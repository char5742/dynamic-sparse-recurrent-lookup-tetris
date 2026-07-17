using Dates
using JLD2
using JSON3
using Lux
using Optimisers
using Random
using Statistics
using Zygote

zeros32(_rng, dims...) = zeros(Float32, dims...)

function build_residual_gate(input_dim::Int; hidden::Int=64)
    return Chain(
        Dense(input_dim => hidden, swish),
        Dense(hidden => 1, tanh; init_weight=zeros32, init_bias=zeros32),
    )
end

function residual_logits(model, ps, st, batch, feature_mean, feature_std, scale)
    first, second, q_first, q_second = batch
    first = (first .- feature_mean) ./ feature_std
    second = (second .- feature_mean) ./ feature_std
    first_residual, st = model(first, ps, st)
    second_residual, st = model(second, ps, st)
    logits = q_second .- q_first .+ scale .* (second_residual .- first_residual)
    return logits, st
end

function gate_objective(model, ps, st, data)
    batch, labels, feature_mean, feature_std, scale, positive_weight = data
    logits, st = residual_logits(
        model, ps, st, batch, feature_mean, feature_std, scale
    )
    weights = ifelse.(labels .> 0.5f0, positive_weight, 1.0f0)
    losses = max.(logits, 0.0f0) .- logits .* labels .+ log1p.(exp.(-abs.(logits)))
    loss = sum(weights .* losses) / sum(weights)
    return loss, st, (;
        accuracy=mean((logits .> 0.0f0) .== (labels .> 0.5f0)),
        flip_rate=mean(logits .> 0.0f0),
        logit_mean=mean(logits),
    )
end

function metrics(model, ps, st, first, second, q_first, q_second, labels,
                 feature_mean, feature_std, scale)
    logits, _ = residual_logits(
        model, ps, st, (first, second, q_first, q_second),
        feature_mean, feature_std, scale
    )
    prediction = vec(logits .> 0.0f0)
    truth = vec(labels .> 0.5f0)
    tp = count(prediction .& truth)
    fp = count(prediction .& .!truth)
    fn = count((.!prediction) .& truth)
    tn = count((.!prediction) .& .!truth)
    return (;
        count=length(truth),
        accuracy=(tp + tn) / length(truth),
        balanced_accuracy=0.5 * (
            tp / max(tp + fn, 1) + tn / max(tn + fp, 1)
        ),
        precision=tp / max(tp + fp, 1),
        recall=tp / max(tp + fn, 1),
        predicted_flip_rate=mean(prediction),
        teacher_flip_rate=mean(truth),
        tp, fp, fn, tn,
    )
end

function main()
    dataset_path = abspath(get(
        ENV,
        "GATE_DATASET_PATH",
        raw"D:\tetris-paper-plus\datasets\bellman_gate\bellman_gate_smoke.jld2",
    ))
    checkpoint_path = abspath(get(
        ENV,
        "GATE_CHECKPOINT_PATH",
        raw"D:\tetris-paper-plus\checkpoints\bellman_gate\gate_smoke.jld2",
    ))
    mkpath(dirname(checkpoint_path))
    data = load(dataset_path)
    episode_ids = Int.(data["episode_ids"])
    unique_episodes = sort(unique(episode_ids))
    length(unique_episodes) >= 3 || error("need at least three teacher episodes")
    validation_episode_count = max(1, parse(Int, get(ENV, "GATE_VALIDATION_EPISODES", "1")))
    validation_ids = last(unique_episodes, validation_episode_count)
    train_indices = findall(id -> !(id in validation_ids), episode_ids)
    validation_indices = findall(id -> id in validation_ids, episode_ids)
    feature_first = Float32.(data["feature_top1"])
    feature_second = Float32.(data["feature_top2"])
    q_first = reshape(Float32.(data["q_top1"]), 1, :)
    q_second = reshape(Float32.(data["q_top2"]), 1, :)
    labels = reshape(Float32.(data["teacher_selects_top2"]), 1, :)
    population = hcat(feature_first[:, train_indices], feature_second[:, train_indices])
    feature_mean = mean(population; dims=2)
    feature_std = max.(std(population; dims=2, corrected=false), 1.0f-4)

    rng = Xoshiro(parse(UInt64, get(ENV, "GATE_TRAIN_SEED", "13135742")))
    hidden = parse(Int, get(ENV, "GATE_HIDDEN", "64"))
    residual_scale = parse(Float32, get(ENV, "GATE_RESIDUAL_SCALE", "1.0"))
    learning_rate = parse(Float32, get(ENV, "GATE_LEARNING_RATE", "3e-4"))
    updates = parse(Int, get(ENV, "GATE_UPDATES", "500"))
    batch_size = min(parse(Int, get(ENV, "GATE_BATCH", "256")), length(train_indices))
    model = build_residual_gate(size(feature_first, 1); hidden)
    ps, st = Lux.setup(rng, model)
    train_state = Lux.Training.TrainState(
        model, ps, st, Optimisers.AdamW(learning_rate, (0.9, 0.999), 1.0f-4)
    )
    positives = count(labels[:, train_indices] .> 0.5f0)
    negatives = length(train_indices) - positives
    positives > 0 || error("teacher never selects top-2; no gate signal")
    positive_weight = Float32(negatives / positives)
    initial_train = metrics(
        model, ps, st, feature_first[:, train_indices], feature_second[:, train_indices],
        q_first[:, train_indices], q_second[:, train_indices], labels[:, train_indices],
        feature_mean, feature_std, residual_scale,
    )
    initial_validation = metrics(
        model, ps, st, feature_first[:, validation_indices], feature_second[:, validation_indices],
        q_first[:, validation_indices], q_second[:, validation_indices], labels[:, validation_indices],
        feature_mean, feature_std, residual_scale,
    )
    best_ps = deepcopy(ps)
    best_st = deepcopy(st)
    best_validation = initial_validation
    best_update = 0
    losses = Float64[]
    started = time()
    @info "Training zero-initialized Bellman residual gate" initial_train initial_validation positives negatives positive_weight

    for update in 1:updates
        indices = rand(rng, train_indices, batch_size)
        batch = (
            feature_first[:, indices], feature_second[:, indices], q_first[:, indices], q_second[:, indices]
        )
        objective_data = (
            batch, labels[:, indices], feature_mean, feature_std,
            residual_scale, positive_weight,
        )
        _, loss, statistics, train_state = Lux.Training.single_train_step!(
            Lux.Training.AutoZygote(), gate_objective, objective_data, train_state
        )
        isfinite(loss) || error("non-finite gate loss at update $update")
        push!(losses, Float64(loss))
        if update == 1 || update % 25 == 0 || update == updates
            validation = metrics(
                model, train_state.parameters, train_state.states,
                feature_first[:, validation_indices], feature_second[:, validation_indices],
                q_first[:, validation_indices], q_second[:, validation_indices],
                labels[:, validation_indices], feature_mean, feature_std, residual_scale,
            )
            @info "Gate update" update loss=Float64(loss) statistics validation elapsed=time()-started
            key = (validation.balanced_accuracy, validation.accuracy)
            best_key = (best_validation.balanced_accuracy, best_validation.accuracy)
            if key > best_key
                best_update = update
                best_validation = validation
                best_ps = deepcopy(train_state.parameters)
                best_st = deepcopy(train_state.states)
            end
        end
    end

    final_train = metrics(
        model, best_ps, best_st, feature_first[:, train_indices], feature_second[:, train_indices],
        q_first[:, train_indices], q_second[:, train_indices], labels[:, train_indices],
        feature_mean, feature_std, residual_scale,
    )
    jldsave(
        checkpoint_path;
        ps=best_ps,
        st=best_st,
        feature_mean,
        feature_std,
        hidden,
        residual_scale,
        parameter_count=Lux.parameterlength(best_ps),
        best_update,
        initial_train,
        initial_validation,
        final_train,
        best_validation,
        training_episode_ids=setdiff(unique_episodes, validation_ids),
        validation_episode_ids=validation_ids,
        dataset_path,
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
    )
    summary = (;
        experiment_id="BG01_gate_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
        generated_at=string(now()),
        dataset_path,
        checkpoint_path,
        backend="Lux+Zygote",
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
        hidden,
        residual_scale,
        parameter_count=Lux.parameterlength(best_ps),
        learning_rate,
        updates,
        batch_size,
        positive_weight,
        initial_train,
        initial_validation,
        final_train,
        best_validation,
        best_update,
        loss_first=first(losses),
        loss_last=last(losses),
        wall_seconds=time()-started,
        completion_reason="configured offline smoke budget completed",
        held_out_test_seeds_used=false,
    )
    open(replace(checkpoint_path, r"\.jld2$" => ".json"), "w") do io
        JSON3.pretty(io, summary)
    end
    @info "Saved Bellman residual gate" summary
    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
