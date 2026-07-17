using Dates
using JLD2
using JSON3
using LinearAlgebra
using Lux
using Optimisers
using Random
using Statistics
using Zygote

include(joinpath(@__DIR__, "models.jl"))

const DEFAULT_DATASET = raw"D:\tetris-paper-plus\datasets\learning\teacher_dev_5742_5749_2000.jld2"
const DEFAULT_CHECKPOINT_ROOT = raw"D:\tetris-paper-plus\checkpoints\film_ab"
const DEFAULT_RUN_ROOT = raw"D:\tetris-paper-plus\runs\film_ab"

function load_dataset(path)
    return jldopen(path, "r") do file
        action_counts = Int.(file["action_counts"])
        max_actions = maximum(action_counts)
        (;
            boards=file["boards"],
            placements=file["placements"][:, :, :, 1:max_actions, :],
            ren=file["ren"],
            back_to_back=file["back_to_back"],
            tspin=file["tspin"][1:max_actions, :],
            queues=file["queues"],
            teacher_q=file["teacher_q"][1:max_actions, :],
            action_counts,
            episode_ids=Int.(file["episode_ids"]),
        )
    end
end

flat_index(action::Int, slot::Int, max_actions::Int) =
    action + (slot - 1) * max_actions

function candidate_batch(dataset, rows::AbstractVector{Int})
    max_actions = size(dataset.placements, 4)
    state_count = length(rows)
    flat_count = max_actions * state_count
    board = zeros(Float32, 24, 10, 1, flat_count)
    placement = zeros(Float32, 24, 10, 1, flat_count)
    ren = zeros(Float32, 1, flat_count)
    back_to_back = zeros(Float32, 1, flat_count)
    tspin = zeros(Float32, 1, flat_count)
    queue = zeros(Float32, 7, 6, flat_count)
    mask = zeros(Float32, max_actions, state_count)
    targets = zeros(Float32, max_actions, state_count)
    for (slot, row) in enumerate(rows)
        count = dataset.action_counts[row]
        teacher = @view dataset.teacher_q[1:count, row]
        center = mean(teacher)
        scale = max(std(teacher; corrected=false), 1.0f-4)
        targets[1:count, slot] .= (teacher .- center) ./ scale
        flat_start = flat_index(1, slot, max_actions)
        flat_stop = flat_index(max_actions, slot, max_actions)
        flat_range = flat_start:flat_stop
        board[:, :, :, flat_range] .= reshape(
            Float32.(@view(dataset.boards[:, :, :, row])), 24, 10, 1, 1
        )
        ren[1, flat_range] .= dataset.ren[1, row]
        back_to_back[1, flat_range] .= dataset.back_to_back[1, row]
        queue[:, :, flat_range] .= reshape(
            Float32.(@view(dataset.queues[:, :, row])), 7, 6, 1
        )
        valid_range = flat_start:(flat_start + count - 1)
        placement[:, :, :, valid_range] .= Float32.(
            @view dataset.placements[:, :, :, 1:count, row]
        )
        tspin[1, valid_range] .= @view dataset.tspin[1:count, row]
        mask[1:count, slot] .= 1.0f0
    end
    return (board, placement, ren, back_to_back, tspin, queue), targets, mask
end

function standardized_listnet(
    predictions, teacher_z, mask; temperature::Float32=1.0f0
)
    temperature > 0 || error("temperature must be positive")
    counts = max.(sum(mask; dims=1), 1.0f0)
    prediction_mean = sum(predictions .* mask; dims=1) ./ counts
    centered = (predictions .- prediction_mean) .* mask
    prediction_scale = sqrt.(sum(centered .^ 2; dims=1) ./ counts .+ 1.0f-4)
    prediction_z = centered ./ prediction_scale
    invalid = (1.0f0 .- mask) .* -1.0f4
    teacher_logits = teacher_z ./ temperature .+ invalid
    student_logits = prediction_z ./ temperature .+ invalid
    teacher_logits = teacher_logits .- maximum(teacher_logits; dims=1)
    student_logits = student_logits .- maximum(student_logits; dims=1)
    teacher_exp = exp.(teacher_logits) .* mask
    teacher_probability = teacher_exp ./ sum(teacher_exp; dims=1)
    student_log_probability = student_logits .-
                              log.(sum(exp.(student_logits) .* mask; dims=1))
    return -sum(teacher_probability .* student_log_probability .* mask) /
           size(predictions, 2)
end

function objective(model, ps, st, batch)
    input, teacher_z, mask, max_actions, state_count, temperature = batch
    raw, next_st = model(input, ps, st)
    predictions = reshape(raw, max_actions, state_count)
    loss = standardized_listnet(predictions, teacher_z, mask; temperature)
    return loss, next_st, (; valid_candidates=sum(mask))
end

function policy_metrics(
    dataset, rows, model, ps, st;
    state_batch::Int=8,
    temperature::Float32=1.0f0,
)
    agreements = Bool[]
    reciprocal_ranks = Float64[]
    correlations = Float64[]
    losses = Float64[]
    for row_batch in Iterators.partition(rows, state_batch)
        batch_rows = collect(row_batch)
        input, teacher_z, mask = candidate_batch(dataset, batch_rows)
        raw, _ = model(input, ps, st)
        predictions = reshape(raw, size(mask))
        push!(
            losses,
            Float64(standardized_listnet(predictions, teacher_z, mask; temperature)),
        )
        for (slot, row) in enumerate(batch_rows)
            count = dataset.action_counts[row]
            prediction = @view predictions[1:count, slot]
            teacher = @view dataset.teacher_q[1:count, row]
            teacher_best = argmax(teacher)
            push!(agreements, argmax(prediction) == teacher_best)
            ordering = sortperm(prediction; rev=true)
            push!(reciprocal_ranks, 1.0 / findfirst(==(teacher_best), ordering))
            if std(prediction; corrected=false) > 1.0f-8 &&
               std(teacher; corrected=false) > 1.0f-8
                push!(correlations, cor(prediction, teacher))
            end
        end
    end
    return (;
        states=length(rows),
        top1_agreement=mean(agreements),
        mean_reciprocal_rank=mean(reciprocal_ranks),
        mean_correlation=isempty(correlations) ? NaN : mean(correlations),
        mean_listwise_cross_entropy=mean(losses),
        random_top1=mean(1.0 ./ dataset.action_counts[rows]),
    )
end

function train_one(
    kind::Symbol,
    model,
    initial_ps,
    initial_st,
    dataset,
    training_rows,
    development_rows,
    row_schedule;
    learning_rate::Float32,
    temperature::Float32,
    run_id::String,
    checkpoint_root::String,
)
    train_state = Lux.Training.TrainState(
        model,
        deepcopy(initial_ps),
        deepcopy(initial_st),
        Optimisers.AdamW(learning_rate, (0.9, 0.999), 1.0f-4),
    )
    initial_metrics = policy_metrics(
        dataset,
        development_rows,
        model,
        train_state.parameters,
        train_state.states;
        temperature,
    )
    losses = Float64[]
    started = time()
    for (update, rows) in enumerate(row_schedule)
        input, targets, mask = candidate_batch(dataset, rows)
        batch = (
            input, targets, mask, size(mask, 1), size(mask, 2), temperature
        )
        _, loss, _, train_state = Lux.Training.single_train_step!(
            Lux.Training.AutoZygote(), objective, batch, train_state
        )
        isfinite(loss) || error("non-finite $kind loss at update $update")
        push!(losses, Float64(loss))
        if update == 1 || update % 50 == 0
            @info "FiLM A/B update" kind update loss=Float64(loss) elapsed=time()-started
        end
    end
    final_metrics = policy_metrics(
        dataset,
        development_rows,
        model,
        train_state.parameters,
        train_state.states;
        temperature,
    )
    checkpoint_path = joinpath(checkpoint_root, "$(run_id)_$(kind).jld2")
    jldsave(
        checkpoint_path;
        ps=train_state.parameters,
        st=train_state.states,
        model_kind=String(kind),
        model_config=(; channels=8, blocks=1, spatial_channels=2),
        parameter_count=Lux.parameterlength(train_state.parameters),
        initial_metrics,
        final_metrics,
        losses,
    )
    return (;
        kind=String(kind),
        checkpoint_path,
        parameter_count=Lux.parameterlength(train_state.parameters),
        macs_per_candidate=analytical_macs(kind),
        initial_metrics,
        final_metrics,
        loss_first=first(losses),
        loss_last=last(losses),
        wall_seconds=time() - started,
    )
end

function main()
    dataset_path = get(ENV, "FILM_AB_DATASET", DEFAULT_DATASET)
    checkpoint_root = get(ENV, "FILM_AB_CHECKPOINT_ROOT", DEFAULT_CHECKPOINT_ROOT)
    run_root = get(ENV, "FILM_AB_RUN_ROOT", DEFAULT_RUN_ROOT)
    mkpath(checkpoint_root)
    mkpath(run_root)
    dataset = load_dataset(dataset_path)
    max_actions = size(dataset.placements, 4)
    max_actions == 74 || error("frozen candidate width must be 74, got $max_actions")
    episodes = sort(unique(dataset.episode_ids))
    development_episodes = episodes[(end - 1):end]
    training_rows = findall(id -> !(id in development_episodes), dataset.episode_ids)
    development_rows = findall(id -> id in development_episodes, dataset.episode_ids)

    seed = parse(UInt64, get(ENV, "FILM_AB_SEED", "517812839"))
    updates = parse(Int, get(ENV, "FILM_AB_UPDATES", "300"))
    state_batch = parse(Int, get(ENV, "FILM_AB_STATE_BATCH", "2"))
    learning_rate = parse(Float32, get(ENV, "FILM_AB_LR", "1e-3"))
    temperature = parse(Float32, get(ENV, "FILM_AB_TEMPERATURE", "0.25"))
    temperature > 0 || error("FILM_AB_TEMPERATURE must be positive")
    schedule_rng = Xoshiro(seed + 0x9e3779b97f4a7c15)
    row_schedule = [rand(schedule_rng, training_rows, state_batch) for _ in 1:updates]

    initialized = matched_initialization(seed)
    plain_initial, _ = initialized.plain(
        first(candidate_batch(dataset, training_rows[1:1])),
        initialized.plain_ps,
        initialized.plain_st,
    )
    film_initial, _ = initialized.film(
        first(candidate_batch(dataset, training_rows[1:1])),
        initialized.film_ps,
        initialized.film_st,
    )
    initial_max_abs_difference = maximum(abs.(plain_initial .- film_initial))
    initial_max_abs_difference <= 1.0f-6 || error(
        "matched initialization failed: $initial_max_abs_difference"
    )

    run_id = get(
        ENV, "FILM_AB_RUN_ID", "film_ab_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
    )
    plain_result = train_one(
        :plain,
        initialized.plain,
        initialized.plain_ps,
        initialized.plain_st,
        dataset,
        training_rows,
        development_rows,
        row_schedule;
        learning_rate,
        temperature,
        run_id,
        checkpoint_root,
    )
    film_result = train_one(
        :film,
        initialized.film,
        initialized.film_ps,
        initialized.film_st,
        dataset,
        training_rows,
        development_rows,
        row_schedule;
        learning_rate,
        temperature,
        run_id,
        checkpoint_root,
    )
    summary = (;
        experiment_id=run_id,
        generated_at=string(now()),
        hypothesis="queue-conditioned FiLM improves candidate ranking under an otherwise matched compact CNN",
        dataset_path,
        effective_max_actions=max_actions,
        seed,
        learning_rate,
        temperature,
        updates,
        state_batch,
        training_episode_ids=sort(unique(dataset.episode_ids[training_rows])),
        development_episode_ids=development_episodes,
        held_out_test_seeds_used=false,
        shared_initialization=true,
        film_condition_zero_initialized=true,
        initial_max_abs_difference,
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
        backend="Lux+Zygote",
        plain=plain_result,
        film=film_result,
        completion_reason="configured matched A/B budget completed",
    )
    summary_path = joinpath(run_root, "$(run_id).json")
    open(summary_path, "w") do io
        JSON3.pretty(io, summary)
    end
    @info "Completed matched FiLM A/B" summary_path plain_result film_result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
