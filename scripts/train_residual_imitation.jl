include(joinpath(@__DIR__, "train_imitation.jl"))

using JLD2

const RESIDUAL_HIDDEN_DIMS = (64, 32)
const BASELINE_SCALE = 100.0f0
const RESIDUAL_RANGE = 1.0f0
const RESIDUAL_REGULARIZATION = 1.0f-2
const RESIDUAL_EVALUATION_STEPS = (0, 50, 100, 200, 400, 800)

zeros32(_rng, dims...) = zeros(Float32, dims...)

function build_residual_model(input_dim::Int)
    return Chain(
        Dense(input_dim => RESIDUAL_HIDDEN_DIMS[1], swish),
        Dense(RESIDUAL_HIDDEN_DIMS[1] => RESIDUAL_HIDDEN_DIMS[2], swish),
        Dense(
            RESIDUAL_HIDDEN_DIMS[2] => 1,
            tanh;
            init_weight=zeros32,
            init_bias=zeros32,
        ),
    )
end

function baseline_scores(normalized_features, feature_mean, feature_std, weights)
    raw_features = normalized_features .* feature_std .+ feature_mean
    heuristic_dim = length(weights)
    return transpose(weights) * raw_features[1:heuristic_dim, :] ./ BASELINE_SCALE
end

function residual_pairwise_objective(model, ps, st, data)
    positive, negative, positive_baseline, negative_baseline = data
    positive_residual, st = model(positive, ps, st)
    negative_residual, st = model(negative, ps, st)
    positive_score = positive_baseline .+ RESIDUAL_RANGE .* positive_residual
    negative_score = negative_baseline .+ RESIDUAL_RANGE .* negative_residual
    z = negative_score .- positive_score
    ranking_loss = mean(max.(z, 0.0f0) .+ log1p.(exp.(-abs.(z))))
    residual_penalty = mean(abs2, positive_residual) + mean(abs2, negative_residual)
    loss = ranking_loss + RESIDUAL_REGULARIZATION * residual_penalty
    accuracy = mean(positive_score .> negative_score)
    return loss, st, (; accuracy, ranking_loss, residual_penalty)
end

function residual_pair_metrics(model, ps, st, data)
    loss, _, stats = residual_pairwise_objective(model, ps, st, data)
    return (;
        loss=Float64(loss),
        accuracy=Float64(stats.accuracy),
        ranking_loss=Float64(stats.ranking_loss),
        residual_penalty=Float64(stats.residual_penalty),
    )
end

function select_hybrid_node(
    state,
    model,
    ps,
    st,
    feature_mean,
    feature_std,
    weights::HeuristicWeights,
)
    nodes = stable_node_list(state)
    isempty(nodes) && return nothing
    features = reduce(hcat, model_feature_vector(state, node) for node in nodes)
    normalized = (features .- feature_mean) ./ feature_std
    residuals, _ = model(normalized, ps, st)
    scores = Float32[
        heuristic_value(state, node, weights) / BASELINE_SCALE for node in nodes
    ]
    scores .+= RESIDUAL_RANGE .* vec(residuals)
    return nodes[argmax(scores)]
end

function play_hybrid_episode(
    seed,
    model,
    ps,
    st,
    feature_mean,
    feature_std,
    weights;
    max_steps=250,
)
    state = GameState(Xoshiro(seed))
    steps = 0
    tetrises = 0
    perfect_clears = 0
    while !state.game_over_flag && steps < max_steps
        node = select_hybrid_node(
            state, model, ps, st, feature_mean, feature_std, weights
        )
        isnothing(node) && break
        blocks_before = sum(state.current_game_board.binary)
        apply_node!(state, node)
        cleared = (blocks_before + 4 - sum(state.current_game_board.binary)) ÷ 10
        tetrises += cleared == 4
        perfect_clears += cleared > 0 && iszero(sum(state.current_game_board.binary))
        steps += 1
    end
    final_features = board_features(state.current_game_board.binary)
    return (;
        score=state.score,
        steps,
        game_over=state.game_over_flag,
        tetrises,
        perfect_clears,
        holes=final_features.holes,
        max_height=final_features.max_height,
    )
end

function evaluate_hybrid_candidate(
    training_step,
    seeds,
    model,
    ps,
    st,
    feature_mean,
    feature_std,
    weights,
)
    episodes = Vector{NamedTuple}(undef, length(seeds))
    Threads.@threads for index in eachindex(seeds)
        seed = seeds[index]
        elapsed = @elapsed episode = play_hybrid_episode(
            seed, model, ps, st, feature_mean, feature_std, weights
        )
        episodes[index] = (; seed, episode..., seconds=elapsed)
    end
    scores = getproperty.(episodes, :score)
    summary = (;
        training_step,
        scores,
        mean_score=mean(scores),
        median_score=median(scores),
        minimum_score=minimum(scores),
        maximum_score=maximum(scores),
        over_12k=count(>=(12_000), scores),
        episodes,
    )
    @info "Hybrid policy evaluation" training_step scores summary.mean_score summary.median_score summary.minimum_score summary.over_12k
    return summary
end

function candidate_key(candidate)
    # Median is the primary robustness target. Mean and worst case break ties.
    return (candidate.median_score, candidate.mean_score, candidate.minimum_score)
end

function residual_main()
    dataset_path = joinpath(ROOT, "datasets", "beam_pairs_v1.jld2")
    isfile(dataset_path) || error("Run scripts/train_imitation.jl once to create $dataset_path")
    data = load(dataset_path)
    train_positive = data["train_positive"]
    train_negative = data["train_negative"]
    validation_positive = data["validation_positive"]
    validation_negative = data["validation_negative"]
    feature_mean = data["feature_mean"]
    feature_std = data["feature_std"]
    weights = best_teacher_weights()
    weight_vector = heuristic_weight_vector(weights)

    train_data = (
        train_positive,
        train_negative,
        baseline_scores(train_positive, feature_mean, feature_std, weight_vector),
        baseline_scores(train_negative, feature_mean, feature_std, weight_vector),
    )
    validation_data = (
        validation_positive,
        validation_negative,
        baseline_scores(validation_positive, feature_mean, feature_std, weight_vector),
        baseline_scores(validation_negative, feature_mean, feature_std, weight_vector),
    )

    rng = Xoshiro(0x5742)
    model = build_residual_model(size(train_positive, 1))
    ps, st = Lux.setup(rng, model)
    train_state = Lux.Training.TrainState(model, ps, st, Optimisers.Adam(3.0f-4))
    evaluation_seed_count = parse(Int, get(ENV, "RESIDUAL_EVAL_SEED_COUNT", "5"))
    evaluation_seeds = collect(4001:(4000 + evaluation_seed_count))
    snapshots = Dict{Int,Any}(0 => (ps=deepcopy(ps), st=deepcopy(st)))
    evaluations = NamedTuple[]
    push!(
        evaluations,
        evaluate_hybrid_candidate(
            0, evaluation_seeds, model, ps, st, feature_mean, feature_std, weights
        ),
    )

    pair_count = size(train_positive, 2)
    batch_size = min(parse(Int, get(ENV, "RESIDUAL_BATCH_SIZE", "512")), pair_count)
    final_step = parse(Int, get(ENV, "RESIDUAL_STEPS", string(last(RESIDUAL_EVALUATION_STEPS))))
    evaluation_steps = filter(<=(final_step), collect(RESIDUAL_EVALUATION_STEPS[2:end]))
    started_at = time()
    for step in 1:final_step
        indices = rand(rng, 1:pair_count, batch_size)
        batch = map(array -> array[:, indices], train_data)
        _, loss, stats, train_state = Lux.Training.single_train_step!(
            Lux.Training.AutoZygote(), residual_pairwise_objective, batch, train_state
        )
        if step == 1 || step % 50 == 0
            validation_metrics = residual_pair_metrics(
                model,
                train_state.parameters,
                train_state.states,
                validation_data,
            )
            @info "Residual training" step loss=Float64(loss) train_accuracy=Float64(stats.accuracy) validation_metrics elapsed=time()-started_at
        end
        if step in evaluation_steps
            snapshots[step] = (
                ps=deepcopy(train_state.parameters), st=deepcopy(train_state.states)
            )
            push!(
                evaluations,
                evaluate_hybrid_candidate(
                    step,
                    evaluation_seeds,
                    model,
                    train_state.parameters,
                    train_state.states,
                    feature_mean,
                    feature_std,
                    weights,
                ),
            )
        end
    end

    _, best_evaluation = findmax(candidate_key, evaluations)
    best_step = evaluations[best_evaluation].training_step
    best_snapshot = snapshots[best_step]
    checkpoint_path = joinpath(ROOT, "checkpoints", "residual_imitation_v1.jld2")
    jldsave(
        checkpoint_path;
        ps=best_snapshot.ps,
        st=best_snapshot.st,
        feature_mean,
        feature_std,
        input_dim=size(train_positive, 1),
        hidden_dims=RESIDUAL_HIDDEN_DIMS,
        baseline_scale=BASELINE_SCALE,
        residual_range=RESIDUAL_RANGE,
        residual_regularization=RESIDUAL_REGULARIZATION,
        best_step,
        evaluation_seeds,
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
    )

    summary = (;
        timestamp=string(now()),
        model="residual_imitation_v1",
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
        best_step,
        gate="maximize (median, mean, minimum) on fixed development seeds",
        evaluations,
    )
    run_path = joinpath(ROOT, "runs", "residual_imitation_v1_summary.json")
    open(run_path, "w") do io
        JSON3.pretty(io, summary)
    end
    @info "Saved gated residual checkpoint" checkpoint_path run_path best_step best=evaluations[best_evaluation]
end

residual_main()
