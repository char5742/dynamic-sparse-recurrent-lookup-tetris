include(joinpath(@__DIR__, "benchmark_legacy_engine.jl"))

using LinearAlgebra
using Lux
using Optimisers
using Zygote
using JLD2
using JSON3
using Dates

const PAIRS_PER_STATE = 8
const BEAM_WIDTH = 4
const BEAM_DISCOUNT = 1.0
const HIDDEN_DIMS = (128, 64)

function collect_pair_data(seed::Int, weights::HeuristicWeights; max_steps::Int=250)
    state = GameState(Xoshiro(seed))
    positive_features = Vector{Vector{Float32}}()
    negative_features = Vector{Vector{Float32}}()
    steps = 0

    while !state.game_over_flag && steps < max_steps
        nodes, selected_index, first_values = rank_beam_nodes(
            state,
            weights;
            beam_width=BEAM_WIDTH,
            discount=BEAM_DISCOUNT,
        )
        isempty(nodes) && break

        selected_feature = model_feature_vector(state, nodes[selected_index])
        hard_negative_order = sortperm(first_values; rev=true)
        pair_count = 0
        for negative_index in hard_negative_order
            negative_index == selected_index && continue
            push!(positive_features, selected_feature)
            push!(negative_features, model_feature_vector(state, nodes[negative_index]))
            pair_count += 1
            pair_count == PAIRS_PER_STATE && break
        end

        apply_node!(state, nodes[selected_index])
        steps += 1
    end

    return (
        positive=reduce(hcat, positive_features),
        negative=reduce(hcat, negative_features),
        score=state.score,
        steps=steps,
        game_over=state.game_over_flag,
    )
end

function pairwise_objective(model, ps, st, data)
    positive, negative = data
    positive_score, st = model(positive, ps, st)
    negative_score, st = model(negative, ps, st)
    z = negative_score .- positive_score
    loss = mean(max.(z, 0.0f0) .+ log1p.(exp.(-abs.(z))))
    accuracy = mean(positive_score .> negative_score)
    return loss, st, (; accuracy)
end

function pair_metrics(model, ps, st, positive, negative)
    loss, _, stats = pairwise_objective(model, ps, st, (positive, negative))
    return Float64(loss), Float64(stats.accuracy)
end

function build_model(input_dim::Int)
    return Chain(
        Dense(input_dim => HIDDEN_DIMS[1], swish),
        Dense(HIDDEN_DIMS[1] => HIDDEN_DIMS[2], swish),
        Dense(HIDDEN_DIMS[2] => 1),
    )
end

function normalize_features(positive, negative, feature_mean, feature_std)
    return (
        (positive .- feature_mean) ./ feature_std,
        (negative .- feature_mean) ./ feature_std,
    )
end

function select_model_node(state, model, ps, st, feature_mean, feature_std)
    nodes = stable_node_list(state)
    isempty(nodes) && return nothing
    features = reduce(hcat, model_feature_vector(state, node) for node in nodes)
    normalized = (features .- feature_mean) ./ feature_std
    scores, _ = model(normalized, ps, st)
    return nodes[argmax(vec(scores))]
end

function play_model_episode(seed, model, ps, st, feature_mean, feature_std; max_steps=250)
    state = GameState(Xoshiro(seed))
    steps = 0
    tetrises = 0
    perfect_clears = 0
    while !state.game_over_flag && steps < max_steps
        node = select_model_node(state, model, ps, st, feature_mean, feature_std)
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

function main()
    train_seed_count = parse(Int, get(ENV, "IMITATION_TRAIN_SEEDS", "6"))
    validation_seed_count = parse(Int, get(ENV, "IMITATION_VALIDATION_SEEDS", "2"))
    training_steps = parse(Int, get(ENV, "IMITATION_STEPS", "1500"))
    batch_size = parse(Int, get(ENV, "IMITATION_BATCH_SIZE", "512"))
    weights = best_teacher_weights()
    train_seeds = collect(2001:(2000 + train_seed_count))
    validation_seeds = collect(3001:(3000 + validation_seed_count))
    all_seeds = vcat(train_seeds, validation_seeds)
    collected = Vector{NamedTuple}(undef, length(all_seeds))

    @info "Collecting beam-teacher ranking pairs" train_seeds validation_seeds threads=Threads.nthreads()
    Threads.@threads for index in eachindex(all_seeds)
        seed = all_seeds[index]
        elapsed = @elapsed result = collect_pair_data(seed, weights)
        collected[index] = (; seed, result..., seconds=elapsed)
        @info "Collected teacher episode" seed result.score result.steps result.game_over pairs=size(result.positive, 2) seconds=elapsed
    end

    train_results = collected[1:length(train_seeds)]
    validation_results = collected[(length(train_seeds) + 1):end]
    train_positive = reduce(hcat, getproperty.(train_results, :positive))
    train_negative = reduce(hcat, getproperty.(train_results, :negative))
    validation_positive = reduce(hcat, getproperty.(validation_results, :positive))
    validation_negative = reduce(hcat, getproperty.(validation_results, :negative))

    feature_population = hcat(train_positive, train_negative)
    feature_mean = mean(feature_population; dims=2)
    feature_std = std(feature_population; dims=2)
    feature_std .= max.(feature_std, 1.0f-4)
    train_positive, train_negative = normalize_features(
        train_positive, train_negative, feature_mean, feature_std
    )
    validation_positive, validation_negative = normalize_features(
        validation_positive, validation_negative, feature_mean, feature_std
    )

    dataset_path = joinpath(ROOT, "datasets", "beam_pairs_v1.jld2")
    jldsave(
        dataset_path;
        train_positive,
        train_negative,
        validation_positive,
        validation_negative,
        feature_mean,
        feature_std,
        train_seeds,
        validation_seeds,
    )
    @info "Saved dataset" dataset_path train_pairs=size(train_positive, 2) validation_pairs=size(validation_positive, 2)

    BLAS.set_num_threads(min(8, Threads.nthreads()))
    rng = Xoshiro(5742)
    input_dim = size(train_positive, 1)
    model = build_model(input_dim)
    ps, st = Lux.setup(rng, model)
    train_state = Lux.Training.TrainState(model, ps, st, Optimisers.Adam(1.0f-3))
    best_validation_loss = Inf
    best_ps = deepcopy(ps)
    best_st = deepcopy(st)
    pair_count = size(train_positive, 2)
    started_at = time()

    for step in 1:training_steps
        indices = rand(rng, 1:pair_count, min(batch_size, pair_count))
        _, loss, stats, train_state = Lux.Training.single_train_step!(
            Lux.Training.AutoZygote(),
            pairwise_objective,
            (train_positive[:, indices], train_negative[:, indices]),
            train_state,
        )

        if step == 1 || step % 100 == 0 || step == training_steps
            validation_loss, validation_accuracy = pair_metrics(
                model,
                train_state.parameters,
                train_state.states,
                validation_positive,
                validation_negative,
            )
            @info "Imitation training" step loss=Float64(loss) train_accuracy=Float64(stats.accuracy) validation_loss validation_accuracy elapsed=time()-started_at
            if validation_loss < best_validation_loss
                best_validation_loss = validation_loss
                best_ps = deepcopy(train_state.parameters)
                best_st = deepcopy(train_state.states)
            end
        end
    end

    checkpoint_path = joinpath(ROOT, "checkpoints", "imitation_v1.jld2")
    jldsave(
        checkpoint_path;
        ps=best_ps,
        st=best_st,
        feature_mean,
        feature_std,
        input_dim,
        hidden_dims=HIDDEN_DIMS,
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
        best_validation_loss,
        train_seeds,
        validation_seeds,
    )
    @info "Saved checkpoint" checkpoint_path best_validation_loss

    BLAS.set_num_threads(1)
    evaluation_seeds = collect(4001:4005)
    evaluations = Vector{NamedTuple}(undef, length(evaluation_seeds))
    Threads.@threads for index in eachindex(evaluation_seeds)
        seed = evaluation_seeds[index]
        elapsed = @elapsed episode = play_model_episode(
            seed, model, best_ps, best_st, feature_mean, feature_std
        )
        evaluations[index] = (; seed, episode..., seconds=elapsed)
    end
    scores = getproperty.(evaluations, :score)
    for result in evaluations
        @info "Model evaluation" result
    end
    summary = (;
        timestamp=string(now()),
        model="imitation_v1",
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
        scores,
        mean_score=mean(scores),
        median_score=median(scores),
        minimum_score=minimum(scores),
        maximum_score=maximum(scores),
        over_12k=count(>=(12_000), scores),
        best_validation_loss,
    )
    run_path = joinpath(ROOT, "runs", "imitation_v1_summary.json")
    open(run_path, "w") do io
        JSON3.pretty(io, summary)
    end
    @info "Evaluation summary" summary run_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
