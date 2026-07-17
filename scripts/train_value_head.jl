include(joinpath(@__DIR__, "evaluate_openvino_checkpoint.jl"))

using JLD2
using Optimisers
using Statistics
using Zygote

mutable struct HeadReplay
    features::Matrix{Float32}
    targets::Matrix{Float32}
    count::Int
    cursor::Int
end

function HeadReplay(feature_count::Int, capacity::Int)
    return HeadReplay(
        Matrix{Float32}(undef, feature_count, capacity),
        Matrix{Float32}(undef, 1, capacity),
        0,
        1,
    )
end

function Base.push!(replay::HeadReplay, feature, target::Float32)
    replay.features[:, replay.cursor] .= feature
    replay.targets[1, replay.cursor] = target
    replay.cursor = replay.cursor == size(replay.features, 2) ? 1 : replay.cursor + 1
    replay.count = min(replay.count + 1, size(replay.features, 2))
    return replay
end

function sample_replay(rng, replay::HeadReplay, batch_size::Int)
    indices = rand(rng, 1:replay.count, min(batch_size, replay.count))
    return replay.features[:, indices], replay.targets[:, indices]
end

function head_objective(model, parameters, state, data)
    features, targets = data
    predictions, state = model(features, parameters, state)
    difference = predictions .- targets
    absolute = abs.(difference)
    loss = mean(ifelse.(absolute .<= 1.0f0, 0.5f0 .* difference .^ 2, absolute .- 0.5f0))
    return loss, state, (; q_mean=mean(predictions), target_mean=mean(targets))
end

function extract_features(feature_inference, input)
    python_features = lock(PYTHON_INFERENCE_LOCK) do
        feature_inference.predict(
            input[1], input[2], input[3], input[4], input[5], input[6]
        )
    end
    return permutedims(pyconvert(Matrix{Float32}, python_features))
end

function state_features(state, feature_inference; next_count::Int=5)
    generation_seconds = @elapsed nodes = stable_node_list(state)
    isempty(nodes) && return nodes, Matrix{Float32}(undef, 249, 0), generation_seconds, 0.0
    input = legacy_candidate_batch(state, nodes; next_count)
    inference_seconds = @elapsed features = extract_features(feature_inference, input)
    return nodes, features, generation_seconds, inference_seconds
end

function head_scores(model, parameters, state, features)
    scores, _ = model(features, parameters, state)
    return vec(scores)
end

function evaluate_head_episode(
    seed,
    feature_inference,
    model,
    parameters,
    model_state;
    max_steps::Int=PAPER_EPISODE_STEPS,
)
    state = GameState(Xoshiro(seed))
    steps = 0
    tspins = 0
    started = time()
    while !state.game_over_flag && steps < max_steps
        nodes, features, _, _ = state_features(state, feature_inference)
        isempty(nodes) && break
        scores = head_scores(model, parameters, model_state, features)
        node = nodes[argmax(scores)]
        tspins += node.tspin > 0
        apply_node!(state, node)
        steps += 1
    end
    return (;
        seed,
        score=state.score,
        steps,
        game_over=state.game_over_flag,
        tspins,
        wall_seconds=time() - started,
    )
end

function evaluate_head(
    training_episode,
    seeds,
    feature_inference,
    model,
    parameters,
    model_state;
    max_steps=PAPER_EPISODE_STEPS,
)
    episodes = [
        evaluate_head_episode(
            seed,
            feature_inference,
            model,
            parameters,
            model_state;
            max_steps,
        ) for seed in seeds
    ]
    scores = getproperty.(episodes, :score)
    summary = (;
        training_episode,
        scores,
        mean_score=mean(scores),
        median_score=median(scores),
        minimum_score=minimum(scores),
        maximum_score=maximum(scores),
        episodes,
    )
    @info "Value-head evaluation" summary
    return summary
end

function main()
    sys = pyimport("sys")
    sys.path.insert(0, joinpath(ROOT, "tools"))
    legacy_openvino = pyimport("legacy_openvino")
    device = get(ENV, "OPENVINO_DEVICE", "NPU")
    feature_inference = legacy_openvino.LegacyOpenVINOFeatures(device, 16)

    parameters, model_state = jldopen(
        joinpath(ROOT, "1313", "mainmodel copy 3.jld2"), "r"
    ) do file
        modernize_legacy_parameters(file["ps"]), Lux.testmode(file["st"])
    end
    legacy_model = LegacyQNetwork()
    model = legacy_model.score_net
    initial_parameters = parameters.score_net
    initial_state = model_state.score_net

    rng = Xoshiro(0x5742_2026)
    learning_rate = parse(Float32, get(ENV, "HEAD_LEARNING_RATE", "1e-5"))
    train_state = Lux.Training.TrainState(
        model,
        deepcopy(initial_parameters),
        deepcopy(initial_state),
        Optimisers.Adam(learning_rate),
    )
    target_parameters = deepcopy(initial_parameters)
    target_state = deepcopy(initial_state)
    replay = HeadReplay(249, parse(Int, get(ENV, "HEAD_REPLAY_CAPACITY", "65536")))
    batch_size = parse(Int, get(ENV, "HEAD_BATCH_SIZE", "128"))
    minimum_replay = parse(Int, get(ENV, "HEAD_MINIMUM_REPLAY", "512"))
    target_interval = parse(Int, get(ENV, "HEAD_TARGET_INTERVAL", "400"))
    episode_count = parse(Int, get(ENV, "HEAD_TRAIN_EPISODES", "20"))
    max_steps = parse(Int, get(ENV, "HEAD_TRAIN_STEPS", string(PAPER_EPISODE_STEPS)))
    evaluation_interval = parse(Int, get(ENV, "HEAD_EVAL_INTERVAL", "5"))
    evaluation_steps = parse(Int, get(ENV, "HEAD_EVAL_STEPS", string(PAPER_EPISODE_STEPS)))
    epsilon_start = parse(Float32, get(ENV, "HEAD_EPSILON_START", "0.05"))
    epsilon_end = parse(Float32, get(ENV, "HEAD_EPSILON_END", "0.01"))
    discount = parse(Float32, get(ENV, "HEAD_DISCOUNT", "0.997"))
    training_seeds = collect(7001:(7000 + episode_count))
    evaluation_seeds = collect(5742:5743)
    update_count = 0
    evaluations = NamedTuple[]
    started = time()

    initial_evaluation = evaluate_head(
            0,
            evaluation_seeds,
            feature_inference,
            model,
            train_state.parameters,
            train_state.states;
            max_steps=evaluation_steps,
        )
    push!(evaluations, initial_evaluation)
    best_evaluation_key = (
        initial_evaluation.median_score,
        initial_evaluation.mean_score,
        initial_evaluation.minimum_score,
    )
    best_episode = 0
    best_parameters = deepcopy(train_state.parameters)
    best_state = deepcopy(train_state.states)
    jldsave(
        joinpath(ROOT, "checkpoints", "value_head_best.jld2");
        ps=best_parameters,
        st=best_state,
        best_episode,
        best_evaluation_key,
        evaluation=initial_evaluation,
        update_count,
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
    )

    for (episode_index, seed) in enumerate(training_seeds)
        epsilon = episode_count == 1 ? epsilon_end :
                  epsilon_start + (epsilon_end - epsilon_start) *
                  Float32(episode_index - 1) / Float32(episode_count - 1)
        state = GameState(Xoshiro(seed))
        nodes, features, generation_seconds, inference_seconds = state_features(
            state, feature_inference
        )
        steps = 0
        episode_loss = Float64[]
        while !state.game_over_flag && steps < max_steps && !isempty(nodes)
            scores = head_scores(
                model, train_state.parameters, train_state.states, features
            )
            selected_index = rand(rng) < epsilon ? rand(rng, eachindex(nodes)) : argmax(scores)
            selected_feature = copy(@view features[:, selected_index])
            previous_score = state.score
            apply_node!(state, nodes[selected_index])
            reward = Float32(state.score - previous_score) / 600.0f0
            steps += 1

            if state.game_over_flag || steps == max_steps
                target = state.game_over_flag ? -1000.0f0 / 600.0f0 : reward
                next_nodes = Node[]
                next_features = Matrix{Float32}(undef, 249, 0)
            else
                next_nodes, next_features, generation, inference_time = state_features(
                    state, feature_inference
                )
                generation_seconds += generation
                inference_seconds += inference_time
                if isempty(next_nodes)
                    target = -1000.0f0 / 600.0f0
                else
                    online_next = head_scores(
                        model,
                        train_state.parameters,
                        train_state.states,
                        next_features,
                    )
                    target_next = head_scores(
                        model, target_parameters, target_state, next_features
                    )
                    target = reward + discount * target_next[argmax(online_next)]
                end
            end
            push!(replay, selected_feature, target)

            if replay.count >= minimum_replay
                batch = sample_replay(rng, replay, batch_size)
                _, loss, stats, train_state = Lux.Training.single_train_step!(
                    Lux.Training.AutoZygote(),
                    head_objective,
                    batch,
                    train_state,
                )
                push!(episode_loss, Float64(loss))
                update_count += 1
                if update_count % target_interval == 0
                    target_parameters = deepcopy(train_state.parameters)
                    target_state = deepcopy(train_state.states)
                end
                if update_count == 1 || update_count % 250 == 0
                    @info "Value-head update" episode_index seed steps update_count loss=Float64(loss) stats replay_count=replay.count elapsed=time()-started
                end
            end
            nodes, features = next_nodes, next_features
        end

        @info "Value-head training episode" episode_index seed epsilon score=state.score steps replay_count=replay.count update_count mean_loss=(isempty(episode_loss) ? NaN : mean(episode_loss)) generation_seconds inference_seconds elapsed=time()-started
        checkpoint_path = joinpath(ROOT, "checkpoints", "value_head_latest.jld2")
        jldsave(
            checkpoint_path;
            ps=train_state.parameters,
            st=train_state.states,
            target_ps=target_parameters,
            target_st=target_state,
            episode_index,
            update_count,
            replay_count=replay.count,
            training_seeds=training_seeds[1:episode_index],
            julia_version=string(VERSION),
            lux_version=string(Base.pkgversion(Lux)),
        )

        if evaluation_interval > 0 &&
           (episode_index % evaluation_interval == 0 || episode_index == episode_count)
            evaluation = evaluate_head(
                    episode_index,
                    evaluation_seeds,
                    feature_inference,
                    model,
                    train_state.parameters,
                    train_state.states;
                    max_steps=evaluation_steps,
                )
            push!(evaluations, evaluation)
            evaluation_key = (
                evaluation.median_score,
                evaluation.mean_score,
                evaluation.minimum_score,
            )
            if evaluation_key > best_evaluation_key
                best_evaluation_key = evaluation_key
                best_episode = episode_index
                best_parameters = deepcopy(train_state.parameters)
                best_state = deepcopy(train_state.states)
                jldsave(
                    joinpath(ROOT, "checkpoints", "value_head_best.jld2");
                    ps=best_parameters,
                    st=best_state,
                    best_episode,
                    best_evaluation_key,
                    evaluation,
                    update_count,
                    julia_version=string(VERSION),
                    lux_version=string(Base.pkgversion(Lux)),
                )
            end
        end
    end

    summary_path = joinpath(ROOT, "runs", "value_head_training.json")
    open(summary_path, "w") do io
        JSON3.pretty(
            io,
            (;
                timestamp=string(now()),
                device,
                learning_rate,
                batch_size,
                minimum_replay,
                target_interval,
                episode_count,
                max_steps,
                epsilon_start,
                epsilon_end,
                discount,
                update_count,
                best_episode,
                best_evaluation_key,
                evaluations,
            ),
        )
    end
    @info "Saved value-head training summary" summary_path evaluations
end

main()
