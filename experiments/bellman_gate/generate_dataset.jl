ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = get(
    ENV,
    "JULIA_PYTHONCALL_EXE",
    raw"D:\tetris-paper-plus\python-env\Scripts\python.exe",
)

const EXPERIMENT_DIR = @__DIR__
const REPOSITORY_ROOT = normpath(joinpath(EXPERIMENT_DIR, "..", ".."))
include(joinpath(REPOSITORY_ROOT, "scripts", "evaluate_openvino_checkpoint.jl"))

using Dates
using JLD2
using JSON3
using PythonCall
using SHA
using Statistics

file_sha256(path) = bytes2hex(open(sha256, path))

function feature_scores(feature_inference, input)
    raw = lock(PYTHON_INFERENCE_LOCK) do
        feature_inference.predict(
            input[1], input[2], input[3], input[4], input[5], input[6]
        )
    end
    return permutedims(pyconvert(Matrix{Float32}, raw))
end

function bellman_value(state, node, q_inference; next_count, discount)
    reward = Float32(node.game_state.score - state.score) / 600.0f0
    if node.game_state.game_over_flag
        return -1000.0f0 / 600.0f0, reward, 0
    end
    next_nodes = stable_node_list(node.game_state)
    isempty(next_nodes) && return -1000.0f0 / 600.0f0, reward, 0
    next_input = legacy_candidate_batch(node.game_state, next_nodes; next_count)
    next_q = openvino_scores(q_inference, next_input)
    return reward + discount * maximum(next_q), reward, length(next_nodes)
end

function main()
    seed_first = parse(Int, get(ENV, "GATE_SEED_FIRST", "72001"))
    episodes = parse(Int, get(ENV, "GATE_EPISODES", "4"))
    max_steps = parse(Int, get(ENV, "GATE_MAX_STEPS", "100"))
    next_count = parse(Int, get(ENV, "GATE_NEXT_COUNT", "5"))
    batch_size = parse(Int, get(ENV, "OPENVINO_BATCH", "16"))
    device = get(ENV, "OPENVINO_DEVICE", "NPU")
    blend = parse(Float32, get(ENV, "GATE_TEACHER_BLEND", "0.5"))
    discount = parse(Float32, get(ENV, "GATE_DISCOUNT", "0.997"))
    output_path = abspath(get(
        ENV,
        "GATE_DATASET_PATH",
        raw"D:\tetris-paper-plus\datasets\bellman_gate\bellman_gate_smoke.jld2",
    ))
    mkpath(dirname(output_path))

    sys = pyimport("sys")
    sys.path.insert(0, joinpath(REPOSITORY_ROOT, "tools"))
    legacy_openvino = pyimport("legacy_openvino")
    @info "Compiling Bellman teacher" device batch_size blend discount
    q_inference = legacy_openvino.LegacyOpenVINOInference(device, batch_size)
    feature_inference = legacy_openvino.LegacyOpenVINOFeatures(device, batch_size)

    feature_top1 = Vector{Vector{Float32}}()
    feature_top2 = Vector{Vector{Float32}}()
    q_top1 = Float32[]
    q_top2 = Float32[]
    bellman_top1 = Float32[]
    bellman_top2 = Float32[]
    reward_top1 = Float32[]
    reward_top2 = Float32[]
    teacher_selects_top2 = Bool[]
    episode_ids = Int32[]
    episode_steps = Int16[]
    seeds = Int[]
    current_candidates = Int16[]
    lookahead_candidates = Int32[]
    episode_summaries = NamedTuple[]
    q_seconds = 0.0
    feature_seconds = 0.0
    lookahead_seconds = 0.0
    generation_seconds = 0.0
    started = time()

    for episode_offset in 0:(episodes - 1)
        seed = seed_first + episode_offset
        state = GameState(Xoshiro(seed))
        steps = 0
        flips = 0
        while !state.game_over_flag && steps < max_steps
            generation_seconds += @elapsed nodes = stable_node_list(state)
            length(nodes) >= 2 || break
            input = legacy_candidate_batch(state, nodes; next_count)
            q_seconds += @elapsed q = openvino_scores(q_inference, input)
            feature_seconds += @elapsed features = feature_scores(feature_inference, input)
            order = partialsortperm(q, 1:2; rev=true)
            first_index, second_index = order
            lookahead_seconds += @elapsed begin
                first_bellman, first_reward, first_next_count = bellman_value(
                    state, nodes[first_index], q_inference; next_count, discount
                )
                second_bellman, second_reward, second_next_count = bellman_value(
                    state, nodes[second_index], q_inference; next_count, discount
                )
            end
            first_teacher = muladd(blend, first_bellman - q[first_index], q[first_index])
            second_teacher = muladd(blend, second_bellman - q[second_index], q[second_index])
            choose_second = second_teacher > first_teacher

            push!(feature_top1, copy(@view features[:, first_index]))
            push!(feature_top2, copy(@view features[:, second_index]))
            push!(q_top1, q[first_index])
            push!(q_top2, q[second_index])
            push!(bellman_top1, first_bellman)
            push!(bellman_top2, second_bellman)
            push!(reward_top1, first_reward)
            push!(reward_top2, second_reward)
            push!(teacher_selects_top2, choose_second)
            push!(episode_ids, Int32(episode_offset + 1))
            push!(episode_steps, Int16(steps + 1))
            push!(seeds, seed)
            push!(current_candidates, Int16(length(nodes)))
            push!(lookahead_candidates, Int32(first_next_count + second_next_count))

            chosen_index = choose_second ? second_index : first_index
            flips += choose_second
            apply_node!(state, nodes[chosen_index])
            steps += 1
        end
        summary = (;
            seed,
            score=state.score,
            steps,
            game_over=state.game_over_flag,
            top2_flips=flips,
            flip_rate=steps == 0 ? 0.0 : flips / steps,
        )
        push!(episode_summaries, summary)
        @info "Bellman teacher episode" summary elapsed=time()-started
    end

    isempty(q_top1) && error("Bellman gate dataset is empty")
    checkpoint_path = joinpath(REPOSITORY_ROOT, "1313", "mainmodel copy 3.jld2")
    metadata = (;
        experiment_id="BG00_teacher_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
        generated_at=string(now()),
        hypothesis="Top-2 one-step Bellman reranking contains a distillable improvement signal.",
        mechanism="Train a zero-initialized residual gate, then remove lookahead at inference.",
        success="Non-trivial flips and positive held-episode imitation signal within the smoke budget.",
        stop="No flips, non-finite values, or configured budget exhausted.",
        source_checkpoint=checkpoint_path,
        source_checkpoint_sha256=file_sha256(checkpoint_path),
        manifest_sha256=file_sha256(joinpath(REPOSITORY_ROOT, "Manifest.toml")),
        generator_sha256=file_sha256(@__FILE__),
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
        openvino_version=pyconvert(String, pyimport("openvino").__version__),
        device,
        batch_size,
        next_count,
        seed_first,
        episodes,
        max_steps,
        blend,
        discount,
        rows=length(q_top1),
        top2_flips=count(identity, teacher_selects_top2),
        top2_flip_rate=mean(teacher_selects_top2),
        mean_old_q_margin=mean(q_top1 .- q_top2),
        q_seconds,
        feature_seconds,
        lookahead_seconds,
        generation_seconds,
        wall_seconds=time() - started,
        episode_summaries,
        held_out_test_seeds_used=false,
    )
    jldsave(
        output_path;
        feature_top1=reduce(hcat, feature_top1),
        feature_top2=reduce(hcat, feature_top2),
        q_top1,
        q_top2,
        bellman_top1,
        bellman_top2,
        reward_top1,
        reward_top2,
        teacher_selects_top2,
        episode_ids,
        episode_steps,
        seeds,
        current_candidates,
        lookahead_candidates,
        metadata,
    )
    open(replace(output_path, r"\.jld2$" => ".json"), "w") do io
        JSON3.pretty(io, metadata)
    end
    @info "Saved Bellman gate dataset" output_path metadata
    return metadata
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
