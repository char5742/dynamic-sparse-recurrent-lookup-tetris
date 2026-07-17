ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = get(
    ENV,
    "JULIA_PYTHONCALL_EXE",
    raw"D:\tetris-paper-plus\python-env\Scripts\python.exe",
)

const EXPERIMENT_DIR = @__DIR__
const REPOSITORY_ROOT = normpath(joinpath(EXPERIMENT_DIR, "..", ".."))
include(joinpath(REPOSITORY_ROOT, "scripts", "evaluate_openvino_checkpoint.jl"))
include(joinpath(EXPERIMENT_DIR, "train_gate.jl"))

using JLD2
using JSON3
using PythonCall
using Statistics

function load_gate(path)
    return jldopen(path, "r") do file
        hidden = Int(file["hidden"])
        model = build_residual_gate(size(file["feature_mean"], 1); hidden)
        return (;
            model,
            ps=file["ps"],
            st=Lux.testmode(file["st"]),
            feature_mean=file["feature_mean"],
            feature_std=file["feature_std"],
            residual_scale=Float32(file["residual_scale"]),
            parameter_count=Lux.parameterlength(file["ps"]),
            hidden,
        )
    end
end

function gate_features(feature_inference, input)
    raw = lock(PYTHON_INFERENCE_LOCK) do
        feature_inference.predict(
            input[1], input[2], input[3], input[4], input[5], input[6]
        )
    end
    return permutedims(pyconvert(Matrix{Float32}, raw))
end

function evaluate_gate_episode(
    seed,
    q_inference,
    feature_inference,
    gate;
    next_count=5,
    max_steps=PAPER_EPISODE_STEPS,
    backend_batch=16,
    decision_threshold=0.0f0,
)
    state = GameState(Xoshiro(seed))
    steps = 0
    flips = 0
    candidates = 0
    logical_model_passes = 0
    q_backend_requests = 0
    feature_backend_requests = 0
    generation_seconds = 0.0
    q_seconds = 0.0
    feature_seconds = 0.0
    gate_seconds = 0.0
    started = time()
    while !state.game_over_flag && steps < max_steps
        generation_seconds += @elapsed nodes = stable_node_list(state)
        isempty(nodes) && break
        input = legacy_candidate_batch(state, nodes; next_count)
        q_seconds += @elapsed q = openvino_scores(q_inference, input)
        feature_seconds += @elapsed features = gate_features(feature_inference, input)
        order = partialsortperm(q, 1:min(2, length(q)); rev=true)
        selected_index = first(order)
        if length(order) == 2
            first_index, second_index = order
            pair_first = @view features[:, first_index:first_index]
            pair_second = @view features[:, second_index:second_index]
            gate_seconds += @elapsed begin
                first_residual, _ = gate.model(
                    (pair_first .- gate.feature_mean) ./ gate.feature_std,
                    gate.ps,
                    gate.st,
                )
                second_residual, _ = gate.model(
                    (pair_second .- gate.feature_mean) ./ gate.feature_std,
                    gate.ps,
                    gate.st,
                )
            end
            first_score = q[first_index] + gate.residual_scale * first_residual[1]
            second_score = q[second_index] + gate.residual_scale * second_residual[1]
            if second_score - first_score > decision_threshold
                selected_index = second_index
                flips += 1
            end
        end
        apply_node!(state, nodes[selected_index])
        steps += 1
        candidates += length(nodes)
        logical_model_passes += 1
        request_count = cld(length(nodes), backend_batch)
        q_backend_requests += request_count
        feature_backend_requests += request_count
        if steps == 1 || steps % 25 == 0 || state.game_over_flag
            @info "Bellman gate episode" seed steps score=state.score flips candidates q_seconds feature_seconds gate_seconds wall_seconds=time()-started
        end
    end
    return (;
        seed,
        score=state.score,
        steps,
        game_over=state.game_over_flag,
        flips,
        flip_rate=steps == 0 ? 0.0 : flips / steps,
        candidate_count=candidates,
        logical_model_passes,
        lookahead_expansions=0,
        q_backend_requests,
        feature_backend_requests,
        physical_backend_requests=q_backend_requests + feature_backend_requests,
        generation_seconds,
        q_seconds,
        feature_seconds,
        gate_seconds,
        inference_seconds=q_seconds + feature_seconds + gate_seconds,
        wall_seconds=time() - started,
    )
end

function main()
    checkpoint_path = abspath(get(
        ENV,
        "GATE_CHECKPOINT_PATH",
        raw"D:\tetris-paper-plus\checkpoints\bellman_gate\gate_smoke.jld2",
    ))
    seeds = parse.(Int, split(get(ENV, "GATE_EVAL_SEEDS", "5748"), ','))
    max_steps = parse(Int, get(ENV, "GATE_EVAL_STEPS", "100"))
    next_count = parse(Int, get(ENV, "GATE_NEXT_COUNT", "5"))
    device = get(ENV, "OPENVINO_DEVICE", "NPU")
    backend_batch = parse(Int, get(ENV, "OPENVINO_BATCH", "16"))
    decision_threshold = parse(Float32, get(ENV, "GATE_DECISION_THRESHOLD", "0.0"))
    gate = load_gate(checkpoint_path)
    sys = pyimport("sys")
    sys.path.insert(0, joinpath(REPOSITORY_ROOT, "tools"))
    legacy_openvino = pyimport("legacy_openvino")
    q_inference = legacy_openvino.LegacyOpenVINOInference(device, backend_batch)
    feature_inference = legacy_openvino.LegacyOpenVINOFeatures(device, backend_batch)
    episodes = [
        evaluate_gate_episode(
            seed, q_inference, feature_inference, gate;
            next_count, max_steps, backend_batch, decision_threshold,
        ) for seed in seeds
    ]
    scores = getproperty.(episodes, :score)
    result = (;
        experiment="bellman_gate_model_only_evaluation",
        checkpoint_path,
        device,
        backend_batch,
        next_count,
        max_steps,
        decision_threshold,
        architecture=(;
            frozen_legacy_backbone=true,
            residual_gate_hidden=gate.hidden,
            residual_gate_parameters=gate.parameter_count,
        ),
        search_budget=(;
            identical_candidate_set=true,
            top_k_after_base_scoring=2,
            logical_model_passes_per_decision=1,
            lookahead_expansions=0,
            note="Current prototype uses separate Q and feature OpenVINO outputs; physical requests are reported.",
        ),
        episodes,
        mean_score=mean(scores),
        median_score=median(scores),
        maximum_score=maximum(scores),
        held_out_test_seeds_used=false,
    )
    output_root = raw"D:\tetris-paper-plus\runs\bellman_gate"
    mkpath(output_root)
    output_path = joinpath(
        output_root, "gate_eval_seed$(join(seeds, '-'))_steps$(max_steps).json"
    )
    open(output_path, "w") do io
        JSON3.pretty(io, result)
    end
    @info "Saved Bellman gate evaluation" output_path result.mean_score result.median_score
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
