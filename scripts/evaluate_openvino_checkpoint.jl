ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = get(
    ENV,
    "JULIA_PYTHONCALL_EXE",
    raw"D:\tetris-paper-plus\python-env\Scripts\python.exe",
)

include(joinpath(@__DIR__, "evaluate_legacy_checkpoint.jl"))

using PythonCall

const PYTHON_INFERENCE_LOCK = ReentrantLock()

function select_openvino_node(state::GameState, inference; next_count::Int=5)
    generation_seconds = @elapsed nodes = stable_node_list(state)
    isempty(nodes) && return nothing, generation_seconds, 0.0, 0
    input = legacy_candidate_batch(state, nodes; next_count)
    inference_seconds = @elapsed begin
        scores = openvino_scores(inference, input)
    end
    all(isfinite, scores) || error("OpenVINO checkpoint produced a non-finite score")
    return nodes[argmax(scores)], generation_seconds, inference_seconds, length(nodes)
end

function openvino_scores(inference, input)
    return lock(PYTHON_INFERENCE_LOCK) do
        pyconvert(
            Vector{Float32},
            inference.predict(
                input[1], input[2], input[3], input[4], input[5], input[6]
            ),
        )
    end
end

function evaluate_openvino_episode(
    seed::Integer, inference; next_count::Int=5, max_steps::Int=PAPER_EPISODE_STEPS
)
    state = GameState(Xoshiro(seed))
    steps = 0
    tetrises = 0
    tspins = 0
    perfect_clears = 0
    generation_seconds = 0.0
    inference_seconds = 0.0
    candidate_count = 0
    started = time()
    while !state.game_over_flag && steps < max_steps
        node, generation, inference_time, candidates = select_openvino_node(
            state, inference; next_count
        )
        isnothing(node) && break
        generation_seconds += generation
        inference_seconds += inference_time
        candidate_count += candidates
        blocks_before = sum(state.current_game_board.binary)
        apply_node!(state, node)
        cleared = (blocks_before + 4 - sum(state.current_game_board.binary)) ÷ 10
        tetrises += cleared == 4
        tspins += node.tspin > 0
        perfect_clears += cleared > 0 && iszero(sum(state.current_game_board.binary))
        steps += 1
        if steps == 1 || steps % 25 == 0 || state.game_over_flag
            @info "OpenVINO checkpoint episode" seed steps score=state.score candidates=candidate_count generation_seconds inference_seconds wall_seconds=time()-started
        end
    end
    return (;
        seed,
        score=state.score,
        steps,
        game_over=state.game_over_flag,
        tetrises,
        tspins,
        perfect_clears,
        candidate_count,
        generation_seconds,
        inference_seconds,
        wall_seconds=time() - started,
        candidates_per_inference_second=candidate_count / inference_seconds,
    )
end

function openvino_main()
    sys = pyimport("sys")
    sys.path.insert(0, joinpath(ROOT, "tools"))
    legacy_openvino = pyimport("legacy_openvino")
    device = get(ENV, "OPENVINO_DEVICE", "NPU")
    batch_size = parse(Int, get(ENV, "OPENVINO_BATCH", "16"))
    @info "Compiling OpenVINO legacy policy" device batch_size
    inference = legacy_openvino.LegacyOpenVINOInference(device, batch_size)
    seed = parse(Int, get(ENV, "LEGACY_EVAL_SEED", "5742"))
    max_steps = parse(Int, get(ENV, "LEGACY_EVAL_STEPS", string(PAPER_EPISODE_STEPS)))
    next_count = parse(Int, get(ENV, "LEGACY_EVAL_NEXT", "5"))
    result = evaluate_openvino_episode(seed, inference; next_count, max_steps)
    @info "OpenVINO checkpoint result" result
    output_path = joinpath(
        ROOT,
        "runs",
        "openvino_checkpoint_$(lowercase(device))_seed$(seed)_next$(next_count).json",
    )
    open(output_path, "w") do io
        JSON3.pretty(
            io,
            (;
                timestamp=string(now()),
                checkpoint=joinpath(ROOT, "1313", "mainmodel copy 3.jld2"),
                julia_version=string(VERSION),
                lux_version=string(Base.pkgversion(Lux)),
                openvino_version=pyconvert(String, pyimport("openvino").__version__),
                device,
                batch_size,
                next_count,
                max_steps,
                result...,
            ),
        )
    end
    @info "Saved OpenVINO evaluation" output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    openvino_main()
end
