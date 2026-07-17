include(joinpath(@__DIR__, "evaluate_openvino_checkpoint.jl"))

function select_ensemble_node(
    state::GameState, inference; weight_next5::Float32=0.5f0
)
    generation_seconds = @elapsed nodes = stable_node_list(state)
    isempty(nodes) && return nothing, generation_seconds, 0.0, 0
    inference_seconds = @elapsed begin
        scores4 = openvino_scores(
            inference, legacy_candidate_batch(state, nodes; next_count=4)
        )
        scores5 = openvino_scores(
            inference, legacy_candidate_batch(state, nodes; next_count=5)
        )
    end
    scores = scores4 .+ weight_next5 .* (scores5 .- scores4)
    return nodes[argmax(scores)], generation_seconds, inference_seconds, length(nodes)
end

function evaluate_ensemble_episode(
    seed, inference; weight_next5::Float32=0.5f0, max_steps::Int=PAPER_EPISODE_STEPS
)
    state = GameState(Xoshiro(seed))
    steps = 0
    candidate_count = 0
    generation_seconds = 0.0
    inference_seconds = 0.0
    started = time()
    while !state.game_over_flag && steps < max_steps
        node, generation, inference_time, candidates = select_ensemble_node(
            state, inference; weight_next5
        )
        isnothing(node) && break
        generation_seconds += generation
        inference_seconds += inference_time
        candidate_count += candidates
        apply_node!(state, node)
        steps += 1
        if steps == 1 || steps % 25 == 0
            @info "NEXT ensemble episode" seed steps score=state.score weight_next5 generation_seconds inference_seconds wall_seconds=time()-started
        end
    end
    return (;
        seed,
        score=state.score,
        steps,
        game_over=state.game_over_flag,
        candidate_count,
        generation_seconds,
        inference_seconds,
        wall_seconds=time() - started,
    )
end

function main()
    sys = pyimport("sys")
    sys.path.insert(0, joinpath(ROOT, "tools"))
    legacy_openvino = pyimport("legacy_openvino")
    inference = legacy_openvino.LegacyOpenVINOInference("NPU", 16)
    seed = parse(Int, get(ENV, "LEGACY_EVAL_SEED", "5742"))
    max_steps = parse(Int, get(ENV, "LEGACY_EVAL_STEPS", string(PAPER_EPISODE_STEPS)))
    weight_next5 = parse(Float32, get(ENV, "ENSEMBLE_WEIGHT_NEXT5", "0.5"))
    result = evaluate_ensemble_episode(seed, inference; weight_next5, max_steps)
    @info "NEXT ensemble result" result weight_next5
    output_path = joinpath(
        ROOT, "runs", "next_ensemble_w$(weight_next5)_seed$(seed).json"
    )
    open(output_path, "w") do io
        JSON3.pretty(io, (; timestamp=string(now()), weight_next5, max_steps, result...))
    end
end

main()
