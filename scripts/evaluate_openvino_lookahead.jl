include(joinpath(@__DIR__, "evaluate_openvino_checkpoint.jl"))

function select_openvino_lookahead_node(
    state::GameState,
    inference;
    next_count::Int=5,
    top_k::Int=4,
    blend::Float32=1.0f0,
    discount::Float32=0.997f0,
    q_margin_threshold::Float32=Inf32,
)
    generation_seconds = @elapsed nodes = stable_node_list(state)
    isempty(nodes) && return nothing, generation_seconds, 0.0, 0, 0
    input = legacy_candidate_batch(state, nodes; next_count)
    inference_seconds = @elapsed scores = openvino_scores(inference, input)
    all(isfinite, scores) || error("OpenVINO checkpoint produced a non-finite score")

    selected_indices = partialsortperm(
        scores, 1:min(top_k, length(scores)); rev=true
    )
    if length(selected_indices) > 1 &&
       scores[selected_indices[1]] - scores[selected_indices[2]] >= q_margin_threshold
        return (
            nodes[selected_indices[1]],
            generation_seconds,
            inference_seconds,
            length(nodes),
            0,
        )
    end
    reranked_scores = fill(-Inf32, length(nodes))
    lookahead_candidates = 0
    for index in selected_indices
        node = nodes[index]
        immediate_reward = Float32(node.game_state.score - state.score) / 600.0f0
        if node.game_state.game_over_flag
            bellman_score = -1000.0f0 / 600.0f0
        else
            generation_seconds += @elapsed next_nodes = stable_node_list(node.game_state)
            lookahead_candidates += length(next_nodes)
            if isempty(next_nodes)
                bellman_score = -1000.0f0 / 600.0f0
            else
                next_input = legacy_candidate_batch(
                    node.game_state, next_nodes; next_count
                )
                inference_seconds += @elapsed next_scores = openvino_scores(
                    inference, next_input
                )
                bellman_score = immediate_reward + discount * maximum(next_scores)
            end
        end
        reranked_scores[index] = muladd(
            blend, bellman_score - scores[index], scores[index]
        )
    end
    best_index = argmax(reranked_scores)
    return (
        nodes[best_index],
        generation_seconds,
        inference_seconds,
        length(nodes),
        lookahead_candidates,
    )
end

function evaluate_openvino_lookahead_episode(
    seed::Integer,
    inference;
    next_count::Int=5,
    top_k::Int=4,
    blend::Float32=1.0f0,
    discount::Float32=0.997f0,
    q_margin_threshold::Float32=Inf32,
    max_steps::Int=PAPER_EPISODE_STEPS,
)
    state = GameState(Xoshiro(seed))
    steps = 0
    generation_seconds = 0.0
    inference_seconds = 0.0
    candidate_count = 0
    lookahead_candidate_count = 0
    tspins = 0
    started = time()
    while !state.game_over_flag && steps < max_steps
        node, generation, inference_time, candidates, lookahead_candidates =
            select_openvino_lookahead_node(
                state,
                inference;
                next_count,
                top_k,
                blend,
                discount,
                q_margin_threshold,
            )
        isnothing(node) && break
        generation_seconds += generation
        inference_seconds += inference_time
        candidate_count += candidates
        lookahead_candidate_count += lookahead_candidates
        tspins += node.tspin > 0
        apply_node!(state, node)
        steps += 1
        if steps == 1 || steps % 25 == 0 || state.game_over_flag
            @info "Lookahead checkpoint episode" seed steps score=state.score candidate_count lookahead_candidate_count generation_seconds inference_seconds wall_seconds=time()-started
        end
    end
    return (;
        seed,
        score=state.score,
        steps,
        game_over=state.game_over_flag,
        tspins,
        candidate_count,
        lookahead_candidate_count,
        generation_seconds,
        inference_seconds,
        wall_seconds=time() - started,
    )
end

function main()
    sys = pyimport("sys")
    sys.path.insert(0, joinpath(ROOT, "tools"))
    legacy_openvino = pyimport("legacy_openvino")
    device = get(ENV, "OPENVINO_DEVICE", "NPU")
    batch_size = parse(Int, get(ENV, "OPENVINO_BATCH", "16"))
    @info "Compiling OpenVINO lookahead policy" device batch_size
    inference = legacy_openvino.LegacyOpenVINOInference(device, batch_size)
    seed = parse(Int, get(ENV, "LEGACY_EVAL_SEED", "5742"))
    max_steps = parse(Int, get(ENV, "LEGACY_EVAL_STEPS", string(PAPER_EPISODE_STEPS)))
    next_count = parse(Int, get(ENV, "LEGACY_EVAL_NEXT", "5"))
    top_k = parse(Int, get(ENV, "LOOKAHEAD_TOP_K", "4"))
    blend = parse(Float32, get(ENV, "LOOKAHEAD_BLEND", "1.0"))
    discount = parse(Float32, get(ENV, "LOOKAHEAD_DISCOUNT", "0.997"))
    q_margin_threshold = parse(
        Float32, get(ENV, "LOOKAHEAD_Q_MARGIN", "Inf")
    )
    result = evaluate_openvino_lookahead_episode(
        seed,
        inference;
        next_count,
        top_k,
        blend,
        discount,
        q_margin_threshold,
        max_steps,
    )
    @info "OpenVINO lookahead result" result top_k blend discount q_margin_threshold
    output_path = joinpath(
        ROOT,
        "runs",
        "openvino_lookahead_k$(top_k)_b$(blend)_m$(isfinite(q_margin_threshold) ? q_margin_threshold : "all")_seed$(seed).json",
    )
    open(output_path, "w") do io
        JSON3.pretty(
            io,
            (;
                timestamp=string(now()),
                device,
                batch_size,
                next_count,
                top_k,
                blend,
                discount,
                q_margin_threshold=(
                    isfinite(q_margin_threshold) ? q_margin_threshold : nothing
                ),
                max_steps,
                result...,
            ),
        )
    end
    @info "Saved OpenVINO lookahead evaluation" output_path
end

main()
