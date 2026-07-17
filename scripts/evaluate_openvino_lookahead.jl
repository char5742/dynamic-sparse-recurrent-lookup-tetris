include(joinpath(@__DIR__, "evaluate_openvino_checkpoint.jl"))
include(joinpath(@__DIR__, "lookahead_artifact_helpers.jl"))

function select_openvino_lookahead_node(
    state::GameState,
    inference;
    next_count::Int=5,
    top_k::Int=S1_TOP_K,
    blend::Float32=S1_BLEND,
    discount::Float32=S1_DISCOUNT,
    q_margin_threshold::Float32=S1_Q_MARGIN_THRESHOLD,
    inference_batch_size::Int=16,
)
    generation_seconds = @elapsed nodes = stable_node_list(state)
    isempty(nodes) && return (
        nothing,
        generation_seconds,
        0.0,
        lookahead_evaluation_accounting(0, Int[], inference_batch_size),
    )
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
            lookahead_evaluation_accounting(
                length(nodes), Int[], inference_batch_size
            ),
        )
    end
    reranked_scores = fill(-Inf32, length(nodes))
    successor_candidate_counts = Int[]
    for index in selected_indices
        node = nodes[index]
        immediate_reward = Float32(node.game_state.score - state.score) / 600.0f0
        if node.game_state.game_over_flag
            bellman_score = -1000.0f0 / 600.0f0
        else
            generation_seconds += @elapsed next_nodes = stable_node_list(node.game_state)
            push!(successor_candidate_counts, length(next_nodes))
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
        lookahead_evaluation_accounting(
            length(nodes), successor_candidate_counts, inference_batch_size
        ),
    )
end

function evaluate_openvino_lookahead_episode(
    seed::Integer,
    inference;
    next_count::Int=5,
    top_k::Int=S1_TOP_K,
    blend::Float32=S1_BLEND,
    discount::Float32=S1_DISCOUNT,
    q_margin_threshold::Float32=S1_Q_MARGIN_THRESHOLD,
    max_steps::Int=PAPER_EPISODE_STEPS,
    inference_batch_size::Int=16,
)
    validate_s1_search_constants(top_k, blend, discount, q_margin_threshold)
    state = GameState(Xoshiro(seed))
    steps = 0
    generation_seconds = 0.0
    inference_seconds = 0.0
    candidate_count = 0
    lookahead_candidate_count = 0
    lookahead_expansions = 0
    logical_model_passes = 0
    physical_backend_requests = 0
    tspins = 0
    started = time()
    while !state.game_over_flag && steps < max_steps
        node, generation, inference_time, accounting =
            select_openvino_lookahead_node(
                state,
                inference;
                next_count,
                top_k,
                blend,
                discount,
                q_margin_threshold,
                inference_batch_size,
            )
        isnothing(node) && break
        generation_seconds += generation
        inference_seconds += inference_time
        candidate_count += accounting.root_candidate_evaluations
        lookahead_candidate_count += accounting.successor_candidate_evaluations
        lookahead_expansions += accounting.lookahead_expansions
        logical_model_passes += accounting.logical_model_passes
        physical_backend_requests += accounting.physical_backend_requests
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
        root_candidate_evaluations=candidate_count,
        successor_candidate_evaluations=lookahead_candidate_count,
        lookahead_expansions,
        logical_model_passes,
        physical_backend_requests,
        generation_seconds,
        inference_seconds,
        wall_seconds=time() - started,
    )
end

function main()
    sys = pyimport("sys")
    sys.path.insert(0, joinpath(ROOT, "tools"))
    legacy_openvino = pyimport("legacy_openvino")
    checkpoint_path = normpath(
        abspath(joinpath(ROOT, "1313", "mainmodel copy 3.jld2"))
    )
    checkpoint_fingerprint = checkpoint_file_fingerprint(checkpoint_path)
    device = get(ENV, "OPENVINO_DEVICE", "NPU")
    batch_size = parse(Int, get(ENV, "OPENVINO_BATCH", "16"))
    @info "Compiling OpenVINO lookahead policy" device batch_size
    inference = legacy_openvino.LegacyOpenVINOInference(device, batch_size)
    seed = parse(Int, get(ENV, "LEGACY_EVAL_SEED", "5742"))
    max_steps = parse(Int, get(ENV, "LEGACY_EVAL_STEPS", string(PAPER_EPISODE_STEPS)))
    next_count = parse(Int, get(ENV, "LEGACY_EVAL_NEXT", "5"))
    top_k = parse(Int, get(ENV, "LOOKAHEAD_TOP_K", string(S1_TOP_K)))
    blend = parse(Float32, get(ENV, "LOOKAHEAD_BLEND", string(S1_BLEND)))
    discount = parse(Float32, get(ENV, "LOOKAHEAD_DISCOUNT", string(S1_DISCOUNT)))
    q_margin_threshold = parse(
        Float32, get(ENV, "LOOKAHEAD_Q_MARGIN", "Inf")
    )
    search_constants = validate_s1_search_constants(
        top_k, blend, discount, q_margin_threshold
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
        inference_batch_size=batch_size,
    )
    @info "OpenVINO lookahead result" result top_k blend discount q_margin_threshold
    output_path = joinpath(
        ROOT,
        "runs",
        lookahead_result_filename(
            seed,
            next_count,
            max_steps;
            device,
            top_k,
            blend,
            discount,
            q_margin_threshold,
        ),
    )
    open(output_path, "w") do io
        JSON3.pretty(
            io,
            (;
                timestamp=string(now()),
                checkpoint=checkpoint_path,
                checkpoint_fingerprint,
                backend="OpenVINO static accelerator with dynamic CPU tail",
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
                search_constants,
                search_budget=(;
                    episode_piece_limit=max_steps,
                    next_count,
                    hold_enabled=true,
                    candidate_order="stable_node_key",
                    selection="top2_one_step_bellman_rerank",
                    inference_batch_size=batch_size,
                ),
                result...,
            ),
        )
    end
    @info "Saved OpenVINO lookahead evaluation" output_path
end

main()
