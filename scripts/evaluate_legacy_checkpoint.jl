include(joinpath(@__DIR__, "benchmark_legacy_engine.jl"))

using TetrisPaperPlus
using Lux
using JLD2
using JSON3
using Dates
using LinearAlgebra

function legacy_mino_one_hot(mino::Union{Nothing,AbstractMino})
    result = zeros(Float32, 7)
    isnothing(mino) && return result
    index = findfirst(candidate -> typeof(candidate) === typeof(mino), Tetris.MINOS)
    !isnothing(index) && (result[index] = 1.0f0)
    return result
end

function legacy_candidate_batch(state::GameState, nodes::Vector{Node}; next_count::Int=5)
    candidate_count = length(nodes)
    board = Array{Float32}(undef, 24, 10, 1, candidate_count)
    placement = Array{Float32}(undef, 24, 10, 1, candidate_count)
    ren = fill(Float32(state.ren), 1, candidate_count)
    back_to_back = fill(Float32(state.back_to_back_flag), 1, candidate_count)
    tspin = Array{Float32}(undef, 1, candidate_count)
    queue = Array{Float32}(undef, 7, 6, candidate_count)

    board[:, :, 1, :] .= reshape(
        Float32.(state.current_game_board.binary), 24, 10, 1, 1
    )
    hold_and_next = [state.hold_mino, state.mino_list[end-4:end]...]
    queue_template = reduce(hcat, legacy_mino_one_hot(mino) for mino in hold_and_next)
    if next_count < 5
        queue_template[:, (2 + next_count):end] .= 0.0f0
    end
    next_count == 0 && (queue_template[:, 1] .= 0.0f0)

    @inbounds for index in eachindex(nodes)
        node = nodes[index]
        placement[:, :, 1, index] .= Float32.(generate_minopos(node.mino, node.position))
        tspin[1, index] = node.tspin > 1 ? 1.0f0 : 0.0f0
        queue[:, :, index] .= queue_template
    end
    return (board, placement, ren, back_to_back, tspin, queue)
end

function select_legacy_node(
    state::GameState,
    model,
    parameters,
    model_state;
    next_count::Int=5,
    inference_batch_size::Int=16,
)
    generation_seconds = @elapsed nodes = stable_node_list(state)
    isempty(nodes) && return nothing, generation_seconds, 0.0, 0
    input = legacy_candidate_batch(state, nodes; next_count)
    score_batches = Matrix{Float32}[]
    inference_seconds = @elapsed begin
        for start_index in 1:inference_batch_size:length(nodes)
            end_index = min(start_index + inference_batch_size - 1, length(nodes))
            batch_input = map(value -> selectdim(value, ndims(value), start_index:end_index), input)
            batch_scores, _ = model(batch_input, parameters, model_state)
            push!(score_batches, Array(batch_scores))
        end
    end
    scores = reduce(hcat, score_batches)
    all(isfinite, scores) || error("Legacy checkpoint produced a non-finite score")
    return nodes[argmax(vec(scores))], generation_seconds, inference_seconds, length(nodes)
end

function evaluate_legacy_episode(
    seed::Integer,
    model,
    parameters,
    model_state;
    next_count::Int=5,
    inference_batch_size::Int=16,
    max_steps::Int=PAPER_EPISODE_STEPS,
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
        node, generation, inference, candidates = select_legacy_node(
            state,
            model,
            parameters,
            model_state;
            next_count,
            inference_batch_size,
        )
        isnothing(node) && break
        generation_seconds += generation
        inference_seconds += inference
        candidate_count += candidates
        blocks_before = sum(state.current_game_board.binary)
        apply_node!(state, node)
        cleared = (blocks_before + 4 - sum(state.current_game_board.binary)) ÷ 10
        tetrises += cleared == 4
        tspins += node.tspin > 0
        perfect_clears += cleared > 0 && iszero(sum(state.current_game_board.binary))
        steps += 1
        if steps == 1 || steps % 25 == 0 || state.game_over_flag
            @info "Legacy checkpoint episode" seed steps score=state.score candidates=candidate_count generation_seconds inference_seconds wall_seconds=time()-started
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

function main()
    checkpoint_path = joinpath(ROOT, "1313", "mainmodel copy 3.jld2")
    parameters, model_state = jldopen(checkpoint_path, "r") do file
        modernize_legacy_parameters(file["ps"]), Lux.testmode(file["st"])
    end
    model = LegacyQNetwork()
    blas_threads = parse(Int, get(ENV, "LEGACY_BLAS_THREADS", "12"))
    BLAS.set_num_threads(min(blas_threads, Threads.nthreads()))
    seed = parse(Int, get(ENV, "LEGACY_EVAL_SEED", "5742"))
    max_steps = parse(Int, get(ENV, "LEGACY_EVAL_STEPS", string(PAPER_EPISODE_STEPS)))
    next_count = parse(Int, get(ENV, "LEGACY_EVAL_NEXT", "5"))
    inference_batch_size = parse(Int, get(ENV, "LEGACY_INFERENCE_BATCH", "16"))
    result = evaluate_legacy_episode(
        seed,
        model,
        parameters,
        model_state;
        next_count,
        inference_batch_size,
        max_steps,
    )
    @info "Legacy checkpoint result" result
    output_path = joinpath(ROOT, "runs", "legacy_checkpoint_seed$(seed)_next$(next_count).json")
    open(output_path, "w") do io
        JSON3.pretty(
            io,
            (;
                timestamp=string(now()),
                checkpoint=checkpoint_path,
                julia_version=string(VERSION),
                lux_version=string(Base.pkgversion(Lux)),
                next_count,
                inference_batch_size,
                max_steps,
                result...,
            ),
        )
    end
    @info "Saved legacy evaluation" output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
