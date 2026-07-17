include(joinpath(@__DIR__, "benchmark_legacy_engine.jl"))

struct BeamState
    state::GameState
    path::Vector{Node}
    priority::Float64
end

function visible_mino_key(state::GameState)
    type_index(mino) = isnothing(mino) ? 0 : something(
        findfirst(candidate -> typeof(candidate) === typeof(mino), Tetris.MINOS), 0
    )
    visible = last(state.mino_list, min(6, length(state.mino_list)))
    return (
        type_index(state.current_mino),
        type_index(state.hold_mino),
        Tuple(type_index.(visible)),
    )
end

function beam_state_key(state::GameState)
    return (
        Tuple(vec(state.current_game_board.binary)),
        visible_mino_key(state),
        state.back_to_back_flag,
        state.ren,
    )
end

function perfect_clear_priority(root_score::Int, state::GameState)
    features = board_features(state.current_game_board.binary)
    reward = state.score - root_score
    return 12.0 * reward -
           1200.0 * features.holes -
           8.0 * features.hole_depth -
           5.0 * features.aggregate_height -
           10.0 * features.bumpiness -
           8.0 * features.max_height -
           2.0 * features.row_transitions
end

function find_perfect_clear(
    initial_state::GameState; depth::Int=5, beam_width::Int=200
)
    root_score = initial_state.score
    beam = [BeamState(GameState(initial_state), Node[], 0.0)]
    started = time()
    for ply in 1:depth
        deduplicated = Dict{Any,BeamState}()
        expanded = 0
        for parent in beam
            nodes = stable_node_list(parent.state)
            expanded += length(nodes)
            for node in nodes
                child = node.game_state
                child.game_over_flag && continue
                path = [parent.path; node]
                if iszero(sum(child.current_game_board.binary)) && child.score > root_score
                    @info "Found perfect clear" ply score_gain=child.score-root_score expanded elapsed=time()-started
                    return path
                end
                priority = perfect_clear_priority(root_score, child)
                candidate = BeamState(child, path, priority)
                key = beam_state_key(child)
                previous = get(deduplicated, key, nothing)
                if isnothing(previous) || candidate.priority > previous.priority
                    deduplicated[key] = candidate
                end
            end
        end
        candidates = collect(values(deduplicated))
        keep = min(beam_width, length(candidates))
        indices = partialsortperm(
            getproperty.(candidates, :priority), 1:keep; rev=true
        )
        beam = candidates[indices]
        @info "Perfect-clear beam" ply expanded unique=length(candidates) kept=length(beam) best=first(beam).priority elapsed=time()-started
    end
    @info "No perfect clear found" depth beam_width elapsed=time()-started
    return Node[]
end

function main()
    seed = parse(Int, get(ENV, "PC_SEED", "5742"))
    depth = parse(Int, get(ENV, "PC_DEPTH", "5"))
    beam_width = parse(Int, get(ENV, "PC_BEAM", "200"))
    state = GameState(Xoshiro(seed))
    path = find_perfect_clear(state; depth, beam_width)
    for (step, node) in enumerate(path)
        apply_node!(state, node)
        @info "Applied perfect-clear path" step score=state.score blocks=sum(state.current_game_board.binary)
    end
    println((; seed, found=!isempty(path), path_length=length(path), score=state.score, blocks=sum(state.current_game_board.binary)))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
