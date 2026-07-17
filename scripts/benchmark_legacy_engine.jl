using Random
using Statistics
using Printf
using Tetris

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "upstream", "TetrisAI", "src", "core", "components", "node.jl"))
include(joinpath(ROOT, "upstream", "TetrisAI", "src", "core", "analyzer.jl"))

Base.@kwdef struct HeuristicWeights
    reward::Float64 = 1.0
    aggregate_height::Float64 = -2.8
    holes::Float64 = -34.0
    bumpiness::Float64 = -3.2
    max_height::Float64 = -5.0
    wells::Float64 = -1.5
    row_transitions::Float64 = -2.5
    column_transitions::Float64 = -3.5
    back_to_back::Float64 = 22.0
    internal_bumpiness::Float64 = 0.0
    right_well_depth::Float64 = 0.0
    right_column_height::Float64 = 0.0
    hold_i_reserve::Float64 = 0.0
    tetris_clear::Float64 = 0.0
    non_tetris_line::Float64 = 0.0
    wasted_i::Float64 = 0.0
    left_stack_range::Float64 = 0.0
    hole_depth::Float64 = 0.0
    tetris_overhang::Float64 = 0.0
    danger_height::Float64 = 16.0
end

best_teacher_weights() = HeuristicWeights(
    reward=0.5,
    aggregate_height=-0.2,
    holes=-70.0,
    bumpiness=0.0,
    max_height=-2.0,
    wells=0.0,
    row_transitions=-0.5,
    column_transitions=-1.0,
    back_to_back=30.0,
    internal_bumpiness=-4.0,
    right_well_depth=12.0,
    right_column_height=-15.0,
    tetris_clear=300.0,
    non_tetris_line=-100.0,
    left_stack_range=-4.0,
)

function board_features(board::AbstractMatrix)
    height, width = size(board)
    heights = zeros(Int, width)
    holes = 0
    hole_depth = 0
    column_transitions = 0

    @inbounds for x in 1:width
        seen_block = false
        blocks_above = 0
        previous = true # upper wall
        for y in 1:height
            occupied = !iszero(board[y, x])
            column_transitions += occupied != previous
            previous = occupied
            if occupied && !seen_block
                heights[x] = height - y + 1
                seen_block = true
                blocks_above += 1
            elseif occupied
                blocks_above += 1
            elseif !occupied && seen_block
                holes += 1
                hole_depth += blocks_above
            end
        end
        column_transitions += !previous # floor is occupied
    end

    row_transitions = 0
    @inbounds for y in 1:height
        previous = true # left wall
        for x in 1:width
            occupied = !iszero(board[y, x])
            row_transitions += occupied != previous
            previous = occupied
        end
        row_transitions += !previous # right wall
    end

    bumpiness = sum(abs(heights[x] - heights[x + 1]) for x in 1:(width - 1))
    internal_bumpiness = sum(abs(heights[x] - heights[x + 1]) for x in 1:(width - 2))
    right_well_depth = max(0, minimum(@view heights[1:(width - 1)]) - heights[width])
    tetris_ceiling = heights[width] + 4
    tetris_overhang = sum(max(0, heights[x] - tetris_ceiling) for x in 1:(width - 1))
    wells = 0
    @inbounds for x in 1:width
        left = x == 1 ? height : heights[x - 1]
        right = x == width ? height : heights[x + 1]
        depth = max(0, min(left, right) - heights[x])
        wells += depth * (depth + 1) ÷ 2
    end

    return (
        aggregate_height=sum(heights),
        holes=holes,
        bumpiness=bumpiness,
        max_height=maximum(heights),
        wells=wells,
        row_transitions=row_transitions,
        column_transitions=column_transitions,
        internal_bumpiness=internal_bumpiness,
        right_well_depth=right_well_depth,
        right_column_height=heights[width],
        left_stack_range=maximum(@view heights[1:(width - 1)]) - minimum(@view heights[1:(width - 1)]),
        hole_depth=hole_depth,
        tetris_overhang=tetris_overhang,
    )
end

function board_max_height(board::AbstractMatrix)
    height, width = size(board)
    max_height = 0
    @inbounds for x in 1:width, y in 1:height
        if !iszero(board[y, x])
            max_height = max(max_height, height - y + 1)
            break
        end
    end
    return max_height
end

function heuristic_value(root::GameState, node::Node, weights::HeuristicWeights)
    features = board_features(node.game_state.current_game_board.binary)
    reward = node.game_state.score - root.score
    hold_i_reserve = node.game_state.hold_mino isa Tetris.IMino && features.right_well_depth < 4
    cleared_lines = (sum(root.current_game_board.binary) + 4 - sum(node.game_state.current_game_board.binary)) ÷ 10
    is_tetris = cleared_lines == 4
    # Below the danger zone, non-Tetris clears waste the prepared well. Close
    # to the ceiling they are allowed as a rescue action instead of forcing a
    # game over for the sake of strategic purity.
    root_max_height = board_max_height(root.current_game_board.binary)
    non_tetris_lines = is_tetris || root_max_height >= weights.danger_height ? 0 : cleared_lines
    wasted_i = node.mino isa Tetris.IMino && !is_tetris && root_max_height < weights.danger_height
    return weights.reward * reward +
           weights.aggregate_height * features.aggregate_height +
           weights.holes * features.holes +
           weights.bumpiness * features.bumpiness +
           weights.max_height * features.max_height +
           weights.wells * features.wells +
           weights.row_transitions * features.row_transitions +
           weights.column_transitions * features.column_transitions +
           weights.back_to_back * node.game_state.back_to_back_flag +
           weights.internal_bumpiness * features.internal_bumpiness +
           weights.right_well_depth * min(features.right_well_depth, 4) +
           weights.right_column_height * features.right_column_height +
           weights.hold_i_reserve * hold_i_reserve +
           weights.tetris_clear * is_tetris +
           weights.non_tetris_line * non_tetris_lines +
           weights.wasted_i * wasted_i +
           weights.left_stack_range * features.left_stack_range +
           weights.hole_depth * features.hole_depth +
           weights.tetris_overhang * features.tetris_overhang
end


function heuristic_feature_vector(root::GameState, node::Node)
    features = board_features(node.game_state.current_game_board.binary)
    reward = node.game_state.score - root.score
    hold_i_reserve = node.game_state.hold_mino isa Tetris.IMino && features.right_well_depth < 4
    cleared_lines = (sum(root.current_game_board.binary) + 4 - sum(node.game_state.current_game_board.binary)) ÷ 10
    is_tetris = cleared_lines == 4
    root_max_height = board_max_height(root.current_game_board.binary)
    non_tetris_lines = is_tetris || root_max_height >= 16 ? 0 : cleared_lines
    wasted_i = node.mino isa Tetris.IMino && !is_tetris && root_max_height < 16

    return Float32[
        reward,
        features.aggregate_height,
        features.holes,
        features.bumpiness,
        features.max_height,
        features.wells,
        features.row_transitions,
        features.column_transitions,
        node.game_state.back_to_back_flag,
        features.internal_bumpiness,
        min(features.right_well_depth, 4),
        features.right_column_height,
        hold_i_reserve,
        is_tetris,
        non_tetris_lines,
        wasted_i,
        features.left_stack_range,
        features.hole_depth,
        features.tetris_overhang,
    ]
end

function heuristic_weight_vector(weights::HeuristicWeights)
    return Float32[
        weights.reward,
        weights.aggregate_height,
        weights.holes,
        weights.bumpiness,
        weights.max_height,
        weights.wells,
        weights.row_transitions,
        weights.column_transitions,
        weights.back_to_back,
        weights.internal_bumpiness,
        weights.right_well_depth,
        weights.right_column_height,
        weights.hold_i_reserve,
        weights.tetris_clear,
        weights.non_tetris_line,
        weights.wasted_i,
        weights.left_stack_range,
        weights.hole_depth,
        weights.tetris_overhang,
    ]
end

function stable_node_key(node::Node)
    mino_index = something(
        findfirst(candidate -> typeof(candidate) === typeof(node.mino), Tetris.MINOS),
        0,
    )
    return (
        Tuple(vec(node.game_state.current_game_board.binary)),
        mino_index,
        Int(node.mino.direction),
        Int(node.position.y),
        Int(node.position.x),
        node.tspin,
        length(node.action_list),
    )
end

function stable_node_list(state::GameState)
    nodes = get_node_list(state)
    sort!(nodes; by=stable_node_key)
    return nodes
end



function select_heuristic_node(state::GameState, weights::HeuristicWeights)
    nodes = stable_node_list(state)
    isempty(nodes) && return nothing
    _, index = findmax(node -> heuristic_value(state, node, weights), nodes)
    return nodes[index]
end

function select_beam_node(
    state::GameState,
    weights::HeuristicWeights;
    beam_width::Int=4,
    discount::Float64=0.8,
)
    nodes, best_index, _ = rank_beam_nodes(state, weights; beam_width, discount)
    isempty(nodes) && return nothing
    return nodes[best_index]
end

function rank_beam_nodes(
    state::GameState,
    weights::HeuristicWeights;
    beam_width::Int=4,
    discount::Float64=0.8,
)
    nodes = stable_node_list(state)
    isempty(nodes) && return nodes, 0, Float64[]
    first_values = [heuristic_value(state, node, weights) for node in nodes]
    width = min(beam_width, length(nodes))
    candidate_indices = partialsortperm(first_values, 1:width; rev=true)

    best_index = candidate_indices[1]
    best_value = -Inf
    for index in candidate_indices
        node = nodes[index]
        if node.game_state.game_over_flag
            combined_value = first_values[index] - 10_000
        else
            next_nodes = stable_node_list(node.game_state)
            future_value = isempty(next_nodes) ? -10_000.0 : maximum(
                child -> heuristic_value(node.game_state, child, weights), next_nodes
            )
            combined_value = first_values[index] + discount * future_value
        end
        if combined_value > best_value
            best_value = combined_value
            best_index = index
        end
    end
    return nodes, best_index, first_values
end


function mino_one_hot(mino::Union{Nothing,AbstractMino})
    result = zeros(Float32, 7)
    isnothing(mino) && return result
    index = findfirst(candidate -> typeof(candidate) === typeof(mino), Tetris.MINOS)
    !isnothing(index) && (result[index] = 1.0f0)
    return result
end


function model_feature_vector(root::GameState, node::Node)
    afterstate = node.game_state
    queue = [afterstate.current_mino, afterstate.hold_mino, afterstate.mino_list[end-4:end]...]
    return vcat(
        heuristic_feature_vector(root, node),
        vec(Float32.(afterstate.current_game_board.binary)),
        reduce(vcat, mino_one_hot(mino) for mino in queue),
    )
end

function apply_node!(state::GameState, node::Node)
    for action in node.action_list
        action!(state, action)
    end
    put_mino!(state)
    return state
end

function play_episode(
    seed::Integer,
    weights::HeuristicWeights;
    max_steps::Int=250,
    beam_width::Int=1,
    beam_discount::Float64=0.8,
)
    state = GameState(Xoshiro(seed))
    steps = 0
    lines_cleared = 0
    tetrises = 0
    tspins = 0
    i_placements = 0
    i_tetrises = 0
    perfect_clears = 0
    while !state.game_over_flag && steps < max_steps
        node = beam_width == 1 ?
               select_heuristic_node(state, weights) :
               select_beam_node(state, weights; beam_width, discount=beam_discount)
        isnothing(node) && break
        blocks_before = sum(state.current_game_board.binary)
        apply_node!(state, node)
        cleared = (blocks_before + 4 - sum(state.current_game_board.binary)) ÷ 10
        lines_cleared += cleared
        tetrises += cleared == 4
        tspins += node.tspin > 0
        placed_i = node.mino isa Tetris.IMino
        i_placements += placed_i
        i_tetrises += placed_i && cleared == 4
        perfect_clears += cleared > 0 && iszero(sum(state.current_game_board.binary))
        steps += 1
    end
    final_features = board_features(state.current_game_board.binary)
    return (;
        score=state.score,
        steps,
        game_over=state.game_over_flag,
        lines_cleared,
        tetrises,
        tspins,
        i_placements,
        i_tetrises,
        perfect_clears,
        back_to_back=state.back_to_back_flag,
        final_holes=final_features.holes,
        final_max_height=final_features.max_height,
    )
end

function benchmark_generation(samples::Int=1_000)
    state = GameState(Xoshiro(0x5742))
    stable_node_list(state) # compile
    GC.gc()
    allocated = @allocated elapsed = @elapsed begin
        for _ in 1:samples
            stable_node_list(state)
        end
    end
    return (
        samples=samples,
        seconds=elapsed,
        generations_per_second=samples / elapsed,
        allocated_mib=allocated / 2.0^20,
        allocated_kib_per_generation=allocated / samples / 2.0^10,
    )
end

function main()
    generation_samples = parse(Int, get(ENV, "BENCH_GENERATION_SAMPLES", "100"))
    episode_count = parse(Int, get(ENV, "BENCH_EPISODES", "3"))
    generation = benchmark_generation(generation_samples)
    @printf("node_generation samples=%d seconds=%.3f rate=%.2f/s allocated=%.1fMiB %.1fKiB/sample\n",
        generation.samples,
        generation.seconds,
        generation.generations_per_second,
        generation.allocated_mib,
        generation.allocated_kib_per_generation)
    flush(stdout)

    weights = HeuristicWeights()
    play_episode(1, weights; max_steps=5) # compile full path
    GC.gc()
    results = Vector{NamedTuple}(undef, episode_count)
    elapsed = @elapsed for seed in eachindex(results)
        results[seed] = play_episode(seed, weights)
        @printf("episode seed=%d score=%d steps=%d game_over=%s lines=%d tetrises=%d tspins=%d i=%d i_tetrises=%d holes=%d max_height=%d\n",
            seed, results[seed].score, results[seed].steps, results[seed].game_over,
            results[seed].lines_cleared, results[seed].tetrises, results[seed].tspins,
            results[seed].i_placements, results[seed].i_tetrises,
            results[seed].final_holes, results[seed].final_max_height)
        flush(stdout)
    end
    scores = getproperty.(results, :score)
    steps = getproperty.(results, :steps)
    @printf("heuristic episodes=%d mean_score=%.1f median_score=%.1f min=%d max=%d steps_per_second=%.2f seconds=%.3f\n",
        length(results), mean(scores), median(scores), minimum(scores), maximum(scores), sum(steps) / elapsed, elapsed)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
