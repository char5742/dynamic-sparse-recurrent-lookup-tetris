module CandidateDeltaInput

"""
Exact, fixed-capacity decomposition of the canonical Tetris sensory input.

The state-common component is prepared once for a board position.  Each legal
candidate then stores only its placement, T-spin flag, and the row permutation
induced by line clearing.  The canonical `board / after / added / removed /
queue / aux` representation can be reconstructed bit-for-bit without retaining
a candidate-sized copy of the common board and queue.
"""

export AUX_FEATURES,
    AUX_LEVELS,
    BOARD_CELLS,
    BOARD_COLUMNS,
    BOARD_ROWS,
    INPUT_RAILS,
    PLACEMENT_CAPACITY,
    QUEUE_PIECES,
    QUEUE_TOKENS,
    CandidateDelta,
    CandidateMaterialization,
    StateCommon,
    pack_candidate_rails!,
    placement_count,
    placement_position,
    prepare_candidate_delta!,
    prepare_state_common!,
    reconstruct_candidate!

const BOARD_ROWS = 24
const BOARD_COLUMNS = 10
const BOARD_CELLS = BOARD_ROWS * BOARD_COLUMNS
const QUEUE_PIECES = 7
const QUEUE_TOKENS = 6
const AUX_FEATURES = 37
const AUX_LEVELS = 8
const PLACEMENT_CAPACITY = 4
const INPUT_RAILS =
    4 * BOARD_CELLS + QUEUE_PIECES * QUEUE_TOKENS +
    AUX_FEATURES * AUX_LEVELS

"""
State data shared by every candidate of one Tetris position.

All buffers are matrices so their storage cannot be resized after construction.
The scalar values use `1 x 1` matrices for the same reason.
"""
struct StateCommon
    board::Matrix{UInt8}
    queue::Matrix{UInt8}
    ren::Matrix{Float32}
    back_to_back::Matrix{Float32}
end

StateCommon() = StateCommon(
    zeros(UInt8, BOARD_ROWS, BOARD_COLUMNS),
    zeros(UInt8, QUEUE_PIECES, QUEUE_TOKENS),
    zeros(Float32, 1, 1),
    zeros(Float32, 1, 1),
)

"""
Candidate-local input and exact line-clear permutation.

`source_to_after[row] == 0` marks a cleared row.  Otherwise it is the row in
the post-clear board that receives that source row.  `after_to_source` is the
inverse map; leading zero rows after a clear map to zero.
"""
struct CandidateDelta
    placement::Matrix{UInt8}
    placement_positions::Memory{UInt16}
    placement_count::Matrix{UInt8}
    tspin::Matrix{Float32}
    line_clear::Matrix{UInt8}
    full_rows::Matrix{Bool}
    source_to_after::Matrix{UInt8}
    after_to_source::Matrix{UInt8}
end

CandidateDelta() = CandidateDelta(
    zeros(UInt8, BOARD_ROWS, BOARD_COLUMNS),
    zeros(UInt16, PLACEMENT_CAPACITY),
    zeros(UInt8, 1, 1),
    zeros(Float32, 1, 1),
    zeros(UInt8, 1, 1),
    falses(BOARD_ROWS, 1),
    zeros(UInt8, BOARD_ROWS, 1),
    zeros(UInt8, BOARD_ROWS, 1),
)

"""Return the number of active raw-placement positions without allocating."""
@inline function placement_count(delta::CandidateDelta)
    return @inbounds Int(delta.placement_count[1])
end

"""
Return one active raw-placement position as a canonical column-major board ID.

Only indices in `1:placement_count(delta)` are valid.  The fixed-capacity
storage is deliberately not exposed as a variable-length view: callers can
iterate these two scalar accessors without constructing a `SubArray`.
"""
@inline function placement_position(delta::CandidateDelta, index::Int)
    count = placement_count(delta)
    1 <= index <= count || throw(BoundsError(delta.placement_positions, index))
    return @inbounds delta.placement_positions[index]
end

"""
Caller-owned reconstruction and geometry scratch for one candidate.

The three board planes and all 37 auxiliary values are observable outputs.
Geometry scratch is retained here so repeated reconstruction performs no heap
allocation.
"""
struct CandidateMaterialization
    after::Matrix{UInt8}
    added::Matrix{UInt8}
    removed::Matrix{UInt8}
    reachable::Matrix{Bool}
    flood_queue::Matrix{Int16}
    heights::Matrix{UInt8}
    holes::Matrix{UInt8}
    wells::Matrix{UInt8}
    aux::Matrix{Float32}
    geometry::Matrix{Int16}
end

CandidateMaterialization() = CandidateMaterialization(
    zeros(UInt8, BOARD_ROWS, BOARD_COLUMNS),
    zeros(UInt8, BOARD_ROWS, BOARD_COLUMNS),
    zeros(UInt8, BOARD_ROWS, BOARD_COLUMNS),
    falses(BOARD_ROWS, BOARD_COLUMNS),
    zeros(Int16, BOARD_CELLS, 1),
    zeros(UInt8, BOARD_COLUMNS, 1),
    zeros(UInt8, BOARD_COLUMNS, 1),
    zeros(UInt8, BOARD_COLUMNS, 1),
    zeros(Float32, AUX_FEATURES, 1),
    zeros(Int16, 4, 1),
)

@inline _bit(value) = value > 0.5 ? UInt8(1) : UInt8(0)
@inline _rail(value::UInt8) = iszero(value) ? 0.0f0 : 1.0f0
@inline _position(board_row::Int, column::Int) =
    UInt16(board_row + (column - 1) * BOARD_ROWS)

@inline function _finish_placement_positions!(
    delta::CandidateDelta,
    active_count::Int,
)
    if active_count > PLACEMENT_CAPACITY
        fill!(delta.placement_positions, UInt16(0))
        delta.placement_count[1] = UInt8(0)
        error("placement contains more than $PLACEMENT_CAPACITY active blocks")
    end
    delta.placement_count[1] = UInt8(active_count)
    return active_count
end

"""Prepare the common board, queue, REN, and back-to-back state once."""
function prepare_state_common!(
    common::StateCommon,
    dataset,
    row::Int,
)
    1 <= row <= size(dataset.boards, 4) ||
        throw(BoundsError(dataset.boards, (:, :, 1, row)))
    @inbounds for column in 1:BOARD_COLUMNS, board_row in 1:BOARD_ROWS
        common.board[board_row, column] =
            _bit(dataset.boards[board_row, column, 1, row])
    end
    @inbounds for token in 1:QUEUE_TOKENS, piece in 1:QUEUE_PIECES
        common.queue[piece, token] = _bit(dataset.queues[piece, token, row])
    end
    common.ren[1] = Float32(dataset.ren[1, row])
    common.back_to_back[1] = Float32(dataset.back_to_back[1, row])
    return common
end

@inline function _finish_row_map!(delta::CandidateDelta, common::StateCommon)
    full_count = 0
    @inbounds for board_row in 1:BOARD_ROWS
        occupied = 0
        for column in 1:BOARD_COLUMNS
            occupied += !iszero(common.board[board_row, column]) ||
                !iszero(delta.placement[board_row, column])
        end
        full = occupied == BOARD_COLUMNS
        delta.full_rows[board_row] = full
        full_count += full
    end

    fill!(delta.after_to_source, 0x00)
    output_row = full_count + 1
    @inbounds for source_row in 1:BOARD_ROWS
        if delta.full_rows[source_row]
            delta.source_to_after[source_row] = 0x00
        else
            destination = UInt8(output_row)
            delta.source_to_after[source_row] = destination
            delta.after_to_source[output_row] = UInt8(source_row)
            output_row += 1
        end
    end
    output_row == BOARD_ROWS + 1 || error("line-clear row-map drift")
    delta.line_clear[1] = UInt8(full_count)
    return full_count
end

"""Prepare a candidate delta from a direct `24 x 10` placement matrix."""
function prepare_candidate_delta!(
    delta::CandidateDelta,
    common::StateCommon,
    placement::AbstractMatrix,
    tspin::Real,
)
    size(placement) == (BOARD_ROWS, BOARD_COLUMNS) || throw(DimensionMismatch(
        "placement must have shape ($BOARD_ROWS, $BOARD_COLUMNS)",
    ))
    fill!(delta.placement_positions, UInt16(0))
    delta.placement_count[1] = UInt8(0)
    active_count = 0
    @inbounds for column in 1:BOARD_COLUMNS, board_row in 1:BOARD_ROWS
        bit = _bit(placement[board_row, column])
        delta.placement[board_row, column] = bit
        if !iszero(bit)
            active_count += 1
            active_count <= PLACEMENT_CAPACITY &&
                (delta.placement_positions[active_count] =
                    _position(board_row, column))
        end
    end
    _finish_placement_positions!(delta, active_count)
    delta.tspin[1] = Float32(tspin)
    _finish_row_map!(delta, common)
    return delta
end

"""Prepare one candidate directly from the canonical teacher dataset shape."""
function prepare_candidate_delta!(
    delta::CandidateDelta,
    common::StateCommon,
    dataset,
    row::Int,
    candidate::Int,
)
    1 <= row <= size(dataset.placements, 5) ||
        throw(BoundsError(dataset.placements, (:, :, 1, candidate, row)))
    1 <= candidate <= size(dataset.placements, 4) ||
        throw(BoundsError(dataset.placements, (:, :, 1, candidate, row)))
    fill!(delta.placement_positions, UInt16(0))
    delta.placement_count[1] = UInt8(0)
    active_count = 0
    @inbounds for column in 1:BOARD_COLUMNS, board_row in 1:BOARD_ROWS
        bit = _bit(dataset.placements[board_row, column, 1, candidate, row])
        delta.placement[board_row, column] = bit
        if !iszero(bit)
            active_count += 1
            active_count <= PLACEMENT_CAPACITY &&
                (delta.placement_positions[active_count] =
                    _position(board_row, column))
        end
    end
    _finish_placement_positions!(delta, active_count)
    delta.tspin[1] = Float32(dataset.tspin[candidate, row])
    _finish_row_map!(delta, common)
    return delta
end

"""
Reconstruct `after`, `added`, `removed`, and the 37 canonical auxiliary values.
"""
function reconstruct_candidate!(
    output::CandidateMaterialization,
    common::StateCommon,
    delta::CandidateDelta,
)
    fill!(output.after, 0x00)
    @inbounds for source_row in 1:BOARD_ROWS
        destination = Int(delta.source_to_after[source_row])
        iszero(destination) && continue
        for column in 1:BOARD_COLUMNS
            output.after[destination, column] =
                (!iszero(common.board[source_row, column]) ||
                 !iszero(delta.placement[source_row, column])) ? 0x01 : 0x00
        end
    end

    @inbounds for column in 1:BOARD_COLUMNS, board_row in 1:BOARD_ROWS
        before = common.board[board_row, column]
        after = output.after[board_row, column]
        output.added[board_row, column] = after > before ? 0x01 : 0x00
        output.removed[board_row, column] = after < before ? 0x01 : 0x00
    end

    fill!(output.heights, 0x00)
    fill!(output.holes, 0x00)
    @inbounds for column in 1:BOARD_COLUMNS
        first_filled = 0
        for board_row in 1:BOARD_ROWS
            if !iszero(output.after[board_row, column])
                first_filled = board_row
                break
            end
        end
        iszero(first_filled) && continue
        output.heights[column] = UInt8(BOARD_ROWS - first_filled + 1)
        holes = 0
        for board_row in first_filled:BOARD_ROWS
            holes += iszero(output.after[board_row, column])
        end
        output.holes[column] = UInt8(holes)
    end

    fill!(output.reachable, false)
    head = 1
    tail = 0
    @inbounds for column in 1:BOARD_COLUMNS
        if iszero(output.after[1, column])
            tail += 1
            linear = 1 + (column - 1) * BOARD_ROWS
            output.flood_queue[tail] = Int16(linear)
            output.reachable[1, column] = true
        end
    end
    @inbounds while head <= tail
        linear = Int(output.flood_queue[head])
        head += 1
        column = div(linear - 1, BOARD_ROWS) + 1
        board_row = linear - (column - 1) * BOARD_ROWS
        if board_row > 1 &&
           iszero(output.after[board_row - 1, column]) &&
           !output.reachable[board_row - 1, column]
            tail += 1
            output.flood_queue[tail] = Int16(linear - 1)
            output.reachable[board_row - 1, column] = true
        end
        if board_row < BOARD_ROWS &&
           iszero(output.after[board_row + 1, column]) &&
           !output.reachable[board_row + 1, column]
            tail += 1
            output.flood_queue[tail] = Int16(linear + 1)
            output.reachable[board_row + 1, column] = true
        end
        if column > 1 &&
           iszero(output.after[board_row, column - 1]) &&
           !output.reachable[board_row, column - 1]
            tail += 1
            output.flood_queue[tail] = Int16(linear - BOARD_ROWS)
            output.reachable[board_row, column - 1] = true
        end
        if column < BOARD_COLUMNS &&
           iszero(output.after[board_row, column + 1]) &&
           !output.reachable[board_row, column + 1]
            tail += 1
            output.flood_queue[tail] = Int16(linear + BOARD_ROWS)
            output.reachable[board_row, column + 1] = true
        end
    end

    cavities = 0
    aggregate_height = 0
    bumpiness = 0
    max_height = 0
    @inbounds for column in 1:BOARD_COLUMNS
        height = Int(output.heights[column])
        aggregate_height += height
        max_height = max(max_height, height)
        column > 1 &&
            (bumpiness += abs(height - Int(output.heights[column - 1])))
        left = column == 1 ? BOARD_ROWS : Int(output.heights[column - 1])
        right = column == BOARD_COLUMNS ?
            BOARD_ROWS : Int(output.heights[column + 1])
        output.wells[column] = UInt8(max(min(left, right) - height, 0))
        for board_row in 1:BOARD_ROWS
            cavities += iszero(output.after[board_row, column]) &&
                !output.reachable[board_row, column]
        end
    end

    output.geometry[1] = Int16(cavities)
    output.geometry[2] = Int16(aggregate_height)
    output.geometry[3] = Int16(bumpiness)
    output.geometry[4] = Int16(max_height)

    @inbounds for column in 1:BOARD_COLUMNS
        output.aux[column] = Float32(output.heights[column]) / 24.0f0
        output.aux[10 + column] = Float32(output.holes[column]) / 24.0f0
        output.aux[20 + column] = Float32(output.wells[column]) / 24.0f0
    end
    output.aux[31] = Float32(cavities) / 240.0f0
    output.aux[32] = Float32(aggregate_height) / 240.0f0
    output.aux[33] = Float32(bumpiness) / 216.0f0
    output.aux[34] = Float32(max_height) / 24.0f0
    output.aux[35] = common.ren[1] / 30.0f0
    output.aux[36] = common.back_to_back[1]
    output.aux[37] = delta.tspin[1]
    return output
end

"""Reconstruct and pack one candidate into the canonical 1,298 Float32 rails."""
function pack_candidate_rails!(
    rails::AbstractVector{Float32},
    common::StateCommon,
    delta::CandidateDelta,
    output::CandidateMaterialization,
)
    length(rails) == INPUT_RAILS || throw(DimensionMismatch(
        "rail vector must have length $INPUT_RAILS",
    ))
    reconstruct_candidate!(output, common, delta)
    rail = 0
    @inbounds for source in (common.board, output.after, output.added, output.removed)
        for column in 1:BOARD_COLUMNS, board_row in 1:BOARD_ROWS
            rail += 1
            rails[rail] = _rail(source[board_row, column])
        end
    end
    @inbounds for token in 1:QUEUE_TOKENS, piece in 1:QUEUE_PIECES
        rail += 1
        rails[rail] = _rail(common.queue[piece, token])
    end
    @inbounds for level in 1:AUX_LEVELS
        threshold = Float32(level) / Float32(AUX_LEVELS)
        for index in 1:AUX_FEATURES
            rail += 1
            rails[rail] = output.aux[index] >= threshold ? 1.0f0 : 0.0f0
        end
    end
    rail == INPUT_RAILS || error("candidate-delta rail packing drift")
    return rails
end

end # module CandidateDeltaInput
