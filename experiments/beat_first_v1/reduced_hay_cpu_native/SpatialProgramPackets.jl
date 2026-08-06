module SpatialProgramPackets

using ..DendriticProgramBank

const Bank = DendriticProgramBank

export AFTER_PLANE,
       BEFORE_PLANE,
       BOARD_COLUMNS,
       BOARD_ROWS,
       PACKET_COUNT,
       PACKET_WIDTH,
       PLANE_COUNT,
       POSITION_COUNT,
       SpatialPacketWorkspace,
       base_packet_grid!,
       base_packet_grid_pullback!,
       candidate_after_packets!,
       candidate_after_packets_pullback!,
       packet_column,
       spatial_program_rows

const BOARD_ROWS = 24
const BOARD_COLUMNS = 10
const POSITION_COUNT = BOARD_ROWS * BOARD_COLUMNS
const PLANE_COUNT = 2
const PACKET_COUNT = POSITION_COUNT * PLANE_COUNT
const PACKET_WIDTH = Bank.PAYLOAD_WIDTH

const BEFORE_PLANE = UInt8(1)
const AFTER_PLANE = UInt8(2)

# The compact address tables enumerate the exact binary 3x3 domain at each
# physical boundary class.  Table four has one disjoint half per semantic
# before/after plane.
const _AFTER_TABLE4_OFFSET = Int32(94_016)

"""Fixed scratch for allocation-free spatial packet forward and reverse."""
struct SpatialPacketWorkspace
    packet::Vector{Float32}
    packet_bar::Vector{Float32}
end

SpatialPacketWorkspace() = SpatialPacketWorkspace(
    zeros(Float32, PACKET_WIDTH),
    zeros(Float32, PACKET_WIDTH),
)

@inline function _check_board(board)
    size(board) == (BOARD_ROWS, BOARD_COLUMNS) || throw(DimensionMismatch(
        "board must have shape ($BOARD_ROWS, $BOARD_COLUMNS)",
    ))
    return nothing
end

@inline function _check_position(position::Integer)
    1 <= position <= POSITION_COUNT ||
        throw(BoundsError(1:POSITION_COUNT, position))
    return Int(position)
end

@inline function _check_plane(plane::Integer)
    plane == BEFORE_PLANE || plane == AFTER_PLANE || throw(ArgumentError(
        "plane must be BEFORE_PLANE or AFTER_PLANE",
    ))
    return UInt8(plane)
end

@inline function _coordinates(position::Int)
    column = div(position - 1, BOARD_ROWS) + 1
    row = position - (column - 1) * BOARD_ROWS
    return row, column
end

"""Column occupied by one `(position, plane)` in the canonical 16 x 480 grid."""
@inline function packet_column(position::Integer, plane::Integer)
    physical_position = _check_position(position)
    physical_plane = _check_plane(plane)
    return physical_position +
           (Int(physical_plane) - Int(BEFORE_PLANE)) * POSITION_COUNT
end

@inline function _occupancy_mask(board, row::Int, column::Int)
    # Out-of-board sites are encoded by the boundary-specific table prefix, so
    # only in-board binary sites consume mask bits.  This gives exactly 4, 6,
    # or 9 bits at corners, edges, or interior positions respectively.
    mask = UInt16(0)
    lane = 0
    @inbounds for column_offset in -1:1, row_offset in -1:1
        local_row = row + row_offset
        local_column = column + column_offset
        if local_row < 1 || local_row > BOARD_ROWS ||
           local_column < 1 || local_column > BOARD_COLUMNS
            continue
        end
        !iszero(board[local_row, local_column]) &&
            (mask |= UInt16(1) << lane)
        lane += 1
    end
    return Int32(mask)
end

@inline function _table1_base(row::Int, column::Int)
    top = row == 1
    bottom = row == BOARD_ROWS
    left = column == 1
    right = column == BOARD_COLUMNS
    top && left && return Int32(768)
    top && right && return Int32(784)
    bottom && left && return Int32(800)
    bottom && right && return Int32(816)
    top && return Int32(512)
    bottom && return Int32(576)
    left && return Int32(640)
    right && return Int32(704)
    return Int32(0)
end

@inline function _table2_base(row::Int, column::Int)
    boundary = row == 1 || row == BOARD_ROWS
    row_prefix = row == 1 ? 0 : row == BOARD_ROWS ? 14_176 :
                 96 + (row - 2) * 640
    position_prefix = boundary ?
        (column == 1 ? 0 : column == BOARD_COLUMNS ? 80 : 16) :
        (column == 1 ? 0 : column == BOARD_COLUMNS ? 576 : 64)
    return Int32(row_prefix + position_prefix)
end

@inline function _table3_base(row::Int, column::Int)
    boundary = column == 1 || column == BOARD_COLUMNS
    column_prefix = column == 1 ? 0 :
        column == BOARD_COLUMNS ? 5_216 : 96 + (column - 2) * 640
    position_prefix = boundary ?
        (row == 1 ? 0 : row == BOARD_ROWS ? 80 : 16) :
        (row == 1 ? 0 : row == BOARD_ROWS ? 576 : 64)
    return Int32(column_prefix + position_prefix)
end

@inline function _table4_base(row::Int, column::Int)
    boundary = column == 1 || column == BOARD_COLUMNS
    column_prefix = column == 1 ? 0 :
        column == BOARD_COLUMNS ? 92_576 :
        1_440 + (column - 2) * 11_392
    position_prefix = boundary ?
        (row == 1 ? 0 : row == BOARD_ROWS ? 1_424 :
         16 + (row - 2) * 64) :
        (row == 1 ? 0 : row == BOARD_ROWS ? 11_328 :
         64 + (row - 2) * 512)
    return Int32(column_prefix + position_prefix)
end

const _TABLE1_BASE = ntuple(POSITION_COUNT) do position
    row, column = _coordinates(position)
    _table1_base(row, column)
end
const _TABLE2_BASE = ntuple(POSITION_COUNT) do position
    row, column = _coordinates(position)
    _table2_base(row, column)
end
const _TABLE3_BASE = ntuple(POSITION_COUNT) do position
    row, column = _coordinates(position)
    _table3_base(row, column)
end
const _TABLE4_BASE = ntuple(POSITION_COUNT) do position
    row, column = _coordinates(position)
    _table4_base(row, column)
end

"""
Return the exact four-row spatial program address for one board position.

The first three semantic rows encode morphology, morphology+row, and
morphology+column.  The fourth encodes morphology+row+column+plane.  Hence
before and after observations can never alias even when their occupancies are
identical.
"""
@inline function _spatial_program_rows(
    board,
    physical_position::Int,
    physical_plane::UInt8,
)
    row, column = _coordinates(physical_position)
    mask = _occupancy_mask(board, row, column)
    plane_offset = physical_plane == AFTER_PLANE ?
        _AFTER_TABLE4_OFFSET : Int32(0)
    return Bank.ProgramRows(
        Int32(Bank.TABLE_ROW_OFFSETS[1]) +
            _TABLE1_BASE[physical_position] + mask + 1,
        Int32(Bank.TABLE_ROW_OFFSETS[2]) +
            _TABLE2_BASE[physical_position] + mask + 1,
        Int32(Bank.TABLE_ROW_OFFSETS[3]) +
            _TABLE3_BASE[physical_position] + mask + 1,
        Int32(Bank.TABLE_ROW_OFFSETS[4]) + plane_offset +
            _TABLE4_BASE[physical_position] + mask + 1,
    )
end

@inline function spatial_program_rows(
    board,
    position::Integer,
    plane::Integer,
)
    _check_board(board)
    physical_position = _check_position(position)
    physical_plane = _check_plane(plane)
    return _spatial_program_rows(board, physical_position, physical_plane)
end

@inline function _check_base_grid(grid)
    size(grid) == (PACKET_WIDTH, PACKET_COUNT) || throw(DimensionMismatch(
        "base packet grid must have shape ($PACKET_WIDTH, $PACKET_COUNT)",
    ))
    return nothing
end

@inline function _check_candidate_matrices(candidate_packets, packet_delta, count)
    size(candidate_packets) == (PACKET_WIDTH, count) ||
        throw(DimensionMismatch(
            "candidate packets must have shape ($PACKET_WIDTH, $count)",
        ))
    size(packet_delta) == (PACKET_WIDTH, count) || throw(DimensionMismatch(
        "packet delta must have shape ($PACKET_WIDTH, $count)",
    ))
    return nothing
end

"""
Materialize the canonical `16 x 480` base packet grid without allocation.

Both semantic planes observe the same common board but own distinct table-four
rows.  Columns `1:240` are before packets and `241:480` are after packets.
"""
function base_packet_grid!(
    destination::AbstractMatrix{Float32},
    workspace::SpatialPacketWorkspace,
    bank::Bank.ProgramBank,
    board,
)
    _check_base_grid(destination)
    _check_board(board)
    @inbounds for plane in (BEFORE_PLANE, AFTER_PLANE)
        plane_offset = (Int(plane) - Int(BEFORE_PLANE)) * POSITION_COUNT
        for position in 1:POSITION_COUNT
            rows = _spatial_program_rows(board, position, plane)
            Bank.program_packet!(workspace.packet, bank, rows)
            column = position + plane_offset
            for lane in 1:PACKET_WIDTH
                destination[lane, column] = workspace.packet[lane]
            end
        end
    end
    return destination
end

"""
Materialize only the requested candidate-after packets and their exact deltas.

Output column `index` corresponds to `after_positions[index]`.  The delta is
`candidate_after_packet - base_after_packet`; no before packet is recomputed.
"""
function candidate_after_packets!(
    candidate_packets::AbstractMatrix{Float32},
    packet_delta::AbstractMatrix{Float32},
    workspace::SpatialPacketWorkspace,
    bank::Bank.ProgramBank,
    base_grid::AbstractMatrix{Float32},
    after_board,
    after_positions::AbstractVector{<:Integer},
)
    count = length(after_positions)
    _check_candidate_matrices(candidate_packets, packet_delta, count)
    _check_base_grid(base_grid)
    _check_board(after_board)
    @inbounds for index in 1:count
        position = _check_position(after_positions[index])
        rows = _spatial_program_rows(after_board, position, AFTER_PLANE)
        Bank.program_packet!(workspace.packet, bank, rows)
        base_column = position + POSITION_COUNT
        for lane in 1:PACKET_WIDTH
            packet = workspace.packet[lane]
            candidate_packets[lane, index] = packet
            packet_delta[lane, index] = packet - base_grid[lane, base_column]
        end
    end
    return candidate_packets, packet_delta
end

"""Pull an arbitrary `16 x 480` base-grid cotangent into sparse bank rows."""
function base_packet_grid_pullback!(
    program_gradient::Bank.SparseProgramGradient,
    workspace::SpatialPacketWorkspace,
    bank::Bank.ProgramBank,
    board,
    base_grid_bar::AbstractMatrix,
)
    _check_base_grid(base_grid_bar)
    _check_board(board)
    @inbounds for plane in (BEFORE_PLANE, AFTER_PLANE)
        plane_offset = (Int(plane) - Int(BEFORE_PLANE)) * POSITION_COUNT
        for position in 1:POSITION_COUNT
            column = position + plane_offset
            for lane in 1:PACKET_WIDTH
                workspace.packet_bar[lane] =
                    Float32(base_grid_bar[lane, column])
            end
            rows = _spatial_program_rows(board, position, plane)
            Bank.program_packet_pullback!(
                program_gradient,
                bank,
                rows,
                workspace.packet_bar,
            )
        end
    end
    return program_gradient
end

"""
Reverse requested candidate packets and packet deltas exactly.

The candidate packet receives `candidate_packet_bar + packet_delta_bar`.
The negative side of the delta is accumulated into the common
`base_grid_bar`, so all candidates can share one later
`base_packet_grid_pullback!`.
"""
function candidate_after_packets_pullback!(
    program_gradient::Bank.SparseProgramGradient,
    base_grid_bar::AbstractMatrix,
    workspace::SpatialPacketWorkspace,
    bank::Bank.ProgramBank,
    after_board,
    after_positions::AbstractVector{<:Integer},
    candidate_packet_bar::AbstractMatrix,
    packet_delta_bar::AbstractMatrix,
)
    count = length(after_positions)
    _check_candidate_matrices(candidate_packet_bar, packet_delta_bar, count)
    _check_base_grid(base_grid_bar)
    _check_board(after_board)
    @inbounds for index in 1:count
        position = _check_position(after_positions[index])
        base_column = position + POSITION_COUNT
        for lane in 1:PACKET_WIDTH
            delta_bar = Float32(packet_delta_bar[lane, index])
            workspace.packet_bar[lane] =
                Float32(candidate_packet_bar[lane, index]) + delta_bar
            base_grid_bar[lane, base_column] -= delta_bar
        end
        rows = _spatial_program_rows(after_board, position, AFTER_PLANE)
        Bank.program_packet_pullback!(
            program_gradient,
            bank,
            rows,
            workspace.packet_bar,
        )
    end
    return program_gradient, base_grid_bar
end

end # module SpatialProgramPackets
