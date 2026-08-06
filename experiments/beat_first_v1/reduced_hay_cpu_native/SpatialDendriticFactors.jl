module SpatialDendriticFactors

using ..ActiveApicalCell
using ..CandidateDeltaInput
using ..DendriticProgramBank
using ..SharedDendriticFactor

const Cell = ActiveApicalCell
const DeltaInput = CandidateDeltaInput
const Bank = DendriticProgramBank
const Factor = SharedDendriticFactor

export AFTER_PLANE,
       BEFORE_PLANE,
       POSITION_COUNT,
       AffectedPositions,
       SpatialFactorScratch,
       affected_count,
       affected_position,
       evaluate_affected_factors!,
       evaluate_all_factors!,
       prepare_affected_positions!,
       pullback_affected_factors!,
       pullback_all_factors!,
       spatial_drive!,
       spatial_program_rows

const POSITION_COUNT = DeltaInput.BOARD_CELLS
const BEFORE_PLANE = UInt8(1)
const AFTER_PLANE = UInt8(2)

# Every 3x3 site is one of empty, occupied, or outside the board.  The latter
# is an explicit symbol rather than implicit zero padding.
const _EMPTY = UInt8(0)
const _OCCUPIED = UInt8(1)
const _BOUNDARY = UInt8(2)

const _EMPTY_BASAL_DRIVE = -0.25f0
const _OCCUPIED_BASAL_DRIVE = 1.00f0
const _BOUNDARY_BASAL_DRIVE = -0.60f0
const _EMPTY_CENTER_DRIVE = -0.35f0
const _OCCUPIED_CENTER_DRIVE = 0.35f0

"""Fixed-capacity set of factors whose 3x3 receptive field changed."""
mutable struct AffectedPositions <: AbstractVector{UInt16}
    positions::Memory{UInt16}
    marked::Memory{UInt8}
    count::Int
    function AffectedPositions()
        positions = Memory{UInt16}(undef, POSITION_COUNT)
        marked = Memory{UInt8}(undef, POSITION_COUNT)
        fill!(positions, UInt16(0))
        fill!(marked, UInt8(0))
        return new(positions, marked, 0)
    end
end

@inline affected_count(affected::AffectedPositions) = affected.count
Base.IndexStyle(::Type{AffectedPositions}) = IndexLinear()
Base.size(affected::AffectedPositions) = (affected.count,)
Base.length(affected::AffectedPositions) = affected.count
@inline Base.getindex(affected::AffectedPositions, index::Int) =
    affected_position(affected, index)

@inline function affected_position(affected::AffectedPositions, index::Integer)
    1 <= index <= affected.count || throw(BoundsError(1:affected.count, index))
    return @inbounds affected.positions[Int(index)]
end

"""All caller-owned buffers required by one factor forward/reverse."""
struct SpatialFactorScratch{T<:AbstractFloat}
    payload::Vector{Float32}
    program::Vector{T}
    default_program::Vector{T}
    drive::Vector{T}
    features::Vector{T}
    dfeatures::Vector{T}
    ddrive::Vector{T}
    dprogram::Vector{T}
    dshared::Vector{T}
    trace::Factor.FactorTrace{T}
    reverse::Factor.FactorScratch{T}
end

function SpatialFactorScratch(::Type{T}=Float32) where {T<:AbstractFloat}
    return SpatialFactorScratch(
        zeros(Float32, Factor.PROGRAM_DIM),
        zeros(T, Factor.PROGRAM_DIM),
        Factor.default_raw_program(T),
        zeros(T, Factor.DRIVE_DIM),
        zeros(T, Factor.FEATURE_DIM),
        zeros(T, Factor.FEATURE_DIM),
        zeros(T, Factor.DRIVE_DIM),
        zeros(T, Factor.PROGRAM_DIM),
        zeros(T, Cell.PARAM_DIM),
        Factor.FactorTrace(T),
        Factor.FactorScratch(T),
    )
end

@inline function _check_board(board)
    size(board) == (DeltaInput.BOARD_ROWS, DeltaInput.BOARD_COLUMNS) ||
        throw(DimensionMismatch(
            "board must have shape ($(DeltaInput.BOARD_ROWS), " *
            "$(DeltaInput.BOARD_COLUMNS))",
        ))
    return nothing
end

@inline function _check_plane(plane::UInt8)
    plane in (BEFORE_PLANE, AFTER_PLANE) ||
        throw(ArgumentError("plane must be BEFORE_PLANE or AFTER_PLANE"))
    return nothing
end

@inline _position(board_row::Int, column::Int) =
    board_row + (column - 1) * DeltaInput.BOARD_ROWS

@inline function _coordinates(position::Int)
    column = div(position - 1, DeltaInput.BOARD_ROWS) + 1
    board_row = position - (column - 1) * DeltaInput.BOARD_ROWS
    return board_row, column
end

@inline function _site(board, board_row::Int, column::Int)
    if board_row < 1 || board_row > DeltaInput.BOARD_ROWS ||
       column < 1 || column > DeltaInput.BOARD_COLUMNS
        return _BOUNDARY
    end
    return iszero(@inbounds(board[board_row, column])) ? _EMPTY : _OCCUPIED
end

@inline function _compact_occupancy_mask(board, board_row::Int, column::Int)
    mask = UInt16(0)
    lane = 0
    @inbounds for column_offset in -1:1, row_offset in -1:1
        site_row = board_row + row_offset
        site_column = column + column_offset
        if site_row < 1 || site_row > DeltaInput.BOARD_ROWS ||
           site_column < 1 || site_column > DeltaInput.BOARD_COLUMNS
            continue
        end
        !iszero(board[site_row, site_column]) &&
            (mask |= UInt16(1) << lane)
        lane += 1
    end
    return Int32(mask)
end

@inline function _table1_base(board_row::Int, column::Int)
    top = board_row == 1
    bottom = board_row == DeltaInput.BOARD_ROWS
    left = column == 1
    right = column == DeltaInput.BOARD_COLUMNS
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

@inline function _table2_base(board_row::Int, column::Int)
    row_boundary = board_row == 1 || board_row == DeltaInput.BOARD_ROWS
    row_prefix = board_row == 1 ? 0 :
        board_row == DeltaInput.BOARD_ROWS ? 14_176 :
        96 + (board_row - 2) * 640
    position_prefix = row_boundary ?
        (column == 1 ? 0 : column == DeltaInput.BOARD_COLUMNS ? 80 : 16) :
        (column == 1 ? 0 : column == DeltaInput.BOARD_COLUMNS ? 576 : 64)
    return Int32(row_prefix + position_prefix)
end

@inline function _table3_base(board_row::Int, column::Int)
    column_boundary = column == 1 || column == DeltaInput.BOARD_COLUMNS
    column_prefix = column == 1 ? 0 :
        column == DeltaInput.BOARD_COLUMNS ? 5_216 :
        96 + (column - 2) * 640
    position_prefix = column_boundary ?
        (board_row == 1 ? 0 :
         board_row == DeltaInput.BOARD_ROWS ? 80 : 16) :
        (board_row == 1 ? 0 :
         board_row == DeltaInput.BOARD_ROWS ? 576 : 64)
    return Int32(column_prefix + position_prefix)
end

@inline function _table4_base(board_row::Int, column::Int)
    column_boundary = column == 1 || column == DeltaInput.BOARD_COLUMNS
    column_prefix = column == 1 ? 0 :
        column == DeltaInput.BOARD_COLUMNS ? 92_576 :
        1_440 + (column - 2) * 11_392
    position_prefix = column_boundary ?
        (board_row == 1 ? 0 :
         board_row == DeltaInput.BOARD_ROWS ? 1_424 :
         16 + (board_row - 2) * 64) :
        (board_row == 1 ? 0 :
         board_row == DeltaInput.BOARD_ROWS ? 11_328 :
         64 + (board_row - 2) * 512)
    return Int32(column_prefix + position_prefix)
end

const _TABLE1_BASE_BY_POSITION = ntuple(POSITION_COUNT) do position
    board_row, column = _coordinates(position)
    _table1_base(board_row, column)
end
const _TABLE2_BASE_BY_POSITION = ntuple(POSITION_COUNT) do position
    board_row, column = _coordinates(position)
    _table2_base(board_row, column)
end
const _TABLE3_BASE_BY_POSITION = ntuple(POSITION_COUNT) do position
    board_row, column = _coordinates(position)
    _table3_base(board_row, column)
end
const _TABLE4_BASE_BY_POSITION = ntuple(POSITION_COUNT) do position
    board_row, column = _coordinates(position)
    _table4_base(board_row, column)
end

"""
Return the four collision-free compact program rows of one spatial factor.

The address is a multiresolution composition rather than four independent
low-cardinality labels:

1. local ternary 3x3 morphology;
2. morphology bound to row;
3. morphology bound to column;
4. morphology bound to row, column, and semantic plane.

The base maps are fixed Int32 tables indexed by absolute board position.  The
only dynamic address component is the 4/6/9-bit occupancy mask of the valid
local sites.  All 208,448 physical bank rows are reachable, tables have
disjoint ranges, and every factor selects exactly four rows without hashing,
bucket collisions, slots, or deduplication.
"""
@inline function spatial_program_rows(
    board::AbstractMatrix,
    position::Integer,
    plane::UInt8,
)
    _check_board(board)
    1 <= position <= POSITION_COUNT ||
        throw(BoundsError(1:POSITION_COUNT, position))
    _check_plane(plane)
    physical_position = Int(position)
    board_row, column = _coordinates(physical_position)
    mask = _compact_occupancy_mask(board, board_row, column)
    table4_plane_offset = plane == AFTER_PLANE ? Int32(94_016) : Int32(0)
    return Bank.ProgramRows(
        Int32(Bank.TABLE_ROW_OFFSETS[1]) +
            _TABLE1_BASE_BY_POSITION[physical_position] + mask + Int32(1),
        Int32(Bank.TABLE_ROW_OFFSETS[2]) +
            _TABLE2_BASE_BY_POSITION[physical_position] + mask + Int32(1),
        Int32(Bank.TABLE_ROW_OFFSETS[3]) +
            _TABLE3_BASE_BY_POSITION[physical_position] + mask + Int32(1),
        Int32(Bank.TABLE_ROW_OFFSETS[4]) + table4_plane_offset +
            _TABLE4_BASE_BY_POSITION[physical_position] + mask + Int32(1),
    )
end

@inline function _basal_drive(site::UInt8, ::Type{T}) where {T}
    site == _EMPTY && return T(_EMPTY_BASAL_DRIVE)
    site == _OCCUPIED && return T(_OCCUPIED_BASAL_DRIVE)
    return T(_BOUNDARY_BASAL_DRIVE)
end

"""
Write the 3x3 local evidence into eight basal drives and one apical drive.

Both binary values are active typed symbols: an occupied basal site drives an
excitatory receptor, while an empty site drives an inhibitory receptor rather
than silence.  The outside-board boundary is a third inhibitory symbol with a
distinct magnitude.  An empty centre likewise recruits a nonzero inhibitory
apical drive.  Consequently neither rail can disappear before learning begins
and a mostly empty board cannot become a repeated positive DC current.
"""
function spatial_drive!(
    drive::AbstractVector{T},
    board::AbstractMatrix,
    position::Integer,
) where {T<:AbstractFloat}
    length(drive) == Factor.DRIVE_DIM || throw(DimensionMismatch(
        "drive must have $(Factor.DRIVE_DIM) coordinates",
    ))
    _check_board(board)
    1 <= position <= POSITION_COUNT ||
        throw(BoundsError(1:POSITION_COUNT, position))
    board_row, column = _coordinates(Int(position))
    basal = 0
    @inbounds for column_offset in -1:1, row_offset in -1:1
        if iszero(row_offset) && iszero(column_offset)
            continue
        end
        basal += 1
        drive[basal] = _basal_drive(
            _site(board, board_row + row_offset, column + column_offset),
            T,
        )
    end
    centre = _site(board, board_row, column)
    drive[Factor.APICAL_DRIVE_INDEX] = centre == _OCCUPIED ?
        T(_OCCUPIED_CENTER_DRIVE) : T(_EMPTY_CENTER_DRIVE)
    return drive
end

@inline function _prepare_program!(
    scratch::SpatialFactorScratch{T},
    bank::Bank.ProgramBank,
    rows::Bank.ProgramRows,
) where {T}
    Bank.accumulate_active_payload!(scratch.payload, bank, rows)
    scale = T(0.5)
    @inbounds for lane in 1:Factor.PROGRAM_DIM
        scratch.program[lane] = scratch.default_program[lane] +
                                T(scratch.payload[lane]) * scale
    end
    return scale
end

@inline function _evaluate_position!(
    scratch::SpatialFactorScratch{T},
    bank::Bank.ProgramBank,
    board::AbstractMatrix,
    position::Int,
    plane::UInt8,
    cache,
) where {T}
    rows = spatial_program_rows(board, position, plane)
    _prepare_program!(scratch, bank, rows)
    spatial_drive!(scratch.drive, board, position)
    control = Factor.factor_forward!(
        scratch.features,
        scratch.trace,
        scratch.drive,
        scratch.program,
        cache,
    )
    return control, rows
end

@inline function _check_feature_output(features, controls)
    size(features) == (Factor.FEATURE_DIM, POSITION_COUNT) ||
        throw(DimensionMismatch(
            "features must have shape ($(Factor.FEATURE_DIM), $POSITION_COUNT)",
        ))
    length(controls) == POSITION_COUNT || throw(DimensionMismatch(
        "controls must have $POSITION_COUNT entries",
    ))
    return nothing
end

"""Evaluate all 240 factors of one board plane."""
function evaluate_all_factors!(
    features::AbstractMatrix{T},
    controls::AbstractVector{T},
    scratch::SpatialFactorScratch{T},
    bank::Bank.ProgramBank,
    board::AbstractMatrix,
    plane::UInt8,
    cache,
) where {T<:AbstractFloat}
    _check_feature_output(features, controls)
    _check_board(board)
    _check_plane(plane)
    @inbounds for position in 1:POSITION_COUNT
        control, _ = _evaluate_position!(
            scratch,
            bank,
            board,
            position,
            plane,
            cache,
        )
        controls[position] = control
        for feature in 1:Factor.FEATURE_DIM
            features[feature, position] = scratch.features[feature]
        end
    end
    return features, controls
end

"""
Evaluate only candidate factors listed in `affected`.

Columns outside the fixed affected list are deliberately left untouched.  A
typed afferent delta reads only these columns, avoiding a 240-factor copy for
each candidate.
"""
function evaluate_affected_factors!(
    features::AbstractMatrix{T},
    controls::AbstractVector{T},
    scratch::SpatialFactorScratch{T},
    bank::Bank.ProgramBank,
    board::AbstractMatrix,
    plane::UInt8,
    cache,
    affected::AffectedPositions,
) where {T<:AbstractFloat}
    _check_feature_output(features, controls)
    _check_board(board)
    _check_plane(plane)
    @inbounds for affected_index in 1:affected.count
        position = Int(affected.positions[affected_index])
        control, _ = _evaluate_position!(
            scratch,
            bank,
            board,
            position,
            plane,
            cache,
        )
        controls[position] = control
        for feature in 1:Factor.FEATURE_DIM
            features[feature, position] = scratch.features[feature]
        end
    end
    return features, controls
end

"""
Find the exact 3x3 dependency closure of cells changed by a candidate.

The returned positions are unique and ordered by canonical column-major board
index.  The scan is capacity-bounded at 240 and performs no allocation.
"""
function prepare_affected_positions!(
    affected::AffectedPositions,
    before::AbstractMatrix,
    after::AbstractMatrix,
)
    _check_board(before)
    _check_board(after)
    fill!(affected.marked, UInt8(0))
    @inbounds for column in 1:DeltaInput.BOARD_COLUMNS
        for board_row in 1:DeltaInput.BOARD_ROWS
            before[board_row, column] == after[board_row, column] && continue
            for factor_column in max(1, column - 1):min(
                DeltaInput.BOARD_COLUMNS,
                column + 1,
            )
                for factor_row in max(1, board_row - 1):min(
                    DeltaInput.BOARD_ROWS,
                    board_row + 1,
                )
                    affected.marked[_position(factor_row, factor_column)] =
                        UInt8(1)
                end
            end
        end
    end
    count = 0
    @inbounds for position in 1:POSITION_COUNT
        iszero(affected.marked[position]) && continue
        count += 1
        affected.positions[count] = UInt16(position)
    end
    affected.count = count
    return affected
end

function prepare_affected_positions!(
    affected::AffectedPositions,
    common::DeltaInput.StateCommon,
    candidate::DeltaInput.CandidateMaterialization,
)
    return prepare_affected_positions!(affected, common.board, candidate.after)
end

@inline function _accumulate_position_pullback!(
    shared_bar::AbstractVector{T},
    bank_bar,
    scratch::SpatialFactorScratch{T},
    bank::Bank.ProgramBank,
    board::AbstractMatrix,
    position::Int,
    plane::UInt8,
    cache,
    derivative_cache,
    feature_bar::AbstractMatrix{T},
    control_bar::T,
) where {T<:AbstractFloat}
    _, rows = _evaluate_position!(scratch, bank, board, position, plane, cache)
    @inbounds for feature in 1:Factor.FEATURE_DIM
        scratch.dfeatures[feature] = feature_bar[feature, position]
    end
    Factor.factor_pullback!(
        scratch.ddrive,
        scratch.dprogram,
        scratch.dshared,
        scratch.reverse,
        scratch.trace,
        scratch.drive,
        scratch.program,
        cache,
        derivative_cache,
        scratch.dfeatures,
        control_bar,
    )
    @inbounds for parameter in 1:Cell.PARAM_DIM
        shared_bar[parameter] += scratch.dshared[parameter]
    end
    scale = T(0.5)
    @inbounds for active_index in 1:Bank.active_count(rows)
        row = Int(Bank.active_row(rows, active_index))
        Bank.accumulate_program_gradient!(
            bank_bar,
            row,
            scratch.dprogram,
            scale,
        )
    end
    return shared_bar, bank_bar
end

@inline function _check_pullback_buffers(shared_bar, bank_bar, bank, feature_bar)
    length(shared_bar) == Cell.PARAM_DIM || throw(DimensionMismatch(
        "shared cotangent must have $(Cell.PARAM_DIM) entries",
    ))
    Bank.program_gradient_row_count(bank_bar) == Bank.bank_row_count(bank) ||
        throw(DimensionMismatch("bank cotangent must match the bank row count"))
    bank_bar isa AbstractMatrix && size(bank_bar, 1) != Factor.PROGRAM_DIM &&
        throw(DimensionMismatch("dense bank cotangent has the wrong payload width"))
    size(feature_bar) == (Factor.FEATURE_DIM, POSITION_COUNT) ||
        throw(DimensionMismatch(
            "feature cotangent must have shape " *
            "($(Factor.FEATURE_DIM), $POSITION_COUNT)",
        ))
    return nothing
end

"""Exact replay pullback for every factor of a board plane."""
function pullback_all_factors!(
    shared_bar::AbstractVector{T},
    bank_bar,
    scratch::SpatialFactorScratch{T},
    bank::Bank.ProgramBank,
    board::AbstractMatrix,
    plane::UInt8,
    cache,
    derivative_cache,
    feature_bar::AbstractMatrix{T},
    control_bar::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:AbstractFloat}
    _check_pullback_buffers(shared_bar, bank_bar, bank, feature_bar)
    control_bar === nothing || length(control_bar) == POSITION_COUNT ||
        throw(DimensionMismatch("control cotangent must have $POSITION_COUNT entries"))
    _check_board(board)
    _check_plane(plane)
    @inbounds for position in 1:POSITION_COUNT
        control = control_bar === nothing ? zero(T) : control_bar[position]
        _accumulate_position_pullback!(
            shared_bar,
            bank_bar,
            scratch,
            bank,
            board,
            position,
            plane,
            cache,
            derivative_cache,
            feature_bar,
            control,
        )
    end
    return shared_bar, bank_bar
end

"""Exact replay pullback for only the candidate-affected factors."""
function pullback_affected_factors!(
    shared_bar::AbstractVector{T},
    bank_bar,
    scratch::SpatialFactorScratch{T},
    bank::Bank.ProgramBank,
    board::AbstractMatrix,
    plane::UInt8,
    cache,
    derivative_cache,
    feature_bar::AbstractMatrix{T},
    affected::AffectedPositions,
    control_bar::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:AbstractFloat}
    _check_pullback_buffers(shared_bar, bank_bar, bank, feature_bar)
    control_bar === nothing || length(control_bar) == POSITION_COUNT ||
        throw(DimensionMismatch("control cotangent must have $POSITION_COUNT entries"))
    _check_board(board)
    _check_plane(plane)
    @inbounds for affected_index in 1:affected.count
        position = Int(affected.positions[affected_index])
        control = control_bar === nothing ? zero(T) : control_bar[position]
        _accumulate_position_pullback!(
            shared_bar,
            bank_bar,
            scratch,
            bank,
            board,
            position,
            plane,
            cache,
            derivative_cache,
            feature_bar,
            control,
        )
    end
    return shared_bar, bank_bar
end

end # module SpatialDendriticFactors
