module CanonicalTetrisInput

"""
Teacher-sufficient, target-free Tetris input for the canonical dendritic graph.

The public types can represent only observations available to every compared
model: the board before placement, the raw placement, role-bound queue/meta,
and candidate T-spin status.  Teacher scores, ranks, selected actions, death
targets, and derived geometry targets are deliberately absent from the type
graph.
"""

export BOARD_ROWS,
       BOARD_COLUMNS,
       BOARD_CELLS,
       PLACEMENT_CAPACITY,
       NEXT_COUNT,
       BoardCell,
       EMPTY,
       OCCUPIED,
       PlacementCell,
       ABSENT,
       PRESENT,
       PieceKind,
       NONE,
       PIECE_I,
       PIECE_O,
       PIECE_T,
       PIECE_S,
       PIECE_Z,
       PIECE_J,
       PIECE_L,
       TruthValue,
       FALSE_VALUE,
       TRUE_VALUE,
       EventToken,
       NO_EVENT,
       EVENT_PRESENT,
       SiteToken,
       SITE_EMPTY,
       SITE_OCCUPIED,
       SITE_PLACED,
       OUTSIDE,
       CandidatePath,
       UNINITIALIZED,
       NO_CLEAR_COW,
       CLEAR_SLOW_PATH,
       StateMeta,
       CandidateMeta,
       StateObservation,
       CandidateObservation,
       TeacherSufficientInput,
       CandidateGeometry,
       derive_candidate!,
       candidate_path,
       clear_count,
       requires_clear_slow_path,
       full_row,
       source_to_after,
       after_to_source,
       hold_piece,
       next_piece,
       ren_value,
       back_to_back_value,
       tspin_value,
       placement_count,
       placement_position,
       no_clear_dirty_count,
       no_clear_dirty_position,
       no_clear_event,
       before_cell,
       placement_cell,
       preclear_cell,
       after_cell,
       before_site,
       preclear_site,
       after_site

const BOARD_ROWS = 24
const BOARD_COLUMNS = 10
const BOARD_CELLS = BOARD_ROWS * BOARD_COLUMNS
const PLACEMENT_CAPACITY = 4
const NEXT_COUNT = 5

# All semantic states have nonzero encodings.  Numeric zero is reserved for
# internal storage sentinels and is never an observation category.
@enum BoardCell::UInt8 begin
    EMPTY = 0x01
    OCCUPIED = 0x02
end

@enum PlacementCell::UInt8 begin
    ABSENT = 0x01
    PRESENT = 0x02
end

@enum PieceKind::UInt8 begin
    NONE = 0x01
    PIECE_I = 0x02
    PIECE_O = 0x03
    PIECE_T = 0x04
    PIECE_S = 0x05
    PIECE_Z = 0x06
    PIECE_J = 0x07
    PIECE_L = 0x08
end

@enum TruthValue::UInt8 begin
    FALSE_VALUE = 0x01
    TRUE_VALUE = 0x02
end

@enum EventToken::UInt8 begin
    NO_EVENT = 0x01
    EVENT_PRESENT = 0x02
end

@enum SiteToken::UInt8 begin
    SITE_EMPTY = 0x01
    SITE_OCCUPIED = 0x02
    SITE_PLACED = 0x03
    OUTSIDE = 0x04
end

@enum CandidatePath::UInt8 begin
    UNINITIALIZED = 0x01
    NO_CLEAR_COW = 0x02
    CLEAR_SLOW_PATH = 0x03
end

"""State-common metadata with queue roles fixed as HOLD and NEXT1--NEXT5."""
struct StateMeta
    hold::PieceKind
    next::NTuple{NEXT_COUNT,PieceKind}
    ren::Int32
    back_to_back::TruthValue

    function StateMeta(
        hold::PieceKind,
        next::NTuple{NEXT_COUNT,PieceKind},
        ren::Integer,
        back_to_back::TruthValue,
    )
        all(piece -> piece != NONE, next) || throw(ArgumentError(
            "NEXT1--NEXT5 must contain concrete pieces; NONE is HOLD-only",
        ))
        0 <= ren <= typemax(Int32) || throw(ArgumentError(
            "REN must be an exact nonnegative Int32 value",
        ))
        return new(hold, next, Int32(ren), back_to_back)
    end
end

"""Candidate-local metadata.  T-spin false remains an explicit value."""
struct CandidateMeta
    tspin::TruthValue
end

"""The immutable-by-construction before-board observation."""
struct StateObservation
    before::Matrix{BoardCell}
    meta::StateMeta

    function StateObservation(
        before::AbstractMatrix{BoardCell},
        meta::StateMeta,
    )
        size(before) == (BOARD_ROWS, BOARD_COLUMNS) || throw(
            DimensionMismatch("before board must have shape 24 x 10"),
        )
        canonical = Matrix{BoardCell}(before)
        @inbounds for row in 1:BOARD_ROWS
            full = true
            for column in 1:BOARD_COLUMNS
                full &= canonical[row, column] == OCCUPIED
            end
            full && throw(ArgumentError(
                "before board contains an uncleared full row at row $row",
            ))
        end
        return new(canonical, meta)
    end
end

"""
Raw placement in the pre-clear coordinate frame.

`ABSENT` is represented at every unused site.  The fixed four-position tuple
is a convenience index and never replaces the full typed placement plane.
"""
struct CandidateObservation
    raw_placement::Matrix{PlacementCell}
    positions::NTuple{PLACEMENT_CAPACITY,UInt16}
    count::UInt8
    meta::CandidateMeta

    function CandidateObservation(
        raw_placement::AbstractMatrix{PlacementCell},
        meta::CandidateMeta,
    )
        size(raw_placement) == (BOARD_ROWS, BOARD_COLUMNS) || throw(
            DimensionMismatch("raw placement must have shape 24 x 10"),
        )
        canonical = Matrix{PlacementCell}(raw_placement)
        position_buffer = zeros(UInt16, PLACEMENT_CAPACITY)
        count = 0
        @inbounds for column in 1:BOARD_COLUMNS, row in 1:BOARD_ROWS
            canonical[row, column] == PRESENT || continue
            count += 1
            count <= PLACEMENT_CAPACITY || throw(ArgumentError(
                "raw placement contains more than four occupied sites",
            ))
            position_buffer[count] = _position(row, column)
        end
        positions = ntuple(index -> position_buffer[index], PLACEMENT_CAPACITY)
        return new(canonical, positions, UInt8(count), meta)
    end
end

"""
The complete observation accepted by the canonical graph.

This object owns no dataset reference and has no route to a teacher target.
"""
struct TeacherSufficientInput
    state::StateObservation
    candidate::CandidateObservation

    function TeacherSufficientInput(
        state::StateObservation,
        candidate::CandidateObservation,
    )
        @inbounds for column in 1:BOARD_COLUMNS, row in 1:BOARD_ROWS
            if state.before[row, column] == OCCUPIED &&
               candidate.raw_placement[row, column] == PRESENT
                throw(ArgumentError(
                    "raw placement overlaps the before board at ($row, $column)",
                ))
            end
        end
        return new(state, candidate)
    end
end

"""Caller-owned exact pre-clear/after geometry and sparse-path metadata."""
mutable struct CandidateGeometry
    preclear::Matrix{BoardCell}
    after::Matrix{BoardCell}
    full_rows::Memory{Bool}
    mu::Memory{UInt8}
    pi::Memory{UInt8}
    dirty_positions::Memory{UInt16}
    dirty_count::UInt8
    cleared_rows::UInt8
    path::CandidatePath
end

function CandidateGeometry()
    preclear = Matrix{BoardCell}(undef, BOARD_ROWS, BOARD_COLUMNS)
    after = Matrix{BoardCell}(undef, BOARD_ROWS, BOARD_COLUMNS)
    full_rows = Memory{Bool}(undef, BOARD_ROWS)
    mu = Memory{UInt8}(undef, BOARD_ROWS)
    pi = Memory{UInt8}(undef, BOARD_ROWS)
    dirty_positions = Memory{UInt16}(undef, PLACEMENT_CAPACITY)
    fill!(preclear, EMPTY)
    fill!(after, EMPTY)
    fill!(full_rows, false)
    fill!(mu, 0x00)
    fill!(pi, 0x00)
    fill!(dirty_positions, UInt16(0))
    return CandidateGeometry(
        preclear,
        after,
        full_rows,
        mu,
        pi,
        dirty_positions,
        UInt8(0),
        UInt8(0),
        UNINITIALIZED,
    )
end

@inline _position(row::Int, column::Int) =
    UInt16(row + (column - 1) * BOARD_ROWS)

@inline _in_bounds(row::Integer, column::Integer) =
    1 <= row <= BOARD_ROWS && 1 <= column <= BOARD_COLUMNS

@inline function _require_derived(geometry::CandidateGeometry)
    geometry.path != UNINITIALIZED || throw(ArgumentError(
        "candidate geometry has not been derived",
    ))
    return nothing
end

@inline function _require_no_clear(geometry::CandidateGeometry)
    _require_derived(geometry)
    geometry.path == NO_CLEAR_COW || throw(ArgumentError(
        "no-clear dirty metadata is unavailable on the clear slow path",
    ))
    return nothing
end

"""
Derive `U = B or P`, exact full rows, source-to-after `mu`, inverse `pi`,
and the destination-coordinate after plane.

For a clear, `mu[source] == 0`.  Otherwise:

    mu(source) = clear_count + count(non-cleared rows <= source)

Leading after rows map to zero in `pi`.  No-clear candidates expose only the
raw-placement dirty set; any clear explicitly selects the full slow path.
"""
function derive_candidate!(
    geometry::CandidateGeometry,
    input::TeacherSufficientInput,
)
    geometry.path = UNINITIALIZED
    fill!(geometry.full_rows, false)
    fill!(geometry.mu, 0x00)
    fill!(geometry.pi, 0x00)
    fill!(geometry.dirty_positions, UInt16(0))
    geometry.dirty_count = UInt8(0)
    geometry.cleared_rows = UInt8(0)

    before = input.state.before
    placement = input.candidate.raw_placement
    @inbounds for column in 1:BOARD_COLUMNS, row in 1:BOARD_ROWS
        if before[row, column] == OCCUPIED
            placement[row, column] == ABSENT || throw(ArgumentError(
                "raw placement overlaps the before board at ($row, $column)",
            ))
            geometry.preclear[row, column] = OCCUPIED
        else
            geometry.preclear[row, column] =
                placement[row, column] == PRESENT ? OCCUPIED : EMPTY
        end
    end

    cleared = 0
    @inbounds for row in 1:BOARD_ROWS
        full = true
        for column in 1:BOARD_COLUMNS
            full &= geometry.preclear[row, column] == OCCUPIED
        end
        geometry.full_rows[row] = full
        cleared += full
    end
    cleared <= PLACEMENT_CAPACITY || error(
        "more than four rows cleared from a legal four-cell placement",
    )
    geometry.cleared_rows = UInt8(cleared)

    fill!(geometry.after, EMPTY)
    destination = cleared + 1
    @inbounds for source in 1:BOARD_ROWS
        if geometry.full_rows[source]
            geometry.mu[source] = 0x00
            continue
        end
        destination <= BOARD_ROWS || error("line-clear row-map overflow")
        geometry.mu[source] = UInt8(destination)
        geometry.pi[destination] = UInt8(source)
        for column in 1:BOARD_COLUMNS
            geometry.after[destination, column] =
                geometry.preclear[source, column]
        end
        destination += 1
    end
    destination == BOARD_ROWS + 1 || error("line-clear row-map drift")

    if iszero(cleared)
        geometry.path = NO_CLEAR_COW
        count = Int(input.candidate.count)
        @inbounds for index in 1:count
            geometry.dirty_positions[index] = input.candidate.positions[index]
        end
        geometry.dirty_count = UInt8(count)
    else
        geometry.path = CLEAR_SLOW_PATH
    end
    return geometry
end

@inline function candidate_path(geometry::CandidateGeometry)
    _require_derived(geometry)
    return geometry.path
end

@inline function clear_count(geometry::CandidateGeometry)
    _require_derived(geometry)
    return Int(geometry.cleared_rows)
end

@inline function requires_clear_slow_path(geometry::CandidateGeometry)
    _require_derived(geometry)
    return geometry.path == CLEAR_SLOW_PATH
end

@inline function full_row(geometry::CandidateGeometry, row::Integer)
    _require_derived(geometry)
    1 <= row <= BOARD_ROWS || throw(BoundsError(1:BOARD_ROWS, row))
    return @inbounds geometry.full_rows[row]
end

@inline function source_to_after(geometry::CandidateGeometry, row::Integer)
    _require_derived(geometry)
    1 <= row <= BOARD_ROWS || throw(BoundsError(1:BOARD_ROWS, row))
    return @inbounds Int(geometry.mu[row])
end

@inline function after_to_source(geometry::CandidateGeometry, row::Integer)
    _require_derived(geometry)
    1 <= row <= BOARD_ROWS || throw(BoundsError(1:BOARD_ROWS, row))
    return @inbounds Int(geometry.pi[row])
end

# These target-free metadata accessors are the canonical input protocol.  Data
# backends extend the same functions for zero-copy state/candidate references;
# graph code must never reach through a concrete storage layout.
@inline hold_piece(input::TeacherSufficientInput) = input.state.meta.hold

@inline function next_piece(input::TeacherSufficientInput, role::Integer)
    1 <= role <= NEXT_COUNT || throw(BoundsError(1:NEXT_COUNT, role))
    return @inbounds input.state.meta.next[Int(role)]
end

@inline ren_value(input::TeacherSufficientInput) = input.state.meta.ren

@inline back_to_back_value(input::TeacherSufficientInput) =
    input.state.meta.back_to_back

@inline tspin_value(input::TeacherSufficientInput) = input.candidate.meta.tspin

@inline placement_count(input::TeacherSufficientInput) =
    Int(input.candidate.count)

@inline function placement_position(
    input::TeacherSufficientInput,
    index::Integer,
)
    count = placement_count(input)
    1 <= index <= count || throw(BoundsError(1:count, index))
    return @inbounds input.candidate.positions[index]
end

@inline function no_clear_dirty_count(geometry::CandidateGeometry)
    _require_no_clear(geometry)
    return Int(geometry.dirty_count)
end

@inline function no_clear_dirty_position(
    geometry::CandidateGeometry,
    index::Integer,
)
    _require_no_clear(geometry)
    count = Int(geometry.dirty_count)
    1 <= index <= count || throw(BoundsError(1:count, index))
    return @inbounds geometry.dirty_positions[index]
end

function no_clear_event(
    geometry::CandidateGeometry,
    row::Integer,
    column::Integer,
)
    _require_no_clear(geometry)
    _in_bounds(row, column) || throw(BoundsError(
        geometry.after,
        (row, column),
    ))
    position = _position(Int(row), Int(column))
    @inbounds for index in 1:Int(geometry.dirty_count)
        geometry.dirty_positions[index] == position && return EVENT_PRESENT
    end
    return NO_EVENT
end

@inline function before_cell(
    input::TeacherSufficientInput,
    row::Integer,
    column::Integer,
)
    _in_bounds(row, column) || throw(BoundsError(
        input.state.before,
        (row, column),
    ))
    return @inbounds input.state.before[row, column]
end

@inline function placement_cell(
    input::TeacherSufficientInput,
    row::Integer,
    column::Integer,
)
    _in_bounds(row, column) || throw(BoundsError(
        input.candidate.raw_placement,
        (row, column),
    ))
    return @inbounds input.candidate.raw_placement[row, column]
end

@inline function preclear_cell(
    geometry::CandidateGeometry,
    row::Integer,
    column::Integer,
)
    _require_derived(geometry)
    _in_bounds(row, column) || throw(BoundsError(
        geometry.preclear,
        (row, column),
    ))
    return @inbounds geometry.preclear[row, column]
end

@inline function after_cell(
    geometry::CandidateGeometry,
    row::Integer,
    column::Integer,
)
    _require_derived(geometry)
    _in_bounds(row, column) || throw(BoundsError(
        geometry.after,
        (row, column),
    ))
    return @inbounds geometry.after[row, column]
end

@inline function before_site(
    input::TeacherSufficientInput,
    row::Integer,
    column::Integer,
)
    _in_bounds(row, column) || return OUTSIDE
    return @inbounds input.state.before[row, column] == OCCUPIED ?
        SITE_OCCUPIED : SITE_EMPTY
end

@inline function preclear_site(
    input::TeacherSufficientInput,
    row::Integer,
    column::Integer,
)
    _in_bounds(row, column) || return OUTSIDE
    @inbounds if input.candidate.raw_placement[row, column] == PRESENT
        return SITE_PLACED
    end
    return @inbounds input.state.before[row, column] == OCCUPIED ?
        SITE_OCCUPIED : SITE_EMPTY
end

@inline function after_site(
    geometry::CandidateGeometry,
    row::Integer,
    column::Integer,
)
    _require_derived(geometry)
    _in_bounds(row, column) || return OUTSIDE
    return @inbounds geometry.after[row, column] == OCCUPIED ?
        SITE_OCCUPIED : SITE_EMPTY
end

end # module CanonicalTetrisInput
