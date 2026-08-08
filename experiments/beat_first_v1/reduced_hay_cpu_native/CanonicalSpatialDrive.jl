module CanonicalSpatialDrive

"""
Target-free, allocation-free spatial drive for the canonical Reduced-Hay graph.

Each 24 x 10 spatial cell receives the ordered Moore neighbourhood on its
eight basal compartments and receives the centre-site, plane and transition
phase identities on the three apical receptor channels.  The three apical
roles are deliberately channel-separated: equal underlying enum values from
different semantic types can therefore never alias.

`spatial_site(accessor, row, column)` is the open accessor protocol.  The
provided wrappers cover the canonical owned input today; a future arena-backed
`CandidateInputRef` only needs to add one method for this protocol.
"""

using ..ActiveApicalCell
using ..CanonicalTetrisInput
using ..OrderedMultiscaleTopology

const Cell = ActiveApicalCell
const Input = CanonicalTetrisInput
const Topology = OrderedMultiscaleTopology

export PHASE_COUNT,
       NEIGHBOR_COUNT,
       BeforeSiteAccessor,
       PreclearSiteAccessor,
       AfterSiteAccessor,
       spatial_site,
       neighbor_offset,
       fill_spatial_drive!

# The graph contract currently has five nonzero transition phases.  This
# module intentionally does not import CanonicalDendriticGraph (which imports
# this helper); any Enum convertible to Int and valued in 1:5 is accepted.
const PHASE_COUNT = 5
const NEIGHBOR_COUNT = Cell.N_BASAL

@assert NEIGHBOR_COUNT == 8
@assert Cell.N_COMPARTMENTS == 9
@assert Cell.INPUT_CHANNELS == 3
@assert Cell.INPUT_DIM == 27
@assert Topology.ROW_COUNT == Input.BOARD_ROWS
@assert Topology.COLUMN_COUNT == Input.BOARD_COLUMNS
@assert Topology.PLANE_COUNT == 2

# Branch order is anatomical and immutable.  Rows increase down the board.
# NW, N, NE, W, E, SW, S, SE map exactly to basal compartments 1:8.
const _NEIGHBOR_OFFSETS = (
    (-1, -1),
    (-1,  0),
    (-1,  1),
    ( 0, -1),
    ( 0,  1),
    ( 1, -1),
    ( 1,  0),
    ( 1,  1),
)

# Fixed positive receptor codes.  Every site class has equal total basal
# drive (0.012), so EMPTY/OCCUPIED/PLACED/OUTSIDE identity cannot be decoded
# merely from energy.  Numeric zero remains exclusively an internal sentinel.
const _SITE_DRIVE = (
    (0.008f0, 0.003f0, 0.001f0), # SITE_EMPTY
    (0.001f0, 0.008f0, 0.003f0), # SITE_OCCUPIED
    (0.003f0, 0.001f0, 0.008f0), # SITE_PLACED
    (0.004f0, 0.004f0, 0.004f0), # OUTSIDE
)

# Apical role separation is structural, not a learned lookup:
#   AMPA <- centre-site identity
#   NMDA <- before/after plane identity
#   GABA <- transition phase identity
# Each semantic value is positive and intervention-distinct within its role.
const _CENTER_DRIVE = (0.003f0, 0.011f0, 0.006f0, 0.014f0)
const _PLANE_DRIVE = (0.004f0, 0.010f0)
const _PHASE_DRIVE = (0.002f0, 0.004f0, 0.006f0, 0.008f0, 0.010f0)

struct BeforeSiteAccessor{I}
    input::I
end

struct PreclearSiteAccessor{I}
    input::I
end

struct AfterSiteAccessor{G}
    geometry::G
end

BeforeSiteAccessor(input::Input.TeacherSufficientInput) =
    BeforeSiteAccessor{Input.TeacherSufficientInput}(input)
PreclearSiteAccessor(input::Input.TeacherSufficientInput) =
    PreclearSiteAccessor{Input.TeacherSufficientInput}(input)
AfterSiteAccessor(geometry::Input.CandidateGeometry) =
    AfterSiteAccessor{Input.CandidateGeometry}(geometry)

"""
Return one explicit `SiteToken` at `(row, column)`.

This function is the only protocol a future borrowed candidate view must
extend.  It is intentionally not restricted to an abstract base type.
"""
@inline spatial_site(accessor::BeforeSiteAccessor, row::Integer, column::Integer) =
    Input.before_site(accessor.input, row, column)

@inline spatial_site(accessor::PreclearSiteAccessor, row::Integer, column::Integer) =
    Input.preclear_site(accessor.input, row, column)

@inline spatial_site(accessor::AfterSiteAccessor, row::Integer, column::Integer) =
    Input.after_site(accessor.geometry, row, column)

@inline function neighbor_offset(branch::Integer)
    1 <= branch <= NEIGHBOR_COUNT || throw(BoundsError(1:NEIGHBOR_COUNT, branch))
    return @inbounds _NEIGHBOR_OFFSETS[Int(branch)]
end

@inline function _site_index(token::Input.SiteToken)
    index = Int(UInt8(token))
    1 <= index <= length(_SITE_DRIVE) || error("invalid SiteToken encoding")
    return index
end

@inline function _plane_index(plane)
    index = Int(plane)
    1 <= index <= Topology.PLANE_COUNT ||
        throw(BoundsError(1:Topology.PLANE_COUNT, index))
    return index
end

@inline function _phase_index(phase)
    index = Int(phase)
    1 <= index <= PHASE_COUNT || throw(BoundsError(1:PHASE_COUNT, index))
    return index
end

"""
Write one complete 27-channel Reduced-Hay input vector.

Basal compartments 1:8 receive NW,N,NE,W,E,SW,S,SE respectively.  The active
apical compartment receives centre, plane and phase on AMPA, NMDA and GABA.
All 27 entries are overwritten with positive `Float32` values; no teacher,
target, learned position embedding or lookup bank is consulted.
"""
@inline function fill_spatial_drive!(
    destination::AbstractVector{Float32},
    accessor,
    row::Integer,
    column::Integer,
    plane,
    phase,
)
    length(destination) == Cell.INPUT_DIM || throw(DimensionMismatch(
        "spatial drive destination must have length $(Cell.INPUT_DIM)",
    ))
    1 <= row <= Input.BOARD_ROWS || throw(BoundsError(1:Input.BOARD_ROWS, row))
    1 <= column <= Input.BOARD_COLUMNS ||
        throw(BoundsError(1:Input.BOARD_COLUMNS, column))
    physical_row = Int(row)
    physical_column = Int(column)
    physical_plane = _plane_index(plane)
    physical_phase = _phase_index(phase)

    @inbounds for branch in 1:NEIGHBOR_COUNT
        delta_row, delta_column = _NEIGHBOR_OFFSETS[branch]
        token = spatial_site(
            accessor,
            physical_row + delta_row,
            physical_column + delta_column,
        )
        drive = _SITE_DRIVE[_site_index(token)]
        destination[Cell.input_index(branch, Cell.INPUT_AMPA)] = drive[1]
        destination[Cell.input_index(branch, Cell.INPUT_NMDA)] = drive[2]
        destination[Cell.input_index(branch, Cell.INPUT_GABA)] = drive[3]
    end

    centre = spatial_site(accessor, physical_row, physical_column)
    centre_index = _site_index(centre)
    apical = Cell.N_COMPARTMENTS
    @inbounds begin
        destination[Cell.input_index(apical, Cell.INPUT_AMPA)] =
            _CENTER_DRIVE[centre_index]
        destination[Cell.input_index(apical, Cell.INPUT_NMDA)] =
            _PLANE_DRIVE[physical_plane]
        destination[Cell.input_index(apical, Cell.INPUT_GABA)] =
            _PHASE_DRIVE[physical_phase]
    end
    return destination
end

end # module CanonicalSpatialDrive
