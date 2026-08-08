using Test

module CanonicalSpatialDriveTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "CanonicalTetrisInput.jl"))
include(joinpath(@__DIR__, "OrderedMultiscaleTopology.jl"))
include(joinpath(@__DIR__, "CanonicalSpatialDrive.jl"))
end
const H = CanonicalSpatialDriveTestHarness
const Cell = H.ActiveApicalCell
const Input = H.CanonicalTetrisInput
const Topology = H.OrderedMultiscaleTopology
const Drive = H.CanonicalSpatialDrive

@enum TestTransitionPhase::UInt8 begin
    TEST_COMMON_BEFORE = 0x01
    TEST_CANDIDATE_AFTER = 0x02
    TEST_MANDATORY_DAG = 0x03
    TEST_EVENT_WAVE = 0x04
    TEST_OUTPUT_POPULATION = 0x05
end

struct SyntheticSiteAccessor
    sites::Matrix{Input.SiteToken}
end

@inline function Drive.spatial_site(
    accessor::SyntheticSiteAccessor,
    row::Integer,
    column::Integer,
)
    if 1 <= row <= Input.BOARD_ROWS && 1 <= column <= Input.BOARD_COLUMNS
        return @inbounds accessor.sites[row, column]
    end
    return Input.OUTSIDE
end

function canonical_fixture()
    before = fill(Input.EMPTY, Input.BOARD_ROWS, Input.BOARD_COLUMNS)
    before[10, 4] = Input.OCCUPIED
    before[12, 5] = Input.OCCUPIED
    placement = fill(Input.ABSENT, Input.BOARD_ROWS, Input.BOARD_COLUMNS)
    placement[10, 5] = Input.PRESENT
    placement[10, 6] = Input.PRESENT
    placement[11, 5] = Input.PRESENT
    placement[11, 6] = Input.PRESENT
    meta = Input.StateMeta(
        Input.NONE,
        (
            Input.PIECE_I,
            Input.PIECE_O,
            Input.PIECE_T,
            Input.PIECE_S,
            Input.PIECE_Z,
        ),
        0,
        Input.FALSE_VALUE,
    )
    state = Input.StateObservation(before, meta)
    candidate = Input.CandidateObservation(
        placement,
        Input.CandidateMeta(Input.FALSE_VALUE),
    )
    input = Input.TeacherSufficientInput(state, candidate)
    geometry = Input.CandidateGeometry()
    Input.derive_candidate!(geometry, input)
    return input, geometry
end

function synthetic_accessor(token::Input.SiteToken=Input.SITE_EMPTY)
    return SyntheticSiteAccessor(fill(
        token,
        Input.BOARD_ROWS,
        Input.BOARD_COLUMNS,
    ))
end

@testset "canonical 3x3 spatial drive" begin
    @test Cell.N_BASAL == Drive.NEIGHBOR_COUNT == 8
    @test Cell.INPUT_DIM == 27
    @test Drive.PHASE_COUNT == 5
    @test Drive.neighbor_offset(1) == (-1, -1)
    @test Drive.neighbor_offset(2) == (-1, 0)
    @test Drive.neighbor_offset(3) == (-1, 1)
    @test Drive.neighbor_offset(4) == (0, -1)
    @test Drive.neighbor_offset(5) == (0, 1)
    @test Drive.neighbor_offset(6) == (1, -1)
    @test Drive.neighbor_offset(7) == (1, 0)
    @test Drive.neighbor_offset(8) == (1, 1)
end

@testset "canonical owned accessors and future-view protocol" begin
    input, geometry = canonical_fixture()
    before = Drive.BeforeSiteAccessor(input)
    preclear = Drive.PreclearSiteAccessor(input)
    after = Drive.AfterSiteAccessor(geometry)

    @test Drive.spatial_site(before, 10, 5) == Input.SITE_EMPTY
    @test Drive.spatial_site(preclear, 10, 5) == Input.SITE_PLACED
    @test Drive.spatial_site(after, 10, 5) == Input.SITE_OCCUPIED
    @test Drive.spatial_site(before, 10, 4) == Input.SITE_OCCUPIED
    @test Drive.spatial_site(before, 0, 5) == Input.OUTSIDE
    @test Drive.spatial_site(preclear, 25, 5) == Input.OUTSIDE
    @test Drive.spatial_site(after, 10, 11) == Input.OUTSIDE

    # A type that does not inherit a helper-owned abstract base can implement
    # the one-method protocol.  CandidateInputRef can use this exact path.
    synthetic = synthetic_accessor(Input.SITE_PLACED)
    @test Drive.spatial_site(synthetic, 1, 1) == Input.SITE_PLACED
    @test Drive.spatial_site(synthetic, 0, 1) == Input.OUTSIDE
    destination = zeros(Float32, Cell.INPUT_DIM)
    @test Drive.fill_spatial_drive!(
        destination,
        synthetic,
        12,
        5,
        Topology.AFTER_PLANE,
        TEST_EVENT_WAVE,
    ) === destination
end

@testset "all site semantics are nonzero, balanced and distinct" begin
    signatures = Vector{Vector{Float32}}()
    basal_totals = Float32[]
    for token in (
        Input.SITE_EMPTY,
        Input.SITE_OCCUPIED,
        Input.SITE_PLACED,
        Input.OUTSIDE,
    )
        destination = zeros(Float32, Cell.INPUT_DIM)
        Drive.fill_spatial_drive!(
            destination,
            synthetic_accessor(token),
            12,
            5,
            Topology.BEFORE_PLANE,
            TEST_COMMON_BEFORE,
        )
        @test all(isfinite, destination)
        @test all(>(0.0f0), destination)
        push!(signatures, destination)
        push!(basal_totals, sum(@view destination[1:(3 * Cell.N_BASAL)]))
    end
    @test length(unique(signatures)) == 4
    @test all(total -> isapprox(total, basal_totals[1]; atol=1.0f-7),
              basal_totals)
end

@testset "ordered neighbours map one-to-one to basal compartments" begin
    row = 12
    column = 5
    baseline_accessor = synthetic_accessor(Input.SITE_EMPTY)
    baseline = zeros(Float32, Cell.INPUT_DIM)
    changed = similar(baseline)
    Drive.fill_spatial_drive!(
        baseline,
        baseline_accessor,
        row,
        column,
        Topology.BEFORE_PLANE,
        TEST_COMMON_BEFORE,
    )

    for branch in 1:Drive.NEIGHBOR_COUNT
        sites = copy(baseline_accessor.sites)
        delta_row, delta_column = Drive.neighbor_offset(branch)
        sites[row + delta_row, column + delta_column] = Input.SITE_OCCUPIED
        Drive.fill_spatial_drive!(
            changed,
            SyntheticSiteAccessor(sites),
            row,
            column,
            Topology.BEFORE_PLANE,
            TEST_COMMON_BEFORE,
        )
        changed_indices = findall(index -> changed[index] != baseline[index],
                                  eachindex(changed))
        @test changed_indices == [
            Cell.input_index(branch, Cell.INPUT_AMPA),
            Cell.input_index(branch, Cell.INPUT_NMDA),
            Cell.input_index(branch, Cell.INPUT_GABA),
        ]
    end

    # Board boundaries are explicit OUTSIDE evidence, never a zero padding.
    corner = zeros(Float32, Cell.INPUT_DIM)
    Drive.fill_spatial_drive!(
        corner,
        baseline_accessor,
        1,
        1,
        Topology.BEFORE_PLANE,
        TEST_COMMON_BEFORE,
    )
    for branch in (1, 2, 3, 4, 6)
        @test all(>(0.0f0), @view corner[
            Cell.input_index(branch, 1):Cell.input_index(branch, 3)
        ])
    end
end

@testset "same enum payload cannot alias centre, plane and phase roles" begin
    # All three interventions change raw payload 1 -> 2, but their neural
    # signatures occupy disjoint apical receptor coordinates.
    row = 12
    column = 5
    empty = synthetic_accessor(Input.SITE_EMPTY)
    base = zeros(Float32, Cell.INPUT_DIM)
    centre_changed = similar(base)
    plane_changed = similar(base)
    phase_changed = similar(base)
    Drive.fill_spatial_drive!(
        base, empty, row, column,
        Topology.BEFORE_PLANE, TEST_COMMON_BEFORE,
    )

    occupied_sites = copy(empty.sites)
    occupied_sites[row, column] = Input.SITE_OCCUPIED
    Drive.fill_spatial_drive!(
        centre_changed,
        SyntheticSiteAccessor(occupied_sites),
        row,
        column,
        Topology.BEFORE_PLANE,
        TEST_COMMON_BEFORE,
    )
    Drive.fill_spatial_drive!(
        plane_changed,
        empty,
        row,
        column,
        Topology.AFTER_PLANE,
        TEST_COMMON_BEFORE,
    )
    Drive.fill_spatial_drive!(
        phase_changed,
        empty,
        row,
        column,
        Topology.BEFORE_PLANE,
        TEST_CANDIDATE_AFTER,
    )

    apical = Cell.N_COMPARTMENTS
    @test findall(!iszero, centre_changed .- base) ==
        [Cell.input_index(apical, Cell.INPUT_AMPA)]
    @test findall(!iszero, plane_changed .- base) ==
        [Cell.input_index(apical, Cell.INPUT_NMDA)]
    @test findall(!iszero, phase_changed .- base) ==
        [Cell.input_index(apical, Cell.INPUT_GABA)]
    @test centre_changed != plane_changed
    @test centre_changed != phase_changed
    @test plane_changed != phase_changed
end

@testset "bounds, Float32 and allocation-free hot path" begin
    input, geometry = canonical_fixture()
    before = Drive.BeforeSiteAccessor(input)
    preclear = Drive.PreclearSiteAccessor(input)
    after = Drive.AfterSiteAccessor(geometry)
    destination = zeros(Float32, Cell.INPUT_DIM)

    Drive.fill_spatial_drive!(
        destination, before, 12, 5,
        Topology.BEFORE_PLANE, TEST_COMMON_BEFORE,
    )
    Drive.fill_spatial_drive!(
        destination, preclear, 12, 5,
        Topology.AFTER_PLANE, TEST_CANDIDATE_AFTER,
    )
    Drive.fill_spatial_drive!(
        destination, after, 12, 5,
        Topology.AFTER_PLANE, TEST_EVENT_WAVE,
    )
    @test @allocated(Drive.fill_spatial_drive!(
        destination, before, 12, 5,
        Topology.BEFORE_PLANE, TEST_COMMON_BEFORE,
    )) == 0
    @test @allocated(Drive.fill_spatial_drive!(
        destination, preclear, 12, 5,
        Topology.AFTER_PLANE, TEST_CANDIDATE_AFTER,
    )) == 0
    @test @allocated(Drive.fill_spatial_drive!(
        destination, after, 12, 5,
        Topology.AFTER_PLANE, TEST_EVENT_WAVE,
    )) == 0

    @test_throws DimensionMismatch Drive.fill_spatial_drive!(
        zeros(Float32, Cell.INPUT_DIM - 1), before, 12, 5,
        Topology.BEFORE_PLANE, TEST_COMMON_BEFORE,
    )
    @test_throws MethodError Drive.fill_spatial_drive!(
        zeros(Float64, Cell.INPUT_DIM), before, 12, 5,
        Topology.BEFORE_PLANE, TEST_COMMON_BEFORE,
    )
    @test_throws BoundsError Drive.fill_spatial_drive!(
        destination, before, 0, 5,
        Topology.BEFORE_PLANE, TEST_COMMON_BEFORE,
    )
    @test_throws BoundsError Drive.fill_spatial_drive!(
        destination, before, 12, 11,
        Topology.BEFORE_PLANE, TEST_COMMON_BEFORE,
    )
    @test_throws BoundsError Drive.fill_spatial_drive!(
        destination, before, 12, 5, 0, TEST_COMMON_BEFORE,
    )
    @test_throws BoundsError Drive.fill_spatial_drive!(
        destination, before, 12, 5, 3, TEST_COMMON_BEFORE,
    )
    @test_throws BoundsError Drive.fill_spatial_drive!(
        destination, before, 12, 5, Topology.BEFORE_PLANE, 0,
    )
    @test_throws BoundsError Drive.fill_spatial_drive!(
        destination, before, 12, 5, Topology.BEFORE_PLANE, 6,
    )
    @test_throws BoundsError Drive.neighbor_offset(0)
    @test_throws BoundsError Drive.neighbor_offset(9)
end
