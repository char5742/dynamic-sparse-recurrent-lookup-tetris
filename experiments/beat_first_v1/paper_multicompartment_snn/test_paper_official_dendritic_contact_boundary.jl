using Test

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropReleaseProductionFinalV3.jl",
))

const BoundaryArena = Main.PaperArenaTrainingFinalProduction
const _BOUNDARY_SOURCE_PATH = Symbol(joinpath(
    @__DIR__,
    "PaperArenaOfficialDendriticContactBoundary.jl",
))

Base.include(
    BoundaryArena,
    String(_BOUNDARY_SOURCE_PATH),
)

@testset "runtime-independent Official Hay contact boundary" begin
    @test BoundaryArena.OFFICIAL_HAY_SEGMENT_COUNT == 642
    @test BoundaryArena.OFFICIAL_HAY_DENDRITIC_RANGE == 2:640
    @test BoundaryArena.OFFICIAL_HAY_DENDRITIC_COUNT == 639
    @test BoundaryArena._official_hay_contact_catalog() ==
        UInt16.(2:640)

    model = (
        sensory_contacts=701,
        recurrent_contacts=701,
        workspace_contacts=701,
        blocks=7,
    )
    inputs, recurrent, workspace =
        BoundaryArena._release_initial_locations(model)
    for locations in (inputs, recurrent, workspace)
        @test all(
            BoundaryArena._official_hay_contact_is_legal,
            locations,
        )
        @test !(UInt16(1) in locations)
        @test !(UInt16(641) in locations)
        @test !(UInt16(642) in locations)
    end
    @test extrema(inputs) == (UInt16(2), UInt16(640))

    proposals = [
        BoundaryArena._official_hay_contact_proposal(
            millisecond,
            contact,
            block,
        )
        for millisecond in 1:1500
        for contact in 1:5
        for block in 1:7
    ]
    @test extrema(proposals) == (2, 640)
    @test all(
        BoundaryArena._official_hay_contact_is_legal,
        proposals,
    )
    @test collect(
        BoundaryArena._official_hay_consolidation_slots(),
    ) == collect(2:640)

    @test which(
        BoundaryArena._release_initial_locations,
        Tuple{typeof(model)},
    ).file == _BOUNDARY_SOURCE_PATH
    @test which(
        BoundaryArena._consolidate_one_location_canonical!,
        Tuple{BoundaryArena.PaperTrainer},
    ).file == _BOUNDARY_SOURCE_PATH
    @test which(
        BoundaryArena._consolidate_workspace_location!,
        Tuple{BoundaryArena.PaperTrainer},
    ).file == _BOUNDARY_SOURCE_PATH
end

