using Test

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropReleaseProductionFinalV4LegalContacts.jl",
))

const LegalArena = Main.PaperArenaTrainingFinalProduction
const _LEGAL_OVERLAY_PATH = Symbol(joinpath(
    @__DIR__,
    "PaperArenaReleaseLegalContactSegmentsV2.jl",
))

@testset "OfficialV2 legal Hay contact segments" begin
    @test LegalArena.RELEASE_OFFICIAL_SEGMENT_COUNT == 642
    @test LegalArena.RELEASE_LEGAL_CONTACT_COUNT == 639
    @test LegalArena.RELEASE_LEGAL_CONTACT_RANGE == 2:640
    @test LegalArena._release_legal_location_catalog() ==
        UInt16.(2:640)

    model = (
        sensory_contacts=701,
        recurrent_contacts=701,
        workspace_contacts=701,
        blocks=7,
    )
    inputs, recurrent, workspace =
        LegalArena._release_initial_locations(model)
    for locations in (inputs, recurrent, workspace)
        @test eltype(locations) === UInt16
        @test all(
            LegalArena._release_location_is_legal,
            locations,
        )
        @test !(UInt16(1) in locations)
        @test !(UInt16(641) in locations)
        @test !(UInt16(642) in locations)
    end
    @test UInt16(2) in inputs
    @test UInt16(640) in inputs

    proposals = UInt16[
        LegalArena._release_legal_proposal(
            millisecond,
            contact,
            block,
        )
        for millisecond in 1:1500
        for contact in 1:5
        for block in 1:7
    ]
    @test all(
        LegalArena._release_location_is_legal,
        proposals,
    )
    @test minimum(proposals) == UInt16(2)
    @test maximum(proposals) == UInt16(640)
    @test !(UInt16(1) in proposals)
    @test !(UInt16(641) in proposals)
    @test !(UInt16(642) in proposals)
    @test collect(
        LegalArena._release_legal_consolidation_slots(),
    ) == collect(2:640)

    @test which(
        LegalArena._release_initial_locations,
        Tuple{typeof(model)},
    ).file == _LEGAL_OVERLAY_PATH
    @test which(
        LegalArena._consolidate_one_location_canonical!,
        Tuple{LegalArena.PaperTrainer},
    ).file == _LEGAL_OVERLAY_PATH
    @test which(
        LegalArena._consolidate_workspace_location!,
        Tuple{LegalArena.PaperTrainer},
    ).file == _LEGAL_OVERLAY_PATH
end

