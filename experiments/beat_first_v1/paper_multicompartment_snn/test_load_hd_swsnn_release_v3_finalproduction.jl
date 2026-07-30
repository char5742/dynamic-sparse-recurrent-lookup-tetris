using Test

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropReleaseV3FinalProduction.jl",
))

const ArenaV3 = Main.HD_RELEASE_V3_ARENA

@testset "release-v3 exact FinalProduction loader" begin
    @test ArenaV3 ===
        Main.PaperArenaTrainingFinalProduction
    @test ArenaV3.Distilled ===
        Main.DistilledElevenStateCellFinal
    @test ArenaV3.ReleaseCell.Final ===
        Main.DistilledElevenStateCellFinal
    @test ArenaV3.RELEASE_ADAPTER_CONTRACT_VERSION == 3
    @test ArenaV3.RELEASE_LEGAL_CONTACT_RANGE == 2:640
    @test ArenaV3.RELEASE_OFFICIAL_SEGMENT_COUNT == 642
    @test isbitstype(ArenaV3.PaperFinalWorkItem)
    @test hasmethod(
        ArenaV3.paper_arena_update!,
        Tuple{ArenaV3.PaperExecutorFinal},
    )
    @test which(
        ArenaV3.paper_arena_update!,
        Tuple{ArenaV3.PaperExecutorFinal},
    ).file == Symbol(joinpath(
        @__DIR__,
        "PaperArenaExecutorFinalBindings.jl",
    ))
end
