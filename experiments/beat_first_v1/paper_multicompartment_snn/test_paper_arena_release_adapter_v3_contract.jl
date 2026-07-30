using Test

const ROOT = @__DIR__

include(joinpath(ROOT, "LoadHDSWSNNTwinPropProduction.jl"))

const ArenaV3Contract =
    Main.HDSWSNNTwinPropProduction.Training

if !isdefined(
    ArenaV3Contract,
    :DistilledElevenStateCellFinal,
)
    Core.eval(
        ArenaV3Contract,
        :(const DistilledElevenStateCellFinal =
            Main.DistilledElevenStateCellFinal),
    )
end

for filename in (
    "PaperArenaReleaseAdapterV3.jl",
    "PaperArenaReleaseLearning.jl",
    "PaperArenaReleaseStructure.jl",
    "PaperArenaReleaseLegalContactMethodsV3.jl",
    "PaperArenaExecutorFinalReleaseV2.jl",
    "PaperArenaExecutorFinalReleaseHotfixV2.jl",
    "PaperArenaExecutorFinalBindings.jl",
)
    Base.include(
        ArenaV3Contract,
        joinpath(ROOT, filename),
    )
end

@testset "FinalProduction release-v3 adapter contract" begin
    @test ArenaV3Contract ===
        Main.PaperArenaTrainingFinalProduction
    @test ArenaV3Contract.RELEASE_ADAPTER_CONTRACT_VERSION == 3
    @test ArenaV3Contract.ReleaseCell ===
        ArenaV3Contract.DistilledElevenStateCellReleaseRuntimeV3
    @test ArenaV3Contract.ReleaseCell.Final ===
        Main.DistilledElevenStateCellFinal
    @test fieldtype(
        ArenaV3Contract.ReleaseCellRuntime,
        1,
    ) === ArenaV3Contract.ReleaseCell.TrustedReleaseRuntime
    @test fieldtype(
        ArenaV3Contract.PaperReleaseAux,
        1,
    ) === ArenaV3Contract.ReleaseCell.TrustedReleaseRuntime
    @test :paper_scale in fieldnames(
        ArenaV3Contract.ReleaseCell.TrustedReleaseRuntime,
    )

    @test ArenaV3Contract.RELEASE_OFFICIAL_SEGMENT_COUNT == 642
    @test ArenaV3Contract.RELEASE_LEGAL_CONTACT_RANGE == 2:640
    @test length(
        ArenaV3Contract._release_legal_location_catalog(),
    ) == 639
    @test eltype(
        ArenaV3Contract._release_legal_location_catalog(),
    ) === UInt16

    model = (
        sensory_contacts=17,
        recurrent_contacts=13,
        workspace_contacts=7,
        blocks=5,
    )
    input_location, recurrent_location, workspace_location =
        ArenaV3Contract._release_initial_locations(model)
    for locations in (
        input_location,
        recurrent_location,
        workspace_location,
    )
        @test eltype(locations) === UInt16
        @test all(location -> 2 <= location <= 640, locations)
        @test all(location -> location != 1, locations)
        @test all(location -> location < 641, locations)
    end
    proposals = UInt16[
        ArenaV3Contract._release_legal_proposal(
            millisecond,
            contact,
            block,
        )
        for millisecond in 1:100,
            contact in 1:17,
            block in 1:5
    ]
    @test all(location -> 2 <= location <= 640, proposals)

    legal_file = Symbol(joinpath(
        ROOT,
        "PaperArenaReleaseLegalContactMethodsV3.jl",
    ))
    @test which(
        ArenaV3Contract._release_location_evidence!,
        Tuple{
            Matrix{Float32},
            Matrix{Float32},
            Matrix{UInt16},
            ArenaV3Contract.ReceptorEligibility,
            ArenaV3Contract.ReleaseCellRuntime,
            Int,
            Int,
            Int,
            Float32,
            Int,
        },
    ).file == legal_file
    @test which(
        ArenaV3Contract._consolidate_one_location_canonical!,
        Tuple{ArenaV3Contract.PaperTrainer},
    ).file == legal_file
    @test which(
        ArenaV3Contract._consolidate_workspace_location!,
        Tuple{ArenaV3Contract.PaperTrainer},
    ).file == legal_file

    @test isdefined(ArenaV3Contract, :PaperExecutorFinal)
    @test isbitstype(ArenaV3Contract.PaperFinalWorkItem)
end
