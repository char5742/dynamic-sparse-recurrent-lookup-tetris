using Test
using Random
using Lux

const ROOT = @__DIR__

include(joinpath(ROOT, "LoadHDSWSNNTwinPropProduction.jl"))

const ScaleArenaV3 =
    Main.HDSWSNNTwinPropProduction.Training

if !isdefined(
    ScaleArenaV3,
    :DistilledElevenStateCellFinal,
)
    Core.eval(
        ScaleArenaV3,
        :(const DistilledElevenStateCellFinal =
            Main.DistilledElevenStateCellFinal),
    )
end

for filename in (
    "PaperArenaReleaseAdapterV3.jl",
    "PaperArenaReleaseScaleModesV3.jl",
)
    Base.include(
        ScaleArenaV3,
        joinpath(ROOT, filename),
    )
end

@testset "release-v3 explicit scale modes" begin
    scale_file = Symbol(joinpath(
        ROOT,
        "PaperArenaReleaseScaleModesV3.jl",
    ))
    @test which(
        ScaleArenaV3.enable_release_runtime!,
        Tuple{ScaleArenaV3.PaperTrainer,AbstractString},
    ).file == scale_file
    @test which(
        ScaleArenaV3.enable_development_release_runtime!,
        Tuple{ScaleArenaV3.PaperTrainer,AbstractString},
    ).file == scale_file
    @test which(
        ScaleArenaV3.paper_preflight_integrity!,
        Tuple{ScaleArenaV3.PaperTrainer},
    ).file == scale_file
    @test which(
        ScaleArenaV3.paper_checkpoint_integrity!,
        Tuple{ScaleArenaV3.PaperTrainer},
    ).file == scale_file
    @test which(
        ScaleArenaV3.paper_end_run_integrity!,
        Tuple{ScaleArenaV3.PaperTrainer},
    ).file == scale_file

    model = ScaleArenaV3.Model.build_paper_model(:tiny)
    parameters = Lux.initialparameters(
        Xoshiro(0x5343414c455633),
        model,
    )
    invalid_artifact =
        joinpath(ROOT, "PaperArenaReleaseAdapterV3.jl")
    trainer = ScaleArenaV3.PaperTrainer(
        model,
        parameters;
        state_batch=1,
        width=1,
        cell_mode=:distilled_frozen,
        cell_artifact=invalid_artifact,
    )

    @test_throws UndefKeywordError(
        :development_scale_chain,
    ) ScaleArenaV3.enable_development_release_runtime!(
        trainer,
        invalid_artifact,
    )
    @test_throws ErrorException ScaleArenaV3.enable_development_release_runtime!(
        trainer,
        invalid_artifact;
        development_scale_chain=false,
    )
    @test !haskey(ScaleArenaV3._RELEASE_SCALE_MODE, trainer)
end
