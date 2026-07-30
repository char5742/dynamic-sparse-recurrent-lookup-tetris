# Sole final production loader for HD-SWSNN-TwinProp release-v2.
#
# Do not combine this loader with provisional DistilledElevenStateCell,
# Production, or ReleaseRuntime-v1 modules.
include(joinpath(@__DIR__, "LoadPaperArenaCanonicalFinal.jl"))

Base.include(
    Main.PaperArenaTrainingFinal,
    joinpath(@__DIR__, "PaperArenaReleaseAdapter.jl"),
)
Base.include(
    Main.PaperArenaTrainingFinal,
    joinpath(@__DIR__, "PaperArenaReleaseLearning.jl"),
)
Base.include(
    Main.PaperArenaTrainingFinal,
    joinpath(@__DIR__, "PaperArenaReleaseStructure.jl"),
)
Base.include(
    Main.PaperArenaTrainingFinal,
    joinpath(@__DIR__, "PaperExecutorHotfixV3.jl"),
)

Core.eval(
    Main.PaperArenaTrainingFinal,
    quote
        export enable_release_runtime!,
            paper_checkpoint_integrity!,
            paper_end_run_integrity!,
            paper_preflight_integrity!
    end,
)
