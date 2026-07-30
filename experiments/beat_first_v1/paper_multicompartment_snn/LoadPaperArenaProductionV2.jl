# Canonical production arena loader used by the final runner.
include(joinpath(@__DIR__, "LoadPaperArenaCanonicalFinal.jl"))

Base.include(
    Main.PaperArenaTrainingFinal,
    joinpath(@__DIR__, "PaperExecutorHotfixV3.jl"),
)
