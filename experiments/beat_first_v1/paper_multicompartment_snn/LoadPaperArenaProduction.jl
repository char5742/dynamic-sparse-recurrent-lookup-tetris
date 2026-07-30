# Canonical production arena loader.
#
# This is the only loader used by the production driver and validation smoke.
include(joinpath(@__DIR__, "LoadPaperArenaCanonicalFinal.jl"))

Base.include(
    Main.PaperArenaTrainingFinal,
    joinpath(@__DIR__, "PaperExecutorHotfixV2.jl"),
)
