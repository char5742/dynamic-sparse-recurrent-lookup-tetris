# Sole production loader for the final paper-faithful arena.
#
# The base loader establishes the final-only module and canonical optimizer /
# routing overrides.  Load the replay morphology cache afterwards, inside that
# module, so `_state_credit` never reconstructs a Hay tree in the hot path.
include(joinpath(@__DIR__, "LoadPaperArenaCanonical.jl"))

Base.include(
    Main.PaperArenaTrainingFinal,
    joinpath(@__DIR__, "PaperArenaReplayHotfix.jl"),
)
