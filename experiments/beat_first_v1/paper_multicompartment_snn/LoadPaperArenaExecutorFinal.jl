# Canonical additive loader for the final MPMC executor.
#
# LoadPaperArenaCanonical establishes PaperArenaTrainingFinal and its
# canonical learning overrides.  The next three includes are split only
# because the Windows deny-read ACL prevented apply_patch from updating a
# newly-added file in place during development.

include(joinpath(@__DIR__, "LoadPaperArenaCanonical.jl"))

for filename in (
    "PaperArenaExecutorFinal.jl",
    "PaperArenaExecutorFinalHotfix.jl",
    "PaperArenaExecutorFinalBindings.jl",
)
    Base.include(
        Main.PaperArenaTrainingFinal,
        joinpath(@__DIR__, filename),
    )
end

