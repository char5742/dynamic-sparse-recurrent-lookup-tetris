# Final canonical release loader.  V2 establishes the single type universe;
# these last bindings select the allocation-safe MPMC update path.
include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropReleaseProductionFinalV2.jl",
))

Base.include(
    HD_RELEASE_V2_ARENA,
    joinpath(
        @__DIR__,
        "PaperArenaExecutorFinalReleaseHotfixV2.jl",
    ),
)
Base.include(
    HD_RELEASE_V2_ARENA,
    joinpath(
        @__DIR__,
        "PaperArenaExecutorFinalBindings.jl",
    ),
)

which(
    HD_RELEASE_V2_ARENA.paper_arena_update!,
    Tuple{HD_RELEASE_V2_ARENA.PaperExecutorFinal},
).file == Symbol(joinpath(
    @__DIR__,
    "PaperArenaExecutorFinalBindings.jl",
)) || error("public final update is not bound to the hotfix path")

const HD_SWSNN_TWINPROP_RELEASE_FINAL_V3 = HD_RELEASE_V2
