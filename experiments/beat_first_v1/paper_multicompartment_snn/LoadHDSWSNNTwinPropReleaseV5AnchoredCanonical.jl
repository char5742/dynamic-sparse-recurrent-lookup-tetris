# Final canonical loader with externally verified sealed-V2 lineage and typed
# RuntimeV5 production/development bundles.

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropReleaseV5SealedV2Canonical.jl",
))

Base.include(
    HD_RELEASE_V5_FINAL_ARENA,
    joinpath(
        @__DIR__,
        "PaperArenaReleaseAdapterV5AnchoredFinal.jl",
    ),
)
Base.include(
    HD_RELEASE_V5_FINAL_BASE,
    joinpath(
        @__DIR__,
        "HDSWSNNTwinPropSealedV5ProductionBundle.jl",
    ),
)

Core.eval(
    HD_RELEASE_V5_FINAL_ARENA,
    quote
        export SealedV2LineageAnchor,
            assert_sealed_v5_lineage_anchor_unchanged!,
            verify_sealed_v5_lineage_anchor
    end,
)

which(
    HD_RELEASE_V5_FINAL_ARENA.enable_release_runtime!,
    Tuple{
        HD_RELEASE_V5_FINAL_ARENA.PaperTrainer,
        AbstractString,
    },
).file == Symbol(joinpath(
    @__DIR__,
    "PaperArenaReleaseAdapterV5AnchoredFinal.jl",
)) || error("release registration is not externally anchored")
which(
    HD_RELEASE_V5_FINAL_BASE.build_production_trainer,
    Tuple{
        HD_RELEASE_V5_FINAL_BASE.ProductionBundle,
        Any,
        Any,
    },
).file == Symbol(joinpath(
    @__DIR__,
    "HDSWSNNTwinPropSealedV5ProductionBundle.jl",
)) || error("legacy ProductionBundle is not fail-closed")

const HD_SWSNN_TWINPROP_RELEASE_V5_ANCHORED_CANONICAL =
    HD_RELEASE_V5_FINAL_BASE
