# Canonical V5Final loader with release-builder and fail-closed registration.

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropReleaseV5SealedV2FinalProduction.jl",
))

Base.include(
    HD_RELEASE_V5_FINAL_ARENA,
    joinpath(
        @__DIR__,
        "PaperArenaReleaseAdapterV5FailClosed.jl",
    ),
)
Base.include(
    HD_RELEASE_V5_FINAL_BASE,
    joinpath(
        @__DIR__,
        "HDSWSNNTwinPropReleaseBuilderV5Final.jl",
    ),
)

which(
    HD_RELEASE_V5_FINAL_ARENA.register_paper_trainer_aux!,
    Tuple{HD_RELEASE_V5_FINAL_ARENA.PaperTrainer},
).file == Symbol(joinpath(
    @__DIR__,
    "PaperArenaReleaseAdapterV5FailClosed.jl",
)) || error("generic distilled registration still bypasses V5Final")
which(
    HD_RELEASE_V5_FINAL_BASE.build_production_trainer,
    Tuple{
        HD_RELEASE_V5_FINAL_BASE.ProductionBundle,
        Any,
        Any,
    },
).file == Symbol(joinpath(
    @__DIR__,
    "HDSWSNNTwinPropReleaseBuilderV5Final.jl",
)) || error("production trainer builder bypasses V5Final")

const HD_SWSNN_TWINPROP_RELEASE_V5_SEALED_V2_CANONICAL =
    HD_RELEASE_V5_FINAL_BASE
