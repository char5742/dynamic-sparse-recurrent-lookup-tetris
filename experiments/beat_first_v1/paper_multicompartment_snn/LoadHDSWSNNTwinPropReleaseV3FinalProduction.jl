# Sole canonical FinalProduction arena loader for release Runtime V3.
#
# The old V2 adapter is intentionally never included.  Every implementation is
# installed into the exact `Main.PaperArenaTrainingFinalProduction` module
# owned by the existing strict production base, so the model, trainer, workers,
# eligibility buffers, and final MPMC work items share one type universe.

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropProductionFinal.jl",
))

const HD_RELEASE_V3_BASE =
    Main.HDSWSNNTwinPropProduction
const HD_RELEASE_V3_ARENA =
    HD_RELEASE_V3_BASE.Training

if !isdefined(
    HD_RELEASE_V3_ARENA,
    :DistilledElevenStateCellFinal,
)
    Core.eval(
        HD_RELEASE_V3_ARENA,
        :(const DistilledElevenStateCellFinal =
            Main.DistilledElevenStateCellFinal),
    )
end

isdefined(HD_RELEASE_V3_ARENA, :ReleaseCell) &&
    error(
        "legacy release runtime was loaded before the canonical V3 adapter",
    )

for filename in (
    "PaperArenaReleaseAdapterV3.jl",
    "PaperArenaReleaseScaleModesV3.jl",
    "PaperArenaReleaseLearning.jl",
    "PaperArenaReleaseStructure.jl",
    "PaperArenaReleaseLegalContactMethodsV3.jl",
)
    Base.include(
        HD_RELEASE_V3_ARENA,
        joinpath(@__DIR__, filename),
    )
end

if !isdefined(
    HD_RELEASE_V3_ARENA,
    :paper_arena_update_hotfinal!,
)
    Base.include(
        HD_RELEASE_V3_ARENA,
        joinpath(
            @__DIR__,
            "PaperArenaExecutorFinalHotfix.jl",
        ),
    )
end
if !hasmethod(
    HD_RELEASE_V3_ARENA.paper_arena_update!,
    Tuple{HD_RELEASE_V3_ARENA.PaperExecutorFinal},
)
    Base.include(
        HD_RELEASE_V3_ARENA,
        joinpath(
            @__DIR__,
            "PaperArenaExecutorFinalBindings.jl",
        ),
    )
end
Base.include(
    HD_RELEASE_V3_ARENA,
    joinpath(
        @__DIR__,
        "PaperArenaExecutorFinalReleaseV2.jl",
    ),
)
Base.include(
    HD_RELEASE_V3_ARENA,
    joinpath(
        @__DIR__,
        "PaperArenaExecutorFinalReleaseHotfixV2.jl",
    ),
)
Base.include(
    HD_RELEASE_V3_ARENA,
    joinpath(
        @__DIR__,
        "PaperArenaExecutorFinalBindings.jl",
    ),
)

Core.eval(
    HD_RELEASE_V3_ARENA,
    quote
        export PaperExecutorFinal,
            enable_release_runtime!,
            enable_development_release_runtime!,
            paper_release_scale_mode,
            paper_checkpoint_integrity!,
            paper_end_run_integrity!,
            paper_preflight_integrity!,
            paper_arena_update_serial_final!
    end,
)

HD_RELEASE_V3_ARENA ===
    Main.PaperArenaTrainingFinalProduction ||
    error("release-v3 arena is not exact FinalProduction")
HD_RELEASE_V3_ARENA.Distilled ===
    Main.DistilledElevenStateCellFinal ||
    error("release-v3 arena has split Final cell identity")
HD_RELEASE_V3_ARENA.ReleaseCell.Final ===
    Main.DistilledElevenStateCellFinal ||
    error("release-v3 runtime has split Final cell identity")
HD_RELEASE_V3_ARENA.RELEASE_ADAPTER_CONTRACT_VERSION == 3 ||
    error("release-v3 adapter contract version differs")
isbitstype(HD_RELEASE_V3_ARENA.PaperFinalWorkItem) ||
    error("release-v3 final MPMC work item is not isbits")
which(
    HD_RELEASE_V3_ARENA.paper_arena_update!,
    Tuple{HD_RELEASE_V3_ARENA.PaperExecutorFinal},
).file == Symbol(joinpath(
    @__DIR__,
    "PaperArenaExecutorFinalBindings.jl",
)) || error("public final update is not bound to final MPMC hotfix")

const HD_SWSNN_TWINPROP_RELEASE_V3_FINAL_PRODUCTION =
    HD_RELEASE_V3_BASE
