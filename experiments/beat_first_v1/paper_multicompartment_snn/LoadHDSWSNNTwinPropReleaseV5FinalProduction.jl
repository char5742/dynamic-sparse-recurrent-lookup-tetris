# Sole canonical FinalProduction loader for sealed RuntimeV5 artifacts.
#
# RuntimeV3 remains a private numerical ABI dependency of RuntimeV5.  It is
# never exposed as an artifact registration path by this loader.

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropProductionFinal.jl",
))

const HD_RELEASE_V5_BASE =
    Main.HDSWSNNTwinPropProduction
const HD_RELEASE_V5_ARENA =
    HD_RELEASE_V5_BASE.Training

if !isdefined(
    HD_RELEASE_V5_ARENA,
    :DistilledElevenStateCellFinal,
)
    Core.eval(
        HD_RELEASE_V5_ARENA,
        :(const DistilledElevenStateCellFinal =
            Main.DistilledElevenStateCellFinal),
    )
end

isdefined(HD_RELEASE_V5_ARENA, :ReleaseCell) &&
    error(
        "legacy release runtime was loaded before canonical V5",
    )

for filename in (
    "PaperArenaReleaseAdapterV5.jl",
    "PaperArenaReleaseLearning.jl",
    "PaperArenaReleaseStructure.jl",
    "PaperArenaReleaseLegalContactMethodsV3.jl",
)
    Base.include(
        HD_RELEASE_V5_ARENA,
        joinpath(@__DIR__, filename),
    )
end

if !isdefined(
    HD_RELEASE_V5_ARENA,
    :paper_arena_update_hotfinal!,
)
    Base.include(
        HD_RELEASE_V5_ARENA,
        joinpath(
            @__DIR__,
            "PaperArenaExecutorFinalHotfix.jl",
        ),
    )
end
if !hasmethod(
    HD_RELEASE_V5_ARENA.paper_arena_update!,
    Tuple{HD_RELEASE_V5_ARENA.PaperExecutorFinal},
)
    Base.include(
        HD_RELEASE_V5_ARENA,
        joinpath(
            @__DIR__,
            "PaperArenaExecutorFinalBindings.jl",
        ),
    )
end
Base.include(
    HD_RELEASE_V5_ARENA,
    joinpath(
        @__DIR__,
        "PaperArenaExecutorFinalReleaseV2.jl",
    ),
)
Base.include(
    HD_RELEASE_V5_ARENA,
    joinpath(
        @__DIR__,
        "PaperArenaExecutorFinalReleaseHotfixV2.jl",
    ),
)
Base.include(
    HD_RELEASE_V5_ARENA,
    joinpath(
        @__DIR__,
        "PaperArenaExecutorFinalBindings.jl",
    ),
)

Core.eval(
    HD_RELEASE_V5_ARENA,
    quote
        export PaperExecutorFinal,
            ReleaseSecurityCell,
            RELEASE_ADAPTER_SECURITY_CONTRACT_VERSION,
            RELEASE_REQUIRED_ARTIFACT_RUNTIME,
            enable_release_runtime!,
            enable_development_release_runtime!,
            paper_release_adapter_contract_version,
            paper_release_scale_mode,
            paper_checkpoint_integrity!,
            paper_end_run_integrity!,
            paper_preflight_integrity!,
            paper_arena_update_serial_final!
    end,
)

HD_RELEASE_V5_ARENA ===
    Main.PaperArenaTrainingFinalProduction ||
    error("release-v5 arena is not exact FinalProduction")
HD_RELEASE_V5_ARENA.Distilled ===
    Main.DistilledElevenStateCellFinal ||
    error("release-v5 arena has split Final cell identity")
HD_RELEASE_V5_ARENA.ReleaseSecurityCell ===
    HD_RELEASE_V5_ARENA.DistilledElevenStateCellReleaseRuntimeV5 ||
    error("release-v5 security runtime identity differs")
HD_RELEASE_V5_ARENA.ReleaseSecurityCell.V3.Final ===
    Main.DistilledElevenStateCellFinal ||
    error("release-v5 numerical runtime has split Final identity")
HD_RELEASE_V5_ARENA.paper_release_adapter_contract_version() == 5 ||
    error("release-v5 adapter security contract differs")
HD_RELEASE_V5_ARENA.RELEASE_REQUIRED_ARTIFACT_RUNTIME ===
    :sealed_runtime_v5 ||
    error("release-v5 artifact runtime declaration differs")
isbitstype(HD_RELEASE_V5_ARENA.PaperFinalWorkItem) ||
    error("release-v5 final MPMC work item is not isbits")
which(
    HD_RELEASE_V5_ARENA.enable_release_runtime!,
    Tuple{
        HD_RELEASE_V5_ARENA.PaperTrainer,
        AbstractString,
    },
).file == Symbol(joinpath(
    @__DIR__,
    "PaperArenaReleaseAdapterV5.jl",
)) || error("public release registration bypasses V5")
which(
    HD_RELEASE_V5_ARENA.paper_arena_update!,
    Tuple{HD_RELEASE_V5_ARENA.PaperExecutorFinal},
).file == Symbol(joinpath(
    @__DIR__,
    "PaperArenaExecutorFinalBindings.jl",
)) || error("public final update is not bound to final MPMC hotfix")

const HD_SWSNN_TWINPROP_RELEASE_V5_FINAL_PRODUCTION =
    HD_RELEASE_V5_BASE
