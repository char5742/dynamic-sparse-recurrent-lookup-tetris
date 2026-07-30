# Sole canonical FinalProduction loader:
# exact sealed ELM release final.v2 + exact frozen RuntimeV5 11-state artifact.

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropProductionFinal.jl",
))

const HD_RELEASE_V5_FINAL_BASE =
    Main.HDSWSNNTwinPropProduction
const HD_RELEASE_V5_FINAL_ARENA =
    HD_RELEASE_V5_FINAL_BASE.Training

if !isdefined(
    HD_RELEASE_V5_FINAL_ARENA,
    :DistilledElevenStateCellFinal,
)
    Core.eval(
        HD_RELEASE_V5_FINAL_ARENA,
        :(const DistilledElevenStateCellFinal =
            Main.DistilledElevenStateCellFinal),
    )
end

isdefined(HD_RELEASE_V5_FINAL_ARENA, :ReleaseCell) &&
    error("legacy release runtime preceded canonical V5Final")

for filename in (
    "PaperArenaReleaseAdapterV5Final.jl",
    "PaperArenaReleaseLearning.jl",
    "PaperArenaReleaseStructure.jl",
    "PaperArenaReleaseLegalContactMethodsV3.jl",
)
    Base.include(
        HD_RELEASE_V5_FINAL_ARENA,
        joinpath(@__DIR__, filename),
    )
end

if !isdefined(
    HD_RELEASE_V5_FINAL_ARENA,
    :paper_arena_update_hotfinal!,
)
    Base.include(
        HD_RELEASE_V5_FINAL_ARENA,
        joinpath(
            @__DIR__,
            "PaperArenaExecutorFinalHotfix.jl",
        ),
    )
end
if !hasmethod(
    HD_RELEASE_V5_FINAL_ARENA.paper_arena_update!,
    Tuple{HD_RELEASE_V5_FINAL_ARENA.PaperExecutorFinal},
)
    Base.include(
        HD_RELEASE_V5_FINAL_ARENA,
        joinpath(
            @__DIR__,
            "PaperArenaExecutorFinalBindings.jl",
        ),
    )
end
for filename in (
    "PaperArenaExecutorFinalReleaseV2.jl",
    "PaperArenaExecutorFinalReleaseHotfixV2.jl",
    "PaperArenaExecutorFinalBindings.jl",
)
    Base.include(
        HD_RELEASE_V5_FINAL_ARENA,
        joinpath(@__DIR__, filename),
    )
end

Core.eval(
    HD_RELEASE_V5_FINAL_ARENA,
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

HD_RELEASE_V5_FINAL_ARENA ===
    Main.PaperArenaTrainingFinalProduction ||
    error("V5Final arena is not exact FinalProduction")
HD_RELEASE_V5_FINAL_ARENA.Distilled ===
    Main.DistilledElevenStateCellFinal ||
    error("V5Final arena has split Final cell identity")
HD_RELEASE_V5_FINAL_ARENA.ReleaseSecurityCell ===
    HD_RELEASE_V5_FINAL_ARENA.
        DistilledElevenStateCellReleaseRuntimeV5Final ||
    error("V5Final security runtime identity differs")
HD_RELEASE_V5_FINAL_ARENA.ReleaseSecurityCell.V3.Final ===
    Main.DistilledElevenStateCellFinal ||
    error("V5Final numerical ABI has split Final identity")
HD_RELEASE_V5_FINAL_ARENA.paper_release_adapter_contract_version() == 5 ||
    error("V5Final adapter contract differs")
HD_RELEASE_V5_FINAL_ARENA.RELEASE_REQUIRED_ARTIFACT_RUNTIME ===
    :sealed_v2_runtime_v5 ||
    error("V5Final artifact declaration differs")
HD_RELEASE_V5_FINAL_ARENA.ReleaseSecurityCell.SEALED_RELEASE_SCHEMA ==
    "hd_swsnn.paper_elm_v2.sealed_release.final.v2" ||
    error("V5Final does not require sealed final.v2")
HD_RELEASE_V5_FINAL_ARENA.ReleaseSecurityCell.
    SEALED_RELEASE_ARTIFACT_KIND ==
    "SealedOfficialELMReleaseV2" ||
    error("V5Final sealed artifact kind differs")
isbitstype(HD_RELEASE_V5_FINAL_ARENA.PaperFinalWorkItem) ||
    error("V5Final MPMC work item is not isbits")
which(
    HD_RELEASE_V5_FINAL_ARENA.enable_release_runtime!,
    Tuple{
        HD_RELEASE_V5_FINAL_ARENA.PaperTrainer,
        AbstractString,
    },
).file == Symbol(joinpath(
    @__DIR__,
    "PaperArenaReleaseAdapterV5Final.jl",
)) || error("public release registration bypasses V5Final")
which(
    HD_RELEASE_V5_FINAL_ARENA.paper_arena_update!,
    Tuple{HD_RELEASE_V5_FINAL_ARENA.PaperExecutorFinal},
).file == Symbol(joinpath(
    @__DIR__,
    "PaperArenaExecutorFinalBindings.jl",
)) || error("public final update bypasses final MPMC")

const HD_SWSNN_TWINPROP_RELEASE_V5_SEALED_V2_FINAL_PRODUCTION =
    HD_RELEASE_V5_FINAL_BASE
