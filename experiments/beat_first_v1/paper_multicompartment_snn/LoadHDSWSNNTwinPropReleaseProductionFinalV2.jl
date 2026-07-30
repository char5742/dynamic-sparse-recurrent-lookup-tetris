# Sole canonical loader for the production release-v2 training chain.
#
# Every component is installed in the exact
# `Main.PaperArenaTrainingFinalProduction` type universe owned by
# `Main.HDSWSNNTwinPropProduction.Training`.
include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropReleaseProduction.jl",
))

const HD_RELEASE_V2 =
    Main.HDSWSNNTwinPropProduction
const HD_RELEASE_V2_ARENA = HD_RELEASE_V2.Training

if !isdefined(
    HD_RELEASE_V2_ARENA,
    :DistilledElevenStateCellFinal,
)
    Core.eval(
        HD_RELEASE_V2_ARENA,
        :(const DistilledElevenStateCellFinal =
            Main.DistilledElevenStateCellFinal),
    )
end

if !isdefined(HD_RELEASE_V2_ARENA, :PaperReleaseAux)
    for filename in (
        "PaperArenaReleaseAdapter.jl",
        "PaperArenaReleaseLearning.jl",
        "PaperArenaReleaseStructure.jl",
    )
        Base.include(
            HD_RELEASE_V2_ARENA,
            joinpath(@__DIR__, filename),
        )
    end
end

if !isdefined(
    HD_RELEASE_V2_ARENA,
    :paper_arena_update_hotfinal!,
)
    Base.include(
        HD_RELEASE_V2_ARENA,
        joinpath(
            @__DIR__,
            "PaperArenaExecutorFinalHotfix.jl",
        ),
    )
end
if !hasmethod(
    HD_RELEASE_V2_ARENA.paper_arena_update!,
    Tuple{HD_RELEASE_V2_ARENA.PaperExecutorFinal},
)
    Base.include(
        HD_RELEASE_V2_ARENA,
        joinpath(
            @__DIR__,
            "PaperArenaExecutorFinalBindings.jl",
        ),
    )
end
Base.include(
    HD_RELEASE_V2_ARENA,
    joinpath(
        @__DIR__,
        "PaperArenaExecutorFinalReleaseV2.jl",
    ),
)

Core.eval(
    HD_RELEASE_V2_ARENA,
    quote
        export PaperExecutorFinal,
            enable_release_runtime!,
            paper_checkpoint_integrity!,
            paper_end_run_integrity!,
            paper_preflight_integrity!,
            paper_arena_update_serial_final!
    end,
)

Base.include(
    HD_RELEASE_V2,
    joinpath(
        @__DIR__,
        "HDSWSNNTwinPropReleaseBuilderV2.jl",
    ),
)

HD_RELEASE_V2_ARENA ===
    Main.PaperArenaTrainingFinalProduction ||
    error("release-v2 arena is not FinalProduction")
HD_RELEASE_V2_ARENA.Distilled ===
    Main.DistilledElevenStateCellFinal ||
    error("release-v2 arena has split Final cell identity")
HD_RELEASE_V2_ARENA.ReleaseCell.Final ===
    Main.DistilledElevenStateCellFinal ||
    error("release-v2 runtime has split Final cell identity")
isdefined(HD_RELEASE_V2_ARENA, :PaperExecutorFinal) ||
    error("release-v2 arena lacks PaperExecutorFinal")
isbitstype(HD_RELEASE_V2_ARENA.PaperFinalWorkItem) ||
    error("release-v2 final work item is not isbits")

const HD_SWSNN_TWINPROP_RELEASE_FINAL_V2 = HD_RELEASE_V2
