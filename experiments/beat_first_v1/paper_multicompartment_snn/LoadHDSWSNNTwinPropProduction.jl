# Sole additive loader for the strict HD-SWSNN-TwinProp production identity.
#
# It loads the strict lineage/freeze gate, then installs the canonical
# optimizer/routing overrides, replay cache, and isbits-MPMC executor into the
# exact same PaperArenaTrainingFinalProduction module.

if !isdefined(Main, :HDSWSNNTwinPropProduction)
    include(joinpath(@__DIR__, "HDSWSNNTwinPropProduction.jl"))
end

const HD_SWSNN_TWINPROP_PRODUCTION =
    Main.HDSWSNNTwinPropProduction
const HD_SWSNN_TWINPROP_PRODUCTION_ARENA =
    HD_SWSNN_TWINPROP_PRODUCTION.Training

if !isdefined(
    HD_SWSNN_TWINPROP_PRODUCTION_ARENA,
    :PaperExecutorFinal,
)
    Core.eval(
        HD_SWSNN_TWINPROP_PRODUCTION_ARENA,
        quote
            @inline sigmoid(value::Float32) =
                ifelse(
                    value >= 0.0f0,
                    inv(1.0f0 + exp(-value)),
                    exp(value) / (1.0f0 + exp(value)),
                )
        end,
    )
    for file in (
        "PaperArenaCanonicalOverrides.jl",
        "PaperArenaReplayHotfix.jl",
        "PaperArenaExecutorFinal.jl",
    )
        Base.include(
            HD_SWSNN_TWINPROP_PRODUCTION_ARENA,
            joinpath(@__DIR__, file),
        )
    end
end

HD_SWSNN_TWINPROP_PRODUCTION_ARENA.Distilled ===
    Main.DistilledElevenStateCellFinal ||
    error("production arena has a split Final-cell type identity")
isbitstype(
    HD_SWSNN_TWINPROP_PRODUCTION_ARENA.PaperFinalWorkItem,
) || error("production executor work item is not isbits")
