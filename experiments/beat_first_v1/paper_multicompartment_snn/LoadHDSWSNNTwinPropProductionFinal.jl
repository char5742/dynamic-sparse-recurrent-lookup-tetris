# Canonical HD-SWSNN-TwinProp release loader.

include(joinpath(@__DIR__, "LoadHDSWSNNTwinPropProduction.jl"))

if !isdefined(
    Main.HDSWSNNTwinPropProduction,
    :_validate_normalizer!,
)
    Base.include(
        Main.HDSWSNNTwinPropProduction,
        joinpath(
            @__DIR__,
            "HDSWSNNTwinPropProductionValidationFinal.jl",
        ),
    )
end

const HD_SWSNN_TWINPROP_RELEASE =
    Main.HDSWSNNTwinPropProduction
