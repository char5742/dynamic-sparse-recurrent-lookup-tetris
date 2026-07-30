# Final release loader: strict official-teacher contract plus the canonical
# Final arena/executor identity.

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropProductionFinal.jl",
))

if !isdefined(
    Main.OfficialNeuronTeacherMetadataProduction,
    :OfficialTeacherMetadataFinal,
)
    Base.include(
        Main.OfficialNeuronTeacherMetadataProduction,
        joinpath(
            @__DIR__,
            "OfficialNeuronTeacherMetadataProductionFinal.jl",
        ),
    )
end

const HD_SWSNN_TWINPROP_CANONICAL_RELEASE =
    Main.HDSWSNNTwinPropProduction
