# Canonical additive loader for the official 1278-input 11-state core.

if !isdefined(Main, :OfficialElevenStateDistillationCore)
    include(joinpath(
        @__DIR__,
        "OfficialElevenStateDistillationCore.jl",
    ))
end

const OFFICIAL_ELEVEN_STATE_DISTILLATION_CORE =
    Main.OfficialElevenStateDistillationCore

# `DistilledParameters` is a struct, whereas the trainable parameter tree is a
# NamedTuple.  Install the explicit finite-array audit used at freeze time.
@eval Main.OfficialElevenStateDistillationCore begin
    function all_finite(parameters::Cell.DistilledParameters)
        return all(array -> all(isfinite, array), (
            parameters.transition_decay,
            parameters.recurrent_weight,
            parameters.input_weight,
            parameters.transition_bias,
            parameters.readout_weight,
            parameters.readout_bias,
            parameters.target_mean,
            parameters.target_scale,
            parameters.initial_state,
            parameters.compartment_projection,
            parameters.region_projection,
        ))
    end
end
