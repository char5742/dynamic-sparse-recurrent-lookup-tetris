# Differentiable execution boundary for a preflight-verified official ELM.
#
# Include this file into `PaperELMTwinOfficialV2Final` after the strict
# format-v3 implementation.  SHA-256 validation remains mandatory at artifact
# load/preflight/checkpoint/run-end boundaries, but is deliberately absent from
# the numerical forward kernel.  Hashing strings and byte buffers inside a
# Zygote trace is both unnecessary and non-differentiable.

export preflight_verified_official_elm!,
    twin_forward_after_preflight,
    twin_step_after_preflight

preflight_verified_official_elm!(
    verified::VerifiedOfficialELMTwin,
) = assert_verified_official_elm(verified)

"""
Run a strictly preflight-verified official ELM without hashing inside AD.

Call `preflight_verified_official_elm!` once at the immutable artifact
boundary before using this kernel.  The verified parameter arrays remain
frozen; gradients may flow only with respect to the supplied synaptic input.
"""
function twin_forward_after_preflight(
    verified::VerifiedOfficialELMTwin,
    input::AbstractArray{<:Real,3};
    normalized::Bool=false,
    initial_state=nothing,
)
    size(input, 1) == OFFICIAL_ELM_INPUT_DIM ||
        throw(DimensionMismatch(
            "official ELM input must have 1278 rows",
        ))
    normalized_input =
        normalize_official_elm_input(verified.normalizer, input)
    output = Core.official_elm_forward(
        verified.model,
        verified.parameters,
        normalized_input;
        initial_state,
    )
    return normalized ?
        output :
        denormalize_official_elm_output(
            verified.normalizer,
            output,
        )
end

function twin_step_after_preflight(
    verified::VerifiedOfficialELMTwin,
    state::OfficialELMState,
    input::AbstractMatrix;
    normalized::Bool=false,
)
    size(input, 1) == OFFICIAL_ELM_INPUT_DIM ||
        throw(DimensionMismatch(
            "official ELM input must have 1278 rows",
        ))
    output = Core.official_elm_step(
        verified.model,
        verified.parameters,
        state,
        input,
    )
    normalized && return output
    voltage = soma_voltage_from_coordinate(output.voltage)
    nmda =
        output.nmda .* verified.normalizer.nmda_scale .+
        verified.normalizer.nmda_mean
    return merge(
        output,
        (;
            voltage_coordinate=output.voltage,
            voltage,
            nmda,
        ),
    )
end

# Preserve the public paper API while moving integrity work to the explicit
# preflight boundary.  Loading a format-v3 verified artifact already performs
# this preflight; long-lived users additionally re-run it at checkpoints and
# run end.
function twin_forward(
    verified::VerifiedOfficialELMTwin,
    input::AbstractArray{<:Real,3};
    kwargs...,
)
    return twin_forward_after_preflight(
        verified,
        input;
        kwargs...,
    )
end

function twin_step(
    verified::VerifiedOfficialELMTwin,
    state::OfficialELMState,
    input::AbstractMatrix;
    kwargs...,
)
    return twin_step_after_preflight(
        verified,
        state,
        input;
        kwargs...,
    )
end
