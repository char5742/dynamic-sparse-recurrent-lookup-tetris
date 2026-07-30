module PaperELMTwinOfficialV2ReleaseExecution

# Gradient-safe execution for the canonical, independently recomputed
# ReleaseAttestation artifact.
#
# Integrity, held-out predictions, fixed fidelity gates, scale evidence, and
# hashes are recomputed before this module creates an execution context.  The
# numerical ELM kernel then runs without SHA/string work inside the AD trace.

using Zygote

const _PARENT = parentmodule(@__MODULE__)
if !isdefined(
    _PARENT,
    :PaperELMTwinOfficialV2ReleaseAttestation,
)
    Base.include(
        _PARENT,
        joinpath(
            @__DIR__,
            "PaperELMTwinOfficialV2ReleaseAttestation.jl",
        ),
    )
end

const Release =
    getfield(
        _PARENT,
        :PaperELMTwinOfficialV2ReleaseAttestation,
    )
const ELM = Release.PaperELMTwinOfficialV2

export VerifiedOfficialELMExecution,
    assert_verified_release_unchanged!,
    load_verified_official_elm_execution,
    verified_release_bundle,
    verified_release_frozen,
    twin_forward_after_verified,
    twin_step_after_verified

struct _VerifiedExecutionToken end
const _VERIFIED_EXECUTION_TOKEN = _VerifiedExecutionToken()

"""
An execution capability created only after ReleaseAttestation recomputation.

`bundle` keeps the exact canonical
`PaperELMTwinOfficialV2ReleaseAttestation.AttestedOfficialELMRelease` type.
The wrapper stores the evidence needed to re-run the same verification at
checkpoint and run-end boundaries.
"""
struct VerifiedOfficialELMExecution{B,H,I,S}
    bundle::B
    heldout::H
    identity::I
    splits::S
    require_production::Bool
    development_scale_chain::Bool
    attestation_sha256::String

    function VerifiedOfficialELMExecution(
        ::_VerifiedExecutionToken,
        bundle::B,
        heldout::H,
        identity::I,
        splits::S,
        require_production::Bool,
        development_scale_chain::Bool,
    ) where {B,H,I,S}
        bundle isa Release.AttestedOfficialELMRelease ||
            error("execution bundle is not the canonical release type")
        return new{B,H,I,S}(
            bundle,
            heldout,
            identity,
            splits,
            require_production,
            development_scale_chain,
            bundle.attestation.attestation_sha256,
        )
    end
end

@inline verified_release_bundle(
    execution::VerifiedOfficialELMExecution,
) = execution.bundle

@inline verified_release_frozen(
    execution::VerifiedOfficialELMExecution,
) = execution.bundle.frozen

function _assert_scale_mode!(
    bundle::Release.AttestedOfficialELMRelease;
    require_production::Bool,
    development_scale_chain::Bool,
)
    outcome = bundle.attestation.payload.outcome
    if require_production
        development_scale_chain === false ||
            error(
                "production execution cannot set " *
                "development_scale_chain=true",
            )
        outcome.paper_scale === true ||
            error("production execution requires paper_scale=true")
        outcome.promotable_production === true ||
            error("production execution is not promotable")
        outcome.development_scale === false ||
            error("production artifact is marked development-scale")
    else
        development_scale_chain === true ||
            error(
                "non-production execution requires the explicit " *
                "development_scale_chain=true flag",
            )
        outcome.paper_scale === false ||
            error("development execution requires paper_scale=false")
        outcome.promotable_production === false ||
            error("development artifact cannot be production-promotable")
        outcome.development_scale === true ||
            error("artifact is not marked development-scale")
    end
    outcome.gate_passed === true ||
        error("official ELM fixed fidelity gate did not pass")
    return bundle
end

"""
Load, independently recompute the held-out fidelity gate, and only then
create a differentiable execution capability.

The default is the unchanged paper-scale production gate.  A rich64
development artifact requires both `require_production=false` and the explicit
`development_scale_chain=true` flag.
"""
function load_verified_official_elm_execution(
    path::AbstractString,
    heldout::Release.OfficialELMHeldoutSet,
    identity::Release.TeacherReleaseIdentity,
    splits::Release.OfficialReleaseSplits;
    require_production::Bool=true,
    development_scale_chain::Bool=false,
)
    bundle = Release.load_verified_official_elm_release(
        path,
        heldout,
        identity,
        splits;
        require_production,
    )
    _assert_scale_mode!(
        bundle;
        require_production,
        development_scale_chain,
    )
    return VerifiedOfficialELMExecution(
        _VERIFIED_EXECUTION_TOKEN,
        bundle,
        heldout,
        identity,
        splits,
        require_production,
        development_scale_chain,
    )
end

"""
Recompute hashes, raw held-out predictions, and fixed gates at a lifecycle
boundary.  This function must not be placed inside an AD trace.
"""
function assert_verified_release_unchanged!(
    execution::VerifiedOfficialELMExecution,
)
    verified = Release.verify_official_elm_release(
        execution.bundle,
        execution.heldout,
        execution.identity,
        execution.splits;
        require_gate=true,
        require_production=execution.require_production,
    )
    verified === execution.bundle ||
        error("release verifier returned another bundle")
    verified.attestation.attestation_sha256 ==
        execution.attestation_sha256 ||
        error("release attestation digest changed")
    _assert_scale_mode!(
        verified;
        require_production=execution.require_production,
        development_scale_chain=
            execution.development_scale_chain,
    )
    return true
end

"""
Differentiable raw official ELM forward after verified preflight.

This is the exact numerical model owned by the canonical ReleaseAttestation
bundle.  No artifact unwrapping, SHA-256 computation, string conversion, or
stored pass-flag trust occurs inside the numerical trace.
"""
function twin_forward_after_verified(
    execution::VerifiedOfficialELMExecution,
    input::AbstractArray{<:Real,3};
    normalized::Bool=false,
    initial_state=nothing,
)
    frozen = verified_release_frozen(execution)
    size(input, 1) == ELM.OFFICIAL_ELM_INPUT_DIM ||
        throw(DimensionMismatch(
            "official ELM input must have 1278 rows",
        ))
    normalized_input = normalized ?
        input :
        ELM.normalize_official_elm_input(
            frozen.normalizer,
            input,
        )
    output = ELM.official_elm_forward(
        frozen.model,
        frozen.parameters,
        normalized_input;
        initial_state,
    )
    return normalized ?
        output :
        ELM.denormalize_official_elm_output(
            frozen.normalizer,
            output,
        )
end

function twin_step_after_verified(
    execution::VerifiedOfficialELMExecution,
    state::ELM.OfficialELMState,
    input::AbstractMatrix;
    normalized::Bool=false,
)
    frozen = verified_release_frozen(execution)
    size(input, 1) == ELM.OFFICIAL_ELM_INPUT_DIM ||
        throw(DimensionMismatch(
            "official ELM input must have 1278 rows",
        ))
    normalized_input = normalized ?
        input :
        (
            input .- frozen.normalizer.input_mean
        ) ./ frozen.normalizer.input_scale
    output = ELM.official_elm_step(
        frozen.model,
        frozen.parameters,
        state,
        normalized_input,
    )
    normalized && return output
    voltage =
        output.voltage .* frozen.normalizer.voltage_scale .+
        frozen.normalizer.voltage_mean
    nmda =
        output.nmda .* frozen.normalizer.nmda_scale .+
        frozen.normalizer.nmda_mean
    return merge(output, (; voltage, nmda))
end

end # module PaperELMTwinOfficialV2ReleaseExecution
