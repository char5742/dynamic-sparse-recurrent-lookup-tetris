module PaperELMTwinOfficialV2IntegrityOverlay

# This module is deliberately additive.  Include PaperELMTwinOfficialV2.jl
# before this file so the canonical model remains owned by its implementation
# module while the promotion/verification trust boundary can evolve
# independently.

using JLD2
using SHA

import ..PaperELMTwinOfficialV2

export OfficialELMVerificationPolicy,
    OfficialELMHeldOutMetrics,
    OfficialELMVerificationAttestation,
    VerifiedOfficialELMTwinEnvelope,
    attest_verified_official_elm,
    assert_verified_official_elm,
    save_verified_official_elm_envelope,
    load_verified_official_elm_envelope

const _ELM = PaperELMTwinOfficialV2
const _ATTESTATION_SCHEMA =
    "hd_swsnn_twinprop.official_elm_verification_attestation.v1"
const _ENVELOPE_KIND = "VerifiedOfficialELMTwinEnvelope"
const _ENVELOPE_VERSION = 1

"""
Thresholds that an independently recomputed held-out result must satisfy.

There are intentionally no implicit defaults: promotion code must state the
quality gate it is enforcing, and artifact loading must supply the same policy
to prevent a serialized artifact from weakening its own gate.
"""
struct OfficialELMVerificationPolicy
    maximum_voltage_rmse::Float64
    minimum_spike_auroc::Float64
    maximum_nmda_rmse::Float64
end

function OfficialELMVerificationPolicy(;
    maximum_voltage_rmse::Real,
    minimum_spike_auroc::Real,
    maximum_nmda_rmse::Real,
)
    voltage = Float64(maximum_voltage_rmse)
    auroc = Float64(minimum_spike_auroc)
    nmda = Float64(maximum_nmda_rmse)
    isfinite(voltage) && voltage >= 0 ||
        throw(ArgumentError("maximum_voltage_rmse must be finite and nonnegative"))
    isfinite(auroc) && 0 <= auroc <= 1 ||
        throw(ArgumentError("minimum_spike_auroc must be in [0,1]"))
    isfinite(nmda) && nmda >= 0 ||
        throw(ArgumentError("maximum_nmda_rmse must be finite and nonnegative"))
    return OfficialELMVerificationPolicy(voltage, auroc, nmda)
end

"""
Metrics recomputed on the sealed held-out split.

`evaluation_result_sha256` in the attestation must identify the immutable
predictions/targets/metrics evidence from which these values were recomputed.
"""
struct OfficialELMHeldOutMetrics
    voltage_rmse::Float64
    spike_auroc::Float64
    nmda_rmse::Float64
    sample_count::Int
    time_steps::Int
    split::String
end

struct OfficialELMVerificationAttestation
    schema::String
    base_artifact_sha256::String
    base_parameter_sha256::String
    candidate_metadata_sha256::String
    metrics::OfficialELMHeldOutMetrics
    policy::OfficialELMVerificationPolicy
    dataset_sha256::String
    evaluator_sha256::String
    evaluation_result_sha256::String
    verification_passed::Bool
end

struct VerifiedOfficialELMTwinEnvelope{F}
    frozen::F
    attestation::OfficialELMVerificationAttestation
    attestation_sha256::String
end

function _digest_token!(context, tag::AbstractString, payload)
    bytes = payload isa AbstractString ? codeunits(payload) : payload
    SHA.update!(context, codeunits(tag))
    SHA.update!(context, UInt8[0x1f])
    SHA.update!(context, codeunits(string(length(bytes))))
    SHA.update!(context, UInt8[0x1e])
    SHA.update!(context, bytes)
    SHA.update!(context, UInt8[0x1d])
    return context
end

function _update_canonical_digest!(context, value)
    if value isa NamedTuple
        _digest_token!(
            context,
            "named-tuple-fields",
            join(String.(keys(value)), ","),
        )
        for (name, child) in pairs(value)
            _digest_token!(context, "field", String(name))
            _update_canonical_digest!(context, child)
        end
    elseif value isa Tuple
        _digest_token!(context, "tuple-length", string(length(value)))
        for child in value
            _update_canonical_digest!(context, child)
        end
    elseif value isa AbstractArray
        _digest_token!(context, "array-eltype", string(eltype(value)))
        _digest_token!(context, "array-size", join(size(value), ","))
        for child in value
            _update_canonical_digest!(context, child)
        end
    elseif value isa AbstractString
        _digest_token!(context, "string", value)
    elseif value isa Symbol
        _digest_token!(context, "symbol", String(value))
    elseif value isa Number || value isa Char
        _digest_token!(context, string(typeof(value)), repr(value))
    elseif value === nothing
        _digest_token!(context, "nothing", "")
    elseif value isa OfficialELMVerificationPolicy ||
           value isa OfficialELMHeldOutMetrics ||
           value isa OfficialELMVerificationAttestation
        _digest_token!(context, "struct-type", string(typeof(value)))
        for field in fieldnames(typeof(value))
            _digest_token!(context, "field", String(field))
            _update_canonical_digest!(context, getfield(value, field))
        end
    else
        throw(ArgumentError(
            "unsupported value in canonical verification digest: " *
            string(typeof(value)),
        ))
    end
    return context
end

function _canonical_sha256(value)
    context = SHA.SHA2_256_CTX()
    _update_canonical_digest!(context, value)
    return bytes2hex(SHA.digest!(context))
end

function _require_sha256(value, label::AbstractString)
    digest = lowercase(String(value))
    occursin(r"^[0-9a-f]{64}$", digest) ||
        throw(ArgumentError("$label must be a 64-character SHA-256 digest"))
    return digest
end

function _assert_unverified_candidate(
    frozen::_ELM.FrozenOfficialELMTwin,
)
    _ELM.assert_frozen_official_elm_unchanged(frozen)
    metadata = frozen.metadata
    metadata isa NamedTuple ||
        error("official ELM candidate metadata must be a NamedTuple")
    hasproperty(metadata, :verification_passed) &&
        metadata.verification_passed === false ||
        error("candidate metadata must remain explicitly unverified")

    for field in (
        :held_out_voltage_rmse,
        :held_out_spike_auroc,
        :held_out_nmda_rmse,
        :verification_policy,
        :verification_attestation,
        :verification_evidence_sha256,
        :evaluation_result_sha256,
    )
        hasproperty(metadata, field) &&
            error("candidate metadata contains reserved gate field $field")
    end

    hasproperty(metadata, :canonical_official_routing) &&
        metadata.canonical_official_routing === true ||
        error("candidate does not certify canonical official routing")
    hasproperty(metadata, :frozen) && metadata.frozen === true ||
        error("candidate metadata does not certify a frozen model")
    hasproperty(metadata, :unpublished_twinprop_checkpoint_identity_claimed) &&
        metadata.unpublished_twinprop_checkpoint_identity_claimed === false ||
        error("candidate makes an unsupported checkpoint-identity claim")
    hasproperty(metadata, :source) &&
        metadata.source == _ELM.official_elm_source_metadata() ||
        error("candidate official-source lineage differs")
    hasproperty(metadata, :input_dim) &&
        metadata.input_dim == _ELM.OFFICIAL_ELM_INPUT_DIM ||
        error("candidate input dimension differs from official ELM")
    hasproperty(metadata, :branches) &&
        metadata.branches == _ELM.OFFICIAL_ELM_BRANCHES ||
        error("candidate branch count differs from official ELM")
    hasproperty(metadata, :synapses_per_branch) &&
        metadata.synapses_per_branch ==
        _ELM.OFFICIAL_ELM_SYNAPSES_PER_BRANCH ||
        error("candidate synapses-per-branch differs from official ELM")
    hasproperty(metadata, :parameter_sha256) &&
        String(metadata.parameter_sha256) == frozen.parameter_sha256 ||
        error("candidate metadata parameter digest differs")
    hasproperty(metadata, :artifact_sha256) &&
        String(metadata.artifact_sha256) == frozen.artifact_sha256 ||
        error("candidate metadata artifact digest differs")
    return true
end

function _held_out_metrics(;
    voltage_rmse::Real,
    spike_auroc::Real,
    nmda_rmse::Real,
    sample_count::Integer,
    time_steps::Integer,
    split::AbstractString,
)
    voltage = Float64(voltage_rmse)
    auroc = Float64(spike_auroc)
    nmda = Float64(nmda_rmse)
    isfinite(voltage) && voltage >= 0 ||
        throw(ArgumentError("held-out voltage RMSE must be finite and nonnegative"))
    isfinite(auroc) && 0 <= auroc <= 1 ||
        throw(ArgumentError("held-out spike AUROC must be in [0,1]"))
    isfinite(nmda) && nmda >= 0 ||
        throw(ArgumentError("held-out NMDA RMSE must be finite and nonnegative"))
    sample_count >= 1 ||
        throw(ArgumentError("held-out sample_count must be positive"))
    time_steps >= 1 ||
        throw(ArgumentError("held-out time_steps must be positive"))
    String(split) == "held_out_test" ||
        throw(ArgumentError("verification requires split=\"held_out_test\""))
    return OfficialELMHeldOutMetrics(
        voltage,
        auroc,
        nmda,
        Int(sample_count),
        Int(time_steps),
        String(split),
    )
end

function _assert_gate_pass(
    metrics::OfficialELMHeldOutMetrics,
    policy::OfficialELMVerificationPolicy,
)
    metrics.voltage_rmse <= policy.maximum_voltage_rmse ||
        error("held-out voltage RMSE fails verification policy")
    metrics.spike_auroc >= policy.minimum_spike_auroc ||
        error("held-out spike AUROC fails verification policy")
    metrics.nmda_rmse <= policy.maximum_nmda_rmse ||
        error("held-out NMDA RMSE fails verification policy")
    return true
end

function _same_policy(
    left::OfficialELMVerificationPolicy,
    right::OfficialELMVerificationPolicy,
)
    return left.maximum_voltage_rmse == right.maximum_voltage_rmse &&
           left.minimum_spike_auroc == right.minimum_spike_auroc &&
           left.maximum_nmda_rmse == right.maximum_nmda_rmse
end

"""
Promote a frozen, explicitly unverified candidate after an external evaluator
has recomputed held-out metrics.

Gate fields are not accepted through a generic metadata merge.  The returned
attestation binds the base artifact, its otherwise-unhashed candidate
metadata, the explicit policy, the recomputed metrics, and dataset/evaluator/
result evidence digests.
"""
function attest_verified_official_elm(
    frozen::_ELM.FrozenOfficialELMTwin;
    voltage_rmse::Real,
    spike_auroc::Real,
    nmda_rmse::Real,
    sample_count::Integer,
    time_steps::Integer,
    policy::OfficialELMVerificationPolicy,
    dataset_sha256,
    evaluator_sha256,
    evaluation_result_sha256,
    split::AbstractString="held_out_test",
)
    _assert_unverified_candidate(frozen)
    metrics = _held_out_metrics(;
        voltage_rmse,
        spike_auroc,
        nmda_rmse,
        sample_count,
        time_steps,
        split,
    )
    _assert_gate_pass(metrics, policy)
    attestation = OfficialELMVerificationAttestation(
        _ATTESTATION_SCHEMA,
        _require_sha256(frozen.artifact_sha256, "base artifact"),
        _require_sha256(frozen.parameter_sha256, "base parameters"),
        _canonical_sha256(frozen.metadata),
        metrics,
        policy,
        _require_sha256(dataset_sha256, "dataset"),
        _require_sha256(evaluator_sha256, "evaluator"),
        _require_sha256(evaluation_result_sha256, "evaluation result"),
        true,
    )
    return VerifiedOfficialELMTwinEnvelope(
        frozen,
        attestation,
        _canonical_sha256(attestation),
    )
end

"""
Verify an attested artifact against caller-owned policy and provenance.

Requiring the expected policy and dataset/evaluator digests prevents the
serialized envelope from choosing a weaker gate or a different evidence
lineage for itself.
"""
function assert_verified_official_elm(
    envelope::VerifiedOfficialELMTwinEnvelope;
    policy::OfficialELMVerificationPolicy,
    dataset_sha256,
    evaluator_sha256,
)
    _assert_unverified_candidate(envelope.frozen)
    attestation = envelope.attestation
    attestation.schema == _ATTESTATION_SCHEMA ||
        error("official ELM verification-attestation schema differs")
    attestation.verification_passed === true ||
        error("official ELM verification attestation is not passed")
    attestation.base_artifact_sha256 ==
        envelope.frozen.artifact_sha256 ||
        error("attestation/base artifact digest differs")
    attestation.base_parameter_sha256 ==
        envelope.frozen.parameter_sha256 ||
        error("attestation/base parameter digest differs")
    attestation.candidate_metadata_sha256 ==
        _canonical_sha256(envelope.frozen.metadata) ||
        error("candidate metadata changed after attestation")
    _same_policy(attestation.policy, policy) ||
        error("serialized verification policy differs from required policy")
    attestation.dataset_sha256 ==
        _require_sha256(dataset_sha256, "dataset") ||
        error("attested dataset digest differs")
    attestation.evaluator_sha256 ==
        _require_sha256(evaluator_sha256, "evaluator") ||
        error("attested evaluator digest differs")
    _require_sha256(
        attestation.evaluation_result_sha256,
        "evaluation result",
    )
    _assert_gate_pass(attestation.metrics, policy)
    _canonical_sha256(attestation) == envelope.attestation_sha256 ||
        error("official ELM verification attestation digest changed")
    return true
end

function save_verified_official_elm_envelope(
    path::AbstractString,
    envelope::VerifiedOfficialELMTwinEnvelope;
    policy::OfficialELMVerificationPolicy,
    dataset_sha256,
    evaluator_sha256,
)
    assert_verified_official_elm(
        envelope;
        policy,
        dataset_sha256,
        evaluator_sha256,
    )
    parent = dirname(abspath(path))
    isdir(parent) || mkpath(parent)
    jldsave(
        path;
        artifact_kind=_ENVELOPE_KIND,
        format_version=_ENVELOPE_VERSION,
        envelope,
    )
    return abspath(path)
end

function load_verified_official_elm_envelope(
    path::AbstractString;
    policy::OfficialELMVerificationPolicy,
    dataset_sha256,
    evaluator_sha256,
)
    isfile(path) ||
        throw(ArgumentError("verified official ELM envelope not found: $path"))
    payload = JLD2.load(path)
    get(payload, "artifact_kind", nothing) == _ENVELOPE_KIND ||
        error("not a verified official ELM envelope")
    get(payload, "format_version", nothing) == _ENVELOPE_VERSION ||
        error("unsupported verified official ELM envelope version")
    envelope = payload["envelope"]
    envelope isa VerifiedOfficialELMTwinEnvelope ||
        error("verified official ELM envelope has the wrong type")
    assert_verified_official_elm(
        envelope;
        policy,
        dataset_sha256,
        evaluator_sha256,
    )
    return envelope
end

end # module PaperELMTwinOfficialV2IntegrityOverlay
