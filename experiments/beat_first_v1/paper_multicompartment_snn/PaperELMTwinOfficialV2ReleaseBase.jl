module PaperELMTwinOfficialV2Release

using JLD2
using Zygote

include(joinpath(@__DIR__, "PaperELMTwinOfficialV2Final.jl"))
const Development = PaperELMTwinOfficialV2Final

export OfficialELMReleaseCandidate,
    OfficialELMReleaseMetrics,
    OfficialELMReleaseAttestation,
    VerifiedOfficialELMRelease,
    CANONICAL_MINIMUM_SPIKE_AUROC,
    CANONICAL_MAXIMUM_VOLTAGE_RMSE_MV,
    CANONICAL_MAXIMUM_NMDA_NORMALIZED_RMSE,
    CANONICAL_HELD_OUT_SPLIT,
    CANONICAL_PAPER_DURATION_MS,
    CANONICAL_PAPER_SAMPLE_DT_MS,
    CANONICAL_PAPER_TIME_STEPS,
    CANONICAL_MINIMUM_HELD_OUT_TRIALS,
    prepare_official_elm_release_candidate,
    assert_official_elm_release_candidate,
    attest_official_elm_release,
    assert_verified_official_elm_release,
    release_attestation_sha256,
    twin_forward,
    twin_step,
    save_verified_official_elm_release,
    load_verified_official_elm_release

const CANONICAL_MINIMUM_SPIKE_AUROC = 0.985
const CANONICAL_MAXIMUM_VOLTAGE_RMSE_MV = 1.0
const CANONICAL_MAXIMUM_NMDA_NORMALIZED_RMSE = 1.0
const CANONICAL_HELD_OUT_SPLIT = "held_out_test"
const CANONICAL_PAPER_DURATION_MS = 10_000
const CANONICAL_PAPER_SAMPLE_DT_MS = 1
const CANONICAL_PAPER_TIME_STEPS =
    CANONICAL_PAPER_DURATION_MS ÷ CANONICAL_PAPER_SAMPLE_DT_MS
const CANONICAL_MINIMUM_HELD_OUT_TRIALS = 2_000
const _RELEASE_SCHEMA =
    "hd_swsnn_twinprop.official_elm_v2.release_attestation.v1"
const _RELEASE_KIND = "VerifiedOfficialELMRelease"
const _RELEASE_FORMAT = 1

struct OfficialELMReleaseCandidate{F}
    frozen::F
    teacher_manifest_sha256::String
    teacher_contract_sha256::String
    candidate_sha256::String
end

struct OfficialELMReleaseMetrics
    voltage_rmse_mv::Float64
    voltage_correlation::Float64
    spike_auroc::Float64
    nmda_normalized_rmse_by_region::NTuple{4,Float64}
    nmda_raw_rmse_by_region::NTuple{4,Float64}
    nmda_correlation_by_region::NTuple{4,Float64}
    held_out_trials::Int
    time_steps::Int
    split::String
    duration_ms::Int
    sample_dt_ms::Int
    paper_scale::Bool
end

struct OfficialELMReleaseAttestation
    schema::String
    candidate_sha256::String
    base_artifact_sha256::String
    base_parameter_sha256::String
    source_metadata_sha256::String
    model_config_sha256::String
    teacher_manifest_sha256::String
    teacher_contract_sha256::String
    evaluator_sha256::String
    evaluation_result_sha256::String
    metrics::OfficialELMReleaseMetrics
    fixed_gate::NamedTuple
    passed::Bool
end

struct VerifiedOfficialELMRelease{C}
    candidate::C
    attestation::OfficialELMReleaseAttestation
    attestation_sha256::String
end

function _require_sha256(label::AbstractString, value)
    digest = lowercase(String(value))
    occursin(r"^[0-9a-f]{64}$", digest) ||
        throw(ArgumentError("$label must be a complete SHA-256"))
    return digest
end

function _required_metadata(metadata, name::Symbol)
    hasproperty(metadata, name) ||
        error("release candidate metadata lacks `$name`")
    return getproperty(metadata, name)
end

function _candidate_payload(frozen, manifest, contract)
    return (;
        schema="hd_swsnn_twinprop.official_elm_v2.release_candidate.v1",
        base_artifact_sha256=frozen.artifact_sha256,
        base_parameter_sha256=frozen.parameter_sha256,
        source_metadata_sha256=Development._digest_hex(
            Development.official_elm_source_metadata(),
        ),
        model_config_sha256=Development._digest_hex(
            frozen.model.config,
        ),
        teacher_manifest_sha256=manifest,
        teacher_contract_sha256=contract,
        held_out_split=CANONICAL_HELD_OUT_SPLIT,
        duration_ms=CANONICAL_PAPER_DURATION_MS,
        sample_dt_ms=CANONICAL_PAPER_SAMPLE_DT_MS,
        paper_scale=true,
    )
end

"""
Prepare a release candidate only when paper-scale provenance is already bound
inside the base artifact hash.

The base metadata must contain the exact teacher hashes, held-out split,
10,000 ms duration, 1 ms sample interval, and `paper_scale=true`.  Smoke and
short-duration artifacts cannot be promoted through this API.
"""
function prepare_official_elm_release_candidate(
    frozen::Development.FrozenOfficialELMTwin,
)
    Development.assert_frozen_official_elm_unchanged(frozen)
    metadata = frozen.metadata
    _required_metadata(metadata, :held_out_split) ==
        CANONICAL_HELD_OUT_SPLIT ||
        error("release candidate split is not held_out_test")
    Int(_required_metadata(metadata, :duration_ms)) ==
        CANONICAL_PAPER_DURATION_MS ||
        error("release candidate duration is not paper-scale 10,000 ms")
    Int(_required_metadata(metadata, :sample_dt_ms)) ==
        CANONICAL_PAPER_SAMPLE_DT_MS ||
        error("release candidate sample_dt_ms is not 1")
    _required_metadata(metadata, :paper_scale) === true ||
        error("release candidate is not paper_scale=true")
    manifest = _require_sha256(
        "official_teacher_manifest_sha256",
        _required_metadata(
            metadata,
            :official_teacher_manifest_sha256,
        ),
    )
    contract = _require_sha256(
        "teacher_contract_sha256",
        _required_metadata(metadata, :teacher_contract_sha256),
    )
    candidate_digest = Development._digest_hex(
        _candidate_payload(frozen, manifest, contract),
    )
    return OfficialELMReleaseCandidate(
        frozen,
        manifest,
        contract,
        candidate_digest,
    )
end

function assert_official_elm_release_candidate(
    candidate::OfficialELMReleaseCandidate,
)
    Development.assert_frozen_official_elm_unchanged(
        candidate.frozen,
    )
    metadata = candidate.frozen.metadata
    _required_metadata(metadata, :held_out_split) ==
        CANONICAL_HELD_OUT_SPLIT ||
        error("release split changed")
    Int(_required_metadata(metadata, :duration_ms)) ==
        CANONICAL_PAPER_DURATION_MS ||
        error("release duration changed")
    Int(_required_metadata(metadata, :sample_dt_ms)) ==
        CANONICAL_PAPER_SAMPLE_DT_MS ||
        error("release sample interval changed")
    _required_metadata(metadata, :paper_scale) === true ||
        error("release paper_scale changed")
    _require_sha256(
        "teacher manifest",
        _required_metadata(
            metadata,
            :official_teacher_manifest_sha256,
        ),
    ) == candidate.teacher_manifest_sha256 ||
        error("release teacher manifest changed")
    _require_sha256(
        "teacher contract",
        _required_metadata(metadata, :teacher_contract_sha256),
    ) == candidate.teacher_contract_sha256 ||
        error("release teacher contract changed")
    Development._digest_hex(_candidate_payload(
        candidate.frozen,
        candidate.teacher_manifest_sha256,
        candidate.teacher_contract_sha256,
    )) == candidate.candidate_sha256 ||
        error("release candidate digest changed")
    return true
end

function _region_tuple(values, label)
    length(values) == 4 ||
        throw(DimensionMismatch("$label must contain four regions"))
    result = ntuple(index -> Float64(values[index]), 4)
    all(isfinite, result) ||
        throw(ArgumentError("$label must be finite"))
    return result
end

function _release_metrics(;
    voltage_rmse_mv,
    voltage_correlation,
    spike_auroc,
    nmda_normalized_rmse_by_region,
    nmda_raw_rmse_by_region,
    nmda_correlation_by_region,
    held_out_trials,
    time_steps,
)
    voltage_rmse = Float64(voltage_rmse_mv)
    voltage_corr = Float64(voltage_correlation)
    auroc = Float64(spike_auroc)
    normalized_nmda = _region_tuple(
        nmda_normalized_rmse_by_region,
        "normalized NMDA RMSE",
    )
    raw_nmda =
        _region_tuple(nmda_raw_rmse_by_region, "raw NMDA RMSE")
    nmda_corr = _region_tuple(
        nmda_correlation_by_region,
        "NMDA correlation",
    )
    isfinite(voltage_rmse) && voltage_rmse >= 0 ||
        throw(ArgumentError("voltage RMSE must be finite/nonnegative"))
    isfinite(voltage_corr) && -1 <= voltage_corr <= 1 ||
        throw(ArgumentError("voltage correlation must be in [-1,1]"))
    isfinite(auroc) && 0 <= auroc <= 1 ||
        throw(ArgumentError("spike AUROC must be in [0,1]"))
    all(>=(0), normalized_nmda) ||
        throw(ArgumentError("normalized NMDA RMSE must be nonnegative"))
    all(>=(0), raw_nmda) ||
        throw(ArgumentError("raw NMDA RMSE must be nonnegative"))
    all(value -> -1 <= value <= 1, nmda_corr) ||
        throw(ArgumentError("NMDA correlations must be in [-1,1]"))
    Int(held_out_trials) >= CANONICAL_MINIMUM_HELD_OUT_TRIALS ||
        error("release requires at least 2,000 held-out trials")
    Int(time_steps) == CANONICAL_PAPER_TIME_STEPS ||
        error("release requires 10,000 one-ms output steps")
    return OfficialELMReleaseMetrics(
        voltage_rmse,
        voltage_corr,
        auroc,
        normalized_nmda,
        raw_nmda,
        nmda_corr,
        Int(held_out_trials),
        Int(time_steps),
        CANONICAL_HELD_OUT_SPLIT,
        CANONICAL_PAPER_DURATION_MS,
        CANONICAL_PAPER_SAMPLE_DT_MS,
        true,
    )
end

function _fixed_gate()
    return (;
        minimum_spike_auroc=CANONICAL_MINIMUM_SPIKE_AUROC,
        maximum_voltage_rmse_mv=
            CANONICAL_MAXIMUM_VOLTAGE_RMSE_MV,
        maximum_nmda_normalized_rmse_by_region=
            ntuple(
                _ -> CANONICAL_MAXIMUM_NMDA_NORMALIZED_RMSE,
                4,
            ),
        minimum_held_out_trials=
            CANONICAL_MINIMUM_HELD_OUT_TRIALS,
        required_time_steps=CANONICAL_PAPER_TIME_STEPS,
        required_split=CANONICAL_HELD_OUT_SPLIT,
        required_duration_ms=CANONICAL_PAPER_DURATION_MS,
        required_sample_dt_ms=CANONICAL_PAPER_SAMPLE_DT_MS,
        required_paper_scale=true,
        threshold_override_allowed=false,
    )
end

function _assert_fixed_gate(metrics::OfficialELMReleaseMetrics)
    metrics.spike_auroc >= CANONICAL_MINIMUM_SPIKE_AUROC ||
        error("release spike AUROC is below 0.985")
    metrics.voltage_rmse_mv <=
        CANONICAL_MAXIMUM_VOLTAGE_RMSE_MV ||
        error("release physical voltage RMSE exceeds 1.0 mV")
    all(
        value ->
            value <= CANONICAL_MAXIMUM_NMDA_NORMALIZED_RMSE,
        metrics.nmda_normalized_rmse_by_region,
    ) || error(
        "a regional normalized NMDA RMSE exceeds 1.0",
    )
    metrics.held_out_trials >=
        CANONICAL_MINIMUM_HELD_OUT_TRIALS ||
        error("release held-out trial count changed")
    metrics.time_steps == CANONICAL_PAPER_TIME_STEPS ||
        error("release held-out duration changed")
    metrics.split == CANONICAL_HELD_OUT_SPLIT ||
        error("release held-out split changed")
    metrics.duration_ms == CANONICAL_PAPER_DURATION_MS ||
        error("release paper duration changed")
    metrics.sample_dt_ms == CANONICAL_PAPER_SAMPLE_DT_MS ||
        error("release sample interval changed")
    metrics.paper_scale ||
        error("release metrics are not paper scale")
    return true
end

release_attestation_sha256(
    attestation::OfficialELMReleaseAttestation,
) = Development._digest_hex(attestation)

"""
Attest using immutable release gates.  No threshold argument exists.
"""
function attest_official_elm_release(
    candidate::OfficialELMReleaseCandidate;
    voltage_rmse_mv,
    voltage_correlation,
    spike_auroc,
    nmda_normalized_rmse_by_region,
    nmda_raw_rmse_by_region,
    nmda_correlation_by_region,
    held_out_trials,
    time_steps,
    evaluator_sha256,
    evaluation_result_sha256,
)
    assert_official_elm_release_candidate(candidate)
    metrics = _release_metrics(;
        voltage_rmse_mv,
        voltage_correlation,
        spike_auroc,
        nmda_normalized_rmse_by_region,
        nmda_raw_rmse_by_region,
        nmda_correlation_by_region,
        held_out_trials,
        time_steps,
    )
    _assert_fixed_gate(metrics)
    attestation = OfficialELMReleaseAttestation(
        _RELEASE_SCHEMA,
        candidate.candidate_sha256,
        candidate.frozen.artifact_sha256,
        candidate.frozen.parameter_sha256,
        Development._digest_hex(
            Development.official_elm_source_metadata(),
        ),
        Development._digest_hex(candidate.frozen.model.config),
        candidate.teacher_manifest_sha256,
        candidate.teacher_contract_sha256,
        _require_sha256("evaluator", evaluator_sha256),
        _require_sha256(
            "evaluation result",
            evaluation_result_sha256,
        ),
        metrics,
        _fixed_gate(),
        true,
    )
    return VerifiedOfficialELMRelease(
        candidate,
        attestation,
        release_attestation_sha256(attestation),
    )
end

function assert_verified_official_elm_release(
    verified::VerifiedOfficialELMRelease,
)
    assert_official_elm_release_candidate(verified.candidate)
    attestation = verified.attestation
    attestation.schema == _RELEASE_SCHEMA ||
        error("release attestation schema changed")
    attestation.passed ||
        error("release attestation is not passed")
    attestation.candidate_sha256 ==
        verified.candidate.candidate_sha256 ||
        error("release candidate digest differs")
    attestation.base_artifact_sha256 ==
        verified.candidate.frozen.artifact_sha256 ||
        error("release base artifact digest differs")
    attestation.base_parameter_sha256 ==
        verified.candidate.frozen.parameter_sha256 ||
        error("release parameter digest differs")
    attestation.source_metadata_sha256 ==
        Development._digest_hex(
            Development.official_elm_source_metadata(),
        ) ||
        error("release source metadata differs")
    attestation.model_config_sha256 ==
        Development._digest_hex(
            verified.candidate.frozen.model.config,
        ) ||
        error("release model config differs")
    attestation.teacher_manifest_sha256 ==
        verified.candidate.teacher_manifest_sha256 ||
        error("release teacher manifest differs")
    attestation.teacher_contract_sha256 ==
        verified.candidate.teacher_contract_sha256 ||
        error("release teacher contract differs")
    attestation.fixed_gate == _fixed_gate() ||
        error("release fixed gate changed")
    _assert_fixed_gate(attestation.metrics)
    release_attestation_sha256(attestation) ==
        verified.attestation_sha256 ||
        error("release attestation digest changed")
    return true
end

"""
Verified inference with integrity checks executed outside the Zygote tape.
"""
function twin_forward(
    verified::VerifiedOfficialELMRelease,
    input::AbstractArray{<:Real,3};
    normalized::Bool=false,
    initial_state=nothing,
)
    Zygote.ignore() do
        assert_verified_official_elm_release(verified)
    end
    frozen = verified.candidate.frozen
    output = Development.Core.official_elm_forward(
        frozen.model,
        frozen.parameters,
        input;
        initial_state,
    )
    return normalized ?
        output :
        Development.denormalize_official_elm_output(
            frozen.normalizer,
            output,
        )
end

function twin_step(
    verified::VerifiedOfficialELMRelease,
    state::Development.OfficialELMState,
    input::AbstractMatrix;
    normalized::Bool=false,
)
    Zygote.ignore() do
        assert_verified_official_elm_release(verified)
    end
    frozen = verified.candidate.frozen
    output = Development.Core.official_elm_step(
        frozen.model,
        frozen.parameters,
        state,
        input,
    )
    normalized && return output
    voltage =
        Development.soma_voltage_from_coordinate(output.voltage)
    nmda =
        output.nmda .* frozen.normalizer.nmda_scale .+
        frozen.normalizer.nmda_mean
    return merge(
        output,
        (;
            voltage_coordinate=output.voltage,
            voltage,
            nmda,
        ),
    )
end

function save_verified_official_elm_release(
    path::AbstractString,
    verified::VerifiedOfficialELMRelease,
)
    assert_verified_official_elm_release(verified)
    parent = dirname(abspath(path))
    isdir(parent) || mkpath(parent)
    jldsave(
        path;
        artifact_kind=_RELEASE_KIND,
        format_version=_RELEASE_FORMAT,
        verified,
    )
    return abspath(path)
end

function load_verified_official_elm_release(path::AbstractString)
    isfile(path) ||
        throw(ArgumentError("verified official ELM release not found: $path"))
    payload = JLD2.load(path)
    get(payload, "artifact_kind", nothing) == _RELEASE_KIND ||
        error("not a verified official ELM release")
    get(payload, "format_version", nothing) == _RELEASE_FORMAT ||
        error("unsupported verified official ELM release version")
    verified = payload["verified"]
    verified isa VerifiedOfficialELMRelease ||
        error("verified official ELM release has the wrong type")
    assert_verified_official_elm_release(verified)
    return verified
end

end # module PaperELMTwinOfficialV2Release
