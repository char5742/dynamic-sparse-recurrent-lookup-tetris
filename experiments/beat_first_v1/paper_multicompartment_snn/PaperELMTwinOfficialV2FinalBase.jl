module PaperELMTwinOfficialV2Final

using JLD2
using SHA
using Statistics

include(joinpath(@__DIR__, "PaperELMTwinOfficialV2.jl"))
const Core = PaperELMTwinOfficialV2

export OfficialELMConfig,
    OfficialELMState,
    OfficialPaperELMTwin,
    OfficialELMNormalizer,
    FrozenOfficialELMTwin,
    OfficialHeldOutAttestation,
    VerifiedOfficialELMTwin,
    OFFICIAL_ELM_INPUT_DIM,
    OFFICIAL_DENDRITIC_LOCATIONS,
    OFFICIAL_ELM_BRANCHES,
    OFFICIAL_ELM_SYNAPSES_PER_BRANCH,
    OFFICIAL_SOMA_CLIP_MV,
    OFFICIAL_SOMA_BIAS_MV,
    OFFICIAL_SOMA_TRAIN_SCALE,
    official_elm_source_metadata,
    official_elm_input_layout,
    official_contact_channel,
    signed_presynaptic_input,
    preprocess_soma_voltage,
    soma_voltage_from_coordinate,
    build_official_elm_twin,
    route_official_input,
    initial_official_elm_state,
    official_elm_step,
    official_elm_forward,
    twin_step,
    twin_forward,
    fit_official_elm_normalizer,
    normalize_official_elm_input,
    denormalize_official_elm_output,
    official_parameter_sha256,
    official_artifact_sha256,
    freeze_official_elm_twin,
    assert_frozen_official_elm_unchanged,
    attest_official_elm_twin,
    attestation_sha256,
    assert_verified_official_elm,
    save_frozen_official_elm,
    load_frozen_official_elm,
    save_verified_official_elm,
    load_verified_official_elm

const OfficialELMConfig = Core.OfficialELMConfig
const OfficialELMState = Core.OfficialELMState
const OfficialPaperELMTwin = Core.OfficialPaperELMTwin
const OFFICIAL_ELM_INPUT_DIM = Core.OFFICIAL_ELM_INPUT_DIM
const OFFICIAL_DENDRITIC_LOCATIONS =
    Core.OFFICIAL_DENDRITIC_LOCATIONS
const OFFICIAL_ELM_BRANCHES = Core.OFFICIAL_ELM_BRANCHES
const OFFICIAL_ELM_SYNAPSES_PER_BRANCH =
    Core.OFFICIAL_ELM_SYNAPSES_PER_BRANCH

const OFFICIAL_SOMA_CLIP_MV = -55.0f0
const OFFICIAL_SOMA_BIAS_MV = -67.7f0
const OFFICIAL_SOMA_TRAIN_SCALE = 0.1f0

const official_elm_input_layout = Core.official_elm_input_layout
const official_contact_channel = Core.official_contact_channel
const signed_presynaptic_input = Core.signed_presynaptic_input
const build_official_elm_twin = Core.build_official_elm_twin
const route_official_input = Core.route_official_input
const initial_official_elm_state = Core.initial_official_elm_state
const official_elm_step = Core.official_elm_step
const official_elm_forward = Core.official_elm_forward

function official_elm_source_metadata()
    return merge(
        Core.official_elm_source_metadata(),
        (;
            canonical_contract_version=2,
            source_derived_input_dim=1_278,
            source_derived_dendritic_locations=639,
            identity_input_normalization=true,
            input_normalization="none",
            soma_training_transform=(
                clip_above_mv=OFFICIAL_SOMA_CLIP_MV,
                bias_mv=OFFICIAL_SOMA_BIAS_MV,
                scale=OFFICIAL_SOMA_TRAIN_SCALE,
            ),
            nmda_train_scaling_is_explicit_extension=true,
        ),
    )
end

preprocess_soma_voltage(voltage_mv) =
    (min.(voltage_mv, OFFICIAL_SOMA_CLIP_MV) .-
     OFFICIAL_SOMA_BIAS_MV) .* OFFICIAL_SOMA_TRAIN_SCALE

soma_voltage_from_coordinate(coordinate) =
    coordinate ./ OFFICIAL_SOMA_TRAIN_SCALE .+
    OFFICIAL_SOMA_BIAS_MV

"""
Fail-closed canonical normalizer.

There are intentionally no input mean/scale fields and no fitted voltage
mean/scale.  The official loader sends signed spikes directly.  Voltage uses
the fixed Spieler clip/bias/scale, and only regional NMDA targets may use
train-split statistics.
"""
struct OfficialELMNormalizer
    nmda_mean::Vector{Float32}
    nmda_scale::Vector{Float32}
end

function OfficialELMNormalizer(
    nmda_mean::AbstractVector{<:Real},
    nmda_scale::AbstractVector{<:Real},
)
    length(nmda_mean) == length(nmda_scale) ||
        throw(DimensionMismatch("NMDA normalizer shapes differ"))
    all(isfinite, nmda_mean) ||
        throw(ArgumentError("NMDA means must be finite"))
    all(value -> isfinite(value) && value > 0, nmda_scale) ||
        throw(ArgumentError("NMDA scales must be finite and positive"))
    return OfficialELMNormalizer(
        Float32.(nmda_mean),
        Float32.(nmda_scale),
    )
end

function fit_official_elm_normalizer(
    input::AbstractArray{<:Real,3},
    target_voltage::AbstractMatrix,
    target_nmda::AbstractArray{<:Real,3},
    indices;
    epsilon::Real=1.0f-5,
)
    isempty(indices) && throw(ArgumentError("normalizer split is empty"))
    size(input, 1) == OFFICIAL_ELM_INPUT_DIM ||
        throw(DimensionMismatch("official ELM input must have 1278 rows"))
    size(input, 3) == size(target_voltage, 2) ==
        size(target_nmda, 3) ||
        throw(DimensionMismatch("normalizer sample dimensions differ"))
    nmda_view = @view target_nmda[:, :, indices]
    nmda_mean = vec(Float32.(mean(nmda_view; dims=(2, 3))))
    nmda_scale = vec(Float32.(std(
        nmda_view;
        dims=(2, 3),
        corrected=false,
    )))
    nmda_scale .= max.(nmda_scale, Float32(epsilon))
    return OfficialELMNormalizer(nmda_mean, nmda_scale)
end

function normalize_official_elm_input(
    ::OfficialELMNormalizer,
    input::AbstractArray{<:Real,3},
)
    size(input, 1) == OFFICIAL_ELM_INPUT_DIM ||
        throw(DimensionMismatch("official ELM input must have 1278 rows"))
    return input
end

function denormalize_official_elm_output(
    normalizer::OfficialELMNormalizer,
    output,
)
    size(output.nmda, 1) == length(normalizer.nmda_mean) ||
        throw(DimensionMismatch("NMDA output/normalizer mismatch"))
    voltage = soma_voltage_from_coordinate(output.voltage)
    nmda =
        output.nmda .*
        reshape(normalizer.nmda_scale, :, 1, 1) .+
        reshape(normalizer.nmda_mean, :, 1, 1)
    return merge(
        output,
        (;
            voltage_coordinate=output.voltage,
            voltage,
            nmda,
        ),
    )
end

function _update_digest!(context, value)
    if value isa NamedTuple
        for (name, child) in pairs(value)
            SHA.update!(context, codeunits(String(name)))
            _update_digest!(context, child)
        end
    elseif value isa AbstractArray
        SHA.update!(context, codeunits(string(eltype(value))))
        SHA.update!(context, codeunits(join(size(value), ",")))
        SHA.update!(context, reinterpret(UInt8, vec(Array(value))))
    elseif value isa OfficialELMNormalizer ||
           value isa Core.OfficialELMConfig
        for field in fieldnames(typeof(value))
            SHA.update!(context, codeunits(String(field)))
            _update_digest!(context, getfield(value, field))
        end
    else
        SHA.update!(context, codeunits(repr(value)))
    end
    return context
end

function _digest_hex(value)
    context = SHA.SHA2_256_CTX()
    _update_digest!(context, value)
    return bytes2hex(SHA.digest!(context))
end

official_parameter_sha256(parameters) = _digest_hex(parameters)

function official_artifact_sha256(
    model::OfficialPaperELMTwin,
    parameters,
    normalizer::OfficialELMNormalizer,
    metadata,
)
    return _digest_hex((;
        config=model.config,
        input_indices=model.input_indices,
        valid_indices_mask=model.valid_indices_mask,
        initial_proto_tau_m=model.initial_proto_tau_m,
        kappa_b=model.kappa_b,
        parameters,
        normalizer,
        metadata,
        source=official_elm_source_metadata(),
    ))
end

const _RESERVED_METADATA_KEYS = (
    :verification_status,
    :verification_passed,
    :verified,
    :verification,
    :held_out_metrics,
    :verification_thresholds,
    :attestation,
    :attestation_sha256,
)

function _reject_verification_metadata(metadata)
    for name in _RESERVED_METADATA_KEYS
        hasproperty(metadata, name) && throw(ArgumentError(
            "verification field `$name` cannot be caller metadata",
        ))
    end
    return metadata
end

struct FrozenOfficialELMTwin{M,P,N,D}
    model::M
    parameters::P
    normalizer::N
    metadata::D
    parameter_sha256::String
    artifact_sha256::String
end

function freeze_official_elm_twin(
    model::OfficialPaperELMTwin,
    parameters,
    normalizer::OfficialELMNormalizer;
    metadata=(;),
)
    _reject_verification_metadata(metadata)
    model.config.num_input == 1_278 ||
        error("official source-derived input_dim attestation failed")
    model.config.num_branch == 45 ||
        error("official branch-count attestation failed")
    model.config.num_synapse_per_branch == 100 ||
        error("official synapse-per-branch attestation failed")
    frozen_parameters = deepcopy(parameters)
    parameter_digest = official_parameter_sha256(frozen_parameters)
    complete_metadata = merge(
        (;
            model_name="Paper-ELM-v2-OfficialRouting-Twin",
            canonical_contract_version=2,
            stage="detailed_cell_to_digital_twin",
            verification_status=:unverified,
            input_dim=1_278,
            dendritic_locations=639,
            branches=45,
            synapses_per_branch=100,
            identity_input_normalization=true,
            soma_clip_mv=OFFICIAL_SOMA_CLIP_MV,
            soma_bias_mv=OFFICIAL_SOMA_BIAS_MV,
            soma_train_scale=OFFICIAL_SOMA_TRAIN_SCALE,
            nmda_train_scaling_is_explicit_extension=true,
            source=official_elm_source_metadata(),
            parameter_sha256=parameter_digest,
        ),
        metadata,
    )
    artifact_digest = official_artifact_sha256(
        model,
        frozen_parameters,
        normalizer,
        complete_metadata,
    )
    return FrozenOfficialELMTwin(
        model,
        frozen_parameters,
        normalizer,
        complete_metadata,
        parameter_digest,
        artifact_digest,
    )
end

function assert_frozen_official_elm_unchanged(
    frozen::FrozenOfficialELMTwin,
)
    official_parameter_sha256(frozen.parameters) ==
        frozen.parameter_sha256 ||
        error("frozen official ELM parameters changed")
    official_artifact_sha256(
        frozen.model,
        frozen.parameters,
        frozen.normalizer,
        frozen.metadata,
    ) == frozen.artifact_sha256 ||
        error("frozen official ELM artifact or metadata changed")
    frozen.metadata.verification_status === :unverified ||
        error("base frozen artifact contains forged verification status")
    return true
end

function twin_forward(
    frozen::FrozenOfficialELMTwin,
    input::AbstractArray{<:Real,3};
    normalized::Bool=false,
    initial_state=nothing,
)
    assert_frozen_official_elm_unchanged(frozen)
    normalized_input =
        normalize_official_elm_input(frozen.normalizer, input)
    output = Core.official_elm_forward(
        frozen.model,
        frozen.parameters,
        normalized_input;
        initial_state,
    )
    return normalized ?
        output :
        denormalize_official_elm_output(frozen.normalizer, output)
end

function twin_step(
    frozen::FrozenOfficialELMTwin,
    state::OfficialELMState,
    input::AbstractMatrix;
    normalized::Bool=false,
)
    assert_frozen_official_elm_unchanged(frozen)
    size(input, 1) == OFFICIAL_ELM_INPUT_DIM ||
        throw(DimensionMismatch("official ELM input must have 1278 rows"))
    output = Core.official_elm_step(
        frozen.model,
        frozen.parameters,
        state,
        input,
    )
    normalized && return output
    voltage = soma_voltage_from_coordinate(output.voltage)
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

struct OfficialHeldOutAttestation{M,T}
    metrics::M
    thresholds::T
    teacher_manifest_sha256::String
    teacher_contract_sha256::String
    evaluator_id::String
    source_metadata_sha256::String
    model_config_sha256::String
    parameter_sha256::String
    base_artifact_sha256::String
    passed::Bool
end

struct VerifiedOfficialELMTwin{M,P,N,D,A}
    model::M
    parameters::P
    normalizer::N
    metadata::D
    parameter_sha256::String
    artifact_sha256::String
    attestation::A
    attestation_sha256::String
end

function _required_value(values, primary::Symbol, alternate::Symbol)
    if hasproperty(values, primary)
        return Float64(getproperty(values, primary))
    elseif hasproperty(values, alternate)
        return Float64(getproperty(values, alternate))
    end
    throw(ArgumentError("missing required field `$primary`"))
end

function _canonical_metrics(metrics)
    result = (;
        voltage_rmse=_required_value(
            metrics,
            :voltage_rmse,
            :held_out_voltage_rmse,
        ),
        spike_auroc=_required_value(
            metrics,
            :spike_auroc,
            :held_out_spike_auroc,
        ),
        nmda_rmse=_required_value(
            metrics,
            :nmda_rmse,
            :held_out_nmda_rmse,
        ),
    )
    all(isfinite, values(result)) ||
        throw(ArgumentError("held-out metrics must be finite"))
    return result
end

function _canonical_thresholds(thresholds)
    result = (;
        max_voltage_rmse=_required_value(
            thresholds,
            :max_voltage_rmse,
            :voltage_rmse_max,
        ),
        min_spike_auroc=_required_value(
            thresholds,
            :min_spike_auroc,
            :spike_auroc_min,
        ),
        max_nmda_rmse=_required_value(
            thresholds,
            :max_nmda_rmse,
            :nmda_rmse_max,
        ),
    )
    all(isfinite, values(result)) ||
        throw(ArgumentError("verification thresholds must be finite"))
    result.max_voltage_rmse >= 0 ||
        throw(ArgumentError("max_voltage_rmse must be non-negative"))
    0 <= result.min_spike_auroc <= 1 ||
        throw(ArgumentError("min_spike_auroc must be in [0,1]"))
    result.max_nmda_rmse >= 0 ||
        throw(ArgumentError("max_nmda_rmse must be non-negative"))
    return result
end

function _passed(metrics, thresholds)
    return metrics.voltage_rmse <= thresholds.max_voltage_rmse &&
           metrics.spike_auroc >= thresholds.min_spike_auroc &&
           metrics.nmda_rmse <= thresholds.max_nmda_rmse
end

function _require_sha256(name::AbstractString, value::AbstractString)
    occursin(r"^[0-9a-fA-F]{64}$", value) ||
        throw(ArgumentError("$name must be a complete SHA-256"))
    return lowercase(String(value))
end

attestation_sha256(attestation::OfficialHeldOutAttestation) =
    _digest_hex(attestation)

function attest_official_elm_twin(
    frozen::FrozenOfficialELMTwin;
    metrics,
    thresholds,
    teacher_manifest_sha256::AbstractString,
    teacher_contract_sha256::AbstractString,
    evaluator_id::AbstractString,
)
    assert_frozen_official_elm_unchanged(frozen)
    canonical_metrics = _canonical_metrics(metrics)
    canonical_thresholds = _canonical_thresholds(thresholds)
    passed = _passed(canonical_metrics, canonical_thresholds)
    passed || error("official ELM held-out verification thresholds failed")
    isempty(evaluator_id) &&
        throw(ArgumentError("evaluator_id must not be empty"))
    attestation = OfficialHeldOutAttestation(
        canonical_metrics,
        canonical_thresholds,
        _require_sha256(
            "teacher_manifest_sha256",
            teacher_manifest_sha256,
        ),
        _require_sha256(
            "teacher_contract_sha256",
            teacher_contract_sha256,
        ),
        String(evaluator_id),
        _digest_hex(official_elm_source_metadata()),
        _digest_hex(frozen.model.config),
        frozen.parameter_sha256,
        frozen.artifact_sha256,
        true,
    )
    digest = attestation_sha256(attestation)
    return VerifiedOfficialELMTwin(
        frozen.model,
        frozen.parameters,
        frozen.normalizer,
        frozen.metadata,
        frozen.parameter_sha256,
        frozen.artifact_sha256,
        attestation,
        digest,
    )
end

function assert_verified_official_elm(
    verified::VerifiedOfficialELMTwin,
)
    base = FrozenOfficialELMTwin(
        verified.model,
        verified.parameters,
        verified.normalizer,
        verified.metadata,
        verified.parameter_sha256,
        verified.artifact_sha256,
    )
    assert_frozen_official_elm_unchanged(base)
    attestation = verified.attestation
    attestation.passed ||
        error("attestation pass status is false")
    _passed(attestation.metrics, attestation.thresholds) ||
        error("attested metrics do not satisfy attested thresholds")
    attestation.source_metadata_sha256 ==
        _digest_hex(official_elm_source_metadata()) ||
        error("attested source metadata changed")
    attestation.model_config_sha256 ==
        _digest_hex(verified.model.config) ||
        error("attested model configuration changed")
    attestation.parameter_sha256 == verified.parameter_sha256 ||
        error("attested parameter digest changed")
    attestation.base_artifact_sha256 == verified.artifact_sha256 ||
        error("attested base artifact digest changed")
    attestation_sha256(attestation) ==
        verified.attestation_sha256 ||
        error("held-out attestation digest changed")
    return true
end

function twin_forward(
    verified::VerifiedOfficialELMTwin,
    input::AbstractArray{<:Real,3};
    kwargs...,
)
    assert_verified_official_elm(verified)
    frozen = FrozenOfficialELMTwin(
        verified.model,
        verified.parameters,
        verified.normalizer,
        verified.metadata,
        verified.parameter_sha256,
        verified.artifact_sha256,
    )
    return twin_forward(frozen, input; kwargs...)
end

function twin_step(
    verified::VerifiedOfficialELMTwin,
    state::OfficialELMState,
    input::AbstractMatrix;
    kwargs...,
)
    assert_verified_official_elm(verified)
    frozen = FrozenOfficialELMTwin(
        verified.model,
        verified.parameters,
        verified.normalizer,
        verified.metadata,
        verified.parameter_sha256,
        verified.artifact_sha256,
    )
    return twin_step(frozen, state, input; kwargs...)
end

function save_frozen_official_elm(
    path::AbstractString,
    frozen::FrozenOfficialELMTwin,
)
    assert_frozen_official_elm_unchanged(frozen)
    parent = dirname(abspath(path))
    isdir(parent) || mkpath(parent)
    jldsave(
        path;
        artifact_kind="FrozenOfficialELMTwinUnverified",
        format_version=3,
        frozen,
    )
    return abspath(path)
end

function load_frozen_official_elm(path::AbstractString)
    isfile(path) ||
        throw(ArgumentError("official ELM artifact not found: $path"))
    payload = JLD2.load(path)
    get(payload, "artifact_kind", nothing) ==
        "FrozenOfficialELMTwinUnverified" ||
        error("not an unverified official ELM artifact")
    get(payload, "format_version", nothing) == 3 ||
        error("unsupported official ELM artifact version")
    frozen = payload["frozen"]
    frozen isa FrozenOfficialELMTwin ||
        error("artifact payload has the wrong type")
    assert_frozen_official_elm_unchanged(frozen)
    return frozen
end

function save_verified_official_elm(
    path::AbstractString,
    verified::VerifiedOfficialELMTwin,
)
    assert_verified_official_elm(verified)
    parent = dirname(abspath(path))
    isdir(parent) || mkpath(parent)
    jldsave(
        path;
        artifact_kind="VerifiedOfficialELMTwin",
        format_version=3,
        verified,
    )
    return abspath(path)
end

function load_verified_official_elm(path::AbstractString)
    isfile(path) ||
        throw(ArgumentError("official ELM artifact not found: $path"))
    payload = JLD2.load(path)
    get(payload, "artifact_kind", nothing) ==
        "VerifiedOfficialELMTwin" ||
        error("artifact is not a strictly attested official ELM twin")
    get(payload, "format_version", nothing) == 3 ||
        error("unsupported verified official ELM artifact version")
    verified = payload["verified"]
    verified isa VerifiedOfficialELMTwin ||
        error("verified artifact payload has the wrong type")
    assert_verified_official_elm(verified)
    return verified
end

end # module PaperELMTwinOfficialV2Final
