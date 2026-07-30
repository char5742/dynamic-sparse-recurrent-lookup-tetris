module PaperELMTwinOfficialV2ReleaseAttestation

using JLD2
using SHA
using Statistics

include(joinpath(@__DIR__, "PaperELMTwinOfficialV2.jl"))
using .PaperELMTwinOfficialV2

export TeacherReleaseIdentity,
    OfficialReleaseSplits,
    OfficialELMHeldoutSet,
    OfficialELMReleaseAttestation,
    AttestedOfficialELMRelease,
    RELEASE_ATTESTATION_SCHEMA,
    RELEASE_CANONICAL_ENCODING,
    MINIMUM_SPIKE_AUROC,
    MAXIMUM_VOLTAGE_RMSE_MV,
    MAXIMUM_NMDA_NORMALIZED_MSE,
    MINIMUM_NMDA_CORRELATION,
    PAPER_TRAIN_POOL_TRIALS,
    PAPER_HELDOUT_TRIALS,
    PAPER_TRIAL_DURATION_MS,
    canonical_sha256,
    release_artifact_payload_sha256,
    recompute_heldout_metrics,
    attest_official_elm_release,
    verify_official_elm_release,
    save_attested_official_elm_release,
    load_checked_official_elm_release,
    load_verified_official_elm_release

const RELEASE_ATTESTATION_SCHEMA =
    "hd_swsnn.paper_elm_v2.release_attestation.final.v1"
const RELEASE_CANONICAL_ENCODING =
    "sha256-tagged-julia-column-major-little-endian-v1"
const RELEASE_ARTIFACT_KIND = "AttestedOfficialELMRelease"
const RELEASE_ARTIFACT_FORMAT_VERSION = 1

# These are release policy, not caller-configurable training options.
const MINIMUM_SPIKE_AUROC = 0.985
const MAXIMUM_VOLTAGE_RMSE_MV = 5.0
const MAXIMUM_NMDA_NORMALIZED_MSE = 0.25
const MINIMUM_NMDA_CORRELATION = 0.80

const PAPER_TRAIN_POOL_TRIALS = 50_000
const PAPER_HELDOUT_TRIALS = 2_000
const PAPER_TRIAL_DURATION_MS = 10_000.0
const PAPER_NMDA_REGIONS = 4

const _MODEL_SOURCE_PATH =
    joinpath(@__DIR__, "PaperELMTwinOfficialV2.jl")
const _ATTESTATION_SOURCE_PATH =
    joinpath(@__DIR__, "PaperELMTwinOfficialV2ReleaseAttestation.jl")

@inline function _valid_sha256(value::AbstractString)
    return occursin(r"^[0-9a-f]{64}$", value)
end

function _required_sha256(value, label)
    text = lowercase(String(value))
    _valid_sha256(text) ||
        throw(ArgumentError("$label must be a lowercase SHA-256"))
    return text
end

"""
Identity obtained from the independently verified teacher manifest.

`connectivity_paper_scale_verified` is evidence from the teacher contract, not
an outcome supplied by the trainer.  Exact trial counts and duration are still
recomputed below; this flag alone can never make an artifact production-scale.
"""
struct TeacherReleaseIdentity
    manifest_sha256::String
    teacher_contract_sha256::String
    validation_from_train_indices::Vector{Int64}
    connectivity_paper_scale_verified::Bool
end

function TeacherReleaseIdentity(
    manifest_sha256,
    teacher_contract_sha256,
    validation_from_train_indices;
    connectivity_paper_scale_verified::Bool=false,
)
    validation = Int64.(collect(validation_from_train_indices))
    isempty(validation) &&
        throw(ArgumentError("manifest validation indices are empty"))
    length(unique(validation)) == length(validation) ||
        throw(ArgumentError("manifest validation indices are not unique"))
    all(>(0), validation) ||
        throw(ArgumentError("manifest validation indices must be positive"))
    return TeacherReleaseIdentity(
        _required_sha256(manifest_sha256, "teacher manifest"),
        _required_sha256(
            teacher_contract_sha256,
            "teacher contract",
        ),
        validation,
        connectivity_paper_scale_verified,
    )
end

"""
Concrete split membership, not caller-reported counts.

The validation membership must exactly equal the manifest-declared
`validation_from_train_indices`; its ordered values and digest are placed in
the attestation.
"""
struct OfficialReleaseSplits
    fit_trial_ids::Vector{Int64}
    validation_trial_ids::Vector{Int64}
    heldout_trial_ids::Vector{Int64}
end

function OfficialReleaseSplits(
    fit_trial_ids,
    validation_trial_ids,
    heldout_trial_ids,
)
    fit = Int64.(collect(fit_trial_ids))
    validation = Int64.(collect(validation_trial_ids))
    heldout = Int64.(collect(heldout_trial_ids))
    isempty(fit) && throw(ArgumentError("fit split is empty"))
    isempty(validation) &&
        throw(ArgumentError("validation split is empty"))
    isempty(heldout) &&
        throw(ArgumentError("held-out split is empty"))
    for (label, ids) in (
        ("fit", fit),
        ("validation", validation),
        ("held-out", heldout),
    )
        all(>(0), ids) ||
            throw(ArgumentError("$label trial IDs must be positive"))
        length(unique(ids)) == length(ids) ||
            throw(ArgumentError("$label trial IDs are not unique"))
    end
    isempty(intersect(fit, validation)) ||
        throw(ArgumentError("fit and validation splits overlap"))
    isempty(intersect(fit, heldout)) ||
        throw(ArgumentError("fit and held-out splits overlap"))
    isempty(intersect(validation, heldout)) ||
        throw(ArgumentError("validation and held-out splits overlap"))
    return OfficialReleaseSplits(fit, validation, heldout)
end

"""
Raw held-out evidence used by the attestor.

No metrics or pass/fail field exists here.  Predictions and all metrics are
computed inside this module from the frozen ELM and these targets.
"""
struct OfficialELMHeldoutSet{I,V,S,N}
    trial_ids::Vector{Int64}
    input::I
    target_voltage::V
    target_spike::S
    target_nmda::N
end

function OfficialELMHeldoutSet(
    trial_ids,
    input,
    target_voltage,
    target_spike,
    target_nmda,
)
    return OfficialELMHeldoutSet(
        Int64.(collect(trial_ids)),
        input,
        target_voltage,
        target_spike,
        target_nmda,
    )
end

struct OfficialELMReleaseAttestation{P}
    payload::P
    attestation_sha256::String
end

struct AttestedOfficialELMRelease{F,A}
    frozen::F
    attestation::A
end

# ---------------------------------------------------------------------------
# Canonical SHA-256 encoding
# ---------------------------------------------------------------------------

function _update_text!(context, value::AbstractString)
    bytes = codeunits(value)
    SHA.update!(context, codeunits(string(length(bytes))))
    SHA.update!(context, UInt8[0x3a])
    SHA.update!(context, bytes)
    return context
end

function _canonical_update!(context, value)
    if value === nothing
        _update_text!(context, "nothing")
    elseif value === missing
        _update_text!(context, "missing")
    elseif value isa Bool
        _update_text!(context, value ? "bool:1" : "bool:0")
    elseif value isa Integer
        _update_text!(
            context,
            "integer:" * string(typeof(value)) * ":" * string(value),
        )
    elseif value isa AbstractFloat
        isfinite(value) ||
            error("canonical release payload contains a non-finite float")
        _update_text!(
            context,
            "float:" * string(typeof(value)) * ":" * bitstring(value),
        )
    elseif value isa AbstractString
        _update_text!(context, "string")
        _update_text!(context, value)
    elseif value isa Symbol
        _update_text!(context, "symbol")
        _update_text!(context, String(value))
    elseif value isa NamedTuple
        _update_text!(context, "namedtuple")
        _update_text!(context, string(length(value)))
        for name in keys(value)
            _update_text!(context, String(name))
            _canonical_update!(context, getproperty(value, name))
        end
    elseif value isa Tuple
        _update_text!(context, "tuple")
        _update_text!(context, string(length(value)))
        for child in value
            _canonical_update!(context, child)
        end
    elseif value isa AbstractDict
        _update_text!(context, "dict")
        ordered = sort!(collect(keys(value)); by=key -> string(key))
        _update_text!(context, string(length(ordered)))
        for key in ordered
            _canonical_update!(context, key)
            _canonical_update!(context, value[key])
        end
    elseif value isa AbstractArray
        _update_text!(context, "array")
        _update_text!(context, string(eltype(value)))
        _canonical_update!(context, Tuple(size(value)))
        if isbitstype(eltype(value))
            ENDIAN_BOM == 0x04030201 ||
                error("canonical array encoder requires little-endian host")
            contiguous = vec(Array(value))
            bytes = reinterpret(UInt8, contiguous)
            _update_text!(context, string(length(bytes)))
            SHA.update!(context, bytes)
        else
            _update_text!(context, string(length(value)))
            for child in value
                _canonical_update!(context, child)
            end
        end
    elseif isstructtype(typeof(value))
        _update_text!(
            context,
            "struct:" * string(parentmodule(typeof(value))) * "." *
            string(nameof(typeof(value))),
        )
        names = fieldnames(typeof(value))
        _update_text!(context, string(length(names)))
        for name in names
            _update_text!(context, String(name))
            _canonical_update!(context, getfield(value, name))
        end
    else
        error("unsupported canonical release payload type $(typeof(value))")
    end
    return context
end

function canonical_sha256(value)
    context = SHA.SHA2_256_CTX()
    _canonical_update!(context, value)
    return bytes2hex(SHA.digest!(context))
end

_file_sha256(path) = bytes2hex(SHA.sha256(read(path)))

function _normalizer_sha256(normalizer)
    return canonical_sha256(normalizer)
end

function _model_structure_sha256(frozen)
    model = frozen.model
    return canonical_sha256((;
        config=model.config,
        input_indices=model.input_indices,
        valid_indices_mask=model.valid_indices_mask,
        initial_proto_tau_m=model.initial_proto_tau_m,
        kappa_b=model.kappa_b,
    ))
end

"""
Hash every logical field of the frozen artifact, including metadata.

This is deliberately stronger than `official_artifact_sha256`, whose stable
model hash excludes mutable release metadata.
"""
function release_artifact_payload_sha256(
    frozen::PaperELMTwinOfficialV2.FrozenOfficialELMTwin,
)
    return canonical_sha256((;
        model=frozen.model,
        parameters=frozen.parameters,
        normalizer=frozen.normalizer,
        metadata=frozen.metadata,
        stored_parameter_sha256=frozen.parameter_sha256,
        stored_artifact_sha256=frozen.artifact_sha256,
    ))
end

# ---------------------------------------------------------------------------
# Held-out metrics and scale evidence
# ---------------------------------------------------------------------------

function _validate_heldout!(
    frozen::PaperELMTwinOfficialV2.FrozenOfficialELMTwin,
    heldout::OfficialELMHeldoutSet,
    identity::TeacherReleaseIdentity,
    splits::OfficialReleaseSplits,
)
    config = frozen.model.config
    identity.validation_from_train_indices ==
        splits.validation_trial_ids ||
        error(
            "validation split differs from manifest " *
            "validation_from_train_indices",
        )
    heldout.trial_ids == splits.heldout_trial_ids ||
        error("held-out trial order differs from release split evidence")
    length(unique(heldout.trial_ids)) == length(heldout.trial_ids) ||
        error("held-out trial IDs are not unique")

    ndims(heldout.input) == 3 ||
        throw(DimensionMismatch("held-out input must be feature x time x trial"))
    features, time_steps, trials = size(heldout.input)
    features == config.num_input ||
        throw(DimensionMismatch("held-out input feature count differs"))
    trials == length(heldout.trial_ids) ||
        throw(DimensionMismatch("held-out trial count differs"))
    size(heldout.target_voltage) == (time_steps, trials) ||
        throw(DimensionMismatch("held-out voltage target shape differs"))
    size(heldout.target_spike) == (time_steps, trials) ||
        throw(DimensionMismatch("held-out spike target shape differs"))
    size(heldout.target_nmda) ==
        (config.nmda_regions, time_steps, trials) ||
        throw(DimensionMismatch("held-out NMDA target shape differs"))

    all(isfinite, heldout.input) ||
        error("held-out input contains non-finite values")
    all(isfinite, heldout.target_voltage) ||
        error("held-out voltage target contains non-finite values")
    all(isfinite, heldout.target_nmda) ||
        error("held-out NMDA target contains non-finite values")
    all(value -> value == 0 || value == 1, heldout.target_spike) ||
        error("held-out spike targets must be exactly zero or one")
    return (; time_steps, trials)
end

function _binary_auroc(scores, labels)
    score = Float64.(vec(scores))
    label = Bool.(vec(labels))
    all(isfinite, score) || error("spike scores contain non-finite values")
    positives = count(identity, label)
    negatives = length(label) - positives
    positives > 0 && negatives > 0 || return NaN

    order = sortperm(score; alg=MergeSort)
    rank_sum = 0.0
    first_index = 1
    while first_index <= length(order)
        last_index = first_index
        tied_score = score[order[first_index]]
        while last_index < length(order) &&
              score[order[last_index + 1]] == tied_score
            last_index += 1
        end
        average_rank = 0.5 * (first_index + last_index)
        for position in first_index:last_index
            label[order[position]] &&
                (rank_sum += average_rank)
        end
        first_index = last_index + 1
    end
    return (
        rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)
end

function _correlation(left, right)
    x = Float64.(vec(left))
    y = Float64.(vec(right))
    length(x) == length(y) ||
        throw(DimensionMismatch("correlation input lengths differ"))
    x_centered = x .- mean(x)
    y_centered = y .- mean(y)
    denominator = sqrt(
        sum(abs2, x_centered) * sum(abs2, y_centered),
    )
    denominator > eps(Float64) || return NaN
    return sum(x_centered .* y_centered) / denominator
end

function recompute_heldout_metrics(
    frozen::PaperELMTwinOfficialV2.FrozenOfficialELMTwin,
    heldout::OfficialELMHeldoutSet,
)
    prediction = PaperELMTwinOfficialV2.twin_forward(
        frozen,
        heldout.input,
    )
    voltage_error =
        Float64.(prediction.voltage) .-
        Float64.(heldout.target_voltage)
    voltage_rmse_mv = sqrt(mean(abs2, voltage_error))
    spike_auroc = _binary_auroc(
        prediction.spike_logit,
        heldout.target_spike,
    )

    regions = size(heldout.target_nmda, 1)
    nmda_rmse_by_region = Vector{Float64}(undef, regions)
    nmda_correlation_by_region = Vector{Float64}(undef, regions)
    nmda_normalized_mse_by_region = Vector{Float64}(undef, regions)
    @inbounds for region in 1:regions
        predicted = @view prediction.nmda[region, :, :]
        target = @view heldout.target_nmda[region, :, :]
        error = Float64.(predicted) .- Float64.(target)
        target64 = Float64.(target)
        centered = target64 .- mean(target64)
        target_variance_sum = sum(abs2, centered)
        nmda_rmse_by_region[region] =
            sqrt(mean(abs2, error))
        nmda_correlation_by_region[region] =
            _correlation(predicted, target)
        nmda_normalized_mse_by_region[region] =
            target_variance_sum > eps(Float64) ?
            sum(abs2, error) / target_variance_sum :
            Inf
    end
    nmda_normalized_mse =
        mean(nmda_normalized_mse_by_region)
    nmda_mean_correlation =
        mean(nmda_correlation_by_region)
    positives = count(value -> value == 1, heldout.target_spike)
    observations = length(heldout.target_spike)

    return (;
        spike_auroc,
        voltage_rmse_mv,
        nmda_normalized_mse,
        nmda_mean_correlation,
        nmda_rmse_by_region,
        nmda_normalized_mse_by_region,
        nmda_correlation_by_region,
        observations,
        spike_positives=positives,
        spike_negatives=observations - positives,
    )
end

function _fixed_gate(metrics)
    finite =
        isfinite(metrics.spike_auroc) &&
        isfinite(metrics.voltage_rmse_mv) &&
        isfinite(metrics.nmda_normalized_mse) &&
        isfinite(metrics.nmda_mean_correlation) &&
        all(isfinite, metrics.nmda_rmse_by_region) &&
        all(isfinite, metrics.nmda_normalized_mse_by_region) &&
        all(isfinite, metrics.nmda_correlation_by_region)
    passed =
        finite &&
        metrics.spike_auroc >= MINIMUM_SPIKE_AUROC &&
        metrics.voltage_rmse_mv <= MAXIMUM_VOLTAGE_RMSE_MV &&
        metrics.nmda_normalized_mse <=
            MAXIMUM_NMDA_NORMALIZED_MSE &&
        metrics.nmda_mean_correlation >=
            MINIMUM_NMDA_CORRELATION
    return (;
        minimum_spike_auroc=MINIMUM_SPIKE_AUROC,
        maximum_voltage_rmse_mv=MAXIMUM_VOLTAGE_RMSE_MV,
        maximum_nmda_normalized_mse=
            MAXIMUM_NMDA_NORMALIZED_MSE,
        minimum_nmda_correlation=MINIMUM_NMDA_CORRELATION,
        passed,
    )
end

function _split_payload(
    frozen,
    identity::TeacherReleaseIdentity,
    splits::OfficialReleaseSplits,
    time_steps::Int,
)
    duration_ms =
        time_steps * Float64(frozen.model.config.delta_t_ms)
    train_pool_count =
        length(splits.fit_trial_ids) +
        length(splits.validation_trial_ids)
    return (;
        fit_count=length(splits.fit_trial_ids),
        validation_count=length(splits.validation_trial_ids),
        heldout_count=length(splits.heldout_trial_ids),
        train_pool_count,
        time_steps,
        delta_t_ms=Float64(frozen.model.config.delta_t_ms),
        trial_duration_ms=duration_ms,
        validation_from_train_indices=
            copy(splits.validation_trial_ids),
        validation_from_train_indices_sha256=
            canonical_sha256(splits.validation_trial_ids),
        fit_trial_ids_sha256=
            canonical_sha256(splits.fit_trial_ids),
        heldout_trial_ids_sha256=
            canonical_sha256(splits.heldout_trial_ids),
        split_membership_sha256=canonical_sha256((;
            fit=splits.fit_trial_ids,
            validation=splits.validation_trial_ids,
            heldout=splits.heldout_trial_ids,
        )),
        teacher_contract_connectivity_paper_scale_verified=
            identity.connectivity_paper_scale_verified,
    )
end

function _is_paper_scale(split_payload, frozen)
    return (
        split_payload.train_pool_count == PAPER_TRAIN_POOL_TRIALS &&
        split_payload.heldout_count == PAPER_HELDOUT_TRIALS &&
        split_payload.trial_duration_ms == PAPER_TRIAL_DURATION_MS &&
        split_payload.teacher_contract_connectivity_paper_scale_verified &&
        frozen.model.config.nmda_regions == PAPER_NMDA_REGIONS
    )
end

function _heldout_sha256(heldout::OfficialELMHeldoutSet)
    return canonical_sha256((;
        trial_ids=heldout.trial_ids,
        input=heldout.input,
        target_voltage=heldout.target_voltage,
        target_spike=heldout.target_spike,
        target_nmda=heldout.target_nmda,
    ))
end

function _attestation_payload(
    frozen::PaperELMTwinOfficialV2.FrozenOfficialELMTwin,
    heldout::OfficialELMHeldoutSet,
    identity::TeacherReleaseIdentity,
    splits::OfficialReleaseSplits,
)
    dimensions = _validate_heldout!(
        frozen,
        heldout,
        identity,
        splits,
    )
    PaperELMTwinOfficialV2.assert_frozen_official_elm_unchanged(frozen)
    metrics = recompute_heldout_metrics(frozen, heldout)
    gate = _fixed_gate(metrics)
    split = _split_payload(
        frozen,
        identity,
        splits,
        dimensions.time_steps,
    )
    paper_scale = _is_paper_scale(split, frozen)
    promotable_production = gate.passed && paper_scale
    model_source_sha256 = _file_sha256(_MODEL_SOURCE_PATH)
    attestation_source_sha256 =
        _file_sha256(_ATTESTATION_SOURCE_PATH)
    upstream_source_sha256 = canonical_sha256(
        PaperELMTwinOfficialV2.official_elm_source_metadata(),
    )
    config_sha256 = canonical_sha256(frozen.model.config)
    model_structure_sha256 = _model_structure_sha256(frozen)
    parameter_sha256 =
        PaperELMTwinOfficialV2.official_parameter_sha256(
            frozen.parameters,
        )
    normalizer_sha256 = _normalizer_sha256(frozen.normalizer)
    official_artifact_sha256 =
        PaperELMTwinOfficialV2.official_artifact_sha256(
            frozen.model,
            frozen.parameters,
            frozen.normalizer,
        )
    artifact_payload_sha256 =
        release_artifact_payload_sha256(frozen)

    return (;
        schema=RELEASE_ATTESTATION_SCHEMA,
        canonical_encoding=RELEASE_CANONICAL_ENCODING,
        artifact_kind=RELEASE_ARTIFACT_KIND,
        model_contract=(;
            model_family="Paper-ELM-v2-OfficialRouting-Twin",
            input_dim=frozen.model.config.num_input,
            branches=frozen.model.config.num_branch,
            synapses_per_branch=
                frozen.model.config.num_synapse_per_branch,
            memory_units=frozen.model.config.num_memory,
            hidden_size=frozen.model.config.hidden_size,
            nmda_regions=frozen.model.config.nmda_regions,
            official_routing=
                frozen.model.config.input_to_synapse_routing,
        ),
        teacher_contract=(;
            manifest_sha256=identity.manifest_sha256,
            teacher_contract_sha256=
                identity.teacher_contract_sha256,
        ),
        hashes=(;
            teacher_manifest_sha256=identity.manifest_sha256,
            teacher_contract_sha256=
                identity.teacher_contract_sha256,
            upstream_source_manifest_sha256=
                upstream_source_sha256,
            model_implementation_source_sha256=
                model_source_sha256,
            attestation_implementation_source_sha256=
                attestation_source_sha256,
            model_config_sha256=config_sha256,
            model_structure_sha256,
            parameter_sha256,
            normalizer_sha256,
            official_artifact_sha256,
            artifact_payload_sha256,
            heldout_dataset_sha256=_heldout_sha256(heldout),
            validation_ids_sha256=
                split.validation_from_train_indices_sha256,
            split_membership_sha256=
                split.split_membership_sha256,
        ),
        split,
        target_contract=(;
            original_paper_targets=(
                "soma_voltage",
                "soma_spike",
            ),
            project_required_extension_targets=(
                "regional_nmda_current",
            ),
            regional_nmda_is_project_extension=true,
            paper_identical_training_claimed=false,
            unpublished_original_checkpoint_identity_claimed=false,
        ),
        metrics,
        fixed_gate=gate,
        outcome=(;
            gate_passed=gate.passed,
            paper_scale,
            promotable_production,
            development_scale=!paper_scale,
            caller_supplied_metrics_accepted=false,
            caller_supplied_pass_flag_accepted=false,
            verification_origin=
                "recomputed_from_frozen_model_and_raw_heldout_targets",
        ),
    )
end

"""
Recompute predictions and construct a hash-bound attestation.

There are deliberately no `metrics`, `passed`, `paper_scale`, or
`promotable` arguments.
"""
function attest_official_elm_release(
    frozen::PaperELMTwinOfficialV2.FrozenOfficialELMTwin,
    heldout::OfficialELMHeldoutSet,
    identity::TeacherReleaseIdentity,
    splits::OfficialReleaseSplits,
)
    payload = _attestation_payload(
        frozen,
        heldout,
        identity,
        splits,
    )
    attestation = OfficialELMReleaseAttestation(
        payload,
        canonical_sha256(payload),
    )
    return AttestedOfficialELMRelease(frozen, attestation)
end

function verify_official_elm_release(
    bundle::AttestedOfficialELMRelease,
    heldout::OfficialELMHeldoutSet,
    identity::TeacherReleaseIdentity,
    splits::OfficialReleaseSplits;
    require_gate::Bool=false,
    require_production::Bool=false,
)
    stored = bundle.attestation
    stored.attestation_sha256 ==
        canonical_sha256(stored.payload) ||
        error("release attestation payload digest mismatch")
    expected = attest_official_elm_release(
        bundle.frozen,
        heldout,
        identity,
        splits,
    )
    stored.attestation_sha256 ==
        expected.attestation.attestation_sha256 ||
        error(
            "release attestation differs from independently " *
            "recomputed evidence",
        )
    canonical_sha256(stored.payload) ==
        canonical_sha256(expected.attestation.payload) ||
        error("release attestation logical payload mismatch")
    stored.payload.hashes.parameter_sha256 ==
        bundle.frozen.parameter_sha256 ||
        error("attested parameter digest differs from frozen artifact")
    stored.payload.hashes.official_artifact_sha256 ==
        bundle.frozen.artifact_sha256 ||
        error("attested model artifact digest differs")
    require_gate &&
        stored.payload.outcome.gate_passed !== true &&
        error("held-out voltage/spike/NMDA release gate did not pass")
    require_production &&
        stored.payload.outcome.promotable_production !== true &&
        error(
            "development-scale artifact is not production/promotable",
        )
    return bundle
end

function save_attested_official_elm_release(
    path::AbstractString,
    bundle::AttestedOfficialELMRelease,
)
    PaperELMTwinOfficialV2.assert_frozen_official_elm_unchanged(
        bundle.frozen,
    )
    bundle.attestation.attestation_sha256 ==
        canonical_sha256(bundle.attestation.payload) ||
        error("refusing to save a corrupt release attestation")
    parent = dirname(abspath(path))
    isdir(parent) || mkpath(parent)
    jldsave(
        path;
        artifact_kind=RELEASE_ARTIFACT_KIND,
        format_version=RELEASE_ARTIFACT_FORMAT_VERSION,
        bundle,
    )
    return abspath(path)
end

function _load_attested_official_elm_release(path::AbstractString)
    isfile(path) ||
        throw(ArgumentError("attested official ELM not found: $path"))
    data = JLD2.load(path)
    get(data, "artifact_kind", nothing) ==
        RELEASE_ARTIFACT_KIND ||
        error("not an attested official ELM release")
    get(data, "format_version", nothing) ==
        RELEASE_ARTIFACT_FORMAT_VERSION ||
        error("unsupported official ELM release attestation version")
    bundle = get(data, "bundle", nothing)
    bundle isa AttestedOfficialELMRelease ||
        error("attested official ELM payload has the wrong type")
    return bundle
end

"""
Load and independently recalculate all hash and metric fields.

This integrity loader returns development-scale artifacts but leaves
`promotable_production=false`; it never trusts a stored verification flag.
"""
function load_checked_official_elm_release(
    path::AbstractString,
    heldout::OfficialELMHeldoutSet,
    identity::TeacherReleaseIdentity,
    splits::OfficialReleaseSplits,
)
    bundle = _load_attested_official_elm_release(path)
    return verify_official_elm_release(
        bundle,
        heldout,
        identity,
        splits;
        require_gate=false,
        require_production=false,
    )
end

"""
Verified loader.  The held-out gate is always recomputed.

Production is required by default.  Development-scale rich64 artifacts may be
opened only with the explicit `require_production=false`; even then their
stored and recomputed `promotable_production` value remains false.
"""
function load_verified_official_elm_release(
    path::AbstractString,
    heldout::OfficialELMHeldoutSet,
    identity::TeacherReleaseIdentity,
    splits::OfficialReleaseSplits;
    require_production::Bool=true,
)
    bundle = _load_attested_official_elm_release(path)
    return verify_official_elm_release(
        bundle,
        heldout,
        identity,
        splits;
        require_gate=true,
        require_production,
    )
end

end # module PaperELMTwinOfficialV2ReleaseAttestation
