module PaperELMTwinOfficialV2ReleaseAttestationFinal

using JLD2
using SHA
using Statistics

# Reuse the one canonical model module/type in the process.  Including the
# model again inside this module would create an incompatible Julia type.
if !isdefined(Main, :PaperELMTwinOfficialV2)
    Base.include(
        Main,
        joinpath(@__DIR__, "PaperELMTwinOfficialV2.jl"),
    )
end
const Twin = Main.PaperELMTwinOfficialV2

export TeacherReleaseIdentity,
    OfficialReleaseSplits,
    OfficialELMHeldoutSet,
    OfficialELMReleaseAttestation,
    AttestedOfficialELMRelease,
    RELEASE_ATTESTATION_SCHEMA,
    RELEASE_CANONICAL_ENCODING,
    MINIMUM_SPIKE_AUROC,
    MAXIMUM_VOLTAGE_RMSE_MV,
    MAXIMUM_NMDA_NORMALIZED_RMSE,
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
    "hd_swsnn.paper_elm_v2.release_attestation.final.v2"
const RELEASE_CANONICAL_ENCODING =
    "sha256-tagged-julia-column-major-little-endian-v1"
const RELEASE_ARTIFACT_KIND = "AttestedOfficialELMReleaseFinal"
const RELEASE_ARTIFACT_FORMAT_VERSION = 2

# Immutable release policy.  No function accepts caller, CLI, or environment
# overrides for these values.
const MINIMUM_SPIKE_AUROC = 0.985
const MAXIMUM_VOLTAGE_RMSE_MV = 1.0
const MAXIMUM_NMDA_NORMALIZED_RMSE = 1.0

const PAPER_TRAIN_POOL_TRIALS = 50_000
const PAPER_HELDOUT_TRIALS = 2_000
const PAPER_TRIAL_DURATION_MS = 10_000.0
const PAPER_NMDA_REGIONS = 4

const _MODEL_SOURCE =
    joinpath(@__DIR__, "PaperELMTwinOfficialV2.jl")
const _ATTESTATION_SOURCE =
    joinpath(@__DIR__, "PaperELMTwinOfficialV2ReleaseAttestationFinal.jl")

@inline _is_sha256(value) =
    occursin(r"^[0-9a-f]{64}$", String(value))

function _sha256(value, label)
    text = lowercase(String(value))
    _is_sha256(text) ||
        throw(ArgumentError("$label must be a lowercase SHA-256"))
    return text
end

"""
Manifest-derived identity.  This carries source identity and scale evidence,
never metric values or a verification outcome.

For the current rich64 teacher, callers must leave
`connectivity_paper_scale_verified=false` and retain a nonempty uncertainty
explanation.  A paper-scale artifact additionally needs exact recomputed
counts and duration; setting this evidence flag alone cannot promote it.
"""
struct TeacherReleaseIdentity
    manifest_sha256::String
    teacher_contract_sha256::String
    validation_from_train_indices::Vector{Int64}
    connectivity_paper_scale_verified::Bool
    paper_scale_uncertainty::String
end

function TeacherReleaseIdentity(
    manifest_sha256,
    teacher_contract_sha256,
    validation_from_train_indices;
    connectivity_paper_scale_verified::Bool=false,
    paper_scale_uncertainty::AbstractString=
        "teacher connectivity scale is not independently verified",
)
    validation = Int64.(collect(validation_from_train_indices))
    isempty(validation) &&
        throw(ArgumentError("manifest validation indices are empty"))
    length(unique(validation)) == length(validation) ||
        throw(ArgumentError("manifest validation indices are not unique"))
    all(>(0), validation) ||
        throw(ArgumentError("manifest validation indices must be positive"))
    uncertainty = String(paper_scale_uncertainty)
    if connectivity_paper_scale_verified
        uncertainty in ("", "none") ||
            throw(ArgumentError(
                "verified connectivity must use uncertainty=\"none\"",
            ))
        uncertainty = "none"
    else
        isempty(strip(uncertainty)) &&
            throw(ArgumentError(
                "unverified paper scale needs uncertainty metadata",
            ))
    end
    return TeacherReleaseIdentity(
        _sha256(manifest_sha256, "teacher manifest"),
        _sha256(teacher_contract_sha256, "teacher contract"),
        validation,
        connectivity_paper_scale_verified,
        uncertainty,
    )
end

struct OfficialReleaseSplits
    fit_trial_ids::Vector{Int64}
    validation_trial_ids::Vector{Int64}
    heldout_trial_ids::Vector{Int64}
end

function OfficialReleaseSplits(fit_ids, validation_ids, heldout_ids)
    split = OfficialReleaseSplits(
        Int64.(collect(fit_ids)),
        Int64.(collect(validation_ids)),
        Int64.(collect(heldout_ids)),
    )
    for (label, ids) in (
        ("fit", split.fit_trial_ids),
        ("validation", split.validation_trial_ids),
        ("held-out", split.heldout_trial_ids),
    )
        isempty(ids) && throw(ArgumentError("$label split is empty"))
        all(>(0), ids) ||
            throw(ArgumentError("$label IDs must be positive"))
        length(unique(ids)) == length(ids) ||
            throw(ArgumentError("$label IDs are not unique"))
    end
    isempty(intersect(
        split.fit_trial_ids,
        split.validation_trial_ids,
    )) || throw(ArgumentError("fit/validation splits overlap"))
    isempty(intersect(
        split.fit_trial_ids,
        split.heldout_trial_ids,
    )) || throw(ArgumentError("fit/held-out splits overlap"))
    isempty(intersect(
        split.validation_trial_ids,
        split.heldout_trial_ids,
    )) || throw(ArgumentError("validation/held-out splits overlap"))
    return split
end

"""
Raw held-out evidence.  Metrics and pass flags are intentionally absent.
"""
struct OfficialELMHeldoutSet{I,V,S,N}
    trial_ids::Vector{Int64}
    input::I
    target_voltage::V
    target_spike::S
    target_nmda::N
end

OfficialELMHeldoutSet(ids, input, voltage, spike, nmda) =
    OfficialELMHeldoutSet(
        Int64.(collect(ids)),
        input,
        voltage,
        spike,
        nmda,
    )

struct OfficialELMReleaseAttestation{P}
    payload::P
    attestation_sha256::String
end

struct AttestedOfficialELMRelease{F,A}
    frozen::F
    attestation::A
end

# -- canonical SHA-256 -------------------------------------------------------

function _text!(context, text)
    bytes = codeunits(String(text))
    SHA.update!(context, codeunits(string(length(bytes))))
    SHA.update!(context, UInt8[0x3a])
    SHA.update!(context, bytes)
end

function _canon!(context, value)
    if value === nothing
        _text!(context, "nothing")
    elseif value === missing
        _text!(context, "missing")
    elseif value isa Bool
        _text!(context, value ? "bool:1" : "bool:0")
    elseif value isa Integer
        _text!(context, "int:" * string(typeof(value)) * ":" * string(value))
    elseif value isa AbstractFloat
        isfinite(value) || error("non-finite canonical payload float")
        _text!(
            context,
            "float:" * string(typeof(value)) * ":" * bitstring(value),
        )
    elseif value isa AbstractString
        _text!(context, "string")
        _text!(context, value)
    elseif value isa Symbol
        _text!(context, "symbol")
        _text!(context, String(value))
    elseif value isa NamedTuple
        _text!(context, "namedtuple:" * string(length(value)))
        for name in keys(value)
            _text!(context, String(name))
            _canon!(context, getproperty(value, name))
        end
    elseif value isa Tuple
        _text!(context, "tuple:" * string(length(value)))
        foreach(child -> _canon!(context, child), value)
    elseif value isa AbstractDict
        keys_sorted = sort!(collect(keys(value)); by=string)
        _text!(context, "dict:" * string(length(keys_sorted)))
        for key in keys_sorted
            _canon!(context, key)
            _canon!(context, value[key])
        end
    elseif value isa AbstractArray
        _text!(context, "array:" * string(eltype(value)))
        _canon!(context, Tuple(size(value)))
        if isbitstype(eltype(value))
            ENDIAN_BOM == 0x04030201 ||
                error("canonical encoder requires little-endian host")
            bytes = reinterpret(UInt8, vec(Array(value)))
            _text!(context, "bytes:" * string(length(bytes)))
            SHA.update!(context, bytes)
        else
            _text!(context, "elements:" * string(length(value)))
            foreach(child -> _canon!(context, child), value)
        end
    elseif isstructtype(typeof(value))
        _text!(
            context,
            "struct:" * string(parentmodule(typeof(value))) * "." *
            string(nameof(typeof(value))),
        )
        for name in fieldnames(typeof(value))
            _text!(context, String(name))
            _canon!(context, getfield(value, name))
        end
    else
        error("unsupported canonical type $(typeof(value))")
    end
    return context
end

function canonical_sha256(value)
    context = SHA.SHA2_256_CTX()
    _canon!(context, value)
    return bytes2hex(SHA.digest!(context))
end

_file_sha(path) = bytes2hex(SHA.sha256(read(path)))

function release_artifact_payload_sha256(frozen)
    frozen isa Twin.FrozenOfficialELMTwin ||
        error("release requires PaperELMTwinOfficialV2 exact frozen type")
    return canonical_sha256((;
        model=frozen.model,
        parameters=frozen.parameters,
        normalizer=frozen.normalizer,
        metadata=frozen.metadata,
        stored_parameter_sha256=frozen.parameter_sha256,
        stored_artifact_sha256=frozen.artifact_sha256,
    ))
end

# -- independent held-out evaluation ---------------------------------------

function _validate!(
    frozen,
    heldout::OfficialELMHeldoutSet,
    identity::TeacherReleaseIdentity,
    splits::OfficialReleaseSplits,
)
    frozen isa Twin.FrozenOfficialELMTwin ||
        error("release requires PaperELMTwinOfficialV2 exact frozen type")
    Twin.assert_frozen_official_elm_unchanged(frozen)
    identity.validation_from_train_indices ==
        splits.validation_trial_ids ||
        error(
            "validation IDs differ from manifest " *
            "validation_from_train_indices",
        )
    heldout.trial_ids == splits.heldout_trial_ids ||
        error("held-out trial order differs from split evidence")
    input = heldout.input
    ndims(input) == 3 ||
        throw(DimensionMismatch("input must be feature x time x trial"))
    features, steps, trials = size(input)
    config = frozen.model.config
    features == config.num_input ||
        throw(DimensionMismatch("held-out input feature count differs"))
    trials == length(heldout.trial_ids) ||
        throw(DimensionMismatch("held-out trial count differs"))
    size(heldout.target_voltage) == (steps, trials) ||
        throw(DimensionMismatch("voltage target shape differs"))
    size(heldout.target_spike) == (steps, trials) ||
        throw(DimensionMismatch("spike target shape differs"))
    size(heldout.target_nmda) ==
        (config.nmda_regions, steps, trials) ||
        throw(DimensionMismatch("NMDA target shape differs"))
    all(isfinite, input) || error("held-out input is non-finite")
    all(isfinite, heldout.target_voltage) ||
        error("held-out voltage target is non-finite")
    all(isfinite, heldout.target_nmda) ||
        error("held-out NMDA target is non-finite")
    all(value -> value == 0 || value == 1, heldout.target_spike) ||
        error("spike target must be exactly zero or one")
    return (; steps, trials)
end

function _auroc(scores, labels)
    score = Float64.(vec(scores))
    label = Bool.(vec(labels))
    positives = count(identity, label)
    negatives = length(label) - positives
    positives > 0 && negatives > 0 || return NaN
    order = sortperm(score; alg=MergeSort)
    rank_sum = 0.0
    first = 1
    while first <= length(order)
        last = first
        while last < length(order) &&
              score[order[last + 1]] == score[order[first]]
            last += 1
        end
        average_rank = (first + last) / 2
        for position in first:last
            label[order[position]] && (rank_sum += average_rank)
        end
        first = last + 1
    end
    return (
        rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)
end

function _correlation(left, right)
    x = Float64.(vec(left))
    y = Float64.(vec(right))
    x .-= mean(x)
    y .-= mean(y)
    denominator = sqrt(sum(abs2, x) * sum(abs2, y))
    denominator > eps(Float64) || return NaN
    return sum(x .* y) / denominator
end

function recompute_heldout_metrics(frozen, heldout::OfficialELMHeldoutSet)
    frozen isa Twin.FrozenOfficialELMTwin ||
        error("metric evaluator requires exact official ELM frozen type")
    prediction = Twin.twin_forward(frozen, heldout.input)
    voltage_error =
        Float64.(prediction.voltage) .-
        Float64.(heldout.target_voltage)
    regions = size(heldout.target_nmda, 1)
    raw_rmse = Vector{Float64}(undef, regions)
    normalized_rmse = Vector{Float64}(undef, regions)
    correlations = Vector{Float64}(undef, regions)
    for region in 1:regions
        predicted = @view prediction.nmda[region, :, :]
        target = @view heldout.target_nmda[region, :, :]
        error = Float64.(predicted) .- Float64.(target)
        centered_target = Float64.(target)
        centered_target .-= mean(centered_target)
        variance_sum = sum(abs2, centered_target)
        raw_rmse[region] = sqrt(mean(abs2, error))
        normalized_rmse[region] = variance_sum > eps(Float64) ?
            sqrt(sum(abs2, error) / variance_sum) : Inf
        correlations[region] = _correlation(predicted, target)
    end
    positives = count(==(1), heldout.target_spike)
    observations = length(heldout.target_spike)
    return (;
        spike_auroc=_auroc(
            prediction.spike_logit,
            heldout.target_spike,
        ),
        voltage_rmse_mv=sqrt(mean(abs2, voltage_error)),
        voltage_correlation=_correlation(
            prediction.voltage,
            heldout.target_voltage,
        ),
        nmda_raw_rmse_by_region=raw_rmse,
        nmda_normalized_rmse_by_region=normalized_rmse,
        nmda_correlation_by_region=correlations,
        observations,
        spike_positives=positives,
        spike_negatives=observations - positives,
    )
end

function _gate(metrics)
    finite =
        isfinite(metrics.spike_auroc) &&
        isfinite(metrics.voltage_rmse_mv) &&
        isfinite(metrics.voltage_correlation) &&
        all(isfinite, metrics.nmda_raw_rmse_by_region) &&
        all(isfinite, metrics.nmda_normalized_rmse_by_region) &&
        all(isfinite, metrics.nmda_correlation_by_region)
    passed =
        finite &&
        metrics.spike_auroc >= MINIMUM_SPIKE_AUROC &&
        metrics.voltage_rmse_mv <= MAXIMUM_VOLTAGE_RMSE_MV &&
        all(
            <=(MAXIMUM_NMDA_NORMALIZED_RMSE),
            metrics.nmda_normalized_rmse_by_region,
        )
    return (;
        minimum_spike_auroc=MINIMUM_SPIKE_AUROC,
        maximum_voltage_rmse_mv=MAXIMUM_VOLTAGE_RMSE_MV,
        maximum_regional_nmda_normalized_rmse=
            MAXIMUM_NMDA_NORMALIZED_RMSE,
        passed,
    )
end

function _split_payload(frozen, identity, splits, steps)
    duration = steps * Float64(frozen.model.config.delta_t_ms)
    train_pool =
        length(splits.fit_trial_ids) +
        length(splits.validation_trial_ids)
    return (;
        fit_count=length(splits.fit_trial_ids),
        validation_count=length(splits.validation_trial_ids),
        train_pool_count=train_pool,
        heldout_count=length(splits.heldout_trial_ids),
        time_steps=steps,
        heldout_observation_bins=
            steps * length(splits.heldout_trial_ids),
        delta_t_ms=Float64(frozen.model.config.delta_t_ms),
        trial_duration_ms=duration,
        validation_from_train_indices=
            copy(splits.validation_trial_ids),
        validation_from_train_indices_sha256=
            canonical_sha256(splits.validation_trial_ids),
        fit_ids_sha256=canonical_sha256(splits.fit_trial_ids),
        heldout_ids_sha256=
            canonical_sha256(splits.heldout_trial_ids),
        split_membership_sha256=canonical_sha256((;
            fit=splits.fit_trial_ids,
            validation=splits.validation_trial_ids,
            heldout=splits.heldout_trial_ids,
        )),
        connectivity_paper_scale_verified=
            identity.connectivity_paper_scale_verified,
        paper_scale_uncertainty=identity.paper_scale_uncertainty,
    )
end

function _paper_scale(split, frozen)
    return (
        split.train_pool_count == PAPER_TRAIN_POOL_TRIALS &&
        split.heldout_count == PAPER_HELDOUT_TRIALS &&
        split.trial_duration_ms == PAPER_TRIAL_DURATION_MS &&
        split.connectivity_paper_scale_verified &&
        frozen.model.config.nmda_regions == PAPER_NMDA_REGIONS
    )
end

function _model_structure_hash(frozen)
    model = frozen.model
    return canonical_sha256((;
        config=model.config,
        input_indices=model.input_indices,
        valid_indices_mask=model.valid_indices_mask,
        initial_proto_tau_m=model.initial_proto_tau_m,
        kappa_b=model.kappa_b,
    ))
end

function _heldout_hash(heldout)
    return canonical_sha256((;
        trial_ids=heldout.trial_ids,
        input=heldout.input,
        voltage=heldout.target_voltage,
        spike=heldout.target_spike,
        nmda=heldout.target_nmda,
    ))
end

function _payload(frozen, heldout, identity, splits)
    dimensions = _validate!(frozen, heldout, identity, splits)
    metrics = recompute_heldout_metrics(frozen, heldout)
    fixed_gate = _gate(metrics)
    split = _split_payload(
        frozen,
        identity,
        splits,
        dimensions.steps,
    )
    paper_scale = _paper_scale(split, frozen)
    promotable = fixed_gate.passed && paper_scale
    parameter_hash =
        Twin.official_parameter_sha256(frozen.parameters)
    official_artifact_hash = Twin.official_artifact_sha256(
        frozen.model,
        frozen.parameters,
        frozen.normalizer,
    )
    return (;
        schema=RELEASE_ATTESTATION_SCHEMA,
        canonical_encoding=RELEASE_CANONICAL_ENCODING,
        artifact_kind=RELEASE_ARTIFACT_KIND,
        teacher_contract=(;
            manifest_sha256=identity.manifest_sha256,
            teacher_contract_sha256=
                identity.teacher_contract_sha256,
        ),
        model_contract=(;
            family="Paper-ELM-v2-OfficialRouting-Twin",
            input_dim=frozen.model.config.num_input,
            branches=frozen.model.config.num_branch,
            synapses_per_branch=
                frozen.model.config.num_synapse_per_branch,
            memory_units=frozen.model.config.num_memory,
            hidden_size=frozen.model.config.hidden_size,
            nmda_regions=frozen.model.config.nmda_regions,
            routing=frozen.model.config.input_to_synapse_routing,
        ),
        hashes=(;
            teacher_manifest_sha256=identity.manifest_sha256,
            teacher_contract_sha256=
                identity.teacher_contract_sha256,
            upstream_source_manifest_sha256=canonical_sha256(
                Twin.official_elm_source_metadata(),
            ),
            model_implementation_source_sha256=
                _file_sha(_MODEL_SOURCE),
            attestation_implementation_source_sha256=
                _file_sha(_ATTESTATION_SOURCE),
            model_config_sha256=
                canonical_sha256(frozen.model.config),
            model_structure_sha256=
                _model_structure_hash(frozen),
            parameter_sha256=parameter_hash,
            normalizer_sha256=
                canonical_sha256(frozen.normalizer),
            official_artifact_sha256=official_artifact_hash,
            artifact_payload_sha256=
                release_artifact_payload_sha256(frozen),
            heldout_dataset_sha256=_heldout_hash(heldout),
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
            unpublished_checkpoint_identity_claimed=false,
        ),
        metrics,
        fixed_gate,
        outcome=(;
            gate_passed=fixed_gate.passed,
            paper_scale,
            promotable_production=promotable,
            development_scale=!paper_scale,
            caller_supplied_metrics_accepted=false,
            caller_supplied_pass_flag_accepted=false,
            verification_origin=
                "recomputed_from_frozen_model_and_raw_heldout_targets",
        ),
    )
end

"""
Construct a release attestation by running the frozen model.

There are intentionally no metrics, threshold, passed, paper-scale, or
promotable arguments.
"""
function attest_official_elm_release(
    frozen,
    heldout::OfficialELMHeldoutSet,
    identity::TeacherReleaseIdentity,
    splits::OfficialReleaseSplits,
)
    payload = _payload(frozen, heldout, identity, splits)
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
    canonical_sha256(stored.payload) ==
        stored.attestation_sha256 ||
        error("stored attestation digest mismatch")
    expected = attest_official_elm_release(
        bundle.frozen,
        heldout,
        identity,
        splits,
    )
    stored.attestation_sha256 ==
        expected.attestation.attestation_sha256 ||
        error("attestation differs from recomputed evidence")
    canonical_sha256(stored.payload) ==
        canonical_sha256(expected.attestation.payload) ||
        error("attestation logical payload differs")
    stored.payload.hashes.parameter_sha256 ==
        bundle.frozen.parameter_sha256 ||
        error("attested parameter hash differs")
    stored.payload.hashes.official_artifact_sha256 ==
        bundle.frozen.artifact_sha256 ||
        error("attested official artifact hash differs")
    require_gate &&
        stored.payload.outcome.gate_passed !== true &&
        error("held-out voltage/spike/NMDA release gate failed")
    require_production &&
        stored.payload.outcome.promotable_production !== true &&
        error("development-scale release is not production/promotable")
    return bundle
end

function save_attested_official_elm_release(path, bundle)
    bundle.frozen isa Twin.FrozenOfficialELMTwin ||
        error("cannot save a non-official ELM release")
    Twin.assert_frozen_official_elm_unchanged(bundle.frozen)
    canonical_sha256(bundle.attestation.payload) ==
        bundle.attestation.attestation_sha256 ||
        error("refusing to save corrupt attestation")
    isdir(dirname(abspath(path))) || mkpath(dirname(abspath(path)))
    jldsave(
        path;
        artifact_kind=RELEASE_ARTIFACT_KIND,
        format_version=RELEASE_ARTIFACT_FORMAT_VERSION,
        bundle,
    )
    return abspath(path)
end

function _load(path)
    isfile(path) || throw(ArgumentError("release not found: $path"))
    data = JLD2.load(path)
    get(data, "artifact_kind", nothing) == RELEASE_ARTIFACT_KIND ||
        error("wrong release artifact kind")
    get(data, "format_version", nothing) ==
        RELEASE_ARTIFACT_FORMAT_VERSION ||
        error("wrong release artifact version")
    bundle = get(data, "bundle", nothing)
    bundle isa AttestedOfficialELMRelease ||
        error("wrong attested release payload type")
    return bundle
end

"""
Integrity loader: recomputes every hash and metric, but may return a failed or
development-scale artifact for diagnostics.  It never promotes it.
"""
function load_checked_official_elm_release(
    path,
    heldout::OfficialELMHeldoutSet,
    identity::TeacherReleaseIdentity,
    splits::OfficialReleaseSplits,
)
    return verify_official_elm_release(
        _load(path),
        heldout,
        identity,
        splits,
    )
end

"""
Verified loader: always requires the immutable held-out gate.  Production is
required by default.  rich64 can be opened only with the explicit
`require_production=false`, and remains non-promotable in its payload.
"""
function load_verified_official_elm_release(
    path,
    heldout::OfficialELMHeldoutSet,
    identity::TeacherReleaseIdentity,
    splits::OfficialReleaseSplits;
    require_production::Bool=true,
)
    return verify_official_elm_release(
        _load(path),
        heldout,
        identity,
        splits;
        require_gate=true,
        require_production,
    )
end

end # module PaperELMTwinOfficialV2ReleaseAttestationFinal
