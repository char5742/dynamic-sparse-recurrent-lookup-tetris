module StreamingOfficialELMReleaseDatasetV3

# Exact final.v2 sealed-release reader.  The V2 reader remains a compatibility
# diagnostic for the superseded final.v1 sealed type; this module never aliases
# or accepts that type.

using JLD2

const _PARENT = parentmodule(@__MODULE__)
if !isdefined(_PARENT, :StreamingOfficialELMReleaseDatasetV2)
    Base.include(
        _PARENT,
        joinpath(
            @__DIR__,
            "StreamingOfficialELMReleaseDatasetV2.jl",
        ),
    )
end
if !isdefined(Main, :PaperELMTwinOfficialV2SealedReleaseV2)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "PaperELMTwinOfficialV2SealedReleaseV2.jl",
        ),
    )
end

const Compat =
    getfield(_PARENT, :StreamingOfficialELMReleaseDatasetV2)
const Sealed = Main.PaperELMTwinOfficialV2SealedReleaseV2
const Twin = Sealed.Twin
const BaseStream = Compat.BaseStream
const Signed = Compat.Signed
const StreamDataset = BaseStream.StreamDataset

export OFFICIAL_ELM_INPUT_DIM,
    SEALED_EXECUTION_TYPE,
    open_sealed_stream_dataset,
    sealed_stream_materialize_window,
    stream_dataset_integrity!,
    stream_target_statistics,
    verify_primary_cache_against_live_sealed_elm!

const OFFICIAL_ELM_INPUT_DIM = 1_278
const SEALED_EXECUTION_TYPE =
    "PaperELMTwinOfficialV2SealedReleaseV2.SealedOfficialELMRelease"
const stream_dataset_integrity! =
    BaseStream.stream_dataset_integrity!
const stream_target_statistics =
    BaseStream.stream_target_statistics

@inline _get(object, name::Symbol, default=nothing) =
    Compat._get(object, name, default)
_required(object, name::Symbol) =
    Compat._required(object, name)
const _assert_sha = Compat._assert_sha
const _assert_time_contract = Compat._assert_time_contract
const _assert_input_contract = Compat._assert_input_contract
const _assert_primary_claim = Compat._assert_primary_claim
const _validate_logit_target = Compat._validate_logit_target
const _bit_exact = Compat._bit_exact

function _verify_bundle_and_manifest!(
    bundle::Sealed.SealedOfficialELMRelease,
    manifest,
    source_teacher_manifest,
    source_teacher_shard_directory;
    require_production,
    scratch_root,
)
    Sealed.verify_sealed_official_elm_release(
        bundle,
        source_teacher_manifest,
        source_teacher_shard_directory;
        require_gate=true,
        require_production,
        scratch_root,
    )
    frozen = bundle.frozen
    frozen isa Twin.FrozenOfficialELMTwin ||
        error("sealed V2 bundle does not own the canonical frozen type")
    Twin.assert_frozen_official_elm_unchanged(frozen)
    payload = bundle.attestation.payload
    String(_required(manifest, :digital_twin_type)) ==
        SEALED_EXECUTION_TYPE ||
        error("bridge did not consume exact final.v2 sealed type")
    String(_required(manifest, :digital_twin_schema)) ==
        Sealed.SEALED_RELEASE_SCHEMA ||
        error("bridge sealed-release schema is not final.v2")
    attestation = _assert_sha(
        _required(manifest, :sealed_attestation_sha256),
        "bridge sealed V2 attestation",
    )
    attestation == bundle.attestation.attestation_sha256 ||
        error("bridge/sealed V2 attestation digest mismatch")
    hashes = _required(manifest, :hashes)
    _assert_sha(
        _required(hashes, :sealed_attestation_sha256),
        "nested bridge sealed V2 attestation",
    ) == attestation ||
        error("top-level/nested sealed V2 attestation differs")
    _assert_sha(
        _required(manifest, :source_manifest_sha256),
        "bridge source manifest",
    ) == payload.teacher.manifest_sha256 ||
        error("bridge source manifest differs from sealed V2 teacher")
    _assert_sha(
        _required(manifest, :source_teacher_contract_sha256),
        "bridge source teacher contract",
    ) == payload.teacher.teacher_contract_sha256 ||
        error("bridge source contract differs from sealed V2 teacher")
    Int(payload.model.input_dim) == OFFICIAL_ELM_INPUT_DIM ||
        error("sealed V2 model input_dim differs")
    payload.outcome.gate_passed === true ||
        error("sealed V2 fixed held-out gate failed")
    if require_production
        payload.outcome.promotable_production === true ||
            error("sealed V2 artifact is not production/promotable")
    else
        payload.outcome.development_scale === true ||
            error("non-production bridge is not canonical development scale")
        payload.outcome.promotable_production === false ||
            error("dev1500 bridge must remain non-promotable")
        payload.split.duration_ms == 1_500.0 ||
            error("canonical development bridge is not dev1500")
    end
    return true
end

function _sealed_dataset(
    parsed::StreamDataset,
    bundle::Sealed.SealedOfficialELMRelease,
    time_contract,
)
    provenance = merge(
        parsed.provenance,
        (;
            sealed_execution_type=SEALED_EXECUTION_TYPE,
            sealed_release_schema=Sealed.SEALED_RELEASE_SCHEMA,
            sealed_attestation_sha256=
                bundle.attestation.attestation_sha256,
            official_elm_input_dim=OFFICIAL_ELM_INPUT_DIM,
            official_elm_input_semantics=Signed.INPUT_SEMANTICS,
            source_teacher_contract_sha256=
                bundle.attestation.payload.teacher.
                    teacher_contract_sha256,
            time_contract,
        ),
    )
    return StreamDataset(
        parsed.manifest_path,
        parsed.root,
        parsed.manifest_sha256,
        parsed.dataset_sha256,
        parsed.manifest,
        parsed.records,
        parsed.total_samples,
        parsed.time_steps,
        OFFICIAL_ELM_INPUT_DIM,
        parsed.train_indices,
        parsed.validation_indices,
        parsed.test_indices,
        parsed.split_code,
        parsed.global_to_shard,
        parsed.diagnostic_time_indices,
        parsed.segment_region,
        parsed.segment_catalog_sha256,
        provenance,
        parsed.frozen_twin_file_sha256,
        parsed.verified_shard_hashes,
        parsed.tracker,
    )
end

"""
Open compact shards against the exact final.v2 sealed ELM type.

There is intentionally no overload accepting final.v1 sealed releases, raw
frozen twins, verified wrappers, or caller-supplied evidence.
"""
function open_sealed_stream_dataset(
    path::AbstractString,
    bundle::Sealed.SealedOfficialELMRelease,
    source_teacher_manifest::AbstractString,
    source_teacher_shard_directory::AbstractString;
    minimum_spike_auroc::Real=0.985,
    verify_shard_hashes::Bool=true,
    require_promotion_eligible::Bool=true,
    require_production::Bool=require_promotion_eligible,
    scratch_root=nothing,
)
    require_production == require_promotion_eligible ||
        error("bridge and sealed V2 scale modes must agree")
    adapter = (;
        model=(; config=(; segments=642, input_dim=6 * 642)),
        parameter_sha256=bundle.frozen.parameter_sha256,
        artifact_sha256=bundle.frozen.artifact_sha256,
    )
    parsed = BaseStream.open_stream_dataset(
        path,
        adapter;
        minimum_spike_auroc,
        verify_shard_hashes,
        require_promotion_eligible,
    )
    _verify_bundle_and_manifest!(
        bundle,
        parsed.manifest,
        source_teacher_manifest,
        source_teacher_shard_directory;
        require_production,
        scratch_root,
    )
    _assert_input_contract(parsed.manifest)
    time_contract =
        _assert_time_contract(parsed.manifest, parsed.time_steps)
    _assert_primary_claim(parsed.manifest)
    return _sealed_dataset(parsed, bundle, time_contract)
end

sealed_stream_materialize_window(
    dataset::StreamDataset,
    global_indices,
    first_time::Integer,
    window::Integer,
) = Compat.sealed_stream_materialize_window(
    dataset,
    global_indices,
    first_time,
    window,
)

"""
Re-run every cached primary trajectory through the exact final.v2 bundle.

Only bounded `1278 × time_chunk × 1` dense input is materialized, recurrent
state is carried between chunks, and all samples/time bins must compare with
zero bit-level delta before this function returns.
"""
function verify_primary_cache_against_live_sealed_elm!(
    dataset::StreamDataset,
    bundle::Sealed.SealedOfficialELMRelease;
    time_chunk::Integer=256,
)
    dataset.input_dim == OFFICIAL_ELM_INPUT_DIM ||
        error("live-cache verification requires signed 1278 input")
    chunk = Int(time_chunk)
    chunk >= 1 ||
        throw(ArgumentError("cache verification chunk must be positive"))
    Twin.assert_frozen_official_elm_unchanged(bundle.frozen)
    maxima = zeros(Float64, 4)
    samples_verified = 0
    time_points_verified = 0
    for (shard_index, record) in enumerate(dataset.records)
        shard = BaseStream._load_shard(dataset, shard_index)
        logit = _validate_logit_target(
            shard,
            dataset.time_steps,
            record.samples,
        )
        for local_index in 1:record.samples
            state = nothing
            for first_time in 1:chunk:dataset.time_steps
                last_time = min(
                    first_time + chunk - 1,
                    dataset.time_steps,
                )
                count = last_time - first_time + 1
                raw = zeros(
                    Float32,
                    OFFICIAL_ELM_INPUT_DIM,
                    count,
                    1,
                )
                Signed.fill_official_raw_window!(
                    @view(raw[:, :, 1]),
                    shard,
                    local_index,
                    first_time,
                    last_time,
                )
                live = Twin.twin_forward(
                    bundle.frozen,
                    raw;
                    normalized=false,
                    initial_state=state,
                )
                state = live.final_state
                maxima[1] = max(
                    maxima[1],
                    _bit_exact(
                        "soma voltage",
                        live.voltage,
                        @view(_required(shard, :target_voltage)[
                            first_time:last_time,
                            local_index:local_index,
                        ]),
                    ),
                )
                maxima[2] = max(
                    maxima[2],
                    _bit_exact(
                        "spike probability",
                        live.spike_probability,
                        @view(_required(shard, :target_spike)[
                            first_time:last_time,
                            local_index:local_index,
                        ]),
                    ),
                )
                maxima[3] = max(
                    maxima[3],
                    _bit_exact(
                        "spike logit",
                        live.spike_logit,
                        @view(logit[
                            first_time:last_time,
                            local_index:local_index,
                        ]),
                    ),
                )
                maxima[4] = max(
                    maxima[4],
                    _bit_exact(
                        "regional NMDA current",
                        live.nmda,
                        @view(_required(shard, :target_nmda)[
                            :,
                            first_time:last_time,
                            local_index:local_index,
                        ]),
                    ),
                )
                window_bytes =
                    sizeof(Float32) * (
                        length(raw) +
                        length(live.voltage) +
                        length(live.spike_probability) +
                        length(live.spike_logit) +
                        length(live.nmda)
                    )
                dataset.tracker.peak_dense_window_bytes = max(
                    dataset.tracker.peak_dense_window_bytes,
                    window_bytes,
                )
                dataset.tracker.peak_combined_bytes = max(
                    dataset.tracker.peak_combined_bytes,
                    record.bytes + window_bytes,
                )
                time_points_verified += count
            end
            samples_verified += 1
        end
    end
    samples_verified == dataset.total_samples ||
        error("not every sealed V2 sample was cache-verified")
    time_points_verified ==
        dataset.total_samples * dataset.time_steps ||
        error("not every sealed V2 time point was cache-verified")
    all(iszero, maxima) ||
        error("sealed V2 primary cache verification was not bit exact")
    Twin.assert_frozen_official_elm_unchanged(bundle.frozen)
    return (;
        cache_verified_all_samples=true,
        bit_exact=true,
        samples_verified,
        time_points_verified,
        soma_voltage_max_delta=maxima[1],
        spike_probability_max_delta=maxima[2],
        spike_logit_max_delta=maxima[3],
        nmda_max_delta=maxima[4],
        primary_targets=(
            "soma_voltage",
            "spike_probability",
            "spike_logit",
            "nmda_soma",
            "nmda_basal",
            "nmda_apical_trunk",
            "nmda_apical_tuft",
        ),
        detailed_model_auxiliary_targets=(
            "calcium_event_sparse",
            "dendritic_voltage_sparse",
        ),
        sealed_execution_type=SEALED_EXECUTION_TYPE,
        official_elm_input_dim=OFFICIAL_ELM_INPUT_DIM,
        sealed_release_schema=Sealed.SEALED_RELEASE_SCHEMA,
        sealed_attestation_sha256=
            bundle.attestation.attestation_sha256,
    )
end

end # module StreamingOfficialELMReleaseDatasetV3
