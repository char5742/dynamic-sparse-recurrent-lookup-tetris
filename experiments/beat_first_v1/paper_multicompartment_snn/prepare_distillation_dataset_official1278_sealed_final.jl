module DistillationDatasetBridgeOfficial1278SealedFinal

# Final official-1278 distillation bridge.
#
# The lower sealed bridge deliberately publishes a fail-closed manifest:
# construction wrote the primary targets from a live frozen ELM, but it does
# not claim that the persisted bytes were replay-verified.  This wrapper opens
# those compact shards without trusting that claim, reconstructs every signed
# 1278-channel input, carries the frozen ELM state across bounded time chunks,
# and compares every persisted primary target bit-for-bit.  Only a successful
# all-sample replay atomically promotes the manifest to `bit_exact=true`.

using JSON3

const _PARENT = parentmodule(@__MODULE__)
if !isdefined(
    _PARENT,
    :DistillationDatasetBridgeOfficial1278Sealed,
)
    Base.include(
        _PARENT,
        joinpath(
            @__DIR__,
            "prepare_distillation_dataset_official1278_sealed.jl",
        ),
    )
end
if !isdefined(_PARENT, :StreamingOfficialELMReleaseDatasetV2)
    Base.include(
        _PARENT,
        joinpath(
            @__DIR__,
            "StreamingOfficialELMReleaseDatasetV2.jl",
        ),
    )
end

const Bridge =
    getfield(_PARENT, :DistillationDatasetBridgeOfficial1278Sealed)
const Stream =
    getfield(_PARENT, :StreamingOfficialELMReleaseDatasetV2)
const Sealed = Bridge.Sealed
const Twin = Bridge.OfficialTwin
const BaseStream = Stream.BaseStream
const BaseBridge = Bridge.BaseBridge
const V6 = Bridge.V6
const ReleaseStreamingPrepareConfig =
    Bridge.ReleaseStreamingPrepareConfig

export FINAL_NEURON_SCHEMA,
    RELEASE_DATASET_SCHEMA,
    RELEASE_SHARD_SCHEMA,
    SEALED_RELEASE_SCHEMA,
    OFFICIAL_INPUT_DIM,
    PRIMARY_REPLAY_SCHEMA,
    ReleaseStreamingPrepareConfig,
    prepare_distillation_dataset_release,
    main

const FINAL_NEURON_SCHEMA = Bridge.FINAL_NEURON_SCHEMA
const RELEASE_DATASET_SCHEMA = Bridge.RELEASE_DATASET_SCHEMA
const RELEASE_SHARD_SCHEMA = Bridge.RELEASE_SHARD_SCHEMA
const SEALED_RELEASE_SCHEMA = Bridge.SEALED_RELEASE_SCHEMA
const OFFICIAL_INPUT_DIM = Bridge.OFFICIAL_INPUT_DIM
const PRIMARY_REPLAY_SCHEMA =
    "hd_swsnn.distillation.primary_cache_live_replay.final.v1"

const PRIMARY_TARGETS = (
    "soma_voltage",
    "spike_probability",
    "spike_logit",
    "regional_nmda_current",
)
const DETAILED_AUXILIARY_EXCLUDED = (
    "calcium_event_sparse",
    "dendritic_voltage_sparse",
)

function _source_and_bundle(
    config::ReleaseStreamingPrepareConfig,
)
    source, _, _ =
        Bridge.FinalBridge._load_release_source(config)
    frozen_path = abspath(config.frozen_twin_path)
    frozen = Twin.load_frozen_official_elm(frozen_path)
    Twin.assert_frozen_official_elm_unchanged(frozen)
    Bridge._validate_official_frozen(frozen)
    BaseBridge._expected_hash(
        "twin parameter",
        frozen.parameter_sha256,
        config.expected_twin_parameter_sha256,
    )
    BaseBridge._expected_hash(
        "twin artifact",
        frozen.artifact_sha256,
        config.expected_twin_artifact_sha256,
    )
    bundle = Sealed.attest_sealed_official_elm_release(
        source.manifest_path,
        source.root,
        frozen,
    )
    # This validates the source hashes, model shape and fixed held-out gate
    # without accepting caller-provided metrics or pass flags.
    Bridge._bridge_twin(bundle, source)
    Sealed.verify_sealed_official_elm_release(
        bundle,
        source.manifest_path,
        source.root;
        require_gate=true,
        require_production=config.require_full_public_counts,
    )
    return source, bundle
end

function _open_unpromoted_replay_dataset(
    path::AbstractString,
    bundle::Sealed.SealedOfficialELMRelease,
    source,
    config::ReleaseStreamingPrepareConfig,
)
    adapter = (;
        model=(; config=(;
            segments=642,
            input_dim=6 * 642,
        )),
        parameter_sha256=bundle.frozen.parameter_sha256,
        artifact_sha256=bundle.frozen.artifact_sha256,
    )
    parsed = BaseStream.open_stream_dataset(
        path,
        adapter;
        minimum_spike_auroc=config.minimum_twin_spike_auroc,
        verify_shard_hashes=true,
        require_promotion_eligible=
            config.require_full_public_counts,
    )
    claim = Stream._required(
        parsed.manifest,
        :primary_cache_live_equality,
    )
    Stream._get(claim, :bit_exact, false) === true &&
        error(
            "new bridge output was marked bit_exact before live replay",
        )
    Stream._verify_bundle_and_manifest!(
        bundle,
        parsed.manifest,
        source.manifest_path,
        source.root;
        require_production=config.require_full_public_counts,
        scratch_root=nothing,
    )
    Stream._assert_input_contract(parsed.manifest)
    time_contract =
        Stream._assert_time_contract(
            parsed.manifest,
            parsed.time_steps,
        )
    provenance = merge(
        parsed.provenance,
        (;
            sealed_execution_type=Stream.SEALED_EXECUTION_TYPE,
            sealed_release_schema=Sealed.SEALED_RELEASE_SCHEMA,
            sealed_attestation_sha256=
                bundle.attestation.attestation_sha256,
            official_elm_input_dim=OFFICIAL_INPUT_DIM,
            official_elm_input_semantics=
                Stream.Signed.INPUT_SEMANTICS,
            source_teacher_contract_sha256=
                bundle.attestation.payload.teacher.
                    teacher_contract_sha256,
            time_contract,
        ),
    )
    return Stream.StreamDataset(
        parsed.manifest_path,
        parsed.root,
        parsed.manifest_sha256,
        parsed.dataset_sha256,
        parsed.manifest,
        parsed.records,
        parsed.total_samples,
        parsed.time_steps,
        OFFICIAL_INPUT_DIM,
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

function _replay_result(report)
    report.cache_verified_all_samples === true ||
        error("live replay did not cover every sample")
    report.bit_exact === true ||
        error("live replay was not bit exact")
    maxima = (
        soma_voltage=Float64(
            report.soma_voltage_max_delta,
        ),
        spike_probability=Float64(
            report.spike_probability_max_delta,
        ),
        spike_logit=Float64(report.spike_logit_max_delta),
        regional_nmda_current=Float64(
            report.nmda_max_delta,
        ),
    )
    all(iszero, values(maxima)) ||
        error("live replay has a nonzero primary-target delta")
    return (;
        all_samples=true,
        bit_exact=true,
        samples_verified=Int(report.samples_verified),
        time_points_verified=Int(
            report.time_points_verified,
        ),
        max_absolute_delta=maxima,
    )
end

function _measurement_payload(
    dataset::Stream.StreamDataset,
    bundle::Sealed.SealedOfficialELMRelease,
    replay_report;
    time_chunk::Integer,
)
    result = _replay_result(replay_report)
    result.samples_verified == dataset.total_samples ||
        error("live replay sample count differs from the cache")
    result.time_points_verified ==
        dataset.total_samples * dataset.time_steps ||
        error("live replay time-point count differs from the cache")
    shard_inventory = Tuple((
        path=record.relative_path,
        sha256=record.sha256,
        bytes=record.bytes,
        samples=record.samples,
        global_first=record.global_first,
        global_last=record.global_last,
    ) for record in dataset.records)
    return (;
        schema=PRIMARY_REPLAY_SCHEMA,
        measurement="postwrite_live_replay",
        execution_type=Stream.SEALED_EXECUTION_TYPE,
        sealed_release_schema=Sealed.SEALED_RELEASE_SCHEMA,
        sealed_attestation_sha256=
            bundle.attestation.attestation_sha256,
        source_manifest_sha256=
            bundle.attestation.payload.teacher.manifest_sha256,
        source_teacher_contract_sha256=
            bundle.attestation.payload.teacher.
                teacher_contract_sha256,
        unpromoted_manifest_sha256=dataset.manifest_sha256,
        shard_inventory_sha256=
            Sealed.canonical_sha256(shard_inventory),
        input_reconstruction_sha256=BaseBridge._sha256_file(
            joinpath(
                @__DIR__,
                "OfficialSignedInputReconstruction.jl",
            ),
        ),
        replay_engine_sha256=BaseBridge._sha256_file(
            joinpath(
                @__DIR__,
                "StreamingOfficialELMReleaseDatasetV2.jl",
            ),
        ),
        time_chunk=Int(time_chunk),
        total_samples=dataset.total_samples,
        time_steps=dataset.time_steps,
        targets=PRIMARY_TARGETS,
        detailed_auxiliary_excluded=
            DETAILED_AUXILIARY_EXCLUDED,
        result,
    )
end

function _promoted_manifest(manifest, measurement)
    measurement.schema == PRIMARY_REPLAY_SCHEMA ||
        error("primary replay measurement schema differs")
    measurement.result.all_samples === true ||
        error("primary replay did not cover all samples")
    measurement.result.bit_exact === true ||
        error("primary replay did not establish bit equality")
    all(
        iszero,
        values(measurement.result.max_absolute_delta),
    ) || error("primary replay contains a nonzero delta")
    claim = (;
        required=true,
        all_samples=true,
        bit_exact=true,
        measurement="postwrite_live_replay",
        measurement_schema=PRIMARY_REPLAY_SCHEMA,
        measurement_sha256=
            Sealed.canonical_sha256(measurement),
        measurement_report=measurement,
        samples_verified=
            measurement.result.samples_verified,
        time_points_verified=
            measurement.result.time_points_verified,
        max_absolute_delta=
            measurement.result.max_absolute_delta,
        targets=PRIMARY_TARGETS,
        detailed_auxiliary_excluded=
            DETAILED_AUXILIARY_EXCLUDED,
        sealed_attestation_sha256=
            measurement.sealed_attestation_sha256,
    )
    promoted =
        JSON3.read(JSON3.write(manifest), Dict{String,Any})
    promoted["primary_cache_live_equality"] = claim
    return promoted
end

function _publish_measurement!(
    manifest_path::AbstractString,
    manifest,
    measurement,
)
    promoted = _promoted_manifest(manifest, measurement)
    Stream._assert_primary_claim(promoted)
    Bridge.Legacy._json_write(manifest_path, promoted)
    return BaseBridge._sha256_file(manifest_path)
end

"""
Prepare compact official-1278 shards and live-verify every persisted primary
target before making the dataset consumable.

If replay fails, the lower bridge output remains fail-closed: its manifest has
no `bit_exact=true`, so the sealed reader refuses it.
"""
function prepare_distillation_dataset_release(
    config::ReleaseStreamingPrepareConfig,
)
    base_report =
        Bridge.prepare_distillation_dataset_release(config)
    source, bundle = _source_and_bundle(config)
    dataset = _open_unpromoted_replay_dataset(
        base_report.output_directory,
        bundle,
        source,
        config,
    )
    replay_report =
        Stream.verify_primary_cache_against_live_sealed_elm!(
            dataset,
            bundle;
            time_chunk=config.time_chunk,
        )
    measurement = _measurement_payload(
        dataset,
        bundle,
        replay_report;
        time_chunk=config.time_chunk,
    )
    manifest_sha256 = _publish_measurement!(
        base_report.manifest_path,
        dataset.manifest,
        measurement,
    )

    # Re-open through the strict public reader after promotion.  This checks
    # the exact sealed type, source lineage, input/time contracts, every shard
    # hash and the promoted primary claim.
    published = Stream.open_sealed_stream_dataset(
        base_report.output_directory,
        bundle,
        source.manifest_path,
        source.root;
        minimum_spike_auroc=config.minimum_twin_spike_auroc,
        verify_shard_hashes=true,
        require_promotion_eligible=
            config.require_full_public_counts,
        require_production=config.require_full_public_counts,
    )
    BaseStream.stream_dataset_integrity!(published)
    return merge(
        base_report,
        (;
            manifest_sha256,
            primary_cache_live_equality=true,
            primary_cache_live_replay_sha256=
                Sealed.canonical_sha256(measurement),
            primary_cache_live_replay=measurement,
            peak_replay_dense_window_bytes=
                dataset.tracker.peak_dense_window_bytes,
            peak_replay_combined_bytes=
                dataset.tracker.peak_combined_bytes,
        ),
    )
end

function main(arguments=ARGS)
    report = prepare_distillation_dataset_release(
        V6._parse_arguments(arguments),
    )
    println(JSON3.write(report))
    return report
end

end # module DistillationDatasetBridgeOfficial1278SealedFinal

if abspath(PROGRAM_FILE) == @__FILE__
    DistillationDatasetBridgeOfficial1278SealedFinal.main()
end
