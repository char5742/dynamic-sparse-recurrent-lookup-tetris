module DistillationDatasetBridgeReleaseProduction

using JSON3

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(_PARENT_MODULE, :DistillationDatasetBridgeReleaseOrdered)
    Base.include(
        _PARENT_MODULE,
        joinpath(
            @__DIR__,
            "prepare_distillation_dataset_release_ordered.jl",
        ),
    )
end

using ..DistillationDatasetBridgeReleaseOrdered
using ..PaperDigitalTwin

const OrderedBridge = DistillationDatasetBridgeReleaseOrdered
const FinalBridge = OrderedBridge.FinalBridge
const V6 = OrderedBridge.V6
const BaseBridge = OrderedBridge.BaseBridge
const ReleaseStreamingPrepareConfig =
    OrderedBridge.ReleaseStreamingPrepareConfig

export FINAL_NEURON_SCHEMA,
    RELEASE_DATASET_SCHEMA,
    RELEASE_SHARD_SCHEMA,
    ReleaseStreamingPrepareConfig,
    prepare_distillation_dataset_release,
    main

const FINAL_NEURON_SCHEMA = OrderedBridge.FINAL_NEURON_SCHEMA
const RELEASE_DATASET_SCHEMA = OrderedBridge.RELEASE_DATASET_SCHEMA
const RELEASE_SHARD_SCHEMA = OrderedBridge.RELEASE_SHARD_SCHEMA

# This two-argument-specialized extension fixes the final manifest binding while
# remaining more specific than the event-order extension and without replacing
# either existing method.
function FinalBridge._release_manifest(
    config::ReleaseStreamingPrepareConfig,
    source::BaseBridge._Source,
    source_counts,
    plan_data,
    frozen,
    integrity_before,
    integrity_after,
    reported_metrics,
    recomputed_gate,
    twin_file_hash,
    config_sha256,
    segment_catalog_sha256,
    selected_segments,
    diagnostic_segments,
    diagnostic_times,
    train_indices,
    validation_indices,
    test_indices,
    shard_records,
    promotion_eligible,
)
    segment_region =
        BaseBridge._segment_regions(source, :official_neuron)
    input_compartment, input_receptor, input_plane =
        BaseBridge._input_anatomy(frozen.model.config)
    split_counts = (;
        train=length(train_indices),
        validation=length(validation_indices),
        test=length(test_indices),
    )
    source_hashes = V6._required(source.manifest, :source_hashes)
    final_generator_sha256 = String(V6._value(
        source_hashes,
        :final_generator_source_sha256,
        "",
    ))
    hashes = (;
        official_modeldb_source_hash=source.modeldb_hash,
        detailed_teacher_hash=source.detailed_teacher_hash,
        detailed_kernel_hash=source.detailed_kernel_hash,
        morphology_hash=source.morphology_hash,
        final_generator_sha256,
        segment_catalog_sha256,
        frozen_twin_parameter_hash=frozen.parameter_sha256,
        frozen_twin_artifact_hash=frozen.artifact_sha256,
        frozen_twin_file_sha256=twin_file_hash,
        source_dataset_hash=source.source_dataset_hash,
        source_manifest_sha256=source.source_manifest_hash,
        config_sha256,
    )
    return (;
        schema=RELEASE_DATASET_SCHEMA,
        shard_schema=RELEASE_SHARD_SCHEMA,
        model_name=HD_SWSNN_TWINPROP_NAME,
        completion_state="complete",
        promotion_eligible,
        created_at_utc=FinalBridge.Dates.format(
            FinalBridge.now(FinalBridge.UTC),
            FinalBridge.dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
        source_kind="official_neuron_final_v2",
        source_teacher_schema=FINAL_NEURON_SCHEMA,
        official_neuron_schema=FINAL_NEURON_SCHEMA,
        source_completion_state=source.completion_state,
        source_public_counts=(;
            train_pool=V6.PAPER_TRAIN_POOL,
            held_out_test=V6.PAPER_HELD_OUT_TEST,
            duration_ms=V6.PAPER_DURATION_MS,
        ),
        observed_source_counts=source_counts,
        validation_derivation=(;
            algorithm="lowest SHA-256(seed:source_sample_id)",
            seed=config.validation_hash_seed,
            requested=config.validation_samples,
            selected=length(plan_data.validation_ids),
            source_train_pool=plan_data.source_train_pool,
            selected_source_sample_ids=plan_data.validation_ids,
            selected_ids_sha256=plan_data.validation_digest,
        ),
        input_representation=
            "compact_ragged_contact_event_location_v2",
        event_order=OrderedBridge.EVENT_ORDER,
        event_window_access=
            "binary-search event_time_bin within each trial offset",
        input_layout=twin_input_layout(frozen),
        input_compartment,
        input_receptor,
        input_plane,
        time_steps=plan_data.time_steps,
        diagnostic_time_indices=diagnostic_times,
        total_samples=
            length(train_indices) +
            length(validation_indices) +
            length(test_indices),
        split_counts,
        train_indices,
        validation_indices,
        test_indices,
        time_chunk=config.time_chunk,
        output_shard_samples=config.output_shard_samples,
        peak_dense_chunk_bytes=
            frozen.model.config.input_dim *
            min(config.time_chunk, plan_data.time_steps) *
            sizeof(Float32),
        dense_memory_scales_with_total_samples=false,
        mixed_supervision=true,
        mixed_supervision_provenance=(;
            target_voltage=
                "actual frozen PaperDigitalTwin inference",
            target_spike=
                "actual frozen PaperDigitalTwin probability",
            target_spike_logit=
                "actual frozen PaperDigitalTwin logit",
            target_nmda=
                "actual frozen PaperDigitalTwin inference",
            target_calcium_event=
                "official detailed Hay/NEURON teacher only",
            target_dendritic_voltage=
                "official detailed Hay/NEURON teacher only",
        ),
        target_schema=V6._target_schema(
            plan_data.time_steps,
            diagnostic_times,
        ),
        digital_twin_gate_passed=true,
        twin_self_report_trusted=false,
        twin_artifact_reported_metrics=reported_metrics,
        recomputed_twin_gate=recomputed_gate,
        integrity_before,
        integrity_after,
        diagnostic_segment_indices=diagnostic_segments,
        selected_dendritic_segments=selected_segments,
        selected_dendritic_semantics=(
            "distal_basal",
            "proximal_apical_trunk",
            "apical_calcium_hot_zone",
            "distal_apical_tuft",
        ),
        segment_region,
        segment_catalog_sha256,
        teacher_hash=source.detailed_teacher_hash,
        detailed_teacher_hash=source.detailed_teacher_hash,
        detailed_kernel_hash=source.detailed_kernel_hash,
        cell_mechanism_sha256=source.detailed_kernel_hash,
        morphology_hash=source.morphology_hash,
        official_modeldb_source_hash=source.modeldb_hash,
        frozen_twin_parameter_hash=frozen.parameter_sha256,
        frozen_twin_artifact_hash=frozen.artifact_sha256,
        frozen_twin_file_sha256=twin_file_hash,
        digital_twin_hash=frozen.artifact_sha256,
        source_dataset_hash=source.source_dataset_hash,
        source_manifest_sha256=source.source_manifest_hash,
        config_sha256,
        hashes,
        shards=shard_records,
    )
end

prepare_distillation_dataset_release(
    config::ReleaseStreamingPrepareConfig,
) = OrderedBridge.prepare_distillation_dataset_release(config)

function main(arguments=ARGS)
    report = prepare_distillation_dataset_release(
        V6._parse_arguments(arguments),
    )
    println(JSON3.write(report))
    return report
end

end # module DistillationDatasetBridgeReleaseProduction

if abspath(PROGRAM_FILE) == @__FILE__
    DistillationDatasetBridgeReleaseProduction.main()
end
