module DistillationDatasetBridgeReleaseOrdered

using JLD2
using JSON3

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(_PARENT_MODULE, :DistillationDatasetBridgeReleaseFinal)
    Base.include(
        _PARENT_MODULE,
        joinpath(
            @__DIR__,
            "prepare_distillation_dataset_release_final.jl",
        ),
    )
end

using ..DistillationDatasetBridgeReleaseFinal
using ..PaperDigitalTwin

const FinalBridge = DistillationDatasetBridgeReleaseFinal
const V6 = FinalBridge.V6
const Legacy = FinalBridge.Legacy
const BaseBridge = FinalBridge.BaseBridge
const ReleaseStreamingPrepareConfig =
    FinalBridge.ReleaseStreamingPrepareConfig

export FINAL_NEURON_SCHEMA,
    RELEASE_DATASET_SCHEMA,
    RELEASE_SHARD_SCHEMA,
    ReleaseStreamingPrepareConfig,
    prepare_distillation_dataset_release,
    main

const FINAL_NEURON_SCHEMA = FinalBridge.FINAL_NEURON_SCHEMA
const RELEASE_DATASET_SCHEMA = FinalBridge.RELEASE_DATASET_SCHEMA
const RELEASE_SHARD_SCHEMA = FinalBridge.RELEASE_SHARD_SCHEMA
const EVENT_ORDER = "time_then_axon"

function _ordered_compact(compact)
    offsets = Int64.(compact.event_trial_offset)
    event_axon = similar(compact.event_axon)
    event_time_bin = similar(compact.event_time_bin)
    event_count = similar(compact.event_count)
    for trial in 1:(length(offsets) - 1)
        first_event = Int(offsets[trial]) + 1
        last_event = Int(offsets[trial + 1])
        first_event > last_event && continue
        source_range = first_event:last_event
        order = sortperm(
            source_range;
            by=index -> (
                compact.event_time_bin[index],
                compact.event_axon[index],
            ),
            alg=Base.Sort.MergeSort,
        )
        for (destination, offset) in
            enumerate(order)
            source = source_range[offset]
            target = first_event + destination - 1
            event_axon[target] = compact.event_axon[source]
            event_time_bin[target] =
                compact.event_time_bin[source]
            event_count[target] = compact.event_count[source]
        end
        issorted(
            zip(
                @view(event_time_bin[source_range]),
                @view(event_axon[source_range]),
            ),
        ) || error("failed to establish time/axon event order")
    end
    return merge(
        compact,
        (;
            event_axon,
            event_time_bin,
            event_count,
            event_order=EVENT_ORDER,
        ),
    )
end

# The original helper is intentionally generic.  This AbstractString-specialized
# method is a non-overwriting extension selected by the release path.
function FinalBridge._write_release_shard(
    staging::AbstractString,
    shard_index,
    source,
    source_shard,
    selected_trials,
    split_code,
    source_ids,
    global_indices,
    diagnostic_segments,
    diagnostic_times,
    selected_segments,
    voltage,
    spike,
    spike_logit,
    nmda,
    calcium,
    dendritic,
    frozen,
    twin_file_hash,
    segment_catalog_sha256,
    config_sha256,
)
    compact = _ordered_compact(
        V6._compact_slice(source_shard, selected_trials),
    )
    shard_name =
        "distillation_release_" *
        lpad(shard_index, 6, '0') * ".jld2"
    shard_path = joinpath(staging, shard_name)
    source_layout = V6._validate_final_shard(source_shard)
    dataset = (;
        schema=RELEASE_SHARD_SCHEMA,
        input_representation=
            "compact_ragged_contact_event_location_v2",
        input=nothing,
        compact...,
        target_voltage=voltage,
        target_spike=spike,
        target_spike_logit=spike_logit,
        target_nmda=nmda,
        target_calcium_event=calcium,
        target_dendritic_voltage=dendritic,
        split_code,
        source_split_code=
            source_layout.split_code[selected_trials],
        source_sample_indices=source_ids,
        global_output_indices=global_indices,
        diagnostic_segment_indices=Int32.(diagnostic_segments),
        diagnostic_time_indices=diagnostic_times,
        selected_dendritic_segments=Int32.(selected_segments),
        mixed_supervision=true,
        digital_twin_gate_passed=true,
        frozen_twin_parameter_hash=frozen.parameter_sha256,
        frozen_twin_artifact_hash=frozen.artifact_sha256,
        frozen_twin_file_sha256=twin_file_hash,
        detailed_teacher_hash=source.detailed_teacher_hash,
        detailed_kernel_hash=source.detailed_kernel_hash,
        morphology_hash=source.morphology_hash,
        official_modeldb_source_hash=source.modeldb_hash,
        segment_catalog_sha256,
        dt_ms=frozen.model.config.dt_ms,
        time_steps=size(voltage, 1),
        source_teacher_schema=FINAL_NEURON_SCHEMA,
        config_sha256,
    )
    Legacy._atomic_shard(shard_path, dataset)
    return (;
        path=shard_name,
        sha256=BaseBridge._sha256_file(shard_path),
        bytes=filesize(shard_path),
        samples=length(selected_trials),
        global_first=Int(first(global_indices)),
        global_last=Int(last(global_indices)),
        split_counts=(;
            train=count(==(BaseBridge.TRAIN_SPLIT), split_code),
            validation=count(
                ==(BaseBridge.VALIDATION_SPLIT),
                split_code,
            ),
            test=count(==(BaseBridge.TEST_SPLIT), split_code),
        ),
    )
end

function FinalBridge._release_manifest(
    config::ReleaseStreamingPrepareConfig,
    source,
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
    arguments = (
        config,
        source,
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
    base = invoke(
        FinalBridge._release_manifest,
        NTuple{20,Any},
        arguments...,
    )
    return merge(
        base,
        (;
            event_order=EVENT_ORDER,
            event_window_access=
                "binary-search event_time_bin within each trial offset",
        ),
    )
end

prepare_distillation_dataset_release(
    config::ReleaseStreamingPrepareConfig,
) = FinalBridge.prepare_distillation_dataset_release(config)

function main(arguments=ARGS)
    report = prepare_distillation_dataset_release(
        V6._parse_arguments(arguments),
    )
    println(JSON3.write(report))
    return report
end

end # module DistillationDatasetBridgeReleaseOrdered

if abspath(PROGRAM_FILE) == @__FILE__
    DistillationDatasetBridgeReleaseOrdered.main()
end
