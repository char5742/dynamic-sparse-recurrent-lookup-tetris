module DistillationDatasetBridgeReleaseFinal

using Dates
using JLD2
using JSON3
using SHA

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(_PARENT_MODULE, :DistillationDatasetBridgeReleaseV6)
    Base.include(
        _PARENT_MODULE,
        joinpath(
            @__DIR__,
            "prepare_distillation_dataset_release_v6.jl",
        ),
    )
end

using ..DistillationDatasetBridgeReleaseV6
using ..PaperDigitalTwin

const V6 = DistillationDatasetBridgeReleaseV6
const Legacy = V6.Legacy
const BaseBridge = V6.BaseBridge
const ReleaseStreamingPrepareConfig = V6.ReleaseStreamingPrepareConfig

export FINAL_NEURON_SCHEMA,
    RELEASE_DATASET_SCHEMA,
    RELEASE_SHARD_SCHEMA,
    ReleaseStreamingPrepareConfig,
    prepare_distillation_dataset_release,
    main

const FINAL_NEURON_SCHEMA = V6.FINAL_NEURON_SCHEMA
const RELEASE_DATASET_SCHEMA = V6.RELEASE_DATASET_SCHEMA
const RELEASE_SHARD_SCHEMA = V6.RELEASE_SHARD_SCHEMA

"""
Load only the canonical paper-final teacher.

The Python generator stores the exact canonical JSON byte string used for its
teacher-contract hash.  JSON parsers legitimately lose the lexical distinction
between `34.0` and `34`, so release verification hashes that raw byte string
and separately proves that its parsed structure equals the manifest contract.
"""
function _load_release_source(
    config::ReleaseStreamingPrepareConfig,
)
    root = abspath(config.dataset_path)
    isdir(root) ||
        error("teacher dataset must be a manifest directory: $root")
    manifest_path = joinpath(root, "manifest.json")
    isfile(manifest_path) ||
        error("teacher dataset has no manifest.json")
    manifest = JSON3.read(read(manifest_path, String))
    String(V6._required(manifest, :schema_name)) ==
        FINAL_NEURON_SCHEMA || error(
        "release promotion accepts only $FINAL_NEURON_SCHEMA",
    )
    String(V6._required(manifest, :model_name)) ==
        HD_SWSNN_TWINPROP_NAME ||
        error("teacher model family is not HD-SWSNN-TwinProp")
    String(V6._required(manifest, :stage)) ==
        "official_hay_neuron_teacher_final" ||
        error("teacher is not the final official Hay/NEURON stage")
    String(V6._required(manifest, :completion_state)) == "complete" ||
        error("official final-v2 generation is incomplete")
    V6._required(
        manifest,
        :modeldb_source_modified_by_generator,
    ) === false ||
        error("official ModelDB checkout was modified by generator")
    V6._required(manifest, :resumable_sidecars_verified) === true ||
        error("official final-v2 sidecars were not verified")
    V6._validate_public_contract(manifest)

    contract = V6._required(manifest, :teacher_contract)
    String(V6._required(contract, :schema_name)) ==
        FINAL_NEURON_SCHEMA ||
        error("embedded teacher contract uses another schema")
    declared_contract = BaseBridge._require_hash(
        "teacher contract",
        V6._required(manifest, :teacher_contract_sha256),
    )
    embedded_contract = BaseBridge._require_hash(
        "embedded teacher contract",
        V6._required(contract, :teacher_contract_sha256),
    )
    embedded_contract == declared_contract ||
        error("embedded teacher-contract hash differs")
    canonical_text = String(
        V6._required(manifest, :teacher_contract_canonical_json),
    )
    V6._sha256_text(canonical_text) == declared_contract ||
        error("raw teacher-contract canonical SHA-256 mismatch")
    canonical_payload =
        JSON3.read(canonical_text, Dict{String,Any})
    contract_payload =
        JSON3.read(JSON3.write(contract), Dict{String,Any})
    delete!(contract_payload, "teacher_contract_sha256")
    canonical_payload == contract_payload ||
        error("raw canonical teacher contract differs structurally")
    V6._canonical_json(V6._required(contract, :config)) ==
        V6._canonical_json(V6._required(manifest, :config)) ||
        error("teacher contract/config differs from manifest")
    V6._canonical_json(V6._required(contract, :source_hashes)) ==
        V6._canonical_json(V6._required(manifest, :source_hashes)) ||
        error("teacher contract/source hashes differs from manifest")

    detailed_teacher,
    detailed_kernel,
    morphology,
    modeldb = BaseBridge._official_hashes(manifest)
    detailed_teacher == declared_contract ||
        error("official teacher lineage is not final contract")
    for (label, actual, expected) in (
        (
            "detailed teacher",
            detailed_teacher,
            config.expected_detailed_teacher_sha256,
        ),
        (
            "detailed kernel",
            detailed_kernel,
            config.expected_detailed_kernel_sha256,
        ),
        (
            "morphology",
            morphology,
            config.expected_morphology_sha256,
        ),
        (
            "ModelDB source",
            modeldb,
            config.expected_modeldb_source_sha256,
        ),
    )
        BaseBridge._expected_hash(label, actual, expected)
    end

    records = collect(V6._required(manifest, :shards))
    isempty(records) && error("teacher manifest has no shards")
    shard_paths = String[]
    shard_hashes = String[]
    expected_first = 1
    train_count = 0
    test_count = 0
    seen_paths = Set{String}()
    for record in records
        relative = String(V6._required(record, :path))
        relative in seen_paths &&
            error("teacher repeats shard path $relative")
        push!(seen_paths, relative)
        path = abspath(joinpath(root, relative))
        V6._inside_root(path, root) ||
            error("teacher shard escapes dataset root: $relative")
        isfile(path) || error("teacher shard is absent: $path")
        Int(V6._required(record, :global_first)) ==
            expected_first ||
            error("teacher shard ranges are not contiguous")
        global_last = Int(V6._required(record, :global_last))
        global_last >= expected_first ||
            error("teacher shard range is empty")
        samples = Int(V6._required(record, :samples))
        samples == global_last - expected_first + 1 ||
            error("teacher shard sample count differs from range")
        expected_first = global_last + 1
        filesize(path) == Int(V6._required(record, :bytes)) ||
            error("teacher shard byte count differs: $relative")
        declared_shard = BaseBridge._require_hash(
            "declared teacher shard",
            V6._required(record, :sha256),
        )
        actual_shard = BaseBridge._sha256_file(path)
        actual_shard == declared_shard || error(
            "teacher shard hash mismatch for $relative",
        )
        BaseBridge._require_hash(
            "shard teacher contract",
            V6._required(record, :teacher_contract_sha256),
        ) == declared_contract ||
            error("teacher shard uses another teacher contract")
        counts = V6._required(record, :split_counts)
        train_count += Int(V6._required(counts, :train))
        test_count += Int(V6._required(counts, :held_out_test))
        push!(shard_paths, path)
        push!(shard_hashes, actual_shard)
    end

    completed = Int(V6._required(manifest, :completed_trials))
    expected_first == completed + 1 ||
        error("teacher shard ranges do not cover completed trials")
    completed == train_count + test_count ||
        error("teacher split counts do not cover completed trials")
    source_config = V6._required(manifest, :config)
    configured_train =
        Int(V6._required(source_config, :train_trials))
    configured_test =
        Int(V6._required(source_config, :test_trials))
    configured_duration =
        Int(V6._required(source_config, :duration_ms))
    configured_train == train_count ||
        error("manifest train count differs from shards")
    configured_test == test_count ||
        error("manifest test count differs from shards")
    if config.require_full_public_counts
        (
            train_count,
            test_count,
            configured_duration,
            completed,
        ) == (
            V6.PAPER_TRAIN_POOL,
            V6.PAPER_HELD_OUT_TEST,
            V6.PAPER_DURATION_MS,
            V6.PAPER_TRAIN_POOL + V6.PAPER_HELD_OUT_TEST,
        ) || error(
            "promotable release requires complete 50k/2k 10-second source",
        )
        Int(V6._required(manifest, :total_segments)) == 642 ||
            error("promotable release source must expose 642 segments")
    end

    manifest_hash = BaseBridge._sha256_file(manifest_path)
    dataset_hash =
        BaseBridge._sha256_strings(manifest_hash, shard_hashes...)
    BaseBridge._expected_hash(
        "source dataset",
        dataset_hash,
        config.expected_source_dataset_sha256,
    )
    source = BaseBridge._Source(
        root,
        manifest_path,
        manifest,
        shard_paths,
        dataset_hash,
        manifest_hash,
        FINAL_NEURON_SCHEMA,
        "complete",
        detailed_teacher,
        detailed_kernel,
        morphology,
        modeldb,
    )
    segment_catalog_sha256 = V6._sha256_text(
        V6._canonical_json(V6._required(manifest, :segments)),
    )
    source_counts = (;
        train_pool=train_count,
        held_out_test=test_count,
        duration_ms=configured_duration,
        completed,
    )
    return source, source_counts, segment_catalog_sha256
end

function _append_split_indices!(
    train_indices,
    validation_indices,
    test_indices,
    global_indices,
    split_code,
)
    for (index, code) in zip(global_indices, split_code)
        if code == BaseBridge.TRAIN_SPLIT
            push!(train_indices, index)
        elseif code == BaseBridge.VALIDATION_SPLIT
            push!(validation_indices, index)
        elseif code == BaseBridge.TEST_SPLIT
            push!(test_indices, index)
        else
            error("prepared split code is invalid")
        end
    end
    return nothing
end

function _write_release_shard(
    staging,
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
    compact = V6._compact_slice(source_shard, selected_trials)
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

function _release_manifest(
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
        created_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
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
        recomputed_twin_gate,
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

function prepare_distillation_dataset_release(
    config::ReleaseStreamingPrepareConfig,
)
    config.time_chunk >= 1 ||
        throw(ArgumentError("time_chunk must be positive"))
    config.output_shard_samples >= 1 ||
        throw(ArgumentError("output_shard_samples must be positive"))
    config.auroc_histogram_bins >= 256 ||
        throw(ArgumentError("auroc_histogram_bins must be >= 256"))
    destination = abspath(config.output_directory)
    ispath(destination) &&
        error("output directory already exists: $destination")
    staging = destination * ".staging." * string(getpid())
    ispath(staging) &&
        error("staging directory already exists: $staging")

    base_config = V6._base_config(config)
    source, source_counts, segment_catalog_sha256 =
        _load_release_source(config)
    frozen, integrity_before, reported_metrics, twin_file_hash =
        BaseBridge._verify_twin(base_config, source)
    plan_data = V6._stream_plan(config, source)
    frozen.model.config.nmda_regions == 4 ||
        error("release distillation requires four NMDA regions")
    frozen.model.config.segments ==
        Int(V6._required(source.manifest, :total_segments)) ||
        error("frozen twin/official segment count differs")
    total_samples = sum(
        length(item.selected) for item in plan_data.plan
    )
    selected_count_tuple = (
        plan_data.split_counts[BaseBridge.TRAIN_SPLIT],
        plan_data.split_counts[BaseBridge.VALIDATION_SPLIT],
        plan_data.split_counts[BaseBridge.TEST_SPLIT],
    )
    promotion_eligible =
        config.require_full_public_counts &&
        source_counts.train_pool == V6.PAPER_TRAIN_POOL &&
        source_counts.held_out_test == V6.PAPER_HELD_OUT_TEST &&
        selected_count_tuple == (
            V6.PAPER_TRAIN_POOL - config.validation_samples,
            config.validation_samples,
            V6.PAPER_HELD_OUT_TEST,
        )
    config.require_full_public_counts && !promotion_eligible &&
        error(
            "promotable release must preserve all 50k train-pool " *
            "and 2k held-out trials",
        )

    config_sha256 = V6._config_hash(config)
    gate = V6._new_gate(config.auroc_histogram_bins)
    shard_records = NamedTuple[]
    train_indices = Int32[]
    validation_indices = Int32[]
    test_indices = Int32[]
    selected_segments = Int[]
    diagnostic_segments = Int[]
    diagnostic_times = Int32[]
    global_output_index = 0
    output_shard_index = 0
    mkpath(staging)
    try
        for source_plan in plan_data.plan
            source_shard = V6._read_final_shard(source_plan.path)
            layout = V6._validate_final_shard(source_shard)
            for first_selected in
                1:config.output_shard_samples:length(source_plan.selected)
                last_selected = min(
                    first_selected +
                    config.output_shard_samples - 1,
                    length(source_plan.selected),
                )
                group_range = first_selected:last_selected
                selected_trials =
                    source_plan.selected[group_range]
                split_code =
                    source_plan.split_code[group_range]
                source_ids =
                    source_plan.source_ids[group_range]
                trial_count = length(selected_trials)
                voltage = Matrix{Float32}(
                    undef,
                    plan_data.time_steps,
                    trial_count,
                )
                spike = similar(voltage)
                spike_logit = similar(voltage)
                nmda = Array{Float32,3}(
                    undef,
                    4,
                    plan_data.time_steps,
                    trial_count,
                )
                chosen_segments,
                selected_rows,
                shard_diagnostic =
                    BaseBridge._diagnostic_selection(
                        base_config,
                        source,
                        source_shard,
                    )
                if isempty(selected_segments)
                    selected_segments = chosen_segments
                    diagnostic_segments = shard_diagnostic
                    diagnostic_times = layout.diagnostic_times
                elseif selected_segments != chosen_segments ||
                       diagnostic_segments != shard_diagnostic ||
                       diagnostic_times != layout.diagnostic_times
                    error("diagnostic mapping changed across shards")
                end
                calcium, dendritic =
                    BaseBridge._detailed_internal_targets(
                        base_config,
                        source_shard,
                        selected_trials,
                        selected_segments,
                        selected_rows,
                    )
                for (output_trial, source_trial) in
                    enumerate(selected_trials)
                    sparse = V6._sparse_sample(
                        source_shard,
                        source_trial,
                        frozen.model.config,
                    )
                    prediction = V6._infer_sample(
                        frozen,
                        sparse,
                        plan_data.time_steps,
                        config.time_chunk,
                    )
                    voltage[:, output_trial] .= prediction.voltage
                    spike[:, output_trial] .= prediction.spike
                    spike_logit[:, output_trial] .=
                        prediction.spike_logit
                    nmda[:, :, output_trial] .= prediction.nmda
                    if split_code[output_trial] ==
                       BaseBridge.TEST_SPLIT
                        Legacy._update_gate!(
                            gate,
                            prediction,
                            @view(
                                V6._required(
                                    source_shard,
                                    :target_voltage,
                                )[:, source_trial]
                            ),
                            @view(
                                V6._required(
                                    source_shard,
                                    :target_spike,
                                )[:, source_trial]
                            ),
                            @view(
                                V6._required(
                                    source_shard,
                                    :target_nmda,
                                )[:, :, source_trial]
                            ),
                        )
                    end
                end

                first_global = global_output_index + 1
                global_output_index += trial_count
                global_indices =
                    Int32.(first_global:global_output_index)
                _append_split_indices!(
                    train_indices,
                    validation_indices,
                    test_indices,
                    global_indices,
                    split_code,
                )
                output_shard_index += 1
                record = _write_release_shard(
                    staging,
                    output_shard_index,
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
                push!(shard_records, record)
            end
        end
        global_output_index == total_samples ||
            error("streaming sample count mismatch")
        recomputed_gate = Legacy._finish_gate(gate)
        recomputed_gate.spike_auroc >=
            config.minimum_twin_spike_auroc || error(
            "frozen twin failed stream-recomputed held-out gate: " *
            "spike AUROC $(recomputed_gate.spike_auroc) < " *
            "$(config.minimum_twin_spike_auroc)",
        )
        integrity_after = assert_frozen_unchanged(
            frozen;
            expected_artifact_sha256=
                integrity_before.artifact_sha256,
        )
        integrity_after.max_delta == 0 ||
            error("frozen twin changed during preparation")
        manifest = _release_manifest(
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
        manifest_path = joinpath(staging, "manifest.json")
        Legacy._json_write(manifest_path, manifest)
        manifest_sha256 =
            BaseBridge._sha256_file(manifest_path)
        mv(staging, destination)
        return (;
            schema=RELEASE_DATASET_SCHEMA,
            output_directory=destination,
            manifest_path=joinpath(destination, "manifest.json"),
            manifest_sha256,
            total_samples,
            shard_count=length(shard_records),
            split_counts=manifest.split_counts,
            promotion_eligible,
            peak_dense_chunk_bytes=
                manifest.peak_dense_chunk_bytes,
            dense_memory_scales_with_total_samples=false,
            recomputed_twin_gate,
            digital_twin_gate_passed=true,
            frozen_max_delta_before=integrity_before.max_delta,
            frozen_max_delta_after=integrity_after.max_delta,
        )
    catch
        ispath(staging) &&
            rm(staging; recursive=true, force=true)
        rethrow()
    end
end

function main(arguments=ARGS)
    report = prepare_distillation_dataset_release(
        V6._parse_arguments(arguments),
    )
    println(JSON3.write(report))
    return report
end

end # module DistillationDatasetBridgeReleaseFinal

if abspath(PROGRAM_FILE) == @__FILE__
    DistillationDatasetBridgeReleaseFinal.main()
end
