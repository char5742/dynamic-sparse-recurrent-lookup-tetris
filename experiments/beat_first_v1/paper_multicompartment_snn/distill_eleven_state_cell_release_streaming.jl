"""
Canonical bounded-memory final-v2 distillation driver.

This driver consumes only the official sharded release-v2 bridge. It never
constructs `[input_dim, full_time, total_samples]`, never promotes the legacy
dense path, and preserves the exact teacher/twin/shard hash chain.
"""

include(joinpath(@__DIR__, "distill_eleven_state_cell_release_v2.jl"))
if !isdefined(Main, :StreamingReleaseDataset)
    include(joinpath(@__DIR__, "StreamingReleaseDataset.jl"))
end
if !isdefined(Main, :StreamingReleaseMetrics)
    include(joinpath(@__DIR__, "StreamingReleaseMetrics.jl"))
end
if !isdefined(Main, :DistilledElevenStateCellReleaseRuntimeV3)
    include(joinpath(
        @__DIR__,
        "DistilledElevenStateCellReleaseRuntimeV3.jl",
    ))
end

const StreamData = Main.StreamingReleaseDataset
const StreamMetrics = Main.StreamingReleaseMetrics
const StreamRuntime =
    Main.DistilledElevenStateCellReleaseRuntimeV3

const STREAM_SEMANTIC_COORDINATE_NAMES = (
    "distal_basal_dendritic_voltage",
    "proximal_apical_trunk_voltage",
    "apical_calcium_hot_zone_voltage",
    "distal_apical_tuft_voltage",
    "soma_nmda_current",
    "basal_nmda_current",
    "apical_trunk_nmda_current",
    "apical_tuft_nmda_current",
    "apical_calcium_context",
    "soma_voltage",
    "calcium_adaptation",
)

Base.@kwdef struct StreamingReleaseDistillationConfig
    dataset::String
    frozen_twin::String
    output::String
    metrics::String
    epochs::Int = 45
    steps_per_epoch::Int = 64
    batch::Int = 32
    window::Int = 256
    learning_rate::Float32 = 0.001f0
    free_rollout_epochs::Int = 15
    metric_time_chunk::Int = 256
    metric_auroc_bins::Int = 65_536
    train_metric_samples::Int = 1_000
    minimum_spike_auroc::Float64 = RELEASE_MINIMUM_SPIKE_AUROC
    seed::UInt64 = 0x005b5a19
    paper_scale::Bool = true
    require_promotion_eligible::Bool = true
end

function _stream_validate_config(config)
    isempty(config.dataset) &&
        throw(ArgumentError("sharded dataset path is required"))
    isempty(config.frozen_twin) &&
        throw(ArgumentError("frozen twin path is required"))
    isempty(config.output) &&
        throw(ArgumentError("output artifact path is required"))
    isempty(config.metrics) &&
        throw(ArgumentError("metrics path is required"))
    config.epochs >= 1 ||
        throw(ArgumentError("epochs must be positive"))
    config.steps_per_epoch >= 1 ||
        throw(ArgumentError("steps_per_epoch must be positive"))
    config.batch >= 1 ||
        throw(ArgumentError("batch must be positive"))
    config.window >= 1 ||
        throw(ArgumentError("window must be positive"))
    config.learning_rate > 0.0f0 ||
        throw(ArgumentError("learning rate must be positive"))
    0 <= config.free_rollout_epochs <= config.epochs ||
        throw(ArgumentError("free_rollout_epochs is invalid"))
    config.metric_time_chunk >= 1 ||
        throw(ArgumentError("metric_time_chunk must be positive"))
    config.metric_auroc_bins >= 256 ||
        throw(ArgumentError("metric_auroc_bins must be at least 256"))
    config.train_metric_samples >= 1 ||
        throw(ArgumentError("train_metric_samples must be positive"))
    config.minimum_spike_auroc >= RELEASE_MINIMUM_SPIKE_AUROC ||
        throw(ArgumentError("streaming release cannot weaken 0.985"))
    config.paper_scale == config.require_promotion_eligible ||
        throw(ArgumentError(
            "paper_scale and require_promotion_eligible must agree",
        ))
    return config
end

function _stream_sequence_loss(
    parameters,
    raw_input,
    target,
    target_mean,
    target_scale,
    segments,
    free_fraction::Float32,
)
    return _release_sequence_loss(
        parameters,
        raw_input,
        target,
        target_mean,
        target_scale,
        segments,
        free_fraction,
    )
end

function _stream_sample_window(
    rng,
    dataset::StreamData.StreamDataset,
    config,
)
    window = min(config.window, dataset.time_steps)
    first_time =
        rand(rng, 1:(dataset.time_steps - window + 1))
    samples =
        rand(rng, dataset.train_indices, config.batch)
    batch = StreamData.stream_materialize_window(
        dataset,
        samples,
        first_time,
        window,
    )
    return batch.raw_input, batch.target
end

function _stream_train(
    rng,
    dataset::StreamData.StreamDataset,
    target_mean,
    target_scale,
    segments,
    config,
)
    parameters = _release_initial_parameters(rng, segments)
    optimizer_state = Optimisers.setup(
        Optimisers.Adam(config.learning_rate),
        parameters,
    )
    history = NamedTuple[]
    for epoch in 1:config.epochs
        free_fraction = _release_fraction(epoch, config)
        total_loss = 0.0
        started = time()
        for _ in 1:config.steps_per_epoch
            input_window, target_window =
                _stream_sample_window(rng, dataset, config)
            loss, gradient = Zygote.withgradient(parameters) do candidate
                _stream_sequence_loss(
                    candidate,
                    input_window,
                    target_window,
                    target_mean,
                    target_scale,
                    segments,
                    free_fraction,
                )
            end
            isfinite(loss) ||
                error("streaming distillation loss is non-finite")
            optimizer_state, parameters = Optimisers.update(
                optimizer_state,
                parameters,
                only(gradient),
            )
            _release_all_finite(parameters) ||
                error("streaming distillation parameter is non-finite")
            total_loss += Float64(loss)
        end
        record = (;
            epoch,
            loss=total_loss / config.steps_per_epoch,
            free_rollout_fraction=free_fraction,
            elapsed_seconds=time() - started,
        )
        push!(history, record)
        @printf(
            "stream release distill %d/%d loss=%.6f free=%.3f time=%.2fs\n",
            epoch,
            config.epochs,
            record.loss,
            free_fraction,
            record.elapsed_seconds,
        )
        flush(stdout)
    end
    return parameters, history
end

@inline function _stream_physical_output(
    raw,
    target_mean,
    target_scale,
)
    output = vec(raw) .* target_scale .+ target_mean
    output[2] = Float32(_release_sigmoid(raw[2]))
    output[7] = Float32(_release_sigmoid(raw[7]))
    return output
end

function _stream_semantic_validity(observed)
    validity = falses(11)
    validity[1:4] .= observed[8:11]
    validity[5:8] .= observed[3:6]
    validity[9] =
        observed[7] && observed[10] && observed[11]
    validity[10] = observed[1]
    validity[11] = observed[7]
    return validity
end

function _stream_update_metrics!(
    metrics::StreamMetrics.StreamingMetrics,
    prediction,
    truth,
    state,
    semantic,
    physical_validity,
    semantic_validity,
)
    @inbounds for coordinate in 1:11
        if physical_validity[coordinate]
            physical_prediction = Float64(prediction[coordinate])
            physical_truth = Float64(truth[coordinate])
            isfinite(physical_prediction) &&
                isfinite(physical_truth) ||
                error("stream physical metric is non-finite")
            StreamMetrics._update_pair!(
                metrics.output_statistics[coordinate],
                physical_prediction,
                physical_truth,
            )
            coordinate == 2 &&
                StreamMetrics._update_auroc!(
                    metrics.spike_auroc,
                    physical_prediction,
                    physical_truth,
                )
            coordinate == 7 &&
                StreamMetrics._update_auroc!(
                    metrics.calcium_auroc,
                    physical_prediction,
                    physical_truth,
                )
        end
        if semantic_validity[coordinate]
            state_value = Float64(state[coordinate])
            target_value = Float64(semantic[coordinate])
            isfinite(state_value) && isfinite(target_value) ||
                error("stream semantic metric is non-finite")
            StreamMetrics._update_pair!(
                metrics.semantic_statistics[coordinate],
                state_value,
                target_value,
            )
        end
    end
    metrics.observations += 1
    return metrics
end

function _stream_conservative_auroc(histogram)
    estimate = StreamMetrics._auroc(histogram)
    isfinite(estimate) ||
        return (; estimate, ambiguity=NaN, lower_bound=NaN)
    denominator =
        Float64(histogram.positive_count) *
        Float64(histogram.negative_count)
    ambiguous_pairs = 0.0
    @inbounds for bin in eachindex(histogram.positive)
        ambiguous_pairs +=
            0.5 *
            Float64(histogram.positive[bin]) *
            Float64(histogram.negative[bin])
    end
    ambiguity = ambiguous_pairs / denominator
    return (;
        estimate,
        ambiguity,
        lower_bound=max(0.0, estimate - ambiguity),
    )
end

function _stream_finalize_metrics(metrics)
    output_rmse = StreamMetrics.output_rmse(metrics)
    output_correlation =
        StreamMetrics.output_correlation(metrics)
    coordinate_rmse = StreamMetrics.semantic_rmse(metrics)
    coordinate_correlation =
        StreamMetrics.semantic_correlation(metrics)
    coordinate_passed = [
        isfinite(coordinate_rmse[coordinate]) &&
        isfinite(coordinate_correlation[coordinate]) &&
        coordinate_rmse[coordinate] <=
            RELEASE_MAXIMUM_COORDINATE_RMSE &&
        coordinate_correlation[coordinate] >=
            RELEASE_MINIMUM_COORDINATE_CORRELATION
        for coordinate in 1:11
    ]
    spike = _stream_conservative_auroc(metrics.spike_auroc)
    calcium = _stream_conservative_auroc(metrics.calcium_auroc)
    return (;
        samples=metrics.samples,
        free_rollout_horizon=metrics.free_rollout_horizon,
        soma_voltage_rmse_mv=output_rmse[1],
        soma_voltage_correlation=output_correlation[1],
        spike_auroc=spike.lower_bound,
        spike_auroc_estimate=spike.estimate,
        spike_auroc_ambiguity_bound=spike.ambiguity,
        nmda_rmse_by_region=output_rmse[3:6],
        nmda_correlation_by_region=output_correlation[3:6],
        calcium_event_auroc=calcium.lower_bound,
        calcium_event_auroc_estimate=calcium.estimate,
        calcium_event_auroc_ambiguity_bound=calcium.ambiguity,
        dendritic_voltage_rmse_mv=output_rmse[8:11],
        semantic_coordinate_names=
            STREAM_SEMANTIC_COORDINATE_NAMES,
        semantic_coordinate_rmse=coordinate_rmse,
        semantic_coordinate_correlation=coordinate_correlation,
        semantic_coordinate_passed=coordinate_passed,
        auroc_is_conservative_lower_bound=true,
        auroc_histogram_bins=length(
            metrics.spike_auroc.positive,
        ),
    )
end

function _stream_metrics(
    parameters,
    dataset::StreamData.StreamDataset,
    indices,
    target_mean,
    target_scale,
    segments;
    time_chunk::Int,
    auroc_bins::Int,
)
    selected = Int.(collect(indices))
    metrics = StreamMetrics.StreamingMetrics(
        length(selected),
        dataset.time_steps;
        auroc_bins,
        maximum_coordinate_rmse=
            RELEASE_MAXIMUM_COORDINATE_RMSE,
        minimum_coordinate_correlation=
            RELEASE_MINIMUM_COORDINATE_CORRELATION,
    )
    for global_index in selected
        state = repeat(parameters.initial_state, 1, 1)
        shard_index =
            Int(dataset.global_to_shard[global_index])
        record = dataset.records[shard_index]
        shard = StreamData._load_shard(dataset, shard_index)
        local_index = global_index - record.global_first + 1
        for first_time in 1:time_chunk:dataset.time_steps
            last_time = min(
                first_time + time_chunk - 1,
                dataset.time_steps,
            )
            count = last_time - first_time + 1
            raw_input =
                zeros(Float32, dataset.input_dim, count)
            target = zeros(Float32, 11, count)
            observed = falses(11, count)
            StreamData._fill_raw_window!(
                raw_input,
                shard,
                local_index,
                first_time,
                last_time,
            )
            StreamData._fill_target_window!(
                target,
                observed,
                dataset,
                shard,
                local_index,
                first_time,
                last_time,
            )
            window_bytes =
                sizeof(Float32) *
                (length(raw_input) + length(target)) +
                sizeof(Bool) * length(observed)
            dataset.tracker.peak_dense_window_bytes = max(
                dataset.tracker.peak_dense_window_bytes,
                window_bytes,
            )
            dataset.tracker.peak_combined_bytes = max(
                dataset.tracker.peak_combined_bytes,
                record.bytes + window_bytes,
            )
            for local_time in 1:count
                input = _release_project_input(
                    @view(raw_input[:, local_time:local_time]),
                    parameters.location_logits,
                    segments,
                )
                state =
                    _release_transition(parameters, state, input)
                raw_output =
                    _release_structured_readout(parameters, state)
                normalized = _release_normalize_target(
                    @view(target[:, local_time:local_time]),
                    target_mean,
                    target_scale,
                )
                semantic = _release_semantic_target(normalized)
                physical = _stream_physical_output(
                    @view(raw_output[:, 1]),
                    target_mean,
                    target_scale,
                )
                physical_validity =
                    @view(observed[:, local_time])
                semantic_validity = _stream_semantic_validity(
                    physical_validity,
                )
                _stream_update_metrics!(
                    metrics,
                    physical,
                    @view(target[:, local_time]),
                    @view(state[:, 1]),
                    @view(semantic[:, 1]),
                    physical_validity,
                    semantic_validity,
                )
            end
        end
    end
    return _stream_finalize_metrics(metrics)
end

function _stream_parse_arguments(arguments)
    options = Dict{String,String}(
        "dataset" => get(
            ENV,
            "HD_TWINPROP_DISTILL_DATASET_DIR",
            "",
        ),
        "frozen-twin" => get(
            ENV,
            "HD_TWINPROP_TWIN_PATH",
            "",
        ),
        "output" => get(
            ENV,
            "HD_TWINPROP_DISTILLED_PATH",
            joinpath(
                @__DIR__,
                "artifacts",
                "distilled_eleven_state_streaming.jld2",
            ),
        ),
        "metrics" => get(
            ENV,
            "HD_TWINPROP_DISTILL_METRICS",
            joinpath(
                @__DIR__,
                "artifacts",
                "distilled_eleven_state_streaming.json",
            ),
        ),
        "epochs" => "45",
        "steps-per-epoch" => "64",
        "batch" => "32",
        "window" => "256",
        "learning-rate" => "0.001",
        "free-rollout-epochs" => "15",
        "metric-time-chunk" => "256",
        "metric-auroc-bins" => "65536",
        "train-metric-samples" => "1000",
        "minimum-spike-auroc" => "0.985",
        "seed" => string(UInt64(0x005b5a19)),
        "paper-scale" => "true",
    )
    index = 1
    while index <= length(arguments)
        token = arguments[index]
        startswith(token, "--") ||
            error("unexpected positional argument: $token")
        key = token[3:end]
        haskey(options, key) || error("unknown option --$key")
        index += 1
        index <= length(arguments) ||
            error("missing value for --$key")
        options[key] = arguments[index]
        index += 1
    end
    paper_scale = parse(Bool, options["paper-scale"])
    return StreamingReleaseDistillationConfig(
        dataset=options["dataset"],
        frozen_twin=options["frozen-twin"],
        output=options["output"],
        metrics=options["metrics"],
        epochs=parse(Int, options["epochs"]),
        steps_per_epoch=parse(
            Int,
            options["steps-per-epoch"],
        ),
        batch=parse(Int, options["batch"]),
        window=parse(Int, options["window"]),
        learning_rate=parse(
            Float32,
            options["learning-rate"],
        ),
        free_rollout_epochs=parse(
            Int,
            options["free-rollout-epochs"],
        ),
        metric_time_chunk=parse(
            Int,
            options["metric-time-chunk"],
        ),
        metric_auroc_bins=parse(
            Int,
            options["metric-auroc-bins"],
        ),
        train_metric_samples=parse(
            Int,
            options["train-metric-samples"],
        ),
        minimum_spike_auroc=parse(
            Float64,
            options["minimum-spike-auroc"],
        ),
        seed=parse(UInt64, options["seed"]),
        paper_scale,
        require_promotion_eligible=paper_scale,
    )
end

function run_streaming_release_distillation(
    config::StreamingReleaseDistillationConfig,
)
    _stream_validate_config(config)
    frozen_twin = ReleaseTwin.load_frozen_twin(config.frozen_twin)
    twin_before = ReleaseTwin.assert_frozen_unchanged(frozen_twin)
    raw_twin_file_sha256 =
        _release_file_sha256(config.frozen_twin)
    dataset = StreamData.open_stream_dataset(
        config.dataset,
        frozen_twin;
        minimum_spike_auroc=config.minimum_spike_auroc,
        verify_shard_hashes=true,
        require_promotion_eligible=
            config.require_promotion_eligible,
    )
    raw_twin_file_sha256 ==
        dataset.frozen_twin_file_sha256 ||
        error("raw frozen-twin file SHA-256 mismatch")
    StreamData.stream_dataset_integrity!(dataset)
    target_mean, target_scale =
        StreamData.stream_target_statistics(dataset)
    config_record = (;
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        batch=config.batch,
        window=min(config.window, dataset.time_steps),
        learning_rate=config.learning_rate,
        free_rollout_epochs=config.free_rollout_epochs,
        metric_time_chunk=config.metric_time_chunk,
        metric_auroc_bins=config.metric_auroc_bins,
        train_metric_samples=config.train_metric_samples,
        minimum_spike_auroc=config.minimum_spike_auroc,
        seed=config.seed,
        paper_scale=config.paper_scale,
        prepared_dataset_schema=
            StreamData.RELEASE_DATASET_SCHEMA,
        training_input_mode="sharded_streaming_v2",
        full_dataset_dense_materialization=false,
        sparse_diagnostic_interpolation="linear_for_fit_exact_mask_for_gate",
        semantic_coordinate_names=
            STREAM_SEMANTIC_COORDINATE_NAMES,
        recurrent_mask_sha256=
            RELEASE_RECURRENT_MASK_SHA256,
        input_mask_sha256=RELEASE_INPUT_MASK_SHA256,
        structured_transition=true,
        structured_readout=true,
        coordinate_wise_semantic_supervision=true,
        soma_spike_is_sole_external_event=true,
        auroc_gate=
            "fixed_histogram_conservative_lower_bound",
    )
    config_hash = _release_value_sha256(config_record)
    trained, history = _stream_train(
        Xoshiro(config.seed),
        dataset,
        target_mean,
        target_scale,
        frozen_twin.model.config.segments,
        config,
    )
    train_metric_count = min(
        config.train_metric_samples,
        length(dataset.train_indices),
    )
    train_metric_indices =
        dataset.train_indices[1:train_metric_count]
    metrics = (;
        train=_stream_metrics(
            trained,
            dataset,
            train_metric_indices,
            target_mean,
            target_scale,
            frozen_twin.model.config.segments;
            time_chunk=config.metric_time_chunk,
            auroc_bins=config.metric_auroc_bins,
        ),
        validation=_stream_metrics(
            trained,
            dataset,
            dataset.validation_indices,
            target_mean,
            target_scale,
            frozen_twin.model.config.segments;
            time_chunk=config.metric_time_chunk,
            auroc_bins=config.metric_auroc_bins,
        ),
        test=_stream_metrics(
            trained,
            dataset,
            dataset.test_indices,
            target_mean,
            target_scale,
            frozen_twin.model.config.segments;
            time_chunk=config.metric_time_chunk,
            auroc_bins=config.metric_auroc_bins,
        ),
    )
    gate = _release_gate(
        metrics.test,
        config.minimum_spike_auroc,
    )
    adapter = (;
        segment_region=dataset.segment_region,
        provenance=dataset.provenance,
        prepared_dataset_file_sha256=
            dataset.dataset_sha256,
    )
    parameters = _release_frozen_parameters(
        trained,
        adapter,
        frozen_twin,
        target_mean,
        target_scale,
        config_hash,
    )
    parameter_hash = ReleaseCell.parameter_sha256(parameters)
    official_segment_region =
        Tuple(String.(dataset.segment_region))
    location_mapping_sha256 = _release_value_sha256((
        dataset.segment_catalog_sha256,
        official_segment_region,
        parameters.compartment_projection,
    ))
    StreamData.stream_dataset_integrity!(dataset)
    twin_after =
        ReleaseTwin.assert_frozen_unchanged(frozen_twin)
    twin_before == twin_after ||
        error("frozen digital twin changed during streaming distillation")
    semantic_coordinate_gate = (;
        passed=all(metrics.test.semantic_coordinate_passed),
        coordinate_names=STREAM_SEMANTIC_COORDINATE_NAMES,
        rmse=metrics.test.semantic_coordinate_rmse,
        correlation=
            metrics.test.semantic_coordinate_correlation,
        per_coordinate_passed=
            metrics.test.semantic_coordinate_passed,
        maximum_rmse=RELEASE_MAXIMUM_COORDINATE_RMSE,
        minimum_correlation=
            RELEASE_MINIMUM_COORDINATE_CORRELATION,
    )
    structured_transition_contract = (;
        recurrent_mask_sha256=
            RELEASE_RECURRENT_MASK_SHA256,
        input_mask_sha256=RELEASE_INPUT_MASK_SHA256,
        structured_readout=true,
        dense_rotational_hidden_basis=false,
        coordinate_wise_semantic_supervision=true,
        semantic_state_scale="normalized_unit_interval",
        location_index_type="UInt16",
    )
    manifest = dataset.manifest
    teacher_counts = (;
        total_samples=dataset.total_samples,
        train=length(dataset.train_indices),
        validation=length(dataset.validation_indices),
        test=length(dataset.test_indices),
        time_steps=dataset.time_steps,
        source_public_counts=_get(
            manifest,
            :source_public_counts,
            nothing,
        ),
        observed_source_counts=_get(
            manifest,
            :observed_source_counts,
            nothing,
        ),
    )
    memory = (;
        peak_loaded_shard_bytes=
            dataset.tracker.peak_loaded_shard_bytes,
        peak_dense_window_bytes=
            dataset.tracker.peak_dense_window_bytes,
        peak_combined_bytes=
            dataset.tracker.peak_combined_bytes,
        windows_materialized=
            dataset.tracker.windows_materialized,
        samples_materialized=
            dataset.tracker.samples_materialized,
        dense_memory_scales_with_total_samples=false,
        full_dataset_dense_materialization=false,
    )
    payload = (;
        schema=ReleaseCell.DISTILLED_ARTIFACT_SCHEMA,
        parameters,
        parameter_sha256=parameter_hash,
        frozen_internal=true,
        ablation_mode=:full,
        provisional=!gate.passed,
        training_input_mode="sharded_streaming_v2",
        prepared_dataset_schema=
            StreamData.RELEASE_DATASET_SCHEMA,
        dense_full_memory_path=false,
        paper_scale=config.paper_scale,
        development_scale=!config.paper_scale,
        teacher_counts,
        official_segment_count=RELEASE_SEGMENTS,
        official_segment_region,
        location_index_type="UInt16",
        semantic_state_scale="normalized_unit_interval",
        location_mapping_sha256,
        semantic_coordinate_gate,
        structured_transition_contract,
        teacher_hash=
            dataset.provenance.detailed_teacher_hash,
        cell_mechanism_sha256=
            dataset.provenance.detailed_kernel_hash,
        detailed_kernel_hash=
            dataset.provenance.detailed_kernel_hash,
        morphology_hash=dataset.provenance.morphology_hash,
        digital_twin_sha256=frozen_twin.artifact_sha256,
        digital_twin_hash=frozen_twin.artifact_sha256,
        frozen_twin_parameter_hash=
            frozen_twin.parameter_sha256,
        frozen_twin_artifact_hash=
            frozen_twin.artifact_sha256,
        raw_twin_file_sha256,
        official_modeldb_source_hash=
            dataset.provenance.official_modeldb_source_hash,
        official_teacher_file_hash=
            dataset.provenance.official_teacher_file_hash,
        official_source_dataset_hash=
            dataset.provenance.official_source_dataset_hash,
        prepared_dataset_manifest_sha256=
            dataset.manifest_sha256,
        prepared_dataset_file_sha256=
            dataset.dataset_sha256,
        distillation_dataset_hash=
            dataset.dataset_sha256,
        distillation_config_hash=config_hash,
        source_segment_catalog_sha256=
            dataset.segment_catalog_sha256,
        mixed_supervision=(;
            twin_targets=(
                :soma_voltage,
                :soma_spike,
                :nmda_current,
            ),
            official_neuron_targets=(
                :calcium_event,
                :dendritic_voltage,
            ),
            upstream_live_frozen_twin_inference=true,
            distiller_reverified_manifest_and_all_shard_hashes=true,
            sparse_diagnostic_gate_uses_only_observed_times=true,
        ),
        frozen_twin_integrity_before=twin_before,
        frozen_twin_integrity_after=twin_after,
        metrics,
        gate,
        config=config_record,
        history,
        memory,
        created_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
    )
    timestamp =
        Dates.format(now(UTC), dateformat"yyyymmddTHHMMSS")
    destination = gate.passed ?
        config.output :
        config.output * ".failed." * timestamp * ".jld2"
    _release_atomic_jldsave(destination; payload)
    if gate.passed
        runtime =
            StreamRuntime.load_release_runtime(destination)
        StreamRuntime.preflight_integrity!(runtime)
        StreamRuntime.checkpoint_integrity!(runtime)
        StreamRuntime.end_run_integrity!(runtime)
    end
    report = (;
        accepted=gate.passed,
        artifact_path=destination,
        artifact_sha256=
            ReleaseCell.artifact_sha256(destination),
        parameter_sha256=parameter_hash,
        paper_scale=config.paper_scale,
        prepared_dataset_schema=
            StreamData.RELEASE_DATASET_SCHEMA,
        distillation_dataset_hash=
            dataset.dataset_sha256,
        location_mapping_sha256,
        semantic_coordinate_gate,
        structured_transition_contract,
        metrics,
        gate,
        history,
        memory,
    )
    _release_atomic_json(config.metrics, report)
    @info "streaming release distillation finished" report.accepted report.paper_scale report.artifact_path report.gate
    gate.passed || error(
        "streaming release gates failed; candidate saved separately at " *
        destination * "; accepted artifact path was not modified",
    )
    return report
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_streaming_release_distillation(
        _stream_parse_arguments(ARGS),
    )
end
