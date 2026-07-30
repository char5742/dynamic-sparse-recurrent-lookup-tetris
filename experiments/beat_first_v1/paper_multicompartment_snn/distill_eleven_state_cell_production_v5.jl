"""
Production-qualified final distillation entry point.

This layer tightens the executable v4 implementation with the release gates
required by HD-SWSNN-TwinProp:

- official 642-segment NEURON source only,
- frozen digital-twin held-out AUROC >= 0.985,
- explicit semantic supervision for every one of the eleven states,
- spike, voltage, NMDA, Ca-event, dendritic-voltage and semantic-state gates,
- a hash-bound 642-segment -> four-latent spatial projection, and
- failed candidates written to a separate `.failed.*` path so an accepted
  artifact can never be overwritten.
"""

include(joinpath(
    @__DIR__,
    "distill_eleven_state_cell_production_v4.jl",
))
if !isdefined(Main, :DistilledElevenStateCellProduction)
    include(joinpath(
        @__DIR__,
        "DistilledElevenStateCellProduction.jl",
    ))
end
const ProductionCell = Main.DistilledElevenStateCellProduction

const PRODUCTION_TWIN_AUROC = 0.985
const PRODUCTION_VOLTAGE_RMSE_MV = 5.0
const PRODUCTION_VOLTAGE_CORRELATION = 0.90
const PRODUCTION_NMDA_CORRELATION = 0.80
const PRODUCTION_CALCIUM_AUROC = 0.80
const PRODUCTION_DENDRITIC_RMSE_MV = 8.0
const PRODUCTION_SEMANTIC_STATE_RMSE = 0.55

function _metadata_value_v5(dataset, name::Symbol, default=nothing)
    direct = _value(dataset, name, nothing)
    direct !== nothing && return direct
    metadata = _value(dataset, :metadata, nothing)
    metadata === nothing && return default
    direct = _value(metadata, name, nothing)
    direct !== nothing && return direct
    hashes = _value(metadata, :hashes, nothing)
    hashes === nothing && return default
    return _value(hashes, name, default)
end

function _required_v5(dataset, names::Tuple)
    for name in names
        value = _metadata_value_v5(dataset, name, nothing)
        if value !== nothing && !isempty(String(value))
            return String(value)
        end
    end
    error("prepared bridge dataset lacks $(join(names, '/'))")
end

function _load_prepared_dataset_v5(path, frozen)
    isfile(path) || error("prepared distillation dataset is absent: $path")
    data = JLD2.load(path)
    haskey(data, "dataset") ||
        error("only prepare_distillation_dataset.jl output is accepted")
    raw = data["dataset"]
    String(_value(raw, :schema, "")) == PREPARED_DATASET_SCHEMA ||
        error("prepared dataset schema mismatch")
    _metadata_value_v5(raw, :mixed_supervision, false) === true ||
        error("prepared dataset is not mixed supervision")

    metadata = _value(raw, :metadata, NamedTuple())
    source_kind = lowercase(String(_value(metadata, :source_kind, "")))
    official_flag = _value(metadata, :official_neuron_source, false)
    source_kind == "official_neuron" && official_flag === true ||
        error("production distillation requires official NEURON source")
    official_schema = String(_value(
        metadata,
        :official_neuron_schema,
        _value(metadata, :source_teacher_schema, ""),
    ))
    official_schema == "hd_swsnn_twinprop.neuron_teacher.v1" ||
        error("official NEURON teacher schema is absent or mismatched")
    completion = lowercase(String(_value(
        metadata,
        :source_completion_state,
        _value(metadata, :completion_state, ""),
    )))
    completion == "complete" ||
        error("official NEURON source is not completion_state=complete")

    twin_gate = _value(metadata, :twin_gate, nothing)
    twin_gate !== nothing &&
        _value(twin_gate, :passed, false) === true ||
        error("prepared dataset lacks passed frozen-twin gate")
    twin_metrics = _value(metadata, :twin_held_out_metrics, nothing)
    twin_metrics === nothing &&
        error("prepared dataset lacks frozen-twin held-out metrics")
    twin_auroc = Float64(_value(twin_metrics, :spike_auroc, NaN))
    isfinite(twin_auroc) && twin_auroc >= PRODUCTION_TWIN_AUROC ||
        error(
            "frozen twin held-out spike AUROC $twin_auroc is below " *
            "$PRODUCTION_TWIN_AUROC",
        )

    input = _value(raw, :input)
    ndims(input) == 3 ||
        throw(DimensionMismatch("input must be input_dim x time x sample"))
    frozen.model.config.segments ==
        ProductionCell.OFFICIAL_HAY_SEGMENTS ||
        error("frozen twin is not the official 642-segment model")
    size(input, 1) == frozen.model.config.input_dim ||
        throw(DimensionMismatch("input/frozen-twin dimension mismatch"))
    time_steps = size(input, 2)
    samples = size(input, 3)
    cached_voltage = _value(raw, :target_voltage)
    cached_spike = _value(raw, :target_spike)
    cached_nmda = _value(raw, :target_nmda)
    target_calcium = _value(raw, :target_calcium_event)
    target_dendritic = _value(raw, :target_dendritic_voltage)
    size(cached_voltage) == (time_steps, samples) ||
        throw(DimensionMismatch("cached twin voltage shape differs"))
    size(cached_spike) == (time_steps, samples) ||
        throw(DimensionMismatch("cached twin spike shape differs"))
    size(cached_nmda) == (4, time_steps, samples) ||
        throw(DimensionMismatch("cached twin NMDA shape differs"))
    size(target_calcium) == (time_steps, samples) ||
        throw(DimensionMismatch("detailed calcium-event shape differs"))
    size(target_dendritic) == (4, time_steps, samples) ||
        throw(DimensionMismatch("detailed dendritic-voltage shape differs"))

    twin_parameter_hash = _required_v5(
        raw,
        (:frozen_twin_parameter_hash,),
    )
    twin_artifact_hash = _required_v5(
        raw,
        (:frozen_twin_artifact_hash, :digital_twin_hash),
    )
    twin_parameter_hash == frozen.parameter_sha256 ||
        error("frozen-twin parameter lineage mismatch")
    twin_artifact_hash == frozen.artifact_sha256 ||
        error("frozen-twin artifact lineage mismatch")

    segment_region = _metadata_value_v5(raw, :segment_region, nothing)
    segment_region === nothing &&
        (segment_region = _metadata_value_v5(
            raw,
            :official_segment_region,
            nothing,
        ))
    segment_region !== nothing &&
        length(segment_region) == ProductionCell.OFFICIAL_HAY_SEGMENTS ||
        error("prepared dataset lacks the official 642 segment-region map")

    train_indices = Int.(_value(raw, :train_indices))
    validation_indices = Int.(_value(raw, :validation_indices))
    test_indices = Int.(_value(raw, :test_indices))
    all(!isempty, (train_indices, validation_indices, test_indices)) ||
        error("prepared dataset split is empty")
    provenance = (;
        detailed_teacher_hash=_required_v5(
            raw,
            (:detailed_teacher_hash, :teacher_contract_sha256),
        ),
        detailed_kernel_hash=_required_v5(
            raw,
            (:detailed_kernel_hash, :cell_mechanism_sha256),
        ),
        morphology_hash=_required_v5(
            raw,
            (:morphology_hash, :morphology_sha256),
        ),
        official_modeldb_source_hash=_required_v5(
            raw,
            (:official_modeldb_source_hash,),
        ),
        official_neuron_schema=official_schema,
        frozen_twin_parameter_hash=twin_parameter_hash,
        frozen_twin_artifact_hash=twin_artifact_hash,
        source_dataset_hash=_required_v5(
            raw,
            (:original_dataset_sha256, :dataset_sha256),
        ),
        source_manifest_hash=_required_v5(
            raw,
            (:source_manifest_sha256,),
        ),
        twin_held_out_spike_auroc=twin_auroc,
    )
    return (;
        schema=PREPARED_DATASET_SCHEMA,
        input=Float32.(input),
        cached_twin_voltage=Float32.(cached_voltage),
        cached_twin_spike=Float32.(cached_spike),
        cached_twin_nmda=Float32.(cached_nmda),
        detailed_calcium_event=Float32.(target_calcium),
        detailed_dendritic_voltage=Float32.(target_dendritic),
        train_indices,
        validation_indices,
        test_indices,
        segment_region,
        provenance,
        file_sha256=_sha256_file(path),
    )
end

function _semantic_target_v5(normalized)
    apical_context = tanh.(
        0.25f0 .* normalized[10:10, :] .+
        0.50f0 .* normalized[11:11, :] .+
        0.25f0 .* normalized[7:7, :],
    )
    return vcat(
        tanh.(normalized[8:11, :]),
        tanh.(normalized[3:6, :]),
        apical_context,
        tanh.(normalized[1:1, :]),
        normalized[7:7, :],
    )
end

function _initial_parameters_v5(rng, segments)
    spatial_logits =
        -2.0f0 .+ 0.02f0 .* randn(rng, Float32, 4, segments)
    # Region-balanced deterministic seed. Learned soft locations subsequently
    # refine every official segment column.
    @inbounds for segment in 1:segments
        spatial_logits[mod1(segment, 4), segment] += 4.0f0
    end
    return (;
        transition_decay_logit=fill(1.4f0, 11),
        recurrent_weight=
            0.08f0 .* randn(rng, Float32, 11, 11) .+
            0.55f0 .* Matrix{Float32}(I, 11, 11),
        input_weight=0.10f0 .* randn(rng, Float32, 11, 16),
        transition_bias=zeros(Float32, 11, 1),
        readout_weight=
            Matrix{Float32}(I, 11, 11) .+
            0.02f0 .* randn(rng, Float32, 11, 11),
        readout_bias=zeros(Float32, 11, 1),
        initial_state=zeros(Float32, 11, 1),
        spatial_logits,
    )
end

function _sequence_loss_v5(
    parameters,
    raw_input,
    target,
    target_mean,
    target_scale,
    segments,
    free_fraction::Float32,
)
    time_steps = size(raw_input, 2)
    batch = size(raw_input, 3)
    state = repeat(parameters.initial_state, 1, batch)
    output_loss = 0.0f0
    spike_loss = 0.0f0
    calcium_loss = 0.0f0
    semantic_loss = 0.0f0
    for time in 1:time_steps
        input = _project_twin_input(
            raw_input[:, time, :],
            parameters.spatial_logits,
            segments,
        )
        predicted_state = _transition(parameters, state, input)
        output = parameters.readout_weight * predicted_state .+
            parameters.readout_bias
        normalized = _normalized_target(
            target[:, time, :],
            target_mean,
            target_scale,
        )
        semantic = _semantic_target_v5(normalized)
        output_loss += sum(abs2, output[1:1, :] .- normalized[1:1, :])
        output_loss += 0.75f0 *
            sum(abs2, output[3:6, :] .- normalized[3:6, :])
        output_loss += 0.50f0 *
            sum(abs2, output[8:11, :] .- normalized[8:11, :])
        spike_loss += sum(_bce_logit.(
            output[2:2, :],
            normalized[2:2, :],
        ))
        calcium_loss += sum(_bce_logit.(
            output[7:7, :],
            normalized[7:7, :],
        ))
        semantic_loss += sum(abs2, predicted_state .- semantic)
        state = free_fraction .* predicted_state .+
            (1.0f0 - free_fraction) .* semantic
    end
    count = Float32(time_steps * batch)
    return output_loss / (9.0f0 * count) +
           4.0f0 * spike_loss / count +
           1.5f0 * calcium_loss / count +
           2.0f0 * semantic_loss / (11.0f0 * count) +
           1.0f-5 * (
               sum(abs2, parameters.recurrent_weight) +
               sum(abs2, parameters.input_weight) +
               sum(abs2, parameters.readout_weight)
           )
end

function _train_v5(
    rng,
    dataset,
    target,
    target_mean,
    target_scale,
    segments,
    config,
)
    parameters = _initial_parameters_v5(rng, segments)
    optimizer_state = Optimisers.setup(
        Optimisers.Adam(config.learning_rate),
        parameters,
    )
    history = NamedTuple[]
    for epoch in 1:config.epochs
        fraction = _free_fraction(epoch, config)
        total = 0.0
        started = time()
        for _ in 1:config.steps_per_epoch
            input_window, target_window = _sample_window(
                rng,
                dataset,
                target,
                dataset.train_indices,
                config,
            )
            loss, gradient = Zygote.withgradient(parameters) do candidate
                _sequence_loss_v5(
                    candidate,
                    input_window,
                    target_window,
                    target_mean,
                    target_scale,
                    segments,
                    fraction,
                )
            end
            isfinite(loss) || error("semantic distillation loss is non-finite")
            optimizer_state, parameters = Optimisers.update(
                optimizer_state,
                parameters,
                only(gradient),
            )
            total += Float64(loss)
        end
        record = (;
            epoch,
            loss=total / config.steps_per_epoch,
            free_rollout_fraction=fraction,
            elapsed_seconds=time() - started,
        )
        push!(history, record)
        @printf(
            "production distill epoch %d/%d loss=%.6f free=%.3f time=%.2fs\n",
            epoch,
            config.epochs,
            record.loss,
            fraction,
            record.elapsed_seconds,
        )
        flush(stdout)
    end
    return parameters, history
end

function _rollout_v5(parameters, raw_input, target, target_mean, target_scale, segments)
    time_steps = size(raw_input, 2)
    batch = size(raw_input, 3)
    state = repeat(parameters.initial_state, 1, batch)
    output = Array{Float32}(undef, 11, time_steps, batch)
    semantic_error = 0.0
    for time in 1:time_steps
        input = _project_twin_input(
            raw_input[:, time, :],
            parameters.spatial_logits,
            segments,
        )
        state = _transition(parameters, state, input)
        output[:, time, :] =
            parameters.readout_weight * state .+
            parameters.readout_bias
        normalized = _normalized_target(
            target[:, time, :],
            target_mean,
            target_scale,
        )
        semantic = _semantic_target_v5(normalized)
        semantic_error += sum(abs2, state .- semantic)
    end
    semantic_rmse = sqrt(
        semantic_error / (11 * time_steps * batch),
    )
    return output, semantic_rmse
end

function _metrics_v5(
    parameters,
    dataset,
    target,
    indices,
    segments,
    target_mean,
    target_scale,
)
    raw, semantic_rmse = _rollout_v5(
        parameters,
        dataset.input[:, :, indices],
        target[:, :, indices],
        target_mean,
        target_scale,
        segments,
    )
    prediction = _physical_output(raw, target_mean, target_scale)
    truth = target[:, :, indices]
    return (;
        samples=length(indices),
        free_rollout_horizon=size(truth, 2),
        soma_voltage_rmse_mv=_rmse(prediction[1, :, :], truth[1, :, :]),
        soma_voltage_correlation=
            _correlation(prediction[1, :, :], truth[1, :, :]),
        spike_auroc=_auroc(prediction[2, :, :], truth[2, :, :]),
        nmda_rmse_by_region=[
            _rmse(prediction[2 + region, :, :], truth[2 + region, :, :])
            for region in 1:4
        ],
        nmda_correlation_by_region=[
            _correlation(
                prediction[2 + region, :, :],
                truth[2 + region, :, :],
            )
            for region in 1:4
        ],
        calcium_event_auroc=
            _auroc(prediction[7, :, :], truth[7, :, :]),
        dendritic_voltage_rmse_mv=[
            _rmse(prediction[7 + region, :, :], truth[7 + region, :, :])
            for region in 1:4
        ],
        semantic_state_rmse=semantic_rmse,
    )
end

function _production_gate_v5(test_metrics, minimum_spike_auroc)
    minimum_spike_auroc >= PRODUCTION_TWIN_AUROC ||
        error("production minimum spike AUROC cannot be below 0.985")
    spike_passed = isfinite(test_metrics.spike_auroc) &&
        test_metrics.spike_auroc >= minimum_spike_auroc
    voltage_passed =
        isfinite(test_metrics.soma_voltage_rmse_mv) &&
        isfinite(test_metrics.soma_voltage_correlation) &&
        test_metrics.soma_voltage_rmse_mv <=
            PRODUCTION_VOLTAGE_RMSE_MV &&
        test_metrics.soma_voltage_correlation >=
            PRODUCTION_VOLTAGE_CORRELATION
    nmda_passed =
        all(isfinite, test_metrics.nmda_rmse_by_region) &&
        all(isfinite, test_metrics.nmda_correlation_by_region) &&
        mean(test_metrics.nmda_correlation_by_region) >=
            PRODUCTION_NMDA_CORRELATION
    calcium_passed =
        isfinite(test_metrics.calcium_event_auroc) &&
        test_metrics.calcium_event_auroc >=
            PRODUCTION_CALCIUM_AUROC
    dendritic_voltage_passed =
        all(isfinite, test_metrics.dendritic_voltage_rmse_mv) &&
        mean(test_metrics.dendritic_voltage_rmse_mv) <=
            PRODUCTION_DENDRITIC_RMSE_MV
    semantic_state_passed =
        isfinite(test_metrics.semantic_state_rmse) &&
        test_metrics.semantic_state_rmse <=
            PRODUCTION_SEMANTIC_STATE_RMSE
    multi_target_passed =
        voltage_passed &&
        nmda_passed &&
        calcium_passed &&
        dendritic_voltage_passed &&
        semantic_state_passed
    return (;
        passed=spike_passed && multi_target_passed,
        minimum_spike_auroc,
        held_out_spike_auroc=test_metrics.spike_auroc,
        spike_passed,
        multi_target_passed,
        voltage_passed,
        nmda_passed,
        calcium_passed,
        dendritic_voltage_passed,
        semantic_state_passed,
        thresholds=(;
            voltage_rmse_mv=PRODUCTION_VOLTAGE_RMSE_MV,
            voltage_correlation=PRODUCTION_VOLTAGE_CORRELATION,
            nmda_mean_correlation=PRODUCTION_NMDA_CORRELATION,
            calcium_auroc=PRODUCTION_CALCIUM_AUROC,
            dendritic_rmse_mv=PRODUCTION_DENDRITIC_RMSE_MV,
            semantic_state_rmse=PRODUCTION_SEMANTIC_STATE_RMSE,
        ),
    )
end

function _parse_production_arguments_v5(arguments)
    config = _parse_arguments(arguments)
    explicit_minimum = any(
        ==("--minimum-spike-auroc"),
        arguments,
    )
    if explicit_minimum
        config.minimum_spike_auroc >= PRODUCTION_TWIN_AUROC ||
            error("production --minimum-spike-auroc must be >= 0.985")
        return config
    end
    return merge(
        config,
        (; minimum_spike_auroc=PRODUCTION_TWIN_AUROC),
    )
end

function run_production_distillation_v5(config)
    _validate_config(config)
    config.minimum_spike_auroc >= PRODUCTION_TWIN_AUROC ||
        error("production distilled gate must be at least 0.985")
    frozen = Twin.load_frozen_twin(config.frozen_twin)
    frozen_before = Twin.assert_frozen_unchanged(frozen)
    dataset = _load_prepared_dataset_v5(config.dataset, frozen)
    live_twin = _run_frozen_twin(frozen, dataset.input)
    cache_check = _verify_cached_twin!(dataset, live_twin)
    target = _target_tensor(dataset, live_twin)
    target_mean, target_scale =
        _target_statistics(target, dataset.train_indices)
    config_record = (;
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        batch=config.batch,
        window=min(config.window, size(dataset.input, 2)),
        learning_rate=config.learning_rate,
        free_rollout_epochs=config.free_rollout_epochs,
        seed=config.seed,
        prepared_dataset_schema=dataset.schema,
        official_neuron_schema=
            dataset.provenance.official_neuron_schema,
        official_segment_count=
            ProductionCell.OFFICIAL_HAY_SEGMENTS,
        semantic_state_supervision=true,
        mixed_supervision=(
            twin=(:soma_voltage, :soma_spike, :nmda_current),
            official_neuron=(:calcium_event, :dendritic_voltage),
        ),
        soma_spike_is_sole_external_event=true,
    )
    config_hash = _sha256_value(config_record)
    trained, history = _train_v5(
        Xoshiro(config.seed),
        dataset,
        target,
        target_mean,
        target_scale,
        frozen.model.config.segments,
        config,
    )
    metrics = (;
        train=_metrics_v5(
            trained,
            dataset,
            target,
            dataset.train_indices,
            frozen.model.config.segments,
            target_mean,
            target_scale,
        ),
        validation=_metrics_v5(
            trained,
            dataset,
            target,
            dataset.validation_indices,
            frozen.model.config.segments,
            target_mean,
            target_scale,
        ),
        test=_metrics_v5(
            trained,
            dataset,
            target,
            dataset.test_indices,
            frozen.model.config.segments,
            target_mean,
            target_scale,
        ),
    )
    gate = _production_gate_v5(
        metrics.test,
        config.minimum_spike_auroc,
    )
    parameters = _freeze(
        trained,
        dataset,
        frozen,
        target_mean,
        target_scale,
        config_hash,
    )
    size(parameters.compartment_projection) ==
        (4, ProductionCell.OFFICIAL_HAY_SEGMENTS) ||
        error("learned location projection is not 4 x 642")
    parameter_hash = Cell.parameter_sha256(parameters)
    location_mapping_hash =
        _sha256_value(parameters.compartment_projection)
    frozen_after = Twin.assert_frozen_unchanged(frozen)
    frozen_before == frozen_after ||
        error("frozen digital twin changed during distillation")

    timestamp = Dates.format(now(UTC), dateformat"yyyymmddTHHMMSS")
    destination = gate.passed ?
        config.output :
        config.output * ".failed." * timestamp * ".jld2"
    payload = (;
        schema=Cell.DISTILLED_ARTIFACT_SCHEMA,
        parameters,
        parameter_sha256=parameter_hash,
        frozen_internal=true,
        ablation_mode=:full,
        official_segment_count=
            ProductionCell.OFFICIAL_HAY_SEGMENTS,
        location_mapping_sha256=location_mapping_hash,
        teacher_hash=dataset.provenance.detailed_teacher_hash,
        cell_mechanism_sha256=
            dataset.provenance.detailed_kernel_hash,
        detailed_kernel_hash=
            dataset.provenance.detailed_kernel_hash,
        morphology_hash=dataset.provenance.morphology_hash,
        digital_twin_sha256=frozen.artifact_sha256,
        digital_twin_hash=frozen.artifact_sha256,
        official_modeldb_source_hash=
            dataset.provenance.official_modeldb_source_hash,
        frozen_twin_parameter_hash=frozen.parameter_sha256,
        frozen_twin_artifact_hash=frozen.artifact_sha256,
        distillation_dataset_hash=dataset.file_sha256,
        distillation_config_hash=config_hash,
        source_dataset_hash=
            dataset.provenance.source_dataset_hash,
        source_manifest_hash=
            dataset.provenance.source_manifest_hash,
        mixed_supervision=(;
            twin_targets=(:soma_voltage, :soma_spike, :nmda_current),
            official_neuron_targets=
                (:calcium_event, :dendritic_voltage),
            live_frozen_twin_inference=true,
            cache_check,
        ),
        frozen_twin_integrity_before=frozen_before,
        frozen_twin_integrity_after=frozen_after,
        metrics,
        gate,
        config=config_record,
        history,
        created_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
    )
    _atomic_jldsave(destination; payload)
    if gate.passed
        runtime =
            ProductionCell.load_production_distilled_artifact(destination)
        ProductionCell.checkpoint_frozen_digest(runtime)
    end
    report = (;
        schema=payload.schema,
        accepted=gate.passed,
        requested_output=config.output,
        artifact_path=destination,
        artifact_sha256=Cell.artifact_sha256(destination),
        parameter_sha256=parameter_hash,
        location_mapping_sha256=location_mapping_hash,
        teacher_hash=payload.teacher_hash,
        digital_twin_sha256=payload.digital_twin_sha256,
        metrics,
        gate,
        cache_check,
        history,
    )
    _atomic_json(config.metrics, report)
    @info "production 11-state distillation finished" report.accepted report.artifact_path report.gate report.metrics.test
    gate.passed || error(
        "production multi-target gate failed; candidate retained at " *
        destination * " and accepted artifact path was not modified",
    )
    return report
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_production_distillation_v5(
        _parse_production_arguments_v5(ARGS),
    )
end
