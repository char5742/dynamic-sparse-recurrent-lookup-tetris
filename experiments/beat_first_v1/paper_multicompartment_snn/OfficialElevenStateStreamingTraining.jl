module OfficialElevenStateStreamingTraining

using Optimisers
using Printf
using Random
using Statistics
using Zygote

const _PARENT = parentmodule(@__MODULE__)
if !isdefined(_PARENT, :OfficialElevenStateDistillationCore)
    Base.include(
        _PARENT,
        joinpath(
            @__DIR__,
            "OfficialElevenStateDistillationCore.jl",
        ),
    )
end
if !isdefined(_PARENT, :StreamingReleaseMetrics)
    Base.include(
        _PARENT,
        joinpath(@__DIR__, "StreamingReleaseMetrics.jl"),
    )
end
const Core =
    getfield(_PARENT, :OfficialElevenStateDistillationCore)
const Metrics = getfield(_PARENT, :StreamingReleaseMetrics)

export MAXIMUM_COORDINATE_RMSE,
    MAXIMUM_DENDRITIC_RMSE_MV,
    MAXIMUM_VOLTAGE_RMSE_MV,
    MINIMUM_CALCIUM_AUROC,
    MINIMUM_COORDINATE_CORRELATION,
    MINIMUM_NMDA_CORRELATION,
    MINIMUM_VOLTAGE_CORRELATION,
    evaluate_streaming,
    gate_metrics,
    train_streaming

const MAXIMUM_VOLTAGE_RMSE_MV = 5.0
const MINIMUM_VOLTAGE_CORRELATION = 0.90
const MINIMUM_NMDA_CORRELATION = 0.80
const MINIMUM_CALCIUM_AUROC = 0.80
const MAXIMUM_DENDRITIC_RMSE_MV = 8.0
const MAXIMUM_COORDINATE_RMSE = 0.70
const MINIMUM_COORDINATE_CORRELATION = 0.70

function _free_fraction(epoch::Int, config)
    teacher_epochs =
        max(config.epochs - config.free_rollout_epochs, 1)
    epoch > teacher_epochs && return 1.0f0
    return Float32(epoch - 1) / Float32(teacher_epochs)
end

function _sample_window(rng, dataset, materialize_window, config)
    window = min(config.window, dataset.time_steps)
    first_time =
        rand(rng, 1:(dataset.time_steps - window + 1))
    samples = rand(rng, dataset.train_indices, config.batch)
    batch = materialize_window(
        dataset,
        samples,
        first_time,
        window,
    )
    return batch.raw_input, batch.target
end

function train_streaming(
    rng,
    dataset,
    materialize_window,
    target_mean,
    target_scale,
    config,
)
    parameters = Core.initial_parameters(rng)
    optimizer_state = Optimisers.setup(
        Optimisers.Adam(config.learning_rate),
        parameters,
    )
    history = NamedTuple[]
    for epoch in 1:config.epochs
        free_fraction = _free_fraction(epoch, config)
        total_loss = 0.0
        started = time()
        for _ in 1:config.steps_per_epoch
            input_window, target_window =
                _sample_window(
                    rng,
                    dataset,
                    materialize_window,
                    config,
                )
            loss, gradient = Zygote.withgradient(parameters) do candidate
                Core.sequence_loss(
                    candidate,
                    input_window,
                    target_window,
                    target_mean,
                    target_scale,
                    free_fraction,
                )
            end
            isfinite(loss) ||
                error("official 11-state distillation loss is non-finite")
            optimizer_state, parameters = Optimisers.update(
                optimizer_state,
                parameters,
                only(gradient),
            )
            Core.all_finite(parameters) ||
                error("official 11-state parameter became non-finite")
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
            "official 1278 -> 11-state %d/%d loss=%.6f free=%.3f time=%.2fs\n",
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

function _semantic_validity(observed)
    validity = falses(11)
    validity[1:4] .= observed[8:11]
    validity[5:8] .= observed[3:6]
    validity[9] =
        observed[7] && observed[10] && observed[11]
    validity[10] = observed[1]
    validity[11] = observed[7]
    return validity
end

function _update_metrics!(
    metrics::Metrics.StreamingMetrics,
    prediction,
    truth,
    state,
    semantic,
    physical_validity,
    semantic_validity,
)
    @inbounds for coordinate in 1:11
        if physical_validity[coordinate]
            predicted = Float64(prediction[coordinate])
            target = Float64(truth[coordinate])
            isfinite(predicted) && isfinite(target) ||
                error("official physical metric is non-finite")
            Metrics._update_pair!(
                metrics.output_statistics[coordinate],
                predicted,
                target,
            )
            coordinate == 2 &&
                Metrics._update_auroc!(
                    metrics.spike_auroc,
                    predicted,
                    target,
                )
            coordinate == 7 &&
                Metrics._update_auroc!(
                    metrics.calcium_auroc,
                    predicted,
                    target,
                )
        end
        if semantic_validity[coordinate]
            predicted = Float64(state[coordinate])
            target = Float64(semantic[coordinate])
            isfinite(predicted) && isfinite(target) ||
                error("official semantic metric is non-finite")
            Metrics._update_pair!(
                metrics.semantic_statistics[coordinate],
                predicted,
                target,
            )
        end
    end
    metrics.observations += 1
    return metrics
end

function _conservative_auroc(histogram)
    estimate = Metrics._auroc(histogram)
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

function _finalize_metrics(metrics)
    output_rmse = Metrics.output_rmse(metrics)
    output_correlation = Metrics.output_correlation(metrics)
    coordinate_rmse = Metrics.semantic_rmse(metrics)
    coordinate_correlation = Metrics.semantic_correlation(metrics)
    coordinate_passed = [
        isfinite(coordinate_rmse[index]) &&
        isfinite(coordinate_correlation[index]) &&
        coordinate_rmse[index] <= MAXIMUM_COORDINATE_RMSE &&
        coordinate_correlation[index] >= MINIMUM_COORDINATE_CORRELATION
        for index in 1:11
    ]
    spike = _conservative_auroc(metrics.spike_auroc)
    calcium = _conservative_auroc(metrics.calcium_auroc)
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
        semantic_coordinate_names=Core.SEMANTIC_COORDINATE_NAMES,
        semantic_coordinate_rmse=coordinate_rmse,
        semantic_coordinate_correlation=coordinate_correlation,
        semantic_coordinate_passed=coordinate_passed,
        sparse_auxiliary_metrics_use_observed_times_only=true,
        interpolated_auxiliary_values_excluded_from_gate=true,
        auroc_is_conservative_lower_bound=true,
        auroc_histogram_bins=length(
            metrics.spike_auroc.positive,
        ),
    )
end

"""
Bounded-memory free rollout evaluation.

The materializer-provided `observed` mask is mandatory.  Calcium, detailed
dendritic voltage, and all semantic coordinates depending on them contribute
to release gates only at actual diagnostic samples, never at interpolated fit
points.
"""
function evaluate_streaming(
    parameters,
    dataset,
    materialize_window,
    indices,
    target_mean,
    target_scale;
    time_chunk::Integer,
    auroc_bins::Integer,
)
    selected = Int.(collect(indices))
    chunk = Int(time_chunk)
    chunk >= 1 || throw(ArgumentError("time_chunk must be positive"))
    metrics = Metrics.StreamingMetrics(
        length(selected),
        dataset.time_steps;
        auroc_bins=Int(auroc_bins),
        maximum_coordinate_rmse=MAXIMUM_COORDINATE_RMSE,
        minimum_coordinate_correlation=
            MINIMUM_COORDINATE_CORRELATION,
    )
    for global_index in selected
        state = repeat(parameters.initial_state, 1, 1)
        for first_time in 1:chunk:dataset.time_steps
            count = min(
                chunk,
                dataset.time_steps - first_time + 1,
            )
            batch = materialize_window(
                dataset,
                [global_index],
                first_time,
                count,
            )
            raw_input = @view batch.raw_input[:, :, 1]
            target = @view batch.target[:, :, 1]
            observed = @view batch.observed[:, :, 1]
            for local_time in 1:count
                input = Core.project_official_input(
                    @view(raw_input[:, local_time:local_time]),
                    parameters.location_logits,
                )
                state = Core.transition(parameters, state, input)
                raw_output =
                    Core.structured_readout(parameters, state)
                normalized = Core.normalize_target(
                    @view(target[:, local_time:local_time]),
                    target_mean,
                    target_scale,
                )
                semantic = Core.semantic_target(normalized)
                physical = Core.physical_output(
                    @view(raw_output[:, 1]),
                    target_mean,
                    target_scale,
                )
                physical_validity =
                    @view observed[:, local_time]
                semantic_validity =
                    _semantic_validity(physical_validity)
                _update_metrics!(
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
    return _finalize_metrics(metrics)
end

function gate_metrics(test_metrics, minimum_spike_auroc::Real)
    spike_passed =
        isfinite(test_metrics.spike_auroc) &&
        test_metrics.spike_auroc >= minimum_spike_auroc
    voltage_passed =
        isfinite(test_metrics.soma_voltage_rmse_mv) &&
        isfinite(test_metrics.soma_voltage_correlation) &&
        test_metrics.soma_voltage_rmse_mv <=
            MAXIMUM_VOLTAGE_RMSE_MV &&
        test_metrics.soma_voltage_correlation >=
            MINIMUM_VOLTAGE_CORRELATION
    nmda_passed =
        all(isfinite, test_metrics.nmda_rmse_by_region) &&
        all(isfinite, test_metrics.nmda_correlation_by_region) &&
        mean(test_metrics.nmda_correlation_by_region) >=
            MINIMUM_NMDA_CORRELATION
    calcium_passed =
        isfinite(test_metrics.calcium_event_auroc) &&
        test_metrics.calcium_event_auroc >=
            MINIMUM_CALCIUM_AUROC
    dendritic_voltage_passed =
        all(isfinite, test_metrics.dendritic_voltage_rmse_mv) &&
        mean(test_metrics.dendritic_voltage_rmse_mv) <=
            MAXIMUM_DENDRITIC_RMSE_MV
    semantic_coordinate_passed =
        all(test_metrics.semantic_coordinate_passed)
    multi_target_passed =
        voltage_passed &&
        nmda_passed &&
        calcium_passed &&
        dendritic_voltage_passed &&
        semantic_coordinate_passed
    return (;
        passed=spike_passed && multi_target_passed,
        minimum_spike_auroc=Float64(minimum_spike_auroc),
        held_out_spike_auroc=test_metrics.spike_auroc,
        spike_passed,
        multi_target_passed,
        voltage_passed,
        nmda_passed,
        calcium_passed,
        dendritic_voltage_passed,
        semantic_state_passed=semantic_coordinate_passed,
        semantic_coordinate_passed,
    )
end

end # module OfficialElevenStateStreamingTraining
