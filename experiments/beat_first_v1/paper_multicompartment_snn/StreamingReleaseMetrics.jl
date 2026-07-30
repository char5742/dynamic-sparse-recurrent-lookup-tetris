module StreamingReleaseMetrics

export StreamingMetrics,
    FixedHistogramAUROC,
    OnlinePairStatistics,
    update!,
    finalize_metrics,
    output_rmse,
    output_correlation,
    semantic_rmse,
    semantic_correlation,
    reset!

const RELEASE_COORDINATE_NAMES = (
    "basal_dendritic_voltage_1",
    "basal_dendritic_voltage_2",
    "apical_dendritic_voltage_1",
    "apical_dendritic_voltage_2",
    "basal_nmda_current_1",
    "basal_nmda_current_2",
    "apical_nmda_current_1",
    "apical_nmda_current_2",
    "apical_calcium_context",
    "soma_voltage",
    "calcium_adaptation",
)

const DEFAULT_MAXIMUM_COORDINATE_RMSE = 0.70
const DEFAULT_MINIMUM_COORDINATE_CORRELATION = 0.70

"""
    OnlinePairStatistics()

Fixed-size, Float64 streaming statistics for a prediction/target pair.
`squared_error` gives exact running RMSE accumulation, while the Welford
moments give a numerically stable Pearson correlation without retaining
observations.
"""
mutable struct OnlinePairStatistics
    count::Int64
    mean_prediction::Float64
    mean_target::Float64
    prediction_m2::Float64
    target_m2::Float64
    comoment::Float64
    squared_error::Float64
end

OnlinePairStatistics() =
    OnlinePairStatistics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

function reset!(statistics::OnlinePairStatistics)
    statistics.count = 0
    statistics.mean_prediction = 0.0
    statistics.mean_target = 0.0
    statistics.prediction_m2 = 0.0
    statistics.target_m2 = 0.0
    statistics.comoment = 0.0
    statistics.squared_error = 0.0
    return statistics
end

@inline function _update_pair!(
    statistics::OnlinePairStatistics,
    prediction::Float64,
    target::Float64,
)
    next_count = statistics.count + 1
    prediction_delta = prediction - statistics.mean_prediction
    target_delta = target - statistics.mean_target

    next_prediction_mean =
        statistics.mean_prediction + prediction_delta / next_count
    next_target_mean = statistics.mean_target + target_delta / next_count

    statistics.prediction_m2 +=
        prediction_delta * (prediction - next_prediction_mean)
    statistics.target_m2 += target_delta * (target - next_target_mean)
    statistics.comoment +=
        prediction_delta * (target - next_target_mean)
    difference = prediction - target
    statistics.squared_error += difference * difference
    statistics.mean_prediction = next_prediction_mean
    statistics.mean_target = next_target_mean
    statistics.count = next_count
    return nothing
end

@inline function _rmse(statistics::OnlinePairStatistics)
    statistics.count == 0 && return NaN
    return sqrt(statistics.squared_error / statistics.count)
end

@inline function _correlation(statistics::OnlinePairStatistics)
    statistics.count == 0 && return NaN
    denominator =
        sqrt(statistics.prediction_m2 * statistics.target_m2)
    denominator <= eps(Float64) && return NaN
    return statistics.comoment / denominator
end

"""
    FixedHistogramAUROC(bins=4096; minimum_score=0.0, maximum_score=1.0)

Bounded-memory AUROC accumulator. Scores are assigned to a fixed histogram;
ties within a bin receive half credit, exactly as rank-based AUROC does for
equal scores. The default range matches physical spike and calcium
probabilities. Out-of-range scores are rejected rather than silently clamped.
"""
mutable struct FixedHistogramAUROC
    positive::Vector{Int64}
    negative::Vector{Int64}
    minimum_score::Float64
    maximum_score::Float64
    positive_count::Int64
    negative_count::Int64
end

function FixedHistogramAUROC(
    bins::Integer=4096;
    minimum_score::Real=0.0,
    maximum_score::Real=1.0,
)
    bins >= 2 || throw(ArgumentError("AUROC histogram needs at least 2 bins"))
    low = Float64(minimum_score)
    high = Float64(maximum_score)
    isfinite(low) && isfinite(high) && low < high ||
        throw(ArgumentError("AUROC score range must be finite and increasing"))
    return FixedHistogramAUROC(
        zeros(Int64, bins),
        zeros(Int64, bins),
        low,
        high,
        0,
        0,
    )
end

function reset!(histogram::FixedHistogramAUROC)
    fill!(histogram.positive, 0)
    fill!(histogram.negative, 0)
    histogram.positive_count = 0
    histogram.negative_count = 0
    return histogram
end

@inline function _histogram_bin(
    histogram::FixedHistogramAUROC,
    score::Float64,
)
    histogram.minimum_score <= score <= histogram.maximum_score ||
        throw(
            DomainError(
                score,
                "AUROC score is outside the configured histogram range",
            ),
        )
    score == histogram.maximum_score && return length(histogram.positive)
    fraction =
        (score - histogram.minimum_score) /
        (histogram.maximum_score - histogram.minimum_score)
    return floor(Int, fraction * length(histogram.positive)) + 1
end

@inline function _update_auroc!(
    histogram::FixedHistogramAUROC,
    score::Float64,
    target::Float64,
)
    bin = _histogram_bin(histogram, score)
    if target >= 0.5
        histogram.positive[bin] += 1
        histogram.positive_count += 1
    else
        histogram.negative[bin] += 1
        histogram.negative_count += 1
    end
    return nothing
end

function _auroc(histogram::FixedHistogramAUROC)
    positive_count = histogram.positive_count
    negative_count = histogram.negative_count
    (positive_count == 0 || negative_count == 0) && return NaN

    lower_negative = Int64(0)
    favourable_pairs = 0.0
    @inbounds for bin in eachindex(histogram.positive)
        positive = histogram.positive[bin]
        negative = histogram.negative[bin]
        favourable_pairs +=
            Float64(positive) *
            (Float64(lower_negative) + 0.5 * Float64(negative))
        lower_negative += negative
    end
    return favourable_pairs /
           (Float64(positive_count) * Float64(negative_count))
end

"""
    StreamingMetrics(samples, free_rollout_horizon; kwargs...)

Create a bounded-memory release-metric accumulator. One call to `update!`
represents one `(time, sample)` observation. `samples` and
`free_rollout_horizon` are retained so `finalize_metrics` has the same
metadata fields as `_release_metrics`.
"""
mutable struct StreamingMetrics
    samples::Int
    free_rollout_horizon::Int
    observations::Int64
    output_statistics::Vector{OnlinePairStatistics}
    semantic_statistics::Vector{OnlinePairStatistics}
    spike_auroc::FixedHistogramAUROC
    calcium_auroc::FixedHistogramAUROC
    maximum_coordinate_rmse::Float64
    minimum_coordinate_correlation::Float64
end

function StreamingMetrics(
    samples::Integer,
    free_rollout_horizon::Integer;
    auroc_bins::Integer=4096,
    minimum_probability::Real=0.0,
    maximum_probability::Real=1.0,
    maximum_coordinate_rmse::Real=DEFAULT_MAXIMUM_COORDINATE_RMSE,
    minimum_coordinate_correlation::Real=
        DEFAULT_MINIMUM_COORDINATE_CORRELATION,
)
    samples >= 0 || throw(ArgumentError("samples must be nonnegative"))
    free_rollout_horizon >= 0 ||
        throw(ArgumentError("free_rollout_horizon must be nonnegative"))
    maximum_rmse = Float64(maximum_coordinate_rmse)
    minimum_correlation = Float64(minimum_coordinate_correlation)
    isfinite(maximum_rmse) && maximum_rmse >= 0.0 ||
        throw(ArgumentError("maximum coordinate RMSE must be finite"))
    isfinite(minimum_correlation) &&
        -1.0 <= minimum_correlation <= 1.0 ||
        throw(
            ArgumentError(
                "minimum coordinate correlation must lie in [-1, 1]",
            ),
        )
    return StreamingMetrics(
        Int(samples),
        Int(free_rollout_horizon),
        0,
        [OnlinePairStatistics() for _ in 1:11],
        [OnlinePairStatistics() for _ in 1:11],
        FixedHistogramAUROC(
            auroc_bins;
            minimum_score=minimum_probability,
            maximum_score=maximum_probability,
        ),
        FixedHistogramAUROC(
            auroc_bins;
            minimum_score=minimum_probability,
            maximum_score=maximum_probability,
        ),
        maximum_rmse,
        minimum_correlation,
    )
end

function reset!(metrics::StreamingMetrics)
    metrics.observations = 0
    foreach(reset!, metrics.output_statistics)
    foreach(reset!, metrics.semantic_statistics)
    reset!(metrics.spike_auroc)
    reset!(metrics.calcium_auroc)
    return metrics
end

@inline function _check_eleven(name::AbstractString, values)
    length(values) == 11 ||
        throw(DimensionMismatch("$name must contain exactly 11 coordinates"))
    return nothing
end

@inline function _valid_coordinate(validity, coordinate::Int)
    value = validity[coordinate]
    value isa Bool && return value
    value isa Integer &&
        (value == 0 || value == 1) &&
        return value == 1
    throw(
        ArgumentError(
            "validity mask entries must be Bool or integer zero/one",
        ),
    )
end

@inline function _finite_float(name::AbstractString, coordinate::Int, value)
    converted = Float64(value)
    isfinite(converted) ||
        throw(
            DomainError(
                value,
                "$name coordinate $coordinate is valid but non-finite",
            ),
        )
    return converted
end

"""
    update!(metrics, prediction, truth, state, semantic, validity)

Update the accumulator for one time/sample pair. All five inputs must contain
11 coordinates. `prediction` and `truth` are physical release outputs;
`state` and `semantic` are the normalized, semantically identified 11-state
cell coordinates. A false validity entry excludes that coordinate from both
the physical and semantic statistics. Valid entries must be finite.
"""
function update!(
    metrics::StreamingMetrics,
    prediction,
    truth,
    state,
    semantic,
    validity,
)
    _check_eleven("prediction", prediction)
    _check_eleven("truth", truth)
    _check_eleven("state", state)
    _check_eleven("semantic", semantic)
    _check_eleven("validity", validity)

    @inbounds for coordinate in 1:11
        _valid_coordinate(validity, coordinate) || continue
        physical_prediction =
            _finite_float("prediction", coordinate, prediction[coordinate])
        physical_truth =
            _finite_float("truth", coordinate, truth[coordinate])
        semantic_state = _finite_float("state", coordinate, state[coordinate])
        semantic_truth =
            _finite_float("semantic", coordinate, semantic[coordinate])

        _update_pair!(
            metrics.output_statistics[coordinate],
            physical_prediction,
            physical_truth,
        )
        _update_pair!(
            metrics.semantic_statistics[coordinate],
            semantic_state,
            semantic_truth,
        )

        coordinate == 2 &&
            _update_auroc!(
                metrics.spike_auroc,
                physical_prediction,
                physical_truth,
            )
        coordinate == 7 &&
            _update_auroc!(
                metrics.calcium_auroc,
                physical_prediction,
                physical_truth,
            )
    end
    metrics.observations += 1
    return metrics
end

output_rmse(metrics::StreamingMetrics) =
    [_rmse(statistics) for statistics in metrics.output_statistics]

output_correlation(metrics::StreamingMetrics) =
    [_correlation(statistics) for statistics in metrics.output_statistics]

semantic_rmse(metrics::StreamingMetrics) =
    [_rmse(statistics) for statistics in metrics.semantic_statistics]

semantic_correlation(metrics::StreamingMetrics) =
    [_correlation(statistics) for statistics in metrics.semantic_statistics]

"""
    finalize_metrics(metrics; require_complete=true)

Return the same NamedTuple fields as `_release_metrics`. With
`require_complete=true`, the number of `update!` calls must equal
`samples * free_rollout_horizon`; coordinate masks may still reduce the valid
count of individual metrics. Histogram AUROCs are bounded-memory
approximations whose ties are exact at histogram-bin resolution.
"""
function finalize_metrics(
    metrics::StreamingMetrics;
    require_complete::Bool=true,
)
    expected =
        Int128(metrics.samples) * Int128(metrics.free_rollout_horizon)
    if require_complete && Int128(metrics.observations) != expected
        throw(
            ArgumentError(
                "received $(metrics.observations) observations; " *
                "expected $expected",
            ),
        )
    end

    physical_rmse = output_rmse(metrics)
    physical_correlation = output_correlation(metrics)
    coordinate_rmse = semantic_rmse(metrics)
    coordinate_correlation = semantic_correlation(metrics)
    coordinate_passed = [
        isfinite(coordinate_rmse[coordinate]) &&
        isfinite(coordinate_correlation[coordinate]) &&
        coordinate_rmse[coordinate] <= metrics.maximum_coordinate_rmse &&
        coordinate_correlation[coordinate] >=
            metrics.minimum_coordinate_correlation
        for coordinate in 1:11
    ]

    return (;
        samples=metrics.samples,
        free_rollout_horizon=metrics.free_rollout_horizon,
        soma_voltage_rmse_mv=physical_rmse[1],
        soma_voltage_correlation=physical_correlation[1],
        spike_auroc=_auroc(metrics.spike_auroc),
        nmda_rmse_by_region=physical_rmse[3:6],
        nmda_correlation_by_region=physical_correlation[3:6],
        calcium_event_auroc=_auroc(metrics.calcium_auroc),
        dendritic_voltage_rmse_mv=physical_rmse[8:11],
        semantic_coordinate_names=RELEASE_COORDINATE_NAMES,
        semantic_coordinate_rmse=coordinate_rmse,
        semantic_coordinate_correlation=coordinate_correlation,
        semantic_coordinate_passed=coordinate_passed,
    )
end

end # module StreamingReleaseMetrics
