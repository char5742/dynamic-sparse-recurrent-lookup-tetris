using Test
using Statistics

include(joinpath(@__DIR__, "StreamingReleaseMetrics.jl"))
using .StreamingReleaseMetrics

function batch_rmse(prediction, truth)
    return sqrt(mean(abs2, Float64.(prediction) .- Float64.(truth)))
end

function batch_correlation(prediction, truth)
    x = Float64.(prediction)
    y = Float64.(truth)
    x .-= mean(x)
    y .-= mean(y)
    denominator = sqrt(sum(abs2, x) * sum(abs2, y))
    denominator <= eps(Float64) && return NaN
    return sum(x .* y) / denominator
end

@testset "bounded-memory release metrics" begin
    samples = 3
    horizon = 4
    observations = samples * horizon
    physical_prediction = Matrix{Float64}(undef, 11, observations)
    physical_truth = Matrix{Float64}(undef, 11, observations)
    state = Matrix{Float64}(undef, 11, observations)
    semantic = Matrix{Float64}(undef, 11, observations)
    validity = trues(11, observations)

    for observation in 1:observations, coordinate in 1:11
        target = 0.15 * coordinate + 0.07 * observation
        physical_truth[coordinate, observation] = target
        physical_prediction[coordinate, observation] =
            target + 0.01 * coordinate * sin(observation)
        semantic[coordinate, observation] =
            0.02 * coordinate + 0.03 * observation
        state[coordinate, observation] =
            semantic[coordinate, observation] +
            0.004 * coordinate * cos(observation)
    end

    # Physical spike/Ca channels are probabilities and use separable bins, so
    # the histogram implementation has exact AUROC=1 in this test.
    for observation in 1:observations
        label = isodd(observation) ? 1.0 : 0.0
        physical_truth[2, observation] = label
        physical_prediction[2, observation] = label == 1.0 ? 0.85 : 0.15
        physical_truth[7, observation] = label
        physical_prediction[7, observation] = label == 1.0 ? 0.75 : 0.25
    end

    # Exercise coordinate-specific missing values without storing them in the
    # streaming accumulator.
    validity[4, 3] = false
    validity[9, 8] = false
    physical_prediction[4, 3] = NaN
    physical_truth[4, 3] = NaN
    state[4, 3] = NaN
    semantic[4, 3] = NaN
    physical_prediction[9, 8] = NaN
    physical_truth[9, 8] = NaN
    state[9, 8] = NaN
    semantic[9, 8] = NaN

    metrics = StreamingMetrics(samples, horizon; auroc_bins=64)
    for observation in 1:observations
        update!(
            metrics,
            view(physical_prediction, :, observation),
            view(physical_truth, :, observation),
            view(state, :, observation),
            view(semantic, :, observation),
            view(validity, :, observation),
        )
    end
    result = finalize_metrics(metrics)

    @test keys(result) == (
        :samples,
        :free_rollout_horizon,
        :soma_voltage_rmse_mv,
        :soma_voltage_correlation,
        :spike_auroc,
        :nmda_rmse_by_region,
        :nmda_correlation_by_region,
        :calcium_event_auroc,
        :dendritic_voltage_rmse_mv,
        :semantic_coordinate_names,
        :semantic_coordinate_rmse,
        :semantic_coordinate_correlation,
        :semantic_coordinate_passed,
    )
    @test result.samples == samples
    @test result.free_rollout_horizon == horizon
    @test result.spike_auroc == 1.0
    @test result.calcium_event_auroc == 1.0

    for coordinate in 1:11
        included = findall(view(validity, coordinate, :))
        expected_output_rmse = batch_rmse(
            physical_prediction[coordinate, included],
            physical_truth[coordinate, included],
        )
        expected_output_correlation = batch_correlation(
            physical_prediction[coordinate, included],
            physical_truth[coordinate, included],
        )
        expected_semantic_rmse = batch_rmse(
            state[coordinate, included],
            semantic[coordinate, included],
        )
        expected_semantic_correlation = batch_correlation(
            state[coordinate, included],
            semantic[coordinate, included],
        )
        @test output_rmse(metrics)[coordinate] ≈ expected_output_rmse atol=1e-13
        @test output_correlation(metrics)[coordinate] ≈
            expected_output_correlation atol=1e-13
        @test result.semantic_coordinate_rmse[coordinate] ≈
            expected_semantic_rmse atol=1e-13
        @test result.semantic_coordinate_correlation[coordinate] ≈
            expected_semantic_correlation atol=1e-13
    end

    @test length(metrics.spike_auroc.positive) == 64
    @test length(metrics.calcium_auroc.negative) == 64
    @test metrics.observations == observations
end

@testset "validation and completeness" begin
    metrics = StreamingMetrics(1, 2; auroc_bins=16)
    values = collect(range(0.1, 0.9; length=11))
    truth = copy(values)
    truth[2] = 1.0
    truth[7] = 0.0
    state = copy(values)
    semantic = copy(values)
    valid = trues(11)

    update!(metrics, values, truth, state, semantic, valid)
    @test_throws ArgumentError finalize_metrics(metrics)
    @test finalize_metrics(metrics; require_complete=false).samples == 1

    bad = copy(values)
    bad[2] = 1.1
    @test_throws DomainError update!(
        StreamingMetrics(1, 1; auroc_bins=16),
        bad,
        truth,
        state,
        semantic,
        valid,
    )

    invalid = copy(valid)
    invalid[2] = false
    bad[2] = NaN
    masked = StreamingMetrics(1, 1; auroc_bins=16)
    update!(masked, bad, truth, state, semantic, invalid)
    @test isnan(finalize_metrics(masked).spike_auroc)
end
