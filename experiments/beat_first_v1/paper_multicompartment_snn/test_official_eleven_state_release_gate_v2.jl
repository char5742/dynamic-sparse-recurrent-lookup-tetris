using Test

include(joinpath(@__DIR__, "OfficialElevenStateReleaseGateV2.jl"))
using .OfficialElevenStateReleaseGateV2

function good_metrics()
    return (;
        evaluation_split="test",
        exact_dataset_split=true,
        evaluated_indices_sha256="a"^64,
        evaluated_index_count=8,
        samples=8,
        evaluated_time_indices_one_based=(501, 1500),
        evaluated_steps_per_trial=1000,
        state_warmup_steps=500,
        expected_dense_observations=8000,
        physical_observation_count_by_coordinate=(
            8000, 8000, 8000, 8000, 8000, 8000,
            80, 80, 80, 80, 80,
        ),
        semantic_observation_count_by_coordinate=ntuple(_ -> 80, 11),
        spike_positive_observations=100,
        spike_negative_observations=7900,
        calcium_positive_observations=10,
        calcium_negative_observations=70,
        sparse_auxiliary_metrics_use_observed_times_only=true,
        interpolated_auxiliary_values_excluded_from_gate=true,
        auroc_is_conservative_lower_bound=true,
        soma_voltage_rmse_mv=1.0,
        soma_voltage_correlation=0.99,
        nmda_rmse_by_region=[0.2, 0.2, 0.2, 0.2],
        nmda_correlation_by_region=[0.9, 0.9, 0.9, 0.9],
        spike_auroc=0.99,
        calcium_event_auroc=0.9,
        dendritic_voltage_rmse_mv=[1.0, 1.0, 1.0, 1.0],
        semantic_coordinate_passed=trues(11),
    )
end

function with_metric(metrics, name, value)
    return merge(metrics, NamedTuple{(name,)}((value,)))
end

@testset "strict final-v2 eleven-state gate" begin
    scale = ones(Float32, 11)
    good = good_metrics()
    gate = strict_release_gate(
        good,
        scale;
        expected_test_indices_sha256="a"^64,
    )
    @test gate.passed
    @test all(gate.nmda_region_passed)
    @test all(gate.dendritic_branch_passed)

    catastrophic_nmda = with_metric(
        good,
        :nmda_rmse_by_region,
        [0.2, 0.2, 0.2, 1.0e9],
    )
    @test !strict_release_gate(
        catastrophic_nmda,
        scale;
        expected_test_indices_sha256="a"^64,
    ).passed

    hidden_bad_correlation = with_metric(
        good,
        :nmda_correlation_by_region,
        [0.99, 0.99, 0.99, -1.0],
    )
    @test !strict_release_gate(
        hidden_bad_correlation,
        scale;
        expected_test_indices_sha256="a"^64,
    ).passed

    catastrophic_dendrite = with_metric(
        good,
        :dendritic_voltage_rmse_mv,
        [1.0, 1.0, 1.0, 31.0],
    )
    @test !strict_release_gate(
        catastrophic_dendrite,
        scale;
        expected_test_indices_sha256="a"^64,
    ).passed

    missing_aux = with_metric(
        good,
        :physical_observation_count_by_coordinate,
        (8000, 8000, 8000, 8000, 8000, 8000, 0, 80, 80, 80, 80),
    )
    @test_throws ErrorException strict_release_gate(
        missing_aux,
        scale;
        expected_test_indices_sha256="a"^64,
    )

    @test_throws ErrorException strict_release_gate(
        with_metric(good, :evaluation_split, "validation"),
        scale;
        expected_test_indices_sha256="a"^64,
    )
    @test_throws ErrorException strict_release_gate(
        good,
        scale;
        expected_test_indices_sha256="b"^64,
    )
end
