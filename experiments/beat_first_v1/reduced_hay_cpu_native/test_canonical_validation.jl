module CanonicalValidationHarness

using Test
using LinearAlgebra

include("CanonicalValidation.jl")
using .CanonicalValidation

@testset "canonical input collision ledger" begin
    encodings = [
        Bool[true, false, true],
        Bool[false, true, false],
        Bool[true, false, true],
        Bool[true, false, true],
    ]
    targets = [
        Float32[1, 2],
        Float32[3, 4],
        Float32[1, 2],
        Float32[1, 3],
    ]
    ledger = input_collision_ledger(encodings, targets; max_examples=1)
    @test ledger.observations == 4
    @test ledger.unique_encodings == 2
    @test ledger.duplicate_observations == 2
    @test ledger.agreeing_duplicates == 1
    @test ledger.disagreeing_duplicates == 1
    @test ledger.colliding_encodings == 1
    @test ledger.maximum_target_distance == 1.0
    @test length(ledger.examples) == 1
    @test ledger.examples[1].first_index == 1
    @test ledger.examples[1].second_index == 4
    @test ncodeunits(ledger.examples[1].encoding_digest) == 64

    exact = input_collision_ledger([(:empty, 1), (:empty, 1)], [0.0, -0.0])
    @test exact.disagreeing_duplicates == 1
    tolerant = input_collision_ledger(
        [(:empty, 1), (:empty, 1)],
        [1.0, 1.01];
        equivalent=(left, right) -> abs(left - right) <= 0.02,
    )
    @test tolerant.agreeing_duplicates == 1

    @test_throws DimensionMismatch input_collision_ledger([1], [1, 2])
    @test_throws ArgumentError input_collision_ledger([1], [1]; max_examples=-1)
    @test_throws ArgumentError input_collision_ledger([NaN], [1.0])
    @test_throws ArgumentError input_collision_ledger([1.0], [Inf])
    @test_throws ArgumentError input_collision_ledger([Ref(1)], [1])
end

@testset "task Jacobian rank is precision-derived" begin
    identity_summary = task_jacobian_rank_summary(Matrix{Float64}(I, 3, 3))
    @test identity_summary.numerical_rank == 3
    @test identity_summary.maximum_rank == 3
    @test identity_summary.stable_rank == 3.0
    @test identity_summary.effective_rank ≈ 3.0
    @test identity_summary.condition_number == 1.0
    @test identity_summary.tolerance == 3eps(Float64)

    low_rank = [1.0 2.0 3.0; 2.0 4.0 6.0]
    low_rank_summary = task_jacobian_rank_summary(low_rank)
    @test low_rank_summary.numerical_rank == 1
    @test low_rank_summary.stable_rank ≈ 1.0

    source32 = Float32[1 0; 0 eps(Float32) / 4]
    summary32 = task_jacobian_rank_summary(source32)
    @test summary32.tolerance == 2eps(Float32)
    @test summary32.numerical_rank == 1
    explicit = task_jacobian_rank_summary(source32; relative_tolerance=0, absolute_tolerance=0)
    @test explicit.numerical_rank == 2

    @test_throws ArgumentError task_jacobian_rank_summary(zeros(0, 2))
    @test_throws ArgumentError task_jacobian_rank_summary([1.0 NaN])
    @test_throws ArgumentError task_jacobian_rank_summary([1.0;;]; relative_tolerance=-1)
    @test_throws ArgumentError task_jacobian_rank_summary([1.0;;]; absolute_tolerance=Inf)
end

@testset "ordered event trajectory digest" begin
    waves = [
        [(source=1, event=:spike), (source=2, event=:plateau)],
        [(source=3, event=:spike)],
        NamedTuple{(:source, :event),Tuple{Int,Symbol}}[],
    ]
    first_digest = event_trajectory_digest(waves)
    repeated_digest = event_trajectory_digest(deepcopy(waves))
    @test first_digest.sha256 == repeated_digest.sha256
    @test first_digest.wave_count == 3
    @test first_digest.event_count == 3
    @test first_digest.unique_event_count == 3
    @test first_digest.events_per_wave == [2, 1, 0]

    reordered = deepcopy(waves)
    reverse!(reordered[1])
    @test event_trajectory_digest(reordered).sha256 != first_digest.sha256
    repartitioned = [waves[1][1:1], vcat(waves[1][2:2], waves[2]), waves[3]]
    @test event_trajectory_digest(repartitioned).sha256 != first_digest.sha256
    @test event_trajectory_digest(Any[]).event_count == 0
    @test_throws ArgumentError event_trajectory_digest([1, 2])
    @test_throws ArgumentError event_trajectory_digest([[NaN]])
end

@testset "free-logit ListNet oracle floor" begin
    oracle = free_logit_listnet_oracle_floor([
        [0.0, 0.0],
        [1000.0, 999.0, -1000.0],
    ])
    @test oracle.state_count == 2
    @test oracle.candidate_count == 5
    @test oracle.state_entropies[1] ≈ log(2)
    @test oracle.mean_cross_entropy == oracle.mean_teacher_entropy
    @test oracle.mean_excess == 0.0
    @test sum(oracle.centered_free_logits[1]) ≈ 0.0 atol=eps(Float64)
    @test all(probabilities -> sum(probabilities) ≈ 1.0, oracle.teacher_probabilities)

    weighted = free_logit_listnet_oracle_floor(
        [[0.0, 0.0], [0.0]];
        inverse_temperature=2,
        state_weights=[3, 1],
    )
    @test weighted.state_weights == [0.75, 0.25]
    @test weighted.mean_teacher_entropy ≈ 0.75log(2)

    @test_throws ArgumentError free_logit_listnet_oracle_floor(Any[])
    @test_throws ArgumentError free_logit_listnet_oracle_floor([Float64[]])
    @test_throws ArgumentError free_logit_listnet_oracle_floor([[NaN]])
    @test_throws ArgumentError free_logit_listnet_oracle_floor([[1.0]]; inverse_temperature=0)
    @test_throws DimensionMismatch free_logit_listnet_oracle_floor(
        [[1.0], [2.0]];
        state_weights=[1.0],
    )
    @test_throws ArgumentError free_logit_listnet_oracle_floor(
        [[1.0], [2.0]];
        state_weights=[0.0, 0.0],
    )
end

@testset "exact/local alignment and confidence helpers" begin
    same = gradient_alignment([1.0, -2.0], [1.0, -2.0])
    @test same.cosine ≈ 1.0
    @test same.optimal_positive_scale ≈ 1.0
    @test isapprox(same.residual_ratio, 0.0; atol=eps(Float64))

    doubled = gradient_alignment([1.0, -2.0], [2.0, -4.0])
    @test doubled.cosine ≈ 1.0
    @test doubled.optimal_positive_scale ≈ 0.5
    @test isapprox(doubled.residual_ratio, 0.0; atol=eps(Float64))

    inverse = gradient_alignment([1.0, 0.0], [-1.0, 0.0])
    @test inverse.cosine == -1.0
    @test inverse.optimal_positive_scale == 0.0
    @test inverse.residual_ratio == 1.0

    silent = gradient_alignment([1.0, 0.0], [0.0, 0.0])
    @test ismissing(silent.cosine)
    @test silent.optimal_positive_scale == 0.0
    @test silent.residual_ratio == 1.0
    zero = gradient_alignment([0.0, 0.0], [0.0, 0.0])
    @test ismissing(zero.cosine)
    @test ismissing(zero.residual_ratio)

    ci = bounded_mean_confidence_interval(fill(1.0, 1_000))
    @test ci.lower_bound > 0.0
    @test ci.mean == 1.0
    @test ci.upper_bound == 1.0
    @test_throws ArgumentError bounded_mean_confidence_interval(Float64[])
    @test_throws ArgumentError bounded_mean_confidence_interval([2.0])
    @test_throws ArgumentError bounded_mean_confidence_interval([0.0]; confidence=1)

    exact_samples = [Float64[1, 2] for _ in 1:1_000]
    local_samples = [Float64[2, 4] for _ in 1:1_000]
    summary = summarize_group_alignment(:synapse, exact_samples, local_samples)
    @test summary.sample_count == 1_000
    @test summary.defined_cosines == 1_000
    @test summary.cosine_interval !== nothing
    @test summary.cosine_interval.lower_bound > 0.0
    @test summary.mean_optimal_positive_scale ≈ 0.5
    @test isapprox(summary.mean_residual_ratio, 0.0; atol=eps(Float64))

    summaries = summarize_group_alignments(
        Dict(:synapse => exact_samples, :cell => exact_samples),
        Dict(:synapse => local_samples, :cell => exact_samples),
    )
    @test getfield.(summaries, :group) == [:cell, :synapse]
    @test_throws ArgumentError summarize_group_alignments(
        Dict(:synapse => exact_samples),
        Dict(:cell => local_samples),
    )
    @test_throws DimensionMismatch gradient_alignment([1.0], [1.0, 2.0])
    @test_throws ArgumentError gradient_alignment([NaN], [1.0])
    @test_throws DimensionMismatch summarize_group_alignment(
        :synapse,
        exact_samples[1:2],
        local_samples[1:1],
    )
end

@testset "fail-closed gate result" begin
    passing = fail_closed_gate(
        :g1,
        [
            gate_check(:collision_free, true; evidence="zero disagreements"),
            gate_check(:rank, () -> true; evidence="derived tolerance"),
        ];
        required=[:collision_free, :rank],
    )
    @test passing.passed
    @test require_gate(passing) === passing

    missing_result = fail_closed_gate(:g2, [gate_check(:exact, true)]; required=[:exact, :tail])
    @test !missing_result.passed
    @test any(check -> check.name == :required_tail, missing_result.checks)
    @test_throws ErrorException require_gate(missing_result)

    empty_result = fail_closed_gate(:empty, GateCheck[])
    @test !empty_result.passed
    @test_throws ArgumentError GateResult(:forged, true, GateCheck[])
    duplicate_result = fail_closed_gate(
        :duplicate,
        [gate_check(:same, true), gate_check(:same, true)],
    )
    @test !duplicate_result.passed
    @test !gate_check(:missing, missing).passed
    @test !gate_check(:absent, nothing).passed
    @test !gate_check(:wrong_type, () -> 1).passed
    thrown = gate_check(:throws, () -> error("expected"))
    @test !thrown.passed
    @test occursin("expected", thrown.evidence)
end

end # module CanonicalValidationHarness
