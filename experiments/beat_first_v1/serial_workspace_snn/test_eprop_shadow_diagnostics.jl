using JSON3
using Test

include(joinpath(@__DIR__, "benchmark_eprop_shadow.jl"))

@testset "shadow gradient comparison diagnostics" begin
    local_gradient = Float32[1, -2, 0, 4]
    vjp_gradient = Float32[2, -1, 0, -4]
    report = gradient_comparison_statistics(
        local_gradient,
        vjp_gradient,
    )
    @test report.count == 4
    @test report.local_norm ≈ sqrt(21)
    @test report.vjp_norm ≈ sqrt(21)
    @test report.norm_ratio ≈ 1.0
    @test report.relative_error ≈ sqrt(66 / 21)
    @test report.dot ≈ -12.0
    @test report.cosine ≈ -12 / 21
    @test report.local_nonzero_fraction == 0.75
    @test report.vjp_nonzero_fraction == 0.75
    @test report.joint_nonzero_fraction == 0.75
    @test report.sign_agreement ≈ 2 / 3

    zero_report = gradient_comparison_statistics(
        zeros(Float32, 3),
        zeros(Float32, 3),
    )
    @test zero_report.norm_ratio === nothing
    @test zero_report.relative_error === nothing
    @test zero_report.cosine === nothing
    @test zero_report.sign_agreement === nothing
    @test_throws DimensionMismatch gradient_comparison_statistics(
        zeros(Float32, 2),
        zeros(Float32, 3),
    )
    @test_throws ErrorException gradient_comparison_statistics(
        Float32[NaN],
        Float32[0],
    )
end

@testset "per-batch distributions retain every diagnostic" begin
    first_record = merge(
        (; batch_ordinal=1, optimizer_step=101),
        gradient_comparison_statistics(
            Float32[1, 2],
            Float32[1, 1],
        ),
    )
    second_record = merge(
        (; batch_ordinal=2, optimizer_step=102),
        gradient_comparison_statistics(
            Float32[-1, 0],
            Float32[1, 0],
        ),
    )
    distributions = comparison_batch_distributions(
        Any[first_record, second_record],
    )
    for metric in (
        :local_norm,
        :vjp_norm,
        :norm_ratio,
        :relative_error,
        :dot,
        :cosine,
        :local_nonzero_fraction,
        :vjp_nonzero_fraction,
        :joint_nonzero_fraction,
        :sign_agreement,
    )
        distribution = getproperty(distributions, metric)
        @test distribution.count == 2
        @test distribution.finite_count == 2
    end
    sparse = scalar_distribution(Any[nothing, NaN, 1.0])
    @test sparse.count == 3
    @test sparse.finite_count == 1
    @test sparse.mean == 1.0
end

@testset "all eleven parameter groups are explicitly covered" begin
    @test length(EPROP_PARAMETER_GROUPS) == 11
    @test length(unique(EPROP_PARAMETER_GROUPS)) == 11
    expected = Set((
        :synapse_weight,
        :input_gain,
        :input_bias,
        :gate_logits,
        :delay_logits,
        :leak_logits,
        :threshold_logits,
        :feedback_gain,
        :workspace_key,
        :query_weight,
        :workspace_decay_logit,
    ))
    @test Set(EPROP_PARAMETER_GROUPS) == expected
    enabled = [
        name
        for name in EPROP_PARAMETER_GROUPS
        if eprop_group_enabled(
            name;
            edge_parameter_mode=:weight_gate_delay,
            node_parameter_mode=:full_state,
            routing_parameter_mode=:three_factor,
        )
    ]
    @test Set(enabled) == expected
    weight_only = [
        name
        for name in EPROP_PARAMETER_GROUPS
        if eprop_group_enabled(
            name;
            edge_parameter_mode=:weight_only,
            node_parameter_mode=:none,
            routing_parameter_mode=:none,
        )
    ]
    @test weight_only == [:synapse_weight]
end

@testset "fresh bias-corrected AdamW next-step diagnostic" begin
    parameter = Float32[2, -4]
    gradient = Float64[0.5, -1.0]
    first_moment = Float32[0.1, -0.2]
    second_moment = Float32[0.04, 0.09]
    optimizer = (;
        learning_rate=0.01f0,
        beta1=0.9f0,
        beta2=0.99f0,
        beta1_power=0.81f0,
        beta2_power=0.9801f0,
        epsilon=1.0f-8,
        weight_decay=0.1f0,
    )
    report = adamw_next_step_statistics(
        parameter,
        gradient,
        first_moment,
        second_moment,
        optimizer,
    )
    expected_updates = Float64[]
    for index in eachindex(parameter)
        moment =
            Float64(optimizer.beta1) * first_moment[index] +
            (1 - Float64(optimizer.beta1)) * gradient[index]
        variance =
            Float64(optimizer.beta2) * second_moment[index] +
            (1 - Float64(optimizer.beta2)) * gradient[index]^2
        direction =
            (moment / (1 - Float64(optimizer.beta1_power))) /
            (
                sqrt(
                    variance /
                    (1 - Float64(optimizer.beta2_power)),
                ) + optimizer.epsilon
            )
        push!(
            expected_updates,
            optimizer.learning_rate * (
                direction +
                optimizer.weight_decay * parameter[index]
            ),
        )
    end
    @test report.count == 2
    @test report.total_update_rms ≈
        sqrt(sum(abs2, expected_updates) / 2)
    @test report.coordinate_update_to_weight_median ≈ median(
        abs.(expected_updates) ./
        (abs.(Float64.(parameter)) .+ optimizer.epsilon),
    )
    @test report.beta1_power_for_next_step ==
        Float64(optimizer.beta1_power)
    @test report.beta2_power_for_next_step ==
        Float64(optimizer.beta2_power)
    @test report.gate_projection_applied == false
    @test report.convention ==
        "fresh_next_step_from_checkpoint_moments_and_mean_batch_gradient"
    underflowed_optimizer = merge(
        optimizer,
        (; beta1_power=0.0f0, beta2_power=0.0f0),
    )
    underflowed_report = adamw_next_step_statistics(
        parameter,
        gradient,
        first_moment,
        second_moment,
        underflowed_optimizer,
    )
    @test underflowed_report.beta1_power_for_next_step == 0.0
    @test underflowed_report.beta2_power_for_next_step == 0.0
    @test isfinite(underflowed_report.total_update_rms)
    @test_throws DimensionMismatch adamw_next_step_statistics(
        parameter,
        zeros(Float32, 3),
        first_moment,
        second_moment,
        optimizer,
    )
end

@testset "production stochastic routing schedule and target labels" begin
    @test benchmark_routing_step(100_000, 0) == 100_000
    @test [
        benchmark_routing_step(100_000, ordinal)
        for ordinal in 1:4
    ] == [100_000, 100_001, 100_002, 100_003]
    @test [
        benchmark_routing_step(100_000, ordinal) + 1
        for ordinal in 1:4
    ] == [100_001, 100_002, 100_003, 100_004]
    @test_throws ErrorException benchmark_routing_step(-1, 0)
    @test_throws ErrorException benchmark_routing_step(0, -1)
    three_factor = eprop_routing_target_label(:three_factor)
    @test occursin("plackett_luce", three_factor)
    @test occursin("different_gradient_targets", three_factor)
    @test occursin(
        "different_gradient_targets",
        eprop_routing_target_label(:local_soft),
    )
    @test eprop_routing_target_label(:none) ==
        "routing_parameter_learning_disabled"
end

@testset "atomic no-clobber JSON output" begin
    mktempdir() do directory
        output = joinpath(directory, "shadow.json")
        checks = Ref(0)
        atomic_eprop_json(
            output,
            (; finite=1.0, missing=nothing);
            precommit_check=() -> (checks[] += 1),
        )
        @test checks[] == 1
        @test isfile(output)
        @test JSON3.read(read(output, String)).finite == 1.0
        @test_throws ErrorException atomic_eprop_json(
            output,
            (; finite=2.0),
        )

        protected = joinpath(directory, "checkpoint.json")
        write(protected, "{}")
        @test_throws ErrorException atomic_eprop_json(
            protected,
            (; forbidden=true);
            protected_paths=(protected,),
        )
        forbidden_root = joinpath(directory, "verified_run")
        mkpath(forbidden_root)
        @test eprop_path_within(
            joinpath(forbidden_root, "analysis.json"),
            forbidden_root,
        )
        @test !eprop_path_within(
            joinpath(directory, "verified_run_sibling", "analysis.json"),
            forbidden_root,
        )
        @test_throws ErrorException atomic_eprop_json(
            joinpath(forbidden_root, "analysis.json"),
            (; forbidden=true);
            forbidden_roots=(forbidden_root,),
        )
        failed = joinpath(directory, "failed.json")
        @test_throws ErrorException atomic_eprop_json(
            failed,
            (; finite=1.0);
            precommit_check=() -> error("bound input changed"),
        )
        @test !ispath(failed)
        @test_throws ErrorException atomic_eprop_json(
            joinpath(directory, "nonfinite.json"),
            (; value=Inf),
        )
        @test_throws ErrorException atomic_eprop_json(
            joinpath(directory, "wrong.txt"),
            (; finite=1.0),
        )
    end
end
