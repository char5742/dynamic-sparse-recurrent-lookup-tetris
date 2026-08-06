using Test

include(joinpath(@__DIR__, "ReducedHayCPU.jl"))
using .ReducedHayCPU

const OptimizerTest = ReducedHayCPU.CanonicalOptimizer

@testset "AdamW late-rate schedule is causal and proportional" begin
    _, initial, _ = build_model(0x4f505449)
    full_parameters = deepcopy(initial)
    half_parameters = deepcopy(initial)
    gradient = OptimizerTest.ParameterGradient(initial)
    gradient.output_bias .= 1.0f0

    full_state = OptimizerTest.AdamWState(full_parameters)
    half_state = OptimizerTest.AdamWState(half_parameters)
    full_config = LocalLearningConfig(
        learning_rate_decay_start=0,
        learning_rate_decay_multiplier=1.0,
    )
    half_config = LocalLearningConfig(
        learning_rate_decay_start=0,
        learning_rate_decay_multiplier=0.5,
    )

    full_before = copy(full_parameters.output_bias)
    half_before = copy(half_parameters.output_bias)
    OptimizerTest.apply_adamw!(
        full_state,
        full_parameters,
        gradient,
        full_config;
        phase=OptimizerTest.OUTPUT_OPTIMIZATION,
    )
    OptimizerTest.apply_adamw!(
        half_state,
        half_parameters,
        gradient,
        half_config;
        phase=OptimizerTest.OUTPUT_OPTIMIZATION,
    )

    full_delta = full_before .- full_parameters.output_bias
    half_delta = half_before .- half_parameters.output_bias
    @test full_state.total_step == half_state.total_step == 1
    @test full_state.output_step == half_state.output_step == 1
    @test full_state.recurrent_step == half_state.recurrent_step == 0
    @test all(isfinite, full_delta)
    @test all(isapprox.(half_delta, 0.5f0 .* full_delta; rtol=2.0f-5, atol=1.0f-8))

    boundary_parameters = deepcopy(initial)
    boundary_state = OptimizerTest.AdamWState(boundary_parameters)
    boundary_config = LocalLearningConfig(
        learning_rate_decay_start=1,
        learning_rate_decay_multiplier=0.5,
    )
    OptimizerTest.apply_adamw!(
        boundary_state,
        boundary_parameters,
        gradient,
        boundary_config;
        phase=OptimizerTest.OUTPUT_OPTIMIZATION,
    )
    @test boundary_state.total_step == 1
    @test boundary_state.output_step == 1
    @test boundary_config.learning_rate_decay_start == 1
    OptimizerTest.apply_adamw!(
        boundary_state,
        boundary_parameters,
        gradient,
        boundary_config;
        phase=OptimizerTest.OUTPUT_OPTIMIZATION,
    )
    @test boundary_state.total_step == 2
    @test boundary_state.output_step == 2
end


@testset "recurrent alpha ramps continuously after delayed start" begin
    _, initial, _ = build_model(0x52414d50)
    full_parameters = deepcopy(initial)
    ramped_parameters = deepcopy(initial)
    gradient = OptimizerTest.ParameterGradient(initial)
    gradient.cell_raw .= 1.0f0
    full_state = OptimizerTest.AdamWState(full_parameters)
    ramped_state = OptimizerTest.AdamWState(ramped_parameters)
    full_config = LocalLearningConfig(
        recurrent_start_update=1,
        recurrent_ramp_updates=1,
        recurrent_multiplier=1.0,
        weight_decay=0.0,
    )
    ramped_config = LocalLearningConfig(
        recurrent_start_update=1,
        recurrent_ramp_updates=10,
        recurrent_multiplier=1.0,
        weight_decay=0.0,
    )
    full_before = copy(full_parameters.cell_raw)
    ramped_before = copy(ramped_parameters.cell_raw)
    OptimizerTest.apply_adamw!(
        full_state,
        full_parameters,
        gradient,
        full_config;
        phase=OptimizerTest.RECURRENT_OPTIMIZATION,
    )
    OptimizerTest.apply_adamw!(
        ramped_state,
        ramped_parameters,
        gradient,
        ramped_config;
        phase=OptimizerTest.RECURRENT_OPTIMIZATION,
    )
    full_delta = full_before .- full_parameters.cell_raw
    ramped_delta = ramped_before .- ramped_parameters.cell_raw
    @test all(isapprox.(
        ramped_delta,
        0.1f0 .* full_delta;
        rtol=2.0f-4,
        atol=5.0f-8,
    ))
end

@testset "output and recurrent AdamW phases are mutually exclusive" begin
    _, initial, _ = build_model(0x50484153)
    parameters = deepcopy(initial)
    state = OptimizerTest.AdamWState(parameters)
    gradient = OptimizerTest.ParameterGradient(initial)
    gradient.output_edge_raw .= 1.0f0
    gradient.cell_raw .= 1.0f0
    config = LocalLearningConfig(
        recurrent_ramp_updates=1,
        recurrent_multiplier=1.0,
        weight_decay=0.0,
    )

    cell_before = copy(parameters.cell_raw)
    output_before = copy(parameters.output_edge_raw)
    OptimizerTest.apply_adamw!(
        state,
        parameters,
        gradient,
        config;
        phase=OptimizerTest.OUTPUT_OPTIMIZATION,
    )
    @test parameters.cell_raw == cell_before
    @test parameters.output_edge_raw != output_before
    @test state.output_step == 1
    @test state.recurrent_step == 0

    output_after = copy(parameters.output_edge_raw)
    OptimizerTest.apply_adamw!(
        state,
        parameters,
        gradient,
        config;
        phase=OptimizerTest.RECURRENT_OPTIMIZATION,
    )
    @test parameters.output_edge_raw == output_after
    @test parameters.cell_raw != cell_before
    @test state.output_step == 1
    @test state.recurrent_step == 1
    @test state.total_step == 2
end

@testset "output decay schedule does not attenuate recurrent learning" begin
    _, initial, _ = build_model(0x44454341)
    full_parameters = deepcopy(initial)
    decayed_parameters = deepcopy(initial)
    full_state = OptimizerTest.AdamWState(full_parameters)
    decayed_state = OptimizerTest.AdamWState(decayed_parameters)
    gradient = OptimizerTest.ParameterGradient(initial)
    gradient.cell_raw .= 1.0f0
    full_config = LocalLearningConfig(
        learning_rate_decay_start=0,
        learning_rate_decay_multiplier=1.0,
        recurrent_ramp_updates=1,
        recurrent_multiplier=1.0,
        weight_decay=0.0,
    )
    decayed_config = LocalLearningConfig(
        learning_rate_decay_start=0,
        learning_rate_decay_multiplier=0.1,
        recurrent_ramp_updates=1,
        recurrent_multiplier=1.0,
        weight_decay=0.0,
    )
    OptimizerTest.apply_adamw!(
        full_state, full_parameters, gradient, full_config;
        phase=OptimizerTest.RECURRENT_OPTIMIZATION,
    )
    OptimizerTest.apply_adamw!(
        decayed_state, decayed_parameters, gradient, decayed_config;
        phase=OptimizerTest.RECURRENT_OPTIMIZATION,
    )
    @test full_parameters.cell_raw == decayed_parameters.cell_raw
end

@testset "auxiliary AdamW cannot mutate numeric-Q edge slots" begin
    _, initial, _ = build_model(0x51534c4f)
    parameters = deepcopy(initial)
    state = OptimizerTest.AdamWState(parameters)
    gradient = OptimizerTest.ParameterGradient(initial)
    gradient.output_bias .= 1.0f0
    q_relations = 1:ReducedHayCPU.OutputCellBank.Q_FANOUT_PER_SOURCE
    first_auxiliary_relation =
        ReducedHayCPU.OutputCellBank.Q_FANOUT_PER_SOURCE + 1
    gradient.output_edge_raw[first_auxiliary_relation, 1] = 1.0f0
    auxiliary_relations = first_auxiliary_relation:size(
        parameters.output_edge_raw,
        1,
    )
    q_before = copy(view(parameters.output_edge_raw, q_relations, :))
    auxiliary_before = copy(view(
        parameters.output_edge_raw,
        auxiliary_relations,
        :,
    ))
    OptimizerTest.apply_adamw!(
        state,
        parameters,
        gradient,
        LocalLearningConfig(weight_decay=0.0);
        phase=OptimizerTest.OUTPUT_OPTIMIZATION,
    )
    @test view(parameters.output_edge_raw, q_relations, :) == q_before
    @test view(parameters.output_edge_raw, auxiliary_relations, :) !=
          auxiliary_before
    @test all(iszero, view(state.first.output_edge_raw, q_relations, :))
    @test all(iszero, view(state.second.output_edge_raw, q_relations, :))
end

@testset "candidate-mean Q Adam is causal and isolated from auxiliary output" begin
    _, initial, _ = build_model(0x5a45524f)
    parameters = deepcopy(initial)
    state = OptimizerTest.AdamWState(parameters)
    gradient = OptimizerTest.ParameterGradient(initial)
    edge_before = copy(parameters.output_edge_raw)
    bias_before = copy(parameters.output_q_basal_bias_raw)
    cell_before = copy(parameters.output_cell_raw)
    gain_before = copy(parameters.output_gain)
    output_bias_before = copy(parameters.output_bias)
    recurrent_before = copy(parameters.cell_raw)
    config = LocalLearningConfig(weight_decay=0.0, clip_norm=100.0)
    OptimizerTest.apply_adamw!(
        state,
        parameters,
        gradient,
        config;
        phase=OptimizerTest.OUTPUT_OPTIMIZATION,
    )
    @test parameters.output_edge_raw == edge_before
    @test parameters.output_q_basal_bias_raw == bias_before
    @test all(iszero, state.first.output_q_edge_raw)
    @test all(iszero, state.second.output_q_edge_raw)
    @test all(iszero, state.first.output_q_basal_bias_raw)
    @test all(iszero, state.second.output_q_basal_bias_raw)
    @test parameters.output_cell_raw == cell_before
    @test parameters.output_gain == gain_before
    @test parameters.output_bias == output_bias_before
    @test parameters.cell_raw == recurrent_before

    large_relation, large_source = 1, 1
    small_relation, small_source = 24, 10
    @test ReducedHayCPU.OutputCellBank.q_relation_output(
        large_relation,
        large_source,
    ) == 1
    @test ReducedHayCPU.OutputCellBank.q_relation_output(
        small_relation,
        small_source,
    ) == 1
    gradient.output_q_edge_raw[large_relation, large_source] = 3.0f0
    gradient.output_q_edge_raw[small_relation, small_source] = 0.03f0
    gradient.output_q_basal_bias_raw[1] = 4.0f0
    edge_before = copy(parameters.output_edge_raw)
    bias_before = copy(parameters.output_q_basal_bias_raw)
    OptimizerTest.apply_adamw!(
        state,
        parameters,
        gradient,
        config;
        phase=OptimizerTest.OUTPUT_OPTIMIZATION,
    )
    @test parameters.output_edge_raw[large_relation, large_source] <
        edge_before[large_relation, large_source]
    @test parameters.output_q_basal_bias_raw[1] < bias_before[1]
    @test isapprox(
        state.first.output_q_edge_raw[large_relation, large_source] /
        state.first.output_q_edge_raw[small_relation, small_source],
        100.0f0;
        rtol=2.0f-5,
    )
    @test isapprox(
        state.second.output_q_edge_raw[large_relation, large_source] /
        state.second.output_q_edge_raw[small_relation, small_source],
        10_000.0f0;
        rtol=2.0f-4,
    )
    @test state.first.output_q_basal_bias_raw[1] > 0.0f0
    @test state.second.output_q_basal_bias_raw[1] > 0.0f0
    first_auxiliary_relation =
        ReducedHayCPU.OutputCellBank.Q_FANOUT_PER_SOURCE + 1
    @test parameters.output_edge_raw[first_auxiliary_relation, 1] ==
        edge_before[first_auxiliary_relation, 1]
    @test parameters.output_cell_raw == cell_before
    @test parameters.output_gain == gain_before
    @test parameters.output_bias == output_bias_before
    @test parameters.cell_raw == recurrent_before
end

@testset "Q cell Adam averages bit gradients and preserves exact sharing" begin
    _, initial, _ = build_model(0x53484152)
    parameters = deepcopy(initial)
    state = OptimizerTest.AdamWState(parameters)
    gradient = OptimizerTest.ParameterGradient(initial)
    q_cells = ReducedHayCPU.Architecture.Q_OUTPUT_CELL_COUNT
    first_auxiliary = q_cells + 1
    cell_parameter = 1

    # One column carries 32 times the unit gradient.  The shared primitive must
    # see its mean, exactly one, rather than either the sum or column one alone.
    gradient.output_cell_raw[cell_parameter, 1] = Float32(q_cells)
    q_before = copy(@view parameters.output_cell_raw[:, 1:q_cells])
    auxiliary_before = copy(@view parameters.output_cell_raw[:, first_auxiliary:end])
    edge_before = copy(parameters.output_edge_raw)
    recurrent_before = copy(parameters.cell_raw)
    OptimizerTest.apply_adamw!(
        state,
        parameters,
        gradient,
        LocalLearningConfig(weight_decay=0.0, clip_norm=100.0);
        phase=OptimizerTest.OUTPUT_OPTIMIZATION,
    )

    @test parameters.output_cell_raw[cell_parameter, 1] <
        q_before[cell_parameter, 1]
    @test isapprox(
        state.first.output_cell_raw[cell_parameter, 1],
        0.1f0;
        rtol=2.0f-6,
        atol=1.0f-8,
    )
    @test isapprox(
        state.second.output_cell_raw[cell_parameter, 1],
        0.001f0;
        rtol=2.0f-5,
        atol=1.0f-9,
    )
    for output in 2:q_cells
        @test @view(parameters.output_cell_raw[:, output]) ==
            @view(parameters.output_cell_raw[:, 1])
        @test @view(state.first.output_cell_raw[:, output]) ==
            @view(state.first.output_cell_raw[:, 1])
        @test @view(state.second.output_cell_raw[:, output]) ==
            @view(state.second.output_cell_raw[:, 1])
    end
    @test @view(parameters.output_cell_raw[:, first_auxiliary:end]) ==
        auxiliary_before
    @test parameters.output_edge_raw == edge_before
    @test parameters.cell_raw == recurrent_before
end

@testset "opposite bit gradients cancel before shared Q-cell Adam" begin
    _, initial, _ = build_model(0x43414e43)
    parameters = deepcopy(initial)
    state = OptimizerTest.AdamWState(parameters)
    gradient = OptimizerTest.ParameterGradient(initial)
    gradient.output_cell_raw[1, 1] = 1.0f0
    gradient.output_cell_raw[1, 2] = -1.0f0
    q_before = copy(@view parameters.output_cell_raw[:, 1:32])
    OptimizerTest.apply_adamw!(
        state,
        parameters,
        gradient,
        LocalLearningConfig(weight_decay=0.0, clip_norm=100.0);
        phase=OptimizerTest.OUTPUT_OPTIMIZATION,
    )
    @test @view(parameters.output_cell_raw[:, 1:32]) == q_before
    @test all(iszero, @view(state.first.output_cell_raw[:, 1:32]))
    @test all(iszero, @view(state.second.output_cell_raw[:, 1:32]))
end

@testset "Q Adam uses one validated candidate mean" begin
    _, initial, _ = build_model(0x4d45414e)
    unit_parameters = deepcopy(initial)
    batch_parameters = deepcopy(initial)
    unit_state = OptimizerTest.AdamWState(unit_parameters)
    batch_state = OptimizerTest.AdamWState(batch_parameters)
    unit_gradient = OptimizerTest.ParameterGradient(initial)
    batch_gradient = OptimizerTest.ParameterGradient(initial)
    unit_gradient.output_q_edge_raw[1, 1] = 0.75f0
    unit_gradient.output_q_basal_bias_raw[1] = -0.5f0
    unit_gradient.output_cell_raw[1, 1] = 8.0f0
    for field in (
        :output_q_edge_raw,
        :output_q_basal_bias_raw,
        :output_cell_raw,
    )
        getfield(batch_gradient, field) .= 52.0f0 .* getfield(
            unit_gradient,
            field,
        )
    end
    config = LocalLearningConfig(
        weight_decay=0.0,
        q_weight_decay=0.0,
        clip_norm=100.0,
    )
    OptimizerTest.apply_adamw!(
        unit_state,
        unit_parameters,
        unit_gradient,
        config;
        phase=OptimizerTest.OUTPUT_OPTIMIZATION,
        q_candidate_count=1,
    )
    OptimizerTest.apply_adamw!(
        batch_state,
        batch_parameters,
        batch_gradient,
        config;
        phase=OptimizerTest.OUTPUT_OPTIMIZATION,
        q_candidate_count=52,
    )
    @test isapprox(
        unit_parameters.output_edge_raw,
        batch_parameters.output_edge_raw;
        rtol=2.0f-6,
        atol=2.0f-7,
    )
    @test isapprox(
        unit_parameters.output_q_basal_bias_raw,
        batch_parameters.output_q_basal_bias_raw;
        rtol=2.0f-6,
        atol=2.0f-7,
    )
    @test isapprox(
        unit_parameters.output_cell_raw,
        batch_parameters.output_cell_raw;
        rtol=2.0f-6,
        atol=2.0f-7,
    )
    @test isapprox(
        unit_state.first.output_cell_raw,
        batch_state.first.output_cell_raw;
        rtol=2.0f-6,
        atol=2.0f-7,
    )
    @test isapprox(
        unit_state.second.output_cell_raw,
        batch_state.second.output_cell_raw;
        rtol=2.0f-6,
        atol=2.0f-7,
    )
end
