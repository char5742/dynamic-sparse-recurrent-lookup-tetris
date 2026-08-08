using Test

include(joinpath(@__DIR__, "CanonicalOptimizer.jl"))
using .CanonicalOptimizer

@inline function approximately_equal(left, right; atol=1.0f-6)
    return isapprox(left, right; atol, rtol=0.0f0)
end

function optimizer_fixture(; frozen_multiplier=0.0f0)
    signed_weight = Float32[0.5, -0.25]
    signed_readout = Float32[0.75]
    conductance = Float32[
        inverse_softplus(0.2f0),
        inverse_softplus(0.8f0),
    ]
    cell_raw = Float32[0.5, -0.5]
    offset = Float32[0.25]
    frozen = Float32[0.4]

    signed_weight_gradient = zeros(Float32, 2)
    signed_readout_gradient = zeros(Float32, 1)
    conductance_gradient = zeros(Float32, 2)
    cell_raw_gradient = zeros(Float32, 2)
    offset_gradient = zeros(Float32, 1)
    frozen_gradient = fill(100.0f0, 1)

    registry = ParameterRegistry(
        ParameterGroup(
            :signed_weight,
            signed_weight,
            signed_weight_gradient,
            SIGNED_WEIGHT;
            lower_bound=-1.0f0,
            upper_bound=1.0f0,
        ),
        ParameterGroup(
            :signed_readout,
            signed_readout,
            signed_readout_gradient,
            SIGNED_READOUT;
            lower_bound=-1.0f0,
            upper_bound=1.0f0,
        ),
        ParameterGroup(
            :conductance,
            conductance,
            conductance_gradient,
            INVERSE_SOFTPLUS_CONDUCTANCE;
            lower_bound=0.05f0,
            upper_bound=1.0f0,
        ),
        ParameterGroup(
            :cell_raw,
            cell_raw,
            cell_raw_gradient,
            CELL_RAW;
            lower_bound=-2.0f0,
            upper_bound=2.0f0,
        ),
        ParameterGroup(
            :offset,
            offset,
            offset_gradient,
            NO_DECAY_RAW;
            lower_bound=-1.0f0,
            upper_bound=1.0f0,
        ),
        ParameterGroup(
            :frozen,
            frozen,
            frozen_gradient,
            SIGNED_WEIGHT;
            multiplier=frozen_multiplier,
            lower_bound=-1.0f0,
            upper_bound=1.0f0,
        ),
    )
    return registry
end

function dual_optimizer_fixture()
    core_parameter = fill(0.25f0, 2, 2)
    semantic_parameter = fill(inverse_softplus(0.4f0), 2, 1, 1, 2)
    event_parameter = fill(inverse_softplus(0.3f0), 3)
    output_cell_parameter = fill(-0.2f0, 2, 1)
    output_projection_parameter = fill(
        inverse_softplus(0.5f0), 2, 2,
    )
    return ParameterRegistry(
        ParameterGroup(
            :core_cell_raw,
            core_parameter,
            zeros(Float32, size(core_parameter)),
            CELL_RAW;
            lower_bound=-4.0f0,
            upper_bound=4.0f0,
        ),
        ParameterGroup(
            :semantic_projection_raw,
            semantic_parameter,
            zeros(Float32, size(semantic_parameter)),
            INVERSE_SOFTPLUS_CONDUCTANCE;
            lower_bound=0.05f0,
            upper_bound=2.0f0,
        ),
        ParameterGroup(
            :event_raw,
            event_parameter,
            zeros(Float32, size(event_parameter)),
            INVERSE_SOFTPLUS_CONDUCTANCE;
            lower_bound=0.05f0,
            upper_bound=2.0f0,
        ),
        ParameterGroup(
            :output_cell_raw,
            output_cell_parameter,
            zeros(Float32, size(output_cell_parameter)),
            CELL_RAW;
            lower_bound=-4.0f0,
            upper_bound=4.0f0,
        ),
        ParameterGroup(
            :output_projection_raw,
            output_projection_parameter,
            zeros(Float32, size(output_projection_parameter)),
            INVERSE_SOFTPLUS_CONDUCTANCE;
            lower_bound=0.05f0,
            upper_bound=2.0f0,
        ),
    )
end

function dual_gradient_lanes(registry::ParameterRegistry)
    analog_buffers = map(group -> group.gradient, registry.groups)
    hard_buffers = (
        zeros(Float32, size(registry.groups[1].parameter)),
        zeros(Float32, size(registry.groups[2].parameter)),
        zeros(Float32, size(registry.groups[3].parameter)),
        nothing,
        nothing,
    )
    return (
        GradientBufferSet(registry, analog_buffers),
        GradientBufferSet(registry, hard_buffers),
    )
end

function optimizer_snapshot(registry, state, analog, hard)
    return (
        parameter=map(group -> copy(group.parameter), registry.groups),
        first=map(moment -> copy(moment.first), state.moments),
        second=map(moment -> copy(moment.second), state.moments),
        group_steps=copy(state.group_steps),
        total_step=state.total_step,
        analog=map(
            buffer -> buffer === nothing ? nothing : copy(buffer),
            analog.buffers,
        ),
        hard=map(
            buffer -> buffer === nothing ? nothing : copy(buffer),
            hard.buffers,
        ),
    )
end

function test_optimizer_snapshot(snapshot, registry, state, analog, hard)
    @test map(group -> group.parameter, registry.groups) ==
        snapshot.parameter
    @test map(moment -> moment.first, state.moments) == snapshot.first
    @test map(moment -> moment.second, state.moments) == snapshot.second
    @test state.group_steps == snapshot.group_steps
    @test state.total_step == snapshot.total_step
    @test isequal(analog.buffers, snapshot.analog)
    @test isequal(hard.buffers, snapshot.hard)
    return nothing
end

@testset "canonical optimizer registry and decay policy" begin
    registry = optimizer_fixture()
    @test parameter_group_names(registry) == (
        :signed_weight,
        :signed_readout,
        :conductance,
        :cell_raw,
        :offset,
        :frozen,
    )
    @test registry_group_count(registry) == 6
    @test uses_weight_decay(SIGNED_WEIGHT)
    @test uses_weight_decay(SIGNED_READOUT)
    @test !uses_weight_decay(INVERSE_SOFTPLUS_CONDUCTANCE)
    @test !uses_weight_decay(CELL_RAW)
    @test !uses_weight_decay(NO_DECAY_RAW)
    @test_throws ArgumentError ParameterRegistry(
        registry.groups[1],
        registry.groups[1],
    )
    aliased_parameter = zeros(Float32, 2)
    first_gradient = zeros(Float32, 2)
    second_gradient = zeros(Float32, 2)
    @test_throws ArgumentError ParameterRegistry(
        ParameterGroup(
            :first_alias,
            aliased_parameter,
            first_gradient,
            SIGNED_WEIGHT,
        ),
        ParameterGroup(
            :second_alias,
            aliased_parameter,
            second_gradient,
            SIGNED_WEIGHT,
        ),
    )
    @test_throws ArgumentError ParameterRegistry(ParameterGroup(
        :self_alias,
        aliased_parameter,
        aliased_parameter,
        SIGNED_WEIGHT,
    ))
end

@testset "normal AdamW, strict freeze and physical projection" begin
    registry = optimizer_fixture()
    state = AdamWState(registry)
    config = AdamWConfig(
        learning_rate=0.1f0,
        beta1=0.0f0,
        beta2=0.0f0,
        epsilon=1.0f-8,
        clip_norm=100.0f0,
        weight_decay=0.2f0,
    )
    weight_before = copy(registry.groups[1].parameter)
    readout_before = copy(registry.groups[2].parameter)
    conductance_before = copy(registry.groups[3].parameter)
    cell_before = copy(registry.groups[4].parameter)
    offset_before = copy(registry.groups[5].parameter)
    frozen_before = copy(registry.groups[6].parameter)

    stats = apply_optimizer_boundary!(state, registry, config)
    @test stats.gradient_norm == 0.0
    @test stats.clip_scale == 1.0f0
    @test stats.active_groups == 5
    @test registry.groups[1].parameter ≈ 0.98f0 .* weight_before
    @test registry.groups[2].parameter ≈ 0.98f0 .* readout_before
    @test registry.groups[3].parameter == conductance_before
    @test registry.groups[4].parameter == cell_before
    @test registry.groups[5].parameter == offset_before
    @test registry.groups[6].parameter == frozen_before
    @test state.group_steps == UInt64[1, 1, 1, 1, 1, 0]
    @test all(iszero, state.moments[6].first)
    @test all(iszero, state.moments[6].second)

    registry.groups[1].parameter[1] = 2.0f0
    registry.groups[3].parameter[1] = inverse_softplus(2.0f0)
    registry.groups[4].parameter[1] = 3.0f0
    registry.groups[5].parameter[1] = 2.0f0
    bounded = apply_optimizer_boundary!(state, registry, config)
    @test bounded.projected_values > 0
    @test all(value -> -1.0f0 <= value <= 1.0f0,
              registry.groups[1].parameter)
    @test all(value -> 0.05f0 <= physical_conductance(value) <= 1.0f0,
              registry.groups[3].parameter)
    @test all(value -> -2.0f0 <= value <= 2.0f0,
              registry.groups[4].parameter)
    @test all(value -> -1.0f0 <= value <= 1.0f0,
              registry.groups[5].parameter)
    @test registry.groups[6].parameter == frozen_before
    @test state.group_steps[6] == 0
end

@testset "Adam bias correction, gradient scale and group multiplier" begin
    parameter = Float32[1.0]
    gradient = Float32[2.0]
    registry = ParameterRegistry(ParameterGroup(
        :scaled,
        parameter,
        gradient,
        NO_DECAY_RAW;
        multiplier=0.25f0,
        lower_bound=-10.0f0,
        upper_bound=10.0f0,
    ))
    config = AdamWConfig(
        learning_rate=0.2f0,
        beta1=0.5f0,
        beta2=0.5f0,
        epsilon=1.0f-8,
        clip_norm=100.0f0,
        weight_decay=0.9f0,
    )
    state = AdamWState(registry)

    first = apply_optimizer_boundary!(
        state,
        registry,
        config;
        gradient_scale=0.5f0,
    )
    @test approximately_equal(parameter[1], 0.95f0)
    @test approximately_equal(Float32(first.gradient_norm), 1.0f0)
    @test approximately_equal(state.moments[1].first[1], 0.5f0)
    @test approximately_equal(state.moments[1].second[1], 0.5f0)

    second = apply_optimizer_boundary!(
        state,
        registry,
        config;
        gradient_scale=0.5f0,
    )
    @test approximately_equal(parameter[1], 0.90f0)
    @test state.group_steps[1] == 2
    @test second.total_step == 2
end

@testset "global clipping and fail-closed preflight" begin
    registry = optimizer_fixture()
    state = AdamWState(registry)
    registry.groups[1].gradient .= Float32[3.0, 4.0]
    config = AdamWConfig(
        learning_rate=0.01f0,
        beta1=0.0f0,
        beta2=0.0f0,
        clip_norm=2.5f0,
        weight_decay=0.0f0,
    )
    stats = apply_optimizer_boundary!(state, registry, config)
    @test approximately_equal(Float32(stats.gradient_norm), 5.0f0)
    @test approximately_equal(stats.clip_scale, 0.5f0)

    parameter_before = copy(registry.groups[1].parameter)
    steps_before = copy(state.group_steps)
    total_before = state.total_step
    registry.groups[1].gradient[1] = NaN32
    @test_throws DomainError apply_optimizer_boundary!(state, registry, config)
    @test registry.groups[1].parameter == parameter_before
    @test state.group_steps == steps_before
    @test state.total_step == total_before

    registry.groups[1].gradient[1] = 0.0f0
    @test_throws ArgumentError apply_optimizer_boundary!(
        state,
        registry,
        config;
        gradient_scale=1.0e300,
    )
    @test registry.groups[1].parameter == parameter_before
    @test state.group_steps == steps_before
    @test state.total_step == total_before
end

@testset "targeted and full moment reset" begin
    registry = optimizer_fixture()
    state = AdamWState(registry)
    registry.groups[1].gradient .= 1.0f0
    config = AdamWConfig(weight_decay=0.0f0)
    apply_optimizer_boundary!(state, registry, config)
    @test all(!iszero, state.moments[1].first)

    reset_moments!(state, registry, :signed_weight, 1)
    @test state.moments[1].first[1] == 0.0f0
    @test state.moments[1].second[1] == 0.0f0
    @test state.moments[1].first[2] != 0.0f0
    @test state.group_steps[1] == 1

    reset_moments!(state, registry, :signed_weight)
    @test all(iszero, state.moments[1].first)
    @test all(iszero, state.moments[1].second)
    @test state.group_steps[1] == 0
end

@testset "due mask preserves inactive Adam groups exactly" begin
    registry = optimizer_fixture(frozen_multiplier=1.0f0)
    state = AdamWState(registry)
    config = AdamWConfig(
        learning_rate=0.01f0,
        beta1=0.5f0,
        beta2=0.75f0,
        clip_norm=100.0f0,
        weight_decay=0.1f0,
    )
    for group in registry.groups
        fill!(group.gradient, 0.25f0)
    end
    apply_optimizer_boundary!(state, registry, config)

    parameter_before = map(group -> copy(group.parameter), registry.groups)
    first_before = map(moment -> copy(moment.first), state.moments)
    second_before = map(moment -> copy(moment.second), state.moments)
    steps_before = copy(state.group_steps)
    due = (true, false, true, false, true, false)
    stats = apply_optimizer_boundary!(
        state, registry, config; due_mask=due,
    )

    @test stats.active_groups == 3
    @test stats.total_step == 2
    @test state.total_step == 2
    @test state.group_steps == steps_before .+ UInt64[1, 0, 1, 0, 1, 0]
    @test Float32(stats.gradient_norm) ≈ Float32(gradient_norm(
        registry; due_mask=due,
    )) atol=1.0f-6 rtol=0.0f0
    for index in (2, 4, 6)
        @test registry.groups[index].parameter == parameter_before[index]
        @test state.moments[index].first == first_before[index]
        @test state.moments[index].second == second_before[index]
    end
    for index in (1, 3, 5)
        @test registry.groups[index].parameter != parameter_before[index]
        @test state.moments[index].first != first_before[index]
        @test state.moments[index].second != second_before[index]
    end

    @test_throws DimensionMismatch apply_optimizer_boundary!(
        state, registry, config; due_mask=(true, false),
    )
    @test_throws ArgumentError apply_optimizer_boundary!(
        state, registry, config; due_mask=(true, false, true, false, true, 1),
    )
    @test_throws ArgumentError apply_optimizer_boundary!(
        state, registry, config; due_mask=trues(6),
    )
end

@testset "steady-state optimizer boundary allocates zero bytes" begin
    registry = optimizer_fixture(frozen_multiplier=1.0f0)
    state = AdamWState(registry)
    config = AdamWConfig(
        learning_rate=1.0f-4,
        clip_norm=10.0f0,
        weight_decay=1.0f-4,
    )
    for group in registry.groups
        fill!(group.gradient, 1.0f-3)
    end
    apply_optimizer_boundary!(state, registry, config)
    apply_optimizer_boundary!(state, registry, config)
    allocated = @allocated apply_optimizer_boundary!(state, registry, config)
    @test allocated == 0

    due = (true, false, true, true, false, true)
    apply_optimizer_boundary!(state, registry, config; due_mask=due)
    masked_allocated = @allocated apply_optimizer_boundary!(
        state, registry, config; due_mask=due,
    )
    @test masked_allocated == 0
end


@testset "dual lanes clip independently and enter Adam exactly once" begin
    registry = dual_optimizer_fixture()
    state = AdamWState(registry)
    analog, hard = dual_gradient_lanes(registry)
    fill!(analog.buffers[1], 0.0f0)
    fill!(hard.buffers[1], 0.0f0)
    analog.buffers[1][1] = 3.0f0
    analog.buffers[1][2] = 4.0f0
    hard.buffers[1][2] = 10.0f0
    analog_before = map(copy, analog.buffers)
    hard_before = map(
        buffer -> buffer === nothing ? nothing : copy(buffer), hard.buffers,
    )
    config = AdamWConfig(
        learning_rate=0.01f0,
        beta1=0.0f0,
        beta2=0.0f0,
        epsilon=1.0f-8,
        clip_norm=2.5f0,
        weight_decay=0.0f0,
    )
    due = (true, false, false, false, false)
    stats = apply_dual_optimizer_boundary!(
        state,
        registry,
        analog,
        hard,
        config;
        due_mask=due,
    )

    @test approximately_equal(
        Float32(stats.analog_gradient_norm), 5.0f0,
    )
    @test approximately_equal(stats.analog_clip_scale, 0.5f0)
    @test approximately_equal(
        Float32(stats.hard_event_gradient_norm), 10.0f0,
    )
    @test approximately_equal(stats.hard_event_clip_scale, 0.25f0)
    @test approximately_equal(
        Float32(stats.combined_gradient_norm), sqrt(22.5f0),
    )
    @test stats.combined_gradient_norm > Float64(config.clip_norm)
    @test state.moments[1].first[1] == 1.5f0
    @test state.moments[1].first[2] == 4.5f0
    @test state.moments[1].second[1] == 2.25f0
    @test state.moments[1].second[2] == 20.25f0
    @test state.group_steps == UInt64[1, 0, 0, 0, 0]
    @test state.total_step == 1
    @test stats.active_groups == 1
    @test analog.buffers == analog_before
    @test hard.buffers == hard_before
end

@testset "dual lane sum rounds each clipped lane before addition" begin
    registry = dual_optimizer_fixture()
    state = AdamWState(registry)
    analog, hard = dual_gradient_lanes(registry)
    analog_value = 0.66898423f0
    analog_scale = 0.24527991f0
    hard_value = 0.8008963f0
    hard_scale = 0.6619079f0
    analog.buffers[1][1] = analog_value
    hard.buffers[1][1] = hard_value
    separately_rounded =
        (analog_value * analog_scale) + (hard_value * hard_scale)
    fused = muladd(analog_value, analog_scale, hard_value * hard_scale)
    @test separately_rounded != fused

    stats = apply_dual_optimizer_boundary!(
        state,
        registry,
        analog,
        hard,
        AdamWConfig(
            learning_rate=0.01f0,
            beta1=0.0f0,
            beta2=0.0f0,
            clip_norm=100.0f0,
            weight_decay=0.0f0,
        );
        analog_gradient_scale=analog_scale,
        hard_event_gradient_scale=hard_scale,
        due_mask=(true, false, false, false, false),
    )
    @test stats.analog_clip_scale == 1.0f0
    @test stats.hard_event_clip_scale == 1.0f0
    @test state.moments[1].first[1] == separately_rounded
    @test state.moments[1].first[1] != fused
end

@testset "dual lanes support either zero lane and explicit structural zeros" begin
    config = AdamWConfig(
        learning_rate=0.01f0,
        beta1=0.5f0,
        beta2=0.75f0,
        clip_norm=100.0f0,
        weight_decay=0.0f0,
    )

    hard_only_registry = dual_optimizer_fixture()
    hard_only_state = AdamWState(hard_only_registry)
    analog_zero, hard_only = dual_gradient_lanes(hard_only_registry)
    fill!(hard_only.buffers[1], 0.25f0)
    hard_stats = apply_dual_optimizer_boundary!(
        hard_only_state,
        hard_only_registry,
        analog_zero,
        hard_only,
        config,
    )
    @test hard_stats.analog_gradient_norm == 0.0
    @test hard_stats.analog_clip_scale == 1.0f0
    @test hard_stats.hard_event_gradient_norm > 0.0
    @test all(==(0.125f0), hard_only_state.moments[1].first)
    @test all(iszero, hard_only_state.moments[4].first)
    @test hard_only_state.group_steps == fill(UInt64(1), 5)
    @test hard_only_state.total_step == 1

    analog_only_registry = dual_optimizer_fixture()
    analog_only_state = AdamWState(analog_only_registry)
    analog_only, _ = dual_gradient_lanes(analog_only_registry)
    fill!(analog_only.buffers[4], -0.5f0)
    structural_zero = GradientBufferSet(
        analog_only_registry,
        (nothing, nothing, nothing, nothing, nothing),
    )
    analog_stats = apply_dual_optimizer_boundary!(
        analog_only_state,
        analog_only_registry,
        analog_only,
        structural_zero,
        config,
    )
    @test analog_stats.analog_gradient_norm > 0.0
    @test analog_stats.hard_event_gradient_norm == 0.0
    @test analog_stats.hard_event_clip_scale == 1.0f0
    @test all(==(-0.25f0), analog_only_state.moments[4].first)
    @test analog_only_state.group_steps == fill(UInt64(1), 5)

    both_zero_registry = dual_optimizer_fixture()
    both_zero_state = AdamWState(both_zero_registry)
    both_zero_analog, _ = dual_gradient_lanes(both_zero_registry)
    both_zero_hard = GradientBufferSet(
        both_zero_registry,
        (nothing, nothing, nothing, nothing, nothing),
    )
    parameters_before = map(
        group -> copy(group.parameter), both_zero_registry.groups,
    )
    zero_stats = apply_dual_optimizer_boundary!(
        both_zero_state,
        both_zero_registry,
        both_zero_analog,
        both_zero_hard,
        config,
    )
    @test zero_stats.analog_gradient_norm == 0.0
    @test zero_stats.hard_event_gradient_norm == 0.0
    @test zero_stats.combined_gradient_norm == 0.0
    @test zero_stats.active_groups == 5
    @test map(group -> group.parameter, both_zero_registry.groups) ==
        parameters_before
    @test both_zero_state.group_steps == fill(UInt64(1), 5)
    @test both_zero_state.total_step == 1
end

@testset "dual five-group due mask preserves inactive groups" begin
    registry = dual_optimizer_fixture()
    state = AdamWState(registry)
    analog, hard = dual_gradient_lanes(registry)
    for buffer in analog.buffers
        fill!(buffer, 0.2f0)
    end
    for buffer in hard.buffers
        buffer === nothing || fill!(buffer, -0.04f0)
    end
    parameters_before = map(group -> copy(group.parameter), registry.groups)
    first_before = map(moment -> copy(moment.first), state.moments)
    second_before = map(moment -> copy(moment.second), state.moments)
    due = (true, false, true, true, false)
    stats = apply_dual_optimizer_boundary!(
        state,
        registry,
        analog,
        hard,
        AdamWConfig(clip_norm=100.0f0, weight_decay=0.0f0);
        analog_gradient_scale=0.5f0,
        hard_event_gradient_scale=2.0f0,
        due_mask=due,
    )
    @test stats.active_groups == 3
    @test state.group_steps == UInt64[1, 0, 1, 1, 0]
    @test state.total_step == 1
    for index in (2, 5)
        @test registry.groups[index].parameter == parameters_before[index]
        @test state.moments[index].first == first_before[index]
        @test state.moments[index].second == second_before[index]
    end
    for index in (1, 3, 4)
        @test registry.groups[index].parameter != parameters_before[index]
        @test state.moments[index].first != first_before[index]
        @test state.moments[index].second != second_before[index]
    end
    @test_throws DimensionMismatch apply_dual_optimizer_boundary!(
        state,
        registry,
        analog,
        hard,
        AdamWConfig();
        due_mask=(true, false),
    )
end

@testset "dual preflight is lane-specific and transactional" begin
    registry = dual_optimizer_fixture()
    state = AdamWState(registry)
    analog, hard = dual_gradient_lanes(registry)
    config = AdamWConfig(clip_norm=10.0f0, weight_decay=0.0f0)

    bad_names = GradientBufferSet(
        (
            :core_cell_raw,
            :event_raw,
            :semantic_projection_raw,
            :output_cell_raw,
            :output_projection_raw,
        ),
        analog.buffers,
    )
    snapshot = optimizer_snapshot(registry, state, analog, hard)
    @test_throws ArgumentError apply_dual_optimizer_boundary!(
        state, registry, bad_names, hard, config,
    )
    test_optimizer_snapshot(snapshot, registry, state, analog, hard)

    wrong_shape_buffers = (
        zeros(Float32, 1),
        analog.buffers[2],
        analog.buffers[3],
        analog.buffers[4],
        analog.buffers[5],
    )
    wrong_shape = GradientBufferSet(registry, wrong_shape_buffers)
    snapshot = optimizer_snapshot(registry, state, analog, hard)
    @test_throws DimensionMismatch apply_dual_optimizer_boundary!(
        state, registry, wrong_shape, hard, config,
    )
    test_optimizer_snapshot(snapshot, registry, state, analog, hard)

    analog.buffers[1][1] = NaN32
    snapshot = optimizer_snapshot(registry, state, analog, hard)
    @test_throws DomainError apply_dual_optimizer_boundary!(
        state, registry, analog, hard, config,
    )
    test_optimizer_snapshot(snapshot, registry, state, analog, hard)
    analog.buffers[1][1] = 0.0f0

    hard.buffers[2][1] = Inf32
    snapshot = optimizer_snapshot(registry, state, analog, hard)
    @test_throws DomainError apply_dual_optimizer_boundary!(
        state, registry, analog, hard, config,
    )
    test_optimizer_snapshot(snapshot, registry, state, analog, hard)
    hard.buffers[2][1] = 0.0f0

    analog.buffers[1][1] = floatmax(Float32)
    hard.buffers[1][1] = floatmax(Float32)
    overflow_config = AdamWConfig(
        clip_norm=floatmax(Float32), weight_decay=0.0f0,
    )
    snapshot = optimizer_snapshot(registry, state, analog, hard)
    @test_throws OverflowError apply_dual_optimizer_boundary!(
        state,
        registry,
        analog,
        hard,
        overflow_config;
        due_mask=(true, false, false, false, false),
    )
    test_optimizer_snapshot(snapshot, registry, state, analog, hard)

    analog.buffers[1][1] = 0.0f0
    hard.buffers[1][1] = 0.0f0
    snapshot = optimizer_snapshot(registry, state, analog, hard)
    @test_throws ArgumentError apply_dual_optimizer_boundary!(
        state,
        registry,
        analog,
        hard,
        config;
        hard_event_gradient_scale=1.0e300,
    )
    test_optimizer_snapshot(snapshot, registry, state, analog, hard)
end

@testset "steady-state dual optimizer boundary allocates zero bytes" begin
    registry = dual_optimizer_fixture()
    state = AdamWState(registry)
    analog, hard = dual_gradient_lanes(registry)
    for buffer in analog.buffers
        fill!(buffer, 1.0f-3)
    end
    for buffer in hard.buffers
        buffer === nothing || fill!(buffer, -2.0f-4)
    end
    config = AdamWConfig(
        learning_rate=1.0f-4,
        clip_norm=10.0f0,
        weight_decay=1.0f-4,
    )
    due = (true, false, true, true, false)
    apply_dual_optimizer_boundary!(
        state, registry, analog, hard, config; due_mask=due,
    )
    apply_dual_optimizer_boundary!(
        state, registry, analog, hard, config; due_mask=due,
    )
    allocated = @allocated apply_dual_optimizer_boundary!(
        state, registry, analog, hard, config; due_mask=due,
    )
    @test allocated == 0
end
