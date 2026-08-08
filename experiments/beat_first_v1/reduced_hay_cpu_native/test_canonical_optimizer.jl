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
