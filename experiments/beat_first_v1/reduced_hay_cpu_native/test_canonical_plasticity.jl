using Test

module CanonicalPlasticityTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "DendriticAxonPacket.jl"))
include(joinpath(@__DIR__, "CanonicalOptimizer.jl"))
include(joinpath(@__DIR__, "CanonicalLocalLearning.jl"))
include(joinpath(@__DIR__, "CanonicalPlasticity.jl"))
end

const Cell = CanonicalPlasticityTestHarness.ActiveApicalCell
const Optimizer = CanonicalPlasticityTestHarness.CanonicalOptimizer
const Local = CanonicalPlasticityTestHarness.CanonicalLocalLearning
const Plasticity = CanonicalPlasticityTestHarness.CanonicalPlasticity

function parameter_fixture(
    cells::Int,
    destinations::AbstractVector{<:Integer};
    cell_multiplier=1.0f0,
    conductance_multiplier=1.0f0,
)
    default = Cell.default_raw_parameters()
    cell_raw = repeat(default, 1, cells)
    cell_gradient = zeros(Float32, size(cell_raw))
    conductance = fill(
        Optimizer.inverse_softplus(0.5f0),
        length(destinations),
    )
    conductance_gradient = zeros(Float32, length(conductance))
    registry = Optimizer.ParameterRegistry(
        Optimizer.ParameterGroup(
            :cell_raw,
            cell_raw,
            cell_gradient,
            Optimizer.CELL_RAW;
            multiplier=cell_multiplier,
        ),
        Optimizer.ParameterGroup(
            :contact_raw,
            conductance,
            conductance_gradient,
            Optimizer.INVERSE_SOFTPLUS_CONDUCTANCE;
            multiplier=conductance_multiplier,
            lower_bound=1.0f-4,
            upper_bound=4.0f0,
        ),
    )
    optimizer = Optimizer.AdamWState(registry)
    for moments in optimizer.moments
        fill!(moments.first, 1.0f0)
        fill!(moments.second, 2.0f0)
    end
    return registry, optimizer
end

function segmented_cell_fixture(
    core_cells::Int=1436,
    output_cells::Int=22;
    core_multiplier=1.0f0,
    output_multiplier=1.0f0,
)
    default = Cell.default_raw_parameters()
    core = repeat(default, 1, core_cells)
    output = repeat(default, 1, output_cells)
    registry = Optimizer.ParameterRegistry(
        Optimizer.ParameterGroup(
            :core_cell_raw,
            core,
            zeros(Float32, size(core)),
            Optimizer.CELL_RAW;
            multiplier=core_multiplier,
        ),
        Optimizer.ParameterGroup(
            :output_cell_raw,
            output,
            zeros(Float32, size(output)),
            Optimizer.CELL_RAW;
            multiplier=output_multiplier,
        ),
    )
    optimizer = Optimizer.AdamWState(registry)
    for moments in optimizer.moments
        fill!(moments.first, 1.0f0)
        fill!(moments.second, 2.0f0)
    end
    return registry, optimizer
end

function segmented_conductance_fixture(;
    semantic_multiplier=1.0f0,
    event_multiplier=1.0f0,
    output_multiplier=1.0f0,
)
    initial = Optimizer.inverse_softplus(0.5f0)
    semantic = fill(initial, 2, 2)
    event = fill(initial, 3)
    output = fill(initial, 1, 2)
    make_group(name, parameter, multiplier) = Optimizer.ParameterGroup(
        name,
        parameter,
        zeros(Float32, size(parameter)),
        Optimizer.INVERSE_SOFTPLUS_CONDUCTANCE;
        multiplier=multiplier,
        lower_bound=1.0f-4,
        upper_bound=4.0f0,
    )
    registry = Optimizer.ParameterRegistry(
        make_group(
            :semantic_projection_raw,
            semantic,
            semantic_multiplier,
        ),
        make_group(:event_raw, event, event_multiplier),
        make_group(
            :output_projection_raw,
            output,
            output_multiplier,
        ),
    )
    optimizer = Optimizer.AdamWState(registry)
    for moments in optimizer.moments
        fill!(moments.first, 1.0f0)
        fill!(moments.second, 2.0f0)
    end
    destinations = (
        UInt16[1, 0, 2, 3],
        UInt16[2, 1, 0],
        UInt16[3, 1],
    )
    return registry, optimizer, destinations
end

mutable struct ResetCounter
    count::Int
    last_linear::Int
end

@inline function (counter::ResetCounter)(::Symbol, index)
    counter.count += 1
    counter.last_linear = index isa CartesianIndex ? index[1] : Int(index)
    return nothing
end

mutable struct ThrowingReset
    count::Int
    throw_after::Int
end

@inline function (reset::ThrowingReset)(::Symbol, index)
    reset.count += 1
    reset.count >= reset.throw_after && error("injected moment-reset failure")
    return nothing
end

mutable struct EligibilityResetCounter
    count::Int
    last::Int
end

@inline function (counter::EligibilityResetCounter)(index::Integer)
    counter.count += 1
    counter.last = Int(index)
    return nothing
end

mutable struct ThrowingEligibilityReset
    count::Int
end

@inline function (reset::ThrowingEligibilityReset)(index::Integer)
    reset.count += 1
    error("injected eligibility-reset failure at $(Int(index))")
end

struct AcceptEveryValidSwap end
@inline (::AcceptEveryValidSwap)(edge, source, old, new, receptor) =
    edge > 0 && source > 0 && old > 0 && new > 0 && receptor > 0

function publish_fixture!(batch, order)
    spikes = (
        UInt32[1, 0],
        UInt32[0, 2],
    )
    observations = (UInt32(2), UInt32(4))
    activity = (
        Float32[0.4, 0.2],
        Float32[0.8, 0.4],
    )
    incoming = (
        Float32[0.6, 0.8],
        Float32[1.0, 1.2],
    )
    third = (
        Float32[2.0, 0.0, 3.0],
        Float32[-1.0, 4.0, 0.0],
    )
    contribution = (
        Float32[0.5, 9.0, 0.0],
        Float32[0.25, -0.5, 7.0],
    )
    contact_activity = (
        Float32[0.1, 0.1, 0.1],
        Float32[0.2, 0.2, 0.2],
    )
    logical_slots = (1, 3)
    for item in order
        Plasticity.record_candidate_plasticity!(
            batch,
            logical_slots[item],
            spikes[item],
            observations[item],
            activity[item],
            incoming[item],
            third[item],
            contribution[item],
            contact_activity[item],
        )
    end
    return batch
end

@testset "coordinator-owned deterministic candidate reduction" begin
    config = Local.PlasticityConfig(
        firing_ema_decay=0.5,
        utility_decay=0.5,
        connection_cost=0.0,
    )
    forward_order = Plasticity.CandidatePlasticityBatch(2, 3, 4)
    reverse_order = Plasticity.CandidatePlasticityBatch(2, 3, 4)
    Plasticity.begin_plasticity_batch!(forward_order)
    Plasticity.begin_plasticity_batch!(reverse_order)
    publish_fixture!(forward_order, (1, 2))
    publish_fixture!(reverse_order, (2, 1))
    state_a = Plasticity.PlasticityState(config, 2, 3)
    state_b = Plasticity.PlasticityState(config, 2, 3)
    stats_a = Plasticity.reduce_candidate_plasticity!(
        state_a,
        forward_order,
        config,
        2,
    )
    stats_b = Plasticity.reduce_candidate_plasticity!(
        state_b,
        reverse_order,
        config,
        2,
    )
    @test stats_a == stats_b
    @test stats_a.candidates == 2
    @test stats_a.observations == 6
    @test stats_a.utility_nonzero == 2
    @test state_a.firing_rate == state_b.firing_rate
    @test state_a.activity_ema == state_b.activity_ema
    @test state_a.incoming_conductance_ema ==
        state_b.incoming_conductance_ema
    @test state_a.utility == state_b.utility
    @test state_a.utility[1] > 0.0f0
    @test state_a.utility[2] > 0.0f0
    @test state_a.utility[3] == 0.0f0
    @test state_a.utility_updates == 2
    @test state_a.reduced_batches == 1
    @test_throws ArgumentError Plasticity.reduce_candidate_plasticity!(
        state_a,
        forward_order,
        config,
        3,
    )
end

@testset "utility requires the same third factor and local contribution" begin
    config = Local.PlasticityConfig(
        firing_ema_decay=0.0,
        utility_decay=0.0,
        connection_cost=0.0,
    )
    batch = Plasticity.CandidatePlasticityBatch(1, 3, 1)
    Plasticity.begin_plasticity_batch!(batch)
    Plasticity.record_candidate_plasticity!(
        batch,
        1,
        UInt32[0],
        1,
        Float32[0],
        Float32[0.2],
        Float32[0, 2, 2],
        Float32[4, 0, -3],
        Float32[0, 0, 0],
    )
    state = Plasticity.PlasticityState(config, 1, 3)
    Plasticity.reduce_candidate_plasticity!(state, batch, config, 1)
    @test state.utility == Float32[0, 0, 6]

    # A nonzero activity cost cannot manufacture positive task utility.
    costly = Local.PlasticityConfig(
        firing_ema_decay=0.0,
        utility_decay=0.0,
        connection_cost=1.0,
    )
    costly_batch = Plasticity.CandidatePlasticityBatch(1, 1, 1)
    Plasticity.begin_plasticity_batch!(costly_batch)
    Plasticity.record_candidate_plasticity!(
        costly_batch,
        1,
        UInt32[0],
        1,
        Float32[0],
        Float32[0],
        Float32[0],
        Float32[9],
        Float32[3],
    )
    costly_state = Plasticity.PlasticityState(costly, 1, 1)
    Plasticity.reduce_candidate_plasticity!(
        costly_state,
        costly_batch,
        costly,
        1,
    )
    @test costly_state.utility[1] == 0.0f0
end

@testset "physical intrinsic homeostasis and synaptic scaling" begin
    config = Local.PlasticityConfig(
        target_rate_min=0.05,
        target_rate_max=0.25,
        threshold_homeostasis_step=0.2,
        adaptation_homeostasis_step=0.1,
        synaptic_scaling_rate=0.2,
    )
    destinations = Int[1, 1, 2, 3, 0]
    registry, optimizer = parameter_fixture(3, destinations)
    cell_group = registry.groups[1]
    contact_group = registry.groups[2]
    reset = Plasticity.OptimizerMomentReset(optimizer, registry)
    state = Plasticity.PlasticityState(config, 3, length(destinations))
    state.firing_rate .= Float32[0.0, 0.1, 0.8]

    threshold_index = Cell.P_SOMA_THRESHOLD_GAP
    adaptation_index = Cell.P_ADAPTATION_GAIN
    threshold_before = [
        Plasticity.cell_physical_parameter(
            cell_group.parameter[threshold_index, cell],
            threshold_index,
        ) for cell in 1:3
    ]
    adaptation_before = [
        Plasticity.cell_physical_parameter(
            cell_group.parameter[adaptation_index, cell],
            adaptation_index,
        ) for cell in 1:3
    ]
    cell_raw_before = copy(cell_group.parameter)
    @test Plasticity.apply_intrinsic_homeostasis!(
        state,
        config,
        false,
        cell_group,
        reset,
    ) == 0
    @test cell_group.parameter == cell_raw_before

    @test Plasticity.apply_intrinsic_homeostasis!(
        state,
        config,
        true,
        cell_group,
        reset,
    ) == 2
    threshold_after = [
        Plasticity.cell_physical_parameter(
            cell_group.parameter[threshold_index, cell],
            threshold_index,
        ) for cell in 1:3
    ]
    adaptation_after = [
        Plasticity.cell_physical_parameter(
            cell_group.parameter[adaptation_index, cell],
            adaptation_index,
        ) for cell in 1:3
    ]
    @test threshold_after[1] < threshold_before[1]
    @test adaptation_after[1] < adaptation_before[1]
    @test threshold_after[2] == threshold_before[2]
    @test adaptation_after[2] == adaptation_before[2]
    @test threshold_after[3] > threshold_before[3]
    @test adaptation_after[3] > adaptation_before[3]
    @test optimizer.moments[1].first[threshold_index, 1] == 0.0f0
    @test optimizer.moments[1].first[adaptation_index, 1] == 0.0f0
    @test optimizer.moments[1].first[threshold_index, 3] == 0.0f0
    @test optimizer.moments[1].first[adaptation_index, 3] == 0.0f0

    physical_before = Optimizer.physical_conductance.(
        copy(contact_group.parameter),
    )
    @test Plasticity.apply_synaptic_scaling!(
        state,
        config,
        true,
        contact_group,
        destinations,
        reset,
    ) == 3
    physical_after = Optimizer.physical_conductance.(contact_group.parameter)
    @test all(physical_after[1:2] .> physical_before[1:2])
    @test physical_after[3] == physical_before[3]
    @test physical_after[4] < physical_before[4]
    @test physical_after[5] == physical_before[5]
    @test all(iszero, optimizer.moments[2].first[[1, 2, 4]])
    @test optimizer.moments[2].first[3] == 1.0f0
    @test optimizer.moments[2].first[5] == 1.0f0
    @test state.synaptic_scaling_events == 3

    # Large physical steps remain finite and inside the cell transform bounds.
    bounded_config = Local.PlasticityConfig(
        target_rate_min=0.05,
        target_rate_max=0.25,
        threshold_homeostasis_step=1.0e6,
        adaptation_homeostasis_step=1.0e6,
    )
    bounded_state = Plasticity.PlasticityState(bounded_config, 3, 5)
    bounded_state.firing_rate .= Float32[0, 0.1, 1]
    Plasticity.apply_intrinsic_homeostasis!(
        bounded_state,
        bounded_config,
        true,
        cell_group,
        reset,
    )
    for cell in (1, 3), parameter in (threshold_index, adaptation_index)
        physical = Plasticity.cell_physical_parameter(
            cell_group.parameter[parameter, cell],
            parameter,
        )
        @test isfinite(cell_group.parameter[parameter, cell])
        @test Cell.PARAMETER_LOWER[parameter] <= physical <=
            Cell.PARAMETER_UPPER[parameter]
    end
end

@testset "segmented canonical cell homeostasis covers all 1458 EMAs" begin
    config = Local.PlasticityConfig(
        target_rate_min=0.05,
        target_rate_max=0.25,
        threshold_homeostasis_step=0.2,
        adaptation_homeostasis_step=0.1,
    )
    registry, optimizer = segmented_cell_fixture()
    groups = registry.groups
    ranges = (1:1436, 1437:1458)
    state = Plasticity.PlasticityState(config, 1458, 0)
    fill!(state.firing_rate, 0.1f0)
    state.firing_rate[1] = 0.0f0
    state.firing_rate[1437] = 1.0f0
    reset = Plasticity.OptimizerMomentReset(optimizer, registry)
    threshold = Cell.P_SOMA_THRESHOLD_GAP
    adaptation = Cell.P_ADAPTATION_GAIN
    core_before = Plasticity.cell_physical_parameter(
        groups[1].parameter[threshold, 1],
        threshold,
    )
    output_before = Plasticity.cell_physical_parameter(
        groups[2].parameter[threshold, 1],
        threshold,
    )

    @test Plasticity.apply_intrinsic_homeostasis!(
        state, config, true, groups, ranges, reset,
    ) == 2
    @test Plasticity.cell_physical_parameter(
        groups[1].parameter[threshold, 1], threshold,
    ) < core_before
    @test Plasticity.cell_physical_parameter(
        groups[2].parameter[threshold, 1], threshold,
    ) > output_before
    @test optimizer.moments[1].first[threshold, 1] == 0.0f0
    @test optimizer.moments[1].first[adaptation, 1] == 0.0f0
    @test optimizer.moments[2].first[threshold, 1] == 0.0f0
    @test optimizer.moments[2].first[adaptation, 1] == 0.0f0
    @test optimizer.moments[2].first[threshold, 2] == 1.0f0
    @test state.homeostasis_events == UInt64(2)

    offset_registry, offset_optimizer = segmented_cell_fixture()
    offset_state = Plasticity.PlasticityState(config, 1458, 0)
    fill!(offset_state.firing_rate, 0.1f0)
    offset_state.firing_rate[1438] = 0.0f0
    @test Plasticity.apply_intrinsic_homeostasis!(
        offset_state,
        config,
        true,
        offset_registry.groups,
        (0, 1436),
        Plasticity.OptimizerMomentReset(offset_optimizer, offset_registry),
    ) == 1
    @test offset_optimizer.moments[2].first[threshold, 2] == 0.0f0
    @test offset_optimizer.moments[2].first[threshold, 1] == 1.0f0
end

@testset "segmented cell freeze and malformed maps are atomic" begin
    config = Local.PlasticityConfig(
        target_rate_min=0.05,
        target_rate_max=0.25,
    )
    registry, optimizer = segmented_cell_fixture(
        2,
        2;
        output_multiplier=0.0f0,
    )
    groups = registry.groups
    reset = Plasticity.OptimizerMomentReset(optimizer, registry)
    state = Plasticity.PlasticityState(config, 4, 0)
    state.firing_rate .= Float32[0.0, 0.1, 1.0, 0.1]
    frozen_before = copy(groups[2].parameter)
    frozen_moments = copy(optimizer.moments[2].first)
    @test Plasticity.apply_intrinsic_homeostasis!(
        state, config, true, groups, (1:2, 3:4), reset,
    ) == 1
    @test groups[2].parameter == frozen_before
    @test optimizer.moments[2].first == frozen_moments

    atomic_registry, atomic_optimizer = segmented_cell_fixture(2, 2)
    atomic_groups = atomic_registry.groups
    atomic_reset = Plasticity.OptimizerMomentReset(
        atomic_optimizer,
        atomic_registry,
    )
    atomic_state = Plasticity.PlasticityState(config, 4, 0)
    atomic_state.firing_rate .= Float32[0.0, 0.1, 1.0, 0.1]
    parameters_before = map(group -> copy(group.parameter), atomic_groups)
    moments_before = map(
        moments -> (copy(moments.first), copy(moments.second)),
        atomic_optimizer.moments,
    )
    @test_throws DimensionMismatch Plasticity.apply_intrinsic_homeostasis!(
        atomic_state, config, true, atomic_groups, (1:2, 4:5), atomic_reset,
    )
    @test_throws DimensionMismatch Plasticity.apply_intrinsic_homeostasis!(
        atomic_state, config, true, atomic_groups, (1:2, 2:3), atomic_reset,
    )
    @test_throws DimensionMismatch Plasticity.apply_intrinsic_homeostasis!(
        atomic_state, config, true, atomic_groups, (1:1, 2:4), atomic_reset,
    )
    @test all(
        atomic_groups[index].parameter == parameters_before[index]
        for index in eachindex(atomic_groups)
    )
    @test atomic_state.homeostasis_events == UInt64(0)

    atomic_groups[2].parameter[1, 2] = NaN32
    core_before = copy(atomic_groups[1].parameter)
    @test_throws DomainError Plasticity.apply_intrinsic_homeostasis!(
        atomic_state, config, true, atomic_groups, (1:2, 3:4), atomic_reset,
    )
    @test atomic_groups[1].parameter == core_before
    @test all(
        atomic_optimizer.moments[index].first == moments_before[index][1] &&
            atomic_optimizer.moments[index].second == moments_before[index][2]
        for index in eachindex(atomic_optimizer.moments)
    )
    @test atomic_state.homeostasis_events == UInt64(0)
end

@testset "segmented conductance scaling is transactional" begin
    config = Local.PlasticityConfig(
        target_rate_min=0.05,
        target_rate_max=0.25,
        synaptic_scaling_rate=0.2,
    )
    registry, optimizer, destinations = segmented_conductance_fixture()
    groups = registry.groups
    state = Plasticity.PlasticityState(config, 3, 0)
    state.firing_rate .= Float32[0.0, 1.0, 0.1]
    reset = Plasticity.OptimizerMomentReset(optimizer, registry)
    physical_before = map(
        group -> Optimizer.physical_conductance.(copy(group.parameter)),
        groups,
    )
    @test Plasticity.apply_synaptic_scaling!(
        state, config, true, groups, destinations, reset,
    ) == 5
    physical_after = map(
        group -> Optimizer.physical_conductance.(group.parameter),
        groups,
    )
    @test physical_after[1][1] > physical_before[1][1]
    @test physical_after[1][2] == physical_before[1][2]
    @test physical_after[1][3] < physical_before[1][3]
    @test physical_after[1][4] == physical_before[1][4]
    @test physical_after[2][1] < physical_before[2][1]
    @test physical_after[2][2] > physical_before[2][2]
    @test physical_after[2][3] == physical_before[2][3]
    @test physical_after[3][1] == physical_before[3][1]
    @test physical_after[3][2] > physical_before[3][2]
    @test all(iszero, optimizer.moments[1].first[[1, 3]])
    @test optimizer.moments[1].first[2] == 1.0f0
    @test all(iszero, optimizer.moments[2].first[1:2])
    @test optimizer.moments[2].first[3] == 1.0f0
    @test optimizer.moments[3].first[1] == 1.0f0
    @test optimizer.moments[3].first[2] == 0.0f0
    @test state.synaptic_scaling_events == UInt64(5)

    frozen_registry, frozen_optimizer, frozen_destinations =
        segmented_conductance_fixture(event_multiplier=0.0f0)
    frozen_state = Plasticity.PlasticityState(config, 3, 0)
    frozen_state.firing_rate .= Float32[0.0, 1.0, 0.1]
    frozen_group_before = copy(frozen_registry.groups[2].parameter)
    frozen_moment_before = copy(frozen_optimizer.moments[2].first)
    @test Plasticity.apply_synaptic_scaling!(
        frozen_state,
        config,
        true,
        frozen_registry.groups,
        frozen_destinations,
        Plasticity.OptimizerMomentReset(frozen_optimizer, frozen_registry),
    ) == 3
    @test frozen_registry.groups[2].parameter == frozen_group_before
    @test frozen_optimizer.moments[2].first == frozen_moment_before
end

@testset "segmented conductance late failures precede all mutation" begin
    config = Local.PlasticityConfig(
        target_rate_min=0.05,
        target_rate_max=0.25,
    )
    registry, optimizer, destinations = segmented_conductance_fixture()
    groups = registry.groups
    state = Plasticity.PlasticityState(config, 3, 0)
    state.firing_rate .= Float32[0.0, 1.0, 0.1]
    reset = Plasticity.OptimizerMomentReset(optimizer, registry)
    parameter_before = map(group -> copy(group.parameter), groups)
    moment_before = map(
        moments -> (copy(moments.first), copy(moments.second)),
        optimizer.moments,
    )
    bad_shape = (destinations[1], destinations[2][1:2], destinations[3])
    @test_throws DimensionMismatch Plasticity.apply_synaptic_scaling!(
        state, config, true, groups, bad_shape, reset,
    )
    bad_destination = (
        destinations[1], destinations[2], UInt16[3, 4],
    )
    @test_throws BoundsError Plasticity.apply_synaptic_scaling!(
        state, config, true, groups, bad_destination, reset,
    )
    @test all(
        groups[index].parameter == parameter_before[index]
        for index in eachindex(groups)
    )

    groups[3].parameter[2] = NaN32
    first_before = copy(groups[1].parameter)
    second_before = copy(groups[2].parameter)
    @test_throws DomainError Plasticity.apply_synaptic_scaling!(
        state, config, true, groups, destinations, reset,
    )
    @test groups[1].parameter == first_before
    @test groups[2].parameter == second_before
    @test all(
        optimizer.moments[index].first == moment_before[index][1] &&
            optimizer.moments[index].second == moment_before[index][2]
        for index in eachindex(optimizer.moments)
    )
    @test state.synaptic_scaling_events == UInt64(0)
end

@testset "slow plasticity callbacks, counters, and segments fail atomically" begin
    config = Local.PlasticityConfig(
        target_rate_min=0.05,
        target_rate_max=0.25,
        threshold_homeostasis_step=0.2,
        adaptation_homeostasis_step=0.1,
        synaptic_scaling_rate=0.2,
    )
    registry, optimizer = segmented_cell_fixture(2, 2)
    groups = registry.groups
    reset = Plasticity.OptimizerMomentReset(optimizer, registry)
    state = Plasticity.PlasticityState(config, 4, 0)
    state.firing_rate .= Float32[0, 0.1, 1, 0.1]
    parameters_before = map(group -> copy(group.parameter), groups)
    moments_before = map(
        moment -> (copy(moment.first), copy(moment.second)),
        optimizer.moments,
    )

    arbitrary = ThrowingReset(0, 1)
    @test_throws ArgumentError Plasticity.apply_intrinsic_homeostasis!(
        state, config, true, groups, (1:2, 3:4), arbitrary,
    )
    @test arbitrary.count == 0
    @test all(
        groups[index].parameter == parameters_before[index]
        for index in eachindex(groups)
    )

    @test_throws ArgumentError Plasticity.apply_intrinsic_homeostasis!(
        state, config, true, (groups[1], groups[1]), (1:2, 3:4), reset,
    )
    @test all(
        groups[index].parameter == parameters_before[index]
        for index in eachindex(groups)
    )

    # Two logical cells would change, so even a counter with one remaining
    # value must fail before either segment is touched.
    state.homeostasis_events = typemax(UInt64) - UInt64(1)
    @test_throws OverflowError Plasticity.apply_intrinsic_homeostasis!(
        state, config, true, groups, (1:2, 3:4), reset,
    )
    @test all(
        groups[index].parameter == parameters_before[index]
        for index in eachindex(groups)
    )
    @test all(
        optimizer.moments[index].first == moments_before[index][1] &&
            optimizer.moments[index].second == moments_before[index][2]
        for index in eachindex(optimizer.moments)
    )
    @test state.homeostasis_events == typemax(UInt64) - UInt64(1)

    # Distinct ParameterGroup wrappers over overlapping views are aliases even
    # though neither the group nor the parameter object is `===`.
    default = Cell.default_raw_parameters()
    aliased_cell_storage = repeat(default, 1, 3)
    left_cell_view = @view aliased_cell_storage[:, 1:2]
    right_cell_view = @view aliased_cell_storage[:, 2:3]
    aliased_cell_groups = (
        Optimizer.ParameterGroup(
            :left_cell_raw,
            left_cell_view,
            zeros(Float32, size(left_cell_view)),
            Optimizer.CELL_RAW,
        ),
        Optimizer.ParameterGroup(
            :right_cell_raw,
            right_cell_view,
            zeros(Float32, size(right_cell_view)),
            Optimizer.CELL_RAW,
        ),
    )
    aliased_cell_before = copy(aliased_cell_storage)
    @test Base.mightalias(
        aliased_cell_groups[1].parameter,
        aliased_cell_groups[2].parameter,
    )
    @test_throws ArgumentError Plasticity.apply_intrinsic_homeostasis!(
        state,
        config,
        true,
        aliased_cell_groups,
        (1:2, 3:4),
        arbitrary,
    )
    @test aliased_cell_storage == aliased_cell_before
    @test arbitrary.count == 0

    invalid_rate = Plasticity.PlasticityState(config, 4, 0)
    invalid_rate.firing_rate[1] = -0.01f0
    @test_throws DomainError Plasticity.apply_intrinsic_homeostasis!(
        invalid_rate, config, false, groups, (1:2, 3:4), reset,
    )
    invalid_rate.firing_rate[1] = 1.01f0
    @test_throws DomainError Plasticity.apply_intrinsic_homeostasis!(
        invalid_rate, config, false, groups, (1:2, 3:4), reset,
    )
    invalid_rate.firing_rate[1] = NaN32
    @test_throws DomainError Plasticity.apply_intrinsic_homeostasis!(
        invalid_rate, config, false, groups, (1:2, 3:4), reset,
    )

    conductance_registry, conductance_optimizer, destinations =
        segmented_conductance_fixture()
    conductance_groups = conductance_registry.groups
    conductance_reset = Plasticity.OptimizerMomentReset(
        conductance_optimizer,
        conductance_registry,
    )
    conductance_state = Plasticity.PlasticityState(config, 3, 0)
    conductance_state.firing_rate .= Float32[0, 1, 0.1]
    conductance_before = map(
        group -> copy(group.parameter), conductance_groups,
    )
    conductance_moments_before = map(
        moment -> (copy(moment.first), copy(moment.second)),
        conductance_optimizer.moments,
    )
    arbitrary = ThrowingReset(0, 1)
    @test_throws ArgumentError Plasticity.apply_synaptic_scaling!(
        conductance_state,
        config,
        true,
        conductance_groups,
        destinations,
        arbitrary,
    )
    @test arbitrary.count == 0
    @test all(
        conductance_groups[index].parameter == conductance_before[index]
        for index in eachindex(conductance_groups)
    )
    @test_throws ArgumentError Plasticity.apply_synaptic_scaling!(
        conductance_state,
        config,
        true,
        (conductance_groups[1], conductance_groups[1]),
        (destinations[1], destinations[1]),
        conductance_reset,
    )
    # Five contacts would change, so four remaining counter values are not
    # enough; the preflight must precede every conductance and moment write.
    conductance_state.synaptic_scaling_events =
        typemax(UInt64) - UInt64(4)
    @test_throws OverflowError Plasticity.apply_synaptic_scaling!(
        conductance_state,
        config,
        true,
        conductance_groups,
        destinations,
        conductance_reset,
    )
    @test all(
        conductance_groups[index].parameter == conductance_before[index]
        for index in eachindex(conductance_groups)
    )
    @test all(
        conductance_optimizer.moments[index].first ==
            conductance_moments_before[index][1] &&
            conductance_optimizer.moments[index].second ==
            conductance_moments_before[index][2]
        for index in eachindex(conductance_optimizer.moments)
    )
    @test conductance_state.synaptic_scaling_events ==
        typemax(UInt64) - UInt64(4)

    aliased_conductance_storage = fill(
        Optimizer.inverse_softplus(0.5f0),
        3,
    )
    left_conductance_view = @view aliased_conductance_storage[1:2]
    right_conductance_view = @view aliased_conductance_storage[2:3]
    make_aliased_conductance(name, parameter) = Optimizer.ParameterGroup(
        name,
        parameter,
        zeros(Float32, size(parameter)),
        Optimizer.INVERSE_SOFTPLUS_CONDUCTANCE;
        lower_bound=1.0f-4,
        upper_bound=4.0f0,
    )
    aliased_conductance_groups = (
        make_aliased_conductance(:left_contact_raw, left_conductance_view),
        make_aliased_conductance(:right_contact_raw, right_conductance_view),
    )
    aliased_conductance_before = copy(aliased_conductance_storage)
    @test Base.mightalias(
        aliased_conductance_groups[1].parameter,
        aliased_conductance_groups[2].parameter,
    )
    @test_throws ArgumentError Plasticity.apply_synaptic_scaling!(
        conductance_state,
        config,
        true,
        aliased_conductance_groups,
        (UInt16[1, 2], UInt16[2, 3]),
        arbitrary,
    )
    @test aliased_conductance_storage == aliased_conductance_before
    @test arbitrary.count == 0

    invalid_conductance_rate = Plasticity.PlasticityState(config, 3, 0)
    invalid_conductance_rate.firing_rate[2] = Inf32
    @test_throws DomainError Plasticity.apply_synaptic_scaling!(
        invalid_conductance_rate,
        config,
        false,
        conductance_groups,
        destinations,
        conductance_reset,
    )
end

@testset "group multiplier zero is a strict plasticity freeze" begin
    config = Local.PlasticityConfig(
        target_rate_min=0.05,
        target_rate_max=0.25,
    )
    destination = Int[1, 2]
    registry, optimizer = parameter_fixture(
        2,
        destination;
        cell_multiplier=0.0f0,
        conductance_multiplier=0.0f0,
    )
    state = Plasticity.PlasticityState(config, 2, 2)
    state.firing_rate .= Float32[0, 1]
    parameters_before = map(group -> copy(group.parameter), registry.groups)
    moments_before = map(
        moments -> (copy(moments.first), copy(moments.second)),
        optimizer.moments,
    )
    reset = Plasticity.OptimizerMomentReset(optimizer, registry)
    @test Plasticity.apply_intrinsic_homeostasis!(
        state,
        config,
        true,
        registry.groups[1],
        reset,
    ) == 0
    @test Plasticity.apply_synaptic_scaling!(
        state,
        config,
        true,
        registry.groups[2],
        destination,
        reset,
    ) == 0
    @test all(
        registry.groups[index].parameter == parameters_before[index]
        for index in eachindex(registry.groups)
    )
    @test all(
        optimizer.moments[index].first == moments_before[index][1] &&
            optimizer.moments[index].second == moments_before[index][2]
        for index in eachindex(optimizer.moments)
    )
end

@testset "canonical no-rewire and one validated optional swap" begin
    source = Int[1, 1, 2, 2]
    destination = UInt16[2, 3, 1, 3]
    receptor = UInt8[1, 2, 1, 2]
    optional = Bool[true, true, true, false]
    proposal = Int[4, 2, 4, 4]
    disabled = Local.PlasticityConfig(structure_enabled=false)
    enabled = Local.PlasticityConfig(
        structure_enabled=true,
        max_swaps_per_node=1,
    )
    registry, optimizer = parameter_fixture(4, destination)
    state = Plasticity.PlasticityState(enabled, 4, 4)
    state.utility .= Float32[0.1, 0.01, 0.2, 0.0]
    destination_before = copy(destination)
    reset = Plasticity.OptimizerMomentReset(optimizer, registry)
    eligibility_reset = EligibilityResetCounter(0, 0)
    @test Plasticity.rewire_one_optional_contact!(
        state,
        disabled,
        true,
        1,
        source,
        destination,
        receptor,
        optional,
        proposal,
        registry.groups[2],
        AcceptEveryValidSwap(),
        reset,
        eligibility_reset,
    ) == 0
    @test destination == destination_before
    @test eligibility_reset.count == 0

    # Edge 2 has lower utility but proposes destination 2, already used by
    # edge 1.  The validated replacement is therefore edge 1 -> destination 4.
    @test Plasticity.rewire_one_optional_contact!(
        state,
        enabled,
        true,
        1,
        source,
        destination,
        receptor,
        optional,
        proposal,
        registry.groups[2],
        AcceptEveryValidSwap(),
        reset,
        eligibility_reset,
    ) == 1
    @test destination == UInt16[4, 3, 1, 3]
    @test length(unique(destination[1:2])) == 2
    @test receptor == UInt8[1, 2, 1, 2]
    @test Optimizer.physical_conductance(
        registry.groups[2].parameter[1],
    ) ≈ enabled.conductance_floor
    @test optimizer.moments[2].first[1] == 0.0f0
    @test optimizer.moments[2].second[1] == 0.0f0
    @test eligibility_reset.count == 1
    @test eligibility_reset.last == 1
    @test state.utility[1] == 0.0f0
    @test state.rewires == 1
end

@testset "rewire reset, bounds, and counter failures are atomic" begin
    source = Int[1, 1, 2, 2]
    receptor = UInt8[1, 2, 1, 2]
    optional = Bool[true, true, true, false]
    proposal = Int[4, 2, 4, 4]
    enabled = Local.PlasticityConfig(
        structure_enabled=true,
        max_swaps_per_node=1,
    )

    function fixture(config=enabled)
        fixture_destination = UInt16[2, 3, 1, 3]
        fixture_registry, fixture_optimizer = parameter_fixture(
            4,
            fixture_destination,
        )
        fixture_state = Plasticity.PlasticityState(config, 4, 4)
        fixture_state.utility .= Float32[0.1, 0.01, 0.2, 0.0]
        return (
            fixture_destination,
            fixture_registry,
            fixture_optimizer,
            fixture_state,
        )
    end

    function snapshot(destination, registry, optimizer, state)
        return (
            destination=copy(destination),
            parameter=copy(registry.groups[2].parameter),
            first=copy(optimizer.moments[2].first),
            second=copy(optimizer.moments[2].second),
            utility=copy(state.utility),
            rewires=state.rewires,
        )
    end

    function attempt!(
        state,
        config,
        destination,
        group,
        moment_reset,
        eligibility_reset,
    )
        return Plasticity.rewire_one_optional_contact!(
            state,
            config,
            true,
            1,
            source,
            destination,
            receptor,
            optional,
            proposal,
            group,
            AcceptEveryValidSwap(),
            moment_reset,
            eligibility_reset,
        )
    end

    destination, registry, optimizer, state = fixture()
    before = snapshot(destination, registry, optimizer, state)
    throwing_reset = ThrowingReset(0, 1)
    eligibility_reset = EligibilityResetCounter(0, 0)
    @test_throws ArgumentError attempt!(
        state,
        enabled,
        destination,
        registry.groups[2],
        throwing_reset,
        eligibility_reset,
    )
    @test throwing_reset.count == 0
    @test eligibility_reset.count == 0
    @test snapshot(destination, registry, optimizer, state) == before

    destination, registry, optimizer, state = fixture()
    foreign_destination,
    foreign_registry,
    foreign_optimizer,
    foreign_state = fixture()
    @test foreign_destination !== destination
    @test foreign_registry !== registry
    @test foreign_optimizer !== optimizer
    @test foreign_state !== state
    before = snapshot(destination, registry, optimizer, state)
    eligibility_reset = EligibilityResetCounter(0, 0)
    @test_throws ArgumentError attempt!(
        state,
        enabled,
        destination,
        registry.groups[2],
        Plasticity.OptimizerMomentReset(
            foreign_optimizer,
            foreign_registry,
        ),
        eligibility_reset,
    )
    @test eligibility_reset.count == 0
    @test snapshot(destination, registry, optimizer, state) == before

    disjoint = Local.PlasticityConfig(
        structure_enabled=true,
        max_swaps_per_node=1,
        conductance_floor=5.0f0,
        conductance_ceiling=6.0f0,
    )
    destination, registry, optimizer, state = fixture(disjoint)
    before = snapshot(destination, registry, optimizer, state)
    eligibility_reset = EligibilityResetCounter(0, 0)
    @test_throws ArgumentError attempt!(
        state,
        disjoint,
        destination,
        registry.groups[2],
        Plasticity.OptimizerMomentReset(optimizer, registry),
        eligibility_reset,
    )
    @test eligibility_reset.count == 0
    @test snapshot(destination, registry, optimizer, state) == before

    destination, registry, optimizer, state = fixture()
    state.rewires = typemax(UInt64)
    before = snapshot(destination, registry, optimizer, state)
    eligibility_reset = EligibilityResetCounter(0, 0)
    @test_throws OverflowError attempt!(
        state,
        enabled,
        destination,
        registry.groups[2],
        Plasticity.OptimizerMomentReset(optimizer, registry),
        eligibility_reset,
    )
    @test eligibility_reset.count == 0
    @test snapshot(destination, registry, optimizer, state) == before

    destination, registry, optimizer, state = fixture()
    before = snapshot(destination, registry, optimizer, state)
    throwing_eligibility = ThrowingEligibilityReset(0)
    @test_throws ErrorException attempt!(
        state,
        enabled,
        destination,
        registry.groups[2],
        Plasticity.OptimizerMomentReset(optimizer, registry),
        throwing_eligibility,
    )
    @test throwing_eligibility.count == 1
    @test snapshot(destination, registry, optimizer, state) == before
end

function publish_duplicate_invariant_fixture!(batch, candidates, order)
    common_spikes = UInt32[1, 0]
    common_activity = Float32[2, 4]
    common_incoming = Float32[6, 8]
    common_task_utility = Float32[2, 0, 0]
    common_contact_activity = Float32[4, 2, 0]
    Plasticity.record_state_common_plasticity!(
        batch,
        1,
        common_spikes,
        UInt8[2, 2],
        common_activity,
        common_incoming,
        common_task_utility,
        common_contact_activity,
    )
    candidate_spikes = UInt32[2, 4]
    candidate_activity = Float32[8, 12]
    candidate_incoming = Float32[16, 20]
    # The scalar loss has already assigned each duplicate its 1/K share.
    candidate_task_utility = Float32[inv(Float32(candidates)), 0, 0]
    candidate_contact_activity = Float32[8, 4, 0]
    for logical_slot in order
        Plasticity.record_candidate_plasticity!(
            batch,
            logical_slot,
            1,
            logical_slot,
            candidate_spikes,
            UInt8[4, 4],
            candidate_activity,
            candidate_incoming,
            candidate_task_utility,
            candidate_contact_activity,
        )
    end
    return batch
end

function plasticity_state_bits(state)
    return (
        copy(reinterpret(UInt8, state.firing_rate)),
        copy(reinterpret(UInt8, state.activity_ema)),
        copy(reinterpret(UInt8, state.incoming_conductance_ema)),
        copy(reinterpret(UInt8, state.utility)),
        state.reduced_batches,
        state.homeostasis_events,
        state.synaptic_scaling_events,
        state.utility_updates,
        state.rewires,
    )
end

@testset "canonical common/candidate reduction and duplicate invariance" begin
    config = Local.PlasticityConfig(
        firing_ema_decay=0.0,
        utility_decay=0.0,
        connection_cost=0.1,
    )
    k1 = Plasticity.CanonicalPlasticityBatch(2, 3, 1, 1)
    k4 = Plasticity.CanonicalPlasticityBatch(2, 3, 1, 4)
    Plasticity.begin_plasticity_batch!(k1)
    Plasticity.begin_plasticity_batch!(k4)
    publish_duplicate_invariant_fixture!(k1, 1, (1,))
    publish_duplicate_invariant_fixture!(k4, 4, (4, 2, 1, 3))
    state_k1 = Plasticity.PlasticityState(config, 2, 3)
    state_k4 = Plasticity.PlasticityState(config, 2, 3)
    stats_k1 = Plasticity.reduce_canonical_plasticity!(
        state_k1,
        k1,
        config,
        Int[0, 1],
    )
    stats_k4 = Plasticity.reduce_canonical_plasticity!(
        state_k4,
        k4,
        config,
        Int[0, 4],
    )
    @test reinterpret(UInt32, state_k1.firing_rate) ==
        reinterpret(UInt32, state_k4.firing_rate)
    @test reinterpret(UInt32, state_k1.activity_ema) ==
        reinterpret(UInt32, state_k4.activity_ema)
    @test reinterpret(UInt32, state_k1.incoming_conductance_ema) ==
        reinterpret(UInt32, state_k4.incoming_conductance_ema)
    @test reinterpret(UInt32, state_k1.utility) ==
        reinterpret(UInt32, state_k4.utility)
    @test state_k1.firing_rate == Float32[0.5, 2 / 3]
    @test state_k1.activity_ema == Float32[10 / 6, 16 / 6]
    @test state_k1.incoming_conductance_ema == Float32[22 / 6, 28 / 6]
    @test state_k1.utility == Float32[1.8, 0, 0]
    @test stats_k1.candidates == 1
    @test stats_k4.candidates == 4
    @test stats_k1.observations == 12
    @test stats_k4.observations == 36
    @test stats_k1.utility_nonzero == stats_k4.utility_nonzero == 1
end

@testset "canonical effective observation mass precedes the ratio" begin
    config = Local.PlasticityConfig(
        firing_ema_decay=0.0,
        utility_decay=0.0,
        connection_cost=0.0,
    )
    batch = Plasticity.CanonicalPlasticityBatch(1, 0, 2, 3)
    Plasticity.begin_plasticity_batch!(batch)
    empty = Float32[]
    Plasticity.record_state_common_plasticity!(
        batch, 1, UInt32[1], UInt8[2], Float32[2], Float32[4],
        empty, empty,
    )
    Plasticity.record_state_common_plasticity!(
        batch, 2, UInt32[4], UInt8[4], Float32[8], Float32[12],
        empty, empty,
    )
    Plasticity.record_candidate_plasticity!(
        batch, 1, 1, 1, UInt32[1], UInt8[2], Float32[4], Float32[6],
        empty, empty,
    )
    for candidate in 2:3
        Plasticity.record_candidate_plasticity!(
            batch, candidate, 2, candidate - 1, UInt32[0], UInt8[8],
            Float32[16], Float32[20], empty, empty,
        )
    end
    state = Plasticity.PlasticityState(config, 1, 0)
    stats = Plasticity.reduce_canonical_plasticity!(
        state,
        batch,
        config,
        Int[0, 1, 3],
    )
    # X_eff/N_eff, not the mean of the two per-state rates.
    @test state.firing_rate[1] == Float32(6 / 16)
    @test state.activity_ema[1] == Float32(30 / 16)
    @test state.incoming_conductance_ema[1] == Float32(42 / 16)
    @test stats.observations == 24
end

@testset "unvisited logical cells are COW/full invariant and unchanged" begin
    config = Local.PlasticityConfig(
        firing_ema_decay=0.5,
        utility_decay=0.0,
        connection_cost=0.0,
    )
    cow = Plasticity.CanonicalPlasticityBatch(2, 0, 1, 1)
    full = Plasticity.CanonicalPlasticityBatch(2, 0, 1, 1)
    empty = Float32[]
    for batch in (cow, full)
        Plasticity.begin_plasticity_batch!(batch)
        Plasticity.record_state_common_plasticity!(
            batch, 1, UInt32[1, 0], UInt8[1, 0],
            Float32[2, 0], Float32[3, 0], empty, empty,
        )
        Plasticity.record_candidate_plasticity!(
            batch, 1, 1, 1, UInt32[0, 0], UInt8[1, 0],
            Float32[4, 0], Float32[5, 0], empty, empty,
        )
    end
    state_cow = Plasticity.PlasticityState(config, 2, 0)
    state_full = Plasticity.PlasticityState(config, 2, 0)
    state_cow.firing_rate .= Float32[0.2, 0.7]
    state_cow.activity_ema .= Float32[0.3, 0.8]
    state_cow.incoming_conductance_ema .= Float32[0.4, 0.9]
    state_full.firing_rate .= state_cow.firing_rate
    state_full.activity_ema .= state_cow.activity_ema
    state_full.incoming_conductance_ema .= state_cow.incoming_conductance_ema
    before_preflight = plasticity_state_bits(state_cow)
    preflight_stats = Plasticity.preflight_canonical_plasticity(
        state_cow, cow, config, Int[0, 1],
    )
    @test plasticity_state_bits(state_cow) == before_preflight
    stats_cow = Plasticity.reduce_canonical_plasticity!(
        state_cow, cow, config, Int[0, 1],
    )
    stats_full = Plasticity.reduce_canonical_plasticity!(
        state_full, full, config, Int[0, 1],
    )
    @test preflight_stats == stats_cow == stats_full
    @test plasticity_state_bits(state_cow) == plasticity_state_bits(state_full)
    @test reinterpret(UInt32, state_cow.firing_rate[2]) ==
        reinterpret(UInt32, Float32(0.7))
    @test reinterpret(UInt32, state_cow.activity_ema[2]) ==
        reinterpret(UInt32, Float32(0.8))
    @test reinterpret(UInt32, state_cow.incoming_conductance_ema[2]) ==
        reinterpret(UInt32, Float32(0.9))

    invalid_work = Plasticity.CanonicalPlasticityBatch(2, 0, 1, 1)
    Plasticity.begin_plasticity_batch!(invalid_work)
    @test_throws DomainError Plasticity.record_state_common_plasticity!(
        invalid_work, 1, UInt32[0, 0], UInt8[1, 0],
        Float32[0, 1], Float32[0, 0], empty, empty,
    )
    @test invalid_work.common.stamp[1] == UInt32(0)
end

function publish_order_fixture!(batch, publication_order)
    common_spikes = (UInt32[1], UInt32[0])
    common_activity = (Float32[1], Float32[3])
    common_incoming = (Float32[2], Float32[4])
    candidate_spikes = (UInt32[0], UInt32[1], UInt32[1], UInt32[0])
    candidate_activity = (
        Float32[2], Float32[4], Float32[6], Float32[8],
    )
    candidate_incoming = (
        Float32[3], Float32[5], Float32[7], Float32[9],
    )
    zero_contact = Float32[0]
    one_contact = Float32[1]
    for encoded in publication_order
        if encoded < 0
            state_slot = -encoded
            Plasticity.record_state_common_plasticity!(
                batch, state_slot, common_spikes[state_slot], UInt8[2],
                common_activity[state_slot], common_incoming[state_slot],
                one_contact, zero_contact,
            )
        else
            state_slot = encoded <= 2 ? 1 : 2
            ordinal = encoded <= 2 ? encoded : encoded - 2
            Plasticity.record_candidate_plasticity!(
                batch, encoded, state_slot, ordinal,
                candidate_spikes[encoded], UInt8[2],
                candidate_activity[encoded], candidate_incoming[encoded],
                Float32[0.25], zero_contact,
            )
        end
    end
    return batch
end

@testset "canonical reduction is bit-stable under worker publication order" begin
    config = Local.PlasticityConfig(
        firing_ema_decay=0.25,
        utility_decay=0.5,
        connection_cost=0.0,
    )
    forward = Plasticity.CanonicalPlasticityBatch(1, 1, 2, 4)
    scrambled = Plasticity.CanonicalPlasticityBatch(1, 1, 2, 4)
    Plasticity.begin_plasticity_batch!(forward)
    Plasticity.begin_plasticity_batch!(scrambled)
    publish_order_fixture!(forward, (-1, 1, 2, -2, 3, 4))
    publish_order_fixture!(scrambled, (4, 2, -2, 1, -1, 3))
    state_forward = Plasticity.PlasticityState(config, 1, 1)
    state_scrambled = Plasticity.PlasticityState(config, 1, 1)
    stats_forward = Plasticity.reduce_canonical_plasticity!(
        state_forward, forward, config, Int[0, 2, 4],
    )
    stats_scrambled = Plasticity.reduce_canonical_plasticity!(
        state_scrambled, scrambled, config, Int[0, 2, 4],
    )
    @test stats_forward == stats_scrambled
    @test reinterpret(UInt32, state_forward.firing_rate) ==
        reinterpret(UInt32, state_scrambled.firing_rate)
    @test reinterpret(UInt32, state_forward.activity_ema) ==
        reinterpret(UInt32, state_scrambled.activity_ema)
    @test reinterpret(UInt32, state_forward.incoming_conductance_ema) ==
        reinterpret(UInt32, state_scrambled.incoming_conductance_ema)
    @test reinterpret(UInt32, state_forward.utility) ==
        reinterpret(UInt32, state_scrambled.utility)
    @test state_forward.reduced_batches == state_scrambled.reduced_batches == 1
end

function canonical_single_state_batch(; states=1, candidates=1, T=Float32)
    batch = Plasticity.CanonicalPlasticityBatch(
        1, 2, states, candidates; T=T,
    )
    Plasticity.begin_plasticity_batch!(batch)
    return batch
end

function publish_common_one!(batch, state_slot=1)
    Plasticity.record_state_common_plasticity!(
        batch, state_slot, UInt32[0], UInt8[1], Float32[0], Float32[0],
        Float32[0, 0], Float32[0, 0],
    )
    return batch
end

function publish_candidate_one!(
    batch,
    logical_slot=1,
    state_slot=1,
    ordinal=1,
)
    Plasticity.record_candidate_plasticity!(
        batch, logical_slot, state_slot, ordinal,
        UInt32[0], UInt8[1], Float32[0], Float32[0],
        Float32[0, 0], Float32[0, 0],
    )
    return batch
end

@testset "canonical utility follows only its explicit analog clock" begin
    config = Local.PlasticityConfig(
        firing_ema_decay=0.0,
        utility_decay=0.5,
        connection_cost=0.0,
    )
    offsets = Int[0, 1]
    batch = canonical_single_state_batch()
    publish_common_one!(batch)
    publish_candidate_one!(batch)
    state = Plasticity.PlasticityState(config, 1, 2)
    state.firing_rate[1] = 0.75f0
    state.utility .= Float32[0.25, 0.5]
    utility_bits = copy(reinterpret(UInt32, state.utility))
    utility_updates = state.utility_updates

    stats = Plasticity.preflight_canonical_plasticity(
        state, batch, config, offsets; utility_due=false,
    )
    @test stats.utility_nonzero == 0
    @test reinterpret(UInt32, state.utility) == utility_bits
    committed = Plasticity.reduce_canonical_plasticity!(
        state, batch, config, offsets; utility_due=false,
    )
    @test committed == stats
    @test reinterpret(UInt32, state.utility) == utility_bits
    @test state.utility_updates == utility_updates
    @test state.firing_rate[1] == 0.0f0
    @test state.reduced_batches == 1

    # `utility_due=false` is also the strict :none-mode contract on an analog
    # tick: persistent utility remains frozen while cell EMAs still publish.
    Plasticity.begin_plasticity_batch!(batch)
    publish_common_one!(batch)
    publish_candidate_one!(batch)
    Plasticity.reduce_canonical_plasticity!(
        state, batch, config, offsets; utility_due=false,
    )
    @test reinterpret(UInt32, state.utility) == utility_bits
    @test state.utility_updates == utility_updates
    @test state.reduced_batches == 2

    due_batch = canonical_single_state_batch()
    publish_common_one!(due_batch)
    publish_candidate_one!(due_batch)
    Plasticity.reduce_canonical_plasticity!(
        state, due_batch, config, offsets; utility_due=true,
    )
    @test state.utility == Float32[0.125, 0.25]
    @test state.utility_updates == utility_updates

    # A nonzero task-utility publication on an inactive clock is rejected
    # before any persistent EMA/counter mutation.
    invalid = canonical_single_state_batch()
    Plasticity.record_state_common_plasticity!(
        invalid, 1, UInt32[0], UInt8[1], Float32[0], Float32[0],
        Float32[1, 0], Float32[0, 0],
    )
    publish_candidate_one!(invalid)
    before = plasticity_state_bits(state)
    @test_throws ArgumentError Plasticity.reduce_canonical_plasticity!(
        state, invalid, config, offsets; utility_due=false,
    )
    @test plasticity_state_bits(state) == before
end

@testset "canonical publication cardinality and slot failures are atomic" begin
    config = Local.PlasticityConfig(
        firing_ema_decay=0.0,
        utility_decay=0.0,
        connection_cost=0.0,
    )

    duplicate_common = canonical_single_state_batch()
    publish_common_one!(duplicate_common)
    common_payload = copy(duplicate_common.common.activity_sum)
    @test_throws ArgumentError publish_common_one!(duplicate_common)
    @test duplicate_common.common.activity_sum == common_payload

    duplicate_candidate = canonical_single_state_batch()
    publish_common_one!(duplicate_candidate)
    publish_candidate_one!(duplicate_candidate)
    candidate_payload = copy(duplicate_candidate.candidate.activity_sum)
    @test_throws ArgumentError publish_candidate_one!(duplicate_candidate)
    @test duplicate_candidate.candidate.activity_sum == candidate_payload

    missing_common = canonical_single_state_batch()
    publish_candidate_one!(missing_common)
    missing_common_state = Plasticity.PlasticityState(config, 1, 2)
    missing_common_before = plasticity_state_bits(missing_common_state)
    @test_throws ArgumentError Plasticity.reduce_canonical_plasticity!(
        missing_common_state, missing_common, config, Int[0, 1],
    )
    @test plasticity_state_bits(missing_common_state) == missing_common_before

    missing_candidate = canonical_single_state_batch()
    publish_common_one!(missing_candidate)
    missing_candidate_state = Plasticity.PlasticityState(config, 1, 2)
    missing_candidate_before = plasticity_state_bits(missing_candidate_state)
    @test_throws ArgumentError Plasticity.reduce_canonical_plasticity!(
        missing_candidate_state, missing_candidate, config, Int[0, 1],
    )
    @test plasticity_state_bits(missing_candidate_state) ==
        missing_candidate_before

    wrong_slot = canonical_single_state_batch(candidates=2)
    publish_common_one!(wrong_slot)
    publish_candidate_one!(wrong_slot, 1, 1, 2)
    publish_candidate_one!(wrong_slot, 2, 1, 1)
    wrong_slot_state = Plasticity.PlasticityState(config, 1, 2)
    wrong_slot_before = plasticity_state_bits(wrong_slot_state)
    @test_throws ArgumentError Plasticity.reduce_canonical_plasticity!(
        wrong_slot_state, wrong_slot, config, Int[0, 2],
    )
    @test plasticity_state_bits(wrong_slot_state) == wrong_slot_before

    extra_common = canonical_single_state_batch(states=2)
    publish_common_one!(extra_common, 1)
    publish_common_one!(extra_common, 2)
    publish_candidate_one!(extra_common)
    extra_state = Plasticity.PlasticityState(config, 1, 2)
    extra_before = plasticity_state_bits(extra_state)
    @test_throws ArgumentError Plasticity.reduce_canonical_plasticity!(
        extra_state, extra_common, config, Int[0, 1],
    )
    @test plasticity_state_bits(extra_state) == extra_before

    stale = canonical_single_state_batch()
    publish_common_one!(stale)
    publish_candidate_one!(stale)
    Plasticity.begin_plasticity_batch!(stale)
    stale_state = Plasticity.PlasticityState(config, 1, 2)
    stale_before = plasticity_state_bits(stale_state)
    @test_throws ArgumentError Plasticity.reduce_canonical_plasticity!(
        stale_state, stale, config, Int[0, 1],
    )
    @test plasticity_state_bits(stale_state) == stale_before

    rollover = canonical_single_state_batch()
    rollover.generation = typemax(UInt32)
    fill!(rollover.common.stamp, typemax(UInt32))
    fill!(rollover.candidate.stamp, typemax(UInt32))
    Plasticity.begin_plasticity_batch!(rollover)
    @test rollover.generation == UInt32(1)
    @test all(iszero, rollover.common.stamp)
    @test all(iszero, rollover.candidate.stamp)
end

@testset "per-use utility survives signed cancellation; common delta cancels" begin
    config = Local.PlasticityConfig(
        firing_ema_decay=0.0,
        utility_decay=0.0,
        connection_cost=0.0,
    )
    batch = canonical_single_state_batch(candidates=2)
    # The common replay sees (+delta) + (-delta) == 0 and therefore publishes
    # zero task utility.  Candidate signed gradients may cancel in the actual
    # parameter gradient, while their chronological abs(M*e) utility remains.
    Plasticity.record_state_common_plasticity!(
        batch, 1, UInt32[0], UInt8[1], Float32[0], Float32[0],
        Float32[0, 0], Float32[0, 0],
    )
    Plasticity.record_candidate_plasticity!(
        batch, 1, 1, 1, UInt32[0], UInt8[1], Float32[0], Float32[0],
        Float32[2, 0], Float32[0, 0],
    )
    Plasticity.record_candidate_plasticity!(
        batch, 2, 1, 2, UInt32[0], UInt8[1], Float32[0], Float32[0],
        Float32[2, 0], Float32[0, 0],
    )
    state = Plasticity.PlasticityState(config, 1, 2)
    Plasticity.reduce_canonical_plasticity!(
        state, batch, config, Int[0, 2],
    )
    @test batch.common.utility_product_sum[:, 1] == Float32[0, 0]
    @test state.utility == Float32[4, 0]
    @test state.utility_updates == 1
end

@testset "canonical NaN, narrowing, and reduction overflow fail atomically" begin
    config = Local.PlasticityConfig(
        firing_ema_decay=0.0,
        utility_decay=0.0,
        connection_cost=0.0,
    )
    invalid = canonical_single_state_batch()
    invalid_common_before = copy(invalid.common.activity_sum)
    @test_throws DomainError Plasticity.record_state_common_plasticity!(
        invalid, 1, UInt32[0], UInt8[1], Float32[NaN], Float32[0],
        Float32[0, 0], Float32[0, 0],
    )
    @test invalid.common.activity_sum == invalid_common_before
    @test invalid.common.stamp[1] == UInt32(0)
    @test_throws DomainError Plasticity.record_candidate_plasticity!(
        invalid, 1, 1, 1, UInt32[0], UInt8[1], Float64[0], Float64[0],
        Float64[1.0e100, 0], Float64[0, 0],
    )
    @test invalid.candidate.stamp[1] == UInt32(0)
    @test_throws DomainError Plasticity.record_state_common_plasticity!(
        invalid, 1, UInt32[0], UInt8[9], Float32[0], Float32[0],
        Float32[0, 0], Float32[0, 0],
    )
    @test_throws DomainError Plasticity.record_state_common_plasticity!(
        invalid, 1, UInt32[2], UInt8[1], Float32[0], Float32[0],
        Float32[0, 0], Float32[0, 0],
    )

    corrupted = canonical_single_state_batch()
    publish_common_one!(corrupted)
    publish_candidate_one!(corrupted)
    corrupted.common.incoming_conductance_sum[1, 1] = NaN32
    corrupted_state = Plasticity.PlasticityState(config, 1, 2)
    corrupted_before = plasticity_state_bits(corrupted_state)
    @test_throws DomainError Plasticity.preflight_canonical_plasticity(
        corrupted_state, corrupted, config, Int[0, 1],
    )
    @test plasticity_state_bits(corrupted_state) == corrupted_before
    @test_throws DomainError Plasticity.reduce_canonical_plasticity!(
        corrupted_state, corrupted, config, Int[0, 1],
    )
    @test plasticity_state_bits(corrupted_state) == corrupted_before

    overflow = canonical_single_state_batch(T=Float64)
    Plasticity.record_state_common_plasticity!(
        overflow, 1, UInt32[0], UInt8[1], Float64[0], Float64[0],
        Float64[1, 1], Float64[0, 0],
    )
    Plasticity.record_candidate_plasticity!(
        overflow, 1, 1, 1, UInt32[0], UInt8[1], Float64[0], Float64[0],
        Float64[1, 1], Float64[0, 0],
    )
    overflow.common.utility_product_sum[1, 1] = 1.0e308
    overflow.candidate.utility_product_sum[1, 1] = 1.0e308
    overflow_state = Plasticity.PlasticityState(
        config, 1, 2; T=Float64,
    )
    overflow_before = plasticity_state_bits(overflow_state)
    @test_throws OverflowError Plasticity.preflight_canonical_plasticity(
        overflow_state, overflow, config, Int[0, 1],
    )
    @test plasticity_state_bits(overflow_state) == overflow_before
    @test_throws OverflowError Plasticity.reduce_canonical_plasticity!(
        overflow_state, overflow, config, Int[0, 1],
    )
    @test plasticity_state_bits(overflow_state) == overflow_before

    persistent = canonical_single_state_batch()
    publish_common_one!(persistent)
    publish_candidate_one!(persistent)
    persistent_state = Plasticity.PlasticityState(config, 1, 2)
    persistent_state.utility[2] = NaN32
    persistent_before = deepcopy(persistent_state)
    @test_throws DomainError Plasticity.reduce_canonical_plasticity!(
        persistent_state, persistent, config, Int[0, 1],
    )
    @test isequal(persistent_state.firing_rate, persistent_before.firing_rate)
    @test isequal(persistent_state.activity_ema, persistent_before.activity_ema)
    @test isequal(
        persistent_state.incoming_conductance_ema,
        persistent_before.incoming_conductance_ema,
    )
    @test isequal(persistent_state.utility, persistent_before.utility)
    @test persistent_state.reduced_batches == persistent_before.reduced_batches

    exhausted = canonical_single_state_batch()
    publish_common_one!(exhausted)
    publish_candidate_one!(exhausted)
    exhausted_state = Plasticity.PlasticityState(config, 1, 2)
    exhausted_state.reduced_batches = typemax(UInt64)
    exhausted_before = plasticity_state_bits(exhausted_state)
    @test_throws OverflowError Plasticity.preflight_canonical_plasticity(
        exhausted_state, exhausted, config, Int[0, 1],
    )
    @test plasticity_state_bits(exhausted_state) == exhausted_before
end

function canonical_allocation_probe()
    config = Local.PlasticityConfig(
        firing_ema_decay=0.5,
        utility_decay=0.5,
        connection_cost=0.1,
    )
    batch = Plasticity.CanonicalPlasticityBatch(2, 2, 2, 4)
    state = Plasticity.PlasticityState(config, 2, 2)
    offsets = Int[0, 2, 4]
    spikes = UInt32[1, 0]
    visits = UInt8[1, 0]
    activity = Float32[0.2, 0]
    incoming = Float32[0.4, 0]
    task_utility = Float32[0.25, 0]
    contact_activity = Float32[0.1, 0]

    # Compile every canonical path before measuring.
    Plasticity.begin_plasticity_batch!(batch)
    for state_slot in 1:2
        Plasticity.record_state_common_plasticity!(
            batch, state_slot, spikes, visits, activity, incoming,
            task_utility, contact_activity,
        )
    end
    for candidate in 1:4
        state_slot = candidate <= 2 ? 1 : 2
        ordinal = candidate <= 2 ? candidate : candidate - 2
        Plasticity.record_candidate_plasticity!(
            batch, candidate, state_slot, ordinal,
            spikes, visits, activity, incoming,
            task_utility, contact_activity,
        )
    end
    Plasticity.preflight_canonical_plasticity(state, batch, config, offsets)
    Plasticity.reduce_canonical_plasticity!(state, batch, config, offsets)

    begin_bytes = @allocated Plasticity.begin_plasticity_batch!(batch)
    common_bytes = @allocated begin
        for state_slot in 1:2
            Plasticity.record_state_common_plasticity!(
                batch, state_slot, spikes, visits, activity, incoming,
                task_utility, contact_activity,
            )
        end
    end
    candidate_bytes = @allocated begin
        for candidate in 1:4
            state_slot = candidate <= 2 ? 1 : 2
            ordinal = candidate <= 2 ? candidate : candidate - 2
            Plasticity.record_candidate_plasticity!(
                batch, candidate, state_slot, ordinal,
                spikes, visits, activity, incoming,
                task_utility, contact_activity,
            )
        end
    end
    preflight_bytes = @allocated Plasticity.preflight_canonical_plasticity(
        state, batch, config, offsets,
    )
    reduce_bytes = @allocated Plasticity.reduce_canonical_plasticity!(
        state, batch, config, offsets,
    )
    fill!(batch.common.utility_product_sum, 0.0f0)
    fill!(batch.candidate.utility_product_sum, 0.0f0)
    Plasticity.preflight_canonical_plasticity(
        state, batch, config, offsets; utility_due=false,
    )
    Plasticity.reduce_canonical_plasticity!(
        state, batch, config, offsets; utility_due=false,
    )
    preflight_inactive_bytes = @allocated begin
        Plasticity.preflight_canonical_plasticity(
            state, batch, config, offsets; utility_due=false,
        )
    end
    reduce_inactive_bytes = @allocated begin
        Plasticity.reduce_canonical_plasticity!(
            state, batch, config, offsets; utility_due=false,
        )
    end
    return (
        begin_bytes,
        common_bytes,
        candidate_bytes,
        preflight_bytes,
        reduce_bytes,
        preflight_inactive_bytes,
        reduce_inactive_bytes,
    )
end

@testset "canonical publication, preflight, and reduction allocate zero bytes" begin
    @test canonical_allocation_probe() == (0, 0, 0, 0, 0, 0, 0)
end

function allocation_probe()
    config = Local.PlasticityConfig(
        firing_ema_decay=0.5,
        utility_decay=0.5,
        connection_cost=0.0,
        target_rate_min=0.05,
        target_rate_max=0.25,
    )
    batch = Plasticity.CandidatePlasticityBatch(2, 2, 2)
    spikes = UInt32[0, 1]
    activity = Float32[0.2, 0.3]
    incoming = Float32[0.4, 0.5]
    third = Float32[1, -2]
    contribution = Float32[0.5, 0.25]
    contact_activity = Float32[0.1, 0.1]
    state = Plasticity.PlasticityState(config, 2, 2)
    destination = Int[1, 2]
    registry, optimizer = parameter_fixture(2, destination)
    reset = Plasticity.OptimizerMomentReset(optimizer, registry)

    Plasticity.begin_plasticity_batch!(batch)
    Plasticity.record_candidate_plasticity!(
        batch, 1, spikes, 2, activity, incoming,
        third, contribution, contact_activity,
    )
    Plasticity.reduce_candidate_plasticity!(state, batch, config, 1)
    Plasticity.apply_intrinsic_homeostasis!(
        state, config, true, registry.groups[1], reset,
    )
    Plasticity.apply_synaptic_scaling!(
        state, config, true, registry.groups[2], destination, reset,
    )

    begin_bytes = @allocated Plasticity.begin_plasticity_batch!(batch)
    record_bytes = @allocated Plasticity.record_candidate_plasticity!(
        batch, 1, spikes, 2, activity, incoming,
        third, contribution, contact_activity,
    )
    reduce_bytes = @allocated Plasticity.reduce_candidate_plasticity!(
        state, batch, config, 1,
    )
    state.firing_rate .= Float32[0, 1]
    homeostasis_bytes = @allocated Plasticity.apply_intrinsic_homeostasis!(
        state, config, true, registry.groups[1], reset,
    )
    scaling_bytes = @allocated Plasticity.apply_synaptic_scaling!(
        state, config, true, registry.groups[2], destination, reset,
    )
    return (
        begin_bytes,
        record_bytes,
        reduce_bytes,
        homeostasis_bytes,
        scaling_bytes,
    )
end

@testset "hot plasticity primitives allocate zero bytes" begin
    bytes = allocation_probe()
    @test bytes == (0, 0, 0, 0, 0)
end

function segmented_allocation_probe()
    config = Local.PlasticityConfig(
        target_rate_min=0.05,
        target_rate_max=0.25,
        threshold_homeostasis_step=0.01,
        adaptation_homeostasis_step=0.01,
        synaptic_scaling_rate=0.01,
    )
    cell_registry, _ = segmented_cell_fixture()
    conductance_registry, _, destinations =
        segmented_conductance_fixture()
    registry = Optimizer.ParameterRegistry(
        cell_registry.groups[1],
        conductance_registry.groups[1],
        conductance_registry.groups[2],
        cell_registry.groups[2],
        conductance_registry.groups[3],
    )
    optimizer = Optimizer.AdamWState(registry)
    for moments in optimizer.moments
        fill!(moments.first, 1.0f0)
        fill!(moments.second, 2.0f0)
    end
    state = Plasticity.PlasticityState(config, 1458, 0)
    fill!(state.firing_rate, 0.1f0)
    state.firing_rate[1] = 0.0f0
    state.firing_rate[2] = 1.0f0
    state.firing_rate[1437] = 1.0f0
    reset = Plasticity.OptimizerMomentReset(optimizer, registry)
    cell_groups = (registry.groups[1], registry.groups[4])
    conductance_groups = (
        registry.groups[2], registry.groups[3], registry.groups[5],
    )
    cell_ranges = (1:1436, 1437:1458)
    cell_offsets = (0, 1436)

    Plasticity.apply_intrinsic_homeostasis!(
        state,
        config,
        true,
        cell_groups,
        cell_ranges,
        reset,
    )
    Plasticity.apply_intrinsic_homeostasis!(
        state,
        config,
        true,
        cell_groups,
        cell_offsets,
        reset,
    )
    Plasticity.apply_synaptic_scaling!(
        state,
        config,
        true,
        conductance_groups,
        destinations,
        reset,
    )

    range_bytes = @allocated Plasticity.apply_intrinsic_homeostasis!(
        state,
        config,
        true,
        cell_groups,
        cell_ranges,
        reset,
    )
    offset_bytes = @allocated Plasticity.apply_intrinsic_homeostasis!(
        state,
        config,
        true,
        cell_groups,
        cell_offsets,
        reset,
    )
    conductance_bytes = @allocated Plasticity.apply_synaptic_scaling!(
        state,
        config,
        true,
        conductance_groups,
        destinations,
        reset,
    )
    return range_bytes, offset_bytes, conductance_bytes
end

@testset "segmented homeostasis and scaling allocate zero bytes" begin
    @test segmented_allocation_probe() == (0, 0, 0)
end
