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

mutable struct ResetCounter
    count::Int
    last_linear::Int
end

@inline function (counter::ResetCounter)(::Symbol, index)
    counter.count += 1
    counter.last_linear = index isa CartesianIndex ? index[1] : Int(index)
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
    registry, _ = parameter_fixture(2, destination)
    reset = ResetCounter(0, 0)

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
