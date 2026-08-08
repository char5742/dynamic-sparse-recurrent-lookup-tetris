using Test
using Random
using LinearAlgebra

module DendriticAxonPacketTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "DendriticAxonPacket.jl"))
end

const H = DendriticAxonPacketTestHarness
const Cell = H.ActiveApicalCell
const Axon = H.DendriticAxonPacket

function recorded_transition(::Type{T}=Float64) where {T<:AbstractFloat}
    raw = T.(Cell.default_raw_parameters())
    cache, derivative_cache = Cell.parameter_caches(raw)
    previous = T.(Cell.initial_state(cache))
    previous[Cell.SOMA_INDEX] = T(-58)
    previous[Cell.ADAPTATION_INDEX] = T(0.03)
    input = zeros(T, Cell.INPUT_DIM)
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        input[Cell.input_index(compartment, Cell.INPUT_AMPA)] =
            T(0.006 + 0.0004 * compartment)
        input[Cell.input_index(compartment, Cell.INPUT_NMDA)] =
            T(0.004 + 0.0003 * compartment)
        input[Cell.input_index(compartment, Cell.INPUT_GABA)] =
            T(0.002 + 0.0002 * compartment)
    end
    next = Cell.cell_step_cached_functional(previous, input, cache)
    return raw, cache, derivative_cache, previous, input, next
end

function packet_objective(previous, input, raw, packet_bar)
    cache = Cell.transform_parameters(raw)
    next = Cell.cell_step_cached_functional(previous, input, cache)
    packet = similar(packet_bar)
    events = zeros(UInt8, Axon.EVENT_DIM)
    Axon.axon_packet!(packet, previous, next, cache)
    Axon.hard_events!(events, previous, next)
    return dot(packet, packet_bar), events
end

function stable_central_difference(values, index, objective; epsilon=1.0e-6)
    plus = copy(values)
    minus = copy(values)
    plus[index] += epsilon
    minus[index] -= epsilon
    plus_value, plus_events = objective(plus)
    minus_value, minus_events = objective(minus)
    @test plus_events == minus_events
    return (plus_value - minus_value) / (2epsilon)
end

@testset "12D dendritic axon contract" begin
    @test Axon.GROUP_COUNT == 4
    @test Axon.FIELD_COUNT == 3
    @test Axon.PACKET_DIM == 12
    @test Axon.PACKET_BYTES == 48
    @test Axon.EVENT_DIM == 5
    @test Cell.INPUT_DIM == 2 * Axon.PACKET_DIM + Cell.INPUT_CHANNELS
    @test Axon.packet_lane(1, Axon.FAST_FIELD) == 1
    @test Axon.packet_lane(4, Axon.INHIBITORY_FIELD) == 12
    @test Axon.plateau_event_lane(1) == 2
    @test Axon.plateau_event_lane(4) == 5

    _, cache, _, previous, _, next = recorded_transition(Float32)
    packet = zeros(Float32, Axon.PACKET_DIM)
    @test Axon.axon_packet!(packet, previous, next, cache) === packet
    @test all(isfinite, packet)
    @test all(value -> 0.0f0 <= value < 1.0f0, packet)
    @test all(group ->
        packet[Axon.packet_lane(group, Axon.FAST_FIELD)] >= 0.0f0 &&
        packet[Axon.packet_lane(group, Axon.NMDA_FIELD)] >= 0.0f0 &&
        packet[Axon.packet_lane(group, Axon.INHIBITORY_FIELD)] >= 0.0f0,
        1:Axon.GROUP_COUNT,
    )

    @test_throws BoundsError Axon.packet_lane(0, 1)
    @test_throws BoundsError Axon.packet_lane(1, 4)
    @test_throws BoundsError Axon.plateau_event_lane(5)
    @test_throws DimensionMismatch Axon.axon_packet!(
        zeros(Float32, 11), previous, next, cache,
    )
    @test_throws DimensionMismatch Axon.axon_packet!(
        packet, zeros(Float32, Cell.STATE_DIM - 1), next, cache,
    )
end

@testset "all required Reduced-Hay evidence reaches the packet" begin
    _, cache, _, previous, _, next = recorded_transition(Float64)
    baseline = zeros(Float64, Axon.PACKET_DIM)
    changed = similar(baseline)
    Axon.axon_packet!(baseline, previous, next, cache)

    probes = (
        (1, Cell.FIELD_AMPA, 0.002, 1, Axon.FAST_FIELD),
        (3, Cell.FIELD_NMDA, 0.002, 2, Axon.NMDA_FIELD),
        (5, Cell.FIELD_PLATEAU, 0.002, 3, Axon.NMDA_FIELD),
        (7, Cell.FIELD_GABA, 0.002, 4, Axon.INHIBITORY_FIELD),
    )
    for (compartment, field, amount, group, packet_field) in probes
        perturbed = copy(next)
        perturbed[Cell.state_index(compartment, field)] += amount
        Axon.axon_packet!(changed, previous, perturbed, cache)
        @test changed[Axon.packet_lane(group, packet_field)] !=
              baseline[Axon.packet_lane(group, packet_field)]
    end

    voltage = copy(next)
    voltage[Cell.state_index(2, Cell.FIELD_VOLTAGE)] += 0.2
    Axon.axon_packet!(changed, previous, voltage, cache)
    @test changed[Axon.packet_lane(1, Axon.FAST_FIELD)] !=
          baseline[Axon.packet_lane(1, Axon.FAST_FIELD)]
    @test changed[Axon.packet_lane(1, Axon.NMDA_FIELD)] !=
          baseline[Axon.packet_lane(1, Axon.NMDA_FIELD)]

    apical = copy(next)
    apical[Cell.state_index(Cell.N_COMPARTMENTS, Cell.FIELD_NMDA)] += 0.002
    Axon.axon_packet!(changed, previous, apical, cache)
    @test any(changed .!= baseline)
    @test all(group ->
        changed[Axon.packet_lane(group, Axon.NMDA_FIELD)] !=
        baseline[Axon.packet_lane(group, Axon.NMDA_FIELD)],
        1:Axon.GROUP_COUNT,
    )

    adaptation = copy(next)
    adaptation[Cell.ADAPTATION_INDEX] += 0.02
    Axon.axon_packet!(changed, previous, adaptation, cache)
    @test all(group ->
        changed[Axon.packet_lane(group, Axon.INHIBITORY_FIELD)] !=
        baseline[Axon.packet_lane(group, Axon.INHIBITORY_FIELD)],
        1:Axon.GROUP_COUNT,
    )

    previous_margin = copy(previous)
    previous_margin[Cell.SOMA_INDEX] += 0.2
    Axon.axon_packet!(changed, previous_margin, next, cache)
    @test any(changed .!= baseline)
end

@testset "hard soma and plateau-group event plane" begin
    _, _, _, previous, _, next = recorded_transition(Float32)
    previous .= 0.0f0
    next .= 0.0f0
    next[Cell.SPIKE_INDEX] = 1.0f0
    threshold = Axon.PLATEAU_EVENT_THRESHOLD
    # onset in group 1, offset in group 2, stable active group 3 and stable
    # inactive group 4.
    next[Cell.state_index(1, Cell.FIELD_PLATEAU)] = threshold * 2
    previous[Cell.state_index(3, Cell.FIELD_PLATEAU)] = threshold * 2
    previous[Cell.state_index(5, Cell.FIELD_PLATEAU)] = threshold * 2
    next[Cell.state_index(6, Cell.FIELD_PLATEAU)] = threshold * 2
    events = fill(UInt8(7), Axon.EVENT_DIM)
    @test Axon.hard_events!(events, previous, next) === events
    @test events == UInt8[1, 1, 1, 0, 0]

    bool_events = falses(Axon.EVENT_DIM)
    Axon.hard_events!(bool_events, previous, next)
    @test bool_events == Bool[1, 1, 1, 0, 0]
    @test_throws DimensionMismatch Axon.hard_events!(
        zeros(UInt8, Axon.EVENT_DIM - 1), previous, next,
    )
end

@testset "exact event-stable packet plus transition pullback" begin
    rng = MersenneTwister(0x12a0c0de)
    raw, cache, derivative_cache, previous, input, next =
        recorded_transition(Float64)
    packet_bar = randn(rng, Axon.PACKET_DIM)
    dnext = zeros(Float64, Cell.STATE_DIM)
    margin_bar = Axon.axon_packet_pullback!(
        dnext, packet_bar, previous, next, cache,
    )
    dprevious = zeros(Float64, Cell.STATE_DIM)
    dinput = zeros(Float64, Cell.INPUT_DIM)
    draw = zeros(Float64, Cell.PARAM_DIM)
    Cell.cell_step_conditional_pullback!(
        dprevious,
        dinput,
        draw,
        previous,
        input,
        cache,
        derivative_cache,
        next,
        dnext,
        0.0,
        0.0,
        margin_bar,
    )

    objective_previous(x) = packet_objective(x, input, raw, packet_bar)
    objective_input(x) = packet_objective(previous, x, raw, packet_bar)
    objective_raw(x) = packet_objective(previous, input, x, packet_bar)
    epsilon = 1.0e-6
    for index in (
        1,
        Cell.state_index(Cell.N_COMPARTMENTS, Cell.FIELD_VOLTAGE),
        Cell.SOMA_INDEX,
        Cell.ADAPTATION_INDEX,
    )
        numerical = stable_central_difference(
            previous, index, objective_previous; epsilon=epsilon,
        )
        @test isapprox(dprevious[index], numerical; rtol=1.5e-3, atol=3.0e-6)
    end
    for index in (
        Cell.input_index(1, Cell.INPUT_AMPA),
        Cell.input_index(5, Cell.INPUT_NMDA),
        Cell.input_index(9, Cell.INPUT_GABA),
    )
        numerical = stable_central_difference(
            input, index, objective_input; epsilon=epsilon,
        )
        @test isapprox(dinput[index], numerical; rtol=1.5e-3, atol=3.0e-6)
    end
    for index in (1, 7, 20, Cell.PARAM_DIM)
        numerical = stable_central_difference(
            raw, index, objective_raw; epsilon=epsilon,
        )
        @test isapprox(draw[index], numerical; rtol=2.5e-3, atol=5.0e-6)
    end
    @test dnext[Cell.SPIKE_INDEX] == 0.0
    @test isfinite(margin_bar)
end

@testset "ordered binary deposit preserves child and receptor identity" begin
    left = Float32.(1:Axon.PACKET_DIM) ./ 32.0f0
    right = Float32.(101:(100 + Axon.PACKET_DIM)) ./ 256.0f0
    context = Float32[0.125, 0.25, 0.375]
    destination = zeros(Float32, Cell.INPUT_DIM)
    @test Axon.ordered_binary_deposit!(
        destination, left, right, context,
    ) === destination
    @inbounds for group in 1:Axon.GROUP_COUNT, field in 1:Axon.FIELD_COUNT
        @test destination[Cell.input_index(group, field)] ==
              left[Axon.packet_lane(group, field)]
        @test destination[Cell.input_index(Axon.GROUP_COUNT + group, field)] ==
              right[Axon.packet_lane(group, field)]
    end
    @inbounds for channel in 1:Cell.INPUT_CHANNELS
        @test destination[Cell.input_index(Cell.N_COMPARTMENTS, channel)] ==
              context[channel]
    end

    swapped = zeros(Float32, Cell.INPUT_DIM)
    Axon.ordered_binary_deposit!(swapped, right, left, context)
    @test swapped != destination

    destination_bar = Float32.(range(-0.4, 0.7; length=Cell.INPUT_DIM))
    left_bar = fill(0.25f0, Axon.PACKET_DIM)
    right_bar = fill(-0.5f0, Axon.PACKET_DIM)
    context_bar = fill(0.75f0, Cell.INPUT_CHANNELS)
    Axon.ordered_binary_deposit_pullback!(
        left_bar, right_bar, context_bar, destination_bar,
    )
    @inbounds for group in 1:Axon.GROUP_COUNT, field in 1:Axon.FIELD_COUNT
        lane = Axon.packet_lane(group, field)
        @test left_bar[lane] == 0.25f0 +
              destination_bar[Cell.input_index(group, field)]
        @test right_bar[lane] == -0.5f0 +
              destination_bar[Cell.input_index(Axon.GROUP_COUNT + group, field)]
    end
    @inbounds for channel in 1:Cell.INPUT_CHANNELS
        @test context_bar[channel] == 0.75f0 +
              destination_bar[Cell.input_index(Cell.N_COMPARTMENTS, channel)]
    end
end

@testset "finite bounds and allocation-free hot kernels" begin
    _, cache, _, previous, _, next = recorded_transition(Float32)
    extreme = copy(next)
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        extreme[Cell.state_index(compartment, Cell.FIELD_VOLTAGE)] =
            isodd(compartment) ? -1.0f4 : 1.0f4
        extreme[Cell.state_index(compartment, Cell.FIELD_AMPA)] = 1.0f3
        extreme[Cell.state_index(compartment, Cell.FIELD_NMDA)] = 1.0f3
        extreme[Cell.state_index(compartment, Cell.FIELD_GABA)] = 1.0f3
        extreme[Cell.state_index(compartment, Cell.FIELD_PLATEAU)] = 1.0f3
    end
    extreme[Cell.ADAPTATION_INDEX] = 1.0f3
    packet = zeros(Float32, Axon.PACKET_DIM)
    Axon.axon_packet!(packet, previous, extreme, cache)
    @test all(isfinite, packet)
    @test all(value -> 0.0f0 <= value < 1.0f0, packet)

    packet_bar = Float32.(range(-0.5, 0.5; length=Axon.PACKET_DIM))
    dnext = zeros(Float32, Cell.STATE_DIM)
    events = zeros(UInt8, Axon.EVENT_DIM)
    left = copy(packet)
    right = reverse(copy(packet))
    context = Float32[0.1, 0.2, 0.3]
    destination = zeros(Float32, Cell.INPUT_DIM)
    destination_bar = Float32.(range(-0.3, 0.4; length=Cell.INPUT_DIM))
    left_bar = zeros(Float32, Axon.PACKET_DIM)
    right_bar = zeros(Float32, Axon.PACKET_DIM)
    context_bar = zeros(Float32, Cell.INPUT_CHANNELS)

    Axon.axon_packet!(packet, previous, next, cache)
    Axon.axon_packet_pullback!(dnext, packet_bar, previous, next, cache)
    Axon.hard_events!(events, previous, next)
    Axon.ordered_binary_deposit!(destination, left, right, context)
    Axon.ordered_binary_deposit_pullback!(
        left_bar, right_bar, context_bar, destination_bar,
    )
    @test @allocated(Axon.axon_packet!(
        packet, previous, next, cache,
    )) == 0
    @test @allocated(Axon.axon_packet_pullback!(
        dnext, packet_bar, previous, next, cache,
    )) == 0
    @test @allocated(Axon.hard_events!(events, previous, next)) == 0
    @test @allocated(Axon.ordered_binary_deposit!(
        destination, left, right, context,
    )) == 0
    @test @allocated(Axon.ordered_binary_deposit_pullback!(
        left_bar, right_bar, context_bar, destination_bar,
    )) == 0
end
