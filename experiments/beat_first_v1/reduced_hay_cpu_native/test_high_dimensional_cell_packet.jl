using Test
using Random
using LinearAlgebra

module HighDimensionalCellPacketTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "HighDimensionalCellPacket.jl"))
end

const H = HighDimensionalCellPacketTestHarness
const Cell = H.ActiveApicalCell
const Packet = H.HighDimensionalCellPacket

function recorded_transition(::Type{T}=Float64) where {T<:AbstractFloat}
    raw = T.(Cell.default_raw_parameters())
    cache, derivative_cache = Cell.parameter_caches(raw)
    previous = T.(Cell.initial_state(cache))
    previous[Cell.SOMA_INDEX] = T(-58)
    # Stay away from the intrinsic `max(adaptation, 0)` kink so the
    # conditional finite-difference oracle has a unique derivative.
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
    Packet.cell_packet!(packet, previous, next, cache)
    return dot(packet, packet_bar), next
end

function central_difference(values, index, objective; epsilon=1.0e-6)
    plus = copy(values)
    minus = copy(values)
    plus[index] += epsilon
    minus[index] -= epsilon
    return (objective(plus) - objective(minus)) / (2epsilon)
end

@testset "47D high-dimensional packet contract" begin
    @test Packet.PACKET_DIM == 47
    @test Packet.PACKET_BYTES == 188
    @test Packet.VOLTAGE_LANE_FIRST == 1
    @test Packet.AMPA_LANE_FIRST == 10
    @test Packet.NMDA_LANE_FIRST == 19
    @test Packet.GABA_LANE_FIRST == 28
    @test Packet.PLATEAU_LANE_FIRST == 37
    @test Packet.MARGIN_LANE == 46
    @test Packet.ADAPTATION_LANE == 47

    raw, cache, _, previous, _, next = recorded_transition(Float32)
    packet = zeros(Float32, Packet.PACKET_DIM)
    @test Packet.cell_packet!(packet, previous, next, cache) === packet
    @test all(isfinite, packet)
    @test all(value -> -1.0f0 < value < 1.0f0, packet)

    covered = falses(Packet.PACKET_DIM)
    @inbounds for field in 1:Cell.COMPARTMENT_STATE_DIM
        for compartment in 1:Cell.N_COMPARTMENTS
            lane = Packet.packet_lane(compartment, field)
            @test !covered[lane]
            covered[lane] = true
        end
    end
    covered[Packet.MARGIN_LANE] = true
    covered[Packet.ADAPTATION_LANE] = true
    @test all(covered)
    @test raw isa Vector{Float32}

    @test_throws DimensionMismatch Packet.cell_packet!(
        zeros(Float32, 46), previous, next, cache,
    )
    @test_throws DimensionMismatch Packet.cell_packet!(
        packet, zeros(Float32, Cell.STATE_DIM - 1), next, cache,
    )
    @test_throws BoundsError Packet.packet_lane(0, Cell.FIELD_VOLTAGE)
    @test_throws BoundsError Packet.packet_lane(1, Cell.COMPARTMENT_STATE_DIM + 1)
end

@testset "every physical field and apical state remains visible" begin
    _, cache, _, previous, _, next = recorded_transition(Float64)
    baseline = zeros(Float64, Packet.PACKET_DIM)
    changed = similar(baseline)
    Packet.cell_packet!(baseline, previous, next, cache)

    probes = (
        (Cell.N_COMPARTMENTS, Cell.FIELD_VOLTAGE, 0.2),
        (3, Cell.FIELD_AMPA, 0.001),
        (4, Cell.FIELD_NMDA, 0.001),
        (5, Cell.FIELD_GABA, 0.001),
        (6, Cell.FIELD_PLATEAU, 0.001),
    )
    for (compartment, field, amount) in probes
        perturbed = copy(next)
        state = Cell.state_index(compartment, field)
        perturbed[state] += amount
        Packet.cell_packet!(changed, previous, perturbed, cache)
        lane = Packet.packet_lane(compartment, field)
        @test changed[lane] != baseline[lane]
        @test count(!iszero, changed .- baseline) >= 1
    end

    adaptation = copy(next)
    adaptation[Cell.ADAPTATION_INDEX] += 0.02
    Packet.cell_packet!(changed, previous, adaptation, cache)
    @test changed[Packet.ADAPTATION_LANE] != baseline[Packet.ADAPTATION_LANE]

    # Margin is reconstructed before soma reset: modifying previous soma
    # changes lane 46 even though no post-reset soma coordinate is exported as
    # a substitute.
    previous_margin = copy(previous)
    previous_margin[Cell.SOMA_INDEX] += 0.2
    Packet.cell_packet!(changed, previous_margin, next, cache)
    @test changed[Packet.MARGIN_LANE] != baseline[Packet.MARGIN_LANE]
end

@testset "exact packet plus transition pullback" begin
    rng = MersenneTwister(0x47c0ffee)
    raw, cache, derivative_cache, previous, input, next =
        recorded_transition(Float64)
    packet_bar = randn(rng, Packet.PACKET_DIM)
    dnext = zeros(Float64, Cell.STATE_DIM)
    margin_bar = Packet.cell_packet_pullback!(
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

    objective_previous(x) = first(packet_objective(x, input, raw, packet_bar))
    objective_input(x) = first(packet_objective(previous, x, raw, packet_bar))
    objective_raw(x) = first(packet_objective(previous, input, x, packet_bar))
    epsilon = 1.0e-6
    for index in (1, Cell.state_index(9, Cell.FIELD_VOLTAGE), Cell.SOMA_INDEX,
                  Cell.ADAPTATION_INDEX)
        numerical = central_difference(
            previous, index, objective_previous; epsilon=epsilon,
        )
        @test isapprox(dprevious[index], numerical; rtol=8.0e-4, atol=2.0e-6)
    end
    for index in (Cell.input_index(1, Cell.INPUT_AMPA),
                  Cell.input_index(5, Cell.INPUT_NMDA),
                  Cell.input_index(9, Cell.INPUT_GABA))
        numerical = central_difference(input, index, objective_input; epsilon=epsilon)
        @test isapprox(dinput[index], numerical; rtol=8.0e-4, atol=2.0e-6)
    end
    for index in (1, 7, 20, Cell.PARAM_DIM)
        numerical = central_difference(raw, index, objective_raw; epsilon=epsilon)
        @test isapprox(draw[index], numerical; rtol=1.5e-3, atol=3.0e-6)
    end
    @test dnext[Cell.SPIKE_INDEX] == 0.0
    @test isfinite(margin_bar)
end

@testset "SoA equivalence and allocation-free hot path" begin
    _, cache, _, previous, _, next = recorded_transition(Float32)
    cell_count = 3
    selected = 2
    states = zeros(Float32, Cell.STATE_DIM, cell_count, 2)
    states[:, selected, 1] .= previous
    states[:, selected, 2] .= next
    vector_packet = zeros(Float32, Packet.PACKET_DIM)
    matrix_packet = fill(17.0f0, Packet.PACKET_DIM, cell_count)
    Packet.cell_packet!(vector_packet, previous, next, cache)
    Packet.cell_packet_column!(matrix_packet, states, selected, 1, 2, cache)
    @test matrix_packet[:, selected] == vector_packet
    @test all(==(17.0f0), matrix_packet[:, 1])
    @test all(==(17.0f0), matrix_packet[:, 3])

    packet_bar = Float32.(range(-0.8, 0.7; length=Packet.PACKET_DIM))
    packet_bar_matrix = zeros(Float32, Packet.PACKET_DIM, cell_count)
    packet_bar_matrix[:, selected] .= packet_bar
    dvector = zeros(Float32, Cell.STATE_DIM)
    dmatrix = zeros(Float32, Cell.STATE_DIM)
    margin_vector = Packet.cell_packet_pullback!(
        dvector, packet_bar, previous, next, cache,
    )
    margin_matrix = Packet.cell_packet_column_pullback!(
        dmatrix, packet_bar_matrix, states, selected, 1, 2, cache,
    )
    @test dmatrix == dvector
    @test margin_matrix == margin_vector

    Packet.cell_packet!(vector_packet, previous, next, cache)
    Packet.cell_packet_column!(matrix_packet, states, selected, 1, 2, cache)
    Packet.cell_packet_pullback!(dvector, packet_bar, previous, next, cache)
    Packet.cell_packet_column_pullback!(
        dmatrix, packet_bar_matrix, states, selected, 1, 2, cache,
    )
    @test @allocated(Packet.cell_packet!(
        vector_packet, previous, next, cache,
    )) == 0
    @test @allocated(Packet.cell_packet_pullback!(
        dvector, packet_bar, previous, next, cache,
    )) == 0
    @test @allocated(Packet.cell_packet_column!(
        matrix_packet, states, selected, 1, 2, cache,
    )) == 0
    @test @allocated(Packet.cell_packet_column_pullback!(
        dmatrix, packet_bar_matrix, states, selected, 1, 2, cache,
    )) == 0
end
