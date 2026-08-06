using Test
using Random
using LinearAlgebra

module StructuredMotifReadoutTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "HighDimensionalCellPacket.jl"))
include(joinpath(@__DIR__, "StructuredMotifReadout.jl"))
end

const H = StructuredMotifReadoutTestHarness
const Cell = H.ActiveApicalCell
const Packet = H.HighDimensionalCellPacket
const Readout = H.StructuredMotifReadout

function _objective(
    packet::Matrix{T},
    parameters::Readout.StructuredReadoutParameters{T},
    direction::Matrix{T},
    scale::T;
    sources=nothing,
) where {T<:AbstractFloat}
    cache = Readout.StructuredReadoutCache(parameters)
    destination = zeros(T, Cell.INPUT_DIM, Readout.OUTPUT_COUNT)
    if sources === nothing
        Readout.deposit_readout!(destination, packet, cache, scale)
    else
        Readout.deposit_readout_selected!(
            destination,
            packet,
            cache,
            sources,
            scale,
        )
    end
    return dot(destination, direction)
end

@testset "structured motif readout contract" begin
    @test Readout.SOURCE_COUNT == 48
    @test Readout.FAMILY_COUNT == 4
    @test Readout.OUTPUT_COUNT == 22
    @test Readout.RELATION_RESIDUAL_SCALE == 0.125f0

    @test Readout.family_index.(1:48) == vcat(
        fill(1, 24),
        fill(2, 10),
        fill(3, 8),
        fill(4, 6),
    )
    @test Readout.positive_branch.(1:4) == [1, 3, 5, 7]
    @test Readout.negative_branch.(1:4) == [2, 4, 6, 8]

    @test Readout.output_field_count(1) == 47
    @test Int.([Readout.output_field(1, slot) for slot in 1:47]) ==
          collect(1:47)
    @test Readout.output_field_count(2) == 24
    death = Int.([
        Readout.output_field(2, slot) for slot in 1:24
    ])
    @test Packet.MARGIN_LANE in death
    @test Packet.ADAPTATION_LANE in death
    @test all(group -> any(field -> field in group, death),
              (1:9, 10:18, 19:27, 28:36, 37:45))
    quantile_fields = Set(Int(Readout.output_field(output, slot))
                          for output in 3:18 for slot in 1:8)
    @test quantile_fields == Set(1:47)
    for output in 19:22
        fields = Int.([Readout.output_field(output, slot) for slot in 1:16])
        @test length(unique(fields)) == 16
        @test Packet.MARGIN_LANE in fields
        @test Packet.ADAPTATION_LANE in fields
        @test all(group -> any(field -> field in group, fields),
                  (1:9, 10:18, 19:27, 28:36, 37:45))
    end

    @test Readout.fixed_weight(Float64, 1, 1) == inv(sqrt(24.0 * 47.0))
    @test Readout.fixed_weight(Float64, 2, 2) == inv(sqrt(10.0 * 24.0))
    @test Readout.fixed_weight(Float64, 3, 3) == inv(sqrt(8.0 * 8.0))
    @test Readout.fixed_weight(Float64, 4, 22) == inv(sqrt(6.0 * 16.0))

    parameters = Readout.initialize_parameters(Float64)
    @test size(parameters.source_gain_raw) == (48, 22)
    @test Readout.stored_parameter_count(parameters) == 1056
    cache = Readout.StructuredReadoutCache(parameters)
    @test cache.source_gain ≈ ones(Float64, 48, 22) atol=2e-16
    @test all(cache.source_gain .>= 0.25)
    @test all(cache.source_gain .<= 2.0)
    @test all(cache.source_gain_derivative .> 0)

    parameters.source_gain_raw[1, 1] = 100.0
    parameters.source_gain_raw[2, 1] = -100.0
    Readout.refresh_cache!(cache, parameters)
    @test cache.source_gain[1, 1] ≈ 2.0
    @test cache.source_gain[2, 1] ≈ 0.25

    gradient = Readout.StructuredReadoutGradient(Float64)
    @test size(gradient.source_gain_raw) == (48, 22)
    gradient.source_gain_raw .= 2
    Readout.clear_gradient!(gradient)
    @test iszero(sum(abs, gradient.source_gain_raw))
    source = Readout.StructuredReadoutGradient(Float64)
    source.source_gain_raw .= 3
    Readout.accumulate_gradient!(gradient, source)
    @test gradient.source_gain_raw == fill(3.0, 48, 22)

    @test_throws DimensionMismatch Readout.StructuredReadoutParameters(
        zeros(Float32, 3, 22),
    )
    @test_throws BoundsError Readout.family_index(0)
    @test_throws BoundsError Readout.family_index(49)
    @test_throws BoundsError Readout.output_field(2, 25)
end

@testset "compartment identity and physical receptor type are retained" begin
    parameters = Readout.initialize_parameters(Float64)
    cache = Readout.StructuredReadoutCache(parameters)
    packet = zeros(Float64, Packet.PACKET_DIM, Readout.SOURCE_COUNT)
    destination = zeros(Float64, Cell.INPUT_DIM, Readout.OUTPUT_COUNT)

    # Q reads every lane.  A signed voltage uses AMPA/GABA at the same rotated
    # physical compartment, not an unrelated lane-modulo receptor.
    packet[1, 1] = 2.0
    Readout.deposit_readout_selected!(destination, packet, cache, [1], 1.0)
    weight = inv(sqrt(24.0 * 47.0))
    @test destination[Cell.input_index(1, Cell.INPUT_AMPA), 1] == 2.0 * weight
    @test iszero(destination[Cell.input_index(1, Cell.INPUT_GABA), 1])

    fill!(destination, 0.0)
    packet[1, 1] = -2.0
    Readout.deposit_readout_selected!(destination, packet, cache, [1], 1.0)
    @test iszero(destination[Cell.input_index(1, Cell.INPUT_AMPA), 1])
    @test destination[Cell.input_index(1, Cell.INPUT_GABA), 1] == 2.0 * weight

    # Conductance and plateau identity determines the receptor exactly.
    fill!(destination, 0.0)
    fill!(packet, 0.0)
    packet[Packet.packet_lane(5, Cell.FIELD_NMDA), 1] = 3.0
    packet[Packet.packet_lane(6, Cell.FIELD_GABA), 1] = 4.0
    packet[Packet.packet_lane(7, Cell.FIELD_PLATEAU), 1] = 5.0
    Readout.deposit_readout_selected!(destination, packet, cache, [1], 1.0)
    @test destination[Cell.input_index(5, Cell.INPUT_NMDA), 1] == 3.0 * weight
    @test destination[Cell.input_index(6, Cell.INPUT_GABA), 1] == 4.0 * weight
    @test destination[Cell.input_index(7, Cell.INPUT_NMDA), 1] == 5.0 * weight

    # Apical fields stay apical for every semantic family.
    fill!(destination, 0.0)
    fill!(packet, 0.0)
    apical_voltage = Packet.packet_lane(9, Cell.FIELD_VOLTAGE)
    packet[apical_voltage, 25] = 1.0
    packet[Packet.packet_lane(9, Cell.FIELD_NMDA), 35] = 1.0
    packet[Packet.ADAPTATION_LANE, 43] = 1.0
    Readout.deposit_readout_selected!(
        destination,
        packet,
        cache,
        [25, 35, 43],
        1.0,
    )
    @test destination[Cell.input_index(9, Cell.INPUT_AMPA), 1] ==
          inv(sqrt(10.0 * 47.0))
    @test destination[Cell.input_index(9, Cell.INPUT_NMDA), 1] ==
          inv(sqrt(8.0 * 47.0))
    @test destination[Cell.input_index(9, Cell.INPUT_GABA), 1] ==
          inv(sqrt(6.0 * 47.0))
end

@testset "source binding forbids within-family permutation invariance" begin
    parameters = Readout.initialize_parameters(Float64)
    # Sources 1 and 2 are both rows, but they own different positive gains.
    parameters.source_gain_raw[1, 3] = -2.0
    parameters.source_gain_raw[2, 3] = 2.0
    cache = Readout.StructuredReadoutCache(parameters)
    packet = zeros(Float64, Packet.PACKET_DIM, Readout.SOURCE_COUNT)
    packet[1, 1] = 0.4
    packet[1, 2] = 1.2
    first = zeros(Float64, Cell.INPUT_DIM, Readout.OUTPUT_COUNT)
    swapped = zeros(Float64, size(first))
    Readout.deposit_readout!(first, packet, cache, 1.0)
    packet[1, 1], packet[1, 2] = packet[1, 2], packet[1, 1]
    Readout.deposit_readout!(swapped, packet, cache, 1.0)
    @test first != swapped
    @test first[Cell.input_index(1, 1), 3] !=
          swapped[Cell.input_index(1, 1), 3]

    # Credit is source specific even though both contacts share family anatomy.
    direction = zeros(Float64, Cell.INPUT_DIM, Readout.OUTPUT_COUNT)
    direction[Cell.input_index(1, 1), 3] = 1.0
    packet_bar = zeros(Float64, size(packet))
    gradient = Readout.StructuredReadoutGradient(Float64)
    Readout.deposit_readout_selected_pullback!(
        packet_bar,
        gradient,
        packet,
        cache,
        [1],
        direction,
        1.0,
    )
    @test !iszero(gradient.source_gain_raw[1, 3])
    @test iszero(gradient.source_gain_raw[2, 3])
end

@testset "full selected and scale semantics" begin
    rng = MersenneTwister(0x5e6d_4e71)
    parameters = Readout.initialize_parameters(Float32)
    parameters.source_gain_raw .= randn(rng, Float32, 48, 22)
    cache = Readout.StructuredReadoutCache(parameters)
    packet = randn(rng, Float32, Packet.PACKET_DIM, Readout.SOURCE_COUNT)
    packet[abs.(packet) .< 0.1f0] .+= 0.25f0

    full = zeros(Float32, Cell.INPUT_DIM, 22)
    selected = zeros(Float32, size(full))
    Readout.deposit_readout!(full, packet, cache, 1.0f0)
    all_sources = collect(Int, 1:Readout.SOURCE_COUNT)
    Readout.deposit_readout_selected!(
        selected,
        packet,
        cache,
        all_sources,
        1.0f0,
    )
    @test selected == full

    partitioned = zeros(Float32, size(full))
    Readout.deposit_readout_selected!(
        partitioned,
        packet,
        cache,
        collect(Int, 1:24),
        1.0f0,
    )
    Readout.deposit_readout_selected!(
        partitioned,
        packet,
        cache,
        collect(Int, 25:48),
        1.0f0,
    )
    @test partitioned == full

    residual = zeros(Float32, size(full))
    Readout.deposit_readout!(
        residual,
        packet,
        cache,
        Readout.RELATION_RESIDUAL_SCALE,
    )
    @test residual ≈ Readout.RELATION_RESIDUAL_SCALE .* full rtol=2e-6

    baseline = randn(rng, Float32, size(full))
    accumulated = copy(baseline)
    Readout.deposit_readout!(accumulated, packet, cache, 1.0f0)
    @test accumulated ≈ baseline + full rtol=2e-6 atol=2e-6

    @test_throws ArgumentError Readout.deposit_readout_selected!(
        selected,
        packet,
        cache,
        [1, 1],
        1.0f0,
    )
    @test_throws BoundsError Readout.deposit_readout_selected!(
        selected,
        packet,
        cache,
        [49],
        1.0f0,
    )
    @test_throws ArgumentError Readout.deposit_readout!(
        full,
        packet,
        cache,
        Float32(Inf),
    )
    @test_throws DimensionMismatch Readout.deposit_readout!(
        zeros(Float32, Cell.INPUT_DIM - 1, 22),
        packet,
        cache,
        1.0f0,
    )
    @test_throws DimensionMismatch Readout.deposit_readout!(
        full,
        zeros(Float32, 15, 48),
        cache,
        1.0f0,
    )
end

@testset "exact full pullback finite differences" begin
    rng = MersenneTwister(0x5a7a_19)
    parameters = Readout.initialize_parameters(Float64)
    parameters.source_gain_raw .= 0.4 .* randn(rng, Float64, 48, 22)
    cache = Readout.StructuredReadoutCache(parameters)
    packet = randn(rng, Float64, Packet.PACKET_DIM, Readout.SOURCE_COUNT)
    packet[abs.(packet) .< 0.15] .+= 0.35
    direction = randn(rng, Float64, Cell.INPUT_DIM, Readout.OUTPUT_COUNT)
    scale = 0.73

    packet_bar = zeros(Float64, size(packet))
    gradient = Readout.StructuredReadoutGradient(Float64)
    Readout.deposit_readout_pullback!(
        packet_bar,
        gradient,
        packet,
        cache,
        direction,
        scale,
    )

    epsilon = 1.0e-6
    for (field, source) in ((1, 1), (5, 24), (10, 25), (16, 34),
                            (3, 35), (11, 42), (4, 43), (13, 48))
        original = packet[field, source]
        packet[field, source] = original + epsilon
        plus = _objective(packet, parameters, direction, scale)
        packet[field, source] = original - epsilon
        minus = _objective(packet, parameters, direction, scale)
        packet[field, source] = original
        @test isapprox(
            packet_bar[field, source],
            (plus - minus) / (2epsilon);
            rtol=5e-7,
            atol=5e-8,
        )
    end

    for (source, output) in ((1, 1), (25, 2), (35, 9), (48, 22))
        original = parameters.source_gain_raw[source, output]
        parameters.source_gain_raw[source, output] = original + epsilon
        plus = _objective(packet, parameters, direction, scale)
        parameters.source_gain_raw[source, output] = original - epsilon
        minus = _objective(packet, parameters, direction, scale)
        parameters.source_gain_raw[source, output] = original
        @test gradient.source_gain_raw[source, output] ≈
              (plus - minus) / (2epsilon) rtol=8e-7 atol=8e-8
    end

    # Pullbacks are additive by contract.
    twice_packet_bar = copy(packet_bar)
    twice_gradient = Readout.StructuredReadoutGradient(Float64)
    twice_gradient.source_gain_raw .= gradient.source_gain_raw
    Readout.deposit_readout_pullback!(
        twice_packet_bar,
        twice_gradient,
        packet,
        cache,
        direction,
        scale,
    )
    @test twice_packet_bar ≈ 2 .* packet_bar rtol=2e-15 atol=2e-15
    @test twice_gradient.source_gain_raw ≈
          2 .* gradient.source_gain_raw rtol=2e-15 atol=2e-15
end

@testset "exact selected pullback and untouched families" begin
    rng = MersenneTwister(0x51ec_7ed)
    parameters = Readout.initialize_parameters(Float64)
    parameters.source_gain_raw .= randn(rng, Float64, 48, 22)
    cache = Readout.StructuredReadoutCache(parameters)
    packet = randn(rng, Float64, Packet.PACKET_DIM, Readout.SOURCE_COUNT)
    packet[abs.(packet) .< 0.2] .+= 0.5
    direction = randn(rng, Float64, Cell.INPUT_DIM, Readout.OUTPUT_COUNT)
    sources = [2, 9, 26, 44]
    scale = 0.125

    packet_bar = zeros(Float64, size(packet))
    gradient = Readout.StructuredReadoutGradient(Float64)
    Readout.deposit_readout_selected_pullback!(
        packet_bar,
        gradient,
        packet,
        cache,
        sources,
        direction,
        scale,
    )
    @test iszero(sum(abs, packet_bar[:, setdiff(1:48, sources)]))
    @test iszero(sum(abs, gradient.source_gain_raw[35:42, :]))

    epsilon = 1.0e-6
    for (field, source) in ((1, 2), (9, 9), (2, 26), (12, 44))
        original = packet[field, source]
        packet[field, source] = original + epsilon
        plus = _objective(
            packet,
            parameters,
            direction,
            scale;
            sources=sources,
        )
        packet[field, source] = original - epsilon
        minus = _objective(
            packet,
            parameters,
            direction,
            scale;
            sources=sources,
        )
        packet[field, source] = original
        @test isapprox(
            packet_bar[field, source],
            (plus - minus) / (2epsilon);
            rtol=7e-7,
            atol=7e-8,
        )
    end

    for (source, output) in ((2, 1), (26, 7), (44, 20))
        original = parameters.source_gain_raw[source, output]
        parameters.source_gain_raw[source, output] = original + epsilon
        plus = _objective(
            packet,
            parameters,
            direction,
            scale;
            sources=sources,
        )
        parameters.source_gain_raw[source, output] = original - epsilon
        minus = _objective(
            packet,
            parameters,
            direction,
            scale;
            sources=sources,
        )
        parameters.source_gain_raw[source, output] = original
        @test gradient.source_gain_raw[source, output] ≈
              (plus - minus) / (2epsilon) rtol=1e-6 atol=1e-7
    end

    all_sources = collect(Int, 1:48)
    full_packet_bar = zeros(Float64, size(packet))
    selected_packet_bar = zeros(Float64, size(full_packet_bar))
    full_gradient = Readout.StructuredReadoutGradient(Float64)
    selected_gradient = Readout.StructuredReadoutGradient(Float64)
    Readout.deposit_readout_pullback!(
        full_packet_bar,
        full_gradient,
        packet,
        cache,
        direction,
        scale,
    )
    Readout.deposit_readout_selected_pullback!(
        selected_packet_bar,
        selected_gradient,
        packet,
        cache,
        all_sources,
        direction,
        scale,
    )
    @test selected_packet_bar == full_packet_bar
    @test selected_gradient.source_gain_raw == full_gradient.source_gain_raw
end

@testset "Float32 hot kernels allocate zero bytes" begin
    rng = MersenneTwister(0xa110_c8)
    parameters = Readout.initialize_parameters(Float32)
    cache = Readout.StructuredReadoutCache(parameters)
    gradient = Readout.StructuredReadoutGradient(Float32)
    packet = randn(rng, Float32, Packet.PACKET_DIM, Readout.SOURCE_COUNT)
    packet[abs.(packet) .< 0.1f0] .+= 0.25f0
    packet_bar = zeros(Float32, size(packet))
    destination = zeros(Float32, Cell.INPUT_DIM, Readout.OUTPUT_COUNT)
    destination_bar = randn(rng, Float32, size(destination))
    sources = UInt8[1, 7, 25, 35, 43, 48]

    Readout.deposit_readout!(destination, packet, cache, 1.0f0)
    fill!(destination, 0.0f0)
    Readout.deposit_readout_selected!(
        destination,
        packet,
        cache,
        sources,
        0.125f0,
    )
    Readout.deposit_readout_pullback!(
        packet_bar,
        gradient,
        packet,
        cache,
        destination_bar,
        1.0f0,
    )
    fill!(packet_bar, 0.0f0)
    Readout.clear_gradient!(gradient)
    Readout.deposit_readout_selected_pullback!(
        packet_bar,
        gradient,
        packet,
        cache,
        sources,
        destination_bar,
        0.125f0,
    )

    @test @allocated(
        Readout.deposit_readout!(destination, packet, cache, 1.0f0),
    ) == 0
    @test @allocated(
        Readout.deposit_readout_selected!(
            destination,
            packet,
            cache,
            sources,
            0.125f0,
        ),
    ) == 0
    @test @allocated(
        Readout.deposit_readout_pullback!(
            packet_bar,
            gradient,
            packet,
            cache,
            destination_bar,
            1.0f0,
        ),
    ) == 0
    @test @allocated(
        Readout.deposit_readout_selected_pullback!(
            packet_bar,
            gradient,
            packet,
            cache,
            sources,
            destination_bar,
            0.125f0,
        ),
    ) == 0
    @test @allocated(Readout.refresh_cache!(cache, parameters)) == 0
    @test @allocated(Readout.clear_gradient!(gradient)) == 0
end
