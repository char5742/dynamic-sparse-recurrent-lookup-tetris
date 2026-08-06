using Test
using Random
using LinearAlgebra

include(joinpath(@__DIR__, "Architecture.jl"))
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "SensoryEncoder.jl"))

const Cell = ActiveApicalCell
const Encoder = SensoryEncoder

@testset "sensory encoder canonical contract" begin
    @test Cell.INPUT_DIM == Cell.N_COMPARTMENTS * Cell.INPUT_CHANNELS
    @test Encoder.INPUT_RAILS == 1_298
    @test Encoder.EXCITATORY_COORDINATES ==
          Encoder.BLOCKS * Encoder.CELLS_PER_BLOCK *
          Cell.N_COMPARTMENTS * 2
    @test Encoder.BOARD_RAILS == 960
    @test Encoder.NONBOARD_RAILS == 338

    excitatory_indices = [
        Encoder.destination_linear_index(Encoder.excitatory_destination(rail))
        for rail in 1:Encoder.INPUT_RAILS
    ]
    codes = [Encoder.rail_code(rail) for rail in 1:Encoder.INPUT_RAILS]
    @test length(unique(excitatory_indices)) == Encoder.INPUT_RAILS
    @test length(unique(codes)) == Encoder.INPUT_RAILS

    seen_blocks = falses(Encoder.BLOCKS)
    seen_compartments = falses(Cell.N_COMPARTMENTS)
    seen_receptors = falses(2)
    for rail in 1:Encoder.INPUT_RAILS
        excitatory = Encoder.excitatory_destination(rail)
        inhibitory = Encoder.gaba_destination(rail)
        compartment = div(excitatory.input - 1, Cell.INPUT_CHANNELS) + 1
        receptor = mod(excitatory.input - 1, Cell.INPUT_CHANNELS) + 1
        @test receptor in (Cell.INPUT_AMPA, Cell.INPUT_NMDA)
        @test inhibitory.input == Cell.input_index(compartment, Cell.INPUT_GABA)
        @test inhibitory.cell == excitatory.cell
        @test inhibitory.block == excitatory.block
        seen_blocks[excitatory.block] = true
        seen_compartments[compartment] = true
        seen_receptors[receptor] = true
    end
    @test all(seen_blocks)
    @test all(@view seen_compartments[1:4])
    @test !any(@view seen_compartments[5:Cell.N_COMPARTMENTS])
    @test all(seen_receptors)
end

@testset "four board planes preserve exact Tetris position" begin
    for column in 1:Encoder.BOARD_COLUMNS, row in 1:Encoder.BOARD_ROWS
        expected_block = (column - 1) * 3 + div(
            row - 1,
            Architecture.SPATIAL_POSITIONS_PER_BLOCK,
        ) + 1
        expected_cell = mod(
            row - 1,
            Architecture.SPATIAL_POSITIONS_PER_BLOCK,
        ) + 1
        destinations = ntuple(Encoder.BOARD_PLANES) do plane
            rail = Encoder.board_rail_index(plane, row, column)
            Encoder.excitatory_destination(rail)
        end

        @test all(destination -> destination.block == expected_block, destinations)
        @test all(destination -> destination.cell == expected_cell, destinations)
        @test length(unique(destination.input for destination in destinations)) == 4
        @test destinations[1].input == Cell.input_index(1, Cell.INPUT_AMPA)
        @test destinations[2].input == Cell.input_index(1, Cell.INPUT_NMDA)
        @test destinations[3].input == Cell.input_index(2, Cell.INPUT_AMPA)
        @test destinations[4].input == Cell.input_index(2, Cell.INPUT_NMDA)
    end
end

@testset "queue and auxiliary rails occupy remaining compartments" begin
    indices = Int[]
    for rail in (Encoder.BOARD_RAILS + 1):Encoder.INPUT_RAILS
        destination = Encoder.excitatory_destination(rail)
        compartment = div(destination.input - 1, Cell.INPUT_CHANNELS) + 1
        @test compartment in 3:4
        push!(indices, Encoder.destination_linear_index(destination))
    end
    @test length(unique(indices)) == Encoder.NONBOARD_RAILS
end

@testset "active apical compartment is sensory-isolated" begin
    rails = ones(Float32, Encoder.INPUT_RAILS)
    raw_gains = Encoder.default_raw_gains()
    encoded = Encoder.encode_sensory(rails, raw_gains)
    for block in 1:Encoder.BLOCKS, cell in 1:Encoder.CELLS_PER_BLOCK,
        compartment in Cell.N_COMPARTMENTS:Cell.N_COMPARTMENTS,
        channel in 1:Cell.INPUT_CHANNELS
        @test encoded[Cell.input_index(compartment, channel), cell, block] == 0.0f0
    end
end

@testset "single rail preserves exact position identity" begin
    raw_gains = Encoder.default_raw_gains()
    rails = zeros(Float32, Encoder.INPUT_RAILS)
    rail = 777
    rails[rail] = 1.0f0
    encoded = Encoder.encode_sensory(rails, raw_gains)
    excitatory = Encoder.excitatory_destination(rail)
    inhibitory = Encoder.gaba_destination(rail)
    associative_excitatory = Encoder.associative_destination(rail)
    associative_inhibitory = Encoder.associative_gaba_destination(rail)
    expected_excitatory = Encoder.bounded_gain(
        raw_gains[Encoder.EXCITATORY_GAIN, rail],
        Encoder.EXCITATORY_GAIN,
    )
    expected_inhibitory = Encoder.bounded_gain(
        raw_gains[Encoder.INHIBITORY_GAIN, rail],
        Encoder.INHIBITORY_GAIN,
    )

    @test encoded[excitatory.input, excitatory.cell, excitatory.block] ==
          expected_excitatory
    @test encoded[inhibitory.input, inhibitory.cell, inhibitory.block] ==
          expected_inhibitory
    @test encoded[
        associative_excitatory.input,
        associative_excitatory.cell,
        associative_excitatory.block,
    ] == expected_excitatory
    @test encoded[
        associative_inhibitory.input,
        associative_inhibitory.cell,
        associative_inhibitory.block,
    ] == expected_inhibitory
    @test count(!iszero, encoded) == 4
end

@testset "bounded E/I gains and functional/in-place equality" begin
    rng = MersenneTwister(0x51e5_0a11)
    raw_gains = Encoder.default_raw_gains()
    rails = Float32.(rand(rng, Bool, Encoder.INPUT_RAILS))
    functional = Encoder.encode_sensory(rails, raw_gains)
    destination = fill(13.0f0, Cell.INPUT_DIM, Encoder.CELLS_PER_BLOCK, Encoder.BLOCKS)
    Encoder.encode_sensory!(destination, rails, raw_gains)
    @test destination == functional
    @test minimum(destination) >= 0.0f0

    excitatory = Encoder.bounded_gain(
        raw_gains[Encoder.EXCITATORY_GAIN, 1],
        Encoder.EXCITATORY_GAIN,
    )
    inhibitory = Encoder.bounded_gain(
        raw_gains[Encoder.INHIBITORY_GAIN, 1],
        Encoder.INHIBITORY_GAIN,
    )
    @test 0.0f0 < inhibitory < excitatory
    @test excitatory < Encoder.EXCITATORY_GAIN_MAX
    @test inhibitory < Encoder.INHIBITORY_GAIN_MAX

    added = zeros(Float32, size(destination))
    Encoder.scatter_add_sensory!(added, rails, raw_gains)
    Encoder.scatter_add_sensory!(added, rails, raw_gains)
    @test added == 2.0f0 .* functional
end

function cached_allocation_probe!(
    destination,
    rail_bar,
    raw_bar,
    input_bar,
    rails,
    gains,
    raw_derivatives,
)
    Encoder.encode_sensory_cached!(destination, rails, gains)
    Encoder.scatter_add_sensory_cached!(destination, rails, gains)
    Encoder.sensory_cached_raw_vjp!(
        rail_bar,
        raw_bar,
        input_bar,
        rails,
        gains,
        raw_derivatives,
    )
    return nothing
end

@testset "exact analytic sensory VJP" begin
    rng = MersenneTwister(0xc0de_1298)
    rails = 0.1 .+ 0.8 .* rand(rng, Float64, Encoder.INPUT_RAILS)
    raw_gains = Float64.(Encoder.default_raw_gains())
    raw_gains .+= 0.15 .* randn(rng, size(raw_gains))
    input_bar = randn(
        rng,
        Float64,
        Cell.INPUT_DIM,
        Encoder.CELLS_PER_BLOCK,
        Encoder.BLOCKS,
    )
    rail_bar = zeros(Float64, Encoder.INPUT_RAILS)
    raw_bar = zeros(Float64, 2, Encoder.INPUT_RAILS)
    Encoder.sensory_vjp!(rail_bar, raw_bar, input_bar, rails, raw_gains)

    objective(test_rails, test_raw) =
        dot(Encoder.encode_sensory(test_rails, test_raw), input_bar)
    epsilon = 1.0e-6
    for rail in (1, 2, 419, 777, Encoder.INPUT_RAILS)
        plus_rails = copy(rails)
        minus_rails = copy(rails)
        plus_rails[rail] += epsilon
        minus_rails[rail] -= epsilon
        finite_difference =
            (objective(plus_rails, raw_gains) -
             objective(minus_rails, raw_gains)) / (2epsilon)
        @test isapprox(rail_bar[rail], finite_difference; rtol=2.0e-6, atol=2.0e-7)

        for gain in (Encoder.EXCITATORY_GAIN, Encoder.INHIBITORY_GAIN)
            plus_raw = copy(raw_gains)
            minus_raw = copy(raw_gains)
            plus_raw[gain, rail] += epsilon
            minus_raw[gain, rail] -= epsilon
            finite_difference =
                (objective(rails, plus_raw) -
                 objective(rails, minus_raw)) / (2epsilon)
            @test isapprox(
                raw_bar[gain, rail],
                finite_difference;
                rtol=3.0e-5,
                atol=2.0e-7,
            )
        end
    end
end

@testset "publish-boundary sensory cache is the exact raw map" begin
    rng = MersenneTwister(0xca5e_1298)
    rails = 0.1 .+ 0.8 .* rand(rng, Float64, Encoder.INPUT_RAILS)
    raw_gains = Float64.(Encoder.default_raw_gains())
    raw_gains .+= 0.4 .* randn(rng, size(raw_gains))
    gains = similar(raw_gains)
    raw_derivatives = similar(raw_gains)
    Encoder.transform_sensory_gains!(gains, raw_derivatives, raw_gains)

    @test all(isfinite, gains)
    @test all(isfinite, raw_derivatives)
    @test all(>(0.0), raw_derivatives)
    for rail in (1, 419, 777, Encoder.INPUT_RAILS)
        for gain in (Encoder.EXCITATORY_GAIN, Encoder.INHIBITORY_GAIN)
            @test gains[gain, rail] ==
                  Encoder.bounded_gain(raw_gains[gain, rail], gain)
            @test raw_derivatives[gain, rail] ==
                  Encoder.bounded_gain_derivative(raw_gains[gain, rail], gain)
        end
    end

    cold = Encoder.encode_sensory(rails, raw_gains)
    cached = similar(cold)
    Encoder.encode_sensory_cached!(cached, rails, gains)
    @test cached == cold

    input_bar = randn(rng, Float64, size(cold))
    cold_rail_bar = zeros(Float64, Encoder.INPUT_RAILS)
    cold_raw_bar = zeros(Float64, size(raw_gains))
    cached_rail_bar = similar(cold_rail_bar)
    cached_raw_bar = similar(cold_raw_bar)
    Encoder.sensory_vjp!(
        cold_rail_bar,
        cold_raw_bar,
        input_bar,
        rails,
        raw_gains,
    )
    Encoder.sensory_cached_raw_vjp!(
        cached_rail_bar,
        cached_raw_bar,
        input_bar,
        rails,
        gains,
        raw_derivatives,
    )
    @test cached_rail_bar == cold_rail_bar
    @test cached_raw_bar == cold_raw_bar

    # The production path consumes only the prepared caches.
    raw_gains .= NaN
    Encoder.encode_sensory_cached!(cached, rails, gains)
    @test cached == cold
end

@testset "fixed arena hot path allocates zero bytes" begin
    rng = MersenneTwister(0xa110_ca7e)
    rails = Float32.(rand(rng, Bool, Encoder.INPUT_RAILS))
    raw_gains = Encoder.default_raw_gains()
    gains = similar(raw_gains)
    raw_derivatives = similar(raw_gains)
    destination = zeros(Float32, Cell.INPUT_DIM, Encoder.CELLS_PER_BLOCK, Encoder.BLOCKS)
    input_bar = randn(
        rng,
        Float32,
        Cell.INPUT_DIM,
        Encoder.CELLS_PER_BLOCK,
        Encoder.BLOCKS,
    )
    rail_bar = zeros(Float32, Encoder.INPUT_RAILS)
    raw_bar = zeros(Float32, 2, Encoder.INPUT_RAILS)

    Encoder.transform_sensory_gains!(gains, raw_derivatives, raw_gains)
    cached_allocation_probe!(
        destination,
        rail_bar,
        raw_bar,
        input_bar,
        rails,
        gains,
        raw_derivatives,
    )
    transform_allocated = @allocated Encoder.transform_sensory_gains!(
        gains,
        raw_derivatives,
        raw_gains,
    )
    hot_allocated = @allocated cached_allocation_probe!(
        destination,
        rail_bar,
        raw_bar,
        input_bar,
        rails,
        gains,
        raw_derivatives,
    )
    @test transform_allocated == 0
    @test hot_allocated == 0
end
