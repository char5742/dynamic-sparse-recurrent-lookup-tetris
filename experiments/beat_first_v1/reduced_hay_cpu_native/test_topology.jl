using Test

module TopologyTestHarness
include(joinpath(@__DIR__, "Architecture.jl"))
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "SensoryEncoder.jl"))
include(joinpath(@__DIR__, "EventGraph.jl"))
include(joinpath(@__DIR__, "Topology.jl"))
end

const Graph = TopologyTestHarness.ReducedHayCPUNativeEventGraph
const Encoder = TopologyTestHarness.SensoryEncoder
using .TopologyTestHarness.Topology

@inline source_block(source::Int) = (source - 1) ÷ CELLS_PER_BLOCK + 1
@inline destination_block(destination::Int) =
    (destination - 1) ÷ CELLS_PER_BLOCK + 1

@inline function absolute_coordinates(block::Int, cell::Int)
    vertical_band = (block - 1) % 3
    column = (block - 1) ÷ 3 + 1
    row = vertical_band * 8 + mod1(cell, 8)
    return row, column
end

@inline function board_cell(row::Int, column::Int, lane::Int)
    vertical_band, cell_zero = divrem(row - 1, 8)
    block = (column - 1) * 3 + vertical_band + 1
    cell = (lane - 1) * 8 + cell_zero + 1
    return (block - 1) * CELLS_PER_BLOCK + cell
end

function literal_nearest_destinations(block::Int, cell::Int)
    source_row, source_column = absolute_coordinates(block, cell)
    lane = (cell - 1) ÷ 8 + 1
    candidates = Tuple{Int,Int,Int,Int}[]
    for column in 1:10, row in 1:24
        destination = board_cell(row, column, lane)
        destination_block_index = destination_block(destination)
        destination_block_index == block && continue
        distance = (row - source_row)^2 + (column - source_column)^2
        push!(candidates, (distance, row, column, destination))
    end
    sort!(candidates)
    return [candidates[index][4] for index in 1:SPATIAL_FANOUT]
end

function central_difference!(values, index::Int, objective; epsilon=1.0e-6)
    original = values[index]
    values[index] = original + epsilon
    positive = objective()
    values[index] = original - epsilon
    negative = objective()
    values[index] = original
    return (positive - negative) / (2 * epsilon)
end

function full_strength_objective!(cache, raw_derivatives, raw, dcache)
    transform_edge_strengths!(cache, raw_derivatives, raw)
    result = zero(eltype(cache))
    @inbounds for index in eachindex(cache)
        result = muladd(cache[index], dcache[index], result)
    end
    return result
end

function directional_difference!(
    raw,
    direction,
    objective;
    epsilon=1.0e-5,
)
    @inbounds for index in eachindex(raw)
        raw[index] += epsilon * direction[index]
    end
    positive = objective()
    @inbounds for index in eachindex(raw)
        raw[index] -= 2epsilon * direction[index]
    end
    negative = objective()
    @inbounds for index in eachindex(raw)
        raw[index] += epsilon * direction[index]
    end
    return (positive - negative) / (2epsilon)
end

function strength_hot_allocations(
    cache::Vector{Float32},
    raw_derivatives::Vector{Float32},
    raw::Array{Float32,3},
    draw::Array{Float32,3},
    dcache::Vector{Float32},
)
    transform_edge_strengths!(cache, raw_derivatives, raw)
    edge_strength_cached_raw_vjp!(draw, raw_derivatives, dcache)
    fill!(draw, 0.0f0)
    forward_bytes = @allocated transform_edge_strengths!(
        cache,
        raw_derivatives,
        raw,
    )
    reverse_bytes = @allocated edge_strength_cached_raw_vjp!(
        draw,
        raw_derivatives,
        dcache,
    )
    return forward_bytes, reverse_bytes
end

@testset "SensoryEncoder and topology share one spatial axis" begin
    graph = build_topology(0x510e527f, Float32)
    for column in 1:Encoder.BOARD_COLUMNS
        for board_row in 1:Encoder.BOARD_ROWS
            rail = Encoder.board_rail_index(1, board_row, column)
            sensory_destination = Encoder.excitatory_destination(rail)
            vertical_band = (board_row - 1) ÷ 8 + 1
            expected_block = (column - 1) * SPATIAL_ROWS + vertical_band
            @test sensory_destination.block == expected_block
            @test sensory_destination.cell == mod1(board_row, 8)

            source = (expected_block - 1) * CELLS_PER_BLOCK +
                     sensory_destination.cell
            first_spatial_slot = (source - 1) * FANOUT + LOCAL_FANOUT + 1
            spatial_slots = first_spatial_slot:(
                first_spatial_slot + SPATIAL_FANOUT - 1
            )
            observed = Int.(graph.destination_cell[spatial_slots])
            expected = literal_nearest_destinations(
                expected_block,
                sensory_destination.cell,
            )
            @test observed == expected
        end
    end


    top_left = literal_nearest_destinations(1, 1)
    top_left_coordinates = absolute_coordinates.(
        destination_block.(top_left),
        mod1.(top_left, 8),
    )
    @test all(coordinates -> coordinates[1] != 24, top_left_coordinates)
    @test all(coordinates -> coordinates[2] != 10, top_left_coordinates)

    bottom_right = literal_nearest_destinations(30, 8)
    bottom_right_coordinates = absolute_coordinates.(
        destination_block.(bottom_right),
        mod1.(bottom_right, 8),
    )
    @test all(coordinates -> coordinates[1] != 1, bottom_right_coordinates)
    @test all(coordinates -> coordinates[2] != 1, bottom_right_coordinates)
end

@testset "canonical topology determinism" begin
    first = build_topology(0x12345678, Float32)
    repeat = build_topology(0x12345678, Float32)
    changed = build_topology(0x12345679, Float32)

    @test BLOCKS == 30
    @test CELLS_PER_BLOCK == 16
    @test first.cell_count == TOTAL_CELLS == 480
    @test first.fanout == FANOUT == 48
    @test (LOCAL_FANOUT, SPATIAL_FANOUT, GLOBAL_FANOUT) == (16, 16, 16)
    @test (EXCITATORY_EDGES_PER_SOURCE, INHIBITORY_EDGES_PER_SOURCE) ==
          (36, 12)
    @test (CURRENT_EDGES_PER_SOURCE, PREVIOUS_EDGES_PER_SOURCE) == (36, 12)
    @test first.destination_cell == repeat.destination_cell
    @test first.destination_compartment == repeat.destination_compartment
    @test first.polarity == repeat.polarity
    @test first.delay_previous == repeat.delay_previous
    @test first.destination_cell != changed.destination_cell
    @test first.destination_compartment != changed.destination_compartment
    @test first.polarity != changed.polarity
    @test first.delay_previous != changed.delay_previous
    for source in 1:TOTAL_CELLS
        first_spatial = (source - 1) * FANOUT + LOCAL_FANOUT + 1
        spatial_slots = first_spatial:(first_spatial + SPATIAL_FANOUT - 1)
        @test first.destination_cell[spatial_slots] ==
              changed.destination_cell[spatial_slots]
    end
end

@testset "golden topology seed and source mapping" begin
    graph = build_topology(0x6a09e667, Float32)
    source = 137
    slots = ((source - 1) * FANOUT + 1):(source * FANOUT)
    @test Int.(graph.destination_cell[slots]) == Int[
        129, 130, 131, 132, 133, 134, 135, 136,
        137, 138, 139, 140, 141, 142, 143, 144,
        128, 89, 185, 80, 176, 90, 186, 127,
        41, 233, 79, 175, 32, 224, 42, 234,
        100, 101, 102, 103, 104, 105, 106, 107,
        108, 109, 110, 111, 112, 145, 146, 147,
    ]
    @test Int.(graph.destination_compartment[slots]) == Int[
        5, 7, 9, 2, 1, 3, 5, 7,
        9, 8, 1, 3, 5, 4, 6, 8,
        1, 3, 2, 4, 6, 8, 7, 9,
        2, 4, 3, 5, 7, 9, 2, 1,
        3, 5, 7, 6, 8, 1, 3, 2,
        4, 6, 8, 1, 9, 2, 4, 6,
    ]
    @test graph.polarity[slots] == UInt8[
        1, 1, 2, 1, 1, 2, 1, 1,
        1, 2, 1, 1, 1, 2, 1, 1,
        2, 1, 1, 1, 2, 1, 1, 1,
        2, 1, 1, 1, 1, 1, 1, 2,
        1, 1, 1, 2, 1, 1, 1, 2,
        1, 1, 2, 1, 1, 1, 2, 1,
    ]
    @test graph.delay_previous[slots] == UInt8[
        0, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0, 0,
    ]
end

@testset "canonical topology structural contract" begin
    graph = build_topology(0x6a09e667, Float32)
    @test length(graph.destination_cell) == EDGE_COUNT
    @test fieldnames(typeof(graph)) == (
        :cell_count,
        :fanout,
        :destination_cell,
        :destination_compartment,
        :polarity,
        :delay_previous,
    )
    @test all(destination -> 1 <= destination <= TOTAL_CELLS, graph.destination_cell)
    @test all(
        compartment -> 1 <= compartment <= TopologyTestHarness.ActiveApicalCell.N_COMPARTMENTS,
        graph.destination_compartment,
    )

    for source in 1:TOTAL_CELLS
        first_slot = (source - 1) * FANOUT + 1
        slots = first_slot:(first_slot + FANOUT - 1)
        destinations = Int.(graph.destination_cell[slots])
        @test length(unique(destinations)) == FANOUT

        block = source_block(source)
        local_destinations = destinations[1:LOCAL_FANOUT]
        @test Set(destination_block.(local_destinations)) == Set((block,))
        @test Set(mod1.(local_destinations, CELLS_PER_BLOCK)) ==
              Set(1:CELLS_PER_BLOCK)

        expected_spatial = literal_nearest_destinations(
            block,
            mod1(source, CELLS_PER_BLOCK),
        )
        spatial_range = (LOCAL_FANOUT + 1):(LOCAL_FANOUT + SPATIAL_FANOUT)
        @test destinations[spatial_range] == expected_spatial
        spatial_blocks = Set(destination_block.(expected_spatial))

        global_destinations = destinations[(LOCAL_FANOUT + SPATIAL_FANOUT + 1):FANOUT]
        @test all(global_destinations) do destination
            destination_block(destination) != block &&
                !(destination_block(destination) in spatial_blocks)
        end

        polarities = graph.polarity[slots]
        @test count(==(Graph.EXCITATORY), polarities) == 36
        @test count(==(Graph.INHIBITORY), polarities) == 12
        delays = graph.delay_previous[slots]
        @test count(==(0x00), delays) == 36
        @test count(==(0x01), delays) == 12
        @test Set(Int.(graph.destination_compartment[slots])) ==
              Set(1:TopologyTestHarness.ActiveApicalCell.N_COMPARTMENTS)
    end
end

@testset "strength transform independent grid" begin
    @test STRENGTH_MIN == 0.01f0 / sqrt(Float32(FANOUT))
    @test STRENGTH_MAX == 0.20f0 / sqrt(Float32(FANOUT))
    @test STRENGTH_DEFAULT == (STRENGTH_MIN + STRENGTH_MAX) / 2.0f0
    raw = initialize_edge_strength_raw(Float64)
    cache = initialize_edge_strength_cache(Float64)
    raw_derivatives = initialize_edge_strength_cache(Float64)
    grid = Float64[-12, -4, -1, 0, 1, 4, 12]
    for index in eachindex(grid)
        raw[index] = grid[index]
    end
    transform_edge_strengths!(cache, raw_derivatives, raw)
    minimum_strength = Float64(STRENGTH_MIN)
    strength_range = Float64(STRENGTH_MAX) - minimum_strength
    dcache = zeros(Float64, EDGE_COUNT)
    for index in eachindex(grid)
        probability = 1.0 / (1.0 + exp(-grid[index]))
        expected = minimum_strength + strength_range * probability
        @test cache[index] ≈ expected rtol=2eps(Float64) atol=0
        dcache[index] = 0.3 + 0.1 * index
    end
    draw = zeros(Float64, size(raw))
    edge_strength_cached_raw_vjp!(draw, raw_derivatives, dcache)
    for index in eachindex(grid)
        probability = 1.0 / (1.0 + exp(-grid[index]))
        expected = dcache[index] * strength_range *
                   probability * (1.0 - probability)
        @test draw[index] ≈ expected rtol=8eps(Float64) atol=1.0e-18
        @test raw_derivatives[index] ≈
              strength_range * probability * (1.0 - probability) rtol=8eps(Float64) atol=1.0e-18
    end
    @test issorted(cache[1:length(grid)])
end

@testset "bounded source-major strength cache" begin
    raw = initialize_edge_strength_raw(Float64)
    cache = initialize_edge_strength_cache(Float64)
    raw_derivatives = initialize_edge_strength_cache(Float64)
    @test size(raw) == (FANOUT, CELLS_PER_BLOCK, BLOCKS)
    @test length(cache) == EDGE_COUNT
    transform_edge_strengths!(cache, raw_derivatives, raw)
    expected_default =
        (Float64(STRENGTH_MIN) + Float64(STRENGTH_MAX)) / 2.0
    @test all(strength -> strength == expected_default, cache)
    @test all(>(0.0), cache)
    @test all(strength -> STRENGTH_MIN <= strength <= STRENGTH_MAX, cache)

    # Julia's column-major order matches relation -> cell -> block and hence
    # EventGraph's flat source-major edge slots exactly.
    raw[3, 2, 4] = 0.75
    transform_edge_strengths!(cache, raw_derivatives, raw)
    source = (4 - 1) * CELLS_PER_BLOCK + 2
    slot = (source - 1) * FANOUT + 3
    @test cache[slot] > STRENGTH_DEFAULT

    dcache = [sin(0.013 * index) for index in 1:EDGE_COUNT]
    draw = zeros(Float64, size(raw))
    edge_strength_cached_raw_vjp!(draw, raw_derivatives, dcache)
    for index in (1, slot, EDGE_COUNT ÷ 2, EDGE_COUNT)
        local_objective = () -> begin
            transform_edge_strengths!(cache, raw_derivatives, raw)
            cache[index] * dcache[index]
        end
        finite_difference = central_difference!(raw, index, local_objective)
        @test draw[index] ≈ finite_difference rtol=1.0e-8 atol=1.0e-10
    end

    first_draw = copy(draw)
    edge_strength_cached_raw_vjp!(draw, raw_derivatives, dcache)
    @test draw ≈ 2 .* first_draw

    # One full-parameter directional derivative catches indexing or source-major
    # permutation errors that isolated coordinate checks can miss.
    for index in eachindex(raw)
        raw[index] = 0.75 * sin(0.007 * index)
        dcache[index] = cos(0.011 * index)
    end
    direction = similar(raw)
    @inbounds for index in eachindex(direction)
        direction[index] = 0.2 * sin(0.017 * index)
    end
    fill!(draw, 0.0)
    transform_edge_strengths!(cache, raw_derivatives, raw)
    edge_strength_cached_raw_vjp!(draw, raw_derivatives, dcache)
    analytic_directional = 0.0
    @inbounds for index in eachindex(draw)
        analytic_directional = muladd(draw[index], direction[index], analytic_directional)
    end
    objective = () -> full_strength_objective!(
        cache,
        raw_derivatives,
        raw,
        dcache,
    )
    finite_directional = directional_difference!(raw, direction, objective)
    @test analytic_directional ≈ finite_directional rtol=1.0e-7 atol=1.0e-10
end

@testset "strength cache allocation and boundaries" begin
    raw = initialize_edge_strength_raw(Float32)
    cache = initialize_edge_strength_cache(Float32)
    raw_derivatives = initialize_edge_strength_cache(Float32)
    draw = similar(raw)
    fill!(draw, 0.0f0)
    dcache = fill(0.25f0, EDGE_COUNT)
    forward_bytes, reverse_bytes = strength_hot_allocations(
        cache,
        raw_derivatives,
        raw,
        draw,
        dcache,
    )
    @test forward_bytes == 0
    @test reverse_bytes == 0

    raw[1] = -80.0f0
    raw[2] = 80.0f0
    transform_edge_strengths!(cache, raw_derivatives, raw)
    @test cache[1] >= STRENGTH_MIN > 0.0f0
    @test cache[2] <= STRENGTH_MAX
    @test cache[1] < cache[2]
    raw[3] = Inf32
    @test_throws ArgumentError transform_edge_strengths!(
        cache,
        raw_derivatives,
        raw,
    )
end
