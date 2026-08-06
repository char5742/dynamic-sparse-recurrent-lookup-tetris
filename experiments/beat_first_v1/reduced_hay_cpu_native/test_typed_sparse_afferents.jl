using Test
using Random

include(joinpath(@__DIR__, "TypedSparseAfferents.jl"))
using .TypedSparseAfferents

function objective_full(graph, features, direction)
    destination = zeros(eltype(features), INPUT_COUNT, DECISION_CELL_COUNT)
    deposit_full!(destination, graph, features)
    return sum(destination .* direction)
end

function objective_delta(graph, candidate, base, affected, direction)
    destination = zeros(eltype(candidate), INPUT_COUNT, DECISION_CELL_COUNT)
    deposit_affected_delta!(destination, graph, candidate, base, affected)
    return sum(destination .* direction)
end

@testset "typed sparse afferent topology" begin
    @test (POSITION_COUNT, FEATURE_COUNT) == (240, 27)
    @test (DECISION_CELL_COUNT, INPUT_COUNT) == (50, 27)
    @test FANOUT == 8

    first = build_typed_sparse_afferents(0x1a2b3c4d)
    repeat = build_typed_sparse_afferents(0x1a2b3c4d)
    changed = build_typed_sparse_afferents(0x1a2b3c4e)
    @test edge_count(first) == 240 * 27 * 8
    @test first.source_position == repeat.source_position
    @test first.source_feature == repeat.source_feature
    @test first.destination_cell == repeat.destination_cell
    @test first.destination_compartment == repeat.destination_compartment
    @test first.receptor == repeat.receptor
    @test first.raw_magnitude == repeat.raw_magnitude
    @test first.destination_cell != changed.destination_cell

    seen_cells = falses(DECISION_CELL_COUNT)
    for position in 1:POSITION_COUNT, feature in 1:FEATURE_COUNT
        targets = Set{Tuple{UInt8,UInt8,UInt8}}()
        expected_receptor = TypedSparseAfferents._feature_receptor(feature)
        for relation in 1:FANOUT
            slot = edge_slot(first, position, feature, relation)
            @test first.source_position[slot] == position
            @test first.source_feature[slot] == feature
            @test first.receptor[slot] == expected_receptor
            @test 1 <= destination_input(first, slot) <= INPUT_COUNT
            target = (
                first.destination_cell[slot],
                first.destination_compartment[slot],
                first.receptor[slot],
            )
            @test !(target in targets)
            push!(targets, target)
            seen_cells[Int(first.destination_cell[slot])] = true
        end
        @test length(targets) == FANOUT
    end
    @test all(seen_cells)
    @test all(raw -> isfinite(raw) && edge_magnitude(raw) > 0.0f0, first.raw_magnitude)
    # Initialization preserves the old single-exposure contact distribution
    # while dividing physical energy by two spatial planes times three phases.
    lower = edge_magnitude(-6.0f0) / 6.0f0
    upper = edge_magnitude(-5.0f0) / 6.0f0
    @test all(
        raw -> lower <= edge_magnitude(raw) < upper,
        first.raw_magnitude,
    )
end

@testset "full base plus affected delta equals full candidate" begin
    rng = MersenneTwister(0x5eed)
    graph = build_typed_sparse_afferents(0x51a7)
    base = randn(rng, Float32, FEATURE_COUNT, POSITION_COUNT)
    candidate = copy(base)
    affected = Int[1, 17, 119, 200, 240]
    for position in affected
        @views candidate[:, position] .+= 0.25f0 .* randn(rng, Float32, FEATURE_COUNT)
    end

    incremental = zeros(Float32, INPUT_COUNT, DECISION_CELL_COUNT)
    full = zeros(Float32, INPUT_COUNT, DECISION_CELL_COUNT)
    deposit_full!(incremental, graph, base)
    deposit_affected_delta!(incremental, graph, candidate, base, affected)
    deposit_full!(full, graph, candidate)
    @test incremental ≈ full rtol=2.0f-5 atol=2.0f-5
    @test all(value -> value >= -2.0f-5, incremental)

    @test_throws ArgumentError deposit_affected_delta!(
        incremental,
        graph,
        candidate,
        base,
        Int[17, 17],
    )
end

@testset "exact full and delta pullbacks" begin
    rng = MersenneTwister(0xc0ffee)
    graph = build_typed_sparse_afferents(0x9911, Float64)
    features = 0.2 .* randn(rng, Float64, FEATURE_COUNT, POSITION_COUNT)
    direction = randn(rng, Float64, INPUT_COUNT, DECISION_CELL_COUNT)
    feature_bar = zeros(Float64, FEATURE_COUNT, POSITION_COUNT)
    raw_bar = zeros(Float64, edge_count(graph))
    deposit_full_pullback!(feature_bar, raw_bar, graph, features, direction)

    epsilon = 1.0e-6
    feature = 11
    position = 137
    original_feature = features[feature, position]
    features[feature, position] = original_feature + epsilon
    plus = objective_full(graph, features, direction)
    features[feature, position] = original_feature - epsilon
    minus = objective_full(graph, features, direction)
    features[feature, position] = original_feature
    numerical_feature = (plus - minus) / (2epsilon)
    @test feature_bar[feature, position] ≈ numerical_feature rtol=2.0e-7 atol=2.0e-7

    slot = edge_slot(graph, position, feature, 3)
    original_raw = graph.raw_magnitude[slot]
    graph.raw_magnitude[slot] = original_raw + epsilon
    plus = objective_full(graph, features, direction)
    graph.raw_magnitude[slot] = original_raw - epsilon
    minus = objective_full(graph, features, direction)
    graph.raw_magnitude[slot] = original_raw
    numerical_raw = (plus - minus) / (2epsilon)
    @test raw_bar[slot] ≈ numerical_raw rtol=5.0e-7 atol=5.0e-7

    base = 0.2 .* randn(rng, Float64, FEATURE_COUNT, POSITION_COUNT)
    candidate = copy(base)
    affected = Int[3, 91, 201]
    for affected_position in affected
        @views candidate[:, affected_position] .+=
            0.1 .* randn(rng, Float64, FEATURE_COUNT)
    end
    candidate_bar = zeros(Float64, FEATURE_COUNT, POSITION_COUNT)
    base_bar = zeros(Float64, FEATURE_COUNT, POSITION_COUNT)
    delta_raw_bar = zeros(Float64, edge_count(graph))
    deposit_affected_delta_pullback!(
        candidate_bar,
        base_bar,
        delta_raw_bar,
        graph,
        candidate,
        base,
        affected,
        direction,
    )
    untouched = 92
    @test all(iszero, @view candidate_bar[:, untouched])
    @test all(iszero, @view base_bar[:, untouched])

    feature = 7
    position = affected[2]
    original = candidate[feature, position]
    candidate[feature, position] = original + epsilon
    plus = objective_delta(graph, candidate, base, affected, direction)
    candidate[feature, position] = original - epsilon
    minus = objective_delta(graph, candidate, base, affected, direction)
    candidate[feature, position] = original
    numerical_candidate = (plus - minus) / (2epsilon)
    @test candidate_bar[feature, position] ≈ numerical_candidate rtol=2.0e-7 atol=2.0e-7

    original_base = base[feature, position]
    base[feature, position] = original_base + epsilon
    plus = objective_delta(graph, candidate, base, affected, direction)
    base[feature, position] = original_base - epsilon
    minus = objective_delta(graph, candidate, base, affected, direction)
    base[feature, position] = original_base
    numerical_base = (plus - minus) / (2epsilon)
    @test base_bar[feature, position] ≈ numerical_base rtol=2.0e-7 atol=2.0e-7
end

@testset "typed afferent hot paths allocate zero bytes" begin
    rng = MersenneTwister(0xa110c)
    graph = build_typed_sparse_afferents(0x7711)
    base = randn(rng, Float32, FEATURE_COUNT, POSITION_COUNT)
    candidate = copy(base)
    affected = Int[4, 77, 181]
    for position in affected
        @views candidate[:, position] .+= 0.1f0 .* randn(rng, Float32, FEATURE_COUNT)
    end
    destination = zeros(Float32, INPUT_COUNT, DECISION_CELL_COUNT)
    direction = randn(rng, Float32, INPUT_COUNT, DECISION_CELL_COUNT)
    feature_bar = zeros(Float32, FEATURE_COUNT, POSITION_COUNT)
    candidate_bar = zeros(Float32, FEATURE_COUNT, POSITION_COUNT)
    base_bar = zeros(Float32, FEATURE_COUNT, POSITION_COUNT)
    raw_bar = zeros(Float32, edge_count(graph))

    deposit_full!(destination, graph, base)
    deposit_affected_delta!(destination, graph, candidate, base, affected)
    deposit_full_pullback!(feature_bar, raw_bar, graph, base, direction)
    deposit_affected_delta_pullback!(
        candidate_bar,
        base_bar,
        raw_bar,
        graph,
        candidate,
        base,
        affected,
        direction,
    )

    @test @allocated(deposit_full!(destination, graph, base)) == 0
    @test @allocated(
        deposit_affected_delta!(destination, graph, candidate, base, affected),
    ) == 0
    @test @allocated(
        deposit_full_pullback!(feature_bar, raw_bar, graph, base, direction),
    ) == 0
    @test @allocated(
        deposit_affected_delta_pullback!(
            candidate_bar,
            base_bar,
            raw_bar,
            graph,
            candidate,
            base,
            affected,
            direction,
        ),
    ) == 0
end
