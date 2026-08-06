using Test
using Random

include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "TypedDendriticAfferents.jl"))

const Cell = ActiveApicalCell
using .TypedDendriticAfferents

function _objective(graph, packet, direction)
    destination = zeros(eltype(packet), Cell.INPUT_DIM, graph.destination_count)
    deposit_typed!(destination, graph, packet)
    return sum(destination .* direction)
end

function _objective_sources(graph, packet, sources, direction)
    destination = zeros(eltype(packet), Cell.INPUT_DIM, graph.destination_count)
    deposit_sources!(destination, graph, packet, sources)
    return sum(destination .* direction)
end

function _objective_delta(graph, candidate, base, sources, direction)
    destination = zeros(eltype(candidate), Cell.INPUT_DIM, graph.destination_count)
    deposit_delta!(destination, graph, candidate, base, sources)
    return sum(destination .* direction)
end

@testset "explicit typed-afferent constructor" begin
    kinds = UInt8[ANALOG_FIELD, HARD_BIT_FIELD]
    source_field = UInt16[1, 2, 1, 2]
    polarity = Int8[-1, 1, 1, 1]
    cells = UInt16[1, 1, 2, 2]
    compartments = UInt8[1, 2, 1, 2]
    receptors = UInt8[AMPA_RECEPTOR, NMDA_RECEPTOR, GABA_RECEPTOR, AMPA_RECEPTOR]
    raw = Float32[-2, -1, 0, 1]
    graph = TypedAfferentGraph(
        2,
        kinds,
        2,
        2,
        source_field,
        polarity,
        cells,
        compartments,
        receptors,
        raw,
    )
    @test graph.source_count == 2
    @test graph.field_count == 2
    @test graph.destination_count == 2
    @test graph.fanout == 2
    @test graph.destination_input[1] == Cell.input_index(1, Cell.INPUT_AMPA)
    @test graph.destination_input[2] == Cell.input_index(2, Cell.INPUT_NMDA)

    # The constructor owns its topology and parameter storage.
    source_field[1] = UInt16(2)
    polarity[1] = Int8(1)
    raw[1] = 9
    @test graph.source_field[1] == UInt16(1)
    @test graph.source_polarity[1] == Int8(-1)
    @test graph.raw_conductance[1] == -2

    bad_polarity = Int8[-1, -1, 1, 1]
    @test_throws ArgumentError TypedAfferentGraph(
        2,
        kinds,
        2,
        2,
        UInt16[1, 2, 1, 2],
        bad_polarity,
        cells,
        compartments,
        receptors,
        Float32[-2, -1, 0, 1],
    )
    @test_throws ArgumentError TypedAfferentGraph(
        2,
        kinds,
        2,
        2,
        UInt16[1, 2, 1],
        Int8[-1, 1, 1],
        UInt16[1, 1, 2],
        UInt8[1, 2, 1],
        UInt8[AMPA_RECEPTOR, NMDA_RECEPTOR, GABA_RECEPTOR],
        Float32[-2, -1, 0],
    )
end

@testset "typed afferent fixed anatomical contract" begin
    kinds = UInt8[ANALOG_FIELD, ANALOG_FIELD, HARD_BIT_FIELD]
    first = build_typed_afferents(
        32,
        kinds,
        11;
        fanout=12,
        seed=0x13a7,
        initial_conductance=0.02,
    )
    repeat = build_typed_afferents(
        32,
        kinds,
        11;
        fanout=12,
        seed=0x13a7,
        initial_conductance=0.02,
    )
    changed = build_typed_afferents(
        32,
        kinds,
        11;
        fanout=12,
        seed=0x13a8,
        initial_conductance=0.02,
    )

    @test contact_count(first) == 32 * 12
    @test first.source_field == repeat.source_field
    @test first.source_polarity == repeat.source_polarity
    @test first.destination_cell == repeat.destination_cell
    @test first.destination_input == repeat.destination_input
    @test first.receptor == repeat.receptor
    @test first.raw_conductance == repeat.raw_conductance
    @test first.destination_cell != changed.destination_cell
    @test Set(first.receptor) == Set(UInt8[AMPA_RECEPTOR, NMDA_RECEPTOR, GABA_RECEPTOR])
    @test Set(
        first.source_polarity[index] for index in eachindex(first.source_polarity)
        if first.field_kind[Int(first.source_field[index])] == ANALOG_FIELD
    ) == Set(Int8[-1, 1])
    @test all(
        index -> first.field_kind[Int(first.source_field[index])] != HARD_BIT_FIELD ||
                 first.source_polarity[index] == Int8(1),
        eachindex(first.source_polarity),
    )
    @test all(raw -> isfinite(raw) && conductance(raw) > 0, first.raw_conductance)

    for source in 1:first.source_count
        seen = Set{Tuple{UInt16,UInt8}}()
        for relation in 1:first.fanout
            slot = contact_slot(first, source, relation)
            typed_destination = (
                first.destination_cell[slot],
                first.destination_input[slot],
            )
            @test !(typed_destination in seen)
            push!(seen, typed_destination)
            @test first.destination_input[slot] == Cell.input_index(
                Int(first.destination_compartment[slot]),
                Int(first.receptor[slot]),
            )
        end
    end
end

@testset "semantic sign never changes receptor identity" begin
    graph = build_typed_afferents(
        1,
        UInt8[ANALOG_FIELD, HARD_BIT_FIELD],
        1;
        fanout=3,
        seed=0x51a7,
        T=Float64,
        initial_conductance=1.0,
    )

    # Construct three explicit anatomical contacts.  A negative-polarity AMPA
    # contact must remain AMPA; positive semantic evidence can independently
    # recruit a fixed GABA contact.  Neither path infers receptor from sign.
    graph.source_field[1] = UInt16(1)
    graph.source_polarity[1] = Int8(-1)
    graph.destination_cell[1] = UInt16(1)
    graph.destination_compartment[1] = UInt8(1)
    graph.receptor[1] = AMPA_RECEPTOR
    graph.destination_input[1] = UInt8(Cell.input_index(1, Cell.INPUT_AMPA))

    graph.source_field[2] = UInt16(1)
    graph.source_polarity[2] = Int8(1)
    graph.destination_cell[2] = UInt16(1)
    graph.destination_compartment[2] = UInt8(1)
    graph.receptor[2] = GABA_RECEPTOR
    graph.destination_input[2] = UInt8(Cell.input_index(1, Cell.INPUT_GABA))

    graph.source_field[3] = UInt16(2)
    graph.source_polarity[3] = Int8(1)
    graph.destination_cell[3] = UInt16(1)
    graph.destination_compartment[3] = UInt8(1)
    graph.receptor[3] = NMDA_RECEPTOR
    graph.destination_input[3] = UInt8(Cell.input_index(1, Cell.INPUT_NMDA))
    fill!(graph.raw_conductance, log(exp(1.0) - 1.0))
    validate_typed_afferents(graph)
    fixed_receptors = copy(graph.receptor)

    packet = reshape(Float64[-2, 0], 2, 1)
    destination = zeros(Float64, Cell.INPUT_DIM, 1)
    deposit_typed!(destination, graph, packet)
    @test destination[Cell.input_index(1, Cell.INPUT_AMPA), 1] ≈ 2.0
    @test iszero(destination[Cell.input_index(1, Cell.INPUT_NMDA), 1])
    @test iszero(destination[Cell.input_index(1, Cell.INPUT_GABA), 1])

    fill!(destination, 0)
    packet[1, 1] = 3
    packet[2, 1] = 1
    deposit_typed!(destination, graph, packet)
    @test iszero(destination[Cell.input_index(1, Cell.INPUT_AMPA), 1])
    @test destination[Cell.input_index(1, Cell.INPUT_NMDA), 1] ≈ 1.0
    @test destination[Cell.input_index(1, Cell.INPUT_GABA), 1] ≈ 3.0
    @test graph.receptor == fixed_receptors
    @test all(value -> value >= 0, destination)

    packet[2, 1] = 0.25
    @test_throws DomainError deposit_typed!(destination, graph, packet)
end

@testset "zero is silence and hard bit has no source cotangent" begin
    graph = build_typed_afferents(
        5,
        UInt8[ANALOG_FIELD, ANALOG_FIELD, HARD_BIT_FIELD],
        4;
        fanout=9,
        seed=0x0,
    )
    packet = zeros(Float32, 3, 5)
    destination = zeros(Float32, Cell.INPUT_DIM, 4)
    deposit_typed!(destination, graph, packet)
    @test all(iszero, destination)

    packet[3, :] .= 1
    deposit_typed!(destination, graph, packet)
    @test any(!iszero, destination)
    packet_bar = zeros(Float32, size(packet))
    raw_bar = zeros(Float32, contact_count(graph))
    direction = ones(Float32, size(destination))
    deposit_typed_pullback!(packet_bar, raw_bar, graph, packet, direction)
    @test all(iszero, @view packet_bar[3, :])
    @test any(!iszero, raw_bar)
end

@testset "source-list deposit visits only selected sources" begin
    rng = MersenneTwister(0x50a7ce)
    graph = build_typed_afferents(
        12,
        UInt8[ANALOG_FIELD, ANALOG_FIELD, HARD_BIT_FIELD],
        7;
        fanout=10,
        seed=0x71a7,
        T=Float64,
        initial_conductance=0.08,
    )
    sources = Int[2, 5, 9, 12]
    packet = zeros(Float64, 3, 12)
    packet[1:2, sources] .= randn(rng, 2, length(sources))
    packet[3, sources] .= rand(rng, Bool, length(sources))
    full = zeros(Float64, Cell.INPUT_DIM, 7)
    sparse = zeros(Float64, Cell.INPUT_DIM, 7)
    deposit_typed!(full, graph, packet)
    deposit_sources!(sparse, graph, packet, sources)
    @test sparse == full
    @test_throws ArgumentError deposit_sources!(
        sparse,
        graph,
        packet,
        Int[2, 5, 5],
    )

    direction = randn(rng, Float64, Cell.INPUT_DIM, 7)
    packet_bar = zeros(Float64, size(packet))
    raw_bar = zeros(Float64, contact_count(graph))
    deposit_sources_pullback!(
        packet_bar,
        raw_bar,
        graph,
        packet,
        sources,
        direction,
    )
    @test all(iszero, @view packet_bar[:, 1])
    @test all(iszero, @view packet_bar[:, 3])
    selected_slots = falses(contact_count(graph))
    for source in sources, relation in 1:graph.fanout
        selected_slots[contact_slot(graph, source, relation)] = true
    end
    @test all(
        slot -> selected_slots[slot] || iszero(raw_bar[slot]),
        eachindex(raw_bar),
    )

    epsilon = 1.0e-6
    source = sources[2]
    field = 1
    original = packet[field, source]
    abs(original) < 0.1 && (packet[field, source] = original + 0.5)
    original = packet[field, source]
    fill!(packet_bar, 0)
    fill!(raw_bar, 0)
    deposit_sources_pullback!(
        packet_bar,
        raw_bar,
        graph,
        packet,
        sources,
        direction,
    )
    packet[field, source] = original + epsilon
    plus = _objective_sources(graph, packet, sources, direction)
    packet[field, source] = original - epsilon
    minus = _objective_sources(graph, packet, sources, direction)
    packet[field, source] = original
    @test packet_bar[field, source] ≈ (plus - minus) / (2epsilon) rtol=2e-7 atol=2e-8

    slot = 0
    for active_source in sources, relation in 1:graph.fanout
        candidate_slot = contact_slot(graph, active_source, relation)
        field = Int(graph.source_field[candidate_slot])
        activity = TypedDendriticAfferents._activity(
            packet[field, active_source],
            graph.field_kind[field],
            graph.source_polarity[candidate_slot],
        )
        if !iszero(activity)
            slot = candidate_slot
            break
        end
    end
    @test slot > 0
    @test abs(raw_bar[slot]) > 1e-10
    original_raw = graph.raw_conductance[slot]
    graph.raw_conductance[slot] = original_raw + epsilon
    plus = _objective_sources(graph, packet, sources, direction)
    graph.raw_conductance[slot] = original_raw - epsilon
    minus = _objective_sources(graph, packet, sources, direction)
    graph.raw_conductance[slot] = original_raw
    @test raw_bar[slot] ≈ (plus - minus) / (2epsilon) rtol=5e-7 atol=2e-8
end

@testset "candidate delta exactly updates cached typed inbox" begin
    rng = MersenneTwister(0xde17a)
    graph = build_typed_afferents(
        14,
        UInt8[ANALOG_FIELD, ANALOG_FIELD, HARD_BIT_FIELD],
        8;
        fanout=11,
        seed=0xa11ce,
        T=Float64,
        initial_conductance=0.12,
    )
    base = randn(rng, Float64, 3, 14)
    base[3, :] .= rand(rng, Bool, 14)
    candidate = copy(base)
    sources = Int[1, 4, 8, 13]
    # Include both analog ReLU sign crossings and hard 0/1 changes.
    candidate[1, 1] = -base[1, 1] - 0.3
    candidate[2, 4] = -base[2, 4] + 0.4
    candidate[1:2, 8] .+= randn(rng, 2)
    candidate[1:2, 13] .-= randn(rng, 2)
    for source in sources
        candidate[3, source] = 1 - base[3, source]
    end

    cached = zeros(Float64, Cell.INPUT_DIM, 8)
    expected = zeros(Float64, Cell.INPUT_DIM, 8)
    deposit_typed!(cached, graph, base)
    deposit_delta!(cached, graph, candidate, base, sources)
    deposit_typed!(expected, graph, candidate)
    @test cached ≈ expected rtol=2e-14 atol=2e-14
    @test minimum(cached) >= -2e-14

    direction = randn(rng, Float64, Cell.INPUT_DIM, 8)
    candidate_bar = zeros(Float64, size(candidate))
    base_bar = zeros(Float64, size(base))
    raw_bar = zeros(Float64, contact_count(graph))
    deposit_delta_pullback!(
        candidate_bar,
        base_bar,
        raw_bar,
        graph,
        candidate,
        base,
        sources,
        direction,
    )
    @test all(iszero, @view candidate_bar[3, :])
    @test all(iszero, @view base_bar[3, :])
    @test all(iszero, @view candidate_bar[:, 2])
    @test all(iszero, @view base_bar[:, 2])

    epsilon = 1.0e-6
    for source in sources, field in 1:2
        original = candidate[field, source]
        abs(original) < 0.1 && (candidate[field, source] = original + 0.5)
        original = candidate[field, source]
        # Recompute because the preceding anti-kink adjustment changes the
        # exact analytical point for this coordinate.
        fill!(candidate_bar, 0)
        fill!(base_bar, 0)
        fill!(raw_bar, 0)
        deposit_delta_pullback!(
            candidate_bar,
            base_bar,
            raw_bar,
            graph,
            candidate,
            base,
            sources,
            direction,
        )
        candidate[field, source] = original + epsilon
        plus = _objective_delta(graph, candidate, base, sources, direction)
        candidate[field, source] = original - epsilon
        minus = _objective_delta(graph, candidate, base, sources, direction)
        candidate[field, source] = original
        @test candidate_bar[field, source] ≈
              (plus - minus) / (2epsilon) rtol=3e-7 atol=3e-8

        original_base = base[field, source]
        abs(original_base) < 0.1 && (base[field, source] = original_base - 0.5)
        original_base = base[field, source]
        fill!(candidate_bar, 0)
        fill!(base_bar, 0)
        fill!(raw_bar, 0)
        deposit_delta_pullback!(
            candidate_bar,
            base_bar,
            raw_bar,
            graph,
            candidate,
            base,
            sources,
            direction,
        )
        base[field, source] = original_base + epsilon
        plus = _objective_delta(graph, candidate, base, sources, direction)
        base[field, source] = original_base - epsilon
        minus = _objective_delta(graph, candidate, base, sources, direction)
        base[field, source] = original_base
        @test base_bar[field, source] ≈
              (plus - minus) / (2epsilon) rtol=3e-7 atol=3e-8
    end

    fill!(candidate_bar, 0)
    fill!(base_bar, 0)
    fill!(raw_bar, 0)
    deposit_delta_pullback!(
        candidate_bar,
        base_bar,
        raw_bar,
        graph,
        candidate,
        base,
        sources,
        direction,
    )
    slot = 0
    for active_source in sources, relation in 1:graph.fanout
        candidate_slot = contact_slot(graph, active_source, relation)
        field = Int(graph.source_field[candidate_slot])
        kind = graph.field_kind[field]
        polarity = graph.source_polarity[candidate_slot]
        activity_delta = TypedDendriticAfferents._activity(
            candidate[field, active_source],
            kind,
            polarity,
        ) - TypedDendriticAfferents._activity(
            base[field, active_source],
            kind,
            polarity,
        )
        if !iszero(activity_delta)
            slot = candidate_slot
            break
        end
    end
    @test slot > 0
    @test abs(raw_bar[slot]) > 1e-10
    original_raw = graph.raw_conductance[slot]
    graph.raw_conductance[slot] = original_raw + epsilon
    plus = _objective_delta(graph, candidate, base, sources, direction)
    graph.raw_conductance[slot] = original_raw - epsilon
    minus = _objective_delta(graph, candidate, base, sources, direction)
    graph.raw_conductance[slot] = original_raw
    @test raw_bar[slot] ≈ (plus - minus) / (2epsilon) rtol=6e-7 atol=3e-8
end

@testset "typed afferent pullback finite differences" begin
    rng = MersenneTwister(0xc0ffee)
    graph = build_typed_afferents(
        7,
        UInt8[ANALOG_FIELD, ANALOG_FIELD, HARD_BIT_FIELD],
        6;
        fanout=11,
        seed=0x9911,
        T=Float64,
        initial_conductance=0.1,
    )
    packet = randn(rng, Float64, 3, 7)
    packet[abs.(packet) .< 0.2] .+= 0.5
    packet[3, :] .= rand(rng, Bool, 7)
    direction = randn(rng, Float64, Cell.INPUT_DIM, 6)
    packet_bar = zeros(Float64, size(packet))
    raw_bar = zeros(Float64, contact_count(graph))
    deposit_typed_pullback!(packet_bar, raw_bar, graph, packet, direction)

    epsilon = 1.0e-6
    checked_packet = 0
    for source in 1:graph.source_count, field in 1:2
        original = packet[field, source]
        packet[field, source] = original + epsilon
        plus = _objective(graph, packet, direction)
        packet[field, source] = original - epsilon
        minus = _objective(graph, packet, direction)
        packet[field, source] = original
        numerical = (plus - minus) / (2epsilon)
        @test packet_bar[field, source] ≈ numerical rtol=2.0e-7 atol=2.0e-8
        checked_packet += 1
    end
    @test checked_packet == 14

    for slot in (1, div(contact_count(graph), 2), contact_count(graph))
        original = graph.raw_conductance[slot]
        graph.raw_conductance[slot] = original + epsilon
        plus = _objective(graph, packet, direction)
        graph.raw_conductance[slot] = original - epsilon
        minus = _objective(graph, packet, direction)
        graph.raw_conductance[slot] = original
        numerical = (plus - minus) / (2epsilon)
        @test raw_bar[slot] ≈ numerical rtol=5.0e-7 atol=2.0e-8
    end
end

@testset "optimizer-step conductance cache preserves exact semantics" begin
    graph = TypedAfferentGraph(
        1,
        UInt8[ANALOG_FIELD],
        1,
        2,
        UInt16[1, 1],
        Int8[1, -1],
        UInt16[1, 1],
        UInt8[1, 2],
        UInt8[AMPA_RECEPTOR, AMPA_RECEPTOR],
        Float32[-0.2, 0.3],
    )
    cache = TypedAfferentCache(graph)
    packet = reshape(Float32[0.75], 1, 1)
    direction = reshape(
        collect(Float32, 1:Cell.INPUT_DIM),
        Cell.INPUT_DIM,
        1,
    )

    uncached = zeros(Float32, Cell.INPUT_DIM, 1)
    cached = zeros(Float32, Cell.INPUT_DIM, 1)
    deposit_typed!(uncached, graph, packet)
    deposit_typed!(cached, graph, cache, packet)
    @test cached == uncached
    @test cache.physical == conductance.(graph.raw_conductance)
    @test cache.derivative ==
          TypedDendriticAfferents._conductance_derivative.(
              graph.raw_conductance,
          )

    packet_bar_uncached = zeros(Float32, 1, 1)
    packet_bar_cached = zeros(Float32, 1, 1)
    raw_bar_uncached = zeros(Float32, 2)
    raw_bar_cached = zeros(Float32, 2)
    deposit_typed_pullback!(
        packet_bar_uncached,
        raw_bar_uncached,
        graph,
        packet,
        direction,
    )
    deposit_typed_pullback!(
        packet_bar_cached,
        raw_bar_cached,
        graph,
        cache,
        packet,
        direction,
    )
    @test packet_bar_cached == packet_bar_uncached
    @test raw_bar_cached == raw_bar_uncached
    @test iszero(raw_bar_cached[2]) # silent opponent was skipped

    # A parameter write does not leak into an in-flight optimizer snapshot.
    stale_value = cached[Cell.input_index(1, Cell.INPUT_AMPA), 1]
    graph.raw_conductance[1] += 0.5f0
    fill!(cached, 0.0f0)
    deposit_typed!(cached, graph, cache, packet)
    @test cached[Cell.input_index(1, Cell.INPUT_AMPA), 1] == stale_value
    refresh_cache!(cache, graph)
    fill!(cached, 0.0f0)
    deposit_typed!(cached, graph, cache, packet)
    @test cached[Cell.input_index(1, Cell.INPUT_AMPA), 1] != stale_value
end

@testset "projected typed-input conductance homeostasis" begin
    source_count = 6
    fanout = 2
    slots = source_count * fanout
    fields = fill(UInt16(1), slots)
    polarities = repeat(Int8[1, -1], source_count)
    cells = Vector{UInt16}(undef, slots)
    compartments = fill(UInt8(1), slots)
    receptors = Vector{UInt8}(undef, slots)
    physical = Float64[
        0.001, 0.001,
        0.020, 0.200,
        0.080, 2.000,
        0.400, 0.005,
        2.000, 0.300,
        4.000, 3.000,
    ]
    for source in 1:source_count
        first = (source - 1) * fanout + 1
        cells[first] = UInt16(1)
        receptors[first] = AMPA_RECEPTOR
        if source <= 3
            cells[first + 1] = UInt16(1)
            receptors[first + 1] = NMDA_RECEPTOR
        else
            cells[first + 1] = UInt16(2)
            receptors[first + 1] = AMPA_RECEPTOR
        end
    end
    raw = TypedDendriticAfferents._inverse_softplus.(physical)
    graph = TypedAfferentGraph(
        source_count,
        UInt8[ANALOG_FIELD],
        2,
        fanout,
        fields,
        polarities,
        cells,
        compartments,
        receptors,
        raw,
    )
    cache = TypedAfferentCache(graph)
    topology = (
        copy(graph.source_field),
        copy(graph.source_polarity),
        copy(graph.destination_cell),
        copy(graph.destination_compartment),
        copy(graph.receptor),
        copy(graph.destination_input),
    )

    function grouped_physical(graph)
        groups = Dict{Tuple{UInt16,UInt8},Vector{Float64}}()
        for slot in 1:contact_count(graph)
            key = (graph.destination_cell[slot], graph.destination_input[slot])
            push!(get!(groups, key, Float64[]), conductance(graph.raw_conductance[slot]))
        end
        return groups
    end
    effective_contacts(values) = sum(values)^2 / sum(abs2, values)

    before = grouped_physical(graph)
    target = 0.2
    floor_ratio = 0.25
    ceiling_ratio = 2.0
    project_conductance_homeostasis!(
        cache,
        graph,
        target,
        floor_ratio,
        ceiling_ratio,
    )
    after = grouped_physical(graph)

    @test keys(after) == keys(before)
    for (key, values) in after
        @test sum(values) / length(values) ≈ target rtol=2e-14 atol=2e-14
        @test minimum(values) >= target * floor_ratio - 2e-15
        @test maximum(values) <= target * ceiling_ratio + 2e-15
        @test effective_contacts(values) >= length(values) / ceiling_ratio - 1e-12
        @test effective_contacts(values) > effective_contacts(before[key])
    end
    @test topology == (
        collect(graph.source_field),
        collect(graph.source_polarity),
        collect(graph.destination_cell),
        collect(graph.destination_compartment),
        collect(graph.receptor),
        collect(graph.destination_input),
    )
    @test cache.physical == conductance.(graph.raw_conductance)
    @test cache.derivative ==
          TypedDendriticAfferents._conductance_derivative.(graph.raw_conductance)

    # A second proximal step is an exact no-op on raw optimizer parameters.
    projected_raw = copy(graph.raw_conductance)
    project_conductance_homeostasis!(
        cache,
        graph,
        target,
        floor_ratio,
        ceiling_ratio,
    )
    @test graph.raw_conductance == projected_raw

    @test_throws ArgumentError project_conductance_homeostasis!(
        cache,
        graph,
        0.0,
        floor_ratio,
        ceiling_ratio,
    )
    @test_throws ArgumentError project_conductance_homeostasis!(
        cache,
        graph,
        target,
        0.0,
        ceiling_ratio,
    )
    @test_throws ArgumentError project_conductance_homeostasis!(
        cache,
        graph,
        target,
        floor_ratio,
        1.0,
    )

    # Compile and then exercise the actual projected path under @allocated.
    hot_graph = build_typed_afferents(
        64,
        UInt8[ANALOG_FIELD, HARD_BIT_FIELD],
        22;
        fanout=12,
        seed=0x70686f6d,
        initial_conductance=0.02,
    )
    hot_cache = TypedAfferentCache(hot_graph)
    project_conductance_homeostasis!(hot_cache, hot_graph, 0.02f0, 0.25f0, 2.0f0)
    hot_graph.raw_conductance[1] =
        TypedDendriticAfferents._inverse_softplus(1.0f0)
    project_conductance_homeostasis!(hot_cache, hot_graph, 0.02f0, 0.25f0, 2.0f0)
    hot_graph.raw_conductance[1] =
        TypedDendriticAfferents._inverse_softplus(1.0f0)
    @test @allocated(
        project_conductance_homeostasis!(
            hot_cache,
            hot_graph,
            0.02f0,
            0.25f0,
            2.0f0,
        ),
    ) == 0
end

@testset "singleton conductance homeostasis preserves learned scale" begin
    target = 0.2
    floor_ratio = 0.25
    ceiling_ratio = 2.0
    lower = target * floor_ratio
    upper = target * ceiling_ratio
    physical = Float64[0.06, 0.30, 0.31, 0.01, 0.70]
    graph = TypedAfferentGraph(
        5,
        UInt8[ANALOG_FIELD],
        2,
        1,
        fill(UInt16(1), 5),
        fill(Int8(1), 5),
        UInt16[1, 1, 1, 2, 2],
        fill(UInt8(1), 5),
        UInt8[
            AMPA_RECEPTOR,
            AMPA_RECEPTOR,
            NMDA_RECEPTOR,
            AMPA_RECEPTOR,
            NMDA_RECEPTOR,
        ],
        TypedDendriticAfferents._inverse_softplus.(physical),
    )
    cache = TypedAfferentCache(graph)
    learned_singleton_raw = graph.raw_conductance[3]

    project_conductance_homeostasis!(
        cache,
        graph,
        target,
        floor_ratio,
        ceiling_ratio,
    )

    # The two-contact group retains its relative learned structure while its
    # bounded mean is projected to the requested target.
    @test (cache.physical[1] + cache.physical[2]) / 2 ≈ target rtol=2e-14 atol=2e-14
    @test cache.physical[1] ≈ 0.08 rtol=2e-14 atol=2e-14
    @test cache.physical[2] ≈ 0.32 rtol=2e-14 atol=2e-14

    # An in-bound learned singleton is not reset to target_mean.
    @test graph.raw_conductance[3] == learned_singleton_raw
    @test cache.physical[3] ≈ 0.31 rtol=2e-14 atol=2e-14

    # Out-of-bound singletons still obey the common safety box.
    @test cache.physical[4] ≈ lower rtol=2e-14 atol=2e-14
    @test cache.physical[5] ≈ upper rtol=2e-14 atol=2e-14
    @test cache.physical == conductance.(graph.raw_conductance)

    # Compile the singleton path, perturb every path again, then prove the
    # actual projection remains allocation-free.
    graph.raw_conductance[1] = TypedDendriticAfferents._inverse_softplus(0.01)
    graph.raw_conductance[3] = TypedDendriticAfferents._inverse_softplus(0.27)
    graph.raw_conductance[4] = TypedDendriticAfferents._inverse_softplus(0.001)
    project_conductance_homeostasis!(
        cache,
        graph,
        target,
        floor_ratio,
        ceiling_ratio,
    )
    graph.raw_conductance[1] = TypedDendriticAfferents._inverse_softplus(0.01)
    graph.raw_conductance[3] = TypedDendriticAfferents._inverse_softplus(0.27)
    graph.raw_conductance[4] = TypedDendriticAfferents._inverse_softplus(0.001)
    @test @allocated(
        project_conductance_homeostasis!(
            cache,
            graph,
            target,
            floor_ratio,
            ceiling_ratio,
        ),
    ) == 0
    @test cache.physical[3] ≈ 0.27 rtol=2e-14 atol=2e-14
end

@testset "typed afferent hot path allocates zero bytes" begin
    rng = MersenneTwister(0xa110c)
    graph = build_typed_afferents(
        64,
        UInt8[ANALOG_FIELD, ANALOG_FIELD, HARD_BIT_FIELD],
        22;
        fanout=12,
        seed=0x7711,
    )
    packet = randn(rng, Float32, 3, 64)
    packet[3, :] .= rand(rng, Bool, 64)
    destination = zeros(Float32, Cell.INPUT_DIM, 22)
    direction = randn(rng, Float32, Cell.INPUT_DIM, 22)
    packet_bar = zeros(Float32, size(packet))
    raw_bar = zeros(Float32, contact_count(graph))
    sources = Int32[2, 7, 19, 41, 63]
    candidate = copy(packet)
    candidate[1:2, sources] .+= 0.2f0 .* randn(rng, Float32, 2, length(sources))
    candidate[3, sources] .= 1 .- packet[3, sources]
    candidate_bar = zeros(Float32, size(packet))
    base_bar = zeros(Float32, size(packet))
    cache = TypedAfferentCache(graph)

    deposit_typed!(destination, graph, packet)
    deposit_typed_pullback!(packet_bar, raw_bar, graph, packet, direction)
    deposit_sources!(destination, graph, packet, sources)
    deposit_sources_pullback!(packet_bar, raw_bar, graph, packet, sources, direction)
    deposit_delta!(destination, graph, candidate, packet, sources)
    deposit_delta_pullback!(
        candidate_bar,
        base_bar,
        raw_bar,
        graph,
        candidate,
        packet,
        sources,
        direction,
    )
    fill!(destination, 0)
    fill!(packet_bar, 0)
    fill!(candidate_bar, 0)
    fill!(base_bar, 0)
    fill!(raw_bar, 0)

    # Compile the cached canonical overloads before allocation assertions.
    deposit_typed!(destination, graph, cache, packet)
    deposit_typed_pullback!(
        packet_bar,
        raw_bar,
        graph,
        cache,
        packet,
        direction,
    )
    deposit_sources!(destination, graph, cache, packet, sources)
    deposit_sources_pullback!(
        packet_bar,
        raw_bar,
        graph,
        cache,
        packet,
        sources,
        direction,
    )
    deposit_delta!(destination, graph, cache, candidate, packet, sources)
    deposit_delta_pullback!(
        candidate_bar,
        base_bar,
        raw_bar,
        graph,
        cache,
        candidate,
        packet,
        sources,
        direction,
    )
    fill!(destination, 0)
    fill!(packet_bar, 0)
    fill!(candidate_bar, 0)
    fill!(base_bar, 0)
    fill!(raw_bar, 0)

    @test @allocated(deposit_typed!(destination, graph, packet)) == 0
    @test @allocated(
        deposit_typed_pullback!(packet_bar, raw_bar, graph, packet, direction),
    ) == 0
    @test @allocated(deposit_sources!(destination, graph, packet, sources)) == 0
    @test @allocated(
        deposit_sources_pullback!(
            packet_bar,
            raw_bar,
            graph,
            packet,
            sources,
            direction,
        ),
    ) == 0
    @test @allocated(
        deposit_delta!(destination, graph, candidate, packet, sources),
    ) == 0
    @test @allocated(
        deposit_delta_pullback!(
            candidate_bar,
            base_bar,
            raw_bar,
            graph,
            candidate,
            packet,
            sources,
            direction,
        ),
    ) == 0
    @test @allocated(deposit_typed!(destination, graph, cache, packet)) == 0
    @test @allocated(
        deposit_typed_pullback!(
            packet_bar,
            raw_bar,
            graph,
            cache,
            packet,
            direction,
        ),
    ) == 0
    @test @allocated(
        deposit_sources!(destination, graph, cache, packet, sources),
    ) == 0
    @test @allocated(
        deposit_sources_pullback!(
            packet_bar,
            raw_bar,
            graph,
            cache,
            packet,
            sources,
            direction,
        ),
    ) == 0
    @test @allocated(
        deposit_delta!(
            destination,
            graph,
            cache,
            candidate,
            packet,
            sources,
        ),
    ) == 0
    @test @allocated(
        deposit_delta_pullback!(
            candidate_bar,
            base_bar,
            raw_bar,
            graph,
            cache,
            candidate,
            packet,
            sources,
            direction,
        ),
    ) == 0
    @test @allocated(refresh_cache!(cache, graph)) == 0
end
