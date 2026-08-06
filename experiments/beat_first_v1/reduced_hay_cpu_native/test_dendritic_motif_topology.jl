using Test
using Random

module DendriticMotifTopologyTestHarness
include(joinpath(@__DIR__, "DendriticMotifTopology.jl"))
end

const Motif =
    DendriticMotifTopologyTestHarness.DendriticMotifTopology

@inline pair_even(value::Int) = isodd(value) ? value + 1 : value - 1
@inline tile_id(band::Int, half::Int) = UInt8(34 + 2 * (band - 1) + half)

function expected_destinations(source::Int)
    if source <= 24
        row = source
        band = div(row - 1, 6) + 1
        return UInt8[
            row,
            pair_even(row),
            tile_id(band, 1),
            tile_id(band, 2),
        ]
    elseif source <= 34
        column = source - 24
        half = div(column - 1, 5) + 1
        return UInt8[
            24 + column,
            24 + pair_even(column),
            tile_id(mod(column - 1, 4) + 1, half),
            tile_id(mod(column + 1, 4) + 1, half),
        ]
    elseif source <= 42
        tile = source - 34
        band = div(tile - 1, 2) + 1
        half = mod(tile - 1, 2) + 1
        return UInt8[
            34 + tile,
            tile_id(band, 3 - half),
            6 * band - 3,
            24 + 5 * half - 2,
        ]
    end
    stripe = source - 42
    return UInt8[
        42 + stripe,
        42 + pair_even(stripe),
        tile_id(mod(stripe - 1, 4) + 1, mod(stripe - 1, 2) + 1),
        tile_id(mod(stripe + 1, 4) + 1, mod(stripe, 2) + 1),
    ]
end

function naive_closure(topology, sources)
    motifs = Set{UInt8}()
    for source in sources
        for rank in 1:Motif.SOURCE_FANOUT
            push!(motifs, Motif.motif_destination(topology, source, rank))
        end
    end
    return sort!(collect(motifs))
end

function closure_allocation(closure, topology, sources, count)
    Motif.fill_affected_motif_closure!(closure, topology, sources, count)
    return @allocated Motif.fill_affected_motif_closure!(
        closure,
        topology,
        sources,
        count,
    )
end

@testset "fixed 24+10+8+6 motif atlas" begin
    topology = Motif.canonical_topology()
    @test Motif.RELATION_SOURCE_COUNT == 48
    @test Motif.MOTIF_COUNT == 48
    @test Motif.SOURCE_FANOUT == 4
    @test Motif.CONTACT_COUNT == 192
    @test Motif.VERTICAL_MOTIF_FIRST == 1
    @test Motif.VERTICAL_MOTIF_COUNT == 24
    @test Motif.WELL_MOTIF_FIRST == 25
    @test Motif.WELL_MOTIF_COUNT == 10
    @test Motif.GEOMETRY_MOTIF_FIRST == 35
    @test Motif.GEOMETRY_MOTIF_COUNT == 8
    @test Motif.ACTION_MOTIF_FIRST == 43
    @test Motif.ACTION_MOTIF_COUNT == 6
    @test !Base.ismutabletype(typeof(topology))
    @test all(
        field -> field isa Memory,
        getfield.(Ref(topology), fieldnames(typeof(topology))),
    )
    @test Motif.canonical_topology() === topology
    @test Motif.validate_topology(topology) === nothing

    @test Motif.vertical_motif.(1:24) == UInt8.(1:24)
    @test Motif.well_motif.(1:10) == UInt8.(25:34)
    @test Motif.geometry_motif.(1:8) == UInt8.(35:42)
    @test Motif.action_motif.(1:6) == UInt8.(43:48)
    @test Motif.motif_class.(1:48) == vcat(
        fill(Motif.VERTICAL_MOTIF, 24),
        fill(Motif.WELL_MOTIF, 10),
        fill(Motif.GEOMETRY_MOTIF, 8),
        fill(Motif.ACTION_MOTIF, 6),
    )
    @test Motif.motif_slot.(1:48) == vcat(
        UInt8.(1:24),
        UInt8.(1:10),
        UInt8.(1:8),
        UInt8.(1:6),
    )
end

@testset "exact four-destination semantic mapping" begin
    topology = Motif.canonical_topology()
    observed = falses(Motif.RELATION_SOURCE_COUNT, Motif.MOTIF_COUNT)
    for source in 1:Motif.RELATION_SOURCE_COUNT
        expected = expected_destinations(source)
        actual = UInt8[
            Motif.motif_destination(topology, source, rank)
            for rank in 1:Motif.SOURCE_FANOUT
        ]
        @test actual == expected
        @test length(unique(actual)) == Motif.SOURCE_FANOUT
        @test all(1 .<= actual .<= Motif.MOTIF_COUNT)
        @test Motif.motif_destination.(source, 1:4) == actual
        for destination in actual
            @test !observed[source, Int(destination)]
            observed[source, Int(destination)] = true
        end
    end
    @test count(observed) == Motif.CONTACT_COUNT

    # Explicit boundary examples make accidental zero/one-based changes loud.
    @test expected_destinations(1) == UInt8[1, 2, 35, 36]
    @test expected_destinations(24) == UInt8[24, 23, 41, 42]
    @test expected_destinations(25) == UInt8[25, 26, 35, 39]
    @test expected_destinations(34) == UInt8[34, 33, 38, 42]
    @test expected_destinations(35) == UInt8[35, 36, 3, 27]
    @test expected_destinations(42) == UInt8[42, 41, 21, 32]
    @test expected_destinations(43) == UInt8[43, 44, 35, 40]
    @test expected_destinations(48) == UInt8[48, 47, 38, 41]
end

@testset "incoming adjacency is exact, sorted, and deterministic" begin
    topology = Motif.canonical_topology()
    total = 0
    for motif in 1:Motif.MOTIF_COUNT
        expected = UInt8[
            source for source in 1:Motif.RELATION_SOURCE_COUNT if
            UInt8(motif) in expected_destinations(source)
        ]
        observed = UInt8[
            Motif.incoming_source(topology, motif, index)
            for index in 1:Motif.incoming_source_count(topology, motif)
        ]
        @test observed == expected
        @test issorted(observed)
        @test length(unique(observed)) == length(observed)
        @test Motif.incoming_source_count(motif) == length(observed)
        @test Motif.incoming_source.(motif, 1:length(observed)) == observed
        total += length(observed)
    end
    @test total == Motif.CONTACT_COUNT
end

@testset "affected closure is exact, ascending, and allocation-free" begin
    topology = Motif.canonical_topology()
    closure = Motif.AffectedMotifClosure()

    sources = UInt8[48, 1, 35, 1, 25, 42, 17, 6]
    expected = naive_closure(topology, sources)
    returned = Motif.fill_affected_motif_closure!(
        closure,
        topology,
        sources,
        length(sources),
    )
    @test returned === closure
    @test collect(closure) == expected
    @test Motif.motif_count(closure) == length(expected)
    @test issorted(collect(closure))
    @test Motif.motif_mask(closure) == foldl(
        (mask, motif) -> mask | (UInt64(1) << (Int(motif) - 1)),
        expected;
        init = UInt64(0),
    )
    @test all(
        Motif.motif_is_affected(closure, motif) ==
        (UInt8(motif) in expected) for motif in 1:Motif.MOTIF_COUNT
    )

    source_mask = foldl(
        (mask, source) -> mask | (UInt64(1) << (Int(source) - 1)),
        sources;
        init = UInt64(0),
    )
    Motif.fill_affected_motif_closure!(closure, topology, source_mask)
    @test collect(closure) == expected
    Motif.fill_affected_motif_closure!(closure, source_mask)
    @test collect(closure) == expected

    Motif.fill_affected_motif_closure!(closure, topology, sources, 0)
    @test isempty(closure)
    @test Motif.motif_count(closure) == 0
    @test Motif.motif_mask(closure) == 0

    all_sources = Memory{UInt8}(UInt8.(1:Motif.RELATION_SOURCE_COUNT))
    Motif.fill_affected_motif_closure!(closure, topology, all_sources)
    @test collect(closure) == UInt8.(1:Motif.MOTIF_COUNT)
    @test Motif.motif_mask(closure) ==
          (UInt64(1) << Motif.MOTIF_COUNT) - UInt64(1)

    hot_sources = Memory{UInt8}(UInt8[1, 25, 35, 43, 48, 12, 33, 39])
    @test closure_allocation(
        closure,
        topology,
        hot_sources,
        length(hot_sources),
    ) == 0

    rng = MersenneTwister(0x4d4f544946)
    for _ in 1:200
        count = rand(rng, 0:48)
        shuffled = UInt8.(randperm(rng, 48))
        selected = @view shuffled[1:count]
        expected_random = naive_closure(topology, selected)
        Motif.fill_affected_motif_closure!(closure, topology, selected)
        @test collect(closure) == expected_random
    end
end

@testset "bounds and malformed-mask checks fail closed" begin
    topology = Motif.canonical_topology()
    closure = Motif.AffectedMotifClosure()
    @test_throws BoundsError Motif.motif_destination(topology, 0, 1)
    @test_throws BoundsError Motif.motif_destination(topology, 49, 1)
    @test_throws BoundsError Motif.motif_destination(topology, 1, 0)
    @test_throws BoundsError Motif.motif_destination(topology, 1, 5)
    @test_throws BoundsError Motif.motif_class(0)
    @test_throws BoundsError Motif.motif_slot(49)
    @test_throws BoundsError Motif.vertical_motif(25)
    @test_throws BoundsError Motif.well_motif(0)
    @test_throws BoundsError Motif.geometry_motif(9)
    @test_throws BoundsError Motif.action_motif(7)
    @test_throws BoundsError Motif.incoming_source(topology, 1, 0)
    @test_throws BoundsError Motif.affected_motif(closure, 1)
    @test_throws BoundsError Motif.motif_is_affected(closure, 0)
    @test_throws BoundsError Motif.fill_affected_motif_closure!(
        closure,
        topology,
        UInt8[1, 2],
        3,
    )
    @test_throws BoundsError Motif.fill_affected_motif_closure!(
        closure,
        topology,
        UInt8[0],
    )
    @test_throws ArgumentError Motif.fill_affected_motif_closure!(
        closure,
        topology,
        UInt64(1) << 48,
    )
end
