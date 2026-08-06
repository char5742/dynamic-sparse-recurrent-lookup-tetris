using Test
using Random

module DendriticRelationTopologyTestHarness
include(joinpath(@__DIR__, "DendriticRelationTopology.jl"))
end

const Relation =
    DendriticRelationTopologyTestHarness.DendriticRelationTopology

@inline function coordinates(position::Integer)
    column = div(Int(position) - 1, Relation.ROW_COUNT) + 1
    row = Int(position) - (column - 1) * Relation.ROW_COUNT
    return row, column
end

@inline function layout_accepts(layout, row::Int, column::Int)
    if layout.kind == Relation.TILE_LOCAL_LAYOUT
        return Int(layout.row_first) <= row <= Int(layout.row_last) &&
               Int(layout.column_first) <= column <=
                   Int(layout.column_last)
    end
    value = Int(layout.row_stride) * (row - 1) +
            Int(layout.column_stride) * (column - 1)
    return mod(value, Int(layout.modulus)) == Int(layout.phase)
end

function naive_affected(topology, plane, positions)
    relations = Set{UInt8}()
    for position in positions
        source = Relation.source_id(plane, position)
        for fanout_index in 1:Relation.source_contact_count(topology, source)
            push!(
                relations,
                Relation.contact_relation(
                    Relation.source_contact(topology, source, fanout_index),
                ),
            )
        end
    end
    return relations
end

function closure_allocation(closure, topology, positions, count)
    Relation.fill_affected_relation_closure!(
        closure,
        topology,
        Relation.AFTER_PLANE,
        positions,
        count,
    )
    return @allocated Relation.fill_affected_relation_closure!(
        closure,
        topology,
        Relation.AFTER_PLANE,
        positions,
        count,
    )
end

function access_allocation(topology, source, relation)
    total = UInt32(0)
    for index in 1:Relation.source_contact_count(topology, source)
        contact = Relation.source_contact(topology, source, index)
        total += UInt32(Relation.contact_position(contact))
        total += UInt32(Relation.contact_relation(contact))
    end
    for index in 1:Relation.incoming_contact_count(topology, relation)
        contact = Relation.incoming_contact(topology, relation, index)
        total += UInt32(Relation.contact_local_role(contact))
    end
    return @allocated begin
        repeated = UInt32(0)
        for index in 1:Relation.source_contact_count(topology, source)
            contact = Relation.source_contact(topology, source, index)
            repeated += UInt32(Relation.contact_position(contact))
            repeated += UInt32(Relation.contact_relation(contact))
        end
        for index in 1:Relation.incoming_contact_count(topology, relation)
            contact = Relation.incoming_contact(topology, relation, index)
            repeated += UInt32(Relation.contact_local_role(contact))
        end
        repeated == total || error("access checksum drift")
    end
end

@testset "fixed 24+10+14 relation contract" begin
    topology = Relation.canonical_topology()
    @test Relation.ROW_COUNT == 24
    @test Relation.COLUMN_COUNT == 10
    @test Relation.LEAF_COUNT == 240
    @test Relation.PLANE_COUNT == 2
    @test Relation.SOURCE_COUNT == 480
    @test Relation.ROW_RELATION_COUNT == 24
    @test Relation.COLUMN_RELATION_COUNT == 10
    @test Relation.CROSS_ACTION_RELATION_COUNT == 14
    @test Relation.RELATION_COUNT == 48
    @test Relation.SOURCE_FANOUT == 4
    @test Relation.CONTACT_COUNT == 1_920
    @test isbitstype(Relation.LeafContactSpec)
    @test isbitstype(Relation.RelationLayout)
    @test !Base.ismutabletype(typeof(topology))
    @test all(
        field -> field isa Memory,
        getfield.(Ref(topology), fieldnames(typeof(topology))),
    )
    @test length(topology.source_offsets) == Relation.SOURCE_COUNT + 1
    @test length(topology.contacts) == Relation.CONTACT_COUNT
    @test length(topology.incoming_offsets) == Relation.RELATION_COUNT + 1
    @test length(topology.incoming_contact_ids) == Relation.CONTACT_COUNT

    @test count(
        relation ->
            Relation.relation_class(topology, relation) ==
            Relation.ROW_LOCAL_RELATION,
        1:Relation.RELATION_COUNT,
    ) == 24
    @test count(
        relation ->
            Relation.relation_class(topology, relation) ==
            Relation.COLUMN_RELATION,
        1:Relation.RELATION_COUNT,
    ) == 10
    @test count(
        relation ->
            Relation.relation_class(topology, relation) ==
            Relation.CROSS_ACTION_RELATION,
        1:Relation.RELATION_COUNT,
    ) == 14
    @test all(
        Relation.relation_slot(topology, Relation.row_relation(row)) == row
        for row in 1:Relation.ROW_COUNT
    )
    @test all(
        Relation.relation_slot(
            topology,
            Relation.column_relation(column),
        ) == column
        for column in 1:Relation.COLUMN_COUNT
    )
    @test all(
        Relation.relation_slot(
            topology,
            Relation.cross_action_relation(slot),
        ) == slot
        for slot in 1:Relation.CROSS_ACTION_RELATION_COUNT
    )
end

@testset "one fixed balanced cross/action layout table" begin
    @test length(Relation.RELATION_LAYOUTS) == 14
    @test count(
        layout -> layout.kind == Relation.TILE_LOCAL_LAYOUT,
        Relation.RELATION_LAYOUTS,
    ) == 8
    @test count(
        layout -> layout.kind == Relation.SMALL_WORLD_LAYOUT,
        Relation.RELATION_LAYOUTS,
    ) == 6

    coverage = zeros(Int, Relation.CROSS_ACTION_RELATION_COUNT)
    for position in 1:Relation.LEAF_COUNT
        row, column = coordinates(position)
        matching_tile_local = Int[]
        matching_small_world = Int[]
        for slot in 1:Relation.CROSS_ACTION_RELATION_COUNT
            layout = Relation.relation_layout(slot)
            if layout_accepts(layout, row, column)
                coverage[slot] += 1
                if layout.kind == Relation.TILE_LOCAL_LAYOUT
                    push!(matching_tile_local, slot)
                else
                    push!(matching_small_world, slot)
                end
            end
        end
        @test length(matching_tile_local) == 1
        @test length(matching_small_world) == 1
    end
    @test coverage[1:8] == fill(30, 8)
    @test coverage[9:14] == fill(40, 6)
end

@testset "source-major contacts preserve plane and absolute position" begin
    topology = Relation.canonical_topology()
    for plane in (Relation.BEFORE_PLANE, Relation.AFTER_PLANE)
        for position in 1:Relation.LEAF_COUNT
            source = Relation.source_id(plane, position)
            @test Relation.source_plane(source) == plane
            @test Relation.source_position(source) == position
            @test Relation.source_contact_count(topology, source) == 4
            row, column = coordinates(position)
            contacts = ntuple(4) do fanout_index
                Relation.source_contact(topology, source, fanout_index)
            end

            @test all(contact.source == source for contact in contacts)
            @test Relation.contact_source_slot.(contacts) ==
                  (UInt8(1), UInt8(2), UInt8(3), UInt8(4))
            @test Relation.contact_id.(contacts) == ntuple(4) do slot
                UInt16((Int(source) - 1) * Relation.SOURCE_FANOUT + slot)
            end
            @test all(
                Relation.contact_position(contact) == position
                for contact in contacts
            )
            @test all(
                Relation.contact_plane(contact) == plane
                for contact in contacts
            )
            @test length(unique(Relation.contact_relation.(contacts))) == 4
            @test Relation.contact_relation(contacts[1]) ==
                  Relation.row_relation(row)
            @test Relation.contact_relation(contacts[2]) ==
                  Relation.column_relation(column)
            @test Relation.relation_class(
                topology,
                Relation.contact_relation(contacts[3]),
            ) == Relation.CROSS_ACTION_RELATION
            @test Relation.relation_class(
                topology,
                Relation.contact_relation(contacts[4]),
            ) == Relation.CROSS_ACTION_RELATION

            @test all(
                Relation.contact_basal_compartment(contact) in
                    1:Relation.BASAL_COMPARTMENT_COUNT
                for contact in contacts
            )

            tile_slot = Int(
                Relation.relation_slot(
                    topology,
                    Relation.contact_relation(contacts[3]),
                ),
            )
            stripe_slot = Int(
                Relation.relation_slot(
                    topology,
                    Relation.contact_relation(contacts[4]),
                ),
            )
            @test Relation.relation_layout(tile_slot).kind ==
                  Relation.TILE_LOCAL_LAYOUT
            @test Relation.relation_layout(stripe_slot).kind ==
                  Relation.SMALL_WORLD_LAYOUT
        end
    end
end

@testset "relation fanin is balanced and duplicate-free" begin
    topology = Relation.canonical_topology()
    observed_contacts = 0
    for relation in 1:Relation.RELATION_COUNT
        relation_class = Relation.relation_class(topology, relation)
        relation_slot = Int(Relation.relation_slot(topology, relation))
        expected_fanin = if relation_class == Relation.ROW_LOCAL_RELATION
            2 * Relation.COLUMN_COUNT
        elseif relation_class == Relation.COLUMN_RELATION
            2 * Relation.ROW_COUNT
        else
            layout = Relation.relation_layout(relation_slot)
            layout.kind == Relation.TILE_LOCAL_LAYOUT ? 60 : 80
        end
        @test Relation.incoming_contact_count(topology, relation) ==
              expected_fanin
        # No cell is a dense global root over all 480 semantic-plane leaves.
        @test expected_fanin < Relation.SOURCE_COUNT

        sources = Set{UInt16}()
        identities = Set{Tuple{UInt8,UInt16}}()
        for incoming_index in 1:expected_fanin
            contact = Relation.incoming_contact(
                topology,
                relation,
                incoming_index,
            )
            @test Relation.contact_relation(contact) == relation
            @test !(contact.source in sources)
            @test !(
                (
                    Relation.contact_plane(contact),
                    Relation.contact_position(contact),
                ) in identities
            )
            push!(sources, contact.source)
            push!(
                identities,
                (
                    Relation.contact_plane(contact),
                    Relation.contact_position(contact),
                ),
            )
        end
        observed_contacts += expected_fanin
    end
    @test observed_contacts == Relation.CONTACT_COUNT
end

@testset "all relation families use the eight basal compartments" begin
    topology = Relation.canonical_topology()
    for relation in 1:Relation.RELATION_COUNT
        relation_class = Relation.relation_class(topology, relation)
        relation_slot = Int(Relation.relation_slot(topology, relation))
        counts = zeros(Int, Relation.PLANE_COUNT, Relation.BASAL_COMPARTMENT_COUNT)
        for incoming_index in
            1:Relation.incoming_contact_count(topology, relation)
            contact = Relation.incoming_contact(
                topology,
                relation,
                incoming_index,
            )
            counts[
                Int(Relation.contact_plane(contact)),
                Int(Relation.contact_basal_compartment(contact)),
            ] += 1
        end
        for plane in 1:Relation.PLANE_COUNT
            plane_counts = @view counts[plane, :]
            @test all(>(0), plane_counts)
            if relation_class == Relation.ROW_LOCAL_RELATION
                @test sum(plane_counts) == 10
                @test maximum(plane_counts) - minimum(plane_counts) <= 1
            elseif relation_class == Relation.COLUMN_RELATION
                @test plane_counts == fill(3, 8)
            elseif Relation.relation_layout(relation_slot).kind ==
                   Relation.TILE_LOCAL_LAYOUT
                @test sum(plane_counts) == 30
                @test maximum(plane_counts) - minimum(plane_counts) <= 1
            else
                @test plane_counts == fill(5, 8)
            end
        end
    end
end

@testset "affected relation closure is exact" begin
    topology = Relation.canonical_topology()
    closure = Relation.AffectedRelationClosure()
    cases = (
        UInt16[],
        UInt16[1],
        UInt16[240],
        UInt16[97, 97],
        UInt16[1, 6, 7, 24, 25, 240],
        UInt16.(1:Relation.LEAF_COUNT),
    )
    for plane in (Relation.BEFORE_PLANE, Relation.AFTER_PLANE)
        for positions in cases
            Relation.fill_affected_relation_closure!(
                closure,
                topology,
                plane,
                positions,
            )
            expected = naive_affected(topology, plane, positions)
            @test Set(closure) == expected
            @test Relation.affected_count(closure) == length(expected)
            @test issorted(collect(closure))
            @test all(
                Relation.relation_is_affected(closure, relation) ==
                    (UInt8(relation) in expected)
                for relation in 1:Relation.RELATION_COUNT
            )
        end
    end

    all_positions = UInt16.(1:Relation.LEAF_COUNT)
    Relation.fill_affected_relation_closure!(
        closure,
        topology,
        Relation.AFTER_PLANE,
        all_positions,
    )
    @test Relation.affected_count(closure) == Relation.RELATION_COUNT
    @test Relation.affected_mask(closure) ==
          (UInt64(1) << Relation.RELATION_COUNT) - UInt64(1)

    rng = MersenneTwister(0x52454c41)
    positions = zeros(UInt16, Relation.LEAF_COUNT)
    for _ in 1:128
        count = rand(rng, 0:Relation.LEAF_COUNT)
        for index in 1:count
            positions[index] = UInt16(rand(rng, 1:Relation.LEAF_COUNT))
        end
        Relation.fill_affected_relation_closure!(
            closure,
            topology,
            Relation.AFTER_PLANE,
            positions,
            count,
        )
        @test Set(closure) == naive_affected(
            topology,
            Relation.AFTER_PLANE,
            @view(positions[1:count]),
        )
    end
end

@testset "bounds fail closed" begin
    topology = Relation.canonical_topology()
    closure = Relation.AffectedRelationClosure()
    @test_throws BoundsError Relation.source_id(0, 1)
    @test_throws BoundsError Relation.source_id(3, 1)
    @test_throws BoundsError Relation.source_id(1, 0)
    @test_throws BoundsError Relation.source_id(1, 241)
    @test_throws BoundsError Relation.source_plane(0)
    @test_throws BoundsError Relation.source_position(481)
    @test_throws BoundsError Relation.row_relation(0)
    @test_throws BoundsError Relation.column_relation(11)
    @test_throws BoundsError Relation.cross_action_relation(15)
    @test_throws BoundsError Relation.relation_layout(0)
    @test_throws BoundsError Relation.source_contact(topology, 1, 0)
    @test_throws BoundsError Relation.source_contact(topology, 1, 5)
    @test_throws BoundsError Relation.incoming_contact(topology, 0, 1)
    @test_throws BoundsError Relation.incoming_contact(topology, 1, 21)
    @test_throws BoundsError Relation.relation_is_affected(closure, 49)
    @test_throws BoundsError Relation.fill_affected_relation_closure!(
        closure,
        topology,
        Relation.AFTER_PLANE,
        UInt16[1],
        2,
    )
    @test_throws BoundsError Relation.fill_affected_relation_closure!(
        closure,
        topology,
        0,
        UInt16[1],
    )
end

@testset "hot topology access and closure allocate nothing" begin
    topology = Relation.canonical_topology()
    closure = Relation.AffectedRelationClosure()
    positions = zeros(UInt16, Relation.LEAF_COUNT)
    positions[1:8] .= UInt16[1, 2, 6, 25, 97, 144, 239, 240]
    @test closure_allocation(closure, topology, positions, 8) == 0
    @test access_allocation(
        topology,
        Relation.source_id(Relation.AFTER_PLANE, 97),
        Relation.cross_action_relation(1),
    ) == 0
end
