using Test

include(joinpath(@__DIR__, "OrderedMultiscaleTopology.jl"))
using .OrderedMultiscaleTopology

const Topology = OrderedMultiscaleTopology

@testset "ordered multiscale exact dimensions and indexing" begin
    topology = canonical_topology()
    @test SPATIAL_COUNT == 480
    @test ROW_INTERNAL_COUNT == 432
    @test COLUMN_INTERNAL_COUNT == 460
    @test MOTIF_COUNT == 32
    @test EVIDENCE_COUNT == 32
    @test OUTPUT_COUNT == 22
    @test NODE_COUNT == 1_458
    @test EDGE_COUNT == 2_472
    @test FULL_PACKET_WIDTH == 12
    @test SEMANTIC_OUTPUT_WIDTH == 3

    @test spatial_node(BEFORE_PLANE, 1, 1) == UInt16(1)
    @test spatial_node(BEFORE_PLANE, 24, 10) == UInt16(240)
    @test spatial_node(AFTER_PLANE, 1, 1) == UInt16(241)
    @test spatial_node(AFTER_PLANE, 24, 10) == UInt16(480)
    @test row_internal_node(BEFORE_PLANE, 1, 1) == UInt16(481)
    @test column_internal_node(BEFORE_PLANE, 1, 1) == UInt16(913)
    @test motif_node(1, 1) == UInt16(1_373)
    @test motif_node(8, 4) == UInt16(1_404)
    @test evidence_node(1) == UInt16(1_405)
    @test output_node(1) == UInt16(1_437)
    @test output_node(22) == UInt16(1_458)

    @test node_class(topology, spatial_node(AFTER_PLANE, 37)) == SPATIAL_CLASS
    @test node_plane(topology, spatial_node(AFTER_PLANE, 37)) == AFTER_PLANE
    @test spatial_position(spatial_node(AFTER_PLANE, 37)) == UInt16(37)
    @test node_class(topology, row_root_node(BEFORE_PLANE, 4)) ==
          ROW_INTERNAL_CLASS
    @test node_class(topology, column_root_node(AFTER_PLANE, 7)) ==
          COLUMN_INTERNAL_CLASS
    @test motif_family(topology, motif_node(6, 3)) == UInt8(6)
    @test motif_slot(topology, motif_node(6, 3)) == UInt8(3)
end
@testset "binary spine preserves complete ordered child packets" begin
    topology = canonical_topology()
    for plane in 1:PLANE_COUNT, row in 1:ROW_COUNT,
        internal in 1:ROW_INTERNAL_PER_ROW
        node = row_internal_node(plane, row, internal)
        @test child_count(topology, node) == 2
        @test child_slot(topology, node, 1) == UInt8(1)
        @test child_slot(topology, node, 2) == UInt8(2)
        left = child_node(topology, node, 1)
        right = child_node(topology, node, 2)
        @test edge_kind(topology, child_edge(topology, node, 1)) ==
              FULL_PACKET_EDGE
        @test edge_kind(topology, child_edge(topology, node, 2)) ==
              FULL_PACKET_EDGE
        @test edge_semantic_role(topology, child_edge(topology, node, 1)) == 0
        @test edge_semantic_role(topology, child_edge(topology, node, 2)) == 0
        @test left < node
        @test right < node
        @test left != right
    end
    for plane in 1:PLANE_COUNT, column in 1:COLUMN_COUNT,
        internal in 1:COLUMN_INTERNAL_PER_COLUMN
        node = column_internal_node(plane, column, internal)
        @test child_count(topology, node) == 2
        @test child_slot(topology, node, 1) == UInt8(1)
        @test child_slot(topology, node, 2) == UInt8(2)
        @test child_node(topology, node, 1) < node
        @test child_node(topology, node, 2) < node
    end

    for plane in 1:PLANE_COUNT, row in 1:ROW_COUNT
        root = row_root_node(plane, row)
        @test interval_first(topology, root) == UInt8(1)
        @test interval_last(topology, root) == UInt8(COLUMN_COUNT)
    end
    for plane in 1:PLANE_COUNT, column in 1:COLUMN_COUNT
        root = column_root_node(plane, column)
        @test interval_first(topology, root) == UInt8(1)
        @test interval_last(topology, root) == UInt8(ROW_COUNT)
    end

    for plane in 1:PLANE_COUNT, position in 1:SPATIAL_COUNT_PER_PLANE
        node = spatial_node(plane, position)
        @test child_count(topology, node) == 0
        @test parent_count(topology, node) == 2
        classes = Set(
            node_class(topology, parent_node(topology, node, index))
            for index in 1:2
        )
        @test classes == Set((ROW_INTERNAL_CLASS, COLUMN_INTERNAL_CLASS))
    end
end

@testset "semantic graph is balanced, ordered and hub-free" begin
    topology = canonical_topology()
    for node in vcat(
        [motif_node(family, slot) for family in 1:8 for slot in 1:4],
        [evidence_node(index) for index in 1:32],
        [output_node(index) for index in 1:22],
    )
        @test child_count(topology, node) == SEMANTIC_FANIN
        sources = UInt16[]
        for index in 1:SEMANTIC_FANIN
            edge = child_edge(topology, node, index)
            @test child_slot(topology, node, index) == UInt8(index)
            @test edge_kind(topology, edge) == PROJECTED_TRIPLET_EDGE
            @test edge_semantic_role(topology, edge) == UInt8(index)
            push!(sources, child_node(topology, node, index))
        end
        @test length(unique(sources)) == SEMANTIC_FANIN
    end

    # A motif compares matching before/after roots in four explicit roles.
    for family in 1:8, slot in 1:4
        node = motif_node(family, slot)
        @test node_plane(topology, child_node(topology, node, 1)) == BEFORE_PLANE
        @test node_plane(topology, child_node(topology, node, 2)) == AFTER_PLANE
        @test node_plane(topology, child_node(topology, node, 3)) == BEFORE_PLANE
        @test node_plane(topology, child_node(topology, node, 4)) == AFTER_PLANE
        @test node_plane(topology, child_node(topology, node, 5)) == BEFORE_PLANE
        @test node_plane(topology, child_node(topology, node, 6)) == AFTER_PLANE
        @test node_plane(topology, child_node(topology, node, 7)) == BEFORE_PLANE
        @test node_plane(topology, child_node(topology, node, 8)) == AFTER_PLANE
    end

    # Evidence signatures are all different and every motif fans out exactly 8.
    signatures = Set{NTuple{8,UInt16}}()
    for evidence in 1:EVIDENCE_COUNT
        node = evidence_node(evidence)
        signature = ntuple(index -> child_node(topology, node, index), 8)
        push!(signatures, signature)
        @test ntuple(
            index -> motif_family(topology, signature[index]),
            8,
        ) == ntuple(UInt8, 8)
    end
    @test length(signatures) == EVIDENCE_COUNT
    for family in 1:8, slot in 1:4
        @test parent_count(topology, motif_node(family, slot)) == 8
    end

    evidence_fanout = [parent_count(topology, evidence_node(index)) for index in 1:32]
    @test minimum(evidence_fanout) == 5
    @test maximum(evidence_fanout) == 6

    row_root_fanout = Int[]
    for plane in 1:2, row in 1:24
        push!(row_root_fanout, parent_count(topology, row_root_node(plane, row)))
    end
    column_root_fanout = Int[]
    for plane in 1:2, column in 1:10
        push!(column_root_fanout,
              parent_count(topology, column_root_node(plane, column)))
    end
    @test extrema(row_root_fanout) == (2, 3)
    @test extrema(column_root_fanout) == (6, 7)
    @test maximum(child_count(topology, node) for node in 1:NODE_COUNT) == 8
end

@testset "bidirectional adjacency and exact topological order" begin
    topology = canonical_topology()
    incoming_seen = falses(EDGE_COUNT)
    outgoing_seen = falses(EDGE_COUNT)
    for node in 1:NODE_COUNT
        for index in 1:child_count(topology, node)
            edge = Int(child_edge(topology, node, index))
            @test !incoming_seen[edge]
            incoming_seen[edge] = true
            @test edge_destination(topology, edge) == UInt16(node)
            @test edge_source(topology, edge) < UInt16(node)
        end
        for index in 1:parent_count(topology, node)
            edge = Int(parent_edge(topology, node, index))
            @test !outgoing_seen[edge]
            outgoing_seen[edge] = true
            @test edge_source(topology, edge) == UInt16(node)
            @test parent_node(topology, node, index) > UInt16(node)
        end
    end
    @test all(incoming_seen)
    @test all(outgoing_seen)
    @test validate_topology(topology) === topology
end

function allocating_oracle(topology, seeds)
    marked = falses(NODE_COUNT)
    for seed in seeds
        marked[Int(seed)] = true
    end
    for node in 1:NODE_COUNT
        marked[node] || continue
        for index in 1:parent_count(topology, node)
            marked[Int(parent_node(topology, node, index))] = true
        end
    end
    return UInt16[node for node in 1:NODE_COUNT if marked[node]]
end

@testset "affected ancestor closure is exact and allocation free" begin
    topology = canonical_topology()
    closure = AffectedClosure()
    seeds = UInt16[
        spatial_node(AFTER_PLANE, 1, 1),
        spatial_node(AFTER_PLANE, 24, 10),
        spatial_node(AFTER_PLANE, 1, 1),
    ]
    expected = allocating_oracle(topology, seeds)
    fill_affected_closure!(closure, topology, seeds)
    @test collect(closure) == expected
    @test issorted(closure)
    @test length(unique(closure)) == length(closure)
    @test affected_forward_node(closure, 1) == first(expected)
    @test affected_reverse_node(closure, 1) == last(expected)

    positions = UInt16[1, 240, 1]
    fill_spatial_affected_closure!(
        closure,
        topology,
        AFTER_PLANE,
        positions,
    )
    @test collect(closure) == expected

    fill_affected_closure!(closure, topology, UInt16[])
    @test affected_count(closure) == 0

    # Warm every method before measuring its fixed-capacity hot path.
    fill_affected_closure!(closure, topology, seeds)
    @test @allocated(fill_affected_closure!(closure, topology, seeds)) == 0
    fill_spatial_affected_closure!(closure, topology, AFTER_PLANE, positions)
    @test @allocated(
        fill_spatial_affected_closure!(
            closure,
            topology,
            AFTER_PLANE,
            positions,
        )
    ) == 0
end

@testset "ordered topology bounds fail closed" begin
    topology = canonical_topology()
    closure = AffectedClosure()
    @test_throws BoundsError spatial_node(0, 1)
    @test_throws BoundsError spatial_node(3, 1)
    @test_throws BoundsError spatial_node(1, 0)
    @test_throws BoundsError spatial_node(1, 241)
    @test_throws BoundsError row_internal_node(1, 0, 1)
    @test_throws BoundsError row_internal_node(1, 1, 10)
    @test_throws BoundsError column_internal_node(1, 11, 1)
    @test_throws BoundsError column_internal_node(1, 1, 24)
    @test_throws BoundsError motif_node(0, 1)
    @test_throws BoundsError motif_node(1, 5)
    @test_throws BoundsError evidence_node(33)
    @test_throws BoundsError output_node(23)
    @test_throws BoundsError child_count(topology, 0)
    @test_throws BoundsError parent_count(topology, NODE_COUNT + 1)
    @test_throws BoundsError edge_source(topology, 0)
    @test_throws BoundsError edge_kind(topology, EDGE_COUNT + 1)
    @test_throws BoundsError child_edge(topology, spatial_node(1, 1), 1)
    @test_throws ArgumentError spatial_position(motif_node(1, 1))
    @test_throws ArgumentError motif_family(topology, evidence_node(1))
    @test_throws BoundsError fill_affected_closure!(
        closure,
        topology,
        UInt16[1],
        2,
    )
    @test_throws BoundsError fill_spatial_affected_closure!(
        closure,
        topology,
        AFTER_PLANE,
        UInt16[241],
        1,
    )
end
