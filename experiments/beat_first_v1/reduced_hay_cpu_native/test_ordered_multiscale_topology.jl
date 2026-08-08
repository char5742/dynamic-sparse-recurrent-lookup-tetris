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
    @test EDGE_COUNT == 2_216
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

@testset "static semantic graph is balanced, ordered and hub-free" begin
    topology = canonical_topology()
    for family in 1:8, slot in 1:4
        node = motif_node(family, slot)
        @test child_count(topology, node) == 0
        @test parent_count(topology, node) == 8
    end
    for node in vcat(
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

    @test all(
        parent_count(topology, row_root_node(plane, row)) == 0
        for plane in 1:2 for row in 1:24
    )
    @test all(
        parent_count(topology, column_root_node(plane, column)) == 0
        for plane in 1:2 for column in 1:10
    )
    @test maximum(child_count(topology, node) for node in 1:NODE_COUNT) == 8
end

function motif_fixture_context(;
    positions=(UInt16(24), UInt16(48), UInt16(0), UInt16(0)),
    count=2,
    clear_last=false,
    hold=UInt8(1),
    next=(UInt8(2), UInt8(3), UInt8(4), UInt8(5), UInt8(6)),
    ren=Int32(0x01020304),
    b2b=UInt8(1),
    tspin=UInt8(2),
)
    if clear_last
        mu = ntuple(row -> row == 24 ? UInt8(0) : UInt8(row + 1), 24)
        mask = UInt32(1) << 23
        cleared = 1
    else
        mu = ntuple(row -> UInt8(row), 24)
        mask = UInt32(0)
        cleared = 0
    end
    return CandidateMotifContext(
        positions,
        count,
        mu,
        mask,
        cleared,
        hold,
        next,
        ren,
        b2b,
        tspin,
    )
end

function motif_kinds(incidence, motif)
    return [
        motif_source(incidence, motif, rank).kind
        for rank in 1:motif_source_count(incidence, motif)
    ]
end

function motif_branches(incidence, motif)
    return [
        motif_source(incidence, motif, rank).branch_slot
        for rank in 1:motif_source_count(incidence, motif)
    ]
end

@testset "candidate semantic incidence realizes all eight motif families" begin
    topology = canonical_topology()
    incidence = CandidateMotifIncidence()
    context = motif_fixture_context()
    fill_candidate_motif_incidence!(incidence, topology, context)

    @test all(1 <= motif_source_count(incidence, motif) <= 8 for motif in 1:32)
    for motif in 1:32
        branches = motif_branches(incidence, motif)
        @test length(unique(branches)) == length(branches)
        @test all(branch -> 1 <= branch <= 8, branches)
    end

    # Families 1--6 use the same canonical placement-slot identity but have
    # distinct typed source contracts.
    @test motif_kinds(incidence, 1) == UInt8[
        MOTIF_SPATIAL_SOURCE,
        MOTIF_SPATIAL_SOURCE,
        MOTIF_RAW_PLACEMENT_SOURCE,
        MOTIF_ROW_REMAP_SOURCE,
    ]
    @test motif_kinds(incidence, 5) == UInt8[
        MOTIF_RAW_PLACEMENT_SOURCE,
        MOTIF_OUTSIDE_SOURCE,
        MOTIF_SPATIAL_SOURCE,
        MOTIF_ROW_REMAP_SOURCE,
    ]
    @test motif_kinds(incidence, 9) == UInt8[
        MOTIF_ROW_ROOT_SOURCE,
        MOTIF_ROW_ROOT_SOURCE,
        MOTIF_RAW_PLACEMENT_SOURCE,
        MOTIF_ROW_REMAP_SOURCE,
    ]
    @test motif_kinds(incidence, 13) == UInt8[
        MOTIF_COLUMN_ROOT_SOURCE,
        MOTIF_COLUMN_ROOT_SOURCE,
        MOTIF_RAW_PLACEMENT_SOURCE,
        MOTIF_ROW_REMAP_SOURCE,
    ]
    @test motif_kinds(incidence, 17) == UInt8[
        MOTIF_ROW_ROOT_SOURCE,
        MOTIF_ROW_ROOT_SOURCE,
        MOTIF_ROW_ROOT_SOURCE,
        MOTIF_ROW_ROOT_SOURCE,
        MOTIF_ROW_ROOT_SOURCE,
        MOTIF_ROW_ROOT_SOURCE,
        MOTIF_RAW_PLACEMENT_SOURCE,
        MOTIF_ROW_REMAP_SOURCE,
    ]
    @test motif_kinds(incidence, 21) == UInt8[
        MOTIF_COLUMN_ROOT_SOURCE,
        MOTIF_COLUMN_ROOT_SOURCE,
        MOTIF_COLUMN_ROOT_SOURCE,
        MOTIF_COLUMN_ROOT_SOURCE,
        MOTIF_COLUMN_ROOT_SOURCE,
        MOTIF_COLUMN_ROOT_SOURCE,
        MOTIF_RAW_PLACEMENT_SOURCE,
        MOTIF_ROW_REMAP_SOURCE,
    ]

    # Unused canonical placement slots are observed ABSENT, not silence.
    for family in 1:6
        motif = (family - 1) * 4 + 3
        @test motif_source_count(incidence, motif) == 1
        source = motif_source(incidence, motif, 1)
        @test source.kind == MOTIF_ABSENT_SOURCE
        @test source.placement_slot == 3
    end

    # Family 7 carries the whole ordered raw footprint and the corresponding
    # exact remap/clear identity on disjoint branches.
    for motif in 25:28
        @test motif_source_count(incidence, motif) == 8
        @test motif_branches(incidence, motif) == UInt8[1, 2, 3, 4, 5, 6, 7, 8]
        @test motif_kinds(incidence, motif) == UInt8[
            MOTIF_RAW_PLACEMENT_SOURCE,
            MOTIF_RAW_PLACEMENT_SOURCE,
            MOTIF_ABSENT_SOURCE,
            MOTIF_ABSENT_SOURCE,
            MOTIF_ROW_REMAP_SOURCE,
            MOTIF_ROW_REMAP_SOURCE,
            MOTIF_ABSENT_SOURCE,
            MOTIF_ABSENT_SOURCE,
        ]
        @test [
            motif_source(incidence, motif, rank).placement_slot
            for rank in 1:8
        ] == UInt8[1, 2, 3, 4, 1, 2, 3, 4]
    end

    # Family 8 partitions nine independent semantic items across four cells;
    # queue order, REN words and booleans never share one scalar token.
    @test motif_source_count(incidence, 29) == 6
    @test motif_kinds(incidence, 29) == fill(MOTIF_QUEUE_SOURCE, 6)
    @test [motif_source(incidence, 29, rank).context_slot for rank in 1:6] ==
          UInt8[1, 2, 3, 4, 5, 6]
    @test motif_kinds(incidence, 30) == UInt8[
        MOTIF_REN_WORD_SOURCE,
        MOTIF_REN_WORD_SOURCE,
        MOTIF_BOOLEAN_SOURCE,
        MOTIF_BOOLEAN_SOURCE,
    ]
    @test [motif_source(incidence, 30, rank).context_slot for rank in 1:4] ==
          UInt8[7, 8, 9, 10]
    @test motif_kinds(incidence, 31) == UInt8[
        MOTIF_QUEUE_SOURCE,
        MOTIF_QUEUE_SOURCE,
        MOTIF_QUEUE_SOURCE,
        MOTIF_QUEUE_SOURCE,
        MOTIF_QUEUE_SOURCE,
        MOTIF_BOOLEAN_SOURCE,
    ]
    @test motif_kinds(incidence, 32) == UInt8[
        MOTIF_QUEUE_SOURCE,
        MOTIF_REN_WORD_SOURCE,
        MOTIF_REN_WORD_SOURCE,
        MOTIF_BOOLEAN_SOURCE,
        MOTIF_BOOLEAN_SOURCE,
    ]
end

@testset "raw placement clear-remap and context interventions are observable" begin
    topology = canonical_topology()
    no_clear = CandidateMotifIncidence()
    clear = CandidateMotifIncidence()
    fill_candidate_motif_incidence!(no_clear, topology, motif_fixture_context())
    fill_candidate_motif_incidence!(
        clear,
        topology,
        motif_fixture_context(clear_last=true),
    )
    for motif in 25:28
        @test any(
            motif_source(clear, motif, rank).kind == MOTIF_CLEARED_ROW_SOURCE
            for rank in 1:8
        )
        @test any(
            motif_source(no_clear, motif, rank) != motif_source(clear, motif, rank)
            for rank in 1:8
        )
    end

    # Exact Int32 REN remains distinguishable above the Float32 integer limit.
    ren_a = CandidateMotifIncidence()
    ren_b = CandidateMotifIncidence()
    fill_candidate_motif_incidence!(
        ren_a,
        topology,
        motif_fixture_context(ren=Int32(16_777_216)),
    )
    fill_candidate_motif_incidence!(
        ren_b,
        topology,
        motif_fixture_context(ren=Int32(16_777_217)),
    )
    @test motif_source(ren_a, 30, 1) != motif_source(ren_b, 30, 1)
    packet_a = zeros(Float32, 12)
    packet_b = zeros(Float32, 12)
    materialize_external_motif_packet!(packet_a, motif_source(ren_a, 30, 1))
    materialize_external_motif_packet!(packet_b, motif_source(ren_b, 30, 1))
    @test packet_a != packet_b
    @test_throws ArgumentError materialize_external_motif_packet!(
        packet_a,
        motif_source(no_clear, 1, 1),
    )

    # NEXT role swap and each explicit boolean change their own descriptors.
    queue_swap = CandidateMotifIncidence()
    fill_candidate_motif_incidence!(queue_swap, topology, motif_fixture_context(
        next=(UInt8(3), UInt8(2), UInt8(4), UInt8(5), UInt8(6)),
    ))
    @test motif_source(no_clear, 29, 2) != motif_source(queue_swap, 29, 2)
    @test motif_source(no_clear, 29, 3) != motif_source(queue_swap, 29, 3)
    hold_change = CandidateMotifIncidence()
    fill_candidate_motif_incidence!(
        hold_change,
        topology,
        motif_fixture_context(hold=UInt8(8)),
    )
    @test motif_source(no_clear, 29, 1) != motif_source(hold_change, 29, 1)
    @test motif_source(no_clear, 32, 1) != motif_source(hold_change, 32, 1)

    b2b_flip = CandidateMotifIncidence()
    fill_candidate_motif_incidence!(
        b2b_flip,
        topology,
        motif_fixture_context(b2b=UInt8(2)),
    )
    @test motif_source(no_clear, 30, 3) != motif_source(b2b_flip, 30, 3)
    @test motif_source(no_clear, 32, 4) != motif_source(b2b_flip, 32, 4)
    b2b_closure = AffectedClosure()
    fill_changed_motif_closure!(
        b2b_closure,
        topology,
        no_clear,
        b2b_flip,
    )
    @test UInt16[
        node for node in b2b_closure if node_class(topology, node) == MOTIF_CLASS
    ] == UInt16[motif_node(8, 2), motif_node(8, 4)]

    tspin_flip = CandidateMotifIncidence()
    fill_candidate_motif_incidence!(
        tspin_flip,
        topology,
        motif_fixture_context(tspin=UInt8(1)),
    )
    closure = AffectedClosure()
    fill_changed_motif_closure!(closure, topology, no_clear, tspin_flip)
    changed_motifs = UInt16[
        node for node in closure if node_class(topology, node) == MOTIF_CLASS
    ]
    @test changed_motifs == UInt16[
        motif_node(8, 2),
        motif_node(8, 3),
        motif_node(8, 4),
    ]
    @test any(node_class(topology, node) == EVIDENCE_CLASS for node in closure)
    @test any(node_class(topology, node) == OUTPUT_CLASS for node in closure)

    # Moving one canonical raw slot changes its six local families plus all
    # four footprint/remap views, but never changes family 8.
    moved = CandidateMotifIncidence()
    fill_candidate_motif_incidence!(moved, topology, motif_fixture_context(
        positions=(UInt16(10 + 2 * 24), UInt16(0), UInt16(0), UInt16(0)),
        count=1,
    ))
    base_one = CandidateMotifIncidence()
    fill_candidate_motif_incidence!(base_one, topology, motif_fixture_context(
        positions=(UInt16(11 + 2 * 24), UInt16(0), UInt16(0), UInt16(0)),
        count=1,
    ))
    fill_changed_motif_closure!(closure, topology, base_one, moved)
    changed_motifs = UInt16[
        node for node in closure if node_class(topology, node) == MOTIF_CLASS
    ]
    @test Set(changed_motifs) == Set(UInt16[
        motif_node(1, 1), motif_node(2, 1), motif_node(3, 1),
        motif_node(4, 1), motif_node(5, 1), motif_node(6, 1),
        motif_node(7, 1), motif_node(7, 2), motif_node(7, 3),
        motif_node(7, 4),
    ])
    @test all(motif_family(topology, node) != 8 for node in changed_motifs)

    # Live spine changes reach only motifs that explicitly name the affected
    # spatial packet or one of its ordered row/column ancestors.
    seed = UInt16[spatial_node(AFTER_PLANE, 24, 1)]
    fill_incidence_affected_closure!(
        closure,
        topology,
        no_clear,
        seed,
    )
    @test motif_node(1, 1) in closure
    @test row_root_node(AFTER_PLANE, 24) in closure
    @test column_root_node(AFTER_PLANE, 1) in closure
    @test any(node_class(topology, node) == OUTPUT_CLASS for node in closure)
    fill_incidence_affected_closure!(closure, topology, no_clear, seed)
    @test @allocated(
        fill_incidence_affected_closure!(closure, topology, no_clear, seed)
    ) == 0
    fill_changed_motif_closure!(closure, topology, no_clear, tspin_flip)
    @test @allocated(
        fill_changed_motif_closure!(closure, topology, no_clear, tspin_flip)
    ) == 0
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
    incidence = CandidateMotifIncidence()
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
    @test_throws BoundsError motif_source_count(incidence, 33)
    @test_throws BoundsError motif_source(incidence, 1, 1)
    @test_throws DimensionMismatch materialize_external_motif_packet!(
        zeros(Float32, 11),
        CandidateMotifSource(
            MOTIF_ABSENT_SOURCE,
            UInt8(0),
            UInt16(0),
            UInt8(0),
            UInt8(0),
            UInt8(1),
            UInt8(1),
            UInt8(1),
            Int32(1),
        ),
    )
    @test_throws ArgumentError motif_fixture_context(
        positions=(UInt16(48), UInt16(24), UInt16(0), UInt16(0)),
    )
    @test_throws ArgumentError CandidateMotifContext(
        (UInt16(24), UInt16(48), UInt16(0), UInt16(0)),
        2,
        ntuple(row -> UInt8(row), 24),
        UInt32(1) << 23,
        1,
        UInt8(1),
        (UInt8(2), UInt8(3), UInt8(4), UInt8(5), UInt8(6)),
        0,
        UInt8(1),
        UInt8(1),
    )
    @test_throws BoundsError fill_incidence_affected_closure!(
        closure,
        topology,
        incidence,
        UInt16[1],
        2,
    )
end
