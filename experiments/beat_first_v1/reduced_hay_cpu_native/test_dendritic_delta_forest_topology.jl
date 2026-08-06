using Test
using Random

module DendriticDeltaForestTopologyTestHarness
include(joinpath(@__DIR__, "ActiveApicalCell.jl"))
include(joinpath(@__DIR__, "CandidateDeltaInput.jl"))
include(joinpath(@__DIR__, "DendriticProgramBank.jl"))
include(joinpath(@__DIR__, "SharedDendriticFactor.jl"))
include(joinpath(@__DIR__, "SpatialDendriticFactors.jl"))
include(joinpath(@__DIR__, "DendriticDeltaForestTopology.jl"))
end

const Delta = DendriticDeltaForestTopologyTestHarness.CandidateDeltaInput
const Spatial = DendriticDeltaForestTopologyTestHarness.SpatialDendriticFactors
const Forest =
    DendriticDeltaForestTopologyTestHarness.DendriticDeltaForestTopology

function naive_ancestor_closure(topology, positions)
    closure = Set{UInt16}()
    frontier = UInt16[]
    for position in positions
        leaf = Forest.leaf_node(position)
        push!(closure, leaf)
        push!(frontier, leaf)
    end
    cursor = 1
    while cursor <= length(frontier)
        node = frontier[cursor]
        cursor += 1
        for parent_index in 1:Forest.parent_count(topology, node)
            parent = Forest.parent_node(topology, node, parent_index)
            if !(parent in closure)
                push!(closure, parent)
                push!(frontier, parent)
            end
        end
    end
    return closure
end

function reachable_anchors(topology, leaf)
    closure = naive_ancestor_closure(topology, UInt16[leaf])
    return Set(node for node in closure if Forest.is_anchor(topology, node))
end

function closure_allocation(closure, topology, positions)
    Forest.fill_dirty_closure!(closure, topology, positions)
    return @allocated Forest.fill_dirty_closure!(
        closure,
        topology,
        positions,
    )
end

function affected_closure_allocation(closure, topology, affected)
    Forest.fill_dirty_closure!(closure, topology, affected)
    return @allocated Forest.fill_dirty_closure!(
        closure,
        topology,
        affected,
    )
end

@testset "fixed crossed delta-forest structure" begin
    topology = Forest.canonical_topology()
    @test Forest.ROW_COUNT == 24
    @test Forest.COLUMN_COUNT == 10
    @test Forest.LEAF_COUNT == 240
    @test Forest.ROW_HALF_COUNT == 48
    @test Forest.ROW_ROOT_COUNT == 24
    @test Forest.COLUMN_GROUP_COUNT == 40
    @test Forest.COLUMN_ROOT_COUNT == 10
    @test Forest.ANCHOR_COUNT == 34
    @test Forest.NODE_COUNT == 362
    @test Forest.NODES_PER_PLANE == 362
    @test Forest.CHILD_EDGE_COUNT == 568
    @test !Base.ismutabletype(typeof(topology))
    @test all(field -> field isa Tuple, getfield.(Ref(topology), fieldnames(typeof(topology))))

    class_counts = Dict(
        Forest.LEAF_CLASS => 0,
        Forest.ROW_HALF_CLASS => 0,
        Forest.ROW_ROOT_CLASS => 0,
        Forest.COLUMN_GROUP_CLASS => 0,
        Forest.COLUMN_ROOT_CLASS => 0,
    )
    child_edges = 0
    parent_edges = 0
    @inbounds for node in 1:Forest.NODE_COUNT
        class_counts[Forest.node_class(topology, node)] += 1
        child_edges += Forest.child_count(topology, node)
        parent_edges += Forest.parent_count(topology, node)
        @test Forest.forward_node(topology, node) == UInt16(node)
        @test Forest.reverse_node(topology, node) ==
              UInt16(Forest.NODE_COUNT - node + 1)
        for child_index in 1:Forest.child_count(topology, node)
            child = Forest.child_node(topology, node, child_index)
            @test child < node
            @test UInt16(node) in (
                Forest.parent_node(topology, child, parent_index)
                for parent_index in 1:Forest.parent_count(topology, child)
            )
        end
    end
    @test child_edges == Forest.CHILD_EDGE_COUNT
    @test parent_edges == Forest.CHILD_EDGE_COUNT
    @test class_counts[Forest.LEAF_CLASS] == 240
    @test class_counts[Forest.ROW_HALF_CLASS] == 48
    @test class_counts[Forest.ROW_ROOT_CLASS] == 24
    @test class_counts[Forest.COLUMN_GROUP_CLASS] == 40
    @test class_counts[Forest.COLUMN_ROOT_CLASS] == 10

    @test all(
        Forest.node_slot(topology, Forest.leaf_node(slot)) == slot
        for slot in 1:Forest.LEAF_COUNT
    )
    @test all(
        Forest.node_slot(topology, Forest.row_half_node(row, half)) ==
            (row - 1) * 2 + half
        for row in 1:24 for half in 1:2
    )
    @test all(
        Forest.node_slot(topology, Forest.column_group_node(column, group)) ==
            (column - 1) * 4 + group
        for column in 1:10 for group in 1:4
    )
    @test all(Forest.is_anchor(topology, Forest.anchor_node(index)) for index in 1:34)
end

@testset "every leaf reaches its row and column anchors" begin
    topology = Forest.canonical_topology()
    for column in 1:Forest.COLUMN_COUNT, row in 1:Forest.ROW_COUNT
        position = row + (column - 1) * Forest.ROW_COUNT
        anchors = reachable_anchors(topology, position)
        @test anchors == Set((
            Forest.row_anchor_node(row),
            Forest.column_anchor_node(column),
        ))
    end
end

@testset "dirty closure is exact and topologically ordered" begin
    topology = Forest.canonical_topology()
    closure = Forest.DirtyClosure()
    cases = (
        UInt16[],
        UInt16[1],
        UInt16[240],
        UInt16[97, 97],
        UInt16[1, 6, 7, 24, 25, 240],
        UInt16.(1:Forest.LEAF_COUNT),
    )
    for positions in cases
        Forest.fill_dirty_closure!(closure, topology, positions)
        actual = Set(
            Forest.dirty_forward_node(closure, index)
            for index in 1:Forest.dirty_count(closure)
        )
        @test actual == naive_ancestor_closure(topology, positions)
        forward = [
            Forest.dirty_forward_node(closure, index)
            for index in 1:Forest.dirty_count(closure)
        ]
        reverse = [
            Forest.dirty_reverse_node(closure, index)
            for index in 1:Forest.dirty_count(closure)
        ]
        @test issorted(forward)
        @test reverse == Base.reverse(forward)
        for node in forward
            for parent_index in 1:Forest.parent_count(topology, node)
                parent = Forest.parent_node(topology, node, parent_index)
                @test findfirst(==(node), forward) < findfirst(==(parent), forward)
            end
        end
    end

    rng = MersenneTwister(0xD31A)
    positions = zeros(UInt16, Forest.LEAF_COUNT)
    for _ in 1:128
        count = rand(rng, 0:Forest.LEAF_COUNT)
        for index in 1:count
            positions[index] = UInt16(rand(rng, 1:Forest.LEAF_COUNT))
        end
        Forest.fill_dirty_closure!(closure, topology, positions, count)
        actual = Set(
            Forest.dirty_forward_node(closure, index)
            for index in 1:Forest.dirty_count(closure)
        )
        @test actual == naive_ancestor_closure(topology, @view(positions[1:count]))
    end
end

@testset "existing candidate affected-position closure" begin
    topology = Forest.canonical_topology()
    common = Delta.StateCommon()
    common.board[20, 4] = UInt8(1)
    common.board[22, 6] = UInt8(1)
    common.board[24, 3] = UInt8(1)
    placement = zeros(UInt8, Delta.BOARD_ROWS, Delta.BOARD_COLUMNS)
    placement[21, 5] = UInt8(1)
    delta = Delta.CandidateDelta()
    Delta.prepare_candidate_delta!(delta, common, placement, 0.0f0)
    materialized = Delta.CandidateMaterialization()
    Delta.reconstruct_candidate!(materialized, common, delta)
    affected = Spatial.AffectedPositions()
    Spatial.prepare_affected_positions!(affected, common, materialized)

    closure = Forest.DirtyClosure()
    Forest.fill_dirty_closure!(closure, topology, affected)
    actual = Set(closure)
    @test actual == naive_ancestor_closure(topology, affected)
    @test Forest.dirty_count(closure) > Spatial.affected_count(affected)

    # A line clear exercises the helper's much larger candidate closure.
    fill!(common.board, UInt8(0))
    fill!(placement, UInt8(0))
    @views common.board[end, 1:9] .= UInt8(1)
    placement[end, 10] = UInt8(1)
    Delta.prepare_candidate_delta!(delta, common, placement, 0.0f0)
    Delta.reconstruct_candidate!(materialized, common, delta)
    Spatial.prepare_affected_positions!(affected, common, materialized)
    Forest.fill_dirty_closure!(closure, topology, affected)
    @test Set(closure) == naive_ancestor_closure(topology, affected)
end

@testset "hot dirty-closure fill allocates nothing" begin
    topology = Forest.canonical_topology()
    closure = Forest.DirtyClosure()
    positions = UInt16[1, 6, 25, 97, 144, 240]
    @test closure_allocation(closure, topology, positions) == 0

    before = zeros(UInt8, Delta.BOARD_ROWS, Delta.BOARD_COLUMNS)
    after = copy(before)
    after[20, 4] = UInt8(1)
    after[21, 5] = UInt8(1)
    affected = Spatial.AffectedPositions()
    Spatial.prepare_affected_positions!(affected, before, after)
    @test affected_closure_allocation(closure, topology, affected) == 0
end
