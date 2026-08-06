module DendriticDeltaForestTopology

"""
Immutable topology of the crossed row/column reduction forest used by the
candidate-delta dendritic model.

Each plane owns 240 spatial leaves.  A leaf contributes to exactly two trees:

* its five-column row half and then its row anchor;
* its six-row column group and then its column anchor.

Node identifiers are also a deterministic topological order.  Leaves precede
all first-level reducers, which precede all anchors.  This lets a dirty closure
propagate marks toward parents with one bounded forward scan and no queue.
"""

export ANCHOR_COUNT,
       CHILD_EDGE_COUNT,
       COLUMN_COUNT,
       COLUMN_GROUP_CLASS,
       COLUMN_GROUP_COUNT,
       COLUMN_ROOT_CLASS,
       COLUMN_ROOT_COUNT,
       DeltaForestTopology,
       DirtyClosure,
       LEAF_CLASS,
       LEAF_COUNT,
       NODE_COUNT,
       NODES_PER_PLANE,
       ROW_COUNT,
       ROW_HALF_CLASS,
       ROW_HALF_COUNT,
       ROW_ROOT_CLASS,
       ROW_ROOT_COUNT,
       anchor_node,
       canonical_topology,
       child_count,
       child_node,
       column_anchor_node,
       column_group_node,
       dirty_count,
       dirty_forward_node,
       dirty_reverse_node,
       fill_dirty_closure!,
       forward_node,
       is_anchor,
       leaf_node,
       node_class,
       node_slot,
       parent_count,
       parent_node,
       reverse_node,
       row_anchor_node,
       row_half_node

const ROW_COUNT = 24
const COLUMN_COUNT = 10
const LEAF_COUNT = ROW_COUNT * COLUMN_COUNT
const ROW_HALF_COUNT = ROW_COUNT * 2
const COLUMN_GROUP_COUNT = COLUMN_COUNT * 4
const ROW_ROOT_COUNT = ROW_COUNT
const COLUMN_ROOT_COUNT = COLUMN_COUNT
const ANCHOR_COUNT = ROW_ROOT_COUNT + COLUMN_ROOT_COUNT

# IDs are ordered by dependency depth, not merely by class name.
const _LEAF_FIRST = 1
const _ROW_HALF_FIRST = _LEAF_FIRST + LEAF_COUNT
const _COLUMN_GROUP_FIRST = _ROW_HALF_FIRST + ROW_HALF_COUNT
const _ROW_ROOT_FIRST = _COLUMN_GROUP_FIRST + COLUMN_GROUP_COUNT
const _COLUMN_ROOT_FIRST = _ROW_ROOT_FIRST + ROW_ROOT_COUNT
const NODE_COUNT = _COLUMN_ROOT_FIRST + COLUMN_ROOT_COUNT - 1
const NODES_PER_PLANE = NODE_COUNT

# 240 leaf->row-half + 48 half->row-root + 240 leaf->column-group
# + 40 group->column-root.
const CHILD_EDGE_COUNT =
    LEAF_COUNT + ROW_HALF_COUNT + LEAF_COUNT + COLUMN_GROUP_COUNT

const LEAF_CLASS = UInt8(1)
const ROW_HALF_CLASS = UInt8(2)
const ROW_ROOT_CLASS = UInt8(3)
const COLUMN_GROUP_CLASS = UInt8(4)
const COLUMN_ROOT_CLASS = UInt8(5)

"""
Fully immutable compressed adjacency for one plane.

Offsets are one-based CSR offsets.  Both adjacency directions contain the
same `CHILD_EDGE_COUNT` edges.  `node_classes` and `node_slots` make the
physical role and the class-local ordinal of every node explicit.
"""
struct DeltaForestTopology
    child_offsets::NTuple{NODE_COUNT + 1,UInt16}
    children::NTuple{CHILD_EDGE_COUNT,UInt16}
    parent_offsets::NTuple{NODE_COUNT + 1,UInt16}
    parents::NTuple{CHILD_EDGE_COUNT,UInt16}
    node_classes::NTuple{NODE_COUNT,UInt8}
    node_slots::NTuple{NODE_COUNT,UInt16}
    forward_order::NTuple{NODE_COUNT,UInt16}
    reverse_order::NTuple{NODE_COUNT,UInt16}
end

@inline function _check_row(row::Integer)
    1 <= row <= ROW_COUNT || throw(BoundsError(1:ROW_COUNT, row))
    return Int(row)
end

@inline function _check_column(column::Integer)
    1 <= column <= COLUMN_COUNT ||
        throw(BoundsError(1:COLUMN_COUNT, column))
    return Int(column)
end

@inline function _check_node(node::Integer)
    1 <= node <= NODE_COUNT || throw(BoundsError(1:NODE_COUNT, node))
    return Int(node)
end

@inline function _check_position(position::Integer)
    1 <= position <= LEAF_COUNT ||
        throw(BoundsError(1:LEAF_COUNT, position))
    return Int(position)
end

@inline leaf_node(position::Integer) = UInt16(_check_position(position))

@inline function row_half_node(row::Integer, half::Integer)
    physical_row = _check_row(row)
    1 <= half <= 2 || throw(BoundsError(1:2, half))
    slot = (physical_row - 1) * 2 + Int(half)
    return UInt16(_ROW_HALF_FIRST + slot - 1)
end

@inline function column_group_node(column::Integer, group::Integer)
    physical_column = _check_column(column)
    1 <= group <= 4 || throw(BoundsError(1:4, group))
    slot = (physical_column - 1) * 4 + Int(group)
    return UInt16(_COLUMN_GROUP_FIRST + slot - 1)
end

@inline row_anchor_node(row::Integer) =
    UInt16(_ROW_ROOT_FIRST + _check_row(row) - 1)

@inline column_anchor_node(column::Integer) =
    UInt16(_COLUMN_ROOT_FIRST + _check_column(column) - 1)

@inline function anchor_node(index::Integer)
    1 <= index <= ANCHOR_COUNT ||
        throw(BoundsError(1:ANCHOR_COUNT, index))
    return UInt16(_ROW_ROOT_FIRST + Int(index) - 1)
end

@inline function _add_edge!(children, parents, parent::Int, child::Int)
    parent > child || error("forest node IDs must be topological")
    push!(children[parent], UInt16(child))
    push!(parents[child], UInt16(parent))
    return nothing
end

function _flatten_adjacency(adjacency)
    offsets = Vector{UInt16}(undef, NODE_COUNT + 1)
    values = Vector{UInt16}()
    sizehint!(values, CHILD_EDGE_COUNT)
    offsets[1] = UInt16(1)
    @inbounds for node in 1:NODE_COUNT
        sort!(adjacency[node])
        append!(values, adjacency[node])
        offsets[node + 1] = UInt16(length(values) + 1)
    end
    length(values) == CHILD_EDGE_COUNT ||
        error("delta-forest edge count drift")
    return Tuple(offsets), Tuple(values)
end

function _build_topology()
    children = [UInt16[] for _ in 1:NODE_COUNT]
    parents = [UInt16[] for _ in 1:NODE_COUNT]

    # Two five-column halves per row.
    for row in 1:ROW_COUNT
        row_root = Int(row_anchor_node(row))
        for half in 1:2
            half_node = Int(row_half_node(row, half))
            first_column = (half - 1) * 5 + 1
            for column in first_column:(first_column + 4)
                position = row + (column - 1) * ROW_COUNT
                _add_edge!(children, parents, half_node, position)
            end
            _add_edge!(children, parents, row_root, half_node)
        end
    end

    # Four six-row groups per column.
    for column in 1:COLUMN_COUNT
        column_root = Int(column_anchor_node(column))
        for group in 1:4
            group_node = Int(column_group_node(column, group))
            first_row = (group - 1) * 6 + 1
            for row in first_row:(first_row + 5)
                position = row + (column - 1) * ROW_COUNT
                _add_edge!(children, parents, group_node, position)
            end
            _add_edge!(children, parents, column_root, group_node)
        end
    end

    child_offsets, flat_children = _flatten_adjacency(children)
    parent_offsets, flat_parents = _flatten_adjacency(parents)

    classes = Vector{UInt8}(undef, NODE_COUNT)
    slots = Vector{UInt16}(undef, NODE_COUNT)
    @inbounds for position in 1:LEAF_COUNT
        classes[position] = LEAF_CLASS
        slots[position] = UInt16(position)
    end
    @inbounds for slot in 1:ROW_HALF_COUNT
        node = _ROW_HALF_FIRST + slot - 1
        classes[node] = ROW_HALF_CLASS
        slots[node] = UInt16(slot)
    end
    @inbounds for slot in 1:COLUMN_GROUP_COUNT
        node = _COLUMN_GROUP_FIRST + slot - 1
        classes[node] = COLUMN_GROUP_CLASS
        slots[node] = UInt16(slot)
    end
    @inbounds for slot in 1:ROW_ROOT_COUNT
        node = _ROW_ROOT_FIRST + slot - 1
        classes[node] = ROW_ROOT_CLASS
        slots[node] = UInt16(slot)
    end
    @inbounds for slot in 1:COLUMN_ROOT_COUNT
        node = _COLUMN_ROOT_FIRST + slot - 1
        classes[node] = COLUMN_ROOT_CLASS
        slots[node] = UInt16(slot)
    end

    forward = ntuple(index -> UInt16(index), NODE_COUNT)
    reverse = ntuple(index -> UInt16(NODE_COUNT - index + 1), NODE_COUNT)
    return DeltaForestTopology(
        child_offsets,
        flat_children,
        parent_offsets,
        flat_parents,
        Tuple(classes),
        Tuple(slots),
        forward,
        reverse,
    )
end

const _CANONICAL_TOPOLOGY = _build_topology()

"""Return the process-wide immutable canonical topology."""
@inline canonical_topology() = _CANONICAL_TOPOLOGY

@inline function child_count(topology::DeltaForestTopology, node::Integer)
    physical_node = _check_node(node)
    return Int(topology.child_offsets[physical_node + 1]) -
           Int(topology.child_offsets[physical_node])
end

@inline function child_node(
    topology::DeltaForestTopology,
    node::Integer,
    child_index::Integer,
)
    physical_node = _check_node(node)
    count = child_count(topology, physical_node)
    1 <= child_index <= count || throw(BoundsError(1:count, child_index))
    flat_index = Int(topology.child_offsets[physical_node]) +
                 Int(child_index) - 1
    return @inbounds topology.children[flat_index]
end

@inline function parent_count(topology::DeltaForestTopology, node::Integer)
    physical_node = _check_node(node)
    return Int(topology.parent_offsets[physical_node + 1]) -
           Int(topology.parent_offsets[physical_node])
end

@inline function parent_node(
    topology::DeltaForestTopology,
    node::Integer,
    parent_index::Integer,
)
    physical_node = _check_node(node)
    count = parent_count(topology, physical_node)
    1 <= parent_index <= count || throw(BoundsError(1:count, parent_index))
    flat_index = Int(topology.parent_offsets[physical_node]) +
                 Int(parent_index) - 1
    return @inbounds topology.parents[flat_index]
end

@inline node_class(topology::DeltaForestTopology, node::Integer) =
    @inbounds topology.node_classes[_check_node(node)]

@inline node_slot(topology::DeltaForestTopology, node::Integer) =
    @inbounds topology.node_slots[_check_node(node)]

@inline function is_anchor(topology::DeltaForestTopology, node::Integer)
    class = node_class(topology, node)
    return class == ROW_ROOT_CLASS || class == COLUMN_ROOT_CLASS
end

@inline forward_node(topology::DeltaForestTopology, index::Integer) =
    @inbounds topology.forward_order[_check_node(index)]

@inline reverse_node(topology::DeltaForestTopology, index::Integer) =
    @inbounds topology.reverse_order[_check_node(index)]

"""Caller-owned, fixed-capacity dirty-closure storage."""
mutable struct DirtyClosure <: AbstractVector{UInt16}
    nodes::Memory{UInt16}
    marked::Memory{UInt8}
    count::Int
    function DirtyClosure()
        nodes = Memory{UInt16}(undef, NODE_COUNT)
        marked = Memory{UInt8}(undef, NODE_COUNT)
        fill!(nodes, UInt16(0))
        fill!(marked, UInt8(0))
        return new(nodes, marked, 0)
    end
end

@inline dirty_count(closure::DirtyClosure) = closure.count
Base.IndexStyle(::Type{DirtyClosure}) = IndexLinear()
Base.size(closure::DirtyClosure) = (closure.count,)
Base.length(closure::DirtyClosure) = closure.count
@inline Base.getindex(closure::DirtyClosure, index::Int) =
    dirty_forward_node(closure, index)

@inline function dirty_forward_node(closure::DirtyClosure, index::Integer)
    1 <= index <= closure.count ||
        throw(BoundsError(1:closure.count, index))
    return @inbounds closure.nodes[Int(index)]
end

@inline function dirty_reverse_node(closure::DirtyClosure, index::Integer)
    1 <= index <= closure.count ||
        throw(BoundsError(1:closure.count, index))
    return @inbounds closure.nodes[closure.count - Int(index) + 1]
end

"""
Fill the exact ancestor closure of `affected_positions` without allocation.

The caller owns `closure`.  Positions may contain duplicates and may be any
indexable collection whose first `affected_count` entries are board positions
in canonical column-major order.  Returned nodes are unique and ordered for
forward evaluation; `dirty_reverse_node` exposes the exact reverse order.
"""
function fill_dirty_closure!(
    closure::DirtyClosure,
    topology::DeltaForestTopology,
    affected_positions,
    affected_count::Integer,
)
    physical_count = Int(affected_count)
    0 <= physical_count <= length(affected_positions) ||
        throw(BoundsError(affected_positions, physical_count))
    fill!(closure.marked, UInt8(0))

    @inbounds for affected_index in 1:physical_count
        position = _check_position(affected_positions[affected_index])
        closure.marked[position] = UInt8(1)
    end

    # Parent IDs are strictly larger than child IDs, so newly marked parents
    # will be visited later in this same scan.  No work queue is necessary.
    @inbounds for node in 1:NODE_COUNT
        iszero(closure.marked[node]) && continue
        first_parent = Int(topology.parent_offsets[node])
        parent_limit = Int(topology.parent_offsets[node + 1]) - 1
        for flat_index in first_parent:parent_limit
            closure.marked[Int(topology.parents[flat_index])] = UInt8(1)
        end
    end

    count = 0
    @inbounds for node in 1:NODE_COUNT
        iszero(closure.marked[node]) && continue
        count += 1
        closure.nodes[count] = UInt16(node)
    end
    closure.count = count
    return closure
end


@inline function fill_dirty_closure!(
    closure::DirtyClosure,
    topology::DeltaForestTopology,
    affected_positions,
)
    return fill_dirty_closure!(
        closure,
        topology,
        affected_positions,
        length(affected_positions),
    )
end


@inline function fill_dirty_closure!(
    closure::DirtyClosure,
    affected_positions,
)
    return fill_dirty_closure!(
        closure,
        _CANONICAL_TOPOLOGY,
        affected_positions,
        length(affected_positions),
    )
end

end # module
