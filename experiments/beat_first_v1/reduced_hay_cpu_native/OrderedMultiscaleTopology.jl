module OrderedMultiscaleTopology

"""
Immutable topology contract for the ordered multiscale Reduced-Hay graph.

The graph has two explicit 24 x 10 spatial planes.  Every spatial node feeds
one ordered binary row tree and one ordered binary column tree.  Row/column
roots then feed 32 semantic motif cells, 32 balanced evidence cells and 22
private output cells.  Node identifiers are a strict topological order, so a
caller-owned affected closure can be propagated with one forward scan and no
queue or allocation.

Information-spine edges transport the complete 12-coordinate axon packet.
Every such receiver has exactly two children with immutable left/right or
upper/lower slots.  Semantic edges instead carry one role-specific 12 -> 3
projection into one of eight ordered basal slots.  Mean pooling, four-child
lossless merges and unbounded semantic hubs are absent by construction.
"""

export AFTER_PLANE,
       BEFORE_PLANE,
       COLUMN_COUNT,
       COLUMN_INTERNAL_CLASS,
       COLUMN_INTERNAL_COUNT,
       COLUMN_INTERNAL_PER_COLUMN,
       COLUMN_INTERNAL_PER_PLANE,
       EDGE_COUNT,
       EVIDENCE_CLASS,
       EVIDENCE_COUNT,
       FULL_PACKET_EDGE,
       FULL_PACKET_WIDTH,
       MOTIF_CLASS,
       MOTIF_COUNT,
       MOTIF_FAMILY_COUNT,
       MOTIF_SLOTS_PER_FAMILY,
       NODE_COUNT,
       OUTPUT_CLASS,
       OUTPUT_COUNT,
       PLANE_COUNT,
       PROJECTED_TRIPLET_EDGE,
       ROW_COUNT,
       ROW_INTERNAL_CLASS,
       ROW_INTERNAL_COUNT,
       ROW_INTERNAL_PER_PLANE,
       ROW_INTERNAL_PER_ROW,
       SEMANTIC_FANIN,
       SEMANTIC_OUTPUT_WIDTH,
       SEMANTIC_ROLE_COUNT,
       SPATIAL_CLASS,
       SPATIAL_COUNT,
       SPATIAL_COUNT_PER_PLANE,
       OrderedTopology,
       AffectedClosure,
       affected_count,
       affected_forward_node,
       affected_reverse_node,
       canonical_topology,
       child_count,
       child_edge,
       child_node,
       child_slot,
       column_internal_node,
       column_root_node,
       edge_destination,
       edge_kind,
       edge_semantic_role,
       edge_source,
       evidence_node,
       fill_affected_closure!,
       fill_spatial_affected_closure!,
       interval_first,
       interval_last,
       motif_family,
       motif_node,
       motif_slot,
       node_class,
       node_plane,
       node_slot,
       output_node,
       parent_count,
       parent_edge,
       parent_node,
       row_internal_node,
       row_root_node,
       spatial_node,
       spatial_position,
       validate_topology

const ROW_COUNT = 24
const COLUMN_COUNT = 10
const PLANE_COUNT = 2
const SPATIAL_COUNT_PER_PLANE = ROW_COUNT * COLUMN_COUNT
const SPATIAL_COUNT = PLANE_COUNT * SPATIAL_COUNT_PER_PLANE

const ROW_INTERNAL_PER_ROW = COLUMN_COUNT - 1
const ROW_INTERNAL_PER_PLANE = ROW_COUNT * ROW_INTERNAL_PER_ROW
const ROW_INTERNAL_COUNT = PLANE_COUNT * ROW_INTERNAL_PER_PLANE

const COLUMN_INTERNAL_PER_COLUMN = ROW_COUNT - 1
const COLUMN_INTERNAL_PER_PLANE = COLUMN_COUNT * COLUMN_INTERNAL_PER_COLUMN
const COLUMN_INTERNAL_COUNT = PLANE_COUNT * COLUMN_INTERNAL_PER_PLANE

const MOTIF_FAMILY_COUNT = 8
const MOTIF_SLOTS_PER_FAMILY = 4
const MOTIF_COUNT = MOTIF_FAMILY_COUNT * MOTIF_SLOTS_PER_FAMILY
const EVIDENCE_COUNT = 32
const OUTPUT_COUNT = 22

const BEFORE_PLANE = UInt8(1)
const AFTER_PLANE = UInt8(2)

const SPATIAL_CLASS = UInt8(1)
const ROW_INTERNAL_CLASS = UInt8(2)
const COLUMN_INTERNAL_CLASS = UInt8(3)
const MOTIF_CLASS = UInt8(4)
const EVIDENCE_CLASS = UInt8(5)
const OUTPUT_CLASS = UInt8(6)

const FULL_PACKET_EDGE = UInt8(1)
const PROJECTED_TRIPLET_EDGE = UInt8(2)
const FULL_PACKET_WIDTH = 12
const SEMANTIC_OUTPUT_WIDTH = 3
const SEMANTIC_ROLE_COUNT = 8
const SEMANTIC_FANIN = 8

const _SPATIAL_FIRST = 1
const _ROW_INTERNAL_FIRST = _SPATIAL_FIRST + SPATIAL_COUNT
const _COLUMN_INTERNAL_FIRST = _ROW_INTERNAL_FIRST + ROW_INTERNAL_COUNT
const _MOTIF_FIRST = _COLUMN_INTERNAL_FIRST + COLUMN_INTERNAL_COUNT
const _EVIDENCE_FIRST = _MOTIF_FIRST + MOTIF_COUNT
const _OUTPUT_FIRST = _EVIDENCE_FIRST + EVIDENCE_COUNT
const NODE_COUNT = _OUTPUT_FIRST + OUTPUT_COUNT - 1

# Two children for every information-spine internal plus eight semantic
# sources for every motif, evidence and output cell.
const _SPINE_EDGE_COUNT = 2 * (ROW_INTERNAL_COUNT + COLUMN_INTERNAL_COUNT)
const _MOTIF_EDGE_COUNT = MOTIF_COUNT * SEMANTIC_FANIN
const _EVIDENCE_EDGE_COUNT = EVIDENCE_COUNT * SEMANTIC_FANIN
const _OUTPUT_EDGE_COUNT = OUTPUT_COUNT * SEMANTIC_FANIN
const EDGE_COUNT = _SPINE_EDGE_COUNT + _MOTIF_EDGE_COUNT +
                   _EVIDENCE_EDGE_COUNT + _OUTPUT_EDGE_COUNT

@assert SPATIAL_COUNT == 480
@assert ROW_INTERNAL_COUNT == 432
@assert COLUMN_INTERNAL_COUNT == 460
@assert NODE_COUNT == 1_458
@assert EDGE_COUNT == 2_472
@assert 2 * FULL_PACKET_WIDTH == 24
@assert 4 * FULL_PACKET_WIDTH > 24

@inline function _check_plane(plane::Integer)
    1 <= plane <= PLANE_COUNT || throw(BoundsError(1:PLANE_COUNT, plane))
    return Int(plane)
end

@inline function _check_position(position::Integer)
    1 <= position <= SPATIAL_COUNT_PER_PLANE ||
        throw(BoundsError(1:SPATIAL_COUNT_PER_PLANE, position))
    return Int(position)
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

@inline function _check_edge(edge::Integer)
    1 <= edge <= EDGE_COUNT || throw(BoundsError(1:EDGE_COUNT, edge))
    return Int(edge)
end

@inline function spatial_node(plane::Integer, position::Integer)
    physical_plane = _check_plane(plane)
    physical_position = _check_position(position)
    return UInt16(
        _SPATIAL_FIRST +
        (physical_plane - 1) * SPATIAL_COUNT_PER_PLANE +
        physical_position - 1,
    )
end

@inline function spatial_node(
    plane::Integer,
    row::Integer,
    column::Integer,
)
    physical_row = _check_row(row)
    physical_column = _check_column(column)
    position = physical_row + (physical_column - 1) * ROW_COUNT
    return spatial_node(plane, position)
end

@inline function spatial_position(node::Integer)
    physical_node = _check_node(node)
    physical_node <= SPATIAL_COUNT || throw(ArgumentError(
        "node $physical_node is not a spatial node",
    ))
    return UInt16(mod(physical_node - _SPATIAL_FIRST,
                      SPATIAL_COUNT_PER_PLANE) + 1)
end

@inline function row_internal_node(
    plane::Integer,
    row::Integer,
    internal::Integer,
)
    physical_plane = _check_plane(plane)
    physical_row = _check_row(row)
    1 <= internal <= ROW_INTERNAL_PER_ROW ||
        throw(BoundsError(1:ROW_INTERNAL_PER_ROW, internal))
    slot = (physical_plane - 1) * ROW_INTERNAL_PER_PLANE +
           (physical_row - 1) * ROW_INTERNAL_PER_ROW + Int(internal)
    return UInt16(_ROW_INTERNAL_FIRST + slot - 1)
end

@inline row_root_node(plane::Integer, row::Integer) =
    row_internal_node(plane, row, ROW_INTERNAL_PER_ROW)

@inline function column_internal_node(
    plane::Integer,
    column::Integer,
    internal::Integer,
)
    physical_plane = _check_plane(plane)
    physical_column = _check_column(column)
    1 <= internal <= COLUMN_INTERNAL_PER_COLUMN ||
        throw(BoundsError(1:COLUMN_INTERNAL_PER_COLUMN, internal))
    slot = (physical_plane - 1) * COLUMN_INTERNAL_PER_PLANE +
           (physical_column - 1) * COLUMN_INTERNAL_PER_COLUMN + Int(internal)
    return UInt16(_COLUMN_INTERNAL_FIRST + slot - 1)
end

@inline column_root_node(plane::Integer, column::Integer) =
    column_internal_node(plane, column, COLUMN_INTERNAL_PER_COLUMN)

@inline function motif_node(family::Integer, slot::Integer)
    1 <= family <= MOTIF_FAMILY_COUNT ||
        throw(BoundsError(1:MOTIF_FAMILY_COUNT, family))
    1 <= slot <= MOTIF_SLOTS_PER_FAMILY ||
        throw(BoundsError(1:MOTIF_SLOTS_PER_FAMILY, slot))
    index = (Int(family) - 1) * MOTIF_SLOTS_PER_FAMILY + Int(slot)
    return UInt16(_MOTIF_FIRST + index - 1)
end

@inline function evidence_node(index::Integer)
    1 <= index <= EVIDENCE_COUNT ||
        throw(BoundsError(1:EVIDENCE_COUNT, index))
    return UInt16(_EVIDENCE_FIRST + Int(index) - 1)
end

@inline function output_node(index::Integer)
    1 <= index <= OUTPUT_COUNT || throw(BoundsError(1:OUTPUT_COUNT, index))
    return UInt16(_OUTPUT_FIRST + Int(index) - 1)
end

"""
Immutable CSR adjacency in both directions.

`child_*` indexes the incoming/source side of a destination node. `parent_*`
indexes outgoing/destination edges of a source node.  Edge metadata is stored
once and both directions reference the same immutable edge identifier.
"""
struct OrderedTopology
    child_offsets::Memory{UInt16}
    child_edges::Memory{UInt16}
    parent_offsets::Memory{UInt16}
    parent_edges::Memory{UInt16}
    edge_sources::Memory{UInt16}
    edge_destinations::Memory{UInt16}
    edge_kinds::Memory{UInt8}
    edge_slots::Memory{UInt8}
    edge_roles::Memory{UInt8}
    node_classes::Memory{UInt8}
    node_planes::Memory{UInt8}
    node_slots::Memory{UInt16}
    interval_firsts::Memory{UInt8}
    interval_lasts::Memory{UInt8}
end

function _fixed_memory(values::AbstractVector{T}) where {T}
    result = Memory{T}(undef, length(values))
    copyto!(result, values)
    return result
end

# A positive tree code names an already-built internal node; a negative code
# names an ordered axis leaf. Post-order numbering makes every internal child
# smaller than its parent and reserves the final index for the root.
function _balanced_binary_template(length::Int)
    left = Vector{Int16}(undef, length - 1)
    right = Vector{Int16}(undef, length - 1)
    firsts = Vector{UInt8}(undef, length - 1)
    lasts = Vector{UInt8}(undef, length - 1)
    next_internal = Ref(0)

    function build(first::Int, last::Int)
        first == last && return Int16(-first)
        midpoint = (first + last) >>> 1
        left_code = build(first, midpoint)
        right_code = build(midpoint + 1, last)
        next_internal[] += 1
        internal = next_internal[]
        left[internal] = left_code
        right[internal] = right_code
        firsts[internal] = UInt8(first)
        lasts[internal] = UInt8(last)
        return Int16(internal)
    end

    root = build(1, length)
    Int(root) == length - 1 || error("binary template root numbering drift")
    return Tuple(left), Tuple(right), Tuple(firsts), Tuple(lasts)
end

const _ROW_LEFT, _ROW_RIGHT, _ROW_FIRSTS, _ROW_LASTS =
    _balanced_binary_template(COLUMN_COUNT)
const _COLUMN_LEFT, _COLUMN_RIGHT, _COLUMN_FIRSTS, _COLUMN_LASTS =
    _balanced_binary_template(ROW_COUNT)

@inline function _row_child(
    plane::Int,
    row::Int,
    code::Int16,
)
    code < 0 && return spatial_node(plane, row, -Int(code))
    return row_internal_node(plane, row, Int(code))
end

@inline function _column_child(
    plane::Int,
    column::Int,
    code::Int16,
)
    code < 0 && return spatial_node(plane, -Int(code), column)
    return column_internal_node(plane, column, Int(code))
end

@inline function _add_edge!(
    sources,
    destinations,
    kinds,
    slots,
    roles,
    source::Integer,
    destination::Integer,
    kind::UInt8,
    slot::Integer,
    role::Integer,
)
    physical_source = Int(source)
    physical_destination = Int(destination)
    physical_source < physical_destination || error(
        "ordered topology edge must point to a later node",
    )
    push!(sources, UInt16(physical_source))
    push!(destinations, UInt16(physical_destination))
    push!(kinds, kind)
    push!(slots, UInt8(slot))
    push!(roles, UInt8(role))
    return nothing
end

@inline function _add_binary_edge!(
    sources,
    destinations,
    kinds,
    slots,
    roles,
    source,
    destination,
    slot,
)
    return _add_edge!(
        sources,
        destinations,
        kinds,
        slots,
        roles,
        source,
        destination,
        FULL_PACKET_EDGE,
        slot,
        0,
    )
end

@inline function _add_semantic_edge!(
    sources,
    destinations,
    kinds,
    slots,
    roles,
    source,
    destination,
    slot,
    role,
)
    return _add_edge!(
        sources,
        destinations,
        kinds,
        slots,
        roles,
        source,
        destination,
        PROJECTED_TRIPLET_EDGE,
        slot,
        role,
    )
end

function _flatten_edge_adjacency(edge_lists, slots)
    offsets = Vector{UInt16}(undef, NODE_COUNT + 1)
    edges = Vector{UInt16}()
    sizehint!(edges, EDGE_COUNT)
    offsets[1] = UInt16(1)
    @inbounds for node in 1:NODE_COUNT
        sort!(edge_lists[node]; by=edge -> (Int(slots[edge]), Int(edge)))
        append!(edges, edge_lists[node])
        offsets[node + 1] = UInt16(length(edges) + 1)
    end
    length(edges) == EDGE_COUNT || error("topology adjacency edge drift")
    return offsets, edges
end

@inline function _evidence_motif_slot(evidence::Int, family::Int)
    value = evidence - 1
    b0 = (value >>> 0) & 1
    b1 = (value >>> 1) & 1
    b2 = (value >>> 2) & 1
    b3 = (value >>> 3) & 1
    b4 = (value >>> 4) & 1
    low, high = family == 1 ? (b0, b1) :
                family == 2 ? (b2, b3) :
                family == 3 ? (b4, b0) :
                family == 4 ? (b1, b2) :
                family == 5 ? (b3, b4) :
                family == 6 ? (xor(b0, b2), xor(b1, b3)) :
                family == 7 ? (xor(b1, b4), xor(b2, b0)) :
                              (xor(b2, b3), xor(b4, b1))
    return 1 + low + 2 * high
end

function _build_topology()
    sources = UInt16[]
    destinations = UInt16[]
    kinds = UInt8[]
    slots = UInt8[]
    roles = UInt8[]
    sizehint!(sources, EDGE_COUNT)
    sizehint!(destinations, EDGE_COUNT)
    sizehint!(kinds, EDGE_COUNT)
    sizehint!(slots, EDGE_COUNT)
    sizehint!(roles, EDGE_COUNT)

    # Lossless ordered row trees.
    @inbounds for plane in 1:PLANE_COUNT, row in 1:ROW_COUNT
        for internal in 1:ROW_INTERNAL_PER_ROW
            destination = row_internal_node(plane, row, internal)
            _add_binary_edge!(
                sources, destinations, kinds, slots, roles,
                _row_child(plane, row, _ROW_LEFT[internal]),
                destination,
                1,
            )
            _add_binary_edge!(
                sources, destinations, kinds, slots, roles,
                _row_child(plane, row, _ROW_RIGHT[internal]),
                destination,
                2,
            )
        end
    end

    # Lossless ordered column trees.
    @inbounds for plane in 1:PLANE_COUNT, column in 1:COLUMN_COUNT
        for internal in 1:COLUMN_INTERNAL_PER_COLUMN
            destination = column_internal_node(plane, column, internal)
            _add_binary_edge!(
                sources, destinations, kinds, slots, roles,
                _column_child(plane, column, _COLUMN_LEFT[internal]),
                destination,
                1,
            )
            _add_binary_edge!(
                sources, destinations, kinds, slots, roles,
                _column_child(plane, column, _COLUMN_RIGHT[internal]),
                destination,
                2,
            )
        end
    end

    # Each motif compares the same two ordered row anchors and column anchors
    # across before/after planes. Sequential anchors distribute 64 uses per
    # axis group as evenly as integer counts permit: row roots 2/3, column
    # roots 6/7. No root becomes a geometry hub.
    @inbounds for motif in 1:MOTIF_COUNT
        family = div(motif - 1, MOTIF_SLOTS_PER_FAMILY) + 1
        family_slot = mod(motif - 1, MOTIF_SLOTS_PER_FAMILY) + 1
        destination = motif_node(family, family_slot)
        row_a = 1 + mod(2 * (motif - 1), ROW_COUNT)
        row_b = 1 + mod(2 * (motif - 1) + 1, ROW_COUNT)
        column_a = 1 + mod(2 * (motif - 1), COLUMN_COUNT)
        column_b = 1 + mod(2 * (motif - 1) + 1, COLUMN_COUNT)
        motif_sources = (
            row_root_node(BEFORE_PLANE, row_a),
            row_root_node(AFTER_PLANE, row_a),
            row_root_node(BEFORE_PLANE, row_b),
            row_root_node(AFTER_PLANE, row_b),
            column_root_node(BEFORE_PLANE, column_a),
            column_root_node(AFTER_PLANE, column_a),
            column_root_node(BEFORE_PLANE, column_b),
            column_root_node(AFTER_PLANE, column_b),
        )
        for role in 1:SEMANTIC_ROLE_COUNT
            _add_semantic_edge!(
                sources, destinations, kinds, slots, roles,
                motif_sources[role], destination, role, role,
            )
        end
    end

    # Every evidence cell receives exactly one source from each of the eight
    # motif families. The five-bit code yields 32 unique source signatures;
    # every motif slot is used exactly eight times.
    @inbounds for evidence in 1:EVIDENCE_COUNT
        destination = evidence_node(evidence)
        for family in 1:MOTIF_FAMILY_COUNT
            source = motif_node(
                family,
                _evidence_motif_slot(evidence, family),
            )
            _add_semantic_edge!(
                sources, destinations, kinds, slots, roles,
                source, destination, family, family,
            )
        end
    end

    # A quarter-cycle stride gives every output eight distinct evidence
    # sources and balances evidence fanout at five or six destinations.
    @inbounds for output in 1:OUTPUT_COUNT
        destination = output_node(output)
        for role in 1:SEMANTIC_ROLE_COUNT
            source_index = 1 + mod((output - 1) + 4 * (role - 1), EVIDENCE_COUNT)
            _add_semantic_edge!(
                sources, destinations, kinds, slots, roles,
                evidence_node(source_index), destination, role, role,
            )
        end
    end

    length(sources) == EDGE_COUNT || error("ordered edge count drift")

    incoming = [UInt16[] for _ in 1:NODE_COUNT]
    outgoing = [UInt16[] for _ in 1:NODE_COUNT]
    @inbounds for edge in 1:EDGE_COUNT
        push!(incoming[Int(destinations[edge])], UInt16(edge))
        push!(outgoing[Int(sources[edge])], UInt16(edge))
    end
    child_offsets, child_edges = _flatten_edge_adjacency(incoming, slots)
    parent_offsets, parent_edges = _flatten_edge_adjacency(outgoing, slots)

    classes = Vector{UInt8}(undef, NODE_COUNT)
    planes = fill(UInt8(0), NODE_COUNT)
    node_slots = Vector{UInt16}(undef, NODE_COUNT)
    firsts = fill(UInt8(0), NODE_COUNT)
    lasts = fill(UInt8(0), NODE_COUNT)

    @inbounds for plane in 1:PLANE_COUNT, position in 1:SPATIAL_COUNT_PER_PLANE
        node = Int(spatial_node(plane, position))
        classes[node] = SPATIAL_CLASS
        planes[node] = UInt8(plane)
        node_slots[node] = UInt16(position)
    end
    @inbounds for plane in 1:PLANE_COUNT, row in 1:ROW_COUNT,
                  internal in 1:ROW_INTERNAL_PER_ROW
        node = Int(row_internal_node(plane, row, internal))
        classes[node] = ROW_INTERNAL_CLASS
        planes[node] = UInt8(plane)
        node_slots[node] = UInt16((row - 1) * ROW_INTERNAL_PER_ROW + internal)
        firsts[node] = _ROW_FIRSTS[internal]
        lasts[node] = _ROW_LASTS[internal]
    end
    @inbounds for plane in 1:PLANE_COUNT, column in 1:COLUMN_COUNT,
                  internal in 1:COLUMN_INTERNAL_PER_COLUMN
        node = Int(column_internal_node(plane, column, internal))
        classes[node] = COLUMN_INTERNAL_CLASS
        planes[node] = UInt8(plane)
        node_slots[node] = UInt16(
            (column - 1) * COLUMN_INTERNAL_PER_COLUMN + internal,
        )
        firsts[node] = _COLUMN_FIRSTS[internal]
        lasts[node] = _COLUMN_LASTS[internal]
    end
    @inbounds for family in 1:MOTIF_FAMILY_COUNT,
                  family_slot in 1:MOTIF_SLOTS_PER_FAMILY
        node = Int(motif_node(family, family_slot))
        classes[node] = MOTIF_CLASS
        node_slots[node] = UInt16(
            (family - 1) * MOTIF_SLOTS_PER_FAMILY + family_slot,
        )
    end
    @inbounds for evidence in 1:EVIDENCE_COUNT
        node = Int(evidence_node(evidence))
        classes[node] = EVIDENCE_CLASS
        node_slots[node] = UInt16(evidence)
    end
    @inbounds for output in 1:OUTPUT_COUNT
        node = Int(output_node(output))
        classes[node] = OUTPUT_CLASS
        node_slots[node] = UInt16(output)
    end

    topology = OrderedTopology(
        _fixed_memory(child_offsets),
        _fixed_memory(child_edges),
        _fixed_memory(parent_offsets),
        _fixed_memory(parent_edges),
        _fixed_memory(sources),
        _fixed_memory(destinations),
        _fixed_memory(kinds),
        _fixed_memory(slots),
        _fixed_memory(roles),
        _fixed_memory(classes),
        _fixed_memory(planes),
        _fixed_memory(node_slots),
        _fixed_memory(firsts),
        _fixed_memory(lasts),
    )
    validate_topology(topology)
    return topology
end

@inline function child_count(topology::OrderedTopology, node::Integer)
    physical_node = _check_node(node)
    return Int(topology.child_offsets[physical_node + 1]) -
           Int(topology.child_offsets[physical_node])
end

@inline function child_edge(
    topology::OrderedTopology,
    node::Integer,
    index::Integer,
)
    physical_node = _check_node(node)
    count = child_count(topology, physical_node)
    1 <= index <= count || throw(BoundsError(1:count, index))
    flat = Int(topology.child_offsets[physical_node]) + Int(index) - 1
    return @inbounds topology.child_edges[flat]
end

@inline child_node(topology::OrderedTopology, node::Integer, index::Integer) =
    edge_source(topology, child_edge(topology, node, index))

@inline child_slot(topology::OrderedTopology, node::Integer, index::Integer) =
    @inbounds topology.edge_slots[Int(child_edge(topology, node, index))]

@inline function parent_count(topology::OrderedTopology, node::Integer)
    physical_node = _check_node(node)
    return Int(topology.parent_offsets[physical_node + 1]) -
           Int(topology.parent_offsets[physical_node])
end

@inline function parent_edge(
    topology::OrderedTopology,
    node::Integer,
    index::Integer,
)
    physical_node = _check_node(node)
    count = parent_count(topology, physical_node)
    1 <= index <= count || throw(BoundsError(1:count, index))
    flat = Int(topology.parent_offsets[physical_node]) + Int(index) - 1
    return @inbounds topology.parent_edges[flat]
end

@inline parent_node(topology::OrderedTopology, node::Integer, index::Integer) =
    edge_destination(topology, parent_edge(topology, node, index))

@inline edge_source(topology::OrderedTopology, edge::Integer) =
    @inbounds topology.edge_sources[_check_edge(edge)]

@inline edge_destination(topology::OrderedTopology, edge::Integer) =
    @inbounds topology.edge_destinations[_check_edge(edge)]

@inline edge_kind(topology::OrderedTopology, edge::Integer) =
    @inbounds topology.edge_kinds[_check_edge(edge)]

@inline edge_semantic_role(topology::OrderedTopology, edge::Integer) =
    @inbounds topology.edge_roles[_check_edge(edge)]

@inline node_class(topology::OrderedTopology, node::Integer) =
    @inbounds topology.node_classes[_check_node(node)]

@inline node_plane(topology::OrderedTopology, node::Integer) =
    @inbounds topology.node_planes[_check_node(node)]

@inline node_slot(topology::OrderedTopology, node::Integer) =
    @inbounds topology.node_slots[_check_node(node)]

@inline interval_first(topology::OrderedTopology, node::Integer) =
    @inbounds topology.interval_firsts[_check_node(node)]

@inline interval_last(topology::OrderedTopology, node::Integer) =
    @inbounds topology.interval_lasts[_check_node(node)]

@inline function motif_family(topology::OrderedTopology, node::Integer)
    node_class(topology, node) == MOTIF_CLASS || throw(ArgumentError(
        "node $(Int(node)) is not a motif node",
    ))
    return UInt8(div(Int(node_slot(topology, node)) - 1,
                     MOTIF_SLOTS_PER_FAMILY) + 1)
end

@inline function motif_slot(topology::OrderedTopology, node::Integer)
    node_class(topology, node) == MOTIF_CLASS || throw(ArgumentError(
        "node $(Int(node)) is not a motif node",
    ))
    return UInt8(mod(Int(node_slot(topology, node)) - 1,
                     MOTIF_SLOTS_PER_FAMILY) + 1)
end

"""Fail closed if any fixed indexing, fan-in or edge-role invariant drifts."""
function validate_topology(topology::OrderedTopology)
    length(topology.edge_sources) == EDGE_COUNT || error("source edge drift")
    length(topology.edge_destinations) == EDGE_COUNT ||
        error("destination edge drift")
    topology.child_offsets[1] == UInt16(1) || error("child CSR origin drift")
    topology.parent_offsets[1] == UInt16(1) || error("parent CSR origin drift")
    topology.child_offsets[end] == UInt16(EDGE_COUNT + 1) ||
        error("child CSR limit drift")
    topology.parent_offsets[end] == UInt16(EDGE_COUNT + 1) ||
        error("parent CSR limit drift")

    child_seen = falses(EDGE_COUNT)
    parent_seen = falses(EDGE_COUNT)
    @inbounds for node in 1:NODE_COUNT
        count = child_count(topology, node)
        class = node_class(topology, node)
        expected = class == SPATIAL_CLASS ? 0 :
                   class == ROW_INTERNAL_CLASS ||
                   class == COLUMN_INTERNAL_CLASS ? 2 : SEMANTIC_FANIN
        count == expected || error("node $node fan-in drift")
        slots_seen = UInt16(0)
        prior_source = 0
        for index in 1:count
            edge = Int(child_edge(topology, node, index))
            child_seen[edge] && error("duplicate child adjacency edge")
            child_seen[edge] = true
            source = Int(edge_source(topology, edge))
            destination = Int(edge_destination(topology, edge))
            destination == node || error("child adjacency destination drift")
            source < destination || error("topological edge order drift")
            source != prior_source || error("duplicate source for one receiver")
            prior_source = source
            slot = Int(topology.edge_slots[edge])
            1 <= slot <= expected || error("child slot drift")
            mask = UInt16(1) << (slot - 1)
            iszero(slots_seen & mask) || error("duplicate child slot")
            slots_seen |= mask
            if expected == 2
                edge_kind(topology, edge) == FULL_PACKET_EDGE ||
                    error("binary edge lost full packet identity")
                edge_semantic_role(topology, edge) == UInt8(0) ||
                    error("binary edge acquired a semantic role")
            else
                edge_kind(topology, edge) == PROJECTED_TRIPLET_EDGE ||
                    error("semantic edge lost projection identity")
                edge_semantic_role(topology, edge) == UInt8(slot) ||
                    error("semantic role/branch identity drift")
            end
        end
        expected_mask = expected == 0 ? UInt16(0) :
            (UInt16(1) << expected) - UInt16(1)
        slots_seen == expected_mask || error("node $node child slots are incomplete")

        for index in 1:parent_count(topology, node)
            edge = Int(parent_edge(topology, node, index))
            parent_seen[edge] && error("duplicate parent adjacency edge")
            parent_seen[edge] = true
            Int(edge_source(topology, edge)) == node ||
                error("parent adjacency source drift")
        end
    end
    all(child_seen) || error("child adjacency omitted an edge")
    all(parent_seen) || error("parent adjacency omitted an edge")

    # Every spatial source has one row-tree and one column-tree parent.
    @inbounds for plane in 1:PLANE_COUNT, position in 1:SPATIAL_COUNT_PER_PLANE
        node = Int(spatial_node(plane, position))
        parent_count(topology, node) == 2 ||
            error("spatial source must enter both crossed covers")
        parent_classes = (
            node_class(topology, parent_node(topology, node, 1)),
            node_class(topology, parent_node(topology, node, 2)),
        )
            Set(parent_classes) == Set((ROW_INTERNAL_CLASS, COLUMN_INTERNAL_CLASS)) ||
            error("spatial crossed-cover identity drift")
    end
    return topology
end

const _CANONICAL_TOPOLOGY = _build_topology()

"""Return the process-wide immutable topology."""
@inline canonical_topology() = _CANONICAL_TOPOLOGY

"""Caller-owned fixed-capacity exact forward ancestor closure."""
mutable struct AffectedClosure <: AbstractVector{UInt16}
    nodes::Memory{UInt16}
    marked::Memory{UInt8}
    count::Int
    function AffectedClosure()
        nodes = Memory{UInt16}(undef, NODE_COUNT)
        marked = Memory{UInt8}(undef, NODE_COUNT)
        fill!(nodes, UInt16(0))
        fill!(marked, UInt8(0))
        return new(nodes, marked, 0)
    end
end

@inline affected_count(closure::AffectedClosure) = closure.count
Base.IndexStyle(::Type{AffectedClosure}) = IndexLinear()
Base.size(closure::AffectedClosure) = (closure.count,)
Base.length(closure::AffectedClosure) = closure.count
@inline Base.getindex(closure::AffectedClosure, index::Int) =
    affected_forward_node(closure, index)

@inline function affected_forward_node(
    closure::AffectedClosure,
    index::Integer,
)
    1 <= index <= closure.count || throw(BoundsError(1:closure.count, index))
    return @inbounds closure.nodes[Int(index)]
end

@inline function affected_reverse_node(
    closure::AffectedClosure,
    index::Integer,
)
    1 <= index <= closure.count || throw(BoundsError(1:closure.count, index))
    return @inbounds closure.nodes[closure.count - Int(index) + 1]
end

"""
Fill the exact downstream ancestor closure of unique or duplicate seed nodes.

Node IDs are topological and every edge points forward. Newly marked parents
are therefore encountered later in the same bounded scan; no queue, hashing,
allocation, or global mutable state is used.
"""
function fill_affected_closure!(
    closure::AffectedClosure,
    topology::OrderedTopology,
    seeds,
    seed_count::Integer,
)
    physical_count = Int(seed_count)
    0 <= physical_count <= length(seeds) ||
        throw(BoundsError(seeds, physical_count))
    fill!(closure.marked, UInt8(0))
    @inbounds for index in 1:physical_count
        closure.marked[_check_node(seeds[index])] = UInt8(1)
    end

    @inbounds for node in 1:NODE_COUNT
        iszero(closure.marked[node]) && continue
        first = Int(topology.parent_offsets[node])
        limit = Int(topology.parent_offsets[node + 1]) - 1
        for flat in first:limit
            edge = Int(topology.parent_edges[flat])
            destination = Int(topology.edge_destinations[edge])
            closure.marked[destination] = UInt8(1)
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

@inline function fill_affected_closure!(
    closure::AffectedClosure,
    topology::OrderedTopology,
    seeds,
)
    return fill_affected_closure!(closure, topology, seeds, length(seeds))
end

@inline function fill_affected_closure!(closure::AffectedClosure, seeds)
    return fill_affected_closure!(
        closure,
        _CANONICAL_TOPOLOGY,
        seeds,
        length(seeds),
    )
end

"""Map board positions in one semantic plane to their exact ancestor closure."""
function fill_spatial_affected_closure!(
    closure::AffectedClosure,
    topology::OrderedTopology,
    plane::Integer,
    positions,
    position_count::Integer,
)
    physical_plane = _check_plane(plane)
    physical_count = Int(position_count)
    0 <= physical_count <= length(positions) ||
        throw(BoundsError(positions, physical_count))
    fill!(closure.marked, UInt8(0))
    @inbounds for index in 1:physical_count
        position = _check_position(positions[index])
        closure.marked[Int(spatial_node(physical_plane, position))] = UInt8(1)
    end

    @inbounds for node in 1:NODE_COUNT
        iszero(closure.marked[node]) && continue
        first = Int(topology.parent_offsets[node])
        limit = Int(topology.parent_offsets[node + 1]) - 1
        for flat in first:limit
            edge = Int(topology.parent_edges[flat])
            destination = Int(topology.edge_destinations[edge])
            closure.marked[destination] = UInt8(1)
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

@inline function fill_spatial_affected_closure!(
    closure::AffectedClosure,
    topology::OrderedTopology,
    plane::Integer,
    positions,
)
    return fill_spatial_affected_closure!(
        closure,
        topology,
        plane,
        positions,
        length(positions),
    )
end

end # module OrderedMultiscaleTopology
