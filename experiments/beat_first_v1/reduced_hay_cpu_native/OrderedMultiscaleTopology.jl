module OrderedMultiscaleTopology

"""
Immutable topology contract for the ordered multiscale Reduced-Hay graph.

The graph has two explicit 24 x 10 spatial planes.  Every spatial node feeds
one ordered binary row tree and one ordered binary column tree.  A separate,
candidate-derived incidence object connects the information spine and typed
candidate inputs to 32 semantic motif cells.  Static motif-to-evidence and
evidence-to-output edges remain immutable.  Node identifiers are a strict
topological order, so a caller-owned affected closure can be propagated with
one forward scan and no queue or allocation.

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
       MOTIF_SOURCE_CAPACITY,
       MOTIF_SPATIAL_SOURCE,
       MOTIF_ROW_ROOT_SOURCE,
       MOTIF_COLUMN_ROOT_SOURCE,
       MOTIF_RAW_PLACEMENT_SOURCE,
       MOTIF_ROW_REMAP_SOURCE,
       MOTIF_CLEARED_ROW_SOURCE,
       MOTIF_OUTSIDE_SOURCE,
       MOTIF_ABSENT_SOURCE,
       MOTIF_QUEUE_SOURCE,
       MOTIF_REN_WORD_SOURCE,
       MOTIF_BOOLEAN_SOURCE,
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
       CandidateMotifContext,
       CandidateMotifIncidence,
       CandidateMotifSource,
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
       fill_candidate_motif_incidence!,
       fill_changed_motif_closure!,
       fill_incidence_affected_closure!,
       fill_spatial_affected_closure!,
       interval_first,
       interval_last,
       motif_family,
       motif_node,
       motif_source,
       motif_source_count,
       motif_source_is_spine,
       materialize_external_motif_packet!,
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
const MOTIF_SOURCE_CAPACITY = SEMANTIC_FANIN

# Candidate-derived motif source kinds.  Zero is never a semantic source.
const MOTIF_SPATIAL_SOURCE = UInt8(1)
const MOTIF_ROW_ROOT_SOURCE = UInt8(2)
const MOTIF_COLUMN_ROOT_SOURCE = UInt8(3)
const MOTIF_RAW_PLACEMENT_SOURCE = UInt8(4)
const MOTIF_ROW_REMAP_SOURCE = UInt8(5)
const MOTIF_CLEARED_ROW_SOURCE = UInt8(6)
const MOTIF_OUTSIDE_SOURCE = UInt8(7)
const MOTIF_ABSENT_SOURCE = UInt8(8)
const MOTIF_QUEUE_SOURCE = UInt8(9)
const MOTIF_REN_WORD_SOURCE = UInt8(10)
const MOTIF_BOOLEAN_SOURCE = UInt8(11)

const _SPATIAL_FIRST = 1
const _ROW_INTERNAL_FIRST = _SPATIAL_FIRST + SPATIAL_COUNT
const _COLUMN_INTERNAL_FIRST = _ROW_INTERNAL_FIRST + ROW_INTERNAL_COUNT
const _MOTIF_FIRST = _COLUMN_INTERNAL_FIRST + COLUMN_INTERNAL_COUNT
const _EVIDENCE_FIRST = _MOTIF_FIRST + MOTIF_COUNT
const _OUTPUT_FIRST = _EVIDENCE_FIRST + EVIDENCE_COUNT
const NODE_COUNT = _OUTPUT_FIRST + OUTPUT_COUNT - 1

# Two children for every information-spine internal plus eight static semantic
# sources for every evidence and output cell.  Motif input incidence is
# candidate-derived and deliberately excluded from the static edge table.
const _SPINE_EDGE_COUNT = 2 * (ROW_INTERNAL_COUNT + COLUMN_INTERNAL_COUNT)
const _EVIDENCE_EDGE_COUNT = EVIDENCE_COUNT * SEMANTIC_FANIN
const _OUTPUT_EDGE_COUNT = OUTPUT_COUNT * SEMANTIC_FANIN
const EDGE_COUNT = _SPINE_EDGE_COUNT + _EVIDENCE_EDGE_COUNT +
                   _OUTPUT_EDGE_COUNT

@assert SPATIAL_COUNT == 480
@assert ROW_INTERNAL_COUNT == 432
@assert COLUMN_INTERNAL_COUNT == 460
@assert NODE_COUNT == 1_458
@assert EDGE_COUNT == 2_216
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
Target-free, isbits candidate metadata used only to derive motif incidence.

`placement_positions` uses the canonical column-major board index
`row + (column-1)*24`; used entries are strictly increasing and unused entries
are zero storage sentinels.  A sentinel is never materialized as an observed
semantic value: the incidence contains an explicit `MOTIF_ABSENT_SOURCE`.
`row_remap` is the exact source-row to after-row map `mu`; cleared rows map to
zero and are cross-checked against `full_row_mask`.
"""
struct CandidateMotifContext
    placement_positions::NTuple{4,UInt16}
    placement_count::UInt8
    row_remap::NTuple{ROW_COUNT,UInt8}
    full_row_mask::UInt32
    cleared_rows::UInt8
    hold_piece::UInt8
    next_pieces::NTuple{5,UInt8}
    ren::Int32
    b2b::UInt8
    tspin::UInt8

    function CandidateMotifContext(
        placement_positions::NTuple{4,UInt16},
        placement_count::Integer,
        row_remap::NTuple{ROW_COUNT,UInt8},
        full_row_mask::UInt32,
        cleared_rows::Integer,
        hold_piece::UInt8,
        next_pieces::NTuple{5,UInt8},
        ren::Integer,
        b2b::UInt8,
        tspin::UInt8,
    )
        0 <= placement_count <= 4 || throw(BoundsError(0:4, placement_count))
        physical_count = Int(placement_count)
        prior = 0
        @inbounds for index in 1:4
            position = Int(placement_positions[index])
            if index <= physical_count
                1 <= position <= SPATIAL_COUNT_PER_PLANE || throw(
                    BoundsError(1:SPATIAL_COUNT_PER_PLANE, position),
                )
                position > prior || throw(ArgumentError(
                    "placement positions must be strictly column-major sorted",
                ))
                prior = position
            else
                iszero(position) || throw(ArgumentError(
                    "unused placement positions must be zero sentinels",
                ))
            end
        end
        iszero(full_row_mask >> ROW_COUNT) || throw(ArgumentError(
            "full-row mask contains bits outside the 24-row board",
        ))
        0 <= cleared_rows <= 4 || throw(BoundsError(0:4, cleared_rows))
        count_ones(full_row_mask) == cleared_rows || throw(ArgumentError(
            "cleared-row count and full-row mask disagree",
        ))
        destination = Int(cleared_rows) + 1
        @inbounds for row in 1:ROW_COUNT
            cleared = !iszero(full_row_mask & (UInt32(1) << (row - 1)))
            mapped = Int(row_remap[row])
            if cleared
                iszero(mapped) || throw(ArgumentError(
                    "cleared source row $row must map to zero",
                ))
            else
                mapped == destination || throw(ArgumentError(
                    "row remap is not the exact stable clear compaction",
                ))
                destination += 1
            end
        end
        destination == ROW_COUNT + 1 || error("row-remap terminal drift")
        1 <= hold_piece <= 8 || throw(ArgumentError(
            "HOLD must be a typed NONE/I/O/T/S/Z/J/L token",
        ))
        all(piece -> 2 <= piece <= 8, next_pieces) || throw(ArgumentError(
            "NEXT1--NEXT5 must be concrete typed piece tokens",
        ))
        0 <= ren <= typemax(Int32) || throw(ArgumentError(
            "REN must be an exact nonnegative Int32 value",
        ))
        1 <= b2b <= 2 || throw(ArgumentError("B2B must be explicit FALSE/TRUE"))
        1 <= tspin <= 2 || throw(ArgumentError(
            "T-spin must be explicit FALSE/TRUE",
        ))
        return new(
            placement_positions,
            UInt8(physical_count),
            row_remap,
            full_row_mask,
            UInt8(cleared_rows),
            hold_piece,
            next_pieces,
            Int32(ren),
            b2b,
            tspin,
        )
    end
end

"""One ordered candidate-derived source deposited onto one motif branch."""
struct CandidateMotifSource
    kind::UInt8
    plane::UInt8
    node::UInt16
    row::UInt8
    column::UInt8
    placement_slot::UInt8
    context_slot::UInt8
    branch_slot::UInt8
    value::Int32
end

const _EMPTY_MOTIF_SOURCE = CandidateMotifSource(
    UInt8(0), UInt8(0), UInt16(0), UInt8(0), UInt8(0), UInt8(0),
    UInt8(0), UInt8(0), Int32(0),
)

"""Caller-owned fixed 8 x 32 candidate motif incidence."""
mutable struct CandidateMotifIncidence
    sources::Memory{CandidateMotifSource}
    counts::Memory{UInt8}
    function CandidateMotifIncidence()
        sources = Memory{CandidateMotifSource}(
            undef,
            MOTIF_SOURCE_CAPACITY * MOTIF_COUNT,
        )
        counts = Memory{UInt8}(undef, MOTIF_COUNT)
        fill!(sources, _EMPTY_MOTIF_SOURCE)
        fill!(counts, UInt8(0))
        return new(sources, counts)
    end
end

@inline function _motif_source_index(motif::Integer, rank::Integer)
    1 <= motif <= MOTIF_COUNT || throw(BoundsError(1:MOTIF_COUNT, motif))
    1 <= rank <= MOTIF_SOURCE_CAPACITY || throw(
        BoundsError(1:MOTIF_SOURCE_CAPACITY, rank),
    )
    return (Int(motif) - 1) * MOTIF_SOURCE_CAPACITY + Int(rank)
end

@inline function motif_source_count(
    incidence::CandidateMotifIncidence,
    motif::Integer,
)
    1 <= motif <= MOTIF_COUNT || throw(BoundsError(1:MOTIF_COUNT, motif))
    return Int(@inbounds incidence.counts[Int(motif)])
end

@inline function motif_source(
    incidence::CandidateMotifIncidence,
    motif::Integer,
    rank::Integer,
)
    count = motif_source_count(incidence, motif)
    1 <= rank <= count || throw(BoundsError(1:count, rank))
    return @inbounds incidence.sources[_motif_source_index(motif, rank)]
end

@inline motif_source_is_spine(source::CandidateMotifSource) =
    source.kind == MOTIF_SPATIAL_SOURCE ||
    source.kind == MOTIF_ROW_ROOT_SOURCE ||
    source.kind == MOTIF_COLUMN_ROOT_SOURCE

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
        expected = class == SPATIAL_CLASS || class == MOTIF_CLASS ? 0 :
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

@inline function _placement_row_column(position::UInt16)
    physical = Int(position)
    row = mod(physical - 1, ROW_COUNT) + 1
    column = div(physical - 1, ROW_COUNT) + 1
    return row, column
end

@inline function _candidate_source(
    kind::UInt8,
    branch::Integer;
    plane::Integer=0,
    node::Integer=0,
    row::Integer=0,
    column::Integer=0,
    placement_slot::Integer=0,
    context_slot::Integer=0,
    value::Integer=0,
)
    1 <= branch <= MOTIF_SOURCE_CAPACITY || throw(
        BoundsError(1:MOTIF_SOURCE_CAPACITY, branch),
    )
    0 <= plane <= PLANE_COUNT || throw(BoundsError(0:PLANE_COUNT, plane))
    0 <= node <= NODE_COUNT || throw(BoundsError(0:NODE_COUNT, node))
    0 <= row <= ROW_COUNT + 1 || throw(BoundsError(0:ROW_COUNT + 1, row))
    0 <= column <= COLUMN_COUNT || throw(BoundsError(0:COLUMN_COUNT, column))
    0 <= placement_slot <= 4 || throw(BoundsError(0:4, placement_slot))
    0 <= context_slot <= typemax(UInt8) || throw(BoundsError(
        0:typemax(UInt8), context_slot,
    ))
    typemin(Int32) <= value <= typemax(Int32) || throw(InexactError(
        :CandidateMotifSource,
        Int32,
        value,
    ))
    spine = kind == MOTIF_SPATIAL_SOURCE ||
            kind == MOTIF_ROW_ROOT_SOURCE ||
            kind == MOTIF_COLUMN_ROOT_SOURCE
    spine == !iszero(node) || throw(ArgumentError(
        "only spine motif sources may own a static node identifier",
    ))
    return CandidateMotifSource(
        kind,
        UInt8(plane),
        UInt16(node),
        UInt8(row),
        UInt8(column),
        UInt8(placement_slot),
        UInt8(context_slot),
        UInt8(branch),
        Int32(value),
    )
end

@inline function _spatial_source(
    plane::Integer,
    row::Integer,
    column::Integer,
    placement_slot::Integer,
    branch::Integer,
)
    return _candidate_source(
        MOTIF_SPATIAL_SOURCE,
        branch;
        plane=plane,
        node=spatial_node(plane, row, column),
        row=row,
        column=column,
        placement_slot=placement_slot,
    )
end

@inline function _row_root_source(
    plane::Integer,
    row::Integer,
    placement_slot::Integer,
    branch::Integer,
)
    return _candidate_source(
        MOTIF_ROW_ROOT_SOURCE,
        branch;
        plane=plane,
        node=row_root_node(plane, row),
        row=row,
        placement_slot=placement_slot,
    )
end

@inline function _column_root_source(
    plane::Integer,
    column::Integer,
    placement_slot::Integer,
    branch::Integer,
)
    return _candidate_source(
        MOTIF_COLUMN_ROOT_SOURCE,
        branch;
        plane=plane,
        node=column_root_node(plane, column),
        column=column,
        placement_slot=placement_slot,
    )
end

@inline function _raw_source(
    context::CandidateMotifContext,
    placement_slot::Integer,
    branch::Integer,
)
    position = context.placement_positions[Int(placement_slot)]
    row, column = _placement_row_column(position)
    return _candidate_source(
        MOTIF_RAW_PLACEMENT_SOURCE,
        branch;
        row=row,
        column=column,
        placement_slot=placement_slot,
        value=Int(position),
    )
end

@inline function _absent_source(
    family::Integer,
    placement_slot::Integer,
    branch::Integer,
)
    return _candidate_source(
        MOTIF_ABSENT_SOURCE,
        branch;
        placement_slot=placement_slot,
        context_slot=family,
        value=family,
    )
end

@inline function _remap_source(
    context::CandidateMotifContext,
    placement_slot::Integer,
    branch::Integer,
)
    position = context.placement_positions[Int(placement_slot)]
    row, column = _placement_row_column(position)
    mapped = Int(context.row_remap[row])
    kind = iszero(mapped) ? MOTIF_CLEARED_ROW_SOURCE : MOTIF_ROW_REMAP_SOURCE
    return _candidate_source(
        kind,
        branch;
        row=row,
        column=column,
        placement_slot=placement_slot,
        value=mapped,
    )
end

@inline function _push_motif_source!(
    incidence::CandidateMotifIncidence,
    motif::Integer,
    source::CandidateMotifSource,
)
    count = motif_source_count(incidence, motif)
    count < MOTIF_SOURCE_CAPACITY || error("motif fan-in exceeds eight")
    @inbounds for rank in 1:count
        previous = incidence.sources[_motif_source_index(motif, rank)]
        previous.branch_slot == source.branch_slot && error(
            "candidate motif branch identity is not unique",
        )
    end
    next = count + 1
    incidence.sources[_motif_source_index(motif, next)] = source
    incidence.counts[Int(motif)] = UInt8(next)
    return nothing
end

@inline function _fill_slot_local_families!(
    incidence::CandidateMotifIncidence,
    context::CandidateMotifContext,
    placement_slot::Int,
)
    if placement_slot > Int(context.placement_count)
        @inbounds for family in 1:6
            _push_motif_source!(
                incidence,
                (family - 1) * MOTIF_SLOTS_PER_FAMILY + placement_slot,
                _absent_source(family, placement_slot, 1),
            )
        end
        return nothing
    end

    position = context.placement_positions[placement_slot]
    row, column = _placement_row_column(position)
    mapped = Int(context.row_remap[row])

    # 1. Placement-centred 3x3 patch.  Spatial packets already encode the
    # ordered centre and eight neighbours; before/after remain separate.
    motif = placement_slot
    _push_motif_source!(incidence, motif,
        _spatial_source(BEFORE_PLANE, row, column, placement_slot, 1))
    if iszero(mapped)
        _push_motif_source!(incidence, motif,
            _remap_source(context, placement_slot, 2))
    else
        _push_motif_source!(incidence, motif,
            _spatial_source(AFTER_PLANE, mapped, column, placement_slot, 2))
    end
    _push_motif_source!(incidence, motif,
        _raw_source(context, placement_slot, 3))
    _push_motif_source!(incidence, motif,
        _remap_source(context, placement_slot, 4))

    # 2. Landing/support: raw footprint, the immediately lower before-site
    # (or explicit OUTSIDE), after-site/remap, and exact row remap.
    motif = MOTIF_SLOTS_PER_FAMILY + placement_slot
    _push_motif_source!(incidence, motif,
        _raw_source(context, placement_slot, 1))
    if row == ROW_COUNT
        _push_motif_source!(incidence, motif, _candidate_source(
            MOTIF_OUTSIDE_SOURCE,
            2;
            row=ROW_COUNT + 1,
            column=column,
            placement_slot=placement_slot,
        ))
    else
        _push_motif_source!(incidence, motif,
            _spatial_source(BEFORE_PLANE, row + 1, column, placement_slot, 2))
    end
    if iszero(mapped)
        _push_motif_source!(incidence, motif,
            _remap_source(context, placement_slot, 3))
    else
        _push_motif_source!(incidence, motif,
            _spatial_source(AFTER_PLANE, mapped, column, placement_slot, 3))
    end
    _push_motif_source!(incidence, motif,
        _remap_source(context, placement_slot, 4))

    # 3. Touched row.
    motif = 2 * MOTIF_SLOTS_PER_FAMILY + placement_slot
    _push_motif_source!(incidence, motif,
        _row_root_source(BEFORE_PLANE, row, placement_slot, 1))
    if iszero(mapped)
        _push_motif_source!(incidence, motif,
            _remap_source(context, placement_slot, 2))
    else
        _push_motif_source!(incidence, motif,
            _row_root_source(AFTER_PLANE, mapped, placement_slot, 2))
    end
    _push_motif_source!(incidence, motif,
        _raw_source(context, placement_slot, 3))
    _push_motif_source!(incidence, motif,
        _remap_source(context, placement_slot, 4))

    # 4. Touched column.
    motif = 3 * MOTIF_SLOTS_PER_FAMILY + placement_slot
    _push_motif_source!(incidence, motif,
        _column_root_source(BEFORE_PLANE, column, placement_slot, 1))
    _push_motif_source!(incidence, motif,
        _column_root_source(AFTER_PLANE, column, placement_slot, 2))
    _push_motif_source!(incidence, motif,
        _raw_source(context, placement_slot, 3))
    _push_motif_source!(incidence, motif,
        _remap_source(context, placement_slot, 4))

    # 5. Three ordered cuts through the six-row band containing the site.
    motif = 4 * MOTIF_SLOTS_PER_FAMILY + placement_slot
    band_first = 1 + 6 * div(row - 1, 6)
    band_rows = (band_first, band_first + 2, band_first + 5)
    branch = 1
    for band_row in band_rows
        _push_motif_source!(incidence, motif,
            _row_root_source(BEFORE_PLANE, band_row, placement_slot, branch))
        branch += 1
        _push_motif_source!(incidence, motif,
            _row_root_source(AFTER_PLANE, band_row, placement_slot, branch))
        branch += 1
    end
    _push_motif_source!(incidence, motif,
        _raw_source(context, placement_slot, 7))
    _push_motif_source!(incidence, motif,
        _remap_source(context, placement_slot, 8))

    # 6. Ordered before/after roots for the three-column shard.  The final
    # one-column shard intentionally uses only four of eight branches.
    motif = 5 * MOTIF_SLOTS_PER_FAMILY + placement_slot
    shard_first = column <= 3 ? 1 : column <= 6 ? 4 : column <= 9 ? 7 : 10
    shard_last = min(shard_first + 2, COLUMN_COUNT)
    branch = 1
    for shard_column in shard_first:shard_last
        _push_motif_source!(incidence, motif, _column_root_source(
            BEFORE_PLANE, shard_column, placement_slot, branch,
        ))
        branch += 1
        _push_motif_source!(incidence, motif, _column_root_source(
            AFTER_PLANE, shard_column, placement_slot, branch,
        ))
        branch += 1
    end
    _push_motif_source!(incidence, motif,
        _raw_source(context, placement_slot, 7))
    _push_motif_source!(incidence, motif,
        _remap_source(context, placement_slot, 8))
    return nothing
end

@inline function _queue_source(
    context_slot::Integer,
    piece::UInt8,
    branch::Integer,
)
    return _candidate_source(
        MOTIF_QUEUE_SOURCE,
        branch;
        context_slot=context_slot,
        value=Int(piece),
    )
end

@inline function _ren_word_source(
    context::CandidateMotifContext,
    high::Bool,
    branch::Integer,
)
    bits = reinterpret(UInt32, context.ren)
    word = high ? UInt16(bits >> 16) : UInt16(bits & 0xffff)
    return _candidate_source(
        MOTIF_REN_WORD_SOURCE,
        branch;
        context_slot=high ? 8 : 7,
        value=Int(word),
    )
end

@inline function _boolean_source(
    context_slot::Integer,
    value::UInt8,
    branch::Integer,
)
    return _candidate_source(
        MOTIF_BOOLEAN_SOURCE,
        branch;
        context_slot=context_slot,
        value=Int(value),
    )
end

function _fill_global_families!(
    incidence::CandidateMotifIncidence,
    context::CandidateMotifContext,
)
    # 7. Every view receives all four ordered footprint slots and all four
    # corresponding remap/clear tokens.  ABSENT is explicit in both halves.
    @inbounds for view in 1:MOTIF_SLOTS_PER_FAMILY
        motif = 6 * MOTIF_SLOTS_PER_FAMILY + view
        for placement_slot in 1:4
            if placement_slot <= Int(context.placement_count)
                _push_motif_source!(incidence, motif,
                    _raw_source(context, placement_slot, placement_slot))
            else
                _push_motif_source!(incidence, motif, _absent_source(
                    7, placement_slot, placement_slot,
                ))
            end
        end
        for placement_slot in 1:4
            if placement_slot <= Int(context.placement_count)
                _push_motif_source!(incidence, motif, _remap_source(
                    context, placement_slot, 4 + placement_slot,
                ))
            else
                _push_motif_source!(incidence, motif, _candidate_source(
                    MOTIF_ABSENT_SOURCE,
                    4 + placement_slot;
                    placement_slot=placement_slot,
                    context_slot=8,
                    value=8,
                ))
            end
        end
    end

    # 8. Nine semantic items cannot be aliased into eight branches.  Four
    # context views retain queue order and split exact Int32 REN into two
    # independently role-bound UInt16 words.
    family_first = 7 * MOTIF_SLOTS_PER_FAMILY
    motif = family_first + 1
    _push_motif_source!(incidence, motif,
        _queue_source(1, context.hold_piece, 1))
    @inbounds for role in 1:5
        _push_motif_source!(incidence, motif,
            _queue_source(role + 1, context.next_pieces[role], role + 1))
    end

    motif = family_first + 2
    _push_motif_source!(incidence, motif, _ren_word_source(context, false, 1))
    _push_motif_source!(incidence, motif, _ren_word_source(context, true, 2))
    _push_motif_source!(incidence, motif,
        _boolean_source(9, context.b2b, 3))
    _push_motif_source!(incidence, motif,
        _boolean_source(10, context.tspin, 4))

    motif = family_first + 3
    @inbounds for role in 1:5
        _push_motif_source!(incidence, motif,
            _queue_source(role + 1, context.next_pieces[role], role))
    end
    _push_motif_source!(incidence, motif,
        _boolean_source(10, context.tspin, 6))

    motif = family_first + 4
    _push_motif_source!(incidence, motif,
        _queue_source(1, context.hold_piece, 1))
    _push_motif_source!(incidence, motif, _ren_word_source(context, false, 2))
    _push_motif_source!(incidence, motif, _ren_word_source(context, true, 3))
    _push_motif_source!(incidence, motif,
        _boolean_source(9, context.b2b, 4))
    _push_motif_source!(incidence, motif,
        _boolean_source(10, context.tspin, 5))
    return nothing
end

"""
Build the complete teacher-free candidate motif incidence.

Families 1--6 are indexed by canonical raw-placement slot.  Family 7 exposes
the whole ordered footprint and exact row clear/remap.  Family 8 partitions
ordered queue and meta without packing independent semantic items together.
"""
function fill_candidate_motif_incidence!(
    incidence::CandidateMotifIncidence,
    ::OrderedTopology,
    context::CandidateMotifContext,
)
    fill!(incidence.sources, _EMPTY_MOTIF_SOURCE)
    fill!(incidence.counts, UInt8(0))
    @inbounds for placement_slot in 1:MOTIF_SLOTS_PER_FAMILY
        _fill_slot_local_families!(incidence, context, placement_slot)
    end
    _fill_global_families!(incidence, context)
    return incidence
end

@inline fill_candidate_motif_incidence!(
    incidence::CandidateMotifIncidence,
    context::CandidateMotifContext,
) = fill_candidate_motif_incidence!(incidence, _CANONICAL_TOPOLOGY, context)

"""
Materialize one non-spine typed source as a bounded 12D packet.

The signed Int32 payload is split into four exact UInt8 lanes; REN is already
split into independently role-bound low/high UInt16 sources.  A spine source
must instead read the current 12D packet at `source.node`, preventing a static
descriptor from masquerading as live neural state.
"""
function materialize_external_motif_packet!(
    destination::AbstractVector{Float32},
    source::CandidateMotifSource,
)
    length(destination) == FULL_PACKET_WIDTH || throw(DimensionMismatch(
        "motif packet destination must have length 12",
    ))
    motif_source_is_spine(source) && throw(ArgumentError(
        "spine motif sources must read the live packet at source.node",
    ))
    iszero(source.kind) && throw(ArgumentError("empty motif source has no packet"))
    fill!(destination, 0.0f0)
    destination[1] = Float32(source.kind) / 16.0f0
    destination[2] = Float32(source.plane) / 2.0f0
    destination[3] = Float32(source.row) / 25.0f0
    destination[4] = Float32(source.column) / 10.0f0
    destination[5] = Float32(source.placement_slot) / 4.0f0
    destination[6] = Float32(source.context_slot) / 16.0f0
    destination[7] = Float32(source.branch_slot) / 8.0f0
    bits = reinterpret(UInt32, source.value)
    @inbounds for byte in 0:3
        destination[8 + byte] = Float32((bits >> (8 * byte)) & 0xff) / 255.0f0
    end
    destination[12] = 1.0f0
    return destination
end

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

@inline function _propagate_static_marked!(
    closure::AffectedClosure,
    topology::OrderedTopology,
)
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
    return nothing
end

@inline function _collect_marked!(closure::AffectedClosure)
    count = 0
    @inbounds for node in 1:NODE_COUNT
        iszero(closure.marked[node]) && continue
        count += 1
        closure.nodes[count] = UInt16(node)
    end
    closure.count = count
    return closure
end

@inline function _mark_incidence_receivers!(
    closure::AffectedClosure,
    incidence::CandidateMotifIncidence,
)
    @inbounds for motif in 1:MOTIF_COUNT
        count = Int(incidence.counts[motif])
        for rank in 1:count
            source = incidence.sources[_motif_source_index(motif, rank)]
            motif_source_is_spine(source) || continue
            if !iszero(closure.marked[Int(source.node)])
                closure.marked[Int(motif_node(
                    div(motif - 1, MOTIF_SLOTS_PER_FAMILY) + 1,
                    mod(motif - 1, MOTIF_SLOTS_PER_FAMILY) + 1,
                ))] = UInt8(1)
                break
            end
        end
    end
    return nothing
end

"""
Propagate changed static spine nodes through candidate-derived motif incidence.

Static spine closure is computed first.  Any motif that reads one of those
live packets is then marked, after which immutable motif/evidence/output edges
complete the closure.  External candidate tokens are handled by comparing two
incidence objects with `fill_changed_motif_closure!`.
"""
function fill_incidence_affected_closure!(
    closure::AffectedClosure,
    topology::OrderedTopology,
    incidence::CandidateMotifIncidence,
    seeds,
    seed_count::Integer,
)
    physical_count = Int(seed_count)
    0 <= physical_count <= length(seeds) || throw(
        BoundsError(seeds, physical_count),
    )
    fill!(closure.marked, UInt8(0))
    @inbounds for index in 1:physical_count
        closure.marked[_check_node(seeds[index])] = UInt8(1)
    end
    _propagate_static_marked!(closure, topology)
    _mark_incidence_receivers!(closure, incidence)
    _propagate_static_marked!(closure, topology)
    return _collect_marked!(closure)
end

@inline function fill_incidence_affected_closure!(
    closure::AffectedClosure,
    topology::OrderedTopology,
    incidence::CandidateMotifIncidence,
    seeds,
)
    return fill_incidence_affected_closure!(
        closure,
        topology,
        incidence,
        seeds,
        length(seeds),
    )
end

@inline function _motif_incidence_changed(
    before::CandidateMotifIncidence,
    after::CandidateMotifIncidence,
    motif::Int,
)
    before_count = Int(@inbounds before.counts[motif])
    after_count = Int(@inbounds after.counts[motif])
    before_count == after_count || return true
    @inbounds for rank in 1:before_count
        before.sources[_motif_source_index(motif, rank)] ==
            after.sources[_motif_source_index(motif, rank)] || return true
    end
    return false
end

"""
Compute the exact static downstream closure of changed candidate incidence.

The optional static seeds cover live before/after packet changes.  Descriptor
differences cover raw placement, clear/remap, queue roles, exact REN words,
B2B and T-spin.  Thus neither candidate tokens nor live packets are disguised
as immutable topology edges.
"""
function fill_changed_motif_closure!(
    closure::AffectedClosure,
    topology::OrderedTopology,
    before::CandidateMotifIncidence,
    after::CandidateMotifIncidence,
    seeds,
    seed_count::Integer,
)
    physical_count = Int(seed_count)
    0 <= physical_count <= length(seeds) || throw(
        BoundsError(seeds, physical_count),
    )
    fill!(closure.marked, UInt8(0))
    @inbounds for index in 1:physical_count
        closure.marked[_check_node(seeds[index])] = UInt8(1)
    end
    _propagate_static_marked!(closure, topology)
    _mark_incidence_receivers!(closure, after)
    @inbounds for motif in 1:MOTIF_COUNT
        _motif_incidence_changed(before, after, motif) || continue
        family = div(motif - 1, MOTIF_SLOTS_PER_FAMILY) + 1
        slot = mod(motif - 1, MOTIF_SLOTS_PER_FAMILY) + 1
        closure.marked[Int(motif_node(family, slot))] = UInt8(1)
    end
    _propagate_static_marked!(closure, topology)
    return _collect_marked!(closure)
end

@inline function fill_changed_motif_closure!(
    closure::AffectedClosure,
    topology::OrderedTopology,
    before::CandidateMotifIncidence,
    after::CandidateMotifIncidence,
)
    return fill_changed_motif_closure!(
        closure,
        topology,
        before,
        after,
        (),
        0,
    )
end

end # module OrderedMultiscaleTopology
