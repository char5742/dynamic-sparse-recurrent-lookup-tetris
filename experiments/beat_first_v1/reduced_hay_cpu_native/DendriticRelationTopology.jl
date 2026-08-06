module DendriticRelationTopology

"""
Fixed, CPU-oriented leaf-to-relation topology for the candidate-delta
high-dimensional dendritic model.

The topology is deliberately a shallow bipartite graph rather than a
reduction tree.  Every board leaf in both semantic planes has four independent
paths:

* one physical-row relation;
* one physical-column relation;
* one `6 x 5` tile-local relation;
* one small-world/global-context stripe.

There is no global reducer, root cut, learned router, or relation-to-relation
edge.  Absolute source position and semantic plane are retained in every
contact specification, so downstream kernels never have to infer identity
from an averaged payload.
"""

export AFTER_PLANE,
       AffectedRelationClosure,
       BEFORE_PLANE,
       BASAL_COMPARTMENT_COUNT,
       COLUMN_RELATION,
       COLUMN_RELATION_COUNT,
       CONTACT_COUNT,
       CROSS_ACTION_RELATION,
       CROSS_ACTION_RELATION_COUNT,
       LeafContactSpec,
       LEAF_COUNT,
       PLANE_COUNT,
       RELATION_COUNT,
       RELATION_LAYOUTS,
       ROW_COUNT,
       COLUMN_COUNT,
       ROW_LOCAL_RELATION,
       ROW_RELATION_COUNT,
       RelationLayout,
       RelationTopology,
       SMALL_WORLD_LAYOUT,
       TILE_LOCAL_LAYOUT,
       SOURCE_COUNT,
       SOURCE_FANOUT,
       affected_count,
       affected_mask,
       affected_relation,
       canonical_topology,
       column_relation,
       contact_basal_compartment,
       contact_id,
       contact_local_role,
       contact_plane,
       contact_position,
       contact_relation,
       contact_source_slot,
       cross_action_relation,
       fill_affected_relation_closure!,
       incoming_contact,
       incoming_contact_count,
       relation_class,
       relation_layout,
       relation_slot,
       relation_is_affected,
       row_relation,
       source_contact,
       source_contact_count,
       source_id,
       source_plane,
       source_position

const ROW_COUNT = 24
const COLUMN_COUNT = 10
const LEAF_COUNT = ROW_COUNT * COLUMN_COUNT
const PLANE_COUNT = 2
const SOURCE_COUNT = LEAF_COUNT * PLANE_COUNT

const BEFORE_PLANE = UInt8(1)
const AFTER_PLANE = UInt8(2)

const ROW_RELATION_COUNT = ROW_COUNT
const COLUMN_RELATION_COUNT = COLUMN_COUNT
const CROSS_ACTION_RELATION_COUNT = 14
const RELATION_COUNT =
    ROW_RELATION_COUNT + COLUMN_RELATION_COUNT + CROSS_ACTION_RELATION_COUNT

const ROW_LOCAL_RELATION = UInt8(1)
const COLUMN_RELATION = UInt8(2)
const CROSS_ACTION_RELATION = UInt8(3)

# Layout tags intentionally occupy a different namespace from the three
# relation-class tags.  Mixing both domains previously made a tile contact
# take the row branch-map path because both happened to use integer one.
const TILE_LOCAL_LAYOUT = UInt8(4)
const SMALL_WORLD_LAYOUT = UInt8(5)

const SOURCE_FANOUT = 4
const CONTACT_COUNT = SOURCE_COUNT * SOURCE_FANOUT
const BASAL_COMPARTMENT_COUNT = 8

const _ROW_RELATION_FIRST = 1
const _COLUMN_RELATION_FIRST = _ROW_RELATION_FIRST + ROW_RELATION_COUNT
const _CROSS_ACTION_RELATION_FIRST =
    _COLUMN_RELATION_FIRST + COLUMN_RELATION_COUNT

"""
One immutable selector for a cross/action relation cell.

A tile layout accepts one explicit `6 x 5` rectangle.  A stripe layout accepts
a board position when

```text
(row_stride * (row - 1) + column_stride * (column - 1)) mod modulus == phase
```

The first eight layouts are the Cartesian product of four six-row bands and
two five-column halves.  The last six form balanced small-world stripes.
Neither family contains a dense all-board root: every leaf belongs to exactly
one layout in each family.
"""
struct RelationLayout
    kind::UInt8
    row_first::UInt8
    row_last::UInt8
    column_first::UInt8
    column_last::UInt8
    modulus::UInt8
    row_stride::UInt8
    column_stride::UInt8
    phase::UInt8
end

const RELATION_LAYOUTS = ntuple(CROSS_ACTION_RELATION_COUNT) do slot
    if slot <= 8
        row_band = div(slot - 1, 2)
        column_half = mod(slot - 1, 2)
        RelationLayout(
            TILE_LOCAL_LAYOUT,
            UInt8(row_band * 6 + 1),
            UInt8((row_band + 1) * 6),
            UInt8(column_half * 5 + 1),
            UInt8((column_half + 1) * 5),
            UInt8(0),
            UInt8(0),
            UInt8(0),
            UInt8(0),
        )
    else
        RelationLayout(
            SMALL_WORLD_LAYOUT,
            UInt8(1),
            UInt8(ROW_COUNT),
            UInt8(1),
            UInt8(COLUMN_COUNT),
            UInt8(6),
            UInt8(1),
            UInt8(5),
            UInt8(slot - 9),
        )
    end
end

"""
Position-preserving source-to-relation contact.

`contact` is the unique 1:1920 trainable-contact coordinate and `source_slot`
is its fixed row/column/tile/stripe lane within one source.  `source` is unique
across both planes.  `position` is the absolute 1:240 column-major board
position and is never replaced by a pooled ordinal.  `local_role` is an
additional relation-local coordinate.  Before and after contacts keep
independent source IDs and typed contact parameters.  Both planes may use all
eight compartments; they are never pooled before the cell.
"""
struct LeafContactSpec
    contact::UInt16
    source::UInt16
    position::UInt16
    relation::UInt8
    local_role::UInt16
    plane::UInt8
    basal_compartment::UInt8
    source_slot::UInt8
end

"""
Immutable owner of fixed-capacity source-major and relation-major arrays.

`Memory` is used instead of a 1,920-element tuple so the Julia compiler does
not specialize one method per tuple coordinate.  The canonical instance is
constructed once and treated as read-only; `Memory` cannot be resized.
"""
struct RelationTopology
    source_offsets::Memory{UInt16}
    contacts::Memory{LeafContactSpec}
    incoming_offsets::Memory{UInt16}
    incoming_contact_ids::Memory{UInt16}
    relation_classes::Memory{UInt8}
    relation_slots::Memory{UInt8}
end

@inline function _check_plane(plane::Integer)
    1 <= plane <= PLANE_COUNT || throw(BoundsError(1:PLANE_COUNT, plane))
    return UInt8(plane)
end

@inline function _check_position(position::Integer)
    1 <= position <= LEAF_COUNT ||
        throw(BoundsError(1:LEAF_COUNT, position))
    return Int(position)
end

@inline function _check_source(source::Integer)
    1 <= source <= SOURCE_COUNT ||
        throw(BoundsError(1:SOURCE_COUNT, source))
    return Int(source)
end

@inline function _check_relation(relation::Integer)
    1 <= relation <= RELATION_COUNT ||
        throw(BoundsError(1:RELATION_COUNT, relation))
    return Int(relation)
end

@inline function _coordinates(position::Int)
    column = div(position - 1, ROW_COUNT) + 1
    row = position - (column - 1) * ROW_COUNT
    return row, column
end

@inline function source_id(plane::Integer, position::Integer)
    physical_plane = Int(_check_plane(plane))
    physical_position = _check_position(position)
    return UInt16((physical_plane - 1) * LEAF_COUNT + physical_position)
end

@inline function source_plane(source::Integer)
    physical_source = _check_source(source)
    return UInt8(div(physical_source - 1, LEAF_COUNT) + 1)
end

@inline function source_position(source::Integer)
    physical_source = _check_source(source)
    return UInt16(mod(physical_source - 1, LEAF_COUNT) + 1)
end

@inline row_relation(row::Integer) = begin
    1 <= row <= ROW_COUNT || throw(BoundsError(1:ROW_COUNT, row))
    UInt8(_ROW_RELATION_FIRST + Int(row) - 1)
end

@inline column_relation(column::Integer) = begin
    1 <= column <= COLUMN_COUNT ||
        throw(BoundsError(1:COLUMN_COUNT, column))
    UInt8(_COLUMN_RELATION_FIRST + Int(column) - 1)
end

@inline cross_action_relation(slot::Integer) = begin
    1 <= slot <= CROSS_ACTION_RELATION_COUNT ||
        throw(BoundsError(1:CROSS_ACTION_RELATION_COUNT, slot))
    UInt8(_CROSS_ACTION_RELATION_FIRST + Int(slot) - 1)
end

@inline function relation_layout(slot::Integer)
    1 <= slot <= CROSS_ACTION_RELATION_COUNT ||
        throw(BoundsError(1:CROSS_ACTION_RELATION_COUNT, slot))
    return @inbounds RELATION_LAYOUTS[Int(slot)]
end

@inline function _layout_accepts(
    layout::RelationLayout,
    row::Int,
    column::Int,
)
    if layout.kind == TILE_LOCAL_LAYOUT
        return Int(layout.row_first) <= row <= Int(layout.row_last) &&
               Int(layout.column_first) <= column <=
                   Int(layout.column_last)
    end
    value = Int(layout.row_stride) * (row - 1) +
            Int(layout.column_stride) * (column - 1)
    return mod(value, Int(layout.modulus)) == Int(layout.phase)
end

@inline function _tile_local_slot(row::Int, column::Int)
    return 2 * div(row - 1, 6) + div(column - 1, 5) + 1
end

@inline function _small_world_slot(row::Int, column::Int)
    return mod((row - 1) + 5 * (column - 1), 6) + 9
end

@inline function _local_role(kind::UInt8, row::Int, column::Int)
    kind == ROW_LOCAL_RELATION && return UInt16(column)
    kind == COLUMN_RELATION && return UInt16(row)
    kind == TILE_LOCAL_LAYOUT && return UInt16(
        mod(row - 1, 6) + 6 * mod(column - 1, 5) + 1,
    )
    # Every stripe contains exactly one row residue per column and six-row
    # band.  This gives a collision-free 1:40 relation-local coordinate.
    return UInt16(div(row - 1, 6) * COLUMN_COUNT + column)
end

@inline function _basal_compartment(
    kind::UInt8,
    row::Int,
    column::Int,
)
    kind == ROW_LOCAL_RELATION &&
        return UInt8(mod(3 * (column - 1), 8) + 1)
    kind == COLUMN_RELATION && return UInt8(mod(row - 1, 8) + 1)
    kind == TILE_LOCAL_LAYOUT && return UInt8(
        mod(mod(row - 1, 6) + 6 * mod(column - 1, 5), 8) + 1,
    )
    return UInt8(2 * div(row - 1, 6) + mod(column - 1, 2) + 1)
end

function _build_topology()
    contacts = Vector{LeafContactSpec}()
    sizehint!(contacts, CONTACT_COUNT)
    source_offsets = Vector{UInt16}(undef, SOURCE_COUNT + 1)
    source_offsets[1] = UInt16(1)

    incoming = [UInt16[] for _ in 1:RELATION_COUNT]

    @inbounds for plane_value in 1:PLANE_COUNT
        plane = UInt8(plane_value)
        for position in 1:LEAF_COUNT
            source = source_id(plane, position)
            row, column = _coordinates(position)
            tile_local_slot = _tile_local_slot(row, column)
            small_world_slot = _small_world_slot(row, column)
            relation_ids = (
                row_relation(row),
                column_relation(column),
                cross_action_relation(tile_local_slot),
                cross_action_relation(small_world_slot),
            )
            kinds = (
                ROW_LOCAL_RELATION,
                COLUMN_RELATION,
                TILE_LOCAL_LAYOUT,
                SMALL_WORLD_LAYOUT,
            )

            for fanout_index in 1:SOURCE_FANOUT
                local_role = _local_role(kinds[fanout_index], row, column)
                contact = LeafContactSpec(
                    UInt16(length(contacts) + 1),
                    source,
                    UInt16(position),
                    relation_ids[fanout_index],
                    local_role,
                    plane,
                    _basal_compartment(kinds[fanout_index], row, column),
                    UInt8(fanout_index),
                )
                push!(contacts, contact)
                push!(incoming[Int(contact.relation)], UInt16(length(contacts)))
            end
            source_offsets[Int(source) + 1] = UInt16(length(contacts) + 1)
        end
    end
    length(contacts) == CONTACT_COUNT || error("relation contact count drift")

    incoming_offsets = Vector{UInt16}(undef, RELATION_COUNT + 1)
    incoming_contact_ids = Vector{UInt16}()
    sizehint!(incoming_contact_ids, CONTACT_COUNT)
    incoming_offsets[1] = UInt16(1)
    @inbounds for relation in 1:RELATION_COUNT
        append!(incoming_contact_ids, incoming[relation])
        incoming_offsets[relation + 1] =
            UInt16(length(incoming_contact_ids) + 1)
    end
    length(incoming_contact_ids) == CONTACT_COUNT ||
        error("incoming contact count drift")

    classes = map(1:RELATION_COUNT) do relation
        relation <= ROW_RELATION_COUNT ? ROW_LOCAL_RELATION :
        relation <= ROW_RELATION_COUNT + COLUMN_RELATION_COUNT ?
            COLUMN_RELATION : CROSS_ACTION_RELATION
    end
    slots = map(1:RELATION_COUNT) do relation
        relation <= ROW_RELATION_COUNT ? UInt8(relation) :
        relation <= ROW_RELATION_COUNT + COLUMN_RELATION_COUNT ?
            UInt8(relation - ROW_RELATION_COUNT) :
            UInt8(relation - ROW_RELATION_COUNT - COLUMN_RELATION_COUNT)
    end

    topology = RelationTopology(
        Memory{UInt16}(source_offsets),
        Memory{LeafContactSpec}(contacts),
        Memory{UInt16}(incoming_offsets),
        Memory{UInt16}(incoming_contact_ids),
        Memory{UInt8}(classes),
        Memory{UInt8}(slots),
    )
    _validate_topology(topology)
    return topology
end

function _validate_topology(topology::RelationTopology)
    seen = falses(SOURCE_COUNT, RELATION_COUNT)
    @inbounds for source in 1:SOURCE_COUNT
        first_contact = Int(topology.source_offsets[source])
        contact_limit = Int(topology.source_offsets[source + 1]) - 1
        contact_limit - first_contact + 1 == SOURCE_FANOUT ||
            error("source fanout drift")
        for contact_index in first_contact:contact_limit
            contact = topology.contacts[contact_index]
            Int(contact.contact) == contact_index ||
                error("contact identity drift")
            Int(contact.source) == source || error("source-major order drift")
            Int(contact.source_slot) == contact_index - first_contact + 1 ||
                error("source slot drift")
            contact.position == source_position(source) ||
                error("contact position identity drift")
            contact.plane == source_plane(source) ||
                error("contact plane identity drift")
            relation = Int(contact.relation)
            seen[source, relation] && error("duplicate source/relation contact")
            seen[source, relation] = true
        end
    end
    return nothing
end

const _CANONICAL_TOPOLOGY = _build_topology()

"""Return the process-wide immutable topology."""
@inline canonical_topology() = _CANONICAL_TOPOLOGY

@inline function source_contact_count(
    topology::RelationTopology,
    source::Integer,
)
    physical_source = _check_source(source)
    return Int(topology.source_offsets[physical_source + 1]) -
           Int(topology.source_offsets[physical_source])
end

@inline function source_contact(
    topology::RelationTopology,
    source::Integer,
    fanout_index::Integer,
)
    physical_source = _check_source(source)
    count = source_contact_count(topology, physical_source)
    1 <= fanout_index <= count || throw(BoundsError(1:count, fanout_index))
    contact_index = Int(topology.source_offsets[physical_source]) +
                    Int(fanout_index) - 1
    return @inbounds topology.contacts[contact_index]
end

@inline function incoming_contact_count(
    topology::RelationTopology,
    relation::Integer,
)
    physical_relation = _check_relation(relation)
    return Int(topology.incoming_offsets[physical_relation + 1]) -
           Int(topology.incoming_offsets[physical_relation])
end

@inline function incoming_contact(
    topology::RelationTopology,
    relation::Integer,
    incoming_index::Integer,
)
    physical_relation = _check_relation(relation)
    count = incoming_contact_count(topology, physical_relation)
    1 <= incoming_index <= count ||
        throw(BoundsError(1:count, incoming_index))
    flat_index = Int(topology.incoming_offsets[physical_relation]) +
                 Int(incoming_index) - 1
    contact_id = @inbounds topology.incoming_contact_ids[flat_index]
    return @inbounds topology.contacts[Int(contact_id)]
end

@inline relation_class(topology::RelationTopology, relation::Integer) =
    @inbounds topology.relation_classes[_check_relation(relation)]

@inline relation_slot(topology::RelationTopology, relation::Integer) =
    @inbounds topology.relation_slots[_check_relation(relation)]

@inline contact_relation(contact::LeafContactSpec) = contact.relation
@inline contact_id(contact::LeafContactSpec) = contact.contact
@inline contact_position(contact::LeafContactSpec) = contact.position
@inline contact_plane(contact::LeafContactSpec) = contact.plane
@inline contact_local_role(contact::LeafContactSpec) = contact.local_role
@inline contact_basal_compartment(contact::LeafContactSpec) =
    contact.basal_compartment
@inline contact_source_slot(contact::LeafContactSpec) = contact.source_slot

"""
Caller-owned exact relation closure.

The 48 relation bits fit in one `UInt64`; `relations` is a fixed-capacity,
ascending list for branch-free hot iteration.  It contains only direct
leaf-to-relation dependencies because this topology has no hidden reducer
chain.
"""
mutable struct AffectedRelationClosure <: AbstractVector{UInt8}
    relations::Memory{UInt8}
    mask::UInt64
    count::Int
    function AffectedRelationClosure()
        relations = Memory{UInt8}(undef, RELATION_COUNT)
        fill!(relations, UInt8(0))
        return new(relations, UInt64(0), 0)
    end
end

@inline affected_count(closure::AffectedRelationClosure) = closure.count
@inline affected_mask(closure::AffectedRelationClosure) = closure.mask
Base.IndexStyle(::Type{AffectedRelationClosure}) = IndexLinear()
Base.size(closure::AffectedRelationClosure) = (closure.count,)
Base.length(closure::AffectedRelationClosure) = closure.count
@inline Base.getindex(closure::AffectedRelationClosure, index::Int) =
    affected_relation(closure, index)

@inline function affected_relation(
    closure::AffectedRelationClosure,
    index::Integer,
)
    1 <= index <= closure.count ||
        throw(BoundsError(1:closure.count, index))
    return @inbounds closure.relations[Int(index)]
end

@inline function relation_is_affected(
    closure::AffectedRelationClosure,
    relation::Integer,
)
    physical_relation = _check_relation(relation)
    bit = UInt64(1) << (physical_relation - 1)
    return !iszero(closure.mask & bit)
end

"""
Fill the exact relation cells touched by changed leaves without allocation.

Duplicate positions are accepted.  `plane` is explicit so a candidate kernel
cannot silently confuse cached before-plane contacts with candidate-local
after-plane contacts.  The relation set is the same geometric closure for
both planes, while each underlying contact remains plane-distinct.
"""
function fill_affected_relation_closure!(
    closure::AffectedRelationClosure,
    topology::RelationTopology,
    plane::Integer,
    changed_positions,
    changed_count::Integer,
)
    physical_plane = _check_plane(plane)
    count = Int(changed_count)
    0 <= count <= length(changed_positions) ||
        throw(BoundsError(changed_positions, count))

    mask = UInt64(0)
    @inbounds for changed_index in 1:count
        position = _check_position(changed_positions[changed_index])
        source = Int(source_id(physical_plane, position))
        first_contact = Int(topology.source_offsets[source])
        contact_limit = Int(topology.source_offsets[source + 1]) - 1
        for contact_index in first_contact:contact_limit
            relation = Int(topology.contacts[contact_index].relation)
            mask |= UInt64(1) << (relation - 1)
        end
    end

    output_count = 0
    @inbounds for relation in 1:RELATION_COUNT
        iszero(mask & (UInt64(1) << (relation - 1))) && continue
        output_count += 1
        closure.relations[output_count] = UInt8(relation)
    end
    closure.mask = mask
    closure.count = output_count
    return closure
end

@inline function fill_affected_relation_closure!(
    closure::AffectedRelationClosure,
    topology::RelationTopology,
    plane::Integer,
    changed_positions,
)
    return fill_affected_relation_closure!(
        closure,
        topology,
        plane,
        changed_positions,
        length(changed_positions),
    )
end

@inline function fill_affected_relation_closure!(
    closure::AffectedRelationClosure,
    plane::Integer,
    changed_positions,
)
    return fill_affected_relation_closure!(
        closure,
        _CANONICAL_TOPOLOGY,
        plane,
        changed_positions,
        length(changed_positions),
    )
end

end # module DendriticRelationTopology
