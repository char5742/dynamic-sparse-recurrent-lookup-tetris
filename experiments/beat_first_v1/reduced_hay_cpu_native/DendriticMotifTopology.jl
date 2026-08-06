module DendriticMotifTopology

"""
Fixed relation-to-motif topology for the CPU-native high-dimensional
Reduced-Hay model.

The 48 relation sources and 48 motif cells share the same semantic atlas:

* `1:24`  -- vertical/row motifs (`V`);
* `25:34` -- well/column motifs (`W`);
* `35:42` -- local geometry motifs (`G`);
* `43:48` -- action/global-stripe motifs (`A`).

Every relation source reaches exactly four *different* motif cells.  The
mapping is fixed, geometry-derived, and contains no learned router or dense
all-to-all reducer.  Candidate-delta execution therefore needs to recompute
only the union of four destinations for each affected relation.
"""

export ACTION_MOTIF,
       ACTION_MOTIF_COUNT,
       ACTION_MOTIF_FIRST,
       AffectedMotifClosure,
       CONTACT_COUNT,
       GEOMETRY_MOTIF,
       GEOMETRY_MOTIF_COUNT,
       GEOMETRY_MOTIF_FIRST,
       MOTIF_COUNT,
       MotifTopology,
       RELATION_SOURCE_COUNT,
       SOURCE_FANOUT,
       VERTICAL_MOTIF,
       VERTICAL_MOTIF_COUNT,
       VERTICAL_MOTIF_FIRST,
       WELL_MOTIF,
       WELL_MOTIF_COUNT,
       WELL_MOTIF_FIRST,
       action_motif,
       affected_motif,
       canonical_topology,
       fill_affected_motif_closure!,
       geometry_motif,
       incoming_source,
       incoming_source_count,
       motif_class,
       motif_count,
       motif_destination,
       motif_is_affected,
       motif_mask,
       motif_slot,
       validate_topology,
       vertical_motif,
       well_motif

const RELATION_SOURCE_COUNT = 48
const MOTIF_COUNT = 48
const SOURCE_FANOUT = 4
const CONTACT_COUNT = RELATION_SOURCE_COUNT * SOURCE_FANOUT

const VERTICAL_MOTIF_FIRST = 1
const VERTICAL_MOTIF_COUNT = 24
const WELL_MOTIF_FIRST = VERTICAL_MOTIF_FIRST + VERTICAL_MOTIF_COUNT
const WELL_MOTIF_COUNT = 10
const GEOMETRY_MOTIF_FIRST = WELL_MOTIF_FIRST + WELL_MOTIF_COUNT
const GEOMETRY_MOTIF_COUNT = 8
const ACTION_MOTIF_FIRST = GEOMETRY_MOTIF_FIRST + GEOMETRY_MOTIF_COUNT
const ACTION_MOTIF_COUNT = 6

const VERTICAL_MOTIF = UInt8(1)
const WELL_MOTIF = UInt8(2)
const GEOMETRY_MOTIF = UInt8(3)
const ACTION_MOTIF = UInt8(4)

@assert ACTION_MOTIF_FIRST + ACTION_MOTIF_COUNT - 1 == MOTIF_COUNT

"""
Immutable source-major and motif-major adjacency.

All arrays have fixed capacity and are stored in `Memory`, so callers cannot
resize the process-wide canonical topology.  Source-major destinations are
ordered by the four semantic mapping rules.  Incoming sources are ascending
within each motif.
"""
struct MotifTopology
    destinations::Memory{UInt8}
    incoming_offsets::Memory{UInt16}
    incoming_sources::Memory{UInt8}
end

@inline function _check_source(source::Integer)
    1 <= source <= RELATION_SOURCE_COUNT ||
        throw(BoundsError(1:RELATION_SOURCE_COUNT, source))
    return Int(source)
end

@inline function _check_motif(motif::Integer)
    1 <= motif <= MOTIF_COUNT || throw(BoundsError(1:MOTIF_COUNT, motif))
    return Int(motif)
end

@inline function _check_rank(rank::Integer)
    1 <= rank <= SOURCE_FANOUT || throw(BoundsError(1:SOURCE_FANOUT, rank))
    return Int(rank)
end

@inline function _check_slot(slot::Integer, count::Int)
    1 <= slot <= count || throw(BoundsError(1:count, slot))
    return Int(slot)
end

@inline _paired_even(value::Int) = isodd(value) ? value + 1 : value - 1

@inline function vertical_motif(slot::Integer)
    physical_slot = _check_slot(slot, VERTICAL_MOTIF_COUNT)
    return UInt8(VERTICAL_MOTIF_FIRST + physical_slot - 1)
end

@inline function well_motif(slot::Integer)
    physical_slot = _check_slot(slot, WELL_MOTIF_COUNT)
    return UInt8(WELL_MOTIF_FIRST + physical_slot - 1)
end

@inline function geometry_motif(slot::Integer)
    physical_slot = _check_slot(slot, GEOMETRY_MOTIF_COUNT)
    return UInt8(GEOMETRY_MOTIF_FIRST + physical_slot - 1)
end

@inline function action_motif(slot::Integer)
    physical_slot = _check_slot(slot, ACTION_MOTIF_COUNT)
    return UInt8(ACTION_MOTIF_FIRST + physical_slot - 1)
end

@inline function _tile_motif(band::Int, half::Int)
    1 <= band <= 4 || throw(BoundsError(1:4, band))
    1 <= half <= 2 || throw(BoundsError(1:2, half))
    return geometry_motif(2 * (band - 1) + half)
end

@inline function motif_class(motif::Integer)
    physical_motif = _check_motif(motif)
    return physical_motif < WELL_MOTIF_FIRST ? VERTICAL_MOTIF :
           physical_motif < GEOMETRY_MOTIF_FIRST ? WELL_MOTIF :
           physical_motif < ACTION_MOTIF_FIRST ? GEOMETRY_MOTIF :
           ACTION_MOTIF
end

@inline function motif_slot(motif::Integer)
    physical_motif = _check_motif(motif)
    return physical_motif < WELL_MOTIF_FIRST ?
           UInt8(physical_motif - VERTICAL_MOTIF_FIRST + 1) :
           physical_motif < GEOMETRY_MOTIF_FIRST ?
           UInt8(physical_motif - WELL_MOTIF_FIRST + 1) :
           physical_motif < ACTION_MOTIF_FIRST ?
           UInt8(physical_motif - GEOMETRY_MOTIF_FIRST + 1) :
           UInt8(physical_motif - ACTION_MOTIF_FIRST + 1)
end

"""Return one of the four fixed motif destinations for a relation source."""
@inline function motif_destination(
    topology::MotifTopology,
    source::Integer,
    rank::Integer,
)
    physical_source = _check_source(source)
    physical_rank = _check_rank(rank)
    index = (physical_source - 1) * SOURCE_FANOUT + physical_rank
    return @inbounds topology.destinations[index]
end

function _source_destinations(source::Int)
    if source <= VERTICAL_MOTIF_COUNT
        row = source
        band = div(row - 1, 6) + 1
        return (
            vertical_motif(row),
            vertical_motif(_paired_even(row)),
            _tile_motif(band, 1),
            _tile_motif(band, 2),
        )
    elseif source < GEOMETRY_MOTIF_FIRST
        column = source - WELL_MOTIF_FIRST + 1
        half = div(column - 1, 5) + 1
        return (
            well_motif(column),
            well_motif(_paired_even(column)),
            _tile_motif(mod(column - 1, 4) + 1, half),
            _tile_motif(mod(column + 1, 4) + 1, half),
        )
    elseif source < ACTION_MOTIF_FIRST
        tile = source - GEOMETRY_MOTIF_FIRST + 1
        band = div(tile - 1, 2) + 1
        half = mod(tile - 1, 2) + 1
        return (
            geometry_motif(tile),
            _tile_motif(band, 3 - half),
            vertical_motif(6 * band - 3),
            well_motif(5 * half - 2),
        )
    end

    stripe = source - ACTION_MOTIF_FIRST + 1
    return (
        action_motif(stripe),
        action_motif(_paired_even(stripe)),
        _tile_motif(mod(stripe - 1, 4) + 1, mod(stripe - 1, 2) + 1),
        _tile_motif(mod(stripe + 1, 4) + 1, mod(stripe, 2) + 1),
    )
end

function _build_topology()
    destinations = Vector{UInt8}(undef, CONTACT_COUNT)
    incoming = [UInt8[] for _ in 1:MOTIF_COUNT]

    @inbounds for source in 1:RELATION_SOURCE_COUNT
        source_destinations = _source_destinations(source)
        for rank in 1:SOURCE_FANOUT
            destination = source_destinations[rank]
            contact = (source - 1) * SOURCE_FANOUT + rank
            destinations[contact] = destination
            push!(incoming[Int(destination)], UInt8(source))
        end
    end

    incoming_offsets = Vector{UInt16}(undef, MOTIF_COUNT + 1)
    incoming_sources = Vector{UInt8}()
    sizehint!(incoming_sources, CONTACT_COUNT)
    incoming_offsets[1] = UInt16(1)
    @inbounds for motif in 1:MOTIF_COUNT
        sort!(incoming[motif])
        append!(incoming_sources, incoming[motif])
        incoming_offsets[motif + 1] = UInt16(length(incoming_sources) + 1)
    end

    topology = MotifTopology(
        Memory{UInt8}(destinations),
        Memory{UInt16}(incoming_offsets),
        Memory{UInt8}(incoming_sources),
    )
    validate_topology(topology)
    return topology
end

"""Fail closed if a motif topology violates the fixed semantic contract."""
function validate_topology(topology::MotifTopology)
    length(topology.destinations) == CONTACT_COUNT ||
        error("motif destination count drift")
    length(topology.incoming_offsets) == MOTIF_COUNT + 1 ||
        error("motif incoming-offset count drift")
    length(topology.incoming_sources) == CONTACT_COUNT ||
        error("motif incoming-source count drift")
    topology.incoming_offsets[1] == UInt16(1) ||
        error("motif incoming offsets must be one-based")
    topology.incoming_offsets[end] == UInt16(CONTACT_COUNT + 1) ||
        error("motif incoming terminal offset drift")

    seen = falses(RELATION_SOURCE_COUNT, MOTIF_COUNT)
    @inbounds for source in 1:RELATION_SOURCE_COUNT
        expected = _source_destinations(source)
        for rank in 1:SOURCE_FANOUT
            destination = Int(motif_destination(topology, source, rank))
            1 <= destination <= MOTIF_COUNT ||
                error("motif destination out of range")
            destination == Int(expected[rank]) ||
                error("motif mapping drift")
            seen[source, destination] &&
                error("duplicate destination within relation source")
            seen[source, destination] = true
        end
    end

    observed_contacts = 0
    @inbounds for motif in 1:MOTIF_COUNT
        first_index = Int(topology.incoming_offsets[motif])
        limit = Int(topology.incoming_offsets[motif + 1]) - 1
        previous_source = 0
        for index in first_index:limit
            source = Int(topology.incoming_sources[index])
            source > previous_source ||
                error("motif incoming sources must be strictly ascending")
            seen[source, motif] || error("motif inverse adjacency drift")
            previous_source = source
            observed_contacts += 1
        end
    end
    observed_contacts == CONTACT_COUNT || error("motif contact count drift")
    return nothing
end

const _CANONICAL_TOPOLOGY = _build_topology()

"""Return the process-wide immutable fixed topology."""
@inline canonical_topology() = _CANONICAL_TOPOLOGY

@inline motif_destination(source::Integer, rank::Integer) =
    motif_destination(_CANONICAL_TOPOLOGY, source, rank)

@inline function incoming_source_count(
    topology::MotifTopology,
    motif::Integer,
)
    physical_motif = _check_motif(motif)
    return Int(topology.incoming_offsets[physical_motif + 1]) -
           Int(topology.incoming_offsets[physical_motif])
end

@inline incoming_source_count(motif::Integer) =
    incoming_source_count(_CANONICAL_TOPOLOGY, motif)

@inline function incoming_source(
    topology::MotifTopology,
    motif::Integer,
    incoming_index::Integer,
)
    physical_motif = _check_motif(motif)
    count = incoming_source_count(topology, physical_motif)
    1 <= incoming_index <= count ||
        throw(BoundsError(1:count, incoming_index))
    flat_index = Int(topology.incoming_offsets[physical_motif]) +
                 Int(incoming_index) - 1
    return @inbounds topology.incoming_sources[flat_index]
end

@inline incoming_source(motif::Integer, incoming_index::Integer) =
    incoming_source(_CANONICAL_TOPOLOGY, motif, incoming_index)

"""
Caller-owned, allocation-free affected-motif closure.

The complete motif set fits in a `UInt64`.  `motifs` is a fixed-capacity
ascending list for the hot forward/reverse loops; duplicate affected relation
sources are harmless.
"""
mutable struct AffectedMotifClosure <: AbstractVector{UInt8}
    motifs::Memory{UInt8}
    mask::UInt64
    count::Int
    function AffectedMotifClosure()
        motifs = Memory{UInt8}(undef, MOTIF_COUNT)
        fill!(motifs, UInt8(0))
        return new(motifs, UInt64(0), 0)
    end
end

Base.IndexStyle(::Type{AffectedMotifClosure}) = IndexLinear()
Base.size(closure::AffectedMotifClosure) = (closure.count,)
Base.length(closure::AffectedMotifClosure) = closure.count
@inline Base.getindex(closure::AffectedMotifClosure, index::Int) =
    affected_motif(closure, index)

@inline motif_count(closure::AffectedMotifClosure) = closure.count
@inline motif_mask(closure::AffectedMotifClosure) = closure.mask

@inline function affected_motif(
    closure::AffectedMotifClosure,
    index::Integer,
)
    1 <= index <= closure.count ||
        throw(BoundsError(1:closure.count, index))
    return @inbounds closure.motifs[Int(index)]
end

@inline function motif_is_affected(
    closure::AffectedMotifClosure,
    motif::Integer,
)
    physical_motif = _check_motif(motif)
    return !iszero(closure.mask & (UInt64(1) << (physical_motif - 1)))
end

@inline function _finish_closure!(
    closure::AffectedMotifClosure,
    mask::UInt64,
)
    output_count = 0
    @inbounds for motif in 1:MOTIF_COUNT
        iszero(mask & (UInt64(1) << (motif - 1))) && continue
        output_count += 1
        closure.motifs[output_count] = UInt8(motif)
    end
    closure.mask = mask
    closure.count = output_count
    return closure
end

"""
Fill the exact motif closure of an affected relation-source list.

Only `affected_count_value` entries are read.  The output list is always
ascending, independent of source order and duplication.
"""
function fill_affected_motif_closure!(
    closure::AffectedMotifClosure,
    topology::MotifTopology,
    affected_sources,
    affected_count_value::Integer,
)
    count = Int(affected_count_value)
    0 <= count <= length(affected_sources) ||
        throw(BoundsError(affected_sources, count))

    mask = UInt64(0)
    @inbounds for affected_index in 1:count
        source = _check_source(affected_sources[affected_index])
        first_contact = (source - 1) * SOURCE_FANOUT + 1
        for rank_offset in 0:(SOURCE_FANOUT - 1)
            motif = Int(topology.destinations[first_contact + rank_offset])
            mask |= UInt64(1) << (motif - 1)
        end
    end
    return _finish_closure!(closure, mask)
end

@inline function fill_affected_motif_closure!(
    closure::AffectedMotifClosure,
    topology::MotifTopology,
    affected_sources,
)
    return fill_affected_motif_closure!(
        closure,
        topology,
        affected_sources,
        length(affected_sources),
    )
end

@inline function fill_affected_motif_closure!(
    closure::AffectedMotifClosure,
    affected_sources,
    affected_count_value::Integer,
)
    return fill_affected_motif_closure!(
        closure,
        _CANONICAL_TOPOLOGY,
        affected_sources,
        affected_count_value,
    )
end

@inline function fill_affected_motif_closure!(
    closure::AffectedMotifClosure,
    affected_sources,
)
    return fill_affected_motif_closure!(
        closure,
        _CANONICAL_TOPOLOGY,
        affected_sources,
        length(affected_sources),
    )
end

"""Fill the motif closure directly from a 48-bit affected-relation mask."""
function fill_affected_motif_closure!(
    closure::AffectedMotifClosure,
    topology::MotifTopology,
    affected_source_mask::UInt64,
)
    valid_mask = (UInt64(1) << RELATION_SOURCE_COUNT) - UInt64(1)
    iszero(affected_source_mask & ~valid_mask) ||
        throw(ArgumentError("affected relation mask has bits above source 48"))

    mask = UInt64(0)
    @inbounds for source in 1:RELATION_SOURCE_COUNT
        iszero(
            affected_source_mask & (UInt64(1) << (source - 1)),
        ) && continue
        first_contact = (source - 1) * SOURCE_FANOUT + 1
        for rank_offset in 0:(SOURCE_FANOUT - 1)
            motif = Int(topology.destinations[first_contact + rank_offset])
            mask |= UInt64(1) << (motif - 1)
        end
    end
    return _finish_closure!(closure, mask)
end

@inline function fill_affected_motif_closure!(
    closure::AffectedMotifClosure,
    affected_source_mask::UInt64,
)
    return fill_affected_motif_closure!(
        closure,
        _CANONICAL_TOPOLOGY,
        affected_source_mask,
    )
end

end # module DendriticMotifTopology
