module CanonicalEventArena

"""
Fixed-memory, candidate-local scheduler for the canonical hard-event graph.

The arena deliberately knows nothing about the Reduced-Hay equations.  A
caller-owned adapter implements `advance_event_cell!` and reads/writes one COW
slot after every source contribution for the current wave has been accumulated.
This keeps the scheduler Jacobi-synchronous and makes same-wave event leakage
impossible by construction.
"""

export CANONICAL_MAX_WAVES,
       NO_EVENT,
       OVERFLOW_ERROR,
       OVERFLOW_FALLBACK,
       OVERFLOW_NONE,
       OVERFLOW_ACTIVE_CAPACITY,
       OVERFLOW_FRONTIER_CAPACITY,
       OVERFLOW_DYNAMIC_CAPACITY,
       OVERFLOW_SEED_MASK_CONFLICT,
       SourceMajorAdjacency,
       DynamicSourceMajorOverlay,
       EventArena,
       EventWaveReport,
       ArenaOverflowError,
       edge_count,
       begin_dynamic_overlay!,
       push_dynamic_edge!,
       seal_dynamic_overlay!,
       active_count,
       state_slot,
       candidate_state,
       begin_candidate!,
       touch_node!,
       seed_event!,
       seed_overlay_only_event!,
       overlay_only_seed_count,
       overlay_only_seed_source,
       overlay_only_seed_mask,
       deliver_event_edge!,
       advance_event_cell!,
       run_event_waves!

# Exact topology bound: the fixed binary information spine has depth five and
# a candidate-dependent column-root -> motif -> evidence path extends it to
# seven. A larger scheduler cap would be arbitrary; a smaller one truncates a
# reachable canonical path.
const CANONICAL_MAX_WAVES = 7
const NO_EVENT = UInt8(0)

const OVERFLOW_ERROR = UInt8(1)
const OVERFLOW_FALLBACK = UInt8(2)

const OVERFLOW_NONE = UInt8(0)
const OVERFLOW_ACTIVE_CAPACITY = UInt8(1)
const OVERFLOW_FRONTIER_CAPACITY = UInt8(2)
const OVERFLOW_DYNAMIC_CAPACITY = UInt8(3)
const OVERFLOW_SEED_MASK_CONFLICT = UInt8(4)

"""Immutable source-major hot adjacency copied out of construction buffers."""
struct SourceMajorAdjacency{T<:AbstractFloat}
    node_count::Int
    inbox_dim::Int
    offsets::Memory{UInt32}
    destination::Memory{UInt16}
    channel::Memory{UInt8}
    trigger_mask::Memory{UInt8}
    weight::Memory{T}
end

@inline edge_count(graph::SourceMajorAdjacency) = length(graph.destination)

"""
Fixed-capacity candidate-owned source-major overlay.

Construction is allocation-free after the cold constructor. Edges may be
pushed in any order. `seal_dynamic_overlay!` sorts the live prefix by
`(source, destination, channel, trigger_bit, raw_index)` and builds one-based
source offsets in place. A trigger is exactly one hard-event bit; interpreting
that bit (including plateau onset versus offset) belongs to the delivery
adapter, which can inspect the current source state.
"""
mutable struct DynamicSourceMajorOverlay
    node_count::Int
    inbox_dim::Int
    capacity::Int
    count::Int
    sealed::Bool
    offsets::Vector{UInt32}
    source::Vector{UInt16}
    destination::Vector{UInt16}
    channel::Vector{UInt8}
    trigger_bit::Vector{UInt8}
    raw_index::Vector{UInt32}
end

function DynamicSourceMajorOverlay(
    node_count::Integer,
    inbox_dim::Integer,
    capacity::Integer,
)
    nodes = _checked_positive(node_count, "node_count")
    inputs = _checked_positive(inbox_dim, "inbox_dim")
    edges = _checked_positive(capacity, "dynamic edge capacity")
    nodes <= typemax(UInt16) || throw(ArgumentError(
        "node_count exceeds UInt16 overlay identity capacity",
    ))
    return DynamicSourceMajorOverlay(
        nodes,
        inputs,
        edges,
        0,
        false,
        zeros(UInt32, nodes + 1),
        zeros(UInt16, edges),
        zeros(UInt16, edges),
        zeros(UInt8, edges),
        zeros(UInt8, edges),
        zeros(UInt32, edges),
    )
end

@inline edge_count(overlay::DynamicSourceMajorOverlay) = overlay.count

"""Discard the prior live prefix without clearing fixed storage."""
function begin_dynamic_overlay!(overlay::DynamicSourceMajorOverlay)
    overlay.count = 0
    overlay.sealed = false
    return overlay
end

@inline function _single_event_bit(value::UInt8)
    return !iszero(value) && iszero(value & (value - UInt8(1)))
end

"""Append one candidate contact to fixed staging storage."""
function push_dynamic_edge!(
    overlay::DynamicSourceMajorOverlay,
    source::Integer,
    destination::Integer,
    channel::Integer,
    trigger_bit::Integer,
    raw_index::Integer,
)
    overlay.sealed && throw(ArgumentError(
        "begin_dynamic_overlay! must be called before pushing a sealed overlay",
    ))
    physical_source = Int(source)
    physical_destination = Int(destination)
    physical_channel = Int(channel)
    trigger_value = Int(trigger_bit)
    physical_raw = Int(raw_index)
    1 <= physical_source <= overlay.node_count || throw(BoundsError(
        1:overlay.node_count, physical_source,
    ))
    1 <= physical_destination <= overlay.node_count || throw(BoundsError(
        1:overlay.node_count, physical_destination,
    ))
    1 <= physical_channel <= overlay.inbox_dim || throw(BoundsError(
        1:overlay.inbox_dim, physical_channel,
    ))
    1 <= trigger_value <= typemax(UInt8) || throw(ArgumentError(
        "dynamic trigger must fit UInt8",
    ))
    physical_trigger = UInt8(trigger_value)
    _single_event_bit(physical_trigger) || throw(ArgumentError(
        "dynamic trigger must contain exactly one hard-event bit",
    ))
    1 <= physical_raw <= typemax(UInt32) || throw(ArgumentError(
        "dynamic raw parameter index must be positive and fit UInt32",
    ))
    requested = overlay.count + 1
    requested <= overlay.capacity || throw(ArenaOverflowError(
        OVERFLOW_DYNAMIC_CAPACITY,
        overlay.capacity,
        requested,
    ))
    @inbounds begin
        overlay.source[requested] = UInt16(physical_source)
        overlay.destination[requested] = UInt16(physical_destination)
        overlay.channel[requested] = UInt8(physical_channel)
        overlay.trigger_bit[requested] = physical_trigger
        overlay.raw_index[requested] = UInt32(physical_raw)
    end
    overlay.count = requested
    return requested
end

@inline function _checked_positive(value::Integer, label::AbstractString)
    physical = Int(value)
    physical >= 1 || throw(ArgumentError("$label must be positive"))
    return physical
end

"""
    SourceMajorAdjacency(node_count, inbox_dim, offsets, destination,
                         channel, trigger_mask, weight)

Construct a fixed source-major graph. Offsets are one-based and every source's
edges must already be sorted by `(destination, channel, trigger_mask)`.  The
constructor copies every array, so later mutation of construction buffers
cannot change the execution graph.
"""
function SourceMajorAdjacency(
    node_count::Integer,
    inbox_dim::Integer,
    offsets,
    destination,
    channel,
    trigger_mask,
    weight::AbstractVector{T},
) where {T<:AbstractFloat}
    nodes = _checked_positive(node_count, "node_count")
    inputs = _checked_positive(inbox_dim, "inbox_dim")
    nodes <= typemax(UInt16) || throw(ArgumentError(
        "node_count exceeds UInt16 arena identity capacity",
    ))
    length(offsets) == nodes + 1 || throw(DimensionMismatch(
        "source offsets must contain node_count + 1 entries",
    ))
    edges = length(destination)
    length(channel) == edges || throw(DimensionMismatch(
        "edge channel length differs from destination length",
    ))
    length(trigger_mask) == edges || throw(DimensionMismatch(
        "edge trigger-mask length differs from destination length",
    ))
    length(weight) == edges || throw(DimensionMismatch(
        "edge weight length differs from destination length",
    ))

    copied_offsets = UInt32[offsets[index] for index in eachindex(offsets)]
    copied_destination = UInt16[destination[index] for index in eachindex(destination)]
    copied_channel = UInt8[channel[index] for index in eachindex(channel)]
    copied_trigger = UInt8[trigger_mask[index] for index in eachindex(trigger_mask)]
    copied_weight = T[weight[index] for index in eachindex(weight)]

    copied_offsets[1] == UInt32(1) || throw(ArgumentError(
        "source offsets must be one-based",
    ))
    copied_offsets[end] == UInt32(edges + 1) || throw(ArgumentError(
        "terminal source offset must equal edge_count + 1",
    ))
    @inbounds for source in 1:nodes
        first_edge = Int(copied_offsets[source])
        limit = Int(copied_offsets[source + 1]) - 1
        first_edge <= limit + 1 || throw(ArgumentError(
            "source offsets must be nondecreasing",
        ))
        previous_destination = 0
        previous_channel = 0
        previous_trigger = 0
        for edge in first_edge:limit
            destination_node = Int(copied_destination[edge])
            input_channel = Int(copied_channel[edge])
            event_trigger = Int(copied_trigger[edge])
            1 <= destination_node <= nodes || throw(ArgumentError(
                "edge destination is outside the graph",
            ))
            1 <= input_channel <= inputs || throw(ArgumentError(
                "edge channel is outside the destination inbox",
            ))
            event_trigger != 0 || throw(ArgumentError(
                "edge trigger mask must be nonzero",
            ))
            isfinite(copied_weight[edge]) || throw(ArgumentError(
                "edge weight must be finite",
            ))
            ordered = destination_node > previous_destination ||
                      (destination_node == previous_destination &&
                       (input_channel > previous_channel ||
                        (input_channel == previous_channel &&
                         event_trigger >= previous_trigger)))
            ordered || throw(ArgumentError(
                "each source's edges must be sorted by destination, channel, trigger mask",
            ))
            previous_destination = destination_node
            previous_channel = input_channel
            previous_trigger = event_trigger
        end
    end
    return SourceMajorAdjacency{T}(
        nodes,
        inputs,
        Memory{UInt32}(copied_offsets),
        Memory{UInt16}(copied_destination),
        Memory{UInt8}(copied_channel),
        Memory{UInt8}(copied_trigger),
        Memory{T}(copied_weight),
    )
end

struct ArenaOverflowError <: Exception
    kind::UInt8
    capacity::Int
    requested::Int
end

function Base.showerror(io::IO, error::ArenaOverflowError)
    label = error.kind == OVERFLOW_ACTIVE_CAPACITY ? "active COW" :
            error.kind == OVERFLOW_FRONTIER_CAPACITY ? "frontier" :
            error.kind == OVERFLOW_DYNAMIC_CAPACITY ? "dynamic overlay" :
            error.kind == OVERFLOW_SEED_MASK_CONFLICT ? "seed mask conflict" :
            "unknown"
    print(
        io,
        label,
        " capacity overflow: capacity=",
        error.capacity,
        " requested=",
        error.requested,
    )
end

"""Allocation-free summary of one candidate's synchronous refinement."""
struct EventWaveReport
    waves_executed::Int
    visited_sources::Int
    scanned_edges::Int
    delivered_edges::Int
    destination_updates::Int
    emitted_events::Int
    terminated_empty::Bool
    hit_wave_limit::Bool
    fallback_requested::Bool
    overflow_kind::UInt8
end

"""
Fixed SoA state and inbox storage for one candidate micro-execution.

Rows are COW slots and columns are state/inbox fields. This makes the same
field contiguous across slots for a future AoSoA cell adapter while retaining
simple scalar access for the reference adapter.
"""
mutable struct EventArena{T<:AbstractFloat}
    node_count::Int
    state_dim::Int
    inbox_dim::Int
    active_capacity::Int
    frontier_capacity::Int
    overflow_policy::UInt8
    base_state::Matrix{T}
    state::Matrix{T}
    inbox::Matrix{T}
    node_generation::Vector{UInt32}
    node_slot::Vector{Int32}
    active_nodes::Vector{UInt16}
    active_count::Int
    candidate_generation::UInt32
    inbox_stamp::Vector{UInt32}
    inbox_epoch::UInt32
    destination_nodes::Vector{UInt16}
    destination_count::Int
    current_nodes::Vector{UInt16}
    current_masks::Vector{UInt8}
    current_count::Int
    next_nodes::Vector{UInt16}
    next_masks::Vector{UInt8}
    next_count::Int
    seed_stamp::Vector{UInt32}
    seed_index::Vector{Int32}
    seed_epoch::UInt32
    overlay_only_nodes::Vector{UInt16}
    overlay_only_masks::Vector{UInt8}
    overlay_only_count::Int
    overlay_only_stamp::Vector{UInt32}
    overlay_only_index::Vector{Int32}
    overlay_only_epoch::UInt32
    accepting_seeds::Bool
    fallback_requested::Bool
    overflow_kind::UInt8
end

@inline function _overflow_policy(policy::Symbol)
    policy === :error && return OVERFLOW_ERROR
    policy === :fallback && return OVERFLOW_FALLBACK
    throw(ArgumentError("overflow policy must be :error or :fallback"))
end

function EventArena(
    node_count::Integer,
    state_dim::Integer,
    inbox_dim::Integer,
    ::Type{T}=Float32;
    active_capacity::Integer=node_count,
    frontier_capacity::Integer=node_count,
    overflow::Symbol=:error,
) where {T<:AbstractFloat}
    nodes = _checked_positive(node_count, "node_count")
    states = _checked_positive(state_dim, "state_dim")
    inputs = _checked_positive(inbox_dim, "inbox_dim")
    active = _checked_positive(active_capacity, "active_capacity")
    frontier = _checked_positive(frontier_capacity, "frontier_capacity")
    active <= nodes || throw(ArgumentError(
        "active_capacity cannot exceed node_count",
    ))
    frontier <= nodes || throw(ArgumentError(
        "frontier_capacity cannot exceed node_count",
    ))
    nodes <= typemax(UInt16) || throw(ArgumentError(
        "node_count exceeds UInt16 arena identity capacity",
    ))
    return EventArena{T}(
        nodes,
        states,
        inputs,
        active,
        frontier,
        _overflow_policy(overflow),
        zeros(T, nodes, states),
        zeros(T, active, states),
        zeros(T, active, inputs),
        zeros(UInt32, nodes),
        zeros(Int32, nodes),
        zeros(UInt16, active),
        0,
        UInt32(0),
        zeros(UInt32, nodes),
        UInt32(0),
        zeros(UInt16, frontier),
        0,
        zeros(UInt16, frontier),
        zeros(UInt8, frontier),
        0,
        zeros(UInt16, frontier),
        zeros(UInt8, frontier),
        0,
        zeros(UInt32, nodes),
        zeros(Int32, nodes),
        UInt32(0),
        zeros(UInt16, frontier),
        zeros(UInt8, frontier),
        0,
        zeros(UInt32, nodes),
        zeros(Int32, nodes),
        UInt32(0),
        false,
        false,
        OVERFLOW_NONE,
    )
end

@inline active_count(arena::EventArena) = arena.active_count

@inline function _check_node(arena::EventArena, node::Integer)
    physical = Int(node)
    1 <= physical <= arena.node_count || throw(BoundsError(
        1:arena.node_count,
        physical,
    ))
    return physical
end

@inline function _advance_epoch(epoch::UInt32, stamps)
    next = epoch + UInt32(1)
    if iszero(next)
        fill!(stamps, UInt32(0))
        return UInt32(1)
    end
    return next
end

"""Discard the previous COW overlay and begin one fresh candidate."""
function begin_candidate!(arena::EventArena)
    arena.candidate_generation = _advance_epoch(
        arena.candidate_generation,
        arena.node_generation,
    )
    arena.seed_epoch = _advance_epoch(arena.seed_epoch, arena.seed_stamp)
    arena.overlay_only_epoch = _advance_epoch(
        arena.overlay_only_epoch,
        arena.overlay_only_stamp,
    )
    arena.active_count = 0
    arena.destination_count = 0
    arena.current_count = 0
    arena.next_count = 0
    arena.overlay_only_count = 0
    arena.accepting_seeds = true
    arena.fallback_requested = false
    arena.overflow_kind = OVERFLOW_NONE
    return arena
end

@inline function _overflow!(
    arena::EventArena,
    kind::UInt8,
    capacity::Int,
    requested::Int,
)
    if arena.overflow_policy == OVERFLOW_ERROR
        throw(ArenaOverflowError(kind, capacity, requested))
    end
    arena.fallback_requested = true
    arena.overflow_kind = kind
    return 0
end

"""Return the candidate COW slot for `node`, copying base state on first touch."""
function touch_node!(arena::EventArena{T}, node::Integer) where {T}
    physical = _check_node(arena, node)
    arena.candidate_generation != UInt32(0) || throw(ArgumentError(
        "begin_candidate! must be called before touching a node",
    ))
    if @inbounds(arena.node_generation[physical]) == arena.candidate_generation
        return Int(@inbounds arena.node_slot[physical])
    end
    requested = arena.active_count + 1
    requested <= arena.active_capacity || return _overflow!(
        arena,
        OVERFLOW_ACTIVE_CAPACITY,
        arena.active_capacity,
        requested,
    )
    slot = requested
    @inbounds for field in 1:arena.state_dim
        arena.state[slot, field] = arena.base_state[physical, field]
    end
    @inbounds for input in 1:arena.inbox_dim
        arena.inbox[slot, input] = zero(T)
    end
    @inbounds begin
        arena.node_generation[physical] = arena.candidate_generation
        arena.node_slot[physical] = Int32(slot)
        arena.active_nodes[slot] = UInt16(physical)
    end
    arena.active_count = slot
    return slot
end

@inline function state_slot(arena::EventArena, node::Integer)
    physical = _check_node(arena, node)
    @inbounds arena.node_generation[physical] == arena.candidate_generation ||
        return 0
    return Int(@inbounds arena.node_slot[physical])
end

@inline function candidate_state(
    arena::EventArena{T},
    node::Integer,
    field::Integer,
) where {T}
    physical = _check_node(arena, node)
    state_field = Int(field)
    1 <= state_field <= arena.state_dim || throw(BoundsError(
        1:arena.state_dim,
        state_field,
    ))
    slot = state_slot(arena, physical)
    return slot == 0 ?
        @inbounds(arena.base_state[physical, state_field]) :
        @inbounds(arena.state[slot, state_field])
end

"""Seed a hard event. Duplicate seed nodes are merged by bitwise OR."""
function seed_event!(arena::EventArena, node::Integer, event_mask::Integer)
    arena.accepting_seeds || throw(ArgumentError(
        "events may only be seeded after begin_candidate! and before run_event_waves!",
    ))
    arena.fallback_requested && return false
    physical = _check_node(arena, node)
    mask = UInt8(event_mask)
    !iszero(mask) || throw(ArgumentError("seed event mask must be nonzero"))
    if @inbounds(arena.seed_stamp[physical]) == arena.seed_epoch
        index = Int(@inbounds arena.seed_index[physical])
        @inbounds arena.current_masks[index] |= mask
        return true
    end
    requested = arena.current_count + 1
    requested <= arena.frontier_capacity || begin
        _overflow!(
            arena,
            OVERFLOW_FRONTIER_CAPACITY,
            arena.frontier_capacity,
            requested,
        )
        return false
    end
    @inbounds begin
        arena.current_nodes[requested] = UInt16(physical)
        arena.current_masks[requested] = mask
        arena.seed_stamp[physical] = arena.seed_epoch
        arena.seed_index[physical] = Int32(requested)
    end
    arena.current_count = requested
    return true
end

"""
Seed a source for wave-one dynamic-overlay delivery only.

This is for candidate-incidence sources whose current packet/state is valid but
which were not re-evaluated in the candidate mandatory closure. Static edges
must not be scanned for such a source. Duplicate overlay-only seeds merge by
OR. If the same source is also a normal seed, normal delivery wins regardless
of call order and this ledger entry is excluded from wave one.
"""
function seed_overlay_only_event!(
    arena::EventArena,
    node::Integer,
    event_mask::Integer,
)
    arena.accepting_seeds || throw(ArgumentError(
        "events may only be seeded after begin_candidate! and before run_event_waves!",
    ))
    arena.fallback_requested && return false
    physical = _check_node(arena, node)
    mask = UInt8(event_mask)
    !iszero(mask) || throw(ArgumentError(
        "overlay-only seed event mask must be nonzero",
    ))
    if @inbounds(arena.overlay_only_stamp[physical]) == arena.overlay_only_epoch
        index = Int(@inbounds arena.overlay_only_index[physical])
        @inbounds arena.overlay_only_masks[index] |= mask
        return true
    end
    requested = arena.overlay_only_count + 1
    requested <= arena.frontier_capacity || begin
        _overflow!(
            arena,
            OVERFLOW_FRONTIER_CAPACITY,
            arena.frontier_capacity,
            requested,
        )
        return false
    end
    @inbounds begin
        arena.overlay_only_nodes[requested] = UInt16(physical)
        arena.overlay_only_masks[requested] = mask
        arena.overlay_only_stamp[physical] = arena.overlay_only_epoch
        arena.overlay_only_index[physical] = Int32(requested)
    end
    arena.overlay_only_count = requested
    return true
end

@inline function _require_sealed_overlay_only(arena::EventArena)
    arena.accepting_seeds && throw(ArgumentError(
        "overlay-only seed order is available only after event execution seals it",
    ))
    arena.fallback_requested && throw(ArgumentError(
        "overlay-only seed order is unavailable after a fallback request",
    ))
    return nothing
end


"""Number of effective wave-one overlay-only sources in sealed source order."""
@inline function overlay_only_seed_count(arena::EventArena)
    _require_sealed_overlay_only(arena)
    return arena.overlay_only_count
end

"""Effective overlay-only source at `index` after deterministic sealing."""
@inline function overlay_only_seed_source(arena::EventArena, index::Integer)
    _require_sealed_overlay_only(arena)
    physical = Int(index)
    1 <= physical <= arena.overlay_only_count || throw(BoundsError(
        1:arena.overlay_only_count, physical,
    ))
    return Int(@inbounds arena.overlay_only_nodes[physical])
end

"""Effective overlay-only source mask at `index` after deterministic sealing."""
@inline function overlay_only_seed_mask(arena::EventArena, index::Integer)
    _require_sealed_overlay_only(arena)
    physical = Int(index)
    1 <= physical <= arena.overlay_only_count || throw(BoundsError(
        1:arena.overlay_only_count, physical,
    ))
    return @inbounds arena.overlay_only_masks[physical]
end

@inline function _swap!(values, left::Int, right::Int)
    temporary = @inbounds values[left]
    @inbounds values[left] = values[right]
    @inbounds values[right] = temporary
    return nothing
end

@inline function _dynamic_key_less(
    overlay::DynamicSourceMajorOverlay,
    left::Int,
    right::Int,
)
    @inbounds begin
        left_source = overlay.source[left]
        right_source = overlay.source[right]
        left_source != right_source && return left_source < right_source
        left_destination = overlay.destination[left]
        right_destination = overlay.destination[right]
        left_destination != right_destination &&
            return left_destination < right_destination
        left_channel = overlay.channel[left]
        right_channel = overlay.channel[right]
        left_channel != right_channel && return left_channel < right_channel
        left_trigger = overlay.trigger_bit[left]
        right_trigger = overlay.trigger_bit[right]
        left_trigger != right_trigger && return left_trigger < right_trigger
        return overlay.raw_index[left] < overlay.raw_index[right]
    end
end

@inline function _swap_dynamic!(
    overlay::DynamicSourceMajorOverlay,
    left::Int,
    right::Int,
)
    _swap!(overlay.source, left, right)
    _swap!(overlay.destination, left, right)
    _swap!(overlay.channel, left, right)
    _swap!(overlay.trigger_bit, left, right)
    _swap!(overlay.raw_index, left, right)
    return nothing
end

@inline function _sift_dynamic_down!(
    overlay::DynamicSourceMajorOverlay,
    root::Int,
    count::Int,
)
    current = root
    while 2 * current <= count
        child = 2 * current
        if child < count && _dynamic_key_less(overlay, child, child + 1)
            child += 1
        end
        !_dynamic_key_less(overlay, current, child) && break
        _swap_dynamic!(overlay, current, child)
        current = child
    end
    return nothing
end

function _sort_dynamic_prefix!(overlay::DynamicSourceMajorOverlay)
    count = overlay.count
    @inbounds for root in div(count, 2):-1:1
        _sift_dynamic_down!(overlay, root, count)
    end
    @inbounds for limit in count:-1:2
        _swap_dynamic!(overlay, 1, limit)
        _sift_dynamic_down!(overlay, 1, limit - 1)
    end
    return overlay
end

@inline function _same_dynamic_edge(
    overlay::DynamicSourceMajorOverlay,
    left::Int,
    right::Int,
)
    @inbounds return overlay.source[left] == overlay.source[right] &&
        overlay.destination[left] == overlay.destination[right] &&
        overlay.channel[left] == overlay.channel[right] &&
        overlay.trigger_bit[left] == overlay.trigger_bit[right]
end

"""Sort and seal the live overlay prefix without allocating."""
function seal_dynamic_overlay!(overlay::DynamicSourceMajorOverlay)
    overlay.sealed && return overlay
    _sort_dynamic_prefix!(overlay)
    @inbounds for edge in 2:overlay.count
        _same_dynamic_edge(overlay, edge - 1, edge) && throw(ArgumentError(
            "duplicate dynamic event contact",
        ))
    end
    edge = 1
    @inbounds for source in 1:overlay.node_count
        overlay.offsets[source] = UInt32(edge)
        while edge <= overlay.count && Int(overlay.source[edge]) == source
            edge += 1
        end
    end
    overlay.offsets[overlay.node_count + 1] = UInt32(overlay.count + 1)
    edge == overlay.count + 1 || error("dynamic source-major seal lost an edge")
    overlay.sealed = true
    return overlay
end

@inline function _sift_down!(nodes, masks, root::Int, count::Int)
    current = root
    while 2 * current <= count
        child = 2 * current
        if child < count && @inbounds(nodes[child] < nodes[child + 1])
            child += 1
        end
        @inbounds nodes[current] >= nodes[child] && break
        _swap!(nodes, current, child)
        masks === nothing || _swap!(masks, current, child)
        current = child
    end
    return nothing
end

"""Allocation-free ascending heapsort of the first `count` entries."""
function _sort_prefix!(nodes, masks, count::Int)
    @inbounds for root in div(count, 2):-1:1
        _sift_down!(nodes, masks, root, count)
    end
    @inbounds for limit in count:-1:2
        _swap!(nodes, 1, limit)
        masks === nothing || _swap!(masks, 1, limit)
        _sift_down!(nodes, masks, 1, limit - 1)
    end
    return nothing
end

"""
Per-edge delivery adapter protocol.

The static method below preserves the diagnostic scalar graph contract. A
canonical Reduced-Hay adapter specializes this function for both static and
dynamic graph types, reads the refreshed source packet/current compartment
state, and deposits a typed AMPA/NMDA/GABA payload. In particular, a plateau
XOR bit is not itself an onset: the adapter must inspect the source's current
plateau group to distinguish onset from offset.
"""
@inline function deliver_event_edge!(
    adapter,
    arena::EventArena{T},
    graph::SourceMajorAdjacency{T},
    source::Int,
    source_mask::UInt8,
    edge::Int,
    destination::Int,
    slot::Int,
    wave::Int,
) where {T<:AbstractFloat}
    input = Int(@inbounds graph.channel[edge])
    @inbounds arena.inbox[slot, input] += graph.weight[edge]
    return nothing
end

# No generic DynamicSourceMajorOverlay method is provided deliberately. The
# dynamic raw index has no scalar meaning inside the scheduler; a model-owned
# typed adapter must define its delivery.

"""
Cell adapter protocol. The adapter must update `arena.state[slot, :]` from the
fully accumulated `arena.inbox[slot, :]` and return a `UInt8` hard-event mask.
Returning a `Bool` is accepted as the single-bit event convenience form.
"""
@inline advance_event_cell!(adapter, arena, node::Int, slot::Int, wave::Int) =
    adapter(arena, node, slot, wave)

@inline _mask_from_result(result::UInt8) = result
@inline _mask_from_result(result::Bool) = result ? UInt8(1) : NO_EVENT
@inline function _mask_from_result(result::Integer)
    0 <= result <= typemax(UInt8) || throw(ArgumentError(
        "cell adapter event mask must fit UInt8",
    ))
    return UInt8(result)
end

@inline function _empty_report(arena::EventArena)
    return EventWaveReport(
        0, 0, 0, 0, 0, 0,
        true,
        false,
        arena.fallback_requested,
        arena.overflow_kind,
    )
end

function _validate_overlay_only_masks!(arena::EventArena)
    @inbounds for index in 1:arena.overlay_only_count
        source = Int(arena.overlay_only_nodes[index])
        arena.seed_stamp[source] == arena.seed_epoch || continue
        normal_index = Int(arena.seed_index[source])
        normal_mask = arena.current_masks[normal_index]
        overlay_mask = arena.overlay_only_masks[index]
        extra = overlay_mask & ~normal_mask
        iszero(extra) && continue
        if arena.overflow_policy == OVERFLOW_ERROR
            throw(ArgumentError(
                "overlay-only seed mask contains bits absent from the normal seed",
            ))
        end
        arena.fallback_requested = true
        arena.overflow_kind = OVERFLOW_SEED_MASK_CONFLICT
        return false
    end
    return true
end

function _sort_and_compact_overlay_only!(arena::EventArena)
    _sort_prefix!(
        arena.overlay_only_nodes,
        arena.overlay_only_masks,
        arena.overlay_only_count,
    )
    destination = 0
    @inbounds for source_index in 1:arena.overlay_only_count
        source = Int(arena.overlay_only_nodes[source_index])
        # Valid same-source overlap is wholly covered by normal delivery and
        # therefore is not part of the effective overlay-only ledger.
        arena.seed_stamp[source] == arena.seed_epoch && continue
        destination += 1
        if destination != source_index
            arena.overlay_only_nodes[destination] =
                arena.overlay_only_nodes[source_index]
            arena.overlay_only_masks[destination] =
                arena.overlay_only_masks[source_index]
        end
    end
    arena.overlay_only_count = destination
    return arena
end

@inline function _dynamic_precedes_static(
    overlay::DynamicSourceMajorOverlay,
    dynamic_edge::Int,
    graph::SourceMajorAdjacency,
    static_edge::Int,
)
    @inbounds begin
        dynamic_destination = overlay.destination[dynamic_edge]
        static_destination = graph.destination[static_edge]
        dynamic_destination != static_destination &&
            return dynamic_destination < static_destination
        dynamic_channel = overlay.channel[dynamic_edge]
        static_channel = graph.channel[static_edge]
        dynamic_channel != static_channel && return dynamic_channel < static_channel
        dynamic_trigger = overlay.trigger_bit[dynamic_edge]
        static_trigger = graph.trigger_mask[static_edge]
        dynamic_trigger != static_trigger && return dynamic_trigger < static_trigger
        # Static precedes dynamic on an equal logical delivery key. This makes
        # the merged order independent of construction order.
        return false
    end
end

@inline function _has_outgoing_contact(
    graph::SourceMajorAdjacency,
    overlay::Nothing,
    source::Int,
)
    return @inbounds graph.offsets[source] != graph.offsets[source + 1]
end

@inline function _has_outgoing_contact(
    graph::SourceMajorAdjacency,
    overlay::DynamicSourceMajorOverlay,
    source::Int,
)
    return @inbounds(graph.offsets[source] != graph.offsets[source + 1]) ||
        @inbounds(overlay.offsets[source] != overlay.offsets[source + 1])
end

@inline function _touch_wave_destination!(
    arena::EventArena{T},
    destination::Int,
) where {T<:AbstractFloat}
    if @inbounds(arena.inbox_stamp[destination]) != arena.inbox_epoch
        requested = arena.destination_count + 1
        if requested > arena.frontier_capacity
            _overflow!(
                arena,
                OVERFLOW_FRONTIER_CAPACITY,
                arena.frontier_capacity,
                requested,
            )
            return 0
        end
        slot = touch_node!(arena, destination)
        iszero(slot) && return 0
        @inbounds for input in 1:arena.inbox_dim
            arena.inbox[slot, input] = zero(T)
        end
        @inbounds begin
            arena.inbox_stamp[destination] = arena.inbox_epoch
            arena.destination_nodes[requested] = UInt16(destination)
        end
        arena.destination_count = requested
        return slot
    end
    return Int(@inbounds arena.node_slot[destination])
end

"""
    run_event_waves!(arena, graph, adapter; max_waves=7)

Run candidate-local Jacobi waves. Current sources are sorted before every
delivery. All matching edges accumulate first; each unique destination is then
advanced exactly once in sorted node order. Events emitted by that advance are
placed in the next frontier and cannot affect the current wave.

With `overflow=:fallback`, a capacity violation returns immediately with
`fallback_requested=true`. The partial COW overlay must be discarded by
`begin_candidate!` before the caller invokes its exact dense fallback.
"""
function run_event_waves!(
    arena::EventArena{T},
    graph::SourceMajorAdjacency{T},
    adapter;
    max_waves::Integer=CANONICAL_MAX_WAVES,
) where {T<:AbstractFloat}
    arena.overlay_only_count == 0 || throw(ArgumentError(
        "overlay-only seeds require a sealed dynamic overlay",
    ))
    return _run_event_waves!(
        arena,
        graph,
        nothing,
        adapter;
        max_waves=max_waves,
    )
end

function run_event_waves!(
    arena::EventArena{T},
    graph::SourceMajorAdjacency{T},
    overlay::DynamicSourceMajorOverlay,
    adapter;
    max_waves::Integer=CANONICAL_MAX_WAVES,
) where {T<:AbstractFloat}
    overlay.sealed || throw(ArgumentError(
        "seal_dynamic_overlay! must run before event execution",
    ))
    overlay.node_count == arena.node_count || throw(DimensionMismatch(
        "arena and dynamic overlay node counts differ",
    ))
    overlay.inbox_dim == arena.inbox_dim || throw(DimensionMismatch(
        "arena and dynamic overlay inbox dimensions differ",
    ))
    return _run_event_waves!(
        arena,
        graph,
        overlay,
        adapter;
        max_waves=max_waves,
    )
end

function _run_event_waves!(
    arena::EventArena{T},
    graph::SourceMajorAdjacency{T},
    overlay::O,
    adapter;
    max_waves::Integer=CANONICAL_MAX_WAVES,
) where {T<:AbstractFloat,O<:Union{Nothing,DynamicSourceMajorOverlay}}
    graph.node_count == arena.node_count || throw(DimensionMismatch(
        "arena and graph node counts differ",
    ))
    graph.inbox_dim == arena.inbox_dim || throw(DimensionMismatch(
        "arena and graph inbox dimensions differ",
    ))
    waves = Int(max_waves)
    1 <= waves <= CANONICAL_MAX_WAVES || throw(ArgumentError(
        "max_waves must be between 1 and $CANONICAL_MAX_WAVES",
    ))
    arena.accepting_seeds || throw(ArgumentError(
        "begin_candidate! must be called exactly once before each event run",
    ))
    if arena.fallback_requested
        arena.accepting_seeds = false
        return _empty_report(arena)
    end
    if !_validate_overlay_only_masks!(arena)
        arena.accepting_seeds = false
        return _empty_report(arena)
    end
    arena.accepting_seeds = false
    arena.current_count == 0 && arena.overlay_only_count == 0 &&
        return _empty_report(arena)
    _sort_prefix!(arena.current_nodes, arena.current_masks, arena.current_count)
    _sort_and_compact_overlay_only!(arena)

    visited_sources = 0
    scanned_edges = 0
    delivered_edges = 0
    destination_updates = 0
    emitted_events = 0
    waves_executed = 0

    @inbounds for wave in 1:waves
        arena.inbox_epoch = _advance_epoch(arena.inbox_epoch, arena.inbox_stamp)
        arena.destination_count = 0
        arena.next_count = 0

        normal_index = 1
        overlay_only_index = wave == 1 ? 1 : arena.overlay_only_count + 1
        while normal_index <= arena.current_count ||
              overlay_only_index <= arena.overlay_only_count
            normal_available = normal_index <= arena.current_count
            overlay_only_available = overlay_only_index <= arena.overlay_only_count
            normal_source = normal_available ?
                Int(@inbounds arena.current_nodes[normal_index]) : typemax(Int)
            overlay_only_source = overlay_only_available ?
                Int(@inbounds arena.overlay_only_nodes[overlay_only_index]) :
                typemax(Int)
            use_overlay_only = overlay_only_source < normal_source
            if normal_available && overlay_only_available &&
               normal_source == overlay_only_source
                # Normal wins: it scans static+dynamic exactly once. The
                # overlay-only mask is deliberately not merged.
                overlay_only_index += 1
            end
            source = use_overlay_only ? overlay_only_source : normal_source
            source_mask = use_overlay_only ?
                @inbounds(arena.overlay_only_masks[overlay_only_index]) :
                @inbounds(arena.current_masks[normal_index])
            use_overlay_only ? (overlay_only_index += 1) : (normal_index += 1)
            visited_sources += 1
            static_edge = use_overlay_only ? 1 : Int(graph.offsets[source])
            static_limit = use_overlay_only ? 0 :
                Int(graph.offsets[source + 1]) - 1
            dynamic_edge = overlay === nothing ? 1 : Int(overlay.offsets[source])
            dynamic_limit = overlay === nothing ? 0 :
                Int(overlay.offsets[source + 1]) - 1
            while static_edge <= static_limit || dynamic_edge <= dynamic_limit
                use_dynamic = static_edge > static_limit ||
                    (dynamic_edge <= dynamic_limit && _dynamic_precedes_static(
                        overlay,
                        dynamic_edge,
                        graph,
                        static_edge,
                    ))
                scanned_edges += 1
                trigger = use_dynamic ?
                    @inbounds(overlay.trigger_bit[dynamic_edge]) :
                    @inbounds(graph.trigger_mask[static_edge])
                if iszero(source_mask & trigger)
                    use_dynamic ? (dynamic_edge += 1) : (static_edge += 1)
                    continue
                end
                delivered_edges += 1
                destination = Int(use_dynamic ?
                    @inbounds(overlay.destination[dynamic_edge]) :
                    @inbounds(graph.destination[static_edge]))
                slot = _touch_wave_destination!(arena, destination)
                if iszero(slot)
                    return EventWaveReport(
                        waves_executed,
                        visited_sources,
                        scanned_edges,
                        delivered_edges,
                        destination_updates,
                        emitted_events,
                        false,
                        false,
                        true,
                        arena.overflow_kind,
                    )
                end
                if use_dynamic
                    deliver_event_edge!(
                        adapter,
                        arena,
                        overlay,
                        source,
                        source_mask,
                        dynamic_edge,
                        destination,
                        slot,
                        wave,
                    )
                    dynamic_edge += 1
                else
                    deliver_event_edge!(
                        adapter,
                        arena,
                        graph,
                        source,
                        source_mask,
                        static_edge,
                        destination,
                        slot,
                        wave,
                    )
                    static_edge += 1
                end
            end
        end

        _sort_prefix!(
            arena.destination_nodes,
            nothing,
            arena.destination_count,
        )
        for destination_index in 1:arena.destination_count
            destination = Int(arena.destination_nodes[destination_index])
            slot = Int(arena.node_slot[destination])
            result = advance_event_cell!(
                adapter,
                arena,
                destination,
                slot,
                wave,
            )
            event_mask = _mask_from_result(result)
            destination_updates += 1
            if !iszero(event_mask)
                emitted_events += 1
                # The event itself is already represented by the updated
                # destination state/mask. A source with no sealed static or
                # dynamic outgoing contact cannot affect a later wave, so
                # enqueueing it would only add an empty drain wave and inflate
                # the exact topology depth by one.
                _has_outgoing_contact(graph, overlay, destination) || continue
                requested = arena.next_count + 1
                # Unique destinations are advanced once, so next events are
                # unique without another generation table.
                requested <= arena.frontier_capacity || begin
                    _overflow!(
                        arena,
                        OVERFLOW_FRONTIER_CAPACITY,
                        arena.frontier_capacity,
                        requested,
                    )
                    return EventWaveReport(
                        waves_executed,
                        visited_sources,
                        scanned_edges,
                        delivered_edges,
                        destination_updates,
                        emitted_events,
                        false,
                        false,
                        true,
                        arena.overflow_kind,
                    )
                end
                arena.next_nodes[requested] = UInt16(destination)
                arena.next_masks[requested] = event_mask
                arena.next_count = requested
            end
        end
        waves_executed = wave

        nodes = arena.current_nodes
        masks = arena.current_masks
        arena.current_nodes = arena.next_nodes
        arena.current_masks = arena.next_masks
        arena.next_nodes = nodes
        arena.next_masks = masks
        arena.current_count = arena.next_count
        arena.next_count = 0

        if arena.current_count == 0
            return EventWaveReport(
                waves_executed,
                visited_sources,
                scanned_edges,
                delivered_edges,
                destination_updates,
                emitted_events,
                true,
                false,
                false,
                OVERFLOW_NONE,
            )
        end
        # The next frontier is already sorted because destinations were sorted
        # and advanced once in that same order.
    end

    return EventWaveReport(
        waves_executed,
        visited_sources,
        scanned_edges,
        delivered_edges,
        destination_updates,
        emitted_events,
        false,
        true,
        false,
        OVERFLOW_NONE,
    )
end

end # module CanonicalEventArena
