module ReducedHayCPUNativeEventGraph

using ..ActiveApicalCell

export ConductanceInbox,
       DelayedPayloadRing,
       EXCITATORY,
       EventGraph,
       EventGraphBuilder,
       INHIBITORY,
       advance_payload_ring!,
       clear_inbox!,
       clear_payload_ring!,
       deliver_payloads!,
       deliver_payloads_vjp!,
       edge_slot,
       freeze_event_graph,
       set_current_payload!,
       set_edge!,
       validate_event_graph

const COMPARTMENT_COUNT = ActiveApicalCell.N_COMPARTMENTS
const RECEPTOR_COUNT = 3
const PAYLOAD_CHANNELS = COMPARTMENT_COUNT * RECEPTOR_COUNT
const CHANNEL_AMPA = 1
const CHANNEL_NMDA = 2
const CHANNEL_GABA = 3
const EXCITATORY = UInt8(1)
const INHIBITORY = UInt8(2)
const _INVALID_CELL = Int32(0)
const _INVALID_BYTE = UInt8(0xff)

"""Two-tap compartment- and receptor-preserving event storage."""
mutable struct DelayedPayloadRing{T<:AbstractFloat}
    current::Matrix{T}
    previous::Matrix{T}
end

function DelayedPayloadRing(cell_count::Integer, ::Type{T}=Float32) where {T<:AbstractFloat}
    cell_count > 0 || throw(ArgumentError("cell_count must be positive"))
    cells = Int(cell_count)
    return DelayedPayloadRing{T}(
        zeros(T, PAYLOAD_CHANNELS, cells),
        zeros(T, PAYLOAD_CHANNELS, cells),
    )
end

@inline function _check_source(ring::DelayedPayloadRing, source::Integer)
    1 <= source <= size(ring.current, 2) || throw(BoundsError(ring.current, source))
    return Int(source)
end

function set_current_payload!(
    ring::DelayedPayloadRing{T},
    source::Integer,
    payload::AbstractVector{T},
) where {T<:AbstractFloat}
    length(payload) == PAYLOAD_CHANNELS || throw(DimensionMismatch(
        "payload must preserve all $PAYLOAD_CHANNELS compartment/receptor channels",
    ))
    source_index = _check_source(ring, source)
    @inbounds for channel in 1:PAYLOAD_CHANNELS
        value = payload[channel]
        isfinite(value) && value >= zero(T) || throw(ArgumentError(
            "payload channels must be finite and nonnegative",
        ))
        ring.current[channel, source_index] = value
    end
    return ring
end

function advance_payload_ring!(ring::DelayedPayloadRing)
    ring.previous, ring.current = ring.current, ring.previous
    fill!(ring.current, zero(eltype(ring.current)))
    return ring
end

function clear_payload_ring!(ring::DelayedPayloadRing)
    fill!(ring.current, zero(eltype(ring.current)))
    fill!(ring.previous, zero(eltype(ring.previous)))
    return ring
end

"""
Source-major fixed-slot graph. Every slot is a live edge; fanout, polarity and
delay are invariant. Destination cell and compartment may be replaced only by
the low-frequency validated structural-plasticity step. There is no runtime
enable mask or per-candidate topology mutation.
"""
struct EventGraph
    cell_count::Int
    fanout::Int
    destination_cell::Memory{Int32}
    destination_compartment::Memory{UInt8}
    polarity::Memory{UInt8}
    delay_previous::Memory{UInt8}
end

"""
Construction-only topology storage.  Every slot begins with an invalid
sentinel and `freeze_event_graph` rejects the builder until every slot has been
explicitly populated by `set_edge!`.
"""
mutable struct EventGraphBuilder
    cell_count::Int
    fanout::Int
    destination_cell::Vector{Int32}
    destination_compartment::Vector{UInt8}
    polarity::Vector{UInt8}
    delay_previous::Vector{UInt8}
end

function EventGraphBuilder(cell_count::Integer, fanout::Integer)
    cell_count > 0 || throw(ArgumentError("cell_count must be positive"))
    fanout > 0 || throw(ArgumentError("fanout must be positive"))
    cells = Int(cell_count)
    degree = Int(fanout)
    slots = Base.checked_mul(cells, degree)
    return EventGraphBuilder(
        cells,
        degree,
        fill(_INVALID_CELL, slots),
        fill(_INVALID_BYTE, slots),
        fill(_INVALID_BYTE, slots),
        fill(_INVALID_BYTE, slots),
    )
end

@inline function edge_slot(
    graph::Union{EventGraph,EventGraphBuilder},
    source::Integer,
    relation::Integer,
)
    1 <= source <= graph.cell_count || throw(BoundsError(1:graph.cell_count, source))
    1 <= relation <= graph.fanout || throw(BoundsError(1:graph.fanout, relation))
    return (Int(source) - 1) * graph.fanout + Int(relation)
end

function set_edge!(
    builder::EventGraphBuilder,
    source::Integer,
    relation::Integer;
    destination_cell::Integer,
    destination_compartment::Integer,
    polarity::Integer,
    delay_previous::Bool=false,
)
    1 <= destination_cell <= builder.cell_count ||
        throw(ArgumentError("destination_cell is outside the graph"))
    1 <= destination_compartment <= COMPARTMENT_COUNT ||
        throw(ArgumentError(
            "destination_compartment must be in 1:$COMPARTMENT_COUNT",
        ))
    polarity in (Int(EXCITATORY), Int(INHIBITORY)) ||
        throw(ArgumentError("polarity must be EXCITATORY or INHIBITORY"))
    slot = edge_slot(builder, source, relation)
    # Cold construction path: retain bounds checks so a damaged builder fails
    # closed instead of turning a malformed topology into an unchecked write.
    builder.destination_cell[slot] = Int32(destination_cell)
    builder.destination_compartment[slot] = UInt8(destination_compartment)
    builder.polarity[slot] = UInt8(polarity)
    builder.delay_previous[slot] = ifelse(delay_previous, 0x01, 0x00)
    return builder
end

@inline function _expected_slots(cell_count::Int, fanout::Int)
    cell_count > 0 || throw(ArgumentError("cell_count must be positive"))
    fanout > 0 || throw(ArgumentError("fanout must be positive"))
    return Base.checked_mul(cell_count, fanout)
end

function _validate_topology_arrays(
    cell_count::Int,
    fanout::Int,
    destination_cell,
    destination_compartment,
    polarity,
    delay_previous,
)
    expected = _expected_slots(cell_count, fanout)
    length(destination_cell) == expected ||
        throw(ArgumentError("destination_cell length violates fixed fanout"))
    length(destination_compartment) == expected ||
        throw(ArgumentError("destination_compartment length violates fixed fanout"))
    length(polarity) == expected ||
        throw(ArgumentError("polarity length violates fixed fanout"))
    length(delay_previous) == expected ||
        throw(ArgumentError("delay mask length violates fixed fanout"))
    @inbounds for slot in 1:expected
        destination = Int(destination_cell[slot])
        compartment = Int(destination_compartment[slot])
        1 <= destination <= cell_count ||
            throw(ArgumentError("destination cell at slot $slot is invalid or unset"))
        1 <= compartment <= COMPARTMENT_COUNT ||
            throw(ArgumentError("destination compartment at slot $slot is invalid or unset"))
        polarity[slot] in (EXCITATORY, INHIBITORY) ||
            throw(ArgumentError("polarity at slot $slot is invalid or unset"))
        delay_previous[slot] <= 0x01 ||
            throw(ArgumentError("delay bit at slot $slot is invalid or unset"))
    end
    return nothing
end

function freeze_event_graph(builder::EventGraphBuilder)
    _validate_topology_arrays(
        builder.cell_count,
        builder.fanout,
        builder.destination_cell,
        builder.destination_compartment,
        builder.polarity,
        builder.delay_previous,
    )
    graph = EventGraph(
        builder.cell_count,
        builder.fanout,
        Memory{Int32}(builder.destination_cell),
        Memory{UInt8}(builder.destination_compartment),
        Memory{UInt8}(builder.polarity),
        Memory{UInt8}(builder.delay_previous),
    )
    return validate_event_graph(graph)
end

function validate_event_graph(graph::EventGraph)
    _validate_topology_arrays(
        graph.cell_count,
        graph.fanout,
        graph.destination_cell,
        graph.destination_compartment,
        graph.polarity,
        graph.delay_previous,
    )
    return graph
end

struct ConductanceInbox{T<:AbstractFloat}
    ampa::Matrix{T}
    nmda::Matrix{T}
    gaba::Matrix{T}
end

function ConductanceInbox(cell_count::Integer, ::Type{T}=Float32) where {T<:AbstractFloat}
    cell_count > 0 || throw(ArgumentError("cell_count must be positive"))
    cells = Int(cell_count)
    return ConductanceInbox{T}(
        zeros(T, COMPARTMENT_COUNT, cells),
        zeros(T, COMPARTMENT_COUNT, cells),
        zeros(T, COMPARTMENT_COUNT, cells),
    )
end

function clear_inbox!(inbox::ConductanceInbox)
    fill!(inbox.ampa, zero(eltype(inbox.ampa)))
    fill!(inbox.nmda, zero(eltype(inbox.nmda)))
    fill!(inbox.gaba, zero(eltype(inbox.gaba)))
    return inbox
end

@inline function _check_delivery_shapes(
    inbox::ConductanceInbox,
    graph::EventGraph,
    ring::DelayedPayloadRing,
)
    size(inbox.ampa) == (COMPARTMENT_COUNT, graph.cell_count) ||
        throw(DimensionMismatch("AMPA inbox shape differs from graph"))
    size(inbox.nmda) == size(inbox.ampa) ||
        throw(DimensionMismatch("NMDA inbox shape differs from AMPA inbox"))
    size(inbox.gaba) == size(inbox.ampa) ||
        throw(DimensionMismatch("GABA inbox shape differs from AMPA inbox"))
    size(ring.current) == (PAYLOAD_CHANNELS, graph.cell_count) ||
        throw(DimensionMismatch("current payload source count differs from graph"))
    size(ring.previous) == (PAYLOAD_CHANNELS, graph.cell_count) ||
        throw(DimensionMismatch("previous payload source count differs from graph"))
    return nothing
end

@inline _payload_channel(source_compartment::Int, receptor::Int) =
    (source_compartment - 1) * RECEPTOR_COUNT + receptor

@inline function _checked_payload(tap, channel::Int, source::Int)
    @inbounds value = tap[channel, source]
    isfinite(value) && value >= zero(value) || throw(ArgumentError(
        "event payload channels must be finite and nonnegative",
    ))
    return value
end

@inline function _checked_edge(graph::EventGraph, slot::Int)
    @inbounds begin
        destination = Int(graph.destination_cell[slot])
        compartment = Int(graph.destination_compartment[slot])
        polarity = graph.polarity[slot]
        delay_previous = graph.delay_previous[slot]
    end
    1 <= destination <= graph.cell_count ||
        throw(ArgumentError("event graph destination cell is invalid"))
    1 <= compartment <= COMPARTMENT_COUNT ||
        throw(ArgumentError("event graph destination compartment is invalid"))
    polarity in (EXCITATORY, INHIBITORY) ||
        throw(ArgumentError("event graph polarity is invalid"))
    delay_previous <= 0x01 ||
        throw(ArgumentError("event graph delay bit is invalid"))
    return destination, compartment, polarity, delay_previous
end

"""
    deliver_payloads!(inbox, graph, strengths, ring, active_sources)

Deliver only the sorted, unique active sources.  All fixed-fanout slots are
    live. Every relation selects one source-cell compartment.
An excitatory edge independently maps that compartment's AMPA and NMDA
channels to the matching destination conductances; an inhibitory edge maps
its GABA channel.  The delay bit independently selects `ring.current` or
`ring.previous` for every edge.
"""
function deliver_payloads!(
    inbox::ConductanceInbox{T},
    graph::EventGraph,
    strengths::AbstractVector{T},
    ring::DelayedPayloadRing{T},
    active_sources::AbstractVector{<:Integer},
) where {T<:AbstractFloat}
    _check_delivery_shapes(inbox, graph, ring)
    length(strengths) == length(graph.polarity) ||
        throw(DimensionMismatch("strength cache shape differs from graph"))
    previous_source = 0
    @inbounds for source_value in active_sources
        source = Int(source_value)
        1 <= source <= graph.cell_count ||
            throw(ArgumentError("active source is outside the graph"))
        source > previous_source ||
            throw(ArgumentError("active sources must be unique and strictly increasing"))
        previous_source = source
        first_slot = (source - 1) * graph.fanout + 1
        last_slot = first_slot + graph.fanout - 1
        for slot in first_slot:last_slot
            destination, compartment, polarity, delayed = _checked_edge(graph, slot)
            strength = strengths[slot]
            isfinite(strength) && strength > zero(T) ||
                throw(ArgumentError("edge strength must be finite and positive"))
            source_compartment = mod(slot - first_slot, COMPARTMENT_COUNT) + 1
            tap = delayed == 0x01 ? ring.previous : ring.current
            if polarity == EXCITATORY
                ampa_payload = _checked_payload(
                    tap,
                    _payload_channel(source_compartment, CHANNEL_AMPA),
                    source,
                )
                nmda_payload = _checked_payload(
                    tap,
                    _payload_channel(source_compartment, CHANNEL_NMDA),
                    source,
                )
                inbox.ampa[compartment, destination] = muladd(
                    strength, ampa_payload,
                    inbox.ampa[compartment, destination],
                )
                inbox.nmda[compartment, destination] = muladd(
                    strength, nmda_payload,
                    inbox.nmda[compartment, destination],
                )
            else
                gaba_payload = _checked_payload(
                    tap,
                    _payload_channel(source_compartment, CHANNEL_GABA),
                    source,
                )
                inbox.gaba[compartment, destination] = muladd(
                    strength,
                    gaba_payload,
                    inbox.gaba[compartment, destination],
                )
            end
        end
    end
    return inbox
end

"""
    deliver_payloads_vjp!(dstrength, dpayload, graph, strengths, ring,
                          active_sources, dinbox)

Exact accumulating reverse of `deliver_payloads!`.  For an excitatory edge,
AMPA and NMDA payload channels receive independent cotangents and both
contribute to the strength cotangent.  Inhibition uses the corresponding GABA
channel and cotangent.  Delay chooses the same current/previous channel column
as the forward pass.
"""
function deliver_payloads_vjp!(
    strength_cotangent::AbstractVector{T},
    payload_cotangent::DelayedPayloadRing{T},
    graph::EventGraph,
    strengths::AbstractVector{T},
    ring::DelayedPayloadRing{T},
    active_sources::AbstractVector{<:Integer},
    inbox_cotangent::ConductanceInbox{T},
) where {T<:AbstractFloat}
    _check_delivery_shapes(inbox_cotangent, graph, ring)
    _check_delivery_shapes(inbox_cotangent, graph, payload_cotangent)
    length(strengths) == length(graph.polarity) ||
        throw(DimensionMismatch("strength cache shape differs from graph"))
    length(strength_cotangent) == length(graph.polarity) ||
        throw(DimensionMismatch("strength cotangent shape differs from graph"))
    strength_cotangent === strengths &&
        throw(ArgumentError("strength cotangent must not alias primal strengths"))
    (payload_cotangent === ring ||
     payload_cotangent.current === ring.current ||
     payload_cotangent.current === ring.previous ||
     payload_cotangent.previous === ring.current ||
     payload_cotangent.previous === ring.previous) &&
        throw(ArgumentError("payload cotangent must not alias the primal payload ring"))

    previous_source = 0
    @inbounds for source_value in active_sources
        source = Int(source_value)
        1 <= source <= graph.cell_count ||
            throw(ArgumentError("active source is outside the graph"))
        source > previous_source ||
            throw(ArgumentError("active sources must be unique and strictly increasing"))
        previous_source = source
        first_slot = (source - 1) * graph.fanout + 1
        last_slot = first_slot + graph.fanout - 1
        for slot in first_slot:last_slot
            destination, compartment, polarity, delayed = _checked_edge(graph, slot)
            strength = strengths[slot]
            isfinite(strength) && strength > zero(T) ||
                throw(ArgumentError("edge strength must be finite and positive"))
            source_compartment = mod(slot - first_slot, COMPARTMENT_COUNT) + 1
            tap = delayed == 0x01 ? ring.previous : ring.current
            tap_bar = delayed == 0x01 ?
                payload_cotangent.previous : payload_cotangent.current
            if polarity == EXCITATORY
                ampa_channel = _payload_channel(
                    source_compartment, CHANNEL_AMPA,
                )
                nmda_channel = _payload_channel(
                    source_compartment, CHANNEL_NMDA,
                )
                ampa_bar = inbox_cotangent.ampa[compartment, destination]
                nmda_bar = inbox_cotangent.nmda[compartment, destination]
                ampa_payload = _checked_payload(tap, ampa_channel, source)
                nmda_payload = _checked_payload(tap, nmda_channel, source)
                strength_cotangent[slot] +=
                    ampa_payload * ampa_bar + nmda_payload * nmda_bar
                tap_bar[ampa_channel, source] += strength * ampa_bar
                tap_bar[nmda_channel, source] += strength * nmda_bar
            else
                gaba_channel = _payload_channel(
                    source_compartment, CHANNEL_GABA,
                )
                gaba_bar = inbox_cotangent.gaba[compartment, destination]
                gaba_payload = _checked_payload(tap, gaba_channel, source)
                strength_cotangent[slot] +=
                    gaba_payload * gaba_bar
                tap_bar[gaba_channel, source] += strength * gaba_bar
            end
        end
    end
    return nothing
end

end # module
