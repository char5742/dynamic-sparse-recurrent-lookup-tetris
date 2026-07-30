module OfficialSignedInputReconstruction

export EVENT_ORDER,
    INPUT_SEMANTICS,
    OFFICIAL_DENDRITIC_LOCATIONS,
    OFFICIAL_ELM_INPUT_DIM,
    fill_official_raw_window!

const OFFICIAL_ELM_INPUT_DIM = 1_278
const OFFICIAL_DENDRITIC_LOCATIONS = 639
const EVENT_ORDER = "time_then_axon"
const INPUT_SEMANTICS =
    "E:+strength*event_count, I:-strength*event_count"

@inline _get(object, name::Symbol, default=nothing) =
    if object isa AbstractDict
        get(object, name, get(object, String(name), default))
    elseif hasproperty(object, name)
        getproperty(object, name)
    else
        default
    end

function _required(object, name::Symbol)
    value = _get(object, name, nothing)
    value === nothing &&
        error("official compact field $(String(name)) is absent")
    return value
end

function _offset_range(offsets, local_index, total, label)
    1 <= local_index < length(offsets) ||
        throw(BoundsError(offsets, local_index))
    first_item = Int(offsets[local_index]) + 1
    last_item = Int(offsets[local_index + 1])
    0 <= first_item - 1 <= last_item <= total ||
        error("$label ragged offsets are invalid")
    return first_item:last_item
end

@inline function _contact_channel(segment::Int, kind::UInt8)
    2 <= segment <= 640 ||
        error("official ELM contacts must target Hay dendrites 2:640")
    location = segment - 1
    kind == UInt8(1) && return location
    kind == UInt8(2) &&
        return OFFICIAL_DENDRITIC_LOCATIONS + location
    error("official ELM contact violates Dale E/I coding")
end

function _assert_event_order(event_time, event_axon, event_range)
    previous_time = typemin(Int)
    previous_axon = typemin(Int)
    @inbounds for event in event_range
        time = event_time[event]
        axon = event_axon[event]
        (
            time > previous_time ||
            (time == previous_time && axon >= previous_axon)
        ) || error("compact events violate time_then_axon order")
        previous_time = time
        previous_axon = axon
    end
    return true
end

"""
Expand one compact trial into exact signed Paper-ELM input.

The destination is `1278 × window`: 639 nonnegative excitatory channels then
639 nonpositive inhibitory channels.  There is no static plane and no
duplicated AMPA/NMDA input plane.
"""
function fill_official_raw_window!(
    destination::AbstractMatrix{Float32},
    shard,
    local_index::Int,
    first_time::Int,
    last_time::Int,
)
    size(destination) ==
        (OFFICIAL_ELM_INPUT_DIM, last_time - first_time + 1) ||
        throw(DimensionMismatch("official raw destination shape differs"))
    fill!(destination, 0.0f0)

    contact_axon = Int.(vec(_required(shard, :contact_axon)))
    contact_segment = Int.(vec(_required(shard, :contact_segment)))
    contact_kind = UInt8.(vec(_required(shard, :contact_kind)))
    contact_strength =
        Float32.(vec(_required(shard, :contact_strength)))
    contact_range = _offset_range(
        vec(_required(shard, :contact_trial_offset)),
        local_index,
        length(contact_axon),
        "contact",
    )
    contacts_by_axon = Dict{Int,Vector{Int}}()
    @inbounds for contact in contact_range
        _contact_channel(
            contact_segment[contact],
            contact_kind[contact],
        )
        strength = contact_strength[contact]
        isfinite(strength) && strength >= 0.0f0 ||
            error("official ELM contact strength is invalid")
        push!(
            get!(contacts_by_axon, contact_axon[contact], Int[]),
            contact,
        )
    end

    event_axon = Int.(vec(_required(shard, :event_axon)))
    event_time = Int.(vec(_required(shard, :event_time_bin)))
    event_count = UInt8.(vec(_required(shard, :event_count)))
    event_range = _offset_range(
        vec(_required(shard, :event_trial_offset)),
        local_index,
        length(event_axon),
        "event",
    )
    _assert_event_order(event_time, event_axon, event_range)
    isempty(event_range) && return destination
    local_times = @view event_time[event_range]
    first_event = searchsortedfirst(local_times, first_time - 1)
    last_event = searchsortedlast(local_times, last_time - 1)
    first_event > last_event && return destination
    offset = first(event_range) - 1
    @inbounds for relative in first_event:last_event
        event = offset + relative
        global_time = event_time[event] + 1
        local_time = global_time - first_time + 1
        multiplicity = Float32(event_count[event])
        multiplicity > 0.0f0 ||
            error("official ELM event multiplicity is not positive")
        for contact in get(
            contacts_by_axon,
            event_axon[event],
            Int[],
        )
            channel = _contact_channel(
                contact_segment[contact],
                contact_kind[contact],
            )
            signed_strength =
                contact_kind[contact] == UInt8(1) ?
                contact_strength[contact] :
                -contact_strength[contact]
            destination[channel, local_time] +=
                signed_strength * multiplicity
        end
    end
    all(isfinite, destination) ||
        error("official ELM stream-expanded input is non-finite")
    all(value -> value >= 0.0f0, @view(destination[1:639, :])) ||
        error("official excitatory input became negative")
    all(value -> value <= 0.0f0, @view(destination[640:1278, :])) ||
        error("official inhibitory input became positive")
    return destination
end

end # module OfficialSignedInputReconstruction
