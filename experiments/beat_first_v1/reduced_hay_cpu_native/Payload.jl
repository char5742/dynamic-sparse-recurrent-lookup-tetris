module Payload

using ..ActiveApicalCell
using ..StateCodec

export ANALOG_GAIN_COUNT,
       PAYLOAD_DIM,
       PLATEAU_EVENT_THRESHOLD,
       GAIN_NMDA,
       GAIN_PLATEAU,
       GAIN_SOMA,
       RAW_GAIN_HIGH,
       RAW_GAIN_LOW,
       payload_amplitude_cached,
       payload_amplitude_cached_raw_vjp!,
       payload_amplitude_cached_raw_vjp_unchecked!,
       payload_amplitude_cached_unchecked,
       payload_amplitude_raw,
       payload_amplitude_raw_vjp!,
       payload_channels_cached_unchecked!,
       payload_channels_cached_raw_vjp_unchecked!,
       payload_channels_event_masked_cached_unchecked!,
       payload_channels_event_masked_cached_raw_vjp_unchecked!,
       has_payload_event,
       compartment_event_gain,
       transform_payload_gains!

const Cell = ActiveApicalCell
const Codec = StateCodec

const ANALOG_GAIN_COUNT = 3
const PAYLOAD_DIM = Cell.INPUT_DIM
const GAIN_SOMA = 1
const GAIN_NMDA = 2
const GAIN_PLATEAU = 3

# CPU-cheap hard sigmoid.  The exact zero corner is the analog ablation and the
# exact one corner bounds recurrent drive without a separate architecture flag.
const RAW_GAIN_LOW = -2.0f0
const RAW_GAIN_HIGH = 2.0f0
const PLATEAU_EVENT_THRESHOLD = 0.01f0
const PLATEAU_EVENT_SURROGATE_WIDTH = 0.01f0

@inline function _plateau_event(state, compartment::Int)
    @inbounds plateau = state[
        Cell.state_index(compartment, Cell.FIELD_PLATEAU)
    ]
    return plateau >= typeof(plateau)(PLATEAU_EVENT_THRESHOLD) ?
           one(plateau) : zero(plateau)
end

@inline function compartment_event_gain(
    state::AbstractVector{T},
    compartment::Int,
    event_floor::T,
) where {T<:AbstractFloat}
    @inbounds spike = clamp(state[Cell.SPIKE_INDEX], zero(T), one(T))
    plateau_event = _plateau_event(state, compartment)
    hard_or = plateau_event + (one(T) - plateau_event) * spike
    return muladd(one(T) - event_floor, hard_or, event_floor)
end

@inline function has_payload_event(
    state::AbstractVector{T},
    event_floor::T,
) where {T<:AbstractFloat}
    event_floor > zero(T) && return true
    @inbounds state[Cell.SPIKE_INDEX] > zero(T) && return true
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        _plateau_event(state, compartment) > zero(T) && return true
    end
    return false
end

@inline function _bounded_gain(raw::T) where {T<:AbstractFloat}
    low = T(RAW_GAIN_LOW)
    high = T(RAW_GAIN_HIGH)
    raw <= low && return zero(T)
    raw >= high && return one(T)
    return (raw - low) / (high - low)
end

@inline function _bounded_gain_derivative(raw::T) where {T<:AbstractFloat}
    low = T(RAW_GAIN_LOW)
    high = T(RAW_GAIN_HIGH)
    return low < raw < high ? inv(high - low) : zero(T)
end

@inline function _check_state(state::AbstractVector{T}) where {T<:AbstractFloat}
    length(state) == Codec.CODEC_DIM ||
        throw(DimensionMismatch("expected $(Codec.CODEC_DIM) normalized states"))
    @inbounds begin
        isfinite(state[Cell.SOMA_INDEX]) ||
            throw(ArgumentError("normalized soma state must be finite"))
        spike = state[Cell.SPIKE_INDEX]
        spike == zero(T) || spike == one(T) ||
            throw(ArgumentError("hard spike state must be zero or one"))
    end
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        nmda = state[Cell.state_index(compartment, Cell.FIELD_NMDA)]
        plateau = state[Cell.state_index(compartment, Cell.FIELD_PLATEAU)]
        isfinite(nmda) && zero(T) <= nmda <= one(T) ||
            throw(ArgumentError("normalized NMDA state must be in [0, 1]"))
        isfinite(plateau) && zero(T) <= plateau <= one(T) ||
            throw(ArgumentError("normalized plateau state must be in [0, 1]"))
    end
    return nothing
end

@inline function _check_raw_gains(raw_gains::AbstractVector)
    length(raw_gains) == ANALOG_GAIN_COUNT ||
        throw(DimensionMismatch("expected $ANALOG_GAIN_COUNT raw payload gains"))
    @inbounds for index in 1:ANALOG_GAIN_COUNT
        isfinite(raw_gains[index]) ||
            throw(ArgumentError("raw payload gains must be finite"))
    end
    return nothing
end

@inline function _check_cached_gains(gains::AbstractVector{T}) where {T<:AbstractFloat}
    length(gains) == ANALOG_GAIN_COUNT ||
        throw(DimensionMismatch("expected $ANALOG_GAIN_COUNT transformed payload gains"))
    @inbounds for index in 1:ANALOG_GAIN_COUNT
        gain = gains[index]
        isfinite(gain) && zero(T) <= gain <= one(T) ||
            throw(ArgumentError("transformed payload gains must be in [0, 1]"))
    end
    return nothing
end

@inline function _check_cached_derivatives(
    raw_derivatives::AbstractVector{T},
) where {T<:AbstractFloat}
    length(raw_derivatives) == ANALOG_GAIN_COUNT ||
        throw(DimensionMismatch("expected $ANALOG_GAIN_COUNT raw derivatives"))
    maximum_derivative = inv(T(RAW_GAIN_HIGH) - T(RAW_GAIN_LOW))
    @inbounds for index in 1:ANALOG_GAIN_COUNT
        derivative = raw_derivatives[index]
        isfinite(derivative) && zero(T) <= derivative <= maximum_derivative ||
            throw(ArgumentError("payload raw derivatives are outside the transform range"))
    end
    return nothing
end

"""
    transform_payload_gains!(gains, raw_derivatives, raw_gains)

Transform exactly three raw gains into caller-owned `[0, 1]` storage and cache
their exact raw derivatives at the same update boundary.  The recurrent hot
path consumes these caches and never repeats the raw transform.
"""
function transform_payload_gains!(
    gains::AbstractVector{T},
    raw_derivatives::AbstractVector{T},
    raw_gains::AbstractVector{T},
) where {T<:AbstractFloat}
    length(gains) == ANALOG_GAIN_COUNT ||
        throw(DimensionMismatch("expected $ANALOG_GAIN_COUNT transformed gains"))
    length(raw_derivatives) == ANALOG_GAIN_COUNT ||
        throw(DimensionMismatch("expected $ANALOG_GAIN_COUNT raw derivatives"))
    _check_raw_gains(raw_gains)
    @inbounds for index in 1:ANALOG_GAIN_COUNT
        gains[index] = _bounded_gain(raw_gains[index])
        raw_derivatives[index] = _bounded_gain_derivative(raw_gains[index])
    end
    return gains, raw_derivatives
end

@inline function _continuous_features(state::AbstractVector{T}) where {T<:AbstractFloat}
    soma_activity = max(state[Cell.SOMA_INDEX], zero(T))
    nmda_sum = zero(T)
    plateau_sum = zero(T)
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        nmda_sum += state[Cell.state_index(compartment, Cell.FIELD_NMDA)]
        plateau_sum += state[Cell.state_index(compartment, Cell.FIELD_PLATEAU)]
    end
    inverse_compartments = inv(T(Cell.N_COMPARTMENTS))
    return soma_activity,
           nmda_sum * inverse_compartments,
           plateau_sum * inverse_compartments
end

@inline function _amplitude_from_gains(
    state::AbstractVector{T},
    gains::AbstractVector{T},
) where {T<:AbstractFloat}
    soma_activity, mean_nmda, mean_plateau = _continuous_features(state)
    @inbounds analog = gains[GAIN_SOMA] * soma_activity +
                       gains[GAIN_NMDA] * mean_nmda +
                       gains[GAIN_PLATEAU] * mean_plateau
    @inbounds return state[Cell.SPIKE_INDEX] + analog
end

"""
    payload_channels_cached_unchecked!(destination, state, gains)

Encode the source cell without erasing compartment or receptor identity.  The
The payload channels have the same `(compartment, AMPA/NMDA/GABA)` layout as a cell
input.  A graph edge reads only the source compartment and receptor channel it
needs, so event delivery keeps fixed fanout and scalar edge arithmetic.
"""
@inline function payload_channels_cached_unchecked!(
    destination::AbstractVector{T},
    state::AbstractVector{T},
    gains::AbstractVector{T},
) where {T<:AbstractFloat}
    @inbounds begin
        spike = state[Cell.SPIKE_INDEX]
        adaptation = max(state[Cell.ADAPTATION_INDEX], zero(T))
        soma_gain = gains[GAIN_SOMA]
        nmda_gain = gains[GAIN_NMDA]
        plateau_gain = gains[GAIN_PLATEAU]
        half = T(0.5)
        for compartment in 1:Cell.N_COMPARTMENTS
            voltage = max(
                state[Cell.state_index(compartment, Cell.FIELD_VOLTAGE)],
                zero(T),
            )
            ampa = state[Cell.state_index(compartment, Cell.FIELD_AMPA)]
            nmda = state[Cell.state_index(compartment, Cell.FIELD_NMDA)]
            gaba = state[Cell.state_index(compartment, Cell.FIELD_GABA)]
            plateau = state[Cell.state_index(compartment, Cell.FIELD_PLATEAU)]
            destination[Cell.input_index(compartment, Cell.INPUT_AMPA)] =
                spike + soma_gain * half * (voltage + ampa)
            destination[Cell.input_index(compartment, Cell.INPUT_NMDA)] =
                spike + nmda_gain * nmda + plateau_gain * plateau
            destination[Cell.input_index(compartment, Cell.INPUT_GABA)] =
                spike + soma_gain * half * (gaba + adaptation)
        end
    end
    return destination
end

"""
Write a hard-event payload without collapsing dendritic identity.  A soma
spike opens every source compartment; otherwise only compartments whose
plateau crossed the fixed hard threshold are allowed to transmit.  The
optional event floor is retained solely for continuous diagnostic arms.
"""
@inline function payload_channels_event_masked_cached_unchecked!(
    destination::AbstractVector{T},
    state::AbstractVector{T},
    gains::AbstractVector{T},
    event_floor::T,
) where {T<:AbstractFloat}
    payload_channels_cached_unchecked!(destination, state, gains)
    active = false
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        gate = compartment_event_gain(state, compartment, event_floor)
        active |= gate > zero(T)
        for receptor in 1:Cell.INPUT_CHANNELS
            channel = Cell.input_index(compartment, receptor)
            destination[channel] *= gate
        end
    end
    return active
end

@inline function _plateau_event_surrogate(value::T) where {T<:AbstractFloat}
    threshold = T(PLATEAU_EVENT_THRESHOLD)
    width = T(PLATEAU_EVENT_SURROGATE_WIDTH)
    return max(zero(T), one(T) - abs(value - threshold) / width)
end

"""Conditional surrogate reverse of the branch-local hard event payload."""
@inline function payload_channels_event_masked_cached_raw_vjp_unchecked!(
    dstate::AbstractVector{T},
    draw_gains::AbstractVector{T},
    state::AbstractVector{T},
    gains::AbstractVector{T},
    raw_derivatives::AbstractVector{T},
    channel_cotangent::AbstractVector{T},
    event_floor::T,
    scaled_channel_cotangent::AbstractVector{T},
    payload_value::AbstractVector{T},
) where {T<:AbstractFloat}
    payload_channels_cached_unchecked!(payload_value, state, gains)
    @inbounds spike = clamp(state[Cell.SPIKE_INDEX], zero(T), one(T))
    spike_bar = zero(T)
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        plateau_event = _plateau_event(state, compartment)
        gate = compartment_event_gain(state, compartment, event_floor)
        gate_bar = zero(T)
        for receptor in 1:Cell.INPUT_CHANNELS
            channel = Cell.input_index(compartment, receptor)
            cotangent = channel_cotangent[channel]
            scaled_channel_cotangent[channel] = gate * cotangent
            gate_bar = muladd(payload_value[channel], cotangent, gate_bar)
        end
        spike_bar += (one(T) - event_floor) *
                     (one(T) - plateau_event) * gate_bar
        plateau_index = Cell.state_index(compartment, Cell.FIELD_PLATEAU)
        dstate[plateau_index] += (one(T) - event_floor) *
            (one(T) - spike) *
            _plateau_event_surrogate(state[plateau_index]) * gate_bar
    end
    dstate[Cell.SPIKE_INDEX] += spike_bar
    payload_channels_cached_raw_vjp_unchecked!(
        dstate,
        draw_gains,
        state,
        gains,
        raw_derivatives,
        scaled_channel_cotangent,
    )
    return nothing
end

"""Exact accumulating VJP of `payload_channels_cached_unchecked!`."""
@inline function payload_channels_cached_raw_vjp_unchecked!(
    dstate::AbstractVector{T},
    draw_gains::AbstractVector{T},
    state::AbstractVector{T},
    gains::AbstractVector{T},
    raw_derivatives::AbstractVector{T},
    channel_cotangent::AbstractVector{T},
) where {T<:AbstractFloat}
    @inbounds begin
        soma_gain = gains[GAIN_SOMA]
        nmda_gain = gains[GAIN_NMDA]
        plateau_gain = gains[GAIN_PLATEAU]
        half = T(0.5)
        adaptation = max(state[Cell.ADAPTATION_INDEX], zero(T))
        spike_bar = zero(T)
        adaptation_bar = zero(T)
        soma_gain_bar = zero(T)
        nmda_gain_bar = zero(T)
        plateau_gain_bar = zero(T)
        for compartment in 1:Cell.N_COMPARTMENTS
            ampa_bar = channel_cotangent[
                Cell.input_index(compartment, Cell.INPUT_AMPA)
            ]
            nmda_bar = channel_cotangent[
                Cell.input_index(compartment, Cell.INPUT_NMDA)
            ]
            gaba_bar = channel_cotangent[
                Cell.input_index(compartment, Cell.INPUT_GABA)
            ]
            spike_bar += ampa_bar + nmda_bar + gaba_bar
            voltage_index = Cell.state_index(compartment, Cell.FIELD_VOLTAGE)
            ampa_index = Cell.state_index(compartment, Cell.FIELD_AMPA)
            nmda_index = Cell.state_index(compartment, Cell.FIELD_NMDA)
            gaba_index = Cell.state_index(compartment, Cell.FIELD_GABA)
            plateau_index = Cell.state_index(compartment, Cell.FIELD_PLATEAU)
            voltage = max(state[voltage_index], zero(T))
            ampa = state[ampa_index]
            nmda = state[nmda_index]
            gaba = state[gaba_index]
            plateau = state[plateau_index]
            if state[voltage_index] > zero(T)
                dstate[voltage_index] += ampa_bar * soma_gain * half
            end
            dstate[ampa_index] += ampa_bar * soma_gain * half
            dstate[nmda_index] += nmda_bar * nmda_gain
            dstate[gaba_index] += gaba_bar * soma_gain * half
            dstate[plateau_index] += nmda_bar * plateau_gain
            adaptation_bar += gaba_bar * soma_gain * half
            soma_gain_bar += half * (
                ampa_bar * (voltage + ampa) +
                gaba_bar * (gaba + adaptation)
            )
            nmda_gain_bar += nmda_bar * nmda
            plateau_gain_bar += nmda_bar * plateau
        end
        dstate[Cell.SPIKE_INDEX] += spike_bar
        if state[Cell.ADAPTATION_INDEX] > zero(T)
            dstate[Cell.ADAPTATION_INDEX] += adaptation_bar
        end
        draw_gains[GAIN_SOMA] +=
            soma_gain_bar * raw_derivatives[GAIN_SOMA]
        draw_gains[GAIN_NMDA] +=
            nmda_gain_bar * raw_derivatives[GAIN_NMDA]
        draw_gains[GAIN_PLATEAU] +=
            plateau_gain_bar * raw_derivatives[GAIN_PLATEAU]
    end
    return nothing
end

"""
    payload_amplitude_cached_unchecked(normalized_state, transformed_gains)

Trusted recurrent hot-path primitive.  The prepared-model boundary guarantees
the exact state/gain shapes and value invariants; this function performs only
the payload arithmetic.
"""
@inline function payload_amplitude_cached_unchecked(
    normalized_state::AbstractVector{T},
    transformed_gains::AbstractVector{T},
) where {T<:AbstractFloat}
    return _amplitude_from_gains(normalized_state, transformed_gains)
end

"""
    payload_amplitude_cached(normalized_state, transformed_gains)

Encode one selected cell as one nonnegative scalar event using a precomputed
three-gain cache.  The scalar contains the hard spike plus positive soma,
mean NMDA, and mean plateau evidence.  All compartments, including the
active apical compartment, participate in both means.
"""
@inline function payload_amplitude_cached(
    normalized_state::AbstractVector{T},
    transformed_gains::AbstractVector{T},
) where {T<:AbstractFloat}
    _check_state(normalized_state)
    _check_cached_gains(transformed_gains)
    return payload_amplitude_cached_unchecked(
        normalized_state,
        transformed_gains,
    )
end

"""
    payload_amplitude_raw(normalized_state, raw_gains)

Reference/cold-path form of the scalar encoder.  Production forward should
transform raw gains once and call `payload_amplitude_cached`.
"""
function payload_amplitude_raw(
    normalized_state::AbstractVector{T},
    raw_gains::AbstractVector{T},
) where {T<:AbstractFloat}
    _check_state(normalized_state)
    _check_raw_gains(raw_gains)
    @inbounds begin
        soma_gain = _bounded_gain(raw_gains[GAIN_SOMA])
        nmda_gain = _bounded_gain(raw_gains[GAIN_NMDA])
        plateau_gain = _bounded_gain(raw_gains[GAIN_PLATEAU])
    end
    soma_activity, mean_nmda, mean_plateau = _continuous_features(normalized_state)
    @inbounds return normalized_state[Cell.SPIKE_INDEX] +
                     soma_gain * soma_activity +
                     nmda_gain * mean_nmda +
                     plateau_gain * mean_plateau
end

@inline function _state_vjp!(
    dstate::AbstractVector{T},
    state::AbstractVector{T},
    soma_gain::T,
    nmda_gain::T,
    plateau_gain::T,
    amplitude_cotangent::T,
) where {T<:AbstractFloat}
    @inbounds begin
        dstate[Cell.SPIKE_INDEX] += amplitude_cotangent
        if state[Cell.SOMA_INDEX] > zero(T)
            dstate[Cell.SOMA_INDEX] += amplitude_cotangent * soma_gain
        end
        compartment_scale = inv(T(Cell.N_COMPARTMENTS))
        nmda_seed = amplitude_cotangent * nmda_gain * compartment_scale
        plateau_seed = amplitude_cotangent * plateau_gain * compartment_scale
        for compartment in 1:Cell.N_COMPARTMENTS
            dstate[Cell.state_index(compartment, Cell.FIELD_NMDA)] += nmda_seed
            dstate[Cell.state_index(compartment, Cell.FIELD_PLATEAU)] += plateau_seed
        end
    end
    return nothing
end

"""
    payload_amplitude_cached_raw_vjp!(dstate, draw_gains, state, gains,
                                      raw_derivatives, damplitude)

Allocation-free production VJP with respect to normalized cell state and the
raw payload parameters.  It consumes both arrays prepared by
[`transform_payload_gains!`](@ref), writes directly in raw coordinates, and
does not evaluate the hard-sigmoid transform.
"""
function payload_amplitude_cached_raw_vjp!(
    dstate::AbstractVector{T},
    draw_gains::AbstractVector{T},
    normalized_state::AbstractVector{T},
    transformed_gains::AbstractVector{T},
    raw_derivatives::AbstractVector{T},
    amplitude_cotangent::T,
) where {T<:AbstractFloat}
    _check_state(normalized_state)
    _check_cached_gains(transformed_gains)
    length(dstate) == Codec.CODEC_DIM ||
        throw(DimensionMismatch("expected $(Codec.CODEC_DIM) state cotangents"))
    length(draw_gains) == ANALOG_GAIN_COUNT ||
        throw(DimensionMismatch("expected $ANALOG_GAIN_COUNT raw gain cotangents"))
    _check_cached_derivatives(raw_derivatives)
    isfinite(amplitude_cotangent) ||
        throw(ArgumentError("payload cotangent must be finite"))
    return payload_amplitude_cached_raw_vjp_unchecked!(
        dstate,
        draw_gains,
        normalized_state,
        transformed_gains,
        raw_derivatives,
        amplitude_cotangent,
    )
end

"""
    payload_amplitude_cached_raw_vjp_unchecked!(dstate, draw_gains, state,
                                                gains, raw_derivatives,
                                                damplitude)

Trusted allocation-free reverse primitive paired with
[`payload_amplitude_cached_unchecked`](@ref).  All arrays and cached invariants
must already have been validated at the prepared-model boundary.
"""
@inline function payload_amplitude_cached_raw_vjp_unchecked!(
    dstate::AbstractVector{T},
    draw_gains::AbstractVector{T},
    normalized_state::AbstractVector{T},
    transformed_gains::AbstractVector{T},
    raw_derivatives::AbstractVector{T},
    amplitude_cotangent::T,
) where {T<:AbstractFloat}
    @inbounds _state_vjp!(
        dstate,
        normalized_state,
        transformed_gains[GAIN_SOMA],
        transformed_gains[GAIN_NMDA],
        transformed_gains[GAIN_PLATEAU],
        amplitude_cotangent,
    )
    soma_activity, mean_nmda, mean_plateau = _continuous_features(normalized_state)
    @inbounds begin
        draw_gains[GAIN_SOMA] += amplitude_cotangent * soma_activity *
                                 raw_derivatives[GAIN_SOMA]
        draw_gains[GAIN_NMDA] += amplitude_cotangent * mean_nmda *
                                 raw_derivatives[GAIN_NMDA]
        draw_gains[GAIN_PLATEAU] += amplitude_cotangent * mean_plateau *
                                    raw_derivatives[GAIN_PLATEAU]
    end
    return nothing
end

"""
    payload_amplitude_raw_vjp!(dstate, draw_gains, state, raw_gains,
                               damplitude)

Allocation-free accumulating reference VJP through both the scalar encoder and
the bounded raw-gain transform.  There is one scalar payload cotangent; receptor
cotangents have already been combined by `EventGraph` according to edge polarity.
"""
function payload_amplitude_raw_vjp!(
    dstate::AbstractVector{T},
    draw_gains::AbstractVector{T},
    normalized_state::AbstractVector{T},
    raw_gains::AbstractVector{T},
    amplitude_cotangent::T,
) where {T<:AbstractFloat}
    _check_state(normalized_state)
    _check_raw_gains(raw_gains)
    length(dstate) == Codec.CODEC_DIM ||
        throw(DimensionMismatch("expected $(Codec.CODEC_DIM) state cotangents"))
    length(draw_gains) == ANALOG_GAIN_COUNT ||
        throw(DimensionMismatch("expected $ANALOG_GAIN_COUNT raw gain cotangents"))
    isfinite(amplitude_cotangent) ||
        throw(ArgumentError("payload cotangent must be finite"))
    @inbounds begin
        soma_gain = _bounded_gain(raw_gains[GAIN_SOMA])
        nmda_gain = _bounded_gain(raw_gains[GAIN_NMDA])
        plateau_gain = _bounded_gain(raw_gains[GAIN_PLATEAU])
    end
    _state_vjp!(
        dstate,
        normalized_state,
        soma_gain,
        nmda_gain,
        plateau_gain,
        amplitude_cotangent,
    )
    soma_activity, mean_nmda, mean_plateau = _continuous_features(normalized_state)
    @inbounds begin
        draw_gains[GAIN_SOMA] += amplitude_cotangent * soma_activity *
                                 _bounded_gain_derivative(raw_gains[GAIN_SOMA])
        draw_gains[GAIN_NMDA] += amplitude_cotangent * mean_nmda *
                                 _bounded_gain_derivative(raw_gains[GAIN_NMDA])
        draw_gains[GAIN_PLATEAU] += amplitude_cotangent * mean_plateau *
                                    _bounded_gain_derivative(raw_gains[GAIN_PLATEAU])
    end
    return nothing
end

end # module Payload
