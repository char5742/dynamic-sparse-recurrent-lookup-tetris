module StateCodec

using ..ActiveApicalCell

export CODEC_DIM,
       COMPARTMENT_VOLTAGE_CENTER,
       COMPARTMENT_VOLTAGE_SCALE,
       SOMA_VOLTAGE_CENTER,
       SOMA_VOLTAGE_SCALE,
       AMPA_SCALE,
       NMDA_SCALE,
       GABA_SCALE,
       ADAPTATION_SCALE,
       encode_state_functional,
       encode_state!,
       state_codec_pullback!

const Cell = ActiveApicalCell
const CODEC_DIM = Cell.STATE_DIM

# Fixed nominal equation references.  They deliberately do not follow
# trainable cell parameters, so codec VJPs cannot silently omit rest-parameter
# cotangents.
const COMPARTMENT_VOLTAGE_CENTER = -65.0f0
const COMPARTMENT_VOLTAGE_SCALE = 10.0f0
const SOMA_VOLTAGE_CENTER = -65.0f0
const SOMA_VOLTAGE_SCALE = 15.0f0
const AMPA_SCALE = 0.25f0
const NMDA_SCALE = 1.0f0
const GABA_SCALE = 0.5f0
const ADAPTATION_SCALE = 2.0f0

@inline function _centered_voltage(value, center::Float32, scale::Float32)
    return tanh((value - oftype(value, center)) / oftype(value, scale))
end

@inline function _nonnegative_bounded(value, scale::Float32)
    if value > zero(value)
        typed_scale = oftype(value, scale)
        return value / (typed_scale + value)
    end
    return zero(value)
end

@inline _physical_unit(value) = clamp(value, zero(value), one(value))

"""
    encode_state_functional(state)

Pure reference codec.  It preserves the canonical cell-state ordering:
all compartment-major `[V, AMPA, NMDA, GABA, plateau]` groups followed by
`[soma, adaptation, spike]`.  Physical resting state maps exactly to zero.
"""
function encode_state_functional(state::AbstractVector{T}) where {T<:Real}
    length(state) == CODEC_DIM || throw(DimensionMismatch("expected $CODEC_DIM cell states"))
    compartments = ntuple(Val(Cell.N_COMPARTMENTS)) do compartment
        base = (compartment - 1) * Cell.COMPARTMENT_STATE_DIM
        (
            _centered_voltage(
                state[base + Cell.FIELD_VOLTAGE],
                COMPARTMENT_VOLTAGE_CENTER,
                COMPARTMENT_VOLTAGE_SCALE,
            ),
            _nonnegative_bounded(state[base + Cell.FIELD_AMPA], AMPA_SCALE),
            _nonnegative_bounded(state[base + Cell.FIELD_NMDA], NMDA_SCALE),
            _nonnegative_bounded(state[base + Cell.FIELD_GABA], GABA_SCALE),
            _physical_unit(state[base + Cell.FIELD_PLATEAU]),
        )
    end
    values = ntuple(Val(CODEC_DIM)) do index
        if index <= Cell.N_COMPARTMENTS * Cell.COMPARTMENT_STATE_DIM
            compartment, field_zero = divrem(
                index - 1,
                Cell.COMPARTMENT_STATE_DIM,
            )
            return compartments[compartment + 1][field_zero + 1]
        end
        index == Cell.SOMA_INDEX && return _centered_voltage(
            state[Cell.SOMA_INDEX],
            SOMA_VOLTAGE_CENTER,
            SOMA_VOLTAGE_SCALE,
        )
        index == Cell.ADAPTATION_INDEX && return _nonnegative_bounded(
            state[Cell.ADAPTATION_INDEX],
            ADAPTATION_SCALE,
        )
        return state[Cell.SPIKE_INDEX]
    end
    return [values...]
end

"""
    encode_state!(destination, state)

Allocation-free Float32 codec.  `destination === state` is supported.
"""
function encode_state!(
    destination::AbstractVector{Float32},
    state::AbstractVector{Float32},
)
    length(destination) == CODEC_DIM || throw(DimensionMismatch("expected $CODEC_DIM encoded states"))
    length(state) == CODEC_DIM || throw(DimensionMismatch("expected $CODEC_DIM cell states"))
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        base = (compartment - 1) * Cell.COMPARTMENT_STATE_DIM
        destination[base + Cell.FIELD_VOLTAGE] = _centered_voltage(
            state[base + Cell.FIELD_VOLTAGE],
            COMPARTMENT_VOLTAGE_CENTER,
            COMPARTMENT_VOLTAGE_SCALE,
        )
        destination[base + Cell.FIELD_AMPA] =
            _nonnegative_bounded(state[base + Cell.FIELD_AMPA], AMPA_SCALE)
        destination[base + Cell.FIELD_NMDA] =
            _nonnegative_bounded(state[base + Cell.FIELD_NMDA], NMDA_SCALE)
        destination[base + Cell.FIELD_GABA] =
            _nonnegative_bounded(state[base + Cell.FIELD_GABA], GABA_SCALE)
        destination[base + Cell.FIELD_PLATEAU] =
            _physical_unit(state[base + Cell.FIELD_PLATEAU])
    end
    @inbounds begin
        destination[Cell.SOMA_INDEX] = _centered_voltage(
            state[Cell.SOMA_INDEX],
            SOMA_VOLTAGE_CENTER,
            SOMA_VOLTAGE_SCALE,
        )
        destination[Cell.ADAPTATION_INDEX] =
            _nonnegative_bounded(state[Cell.ADAPTATION_INDEX], ADAPTATION_SCALE)
        destination[Cell.SPIKE_INDEX] = state[Cell.SPIKE_INDEX]
    end
    return destination
end

"""
    state_codec_pullback!(dstate, state, dencoded)

Exact analytic VJP of the continuous codec.  Plateau clipping has unit
derivative only in its physical interior; the spike coordinate is an identity
readout, while spike generation remains a discrete decision in the cell.
"""
@inline function state_codec_pullback!(
    dstate::AbstractVector{T},
    state::AbstractVector{T},
    dencoded::AbstractVector{T},
) where {T<:AbstractFloat}
    length(dstate) == CODEC_DIM || throw(DimensionMismatch("expected $CODEC_DIM state cotangents"))
    length(state) == CODEC_DIM || throw(DimensionMismatch("expected $CODEC_DIM cell states"))
    length(dencoded) == CODEC_DIM || throw(DimensionMismatch("expected $CODEC_DIM encoded cotangents"))
    fill!(dstate, zero(T))
    @inbounds for compartment in 1:Cell.N_COMPARTMENTS
        base = (compartment - 1) * Cell.COMPARTMENT_STATE_DIM

        voltage_index = base + Cell.FIELD_VOLTAGE
        voltage_encoded = _centered_voltage(
            state[voltage_index],
            COMPARTMENT_VOLTAGE_CENTER,
            COMPARTMENT_VOLTAGE_SCALE,
        )
        dstate[voltage_index] = dencoded[voltage_index] *
                                (one(T) - voltage_encoded * voltage_encoded) /
                                T(COMPARTMENT_VOLTAGE_SCALE)

        ampa_index = base + Cell.FIELD_AMPA
        if state[ampa_index] > zero(T)
            denominator = T(AMPA_SCALE) + state[ampa_index]
            dstate[ampa_index] = dencoded[ampa_index] * T(AMPA_SCALE) /
                                 (denominator * denominator)
        end

        nmda_index = base + Cell.FIELD_NMDA
        if state[nmda_index] > zero(T)
            denominator = T(NMDA_SCALE) + state[nmda_index]
            dstate[nmda_index] = dencoded[nmda_index] * T(NMDA_SCALE) /
                                 (denominator * denominator)
        end

        gaba_index = base + Cell.FIELD_GABA
        if state[gaba_index] > zero(T)
            denominator = T(GABA_SCALE) + state[gaba_index]
            dstate[gaba_index] = dencoded[gaba_index] * T(GABA_SCALE) /
                                 (denominator * denominator)
        end

        plateau_index = base + Cell.FIELD_PLATEAU
        if zero(T) < state[plateau_index] < one(T)
            dstate[plateau_index] = dencoded[plateau_index]
        end
    end

    @inbounds begin
        soma_encoded = _centered_voltage(
            state[Cell.SOMA_INDEX],
            SOMA_VOLTAGE_CENTER,
            SOMA_VOLTAGE_SCALE,
        )
        dstate[Cell.SOMA_INDEX] = dencoded[Cell.SOMA_INDEX] *
                                  (one(T) - soma_encoded * soma_encoded) /
                                  T(SOMA_VOLTAGE_SCALE)
        if state[Cell.ADAPTATION_INDEX] > zero(T)
            denominator = T(ADAPTATION_SCALE) + state[Cell.ADAPTATION_INDEX]
            dstate[Cell.ADAPTATION_INDEX] = dencoded[Cell.ADAPTATION_INDEX] *
                                            T(ADAPTATION_SCALE) /
                                            (denominator * denominator)
        end
        dstate[Cell.SPIKE_INDEX] = dencoded[Cell.SPIKE_INDEX]
    end
    return dstate
end

end # module StateCodec
