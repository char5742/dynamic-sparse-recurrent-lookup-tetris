module HighDimensionalCellPacket

using ..ActiveApicalCell

const Cell = ActiveApicalCell

export PACKET_DIM,
       PACKET_BYTES,
       VOLTAGE_LANE_FIRST,
       AMPA_LANE_FIRST,
       NMDA_LANE_FIRST,
       GABA_LANE_FIRST,
       PLATEAU_LANE_FIRST,
       MARGIN_LANE,
       ADAPTATION_LANE,
       VOLTAGE_CENTER,
       VOLTAGE_SCALE,
       CONDUCTANCE_SCALE,
       PLATEAU_SCALE,
       MARGIN_SCALE,
       ADAPTATION_SCALE,
       packet_lane,
       cell_packet!,
       cell_packet_pullback!,
       cell_packet_column!,
       cell_packet_column_pullback!

"""
Every meaningful continuous Reduced-Hay transition coordinate.

Nine compartments each expose voltage, AMPA, NMDA, GABA and plateau state
(45 lanes). The exact pre-reset soma margin and adaptation add two lanes. The
hard spike remains on the separate event/control plane and is never smuggled
into the continuous task VJP.
"""
const PACKET_DIM = Cell.STATE_DIM - 1
const PACKET_BYTES = PACKET_DIM * sizeof(Float32)

const VOLTAGE_LANE_FIRST = 1
const AMPA_LANE_FIRST = VOLTAGE_LANE_FIRST + Cell.N_COMPARTMENTS
const NMDA_LANE_FIRST = AMPA_LANE_FIRST + Cell.N_COMPARTMENTS
const GABA_LANE_FIRST = NMDA_LANE_FIRST + Cell.N_COMPARTMENTS
const PLATEAU_LANE_FIRST = GABA_LANE_FIRST + Cell.N_COMPARTMENTS
const MARGIN_LANE = PLATEAU_LANE_FIRST + Cell.N_COMPARTMENTS
const ADAPTATION_LANE = MARGIN_LANE + 1

# Fixed physical units keep the packet map independent of trainable readout
# parameters. Softsign gives even an exact zero a nonzero local derivative.
const VOLTAGE_CENTER = -65.0f0
const VOLTAGE_SCALE = 5.0f0
const CONDUCTANCE_SCALE = 0.03125f0
const PLATEAU_SCALE = 0.03125f0
const MARGIN_SCALE = 5.0f0
const ADAPTATION_SCALE = 0.25f0

@inline _softsign(value) = value / (one(value) + abs(value))

@inline function _softsign_derivative(value)
    denominator = one(value) + abs(value)
    return inv(denominator * denominator)
end

@inline function packet_lane(compartment::Integer, field::Integer)
    1 <= compartment <= Cell.N_COMPARTMENTS ||
        throw(BoundsError(1:Cell.N_COMPARTMENTS, compartment))
    1 <= field <= Cell.COMPARTMENT_STATE_DIM ||
        throw(BoundsError(1:Cell.COMPARTMENT_STATE_DIM, field))
    first = field == Cell.FIELD_VOLTAGE ? VOLTAGE_LANE_FIRST :
            field == Cell.FIELD_AMPA ? AMPA_LANE_FIRST :
            field == Cell.FIELD_NMDA ? NMDA_LANE_FIRST :
            field == Cell.FIELD_GABA ? GABA_LANE_FIRST :
            PLATEAU_LANE_FIRST
    return first + Int(compartment) - 1
end

@inline function _check_packet(packet)
    length(packet) == PACKET_DIM || throw(DimensionMismatch(
        "cell packet must contain exactly $PACKET_DIM coordinates",
    ))
    return nothing
end

@inline function _check_state(state)
    length(state) == Cell.STATE_DIM || throw(DimensionMismatch(
        "cell state must contain $(Cell.STATE_DIM) coordinates",
    ))
    return nothing
end

@inline function _scale(::Type{T}, field::Int) where {T<:AbstractFloat}
    field == Cell.FIELD_VOLTAGE && return T(VOLTAGE_SCALE)
    field == Cell.FIELD_PLATEAU && return T(PLATEAU_SCALE)
    return T(CONDUCTANCE_SCALE)
end

@inline function _center(::Type{T}, field::Int) where {T<:AbstractFloat}
    return field == Cell.FIELD_VOLTAGE ? T(VOLTAGE_CENTER) : zero(T)
end

"""Write all 47 continuous transition coordinates into a bounded packet."""
function cell_packet!(
    packet::AbstractVector{T},
    previous_state::AbstractVector{T},
    next_state::AbstractVector{T},
    cache::Cell.CellParameterCache{T},
) where {T<:AbstractFloat}
    _check_packet(packet)
    _check_state(previous_state)
    _check_state(next_state)
    @inbounds for field in 1:Cell.COMPARTMENT_STATE_DIM
        center = _center(T, field)
        inv_scale = inv(_scale(T, field))
        for compartment in 1:Cell.N_COMPARTMENTS
            lane = packet_lane(compartment, field)
            state = Cell.state_index(compartment, field)
            packet[lane] = _softsign(
                (next_state[state] - center) * inv_scale,
            )
        end
    end
    margin = Cell.spike_margin_from_transition(
        previous_state,
        next_state,
        cache,
    )
    packet[MARGIN_LANE] = _softsign(margin / T(MARGIN_SCALE))
    packet[ADAPTATION_LANE] = _softsign(
        next_state[Cell.ADAPTATION_INDEX] / T(ADAPTATION_SCALE),
    )
    return packet
end

"""Allocation-free structure-of-arrays form of [`cell_packet!`](@ref)."""
function cell_packet_column!(
    packet::AbstractMatrix{T},
    states::AbstractArray{T,3},
    cell::Integer,
    previous_column::Integer,
    next_column::Integer,
    cache::Cell.CellParameterCache{T},
) where {T<:AbstractFloat}
    size(packet, 1) == PACKET_DIM || throw(DimensionMismatch(
        "cell packet matrix must have $PACKET_DIM rows",
    ))
    size(states, 1) == Cell.STATE_DIM || throw(DimensionMismatch(
        "cell state tensor must have $(Cell.STATE_DIM) rows",
    ))
    selected = Int(cell)
    1 <= selected <= size(packet, 2) ||
        throw(BoundsError(axes(packet, 2), selected))
    1 <= selected <= size(states, 2) ||
        throw(BoundsError(axes(states, 2), selected))
    previous = Int(previous_column)
    next = Int(next_column)
    1 <= previous <= size(states, 3) ||
        throw(BoundsError(axes(states, 3), previous))
    1 <= next <= size(states, 3) ||
        throw(BoundsError(axes(states, 3), next))
    @inbounds for field in 1:Cell.COMPARTMENT_STATE_DIM
        center = _center(T, field)
        inv_scale = inv(_scale(T, field))
        for compartment in 1:Cell.N_COMPARTMENTS
            lane = packet_lane(compartment, field)
            state = Cell.state_index(compartment, field)
            packet[lane, selected] = _softsign(
                (states[state, selected, next] - center) * inv_scale,
            )
        end
    end
    margin = Cell.spike_margin_from_transition(
        @view(states[:, selected, previous]),
        @view(states[:, selected, next]),
        cache,
    )
    packet[MARGIN_LANE, selected] =
        _softsign(margin / T(MARGIN_SCALE))
    packet[ADAPTATION_LANE, selected] = _softsign(
        states[Cell.ADAPTATION_INDEX, selected, next] /
        T(ADAPTATION_SCALE),
    )
    return packet
end

"""
Exact packet-map cotangent excluding the margin path.

`dnext_state` is overwritten with the 45 compartment and adaptation
cotangents. The returned scalar is the exact cotangent of the *uncompanded*
pre-reset margin and must be passed as `direct_margin_cotangent` to
`cell_step_conditional_pullback!`.
"""
function cell_packet_pullback!(
    dnext_state::AbstractVector{T},
    packet_bar::AbstractVector{T},
    previous_state::AbstractVector{T},
    next_state::AbstractVector{T},
    cache::Cell.CellParameterCache{T},
) where {T<:AbstractFloat}
    _check_state(dnext_state)
    _check_packet(packet_bar)
    _check_state(previous_state)
    _check_state(next_state)
    fill!(dnext_state, zero(T))
    @inbounds for field in 1:Cell.COMPARTMENT_STATE_DIM
        center = _center(T, field)
        inv_scale = inv(_scale(T, field))
        for compartment in 1:Cell.N_COMPARTMENTS
            lane = packet_lane(compartment, field)
            state = Cell.state_index(compartment, field)
            normalized = (next_state[state] - center) * inv_scale
            dnext_state[state] = packet_bar[lane] *
                _softsign_derivative(normalized) * inv_scale
        end
    end
    adaptation_normalized =
        next_state[Cell.ADAPTATION_INDEX] / T(ADAPTATION_SCALE)
    dnext_state[Cell.ADAPTATION_INDEX] = packet_bar[ADAPTATION_LANE] *
        _softsign_derivative(adaptation_normalized) /
        T(ADAPTATION_SCALE)
    dnext_state[Cell.SPIKE_INDEX] = zero(T)
    margin = Cell.spike_margin_from_transition(
        previous_state,
        next_state,
        cache,
    )
    return packet_bar[MARGIN_LANE] *
           _softsign_derivative(margin / T(MARGIN_SCALE)) /
           T(MARGIN_SCALE)
end

"""Structure-of-arrays form of [`cell_packet_pullback!`](@ref)."""
function cell_packet_column_pullback!(
    dnext_state::AbstractVector{T},
    packet_bar::AbstractMatrix{T},
    states::AbstractArray{T,3},
    cell::Integer,
    previous_column::Integer,
    next_column::Integer,
    cache::Cell.CellParameterCache{T},
) where {T<:AbstractFloat}
    _check_state(dnext_state)
    size(packet_bar, 1) == PACKET_DIM || throw(DimensionMismatch(
        "cell packet cotangent matrix must have $PACKET_DIM rows",
    ))
    size(states, 1) == Cell.STATE_DIM || throw(DimensionMismatch(
        "cell state tensor must have $(Cell.STATE_DIM) rows",
    ))
    selected = Int(cell)
    1 <= selected <= size(packet_bar, 2) ||
        throw(BoundsError(axes(packet_bar, 2), selected))
    1 <= selected <= size(states, 2) ||
        throw(BoundsError(axes(states, 2), selected))
    previous = Int(previous_column)
    next = Int(next_column)
    1 <= previous <= size(states, 3) ||
        throw(BoundsError(axes(states, 3), previous))
    1 <= next <= size(states, 3) ||
        throw(BoundsError(axes(states, 3), next))
    fill!(dnext_state, zero(T))
    @inbounds for field in 1:Cell.COMPARTMENT_STATE_DIM
        center = _center(T, field)
        inv_scale = inv(_scale(T, field))
        for compartment in 1:Cell.N_COMPARTMENTS
            lane = packet_lane(compartment, field)
            state = Cell.state_index(compartment, field)
            normalized =
                (states[state, selected, next] - center) * inv_scale
            dnext_state[state] = packet_bar[lane, selected] *
                _softsign_derivative(normalized) * inv_scale
        end
    end
    adaptation_normalized =
        states[Cell.ADAPTATION_INDEX, selected, next] /
        T(ADAPTATION_SCALE)
    dnext_state[Cell.ADAPTATION_INDEX] =
        packet_bar[ADAPTATION_LANE, selected] *
        _softsign_derivative(adaptation_normalized) /
        T(ADAPTATION_SCALE)
    dnext_state[Cell.SPIKE_INDEX] = zero(T)
    margin = Cell.spike_margin_from_transition(
        @view(states[:, selected, previous]),
        @view(states[:, selected, next]),
        cache,
    )
    return packet_bar[MARGIN_LANE, selected] *
           _softsign_derivative(margin / T(MARGIN_SCALE)) /
           T(MARGIN_SCALE)
end

@assert PACKET_DIM == 47
@assert PACKET_BYTES == 188
@assert ADAPTATION_LANE == PACKET_DIM

end # module HighDimensionalCellPacket
