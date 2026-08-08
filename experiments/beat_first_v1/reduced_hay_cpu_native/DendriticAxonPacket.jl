module DendriticAxonPacket

using ..ActiveApicalCell

const Cell = ActiveApicalCell

export GROUP_COUNT,
       FIELD_COUNT,
       PACKET_DIM,
       PACKET_BYTES,
       FAST_FIELD,
       NMDA_FIELD,
       INHIBITORY_FIELD,
       EVENT_DIM,
       SOMA_EVENT,
       PLATEAU_EVENT_FIRST,
       PLATEAU_EVENT_THRESHOLD,
       packet_lane,
       plateau_event_lane,
       axon_packet!,
       axon_packet_pullback!,
       hard_events!,
       ordered_binary_deposit!,
       ordered_binary_deposit_pullback!

"""
The canonical inter-cell message is four ordered branch-pair groups, each
carrying fast excitation, voltage-unblocked NMDA/plateau evidence and
inhibitory/opponent evidence.  All twelve coordinates are bounded and
nonnegative.  The remaining continuous Reduced-Hay state stays private to the
cell; hard events travel separately.
"""
const GROUP_COUNT = 4
const FIELD_COUNT = 3
const PACKET_DIM = GROUP_COUNT * FIELD_COUNT
const PACKET_BYTES = PACKET_DIM * sizeof(Float32)

const FAST_FIELD = 1
const NMDA_FIELD = 2
const INHIBITORY_FIELD = 3

const EVENT_DIM = 1 + GROUP_COUNT
const SOMA_EVENT = 1
const PLATEAU_EVENT_FIRST = 2
const PLATEAU_EVENT_THRESHOLD = 0.01f0

# Fixed physical units deliberately keep the axon map independent of the
# trainable cell-parameter transform.  The only cache-dependent coordinate is
# the exact pre-reset soma margin, whose cotangent is returned explicitly to
# the owning cell transition VJP.
const VOLTAGE_CENTER = -65.0f0
const VOLTAGE_SCALE = 5.0f0
const NMDA_HALF_VOLTAGE = -55.0f0
const NMDA_SLOPE = 5.0f0
const CONDUCTANCE_SCALE = 0.03125f0
const PLATEAU_SCALE = 0.03125f0
const ADAPTATION_SCALE = 0.25f0
const MARGIN_SCALE = 5.0f0
const SMOOTH_POSITIVE_EPSILON = 1.0f-3

const APICAL_FAST_GAIN = 0.25f0
const APICAL_NMDA_GAIN = 0.50f0
const APICAL_INHIBITORY_GAIN = 0.25f0
const ADAPTATION_INHIBITORY_GAIN = 0.25f0
const SOMA_EVIDENCE_GAIN = 0.125f0

@inline function packet_lane(group::Integer, field::Integer)
    1 <= group <= GROUP_COUNT || throw(BoundsError(1:GROUP_COUNT, group))
    1 <= field <= FIELD_COUNT || throw(BoundsError(1:FIELD_COUNT, field))
    return (Int(group) - 1) * FIELD_COUNT + Int(field)
end

@inline function plateau_event_lane(group::Integer)
    1 <= group <= GROUP_COUNT || throw(BoundsError(1:GROUP_COUNT, group))
    return PLATEAU_EVENT_FIRST + Int(group) - 1
end

@inline function _check_state(state)
    length(state) == Cell.STATE_DIM || throw(DimensionMismatch(
        "cell state must contain $(Cell.STATE_DIM) coordinates",
    ))
    return nothing
end

@inline function _check_packet(packet)
    length(packet) == PACKET_DIM || throw(DimensionMismatch(
        "axon packet must contain exactly $PACKET_DIM coordinates",
    ))
    return nothing
end

@inline function _check_events(events)
    length(events) == EVENT_DIM || throw(DimensionMismatch(
        "event plane must contain exactly $EVENT_DIM hard bits",
    ))
    return nothing
end

@inline _positive(value) = max(value, zero(value))

@inline function _positive_derivative(value)
    return value > zero(value) ? one(value) : zero(value)
end

@inline function _smooth_positive(value::T) where {T<:AbstractFloat}
    epsilon = T(SMOOTH_POSITIVE_EPSILON)
    return T(0.5) * (value + sqrt(muladd(value, value, epsilon * epsilon)))
end

@inline function _smooth_positive_derivative(value::T) where {T<:AbstractFloat}
    epsilon = T(SMOOTH_POSITIVE_EPSILON)
    radius = sqrt(muladd(value, value, epsilon * epsilon))
    return T(0.5) * (one(T) + value / radius)
end

@inline function _bounded_positive(value)
    return value / (one(value) + value)
end

@inline function _bounded_positive_derivative(value)
    denominator = one(value) + value
    return inv(denominator * denominator)
end

@inline function _sigmoid(value)
    if value >= zero(value)
        inverse = exp(-value)
        return inv(one(value) + inverse)
    end
    exponential = exp(value)
    return exponential / (one(value) + exponential)
end

@inline function _branch_evidence(
    state::AbstractVector{T},
    compartment::Int,
) where {T<:AbstractFloat}
    voltage = @inbounds state[Cell.state_index(compartment, Cell.FIELD_VOLTAGE)]
    ampa = @inbounds state[Cell.state_index(compartment, Cell.FIELD_AMPA)]
    nmda = @inbounds state[Cell.state_index(compartment, Cell.FIELD_NMDA)]
    gaba = @inbounds state[Cell.state_index(compartment, Cell.FIELD_GABA)]
    plateau = @inbounds state[Cell.state_index(compartment, Cell.FIELD_PLATEAU)]

    voltage_normalized = (voltage - T(VOLTAGE_CENTER)) / T(VOLTAGE_SCALE)
    voltage_positive = _smooth_positive(voltage_normalized)
    voltage_negative = _smooth_positive(-voltage_normalized)
    ampa_positive = _positive(ampa) / T(CONDUCTANCE_SCALE)
    nmda_positive = _positive(nmda) / T(CONDUCTANCE_SCALE)
    gaba_positive = _positive(gaba) / T(CONDUCTANCE_SCALE)
    plateau_positive = _positive(plateau) / T(PLATEAU_SCALE)
    unblock = _sigmoid(
        (voltage - T(NMDA_HALF_VOLTAGE)) / T(NMDA_SLOPE),
    )
    fast = ampa_positive + voltage_positive
    slow = nmda_positive * unblock + plateau_positive
    inhibitory = gaba_positive + voltage_negative
    return fast, slow, inhibitory
end

@inline function _accumulate_branch_pullback!(
    dstate::AbstractVector{T},
    state::AbstractVector{T},
    compartment::Int,
    dfast::T,
    dslow::T,
    dinhibitory::T,
) where {T<:AbstractFloat}
    voltage_index = Cell.state_index(compartment, Cell.FIELD_VOLTAGE)
    ampa_index = Cell.state_index(compartment, Cell.FIELD_AMPA)
    nmda_index = Cell.state_index(compartment, Cell.FIELD_NMDA)
    gaba_index = Cell.state_index(compartment, Cell.FIELD_GABA)
    plateau_index = Cell.state_index(compartment, Cell.FIELD_PLATEAU)

    voltage = @inbounds state[voltage_index]
    ampa = @inbounds state[ampa_index]
    nmda = @inbounds state[nmda_index]
    gaba = @inbounds state[gaba_index]
    plateau = @inbounds state[plateau_index]
    voltage_normalized = (voltage - T(VOLTAGE_CENTER)) / T(VOLTAGE_SCALE)
    nmda_positive = _positive(nmda) / T(CONDUCTANCE_SCALE)
    unblock = _sigmoid(
        (voltage - T(NMDA_HALF_VOLTAGE)) / T(NMDA_SLOPE),
    )

    dvoltage = dfast * _smooth_positive_derivative(voltage_normalized) /
               T(VOLTAGE_SCALE)
    dvoltage -= dinhibitory * _smooth_positive_derivative(-voltage_normalized) /
                T(VOLTAGE_SCALE)
    dvoltage += dslow * nmda_positive * unblock * (one(T) - unblock) /
                T(NMDA_SLOPE)
    @inbounds begin
        dstate[voltage_index] += dvoltage
        dstate[ampa_index] += dfast * _positive_derivative(ampa) /
                              T(CONDUCTANCE_SCALE)
        dstate[nmda_index] += dslow * unblock * _positive_derivative(nmda) /
                              T(CONDUCTANCE_SCALE)
        dstate[gaba_index] += dinhibitory * _positive_derivative(gaba) /
                              T(CONDUCTANCE_SCALE)
        dstate[plateau_index] += dslow * _positive_derivative(plateau) /
                                 T(PLATEAU_SCALE)
    end
    return dstate
end

@inline function _global_evidence(
    previous_state::AbstractVector{T},
    next_state::AbstractVector{T},
    cache::Cell.CellParameterCache{T},
) where {T<:AbstractFloat}
    apical_fast, apical_slow, apical_inhibitory =
        _branch_evidence(next_state, Cell.N_COMPARTMENTS)
    apical_excitation_raw = T(0.5) * (apical_fast + apical_slow)
    apical_excitation = _bounded_positive(apical_excitation_raw)
    apical_inhibition = _bounded_positive(apical_inhibitory)
    adaptation = _positive(next_state[Cell.ADAPTATION_INDEX]) /
                 T(ADAPTATION_SCALE)
    margin = Cell.spike_margin_from_transition(previous_state, next_state, cache)
    margin_normalized = margin / T(MARGIN_SCALE)
    margin_positive = _smooth_positive(margin_normalized)
    margin_negative = _smooth_positive(-margin_normalized)
    return apical_fast,
           apical_slow,
           apical_inhibitory,
           apical_excitation_raw,
           apical_excitation,
           apical_inhibition,
           adaptation,
           margin_normalized,
           margin_positive,
           margin_negative
end

"""Write the bounded nonnegative 12D axon packet for one cell transition."""
function axon_packet!(
    packet::AbstractVector{T},
    previous_state::AbstractVector{T},
    next_state::AbstractVector{T},
    cache::Cell.CellParameterCache{T},
) where {T<:AbstractFloat}
    _check_packet(packet)
    _check_state(previous_state)
    _check_state(next_state)
    _, _, _, _, apical_excitation, apical_inhibition, adaptation, _,
        margin_positive, margin_negative =
        _global_evidence(previous_state, next_state, cache)

    @inbounds for group in 1:GROUP_COUNT
        first_branch = 2 * group - 1
        second_branch = first_branch + 1
        fast_first, slow_first, inhibitory_first =
            _branch_evidence(next_state, first_branch)
        fast_second, slow_second, inhibitory_second =
            _branch_evidence(next_state, second_branch)
        pair_fast = T(0.5) * (fast_first + fast_second)
        pair_slow = T(0.5) * (slow_first + slow_second)
        pair_inhibitory = T(0.5) * (inhibitory_first + inhibitory_second)

        fast_raw = pair_fast * (
            one(T) + T(APICAL_FAST_GAIN) * apical_excitation
        ) + T(SOMA_EVIDENCE_GAIN) * margin_positive
        slow_raw = pair_slow * (
            one(T) + T(APICAL_NMDA_GAIN) * apical_excitation
        )
        inhibitory_raw = pair_inhibitory +
            T(APICAL_INHIBITORY_GAIN) * apical_inhibition +
            T(ADAPTATION_INHIBITORY_GAIN) * adaptation +
            T(SOMA_EVIDENCE_GAIN) * margin_negative

        packet[packet_lane(group, FAST_FIELD)] =
            _bounded_positive(fast_raw)
        packet[packet_lane(group, NMDA_FIELD)] =
            _bounded_positive(slow_raw)
        packet[packet_lane(group, INHIBITORY_FIELD)] =
            _bounded_positive(inhibitory_raw)
    end
    return packet
end

"""
Exact continuous packet-map pullback.

`dnext_state` is overwritten.  The returned scalar is the cotangent of the
uncompanded pre-reset soma margin and must be passed as
`direct_margin_cotangent` to `cell_step_conditional_pullback!`.  Hard events
are deliberately absent from this VJP.
"""
function axon_packet_pullback!(
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

    apical_fast, apical_slow, apical_inhibitory,
        apical_excitation_raw, apical_excitation, apical_inhibition,
        adaptation, margin_normalized, margin_positive, margin_negative =
        _global_evidence(previous_state, next_state, cache)

    d_apical_excitation = zero(T)
    d_apical_inhibition = zero(T)
    d_adaptation = zero(T)
    d_margin_positive = zero(T)
    d_margin_negative = zero(T)

    @inbounds for group in 1:GROUP_COUNT
        first_branch = 2 * group - 1
        second_branch = first_branch + 1
        fast_first, slow_first, inhibitory_first =
            _branch_evidence(next_state, first_branch)
        fast_second, slow_second, inhibitory_second =
            _branch_evidence(next_state, second_branch)
        pair_fast = T(0.5) * (fast_first + fast_second)
        pair_slow = T(0.5) * (slow_first + slow_second)
        pair_inhibitory = T(0.5) * (inhibitory_first + inhibitory_second)

        fast_multiplier = one(T) +
                          T(APICAL_FAST_GAIN) * apical_excitation
        slow_multiplier = one(T) +
                          T(APICAL_NMDA_GAIN) * apical_excitation
        fast_raw = pair_fast * fast_multiplier +
                   T(SOMA_EVIDENCE_GAIN) * margin_positive
        slow_raw = pair_slow * slow_multiplier
        inhibitory_raw = pair_inhibitory +
            T(APICAL_INHIBITORY_GAIN) * apical_inhibition +
            T(ADAPTATION_INHIBITORY_GAIN) * adaptation +
            T(SOMA_EVIDENCE_GAIN) * margin_negative

        d_fast_raw = packet_bar[packet_lane(group, FAST_FIELD)] *
                     _bounded_positive_derivative(fast_raw)
        d_slow_raw = packet_bar[packet_lane(group, NMDA_FIELD)] *
                     _bounded_positive_derivative(slow_raw)
        d_inhibitory_raw =
            packet_bar[packet_lane(group, INHIBITORY_FIELD)] *
            _bounded_positive_derivative(inhibitory_raw)

        d_pair_fast = d_fast_raw * fast_multiplier
        d_pair_slow = d_slow_raw * slow_multiplier
        d_pair_inhibitory = d_inhibitory_raw
        d_apical_excitation +=
            d_fast_raw * pair_fast * T(APICAL_FAST_GAIN) +
            d_slow_raw * pair_slow * T(APICAL_NMDA_GAIN)
        d_apical_inhibition +=
            d_inhibitory_raw * T(APICAL_INHIBITORY_GAIN)
        d_adaptation +=
            d_inhibitory_raw * T(ADAPTATION_INHIBITORY_GAIN)
        d_margin_positive += d_fast_raw * T(SOMA_EVIDENCE_GAIN)
        d_margin_negative += d_inhibitory_raw * T(SOMA_EVIDENCE_GAIN)

        _accumulate_branch_pullback!(
            dnext_state,
            next_state,
            first_branch,
            T(0.5) * d_pair_fast,
            T(0.5) * d_pair_slow,
            T(0.5) * d_pair_inhibitory,
        )
        _accumulate_branch_pullback!(
            dnext_state,
            next_state,
            second_branch,
            T(0.5) * d_pair_fast,
            T(0.5) * d_pair_slow,
            T(0.5) * d_pair_inhibitory,
        )
    end

    d_apical_excitation_raw = d_apical_excitation *
        _bounded_positive_derivative(apical_excitation_raw)
    d_apical_fast = T(0.5) * d_apical_excitation_raw
    d_apical_slow = T(0.5) * d_apical_excitation_raw
    d_apical_inhibitory = d_apical_inhibition *
        _bounded_positive_derivative(apical_inhibitory)
    _accumulate_branch_pullback!(
        dnext_state,
        next_state,
        Cell.N_COMPARTMENTS,
        d_apical_fast,
        d_apical_slow,
        d_apical_inhibitory,
    )
    adaptation_raw = next_state[Cell.ADAPTATION_INDEX]
    dnext_state[Cell.ADAPTATION_INDEX] +=
        d_adaptation * _positive_derivative(adaptation_raw) /
        T(ADAPTATION_SCALE)
    dnext_state[Cell.SPIKE_INDEX] = zero(T)

    return (
        d_margin_positive * _smooth_positive_derivative(margin_normalized) -
        d_margin_negative * _smooth_positive_derivative(-margin_normalized)
    ) / T(MARGIN_SCALE)
end

"""
Write the hard control plane: soma spike followed by four plateau-group
onset/offset bits.  A group event is an XOR of its previous and next active
states; stable activity does not repeatedly enqueue work.
"""
function hard_events!(
    events::AbstractVector,
    previous_state::AbstractVector{T},
    next_state::AbstractVector{T},
) where {T<:AbstractFloat}
    _check_events(events)
    _check_state(previous_state)
    _check_state(next_state)
    one_event = one(eltype(events))
    zero_event = zero(eltype(events))
    events[SOMA_EVENT] = next_state[Cell.SPIKE_INDEX] >= T(0.5) ?
                         one_event : zero_event
    threshold = T(PLATEAU_EVENT_THRESHOLD)
    @inbounds for group in 1:GROUP_COUNT
        first_branch = 2 * group - 1
        second_branch = first_branch + 1
        previous_active =
            previous_state[Cell.state_index(first_branch, Cell.FIELD_PLATEAU)] >= threshold ||
            previous_state[Cell.state_index(second_branch, Cell.FIELD_PLATEAU)] >= threshold
        next_active =
            next_state[Cell.state_index(first_branch, Cell.FIELD_PLATEAU)] >= threshold ||
            next_state[Cell.state_index(second_branch, Cell.FIELD_PLATEAU)] >= threshold
        events[plateau_event_lane(group)] = previous_active != next_active ?
                                              one_event : zero_event
    end
    return events
end

"""
Add an ordered binary merge to one 27-channel Reduced-Hay inbox.

The left packet occupies basal compartments 1:4, the right packet basal
compartments 5:8, and the three caller-supplied context channels occupy the
apical compartment.  No two children ever share an input bin.
"""
function ordered_binary_deposit!(
    destination::AbstractVector{T},
    left::AbstractVector{T},
    right::AbstractVector{T},
    apical_context::AbstractVector{T},
) where {T<:AbstractFloat}
    length(destination) == Cell.INPUT_DIM || throw(DimensionMismatch(
        "ordered merge destination must have $(Cell.INPUT_DIM) channels",
    ))
    _check_packet(left)
    _check_packet(right)
    length(apical_context) == Cell.INPUT_CHANNELS || throw(DimensionMismatch(
        "apical context must have $(Cell.INPUT_CHANNELS) channels",
    ))
    @inbounds for group in 1:GROUP_COUNT, field in 1:FIELD_COUNT
        destination[Cell.input_index(group, field)] +=
            left[packet_lane(group, field)]
        destination[Cell.input_index(GROUP_COUNT + group, field)] +=
            right[packet_lane(group, field)]
    end
    @inbounds for channel in 1:Cell.INPUT_CHANNELS
        destination[Cell.input_index(Cell.N_COMPARTMENTS, channel)] +=
            apical_context[channel]
    end
    return destination
end

"""Exact additive pullback of [`ordered_binary_deposit!`](@ref)."""
function ordered_binary_deposit_pullback!(
    left_bar::AbstractVector{T},
    right_bar::AbstractVector{T},
    apical_context_bar::AbstractVector{T},
    destination_bar::AbstractVector{T},
) where {T<:AbstractFloat}
    _check_packet(left_bar)
    _check_packet(right_bar)
    length(apical_context_bar) == Cell.INPUT_CHANNELS ||
        throw(DimensionMismatch(
            "apical context cotangent must have $(Cell.INPUT_CHANNELS) channels",
        ))
    length(destination_bar) == Cell.INPUT_DIM || throw(DimensionMismatch(
        "ordered merge cotangent must have $(Cell.INPUT_DIM) channels",
    ))
    @inbounds for group in 1:GROUP_COUNT, field in 1:FIELD_COUNT
        left_bar[packet_lane(group, field)] +=
            destination_bar[Cell.input_index(group, field)]
        right_bar[packet_lane(group, field)] +=
            destination_bar[Cell.input_index(GROUP_COUNT + group, field)]
    end
    @inbounds for channel in 1:Cell.INPUT_CHANNELS
        apical_context_bar[channel] +=
            destination_bar[Cell.input_index(Cell.N_COMPARTMENTS, channel)]
    end
    return left_bar, right_bar, apical_context_bar
end

@assert Cell.N_BASAL == 2 * GROUP_COUNT
@assert Cell.INPUT_DIM == 2 * PACKET_DIM + Cell.INPUT_CHANNELS
@assert PACKET_DIM == 12
@assert PACKET_BYTES == 48
@assert EVENT_DIM == 5

end # module DendriticAxonPacket
