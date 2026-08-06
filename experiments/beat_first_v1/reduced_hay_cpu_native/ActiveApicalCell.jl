module ActiveApicalCell

using LinearAlgebra

# Canonical state order is compartment-major:
#   [V, g_AMPA, g_NMDA, g_GABA, plateau] for every basal compartment,
#   the same five states for the final active apical compartment,
#   then [soma_voltage, adaptation, hard_spike].
# Inputs are [AMPA, NMDA, GABA] for each compartment in the same order.

export N_BASAL,
       N_COMPARTMENTS,
       COMPARTMENT_STATE_DIM,
       STATE_DIM,
       INPUT_CHANNELS,
       INPUT_DIM,
       PARAM_DIM,
       SPIKE_SURROGATE_WIDTH,
       FIELD_VOLTAGE,
       FIELD_AMPA,
       FIELD_NMDA,
       FIELD_GABA,
       FIELD_PLATEAU,
       INPUT_AMPA,
       INPUT_NMDA,
       INPUT_GABA,
       SOMA_INDEX,
       ADAPTATION_INDEX,
       SPIKE_INDEX,
       PARAMETER_NAMES,
       CellParameterCache,
       CellParameterDerivativeCache,
       state_index,
       input_index,
       default_raw_parameters,
       transform_parameters,
       transform_parameter_derivatives,
       parameter_caches,
       initial_state,
       initial_state!,
       initial_state_pullback!,
       spike_margin_from_transition,
       spike_surrogate_value,
       spike_surrogate_derivative,
       cell_step_functional,
       cell_step_cached_functional,
       cell_step!,
       cell_step_conditional_pullback!,
       cell_step_pullback!

const N_BASAL = 8
const N_COMPARTMENTS = N_BASAL + 1
const COMPARTMENT_STATE_DIM = 5
const STATE_DIM = N_COMPARTMENTS * COMPARTMENT_STATE_DIM + 3
const INPUT_CHANNELS = 3
const INPUT_DIM = N_COMPARTMENTS * INPUT_CHANNELS

const FIELD_VOLTAGE = 1
const FIELD_AMPA = 2
const FIELD_NMDA = 3
const FIELD_GABA = 4
const FIELD_PLATEAU = 5

const INPUT_AMPA = 1
const INPUT_NMDA = 2
const INPUT_GABA = 3

const SOMA_INDEX = N_COMPARTMENTS * COMPARTMENT_STATE_DIM + 1
const ADAPTATION_INDEX = SOMA_INDEX + 1
const SPIKE_INDEX = ADAPTATION_INDEX + 1
const SPIKE_SURROGATE_WIDTH = 5.0f0
const AMPA_INPUT_SCALE = 1.0f0
const NMDA_INPUT_SCALE = 1.0f0
const GABA_INPUT_SCALE = 1.0f0

@inline function state_index(compartment::Integer, field::Integer)
    1 <= compartment <= N_COMPARTMENTS || throw(BoundsError(1:N_COMPARTMENTS, compartment))
    1 <= field <= COMPARTMENT_STATE_DIM || throw(BoundsError(1:COMPARTMENT_STATE_DIM, field))
    return (compartment - 1) * COMPARTMENT_STATE_DIM + field
end

@inline function input_index(compartment::Integer, channel::Integer)
    1 <= compartment <= N_COMPARTMENTS || throw(BoundsError(1:N_COMPARTMENTS, compartment))
    1 <= channel <= INPUT_CHANNELS || throw(BoundsError(1:INPUT_CHANNELS, channel))
    return (compartment - 1) * INPUT_CHANNELS + channel
end

const CORE_PARAMETER_NAMES = (
    :basal_dt,
    :apical_dt,
    :ampa_decay,
    :nmda_decay,
    :gaba_decay,
    :plateau_decay,
    :ampa_max,
    :nmda_max,
    :gaba_max,
    :compartment_rest,
    :excitatory_reversal,
    :inhibitory_gap,
    :nmda_half_voltage,
    :nmda_slope,
    :plateau_threshold,
    :plateau_slope,
    :plateau_gain,
    :plateau_current,
    :bap_gain,
    :basal_to_soma,
    :apical_to_soma,
    :apical_modulation,
    :signal_scale,
    :soma_dt,
    :soma_rest,
    :soma_threshold_gap,
    :soma_reset_gap,
    :adaptation_decay,
    :adaptation_gain,
    :adaptation_coupling,
)

const PARAMETER_NAMES = (
    CORE_PARAMETER_NAMES...,
    ntuple(index -> Symbol("basal_dt_multiplier_", index), Val(N_BASAL))...,
    ntuple(index -> Symbol("basal_role_", index), Val(N_BASAL))...,
)

const PARAM_DIM = length(PARAMETER_NAMES)

const P_BASAL_DT = 1
const P_APICAL_DT = 2
const P_AMPA_DECAY = 3
const P_NMDA_DECAY = 4
const P_GABA_DECAY = 5
const P_PLATEAU_DECAY = 6
const P_AMPA_MAX = 7
const P_NMDA_MAX = 8
const P_GABA_MAX = 9
const P_COMPARTMENT_REST = 10
const P_EXCITATORY_REVERSAL = 11
const P_INHIBITORY_GAP = 12
const P_NMDA_HALF_VOLTAGE = 13
const P_NMDA_SLOPE = 14
const P_PLATEAU_THRESHOLD = 15
const P_PLATEAU_SLOPE = 16
const P_PLATEAU_GAIN = 17
const P_PLATEAU_CURRENT = 18
const P_BAP_GAIN = 19
const P_BASAL_TO_SOMA = 20
const P_APICAL_TO_SOMA = 21
const P_APICAL_MODULATION = 22
const P_SIGNAL_SCALE = 23
const P_SOMA_DT = 24
const P_SOMA_REST = 25
const P_SOMA_THRESHOLD_GAP = 26
const P_SOMA_RESET_GAP = 27
const P_ADAPTATION_DECAY = 28
const P_ADAPTATION_GAIN = 29
const P_ADAPTATION_COUPLING = 30
const FIRST_BASAL_DT_MULTIPLIER = length(CORE_PARAMETER_NAMES) + 1
const FIRST_BASAL_ROLE = FIRST_BASAL_DT_MULTIPLIER + N_BASAL

const PARAMETER_LOWER = (
    0.001f0, 0.001f0,
    0.01f0, 0.01f0, 0.01f0, 0.01f0,
    0.0f0, 0.0f0, 0.0f0,
    -80.0f0, -10.0f0, 1.0f0,
    -70.0f0, 1.0f0,
    -70.0f0, 1.0f0,
    0.0f0, 0.0f0, 0.0f0,
    0.0f0, 0.0f0, 0.0f0, 1.0f0,
    0.001f0, -80.0f0, 1.0f0, 1.0f0,
    0.01f0, 0.0f0, 0.0f0,
    ntuple(_ -> 0.5f0, Val(N_BASAL))...,
    ntuple(_ -> 0.05f0, Val(N_BASAL))...,
)

const PARAMETER_UPPER = (
    0.3f0, 0.5f0,
    0.999f0, 0.9995f0, 0.999f0, 0.9995f0,
    2.0f0, 2.0f0, 2.0f0,
    -50.0f0, 10.0f0, 40.0f0,
    -10.0f0, 30.0f0,
    -10.0f0, 30.0f0,
    1.0f0, 30.0f0, 30.0f0,
    30.0f0, 30.0f0, 0.95f0, 40.0f0,
    0.5f0, -50.0f0, 40.0f0, 40.0f0,
    0.9995f0, 20.0f0, 10.0f0,
    ntuple(_ -> 1.5f0, Val(N_BASAL))...,
    ntuple(_ -> 1.0f0, Val(N_BASAL))...,
)

const DEFAULT_PARAMETER_VALUES = (
    0.25f0, 0.25f0,
    0.25f0, 0.60f0, 0.50f0, 0.65f0,
    0.80f0, 0.60f0, 0.20f0,
    -65.0f0, 0.0f0, 13.0f0,
    -62.0f0, 5.0f0,
    -63.0f0, 3.0f0,
    0.60f0, 8.0f0, 6.0f0,
    16.0f0, 8.0f0, 0.5f0, 4.0f0,
    0.40f0, -65.0f0, 3.75f0, 2.50f0,
    0.70f0, 2.0f0, 1.5f0,
    ntuple(_ -> 1.0f0, Val(N_BASAL))...,
    # Branches 1--4 carry the direct sensory map.  Recurrent-only branches
    # start weak but remain trainable, so added dendritic capacity does not
    # dilute the established sensory drive at initialization.
    ntuple(compartment -> compartment <= 4 ? 0.5f0 : 0.1f0, Val(N_BASAL))...,
)

struct CellParameterCache{T}
    basal_dt::T
    apical_dt::T
    ampa_decay::T
    nmda_decay::T
    gaba_decay::T
    plateau_decay::T
    ampa_max::T
    nmda_max::T
    gaba_max::T
    compartment_rest::T
    excitatory_reversal::T
    inhibitory_reversal::T
    nmda_half_voltage::T
    inv_nmda_slope::T
    plateau_threshold::T
    inv_plateau_slope::T
    plateau_gain::T
    plateau_current::T
    bap_gain::T
    basal_to_soma::T
    apical_to_soma::T
    apical_modulation::T
    inv_signal_scale::T
    soma_decay::T
    soma_rest::T
    soma_threshold::T
    soma_reset::T
    adaptation_decay::T
    adaptation_gain::T
    adaptation_coupling::T
    basal_dt_multiplier::NTuple{N_BASAL,T}
    basal_role::NTuple{N_BASAL,T}
end

"""
Raw-parameter transform derivatives cached once per optimizer step.  The
diagonal tuple covers independent bounded transforms.  `basal_role_jacobian`
is the 4x4 Jacobian (column-major) of the normalized basal role weights.
"""
const INDEPENDENT_PARAMETER_DIM = FIRST_BASAL_ROLE - 1

struct CellParameterDerivativeCache{T}
    diagonal::NTuple{INDEPENDENT_PARAMETER_DIM,T}
    basal_role_jacobian::NTuple{N_BASAL * N_BASAL,T}
end

@inline _sigmoid(x) = inv(one(x) + exp(-x))
@inline _positive(x) = max(x, zero(x))
@inline _unit_interval(x) = clamp(x, zero(x), one(x))
@inline _hard_event(x) = ifelse(x >= oftype(x, 0.5), one(x), zero(x))
@inline _threshold_event(x) = ifelse(x >= zero(x), one(x), zero(x))

@inline function _soft_threshold_event(margin::T, width::T) where {T<:Real}
    margin <= -width && return zero(T)
    margin >= width && return one(T)
    if margin < zero(T)
        shifted = margin + width
        return shifted * shifted / (T(2) * width * width)
    end
    shifted = width - margin
    return one(T) - shifted * shifted / (T(2) * width * width)
end

"""Continuous event value whose derivative is the canonical spike surrogate."""
@inline spike_surrogate_value(margin::T) where {T<:Real} =
    spike_surrogate_value(margin, oftype(margin, SPIKE_SURROGATE_WIDTH))

@inline function spike_surrogate_value(margin::T, width::T) where {T<:Real}
    width > zero(T) || throw(ArgumentError("surrogate width must be positive"))
    return _soft_threshold_event(margin, width)
end

@inline function _smoothed_threshold_event(
    margin::T,
    smoothing::T,
) where {T<:Real}
    hard = _threshold_event(margin)
    iszero(smoothing) && return hard
    soft = _soft_threshold_event(margin, oftype(margin, SPIKE_SURROGATE_WIDTH))
    return muladd(smoothing, soft - hard, hard)
end

@inline function _smoothed_previous_event(
    spike::T,
    smoothing::T,
) where {T<:Real}
    hard = _hard_event(spike)
    iszero(smoothing) && return hard
    return muladd(smoothing, _unit_interval(spike) - hard, hard)
end

"""
    spike_surrogate_derivative(margin[, width])

Triangular surrogate derivative for the hard-spike margin
`soma_pre_reset - threshold`.  The canonical pullback uses it exactly once
after aggregating all event cotangents.
"""
@inline spike_surrogate_derivative(margin::T) where {T<:Real} =
    spike_surrogate_derivative(margin, oftype(margin, SPIKE_SURROGATE_WIDTH))

@inline function spike_surrogate_derivative(margin::T, width::T) where {T<:Real}
    width > zero(T) || throw(ArgumentError("surrogate width must be positive"))
    distance = abs(margin) / width
    return ifelse(distance < one(T), (one(T) - distance) / width, zero(T))
end

@inline function _bounded(raw, lower, upper)
    lo = oftype(raw, lower)
    hi = oftype(raw, upper)
    return lo + (hi - lo) * _sigmoid(raw)
end

@inline function _conductance_drive(input_value, maximum, scale::Float32)
    positive_input = _positive(input_value)
    return maximum * (one(input_value) - exp(-positive_input / oftype(input_value, scale)))
end

function default_raw_parameters(::Type{T}=Float32) where {T<:AbstractFloat}
    raw = Vector{T}(undef, PARAM_DIM)
    @inbounds for i in 1:PARAM_DIM
        lo = T(PARAMETER_LOWER[i])
        hi = T(PARAMETER_UPPER[i])
        value = T(DEFAULT_PARAMETER_VALUES[i])
        probability = clamp((value - lo) / (hi - lo), eps(T), one(T) - eps(T))
        raw[i] = log(probability / (one(T) - probability))
    end
    return raw
end

function transform_parameters(raw::AbstractVector{T}) where {T<:Real}
    length(raw) == PARAM_DIM || throw(DimensionMismatch("expected $PARAM_DIM raw cell parameters"))
    bounded(i) = _bounded(raw[i], PARAMETER_LOWER[i], PARAMETER_UPPER[i])
    role_strength = ntuple(Val(N_BASAL)) do compartment
        bounded(FIRST_BASAL_ROLE + compartment - 1)
    end
    role_total = sum(role_strength)
    compartment_rest = bounded(P_COMPARTMENT_REST)
    inhibitory_gap = bounded(P_INHIBITORY_GAP)
    inhibitory_reversal = compartment_rest - inhibitory_gap
    nmda_slope = bounded(P_NMDA_SLOPE)
    plateau_slope = bounded(P_PLATEAU_SLOPE)
    signal_scale = bounded(P_SIGNAL_SCALE)
    soma_dt = bounded(P_SOMA_DT)
    soma_rest = bounded(P_SOMA_REST)
    soma_threshold_gap = bounded(P_SOMA_THRESHOLD_GAP)
    soma_threshold = soma_rest + soma_threshold_gap
    soma_reset_gap = bounded(P_SOMA_RESET_GAP)
    soma_reset = soma_threshold - soma_reset_gap
    return CellParameterCache(
        bounded(P_BASAL_DT),
        bounded(P_APICAL_DT),
        bounded(P_AMPA_DECAY),
        bounded(P_NMDA_DECAY),
        bounded(P_GABA_DECAY),
        bounded(P_PLATEAU_DECAY),
        bounded(P_AMPA_MAX),
        bounded(P_NMDA_MAX),
        bounded(P_GABA_MAX),
        compartment_rest,
        bounded(P_EXCITATORY_REVERSAL),
        inhibitory_reversal,
        bounded(P_NMDA_HALF_VOLTAGE),
        inv(nmda_slope),
        bounded(P_PLATEAU_THRESHOLD),
        inv(plateau_slope),
        bounded(P_PLATEAU_GAIN),
        bounded(P_PLATEAU_CURRENT),
        bounded(P_BAP_GAIN),
        bounded(P_BASAL_TO_SOMA),
        bounded(P_APICAL_TO_SOMA),
        bounded(P_APICAL_MODULATION),
        inv(signal_scale),
        exp(-soma_dt),
        soma_rest,
        soma_threshold,
        soma_reset,
        bounded(P_ADAPTATION_DECAY),
        bounded(P_ADAPTATION_GAIN),
        bounded(P_ADAPTATION_COUPLING),
        ntuple(
            compartment -> bounded(
                FIRST_BASAL_DT_MULTIPLIER + compartment - 1,
            ),
            Val(N_BASAL),
        ),
        ntuple(
            compartment -> role_strength[compartment] / role_total,
            Val(N_BASAL),
        ),
    )
end

@inline function _bounded_derivative(raw, index::Int)
    probability = _sigmoid(raw[index])
    return (oftype(probability, PARAMETER_UPPER[index]) -
            oftype(probability, PARAMETER_LOWER[index])) * probability * (one(probability) - probability)
end

function transform_parameter_derivatives(raw::AbstractVector{T}) where {T<:Real}
    length(raw) == PARAM_DIM || throw(DimensionMismatch("expected $PARAM_DIM raw cell parameters"))
    diagonal = ntuple(i -> _bounded_derivative(raw, i), Val(INDEPENDENT_PARAMETER_DIM))
    strengths = ntuple(Val(N_BASAL)) do compartment
        index = FIRST_BASAL_ROLE + compartment - 1
        _bounded(raw[index], PARAMETER_LOWER[index], PARAMETER_UPPER[index])
    end
    total = sum(strengths)
    role_jacobian = ntuple(Val(N_BASAL * N_BASAL)) do linear_index
        output_role = (linear_index - 1) % N_BASAL + 1
        raw_role = (linear_index - 1) ÷ N_BASAL + 1
        numerator = ifelse(output_role == raw_role, total, zero(T)) - strengths[output_role]
        return numerator / (total * total) *
               _bounded_derivative(raw, FIRST_BASAL_ROLE + raw_role - 1)
    end
    return CellParameterDerivativeCache(diagonal, role_jacobian)
end

parameter_caches(raw::AbstractVector{T}) where {T<:Real} =
    (transform_parameters(raw), transform_parameter_derivatives(raw))

@inline function _basal_dt_multiplier(cache::CellParameterCache, compartment::Int)
    return cache.basal_dt_multiplier[compartment]
end

@inline function _basal_role(cache::CellParameterCache, compartment::Int)
    return cache.basal_role[compartment]
end

function initial_state!(
    destination::AbstractVector{T},
    cache::CellParameterCache{T},
) where {T<:Real}
    length(destination) == STATE_DIM ||
        throw(DimensionMismatch("expected $STATE_DIM destination states"))
    fill!(destination, zero(T))
    @inbounds for compartment in 1:N_COMPARTMENTS
        destination[state_index(compartment, FIELD_VOLTAGE)] = cache.compartment_rest
    end
    @inbounds destination[SOMA_INDEX] = cache.soma_rest
    return destination
end

function initial_state(cache::CellParameterCache{T}) where {T<:Real}
    destination = Vector{T}(undef, STATE_DIM)
    return initial_state!(destination, cache)
end

initial_state(raw::AbstractVector{T}) where {T<:Real} = initial_state(transform_parameters(raw))

"""
    initial_state_pullback!(draw, dinitial, derivative_cache)

Accumulate the cotangent of the canonical resting initial state into raw cell
parameter cotangents.  Call this once at the root of a trajectory after all
cell-step pullbacks.  Only `compartment_rest` (five voltage coordinates) and
`soma_rest` affect initialization; every other raw parameter receives no
initial-state contribution.
"""
function initial_state_pullback!(
    draw::AbstractVector{T},
    dinitial::AbstractVector{T},
    derivative_cache::CellParameterDerivativeCache{T},
) where {T<:AbstractFloat}
    length(draw) == PARAM_DIM || throw(DimensionMismatch("expected $PARAM_DIM parameter cotangents"))
    length(dinitial) == STATE_DIM || throw(DimensionMismatch("expected $STATE_DIM initial-state cotangents"))
    compartment_rest_cotangent = zero(T)
    @inbounds for compartment in 1:N_COMPARTMENTS
        compartment_rest_cotangent += dinitial[state_index(compartment, FIELD_VOLTAGE)]
    end
    @inbounds begin
        draw[P_COMPARTMENT_REST] += compartment_rest_cotangent *
                                    derivative_cache.diagonal[P_COMPARTMENT_REST]
        draw[P_SOMA_REST] += dinitial[SOMA_INDEX] *
                             derivative_cache.diagonal[P_SOMA_REST]
    end
    return draw
end

@inline function _compartment_step(
    voltage,
    ampa,
    nmda,
    gaba,
    plateau,
    ampa_input,
    nmda_input,
    gaba_input,
    previous_spike,
    cache::CellParameterCache,
    compartment::Int,
    spike_smoothing=zero(previous_spike),
)
    is_apical = compartment == N_COMPARTMENTS
    ampa_drive = _conductance_drive(ampa_input, cache.ampa_max, AMPA_INPUT_SCALE)
    nmda_drive = _conductance_drive(nmda_input, cache.nmda_max, NMDA_INPUT_SCALE)
    gaba_drive = _conductance_drive(gaba_input, cache.gaba_max, GABA_INPUT_SCALE)
    ampa_next = cache.ampa_decay * _positive(ampa) +
                (one(ampa) - cache.ampa_decay) * ampa_drive
    nmda_next = cache.nmda_decay * _positive(nmda) +
                (one(nmda) - cache.nmda_decay) * nmda_drive
    gaba_next = cache.gaba_decay * _positive(gaba) +
                (one(gaba) - cache.gaba_decay) * gaba_drive

    nmda_unblock = _sigmoid((voltage - cache.nmda_half_voltage) * cache.inv_nmda_slope)
    excitation = ampa_next + nmda_next * nmda_unblock
    ligand_gate = one(excitation) - exp(-excitation)
    voltage_gate = _sigmoid((voltage - cache.plateau_threshold) * cache.inv_plateau_slope)
    plateau_target = cache.plateau_gain * voltage_gate * ligand_gate
    plateau_previous = _unit_interval(plateau)
    plateau_next = cache.plateau_decay * plateau_previous +
                   (one(plateau_previous) - cache.plateau_decay) * plateau_target

    excitatory_conductance = ampa_next + nmda_next * nmda_unblock
    inhibitory_conductance = gaba_next
    bap_drive = ifelse(
        is_apical,
        cache.bap_gain * _smoothed_previous_event(previous_spike, spike_smoothing),
        zero(voltage),
    )
    dt = is_apical ?
        cache.apical_dt :
        cache.basal_dt * _basal_dt_multiplier(cache, compartment)
    total_conductance = one(voltage) + excitatory_conductance + inhibitory_conductance
    equilibrium_voltage = (
        cache.compartment_rest +
        excitatory_conductance * cache.excitatory_reversal +
        inhibitory_conductance * cache.inhibitory_reversal +
        cache.plateau_current * plateau_next + bap_drive
    ) / total_conductance
    relaxation = one(voltage) - exp(-dt * total_conductance)
    voltage_next = voltage + relaxation * (equilibrium_voltage - voltage)
    return (voltage_next, ampa_next, nmda_next, gaba_next, plateau_next)
end

@inline function _compartment_signal(voltage, cache::CellParameterCache)
    return tanh((voltage - cache.compartment_rest) * cache.inv_signal_scale)
end

function cell_step_cached_functional(
    state::AbstractVector{T},
    input::AbstractVector{T},
    cache::CellParameterCache,
    spike_smoothing::T=zero(T),
) where {T<:Real}
    length(state) == STATE_DIM || throw(DimensionMismatch("expected $STATE_DIM cell states"))
    length(input) == INPUT_DIM || throw(DimensionMismatch("expected $INPUT_DIM cell inputs"))

    previous_spike = state[SPIKE_INDEX]
    next_compartments = ntuple(Val(N_COMPARTMENTS)) do compartment
        base = (compartment - 1) * COMPARTMENT_STATE_DIM
        input_base = (compartment - 1) * INPUT_CHANNELS
        _compartment_step(
            state[base + FIELD_VOLTAGE],
            state[base + FIELD_AMPA],
            state[base + FIELD_NMDA],
            state[base + FIELD_GABA],
            state[base + FIELD_PLATEAU],
            input[input_base + INPUT_AMPA],
            input[input_base + INPUT_NMDA],
            input[input_base + INPUT_GABA],
            previous_spike,
            cache,
            compartment,
            spike_smoothing,
        )
    end

    basal_signal = zero(T)
    @inbounds for compartment in 1:N_BASAL
        signal = _compartment_signal(
            next_compartments[compartment][FIELD_VOLTAGE],
            cache,
        )
        basal_signal = muladd(_basal_role(cache, compartment), signal, basal_signal)
    end
    apical_signal = _compartment_signal(
        next_compartments[N_COMPARTMENTS][FIELD_VOLTAGE],
        cache,
    )
    modulation = one(apical_signal) + cache.apical_modulation * apical_signal
    soma_drive = cache.basal_to_soma * basal_signal * modulation +
                 cache.apical_to_soma * apical_signal

    soma_previous = state[SOMA_INDEX]
    adaptation_previous = _positive(state[ADAPTATION_INDEX])
    soma_target = cache.soma_rest + soma_drive -
                  cache.adaptation_coupling * adaptation_previous
    soma_relaxation = one(soma_previous) - cache.soma_decay
    soma_pre_reset = soma_previous + soma_relaxation * (soma_target - soma_previous)
    spike_next = _smoothed_threshold_event(
        soma_pre_reset - cache.soma_threshold,
        spike_smoothing,
    )
    soma_next = muladd(
        spike_next,
        cache.soma_reset - soma_pre_reset,
        soma_pre_reset,
    )
    adaptation_target = cache.adaptation_gain * spike_next
    adaptation_next = cache.adaptation_decay * adaptation_previous +
                      (one(adaptation_previous) - cache.adaptation_decay) * adaptation_target

    values = ntuple(Val(STATE_DIM)) do index
        if index <= N_COMPARTMENTS * COMPARTMENT_STATE_DIM
            compartment, field_zero = divrem(index - 1, COMPARTMENT_STATE_DIM)
            return next_compartments[compartment + 1][field_zero + 1]
        end
        index == SOMA_INDEX && return soma_next
        index == ADAPTATION_INDEX && return adaptation_next
        return spike_next
    end
    return [values...]
end

"""
Reconstruct the exact pre-reset soma margin from one recorded transition.

Only compartment coordinates from `next_state` are read; soma reset and the
hard spike therefore cannot erase the decision variable used by a local
eligibility rule.
"""
@inline function spike_margin_from_transition(
    state::AbstractVector{T},
    next_state::AbstractVector{T},
    cache::CellParameterCache{T},
) where {T<:AbstractFloat}
    length(state) == STATE_DIM || throw(DimensionMismatch(
        "expected $STATE_DIM previous cell states",
    ))
    length(next_state) == STATE_DIM || throw(DimensionMismatch(
        "expected $STATE_DIM next cell states",
    ))
    basal_signal = zero(T)
    @inbounds for compartment in 1:N_BASAL
        basal_signal = muladd(
            _basal_role(cache, compartment),
            _compartment_signal(
                next_state[state_index(compartment, FIELD_VOLTAGE)],
                cache,
            ),
            basal_signal,
        )
    end
    apical_signal = _compartment_signal(
        next_state[state_index(N_COMPARTMENTS, FIELD_VOLTAGE)],
        cache,
    )
    modulation = one(T) + cache.apical_modulation * apical_signal
    soma_drive = cache.basal_to_soma * basal_signal * modulation +
                 cache.apical_to_soma * apical_signal
    soma_previous = @inbounds state[SOMA_INDEX]
    adaptation_previous = _positive(@inbounds state[ADAPTATION_INDEX])
    soma_target = cache.soma_rest + soma_drive -
                  cache.adaptation_coupling * adaptation_previous
    soma_pre_reset = soma_previous +
        (one(T) - cache.soma_decay) * (soma_target - soma_previous)
    return soma_pre_reset - cache.soma_threshold
end

function cell_step_functional(
    state::AbstractVector{T},
    input::AbstractVector{T},
    raw_parameters::AbstractVector{T},
) where {T<:Real}
    return cell_step_cached_functional(state, input, transform_parameters(raw_parameters))
end

function cell_step!(
    destination::AbstractVector{Float32},
    state::AbstractVector{Float32},
    input::AbstractVector{Float32},
    cache::CellParameterCache{Float32},
    spike_smoothing::Float32=0.0f0,
)
    length(destination) == STATE_DIM || throw(DimensionMismatch("expected $STATE_DIM destination states"))
    length(state) == STATE_DIM || throw(DimensionMismatch("expected $STATE_DIM cell states"))
    length(input) == INPUT_DIM || throw(DimensionMismatch("expected $INPUT_DIM cell inputs"))
    0.0f0 <= spike_smoothing <= 1.0f0 || throw(ArgumentError(
        "spike smoothing must be in [0, 1]",
    ))

    previous_spike = @inbounds state[SPIKE_INDEX]
    @inbounds for compartment in 1:N_COMPARTMENTS
        state_base = (compartment - 1) * COMPARTMENT_STATE_DIM
        input_base = (compartment - 1) * INPUT_CHANNELS
        # An exactly resting basal branch with no incoming conductance is a
        # fixed point of `_compartment_step`. Preserve that state directly so
        # inactive high-dimensional capacity costs only five comparisons and
        # copies, while any sensory or recurrent event immediately restores
        # the full Hay-derived dynamics. The apical compartment is excluded
        # because a previous soma event can drive it through bAP.
        dormant = compartment <= N_BASAL &&
                  state[state_base + FIELD_VOLTAGE] == cache.compartment_rest &&
                  state[state_base + FIELD_AMPA] == 0.0f0 &&
                  state[state_base + FIELD_NMDA] == 0.0f0 &&
                  state[state_base + FIELD_GABA] == 0.0f0 &&
                  state[state_base + FIELD_PLATEAU] == 0.0f0 &&
                  input[input_base + INPUT_AMPA] == 0.0f0 &&
                  input[input_base + INPUT_NMDA] == 0.0f0 &&
                  input[input_base + INPUT_GABA] == 0.0f0
        if dormant
            destination[state_base + FIELD_VOLTAGE] = cache.compartment_rest
            destination[state_base + FIELD_AMPA] = 0.0f0
            destination[state_base + FIELD_NMDA] = 0.0f0
            destination[state_base + FIELD_GABA] = 0.0f0
            destination[state_base + FIELD_PLATEAU] = 0.0f0
            continue
        end
        next_values = _compartment_step(
            state[state_base + FIELD_VOLTAGE],
            state[state_base + FIELD_AMPA],
            state[state_base + FIELD_NMDA],
            state[state_base + FIELD_GABA],
            state[state_base + FIELD_PLATEAU],
            input[input_base + INPUT_AMPA],
            input[input_base + INPUT_NMDA],
            input[input_base + INPUT_GABA],
            previous_spike,
            cache,
            compartment,
            spike_smoothing,
        )
        destination[state_base + FIELD_VOLTAGE] = next_values[FIELD_VOLTAGE]
        destination[state_base + FIELD_AMPA] = next_values[FIELD_AMPA]
        destination[state_base + FIELD_NMDA] = next_values[FIELD_NMDA]
        destination[state_base + FIELD_GABA] = next_values[FIELD_GABA]
        destination[state_base + FIELD_PLATEAU] = next_values[FIELD_PLATEAU]
    end

    @inbounds begin
        basal_signal = 0.0f0
        for compartment in 1:N_BASAL
            voltage = destination[state_index(compartment, FIELD_VOLTAGE)]
            voltage == cache.compartment_rest && continue
            signal = _compartment_signal(voltage, cache)
            basal_signal = muladd(
                _basal_role(cache, compartment),
                signal,
                basal_signal,
            )
        end
        apical_voltage = destination[state_index(N_COMPARTMENTS, FIELD_VOLTAGE)]
        apical_signal = apical_voltage == cache.compartment_rest ?
            0.0f0 : _compartment_signal(apical_voltage, cache)
        modulation = 1.0f0 + cache.apical_modulation * apical_signal
        soma_drive = cache.basal_to_soma * basal_signal * modulation +
                     cache.apical_to_soma * apical_signal
        soma_previous = state[SOMA_INDEX]
        adaptation_previous = _positive(state[ADAPTATION_INDEX])
        soma_target = cache.soma_rest + soma_drive -
                      cache.adaptation_coupling * adaptation_previous
        soma_relaxation = 1.0f0 - cache.soma_decay
        soma_pre_reset = soma_previous + soma_relaxation * (soma_target - soma_previous)
        spike_next = _smoothed_threshold_event(
            soma_pre_reset - cache.soma_threshold,
            spike_smoothing,
        )
        destination[SOMA_INDEX] = muladd(
            spike_next,
            cache.soma_reset - soma_pre_reset,
            soma_pre_reset,
        )
        adaptation_target = cache.adaptation_gain * spike_next
        destination[ADAPTATION_INDEX] = cache.adaptation_decay * adaptation_previous +
                                        (1.0f0 - cache.adaptation_decay) * adaptation_target
        destination[SPIKE_INDEX] = spike_next
    end
    return destination
end

function _cell_step_pullback!(
    dstate::AbstractVector{T},
    dinput::AbstractVector{T},
    draw::AbstractVector{T},
    state::AbstractVector{T},
    input::AbstractVector{T},
    cache::CellParameterCache{T},
    derivative_cache::CellParameterDerivativeCache{T},
    next_state::AbstractVector{T},
    dnext::AbstractVector{T},
    event_cotangent::T,
    spike_smoothing::T,
    direct_margin_cotangent::T,
    propagate_downstream_event_surrogate::Bool,
) where {T<:AbstractFloat}
    length(dstate) == STATE_DIM || throw(DimensionMismatch("expected $STATE_DIM state cotangents"))
    length(dinput) == INPUT_DIM || throw(DimensionMismatch("expected $INPUT_DIM input cotangents"))
    length(draw) == PARAM_DIM || throw(DimensionMismatch("expected $PARAM_DIM parameter cotangents"))
    length(state) == STATE_DIM || throw(DimensionMismatch("expected $STATE_DIM cell states"))
    length(input) == INPUT_DIM || throw(DimensionMismatch("expected $INPUT_DIM cell inputs"))
    length(next_state) == STATE_DIM || throw(DimensionMismatch("expected $STATE_DIM next states"))
    length(dnext) == STATE_DIM || throw(DimensionMismatch("expected $STATE_DIM next-state cotangents"))
    zero(T) <= spike_smoothing <= one(T) || throw(ArgumentError(
        "spike smoothing must be in [0, 1]",
    ))

    fill!(dstate, zero(T))
    fill!(dinput, zero(T))
    fill!(draw, zero(T))

    @inbounds begin
        basal_signal = zero(T)
        for compartment in 1:N_BASAL
            signal = _compartment_signal(
                next_state[state_index(compartment, FIELD_VOLTAGE)],
                cache,
            )
            basal_signal = muladd(
                _basal_role(cache, compartment),
                signal,
                basal_signal,
            )
        end
        apical_signal = _compartment_signal(
            next_state[state_index(N_COMPARTMENTS, FIELD_VOLTAGE)],
            cache,
        )
        modulation = one(T) + cache.apical_modulation * apical_signal
        soma_drive = cache.basal_to_soma * basal_signal * modulation +
                     cache.apical_to_soma * apical_signal

        soma_previous = state[SOMA_INDEX]
        adaptation_raw = state[ADAPTATION_INDEX]
        adaptation_previous = _positive(adaptation_raw)
        soma_target = cache.soma_rest + soma_drive -
                      cache.adaptation_coupling * adaptation_previous
        soma_relaxation = one(T) - cache.soma_decay
        soma_pre_reset = soma_previous + soma_relaxation * (soma_target - soma_previous)
        spike_value = next_state[SPIKE_INDEX]
        adaptation_target = cache.adaptation_gain * spike_value

        d_adaptation_next = dnext[ADAPTATION_INDEX]
        draw[P_ADAPTATION_DECAY] +=
            d_adaptation_next * (adaptation_previous - adaptation_target)
        d_adaptation_previous = d_adaptation_next * cache.adaptation_decay
        draw[P_ADAPTATION_GAIN] +=
            d_adaptation_next * (one(T) - cache.adaptation_decay) * spike_value

        d_soma_pre_reset = (one(T) - spike_value) * dnext[SOMA_INDEX]
        draw[P_SOMA_RESET_GAP] += spike_value * dnext[SOMA_INDEX]

        event_seed = event_cotangent
        if propagate_downstream_event_surrogate
            event_seed += dnext[SPIKE_INDEX] +
                (one(T) - cache.adaptation_decay) * cache.adaptation_gain *
                    dnext[ADAPTATION_INDEX] +
                (cache.soma_reset - soma_pre_reset) * dnext[SOMA_INDEX]
        end
        # `_soft_threshold_event` has exactly the same triangular derivative
        # used as the STE for the hard event.  Blending soft and hard forward
        # values therefore must not attenuate credit by `spike_smoothing`.
        # Keeping this derivative invariant also avoids a discontinuous jump
        # when an annealing schedule reaches the hard endpoint.
        d_margin = direct_margin_cotangent +
            event_seed * spike_surrogate_derivative(
                soma_pre_reset - cache.soma_threshold,
            )
        d_soma_pre_reset += d_margin
        draw[P_SOMA_THRESHOLD_GAP] -= d_margin

        d_soma_relaxation = d_soma_pre_reset * (soma_target - soma_previous)
        d_soma_target = d_soma_pre_reset * soma_relaxation
        dstate[SOMA_INDEX] += d_soma_pre_reset * (one(T) - soma_relaxation)
        draw[P_SOMA_DT] += d_soma_relaxation * cache.soma_decay
        draw[P_SOMA_REST] += d_soma_target
        d_soma_drive = d_soma_target
        draw[P_ADAPTATION_COUPLING] -= d_soma_target * adaptation_previous
        d_adaptation_previous -= d_soma_target * cache.adaptation_coupling
        if adaptation_raw > zero(T)
            dstate[ADAPTATION_INDEX] += d_adaptation_previous
        end

        draw[P_BASAL_TO_SOMA] += d_soma_drive * basal_signal * modulation
        d_basal_signal = d_soma_drive * cache.basal_to_soma * modulation
        for compartment in 1:N_BASAL
            signal = _compartment_signal(
                next_state[state_index(compartment, FIELD_VOLTAGE)],
                cache,
            )
            draw[FIRST_BASAL_ROLE + compartment - 1] +=
                d_basal_signal * signal
        end
        d_modulation = d_soma_drive * cache.basal_to_soma * basal_signal
        draw[P_APICAL_MODULATION] += d_modulation * apical_signal
        d_apical_signal = d_modulation * cache.apical_modulation
        draw[P_APICAL_TO_SOMA] += d_soma_drive * apical_signal
        d_apical_signal += d_soma_drive * cache.apical_to_soma

        previous_spike_event = _smoothed_previous_event(
            state[SPIKE_INDEX],
            spike_smoothing,
        )

        for compartment in 1:N_COMPARTMENTS
            state_base = (compartment - 1) * COMPARTMENT_STATE_DIM
            input_base = (compartment - 1) * INPUT_CHANNELS
            voltage = state[state_base + FIELD_VOLTAGE]
            ampa = state[state_base + FIELD_AMPA]
            nmda = state[state_base + FIELD_NMDA]
            gaba = state[state_base + FIELD_GABA]
            plateau = state[state_base + FIELD_PLATEAU]
            ampa_input = input[input_base + INPUT_AMPA]
            nmda_input = input[input_base + INPUT_NMDA]
            gaba_input = input[input_base + INPUT_GABA]

            ampa_positive = _positive(ampa)
            nmda_positive = _positive(nmda)
            gaba_positive = _positive(gaba)
            ampa_input_positive = _positive(ampa_input)
            nmda_input_positive = _positive(nmda_input)
            gaba_input_positive = _positive(gaba_input)
            ampa_input_decay = exp(-ampa_input_positive / T(AMPA_INPUT_SCALE))
            nmda_input_decay = exp(-nmda_input_positive / T(NMDA_INPUT_SCALE))
            gaba_input_decay = exp(-gaba_input_positive / T(GABA_INPUT_SCALE))
            ampa_drive = cache.ampa_max * (one(T) - ampa_input_decay)
            nmda_drive = cache.nmda_max * (one(T) - nmda_input_decay)
            gaba_drive = cache.gaba_max * (one(T) - gaba_input_decay)
            ampa_next = cache.ampa_decay * ampa_positive +
                        (one(T) - cache.ampa_decay) * ampa_drive
            nmda_next = cache.nmda_decay * nmda_positive +
                        (one(T) - cache.nmda_decay) * nmda_drive
            gaba_next = cache.gaba_decay * gaba_positive +
                        (one(T) - cache.gaba_decay) * gaba_drive

            nmda_argument = (voltage - cache.nmda_half_voltage) * cache.inv_nmda_slope
            nmda_unblock = _sigmoid(nmda_argument)
            excitatory_conductance = ampa_next + nmda_next * nmda_unblock
            inhibitory_conductance = gaba_next
            ligand_gate = one(T) - exp(-excitatory_conductance)
            voltage_argument = (voltage - cache.plateau_threshold) * cache.inv_plateau_slope
            voltage_gate = _sigmoid(voltage_argument)
            plateau_target = cache.plateau_gain * voltage_gate * ligand_gate
            plateau_previous = _unit_interval(plateau)
            plateau_next = cache.plateau_decay * plateau_previous +
                           (one(T) - cache.plateau_decay) * plateau_target

            is_apical = compartment == N_COMPARTMENTS
            bap_drive = ifelse(is_apical, cache.bap_gain * previous_spike_event, zero(T))
            dt_multiplier = is_apical ?
                one(T) : _basal_dt_multiplier(cache, compartment)
            dt = is_apical ? cache.apical_dt : cache.basal_dt * dt_multiplier
            total_conductance = one(T) + excitatory_conductance + inhibitory_conductance
            voltage_numerator = cache.compartment_rest +
                                excitatory_conductance * cache.excitatory_reversal +
                                inhibitory_conductance * cache.inhibitory_reversal +
                                cache.plateau_current * plateau_next + bap_drive
            equilibrium_voltage = voltage_numerator / total_conductance
            voltage_decay = exp(-dt * total_conductance)
            relaxation = one(T) - voltage_decay

            signal = _compartment_signal(
                next_state[state_index(compartment, FIELD_VOLTAGE)],
                cache,
            )
            d_signal = is_apical ?
                d_apical_signal :
                d_basal_signal * _basal_role(cache, compartment)
            d_signal_argument = d_signal * (one(T) - signal * signal)
            d_voltage_next = dnext[state_base + FIELD_VOLTAGE] +
                             d_signal_argument * cache.inv_signal_scale
            draw[P_COMPARTMENT_REST] -= d_signal_argument * cache.inv_signal_scale
            draw[P_SIGNAL_SCALE] -= d_signal_argument *
                                    (next_state[state_base + FIELD_VOLTAGE] - cache.compartment_rest) *
                                    cache.inv_signal_scale * cache.inv_signal_scale

            d_ampa_next = dnext[state_base + FIELD_AMPA]
            d_nmda_next = dnext[state_base + FIELD_NMDA]
            d_gaba_next = dnext[state_base + FIELD_GABA]
            d_plateau_next = dnext[state_base + FIELD_PLATEAU]

            d_voltage = d_voltage_next * (one(T) - relaxation)
            d_relaxation = d_voltage_next * (equilibrium_voltage - voltage)
            d_equilibrium_voltage = d_voltage_next * relaxation
            d_dt = d_relaxation * voltage_decay * total_conductance
            d_total_conductance = d_relaxation * voltage_decay * dt
            d_voltage_numerator = d_equilibrium_voltage / total_conductance
            d_total_conductance -=
                d_equilibrium_voltage * equilibrium_voltage / total_conductance

            if is_apical
                draw[P_APICAL_DT] += d_dt
            else
                draw[P_BASAL_DT] += d_dt * dt_multiplier
                draw[FIRST_BASAL_DT_MULTIPLIER + compartment - 1] += d_dt * cache.basal_dt
            end

            draw[P_COMPARTMENT_REST] += d_voltage_numerator
            d_excitatory_conductance = d_voltage_numerator * cache.excitatory_reversal +
                                       d_total_conductance
            draw[P_EXCITATORY_REVERSAL] +=
                d_voltage_numerator * excitatory_conductance
            d_inhibitory_conductance = d_voltage_numerator * cache.inhibitory_reversal +
                                       d_total_conductance
            draw[P_INHIBITORY_GAP] +=
                d_voltage_numerator * inhibitory_conductance
            d_plateau_next += d_voltage_numerator * cache.plateau_current
            draw[P_PLATEAU_CURRENT] += d_voltage_numerator * plateau_next
            if is_apical
                draw[P_BAP_GAIN] += d_voltage_numerator * previous_spike_event
                # The previous-event path uses the identity STE for both the
                # hard event and its unit-interval soft relaxation.
                dstate[SPIKE_INDEX] += d_voltage_numerator * cache.bap_gain
            end

            draw[P_PLATEAU_DECAY] +=
                d_plateau_next * (plateau_previous - plateau_target)
            d_plateau_previous = d_plateau_next * cache.plateau_decay
            d_plateau_target = d_plateau_next * (one(T) - cache.plateau_decay)
            draw[P_PLATEAU_GAIN] += d_plateau_target * voltage_gate * ligand_gate
            d_voltage_gate = d_plateau_target * cache.plateau_gain * ligand_gate
            d_ligand_gate = d_plateau_target * cache.plateau_gain * voltage_gate
            d_excitatory_conductance +=
                d_ligand_gate * exp(-excitatory_conductance)
            if zero(T) < plateau < one(T)
                dstate[state_base + FIELD_PLATEAU] += d_plateau_previous
            end

            d_voltage_argument = d_voltage_gate * voltage_gate * (one(T) - voltage_gate)
            d_voltage += d_voltage_argument * cache.inv_plateau_slope
            draw[P_PLATEAU_THRESHOLD] -= d_voltage_argument * cache.inv_plateau_slope
            draw[P_PLATEAU_SLOPE] -= d_voltage_argument *
                                     (voltage - cache.plateau_threshold) *
                                     cache.inv_plateau_slope * cache.inv_plateau_slope

            d_ampa_next += d_excitatory_conductance
            d_nmda_next += d_excitatory_conductance * nmda_unblock
            d_nmda_unblock = d_excitatory_conductance * nmda_next
            d_gaba_next += d_inhibitory_conductance

            d_nmda_argument = d_nmda_unblock * nmda_unblock * (one(T) - nmda_unblock)
            d_voltage += d_nmda_argument * cache.inv_nmda_slope
            draw[P_NMDA_HALF_VOLTAGE] -= d_nmda_argument * cache.inv_nmda_slope
            draw[P_NMDA_SLOPE] -= d_nmda_argument *
                                  (voltage - cache.nmda_half_voltage) *
                                  cache.inv_nmda_slope * cache.inv_nmda_slope

            draw[P_AMPA_DECAY] += d_ampa_next * (ampa_positive - ampa_drive)
            d_ampa_drive = d_ampa_next * (one(T) - cache.ampa_decay)
            draw[P_AMPA_MAX] += d_ampa_drive * (one(T) - ampa_input_decay)
            if ampa > zero(T)
                dstate[state_base + FIELD_AMPA] += d_ampa_next * cache.ampa_decay
            end
            if ampa_input > zero(T)
                dinput[input_base + INPUT_AMPA] +=
                    d_ampa_drive * cache.ampa_max * ampa_input_decay / T(AMPA_INPUT_SCALE)
            end

            draw[P_NMDA_DECAY] += d_nmda_next * (nmda_positive - nmda_drive)
            d_nmda_drive = d_nmda_next * (one(T) - cache.nmda_decay)
            draw[P_NMDA_MAX] += d_nmda_drive * (one(T) - nmda_input_decay)
            if nmda > zero(T)
                dstate[state_base + FIELD_NMDA] += d_nmda_next * cache.nmda_decay
            end
            if nmda_input > zero(T)
                dinput[input_base + INPUT_NMDA] +=
                    d_nmda_drive * cache.nmda_max * nmda_input_decay / T(NMDA_INPUT_SCALE)
            end

            draw[P_GABA_DECAY] += d_gaba_next * (gaba_positive - gaba_drive)
            d_gaba_drive = d_gaba_next * (one(T) - cache.gaba_decay)
            draw[P_GABA_MAX] += d_gaba_drive * (one(T) - gaba_input_decay)
            if gaba > zero(T)
                dstate[state_base + FIELD_GABA] += d_gaba_next * cache.gaba_decay
            end
            if gaba_input > zero(T)
                dinput[input_base + INPUT_GABA] +=
                    d_gaba_drive * cache.gaba_max * gaba_input_decay / T(GABA_INPUT_SCALE)
            end

            dstate[state_base + FIELD_VOLTAGE] += d_voltage
        end

        role_gradient = ntuple(
            compartment -> draw[FIRST_BASAL_ROLE + compartment - 1],
            Val(N_BASAL),
        )
        compartment_rest_gradient = draw[P_COMPARTMENT_REST]
        inhibitory_reversal_gradient = draw[P_INHIBITORY_GAP]
        soma_rest_gradient = draw[P_SOMA_REST]
        soma_threshold_gradient = draw[P_SOMA_THRESHOLD_GAP]
        soma_reset_gradient = draw[P_SOMA_RESET_GAP]

        for parameter in 1:(FIRST_BASAL_ROLE - 1)
            if parameter != P_COMPARTMENT_REST &&
               parameter != P_INHIBITORY_GAP &&
               parameter != P_SOMA_REST &&
               parameter != P_SOMA_THRESHOLD_GAP &&
               parameter != P_SOMA_RESET_GAP
                draw[parameter] *= derivative_cache.diagonal[parameter]
            end
        end
        draw[P_COMPARTMENT_REST] =
            (compartment_rest_gradient + inhibitory_reversal_gradient) *
            derivative_cache.diagonal[P_COMPARTMENT_REST]
        draw[P_INHIBITORY_GAP] =
            -inhibitory_reversal_gradient * derivative_cache.diagonal[P_INHIBITORY_GAP]
        draw[P_SOMA_REST] =
            (soma_rest_gradient + soma_threshold_gradient + soma_reset_gradient) *
            derivative_cache.diagonal[P_SOMA_REST]
        draw[P_SOMA_THRESHOLD_GAP] =
            (soma_threshold_gradient + soma_reset_gradient) *
            derivative_cache.diagonal[P_SOMA_THRESHOLD_GAP]
        draw[P_SOMA_RESET_GAP] =
            -soma_reset_gradient * derivative_cache.diagonal[P_SOMA_RESET_GAP]

        for raw_role in 1:N_BASAL
            jacobian_base = (raw_role - 1) * N_BASAL
            value = zero(T)
            for output_role in 1:N_BASAL
                value = muladd(
                    role_gradient[output_role],
                    derivative_cache.basal_role_jacobian[
                        jacobian_base + output_role
                    ],
                    value,
                )
            end
            draw[FIRST_BASAL_ROLE + raw_role - 1] = value
        end
    end
    return dstate, dinput, draw
end

"""
    cell_step_pullback!(...)

One-step reverse with the canonical triangular surrogate propagated through
all downstream uses of the hard event (spike state, reset, adaptation and
bAP).  This is the discrete-control learner; it is intentionally not described
as an exact derivative of a hard threshold.
"""
function cell_step_pullback!(
    dstate::AbstractVector{T},
    dinput::AbstractVector{T},
    draw::AbstractVector{T},
    state::AbstractVector{T},
    input::AbstractVector{T},
    cache::CellParameterCache{T},
    derivative_cache::CellParameterDerivativeCache{T},
    next_state::AbstractVector{T},
    dnext::AbstractVector{T},
    event_cotangent::T,
    spike_smoothing::T=zero(T),
    direct_margin_cotangent::T=zero(T),
) where {T<:AbstractFloat}
    return _cell_step_pullback!(
        dstate,
        dinput,
        draw,
        state,
        input,
        cache,
        derivative_cache,
        next_state,
        dnext,
        event_cotangent,
        spike_smoothing,
        direct_margin_cotangent,
        true,
    )
end

"""
    cell_step_conditional_pullback!(...)

Exact derivative of the continuous state transition conditional on the
recorded hard-event sequence.  Reset, adaptation and bAP still use the hard
event in forward, but their derivative through the threshold is zero, as it is
almost everywhere.  Only an explicit `event_cotangent` receives a triangular
surrogate.  This keeps continuous ListNet credit mathematically separate from
the optional hard-control learner.
"""
function cell_step_conditional_pullback!(
    dstate::AbstractVector{T},
    dinput::AbstractVector{T},
    draw::AbstractVector{T},
    state::AbstractVector{T},
    input::AbstractVector{T},
    cache::CellParameterCache{T},
    derivative_cache::CellParameterDerivativeCache{T},
    next_state::AbstractVector{T},
    dnext::AbstractVector{T},
    event_cotangent::T=zero(T),
    spike_smoothing::T=zero(T),
    direct_margin_cotangent::T=zero(T),
) where {T<:AbstractFloat}
    return _cell_step_pullback!(
        dstate,
        dinput,
        draw,
        state,
        input,
        cache,
        derivative_cache,
        next_state,
        dnext,
        event_cotangent,
        spike_smoothing,
        direct_margin_cotangent,
        false,
    )
end

end # module ActiveApicalCell
