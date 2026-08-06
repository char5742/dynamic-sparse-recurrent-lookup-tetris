module CompactDendriticNode

using ..ActiveApicalCell

const Cell = ActiveApicalCell

export PHASE_COUNT,
       DRIVE_DIM,
       ANALOG_DIM,
       PAYLOAD_DIM,
       CENTERED_MARGIN_INDEX,
       MEAN_PLATEAU_INDEX,
       HARD_EVENT_INDEX,
       CENTERED_MARGIN_SCALE,
       MEAN_PLATEAU_SCALE,
       EXCITATORY_DRIVE_SCALE,
       INHIBITORY_DRIVE_SCALE,
       NodeTrace,
       NodeScratch,
       node_forward!,
       node_pullback!

"""One driven phase followed by two input-free relaxation phases."""
const PHASE_COUNT = 3

"""Eight signed basal drives followed by one signed apical drive."""
const DRIVE_DIM = Cell.N_BASAL + 1

"""The differentiable part of the compact node payload."""
const ANALOG_DIM = 2

"""Two continuous coordinates plus one hard event/control coordinate."""
const PAYLOAD_DIM = ANALOG_DIM + 1
const CENTERED_MARGIN_INDEX = 1
const MEAN_PLATEAU_INDEX = 2
const HARD_EVENT_INDEX = 3

"""Five millivolts is one dimensionless centered-margin unit."""
const CENTERED_MARGIN_SCALE = 5.0f0

"""One thirty-second mean plateau occupancy is one dimensionless unit."""
const MEAN_PLATEAU_SCALE = 0.03125f0

"""
Semantic input calibration for positive AMPA/NMDA evidence.

The underlying conductance transform gives one unit of positive signed drive
two excitatory receptor paths, while negative evidence has one GABA path.
This fixed boundary calibration equalizes their three-phase soma-margin
effect without changing the Reduced Hay cell equation.
"""
const EXCITATORY_DRIVE_SCALE = 0.035f0

"""Negative signed evidence is the reference GABA drive scale."""
const INHIBITORY_DRIVE_SCALE = 1.0f0

const APICAL_DRIVE_INDEX = DRIVE_DIM
const SOMA_THRESHOLD_GAP_RAW_INDEX = something(
    findfirst(==(:soma_threshold_gap), Cell.PARAMETER_NAMES),
)

"""
Caller-owned trajectory for one compact dendritic node.

The node remains a full 48-state Reduced Hay cell.  `states[:, 1]` is the
parameter-dependent resting state and the next three columns are the driven
phase and two silent relaxation phases.  Only the final compact payload is
exported; no internal conductance or compartment state is discarded locally.
"""
struct NodeTrace{T<:AbstractFloat}
    states::Matrix{T}
    driven_input::Vector{T}
    silent_input::Vector{T}
    margins::Vector{T}
    events::Vector{T}
end

function NodeTrace(::Type{T}=Float32) where {T<:AbstractFloat}
    return NodeTrace(
        Matrix{T}(undef, Cell.STATE_DIM, PHASE_COUNT + 1),
        Vector{T}(undef, Cell.INPUT_DIM),
        zeros(T, Cell.INPUT_DIM),
        Vector{T}(undef, PHASE_COUNT),
        Vector{T}(undef, PHASE_COUNT),
    )
end

"""Caller-owned reverse buffers for the three cell transitions."""
struct NodeScratch{T<:AbstractFloat}
    dstate::Vector{T}
    dinput::Vector{T}
    draw_step::Vector{T}
    dnext::Vector{T}
end

function NodeScratch(::Type{T}=Float32) where {T<:AbstractFloat}
    return NodeScratch(
        Vector{T}(undef, Cell.STATE_DIM),
        Vector{T}(undef, Cell.INPUT_DIM),
        Vector{T}(undef, Cell.PARAM_DIM),
        Vector{T}(undef, Cell.STATE_DIM),
    )
end

@inline function _check_trace(trace::NodeTrace)
    size(trace.states) == (Cell.STATE_DIM, PHASE_COUNT + 1) ||
        throw(DimensionMismatch("node trace states have the wrong shape"))
    length(trace.driven_input) == Cell.INPUT_DIM ||
        throw(DimensionMismatch("node trace driven input has the wrong length"))
    length(trace.silent_input) == Cell.INPUT_DIM ||
        throw(DimensionMismatch("node trace silent input has the wrong length"))
    length(trace.margins) == PHASE_COUNT ||
        throw(DimensionMismatch("node trace margins have the wrong length"))
    length(trace.events) == PHASE_COUNT ||
        throw(DimensionMismatch("node trace events have the wrong length"))
    return nothing
end

"""
Encode signed evidence without conflating inhibition with silence.

Positive drive recruits both the fast AMPA and slow voltage-dependent NMDA
channels.  Negative drive recruits GABA.  A zero drive writes no event.
"""
@inline function _write_driven_input!(
    input::AbstractVector{T},
    drive::AbstractVector{T},
) where {T<:AbstractFloat}
    fill!(input, zero(T))
    @inbounds for branch in 1:Cell.N_BASAL
        value = drive[branch]
        excitatory = max(value, zero(T)) * T(EXCITATORY_DRIVE_SCALE)
        inhibitory = max(-value, zero(T)) * T(INHIBITORY_DRIVE_SCALE)
        input[Cell.input_index(branch, Cell.INPUT_AMPA)] = excitatory
        input[Cell.input_index(branch, Cell.INPUT_NMDA)] = excitatory
        input[Cell.input_index(branch, Cell.INPUT_GABA)] = inhibitory
    end
    @inbounds begin
        value = drive[APICAL_DRIVE_INDEX]
        compartment = Cell.N_COMPARTMENTS
        excitatory = max(value, zero(T)) * T(EXCITATORY_DRIVE_SCALE)
        inhibitory = max(-value, zero(T)) * T(INHIBITORY_DRIVE_SCALE)
        input[Cell.input_index(compartment, Cell.INPUT_AMPA)] = excitatory
        input[Cell.input_index(compartment, Cell.INPUT_NMDA)] = excitatory
        input[Cell.input_index(compartment, Cell.INPUT_GABA)] = inhibitory
    end
    return input
end

@inline function _cell_step!(
    destination::AbstractVector{Float32},
    state::AbstractVector{Float32},
    input::AbstractVector{Float32},
    cache::Cell.CellParameterCache{Float32},
)
    return Cell.cell_step!(destination, state, input, cache)
end

# Float64 is retained only for focused derivative oracles.  The production
# Float32 method above is the allocation-free in-place kernel.
function _cell_step!(
    destination::AbstractVector{T},
    state::AbstractVector{T},
    input::AbstractVector{T},
    cache::Cell.CellParameterCache{T},
) where {T<:AbstractFloat}
    copyto!(destination, Cell.cell_step_cached_functional(state, input, cache))
    return destination
end

@inline function _resting_margin(cache::Cell.CellParameterCache{T}) where {T}
    return cache.soma_rest - cache.soma_threshold
end

"""
    node_forward!(payload, trace, drive, cache) -> hard_event

Run the full 48-state Reduced Hay cell for three phases.  External evidence is
present only in phase one.  Phases two and three use an exactly zero input so
NMDA, plateau, adaptation, reset and bAP evolve from the cell's own state.

The compact payload is

1. pre-reset soma margin minus its resting margin, divided by 5 mV,
2. mean final basal plateau occupancy divided by 1/32, and
3. exact OR of the three phase hard events (0 or 1).

Coordinate three is a control coordinate, not a continuous feature.  Its
surrogate cotangent is accepted separately by `node_pullback!`.
"""
function node_forward!(
    payload::AbstractVector{T},
    trace::NodeTrace{T},
    drive::AbstractVector{T},
    cache::Cell.CellParameterCache{T},
) where {T<:AbstractFloat}
    length(payload) == PAYLOAD_DIM ||
        throw(DimensionMismatch("expected $PAYLOAD_DIM payload coordinates"))
    length(drive) == DRIVE_DIM ||
        throw(DimensionMismatch("expected $DRIVE_DIM signed drives"))
    _check_trace(trace)

    Cell.initial_state!(@view(trace.states[:, 1]), cache)
    _write_driven_input!(trace.driven_input, drive)
    @inbounds for phase in 1:PHASE_COUNT
        previous_state = @view trace.states[:, phase]
        state = @view trace.states[:, phase + 1]
        phase_input = phase == 1 ? trace.driven_input : trace.silent_input
        _cell_step!(state, previous_state, phase_input, cache)
        trace.margins[phase] = Cell.spike_margin_from_transition(
            previous_state,
            state,
            cache,
        )
        trace.events[phase] = state[Cell.SPIKE_INDEX]
    end

    final_state = @view trace.states[:, PHASE_COUNT + 1]
    plateau_sum = zero(T)
    @inbounds for branch in 1:Cell.N_BASAL
        plateau_sum += final_state[Cell.state_index(branch, Cell.FIELD_PLATEAU)]
    end
    # A compact node event means that the node fired at least once during its
    # complete three-phase integration window.  Reading only phase three
    # would erase a valid early event after reset/adaptation silenced the
    # final phase.  Since each phase event is an exact bit, `max` is exact OR.
    any_event = zero(T)
    @inbounds for phase in 1:PHASE_COUNT
        any_event = max(any_event, trace.events[phase])
    end
    @inbounds begin
        payload[CENTERED_MARGIN_INDEX] =
            (trace.margins[PHASE_COUNT] - _resting_margin(cache)) /
            T(CENTERED_MARGIN_SCALE)
        payload[MEAN_PLATEAU_INDEX] =
            plateau_sum / (T(Cell.N_BASAL) * T(MEAN_PLATEAU_SCALE))
        payload[HARD_EVENT_INDEX] = any_event
    end
    return any_event
end

@inline function _event_argmax_phase(trace::NodeTrace{T}) where {T}
    # `>` deliberately keeps the earliest phase on an exact tie, giving the
    # OR surrogate H(max margin) one deterministic generalized derivative.
    phase_index = 1
    maximum_margin = @inbounds trace.margins[1]
    @inbounds for phase in 2:PHASE_COUNT
        margin = trace.margins[phase]
        if margin > maximum_margin
            maximum_margin = margin
            phase_index = phase
        end
    end
    return phase_index
end

@inline function _driven_input_pullback!(
    ddrive::AbstractVector{T},
    dinput::AbstractVector{T},
    drive::AbstractVector{T},
) where {T<:AbstractFloat}
    @inbounds for branch in 1:Cell.N_BASAL
        value = drive[branch]
        if value > zero(T)
            ddrive[branch] += T(EXCITATORY_DRIVE_SCALE) * (
                dinput[Cell.input_index(branch, Cell.INPUT_AMPA)] +
                dinput[Cell.input_index(branch, Cell.INPUT_NMDA)]
            )
        elseif value < zero(T)
            ddrive[branch] -= T(INHIBITORY_DRIVE_SCALE) *
                dinput[Cell.input_index(branch, Cell.INPUT_GABA)]
        end
    end
    @inbounds begin
        value = drive[APICAL_DRIVE_INDEX]
        compartment = Cell.N_COMPARTMENTS
        if value > zero(T)
            ddrive[APICAL_DRIVE_INDEX] += T(EXCITATORY_DRIVE_SCALE) * (
                dinput[Cell.input_index(compartment, Cell.INPUT_AMPA)] +
                dinput[Cell.input_index(compartment, Cell.INPUT_NMDA)]
            )
        elseif value < zero(T)
            ddrive[APICAL_DRIVE_INDEX] -= T(INHIBITORY_DRIVE_SCALE) *
                dinput[Cell.input_index(compartment, Cell.INPUT_GABA)]
        end
    end
    return ddrive
end

"""
    node_pullback!(ddrive, draw_shared, scratch, trace, drive, cache,
                   derivative_cache, danalog[, event_cotangent])

Reverse the two continuous payload coordinates exactly, conditional on the
recorded hard-event sequence.  `danalog` has length two and therefore cannot
silently differentiate the hard event.  An optional nonzero
`event_cotangent` explicitly enables `ActiveApicalCell`'s triangular spike
surrogate for the control coordinate only.  OR is relaxed as H(maximum phase
margin), with the cotangent sent to the earliest maximum-margin phase on a
tie.  It is a surrogate control learner, not part of the exact analog VJP.
"""
function node_pullback!(
    ddrive::AbstractVector{T},
    draw_shared::AbstractVector{T},
    scratch::NodeScratch{T},
    trace::NodeTrace{T},
    drive::AbstractVector{T},
    cache::Cell.CellParameterCache{T},
    derivative_cache::Cell.CellParameterDerivativeCache{T},
    danalog::AbstractVector{T},
    event_cotangent::T=zero(T),
) where {T<:AbstractFloat}
    length(ddrive) == DRIVE_DIM ||
        throw(DimensionMismatch("expected $DRIVE_DIM drive cotangents"))
    length(draw_shared) == Cell.PARAM_DIM ||
        throw(DimensionMismatch("expected $(Cell.PARAM_DIM) shared cotangents"))
    length(drive) == DRIVE_DIM ||
        throw(DimensionMismatch("expected $DRIVE_DIM signed drives"))
    length(danalog) == ANALOG_DIM || throw(DimensionMismatch(
        "expected exactly $ANALOG_DIM analog cotangents; pass the hard-event " *
        "cotangent as the separate final argument",
    ))
    _check_trace(trace)

    fill!(ddrive, zero(T))
    fill!(draw_shared, zero(T))
    fill!(scratch.dnext, zero(T))
    @inbounds for branch in 1:Cell.N_BASAL
        scratch.dnext[Cell.state_index(branch, Cell.FIELD_PLATEAU)] +=
            danalog[MEAN_PLATEAU_INDEX] /
            (T(Cell.N_BASAL) * T(MEAN_PLATEAU_SCALE))
    end

    margin_cotangent = @inbounds(
        danalog[CENTERED_MARGIN_INDEX] / T(CENTERED_MARGIN_SCALE)
    )
    event_phase = _event_argmax_phase(trace)
    @inbounds for phase in PHASE_COUNT:-1:1
        previous_state = @view trace.states[:, phase]
        state = @view trace.states[:, phase + 1]
        Cell.cell_step_conditional_pullback!(
            scratch.dstate,
            scratch.dinput,
            scratch.draw_step,
            previous_state,
            phase == 1 ? trace.driven_input : trace.silent_input,
            cache,
            derivative_cache,
            state,
            scratch.dnext,
            phase == event_phase ? event_cotangent : zero(T),
            zero(T),
            phase == PHASE_COUNT ? margin_cotangent : zero(T),
        )
        for parameter in 1:Cell.PARAM_DIM
            draw_shared[parameter] += scratch.draw_step[parameter]
        end
        phase == 1 && _driven_input_pullback!(
            ddrive,
            scratch.dinput,
            drive,
        )
        copyto!(scratch.dnext, scratch.dstate)
    end

    # Traverse the parameterized resting state after all three transitions.
    Cell.initial_state_pullback!(draw_shared, scratch.dnext, derivative_cache)

    # The exported margin is centered by subtracting the resting margin.
    # resting_margin == -soma_threshold_gap, so the raw correction is +d(gap).
    @inbounds draw_shared[SOMA_THRESHOLD_GAP_RAW_INDEX] +=
        margin_cotangent *
        derivative_cache.diagonal[SOMA_THRESHOLD_GAP_RAW_INDEX]
    return ddrive, draw_shared
end

end # module CompactDendriticNode
