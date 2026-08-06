module SharedDendriticFactor

using ..ActiveApicalCell

const Cell = ActiveApicalCell

export PHASE_COUNT,
       DRIVE_DIM,
       PROGRAM_DIM,
       FEATURE_DIM,
       APICAL_DRIVE_INDEX,
       APICAL_VOLTAGE_FEATURE,
       SOMA_MARGIN_FEATURE,
       ADAPTATION_FEATURE,
       FactorTrace,
       FactorScratch,
       default_raw_program,
       factor_forward!,
       factor_pullback!

"""Three fixed integration phases expose NMDA and plateau dynamics."""
const PHASE_COUNT = 3

"""Eight basal drives followed by one signed apical drive."""
const DRIVE_DIM = Cell.N_BASAL + 1
const APICAL_DRIVE_INDEX = DRIVE_DIM

"""Eight excitatory and eight inhibitory raw gains."""
const PROGRAM_DIM = 2 * Cell.N_BASAL
const FIRST_INHIBITORY_PROGRAM = Cell.N_BASAL + 1

# External features deliberately exclude the hard soma spike.  The first 24
# coordinates preserve branch identity instead of averaging it away.
const BRANCH_VOLTAGE_FEATURE = 1
const BRANCH_NMDA_FEATURE = BRANCH_VOLTAGE_FEATURE + Cell.N_BASAL
const BRANCH_PLATEAU_FEATURE = BRANCH_NMDA_FEATURE + Cell.N_BASAL
const APICAL_VOLTAGE_FEATURE = BRANCH_PLATEAU_FEATURE + Cell.N_BASAL
const SOMA_MARGIN_FEATURE = APICAL_VOLTAGE_FEATURE + 1
const ADAPTATION_FEATURE = SOMA_MARGIN_FEATURE + 1
const FEATURE_DIM = ADAPTATION_FEATURE
const VOLTAGE_FEATURE_SCALE = 10.0f0
const MARGIN_FEATURE_SCALE = 5.0f0

"""
Caller-owned trajectory for one factor evaluation.

`states[:, 1]` is the parameter-dependent resting state and the remaining
columns are the three phase results.  `input` is shared by all three phases;
the program and drive are fixed during one factor evaluation.
"""
struct FactorTrace{T<:AbstractFloat}
    states::Matrix{T}
    input::Vector{T}
    silent_input::Vector{T}
    margins::Vector{T}
    spikes::Vector{T}
end

function FactorTrace(::Type{T}=Float32) where {T<:AbstractFloat}
    return FactorTrace(
        Matrix{T}(undef, Cell.STATE_DIM, PHASE_COUNT + 1),
        Vector{T}(undef, Cell.INPUT_DIM),
        zeros(T, Cell.INPUT_DIM),
        Vector{T}(undef, PHASE_COUNT),
        Vector{T}(undef, PHASE_COUNT),
    )
end

"""Caller-owned reverse buffers.  No buffer aliases a stored trajectory."""
struct FactorScratch{T<:AbstractFloat}
    dstate::Vector{T}
    dinput::Vector{T}
    draw_step::Vector{T}
    dnext::Vector{T}
end

function FactorScratch(::Type{T}=Float32) where {T<:AbstractFloat}
    return FactorScratch(
        Vector{T}(undef, Cell.STATE_DIM),
        Vector{T}(undef, Cell.INPUT_DIM),
        Vector{T}(undef, Cell.PARAM_DIM),
        Vector{T}(undef, Cell.STATE_DIM),
    )
end

@inline function _softplus(value::T) where {T<:AbstractFloat}
    return max(value, zero(T)) + log1p(exp(-abs(value)))
end

@inline _softplus_derivative(value::T) where {T<:AbstractFloat} =
    inv(one(T) + exp(-value))

@inline function _inverse_softplus(value::T) where {T<:AbstractFloat}
    value > zero(T) || throw(ArgumentError("program gain must be positive"))
    return value + log(-expm1(-value))
end

"""
    default_raw_program([T])

Return a quiet but non-dormant program.  Excitatory and inhibitory contacts
remain separate.  One excitatory gain drives both AMPA and NMDA, preserving
fast evidence and slow voltage-dependent evidence with one stored value.
"""
function default_raw_program(::Type{T}=Float32) where {T<:AbstractFloat}
    raw = Vector{T}(undef, PROGRAM_DIM)
    # At initialization an occupied symbol and an equally strong empty-symbol
    # contact have equal physical magnitude.  Their receptor identity, not an
    # arbitrary gain imbalance, determines the sign of their evidence.
    excitatory = _inverse_softplus(T(0.55))
    inhibitory = _inverse_softplus(T(0.55))
    @inbounds for branch in 1:Cell.N_BASAL
        raw[branch] = excitatory
        raw[FIRST_INHIBITORY_PROGRAM + branch - 1] = inhibitory
    end
    return raw
end

@inline function _check_trace(trace::FactorTrace{T}) where {T}
    size(trace.states) == (Cell.STATE_DIM, PHASE_COUNT + 1) ||
        throw(DimensionMismatch("factor trace states have the wrong shape"))
    length(trace.input) == Cell.INPUT_DIM ||
        throw(DimensionMismatch("factor trace input has the wrong length"))
    length(trace.silent_input) == Cell.INPUT_DIM ||
        throw(DimensionMismatch(
            "factor trace silent input has the wrong length",
        ))
    length(trace.margins) == PHASE_COUNT ||
        throw(DimensionMismatch("factor trace margins have the wrong length"))
    length(trace.spikes) == PHASE_COUNT ||
        throw(DimensionMismatch("factor trace spikes have the wrong length"))
    return nothing
end

@inline function _write_input!(
    input::AbstractVector{T},
    drive::AbstractVector{T},
    raw_program::AbstractVector{T},
) where {T<:AbstractFloat}
    fill!(input, zero(T))
    @inbounds for branch in 1:Cell.N_BASAL
        activity = drive[branch]
        if activity > zero(T)
            excitatory = activity * _softplus(raw_program[branch])
            input[Cell.input_index(branch, Cell.INPUT_AMPA)] = excitatory
            input[Cell.input_index(branch, Cell.INPUT_NMDA)] = excitatory
        elseif activity < zero(T)
            inhibitory = -activity * _softplus(
                raw_program[FIRST_INHIBITORY_PROGRAM + branch - 1],
            )
            input[Cell.input_index(branch, Cell.INPUT_GABA)] = inhibitory
        end
    end

    # The apical coordinate is signed context, not another learned branch.
    # Positive context opens AMPA+NMDA; negative context recruits inhibition.
    apical = drive[APICAL_DRIVE_INDEX]
    apical_compartment = Cell.N_COMPARTMENTS
    input[Cell.input_index(apical_compartment, Cell.INPUT_AMPA)] =
        max(apical, zero(T))
    input[Cell.input_index(apical_compartment, Cell.INPUT_NMDA)] =
        max(apical, zero(T))
    input[Cell.input_index(apical_compartment, Cell.INPUT_GABA)] =
        max(-apical, zero(T))
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

# The Float64 path is for focused derivative tests.  The production Float32
# method above reuses the allocation-free in-place Reduced Hay kernel.
function _cell_step!(
    destination::AbstractVector{T},
    state::AbstractVector{T},
    input::AbstractVector{T},
    cache::Cell.CellParameterCache{T},
) where {T<:AbstractFloat}
    copyto!(destination, Cell.cell_step_cached_functional(state, input, cache))
    return destination
end

@inline function _write_features!(
    features::AbstractVector{T},
    initial_state::AbstractVector{T},
    previous_state::AbstractVector{T},
    state::AbstractVector{T},
    margin::T,
) where {T<:AbstractFloat}
    @inbounds for branch in 1:Cell.N_BASAL
        features[BRANCH_VOLTAGE_FEATURE + branch - 1] =
            (state[Cell.state_index(branch, Cell.FIELD_VOLTAGE)] -
             initial_state[Cell.state_index(branch, Cell.FIELD_VOLTAGE)]) /
            T(VOLTAGE_FEATURE_SCALE)
        features[BRANCH_NMDA_FEATURE + branch - 1] =
            state[Cell.state_index(branch, Cell.FIELD_NMDA)]
        features[BRANCH_PLATEAU_FEATURE + branch - 1] =
            state[Cell.state_index(branch, Cell.FIELD_PLATEAU)]
    end
    @inbounds begin
        apical_voltage_index = Cell.state_index(
            Cell.N_COMPARTMENTS,
            Cell.FIELD_VOLTAGE,
        )
        features[APICAL_VOLTAGE_FEATURE] =
            (state[apical_voltage_index] - initial_state[apical_voltage_index]) /
            T(VOLTAGE_FEATURE_SCALE)
        # `margin` is reconstructed before soma reset.  The hard spike remains
        # a control output and never becomes an analog feature coordinate.
        features[SOMA_MARGIN_FEATURE] = margin / T(MARGIN_FEATURE_SCALE)
        features[ADAPTATION_FEATURE] = state[Cell.ADAPTATION_INDEX]
    end
    return features
end

"""
    factor_forward!(features, trace, drive, raw_program, cache) -> hard_spike

Integrate one high-dimensional factor for three phases.  The typed sensory
event is injected only in phase one; phases two and three expose the cell's
own NMDA, plateau and adaptation dynamics without repeatedly adding the same
DC current.  The 27 continuous
features preserve branch voltage, NMDA and plateau identity, apical voltage,
the pre-reset soma margin, and adaptation.  The returned hard spike is solely
the event/control plane.
"""
function factor_forward!(
    features::AbstractVector{T},
    trace::FactorTrace{T},
    drive::AbstractVector{T},
    raw_program::AbstractVector{T},
    cache::Cell.CellParameterCache{T},
) where {T<:AbstractFloat}
    length(features) == FEATURE_DIM ||
        throw(DimensionMismatch("expected $FEATURE_DIM factor features"))
    length(drive) == DRIVE_DIM ||
        throw(DimensionMismatch("expected $DRIVE_DIM factor drives"))
    length(raw_program) == PROGRAM_DIM ||
        throw(DimensionMismatch("expected $PROGRAM_DIM raw program values"))
    _check_trace(trace)

    Cell.initial_state!(@view(trace.states[:, 1]), cache)
    _write_input!(trace.input, drive, raw_program)
    @inbounds for phase in 1:PHASE_COUNT
        previous_state = @view trace.states[:, phase]
        state = @view trace.states[:, phase + 1]
        phase_input = phase == 1 ? trace.input : trace.silent_input
        _cell_step!(state, previous_state, phase_input, cache)
        trace.margins[phase] = Cell.spike_margin_from_transition(
            previous_state,
            state,
            cache,
        )
        trace.spikes[phase] = state[Cell.SPIKE_INDEX]
    end
    final_previous = @view trace.states[:, PHASE_COUNT]
    final_state = @view trace.states[:, PHASE_COUNT + 1]
    _write_features!(
        features,
        @view(trace.states[:, 1]),
        final_previous,
        final_state,
        trace.margins[PHASE_COUNT],
    )
    return trace.spikes[PHASE_COUNT]
end

@inline function _seed_feature_pullback!(
    dnext::AbstractVector{T},
    dfeatures::AbstractVector{T},
) where {T<:AbstractFloat}
    fill!(dnext, zero(T))
    @inbounds for branch in 1:Cell.N_BASAL
        dnext[Cell.state_index(branch, Cell.FIELD_VOLTAGE)] +=
            dfeatures[BRANCH_VOLTAGE_FEATURE + branch - 1] /
            T(VOLTAGE_FEATURE_SCALE)
        dnext[Cell.state_index(branch, Cell.FIELD_NMDA)] +=
            dfeatures[BRANCH_NMDA_FEATURE + branch - 1]
        dnext[Cell.state_index(branch, Cell.FIELD_PLATEAU)] +=
            dfeatures[BRANCH_PLATEAU_FEATURE + branch - 1]
    end
    @inbounds begin
        dnext[Cell.state_index(Cell.N_COMPARTMENTS, Cell.FIELD_VOLTAGE)] +=
            dfeatures[APICAL_VOLTAGE_FEATURE] / T(VOLTAGE_FEATURE_SCALE)
        dnext[Cell.ADAPTATION_INDEX] += dfeatures[ADAPTATION_FEATURE]
    end
    return @inbounds dfeatures[SOMA_MARGIN_FEATURE] / T(MARGIN_FEATURE_SCALE)
end

@inline function _input_pullback!(
    ddrive::AbstractVector{T},
    draw_program::AbstractVector{T},
    dinput::AbstractVector{T},
    drive::AbstractVector{T},
    raw_program::AbstractVector{T},
) where {T<:AbstractFloat}
    @inbounds for branch in 1:Cell.N_BASAL
        activity = drive[branch]
        excitatory_index = branch
        inhibitory_index = FIRST_INHIBITORY_PROGRAM + branch - 1
        excitatory_input_bar =
            dinput[Cell.input_index(branch, Cell.INPUT_AMPA)] +
            dinput[Cell.input_index(branch, Cell.INPUT_NMDA)]
        inhibitory_input_bar =
            dinput[Cell.input_index(branch, Cell.INPUT_GABA)]
        excitatory_gain = _softplus(raw_program[excitatory_index])
        inhibitory_gain = _softplus(raw_program[inhibitory_index])
        if activity > zero(T)
            ddrive[branch] += excitatory_input_bar * excitatory_gain
            draw_program[excitatory_index] +=
                excitatory_input_bar * activity *
                _softplus_derivative(raw_program[excitatory_index])
        elseif activity < zero(T)
            ddrive[branch] -= inhibitory_input_bar * inhibitory_gain
            draw_program[inhibitory_index] +=
                inhibitory_input_bar * (-activity) *
                _softplus_derivative(raw_program[inhibitory_index])
        end
    end

    apical = drive[APICAL_DRIVE_INDEX]
    apical_compartment = Cell.N_COMPARTMENTS
    if apical > zero(T)
        ddrive[APICAL_DRIVE_INDEX] +=
            dinput[Cell.input_index(apical_compartment, Cell.INPUT_AMPA)] +
            dinput[Cell.input_index(apical_compartment, Cell.INPUT_NMDA)]
    elseif apical < zero(T)
        ddrive[APICAL_DRIVE_INDEX] -=
            dinput[Cell.input_index(apical_compartment, Cell.INPUT_GABA)]
    end
    return ddrive, draw_program
end

"""
    factor_pullback!(ddrive, draw_program, draw_shared, scratch, trace,
                     drive, raw_program, cache, derivative_cache, dfeatures,
                     control_cotangent=0)

Reverse all three phases into the factor drive, the 16-value local program,
and the shared raw Reduced Hay parameters.  The final hard control event has a
separate optional cotangent.  The analog feature cotangent never reads the
hard spike.  After the three transitions, the reverse explicitly traverses
the parameter-dependent resting initial state.
"""
function factor_pullback!(
    ddrive::AbstractVector{T},
    draw_program::AbstractVector{T},
    draw_shared::AbstractVector{T},
    scratch::FactorScratch{T},
    trace::FactorTrace{T},
    drive::AbstractVector{T},
    raw_program::AbstractVector{T},
    cache::Cell.CellParameterCache{T},
    derivative_cache::Cell.CellParameterDerivativeCache{T},
    dfeatures::AbstractVector{T},
    control_cotangent::T=zero(T),
) where {T<:AbstractFloat}
    length(ddrive) == DRIVE_DIM ||
        throw(DimensionMismatch("expected $DRIVE_DIM drive cotangents"))
    length(draw_program) == PROGRAM_DIM ||
        throw(DimensionMismatch("expected $PROGRAM_DIM program cotangents"))
    length(draw_shared) == Cell.PARAM_DIM ||
        throw(DimensionMismatch("expected $(Cell.PARAM_DIM) shared cotangents"))
    length(drive) == DRIVE_DIM ||
        throw(DimensionMismatch("expected $DRIVE_DIM factor drives"))
    length(raw_program) == PROGRAM_DIM ||
        throw(DimensionMismatch("expected $PROGRAM_DIM raw program values"))
    length(dfeatures) == FEATURE_DIM ||
        throw(DimensionMismatch("expected $FEATURE_DIM feature cotangents"))
    _check_trace(trace)

    fill!(ddrive, zero(T))
    fill!(draw_program, zero(T))
    fill!(draw_shared, zero(T))
    margin_cotangent = _seed_feature_pullback!(scratch.dnext, dfeatures)

    @inbounds for phase in PHASE_COUNT:-1:1
        previous_state = @view trace.states[:, phase]
        state = @view trace.states[:, phase + 1]
        Cell.cell_step_conditional_pullback!(
            scratch.dstate,
            scratch.dinput,
            scratch.draw_step,
            previous_state,
            phase == 1 ? trace.input : trace.silent_input,
            cache,
            derivative_cache,
            state,
            scratch.dnext,
            phase == PHASE_COUNT ? control_cotangent : zero(T),
            zero(T),
            phase == PHASE_COUNT ? margin_cotangent : zero(T),
        )
        for parameter in 1:Cell.PARAM_DIM
            draw_shared[parameter] += scratch.draw_step[parameter]
        end
        if phase == 1
            _input_pullback!(
                ddrive,
                draw_program,
                scratch.dinput,
                drive,
                raw_program,
            )
        end
        copyto!(scratch.dnext, scratch.dstate)
    end

    # `scratch.dnext` is now the cotangent of trace.states[:, 1].  The resting
    # voltages also appear explicitly in the DC-free voltage features.  Add
    # those negative initial-state terms before traversing the parameterized
    # resting-state constructor.
    @inbounds for branch in 1:Cell.N_BASAL
        scratch.dnext[Cell.state_index(branch, Cell.FIELD_VOLTAGE)] -=
            dfeatures[BRANCH_VOLTAGE_FEATURE + branch - 1] /
            T(VOLTAGE_FEATURE_SCALE)
    end
    @inbounds scratch.dnext[Cell.state_index(
        Cell.N_COMPARTMENTS,
        Cell.FIELD_VOLTAGE,
    )] -= dfeatures[APICAL_VOLTAGE_FEATURE] / T(VOLTAGE_FEATURE_SCALE)

    # The resting voltages depend on compartment_rest and soma_rest, so
    # omitting this step silently loses a real shared-parameter path.
    Cell.initial_state_pullback!(
        draw_shared,
        scratch.dnext,
        derivative_cache,
    )
    return ddrive, draw_program, draw_shared
end

end # module SharedDendriticFactor
