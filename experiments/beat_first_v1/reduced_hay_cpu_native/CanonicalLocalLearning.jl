module CanonicalLocalLearning

using LinearAlgebra
using ..ActiveApicalCell
using ..DendriticAxonPacket

const Cell = ActiveApicalCell
const Axon = DendriticAxonPacket

export LOCAL_OBSERVATION_DIM,
       LocalParameterBasis,
       AnalogEligibilityState,
       HardEventEligibilityState,
       StructuralUtilityState,
       EligibilityScratch,
       FixedLocalSignalMap,
       LearningSchedule,
       LearningClockState,
       DuePlasticityClocks,
       ReplayPhase,
       TwoPassListNetReplay,
       reset_replay!,
       reset_eligibility!,
       advance_clocks!,
       record_teacher_free_forward!,
       seal_listnet_deltas!,
       copy_replay_delta!,
       finish_replay!,
       project_learning_signal!,
       accumulate_active_apical_transition!,
       accumulate_analog_gradient!,
       accumulate_packet_gradient!,
       accumulate_hard_event_gradient!,
       update_structural_utility!,
       update_packet_structural_utility!,
       continuous_observation!

# Local learning observes every continuous compartment coordinate, the exact
# pre-reset soma margin, and adaptation.  The hard spike is deliberately kept
# on the separate control plane.
const LOCAL_OBSERVATION_DIM =
    Cell.N_COMPARTMENTS * Cell.COMPARTMENT_STATE_DIM + 2
const MARGIN_OBSERVATION =
    Cell.N_COMPARTMENTS * Cell.COMPARTMENT_STATE_DIM + 1
const ADAPTATION_OBSERVATION = MARGIN_OBSERVATION + 1

@assert LOCAL_OBSERVATION_DIM == Cell.STATE_DIM - 1

@inline function _require_positive(name::AbstractString, value::Integer)
    value > 0 || throw(ArgumentError("$name must be positive"))
    return Int(value)
end

@inline function _require_unit_interval(name::AbstractString, value::T) where {T<:AbstractFloat}
    isfinite(value) && zero(T) <= value <= one(T) || throw(ArgumentError(
        "$name must be finite and in [0, 1]",
    ))
    return value
end

"""
    LocalParameterBasis(parameter_count; T=Float32, include_cell_parameters=true)

Teacher-free local parameterization for one cell transition. `raw_basis`
maps local parameters to the 46 raw Reduced-Hay parameters. `input_basis`
maps them to the 27 typed receptor inputs. Graph/contact code refreshes the
input columns from the presynaptic packet before each replay transition.

This is the explicit adapter boundary for the canonical 12D axon packet. The
learning kernel does not depend on packet lane layout; the packet/graph owner
must express each contact derivative in `input_basis`.
"""
struct LocalParameterBasis{T<:AbstractFloat}
    raw_basis::Matrix{T}
    input_basis::Matrix{T}
end

function LocalParameterBasis(
    parameter_count::Integer;
    T::Type{<:AbstractFloat}=Float32,
    include_cell_parameters::Bool=true,
)
    count = _require_positive("parameter_count", parameter_count)
    raw = zeros(T, Cell.PARAM_DIM, count)
    input = zeros(T, Cell.INPUT_DIM, count)
    if include_cell_parameters
        count >= Cell.PARAM_DIM || throw(ArgumentError(
            "including cell parameters requires at least $(Cell.PARAM_DIM) columns",
        ))
        @inbounds for parameter in 1:Cell.PARAM_DIM
            raw[parameter, parameter] = one(T)
        end
    end
    return LocalParameterBasis{T}(raw, input)
end

function _check_basis(basis::LocalParameterBasis, parameter_count::Int)
    size(basis.raw_basis) == (Cell.PARAM_DIM, parameter_count) || throw(
        DimensionMismatch("raw parameter basis has the wrong shape"),
    )
    size(basis.input_basis) == (Cell.INPUT_DIM, parameter_count) || throw(
        DimensionMismatch("input parameter basis has the wrong shape"),
    )
    return nothing
end

"""Forward sensitivities and current continuous e-prop eligibility."""
mutable struct AnalogEligibilityState{T<:AbstractFloat}
    state_sensitivity::Matrix{T}
    eligibility::Matrix{T}
    packet_eligibility::Matrix{T}
    touched::Bool
    transition_count::Int
end

function AnalogEligibilityState(
    parameter_count::Integer;
    T::Type{<:AbstractFloat}=Float32,
)
    count = _require_positive("parameter_count", parameter_count)
    return AnalogEligibilityState{T}(
        zeros(T, Cell.STATE_DIM, count),
        zeros(T, LOCAL_OBSERVATION_DIM, count),
        zeros(T, Axon.PACKET_DIM, count),
        false,
        0,
    )
end

"""Eligibility for the hard spike/event decision, separate from analog credit."""
mutable struct HardEventEligibilityState{T<:AbstractFloat}
    trace::Vector{T}
    touched::Bool
    transition_count::Int
end

function HardEventEligibilityState(
    parameter_count::Integer;
    T::Type{<:AbstractFloat}=Float32,
)
    count = _require_positive("parameter_count", parameter_count)
    return HardEventEligibilityState{T}(zeros(T, count), false, 0)
end

"""Slow utility state. It never aliases analog or hard-event gradients."""
mutable struct StructuralUtilityState{T<:AbstractFloat}
    utility::Vector{T}
    update_count::Int
end

function StructuralUtilityState(
    parameter_count::Integer;
    T::Type{<:AbstractFloat}=Float32,
)
    count = _require_positive("parameter_count", parameter_count)
    return StructuralUtilityState{T}(zeros(T, count), 0)
end

"""Fixed storage for conditional local Jacobians and forward sensitivities."""
struct EligibilityScratch{T<:AbstractFloat}
    state_jacobian::Matrix{T}
    input_jacobian::Matrix{T}
    raw_jacobian::Matrix{T}
    next_sensitivity::Matrix{T}
    direct_sensitivity::Matrix{T}
    dstate::Vector{T}
    dinput::Vector{T}
    draw::Vector{T}
    dnext::Vector{T}
    margin_sensitivity::Vector{T}
    packet_bar::Vector{T}
    packet_dnext::Vector{T}
end

function EligibilityScratch(
    parameter_count::Integer;
    T::Type{<:AbstractFloat}=Float32,
)
    count = _require_positive("parameter_count", parameter_count)
    return EligibilityScratch{T}(
        zeros(T, Cell.STATE_DIM, Cell.STATE_DIM),
        zeros(T, Cell.STATE_DIM, Cell.INPUT_DIM),
        zeros(T, Cell.STATE_DIM, Cell.PARAM_DIM),
        zeros(T, Cell.STATE_DIM, count),
        zeros(T, Cell.STATE_DIM, count),
        zeros(T, Cell.STATE_DIM),
        zeros(T, Cell.INPUT_DIM),
        zeros(T, Cell.PARAM_DIM),
        zeros(T, Cell.STATE_DIM),
        zeros(T, count),
        zeros(T, Axon.PACKET_DIM),
        zeros(T, Cell.STATE_DIM),
    )
end

function _check_learning_shapes(
    analog::AnalogEligibilityState{T},
    event::HardEventEligibilityState{T},
    basis::LocalParameterBasis{T},
    scratch::EligibilityScratch{T},
) where {T<:AbstractFloat}
    parameter_count = size(analog.state_sensitivity, 2)
    size(analog.state_sensitivity) == (Cell.STATE_DIM, parameter_count) ||
        throw(DimensionMismatch("analog state sensitivity has the wrong shape"))
    size(analog.eligibility) == (LOCAL_OBSERVATION_DIM, parameter_count) ||
        throw(DimensionMismatch("analog eligibility has the wrong shape"))
    size(analog.packet_eligibility) == (Axon.PACKET_DIM, parameter_count) ||
        throw(DimensionMismatch("axon packet eligibility has the wrong shape"))
    length(event.trace) == parameter_count || throw(DimensionMismatch(
        "hard-event eligibility has the wrong length",
    ))
    _check_basis(basis, parameter_count)
    size(scratch.next_sensitivity) == (Cell.STATE_DIM, parameter_count) ||
        throw(DimensionMismatch("eligibility scratch has the wrong parameter width"))
    return parameter_count
end

"""
Seed-fixed, non-trainable maps for global raw derivatives and local prediction
errors. The constructor uses a counter hash rather than process-global RNG, so
the same `(seed, family, cell)` is reproducible independently of call order.
"""
struct FixedLocalSignalMap{T<:AbstractFloat}
    global_feedback::Matrix{T}
    predictor_feedback::Matrix{T}
    seed::UInt64
    family::UInt32
    cell::UInt32
end

@inline function _splitmix64(value::UInt64)
    value += 0x9e3779b97f4a7c15
    value = xor(value, value >> 30) * 0xbf58476d1ce4e5b9
    value = xor(value, value >> 27) * 0x94d049bb133111eb
    return xor(value, value >> 31)
end

@inline function _rademacher(
    ::Type{T},
    seed::UInt64,
    family::UInt32,
    cell::UInt32,
    row::Int,
    column::Int,
    scale::T,
) where {T<:AbstractFloat}
    counter = xor(xor(seed, UInt64(family) << 32), UInt64(cell))
    counter = xor(counter, UInt64(row) * 0xd6e8feb86659fd93)
    counter = xor(counter, UInt64(column) * 0xa5a3564e27f8862f)
    return isodd(_splitmix64(counter)) ? scale : -scale
end

function FixedLocalSignalMap(
    output_dim::Integer,
    predictor_dim::Integer=0;
    observation_dim::Integer=LOCAL_OBSERVATION_DIM,
    seed::Integer=0x5eed,
    family::Integer=1,
    cell::Integer=1,
    scale::Real=1,
    T::Type{<:AbstractFloat}=Float32,
)
    outputs = _require_positive("output_dim", output_dim)
    predictors = Int(predictor_dim)
    predictors >= 0 || throw(ArgumentError("predictor_dim must be nonnegative"))
    observations = _require_positive("observation_dim", observation_dim)
    family >= 0 || throw(ArgumentError("family must be nonnegative"))
    cell >= 0 || throw(ArgumentError("cell must be nonnegative"))
    finite_scale = T(scale)
    isfinite(finite_scale) && finite_scale >= zero(T) || throw(ArgumentError(
        "feedback scale must be finite and nonnegative",
    ))
    global_scale = finite_scale / sqrt(T(outputs))
    predictor_scale = predictors == 0 ? zero(T) : finite_scale / sqrt(T(predictors))
    global_matrix = Matrix{T}(undef, observations, outputs)
    predictor = Matrix{T}(undef, observations, predictors)
    seed_value = UInt64(seed)
    family_value = UInt32(family)
    cell_value = UInt32(cell)
    @inbounds for column in 1:outputs, row in 1:observations
        global_matrix[row, column] = _rademacher(
            T, seed_value, family_value, cell_value, row, column, global_scale,
        )
    end
    @inbounds for column in 1:predictors, row in 1:observations
        predictor[row, column] = _rademacher(
            T, xor(seed_value, 0xc6bc279692b5c323),
            family_value, cell_value, row, column, predictor_scale,
        )
    end
    return FixedLocalSignalMap{T}(
        global_matrix, predictor, seed_value, family_value, cell_value,
    )
end

"""Compute `B * delta_raw + C * local_error` without any trainable readout."""
function project_learning_signal!(
    destination::AbstractVector{T},
    map::FixedLocalSignalMap{T},
    delta_raw::AbstractVector{T},
) where {T<:AbstractFloat}
    length(destination) == size(map.global_feedback, 1) || throw(
        DimensionMismatch("learning signal destination has the wrong length"),
    )
    length(delta_raw) == size(map.global_feedback, 2) || throw(
        DimensionMismatch("raw loss derivative has the wrong length"),
    )
    size(map.predictor_feedback, 2) == 0 || throw(ArgumentError(
        "this fixed map requires an explicit local prediction error",
    ))
    mul!(destination, map.global_feedback, delta_raw)
    return destination
end

function project_learning_signal!(
    destination::AbstractVector{T},
    map::FixedLocalSignalMap{T},
    delta_raw::AbstractVector{T},
    local_error::AbstractVector{T},
) where {T<:AbstractFloat}
    length(destination) == size(map.global_feedback, 1) || throw(
        DimensionMismatch("learning signal destination has the wrong length"),
    )
    length(delta_raw) == size(map.global_feedback, 2) || throw(
        DimensionMismatch("raw loss derivative has the wrong length"),
    )
    length(local_error) == size(map.predictor_feedback, 2) || throw(
        DimensionMismatch("local prediction error has the wrong length"),
    )
    mul!(destination, map.global_feedback, delta_raw)
    mul!(destination, map.predictor_feedback, local_error, one(T), one(T))
    return destination
end

"""Independent clocks for analog, hard-event, homeostatic and structure work."""
struct LearningSchedule
    analog_interval::Int
    hard_event_interval::Int
    homeostasis_interval::Int
    structure_interval::Int
end

function LearningSchedule(;
    analog_interval::Integer=1,
    hard_event_interval::Integer=4,
    homeostasis_interval::Integer=128,
    structure_interval::Integer=4096,
)
    return LearningSchedule(
        _require_positive("analog_interval", analog_interval),
        _require_positive("hard_event_interval", hard_event_interval),
        _require_positive("homeostasis_interval", homeostasis_interval),
        _require_positive("structure_interval", structure_interval),
    )
end

struct DuePlasticityClocks
    analog::Bool
    hard_event::Bool
    homeostasis::Bool
    structure::Bool
end

mutable struct LearningClockState
    update::Int
    analog_ticks::Int
    hard_event_ticks::Int
    homeostasis_ticks::Int
    structure_ticks::Int
end

LearningClockState() = LearningClockState(0, 0, 0, 0, 0)

function advance_clocks!(clock::LearningClockState, schedule::LearningSchedule)
    clock.update == typemax(Int) && throw(OverflowError("learning clock overflow"))
    clock.update += 1
    analog = mod(clock.update, schedule.analog_interval) == 0
    hard_event = mod(clock.update, schedule.hard_event_interval) == 0
    homeostasis = mod(clock.update, schedule.homeostasis_interval) == 0
    structure = mod(clock.update, schedule.structure_interval) == 0
    clock.analog_ticks += analog
    clock.hard_event_ticks += hard_event
    clock.homeostasis_ticks += homeostasis
    clock.structure_ticks += structure
    return DuePlasticityClocks(analog, hard_event, homeostasis, structure)
end

@enum ReplayPhase::UInt8 begin
    COLLECTING_FORWARD = 0x01
    DELTAS_SEALED = 0x02
    REPLAYING = 0x03
    REPLAY_COMPLETE = 0x04
end

"""
Fixed-memory two-pass boundary. Pass 1 records only candidate trajectory
digests; it has no teacher/target field. The caller computes ListNet after all
candidates finish and seals only the resulting raw 22D derivatives. Replay
must present the same digest, making the hard trajectory deterministic.
"""
mutable struct TwoPassListNetReplay{T<:AbstractFloat}
    raw_delta::Matrix{T}
    forward_digest::Vector{UInt64}
    recorded::BitVector
    replayed::BitVector
    candidate_count::Int
    phase::ReplayPhase
end

function TwoPassListNetReplay(
    output_dim::Integer,
    capacity::Integer;
    T::Type{<:AbstractFloat}=Float32,
)
    outputs = _require_positive("output_dim", output_dim)
    candidates = _require_positive("capacity", capacity)
    return TwoPassListNetReplay{T}(
        zeros(T, outputs, candidates),
        zeros(UInt64, candidates),
        falses(candidates),
        falses(candidates),
        0,
        COLLECTING_FORWARD,
    )
end

function reset_replay!(replay::TwoPassListNetReplay)
    fill!(replay.raw_delta, zero(eltype(replay.raw_delta)))
    fill!(replay.forward_digest, 0)
    fill!(replay.recorded, false)
    fill!(replay.replayed, false)
    replay.candidate_count = 0
    replay.phase = COLLECTING_FORWARD
    return replay
end

function record_teacher_free_forward!(
    replay::TwoPassListNetReplay,
    candidate::Integer,
    trajectory_digest::Integer,
)
    replay.phase == COLLECTING_FORWARD || throw(ArgumentError(
        "teacher-free forward records are accepted only in pass 1",
    ))
    selected = Int(candidate)
    checkbounds(replay.recorded, selected)
    replay.recorded[selected] && throw(ArgumentError(
        "candidate $selected was already recorded",
    ))
    replay.recorded[selected] = true
    trajectory_digest >= 0 || throw(ArgumentError(
        "trajectory digest must be nonnegative",
    ))
    replay.forward_digest[selected] = UInt64(trajectory_digest)
    replay.candidate_count = max(replay.candidate_count, selected)
    return replay
end

function seal_listnet_deltas!(
    replay::TwoPassListNetReplay{T},
    delta_raw::AbstractMatrix{T},
    candidate_count::Integer,
) where {T<:AbstractFloat}
    replay.phase == COLLECTING_FORWARD || throw(ArgumentError(
        "ListNet derivatives can be sealed only after pass 1",
    ))
    count = Int(candidate_count)
    1 <= count <= size(replay.raw_delta, 2) || throw(BoundsError(
        axes(replay.raw_delta, 2), count,
    ))
    replay.candidate_count == count || throw(ArgumentError(
        "recorded candidate range does not match candidate_count",
    ))
    all(@view(replay.recorded[1:count])) || throw(ArgumentError(
        "all candidates must finish before ListNet derivatives are sealed",
    ))
    size(delta_raw) == (size(replay.raw_delta, 1), count) || throw(
        DimensionMismatch("raw ListNet derivative matrix has the wrong shape"),
    )
    all(isfinite, delta_raw) || throw(ArgumentError(
        "raw ListNet derivatives must be finite",
    ))
    copyto!(@view(replay.raw_delta[:, 1:count]), delta_raw)
    replay.phase = DELTAS_SEALED
    return replay
end

function copy_replay_delta!(
    destination::AbstractVector{T},
    replay::TwoPassListNetReplay{T},
    candidate::Integer,
    trajectory_digest::Integer,
) where {T<:AbstractFloat}
    replay.phase in (DELTAS_SEALED, REPLAYING) || throw(ArgumentError(
        "candidate replay requires sealed ListNet derivatives",
    ))
    selected = Int(candidate)
    1 <= selected <= replay.candidate_count || throw(BoundsError(
        1:replay.candidate_count, selected,
    ))
    length(destination) == size(replay.raw_delta, 1) || throw(
        DimensionMismatch("replay derivative destination has the wrong length"),
    )
    replay.replayed[selected] && throw(ArgumentError(
        "candidate $selected was replayed more than once",
    ))
    trajectory_digest >= 0 || throw(ArgumentError(
        "trajectory digest must be nonnegative",
    ))
    replay.forward_digest[selected] == UInt64(trajectory_digest) || throw(ArgumentError(
        "candidate $selected replay trajectory differs from pass 1",
    ))
    copyto!(destination, @view(replay.raw_delta[:, selected]))
    replay.replayed[selected] = true
    replay.phase = REPLAYING
    return destination
end

function finish_replay!(replay::TwoPassListNetReplay)
    replay.phase == REPLAYING || throw(ArgumentError(
        "replay cannot finish before at least one candidate is replayed",
    ))
    all(@view(replay.replayed[1:replay.candidate_count])) || throw(ArgumentError(
        "all candidates must be replayed exactly once",
    ))
    replay.phase = REPLAY_COMPLETE
    return replay
end

function reset_eligibility!(
    analog::AnalogEligibilityState,
    event::HardEventEligibilityState,
)
    fill!(analog.state_sensitivity, 0)
    fill!(analog.eligibility, 0)
    fill!(analog.packet_eligibility, 0)
    analog.touched = false
    analog.transition_count = 0
    fill!(event.trace, 0)
    event.touched = false
    event.transition_count = 0
    return nothing
end

"""Write the local 47D continuous observation; teacher state is not an input."""
function continuous_observation!(
    destination::AbstractVector{T},
    previous_state::AbstractVector{T},
    next_state::AbstractVector{T},
    cache::Cell.CellParameterCache{T},
) where {T<:AbstractFloat}
    length(destination) == LOCAL_OBSERVATION_DIM || throw(DimensionMismatch(
        "continuous observation has the wrong length",
    ))
    length(previous_state) == Cell.STATE_DIM || throw(DimensionMismatch(
        "previous state has the wrong length",
    ))
    length(next_state) == Cell.STATE_DIM || throw(DimensionMismatch(
        "next state has the wrong length",
    ))
    @inbounds for state in 1:(Cell.N_COMPARTMENTS * Cell.COMPARTMENT_STATE_DIM)
        destination[state] = next_state[state]
    end
    destination[MARGIN_OBSERVATION] = Cell.spike_margin_from_transition(
        previous_state, next_state, cache,
    )
    destination[ADAPTATION_OBSERVATION] = next_state[Cell.ADAPTATION_INDEX]
    return destination
end

function _conditional_jacobians!(
    scratch::EligibilityScratch{T},
    previous_state::AbstractVector{T},
    input::AbstractVector{T},
    cache::Cell.CellParameterCache{T},
    derivative_cache::Cell.CellParameterDerivativeCache{T},
    next_state::AbstractVector{T},
) where {T<:AbstractFloat}
    fill!(scratch.state_jacobian, zero(T))
    fill!(scratch.input_jacobian, zero(T))
    fill!(scratch.raw_jacobian, zero(T))
    @inbounds for output in 1:(Cell.STATE_DIM - 1)
        fill!(scratch.dnext, zero(T))
        scratch.dnext[output] = one(T)
        Cell.cell_step_conditional_pullback!(
            scratch.dstate,
            scratch.dinput,
            scratch.draw,
            previous_state,
            input,
            cache,
            derivative_cache,
            next_state,
            scratch.dnext,
        )
        copyto!(@view(scratch.state_jacobian[output, :]), scratch.dstate)
        copyto!(@view(scratch.input_jacobian[output, :]), scratch.dinput)
        copyto!(@view(scratch.raw_jacobian[output, :]), scratch.draw)
    end
    # Hard spikes are fixed observations in the conditional analog model.
    fill!(@view(scratch.state_jacobian[Cell.SPIKE_INDEX, :]), zero(T))
    fill!(@view(scratch.input_jacobian[Cell.SPIKE_INDEX, :]), zero(T))
    fill!(@view(scratch.raw_jacobian[Cell.SPIKE_INDEX, :]), zero(T))
    return scratch
end

function _margin_sensitivity!(
    destination::AbstractVector{T},
    scratch::EligibilityScratch{T},
    previous_sensitivity::AbstractMatrix{T},
    basis::LocalParameterBasis{T},
    previous_state::AbstractVector{T},
    input::AbstractVector{T},
    cache::Cell.CellParameterCache{T},
    derivative_cache::Cell.CellParameterDerivativeCache{T},
    next_state::AbstractVector{T},
) where {T<:AbstractFloat}
    fill!(scratch.dnext, zero(T))
    Cell.cell_step_conditional_pullback!(
        scratch.dstate,
        scratch.dinput,
        scratch.draw,
        previous_state,
        input,
        cache,
        derivative_cache,
        next_state,
        scratch.dnext,
        zero(T),
        zero(T),
        one(T),
    )
    mul!(destination, transpose(previous_sensitivity), scratch.dstate)
    mul!(destination, transpose(basis.raw_basis), scratch.draw, one(T), one(T))
    mul!(destination, transpose(basis.input_basis), scratch.dinput, one(T), one(T))
    return destination
end

function _packet_eligibility!(
    destination::AbstractMatrix{T},
    scratch::EligibilityScratch{T},
    next_sensitivity::AbstractMatrix{T},
    previous_state::AbstractVector{T},
    next_state::AbstractVector{T},
    cache::Cell.CellParameterCache{T},
) where {T<:AbstractFloat}
    size(destination) == (Axon.PACKET_DIM, size(next_sensitivity, 2)) ||
        throw(DimensionMismatch("axon packet eligibility has the wrong shape"))
    @inbounds for lane in 1:Axon.PACKET_DIM
        fill!(scratch.packet_bar, zero(T))
        scratch.packet_bar[lane] = one(T)
        margin_cotangent = Axon.axon_packet_pullback!(
            scratch.packet_dnext,
            scratch.packet_bar,
            previous_state,
            next_state,
            cache,
        )
        for parameter in axes(next_sensitivity, 2)
            value = margin_cotangent * scratch.margin_sensitivity[parameter]
            for state in 1:(Cell.STATE_DIM - 1)
                value = muladd(
                    scratch.packet_dnext[state],
                    next_sensitivity[state, parameter],
                    value,
                )
            end
            destination[lane, parameter] = value
        end
    end
    return destination
end

"""
    accumulate_active_apical_transition!(...; touched, event_decay)

Generate teacher-free multi-compartment eligibility for one recorded hard
trajectory transition. The conditional analog trace never differentiates the
hard threshold. Hard-event eligibility is generated separately from the exact
pre-reset margin and a bounded triangular surrogate. `event_decay=0` implements
the canonical instantaneous `T = H'(margin) * dmargin/dp`; a nonzero value is
an explicit slower synaptic tag rather than an implicit part of analog credit.

If `touched=false`, no state or trace is changed. This makes an unvisited cell
strictly different from a mandatory-sweep cell whose recorded hard spike is 0.
"""
function accumulate_active_apical_transition!(
    analog::AnalogEligibilityState{T},
    event::HardEventEligibilityState{T},
    scratch::EligibilityScratch{T},
    basis::LocalParameterBasis{T},
    previous_state::AbstractVector{T},
    input::AbstractVector{T},
    raw_parameters::AbstractVector{T},
    next_state::AbstractVector{T};
    touched::Bool,
    event_decay::T=zero(T),
) where {T<:AbstractFloat}
    parameter_count = _check_learning_shapes(analog, event, basis, scratch)
    length(previous_state) == Cell.STATE_DIM || throw(DimensionMismatch(
        "previous state has the wrong length",
    ))
    length(input) == Cell.INPUT_DIM || throw(DimensionMismatch(
        "cell input has the wrong length",
    ))
    length(raw_parameters) == Cell.PARAM_DIM || throw(DimensionMismatch(
        "raw cell parameters have the wrong length",
    ))
    length(next_state) == Cell.STATE_DIM || throw(DimensionMismatch(
        "next state has the wrong length",
    ))
    _require_unit_interval("event_decay", event_decay)
    touched || return false
    all(isfinite, previous_state) && all(isfinite, input) &&
        all(isfinite, raw_parameters) && all(isfinite, next_state) ||
        throw(ArgumentError("cell transition must be finite"))

    cache, derivative_cache = Cell.parameter_caches(raw_parameters)
    _conditional_jacobians!(
        scratch, previous_state, input, cache, derivative_cache, next_state,
    )
    previous_sensitivity = analog.state_sensitivity
    _margin_sensitivity!(
        scratch.margin_sensitivity,
        scratch,
        previous_sensitivity,
        basis,
        previous_state,
        input,
        cache,
        derivative_cache,
        next_state,
    )

    mul!(
        scratch.next_sensitivity,
        scratch.state_jacobian,
        previous_sensitivity,
    )
    mul!(scratch.direct_sensitivity, scratch.raw_jacobian, basis.raw_basis)
    scratch.next_sensitivity .+= scratch.direct_sensitivity
    mul!(scratch.direct_sensitivity, scratch.input_jacobian, basis.input_basis)
    scratch.next_sensitivity .+= scratch.direct_sensitivity
    fill!(@view(scratch.next_sensitivity[Cell.SPIKE_INDEX, :]), zero(T))

    @inbounds begin
        continuous_count = Cell.N_COMPARTMENTS * Cell.COMPARTMENT_STATE_DIM
        for parameter in 1:parameter_count
            for state in 1:continuous_count
                analog.eligibility[state, parameter] =
                    scratch.next_sensitivity[state, parameter]
            end
            analog.eligibility[MARGIN_OBSERVATION, parameter] =
                scratch.margin_sensitivity[parameter]
            analog.eligibility[ADAPTATION_OBSERVATION, parameter] =
                scratch.next_sensitivity[Cell.ADAPTATION_INDEX, parameter]
        end
    end
    _packet_eligibility!(
        analog.packet_eligibility,
        scratch,
        scratch.next_sensitivity,
        previous_state,
        next_state,
        cache,
    )
    copyto!(analog.state_sensitivity, scratch.next_sensitivity)

    margin = Cell.spike_margin_from_transition(
        previous_state, next_state, cache,
    )
    surrogate = Cell.spike_surrogate_derivative(margin)
    @inbounds for parameter in 1:parameter_count
        event.trace[parameter] = muladd(
            event_decay,
            event.trace[parameter],
            surrogate * scratch.margin_sensitivity[parameter],
        )
    end
    analog.touched = true
    analog.transition_count += 1
    event.touched = true
    event.transition_count += 1
    return true
end

function accumulate_packet_gradient!(
    gradient::AbstractVector{T},
    learning_signal::AbstractVector{T},
    analog::AnalogEligibilityState{T};
    scale::T=one(T),
    due::Bool=true,
) where {T<:AbstractFloat}
    length(gradient) == size(analog.packet_eligibility, 2) || throw(
        DimensionMismatch("packet gradient has the wrong length"),
    )
    length(learning_signal) == size(analog.packet_eligibility, 1) || throw(
        DimensionMismatch("packet learning signal has the wrong length"),
    )
    isfinite(scale) || throw(ArgumentError("packet gradient scale must be finite"))
    if due && analog.touched && !iszero(scale)
        mul!(
            gradient,
            transpose(analog.packet_eligibility),
            learning_signal,
            scale,
            one(T),
        )
    end
    return gradient
end

function accumulate_analog_gradient!(
    gradient::AbstractVector{T},
    learning_signal::AbstractVector{T},
    analog::AnalogEligibilityState{T};
    scale::T=one(T),
    due::Bool=true,
) where {T<:AbstractFloat}
    length(gradient) == size(analog.eligibility, 2) || throw(
        DimensionMismatch("analog gradient has the wrong length"),
    )
    length(learning_signal) == size(analog.eligibility, 1) || throw(
        DimensionMismatch("analog learning signal has the wrong length"),
    )
    isfinite(scale) || throw(ArgumentError("analog gradient scale must be finite"))
    if due && analog.touched && !iszero(scale)
        mul!(gradient, transpose(analog.eligibility), learning_signal, scale, one(T))
    end
    return gradient
end

function accumulate_hard_event_gradient!(
    gradient::AbstractVector{T},
    control_signal::T,
    event::HardEventEligibilityState{T};
    scale::T=one(T),
    due::Bool=true,
) where {T<:AbstractFloat}
    length(gradient) == length(event.trace) || throw(DimensionMismatch(
        "hard-event gradient has the wrong length",
    ))
    isfinite(control_signal) && isfinite(scale) || throw(ArgumentError(
        "hard-event control signal and scale must be finite",
    ))
    if due && event.touched && !iszero(control_signal) && !iszero(scale)
        coefficient = control_signal * scale
        @inbounds for parameter in eachindex(gradient, event.trace)
            gradient[parameter] = muladd(
                coefficient, event.trace[parameter], gradient[parameter],
            )
        end
    end
    return gradient
end

function update_structural_utility!(
    state::StructuralUtilityState{T},
    learning_signal::AbstractVector{T},
    analog::AnalogEligibilityState{T};
    decay::T=T(0.999),
    due::Bool=true,
) where {T<:AbstractFloat}
    length(state.utility) == size(analog.eligibility, 2) || throw(
        DimensionMismatch("structural utility has the wrong length"),
    )
    length(learning_signal) == size(analog.eligibility, 1) || throw(
        DimensionMismatch("structural learning signal has the wrong length"),
    )
    _require_unit_interval("utility decay", decay)
    due || return state
    analog.touched || return state
    signal_norm = norm(learning_signal)
    @inbounds for parameter in eachindex(state.utility)
        eligibility = @view analog.eligibility[:, parameter]
        eligibility_norm = norm(eligibility)
        denominator = signal_norm * eligibility_norm
        contribution = iszero(denominator) ? zero(T) :
            abs(dot(learning_signal, eligibility)) /
            (denominator + eps(T))
        state.utility[parameter] = muladd(
            decay, state.utility[parameter], contribution,
        )
    end
    state.update_count += 1
    return state
end

function update_packet_structural_utility!(
    state::StructuralUtilityState{T},
    learning_signal::AbstractVector{T},
    analog::AnalogEligibilityState{T};
    decay::T=T(0.999),
    due::Bool=true,
) where {T<:AbstractFloat}
    length(state.utility) == size(analog.packet_eligibility, 2) || throw(
        DimensionMismatch("structural utility has the wrong length"),
    )
    length(learning_signal) == size(analog.packet_eligibility, 1) || throw(
        DimensionMismatch("packet structural learning signal has the wrong length"),
    )
    _require_unit_interval("utility decay", decay)
    due || return state
    analog.touched || return state
    signal_norm = norm(learning_signal)
    @inbounds for parameter in eachindex(state.utility)
        eligibility = @view analog.packet_eligibility[:, parameter]
        eligibility_norm = norm(eligibility)
        denominator = signal_norm * eligibility_norm
        contribution = iszero(denominator) ? zero(T) :
            abs(dot(learning_signal, eligibility)) /
            (denominator + eps(T))
        state.utility[parameter] = muladd(
            decay, state.utility[parameter], contribution,
        )
    end
    state.update_count += 1
    return state
end

end # module CanonicalLocalLearning
