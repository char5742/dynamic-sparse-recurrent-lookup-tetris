module CanonicalLocalLearning

using LinearAlgebra
using SHA
using ..ActiveApicalCell
using ..DendriticAxonPacket

const Cell = ActiveApicalCell
const Axon = DendriticAxonPacket

export LOCAL_OBSERVATION_DIM,
       ChronologicalTransitionLink,
       ContractedLocalAdjoint,
       ContractedAdjointArena,
       ContractedAdjointScratch,
       CausalEventControl,
       StructuralUtilityState,
       FixedLocalSignalMap,
       PlasticityConfig,
       LocalLearningConfig,
       LearningSchedule,
       LearningClockState,
       DuePlasticityClocks,
       ReplayPhase,
       TwoPassListNetReplay,
       reset_replay!,
       begin_local_adjoint!,
       add_terminal_seed!,
       finish_local_adjoint!,
       reset_adjoint_arena!,
       preview_clocks,
       commit_clocks!,
       advance_clocks!,
       record_teacher_free_forward!,
       seal_listnet_deltas!,
       copy_replay_delta!,
       finish_replay!,
       project_learning_signal!,
       contract_replayed_transition!,
       raw_parameter_cotangent,
       input_cotangent,
       update_structural_utility!,
       config_summary,
       config_fingerprint,
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
    ChronologicalTransitionLink(record, predecessor, packet_version)

Immutable identity of one replayed cell transition. `record` is the exact
post-transition state version and `predecessor` is the version consumed by the
forward cell kernel (`0` denotes the anatomical/common root). `packet_version`
identifies the packet produced by this exact transition; graph/contact replay
must never substitute a cell's final packet.
"""
struct ChronologicalTransitionLink
    record::Int
    predecessor::Int
    packet_version::Int
    function ChronologicalTransitionLink(
        record::Int,
        predecessor::Int,
        packet_version::Int,
    )
        record > 0 || throw(ArgumentError(
            "transition record must be positive",
        ))
        predecessor >= 0 || throw(ArgumentError(
            "transition predecessor must be nonnegative",
        ))
        predecessor != record || throw(ArgumentError(
            "a transition cannot be its own predecessor",
        ))
        packet_version > 0 || throw(ArgumentError(
            "packet version must be positive",
        ))
        return new(record, predecessor, packet_version)
    end
end

function ChronologicalTransitionLink(
    record::Integer,
    predecessor::Integer,
    packet_version::Integer=record,
)
    return ChronologicalTransitionLink(
        Int(record), Int(predecessor), Int(packet_version),
    )
end

"""
Reverse local-adjoint state for one cell in one alternative world.

Only the 48-dimensional cell-state cotangent is retained. There is no
parameter-width dimension: the state is therefore independent of the number
of shared semantic/event contacts. The generic primitive supports an explicit
terminal seed, but the canonical Graph stops candidate alternative worlds at
`initial_core`; it does not propagate their 48D root cotangents into common.
"""
mutable struct ContractedLocalAdjoint{T<:AbstractFloat}
    state_bar::Vector{T}
    expected_record::Int
    active::Bool
    touched::Bool
    visited_transition_count::Int
    conditional_pullback_count::Int
end

function ContractedLocalAdjoint(; T::Type{<:AbstractFloat}=Float32)
    return ContractedLocalAdjoint{T}(
        zeros(T, Cell.STATE_DIM), 0, false, false, 0, 0,
    )
end

"""
Generation-stamped production arena for every core cell owned by one worker.

The `48 × node_count` cotangent slab is contiguous and parameter-width
independent. Metadata are parallel primitive arrays, so replay addresses a
cell by integer column and creates neither a per-cell object nor a SubArray
view. `reset_adjoint_arena!` advances a generation instead of clearing the
whole slab.
"""
mutable struct ContractedAdjointArena{T<:AbstractFloat}
    state_bar::Matrix{T}
    expected_record::Vector{Int}
    active::Vector{Bool}
    touched::Vector{Bool}
    visited_transition_count::Vector{Int}
    conditional_pullback_count::Vector{Int}
    generation::Vector{UInt32}
    current_generation::UInt32
end

function ContractedAdjointArena(
    node_count::Integer;
    T::Type{<:AbstractFloat}=Float32,
)
    count = _require_positive("adjoint arena node_count", node_count)
    return ContractedAdjointArena{T}(
        zeros(T, Cell.STATE_DIM, count),
        zeros(Int, count),
        fill(false, count),
        fill(false, count),
        zeros(Int, count),
        zeros(Int, count),
        zeros(UInt32, count),
        UInt32(1),
    )
end

function reset_adjoint_arena!(arena::ContractedAdjointArena)
    if arena.current_generation == typemax(UInt32)
        fill!(arena.generation, UInt32(0))
        arena.current_generation = UInt32(1)
    else
        arena.current_generation += UInt32(1)
    end
    return arena
end

@inline function _check_arena_node(arena::ContractedAdjointArena, node::Integer)
    selected = Int(node)
    checkbounds(arena.expected_record, selected)
    return selected
end

@inline function _arena_is_current(
    arena::ContractedAdjointArena,
    node::Int,
)
    return arena.generation[node] == arena.current_generation
end

"""
Fixed-width scratch for one contracted transition.

The dimensions are properties of the Reduced-Hay cell/packet only. In
particular, no array scales with the number of learnable parameters.
`draw` and `dinput` are the transition's direct contributions; the graph owner
must reduce `draw` into shared cell parameters and scatter `dinput` through
the exact chronological deposit records without propagating cotangents into
the source cell.
"""
struct ContractedAdjointScratch{T<:AbstractFloat}
    dstate::Vector{T}
    dinput::Vector{T}
    draw::Vector{T}
    dnext::Vector{T}
    packet_bar::Vector{T}
    packet_dnext::Vector{T}
end

function ContractedAdjointScratch(; T::Type{<:AbstractFloat}=Float32)
    return ContractedAdjointScratch{T}(
        zeros(T, Cell.STATE_DIM),
        zeros(T, Cell.INPUT_DIM),
        zeros(T, Cell.PARAM_DIM),
        zeros(T, Cell.STATE_DIM),
        zeros(T, Axon.PACKET_DIM),
        zeros(T, Cell.STATE_DIM),
    )
end

"""
Explicit hard-event control boundary.

The initial canonical model has no causal continuation/energy estimator, so
only `CausalEventControl()` is accepted by the contracted kernel. A nonzero or
connected control fails closed instead of silently reusing the analog ListNet
signal. Once a fused two-lane cell VJP exists, a connected estimator can be
implemented without merging the analog and hard-event clocks.
"""
struct CausalEventControl{T<:AbstractFloat}
    advantage::T
    connected::Bool
end

CausalEventControl(; T::Type{<:AbstractFloat}=Float32) =
    CausalEventControl{T}(zero(T), false)

function CausalEventControl(
    advantage::T;
    connected::Bool,
) where {T<:AbstractFloat}
    isfinite(advantage) || throw(ArgumentError(
        "hard-event advantage must be finite",
    ))
    return CausalEventControl{T}(advantage, connected)
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
    predictor_scale::Real=scale,
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
    finite_predictor_scale = T(predictor_scale)
    isfinite(finite_predictor_scale) && finite_predictor_scale >= zero(T) ||
        throw(ArgumentError(
            "predictor feedback scale must be finite and nonnegative",
        ))
    predictor_lane_scale = predictors == 0 ? zero(T) :
        finite_predictor_scale / sqrt(T(predictors))
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
            family_value, cell_value, row, column, predictor_lane_scale,
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

"""
One immutable owner of slow biological plasticity controls. The canonical
information spine is not rewired; these settings apply only to optional event
contacts and intrinsic cell homeostasis. `structure_enabled=false` is the
initial canonical setting even though utility may be collected in shadow.
"""
struct PlasticityConfig
    firing_ema_decay::Float32
    target_rate_min::Float32
    target_rate_max::Float32
    threshold_homeostasis_step::Float32
    adaptation_homeostasis_step::Float32
    synaptic_scaling_rate::Float32
    conductance_floor::Float32
    conductance_ceiling::Float32
    structure_enabled::Bool
    utility_decay::Float32
    connection_cost::Float32
    max_swaps_per_node::Int
end

function PlasticityConfig(;
    firing_ema_decay::Real=0.99,
    target_rate_min::Real=0.002,
    target_rate_max::Real=0.25,
    threshold_homeostasis_step::Real=0.04,
    adaptation_homeostasis_step::Real=0.02,
    synaptic_scaling_rate::Real=0.05,
    conductance_floor::Real=1.0e-4,
    conductance_ceiling::Real=4.0,
    structure_enabled::Bool=false,
    utility_decay::Real=0.999,
    connection_cost::Real=1.0e-6,
    max_swaps_per_node::Integer=1,
)
    ema = Float32(firing_ema_decay)
    minimum_rate = Float32(target_rate_min)
    maximum_rate = Float32(target_rate_max)
    threshold_step = Float32(threshold_homeostasis_step)
    adaptation_step = Float32(adaptation_homeostasis_step)
    scaling_rate = Float32(synaptic_scaling_rate)
    floor = Float32(conductance_floor)
    ceiling = Float32(conductance_ceiling)
    decay = Float32(utility_decay)
    cost = Float32(connection_cost)
    isfinite(ema) && 0.0f0 <= ema < 1.0f0 || throw(ArgumentError(
        "firing EMA decay must be finite and in [0, 1)",
    ))
    isfinite(minimum_rate) && isfinite(maximum_rate) &&
        0.0f0 <= minimum_rate < maximum_rate <= 1.0f0 || throw(ArgumentError(
            "target firing-rate range must satisfy 0 <= min < max <= 1",
        ))
    isfinite(threshold_step) && threshold_step >= 0.0f0 || throw(ArgumentError(
        "threshold homeostasis step must be finite and nonnegative",
    ))
    isfinite(adaptation_step) && adaptation_step >= 0.0f0 || throw(ArgumentError(
        "adaptation homeostasis step must be finite and nonnegative",
    ))
    isfinite(scaling_rate) && scaling_rate >= 0.0f0 || throw(ArgumentError(
        "synaptic scaling rate must be finite and nonnegative",
    ))
    isfinite(floor) && isfinite(ceiling) && 0.0f0 < floor < ceiling ||
        throw(ArgumentError(
            "physical conductance bounds must satisfy 0 < floor < ceiling",
        ))
    isfinite(decay) && 0.0f0 <= decay < 1.0f0 || throw(ArgumentError(
        "utility decay must be finite and in [0, 1)",
    ))
    isfinite(cost) && cost >= 0.0f0 || throw(ArgumentError(
        "connection cost must be finite and nonnegative",
    ))
    0 <= max_swaps_per_node <= 1 || throw(ArgumentError(
        "canonical structural plasticity permits at most one swap per node",
    ))
    return PlasticityConfig(
        ema,
        minimum_rate,
        maximum_rate,
        threshold_step,
        adaptation_step,
        scaling_rate,
        floor,
        ceiling,
        structure_enabled,
        decay,
        cost,
        Int(max_swaps_per_node),
    )
end

"""
The single checkpointed owner of canonical local-learning controls.

`eligibility_decay` applies only to the optional hard-event synaptic tag. The
continuous multi-compartment e-prop state already contains the physical Hay
decays through its local Jacobian and is never silently decayed a second time.
The factorized local objective is the explicit sum of the 47-dimensional
continuous term and the 12-dimensional packet term. Accordingly,
`utility_mode` is either `:combined`, which collects the absolute per-use
contribution of that same summed objective, or `:none`, which disables utility
collection. Utility collection may run in shadow while
`plasticity.structure_enabled` remains false.
"""
struct LocalLearningConfig
    schedule::LearningSchedule
    feedback_seed::UInt64
    feedback_scale::Float32
    predictor_scale::Float32
    predictor_dim::Int
    eligibility_decay::Float32
    analog_multiplier::Float32
    hard_event_multiplier::Float32
    utility_mode::Symbol
    plasticity::PlasticityConfig
end


function LocalLearningConfig(;
    schedule::LearningSchedule=LearningSchedule(),
    feedback_seed::Integer=0x4445434f4c4c4532,
    feedback_scale::Real=1.0,
    predictor_scale::Real=0.0,
    predictor_dim::Integer=0,
    eligibility_decay::Real=0.0,
    analog_multiplier::Real=1.0,
    # Fail closed until a causal continuation/energy estimator and fused
    # analog/event cell VJP are connected.
    hard_event_multiplier::Real=0.0,
    utility_mode::Symbol=:combined,
    plasticity::PlasticityConfig=PlasticityConfig(),
)
    feedback_seed >= 0 || throw(ArgumentError(
        "feedback seed must be nonnegative",
    ))
    feedback = Float32(feedback_scale)
    predictor = Float32(predictor_scale)
    decay = Float32(eligibility_decay)
    analog = Float32(analog_multiplier)
    hard_event = Float32(hard_event_multiplier)
    predictor_dim >= 0 || throw(ArgumentError(
        "predictor dimension must be nonnegative",
    ))
    isfinite(feedback) && feedback >= 0.0f0 || throw(ArgumentError(
        "feedback scale must be finite and nonnegative",
    ))
    isfinite(predictor) && predictor >= 0.0f0 || throw(ArgumentError(
        "predictor scale must be finite and nonnegative",
    ))
    isfinite(decay) && 0.0f0 <= decay < 1.0f0 || throw(ArgumentError(
        "eligibility decay must be finite and in [0, 1)",
    ))
    isfinite(analog) && analog >= 0.0f0 || throw(ArgumentError(
        "analog multiplier must be finite and nonnegative",
    ))
    isfinite(hard_event) && hard_event >= 0.0f0 || throw(ArgumentError(
        "hard-event multiplier must be finite and nonnegative",
    ))
    utility_mode in (:combined, :none) || throw(ArgumentError(
        "utility mode must be :combined or :none",
    ))
    return LocalLearningConfig(
        schedule,
        UInt64(feedback_seed),
        feedback,
        predictor,
        Int(predictor_dim),
        decay,
        analog,
        hard_event,
        utility_mode,
        plasticity,
    )
end

function FixedLocalSignalMap(
    output_dim::Integer,
    config::LocalLearningConfig;
    observation_dim::Integer=LOCAL_OBSERVATION_DIM,
    family::Integer=1,
    cell::Integer=1,
    T::Type{<:AbstractFloat}=Float32,
)
    return FixedLocalSignalMap(
        output_dim,
        config.predictor_dim;
        observation_dim=observation_dim,
        seed=config.feedback_seed,
        family=family,
        cell=cell,
        scale=config.feedback_scale,
        predictor_scale=config.predictor_scale,
        T=T,
    )
end

function config_summary(config::LocalLearningConfig)
    schedule = config.schedule
    plasticity = config.plasticity
    values = (
        "analog_interval=$(schedule.analog_interval)",
        "hard_event_interval=$(schedule.hard_event_interval)",
        "homeostasis_interval=$(schedule.homeostasis_interval)",
        "structure_interval=$(schedule.structure_interval)",
        "feedback_seed=$(config.feedback_seed)",
        "feedback_scale=$(config.feedback_scale)",
        "predictor_scale=$(config.predictor_scale)",
        "predictor_dim=$(config.predictor_dim)",
        "eligibility_decay=$(config.eligibility_decay)",
        "analog_multiplier=$(config.analog_multiplier)",
        "hard_event_multiplier=$(config.hard_event_multiplier)",
        "utility_mode=$(config.utility_mode)",
        "firing_ema_decay=$(plasticity.firing_ema_decay)",
        "target_rate_min=$(plasticity.target_rate_min)",
        "target_rate_max=$(plasticity.target_rate_max)",
        "threshold_homeostasis_step=$(plasticity.threshold_homeostasis_step)",
        "adaptation_homeostasis_step=$(plasticity.adaptation_homeostasis_step)",
        "synaptic_scaling_rate=$(plasticity.synaptic_scaling_rate)",
        "conductance_floor=$(plasticity.conductance_floor)",
        "conductance_ceiling=$(plasticity.conductance_ceiling)",
        "structure_enabled=$(plasticity.structure_enabled)",
        "utility_decay=$(plasticity.utility_decay)",
        "connection_cost=$(plasticity.connection_cost)",
        "max_swaps_per_node=$(plasticity.max_swaps_per_node)",
    )
    return join(values, ' ')
end

config_fingerprint(config::LocalLearningConfig) =
    bytes2hex(sha256(config_summary(config)))

function Base.show(io::IO, config::LocalLearningConfig)
    print(io, "LocalLearningConfig(", config_summary(config), ')')
end

struct DuePlasticityClocks
    analog::Bool
    hard_event::Bool
    homeostasis::Bool
    structure::Bool
    expected_update::Int
    expected_ticks::NTuple{4,Int}
    schedule_intervals::NTuple{4,Int}
end

# Zero-token sentinel for initialization/display only. A transactional commit
# accepts only a value returned by `preview_clocks` for the current clock.
DuePlasticityClocks(
    analog::Bool,
    hard_event::Bool,
    homeostasis::Bool,
    structure::Bool,
) = DuePlasticityClocks(
    analog,
    hard_event,
    homeostasis,
    structure,
    0,
    (0, 0, 0, 0),
    (0, 0, 0, 0),
)

mutable struct LearningClockState
    update::Int
    analog_ticks::Int
    hard_event_ticks::Int
    homeostasis_ticks::Int
    structure_ticks::Int
end

LearningClockState() = LearningClockState(0, 0, 0, 0, 0)

@inline function _checked_tick(current::Int, due::Bool, name::AbstractString)
    if due && current == typemax(Int)
        throw(OverflowError("$name learning tick overflow"))
    end
    return current + Int(due)
end

"""
    preview_clocks(clock, schedule) -> DuePlasticityClocks

Purely compute the due mask and transaction token for the next successful
update. The clock and all tick counters remain bit-identical. Every overflow
that a matching commit could encounter is checked here before mutation.
"""
function preview_clocks(clock::LearningClockState, schedule::LearningSchedule)
    clock.update == typemax(Int) && throw(OverflowError("learning clock overflow"))
    next_update = clock.update + 1
    analog = mod(next_update, schedule.analog_interval) == 0
    hard_event = mod(next_update, schedule.hard_event_interval) == 0
    homeostasis = mod(next_update, schedule.homeostasis_interval) == 0
    structure = mod(next_update, schedule.structure_interval) == 0
    expected_ticks = (
        _checked_tick(clock.analog_ticks, analog, "analog"),
        _checked_tick(clock.hard_event_ticks, hard_event, "hard-event"),
        _checked_tick(clock.homeostasis_ticks, homeostasis, "homeostasis"),
        _checked_tick(clock.structure_ticks, structure, "structure"),
    )
    schedule_intervals = (
        schedule.analog_interval,
        schedule.hard_event_interval,
        schedule.homeostasis_interval,
        schedule.structure_interval,
    )
    return DuePlasticityClocks(
        analog,
        hard_event,
        homeostasis,
        structure,
        next_update,
        expected_ticks,
        schedule_intervals,
    )
end

"""
    commit_clocks!(clock, schedule, expected_due)

Atomically advance an already-previewed learning clock. The expected token and
all four due bits must match a fresh pure preview. Mismatch, duplicate commit,
or overflow fails before any field is changed.
"""
function commit_clocks!(
    clock::LearningClockState,
    schedule::LearningSchedule,
    expected_due::DuePlasticityClocks,
)
    actual = preview_clocks(clock, schedule)
    actual == expected_due || throw(ArgumentError(
        "learning clock preview is stale or does not match the schedule",
    ))
    clock.update = actual.expected_update
    clock.analog_ticks = actual.expected_ticks[1]
    clock.hard_event_ticks = actual.expected_ticks[2]
    clock.homeostasis_ticks = actual.expected_ticks[3]
    clock.structure_ticks = actual.expected_ticks[4]
    return actual
end

"""Oracle convenience: preview and immediately commit one update."""
function advance_clocks!(clock::LearningClockState, schedule::LearningSchedule)
    due = preview_clocks(clock, schedule)
    return commit_clocks!(clock, schedule, due)
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

"""
    begin_local_adjoint!(state, terminal_record; terminal_seed=nothing)

Begin reverse replay of one cell chain. `terminal_record` is the newest exact
state version in this world. `terminal_seed` is a generic/oracle facility.
Canonical candidate replay stops at `initial_core`; common replay instead uses
the fixed-feedback projection of the aggregate raw 22D ListNet derivative and
runs once per state, without a candidate-root state seed.
"""
function begin_local_adjoint!(
    state::ContractedLocalAdjoint{T},
    terminal_record::Integer;
    terminal_seed::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:AbstractFloat}
    terminal = Int(terminal_record)
    terminal >= 0 || throw(ArgumentError(
        "terminal record must be nonnegative",
    ))
    fill!(state.state_bar, zero(T))
    if terminal_seed !== nothing
        length(terminal_seed) == Cell.STATE_DIM || throw(DimensionMismatch(
            "terminal seed has the wrong length",
        ))
        all(isfinite, terminal_seed) || throw(ArgumentError(
            "terminal seed must be finite",
        ))
        copyto!(state.state_bar, terminal_seed)
    end
    state.expected_record = terminal
    state.active = true
    state.touched = false
    state.visited_transition_count = 0
    state.conditional_pullback_count = 0
    return state
end

function begin_local_adjoint!(
    arena::ContractedAdjointArena{T},
    node::Integer,
    terminal_record::Integer;
    terminal_seed::Union{Nothing,AbstractVector{T}}=nothing,
) where {T<:AbstractFloat}
    selected = _check_arena_node(arena, node)
    terminal = Int(terminal_record)
    terminal >= 0 || throw(ArgumentError(
        "terminal record must be nonnegative",
    ))
    @inbounds for index in 1:Cell.STATE_DIM
        arena.state_bar[index, selected] = zero(T)
    end
    if terminal_seed !== nothing
        length(terminal_seed) == Cell.STATE_DIM || throw(DimensionMismatch(
            "terminal seed has the wrong length",
        ))
        all(isfinite, terminal_seed) || throw(ArgumentError(
            "terminal seed must be finite",
        ))
        @inbounds for index in 1:Cell.STATE_DIM
            arena.state_bar[index, selected] = terminal_seed[index]
        end
    end
    arena.expected_record[selected] = terminal
    arena.active[selected] = true
    arena.touched[selected] = false
    arena.visited_transition_count[selected] = 0
    arena.conditional_pullback_count[selected] = 0
    arena.generation[selected] = arena.current_generation
    return selected
end

"""Add a generic/oracle terminal cotangent to an active adjoint seed."""
function add_terminal_seed!(
    state::ContractedLocalAdjoint{T},
    seed::AbstractVector{T},
) where {T<:AbstractFloat}
    state.active || throw(ArgumentError(
        "local adjoint must be active before adding a terminal seed",
    ))
    length(seed) == Cell.STATE_DIM || throw(DimensionMismatch(
        "terminal seed has the wrong length",
    ))
    all(isfinite, seed) || throw(ArgumentError(
        "terminal seed must be finite",
    ))
    @inbounds for index in eachindex(state.state_bar, seed)
        state.state_bar[index] += seed[index]
    end
    return state
end

function add_terminal_seed!(
    arena::ContractedAdjointArena{T},
    node::Integer,
    seed::AbstractVector{T},
) where {T<:AbstractFloat}
    selected = _check_arena_node(arena, node)
    _arena_is_current(arena, selected) && arena.active[selected] ||
        throw(ArgumentError(
            "local adjoint column must be active before adding a terminal seed",
        ))
    length(seed) == Cell.STATE_DIM || throw(DimensionMismatch(
        "terminal seed has the wrong length",
    ))
    all(isfinite, seed) || throw(ArgumentError(
        "terminal seed must be finite",
    ))
    @inbounds for index in 1:Cell.STATE_DIM
        arena.state_bar[index, selected] += seed[index]
    end
    return selected
end

"""
Allocation-free generic/oracle column-to-column seed reduction.

This is not part of the canonical Graph call chain: candidate alternative
worlds are stop-gradient at `initial_core` and common receives aggregate raw
22D fixed-feedback, not their 48D state cotangents.
"""
function add_terminal_seed!(
    destination::ContractedAdjointArena{T},
    destination_node::Integer,
    source::ContractedAdjointArena{T},
    source_node::Integer,
) where {T<:AbstractFloat}
    target = _check_arena_node(destination, destination_node)
    origin = _check_arena_node(source, source_node)
    _arena_is_current(destination, target) && destination.active[target] ||
        throw(ArgumentError(
            "destination local-adjoint column must be active",
        ))
    _arena_is_current(source, origin) || throw(ArgumentError(
        "source local-adjoint column is stale",
    ))
    @inbounds for index in 1:Cell.STATE_DIM
        destination.state_bar[index, target] += source.state_bar[index, origin]
    end
    return target
end

"""Verify that reverse replay reached the declared chronological root."""
function finish_local_adjoint!(
    state::ContractedLocalAdjoint,
    root_record::Integer=0,
)
    state.active || throw(ArgumentError("local adjoint is not active"))
    root = Int(root_record)
    root >= 0 || throw(ArgumentError("root record must be nonnegative"))
    state.expected_record == root || throw(ArgumentError(
        "local replay stopped at record $(state.expected_record), expected $root",
    ))
    state.conditional_pullback_count == state.visited_transition_count ||
        error("each visited transition must execute exactly one cell pullback")
    state.active = false
    return state.state_bar
end

function finish_local_adjoint!(
    arena::ContractedAdjointArena,
    node::Integer,
    root_record::Integer=0,
)
    selected = _check_arena_node(arena, node)
    _arena_is_current(arena, selected) && arena.active[selected] ||
        throw(ArgumentError("local adjoint column is not active"))
    root = Int(root_record)
    root >= 0 || throw(ArgumentError("root record must be nonnegative"))
    arena.expected_record[selected] == root || throw(ArgumentError(
        "local replay stopped at record $(arena.expected_record[selected]), " *
        "expected $root",
    ))
    arena.conditional_pullback_count[selected] ==
        arena.visited_transition_count[selected] ||
        error("each visited transition must execute exactly one cell pullback")
    arena.active[selected] = false
    return selected
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

@inline function _check_transition_shapes(
    scratch::ContractedAdjointScratch{T},
    previous_state::AbstractVector{T},
    input::AbstractVector{T},
    next_state::AbstractVector{T},
    continuous_signal::AbstractVector{T},
    packet_signal::AbstractVector{T},
) where {T<:AbstractFloat}
    length(scratch.dstate) == Cell.STATE_DIM || throw(DimensionMismatch(
        "local-adjoint scratch has the wrong cell-state length",
    ))
    length(previous_state) == Cell.STATE_DIM || throw(DimensionMismatch(
        "previous state has the wrong length",
    ))
    length(input) == Cell.INPUT_DIM || throw(DimensionMismatch(
        "cell input has the wrong length",
    ))
    length(next_state) == Cell.STATE_DIM || throw(DimensionMismatch(
        "next state has the wrong length",
    ))
    length(continuous_signal) == LOCAL_OBSERVATION_DIM || throw(
        DimensionMismatch("continuous learning signal has the wrong length"),
    )
    length(packet_signal) == Axon.PACKET_DIM || throw(
        DimensionMismatch("packet learning signal has the wrong length"),
    )
    return nothing
end

@inline function _check_event_control(control::CausalEventControl)
    if control.connected || !iszero(control.advantage)
        throw(ArgumentError(
            "hard-event control is disconnected: a causal continuation/energy " *
            "estimator and fused two-lane cell VJP are required",
        ))
    end
    return nothing
end

@inline function _contract_seeded_transition!(
    scratch::ContractedAdjointScratch{T},
    previous_state::AbstractVector{T},
    input::AbstractVector{T},
    cache::Cell.CellParameterCache{T},
    derivative_cache::Cell.CellParameterDerivativeCache{T},
    next_state::AbstractVector{T},
    continuous_signal::AbstractVector{T},
    packet_signal::AbstractVector{T},
    eligibility_scale::T,
) where {T<:AbstractFloat}
    continuous_count = Cell.N_COMPARTMENTS * Cell.COMPARTMENT_STATE_DIM
    @inbounds for index in 1:continuous_count
        scratch.dnext[index] = muladd(
            eligibility_scale,
            continuous_signal[index],
            scratch.dnext[index],
        )
    end
    scratch.dnext[Cell.ADAPTATION_INDEX] = muladd(
        eligibility_scale,
        continuous_signal[ADAPTATION_OBSERVATION],
        scratch.dnext[Cell.ADAPTATION_INDEX],
    )
    direct_margin_cotangent =
        eligibility_scale * continuous_signal[MARGIN_OBSERVATION]

    @inbounds for lane in 1:Axon.PACKET_DIM
        scratch.packet_bar[lane] = eligibility_scale * packet_signal[lane]
    end
    direct_margin_cotangent += Axon.axon_packet_pullback!(
        scratch.packet_dnext,
        scratch.packet_bar,
        previous_state,
        next_state,
        cache,
    )
    @inbounds for index in 1:(Cell.STATE_DIM - 1)
        scratch.dnext[index] += scratch.packet_dnext[index]
    end

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
        direct_margin_cotangent,
    )
    return nothing
end

"""
    contract_replayed_transition!(...; touched, eligibility_scale,
                                  event_control)

Contract one recorded transition during pass-2 replay. The forward trajectory
and its local observables are teacher-free; the post-hoc `continuous_signal`
and `packet_signal` only select the already-defined local derivatives.

This is the reverse form of `sum_t m_t' * E_t`: all future local recurrence is
carried by one 48D `state_bar`, and exactly one conditional cell pullback is
executed for every visited transition. Continuous cotangents stop at the
destination input. The graph owner is responsible for using `dinput` only to
differentiate the local deposit/contact parameters and must not propagate it
through the source packet into another cell.
"""
function contract_replayed_transition!(
    state::ContractedLocalAdjoint{T},
    scratch::ContractedAdjointScratch{T},
    link::ChronologicalTransitionLink,
    previous_state::AbstractVector{T},
    input::AbstractVector{T},
    cache::Cell.CellParameterCache{T},
    derivative_cache::Cell.CellParameterDerivativeCache{T},
    next_state::AbstractVector{T},
    continuous_signal::AbstractVector{T},
    packet_signal::AbstractVector{T};
    touched::Bool,
    eligibility_scale::T=one(T),
    event_control::CausalEventControl{T}=CausalEventControl(; T=T),
) where {T<:AbstractFloat}
    _check_transition_shapes(
        scratch,
        previous_state,
        input,
        next_state,
        continuous_signal,
        packet_signal,
    )
    isfinite(eligibility_scale) || throw(ArgumentError(
        "eligibility scale must be finite",
    ))
    _check_event_control(event_control)
    touched || return false
    state.active || throw(ArgumentError(
        "begin_local_adjoint! must be called before replay contraction",
    ))
    state.expected_record == link.record || throw(ArgumentError(
        "out-of-order local replay: expected record $(state.expected_record), " *
        "received $(link.record)",
    ))
    all(isfinite, previous_state) && all(isfinite, input) &&
        all(isfinite, next_state) &&
        all(isfinite, continuous_signal) && all(isfinite, packet_signal) ||
        throw(ArgumentError("local replay inputs must be finite"))

    copyto!(scratch.dnext, state.state_bar)
    _contract_seeded_transition!(
        scratch,
        previous_state,
        input,
        cache,
        derivative_cache,
        next_state,
        continuous_signal,
        packet_signal,
        eligibility_scale,
    )
    copyto!(state.state_bar, scratch.dstate)
    state.expected_record = link.predecessor
    state.touched = true
    state.visited_transition_count += 1
    state.conditional_pullback_count += 1
    return true
end

"""
Production arena overload. It addresses the cotangent slab by integer column,
uses precomputed parameter caches, and creates no per-transition view/object.
"""
function contract_replayed_transition!(
    arena::ContractedAdjointArena{T},
    node::Integer,
    scratch::ContractedAdjointScratch{T},
    link::ChronologicalTransitionLink,
    previous_state::AbstractVector{T},
    input::AbstractVector{T},
    cache::Cell.CellParameterCache{T},
    derivative_cache::Cell.CellParameterDerivativeCache{T},
    next_state::AbstractVector{T},
    continuous_signal::AbstractVector{T},
    packet_signal::AbstractVector{T};
    touched::Bool,
    eligibility_scale::T=one(T),
    event_control::CausalEventControl{T}=CausalEventControl(; T=T),
) where {T<:AbstractFloat}
    _check_transition_shapes(
        scratch,
        previous_state,
        input,
        next_state,
        continuous_signal,
        packet_signal,
    )
    isfinite(eligibility_scale) || throw(ArgumentError(
        "eligibility scale must be finite",
    ))
    _check_event_control(event_control)
    touched || return false
    selected = _check_arena_node(arena, node)
    _arena_is_current(arena, selected) && arena.active[selected] ||
        throw(ArgumentError(
            "begin_local_adjoint! must activate the arena column before replay",
        ))
    arena.expected_record[selected] == link.record || throw(ArgumentError(
        "out-of-order local replay: expected record " *
        "$(arena.expected_record[selected]), received $(link.record)",
    ))
    all(isfinite, previous_state) && all(isfinite, input) &&
        all(isfinite, next_state) &&
        all(isfinite, continuous_signal) && all(isfinite, packet_signal) ||
        throw(ArgumentError("local replay inputs must be finite"))

    @inbounds for index in 1:Cell.STATE_DIM
        scratch.dnext[index] = arena.state_bar[index, selected]
    end
    _contract_seeded_transition!(
        scratch,
        previous_state,
        input,
        cache,
        derivative_cache,
        next_state,
        continuous_signal,
        packet_signal,
        eligibility_scale,
    )
    @inbounds for index in 1:Cell.STATE_DIM
        arena.state_bar[index, selected] = scratch.dstate[index]
    end
    arena.expected_record[selected] = link.predecessor
    arena.touched[selected] = true
    arena.visited_transition_count[selected] += 1
    arena.conditional_pullback_count[selected] += 1
    return true
end

# Focused-test/oracle convenience overloads. Production must pass the caches
# already owned by the model rather than rebuilding them per transition.
function contract_replayed_transition!(
    state::ContractedLocalAdjoint{T},
    scratch::ContractedAdjointScratch{T},
    link::ChronologicalTransitionLink,
    previous_state::AbstractVector{T},
    input::AbstractVector{T},
    raw_parameters::AbstractVector{T},
    next_state::AbstractVector{T},
    continuous_signal::AbstractVector{T},
    packet_signal::AbstractVector{T};
    kwargs...,
) where {T<:AbstractFloat}
    length(raw_parameters) == Cell.PARAM_DIM || throw(DimensionMismatch(
        "raw cell parameters have the wrong length",
    ))
    all(isfinite, raw_parameters) || throw(ArgumentError(
        "raw cell parameters must be finite",
    ))
    cache, derivative_cache = Cell.parameter_caches(raw_parameters)
    return contract_replayed_transition!(
        state,
        scratch,
        link,
        previous_state,
        input,
        cache,
        derivative_cache,
        next_state,
        continuous_signal,
        packet_signal;
        kwargs...,
    )
end

function contract_replayed_transition!(
    arena::ContractedAdjointArena{T},
    node::Integer,
    scratch::ContractedAdjointScratch{T},
    link::ChronologicalTransitionLink,
    previous_state::AbstractVector{T},
    input::AbstractVector{T},
    raw_parameters::AbstractVector{T},
    next_state::AbstractVector{T},
    continuous_signal::AbstractVector{T},
    packet_signal::AbstractVector{T};
    kwargs...,
) where {T<:AbstractFloat}
    length(raw_parameters) == Cell.PARAM_DIM || throw(DimensionMismatch(
        "raw cell parameters have the wrong length",
    ))
    all(isfinite, raw_parameters) || throw(ArgumentError(
        "raw cell parameters must be finite",
    ))
    cache, derivative_cache = Cell.parameter_caches(raw_parameters)
    return contract_replayed_transition!(
        arena,
        node,
        scratch,
        link,
        previous_state,
        input,
        cache,
        derivative_cache,
        next_state,
        continuous_signal,
        packet_signal;
        kwargs...,
    )
end

@inline raw_parameter_cotangent(scratch::ContractedAdjointScratch) = scratch.draw
@inline input_cotangent(scratch::ContractedAdjointScratch) = scratch.dinput

"""
Update slow structural utility from an already-contracted parameter
contribution. Canonical callers in `:combined` mode pass the per-use sum of the
47-dimensional continuous and 12-dimensional packet objective contributions;
the two terms are never selected independently. Callers in `:none` mode must
leave utility strictly zero. Since each combined contribution is
`m * eligibility`, zero third factor or zero eligibility independently produces
zero utility. Shared parameters are updated once after the graph owner has
deterministically reduced all uses.
"""
function update_structural_utility!(
    state::StructuralUtilityState{T},
    contracted_contribution::AbstractVector{T};
    decay::T=T(0.999),
    normalization::T=one(T),
    due::Bool=true,
) where {T<:AbstractFloat}
    length(state.utility) == length(contracted_contribution) || throw(
        DimensionMismatch("structural utility contribution has the wrong length"),
    )
    _require_unit_interval("utility decay", decay)
    isfinite(normalization) && normalization >= zero(T) || throw(ArgumentError(
        "utility normalization must be finite and nonnegative",
    ))
    due || return state
    denominator = normalization + eps(T)
    @inbounds for parameter in eachindex(state.utility, contracted_contribution)
        contribution = abs(contracted_contribution[parameter]) / denominator
        state.utility[parameter] = muladd(
            decay, state.utility[parameter], contribution,
        )
    end
    state.update_count += 1
    return state
end

end # module CanonicalLocalLearning
