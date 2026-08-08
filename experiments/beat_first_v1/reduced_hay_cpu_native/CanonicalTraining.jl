module CanonicalTraining

using SHA
using ..CanonicalBarrierless
using ..CanonicalCheckpoint
using ..CanonicalDendriticGraph
using ..CanonicalExperimentData
using ..CanonicalListNet
using ..CanonicalLocalLearning
using ..CanonicalOptimizer
using ..CanonicalPlasticity
using ..DendriticAxonPacket
using ..DendriticOutputPopulation
using ..ReducedHayCPUSampler

const Barrierless = CanonicalBarrierless
const Checkpoint = CanonicalCheckpoint
const Data = CanonicalExperimentData
const Graph = CanonicalDendriticGraph
const ListNet = CanonicalListNet
const Local = CanonicalLocalLearning
const Optimizer = CanonicalOptimizer
const Plasticity = CanonicalPlasticity
const Axon = DendriticAxonPacket
const Output = DendriticOutputPopulation
const Sampler = ReducedHayCPUSampler

export AuxiliaryLossConfig,
       OptimizerGroupConfig,
       CanonicalTrainingConfig,
       CanonicalLossResult,
       MechanismActivation,
       MechanismCounters,
       MechanismCounterState,
       CumulativeMechanismCounters,
       TrainingUpdateResult,
       DendriticTrainingAdapter,
       CanonicalTrainer,
       mechanism_counts,
       cumulative_mechanism_counts,
       checkpoint_components,
       restore_training_checkpoint!,
       training_config_fingerprint,
       training_config_summary,
       update_count,
       train_update!,
       with_training_team

const OUTPUT_DIM = Output.OUTPUT_DIM
const QUANTILE_RANGE = Output.QUANTILE_RANGE
const GEOMETRY_RANGE = Output.GEOMETRY_RANGE
const QUANTILE_COUNT = length(QUANTILE_RANGE)
const GEOMETRY_COUNT = length(GEOMETRY_RANGE)
const _CANDIDATE_REDUCTION_SLOT = UInt8(1)
const _STATE_COMMON_REDUCTION_SLOT = UInt8(2)

const _MECHANISM_KEYS = (
    :decolle_signal_nonzero,
    :subthreshold_updates,
    :nonspiking_updates,
    :hard_event_control_updates,
    :homeostasis_events,
    :synaptic_scaling_events,
    :utility_updates,
    :rewires,
)

"""Weights for the non-Q portion of the fixed 22-dimensional ABI."""
struct AuxiliaryLossConfig
    death_weight::Float32
    quantile_weight::Float32
    geometry_weight::Float32
    huber_delta::Float32

    function AuxiliaryLossConfig(
        death_weight::Real,
        quantile_weight::Real,
        geometry_weight::Real,
        huber_delta::Real,
    )
        values = map(Float32, (
            death_weight, quantile_weight, geometry_weight, huber_delta,
        ))
        all(isfinite, values) || throw(ArgumentError(
            "auxiliary loss values must be finite",
        ))
        all(value -> value >= 0.0f0, values[1:3]) || throw(ArgumentError(
            "auxiliary loss weights must be nonnegative",
        ))
        values[4] > 0.0f0 || throw(ArgumentError(
            "auxiliary Huber delta must be positive",
        ))
        return new(values...)
    end
end

AuxiliaryLossConfig(;
    death_weight::Real=0.10,
    quantile_weight::Real=0.05,
    geometry_weight::Real=0.10,
    huber_delta::Real=1.0,
) = AuxiliaryLossConfig(
    death_weight, quantile_weight, geometry_weight, huber_delta,
)

"""
Static registry metadata. Intervals live only in `LocalLearningConfig`; these
multipliers never change across an optimizer step or checkpoint resume.
"""
struct OptimizerGroupConfig
    core_cell_multiplier::Float32
    semantic_projection_multiplier::Float32
    event_multiplier::Float32
    output_cell_multiplier::Float32
    output_projection_multiplier::Float32
    conductance_floor::Float32
    conductance_ceiling::Float32

    function OptimizerGroupConfig(
        core_cell_multiplier::Real,
        semantic_projection_multiplier::Real,
        event_multiplier::Real,
        output_cell_multiplier::Real,
        output_projection_multiplier::Real,
        conductance_floor::Real,
        conductance_ceiling::Real,
    )
        multipliers = map(Float32, (
            core_cell_multiplier,
            semantic_projection_multiplier,
            event_multiplier,
            output_cell_multiplier,
            output_projection_multiplier,
        ))
        all(value -> isfinite(value) && value >= 0.0f0, multipliers) ||
            throw(ArgumentError(
                "optimizer group multipliers must be finite and nonnegative",
            ))
        floor = Float32(conductance_floor)
        ceiling = Float32(conductance_ceiling)
        isfinite(floor) && isfinite(ceiling) && 0.0f0 < floor < ceiling ||
            throw(ArgumentError(
                "conductance bounds must satisfy 0 < floor < ceiling",
            ))
        return new(multipliers..., floor, ceiling)
    end
end

OptimizerGroupConfig(;
    core_cell_multiplier::Real=1.0,
    semantic_projection_multiplier::Real=1.0,
    event_multiplier::Real=1.0,
    output_cell_multiplier::Real=1.0,
    output_projection_multiplier::Real=1.0,
    conductance_floor::Real=1.0e-4,
    conductance_ceiling::Real=4.0,
) = OptimizerGroupConfig(
    core_cell_multiplier,
    semantic_projection_multiplier,
    event_multiplier,
    output_cell_multiplier,
    output_projection_multiplier,
    conductance_floor,
    conductance_ceiling,
)

"""The sole immutable owner of production-training numerical controls."""
struct CanonicalTrainingConfig{L<:ListNet.ListNetConfig}
    listnet::L
    auxiliary::AuxiliaryLossConfig
    local_learning::Local.LocalLearningConfig
    optimizer::Optimizer.AdamWConfig
    groups::OptimizerGroupConfig
end

function CanonicalTrainingConfig(;
    listnet::ListNet.ListNetConfig=ListNet.ListNetConfig(Float32),
    auxiliary::AuxiliaryLossConfig=AuxiliaryLossConfig(),
    local_learning::Local.LocalLearningConfig=Local.LocalLearningConfig(),
    optimizer::Optimizer.AdamWConfig=Optimizer.AdamWConfig(),
    groups::OptimizerGroupConfig=OptimizerGroupConfig(
        conductance_floor=local_learning.plasticity.conductance_floor,
        conductance_ceiling=local_learning.plasticity.conductance_ceiling,
    ),
)
    groups.conductance_floor == local_learning.plasticity.conductance_floor ||
        throw(ArgumentError(
            "optimizer and plasticity conductance floors differ",
        ))
    groups.conductance_ceiling == local_learning.plasticity.conductance_ceiling ||
        throw(ArgumentError(
            "optimizer and plasticity conductance ceilings differ",
        ))
    return CanonicalTrainingConfig(
        listnet, auxiliary, local_learning, optimizer, groups,
    )
end

function training_config_summary(config::CanonicalTrainingConfig)
    listnet = config.listnet
    auxiliary = config.auxiliary
    optimizer = config.optimizer
    groups = config.groups
    return join((
        "listnet_temperature=$(listnet.temperature)",
        "listnet_scale_floor=$(listnet.scale_floor)",
        "listnet_q_huber_weight=$(listnet.q_huber_weight)",
        "listnet_huber_delta=$(listnet.huber_delta)",
        "death_weight=$(auxiliary.death_weight)",
        "quantile_weight=$(auxiliary.quantile_weight)",
        "geometry_weight=$(auxiliary.geometry_weight)",
        "auxiliary_huber_delta=$(auxiliary.huber_delta)",
        Local.config_summary(config.local_learning),
        "adam_learning_rate=$(optimizer.learning_rate)",
        "adam_beta1=$(optimizer.beta1)",
        "adam_beta2=$(optimizer.beta2)",
        "adam_epsilon=$(optimizer.epsilon)",
        "adam_clip_norm=$(optimizer.clip_norm)",
        "adam_weight_decay=$(optimizer.weight_decay)",
        "core_cell_multiplier=$(groups.core_cell_multiplier)",
        "semantic_projection_multiplier=$(groups.semantic_projection_multiplier)",
        "event_multiplier=$(groups.event_multiplier)",
        "output_cell_multiplier=$(groups.output_cell_multiplier)",
        "output_projection_multiplier=$(groups.output_projection_multiplier)",
        "conductance_floor=$(groups.conductance_floor)",
        "conductance_ceiling=$(groups.conductance_ceiling)",
    ), ' ')
end

training_config_fingerprint(config::CanonicalTrainingConfig) =
    bytes2hex(sha256(training_config_summary(config)))

function Base.show(io::IO, config::CanonicalTrainingConfig)
    print(io, "CanonicalTrainingConfig(", training_config_summary(config), ')')
end

"""Mechanisms which are mathematically due and have a nonzero driver."""
struct MechanismActivation
    decolle_signal_nonzero::Bool
    subthreshold_updates::Bool
    nonspiking_updates::Bool
    hard_event_control_updates::Bool
    homeostasis_events::Bool
    synaptic_scaling_events::Bool
    utility_updates::Bool
    rewires::Bool
end

MechanismActivation() = MechanismActivation(
    false, false, false, false, false, false, false, false,
)

"""Immutable per-update mechanism telemetry."""
struct MechanismCounters
    decolle_signal_nonzero::Int64
    subthreshold_updates::Int64
    nonspiking_updates::Int64
    hard_event_control_updates::Int64
    homeostasis_events::Int64
    synaptic_scaling_events::Int64
    utility_updates::Int64
    rewires::Int64

    function MechanismCounters(values::Vararg{Integer,8})
        all(value -> value >= 0, values) || throw(ArgumentError(
            "mechanism counters must be nonnegative",
        ))
        return new(map(Int64, values)...)
    end
end

MechanismCounters() = MechanismCounters(0, 0, 0, 0, 0, 0, 0, 0)

"""Mutable worker-local accumulator; no hot-path atomics are required."""
mutable struct MechanismCounterState
    decolle_signal_nonzero::Int64
    subthreshold_updates::Int64
    nonspiking_updates::Int64
    hard_event_control_updates::Int64
    homeostasis_events::Int64
    synaptic_scaling_events::Int64
    utility_updates::Int64
    rewires::Int64
end

MechanismCounterState() = MechanismCounterState(0, 0, 0, 0, 0, 0, 0, 0)

"""Training-owned cumulative telemetry committed once per successful update."""
mutable struct CumulativeMechanismCounters
    decolle_signal_nonzero::UInt64
    subthreshold_updates::UInt64
    nonspiking_updates::UInt64
    hard_event_control_updates::UInt64
    homeostasis_events::UInt64
    synaptic_scaling_events::UInt64
    utility_updates::UInt64
    rewires::UInt64
end

CumulativeMechanismCounters() = CumulativeMechanismCounters(
    0, 0, 0, 0, 0, 0, 0, 0,
)

@inline function _reset!(state::MechanismCounterState)
    @inbounds for name in _MECHANISM_KEYS
        setfield!(state, name, Int64(0))
    end
    return state
end

@inline _snapshot(state::MechanismCounterState) = MechanismCounters(
    state.decolle_signal_nonzero,
    state.subthreshold_updates,
    state.nonspiking_updates,
    state.hard_event_control_updates,
    state.homeostasis_events,
    state.synaptic_scaling_events,
    state.utility_updates,
    state.rewires,
)

@inline function _add(left::MechanismCounters, right::MechanismCounters)
    return MechanismCounters(
        left.decolle_signal_nonzero + right.decolle_signal_nonzero,
        left.subthreshold_updates + right.subthreshold_updates,
        left.nonspiking_updates + right.nonspiking_updates,
        left.hard_event_control_updates + right.hard_event_control_updates,
        left.homeostasis_events + right.homeostasis_events,
        left.synaptic_scaling_events + right.synaptic_scaling_events,
        left.utility_updates + right.utility_updates,
        left.rewires + right.rewires,
    )
end

@inline function _accumulate_local_report!(state::MechanismCounterState, report)
    state.decolle_signal_nonzero += Int64(report.signal_nonzero)
    state.subthreshold_updates += Int64(report.conditional_pullbacks)
    state.nonspiking_updates += Int64(report.nonspiking_transitions)
    state.hard_event_control_updates +=
        Int64(report.event_control_source_transitions)
    state.utility_updates += Int64(report.utility_updates)
    return state
end

struct CanonicalLossResult
    total_loss::Float32
    listnet_kl::Float32
    q_huber_loss::Float32
    death_loss::Float32
    quantile_loss::Float32
    geometry_loss::Float32
    valid_candidates::Int
end

struct TrainingUpdateResult
    loss::CanonicalLossResult
    mechanisms::MechanismCounters
    optimizer::Optimizer.DualOptimizerStepStats
    update::UInt64
end

mutable struct DendriticTrainingWorker{G,L}
    graph::G
    local_learner::L
    delta22::Vector{Float32}
    counters::MechanismCounterState
    hard_event_deliveries::Int64
    worker_slot::UInt16
end

mutable struct DendriticTrainingAdapter{C,R,A,S,PB,PS,AF,AO,HR,HZ} <:
               Barrierless.AbstractCanonicalGraphAdapter
    model::Graph.CanonicalModel
    config::C
    local_signals::S
    common_states::Vector{Graph.ModelState}
    common_signature::Vector{Graph.TrajectorySignature}
    common_prepare_generation::Vector{UInt64}
    common_prepare_owner::Vector{UInt16}
    components::Vector{Output.OutputComponents{Float32}}
    component_bars::Vector{Output.OutputComponentGradient{Float32}}
    forward_signature::Vector{Graph.TrajectorySignature}
    replay_signature::Vector{Graph.TrajectorySignature}
    forward_seen::Vector{UInt8}
    replay_seen::Vector{UInt8}
    ordinal_flat::Vector{Int32}
    ordinal_state::Vector{Int16}
    state_offsets::Vector{Int32}
    state_value::Vector{Float32}
    state_value_delta::Vector{Float32}
    state_delta22::Matrix{Float32}
    q_bar::Vector{Float32}
    raw_output::Matrix{Float32}
    raw_delta::Matrix{Float32}
    student_q::Matrix{Float32}
    q_gradient::Matrix{Float32}
    listnet_scratch::ListNet.ListNetScratch{Float32}
    listnet_result::Union{Nothing,ListNet.ListNetResult{Float32}}
    loss::CanonicalLossResult
    reduced_gradient::Graph.ModelGradient
    gradient_slots::Vector{Graph.ModelGradient}
    reduced_hard_gradient::Graph.ModelHardEventGradient
    hard_delta_slots::Vector{Graph.ModelHardEventDelta}
    common_hard_seed::Array{Float32,3}
    common_hard_seed_generation::Vector{UInt64}
    common_hard_seed_consumed::Vector{UInt8}
    hard_delivery_slots::Vector{Int64}
    hard_event_deliveries::Int64
    plasticity_batch::PB
    plasticity_state::PS
    event_destination::Vector{UInt16}
    zero_based_state_offsets::Vector{Int32}
    mechanism_slots::Vector{MechanismCounters}
    slot_generation::Vector{UInt64}
    slot_kind::Vector{UInt8}
    slot_logical_first::Vector{Int32}
    slot_logical_last::Vector{Int32}
    slot_owner::Vector{UInt16}
    mechanisms::MechanismCounters
    cumulative_mechanisms::CumulativeMechanismCounters
    expected::MechanismActivation
    clocks::Local.LearningClockState
    due::Local.DuePlasticityClocks
    registry::R
    optimizer_state::A
    analog_full_buffers::AF
    analog_output_buffers::AO
    hard_recurrent_buffers::HR
    hard_zero_buffers::HZ
    candidate_chunk_size::Int
    candidate_slot_capacity::Int
    slot_capacity::Int
    active_generation::UInt64
    optimizer_boundaries::Int
    updates::UInt64
end

@inline function _copy_component!(
    destination::Output.OutputComponents{Float32},
    source::Output.OutputComponents{Float32},
)
    destination.value = source.value
    destination.advantage = source.advantage
    destination.death = source.death
    copyto!(destination.geometry, source.geometry)
    destination.uncertainty_raw = source.uncertainty_raw
    return destination
end

function _parameter_registry(
    model::Graph.CanonicalModel,
    gradient::Graph.ModelGradient,
    config::OptimizerGroupConfig,
)
    parameter = Graph.parameter_components(model.parameters)
    bar = Graph.gradient_components(gradient)
    lower = config.conductance_floor
    upper = config.conductance_ceiling
    return Optimizer.ParameterRegistry(
        Optimizer.ParameterGroup(
            :core_cell_raw,
            parameter.core_cell_raw,
            bar.core_cell_raw,
            Optimizer.CELL_RAW;
            multiplier=config.core_cell_multiplier,
        ),
        Optimizer.ParameterGroup(
            :semantic_projection_raw,
            parameter.semantic_projection_raw,
            bar.semantic_projection_raw,
            Optimizer.INVERSE_SOFTPLUS_CONDUCTANCE;
            multiplier=config.semantic_projection_multiplier,
            lower_bound=lower,
            upper_bound=upper,
        ),
        Optimizer.ParameterGroup(
            :event_raw,
            parameter.event_raw,
            bar.event_raw,
            Optimizer.INVERSE_SOFTPLUS_CONDUCTANCE;
            multiplier=config.event_multiplier,
            lower_bound=lower,
            upper_bound=upper,
        ),
        Optimizer.ParameterGroup(
            :output_cell_raw,
            parameter.output_cell_raw,
            bar.output_cell_raw,
            Optimizer.CELL_RAW;
            multiplier=config.output_cell_multiplier,
        ),
        Optimizer.ParameterGroup(
            :output_projection_raw,
            parameter.output_projection_raw,
            bar.output_projection_raw,
            Optimizer.INVERSE_SOFTPLUS_CONDUCTANCE;
            multiplier=config.output_projection_multiplier,
            lower_bound=lower,
            upper_bound=upper,
        ),
    )
end

@inline function _hard_event_due(adapter::DendriticTrainingAdapter)
    local_config = adapter.config.local_learning
    return adapter.due.hard_event &&
        local_config.hard_event_multiplier > 0.0f0
end

function _hard_event_gradient(model::Graph.CanonicalModel)
    parameters = model.parameters
    return Graph.ModelHardEventGradient(
        zeros(Float32, size(parameters.core_cell_raw)),
        zeros(Float32, size(parameters.semantic_projection_raw)),
        zeros(Float32, length(parameters.event_raw)),
    )
end

function DendriticTrainingAdapter(
    model::Graph.CanonicalModel,
    batch::Data.CanonicalBatch,
    config::CanonicalTrainingConfig=CanonicalTrainingConfig();
    candidate_chunk_size::Integer=4,
)
    chunk = Int(candidate_chunk_size)
    chunk >= 1 || throw(ArgumentError(
        "candidate_chunk_size must be positive",
    ))
    input = batch.input
    states = input.state_batch
    capacity = input.capacity
    model.config.max_candidates >= chunk || throw(ArgumentError(
        "graph candidate scratch must hold one scheduler microbatch",
    ))
    config.local_learning.plasticity.structure_enabled && throw(ArgumentError(
        "canonical fixed-spine training requires structure_enabled=false",
    ))
    candidate_slot_capacity = cld(capacity, chunk)
    slot_capacity = candidate_slot_capacity + states
    hard_seed_capacity = Base.checked_mul(
        chunk, Graph.MAX_COMMON_HARD_EVENT_SEEDS_PER_CANDIDATE,
    )
    local_signals = Graph.initialize_local_signal_maps(
        model, config.local_learning,
    )
    common_states = [Graph.initialize_state(model) for _ in 1:states]
    reduced_gradient = Graph.initialize_gradient(model)
    gradient_slots = [
        Graph.initialize_gradient(model) for _ in 1:slot_capacity
    ]
    reduced_hard_gradient = _hard_event_gradient(model)
    hard_delta_slots = [
        Graph.initialize_hard_event_delta(model, hard_seed_capacity)
        for _ in 1:slot_capacity
    ]
    registry = _parameter_registry(model, reduced_gradient, config.groups)
    optimizer_state = Optimizer.AdamWState(registry)
    analog = Graph.gradient_components(reduced_gradient)
    hard = Graph.hard_gradient_components(reduced_hard_gradient)
    analog_full_buffers = Optimizer.GradientBufferSet(registry, (
        analog.core_cell_raw,
        analog.semantic_projection_raw,
        analog.event_raw,
        analog.output_cell_raw,
        analog.output_projection_raw,
    ))
    analog_output_buffers = Optimizer.GradientBufferSet(registry, (
        nothing,
        nothing,
        nothing,
        analog.output_cell_raw,
        analog.output_projection_raw,
    ))
    hard_recurrent_buffers = Optimizer.GradientBufferSet(registry, (
        hard.core_cell_raw,
        hard.semantic_projection_raw,
        hard.event_raw,
        nothing,
        nothing,
    ))
    hard_zero_buffers = Optimizer.GradientBufferSet(
        registry, (nothing, nothing, nothing, nothing, nothing),
    )
    event_parameter_total = Graph.event_parameter_count(model)
    event_destination = Vector{UInt16}(undef, event_parameter_total)
    contact_count = 0
    seen_shared_gain = false
    @inbounds for contact in 1:event_parameter_total
        destination = Graph.event_parameter_descriptor(model, contact).destination
        event_destination[contact] = destination
        if iszero(destination)
            seen_shared_gain = true
        else
            seen_shared_gain && error(
                "anatomical event contacts must precede shared event gains",
            )
            contact_count += 1
        end
    end
    plasticity_batch = Plasticity.CanonicalPlasticityBatch(
        Graph.TOTAL_NODE_COUNT,
        contact_count,
        states,
        capacity,
    )
    plasticity_state = Plasticity.PlasticityState(
        config.local_learning.plasticity,
        Graph.TOTAL_NODE_COUNT,
        contact_count,
    )
    zero_loss = CanonicalLossResult(
        0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0,
    )
    return DendriticTrainingAdapter(
        model,
        config,
        local_signals,
        common_states,
        fill(Graph.TrajectorySignature(), states),
        zeros(UInt64, states),
        zeros(UInt16, states),
        [Output.OutputComponents(Float32) for _ in 1:capacity],
        [Output.OutputComponentGradient(Float32) for _ in 1:capacity],
        fill(Graph.TrajectorySignature(), capacity),
        fill(Graph.TrajectorySignature(), capacity),
        zeros(UInt8, capacity),
        zeros(UInt8, capacity),
        zeros(Int32, capacity),
        zeros(Int16, capacity),
        zeros(Int32, states + 1),
        zeros(Float32, states),
        zeros(Float32, states),
        zeros(Float32, OUTPUT_DIM, states),
        zeros(Float32, capacity),
        zeros(Float32, OUTPUT_DIM, capacity),
        zeros(Float32, OUTPUT_DIM, capacity),
        zeros(Float32, input.width, states),
        zeros(Float32, input.width, states),
        ListNet.ListNetScratch(input.width, Float32),
        nothing,
        zero_loss,
        reduced_gradient,
        gradient_slots,
        reduced_hard_gradient,
        hard_delta_slots,
        zeros(Float32, Axon.EVENT_DIM, Graph.CORE_NODE_COUNT, states),
        zeros(UInt64, states),
        zeros(UInt8, states),
        zeros(Int64, slot_capacity),
        Int64(0),
        plasticity_batch,
        plasticity_state,
        event_destination,
        zeros(Int32, states + 1),
        fill(MechanismCounters(), slot_capacity),
        zeros(UInt64, slot_capacity),
        zeros(UInt8, slot_capacity),
        zeros(Int32, slot_capacity),
        zeros(Int32, slot_capacity),
        zeros(UInt16, slot_capacity),
        MechanismCounters(),
        CumulativeMechanismCounters(),
        MechanismActivation(),
        Local.LearningClockState(),
        Local.DuePlasticityClocks(false, false, false, false),
        registry,
        optimizer_state,
        analog_full_buffers,
        analog_output_buffers,
        hard_recurrent_buffers,
        hard_zero_buffers,
        chunk,
        candidate_slot_capacity,
        slot_capacity,
        UInt64(0),
        0,
        UInt64(0),
    )
end

function Barrierless.create_worker_arena(
    adapter::DendriticTrainingAdapter,
    worker_slot::Int,
)
    worker_slot >= 1 || throw(ArgumentError("worker_slot must be positive"))
    graph = Graph.initialize_worker(adapter.model)
    hard_seed_capacity = Base.checked_mul(
        adapter.candidate_chunk_size,
        Graph.MAX_COMMON_HARD_EVENT_SEEDS_PER_CANDIDATE,
    )
    learner = Graph.initialize_local_learner(
        adapter.model,
        adapter.local_signals;
        hard_event_seed_capacity=hard_seed_capacity,
    )
    return DendriticTrainingWorker(
        graph,
        learner,
        zeros(Float32, OUTPUT_DIM),
        MechanismCounterState(),
        Int64(0),
        UInt16(worker_slot),
    )
end

Barrierless.state_count(
    ::DendriticTrainingAdapter,
    batch::Data.CanonicalBatch,
) = batch.input.state_batch

Barrierless.candidate_count(
    ::DendriticTrainingAdapter,
    batch::Data.CanonicalBatch,
) = batch.input.valid_count

function Barrierless.state_candidate_bounds(
    adapter::DendriticTrainingAdapter,
    ::Data.CanonicalBatch,
    state::Int,
)
    1 <= state < length(adapter.state_offsets) || throw(BoundsError(
        adapter.state_offsets, state,
    ))
    first = Int(@inbounds adapter.state_offsets[state])
    last = Int(@inbounds adapter.state_offsets[state + 1]) - 1
    return first, last
end

function _validate_batch_shape!(
    adapter::DendriticTrainingAdapter,
    batch::Data.CanonicalBatch,
)
    input = batch.input
    input.state_batch == length(adapter.common_states) || throw(DimensionMismatch(
        "state batch changed across persistent training team",
    ))
    input.width == size(adapter.student_q, 1) || throw(DimensionMismatch(
        "candidate width changed across persistent training team",
    ))
    input.capacity == length(adapter.components) || throw(DimensionMismatch(
        "candidate capacity changed across persistent training team",
    ))
    input.valid_count >= 1 || throw(ArgumentError(
        "prepared batch contains no valid candidates",
    ))
    input.valid_count <= input.capacity || throw(ArgumentError(
        "prepared batch valid_count exceeds capacity",
    ))
    return nothing
end

@inline _clear_gradient_groups!(::Tuple{}) = nothing

@inline function _clear_gradient_groups!(groups::Tuple)
    fill!(first(groups).gradient, 0.0f0)
    _clear_gradient_groups!(Base.tail(groups))
    return nothing
end

function Barrierless.prepare_batch!(
    adapter::DendriticTrainingAdapter,
    batch::Data.CanonicalBatch,
)
    _validate_batch_shape!(adapter, batch)
    adapter.active_generation != typemax(UInt64) || throw(OverflowError(
        "canonical barrierless attempt generation overflow",
    ))
    adapter.active_generation += UInt64(1)
    input = batch.input
    valid = input.valid_count
    fill!(adapter.forward_seen, 0x00)
    fill!(adapter.replay_seen, 0x00)
    fill!(adapter.raw_output, 0.0f0)
    fill!(adapter.raw_delta, 0.0f0)
    fill!(adapter.student_q, 0.0f0)
    fill!(adapter.q_gradient, 0.0f0)
    fill!(adapter.state_delta22, 0.0f0)
    fill!(adapter.state_value_delta, 0.0f0)
    _clear_gradient_groups!(adapter.registry.groups)
    Plasticity.begin_plasticity_batch!(adapter.plasticity_batch)
    adapter.optimizer_boundaries = 0
    adapter.mechanisms = MechanismCounters()
    adapter.expected = MechanismActivation()
    adapter.due = Local.preview_clocks(
        adapter.clocks, adapter.config.local_learning.schedule,
    )

    ordinal = 1
    adapter.state_offsets[1] = Int32(1)
    adapter.zero_based_state_offsets[1] = Int32(0)
    @inbounds for state in 1:input.state_batch
        count = Data.candidate_count(input, state)
        for candidate in 1:count
            ordinal <= valid || error("candidate compact order underflow")
            expected_flat = (state - 1) * input.width + candidate
            flat = Int(input.valid_flats[ordinal])
            flat == expected_flat || throw(ArgumentError(
                "valid candidate order is not state-major and contiguous",
            ))
            adapter.ordinal_flat[ordinal] = Int32(flat)
            adapter.ordinal_state[ordinal] = Int16(state)
            ordinal += 1
        end
        adapter.state_offsets[state + 1] = Int32(ordinal)
        adapter.zero_based_state_offsets[state + 1] = Int32(ordinal - 1)
    end
    ordinal == valid + 1 || error("candidate compact order overflow")
    required_slots = cld(valid, adapter.candidate_chunk_size)
    required_slots <= adapter.candidate_slot_capacity || error(
        "microbatch slot storage is smaller than the scheduler partition",
    )
    required_slots + input.state_batch <= adapter.slot_capacity || error(
        "state-common slot storage is smaller than the scheduler partition",
    )
    fill!(adapter.hard_delivery_slots, Int64(0))
    adapter.hard_event_deliveries = Int64(0)
    Graph.clear_hard_event_gradient!(adapter.reduced_hard_gradient)
    if _hard_event_due(adapter)
        fill!(adapter.common_hard_seed, 0.0f0)
        fill!(adapter.common_hard_seed_generation, UInt64(0))
        fill!(adapter.common_hard_seed_consumed, UInt8(0))
        @inbounds for slot in 1:(required_slots + input.state_batch)
            Graph.clear_hard_event_delta!(adapter.hard_delta_slots[slot])
        end
    end
    return adapter
end

function Barrierless.prepare_state_common!(
    adapter::DendriticTrainingAdapter,
    worker::DendriticTrainingWorker,
    batch::Data.CanonicalBatch,
    state::Int,
)
    1 <= state <= batch.input.state_batch || throw(BoundsError(
        adapter.common_states, state,
    ))
    generation = adapter.active_generation
    @inbounds adapter.common_prepare_generation[state] != generation || error(
        "state-common preparation published twice for state $state",
    )
    common = @inbounds adapter.common_states[state]
    Graph.prepare_state_common!(
        adapter.model,
        common,
        worker.graph,
        Data.state_input(batch.input, state),
    )
    @inbounds begin
        adapter.state_value[state] = common.state_value
        adapter.common_signature[state] = common.common_signature
        adapter.common_prepare_owner[state] = worker.worker_slot
        # Publication stamp is deliberately last.  The scheduler phase barrier
        # is the acquire boundary for the coordinator and candidate workers.
        adapter.common_prepare_generation[state] = generation
    end
    return nothing
end

function Barrierless.finish_state_common_phase!(
    adapter::DendriticTrainingAdapter,
    batch::Data.CanonicalBatch,
    state_total::Int,
)
    state_total == batch.input.state_batch || throw(DimensionMismatch(
        "scheduler state count differs from the canonical batch",
    ))
    generation = adapter.active_generation
    @inbounds for state in 1:state_total
        adapter.common_prepare_generation[state] == generation || error(
            "state-common preparation slot $state is missing or stale",
        )
        !iszero(adapter.common_prepare_owner[state]) || error(
            "state-common preparation slot $state has no worker owner",
        )
        adapter.common_states[state].ready || error(
            "state-common preparation slot $state is not finalized",
        )
        adapter.common_states[state].common_signature ==
            adapter.common_signature[state] || error(
                "state-common preparation signature changed before forward",
            )
    end
    return nothing
end

function Barrierless.begin_microbatch!(
    ::DendriticTrainingAdapter,
    worker::DendriticTrainingWorker,
    ::Data.CanonicalBatch,
    phase::UInt8,
    ::Int,
    ::Int,
    ::Int,
)
    Graph.reset_candidate_set!(worker.graph)
    if phase == Barrierless.REPLAY_PASS
        Graph.clear_gradient!(worker.graph)
        Graph.begin_local_microbatch!(worker.local_learner)
        _reset!(worker.counters)
        worker.hard_event_deliveries = Int64(0)
    elseif phase != Barrierless.FORWARD_PASS
        throw(ArgumentError("unknown canonical training phase $phase"))
    end
    return nothing
end

@inline function _flat_for_ordinal(
    adapter::DendriticTrainingAdapter,
    ordinal::Int,
)
    return Int(@inbounds adapter.ordinal_flat[ordinal])
end

@inline function _state_for_ordinal(
    adapter::DendriticTrainingAdapter,
    ordinal::Int,
)
    return Int(@inbounds adapter.ordinal_state[ordinal])
end

function Barrierless.run_candidate!(
    adapter::DendriticTrainingAdapter,
    worker::DendriticTrainingWorker,
    batch::Data.CanonicalBatch,
    ordinal::Int,
)
    flat = _flat_for_ordinal(adapter, ordinal)
    state = _state_for_ordinal(adapter, ordinal)
    component, signature = Graph.forward_candidate!(
        adapter.model,
        adapter.common_states[state],
        worker.graph,
        Data.candidate_input(batch.input, flat);
        mode=:cow,
    )
    _copy_component!(adapter.components[ordinal], component)
    adapter.forward_signature[ordinal] = signature
    adapter.forward_seen[ordinal] = 0x01
    return nothing
end

Barrierless.finish_forward_microbatch!(
    ::DendriticTrainingAdapter,
    ::DendriticTrainingWorker,
    ::Data.CanonicalBatch,
    ::Int,
    ::Int,
    ::Int,
) = nothing

@inline function _huber(error::Float32, delta::Float32)
    absolute = abs(error)
    return absolute <= delta ?
        0.5f0 * error * error : delta * (absolute - 0.5f0 * delta)
end

@inline _huber_derivative(error::Float32, delta::Float32) =
    clamp(error, -delta, delta)

@inline function _sigmoid(value::Float32)
    if value >= 0.0f0
        inverse = exp(-value)
        return inv(1.0f0 + inverse)
    end
    exponential = exp(value)
    return exponential / (1.0f0 + exponential)
end

function _finalize_auxiliary_loss!(
    adapter::DendriticTrainingAdapter,
    batch::Data.CanonicalBatch,
)
    config = adapter.config.auxiliary
    targets = batch.teacher
    valid = batch.input.valid_count
    death_count = 0
    @inbounds for ordinal in 1:valid
        flat = _flat_for_ordinal(adapter, ordinal)
        state = _state_for_ordinal(adapter, ordinal)
        candidate = flat - (state - 1) * batch.input.width
        death_count += targets.death_mask[candidate, state] != 0.0f0
    end
    inverse_death = inv(Float32(max(death_count, 1)))
    inverse_valid = inv(Float32(valid))
    death_loss = 0.0f0
    quantile_loss = 0.0f0
    geometry_loss = 0.0f0

    @inbounds for ordinal in 1:valid
        flat = _flat_for_ordinal(adapter, ordinal)
        state = _state_for_ordinal(adapter, ordinal)
        candidate = flat - (state - 1) * batch.input.width
        if targets.death_mask[candidate, state] != 0.0f0
            logit = adapter.raw_output[Output.DEATH_INDEX, ordinal]
            label = targets.death[candidate, state]
            death_loss += (
                max(logit, 0.0f0) - logit * label +
                log1p(exp(-abs(logit)))
            ) * inverse_death
            adapter.raw_delta[Output.DEATH_INDEX, ordinal] =
                config.death_weight * (_sigmoid(logit) - label) *
                inverse_death
        end

        quantile_scale = config.quantile_weight * inverse_valid /
            Float32(QUANTILE_COUNT)
        for output in QUANTILE_RANGE
            error = adapter.raw_output[output, ordinal] -
                targets.raw22[output, flat]
            quantile_loss += _huber(error, config.huber_delta) *
                inverse_valid / Float32(QUANTILE_COUNT)
            adapter.raw_delta[output, ordinal] = quantile_scale *
                _huber_derivative(error, config.huber_delta)
        end

        geometry_scale = config.geometry_weight * inverse_valid /
            Float32(GEOMETRY_COUNT)
        for output in GEOMETRY_RANGE
            error = adapter.raw_output[output, ordinal] -
                targets.raw22[output, flat]
            geometry_loss += _huber(error, config.huber_delta) *
                inverse_valid / Float32(GEOMETRY_COUNT)
            adapter.raw_delta[output, ordinal] = geometry_scale *
                _huber_derivative(error, config.huber_delta)
        end
    end
    return death_loss, quantile_loss, geometry_loss
end

function _assemble_component_bars!(
    adapter::DendriticTrainingAdapter,
    batch::Data.CanonicalBatch,
)
    @inbounds for state in 1:batch.input.state_batch
        first = Int(adapter.state_offsets[state])
        last = Int(adapter.state_offsets[state + 1]) - 1
        count = last - first + 1
        q_sum = 0.0
        for output in 1:OUTPUT_DIM
            signal_sum = 0.0
            for ordinal in first:last
                signal_sum += Float64(adapter.raw_delta[output, ordinal])
            end
            adapter.state_delta22[output, state] = Float32(signal_sum)
        end
        for ordinal in first:last
            # Form the private output-population cotangent first.  Only the
            # centered advantage lane depends on the complete candidate set;
            # patch that one scalar after its state mean is known.  This also
            # avoids a second SIMD Q reduction with a different reassociation.
            qbar = Output.assemble_output_pullback!(
                adapter.component_bars[ordinal],
                @view(adapter.raw_delta[:, ordinal]),
                adapter.components[ordinal],
                0.0f0,
            )
            adapter.q_bar[ordinal] = qbar
            q_sum += Float64(qbar)
        end
        adapter.state_value_delta[state] = Float32(q_sum)
        q_mean = Float32(q_sum / Float64(count))
        for ordinal in first:last
            bar = adapter.component_bars[ordinal]
            bar.advantage = adapter.q_bar[ordinal] - q_mean
            # V(s) is shared and receives its complete cotangent exactly once
            # in local_replay_state_common!, never once per candidate.
            bar.value = 0.0f0
        end
    end
    return nothing
end

@inline function _any_nonzero_signal(
    adapter::DendriticTrainingAdapter,
    valid::Int,
)
    @inbounds for ordinal in 1:valid, output in 1:OUTPUT_DIM
        iszero(adapter.raw_delta[output, ordinal]) || return true
    end
    return false
end

function _expected_mechanisms(
    adapter::DendriticTrainingAdapter,
    has_signal::Bool,
)
    local_config = adapter.config.local_learning
    due = adapter.due
    analog = due.analog && local_config.analog_multiplier > 0.0f0 &&
        has_signal && (
            local_config.feedback_scale > 0.0f0 ||
            local_config.predictor_scale > 0.0f0
        )
    utility = due.analog && local_config.analog_multiplier > 0.0f0 &&
        has_signal && local_config.utility_mode !== :none
    hard_event = _hard_event_due(adapter)
    return MechanismActivation(
        analog,
        analog,
        analog,
        hard_event,
        false,
        false,
        utility,
        false,
    )
end

function Barrierless.finalize_listnet!(
    adapter::DendriticTrainingAdapter,
    batch::Data.CanonicalBatch,
)
    valid = batch.input.valid_count
    all(==(0x01), @view(adapter.forward_seen[1:valid])) || error(
        "ListNet boundary observed an incomplete candidate set",
    )
    fill!(adapter.raw_delta, 0.0f0)
    fill!(adapter.student_q, 0.0f0)
    fill!(adapter.q_gradient, 0.0f0)

    @inbounds for state in 1:batch.input.state_batch
        first = Int(adapter.state_offsets[state])
        last = Int(adapter.state_offsets[state + 1]) - 1
        count = last - first + 1
        Graph.assemble_candidate_set!(
            @view(adapter.raw_output[:, first:last]),
            adapter.state_value[state],
            @view(adapter.components[first:last]),
            count,
        )
        for candidate in 1:count
            adapter.student_q[candidate, state] =
                adapter.raw_output[Output.Q_INDEX, first + candidate - 1]
        end
    end

    result = ListNet.listnet_loss_and_gradient!(
        adapter.q_gradient,
        adapter.listnet_scratch,
        adapter.student_q,
        batch.teacher.teacher_q,
        batch.input.counts,
        adapter.config.listnet,
    )
    adapter.listnet_result = result
    @inbounds for state in 1:batch.input.state_batch
        first = Int(adapter.state_offsets[state])
        count = Int(batch.input.counts[state])
        for candidate in 1:count
            adapter.raw_delta[Output.Q_INDEX, first + candidate - 1] =
                adapter.q_gradient[candidate, state]
        end
    end
    death_loss, quantile_loss, geometry_loss =
        _finalize_auxiliary_loss!(adapter, batch)
    _assemble_component_bars!(adapter, batch)
    auxiliary = adapter.config.auxiliary
    total = result.total_loss +
        auxiliary.death_weight * death_loss +
        auxiliary.quantile_weight * quantile_loss +
        auxiliary.geometry_weight * geometry_loss
    adapter.loss = CanonicalLossResult(
        total,
        result.listnet_kl,
        result.q_huber_loss,
        death_loss,
        quantile_loss,
        geometry_loss,
        valid,
    )
    adapter.expected = _expected_mechanisms(
        adapter, _any_nonzero_signal(adapter, valid),
    )
    return adapter.loss
end

function Barrierless.replay_candidate!(
    adapter::DendriticTrainingAdapter,
    worker::DendriticTrainingWorker,
    batch::Data.CanonicalBatch,
    ordinal::Int,
)
    state = _state_for_ordinal(adapter, ordinal)
    flat = _flat_for_ordinal(adapter, ordinal)
    copyto!(worker.delta22, @view(adapter.raw_delta[:, ordinal]))
    report = Graph.local_replay_candidate!(
        adapter.model,
        adapter.common_states[state],
        worker.graph,
        worker.local_learner,
        Data.candidate_input(batch.input, flat),
        worker.delta22,
        adapter.component_bars[ordinal],
        adapter.due;
        expected_signature=adapter.forward_signature[ordinal],
        mode=:cow,
        hard_state_id=state,
        hard_candidate_ordinal=ordinal,
    )
    _accumulate_local_report!(worker.counters, report)
    worker.hard_event_deliveries = Base.checked_add(
        worker.hard_event_deliveries,
        Int64(report.event_control_deliveries),
    )
    observation = Graph.local_plasticity_observation(worker.local_learner)
    Plasticity.record_candidate_plasticity!(
        adapter.plasticity_batch,
        ordinal,
        state,
        ordinal - Int(adapter.state_offsets[state]) + 1,
        observation.spike_count,
        observation.visit_count,
        observation.activity_sum,
        observation.incoming_conductance_sum,
        observation.task_utility_sum,
        observation.contact_activity_sum,
    )
    # The Graph method validates `expected_signature` before returning.
    adapter.replay_signature[ordinal] = adapter.forward_signature[ordinal]
    adapter.replay_seen[ordinal] = 0x01
    return nothing
end

@inline function _publish_reduction_slot!(
    adapter::DendriticTrainingAdapter,
    worker::DendriticTrainingWorker,
    slot::Int,
    kind::UInt8,
    logical_first::Int,
    logical_last::Int,
)
    1 <= slot <= adapter.slot_capacity || throw(BoundsError(
        adapter.gradient_slots, slot,
    ))
    generation = adapter.active_generation
    @inbounds adapter.slot_generation[slot] != generation || error(
        "reduction slot $slot was published twice in one attempt",
    )
    destination = @inbounds adapter.gradient_slots[slot]
    Graph.clear_gradient!(destination)
    Graph.accumulate_gradient!(destination, worker.graph.gradient)
    if _hard_event_due(adapter)
        Graph.publish_hard_event_delta!(
            @inbounds(adapter.hard_delta_slots[slot]),
            worker.local_learner,
            adapter.model,
            logical_first,
            logical_last,
        )
    end
    @inbounds begin
        adapter.mechanism_slots[slot] = _snapshot(worker.counters)
        adapter.hard_delivery_slots[slot] = worker.hard_event_deliveries
        adapter.slot_kind[slot] = kind
        adapter.slot_logical_first[slot] = Int32(logical_first)
        adapter.slot_logical_last[slot] = Int32(logical_last)
        adapter.slot_owner[slot] = worker.worker_slot
        # Publish after the complete payload and logical identity.
        adapter.slot_generation[slot] = generation
    end
    return nothing
end

function Barrierless.reduce_worker!(
    adapter::DendriticTrainingAdapter,
    worker::DendriticTrainingWorker,
    ::Data.CanonicalBatch,
    slot::Int,
    first::Int,
    last::Int,
)
    slot <= adapter.candidate_slot_capacity || throw(BoundsError(
        adapter.gradient_slots, slot,
    ))
    _publish_reduction_slot!(
        adapter,
        worker,
        slot,
        _CANDIDATE_REDUCTION_SLOT,
        first,
        last,
    )
    return nothing
end


function Barrierless.finish_candidate_replay_phase!(
    adapter::DendriticTrainingAdapter,
    batch::Data.CanonicalBatch,
    microbatch_count::Int,
)
    _hard_event_due(adapter) || return nothing
    expected_count = cld(
        batch.input.valid_count, adapter.candidate_chunk_size,
    )
    microbatch_count == expected_count || throw(DimensionMismatch(
        "candidate replay phase reported the wrong microbatch count",
    ))
    microbatch_count <= adapter.candidate_slot_capacity || throw(BoundsError(
        adapter.hard_delta_slots, microbatch_count,
    ))
    generation = adapter.active_generation
    @inbounds for state in 1:batch.input.state_batch
        adapter.common_hard_seed_generation[state] != generation || error(
            "candidate hard-event seed phase was finalized twice",
        )
        iszero(adapter.common_hard_seed_consumed[state]) || error(
            "candidate hard-event seed phase observed a consumed state",
        )
    end
    @inbounds for slot in 1:microbatch_count
        first = (slot - 1) * adapter.candidate_chunk_size + 1
        last = min(
            slot * adapter.candidate_chunk_size,
            batch.input.valid_count,
        )
        _preflight_reduction_slot!(
            adapter,
            slot,
            _CANDIDATE_REDUCTION_SLOT,
            first,
            last,
        )
        delta = adapter.hard_delta_slots[slot]
        Graph.accumulate_common_hard_event_seeds!(
            adapter.common_hard_seed,
            delta,
            adapter.model;
            expected_first_candidate=first,
            expected_last_candidate=last,
        )
    end
    @inbounds for state in 1:batch.input.state_batch
        # Per-state readiness is published only after every candidate slot has
        # been reduced in ascending logical/physical delivery order.
        adapter.common_hard_seed_generation[state] = generation
    end
    return nothing
end


function Barrierless.replay_state_common!(
    adapter::DendriticTrainingAdapter,
    worker::DendriticTrainingWorker,
    batch::Data.CanonicalBatch,
    state::Int,
    reduction_slot::Int,
)
    1 <= state <= batch.input.state_batch || throw(BoundsError(
        adapter.common_states, state,
    ))
    @inbounds adapter.common_prepare_generation[state] ==
        adapter.active_generation || error(
            "state-common replay observed an unprepared state $state",
        )
    hard_due = _hard_event_due(adapter)
    if hard_due
        @inbounds adapter.common_hard_seed_generation[state] ==
            adapter.active_generation || error(
                "state-common hard-event seed slab is missing or stale for state $state",
            )
        @inbounds iszero(adapter.common_hard_seed_consumed[state]) || error(
            "state-common hard-event seed slab was already consumed for state $state",
        )
    end
    Graph.reset_candidate_set!(worker.graph)
    Graph.clear_gradient!(worker.graph)
    Graph.begin_local_microbatch!(worker.local_learner)
    _reset!(worker.counters)
    worker.hard_event_deliveries = Int64(0)
    if hard_due
        Graph.load_common_hard_event_seeds!(
            worker.local_learner,
            @view(adapter.common_hard_seed[:, :, state]),
            state,
        )
    end
    common = @inbounds adapter.common_states[state]
    value_delta = @inbounds adapter.state_value_delta[state]
    expected_signature = @inbounds adapter.common_signature[state]
    report = Graph.local_replay_state_common!(
        adapter.model,
        common,
        worker.graph,
        worker.local_learner,
        Data.state_input(batch.input, state),
        @view(adapter.state_delta22[:, state]),
        value_delta,
        adapter.due;
        expected_signature,
        hard_state_id=state,
    )
    report.signature == expected_signature || error(
        "state-common replay signature changed after ListNet boundary",
    )
    worker.hard_event_deliveries = Base.checked_add(
        worker.hard_event_deliveries,
        Int64(report.event_control_deliveries),
    )
    if hard_due
        !Graph.common_hard_event_seed_loaded(worker.local_learner) || error(
            "state-common hard-event replay left its seed slab loaded",
        )
        Graph.common_hard_event_seed_consumed(worker.local_learner) || error(
            "state-common hard-event replay did not consume its seed slab",
        )
        !Graph.common_hard_event_seed_poisoned(worker.local_learner) || error(
            "state-common hard-event replay left a poisoned seed slab",
        )
    end
    _accumulate_local_report!(worker.counters, report)
    observation = Graph.local_plasticity_observation(worker.local_learner)
    Plasticity.record_state_common_plasticity!(
        adapter.plasticity_batch,
        state,
        observation.spike_count,
        observation.visit_count,
        observation.activity_sum,
        observation.incoming_conductance_sum,
        observation.task_utility_sum,
        observation.contact_activity_sum,
    )
    _publish_reduction_slot!(
        adapter,
        worker,
        reduction_slot,
        _STATE_COMMON_REDUCTION_SLOT,
        0,
        0,
    )
    hard_due && (@inbounds adapter.common_hard_seed_consumed[state] = 0x01)
    return nothing
end

@inline function _preflight_reduction_slot!(
    adapter::DendriticTrainingAdapter,
    slot::Int,
    kind::UInt8,
    logical_first::Int,
    logical_last::Int,
)
    @inbounds begin
        adapter.slot_generation[slot] == adapter.active_generation || error(
            "reduction slot $slot is missing or stale",
        )
        adapter.slot_kind[slot] == kind || error(
            "reduction slot $slot has the wrong ownership kind",
        )
        adapter.slot_logical_first[slot] == logical_first || error(
            "reduction slot $slot has the wrong logical start",
        )
        adapter.slot_logical_last[slot] == logical_last || error(
            "reduction slot $slot has the wrong logical end",
        )
        !iszero(adapter.slot_owner[slot]) || error(
            "reduction slot $slot has no worker owner",
        )
    end
    return nothing
end

@inline function _accumulate_hard_event_array!(destination, source)
    axes(destination) == axes(source) || throw(DimensionMismatch(
        "hard-event gradient groups have different axes",
    ))
    @inbounds @simd for index in eachindex(destination, source)
        destination[index] += source[index]
    end
    return destination
end

@inline function _accumulate_hard_event_gradient!(
    destination::Graph.ModelHardEventGradient,
    source::Graph.ModelHardEventGradient,
)
    _accumulate_hard_event_array!(
        destination.core_cell_raw, source.core_cell_raw,
    )
    _accumulate_hard_event_array!(
        destination.semantic_projection_raw, source.semantic_projection_raw,
    )
    _accumulate_hard_event_array!(destination.event_raw, source.event_raw)
    return destination
end

function Barrierless.deterministic_reduce!(
    adapter::DendriticTrainingAdapter,
    batch::Data.CanonicalBatch,
    microbatch_count::Int,
)
    valid = batch.input.valid_count
    all(==(0x01), @view(adapter.replay_seen[1:valid])) || error(
        "deterministic reduction observed an incomplete replay set",
    )
    microbatch_count <= adapter.candidate_slot_capacity || throw(BoundsError(
        adapter.gradient_slots, microbatch_count,
    ))
    state_total = batch.input.state_batch
    total_slots = microbatch_count + state_total
    total_slots <= adapter.slot_capacity || throw(BoundsError(
        adapter.gradient_slots, total_slots,
    ))
    @inbounds for slot in 1:microbatch_count
        first = (slot - 1) * adapter.candidate_chunk_size + 1
        last = min(
            slot * adapter.candidate_chunk_size,
            batch.input.valid_count,
        )
        _preflight_reduction_slot!(
            adapter,
            slot,
            _CANDIDATE_REDUCTION_SLOT,
            first,
            last,
        )
    end
    @inbounds for state in 1:state_total
        _preflight_reduction_slot!(
            adapter,
            microbatch_count + state,
            _STATE_COMMON_REDUCTION_SLOT,
            0,
            0,
        )
    end

    hard_due = _hard_event_due(adapter)
    if hard_due
        @inbounds for slot in 1:microbatch_count
            delta = adapter.hard_delta_slots[slot]
            Graph.hard_event_delta_sealed(delta) || error(
                "candidate hard-event delta slot $slot is not sealed",
            )
            Graph.hard_event_delta_candidate_range(delta) == (
                (slot - 1) * adapter.candidate_chunk_size + 1,
                min(
                    slot * adapter.candidate_chunk_size,
                    batch.input.valid_count,
                ),
            ) || error(
                "candidate hard-event delta slot $slot has the wrong range",
            )
            delta.reduced || error(
                "candidate hard-event delta slot $slot missed phase reduction",
            )
        end
        @inbounds for state in 1:state_total
            adapter.common_hard_seed_generation[state] ==
                adapter.active_generation || error(
                    "state-common hard-event seed slab $state is missing or stale",
                )
            adapter.common_hard_seed_consumed[state] == 0x01 || error(
                "state-common hard-event seed slab $state was not consumed exactly once",
            )
            slot = microbatch_count + state
            delta = adapter.hard_delta_slots[slot]
            Graph.accumulate_common_hard_event_seeds!(
                adapter.common_hard_seed,
                delta,
                adapter.model;
                expected_first_candidate=0,
                expected_last_candidate=0,
            )
        end
    end

    # No reduction scratch is touched until every worker publication has been
    # validated against this attempt generation and its logical owner.
    Graph.clear_gradient!(adapter.reduced_gradient)
    Graph.clear_hard_event_gradient!(adapter.reduced_hard_gradient)
    mechanisms = MechanismCounters()
    hard_event_deliveries = Int64(0)
    @inbounds for slot in 1:total_slots
        Graph.accumulate_gradient!(
            adapter.reduced_gradient, adapter.gradient_slots[slot],
        )
        if hard_due
            _accumulate_hard_event_gradient!(
                adapter.reduced_hard_gradient,
                Graph.hard_event_gradient(adapter.hard_delta_slots[slot]),
            )
            hard_event_deliveries = Base.checked_add(
                hard_event_deliveries,
                adapter.hard_delivery_slots[slot],
            )
        end
        mechanisms = _add(mechanisms, adapter.mechanism_slots[slot])
    end
    adapter.mechanisms = mechanisms
    adapter.hard_event_deliveries = hard_event_deliveries
    return adapter.reduced_gradient
end

@inline function _assert_counter(active::Bool, count::Int64, name::Symbol)
    active && count == 0 && error(
        "canonical mechanism $name was active but fired zero times",
    )
    return nothing
end

function _assert_mechanisms!(
    expected::MechanismActivation,
    counts::MechanismCounters;
    include_slow::Bool,
)
    _assert_counter(
        expected.decolle_signal_nonzero,
        counts.decolle_signal_nonzero,
        :decolle_signal_nonzero,
    )
    _assert_counter(
        expected.subthreshold_updates,
        counts.subthreshold_updates,
        :subthreshold_updates,
    )
    _assert_counter(
        expected.nonspiking_updates,
        counts.nonspiking_updates,
        :nonspiking_updates,
    )
    _assert_counter(
        expected.hard_event_control_updates,
        counts.hard_event_control_updates,
        :hard_event_control_updates,
    )
    _assert_counter(
        expected.utility_updates,
        counts.utility_updates,
        :utility_updates,
    )
    if include_slow
        _assert_counter(
            expected.homeostasis_events,
            counts.homeostasis_events,
            :homeostasis_events,
        )
        _assert_counter(
            expected.synaptic_scaling_events,
            counts.synaptic_scaling_events,
            :synaptic_scaling_events,
        )
        _assert_counter(expected.rewires, counts.rewires, :rewires)
    end
    return nothing
end

@inline function _checked_cumulative_add(
    current::UInt64,
    increment::Int64,
    name::Symbol,
)
    increment >= 0 || throw(ArgumentError(
        "per-update mechanism $name cannot be negative",
    ))
    amount = UInt64(increment)
    current <= typemax(UInt64) - amount || throw(OverflowError(
        "cumulative mechanism $name overflow",
    ))
    return current + amount
end

"""Preflight cumulative telemetry before any optimizer/plasticity mutation."""
function _preflight_cumulative_mechanisms!(
    cumulative::CumulativeMechanismCounters,
    update::MechanismCounters,
)
    _checked_cumulative_add(
        cumulative.decolle_signal_nonzero,
        update.decolle_signal_nonzero,
        :decolle_signal_nonzero,
    )
    _checked_cumulative_add(
        cumulative.subthreshold_updates,
        update.subthreshold_updates,
        :subthreshold_updates,
    )
    _checked_cumulative_add(
        cumulative.nonspiking_updates,
        update.nonspiking_updates,
        :nonspiking_updates,
    )
    _checked_cumulative_add(
        cumulative.hard_event_control_updates,
        update.hard_event_control_updates,
        :hard_event_control_updates,
    )
    _checked_cumulative_add(
        cumulative.utility_updates,
        update.utility_updates,
        :utility_updates,
    )
    _checked_cumulative_add(
        cumulative.rewires,
        update.rewires,
        :rewires,
    )
    return nothing
end

function _preflight_cumulative_slow_mechanisms!(
    adapter::DendriticTrainingAdapter,
)
    cumulative = adapter.cumulative_mechanisms
    cumulative.homeostasis_events <=
        typemax(UInt64) - UInt64(Graph.TOTAL_NODE_COUNT) || throw(
            OverflowError("cumulative mechanism homeostasis_events overflow"),
        )
    cumulative.synaptic_scaling_events <=
        typemax(UInt64) - UInt64(length(adapter.event_destination)) || throw(
            OverflowError(
                "cumulative mechanism synaptic_scaling_events overflow",
            ),
        )
    return nothing
end

function _commit_cumulative_mechanisms!(
    cumulative::CumulativeMechanismCounters,
    update::MechanismCounters,
)
    cumulative.decolle_signal_nonzero += UInt64(update.decolle_signal_nonzero)
    cumulative.subthreshold_updates += UInt64(update.subthreshold_updates)
    cumulative.nonspiking_updates += UInt64(update.nonspiking_updates)
    cumulative.hard_event_control_updates +=
        UInt64(update.hard_event_control_updates)
    cumulative.homeostasis_events += UInt64(update.homeostasis_events)
    cumulative.synaptic_scaling_events +=
        UInt64(update.synaptic_scaling_events)
    cumulative.utility_updates += UInt64(update.utility_updates)
    cumulative.rewires += UInt64(update.rewires)
    return cumulative
end

@inline function _optimizer_due_mask(adapter::DendriticTrainingAdapter)
    local_config = adapter.config.local_learning
    analog = adapter.due.analog && local_config.analog_multiplier > 0.0f0
    hard = adapter.due.hard_event &&
        local_config.hard_event_multiplier > 0.0f0
    recurrent = analog || hard
    return (recurrent, recurrent, recurrent, true, true)
end

@inline function _apply_dual_optimizer_boundary!(
    adapter::DendriticTrainingAdapter,
    analog_buffers,
    hard_buffers,
    due_mask::Tuple,
)
    return Optimizer.apply_dual_optimizer_boundary!(
        adapter.optimizer_state,
        adapter.registry,
        analog_buffers,
        hard_buffers,
        adapter.config.optimizer;
        analog_gradient_scale=1.0f0,
        hard_event_gradient_scale=1.0f0,
        due_mask,
    )
end

@noinline function _nonfinite_optimizer_state(value, group::Symbol, field::Symbol)
    throw(DomainError(value, (group=group, field=field)))
end

@inline function _assert_all_finite(
    array,
    group::Symbol,
    field::Symbol,
)
    @inbounds for value in array
        isfinite(value) || _nonfinite_optimizer_state(value, group, field)
    end
    return nothing
end


@inline function _preflight_optimizer_groups!(
    ::Tuple{},
    ::Tuple{},
    ::Vector{UInt64},
    ::Tuple{},
    ::Int,
)
    return nothing
end

@inline function _preflight_optimizer_groups!(
    groups::Tuple,
    moments::Tuple,
    group_steps::Vector{UInt64},
    due_mask::Tuple,
    index::Int,
)
    group = first(groups)
    moment = first(moments)
    _assert_all_finite(group.parameter, group.name, :parameter)
    _assert_all_finite(moment.first, group.name, :first_moment)
    _assert_all_finite(moment.second, group.name, :second_moment)
    group.multiplier > 0.0f0 &&
        _assert_all_finite(group.gradient, group.name, :gradient)
    step_at_limit = @inbounds group_steps[index] == typemax(UInt64)
    first(due_mask) && group.multiplier > 0.0f0 && step_at_limit && throw(
            OverflowError("optimizer group clock overflow for $(group.name)"),
        )
    _preflight_optimizer_groups!(
        Base.tail(groups),
        Base.tail(moments),
        group_steps,
        Base.tail(due_mask),
        index + 1,
    )
    return nothing
end

function _preflight_optimizer_transaction!(
    adapter::DendriticTrainingAdapter,
    due_mask::Tuple,
)
    state = adapter.optimizer_state
    registry = adapter.registry
    Optimizer.assert_registry_match(state, registry)
    length(due_mask) == length(registry.groups) || throw(DimensionMismatch(
        "optimizer due mask differs from the canonical registry",
    ))
    state.total_step != typemax(UInt64) || throw(OverflowError(
        "total optimizer clock overflow",
    ))
    _preflight_optimizer_groups!(
        registry.groups,
        state.moments,
        state.group_steps,
        due_mask,
        1,
    )
    Optimizer.gradient_norm(registry; due_mask)
    return nothing
end

@inline function _preflight_slow_counter_clocks!(
    adapter::DendriticTrainingAdapter,
)
    adapter.due.homeostasis || return nothing
    state = adapter.plasticity_state
    state.homeostasis_events <=
        typemax(UInt64) - UInt64(Graph.TOTAL_NODE_COUNT) || throw(
            OverflowError("intrinsic-homeostasis event counter overflow"),
        )
    state.synaptic_scaling_events <=
        typemax(UInt64) - UInt64(length(adapter.event_destination)) || throw(
            OverflowError("synaptic-scaling event counter overflow"),
        )
    return nothing
end

@inline function _homeostasis_driver_exists(adapter::DendriticTrainingAdapter)
    due = adapter.due.homeostasis
    config = adapter.config.local_learning.plasticity
    enabled = config.threshold_homeostasis_step > 0.0f0 ||
        config.adaptation_homeostasis_step > 0.0f0
    due && enabled || return false
    core_enabled = adapter.registry.groups[1].multiplier > 0.0f0
    output_enabled = adapter.registry.groups[4].multiplier > 0.0f0
    @inbounds for cell in eachindex(adapter.plasticity_state.firing_rate)
        active_group = cell <= Graph.CORE_NODE_COUNT ?
            core_enabled : output_enabled
        active_group || continue
        rate = adapter.plasticity_state.firing_rate[cell]
        (rate < config.target_rate_min || rate > config.target_rate_max) &&
            return true
    end
    return false
end

@inline function _scaling_driver_exists(adapter::DendriticTrainingAdapter)
    due = adapter.due.homeostasis
    config = adapter.config.local_learning.plasticity
    due && config.synaptic_scaling_rate > 0.0f0 &&
        adapter.registry.groups[3].multiplier > 0.0f0 || return false
    @inbounds for destination in adapter.event_destination
        iszero(destination) && continue
        rate = adapter.plasticity_state.firing_rate[Int(destination)]
        (rate < config.target_rate_min || rate > config.target_rate_max) &&
            return true
    end
    return false
end

@inline function _set_slow_expectation!(
    adapter::DendriticTrainingAdapter,
    homeostasis::Bool,
    scaling::Bool,
)
    current = adapter.expected
    adapter.expected = MechanismActivation(
        current.decolle_signal_nonzero,
        current.subthreshold_updates,
        current.nonspiking_updates,
        current.hard_event_control_updates,
        homeostasis,
        scaling,
        current.utility_updates,
        false,
    )
    return adapter.expected
end

@noinline function _reject_preflight_reset(name, index)
    error("plasticity preflight attempted to reset optimizer moments")
end

function Barrierless.apply_update!(
    adapter::DendriticTrainingAdapter,
    ::Data.CanonicalBatch,
)
    adapter.optimizer_boundaries == 0 || error(
        "more than one optimizer boundary in a canonical update",
    )
    # No parameter is touched if the forward/local mechanisms already violate
    # their declared contract.
    _assert_mechanisms!(
        adapter.expected, adapter.mechanisms; include_slow=false,
    )
    if adapter.expected.hard_event_control_updates
        adapter.hard_event_deliveries > 0 || error(
            "hard-event control was active but evaluated zero deliveries",
        )
    end
    adapter.updates != typemax(UInt64) || throw(OverflowError(
        "canonical training update counter overflow",
    ))
    schedule = adapter.config.local_learning.schedule
    Local.preview_clocks(adapter.clocks, schedule) == adapter.due || throw(
        ArgumentError("canonical learning-clock preview became stale"),
    )
    due_mask = _optimizer_due_mask(adapter)
    _preflight_optimizer_transaction!(adapter, due_mask)
    _preflight_slow_counter_clocks!(adapter)
    _preflight_cumulative_mechanisms!(
        adapter.cumulative_mechanisms,
        adapter.mechanisms,
    )
    _preflight_cumulative_slow_mechanisms!(adapter)
    local_config = adapter.config.local_learning
    plasticity_config = adapter.config.local_learning.plasticity
    utility_due = adapter.due.analog &&
        local_config.analog_multiplier > 0.0f0 &&
        local_config.utility_mode === :combined
    plasticity_stats = Plasticity.preflight_canonical_plasticity(
        adapter.plasticity_state,
        adapter.plasticity_batch,
        plasticity_config,
        adapter.zero_based_state_offsets,
        ;
        utility_due,
    )
    cell_groups = (
        adapter.registry.groups[1],
        adapter.registry.groups[4],
    )
    cell_offsets = (0, Graph.CORE_NODE_COUNT)
    Plasticity.apply_intrinsic_homeostasis!(
        adapter.plasticity_state,
        plasticity_config,
        false,
        cell_groups,
        cell_offsets,
        _reject_preflight_reset,
    ) == 0 || error("homeostasis preflight mutated parameters")
    Plasticity.apply_synaptic_scaling!(
        adapter.plasticity_state,
        plasticity_config,
        false,
        adapter.registry.groups[3],
        adapter.event_destination,
        _reject_preflight_reset,
    ) == 0 || error("synaptic-scaling preflight mutated parameters")

    # The transaction begins only after every fallible persistent-state check.
    local_config = adapter.config.local_learning
    analog_due = adapter.due.analog &&
        local_config.analog_multiplier > 0.0f0
    hard_due = _hard_event_due(adapter)
    optimizer_stats = if analog_due
        if hard_due
            _apply_dual_optimizer_boundary!(
                adapter,
                adapter.analog_full_buffers,
                adapter.hard_recurrent_buffers,
                due_mask,
            )
        else
            _apply_dual_optimizer_boundary!(
                adapter,
                adapter.analog_full_buffers,
                adapter.hard_zero_buffers,
                due_mask,
            )
        end
    elseif hard_due
        _apply_dual_optimizer_boundary!(
            adapter,
            adapter.analog_output_buffers,
            adapter.hard_recurrent_buffers,
            due_mask,
        )
    else
        _apply_dual_optimizer_boundary!(
            adapter,
            adapter.analog_output_buffers,
            adapter.hard_zero_buffers,
            due_mask,
        )
    end
    adapter.optimizer_boundaries = 1
    committed_stats = Plasticity.reduce_canonical_plasticity!(
        adapter.plasticity_state,
        adapter.plasticity_batch,
        plasticity_config,
        adapter.zero_based_state_offsets,
        ;
        utility_due,
    )
    committed_stats == plasticity_stats || error(
        "canonical plasticity commit diverged from its pure preflight",
    )
    reset = Plasticity.OptimizerMomentReset(
        adapter.optimizer_state, adapter.registry,
    )
    homeostasis_expected = _homeostasis_driver_exists(adapter)
    scaling_expected = _scaling_driver_exists(adapter)
    homeostasis_events = Plasticity.apply_intrinsic_homeostasis!(
        adapter.plasticity_state,
        plasticity_config,
        adapter.due.homeostasis,
        cell_groups,
        cell_offsets,
        reset,
    )
    scaling_events = Plasticity.apply_synaptic_scaling!(
        adapter.plasticity_state,
        plasticity_config,
        adapter.due.homeostasis,
        adapter.registry.groups[3],
        adapter.event_destination,
        reset,
    )
    _set_slow_expectation!(
        adapter, homeostasis_expected, scaling_expected,
    )
    adapter.mechanisms = _add(
        adapter.mechanisms,
        MechanismCounters(
            0, 0, 0, 0, homeostasis_events, scaling_events, 0, 0,
        ),
    )
    _assert_mechanisms!(
        adapter.expected, adapter.mechanisms; include_slow=true,
    )
    _commit_cumulative_mechanisms!(
        adapter.cumulative_mechanisms,
        adapter.mechanisms,
    )
    Graph.refresh_cache!(adapter.model)
    @inbounds for state in adapter.common_states
        Graph.refresh_state_initial!(adapter.model, state)
    end
    Local.commit_clocks!(adapter.clocks, schedule, adapter.due)
    adapter.updates += UInt64(1)
    return TrainingUpdateResult(
        adapter.loss,
        adapter.mechanisms,
        optimizer_stats,
        adapter.updates,
    )
end

"""Active handle valid only inside one persistent native-worker team."""
mutable struct CanonicalTrainer{S}
    session::S
    active::Bool
end

function CanonicalTrainer(session::S) where {S<:Barrierless.CanonicalSession}
    session.executor.adapter isa DendriticTrainingAdapter || throw(
        ArgumentError("trainer requires the concrete dendritic adapter"),
    )
    return CanonicalTrainer{S}(session, false)
end

@inline update_count(trainer::CanonicalTrainer) =
    trainer.session.executor.adapter.updates

@inline mechanism_counts(trainer::CanonicalTrainer) =
    trainer.session.executor.adapter.mechanisms

@inline cumulative_mechanism_counts(trainer::CanonicalTrainer) =
    trainer.session.executor.adapter.cumulative_mechanisms

@inline function _checkpoint_group_contract(group)
    return Checkpoint.ParameterGroupContract(
        group.name,
        group.transform_kind,
        group.multiplier,
        group.lower_bound,
        group.upper_bound,
        group.projected_lower_raw,
        group.projected_upper_raw,
        Tuple(size(group.parameter)),
    )
end

function _checkpoint_registry(adapter::DendriticTrainingAdapter)
    groups = adapter.registry.groups
    length(groups) == 5 || throw(DimensionMismatch(
        "canonical checkpoint requires exactly five parameter groups",
    ))
    return (
        _checkpoint_group_contract(groups[1]),
        _checkpoint_group_contract(groups[2]),
        _checkpoint_group_contract(groups[3]),
        _checkpoint_group_contract(groups[4]),
        _checkpoint_group_contract(groups[5]),
    )
end

@inline function _checkpoint_parameter_state(arrays::Tuple)
    length(arrays) == 5 || throw(DimensionMismatch(
        "canonical parameter state requires exactly five arrays",
    ))
    return Checkpoint.CanonicalParameterState(
        arrays[1], arrays[2], arrays[3], arrays[4], arrays[5],
    )
end

function _checkpoint_optimizer_state(
    adapter::DendriticTrainingAdapter,
    registry::NTuple{5,Checkpoint.ParameterGroupContract},
)
    state = adapter.optimizer_state
    Optimizer.assert_registry_match(state, adapter.registry)
    length(state.group_steps) == 5 || throw(DimensionMismatch(
        "optimizer group clocks differ from the canonical registry",
    ))
    parameters = _checkpoint_parameter_state(
        map(group -> group.parameter, adapter.registry.groups),
    )
    first_moments = _checkpoint_parameter_state(
        map(moment -> moment.first, state.moments),
    )
    second_moments = _checkpoint_parameter_state(
        map(moment -> moment.second, state.moments),
    )
    group_steps = (
        state.group_steps[1],
        state.group_steps[2],
        state.group_steps[3],
        state.group_steps[4],
        state.group_steps[5],
    )
    return Checkpoint.CanonicalOptimizerStateSnapshot(
        registry,
        parameters,
        first_moments,
        second_moments,
        group_steps,
        state.total_step,
    )
end

@inline function _checkpoint_clock(adapter::DendriticTrainingAdapter)
    clock = adapter.clocks
    return Checkpoint.LearningClockSnapshot(
        clock.update,
        clock.analog_ticks,
        clock.hard_event_ticks,
        clock.homeostasis_ticks,
        clock.structure_ticks,
    )
end

@inline function _checkpoint_mechanisms(adapter::DendriticTrainingAdapter)
    counters = adapter.cumulative_mechanisms
    return Checkpoint.MechanismCounterSnapshot(
        counters.decolle_signal_nonzero,
        counters.subthreshold_updates,
        counters.nonspiking_updates,
        counters.hard_event_control_updates,
        counters.homeostasis_events,
        counters.synaptic_scaling_events,
        counters.utility_updates,
        counters.rewires,
    )
end

@inline function _checkpoint_plasticity(adapter::DendriticTrainingAdapter)
    state = adapter.plasticity_state
    return Checkpoint.PlasticityStateSnapshot(
        state.firing_rate,
        state.activity_ema,
        state.incoming_conductance_ema,
        state.utility,
        state.reduced_batches,
        state.homeostasis_events,
        state.synaptic_scaling_events,
        state.utility_updates,
        state.rewires,
    )
end

function _validate_checkpoint_run_boundary!(
    trainer::CanonicalTrainer,
    sampler::Sampler.DeterministicEpochSampler,
    contract::Checkpoint.CanonicalRunContract,
)
    Checkpoint.run_contract_fingerprint(contract)
    session = trainer.session
    executor = session.executor
    adapter = executor.adapter
    input = executor.batch.input
    input.state_batch == contract.state_batch || throw(DimensionMismatch(
        "live state batch differs from the run contract",
    ))
    input.width == contract.candidate_width || throw(DimensionMismatch(
        "live candidate width differs from the run contract",
    ))
    executor.candidate_chunk_size == contract.chunk_size || throw(
        ArgumentError("live executor chunk differs from the run contract"),
    )
    adapter.candidate_chunk_size == contract.chunk_size || throw(
        ArgumentError("live adapter chunk differs from the run contract"),
    )
    session.scheduler.worker_count == contract.workers || throw(ArgumentError(
        "live worker count differs from the run contract",
    ))
    Barrierless.SchedulerCore.Queue.capacity(session.scheduler.queue) ==
        contract.queue_capacity || throw(ArgumentError(
            "live queue capacity differs from the run contract",
        ))
    session.scheduler.binding_mode === contract.binding || throw(ArgumentError(
        "live scheduler binding differs from the run contract",
    ))
    training_config_fingerprint(adapter.config) ==
        contract.training_config_fingerprint || throw(ArgumentError(
            "live training configuration differs from the run contract",
        ))
    Checkpoint.architecture_fingerprint(adapter.model.config) ==
        contract.architecture_fingerprint || throw(ArgumentError(
            "live architecture differs from the run contract",
        ))
    Checkpoint.topology_fingerprint(adapter.model) ==
        contract.topology_fingerprint || throw(ArgumentError(
            "live topology/order differs from the run contract",
        ))
    sampler.seed == contract.sampler_seed || throw(ArgumentError(
        "live sampler seed differs from the run contract",
    ))
    length(sampler.source_rows) == contract.training_state_count || throw(
        DimensionMismatch("live sampler source count differs from the dataset"),
    )
    adapter.updates <= UInt64(contract.planned_max_updates) || throw(
        ArgumentError("live updates exceed the run plan"),
    )
    return nothing
end

"""Extract one complete schema2 snapshot from an idle production trainer."""
function checkpoint_components(
    trainer::CanonicalTrainer,
    sampler::Sampler.DeterministicEpochSampler,
    contract::Checkpoint.CanonicalRunContract,
)
    trainer.active && error("cannot checkpoint an active train_update!")
    executor = trainer.session.executor
    executor.update_active && error("cannot checkpoint an active executor update")
    _validate_checkpoint_run_boundary!(trainer, sampler, contract)
    registry = _checkpoint_registry(executor.adapter)
    return Checkpoint.build_training_snapshot(
        executor.adapter.model,
        executor.adapter.config.local_learning,
        executor.adapter.config.optimizer,
        _checkpoint_optimizer_state(executor.adapter, registry),
        _checkpoint_clock(executor.adapter),
        _checkpoint_mechanisms(executor.adapter),
        executor.adapter.updates,
        _checkpoint_plasticity(executor.adapter),
        Checkpoint.SamplerStateSnapshot(Sampler.sampler_snapshot(sampler)),
        contract,
    )
end

@inline _all_zero(values) = all(iszero, values)

@inline function _mechanisms_are_zero(counters)
    return all(name -> iszero(getfield(counters, name)), _MECHANISM_KEYS)
end

function _assert_virgin_restore_target!(
    trainer::CanonicalTrainer,
    sampler::Sampler.DeterministicEpochSampler,
)
    trainer.active && error("checkpoint restore requires an inactive trainer")
    session = trainer.session
    executor = session.executor
    adapter = executor.adapter
    executor.update_active && error(
        "checkpoint restore requires an inactive executor",
    )
    executor.state_total == 0 && executor.candidate_total == 0 &&
        executor.microbatch_total == 0 || throw(ArgumentError(
            "checkpoint restore target has executed an update attempt",
        ))
    adapter.updates == 0 && adapter.active_generation == 0 &&
        adapter.optimizer_boundaries == 0 || throw(ArgumentError(
            "checkpoint restore target is not virgin",
        ))
    Optimizer.assert_registry_match(adapter.optimizer_state, adapter.registry)
    adapter.optimizer_state.total_step == 0 &&
        all(iszero, adapter.optimizer_state.group_steps) || throw(ArgumentError(
            "checkpoint restore target has optimizer progress",
        ))
    @inbounds for moments in adapter.optimizer_state.moments
        _all_zero(moments.first) && _all_zero(moments.second) || throw(
            ArgumentError("checkpoint restore target has optimizer moments"),
        )
    end
    clock = adapter.clocks
    clock.update == 0 && clock.analog_ticks == 0 &&
        clock.hard_event_ticks == 0 && clock.homeostasis_ticks == 0 &&
        clock.structure_ticks == 0 || throw(ArgumentError(
            "checkpoint restore target has learning-clock progress",
        ))
    adapter.due == Local.DuePlasticityClocks(false, false, false, false) ||
        throw(ArgumentError("checkpoint restore target has a due-clock token"))
    adapter.expected == MechanismActivation() || throw(ArgumentError(
        "checkpoint restore target has mechanism activation state",
    ))
    _mechanisms_are_zero(adapter.mechanisms) &&
        _mechanisms_are_zero(adapter.cumulative_mechanisms) || throw(
            ArgumentError("checkpoint restore target has mechanism telemetry"),
        )
    plasticity = adapter.plasticity_state
    plasticity_config = adapter.config.local_learning.plasticity
    initial_rate = Float32(
        (plasticity_config.target_rate_min +
         plasticity_config.target_rate_max) / 2,
    )
    all(==(initial_rate), plasticity.firing_rate) &&
        _all_zero(plasticity.activity_ema) &&
        _all_zero(plasticity.incoming_conductance_ema) &&
        _all_zero(plasticity.utility) || throw(ArgumentError(
            "checkpoint restore target has non-constructor plasticity state",
        ))
    plasticity.reduced_batches == 0 && plasticity.homeostasis_events == 0 &&
        plasticity.synaptic_scaling_events == 0 &&
        plasticity.utility_updates == 0 && plasticity.rewires == 0 || throw(
            ArgumentError("checkpoint restore target has plasticity progress"),
        )
    adapter.listnet_result === nothing || throw(ArgumentError(
        "checkpoint restore target has a published ListNet result",
    ))
    adapter.hard_event_deliveries == 0 &&
        _all_zero(adapter.common_prepare_generation) &&
        _all_zero(adapter.common_hard_seed_generation) &&
        _all_zero(adapter.slot_generation) || throw(ArgumentError(
            "checkpoint restore target has replay/publication state",
        ))
    Sampler.sampler_consumed_rows(sampler) == 0 || throw(ArgumentError(
        "checkpoint restore requires a virgin sampler",
    ))
    return nothing
end

function _validate_prepared_against_live!(
    trainer::CanonicalTrainer,
    sampler::Sampler.DeterministicEpochSampler,
    prepared::Checkpoint.PreparedTrainingCheckpoint,
)
    snapshot = Checkpoint.validate_prepared_checkpoint(prepared)
    adapter = trainer.session.executor.adapter
    isequal(snapshot.learning_config, adapter.config.local_learning) || throw(
        ArgumentError("prepared learning configuration differs from the trainer"),
    )
    isequal(snapshot.optimizer_config, adapter.config.optimizer) || throw(
        ArgumentError("prepared optimizer configuration differs from the trainer"),
    )
    snapshot.optimizer.registry == _checkpoint_registry(adapter) || throw(
        ArgumentError("prepared parameter registry differs from the trainer"),
    )
    _validate_checkpoint_run_boundary!(trainer, sampler, snapshot.run_contract)
    sampler.seed == prepared.sampler.seed &&
        sampler.source_sha256 == prepared.sampler.source_sha256 &&
        sampler.source_rows == prepared.sampler.source_rows || throw(
            ArgumentError("prepared sampler source differs from the live sampler"),
        )
    _assert_virgin_restore_target!(trainer, sampler)
    return snapshot
end

@inline function _restore_parameter_state!(arrays::Tuple, state)
    saved = (
        state.core_cell_raw,
        state.semantic_projection_raw,
        state.event_raw,
        state.output_cell_raw,
        state.output_projection_raw,
    )
    @inbounds for index in 1:5
        copyto!(arrays[index], saved[index])
    end
    return nothing
end

"""Commit a fully prepared schema2 payload into a virgin inactive trainer."""
function restore_training_checkpoint!(
    trainer::CanonicalTrainer,
    sampler::Sampler.DeterministicEpochSampler,
    prepared::Checkpoint.PreparedTrainingCheckpoint,
)
    snapshot = _validate_prepared_against_live!(trainer, sampler, prepared)
    adapter = trainer.session.executor.adapter
    optimizer = adapter.optimizer_state
    _restore_parameter_state!(
        map(group -> group.parameter, adapter.registry.groups),
        snapshot.optimizer.parameters,
    )
    _restore_parameter_state!(
        map(moment -> moment.first, optimizer.moments),
        snapshot.optimizer.first_moments,
    )
    _restore_parameter_state!(
        map(moment -> moment.second, optimizer.moments),
        snapshot.optimizer.second_moments,
    )
    copyto!(optimizer.group_steps, snapshot.optimizer.group_steps)
    optimizer.total_step = snapshot.optimizer.total_step

    clock = snapshot.learning_clock
    adapter.clocks.update = clock.update
    adapter.clocks.analog_ticks = clock.analog_ticks
    adapter.clocks.hard_event_ticks = clock.hard_event_ticks
    adapter.clocks.homeostasis_ticks = clock.homeostasis_ticks
    adapter.clocks.structure_ticks = clock.structure_ticks

    mechanisms = snapshot.cumulative_mechanisms
    cumulative = adapter.cumulative_mechanisms
    @inbounds for name in _MECHANISM_KEYS
        setfield!(cumulative, name, getfield(mechanisms, name))
    end

    saved_plasticity = snapshot.plasticity
    plasticity = adapter.plasticity_state
    copyto!(plasticity.firing_rate, saved_plasticity.firing_rate)
    copyto!(plasticity.activity_ema, saved_plasticity.activity_ema)
    copyto!(
        plasticity.incoming_conductance_ema,
        saved_plasticity.incoming_conductance_ema,
    )
    copyto!(plasticity.utility, saved_plasticity.utility)
    plasticity.reduced_batches = saved_plasticity.reduced_batches
    plasticity.homeostasis_events = saved_plasticity.homeostasis_events
    plasticity.synaptic_scaling_events =
        saved_plasticity.synaptic_scaling_events
    plasticity.utility_updates = saved_plasticity.utility_updates
    plasticity.rewires = saved_plasticity.rewires

    restored_sampler = prepared.sampler
    copyto!(sampler.permutation, restored_sampler.permutation)
    sampler.position.epoch = restored_sampler.position.epoch
    sampler.position.cursor = restored_sampler.position.cursor
    adapter.updates = snapshot.training_updates

    # Scratch and latest per-update telemetry are deliberately not serialized.
    adapter.mechanisms = MechanismCounters()
    adapter.expected = MechanismActivation()
    adapter.due = Local.DuePlasticityClocks(false, false, false, false)
    adapter.listnet_result = nothing
    adapter.hard_event_deliveries = 0
    Graph.refresh_cache!(adapter.model)
    @inbounds for state in adapter.common_states
        Graph.refresh_state_initial!(adapter.model, state)
    end
    return trainer
end

"""Prepare and restore one loaded typed snapshot without exposing registry internals."""
function restore_training_checkpoint!(
    trainer::CanonicalTrainer,
    sampler::Sampler.DeterministicEpochSampler,
    snapshot::Checkpoint.CanonicalTrainingStateSnapshot,
    contract::Checkpoint.CanonicalRunContract,
)
    adapter = trainer.session.executor.adapter
    prepared = Checkpoint.prepare_training_checkpoint(
        snapshot,
        adapter.model,
        _checkpoint_registry(adapter),
        adapter.config.local_learning,
        adapter.config.optimizer,
        contract,
        sampler.source_rows,
    )
    return restore_training_checkpoint!(trainer, sampler, prepared)
end

"""
The sole production learning operation. There is deliberately no exact-oracle
mode flag; exact reverse is reachable only from the diagnostic module.
"""
function train_update!(
    trainer::CanonicalTrainer,
    batch::Data.CanonicalBatch,
)
    trainer.active && error("nested canonical train_update! rejected")
    executor = trainer.session.executor
    batch.input.state_batch == executor.batch.input.state_batch ||
        throw(DimensionMismatch("state batch changed across training team"))
    batch.input.width == executor.batch.input.width ||
        throw(DimensionMismatch("candidate width changed across training team"))
    executor.batch = batch
    trainer.active = true
    try
        return Barrierless.train_update!(trainer.session)
    finally
        trainer.active = false
    end
end

"""Run a caller-owned training loop on one persistent candidate-level team."""
function with_training_team(
    body::F,
    model::Graph.CanonicalModel,
    batch::Data.CanonicalBatch,
    config::CanonicalTrainingConfig=CanonicalTrainingConfig();
    workers::Integer=Base.Threads.nthreads(:default),
    queue_capacity::Integer=64,
    candidate_chunk_size::Integer=4,
    binding_mode::Symbol=:none,
    log_config::Bool=true,
    log_io::IO=stdout,
) where {F}
    worker_count = Int(workers)
    chunk = Int(candidate_chunk_size)
    adapter = DendriticTrainingAdapter(
        model, batch, config; candidate_chunk_size=chunk,
    )
    executor = Barrierless.CanonicalExecutor(
        adapter,
        batch;
        worker_capacity=worker_count,
        candidate_chunk_size=chunk,
    )
    log_config && println(log_io, training_config_summary(config))
    return Barrierless.run_executor_team!(
        executor;
        workers=worker_count,
        queue_capacity,
        binding_mode,
    ) do session
        body(CanonicalTrainer(session))
    end
end

end # module CanonicalTraining
