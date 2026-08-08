module CanonicalTraining

using SHA
using ..CanonicalBarrierless
using ..CanonicalDendriticGraph
using ..CanonicalExperimentData
using ..CanonicalListNet
using ..CanonicalLocalLearning
using ..CanonicalOptimizer
using ..CanonicalPlasticity
using ..DendriticOutputPopulation

const Barrierless = CanonicalBarrierless
const Data = CanonicalExperimentData
const Graph = CanonicalDendriticGraph
const ListNet = CanonicalListNet
const Local = CanonicalLocalLearning
const Optimizer = CanonicalOptimizer
const Plasticity = CanonicalPlasticity
const Output = DendriticOutputPopulation

export AuxiliaryLossConfig,
       OptimizerGroupConfig,
       CanonicalTrainingConfig,
       CanonicalLossResult,
       MechanismActivation,
       MechanismCounters,
       MechanismCounterState,
       TrainingUpdateResult,
       DendriticTrainingAdapter,
       CanonicalTrainer,
       mechanism_counts,
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

const _MECHANISM_KEYS = (
    :decolle_signal_nonzero,
    :subthreshold_updates,
    :nonspiking_updates,
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
    homeostasis_events::Bool
    synaptic_scaling_events::Bool
    utility_updates::Bool
    rewires::Bool
end

MechanismActivation() = MechanismActivation(
    false, false, false, false, false, false, false,
)

"""Immutable per-update mechanism telemetry."""
struct MechanismCounters
    decolle_signal_nonzero::Int64
    subthreshold_updates::Int64
    nonspiking_updates::Int64
    homeostasis_events::Int64
    synaptic_scaling_events::Int64
    utility_updates::Int64
    rewires::Int64

    function MechanismCounters(values::Vararg{Integer,7})
        all(value -> value >= 0, values) || throw(ArgumentError(
            "mechanism counters must be nonnegative",
        ))
        return new(map(Int64, values)...)
    end
end

MechanismCounters() = MechanismCounters(0, 0, 0, 0, 0, 0, 0)

"""Mutable worker-local accumulator; no hot-path atomics are required."""
mutable struct MechanismCounterState
    decolle_signal_nonzero::Int64
    subthreshold_updates::Int64
    nonspiking_updates::Int64
    homeostasis_events::Int64
    synaptic_scaling_events::Int64
    utility_updates::Int64
    rewires::Int64
end

MechanismCounterState() = MechanismCounterState(0, 0, 0, 0, 0, 0, 0)

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
    optimizer::Optimizer.OptimizerStepStats
    update::UInt64
end

mutable struct DendriticTrainingWorker{G,L}
    graph::G
    local_learner::L
    delta22::Vector{Float32}
    counters::MechanismCounterState
end

mutable struct DendriticTrainingAdapter{R,A,S,L,PB,PS} <:
               Barrierless.AbstractCanonicalGraphAdapter
    model::Graph.CanonicalModel
    config::CanonicalTrainingConfig
    local_signals::S
    common_states::Vector{Graph.ModelState}
    common_worker::Graph.ModelWorker
    common_local::L
    common_signature::Vector{Graph.TrajectorySignature}
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
    plasticity_batch::PB
    plasticity_state::PS
    event_destination::Vector{UInt16}
    zero_based_state_offsets::Vector{Int32}
    mechanism_slots::Vector{MechanismCounters}
    mechanisms::MechanismCounters
    expected::MechanismActivation
    clocks::Local.LearningClockState
    due::Local.DuePlasticityClocks
    registry::R
    optimizer_state::A
    candidate_chunk_size::Int
    slot_capacity::Int
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
    iszero(config.local_learning.hard_event_multiplier) || throw(ArgumentError(
        "hard-event causal credit is not connected; hard_event_multiplier must be zero",
    ))
    config.local_learning.plasticity.structure_enabled && throw(ArgumentError(
        "canonical fixed-spine training requires structure_enabled=false",
    ))
    slot_capacity = cld(capacity, chunk)
    local_signals = Graph.initialize_local_signal_maps(
        model, config.local_learning,
    )
    common_states = [Graph.initialize_state(model) for _ in 1:states]
    common_worker = Graph.initialize_worker(model)
    common_local = Graph.initialize_local_learner(model, local_signals)
    reduced_gradient = Graph.initialize_gradient(model)
    gradient_slots = [
        Graph.initialize_gradient(model) for _ in 1:slot_capacity
    ]
    registry = _parameter_registry(model, reduced_gradient, config.groups)
    optimizer_state = Optimizer.AdamWState(registry)
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
        common_worker,
        common_local,
        fill(Graph.TrajectorySignature(), states),
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
        plasticity_batch,
        plasticity_state,
        event_destination,
        zeros(Int32, states + 1),
        fill(MechanismCounters(), slot_capacity),
        MechanismCounters(),
        MechanismActivation(),
        Local.LearningClockState(),
        Local.DuePlasticityClocks(false, false, false, false),
        registry,
        optimizer_state,
        chunk,
        slot_capacity,
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
    learner = Graph.initialize_local_learner(adapter.model, adapter.local_signals)
    return DendriticTrainingWorker(
        graph,
        learner,
        zeros(Float32, OUTPUT_DIM),
        MechanismCounterState(),
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

function Barrierless.prepare_batch!(
    adapter::DendriticTrainingAdapter,
    batch::Data.CanonicalBatch,
)
    _validate_batch_shape!(adapter, batch)
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
    Optimizer.clear_gradients!(adapter.registry)
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
        Graph.prepare_state_common!(
            adapter.model,
            adapter.common_states[state],
            adapter.common_worker,
            Data.state_input(input, state),
        )
        adapter.state_value[state] = adapter.common_states[state].state_value
        adapter.common_signature[state] =
            adapter.common_states[state].common_signature
    end
    ordinal == valid + 1 || error("candidate compact order overflow")
    required_slots = cld(valid, adapter.candidate_chunk_size)
    required_slots <= adapter.slot_capacity || error(
        "microbatch slot storage is smaller than the scheduler partition",
    )
    return adapter
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
    return MechanismActivation(
        analog,
        analog,
        analog,
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
    )
    _accumulate_local_report!(worker.counters, report)
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

function Barrierless.reduce_worker!(
    adapter::DendriticTrainingAdapter,
    worker::DendriticTrainingWorker,
    ::Data.CanonicalBatch,
    slot::Int,
    ::Int,
    ::Int,
)
    1 <= slot <= adapter.slot_capacity || throw(BoundsError(
        adapter.gradient_slots, slot,
    ))
    destination = adapter.gradient_slots[slot]
    Graph.clear_gradient!(destination)
    Graph.accumulate_gradient!(destination, worker.graph.gradient)
    adapter.mechanism_slots[slot] = _snapshot(worker.counters)
    return nothing
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
    microbatch_count <= adapter.slot_capacity || throw(BoundsError(
        adapter.gradient_slots, microbatch_count,
    ))
    Graph.clear_gradient!(adapter.reduced_gradient)
    mechanisms = MechanismCounters()
    @inbounds for slot in 1:microbatch_count
        Graph.accumulate_gradient!(
            adapter.reduced_gradient, adapter.gradient_slots[slot],
        )
        mechanisms = _add(mechanisms, adapter.mechanism_slots[slot])
    end

    # Candidate COW worlds stop at `initial_core`.  The state-common chronology
    # is regenerated and credited exactly once per state, only after every
    # candidate replay has joined and in fixed state order.
    @inbounds for state in 1:batch.input.state_batch
        Graph.reset_candidate_set!(adapter.common_worker)
        Graph.clear_gradient!(adapter.common_worker)
        Graph.begin_local_microbatch!(adapter.common_local)
        report = Graph.local_replay_state_common!(
            adapter.model,
            adapter.common_states[state],
            adapter.common_worker,
            adapter.common_local,
            Data.state_input(batch.input, state),
            @view(adapter.state_delta22[:, state]),
            adapter.state_value_delta[state],
            adapter.due;
            expected_signature=adapter.common_signature[state],
        )
        report.signature == adapter.common_signature[state] || error(
            "state-common replay signature changed after ListNet boundary",
        )
        Graph.accumulate_gradient!(
            adapter.reduced_gradient, adapter.common_worker.gradient,
        )
        common_counts = MechanismCounterState()
        _accumulate_local_report!(common_counts, report)
        mechanisms = _add(mechanisms, _snapshot(common_counts))
        observation = Graph.local_plasticity_observation(adapter.common_local)
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
    end
    adapter.mechanisms = mechanisms
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

@inline function _optimizer_due_mask(adapter::DendriticTrainingAdapter)
    local_config = adapter.config.local_learning
    analog = adapter.due.analog && local_config.analog_multiplier > 0.0f0
    hard = adapter.due.hard_event &&
        local_config.hard_event_multiplier > 0.0f0
    return (analog, analog, analog || hard, true, true)
end

@inline function _assert_all_finite(array, label::AbstractString)
    @inbounds for value in array
        isfinite(value) || throw(DomainError(
            value, "non-finite value in $label",
        ))
    end
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
    @inbounds for index in eachindex(registry.groups)
        group = registry.groups[index]
        moments = state.moments[index]
        _assert_all_finite(group.parameter, "$(group.name) parameters")
        _assert_all_finite(moments.first, "$(group.name) first moments")
        _assert_all_finite(moments.second, "$(group.name) second moments")
        group.multiplier > 0.0f0 &&
            _assert_all_finite(group.gradient, "$(group.name) gradients")
        due_mask[index] && group.multiplier > 0.0f0 &&
            state.group_steps[index] == typemax(UInt64) && throw(
                OverflowError("optimizer group clock overflow for $(group.name)"),
            )
    end
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
    optimizer_stats = Optimizer.apply_optimizer_boundary!(
        adapter.optimizer_state,
        adapter.registry,
        adapter.config.optimizer;
        due_mask,
    )
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
            0, 0, 0, homeostasis_events, scaling_events, 0, 0,
        ),
    )
    _assert_mechanisms!(
        adapter.expected, adapter.mechanisms; include_slow=true,
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
