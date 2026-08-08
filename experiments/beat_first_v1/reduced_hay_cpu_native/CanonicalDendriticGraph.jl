module CanonicalDendriticGraph

using Random

using ..ActiveApicalCell
using ..CanonicalTetrisInput
using ..DendriticAxonPacket
using ..OrderedMultiscaleTopology
using ..DendriticOutputPopulation
using ..CanonicalEventArena
using ..CanonicalSpatialDrive
using ..CanonicalLocalLearning

const Cell = ActiveApicalCell
const Input = CanonicalTetrisInput
const Axon = DendriticAxonPacket
const Topology = OrderedMultiscaleTopology
const Output = DendriticOutputPopulation
const Events = CanonicalEventArena
const Spatial = CanonicalSpatialDrive
const Local = CanonicalLocalLearning

export CORE_NODE_COUNT,
       TOTAL_NODE_COUNT,
       MAX_COMMON_HARD_EVENT_SEEDS_PER_CANDIDATE,
       GraphConfig,
       ModelParameters,
       ModelCache,
       ModelState,
       ModelGradient,
       ModelHardEventGradient,
       ModelHardEventSeedRecord,
       ModelHardEventDelta,
       ModelWorker,
       ModelForwardStats,
       TrajectorySignature,
       TransitionPhase,
       AnalogDepositKind,
       BINARY_PACKET_DEPOSIT,
       SEMANTIC_PACKET_DEPOSIT,
       COMMON_SOURCE_RECORD,
       is_common_source_record,
       AnalogDepositRecord,
       EventDeliveryRecord,
       OutputEvidenceRecord,
       RecordedCountManifest,
       ReplayProvenance,
       ModelLocalSignalMaps,
       ModelLocalLearner,
       LocalPlasticityObservation,
       LocalReplayCounters,
       LocalReplayReport,
       COMMON_BEFORE,
       CANDIDATE_AFTER,
       MANDATORY_DAG,
       EVENT_WAVE,
       OUTPUT_POPULATION,
       TransitionTape,
       initialize_model,
       initialize_state,
       initialize_worker,
       initialize_local_signal_maps,
       initialize_local_learner,
       begin_local_microbatch!,
       hard_event_gradient,
       hard_event_delivery_gradient,
       clear_hard_event_gradient!,
       hard_gradient_components,
       initialize_hard_event_delta,
       clear_hard_event_delta!,
       publish_hard_event_delta!,
       hard_event_seed_count,
       hard_event_seed_record,
       hard_event_delta_sealed,
       hard_event_delta_candidate_range,
       accumulate_common_hard_event_seeds!,
       load_common_hard_event_seeds!,
       common_hard_event_seed_loaded,
       common_hard_event_seed_consumed,
       common_hard_event_seed_poisoned,
       local_plasticity_observation,
       local_replay_candidate!,
       local_replay_state_common!,
       reset_candidate_set!,
       refresh_state_initial!,
       sync_state_common!,
       initialize_gradient,
       refresh_cache!,
       clear_gradient!,
       accumulate_gradient!,
       parameter_components,
       gradient_components,
       stored_parameter_count,
       prepare_state_common!,
       forward_candidate!,
       assemble_candidate_set!,
       conditional_reverse_candidate!,
       replay_state_common!,
       conditional_reverse_state_value!,
       latest_outputs,
       latest_candidate_count,
       candidate_signature,
       candidate_provenance,
       analog_deposit_count,
       analog_deposit_record,
       analog_deposit_packet_lane,
       record_analog_deposit_range,
       event_delivery_record_count,
       event_delivery_record,
       record_event_delivery_head,
       next_event_delivery_record,
       output_evidence_record_count,
       output_evidence_record,
       output_evidence_packet_lane,
       recorded_count_manifest,
       provenance_parameter_digest,
       provenance_input_digest,
       provenance_signature,
       provenance_sealed,
       candidate_state,
       candidate_packet,
       motif_state,
       motif_packet,
       evidence_state,
       evidence_packet,
       DynamicEventContactDescriptor,
       EventParameterKind,
       STATIC_EVENT_CONTACT,
       DYNAMIC_EVENT_CONTACT,
       SHARED_EVENT_KIND_GAIN,
       EventParameterDescriptor,
       event_parameter_count,
       event_parameter_descriptor,
       dynamic_event_contact_count,
       dynamic_event_pair_index,
       dynamic_event_pair_descriptor,
       dynamic_event_parameter_index,
       event_kind_parameter_index,
       wave_one_event_mask

const TOTAL_NODE_COUNT = Topology.NODE_COUNT
const CORE_NODE_COUNT = TOTAL_NODE_COUNT - Topology.OUTPUT_COUNT
const SEMANTIC_CLASS_COUNT = 2 # motif and evidence
const EVENT_TRIGGER_MASK = UInt8((1 << Axon.EVENT_DIM) - 1)
# Only motif families1:6 have live spine afferents. Families1:4 expose two
# live branches per slot and families5:6 expose six. Families7:8 are external
# candidate/context descriptors and therefore cannot emit cell events.
const DYNAMIC_MOTIF_EVENT_COUNT =
    4 * Topology.MOTIF_SLOTS_PER_FAMILY * 2 +
    2 * Topology.MOTIF_SLOTS_PER_FAMILY * 6
const DYNAMIC_MOTIF_EVENT_EDGE_CAPACITY =
    DYNAMIC_MOTIF_EVENT_COUNT * Axon.EVENT_DIM
const EVENT_KIND_PARAMETER_COUNT = Axon.EVENT_DIM
const MAX_COMMON_HARD_EVENT_SEEDS_PER_CANDIDATE =
    DYNAMIC_MOTIF_EVENT_EDGE_CAPACITY # 80 dynamic contacts × five typed lanes

"""Checkpoint-stable anatomical identity of one live dynamic event contact."""
struct DynamicEventContactDescriptor
    family::UInt8
    slot::UInt8
    branch::UInt8
end

@enum EventParameterKind::UInt8 begin
    STATIC_EVENT_CONTACT = 0x01
    DYNAMIC_EVENT_CONTACT = 0x02
    SHARED_EVENT_KIND_GAIN = 0x03
end

"""
Allocation-free, checkpoint-facing identity for one `event_raw` coordinate.

For static contacts, `source`, `destination`, and `channel` are populated.  For
dynamic contacts, the candidate-dependent source is zero and the stable
`family/slot/branch` anatomy plus motif destination are populated.  Shared
event-kind gains populate only `lane`.
"""
struct EventParameterDescriptor
    kind::EventParameterKind
    source::UInt16
    destination::UInt16
    channel::UInt8
    family::UInt8
    slot::UInt8
    branch::UInt8
    lane::UInt8
end

@assert TOTAL_NODE_COUNT == 1_458
@assert CORE_NODE_COUNT == 1_436
@assert Topology.OUTPUT_COUNT == Output.OUTPUT_CELLS == Output.OUTPUT_DIM
@assert Axon.PACKET_DIM == 12
@assert Cell.STATE_DIM == 48

struct GraphConfig
    max_candidates::Int
    max_event_waves::Int
    tape_capacity::Int
    event_overflow::Symbol

    function GraphConfig(
        max_candidates::Integer=128,
        max_event_waves::Integer=Events.CANONICAL_MAX_WAVES,
        tape_capacity::Integer=12_288,
        event_overflow::Symbol=:error,
    )
        max_candidates > 0 || throw(ArgumentError(
            "max_candidates must be positive",
        ))
        0 <= max_event_waves <= Events.CANONICAL_MAX_WAVES || throw(
            ArgumentError(
                "max_event_waves must be in 0:$(Events.CANONICAL_MAX_WAVES)",
            ),
        )
        required_tape_capacity = CORE_NODE_COUNT * (1 + Int(max_event_waves))
        tape_capacity >= required_tape_capacity || throw(ArgumentError(
            "tape capacity must hold the full mandatory traversal and one " *
            "unique destination transition per event wave " *
            "(minimum $required_tape_capacity)",
        ))
        event_overflow in (:error, :fallback) || throw(ArgumentError(
            "event_overflow must be :error or :fallback",
        ))
        return new(
            Int(max_candidates),
            Int(max_event_waves),
            Int(tape_capacity),
            event_overflow,
        )
    end
end

GraphConfig(; max_candidates::Integer=128,
              max_event_waves::Integer=Events.CANONICAL_MAX_WAVES,
              tape_capacity::Integer=12_288,
              event_overflow::Symbol=:error) = GraphConfig(
    max_candidates,
    max_event_waves,
    tape_capacity,
    event_overflow,
)

"""
The only trainable owners of the canonical graph.

`core_cell_raw` covers every non-output Reduced-Hay cell.  Information-spine
edges have no trainable bottleneck.  Motif/evidence semantic compression uses
one receptor-diagonal projection per receiver class and ordered branch role.
The output population owns its complete private-cell parameters.
"""
struct ModelParameters
    core_cell_raw::Matrix{Float32}
    semantic_projection_raw::Array{Float32,4}
    event_raw::Vector{Float32}
    output::Output.OutputPopulationParameters{Float32}
end

parameter_components(parameters::ModelParameters) = (
    core_cell_raw=parameters.core_cell_raw,
    semantic_projection_raw=parameters.semantic_projection_raw,
    event_raw=parameters.event_raw,
    output_cell_raw=parameters.output.cell_raw,
    output_projection_raw=parameters.output.projection_raw,
)

struct ModelGradient
    core_cell_raw::Matrix{Float32}
    semantic_projection_raw::Array{Float32,4}
    event_raw::Vector{Float32}
    output::Output.OutputPopulationGradient{Float32}
end

"""
Gradient lane owned exclusively by hard-event source control.

The ordinary analog clock owns its receiver contact/kind derivatives in
`ModelGradient.event_raw`. Independently, the hard clock owns both the direct
continuation derivative of every evaluated delivery and the returned scalar
source-control advantage. The source cell-local surrogate may differentiate
that transition's own sealed semantic/event input parameters, but never
propagates a state or packet cotangent into another cell. The three recurrent
parameter groups remain separate from the analog gradient while sharing no
dense eligibility matrix.
"""
struct ModelHardEventGradient
    core_cell_raw::Matrix{Float32}
    semantic_projection_raw::Array{Float32,4}
    event_raw::Vector{Float32}
end

function clear_hard_event_gradient!(gradient::ModelHardEventGradient)
    fill!(gradient.core_cell_raw, 0.0f0)
    fill!(gradient.semantic_projection_raw, 0.0f0)
    fill!(gradient.event_raw, 0.0f0)
    return gradient
end

hard_gradient_components(gradient::ModelHardEventGradient) = (
    core_cell_raw=gradient.core_cell_raw,
    semantic_projection_raw=gradient.semantic_projection_raw,
    event_raw=gradient.event_raw,
)

"""One actual candidate delivery whose source is finalized state-common."""
struct ModelHardEventSeedRecord
    state_id::UInt16
    candidate_ordinal::Int32
    delivery_ordinal::UInt32
    source_node::UInt16
    lane::UInt8
    advantage::Float32
end

"""
Fixed-capacity publication slot for one deterministic candidate replay range.

The recurrent hard gradient is reduced independently from the ordinary analog
gradient. `seed_*` contains one record for every actual candidate delivery
whose source record is `COMMON_SOURCE_RECORD`, including a zero advantage.
Records are published in logical-candidate then physical-delivery order.
`sealed` is written last, after the gradient, ledger, model digest, payload
digests, and logical range are complete. Both compact advantages and all three
hard-gradient groups are covered by the sealed payload digests.
"""
mutable struct ModelHardEventDelta
    gradient::ModelHardEventGradient
    seed_state_id::Vector{UInt16}
    seed_candidate_ordinal::Vector{Int32}
    seed_delivery_ordinal::Vector{UInt32}
    seed_source_node::Vector{UInt16}
    seed_lane::Vector{UInt8}
    seed_advantage::Vector{Float32}
    seed_count::Int
    parameter_digest::UInt64
    logical_first::Int32
    logical_last::Int32
    seed_identity_digest::UInt64
    hard_gradient_digest::UInt64
    sealed::Bool
    reduced::Bool
end

@inline hard_event_seed_count(delta::ModelHardEventDelta) = delta.seed_count
@inline hard_event_delta_sealed(delta::ModelHardEventDelta) = delta.sealed
@inline hard_event_delta_candidate_range(delta::ModelHardEventDelta) = (
    Int(delta.logical_first),
    Int(delta.logical_last),
)
hard_event_gradient(delta::ModelHardEventDelta) = delta.gradient

@inline function hard_event_seed_record(
    delta::ModelHardEventDelta,
    index::Integer,
)
    slot = Int(index)
    1 <= slot <= delta.seed_count || throw(BoundsError(delta, slot))
    return @inbounds ModelHardEventSeedRecord(
        delta.seed_state_id[slot],
        delta.seed_candidate_ordinal[slot],
        delta.seed_delivery_ordinal[slot],
        delta.seed_source_node[slot],
        delta.seed_lane[slot],
        delta.seed_advantage[slot],
    )
end

gradient_components(gradient::ModelGradient) = (
    core_cell_raw=gradient.core_cell_raw,
    semantic_projection_raw=gradient.semantic_projection_raw,
    event_raw=gradient.event_raw,
    output_cell_raw=gradient.output.cell_raw,
    output_projection_raw=gradient.output.projection_raw,
)

mutable struct ModelCache
    core_cell::Vector{Cell.CellParameterCache{Float32}}
    core_derivative::Vector{Cell.CellParameterDerivativeCache{Float32}}
    semantic_projection::Array{Float32,4}
    semantic_projection_derivative::Array{Float32,4}
    event_weight::Vector{Float32}
    event_graph::Events.SourceMajorAdjacency{Float32}
    output::Output.OutputPopulationCache{Float32}
    revision::UInt64
    parameter_digest::UInt64
end

struct CanonicalModel
    config::GraphConfig
    topology::Topology.OrderedTopology
    parameters::ModelParameters
    cache::ModelCache
end

@enum TransitionPhase::UInt8 begin
    COMMON_BEFORE = 0x01
    CANDIDATE_AFTER = 0x02
    MANDATORY_DAG = 0x03
    EVENT_WAVE = 0x04
    OUTPUT_POPULATION = 0x05
end

"""Exact hard-trajectory identity used by replay and finite differences."""
struct TrajectorySignature
    soma_hash::UInt64
    plateau_hash::UInt64
    frontier_hash::UInt64
    delivery_hash::UInt64
    delivery_count::Int
    transition_count::Int
    event_waves::Int
    terminated_empty::Bool
    hit_wave_limit::Bool
end

TrajectorySignature() = TrajectorySignature(
    UInt64(0), UInt64(0), UInt64(0), UInt64(0), 0, 0, 0, true, false,
)

"""Model-shared, seed-fixed two-block local feedback maps."""
struct ModelLocalSignalMaps
    config::Local.LocalLearningConfig
    continuous::Vector{Local.FixedLocalSignalMap{Float32}}
    packet::Vector{Local.FixedLocalSignalMap{Float32}}
end

mutable struct LocalReplayCounters
    candidate_replays::Int
    common_replays::Int
    visited_transitions::Int
    conditional_pullbacks::Int
    signal_nonzero::Int
    nonspiking_transitions::Int
    semantic_parameter_updates::Int
    event_receiver_updates::Int
    utility_updates::Int
    output_replays::Int
    event_control_deliveries::Int
    event_control_source_transitions::Int
    event_control_soma_sources::Int
    event_control_plateau_sources::Int
    event_control_common_seeds::Int
    event_control_semantic_updates::Int
    event_control_event_parameter_updates::Int
end

LocalReplayCounters() = LocalReplayCounters(
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)

"""
Fixed publication payload for the most recently replayed logical world.

Cell observations are teacher-free trajectory statistics. Event-contact
utility is accumulated per physical delivery *before* shared-parameter
contributions can cancel. The vectors cover the 2,120 anatomical contacts;
the five trainable shared kind gains are explicitly non-rewirable and absent.
"""
mutable struct LocalPlasticityObservation
    spike_count::Vector{UInt8}
    visit_count::Vector{UInt8}
    activity_sum::Vector{Float32}
    incoming_conductance_sum::Vector{Float32}
    task_utility_sum::Vector{Float32}
    contact_activity_sum::Vector{Float32}
end

struct LocalReplayReport
    signature::TrajectorySignature
    visited_transitions::Int
    conditional_pullbacks::Int
    signal_nonzero::Int
    nonspiking_transitions::Int
    semantic_parameter_updates::Int
    event_receiver_updates::Int
    utility_updates::Int
    output_replays::Int
    event_control_deliveries::Int
    event_control_source_transitions::Int
    event_control_soma_sources::Int
    event_control_plateau_sources::Int
    event_control_common_seeds::Int
    event_control_semantic_updates::Int
    event_control_event_parameter_updates::Int
end

mutable struct ModelLocalLearner
    signals::ModelLocalSignalMaps
    arena::Local.ContractedAdjointArena{Float32}
    scratch::Local.ContractedAdjointScratch{Float32}
    continuous_signal::Vector{Float32}
    packet_signal::Vector{Float32}
    fused_arena::Local.FusedContractedAdjointArena{Float32}
    fused_scratch::Local.FusedContractedAdjointScratch{Float32}
    continuous_signal_by_node::Matrix{Float32}
    packet_signal_by_node::Matrix{Float32}
    receiver_input_cotangent::Matrix{Float32}
    logical_record::Vector{Bool}
    source_event_advantage::Matrix{Float32}
    event_delivery_advantage::Vector{Float32}
    source_delivery_next::Vector{Int32}
    source_delivery_head_by_record::Vector{Int32}
    source_delivery_tail_by_record::Vector{Int32}
    event_delivery_ready::Vector{Bool}
    common_source_event_advantage::Matrix{Float32}
    common_source_seed_loaded::Bool
    common_source_seed_consumed::Bool
    common_source_seed_in_progress::Bool
    common_source_seed_state_id::UInt16
    hard_seed_state_id::Vector{UInt16}
    hard_seed_candidate_ordinal::Vector{Int32}
    hard_seed_delivery_ordinal::Vector{UInt32}
    hard_seed_source_node::Vector{UInt16}
    hard_seed_lane::Vector{UInt8}
    hard_seed_advantage::Vector{Float32}
    hard_seed_count::Int
    hard_expected_seed_count::Int
    hard_seed_identity_digest::UInt64
    first_hard_candidate_ordinal::Int32
    last_hard_candidate_ordinal::Int32
    hard_parameter_digest::UInt64
    hard_gradient::ModelHardEventGradient
    delivery_event_gradient::Vector{Float32}
    plasticity::LocalPlasticityObservation
    counters::LocalReplayCounters
end

mutable struct ModelState
    initial_core::Matrix{Float32}
    common_state::Matrix{Float32}
    common_packet::Matrix{Float32}
    common_input::Matrix{Float32}
    common_seed_mask::Vector{UInt8}
    common_event_mask::Vector{UInt8}
    output_initial::Matrix{Float32}
    state_value::Float32
    state_value_components::Output.OutputComponents{Float32}
    state_value_hard_event::Vector{Float32}
    state_value_tape::Output.OutputPopulationTape{Float32}
    common_signature::TrajectorySignature
    prepared_revision::UInt64
    epoch::UInt64
    fingerprint::UInt64
    ready::Bool
end

"""One fixed replay tape; pass one stores only `TrajectorySignature`."""
mutable struct TransitionTape
    node::Vector{UInt16}
    phase::Vector{UInt8}
    wave::Vector{UInt8}
    event_mask::Vector{UInt8}
    previous_record::Vector{Int32}
    latest_record::Vector{Int32}
    mandatory_record::Vector{Int32}
    previous_state::Matrix{Float32}
    input::Matrix{Float32}
    next_state::Matrix{Float32}
    packet::Matrix{Float32}
    count::Int
    signature::TrajectorySignature
end

@enum AnalogDepositKind::UInt8 begin
    BINARY_PACKET_DEPOSIT = 0x01
    SEMANTIC_PACKET_DEPOSIT = 0x02
end

"""
Named event-provenance identity for a finalized state-common source.

The sentinel is valid only for a candidate wave-one dynamic-overlay delivery
from a source outside the logical candidate closure. Event sources are always
real core nodes, so `(source_node > 0, source_record == COMMON_SOURCE_RECORD)`
is distinct from an external analog/output constant `(0, 0)`.
"""
const COMMON_SOURCE_RECORD = Int32(0)

@inline is_common_source_record(
    source_node::Integer,
    source_record::Integer,
) = source_node > 0 && source_record == COMMON_SOURCE_RECORD

"""One immutable view of an authoritative mandatory-analog deposit."""
struct AnalogDepositRecord
    kind::AnalogDepositKind
    destination_record::Int32
    source_node::UInt16
    source_record::Int32
    branch::UInt8
    semantic_role::UInt8
    semantic_class::UInt8
    ordinal::UInt32
end

"""One actually delivered hard-event lane after polarity/channel resolution."""
struct EventDeliveryRecord
    destination_record::Int32
    source_node::UInt16
    source_record::Int32
    source_mask::UInt8
    lane::UInt8
    destination_branch::UInt8
    polarity::UInt8
    resolved_channel::UInt8
    contact_parameter::UInt16
    kind_parameter::UInt16
    scale::Float32
    wave::UInt8
    ordinal::UInt32
end

@inline is_common_source_record(record::EventDeliveryRecord) =
    is_common_source_record(record.source_node, record.source_record)

"""One exact packet binding consumed by a physical output cell."""
struct OutputEvidenceRecord
    source_node::UInt16
    source_record::Int32
    output_cell::UInt8
    evidence_rank::UInt8
    ordinal::UInt16
end

struct RecordedCountManifest
    transitions::Int
    analog_deposits::Int
    event_deliveries::Int
    output_bindings::Int
end

"""
Fixed-capacity chronological provenance for one reusable replay world.

Packet primals are copied into the analog/output ledgers. Event records keep
stable raw-parameter indices and the resolved typed channel, so replay never
has to infer polarity, incidence rank, or parameter order from a later graph
snapshot. `COMMON_SOURCE_RECORD` denotes a candidate wave-one dynamic-overlay
read from the finalized state-common snapshot; positive event identities index
the accompanying `TransitionTape`. Analog/output `(source_node, source_record)
== (0, 0)` remains the separate external-constant identity.
"""
mutable struct ReplayProvenance
    analog_kind::Vector{UInt8}
    analog_destination_record::Vector{Int32}
    analog_source_node::Vector{UInt16}
    analog_source_record::Vector{Int32}
    analog_branch::Vector{UInt8}
    analog_semantic_role::Vector{UInt8}
    analog_semantic_class::Vector{UInt8}
    analog_ordinal::Vector{UInt32}
    analog_packet::Matrix{Float32}
    analog_first_by_record::Vector{Int32}
    analog_count_by_record::Vector{UInt8}
    analog_count::Int
    event_destination_record::Vector{Int32}
    event_source_node::Vector{UInt16}
    event_source_record::Vector{Int32}
    event_source_mask::Vector{UInt8}
    event_lane::Vector{UInt8}
    event_destination_branch::Vector{UInt8}
    event_polarity::Vector{UInt8}
    event_resolved_channel::Vector{UInt8}
    event_contact_parameter::Vector{UInt16}
    event_kind_parameter::Vector{UInt16}
    event_scale::Vector{Float32}
    event_wave::Vector{UInt8}
    event_ordinal::Vector{UInt32}
    event_next::Vector{Int32}
    event_head_by_node::Vector{Int32}
    event_tail_by_node::Vector{Int32}
    event_head_by_record::Vector{Int32}
    event_count::Int
    active_event_wave::Int
    output_source_node::Vector{UInt16}
    output_source_record::Vector{Int32}
    output_cell::Vector{UInt8}
    output_rank::Vector{UInt8}
    output_ordinal::Vector{UInt16}
    output_packet::Matrix{Float32}
    output_count::Int
    parameter_digest::UInt64
    input_digest::UInt64
    signature::TrajectorySignature
    sealed::Bool
end

function ReplayProvenance(
    tape_capacity::Integer,
    analog_capacity::Integer,
    event_capacity::Integer,
    output_capacity::Integer,
)
    tape_n = Int(tape_capacity)
    analog_n = Int(analog_capacity)
    event_n = Int(event_capacity)
    output_n = Int(output_capacity)
    minimum((tape_n, analog_n, event_n, output_n)) >= 1 || throw(
        ArgumentError("provenance capacities must be positive"),
    )
    event_n <= typemax(UInt32) || throw(ArgumentError(
        "event provenance capacity exceeds UInt32 ordinal identity",
    ))
    return ReplayProvenance(
        zeros(UInt8, analog_n), zeros(Int32, analog_n),
        zeros(UInt16, analog_n), zeros(Int32, analog_n),
        zeros(UInt8, analog_n), zeros(UInt8, analog_n),
        zeros(UInt8, analog_n), zeros(UInt32, analog_n),
        Matrix{Float32}(undef, Axon.PACKET_DIM, analog_n),
        zeros(Int32, tape_n), zeros(UInt8, tape_n), 0,
        zeros(Int32, event_n), zeros(UInt16, event_n),
        zeros(Int32, event_n), zeros(UInt8, event_n),
        zeros(UInt8, event_n), zeros(UInt8, event_n),
        zeros(UInt8, event_n), zeros(UInt8, event_n),
        zeros(UInt16, event_n), zeros(UInt16, event_n),
        zeros(Float32, event_n), zeros(UInt8, event_n),
        zeros(UInt32, event_n), zeros(Int32, event_n),
        zeros(Int32, CORE_NODE_COUNT), zeros(Int32, CORE_NODE_COUNT),
        zeros(Int32, tape_n), 0, 0,
        zeros(UInt16, output_n), zeros(Int32, output_n),
        zeros(UInt8, output_n), zeros(UInt8, output_n),
        zeros(UInt16, output_n),
        Matrix{Float32}(undef, Axon.PACKET_DIM, output_n), 0,
        UInt64(0), UInt64(0), TrajectorySignature(), false,
    )
end

@inline analog_deposit_count(tape::ReplayProvenance) = tape.analog_count
@inline event_delivery_record_count(tape::ReplayProvenance) = tape.event_count
@inline output_evidence_record_count(tape::ReplayProvenance) = tape.output_count
@inline provenance_parameter_digest(tape::ReplayProvenance) = tape.parameter_digest
@inline provenance_input_digest(tape::ReplayProvenance) = tape.input_digest
@inline provenance_signature(tape::ReplayProvenance) = tape.signature
@inline provenance_sealed(tape::ReplayProvenance) = tape.sealed

@inline function analog_deposit_record(tape::ReplayProvenance, index::Integer)
    slot = Int(index)
    1 <= slot <= tape.analog_count || throw(BoundsError(tape, slot))
    return @inbounds AnalogDepositRecord(
        AnalogDepositKind(tape.analog_kind[slot]),
        tape.analog_destination_record[slot], tape.analog_source_node[slot],
        tape.analog_source_record[slot], tape.analog_branch[slot],
        tape.analog_semantic_role[slot], tape.analog_semantic_class[slot],
        tape.analog_ordinal[slot],
    )
end

@inline function analog_deposit_packet_lane(
    tape::ReplayProvenance,
    index::Integer,
    lane::Integer,
)
    slot = Int(index); packet_lane = Int(lane)
    1 <= slot <= tape.analog_count || throw(BoundsError(tape, slot))
    1 <= packet_lane <= Axon.PACKET_DIM || throw(BoundsError(1:Axon.PACKET_DIM, packet_lane))
    return @inbounds tape.analog_packet[packet_lane, slot]
end

@inline function record_analog_deposit_range(
    tape::ReplayProvenance,
    record::Integer,
)
    slot = Int(record)
    1 <= slot <= length(tape.analog_first_by_record) || throw(BoundsError(tape, slot))
    return (
        @inbounds(Int(tape.analog_first_by_record[slot])),
        @inbounds(Int(tape.analog_count_by_record[slot])),
    )
end

@inline function event_delivery_record(tape::ReplayProvenance, index::Integer)
    slot = Int(index)
    1 <= slot <= tape.event_count || throw(BoundsError(tape, slot))
    return @inbounds EventDeliveryRecord(
        tape.event_destination_record[slot], tape.event_source_node[slot],
        tape.event_source_record[slot], tape.event_source_mask[slot],
        tape.event_lane[slot], tape.event_destination_branch[slot],
        tape.event_polarity[slot], tape.event_resolved_channel[slot],
        tape.event_contact_parameter[slot], tape.event_kind_parameter[slot],
        tape.event_scale[slot], tape.event_wave[slot],
        tape.event_ordinal[slot],
    )
end

@inline function record_event_delivery_head(
    tape::ReplayProvenance,
    record::Integer,
)
    slot = Int(record)
    1 <= slot <= length(tape.event_head_by_record) || throw(BoundsError(tape, slot))
    return Int(@inbounds tape.event_head_by_record[slot])
end

@inline function next_event_delivery_record(
    tape::ReplayProvenance,
    delivery::Integer,
)
    slot = Int(delivery)
    1 <= slot <= tape.event_count || throw(BoundsError(tape, slot))
    return Int(@inbounds tape.event_next[slot])
end

@inline function output_evidence_record(tape::ReplayProvenance, index::Integer)
    slot = Int(index)
    1 <= slot <= tape.output_count || throw(BoundsError(tape, slot))
    return @inbounds OutputEvidenceRecord(
        tape.output_source_node[slot], tape.output_source_record[slot],
        tape.output_cell[slot], tape.output_rank[slot],
        tape.output_ordinal[slot],
    )
end

@inline function output_evidence_packet_lane(
    tape::ReplayProvenance,
    index::Integer,
    lane::Integer,
)
    slot = Int(index); packet_lane = Int(lane)
    1 <= slot <= tape.output_count || throw(BoundsError(tape, slot))
    1 <= packet_lane <= Axon.PACKET_DIM || throw(BoundsError(1:Axon.PACKET_DIM, packet_lane))
    return @inbounds tape.output_packet[packet_lane, slot]
end

function TransitionTape(capacity::Integer)
    capacity > 0 || throw(ArgumentError("tape capacity must be positive"))
    physical = Int(capacity)
    physical <= typemax(Int32) || throw(ArgumentError(
        "tape capacity exceeds Int32 record identity",
    ))
    return TransitionTape(
        zeros(UInt16, physical),
        zeros(UInt8, physical),
        zeros(UInt8, physical),
        zeros(UInt8, physical),
        zeros(Int32, physical),
        zeros(Int32, CORE_NODE_COUNT),
        zeros(Int32, CORE_NODE_COUNT),
        Matrix{Float32}(undef, Cell.STATE_DIM, physical),
        Matrix{Float32}(undef, Cell.INPUT_DIM, physical),
        Matrix{Float32}(undef, Cell.STATE_DIM, physical),
        Matrix{Float32}(undef, Axon.PACKET_DIM, physical),
        0,
        TrajectorySignature(),
    )
end

mutable struct ModelForwardStats
    common_transitions::Int
    common_event_transitions::Int
    common_event_waves::Int
    mandatory_transitions::Int
    event_transitions::Int
    output_transitions::Int
    event_waves::Int
    hard_events::Int
    clear_slow_paths::Int
end

ModelForwardStats() = ModelForwardStats(0, 0, 0, 0, 0, 0, 0, 0, 0)

mutable struct ModelWorker
    geometry::Input.CandidateGeometry
    motif_incidence::Topology.CandidateMotifIncidence
    closure::Topology.AffectedClosure
    arena::Events.EventArena{Float32}
    common_replay_state::ModelState
    dynamic_overlay::Events.DynamicSourceMajorOverlay
    packet_by_slot::Matrix{Float32}
    event_by_slot::Matrix{UInt8}
    seeds::Vector{UInt16}
    seed_count::Int
    context::Vector{Float32}
    input_scratch::Vector{Float32}
    previous_scratch::Vector{Float32}
    next_scratch::Vector{Float32}
    packet_scratch::Vector{Float32}
    event_scratch::Vector{UInt8}
    output_evidence::Array{Float32,3}
    output_evidence_count::Vector{UInt8}
    output_hard_event::Vector{Float32}
    output_tape::Output.OutputPopulationTape{Float32}
    output_scratch::Output.OutputPopulationScratch{Float32}
    output_dbase::Matrix{Float32}
    output_devidence::Array{Float32,3}
    components::Vector{Output.OutputComponents{Float32}}
    component_bars::Vector{Output.OutputComponentGradient{Float32}}
    outputs::Matrix{Float32}
    advantages::Vector{Float32}
    signatures::Vector{TrajectorySignature}
    candidate_count::Int
    tape::TransitionTape
    provenance::ReplayProvenance
    gradient::ModelGradient
    core_packet_bar::Matrix{Float32}
    core_state_bar::Matrix{Float32}
    record_packet_bar::Matrix{Float32}
    record_state_bar::Matrix{Float32}
    dstate_scratch::Vector{Float32}
    dinput_scratch::Vector{Float32}
    draw_scratch::Vector{Float32}
    dnext_scratch::Vector{Float32}
    stats::ModelForwardStats
    event_delivery_hash::UInt64
    event_delivery_count::Int
    prepared_state_epoch::UInt64
    prepared_state_fingerprint::UInt64
    prepared_cache_revision::UInt64
end

latest_outputs(worker::ModelWorker) = @view worker.outputs[:, 1:worker.candidate_count]
latest_candidate_count(worker::ModelWorker) = worker.candidate_count
candidate_signature(worker::ModelWorker, candidate::Integer) =
    worker.signatures[Int(candidate)]
candidate_provenance(worker::ModelWorker) = worker.provenance
recorded_count_manifest(worker::ModelWorker) = RecordedCountManifest(
    worker.tape.count,
    worker.provenance.analog_count,
    worker.provenance.event_count,
    worker.provenance.output_count,
)

function reset_candidate_set!(worker::ModelWorker)
    worker.candidate_count = 0
    return worker
end

@inline _softplus(value::Float32) = max(value, 0.0f0) + log1p(exp(-abs(value)))
@inline function _softplus_derivative(value::Float32)
    value >= 0.0f0 && return inv(1.0f0 + exp(-value))
    exponential = exp(value)
    return exponential / (1.0f0 + exponential)
end

@inline function _zero_input!(destination::AbstractVector{Float32})
    @inbounds for index in 1:Cell.INPUT_DIM
        destination[index] = 0.0f0
    end
    return destination
end
@inline _inverse_softplus(value::Float32) = log(expm1(value))

function _initial_parameters(rng::AbstractRNG)
    default_raw = Cell.default_raw_parameters(Float32)
    cell_raw = Matrix{Float32}(undef, Cell.PARAM_DIM, CORE_NODE_COUNT)
    @inbounds for node in 1:CORE_NODE_COUNT, parameter in 1:Cell.PARAM_DIM
        # Tiny deterministic symmetry breaking; no position lookup exists.
        cell_raw[parameter, node] = default_raw[parameter] +
            1.0f-3 * randn(rng, Float32)
    end
    semantic_raw = Array{Float32,4}(
        undef,
        Axon.GROUP_COUNT,
        Cell.INPUT_CHANNELS,
        Topology.SEMANTIC_ROLE_COUNT,
        SEMANTIC_CLASS_COUNT,
    )
    @inbounds for index in eachindex(semantic_raw)
        semantic_raw[index] = _inverse_softplus(0.05f0)
    end
    # One event weight per static core edge; optional events only refine the
    # packet-selected mandatory state and never replace the analog sweep.
    core_event_edges = _core_event_edge_count(Topology.canonical_topology())
    contact_count = core_event_edges + DYNAMIC_MOTIF_EVENT_COUNT
    event_raw = Vector{Float32}(
        undef,
        contact_count + EVENT_KIND_PARAMETER_COUNT,
    )
    fill!(@view(event_raw[1:core_event_edges]), _inverse_softplus(0.02f0))
    fill!(
        @view(event_raw[(core_event_edges + 1):contact_count]),
        _inverse_softplus(0.10f0),
    )
    # Five shared typed gains separate soma and the four plateau groups
    # without multiplying anatomical contact parameters.
    fill!(
        @view(event_raw[(contact_count + 1):end]),
        _inverse_softplus(1.0f0),
    )
    return ModelParameters(
        cell_raw,
        semantic_raw,
        event_raw,
        Output.initialize_parameters(Float32),
    )
end

function _core_event_edge_count(topology::Topology.OrderedTopology)
    count = 0
    @inbounds for edge in 1:Topology.EDGE_COUNT
        Int(Topology.edge_destination(topology, edge)) <= CORE_NODE_COUNT &&
            (count += 1)
    end
    return count
end

@inline function _event_delivery_capacity(model::CanonicalModel)
    return max(
        1,
        (_core_event_edge_count(model.topology) * Axon.EVENT_DIM +
         DYNAMIC_MOTIF_EVENT_EDGE_CAPACITY) *
        max(1, model.config.max_event_waves),
    )
end

function _build_event_graph(
    topology::Topology.OrderedTopology,
    event_weight::AbstractVector{Float32},
)
    edge_count = _core_event_edge_count(topology)
    length(event_weight) >= edge_count || throw(DimensionMismatch(
        "event parameter count differs from static core adjacency",
    ))
    offsets = Vector{UInt32}(undef, CORE_NODE_COUNT + 1)
    destination = UInt16[]
    channel = UInt8[]
    trigger = UInt8[]
    weights = Float32[]
    sizehint!(destination, edge_count)
    sizehint!(channel, edge_count)
    sizehint!(trigger, edge_count)
    sizehint!(weights, edge_count)
    event_index = 0
    offsets[1] = UInt32(1)
    @inbounds for source in 1:CORE_NODE_COUNT
        local_edges = Tuple{UInt16,UInt8,UInt8,Float32}[]
        for parent_index in 1:Topology.parent_count(topology, source)
            edge = Int(Topology.parent_edge(topology, source, parent_index))
            target = Int(Topology.edge_destination(topology, edge))
            target <= CORE_NODE_COUNT || continue
            slot = Int(Topology.child_slot(
                topology,
                target,
                findfirst(index -> Int(Topology.child_edge(topology, target, index)) == edge,
                          1:Topology.child_count(topology, target)),
            ))
            target_class = Topology.node_class(topology, target)
            # Ordered binary children own disjoint four-branch groups.  The
            # delivery callback refines this base branch by plateau group;
            # semantic receivers already expose the exact destination branch.
            base_branch = if target_class == Topology.ROW_INTERNAL_CLASS ||
                             target_class == Topology.COLUMN_INTERNAL_CLASS
                slot == 1 ? 1 : 1 + Axon.GROUP_COUNT
            else
                min(slot, Cell.N_BASAL)
            end
            input_channel = UInt8(Cell.input_index(
                base_branch,
                Cell.INPUT_AMPA,
            ))
            push!(local_edges, (
                UInt16(target), input_channel, EVENT_TRIGGER_MASK,
                0.0f0,
            ))
        end
        sort!(local_edges; by=item -> (item[1], item[2], item[3]))
        for item in local_edges
            event_index += 1
            push!(destination, item[1])
            push!(channel, item[2])
            push!(trigger, item[3])
            # event_raw is defined in this final source-major sorted order;
            # adjacency edge id and optimizer parameter id are identical.
            push!(weights, event_weight[event_index])
        end
        offsets[source + 1] = UInt32(length(destination) + 1)
    end
    return Events.SourceMajorAdjacency(
        CORE_NODE_COUNT,
        Cell.INPUT_DIM,
        offsets,
        destination,
        channel,
        trigger,
        weights,
    )
end

@inline function _hash_parameter_array(
    hash::UInt64,
    values::AbstractArray{Float32},
)
    local_hash = hash
    @inbounds for value in values
        local_hash = _mix_hash(local_hash, UInt64(reinterpret(UInt32, value)))
    end
    return local_hash
end

function _parameter_digest(parameters::ModelParameters)
    hash = UInt64(0xcbf29ce484222325)
    hash = _hash_parameter_array(hash, parameters.core_cell_raw)
    hash = _hash_parameter_array(hash, parameters.semantic_projection_raw)
    hash = _hash_parameter_array(hash, parameters.event_raw)
    hash = _hash_parameter_array(hash, parameters.output.cell_raw)
    hash = _hash_parameter_array(hash, parameters.output.projection_raw)
    return hash
end

function _cache(parameters::ModelParameters, topology::Topology.OrderedTopology)
    cell_cache = Vector{Cell.CellParameterCache{Float32}}(undef, CORE_NODE_COUNT)
    derivative = Vector{Cell.CellParameterDerivativeCache{Float32}}(
        undef,
        CORE_NODE_COUNT,
    )
    @inbounds for node in 1:CORE_NODE_COUNT
        cell_cache[node], derivative[node] = Cell.parameter_caches(
            @view(parameters.core_cell_raw[:, node]),
        )
    end
    projection = similar(parameters.semantic_projection_raw)
    projection_derivative = similar(projection)
    @inbounds @simd for index in eachindex(projection)
        raw = parameters.semantic_projection_raw[index]
        projection[index] = _softplus(raw)
        projection_derivative[index] = _softplus_derivative(raw)
    end
    event_weight = similar(parameters.event_raw)
    @inbounds @simd for index in eachindex(event_weight)
        event_weight[index] = _softplus(parameters.event_raw[index])
    end
    return ModelCache(
        cell_cache,
        derivative,
        projection,
        projection_derivative,
        event_weight,
        _build_event_graph(topology, @view(event_weight[1:_core_event_edge_count(topology)])),
        Output.OutputPopulationCache(parameters.output),
        UInt64(1),
        _parameter_digest(parameters),
    )
end

function initialize_model(
    rng::AbstractRNG=MersenneTwister(0),
    config::GraphConfig=GraphConfig(),
)
    topology = Topology.canonical_topology()
    parameters = _initial_parameters(rng)
    return CanonicalModel(config, topology, parameters, _cache(parameters, topology))
end

function refresh_cache!(model::CanonicalModel)
    cache = model.cache
    cache.revision == typemax(UInt64) && throw(OverflowError(
        "parameter cache revision exhausted",
    ))
    parameters = model.parameters
    @inbounds for node in 1:CORE_NODE_COUNT
        cell_cache, derivative_cache = Cell.parameter_caches(
            @view(parameters.core_cell_raw[:, node]),
        )
        cache.core_cell[node] = cell_cache
        cache.core_derivative[node] = derivative_cache
    end
    @inbounds @simd for index in eachindex(cache.semantic_projection)
        raw = parameters.semantic_projection_raw[index]
        cache.semantic_projection[index] = _softplus(raw)
        cache.semantic_projection_derivative[index] = _softplus_derivative(raw)
    end
    @inbounds @simd for index in eachindex(cache.event_weight)
        cache.event_weight[index] = _softplus(parameters.event_raw[index])
    end
    static_count = _core_event_edge_count(model.topology)
    @inbounds @simd for edge in 1:static_count
        cache.event_graph.weight[edge] = cache.event_weight[edge]
    end
    Output.refresh_cache!(model.cache.output, model.parameters.output)
    cache.revision += UInt64(1)
    cache.parameter_digest = _parameter_digest(parameters)
    return cache
end

function initialize_state(model::CanonicalModel)
    initial_core = Matrix{Float32}(undef, Cell.STATE_DIM, CORE_NODE_COUNT)
    @inbounds for node in 1:CORE_NODE_COUNT
        Cell.initial_state!(@view(initial_core[:, node]), model.cache.core_cell[node])
    end
    output_initial = Matrix{Float32}(undef, Cell.STATE_DIM, Output.OUTPUT_CELLS)
    Output.output_initial_state!(output_initial, model.cache.output)
    return ModelState(
        initial_core,
        copy(initial_core),
        zeros(Float32, Axon.PACKET_DIM, CORE_NODE_COUNT),
        zeros(Float32, Cell.INPUT_DIM, CORE_NODE_COUNT),
        zeros(UInt8, CORE_NODE_COUNT),
        zeros(UInt8, CORE_NODE_COUNT),
        output_initial,
        0.0f0,
        Output.OutputComponents(Float32),
        zeros(Float32, Output.OUTPUT_CELLS),
        Output.OutputPopulationTape(Float32),
        TrajectorySignature(),
        model.cache.revision,
        UInt64(0),
        UInt64(0),
        false,
    )
end

function refresh_state_initial!(model::CanonicalModel, state::ModelState)
    @inbounds for node in 1:CORE_NODE_COUNT
        Cell.initial_state!(
            @view(state.initial_core[:, node]),
            model.cache.core_cell[node],
        )
    end
    Output.output_initial_state!(state.output_initial, model.cache.output)
    state.prepared_revision = model.cache.revision
    state.ready = false
    state.fingerprint = UInt64(0)
    state.common_signature = TrajectorySignature()
    return state
end

function _gradient()
    return ModelGradient(
        zeros(Float32, Cell.PARAM_DIM, CORE_NODE_COUNT),
        zeros(Float32, Axon.GROUP_COUNT, Cell.INPUT_CHANNELS,
              Topology.SEMANTIC_ROLE_COUNT, SEMANTIC_CLASS_COUNT),
        zeros(
            Float32,
            _core_event_edge_count(Topology.canonical_topology()) +
            DYNAMIC_MOTIF_EVENT_COUNT + EVENT_KIND_PARAMETER_COUNT,
        ),
        Output.OutputPopulationGradient(Float32),
    )
end

initialize_gradient(::CanonicalModel) = _gradient()

function _hard_event_gradient(model::CanonicalModel)
    return ModelHardEventGradient(
        zeros(Float32, Cell.PARAM_DIM, CORE_NODE_COUNT),
        zeros(Float32, size(model.parameters.semantic_projection_raw)),
        zeros(Float32, length(model.parameters.event_raw)),
    )
end

"""
Allocate one fixed hard-event publication slot.

The default compact capacity covers exactly one candidate. A replay chunk must
pass `candidate_chunk_size * MAX_COMMON_HARD_EVENT_SEEDS_PER_CANDIDATE`.
"""
function initialize_hard_event_delta(
    model::CanonicalModel,
    seed_capacity::Integer=MAX_COMMON_HARD_EVENT_SEEDS_PER_CANDIDATE,
)
    capacity = Int(seed_capacity)
    capacity >= 1 || throw(ArgumentError(
        "hard-event delta seed capacity must be positive",
    ))
    return ModelHardEventDelta(
        _hard_event_gradient(model),
        zeros(UInt16, capacity),
        zeros(Int32, capacity),
        zeros(UInt32, capacity),
        zeros(UInt16, capacity),
        zeros(UInt8, capacity),
        zeros(Float32, capacity),
        0,
        UInt64(0),
        Int32(0),
        Int32(0),
        UInt64(0xcbf29ce484222325),
        UInt64(0xcbf29ce484222325),
        false,
        false,
    )
end

function clear_hard_event_delta!(delta::ModelHardEventDelta)
    clear_hard_event_gradient!(delta.gradient)
    delta.seed_count = 0
    delta.parameter_digest = UInt64(0)
    delta.logical_first = Int32(0)
    delta.logical_last = Int32(0)
    delta.seed_identity_digest = UInt64(0xcbf29ce484222325)
    delta.hard_gradient_digest = UInt64(0xcbf29ce484222325)
    delta.sealed = false
    delta.reduced = false
    return delta
end

function initialize_worker(model::CanonicalModel)
    config = model.config
    arena = Events.EventArena(
        CORE_NODE_COUNT,
        Cell.STATE_DIM,
        Cell.INPUT_DIM,
        Float32;
        active_capacity=CORE_NODE_COUNT,
        frontier_capacity=CORE_NODE_COUNT,
        overflow=config.event_overflow,
    )
    return ModelWorker(
        Input.CandidateGeometry(),
        Topology.CandidateMotifIncidence(),
        Topology.AffectedClosure(),
        arena,
        initialize_state(model),
        Events.DynamicSourceMajorOverlay(
            CORE_NODE_COUNT,
            Cell.INPUT_DIM,
            DYNAMIC_MOTIF_EVENT_EDGE_CAPACITY,
        ),
        zeros(Float32, Axon.PACKET_DIM, CORE_NODE_COUNT),
        zeros(UInt8, CORE_NODE_COUNT, Axon.EVENT_DIM),
        zeros(UInt16, CORE_NODE_COUNT),
        0,
        zeros(Float32, Cell.INPUT_CHANNELS),
        zeros(Float32, Cell.INPUT_DIM),
        zeros(Float32, Cell.STATE_DIM),
        zeros(Float32, Cell.STATE_DIM),
        zeros(Float32, Axon.PACKET_DIM),
        zeros(UInt8, Axon.EVENT_DIM),
        zeros(Float32, Output.EVIDENCE_DIM, Output.MAX_EVIDENCE,
              Output.OUTPUT_CELLS),
        zeros(UInt8, Output.OUTPUT_CELLS),
        zeros(Float32, Output.OUTPUT_CELLS),
        Output.OutputPopulationTape(Float32),
        Output.OutputPopulationScratch(Float32),
        zeros(Float32, Cell.STATE_DIM, Output.OUTPUT_CELLS),
        zeros(Float32, Output.EVIDENCE_DIM, Output.MAX_EVIDENCE,
              Output.OUTPUT_CELLS),
        [Output.OutputComponents(Float32) for _ in 1:config.max_candidates],
        [Output.OutputComponentGradient(Float32) for _ in 1:config.max_candidates],
        zeros(Float32, Output.OUTPUT_DIM, config.max_candidates),
        zeros(Float32, config.max_candidates),
        fill(TrajectorySignature(), config.max_candidates),
        0,
        TransitionTape(config.tape_capacity),
        ReplayProvenance(
            config.tape_capacity,
            CORE_NODE_COUNT * Cell.N_BASAL,
            _event_delivery_capacity(model),
            Output.OUTPUT_CELLS * Output.MAX_EVIDENCE,
        ),
        _gradient(),
        zeros(Float32, Axon.PACKET_DIM, CORE_NODE_COUNT),
        zeros(Float32, Cell.STATE_DIM, CORE_NODE_COUNT),
        zeros(Float32, Axon.PACKET_DIM, config.tape_capacity),
        zeros(Float32, Cell.STATE_DIM, config.tape_capacity),
        zeros(Float32, Cell.STATE_DIM),
        zeros(Float32, Cell.INPUT_DIM),
        zeros(Float32, Cell.PARAM_DIM),
        zeros(Float32, Cell.STATE_DIM),
        ModelForwardStats(),
        UInt64(0xcbf29ce484222325),
        0,
        UInt64(0),
        UInt64(0),
        UInt64(0),
    )
end

function sync_state_common!(
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
)
    state.ready || throw(ArgumentError(
        "state-common must be prepared before worker synchronization",
    ))
    state.prepared_revision == model.cache.revision || throw(ArgumentError(
        "state initial conditions are stale for the current parameter cache",
    ))
    if worker.prepared_state_epoch != state.epoch ||
       worker.prepared_state_fingerprint != state.fingerprint ||
       worker.prepared_cache_revision != model.cache.revision
        @inbounds for node in 1:CORE_NODE_COUNT, field in 1:Cell.STATE_DIM
            worker.arena.base_state[node, field] = state.common_state[field, node]
        end
        worker.prepared_state_epoch = state.epoch
        worker.prepared_state_fingerprint = state.fingerprint
        worker.prepared_cache_revision = model.cache.revision
    end
    return worker
end

function initialize_local_signal_maps(
    ::CanonicalModel,
    config::Local.LocalLearningConfig=Local.LocalLearningConfig(),
)
    config.predictor_dim == 0 || throw(ArgumentError(
        "canonical local replay requires predictor_dim == 0",
    ))
    config.utility_mode in (:combined, :none) || throw(ArgumentError(
        "factorized two-block replay supports utility_mode=:combined or :none",
    ))
    continuous_scale = config.feedback_scale /
        sqrt(Float32(2 * Local.LOCAL_OBSERVATION_DIM))
    packet_scale = config.feedback_scale /
        sqrt(Float32(2 * Axon.PACKET_DIM))
    continuous = Vector{Local.FixedLocalSignalMap{Float32}}(
        undef,
        CORE_NODE_COUNT,
    )
    packet = similar(continuous)
    @inbounds for node in 1:CORE_NODE_COUNT
        continuous[node] = Local.FixedLocalSignalMap(
            Output.OUTPUT_DIM,
            0;
            observation_dim=Local.LOCAL_OBSERVATION_DIM,
            seed=config.feedback_seed,
            family=1,
            cell=node,
            scale=continuous_scale,
            predictor_scale=0.0f0,
            T=Float32,
        )
        packet[node] = Local.FixedLocalSignalMap(
            Output.OUTPUT_DIM,
            0;
            observation_dim=Axon.PACKET_DIM,
            seed=config.feedback_seed,
            family=2,
            cell=node,
            scale=packet_scale,
            predictor_scale=0.0f0,
            T=Float32,
        )
    end
    return ModelLocalSignalMaps(config, continuous, packet)
end

"""
Allocate the graph-local learner and its fixed hard-event sidecar.

`hard_event_seed_capacity` defaults to one candidate. Multi-candidate chunks
must reserve `chunk_size * MAX_COMMON_HARD_EVENT_SEEDS_PER_CANDIDATE`.
"""
function initialize_local_learner(
    model::CanonicalModel,
    signals::ModelLocalSignalMaps,
    ; hard_event_seed_capacity::Integer=
        MAX_COMMON_HARD_EVENT_SEEDS_PER_CANDIDATE,
)
    length(signals.continuous) == CORE_NODE_COUNT || throw(DimensionMismatch(
        "continuous local-map population does not match the graph",
    ))
    length(signals.packet) == CORE_NODE_COUNT || throw(DimensionMismatch(
        "packet local-map population does not match the graph",
    ))
    seed_capacity = Int(hard_event_seed_capacity)
    seed_capacity >= 1 || throw(ArgumentError(
        "hard-event learner seed capacity must be positive",
    ))
    delivery_capacity = _event_delivery_capacity(model)
    return ModelLocalLearner(
        signals,
        Local.ContractedAdjointArena(CORE_NODE_COUNT; T=Float32),
        Local.ContractedAdjointScratch(; T=Float32),
        zeros(Float32, Local.LOCAL_OBSERVATION_DIM),
        zeros(Float32, Axon.PACKET_DIM),
        Local.FusedContractedAdjointArena(CORE_NODE_COUNT; T=Float32),
        Local.FusedContractedAdjointScratch(; T=Float32),
        zeros(Float32, Local.LOCAL_OBSERVATION_DIM, CORE_NODE_COUNT),
        zeros(Float32, Axon.PACKET_DIM, CORE_NODE_COUNT),
        zeros(Float32, Cell.INPUT_DIM, model.config.tape_capacity),
        fill(false, model.config.tape_capacity),
        zeros(Float32, Axon.EVENT_DIM, model.config.tape_capacity),
        zeros(Float32, delivery_capacity),
        zeros(Int32, delivery_capacity),
        zeros(Int32, model.config.tape_capacity),
        zeros(Int32, model.config.tape_capacity),
        fill(false, delivery_capacity),
        zeros(Float32, Axon.EVENT_DIM, CORE_NODE_COUNT),
        false,
        false,
        false,
        UInt16(0),
        zeros(UInt16, seed_capacity),
        zeros(Int32, seed_capacity),
        zeros(UInt32, seed_capacity),
        zeros(UInt16, seed_capacity),
        zeros(UInt8, seed_capacity),
        zeros(Float32, seed_capacity),
        0,
        0,
        UInt64(0xcbf29ce484222325),
        Int32(0),
        Int32(0),
        UInt64(0),
        _hard_event_gradient(model),
        zeros(Float32, length(model.parameters.event_raw)),
        LocalPlasticityObservation(
            zeros(UInt8, TOTAL_NODE_COUNT),
            zeros(UInt8, TOTAL_NODE_COUNT),
            zeros(Float32, TOTAL_NODE_COUNT),
            zeros(Float32, TOTAL_NODE_COUNT),
            zeros(
                Float32,
                event_parameter_count(model) - EVENT_KIND_PARAMETER_COUNT,
            ),
            zeros(
                Float32,
                event_parameter_count(model) - EVENT_KIND_PARAMETER_COUNT,
            ),
        ),
        LocalReplayCounters(),
    )
end

local_plasticity_observation(learner::ModelLocalLearner) = learner.plasticity
hard_event_gradient(learner::ModelLocalLearner) = learner.hard_gradient
hard_event_delivery_gradient(learner::ModelLocalLearner) =
    learner.delivery_event_gradient

function _reset_local_plasticity!(learner::ModelLocalLearner)
    observation = learner.plasticity
    fill!(observation.spike_count, UInt8(0))
    fill!(observation.visit_count, UInt8(0))
    fill!(observation.activity_sum, 0.0f0)
    fill!(observation.incoming_conductance_sum, 0.0f0)
    fill!(observation.task_utility_sum, 0.0f0)
    fill!(observation.contact_activity_sum, 0.0f0)
    return observation
end

function begin_local_microbatch!(learner::ModelLocalLearner)
    counters = learner.counters
    counters.candidate_replays = 0
    counters.common_replays = 0
    counters.visited_transitions = 0
    counters.conditional_pullbacks = 0
    counters.signal_nonzero = 0
    counters.nonspiking_transitions = 0
    counters.semantic_parameter_updates = 0
    counters.event_receiver_updates = 0
    counters.utility_updates = 0
    counters.output_replays = 0
    counters.event_control_deliveries = 0
    counters.event_control_source_transitions = 0
    counters.event_control_soma_sources = 0
    counters.event_control_plateau_sources = 0
    counters.event_control_common_seeds = 0
    counters.event_control_semantic_updates = 0
    counters.event_control_event_parameter_updates = 0
    Local.reset_adjoint_arena!(learner.arena)
    Local.reset_adjoint_arena!(learner.fused_arena)
    fill!(learner.source_event_advantage, 0.0f0)
    fill!(learner.common_source_event_advantage, 0.0f0)
    learner.common_source_seed_loaded = false
    learner.common_source_seed_consumed = false
    learner.common_source_seed_in_progress = false
    learner.common_source_seed_state_id = UInt16(0)
    learner.hard_seed_count = 0
    learner.hard_expected_seed_count = 0
    learner.hard_seed_identity_digest = UInt64(0xcbf29ce484222325)
    learner.first_hard_candidate_ordinal = Int32(0)
    learner.last_hard_candidate_ordinal = Int32(0)
    learner.hard_parameter_digest = UInt64(0)
    clear_hard_event_gradient!(learner.hard_gradient)
    fill!(learner.delivery_event_gradient, 0.0f0)
    _reset_local_plasticity!(learner)
    return learner
end

@inline common_hard_event_seed_loaded(learner::ModelLocalLearner) =
    learner.common_source_seed_loaded
@inline common_hard_event_seed_consumed(learner::ModelLocalLearner) =
    learner.common_source_seed_consumed
@inline common_hard_event_seed_poisoned(learner::ModelLocalLearner) =
    learner.common_source_seed_in_progress

@inline function _bind_hard_parameter_digest!(
    learner::ModelLocalLearner,
    digest::UInt64,
)
    if iszero(learner.hard_parameter_digest)
        learner.hard_parameter_digest = digest
    else
        learner.hard_parameter_digest == digest || throw(ArgumentError(
            "hard-event microbatch crosses parameter snapshots",
        ))
    end
    return nothing
end

@inline function _copy_hard_event_gradient!(
    destination::ModelHardEventGradient,
    source::ModelHardEventGradient,
)
    size(destination.core_cell_raw) == size(source.core_cell_raw) || throw(
        DimensionMismatch("hard-event core gradient shape changed"),
    )
    size(destination.semantic_projection_raw) ==
        size(source.semantic_projection_raw) || throw(DimensionMismatch(
            "hard-event semantic gradient shape changed",
        ))
    length(destination.event_raw) == length(source.event_raw) || throw(
        DimensionMismatch("hard-event parameter gradient shape changed"),
    )
    copyto!(destination.core_cell_raw, source.core_cell_raw)
    copyto!(destination.semantic_projection_raw, source.semantic_projection_raw)
    copyto!(destination.event_raw, source.event_raw)
    return destination
end

@inline function _hard_seed_identity_digest(
    hash::UInt64,
    state_id::Int,
    candidate::Int,
    delivery::Int,
    source::Int,
    lane::Int,
    advantage::Float32,
)
    hash = _mix_hash(hash, UInt64(state_id))
    hash = _mix_hash(hash, UInt64(candidate))
    hash = _mix_hash(hash, UInt64(delivery))
    hash = _mix_hash(hash, UInt64(source))
    hash = _mix_hash(hash, UInt64(lane))
    return _mix_hash(hash, UInt64(reinterpret(UInt32, advantage)))
end

function _hard_gradient_payload_digest(gradient::ModelHardEventGradient)
    hash = UInt64(0xcbf29ce484222325)
    @inbounds for value in gradient.core_cell_raw
        hash = _mix_hash(hash, UInt64(reinterpret(UInt32, value)))
    end
    @inbounds for value in gradient.semantic_projection_raw
        hash = _mix_hash(hash, UInt64(reinterpret(UInt32, value)))
    end
    @inbounds for value in gradient.event_raw
        hash = _mix_hash(hash, UInt64(reinterpret(UInt32, value)))
    end
    return hash
end

function _validate_learner_hard_seed_ledger!(
    learner::ModelLocalLearner,
    logical_first::Int,
    logical_last::Int,
)
    count = learner.hard_seed_count
    0 <= count <= length(learner.hard_seed_state_id) || error(
        "hard-event learner seed count exceeds its fixed capacity",
    )
    count == learner.hard_expected_seed_count || error(
        "hard-event learner seed count differs from evaluated sentinel coverage",
    )
    identity_digest = UInt64(0xcbf29ce484222325)
    previous_candidate = 0
    previous_state = 0
    previous_delivery = 0
    @inbounds for slot in 1:count
        state_id = Int(learner.hard_seed_state_id[slot])
        candidate = Int(learner.hard_seed_candidate_ordinal[slot])
        delivery = Int(learner.hard_seed_delivery_ordinal[slot])
        source = Int(learner.hard_seed_source_node[slot])
        lane = Int(learner.hard_seed_lane[slot])
        advantage = learner.hard_seed_advantage[slot]
        state_id >= 1 || error("hard-event seed has no state identity")
        logical_first <= candidate <= logical_last || error(
            "hard-event seed candidate is outside its published logical range",
        )
        delivery >= 1 || error("hard-event seed has no physical delivery ordinal")
        1 <= source <= CORE_NODE_COUNT || error(
            "hard-event seed source is outside the core graph",
        )
        1 <= lane <= Axon.EVENT_DIM || error(
            "hard-event seed lane is outside the typed event ABI",
        )
        isfinite(advantage) || error("hard-event seed advantage is not finite")
        if candidate == previous_candidate
            state_id == previous_state || error(
                "one candidate hard-event ledger crosses state identities",
            )
            delivery > previous_delivery || error(
                "hard-event records are not in physical delivery order",
            )
        else
            candidate > previous_candidate || error(
                "hard-event seed candidates are not strictly ordered",
            )
            previous_candidate = candidate
            previous_state = state_id
        end
        previous_delivery = delivery
        identity_digest = _hard_seed_identity_digest(
            identity_digest,
            state_id,
            candidate,
            delivery,
            source,
            lane,
            advantage,
        )
    end
    identity_digest == learner.hard_seed_identity_digest || error(
        "hard-event learner seed identity digest changed",
    )
    return nothing
end

function publish_hard_event_delta!(
    delta::ModelHardEventDelta,
    learner::ModelLocalLearner,
    model::CanonicalModel,
    logical_first_candidate::Integer,
    logical_last_candidate::Integer,
)
    delta.sealed && error("hard-event delta slot is already sealed")
    learner.common_source_seed_loaded && error(
        "loaded common hard-event seeds must be consumed before publication",
    )
    learner.common_source_seed_in_progress && error(
        "failed common hard-event replay must be reset before publication",
    )
    first_candidate = Int(logical_first_candidate)
    last_candidate = Int(logical_last_candidate)
    typemin(Int32) <= first_candidate <= typemax(Int32) || throw(ArgumentError(
        "hard-event logical first candidate does not fit Int32",
    ))
    typemin(Int32) <= last_candidate <= typemax(Int32) || throw(ArgumentError(
        "hard-event logical last candidate does not fit Int32",
    ))
    if learner.first_hard_candidate_ordinal == 0
        first_candidate == 0 && last_candidate == 0 || throw(ArgumentError(
            "hard-event publication range is nonzero without candidate replay",
        ))
    else
        first_candidate == Int(learner.first_hard_candidate_ordinal) &&
            last_candidate == Int(learner.last_hard_candidate_ordinal) || throw(
                ArgumentError(
                    "hard-event publication range differs from replayed candidates",
                ),
            )
    end
    count = learner.hard_seed_count
    delta_capacity = length(delta.seed_state_id)
    length(delta.seed_candidate_ordinal) == delta_capacity &&
        length(delta.seed_delivery_ordinal) == delta_capacity &&
        length(delta.seed_source_node) == delta_capacity &&
        length(delta.seed_lane) == delta_capacity &&
        length(delta.seed_advantage) == delta_capacity || throw(DimensionMismatch(
            "hard-event delta compact seed arrays have unequal capacities",
        ))
    count <= delta_capacity || throw(OverflowError(
        "hard-event delta compact seed capacity exceeded",
    ))
    (iszero(learner.hard_parameter_digest) ||
     learner.hard_parameter_digest == model.cache.parameter_digest) || throw(
        ArgumentError("hard-event learner parameter digest changed"),
    )
    _validate_learner_hard_seed_ledger!(
        learner,
        first_candidate,
        last_candidate,
    )
    all(isfinite, learner.hard_gradient.core_cell_raw) || error(
        "hard-event core gradient is not finite",
    )
    all(isfinite, learner.hard_gradient.semantic_projection_raw) || error(
        "hard-event semantic gradient is not finite",
    )
    all(isfinite, learner.hard_gradient.event_raw) || error(
        "hard-event parameter gradient is not finite",
    )

    _copy_hard_event_gradient!(delta.gradient, learner.hard_gradient)
    count == 0 || begin
        copyto!(delta.seed_state_id, 1, learner.hard_seed_state_id, 1, count)
        copyto!(
            delta.seed_candidate_ordinal,
            1,
            learner.hard_seed_candidate_ordinal,
            1,
            count,
        )
        copyto!(
            delta.seed_delivery_ordinal,
            1,
            learner.hard_seed_delivery_ordinal,
            1,
            count,
        )
        copyto!(delta.seed_source_node, 1, learner.hard_seed_source_node, 1, count)
        copyto!(delta.seed_lane, 1, learner.hard_seed_lane, 1, count)
        copyto!(delta.seed_advantage, 1, learner.hard_seed_advantage, 1, count)
    end
    delta.seed_count = count
    delta.parameter_digest = model.cache.parameter_digest
    delta.logical_first = Int32(first_candidate)
    delta.logical_last = Int32(last_candidate)
    delta.seed_identity_digest = learner.hard_seed_identity_digest
    delta.hard_gradient_digest = _hard_gradient_payload_digest(
        learner.hard_gradient,
    )
    delta.reduced = false
    delta.sealed = true # publication fence: always written last
    return delta
end

function accumulate_common_hard_event_seeds!(
    destination::AbstractArray{Float32,3},
    delta::ModelHardEventDelta,
    model::CanonicalModel;
    expected_first_candidate::Integer=delta.logical_first,
    expected_last_candidate::Integer=delta.logical_last,
)
    size(destination, 1) == Axon.EVENT_DIM || throw(DimensionMismatch(
        "common hard-event seed destination must have five event lanes",
    ))
    size(destination, 2) == CORE_NODE_COUNT || throw(DimensionMismatch(
        "common hard-event seed destination node axis differs from the graph",
    ))
    delta.sealed || error("hard-event delta must be sealed before reduction")
    delta.reduced && error("hard-event delta was already reduced")
    delta.parameter_digest == model.cache.parameter_digest || throw(
        ArgumentError("hard-event delta parameter digest changed"),
    )
    delta_capacity = length(delta.seed_state_id)
    length(delta.seed_candidate_ordinal) == delta_capacity &&
        length(delta.seed_delivery_ordinal) == delta_capacity &&
        length(delta.seed_source_node) == delta_capacity &&
        length(delta.seed_lane) == delta_capacity &&
        length(delta.seed_advantage) == delta_capacity &&
        0 <= delta.seed_count <= delta_capacity || throw(DimensionMismatch(
            "hard-event delta compact seed arrays/count are inconsistent",
        ))
    Int(delta.logical_first) == Int(expected_first_candidate) &&
        Int(delta.logical_last) == Int(expected_last_candidate) || throw(
            ArgumentError("hard-event delta logical range changed"),
        )
    all(isfinite, destination) || throw(ArgumentError(
        "common hard-event reduction destination must be finite",
    ))
    previous_candidate = 0
    previous_state = 0
    previous_delivery = 0
    absolute_seed_sum = 0.0
    identity_digest = UInt64(0xcbf29ce484222325)
    @inbounds for slot in 1:delta.seed_count
        state_id = Int(delta.seed_state_id[slot])
        candidate = Int(delta.seed_candidate_ordinal[slot])
        delivery = Int(delta.seed_delivery_ordinal[slot])
        source = Int(delta.seed_source_node[slot])
        lane = Int(delta.seed_lane[slot])
        advantage = delta.seed_advantage[slot]
        1 <= state_id <= size(destination, 3) || throw(ArgumentError(
            "hard-event delta state identity is outside the reduction batch",
        ))
        Int(expected_first_candidate) <= candidate <=
            Int(expected_last_candidate) || throw(ArgumentError(
                "hard-event seed candidate is outside its logical range",
            ))
        delivery >= 1 || error(
            "hard-event delta has no physical delivery ordinal",
        )
        1 <= source <= CORE_NODE_COUNT || error(
            "hard-event delta source is outside the core graph",
        )
        1 <= lane <= Axon.EVENT_DIM || error(
            "hard-event delta lane is outside the typed event ABI",
        )
        isfinite(advantage) || error(
            "hard-event delta contains a nonfinite seed value",
        )
        absolute_seed_sum += abs(Float64(advantage))
        absolute_seed_sum <= Float64(floatmax(Float32)) || throw(OverflowError(
            "hard-event delta absolute seed sum exceeds Float32",
        ))
        if candidate == previous_candidate
            state_id == previous_state || error(
                "one reduced candidate crosses state identities",
            )
            delivery > previous_delivery || error(
                "hard-event delta records are not in physical delivery order",
            )
        else
            candidate > previous_candidate || error(
                "hard-event delta candidates are not strictly ordered",
            )
            previous_candidate = candidate
            previous_state = state_id
        end
        previous_delivery = delivery
        identity_digest = _hard_seed_identity_digest(
            identity_digest,
            state_id,
            candidate,
            delivery,
            source,
            lane,
            advantage,
        )
    end
    identity_digest == delta.seed_identity_digest || error(
        "hard-event delta seed identity digest changed",
    )
    _hard_gradient_payload_digest(delta.gradient) ==
        delta.hard_gradient_digest || error(
            "hard-event delta gradient payload digest changed",
        )
    @inbounds for slot in 1:delta.seed_count
        state_id = Int(delta.seed_state_id[slot])
        source = Int(delta.seed_source_node[slot])
        lane = Int(delta.seed_lane[slot])
        absolute_seed_sum <= Float64(floatmax(Float32)) -
            abs(Float64(destination[lane, source, state_id])) || throw(OverflowError(
                "common hard-event reduction would overflow Float32",
            ))
    end
    @inbounds for slot in 1:delta.seed_count
        state_id = Int(delta.seed_state_id[slot])
        source = Int(delta.seed_source_node[slot])
        lane = Int(delta.seed_lane[slot])
        destination[lane, source, state_id] += delta.seed_advantage[slot]
    end
    delta.reduced = true
    return destination
end

function load_common_hard_event_seeds!(
    learner::ModelLocalLearner,
    source::AbstractMatrix{Float32},
    state_id::Integer,
)
    size(source) == size(learner.common_source_event_advantage) || throw(
        DimensionMismatch("common hard-event seed slab differs from the graph"),
    )
    1 <= Int(state_id) <= typemax(UInt16) || throw(ArgumentError(
        "common hard-event seed state identity must fit positive UInt16",
    ))
    learner.common_source_seed_loaded && error(
        "common hard-event seed slab is already loaded",
    )
    learner.common_source_seed_consumed && error(
        "begin_local_microbatch! is required before reloading common hard-event seeds",
    )
    learner.common_source_seed_in_progress && error(
        "failed common hard-event replay must be reset before reloading seeds",
    )
    learner.first_hard_candidate_ordinal == 0 &&
        learner.last_hard_candidate_ordinal == 0 &&
        learner.hard_seed_count == 0 || error(
            "common hard-event seeds require a fresh common replay learner",
        )
    all(isfinite, source) || throw(ArgumentError(
        "common hard-event seed slab must be finite",
    ))
    copyto!(learner.common_source_event_advantage, source)
    learner.common_source_seed_state_id = UInt16(state_id)
    learner.common_source_seed_consumed = false
    learner.common_source_seed_in_progress = false
    learner.common_source_seed_loaded = true
    return learner
end

stored_parameter_count(model::CanonicalModel) =
    length(model.parameters.core_cell_raw) +
    length(model.parameters.semantic_projection_raw) +
    length(model.parameters.event_raw) +
    Output.stored_parameter_count(model.parameters.output)

function clear_gradient!(gradient::ModelGradient)
    fill!(gradient.core_cell_raw, 0.0f0)
    fill!(gradient.semantic_projection_raw, 0.0f0)
    fill!(gradient.event_raw, 0.0f0)
    Output.clear_gradient!(gradient.output)
    return gradient
end

clear_gradient!(worker::ModelWorker) = clear_gradient!(worker.gradient)

function accumulate_gradient!(destination::ModelGradient, source::ModelGradient)
    destination.core_cell_raw .+= source.core_cell_raw
    destination.semantic_projection_raw .+= source.semantic_projection_raw
    destination.event_raw .+= source.event_raw
    Output.accumulate_gradient!(destination.output, source.output)
    return destination
end

@inline function _reset_tape!(tape::TransitionTape)
    tape.count = 0
    tape.signature = TrajectorySignature()
    fill!(tape.latest_record, Int32(0))
    fill!(tape.mandatory_record, Int32(0))
    return tape
end

function _reset_provenance!(
    tape::ReplayProvenance,
    parameter_digest::UInt64,
    input_digest::UInt64,
)
    tape.analog_count = 0
    tape.event_count = 0
    tape.output_count = 0
    tape.active_event_wave = 0
    fill!(tape.analog_first_by_record, Int32(0))
    fill!(tape.analog_count_by_record, UInt8(0))
    fill!(tape.event_head_by_node, Int32(0))
    fill!(tape.event_tail_by_node, Int32(0))
    fill!(tape.event_head_by_record, Int32(0))
    tape.parameter_digest = parameter_digest
    tape.input_digest = input_digest
    tape.signature = TrajectorySignature()
    tape.sealed = false
    return tape
end

@inline function _append_analog_deposit!(
    tape::ReplayProvenance,
    kind::AnalogDepositKind,
    source_node::Int,
    source_record::Int,
    packet::AbstractVector{Float32},
    branch::Int,
    semantic_role::Int,
    semantic_class::Int,
)
    slot = tape.analog_count + 1
    slot <= length(tape.analog_kind) || throw(OverflowError(
        "mandatory analog provenance capacity exceeded",
    ))
    tape.analog_kind[slot] = UInt8(kind)
    tape.analog_destination_record[slot] = Int32(0)
    tape.analog_source_node[slot] = UInt16(source_node)
    tape.analog_source_record[slot] = Int32(source_record)
    tape.analog_branch[slot] = UInt8(branch)
    tape.analog_semantic_role[slot] = UInt8(semantic_role)
    tape.analog_semantic_class[slot] = UInt8(semantic_class)
    tape.analog_ordinal[slot] = UInt32(slot)
    @inbounds @simd for lane in 1:Axon.PACKET_DIM
        tape.analog_packet[lane, slot] = packet[lane]
    end
    tape.analog_count = slot
    return slot
end

@inline function _bind_analog_deposits!(
    tape::ReplayProvenance,
    first::Int,
    destination_record::Int,
)
    count = tape.analog_count - first + 1
    0 <= count <= Cell.N_BASAL || throw(ArgumentError(
        "one mandatory transition exceeded eight anatomical deposits",
    ))
    count == 0 && return nothing
    @inbounds for slot in first:tape.analog_count
        tape.analog_destination_record[slot] = Int32(destination_record)
    end
    tape.analog_first_by_record[destination_record] = Int32(first)
    tape.analog_count_by_record[destination_record] = UInt8(count)
    return nothing
end

@inline function _rollback_analog_deposits!(
    tape::ReplayProvenance,
    first::Int,
)
    tape.analog_count = first - 1
    return nothing
end

@inline function _begin_event_delivery_wave!(
    tape::ReplayProvenance,
    wave::Int,
)
    if tape.active_event_wave != wave
        fill!(tape.event_head_by_node, Int32(0))
        fill!(tape.event_tail_by_node, Int32(0))
        tape.active_event_wave = wave
    end
    return nothing
end

@inline function _validate_event_source_record(
    tape::TransitionTape,
    source_node::Int,
    source_record::Int,
    common_phase::Bool,
    candidate_affected::Bool,
    wave::Int,
    append_time::Bool,
)
    1 <= source_node <= CORE_NODE_COUNT || throw(ArgumentError(
        "event provenance source_node must identify a core node",
    ))
    Int(COMMON_SOURCE_RECORD) <= source_record <= typemax(Int32) || throw(
        ArgumentError(
            "event provenance source_record must fit the nonnegative Int32 ABI",
        ),
    )
    expected_common = !common_phase && wave == 1 && !candidate_affected
    is_common_source_record(source_node, source_record) == expected_common || throw(
        ArgumentError(
            expected_common ?
            "candidate overlay finalized-common source must use COMMON_SOURCE_RECORD" :
            "COMMON_SOURCE_RECORD is valid only for a candidate wave-one overlay source",
        ),
    )
    if expected_common
        if append_time
            Int(@inbounds tape.latest_record[source_node]) ==
                Int(COMMON_SOURCE_RECORD) || throw(ArgumentError(
                    "candidate overlay common source already has a logical candidate record",
                ))
        end
    else
        1 <= source_record <= tape.count || throw(ArgumentError(
            "event provenance source_record is outside the transition tape",
        ))
        Int(@inbounds tape.node[source_record]) == source_node || throw(
            ArgumentError(
                "event provenance source_record does not belong to source_node",
            ),
        )
    end
    return nothing
end

@inline function _append_event_delivery!(
    provenance::ReplayProvenance,
    tape::TransitionTape,
    common_phase::Bool,
    candidate_affected::Bool,
    destination::Int,
    source_node::Int,
    source_record::Int,
    source_mask::UInt8,
    lane::Int,
    resolved_channel::Int,
    contact_parameter::Int,
    kind_parameter::Int,
    scale::Float32,
    wave::Int,
)
    _validate_event_source_record(
        tape,
        source_node,
        source_record,
        common_phase,
        candidate_affected,
        wave,
        true,
    )
    _begin_event_delivery_wave!(provenance, wave)
    slot = provenance.event_count + 1
    slot <= length(provenance.event_source_node) || throw(OverflowError(
        "typed event-delivery provenance capacity exceeded",
    ))
    provenance.event_destination_record[slot] = Int32(0)
    provenance.event_source_node[slot] = UInt16(source_node)
    provenance.event_source_record[slot] = Int32(source_record)
    provenance.event_source_mask[slot] = source_mask
    provenance.event_lane[slot] = UInt8(lane)
    provenance.event_destination_branch[slot] = UInt8(
        div(resolved_channel - 1, Cell.INPUT_CHANNELS) + 1,
    )
    receptor = mod(resolved_channel - 1, Cell.INPUT_CHANNELS) + 1
    provenance.event_polarity[slot] = UInt8(
        lane == Axon.SOMA_EVENT ? 0 :
        receptor == Cell.INPUT_NMDA ? 1 : 2,
    )
    provenance.event_resolved_channel[slot] = UInt8(resolved_channel)
    provenance.event_contact_parameter[slot] = UInt16(contact_parameter)
    provenance.event_kind_parameter[slot] = UInt16(kind_parameter)
    provenance.event_scale[slot] = scale
    provenance.event_wave[slot] = UInt8(wave)
    provenance.event_ordinal[slot] = UInt32(slot)
    provenance.event_next[slot] = Int32(0)
    previous = Int(@inbounds provenance.event_tail_by_node[destination])
    if previous == 0
        provenance.event_head_by_node[destination] = Int32(slot)
    else
        provenance.event_next[previous] = Int32(slot)
    end
    provenance.event_tail_by_node[destination] = Int32(slot)
    provenance.event_count = slot
    return slot
end

@inline function _bind_event_deliveries!(
    tape::ReplayProvenance,
    destination::Int,
    destination_record::Int,
)
    head = Int(@inbounds tape.event_head_by_node[destination])
    tape.event_head_by_record[destination_record] = Int32(head)
    slot = head
    while slot != 0
        @inbounds tape.event_destination_record[slot] = Int32(destination_record)
        slot = Int(@inbounds tape.event_next[slot])
    end
    return nothing
end

@inline function _reset_output_provenance!(tape::ReplayProvenance)
    tape.output_count = 0
    return nothing
end

@inline function _append_output_evidence!(
    tape::ReplayProvenance,
    source_node::Int,
    source_record::Int,
    output_cell::Int,
    evidence_rank::Int,
    packet::AbstractVector{Float32},
)
    slot = tape.output_count + 1
    slot <= length(tape.output_source_node) || throw(OverflowError(
        "output-evidence provenance capacity exceeded",
    ))
    tape.output_source_node[slot] = UInt16(source_node)
    tape.output_source_record[slot] = Int32(source_record)
    tape.output_cell[slot] = UInt8(output_cell)
    tape.output_rank[slot] = UInt8(evidence_rank)
    tape.output_ordinal[slot] = UInt16(slot)
    @inbounds @simd for lane in 1:Axon.PACKET_DIM
        tape.output_packet[lane, slot] = packet[lane]
    end
    tape.output_count = slot
    return slot
end

@inline function _mix_hash(hash::UInt64, value::UInt64)
    return xor(hash, value) * UInt64(0x00000100000001b3)
end

function _record_transition!(
    tape::TransitionTape,
    node::Int,
    phase::TransitionPhase,
    wave::Int,
    previous_state::AbstractVector{Float32},
    input::AbstractVector{Float32},
    next_state::AbstractVector{Float32},
    packet::AbstractVector{Float32},
    event_mask::UInt8,
    logical_version::Bool,
)
    slot = tape.count + 1
    slot <= length(tape.node) || throw(OverflowError(
        "canonical replay tape capacity exceeded",
    ))
    tape.node[slot] = UInt16(node)
    tape.phase[slot] = UInt8(phase)
    tape.wave[slot] = UInt8(wave)
    tape.event_mask[slot] = event_mask
    tape.previous_record[slot] = logical_version ? tape.latest_record[node] : Int32(0)
    copyto!(@view(tape.previous_state[:, slot]), previous_state)
    copyto!(@view(tape.input[:, slot]), input)
    copyto!(@view(tape.next_state[:, slot]), next_state)
    copyto!(@view(tape.packet[:, slot]), packet)
    if logical_version
        tape.latest_record[node] = Int32(slot)
        phase == MANDATORY_DAG && (tape.mandatory_record[node] = Int32(slot))
    end
    tape.count = slot
    return slot
end

@inline function _event_mask(events::AbstractVector{UInt8})
    mask = UInt8(0)
    @inbounds for lane in 1:Axon.EVENT_DIM
        iszero(events[lane]) || (mask |= UInt8(1 << (lane - 1)))
    end
    return mask
end

@inline function _slot(worker::ModelWorker, node::Int)
    return Events.state_slot(worker.arena, node)
end

@inline function _node_state(
    worker::ModelWorker,
    state::ModelState,
    node::Int,
)
    slot = _slot(worker, node)
    return slot == 0 ? @view(state.common_state[:, node]) :
                       @view(worker.arena.state[slot, :])
end

@inline function _node_packet(
    worker::ModelWorker,
    state::ModelState,
    node::Int,
)
    slot = _slot(worker, node)
    return slot == 0 ? @view(state.common_packet[:, node]) :
                       @view(worker.packet_by_slot[:, slot])
end

@inline function _append_node_analog_deposit!(
    worker::ModelWorker,
    state::ModelState,
    kind::AnalogDepositKind,
    source::Int,
    branch::Int,
    semantic_role::Int,
    semantic_class::Int,
)
    slot = _slot(worker, source)
    source_record = Int(@inbounds worker.tape.mandatory_record[source])
    if iszero(slot)
        return _append_analog_deposit!(
            worker.provenance, kind, source, source_record,
            @view(state.common_packet[:, source]), branch,
            semantic_role, semantic_class,
        )
    end
    return _append_analog_deposit!(
        worker.provenance, kind, source, source_record,
        @view(worker.packet_by_slot[:, slot]), branch,
        semantic_role, semantic_class,
    )
end

@inline function _append_node_output_evidence!(
    worker::ModelWorker,
    state::ModelState,
    source::Int,
    output_cell::Int,
    evidence_rank::Int,
)
    slot = _slot(worker, source)
    source_record = Int(@inbounds worker.tape.latest_record[source])
    if iszero(slot)
        return _append_output_evidence!(
            worker.provenance, source, source_record, output_cell,
            evidence_rank, @view(state.common_packet[:, source]),
        )
    end
    return _append_output_evidence!(
        worker.provenance, source, source_record, output_cell,
        evidence_rank, @view(worker.packet_by_slot[:, slot]),
    )
end

@inline function candidate_state(
    worker::ModelWorker,
    state::ModelState,
    node::Integer,
)
    physical = Int(node)
    1 <= physical <= CORE_NODE_COUNT || throw(BoundsError(1:CORE_NODE_COUNT, physical))
    return _node_state(worker, state, physical)
end

@inline function candidate_packet(
    worker::ModelWorker,
    state::ModelState,
    node::Integer,
)
    physical = Int(node)
    1 <= physical <= CORE_NODE_COUNT || throw(BoundsError(1:CORE_NODE_COUNT, physical))
    return _node_packet(worker, state, physical)
end

@inline motif_state(worker::ModelWorker, state::ModelState, family::Integer, slot::Integer) =
    candidate_state(worker, state, Topology.motif_node(family, slot))
@inline motif_packet(worker::ModelWorker, state::ModelState, family::Integer, slot::Integer) =
    candidate_packet(worker, state, Topology.motif_node(family, slot))
@inline evidence_state(worker::ModelWorker, state::ModelState, index::Integer) =
    candidate_state(worker, state, Topology.evidence_node(index))
@inline evidence_packet(worker::ModelWorker, state::ModelState, index::Integer) =
    candidate_packet(worker, state, Topology.evidence_node(index))

@inline function _semantic_class(topology::Topology.OrderedTopology, node::Int)
    class = Topology.node_class(topology, node)
    class == Topology.MOTIF_CLASS && return 1
    class == Topology.EVIDENCE_CLASS && return 2
    throw(ArgumentError("node $node is not a core semantic receiver"))
end

@inline dynamic_event_contact_count(::CanonicalModel) = DYNAMIC_MOTIF_EVENT_COUNT

@inline function _dynamic_family_branch_ordinal(family::Int, branch::Int)
    1 <= family <= Topology.MOTIF_FAMILY_COUNT || throw(BoundsError(
        1:Topology.MOTIF_FAMILY_COUNT,
        family,
    ))
    if family == 1 || family == 3 || family == 4
        1 <= branch <= 2 || throw(ArgumentError(
            "motif family $family branch $branch has no live event contact",
        ))
        return branch
    elseif family == 2
        2 <= branch <= 3 || throw(ArgumentError(
            "motif family 2 branch $branch has no live event contact",
        ))
        return branch - 1
    elseif family == 5 || family == 6
        1 <= branch <= 6 || throw(ArgumentError(
            "motif family $family branch $branch has no live event contact",
        ))
        return branch
    end
    throw(ArgumentError(
        "motif family $family has no live upstream event contacts",
    ))
end

@inline function _dynamic_family_prefix(family::Int)
    family == 1 && return 0
    family == 2 && return 2
    family == 3 && return 4
    family == 4 && return 6
    family == 5 && return 8
    family == 6 && return 14
    1 <= family <= Topology.MOTIF_FAMILY_COUNT || throw(BoundsError(
        1:Topology.MOTIF_FAMILY_COUNT,
        family,
    ))
    throw(ArgumentError(
        "motif family $family has no live upstream event contacts",
    ))
end

"""
Return the checkpoint-stable slot-major compact identity in `1:80`.

Each slot owns twenty anatomical contacts in family order.  Family 2 uses
physical branches `(2,3)`; no caller is permitted to reinterpret the compact
ordinal as a physical branch.
"""
@inline function dynamic_event_pair_index(
    family::Integer,
    slot::Integer,
    branch::Integer,
)
    physical_family = Int(family)
    physical_slot = Int(slot)
    physical_branch = Int(branch)
    1 <= physical_slot <= Topology.MOTIF_SLOTS_PER_FAMILY || throw(BoundsError(
        1:Topology.MOTIF_SLOTS_PER_FAMILY,
        physical_slot,
    ))
    ordinal = _dynamic_family_branch_ordinal(
        physical_family,
        physical_branch,
    )
    return 20 * (physical_slot - 1) +
        _dynamic_family_prefix(physical_family) + ordinal
end

"""Inverse of `dynamic_event_pair_index`, used by checkpoints and audits."""
@inline function dynamic_event_pair_descriptor(index::Integer)
    compact = Int(index)
    1 <= compact <= DYNAMIC_MOTIF_EVENT_COUNT || throw(BoundsError(
        1:DYNAMIC_MOTIF_EVENT_COUNT,
        compact,
    ))
    slot = div(compact - 1, 20) + 1
    local_index = mod(compact - 1, 20) + 1
    family, branch = if local_index <= 2
        1, local_index
    elseif local_index <= 4
        2, local_index - 1
    elseif local_index <= 6
        3, local_index - 4
    elseif local_index <= 8
        4, local_index - 6
    elseif local_index <= 14
        5, local_index - 8
    else
        6, local_index - 14
    end
    return DynamicEventContactDescriptor(
        UInt8(family),
        UInt8(slot),
        UInt8(branch),
    )
end

@inline function dynamic_event_parameter_index(
    model::CanonicalModel,
    family::Integer,
    slot::Integer,
    branch::Integer,
)
    return _core_event_edge_count(model.topology) +
        dynamic_event_pair_index(family, slot, branch)
end

@inline function _dynamic_event_parameter_index(
    model::CanonicalModel,
    motif::Int,
    branch_slot::Int,
)
    family = div(motif - 1, Topology.MOTIF_SLOTS_PER_FAMILY) + 1
    slot = mod(motif - 1, Topology.MOTIF_SLOTS_PER_FAMILY) + 1
    return dynamic_event_parameter_index(model, family, slot, branch_slot)
end

@inline function _event_contact_parameter_count(model::CanonicalModel)
    return _core_event_edge_count(model.topology) + DYNAMIC_MOTIF_EVENT_COUNT
end

@inline function event_kind_parameter_index(
    model::CanonicalModel,
    lane::Integer,
)
    physical_lane = Int(lane)
    1 <= physical_lane <= Axon.EVENT_DIM || throw(BoundsError(
        1:Axon.EVENT_DIM,
        physical_lane,
    ))
    return _event_contact_parameter_count(model) + physical_lane
end

@inline event_parameter_count(model::CanonicalModel) =
    length(model.parameters.event_raw)

@inline function _static_event_source(
    graph::Events.SourceMajorAdjacency,
    edge::Int,
)
    1 <= edge <= length(graph.destination) || throw(BoundsError(
        1:length(graph.destination),
        edge,
    ))
    low = 1
    high = graph.node_count
    while low <= high
        source = (low + high) >>> 1
        first_edge = Int(@inbounds graph.offsets[source])
        last_edge = Int(@inbounds graph.offsets[source + 1]) - 1
        if edge < first_edge
            high = source - 1
        elseif edge > last_edge
            low = source + 1
        else
            return source
        end
    end
    error("static event edge $edge is absent from source-major adjacency")
end

"""
Return the canonical, allocation-free identity of `event_raw[index]`.

The order is the source-major static prefix, the slot-major live anatomical
dynamic contacts, then the five shared `(soma,p1,p2,p3,p4)` event-kind gains.
"""
function event_parameter_descriptor(
    model::CanonicalModel,
    index::Integer,
)
    parameter = Int(index)
    1 <= parameter <= event_parameter_count(model) || throw(BoundsError(
        1:event_parameter_count(model),
        parameter,
    ))
    static_count = _core_event_edge_count(model.topology)
    if parameter <= static_count
        graph = model.cache.event_graph
        return EventParameterDescriptor(
            STATIC_EVENT_CONTACT,
            UInt16(_static_event_source(graph, parameter)),
            @inbounds(graph.destination[parameter]),
            @inbounds(graph.channel[parameter]),
            0x00,
            0x00,
            0x00,
            0x00,
        )
    end
    dynamic_limit = static_count + DYNAMIC_MOTIF_EVENT_COUNT
    if parameter <= dynamic_limit
        descriptor = dynamic_event_pair_descriptor(parameter - static_count)
        destination = Topology.motif_node(
            Int(descriptor.family),
            Int(descriptor.slot),
        )
        channel = Cell.input_index(
            Int(descriptor.branch),
            Cell.INPUT_AMPA,
        )
        return EventParameterDescriptor(
            DYNAMIC_EVENT_CONTACT,
            0x0000,
            destination,
            UInt8(channel),
            descriptor.family,
            descriptor.slot,
            descriptor.branch,
            0x00,
        )
    end
    lane = parameter - dynamic_limit
    return EventParameterDescriptor(
        SHARED_EVENT_KIND_GAIN,
        0x0000,
        0x0000,
        0x00,
        0x00,
        0x00,
        0x00,
        UInt8(lane),
    )
end

function _fill_binary_input!(
    destination::AbstractVector{Float32},
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
    node::Int,
)
    _zero_input!(destination)
    Topology.child_count(model.topology, node) == 2 || error(
        "ordered spine node $node lost binary fan-in",
    )
    left = Int(Topology.child_node(model.topology, node, 1))
    right = Int(Topology.child_node(model.topology, node, 2))
    fill!(worker.context, 0.0f0)
    Axon.ordered_binary_deposit!(
        destination,
        _node_packet(worker, state, left),
        _node_packet(worker, state, right),
        worker.context,
    )
    _append_node_analog_deposit!(
        worker, state, BINARY_PACKET_DEPOSIT, left, 1, 0, 0,
    )
    _append_node_analog_deposit!(
        worker, state, BINARY_PACKET_DEPOSIT, right,
        1 + Axon.GROUP_COUNT, 0, 0,
    )
    return destination
end

function _fill_static_semantic_input!(
    destination::AbstractVector{Float32},
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
    node::Int,
)
    _zero_input!(destination)
    semantic_class = _semantic_class(model.topology, node)
    count = Topology.child_count(model.topology, node)
    count <= Cell.N_BASAL || error("semantic receiver fan-in exceeded eight")
    @inbounds for source_index in 1:count
        edge = Int(Topology.child_edge(model.topology, node, source_index))
        source = Int(Topology.edge_source(model.topology, edge))
        role = Int(Topology.edge_semantic_role(model.topology, edge))
        branch = Int(Topology.child_slot(model.topology, node, source_index))
        packet = _node_packet(worker, state, source)
        for receptor in 1:Cell.INPUT_CHANNELS
            drive = 0.0f0
            for group in 1:Axon.GROUP_COUNT
                drive = muladd(
                    model.cache.semantic_projection[
                        group, receptor, role, semantic_class,
                    ],
                    packet[Axon.packet_lane(group, receptor)],
                    drive,
                )
            end
            destination[Cell.input_index(branch, receptor)] = drive
        end
        _append_node_analog_deposit!(
            worker, state, SEMANTIC_PACKET_DEPOSIT, source,
            branch, role, semantic_class,
        )
    end
    return destination
end

@inline function _project_packet!(
    destination::AbstractVector{Float32},
    model::CanonicalModel,
    packet::AbstractVector{Float32},
    branch::Int,
    role::Int,
    semantic_class::Int,
)
    1 <= branch <= Cell.N_BASAL || throw(BoundsError(1:Cell.N_BASAL, branch))
    1 <= role <= Topology.SEMANTIC_ROLE_COUNT || throw(
        BoundsError(1:Topology.SEMANTIC_ROLE_COUNT, role),
    )
    @inbounds for receptor in 1:Cell.INPUT_CHANNELS
        drive = 0.0f0
        for group in 1:Axon.GROUP_COUNT
            drive = muladd(
                model.cache.semantic_projection[
                    group, receptor, role, semantic_class,
                ],
                packet[Axon.packet_lane(group, receptor)],
                drive,
            )
        end
        destination[Cell.input_index(branch, receptor)] += drive
    end
    return destination
end

function _motif_context(
    input,
    geometry::Input.CandidateGeometry,
)
    placement_count = Input.placement_count(input)
    positions = ntuple(Val(4)) do index
        index <= placement_count ? Input.placement_position(input, index) : UInt16(0)
    end
    row_remap = ntuple(Val(Input.BOARD_ROWS)) do row
        UInt8(Input.source_to_after(geometry, row))
    end
    full_row_mask = UInt32(0)
    @inbounds for row in 1:Input.BOARD_ROWS
        Input.full_row(geometry, row) || continue
        full_row_mask |= UInt32(1) << (row - 1)
    end
    return Topology.CandidateMotifContext(
        positions,
        placement_count,
        row_remap,
        full_row_mask,
        Input.clear_count(geometry),
        UInt8(Input.hold_piece(input)),
        ntuple(Val(Input.NEXT_COUNT)) do index
            UInt8(Input.next_piece(input, index))
        end,
        Input.ren_value(input),
        UInt8(Input.back_to_back_value(input)),
        UInt8(Input.tspin_value(input)),
    )
end

function _fill_dynamic_motif_input!(
    destination::AbstractVector{Float32},
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
    node::Int,
)
    _zero_input!(destination)
    motif = (node - Int(Topology.motif_node(1, 1))) + 1
    family = Int(Topology.motif_family(model.topology, node))
    count = Topology.motif_source_count(worker.motif_incidence, motif)
    count <= Cell.N_BASAL || error("candidate motif fan-in exceeded eight")
    @inbounds for rank in 1:count
        source = Topology.motif_source(worker.motif_incidence, motif, rank)
        if Topology.motif_source_is_spine(source)
            source_node = Int(source.node)
            packet = _node_packet(worker, state, source_node)
            _project_packet!(
                destination,
                model,
                packet,
                Int(source.branch_slot),
                family,
                1,
            )
            _append_node_analog_deposit!(
                worker, state, SEMANTIC_PACKET_DEPOSIT, source_node,
                Int(source.branch_slot), family, 1,
            )
        else
            Topology.materialize_external_motif_packet!(
                worker.packet_scratch,
                source,
            )
            _project_packet!(
                destination,
                model,
                worker.packet_scratch,
                Int(source.branch_slot),
                family,
                1,
            )
            _append_analog_deposit!(
                worker.provenance, SEMANTIC_PACKET_DEPOSIT, 0, 0,
                worker.packet_scratch, Int(source.branch_slot), family, 1,
            )
        end
    end
    return destination
end

function _fill_candidate_input!(
    destination::AbstractVector{Float32},
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
    input,
    node::Int,
)
    class = Topology.node_class(model.topology, node)
    if class == Topology.SPATIAL_CLASS
        plane = Topology.node_plane(model.topology, node)
        position = Int(Topology.spatial_position(node))
        row = mod(position - 1, Input.BOARD_ROWS) + 1
        column = div(position - 1, Input.BOARD_ROWS) + 1
        accessor = plane == Topology.BEFORE_PLANE ?
            Spatial.BeforeSiteAccessor(input) :
            Spatial.AfterSiteAccessor(worker.geometry)
        Spatial.fill_spatial_drive!(
            destination,
            accessor,
            row,
            column,
            plane,
            plane == Topology.BEFORE_PLANE ? COMMON_BEFORE : CANDIDATE_AFTER,
        )
    elseif class == Topology.ROW_INTERNAL_CLASS ||
           class == Topology.COLUMN_INTERNAL_CLASS
        _fill_binary_input!(destination, model, state, worker, node)
    elseif class == Topology.MOTIF_CLASS
        _fill_dynamic_motif_input!(destination, model, state, worker, node)
    elseif class == Topology.EVIDENCE_CLASS
        _fill_static_semantic_input!(destination, model, state, worker, node)
    else
        throw(ArgumentError("output cells are owned by DendriticOutputPopulation"))
    end
    return destination
end

function _transition!(
    model::CanonicalModel,
    worker::ModelWorker,
    node::Int,
    previous::AbstractVector{Float32},
    input::AbstractVector{Float32},
    destination::AbstractVector{Float32},
    packet::AbstractVector{Float32},
    events::AbstractVector{UInt8},
    phase::TransitionPhase,
    wave::Int,
    record::Bool,
)
    Cell.cell_step!(destination, previous, input, model.cache.core_cell[node])
    Axon.axon_packet!(packet, previous, destination, model.cache.core_cell[node])
    Axon.hard_events!(events, previous, destination)
    mask = _event_mask(events)
    record && _record_transition!(
        worker.tape,
        node,
        phase,
        wave,
        previous,
        input,
        destination,
        packet,
        mask,
        phase != MANDATORY_DAG || _logical_candidate_affected(worker, node),
    )
    return mask
end

function _fill_evidence_for_output!(
    worker::ModelWorker,
    model::CanonicalModel,
    state::ModelState,
)
    fill!(worker.output_evidence, 0.0f0)
    fill!(worker.output_evidence_count, UInt8(0))
    _reset_output_provenance!(worker.provenance)
    @inbounds for output_cell in 3:Output.OUTPUT_CELLS
        node = Int(Topology.output_node(output_cell))
        count = Topology.child_count(model.topology, node)
        count <= Output.MAX_EVIDENCE || error("output fan-in exceeded eight")
        worker.output_evidence_count[output_cell] = UInt8(count)
        for source_index in 1:count
            source = Int(Topology.child_node(
                model.topology,
                node,
                source_index,
            ))
            copyto!(
                @view(worker.output_evidence[:, source_index, output_cell]),
                _node_packet(worker, state, source),
            )
            _append_node_output_evidence!(
                worker, state, source, output_cell, source_index,
            )
        end
    end
    return worker.output_evidence
end

function _state_fingerprint(input)
    hash = UInt64(0xcbf29ce484222325)
    @inbounds for column in 1:Input.BOARD_COLUMNS, row in 1:Input.BOARD_ROWS
        hash = _mix_hash(hash, UInt64(Input.before_cell(input, row, column)))
    end
    # State metadata are part of the teacher-sufficient state identity.  The
    # public accessor functions are supplied by CanonicalTetrisInput.
    hash = _mix_hash(hash, UInt64(Input.hold_piece(input)))
    for index in 1:Input.NEXT_COUNT
        hash = _mix_hash(hash, UInt64(Input.next_piece(input, index)))
    end
    hash = _mix_hash(hash, UInt64(Input.ren_value(input)))
    hash = _mix_hash(hash, UInt64(Input.back_to_back_value(input)))
    return hash
end

"""
Assemble one state after all candidate-local physical output populations ran.
The seven-dimensional independent output geometry remains explicit; the
external 22D quantile ABI is assembled only here.
"""
function assemble_candidate_set!(
    destination::AbstractMatrix{Float32},
    state_value::Float32,
    components::AbstractVector{<:Output.OutputComponents{Float32}},
    candidate_count::Integer,
)
    count = Int(candidate_count)
    1 <= count <= length(components) || throw(BoundsError(components, count))
    size(destination) == (Output.OUTPUT_DIM, count) || throw(
        DimensionMismatch("destination must have shape (22, $count)"),
    )
    advantage_sum = 0.0
    @inbounds for candidate in 1:count
        advantage_sum += Float64(components[candidate].advantage)
    end
    advantage_mean = Float32(advantage_sum / count)
    @inbounds for candidate in 1:count
        # V(s) is computed once from BEFORE-only state-common evidence.  The
        # candidate-private output population owns A/death/geometry/sigma but
        # never substitutes a candidate-conditioned value.
        components[candidate].value = state_value
        Output.assemble_output!(
            @view(destination[:, candidate]),
            components[candidate],
            advantage_mean,
        )
    end
    return destination
end


function _fill_state_context_packet!(
    destination::AbstractVector{Float32},
    input,
    context_role::Int,
)
    length(destination) == Axon.PACKET_DIM || throw(DimensionMismatch(
        "state-context packet must have length 12",
    ))
    1 <= context_role <= 6 || throw(BoundsError(1:6, context_role))
    piece = context_role == 1 ? Input.hold_piece(input) :
            Input.next_piece(input, context_role - 1)
    ren_bits = reinterpret(UInt32, Input.ren_value(input))
    fill!(destination, 0.0f0)
    destination[1] = Float32(context_role) / 6.0f0
    destination[2] = Float32(UInt8(piece)) / 8.0f0
    destination[3] = 1.0f0 # typed state-context identity
    @inbounds for byte in 0:3
        destination[4 + byte] =
            Float32((ren_bits >> (8 * byte)) & 0xff) / 255.0f0
    end
    destination[8] = Float32(UInt8(Input.back_to_back_value(input))) / 2.0f0
    destination[9] = context_role == 1 ? 1.0f0 : 0.5f0
    destination[10] = piece == Input.NONE ? 0.5f0 : 1.0f0
    destination[11] = 1.0f0
    destination[12] = 1.0f0
    return destination
end

function _prepare_shared_state_value!(
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
    input,
)
    evidence = worker.output_evidence
    counts = worker.output_evidence_count
    fill!(evidence, 0.0f0)
    fill!(counts, UInt8(0))
    _reset_output_provenance!(worker.provenance)
    # Exact 16-fibre state-common contract: all ten ordered BEFORE column
    # roots plus six role-bound HOLD/NEXT context packets.  Cell 1 receives
    # columns1:5,HOLD,NEXT1,NEXT2; cell2 receives columns6:10,NEXT3:5.
    # REN and B2B are repeated in every context packet, so V(s) is invariant
    # to candidate order yet remains teacher-sufficient for the state.
    @inbounds for value_cell in 1:2
        source_rank = 0
        first_column = value_cell == 1 ? 1 : 6
        for column in first_column:(first_column + 4)
            source_rank += 1
            source = Int(Topology.column_root_node(
                Topology.BEFORE_PLANE,
                column,
            ))
            copyto!(
                @view(evidence[:, source_rank, value_cell]),
                @view(state.common_packet[:, source]),
            )
            _append_output_evidence!(
                worker.provenance,
                source,
                Int(@inbounds worker.tape.latest_record[source]),
                value_cell,
                source_rank,
                @view(state.common_packet[:, source]),
            )
        end
        context_first = value_cell == 1 ? 1 : 4
        for context_role in context_first:(context_first + 2)
            source_rank += 1
            _fill_state_context_packet!(
                @view(evidence[:, source_rank, value_cell]),
                input,
                context_role,
            )
            _append_output_evidence!(
                worker.provenance,
                0,
                0,
                value_cell,
                source_rank,
                @view(evidence[:, source_rank, value_cell]),
            )
        end
        counts[value_cell] = UInt8(source_rank)
    end
    Output.value_output_population_forward!(
        state.state_value_components,
        state.state_value_hard_event,
        state.state_value_tape,
        state.output_initial,
        evidence,
        counts,
        model.parameters.output,
        model.cache.output,
    )
    state.state_value = state.state_value_components.value
    _seal_replay_provenance!(
        model,
        state,
        worker,
        state.common_signature,
        true,
    )
    return state.state_value
end

function _reset_stats!(stats::ModelForwardStats)
    stats.common_transitions = 0
    stats.common_event_transitions = 0
    stats.common_event_waves = 0
    stats.mandatory_transitions = 0
    stats.event_transitions = 0
    stats.output_transitions = 0
    stats.event_waves = 0
    stats.hard_events = 0
    stats.clear_slow_paths = 0
    return stats
end

function _finalize_common_signature!(
    state::ModelState,
    worker::ModelWorker,
    report::Events.EventWaveReport,
)
    soma_hash = UInt64(0xcbf29ce484222325)
    plateau_hash = UInt64(0xcbf29ce484222325)
    frontier_hash = UInt64(0xcbf29ce484222325)
    @inbounds for node in 1:CORE_NODE_COUNT
        mask = state.common_seed_mask[node]
        iszero(mask) && continue
        soma_hash = _mix_hash(
            soma_hash,
            (UInt64(node) << 1) | UInt64(mask & 0x01),
        )
        plateau_hash = _mix_hash(
            plateau_hash,
            (UInt64(node) << 4) | UInt64(mask >> 1),
        )
        frontier_hash = _mix_hash(
            frontier_hash,
            UInt64(node) | (UInt64(mask) << 32),
        )
    end
    @inbounds for record in 1:worker.tape.count
        TransitionPhase(worker.tape.phase[record]) == EVENT_WAVE || continue
        node = Int(worker.tape.node[record])
        mask = worker.tape.event_mask[record]
        wave = Int(worker.tape.wave[record])
        soma_hash = _mix_hash(
            soma_hash,
            (UInt64(node) << 1) | UInt64(mask & 0x01),
        )
        plateau_hash = _mix_hash(
            plateau_hash,
            (UInt64(node) << 4) | UInt64(mask >> 1),
        )
        frontier_hash = _mix_hash(
            frontier_hash,
            UInt64(node) | (UInt64(wave) << 24) | (UInt64(mask) << 32),
        )
    end
    frontier_hash = _mix_hash(frontier_hash, UInt64(report.visited_sources))
    frontier_hash = _mix_hash(frontier_hash, UInt64(report.delivered_edges))
    frontier_hash = _mix_hash(frontier_hash, UInt64(report.emitted_events))
    signature = TrajectorySignature(
        soma_hash,
        plateau_hash,
        frontier_hash,
        worker.event_delivery_hash,
        worker.event_delivery_count,
        worker.stats.common_transitions + worker.stats.common_event_transitions,
        report.waves_executed,
        report.terminated_empty,
        report.hit_wave_limit,
    )
    worker.tape.signature = signature
    state.common_signature = signature
    return signature
end

function _run_common_event_waves!(
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
)
    Events.begin_candidate!(worker.arena)
    @inbounds for node in 1:CORE_NODE_COUNT
        mask = state.common_seed_mask[node]
        iszero(mask) || Events.seed_event!(worker.arena, node, mask)
    end
    report = Events.run_event_waves!(
        worker.arena,
        model.cache.event_graph,
        _GraphEventStepper(model, state, worker, true);
        max_waves=model.config.max_event_waves,
    )
    report.fallback_requested && throw(OverflowError(
        "state-common event arena requested a dense fallback",
    ))
    report.hit_wave_limit && throw(ErrorException(
        "state-common static event frontier remained nonempty at the configured wave limit",
    ))
    worker.stats.common_event_waves = report.waves_executed

    # The refined state is the sole candidate-independent base. Untouched
    # nodes retain their mandatory common transition and seed mask; touched
    # nodes publish the latest event-wave state, packet, and emitted mask.
    @inbounds for active in 1:Events.active_count(worker.arena)
        node = Int(worker.arena.active_nodes[active])
        slot = Events.state_slot(worker.arena, node)
        copyto!(
            @view(state.common_state[:, node]),
            @view(worker.arena.state[slot, :]),
        )
        copyto!(
            @view(state.common_packet[:, node]),
            @view(worker.packet_by_slot[:, slot]),
        )
        state.common_event_mask[node] = _event_mask(
            @view(worker.event_by_slot[slot, :]),
        )
    end
    @inbounds for node in 1:CORE_NODE_COUNT, field in 1:Cell.STATE_DIM
        worker.arena.base_state[node, field] = state.common_state[field, node]
    end
    _finalize_common_signature!(state, worker, report)
    return report
end

"""
Compute both candidate-independent anatomical planes from B.  The AFTER
baseline is not copied from BEFORE: its cells have independently trainable
parameters, so it must be evaluated with AFTER-plane identity before COW.
Motif and evidence cells are candidate-defined and remain at their own rest
state until `forward_candidate!`.
"""
function prepare_state_common!(
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
    input,
)
    state.epoch == typemax(UInt64) && throw(OverflowError(
        "state-common epoch exhausted",
    ))
    state.prepared_revision == model.cache.revision ||
        refresh_state_initial!(model, state)
    state.ready = false
    state.fingerprint = _state_fingerprint(input)
    worker.candidate_count = 0
    _reset_stats!(worker.stats)
    _reset_tape!(worker.tape)
    _reset_provenance!(
        worker.provenance,
        model.cache.parameter_digest,
        state.fingerprint,
    )
    worker.event_delivery_hash = UInt64(0xcbf29ce484222325)
    worker.event_delivery_count = 0
    accessor = Spatial.BeforeSiteAccessor(input)
    spatial_and_spine_limit = Topology.SPATIAL_COUNT +
        Topology.ROW_INTERNAL_COUNT + Topology.COLUMN_INTERNAL_COUNT

    @inbounds for node in 1:spatial_and_spine_limit
        first_deposit = worker.provenance.analog_count + 1
        class = Topology.node_class(model.topology, node)
        if class == Topology.SPATIAL_CLASS
            plane = Topology.node_plane(model.topology, node)
            position = Int(Topology.spatial_position(node))
            row = mod(position - 1, Input.BOARD_ROWS) + 1
            column = div(position - 1, Input.BOARD_ROWS) + 1
            Spatial.fill_spatial_drive!(
                @view(state.common_input[:, node]),
                accessor,
                row,
                column,
                plane,
                plane == Topology.BEFORE_PLANE ?
                    COMMON_BEFORE : CANDIDATE_AFTER,
            )
        else
            # During common preparation no COW slot exists, so expose the
            # already-computed common packets through a tiny direct merge.
            fill!(@view(state.common_input[:, node]), 0.0f0)
            left = Int(Topology.child_node(model.topology, node, 1))
            right = Int(Topology.child_node(model.topology, node, 2))
            fill!(worker.context, 0.0f0)
            Axon.ordered_binary_deposit!(
                @view(state.common_input[:, node]),
                @view(state.common_packet[:, left]),
                @view(state.common_packet[:, right]),
                worker.context,
            )
            _append_analog_deposit!(
                worker.provenance,
                BINARY_PACKET_DEPOSIT,
                left,
                Int(worker.tape.latest_record[left]),
                @view(state.common_packet[:, left]),
                1,
                0,
                0,
            )
            _append_analog_deposit!(
                worker.provenance,
                BINARY_PACKET_DEPOSIT,
                right,
                Int(worker.tape.latest_record[right]),
                @view(state.common_packet[:, right]),
                1 + Axon.GROUP_COUNT,
                0,
                0,
            )
        end
        plane = Int(Topology.node_plane(model.topology, node))
        mask = _transition!(
            model,
            worker,
            node,
            @view(state.initial_core[:, node]),
            @view(state.common_input[:, node]),
            @view(state.common_state[:, node]),
            @view(state.common_packet[:, node]),
            worker.event_scratch,
            plane == Int(Topology.BEFORE_PLANE) ? COMMON_BEFORE : CANDIDATE_AFTER,
            0,
            true,
        )
        _bind_analog_deposits!(
            worker.provenance,
            first_deposit,
            worker.tape.count,
        )
        state.common_seed_mask[node] = mask
        state.common_event_mask[node] = mask
        worker.stats.hard_events += count_ones(mask)
        worker.stats.common_transitions += 1
    end

    # Candidate-defined cells have an explicit rest packet. They are always
    # in the mandatory candidate closure, so this is never used as a silent
    # replacement for candidate semantics.
    @inbounds for node in (spatial_and_spine_limit + 1):CORE_NODE_COUNT
        copyto!(@view(state.common_state[:, node]), @view(state.initial_core[:, node]))
        fill!(@view(state.common_input[:, node]), 0.0f0)
        state.common_seed_mask[node] = UInt8(0)
        state.common_event_mask[node] = UInt8(0)
        Axon.axon_packet!(
            @view(state.common_packet[:, node]),
            @view(state.initial_core[:, node]),
            @view(state.initial_core[:, node]),
            model.cache.core_cell[node],
        )
    end

    # EventArena uses node-major base storage.
    @inbounds for node in 1:CORE_NODE_COUNT, field in 1:Cell.STATE_DIM
        worker.arena.base_state[node, field] = state.common_state[field, node]
    end
    if model.config.max_event_waves > 0
        _run_common_event_waves!(model, state, worker)
    else
        _finalize_common_signature!(
            state,
            worker,
            Events.EventWaveReport(
                0, 0, 0, 0, 0, 0, true, false, false,
                Events.OVERFLOW_NONE,
            ),
        )
    end
    state.epoch += UInt64(1)
    _prepare_shared_state_value!(model, state, worker, input)
    state.ready = true
    worker.prepared_state_epoch = state.epoch
    worker.prepared_state_fingerprint = state.fingerprint
    worker.prepared_cache_revision = model.cache.revision
    return state
end

@inline function _push_seed!(worker::ModelWorker, node::Integer)
    count = worker.seed_count + 1
    count <= length(worker.seeds) || throw(OverflowError(
        "candidate seed capacity exceeded",
    ))
    worker.seeds[count] = UInt16(node)
    worker.seed_count = count
    return nothing
end

"""
Build the candidate-derived hard-event overlay from the same semantic
incidence used by the mandatory analog DAG.  External descriptors have no
upstream cell and therefore cannot emit events.  Every live spine contact
owns five single-bit physical edges but one stable trainable conductance,
indexed by `(motif, destination branch)` rather than incidence rank.
"""
function _prepare_candidate_event_overlay!(
    model::CanonicalModel,
    worker::ModelWorker,
)
    overlay = worker.dynamic_overlay
    Events.begin_dynamic_overlay!(overlay)
    if model.config.max_event_waves > 0
        @inbounds for motif in 1:Topology.MOTIF_COUNT
            destination = Int(Topology.motif_node(
                div(motif - 1, Topology.MOTIF_SLOTS_PER_FAMILY) + 1,
                mod(motif - 1, Topology.MOTIF_SLOTS_PER_FAMILY) + 1,
            ))
            source_count = Topology.motif_source_count(
                worker.motif_incidence,
                motif,
            )
            for rank in 1:source_count
                source = Topology.motif_source(
                    worker.motif_incidence,
                    motif,
                    rank,
                )
                Topology.motif_source_is_spine(source) || continue
                branch_slot = Int(source.branch_slot)
                raw_index = _dynamic_event_parameter_index(
                    model,
                    motif,
                    branch_slot,
                )
                base_channel = Cell.input_index(
                    branch_slot,
                    Cell.INPUT_AMPA,
                )
                for lane in 1:Axon.EVENT_DIM
                    Events.push_dynamic_edge!(
                        overlay,
                        Int(source.node),
                        destination,
                        base_channel,
                        UInt8(1 << (lane - 1)),
                        raw_index,
                    )
                end
            end
        end
    end
    Events.seal_dynamic_overlay!(overlay)
    return overlay
end

function _prepare_candidate_closure!(
    model::CanonicalModel,
    worker::ModelWorker,
    input,
)
    Input.derive_candidate!(worker.geometry, input)
    context = _motif_context(input, worker.geometry)
    Topology.fill_candidate_motif_incidence!(
        worker.motif_incidence,
        model.topology,
        context,
    )
    _prepare_candidate_event_overlay!(model, worker)
    worker.seed_count = 0

    if Input.requires_clear_slow_path(worker.geometry)
        worker.stats.clear_slow_paths += 1
        @inbounds for position in 1:Input.BOARD_CELLS
            _push_seed!(worker, Topology.spatial_node(
                Topology.AFTER_PLANE,
                position,
            ))
        end
    else
        # Every spatial cell reads a local 3x3 patch.  One changed placement
        # therefore invalidates all AFTER centres in its Chebyshev-1
        # neighbourhood, not merely the four placement centres themselves.
        @inbounds for dirty_index in 1:Input.no_clear_dirty_count(worker.geometry)
            position = Int(Input.no_clear_dirty_position(
                worker.geometry,
                dirty_index,
            ))
            row = mod(position - 1, Input.BOARD_ROWS) + 1
            column = div(position - 1, Input.BOARD_ROWS) + 1
            for delta_column in -1:1, delta_row in -1:1
                centre_row = row + delta_row
                centre_column = column + delta_column
                1 <= centre_row <= Input.BOARD_ROWS || continue
                1 <= centre_column <= Input.BOARD_COLUMNS || continue
                _push_seed!(worker, Topology.spatial_node(
                    Topology.AFTER_PLANE,
                    centre_row,
                    centre_column,
                ))
            end
        end
    end

    # Candidate descriptors (raw placement, exact clear/remap, queue, REN,
    # B2B and T-spin) make every motif receiver candidate-defined even when
    # its live spine source did not change.
    @inbounds for family in 1:Topology.MOTIF_FAMILY_COUNT,
                  slot in 1:Topology.MOTIF_SLOTS_PER_FAMILY
        _push_seed!(worker, Topology.motif_node(family, slot))
    end
    Topology.fill_incidence_affected_closure!(
        worker.closure,
        model.topology,
        worker.motif_incidence,
        worker.seeds,
        worker.seed_count,
    )
    return worker.closure
end

@inline function _logical_candidate_affected(worker::ModelWorker, node::Int)
    return !iszero(worker.closure.marked[node])
end

function _validate_event_delivery_provenance!(
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
    common_phase::Bool,
)
    provenance = worker.provenance
    tape = worker.tape
    first_dynamic = Events.edge_count(model.cache.event_graph) + 1
    last_dynamic = first_dynamic + dynamic_event_contact_count(model) - 1
    @inbounds for delivery in 1:provenance.event_count
        source = Int(provenance.event_source_node[delivery])
        source_record = Int(provenance.event_source_record[delivery])
        wave = Int(provenance.event_wave[delivery])
        1 <= source <= CORE_NODE_COUNT || throw(ArgumentError(
            "sealed event provenance source_node must identify a core node",
        ))
        candidate_affected = common_phase ? true :
            _logical_candidate_affected(worker, source)
        _validate_event_source_record(
            tape,
            source,
            source_record,
            common_phase,
            candidate_affected,
            wave,
            false,
        )
        if is_common_source_record(source, source_record)
            state.ready || throw(ArgumentError(
                "sealed candidate common-source provenance requires a prepared state",
            ))
            provenance.event_source_mask[delivery] ==
                state.common_event_mask[source] || throw(ArgumentError(
                    "sealed candidate common-source mask differs from finalized state-common",
                ))
            contact = Int(provenance.event_contact_parameter[delivery])
            first_dynamic <= contact <= last_dynamic || throw(ArgumentError(
                "COMMON_SOURCE_RECORD is restricted to dynamic-overlay contacts",
            ))
        end
    end
    return nothing
end

function _seal_replay_provenance!(
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
    signature::TrajectorySignature,
    common_phase::Bool,
)
    provenance = worker.provenance
    provenance.sealed && throw(ArgumentError(
        "replay provenance is already sealed",
    ))
    _validate_event_delivery_provenance!(
        model,
        state,
        worker,
        common_phase,
    )
    provenance.signature = signature
    provenance.sealed = true
    return provenance
end

"""
Return the hard-event mask that owns a source at the start of candidate wave one.

Candidate-affected sources use their mandatory candidate transition. Sources
outside the candidate closure retain the finalized state-common event mask.
The latter are eligible for dynamic-overlay delivery only; scanning their
static anatomy again would duplicate state-common work.
"""
@inline function wave_one_event_mask(
    state::ModelState,
    worker::ModelWorker,
    node::Integer,
)
    source = Int(node)
    1 <= source <= CORE_NODE_COUNT || throw(BoundsError(
        1:CORE_NODE_COUNT,
        source,
    ))
    record = Int(@inbounds worker.tape.mandatory_record[source])
    return is_common_source_record(source, record) ?
        @inbounds(state.common_event_mask[source]) :
        @inbounds(worker.tape.event_mask[record])
end

"""
Seed every live dynamic-overlay source that was not recomputed by the
candidate mandatory closure. The sealed overlay is source-major, so the
single pass is deterministic and duplicate contacts never duplicate a source
ledger entry. Static edges remain disabled for these wave-one sources.
"""
function _seed_overlay_only_dynamic_sources!(
    state::ModelState,
    worker::ModelWorker,
)
    overlay = worker.dynamic_overlay
    previous_source = 0
    @inbounds for edge in 1:Events.edge_count(overlay)
        source = Int(overlay.source[edge])
        source == previous_source && continue
        previous_source = source
        _logical_candidate_affected(worker, source) && continue
        mask = wave_one_event_mask(state, worker, source)
        iszero(mask) || Events.seed_overlay_only_event!(
            worker.arena,
            source,
            mask,
        )
    end
    return nothing
end

function _run_mandatory_node!(
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
    input,
    node::Int,
    seed_events::Bool,
)
    first_deposit = worker.provenance.analog_count + 1
    _fill_candidate_input!(
        worker.input_scratch,
        model,
        state,
        worker,
        input,
        node,
    )
    slot = Events.touch_node!(worker.arena, node)
    iszero(slot) && throw(OverflowError("candidate COW arena overflow"))
    previous = @view(state.initial_core[:, node])
    mask = _transition!(
        model,
        worker,
        node,
        previous,
        worker.input_scratch,
        worker.next_scratch,
        worker.packet_scratch,
        worker.event_scratch,
        MANDATORY_DAG,
        0,
        true,
    )
    if seed_events
        _bind_analog_deposits!(
            worker.provenance,
            first_deposit,
            worker.tape.count,
        )
    else
        _rollback_analog_deposits!(worker.provenance, first_deposit)
    end
    if seed_events
        copyto!(@view(worker.arena.state[slot, :]), worker.next_scratch)
        copyto!(@view(worker.packet_by_slot[:, slot]), worker.packet_scratch)
        copyto!(@view(worker.event_by_slot[slot, :]), worker.event_scratch)
    else
        # `:full` executes closure-unmarked nodes as a work/provenance oracle,
        # but their semantic version is the finalized event-refined common
        # base. Publishing the work-only mandatory transition would make its
        # descendants read a different world from COW. Restore the common
        # version in the slot after recording the independent transition.
        copyto!(
            @view(worker.arena.state[slot, :]),
            @view(state.common_state[:, node]),
        )
        copyto!(
            @view(worker.packet_by_slot[:, slot]),
            @view(state.common_packet[:, node]),
        )
        common_mask = state.common_event_mask[node]
        @inbounds for lane in 1:Axon.EVENT_DIM
            worker.event_by_slot[slot, lane] = UInt8(
                !iszero(common_mask & UInt8(1 << (lane - 1))),
            )
        end
    end
    worker.stats.mandatory_transitions += 1
    worker.stats.hard_events += count_ones(mask)
    seed_events && !iszero(mask) && Events.seed_event!(worker.arena, node, mask)
    return mask
end

struct _GraphEventStepper
    model::CanonicalModel
    state::ModelState
    worker::ModelWorker
    common_phase::Bool
end

@inline _GraphEventStepper(
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
) = _GraphEventStepper(model, state, worker, false)

@inline function _event_destination_branch(
    model::CanonicalModel,
    destination::Int,
    encoded_channel::Int,
    lane::Int,
)
    base_branch = div(encoded_channel - 1, Cell.INPUT_CHANNELS) + 1
    destination_class = Topology.node_class(model.topology, destination)
    if lane > Axon.SOMA_EVENT &&
       (destination_class == Topology.ROW_INTERNAL_CLASS ||
        destination_class == Topology.COLUMN_INTERNAL_CLASS)
        # A binary child owns a four-branch group:1:4 for left,5:8 for right.
        return base_branch + (lane - Axon.PLATEAU_EVENT_FIRST)
    end
    return base_branch
end

@inline function _source_plateau_group_active(
    source_state::AbstractVector{Float32},
    group::Int,
)
    first_branch = 2 * group - 1
    second_branch = first_branch + 1
    threshold = Float32(Axon.PLATEAU_EVENT_THRESHOLD)
    return @inbounds(
        source_state[Cell.state_index(
            first_branch,
            Cell.FIELD_PLATEAU,
        )] >= threshold ||
        source_state[Cell.state_index(
            second_branch,
            Cell.FIELD_PLATEAU,
        )] >= threshold
    )
end

@inline function _event_delivery_channel_and_scale(
    model::CanonicalModel,
    destination::Int,
    encoded_channel::Int,
    lane::Int,
    source_state::AbstractVector{Float32},
)
    branch = _event_destination_branch(
        model,
        destination,
        encoded_channel,
        lane,
    )
    if lane == Axon.SOMA_EVENT
        return Cell.input_index(branch, Cell.INPUT_AMPA), 1.0f0
    end
    group = lane - Axon.PLATEAU_EVENT_FIRST + 1
    receptor = _source_plateau_group_active(source_state, group) ?
        Cell.INPUT_NMDA : Cell.INPUT_GABA
    # Preserve plateau-group identity after local conductance summing. Every
    # subset of the four groups has a distinct fixed code.
    scale = Float32(1 << (group - 1)) * 0.125f0
    return Cell.input_index(branch, receptor), scale
end

@inline function _deliver_typed_event_from_state!(
    stepper::_GraphEventStepper,
    arena::Events.EventArena{Float32},
    source_state::AbstractVector{Float32},
    source_record::Int,
    candidate_affected::Bool,
    source::Int,
    source_mask::UInt8,
    destination::Int,
    slot::Int,
    encoded_channel::Int,
    trigger_mask::UInt8,
    weight::Float32,
    raw_index::Int,
    wave::Int,
)
    if is_common_source_record(source, source_record)
        stepper.state.ready || throw(ArgumentError(
            "candidate overlay common source requires a prepared state",
        ))
        source_mask == @inbounds(stepper.state.common_event_mask[source]) ||
            throw(ArgumentError(
                "candidate overlay source mask differs from finalized state-common",
            ))
    end
    @inbounds for lane in 1:Axon.EVENT_DIM
        bit = UInt8(1 << (lane - 1))
        iszero(source_mask & trigger_mask & bit) && continue
        channel, scale = _event_delivery_channel_and_scale(
            stepper.model,
            destination,
            encoded_channel,
            lane,
            source_state,
        )
        kind_index = event_kind_parameter_index(stepper.model, lane)
        kind_weight = stepper.model.cache.event_weight[kind_index]
        arena.inbox[slot, channel] += weight * kind_weight * scale
        _append_event_delivery!(
            stepper.worker.provenance,
            stepper.worker.tape,
            stepper.common_phase,
            candidate_affected,
            destination,
            source,
            source_record,
            source_mask,
            lane,
            channel,
            raw_index,
            kind_index,
            scale,
            wave,
        )
        # Exact replay identity includes the actual typed delivery, not just
        # the source event mask. Onset/offset can share a hard bit while
        # selecting NMDA versus GABA, so channel is a mandatory decision.
        hash = stepper.worker.event_delivery_hash
        hash = _mix_hash(hash, UInt64(wave))
        hash = _mix_hash(hash, UInt64(source))
        hash = _mix_hash(hash, UInt64(destination))
        hash = _mix_hash(hash, UInt64(raw_index))
        hash = _mix_hash(hash, UInt64(bit))
        hash = _mix_hash(hash, UInt64(channel))
        hash = _mix_hash(hash, UInt64(reinterpret(UInt32, scale)))
        stepper.worker.event_delivery_hash = hash
        stepper.worker.event_delivery_count += 1
    end
    return nothing
end

@inline function _deliver_typed_event!(
    stepper::_GraphEventStepper,
    arena::Events.EventArena{Float32},
    source::Int,
    source_mask::UInt8,
    destination::Int,
    slot::Int,
    encoded_channel::Int,
    trigger_mask::UInt8,
    weight::Float32,
    raw_index::Int,
    wave::Int,
)
    source_slot = Events.state_slot(arena, source)
    candidate_affected = stepper.common_phase ? true :
        _logical_candidate_affected(stepper.worker, source)
    # Keep column-major common-state and row-major COW-state views behind
    # separate typed calls. Joining those two SubArray layouts into one local
    # union forces a heap allocation on every typed delivery.
    if stepper.common_phase && iszero(source_slot)
        return _deliver_typed_event_from_state!(
            stepper, arena, @view(stepper.state.common_state[:, source]),
            Int(@inbounds stepper.worker.tape.latest_record[source]),
            candidate_affected,
            source, source_mask, destination, slot, encoded_channel,
            trigger_mask, weight, raw_index, wave,
        )
    elseif !stepper.common_phase && wave == 1 && !candidate_affected
        return _deliver_typed_event_from_state!(
            stepper, arena, @view(stepper.state.common_state[:, source]),
            Int(COMMON_SOURCE_RECORD),
            candidate_affected,
            source, source_mask, destination, slot, encoded_channel,
            trigger_mask, weight, raw_index, wave,
        )
    end
    iszero(source_slot) && error("event source has no current COW state")
    return _deliver_typed_event_from_state!(
        stepper, arena, @view(arena.state[source_slot, :]),
        Int(@inbounds stepper.worker.tape.latest_record[source]),
        candidate_affected,
        source, source_mask, destination, slot, encoded_channel,
        trigger_mask, weight, raw_index, wave,
    )
end

function Events.deliver_event_edge!(
    stepper::_GraphEventStepper,
    arena::Events.EventArena{Float32},
    graph::Events.SourceMajorAdjacency{Float32},
    source::Int,
    source_mask::UInt8,
    edge::Int,
    destination::Int,
    slot::Int,
    wave::Int,
)
    _deliver_typed_event!(
        stepper,
        arena,
        source,
        source_mask,
        destination,
        slot,
        Int(@inbounds(graph.channel[edge])),
        @inbounds(graph.trigger_mask[edge]),
        @inbounds(graph.weight[edge]),
        edge,
        wave,
    )
    return nothing
end

function Events.deliver_event_edge!(
    stepper::_GraphEventStepper,
    arena::Events.EventArena{Float32},
    overlay::Events.DynamicSourceMajorOverlay,
    source::Int,
    source_mask::UInt8,
    edge::Int,
    destination::Int,
    slot::Int,
    wave::Int,
)
    raw_index = Int(@inbounds overlay.raw_index[edge])
    _deliver_typed_event!(
        stepper,
        arena,
        source,
        source_mask,
        destination,
        slot,
        Int(@inbounds(overlay.channel[edge])),
        @inbounds(overlay.trigger_bit[edge]),
        @inbounds(stepper.model.cache.event_weight[raw_index]),
        raw_index,
        wave,
    )
    return nothing
end

function (stepper::_GraphEventStepper)(
    arena::Events.EventArena{Float32},
    node::Int,
    slot::Int,
    wave::Int,
)
    model = stepper.model
    worker = stepper.worker
    copyto!(worker.previous_scratch, @view(arena.state[slot, :]))
    copyto!(worker.input_scratch, @view(arena.inbox[slot, :]))
    mask = _transition!(
        model,
        worker,
        node,
        worker.previous_scratch,
        worker.input_scratch,
        worker.next_scratch,
        worker.packet_scratch,
        worker.event_scratch,
        EVENT_WAVE,
        wave,
        true,
    )
    _bind_event_deliveries!(worker.provenance, node, worker.tape.count)
    copyto!(@view(arena.state[slot, :]), worker.next_scratch)
    copyto!(@view(worker.packet_by_slot[:, slot]), worker.packet_scratch)
    copyto!(@view(worker.event_by_slot[slot, :]), worker.event_scratch)
    if stepper.common_phase
        worker.stats.common_event_transitions += 1
    else
        worker.stats.event_transitions += 1
    end
    worker.stats.hard_events += count_ones(mask)
    return mask
end

function _candidate_fingerprint(
    input,
    geometry::Input.CandidateGeometry,
)
    hash = _state_fingerprint(input)
    @inbounds for index in 1:Input.placement_count(input)
        hash = _mix_hash(hash, UInt64(Input.placement_position(input, index)))
    end
    hash = _mix_hash(hash, UInt64(Input.tspin_value(input)))
    hash = _mix_hash(hash, UInt64(Input.clear_count(geometry)))
    @inbounds for row in 1:Input.BOARD_ROWS
        hash = _mix_hash(hash, UInt64(Input.source_to_after(geometry, row)))
    end
    return hash
end

function _finalize_signature!(
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
    candidate_hash::UInt64,
    report::Events.EventWaveReport,
)
    soma_hash = UInt64(0xcbf29ce484222325)
    plateau_hash = UInt64(0xcbf29ce484222325)
    frontier_hash = candidate_hash
    logical_transitions = Output.OUTPUT_CELLS - 2
    @inbounds for record in 1:worker.tape.count
        node = Int(worker.tape.node[record])
        phase = TransitionPhase(worker.tape.phase[record])
        phase == MANDATORY_DAG &&
            !_logical_candidate_affected(worker, node) && continue
        mask = worker.tape.event_mask[record]
        logical_transitions += 1
        soma_hash = _mix_hash(
            soma_hash,
            (UInt64(node) << 1) | UInt64(mask & 0x01),
        )
        plateau_hash = _mix_hash(
            plateau_hash,
            (UInt64(node) << 4) | UInt64(mask >> 1),
        )
        decision = UInt64(node) |
            (UInt64(worker.tape.phase[record]) << 16) |
            (UInt64(worker.tape.wave[record]) << 24) |
            (UInt64(mask) << 32)
        frontier_hash = _mix_hash(frontier_hash, decision)
    end
    @inbounds for cell in 3:Output.OUTPUT_CELLS
        bit = worker.output_hard_event[cell] > 0.5f0 ? UInt64(1) : UInt64(0)
        soma_hash = _mix_hash(
            soma_hash,
            (UInt64(CORE_NODE_COUNT + cell) << 1) | bit,
        )
        frontier_hash = _mix_hash(
            frontier_hash,
            (UInt64(CORE_NODE_COUNT + cell) << 1) | bit,
        )
    end
    frontier_hash = _mix_hash(frontier_hash, UInt64(report.visited_sources))
    frontier_hash = _mix_hash(frontier_hash, UInt64(report.delivered_edges))
    frontier_hash = _mix_hash(frontier_hash, UInt64(report.emitted_events))
    signature = TrajectorySignature(
        soma_hash,
        plateau_hash,
        frontier_hash,
        worker.event_delivery_hash,
        worker.event_delivery_count,
        logical_transitions,
        report.waves_executed,
        report.terminated_empty,
        report.hit_wave_limit,
    )
    worker.tape.signature = signature
    return signature
end

"""
Run one real candidate through the 1,458-cell canonical graph.

`:cow` visits the exact candidate closure; `:full` independently recomputes
all 1,436 non-output cells from their anatomical initial state.  Both modes
seed optional event waves only from the logical candidate closure, so the full
path is a work-provenance oracle rather than a different dynamical system.
"""
function forward_candidate!(
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
    input;
    mode::Symbol=:cow,
)
    state.ready || throw(ArgumentError(
        "prepare_state_common! must run before candidate forward",
    ))
    _state_fingerprint(input) == state.fingerprint || throw(ArgumentError(
        "candidate does not belong to the prepared state-common input",
    ))
    mode in (:cow, :full) || throw(ArgumentError("mode must be :cow or :full"))
    worker.candidate_count < model.config.max_candidates || throw(OverflowError(
        "candidate output capacity exceeded",
    ))
    sync_state_common!(model, state, worker)
    _prepare_candidate_closure!(model, worker, input)
    Events.begin_candidate!(worker.arena)
    _reset_tape!(worker.tape)
    candidate_hash = _candidate_fingerprint(input, worker.geometry)
    _reset_provenance!(
        worker.provenance,
        model.cache.parameter_digest,
        candidate_hash,
    )
    worker.event_delivery_hash = UInt64(0xcbf29ce484222325)
    worker.event_delivery_count = 0
    worker.stats.mandatory_transitions = 0
    worker.stats.event_transitions = 0
    worker.stats.output_transitions = 0
    worker.stats.event_waves = 0
    worker.stats.hard_events = 0

    if mode === :full
        @inbounds for node in 1:CORE_NODE_COUNT
            _run_mandatory_node!(
                model,
                state,
                worker,
                input,
                node,
                _logical_candidate_affected(worker, node),
            )
        end
    else
        @inbounds for index in 1:Topology.affected_count(worker.closure)
            node = Int(Topology.affected_forward_node(worker.closure, index))
            node <= CORE_NODE_COUNT || continue
            _run_mandatory_node!(
                model,
                state,
                worker,
                input,
                node,
                true,
            )
        end
    end

    report = if model.config.max_event_waves > 0
        _seed_overlay_only_dynamic_sources!(state, worker)
        Events.run_event_waves!(
            worker.arena,
            model.cache.event_graph,
            worker.dynamic_overlay,
            _GraphEventStepper(model, state, worker);
            max_waves=model.config.max_event_waves,
        )
    else
        # max_event_waves=0 is the explicit mandatory-DAG ablation.
        Events.EventWaveReport(0, 0, 0, 0, 0, 0, true, false, false,
                               Events.OVERFLOW_NONE)
    end
    report.fallback_requested && throw(OverflowError(
        "event arena requested a dense fallback",
    ))
    worker.stats.event_waves = report.waves_executed

    _fill_evidence_for_output!(worker, model, state)
    worker.output_evidence_count[1] = UInt8(0)
    worker.output_evidence_count[2] = UInt8(0)
    candidate_slot = worker.candidate_count + 1
    component = worker.components[candidate_slot]
    Output.candidate_output_population_forward!(
        component,
        worker.output_hard_event,
        worker.output_tape,
        state.output_initial,
        worker.output_evidence,
        worker.output_evidence_count,
        model.parameters.output,
        model.cache.output,
    )
    component.value = state.state_value
    worker.stats.output_transitions = Output.OUTPUT_CELLS - 2
    worker.stats.hard_events += count(
        value -> value > 0.5f0,
        @view(worker.output_hard_event[3:Output.OUTPUT_CELLS]),
    )
    signature = _finalize_signature!(
        model,
        state,
        worker,
        candidate_hash,
        report,
    )
    _seal_replay_provenance!(model, state, worker, signature, false)
    worker.signatures[candidate_slot] = signature
    worker.advantages[candidate_slot] = component.advantage
    worker.candidate_count = candidate_slot
    return component, signature
end

@inline function _has_nonzero(values::AbstractVector{Float32})
    @inbounds for value in values
        iszero(value) || return true
    end
    return false
end

@inline function _mandatory_packet_record(worker::ModelWorker, node::Int)
    return Int(@inbounds worker.tape.mandatory_record[node])
end

@inline function _latest_packet_record(worker::ModelWorker, node::Int)
    return Int(@inbounds worker.tape.latest_record[node])
end

@inline function _packet_at_record(
    worker::ModelWorker,
    state::ModelState,
    node::Int,
    record::Int,
)
    return is_common_source_record(node, record) ?
        @view(state.common_packet[:, node]) :
        @view(worker.tape.packet[:, record])
end

@inline function _packet_bar_at_record(
    worker::ModelWorker,
    node::Int,
    record::Int,
)
    return is_common_source_record(node, record) ?
        @view(worker.core_packet_bar[:, node]) :
        @view(worker.record_packet_bar[:, record])
end

@inline function _mandatory_packet(
    worker::ModelWorker,
    state::ModelState,
    node::Int,
)
    return _packet_at_record(
        worker,
        state,
        node,
        _mandatory_packet_record(worker, node),
    )
end

@inline function _mandatory_packet_bar(worker::ModelWorker, node::Int)
    return _packet_bar_at_record(
        worker,
        node,
        _mandatory_packet_record(worker, node),
    )
end

@inline function _accumulate_latest_packet_bar!(
    worker::ModelWorker,
    node::Int,
    source_bar::AbstractVector{Float32},
)
    destination = _packet_bar_at_record(
        worker,
        node,
        _latest_packet_record(worker, node),
    )
    @inbounds @simd for lane in 1:Axon.PACKET_DIM
        destination[lane] += source_bar[lane]
    end
    return nothing
end

function _reverse_projected_source!(
    model::CanonicalModel,
    gradient::ModelGradient,
    packet_bar,
    packet::AbstractVector{Float32},
    input_bar::AbstractVector{Float32},
    branch::Int,
    role::Int,
    semantic_class::Int,
)
    @inbounds for receptor in 1:Cell.INPUT_CHANNELS
        local_bar = input_bar[Cell.input_index(branch, receptor)]
        for group in 1:Axon.GROUP_COUNT
            lane = Axon.packet_lane(group, receptor)
            coefficient = model.cache.semantic_projection[
                group, receptor, role, semantic_class,
            ]
            packet_bar === nothing || (packet_bar[lane] += local_bar * coefficient)
            gradient.semantic_projection_raw[
                group, receptor, role, semantic_class,
            ] += local_bar * packet[lane] *
                 model.cache.semantic_projection_derivative[
                     group, receptor, role, semantic_class,
                 ]
        end
    end
    return nothing
end

function _reverse_mandatory_input!(
    model::CanonicalModel,
    worker::ModelWorker,
    destination_record::Int,
    input_bar::AbstractVector{Float32},
)
    provenance = worker.provenance
    first, count = record_analog_deposit_range(
        provenance,
        destination_record,
    )
    count == 0 && return nothing
    kind = AnalogDepositKind(@inbounds provenance.analog_kind[first])
    if kind == BINARY_PACKET_DEPOSIT
        count == 2 || error("recorded binary deposit lost one child")
        left_source = Int(@inbounds provenance.analog_source_node[first])
        right_source = Int(@inbounds provenance.analog_source_node[first + 1])
        left_record = Int(@inbounds provenance.analog_source_record[first])
        right_record = Int(@inbounds provenance.analog_source_record[first + 1])
        fill!(worker.context, 0.0f0)
        Axon.ordered_binary_deposit_pullback!(
            _packet_bar_at_record(worker, left_source, left_record),
            _packet_bar_at_record(worker, right_source, right_record),
            worker.context,
            input_bar,
        )
        return nothing
    end
    kind == SEMANTIC_PACKET_DEPOSIT || error(
        "unknown recorded analog deposit kind",
    )
    @inbounds for deposit in first:(first + count - 1)
        source_node = Int(provenance.analog_source_node[deposit])
        source_record = Int(provenance.analog_source_record[deposit])
        packet_bar = source_node == 0 ? nothing :
            _packet_bar_at_record(worker, source_node, source_record)
        _reverse_projected_source!(
            model,
            worker.gradient,
            packet_bar,
            @view(provenance.analog_packet[:, deposit]),
            input_bar,
            Int(provenance.analog_branch[deposit]),
            Int(provenance.analog_semantic_role[deposit]),
            Int(provenance.analog_semantic_class[deposit]),
        )
    end
    return nothing
end

function _reverse_event_input!(
    model::CanonicalModel,
    worker::ModelWorker,
    destination_record::Int,
    input_bar::AbstractVector{Float32},
)
    provenance = worker.provenance
    delivery = record_event_delivery_head(provenance, destination_record)
    while delivery != 0
        @inbounds begin
            channel = Int(provenance.event_resolved_channel[delivery])
            scale = provenance.event_scale[delivery]
            raw_index = Int(provenance.event_contact_parameter[delivery])
            kind_index = Int(provenance.event_kind_parameter[delivery])
        end
        local_bar = @inbounds(input_bar[channel]) * scale
        contact_weight = @inbounds model.cache.event_weight[raw_index]
        kind_weight = @inbounds model.cache.event_weight[kind_index]
        @inbounds worker.gradient.event_raw[raw_index] +=
            local_bar * kind_weight *
            _softplus_derivative(model.parameters.event_raw[raw_index])
        @inbounds worker.gradient.event_raw[kind_index] +=
            local_bar * contact_weight *
            _softplus_derivative(model.parameters.event_raw[kind_index])
        delivery = next_event_delivery_record(provenance, delivery)
    end
    return nothing
end

function _scatter_local_input_parameters!(
    model::CanonicalModel,
    worker::ModelWorker,
    learner::ModelLocalLearner,
    record::Int,
    input_cotangent::AbstractVector{Float32},
)
    provenance = worker.provenance
    phase = TransitionPhase(@inbounds worker.tape.phase[record])
    if phase == EVENT_WAVE
        delivery = record_event_delivery_head(provenance, record)
        while delivery != 0
            @inbounds begin
                channel = Int(provenance.event_resolved_channel[delivery])
                scale = provenance.event_scale[delivery]
                raw_index = Int(provenance.event_contact_parameter[delivery])
                kind_index = Int(provenance.event_kind_parameter[delivery])
            end
            local_bar = @inbounds(input_cotangent[channel]) * scale
            if !iszero(local_bar)
                contact_weight = @inbounds model.cache.event_weight[raw_index]
                kind_weight = @inbounds model.cache.event_weight[kind_index]
                contact_contribution =
                    local_bar * kind_weight *
                    _softplus_derivative(model.parameters.event_raw[raw_index])
                kind_contribution =
                    local_bar * contact_weight *
                    _softplus_derivative(model.parameters.event_raw[kind_index])
                @inbounds worker.gradient.event_raw[raw_index] +=
                    contact_contribution
                @inbounds worker.gradient.event_raw[kind_index] +=
                    kind_contribution
                # Structural utility is anatomical-contact only. Shared kind
                # gains remain trainable but are explicitly non-rewirable.
                learner.counters.event_receiver_updates += 1
                if learner.signals.config.utility_mode === :combined
                    @inbounds learner.plasticity.task_utility_sum[raw_index] +=
                        abs(contact_contribution)
                    learner.counters.utility_updates += 1
                end
            end
            delivery = next_event_delivery_record(provenance, delivery)
        end
        return nothing
    end

    first, count = record_analog_deposit_range(provenance, record)
    count == 0 && return nothing
    @inbounds for deposit in first:(first + count - 1)
        AnalogDepositKind(provenance.analog_kind[deposit]) ==
            SEMANTIC_PACKET_DEPOSIT || continue
        branch = Int(provenance.analog_branch[deposit])
        role = Int(provenance.analog_semantic_role[deposit])
        semantic_class = Int(provenance.analog_semantic_class[deposit])
        changed = false
        for receptor in 1:Cell.INPUT_CHANNELS
            local_bar = input_cotangent[Cell.input_index(branch, receptor)]
            iszero(local_bar) && continue
            for group in 1:Axon.GROUP_COUNT
                lane = Axon.packet_lane(group, receptor)
                worker.gradient.semantic_projection_raw[
                    group, receptor, role, semantic_class,
                ] += local_bar * provenance.analog_packet[lane, deposit] *
                     model.cache.semantic_projection_derivative[
                         group, receptor, role, semantic_class,
                     ]
            end
            changed = true
        end
        changed && (learner.counters.semantic_parameter_updates += 1)
    end
    return nothing
end

function _collect_local_plasticity!(
    model::CanonicalModel,
    worker::ModelWorker,
    learner::ModelLocalLearner,
    output_tape::Output.OutputPopulationTape{Float32},
    first_output_cell::Int,
    last_output_cell::Int,
)
    observation = _reset_local_plasticity!(learner)
    maximum_visits = UInt8(1 + model.config.max_event_waves)
    @inbounds for node in 1:CORE_NODE_COUNT
        record = Int(worker.tape.latest_record[node])
        while record != 0
            visits = observation.visit_count[node]
            visits < maximum_visits || error(
                "logical cell visit count exceeds the configured event horizon",
            )
            observation.visit_count[node] = visits + UInt8(1)
            observation.spike_count[node] += UInt8(
                !iszero(worker.tape.event_mask[record] & UInt8(0x01)),
            )
            observation.activity_sum[node] += Cell.spike_surrogate_value(
                Cell.spike_margin_from_transition(
                    @view(worker.tape.previous_state[:, record]),
                    @view(worker.tape.next_state[:, record]),
                    model.cache.core_cell[node],
                ),
            )
            phase = TransitionPhase(worker.tape.phase[record])
            if phase in (COMMON_BEFORE, CANDIDATE_AFTER, MANDATORY_DAG) &&
               Topology.node_class(model.topology, node) == Topology.SPATIAL_CLASS
                observation.incoming_conductance_sum[node] +=
                    Float32(Cell.INPUT_DIM)
            end
            record = Int(worker.tape.previous_record[record])
        end
    end

    provenance = worker.provenance
    structural_contacts = length(observation.contact_activity_sum)
    @inbounds for deposit in 1:provenance.analog_count
        destination_record = Int(provenance.analog_destination_record[deposit])
        destination_node = Int(worker.tape.node[destination_record])
        kind = AnalogDepositKind(provenance.analog_kind[deposit])
        if kind == BINARY_PACKET_DEPOSIT
            # Ordered binary anatomy has one fixed unit conductance for every
            # packet group/receptor lane.
            observation.incoming_conductance_sum[destination_node] +=
                Float32(Axon.PACKET_DIM)
        else
            role = Int(provenance.analog_semantic_role[deposit])
            semantic_class = Int(provenance.analog_semantic_class[deposit])
            physical = 0.0f0
            for receptor in 1:Cell.INPUT_CHANNELS, group in 1:Axon.GROUP_COUNT
                physical += model.cache.semantic_projection[
                    group, receptor, role, semantic_class,
                ]
            end
            observation.incoming_conductance_sum[destination_node] += physical
        end
    end
    @inbounds for delivery in 1:provenance.event_count
        raw_index = Int(provenance.event_contact_parameter[delivery])
        1 <= raw_index <= structural_contacts || error(
            "event delivery references a non-anatomical contact parameter",
        )
        kind_index = Int(provenance.event_kind_parameter[delivery])
        delivered = abs(
            provenance.event_scale[delivery] *
            model.cache.event_weight[kind_index],
        )
        observation.contact_activity_sum[raw_index] += delivered
        destination_record = Int(provenance.event_destination_record[delivery])
        destination_node = Int(worker.tape.node[destination_record])
        observation.incoming_conductance_sum[destination_node] +=
            delivered * model.cache.event_weight[raw_index]
    end

    @inbounds for output_cell in first_output_cell:last_output_cell
        node = CORE_NODE_COUNT + output_cell
        observation.visit_count[node] = UInt8(1)
        observation.spike_count[node] = UInt8(
            !iszero(output_tape.event[output_cell]),
        )
        observation.activity_sum[node] =
            Cell.spike_surrogate_value(output_tape.margin[output_cell])
        observation.incoming_conductance_sum[node] = 0.0f0
    end
    @inbounds for binding in 1:provenance.output_count
        output_cell = Int(provenance.output_cell[binding])
        first_output_cell <= output_cell <= last_output_cell || continue
        role = Output.cell_role(output_cell)
        node = CORE_NODE_COUNT + output_cell
        for receptor in 1:Cell.INPUT_CHANNELS, group in 1:Axon.GROUP_COUNT
            observation.incoming_conductance_sum[node] +=
                model.cache.output.projection[group, receptor, role]
        end
    end
    return observation
end

function _contract_analog_local_records!(
    model::CanonicalModel,
    worker::ModelWorker,
    learner::ModelLocalLearner,
    raw_delta::AbstractVector{Float32},
)
    length(raw_delta) == Output.OUTPUT_DIM || throw(DimensionMismatch(
        "local raw derivative must have length 22",
    ))
    all(isfinite, raw_delta) || throw(ArgumentError(
        "local raw derivative must be finite",
    ))
    Local.reset_adjoint_arena!(learner.arena)
    visited_before = learner.counters.visited_transitions
    pullbacks_before = learner.counters.conditional_pullbacks
    @inbounds for node in 1:CORE_NODE_COUNT
        terminal_record = Int(worker.tape.latest_record[node])
        terminal_record == 0 && continue
        Local.project_learning_signal!(
            learner.continuous_signal,
            learner.signals.continuous[node],
            raw_delta,
        )
        Local.project_learning_signal!(
            learner.packet_signal,
            learner.signals.packet[node],
            raw_delta,
        )
        (_has_nonzero(learner.continuous_signal) ||
         _has_nonzero(learner.packet_signal)) &&
            (learner.counters.signal_nonzero += 1)
        Local.begin_local_adjoint!(learner.arena, node, terminal_record)
        record = terminal_record
        while record != 0
            predecessor = Int(worker.tape.previous_record[record])
            link = Local.ChronologicalTransitionLink(
                record,
                predecessor,
                record,
            )
            Local.contract_replayed_transition!(
                learner.arena,
                node,
                learner.scratch,
                link,
                @view(worker.tape.previous_state[:, record]),
                @view(worker.tape.input[:, record]),
                model.cache.core_cell[node],
                model.cache.core_derivative[node],
                @view(worker.tape.next_state[:, record]),
                learner.continuous_signal,
                learner.packet_signal;
                touched=true,
                eligibility_scale=learner.signals.config.analog_multiplier,
            )
            draw = Local.raw_parameter_cotangent(learner.scratch)
            for parameter in 1:Cell.PARAM_DIM
                worker.gradient.core_cell_raw[parameter, node] += draw[parameter]
            end
            _scatter_local_input_parameters!(
                model,
                worker,
                learner,
                record,
                Local.input_cotangent(learner.scratch),
            )
            learner.counters.visited_transitions += 1
            learner.counters.conditional_pullbacks += 1
            iszero(worker.tape.event_mask[record] & UInt8(0x01)) &&
                (learner.counters.nonspiking_transitions += 1)
            record = predecessor
        end
        Local.finish_local_adjoint!(learner.arena, node, 0)
    end
    learner.counters.visited_transitions - visited_before ==
        learner.counters.conditional_pullbacks - pullbacks_before || error(
            "local replay must execute one conditional pullback per transition",
        )
    return nothing
end

function _mark_logical_local_records!(
    worker::ModelWorker,
    learner::ModelLocalLearner,
)
    count = worker.tape.count
    fill!(@view(learner.logical_record[1:count]), false)
    @inbounds for node in 1:CORE_NODE_COUNT
        record = Int(worker.tape.latest_record[node])
        while record != 0
            learner.logical_record[record] = true
            record = Int(worker.tape.previous_record[record])
        end
    end
    return nothing
end

function _preflight_candidate_hard_seed_capacity!(
    worker::ModelWorker,
    learner::ModelLocalLearner,
)
    _mark_logical_local_records!(worker, learner)
    sentinel_delivery_count = 0
    provenance = worker.provenance
    @inbounds for delivery in 1:provenance.event_count
        destination_record = Int(provenance.event_destination_record[delivery])
        1 <= destination_record <= worker.tape.count || error(
            "hard-event delivery destination is outside the replay tape",
        )
        learner.logical_record[destination_record] || continue
        source_node = Int(provenance.event_source_node[delivery])
        source_record = Int(provenance.event_source_record[delivery])
        is_common_source_record(source_node, source_record) || continue
        sentinel_delivery_count += 1
    end
    sentinel_delivery_count <= MAX_COMMON_HARD_EVENT_SEEDS_PER_CANDIDATE || error(
        "candidate common-source deliveries exceed the anatomical upper bound",
    )
    learner.hard_seed_count + sentinel_delivery_count <=
        length(learner.hard_seed_state_id) || throw(OverflowError(
            "hard-event learner compact seed capacity exceeded before replay",
        ))
    return nothing
end

function _prepare_hard_event_source_fanout!(
    worker::ModelWorker,
    learner::ModelLocalLearner,
    allow_common_source::Bool,
)
    provenance = worker.provenance
    count = provenance.event_count
    count <= length(learner.event_delivery_advantage) || error(
        "hard-event delivery scratch is smaller than sealed provenance",
    )
    fill!(@view(learner.event_delivery_advantage[1:count]), 0.0f0)
    fill!(@view(learner.event_delivery_ready[1:count]), false)
    fill!(@view(learner.source_delivery_next[1:count]), Int32(0))
    fill!(
        @view(learner.source_delivery_head_by_record[1:worker.tape.count]),
        Int32(0),
    )
    fill!(
        @view(learner.source_delivery_tail_by_record[1:worker.tape.count]),
        Int32(0),
    )
    allow_common_source && fill!(learner.common_source_event_advantage, 0.0f0)

    @inbounds for delivery in 1:count
        Int(provenance.event_ordinal[delivery]) == delivery || error(
            "hard-event physical delivery ordinal changed before replay",
        )
        destination_record = Int(provenance.event_destination_record[delivery])
        1 <= destination_record <= worker.tape.count || error(
            "hard-event delivery destination is outside the replay tape",
        )
        learner.logical_record[destination_record] || continue
        TransitionPhase(worker.tape.phase[destination_record]) == EVENT_WAVE || error(
            "hard-event delivery destination is not an event-wave transition",
        )
        Int(provenance.event_wave[delivery]) ==
            Int(worker.tape.wave[destination_record]) || error(
                "hard-event delivery wave differs from its destination transition",
            )
        source_node = Int(provenance.event_source_node[delivery])
        source_record = Int(provenance.event_source_record[delivery])
        lane = Int(provenance.event_lane[delivery])
        1 <= lane <= Axon.EVENT_DIM || error(
            "hard-event delivery lane is outside the typed event ABI",
        )
        if is_common_source_record(source_node, source_record)
            allow_common_source || error(
                "state-common replay contains a candidate common-source sentinel",
            )
            Int(provenance.event_wave[delivery]) == 1 || error(
                "common-source sentinel is valid only in candidate wave one",
            )
            continue
        end
        1 <= source_record < destination_record || error(
            "hard-event source is not chronologically earlier than its destination",
        )
        learner.logical_record[source_record] || error(
            "hard-event source transition is outside the logical replay world",
        )
        Int(worker.tape.node[source_record]) == source_node || error(
            "hard-event source record/node identity changed before replay",
        )
        provenance.event_source_mask[delivery] ==
            worker.tape.event_mask[source_record] || error(
                "hard-event source mask changed before replay",
            )
        delivery_wave = Int(provenance.event_wave[delivery])
        source_wave = Int(worker.tape.wave[source_record])
        if delivery_wave == 1
            source_wave == 0 || error(
                "wave-one hard-event source is not a mandatory transition",
            )
        else
            TransitionPhase(worker.tape.phase[source_record]) == EVENT_WAVE &&
                source_wave == delivery_wave - 1 || error(
                    "hard-event source transition is not from the preceding wave",
                )
        end
        previous = Int(learner.source_delivery_tail_by_record[source_record])
        if previous == 0
            learner.source_delivery_head_by_record[source_record] = Int32(delivery)
        else
            learner.source_delivery_next[previous] = Int32(delivery)
        end
        learner.source_delivery_tail_by_record[source_record] = Int32(delivery)
    end
    return nothing
end

@inline function _gather_hard_event_source_advantages!(
    worker::ModelWorker,
    learner::ModelLocalLearner,
    source_record::Int,
)
    provenance = worker.provenance
    delivery = Int(@inbounds learner.source_delivery_head_by_record[source_record])
    while delivery != 0
        @inbounds learner.event_delivery_ready[delivery] || error(
            "hard-event destination advantage was not ready before source replay",
        )
        lane = Int(@inbounds provenance.event_lane[delivery])
        @inbounds learner.source_event_advantage[lane, source_record] +=
            learner.event_delivery_advantage[delivery]
        delivery = Int(@inbounds learner.source_delivery_next[delivery])
    end
    return nothing
end

function _publish_candidate_common_source_deliveries!(
    worker::ModelWorker,
    learner::ModelLocalLearner,
    state_id::Int,
    candidate_ordinal::Int,
)
    provenance = worker.provenance
    appended = 0
    @inbounds for delivery in 1:provenance.event_count
        destination_record = Int(provenance.event_destination_record[delivery])
        learner.logical_record[destination_record] || continue
        source_node = Int(provenance.event_source_node[delivery])
        source_record = Int(provenance.event_source_record[delivery])
        is_common_source_record(source_node, source_record) || continue
        learner.event_delivery_ready[delivery] || error(
            "common-source delivery advantage was not evaluated",
        )
        lane = Int(provenance.event_lane[delivery])
        slot = learner.hard_seed_count + 1
        slot <= length(learner.hard_seed_state_id) || error(
            "hard-event learner seed preflight was inconsistent",
        )
        learner.hard_seed_state_id[slot] = UInt16(state_id)
        learner.hard_seed_candidate_ordinal[slot] = Int32(candidate_ordinal)
        learner.hard_seed_delivery_ordinal[slot] =
            provenance.event_ordinal[delivery]
        learner.hard_seed_source_node[slot] = UInt16(source_node)
        learner.hard_seed_lane[slot] = UInt8(lane)
        learner.hard_seed_advantage[slot] =
            learner.event_delivery_advantage[delivery]
        learner.hard_seed_count = slot
        learner.hard_seed_identity_digest = _hard_seed_identity_digest(
            learner.hard_seed_identity_digest,
            state_id,
            candidate_ordinal,
            delivery,
            source_node,
            lane,
            learner.event_delivery_advantage[delivery],
        )
        appended += 1
    end
    appended <= MAX_COMMON_HARD_EVENT_SEEDS_PER_CANDIDATE || error(
        "candidate sentinel coverage exceeds its anatomical bound",
    )
    learner.hard_expected_seed_count += appended
    learner.hard_expected_seed_count == learner.hard_seed_count || error(
        "hard-event candidate sentinel coverage was not published exactly once",
    )
    return nothing
end

@inline function _event_control_advantages(
    learner::ModelLocalLearner,
    record::Int,
)
    return ntuple(
        lane -> @inbounds(learner.source_event_advantage[lane, record]),
        Val(Axon.EVENT_DIM),
    )
end

function _return_event_source_advantages!(
    model::CanonicalModel,
    worker::ModelWorker,
    learner::ModelLocalLearner,
    destination_record::Int,
    allow_common_source::Bool,
)
    provenance = worker.provenance
    delivery = record_event_delivery_head(provenance, destination_record)
    while delivery != 0
        @inbounds begin
            source_node = Int(provenance.event_source_node[delivery])
            source_record = Int(provenance.event_source_record[delivery])
            source_mask = provenance.event_source_mask[delivery]
            lane = Int(provenance.event_lane[delivery])
            channel = Int(provenance.event_resolved_channel[delivery])
            raw_index = Int(provenance.event_contact_parameter[delivery])
            kind_index = Int(provenance.event_kind_parameter[delivery])
            scale = provenance.event_scale[delivery]
        end
        bit = UInt8(1) << (lane - 1)
        !iszero(source_mask & bit) || error(
            "delivered event lane is absent from its source mask",
        )
        amplitude = scale * model.cache.event_weight[raw_index] *
            model.cache.event_weight[kind_index]
        input_bar = @inbounds learner.receiver_input_cotangent[
            channel,
            destination_record,
        ]
        advantage = muladd(
            input_bar,
            amplitude,
            learner.signals.config.hard_event_energy_cost,
        )
        if is_common_source_record(source_node, source_record)
            allow_common_source || error(
                "state-common replay cannot contain a common-source sentinel",
            )
            learner.counters.event_control_common_seeds += 1
        end
        @inbounds learner.event_delivery_ready[delivery] && error(
            "hard-event delivery advantage was evaluated twice",
        )
        @inbounds learner.event_delivery_advantage[delivery] = advantage
        @inbounds learner.event_delivery_ready[delivery] = true

        # The continuation component also differentiates the evaluated
        # outgoing delivery's own contact and shared kind gain. The source
        # presence cost belongs only to the source transition surrogate.
        direct_bar = input_bar * scale *
            learner.signals.config.hard_event_multiplier
        if !iszero(direct_bar)
            contact_weight = @inbounds model.cache.event_weight[raw_index]
            kind_weight = @inbounds model.cache.event_weight[kind_index]
            contact_contribution = direct_bar * kind_weight *
                _softplus_derivative(model.parameters.event_raw[raw_index])
            kind_contribution = direct_bar * contact_weight *
                _softplus_derivative(model.parameters.event_raw[kind_index])
            @inbounds learner.hard_gradient.event_raw[raw_index] +=
                contact_contribution
            @inbounds learner.hard_gradient.event_raw[kind_index] +=
                kind_contribution
            @inbounds learner.delivery_event_gradient[raw_index] +=
                contact_contribution
            @inbounds learner.delivery_event_gradient[kind_index] +=
                kind_contribution
            learner.counters.event_control_event_parameter_updates += 1
        end
        learner.counters.event_control_deliveries += 1
        delivery = next_event_delivery_record(provenance, delivery)
    end
    return nothing
end

function _seed_common_source_advantages!(
    worker::ModelWorker,
    learner::ModelLocalLearner,
)
    learner.common_source_seed_loaded || error(
        "state-common hard replay requires an explicitly loaded seed slab",
    )
    learner.common_source_seed_consumed && error(
        "state-common hard-event seed slab was already consumed",
    )
    learner.common_source_seed_in_progress || error(
        "state-common hard-event seed slab is not in its consuming phase",
    )
    @inbounds for node in 1:CORE_NODE_COUNT
        record = Int(worker.tape.latest_record[node])
        for lane in 1:Axon.EVENT_DIM
            advantage = learner.common_source_event_advantage[lane, node]
            iszero(advantage) && continue
            record > 0 || error(
                "common hard-event source has no replay transition",
            )
            !iszero(worker.tape.event_mask[record] &
                (UInt8(1) << (lane - 1))) || error(
                "aggregated common hard-event lane changed during replay",
            )
            learner.source_event_advantage[lane, record] += advantage
        end
    end
    return nothing
end

function _scatter_fused_analog_inputs!(
    model::CanonicalModel,
    worker::ModelWorker,
    learner::ModelLocalLearner,
)
    @inbounds for node in 1:CORE_NODE_COUNT
        record = Int(worker.tape.latest_record[node])
        while record != 0
            for channel in 1:Cell.INPUT_DIM
                learner.scratch.dinput[channel] =
                    learner.signals.config.analog_multiplier *
                    learner.receiver_input_cotangent[channel, record]
            end
            _scatter_local_input_parameters!(
                model,
                worker,
                learner,
                record,
                learner.scratch.dinput,
            )
            record = Int(worker.tape.previous_record[record])
        end
    end
    return nothing
end


function _scatter_hard_source_input_parameters!(
    model::CanonicalModel,
    worker::ModelWorker,
    learner::ModelLocalLearner,
    record::Int,
)
    provenance = worker.provenance
    phase = TransitionPhase(@inbounds worker.tape.phase[record])
    if phase == EVENT_WAVE
        delivery = record_event_delivery_head(provenance, record)
        while delivery != 0
            @inbounds begin
                channel = Int(provenance.event_resolved_channel[delivery])
                scale = provenance.event_scale[delivery]
                raw_index = Int(provenance.event_contact_parameter[delivery])
                kind_index = Int(provenance.event_kind_parameter[delivery])
            end
            local_bar = Local.fused_event_input_cotangent(
                learner.fused_scratch,
                channel,
            ) * learner.signals.config.hard_event_multiplier * scale
            if !iszero(local_bar)
                contact_weight = @inbounds model.cache.event_weight[raw_index]
                kind_weight = @inbounds model.cache.event_weight[kind_index]
                @inbounds learner.hard_gradient.event_raw[raw_index] +=
                    local_bar * kind_weight *
                    _softplus_derivative(model.parameters.event_raw[raw_index])
                @inbounds learner.hard_gradient.event_raw[kind_index] +=
                    local_bar * contact_weight *
                    _softplus_derivative(model.parameters.event_raw[kind_index])
                learner.counters.event_control_event_parameter_updates += 1
            end
            delivery = next_event_delivery_record(provenance, delivery)
        end
        return nothing
    end

    first, count = record_analog_deposit_range(provenance, record)
    count == 0 && return nothing
    @inbounds for deposit in first:(first + count - 1)
        AnalogDepositKind(provenance.analog_kind[deposit]) ==
            SEMANTIC_PACKET_DEPOSIT || continue
        branch = Int(provenance.analog_branch[deposit])
        role = Int(provenance.analog_semantic_role[deposit])
        semantic_class = Int(provenance.analog_semantic_class[deposit])
        changed = false
        for receptor in 1:Cell.INPUT_CHANNELS
            local_bar = Local.fused_event_input_cotangent(
                learner.fused_scratch,
                Cell.input_index(branch, receptor),
            ) * learner.signals.config.hard_event_multiplier
            iszero(local_bar) && continue
            for group in 1:Axon.GROUP_COUNT
                lane = Axon.packet_lane(group, receptor)
                learner.hard_gradient.semantic_projection_raw[
                    group,
                    receptor,
                    role,
                    semantic_class,
                ] += local_bar * provenance.analog_packet[lane, deposit] *
                     model.cache.semantic_projection_derivative[
                         group,
                         receptor,
                         role,
                         semantic_class,
                     ]
            end
            changed = true
        end
        changed && (learner.counters.event_control_semantic_updates += 1)
    end
    return nothing
end

function _contract_fused_local_records!(
    model::CanonicalModel,
    worker::ModelWorker,
    learner::ModelLocalLearner,
    raw_delta::AbstractVector{Float32};
    write_analog::Bool,
    allow_common_source::Bool,
    consume_common_source::Bool,
    hard_state_id::Int,
    hard_candidate_ordinal::Int,
)
    length(raw_delta) == Output.OUTPUT_DIM || throw(DimensionMismatch(
        "local raw derivative must have length 22",
    ))
    all(isfinite, raw_delta) || throw(ArgumentError(
        "local raw derivative must be finite",
    ))
    config = learner.signals.config
    config.hard_event_multiplier > 0.0f0 || error(
        "fused hard-event replay requires a positive multiplier",
    )
    Local.reset_adjoint_arena!(learner.fused_arena)
    fill!(learner.source_event_advantage, 0.0f0)
    @views fill!(
        learner.receiver_input_cotangent[:, 1:worker.tape.count],
        0.0f0,
    )
    _mark_logical_local_records!(worker, learner)
    _prepare_hard_event_source_fanout!(
        worker,
        learner,
        allow_common_source,
    )
    visited_before = learner.counters.visited_transitions
    pullbacks_before = learner.counters.conditional_pullbacks
    @inbounds for node in 1:CORE_NODE_COUNT
        terminal_record = Int(worker.tape.latest_record[node])
        terminal_record == 0 && continue
        continuous = @view learner.continuous_signal_by_node[:, node]
        packet = @view learner.packet_signal_by_node[:, node]
        Local.project_learning_signal!(
            continuous,
            learner.signals.continuous[node],
            raw_delta,
        )
        Local.project_learning_signal!(
            packet,
            learner.signals.packet[node],
            raw_delta,
        )
        (_has_nonzero(continuous) || _has_nonzero(packet)) &&
            (learner.counters.signal_nonzero += 1)
        Local.begin_fused_local_adjoint!(
            learner.fused_arena,
            node,
            terminal_record,
        )
    end
    consume_common_source && _seed_common_source_advantages!(worker, learner)

    @inbounds for record in worker.tape.count:-1:1
        learner.logical_record[record] || continue
        _gather_hard_event_source_advantages!(worker, learner, record)
        node = Int(worker.tape.node[record])
        predecessor = Int(worker.tape.previous_record[record])
        mask = worker.tape.event_mask[record]
        advantages = _event_control_advantages(learner, record)
        has_advantage = any(!iszero, advantages)
        control = Local.CausalEventControl(
            mask,
            advantages;
            connected=!iszero(mask),
        )
        Local.contract_replayed_transition_fused!(
            learner.fused_arena,
            node,
            learner.fused_scratch,
            Local.ChronologicalTransitionLink(record, predecessor, record),
            @view(worker.tape.previous_state[:, record]),
            @view(worker.tape.input[:, record]),
            model.cache.core_cell[node],
            model.cache.core_derivative[node],
            @view(worker.tape.next_state[:, record]),
            @view(learner.continuous_signal_by_node[:, node]),
            @view(learner.packet_signal_by_node[:, node]);
            touched=true,
            # The receiver lane is the unscaled local objective. Its dinput
            # defines a_e independently of the analog optimizer multiplier.
            analog_scale=1.0f0,
            # The event lane is the unscaled source eligibility. Its own
            # optimizer multiplier is applied only when scattering the three
            # hard-gradient parameter groups.
            event_scale=1.0f0,
            event_control=control,
        )
        for parameter in 1:Cell.PARAM_DIM
            write_analog && (@inbounds worker.gradient.core_cell_raw[
                parameter,
                node,
            ] += config.analog_multiplier * Local.fused_analog_raw_cotangent(
                learner.fused_scratch,
                parameter,
            ))
            @inbounds learner.hard_gradient.core_cell_raw[
                parameter,
                node,
            ] += config.hard_event_multiplier * Local.fused_event_raw_cotangent(
                learner.fused_scratch,
                parameter,
            )
        end
        _scatter_hard_source_input_parameters!(
            model,
            worker,
            learner,
            record,
        )
        for channel in 1:Cell.INPUT_DIM
            @inbounds learner.receiver_input_cotangent[channel, record] =
                Local.fused_analog_input_cotangent(
                    learner.fused_scratch,
                    channel,
                )
        end
        phase = TransitionPhase(worker.tape.phase[record])
        phase == EVENT_WAVE && _return_event_source_advantages!(
            model,
            worker,
            learner,
            record,
            allow_common_source,
        )
        if has_advantage
            learner.counters.event_control_source_transitions += 1
            !iszero(advantages[Axon.SOMA_EVENT]) &&
                (learner.counters.event_control_soma_sources += 1)
            for lane in Axon.PLATEAU_EVENT_FIRST:Axon.EVENT_DIM
                !iszero(advantages[lane]) &&
                    (learner.counters.event_control_plateau_sources += 1)
            end
        end
        learner.counters.visited_transitions += 1
        learner.counters.conditional_pullbacks += 1
        iszero(mask & UInt8(0x01)) &&
            (learner.counters.nonspiking_transitions += 1)
    end
    @inbounds for delivery in 1:worker.provenance.event_count
        destination_record = Int(
            worker.provenance.event_destination_record[delivery],
        )
        learner.logical_record[destination_record] || continue
        learner.event_delivery_ready[delivery] || error(
            "logical hard-event delivery was not evaluated exactly once",
        )
    end
    allow_common_source && _publish_candidate_common_source_deliveries!(
        worker,
        learner,
        hard_state_id,
        hard_candidate_ordinal,
    )
    write_analog && _scatter_fused_analog_inputs!(model, worker, learner)
    @inbounds for node in 1:CORE_NODE_COUNT
        Int(worker.tape.latest_record[node]) == 0 && continue
        Local.finish_fused_local_adjoint!(learner.fused_arena, node, 0)
    end
    learner.counters.visited_transitions - visited_before ==
        learner.counters.conditional_pullbacks - pullbacks_before || error(
            "fused replay must execute one conditional pullback per transition",
        )
    if consume_common_source
        fill!(learner.common_source_event_advantage, 0.0f0)
        learner.common_source_seed_loaded = false
        learner.common_source_seed_consumed = true
        learner.common_source_seed_in_progress = false
    end
    return nothing
end

function _contract_due_local_records!(
    model::CanonicalModel,
    worker::ModelWorker,
    learner::ModelLocalLearner,
    raw_delta::AbstractVector{Float32},
    due::Local.DuePlasticityClocks;
    allow_common_source::Bool,
    consume_common_source::Bool=false,
    hard_state_id::Int=0,
    hard_candidate_ordinal::Int=0,
)
    hard_due = due.hard_event &&
        learner.signals.config.hard_event_multiplier > 0.0f0
    if hard_due
        _contract_fused_local_records!(
            model,
            worker,
            learner,
            raw_delta;
            write_analog=due.analog,
            allow_common_source,
            consume_common_source,
            hard_state_id,
            hard_candidate_ordinal,
        )
    elseif due.analog
        _contract_analog_local_records!(model, worker, learner, raw_delta)
    end
    return nothing
end

@inline function _local_report(
    signature::TrajectorySignature,
    counters::LocalReplayCounters,
    before::NTuple{15,Int},
)
    return LocalReplayReport(
        signature,
        counters.visited_transitions - before[1],
        counters.conditional_pullbacks - before[2],
        counters.signal_nonzero - before[3],
        counters.nonspiking_transitions - before[4],
        counters.semantic_parameter_updates - before[5],
        counters.event_receiver_updates - before[6],
        counters.utility_updates - before[7],
        counters.output_replays - before[8],
        counters.event_control_deliveries - before[9],
        counters.event_control_source_transitions - before[10],
        counters.event_control_soma_sources - before[11],
        counters.event_control_plateau_sources - before[12],
        counters.event_control_common_seeds - before[13],
        counters.event_control_semantic_updates - before[14],
        counters.event_control_event_parameter_updates - before[15],
    )
end

@inline function _local_counter_snapshot(counters::LocalReplayCounters)
    return (
        counters.visited_transitions,
        counters.conditional_pullbacks,
        counters.signal_nonzero,
        counters.nonspiking_transitions,
        counters.semantic_parameter_updates,
        counters.event_receiver_updates,
        counters.utility_updates,
        counters.output_replays,
        counters.event_control_deliveries,
        counters.event_control_source_transitions,
        counters.event_control_soma_sources,
        counters.event_control_plateau_sources,
        counters.event_control_common_seeds,
        counters.event_control_semantic_updates,
        counters.event_control_event_parameter_updates,
    )
end

function _replay_candidate_world!(
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
    input,
    expected_signature::Union{Nothing,TrajectorySignature},
    mode::Symbol,
)
    saved_candidate_count = worker.candidate_count
    restore_slot = saved_candidate_count > 0
    saved_component = worker.components[1]
    saved_value = saved_component.value
    saved_advantage = saved_component.advantage
    saved_death = saved_component.death
    saved_geometry = ntuple(
        index -> saved_component.geometry[index],
        Val(Output.GEOMETRY_COUNT),
    )
    saved_uncertainty = saved_component.uncertainty_raw
    saved_signature = worker.signatures[1]
    saved_worker_advantage = worker.advantages[1]
    replay_signature = try
        worker.candidate_count = 0
        _, signature = forward_candidate!(model, state, worker, input; mode=mode)
        signature
    finally
        worker.candidate_count = saved_candidate_count
        if restore_slot
            saved_component.value = saved_value
            saved_component.advantage = saved_advantage
            saved_component.death = saved_death
            @inbounds for index in 1:Output.GEOMETRY_COUNT
                saved_component.geometry[index] = saved_geometry[index]
            end
            saved_component.uncertainty_raw = saved_uncertainty
            worker.signatures[1] = saved_signature
            worker.advantages[1] = saved_worker_advantage
        end
    end
    expected_signature === nothing || replay_signature == expected_signature ||
        throw(ArgumentError("candidate replay hard trajectory changed"))
    worker.provenance.sealed || error("candidate replay provenance was not sealed")
    worker.provenance.parameter_digest == model.cache.parameter_digest ||
        throw(ArgumentError("candidate replay parameter snapshot changed"))
    return replay_signature
end

"""Exact continuous reverse conditional on the replayed hard trajectory."""
function conditional_reverse_candidate!(
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
    input,
    component_bar::Output.OutputComponentGradient{Float32};
    expected_signature::Union{Nothing,TrajectorySignature}=nothing,
    mode::Symbol=:cow,
)
    iszero(component_bar.value) || throw(ArgumentError(
        "candidate reverse must not credit shared V(s); use state-value reverse once",
    ))
    _replay_candidate_world!(
        model, state, worker, input, expected_signature, mode,
    )
    fill!(worker.core_packet_bar, 0.0f0)
    fill!(worker.core_state_bar, 0.0f0)
    @views fill!(worker.record_packet_bar[:, 1:worker.tape.count], 0.0f0)
    @views fill!(worker.record_state_bar[:, 1:worker.tape.count], 0.0f0)

    Output.candidate_output_population_pullback!(
        worker.output_dbase,
        worker.output_devidence,
        worker.gradient.output,
        worker.output_scratch,
        worker.output_tape,
        model.parameters.output,
        model.cache.output,
        component_bar,
    )
    Output.candidate_output_initial_state_pullback!(
        worker.gradient.output,
        worker.output_scratch,
        worker.output_dbase,
        model.cache.output,
    )
    provenance = worker.provenance
    @inbounds for binding in 1:provenance.output_count
        output_cell = Int(provenance.output_cell[binding])
        output_cell >= 3 || continue
        rank = Int(provenance.output_rank[binding])
        source = Int(provenance.output_source_node[binding])
        source_record = Int(provenance.output_source_record[binding])
        destination = _packet_bar_at_record(worker, source, source_record)
        for lane in 1:Axon.PACKET_DIM
            destination[lane] += worker.output_devidence[lane, rank, output_cell]
        end
    end

    @inbounds for record in worker.tape.count:-1:1
        node = Int(worker.tape.node[record])
        phase = TransitionPhase(worker.tape.phase[record])
        phase == MANDATORY_DAG &&
            !_logical_candidate_affected(worker, node) && continue
        packet_bar = @view(worker.record_packet_bar[:, record])
        margin_bar = Axon.axon_packet_pullback!(
            worker.dnext_scratch,
            packet_bar,
            @view(worker.tape.previous_state[:, record]),
            @view(worker.tape.next_state[:, record]),
            model.cache.core_cell[node],
        )
        state_bar = @view(worker.record_state_bar[:, record])
        @inbounds for field in 1:Cell.STATE_DIM
            worker.dnext_scratch[field] += state_bar[field]
        end
        Cell.cell_step_conditional_pullback!(
            worker.dstate_scratch,
            worker.dinput_scratch,
            worker.draw_scratch,
            @view(worker.tape.previous_state[:, record]),
            @view(worker.tape.input[:, record]),
            model.cache.core_cell[node],
            model.cache.core_derivative[node],
            @view(worker.tape.next_state[:, record]),
            worker.dnext_scratch,
            0.0f0,
            0.0f0,
            margin_bar,
        )
        @views worker.gradient.core_cell_raw[:, node] .+= worker.draw_scratch
        if phase == EVENT_WAVE
            previous_record = Int(worker.tape.previous_record[record])
            previous_bar = previous_record == 0 ?
                @view(worker.core_state_bar[:, node]) :
                @view(worker.record_state_bar[:, previous_record])
            @inbounds @simd for field in 1:Cell.STATE_DIM
                previous_bar[field] += worker.dstate_scratch[field]
            end
            _reverse_event_input!(
                model,
                worker,
                record,
                worker.dinput_scratch,
            )
        else
            _reverse_mandatory_input!(
                model,
                worker,
                record,
                worker.dinput_scratch,
            )
            fill!(worker.draw_scratch, 0.0f0)
            Cell.initial_state_pullback!(
                worker.draw_scratch,
                worker.dstate_scratch,
                model.cache.core_derivative[node],
            )
            @views worker.gradient.core_cell_raw[:, node] .+= worker.draw_scratch
        end
    end
    return worker.gradient
end

"""
Replay one candidate with seed-fixed teacher-free local adjoints.

`raw_delta` drives only the two fixed feedback blocks. `component_bar` is the
exact private output-population VJP; its returned evidence cotangent is not
propagated into the recurrent graph. The candidate anatomical/common root is
a stop-gradient boundary.
"""
function local_replay_candidate!(
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
    learner::ModelLocalLearner,
    input,
    raw_delta::AbstractVector{Float32},
    component_bar::Output.OutputComponentGradient{Float32},
    due::Local.DuePlasticityClocks;
    expected_signature::TrajectorySignature,
    mode::Symbol=:cow,
    hard_state_id::Integer=0,
    hard_candidate_ordinal::Integer=0,
)
    length(raw_delta) == Output.OUTPUT_DIM || throw(DimensionMismatch(
        "candidate local raw derivative must have length 22",
    ))
    all(isfinite, raw_delta) || throw(ArgumentError(
        "candidate local raw derivative must be finite",
    ))
    iszero(component_bar.value) || throw(ArgumentError(
        "candidate local replay cannot credit shared V(s)",
    ))
    hard_due = due.hard_event &&
        learner.signals.config.hard_event_multiplier > 0.0f0
    state_identity = Int(hard_state_id)
    candidate_identity = Int(hard_candidate_ordinal)
    if hard_due
        1 <= state_identity <= typemax(UInt16) || throw(ArgumentError(
            "hard-event candidate state identity must fit positive UInt16",
        ))
        1 <= candidate_identity <= typemax(Int32) || throw(ArgumentError(
            "hard-event candidate ordinal must fit positive Int32",
        ))
        previous_candidate = Int(learner.last_hard_candidate_ordinal)
        (previous_candidate == 0 || candidate_identity == previous_candidate + 1) ||
            throw(ArgumentError(
                "hard-event candidates must replay in contiguous logical order",
            ))
    end
    (learner.common_source_seed_loaded ||
     learner.common_source_seed_consumed ||
     learner.common_source_seed_in_progress) && error(
        "candidate replay requires a fresh learner without common hard-event seeds",
    )
    before = _local_counter_snapshot(learner.counters)
    signature = _replay_candidate_world!(
        model,
        state,
        worker,
        input,
        expected_signature,
        mode,
    )
    if hard_due
        _bind_hard_parameter_digest!(
            learner,
            worker.provenance.parameter_digest,
        )
        _preflight_candidate_hard_seed_capacity!(worker, learner)
    end
    _collect_local_plasticity!(
        model,
        worker,
        learner,
        worker.output_tape,
        first(Output.ADVANTAGE_CELLS),
        Output.OUTPUT_CELLS,
    )
    Output.candidate_output_population_pullback!(
        worker.output_dbase,
        worker.output_devidence,
        worker.gradient.output,
        worker.output_scratch,
        worker.output_tape,
        model.parameters.output,
        model.cache.output,
        component_bar,
    )
    Output.candidate_output_initial_state_pullback!(
        worker.gradient.output,
        worker.output_scratch,
        worker.output_dbase,
        model.cache.output,
    )
    learner.counters.output_replays += 1
    _contract_due_local_records!(
        model,
        worker,
        learner,
        raw_delta,
        due;
        allow_common_source=true,
        hard_state_id=state_identity,
        hard_candidate_ordinal=candidate_identity,
    )
    if hard_due
        learner.first_hard_candidate_ordinal == 0 &&
            (learner.first_hard_candidate_ordinal = Int32(candidate_identity))
        learner.last_hard_candidate_ordinal = Int32(candidate_identity)
    end
    learner.counters.candidate_replays += 1
    return _local_report(signature, learner.counters, before)
end

"""
Regenerate the full state-common mandatory/event chronology in the worker's
single reusable tape without retaining a trajectory per state.

The regeneration targets `worker.common_replay_state`; the authoritative
pass-one `state` is read-only even when replay or signature validation fails.
A hard signature mismatch fails before any reverse update.
"""
function replay_state_common!(
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
    input;
    expected_signature::TrajectorySignature=state.common_signature,
)
    state.ready || throw(ArgumentError(
        "prepare_state_common! must run before common replay",
    ))
    state.prepared_revision == model.cache.revision || throw(ArgumentError(
        "state-common replay parameter revision differs from pass one",
    ))
    saved_candidate_count = worker.candidate_count
    saved_value_bits = reinterpret(UInt32, state.state_value)
    replay = worker.common_replay_state
    try
        prepare_state_common!(model, replay, worker, input)
    finally
        worker.candidate_count = saved_candidate_count
    end
    replay_signature = replay.common_signature
    replay_signature == expected_signature || throw(ArgumentError(
        "state-common replay hard trajectory changed",
    ))
    worker.provenance.sealed || error("state-common replay provenance was not sealed")
    worker.provenance.parameter_digest == model.cache.parameter_digest ||
        throw(ArgumentError("state-common replay parameter snapshot changed"))
    replay.fingerprint == state.fingerprint || throw(ArgumentError(
        "state-common replay input fingerprint changed",
    ))
    reinterpret(UInt32, replay.state_value) == saved_value_bits || throw(
        ArgumentError("state-common replay continuous value changed"),
    )
    @inbounds for node in 1:CORE_NODE_COUNT
        for field in 1:Cell.STATE_DIM
            reinterpret(UInt32, replay.common_state[field, node]) ==
                reinterpret(UInt32, state.common_state[field, node]) || throw(
                    ArgumentError("state-common replay continuous state changed"),
                )
        end
        for lane in 1:Axon.PACKET_DIM
            reinterpret(UInt32, replay.common_packet[lane, node]) ==
                reinterpret(UInt32, state.common_packet[lane, node]) || throw(
                    ArgumentError("state-common replay packet changed"),
                )
        end
    end
    return replay_signature
end

function _reverse_common_records!(
    model::CanonicalModel,
    worker::ModelWorker,
)
    @inbounds for record in worker.tape.count:-1:1
        node = Int(worker.tape.node[record])
        packet_bar = @view(worker.record_packet_bar[:, record])
        state_bar = @view(worker.record_state_bar[:, record])
        (_has_nonzero(packet_bar) || _has_nonzero(state_bar)) || continue
        margin_bar = Axon.axon_packet_pullback!(
            worker.dnext_scratch,
            packet_bar,
            @view(worker.tape.previous_state[:, record]),
            @view(worker.tape.next_state[:, record]),
            model.cache.core_cell[node],
        )
        for field in 1:Cell.STATE_DIM
            worker.dnext_scratch[field] += state_bar[field]
        end
        Cell.cell_step_conditional_pullback!(
            worker.dstate_scratch,
            worker.dinput_scratch,
            worker.draw_scratch,
            @view(worker.tape.previous_state[:, record]),
            @view(worker.tape.input[:, record]),
            model.cache.core_cell[node],
            model.cache.core_derivative[node],
            @view(worker.tape.next_state[:, record]),
            worker.dnext_scratch,
            0.0f0,
            0.0f0,
            margin_bar,
        )
        @views worker.gradient.core_cell_raw[:, node] .+= worker.draw_scratch
        phase = TransitionPhase(worker.tape.phase[record])
        if phase == EVENT_WAVE
            previous_record = Int(worker.tape.previous_record[record])
            previous_bar = previous_record == 0 ?
                @view(worker.core_state_bar[:, node]) :
                @view(worker.record_state_bar[:, previous_record])
            for field in 1:Cell.STATE_DIM
                previous_bar[field] += worker.dstate_scratch[field]
            end
            _reverse_event_input!(
                model,
                worker,
                record,
                worker.dinput_scratch,
            )
        else
            _reverse_mandatory_input!(
                model,
                worker,
                record,
                worker.dinput_scratch,
            )
            fill!(worker.draw_scratch, 0.0f0)
            Cell.initial_state_pullback!(
                worker.draw_scratch,
                worker.dstate_scratch,
                model.cache.core_derivative[node],
            )
            @views worker.gradient.core_cell_raw[:, node] .+= worker.draw_scratch
        end
    end
    return nothing
end

"""Credit shared BEFORE-only V(s) once through its full common chronology."""
function conditional_reverse_state_value!(
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
    input,
    value_bar::Float32,
    ; expected_signature::TrajectorySignature=state.common_signature,
)
    replay_state_common!(
        model,
        state,
        worker,
        input;
        expected_signature=expected_signature,
    )
    replay = worker.common_replay_state
    fill!(worker.core_packet_bar, 0.0f0)
    fill!(worker.core_state_bar, 0.0f0)
    @views fill!(worker.record_packet_bar[:, 1:worker.tape.count], 0.0f0)
    @views fill!(worker.record_state_bar[:, 1:worker.tape.count], 0.0f0)
    component_bar = worker.component_bars[1]
    Output.clear_component_gradient!(component_bar)
    component_bar.value = value_bar
    Output.value_output_population_pullback!(
        worker.output_dbase,
        worker.output_devidence,
        worker.gradient.output,
        worker.output_scratch,
        replay.state_value_tape,
        model.parameters.output,
        model.cache.output,
        component_bar,
    )
    Output.value_output_initial_state_pullback!(
        worker.gradient.output,
        worker.output_scratch,
        worker.output_dbase,
        model.cache.output,
    )
    provenance = worker.provenance
    @inbounds for binding in 1:provenance.output_count
        output_cell = Int(provenance.output_cell[binding])
        output_cell <= 2 || continue
        source = Int(provenance.output_source_node[binding])
        source == 0 && continue # exact state-context constant
        source_record = Int(provenance.output_source_record[binding])
        rank = Int(provenance.output_rank[binding])
        destination = _packet_bar_at_record(worker, source, source_record)
        for lane in 1:Axon.PACKET_DIM
            destination[lane] += worker.output_devidence[lane, rank, output_cell]
        end
    end
    _reverse_common_records!(model, worker)
    return worker.gradient
end

"""
Replay the state-common chronology exactly once with the loss-normalized sum
of candidate raw derivatives. Shared V(s) receives its exact private output
credit once; its evidence cotangent is deliberately discarded from the local
recurrent objective.
"""
function local_replay_state_common!(
    model::CanonicalModel,
    state::ModelState,
    worker::ModelWorker,
    learner::ModelLocalLearner,
    input,
    aggregate_raw_delta::AbstractVector{Float32},
    shared_value_bar::Float32,
    due::Local.DuePlasticityClocks;
    expected_signature::TrajectorySignature,
    hard_state_id::Integer=0,
)
    length(aggregate_raw_delta) == Output.OUTPUT_DIM || throw(DimensionMismatch(
        "common local raw derivative must have length 22",
    ))
    all(isfinite, aggregate_raw_delta) || throw(ArgumentError(
        "common local raw derivative must be finite",
    ))
    isfinite(shared_value_bar) || throw(ArgumentError(
        "shared value cotangent must be finite",
    ))
    hard_due = due.hard_event &&
        learner.signals.config.hard_event_multiplier > 0.0f0
    state_identity = Int(hard_state_id)
    if hard_due
        1 <= state_identity <= typemax(UInt16) || throw(ArgumentError(
            "common hard-event replay state identity must fit positive UInt16",
        ))
        learner.common_source_seed_loaded || error(
            "common hard-event replay requires a loaded state seed slab",
        )
        learner.common_source_seed_consumed && error(
            "common hard-event replay seed slab was already consumed",
        )
        learner.common_source_seed_in_progress && error(
            "failed common hard-event replay must be reset before retry",
        )
        Int(learner.common_source_seed_state_id) == state_identity || throw(
            ArgumentError(
                "loaded common hard-event seed slab belongs to another state",
            ),
        )
    elseif learner.common_source_seed_loaded ||
           learner.common_source_seed_in_progress
        error("common hard-event seeds were staged on a non-hard replay clock")
    end
    hard_due && (learner.common_source_seed_in_progress = true)
    before = _local_counter_snapshot(learner.counters)
    signature = replay_state_common!(
        model,
        state,
        worker,
        input;
        expected_signature,
    )
    hard_due && _bind_hard_parameter_digest!(
        learner,
        worker.provenance.parameter_digest,
    )
    replay = worker.common_replay_state
    _collect_local_plasticity!(
        model,
        worker,
        learner,
        replay.state_value_tape,
        first(Output.VALUE_CELLS),
        last(Output.VALUE_CELLS),
    )
    component_bar = worker.component_bars[1]
    Output.clear_component_gradient!(component_bar)
    component_bar.value = shared_value_bar
    Output.value_output_population_pullback!(
        worker.output_dbase,
        worker.output_devidence,
        worker.gradient.output,
        worker.output_scratch,
        replay.state_value_tape,
        model.parameters.output,
        model.cache.output,
        component_bar,
    )
    Output.value_output_initial_state_pullback!(
        worker.gradient.output,
        worker.output_scratch,
        worker.output_dbase,
        model.cache.output,
    )
    learner.counters.output_replays += 1
    _contract_due_local_records!(
        model,
        worker,
        learner,
        aggregate_raw_delta,
        due;
        allow_common_source=false,
        consume_common_source=true,
        hard_state_id=state_identity,
    )
    learner.counters.common_replays += 1
    return _local_report(signature, learner.counters, before)
end


end # module CanonicalDendriticGraph
