module CandidateDeltaRelationGraph

using ..ActiveApicalCell
using ..CandidateDeltaInput
using ..DendriticProgramBank
using ..SpatialProgramPackets
using ..DendriticRelationTopology
using ..DendriticMotifTopology
using ..TypedDendriticAfferents
using ..HighDimensionalCellPacket
using ..TypedRelationCellBank
using ..TypedRelationContext
using ..TypedOutputCellBank
using ..StructuredMotifReadout

const Cell = ActiveApicalCell
const Delta = CandidateDeltaInput
const Bank = DendriticProgramBank
const Spatial = SpatialProgramPackets
const Topology = DendriticRelationTopology
const MotifTopology = DendriticMotifTopology
const Afferents = TypedDendriticAfferents
const Packet = HighDimensionalCellPacket
const Relations = TypedRelationCellBank
const Context = TypedRelationContext
const Outputs = TypedOutputCellBank
const Readout = StructuredMotifReadout

export ModelCache,
       ModelForwardStats,
       ModelGradient,
       ModelParameters,
       ModelState,
       ModelWorker,
       AffectedPositions,
       affected_count,
       clear_gradient!,
       finish_state_pullback!,
       forward_candidate!,
       forward_candidate_full_overlay!,
       forward_stats,
       initialize_model,
       prepare_affected_positions!,
       prepare_candidate!,
       prepare_state!,
       pullback_candidate!,
       refresh_cache!,
       stored_parameter_count

"""Fixed payload width emitted by the bit-serial dendritic program bank."""
const PROGRAM_PACKET_DIM = Spatial.PACKET_WIDTH

"""Full continuous Reduced-Hay transition width exported between cell banks."""
const CELL_PACKET_DIM = Packet.PACKET_DIM
const PROGRAM_SOURCES = Spatial.PACKET_COUNT
const RELATION_CELLS = Relations.RELATION_CELLS
const MOTIF_CELLS = MotifTopology.MOTIF_COUNT
const OUTPUT_CELLS = Outputs.OUTPUT_CELLS
const PLACEMENT_SOURCES = Delta.BOARD_CELLS

const OPPONENT_PAIR_SIZE = 2
const LEAF_RELATION_FANOUT = PROGRAM_PACKET_DIM * OPPONENT_PAIR_SIZE

# Every signed voltage and soma margin uses an opponent pair.  Conductance,
# plateau and adaptation coordinates are physically non-negative, so a single
# positive contact carries each without wasting typed-input capacity.  This
# leaves each five-field compartment bundle intact at one motif destination.
const RELATION_MOTIF_FANOUT =
    Cell.N_COMPARTMENTS * (Cell.COMPARTMENT_STATE_DIM + 1) + 3
const PLACEMENT_RELATION_FANOUT = Topology.SOURCE_FANOUT

const LEAF_RELATION_INITIAL_CONDUCTANCE = 0.1f0
const RELATION_MOTIF_INITIAL_CONDUCTANCE = 0.1f0
const PLACEMENT_INITIAL_CONDUCTANCE = 0.25f0

@inline _position(row::Int, column::Int) =
    row + (column - 1) * Delta.BOARD_ROWS

@inline function _raw_conductance(value::Float32)
    value > 0.0f0 || throw(ArgumentError("conductance must be positive"))
    return value + log(-expm1(-value))
end

"""Build the fixed 480-packet to 48-relation anatomical graph."""
function _build_leaf_relation_graph()
    topology = Topology.canonical_topology()
    slots = PROGRAM_SOURCES * LEAF_RELATION_FANOUT
    fields = Vector{UInt16}(undef, slots)
    polarities = fill(Int8(1), slots)
    cells = Vector{UInt16}(undef, slots)
    compartments = Vector{UInt8}(undef, slots)
    receptors = Vector{UInt8}(undef, slots)
    raw = fill(_raw_conductance(LEAF_RELATION_INITIAL_CONDUCTANCE), slots)

    @inbounds for source in 1:PROGRAM_SOURCES
        for field in 1:PROGRAM_PACKET_DIM
            topology_slot = mod(field - 1, Topology.SOURCE_FANOUT) + 1
            specification = Topology.source_contact(
                topology,
                source,
                topology_slot,
            )
            receptor = UInt8(
                mod(source + field - 2, Cell.INPUT_CHANNELS) + 1,
            )
            # Every 16D packet lane is exposed exactly once to the four
            # semantic topology families, while the two field octaves cover
            # all eight basal compartments.
            positive_compartment = mod(field - 1, Cell.N_BASAL) + 1
            negative_compartment = mod(positive_compartment, Cell.N_BASAL) + 1
            for member in 1:OPPONENT_PAIR_SIZE
                local_slot = (field - 1) * OPPONENT_PAIR_SIZE + member
                slot = (source - 1) * LEAF_RELATION_FANOUT + local_slot
                fields[slot] = UInt16(field)
                polarities[slot] = member == 1 ? Int8(1) : Int8(-1)
                cells[slot] = UInt16(specification.relation)
                compartments[slot] = UInt8(
                    member == 1 ? positive_compartment : negative_compartment,
                )
                receptors[slot] = receptor
            end
        end
    end
    return Afferents.TypedAfferentGraph(
        PROGRAM_SOURCES,
        fill(Afferents.ANALOG_FIELD, PROGRAM_PACKET_DIM),
        RELATION_CELLS,
        LEAF_RELATION_FANOUT,
        fields,
        polarities,
        cells,
        compartments,
        receptors,
        raw,
    )
end

"""Build the fixed semantic 48-relation to 48-motif anatomical graph."""
function _build_relation_motif_graph()
    topology = MotifTopology.canonical_topology()
    slots = RELATION_CELLS * RELATION_MOTIF_FANOUT
    fields = Vector{UInt16}(undef, slots)
    polarities = fill(Int8(1), slots)
    cells = Vector{UInt16}(undef, slots)
    compartments = Vector{UInt8}(undef, slots)
    receptors = Vector{UInt8}(undef, slots)
    raw = fill(_raw_conductance(RELATION_MOTIF_INITIAL_CONDUCTANCE), slots)

    @inbounds for source in 1:RELATION_CELLS
        family = source <= 24 ? 1 : source <= 34 ? 2 : source <= 42 ? 3 : 4
        phase = 2 * (family - 1)
        local_slot = 0
        for source_compartment in 1:Cell.N_COMPARTMENTS
            # A physical compartment's V/AMPA/NMDA/GABA/plateau coordinates
            # must remain co-located.  Splitting them by lane would destroy
            # precisely the dendritic interaction the motif cell is meant to
            # compute.  Nine bundles are distributed 3/2/2/2 over the four
            # fixed semantic destinations.
            destination_rank = mod(
                source_compartment - 1,
                MotifTopology.SOURCE_FANOUT,
            ) + 1
            bundle_rank = div(
                source_compartment - 1,
                MotifTopology.SOURCE_FANOUT,
            )
            destination = MotifTopology.motif_destination(
                topology,
                source,
                destination_rank,
            )
            target_a = 1 + mod(3 * bundle_rank + phase, Cell.N_COMPARTMENTS)
            target_b = 1 + mod(3 * bundle_rank + 1 + phase, Cell.N_COMPARTMENTS)
            target_c = 1 + mod(3 * bundle_rank + 2 + phase, Cell.N_COMPARTMENTS)

            voltage_field = Packet.packet_lane(
                source_compartment,
                Cell.FIELD_VOLTAGE,
            )
            for (polarity, target, receptor) in (
                (Int8(1), target_a, Cell.INPUT_AMPA),
                (Int8(-1), target_a, Cell.INPUT_GABA),
            )
                local_slot += 1
                slot = (source - 1) * RELATION_MOTIF_FANOUT + local_slot
                fields[slot] = UInt16(voltage_field)
                polarities[slot] = polarity
                cells[slot] = UInt16(destination)
                compartments[slot] = UInt8(target)
                receptors[slot] = UInt8(receptor)
            end

            for (field_kind, target, receptor) in (
                (Cell.FIELD_AMPA, target_b, Cell.INPUT_AMPA),
                (Cell.FIELD_NMDA, target_b, Cell.INPUT_NMDA),
                (Cell.FIELD_GABA, target_b, Cell.INPUT_GABA),
                (Cell.FIELD_PLATEAU, target_c, Cell.INPUT_NMDA),
            )
                local_slot += 1
                slot = (source - 1) * RELATION_MOTIF_FANOUT + local_slot
                fields[slot] = UInt16(Packet.packet_lane(
                    source_compartment,
                    field_kind,
                ))
                polarities[slot] = Int8(1)
                cells[slot] = UInt16(destination)
                compartments[slot] = UInt8(target)
                receptors[slot] = UInt8(receptor)
            end
        end

        # The transition margin and adaptation belong beside the apical/soma
        # bundle (source compartment nine, semantic destination one).
        soma_destination = MotifTopology.motif_destination(
            topology,
            source,
            1,
        )
        soma_bundle_rank = div(Cell.N_COMPARTMENTS - 1, MotifTopology.SOURCE_FANOUT)
        soma_a = 1 + mod(3 * soma_bundle_rank + phase, Cell.N_COMPARTMENTS)
        soma_c = 1 + mod(3 * soma_bundle_rank + 2 + phase, Cell.N_COMPARTMENTS)
        for (field, polarity, target, receptor) in (
            (Packet.MARGIN_LANE, Int8(1), soma_c, Cell.INPUT_AMPA),
            (Packet.MARGIN_LANE, Int8(-1), soma_c, Cell.INPUT_GABA),
            (Packet.ADAPTATION_LANE, Int8(1), soma_a, Cell.INPUT_NMDA),
        )
            local_slot += 1
            slot = (source - 1) * RELATION_MOTIF_FANOUT + local_slot
            fields[slot] = UInt16(field)
            polarities[slot] = polarity
            cells[slot] = UInt16(soma_destination)
            compartments[slot] = UInt8(target)
            receptors[slot] = UInt8(receptor)
        end
        local_slot == RELATION_MOTIF_FANOUT || error(
            "relation-motif fanout construction drifted",
        )
    end
    return Afferents.TypedAfferentGraph(
        RELATION_CELLS,
        fill(Afferents.ANALOG_FIELD, CELL_PACKET_DIM),
        MOTIF_CELLS,
        RELATION_MOTIF_FANOUT,
        fields,
        polarities,
        cells,
        compartments,
        receptors,
        raw,
    )
end

"""Raw placement contacts preserve pre-clear action identity in relations."""
function _build_placement_relation_graph()
    topology = Topology.canonical_topology()
    slots = PLACEMENT_SOURCES * PLACEMENT_RELATION_FANOUT
    fields = fill(UInt16(1), slots)
    polarities = fill(Int8(1), slots)
    cells = Vector{UInt16}(undef, slots)
    compartments = Vector{UInt8}(undef, slots)
    receptors = Vector{UInt8}(undef, slots)
    raw = fill(_raw_conductance(PLACEMENT_INITIAL_CONDUCTANCE), slots)
    @inbounds for position in 1:PLACEMENT_SOURCES
        after_source = Int(Topology.source_id(Topology.AFTER_PLANE, position))
        for topology_slot in 1:Topology.SOURCE_FANOUT
            specification = Topology.source_contact(
                topology,
                after_source,
                topology_slot,
            )
            slot = (position - 1) * PLACEMENT_RELATION_FANOUT + topology_slot
            cells[slot] = UInt16(specification.relation)
            compartments[slot] = specification.basal_compartment
            receptors[slot] = UInt8(
                mod(position + topology_slot - 2, Cell.INPUT_CHANNELS) + 1,
            )
        end
    end
    return Afferents.TypedAfferentGraph(
        PLACEMENT_SOURCES,
        UInt8[Afferents.HARD_BIT_FIELD],
        RELATION_CELLS,
        PLACEMENT_RELATION_FANOUT,
        fields,
        polarities,
        cells,
        compartments,
        receptors,
        raw,
    )
end

"""
The sole parameter owner of the HD candidate-delta relation graph.

Receptor, source-field, polarity, destination and compartment identities are
immutable arrays inside each afferent graph.  Only their non-negative physical
conductances are trainable.
"""
struct ModelParameters
    program_bank::Bank.ProgramBank
    leaf_relation::Afferents.TypedAfferentGraph{Float32}
    relation::Relations.RelationParameters{Float32}
    relation_motif::Afferents.TypedAfferentGraph{Float32}
    motif::Relations.RelationParameters{Float32}
    context::Context.RelationContextGraphs{Float32}
    placement_relation::Afferents.TypedAfferentGraph{Float32}
    motif_readout::Readout.StructuredReadoutParameters{Float32}
    output::Outputs.TypedOutputParameters{Float32}
end

function initialize_model()
    return ModelParameters(
        Bank.ProgramBank(),
        _build_leaf_relation_graph(),
        Relations.initialize_parameters(),
        _build_relation_motif_graph(),
        Relations.initialize_parameters(),
        Context.build_relation_context(),
        _build_placement_relation_graph(),
        Readout.initialize_parameters(),
        Outputs.initialize_parameters(),
    )
end

@inline function stored_parameter_count(parameters::ModelParameters)
    return length(parameters.program_bank.payload) +
           length(parameters.leaf_relation.raw_conductance) +
           Relations.stored_parameter_count(parameters.relation) +
           length(parameters.relation_motif.raw_conductance) +
           Relations.stored_parameter_count(parameters.motif) +
           Afferents.contact_count(parameters.context.common_relation) +
           Afferents.contact_count(parameters.context.common_output) +
           Afferents.contact_count(parameters.context.aux_relation) +
           length(parameters.placement_relation.raw_conductance) +
           Readout.stored_parameter_count(parameters.motif_readout) +
           Outputs.stored_parameter_count(parameters.output)
end

mutable struct ModelCache
    relation::Relations.RelationCache{Float32}
    motif::Relations.RelationCache{Float32}
    output::Outputs.TypedOutputCache{Float32}
    leaf_relation::Afferents.TypedAfferentCache{Float32}
    relation_motif::Afferents.TypedAfferentCache{Float32}
    common_relation::Afferents.TypedAfferentCache{Float32}
    common_output::Afferents.TypedAfferentCache{Float32}
    aux_relation::Afferents.TypedAfferentCache{Float32}
    placement_relation::Afferents.TypedAfferentCache{Float32}
    motif_readout::Readout.StructuredReadoutCache{Float32}
end

ModelCache(parameters::ModelParameters) = ModelCache(
    Relations.RelationCache(parameters.relation),
    Relations.RelationCache(parameters.motif),
    Outputs.TypedOutputCache(parameters.output),
    Afferents.TypedAfferentCache(parameters.leaf_relation),
    Afferents.TypedAfferentCache(parameters.relation_motif),
    Afferents.TypedAfferentCache(parameters.context.common_relation),
    Afferents.TypedAfferentCache(parameters.context.common_output),
    Afferents.TypedAfferentCache(parameters.context.aux_relation),
    Afferents.TypedAfferentCache(parameters.placement_relation),
    Readout.StructuredReadoutCache(parameters.motif_readout),
)

function refresh_cache!(cache::ModelCache, parameters::ModelParameters)
    Relations.refresh_cache!(cache.relation, parameters.relation)
    Relations.refresh_cache!(cache.motif, parameters.motif)
    Outputs.refresh_cache!(cache.output, parameters.output)
    Afferents.refresh_cache!(cache.leaf_relation, parameters.leaf_relation)
    Afferents.refresh_cache!(cache.relation_motif, parameters.relation_motif)
    Afferents.refresh_cache!(
        cache.common_relation,
        parameters.context.common_relation,
    )
    Afferents.refresh_cache!(
        cache.common_output,
        parameters.context.common_output,
    )
    Afferents.refresh_cache!(cache.aux_relation, parameters.context.aux_relation)
    Afferents.refresh_cache!(
        cache.placement_relation,
        parameters.placement_relation,
    )
    Readout.refresh_cache!(cache.motif_readout, parameters.motif_readout)
    return cache
end

"""Sparse program gradient and every continuously trainable graph group."""
struct ModelGradient
    program::Bank.SparseProgramGradient
    leaf_relation::Vector{Float32}
    relation::Relations.RelationGradient{Float32}
    relation_motif::Vector{Float32}
    motif::Relations.RelationGradient{Float32}
    context::Context.RelationContextGradient{Float32}
    placement_relation::Vector{Float32}
    motif_readout::Readout.StructuredReadoutGradient{Float32}
    output::Outputs.TypedOutputGradient{Float32}
end

function ModelGradient(
    parameters::ModelParameters;
    active_program_capacity::Int=131_072,
)
    return ModelGradient(
        Bank.SparseProgramGradient(
            parameters.program_bank,
            min(
                active_program_capacity,
                Bank.bank_row_count(parameters.program_bank),
            ),
        ),
        zeros(Float32, Afferents.contact_count(parameters.leaf_relation)),
        Relations.RelationGradient(Float32),
        zeros(Float32, Afferents.contact_count(parameters.relation_motif)),
        Relations.RelationGradient(Float32),
        Context.RelationContextGradient(parameters.context),
        zeros(Float32, Afferents.contact_count(parameters.placement_relation)),
        Readout.StructuredReadoutGradient(Float32),
        Outputs.TypedOutputGradient(Float32),
    )
end

function clear_gradient!(gradient::ModelGradient)
    Bank.reset_sparse_gradient!(gradient.program)
    fill!(gradient.leaf_relation, 0.0f0)
    Relations.clear_gradient!(gradient.relation)
    fill!(gradient.relation_motif, 0.0f0)
    Relations.clear_gradient!(gradient.motif)
    Context.clear_context_gradient!(gradient.context)
    fill!(gradient.placement_relation, 0.0f0)
    Readout.clear_gradient!(gradient.motif_readout)
    Outputs.clear_gradient!(gradient.output)
    return gradient
end

"""Fixed 3x3 semantic-address closure of board cells changed by a candidate."""
mutable struct AffectedPositions <: AbstractVector{UInt16}
    positions::Memory{UInt16}
    after_sources::Memory{UInt16}
    marked::Memory{UInt8}
    count::Int
    function AffectedPositions()
        positions = Memory{UInt16}(undef, Spatial.POSITION_COUNT)
        sources = Memory{UInt16}(undef, Spatial.POSITION_COUNT)
        marked = Memory{UInt8}(undef, Spatial.POSITION_COUNT)
        fill!(positions, UInt16(0))
        fill!(sources, UInt16(0))
        fill!(marked, UInt8(0))
        return new(positions, sources, marked, 0)
    end
end

@inline affected_count(affected::AffectedPositions) = affected.count
Base.IndexStyle(::Type{AffectedPositions}) = IndexLinear()
Base.size(affected::AffectedPositions) = (affected.count,)
Base.length(affected::AffectedPositions) = affected.count
@inline function Base.getindex(affected::AffectedPositions, index::Int)
    1 <= index <= affected.count || throw(BoundsError(1:affected.count, index))
    return @inbounds affected.positions[index]
end

function prepare_affected_positions!(
    affected::AffectedPositions,
    before::AbstractMatrix,
    after::AbstractMatrix,
)
    size(before) == (Delta.BOARD_ROWS, Delta.BOARD_COLUMNS) ||
        throw(DimensionMismatch("before board must be 24 x 10"))
    size(after) == (Delta.BOARD_ROWS, Delta.BOARD_COLUMNS) ||
        throw(DimensionMismatch("after board must be 24 x 10"))
    fill!(affected.marked, UInt8(0))
    @inbounds for column in 1:Delta.BOARD_COLUMNS, row in 1:Delta.BOARD_ROWS
        before[row, column] == after[row, column] && continue
        for local_column in max(1, column - 1):min(Delta.BOARD_COLUMNS, column + 1)
            for local_row in max(1, row - 1):min(Delta.BOARD_ROWS, row + 1)
                affected.marked[_position(local_row, local_column)] = UInt8(1)
            end
        end
    end
    count = 0
    @inbounds for position in 1:Spatial.POSITION_COUNT
        iszero(affected.marked[position]) && continue
        count += 1
        affected.positions[count] = UInt16(position)
        affected.after_sources[count] = UInt16(
            Spatial.packet_column(position, Spatial.AFTER_PLANE),
        )
    end
    affected.count = count
    return affected
end

"""All base trajectories and grouped cotangents owned by one Tetris state."""
struct ModelState
    common::Delta.StateCommon
    program_packets::Matrix{Float32}
    program_packet_bar::Matrix{Float32}
    relation_initial_state::Matrix{Float32}
    relation_inbox::Matrix{Float32}
    relation_packet::Matrix{Float32}
    relation_event::Vector{Float32}
    relation_tape::Relations.RelationTape{Float32}
    relation_final_state_bar::Matrix{Float32}
    relation_packet_bar::Matrix{Float32}
    relation_initial_state_bar::Matrix{Float32}
    relation_inbox_bar::Matrix{Float32}
    motif_initial_state::Matrix{Float32}
    motif_inbox::Matrix{Float32}
    motif_packet::Matrix{Float32}
    motif_event::Vector{Float32}
    motif_tape::Relations.RelationTape{Float32}
    motif_final_state_bar::Matrix{Float32}
    motif_packet_bar::Matrix{Float32}
    motif_initial_state_bar::Matrix{Float32}
    motif_inbox_bar::Matrix{Float32}
    output_initial_state::Matrix{Float32}
    output_inbox::Matrix{Float32}
    output_value::Vector{Float32}
    output_event::Vector{Float32}
    output_tape::Outputs.TypedOutputTape{Float32}
    output_final_state_bar::Matrix{Float32}
    output_inbox_bar::Matrix{Float32}
    output_initial_state_bar::Matrix{Float32}
    context::Context.RelationContextScratch{Float32}
end

function ModelState()
    return ModelState(
        Delta.StateCommon(),
        zeros(Float32, PROGRAM_PACKET_DIM, PROGRAM_SOURCES),
        zeros(Float32, PROGRAM_PACKET_DIM, PROGRAM_SOURCES),
        zeros(Float32, Cell.STATE_DIM, RELATION_CELLS),
        zeros(Float32, Cell.INPUT_DIM, RELATION_CELLS),
        zeros(Float32, CELL_PACKET_DIM, RELATION_CELLS),
        zeros(Float32, RELATION_CELLS),
        Relations.RelationTape(Float32),
        zeros(Float32, Cell.STATE_DIM, RELATION_CELLS),
        zeros(Float32, CELL_PACKET_DIM, RELATION_CELLS),
        zeros(Float32, Cell.STATE_DIM, RELATION_CELLS),
        zeros(Float32, Cell.INPUT_DIM, RELATION_CELLS),
        zeros(Float32, Cell.STATE_DIM, MOTIF_CELLS),
        zeros(Float32, Cell.INPUT_DIM, MOTIF_CELLS),
        zeros(Float32, CELL_PACKET_DIM, MOTIF_CELLS),
        zeros(Float32, MOTIF_CELLS),
        Relations.RelationTape(Float32),
        zeros(Float32, Cell.STATE_DIM, MOTIF_CELLS),
        zeros(Float32, CELL_PACKET_DIM, MOTIF_CELLS),
        zeros(Float32, Cell.STATE_DIM, MOTIF_CELLS),
        zeros(Float32, Cell.INPUT_DIM, MOTIF_CELLS),
        zeros(Float32, Cell.STATE_DIM, OUTPUT_CELLS),
        zeros(Float32, Cell.INPUT_DIM, OUTPUT_CELLS),
        zeros(Float32, OUTPUT_CELLS),
        zeros(Float32, OUTPUT_CELLS),
        Outputs.TypedOutputTape(Float32),
        zeros(Float32, Cell.STATE_DIM, OUTPUT_CELLS),
        zeros(Float32, Cell.INPUT_DIM, OUTPUT_CELLS),
        zeros(Float32, Cell.STATE_DIM, OUTPUT_CELLS),
        Context.RelationContextScratch(Float32),
    )
end

"""Candidate-local COW overlay, typed inbox and exact reverse scratch."""
struct ModelWorker
    delta::Delta.CandidateDelta
    materialization::Delta.CandidateMaterialization
    affected::AffectedPositions
    closure::Topology.AffectedRelationClosure
    motif_closure::MotifTopology.AffectedMotifClosure
    spatial::Spatial.SpatialPacketWorkspace
    candidate_packets::Matrix{Float32}
    packet_delta::Matrix{Float32}
    packet_delta_grid::Matrix{Float32}
    candidate_relation_packet::Matrix{Float32}
    relation_packet_delta::Matrix{Float32}
    relation_delta_inbox::Matrix{Float32}
    relation_event::Vector{Float32}
    relation_tape::Relations.RelationTape{Float32}
    relation_scratch::Relations.RelationScratch{Float32}
    candidate_motif_packet::Matrix{Float32}
    motif_packet_delta::Matrix{Float32}
    motif_delta_inbox::Matrix{Float32}
    motif_event::Vector{Float32}
    motif_tape::Relations.RelationTape{Float32}
    motif_scratch::Relations.RelationScratch{Float32}
    output_delta_inbox::Matrix{Float32}
    output_event::Vector{Float32}
    output_tape::Outputs.TypedOutputTape{Float32}
    output_scratch::Outputs.TypedOutputScratch{Float32}
    context::Context.RelationContextScratch{Float32}
    placement_packet::Matrix{Float32}
    placement_packet_bar::Matrix{Float32}
    output_base_state_bar::Matrix{Float32}
    output_inbox_bar::Matrix{Float32}
    relation_initial_state_bar::Matrix{Float32}
    relation_inbox_bar::Matrix{Float32}
    relation_packet_delta_bar::Matrix{Float32}
    motif_initial_state_bar::Matrix{Float32}
    motif_inbox_bar::Matrix{Float32}
    motif_packet_delta_bar::Matrix{Float32}
    packet_delta_grid_bar::Matrix{Float32}
    candidate_packet_bar::Matrix{Float32}
    packet_delta_bar::Matrix{Float32}
end

function ModelWorker()
    return ModelWorker(
        Delta.CandidateDelta(),
        Delta.CandidateMaterialization(),
        AffectedPositions(),
        Topology.AffectedRelationClosure(),
        MotifTopology.AffectedMotifClosure(),
        Spatial.SpatialPacketWorkspace(),
        zeros(Float32, PROGRAM_PACKET_DIM, Spatial.POSITION_COUNT),
        zeros(Float32, PROGRAM_PACKET_DIM, Spatial.POSITION_COUNT),
        zeros(Float32, PROGRAM_PACKET_DIM, PROGRAM_SOURCES),
        zeros(Float32, CELL_PACKET_DIM, RELATION_CELLS),
        zeros(Float32, CELL_PACKET_DIM, RELATION_CELLS),
        zeros(Float32, Cell.INPUT_DIM, RELATION_CELLS),
        zeros(Float32, RELATION_CELLS),
        Relations.RelationTape(Float32),
        Relations.RelationScratch(Float32),
        zeros(Float32, CELL_PACKET_DIM, MOTIF_CELLS),
        zeros(Float32, CELL_PACKET_DIM, MOTIF_CELLS),
        zeros(Float32, Cell.INPUT_DIM, MOTIF_CELLS),
        zeros(Float32, MOTIF_CELLS),
        Relations.RelationTape(Float32),
        Relations.RelationScratch(Float32),
        zeros(Float32, Cell.INPUT_DIM, OUTPUT_CELLS),
        zeros(Float32, OUTPUT_CELLS),
        Outputs.TypedOutputTape(Float32),
        Outputs.TypedOutputScratch(Float32),
        Context.RelationContextScratch(Float32),
        zeros(Float32, 1, PLACEMENT_SOURCES),
        zeros(Float32, 1, PLACEMENT_SOURCES),
        zeros(Float32, Cell.STATE_DIM, OUTPUT_CELLS),
        zeros(Float32, Cell.INPUT_DIM, OUTPUT_CELLS),
        zeros(Float32, Cell.STATE_DIM, RELATION_CELLS),
        zeros(Float32, Cell.INPUT_DIM, RELATION_CELLS),
        zeros(Float32, CELL_PACKET_DIM, RELATION_CELLS),
        zeros(Float32, Cell.STATE_DIM, MOTIF_CELLS),
        zeros(Float32, Cell.INPUT_DIM, MOTIF_CELLS),
        zeros(Float32, CELL_PACKET_DIM, MOTIF_CELLS),
        zeros(Float32, PROGRAM_PACKET_DIM, PROGRAM_SOURCES),
        zeros(Float32, PROGRAM_PACKET_DIM, Spatial.POSITION_COUNT),
        zeros(Float32, PROGRAM_PACKET_DIM, Spatial.POSITION_COUNT),
    )
end

struct ModelForwardStats
    affected_positions::Int32
    affected_relations::Int32
    affected_motifs::Int32
    base_relation_events::Int32
    candidate_relation_events::Int32
    base_motif_events::Int32
    candidate_motif_events::Int32
    base_output_events::Int32
    candidate_output_events::Int32
end

@inline function forward_stats(state::ModelState, worker::ModelWorker)
    candidate_relation_events = 0
    @inbounds for index in eachindex(worker.closure)
        relation = Int(worker.closure[index])
        candidate_relation_events +=
            !iszero(worker.relation_tape.events[1, relation])
    end
    candidate_motif_events = 0
    @inbounds for index in eachindex(worker.motif_closure)
        motif = Int(worker.motif_closure[index])
        candidate_motif_events += !iszero(worker.motif_tape.events[1, motif])
    end
    return ModelForwardStats(
        Int32(affected_count(worker.affected)),
        Int32(Topology.affected_count(worker.closure)),
        Int32(MotifTopology.motif_count(worker.motif_closure)),
        Int32(Relations.hard_event_count(state.relation_tape)),
        Int32(candidate_relation_events),
        Int32(Relations.hard_event_count(state.motif_tape)),
        Int32(candidate_motif_events),
        Int32(Outputs.hard_event_count(state.output_tape)),
        Int32(Outputs.hard_event_count(worker.output_tape)),
    )
end

@inline function _fill_placement_packet!(worker::ModelWorker)
    count = Delta.placement_count(worker.delta)
    count <= Delta.PLACEMENT_CAPACITY || error(
        "placement contains more than $(Delta.PLACEMENT_CAPACITY) active blocks",
    )
    @inbounds for index in 1:count
        position = Int(Delta.placement_position(worker.delta, index))
        worker.placement_packet[1, position] = 1.0f0
    end
    return worker.placement_packet
end

"""Union board dependencies with every actually driven context destination."""
function _complete_candidate_closure!(
    worker::ModelWorker,
    parameters::ModelParameters,
)
    mask = worker.closure.mask
    aux_graph = parameters.context.aux_relation
    @inbounds for slot in 1:Afferents.contact_count(aux_graph)
        relation = Int(aux_graph.destination_cell[slot])
        mask |= UInt64(1) << (relation - 1)
    end
    placement_graph = parameters.placement_relation
    @inbounds for index in 1:Delta.placement_count(worker.delta)
        source = Int(Delta.placement_position(worker.delta, index))
        first_slot = (source - 1) * placement_graph.fanout + 1
        for rank in 1:placement_graph.fanout
            relation = Int(placement_graph.destination_cell[first_slot + rank - 1])
            mask |= UInt64(1) << (relation - 1)
        end
    end
    count = 0
    @inbounds for relation in 1:RELATION_CELLS
        iszero(mask & (UInt64(1) << (relation - 1))) && continue
        count += 1
        worker.closure.relations[count] = UInt8(relation)
    end
    worker.closure.mask = mask
    worker.closure.count = count
    MotifTopology.fill_affected_motif_closure!(
        worker.motif_closure,
        MotifTopology.canonical_topology(),
        worker.closure.relations,
        worker.closure.count,
    )
    return worker.closure
end

@inline function _deposit_common_context!(
    relation_inbox,
    output_inbox,
    parameters::ModelParameters,
    cache::ModelCache,
    common,
    scratch,
)
    Context.pack_state_common!(scratch.common_packet, common)
    Afferents.deposit_typed!(
        relation_inbox,
        parameters.context.common_relation,
        cache.common_relation,
        scratch.common_packet,
    )
    Afferents.deposit_typed!(
        output_inbox,
        parameters.context.common_output,
        cache.common_output,
        scratch.common_packet,
    )
    return relation_inbox, output_inbox
end

@inline function _deposit_candidate_aux_context!(
    relation_inbox,
    parameters::ModelParameters,
    cache::ModelCache,
    materialization,
    scratch,
)
    Context.pack_candidate_aux!(scratch.aux_packet, materialization)
    Afferents.deposit_typed!(
        relation_inbox,
        parameters.context.aux_relation,
        cache.aux_relation,
        scratch.aux_packet,
    )
    return relation_inbox
end

@inline function _pullback_common_context!(
    scratch,
    gradient::ModelGradient,
    parameters::ModelParameters,
    cache::ModelCache,
    relation_inbox_bar,
    output_inbox_bar,
)
    Afferents.deposit_typed_pullback!(
        scratch.common_packet_bar,
        gradient.context.common_relation_raw,
        parameters.context.common_relation,
        cache.common_relation,
        scratch.common_packet,
        relation_inbox_bar,
    )
    Afferents.deposit_typed_pullback!(
        scratch.common_packet_bar,
        gradient.context.common_output_raw,
        parameters.context.common_output,
        cache.common_output,
        scratch.common_packet,
        output_inbox_bar,
    )
    return scratch.common_packet_bar, gradient
end

@inline function _pullback_candidate_aux_context!(
    scratch,
    gradient::ModelGradient,
    parameters::ModelParameters,
    cache::ModelCache,
    relation_inbox_bar,
)
    Afferents.deposit_typed_pullback!(
        scratch.aux_packet_bar,
        gradient.context.aux_relation_raw,
        parameters.context.aux_relation,
        cache.aux_relation,
        scratch.aux_packet,
        relation_inbox_bar,
    )
    return scratch.aux_packet_bar, gradient
end

function prepare_state!(
    state::ModelState,
    worker::ModelWorker,
    parameters::ModelParameters,
    cache::ModelCache,
)
    Spatial.base_packet_grid!(
        state.program_packets,
        worker.spatial,
        parameters.program_bank,
        state.common.board,
    )

    fill!(state.relation_inbox, 0.0f0)
    fill!(state.motif_inbox, 0.0f0)
    fill!(state.output_inbox, 0.0f0)
    Afferents.deposit_typed!(
        state.relation_inbox,
        parameters.leaf_relation,
        cache.leaf_relation,
        state.program_packets,
    )
    _deposit_common_context!(
        state.relation_inbox,
        state.output_inbox,
        parameters,
        cache,
        state.common,
        state.context,
    )
    Relations.relation_initial_state!(state.relation_initial_state, cache.relation)
    Relations.relation_forward!(
        state.relation_packet,
        state.relation_event,
        state.relation_tape,
        state.relation_initial_state,
        state.relation_inbox,
        parameters.relation,
        cache.relation,
    )

    Afferents.deposit_typed!(
        state.motif_inbox,
        parameters.relation_motif,
        cache.relation_motif,
        state.relation_packet,
    )
    Relations.relation_initial_state!(state.motif_initial_state, cache.motif)
    Relations.relation_forward!(
        state.motif_packet,
        state.motif_event,
        state.motif_tape,
        state.motif_initial_state,
        state.motif_inbox,
        parameters.motif,
        cache.motif,
    )
    Readout.deposit_readout!(
        state.output_inbox,
        state.motif_packet,
        cache.motif_readout,
        1.0f0,
    )
    Readout.deposit_readout!(
        state.output_inbox,
        state.relation_packet,
        cache.motif_readout,
        Readout.RELATION_RESIDUAL_SCALE,
    )
    Outputs.output_initial_state!(state.output_initial_state, cache.output)
    Outputs.typed_output_forward!(
        state.output_value,
        state.output_event,
        state.output_tape,
        state.output_initial_state,
        state.output_inbox,
        parameters.output,
        cache.output,
    )

    fill!(state.program_packet_bar, 0.0f0)
    fill!(state.relation_final_state_bar, 0.0f0)
    fill!(state.relation_packet_bar, 0.0f0)
    fill!(state.relation_initial_state_bar, 0.0f0)
    fill!(state.relation_inbox_bar, 0.0f0)
    fill!(state.motif_final_state_bar, 0.0f0)
    fill!(state.motif_packet_bar, 0.0f0)
    fill!(state.motif_initial_state_bar, 0.0f0)
    fill!(state.motif_inbox_bar, 0.0f0)
    fill!(state.output_final_state_bar, 0.0f0)
    fill!(state.output_inbox_bar, 0.0f0)
    fill!(state.output_initial_state_bar, 0.0f0)
    Context.clear_packet_bars!(state.context)
    return state
end

function prepare_state!(
    state::ModelState,
    worker::ModelWorker,
    parameters::ModelParameters,
    cache::ModelCache,
    dataset,
    row::Int,
)
    Delta.prepare_state_common!(state.common, dataset, row)
    return prepare_state!(state, worker, parameters, cache)
end

function prepare_candidate!(
    worker::ModelWorker,
    state::ModelState,
    parameters::ModelParameters,
    placement::AbstractMatrix,
    tspin::Real,
)
    Delta.prepare_candidate_delta!(worker.delta, state.common, placement, tspin)
    Delta.reconstruct_candidate!(worker.materialization, state.common, worker.delta)
    prepare_affected_positions!(
        worker.affected,
        state.common.board,
        worker.materialization.after,
    )
    Topology.fill_affected_relation_closure!(
        worker.closure,
        Topology.canonical_topology(),
        Topology.AFTER_PLANE,
        worker.affected.positions,
        worker.affected.count,
    )
    _fill_placement_packet!(worker)
    _complete_candidate_closure!(worker, parameters)
    return worker
end

function prepare_candidate!(
    worker::ModelWorker,
    state::ModelState,
    parameters::ModelParameters,
    dataset,
    row::Int,
    candidate::Int,
)
    Delta.prepare_candidate_delta!(
        worker.delta,
        state.common,
        dataset,
        row,
        candidate,
    )
    Delta.reconstruct_candidate!(worker.materialization, state.common, worker.delta)
    prepare_affected_positions!(
        worker.affected,
        state.common.board,
        worker.materialization.after,
    )
    Topology.fill_affected_relation_closure!(
        worker.closure,
        Topology.canonical_topology(),
        Topology.AFTER_PLANE,
        worker.affected.positions,
        worker.affected.count,
    )
    _fill_placement_packet!(worker)
    _complete_candidate_closure!(worker, parameters)
    return worker
end

@inline function _candidate_program_delta!(
    worker::ModelWorker,
    state::ModelState,
    parameters::ModelParameters,
    positions,
)
    count = length(positions)
    candidate = @view worker.candidate_packets[:, 1:count]
    delta = @view worker.packet_delta[:, 1:count]
    Spatial.candidate_after_packets!(
        candidate,
        delta,
        worker.spatial,
        parameters.program_bank,
        state.program_packets,
        worker.materialization.after,
        positions,
    )
    @inbounds for index in 1:count
        source = Spatial.packet_column(positions[index], Spatial.AFTER_PLANE)
        for lane in 1:PROGRAM_PACKET_DIM
            worker.packet_delta_grid[lane, source] = delta[lane, index]
        end
    end
    return candidate, delta
end

@inline function _relation_delta!(worker::ModelWorker, state::ModelState)
    @inbounds for index in eachindex(worker.closure)
        relation = Int(worker.closure[index])
        for lane in 1:CELL_PACKET_DIM
            worker.relation_packet_delta[lane, relation] =
                worker.candidate_relation_packet[lane, relation] -
                state.relation_packet[lane, relation]
        end
    end
    return worker.relation_packet_delta
end

@inline function _motif_delta!(worker::ModelWorker, state::ModelState)
    @inbounds for index in eachindex(worker.motif_closure)
        motif = Int(worker.motif_closure[index])
        for lane in 1:CELL_PACKET_DIM
            worker.motif_packet_delta[lane, motif] =
                worker.candidate_motif_packet[lane, motif] -
                state.motif_packet[lane, motif]
        end
    end
    return worker.motif_packet_delta
end

@inline function _active_placement_positions(worker::ModelWorker)
    return @view worker.delta.placement_positions[1:Delta.placement_count(worker.delta)]
end

function _forward_prepared_candidate!(
    raw_output::AbstractVector{Float32},
    worker::ModelWorker,
    state::ModelState,
    parameters::ModelParameters,
    cache::ModelCache,
    program_positions,
    program_sources,
)
    _candidate_program_delta!(worker, state, parameters, program_positions)

    fill!(worker.relation_delta_inbox, 0.0f0)
    fill!(worker.motif_delta_inbox, 0.0f0)
    fill!(worker.output_delta_inbox, 0.0f0)
    _deposit_candidate_aux_context!(
        worker.relation_delta_inbox,
        parameters,
        cache,
        worker.materialization,
        worker.context,
    )
    placement_sources = _active_placement_positions(worker)
    Afferents.deposit_sources!(
        worker.relation_delta_inbox,
        parameters.placement_relation,
        cache.placement_relation,
        worker.placement_packet,
        placement_sources,
    )
    Afferents.deposit_sources!(
        worker.relation_delta_inbox,
        parameters.leaf_relation,
        cache.leaf_relation,
        worker.packet_delta_grid,
        program_sources,
    )
    Relations.relation_forward_selected!(
        worker.candidate_relation_packet,
        worker.relation_event,
        worker.relation_tape,
        @view(state.relation_tape.states[:, :, Relations.PHASE_COUNT + 1]),
        worker.relation_delta_inbox,
        parameters.relation,
        cache.relation,
        worker.closure,
    )
    _relation_delta!(worker, state)

    Afferents.deposit_sources!(
        worker.motif_delta_inbox,
        parameters.relation_motif,
        cache.relation_motif,
        worker.relation_packet_delta,
        worker.closure,
    )
    Relations.relation_forward_selected!(
        worker.candidate_motif_packet,
        worker.motif_event,
        worker.motif_tape,
        @view(state.motif_tape.states[:, :, Relations.PHASE_COUNT + 1]),
        worker.motif_delta_inbox,
        parameters.motif,
        cache.motif,
        worker.motif_closure,
    )
    _motif_delta!(worker, state)
    Readout.deposit_readout_selected!(
        worker.output_delta_inbox,
        worker.motif_packet_delta,
        cache.motif_readout,
        worker.motif_closure,
        1.0f0,
    )
    Readout.deposit_readout_selected!(
        worker.output_delta_inbox,
        worker.relation_packet_delta,
        cache.motif_readout,
        worker.closure,
        Readout.RELATION_RESIDUAL_SCALE,
    )
    Outputs.typed_output_forward!(
        raw_output,
        worker.output_event,
        worker.output_tape,
        state.output_tape.next_state,
        worker.output_delta_inbox,
        parameters.output,
        cache.output,
    )
    return raw_output
end

function forward_candidate!(
    raw_output::AbstractVector{Float32},
    worker::ModelWorker,
    state::ModelState,
    parameters::ModelParameters,
    cache::ModelCache,
    placement::AbstractMatrix,
    tspin::Real,
)
    prepare_candidate!(worker, state, parameters, placement, tspin)
    positions = @view worker.affected.positions[1:worker.affected.count]
    sources = @view worker.affected.after_sources[1:worker.affected.count]
    return _forward_prepared_candidate!(
        raw_output,
        worker,
        state,
        parameters,
        cache,
        positions,
        sources,
    )
end

function forward_candidate!(
    raw_output::AbstractVector{Float32},
    worker::ModelWorker,
    state::ModelState,
    parameters::ModelParameters,
    cache::ModelCache,
    dataset,
    row::Int,
    candidate::Int,
)
    prepare_candidate!(worker, state, parameters, dataset, row, candidate)
    positions = @view worker.affected.positions[1:worker.affected.count]
    sources = @view worker.affected.after_sources[1:worker.affected.count]
    return _forward_prepared_candidate!(
        raw_output,
        worker,
        state,
        parameters,
        cache,
        positions,
        sources,
    )
end

"""
Dense source oracle for the sparse candidate program path.

All 240 after-plane packets and all 240 signed deltas are materialized and
deposited.  The same exact relation closure is advanced because a zero-delta
relation is not a candidate event and must retain the common base state.
"""
function forward_candidate_full_overlay!(
    raw_output::AbstractVector{Float32},
    worker::ModelWorker,
    state::ModelState,
    parameters::ModelParameters,
    cache::ModelCache,
    placement::AbstractMatrix,
    tspin::Real,
)
    prepare_candidate!(worker, state, parameters, placement, tspin)
    positions = 1:Spatial.POSITION_COUNT
    sources = (Spatial.POSITION_COUNT + 1):Spatial.PACKET_COUNT
    return _forward_prepared_candidate!(
        raw_output,
        worker,
        state,
        parameters,
        cache,
        positions,
        sources,
    )
end

@inline function _accumulate_matrix!(destination, source)
    @inbounds @simd for index in eachindex(destination, source)
        destination[index] += source[index]
    end
    return destination
end

"""
Reverse one sparse candidate into candidate-local parameters and grouped base
cotangents.  The hard relation/output events are control-only; this is the
exact continuous VJP conditional on the recorded event sequence.
"""
function pullback_candidate!(
    gradient::ModelGradient,
    raw_bar::AbstractVector{Float32},
    worker::ModelWorker,
    state::ModelState,
    parameters::ModelParameters,
    cache::ModelCache,
)
    length(raw_bar) == OUTPUT_CELLS || throw(DimensionMismatch(
        "raw output cotangent must have $OUTPUT_CELLS values",
    ))
    positions = @view worker.affected.positions[1:worker.affected.count]
    sources = @view worker.affected.after_sources[1:worker.affected.count]
    placement_sources = _active_placement_positions(worker)

    fill!(worker.packet_delta_grid_bar, 0.0f0)
    fill!(worker.relation_packet_delta_bar, 0.0f0)
    fill!(worker.motif_packet_delta_bar, 0.0f0)
    fill!(worker.placement_packet_bar, 0.0f0)
    Context.clear_packet_bars!(worker.context)

    Outputs.typed_output_pullback!(
        worker.output_base_state_bar,
        worker.output_inbox_bar,
        gradient.output,
        worker.output_scratch,
        worker.output_tape,
        parameters.output,
        cache.output,
        raw_bar,
    )
    _accumulate_matrix!(
        state.output_final_state_bar,
        worker.output_base_state_bar,
    )

    Readout.deposit_readout_selected_pullback!(
        worker.motif_packet_delta_bar,
        gradient.motif_readout,
        worker.motif_packet_delta,
        cache.motif_readout,
        worker.motif_closure,
        worker.output_inbox_bar,
        1.0f0,
    )
    Readout.deposit_readout_selected_pullback!(
        worker.relation_packet_delta_bar,
        gradient.motif_readout,
        worker.relation_packet_delta,
        cache.motif_readout,
        worker.closure,
        worker.output_inbox_bar,
        Readout.RELATION_RESIDUAL_SCALE,
    )
    # motif_packet_delta = candidate_motif_packet - base_motif_packet
    @inbounds for index in eachindex(worker.motif_closure)
        motif = Int(worker.motif_closure[index])
        for lane in 1:CELL_PACKET_DIM
            cotangent = worker.motif_packet_delta_bar[lane, motif]
            state.motif_packet_bar[lane, motif] -= cotangent
        end
    end
    Relations.relation_pullback_selected!(
        worker.motif_initial_state_bar,
        worker.motif_inbox_bar,
        gradient.motif,
        worker.motif_scratch,
        worker.motif_tape,
        parameters.motif,
        cache.motif,
        worker.motif_packet_delta_bar,
        worker.motif_closure,
    )
    @inbounds for index in eachindex(worker.motif_closure)
        motif = Int(worker.motif_closure[index])
        for cell_state in 1:Cell.STATE_DIM
            state.motif_final_state_bar[cell_state, motif] +=
                worker.motif_initial_state_bar[cell_state, motif]
        end
    end
    Afferents.deposit_sources_pullback!(
        worker.relation_packet_delta_bar,
        gradient.relation_motif,
        parameters.relation_motif,
        cache.relation_motif,
        worker.relation_packet_delta,
        worker.closure,
        worker.motif_inbox_bar,
    )
    # relation_packet_delta = candidate_relation_packet - base_relation_packet
    @inbounds for index in eachindex(worker.closure)
        relation = Int(worker.closure[index])
        for lane in 1:CELL_PACKET_DIM
            cotangent = worker.relation_packet_delta_bar[lane, relation]
            state.relation_packet_bar[lane, relation] -= cotangent
        end
    end
    Relations.relation_pullback_selected!(
        worker.relation_initial_state_bar,
        worker.relation_inbox_bar,
        gradient.relation,
        worker.relation_scratch,
        worker.relation_tape,
        parameters.relation,
        cache.relation,
        worker.relation_packet_delta_bar,
        worker.closure,
    )
    @inbounds for index in eachindex(worker.closure)
        relation = Int(worker.closure[index])
        for cell_state in 1:Cell.STATE_DIM
            state.relation_final_state_bar[cell_state, relation] +=
                worker.relation_initial_state_bar[cell_state, relation]
        end
    end

    Afferents.deposit_sources_pullback!(
        worker.packet_delta_grid_bar,
        gradient.leaf_relation,
        parameters.leaf_relation,
        cache.leaf_relation,
        worker.packet_delta_grid,
        sources,
        worker.relation_inbox_bar,
    )
    Afferents.deposit_sources_pullback!(
        worker.placement_packet_bar,
        gradient.placement_relation,
        parameters.placement_relation,
        cache.placement_relation,
        worker.placement_packet,
        placement_sources,
        worker.relation_inbox_bar,
    )
    _pullback_candidate_aux_context!(
        worker.context,
        gradient,
        parameters,
        cache,
        worker.relation_inbox_bar,
    )

    count = length(positions)
    candidate_bar = @view worker.candidate_packet_bar[:, 1:count]
    delta_bar = @view worker.packet_delta_bar[:, 1:count]
    @inbounds for index in 1:count
        source = Int(sources[index])
        for lane in 1:PROGRAM_PACKET_DIM
            candidate_bar[lane, index] = 0.0f0
            delta_bar[lane, index] =
                worker.packet_delta_grid_bar[lane, source]
        end
    end
    Spatial.candidate_after_packets_pullback!(
        gradient.program,
        state.program_packet_bar,
        worker.spatial,
        parameters.program_bank,
        worker.materialization.after,
        positions,
        candidate_bar,
        delta_bar,
    )
    return gradient
end

"""Reverse the common output transition from all grouped candidate credit."""
function _base_output_pullback!(
    gradient::ModelGradient,
    worker::ModelWorker,
    state::ModelState,
    cache::ModelCache,
)
    scratch = worker.output_scratch
    @inbounds for output in 1:OUTPUT_CELLS
        Cell.cell_step_conditional_pullback!(
            scratch.dstate,
            scratch.dinput,
            scratch.draw_step,
            @view(state.output_tape.base_state[:, output]),
            @view(state.output_tape.inbox[:, output]),
            cache.output.cell[output],
            cache.output.derivative[output],
            @view(state.output_tape.next_state[:, output]),
            @view(state.output_final_state_bar[:, output]),
            0.0f0,
            0.0f0,
            0.0f0,
        )
        scratch.dstate[Cell.SPIKE_INDEX] = 0.0f0
        for parameter in 1:Cell.PARAM_DIM
            gradient.output.cell_raw[parameter, output] +=
                scratch.draw_step[parameter]
        end
        for channel in 1:Cell.INPUT_DIM
            state.output_inbox_bar[channel, output] = scratch.dinput[channel]
        end
        for cell_state in 1:Cell.STATE_DIM
            state.output_initial_state_bar[cell_state, output] =
                scratch.dstate[cell_state]
        end
    end
    Outputs.output_initial_state_pullback!(
        gradient.output,
        scratch,
        state.output_initial_state_bar,
        cache.output,
    )
    return gradient
end

"""Reverse the common relation transition including candidate state credit."""
function _base_relation_pullback!(
    gradient::ModelGradient,
    worker::ModelWorker,
    state::ModelState,
    cache::ModelCache,
)
    scratch = worker.relation_scratch
    @inbounds for relation in 1:RELATION_CELLS
        margin_bar = Packet.cell_packet_column_pullback!(
            scratch.dnext,
            state.relation_packet_bar,
            state.relation_tape.states,
            relation,
            1,
            Relations.PHASE_COUNT + 1,
            cache.relation.cell[relation],
        )
        for cell_state in 1:Cell.STATE_DIM
            scratch.dnext[cell_state] +=
                state.relation_final_state_bar[cell_state, relation]
        end
        Cell.cell_step_conditional_pullback!(
            scratch.dstate,
            scratch.dinput,
            scratch.draw_step,
            @view(state.relation_tape.states[:, relation, 1]),
            @view(state.relation_tape.driven_input[:, relation]),
            cache.relation.cell[relation],
            cache.relation.derivative[relation],
            @view(state.relation_tape.states[:, relation, 2]),
            scratch.dnext,
            0.0f0,
            0.0f0,
            margin_bar,
        )
        scratch.dstate[Cell.SPIKE_INDEX] = 0.0f0
        for parameter in 1:Cell.PARAM_DIM
            gradient.relation.cell_raw[parameter, relation] +=
                scratch.draw_step[parameter]
        end
        for channel in 1:Cell.INPUT_DIM
            state.relation_inbox_bar[channel, relation] =
                scratch.dinput[channel]
        end
        for cell_state in 1:Cell.STATE_DIM
            state.relation_initial_state_bar[cell_state, relation] =
                scratch.dstate[cell_state]
        end
    end
    Relations.relation_initial_state_pullback!(
        gradient.relation,
        scratch,
        state.relation_initial_state_bar,
        cache.relation,
    )
    return gradient
end

"""Reverse the common motif transition including every candidate COW bar."""
function _base_motif_pullback!(
    gradient::ModelGradient,
    worker::ModelWorker,
    state::ModelState,
    cache::ModelCache,
)
    scratch = worker.motif_scratch
    @inbounds for motif in 1:MOTIF_CELLS
        margin_bar = Packet.cell_packet_column_pullback!(
            scratch.dnext,
            state.motif_packet_bar,
            state.motif_tape.states,
            motif,
            1,
            Relations.PHASE_COUNT + 1,
            cache.motif.cell[motif],
        )
        for cell_state in 1:Cell.STATE_DIM
            scratch.dnext[cell_state] +=
                state.motif_final_state_bar[cell_state, motif]
        end
        Cell.cell_step_conditional_pullback!(
            scratch.dstate,
            scratch.dinput,
            scratch.draw_step,
            @view(state.motif_tape.states[:, motif, 1]),
            @view(state.motif_tape.driven_input[:, motif]),
            cache.motif.cell[motif],
            cache.motif.derivative[motif],
            @view(state.motif_tape.states[:, motif, 2]),
            scratch.dnext,
            0.0f0,
            0.0f0,
            margin_bar,
        )
        scratch.dstate[Cell.SPIKE_INDEX] = 0.0f0
        for parameter in 1:Cell.PARAM_DIM
            gradient.motif.cell_raw[parameter, motif] +=
                scratch.draw_step[parameter]
        end
        for channel in 1:Cell.INPUT_DIM
            state.motif_inbox_bar[channel, motif] = scratch.dinput[channel]
        end
        for cell_state in 1:Cell.STATE_DIM
            state.motif_initial_state_bar[cell_state, motif] =
                scratch.dstate[cell_state]
        end
    end
    Relations.relation_initial_state_pullback!(
        gradient.motif,
        scratch,
        state.motif_initial_state_bar,
        cache.motif,
    )
    return gradient
end

"""
Finish one state after all candidate pullbacks.

Shared base output, relation and program trajectories are reversed exactly
once.  This is the grouped reverse counterpart of state-common COW forward.
"""
function finish_state_pullback!(
    gradient::ModelGradient,
    worker::ModelWorker,
    state::ModelState,
    parameters::ModelParameters,
    cache::ModelCache,
)
    _base_output_pullback!(gradient, worker, state, cache)

    Readout.deposit_readout_pullback!(
        state.motif_packet_bar,
        gradient.motif_readout,
        state.motif_packet,
        cache.motif_readout,
        state.output_inbox_bar,
        1.0f0,
    )
    Readout.deposit_readout_pullback!(
        state.relation_packet_bar,
        gradient.motif_readout,
        state.relation_packet,
        cache.motif_readout,
        state.output_inbox_bar,
        Readout.RELATION_RESIDUAL_SCALE,
    )

    _base_motif_pullback!(gradient, worker, state, cache)
    Afferents.deposit_typed_pullback!(
        state.relation_packet_bar,
        gradient.relation_motif,
        parameters.relation_motif,
        cache.relation_motif,
        state.relation_packet,
        state.motif_inbox_bar,
    )

    _base_relation_pullback!(gradient, worker, state, cache)
    Afferents.deposit_typed_pullback!(
        state.program_packet_bar,
        gradient.leaf_relation,
        parameters.leaf_relation,
        cache.leaf_relation,
        state.program_packets,
        state.relation_inbox_bar,
    )
    _pullback_common_context!(
        state.context,
        gradient,
        parameters,
        cache,
        state.relation_inbox_bar,
        state.output_inbox_bar,
    )
    Spatial.base_packet_grid_pullback!(
        gradient.program,
        worker.spatial,
        parameters.program_bank,
        state.common.board,
        state.program_packet_bar,
    )
    return gradient
end

end # module CandidateDeltaRelationGraph
