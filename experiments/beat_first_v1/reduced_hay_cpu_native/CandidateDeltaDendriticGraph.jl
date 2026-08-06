module CandidateDeltaDendriticGraph

using ..ActiveApicalCell
using ..CandidateDeltaInput
using ..CompactDendriticNode
using ..DendriticDeltaForestTopology
using ..DendriticProgramBank
using ..DendriticDeltaForest
using ..DendriticForestOutput

const Cell = ActiveApicalCell
const Delta = CandidateDeltaInput
const Node = CompactDendriticNode
const Topology = DendriticDeltaForestTopology
const Bank = DendriticProgramBank
const Forest = DendriticDeltaForest
const Output = DendriticForestOutput

export AffectedLeaves,
       ModelParameters,
       ModelCache,
       ModelGradient,
       ModelState,
       ModelWorker,
       ModelForwardStats,
       affected_count,
       clear_gradient!,
       finish_state_pullback!,
       forward_candidate!,
       forward_stats,
       initialize_model,
       prepare_affected_leaves!,
       prepare_candidate!,
       prepare_state!,
       pullback_candidate!,
       refresh_cache!,
       stored_parameter_count

"""
The single canonical candidate-delta dendritic forest.

Leaves own a complete Reduced Hay cell through `leaf_shared_raw`.  The crossed
row/column forest owns its four internal cell classes and signed compact
contacts.  Twenty-two private Reduced Hay output cells consume the two sets of
34 root payloads plus signed Tetris context and the exact placement.  No 27-D
factor, typed random afferent graph, scalar decision bank, or compatibility
path remains in this module.
"""
struct ModelParameters
    leaf_shared_raw::Vector{Float32}
    program_bank::Bank.ProgramBank
    forest::Forest.PlaneParameters{Float32}
    output::Output.ForestOutputParameters{Float32}
end

function initialize_model()
    return ModelParameters(
        Cell.default_raw_parameters(Float32),
        Bank.ProgramBank(),
        Forest.initialize_parameters(),
        Output.initialize_parameters(),
    )
end

@inline function stored_parameter_count(parameters::ModelParameters)
    return length(parameters.leaf_shared_raw) +
           length(parameters.program_bank.payload) +
           Forest.stored_parameter_count(parameters.forest) +
           Output.stored_parameter_count(parameters.output)
end

mutable struct ModelCache
    leaf::Cell.CellParameterCache{Float32}
    leaf_derivative::Cell.CellParameterDerivativeCache{Float32}
    forest::Forest.PlaneCache{Float32}
    output::Output.ForestOutputCache{Float32}
end

function ModelCache(parameters::ModelParameters)
    leaf, leaf_derivative = Cell.parameter_caches(
        parameters.leaf_shared_raw,
    )
    return ModelCache(
        leaf,
        leaf_derivative,
        Forest.PlaneCache(parameters.forest),
        Output.ForestOutputCache(parameters.output),
    )
end

function refresh_cache!(cache::ModelCache, parameters::ModelParameters)
    cache.leaf = Cell.transform_parameters(parameters.leaf_shared_raw)
    cache.leaf_derivative =
        Cell.transform_parameter_derivatives(parameters.leaf_shared_raw)
    Forest.refresh_cache!(cache.forest, parameters.forest)
    Output.refresh_cache!(cache.output, parameters.output)
    return cache
end

"""Sparse program gradient plus all always-active dense forest gradients."""
struct ModelGradient
    leaf_shared_raw::Vector{Float32}
    program::Bank.SparseProgramGradient
    forest::Forest.PlaneGradient{Float32}
    output::Output.ForestOutputGradient{Float32}
end

function ModelGradient(
    parameters::ModelParameters;
    active_program_capacity::Integer=min(
        Bank.bank_row_count(parameters.program_bank),
        16_384,
    ),
)
    return ModelGradient(
        zeros(Float32, Cell.PARAM_DIM),
        Bank.SparseProgramGradient(
            parameters.program_bank,
            active_program_capacity,
        ),
        Forest.PlaneGradient(parameters.forest),
        Output.ForestOutputGradient(Float32),
    )
end

function clear_gradient!(gradient::ModelGradient)
    fill!(gradient.leaf_shared_raw, 0.0f0)
    Bank.reset_sparse_gradient!(gradient.program)
    Forest.clear_gradient!(gradient.forest)
    Output.clear_gradient!(gradient.output)
    return gradient
end

"""Fixed-capacity exact 3x3 leaf dependency closure of changed board cells."""
mutable struct AffectedLeaves <: AbstractVector{UInt16}
    positions::Memory{UInt16}
    marked::Memory{UInt8}
    count::Int
    function AffectedLeaves()
        positions = Memory{UInt16}(undef, Topology.LEAF_COUNT)
        marked = Memory{UInt8}(undef, Topology.LEAF_COUNT)
        fill!(positions, UInt16(0))
        fill!(marked, UInt8(0))
        return new(positions, marked, 0)
    end
end

@inline affected_count(affected::AffectedLeaves) = affected.count
Base.IndexStyle(::Type{AffectedLeaves}) = IndexLinear()
Base.size(affected::AffectedLeaves) = (affected.count,)
Base.length(affected::AffectedLeaves) = affected.count
@inline function Base.getindex(affected::AffectedLeaves, index::Int)
    1 <= index <= affected.count ||
        throw(BoundsError(1:affected.count, index))
    return @inbounds affected.positions[index]
end

@inline _position(row::Int, column::Int) =
    row + (column - 1) * Delta.BOARD_ROWS

"""
Find every forest leaf whose signed 3x3 morphology changes.

This is intentionally owned by the canonical model.  Depending on the retired
`SpatialDendriticFactors` merely for its affected-position helper would make
the old 27-D factor path a hidden production dependency.
"""
function prepare_affected_leaves!(
    affected::AffectedLeaves,
    before::AbstractMatrix,
    after::AbstractMatrix,
)
    size(before) == (Delta.BOARD_ROWS, Delta.BOARD_COLUMNS) ||
        throw(DimensionMismatch("before board must be 24 x 10"))
    size(after) == (Delta.BOARD_ROWS, Delta.BOARD_COLUMNS) ||
        throw(DimensionMismatch("after board must be 24 x 10"))
    fill!(affected.marked, UInt8(0))
    @inbounds for column in 1:Delta.BOARD_COLUMNS
        for row in 1:Delta.BOARD_ROWS
            before[row, column] == after[row, column] && continue
            for leaf_column in max(1, column - 1):min(
                Delta.BOARD_COLUMNS,
                column + 1,
            )
                for leaf_row in max(1, row - 1):min(
                    Delta.BOARD_ROWS,
                    row + 1,
                )
                    affected.marked[_position(leaf_row, leaf_column)] =
                        UInt8(1)
                end
            end
        end
    end
    count = 0
    @inbounds for position in 1:Topology.LEAF_COUNT
        iszero(affected.marked[position]) && continue
        count += 1
        affected.positions[count] = UInt16(position)
    end
    affected.count = count
    return affected
end

@inline function prepare_affected_leaves!(
    affected::AffectedLeaves,
    common::Delta.StateCommon,
    materialization::Delta.CandidateMaterialization,
)
    return prepare_affected_leaves!(
        affected,
        common.board,
        materialization.after,
    )
end

"""State-common planes and grouped cotangents for one Tetris state."""
struct ModelState
    common::Delta.StateCommon
    before::Forest.PlaneState{Float32}
    after_base::Forest.PlaneState{Float32}
    before_bar::Matrix{Float32}
    after_base_bar::Matrix{Float32}
end

ModelState() = ModelState(
    Delta.StateCommon(),
    Forest.PlaneState(Float32),
    Forest.PlaneState(Float32),
    zeros(Float32, Forest.PAYLOAD_DIM, Topology.NODE_COUNT),
    zeros(Float32, Forest.PAYLOAD_DIM, Topology.NODE_COUNT),
)

"""All candidate-local COW, replay, output, and reverse storage of one worker."""
struct ModelWorker
    delta::Delta.CandidateDelta
    materialization::Delta.CandidateMaterialization
    affected::AffectedLeaves
    closure::Topology.DirtyClosure
    candidate_after::Forest.COWPlaneState{Float32}
    forest_workspace::Forest.PlaneWorkspace{Float32}
    anchors::Array{Float32,3}
    anchor_bar::Array{Float32,3}
    context::Vector{Float32}
    context_bar::Vector{Float32}
    output_events::Vector{Float32}
    output_tape::Output.ForestOutputTape{Float32}
    output_scratch::Output.ForestOutputScratch{Float32}
    output_gradient::Output.ForestOutputGradient{Float32}
end

ModelWorker() = ModelWorker(
    Delta.CandidateDelta(),
    Delta.CandidateMaterialization(),
    AffectedLeaves(),
    Topology.DirtyClosure(),
    Forest.COWPlaneState(Float32),
    Forest.PlaneWorkspace(Float32),
    zeros(
        Float32,
        Forest.PAYLOAD_DIM,
        Topology.ANCHOR_COUNT,
        Output.PLANE_COUNT,
    ),
    zeros(
        Float32,
        Forest.PAYLOAD_DIM,
        Topology.ANCHOR_COUNT,
        Output.PLANE_COUNT,
    ),
    zeros(Float32, Output.CONTEXT_DIM),
    zeros(Float32, Output.CONTEXT_DIM),
    zeros(Float32, Output.OUTPUT_CHANNELS),
    Output.ForestOutputTape(Float32),
    Output.ForestOutputScratch(Float32),
    Output.ForestOutputGradient(Float32),
)

"""Allocation-free, explicitly separated execution diagnostics."""
struct ModelForwardStats
    before::Forest.PlaneForwardStats
    after_base::Forest.PlaneForwardStats
    candidate_after::Forest.PlaneForwardStats
    output_hard_events::Int32
    output_event_denominator::Int32
end

@inline function forward_stats(state::ModelState, worker::ModelWorker)
    return ModelForwardStats(
        state.before.stats,
        state.after_base.stats,
        worker.candidate_after.stats,
        Int32(Output.hard_event_count(worker.output_tape)),
        Int32(Output.hard_event_denominator()),
    )
end

"""Evaluate both semantic base planes exactly once and clear grouped bars."""
function prepare_state!(
    state::ModelState,
    worker::ModelWorker,
    parameters::ModelParameters,
    cache::ModelCache,
)
    Forest.base_forward!(
        state.before,
        worker.forest_workspace,
        parameters.forest,
        cache.forest,
        parameters.program_bank,
        state.common.board,
        Forest.BEFORE_PLANE,
        cache.leaf,
    )
    Forest.base_forward!(
        state.after_base,
        worker.forest_workspace,
        parameters.forest,
        cache.forest,
        parameters.program_bank,
        state.common.board,
        Forest.AFTER_PLANE,
        cache.leaf,
    )
    fill!(state.before_bar, 0.0f0)
    fill!(state.after_base_bar, 0.0f0)
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

@inline function _materialize_candidate!(
    worker::ModelWorker,
    state::ModelState,
)
    Delta.reconstruct_candidate!(
        worker.materialization,
        state.common,
        worker.delta,
    )
    prepare_affected_leaves!(
        worker.affected,
        state.common,
        worker.materialization,
    )
    return worker
end

function prepare_candidate!(
    worker::ModelWorker,
    state::ModelState,
    parameters::ModelParameters,
    cache::ModelCache,
    placement::AbstractMatrix,
    tspin::Real,
)
    Delta.prepare_candidate_delta!(
        worker.delta,
        state.common,
        placement,
        tspin,
    )
    return _materialize_candidate!(worker, state)
end

function prepare_candidate!(
    worker::ModelWorker,
    state::ModelState,
    parameters::ModelParameters,
    cache::ModelCache,
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
    return _materialize_candidate!(worker, state)
end

@inline function _copy_output_inputs!(
    worker::ModelWorker,
    state::ModelState,
)
    Forest.copy_anchor_payloads!(
        @view(worker.anchors[:, :, Output.BEFORE_PLANE]),
        state.before,
    )
    Forest.copy_anchor_payloads!(
        @view(worker.anchors[:, :, Output.AFTER_PLANE]),
        worker.candidate_after,
        state.after_base,
    )
    Output.fill_context!(
        worker.context,
        state.common,
        worker.materialization,
    )
    return worker
end

@inline function _forward_prepared_candidate!(
    raw_output::AbstractVector{Float32},
    worker::ModelWorker,
    state::ModelState,
    parameters::ModelParameters,
    cache::ModelCache,
)
    Forest.candidate_forward!(
        worker.candidate_after,
        state.after_base,
        worker.forest_workspace,
        parameters.forest,
        cache.forest,
        parameters.program_bank,
        worker.materialization.after,
        Forest.AFTER_PLANE,
        cache.leaf,
        worker.affected,
        worker.closure,
    )
    _copy_output_inputs!(worker, state)
    Output.forest_output_forward!(
        raw_output,
        worker.output_events,
        worker.output_tape,
        worker.anchors,
        worker.context,
        worker.delta.placement,
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
    prepare_candidate!(worker, state, parameters, cache, placement, tspin)
    return _forward_prepared_candidate!(
        raw_output,
        worker,
        state,
        parameters,
        cache,
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
    prepare_candidate!(
        worker,
        state,
        parameters,
        cache,
        dataset,
        row,
        candidate,
    )
    return _forward_prepared_candidate!(
        raw_output,
        worker,
        state,
        parameters,
        cache,
    )
end

@inline function _accumulate_before_anchor_bar!(
    state::ModelState,
    anchor_bar::AbstractArray{Float32,3},
)
    @inbounds for anchor in 1:Topology.ANCHOR_COUNT
        node = Int(Topology.anchor_node(anchor))
        for lane in 1:Forest.PAYLOAD_DIM
            state.before_bar[lane, node] +=
                anchor_bar[lane, anchor, Output.BEFORE_PLANE]
        end
    end
    return state.before_bar
end

"""Replay and reverse one prepared candidate into the grouped base DAG."""
function pullback_candidate!(
    gradient::ModelGradient,
    raw_output::AbstractVector{Float32},
    raw_bar::AbstractVector{Float32},
    worker::ModelWorker,
    state::ModelState,
    parameters::ModelParameters,
    cache::ModelCache,
)
    # Candidate storage is worker-local and reused by every candidate.  Replay
    # both the COW forest and its 22 output cells before exact conditional VJP.
    _forward_prepared_candidate!(
        raw_output,
        worker,
        state,
        parameters,
        cache,
    )
    Output.forest_output_pullback!(
        worker.anchor_bar,
        worker.context_bar,
        worker.output_gradient,
        worker.output_scratch,
        worker.output_tape,
        worker.anchors,
        worker.context,
        worker.delta.placement,
        parameters.output,
        cache.output,
        raw_bar,
    )
    Output.accumulate_gradient!(gradient.output, worker.output_gradient)
    _accumulate_before_anchor_bar!(state, worker.anchor_bar)
    Forest.candidate_reverse!(
        gradient.forest,
        gradient.leaf_shared_raw,
        gradient.program,
        state.after_base_bar,
        worker.forest_workspace,
        worker.candidate_after,
        state.after_base,
        parameters.forest,
        cache.forest,
        parameters.program_bank,
        worker.materialization.after,
        Forest.AFTER_PLANE,
        cache.leaf,
        cache.leaf_derivative,
        @view(worker.anchor_bar[:, :, Output.AFTER_PLANE]),
        worker.closure,
    )
    return gradient
end

"""Reverse both common planes once after every candidate was replayed."""
function finish_state_pullback!(
    gradient::ModelGradient,
    worker::ModelWorker,
    state::ModelState,
    parameters::ModelParameters,
    cache::ModelCache,
)
    Forest.base_reverse!(
        gradient.forest,
        gradient.leaf_shared_raw,
        gradient.program,
        state.before_bar,
        worker.forest_workspace,
        state.before,
        parameters.forest,
        cache.forest,
        parameters.program_bank,
        state.common.board,
        Forest.BEFORE_PLANE,
        cache.leaf,
        cache.leaf_derivative,
    )
    Forest.base_reverse!(
        gradient.forest,
        gradient.leaf_shared_raw,
        gradient.program,
        state.after_base_bar,
        worker.forest_workspace,
        state.after_base,
        parameters.forest,
        cache.forest,
        parameters.program_bank,
        state.common.board,
        Forest.AFTER_PLANE,
        cache.leaf,
        cache.leaf_derivative,
    )
    return gradient
end

end # module CandidateDeltaDendriticGraph
