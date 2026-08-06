module ReducedHayCPUNativeArena

using ..ReducedHayCPUNativeModel

export STATE_BATCH,
    CANDIDATE_WIDTH,
    CAPACITY,
    CANDIDATE_TAPE_BYTES,
    ARENA_TAPE_BYTES,
    FixedBatchArena,
    ArenaWorker,
    arena_payload_bytes,
    arena_generation,
    begin_batch!,
    forward_candidate!,
    replay_candidate!

const Model = ReducedHayCPUNativeModel

const STATE_BATCH = 8
const CANDIDATE_WIDTH = 80
const CAPACITY = STATE_BATCH * CANDIDATE_WIDTH

const ANCHOR_VALUES = Model.STATE_DIM * Model.CELLS_PER_BLOCK * Model.BLOCKS
const RECURRENT_VALUES =
    Model.STATE_DIM * Model.CELLS_PER_BLOCK * Model.BLOCKS *
    Model.RECURRENT_STEPS
const RECURRENT_INPUT_VALUES =
    Model.Cell.INPUT_DIM * Model.CELLS_PER_BLOCK * Model.BLOCKS *
    Model.RECURRENT_STEPS
const CANDIDATE_TAPE_BYTES =
    sizeof(Float32) * (ANCHOR_VALUES + RECURRENT_VALUES + RECURRENT_INPUT_VALUES)
const ARENA_TAPE_BYTES = CAPACITY * CANDIDATE_TAPE_BYTES

"""
The complete persistent forward tape for one fixed `8 x 80` training batch.

Candidate `flat` occupies one contiguous slab along the final axis.  The tape
contains one physical sensory anchor, every recurrent post-update state, and
the exact input consumed by every cell at every cycle. Encoded copies, pending
conductances, and event rings are worker-local and never duplicated here.
"""
mutable struct FixedBatchArena
    physical_anchor::Array{Float32,4}
    physical_recurrent::Array{Float32,5}
    recurrent_inputs::Array{Float32,5}
    generation::UInt64

    function FixedBatchArena(
        physical_anchor::Array{Float32,4},
        physical_recurrent::Array{Float32,5},
        recurrent_inputs::Array{Float32,5},
        generation::UInt64,
    )
        size(physical_anchor) == (
            Model.STATE_DIM,
            Model.CELLS_PER_BLOCK,
            Model.BLOCKS,
            CAPACITY,
        ) || throw(DimensionMismatch("physical_anchor has the wrong shape"))
        size(physical_recurrent) == (
            Model.STATE_DIM,
            Model.CELLS_PER_BLOCK,
            Model.BLOCKS,
            Model.RECURRENT_STEPS,
            CAPACITY,
        ) || throw(DimensionMismatch("physical_recurrent has the wrong shape"))
        size(recurrent_inputs) == (
            Model.Cell.INPUT_DIM,
            Model.CELLS_PER_BLOCK,
            Model.BLOCKS,
            Model.RECURRENT_STEPS,
            CAPACITY,
        ) || throw(DimensionMismatch("recurrent_inputs has the wrong shape"))
        return new(
            physical_anchor,
            physical_recurrent,
            recurrent_inputs,
            generation,
        )
    end
end

function FixedBatchArena()
    return FixedBatchArena(
        Array{Float32}(
            undef,
            Model.STATE_DIM,
            Model.CELLS_PER_BLOCK,
            Model.BLOCKS,
            CAPACITY,
        ),
        Array{Float32}(
            undef,
            Model.STATE_DIM,
            Model.CELLS_PER_BLOCK,
            Model.BLOCKS,
            Model.RECURRENT_STEPS,
            CAPACITY,
        ),
        Array{Float32}(
            undef,
            Model.Cell.INPUT_DIM,
            Model.CELLS_PER_BLOCK,
            Model.BLOCKS,
            Model.RECURRENT_STEPS,
            CAPACITY,
        ),
        UInt64(0),
    )
end

@inline arena_generation(arena::FixedBatchArena) = arena.generation

"""Pin all candidate tapes in the next batch to one prepared generation."""
function begin_batch!(
    arena::FixedBatchArena,
    prepared::Model.PreparedModelState{Float32},
)
    generation = Model.prepared_generation(prepared)
    Model.assert_generation(prepared, generation)
    arena.generation = generation
    return generation
end

"""The numeric tape payload, excluding Julia array headers and alignment."""
@inline function arena_payload_bytes(arena::FixedBatchArena)
    return sizeof(eltype(arena.physical_anchor)) *
               length(arena.physical_anchor) +
           sizeof(eltype(arena.physical_recurrent)) *
               length(arena.physical_recurrent) +
           sizeof(eltype(arena.recurrent_inputs)) *
               length(arena.recurrent_inputs)
end

"""
One worker's sole owning Model staging buffers and forward scratch.

No candidate slice or view is retained.  The barrierless scheduler owns the
range claim that makes each `flat` commit unique; an arbitrary dequeuing worker
or coordinator helper may execute that claimed flat.  Arena deliberately adds
neither a static worker mapping nor a second atomic ownership protocol.
"""
struct ArenaWorker
    buffers::Model.ForwardBuffers{Float32}
    scratch::Model.ForwardScratch{Float32}
    diagnostics::Model.FullForwardDiagnostics{Float32}
end

function ArenaWorker()
    buffers = Model.ForwardBuffers(Float32)
    return ArenaWorker(
        buffers,
        Model.ForwardScratch(Float32),
        Model.FullForwardDiagnostics(Float32),
    )
end

@inline function _checked_flat(flat::Integer)
    candidate = Int(flat)
    1 <= candidate <= CAPACITY || throw(BoundsError(1:CAPACITY, candidate))
    return candidate
end

@inline function _check_batch_storage(
    raw_output::AbstractMatrix{Float32},
    rails::AbstractMatrix{Float32},
)
    size(raw_output) == (Model.OUTPUT_DIM, CAPACITY) || throw(
        DimensionMismatch("raw_output must have shape (22, 640)"),
    )
    size(rails) == (Model.INPUT_RAILS, CAPACITY) || throw(
        DimensionMismatch("rails must have shape (1298, 640)"),
    )
    return nothing
end

@inline function _commit_candidate!(
    arena::FixedBatchArena,
    raw_output::AbstractMatrix{Float32},
    buffers::Model.ForwardBuffers{Float32},
    scratch::Model.ForwardScratch{Float32},
    diagnostics::Union{Nothing,Model.FullForwardDiagnostics{Float32}},
    flat::Int,
)
    anchor_offset = (flat - 1) * ANCHOR_VALUES + 1
    recurrent_offset = (flat - 1) * RECURRENT_VALUES + 1
    raw_offset = (flat - 1) * Model.OUTPUT_DIM + 1
    copyto!(
        arena.physical_anchor,
        anchor_offset,
        buffers.physical_anchor,
        1,
        ANCHOR_VALUES,
    )
    copyto!(
        arena.physical_recurrent,
        recurrent_offset,
        buffers.physical_recurrent,
        1,
        RECURRENT_VALUES,
    )
    recurrent_input_offset = (flat - 1) * RECURRENT_INPUT_VALUES + 1
    copyto!(
        arena.recurrent_inputs,
        recurrent_input_offset,
        scratch.recurrent_inputs,
        1,
        RECURRENT_INPUT_VALUES,
    )
    copyto!(
        raw_output,
        raw_offset,
        buffers.raw_output,
        1,
        Model.OUTPUT_DIM,
    )
    return nothing
end

"""
Run one route-free candidate through the sole Model kernel, then commit its
trajectory transactionally from worker-owned staging.
"""
function forward_candidate!(
    arena::FixedBatchArena,
    raw_output::AbstractMatrix{Float32},
    worker::ArenaWorker,
    flat::Integer,
    model::Model.CPUHayModel{Float32},
    prepared::Model.PreparedModelState{Float32},
    rails::AbstractMatrix{Float32};
    record_diagnostics::Bool=false,
    event_floor::Float32=0.0f0,
    spike_smoothing::Float32=0.0f0,
)
    candidate = _checked_flat(flat)
    _check_batch_storage(raw_output, rails)
    Model.assert_generation(prepared, arena.generation)
    rail_column = @view rails[:, candidate]
    if record_diagnostics
        Model.forward_candidate!(
            worker.buffers,
            worker.scratch,
            model,
            prepared,
            rail_column,
            worker.diagnostics;
            expected_generation=arena.generation,
            event_floor,
            spike_smoothing,
        )
    else
        Model.forward_candidate!(
            worker.buffers,
            worker.scratch,
            model,
            prepared,
            rail_column,
            Model.NoForwardDiagnostics();
            expected_generation=arena.generation,
            event_floor,
            spike_smoothing,
        )
    end
    _commit_candidate!(
        arena,
        raw_output,
        worker.buffers,
        worker.scratch,
        record_diagnostics ? worker.diagnostics : nothing,
        candidate,
    )
    return nothing
end

"""
Regenerate one candidate trajectory with the current published parameters.
"""
function replay_candidate!(
    arena::FixedBatchArena,
    raw_output::AbstractMatrix{Float32},
    worker::ArenaWorker,
    flat::Integer,
    model::Model.CPUHayModel{Float32},
    prepared::Model.PreparedModelState{Float32},
    rails::AbstractMatrix{Float32},
    ;
    record_diagnostics::Bool=false,
    event_floor::Float32=0.0f0,
    spike_smoothing::Float32=0.0f0,
)
    candidate = _checked_flat(flat)
    _check_batch_storage(raw_output, rails)
    Model.assert_generation(prepared, arena.generation)
    rail_column = @view rails[:, candidate]
    if record_diagnostics
        Model.forward_candidate!(
            worker.buffers,
            worker.scratch,
            model,
            prepared,
            rail_column,
            worker.diagnostics;
            expected_generation=arena.generation,
            event_floor,
            spike_smoothing,
        )
    else
        Model.forward_candidate!(
            worker.buffers,
            worker.scratch,
            model,
            prepared,
            rail_column,
            Model.NoForwardDiagnostics();
            expected_generation=arena.generation,
            event_floor,
            spike_smoothing,
        )
    end
    _commit_candidate!(
        arena,
        raw_output,
        worker.buffers,
        worker.scratch,
        record_diagnostics ? worker.diagnostics : nothing,
        candidate,
    )
    return nothing
end

end # module ReducedHayCPUNativeArena
