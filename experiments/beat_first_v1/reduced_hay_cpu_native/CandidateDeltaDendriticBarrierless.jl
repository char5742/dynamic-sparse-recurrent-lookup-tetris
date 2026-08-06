module CandidateDeltaDendriticBarrierless

using ..BarrierlessScheduler
using ..DendriticProgramBank
using ..CandidateDeltaDendriticGraph
using ..TetrisRankingBatch

const SchedulerCore = BarrierlessScheduler
const Bank = DendriticProgramBank
const Model = CandidateDeltaDendriticGraph
const Ranking = TetrisRankingBatch

export BarrierlessExactExecutor,
       BarrierlessExactSession,
       ForwardDiagnostics,
       backward_batch!,
       clear_batch_gradient!,
       forward_batch!,
       forward_loss_backward!,
       latest_forward_diagnostics,
       latest_loss,
       loss_and_raw_gradient!,
       run_executor_team!,
       scheduler_report

const _PREPARE_STATE = UInt16(1)
const _FORWARD_CANDIDATE = UInt16(2)
const _BACKWARD_STATE = UInt16(3)

const _LEAF_EVENTS = 1
const _INTERNAL_EVENTS = 2
const _ROOT_EVENTS = 3
const _OUTPUT_EVENTS = 4
const _EVALUATED_NODES = 5
const _DIRTY_LEAVES = 6
const _DIRTY_ANCESTORS = 7
const _COMPACT_MESSAGES = 8
const _DIAGNOSTIC_COUNT = 8

"""One complete allocation-free forward diagnostic snapshot."""
struct ForwardDiagnostics
    leaf_events::Int64
    internal_events::Int64
    root_events::Int64
    output_events::Int64
    evaluated_nodes::Int64
    dirty_leaves::Int64
    dirty_ancestors::Int64
    compact_messages::Int64
end

"""
Fixed DDF arena with three scheduler phases.

State-common forest planes are evaluated in phase one. Candidate COW forests
and the 22-cell output cascade are evaluated in phase two. Exact reverse owns
one state per job in phase three so grouped base-plane cotangents remain
race-free. Native workers may steal jobs in every phase; the coordinator still
reduces state gradients in ascending state-slot order.
"""
mutable struct BarrierlessExactExecutor{D}
    parameters::Model.ModelParameters
    cache::Model.ModelCache
    gradient::Model.ModelGradient
    states::Vector{Model.ModelState}
    workers::Vector{Model.ModelWorker}
    state_gradients::Vector{Model.ModelGradient}
    loss_scratch::Ranking.LossScratch
    loss_sink::Base.RefValue{Ranking.SupervisedLoss}
    batch::Ranking.Batch
    dataset::D
    state_batch::Int
    width::Int
    candidate_chunk_size::Int
    diagnostic_counters::Matrix{Int64}
end

@inline function _state_program_capacity(
    parameters::Model.ModelParameters,
    width::Int,
)
    # Four semantic rows are touched by every evaluated leaf. Two full base
    # planes plus one conservative full candidate plane per stored candidate
    # bound the number of distinct rows without allocating a dense gradient.
    leaf_visits = Base.checked_mul(
        2 + width,
        Model.Topology.LEAF_COUNT,
    )
    return min(
        Model.Bank.bank_row_count(parameters.program_bank),
        Base.checked_mul(leaf_visits, Model.Bank.MAX_ACTIVE_ROWS),
    )
end

@inline function _batch_program_capacity(
    parameters::Model.ModelParameters,
    state_batch::Int,
    width::Int,
)
    return min(
        Model.Bank.bank_row_count(parameters.program_bank),
        Base.checked_mul(
            state_batch,
            _state_program_capacity(parameters, width),
        ),
    )
end

function BarrierlessExactExecutor(
    parameters::Model.ModelParameters,
    batch::Ranking.Batch,
    dataset::D;
    state_program_capacity::Union{Nothing,Integer}=nothing,
    batch_program_capacity::Union{Nothing,Integer}=nothing,
    worker_capacity::Integer=Base.Threads.nthreads(:default),
    candidate_chunk_size::Integer=4,
) where {D<:Ranking.ValidatedDataset}
    batch.width == dataset.candidate_width || throw(DimensionMismatch(
        "batch width $(batch.width) differs from dataset width " *
        "$(dataset.candidate_width)",
    ))
    states = batch.state_batch
    width = batch.width
    workers = Int(worker_capacity)
    workers >= 1 || throw(ArgumentError("worker_capacity must be positive"))
    chunk = Int(candidate_chunk_size)
    chunk >= 1 || throw(ArgumentError(
        "candidate_chunk_size must be positive",
    ))
    state_capacity = isnothing(state_program_capacity) ?
        _state_program_capacity(parameters, width) :
        Int(state_program_capacity)
    batch_capacity = isnothing(batch_program_capacity) ?
        _batch_program_capacity(parameters, states, width) :
        Int(batch_program_capacity)
    return BarrierlessExactExecutor{D}(
        parameters,
        Model.ModelCache(parameters),
        Model.ModelGradient(
            parameters;
            active_program_capacity=batch_capacity,
        ),
        [Model.ModelState() for _ in 1:states],
        [Model.ModelWorker() for _ in 1:workers],
        [Model.ModelGradient(
            parameters;
            active_program_capacity=state_capacity,
        ) for _ in 1:states],
        Ranking.LossScratch(width, states),
        Ref{Ranking.SupervisedLoss}(),
        batch,
        dataset,
        states,
        width,
        chunk,
        zeros(Int64, _DIAGNOSTIC_COUNT, workers),
    )
end

@inline function _record_forest_stats!(
    executor::BarrierlessExactExecutor,
    worker_slot::Int,
    stats::Model.Forest.PlaneForwardStats,
    candidate_delta::Bool,
)
    counters = executor.diagnostic_counters
    @inbounds begin
        counters[_LEAF_EVENTS, worker_slot] += stats.leaf_hard_events
        counters[_INTERNAL_EVENTS, worker_slot] += stats.internal_hard_events
        counters[_ROOT_EVENTS, worker_slot] += stats.root_hard_events
        counters[_EVALUATED_NODES, worker_slot] +=
            stats.dirty_leaves + stats.dirty_ancestors
        counters[_COMPACT_MESSAGES, worker_slot] += stats.compact_messages
        if candidate_delta
            counters[_DIRTY_LEAVES, worker_slot] += stats.dirty_leaves
            counters[_DIRTY_ANCESTORS, worker_slot] += stats.dirty_ancestors
        end
    end
    return nothing
end

@inline function _prepare_state!(
    executor::BarrierlessExactExecutor,
    worker_slot::Int,
    state_slot::Int,
)
    batch = executor.batch
    row = @inbounds batch.rows[state_slot]
    state = @inbounds executor.states[state_slot]
    worker = @inbounds executor.workers[worker_slot]
    Model.prepare_state!(
        state,
        worker,
        executor.parameters,
        executor.cache,
        executor.dataset,
        row,
    )
    _record_forest_stats!(executor, worker_slot, state.before.stats, false)
    _record_forest_stats!(executor, worker_slot, state.after_base.stats, false)
    return nothing
end

@inline function _forward_candidate_range!(
    executor::BarrierlessExactExecutor,
    worker_slot::Int,
    first_ordinal::Int,
    last_ordinal::Int,
)
    batch = executor.batch
    worker = @inbounds executor.workers[worker_slot]
    @inbounds for ordinal in first_ordinal:last_ordinal
        flat = Int(batch.valid_flats[ordinal])
        state_slot, candidate = Ranking.state_candidate(flat, executor.width)
        row = batch.rows[state_slot]
        Model.forward_candidate!(
            @view(batch.raw[:, flat]),
            worker,
            executor.states[state_slot],
            executor.parameters,
            executor.cache,
            executor.dataset,
            row,
            candidate,
        )
        _record_forest_stats!(
            executor,
            worker_slot,
            worker.candidate_after.stats,
            true,
        )
        executor.diagnostic_counters[_OUTPUT_EVENTS, worker_slot] +=
            Model.Output.hard_event_count(worker.output_tape)
    end
    return nothing
end

@inline function _backward_state!(
    executor::BarrierlessExactExecutor,
    worker_slot::Int,
    state_slot::Int,
)
    batch = executor.batch
    row = @inbounds batch.rows[state_slot]
    state = @inbounds executor.states[state_slot]
    worker = @inbounds executor.workers[worker_slot]
    gradient = @inbounds executor.state_gradients[state_slot]
    Model.clear_gradient!(gradient)
    Model.prepare_state!(
        state,
        worker,
        executor.parameters,
        executor.cache,
        executor.dataset,
        row,
    )
    count = Int(@inbounds batch.counts[state_slot])
    offset = (state_slot - 1) * executor.width
    @inbounds for candidate in 1:count
        flat = offset + candidate
        Model.prepare_candidate!(
            worker,
            state,
            executor.parameters,
            executor.cache,
            executor.dataset,
            row,
            candidate,
        )
        Model.pullback_candidate!(
            gradient,
            @view(batch.raw[:, flat]),
            @view(batch.raw_gradient[:, flat]),
            worker,
            state,
            executor.parameters,
            executor.cache,
        )
    end
    Model.finish_state_pullback!(
        gradient,
        worker,
        state,
        executor.parameters,
        executor.cache,
    )
    return nothing
end

function SchedulerCore.dispatch_work!(
    executor::BarrierlessExactExecutor,
    worker_slot::Int,
    item::SchedulerCore.WorkItem,
)
    phase = item.phase
    first = Int(item.first)
    last = Int(item.last)
    if phase == _PREPARE_STATE
        @inbounds for state_slot in first:last
            _prepare_state!(executor, worker_slot, state_slot)
        end
    elseif phase == _FORWARD_CANDIDATE
        _forward_candidate_range!(executor, worker_slot, first, last)
    elseif phase == _BACKWARD_STATE
        @inbounds for state_slot in first:last
            _backward_state!(executor, worker_slot, state_slot)
        end
    else
        throw(ArgumentError("unknown DDF barrierless phase $phase"))
    end
    return nothing
end

"""Active process-wide worker team for repeated fixed-arena updates."""
mutable struct BarrierlessExactSession{E,S}
    executor::E
    scheduler::S
end

function run_executor_team!(
    body::F,
    executor::BarrierlessExactExecutor;
    workers::Integer=Base.Threads.nthreads(:default),
    queue_capacity::Integer=64,
    binding_mode::Symbol=:none,
) where {F}
    Int(workers) <= length(executor.workers) || throw(ArgumentError(
        "workers exceeds executor worker_capacity $(length(executor.workers))",
    ))
    scheduler = SchedulerCore.Scheduler(
        executor;
        workers,
        queue_capacity,
        binding_mode,
    )
    return SchedulerCore.run_team!(scheduler) do active_scheduler
        body(BarrierlessExactSession(executor, active_scheduler))
    end
end

@inline scheduler_report(session::BarrierlessExactSession) =
    SchedulerCore.teardown_report(session.scheduler)

@inline function _check_rows(executor::BarrierlessExactExecutor)
    @inbounds for state_slot in 1:executor.state_batch
        row = executor.batch.rows[state_slot]
        1 <= row <= executor.dataset.state_count || throw(BoundsError(
            1:executor.dataset.state_count,
            row,
        ))
    end
    return nothing
end

"""Evaluate every state-common plane and valid candidate through the team."""
function forward_batch!(session::BarrierlessExactSession)
    executor = session.executor
    _check_rows(executor)
    Ranking.prepare_batch_metadata!(executor.batch, executor.dataset)
    Model.refresh_cache!(executor.cache, executor.parameters)
    fill!(executor.diagnostic_counters, 0)
    SchedulerCore.run_phase!(
        session.scheduler,
        _PREPARE_STATE,
        1,
        executor.state_batch;
        chunk_size=1,
    )
    SchedulerCore.run_phase!(
        session.scheduler,
        _FORWARD_CANDIDATE,
        1,
        executor.batch.valid_count;
        chunk_size=executor.candidate_chunk_size,
    )
    return executor.batch.raw
end

"""Compute the complete 22-channel supervised cotangent."""
function loss_and_raw_gradient!(session::BarrierlessExactSession)
    executor = session.executor
    executor.loss_sink[] = Ranking.supervised_loss_and_raw_gradient!(
        executor.batch,
        executor.loss_scratch,
    )
    return executor.loss_sink[]
end

function loss_and_raw_gradient!(
    session::BarrierlessExactSession,
    sink::Base.RefValue{Ranking.SupervisedLoss},
)
    executor = session.executor
    sink[] = Ranking.supervised_loss_and_raw_gradient!(
        executor.batch,
        executor.loss_scratch,
    )
    return nothing
end

@inline latest_loss(session::BarrierlessExactSession) =
    session.executor.loss_sink[]

function latest_forward_diagnostics(session::BarrierlessExactSession)
    counters = session.executor.diagnostic_counters
    totals = ntuple(
        lane -> sum(@view(counters[lane, :])),
        _DIAGNOSTIC_COUNT,
    )
    return ForwardDiagnostics(totals...)
end

function clear_batch_gradient!(executor::BarrierlessExactExecutor)
    Model.clear_gradient!(executor.gradient)
    return executor.gradient
end

@inline function _add_array!(destination, source)
    @inbounds @simd for index in eachindex(destination, source)
        destination[index] += source[index]
    end
    return destination
end

@inline function _merge_sparse_program!(
    destination::Bank.SparseProgramGradient,
    source::Bank.SparseProgramGradient,
)
    @inbounds for slot in 1:Bank.active_gradient_count(source)
        row = Int(Bank.active_gradient_row(source, slot))
        Bank.accumulate_program_gradient!(
            destination,
            row,
            @view(source.values[:, slot]),
            1.0f0,
        )
    end
    return destination
end

@inline function _merge_gradient!(
    destination::Model.ModelGradient,
    source::Model.ModelGradient,
)
    _add_array!(destination.leaf_shared_raw, source.leaf_shared_raw)
    _merge_sparse_program!(destination.program, source.program)
    _add_array!(destination.forest.internal_raw, source.forest.internal_raw)
    _add_array!(
        destination.forest.child_contact,
        source.forest.child_contact,
    )
    Model.Output.accumulate_gradient!(destination.output, source.output)
    return destination
end

"""Exact per-state reverse followed by stable state-slot reduction."""
function backward_batch!(session::BarrierlessExactSession)
    executor = session.executor
    SchedulerCore.run_phase!(
        session.scheduler,
        _BACKWARD_STATE,
        1,
        executor.state_batch;
        chunk_size=1,
    )
    clear_batch_gradient!(executor)
    @inbounds for state_slot in 1:executor.state_batch
        _merge_gradient!(
            executor.gradient,
            executor.state_gradients[state_slot],
        )
    end
    return executor.gradient
end

"""Forward, supervised ListNet/auxiliary loss, exact reverse, and reduction."""
function forward_loss_backward!(session::BarrierlessExactSession)
    forward_batch!(session)
    loss = loss_and_raw_gradient!(session)
    backward_batch!(session)
    return loss
end

"""Allocation-free update form writing the loss into caller-owned storage."""
function forward_loss_backward!(
    session::BarrierlessExactSession,
    sink::Base.RefValue{Ranking.SupervisedLoss},
)
    forward_batch!(session)
    loss_and_raw_gradient!(session, sink)
    backward_batch!(session)
    return nothing
end

end # module CandidateDeltaDendriticBarrierless
