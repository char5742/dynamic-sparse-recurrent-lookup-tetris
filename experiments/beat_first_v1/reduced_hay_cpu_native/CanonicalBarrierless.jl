module CanonicalBarrierless

using ..BarrierlessScheduler

const SchedulerCore = BarrierlessScheduler

export AbstractCanonicalGraphAdapter,
       CanonicalExecutor,
       CanonicalSession,
       FORWARD_PASS,
       PREPARE_COMMON_PASS,
       REPLAY_PASS,
       REPLAY_COMMON_PASS,
       apply_update!,
       begin_microbatch!,
       candidate_count,
       create_worker_arena,
       deterministic_reduce!,
       finalize_listnet!,
       finish_state_common_phase!,
       finish_forward_microbatch!,
       prepare_batch!,
       prepare_state_common!,
       reduce_worker!,
       replay_candidate!,
       replay_state_common!,
       run_candidate!,
       run_executor_team!,
       scheduler_report,
       serial_reference_update!,
       state_candidate_bounds,
       state_count,
       train_update!

const PREPARE_COMMON_PASS = UInt8(1)
const FORWARD_PASS = UInt8(2)
const REPLAY_PASS = UInt8(3)
const REPLAY_COMMON_PASS = UInt8(4)

const _PREPARE_COMMON_STATES = UInt16(1)
const _FORWARD_CANDIDATES = UInt16(2)
const _REPLAY_CANDIDATES = UInt16(3)
const _REPLAY_COMMON_STATES = UInt16(4)

"""
Adapter boundary between the persistent MPMC team and `CanonicalDendriticGraph`.

The graph module owns every numerical object: the batch, state/candidate map,
worker arena, hard-event waves, ListNet scratch, deterministic reduction slots,
optimizer and metrics.  This module owns only the execution order.  Concrete
adapters must implement every exported hook below; there are intentionally no
fallback implementations which could silently skip part of an update.
"""
abstract type AbstractCanonicalGraphAdapter end

function create_worker_arena end
function prepare_batch! end
function prepare_state_common! end
function finish_state_common_phase! end
function state_count end
function candidate_count end
function state_candidate_bounds end
function begin_microbatch! end
function run_candidate! end
function finish_forward_microbatch! end
function finalize_listnet! end
function replay_candidate! end
function replay_state_common! end
function reduce_worker! end
function deterministic_reduce! end
function apply_update! end

"""
Persistent, fixed-memory owner submitted to `BarrierlessScheduler`.

One scheduler item is one deterministic candidate microbatch.  A native worker
that claims the item retains its worker-local arena for every mandatory DAG
step and every hard-event refinement wave of every candidate in that item.
There are no wave-level jobs and therefore no inter-wave global barrier.
"""
mutable struct CanonicalExecutor{A<:AbstractCanonicalGraphAdapter,B,W}
    adapter::A
    batch::B
    workers::Vector{W}
    candidate_chunk_size::Int
    state_total::Int
    candidate_total::Int
    microbatch_total::Int
    update_active::Bool
end

function CanonicalExecutor(
    adapter::A,
    batch::B;
    worker_capacity::Integer=Base.Threads.nthreads(:default),
    candidate_chunk_size::Integer=4,
) where {A<:AbstractCanonicalGraphAdapter,B}
    capacity = Int(worker_capacity)
    capacity >= 1 || throw(ArgumentError("worker_capacity must be positive"))
    chunk = Int(candidate_chunk_size)
    chunk >= 1 || throw(ArgumentError(
        "candidate_chunk_size must be positive",
    ))

    first_worker = create_worker_arena(adapter, 1)
    workers = Vector{typeof(first_worker)}(undef, capacity)
    workers[1] = first_worker
    @inbounds for worker_slot in 2:capacity
        workers[worker_slot] = create_worker_arena(adapter, worker_slot)
    end
    return CanonicalExecutor{A,B,typeof(first_worker)}(
        adapter,
        batch,
        workers,
        chunk,
        0,
        0,
        0,
        false,
    )
end

"""Validate that state-local candidates form one exact ordinal partition."""
function _validate_candidate_partition!(executor::CanonicalExecutor)
    adapter = executor.adapter
    batch = executor.batch
    states = Int(state_count(adapter, batch))
    candidates = Int(candidate_count(adapter, batch))
    states >= 1 || throw(ArgumentError("a batch must contain at least one state"))
    candidates >= 1 || throw(ArgumentError(
        "a batch must contain at least one candidate",
    ))

    expected_first = 1
    @inbounds for state_slot in 1:states
        first, last = state_candidate_bounds(adapter, batch, state_slot)
        first_value = Int(first)
        last_value = Int(last)
        first_value == expected_first || throw(ArgumentError(
            "state candidate ranges must be contiguous and ordered",
        ))
        last_value >= first_value || throw(ArgumentError(
            "every state must own at least one candidate",
        ))
        last_value <= candidates || throw(ArgumentError(
            "state candidate range exceeds candidate_count",
        ))
        expected_first = last_value + 1
    end
    expected_first == candidates + 1 || throw(ArgumentError(
        "state candidate ranges do not cover candidate_count",
    ))

    executor.state_total = states
    executor.candidate_total = candidates
    executor.microbatch_total = cld(candidates, executor.candidate_chunk_size)
    return nothing
end

@inline function _prepare_common_state!(
    executor::CanonicalExecutor,
    worker_slot::Int,
    first::Int,
    last::Int,
)
    first == last || error("state-common preparation jobs must be indivisible")
    worker = @inbounds executor.workers[worker_slot]
    prepare_state_common!(
        executor.adapter, worker, executor.batch, first,
    )
    return nothing
end

@inline function _microbatch_slot(
    executor::CanonicalExecutor,
    first::Int,
    last::Int,
)
    chunk = executor.candidate_chunk_size
    slot = fld(first - 1, chunk) + 1
    first == (slot - 1) * chunk + 1 || error(
        "scheduler emitted a noncanonical microbatch start",
    )
    last == min(slot * chunk, executor.candidate_total) || error(
        "scheduler emitted a noncanonical microbatch end",
    )
    return slot
end

@inline function _replay_common_state!(
    executor::CanonicalExecutor,
    worker_slot::Int,
    first::Int,
    last::Int,
)
    first == last || error("state-common replay jobs must be indivisible")
    state_slot = first
    reduction_slot = executor.microbatch_total + state_slot
    worker = @inbounds executor.workers[worker_slot]
    replay_state_common!(
        executor.adapter,
        worker,
        executor.batch,
        state_slot,
        reduction_slot,
    )
    return nothing
end

@inline function _forward_microbatch!(
    executor::CanonicalExecutor,
    worker_slot::Int,
    first::Int,
    last::Int,
)
    adapter = executor.adapter
    batch = executor.batch
    worker = @inbounds executor.workers[worker_slot]
    slot = _microbatch_slot(executor, first, last)
    begin_microbatch!(
        adapter, worker, batch, FORWARD_PASS, slot, first, last,
    )
    @inbounds for ordinal in first:last
        # This call owns the complete candidate trajectory, including all
        # synchronous hard-event waves.  It must not publish nested work.
        run_candidate!(adapter, worker, batch, ordinal)
    end
    finish_forward_microbatch!(adapter, worker, batch, slot, first, last)
    return nothing
end

@inline function _replay_microbatch!(
    executor::CanonicalExecutor,
    worker_slot::Int,
    first::Int,
    last::Int,
)
    adapter = executor.adapter
    batch = executor.batch
    worker = @inbounds executor.workers[worker_slot]
    slot = _microbatch_slot(executor, first, last)
    begin_microbatch!(
        adapter, worker, batch, REPLAY_PASS, slot, first, last,
    )
    @inbounds for ordinal in first:last
        # Replay regenerates eligibility in chronological order.  The adapter
        # writes only worker-local state until `reduce_worker!` commits this
        # deterministic microbatch slot.
        replay_candidate!(adapter, worker, batch, ordinal)
    end
    reduce_worker!(adapter, worker, batch, slot, first, last)
    return nothing
end

function SchedulerCore.dispatch_work!(
    executor::CanonicalExecutor,
    worker_slot::Int,
    item::SchedulerCore.WorkItem,
)
    first = Int(item.first)
    last = Int(item.last)
    if item.phase == _PREPARE_COMMON_STATES
        _prepare_common_state!(executor, worker_slot, first, last)
    elseif item.phase == _FORWARD_CANDIDATES
        _forward_microbatch!(executor, worker_slot, first, last)
    elseif item.phase == _REPLAY_CANDIDATES
        _replay_microbatch!(executor, worker_slot, first, last)
    elseif item.phase == _REPLAY_COMMON_STATES
        _replay_common_state!(executor, worker_slot, first, last)
    else
        throw(ArgumentError(
            "unknown canonical barrierless phase $(item.phase)",
        ))
    end
    return nothing
end

"""Active view of a one-shot persistent native-worker team."""
struct CanonicalSession{E,S}
    executor::E
    scheduler::S
end

function run_executor_team!(
    body::F,
    executor::CanonicalExecutor;
    workers::Integer=Base.Threads.nthreads(:default),
    queue_capacity::Integer=64,
    binding_mode::Symbol=:none,
) where {F}
    worker_count = Int(workers)
    worker_count <= length(executor.workers) || throw(ArgumentError(
        "workers exceeds executor worker_capacity $(length(executor.workers))",
    ))
    scheduler = SchedulerCore.Scheduler(
        executor;
        workers=worker_count,
        queue_capacity,
        binding_mode,
    )
    return SchedulerCore.run_team!(scheduler) do active_scheduler
        body(CanonicalSession(executor, active_scheduler))
    end
end

@inline scheduler_report(session::CanonicalSession) =
    SchedulerCore.teardown_report(session.scheduler)

"""
Run one complete canonical update in the persistent team.

The two global mathematical boundaries are explicit:

1. every state-common prefix is complete before candidate forward;
2. every candidate forward is complete before `finalize_listnet!`; and
3. every candidate and state-common replay is committed before ascending-slot
   `deterministic_reduce!` and `apply_update!`.

`finalize_listnet!` must use `state_candidate_bounds` and finalize each state
only from its complete candidate set.  `reduce_worker!` must copy/sum the
worker-local accumulator into the supplied deterministic `microbatch_slot`;
it must not update parameters.  `deterministic_reduce!` consumes those slots
in ascending order, independently of which native worker produced them.
"""
function train_update!(session::CanonicalSession)
    executor = session.executor
    Base.Threads.threadid() == 1 || error(
        "only coordinator worker slot one may start an update",
    )
    executor.update_active && error("nested canonical update rejected")
    executor.update_active = true
    try
        prepare_batch!(executor.adapter, executor.batch)
        _validate_candidate_partition!(executor)
        SchedulerCore.run_phase!(
            session.scheduler,
            _PREPARE_COMMON_STATES,
            1,
            executor.state_total;
            chunk_size=1,
        )
        finish_state_common_phase!(
            executor.adapter, executor.batch, executor.state_total,
        )
        SchedulerCore.run_phase!(
            session.scheduler,
            _FORWARD_CANDIDATES,
            1,
            executor.candidate_total;
            chunk_size=executor.candidate_chunk_size,
        )

        # This call is deliberately outside the worker phase.  No state may
        # observe a partial candidate list when its ListNet delta is formed.
        finalize_listnet!(executor.adapter, executor.batch)

        SchedulerCore.run_phase!(
            session.scheduler,
            _REPLAY_CANDIDATES,
            1,
            executor.candidate_total;
            chunk_size=executor.candidate_chunk_size,
        )
        SchedulerCore.run_phase!(
            session.scheduler,
            _REPLAY_COMMON_STATES,
            1,
            executor.state_total;
            chunk_size=1,
        )

        deterministic_reduce!(
            executor.adapter,
            executor.batch,
            executor.microbatch_total,
        )
        return apply_update!(executor.adapter, executor.batch)
    finally
        executor.update_active = false
    end
end

"""
Diagnostic serial orchestration oracle using the identical adapter hooks.

This is not an alternative production trainer.  It exists only to prove that
the MPMC execution preserves the canonical forward/ListNet/replay/reduction
boundaries.  The same deterministic microbatch partition is retained.
"""
function serial_reference_update!(executor::CanonicalExecutor)
    executor.update_active && error("nested canonical update rejected")
    executor.update_active = true
    try
        prepare_batch!(executor.adapter, executor.batch)
        _validate_candidate_partition!(executor)
        @inbounds for state_slot in 1:executor.state_total
            _prepare_common_state!(executor, 1, state_slot, state_slot)
        end
        finish_state_common_phase!(
            executor.adapter, executor.batch, executor.state_total,
        )
        candidates = executor.candidate_total
        chunk = executor.candidate_chunk_size
        first = 1
        while first <= candidates
            last = min(first + chunk - 1, candidates)
            _forward_microbatch!(executor, 1, first, last)
            first = last + 1
        end
        finalize_listnet!(executor.adapter, executor.batch)
        first = 1
        while first <= candidates
            last = min(first + chunk - 1, candidates)
            _replay_microbatch!(executor, 1, first, last)
            first = last + 1
        end
        @inbounds for state_slot in 1:executor.state_total
            _replay_common_state!(executor, 1, state_slot, state_slot)
        end
        deterministic_reduce!(
            executor.adapter,
            executor.batch,
            executor.microbatch_total,
        )
        return apply_update!(executor.adapter, executor.batch)
    finally
        executor.update_active = false
    end
end

end # module CanonicalBarrierless
