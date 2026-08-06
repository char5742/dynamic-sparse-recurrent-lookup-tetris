module BarrierlessScheduler

include(joinpath(
    @__DIR__, "..", "episodic_vit_recurrent_lookup", "bounded_mpmc_queue.jl",
))
include(joinpath(
    @__DIR__, "..", "episodic_vit_recurrent_lookup", "windows_cpu_sets.jl",
))

const Queue = BoundedMPMCRing
const CpuSets = WinCpuSets

export WorkItem, Scheduler, SchedulerFailure, TeardownReport,
       dispatch_work!, run_team!, run_phase!, teardown_report

"""A fixed-size queue envelope.  Phase zero is reserved for the empty value."""
struct WorkItem
    phase::UInt16
    generation::UInt32
    first::Int32
    last::Int32
end

Base.zero(::Type{WorkItem}) = WorkItem(UInt16(0), UInt32(0), Int32(0), Int32(-1))
isbitstype(WorkItem) || error("WorkItem must remain isbits")

"""Extend this method for the concrete owner stored by `Scheduler`."""
function dispatch_work! end

struct TeardownReport
    entered_threads::Int
    bound_threads::Int
    cleared_threads::Int
    exited_threads::Int
    executed_jobs::Int
    coordinator_jobs::Int
    queue_empty::Bool
    queue_closed::Bool
    remaining::Int
    active_dispatches::Int
    phase_idle::Bool
    binding_plan_inactive::Bool
    binding_generations_clear::Bool
end

struct FailureEntry
    source::Symbol
    worker_slot::Int
    exception::Any
    backtrace::Any
end

struct SchedulerFailure <: Exception
    entries::Vector{FailureEntry}
    report::TeardownReport
end

function Base.showerror(io::IO, failure::SchedulerFailure)
    print(io, "barrierless team failed with ", length(failure.entries), " error(s)")
    for entry in failure.entries
        print(io, "\n  [", entry.source)
        entry.worker_slot > 0 && print(io, " worker=", entry.worker_slot)
        print(io, "] ")
        showerror(io, entry.exception)
    end
end

struct TeamAborted <: Exception end

const PROCESS_TEAM_ACTIVE = Base.Threads.Atomic{UInt8}(0)

@inline _inside_threaded_region() =
    ccall(:jl_in_threaded_region, Cint, ()) != 0

mutable struct Scheduler{Owner}
    owner::Owner
    queue::Queue.BoundedMPMCQueue{WorkItem}
    worker_count::Int
    julia_thread_count::Int
    binding_mode::Symbol
    generation::Base.Threads.Atomic{UInt32}
    remaining::Base.Threads.Atomic{Int}
    active_dispatches::Base.Threads.Atomic{Int}
    phase_active::Base.Threads.Atomic{UInt8}
    shutdown_requested::Base.Threads.Atomic{UInt8}
    failure_published::Base.Threads.Atomic{UInt8}
    entered_threads::Base.Threads.Atomic{Int}
    ready_workers::Base.Threads.Atomic{Int}
    bound_threads::Base.Threads.Atomic{Int}
    cleared_threads::Base.Threads.Atomic{Int}
    exited_threads::Base.Threads.Atomic{Int}
    executed_jobs::Base.Threads.Atomic{Int}
    jobs_by_worker::Vector{Base.Threads.Atomic{Int}}
    worker_failures::Vector{Any}
    thread_failures::Vector{Any}
    teardown_failures::Vector{Any}
    binding_plan_failure::Base.RefValue{Any}
    coordinator_failure::Base.RefValue{Any}
    threading_failure::Base.RefValue{Any}
    startup_event::Base.Event
    started::Bool
    finished::Bool
    final_report::Union{Nothing,TeardownReport}
end

function Scheduler(
    owner::Owner;
    workers::Integer=Base.Threads.nthreads(:default),
    queue_capacity::Integer=1024,
    binding_mode::Symbol=:none,
) where {Owner}
    worker_count = Int(workers)
    julia_thread_count = Base.Threads.nthreads(:default)
    worker_count >= 1 || throw(ArgumentError("workers must be positive"))
    worker_count <= julia_thread_count || throw(ArgumentError(
        "workers exceeds the Julia default thread count",
    ))
    Base.Threads.nthreads(:interactive) == 0 || throw(ArgumentError(
        "the interactive Julia thread pool must be disabled",
    ))
    capacity = Int(queue_capacity)
    capacity >= 2 || throw(ArgumentError("queue_capacity must be at least two"))
    ispow2(capacity) || throw(ArgumentError(
        "queue_capacity must be a power of two",
    ))
    binding_mode in (:none, :all, :p_only) || throw(ArgumentError(
        "binding_mode must be :none, :all, or :p_only",
    ))
    return Scheduler{Owner}(
        owner,
        Queue.BoundedMPMCQueue{WorkItem}(capacity, zero(WorkItem)),
        worker_count,
        julia_thread_count,
        binding_mode,
        Base.Threads.Atomic{UInt32}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{UInt8}(0),
        Base.Threads.Atomic{UInt8}(0),
        Base.Threads.Atomic{UInt8}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        [Base.Threads.Atomic{Int}(0) for _ in 1:worker_count],
        Any[nothing for _ in 1:worker_count],
        Any[nothing for _ in 1:julia_thread_count],
        Any[nothing for _ in 1:julia_thread_count],
        Ref{Any}(nothing),
        Ref{Any}(nothing),
        Ref{Any}(nothing),
        Base.Event(true),
        false,
        false,
        nothing,
    )
end

@inline _failed(scheduler::Scheduler) = !iszero(scheduler.failure_published[])

@inline function _publish_abort!(scheduler::Scheduler)
    Base.Threads.atomic_xchg!(scheduler.failure_published, UInt8(1))
    Base.Threads.atomic_xchg!(scheduler.shutdown_requested, UInt8(1))
    notify(scheduler.startup_event)
    Queue.close!(scheduler.queue)
    return nothing
end

function _record_worker_failure!(
    scheduler::Scheduler,
    worker_slot::Int,
    exception,
    backtrace,
)
    @inbounds scheduler.worker_failures[worker_slot] = (exception, backtrace)
    _publish_abort!(scheduler)
    return nothing
end

function _record_thread_failure!(
    scheduler::Scheduler,
    worker_slot::Int,
    exception,
    backtrace,
)
    @inbounds scheduler.thread_failures[worker_slot] = (exception, backtrace)
    _publish_abort!(scheduler)
    return nothing
end

function _record_teardown_failure!(
    scheduler::Scheduler,
    worker_slot::Int,
    exception,
    backtrace,
)
    @inbounds scheduler.teardown_failures[worker_slot] = (exception, backtrace)
    _publish_abort!(scheduler)
    return nothing
end

@inline function _complete_item!(scheduler::Scheduler)
    previous = Base.Threads.atomic_add!(scheduler.remaining, -1)
    if previous <= 0
        _record_worker_failure!(
            scheduler,
            1,
            ErrorException("remaining work counter underflow"),
            backtrace(),
        )
    elseif previous == 1
        Queue.wake_consumers!(scheduler.queue)
    end
    return nothing
end

@inline function _execute_item!(
    scheduler::Scheduler,
    worker_slot::Int,
    item::WorkItem,
)
    item.phase != 0 || begin
        _record_worker_failure!(
            scheduler, worker_slot,
            ErrorException("phase zero cannot be dispatched"), backtrace(),
        )
        return false
    end
    item.generation == scheduler.generation[] || begin
        _record_worker_failure!(
            scheduler, worker_slot,
            ErrorException("stale work generation"), backtrace(),
        )
        return false
    end
    item.first <= item.last || begin
        _record_worker_failure!(
            scheduler, worker_slot,
            ErrorException("invalid work range"), backtrace(),
        )
        return false
    end

    Base.Threads.atomic_add!(scheduler.active_dispatches, 1)
    # This second check closes the dequeue-to-claim race.  A dispatch that
    # passes it is logically active before any later failure publication.
    if !iszero(scheduler.shutdown_requested[]) || _failed(scheduler)
        Base.Threads.atomic_add!(scheduler.active_dispatches, -1)
        return false
    end
    succeeded = false
    try
        dispatch_work!(scheduler.owner, worker_slot, item)
        succeeded = true
    catch exception
        _record_worker_failure!(
            scheduler, worker_slot, exception, catch_backtrace(),
        )
    finally
        Base.Threads.atomic_add!(scheduler.active_dispatches, -1)
    end
    succeeded || return false
    Base.Threads.atomic_add!(scheduler.executed_jobs, 1)
    Base.Threads.atomic_add!(@inbounds(scheduler.jobs_by_worker[worker_slot]), 1)
    _complete_item!(scheduler)
    return true
end

"""Claim a dequeued item only while the team remains dispatchable."""
@inline function _dispatch_dequeued_item!(
    scheduler::Scheduler,
    worker_slot::Int,
    item::WorkItem,
)
    if !iszero(scheduler.shutdown_requested[]) || _failed(scheduler)
        return false
    end
    return _execute_item!(scheduler, worker_slot, item)
end

function _worker_loop!(scheduler::Scheduler, worker_slot::Int)
    while iszero(scheduler.shutdown_requested[])
        available, item = Queue.dequeue_wait!(scheduler.queue; timeout_ms=100)
        if available
            # Failure can be published while this worker is waking/dequeuing.
            # Recheck after ownership transfer and immediately before dispatch.
            _dispatch_dequeued_item!(scheduler, worker_slot, item) ||
                return nothing
        elseif Queue.isclosed(scheduler.queue)
            return nothing
        end
    end
    return nothing
end

@inline function _next_generation!(scheduler::Scheduler)
    current = scheduler.generation[]
    current == typemax(UInt32) && error("work generation overflow")
    next = current + UInt32(1)
    scheduler.generation[] = next
    return next
end

@inline function _publish_or_help!(
    scheduler::Scheduler,
    item::WorkItem,
)
    while !Queue.try_enqueue!(scheduler.queue, item)
        _failed(scheduler) && throw(TeamAborted())
        Queue.isclosed(scheduler.queue) && error(
            "work queue closed during phase publication",
        )
        available, queued_item = Queue.try_dequeue!(scheduler.queue)
        if available
            _dispatch_dequeued_item!(scheduler, 1, queued_item) ||
                throw(TeamAborted())
        else
            GC.safepoint()
        end
    end
    return nothing
end

@inline function _checked_i32(value::Int, label::String)
    typemin(Int32) <= value <= typemax(Int32) || throw(ArgumentError(
        "$label does not fit in Int32",
    ))
    return Int32(value)
end

"""
    run_phase!(scheduler, phase, first, last; chunk_size)

Run one flat range.  All chunks except the final chunk are published before
worker slot one begins its reserved chunk.  This keeps coordinator
participation deterministic without forcing every other worker to idle for
one full chunk at the start of each phase.
"""
function run_phase!(
    scheduler::Scheduler,
    phase::Integer,
    first::Integer,
    last::Integer;
    chunk_size::Integer,
)
    scheduler.started || error("the worker team is not running")
    scheduler.finished && error("the worker team has already been torn down")
    Base.Threads.threadid() == 1 || error(
        "only coordinator worker slot one may start a phase",
    )
    _failed(scheduler) && throw(TeamAborted())
    phase_value = Int(phase)
    1 <= phase_value <= typemax(UInt16) || throw(ArgumentError(
        "phase must be in 1:$(typemax(UInt16))",
    ))
    first_value = Int(first)
    last_value = Int(last)
    chunk = Int(chunk_size)
    chunk >= 1 || throw(ArgumentError("chunk_size must be positive"))
    first_value <= last_value || return nothing
    _checked_i32(first_value, "first")
    _checked_i32(last_value, "last")
    Base.Threads.atomic_cas!(
        scheduler.phase_active, UInt8(0), UInt8(1),
    ) == 0 || error("nested or overlapping scheduler phase rejected")

    try
        Queue.approx_length(scheduler.queue) == 0 || error(
            "queue is not empty at the phase boundary",
        )
        scheduler.remaining[] == 0 || error(
            "remaining work is nonzero at the phase boundary",
        )
        generation = _next_generation!(scheduler)
        span = last_value - first_value + 1
        job_count = cld(span, chunk)
        scheduler.remaining[] = job_count

        chunk_first = first_value
        reserved_item = zero(WorkItem)
        while chunk_first <= last_value
            _failed(scheduler) && throw(TeamAborted())
            remaining_span = last_value - chunk_first + 1
            chunk_last = chunk >= remaining_span ?
                last_value : chunk_first + chunk - 1
            item = WorkItem(
                UInt16(phase_value), generation,
                _checked_i32(chunk_first, "first"),
                _checked_i32(chunk_last, "last"),
            )
            if chunk_last == last_value
                reserved_item = item
            else
                _publish_or_help!(scheduler, item)
            end
            chunk_first = chunk_last + 1
        end
        reserved_item.phase == UInt16(phase_value) || error(
            "failed to reserve the coordinator phase chunk",
        )
        _execute_item!(scheduler, 1, reserved_item) || throw(TeamAborted())

        while scheduler.remaining[] != 0
            _failed(scheduler) && throw(TeamAborted())
            available, item = Queue.try_dequeue!(scheduler.queue)
            if available
                _dispatch_dequeued_item!(scheduler, 1, item) ||
                    throw(TeamAborted())
                continue
            end
            expected = Queue.item_epoch(scheduler.queue)
            scheduler.remaining[] == 0 && break
            _failed(scheduler) && throw(TeamAborted())
            Queue.isclosed(scheduler.queue) && error(
                "work queue closed before phase completion",
            )
            Queue.wait_for_item_change!(
                scheduler.queue, expected; timeout_ms=100,
            )
        end
        _failed(scheduler) && throw(TeamAborted())
        scheduler.active_dispatches[] == 0 || error(
            "phase completed while a dispatch remained active",
        )
        Queue.approx_length(scheduler.queue) == 0 || error(
            "phase completed with queued work",
        )
        return nothing
    finally
        scheduler.phase_active[] = UInt8(0)
    end
end

function _teardown_report(scheduler::Scheduler)
    return TeardownReport(
        scheduler.entered_threads[],
        scheduler.bound_threads[],
        scheduler.cleared_threads[],
        scheduler.exited_threads[],
        scheduler.executed_jobs[],
        @inbounds(scheduler.jobs_by_worker[1][]),
        Queue.approx_length(scheduler.queue) == 0,
        Queue.isclosed(scheduler.queue),
        scheduler.remaining[],
        scheduler.active_dispatches[],
        iszero(scheduler.phase_active[]),
        CpuSets.ACTIVE_PLAN[] === nothing,
        all(iszero, CpuSets.BOUND_GENERATION),
    )
end

teardown_report(scheduler::Scheduler) = scheduler.final_report === nothing ?
    _teardown_report(scheduler) : scheduler.final_report

function _teardown_invariant_errors(
    scheduler::Scheduler,
    report::TeardownReport,
)
    errors = FailureEntry[]
    expected_threads = scheduler.julia_thread_count
    report.entered_threads == expected_threads || push!(errors, FailureEntry(
        :invariant, 0,
        ErrorException("not every Julia thread entered the team"), nothing,
    ))
    report.bound_threads == expected_threads || push!(errors, FailureEntry(
        :invariant, 0,
        ErrorException("not every Julia thread completed CPU binding"), nothing,
    ))
    report.cleared_threads == expected_threads || push!(errors, FailureEntry(
        :invariant, 0,
        ErrorException("not every Julia thread cleared CPU binding"), nothing,
    ))
    report.exited_threads == expected_threads || push!(errors, FailureEntry(
        :invariant, 0,
        ErrorException("not every Julia thread exited the team"), nothing,
    ))
    report.queue_empty || push!(errors, FailureEntry(
        :invariant, 0, ErrorException("queue is not empty after teardown"), nothing,
    ))
    report.queue_closed || push!(errors, FailureEntry(
        :invariant, 0, ErrorException("queue is not closed after teardown"), nothing,
    ))
    report.remaining == 0 || push!(errors, FailureEntry(
        :invariant, 0,
        ErrorException("remaining work is nonzero after teardown"), nothing,
    ))
    report.active_dispatches == 0 || push!(errors, FailureEntry(
        :invariant, 0,
        ErrorException("a dispatch is active after teardown"), nothing,
    ))
    report.phase_idle || push!(errors, FailureEntry(
        :invariant, 0, ErrorException("phase is active after teardown"), nothing,
    ))
    report.binding_plan_inactive || push!(errors, FailureEntry(
        :invariant, 0,
        ErrorException("CPU binding plan remains active after teardown"), nothing,
    ))
    report.binding_generations_clear || push!(errors, FailureEntry(
        :invariant, 0,
        ErrorException("CPU binding generation remains live after teardown"), nothing,
    ))
    return errors
end

"""
Deactivate the process-global CPU binding plan after every native thread has
cleared and verified its own affinity.  `WinCpuSets` has no public plan-release
operation, so this is the single narrowly scoped access to its logical state.
The scheduler process guard remains held until both postconditions are true.
"""
function _deactivate_cpu_binding_plan!()
    CpuSets.ACTIVE_PLAN[] = nothing
    fill!(CpuSets.BOUND_GENERATION, UInt64(0))
    CpuSets.ACTIVE_PLAN[] === nothing || error(
        "CPU binding plan deactivation failed",
    )
    all(iszero, CpuSets.BOUND_GENERATION) || error(
        "CPU binding generation reset failed",
    )
    return nothing
end

function _drain_aborted_queue!(scheduler::Scheduler)
    while true
        available, _ = Queue.try_dequeue!(scheduler.queue)
        available || break
    end
    scheduler.remaining[] = 0
    scheduler.phase_active[] = UInt8(0)
    return nothing
end

function _collect_failures(
    scheduler::Scheduler,
    report::TeardownReport,
)
    failures = FailureEntry[]
    for worker_slot in eachindex(scheduler.worker_failures)
        payload = @inbounds scheduler.worker_failures[worker_slot]
        payload === nothing || push!(failures, FailureEntry(
            :worker, worker_slot, payload[1], payload[2],
        ))
    end
    for worker_slot in eachindex(scheduler.thread_failures)
        payload = @inbounds scheduler.thread_failures[worker_slot]
        payload === nothing || push!(failures, FailureEntry(
            :thread, worker_slot, payload[1], payload[2],
        ))
    end
    coordinator = scheduler.coordinator_failure[]
    coordinator === nothing || push!(failures, FailureEntry(
        :coordinator, 1, coordinator[1], coordinator[2],
    ))
    threading = scheduler.threading_failure[]
    threading === nothing || push!(failures, FailureEntry(
        :threading_runtime, 0, threading[1], threading[2],
    ))
    for worker_slot in eachindex(scheduler.teardown_failures)
        payload = @inbounds scheduler.teardown_failures[worker_slot]
        payload === nothing || push!(failures, FailureEntry(
            :teardown, worker_slot, payload[1], payload[2],
        ))
    end
    binding_plan = scheduler.binding_plan_failure[]
    binding_plan === nothing || push!(failures, FailureEntry(
        :teardown, 0, binding_plan[1], binding_plan[2],
    ))
    append!(failures, _teardown_invariant_errors(scheduler, report))
    return failures
end

"""
    run_team!(body, scheduler)

Open the single process-wide native-worker region.  Slot one runs `body` and
also executes work submitted by `run_phase!`; all bindings are cleared and
verified before this function returns or throws.
"""
function run_team!(body::Body, scheduler::Scheduler) where {Body}
    scheduler.started && error("this scheduler is already running")
    scheduler.finished && error("this scheduler is one-shot and already finished")
    Queue.isclosed(scheduler.queue) && error("scheduler queue is already closed")
    _inside_threaded_region() && error(
        "starting a worker team inside an existing threaded region is rejected",
    )
    result = Ref{Any}(nothing)
    acquired = Base.Threads.atomic_cas!(
        PROCESS_TEAM_ACTIVE, UInt8(0), UInt8(1),
    ) == 0
    acquired || error("nested or concurrent process-wide worker team rejected")
    scheduler.started = true

    try
        CpuSets.configure_worker_bindings(
            scheduler.binding_mode, scheduler.worker_count,
        )
        try
            Base.Threads.threading_run(worker_slot -> begin
                Base.Threads.atomic_add!(scheduler.entered_threads, 1)
                notify(scheduler.startup_event)
                try
                    CpuSets.bind_current_worker!(worker_slot)
                    Base.Threads.atomic_add!(scheduler.bound_threads, 1)
                    if worker_slot <= scheduler.worker_count
                        Base.Threads.atomic_add!(scheduler.ready_workers, 1)
                    end
                    notify(scheduler.startup_event)

                    if worker_slot == 1
                        while scheduler.entered_threads[] < scheduler.julia_thread_count ||
                              scheduler.ready_workers[] < scheduler.worker_count
                            _failed(scheduler) && throw(TeamAborted())
                            wait(scheduler.startup_event)
                        end
                        try
                            result[] = body(scheduler)
                        catch exception
                            if !(exception isa TeamAborted && _failed(scheduler))
                                scheduler.coordinator_failure[] =
                                    (exception, catch_backtrace())
                                _publish_abort!(scheduler)
                            end
                        end
                        if !_failed(scheduler) && (
                            scheduler.remaining[] != 0 ||
                            scheduler.active_dispatches[] != 0 ||
                            scheduler.phase_active[] != 0 ||
                            Queue.approx_length(scheduler.queue) != 0
                        )
                            scheduler.coordinator_failure[] = (
                                ErrorException(
                                    "coordinator returned outside a quiescent phase boundary",
                                ),
                                backtrace(),
                            )
                            _publish_abort!(scheduler)
                        end
                        Base.Threads.atomic_xchg!(
                            scheduler.shutdown_requested, UInt8(1),
                        )
                        Queue.close!(scheduler.queue)
                    elseif worker_slot <= scheduler.worker_count
                        _worker_loop!(scheduler, worker_slot)
                    end
                catch exception
                    if !(exception isa TeamAborted && _failed(scheduler))
                        _record_thread_failure!(
                            scheduler, worker_slot, exception, catch_backtrace(),
                        )
                    end
                finally
                    try
                        CpuSets.clear_current_binding!()
                        Base.Threads.atomic_add!(scheduler.cleared_threads, 1)
                    catch exception
                        _record_teardown_failure!(
                            scheduler, worker_slot, exception, catch_backtrace(),
                        )
                    end
                    Base.Threads.atomic_add!(scheduler.exited_threads, 1)
                    notify(scheduler.startup_event)
                end
                return nothing
            end, true)
        catch exception
            scheduler.threading_failure[] = (exception, catch_backtrace())
            _publish_abort!(scheduler)
        end
    catch exception
        scheduler.threading_failure[] = (exception, catch_backtrace())
        _publish_abort!(scheduler)
    finally
        Queue.close!(scheduler.queue)
        _drain_aborted_queue!(scheduler)
        scheduler.started = false
        scheduler.finished = true
        try
            _deactivate_cpu_binding_plan!()
        catch exception
            scheduler.binding_plan_failure[] =
                (exception, catch_backtrace())
            Base.Threads.atomic_xchg!(
                scheduler.failure_published, UInt8(1),
            )
        finally
            try
                # Capture logical binding postconditions while this scheduler
                # still owns the process-global team guard.
                scheduler.final_report = _teardown_report(scheduler)
            finally
                Base.Threads.atomic_xchg!(PROCESS_TEAM_ACTIVE, UInt8(0))
            end
        end
    end

    report = scheduler.final_report
    report === nothing && error("teardown report was not captured")
    failures = _collect_failures(scheduler, report)
    isempty(failures) || throw(SchedulerFailure(failures, report))
    return result[]
end

run_team!(scheduler::Scheduler, body::Body) where {Body} =
    run_team!(body, scheduler)

end # module BarrierlessScheduler
