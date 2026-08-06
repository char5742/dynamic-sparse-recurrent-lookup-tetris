using Profile
using Test

include(joinpath(@__DIR__, "BarrierlessScheduler.jl"))
using .BarrierlessScheduler

const Threads = Base.Threads

mutable struct ExactOwner
    values::Vector{Int64}
    seen::Vector{Int}
end

function BarrierlessScheduler.dispatch_work!(
    owner::ExactOwner,
    worker_slot::Int,
    item::WorkItem,
)
    phase = Int(item.phase)
    @inbounds for index in Int(item.first):Int(item.last)
        if phase == 1
            owner.values[index] = Int64(index * index + 17)
            owner.seen[index] += 1
        elseif phase == 2
            owner.seen[index] == 1 || error("phase boundary was crossed")
            owner.values[index] = 3 * owner.values[index] - Int64(index)
            owner.seen[index] += 1
        else
            error("unexpected phase $phase")
        end
    end
    return nothing
end

function run_exact_case(worker_count::Int; count::Int=257)
    owner = ExactOwner(zeros(Int64, count), zeros(Int, count))
    scheduler = Scheduler(
        owner; workers=worker_count, queue_capacity=8, binding_mode=:none,
    )
    result = run_team!(scheduler) do active_scheduler
        run_phase!(active_scheduler, 9, 1, 0; chunk_size=1)
        run_phase!(active_scheduler, 1, 1, count; chunk_size=3)
        @test active_scheduler.remaining[] == 0
        @test BarrierlessScheduler.Queue.approx_length(
            active_scheduler.queue,
        ) == 0
        run_phase!(active_scheduler, 2, 1, count; chunk_size=5)
        return :completed
    end
    return owner, scheduler, result
end

mutable struct AllocationOwner
    values::Vector{Int}
    profiled_bytes::Int
    profiled_allocations::Int
    measured_phases::Int
end

function BarrierlessScheduler.dispatch_work!(
    owner::AllocationOwner,
    worker_slot::Int,
    item::WorkItem,
)
    phase = Int(item.phase)
    @inbounds for index in Int(item.first):Int(item.last)
        owner.values[index] = phase * index + worker_slot - worker_slot
    end
    return nothing
end

function measure_hot_phases!(scheduler::Scheduler, count::Int)
    # Warm every scheduler/queue path before enabling the allocation recorder.
    for _ in 1:8
        run_phase!(scheduler, 2, 1, count; chunk_size=7)
    end
    GC.gc()
    phases = 128
    Profile.Allocs.clear()
    Profile.Allocs.start(; sample_rate=1.0)
    try
        for _ in 1:phases
            run_phase!(scheduler, 2, 1, count; chunk_size=7)
        end
    finally
        Profile.Allocs.stop()
    end
    allocations = Profile.Allocs.fetch().allocs
    scheduler.owner.profiled_allocations = length(allocations)
    scheduler.owner.profiled_bytes = sum(
        allocation -> Int(allocation.size), allocations; init=0,
    )
    scheduler.owner.measured_phases = phases
    Profile.Allocs.clear()
    return nothing
end

struct DispatchSentinel <: Exception
    target::Int
end

Base.showerror(io::IO, exception::DispatchSentinel) =
    print(io, "dispatch sentinel at ", exception.target)

struct FailingOwner
    target::Int
end

function BarrierlessScheduler.dispatch_work!(
    owner::FailingOwner,
    worker_slot::Int,
    item::WorkItem,
)
    Int(item.first) <= owner.target <= Int(item.last) &&
        throw(DispatchSentinel(owner.target))
    return nothing
end

struct NoopOwner end

function BarrierlessScheduler.dispatch_work!(
    ::NoopOwner,
    worker_slot::Int,
    item::WorkItem,
)
    return nothing
end

function sources(exception::SchedulerFailure)
    return Set(entry.source for entry in exception.entries)
end

function binding_state_is_clear(scheduler::Scheduler)
    report = teardown_report(scheduler)
    return report.binding_plan_inactive &&
        report.binding_generations_clear &&
        BarrierlessScheduler.CpuSets.ACTIVE_PLAN[] === nothing &&
        all(iszero, BarrierlessScheduler.CpuSets.BOUND_GENERATION) &&
        isempty(
            BarrierlessScheduler.CpuSets._selected_cpu_sets_current_thread(),
        )
end

@testset "work item envelope" begin
    @test isbitstype(WorkItem)
    @test sizeof(WorkItem) == 16
    @test_throws ArgumentError Scheduler(NoopOwner(); queue_capacity=3)
    @test_throws ArgumentError Scheduler(NoopOwner(); workers=0)
end

@testset "serial and parallel equivalence" begin
    serial_owner, serial_scheduler, serial_result = run_exact_case(1)
    parallel_workers = min(4, Threads.nthreads(:default))
    parallel_owner, parallel_scheduler, parallel_result =
        run_exact_case(parallel_workers)

    expected = Int64[3 * (index * index + 17) - index for index in 1:257]
    @test serial_result === :completed
    @test parallel_result === :completed
    @test serial_owner.values == expected
    @test parallel_owner.values == expected
    @test serial_owner.values == parallel_owner.values
    @test all(==(2), serial_owner.seen)
    @test all(==(2), parallel_owner.seen)

    for scheduler in (serial_scheduler, parallel_scheduler)
        report = teardown_report(scheduler)
        @test report.queue_empty
        @test report.queue_closed
        @test report.remaining == 0
        @test report.active_dispatches == 0
        @test report.phase_idle
        @test report.binding_plan_inactive
        @test report.binding_generations_clear
        @test report.entered_threads == Threads.nthreads(:default)
        @test report.bound_threads == Threads.nthreads(:default)
        @test report.cleared_threads == Threads.nthreads(:default)
        @test report.exited_threads == Threads.nthreads(:default)
        @test report.coordinator_jobs >= 2
        @test binding_state_is_clear(scheduler)
        @test_throws ErrorException run_team!(scheduler) do _
            nothing
        end
    end
end

function run_process_exit_probe!()
    scheduler = Scheduler(
        NoopOwner(); workers=1, queue_capacity=4, binding_mode=:none,
    )
    result = run_team!(scheduler) do active_scheduler
        # More jobs than queue slots reproduces the original serial
        # backpressure deadlock if coordinator help/drain regresses.
        run_phase!(active_scheduler, 1, 1, 257; chunk_size=1)
        return :probe_complete
    end
    result === :probe_complete || error("exit probe lost its result")
    binding_state_is_clear(scheduler) || error(
        "exit probe retained CPU binding state",
    )
    return nothing
end

if "--exit-probe-child" in ARGS
    run_process_exit_probe!()
    exit(0)
end

@testset "coordinator participates and hot phase is allocation-free" begin
    worker_count = min(4, Threads.nthreads(:default))
    owner = AllocationOwner(zeros(Int, 1025), -1, -1, 0)
    scheduler = Scheduler(
        owner; workers=worker_count, queue_capacity=8, binding_mode=:none,
    )
    run_team!(scheduler) do active_scheduler
        measure_hot_phases!(active_scheduler, length(owner.values))
    end
    @test owner.values == [2 * index for index in eachindex(owner.values)]
    @test owner.measured_phases == 128
    @test owner.profiled_allocations == 0
    @test owner.profiled_bytes == 0
    @test teardown_report(scheduler).coordinator_jobs >= 2
end

@testset "worker failure propagates and teardown remains terminal" begin
    worker_count = min(4, Threads.nthreads(:default))
    scheduler = Scheduler(
        FailingOwner(9); workers=worker_count, queue_capacity=4,
        binding_mode=:none,
    )
    captured = try
        run_team!(scheduler) do active_scheduler
            run_phase!(active_scheduler, 1, 1, 128; chunk_size=1)
        end
        nothing
    catch exception
        exception
    end
    @test captured isa SchedulerFailure
    @test :worker in sources(captured)
    @test any(
        entry.exception isa DispatchSentinel for entry in captured.entries
    )
    report = teardown_report(scheduler)
    @test report.queue_empty
    @test report.queue_closed
    @test report.remaining == 0
    @test report.active_dispatches == 0
    @test report.phase_idle
    @test report.cleared_threads == Threads.nthreads(:default)
    @test report.exited_threads == Threads.nthreads(:default)
    @test binding_state_is_clear(scheduler)

    fresh = Scheduler(NoopOwner(); workers=1, queue_capacity=4)
    @test run_team!(fresh) do active_scheduler
        run_phase!(active_scheduler, 1, 1, 4; chunk_size=1)
        :fresh_team_completed
    end === :fresh_team_completed
end

@testset "coordinator failure is aggregated" begin
    scheduler = Scheduler(NoopOwner(); workers=1, queue_capacity=4)
    captured = try
        run_team!(scheduler) do _
            error("coordinator sentinel")
        end
        nothing
    catch exception
        exception
    end
    @test captured isa SchedulerFailure
    @test :coordinator in sources(captured)
    @test occursin("coordinator sentinel", sprint(showerror, captured))
    @test teardown_report(scheduler).queue_empty
    @test teardown_report(scheduler).cleared_threads ==
        Threads.nthreads(:default)
    @test binding_state_is_clear(scheduler)
end

@testset "teardown rejects a nonquiescent coordinator boundary" begin
    scheduler = Scheduler(NoopOwner(); workers=1, queue_capacity=4)
    captured = try
        run_team!(scheduler) do active_scheduler
            active_scheduler.remaining[] = 1
            nothing
        end
        nothing
    catch exception
        exception
    end
    @test captured isa SchedulerFailure
    @test :coordinator in sources(captured)
    @test occursin("quiescent phase boundary", sprint(showerror, captured))
    report = teardown_report(scheduler)
    @test report.queue_empty
    @test report.remaining == 0
    @test report.cleared_threads == Threads.nthreads(:default)
    @test binding_state_is_clear(scheduler)
end


@testset "failure published after dequeue prevents dispatch" begin
    owner = AllocationOwner(zeros(Int, 1), -1, -1, 0)
    scheduler = Scheduler(owner; workers=1, queue_capacity=4)
    scheduler.generation[] = UInt32(7)
    scheduler.remaining[] = 1
    item = WorkItem(UInt16(1), UInt32(7), Int32(1), Int32(1))

    @test BarrierlessScheduler.Queue.try_enqueue!(scheduler.queue, item)
    available, dequeued =
        BarrierlessScheduler.Queue.try_dequeue!(scheduler.queue)
    @test available
    Threads.atomic_xchg!(scheduler.shutdown_requested, UInt8(1))
    Threads.atomic_xchg!(scheduler.failure_published, UInt8(1))
    before = copy(owner.values)
    @test !BarrierlessScheduler._dispatch_dequeued_item!(
        scheduler, 1, dequeued,
    )
    @test owner.values == before
    @test scheduler.active_dispatches[] == 0
    @test scheduler.remaining[] == 1
    BarrierlessScheduler.Queue.close!(scheduler.queue)
end

@testset "nested teams and threaded regions are rejected" begin
    outer = Scheduler(NoopOwner(); workers=1, queue_capacity=4)
    inner = Scheduler(NoopOwner(); workers=1, queue_capacity=4)
    run_team!(outer) do active_scheduler
        @test_throws ErrorException run_team!(inner) do _
            nothing
        end
        @test_throws ErrorException run_team!(active_scheduler) do _
            nothing
        end
    end


    guarded = Scheduler(NoopOwner(); workers=1, queue_capacity=4)
    Threads.atomic_xchg!(
        BarrierlessScheduler.PROCESS_TEAM_ACTIVE, UInt8(1),
    )
    try
        @test_throws ErrorException run_team!(guarded) do _
            nothing
        end
    finally
        Threads.atomic_xchg!(
            BarrierlessScheduler.PROCESS_TEAM_ACTIVE, UInt8(0),
        )
    end
    @test run_team!(guarded) do _
        :atomic_guard_released
    end === :atomic_guard_released

    threaded_rejected = Threads.Atomic{Int}(0)
    Base.Threads.threading_run(worker_slot -> begin
        worker_slot == 1 || return nothing
        candidate = Scheduler(NoopOwner(); workers=1, queue_capacity=4)
        try
            run_team!(candidate) do _
                nothing
            end
        catch exception
            occursin("threaded region", sprint(showerror, exception)) &&
                Threads.atomic_add!(threaded_rejected, 1)
        end
        return nothing
    end, true)
    @test threaded_rejected[] == 1

    after_rejection = Scheduler(NoopOwner(); workers=1, queue_capacity=4)
    @test run_team!(after_rejection) do _
        :guard_released
    end === :guard_released
end


@testset "failure aggregation retains thread and teardown categories" begin
    scheduler = Scheduler(NoopOwner(); workers=1, queue_capacity=4)
    scheduler.thread_failures[1] =
        (ErrorException("thread failure"), nothing)
    scheduler.teardown_failures[1] =
        (ErrorException("teardown failure"), nothing)
    report = BarrierlessScheduler._teardown_report(scheduler)
    entries = BarrierlessScheduler._collect_failures(scheduler, report)
    categories = Set(entry.source for entry in entries)
    @test :thread in categories
    @test :teardown in categories
end


@testset "external serial-backpressure process exits" begin
    project_directory = dirname(Base.active_project())
    command = `$(Base.julia_cmd()) --project=$project_directory --threads=4,0 $(@__FILE__) --exit-probe-child`
    process = run(
        pipeline(command; stdout=devnull, stderr=devnull); wait=false,
    )
    deadline = time() + 15.0
    while Base.process_running(process) && time() < deadline
        sleep(0.01)
    end
    exited_before_deadline = !Base.process_running(process)
    if !exited_before_deadline
        # This is the exact child created above; never terminate by name.
        kill(process)
    end
    wait(process)
    @test exited_before_deadline
    @test exited_before_deadline && success(process)
end
