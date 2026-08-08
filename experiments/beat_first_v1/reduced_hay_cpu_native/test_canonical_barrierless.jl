using Test

module CanonicalBarrierlessTestHarness
include(joinpath(@__DIR__, "BarrierlessScheduler.jl"))
include(joinpath(@__DIR__, "CanonicalBarrierless.jl"))
end

const H = CanonicalBarrierlessTestHarness
const SchedulerCore = H.BarrierlessScheduler
const Parallel = H.CanonicalBarrierless
const Threads = Base.Threads

struct FakeBatch
    state_counts::Vector{Int}
    state_offsets::Vector{Int}
    raw::Vector{Int64}
    raw_gradient::Vector{Int64}
end

function FakeBatch(counts::Vector{Int})
    offsets = Vector{Int}(undef, length(counts) + 1)
    offsets[1] = 1
    @inbounds for state in eachindex(counts)
        offsets[state + 1] = offsets[state] + counts[state]
    end
    candidates = offsets[end] - 1
    return FakeBatch(
        copy(counts),
        offsets,
        zeros(Int64, candidates),
        zeros(Int64, candidates),
    )
end

mutable struct FakeArena
    worker_slot::Int
    accumulator::Int64
    candidate_owner::Int
end

mutable struct FakeAdapter <: Parallel.AbstractCanonicalGraphAdapter
    forward_seen::Vector{Int}
    replay_seen::Vector{Int}
    candidate_worker::Vector{Int}
    state_prepare_seen::Vector{Int}
    state_replay_seen::Vector{Int}
    partials::Vector{Int64}
    reduced::Int64
    parameter::Int64
    listnet_states::Int
    listnet_ready::Bool
    replay_before_listnet::Int
    updates::Int
    fail_ordinal::Int
end

function FakeAdapter(
    candidates::Int,
    states::Int,
    max_microbatches::Int;
    fail_ordinal::Int=0,
)
    return FakeAdapter(
        zeros(Int, candidates),
        zeros(Int, candidates),
        zeros(Int, candidates),
        zeros(Int, states),
        zeros(Int, states),
        zeros(Int64, max_microbatches + states),
        0,
        100_000,
        0,
        false,
        0,
        0,
        fail_ordinal,
    )
end

Parallel.create_worker_arena(::FakeAdapter, worker_slot::Int) =
    FakeArena(worker_slot, 0, 0)

Parallel.state_count(::FakeAdapter, batch::FakeBatch) =
    length(batch.state_counts)

Parallel.candidate_count(::FakeAdapter, batch::FakeBatch) =
    length(batch.raw)

function Parallel.state_candidate_bounds(
    ::FakeAdapter,
    batch::FakeBatch,
    state_slot::Int,
)
    first = @inbounds batch.state_offsets[state_slot]
    return first, @inbounds(batch.state_offsets[state_slot + 1] - 1)
end

function Parallel.prepare_batch!(adapter::FakeAdapter, batch::FakeBatch)
    fill!(adapter.forward_seen, 0)
    fill!(adapter.replay_seen, 0)
    fill!(adapter.candidate_worker, 0)
    fill!(adapter.state_prepare_seen, 0)
    fill!(adapter.state_replay_seen, 0)
    fill!(adapter.partials, 0)
    fill!(batch.raw, 0)
    fill!(batch.raw_gradient, 0)
    adapter.reduced = 0
    adapter.listnet_states = 0
    adapter.listnet_ready = false
    adapter.replay_before_listnet = 0
    return nothing
end

function Parallel.prepare_state_common!(
    adapter::FakeAdapter,
    ::FakeArena,
    ::FakeBatch,
    state::Int,
)
    @inbounds adapter.state_prepare_seen[state] += 1
    return nothing
end

function Parallel.finish_state_common_phase!(
    adapter::FakeAdapter,
    ::FakeBatch,
    state_total::Int,
)
    all(==(1), @view(adapter.state_prepare_seen[1:state_total])) || error(
        "state-common preparation phase was incomplete",
    )
    return nothing
end

function Parallel.begin_microbatch!(
    ::FakeAdapter,
    arena::FakeArena,
    ::FakeBatch,
    pass::UInt8,
    slot::Int,
    first::Int,
    last::Int,
)
    arena.accumulator = 0
    arena.candidate_owner = 0
    return nothing
end

function Parallel.run_candidate!(
    adapter::FakeAdapter,
    arena::FakeArena,
    batch::FakeBatch,
    ordinal::Int,
)
    ordinal == adapter.fail_ordinal && error("candidate failure sentinel")
    @inbounds adapter.forward_seen[ordinal] += 1
    @inbounds adapter.candidate_worker[ordinal] = arena.worker_slot

    # Simulate four synchronous event waves.  The owner check demonstrates
    # that no wave is resubmitted to another worker.
    arena.candidate_owner = arena.worker_slot
    value = Int64(ordinal)
    @inbounds for wave in 1:4
        arena.candidate_owner == arena.worker_slot || error(
            "candidate wave migrated between workers",
        )
        value = 3value + Int64(wave)
    end
    @inbounds batch.raw[ordinal] = value
    return nothing
end

function Parallel.finish_forward_microbatch!(
    ::FakeAdapter,
    ::FakeArena,
    ::FakeBatch,
    slot::Int,
    first::Int,
    last::Int,
)
    return nothing
end

function Parallel.finalize_listnet!(adapter::FakeAdapter, batch::FakeBatch)
    all(==(1), adapter.forward_seen) || error(
        "ListNet observed an incomplete candidate set",
    )
    @inbounds for state_slot in eachindex(batch.state_counts)
        first = batch.state_offsets[state_slot]
        last = batch.state_offsets[state_slot + 1] - 1
        count = last - first + 1
        total = Int64(0)
        for ordinal in first:last
            total += batch.raw[ordinal]
        end
        # Integer centering is sufficient for this orchestration oracle and
        # makes reduction order observable without floating-point ambiguity.
        for ordinal in first:last
            batch.raw_gradient[ordinal] =
                Int64(count) * batch.raw[ordinal] - total
        end
        adapter.listnet_states += 1
    end
    adapter.listnet_ready = true
    return nothing
end

function Parallel.replay_candidate!(
    adapter::FakeAdapter,
    arena::FakeArena,
    batch::FakeBatch,
    ordinal::Int,
)
    if !adapter.listnet_ready
        adapter.replay_before_listnet += 1
        error("candidate replay crossed the ListNet boundary")
    end
    @inbounds adapter.replay_seen[ordinal] += 1
    @inbounds arena.accumulator +=
        batch.raw_gradient[ordinal] * Int64(ordinal + 7)
    return nothing
end

function Parallel.reduce_worker!(
    adapter::FakeAdapter,
    arena::FakeArena,
    ::FakeBatch,
    microbatch_slot::Int,
    first::Int,
    last::Int,
)
    @inbounds adapter.partials[microbatch_slot] = arena.accumulator
    return nothing
end

function Parallel.replay_state_common!(
    adapter::FakeAdapter,
    ::FakeArena,
    ::FakeBatch,
    state::Int,
    reduction_slot::Int,
)
    @inbounds adapter.state_replay_seen[state] += 1
    @inbounds adapter.partials[reduction_slot] = Int64(10_000 * state)
    return nothing
end

function Parallel.deterministic_reduce!(
    adapter::FakeAdapter,
    batch::FakeBatch,
    microbatch_count::Int,
)
    total = Int64(0)
    total_slots = microbatch_count + length(batch.state_counts)
    @inbounds for slot in 1:total_slots
        total += adapter.partials[slot]
    end
    adapter.reduced = total
    return nothing
end

function Parallel.apply_update!(adapter::FakeAdapter, ::FakeBatch)
    adapter.parameter -= adapter.reduced
    adapter.updates += 1
    return adapter.parameter
end

function run_parallel(
    counts::Vector{Int};
    workers::Int,
    chunk::Int,
    fail_ordinal::Int=0,
)
    batch = FakeBatch(counts)
    adapter = FakeAdapter(
        length(batch.raw),
        length(batch.state_counts),
        cld(length(batch.raw), chunk);
        fail_ordinal,
    )
    executor = Parallel.CanonicalExecutor(
        adapter,
        batch;
        worker_capacity=workers,
        candidate_chunk_size=chunk,
    )
    session_sink = Ref{Any}(nothing)
    result = Parallel.run_executor_team!(
        executor;
        workers,
        queue_capacity=16,
    ) do session
        session_sink[] = session
        Parallel.train_update!(session)
    end
    return adapter, batch, executor, session_sink[], result
end

@testset "candidate ownership and state-complete ListNet boundary" begin
    workers = min(4, Threads.nthreads(:default))
    adapter, batch, executor, session, result = run_parallel(
        Int[3, 2, 4]; workers, chunk=2,
    )
    @test result == adapter.parameter
    @test adapter.forward_seen == ones(Int, 9)
    @test adapter.replay_seen == ones(Int, 9)
    @test adapter.state_prepare_seen == ones(Int, 3)
    @test adapter.state_replay_seen == ones(Int, 3)
    @test all(!iszero, adapter.candidate_worker)
    @test adapter.listnet_states == 3
    @test adapter.replay_before_listnet == 0
    @test executor.microbatch_total == 5

    report = Parallel.scheduler_report(session)
    @test report.entered_threads == Threads.nthreads(:default)
    @test report.exited_threads == Threads.nthreads(:default)
    @test report.queue_empty
    @test report.queue_closed
    @test report.remaining == 0
    @test report.active_dispatches == 0
    @test report.phase_idle
end

@testset "parallel matches serial and is scheduling deterministic" begin
    counts = Int[3, 2, 4]
    workers = min(4, Threads.nthreads(:default))

    serial_batch = FakeBatch(counts)
    serial_adapter = FakeAdapter(length(serial_batch.raw), length(counts), 5)
    serial_executor = Parallel.CanonicalExecutor(
        serial_adapter,
        serial_batch;
        worker_capacity=1,
        candidate_chunk_size=2,
    )
    serial_result = Parallel.serial_reference_update!(serial_executor)

    first = run_parallel(counts; workers, chunk=2)
    second = run_parallel(counts; workers, chunk=2)
    @test first[2].raw == serial_batch.raw
    @test first[2].raw_gradient == serial_batch.raw_gradient
    @test first[1].reduced == serial_adapter.reduced
    @test first[5] == serial_result
    @test second[2].raw == first[2].raw
    @test second[2].raw_gradient == first[2].raw_gradient
    @test second[1].reduced == first[1].reduced
    @test second[5] == first[5]

    # A different deterministic partition must also produce the same result
    # for this exact-integer oracle.
    different_chunk = run_parallel(counts; workers, chunk=3)
    @test different_chunk[2].raw == first[2].raw
    @test different_chunk[2].raw_gradient == first[2].raw_gradient
    @test different_chunk[1].reduced == first[1].reduced
    @test different_chunk[5] == first[5]
end

@testset "warmed canonical update allocates zero bytes" begin
    counts = Int[3, 2, 4]
    workers = min(4, Threads.nthreads(:default))
    batch = FakeBatch(counts)
    adapter = FakeAdapter(length(batch.raw), length(counts), 5)
    executor = Parallel.CanonicalExecutor(
        adapter,
        batch;
        worker_capacity=workers,
        candidate_chunk_size=2,
    )
    allocated = Ref{Int}(-1)
    Parallel.run_executor_team!(
        executor;
        workers,
        queue_capacity=16,
    ) do session
        Parallel.train_update!(session)
        Parallel.train_update!(session)
        allocated[] = @allocated Parallel.train_update!(session)
        nothing
    end
    @test allocated[] == 0
end

@testset "candidate failure tears down the persistent team" begin
    counts = Int[3, 2, 4]
    workers = min(4, Threads.nthreads(:default))
    batch = FakeBatch(counts)
    adapter = FakeAdapter(
        length(batch.raw), length(counts), 5; fail_ordinal=4,
    )
    executor = Parallel.CanonicalExecutor(
        adapter,
        batch;
        worker_capacity=workers,
        candidate_chunk_size=2,
    )
    captured = try
        Parallel.run_executor_team!(
            executor;
            workers,
            queue_capacity=16,
        ) do session
            Parallel.train_update!(session)
        end
        nothing
    catch exception
        exception
    end
    @test captured isa SchedulerCore.SchedulerFailure
    @test occursin("candidate failure sentinel", sprint(showerror, captured))
    report = captured.report
    @test report.queue_empty
    @test report.queue_closed
    @test report.remaining == 0
    @test report.active_dispatches == 0
    @test report.phase_idle
    @test report.entered_threads == Threads.nthreads(:default)
    @test report.exited_threads == Threads.nthreads(:default)
end

@testset "state candidate partition fails closed" begin
    batch = FakeBatch(Int[2, 2])
    batch.state_offsets[2] = 5
    adapter = FakeAdapter(4, 2, 2)
    executor = Parallel.CanonicalExecutor(
        adapter,
        batch;
        worker_capacity=1,
        candidate_chunk_size=2,
    )
    @test_throws ArgumentError Parallel.serial_reference_update!(executor)
end
