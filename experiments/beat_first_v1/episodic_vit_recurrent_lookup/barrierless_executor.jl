"""
Barrierless execution core for the EVRL four-state teacher update.

This file is included inside `EpisodicViTRecurrentLookupTeacherTraining`; it
deliberately has no module wrapper.  `Queue` must name `BoundedMPMCRing` and
`CpuSets`, `Model`, the teacher loss helpers, and `_prepare_flat_jobs!` must be
defined by the including module.

The executor owns scheduling state only.  Model and optimizer objects are not
stored in checkpoints, and model parameters remain read-only until the caller
has waited for all four states to become gradient-ready.
"""

@enum BarrierlessJobKind::UInt8 begin
    BARRIERLESS_NO_JOB = 0
    BARRIERLESS_PREPARE = 1
    BARRIERLESS_FORWARD_STEP = 2
    BARRIERLESS_STATE_LOSS_VJP = 3
    BARRIERLESS_BACKWARD = 4
end

@enum BarrierlessCandidatePhase::UInt8 begin
    BARRIERLESS_CANDIDATE_IDLE = 0
    BARRIERLESS_CANDIDATE_PREPARE = 1
    BARRIERLESS_CANDIDATE_FORWARD = 2
    BARRIERLESS_CANDIDATE_WAIT_BACKWARD = 3
    BARRIERLESS_CANDIDATE_DONE = 4
end

@enum BarrierlessStatePhase::UInt8 begin
    BARRIERLESS_STATE_IDLE = 0
    BARRIERLESS_STATE_FORWARD = 1
    BARRIERLESS_STATE_LOSS = 2
    BARRIERLESS_STATE_BACKWARD = 3
    BARRIERLESS_STATE_GRADIENT_READY = 4
end

"""An allocation-free queue payload.  `generation` rejects stale work."""
struct BarrierlessJob
    kind::UInt8
    target::UInt16
    generation::UInt32
end

BarrierlessJob(
    kind::BarrierlessJobKind, target::Integer, generation::UInt32,
) = BarrierlessJob(UInt8(kind), UInt16(target), generation)

Base.zero(::Type{BarrierlessJob}) = BarrierlessJob(
    UInt8(BARRIERLESS_NO_JOB), UInt16(0), UInt32(0),
)

isbitstype(BarrierlessJob) || error("BarrierlessJob must remain isbits")

mutable struct BarrierlessCandidateRuntime
    state_slot::Int16
    candidate::Int16
    phase::UInt8
    trajectory::Union{Nothing,Model.ForwardTrajectoryState}
end

BarrierlessCandidateRuntime() = BarrierlessCandidateRuntime(
    Int16(0), Int16(0), UInt8(BARRIERLESS_CANDIDATE_IDLE), nothing,
)

mutable struct BarrierlessStateRuntime
    candidate_count::Int16
    remaining_forward::Base.Threads.Atomic{Int}
    remaining_backward::Base.Threads.Atomic{Int}
    phase::Base.Threads.Atomic{UInt8}
    candidate_jobs::Vector{UInt16}
    backward_jobs::Vector{BarrierlessJob}
    depth_values::Vector{Int}
end

function BarrierlessStateRuntime()
    return BarrierlessStateRuntime(
        Int16(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{UInt8}(UInt8(BARRIERLESS_STATE_IDLE)),
        zeros(UInt16, LEARNER_WIDTH),
        fill(zero(BarrierlessJob), LEARNER_WIDTH),
        zeros(Int, LEARNER_WIDTH),
    )
end

const BARRIERLESS_PHASE_PREPARE = 1
const BARRIERLESS_PHASE_FORWARD = 2
const BARRIERLESS_PHASE_LOSS_VJP = 3
const BARRIERLESS_PHASE_BACKWARD = 4
const BARRIERLESS_PHASE_QUEUE_WAIT = 5
const BARRIERLESS_PHASE_COUNT = 5

# Benchmark phase IDs are deliberately separate from the worker-local phase
# IDs above.  The first five are measured by the persistent workers; data
# packing, reduction, and optimizer work are coordinator-owned phases.  GC is
# reported from Julia's cumulative GC clock at the exact update boundary.
const BARRIERLESS_BENCH_DATA_PACK = 1
const BARRIERLESS_BENCH_PREPARE = 2
const BARRIERLESS_BENCH_FORWARD = 3
const BARRIERLESS_BENCH_QUEUE_WAIT = 4
const BARRIERLESS_BENCH_LOSS_VJP = 5
const BARRIERLESS_BENCH_BACKWARD = 6
const BARRIERLESS_BENCH_GRADIENT_REDUCE = 7
const BARRIERLESS_BENCH_OPTIMIZER = 8
const BARRIERLESS_BENCH_GC = 9
const BARRIERLESS_BENCH_PHASE_COUNT = 9

const BARRIERLESS_BENCH_PHASE_NAMES = (
    :data_pack,
    :prepare,
    :recurrent_forward,
    :queue_wait,
    :loss_vjp,
    :backward,
    :gradient_reduce,
    :optimizer,
    :gc,
)

const BARRIERLESS_WORKER_TO_BENCH_PHASE = (
    BARRIERLESS_BENCH_PREPARE,
    BARRIERLESS_BENCH_FORWARD,
    BARRIERLESS_BENCH_LOSS_VJP,
    BARRIERLESS_BENCH_BACKWARD,
    BARRIERLESS_BENCH_QUEUE_WAIT,
)

# Allocation counters exposed by Julia's GC are process-wide.  They are exact
# for coordinator-owned, non-overlapping intervals, but cannot be partitioned
# between concurrently executing worker phases without double-counting.  Keep
# that distinction explicit in benchmark output instead of reporting the zero
# storage slots for worker phases as measured zero allocation.
@inline function _barrierless_phase_allocation_measurement(phase::Int)
    if phase == BARRIERLESS_BENCH_DATA_PACK ||
            phase == BARRIERLESS_BENCH_GRADIENT_REDUCE ||
            phase == BARRIERLESS_BENCH_OPTIMIZER
        return :process_gc_delta
    elseif phase == BARRIERLESS_BENCH_GC
        return :update_total_gc_delta
    end
    return :unavailable_parallel_worker_phase
end

@inline function _barrierless_sampled_allocation_phase(stacktrace)
    optimizer = false
    gradient_reduce = false
    backward = false
    recurrent_forward = false
    @inbounds for frame in stacktrace
        name = String(frame.func)
        optimizer |= startswith(name, "_barrierless_optimize_") ||
            name == "_barrierless_direct_sparse_adam_step!" ||
            name == "_barrierless_enqueue_optimizer_work!" ||
            startswith(name, "_adam_update_") ||
            name == "optimizer_step!"
        gradient_reduce |= startswith(name, "_barrierless_reduce_") ||
            startswith(name, "_barrierless_scale_") ||
            name == "_barrierless_prepare_bank_rows!" ||
            name == "_barrierless_enqueue_reduction_work!" ||
            name == "_barrierless_enqueue_scale_work!" ||
            name == "gradient_norm" ||
            name == "scale_gradients!"
        backward |= name == "backward_trajectory!" ||
            name == "_barrierless_backward!" ||
            occursin("_vjp!", name)
        recurrent_forward |= name == "prepare_trajectory" ||
            name == "advance_trajectory!" ||
            name == "finalize_trajectory" ||
            name == "forward_trajectory" ||
            name == "_barrierless_prepare!" ||
            name == "_barrierless_forward_step!" ||
            occursin("_forward", name)
    end
    optimizer && return :optimizer
    gradient_reduce && return :gradient_reduce
    backward && return :backward
    recurrent_forward && return :recurrent_forward
    return :other
end

# Julia 1.12 records an allocation stack with the allocator/runtime frames at
# the leaf followed by the Julia caller.  Report the first actionable Julia
# source frame, while retaining a valid Julia frame as a fallback for samples
# that never leave Base/stdlib code.
@inline function _barrierless_is_runtime_allocation_frame(file::String)
    normalized = lowercase(replace(file, '\\' => '/'))
    return startswith(normalized, "./") ||
        occursin("/workdir/src/", normalized) ||
        occursin("/workdir/base/", normalized) ||
        occursin("/share/julia/base/", normalized) ||
        occursin("/share/julia/stdlib/", normalized)
end

function _barrierless_sampled_allocation_site(stacktrace)
    fallback = nothing
    @inbounds for frame in stacktrace
        frame.from_c && continue
        line = Int(frame.line)
        line > 0 || continue
        function_name = String(frame.func)
        file = String(frame.file)
        isempty(function_name) && continue
        isempty(file) && continue
        fallback === nothing && (fallback = (function_name, file, line))
        _barrierless_is_runtime_allocation_frame(file) && continue
        return (function_name, file, line)
    end
    return fallback === nothing ? ("<unknown>", "<unknown>", -1) : fallback
end

@inline function _barrierless_allocation_site_precedes(left, right)
    left_values = last(left)
    right_values = last(right)
    left_values[1] != right_values[1] &&
        return left_values[1] > right_values[1]
    left_values[2] != right_values[2] &&
        return left_values[2] > right_values[2]
    return first(left) < first(right)
end

"""Decode one sampled warmup update into non-overlapping allocation phases."""
function summarize_barrierless_allocation_profile(
    profile;
    sample_rate::Float64,
    update::Int,
)
    0.0 < sample_rate <= 1.0 || error("allocation sample rate must be in (0,1]")
    forward_bytes = UInt128(0)
    backward_bytes = UInt128(0)
    reduce_bytes = UInt128(0)
    optimizer_bytes = UInt128(0)
    other_bytes = UInt128(0)
    forward_count = 0
    backward_count = 0
    reduce_count = 0
    optimizer_count = 0
    other_count = 0
    allocation_sites = Dict{
        Tuple{String,String,Int,String},
        Tuple{UInt128,Int},
    }()
    for allocation in profile.allocs
        bytes = UInt128(allocation.size)
        phase = _barrierless_sampled_allocation_phase(allocation.stacktrace)
        if phase === :recurrent_forward
            forward_bytes += bytes
            forward_count += 1
        elseif phase === :backward
            backward_bytes += bytes
            backward_count += 1
        elseif phase === :gradient_reduce
            reduce_bytes += bytes
            reduce_count += 1
        elseif phase === :optimizer
            optimizer_bytes += bytes
            optimizer_count += 1
        else
            other_bytes += bytes
            other_count += 1
        end
        function_name, file, line =
            _barrierless_sampled_allocation_site(allocation.stacktrace)
        site_key = (function_name, file, line, String(phase))
        previous = get(allocation_sites, site_key, (UInt128(0), 0))
        allocation_sites[site_key] =
            (previous[1] + bytes, previous[2] + 1)
    end
    sampled_bytes = (;
        recurrent_forward=forward_bytes,
        backward=backward_bytes,
        gradient_reduce=reduce_bytes,
        optimizer=optimizer_bytes,
        other=other_bytes,
    )
    sampled_allocations = (;
        recurrent_forward=forward_count,
        backward=backward_count,
        gradient_reduce=reduce_count,
        optimizer=optimizer_count,
        other=other_count,
    )
    estimated_bytes_per_update = map(
        bytes -> Float64(bytes) / sample_rate,
        sampled_bytes,
    )
    ranked_sites = collect(allocation_sites)
    sort!(ranked_sites; lt=_barrierless_allocation_site_precedes)
    top_site_count = min(length(ranked_sites), 20)
    top_allocation_sites = map(@view ranked_sites[1:top_site_count]) do entry
        key = first(entry)
        values = last(entry)
        return NamedTuple{
            (:function, :file, :line, :category, :sampled_bytes, :count),
        }((key[1], key[2], key[3], key[4], values[1], values[2]))
    end
    return (;
        provenance="Profile.Allocs sampled warmup update",
        update,
        sample_rate,
        sampled_allocation_count=length(profile.allocs),
        sampled_allocations,
        sampled_bytes,
        estimated_bytes_per_update,
        top_allocation_sites,
    )
end

"""
Measurement-only state.  It is owned by `BarrierlessExecutor`, which is never
serialized in a v1 training checkpoint, so enabling or resetting counters does
not change checkpoint or optimizer-state compatibility.

`phase_wall_nanoseconds` for worker phases is aggregate native-worker wall
time.  This is intentional: asynchronous phases overlap, so there is no unique
calendar-time partition.  The summary also returns capacity-normalized phase
wall time (`aggregate / active_workers`) and the exact executor calendar wall.
"""
mutable struct BarrierlessBenchmarkStatistics
    enabled::Base.Threads.Atomic{UInt32}
    measured_updates::UInt64
    overall_wall_nanoseconds::UInt128
    overall_cpu_ticks_100ns::UInt128
    executor_wall_nanoseconds::UInt128
    executor_cpu_ticks_100ns::UInt128
    allocation_bytes::UInt128
    executor_allocation_bytes::UInt128
    gc_nanoseconds::UInt128
    phase_wall_nanoseconds::Vector{UInt128}
    phase_cpu_ticks_100ns::Vector{UInt128}
    phase_allocation_bytes::Vector{UInt128}
    phase_gc_nanoseconds::Vector{UInt128}
    jobs::UInt128
    chunks::UInt128
    update_open::Bool
    executor_open::Bool
    update_wall_started::UInt64
    update_cpu_started::UInt64
    update_gc_started::Base.GC_Num
    executor_wall_started::UInt64
    executor_cpu_started::UInt64
    executor_gc_started::Base.GC_Num
end

function BarrierlessBenchmarkStatistics()
    gc_snapshot = Base.gc_num()
    return BarrierlessBenchmarkStatistics(
        Base.Threads.Atomic{UInt32}(UInt32(0)),
        UInt64(0), UInt128(0), UInt128(0), UInt128(0), UInt128(0),
        UInt128(0), UInt128(0), UInt128(0),
        zeros(UInt128, BARRIERLESS_BENCH_PHASE_COUNT),
        zeros(UInt128, BARRIERLESS_BENCH_PHASE_COUNT),
        zeros(UInt128, BARRIERLESS_BENCH_PHASE_COUNT),
        zeros(UInt128, BARRIERLESS_BENCH_PHASE_COUNT),
        UInt128(0), UInt128(0), false, false,
        UInt64(0), UInt64(0), gc_snapshot,
        UInt64(0), UInt64(0), gc_snapshot,
    )
end

"""An allocation-free coordinator phase measurement token."""
struct BarrierlessPhaseMeasurement
    phase::UInt8
    active::Bool
    wall_started::UInt64
    cpu_started::UInt64
    gc_started::Base.GC_Num
end

@inline function _barrierless_phase_id(phase::Symbol)
    phase === :data_pack && return BARRIERLESS_BENCH_DATA_PACK
    phase === :prepare && return BARRIERLESS_BENCH_PREPARE
    phase === :recurrent_forward && return BARRIERLESS_BENCH_FORWARD
    phase === :queue_wait && return BARRIERLESS_BENCH_QUEUE_WAIT
    phase === :loss_vjp && return BARRIERLESS_BENCH_LOSS_VJP
    phase === :backward && return BARRIERLESS_BENCH_BACKWARD
    phase === :gradient_reduce && return BARRIERLESS_BENCH_GRADIENT_REDUCE
    phase === :optimizer && return BARRIERLESS_BENCH_OPTIMIZER
    phase === :gc && return BARRIERLESS_BENCH_GC
    throw(ArgumentError("unknown barrierless benchmark phase $phase"))
end

@inline function _barrierless_gc_diff(started::Base.GC_Num)
    return Base.GC_Diff(Base.gc_num(), started)
end

mutable struct BarrierlessWorkerStatistics
    jobs::UInt64
    chunks::UInt64
    phase_wall_nanoseconds::Vector{UInt128}
    phase_cpu_ticks_100ns::Vector{UInt128}
end

BarrierlessWorkerStatistics() = BarrierlessWorkerStatistics(
    0, 0, zeros(UInt128, BARRIERLESS_PHASE_COUNT),
    zeros(UInt128, BARRIERLESS_PHASE_COUNT),
)

function reset_barrierless_statistics!(statistics::BarrierlessWorkerStatistics)
    statistics.jobs = 0
    statistics.chunks = 0
    fill!(statistics.phase_wall_nanoseconds, 0)
    fill!(statistics.phase_cpu_ticks_100ns, 0)
    return statistics
end

mutable struct BarrierlessWorkerRuntime
    dequeue_buffer::Vector{BarrierlessJob}
    continuation_buffer::Vector{BarrierlessJob}
    statistics::BarrierlessWorkerStatistics
end

function BarrierlessWorkerRuntime(maximum_chunk::Int)
    return BarrierlessWorkerRuntime(
        fill(zero(BarrierlessJob), maximum_chunk),
        fill(zero(BarrierlessJob), maximum_chunk),
        BarrierlessWorkerStatistics(),
    )
end

"""Immutable, update-owned values observed by every worker."""
struct BarrierlessEpochContext{T,B,H}
    trainer::T
    batches::B
    expected_update::Int
    hyperparameters::H
    baseline::Float32
    temperature::Float32
    generation::UInt32
end

"""
Persistent native-worker team and the scheduling-only state of one update.

The caller supplies exactly one gradient accumulator per active worker.  Those
accumulators are single-owner while the DAG is live and must be merged in
worker-slot order only after `wait_barrierless_gradients!` returns.
"""
mutable struct BarrierlessExecutor
    queue::Queue.BoundedMPMCQueue{BarrierlessJob}
    active_workers::Int
    julia_workers::Int
    fixed_chunk_size::Int
    adaptive_tail::Bool
    candidates::Vector{BarrierlessCandidateRuntime}
    states::Vector{BarrierlessStateRuntime}
    workers::Vector{BarrierlessWorkerRuntime}
    worker_accumulators::Vector{Model.GradientAccumulator}
    ffn_scale_contributions::Matrix{Float32}
    initial_jobs::Vector{BarrierlessJob}
    state_results::Vector{Any}
    epoch_context::Base.RefValue{Any}
    generation::Base.Threads.Atomic{UInt32}
    forward_active::Base.Threads.Atomic{Int}
    gradient_ready_states::Base.Threads.Atomic{Int}
    active_dispatches::Base.Threads.Atomic{Int}
    boundary_workers::Base.Threads.Atomic{Int}
    update_inflight::Base.Threads.Atomic{UInt32}
    shutdown_requested::Base.Threads.Atomic{UInt32}
    failure_worker::Base.Threads.Atomic{Int}
    worker_failures::Vector{Any}
    ready_workers::Base.Threads.Atomic{Int}
    booted_workers::Base.Threads.Atomic{Int}
    startup_event::Base.Event
    worker_bindings::Vector{Any}
    benchmark_statistics::BarrierlessBenchmarkStatistics
    started::Bool
end

function BarrierlessExecutor(
    worker_accumulators::Vector{Model.GradientAccumulator};
    active_workers::Int=length(worker_accumulators),
    fixed_chunk_size::Int=8,
    adaptive_tail::Bool=false,
    queue_capacity::Int=1024,
)
    active_workers >= 1 || throw(ArgumentError("active_workers must be positive"))
    length(worker_accumulators) == active_workers || throw(ArgumentError(
        "one worker-local accumulator is required per active worker",
    ))
    fixed_chunk_size in (8, 16, 32) || throw(ArgumentError(
        "fixed chunk size must be 8, 16, or 32",
    ))
    ispow2(queue_capacity) || throw(ArgumentError("queue capacity must be a power of two"))
    queue_capacity >= 2 * MAX_FLAT_CANDIDATES || throw(ArgumentError(
        "queue needs headroom for control and backward fanout",
    ))
    julia_workers = Base.Threads.nthreads(:default)
    active_workers <= julia_workers || throw(ArgumentError(
        "active worker count exceeds the Julia default pool",
    ))
    maximum_chunk = max(fixed_chunk_size, 8)
    return BarrierlessExecutor(
        Queue.BoundedMPMCQueue{BarrierlessJob}(
            queue_capacity, zero(BarrierlessJob),
        ),
        active_workers,
        julia_workers,
        fixed_chunk_size,
        adaptive_tail,
        [BarrierlessCandidateRuntime() for _ in 1:MAX_FLAT_CANDIDATES],
        [BarrierlessStateRuntime() for _ in 1:TRAINING_STATE_BATCH],
        [BarrierlessWorkerRuntime(maximum_chunk) for _ in 1:active_workers],
        worker_accumulators,
        zeros(Float32, Model.MAX_RECURRENT_STEPS, MAX_FLAT_CANDIDATES),
        fill(zero(BarrierlessJob), MAX_FLAT_CANDIDATES),
        Any[nothing for _ in 1:TRAINING_STATE_BATCH],
        Ref{Any}(nothing),
        Base.Threads.Atomic{UInt32}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{UInt32}(0),
        Base.Threads.Atomic{UInt32}(0),
        Base.Threads.Atomic{Int}(0),
        Any[nothing for _ in 1:julia_workers],
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Event(true),
        Any[nothing for _ in 1:julia_workers],
        BarrierlessBenchmarkStatistics(),
        false,
    )
end

@inline function _barrierless_chunk_size(executor::BarrierlessExecutor)
    executor.adaptive_tail || return executor.fixed_chunk_size
    active = executor.forward_active[]
    return active >= 160 ? 8 : active >= 80 ? 4 : active >= 40 ? 2 : 1
end

@inline function _barrierless_record_phase!(
    statistics::BarrierlessWorkerStatistics,
    phase::Int,
    wall_started::UInt64,
    cpu_started::UInt64,
)
    statistics.phase_wall_nanoseconds[phase] += UInt128(time_ns() - wall_started)
    statistics.phase_cpu_ticks_100ns[phase] += UInt128(
        CpuSets.thread_cpu_ticks_100ns() - cpu_started,
    )
    return nothing
end

@inline _barrierless_measurement_enabled(executor::BarrierlessExecutor) =
    executor.benchmark_statistics.enabled[] != 0

@inline function _barrierless_nonnegative_u128(value::Integer)
    value >= 0 || error("barrierless measurement counter moved backwards")
    return UInt128(value)
end

"""
Reset every benchmark counter at an optimizer/update boundary.

Call this immediately after the ten compile-warmup updates.  Passing
`enabled=true` makes the next completed update measurement update 1; passing
`enabled=false` removes per-job clock calls from the full-training hot path.
"""
function reset_barrierless_benchmark_statistics!(
    executor::BarrierlessExecutor;
    enabled::Bool=true,
)
    executor.update_inflight[] == 0 || error(
        "benchmark counters may be reset only at an update boundary",
    )
    statistics = executor.benchmark_statistics
    statistics.update_open && error("benchmark update measurement is still open")
    statistics.executor_open && error("executor measurement is still open")
    statistics.measured_updates = 0
    statistics.overall_wall_nanoseconds = 0
    statistics.overall_cpu_ticks_100ns = 0
    statistics.executor_wall_nanoseconds = 0
    statistics.executor_cpu_ticks_100ns = 0
    statistics.allocation_bytes = 0
    statistics.executor_allocation_bytes = 0
    statistics.gc_nanoseconds = 0
    fill!(statistics.phase_wall_nanoseconds, 0)
    fill!(statistics.phase_cpu_ticks_100ns, 0)
    fill!(statistics.phase_allocation_bytes, 0)
    fill!(statistics.phase_gc_nanoseconds, 0)
    statistics.jobs = 0
    statistics.chunks = 0
    for worker in executor.workers
        reset_barrierless_statistics!(worker.statistics)
    end
    Base.Threads.atomic_xchg!(
        statistics.enabled, enabled ? UInt32(1) : UInt32(0),
    )
    return statistics
end

function disable_barrierless_benchmark_statistics!(executor::BarrierlessExecutor)
    return reset_barrierless_benchmark_statistics!(executor; enabled=false)
end

"""Start a coordinator-owned phase such as reduction or optimizer."""
function begin_barrierless_phase_measurement(
    executor::BarrierlessExecutor,
    phase::Symbol,
)
    phase_id = _barrierless_phase_id(phase)
    active = _barrierless_measurement_enabled(executor)
    if !active
        return BarrierlessPhaseMeasurement(
            UInt8(phase_id), false, UInt64(0), UInt64(0),
            executor.benchmark_statistics.update_gc_started,
        )
    end
    executor.benchmark_statistics.update_open || error(
        "phase measurement requires an open benchmark update",
    )
    return BarrierlessPhaseMeasurement(
        UInt8(phase_id), true, time_ns(),
        CpuSets.process_cpu_ticks_100ns(), Base.gc_num(),
    )
end

"""Finish a coordinator-owned phase token exactly once."""
function finish_barrierless_phase_measurement!(
    executor::BarrierlessExecutor,
    measurement::BarrierlessPhaseMeasurement,
)
    measurement.active || return nothing
    _barrierless_measurement_enabled(executor) || error(
        "benchmark measurement was disabled while a phase was open",
    )
    phase = Int(measurement.phase)
    1 <= phase <= BARRIERLESS_BENCH_PHASE_COUNT || error(
        "invalid barrierless benchmark phase $phase",
    )
    statistics = executor.benchmark_statistics
    wall_finished = time_ns()
    cpu_finished = CpuSets.process_cpu_ticks_100ns()
    gc_delta = _barrierless_gc_diff(measurement.gc_started)
    statistics.phase_wall_nanoseconds[phase] += _barrierless_nonnegative_u128(
        wall_finished - measurement.wall_started,
    )
    statistics.phase_cpu_ticks_100ns[phase] += _barrierless_nonnegative_u128(
        cpu_finished - measurement.cpu_started,
    )
    statistics.phase_allocation_bytes[phase] += _barrierless_nonnegative_u128(
        gc_delta.allocd,
    )
    statistics.phase_gc_nanoseconds[phase] += _barrierless_nonnegative_u128(
        gc_delta.total_time,
    )
    return nothing
end

@inline function _barrierless_begin_update_measurement!(executor)
    _barrierless_measurement_enabled(executor) || return nothing
    statistics = executor.benchmark_statistics
    statistics.update_open && error("benchmark update measurement is already open")
    statistics.executor_open && error("stale executor measurement is still open")
    statistics.update_wall_started = time_ns()
    statistics.update_cpu_started = CpuSets.process_cpu_ticks_100ns()
    statistics.update_gc_started = Base.gc_num()
    statistics.update_open = true
    return nothing
end


"""
Open overall update measurement before sampler/data packing.

This optional public hook lets the CLI include `next_batch!` and
`pack_batch!` in the exact GetProcessTimes interval.  If it is not used,
`begin_barrierless_update!` opens the interval itself for compatibility.
"""
function begin_barrierless_update_measurement!(executor::BarrierlessExecutor)
    executor.update_inflight[] == 0 || error(
        "external update measurement must begin before work publication",
    )
    _barrierless_begin_update_measurement!(executor)
    return nothing
end

barrierless_update_measurement_open(executor::BarrierlessExecutor) =
    executor.benchmark_statistics.update_open

@inline function _barrierless_begin_executor_measurement!(executor)
    _barrierless_measurement_enabled(executor) || return nothing
    statistics = executor.benchmark_statistics
    statistics.update_open || error("executor measurement has no open update")
    statistics.executor_open && error("executor measurement is already open")
    statistics.executor_wall_started = time_ns()
    statistics.executor_cpu_started = CpuSets.process_cpu_ticks_100ns()
    statistics.executor_gc_started = Base.gc_num()
    statistics.executor_open = true
    return nothing
end

function _barrierless_finish_executor_measurement!(executor)
    _barrierless_measurement_enabled(executor) || return nothing
    statistics = executor.benchmark_statistics
    statistics.executor_open || error("executor measurement is not open")
    wall_finished = time_ns()
    cpu_finished = CpuSets.process_cpu_ticks_100ns()
    gc_delta = _barrierless_gc_diff(statistics.executor_gc_started)
    statistics.executor_wall_nanoseconds += _barrierless_nonnegative_u128(
        wall_finished - statistics.executor_wall_started,
    )
    statistics.executor_cpu_ticks_100ns += _barrierless_nonnegative_u128(
        cpu_finished - statistics.executor_cpu_started,
    )
    statistics.executor_allocation_bytes += _barrierless_nonnegative_u128(
        gc_delta.allocd,
    )

    statistics.executor_open = false
    return nothing
end

function _barrierless_collect_worker_statistics!(executor)
    _barrierless_measurement_enabled(executor) || return nothing
    statistics = executor.benchmark_statistics
    # Background workers remain available for parameter jobs after the
    # candidate timing window closes.  Aggregate their candidate-only counters
    # only after the final post-phase rendezvous, when every native worker is
    # quiescent and no late queue-wait publication can race this fixed-order
    # diagnostic reduction.
    @inbounds for worker in executor.workers
        local_statistics = worker.statistics
        statistics.jobs += UInt128(local_statistics.jobs)
        statistics.chunks += UInt128(local_statistics.chunks)
        for worker_phase in 1:BARRIERLESS_PHASE_COUNT
            benchmark_phase = BARRIERLESS_WORKER_TO_BENCH_PHASE[worker_phase]
            statistics.phase_wall_nanoseconds[benchmark_phase] +=
                local_statistics.phase_wall_nanoseconds[worker_phase]
            statistics.phase_cpu_ticks_100ns[benchmark_phase] +=
                local_statistics.phase_cpu_ticks_100ns[worker_phase]
        end
    end
    return nothing
end

function _barrierless_finish_update_measurement!(executor)
    _barrierless_measurement_enabled(executor) || return nothing
    statistics = executor.benchmark_statistics
    statistics.update_open || error("benchmark update measurement is not open")
    statistics.executor_open && error("cannot finish update with executor measurement open")
    wall_finished = time_ns()
    cpu_finished = CpuSets.process_cpu_ticks_100ns()
    gc_delta = _barrierless_gc_diff(statistics.update_gc_started)
    statistics.overall_wall_nanoseconds += _barrierless_nonnegative_u128(
        wall_finished - statistics.update_wall_started,
    )
    statistics.overall_cpu_ticks_100ns += _barrierless_nonnegative_u128(
        cpu_finished - statistics.update_cpu_started,
    )
    allocation_delta = _barrierless_nonnegative_u128(gc_delta.allocd)
    gc_nanoseconds = _barrierless_nonnegative_u128(gc_delta.total_time)
    statistics.allocation_bytes += allocation_delta
    statistics.gc_nanoseconds += gc_nanoseconds
    # GC is an explicit diagnostic phase.  It overlaps the functional phase
    # in which the collection occurred, so phase sums are not asserted to be a
    # disjoint wall-time decomposition.
    statistics.phase_wall_nanoseconds[BARRIERLESS_BENCH_GC] += gc_nanoseconds
    statistics.phase_cpu_ticks_100ns[BARRIERLESS_BENCH_GC] +=
        cld(gc_nanoseconds, UInt128(100))
    statistics.phase_allocation_bytes[BARRIERLESS_BENCH_GC] += allocation_delta
    statistics.phase_gc_nanoseconds[BARRIERLESS_BENCH_GC] += gc_nanoseconds
    statistics.measured_updates += 1
    statistics.update_open = false
    return nothing
end

function _barrierless_abort_measurement!(executor)
    statistics = executor.benchmark_statistics
    statistics.update_open = false
    statistics.executor_open = false
    return nothing
end


"""Cancel an externally opened interval after a pre-publication exception."""
function abort_barrierless_update_measurement!(executor::BarrierlessExecutor)
    executor.update_inflight[] == 0 || error(
        "published work must reach its normal update boundary",
    )
    _barrierless_abort_measurement!(executor)
    return nothing
end

"""Return stable benchmark counters without mutating the executor."""
function barrierless_benchmark_summary(
    executor::BarrierlessExecutor;
    sampled_allocation_profile=nothing,
)
    statistics = executor.benchmark_statistics
    measured_updates = Int(statistics.measured_updates)
    overall_wall_seconds = Float64(statistics.overall_wall_nanoseconds) * 1.0e-9
    overall_cpu_seconds = Float64(statistics.overall_cpu_ticks_100ns) * 1.0e-7
    executor_wall_seconds = Float64(statistics.executor_wall_nanoseconds) * 1.0e-9
    executor_cpu_seconds = Float64(statistics.executor_cpu_ticks_100ns) * 1.0e-7
    worker_capacity = max(executor.active_workers, 1)
    phases = ntuple(BARRIERLESS_BENCH_PHASE_COUNT) do phase
        aggregate_wall_seconds = Float64(
            statistics.phase_wall_nanoseconds[phase],
        ) * 1.0e-9
        cpu_seconds = Float64(
            statistics.phase_cpu_ticks_100ns[phase],
        ) * 1.0e-7
        allocation_bytes = UInt128(statistics.phase_allocation_bytes[phase])
        allocation_measurement = _barrierless_phase_allocation_measurement(phase)
        allocation_is_measured = allocation_measurement !==
            :unavailable_parallel_worker_phase
        return (;
            name=BARRIERLESS_BENCH_PHASE_NAMES[phase],
            aggregate_wall_seconds,
            capacity_normalized_wall_seconds=aggregate_wall_seconds / worker_capacity,
            cpu_seconds,
            allocation_bytes,
            allocation_bytes_per_update=allocation_is_measured ?
                (measured_updates > 0 ? Float64(allocation_bytes) / measured_updates : 0.0) :
                nothing,
            allocation_measurement,
            gc_seconds=Float64(statistics.phase_gc_nanoseconds[phase]) * 1.0e-9,
        )
    end
    allocation_bytes = UInt128(statistics.allocation_bytes)
    executor_allocation_bytes = UInt128(statistics.executor_allocation_bytes)
    return (;
        enabled=statistics.enabled[] != 0,
        updates=measured_updates,
        overall_wall_seconds,
        overall_cpu_seconds,
        overall_cpu_percent=overall_wall_seconds > 0 ?
            100.0 * overall_cpu_seconds / (overall_wall_seconds * worker_capacity) : 0.0,
        executor_wall_seconds,
        executor_cpu_seconds,
        executor_cpu_percent=executor_wall_seconds > 0 ?
            100.0 * executor_cpu_seconds / (executor_wall_seconds * worker_capacity) : 0.0,
        updates_per_second=overall_wall_seconds > 0 ?
            statistics.measured_updates / overall_wall_seconds : 0.0,
        allocation_bytes,
        allocation_bytes_per_update=measured_updates > 0 ?
            Float64(allocation_bytes) / measured_updates : 0.0,
        executor_allocation_bytes,
        executor_allocation_bytes_per_update=measured_updates > 0 ?
            Float64(executor_allocation_bytes) / measured_updates : 0.0,
        gc_seconds=Float64(statistics.gc_nanoseconds) * 1.0e-9,
        jobs=UInt128(statistics.jobs),
        chunks=UInt128(statistics.chunks),
        phases,
        allocation_bytes_per_update_by_phase=(;
            recurrent_forward=phases[BARRIERLESS_BENCH_FORWARD].allocation_bytes_per_update,
            backward=phases[BARRIERLESS_BENCH_BACKWARD].allocation_bytes_per_update,
            gradient_reduce=phases[BARRIERLESS_BENCH_GRADIENT_REDUCE].allocation_bytes_per_update,
            optimizer=phases[BARRIERLESS_BENCH_OPTIMIZER].allocation_bytes_per_update,
        ),
        sampled_allocation_profile,
        estimated_allocation_bytes_per_update_by_phase=
            sampled_allocation_profile === nothing ? (;
                recurrent_forward=nothing,
                backward=nothing,
                gradient_reduce=nothing,
                optimizer=nothing,
                other=nothing,
            ) : sampled_allocation_profile.estimated_bytes_per_update,
    )
end

function _barrierless_mark_failed!(
    executor::BarrierlessExecutor,
    worker_slot::Int,
    exception,
    backtrace,
)
    @inbounds executor.worker_failures[worker_slot] = (exception, backtrace)
    observed = Base.Threads.atomic_cas!(executor.failure_worker, 0, worker_slot)
    if observed == 0
        Base.Threads.atomic_xchg!(executor.shutdown_requested, UInt32(1))
        notify(executor.startup_event)
        Queue.close!(executor.queue)
    end
    return nothing
end

function _barrierless_throw_failure(executor::BarrierlessExecutor)
    worker = executor.failure_worker[]
    iszero(worker) && return nothing
    payload = @inbounds executor.worker_failures[worker]
    payload === nothing && error("barrierless worker $worker failed without a payload")
    exception, backtrace = payload
    throw(Base.CapturedException(exception, backtrace))
end

@inline function _barrierless_enqueue!(executor, job::BarrierlessJob)
    Queue.enqueue_wait!(executor.queue, job) || error("barrierless queue closed")
    return nothing
end

@inline function _barrierless_enqueue_batch!(executor, jobs, count::Int)
    count == 0 && return nothing
    Queue.enqueue_batch_wait!(executor.queue, jobs, count) ||
        error("barrierless queue closed")
    return nothing
end

function _barrierless_prepare!(executor, context, flat_job::Int)
    trainer = context.trainer
    scheduler = trainer.scheduler
    runtime = @inbounds executor.candidates[flat_job]
    runtime.phase == UInt8(BARRIERLESS_CANDIDATE_PREPARE) ||
        error("candidate $flat_job is not in PREPARE")
    state_slot = Int(runtime.state_slot)
    candidate = Int(runtime.candidate)
    workspace = scheduler.state_workspaces[state_slot]
    forced_code = workspace.forced_depths[candidate]
    trajectory = Model.prepare_trajectory(
        trainer.model,
        _candidate_input(context.batches[state_slot], candidate);
        rng=Xoshiro(workspace.candidate_seeds[candidate]),
        training=true,
        forced_depth=iszero(forced_code) ? nothing : Int(forced_code),
        temperature=context.temperature,
        memory_buffer=workspace.memory_buffers[candidate],
        inverse_rms_buffer=workspace.inverse_rms_buffers[candidate],
        arena=workspace.forward_arenas[candidate],
    )
    runtime.trajectory = trajectory
    runtime.phase = UInt8(BARRIERLESS_CANDIDATE_FORWARD)
    # Keep the legacy scheduler mirror populated for smoke/debug hooks.
    scheduler.trajectory_states[flat_job] = trajectory
    return BarrierlessJob(
        BARRIERLESS_FORWARD_STEP, flat_job, context.generation,
    )
end

function _barrierless_forward_step!(executor, context, flat_job::Int)
    trainer = context.trainer
    scheduler = trainer.scheduler
    runtime = @inbounds executor.candidates[flat_job]
    runtime.phase == UInt8(BARRIERLESS_CANDIDATE_FORWARD) ||
        error("candidate $flat_job is not in FORWARD_STEP")
    trajectory = runtime.trajectory
    trajectory === nothing && error("candidate $flat_job lost its trajectory")
    stopped = Model.advance_trajectory!(trainer.model, trajectory)
    if !stopped
        return BarrierlessJob(
            BARRIERLESS_FORWARD_STEP, flat_job, context.generation,
        )
    end

    state_slot = Int(runtime.state_slot)
    candidate = Int(runtime.candidate)
    workspace = scheduler.state_workspaces[state_slot]
    result = Model.finalize_trajectory(trainer.model, trajectory)
    # Publication order is part of the state-local barrier: all candidate data
    # is written before the seq-cst remaining_forward decrement.
    workspace.raw[:, candidate] .= result.output
    workspace.depths[candidate] = Int16(result.depth)
    workspace.tapes[candidate] = result.tape
    runtime.phase = UInt8(BARRIERLESS_CANDIDATE_WAIT_BACKWARD)
    Base.Threads.atomic_add!(executor.forward_active, -1)
    state = @inbounds executor.states[state_slot]
    previous = Base.Threads.atomic_add!(state.remaining_forward, -1)
    previous >= 1 || error("state $state_slot forward counter underflow")
    if previous == 1
        state.phase[] = UInt8(BARRIERLESS_STATE_LOSS)
        # Control jobs are published immediately, rather than waiting for the
        # current worker's continuation batch.
        _barrierless_enqueue!(executor, BarrierlessJob(
            BARRIERLESS_STATE_LOSS_VJP, state_slot, context.generation,
        ))
    end
    return zero(BarrierlessJob)
end

function _barrierless_state_loss_vjp!(
    executor, context, worker_slot::Int, state_slot::Int,
)
    trainer = context.trainer
    scheduler = trainer.scheduler
    state = @inbounds executor.states[state_slot]
    state.phase[] == UInt8(BARRIERLESS_STATE_LOSS) ||
        error("state $state_slot is not in LOSS/VJP")
    batch = context.batches[state_slot]
    workspace = scheduler.state_workspaces[state_slot]
    raw = workspace.raw
    hard_negative = _hard_negative_selection(
        raw, batch, context.hyperparameters.loss.margin_mode,
    )
    components = _weighted_components(
        raw, batch, context.hyperparameters; hard_negative,
    )
    loss, raw_gradient = _loss_output_vjp(
        raw, batch, context.hyperparameters; hard_negative,
    )
    scheduler.raw_gradients[state_slot] .= raw_gradient
    scheduler.state_losses[state_slot] = loss
    count = Int(state.candidate_count)
    Model.lookup_balance_stats!(
        workspace.lookup_balance_stats, workspace.tapes, count,
    )
    halt_probe = _apply_halt_probes!(
        trainer,
        batch,
        workspace,
        state_slot,
        context.expected_update,
        context.hyperparameters,
        worker_slot,
    )
    @inbounds for candidate in 1:count
        state.depth_values[candidate] = Int(workspace.depths[candidate])
    end
    executor.state_results[state_slot] = (;
        loss,
        components=_component_record(components),
        candidate_count=count,
        depths=copy(@view(state.depth_values[1:count])),
        halt_probe,
    )

    # Stable depth-descending fanout.  Equal-depth candidates retain ascending
    # candidate order.  No sort workspace or job vector is allocated here.
    output_index = 0
    @inbounds for depth in Model.MAX_RECURRENT_STEPS:-1:1
        for candidate in 1:count
            state.depth_values[candidate] == depth || continue
            output_index += 1
            flat_job = Int(state.candidate_jobs[candidate])
            state.backward_jobs[output_index] = BarrierlessJob(
                BARRIERLESS_BACKWARD, flat_job, context.generation,
            )
        end
    end
    output_index == count || error("state $state_slot depth fanout is incomplete")
    state.remaining_backward[] = count
    state.phase[] = UInt8(BARRIERLESS_STATE_BACKWARD)
    _barrierless_enqueue_batch!(executor, state.backward_jobs, count)
    return nothing
end

function _barrierless_backward!(
    executor,
    context,
    worker_slot::Int,
    flat_job::Int,
)
    trainer = context.trainer
    scheduler = trainer.scheduler
    runtime = @inbounds executor.candidates[flat_job]
    runtime.phase == UInt8(BARRIERLESS_CANDIDATE_WAIT_BACKWARD) ||
        error("candidate $flat_job is not waiting for BACKWARD")
    state_slot = Int(runtime.state_slot)
    candidate = Int(runtime.candidate)
    workspace = scheduler.state_workspaces[state_slot]
    tape = workspace.tapes[candidate]
    tape === nothing && error("candidate $flat_job lost its training tape")
    accumulator = @inbounds executor.worker_accumulators[worker_slot]
    ffn_scale_contributions = @view executor.ffn_scale_contributions[:, flat_job]
    Model.backward_trajectory!(
        accumulator,
        trainer.model,
        tape,
        @view(scheduler.raw_gradients[state_slot][:, candidate]);
        realized_loss=scheduler.state_losses[state_slot],
        baseline=context.baseline,
        compute_price=context.hyperparameters.halting.compute_price,
        policy_weight=context.hyperparameters.halting.policy_weight,
        entropy_weight=context.hyperparameters.halting.entropy_weight,
        halt_probe_mode=context.hyperparameters.halting.probe_candidates_per_state > 0,
        halt_probe_target=workspace.halt_probe_targets[candidate],
        halt_probe_weight=context.hyperparameters.halting.probe_weight,
        lookup_balance_stats=workspace.lookup_balance_stats,
        lookup_balance_weight=context.hyperparameters.routing.balance_weight,
        temperature=context.temperature,
        ffn_scale_contributions,
    )
    # `cell_bias` receives one Float32 addition per active cell token.  Keep
    # those exact per-token summands in candidate-owned storage so the update
    # boundary can replay the canonical state/candidate/token order.  The
    # forward input buffer is dead after the trajectory tape has been built,
    # and every candidate owns a distinct buffer, so this is race-free and
    # introduces no allocation.
    cell_bias_contributions = workspace.memory_buffers[candidate]
    memory_gradient = accumulator.backward_scratch.memory_gradient
    @inbounds for token in 1:Model.CELL_COUNT
        @simd for coordinate in 1:Model.MODEL_DIM
            cell_bias_contributions[coordinate, token] =
                memory_gradient[coordinate, token]
        end
    end
    runtime.trajectory = nothing
    runtime.phase = UInt8(BARRIERLESS_CANDIDATE_DONE)
    scheduler.trajectory_states[flat_job] = nothing
    state = @inbounds executor.states[state_slot]
    previous = Base.Threads.atomic_add!(state.remaining_backward, -1)
    previous >= 1 || error("state $state_slot backward counter underflow")
    if previous == 1
        state.phase[] = UInt8(BARRIERLESS_STATE_GRADIENT_READY)
        ready_before = Base.Threads.atomic_add!(executor.gradient_ready_states, 1)
        ready_before < TRAINING_STATE_BATCH || error("gradient-ready counter overflow")
        if ready_before + 1 == TRAINING_STATE_BATCH
            # Wake slot 1 if its coordinator/drainer is observing an empty
            # queue.  It re-checks gradient_ready_states before parking again.
            Queue.wake_consumers!(executor.queue)
        end
    end
    return nothing
end


"""Restore the serial state/candidate/reverse-step sum for one sensitive scalar.

Every other gradient remains worker-local and is reduced in worker-slot order.
The per-step FFN residual-scale contribution is inexpensive to retain and is
replayed only after all candidate jobs finish, so its Float32 sum is identical
to the canonical single-worker traversal without serializing candidate VJPs.
"""
function _barrierless_finalize_ordered_ffn_scale!(executor, context)
    for accumulator in executor.worker_accumulators
        accumulator.dense[:ffn_scale_logit][1] = 0.0f0
    end
    target = executor.worker_accumulators[1].dense[:ffn_scale_logit]
    scheduler = context.trainer.scheduler
    @inbounds for state_slot in 1:TRAINING_STATE_BATCH
        state = executor.states[state_slot]
        workspace = scheduler.state_workspaces[state_slot]
        for candidate in 1:Int(state.candidate_count)
            flat_job = Int(state.candidate_jobs[candidate])
            tape = workspace.tapes[candidate]
            tape === nothing && error("candidate $flat_job lost its ordered scalar tape")
            for step_index in length(tape.steps):-1:1
                target[1] += executor.ffn_scale_contributions[step_index, flat_job]
            end
        end
    end
    return nothing
end

"""Restore the serial state/candidate/token sum for the shared cell bias.

Candidate VJPs run in queue completion order and therefore cannot accumulate
`cell_bias` in the canonical single-worker Float32 order directly.  Each
candidate retained the exact raw input-token cotangent in its otherwise-dead
forward memory buffer.  Replaying those summands here preserves both the
serial active-zero test and its state-major, candidate-major, token-major
addition order without serializing the expensive candidate VJPs.
"""
function _barrierless_finalize_ordered_cell_bias!(executor, context)
    for accumulator in executor.worker_accumulators
        fill!(accumulator.dense[:cell_bias], 0.0f0)
    end
    # `dense` intentionally stores arrays of several ranks.  Recover the
    # concrete rank once here so the scalar replay loop does not dynamically
    # dispatch (and box) every Float32 addition.
    target = executor.worker_accumulators[1].dense[:cell_bias]::Vector{Float32}
    scheduler = context.trainer.scheduler
    @inbounds for state_slot in 1:TRAINING_STATE_BATCH
        state = executor.states[state_slot]
        workspace = scheduler.state_workspaces[state_slot]
        for candidate in 1:Int(state.candidate_count)
            contributions = workspace.memory_buffers[candidate]::Matrix{Float32}
            for token in 1:Model.CELL_COUNT
                # Inactive tokens contain an all-zero column.  Adding those
                # zeros is numerically identical to the canonical serial skip,
                # while avoiding a boxed Bool SIMD reduction and a second pass
                # over every candidate/token column.
                @simd for coordinate in 1:Model.MODEL_DIM
                    target[coordinate] += contributions[coordinate, token]
                end
            end
        end
    end
    return nothing
end

function _barrierless_dispatch_job!(
    executor::BarrierlessExecutor,
    worker::BarrierlessWorkerRuntime,
    worker_slot::Int,
    context::BarrierlessEpochContext,
    job::BarrierlessJob,
    continuation_count::Int,
)
    job.generation == context.generation || error(
        "stale barrierless job generation $(job.generation), current $(context.generation)",
    )
    if barrierless_is_post_job(job.kind)
        return barrierless_dispatch_post_job!(
            executor,
            worker,
            worker_slot,
            context,
            job,
            continuation_count,
        )
    end
    kind = BarrierlessJobKind(job.kind)
    measure = _barrierless_measurement_enabled(executor)
    wall_started = measure ? time_ns() : UInt64(0)
    cpu_started = measure ? CpuSets.thread_cpu_ticks_100ns() : UInt64(0)
    continuation = zero(BarrierlessJob)
    phase = 0
    if kind == BARRIERLESS_PREPARE
        phase = BARRIERLESS_PHASE_PREPARE
        continuation = _barrierless_prepare!(executor, context, Int(job.target))
    elseif kind == BARRIERLESS_FORWARD_STEP
        phase = BARRIERLESS_PHASE_FORWARD
        continuation = _barrierless_forward_step!(executor, context, Int(job.target))
    elseif kind == BARRIERLESS_STATE_LOSS_VJP
        phase = BARRIERLESS_PHASE_LOSS_VJP
        _barrierless_state_loss_vjp!(
            executor, context, worker_slot, Int(job.target),
        )
    elseif kind == BARRIERLESS_BACKWARD
        phase = BARRIERLESS_PHASE_BACKWARD
        _barrierless_backward!(executor, context, worker_slot, Int(job.target))
    else
        error("unknown barrierless job kind $(job.kind)")
    end
    measure && _barrierless_record_phase!(
        worker.statistics, phase, wall_started, cpu_started,
    )
    if continuation.kind != UInt8(BARRIERLESS_NO_JOB)
        continuation_count += 1
        @inbounds worker.continuation_buffer[continuation_count] = continuation
    end
    return continuation_count
end

function _barrierless_take_batch!(
    executor::BarrierlessExecutor,
    worker::BarrierlessWorkerRuntime,
    stop_at_gradient_boundary::Bool=false,
    stop_at_postphase_boundary::Bool=false,
)
    queue = executor.queue
    while true
        stop_at_gradient_boundary &&
            executor.gradient_ready_states[] == TRAINING_STATE_BATCH &&
            executor.active_dispatches[] == 0 &&
            return 0
        stop_at_postphase_boundary &&
            executor.gradient_ready_states[] == TRAINING_STATE_BATCH &&
            barrierless_postphase_complete(executor) &&
            executor.active_dispatches[] == 0 &&
            return 0
        executor.shutdown_requested[] != 0 && isempty(queue) && return 0
        # Candidate jobs use the selected benchmark chunk.  The post-gradient
        # tail contains only a few dozen disjoint parameter jobs; claiming a
        # candidate-sized chunk there would let one worker monopolize the
        # entire reduce or optimizer phase.
        chunk = executor.gradient_ready_states[] == TRAINING_STATE_BATCH &&
                !barrierless_postphase_complete(executor) ? 1 :
                _barrierless_chunk_size(executor)
        count = Queue.try_dequeue_batch!(queue, worker.dequeue_buffer, chunk)
        count > 0 && return count

        # Required double-check pattern: observe the notification epoch, retry
        # dequeue, re-check the external predicate, then park in WaitOnAddress.
        expected = Queue.item_epoch(queue)
        count = Queue.try_dequeue_batch!(queue, worker.dequeue_buffer, chunk)
        count > 0 && return count
        stop_at_gradient_boundary &&
            executor.gradient_ready_states[] == TRAINING_STATE_BATCH &&
            executor.active_dispatches[] == 0 &&
            return 0
        stop_at_postphase_boundary &&
            executor.gradient_ready_states[] == TRAINING_STATE_BATCH &&
            barrierless_postphase_complete(executor) &&
            executor.active_dispatches[] == 0 &&
            return 0
        executor.shutdown_requested[] != 0 && isempty(queue) && return 0
        measure = _barrierless_measurement_enabled(executor) &&
            executor.gradient_ready_states[] < TRAINING_STATE_BATCH
        wall_started = measure ? time_ns() : UInt64(0)
        cpu_started = measure ? CpuSets.thread_cpu_ticks_100ns() : UInt64(0)
        Queue.wait_for_item_change!(queue, expected; timeout_ms=100)
        measure && _barrierless_record_phase!(
            worker.statistics, BARRIERLESS_PHASE_QUEUE_WAIT,
            wall_started, cpu_started,
        )
    end
end

function _barrierless_process_batch!(
    executor::BarrierlessExecutor,
    worker::BarrierlessWorkerRuntime,
    worker_slot::Int,
    count::Int,
)
    Base.Threads.atomic_add!(executor.active_dispatches, 1)
    try
        context = executor.epoch_context[]
        context isa BarrierlessEpochContext || error(
            "barrierless queue published work without an epoch context",
        )
        continuation_count = 0
        candidate_jobs = 0
        @inbounds for index in 1:count
            is_post_job = barrierless_is_post_job(
                worker.dequeue_buffer[index].kind,
            )
            continuation_count = _barrierless_dispatch_job!(
                executor, worker, worker_slot, context,
                worker.dequeue_buffer[index], continuation_count,
            )
            if !is_post_job
                worker.statistics.jobs += 1
                candidate_jobs += 1
            end
        end
        candidate_jobs > 0 && (worker.statistics.chunks += 1)
        _barrierless_enqueue_batch!(
            executor, worker.continuation_buffer, continuation_count,
        )
    finally
        previous = Base.Threads.atomic_add!(executor.active_dispatches, -1)
        previous >= 1 || error("active dispatch counter underflow")
        if previous == 1 &&
           executor.gradient_ready_states[] == TRAINING_STATE_BATCH
            # The fourth state can become ready before the final worker has
            # recorded phase statistics and returned from its claimed batch.
            # Wake the slot-1 drainer only after that publication is complete.
            Queue.wake_consumers!(executor.queue)
        end
    end
    return nothing
end

function _barrierless_wait_for_next_generation!(
    executor::BarrierlessExecutor,
    completed_generation::UInt32,
)
    queue = executor.queue
    while executor.shutdown_requested[] == 0 &&
          executor.generation[] == completed_generation
        expected = Queue.item_epoch(queue)
        executor.shutdown_requested[] != 0 && return nothing
        executor.generation[] != completed_generation && return nothing
        Queue.wait_for_item_change!(queue, expected; timeout_ms=100)
    end
    return nothing
end

function _barrierless_worker_entry!(executor::BarrierlessExecutor, worker_slot::Int)
    worker = @inbounds executor.workers[worker_slot]
    # Do not enter the measured queue-wait loop before the first update is
    # published.  Besides excluding data-loading time, this makes the first
    # per-worker statistics reset race-free.
    if executor.generation[] == 0
        _barrierless_wait_for_next_generation!(executor, UInt32(0))
        executor.shutdown_requested[] != 0 && return nothing
    end
    while true
        count = _barrierless_take_batch!(executor, worker, false, true)
        if count == 0
            executor.shutdown_requested[] != 0 && return nothing
            executor.update_inflight[] == 1 || error(
                "worker reached a gradient boundary without an active update",
            )
            executor.gradient_ready_states[] == TRAINING_STATE_BATCH || error(
                "worker left the queue before every state was gradient-ready",
            )
            barrierless_postphase_complete(executor) || error(
                "worker left the queue before reduction and optimizer completed",
            )
            executor.active_dispatches[] == 0 || error(
                "worker left the queue while another worker was dispatching",
            )
            completed_generation = executor.generation[]
            previous = Base.Threads.atomic_add!(executor.boundary_workers, 1)
            previous < executor.active_workers - 1 || error(
                "barrierless boundary-worker counter overflow",
            )
            Queue.wake_consumers!(executor.queue)
            _barrierless_wait_for_next_generation!(
                executor, completed_generation,
            )
            executor.shutdown_requested[] != 0 && return nothing
            continue
        end
        _barrierless_process_batch!(executor, worker, worker_slot, count)
    end
end

"""
Drain the current update on native worker slot 1 until the exact all-gradient
boundary.  The caller is therefore both coordinator and the twentieth worker;
it never parks native thread 1 while candidate work is available elsewhere.
"""
function _barrierless_drain_update!(
    executor::BarrierlessExecutor,
    worker_slot::Int=1,
)
    worker_slot == 1 || throw(ArgumentError(
        "the synchronous coordinator must own worker slot 1",
    ))
    executor.update_inflight[] == 1 || error("no barrierless update is active")
    worker = @inbounds executor.workers[worker_slot]
    while executor.gradient_ready_states[] < TRAINING_STATE_BATCH ||
          executor.active_dispatches[] != 0
        _barrierless_throw_failure(executor)
        count = _barrierless_take_batch!(executor, worker, true)
        count == 0 && break
        _barrierless_process_batch!(executor, worker, worker_slot, count)
    end
    _barrierless_throw_failure(executor)
    executor.gradient_ready_states[] == TRAINING_STATE_BATCH || error(
        "barrierless drain ended before every state became gradient-ready",
    )
    executor.active_dispatches[] == 0 || error(
        "barrierless drain ended with a worker still dispatching jobs",
    )
    Queue.approx_length(executor.queue) == 0 || error(
        "queue retained work after every state became gradient-ready",
    )
    return executor.state_results
end

"""
Run one persistent native-worker team around the caller's complete training
driver.  `coordinator_body(executor)` executes on pinned native worker slot 1.
Slots 2:N remain in the shared WaitOnAddress queue for the lifetime of the
driver; slot 1 joins that same queue via `_barrierless_drain_update!` during
each update.  No background sticky coordinator Task is created.
"""
function run_with_barrierless_team!(
    coordinator_body::F,
    executor::BarrierlessExecutor,
) where {F}
    executor.started && error("barrierless worker team is already started")
    Queue.isclosed(executor.queue) && error("cannot start a closed barrierless queue")
    executor.active_workers >= 1 || error("worker slot 1 must be active")
    executor.ready_workers[] = 0
    executor.booted_workers[] = 0
    executor.shutdown_requested[] = 0
    executor.failure_worker[] = 0
    reset(executor.startup_event)
    executor.started = true
    coordinator_result = Ref{Any}(nothing)

    try
        Base.Threads.threading_run(worker_slot -> begin
            try
                # Every native Julia worker applies/clears and verifies the
                # configured CPU Set once, including inactive P-only slots.
                binding = CpuSets.bind_current_worker!(worker_slot)
                @inbounds executor.worker_bindings[worker_slot] = binding
                booted_before = Base.Threads.atomic_add!(
                    executor.booted_workers, 1,
                )
                if booted_before + 1 == executor.julia_workers
                    notify(executor.startup_event)
                end
                worker_slot <= executor.active_workers || return nothing
                ready_before = Base.Threads.atomic_add!(executor.ready_workers, 1)
                if ready_before + 1 == executor.active_workers
                    notify(executor.startup_event)
                end

                if worker_slot == 1
                    while executor.booted_workers[] < executor.julia_workers ||
                          executor.ready_workers[] < executor.active_workers
                        _barrierless_throw_failure(executor)
                        wait(executor.startup_event)
                    end
                    _barrierless_throw_failure(executor)
                    coordinator_result[] = coordinator_body(executor)
                    executor.update_inflight[] == 0 || error(
                        "coordinator returned before the optimizer update boundary",
                    )
                    Base.Threads.atomic_xchg!(
                        executor.shutdown_requested, UInt32(1),
                    )
                    Queue.close!(executor.queue)
                else
                    _barrierless_worker_entry!(executor, worker_slot)
                end
                return nothing
            catch exception
                _barrierless_mark_failed!(
                    executor, min(worker_slot, length(executor.worker_failures)),
                    exception, catch_backtrace(),
                )
                return nothing
            end
        end, true)
    finally
        executor.started = false
    end
    _barrierless_throw_failure(executor)
    return coordinator_result[]
end

run_with_barrierless_team!(executor::BarrierlessExecutor, body::F) where {F} =
    run_with_barrierless_team!(body, executor)

"""
Publish one four-state update.  RNG preparation stays serial and in the legacy
state-major/candidate-major order through `_prepare_flat_jobs!`.
"""
function begin_barrierless_update!(
    executor::BarrierlessExecutor,
    trainer,
    batches;
    expected_update::Int,
    hyperparameters,
    baseline::Float32=trainer.baseline,
)
    executor.started || error("barrierless worker team is not started")
    _barrierless_throw_failure(executor)
    iszero(hyperparameters.optimizer.bank_weight_decay) || error(
        "barrierless queue requires zero bank decay so forward remains read-only",
    )
    Base.Threads.atomic_cas!(executor.update_inflight, UInt32(0), UInt32(1)) == 0 ||
        error("a barrierless update is already in flight")
    Queue.approx_length(executor.queue) == 0 || error(
        "barrierless queue is not empty at update boundary",
    )
    measurement_preopened = executor.benchmark_statistics.update_open
    try
        measurement_preopened || _barrierless_begin_update_measurement!(executor)
        executor.gradient_ready_states[] = 0
        executor.active_dispatches[] = 0
        executor.boundary_workers[] = 0
        barrierless_begin_postphase_update!(executor)
        fill!(executor.state_results, nothing)
        for worker in executor.workers
            reset_barrierless_statistics!(worker.statistics)
        end
        for accumulator in executor.worker_accumulators
            Model.reset_gradients!(accumulator)
        end
        executor.generation[] == typemax(UInt32) && error(
            "barrierless generation counter overflow",
        )
        generation = executor.generation[] + UInt32(1)
        if measurement_preopened
            # The caller already measured sampler/host data packing.  Keep the
            # scheduler flatten in the exact overall interval without double
            # counting the externally owned data-pack phase.
            total = _prepare_flat_jobs!(
                trainer, batches, expected_update, hyperparameters,
            )
        else
            data_pack_measurement = begin_barrierless_phase_measurement(
                executor, :data_pack,
            )
            total = try
                _prepare_flat_jobs!(
                    trainer, batches, expected_update, hyperparameters,
                )
            finally
                finish_barrierless_phase_measurement!(
                    executor, data_pack_measurement,
                )
            end
        end
        total <= MAX_FLAT_CANDIDATES || error("too many flattened candidates")

        for state_slot in 1:TRAINING_STATE_BATCH
            state = @inbounds executor.states[state_slot]
            count = _valid_candidate_count(batches[state_slot])
            state.candidate_count = Int16(count)
            state.remaining_forward[] = count
            state.remaining_backward[] = 0
            state.phase[] = UInt8(BARRIERLESS_STATE_FORWARD)
            fill!(state.candidate_jobs, 0)
            fill!(state.backward_jobs, zero(BarrierlessJob))
            fill!(state.depth_values, 0)
        end

        scheduler = trainer.scheduler
        @inbounds for flat_job in 1:total
            state_slot = Int(scheduler.job_states[flat_job])
            candidate = Int(scheduler.job_candidates[flat_job])
            runtime = executor.candidates[flat_job]
            runtime.state_slot = Int16(state_slot)
            runtime.candidate = Int16(candidate)
            runtime.phase = UInt8(BARRIERLESS_CANDIDATE_PREPARE)
            runtime.trajectory = nothing
            executor.states[state_slot].candidate_jobs[candidate] = UInt16(flat_job)
            executor.initial_jobs[flat_job] = BarrierlessJob(
                BARRIERLESS_PREPARE, flat_job, generation,
            )
        end
        @inbounds for flat_job in (total + 1):MAX_FLAT_CANDIDATES
            runtime = executor.candidates[flat_job]
            runtime.phase = UInt8(BARRIERLESS_CANDIDATE_IDLE)
            runtime.trajectory = nothing
        end
        context = BarrierlessEpochContext(
            trainer, batches, expected_update, hyperparameters,
            baseline, Float32(routing_temperature(
                expected_update, hyperparameters.routing,
            )), generation,
        )
        executor.epoch_context[] = context
        executor.forward_active[] = total
        _barrierless_begin_executor_measurement!(executor)
        # Publish the epoch only after serial preparation succeeds.  The
        # persistent workers use this atomic word as their between-update
        # release condition, so a preparation exception must leave it intact.
        executor.generation[] = generation
        _barrierless_enqueue_batch!(executor, executor.initial_jobs, total)
        return total
    catch
        barrierless_abort_postphase_update!(executor)
        _barrierless_abort_measurement!(executor)
        executor.update_inflight[] = UInt32(0)
        rethrow()
    end
end

"""
Yield the caller task until every state's candidate gradients are complete.

This deliberately does not clear `update_inflight`: the caller must first
perform fixed-order reduction, one exact 4-state mean, and the optimizer, then
call `finish_barrierless_update!`.  That enforces the no-stale-weight boundary.
"""
function wait_barrierless_gradients!(executor::BarrierlessExecutor)
    Base.Threads.threadid() == 1 || error(
        "wait_barrierless_gradients! must run in the slot-1 coordinator",
    )
    context = executor.epoch_context[]
    context isa BarrierlessEpochContext || error(
        "barrierless gradient boundary lost its epoch context",
    )
    state_results = _barrierless_drain_update!(executor, 1)
    _barrierless_finalize_ordered_ffn_scale!(executor, context)
    _barrierless_finalize_ordered_cell_bias!(executor, context)
    _barrierless_finish_executor_measurement!(executor)
    return state_results
end

"""
Release the update boundary only after reduction and optimizer finish.

`finish_measurement=false` is used by the training driver so that the exact
overall-update interval remains open while it aggregates loss/depth records,
updates counters, and constructs the returned step record.  The default keeps
the original standalone-call semantics used by the correctness smoke.
"""
function finish_barrierless_update!(
    executor::BarrierlessExecutor;
    finish_measurement::Bool=true,
)
    executor.update_inflight[] == 1 || error("no barrierless update is active")
    executor.gradient_ready_states[] == TRAINING_STATE_BATCH || error(
        "cannot finish before all state gradients are ready",
    )
    executor.active_dispatches[] == 0 || error(
        "cannot finish while a worker is still dispatching jobs",
    )
    barrierless_postphase_complete(executor) || error(
        "cannot finish before reduction and optimizer complete",
    )
    Queue.wake_consumers!(executor.queue)
    boundary_target = executor.active_workers - 1
    while executor.boundary_workers[] < boundary_target
        _barrierless_throw_failure(executor)
        expected = Queue.item_epoch(executor.queue)
        executor.boundary_workers[] >= boundary_target && break
        Queue.wait_for_item_change!(executor.queue, expected; timeout_ms=100)
    end
    _barrierless_throw_failure(executor)
    executor.boundary_workers[] == executor.active_workers - 1 || error(
        "cannot finish before every background worker reaches the boundary",
    )
    Queue.approx_length(executor.queue) == 0 || error(
        "cannot finish with queued work",
    )
    _barrierless_collect_worker_statistics!(executor)
    finish_measurement && _barrierless_finish_update_measurement!(executor)
    executor.epoch_context[] = nothing
    executor.update_inflight[] = UInt32(0)
    return executor
end

"""Close a deferred exact overall-update interval at the return boundary."""
function finish_barrierless_update_measurement!(executor::BarrierlessExecutor)
    executor.update_inflight[] == 0 || error(
        "overall update measurement may close only after the update boundary",
    )
    _barrierless_finish_update_measurement!(executor)
    return nothing
end

"""
Request early team shutdown from the slot-1 coordinator after an update
boundary.  Normal lifetime shutdown is automatic when the coordinator body
passed to `run_with_barrierless_team!` returns.
"""
function shutdown_barrierless_team!(executor::BarrierlessExecutor)
    executor.started || return executor
    executor.update_inflight[] == 0 || error(
        "shutdown is allowed only after the optimizer update boundary",
    )
    Base.Threads.atomic_xchg!(executor.shutdown_requested, UInt32(1))
    Queue.close!(executor.queue)
    return executor
end

barrierless_worker_accumulators(executor::BarrierlessExecutor) =
    executor.worker_accumulators

barrierless_state_results(executor::BarrierlessExecutor) = executor.state_results

function barrierless_worker_statistics(executor::BarrierlessExecutor)
    return [
        (;
            worker_slot,
            jobs=statistics.jobs,
            chunks=statistics.chunks,
            phase_wall_nanoseconds=copy(statistics.phase_wall_nanoseconds),
            phase_cpu_ticks_100ns=copy(statistics.phase_cpu_ticks_100ns),
        )
        for (worker_slot, worker) in enumerate(executor.workers)
        for statistics in (worker.statistics,)
    ]
end

barrierless_worker_bindings(executor::BarrierlessExecutor) =
    copy(executor.worker_bindings)
