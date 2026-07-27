module BarrierlessWorkspaceTraining

using LinearAlgebra
using Lux
using Optimisers
using Statistics
using Zygote

if !isdefined(Main, :BoundedMPMCRing)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "..",
            "episodic_vit_recurrent_lookup",
            "bounded_mpmc_queue.jl",
        ),
    )
end
if !isdefined(Main, :WinCpuSets)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "..",
            "episodic_vit_recurrent_lookup",
            "windows_cpu_sets.jl",
        ),
    )
end
if !isdefined(Main, :SerialWorkspaceSNN)
    Base.include(Main, joinpath(@__DIR__, "SerialWorkspaceSNN.jl"))
end
if !isdefined(Main, :BeatFirstTrainingCore)
    Base.include(Main, joinpath(@__DIR__, "..", "training", "core.jl"))
end

const Queue = Main.BoundedMPMCRing
const CpuSets = Main.WinCpuSets
const Model = Main.SerialWorkspaceSNN
const TrainingCore = Main.BeatFirstTrainingCore

export BarrierlessExecutor,
    barrierless_gradient!,
    barrierless_update!,
    gradient_max_abs_difference,
    raw_matrix,
    run_with_barrierless_team!,
    tree_norm

@enum WorkKind::UInt8 begin
    NO_WORK = 0
    FORWARD_WORK = 1
    BACKWARD_WORK = 2
end

struct WorkItem
    kind::UInt8
    target::UInt16
    generation::UInt32
end

WorkItem(kind::WorkKind, target::Integer, generation::UInt32) =
    WorkItem(UInt8(kind), UInt16(target), generation)

Base.zero(::Type{WorkItem}) = WorkItem(UInt8(NO_WORK), UInt16(0), UInt32(0))
isbitstype(WorkItem) || error("barrierless work items must remain isbits")

mutable struct WorkerStatistics
    forward_jobs::Vector{Int}
    backward_jobs::Vector{Int}
    forward_wall_ns::Vector{UInt64}
    backward_wall_ns::Vector{UInt64}
    forward_cpu_ticks::Vector{UInt64}
    backward_cpu_ticks::Vector{UInt64}
end

function WorkerStatistics(workers::Int)
    return WorkerStatistics(
        zeros(Int, workers),
        zeros(Int, workers),
        zeros(UInt64, workers),
        zeros(UInt64, workers),
        zeros(UInt64, workers),
        zeros(UInt64, workers),
    )
end

function reset!(statistics::WorkerStatistics)
    fill!(statistics.forward_jobs, 0)
    fill!(statistics.backward_jobs, 0)
    fill!(statistics.forward_wall_ns, 0)
    fill!(statistics.backward_wall_ns, 0)
    fill!(statistics.forward_cpu_ticks, 0)
    fill!(statistics.backward_cpu_ticks, 0)
    return statistics
end

mutable struct EpochContext
    model::Any
    parameters::Any
    states::Any
    batch::Any
    ranges::Vector{UnitRange{Int}}
    raw_chunks::Vector{Any}
    pullbacks::Vector{Any}
    output_cotangent::Any
    gradients::Vector{Any}
    generation::UInt32
end

mutable struct BarrierlessExecutor
    queue::Queue.BoundedMPMCQueue{WorkItem}
    active_workers::Int
    julia_workers::Int
    chunk_size::Int
    cpuset_mode::Symbol
    epoch::Base.RefValue{Any}
    generation::Base.Threads.Atomic{UInt32}
    remaining_forward::Base.Threads.Atomic{Int}
    remaining_backward::Base.Threads.Atomic{Int}
    shutdown_requested::Base.Threads.Atomic{UInt32}
    ready_workers::Base.Threads.Atomic{Int}
    booted_workers::Base.Threads.Atomic{Int}
    failure_worker::Base.Threads.Atomic{Int}
    failures::Vector{Any}
    bindings::Vector{Any}
    startup_event::Base.Event
    statistics::WorkerStatistics
    started::Bool
end

function BarrierlessExecutor(;
    active_workers::Int=Base.Threads.nthreads(:default),
    chunk_size::Int=4,
    cpuset_mode::Symbol=:none,
    queue_capacity::Int=512,
)
    julia_workers = Base.Threads.nthreads(:default)
    Base.Threads.nthreads(:interactive) == 0 || error(
        "launch Julia with --threads=N,0 for deterministic native worker slots",
    )
    2 <= active_workers <= julia_workers || throw(ArgumentError(
        "active_workers must be in 2:$julia_workers",
    ))
    chunk_size in (1, 2, 4, 8, 16, 32) || throw(ArgumentError(
        "chunk_size must be one of 1,2,4,8,16,32",
    ))
    cpuset_mode in (:none, :all, :p_only) || throw(ArgumentError(
        "cpuset_mode must be :none, :all, or :p_only",
    ))
    ispow2(queue_capacity) || throw(ArgumentError("queue capacity must be a power of two"))
    queue_capacity >= 256 || throw(ArgumentError("queue capacity must be at least 256"))
    return BarrierlessExecutor(
        Queue.BoundedMPMCQueue{WorkItem}(queue_capacity, zero(WorkItem)),
        active_workers,
        julia_workers,
        chunk_size,
        cpuset_mode,
        Ref{Any}(nothing),
        Base.Threads.Atomic{UInt32}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{UInt32}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Any[nothing for _ in 1:julia_workers],
        Any[nothing for _ in 1:julia_workers],
        Base.Event(true),
        WorkerStatistics(active_workers),
        false,
    )
end

function raw_matrix(output)
    candidates = length(output.q)
    return vcat(
        reshape(output.q, 1, candidates),
        reshape(output.death_logit, 1, candidates),
        output.quantiles,
        output.geometry,
    )
end

function _output_from_raw(raw::AbstractMatrix)
    size(raw, 1) == 22 || throw(DimensionMismatch("raw output must have 22 rows"))
    return (;
        q=vec(raw[1:1, :]),
        death_logit=vec(raw[2:2, :]),
        quantiles=raw[3:18, :],
        geometry=raw[19:22, :],
    )
end

function _slice_input(input, range::UnitRange{Int})
    return (;
        board=@view(input.board[:, :, :, range]),
        candidate=@view(input.candidate[:, :, :, range]),
        difference=@view(input.difference[:, :, :, range]),
        aux=@view(input.aux[:, range]),
        next_hold=@view(input.next_hold[:, :, range]),
        local_mask=@view(input.local_mask[:, :, :, range]),
    )
end

function _candidate_ranges(batch, chunk_size::Int)
    width, state_batch = size(batch.mask)
    ranges = UnitRange{Int}[]
    for slot in 1:state_batch
        count = Int(sum(@view(batch.mask[:, slot])))
        1 <= count <= width || error("invalid candidate count $count in state slot $slot")
        offset = (slot - 1) * width
        first_candidate = 1
        while first_candidate <= count
            last_candidate = min(first_candidate + chunk_size - 1, count)
            push!(ranges, (offset + first_candidate):(offset + last_candidate))
            first_candidate = last_candidate + 1
        end
    end
    length(ranges) <= typemax(UInt16) || error("too many candidate chunks")
    return ranges
end

function tree_norm(value)
    value === nothing && return 0.0
    value isa AbstractArray && return sqrt(sum(abs2, Float64.(value)))
    value isa NamedTuple && return sqrt(sum(
        tree_norm(child)^2 for child in values(value);
        init=0.0,
    ))
    value isa Tuple && return sqrt(sum(tree_norm(child)^2 for child in value; init=0.0))
    return 0.0
end

_gradient_copy(::Nothing) = nothing
_gradient_copy(value::AbstractArray) = copy(value)
_gradient_copy(value::NamedTuple) =
    NamedTuple{keys(value)}(map(_gradient_copy, values(value)))
_gradient_copy(value::Tuple) = map(_gradient_copy, value)

function _gradient_add!(destination, source)
    source === nothing && return destination
    destination === nothing && error("gradient destination is missing")
    if destination isa AbstractArray
        destination .+= source
    elseif destination isa NamedTuple
        for key in keys(destination)
            _gradient_add!(getproperty(destination, key), getproperty(source, key))
        end
    elseif destination isa Tuple
        for index in eachindex(destination)
            _gradient_add!(destination[index], source[index])
        end
    else
        error("unsupported gradient node $(typeof(destination))")
    end
    return destination
end

function _gradient_scale!(value, scale::Float32)
    value === nothing && return value
    if value isa AbstractArray
        value .*= scale
    elseif value isa NamedTuple || value isa Tuple
        for child in values(value)
            _gradient_scale!(child, scale)
        end
    else
        error("unsupported gradient node $(typeof(value))")
    end
    return value
end

function gradient_max_abs_difference(left, right)
    left === nothing && right === nothing && return 0.0
    (left === nothing) == (right === nothing) || return Inf
    if left isa AbstractArray
        return maximum(abs.(Float64.(left) .- Float64.(right)); init=0.0)
    elseif left isa NamedTuple
        keys(left) == keys(right) || return Inf
        return maximum(
            gradient_max_abs_difference(getproperty(left, key), getproperty(right, key))
            for key in keys(left);
            init=0.0,
        )
    elseif left isa Tuple
        length(left) == length(right) || return Inf
        return maximum(
            gradient_max_abs_difference(left[index], right[index])
            for index in eachindex(left);
            init=0.0,
        )
    end
    return left == right ? 0.0 : Inf
end

function _record_work!(
    statistics::WorkerStatistics,
    worker_slot::Int,
    kind::UInt8,
    wall_started::UInt64,
    cpu_started::UInt64,
)
    wall = time_ns() - wall_started
    cpu = CpuSets.thread_cpu_ticks_100ns() - cpu_started
    if kind == UInt8(FORWARD_WORK)
        statistics.forward_jobs[worker_slot] += 1
        statistics.forward_wall_ns[worker_slot] += wall
        statistics.forward_cpu_ticks[worker_slot] += cpu
    elseif kind == UInt8(BACKWARD_WORK)
        statistics.backward_jobs[worker_slot] += 1
        statistics.backward_wall_ns[worker_slot] += wall
        statistics.backward_cpu_ticks[worker_slot] += cpu
    end
    return nothing
end

@inline function _complete_work!(
    executor::BarrierlessExecutor,
    remaining::Base.Threads.Atomic{Int},
)
    previous = Base.Threads.atomic_add!(remaining, -1)
    previous >= 1 || error("barrierless remaining-work counter underflow")
    previous == 1 && Queue.wake_consumers!(executor.queue)
    return nothing
end

function _dispatch!(
    executor::BarrierlessExecutor,
    worker_slot::Int,
    work::WorkItem,
)
    work.generation == executor.generation[] || error("stale barrierless work generation")
    context = executor.epoch[]
    context isa EpochContext || error("barrierless epoch context is missing")
    target = Int(work.target)
    1 <= target <= length(context.ranges) || error("invalid work target $target")
    range = context.ranges[target]
    wall_started = time_ns()
    cpu_started = CpuSets.thread_cpu_ticks_100ns()
    if work.kind == UInt8(FORWARD_WORK)
        input = _slice_input(context.batch.inputs, range)
        raw, pullback = Zygote.pullback(context.parameters) do parameters
            output, _ = context.model(input, parameters, context.states)
            raw_matrix(output)
        end
        context.raw_chunks[target] = Matrix{Float32}(raw)
        context.pullbacks[target] = pullback
        _record_work!(
            executor.statistics,
            worker_slot,
            work.kind,
            wall_started,
            cpu_started,
        )
        _complete_work!(executor, executor.remaining_forward)
    elseif work.kind == UInt8(BACKWARD_WORK)
        pullback = context.pullbacks[target]
        pullback === nothing && error("backward work has no forward pullback")
        cotangent = @view(context.output_cotangent[:, range])
        gradient = only(pullback(cotangent))
        context.gradients[target] = gradient
        context.pullbacks[target] = nothing
        _record_work!(
            executor.statistics,
            worker_slot,
            work.kind,
            wall_started,
            cpu_started,
        )
        _complete_work!(executor, executor.remaining_backward)
    else
        error("unknown barrierless work kind $(work.kind)")
    end
    return nothing
end

function _mark_failure!(
    executor::BarrierlessExecutor,
    worker_slot::Int,
    exception,
    backtrace,
)
    executor.failures[worker_slot] = (exception, backtrace)
    Base.Threads.atomic_cas!(executor.failure_worker, 0, worker_slot)
    Base.Threads.atomic_xchg!(executor.shutdown_requested, UInt32(1))
    Queue.close!(executor.queue)
    notify(executor.startup_event)
    return nothing
end

function _throw_failure(executor::BarrierlessExecutor)
    worker = executor.failure_worker[]
    worker == 0 && return nothing
    payload = executor.failures[worker]
    payload === nothing && error("barrierless worker $worker failed without payload")
    exception, backtrace = payload
    Base.showerror(stderr, exception, backtrace)
    println(stderr)
    error("barrierless worker $worker failed: $(sprint(showerror, exception))")
end

function _worker_entry!(executor::BarrierlessExecutor, worker_slot::Int)
    while executor.shutdown_requested[] == 0
        available, work = Queue.dequeue_wait!(executor.queue; timeout_ms=100)
        if !available
            Queue.isclosed(executor.queue) && return nothing
            continue
        end
        _dispatch!(executor, worker_slot, work)
    end
    return nothing
end

function _coordinator_drain!(
    executor::BarrierlessExecutor,
    remaining::Base.Threads.Atomic{Int},
)
    while remaining[] > 0
        _throw_failure(executor)
        available, work = Queue.try_dequeue!(executor.queue)
        if available
            _dispatch!(executor, 1, work)
            continue
        end
        expected = Queue.item_epoch(executor.queue)
        remaining[] == 0 && break
        Queue.wait_for_item_change!(executor.queue, expected; timeout_ms=10)
    end
    _throw_failure(executor)
    remaining[] == 0 || error("barrierless phase ended before all work completed")
    return nothing
end

function _enqueue_phase!(
    executor::BarrierlessExecutor,
    kind::WorkKind,
    count::Int,
    generation::UInt32,
)
    remaining = kind === FORWARD_WORK ?
        executor.remaining_forward : executor.remaining_backward
    remaining[] = count
    for target in 1:count
        Queue.enqueue_wait!(
            executor.queue,
            WorkItem(kind, target, generation);
            timeout_ms=10_000,
        ) || error("barrierless queue closed while publishing work")
    end
    return nothing
end

function _phase_snapshot(statistics::WorkerStatistics)
    return (;
        forward_jobs=sum(statistics.forward_jobs),
        backward_jobs=sum(statistics.backward_jobs),
        forward_wall_ns=sum(statistics.forward_wall_ns),
        backward_wall_ns=sum(statistics.backward_wall_ns),
        forward_cpu_ticks=sum(statistics.forward_cpu_ticks),
        backward_cpu_ticks=sum(statistics.backward_cpu_ticks),
        per_worker=[
            (;
                worker=slot,
                forward_jobs=statistics.forward_jobs[slot],
                backward_jobs=statistics.backward_jobs[slot],
                forward_wall_seconds=statistics.forward_wall_ns[slot] * 1.0e-9,
                backward_wall_seconds=statistics.backward_wall_ns[slot] * 1.0e-9,
                forward_cpu_seconds=statistics.forward_cpu_ticks[slot] * 1.0e-7,
                backward_cpu_seconds=statistics.backward_cpu_ticks[slot] * 1.0e-7,
            )
            for slot in eachindex(statistics.forward_jobs)
        ],
    )
end

function _structure_penalty(parameters, structure_weight::Float32)
    density = mean(sigmoid.(parameters.gate_logits))
    return structure_weight * (density - 0.50f0)^2, density
end

function _add_structure_gradient!(
    gradient,
    parameters,
    structure_weight::Float32,
)
    iszero(structure_weight) && return (loss=0.0f0, density=mean(
        sigmoid.(parameters.gate_logits),
    ))
    probability = sigmoid.(parameters.gate_logits)
    density = mean(probability)
    loss = structure_weight * (density - 0.50f0)^2
    coefficient =
        2.0f0 * structure_weight * (density - 0.50f0) / Float32(length(probability))
    gradient.gate_logits .+= coefficient .* probability .* (1.0f0 .- probability)
    return (; loss, density)
end

"""
Compute one exact synchronous gradient with a persistent MPMC worker team.

Candidate chunks may finish in any order, but raw outputs and gradients are
assembled/reduced in canonical candidate order. Parameters remain read-only
until every backward job has completed.
"""
function barrierless_gradient!(
    executor::BarrierlessExecutor,
    model,
    parameters,
    states,
    batch;
    structure_weight::Real=0.01f0,
)
    executor.started || error("run barrierless_gradient! inside run_with_barrierless_team!")
    executor.epoch[] === nothing || error("previous barrierless epoch was not released")
    Queue.approx_length(executor.queue) == 0 || error("barrierless queue is not empty")
    generation = Base.Threads.atomic_add!(executor.generation, UInt32(1)) + UInt32(1)
    ranges = _candidate_ranges(batch, executor.chunk_size)
    count = length(ranges)
    context = EpochContext(
        model,
        parameters,
        states,
        batch,
        ranges,
        Any[nothing for _ in 1:count],
        Any[nothing for _ in 1:count],
        nothing,
        Any[nothing for _ in 1:count],
        generation,
    )
    executor.epoch[] = context
    reset!(executor.statistics)
    process_cpu_started = CpuSets.process_cpu_ticks_100ns()
    total_started = time_ns()

    forward_started = time_ns()
    _enqueue_phase!(executor, FORWARD_WORK, count, generation)
    _coordinator_drain!(executor, executor.remaining_forward)
    forward_seconds = (time_ns() - forward_started) * 1.0e-9

    width, state_batch = size(batch.mask)
    raw = zeros(Float32, 22, width * state_batch)
    for target in 1:count
        raw[:, ranges[target]] .= context.raw_chunks[target]
        context.raw_chunks[target] = nothing
    end

    loss_started = time_ns()
    task_loss, loss_pullback = Zygote.pullback(raw) do candidate_raw
        TrainingCore.supervised_components(
            _output_from_raw(candidate_raw),
            batch,
        ).composite_loss
    end
    output_cotangent = only(loss_pullback(one(task_loss)))
    context.output_cotangent = output_cotangent
    loss_vjp_seconds = (time_ns() - loss_started) * 1.0e-9

    backward_started = time_ns()
    _enqueue_phase!(executor, BACKWARD_WORK, count, generation)
    _coordinator_drain!(executor, executor.remaining_backward)
    backward_seconds = (time_ns() - backward_started) * 1.0e-9

    reduce_started = time_ns()
    gradient = _gradient_copy(context.gradients[1])
    context.gradients[1] = nothing
    for target in 2:count
        _gradient_add!(gradient, context.gradients[target])
        context.gradients[target] = nothing
    end
    structure = _add_structure_gradient!(
        gradient,
        parameters,
        Float32(structure_weight),
    )
    reduction_seconds = (time_ns() - reduce_started) * 1.0e-9
    total_loss = Float32(task_loss) + structure.loss
    all(isfinite, (total_loss, tree_norm(gradient))) || error(
        "barrierless loss or gradient is non-finite",
    )
    components = TrainingCore.supervised_components(_output_from_raw(raw), batch)
    worker = _phase_snapshot(executor.statistics)
    total_seconds = (time_ns() - total_started) * 1.0e-9
    process_cpu_seconds =
        (CpuSets.process_cpu_ticks_100ns() - process_cpu_started) * 1.0e-7
    executor.epoch[] = nothing
    metrics = (;
        active_workers=executor.active_workers,
        chunk_size=executor.chunk_size,
        chunks=count,
        candidates=Int(sum(batch.mask)),
        state_batch,
        forward_seconds,
        loss_vjp_seconds,
        backward_seconds,
        reduction_seconds,
        total_seconds,
        process_cpu_seconds,
        whole_machine_cpu_utilization_percent=
            100.0 * process_cpu_seconds /
            max(total_seconds * executor.julia_workers, eps(Float64)),
        active_worker_cpu_utilization_percent=
            100.0 * process_cpu_seconds /
            max(total_seconds * executor.active_workers, eps(Float64)),
        worker,
    )
    return (;
        loss=total_loss,
        task_loss=Float32(task_loss),
        structure_loss=structure.loss,
        gate_density=structure.density,
        components,
        gradient,
        raw,
        metrics,
    )
end

function barrierless_update!(
    executor::BarrierlessExecutor,
    model,
    parameters,
    states,
    optimizer_state,
    batch;
    structure_weight::Real=0.01f0,
)
    gradient_result = barrierless_gradient!(
        executor,
        model,
        parameters,
        states,
        batch;
        structure_weight,
    )
    optimizer_started = time_ns()
    next_optimizer_state, next_parameters = Optimisers.update(
        optimizer_state,
        parameters,
        gradient_result.gradient,
    )
    optimizer_seconds = (time_ns() - optimizer_started) * 1.0e-9
    return merge(
        gradient_result,
        (; parameters=next_parameters, optimizer_state=next_optimizer_state,
           optimizer_seconds),
    )
end

function run_with_barrierless_team!(
    body::F,
    executor::BarrierlessExecutor,
) where {F}
    executor.started && error("barrierless team is already started")
    Queue.isclosed(executor.queue) && error("cannot restart a closed queue")
    topology = CpuSets.discover_topology()
    binding_plan = CpuSets.configure_worker_bindings(
        executor.cpuset_mode,
        executor.active_workers,
        topology,
    )
    executor.ready_workers[] = 0
    executor.booted_workers[] = 0
    executor.failure_worker[] = 0
    executor.shutdown_requested[] = 0
    reset(executor.startup_event)
    executor.started = true
    result = Ref{Any}(nothing)
    try
        Base.Threads.threading_run(worker_slot -> begin
            try
                binding = CpuSets.bind_current_worker!(worker_slot)
                executor.bindings[worker_slot] = binding
                booted = Base.Threads.atomic_add!(executor.booted_workers, 1) + 1
                booted == executor.julia_workers && notify(executor.startup_event)
                worker_slot <= executor.active_workers || return nothing
                ready = Base.Threads.atomic_add!(executor.ready_workers, 1) + 1
                ready == executor.active_workers && notify(executor.startup_event)
                if worker_slot == 1
                    while executor.booted_workers[] < executor.julia_workers ||
                          executor.ready_workers[] < executor.active_workers
                        _throw_failure(executor)
                        wait(executor.startup_event)
                    end
                    result[] = body(executor)
                    executor.epoch[] === nothing || error(
                        "coordinator returned with an active epoch",
                    )
                    Base.Threads.atomic_xchg!(executor.shutdown_requested, UInt32(1))
                    Queue.close!(executor.queue)
                else
                    _worker_entry!(executor, worker_slot)
                end
                return nothing
            catch exception
                _mark_failure!(
                    executor,
                    min(worker_slot, length(executor.failures)),
                    exception,
                    catch_backtrace(),
                )
                return nothing
            end
        end, true)
    finally
        executor.started = false
    end
    _throw_failure(executor)
    return (result=result[], binding_plan, bindings=copy(executor.bindings))
end

run_with_barrierless_team!(executor::BarrierlessExecutor, body::F) where {F} =
    run_with_barrierless_team!(body, executor)

end # module
