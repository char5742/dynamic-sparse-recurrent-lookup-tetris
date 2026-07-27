using JSON3
using LinearAlgebra
using Lux
using Optimisers
using Random
using Statistics
using Zygote

include(joinpath(@__DIR__, "SerialWorkspaceSNN.jl"))
include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "BarrierlessWorkspaceTraining.jl"))
using .SerialWorkspaceSNN
using .BeatFirstTrainingCore
using .BarrierlessWorkspaceTraining

const CpuSets = Main.WinCpuSets
const DEFAULT_DATASET = raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const MODEL_SEED = UInt64(2026072703)

env_int(name, default; minimum=1) = begin
    value = parse(Int, get(ENV, name, string(default)))
    value >= minimum || error("$name must be >= $minimum")
    value
end

function training_rows_only(dataset)
    if hasproperty(dataset, :predefined_split)
        rows = findall(==(:train), dataset.predefined_split)
        !isempty(rows) && return Int.(rows)
    end
    return collect(eachindex(dataset.action_counts))
end

function serial_objective(model, ps, st, batch, structure_weight::Float32)
    output, _ = model(batch.inputs, ps, st)
    components = supervised_components(output, batch)
    density = mean(sigmoid.(ps.gate_logits))
    return components.composite_loss + structure_weight * (density - 0.50f0)^2
end

function serial_update!(
    model,
    ps,
    st,
    optimizer_state,
    batch,
    structure_weight::Float32,
)
    loss, pullback = Zygote.pullback(ps) do parameters
        serial_objective(model, parameters, st, batch, structure_weight)
    end
    gradient = only(pullback(one(loss)))
    next_optimizer_state, next_ps = Optimisers.update(
        optimizer_state, ps, gradient,
    )
    return (;
        loss=Float32(loss),
        parameters=next_ps,
        optimizer_state=next_optimizer_state,
        gradient_norm=tree_norm(gradient),
    )
end

function summarize_phase(records, field)
    isempty(records) && return 0.0
    return mean(Float64(getproperty(record.metrics, field)) for record in records)
end

function benchmark_serial(
    model,
    ps,
    st,
    optimizer,
    batch,
    warmup_updates::Int,
    measured_updates::Int,
    structure_weight::Float32,
)
    optimizer_state = Optimisers.setup(optimizer, ps)
    last = nothing
    for _ in 1:warmup_updates
        last = serial_update!(
            model, ps, st, optimizer_state, batch, structure_weight,
        )
        ps = last.parameters
        optimizer_state = last.optimizer_state
    end
    GC.gc()
    cpu_started = CpuSets.process_cpu_ticks_100ns()
    timed = @timed begin
        for _ in 1:measured_updates
            last = serial_update!(
                model, ps, st, optimizer_state, batch, structure_weight,
            )
            ps = last.parameters
            optimizer_state = last.optimizer_state
        end
    end
    cpu_seconds = (CpuSets.process_cpu_ticks_100ns() - cpu_started) * 1.0e-7
    state_batch = size(batch.mask, 2)
    return (;
        mode=:serial,
        parameters=ps,
        optimizer_state,
        last_loss=last.loss,
        last_gradient_norm=last.gradient_norm,
        measured_updates,
        state_batch,
        wall_seconds=timed.time,
        cpu_seconds,
        updates_per_second=measured_updates / timed.time,
        states_per_second=measured_updates * state_batch / timed.time,
        allocation_bytes=timed.bytes,
        allocation_bytes_per_update=timed.bytes / measured_updates,
        gc_seconds=timed.gctime,
        gc_fraction=timed.gctime / timed.time,
        whole_machine_cpu_utilization_percent=
            100.0 * cpu_seconds /
            (timed.time * Threads.nthreads(:default)),
        active_worker_cpu_utilization_percent=
            100.0 * cpu_seconds / timed.time,
    )
end

function benchmark_barrierless(
    model,
    ps,
    st,
    optimizer,
    batch,
    warmup_updates::Int,
    measured_updates::Int,
    structure_weight::Float32,
    active_workers::Int,
    chunk_size::Int,
    cpuset_mode::Symbol,
)
    optimizer_state = Optimisers.setup(optimizer, ps)
    executor = BarrierlessExecutor(
        ; active_workers, chunk_size, cpuset_mode,
    )
    team = run_with_barrierless_team!(executor) do running
        last = nothing
        for _ in 1:warmup_updates
            last = barrierless_update!(
                running,
                model,
                ps,
                st,
                optimizer_state,
                batch;
                structure_weight,
            )
            ps = last.parameters
            optimizer_state = last.optimizer_state
        end
        GC.gc()
        records = Any[]
        cpu_started = CpuSets.process_cpu_ticks_100ns()
        timed = @timed begin
            for _ in 1:measured_updates
                last = barrierless_update!(
                    running,
                    model,
                    ps,
                    st,
                    optimizer_state,
                    batch;
                    structure_weight,
                )
                ps = last.parameters
                optimizer_state = last.optimizer_state
                # Keep only compact telemetry. Retaining each chunk's pullback
                # gradient/result would make the benchmark itself consume
                # several model copies and distort GC/allocation measurements.
                push!(records, (;
                    metrics=last.metrics,
                    optimizer_seconds=last.optimizer_seconds,
                ))
            end
        end
        cpu_seconds =
            (CpuSets.process_cpu_ticks_100ns() - cpu_started) * 1.0e-7
        state_batch = size(batch.mask, 2)
        worker_jobs = [
            (;
                worker=slot,
                forward_jobs=sum(
                    record.metrics.worker.per_worker[slot].forward_jobs
                    for record in records
                ),
                backward_jobs=sum(
                    record.metrics.worker.per_worker[slot].backward_jobs
                    for record in records
                ),
                cpu_seconds=sum(
                    record.metrics.worker.per_worker[slot].forward_cpu_seconds +
                    record.metrics.worker.per_worker[slot].backward_cpu_seconds
                    for record in records
                ),
            )
            for slot in 1:active_workers
        ]
        return (;
            mode=:barrierless,
            parameters=ps,
            optimizer_state,
            last_loss=last.loss,
            last_gradient_norm=tree_norm(last.gradient),
            measured_updates,
            state_batch,
            active_workers,
            chunk_size,
            cpuset_mode,
            chunks_per_update=last.metrics.chunks,
            candidates_per_update=last.metrics.candidates,
            wall_seconds=timed.time,
            cpu_seconds,
            updates_per_second=measured_updates / timed.time,
            states_per_second=measured_updates * state_batch / timed.time,
            allocation_bytes=timed.bytes,
            allocation_bytes_per_update=timed.bytes / measured_updates,
            gc_seconds=timed.gctime,
            gc_fraction=timed.gctime / timed.time,
            whole_machine_cpu_utilization_percent=
                100.0 * cpu_seconds /
                (timed.time * Threads.nthreads(:default)),
            active_worker_cpu_utilization_percent=
                100.0 * cpu_seconds / (timed.time * active_workers),
            mean_forward_seconds=summarize_phase(records, :forward_seconds),
            mean_loss_vjp_seconds=summarize_phase(records, :loss_vjp_seconds),
            mean_backward_seconds=summarize_phase(records, :backward_seconds),
            mean_reduction_seconds=summarize_phase(records, :reduction_seconds),
            mean_optimizer_seconds=mean(record.optimizer_seconds for record in records),
            worker_jobs,
        )
    end
    result = team.result
    return merge(
        result,
        (;
            binding_plan=team.binding_plan,
            bindings=team.bindings[1:active_workers],
        ),
    )
end

function without_training_state(result::NamedTuple)
    retained = filter(
        key -> key ∉ (:parameters, :optimizer_state),
        keys(result),
    )
    return NamedTuple{Tuple(retained)}(
        Tuple(getproperty(result, key) for key in retained),
    )
end

function main()
    Threads.nthreads(:interactive) == 0 || error(
        "launch with --threads=N,0",
    )
    BLAS.set_num_threads(1)
    mode = Symbol(lowercase(get(ENV, "SWSNN_BENCH_MODE", "serial")))
    preset = Symbol(get(ENV, "SWSNN_PRESET", "scaled"))
    state_batch = env_int("SWSNN_STATE_BATCH", 1)
    warmup_updates = env_int("SWSNN_WARMUP_UPDATES", 1; minimum=0)
    measured_updates = env_int("SWSNN_MEASURED_UPDATES", 3)
    active_workers = env_int(
        "SWSNN_ACTIVE_WORKERS",
        Threads.nthreads(:default),
    )
    chunk_size = env_int("SWSNN_CHUNK_SIZE", 4)
    cpuset_mode = Symbol(lowercase(get(ENV, "SWSNN_CPUSET_MODE", "none")))
    structure_weight = parse(
        Float32,
        get(ENV, "SWSNN_STRUCTURE_WEIGHT", "0.01"),
    )
    dataset_path = abspath(get(ENV, "SWSNN_DATASET", DEFAULT_DATASET))

    dataset = load_teacher_dataset(
        dataset_path;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=state_batch,
    )
    rows = training_rows_only(dataset)[1:state_batch]
    width = 16 * cld(maximum(dataset.action_counts), 16)
    batch = allocate_host_batch(state_batch; max_candidates=width)
    pack_batch!(batch, dataset, rows)
    model = build_model(preset)
    ps, st = Lux.setup(Xoshiro(MODEL_SEED), model)
    optimizer = Optimisers.AdamW(5.0f-4, (0.9, 0.999), 1.0f-5)
    topology = graph_topology(model, ps)

    result = if mode === :serial
        benchmark_serial(
            model,
            ps,
            st,
            optimizer,
            batch,
            warmup_updates,
            measured_updates,
            structure_weight,
        )
    elseif mode === :barrierless
        benchmark_barrierless(
            model,
            ps,
            st,
            optimizer,
            batch,
            warmup_updates,
            measured_updates,
            structure_weight,
            active_workers,
            chunk_size,
            cpuset_mode,
        )
    else
        error("SWSNN_BENCH_MODE must be serial or barrierless")
    end
    report = (;
        conditions=(;
            mode,
            preset,
            topology,
            parameter_count=parameter_count(ps),
            julia_threads=Threads.nthreads(:default),
            blas_threads=BLAS.get_num_threads(),
            state_batch,
            warmup_updates,
            measured_updates,
            rows,
            candidate_counts=dataset.action_counts[rows],
        ),
        result=without_training_state(result),
    )
    output = strip(get(ENV, "SWSNN_BENCH_OUTPUT", ""))
    if !isempty(output)
        mkpath(dirname(abspath(output)))
        open(abspath(output), "w") do io
            JSON3.pretty(io, report)
            write(io, '\n')
        end
    end
    println(JSON3.write(report))
    return report
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
