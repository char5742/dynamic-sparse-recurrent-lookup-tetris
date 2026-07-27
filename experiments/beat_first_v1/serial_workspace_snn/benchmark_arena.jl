using JSON3
using LinearAlgebra
using Lux
using Random

include(joinpath(@__DIR__, "SerialWorkspaceSNN.jl"))
include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ArenaWorkspaceTraining.jl"))
using .SerialWorkspaceSNN
using .BeatFirstTrainingCore
using .ArenaWorkspaceTraining

const DATASET_PATH = raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const MODEL_SEED = UInt64(2026072703)
const BENCHMARK_CONFIGS = (
    (state_batch=4, workers=8, cpuset=:p_only),
    (state_batch=4, workers=12, cpuset=:none),
    (state_batch=4, workers=20, cpuset=:all),
    (state_batch=8, workers=20, cpuset=:all),
)

mutable struct BenchmarkSums
    wall::Float64
    cpu::Float64
    allocation_bytes::Int128
    gc::Float64
    pack::Float64
    forward::Float64
    loss::Float64
    backward::Float64
    optimizer::Float64
    whole_cpu::Float64
    active_cpu::Float64
end

BenchmarkSums() = BenchmarkSums(
    0.0, 0.0, Int128(0), 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)

function accumulate!(sums::BenchmarkSums, metrics)
    sums.wall += metrics.wall_seconds
    sums.cpu += metrics.cpu_seconds
    sums.allocation_bytes += metrics.allocation_bytes
    sums.gc += metrics.gc_seconds
    sums.pack += metrics.pack_seconds
    sums.forward += metrics.forward_seconds
    sums.loss += metrics.loss_seconds
    sums.backward += metrics.backward_seconds
    sums.optimizer += metrics.optimizer_seconds
    sums.whole_cpu += metrics.whole_machine_cpu_percent
    sums.active_cpu += metrics.active_worker_cpu_percent
    return sums
end

function training_rows_only(dataset)
    if hasproperty(dataset, :predefined_split)
        rows = findall(==(:train), dataset.predefined_split)
        !isempty(rows) && return Int.(rows)
    end
    return collect(eachindex(dataset.action_counts))
end

function run_configuration(
    model,
    initial_parameters,
    dataset,
    rows,
    width,
    config,
    warmup_updates,
    measured_updates,
)
    trainer = ArenaTrainer(
        model,
        copy_parameters(initial_parameters);
        state_batch=config.state_batch,
        width,
        parameter_shard_size=4096,
    )
    trainer.arena.rows .= @view rows[1:config.state_batch]
    executor = ArenaExecutor(
        trainer,
        dataset;
        active_workers=config.workers,
        cpuset_mode=config.cpuset,
    )
    team = run_with_arena_team!(executor) do running
        for _ in 1:warmup_updates
            arena_update!(running; structural_interval=25)
        end
        GC.gc()
        sums = BenchmarkSums()
        outer_gc_started = Base.gc_num()
        outer_cpu_started = Main.WinCpuSets.process_cpu_ticks_100ns()
        outer_wall_started = time_ns()
        for _ in 1:measured_updates
            arena_update!(running; structural_interval=25)
            accumulate!(sums, trainer.metrics)
        end
        outer_wall = (time_ns() - outer_wall_started) * 1.0e-9
        outer_cpu =
            (Main.WinCpuSets.process_cpu_ticks_100ns() - outer_cpu_started) *
            1.0e-7
        outer_gc = Base.GC_Diff(Base.gc_num(), outer_gc_started)
        worker_jobs = [
            (;
                worker=slot,
                jobs=executor.workers[slot].jobs,
                cpu_seconds=executor.workers[slot].cpu_ticks * 1.0e-7,
            )
            for slot in 1:executor.active_workers
        ]
        return (;
            state_batch=config.state_batch,
            workers=config.workers,
            cpuset=config.cpuset,
            warmup_updates,
            measured_updates,
            candidate_counts=Int.(trainer.arena.counts),
            candidates_per_update=trainer.arena.valid_count,
            states_per_second=
                measured_updates * config.state_batch / outer_wall,
            updates_per_second=measured_updates / outer_wall,
            outer_wall_seconds=outer_wall,
            outer_cpu_seconds=outer_cpu,
            outer_whole_machine_cpu_percent=
                100.0 * outer_cpu / (outer_wall * Threads.nthreads(:default)),
            outer_active_worker_cpu_percent=
                100.0 * outer_cpu / (outer_wall * config.workers),
            outer_allocation_bytes=Int128(outer_gc.allocd),
            outer_allocation_bytes_per_update=
                Float64(outer_gc.allocd) / measured_updates,
            outer_gc_seconds=Float64(outer_gc.total_time) * 1.0e-9,
            measured_update_allocation_bytes=sums.allocation_bytes,
            measured_update_allocation_bytes_per_update=
                Float64(sums.allocation_bytes) / measured_updates,
            measured_update_gc_seconds=sums.gc,
            mean_pack_seconds=sums.pack / measured_updates,
            mean_forward_seconds=sums.forward / measured_updates,
            mean_loss_seconds=sums.loss / measured_updates,
            mean_backward_seconds=sums.backward / measured_updates,
            mean_optimizer_seconds=sums.optimizer / measured_updates,
            mean_reported_whole_machine_cpu_percent=
                sums.whole_cpu / measured_updates,
            mean_reported_active_worker_cpu_percent=
                sums.active_cpu / measured_updates,
            last_loss=trainer.last_loss.composite_loss,
            last_gradient_norm=trainer.last_gradient_norm,
            worker_jobs,
        )
    end
    return merge(
        team.result,
        (;
            bindings_verified=all(
                binding -> binding !== nothing && binding.verified,
                team.bindings,
            ),
            cpu_set_ids=[
                binding.cpu_set_id
                for binding in team.bindings[1:config.workers]
            ],
        ),
    )
end

function main()
    Threads.nthreads(:interactive) == 0 || error("launch with --threads=N,0")
    Threads.nthreads(:default) == 20 || error(
        "benchmark grid requires --threads=20,0",
    )
    BLAS.set_num_threads(1)
    warmup_updates = parse(Int, get(ENV, "SWSNN_WARMUP_UPDATES", "2"))
    measured_updates = parse(Int, get(ENV, "SWSNN_MEASURED_UPDATES", "20"))
    dataset = load_teacher_dataset(
        abspath(get(ENV, "SWSNN_DATASET", DATASET_PATH));
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=8,
    )
    rows = training_rows_only(dataset)
    width = 16 * cld(maximum(dataset.action_counts), 16)
    model = build_model(:scaled)
    initial_parameters, _ = Lux.setup(Xoshiro(MODEL_SEED), model)
    results = Any[]
    for (index, config) in enumerate(BENCHMARK_CONFIGS)
        println(stderr, "arena benchmark $(index)/$(length(BENCHMARK_CONFIGS)): $config")
        result = run_configuration(
            model,
            initial_parameters,
            dataset,
            rows,
            width,
            config,
            warmup_updates,
            measured_updates,
        )
        push!(results, result)
        println(stderr, JSON3.write(result))
        GC.gc()
    end
    report = (;
        conditions=(;
            model=:scaled,
            graph=graph_topology(model, initial_parameters),
            parameter_count=parameter_count(initial_parameters),
            julia_threads=Threads.nthreads(:default),
            blas_threads=BLAS.get_num_threads(),
            warmup_updates,
            measured_updates,
        ),
        results,
    )
    output = abspath(get(
        ENV,
        "SWSNN_ARENA_BENCH_OUTPUT",
        joinpath(@__DIR__, "trained", "arena_tuning.json"),
    ))
    mkpath(dirname(output))
    open(output, "w") do io
        JSON3.pretty(io, report)
        write(io, '\n')
    end
    println(JSON3.write((; output, results)))
    return report
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
