using JSON3
using LinearAlgebra
using Lux
using Optimisers
using Random

include(joinpath(@__DIR__, "benchmark_barrierless.jl"))

const DEFAULT_TUNING_CONFIGS = (
    (mode=:serial, state_batch=1, workers=1, chunk=0, cpuset=:none),
    (mode=:barrierless, state_batch=1, workers=4, chunk=8, cpuset=:none),
    (mode=:barrierless, state_batch=1, workers=8, chunk=4, cpuset=:none),
    (mode=:barrierless, state_batch=1, workers=8, chunk=4, cpuset=:p_only),
    (mode=:barrierless, state_batch=1, workers=12, chunk=2, cpuset=:none),
    (mode=:barrierless, state_batch=1, workers=20, chunk=2, cpuset=:all),
    (mode=:barrierless, state_batch=1, workers=20, chunk=1, cpuset=:all),
    (mode=:serial, state_batch=4, workers=1, chunk=0, cpuset=:none),
    (mode=:barrierless, state_batch=4, workers=8, chunk=8, cpuset=:p_only),
    (mode=:barrierless, state_batch=4, workers=12, chunk=8, cpuset=:none),
    (mode=:barrierless, state_batch=4, workers=20, chunk=8, cpuset=:all),
    (mode=:barrierless, state_batch=4, workers=20, chunk=4, cpuset=:all),
)

const CHUNK_TUNING_CONFIGS = (
    (mode=:barrierless, state_batch=4, workers=4, chunk=32, cpuset=:none),
    (mode=:barrierless, state_batch=4, workers=8, chunk=32, cpuset=:none),
    (mode=:barrierless, state_batch=4, workers=8, chunk=32, cpuset=:p_only),
    (mode=:barrierless, state_batch=4, workers=8, chunk=16, cpuset=:p_only),
    (mode=:barrierless, state_batch=4, workers=12, chunk=16, cpuset=:none),
    (mode=:serial, state_batch=8, workers=1, chunk=0, cpuset=:none),
    (mode=:barrierless, state_batch=8, workers=8, chunk=32, cpuset=:p_only),
    (mode=:barrierless, state_batch=8, workers=8, chunk=16, cpuset=:p_only),
    (mode=:barrierless, state_batch=8, workers=12, chunk=32, cpuset=:none),
    (mode=:barrierless, state_batch=8, workers=20, chunk=16, cpuset=:all),
)

function compact_result(result)
    common = (;
        mode=result.mode,
        last_loss=result.last_loss,
        last_gradient_norm=result.last_gradient_norm,
        measured_updates=result.measured_updates,
        state_batch=result.state_batch,
        wall_seconds=result.wall_seconds,
        cpu_seconds=result.cpu_seconds,
        updates_per_second=result.updates_per_second,
        states_per_second=result.states_per_second,
        allocation_bytes_per_update=result.allocation_bytes_per_update,
        gc_seconds=result.gc_seconds,
        gc_fraction=result.gc_fraction,
        whole_machine_cpu_utilization_percent=
            result.whole_machine_cpu_utilization_percent,
        active_worker_cpu_utilization_percent=
            result.active_worker_cpu_utilization_percent,
    )
    result.mode === :serial && return common
    return merge(common, (;
        active_workers=result.active_workers,
        chunk_size=result.chunk_size,
        cpuset_mode=result.cpuset_mode,
        chunks_per_update=result.chunks_per_update,
        candidates_per_update=result.candidates_per_update,
        mean_forward_seconds=result.mean_forward_seconds,
        mean_loss_vjp_seconds=result.mean_loss_vjp_seconds,
        mean_backward_seconds=result.mean_backward_seconds,
        mean_reduction_seconds=result.mean_reduction_seconds,
        mean_optimizer_seconds=result.mean_optimizer_seconds,
        worker_jobs=result.worker_jobs,
        bindings_verified=all(binding.verified for binding in result.bindings),
        cpu_set_ids=[binding.cpu_set_id for binding in result.bindings],
    ))
end

function main_tuning()
    Threads.nthreads(:interactive) == 0 || error("launch with --threads=N,0")
    Threads.nthreads(:default) == 20 || error(
        "the default tuning grid requires exactly 20 Julia threads",
    )
    BLAS.set_num_threads(1)
    warmup_updates = env_int("SWSNN_WARMUP_UPDATES", 1; minimum=0)
    measured_updates = env_int("SWSNN_MEASURED_UPDATES", 5)
    structure_weight = parse(
        Float32,
        get(ENV, "SWSNN_STRUCTURE_WEIGHT", "0.01"),
    )
    grid_name = Symbol(lowercase(get(ENV, "SWSNN_TUNING_GRID", "default")))
    configs = if grid_name === :default
        DEFAULT_TUNING_CONFIGS
    elseif grid_name === :chunks
        CHUNK_TUNING_CONFIGS
    else
        error("SWSNN_TUNING_GRID must be default or chunks")
    end
    dataset_path = abspath(get(ENV, "SWSNN_DATASET", DEFAULT_DATASET))
    maximum_state_batch = maximum(config.state_batch for config in configs)
    dataset = load_teacher_dataset(
        dataset_path;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=maximum_state_batch,
    )
    training_rows = training_rows_only(dataset)
    width = 16 * cld(maximum(dataset.action_counts), 16)
    batches = Dict{Int,Any}()
    for state_batch in unique(config.state_batch for config in configs)
        rows = training_rows[1:state_batch]
        batch = allocate_host_batch(state_batch; max_candidates=width)
        pack_batch!(batch, dataset, rows)
        batches[state_batch] = batch
    end

    model = build_model(:scaled)
    initial_parameters, states = Lux.setup(Xoshiro(MODEL_SEED), model)
    optimizer = Optimisers.AdamW(5.0f-4, (0.9, 0.999), 1.0f-5)
    results = Any[]
    for (index, config) in enumerate(configs)
        println(stderr, "tuning $(index)/$(length(configs)): $config")
        parameters = deepcopy(initial_parameters)
        batch = batches[config.state_batch]
        result = if config.mode === :serial
            benchmark_serial(
                model,
                parameters,
                states,
                optimizer,
                batch,
                warmup_updates,
                measured_updates,
                structure_weight,
            )
        else
            benchmark_barrierless(
                model,
                parameters,
                states,
                optimizer,
                batch,
                warmup_updates,
                measured_updates,
                structure_weight,
                config.workers,
                config.chunk,
                config.cpuset,
            )
        end
        compact = compact_result(result)
        push!(results, compact)
        println(stderr, JSON3.write(compact))
        GC.gc()
    end

    report = (;
        conditions=(;
            preset=:scaled,
            topology=graph_topology(model, initial_parameters),
            parameter_count=parameter_count(initial_parameters),
            julia_threads=Threads.nthreads(:default),
            blas_threads=BLAS.get_num_threads(),
            grid_name,
            warmup_updates,
            measured_updates,
            candidate_counts=Dict(
                string(state_batch) =>
                    dataset.action_counts[training_rows[1:state_batch]]
                for state_batch in keys(batches)
            ),
        ),
        results,
    )
    output = abspath(get(
        ENV,
        "SWSNN_TUNING_OUTPUT",
        joinpath(
            @__DIR__,
            "trained",
            "barrierless_tuning_grid.json",
        ),
    ))
    mkpath(dirname(output))
    open(output, "w") do io
        JSON3.pretty(io, report)
        write(io, '\n')
    end
    println(JSON3.write((; output, results)))
    return report
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_tuning()
