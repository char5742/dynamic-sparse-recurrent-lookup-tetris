include(joinpath(@__DIR__, "common.jl"))
using Reactant

const STATE_BATCH = parse(Int, get(ENV, "AD_RETRY_STATE_BATCH", "4"))
const STEPS = parse(Int, get(ENV, "AD_RETRY_STEPS", "1000"))
const BLAS_THREADS = parse(Int, get(ENV, "AD_RETRY_BLAS_THREADS", "10"))
const OUTPUT = get(
    ENV,
    "AD_RETRY_OUTPUT",
    joinpath(@__DIR__, "artifacts", "reactant_b$(STATE_BATCH)_n$(STEPS).json"),
)

BLAS.set_num_threads(BLAS_THREADS)
Reactant.set_default_backend("cpu")

function main()
    problem = load_fixed_problem(STATE_BATCH)
    device = Lux.reactant_device(; force=true)
    parameters, state, batch = device((problem.parameters, problem.state, problem.batch))
    train_state = new_train_state(problem; reactant_values=(parameters, state))
    times = Float64[]
    allocations = Int[]
    gc_times = Float64[]
    losses = Float64[]
    barrier_times = Float64[]
    thunk_ids = UInt64[]
    start_wall = time_ns()

    for step in 1:STEPS
        measurement = @timed Lux.Training.single_train_step!(
            AutoEnzyme(),
            problem.objective,
            batch,
            train_state;
            return_gradients=Val(false),
            sync=true,
        )
        _, loss, _, train_state = measurement.value
        barrier_start = time_ns()
        Reactant.synchronize(loss)
        foreach(Reactant.synchronize, tree_arrays(train_state.parameters))
        push!(barrier_times, (time_ns() - barrier_start) / 1.0e9)
        if hasfield(typeof(train_state.cache.extras), :compiled_grad_and_step_function)
            push!(thunk_ids, UInt64(objectid(train_state.cache.extras.compiled_grad_and_step_function)))
        end
        push!(times, measurement.time)
        push!(allocations, measurement.bytes)
        push!(gc_times, measurement.gctime)
        if step <= 5 || step in (10, 100, 500, 1000)
            host_loss = Float64(Reactant.to_number(loss))
            push!(losses, host_loss)
            @printf("reactant batch=%d step=%d seconds=%.6f barrier=%.9f loss=%.9f\n", STATE_BATCH, step, measurement.time, barrier_times[end], host_loss)
            flush(stdout)
        end
    end

    memory = process_memory_bytes()
    document = (;
        status="completed",
        generated_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        backend="reactant_enzyme_mlir",
        versions=package_versions(; reactant=string(Base.pkgversion(Reactant))),
        julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
        workload=input_provenance(problem),
        requested_steps=STEPS,
        completed_steps=length(times),
        full_loop_wall_seconds=(time_ns() - start_wall) / 1.0e9,
        first_update_seconds=first(times),
        steady_median_seconds=median(@view times[min(6, length(times)):end]),
        steady_median_allocated_bytes=Int(round(median(@view allocations[min(6, length(allocations)):end]))),
        windows=window_summaries(times, allocations, gc_times),
        checkpoint_losses=losses,
        final_parameters=flat_parameters(train_state.parameters),
        explicit_barrier_median_seconds=median(barrier_times),
        explicit_barrier_max_seconds=maximum(barrier_times),
        compiled_thunk_unique_ids=length(unique(thunk_ids)),
        compiled_thunk_observations=length(thunk_ids),
        no_recompile_observed=!isempty(thunk_ids) && length(unique(thunk_ids)) == 1,
        times,
        allocations,
        peak_working_set_bytes=memory.peak,
        current_working_set_bytes=memory.current,
    )
    write_json(OUTPUT, document)
    @printf("wrote %s wall=%.6f peak=%.3fGiB\n", OUTPUT, document.full_loop_wall_seconds, memory.peak / 2.0^30)
end

main()
