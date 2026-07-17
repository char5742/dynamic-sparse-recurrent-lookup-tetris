include(joinpath(@__DIR__, "common.jl"))
using Reactant

const STATE_BATCH = parse(Int, get(ENV, "AD_BLIND_STATE_BATCH", "4"))
const STEPS = parse(Int, get(ENV, "AD_BLIND_STEPS", "1000"))
const BLAS_THREADS = parse(Int, get(ENV, "AD_BLIND_BLAS_THREADS", "10"))
const OUTPUT = get(
    ENV,
    "AD_BLIND_OUTPUT",
    joinpath(BLIND_OUTPUT_ROOT, "reactant_b$(STATE_BATCH)_n$(STEPS).json"),
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
    barriers = Float64[]
    thunk_ids = UInt64[]
    parameter_checkpoints = Dict{String,Any}()
    checkpoints = checkpoint_indices(STEPS)
    started = time_ns()

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
        barrier_started = time_ns()
        Reactant.synchronize(loss)
        foreach(Reactant.synchronize, tree_arrays(train_state.parameters))
        push!(barriers, (time_ns() - barrier_started) / 1.0e9)
        if hasfield(typeof(train_state.cache.extras), :compiled_grad_and_step_function)
            push!(thunk_ids, UInt64(objectid(train_state.cache.extras.compiled_grad_and_step_function)))
        end
        host_loss = Float64(Reactant.to_number(loss))
        push!(times, measurement.time)
        push!(allocations, measurement.bytes)
        push!(gc_times, measurement.gctime)
        push!(losses, host_loss)
        if step in checkpoints
            parameter_checkpoints[string(step)] = flat_parameters(train_state.parameters)
            @printf(
                "reactant batch=%d step=%d seconds=%.6f bytes=%d barrier=%.9f loss=%.9f\n",
                STATE_BATCH,
                step,
                measurement.time,
                measurement.bytes,
                barriers[end],
                host_loss,
            )
            flush(stdout)
        end
    end

    memory = process_memory_bytes()
    document = (;
        status="completed",
        generated_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        backend="lux_reactant_enzyme_mlir",
        versions=package_versions(; reactant=string(Base.pkgversion(Reactant))),
        julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
        workload=input_provenance(problem),
        source=blind_source_provenance(),
        warmup_policy="first five trajectory updates excluded from warm steady statistics",
        ad=(;
            implementation="Lux.Training.single_train_step! + Reactant CPU + EnzymeMLIR",
            return_gradients=false,
            sync=true,
            persistent_returned_train_state=true,
            optimizer_fused_in_compiled_step=true,
            separate_loss_forward=false,
        ),
        optimizer=optimizer_description(),
        requested_steps=STEPS,
        completed_steps=length(times),
        full_loop_wall_seconds=(time_ns() - started) / 1.0e9,
        first_update_seconds=first(times),
        steady_median_seconds=median(@view times[min(6, length(times)):end]),
        steady_median_allocated_bytes=Int(round(median(@view allocations[min(6, length(allocations)):end]))),
        windows=window_summaries(times, allocations, gc_times),
        losses,
        times,
        allocations,
        gc_times,
        parameter_checkpoints,
        final_parameters=flat_parameters(train_state.parameters),
        explicit_barrier_median_seconds=median(barriers),
        explicit_barrier_max_seconds=maximum(barriers),
        compiled_thunk_unique_ids=length(unique(thunk_ids)),
        compiled_thunk_observations=length(thunk_ids),
        no_recompile_observed=!isempty(thunk_ids) && length(unique(thunk_ids)) == 1,
        peak_working_set_bytes=memory.peak,
        current_working_set_bytes=memory.current,
    )
    write_json(OUTPUT, document)
    @printf("wrote %s wall=%.6f peak=%.3fGiB\n", OUTPUT, document.full_loop_wall_seconds, memory.peak / 2.0^30)
end

main()

