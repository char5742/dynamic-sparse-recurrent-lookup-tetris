include(joinpath(@__DIR__, "common.jl"))
using ADTypes: AutoEnzyme
using Reactant

const BATCH_SIZE = parse(Int, get(ENV, "AD_CLEAN_BATCH", "16"))
const STEPS = parse(Int, get(ENV, "AD_CLEAN_STEPS", "1000"))
const BLAS_THREADS = parse(Int, get(ENV, "AD_CLEAN_BLAS_THREADS", "20"))
const OUTPUT = get(
    ENV,
    "AD_CLEAN_OUTPUT",
    joinpath(@__DIR__, "artifacts", "reactant_b$(BATCH_SIZE)_n$(STEPS).json"),
)

BLAS.set_num_threads(BLAS_THREADS)
Reactant.set_default_backend("cpu")

function main()
    problem = make_problem(BATCH_SIZE)
    device = Lux.reactant_device(; force=true)
    parameters, state, batch = device((problem.parameters, problem.state, problem.batch))
    train_state = Lux.Training.TrainState(problem.model, parameters, state, OPTIMIZER)
    before_updates_memory = process_memory_bytes()

    times = Float64[]
    allocations = Int[]
    gc_times = Float64[]
    losses = Float64[]
    compiled_thunk_ids = UInt64[]
    started = time_ns()

    for step in 1:STEPS
        measurement = @timed begin
            _, loss, _, train_state = Lux.Training.single_train_step!(
                AutoEnzyme(),
                problem.objective,
                batch,
                train_state;
                return_gradients=Val(false),
                sync=true,
            )
            Reactant.synchronize(loss)
            foreach(Reactant.synchronize, tree_arrays(train_state.parameters))
            (train_state, loss)
        end
        train_state, loss = measurement.value
        host_loss = Float64(Reactant.to_number(loss))
        isfinite(host_loss) || error("non-finite loss at update $step")
        push!(times, measurement.time)
        push!(allocations, measurement.bytes)
        push!(gc_times, measurement.gctime)
        push!(losses, host_loss)
        if hasfield(typeof(train_state.cache.extras), :compiled_grad_and_step_function)
            push!(compiled_thunk_ids, UInt64(objectid(train_state.cache.extras.compiled_grad_and_step_function)))
        end
        if step <= 3 || step in (10, 16, 64, 100, 500, 1000)
            @printf("reactant batch=%d step=%d seconds=%.9f bytes=%d loss=%.9f\n", BATCH_SIZE, step, measurement.time, measurement.bytes, host_loss)
            flush(stdout)
        end
    end

    after_updates_memory = process_memory_bytes()
    finite_tree(train_state.parameters) || error("non-finite final parameters")
    document = (;
        status="completed",
        generated_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        backend="reactant_enzyme_mlir",
        implementation="Lux.Training.single_train_step! + AutoEnzyme + Reactant CPU",
        return_primal_from_ad=true,
        return_gradients=false,
        separate_loss_forward=false,
        optimizer_update_inside_timing=true,
        optimizer_fused_in_compiled_step=true,
        completion_barrier_inside_timing=true,
        versions=package_versions(; reactant=string(Base.pkgversion(Reactant))),
        platform=(; kernel=string(Sys.KERNEL), machine=string(Sys.MACHINE), cpu=Sys.CPU_NAME, word_size=Sys.WORD_SIZE),
        julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
        workload=workload_description(problem),
        source_sha256=source_hashes(),
        requested_steps=STEPS,
        completed_steps=length(times),
        full_loop_wall_seconds=(time_ns() - started) / 1.0e9,
        first_update_compile_inclusive_seconds=first(times),
        steady=steady_summary(times, allocations, gc_times),
        checkpoints=checkpoint_summaries(times, allocations),
        stability=(; all_losses_finite=all(isfinite, losses), initial_loss=first(losses), final_loss=last(losses), minimum_loss=minimum(losses), maximum_loss=maximum(losses), final_parameters_finite=true),
        process_memory=(; before_updates=before_updates_memory, after_updates=after_updates_memory),
        compiled_thunk_observations=length(compiled_thunk_ids),
        compiled_thunk_unique_ids=length(unique(compiled_thunk_ids)),
        no_recompile_observed=!isempty(compiled_thunk_ids) && length(unique(compiled_thunk_ids)) == 1,
        times,
        allocations,
        gc_times,
        losses,
        final_parameter_sha256=array_sha256(tree_arrays(train_state.parameters)...),
    )
    write_json(OUTPUT, document)
    @printf("wrote %s steady=%.3f step/s peak=%.3f GiB\n", OUTPUT, document.steady.steady_updates_per_second, after_updates_memory.peak / 2.0^30)
end

main()
