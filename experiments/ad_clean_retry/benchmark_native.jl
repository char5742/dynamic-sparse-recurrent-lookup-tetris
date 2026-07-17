include(joinpath(@__DIR__, "common.jl"))

const BACKEND = Symbol(get(ENV, "AD_CLEAN_BACKEND", "zygote"))
const BATCH_SIZE = parse(Int, get(ENV, "AD_CLEAN_BATCH", "16"))
const STEPS = parse(Int, get(ENV, "AD_CLEAN_STEPS", "1000"))
const BLAS_THREADS = parse(Int, get(ENV, "AD_CLEAN_BLAS_THREADS", "20"))
const OUTPUT = get(
    ENV,
    "AD_CLEAN_OUTPUT",
    joinpath(@__DIR__, "artifacts", "$(BACKEND)_b$(BATCH_SIZE)_n$(STEPS).json"),
)

BLAS.set_num_threads(BLAS_THREADS)

function main()
    BACKEND in (:zygote, :enzyme_static, :enzyme_runtime) || error("unknown backend: $BACKEND")
    problem = make_problem(BATCH_SIZE)
    parameters = deepcopy(problem.parameters)
    optimizer_state = Optimisers.setup(OPTIMIZER, parameters)
    shadow = BACKEND === :zygote ? nothing : Enzyme.make_zero(parameters)
    before_updates_memory = process_memory_bytes()

    times = Float64[]
    allocations = Int[]
    gc_times = Float64[]
    losses = Float64[]
    started = time_ns()

    for step in 1:STEPS
        measurement = if BACKEND === :zygote
            @timed begin
                gradient, loss = zygote_gradient(problem, parameters)
                Optimisers.update!(optimizer_state, parameters, gradient)
                loss
            end
        else
            mode = BACKEND === :enzyme_static ? :static : :runtime
            @timed begin
                gradient, loss = enzyme_gradient!(shadow, problem, parameters, mode)
                Optimisers.update!(optimizer_state, parameters, gradient)
                loss
            end
        end
        loss = Float64(measurement.value)
        isfinite(loss) || error("non-finite loss at update $step")
        push!(times, measurement.time)
        push!(allocations, measurement.bytes)
        push!(gc_times, measurement.gctime)
        push!(losses, loss)
        if step <= 3 || step in (10, 16, 64, 100, 500, 1000)
            finite_tree(parameters) || error("non-finite parameters at update $step")
            @printf("%s batch=%d step=%d seconds=%.9f bytes=%d loss=%.9f\n", BACKEND, BATCH_SIZE, step, measurement.time, measurement.bytes, loss)
            flush(stdout)
        end
    end

    after_updates_memory = process_memory_bytes()
    finite_tree(parameters) || error("non-finite final parameters")
    document = (;
        status="completed",
        generated_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        backend=String(BACKEND),
        implementation=BACKEND === :zygote ? "Zygote.withgradient + Optimisers.update!" : "standalone Enzyme.autodiff + Optimisers.update!",
        enzyme_activity=BACKEND === :zygote ? nothing : String(BACKEND === :enzyme_static ? :static : :runtime),
        return_primal_from_ad=true,
        separate_loss_forward=false,
        optimizer_update_inside_timing=true,
        completion_barrier_inside_timing=true,
        versions=package_versions(),
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
        times,
        allocations,
        gc_times,
        losses,
        final_parameter_sha256=array_sha256(tree_arrays(parameters)...),
    )
    write_json(OUTPUT, document)
    @printf("wrote %s steady=%.3f step/s peak=%.3f GiB\n", OUTPUT, document.steady.steady_updates_per_second, after_updates_memory.peak / 2.0^30)
end

main()
