include(joinpath(@__DIR__, "common.jl"))

const BACKEND = Symbol(get(ENV, "AD_RETRY_BACKEND", "zygote"))
const STATE_BATCH = parse(Int, get(ENV, "AD_RETRY_STATE_BATCH", "4"))
const STEPS = parse(Int, get(ENV, "AD_RETRY_STEPS", "100"))
const BLAS_THREADS = parse(Int, get(ENV, "AD_RETRY_BLAS_THREADS", "10"))
const OUTPUT = get(
    ENV,
    "AD_RETRY_OUTPUT",
    joinpath(@__DIR__, "artifacts", "$(BACKEND)_b$(STATE_BATCH)_n$(STEPS).json"),
)

BLAS.set_num_threads(BLAS_THREADS)

function main()
    problem = load_fixed_problem(STATE_BATCH)
    parameters = deepcopy(problem.parameters)
    optimizer_state = Optimisers.setup(OPTIMIZER, parameters)
    shadow = BACKEND in (:enzyme_direct, :enzyme_direct_runtime) ? Enzyme.make_zero(parameters) : nothing
    train_state = BACKEND in (:enzyme_lux, :enzyme_lux_runtime) ? new_train_state(problem) : nothing

    times = Float64[]
    allocations = Int[]
    gc_times = Float64[]
    losses = Float64[]
    start_wall = time_ns()

    for step in 1:STEPS
        measurement = if BACKEND === :zygote
            @timed begin
                gradient, loss = native_zygote_gradient(problem, parameters)
                Optimisers.update!(optimizer_state, parameters, gradient)
                (gradient, loss, parameters)
            end
        elseif BACKEND in (:enzyme_direct, :enzyme_direct_runtime)
            @timed begin
                gradient, loss = native_enzyme_gradient!(
                    shadow,
                    problem,
                    parameters;
                    runtime_activity=BACKEND === :enzyme_direct_runtime,
                )
                Optimisers.update!(optimizer_state, parameters, gradient)
                (gradient, loss, parameters)
            end
        elseif BACKEND in (:enzyme_lux, :enzyme_lux_runtime)
            @timed begin
                ad = BACKEND === :enzyme_lux_runtime ?
                     AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse)) :
                     AutoEnzyme()
                gradient, loss, _, train_state = Lux.Training.single_train_step!(
                    ad, problem.objective, problem.batch, train_state
                )
                (gradient, loss, train_state.parameters)
            end
        else
            error("unknown native backend $BACKEND")
        end
        gradient, loss, updated_parameters = measurement.value
        finite_tree(gradient) || error("non-finite gradient at update $step")
        finite_tree(updated_parameters) || error("non-finite parameters at update $step")
        isfinite(loss) || error("non-finite loss at update $step")
        push!(times, measurement.time)
        push!(allocations, measurement.bytes)
        push!(gc_times, measurement.gctime)
        push!(losses, Float64(loss))
        if step <= 5 || step in (10, 100, 500, 1000)
            @printf("%s batch=%d step=%d seconds=%.6f loss=%.9f\n", BACKEND, STATE_BATCH, step, measurement.time, loss)
            flush(stdout)
        end
    end

    final_parameters = BACKEND in (:enzyme_lux, :enzyme_lux_runtime) ? train_state.parameters : parameters
    memory = process_memory_bytes()
    document = (;
        status="completed",
        generated_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        backend=String(BACKEND),
        versions=package_versions(),
        julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
        workload=input_provenance(problem),
        activities=BACKEND in (:enzyme_direct, :enzyme_direct_runtime) ? (;
            mode="ReverseWithPrimal",
            return_value="Active",
            parameters="Duplicated with preallocated/reused shadow",
            objective_model_state_data="Const",
            runtime_activity=BACKEND === :enzyme_direct_runtime,
            separate_loss_forward=false,
        ) : nothing,
        requested_steps=STEPS,
        completed_steps=length(times),
        full_loop_wall_seconds=(time_ns() - start_wall) / 1.0e9,
        first_update_seconds=first(times),
        steady_median_seconds=median(@view times[min(6, length(times)):end]),
        steady_median_allocated_bytes=Int(round(median(@view allocations[min(6, length(allocations)):end]))),
        windows=window_summaries(times, allocations, gc_times),
        times,
        allocations,
        losses,
        final_parameters=flat_parameters(final_parameters),
        peak_working_set_bytes=memory.peak,
        current_working_set_bytes=memory.current,
    )
    write_json(OUTPUT, document)
    @printf("wrote %s wall=%.6f peak=%.3fGiB\n", OUTPUT, document.full_loop_wall_seconds, memory.peak / 2.0^30)
end

main()
