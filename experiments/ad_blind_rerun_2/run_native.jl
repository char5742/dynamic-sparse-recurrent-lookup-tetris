include(joinpath(@__DIR__, "common.jl"))

const BACKEND = Symbol(get(ENV, "AD_BLIND_BACKEND", "zygote"))
const STATE_BATCH = parse(Int, get(ENV, "AD_BLIND_STATE_BATCH", "4"))
const STEPS = parse(Int, get(ENV, "AD_BLIND_STEPS", "100"))
const BLAS_THREADS = parse(Int, get(ENV, "AD_BLIND_BLAS_THREADS", "10"))
const OUTPUT = get(
    ENV,
    "AD_BLIND_OUTPUT",
    joinpath(BLIND_OUTPUT_ROOT, "$(BACKEND)_b$(STATE_BATCH)_n$(STEPS).json"),
)

BLAS.set_num_threads(BLAS_THREADS)

function main()
    BACKEND in (:zygote, :enzyme_runtime, :enzyme_runtime_strongzero, :enzyme_static) ||
        error("unknown backend $BACKEND")
    problem = load_fixed_problem(STATE_BATCH)
    parameters = deepcopy(problem.parameters)
    optimizer_state = Optimisers.setup(OPTIMIZER, parameters)
    shadow = BACKEND === :zygote ? nothing : Enzyme.make_zero(parameters)

    times = Float64[]
    allocations = Int[]
    gc_times = Float64[]
    losses = Float64[]
    parameter_checkpoints = Dict{String,Any}()
    checkpoints = checkpoint_indices(STEPS)
    started = time_ns()

    for step in 1:STEPS
        measurement = if BACKEND === :zygote
            @timed begin
                gradient, loss = native_zygote_gradient(problem, parameters)
                Optimisers.update!(optimizer_state, parameters, gradient)
                loss
            end
        else
            snapshot = step == 1 ? primal_snapshot(parameters, problem) : nothing
            @timed begin
                gradient, loss = direct_enzyme_gradient!(shadow, problem, parameters, BACKEND)
                if step == 1
                    mutation = primal_mutation_report(snapshot, parameters, problem)
                    mutation.unchanged || error(
                        "invalid standalone Enzyme path: gradient call mutated primal inputs: $mutation"
                    )
                end
                Optimisers.update!(optimizer_state, parameters, gradient)
                loss
            end
        end
        loss = measurement.value
        finite_tree(parameters) || error("non-finite parameters at update $step")
        isfinite(loss) || error("non-finite loss at update $step")
        push!(times, measurement.time)
        push!(allocations, measurement.bytes)
        push!(gc_times, measurement.gctime)
        push!(losses, Float64(loss))
        if step in checkpoints
            parameter_checkpoints[string(step)] = flat_parameters(parameters)
            @printf(
                "%s batch=%d step=%d seconds=%.6f bytes=%d loss=%.9f\n",
                BACKEND,
                STATE_BATCH,
                step,
                measurement.time,
                measurement.bytes,
                loss,
            )
            flush(stdout)
        end
    end

    memory = process_memory_bytes()
    document = (;
        status="completed",
        generated_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        backend=String(BACKEND),
        versions=package_versions(),
        julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
        workload=input_provenance(problem),
        source=blind_source_provenance(),
        warmup_policy="first five trajectory updates excluded from warm steady statistics",
        ad=BACKEND === :zygote ? (;
            implementation="Zygote.withgradient",
            separate_loss_forward=false,
        ) : (;
            implementation="standalone Enzyme.autodiff",
            mode=BACKEND === :enzyme_static ? "ReverseWithPrimal" :
                 BACKEND === :enzyme_runtime ? "set_runtime_activity(ReverseWithPrimal)" :
                 "set_strong_zero(set_runtime_activity(ReverseWithPrimal))",
            return_value="Active",
            parameters="Duplicated with one preallocated/reused shadow",
            shadow_zeroed_before_each_update=true,
            objective_model_state_batch="Const",
            primal_input_mutation_guard=true,
            separate_loss_forward=false,
            split_tape=false,
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
        final_parameters=flat_parameters(parameters),
        peak_working_set_bytes=memory.peak,
        current_working_set_bytes=memory.current,
    )
    write_json(OUTPUT, document)
    @printf("wrote %s wall=%.6f peak=%.3fGiB\n", OUTPUT, document.full_loop_wall_seconds, memory.peak / 2.0^30)
end

main()
