include(joinpath(@__DIR__, "common.jl"))

const STEPS = parse(Int, get(ENV, "AD_STEPS", "100"))
const BATCHES = parse.(Int, split(get(ENV, "AD_BATCHES", "16,32,64,128"), ','))
const BACKENDS = Symbol.(split(get(ENV, "AD_BACKENDS", "zygote,enzyme"), ','))
const CONTAMINATION = get(ENV, "AD_CONTAMINATION", "none observed")
const RUN_NUMERICAL_GATE = get(ENV, "AD_NUMERICAL_GATE", "true") == "true"
const OUTPUT = get(
    ENV, "AD_OUTPUT", joinpath(@__DIR__, "native_results.json")
)

function gradient_for(backend::Symbol, model, ps, st, data)
    backend === :zygote && return native_zygote_gradient(model, ps, st, data)
    backend === :enzyme && return native_enzyme_gradient(model, ps, st, data)
    error("unsupported native backend: $backend")
end

function run_backend(backend::Symbol, problem, steps::Int)
    ps = deepcopy(problem.ps)
    optimizer_state = Optimisers.setup(Optimisers.Adam(LEARNING_RATE), ps)
    times = Float64[]
    allocations = Int[]
    losses = Float64[]
    failure = nothing

    for step in 1:steps
        measurement = @timed begin
            gradient, loss = gradient_for(
                backend, problem.model, ps, problem.st, problem.data
            )
            optimizer_state, ps = Optimisers.update(
                optimizer_state, ps, gradient
            )
            (loss, finite_tree(gradient) && finite_tree(ps))
        end
        push!(times, measurement.time)
        push!(allocations, measurement.bytes)
        loss, finite = measurement.value
        push!(losses, Float64(loss))
        if !finite || !isfinite(loss)
            failure = "non-finite update at step $step"
            break
        end
        if measurement.time > 10.0 && step > 1
            failure = "steady update exceeded 10 seconds at step $step"
            break
        end
    end

    completed = length(times)
    steady = completed > 1 ? @view(times[2:end]) : times
    projected_100 = times[1] + (100 - 1) * median(steady)
    memory = process_memory_bytes()
    return (;
        backend=String(backend),
        completed_steps=completed,
        requested_steps=steps,
        first_update_seconds=times[1],
        steady_median_seconds=median(steady),
        steady_updates_per_second=inv(median(steady)),
        measured_total_seconds=sum(times),
        projected_compile_inclusive_100_seconds=projected_100,
        steady_median_allocated_bytes=Int(round(median(allocations[2:end]))),
        initial_loss=first(losses),
        final_loss=last(losses),
        finite=failure === nothing,
        exit_reason=something(failure, "completed"),
        peak_working_set_bytes=memory.peak,
        current_working_set_bytes=memory.current,
    )
end

results = Any[]
for batch in BATCHES
    problem = make_problem(batch)

    reference_loss = missing
    enzyme_loss = missing
    numerical = missing
    if RUN_NUMERICAL_GATE
        reference_gradient, reference_loss = native_zygote_gradient(
            problem.model, problem.ps, problem.st, problem.data
        )
        enzyme_gradient, enzyme_loss = native_enzyme_gradient(
            problem.model, problem.ps, problem.st, problem.data
        )
        numerical = numerical_comparison(reference_gradient, enzyme_gradient)
    end

    batch_runs = Any[]
    for backend in BACKENDS
        run = run_backend(backend, problem, STEPS)
        push!(batch_runs, run)
        @printf(
            "%7s b=%3d first=%8.3fs steady=%8.3fs (%6.2f update/s) steps=%d reason=%s\n",
            backend,
            batch,
            run.first_update_seconds,
            run.steady_median_seconds,
            run.steady_updates_per_second,
            run.completed_steps,
            run.exit_reason,
        )
        flush(stdout)
    end
    push!(results, (;
        batch,
        parameter_count=parameter_count(problem.ps),
        zygote_loss=reference_loss,
        enzyme_loss,
        enzyme_vs_zygote=numerical,
        runs=batch_runs,
    ))
end

document = (;
    generated_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
    versions=package_versions(),
    threads=Threads.nthreads(),
    blas_threads=BLAS.get_num_threads(),
    contamination=CONTAMINATION,
    numerical_gate_preceded_timing=RUN_NUMERICAL_GATE,
    results,
)
write_json(OUTPUT, document)
