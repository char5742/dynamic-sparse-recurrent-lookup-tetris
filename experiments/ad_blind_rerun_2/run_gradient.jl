include(joinpath(@__DIR__, "common.jl"))

const BACKEND = Symbol(get(ENV, "AD_BLIND_BACKEND", "zygote"))
const STATE_BATCH = parse(Int, get(ENV, "AD_BLIND_STATE_BATCH", "4"))
const CALLS = parse(Int, get(ENV, "AD_BLIND_CALLS", "100"))
const BLAS_THREADS = parse(Int, get(ENV, "AD_BLIND_BLAS_THREADS", "10"))
const OUTPUT = get(
    ENV,
    "AD_BLIND_OUTPUT",
    joinpath(BLIND_OUTPUT_ROOT, "gradient_$(BACKEND)_b$(STATE_BATCH)_n$(CALLS).json"),
)
BLAS.set_num_threads(BLAS_THREADS)

function main()
    BACKEND in (:zygote, :enzyme_runtime) || error("unsupported backend $BACKEND")
    problem = load_fixed_problem(STATE_BATCH)
    parameters = deepcopy(problem.parameters)
    shadow = BACKEND === :enzyme_runtime ? Enzyme.make_zero(parameters) : nothing
    times = Float64[]
    allocations = Int[]
    gc_times = Float64[]
    losses = Float64[]

    for call in 1:CALLS
        measurement = if BACKEND === :zygote
            @timed native_zygote_gradient(problem, parameters)
        else
            @timed direct_enzyme_gradient!(shadow, problem, parameters, BACKEND)
        end
        gradient, loss = measurement.value
        finite_tree(gradient) || error("non-finite gradient at call $call")
        push!(times, measurement.time)
        push!(allocations, measurement.bytes)
        push!(gc_times, measurement.gctime)
        push!(losses, Float64(loss))
    end

    memory = process_memory_bytes()
    document = (;
        status="completed",
        generated_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        benchmark="gradient call only; no optimizer setup or update in timed region",
        backend=String(BACKEND),
        versions=package_versions(),
        julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
        workload=input_provenance(problem),
        source=blind_source_provenance(),
        requested_calls=CALLS,
        completed_calls=length(times),
        first_call_seconds=first(times),
        warmup_policy="first five repeated fixed-parameter calls excluded",
        warm_median_seconds=median(@view times[min(6, length(times)):end]),
        warm_median_allocated_bytes=Int(round(median(@view allocations[min(6, length(allocations)):end]))),
        windows=window_summaries(times, allocations, gc_times),
        losses,
        times,
        allocations,
        gc_times,
        peak_working_set_bytes=memory.peak,
        current_working_set_bytes=memory.current,
    )
    write_json(OUTPUT, document)
    @printf(
        "wrote %s first=%.6f warm=%.6f bytes=%d\n",
        OUTPUT,
        document.first_call_seconds,
        document.warm_median_seconds,
        document.warm_median_allocated_bytes,
    )
end

main()
