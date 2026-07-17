include(joinpath(@__DIR__, "common.jl"))

const BATCH_SIZE = parse(Int, get(ENV, "AD_CLEAN_BATCH", "16"))
const BLAS_THREADS = parse(Int, get(ENV, "AD_CLEAN_BLAS_THREADS", "20"))
const OUTPUT = get(ENV, "AD_CLEAN_OUTPUT", joinpath(@__DIR__, "artifacts", "numerics_native_b$(BATCH_SIZE).json"))

BLAS.set_num_threads(BLAS_THREADS)

function try_enzyme(problem, zygote_gradient_value, zygote_loss, mode::Symbol)
    shadow = Enzyme.make_zero(problem.parameters)
    try
        enzyme_gradient_value, enzyme_loss = enzyme_gradient!(shadow, problem, problem.parameters, mode)
        zygote_parameters = deepcopy(problem.parameters)
        enzyme_parameters = deepcopy(problem.parameters)
        zygote_optimizer = Optimisers.setup(OPTIMIZER, zygote_parameters)
        enzyme_optimizer = Optimisers.setup(OPTIMIZER, enzyme_parameters)
        Optimisers.update!(zygote_optimizer, zygote_parameters, zygote_gradient_value)
        Optimisers.update!(enzyme_optimizer, enzyme_parameters, enzyme_gradient_value)
        return (;
            status="completed",
            error=nothing,
            zygote_loss=Float64(zygote_loss),
            enzyme_loss=Float64(enzyme_loss),
            loss_absolute_error=abs(Float64(enzyme_loss) - Float64(zygote_loss)),
            gradient_vs_zygote=numeric_comparison(zygote_gradient_value, enzyme_gradient_value),
            one_update_parameters_vs_zygote=numeric_comparison(zygote_parameters, enzyme_parameters),
        )
    catch exception
        return (; status="failed", error=sprint(showerror, exception, catch_backtrace()))
    end
end

function main()
    problem = make_problem(BATCH_SIZE)
    gradient, loss = zygote_gradient(problem, problem.parameters)
    document = (;
        status="completed",
        generated_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        batch_size=BATCH_SIZE,
        versions=package_versions(),
        julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
        workload=workload_description(problem),
        enzyme_static=try_enzyme(problem, gradient, loss, :static),
        enzyme_runtime=try_enzyme(problem, gradient, loss, :runtime),
    )
    write_json(OUTPUT, document)
    println(JSON3.pretty(document))
end

main()
