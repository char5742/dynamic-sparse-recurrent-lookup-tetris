include(joinpath(@__DIR__, "common.jl"))

const STATE_BATCH = parse(Int, get(ENV, "AD_RETRY_STATE_BATCH", "4"))
const OUTPUT = get(
    ENV,
    "AD_RETRY_OUTPUT",
    joinpath(@__DIR__, "artifacts", "native_numerics_b$(STATE_BATCH).json"),
)
BLAS.set_num_threads(parse(Int, get(ENV, "AD_RETRY_BLAS_THREADS", "10")))

problem = load_fixed_problem(STATE_BATCH)
zygote_gradient, zygote_loss = native_zygote_gradient(problem, problem.parameters)
enzyme_shadow = Enzyme.make_zero(problem.parameters)
static_activity_result = if get(ENV, "AD_RETRY_SKIP_STATIC", "false") == "true"
    (; status="failed_in_preserved_prior_attempt", error="EnzymeRuntimeActivityError; see native_numerics_b4_static_failure.log")
else
    try
        native_enzyme_gradient!(enzyme_shadow, problem, problem.parameters)
        (; status="completed", error=nothing)
    catch exception
        (; status="failed", error=sprint(showerror, exception))
    end
end
enzyme_gradient, enzyme_loss = native_enzyme_gradient!(
    enzyme_shadow, problem, problem.parameters; runtime_activity=true
)

zygote_parameters = deepcopy(problem.parameters)
enzyme_parameters = deepcopy(problem.parameters)
zygote_optimizer = Optimisers.setup(OPTIMIZER, zygote_parameters)
enzyme_optimizer = Optimisers.setup(OPTIMIZER, enzyme_parameters)
Optimisers.update!(zygote_optimizer, zygote_parameters, zygote_gradient)
Optimisers.update!(enzyme_optimizer, enzyme_parameters, enzyme_gradient)

document = (;
    status="completed",
    generated_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
    versions=package_versions(),
    julia_threads=Threads.nthreads(),
    blas_threads=BLAS.get_num_threads(),
    workload=input_provenance(problem),
    activities=(;
        mode="ReverseWithPrimal",
        return_value="Active",
        parameters="Duplicated with preallocated shadow",
        objective_model_state_data="Const",
        runtime_activity=true,
        separate_loss_forward=false,
    ),
    static_activity_attempt=static_activity_result,
    losses=(; zygote=zygote_loss, enzyme=enzyme_loss, absolute_error=abs(zygote_loss - enzyme_loss)),
    enzyme_gradient_vs_zygote=numerical_comparison(zygote_gradient, enzyme_gradient),
    one_update_enzyme_parameters_vs_zygote=numerical_comparison(zygote_parameters, enzyme_parameters),
)
write_json(OUTPUT, document)
println(JSON3.pretty(document))
