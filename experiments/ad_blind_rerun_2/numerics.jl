include(joinpath(@__DIR__, "common.jl"))

const STATE_BATCH = parse(Int, get(ENV, "AD_BLIND_STATE_BATCH", "4"))
const BLAS_THREADS = parse(Int, get(ENV, "AD_BLIND_BLAS_THREADS", "10"))
const OUTPUT = get(
    ENV,
    "AD_BLIND_OUTPUT",
    joinpath(BLIND_OUTPUT_ROOT, "numerics_b$(STATE_BATCH).json"),
)
BLAS.set_num_threads(BLAS_THREADS)

function gradient_diagnostics(reference, candidate)
    ref = Float64.(flat_parameters(reference))
    cand = Float64.(flat_parameters(candidate))
    delta = cand .- ref
    magnitudes = abs.(ref)
    sign_mismatch = signbit.(ref) .!= signbit.(cand)
    material_sign_mismatch = sign_mismatch .& (max.(abs.(ref), abs.(cand)) .> 1.0e-8)
    maximum_index = argmax(abs.(delta))
    return merge(
        numerical_comparison(reference, candidate),
        (;
            reference_l2=norm(ref),
            candidate_l2=norm(cand),
            maximum_error_flat_index=maximum_index,
            maximum_error_reference=ref[maximum_index],
            maximum_error_candidate=cand[maximum_index],
            exact_zero_reference_count=count(iszero, ref),
            exact_zero_candidate_count=count(iszero, cand),
            sign_mismatch_count=count(sign_mismatch),
            material_sign_mismatch_count=count(material_sign_mismatch),
            reference_abs_quantiles=(
                q0=minimum(magnitudes),
                q25=quantile(magnitudes, 0.25),
                q50=quantile(magnitudes, 0.50),
                q75=quantile(magnitudes, 0.75),
                q99=quantile(magnitudes, 0.99),
                q100=maximum(magnitudes),
            ),
        ),
    )
end

function one_update(problem, gradient)
    parameters = deepcopy(problem.parameters)
    optimizer_state = Optimisers.setup(OPTIMIZER, parameters)
    Optimisers.update!(optimizer_state, parameters, gradient)
    return parameters
end

problem = load_fixed_problem(STATE_BATCH)
zygote_gradient, zygote_loss = native_zygote_gradient(problem, problem.parameters)
enzyme_runtime = Enzyme.make_zero(problem.parameters)
runtime_gradient, runtime_loss = direct_enzyme_gradient!(
    enzyme_runtime, problem, problem.parameters, :enzyme_runtime
)
enzyme_strongzero = Enzyme.make_zero(problem.parameters)
strongzero_result = try
    strongzero_gradient, strongzero_loss = direct_enzyme_gradient!(
        enzyme_strongzero, problem, problem.parameters, :enzyme_runtime_strongzero
    )
    (;
        status="completed",
        loss=strongzero_loss,
        gradient=gradient_diagnostics(zygote_gradient, strongzero_gradient),
        one_update_parameters=numerical_comparison(
            one_update(problem, zygote_gradient), one_update(problem, strongzero_gradient)
        ),
    )
catch exception
    (; status="failed", error=sprint(showerror, exception))
end
static_result = try
    static_shadow = Enzyme.make_zero(problem.parameters)
    static_gradient, static_loss = direct_enzyme_gradient!(
        static_shadow, problem, problem.parameters, :enzyme_static
    )
    (;
        status="completed",
        loss=static_loss,
        gradient=gradient_diagnostics(zygote_gradient, static_gradient),
    )
catch exception
    (; status="failed", error=sprint(showerror, exception))
end

document = (;
    status="completed",
    generated_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
    versions=package_versions(),
    julia_threads=Threads.nthreads(),
    blas_threads=BLAS.get_num_threads(),
    workload=input_provenance(problem),
    source=blind_source_provenance(),
    zygote_loss,
    runtime=(;
        loss=runtime_loss,
        loss_absolute_error=abs(runtime_loss - zygote_loss),
        gradient=gradient_diagnostics(zygote_gradient, runtime_gradient),
        one_update_parameters=numerical_comparison(
            one_update(problem, zygote_gradient), one_update(problem, runtime_gradient)
        ),
    ),
    strongzero=strongzero_result,
    static=static_result,
)
write_json(OUTPUT, document)
println(JSON3.pretty(document))

