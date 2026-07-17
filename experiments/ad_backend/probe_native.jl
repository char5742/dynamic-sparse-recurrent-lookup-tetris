include(joinpath(@__DIR__, "common.jl"))

batch_size = parse(Int, get(ENV, "AD_BATCH", "16"))
problem = make_problem(batch_size)
println("parameters=$(parameter_count(problem.ps)) batch=$batch_size")

zygote_seconds = @elapsed zygote_gradient, zygote_loss = native_zygote_gradient(
    problem.model, problem.ps, problem.st, problem.data
)
println("zygote loss=$zygote_loss seconds=$zygote_seconds finite=$(finite_tree(zygote_gradient))")

enzyme_seconds = @elapsed enzyme_gradient, enzyme_loss = native_enzyme_gradient(
    problem.model, problem.ps, problem.st, problem.data
)
println("enzyme loss=$enzyme_loss seconds=$enzyme_seconds finite=$(finite_tree(enzyme_gradient))")
println(numerical_comparison(zygote_gradient, enzyme_gradient))

_, zygote_ps = optimizer_step(deepcopy(problem.ps), zygote_gradient)
_, enzyme_ps = optimizer_step(deepcopy(problem.ps), enzyme_gradient)
println((; one_step=numerical_comparison(zygote_ps, enzyme_ps)))
