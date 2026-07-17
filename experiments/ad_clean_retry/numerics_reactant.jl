include(joinpath(@__DIR__, "common.jl"))
using ADTypes: AutoEnzyme
using Reactant

const BATCH_SIZE = parse(Int, get(ENV, "AD_CLEAN_BATCH", "16"))
const BLAS_THREADS = parse(Int, get(ENV, "AD_CLEAN_BLAS_THREADS", "20"))
const OUTPUT = get(ENV, "AD_CLEAN_OUTPUT", joinpath(@__DIR__, "artifacts", "numerics_reactant_b$(BATCH_SIZE).json"))

BLAS.set_num_threads(BLAS_THREADS)
Reactant.set_default_backend("cpu")

function main()
    problem = make_problem(BATCH_SIZE)
    zygote_gradient_value, zygote_loss = zygote_gradient(problem, problem.parameters)
    zygote_parameters = deepcopy(problem.parameters)
    zygote_optimizer = Optimisers.setup(OPTIMIZER, zygote_parameters)
    Optimisers.update!(zygote_optimizer, zygote_parameters, zygote_gradient_value)

    device = Lux.reactant_device(; force=true)
    parameters, state, batch = device((problem.parameters, problem.state, problem.batch))
    train_state = Lux.Training.TrainState(problem.model, parameters, state, OPTIMIZER)
    reactant_gradient, reactant_loss, _, train_state = Lux.Training.single_train_step!(
        AutoEnzyme(),
        problem.objective,
        batch,
        train_state;
        return_gradients=Val(true),
        sync=true,
    )
    Reactant.synchronize(reactant_loss)
    foreach(Reactant.synchronize, tree_arrays(reactant_gradient))
    foreach(Reactant.synchronize, tree_arrays(train_state.parameters))

    document = (;
        status="completed",
        generated_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        batch_size=BATCH_SIZE,
        versions=package_versions(; reactant=string(Base.pkgversion(Reactant))),
        julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
        workload=workload_description(problem),
        zygote_loss=Float64(zygote_loss),
        reactant_loss=Float64(Reactant.to_number(reactant_loss)),
        loss_absolute_error=abs(Float64(Reactant.to_number(reactant_loss)) - Float64(zygote_loss)),
        gradient_vs_zygote=numeric_comparison(zygote_gradient_value, reactant_gradient),
        one_update_parameters_vs_zygote=numeric_comparison(zygote_parameters, train_state.parameters),
    )
    write_json(OUTPUT, document)
    println(JSON3.pretty(document))
end

main()
