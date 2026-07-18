using JSON3
using Lux
using Optimisers
using Random

include(joinpath(@__DIR__, "core.jl"))
include(joinpath(@__DIR__, "..", "models", "models.jl"))
using .BeatFirstTrainingCore
using .BeatFirstModels

const BACKEND_KIND = lowercase(get(ENV, "BEAT_SMOKE_BACKEND", "native"))
const BACKEND_API = if BACKEND_KIND == "reactant"
    include(joinpath(@__DIR__, "..", "backend", "fixedshape_learner.jl"))
    BeatFirstFixedShapeBackend
else
    include(joinpath(@__DIR__, "native_backend.jl"))
    BeatFirstNativeBackend
end

allfinite(x::AbstractArray) = all(isfinite, x)
allfinite(x::NamedTuple) = all(allfinite, values(x))
allfinite(x::Tuple) = all(allfinite, x)
allfinite(x) = true

function main()
    dataset = load_teacher_dataset(get(
        ENV,
        "BEAT_TEACHER_DATASET",
        raw"D:\tetris-paper-plus\datasets\learning\teacher_plus_dagger_c13_round1.jld2",
    ))
    state_batch = parse(Int, get(ENV, "BEAT_SMOKE_STATE_BATCH", "1"))
    batch = allocate_host_batch(state_batch)
    pack_batch!(batch, dataset, collect(1:state_batch))
    kind = Symbol(get(ENV, "BEAT_SMOKE_MODEL", "preact_eca"))
    setup = setup_model(kind, Xoshiro(0x42656174536d6f6b65); n_quantiles=16)
    learner = BACKEND_API.init_backend(
        setup.model,
        setup.ps,
        setup.st,
        Optimisers.AdamW(3.0f-4, (0.9, 0.999), 1.0f-4),
        supervised_objective,
        batch;
        max_candidates=MAX_CANDIDATES,
        backend="cpu",
    )
    started = time()
    step = BACKEND_API.train_step!(learner, batch)
    checkpoint = BACKEND_API.host_checkpoint(learner)
    ps = hasproperty(checkpoint, :ps) ? checkpoint.ps : checkpoint.parameters
    st = hasproperty(checkpoint, :st) ? checkpoint.st : checkpoint.states
    allfinite(ps) || error("smoke checkpoint contains non-finite parameters")
    output, _ = setup.model(batch.inputs, ps, Lux.testmode(st))
    allfinite(output) || error("smoke output contains a non-finite value")
    result = (;
        backend=BACKEND_KIND,
        model=String(kind),
        parameter_count=setup.meta.parameters,
        state_batch,
        loss=Float64(step.loss),
        wall_seconds=time() - started,
        backend_step=step,
        q_shape=size(output.q),
        death_shape=size(output.death_logit),
        quantile_shape=size(output.quantiles),
    )
    JSON3.pretty(stdout, result)
    println()
    return result
end

main()
