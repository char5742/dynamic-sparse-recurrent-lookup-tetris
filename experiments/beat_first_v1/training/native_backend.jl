module BeatFirstNativeBackend

using Lux
using Zygote

export NativeLearner, init_backend, train_step!, host_checkpoint, predict

mutable struct NativeLearner
    train_state
    objective
    step::Int
end

function init_backend(
    model,
    ps,
    st,
    optimiser,
    objective,
    _host_template=nothing;
    max_candidates::Int=208,
    backend::AbstractString="cpu",
)
    1 <= max_candidates <= 208 || error(
        "native beat-first learner candidate width must be in 1:208",
    )
    lowercase(backend) in ("cpu", "native", "zygote") || error(
        "unsupported native backend device: $backend",
    )
    return NativeLearner(
        Lux.Training.TrainState(model, ps, st, optimiser), objective, 0,
    )
end

function train_step!(learner::NativeLearner, batch; synchronize::Bool=true)
    started = time()
    _, loss, statistics, next_train_state = Lux.Training.single_train_step!(
        Lux.Training.AutoZygote(), learner.objective, batch, learner.train_state,
    )
    learner.train_state = next_train_state
    learner.step += 1
    wall = time() - started
    return merge(
        (; loss, wall_seconds=wall, pack_seconds=0.0, transfer_seconds=0.0,
           update_seconds=wall, recompiled=false, step=learner.step),
        statistics,
    )
end

function host_checkpoint(learner::NativeLearner)
    return (;
        ps=learner.train_state.parameters,
        st=learner.train_state.states,
        optimizer_state=learner.train_state.optimizer_state,
        step=learner.step,
    )
end

function predict(learner::NativeLearner, batch)
    output, _ = learner.train_state.model(
        batch.inputs, learner.train_state.parameters, learner.train_state.states,
    )
    return output
end

end # module
