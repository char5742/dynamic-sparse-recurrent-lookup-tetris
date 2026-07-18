module BeatFirstNativeBackend

using Lux
using Zygote

export NativeLearner, init_backend, train_step!, host_parameters, host_checkpoint, predict

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
    restore=nothing,
)
    1 <= max_candidates <= 208 || error(
        "native beat-first learner candidate width must be in 1:208",
    )
    lowercase(backend) in ("cpu", "native", "zygote") || error(
        "unsupported native backend device: $backend",
    )
    fresh = Lux.Training.TrainState(model, ps, st, optimiser)
    train_state = if restore === nothing
        fresh
    else
        required = (:parameters, :states, :optimizer_state, :step, :backend_updates)
        all(name -> hasproperty(restore, name), required) || error(
            "native restore is missing one of $(required)",
        )
        step = Int(restore.step)
        step == Int(restore.backend_updates) || error(
            "native restored TrainState and backend update counters differ",
        )
        # Lux 1.31.4 layout, shared with the pinned Reactant restore path.
        Lux.Training.TrainState(
            fresh.cache,
            fresh.objective_function,
            fresh.allocator_cache,
            fresh.model,
            restore.parameters,
            restore.states,
            fresh.optimizer,
            restore.optimizer_state,
            step,
        )
    end
    restored_step = restore === nothing ? 0 : Int(restore.step)
    return NativeLearner(train_state, objective, restored_step)
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
        backend_updates=learner.step,
    )
end

host_parameters(learner::NativeLearner) = (;
    ps=learner.train_state.parameters,
    st=learner.train_state.states,
    step=learner.step,
    backend_updates=learner.step,
)

function predict(learner::NativeLearner, batch)
    output, _ = learner.train_state.model(
        batch.inputs, learner.train_state.parameters, learner.train_state.states,
    )
    return output
end

end # module
