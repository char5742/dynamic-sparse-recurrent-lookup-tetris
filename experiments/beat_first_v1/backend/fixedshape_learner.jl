module BeatFirstFixedShapeBackend

using ADTypes: AutoEnzyme
using Lux
using Optimisers
using Reactant

export FixedShapeLearner, init_backend, train_step!, host_parameters, host_checkpoint, compiled_thunk_id

const DEFAULT_MAX_CANDIDATES = 208

"""Persistent Reactant learner for one fixed candidate-batch shape.

`host_batch` is owned by the learner and must be a NamedTuple with `inputs`,
`targets`, and a `Float32` `mask` of shape `(max_candidates, batch_size)`.
The objective follows Lux's `(model, ps, st, batch) -> (loss, st, stats)`
contract. Replay packing remains native Julia and mutates `host_batch`.
"""
mutable struct FixedShapeLearner{M,O,D,H,S}
    model::M
    objective::O
    device::D
    host_batch::H
    host_signature::S
    # Lux adds a compiled cache on the first Reactant step, changing the
    # concrete TrainState type. Keep this host-side control field open.
    train_state::Any
    max_candidates::Int
    batch_size::Int
    first_thunk_id::Union{Nothing,UInt64}
    updates::Int
end

function _visit_arrays!(destination::Vector{Any}, value)
    if value isa NamedTuple || value isa Tuple
        foreach(item -> _visit_arrays!(destination, item), values(value))
    elseif value isa AbstractArray
        push!(destination, value)
    else
        throw(ArgumentError("fixed-shape batch leaves must be arrays, got $(typeof(value))"))
    end
    return destination
end

_arrays(value) = _visit_arrays!(Any[], value)
_signature(value) = Tuple((eltype(array), size(array)) for array in _arrays(value))

function _optimizer_rule_signature(value)
    if value isa NamedTuple
        return (:named_tuple, keys(value), map(_optimizer_rule_signature, values(value)))
    elseif value isa Tuple
        return (:tuple, map(_optimizer_rule_signature, value))
    elseif value isa AbstractArray
        host = Array(value)
        return (:array, eltype(host), size(host), Tuple(vec(host)))
    elseif value === nothing || value isa Number || value isa Bool ||
            value isa Symbol || value isa AbstractString
        return (:scalar, typeof(value), value)
    end
    type = typeof(value)
    isstructtype(type) && fieldcount(type) > 0 || error(
        "unsupported optimizer-rule field $(type)",
    )
    return (
        :struct,
        string(parentmodule(type)),
        string(nameof(type)),
        fieldnames(type),
        ntuple(index -> _optimizer_rule_signature(getfield(value, index)), fieldcount(type)),
    )
end

function _optimizer_state_layout(value)
    if value isa NamedTuple
        return (:named_tuple, keys(value), map(_optimizer_state_layout, values(value)))
    elseif value isa Tuple
        return (:tuple, map(_optimizer_state_layout, value))
    elseif value isa AbstractArray
        return (:array, eltype(value), size(value))
    elseif value === nothing || value isa Number || value isa Bool
        return (:scalar, typeof(value))
    end
    type = typeof(value)
    isstructtype(type) && fieldcount(type) > 0 || error(
        "unsupported optimizer-state field $(type)",
    )
    return (
        :struct,
        string(parentmodule(type)),
        string(nameof(type)),
        fieldnames(type),
        ntuple(index -> _optimizer_state_layout(getfield(value, index)), fieldcount(type)),
    )
end

function _optimizer_tree_signature(value)
    if value isa Optimisers.Leaf
        return (
            :leaf,
            _optimizer_rule_signature(value.rule),
            _optimizer_state_layout(value.state),
            value.frozen,
        )
    elseif value isa NamedTuple
        return (:named_tuple, keys(value), map(_optimizer_tree_signature, values(value)))
    elseif value isa Tuple
        return (:tuple, map(_optimizer_tree_signature, value))
    end
    error("optimizer tree contains an unsupported node $(typeof(value))")
end

function _restore_optimizer_state_value(template, saved, array_device, number_device)
    if template isa NamedTuple
        saved isa NamedTuple && keys(saved) == keys(template) || error(
            "restored optimizer NamedTuple state is incompatible",
        )
        restored = map(
            (template_value, saved_value) -> _restore_optimizer_state_value(
                template_value, saved_value, array_device, number_device,
            ),
            values(template),
            values(saved),
        )
        return NamedTuple{keys(template)}(restored)
    elseif template isa Tuple
        saved isa Tuple && length(saved) == length(template) || error(
            "restored optimizer Tuple state is incompatible",
        )
        return map(
            (template_value, saved_value) -> _restore_optimizer_state_value(
                template_value, saved_value, array_device, number_device,
            ),
            template,
            saved,
        )
    elseif template isa AbstractArray
        saved isa AbstractArray || error("restored optimizer array state is missing")
        size(saved) == size(template) || error("restored optimizer array shape changed")
        eltype(saved) == eltype(template) || error("restored optimizer array eltype changed")
        restored = array_device(saved)
        typeof(restored) === typeof(template) || error(
            "restored optimizer array did not recover its Reactant device type",
        )
        return restored
    elseif template isa Number
        saved isa Number || error("restored optimizer scalar state is missing")
        host_template = Reactant.to_number(template)
        restored = number_device(convert(typeof(host_template), saved))
        typeof(restored) === typeof(template) || error(
            "restored optimizer scalar did not recover its tracked Reactant type",
        )
        return restored
    end
    error("unsupported optimizer state template $(typeof(template))")
end

function _restore_optimizer_tree(template, saved, array_device, number_device)
    if template isa Optimisers.Leaf
        saved isa Optimisers.Leaf || error("restored optimizer leaf is missing")
        saved.frozen == template.frozen || error("restored optimizer frozen flag changed")
        return Optimisers.Leaf(
            template.rule,
            _restore_optimizer_state_value(
                template.state, saved.state, array_device, number_device,
            );
            frozen=template.frozen,
        )
    elseif template isa NamedTuple
        saved isa NamedTuple && keys(saved) == keys(template) || error(
            "restored optimizer tree keys changed",
        )
        restored = map(
            (template_value, saved_value) -> _restore_optimizer_tree(
                template_value, saved_value, array_device, number_device,
            ),
            values(template),
            values(saved),
        )
        return NamedTuple{keys(template)}(restored)
    elseif template isa Tuple
        saved isa Tuple && length(saved) == length(template) || error(
            "restored optimizer tree arity changed",
        )
        return map(
            (template_value, saved_value) -> _restore_optimizer_tree(
                template_value, saved_value, array_device, number_device,
            ),
            template,
            saved,
        )
    end
    error("optimizer tree contains an unsupported node $(typeof(template))")
end

function _validate_batch(batch, signature, max_candidates::Int, batch_size::Int)
    batch isa NamedTuple || throw(ArgumentError("batch must be a NamedTuple"))
    keys(batch) == (:inputs, :targets, :mask) || throw(
        ArgumentError("batch fields must be exactly (:inputs, :targets, :mask)")
    )
    mask = batch.mask
    mask isa Matrix{Float32} || throw(ArgumentError("mask must be Matrix{Float32}"))
    size(mask) == (max_candidates, batch_size) || throw(
        DimensionMismatch(
            "mask must have shape ($max_candidates, $batch_size), got $(size(mask))"
        )
    )
    _signature(batch) == signature || throw(
        DimensionMismatch("packed batch changed array count, element type, or shape")
    )
    all(isfinite, mask) || throw(ArgumentError("mask contains a non-finite value"))
    all(value -> value == 0.0f0 || value == 1.0f0, mask) || throw(
        ArgumentError("mask must contain only 0f0 and 1f0")
    )
    all(>(0.0f0), vec(sum(mask; dims=1))) || throw(
        ArgumentError("every state must contain at least one valid candidate")
    )
    return batch
end

"""Create the persistent Reactant TrainState and reusable host batch.

The first `train_step!` performs compilation. Supplying later data with the
same leaf shapes changes values without changing the compiled train step.
"""
function init_backend(
    model,
    parameters,
    states,
    optimiser,
    objective,
    host_template;
    max_candidates::Int=DEFAULT_MAX_CANDIDATES,
    backend::AbstractString="cpu",
    restore=nothing,
)
    max_candidates > 0 || throw(ArgumentError("max_candidates must be positive"))
    Reactant.set_default_backend(backend)
    host_batch = deepcopy(host_template)
    signature = _signature(host_batch)
    host_template.mask isa AbstractMatrix || throw(ArgumentError("mask must be a matrix"))
    batch_size = size(host_template.mask, 2)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    _validate_batch(host_batch, signature, max_candidates, batch_size)

    device = Lux.reactant_device(; force=true)
    device_parameters, device_states = device((parameters, states))
    fresh_train_state = Lux.Training.TrainState(
        model, device_parameters, device_states, optimiser
    )
    backend_updates = 0
    train_state = if restore === nothing
        fresh_train_state
    else
        required = (:parameters, :states, :optimizer_state, :step, :backend_updates)
        all(name -> hasproperty(restore, name), required) || error(
            "Reactant restore is missing one of $(required)",
        )
        restored_step = Int(restore.step)
        backend_updates = Int(restore.backend_updates)
        restored_step >= 0 || error("restored TrainState step is negative")
        backend_updates >= 0 || error("restored backend update count is negative")
        restored_step == backend_updates || error(
            "restored TrainState step=$restored_step != backend updates=$backend_updates",
        )
        fresh_host_optimizer_state = Lux.cpu_device()(
            fresh_train_state.optimizer_state,
        )
        _optimizer_tree_signature(restore.optimizer_state) ==
            _optimizer_tree_signature(fresh_host_optimizer_state) || error(
            "restored optimizer state layout or pinned rule is incompatible",
        )
        restored_parameters, restored_states = device((
            restore.parameters,
            restore.states,
        ))
        number_device = Lux.reactant_device(; force=true, track_numbers=AbstractFloat)
        restored_optimizer_state = _restore_optimizer_tree(
            fresh_train_state.optimizer_state,
            restore.optimizer_state,
            device,
            number_device,
        )
        typeof(restored_parameters) === typeof(fresh_train_state.parameters) || error(
            "restored parameter tree is incompatible with the fresh model",
        )
        typeof(restored_states) === typeof(fresh_train_state.states) || error(
            "restored state tree is incompatible with the fresh model",
        )
        typeof(restored_optimizer_state) === typeof(fresh_train_state.optimizer_state) ||
            error("restored optimizer state did not recover the fresh Reactant type")
        # Lux 1.31.4 pins this internal layout. Preserve the fresh cache,
        # allocator, objective slot, model, and Reactant-compatible optimizer;
        # replace only the four durable training fields.
        Lux.Training.TrainState(
            fresh_train_state.cache,
            fresh_train_state.objective_function,
            fresh_train_state.allocator_cache,
            fresh_train_state.model,
            restored_parameters,
            restored_states,
            fresh_train_state.optimizer,
            restored_optimizer_state,
            restored_step,
        )
    end
    return FixedShapeLearner(
        model,
        objective,
        device,
        host_batch,
        signature,
        train_state,
        max_candidates,
        batch_size,
        nothing,
        backend_updates,
    )
end

"""Return the identity of Lux's cached compiled train step, when available."""
function compiled_thunk_id(learner::FixedShapeLearner)
    cache = learner.train_state.cache
    cache === nothing && return nothing
    hasproperty(cache, :extras) || return nothing
    extras = cache.extras
    hasproperty(extras, :compiled_grad_and_step_function) || return nothing
    return UInt64(objectid(extras.compiled_grad_and_step_function))
end

function _synchronize_parameters(parameters)
    foreach(Reactant.synchronize, _arrays(parameters))
    return nothing
end

"""Pack one Replay batch and complete one synchronized optimizer update.

The reported wall time includes native packing, host-to-Reactant transfer,
forward/loss/backward/optimizer update, and synchronization of both the loss
and parameters. Checkpoint conversion is deliberately separate.
"""
function train_step!(learner::FixedShapeLearner, pack!::F) where {F}
    wall_start = time_ns()

    pack_start = time_ns()
    pack!(learner.host_batch)
    _validate_batch(
        learner.host_batch,
        learner.host_signature,
        learner.max_candidates,
        learner.batch_size,
    )
    pack_seconds = (time_ns() - pack_start) / 1.0e9

    transfer_start = time_ns()
    device_batch = learner.device(learner.host_batch)
    transfer_seconds = (time_ns() - transfer_start) / 1.0e9

    update_start = time_ns()
    _, loss, statistics, train_state = Lux.Training.single_train_step!(
        AutoEnzyme(),
        learner.objective,
        device_batch,
        learner.train_state;
        return_gradients=Val(false),
        sync=true,
    )
    learner.train_state = train_state
    Reactant.synchronize(loss)
    _synchronize_parameters(learner.train_state.parameters)
    host_loss = Float64(Reactant.to_number(loss))
    isfinite(host_loss) || error("non-finite loss at update $(learner.updates + 1)")
    # Objective statistics are outside the differentiated result.  Copying
    # their small scalar/vector leaves lets RL update PER priorities without
    # repeating the current-state forward and loss on the host.
    host_statistics = Lux.cpu_device()(statistics)
    update_seconds = (time_ns() - update_start) / 1.0e9

    learner.updates += 1
    thunk_id = compiled_thunk_id(learner)
    if learner.first_thunk_id === nothing
        learner.first_thunk_id = thunk_id
    end
    recompiled = learner.first_thunk_id !== nothing && thunk_id !== learner.first_thunk_id

    base = (;
        loss=host_loss,
        wall_seconds=(time_ns() - wall_start) / 1.0e9,
        pack_seconds,
        transfer_seconds,
        update_seconds,
        step=learner.updates,
        compiled_thunk_id=thunk_id,
        recompiled,
    )
    return merge(base, host_statistics)
end

function _copy_batch!(destination, source)
    destination_signature = _signature(destination)
    source_signature = _signature(source)
    destination_signature == source_signature || throw(
        DimensionMismatch("source batch does not match the fixed host batch")
    )
    foreach(copyto!, _arrays(destination), _arrays(source))
    return destination
end

"""Convenience path for callers that already own a complete fixed batch."""
train_step!(learner::FixedShapeLearner, batch::NamedTuple) =
    train_step!(learner, host -> _copy_batch!(host, batch))

"""Copy a synchronized training snapshot back to native CPU arrays.

Call this outside throughput measurements; it is intentionally not part of
the compiled update or `train_step!`.
"""
function host_checkpoint(learner::FixedShapeLearner)
    _synchronize_parameters(learner.train_state.parameters)
    cpu = Lux.cpu_device()
    parameters, states, optimizer_state = cpu((
        learner.train_state.parameters,
        learner.train_state.states,
        learner.train_state.optimizer_state,
    ))
    return (;
        parameters,
        states,
        optimizer=learner.train_state.optimizer,
        optimizer_state,
        step=learner.train_state.step,
        backend_updates=learner.updates,
    )
end

"""Copy only online parameters/state for host-side Double-DQN targets.

Unlike `host_checkpoint`, this deliberately avoids transferring the AdamW
optimizer tree on every learner update. Durable optimizer state is copied only
at the 100--250 update checkpoint boundary.
"""
function host_parameters(learner::FixedShapeLearner)
    _synchronize_parameters(learner.train_state.parameters)
    cpu = Lux.cpu_device()
    parameters, states = cpu((
        learner.train_state.parameters,
        learner.train_state.states,
    ))
    return (;
        ps=parameters,
        st=states,
        step=Int(learner.train_state.step),
        backend_updates=learner.updates,
    )
end

end # module
