module PaperCheckpoint

using Dates
using JLD2
using Serialization
using SHA

if !isdefined(Main, :BeatFirstTrainingCore)
    Base.include(
        Main,
        joinpath(@__DIR__, "..", "training", "core.jl"),
    )
end
if !isdefined(Main, :PaperArenaTraining)
    Base.include(
        Main,
        joinpath(@__DIR__, "PaperArenaTraining.jl"),
    )
end

const Core = Main.BeatFirstTrainingCore
const Training = Main.PaperArenaTraining
const CHECKPOINT_SCHEMA =
    "paper-mechanism-multicompartment-workspace-snn-checkpoint-v1"

const TRAINER_SCALAR_FIELDS = (
    :utility_decay,
    :utility_connection_cost,
    :structural_interval,
    :location_interval,
    :global_signal_scale,
    :local_signal_scale,
    :routing_entropy_weight,
    :routing_entropy_floor,
    :routing_load_weight,
)

const TRAINER_ARRAY_FIELDS = (
    :projection,
    :fixed_projection,
    :gate_mask,
    :compartment_for_edge,
    :receptor_for_edge,
    :synapse_type_for_edge,
    :synapse_utility,
    :location_utility,
    :compartment_utility,
)

export CHECKPOINT_SCHEMA,
    load_paper_checkpoint,
    paper_checkpoint_sha256,
    restore_paper_checkpoint!,
    save_paper_checkpoint

function _copy_tree(tree::NamedTuple)
    return NamedTuple{keys(tree)}(map(_copy_tree, values(tree)))
end

_copy_tree(tree::Tuple) = map(_copy_tree, tree)
_copy_tree(tree::AbstractArray) = copy(tree)
_copy_tree(tree::AbstractDict) =
    Dict(key => _copy_tree(value) for (key, value) in tree)
_copy_tree(tree) = deepcopy(tree)

function _copy_tree!(
    destination::NamedTuple,
    source::NamedTuple,
    label::AbstractString,
)
    keys(destination) == keys(source) ||
        error("$label parameter registry differs")
    @inbounds for name in keys(destination)
        _copy_tree!(
            getproperty(destination, name),
            getproperty(source, name),
            "$label.$name",
        )
    end
    return destination
end

function _copy_tree!(
    destination::Tuple,
    source::Tuple,
    label::AbstractString,
)
    length(destination) == length(source) ||
        error("$label tuple length differs")
    @inbounds for index in eachindex(destination, source)
        _copy_tree!(
            destination[index],
            source[index],
            "$label[$index]",
        )
    end
    return destination
end

function _copy_tree!(
    destination::AbstractArray,
    source::AbstractArray,
    label::AbstractString,
)
    size(destination) == size(source) ||
        error("$label shape differs")
    eltype(destination) == eltype(source) ||
        error("$label element type differs")
    copyto!(destination, source)
    return destination
end

function _copy_tree!(
    destination::AbstractDict,
    source::AbstractDict,
    label::AbstractString,
)
    keys(destination) == keys(source) ||
        error("$label dictionary keys differ")
    for key in keys(destination)
        _copy_tree!(
            destination[key],
            source[key],
            "$label[$key]",
        )
    end
    return destination
end

function _copy_tree!(destination, source, label::AbstractString)
    destination == source ||
        error("$label immutable value differs")
    return destination
end

function _serialized_sha256(value)
    stream = IOBuffer()
    Serialization.serialize(stream, value)
    return bytes2hex(SHA.sha256(take!(stream)))
end

function _model_signature(model)
    scalar_fields = Dict{Symbol,Any}()
    for name in propertynames(model)
        value = getproperty(model, name)
        if value isa Union{
            Nothing,
            Bool,
            Integer,
            AbstractFloat,
            Symbol,
            AbstractString,
        }
            scalar_fields[name] = value
        end
    end
    return (;
        model_type=string(typeof(model)),
        serialized_sha256=_serialized_sha256(model),
        scalar_fields,
    )
end

function _arena_signature(trainer)
    arena = Training.paper_training_arena(trainer)
    return (;
        arena_type=string(typeof(arena)),
        state_batch=Int(getproperty(arena, :state_batch)),
        width=Int(getproperty(arena, :width)),
    )
end

function _optimizer_state(optimizer)
    state = Dict{Symbol,Any}()
    for name in (
        :learning_rate,
        :beta1,
        :beta2,
        :beta1_power,
        :beta2_power,
        :epsilon,
        :weight_decay,
        :step,
    )
        hasproperty(optimizer, name) &&
            (state[name] = getproperty(optimizer, name))
    end
    return state
end

function _trainer_scalar_state(trainer)
    state = Dict{Symbol,Any}()
    for name in TRAINER_SCALAR_FIELDS
        hasproperty(trainer, name) &&
            (state[name] = getproperty(trainer, name))
    end
    return state
end

function _trainer_array_state(trainer)
    state = Dict{Symbol,Any}()
    for name in TRAINER_ARRAY_FIELDS
        hasproperty(trainer, name) &&
            (state[name] = _copy_tree(getproperty(trainer, name)))
    end
    return state
end

function paper_checkpoint_sha256(path::AbstractString)
    isfile(path) || error("checkpoint is absent: $path")
    return bytes2hex(SHA.sha256(read(path)))
end

function save_paper_checkpoint(
    path::AbstractString,
    trainer,
    sampler,
    run_config;
    update::Integer=trainer.optimizer.step,
)
    optimizer = trainer.optimizer
    Int(update) == Int(optimizer.step) ||
        error("checkpoint update and optimizer step differ")
    payload = (;
        checkpoint_schema=CHECKPOINT_SCHEMA,
        saved_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
        update=Int(update),
        model_signature=_model_signature(trainer.model),
        arena_signature=_arena_signature(trainer),
        run_config,
        parameters=_copy_tree(trainer.parameters),
        initial_parameters=_copy_tree(
            trainer.initial_parameters,
        ),
        optimizer_first_moment=_copy_tree(
            optimizer.first_moment,
        ),
        optimizer_second_moment=_copy_tree(
            optimizer.second_moment,
        ),
        optimizer_state=_optimizer_state(optimizer),
        trainer_scalar_state=_trainer_scalar_state(trainer),
        trainer_array_state=_trainer_array_state(trainer),
        sampler_snapshot=Core.sampler_snapshot(sampler),
    )
    destination = Core.atomic_jldsave(path; payload)
    return (;
        path=destination,
        sha256=paper_checkpoint_sha256(destination),
        update=Int(update),
        schema=CHECKPOINT_SCHEMA,
    )
end

function load_paper_checkpoint(path::AbstractString)
    source = abspath(path)
    isfile(source) || error("checkpoint is absent: $source")
    data = JLD2.load(source)
    haskey(data, "payload") ||
        error("checkpoint has no payload")
    payload = data["payload"]
    hasproperty(payload, :checkpoint_schema) &&
        payload.checkpoint_schema == CHECKPOINT_SCHEMA ||
        error("unsupported paper-model checkpoint schema")
    return payload
end

function _restore_optimizer_state!(optimizer, state)
    for (raw_name, value) in state
        name = Symbol(raw_name)
        hasproperty(optimizer, name) ||
            error("optimizer no longer has checkpoint field $name")
        destination = getproperty(optimizer, name)
        converted = convert(typeof(destination), value)
        setproperty!(optimizer, name, converted)
    end
    return optimizer
end

function _restore_trainer_scalar_state!(trainer, state)
    for (raw_name, value) in state
        name = Symbol(raw_name)
        hasproperty(trainer, name) ||
            error("trainer no longer has checkpoint field $name")
        destination = getproperty(trainer, name)
        converted = convert(typeof(destination), value)
        setproperty!(trainer, name, converted)
    end
    return trainer
end

function _restore_trainer_array_state!(trainer, state)
    for (raw_name, value) in state
        name = Symbol(raw_name)
        hasproperty(trainer, name) ||
            error("trainer no longer has checkpoint array $name")
        _copy_tree!(
            getproperty(trainer, name),
            value,
            "trainer.$name",
        )
    end
    return trainer
end

function _refresh_cache_if_supported!(trainer)
    isdefined(Training, :refresh_paper_cache!) ||
        return trainer
    refresh! = getfield(Training, :refresh_paper_cache!)
    if applicable(refresh!, trainer)
        refresh!(trainer)
    elseif hasproperty(trainer, :cache) &&
           hasproperty(trainer, :gate_mask) &&
           applicable(
               refresh!,
               trainer.cache,
               trainer.parameters,
               trainer.gate_mask,
           )
        refresh!(
            trainer.cache,
            trainer.parameters,
            trainer.gate_mask,
        )
    else
        error(
            "refresh_paper_cache! exists but has no supported " *
            "checkpoint restore method",
        )
    end
    return trainer
end

function restore_paper_checkpoint!(
    trainer,
    payload,
    training_rows,
)
    payload.model_signature == _model_signature(trainer.model) ||
        error("checkpoint model signature differs")
    payload.arena_signature == _arena_signature(trainer) ||
        error("checkpoint arena signature differs")

    _copy_tree!(
        trainer.parameters,
        payload.parameters,
        "parameters",
    )
    _copy_tree!(
        trainer.initial_parameters,
        payload.initial_parameters,
        "initial parameters",
    )
    _copy_tree!(
        trainer.optimizer.first_moment,
        payload.optimizer_first_moment,
        "optimizer first moment",
    )
    _copy_tree!(
        trainer.optimizer.second_moment,
        payload.optimizer_second_moment,
        "optimizer second moment",
    )
    _restore_optimizer_state!(
        trainer.optimizer,
        payload.optimizer_state,
    )
    trainer.optimizer.step == Int(payload.update) ||
        error("checkpoint update and optimizer step differ")
    _restore_trainer_scalar_state!(
        trainer,
        payload.trainer_scalar_state,
    )
    _restore_trainer_array_state!(
        trainer,
        payload.trainer_array_state,
    )
    _refresh_cache_if_supported!(trainer)
    return Core.restore_sampler(
        training_rows,
        payload.sampler_snapshot,
    )
end

end # module
