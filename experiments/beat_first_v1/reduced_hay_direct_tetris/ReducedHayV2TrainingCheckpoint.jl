module ReducedHayV2TrainingCheckpoint

using Dates
using JLD2
using SHA

if !isdefined(Main, :BeatFirstTrainingCore)
    Base.include(
        Main,
        joinpath(@__DIR__, "..", "training", "core.jl"),
    )
end
if !isdefined(Main, :ReducedHayV2ArenaTraining)
    Base.include(
        Main,
        joinpath(@__DIR__, "ReducedHayV2ArenaTraining.jl"),
    )
end

const Core = Main.BeatFirstTrainingCore
const Training = Main.ReducedHayV2ArenaTraining
const CHECKPOINT_SCHEMA =
    "reduced-hay-v2-decolle-eprop-arena-checkpoint-v1"

export CHECKPOINT_SCHEMA,
    load_reduced_hay_v2_checkpoint,
    reduced_hay_v2_checkpoint_sha256,
    restore_reduced_hay_v2_checkpoint!,
    save_reduced_hay_v2_checkpoint

function _copy_tree(tree)
    return NamedTuple{keys(tree)}(map(copy, values(tree)))
end

function _copy_tree!(destination, source, label::AbstractString)
    keys(destination) == keys(source) ||
        error("$label parameter registry differs")
    @inbounds for name in keys(destination)
        target = getproperty(destination, name)
        value = getproperty(source, name)
        size(target) == size(value) ||
            error("$label shape differs for $name")
        copyto!(target, value)
    end
    return destination
end

function _model_signature(model)
    return (;
        blocks=model.blocks,
        cells_per_block=model.cells_per_block,
        branches=model.branches,
        readout_per_cell=model.readout_per_cell,
        node_dim=model.node_dim,
        fanout=model.fanout,
        cycles=model.cycles,
        workspace_k=model.workspace_k,
        hidden=model.hidden,
        spike_temperature=model.spike_temperature,
        route_temperature=model.route_temperature,
        variant=model.variant,
        sensory_fanin=model.sensory_fanin,
        sensory_cycles=model.sensory_cycles,
        fixed_recurrent_fanout=model.fixed_recurrent_fanout,
    )
end

function reduced_hay_v2_checkpoint_sha256(path::AbstractString)
    isfile(path) || error("checkpoint is absent: $path")
    return bytes2hex(SHA.sha256(read(path)))
end

function save_reduced_hay_v2_checkpoint(
    path::AbstractString,
    trainer,
    sampler,
    run_config;
    update::Integer=trainer.optimizer.step,
)
    optimizer = trainer.optimizer
    payload = (;
        checkpoint_schema=CHECKPOINT_SCHEMA,
        saved_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
        update=Int(update),
        model_signature=_model_signature(trainer.model),
        arena_signature=(;
            state_batch=trainer.tape.base.state_batch,
            width=trainer.tape.base.width,
        ),
        run_config,
        parameters=_copy_tree(trainer.parameters),
        initial_parameters=_copy_tree(trainer.initial_parameters),
        optimizer_first_moment=_copy_tree(optimizer.first_moment),
        optimizer_second_moment=_copy_tree(optimizer.second_moment),
        optimizer_state=(;
            learning_rate=optimizer.learning_rate,
            beta1=optimizer.beta1,
            beta2=optimizer.beta2,
            beta1_power=optimizer.beta1_power,
            beta2_power=optimizer.beta2_power,
            epsilon=optimizer.epsilon,
            weight_decay=optimizer.weight_decay,
            step=optimizer.step,
        ),
        trainer_state=(;
            utility_decay=trainer.utility_decay,
            utility_connection_cost=
                trainer.utility_connection_cost,
            structural_interval=trainer.structural_interval,
            branch_interval=trainer.branch_interval,
            global_signal_scale=
                trainer.global_signal_scale,
            local_signal_scale=trainer.local_signal_scale,
            routing_entropy_weight=
                trainer.routing_entropy_weight,
            routing_entropy_floor=
                trainer.routing_entropy_floor,
            routing_load_weight=
                trainer.routing_load_weight,
        ),
        projection=copy(trainer.projection),
        branch_for_edge=copy(trainer.branch_for_edge),
        gate_mask=copy(trainer.gate_mask),
        synapse_utility=copy(trainer.synapse_utility),
        branch_utility=copy(trainer.branch_utility),
        sampler_snapshot=Core.sampler_snapshot(sampler),
    )
    destination = Core.atomic_jldsave(
        path;
        payload,
    )
    return (;
        path=destination,
        sha256=reduced_hay_v2_checkpoint_sha256(destination),
        update=Int(update),
    )
end

function load_reduced_hay_v2_checkpoint(path::AbstractString)
    source = abspath(path)
    isfile(source) || error("checkpoint is absent: $source")
    data = JLD2.load(source)
    haskey(data, "payload") ||
        error("checkpoint has no payload")
    payload = data["payload"]
    hasproperty(payload, :checkpoint_schema) &&
        payload.checkpoint_schema == CHECKPOINT_SCHEMA ||
        error("unsupported Reduced Hay v2 checkpoint schema")
    return payload
end

function restore_reduced_hay_v2_checkpoint!(
    trainer,
    payload,
    training_rows,
)
    payload.model_signature == _model_signature(trainer.model) ||
        error("checkpoint model signature differs")
    payload.arena_signature == (;
        state_batch=trainer.tape.base.state_batch,
        width=trainer.tape.base.width,
    ) || error("checkpoint arena signature differs")

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
        "first moment",
    )
    _copy_tree!(
        trainer.optimizer.second_moment,
        payload.optimizer_second_moment,
        "second moment",
    )
    optimizer_state = payload.optimizer_state
    optimizer = trainer.optimizer
    optimizer.learning_rate =
        Float32(optimizer_state.learning_rate)
    optimizer.beta1 = Float32(optimizer_state.beta1)
    optimizer.beta2 = Float32(optimizer_state.beta2)
    optimizer.beta1_power =
        Float32(optimizer_state.beta1_power)
    optimizer.beta2_power =
        Float32(optimizer_state.beta2_power)
    optimizer.epsilon = Float32(optimizer_state.epsilon)
    optimizer.weight_decay =
        Float32(optimizer_state.weight_decay)
    optimizer.step = Int(optimizer_state.step)
    optimizer.step == Int(payload.update) ||
        error("checkpoint update and optimizer step differ")
    trainer_state = payload.trainer_state
    trainer.utility_decay =
        Float32(trainer_state.utility_decay)
    trainer.utility_connection_cost =
        Float32(trainer_state.utility_connection_cost)
    trainer.structural_interval =
        Int(trainer_state.structural_interval)
    trainer.branch_interval =
        Int(trainer_state.branch_interval)
    trainer.global_signal_scale =
        Float32(trainer_state.global_signal_scale)
    trainer.local_signal_scale =
        Float32(trainer_state.local_signal_scale)
    trainer.routing_entropy_weight =
        Float32(trainer_state.routing_entropy_weight)
    trainer.routing_entropy_floor =
        Float32(trainer_state.routing_entropy_floor)
    trainer.routing_load_weight =
        Float32(trainer_state.routing_load_weight)

    size(trainer.projection) == size(payload.projection) ||
        error("checkpoint projection shape differs")
    copyto!(trainer.projection, payload.projection)
    size(trainer.branch_for_edge) ==
        size(payload.branch_for_edge) ||
        error("checkpoint branch map shape differs")
    copyto!(
        trainer.branch_for_edge,
        payload.branch_for_edge,
    )
    all(
        branch ->
            1 <= branch <= trainer.model.branches,
        trainer.branch_for_edge,
    ) || error("checkpoint branch map is invalid")
    size(trainer.gate_mask) == size(payload.gate_mask) ||
        error("checkpoint gate mask shape differs")
    copyto!(trainer.gate_mask, payload.gate_mask)
    expected_fanout =
        trainer.model.fixed_recurrent_fanout
    all(
        source ->
            count(@view(trainer.gate_mask[source, :])) ==
            expected_fanout,
        axes(trainer.gate_mask, 1),
    ) || error("checkpoint gate mask violates fixed fanout")
    size(trainer.synapse_utility) ==
        size(payload.synapse_utility) ||
        error("checkpoint synapse utility shape differs")
    size(trainer.branch_utility) ==
        size(payload.branch_utility) ||
        error("checkpoint branch utility shape differs")
    copyto!(
        trainer.synapse_utility,
        payload.synapse_utility,
    )
    copyto!(
        trainer.branch_utility,
        payload.branch_utility,
    )
    Training.refresh_dendritic_cache!(
        trainer.cache,
        trainer.parameters,
        trainer.gate_mask,
    )
    return Core.restore_sampler(
        training_rows,
        payload.sampler_snapshot,
    )
end

end # module
