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
const LEGACY_CHECKPOINT_SCHEMA =
    "reduced-hay-v2-decolle-eprop-arena-checkpoint-v1"
const PREVIOUS_CHECKPOINT_SCHEMA =
    "reduced-hay-v2-decolle-eprop-arena-checkpoint-v2"
const APICAL_V3_CHECKPOINT_SCHEMA =
    "reduced-hay-v2-apical-credit-arena-checkpoint-v3"
const LAYERED_V4_CHECKPOINT_SCHEMA =
    "reduced-hay-v2-layered-feedback-arena-checkpoint-v4"
const CHECKPOINT_SCHEMA =
    "reduced-hay-v2-exact-slot-axis-arena-checkpoint-v5"
const PARAMETER_REGISTRY_SCHEMA =
    "reduced-hay-v2-dynamic-parameter-registry-v1"

export CHECKPOINT_SCHEMA,
    load_reduced_hay_v2_checkpoint,
    reduced_hay_v2_checkpoint_sha256,
    restore_reduced_hay_v2_checkpoint!,
    save_reduced_hay_v2_checkpoint

function _copy_tree(tree)
    return NamedTuple{keys(tree)}(map(copy, values(tree)))
end

function _parameter_registry_signature(tree)
    return (;
        schema=PARAMETER_REGISTRY_SCHEMA,
        names=Tuple(keys(tree)),
        shapes=Tuple(map(size, values(tree))),
        element_types=Tuple(map(value -> string(eltype(value)), values(tree))),
    )
end

function _validate_parameter_registry(tree, signature, label::AbstractString)
    signature.schema == PARAMETER_REGISTRY_SCHEMA ||
        error("unsupported $label parameter registry schema")
    current = _parameter_registry_signature(tree)
    current == signature || error("$label parameter registry differs")
    return nothing
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

function _copy_tree_compatible!(
    destination,
    source,
    label::AbstractString,
)
    @inbounds for name in keys(destination)
        hasproperty(source, name) || continue
        target = getproperty(destination, name)
        value = getproperty(source, name)
        size(target) == size(value) ||
            error("$label shape differs for $name")
        copyto!(target, value)
    end
    return destination
end

function _model_signature(model)
    topology = Training.Model.reduced_hay_topology(model)
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
        head_readout=model.head_readout,
        sensory_layout=model.sensory_layout,
        route_revisit_policy=model.route_revisit_policy,
        communication_init=model.communication_init,
        apical_response=model.apical_response,
        cell_export=model.cell_export,
        workspace_binding=model.workspace_binding,
        workspace_layout=model.workspace_layout,
        route_dim=model.route_dim,
        head_layout=model.head_layout,
        head_state_rank=model.head_state_rank,
        branch_bias_mode=model.branch_bias_mode,
        workspace_binding_version=topology.workspace_binding,
        spatial_binding_seed=topology.spatial_binding_seed,
        temporal_binding_seed=topology.temporal_binding_seed,
        sensory_fanin=model.sensory_fanin,
        sensory_cycles=model.sensory_cycles,
        fixed_recurrent_fanout=model.fixed_recurrent_fanout,
    )
end

function _normalized_model_signature(signature)
    communication_init=hasproperty(signature, :communication_init) ?
        signature.communication_init : :random
    apical_response=hasproperty(signature, :apical_response) ?
        signature.apical_response :
        communication_init === :zero ? :centered_v2 : :uncentered_v1
    cell_export=hasproperty(signature, :cell_export) ?
        signature.cell_export : :legacy6
    workspace_binding=hasproperty(signature, :workspace_binding) ?
        signature.workspace_binding : :none
    workspace_binding_version=hasproperty(
        signature,
        :workspace_binding_version,
    ) ? signature.workspace_binding_version : workspace_binding
    spatial_binding_seed=hasproperty(
        signature,
        :spatial_binding_seed,
    ) ? signature.spatial_binding_seed : nothing
    temporal_binding_seed=hasproperty(
        signature,
        :temporal_binding_seed,
    ) ? signature.temporal_binding_seed : nothing
    workspace_layout=hasproperty(signature, :workspace_layout) ?
        signature.workspace_layout : :single_vector
    route_dim=hasproperty(signature, :route_dim) ?
        signature.route_dim : signature.node_dim
    head_layout=hasproperty(signature, :head_layout) ?
        signature.head_layout : :dense_mlp
    head_state_rank=hasproperty(signature, :head_state_rank) ?
        signature.head_state_rank : 0
    branch_bias_mode=hasproperty(signature, :branch_bias_mode) ?
        signature.branch_bias_mode : :raw
    return (;
        blocks=signature.blocks,
        cells_per_block=signature.cells_per_block,
        branches=signature.branches,
        readout_per_cell=signature.readout_per_cell,
        node_dim=signature.node_dim,
        fanout=signature.fanout,
        cycles=signature.cycles,
        workspace_k=signature.workspace_k,
        hidden=signature.hidden,
        spike_temperature=signature.spike_temperature,
        route_temperature=signature.route_temperature,
        variant=signature.variant,
        head_readout=hasproperty(signature, :head_readout) ?
            signature.head_readout : :pooled,
        sensory_layout=hasproperty(signature, :sensory_layout) ?
            signature.sensory_layout : :hashed,
        route_revisit_policy=hasproperty(
            signature,
            :route_revisit_policy,
        ) ? signature.route_revisit_policy : :allow,
        communication_init,
        apical_response,
        cell_export,
        workspace_binding,
        workspace_layout,
        route_dim,
        head_layout,
        head_state_rank,
        branch_bias_mode,
        workspace_binding_version,
        spatial_binding_seed,
        temporal_binding_seed,
        sensory_fanin=signature.sensory_fanin,
        sensory_cycles=signature.sensory_cycles,
        fixed_recurrent_fanout=signature.fixed_recurrent_fanout,
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
        parameter_registry=
            _parameter_registry_signature(trainer.parameters),
        parameters=_copy_tree(trainer.parameters),
        initial_parameters=_copy_tree(trainer.initial_parameters),
        optimizer_first_moment=_copy_tree(optimizer.first_moment),
        optimizer_second_moment=_copy_tree(optimizer.second_moment),
        recurrent_gradient_accumulator=
            _copy_tree(trainer.recurrent_gradient_accumulator),
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
            recurrent_learning_rate_multiplier=
                trainer.recurrent_learning_rate_multiplier,
            sensory_learning_rate_multiplier=
                trainer.sensory_learning_rate_multiplier,
            routing_learning_rate_multiplier=
                trainer.routing_learning_rate_multiplier,
            communication_learning_rate_multiplier=
                trainer.communication_learning_rate_multiplier,
            recurrent_accumulation_steps=
                trainer.recurrent_accumulation_steps,
            recurrent_accumulation_count=
                trainer.recurrent_accumulation_count,
            recurrent_beta1_power=trainer.recurrent_beta1_power,
            recurrent_beta2_power=trainer.recurrent_beta2_power,
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
        payload.checkpoint_schema in (
            CHECKPOINT_SCHEMA,
            LAYERED_V4_CHECKPOINT_SCHEMA,
            APICAL_V3_CHECKPOINT_SCHEMA,
            PREVIOUS_CHECKPOINT_SCHEMA,
            LEGACY_CHECKPOINT_SCHEMA,
        ) ||
        error("unsupported Reduced Hay v2 checkpoint schema")
    return payload
end

function restore_reduced_hay_v2_checkpoint!(
    trainer,
    payload,
    training_rows,
)
    _normalized_model_signature(payload.model_signature) ==
        _normalized_model_signature(_model_signature(trainer.model)) ||
        error("checkpoint model signature differs")
    payload.arena_signature == (;
        state_batch=trainer.tape.base.state_batch,
        width=trainer.tape.base.width,
    ) || error("checkpoint arena signature differs")

    if payload.checkpoint_schema == CHECKPOINT_SCHEMA
        hasproperty(payload, :parameter_registry) ||
            error("v5 checkpoint omits parameter registry")
        for (tree, label) in (
            (trainer.parameters, "parameters"),
            (trainer.initial_parameters, "initial parameters"),
            (trainer.optimizer.first_moment, "first moment"),
            (trainer.optimizer.second_moment, "second moment"),
            (
                trainer.recurrent_gradient_accumulator,
                "recurrent gradient accumulator",
            ),
        )
            _validate_parameter_registry(
                tree,
                payload.parameter_registry,
                label,
            )
        end
    end

    legacy = payload.checkpoint_schema ==
             LEGACY_CHECKPOINT_SCHEMA
    compatible = payload.checkpoint_schema != CHECKPOINT_SCHEMA

    copy_parameters! = compatible ?
        _copy_tree_compatible! : _copy_tree!
    copy_parameters!(
        trainer.parameters,
        payload.parameters,
        "parameters",
    )
    copy_parameters!(
        trainer.initial_parameters,
        payload.initial_parameters,
        "initial parameters",
    )
    copy_parameters!(
        trainer.optimizer.first_moment,
        payload.optimizer_first_moment,
        "first moment",
    )
    copy_parameters!(
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
    trainer.global_signal_scale = legacy ?
        1.0f0 : Float32(trainer_state.global_signal_scale)
    trainer.local_signal_scale = legacy ?
        1.0f0 : Float32(trainer_state.local_signal_scale)
    if hasproperty(
        trainer_state,
        :recurrent_learning_rate_multiplier,
    )
        trainer.recurrent_learning_rate_multiplier = Float32(
            trainer_state.recurrent_learning_rate_multiplier,
        )
    end
    trainer.sensory_learning_rate_multiplier = hasproperty(
        trainer_state,
        :sensory_learning_rate_multiplier,
    ) ? Float32(
        trainer_state.sensory_learning_rate_multiplier,
    ) : trainer.recurrent_learning_rate_multiplier
    trainer.routing_learning_rate_multiplier = hasproperty(
        trainer_state,
        :routing_learning_rate_multiplier,
    ) ? Float32(
        trainer_state.routing_learning_rate_multiplier,
    ) : trainer.recurrent_learning_rate_multiplier
    trainer.communication_learning_rate_multiplier = hasproperty(
        trainer_state,
        :communication_learning_rate_multiplier,
    ) ? Float32(
        trainer_state.communication_learning_rate_multiplier,
    ) : trainer.recurrent_learning_rate_multiplier
    if hasproperty(trainer_state, :recurrent_accumulation_steps)
        trainer.recurrent_accumulation_steps = Int(
            trainer_state.recurrent_accumulation_steps,
        )
        trainer.recurrent_accumulation_count = Int(
            trainer_state.recurrent_accumulation_count,
        )
        trainer.recurrent_beta1_power = Float32(
            trainer_state.recurrent_beta1_power,
        )
        trainer.recurrent_beta2_power = Float32(
            trainer_state.recurrent_beta2_power,
        )
        hasproperty(payload, :recurrent_gradient_accumulator) ||
            error("checkpoint omits recurrent gradient accumulator")
        copy_parameters!(
            trainer.recurrent_gradient_accumulator,
            payload.recurrent_gradient_accumulator,
            "recurrent gradient accumulator",
        )
    else
        trainer.recurrent_accumulation_count = 0
        trainer.recurrent_beta1_power = optimizer.beta1_power
        trainer.recurrent_beta2_power = optimizer.beta2_power
        for array in values(trainer.recurrent_gradient_accumulator)
            fill!(array, 0.0f0)
        end
    end
    trainer.recurrent_optimizer_due = false
    trainer.routing_entropy_weight =
        Float32(trainer_state.routing_entropy_weight)
    trainer.routing_entropy_floor =
        Float32(trainer_state.routing_entropy_floor)
    trainer.routing_load_weight =
        Float32(trainer_state.routing_load_weight)

    size(trainer.projection) == size(payload.projection) ||
        error("checkpoint projection shape differs")
    copyto!(trainer.projection, payload.projection)
    if legacy
        copyto!(
            trainer.parameters.local_readout,
            trainer.projection,
        )
        copyto!(
            trainer.initial_parameters.local_readout,
            trainer.projection,
        )
        fill!(trainer.parameters.local_readout_bias, 0.0f0)
        fill!(
            trainer.initial_parameters.local_readout_bias,
            0.0f0,
        )
        for moment in (
            trainer.optimizer.first_moment,
            trainer.optimizer.second_moment,
        )
            fill!(moment.local_readout, 0.0f0)
            fill!(moment.local_readout_bias, 0.0f0)
        end
    end
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
