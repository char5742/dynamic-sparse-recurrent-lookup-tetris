module DynamicSparseRecurrentLookupTeacherTraining

using Dates
using JSON3
using Random
using Serialization
using SHA
using Statistics
using Zygote

if !isdefined(Main, :BeatFirstThreeLayerTeacherTraining)
    Base.include(Main, joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "teacher_training.jl"))
end
if !isdefined(Main, :DynamicSparseRecurrentLookup)
    Base.include(Main, joinpath(@__DIR__, "DynamicSparseRecurrentLookup.jl"))
end

const ParentTraining = Main.BeatFirstThreeLayerTeacherTraining
const TrainingCore = ParentTraining.BeatFirstTrainingCore
const SparseFeatures = Main.SparseDynamic3Layer.BeatFirstSparseFeatures
const Model = Main.DynamicSparseRecurrentLookup
const InputAdapter = Main.ResidualLookupSlide

const EXPERIMENT_ID = :dynamic_sparse_recurrent_lookup_v1
const LEARNER_WIDTH = 80
const MODEL_SEED = UInt64(0x4453524c5f4d4f44)
const HALT_SEED = UInt64(0x4453524c5f48414c)
const SPLIT_SEED = UInt64(0x334c5f53504c4954)
const SAMPLER_SEED = UInt64(0x4453524c5f53414d)
const HELD_SEED = SPLIT_SEED + UInt64(202)
const HELD_STATES = 128
const DEFAULT_DATASET = raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const DEFAULT_OUTPUT = raw"D:\tetris-paper-plus\runs\beat_first_v1\dynamic_sparse_recurrent_lookup"

@inline _property_or(value, name::Symbol, default) =
    value !== nothing && hasproperty(value, name) ? getproperty(value, name) : default

function _float32_env(name::AbstractString, default; nonnegative::Bool=false)
    value = parse(Float32, strip(get(ENV, name, string(default))))
    isfinite(value) || error("$name must be finite")
    if nonnegative
        value >= 0.0f0 || error("$name must be nonnegative")
    else
        value > 0.0f0 || error("$name must be positive")
    end
    return value
end

function _int_env(name::AbstractString, default; minimum::Int=0)
    value = parse(Int, strip(get(ENV, name, string(default))))
    value >= minimum || error("$name must be at least $minimum")
    return value
end

function _margin_mode(raw)
    normalized = replace(lowercase(strip(String(raw))), '-' => '_')
    normalized in ("fixed_top2", "fixed_teacher_top2") &&
        return TrainingCore.FIXED_TEACHER_TOP2_MARGIN_MODE
    normalized in ("hard_negative", "student_hard_negative") &&
        return TrainingCore.STUDENT_HARD_NEGATIVE_MARGIN_MODE
    error("DSRL_MARGIN_MODE must be fixed_top2 or student_hard_negative")
end

function _legacy_hyperparameters(maximum_updates::Int, payload)
    parent_config = payload === nothing ? nothing : payload.config
    parent_maximum = Int(_property_or(parent_config, :maximum_updates, maximum_updates))
    parent_objective = _property_or(parent_config, :objective, nothing)
    parent_halting = _property_or(parent_config, :halting, nothing)
    return (;
        optimizer=(;
            beta1=0.9f0,
            beta2=0.999f0,
            epsilon=1.0f-8,
            bank_learning_rate=1.0f-4,
            bh4_learning_rate=2.0f-4,
            alpha_learning_rate=1.0f-4,
            head_learning_rate=1.0f-4,
            halt_learning_rate=5.0f-5,
            reinject_learning_rate=1.0f-4,
            bank_weight_decay=0.0f0,
            dense_weight_decay=0.0f0,
        ),
        loss=(;
            listnet_weight=1.0f0,
            old_q_weight=Float32(_property_or(parent_objective, :old_q_weight, TrainingCore.OLD_Q_WEIGHT)),
            margin_weight=Float32(_property_or(parent_objective, :margin_weight, 1.0f0)),
            death_weight=Float32(_property_or(parent_objective, :death_weight, TrainingCore.DEATH_WEIGHT)),
            quantile_weight=Float32(_property_or(parent_objective, :quantile_weight, TrainingCore.QUANTILE_TEACHER_WEIGHT)),
            geometry_weight=Float32(_property_or(parent_objective, :geometry_weight, TrainingCore.GEOMETRY_WEIGHT)),
            margin_mode=_margin_mode(_property_or(parent_objective, :margin_mode, TrainingCore.FIXED_TEACHER_TOP2_MARGIN_MODE)),
            hard_negative_margin_floor=Float32(_property_or(
                parent_objective, :hard_negative_margin_floor, 0.0f0,
            )),
        ),
        routing=(;
            start_temperature=1.0f0,
            end_temperature=0.25f0,
            anneal_start_update=0,
            anneal_end_update=parent_maximum,
        ),
        halting=(;
            warmup_updates=Int(_property_or(parent_config, :warmup_updates, 1_000)),
            compute_price=Float32(_property_or(parent_halting, :compute_price, 0.02f0)),
            policy_weight=Float32(_property_or(parent_halting, :policy_weight, 0.05f0)),
            entropy_weight=Float32(_property_or(parent_halting, :entropy_weight, 0.001f0)),
        ),
    )
end

function inherited_hyperparameters(maximum_updates::Int, payload=nothing)
    return payload !== nothing && hasproperty(payload.config, :hyperparameters) ?
        payload.config.hyperparameters : _legacy_hyperparameters(maximum_updates, payload)
end

function runtime_hyperparameters(maximum_updates::Int, payload=nothing)
    inherited = inherited_hyperparameters(maximum_updates, payload)
    inherited_optimizer = inherited.optimizer
    inherited_loss = inherited.loss
    inherited_routing = inherited.routing
    inherited_halting = inherited.halting
    optimizer = (;
        beta1=_float32_env("DSRL_ADAM_BETA1", inherited_optimizer.beta1; nonnegative=true),
        beta2=_float32_env("DSRL_ADAM_BETA2", inherited_optimizer.beta2; nonnegative=true),
        epsilon=_float32_env("DSRL_ADAM_EPSILON", inherited_optimizer.epsilon),
        bank_learning_rate=_float32_env("DSRL_LR_BANK", inherited_optimizer.bank_learning_rate),
        bh4_learning_rate=_float32_env("DSRL_LR_BH4", inherited_optimizer.bh4_learning_rate),
        alpha_learning_rate=_float32_env("DSRL_LR_ALPHA", inherited_optimizer.alpha_learning_rate),
        head_learning_rate=_float32_env("DSRL_LR_HEAD", inherited_optimizer.head_learning_rate),
        halt_learning_rate=_float32_env("DSRL_LR_HALT", inherited_optimizer.halt_learning_rate),
        reinject_learning_rate=_float32_env("DSRL_LR_REINJECT", inherited_optimizer.reinject_learning_rate),
        bank_weight_decay=_float32_env("DSRL_WD_BANK", inherited_optimizer.bank_weight_decay; nonnegative=true),
        dense_weight_decay=_float32_env("DSRL_WD_DENSE", inherited_optimizer.dense_weight_decay; nonnegative=true),
    )
    0.0f0 <= optimizer.beta1 < 1.0f0 || error("DSRL_ADAM_BETA1 must be in [0, 1)")
    0.0f0 <= optimizer.beta2 < 1.0f0 || error("DSRL_ADAM_BETA2 must be in [0, 1)")
    Float64(optimizer.bank_learning_rate) * Float64(optimizer.bank_weight_decay) < 1.0 ||
        error("DSRL_LR_BANK * DSRL_WD_BANK must be less than one")
    maximum(Float64.((
        optimizer.bh4_learning_rate,
        optimizer.alpha_learning_rate,
        optimizer.head_learning_rate,
        optimizer.halt_learning_rate,
        optimizer.reinject_learning_rate,
    ))) * Float64(optimizer.dense_weight_decay) < 1.0 ||
        error("every dense learning rate times DSRL_WD_DENSE must be less than one")
    loss = (;
        listnet_weight=_float32_env("DSRL_LOSS_LISTNET", inherited_loss.listnet_weight; nonnegative=true),
        old_q_weight=_float32_env("DSRL_LOSS_OLD_Q", inherited_loss.old_q_weight; nonnegative=true),
        margin_weight=_float32_env("DSRL_LOSS_MARGIN", inherited_loss.margin_weight; nonnegative=true),
        death_weight=_float32_env("DSRL_LOSS_DEATH", inherited_loss.death_weight; nonnegative=true),
        quantile_weight=_float32_env("DSRL_LOSS_QUANTILE", inherited_loss.quantile_weight; nonnegative=true),
        geometry_weight=_float32_env("DSRL_LOSS_GEOMETRY", inherited_loss.geometry_weight; nonnegative=true),
        margin_mode=_margin_mode(get(ENV, "DSRL_MARGIN_MODE", string(inherited_loss.margin_mode))),
        hard_negative_margin_floor=_float32_env(
            "DSRL_HARDNEG_MARGIN_FLOOR",
            _property_or(inherited_loss, :hard_negative_margin_floor, 0.0f0);
            nonnegative=true,
        ),
    )
    routing = (;
        start_temperature=_float32_env("DSRL_ROUTE_TEMP_START", inherited_routing.start_temperature),
        end_temperature=_float32_env("DSRL_ROUTE_TEMP_END", inherited_routing.end_temperature),
        anneal_start_update=_int_env("DSRL_ROUTE_ANNEAL_START", inherited_routing.anneal_start_update),
        anneal_end_update=_int_env("DSRL_ROUTE_ANNEAL_END", inherited_routing.anneal_end_update; minimum=1),
    )
    routing.anneal_end_update > routing.anneal_start_update || error(
        "routing anneal end must be after its start",
    )
    halting = (;
        warmup_updates=_int_env("DSRL_WARMUP_UPDATES", inherited_halting.warmup_updates),
        compute_price=_float32_env("DSRL_COMPUTE_PRICE", inherited_halting.compute_price; nonnegative=true),
        policy_weight=_float32_env("DSRL_POLICY_WEIGHT", inherited_halting.policy_weight; nonnegative=true),
        entropy_weight=_float32_env("DSRL_ENTROPY_WEIGHT", inherited_halting.entropy_weight; nonnegative=true),
    )
    return (; optimizer, loss, routing, halting)
end

function objective_contract(hyperparameters)
    loss = hyperparameters.loss
    return (;
        objective_mode=TrainingCore.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
        margin_mode=loss.margin_mode,
        hard_negative_margin_floor=loss.hard_negative_margin_floor,
        listnet_weight=loss.listnet_weight,
        margin_weight=loss.margin_weight,
        listnet_temperature=TrainingCore.LISTNET_TEMPERATURE,
        old_q_weight=loss.old_q_weight,
        death_weight=loss.death_weight,
        quantile_weight=loss.quantile_weight,
        geometry_weight=loss.geometry_weight,
    )
end

@inline function routing_temperature(update::Int, schedule)
    progress = clamp(
        Float32(update - schedule.anneal_start_update) /
        Float32(max(schedule.anneal_end_update - schedule.anneal_start_update, 1)),
        0.0f0,
        1.0f0,
    )
    return schedule.start_temperature -
        (schedule.start_temperature - schedule.end_temperature) * progress
end

mutable struct TeacherWorkspace
    q64::Vector{Float32}
    x496::Vector{Float32}
    context::Vector{Float32}
    next_hold::Vector{Float32}
    raw::Matrix{Float32}
    tapes::Vector{Union{Nothing,Model.TrajectoryTape}}
    depths::Vector{Int16}
end

function TeacherWorkspace()
    return TeacherWorkspace(
        zeros(Float32, SparseFeatures.ROUTE_FEATURES),
        zeros(Float32, SparseFeatures.VALUE_FEATURES),
        zeros(Float32, InputAdapter.CONTEXT_DIM),
        zeros(Float32, InputAdapter.NEXT_HOLD_DIM),
        zeros(Float32, Model.OUTPUT_DIM, LEARNER_WIDTH),
        Union{Nothing,Model.TrajectoryTape}[nothing for _ in 1:LEARNER_WIDTH],
        zeros(Int16, LEARNER_WIDTH),
    )
end

mutable struct TeacherTrainer
    model::Model.DynamicLookupModel
    optimizer::Model.DynamicLookupOptimizer
    halt_rng::Xoshiro
    usage::Model.RouteUsage
    workspace::TeacherWorkspace
    baseline::Float32
    update::Int
    timed_updates::UInt64
    timed_candidates::UInt64
    timed_recurrent_steps::UInt64
    training_nanoseconds::UInt128
end

function initialize_trainer(hyperparameters=runtime_hyperparameters(20_000))
    model = Model.initialize_model(Xoshiro(MODEL_SEED))
    Model.parameter_count(model) == Model.TOTAL_PARAMETERS || error("model geometry changed")
    optimizer = hyperparameters.optimizer
    return TeacherTrainer(
        model,
        Model.initialize_optimizer(
            model;
            beta1=optimizer.beta1,
            beta2=optimizer.beta2,
            epsilon=optimizer.epsilon,
            bank_learning_rate=optimizer.bank_learning_rate,
            bank_weight_decay=optimizer.bank_weight_decay,
        ),
        Xoshiro(HALT_SEED),
        Model.RouteUsage(),
        TeacherWorkspace(),
        8.0f0,
        0,
        0,
        0,
        0,
        0,
    )
end

function _valid_candidate_count(batch)
    size(batch.mask) == (LEARNER_WIDTH, 1) || throw(DimensionMismatch("teacher batch geometry changed"))
    count = Int(sum(@view batch.mask[:, 1]))
    1 <= count <= LEARNER_WIDTH || error("invalid candidate count")
    all(@view(batch.mask[1:count, 1]) .== 1.0f0) || error("candidate mask is not prefix-valid")
    if count < LEARNER_WIDTH
        all(@view(batch.mask[(count + 1):end, 1]) .== 0.0f0) || error("candidate padding is nonzero")
    end
    return count
end

function _candidate_input!(workspace::TeacherWorkspace, batch, candidate::Int)
    SparseFeatures.split_candidate_features!(workspace.q64, workspace.x496, batch.inputs, candidate)
    position = 1
    @inbounds for aux_index in SparseFeatures.ROUTE_AUX_INDICES
        workspace.context[position] = batch.inputs.aux[aux_index, candidate]
        position += 1
    end
    position == InputAdapter.CONTEXT_DIM + 1 || error("context width changed")
    position = 1
    @inbounds for token in 1:SparseFeatures.NEXT_HOLD_TOKENS, piece in 1:SparseFeatures.NEXT_HOLD_PIECES
        workspace.next_hold[position] = batch.inputs.next_hold[piece, token, candidate]
        position += 1
    end
    return InputAdapter.ResidualLookupInput(workspace.x496, workspace.context, workspace.next_hold)
end

function predict_raw!(
    trainer::TeacherTrainer,
    batch;
    training::Bool,
    expected_update::Int=trainer.update,
    hyperparameters=runtime_hyperparameters(20_000),
    record_tapes::Bool=training,
)
    count = _valid_candidate_count(batch)
    workspace = trainer.workspace
    fill!(workspace.raw, 0.0f0)
    fill!(workspace.tapes, nothing)
    fill!(workspace.depths, 0)
    temperature = routing_temperature(expected_update, hyperparameters.routing)
    materialize = training ? ((block, columns) ->
        InputAdapter.materialize_selected_columns!(
            trainer.model.banks[block], trainer.optimizer.bank_states[block], columns,
        )) : nothing
    warmup = training && expected_update <= hyperparameters.halting.warmup_updates
    for candidate in 1:count
        input = _candidate_input!(workspace, batch, candidate)
        forced_depth = warmup ? rand(trainer.halt_rng, Model.MIN_RECURRENT_STEPS:Model.WARMUP_MAX_STEPS) : nothing
        result = Model.forward_trajectory(
            trainer.model,
            input;
            rng=training ? trainer.halt_rng : nothing,
            training,
            forced_depth,
            temperature,
            usage=training ? trainer.usage : nothing,
            materialize,
        )
        workspace.raw[:, candidate] .= result.output
        workspace.depths[candidate] = Int16(result.depth)
        record_tapes && (workspace.tapes[candidate] = result.tape)
    end
    all(isfinite, workspace.raw) || error("model output is non-finite")
    return workspace.raw, count
end

raw_output(raw::AbstractMatrix) = ParentTraining.raw_output(raw)

function _hard_negative_selection(raw::AbstractMatrix, batch, margin_mode)
    margin_mode === TrainingCore.STUDENT_HARD_NEGATIVE_MARGIN_MODE || return nothing
    output = raw_output(raw)
    return TrainingCore.hard_negative_selection(
        reshape(output.q, size(batch.mask)),
        batch,
    )
end

@inline function _weighted_loss(components, loss_weights)
    return loss_weights.listnet_weight * components.listnet_loss +
        loss_weights.old_q_weight * components.old_q_loss +
        loss_weights.margin_weight * components.margin_loss +
        loss_weights.death_weight * components.death_loss +
        loss_weights.quantile_weight * components.quantile_teacher_loss +
        loss_weights.geometry_weight * components.geometry_loss
end

function _weighted_components(raw, batch, hyperparameters; hard_negative=nothing)
    loss_weights = hyperparameters.loss
    components = TrainingCore.supervised_components(
        raw_output(raw),
        batch;
        margin_weight=loss_weights.margin_weight,
        margin_mode=loss_weights.margin_mode,
        objective_mode=TrainingCore.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
        hard_negative,
        hard_negative_margin_floor=loss_weights.hard_negative_margin_floor,
    )
    composite_loss = _weighted_loss(components, loss_weights)
    effective_raw_top_gap_weight =
        loss_weights.margin_mode === TrainingCore.FIXED_TEACHER_TOP2_MARGIN_MODE ?
        loss_weights.margin_weight : 0.0f0
    return merge(components, (;
        composite_loss,
        effective_listnet_weight=loss_weights.listnet_weight,
        effective_margin_weight=loss_weights.margin_weight,
        effective_raw_top_gap_weight,
    ))
end

function _loss_output_vjp(raw::Matrix{Float32}, batch, hyperparameters; hard_negative=nothing)
    loss, pullback = Zygote.pullback(raw) do candidate_outputs
        _weighted_components(
            candidate_outputs,
            batch,
            hyperparameters;
            hard_negative,
        ).composite_loss
    end
    gradient = only(pullback(one(loss)))
    gradient === nothing && error("loss VJP returned no gradient")
    result = Matrix{Float32}(gradient)
    all(isfinite, result) || error("loss VJP is non-finite")
    return Float32(loss), result
end

function _component_record(components)
    names = (
        :composite_loss, :listnet_loss, :old_q_loss, :q_huber_loss,
        :margin_loss, :raw_top_gap_loss, :death_loss, :quantile_teacher_loss,
        :geometry_loss, :line_clear_loss, :max_height_loss, :holes_loss,
        :cavities_loss, :valid_candidates,
    )
    return NamedTuple{names}(Tuple(Float64(getproperty(components, name)) for name in names))
end

function _accumulate_state_gradient!(
    trainer::TeacherTrainer,
    batch,
    accumulator::Model.GradientAccumulator;
    expected_update::Int,
    hyperparameters,
    baseline::Float32,
)
    raw, candidate_count = predict_raw!(
        trainer, batch; training=true, expected_update, hyperparameters,
    )
    hard_negative = _hard_negative_selection(raw, batch, hyperparameters.loss.margin_mode)
    components = _weighted_components(raw, batch, hyperparameters; hard_negative)
    loss, raw_gradient = _loss_output_vjp(
        raw,
        batch,
        hyperparameters;
        hard_negative,
    )
    temperature = routing_temperature(expected_update, hyperparameters.routing)
    for candidate in 1:candidate_count
        tape = trainer.workspace.tapes[candidate]
        tape === nothing && error("training trajectory is missing")
        Model.backward_trajectory!(
            accumulator,
            trainer.model,
            tape,
            @view(raw_gradient[:, candidate]);
            realized_loss=loss,
            baseline,
            compute_price=hyperparameters.halting.compute_price,
            policy_weight=hyperparameters.halting.policy_weight,
            entropy_weight=hyperparameters.halting.entropy_weight,
            temperature,
        )
    end
    depths = Int.(view(trainer.workspace.depths, 1:candidate_count))
    return (;
        loss,
        components=_component_record(components),
        candidate_count,
        depths,
    )
end

function _mean_component_records(records)
    names = propertynames(first(records))
    values = map(names) do name
        mean(getproperty(record, name) for record in records)
    end
    return NamedTuple{names}(values)
end

function train_accumulated_step!(
    trainer::TeacherTrainer,
    batches;
    expected_update::Int,
    hyperparameters,
)
    expected_update == trainer.update + 1 || error("non-adjacent training update")
    isempty(batches) && error("training state batch is empty")
    started = time_ns()
    accumulator = Model.GradientAccumulator()
    baseline = trainer.baseline
    temperature = routing_temperature(expected_update, hyperparameters.routing)
    state_results = map(batches) do batch
        _accumulate_state_gradient!(
            trainer,
            batch,
            accumulator;
            expected_update,
            hyperparameters,
            baseline,
        )
    end
    state_batch = length(state_results)
    Model.scale_gradients!(accumulator, inv(Float32(state_batch)))
    optimizer_hyperparameters = hyperparameters.optimizer
    optimizer = Model.optimizer_step!(
        trainer.model,
        trainer.optimizer,
        accumulator;
        beta1=optimizer_hyperparameters.beta1,
        beta2=optimizer_hyperparameters.beta2,
        epsilon=optimizer_hyperparameters.epsilon,
        bh4_learning_rate=optimizer_hyperparameters.bh4_learning_rate,
        alpha_learning_rate=optimizer_hyperparameters.alpha_learning_rate,
        head_learning_rate=optimizer_hyperparameters.head_learning_rate,
        halt_learning_rate=optimizer_hyperparameters.halt_learning_rate,
        reinject_learning_rate=optimizer_hyperparameters.reinject_learning_rate,
        dense_weight_decay=optimizer_hyperparameters.dense_weight_decay,
    )
    loss = mean(result.loss for result in state_results)
    component_records = [result.components for result in state_results]
    candidate_count = sum(result.candidate_count for result in state_results)
    depths = reduce(vcat, (result.depths for result in state_results))
    mean_depth = mean(depths)
    realized_cost = mean(
        result.loss + hyperparameters.halting.compute_price * Float32(mean(result.depths))
        for result in state_results
    )
    trainer.baseline = muladd(0.99f0, trainer.baseline, 0.01f0 * realized_cost)
    trainer.update = expected_update
    trainer.timed_updates += 1
    trainer.timed_candidates += UInt64(candidate_count)
    trainer.timed_recurrent_steps += UInt64(sum(depths))
    elapsed = UInt128(time_ns() - started)
    trainer.training_nanoseconds += elapsed
    trainer.optimizer.step == UInt64(expected_update) || error("optimizer clock diverged")
    return (;
        update=expected_update,
        state_batch,
        candidate_count,
        loss=Float64(loss),
        components=_mean_component_records(component_records),
        mean_depth,
        minimum_depth=minimum(depths),
        maximum_depth=maximum(depths),
        warmup=expected_update <= hyperparameters.halting.warmup_updates,
        routing_temperature=Float64(temperature),
        baseline=Float64(trainer.baseline),
        optimizer,
        training_seconds=Float64(elapsed) * 1.0e-9,
    )
end


function train_step!(trainer::TeacherTrainer, batch; expected_update::Int, hyperparameters)
    return train_accumulated_step!(
        trainer,
        (batch,);
        expected_update,
        hyperparameters,
    )
end

function held_evaluation(trainer, dataset, rows, host_batch; hyperparameters)
    depths = Int[]
    metrics = TrainingCore.evaluation_metrics(
        dataset,
        Int.(rows),
        host_batch,
        batch -> begin
            raw, count = predict_raw!(
                trainer, batch; training=false, expected_update=trainer.update,
                hyperparameters, record_tapes=false,
            )
            append!(depths, Int.(view(trainer.workspace.depths, 1:count)))
            raw_output(raw)
        end;
        margin_weight=hyperparameters.loss.margin_weight,
        margin_mode=hyperparameters.loss.margin_mode,
        objective_mode=TrainingCore.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
        hard_negative_margin_floor=hyperparameters.loss.hard_negative_margin_floor,
    )
    metrics = merge(metrics, (;
        composite_loss=_weighted_loss(metrics, hyperparameters.loss),
        effective_listnet_weight=hyperparameters.loss.listnet_weight,
        effective_margin_weight=hyperparameters.loss.margin_weight,
        effective_raw_top_gap_weight=
            hyperparameters.loss.margin_mode === TrainingCore.FIXED_TEACHER_TOP2_MARGIN_MODE ?
            hyperparameters.loss.margin_weight : 0.0f0,
    ))
    return (;
        metrics,
        mean_depth=mean(depths),
        minimum_depth=minimum(depths),
        maximum_depth=maximum(depths),
    )
end

function _throughput(trainer)
    seconds = Float64(trainer.training_nanoseconds) * 1.0e-9
    return (;
        training_seconds=seconds,
        updates_per_second=seconds == 0 ? 0.0 : Float64(trainer.timed_updates) / seconds,
        candidates_per_second=seconds == 0 ? 0.0 : Float64(trainer.timed_candidates) / seconds,
        recurrent_steps_per_second=seconds == 0 ? 0.0 : Float64(trainer.timed_recurrent_steps) / seconds,
        mean_training_depth=trainer.timed_candidates == 0 ? 0.0 :
            Float64(trainer.timed_recurrent_steps) / Float64(trainer.timed_candidates),
    )
end

function _counter_snapshot(trainer)
    return (;
        update=trainer.update,
        timed_updates=trainer.timed_updates,
        timed_candidates=trainer.timed_candidates,
        timed_recurrent_steps=trainer.timed_recurrent_steps,
        training_nanoseconds=trainer.training_nanoseconds,
    )
end

function _segment_throughput(trainer, start)
    updates = trainer.timed_updates - start.timed_updates
    candidates = trainer.timed_candidates - start.timed_candidates
    recurrent_steps = trainer.timed_recurrent_steps - start.timed_recurrent_steps
    nanoseconds = trainer.training_nanoseconds - start.training_nanoseconds
    seconds = Float64(nanoseconds) * 1.0e-9
    return (;
        start_update=start.update,
        completed_updates=Int(updates),
        training_seconds=seconds,
        updates_per_second=seconds == 0 ? 0.0 : Float64(updates) / seconds,
        candidates_per_second=seconds == 0 ? 0.0 : Float64(candidates) / seconds,
        recurrent_steps_per_second=seconds == 0 ? 0.0 : Float64(recurrent_steps) / seconds,
        mean_training_depth=candidates == 0 ? 0.0 :
            Float64(recurrent_steps) / Float64(candidates),
    )
end

function _sha256_file(path)
    open(path, "r") do io
        return bytes2hex(SHA.sha256(io))
    end
end

function _source_paths()
    return [
        joinpath(@__DIR__, "DynamicSparseRecurrentLookup.jl"),
        joinpath(@__DIR__, "teacher_training.jl"),
        joinpath(@__DIR__, "..", "residual_lookup_slide", "geometry.jl"),
        joinpath(@__DIR__, "..", "residual_lookup_slide", "hash.jl"),
        joinpath(@__DIR__, "..", "residual_lookup_slide", "optimizer.jl"),
    ]
end

function source_fingerprint()
    io = IOBuffer()
    for path in _source_paths()
        write(io, codeunits(abspath(path)))
        write(io, read(path))
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function _dataset_manifest_sha256(dataset_path)
    manifest = joinpath(dataset_path, "manifest.json")
    isfile(manifest) || error("teacher dataset manifest is missing")
    return _sha256_file(manifest)
end

function _append_jsonl(path, record)
    open(path, "a") do io
        JSON3.write(io, record)
        write(io, '\n')
        flush(io)
    end
end

function _write_json(path, record)
    open(path, "w") do io
        JSON3.pretty(io, record)
        write(io, '\n')
        flush(io)
    end
end

function save_checkpoint(path, trainer, sampler, history, config, split_metadata)
    consumed_states = TrainingCore.sampler_consumed_states(sampler)
    payload = (;
        format="dynamic-sparse-recurrent-lookup-checkpoint",
        version=1,
        update=trainer.update,
        model=trainer.model,
        optimizer=trainer.optimizer,
        halt_rng=copy(trainer.halt_rng),
        usage=trainer.usage,
        baseline=trainer.baseline,
        timed_updates=trainer.timed_updates,
        timed_candidates=trainer.timed_candidates,
        timed_recurrent_steps=trainer.timed_recurrent_steps,
        training_nanoseconds=trainer.training_nanoseconds,
        consumed_states,
        sampler_snapshot=TrainingCore.sampler_snapshot(sampler),
        history,
        config,
        split_metadata,
    )
    temporary = path * ".tmp-" * string(getpid())
    open(temporary, "w") do io
        serialize(io, payload)
        flush(io)
    end
    mv(temporary, path; force=true)
    return (;
        path=abspath(path),
        bytes=filesize(path),
        sha256=_sha256_file(path),
        update=trainer.update,
        consumed_states,
    )
end

function read_checkpoint(path::AbstractString, expected_sha256::AbstractString="")
    checkpoint_path = abspath(path)
    isfile(checkpoint_path) || error("resume checkpoint does not exist: $checkpoint_path")
    actual_sha256 = _sha256_file(checkpoint_path)
    expected = lowercase(strip(expected_sha256))
    isempty(expected) || actual_sha256 == expected || error("resume checkpoint SHA-256 differs")
    payload = open(checkpoint_path, "r") do io
        deserialize(io)
    end
    hasproperty(payload, :format) && payload.format == "dynamic-sparse-recurrent-lookup-checkpoint" ||
        error("resume checkpoint format differs")
    hasproperty(payload, :version) && Int(payload.version) == 1 ||
        error("unsupported resume checkpoint version")
    return payload, (;
        path=checkpoint_path,
        bytes=filesize(checkpoint_path),
        sha256=actual_sha256,
    )
end

function restore_checkpoint(payload, split, split_metadata, dataset_manifest_sha256)
    required = (
        :update, :model, :optimizer, :halt_rng, :usage, :baseline,
        :timed_updates, :timed_candidates, :timed_recurrent_steps,
        :training_nanoseconds, :sampler_snapshot, :history, :config,
        :split_metadata,
    )
    all(name -> hasproperty(payload, name), required) || error("resume checkpoint is incomplete")
    update = Int(payload.update)
    update >= 0 || error("resume update is negative")
    Model.parameter_count(payload.model) == Model.TOTAL_PARAMETERS || error(
        "resume model geometry differs from DSRL_WTA_CHOICES/DSRL_TABLES_PER_BLOCK",
    )
    payload.optimizer.step == UInt64(update) || error("resume optimizer clock differs")
    payload.optimizer.dense.step == UInt64(update) || error("resume dense optimizer clock differs")
    all(state.global_step == UInt64(update) for state in payload.optimizer.bank_states) ||
        error("resume sparse optimizer clocks differ")
    resume_topology = _property_or(payload.config, :model, nothing)
    resume_lookup_width = Int(_property_or(resume_topology, :rows_per_table_lookup, 1))
    resume_lookup_width in (1, 2, 3) || error("resume lookup width is unsupported")
    resume_lookup_width == Model.ROWS_PER_TABLE_LOOKUP ||
        haskey(ENV, "DSRL_ROWS_PER_TABLE_LOOKUP") || error(
            "changing resume lookup width requires explicit DSRL_ROWS_PER_TABLE_LOOKUP",
        )
    payload.split_metadata == split_metadata || error("resume split metadata differs")
    hasproperty(payload.config, :dataset_manifest_sha256) || error(
        "resume config lacks dataset manifest SHA-256",
    )
    payload.config.dataset_manifest_sha256 == dataset_manifest_sha256 || error(
        "resume dataset manifest differs",
    )
    hasproperty(payload.config, :julia_version) && payload.config.julia_version == string(VERSION) ||
        error("resume Julia version differs")
    sampler = TrainingCore.restore_sampler(split.training_rows, payload.sampler_snapshot)
    expected_consumed_states = hasproperty(payload, :consumed_states) ?
        Int(payload.consumed_states) : update
    TrainingCore.sampler_consumed_states(sampler) == expected_consumed_states || error(
        "resume sampler position differs from recorded consumed states",
    )
    size(payload.usage.counts) == (Model.ROWS_PER_TABLE, Model.TABLES_PER_BLOCK, Model.BLOCKS) ||
        error("resume route-usage geometry differs")
    trainer = TeacherTrainer(
        payload.model,
        payload.optimizer,
        copy(payload.halt_rng),
        payload.usage,
        TeacherWorkspace(),
        Float32(payload.baseline),
        update,
        UInt64(payload.timed_updates),
        UInt64(payload.timed_candidates),
        UInt64(payload.timed_recurrent_steps),
        UInt128(payload.training_nanoseconds),
    )
    return trainer, sampler, copy(payload.history)
end

function _evaluation_record(
    trainer,
    dataset,
    held_rows,
    host_batch,
    last_step;
    hyperparameters,
    segment_start,
)
    held = held_evaluation(trainer, dataset, held_rows, host_batch; hyperparameters)
    metrics = held.metrics
    return (;
        update=trainer.update,
        experiment_id=EXPERIMENT_ID,
        held_signal=(;
            loss=Float64(metrics.composite_loss),
            ndcg=Float64(metrics.ndcg),
            pairwise_accuracy=Float64(metrics.pairwise_accuracy),
            top1_agreement=Float64(metrics.top1_agreement),
            action_margin=Float64(metrics.action_margin),
        ),
        recurrent_depth=(;
            mean=held.mean_depth,
            minimum=held.minimum_depth,
            maximum=held.maximum_depth,
        ),
        route_usage=Model.usage_summary(trainer.usage),
        throughput=_throughput(trainer),
        segment_throughput=_segment_throughput(trainer, segment_start),
        last_step,
        parameter_contract=Model.topology(trainer.model),
        objective=objective_contract(hyperparameters),
        source_fingerprint=source_fingerprint(),
    )
end

function teacher_signal_cli_main()
    maximum_updates = parse(Int, get(ENV, "DSRL_MAX_UPDATES", "20000"))
    maximum_updates >= 1 || error("maximum updates must be positive")
    dataset_path = abspath(get(ENV, "DSRL_DATASET", DEFAULT_DATASET))
    output_root = abspath(get(ENV, "DSRL_OUTPUT", DEFAULT_OUTPUT))
    run_id = strip(get(ENV, "DSRL_RUN_ID", ""))
    isempty(run_id) && (run_id = "dynamic_sparse_recurrent_lookup_v1_" * Dates.format(now(), "yyyymmddTHHMMSS"))
    occursin(r"^[A-Za-z0-9_.-]+$", run_id) || error("unsafe run ID")
    run_dir = joinpath(output_root, run_id)
    ispath(run_dir) && error("fresh run already exists: $run_dir")
    write_checkpoints = get(ENV, "DSRL_WRITE_CHECKPOINTS", "1") == "1"
    resume_path = strip(get(ENV, "DSRL_RESUME_CHECKPOINT", ""))
    resume_sha256 = strip(get(ENV, "DSRL_RESUME_SHA256", ""))
    resume_payload, resume_artifact = isempty(resume_path) ? (nothing, nothing) :
        read_checkpoint(resume_path, resume_sha256)
    inherited_state_batch = resume_payload === nothing ? 1 :
        Int(_property_or(resume_payload.config, :state_batch, 1))
    state_batch = _int_env("DSRL_STATE_BATCH", inherited_state_batch; minimum=1)
    hyperparameters = runtime_hyperparameters(maximum_updates, resume_payload)
    default_interval = maximum_updates < 2_500 ? maximum_updates : 2_500
    evaluation_interval = _int_env("DSRL_EVAL_INTERVAL", default_interval; minimum=1)
    checkpoint_interval = _int_env("DSRL_CHECKPOINT_INTERVAL", default_interval; minimum=1)

    dataset = TrainingCore.load_teacher_dataset(
        dataset_path; max_candidates=TrainingCore.MAX_CANDIDATES,
        allow_partial_dataset=false,
    )
    maximum(dataset.action_counts) <= LEARNER_WIDTH || error("teacher data exceeds learner width")
    split = ParentTraining.episode_separated_split(
        dataset; seed=SPLIT_SEED, validation_fraction=0.20,
    )
    held_rows = ParentTraining.fixed_evaluation_subset(
        split.validation_rows, HELD_STATES, HELD_SEED,
    )
    split_metadata = (;
        held_rows=Int.(held_rows),
        training_groups=copy(split.training_groups),
        validation_groups=copy(split.validation_groups),
        predefined=split.predefined,
    )
    dataset_manifest_sha256 = _dataset_manifest_sha256(dataset_path)
    trainer, sampler, history = if resume_payload === nothing
        fresh_trainer = initialize_trainer(hyperparameters)
        fresh_sampler = TrainingCore.EpochSampler(split.training_rows, Xoshiro(SAMPLER_SEED))
        fresh_trainer, fresh_sampler, Any[]
    else
        restored_trainer, restored_sampler, restored_history = restore_checkpoint(
            resume_payload,
            split,
            split_metadata,
            dataset_manifest_sha256,
        )
        inherited_optimizer = inherited_hyperparameters(
            maximum_updates,
            resume_payload,
        ).optimizer
        for name in (:beta1, :beta2, :epsilon)
            getproperty(hyperparameters.optimizer, name) == getproperty(inherited_optimizer, name) ||
                error("$name may change only on a fresh run")
        end
        Model.configure_bank_optimizer!(
            restored_trainer.optimizer;
            beta1=hyperparameters.optimizer.beta1,
            beta2=hyperparameters.optimizer.beta2,
            epsilon=hyperparameters.optimizer.epsilon,
            bank_learning_rate=hyperparameters.optimizer.bank_learning_rate,
            bank_weight_decay=hyperparameters.optimizer.bank_weight_decay,
        )
        restored_trainer, restored_sampler, restored_history
    end
    maximum_updates > trainer.update || error(
        "DSRL_MAX_UPDATES is an absolute target and must exceed the starting update",
    )
    segment_start = _counter_snapshot(trainer)
    parent_checkpoint = resume_payload === nothing ? nothing : (;
        resume_artifact...,
        update=Int(resume_payload.update),
        source_fingerprint=String(_property_or(resume_payload.config, :source_fingerprint, "")),
        rows_per_table_lookup=Int(_property_or(
            _property_or(resume_payload.config, :model, nothing),
            :rows_per_table_lookup,
            1,
        )),
    )
    config = (;
        experiment_id=EXPERIMENT_ID,
        run_id,
        dataset_path,
        dataset_manifest_sha256,
        source_fingerprint=source_fingerprint(),
        julia_version=string(VERSION),
        maximum_updates,
        evaluation_interval,
        checkpoint_interval,
        state_batch,
        starting_consumed_states=TrainingCore.sampler_consumed_states(sampler),
        warmup_updates=hyperparameters.halting.warmup_updates,
        hyperparameters,
        parent_checkpoint,
        model=Model.topology(trainer.model),
        objective=objective_contract(hyperparameters),
        halting=(;
            mode="random-depth-warmup-then-sampled-hard-hazard",
            minimum_steps=Model.MIN_RECURRENT_STEPS,
            maximum_steps=Model.MAX_RECURRENT_STEPS,
            compute_price=hyperparameters.halting.compute_price,
            policy_weight=hyperparameters.halting.policy_weight,
            entropy_weight=hyperparameters.halting.entropy_weight,
        ),
        forbidden_game_seeds_touched=false,
    )

    mkpath(joinpath(run_dir, "checkpoints"))
    _write_json(joinpath(run_dir, "config.json"), config)
    metrics_path = joinpath(run_dir, "metrics.jsonl")
    host_batches = [
        TrainingCore.allocate_host_batch(1; max_candidates=LEARNER_WIDTH)
        for _ in 1:state_batch
    ]
    host_batch = first(host_batches)
    initial = _evaluation_record(
        trainer,
        dataset,
        held_rows,
        host_batch,
        nothing;
        hyperparameters,
        segment_start,
    )
    push!(history, initial)
    _append_jsonl(metrics_path, initial)
    @info "Dynamic recurrent LookupFFN initial teacher signal" held=initial.held_signal depth=initial.recurrent_depth

    last_step = nothing
    while trainer.update < maximum_updates
        rows = TrainingCore.next_batch!(sampler, state_batch)
        for (batch, row) in zip(host_batches, rows)
            TrainingCore.pack_batch!(batch, dataset, [row])
        end
        last_step = train_accumulated_step!(
            trainer, host_batches;
            expected_update=trainer.update + 1,
            hyperparameters,
        )
        if trainer.update % 100 == 0
            @info "Dynamic recurrent LookupFFN progress" update=trainer.update loss=last_step.loss mean_depth=last_step.mean_depth throughput=_segment_throughput(trainer, segment_start)
        end
        if trainer.update % evaluation_interval == 0 || trainer.update == maximum_updates
            record = _evaluation_record(
                trainer,
                dataset,
                held_rows,
                host_batch,
                last_step;
                hyperparameters,
                segment_start,
            )
            push!(history, record)
            _append_jsonl(metrics_path, record)
            @info "Dynamic recurrent LookupFFN teacher signal" update=trainer.update held=record.held_signal depth=record.recurrent_depth throughput=record.throughput
        end
        if write_checkpoints &&
           (trainer.update % checkpoint_interval == 0 || trainer.update == maximum_updates)
            checkpoint_path = joinpath(
                run_dir, "checkpoints",
                "checkpoint_" * lpad(string(trainer.update), 9, '0') * ".jls",
            )
            artifact = save_checkpoint(
                checkpoint_path, trainer, sampler, history, config, split_metadata,
            )
            _write_json(joinpath(run_dir, "latest.json"), artifact)
            @info "Dynamic recurrent LookupFFN checkpoint" artifact
        end
    end
    trainer.update == maximum_updates || error("training stopped before target")
    summary = (;
        status="complete",
        run_dir=abspath(run_dir),
        metrics_path=abspath(metrics_path),
        update=trainer.update,
        consumed_states=TrainingCore.sampler_consumed_states(sampler),
        final=last(history),
        throughput=_throughput(trainer),
        segment_throughput=_segment_throughput(trainer, segment_start),
        parent_checkpoint,
        source_fingerprint=config.source_fingerprint,
    )
    _write_json(joinpath(run_dir, "summary.json"), summary)
    return summary
end

export TeacherTrainer, initialize_trainer, predict_raw!, train_step!, train_accumulated_step!, held_evaluation,
       save_checkpoint, read_checkpoint, restore_checkpoint, teacher_signal_cli_main,
       runtime_hyperparameters, source_fingerprint

end
