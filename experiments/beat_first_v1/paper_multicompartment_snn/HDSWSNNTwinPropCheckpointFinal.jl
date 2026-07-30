module HDSWSNNTwinPropCheckpointFinal

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
const MODEL_FAMILY = "HD-SWSNN-TwinProp"
const CHECKPOINT_SCHEMA =
    "hd-swsnn-twinprop-checkpoint-final-v1"

const REQUIRED_PROVENANCE_FIELDS = (
    :model_family,
    :cell_mode,
    :teacher_hash,
    :digital_twin_hash,
    :distilled_artifact_hash,
    :internal_parameter_sha256,
    :frozen_internal,
    :cell_mechanism_sha256,
    :morphology_sha256,
    :dt_ms,
    :substeps_per_cycle,
    :cycles,
    :decision_window_ms,
    :ablation_mode,
)

const REQUIRED_AUX_FIELDS = (
    :input_location,
    :recurrent_location,
    :workspace_location,
    :input_location_utility,
    :recurrent_location_utility,
    :workspace_location_utility,
    :contact_capacity,
    :regional_projection,
)

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

export CHECKPOINT_SCHEMA,
    MODEL_FAMILY,
    cell_artifact_metadata,
    checkpoint_sha256,
    load_checkpoint,
    restore_checkpoint!,
    save_checkpoint

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
    Set(keys(destination)) == Set(keys(source)) ||
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
    isequal(destination, source) ||
        error("$label immutable value differs")
    return destination
end

function _tree_equal(left::NamedTuple, right::NamedTuple)
    keys(left) == keys(right) || return false
    return all(
        name -> _tree_equal(
            getproperty(left, name),
            getproperty(right, name),
        ),
        keys(left),
    )
end

function _tree_equal(left::Tuple, right::Tuple)
    length(left) == length(right) || return false
    return all(
        index -> _tree_equal(left[index], right[index]),
        eachindex(left),
    )
end

function _tree_equal(left::AbstractArray, right::AbstractArray)
    size(left) == size(right) || return false
    eltype(left) == eltype(right) || return false
    return isequal(left, right)
end

function _tree_equal(left::AbstractDict, right::AbstractDict)
    Set(keys(left)) == Set(keys(right)) || return false
    return all(
        key -> _tree_equal(left[key], right[key]),
        keys(left),
    )
end

_tree_equal(left, right) = isequal(left, right)

function _serialized_sha256(value)
    stream = IOBuffer()
    Serialization.serialize(stream, value)
    return bytes2hex(SHA.sha256(take!(stream)))
end

function checkpoint_sha256(path::AbstractString)
    isfile(path) || error("artifact is absent: $path")
    return bytes2hex(SHA.sha256(read(path)))
end

function _mapping_value(source, name::Symbol)
    source === nothing && return nothing
    hasproperty(source, name) &&
        return getproperty(source, name)
    if source isa AbstractDict
        haskey(source, name) && return source[name]
        text = String(name)
        haskey(source, text) && return source[text]
    end
    return nothing
end

function _nested_value(source, names)
    for name in names
        value = _mapping_value(source, name)
        value === nothing || return value
    end
    for container_name in (
        :payload,
        :metadata,
        :provenance,
        :manifest,
        :training,
        :teacher,
        :digital_twin,
        :distillation,
        :config,
        :parameters,
    )
        child = _mapping_value(source, container_name)
        child === nothing && continue
        value = _nested_value(child, names)
        value === nothing || return value
    end
    return nothing
end

function _required_hash(source, names, label::AbstractString)
    raw = _nested_value(source, names)
    raw === nothing &&
        error("distilled artifact has no $label")
    value = lowercase(strip(String(raw)))
    occursin(r"^[0-9a-f]{64}$", value) ||
        error("$label must be exactly 64 hexadecimal digits")
    return value
end

function _required_number(source, names, label::AbstractString)
    raw = _nested_value(source, names)
    raw === nothing &&
        error("distilled artifact has no $label")
    return Float64(raw)
end

function _required_bool(source, names, label::AbstractString)
    raw = _nested_value(source, names)
    raw === nothing &&
        error("distilled artifact has no $label")
    raw isa Bool || error("$label must be Boolean")
    return raw
end

function cell_artifact_metadata(path::AbstractString)
    source = abspath(path)
    isfile(source) ||
        error("distilled cell artifact is absent: $source")
    data = JLD2.load(source)
    artifact_hash = checkpoint_sha256(source)
    schema = _nested_value(
        data,
        (:schema, :artifact_schema, :format),
    )
    schema === nothing &&
        error("distilled artifact has no schema")
    frozen_internal = _required_bool(
        data,
        (:frozen_internal, :internal_frozen, :is_frozen),
        "frozen_internal",
    )
    frozen_internal ||
        error("distilled artifact is not marked frozen")
    trainable = _nested_value(
        data,
        (:trainable_parameters,),
    )
    trainable === nothing ||
        isempty(trainable) ||
        error("distilled artifact exposes trainable internal parameters")
    return (;
        path=source,
        artifact_schema=String(schema),
        teacher_hash=_required_hash(
            data,
            (
                :teacher_hash,
                :detailed_teacher_hash,
                :teacher_sha256,
                :teacher_dataset_sha256,
            ),
            "teacher_hash",
        ),
        digital_twin_hash=_required_hash(
            data,
            (
                :digital_twin_hash,
                :digital_twin_sha256,
                :twin_hash,
                :twin_sha256,
            ),
            "digital_twin_hash",
        ),
        distilled_artifact_hash=artifact_hash,
        internal_parameter_sha256=_required_hash(
            data,
            (
                :internal_parameter_sha256,
                :parameter_sha256,
                :distilled_parameter_sha256,
            ),
            "internal_parameter_sha256",
        ),
        cell_mechanism_sha256=_required_hash(
            data,
            (
                :cell_mechanism_sha256,
                :mechanism_sha256,
                :detailed_cell_sha256,
            ),
            "cell_mechanism_sha256",
        ),
        morphology_sha256=_required_hash(
            data,
            (
                :morphology_sha256,
                :morphology_hash,
                :tree_sha256,
            ),
            "morphology_sha256",
        ),
        dt_ms=_required_number(
            data,
            (:dt_ms, :time_step_ms),
            "dt_ms",
        ),
        frozen_internal,
    )
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

function _require_property(source, name::Symbol, label)
    hasproperty(source, name) ||
        error("$label has no required field $name")
    return getproperty(source, name)
end

function _provenance(source, label::AbstractString)
    values = NamedTuple{REQUIRED_PROVENANCE_FIELDS}(
        map(
            name -> _require_property(source, name, label),
            REQUIRED_PROVENANCE_FIELDS,
        ),
    )
    values.model_family == MODEL_FAMILY ||
        error("$label model_family is not $MODEL_FAMILY")
    values.cell_mode in ("distilled-frozen", "detailed-control") ||
        error("$label has unsupported cell_mode")
    values.frozen_internal isa Bool ||
        error("$label frozen_internal is not Boolean")
    values.cell_mode == "distilled-frozen" &&
        values.frozen_internal != true &&
        error("$label distilled cell is not frozen")
    values.dt_ms > 0 ||
        error("$label dt_ms must be positive")
    values.cycles > 0 ||
        error("$label cycles must be positive")
    values.substeps_per_cycle > 0 ||
        error("$label substeps_per_cycle must be positive")
    expected_window =
        Float64(values.dt_ms) *
        Int(values.cycles) *
        Int(values.substeps_per_cycle)
    isapprox(
        Float64(values.decision_window_ms),
        expected_window;
        rtol=0,
        atol=eps(expected_window) * 4,
    ) || error("$label decision_window_ms is inconsistent")
    for name in (
        :teacher_hash,
        :digital_twin_hash,
        :distilled_artifact_hash,
        :internal_parameter_sha256,
        :cell_mechanism_sha256,
        :morphology_sha256,
    )
        value = lowercase(strip(String(getproperty(values, name))))
        occursin(r"^[0-9a-f]{64}$", value) ||
            error("$label $name is not a SHA-256 value")
    end
    return values
end

function _verify_model_provenance!(trainer, provenance, label)
    model = trainer.model
    Int(provenance.cycles) == Int(getproperty(model, :cycles)) ||
        error("$label cycles differ from current model")
    Int(provenance.substeps_per_cycle) ==
        Int(getproperty(model, :substeps_per_cycle)) ||
        error("$label substeps_per_cycle differs from current model")
    return provenance
end

function _verify_aux_snapshot!(snapshot, label)
    for name in REQUIRED_AUX_FIELDS
        _mapping_value(snapshot, name) === nothing &&
            error("$label has no required auxiliary field $name")
    end
    return snapshot
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

function _restore_scalar_state!(destination, state, label)
    for (raw_name, value) in state
        name = Symbol(raw_name)
        hasproperty(destination, name) ||
            error("$label no longer has field $name")
        old_value = getproperty(destination, name)
        setproperty!(
            destination,
            name,
            convert(typeof(old_value), value),
        )
    end
    return destination
end

function _refresh_cache_if_supported!(trainer)
    isdefined(Training, :refresh_paper_cache!) ||
        return trainer
    refresh! = getfield(Training, :refresh_paper_cache!)
    applicable(refresh!, trainer) ||
        error("refresh_paper_cache! has no trainer method")
    refresh!(trainer)
    return trainer
end

function _current_internal_audit(trainer)
    return (;
        artifact_sha256=lowercase(String(
            Training.paper_internal_sha256(trainer),
        )),
        parameter_sha256=lowercase(String(
            Training.paper_internal_parameter_sha256(trainer),
        )),
        max_delta=Float64(
            Training.paper_internal_max_delta(trainer),
        ),
    )
end

function _verify_internal!(
    trainer,
    provenance,
    label::AbstractString,
)
    audit = _current_internal_audit(trainer)
    if provenance.cell_mode == "distilled-frozen"
        audit.artifact_sha256 ==
            lowercase(String(
                provenance.distilled_artifact_hash,
            )) || error("$label artifact hash differs")
        audit.parameter_sha256 ==
            lowercase(String(
                provenance.internal_parameter_sha256,
            )) || error("$label in-memory parameter hash differs")
    end
    audit.max_delta == 0.0 ||
        error("$label changed frozen internal parameters")
    return audit
end

function save_checkpoint(
    path::AbstractString,
    trainer,
    sampler,
    run_config;
    update::Integer=trainer.optimizer.step,
)
    optimizer = trainer.optimizer
    Int(update) == Int(optimizer.step) ||
        error("checkpoint update and optimizer step differ")
    provenance = _provenance(run_config, "run_config")
    _verify_model_provenance!(
        trainer,
        provenance,
        "run_config",
    )
    internal = _verify_internal!(
        trainer,
        provenance,
        "checkpoint save",
    )
    aux_snapshot = _copy_tree(
        Training.paper_aux_snapshot(trainer),
    )
    _verify_aux_snapshot!(
        aux_snapshot,
        "checkpoint auxiliary snapshot",
    )
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
        provenance,
        model_family=provenance.model_family,
        cell_mode=provenance.cell_mode,
        teacher_hash=provenance.teacher_hash,
        digital_twin_hash=provenance.digital_twin_hash,
        distilled_artifact_hash=
            provenance.distilled_artifact_hash,
        internal_parameter_sha256=
            provenance.internal_parameter_sha256,
        frozen_internal=provenance.frozen_internal,
        cell_mechanism_sha256=
            provenance.cell_mechanism_sha256,
        morphology_sha256=provenance.morphology_sha256,
        dt_ms=provenance.dt_ms,
        substeps_per_cycle=provenance.substeps_per_cycle,
        cycles=provenance.cycles,
        decision_window_ms=provenance.decision_window_ms,
        ablation_mode=provenance.ablation_mode,
        loaded_internal_artifact_sha256=
            internal.artifact_sha256,
        loaded_internal_parameter_sha256=
            internal.parameter_sha256,
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
        aux_snapshot,
        sampler_snapshot=Core.sampler_snapshot(sampler),
    )
    destination = Core.atomic_jldsave(path; payload)
    return (;
        path=destination,
        sha256=checkpoint_sha256(destination),
        update=Int(update),
        schema=CHECKPOINT_SCHEMA,
        internal_artifact_sha256=
            internal.artifact_sha256,
        internal_parameter_sha256=
            internal.parameter_sha256,
    )
end

function load_checkpoint(path::AbstractString)
    source = abspath(path)
    isfile(source) || error("checkpoint is absent: $source")
    data = JLD2.load(source)
    haskey(data, "payload") ||
        error("checkpoint has no payload")
    payload = data["payload"]
    hasproperty(payload, :checkpoint_schema) &&
        payload.checkpoint_schema == CHECKPOINT_SCHEMA ||
        error("unsupported HD-SWSNN-TwinProp checkpoint schema")
    return payload
end

function restore_checkpoint!(
    trainer,
    payload,
    training_rows;
    current_run_config,
)
    payload.model_signature == _model_signature(trainer.model) ||
        error("checkpoint model signature differs")
    payload.arena_signature == _arena_signature(trainer) ||
        error("checkpoint arena signature differs")

    saved = _provenance(
        payload.run_config,
        "checkpoint run_config",
    )
    payload.provenance == saved ||
        error("checkpoint provenance copy differs")
    top_level = _provenance(payload, "checkpoint payload")
    top_level == saved ||
        error("checkpoint top-level provenance differs")
    current = _provenance(
        current_run_config,
        "current run_config",
    )
    _verify_model_provenance!(
        trainer,
        current,
        "current run_config",
    )
    saved == current ||
        error(
            "checkpoint provenance differs from current trainer/run config",
        )
    internal_before = _verify_internal!(
        trainer,
        current,
        "checkpoint restore preflight",
    )
    internal_before.artifact_sha256 ==
        lowercase(String(
            payload.loaded_internal_artifact_sha256,
        )) || error("checkpoint loaded artifact hash differs")
    internal_before.parameter_sha256 ==
        lowercase(String(
            payload.loaded_internal_parameter_sha256,
        )) || error("checkpoint loaded parameter hash differs")

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
    _restore_scalar_state!(
        trainer.optimizer,
        payload.optimizer_state,
        "optimizer",
    )
    trainer.optimizer.step == Int(payload.update) ||
        error("checkpoint update and optimizer step differ")
    _restore_scalar_state!(
        trainer,
        payload.trainer_scalar_state,
        "trainer",
    )
    _verify_aux_snapshot!(
        payload.aux_snapshot,
        "checkpoint auxiliary snapshot",
    )
    Training.restore_paper_aux_snapshot!(
        trainer,
        _copy_tree(payload.aux_snapshot),
    )
    restored_aux = Training.paper_aux_snapshot(trainer)
    _tree_equal(restored_aux, payload.aux_snapshot) ||
        error("restored auxiliary state differs from checkpoint")
    _refresh_cache_if_supported!(trainer)
    internal_after = _verify_internal!(
        trainer,
        current,
        "checkpoint restore completion",
    )
    internal_after == internal_before ||
        error("checkpoint restore mutated frozen internal parameters")
    return Core.restore_sampler(
        training_rows,
        payload.sampler_snapshot,
    )
end

end # module
