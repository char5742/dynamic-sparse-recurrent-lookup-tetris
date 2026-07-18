using Dates
using JLD2
using JSON3
using Lux
using Optimisers
using Random
using SHA
using Statistics
using Zygote

const TRAINING_DIR = @__DIR__
const EXPERIMENT_DIR = normpath(joinpath(TRAINING_DIR, ".."))

include(joinpath(TRAINING_DIR, "core.jl"))
include(joinpath(TRAINING_DIR, "native_backend.jl"))
include(joinpath(TRAINING_DIR, "metrics_resume.jl"))
using .BeatFirstTrainingCore
using .BeatFirstMetricsResume

const MODEL_SOURCE = abspath(get(
    ENV, "BEAT_MODEL_SOURCE", joinpath(EXPERIMENT_DIR, "models", "models.jl"),
))
isfile(MODEL_SOURCE) || error("BEAT_MODEL_SOURCE does not exist: $MODEL_SOURCE")
include(MODEL_SOURCE)
const MODEL_MODULE_NAME = Symbol(get(ENV, "BEAT_MODEL_MODULE", "BeatFirstModels"))
isdefined(Main, MODEL_MODULE_NAME) || error("model module $MODEL_MODULE_NAME was not defined")
const MODEL_API = getfield(Main, MODEL_MODULE_NAME)

const BACKEND_KIND = lowercase(get(ENV, "BEAT_BACKEND", "reactant"))
const BACKEND_API = if BACKEND_KIND in ("native", "zygote", "cpu")
    BeatFirstNativeBackend
else
    source = abspath(get(
        ENV,
        "BEAT_BACKEND_SOURCE",
        joinpath(EXPERIMENT_DIR, "backend", "fixedshape_learner.jl"),
    ))
    isfile(source) || error("BEAT_BACKEND_SOURCE does not exist: $source")
    include(source)
    module_name = Symbol(get(ENV, "BEAT_BACKEND_MODULE", "BeatFirstFixedShapeBackend"))
    isdefined(Main, module_name) || error("backend module $module_name was not defined")
    getfield(Main, module_name)
end

# Exactly the two surviving architectures, each at the model registry's frozen
# Small and Medium presets. ConvNeXt-only and ad-hoc sizes are intentionally not
# accepted by this convergence driver.
const CONVERGENCE_VARIANTS = (
    :preact_eca,
    :preact_eca_medium,
    :gravity_film,
    :gravity_film_medium,
)

sha256_file(path) = bytes2hex(open(sha256, path))
sha256_text(value::AbstractString) = bytes2hex(sha256(codeunits(value)))

function source_closure_sha256(paths)
    normalized = sort(abspath.(collect(paths)))
    all(isfile, normalized) || error("source closure contains a missing file")
    payload = join(
        (replace(path, '\\' => '/') * "\0" * sha256_file(path) for path in normalized),
        "\n",
    )
    return sha256_text(payload)
end

function _git_metadata()
    repository_root = normpath(joinpath(EXPERIMENT_DIR, "..", ".."))
    commit = readchomp(`git -C $repository_root rev-parse HEAD`)
    tracked_status = strip(readchomp(
        `git -C $repository_root status --porcelain --untracked-files=no`,
    ))
    isempty(tracked_status) || error(
        "supervised training requires a committed tracked worktree; status: $tracked_status",
    )
    return (; commit, tracked_worktree_clean=true)
end

const CHECKPOINT_FORMAT_VERSION = 2
const RESUME_CONFIG_FIELDS = (
    :experiment_id,
    :variant,
    :family,
    :preset,
    :model_config,
    :total_parameter_count,
    :trainable_parameter_count,
    :dataset_path,
    :dataset_sha256,
    :dataset_states,
    :dataset_part_integrity_verified,
    :dataset_verified_part_count,
    :storage_candidate_width,
    :candidate_width,
    :observed_candidate_width,
    :split_kind,
    :split_seed,
    :training_groups,
    :validation_groups,
    :training_states,
    :validation_states,
    :training_eval_states,
    :validation_eval_states,
    :partial_dataset_allowed,
    :model_source_sha256,
    :model_source_closure_sha256,
    :training_source_sha256,
    :training_core_source_sha256,
    :metrics_resume_source_sha256,
    :backend,
    :backend_source_sha256,
    :backend_device,
    :state_batch,
    :n_quantiles,
    :learning_rate,
    :weight_decay,
    :seed,
    :evaluation_interval,
    :min_epochs,
    :max_epochs,
    :minimum_updates,
    :maximum_updates,
    :patience,
    :gradient_evaluation_interval,
    :stability_top1_delta,
    :stability_loss_ratio,
    :git_commit,
    :git_tracked_worktree_clean,
    :julia_version,
    :lux_version,
    :optimisers_version,
    :zygote_version,
    :manifest_sha256,
)

function _read_resume_checkpoint(path::AbstractString)
    isfile(path) || error("BEAT_RESUME_CHECKPOINT does not exist: $path")
    return jldopen(path, "r") do file
        required = (
            "checkpoint_format_version", "ps", "st", "optimizer_state",
            "train_state_step", "backend_updates", "trainer_state",
            "sampler_state", "control_state", "config", "metrics", "status",
        )
        missing = filter(name -> !haskey(file, name), required)
        isempty(missing) || error(
            "resume checkpoint is missing required fields: $(join(missing, ", "))",
        )
        Int(file["checkpoint_format_version"]) == CHECKPOINT_FORMAT_VERSION || error(
            "unsupported resume checkpoint format $(file["checkpoint_format_version"])",
        )
        return (;
            path=abspath(path),
            ps=file["ps"],
            st=file["st"],
            optimizer_state=file["optimizer_state"],
            train_state_step=Int(file["train_state_step"]),
            backend_updates=Int(file["backend_updates"]),
            trainer_state=file["trainer_state"],
            sampler_state=file["sampler_state"],
            control_state=file["control_state"],
            config=file["config"],
            metrics=file["metrics"],
            status=file["status"],
        )
    end
end

function _validate_resume_config!(saved, current)
    for name in RESUME_CONFIG_FIELDS
        hasproperty(saved, name) || error("resume config is missing $name")
        hasproperty(current, name) || error("current config is missing $name")
        isequal(getproperty(saved, name), getproperty(current, name)) || error(
            "resume config mismatch for $name: checkpoint=$(repr(getproperty(saved, name))) " *
            "current=$(repr(getproperty(current, name)))",
        )
    end
    return nothing
end

mutable struct ConvergenceRun
    variant::Symbol
    model
    learner
    meta
    update::Int
    packed_states::Int
    training_wall_seconds::Float64
    history::Vector{Any}
    last_gradient_diagnostics
    first_update_wall_seconds
    backend_recompile_count::Int
end

function _host_checkpoint(run::ConvergenceRun)
    raw = BACKEND_API.host_checkpoint(run.learner)
    train_state_step = Int(raw.step)
    backend_updates = hasproperty(raw, :backend_updates) ?
                      Int(raw.backend_updates) : train_state_step
    return (;
        ps=hasproperty(raw, :ps) ? raw.ps : raw.parameters,
        st=hasproperty(raw, :st) ? raw.st : raw.states,
        optimizer_state=raw.optimizer_state,
        train_state_step,
        backend_updates,
    )
end

function _parameter_arrays(value)
    arrays = Any[]
    function visit(item)
        if item isa NamedTuple || item isa Tuple
            foreach(visit, values(item))
        elseif item isa AbstractArray
            push!(arrays, item)
        end
    end
    visit(value)
    return arrays
end

parameter_count(value) = sum(length, _parameter_arrays(value); init=0)
parameter_norm(value) = sqrt(sum(
    sum(abs2, Float64.(Array(array))) for array in _parameter_arrays(value);
    init=0.0,
))

function _gradient_leaves(parameters, gradients)
    records = NamedTuple[]
    function visit(path::String, parameter, gradient)
        if parameter isa NamedTuple
            for key in keys(parameter)
                child_gradient = gradient === nothing ? nothing : getproperty(gradient, key)
                child_path = isempty(path) ? String(key) : "$path.$key"
                visit(child_path, getproperty(parameter, key), child_gradient)
            end
        elseif parameter isa Tuple
            for index in eachindex(parameter)
                child_gradient = gradient === nothing ? nothing : gradient[index]
                visit("$path.$index", parameter[index], child_gradient)
            end
        elseif parameter isa AbstractArray
            norm = gradient isa AbstractArray ?
                sqrt(sum(abs2, Float64.(Array(gradient)))) : 0.0
            push!(records, (; path, parameters=length(parameter), gradient_norm=norm))
        end
    end
    visit("", parameters, gradients)
    return records
end

function _prefix_gradient_norm(leaves, prefix::AbstractString)
    return sqrt(sum(
        record.gradient_norm^2 for record in leaves if startswith(record.path, prefix);
        init=0.0,
    ))
end

function _gradient_diagnostics(run::ConvergenceRun, checkpoint, batch; witness::Bool=false)
    gradient = only(Zygote.gradient(
        ps -> first(supervised_objective(run.model, ps, checkpoint.st, batch)),
        checkpoint.ps,
    ))
    leaves = _gradient_leaves(checkpoint.ps, gradient)
    global_norm = sqrt(sum(record.gradient_norm^2 for record in leaves; init=0.0))
    result = (;
        global_gradient_norm=global_norm,
        parameter_norm=parameter_norm(checkpoint.ps),
        finite=isfinite(global_norm),
    )
    witness || return result

    depth = Int(run.meta.depth)
    prefixes = run.meta.family === :preact_eca ? (;
        first="stem",
        middle="blocks.layer_$(cld(depth, 2))",
        final="blocks.layer_$depth",
    ) : (;
        first="board_stem",
        middle="blocks.layer_$(cld(depth, 2))",
        final="blocks.layer_$depth",
    )
    witness_for(prefix) = (;
        path=prefix,
        gradient_norm=_prefix_gradient_norm(leaves, prefix),
    )
    paths = (;
        first=witness_for(prefixes.first),
        middle=witness_for(prefixes.middle),
        final=witness_for(prefixes.final),
        queue=witness_for("queue"),
        aux=witness_for("aux"),
        shared_head=witness_for("heads.shared."),
        q_head=witness_for("heads.q."),
        death_head=witness_for("heads.death."),
        quantile_head=witness_for("heads.quantiles."),
        geometry_head=witness_for("heads.geometry."),
    )
    passed = all(item -> isfinite(item.gradient_norm) && item.gradient_norm > 0.0, values(paths))
    passed || error("end-to-end gradient witness failed for $(run.variant): $paths")
    return merge(result, (; backbone_gradient_witness=paths, witness_passed=passed))
end

function _group_split(dataset, seed::UInt64, fraction::Float64)
    if hasproperty(dataset, :predefined_split) &&
       any(split -> split !== :unspecified, dataset.predefined_split)
        allowed = Set((:train, :validation))
        all(split -> split in allowed, dataset.predefined_split) || error(
            "predefined dataset split contains an unknown value",
        )
        training_rows = findall(==(:train), dataset.predefined_split)
        validation_rows = findall(==(:validation), dataset.predefined_split)
        isempty(training_rows) && error("predefined split has no training rows")
        isempty(validation_rows) && error("predefined split has no validation rows")
        training_groups = sort(unique(dataset.split_group_ids[training_rows]))
        validation_groups = sort(unique(dataset.split_group_ids[validation_rows]))
        isempty(intersect(training_groups, validation_groups)) || error(
            "seed leakage across predefined train/validation split",
        )
        return (; training_rows, validation_rows, validation_groups,
                training_groups, predefined=true)
    end
    groups = sort(unique(dataset.split_group_ids))
    length(groups) >= 2 || error("at least two episode/seed groups are required")
    explicit = strip(get(ENV, "BEAT_VALIDATION_GROUPS", ""))
    validation_groups = if isempty(explicit)
        shuffled = copy(groups)
        shuffle!(Xoshiro(seed), shuffled)
        count = clamp(round(Int, length(groups) * fraction), 1, length(groups) - 1)
        sort(shuffled[1:count])
    else
        requested = sort(unique(parse.(Int, split(explicit, ','))))
        all(group -> group in groups, requested) || error("unknown validation group")
        1 <= length(requested) < length(groups) || error("invalid validation group count")
        requested
    end
    validation_set = Set(validation_groups)
    validation_rows = findall(group -> group in validation_set, dataset.split_group_ids)
    training_rows = findall(group -> !(group in validation_set), dataset.split_group_ids)
    isempty(training_rows) && error("empty training split")
    isempty(validation_rows) && error("empty validation split")
    return (; training_rows, validation_rows, validation_groups,
            training_groups=sort(setdiff(groups, validation_groups)), predefined=false)
end

function _fixed_subset(rows, maximum_states::Int, batch_size::Int, seed::UInt64)
    maximum_states >= batch_size || error("evaluation state cap is below fixed batch size")
    selected = copy(rows)
    shuffle!(Xoshiro(seed), selected)
    resize!(selected, min(length(selected), maximum_states))
    usable = length(selected) - mod(length(selected), batch_size)
    usable >= batch_size || error("split is smaller than one fixed evaluation batch")
    resize!(selected, usable)
    return selected
end

function _all_finite(value)
    for item in values(value)
        item isa Number && !isfinite(item) && return false
    end
    return true
end

function _last_two_stable(history; top1_delta::Float64, loss_ratio::Float64)
    length(history) >= 2 || return false
    previous, current = history[end - 1], history[end]
    _all_finite(previous.training) && _all_finite(current.training) &&
        _all_finite(previous.validation) && _all_finite(current.validation) || return false
    isfinite(previous.global_gradient_norm) && isfinite(current.global_gradient_norm) &&
        isfinite(previous.parameter_norm) && isfinite(current.parameter_norm) || return false
    abs(current.validation.top1_agreement - previous.validation.top1_agreement) <= top1_delta ||
        return false
    previous_loss = max(abs(previous.validation.composite_loss), 1.0e-8)
    abs(current.validation.composite_loss - previous.validation.composite_loss) /
        previous_loss <= loss_ratio || return false
    return isfinite(current.validation.q_mean) && isfinite(current.validation.q_std) &&
           isfinite(current.validation.action_margin)
end

function _classification(history, eligible::Bool)
    eligible && return "game_eligible"
    current = last(history)
    gap = current.training.top1_agreement - current.validation.top1_agreement
    gap >= 0.10 && return "overfit"
    current.training.top1_agreement < 0.80 && return "underfit_or_optimization_headroom"
    current.validation.top1_agreement < 0.80 && return "generalization_headroom"
    return "unstable_or_insufficient_epoch"
end

function _evaluate!(
    run::ConvergenceRun,
    dataset,
    host_batch,
    training_eval_rows,
    validation_eval_rows,
    witness_batch,
    training_rows_count::Int,
    witness::Bool,
    compute_gradient::Bool,
    stability_top1_delta::Float64,
    stability_loss_ratio::Float64,
)
    checkpoint = _host_checkpoint(run)
    test_state = Lux.testmode(checkpoint.st)
    predictor = batch -> first(run.model(batch.inputs, checkpoint.ps, test_state))
    training = evaluation_metrics(dataset, training_eval_rows, host_batch, predictor)
    validation = evaluation_metrics(dataset, validation_eval_rows, host_batch, predictor)
    if compute_gradient
        run.last_gradient_diagnostics = merge(
            _gradient_diagnostics(run, checkpoint, witness_batch; witness),
            (; observed_at_update=run.update),
        )
    end
    gradients = run.last_gradient_diagnostics
    gradients === nothing && error("no gradient diagnostic has been recorded")
    elapsed = max(run.training_wall_seconds, eps(Float64))
    record = (;
        update=run.update,
        epoch_equivalent=run.packed_states / training_rows_count,
        updates_per_second=run.update / elapsed,
        training,
        validation,
        global_gradient_norm=gradients.global_gradient_norm,
        parameter_norm=parameter_norm(checkpoint.ps),
        gradients_finite=gradients.finite,
        gradient_observed_at_update=gradients.observed_at_update,
        gradient_fresh=compute_gradient,
        first_update_wall_seconds=run.first_update_wall_seconds,
        backend_recompile_count=run.backend_recompile_count,
        backbone_gradient_witness=witness ? gradients.backbone_gradient_witness : nothing,
        witness_passed=witness ? gradients.witness_passed : true,
    )
    push!(run.history, record)
    stable = _last_two_stable(
        run.history; top1_delta=stability_top1_delta, loss_ratio=stability_loss_ratio,
    )
    finite_q_margin = isfinite(validation.q_mean) && isfinite(validation.q_std) &&
                      isfinite(validation.action_margin)
    return merge(record, (;
        last_two_finite_stable=stable,
        finite_q_and_margin=finite_q_margin,
    )), checkpoint
end

function _save_checkpoint(
    path,
    run,
    checkpoint,
    record,
    config,
    status,
    sampler,
    control_state,
)
    checkpoint.train_state_step == run.update || error(
        "TrainState step $(checkpoint.train_state_step) != trainer update $(run.update)",
    )
    checkpoint.backend_updates == run.update || error(
        "backend updates $(checkpoint.backend_updates) != trainer update $(run.update)",
    )
    sampler_consumed_states(sampler) == run.packed_states || error(
        "sampler position does not match trainer packed state count",
    )
    atomic_jldsave(
        path;
        checkpoint_format_version=CHECKPOINT_FORMAT_VERSION,
        model_kind=String(run.variant),
        ps=checkpoint.ps,
        st=checkpoint.st,
        optimizer_state=checkpoint.optimizer_state,
        train_state_step=checkpoint.train_state_step,
        backend_updates=checkpoint.backend_updates,
        update=run.update,
        metrics=record,
        history=run.history,
        trainer_state=(;
            update=run.update,
            packed_states=run.packed_states,
            training_wall_seconds=run.training_wall_seconds,
            history=run.history,
            last_gradient_diagnostics=run.last_gradient_diagnostics,
            first_update_wall_seconds=run.first_update_wall_seconds,
            backend_recompile_count=run.backend_recompile_count,
        ),
        sampler_state=sampler_snapshot(sampler),
        control_state,
        meta=run.meta,
        config,
        status,
    )
    return (; path=abspath(path), bytes=filesize(path), sha256=sha256_file(path))
end

function _write_json(path, value)
    open(path, "w") do io
        JSON3.pretty(io, value)
    end
end

function main()
    dataset_path = abspath(get(
        ENV, "BEAT_TEACHER_DATASET",
        raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
    ))
    run_root = abspath(get(
        ENV, "BEAT_RUN_ROOT", raw"D:\tetris-paper-plus\runs\beat_first_v1",
    ))
    checkpoint_root = abspath(get(
        ENV, "BEAT_CHECKPOINT_ROOT", raw"D:\tetris-paper-plus\checkpoints\beat_first_v1",
    ))
    mkpath(run_root)
    mkpath(checkpoint_root)

    resume_value = strip(get(ENV, "BEAT_RESUME_CHECKPOINT", ""))
    resume_path = isempty(resume_value) ? nothing : abspath(resume_value)
    resume = resume_path === nothing ? nothing : _read_resume_checkpoint(resume_path)

    variant = Symbol(get(ENV, "BEAT_VARIANT", "preact_eca"))
    variant in CONVERGENCE_VARIANTS || error(
        "BEAT_VARIANT must be one of $(CONVERGENCE_VARIANTS); got $variant",
    )
    state_batch = parse(Int, get(ENV, "BEAT_STATE_BATCH", "4"))
    n_quantiles = parse(Int, get(ENV, "BEAT_N_QUANTILES", "16"))
    learning_rate = parse(Float32, get(ENV, "BEAT_LEARNING_RATE", "3e-4"))
    weight_decay = parse(Float32, get(ENV, "BEAT_WEIGHT_DECAY", "1e-4"))
    seed = parse(UInt64, get(ENV, "BEAT_TRAIN_SEED", "2026071801"))
    split_seed = parse(UInt64, get(ENV, "BEAT_SPLIT_SEED", "2026071817"))
    validation_fraction = parse(Float64, get(ENV, "BEAT_VALIDATION_FRACTION", "0.10"))
    0.0 < validation_fraction < 1.0 || error("validation fraction must be in (0,1)")
    evaluation_interval = parse(Int, get(ENV, "BEAT_EVAL_INTERVAL", "250"))
    100 <= evaluation_interval <= 250 || error("BEAT_EVAL_INTERVAL must be in 100:250")
    min_epochs = parse(Float64, get(ENV, "BEAT_MIN_EPOCHS", "1"))
    max_epochs = parse(Float64, get(ENV, "BEAT_MAX_EPOCHS", "3"))
    1.0 <= min_epochs <= max_epochs || error("require 1 <= min epochs <= max epochs")
    patience = parse(Int, get(ENV, "BEAT_EARLY_STOP_PATIENCE", "3"))
    train_eval_max = parse(Int, get(ENV, "BEAT_TRAIN_EVAL_STATES", "64"))
    validation_eval_max = parse(Int, get(ENV, "BEAT_VAL_EVAL_STATES", "128"))
    stability_top1_delta = parse(Float64, get(ENV, "BEAT_STABLE_TOP1_DELTA", "0.03"))
    stability_loss_ratio = parse(Float64, get(ENV, "BEAT_STABLE_LOSS_RATIO", "0.25"))
    gradient_evaluation_interval = parse(Int, get(ENV, "BEAT_GRAD_EVAL_INTERVAL", "5"))
    gradient_evaluation_interval >= 1 || error("BEAT_GRAD_EVAL_INTERVAL must be positive")
    default_run_id = resume === nothing ?
        "$(variant)_$(Dates.format(now(), "yyyymmdd_HHMMSS"))" :
        String(resume.config.experiment_id)
    run_id = get(ENV, "BEAT_RUN_ID", default_run_id)

    cache_max = parse(Int, get(ENV, "BEAT_GEOMETRY_CACHE_MAX_STATES", "2048"))
    dataset = load_teacher_dataset(dataset_path; geometry_cache_max_states=cache_max)
    storage_candidate_width = size(dataset.teacher_q, 1)
    observed_candidate_width = maximum(dataset.action_counts)
    storage_candidate_width >= observed_candidate_width || error(
        "materialized candidate width is below an observed action count",
    )
    candidate_width = 16 * cld(observed_candidate_width, 16)
    split = _group_split(dataset, split_seed, validation_fraction)
    length(split.training_rows) >= state_batch || error("training split is smaller than batch")
    length(split.validation_rows) >= state_batch || error("validation split is smaller than batch")
    training_eval_rows = _fixed_subset(
        split.training_rows, train_eval_max, state_batch, seed + 0x101,
    )
    validation_eval_rows = _fixed_subset(
        split.validation_rows, validation_eval_max, state_batch, seed + 0x202,
    )

    host_batch = allocate_host_batch(state_batch; max_candidates=candidate_width)
    witness_batch = allocate_host_batch(state_batch; max_candidates=candidate_width)
    pack_batch!(host_batch, dataset, split.training_rows[1:state_batch])
    pack_batch!(witness_batch, dataset, training_eval_rows[1:state_batch])

    setup = MODEL_API.setup_model(variant, Xoshiro(seed); n_quantiles)
    total_count = Int(setup.meta.parameters)
    trainable_count = parameter_count(setup.ps)
    trainable_count == total_count || error(
        "trainable_count=$trainable_count != total_count=$total_count",
    )
    optimiser = Optimisers.AdamW(learning_rate, (0.9, 0.999), weight_decay)
    git = _git_metadata()

    minimum_updates = ceil(Int, min_epochs * length(split.training_rows) / state_batch)
    default_maximum = ceil(Int, max_epochs * length(split.training_rows) / state_batch)
    maximum_updates = parse(Int, get(ENV, "BEAT_MAX_UPDATES", string(default_maximum)))
    maximum_updates >= minimum_updates || error(
        "BEAT_MAX_UPDATES=$maximum_updates is below minimum $minimum_updates",
    )
    manifest = joinpath(dirname(Base.active_project()), "Manifest.toml")
    backend_source_path = BACKEND_KIND in ("native", "zygote", "cpu") ?
        joinpath(TRAINING_DIR, "native_backend.jl") :
        abspath(get(
            ENV,
            "BEAT_BACKEND_SOURCE",
            joinpath(EXPERIMENT_DIR, "backend", "fixedshape_learner.jl"),
        ))
    model_source_closure = (
        MODEL_SOURCE,
        joinpath(dirname(MODEL_SOURCE), "common.jl"),
        joinpath(dirname(MODEL_SOURCE), "preact_eca.jl"),
        joinpath(dirname(MODEL_SOURCE), "gravity_film_convnext.jl"),
        joinpath(dirname(MODEL_SOURCE), "tetris_convnext.jl"),
    )
    training_core_source = joinpath(TRAINING_DIR, "core.jl")
    metrics_resume_source = joinpath(TRAINING_DIR, "metrics_resume.jl")
    config = (;
        experiment_id=run_id,
        variant=String(variant),
        family=String(setup.meta.family),
        preset=String(setup.meta.preset),
        model_config=setup.meta.config,
        total_parameter_count=total_count,
        trainable_parameter_count=trainable_count,
        all_parameters_end_to_end=trainable_count == total_count,
        dataset_path,
        dataset_sha256=isdir(dataset_path) ?
            sha256_file(joinpath(dataset_path, "manifest.json")) : sha256_file(dataset_path),
        dataset_states=length(dataset.action_counts),
        dataset_part_integrity_verified=dataset.part_integrity_verified,
        dataset_verified_part_count=dataset.verified_part_count,
        storage_candidate_width,
        candidate_width,
        observed_candidate_width,
        split_kind=split.predefined ? "manifest_predefined_seed_group" : "episode_or_seed_group",
        split_seed,
        training_groups=split.training_groups,
        validation_groups=split.validation_groups,
        training_states=length(split.training_rows),
        validation_states=length(split.validation_rows),
        training_eval_states=length(training_eval_rows),
        validation_eval_states=length(validation_eval_rows),
        partial_dataset_allowed=dataset.partial_dataset_allowed,
        optional_target_coverage=optional_target_coverage(dataset),
        model_source=MODEL_SOURCE,
        model_source_sha256=sha256_file(MODEL_SOURCE),
        model_source_closure_sha256=source_closure_sha256(model_source_closure),
        training_source=abspath(@__FILE__),
        training_source_sha256=sha256_file(abspath(@__FILE__)),
        training_core_source=training_core_source,
        training_core_source_sha256=sha256_file(training_core_source),
        metrics_resume_source=metrics_resume_source,
        metrics_resume_source_sha256=sha256_file(metrics_resume_source),
        backend=BACKEND_KIND,
        backend_source=backend_source_path,
        backend_source_sha256=sha256_file(backend_source_path),
        backend_device=get(ENV, "BEAT_BACKEND_DEVICE", "cpu"),
        state_batch,
        n_quantiles,
        learning_rate,
        weight_decay,
        seed,
        evaluation_interval,
        min_epochs,
        max_epochs,
        minimum_updates,
        maximum_updates,
        patience,
        gradient_evaluation_interval,
        stability_top1_delta,
        stability_loss_ratio,
        git_commit=git.commit,
        git_tracked_worktree_clean=git.tracked_worktree_clean,
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
        optimisers_version=string(Base.pkgversion(Optimisers)),
        zygote_version=string(Base.pkgversion(Zygote)),
        project=Base.active_project(),
        manifest_sha256=isfile(manifest) ? sha256_file(manifest) : nothing,
        held_out_test_seeds_used=false,
    )

    summary_path = joinpath(run_root, "$run_id.json")
    metrics_path = joinpath(run_root, "$(run_id)_metrics.jsonl")
    latest_path = joinpath(checkpoint_root, "$(run_id)_latest.jld2")
    best_path = joinpath(checkpoint_root, "$(run_id)_best.jld2")
    eligible_best_path = joinpath(checkpoint_root, "$(run_id)_eligible_best.jld2")
    if resume === nothing
        isfile(metrics_path) && error("metrics path already exists: $metrics_path")
    else
        _validate_resume_config!(resume.config, config)
        lowercase(normpath(resume.path)) == lowercase(normpath(latest_path)) || error(
            "resume must use the run's latest checkpoint: expected $latest_path, got $(resume.path)",
        )
        repair_metrics_log!(
            metrics_path,
            (;
                update=Int(resume.trainer_state.update),
                history=resume.trainer_state.history,
                metrics=merge(resume.metrics, resume.status),
            ),
            latest_path;
            artifact_field=:latest_checkpoint,
            allowed_extra_fields=(
                :last_two_finite_stable,
                :finite_q_and_margin,
                :game_eligible,
                :classification,
                :trainable_count,
                :total_count,
                :promotion_rule,
                :latest_checkpoint,
            ),
        )
        BACKEND_KIND in ("native", "zygote", "cpu") && error(
            "durable resume is currently restricted to the pinned Reactant backend",
        )
    end

    backend_restore = resume === nothing ? nothing : (;
        parameters=resume.ps,
        states=resume.st,
        optimizer_state=resume.optimizer_state,
        step=resume.train_state_step,
        backend_updates=resume.backend_updates,
    )
    learner = if BACKEND_KIND in ("native", "zygote", "cpu")
        BACKEND_API.init_backend(
            setup.model, setup.ps, setup.st, optimiser, supervised_objective, host_batch;
            max_candidates=candidate_width,
            backend=get(ENV, "BEAT_BACKEND_DEVICE", "cpu"),
        )
    else
        BACKEND_API.init_backend(
            setup.model, setup.ps, setup.st, optimiser, supervised_objective, host_batch;
            max_candidates=candidate_width,
            backend=get(ENV, "BEAT_BACKEND_DEVICE", "cpu"),
            restore=backend_restore,
        )
    end

    trainer_state = resume === nothing ? nothing : resume.trainer_state
    if trainer_state !== nothing
        required = (
            :update, :packed_states, :training_wall_seconds, :history,
            :last_gradient_diagnostics, :first_update_wall_seconds,
            :backend_recompile_count,
        )
        all(name -> hasproperty(trainer_state, name), required) || error(
            "resume trainer state is incomplete",
        )
    end
    run = if resume === nothing
        ConvergenceRun(
            variant, setup.model, learner, setup.meta, 0, 0, 0.0, Any[], nothing,
            nothing, 0,
        )
    else
        restored = ConvergenceRun(
            variant,
            setup.model,
            learner,
            setup.meta,
            Int(trainer_state.update),
            Int(trainer_state.packed_states),
            Float64(trainer_state.training_wall_seconds),
            Any[item for item in trainer_state.history],
            trainer_state.last_gradient_diagnostics,
            trainer_state.first_update_wall_seconds,
            Int(trainer_state.backend_recompile_count),
        )
        restored.update == resume.train_state_step == resume.backend_updates || error(
            "trainer, TrainState, and backend update counters disagree in checkpoint",
        )
        restored.packed_states == restored.update * state_batch || error(
            "packed state count is inconsistent with update count and batch size",
        )
        isempty(restored.history) && error("resume checkpoint has empty training history")
        Int(last(restored.history).update) == restored.update || error(
            "resume history does not end at the checkpoint update",
        )
        restored
    end
    sampler = resume === nothing ?
        EpochSampler(split.training_rows, Xoshiro(seed + 0x9e3779b97f4a7c15)) :
        restore_sampler(split.training_rows, resume.sampler_state)
    sampler_consumed_states(sampler) == run.packed_states || error(
        "restored sampler position does not match packed state count",
    )

    control = resume === nothing ? nothing : resume.control_state
    if control !== nothing
        required = (
            :best_top1, :best_update, :eligible_best_top1, :eligible_best_update,
            :stale_evaluations, :evaluations_completed,
        )
        all(name -> hasproperty(control, name), required) || error(
            "resume control state is incomplete",
        )
    end
    best_top1 = control === nothing ? -Inf : Float64(control.best_top1)
    best_update = control === nothing ? 0 : Int(control.best_update)
    eligible_best_top1 = control === nothing ? -Inf : Float64(control.eligible_best_top1)
    eligible_best_update = control === nothing ? 0 : Int(control.eligible_best_update)
    stale_evaluations = control === nothing ? 0 : Int(control.stale_evaluations)
    evaluations_completed = control === nothing ? 0 : Int(control.evaluations_completed)
    if resume !== nothing
        run.last_gradient_diagnostics === nothing && error(
            "resume checkpoint has no gradient diagnostics",
        )
        evaluations_completed == length(run.history) - 1 || error(
            "evaluation counter does not match checkpoint history",
        )
        0 <= stale_evaluations <= evaluations_completed || error(
            "stale evaluation counter is outside the valid range",
        )
        0 <= best_update <= run.update || error("best update counter is invalid")
        0 <= eligible_best_update <= run.update || error(
            "eligible-best update counter is invalid",
        )
    end
    control_state() = (;
        best_top1,
        best_update,
        eligible_best_top1,
        eligible_best_update,
        stale_evaluations,
        evaluations_completed,
    )
    resumed_plateau = resume !== nothing &&
                      last(run.history).epoch_equivalent >= min_epochs &&
                      stale_evaluations >= patience
    completion_reason = resumed_plateau ? "validation_plateau" : "maximum_updates"

    if resume !== nothing
        # `latest` is the authoritative commit point. Repair derived best files
        # if interruption occurred after latest but before their atomic writes.
        restored_checkpoint = (;
            ps=resume.ps,
            st=resume.st,
            optimizer_state=resume.optimizer_state,
            train_state_step=resume.train_state_step,
            backend_updates=resume.backend_updates,
        )
        best_update > 0 && best_update == run.update && _save_checkpoint(
            best_path, run, restored_checkpoint, resume.metrics, config, resume.status,
            sampler, control_state(),
        )
        eligible_best_update > 0 && eligible_best_update == run.update && _save_checkpoint(
            eligible_best_path, run, restored_checkpoint, resume.metrics, config,
            resume.status, sampler, control_state(),
        )
    end

    if resume === nothing
        # Update zero supplies the mandatory one-time first/middle/final
        # backbone witness before an optimizer step can hide a frozen path.
        initial, initial_checkpoint = _evaluate!(
            run, dataset, host_batch, training_eval_rows, validation_eval_rows,
            witness_batch, length(split.training_rows), true, true,
            stability_top1_delta, stability_loss_ratio,
        )
        initial_status = (;
            game_eligible=false,
            classification="pretraining_baseline",
            trainable_count,
            total_count,
        )
        initial_latest = _save_checkpoint(
            latest_path, run, initial_checkpoint, initial, config, initial_status,
            sampler, control_state(),
        )
        open(metrics_path, "a") do io
            JSON3.write(io, merge(initial, initial_status, (; latest_checkpoint=initial_latest)))
            write(io, '\n')
        end
    else
        @info "Resumed convergence run" checkpoint=resume.path update=run.update packed_states=run.packed_states evaluations_completed
    end

    while run.update < maximum_updates && !resumed_plateau
        rows = next_batch!(sampler, state_batch)
        pack_started = time()
        pack_batch!(host_batch, dataset, rows)
        pack_seconds = time() - pack_started
        statistics = BACKEND_API.train_step!(run.learner, host_batch)
        loss = Float64(statistics.loss)
        isfinite(loss) || error("non-finite loss at update $(run.update + 1)")
        run.first_update_wall_seconds === nothing &&
            (run.first_update_wall_seconds = Float64(statistics.wall_seconds))
        hasproperty(statistics, :recompiled) && Bool(statistics.recompiled) &&
            (run.backend_recompile_count += 1)
        run.backend_recompile_count == 0 || error(
            "fixed-shape backend recompiled at update $(run.update + 1)",
        )
        hasproperty(statistics, :step) && Int(statistics.step) == run.update + 1 || error(
            "backend step does not match the next trainer update",
        )
        run.update += 1
        run.packed_states += state_batch
        run.training_wall_seconds += Float64(statistics.wall_seconds) + pack_seconds

        evaluate_now = run.update % evaluation_interval == 0 || run.update == maximum_updates
        evaluate_now || continue
        evaluations_completed += 1
        compute_gradient = evaluations_completed % gradient_evaluation_interval == 0 ||
                           run.update == maximum_updates
        record, checkpoint = _evaluate!(
            run, dataset, host_batch, training_eval_rows, validation_eval_rows,
            witness_batch, length(split.training_rows), false, compute_gradient,
            stability_top1_delta, stability_loss_ratio,
        )
        epoch_ok = record.epoch_equivalent >= 1.0
        top1_ok = record.validation.top1_agreement >= 0.80
        eligible = epoch_ok && top1_ok && record.last_two_finite_stable &&
                   record.finite_q_and_margin && record.gradients_finite
        status = (;
            game_eligible=eligible,
            classification=_classification(run.history, eligible),
            trainable_count,
            total_count,
            promotion_rule="epoch>=1 && val_top1>=0.80 && last2_finite_stable && finite_Q_margin",
        )
        new_best = record.validation.top1_agreement > best_top1
        if new_best
            best_top1 = record.validation.top1_agreement
            best_update = run.update
            stale_evaluations = 0
        else
            stale_evaluations += 1
        end
        new_eligible_best = eligible &&
                            record.validation.top1_agreement > eligible_best_top1
        if new_eligible_best
            eligible_best_top1 = record.validation.top1_agreement
            eligible_best_update = run.update
        end
        durable_control = control_state()
        latest = _save_checkpoint(
            latest_path, run, checkpoint, record, config, status, sampler, durable_control,
        )
        new_best && _save_checkpoint(
            best_path, run, checkpoint, record, config, status, sampler, durable_control,
        )
        new_eligible_best && _save_checkpoint(
            eligible_best_path, run, checkpoint, record, config, status, sampler,
            durable_control,
        )
        open(metrics_path, "a") do io
            JSON3.write(io, merge(record, status, (; latest_checkpoint=latest)))
            write(io, '\n')
        end
        @info "Convergence checkpoint" variant update=run.update epoch=record.epoch_equivalent updates_per_second=record.updates_per_second train_top1=record.training.top1_agreement val_top1=record.validation.top1_agreement val_loss=record.validation.composite_loss classification=status.classification game_eligible=eligible

        if record.epoch_equivalent >= min_epochs && stale_evaluations >= patience
            completion_reason = "validation_plateau"
            break
        end
    end

    final_record = last(run.history)
    final_epoch_ok = final_record.epoch_equivalent >= 1.0
    final_top1_ok = final_record.validation.top1_agreement >= 0.80
    final_stable = _last_two_stable(
        run.history; top1_delta=stability_top1_delta, loss_ratio=stability_loss_ratio,
    )
    game_eligible = final_epoch_ok && final_top1_ok && final_stable &&
                    isfinite(final_record.validation.q_mean) &&
                    isfinite(final_record.validation.q_std) &&
                    isfinite(final_record.validation.action_margin) &&
                    final_record.gradients_finite
    summary = (;
        generated_at=string(now()),
        config,
        completion_reason,
        completed_updates=run.update,
        completed_epoch_equivalent=final_record.epoch_equivalent,
        best_validation_top1=best_top1,
        best_update,
        eligible_best_validation_top1=isfinite(eligible_best_top1) ? eligible_best_top1 : nothing,
        eligible_best_update=eligible_best_update == 0 ? nothing : eligible_best_update,
        final=final_record,
        game_eligible,
        classification=_classification(run.history, game_eligible),
        latest_checkpoint=latest_path,
        best_checkpoint=isfile(best_path) ? best_path : nothing,
        eligible_best_checkpoint=isfile(eligible_best_path) ? eligible_best_path : nothing,
        metrics_jsonl=metrics_path,
        next_action=game_eligible ?
            "eligible for fixed-development-game evaluation" :
            "do not run game promotion; inspect underfit/overfit/headroom classification",
    )
    _write_json(summary_path, summary)
    @info "Supervised convergence run complete" summary_path game_eligible classification=summary.classification best_validation_top1=best_top1
    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
