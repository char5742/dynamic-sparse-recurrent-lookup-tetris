# Critical-path full-model recovery for the development-scale Official ELM.
#
# This is intentionally a separate recovery lineage.  It reads an immutable
# parent checkpoint, resets Adam and the cosine schedule, then updates both the
# reservoir and the six-row output head.  Only fit and derived-validation
# targets are opened.  No output is ever written under the original 3 x 35 run.
#
# The 16-update preflight is the prefix of the 140-update run:
#   --updates 16  --cosine-t-max 140 --resume true
# followed, in the same recovery run root, by:
#   --updates 140 --cosine-t-max 140 --resume true

include(joinpath(
    @__DIR__,
    "train_paper_elm_twin_official_full.jl",
))

module TrainPaperELMReservoirWarmstartCritical

using Dates
using JLD2
using JSON3
using LinearAlgebra
using Optimisers
using Random
using SHA
using Statistics
using Zygote

const Full = Main.TrainPaperELMTwinOfficialFull
const Development = Full.Development
const Twin = Full.Twin
const Sealed = Full.Sealed

const PROJECT_ROOT = dirname(dirname(dirname(@__DIR__)))
const DEFAULT_DATASET =
    raw"C:\tmp\hd_swsnn_neuron_teacher_final_dev1500_release"
const DEFAULT_PARENT_CHECKPOINT = joinpath(
    PROJECT_ROOT,
    "runs",
    "paper_elm_official_dev1500",
    "paper_elm_dev1500_3x35_20260729t1053z",
    "checkpoints",
    "restart_3",
    "epoch_031.jld2",
)
const DEFAULT_RUN_ROOT = joinpath(
    PROJECT_ROOT,
    "runs",
    "paper_elm_official_dev1500",
    "reservoir_warmstart_r3e31",
)
const DEFAULT_SEED = UInt64(0x52534552564f4952)
const DEFAULT_UPDATES = 16
const DEFAULT_COSINE_T_MAX = 140
const DEFAULT_VALIDATION_EVERY = 16
const DEFAULT_LEARNING_RATE = 5.0f-5
const HEAD_NAMES = (:output_weight, :output_bias)

struct Options
    dataset::String
    parent_checkpoint::String
    run_root::String
    run_id::String
    updates::Int
    cosine_t_max::Int
    validation_every::Int
    batch_size::Int
    seed::UInt64
    learning_rate::Float32
    blas_threads::Int
    positive_aware::Bool
    resume::Bool
end

function _is_within(path::AbstractString, root::AbstractString)
    relative = relpath(abspath(path), abspath(root))
    return relative == "." || !(
        relative == ".." ||
        startswith(relative, "..\\") ||
        startswith(relative, "../")
    )
end

function Options(;
    dataset::AbstractString=DEFAULT_DATASET,
    parent_checkpoint::AbstractString=DEFAULT_PARENT_CHECKPOINT,
    run_root::AbstractString=DEFAULT_RUN_ROOT,
    run_id::AbstractString="reservoir-warmstart-r3e31",
    updates::Integer=DEFAULT_UPDATES,
    cosine_t_max::Integer=DEFAULT_COSINE_T_MAX,
    validation_every::Integer=DEFAULT_VALIDATION_EVERY,
    batch_size::Integer=8,
    seed::Integer=DEFAULT_SEED,
    learning_rate::Real=DEFAULT_LEARNING_RATE,
    blas_threads::Integer=20,
    positive_aware::Bool=true,
    resume::Bool=true,
)
    updates >= 1 || throw(ArgumentError("updates must be positive"))
    cosine_t_max >= updates || throw(ArgumentError(
        "cosine_t_max must cover the requested total updates",
    ))
    validation_every >= 1 || throw(ArgumentError(
        "validation_every must be positive",
    ))
    batch_size == 8 || throw(ArgumentError(
        "canonical development batch_size must be 8",
    ))
    learning_rate > 0 || throw(ArgumentError(
        "learning_rate must be positive",
    ))
    blas_threads >= 1 || throw(ArgumentError(
        "blas_threads must be positive",
    ))
    isempty(run_id) && throw(ArgumentError("run_id must not be empty"))

    resolved_parent = abspath(String(parent_checkpoint))
    resolved_run_root = abspath(String(run_root))
    parent_evidence_root =
        dirname(dirname(dirname(resolved_parent)))
    _is_within(resolved_run_root, parent_evidence_root) &&
        throw(ArgumentError(
            "recovery run_root must be outside the immutable parent run",
        ))
    return Options(
        abspath(String(dataset)),
        resolved_parent,
        resolved_run_root,
        String(run_id),
        Int(updates),
        Int(cosine_t_max),
        Int(validation_every),
        Int(batch_size),
        UInt64(seed),
        Float32(learning_rate),
        Int(blas_threads),
        positive_aware,
        resume,
    )
end

@inline _sha256_file(path) = bytes2hex(SHA.sha256(read(path)))

function _parameter_subset(parameters, include_head::Bool)
    names = Tuple(
        name for name in propertynames(parameters)
        if (name in HEAD_NAMES) == include_head
    )
    return NamedTuple{names}(Tuple(
        getproperty(parameters, name) for name in names
    ))
end

_reservoir_sha256(parameters) =
    Twin.official_parameter_sha256(
        _parameter_subset(parameters, false),
    )

_head_sha256(parameters) =
    Twin.official_parameter_sha256(
        _parameter_subset(parameters, true),
    )

function _append_event(path, value)
    mkpath(dirname(path))
    open(path, "a") do io
        JSON3.write(io, value)
        write(io, '\n')
        flush(io)
    end
    return nothing
end

_event_path(options) =
    joinpath(options.run_root, "events.jsonl")

_checkpoint_dir(options) =
    joinpath(options.run_root, "checkpoints")

function _checkpoint_path(options, update)
    return joinpath(
        _checkpoint_dir(options),
        "update_$(lpad(update, 6, '0')).jld2",
    )
end

function _latest_checkpoint(options)
    directory = _checkpoint_dir(options)
    isdir(directory) || return nothing
    candidates = Tuple{Int,String}[]
    for path in readdir(directory; join=true)
        matched = match(r"^update_(\d{6})\.jld2$", basename(path))
        matched === nothing && continue
        update = parse(Int, only(matched.captures))
        update <= options.updates || continue
        push!(candidates, (update, path))
    end
    isempty(candidates) && return nothing
    sort!(candidates; by=first)
    return last(candidates)[2]
end

function _verify_parent!(checkpoint, options, dataset)
    String(checkpoint["artifact_kind"]) ==
        "PaperELMTwinOfficialFullCheckpoint" ||
        error("parent is not an Official full-training checkpoint")
    Int(checkpoint["format_version"]) == 1 ||
        error("parent checkpoint format differs")
    String(checkpoint["manifest_sha256"]) ==
        dataset.manifest_sha256 ||
        error("parent manifest digest differs")
    String(checkpoint["teacher_contract_sha256"]) ==
        dataset.teacher_contract_sha256 ||
        error("parent teacher contract digest differs")
    actual =
        Twin.official_parameter_sha256(checkpoint["parameters"])
    actual == String(checkpoint["parameter_sha256"]) ||
        error("parent parameter digest differs")
    return nothing
end

function _verify_recovery_checkpoint!(
    checkpoint,
    options,
    lineage,
)
    String(checkpoint["artifact_kind"]) ==
        "PaperELMFullWarmstartRecoveryCheckpoint" ||
        error("recovery checkpoint artifact kind differs")
    Int(checkpoint["format_version"]) == 1 ||
        error("recovery checkpoint format differs")
    String(checkpoint["run_id"]) == options.run_id ||
        error("recovery checkpoint run_id differs")
    String(checkpoint["parent_checkpoint_sha256"]) ==
        lineage.parent_checkpoint_sha256 ||
        error("recovery parent checkpoint digest differs")
    String(checkpoint["parent_parameter_sha256"]) ==
        lineage.parent_parameter_sha256 ||
        error("recovery parent parameter digest differs")
    String(checkpoint["parent_reservoir_sha256"]) ==
        lineage.parent_reservoir_sha256 ||
        error("recovery parent reservoir digest differs")
    String(checkpoint["manifest_sha256"]) ==
        lineage.manifest_sha256 ||
        error("recovery manifest digest differs")
    String(checkpoint["teacher_contract_sha256"]) ==
        lineage.teacher_contract_sha256 ||
        error("recovery teacher contract digest differs")
    UInt64(checkpoint["seed"]) == options.seed ||
        error("recovery seed differs")
    Int(checkpoint["cosine_t_max"]) == options.cosine_t_max ||
        error("recovery cosine horizon differs")
    Float32(checkpoint["base_learning_rate"]) ==
        options.learning_rate ||
        error("recovery base learning rate differs")
    actual =
        Twin.official_parameter_sha256(checkpoint["parameters"])
    actual == String(checkpoint["parameter_sha256"]) ||
        error("recovery parameter digest differs")
    Int(checkpoint["update_index"]) <= options.updates ||
        error("recovery checkpoint is beyond requested target")
    Bool(checkpoint["heldout_targets_opened"]) == false ||
        error("recovery checkpoint claims heldout target access")
    return nothing
end

function _save_checkpoint(
    options,
    lineage,
    update_index,
    model,
    parameters,
    optimizer_state,
    normalizer,
    rng,
    metrics,
)
    path = _checkpoint_path(options, update_index)
    mkpath(dirname(path))
    temporary = path * ".partial"
    parameter_sha256 =
        Twin.official_parameter_sha256(parameters)
    reservoir_sha256 = _reservoir_sha256(parameters)
    head_sha256 = _head_sha256(parameters)
    JLD2.jldsave(
        temporary;
        artifact_kind="PaperELMFullWarmstartRecoveryCheckpoint",
        format_version=1,
        run_id=options.run_id,
        update_index,
        target_updates=options.updates,
        cosine_t_max=options.cosine_t_max,
        base_learning_rate=options.learning_rate,
        seed=options.seed,
        model,
        parameters,
        optimizer_state,
        normalizer,
        rng,
        validation_metrics=metrics,
        validation_completed=metrics !== nothing,
        parameter_sha256,
        reservoir_sha256,
        head_sha256,
        parent_checkpoint=options.parent_checkpoint,
        parent_checkpoint_sha256=
            lineage.parent_checkpoint_sha256,
        parent_parameter_sha256=
            lineage.parent_parameter_sha256,
        parent_reservoir_sha256=
            lineage.parent_reservoir_sha256,
        parent_head_sha256=lineage.parent_head_sha256,
        manifest_sha256=lineage.manifest_sha256,
        teacher_contract_sha256=
            lineage.teacher_contract_sha256,
        optimizer_reset_at_parent=true,
        cosine_schedule_reset_at_parent=true,
        full_reservoir_and_head_update=true,
        separate_from_3x35_training_evidence=true,
        heldout_targets_opened=false,
    )
    mv(temporary, path; force=true)
    return (;
        path,
        checkpoint_file_sha256=_sha256_file(path),
        parameter_sha256,
        reservoir_sha256,
        head_sha256,
    )
end

function _positive_window_catalog(dataset)
    cache = Dict{Int,Any}()
    catalog = Dict{Int,Vector{Int}}()
    for source_id in dataset.fit_ids
        id = Int(source_id)
        record_index, _, item =
            Development._record_for_id(dataset, id)
        data =
            Development._numeric!(cache, dataset, record_index)
        Sealed._validate_numeric!(data)
        target = @view data["target_spike"][:, item]
        prefix = zeros(Int, length(target) + 1)
        @inbounds for index in eachindex(target)
            (target[index] == 0 || target[index] == 1) ||
                error("fit spike target is not binary")
            prefix[index + 1] =
                prefix[index] + Int(target[index])
        end
        starts = Int[]
        for start in (
            Development.FIRST_RANDOM_START:
            Development.LAST_RANDOM_START
        )
            stop = start + Development.TRAIN_WINDOW - 1
            prefix[stop + 1] - prefix[start] > 0 &&
                push!(starts, start)
        end
        catalog[id] = starts
    end
    any(!isempty, values(catalog)) ||
        error("fit split has no spike-positive training window")
    return catalog
end

function _sample_batch_spec(
    rng,
    dataset,
    positive_catalog,
    options,
)
    order = randperm(rng, length(dataset.fit_ids))
    ids = Int.(
        dataset.fit_ids[order[1:options.batch_size]],
    )
    starts = rand(
        rng,
        (
            Development.FIRST_RANDOM_START:
            Development.LAST_RANDOM_START
        ),
        options.batch_size,
    )
    anchor_slot = 0
    if options.positive_aware
        eligible_slots = [
            slot for slot in eachindex(ids)
            if !isempty(positive_catalog[ids[slot]])
        ]
        if isempty(eligible_slots)
            positive_ids = [
                id for id in Int.(dataset.fit_ids)
                if !isempty(positive_catalog[id])
            ]
            replacement_pool =
                setdiff(positive_ids, ids)
            isempty(replacement_pool) &&
                error("cannot choose a distinct positive anchor")
            anchor_slot = rand(rng, eachindex(ids))
            ids[anchor_slot] = rand(rng, replacement_pool)
        else
            anchor_slot = rand(rng, eligible_slots)
        end
        choices = positive_catalog[ids[anchor_slot]]
        starts[anchor_slot] = rand(rng, choices)
    end
    return ids, starts, anchor_slot
end

@inline function _bce_with_logits(logit, target)
    return max(logit, zero(logit)) -
           logit * target +
           log1p(exp(-abs(logit)))
end

function _balanced_objective(
    model,
    parameters,
    normalizer,
    batch,
)
    prediction = Twin.Core.official_elm_forward(
        model,
        parameters,
        batch.input,
    )
    target_voltage =
        Twin.preprocess_soma_voltage(batch.target_voltage)
    voltage_mse =
        mean(abs2, prediction.voltage .- target_voltage)

    spike_loss = _bce_with_logits.(
        prediction.spike_logit,
        batch.target_spike,
    )
    positives = sum(batch.target_spike)
    negatives =
        Float32(length(batch.target_spike)) - positives
    positives > 0 || error(
        "positive-aware batch unexpectedly has no spike",
    )
    negatives > 0 || error(
        "training batch unexpectedly has no negative spike",
    )
    positive_bce =
        sum(spike_loss .* batch.target_spike) / positives
    negative_bce =
        sum(
            spike_loss .* (1.0f0 .- batch.target_spike),
        ) / negatives
    spike_balanced_bce =
        0.5f0 * positive_bce + 0.5f0 * negative_bce

    target_nmda =
        (
            batch.target_nmda .-
            reshape(normalizer.nmda_mean, :, 1, 1)
        ) ./ reshape(normalizer.nmda_scale, :, 1, 1)
    nmda_extension_loss =
        mean(abs2, prediction.nmda .- target_nmda)

    paper_loss =
        0.5f0 * voltage_mse +
        0.5f0 * spike_balanced_bce
    total =
        Development.PAPER_LOSS_WEIGHT * paper_loss +
        Development.NMDA_EXTENSION_WEIGHT *
        nmda_extension_loss
    return total, (;
        total,
        paper_loss,
        voltage_mse,
        spike_balanced_bce,
        positive_bce,
        negative_bce,
        nmda_extension_loss,
        spike_positives=Int(round(positives)),
        spike_negatives=Int(round(negatives)),
    )
end

mutable struct _Moments
    n::Int
    sum_y::Float64
    sum_y2::Float64
    error2::Float64
end

_Moments() = _Moments(0, 0.0, 0.0, 0.0)

function _update!(moments::_Moments, predicted, target)
    for index in eachindex(predicted, target)
        x = Float64(predicted[index])
        y = Float64(target[index])
        isfinite(x) && isfinite(y) ||
            error("validation observation is non-finite")
        moments.n += 1
        moments.sum_y += y
        moments.sum_y2 += y * y
        difference = x - y
        moments.error2 += difference * difference
    end
    return moments
end

_rmse(moments::_Moments) =
    moments.n > 0 ?
    sqrt(moments.error2 / moments.n) :
    NaN

function _normalized_rmse(moments::_Moments)
    moments.n > 0 || return NaN
    centered =
        moments.sum_y2 -
        moments.sum_y * moments.sum_y / moments.n
    if centered > eps(Float64)
        return sqrt(moments.error2 / centered)
    end
    # These moments are already in fit-normalized coordinates, so their
    # RMSE is exactly raw_RMSE / max(abs(frozen_scale), 1e-5), matching the
    # corrected sealed-V2 zero-variance contract.
    return _rmse(moments)
end

function _exact_auroc(scores, labels)
    length(scores) == length(labels) ||
        throw(DimensionMismatch("AUROC scores and labels differ"))
    binary = labels .== UInt8(1)
    positives = count(binary)
    negatives = length(binary) - positives
    positives > 0 && negatives > 0 ||
        error("validation AUROC requires both spike classes")
    order = sortperm(scores; alg=MergeSort)
    rank_sum = 0.0
    first_index = 1
    while first_index <= length(order)
        last_index = first_index
        while last_index < length(order) &&
              scores[order[last_index + 1]] ==
              scores[order[first_index]]
            last_index += 1
        end
        average_rank = (first_index + last_index) / 2
        @inbounds for position in first_index:last_index
            binary[order[position]] &&
                (rank_sum += average_rank)
        end
        first_index = last_index + 1
    end
    return (
        rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)
end

function _validation_metrics(
    dataset,
    model,
    parameters,
    normalizer,
)
    voltage = _Moments()
    nmda = [_Moments() for _ in 1:Development.NMDA_REGIONS]
    spike_scores = Float32[]
    spike_labels = UInt8[]
    cache = Dict{Int,Any}()
    seen = Int[]
    window_evaluations = 0

    for source_id in dataset.validation_ids
        id = Int(source_id)
        record_index, _, item =
            Development._record_for_id(dataset, id)
        data =
            Development._numeric!(cache, dataset, record_index)
        Sealed._validate_numeric!(data)
        steps = size(data["target_voltage"], 1)
        steps == 1_500 ||
            error("validation trial must contain 1500 bins")
        push!(seen, id)
        for (window_index, start_step) in
            enumerate(Sealed._paper_window_starts(steps))
            input, actual_steps = Sealed._paper_window_input(
                data,
                item,
                start_step,
                steps,
            )
            prediction = Twin.Core.official_elm_forward(
                model,
                parameters,
                input;
                initial_state=nothing,
            )
            window_evaluations += 1
            local_keep_first =
                window_index == 1 ?
                1 :
                Sealed.PAPER_EVALUATION_OVERLAP_STEPS + 1
            local_keep_first <= actual_steps || continue
            global_keep_first =
                start_step + local_keep_first - 1
            global_keep_last =
                start_step + actual_steps - 1
            metric_global_first =
                max(global_keep_first, 501)
            metric_global_first > global_keep_last && continue
            local_metric_first =
                local_keep_first +
                metric_global_first - global_keep_first
            local_range = local_metric_first:actual_steps
            target_range =
                metric_global_first:global_keep_last

            predicted_mv =
                Twin.soma_voltage_from_coordinate(
                    @view(prediction.voltage[
                        local_range,
                        :,
                    ]),
                )
            target_mv = min.(
                @view(
                    data["target_voltage"][
                        target_range,
                        item:item,
                    ]
                ),
                Twin.OFFICIAL_SOMA_CLIP_MV,
            )
            _update!(voltage, predicted_mv, target_mv)

            score = @view prediction.spike_logit[
                local_range,
                :,
            ]
            label = @view data["target_spike"][
                target_range,
                item:item,
            ]
            for index in eachindex(score, label)
                label[index] == 0 || label[index] == 1 ||
                    error("validation spike label is not binary")
                push!(spike_scores, Float32(score[index]))
                push!(spike_labels, UInt8(label[index]))
            end

            for region in 1:Development.NMDA_REGIONS
                target_coordinate =
                    (
                        @view(data["target_nmda"][
                            region,
                            target_range,
                            item:item,
                        ]) .-
                        normalizer.nmda_mean[region]
                    ) ./ normalizer.nmda_scale[region]
                _update!(
                    nmda[region],
                    @view(prediction.nmda[
                        region,
                        local_range,
                        :,
                    ]),
                    target_coordinate,
                )
            end
        end
    end

    sort!(seen)
    seen == sort(Int.(collect(dataset.validation_ids))) ||
        error("derived-validation membership differs")
    expected = length(dataset.validation_ids) * 1_000
    voltage.n == expected ||
        error("validation voltage bin count differs")
    length(spike_scores) == expected ||
        error("validation spike bin count differs")
    all(value -> value.n == expected, nmda) ||
        error("validation NMDA bin count differs")

    voltage_rmse_mv = _rmse(voltage)
    spike_auroc =
        _exact_auroc(spike_scores, spike_labels)
    nmda_rmse =
        [_normalized_rmse(value) for value in nmda]
    all(isfinite, (
        voltage_rmse_mv,
        spike_auroc,
        nmda_rmse...,
    )) || error("validation metrics are non-finite")
    gate_pass =
        voltage_rmse_mv <=
            Sealed.MAXIMUM_VOLTAGE_RMSE_MV &&
        spike_auroc >= Sealed.MINIMUM_SPIKE_AUROC &&
        all(
            value ->
                value <=
                Sealed.MAXIMUM_REGIONAL_NMDA_NORMALIZED_RMSE,
            nmda_rmse,
        )
    return (;
        clipped_physical_voltage_rmse_mv=voltage_rmse_mv,
        voltage_target_clip_max_mv=
            Twin.OFFICIAL_SOMA_CLIP_MV,
        spike_auroc,
        spike_auroc_exact_tie_aware=true,
        nmda_normalized_rmse_by_region=nmda_rmse,
        validation_bins=expected,
        spike_positives=count(==(UInt8(1)), spike_labels),
        spike_negatives=count(==(UInt8(0)), spike_labels),
        evaluation_window_count=window_evaluations,
        evaluation_window_steps=
            Sealed.PAPER_EVALUATION_WINDOW_STEPS,
        evaluation_overlap_steps=
            Sealed.PAPER_EVALUATION_OVERLAP_STEPS,
        evaluation_stride_steps=
            Sealed.PAPER_EVALUATION_STRIDE_STEPS,
        recurrent_state_reset_each_window=true,
        validation_gate_pass=gate_pass,
        minimum_spike_auroc=Sealed.MINIMUM_SPIKE_AUROC,
        maximum_voltage_rmse_mv=
            Sealed.MAXIMUM_VOLTAGE_RMSE_MV,
        maximum_regional_nmda_normalized_rmse=
            Sealed.MAXIMUM_REGIONAL_NMDA_NORMALIZED_RMSE,
    )
end

@inline function _cosine_learning_rate(options, update)
    return options.learning_rate * 0.5f0 * (
        1.0f0 +
        cos(
            Float32(pi) *
            Float32(update - 1) /
            Float32(options.cosine_t_max),
        )
    )
end

function _fresh_state(options, parent_parameters)
    parameters = deepcopy(parent_parameters)
    return (;
        update_index=0,
        parameters,
        optimizer_state=Optimisers.setup(
            Optimisers.Adam(options.learning_rate),
            parameters,
        ),
        rng=Xoshiro(options.seed),
        resumed_from=nothing,
    )
end

function _load_or_initialize(
    options,
    lineage,
    parent_parameters,
)
    latest = options.resume ? _latest_checkpoint(options) : nothing
    latest === nothing &&
        return _fresh_state(options, parent_parameters)
    checkpoint = JLD2.load(latest)
    _verify_recovery_checkpoint!(
        checkpoint,
        options,
        lineage,
    )
    return (;
        update_index=Int(checkpoint["update_index"]),
        parameters=checkpoint["parameters"],
        optimizer_state=checkpoint["optimizer_state"],
        rng=checkpoint["rng"],
        resumed_from=latest,
    )
end

function run(options::Options)
    BLAS.set_num_threads(options.blas_threads)
    started = time()
    mkpath(options.run_root)
    manifest_path = joinpath(
        options.dataset,
        Development.MANIFEST_NAME,
    )
    dataset = Sealed._verify_manifest_and_shards(
        manifest_path,
        options.dataset,
    )
    length(dataset.fit_ids) == 32 ||
        error("development fit split must contain 32 trials")
    length(dataset.validation_ids) == 8 ||
        error("derived-validation split must contain 8 trials")
    length(dataset.heldout_ids) == 8 ||
        error("sealed heldout split inventory differs")

    parent = JLD2.load(options.parent_checkpoint)
    _verify_parent!(parent, options, dataset)
    model = parent["model"]
    parent_parameters = parent["parameters"]
    normalizer = parent["normalizer"]
    lineage = (;
        parent_checkpoint_sha256=
            _sha256_file(options.parent_checkpoint),
        parent_parameter_sha256=
            Twin.official_parameter_sha256(
                parent_parameters,
            ),
        parent_reservoir_sha256=
            _reservoir_sha256(parent_parameters),
        parent_head_sha256=_head_sha256(parent_parameters),
        manifest_sha256=dataset.manifest_sha256,
        teacher_contract_sha256=
            dataset.teacher_contract_sha256,
    )
    positive_catalog =
        _positive_window_catalog(dataset)
    positive_trial_count =
        count(!isempty, values(positive_catalog))
    state = _load_or_initialize(
        options,
        lineage,
        parent_parameters,
    )
    parameters = state.parameters
    optimizer_state = state.optimizer_state
    rng = state.rng
    start_update = state.update_index + 1
    event_path = _event_path(options)

    started_event = (;
        event="reservoir_warmstart_started",
        timestamp=string(now(UTC)),
        pid=getpid(),
        run_id=options.run_id,
        requested_target_updates=options.updates,
        resume_update=state.update_index,
        resumed_from=state.resumed_from,
        parent_checkpoint=options.parent_checkpoint,
        parent_checkpoint_sha256=
            lineage.parent_checkpoint_sha256,
        parent_parameter_sha256=
            lineage.parent_parameter_sha256,
        parent_reservoir_sha256=
            lineage.parent_reservoir_sha256,
        parent_head_sha256=lineage.parent_head_sha256,
        initial_parameter_sha256=
            Twin.official_parameter_sha256(parameters),
        initial_reservoir_sha256=
            _reservoir_sha256(parameters),
        initial_head_sha256=_head_sha256(parameters),
        parent_restart_index=Int(parent["restart_index"]),
        parent_epoch=Int(parent["epoch"]),
        parent_update_index=Int(parent["update_index"]),
        optimizer_reset_at_parent=state.update_index == 0,
        cosine_schedule_reset_at_parent=true,
        cosine_t_max=options.cosine_t_max,
        base_learning_rate=options.learning_rate,
        full_reservoir_and_head_update=true,
        positive_aware_window_sampling=
            options.positive_aware,
        class_balanced_spike_bce=true,
        voltage_loss_preserved=true,
        nmda_loss_preserved=true,
        fit_trials=length(dataset.fit_ids),
        validation_trials=length(dataset.validation_ids),
        positive_capable_fit_trials=positive_trial_count,
        heldout_inventory_count=length(dataset.heldout_ids),
        heldout_targets_opened=false,
        separate_from_3x35_training_evidence=true,
        validation_every=options.validation_every,
        julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
    )
    _append_event(event_path, started_event)
    println(JSON3.write(started_event))
    flush(stdout)

    for update in start_update:options.updates
        update_started = time()
        ids, starts, anchor_slot = _sample_batch_spec(
            rng,
            dataset,
            positive_catalog,
            options,
        )
        batch = Development._materialize_batch(
            dataset,
            ids,
            starts,
        )
        learning_rate =
            _cosine_learning_rate(options, update)
        Optimisers.adjust!(
            optimizer_state,
            learning_rate,
        )
        objective, gradients =
            Zygote.withgradient(parameters) do candidate
                _balanced_objective(
                    model,
                    candidate,
                    normalizer,
                    batch,
                )
            end
        loss, components = objective
        isfinite(loss) ||
            error("non-finite loss at recovery update $update")
        gradient = only(gradients)
        reservoir_gradient =
            _parameter_subset(gradient, false)
        head_gradient =
            _parameter_subset(gradient, true)
        reservoir_norm2, reservoir_finite, _ =
            Development._gradient_stats(
                reservoir_gradient,
            )
        head_norm2, head_finite, _ =
            Development._gradient_stats(head_gradient)
        reservoir_finite && head_finite ||
            error("non-finite recovery gradient at update $update")
        reservoir_norm2 > 0 ||
            error("zero reservoir gradient at update $update")
        head_norm2 > 0 ||
            error("zero head gradient at update $update")
        optimizer_state, parameters =
            Optimisers.update(
                optimizer_state,
                parameters,
                gradient,
            )
        update_event = (;
            event="recovery_update_completed",
            timestamp=string(now(UTC)),
            run_id=options.run_id,
            update,
            target_updates=options.updates,
            learning_rate,
            loss,
            components,
            reservoir_gradient_norm=
                sqrt(reservoir_norm2),
            head_gradient_norm=sqrt(head_norm2),
            sample_ids=ids,
            crop_starts=starts,
            positive_anchor_slot=anchor_slot,
            elapsed_seconds=time() - update_started,
            heldout_targets_opened=false,
        )
        _append_event(event_path, update_event)
        println(JSON3.write(update_event))
        flush(stdout)

        if update % options.validation_every == 0 ||
           update == options.updates
            presaved = _save_checkpoint(
                options,
                lineage,
                update,
                model,
                parameters,
                optimizer_state,
                normalizer,
                rng,
                nothing,
            )
            prevalidation_event = (;
                event="recovery_prevalidation_checkpoint_saved",
                timestamp=string(now(UTC)),
                run_id=options.run_id,
                update,
                checkpoint_file=replace(
                    relpath(presaved.path, options.run_root),
                    '\\' => '/',
                ),
                checkpoint_file_sha256=
                    presaved.checkpoint_file_sha256,
                parameter_sha256=presaved.parameter_sha256,
                reservoir_sha256=presaved.reservoir_sha256,
                head_sha256=presaved.head_sha256,
                validation_completed=false,
                heldout_targets_opened=false,
            )
            _append_event(event_path, prevalidation_event)
            println(JSON3.write(prevalidation_event))
            flush(stdout)
            metrics = _validation_metrics(
                dataset,
                model,
                parameters,
                normalizer,
            )
            saved = _save_checkpoint(
                options,
                lineage,
                update,
                model,
                parameters,
                optimizer_state,
                normalizer,
                rng,
                metrics,
            )
            reservoir_changed =
                saved.reservoir_sha256 !=
                lineage.parent_reservoir_sha256
            head_changed =
                saved.head_sha256 !=
                lineage.parent_head_sha256
            reservoir_changed ||
                error("recovery did not change reservoir parameters")
            head_changed ||
                error("recovery did not change head parameters")
            validation_event = (;
                event="recovery_validation_completed",
                timestamp=string(now(UTC)),
                run_id=options.run_id,
                update,
                target_updates=options.updates,
                metrics,
                checkpoint_file=replace(
                    relpath(saved.path, options.run_root),
                    '\\' => '/',
                ),
                checkpoint_file_sha256=
                    saved.checkpoint_file_sha256,
                parent_checkpoint_sha256=
                    lineage.parent_checkpoint_sha256,
                parent_parameter_sha256=
                    lineage.parent_parameter_sha256,
                parent_reservoir_sha256=
                    lineage.parent_reservoir_sha256,
                parameter_sha256=saved.parameter_sha256,
                reservoir_sha256=saved.reservoir_sha256,
                head_sha256=saved.head_sha256,
                reservoir_changed,
                head_changed,
                heldout_targets_opened=false,
            )
            _append_event(event_path, validation_event)
            println(JSON3.write(validation_event))
            flush(stdout)
        end
    end

    completed = (;
        event="reservoir_warmstart_completed",
        timestamp=string(now(UTC)),
        run_id=options.run_id,
        update_index=options.updates,
        cosine_t_max=options.cosine_t_max,
        elapsed_seconds=time() - started,
        parent_checkpoint_sha256=
            lineage.parent_checkpoint_sha256,
        parent_parameter_sha256=
            lineage.parent_parameter_sha256,
        final_parameter_sha256=
            Twin.official_parameter_sha256(parameters),
        final_reservoir_sha256=
            _reservoir_sha256(parameters),
        final_head_sha256=_head_sha256(parameters),
        heldout_targets_opened=false,
        separate_from_3x35_training_evidence=true,
    )
    _append_event(event_path, completed)
    println(JSON3.write(completed))
    flush(stdout)
    return completed
end

function run_minimal_unit()
    _exact_auroc(
        Float32[0, 1, 2, 3],
        UInt8[0, 0, 1, 1],
    ) == 1.0 || error("perfect AUROC unit failed")
    _exact_auroc(
        zeros(Float32, 4),
        UInt8[0, 1, 0, 1],
    ) == 0.5 || error("tied AUROC unit failed")
    zero_variance = _Moments()
    _update!(
        zero_variance,
        Float64[0.5, 1.5],
        Float64[0, 0],
    )
    isapprox(
        _normalized_rmse(zero_variance),
        sqrt(5.0) / 2.0;
        rtol=0,
        atol=1e-15,
    ) || error("zero-variance NMDA contract unit failed")
    parameters = (
        reservoir=Float32[1, 2],
        output_weight=ones(Float32, 1, 2),
        output_bias=zeros(Float32, 1),
    )
    propertynames(_parameter_subset(parameters, false)) ==
        (:reservoir,) ||
        error("reservoir partition unit failed")
    propertynames(_parameter_subset(parameters, true)) ==
        HEAD_NAMES ||
        error("head partition unit failed")
    options = Options(
        parent_checkpoint=DEFAULT_PARENT_CHECKPOINT,
        run_root=DEFAULT_RUN_ROOT * "_unit",
        updates=16,
        cosine_t_max=140,
    )
    _cosine_learning_rate(options, 1) ==
        options.learning_rate ||
        error("cosine reset unit failed")
    return true
end

function _parse_bool(value)
    normalized = lowercase(String(value))
    normalized in ("true", "1", "yes") && return true
    normalized in ("false", "0", "no") && return false
    throw(ArgumentError("invalid boolean: $value"))
end

function _parse(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        startswith(arguments[index], "--") ||
            error("unexpected positional argument: $(arguments[index])")
        index < length(arguments) ||
            error("missing value for $(arguments[index])")
        values[arguments[index][3:end]] =
            arguments[index + 1]
        index += 2
    end
    return Options(
        dataset=get(values, "dataset", DEFAULT_DATASET),
        parent_checkpoint=get(
            values,
            "parent-checkpoint",
            DEFAULT_PARENT_CHECKPOINT,
        ),
        run_root=get(
            values,
            "run-root",
            DEFAULT_RUN_ROOT,
        ),
        run_id=get(
            values,
            "run-id",
            "reservoir-warmstart-r3e31",
        ),
        updates=parse(
            Int,
            get(values, "updates", string(DEFAULT_UPDATES)),
        ),
        cosine_t_max=parse(
            Int,
            get(
                values,
                "cosine-t-max",
                string(DEFAULT_COSINE_T_MAX),
            ),
        ),
        validation_every=parse(
            Int,
            get(
                values,
                "validation-every",
                string(DEFAULT_VALIDATION_EVERY),
            ),
        ),
        batch_size=parse(
            Int,
            get(values, "batch-size", "8"),
        ),
        seed=parse(
            UInt64,
            get(values, "seed", string(DEFAULT_SEED)),
        ),
        learning_rate=parse(
            Float32,
            get(
                values,
                "learning-rate",
                string(DEFAULT_LEARNING_RATE),
            ),
        ),
        blas_threads=parse(
            Int,
            get(values, "blas-threads", "20"),
        ),
        positive_aware=_parse_bool(
            get(values, "positive-aware", "true"),
        ),
        resume=_parse_bool(
            get(values, "resume", "true"),
        ),
    )
end

main(arguments=ARGS) = run(_parse(arguments))

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    TrainPaperELMReservoirWarmstartCritical.main(ARGS)
end
