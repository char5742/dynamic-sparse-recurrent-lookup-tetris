using Dates
using JLD2
using JSON3
using LinearAlgebra
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ReducedHayDirectTraining.jl"))
include(joinpath(@__DIR__, "BudgetMatchedPointSNN.jl"))
include(joinpath(@__DIR__, "BudgetMatchedGRU.jl"))

using .BeatFirstTrainingCore
using .ReducedHayWorkspaceSNN
using .ReducedHayDirectTraining
using .BudgetMatchedPointSNN
using .BudgetMatchedGRU

const Arena = Main.ArenaWorkspaceTraining
const DEFAULT_DATASET =
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const DEFAULT_OUTPUT_ROOT =
    raw"D:\tetris-paper-plus\runs\reduced_hay_direct_validation"
const VALIDATION_SCHEMA =
    "reduced-hay-budget-validation-v4"

function _parse(arguments)
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        startswith(arguments[index], "--") ||
            error("unexpected argument $(arguments[index])")
        index < length(arguments) ||
            error("missing value for $(arguments[index])")
        name = arguments[index][3:end]
        haskey(values, name) &&
            error("option repeated: --$name")
        values[name] = arguments[index + 1]
        index += 2
    end
    output_dir = get(values, "output-dir", "")
    resume_dir = strip(get(values, "resume-dir", ""))
    resume_update = parse(Int, get(values, "resume-update", "0"))
    arms = Symbol.(
        split(get(values, "arms", "point,reduced,gru"), ','),
    )
    updates = parse(Int, get(values, "updates", "64"))
    milestone_text = strip(get(
        values,
        "evaluation-milestones",
        "",
    ))
    evaluation_milestones = if isempty(milestone_text)
        Int[updates]
    else
        sort!(unique!(parse.(Int, split(milestone_text, ','))))
    end
    updates in evaluation_milestones ||
        push!(evaluation_milestones, updates)
    sort!(evaluation_milestones)
    return (;
        dataset=abspath(get(values, "dataset", DEFAULT_DATASET)),
        output_root=abspath(get(
            values,
            "output-root",
            DEFAULT_OUTPUT_ROOT,
        )),
        output_dir=isempty(output_dir) ?
            nothing : abspath(output_dir),
        resume_dir=isempty(resume_dir) ?
            nothing : abspath(resume_dir),
        resume_update,
        arms,
        updates,
        evaluation_milestones,
        width=parse(Int, get(values, "width", "40")),
        validation_states=parse(
            Int,
            get(values, "validation-states", "16"),
        ),
        learning_rate=parse(
            Float32,
            get(values, "learning-rate", "0.0002"),
        ),
        weight_decay=parse(
            Float32,
            get(values, "weight-decay", "0.00001"),
        ),
        model_seed=parse(Int, get(values, "model-seed", "20260730")),
        schedule_seed=parse(
            Int,
            get(values, "schedule-seed", "20260731"),
        ),
        panel_seed=parse(Int, get(values, "panel-seed", "20260732")),
    )
end

function _validate(options)
    isdir(options.dataset) ||
        error("teacher dataset is absent: $(options.dataset)")
    options.updates > 0 || error("updates must be positive")
    if options.resume_dir === nothing
        options.resume_update == 0 ||
            error("resume-update requires resume-dir")
    else
        isdir(options.resume_dir) ||
            error("resume directory is absent: $(options.resume_dir)")
        0 < options.resume_update < options.updates ||
            error("resume-update must lie within 1:(updates-1)")
    end
    all(
        milestone ->
            options.resume_update < milestone <= options.updates,
        options.evaluation_milestones,
    ) || error(
        "evaluation milestones must follow resume-update and not exceed updates",
    )
    last(options.evaluation_milestones) == options.updates ||
        error("final update must be an evaluation milestone")
    options.width > 0 || error("width must be positive")
    options.validation_states > 0 ||
        error("validation states must be positive")
    options.learning_rate > 0.0f0 ||
        error("learning rate must be positive")
    options.weight_decay >= 0.0f0 ||
        error("weight decay must be nonnegative")
    isempty(options.arms) && error("at least one arm is required")
    all(arm in (:point, :reduced, :gru) for arm in options.arms) ||
        error("reference runner supports point,reduced,gru")
    length(unique(options.arms)) == length(options.arms) ||
        error("arm names must be unique")
    Threads.nthreads(:default) == 1 ||
        error("reference validation requires exactly one Julia thread")
    return options
end

function _reserve_output(options)
    if options.output_dir !== nothing
        ispath(options.output_dir) &&
            error("output directory already exists: $(options.output_dir)")
        mkpath(options.output_dir)
        return options.output_dir
    end
    stamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    path = joinpath(
        options.output_root,
        "budget_validation_$(stamp)_u$(options.updates)",
    )
    ispath(path) && error("output directory already exists: $path")
    mkpath(path)
    return path
end

function _source_revision()
    try
        revision = strip(readchomp(`git rev-parse HEAD`))
        dirty = !isempty(strip(readchomp(
            `git status --porcelain --untracked-files=no`,
        )))
        return dirty ? revision * "-dirty" : revision
    catch
        return "unknown"
    end
end

function _eligible_rows(dataset, split::Symbol, width::Int)
    rows = Int[
        index
        for index in eachindex(dataset.action_counts)
        if dataset.predefined_split[index] === split &&
           dataset.action_counts[index] <= width
    ]
    isempty(rows) &&
        error("no $split rows fit candidate width $width")
    return rows
end

function _schedule(
    source::Vector{Int},
    count::Int,
    seed::Int,
)
    rng = Xoshiro(UInt64(seed))
    permutation = copy(source)
    result = Vector{Int}(undef, count)
    cursor = 1
    for index in 1:count
        if cursor == 1
            shuffle!(rng, permutation)
        end
        result[index] = permutation[cursor]
        cursor += 1
        cursor > length(permutation) && (cursor = 1)
    end
    return result
end

function _arm(name::Symbol)
    name === :point && return (
        model=build_budget_point_snn(),
        raw_function=budget_point_raw,
        activity=false,
    )
    name === :reduced && return (
        model=build_reduced_hay_model(:tiny),
        raw_function=reduced_hay_raw,
        activity=true,
    )
    name === :gru && return (
        model=DiagonalGRUBaseline(),
        raw_function=budget_gru_raw,
        activity=false,
    )
    error("unsupported arm $name")
end

function _parameter_count(parameters)
    return sum(length, values(parameters); init=0)
end

function _ndcg(prediction, teacher)
    count = length(teacher)
    teacher_order = sortperm(teacher; rev=true, alg=MergeSort)
    relevance = zeros(Float64, count)
    for (rank, action) in enumerate(teacher_order)
        relevance[action] = count - rank
    end
    prediction_order =
        sortperm(prediction; rev=true, alg=MergeSort)
    dcg = 0.0
    idcg = 0.0
    for rank in 1:count
        discount = inv(log2(rank + 1.0))
        dcg += relevance[prediction_order[rank]] * discount
        idcg += (count - rank) * discount
    end
    return iszero(idcg) ? 1.0 : dcg / idcg
end

function _pairwise(prediction, teacher)
    correct = 0
    compared = 0
    for left in 1:(length(teacher) - 1)
        for right in (left + 1):length(teacher)
            teacher_difference = teacher[left] - teacher[right]
            iszero(teacher_difference) && continue
            prediction_difference =
                prediction[left] - prediction[right]
            correct +=
                prediction_difference * teacher_difference > 0
            compared += 1
        end
    end
    return iszero(compared) ? 1.0 : correct / compared
end

function _effective_rank(values::AbstractMatrix)
    size(values, 2) <= 1 && return 1.0
    centered = Float64.(values)
    centered .-= mean(centered; dims=2)
    singular = svdvals(centered)
    energy = sum(abs2, singular)
    iszero(energy) && return 0.0
    entropy = 0.0
    for value in singular
        probability = value * value / energy
        probability > 0.0 &&
            (entropy -= probability * log(probability))
    end
    return exp(entropy)
end

function _reduced_activity(
    trainer,
    parameters,
    ;
    ablation::Symbol=:none,
)
    arena = trainer.arena
    valid = Int.(arena.valid_flats[1:arena.valid_count])
    rails = @view arena.rails[:, valid]
    dynamics = reduced_hay_dynamics(
        trainer.model,
        rails,
        parameters,
        ;
        _ablation_scales(ablation)...,
    )
    branch = reshape(
        dynamics.branch_voltage,
        :,
        length(valid),
    )
    return (;
        active_spike_rate=Float64(dynamics.active_spike_rate),
        soma_spike_rate=Float64(dynamics.soma_spike_rate),
        plateau_mean=Float64(mean(dynamics.plateau)),
        plateau_active_fraction=Float64(mean(
            dynamics.plateau .> 0.05f0,
        )),
        nmda_mean=Float64(mean(dynamics.nmda)),
        branch_effective_rank=_effective_rank(branch),
    )
end

function _evaluate!(
    trainer,
    dataset,
    rows::Vector{Int};
    parameters=trainer.parameters,
    collect_activity::Bool=false,
    ablation::Symbol=:none,
)
    composite = 0.0
    q_huber = 0.0
    listnet = 0.0
    top1 = 0
    ndcg = 0.0
    pairwise = 0.0
    activity_sum = Dict(
        :active_spike_rate => 0.0,
        :soma_spike_rate => 0.0,
        :plateau_mean => 0.0,
        :plateau_active_fraction => 0.0,
        :nmda_mean => 0.0,
        :branch_effective_rank => 0.0,
    )
    for row in rows
        pack_rows!(trainer, dataset, [row])
        raw = if ablation === :none
            trainer.raw_function(
                trainer.model,
                trainer.arena.rails,
                parameters,
            )
        else
            reduced_hay_raw(
                trainer.model,
                trainer.arena.rails,
                parameters;
                _ablation_scales(ablation)...,
            )
        end
        copyto!(trainer.arena.raw, raw)
        loss = Arena.loss_and_raw_gradient!(
            trainer.arena,
            trainer.loss_scratch,
            0.5f0,
            0.0f0,
        )
        composite += Float64(loss.composite_loss)
        q_huber += Float64(loss.q_huber_loss)
        listnet += Float64(loss.listnet_loss)
        count = Int(trainer.arena.counts[1])
        prediction = @view raw[1, 1:count]
        teacher = @view dataset.teacher_q[1:count, row]
        top1 += argmax(prediction) == argmax(teacher)
        ndcg += _ndcg(prediction, teacher)
        pairwise += _pairwise(prediction, teacher)
        if collect_activity
            activity = _reduced_activity(
                trainer,
                parameters;
                ablation,
            )
            for name in keys(activity_sum)
                activity_sum[name] += getproperty(activity, name)
            end
        end
    end
    inverse = inv(Float64(length(rows)))
    metrics = (;
        states=length(rows),
        composite_loss=composite * inverse,
        listnet_loss=listnet * inverse,
        q_huber_loss=q_huber * inverse,
        top1=top1 * inverse,
        ndcg=ndcg * inverse,
        pairwise=pairwise * inverse,
    )
    activity = collect_activity ? NamedTuple{Tuple(keys(activity_sum))}(
        Tuple(value * inverse for value in values(activity_sum)),
    ) : nothing
    return (; metrics, activity)
end

function _ablation_scales(mode::Symbol)
    mode === :none && return (
        plateau_scale=1.0f0,
        apical_scale=1.0f0,
        recurrent_scale=1.0f0,
    )
    mode === :plateau_off && return (
        plateau_scale=0.0f0,
        apical_scale=1.0f0,
        recurrent_scale=1.0f0,
    )
    mode === :apical_off && return (
        plateau_scale=1.0f0,
        apical_scale=0.0f0,
        recurrent_scale=1.0f0,
    )
    mode === :recurrent_off && return (
        plateau_scale=1.0f0,
        apical_scale=1.0f0,
        recurrent_scale=0.0f0,
    )
    error("unknown ablation $mode")
end

function _warmup!(
    arm,
    dataset,
    row::Int,
    options,
)
    trainer = CanonicalDirectTrainer(
        arm.model,
        arm.raw_function;
        rng=MersenneTwister(options.model_seed),
        state_batch=1,
        width=options.width,
        learning_rate=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    pack_rows!(trainer, dataset, [row])
    direct_update!(trainer)
    _evaluate!(
        trainer,
        dataset,
        [row];
        collect_activity=arm.activity,
    )
    return nothing
end

function _restore_checkpoint!(
    trainer,
    name::Symbol,
    train_schedule::Vector{Int},
    validation_rows::Vector{Int},
    options,
)
    options.resume_dir === nothing && return nothing
    checkpoint_path = joinpath(
        options.resume_dir,
        "checkpoint_$(String(name))_u$(options.resume_update).jld2",
    )
    isfile(checkpoint_path) ||
        error("resume checkpoint is absent: $checkpoint_path")
    checkpoint = JLD2.load(checkpoint_path)
    for field in (
        "parameters",
        "optimizer_state",
        "update",
        "model_seed",
        "schedule_seed",
        "train_schedule",
        "validation_rows",
        "arm",
    )
        haskey(checkpoint, field) ||
            error("resume checkpoint lacks $field: $checkpoint_path")
    end
    Int(checkpoint["update"]) == options.resume_update ||
        error("resume update mismatch: $checkpoint_path")
    Int(checkpoint["model_seed"]) == options.model_seed ||
        error("resume model seed mismatch: $checkpoint_path")
    Int(checkpoint["schedule_seed"]) == options.schedule_seed ||
        error("resume schedule seed mismatch: $checkpoint_path")
    String(checkpoint["arm"]) == String(name) ||
        error("resume arm mismatch: $checkpoint_path")
    saved_schedule = Int.(checkpoint["train_schedule"])
    length(saved_schedule) >= options.resume_update ||
        error("resume schedule is too short: $checkpoint_path")
    saved_schedule[1:options.resume_update] ==
        train_schedule[1:options.resume_update] ||
        error("resume training schedule prefix mismatch: $checkpoint_path")
    Int.(checkpoint["validation_rows"]) == validation_rows ||
        error("resume validation panel mismatch: $checkpoint_path")
    trainer.parameters = checkpoint["parameters"]
    trainer.optimizer_state = checkpoint["optimizer_state"]
    trainer.updates = options.resume_update
    return checkpoint_path
end

function _train_arm!(
    name::Symbol,
    dataset,
    train_schedule::Vector{Int},
    validation_rows::Vector{Int},
    options,
    output_dir::AbstractString,
)
    arm = _arm(name)
    _warmup!(arm, dataset, first(train_schedule), options)
    trainer = CanonicalDirectTrainer(
        arm.model,
        arm.raw_function;
        rng=MersenneTwister(options.model_seed),
        state_batch=1,
        width=options.width,
        learning_rate=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    resume_checkpoint = _restore_checkpoint!(
        trainer,
        name,
        train_schedule,
        validation_rows,
        options,
    )
    before_timed = @timed _evaluate!(
        trainer,
        dataset,
        validation_rows;
        collect_activity=arm.activity,
    )
    before = before_timed.value
    start_update = options.resume_update
    learning_curve = Dict{String,Any}(
        string(start_update) => before,
    )
    evaluation_wall_seconds = Dict{String,Float64}(
        string(start_update) => before_timed.time,
    )
    milestone_checkpoints = Dict{String,String}()
    continuation_updates = length(train_schedule) - start_update
    loss_trace = Vector{Float64}(undef, continuation_updates)
    gradient_trace = similar(loss_trace)
    GC.gc()
    maximum_rss_before = Sys.maxrss()
    training_wall_seconds = 0.0
    training_allocation_bytes = 0
    training_gc_seconds = 0.0
    segment_start = start_update + 1
    for milestone in options.evaluation_milestones
        timed = @timed begin
            for update in segment_start:milestone
                row = train_schedule[update]
                pack_rows!(trainer, dataset, [row])
                loss = direct_update!(trainer)
                trace_index = update - start_update
                loss_trace[trace_index] =
                    Float64(loss.composite_loss)
                gradient_trace[trace_index] =
                    trainer.last_gradient_norm
                if update == start_update + 1 ||
                   update == length(train_schedule) ||
                   update % max(div(length(train_schedule), 4), 1) == 0
                    @printf(
                        "arm=%s update=%d/%d loss=%.6f gradient=%.6f\n",
                        String(name),
                        update,
                        length(train_schedule),
                        loss.composite_loss,
                        trainer.last_gradient_norm,
                    )
                    flush(stdout)
                end
            end
        end
        training_wall_seconds += timed.time
        training_allocation_bytes += timed.bytes
        training_gc_seconds += timed.gctime
        evaluation_timed = @timed _evaluate!(
            trainer,
            dataset,
            validation_rows;
            collect_activity=arm.activity,
        )
        evaluation = evaluation_timed.value
        milestone_key = string(milestone)
        learning_curve[milestone_key] = evaluation
        evaluation_wall_seconds[milestone_key] =
            evaluation_timed.time
        checkpoint_path = joinpath(
            output_dir,
            "checkpoint_$(String(name))_u$(milestone).jld2",
        )
        JLD2.jldsave(
            checkpoint_path;
            parameters=trainer.parameters,
            optimizer_state=trainer.optimizer_state,
            update=milestone,
            model_seed=options.model_seed,
            schedule_seed=options.schedule_seed,
            train_schedule,
            validation_rows,
            arm=String(name),
        )
        milestone_checkpoints[milestone_key] = checkpoint_path
        @printf(
            "held arm=%s update=%d loss=%.6f top1=%.6f ndcg=%.6f pairwise=%.6f\n",
            String(name),
            milestone,
            evaluation.metrics.composite_loss,
            evaluation.metrics.top1,
            evaluation.metrics.ndcg,
            evaluation.metrics.pairwise,
        )
        flush(stdout)
        segment_start = milestone + 1
    end
    maximum_rss_after = Sys.maxrss()
    after = learning_curve[string(length(train_schedule))]
    ablations = nothing
    if name === :reduced
        ablations = Dict{String,Any}()
        for mode in (:plateau_off, :apical_off, :recurrent_off)
            result = _evaluate!(
                trainer,
                dataset,
                validation_rows;
                collect_activity=true,
                ablation=mode,
            )
            ablations[String(mode)] = result
        end
    end
    result = Dict{String,Any}(
        "status" => "complete",
        "parameter_count" =>
            _parameter_count(trainer.parameters),
        "resumed_from" => resume_checkpoint,
        "before" => before,
        "after" => after,
        "learning_curve" => learning_curve,
        "evaluation_wall_seconds" => evaluation_wall_seconds,
        "milestone_checkpoints" => milestone_checkpoints,
        "ablations" => ablations,
        "training" => Dict(
            "updates" => continuation_updates,
            "update_start" => start_update + 1,
            "update_end" => length(train_schedule),
            "loss_first" => first(loss_trace),
            "loss_last" => last(loss_trace),
            "loss_mean" => mean(loss_trace),
            "gradient_first" => first(gradient_trace),
            "gradient_last" => last(gradient_trace),
            "wall_seconds" => training_wall_seconds,
            "updates_per_second" =>
                continuation_updates / training_wall_seconds,
            "allocation_bytes" => training_allocation_bytes,
            "allocation_bytes_per_update" =>
                training_allocation_bytes / continuation_updates,
            "gc_seconds" => training_gc_seconds,
            "maxrss_before" => maximum_rss_before,
            "maxrss_after" => maximum_rss_after,
            "loss_trace" => loss_trace,
            "gradient_trace" => gradient_trace,
        ),
    )
    return (; trainer, result)
end

function _write_json(path::AbstractString, payload)
    temporary = path * ".tmp"
    open(temporary, "w") do io
        JSON3.pretty(io, payload)
        println(io)
    end
    mv(temporary, path; force=true)
    return path
end

function main(arguments=ARGS)
    options = _validate(_parse(arguments))
    LinearAlgebra.BLAS.set_num_threads(1)
    dataset = load_teacher_dataset(options.dataset)
    train_pool = _eligible_rows(dataset, :train, options.width)
    validation_pool =
        _eligible_rows(dataset, :validation, options.width)
    train_schedule = _schedule(
        train_pool,
        options.updates,
        options.schedule_seed,
    )
    validation_rows = _schedule(
        validation_pool,
        options.validation_states,
        options.panel_seed,
    )
    output_dir = _reserve_output(options)
    results = Dict{String,Any}(
        "schema" => VALIDATION_SCHEMA,
        "created_at" => string(now()),
        "revision" => _source_revision(),
        "dataset" => options.dataset,
        "options" => Dict(
            "arms" => String.(options.arms),
            "updates" => options.updates,
            "resume_dir" => options.resume_dir,
            "resume_update" => options.resume_update,
            "evaluation_milestones" =>
                options.evaluation_milestones,
            "width" => options.width,
            "validation_states" => options.validation_states,
            "learning_rate" => options.learning_rate,
            "weight_decay" => options.weight_decay,
            "model_seed" => options.model_seed,
            "schedule_seed" => options.schedule_seed,
            "panel_seed" => options.panel_seed,
            "julia_threads" => Threads.nthreads(:default),
            "blas_threads" => BLAS.get_num_threads(),
        ),
        "train_schedule" => train_schedule,
        "validation_rows" => validation_rows,
        "frozen_distilled_11_state" => Dict(
            "status" => "unavailable",
            "reason" =>
                "no qualified frozen 11-state artifact is present",
        ),
        "arms" => Dict{String,Any}(),
    )
    results_path = joinpath(output_dir, "results.json")
    _write_json(results_path, results)
    for name in options.arms
        @info "starting validation arm" name
        run = _train_arm!(
            name,
            dataset,
            train_schedule,
            validation_rows,
            options,
            output_dir,
        )
        results["arms"][String(name)] = run.result
        checkpoint_path =
            joinpath(output_dir, "checkpoint_$(String(name)).jld2")
        JLD2.jldsave(
            checkpoint_path;
            parameters=run.trainer.parameters,
            optimizer_state=run.trainer.optimizer_state,
            update=options.updates,
            model_seed=options.model_seed,
            schedule_seed=options.schedule_seed,
            train_schedule,
            validation_rows,
            arm=String(name),
        )
        run.result["checkpoint"] = checkpoint_path
        _write_json(results_path, results)
    end
    println("RESULTS\t", results_path)
    return (; output_dir, results_path, results)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
