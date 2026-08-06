module CanonicalRelationScratch

using Dates
using JSON3
using LinearAlgebra

const EXPERIMENT_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(EXPERIMENT_ROOT, "training", "core.jl"))
include(joinpath(@__DIR__, "ReducedHayCPU.jl"))
include(joinpath(@__DIR__, "DevelopmentValidationPanel.jl"))
include(joinpath(@__DIR__, "ExperimentData.jl"))

const Root = ReducedHayCPU
const Data = ReducedHayCPUExperimentData
const Ranking = Root.CanonicalRanking
const Parallel = Root.CanonicalBarrierless
const Optimizer = Root.CanonicalOptimizer
const Sampler = Root.CanonicalSampler
const Checkpoint = Root.CanonicalCheckpoint

export Options,
    learning_rate_at,
    main,
    parse_options,
    run_contract,
    usage,
    validate_options

const SAMPLER_SEED = UInt64(0x72656c6174696f6e)
const RUN_CONTRACT_SCHEMA = "hd-relation-graph-full-data-v1"
const PROGRESS_FILE = "progress.jsonl"
const CHECKPOINT_DIRECTORY = "checkpoints"
const LATEST_POINTER = "latest_checkpoint.txt"

Base.@kwdef struct Options
    mode::Symbol = :unset
    dataset::String = Data.DEFAULT_DATASET
    run_dir::String = ""
    resume_checkpoint::String = ""
    updates::Int = 1_000
    log_interval::Int = 100
    evaluation_interval::Int = 1_000
    checkpoint_interval::Int = 1_000
    learning_rate::Float32 = 3.0f-3
    warmup_updates::Int = 100
    finish_learning_rate::Float32 = 3.0f-4
    workers::Int = min(20, Threads.nthreads(:default))
    candidate_chunk::Int = 4
end

function usage(io::IO=stdout)
    println(io, "Canonical HD relation-graph full-data trainer")
    println(io, "usage:")
    println(io, "  julia --threads=20,0 train_scratch.jl scratch --run-dir PATH [options]")
    println(io, "  julia --threads=20,0 train_scratch.jl resume  --run-dir PATH [--checkpoint PATH] [options]")
    println(io, "")
    println(io, "Required scientific schedule options:")
    println(io, "  --updates N --log-every N --evaluate-every N --checkpoint-every N")
    println(io, "  --learning-rate X --warmup-updates N --finish-learning-rate X")
    println(io, "  --workers N --candidate-chunk N --dataset PATH")
    println(io, "")
    println(io, "State batch 8, candidate width 80, sampler seed, model topology, and")
    println(io, "development panel are fixed and cannot be changed from the CLI.")
end

function validate_options(options::Options)
    options.mode in (:scratch, :resume) || error(
        "first argument must be scratch or resume",
    )
    isempty(options.run_dir) && error("--run-dir is required")
    options.updates >= 1 || error("--updates must be positive")
    options.log_interval >= 1 || error("--log-every must be positive")
    options.evaluation_interval >= 1 || error(
        "--evaluate-every must be positive",
    )
    options.checkpoint_interval >= 1 || error(
        "--checkpoint-every must be positive",
    )
    isfinite(options.learning_rate) && options.learning_rate > 0.0f0 || error(
        "--learning-rate must be finite and positive",
    )
    isfinite(options.finish_learning_rate) &&
        options.finish_learning_rate > 0.0f0 || error(
        "--finish-learning-rate must be finite and positive",
    )
    options.finish_learning_rate <= options.learning_rate || error(
        "--finish-learning-rate cannot exceed --learning-rate",
    )
    0 <= options.warmup_updates < options.updates || error(
        "--warmup-updates must lie in 0:(updates - 1)",
    )
    1 <= options.workers <= Threads.nthreads(:default) || error(
        "--workers must fit the Julia default thread pool",
    )
    options.candidate_chunk >= 1 || error(
        "--candidate-chunk must be positive",
    )
    options.mode === :scratch && !isempty(options.resume_checkpoint) && error(
        "scratch mode does not accept --checkpoint",
    )
    return options
end

function parse_options(args=ARGS)
    isempty(args) && error("missing scratch/resume mode")
    args[1] == "--help" && (usage(); return nothing)
    mode = args[1] == "scratch" ? :scratch :
        args[1] == "resume" ? :resume :
        error("first argument must be scratch or resume")
    specifications = Dict{String,Tuple{Symbol,DataType}}(
        "--dataset" => (:dataset, String),
        "--run-dir" => (:run_dir, String),
        "--checkpoint" => (:resume_checkpoint, String),
        "--updates" => (:updates, Int),
        "--log-every" => (:log_interval, Int),
        "--evaluate-every" => (:evaluation_interval, Int),
        "--checkpoint-every" => (:checkpoint_interval, Int),
        "--learning-rate" => (:learning_rate, Float32),
        "--warmup-updates" => (:warmup_updates, Int),
        "--finish-learning-rate" => (:finish_learning_rate, Float32),
        "--workers" => (:workers, Int),
        "--candidate-chunk" => (:candidate_chunk, Int),
    )
    values = Dict{Symbol,Any}(:mode => mode)
    index = 2
    while index <= length(args)
        token = args[index]
        token == "--help" && (usage(); return nothing)
        startswith(token, "--") || error("unknown positional argument: $token")
        equals = findfirst(==('='), token)
        if isnothing(equals)
            index < length(args) || error("missing value after $token")
            key = token
            raw = args[index + 1]
            index += 2
        else
            key = token[1:(equals - 1)]
            raw = token[(equals + 1):end]
            index += 1
        end
        haskey(specifications, key) || error("unknown option: $key")
        field, kind = specifications[key]
        haskey(values, field) && error("duplicate option: $key")
        values[field] = kind === String ? String(raw) : parse(kind, raw)
    end
    options = Options(; values...)
    options = Options(;
        (name => (
            name in (:dataset, :run_dir, :resume_checkpoint) &&
            !isempty(getfield(options, name)) ?
                abspath(getfield(options, name)) : getfield(options, name)
        ) for name in fieldnames(Options))...,
    )
    return validate_options(options)
end

"""Linear warmup followed by cosine decay to the explicit finish rate."""
function learning_rate_at(options::Options, completed_update::Integer)
    completed_update isa Bool && throw(ArgumentError("update cannot be Bool"))
    update = Int(completed_update)
    0 <= update <= options.updates || throw(BoundsError(0:options.updates, update))
    if update == 0
        return options.warmup_updates == 0 ?
            options.learning_rate : 0.0f0
    elseif update <= options.warmup_updates
        return options.learning_rate *
            Float32(update / options.warmup_updates)
    end
    decay_updates = options.updates - options.warmup_updates
    progress = Float32(
        (update - options.warmup_updates) / decay_updates,
    )
    cosine = 0.5f0 * (1.0f0 + cos(Float32(pi) * progress))
    return options.finish_learning_rate +
        (options.learning_rate - options.finish_learning_rate) * cosine
end

@inline function _optimizer_policy_record(config::Optimizer.OptimizerConfig)
    names = fieldnames(Optimizer.OptimizerConfig)
    return NamedTuple{names}(ntuple(
        index -> getfield(config, names[index]),
        length(names),
    ))
end

function _base_optimizer_config(options::Options, learning_rate::Real)
    return Optimizer.OptimizerConfig(;
        learning_rate,
        clip_norm=1.0f0,
        weight_decay=1.0f-4,
    )
end

function run_contract(
    options::Options,
    data::Data.ExperimentDataset,
    parameters::Root.ModelParameters,
)
    closure = Checkpoint.canonical_source_closure()
    base_optimizer = _base_optimizer_config(options, options.learning_rate)
    return (;
        schema=RUN_CONTRACT_SCHEMA,
        source_fingerprint=closure.aggregate,
        dataset_root=data.root,
        dataset_manifest_sha256=data.manifest_sha256,
        training_rows=length(data.train_rows),
        training_rows_sha256=Data.ordered_rows_sha256(data.train_rows),
        development_rows=data.development.states,
        development_rows_sha256=data.development.rows_sha256,
        model_fingerprint=Checkpoint.model_fingerprint(parameters),
        state_batch=Data.STATE_BATCH,
        candidate_width=Data.CANDIDATE_WIDTH,
        sampler_seed=SAMPLER_SEED,
        workers=options.workers,
        candidate_chunk=options.candidate_chunk,
        target_updates=options.updates,
        log_interval=options.log_interval,
        evaluation_interval=options.evaluation_interval,
        checkpoint_interval=options.checkpoint_interval,
        learning_rate_schedule=(;
            kind="linear-warmup-cosine-finish-v1",
            peak=options.learning_rate,
            warmup_updates=options.warmup_updates,
            finish=options.finish_learning_rate,
            finish_update=options.updates,
        ),
        optimizer_policy=_optimizer_policy_record(base_optimizer),
    )
end

function _prepare_run_directory!(options::Options)
    if options.mode === :scratch
        if ispath(options.run_dir)
            isempty(readdir(options.run_dir)) || error(
                "scratch run directory is not empty: $(options.run_dir)",
            )
        else
            mkpath(options.run_dir)
        end
    else
        isdir(options.run_dir) || error(
            "resume run directory does not exist: $(options.run_dir)",
        )
    end
    mkpath(joinpath(options.run_dir, CHECKPOINT_DIRECTORY))
    return options.run_dir
end

@inline _progress_path(options::Options) =
    joinpath(options.run_dir, PROGRESS_FILE)

function _append_json!(path::AbstractString, record)
    open(path, "a") do io
        JSON3.write(io, record)
        write(io, '\n')
        flush(io)
    end
    return record
end

function _write_pointer!(run_dir::AbstractString, checkpoint::AbstractString)
    destination = joinpath(run_dir, LATEST_POINTER)
    temporary = destination * ".tmp"
    relative = relpath(abspath(checkpoint), abspath(run_dir))
    open(temporary, "w") do io
        println(io, relative)
    end
    mv(temporary, destination; force=true)
    return destination
end

function _latest_checkpoint(run_dir::AbstractString)
    pointer = joinpath(run_dir, LATEST_POINTER)
    isfile(pointer) || error("resume checkpoint pointer does not exist: $pointer")
    relative = strip(read(pointer, String))
    isempty(relative) && error("resume checkpoint pointer is empty")
    candidate = normpath(joinpath(run_dir, relative))
    isfile(candidate) || error("resume checkpoint does not exist: $candidate")
    return abspath(candidate)
end

function _checkpoint_path(options::Options, update::Int)
    return joinpath(
        options.run_dir,
        CHECKPOINT_DIRECTORY,
        "checkpoint_" * lpad(string(update), 9, '0') * ".jls",
    )
end

function _save_checkpoint!(options, trainer, sampler, update, contract)
    destination = _checkpoint_path(options, update)
    isfile(destination) && error("refusing to overwrite checkpoint: $destination")
    Checkpoint.save_checkpoint(
        destination,
        trainer,
        sampler;
        update,
        run_contract=contract,
    )
    _write_pointer!(options.run_dir, destination)
    return destination
end

@inline function _development_record(metrics::Data.DevelopmentMetrics)
    return (;
        states=metrics.states,
        candidates=metrics.candidates,
        composite_loss=metrics.composite_loss,
        composite_excess=metrics.composite_excess,
        q_listnet_cross_entropy=metrics.q_listnet_cross_entropy,
        q_teacher_entropy=metrics.q_teacher_entropy,
        q_excess=metrics.q_excess,
        legacy_stable_top1=metrics.legacy_stable_top1,
        tie_aware_top1=metrics.tie_aware_top1,
        ndcg=metrics.ndcg,
        pairwise_accuracy=metrics.pairwise_accuracy,
    )
end

@inline function _training_record(
    update::Int,
    metrics,
    learning_rate::Float32,
    sampler,
    elapsed_seconds::Float64,
    window_updates::Int,
)
    updates_per_second = window_updates / elapsed_seconds
    return (;
        kind="train",
        timestamp=string(Dates.now()),
        update,
        consumed_states=string(Sampler.sampler_consumed_rows(sampler)),
        learning_rate=Float64(learning_rate),
        composite_loss=Float64(metrics.loss.composite_loss),
        composite_excess=Float64(metrics.excess_loss),
        q_excess=Float64(metrics.loss.listnet_kl),
        tie_aware_top1=Float64(metrics.tie_top1),
        gradient_norm=metrics.gradient_norm,
        clip_scale=Float64(metrics.clip_scale),
        active_program_rows=metrics.active_program_rows,
        base_relation_events=metrics.base_relation_events,
        candidate_relation_events=metrics.candidate_relation_events,
        base_motif_events=metrics.base_motif_events,
        candidate_motif_events=metrics.candidate_motif_events,
        base_output_events=metrics.base_output_events,
        candidate_output_events=metrics.candidate_output_events,
        affected_positions=metrics.affected_positions,
        affected_relations=metrics.affected_relations,
        affected_motifs=metrics.affected_motifs,
        updates_per_second,
        states_per_second=Data.STATE_BATCH * updates_per_second,
    )
end

function _print_record(record)
    println(JSON3.write(record))
    flush(stdout)
    return record
end

function _initialize_training(options::Options, data::Data.ExperimentDataset)
    parameters = Root.initialize_model()
    contract = run_contract(options, data, parameters)
    batch = Ranking.Batch(Data.STATE_BATCH, Data.CANDIDATE_WIDTH)

    if options.mode === :scratch
        sampler = Sampler.DeterministicEpochSampler(
            data.train_rows,
            SAMPLER_SEED,
        )
        config = _base_optimizer_config(options, learning_rate_at(options, 0))
        trainer = Root.BarrierlessRelationGraphTrainer(
            parameters,
            batch,
            data.ranking;
            optimizer_config=config,
            worker_capacity=options.workers,
            candidate_chunk_size=options.candidate_chunk,
        )
        return trainer, sampler, 0, contract
    end

    checkpoint_path = isempty(options.resume_checkpoint) ?
        _latest_checkpoint(options.run_dir) : options.resume_checkpoint
    snapshot = Checkpoint.load_checkpoint(checkpoint_path)
    snapshot.update < options.updates || error(
        "checkpoint update must be below target updates",
    )
    scheduled = learning_rate_at(options, snapshot.update)
    snapshot.optimizer_config.learning_rate == scheduled || error(
        "checkpoint learning rate differs from the declared schedule",
    )
    trainer = Root.BarrierlessRelationGraphTrainer(
        parameters,
        batch,
        data.ranking;
        optimizer_config=snapshot.optimizer_config,
        worker_capacity=options.workers,
        candidate_chunk_size=options.candidate_chunk,
    )
    restored = Checkpoint.restore_checkpoint!(
        trainer,
        snapshot,
        data.train_rows;
        expected_run_contract=contract,
    )
    return trainer, restored.sampler, restored.update, contract
end

function main(args=ARGS)
    options = parse_options(args)
    isnothing(options) && return nothing
    Threads.nthreads(:interactive) == 0 || error(
        "canonical scheduler requires --threads=WORKERS,0",
    )
    BLAS.set_num_threads(1)
    _prepare_run_directory!(options)
    data = Data.load_experiment_data(options.dataset)
    trainer, sampler, update, contract = _initialize_training(options, data)
    progress = _progress_path(options)
    if options.mode === :scratch
        isfile(progress) && error("scratch progress file already exists: $progress")
    else
        isfile(progress) || error("resume progress file does not exist: $progress")
    end
    evaluator = Data.DevelopmentEvaluator(data)
    source_closure = Checkpoint.canonical_source_closure()
    start_record = (;
        kind=options.mode === :scratch ? "start" : "resume",
        timestamp=string(Dates.now()),
        update,
        target_updates=options.updates,
        run_contract_fingerprint=Checkpoint.run_contract_fingerprint(contract),
        source_fingerprint=source_closure.aggregate,
        dataset_manifest_sha256=data.manifest_sha256,
        training_rows_sha256=Data.ordered_rows_sha256(data.train_rows),
        development_rows_sha256=data.development.rows_sha256,
        parameters=Root.stored_parameter_count(trainer.parameters),
        state_batch=Data.STATE_BATCH,
        candidate_width=Data.CANDIDATE_WIDTH,
        workers=options.workers,
        candidate_chunk=options.candidate_chunk,
    )
    _append_json!(progress, start_record)
    _print_record(start_record)

    last_metrics = nothing
    last_log_update = update
    window_started = time_ns()
    last_evaluation = -1
    last_checkpoint = -1
    Root.run_trainer_team!(
        trainer;
        workers=options.workers,
        queue_capacity=64,
        binding_mode=:none,
    ) do session
        while update < options.updates
            update += 1
            Sampler.next_batch!(trainer.batch.rows, sampler)
            Data.assert_training_rows!(data, trainer.batch.rows)
            learning_rate = learning_rate_at(options, update)
            Parallel.set_learning_rate!(trainer, learning_rate)
            last_metrics = Root.train_update!(session)

            if update % options.log_interval == 0 || update == options.updates
                elapsed = (time_ns() - window_started) * 1.0e-9
                record = _training_record(
                    update,
                    last_metrics,
                    learning_rate,
                    sampler,
                    elapsed,
                    update - last_log_update,
                )
                _append_json!(progress, record)
                _print_record(record)
                last_log_update = update
                window_started = time_ns()
            end

            if update % options.evaluation_interval == 0 ||
               update == options.updates
                metrics = Data.evaluate_development!(session, data, evaluator)
                record = merge((;
                    kind="development_validation",
                    timestamp=string(Dates.now()),
                    update,
                    learning_rate=Float64(learning_rate),
                    panel_rows_sha256=data.development.rows_sha256,
                    held_test_touched=false,
                    sealed_game_seed_touched=false,
                ), _development_record(metrics))
                _append_json!(progress, record)
                _print_record(record)
                last_evaluation = update
            end

            if update % options.checkpoint_interval == 0 ||
               update == options.updates
                checkpoint = _save_checkpoint!(
                    options,
                    trainer,
                    sampler,
                    update,
                    contract,
                )
                record = (;
                    kind="checkpoint",
                    timestamp=string(Dates.now()),
                    update,
                    path=checkpoint,
                    bytes=filesize(checkpoint),
                )
                _append_json!(progress, record)
                _print_record(record)
                last_checkpoint = update
            end
        end
    end
    last_evaluation == update || error("final development evaluation was not recorded")
    last_checkpoint == update || error("final checkpoint was not recorded")
    complete = (;
        kind="complete",
        timestamp=string(Dates.now()),
        update,
        consumed_states=string(Sampler.sampler_consumed_rows(sampler)),
        progress,
        checkpoint=_checkpoint_path(options, update),
    )
    _append_json!(progress, complete)
    _print_record(complete)
    return complete
end

end # module CanonicalRelationScratch

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    CanonicalRelationScratch.main()
end
