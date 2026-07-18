using Dates
using JLD2
using JSON3
using Lux
using Optimisers
using Random
using SHA
using Statistics

const TRAINING_DIR = @__DIR__
const EXPERIMENT_DIR = normpath(joinpath(TRAINING_DIR, ".."))
const REPOSITORY_ROOT = normpath(joinpath(EXPERIMENT_DIR, "..", ".."))

include(joinpath(TRAINING_DIR, "core.jl"))
include(joinpath(TRAINING_DIR, "native_backend.jl"))
using .BeatFirstTrainingCore

# The selected source is included before main is compiled, avoiding runtime
# method-definition/world-age problems while keeping the training driver
# independent of the three architecture implementations.
const MODEL_SOURCE = abspath(get(
    ENV,
    "BEAT_MODEL_SOURCE",
    joinpath(EXPERIMENT_DIR, "models", "models.jl"),
))
isfile(MODEL_SOURCE) || error("BEAT_MODEL_SOURCE does not exist: $MODEL_SOURCE")
include(MODEL_SOURCE)
const MODEL_MODULE_NAME = Symbol(get(ENV, "BEAT_MODEL_MODULE", "BeatFirstModels"))
isdefined(Main, MODEL_MODULE_NAME) || error(
    "model source did not define module $MODEL_MODULE_NAME",
)
const MODEL_API = getfield(Main, MODEL_MODULE_NAME)

const BACKEND_KIND = lowercase(get(ENV, "BEAT_BACKEND", "native"))
const BACKEND_API = if BACKEND_KIND in ("native", "zygote", "cpu")
    BeatFirstNativeBackend
else
    source = abspath(get(ENV, "BEAT_BACKEND_SOURCE", ""))
    isfile(source) || error("BEAT_BACKEND_SOURCE does not exist: $source")
    include(source)
    module_name = Symbol(get(ENV, "BEAT_BACKEND_MODULE", "BeatFirstFixedShapeBackend"))
    isdefined(Main, module_name) || error("backend source did not define module $module_name")
    getfield(Main, module_name)
end

sha256_file(path) = bytes2hex(open(sha256, path))

mutable struct CandidateRun
    kind::Symbol
    model
    learner
    meta
    update::Int
    train_seconds::Float64
    losses::Vector{Float64}
    stages::Vector{Any}
end

function _parse_update_budgets(value::AbstractString)
    result = parse.(Int, split(value, ','))
    length(result) == 3 || error("BEAT_HALVING_UPDATES must contain three comma-separated budgets")
    all(>(0), result) || error("halving update budgets must be positive")
    return result
end

function _setup_model(kind::Symbol, rng, n_quantiles::Int)
    isdefined(MODEL_API, :setup_model) || error("model module lacks setup_model")
    setup = MODEL_API.setup_model(kind, rng; n_quantiles)
    model, ps, st, meta = setup.model, setup.ps, setup.st, setup.meta
    output_keys = (:q, :death_logit, :quantiles)
    return model, ps, st, merge((; output_keys), meta)
end

function _init_run(
    kind::Symbol,
    seed::UInt64,
    n_quantiles::Int,
    learning_rate::Float32,
    weight_decay::Float32,
    host_template,
)
    model, ps, st, meta = _setup_model(kind, Xoshiro(seed), n_quantiles)
    optimiser = Optimisers.AdamW(
        learning_rate, (0.9, 0.999), weight_decay,
    )
    learner = BACKEND_API.init_backend(
        model,
        ps,
        st,
        optimiser,
        supervised_objective,
        host_template;
        max_candidates=MAX_CANDIDATES,
        backend=get(ENV, "BEAT_BACKEND_DEVICE", "cpu"),
    )
    return CandidateRun(kind, model, learner, meta, 0, 0.0, Float64[], Any[])
end

function _host_checkpoint(run::CandidateRun)
    raw = BACKEND_API.host_checkpoint(run.learner)
    return (;
        ps=hasproperty(raw, :ps) ? raw.ps : raw.parameters,
        st=hasproperty(raw, :st) ? raw.st : raw.states,
        optimizer_state=raw.optimizer_state,
        step=raw.step,
    )
end

function _predictor(run::CandidateRun)
    checkpoint = _host_checkpoint(run)
    test_state = Lux.testmode(checkpoint.st)
    return batch -> first(run.model(batch.inputs, checkpoint.ps, test_state))
end

function _train_stage!(
    run::CandidateRun,
    dataset,
    host_batch,
    row_schedule,
)
    stage_started = time()
    first_loss = nothing
    last_statistics = nothing
    @info "Beat-first training stage started" kind=run.kind stage_updates=length(row_schedule) current_update=run.update
    for (stage_update, rows) in enumerate(row_schedule)
        pack_started = time()
        pack_batch!(host_batch, dataset, rows)
        pack_seconds = time() - pack_started
        statistics = BACKEND_API.train_step!(run.learner, host_batch)
        loss = Float64(statistics.loss)
        isfinite(loss) || error("$(run.kind) produced non-finite loss at update $(run.update + 1)")
        first_loss = isnothing(first_loss) ? loss : first_loss
        push!(run.losses, loss)
        run.update += 1
        run.train_seconds += Float64(statistics.wall_seconds) + pack_seconds
        last_statistics = merge(statistics, (; external_pack_seconds=pack_seconds))
        if stage_update == 1 || stage_update % 25 == 0 || stage_update == length(row_schedule)
            @info "Beat-first training progress" kind=run.kind stage_update total_update=run.update loss updates_per_second=stage_update/(time()-stage_started)
        end
    end
    elapsed = time() - stage_started
    return (;
        updates=length(row_schedule),
        first_loss,
        last_loss=last(run.losses),
        wall_seconds=elapsed,
        updates_per_second=length(row_schedule) / elapsed,
        candidates_per_second=length(row_schedule) * length(host_batch.mask) / elapsed,
        last_statistics,
    )
end

function _save_checkpoint(
    run::CandidateRun,
    path::AbstractString,
    metrics,
    stage::Int,
    config,
)
    checkpoint = _host_checkpoint(run)
    mkpath(dirname(path))
    jldsave(
        path;
        model_kind=String(run.kind),
        ps=checkpoint.ps,
        st=checkpoint.st,
        optimizer_state=checkpoint.optimizer_state,
        update=run.update,
        stage,
        metrics,
        meta=run.meta,
        config,
        losses=run.losses,
    )
    return (; path=abspath(path), bytes=filesize(path), sha256=sha256_file(path))
end

function _json_safe_meta(meta)
    return Dict(String(key) => string(value) for (key, value) in pairs(meta))
end

function main()
    dataset_path = abspath(get(
        ENV,
        "BEAT_TEACHER_DATASET",
        raw"D:\tetris-paper-plus\datasets\learning\teacher_plus_dagger_c13_round1.jld2",
    ))
    checkpoint_root = abspath(get(
        ENV,
        "BEAT_CHECKPOINT_ROOT",
        raw"D:\tetris-paper-plus\checkpoints\beat_first_v1",
    ))
    run_root = abspath(get(
        ENV,
        "BEAT_RUN_ROOT",
        raw"D:\tetris-paper-plus\runs\beat_first_v1",
    ))
    mkpath(checkpoint_root)
    mkpath(run_root)
    dataset = load_teacher_dataset(dataset_path)
    episodes = sort(unique(dataset.episode_ids))
    validation_episode_count = parse(Int, get(ENV, "BEAT_VALIDATION_EPISODES", "2"))
    1 <= validation_episode_count < length(episodes) || error("invalid validation episode count")
    validation_episodes = episodes[(end - validation_episode_count + 1):end]
    training_rows = findall(id -> !(id in validation_episodes), dataset.episode_ids)
    validation_rows = findall(id -> id in validation_episodes, dataset.episode_ids)
    isempty(training_rows) && error("empty teacher training split")
    isempty(validation_rows) && error("empty held-out teacher split")

    single_model = strip(get(ENV, "BEAT_SINGLE_MODEL", ""))
    model_kinds = if isempty(single_model)
        Symbol.(split(get(
            ENV,
            "BEAT_MODEL_KINDS",
            join(string.(MODEL_API.MODEL_KINDS), ','),
        ), ','))
    else
        [Symbol(single_model)]
    end
    length(model_kinds) in (1, 3) || error("trainer requires one lazy model or three halving models")
    length(unique(model_kinds)) == length(model_kinds) || error("model kinds must be distinct")
    all(kind -> kind in MODEL_API.MODEL_KINDS, model_kinds) || error("unknown model kind")
    budgets = _parse_update_budgets(get(ENV, "BEAT_HALVING_UPDATES", "100,200,500"))
    max_stage = parse(Int, get(ENV, "BEAT_MAX_HALVING_STAGE", "3"))
    1 <= max_stage <= 3 || error("BEAT_MAX_HALVING_STAGE must be in 1:3")
    state_batch = parse(Int, get(ENV, "BEAT_STATE_BATCH", "2"))
    n_quantiles = parse(Int, get(ENV, "BEAT_N_QUANTILES", "16"))
    learning_rate = parse(Float32, get(ENV, "BEAT_LEARNING_RATE", "3e-4"))
    weight_decay = parse(Float32, get(ENV, "BEAT_WEIGHT_DECAY", "1e-4"))
    seed = parse(UInt64, get(ENV, "BEAT_TRAIN_SEED", "2026071801"))
    run_id = get(ENV, "BEAT_RUN_ID", "beat_teacher_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")

    config = (;
        run_id,
        dataset_path,
        dataset_sha256=sha256_file(dataset_path),
        model_source=MODEL_SOURCE,
        model_source_sha256=sha256_file(MODEL_SOURCE),
        model_kinds=String.(model_kinds),
        single_model_mode=length(model_kinds) == 1,
        backend=BACKEND_KIND,
        backend_device=get(ENV, "BEAT_BACKEND_DEVICE", "cpu"),
        budgets,
        max_stage,
        state_batch,
        n_quantiles,
        learning_rate,
        weight_decay,
        seed,
        training_episode_ids=sort(unique(dataset.episode_ids[training_rows])),
        validation_episode_ids=validation_episodes,
        held_out_test_seeds_used=false,
        optional_target_coverage=optional_target_coverage(dataset),
    )
    host_batch = allocate_host_batch(state_batch)
    length(training_rows) >= state_batch || error("training split is smaller than state batch")
    # Reactant validates its owned host template before the first compile.
    # Supply one real nonempty fixed-shape batch; subsequent packs change values only.
    pack_batch!(host_batch, dataset, training_rows[1:state_batch])
    candidates = CandidateRun[]
    for kind in model_kinds
        model_index = findfirst(==(kind), MODEL_API.MODEL_KINDS)
        push!(candidates, _init_run(
            kind,
            seed + UInt64(model_index - 1),
            n_quantiles,
            learning_rate,
            weight_decay,
            host_batch,
        ))
    end

    started = time()
    rng = Xoshiro(seed + 0x9e3779b97f4a7c15)
    stage_results = Any[]
    survivors = candidates
    survivor_counts = length(candidates) == 1 ? (1, 1, 1) : (2, 1, 1)
    for stage in 1:max_stage
        stage_schedule = [rand(rng, training_rows, state_batch) for _ in 1:budgets[stage]]
        evaluations = Any[]
        for run in survivors
            training = _train_stage!(run, dataset, host_batch, stage_schedule)
            metrics = teacher_metrics(
                dataset,
                validation_rows,
                host_batch,
                _predictor(run),
            )
            checkpoint_path = joinpath(
                checkpoint_root, "$(run_id)_$(run.kind)_stage$(stage).jld2",
            )
            checkpoint = _save_checkpoint(run, checkpoint_path, metrics, stage, config)
            record = (;
                kind=String(run.kind),
                stage,
                total_updates=run.update,
                parameter_count=get(run.meta, :parameters, nothing),
                training,
                metrics,
                checkpoint,
            )
            push!(run.stages, record)
            push!(evaluations, record)
            @info "Beat-first teacher stage complete" record
        end
        order = sortperm(
            survivors;
            by=run -> promotion_key(last(run.stages).metrics),
            rev=true,
            alg=MergeSort,
        )
        keep = min(survivor_counts[stage], length(survivors))
        promoted = survivors[order[1:keep]]
        push!(stage_results, (;
            stage,
            evaluations,
            promoted=String.(getfield.(promoted, :kind)),
            selection_metric="lexicographic(top1, ndcg, pairwise, -old_q_huber)",
        ))
        survivors = promoted
    end

    champion = length(survivors) == 1 ? only(survivors) : nothing
    champion_record = isnothing(champion) ? nothing : last(champion.stages)
    summary = (;
        experiment_id=run_id,
        generated_at=string(now()),
        config,
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
        model_metadata=Dict(String(run.kind) => _json_safe_meta(run.meta) for run in candidates),
        stages=stage_results,
        champion=isnothing(champion) ? nothing : String(champion.kind),
        champion_checkpoint=isnothing(champion_record) ? nothing : champion_record.checkpoint,
        champion_teacher_metrics=isnothing(champion_record) ? nothing : champion_record.metrics,
        current_survivors=String.(getfield.(survivors, :kind)),
        total_wall_seconds=time() - started,
        completion_reason=max_stage == 3 ?
            "configured 3-to-2-to-1 teacher successive halving completed" :
            "configured teacher halving stage $max_stage completed",
        next_action=max_stage == 3 ?
            "run fixed development games, then promote only the champion to end-to-end RL" :
            "review stage metrics and continue the frozen halving schedule with only current survivors",
    )
    summary_path = joinpath(run_root, "$(run_id).json")
    open(summary_path, "w") do io
        JSON3.pretty(io, summary)
    end
    @info "Beat-first teacher pretraining complete" summary_path champion=summary.champion survivors=summary.current_survivors
    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
