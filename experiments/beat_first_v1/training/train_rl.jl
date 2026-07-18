using Dates
using JLD2
using JSON3
using Lux
using Optimisers
using Random
using SHA
using Statistics
using Zygote

const RL_TRAINING_DIR = @__DIR__
const RL_EXPERIMENT_DIR = normpath(joinpath(RL_TRAINING_DIR, ".."))
const RL_REPOSITORY_ROOT = normpath(joinpath(RL_EXPERIMENT_DIR, "..", ".."))

include(joinpath(RL_TRAINING_DIR, "core.jl"))
include(joinpath(RL_TRAINING_DIR, "rl_stage2.jl"))
include(joinpath(RL_TRAINING_DIR, "metrics_resume.jl"))
include(joinpath(RL_TRAINING_DIR, "native_backend.jl"))
include(joinpath(RL_EXPERIMENT_DIR, "backend", "fixedshape_learner.jl"))
include(joinpath(RL_EXPERIMENT_DIR, "models", "models.jl"))
# Canonical GameState, stable candidate order, legacy_candidate_batch and
# apply_node!; its guarded CLI main is not executed by this runner.
include(joinpath(RL_REPOSITORY_ROOT, "scripts", "evaluate_legacy_checkpoint.jl"))
# Reuse the exact candidate feature packer used by paired evaluation.
include(joinpath(RL_EXPERIMENT_DIR, "evaluation", "beat_first_adapter.jl"))

using .BeatFirstModels
using .BeatFirstMetricsResume
using .BeatFirstRLStage2
using .BeatFirstTrainingCore

const RL_CHECKPOINT_FORMAT_VERSION = 1
const RL_RESERVED_SEEDS = Set(vcat(collect(5756:5757), collect(8001:8008), collect(91001:91032)))

sha256_file(path::AbstractString) = bytes2hex(open(sha256, path))
sha256_text(value::AbstractString) = bytes2hex(sha256(codeunits(value)))

function _git_metadata()
    commit = readchomp(`git -C $RL_REPOSITORY_ROOT rev-parse HEAD`)
    tracked_status = readchomp(
        `git -C $RL_REPOSITORY_ROOT status --porcelain --untracked-files=no`,
    )
    isempty(tracked_status) || error(
        "RL training requires a committed tracked worktree; status: $tracked_status",
    )
    return (; commit, tracked_worktree_clean=true)
end

function _boolean_env(name::AbstractString, default::Bool)
    value = lowercase(strip(get(ENV, name, string(default))))
    value in ("1", "true", "yes") && return true
    value in ("0", "false", "no") && return false
    error("$name must be true or false")
end

function _write_json_atomic(path::AbstractString, value)
    mkpath(dirname(path))
    temporary = path * ".tmp"
    isfile(temporary) && rm(temporary; force=true)
    open(temporary, "w") do io
        JSON3.pretty(io, value)
        write(io, '\n')
    end
    mv(temporary, path; force=true)
    return path
end

function _copy_tree!(destination, source)
    keys(destination) == keys(source) || error("array-tree keys differ")
    for key in keys(destination)
        left, right = getproperty(destination, key), getproperty(source, key)
        if left isa NamedTuple
            _copy_tree!(left, right)
        else
            copyto!(left, right)
        end
    end
    return destination
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
                visit(
                    isempty(path) ? String(key) : "$path.$key",
                    getproperty(parameter, key),
                    gradient === nothing ? nothing : getproperty(gradient, key),
                )
            end
        elseif parameter isa Tuple
            for index in eachindex(parameter)
                visit("$path.$index", parameter[index], gradient === nothing ? nothing : gradient[index])
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

function _gradient_witness(model, ps, st, batch, meta)
    gradient = only(Zygote.gradient(
        parameters -> first(quantile_td_objective(model, parameters, st, batch)), ps,
    ))
    leaves = _gradient_leaves(ps, gradient)
    depth = Int(meta.depth)
    prefixes = meta.family === :preact_eca ? (;
        first="stem", middle="blocks.layer_$(cld(depth, 2))", final="blocks.layer_$depth",
    ) : (;
        first="board_stem", middle="blocks.layer_$(cld(depth, 2))", final="blocks.layer_$depth",
    )
    witness(prefix) = (;
        path=prefix,
        gradient_norm=sqrt(sum(
            leaf.gradient_norm^2 for leaf in leaves if startswith(leaf.path, prefix);
            init=0.0,
        )),
    )
    paths = (; first=witness(prefixes.first), middle=witness(prefixes.middle), final=witness(prefixes.final))
    passed = all(item -> isfinite(item.gradient_norm) && item.gradient_norm > 0, values(paths))
    passed || error("RL end-to-end gradient witness failed: $paths")
    global_norm = sqrt(sum(leaf.gradient_norm^2 for leaf in leaves; init=0.0))
    return (; global_gradient_norm=global_norm, parameter_norm=parameter_norm(ps), paths, passed)
end

function _load_promoted_checkpoint(path::AbstractString, promotion_path::AbstractString)
    isfile(path) || error("promoted supervised checkpoint does not exist: $path")
    fingerprint = (; absolute_path=abspath(path), bytes=filesize(path), sha256=sha256_file(path))
    payload = jldopen(path, "r") do file
        required = ("model_kind", "ps", "st", "meta", "config", "status")
        missing = filter(name -> !haskey(file, name), required)
        isempty(missing) || error("supervised checkpoint is missing $(join(missing, ", "))")
        status = file["status"]
        hasproperty(status, :game_eligible) && Bool(status.game_eligible) || error(
            "RL requires a supervised checkpoint with game_eligible=true",
        )
        config = file["config"]
        Int(config.trainable_parameter_count) == Int(config.total_parameter_count) || error(
            "promoted checkpoint was not trained end-to-end",
        )
        return (;
            kind=Symbol(String(file["model_kind"])),
            ps=file["ps"], st=file["st"], meta=file["meta"], config, status,
        )
    end
    payload.kind in BeatFirstModels.TRAINING_MODEL_KINDS || error(
        "stage-2 RL only accepts the two surviving PreAct/Gravity families",
    )
    isfile(promotion_path) || error(
        "BEAT_RL_PROMOTION_RECORD is mandatory; run the fixed development paired gate first",
    )
    promotion = JSON3.read(read(promotion_path, String))
    String(promotion.status) == "complete" || error("development promotion record is incomplete")
    String(promotion.stage) == "dev" || error("RL promotion record must be stage=dev")
    Int.(collect(promotion.seeds)) == [5756, 5757] || error(
        "RL promotion record must use exactly development seeds 5756 and 5757",
    )
    rules = promotion.rules
    Int(rules.max_pieces) == 250 || error("RL promotion must use max_pieces=250")
    Int(rules.next_count) == 5 || error("RL promotion must use NEXT=5")
    Bool(rules.hold_enabled) || error("RL promotion must keep HOLD enabled")
    String(rules.candidate_order) == "stable_node_key (duplicates preserved)" || error(
        "RL promotion candidate order differs from the frozen evaluator",
    )
    String(rules.tie_break) == "strict first maximum" || error(
        "RL promotion tie-breaking differs from the frozen evaluator",
    )
    Int(rules.lookahead_expansions) == 0 || error(
        "RL promotion must use zero lookahead expansions",
    )
    Int(rules.logical_full_candidate_score_calls_per_decision) == 1 || error(
        "RL promotion must use one full candidate score call per decision",
    )
    String(promotion.candidate.checkpoint.sha256) == fingerprint.sha256 || error(
        "promotion record checkpoint hash does not match the supervised checkpoint",
    )
    summary = promotion.summary
    Int(summary.episodes) == 2 || error("development promotion needs exactly two paired episodes")
    pairs = collect(promotion.pairs)
    length(pairs) == 2 || error("development promotion record must contain two pairs")
    Int[pair.seed for pair in pairs] == [5756, 5757] || error(
        "development pair seeds differ from the frozen schedule",
    )
    Float64(summary.paired_differences.mean) >= 500 || error(
        "development paired mean is below the preregistered +500 gate",
    )
    Int(summary.wins) * 2 > Int(summary.episodes) || error(
        "development paired differences are not positive on a strict majority",
    )
    Float64(summary.candidate_completion_rate) >= Float64(summary.old_completion_rate) || error(
        "development completion rate regressed",
    )
    rebuilt = BeatFirstModels.build_model(
        payload.kind; n_quantiles=Int(payload.meta.n_quantiles), payload.meta.config...,
    )
    parameter_count(payload.ps) == Int(payload.meta.parameters) || error(
        "promoted checkpoint parameter tree does not match metadata",
    )
    return merge(payload, (; model=rebuilt, fingerprint, promotion_path=abspath(promotion_path),
                            promotion_sha256=sha256_file(promotion_path)))
end

function _allocate_materialized_dataset(width::Int, batch_size::Int)
    return (;
        boards=zeros(UInt8, 24, 10, 1, batch_size),
        placements=zeros(UInt8, 24, 10, 1, width, batch_size),
        ren=zeros(Float32, 1, batch_size),
        back_to_back=zeros(Float32, 1, batch_size),
        tspin=zeros(Float32, width, batch_size),
        queues=zeros(UInt8, 7, 6, batch_size),
        teacher_q=zeros(Float32, width, batch_size),
        action_counts=zeros(Int, batch_size),
        selected_actions=ones(Int, batch_size),
        rewards=zeros(Float32, batch_size),
        seed_ids=zeros(Int, batch_size),
        episode_ids=zeros(Int, batch_size),
        split_group_ids=zeros(Int, batch_size),
        predefined_split=fill(:train, batch_size),
        episode_steps=collect(1:batch_size),
        terminal=falses(batch_size),
        candidate_death=falses(width, batch_size),
        candidate_death_available=falses(batch_size),
        geometry_cache=nothing,
    )
end

function _fill_materialized!(dataset, sets, actions, width::Int)
    length(sets) == length(actions) || throw(DimensionMismatch("candidate sets/actions differ"))
    for array in (
        dataset.boards, dataset.placements, dataset.ren, dataset.back_to_back,
        dataset.tspin, dataset.queues, dataset.teacher_q,
    )
        fill!(array, zero(eltype(array)))
    end
    for (slot, set) in enumerate(sets)
        count = length(set)
        count <= width || error(
            "online candidate count $count exceeds compiled width $width; no truncation is allowed",
        )
        materialized = materialize_candidate_set(set)
        dataset.boards[:, :, 1, slot] .= materialized.board
        dataset.placements[:, :, 1, 1:count, slot] .= materialized.placements
        dataset.ren[1, slot] = materialized.ren
        dataset.back_to_back[1, slot] = materialized.back_to_back
        dataset.tspin[1:count, slot] .= materialized.tspin
        dataset.queues[:, :, slot] .= materialized.queue
        if !isempty(materialized.teacher_q)
            dataset.teacher_q[1:count, slot] .= materialized.teacher_q
        end
        dataset.action_counts[slot] = count
        dataset.selected_actions[slot] = Int(actions[slot])
    end
    return dataset
end

function _allocate_rl_batch(width::Int, batch_size::Int, n_quantiles::Int)
    packed = allocate_host_batch(batch_size; max_candidates=width)
    targets = (;
        action_mask=zeros(Float32, width, batch_size),
        target_quantiles=zeros(Float32, n_quantiles, batch_size),
        importance_weights=ones(Float32, batch_size),
        teacher_q=zeros(Float32, width, batch_size),
        teacher_mask=zeros(Float32, width, batch_size),
        teacher_weight=zeros(Float32, 1),
    )
    return (; inputs=packed.inputs, targets, mask=packed.mask)
end

mutable struct RLPacker
    width::Int
    batch_size::Int
    current_dataset
    next_dataset
    current_packed
    next_packed
    rl_batch
end

function RLPacker(width::Int, batch_size::Int, n_quantiles::Int)
    return RLPacker(
        width, batch_size,
        _allocate_materialized_dataset(width, batch_size),
        _allocate_materialized_dataset(width, batch_size),
        allocate_host_batch(batch_size; max_candidates=width),
        allocate_host_batch(batch_size; max_candidates=width),
        _allocate_rl_batch(width, batch_size, n_quantiles),
    )
end

function _prepare_inputs!(packer::RLPacker, transitions, importance_weights)
    current_sets = [transition.current for transition in transitions]
    actions = Int[transition.action for transition in transitions]
    next_sets = CompactCandidateSet[
        transition.bootstrap === nothing ? transition.current : transition.bootstrap
        for transition in transitions
    ]
    _fill_materialized!(packer.current_dataset, current_sets, actions, packer.width)
    _fill_materialized!(packer.next_dataset, next_sets, ones(Int, packer.batch_size), packer.width)
    pack_batch!(packer.current_packed, packer.current_dataset, collect(1:packer.batch_size))
    pack_batch!(packer.next_packed, packer.next_dataset, collect(1:packer.batch_size))
    _copy_tree!(packer.rl_batch.inputs, packer.current_packed.inputs)
    copyto!(packer.rl_batch.mask, packer.current_packed.mask)
    targets = packer.rl_batch.targets
    fill!(targets.action_mask, 0.0f0)
    fill!(targets.teacher_q, 0.0f0)
    fill!(targets.teacher_mask, 0.0f0)
    copyto!(targets.importance_weights, importance_weights)
    for (slot, transition) in enumerate(transitions)
        action = Int(transition.action)
        targets.action_mask[action, slot] = 1.0f0
        labels = transition.current.teacher_q
        if !isempty(labels)
            targets.teacher_q[1:length(labels), slot] .= labels
            targets.teacher_mask[1:length(labels), slot] .= 1.0f0
        end
    end
    return packer.rl_batch, packer.next_packed
end

function _host_targets!(batch, next_packed, transitions, model, online, ema)
    online_output, _ = model(next_packed.inputs, online.ps, Lux.testmode(online.st))
    target_output, _ = model(next_packed.inputs, ema.ps, Lux.testmode(ema.st))
    rewards = Float32[transition.return_n for transition in transitions]
    discounts = Float32[transition.bootstrap_discount for transition in transitions]
    targets = double_dqn_quantile_targets(
        online_output, target_output, next_packed.mask, rewards, discounts,
    )
    copyto!(batch.targets.target_quantiles, targets)
    return targets
end

struct OpenVINOTeacherHook{M,T}
    module_api::M
    teacher::T
end

function _build_teacher()
    source = abspath(get(
        ENV, "BEAT_RL_TEACHER_SOURCE", joinpath(RL_TRAINING_DIR, "rl_teacher_openvino.jl"),
    ))
    isfile(source) || error("old-teacher hook does not exist: $source")
    include(source)
    isdefined(Main, :BeatFirstRLOpenVINOTeacher) || error(
        "old-teacher hook must define BeatFirstRLOpenVINOTeacher",
    )
    api = Main.BeatFirstRLOpenVINOTeacher
    teacher = Base.invokelatest(
        api.build_teacher;
        device=get(ENV, "BEAT_RL_TEACHER_DEVICE", "NPU"),
        batch_size=parse(Int, get(ENV, "BEAT_RL_TEACHER_BATCH", "16")),
    )
    return OpenVINOTeacherHook(api, teacher), source
end

_teacher_scores(hook::OpenVINOTeacherHook, input) =
    Base.invokelatest(hook.module_api.teacher_scores, hook.teacher, input)
_teacher_metadata(hook::OpenVINOTeacherHook) =
    Base.invokelatest(hook.module_api.teacher_metadata, hook.teacher)

function _student_scores(model, snapshot, state, nodes)
    input = Main.BeatFirstCandidateAdapter._pack_candidate_set(state, nodes)
    output, _ = model(input, snapshot.ps, Lux.testmode(snapshot.st))
    # Stage 2 optimizes the return distribution. Its online policy therefore
    # uses mean quantiles, matching Double-DQN action selection and the final
    # checkpoint adapter; scalar q remains the decaying teacher auxiliary.
    scores = Float32.(vec(Array(mean(output.quantiles; dims=1))))
    length(scores) == length(nodes) || error("student returned the wrong candidate count")
    all(isfinite, scores) || error("student returned non-finite Q")
    return scores
end

function _train_seed(seed::Int)
    seed in RL_RESERVED_SEEDS && error("RL attempted to consume reserved evaluation seed $seed")
    200_000 <= seed < 900_000 || error("RL train seeds must remain in the registered train-only range")
    return seed
end

function _rollout_episode!(
    replay, model, snapshot, teacher_hook, seed::Int;
    epsilon::Float64, max_steps::Int, n_step::Int, discount::Float32,
)
    _train_seed(seed)
    rollout_started = time()
    state = GameState(Xoshiro(seed))
    behavior_rng = Xoshiro(UInt64(seed) ⊻ 0x726c5f6461676765)
    sets = CompactCandidateSet[]
    actions = Int[]
    rewards = Float32[]
    terminals = Bool[]
    teacher_seconds = 0.0
    student_seconds = 0.0
    while !state.game_over_flag && length(sets) < max_steps
        nodes = stable_node_list(state)
        isempty(nodes) && break
        input = legacy_candidate_batch(state, nodes; next_count=5)
        labels = Float32[]
        teacher_seconds += @elapsed labels = _teacher_scores(teacher_hook, input)
        length(labels) == length(nodes) || error("old teacher returned the wrong candidate count")
        scores = Float32[]
        student_seconds += @elapsed scores = _student_scores(model, snapshot, state, nodes)
        action = rand(behavior_rng) < epsilon ? rand(behavior_rng, eachindex(nodes)) : argmax(scores)
        push!(sets, compact_candidate_set(input; teacher_q=labels))
        push!(actions, action)
        previous_score = state.score
        apply_node!(state, nodes[action])
        push!(rewards, scaled_game_reward(
            state.score - previous_score, state.game_over_flag,
        ))
        push!(terminals, state.game_over_flag)
    end
    episode_ids = fill(seed, length(sets))
    transition_data = nstep_transitions(
        rewards, episode_ids, collect(1:length(sets)), terminals; n=n_step, discount,
    )
    insertable_rows = insertable_nstep_rows(terminals; n=n_step)
    for row in insertable_rows
        bootstrap_row = transition_data.bootstrap_rows[row]
        bootstrap = bootstrap_row == 0 ? nothing : sets[bootstrap_row]
        push_transition!(replay, CompactTransition(
            sets[row], Int16(actions[row]), transition_data.returns[row], bootstrap,
            transition_data.bootstrap_discounts[row],
        ))
    end
    isempty(sets) && error("student rollout produced no transitions")
    return (;
        seed, steps=length(sets), final_score=state.score, game_over=state.game_over_flag,
        inserted=length(insertable_rows), truncated=!state.game_over_flag,
        teacher_seconds, student_seconds,
        wall_seconds=time() - rollout_started,
    )
end

function _snapshot(api, learner)
    raw = api.host_checkpoint(learner)
    return (;
        ps=hasproperty(raw, :ps) ? raw.ps : raw.parameters,
        st=hasproperty(raw, :st) ? raw.st : raw.states,
        optimizer_state=raw.optimizer_state,
        step=Int(raw.step),
        backend_updates=hasproperty(raw, :backend_updates) ? Int(raw.backend_updates) : Int(raw.step),
    )
end

function _parameter_snapshot(api, learner)
    raw = api.host_parameters(learner)
    return (;
        ps=raw.ps, st=raw.st, step=Int(raw.step),
        backend_updates=Int(raw.backend_updates),
    )
end

function _refresh_online_ema(
    api, learner, ema, previous_update::Int, current_update::Int, tau::Float32,
)
    current_update > previous_update || error("host target refresh did not advance")
    snapshot = _parameter_snapshot(api, learner)
    snapshot.step == current_update || error("host target refresh step mismatch")
    elapsed = current_update - previous_update
    effective_tau = Float32(1 - (1 - Float64(tau))^elapsed)
    ema_parameters!(ema.ps, snapshot.ps; tau=effective_tau)
    online = (; ps=snapshot.ps, st=snapshot.st)
    refreshed_ema = (; ps=ema.ps, st=snapshot.st)
    return online, refreshed_ema, current_update
end

function _init_learner(kind::String, model, snapshot, optimiser, batch; restore=nothing, width::Int)
    if kind == "reactant"
        api = Main.BeatFirstFixedShapeBackend
    elseif kind == "native"
        api = Main.BeatFirstNativeBackend
    else
        error("effective RL backend must be reactant or native")
    end
    learner = api.init_backend(
        model, snapshot.ps, snapshot.st, optimiser, quantile_td_objective, batch;
        max_candidates=width,
        backend=kind == "reactant" ? get(ENV, "BEAT_RL_BACKEND_DEVICE", "cpu") : "cpu",
        restore,
    )
    return api, learner
end

function _metric_summary(samples)
    isempty(samples) && return nothing
    average(name) = mean(Float64(getproperty(sample, name)) for sample in samples)
    update_wall_seconds = average(:update_wall_seconds)
    return (;
        updates=length(samples),
        qr_loss=average(:qr_loss), teacher_loss=average(:teacher_loss),
        total_loss=average(:total_loss), mean_absolute_td=average(:mean_absolute_td),
        mean_return=average(:mean_return), teacher_weight=last(samples).teacher_weight,
        update_wall_seconds,
        updates_per_second=1 / max(update_wall_seconds, eps(Float64)),
        compiled_learner_wall_seconds=average(:compiled_learner_wall_seconds),
    )
end

_host_scalar(value::Number) = Float64(value)
function _host_scalar(value::AbstractArray)
    length(value) == 1 || error("expected a scalar statistic, got size $(size(value))")
    return Float64(only(Array(value)))
end

function _reactant_lowering_failure(exception)
    evidence = lowercase(string(typeof(exception), " ", sprint(showerror, exception)))
    return any(
        marker -> occursin(marker, evidence),
        (
            "mlir", "enzymemlir", "xla", "lowering", "failed to lower",
            "compilation", "failed to compile", "trace failed", "tracing failed",
        ),
    )
end

function _source_hashes(teacher_source)
    paths = (;
        runner=abspath(@__FILE__),
        rl_primitives=joinpath(RL_TRAINING_DIR, "rl_stage2.jl"),
        metrics_resume=joinpath(RL_TRAINING_DIR, "metrics_resume.jl"),
        core=joinpath(RL_TRAINING_DIR, "core.jl"),
        native_backend=joinpath(RL_TRAINING_DIR, "native_backend.jl"),
        reactant_backend=joinpath(RL_EXPERIMENT_DIR, "backend", "fixedshape_learner.jl"),
        models=joinpath(RL_EXPERIMENT_DIR, "models", "models.jl"),
        model_common=joinpath(RL_EXPERIMENT_DIR, "models", "common.jl"),
        model_preact=joinpath(RL_EXPERIMENT_DIR, "models", "preact_eca.jl"),
        model_gravity=joinpath(RL_EXPERIMENT_DIR, "models", "gravity_film_convnext.jl"),
        candidate_adapter=joinpath(RL_EXPERIMENT_DIR, "evaluation", "beat_first_adapter.jl"),
        canonical_engine=joinpath(RL_REPOSITORY_ROOT, "scripts", "benchmark_legacy_engine.jl"),
        canonical_input=joinpath(RL_REPOSITORY_ROOT, "scripts", "evaluate_legacy_checkpoint.jl"),
        canonical_node=joinpath(
            RL_REPOSITORY_ROOT, "upstream", "TetrisAI", "src", "core", "components", "node.jl",
        ),
        canonical_analyzer=joinpath(
            RL_REPOSITORY_ROOT, "upstream", "TetrisAI", "src", "core", "analyzer.jl",
        ),
        openvino_teacher=joinpath(RL_REPOSITORY_ROOT, "tools", "legacy_openvino.py"),
        teacher_hook=teacher_source,
    )
    return NamedTuple{keys(paths)}(Tuple((; path, sha256=sha256_file(path)) for path in values(paths)))
end

function _load_resume(path::AbstractString)
    return jldopen(path, "r") do file
        Int(file["checkpoint_format_version"]) == RL_CHECKPOINT_FORMAT_VERSION || error(
            "unsupported RL checkpoint format",
        )
        return (;
            model_kind=Symbol(String(file["model_kind"])), ps=file["ps"], st=file["st"],
            optimizer_state=file["optimizer_state"], update=Int(file["update"]),
            backend_updates=Int(file["backend_updates"]), ema_ps=file["ema_ps"],
            ema_st=file["ema_st"], replay=file["replay"], rng=file["sampling_rng"],
            next_train_seed=Int(file["next_train_seed"]), update_credit=Float64(file["update_credit"]),
            actor_episodes=Int(file["actor_episodes"]), actor_steps=Int(file["actor_steps"]),
            actor_wall_seconds=Float64(file["actor_wall_seconds"]),
            teacher_seconds=Float64(file["teacher_seconds"]),
            student_seconds=Float64(file["student_seconds"]),
            learner_iteration_wall_seconds=Float64(file["learner_iteration_wall_seconds"]),
            history=Any[item for item in file["history"]], config=file["config"],
            config_sha256=String(file["config_sha256"]), effective_backend=String(file["effective_backend"]),
            gradient_witness=file["gradient_witness"], metrics=file["metrics"],
        )
    end
end

function _save_checkpoint!(
    path, model_kind, snapshot, ema, replay, rng, state, history, config,
    config_sha256, source_hashes, gradient_witness, metrics,
)
    snapshot.step == state.update == snapshot.backend_updates || error(
        "RL trainer/backend update counters differ",
    )
    atomic_jldsave(
        path;
        checkpoint_format_version=RL_CHECKPOINT_FORMAT_VERSION,
        stage=2,
        model_kind=String(model_kind),
        ps=snapshot.ps, st=snapshot.st, optimizer_state=snapshot.optimizer_state,
        update=state.update, train_state_step=snapshot.step,
        backend_updates=snapshot.backend_updates,
        ema_ps=ema.ps, ema_st=ema.st,
        replay, sampling_rng=deepcopy(rng), next_train_seed=state.next_train_seed,
        update_credit=state.update_credit, actor_episodes=state.actor_episodes,
        actor_steps=state.actor_steps, actor_wall_seconds=state.actor_wall_seconds,
        teacher_seconds=state.teacher_seconds, student_seconds=state.student_seconds,
        learner_iteration_wall_seconds=state.learner_iteration_wall_seconds,
        history, config, config_sha256, source_hashes,
        gradient_witness, metrics, effective_backend=state.effective_backend,
        meta=merge(config.model_meta, (; ranking_source=:mean_quantiles, stage=2)),
        status=(; stage="rl", full_parameter_end_to_end=true, held_out_seeds_used=false),
    )
    fingerprint = (; path=abspath(path), bytes=filesize(path), sha256=sha256_file(path))
    _write_json_atomic(path * ".json", (;
        generated_at=string(now()), checkpoint=fingerprint, config_sha256,
        manifest_sha256=config.manifest_sha256, source_hashes, metrics,
    ))
    return fingerprint
end

mutable struct RLControl
    update::Int
    next_train_seed::Int
    update_credit::Float64
    actor_episodes::Int
    actor_steps::Int
    actor_wall_seconds::Float64
    teacher_seconds::Float64
    student_seconds::Float64
    learner_iteration_wall_seconds::Float64
    effective_backend::String
end

function rl_main()
    promoted_path = abspath(get(ENV, "BEAT_RL_PROMOTED_CHECKPOINT", ""))
    promotion_path = abspath(get(ENV, "BEAT_RL_PROMOTION_RECORD", ""))
    isempty(strip(get(ENV, "BEAT_RL_PROMOTED_CHECKPOINT", ""))) && error(
        "BEAT_RL_PROMOTED_CHECKPOINT is required",
    )
    isempty(strip(get(ENV, "BEAT_RL_PROMOTION_RECORD", ""))) && error(
        "BEAT_RL_PROMOTION_RECORD is required",
    )
    promoted = _load_promoted_checkpoint(promoted_path, promotion_path)

    run_root = abspath(get(ENV, "BEAT_RL_RUN_ROOT", raw"D:\tetris-paper-plus\runs\beat_first_v1"))
    checkpoint_root = abspath(get(
        ENV, "BEAT_RL_CHECKPOINT_ROOT", raw"D:\tetris-paper-plus\checkpoints\beat_first_v1",
    ))
    mkpath(run_root); mkpath(checkpoint_root)
    resume_value = strip(get(ENV, "BEAT_RL_RESUME_CHECKPOINT", ""))
    resume = isempty(resume_value) ? nothing : _load_resume(abspath(resume_value))
    run_id = resume === nothing ? get(
        ENV, "BEAT_RL_RUN_ID",
        "$(promoted.kind)_rl_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
    ) : String(resume.config.experiment_id)

    width = parse(Int, get(
        ENV, "BEAT_RL_CANDIDATE_WIDTH", string(BeatFirstTrainingCore.MAX_CANDIDATES),
    ))
    width == BeatFirstTrainingCore.MAX_CANDIDATES || error(
        "Stage-2 RL requires the proven safe candidate width " *
        "$(BeatFirstTrainingCore.MAX_CANDIDATES); width migration is not implemented",
    )
    batch_size = parse(Int, get(ENV, "BEAT_RL_BATCH_SIZE", "4"))
    replay_capacity = parse(Int, get(ENV, "BEAT_RL_REPLAY_CAPACITY", "32768"))
    replay_warmup = parse(Int, get(ENV, "BEAT_RL_REPLAY_WARMUP", "1024"))
    batch_size > 0 || error("RL batch size must be positive")
    replay_capacity > 0 || error("replay capacity must be positive")
    replay_warmup >= batch_size || error("replay warmup is below batch size")
    replay_warmup <= replay_capacity || error("replay warmup exceeds capacity")
    max_updates = parse(Int, get(ENV, "BEAT_RL_MAX_UPDATES", "50000"))
    max_updates > 0 || error("RL maximum updates must be positive")
    checkpoint_interval = parse(Int, get(ENV, "BEAT_RL_CHECKPOINT_INTERVAL", "250"))
    100 <= checkpoint_interval <= 250 || error("RL checkpoint interval must be in 100:250")
    n_step = parse(Int, get(ENV, "BEAT_RL_N_STEP", "3"))
    n_step == 3 || error("beat-first stage 2 freezes n-step at 3")
    discount = parse(Float32, get(ENV, "BEAT_RL_DISCOUNT", "0.997"))
    ema_tau = parse(Float32, get(ENV, "BEAT_RL_EMA_TAU", "0.005"))
    0.0f0 <= discount <= 1.0f0 || error("RL discount must be in [0,1]")
    0.0f0 < ema_tau <= 1.0f0 || error("EMA tau must be in (0,1]")
    teacher_initial = parse(Float32, get(ENV, "BEAT_RL_TEACHER_WEIGHT", "0.25"))
    teacher_decay_updates = parse(Int, get(ENV, "BEAT_RL_TEACHER_DECAY_UPDATES", "50000"))
    target_refresh_interval = parse(Int, get(ENV, "BEAT_RL_TARGET_REFRESH_INTERVAL", "16"))
    replay_ratio = parse(Float64, get(ENV, "BEAT_RL_REPLAY_RATIO", "1.0"))
    epsilon = parse(Float64, get(ENV, "BEAT_RL_EPSILON", "0.05"))
    max_episode_steps = parse(Int, get(ENV, "BEAT_RL_EPISODE_STEPS", "250"))
    teacher_initial >= 0 || error("teacher weight must be nonnegative")
    teacher_decay_updates > 0 || error("teacher decay updates must be positive")
    target_refresh_interval > 0 || error("target refresh interval must be positive")
    replay_ratio > 0 || error("replay ratio must be positive")
    0.0 <= epsilon <= 1.0 || error("RL epsilon must be in [0,1]")
    max_episode_steps >= n_step || error("RL episode limit must be at least n-step")
    train_seed_start = parse(Int, get(ENV, "BEAT_RL_TRAIN_SEED_START", "200001"))
    learning_rate = parse(Float32, get(ENV, "BEAT_RL_LEARNING_RATE", "1e-4"))
    weight_decay = parse(Float32, get(ENV, "BEAT_RL_WEIGHT_DECAY", "1e-4"))
    learning_rate > 0 || error("RL learning rate must be positive")
    weight_decay >= 0 || error("RL weight decay must be nonnegative")
    allow_native_fallback = _boolean_env("BEAT_RL_ALLOW_NATIVE_FALLBACK", true)
    requested_backend = lowercase(get(ENV, "BEAT_RL_BACKEND", "reactant"))
    requested_backend in ("reactant", "native") || error("BEAT_RL_BACKEND must be reactant or native")

    teacher_hook, teacher_source = _build_teacher()
    source_hashes = _source_hashes(teacher_source)
    git = _git_metadata()
    manifest = joinpath(dirname(Base.active_project()), "Manifest.toml")
    model_meta = promoted.meta
    total_count = Int(model_meta.parameters)
    trainable_count = parameter_count(promoted.ps)
    total_count == trainable_count || error("RL trainable parameters do not equal total parameters")
    config = (;
        experiment_id=run_id, model_kind=String(promoted.kind), model_meta,
        total_parameter_count=total_count, trainable_parameter_count=trainable_count,
        full_parameter_end_to_end=true,
        promoted_checkpoint=promoted.fingerprint,
        promotion_record=(; path=promoted.promotion_path, sha256=promoted.promotion_sha256),
        requested_backend, allow_native_fallback,
        backend_device=get(ENV, "BEAT_RL_BACKEND_DEVICE", "cpu"),
        candidate_width=width, no_candidate_truncation=true,
        batch_size, replay_capacity, replay_warmup, max_updates, checkpoint_interval,
        n_step, discount, ema_tau, teacher_initial, teacher_decay_updates,
        target_refresh_interval,
        time_limit_policy="drop the final n transitions when no boundary bootstrap state is observed",
        target_network_policy="periodic Polyak refresh with tau compounded over the refresh interval",
        replay_ratio, epsilon, max_episode_steps, train_seed_start,
        learning_rate, weight_decay, teacher=_teacher_metadata(teacher_hook),
        train_seed_policy="200000:899999 excluding dev 5756:5757, validation 8001:8008, sealed 91001:91032",
        dev_validation_sealed_used_for_training=false,
        julia_version=string(VERSION), lux_version=string(Base.pkgversion(Lux)),
        optimisers_version=string(Base.pkgversion(Optimisers)),
        zygote_version=string(Base.pkgversion(Zygote)),
        project=Base.active_project(),
        manifest_sha256=isfile(manifest) ? sha256_file(manifest) : nothing,
        git_commit=git.commit, git_tracked_worktree_clean=git.tracked_worktree_clean,
        source_hashes,
    )
    config_sha256 = sha256_text(JSON3.write(config))
    if resume !== nothing
        resume.config_sha256 == config_sha256 || error("RL resume configuration/source mismatch")
        resume.model_kind == promoted.kind || error("RL resume model kind mismatch")
    end

    replay = resume === nothing ? PrioritizedReplay(replay_capacity) : resume.replay
    length(replay.entries) == replay_capacity || error("resumed replay capacity differs")
    sampling_rng = resume === nothing ? Xoshiro(0x726c5f7065725f31) : resume.rng
    control = resume === nothing ?
        RLControl(0, train_seed_start, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, requested_backend) :
        RLControl(
            resume.update, resume.next_train_seed, resume.update_credit,
            resume.actor_episodes, resume.actor_steps, resume.actor_wall_seconds,
            resume.teacher_seconds, resume.student_seconds,
            resume.learner_iteration_wall_seconds, resume.effective_backend,
        )
    history = resume === nothing ? Any[] : resume.history
    packer = RLPacker(width, batch_size, Int(model_meta.n_quantiles))
    optimiser = Optimisers.AdamW(learning_rate, (0.9, 0.999), weight_decay)
    initial_snapshot = resume === nothing ?
        (; ps=promoted.ps, st=promoted.st, optimizer_state=nothing, step=0, backend_updates=0) :
        (; ps=resume.ps, st=resume.st, optimizer_state=resume.optimizer_state,
           step=resume.update, backend_updates=resume.backend_updates)
    restore = resume === nothing ? nothing : (;
        parameters=resume.ps, states=resume.st, optimizer_state=resume.optimizer_state,
        step=resume.update, backend_updates=resume.backend_updates,
    )
    backend_api, learner = _init_learner(
        control.effective_backend, promoted.model, initial_snapshot, optimiser,
        packer.rl_batch; restore, width,
    )
    online = resume === nothing ? (; ps=promoted.ps, st=promoted.st) : (; ps=resume.ps, st=resume.st)
    ema = resume === nothing ? (; ps=deepcopy(promoted.ps), st=deepcopy(promoted.st)) :
          (; ps=resume.ema_ps, st=resume.ema_st)
    last_host_refresh_update = control.update
    gradient_witness = resume === nothing ? nothing : resume.gradient_witness

    latest_path = joinpath(checkpoint_root, "$(run_id)_latest.jld2")
    metrics_path = joinpath(run_root, "$(run_id)_metrics.jsonl")
    summary_path = joinpath(run_root, "$(run_id).json")
    if resume !== nothing
        lowercase(abspath(resume_value)) == lowercase(abspath(latest_path)) || error(
            "durable RL resume must use this run's latest checkpoint",
        )
        repair_metrics_log!(metrics_path, resume, latest_path)
    elseif isfile(metrics_path) || isfile(latest_path)
        error("RL run id already exists: $run_id")
    end

    # Student-visited DAgger rollouts fill the replay before any optimizer step.
    while replay.count < replay_warmup
        rollout = _rollout_episode!(
            replay, promoted.model, online, teacher_hook, control.next_train_seed;
            epsilon, max_steps=max_episode_steps, n_step, discount,
        )
        control.next_train_seed += 1
        control.actor_episodes += 1
        control.actor_steps += rollout.steps
        control.actor_wall_seconds += rollout.wall_seconds
        control.teacher_seconds += rollout.teacher_seconds
        control.student_seconds += rollout.student_seconds
        control.update_credit += replay_ratio * rollout.inserted
        @info "RL DAgger warmup" seed=rollout.seed score=rollout.final_score replay_count=replay.count
    end

    interval_samples = Any[]
    last_rollout = nothing
    while control.update < max_updates
        if control.update_credit < 1
            if last_host_refresh_update < control.update
                online, ema, last_host_refresh_update = _refresh_online_ema(
                    backend_api, learner, ema, last_host_refresh_update,
                    control.update, ema_tau,
                )
            end
            last_rollout = _rollout_episode!(
                replay, promoted.model, online, teacher_hook, control.next_train_seed;
                epsilon, max_steps=max_episode_steps, n_step, discount,
            )
            control.next_train_seed += 1
            control.actor_episodes += 1
            control.actor_steps += last_rollout.steps
            control.actor_wall_seconds += last_rollout.wall_seconds
            control.teacher_seconds += last_rollout.teacher_seconds
            control.student_seconds += last_rollout.student_seconds
            control.update_credit += replay_ratio * last_rollout.inserted
        end

        iteration_started = time()
        beta = 0.4 + 0.6 * min(control.update / max(max_updates, 1), 1)
        indices, importance = sample_indices(sampling_rng, replay, batch_size; beta)
        transitions = sampled_transitions(replay, indices)
        batch, next_packed = _prepare_inputs!(packer, transitions, importance)
        _host_targets!(
            batch, next_packed, transitions, promoted.model, online, ema,
        )
        batch.targets.teacher_weight[1] = teacher_weight(
            control.update + 1; initial=teacher_initial, decay_updates=teacher_decay_updates,
        )
        gradient_witness === nothing && (gradient_witness = _gradient_witness(
            promoted.model, online.ps, Lux.testmode(online.st), batch, model_meta,
        ))

        step_stats = nothing
        try
            step_stats = backend_api.train_step!(learner, batch)
        catch exception
            if control.update == 0 && control.effective_backend == "reactant" &&
               allow_native_fallback && _reactant_lowering_failure(exception)
                @warn "Reactant first update failed; using the registered Native Zygote fallback" exception
                control.effective_backend = "native"
                backend_api, learner = _init_learner(
                    "native", promoted.model,
                    (; ps=online.ps, st=online.st, optimizer_state=nothing, step=0, backend_updates=0),
                    optimiser, batch; width,
                )
                step_stats = backend_api.train_step!(learner, batch)
            else
                rethrow()
            end
        end
        hasproperty(step_stats, :recompiled) && Bool(step_stats.recompiled) && error(
            "fixed-shape RL update recompiled",
        )
        hasproperty(step_stats, :step) && Int(step_stats.step) == control.update + 1 || error(
            "backend step differs from the next RL update",
        )
        priority_errors = Float32.(vec(Array(step_stats.priority_errors)))
        length(priority_errors) == batch_size || error(
            "compiled objective returned the wrong PER priority count",
        )
        all(isfinite, priority_errors) || error("compiled objective returned non-finite TD errors")
        update_priorities!(replay, indices, priority_errors)
        control.update += 1
        control.update_credit -= 1
        push!(interval_samples, (;
            qr_loss=_host_scalar(step_stats.qr_loss),
            teacher_loss=_host_scalar(step_stats.teacher_loss),
            total_loss=Float64(step_stats.loss),
            mean_absolute_td=mean(abs, priority_errors),
            mean_return=mean(transition.return_n for transition in transitions),
            teacher_weight=Float64(batch.targets.teacher_weight[1]),
            update_wall_seconds=time() - iteration_started,
            compiled_learner_wall_seconds=Float64(step_stats.wall_seconds),
        ))
        control.learner_iteration_wall_seconds += time() - iteration_started

        checkpoint_now = control.update % checkpoint_interval == 0 || control.update == max_updates
        if !checkpoint_now && control.update % target_refresh_interval == 0
            online, ema, last_host_refresh_update = _refresh_online_ema(
                backend_api, learner, ema, last_host_refresh_update,
                control.update, ema_tau,
            )
        end
        checkpoint_now || continue
        snapshot = _snapshot(backend_api, learner)
        snapshot.step == control.update || error("checkpoint step differs from RL update")
        if last_host_refresh_update < control.update
            elapsed = control.update - last_host_refresh_update
            effective_tau = Float32(1 - (1 - Float64(ema_tau))^elapsed)
            ema_parameters!(ema.ps, snapshot.ps; tau=effective_tau)
            online = (; ps=snapshot.ps, st=snapshot.st)
            ema = (; ps=ema.ps, st=snapshot.st)
            last_host_refresh_update = control.update
        end
        metrics = (;
            update=control.update, replay_count=replay.count,
            replay_insertions=replay.insertions, actor_episodes=control.actor_episodes,
            actor_steps=control.actor_steps, next_train_seed=control.next_train_seed,
            actor_wall_seconds=control.actor_wall_seconds,
            teacher_seconds=control.teacher_seconds,
            student_seconds=control.student_seconds,
            learner_iteration_wall_seconds=control.learner_iteration_wall_seconds,
            end_to_end_wall_seconds=control.actor_wall_seconds + control.learner_iteration_wall_seconds,
            end_to_end_updates_per_second=control.update / max(
                control.actor_wall_seconds + control.learner_iteration_wall_seconds,
                eps(Float64),
            ),
            update_credit=control.update_credit, effective_backend=control.effective_backend,
            target_refresh_interval, last_host_refresh_update,
            interval=_metric_summary(interval_samples), parameter_norm=parameter_norm(snapshot.ps),
            ema_parameter_norm=parameter_norm(ema.ps), gradient_witness,
            last_rollout,
            held_out_dev_validation_sealed_metrics_used=false,
        )
        push!(history, metrics)
        artifact = _save_checkpoint!(
            latest_path, promoted.kind, snapshot, ema, replay, sampling_rng, control,
            history, config, config_sha256, source_hashes, gradient_witness, metrics,
        )
        open(metrics_path, "a") do io
            JSON3.write(io, merge(metrics, (; checkpoint=artifact)))
            write(io, '\n')
        end
        empty!(interval_samples)
        @info "RL checkpoint" update=control.update replay=replay.count backend=control.effective_backend qr_loss=metrics.interval.qr_loss checkpoint=artifact.path
    end

    final = last(history)
    summary = (;
        generated_at=string(now()), config, config_sha256,
        completed_updates=control.update, final,
        latest_checkpoint=latest_path, latest_checkpoint_sha256=sha256_file(latest_path),
        metrics_jsonl=metrics_path,
        next_action="development-game evaluation only; validation and sealed remain locked",
    )
    _write_json_atomic(summary_path, summary)
    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    rl_main()
end
