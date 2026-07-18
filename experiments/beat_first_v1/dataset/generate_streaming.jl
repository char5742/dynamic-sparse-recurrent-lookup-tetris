ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = get(
    ENV,
    "JULIA_PYTHONCALL_EXE",
    raw"D:\tetris-paper-plus\python-env\Scripts\python.exe",
)

using Dates
using JSON3
using PythonCall
using Random
using SHA

const DATASET_DIR = @__DIR__
const EXPERIMENT_DIR = normpath(joinpath(DATASET_DIR, ".."))
const REPOSITORY_ROOT = normpath(joinpath(EXPERIMENT_DIR, "..", ".."))

# Canonical engine ordering, old-checkpoint inputs, state mutation, and the
# historical NPU16 + actual-size CPU-tail scorer are reused verbatim.
include(joinpath(REPOSITORY_ROOT, "scripts", "evaluate_openvino_checkpoint.jl"))
include(joinpath(EXPERIMENT_DIR, "training", "core.jl"))
include(joinpath(DATASET_DIR, "schema.jl"))
using .BeatFirstTeacherDatasetV2

const STUDENT_CHECKPOINT = strip(get(ENV, "BEAT_DATASET_STUDENT_CHECKPOINT", ""))
const STUDENT_ADAPTER_SOURCE = abspath(get(
    ENV,
    "BEAT_DATASET_STUDENT_ADAPTER",
    joinpath(EXPERIMENT_DIR, "evaluation", "beat_first_adapter.jl"),
))
if !isempty(STUDENT_CHECKPOINT)
    isfile(STUDENT_ADAPTER_SOURCE) || error(
        "student adapter does not exist: $STUDENT_ADAPTER_SOURCE",
    )
    include(STUDENT_ADAPTER_SOURCE)
end

function _fingerprint(path::AbstractString)
    absolute = abspath(path)
    isfile(absolute) || error("required file does not exist: $absolute")
    return (;
        path=absolute,
        bytes=filesize(absolute),
        sha256=bytes2hex(open(sha256, absolute)),
    )
end

function _interleave(left, right)
    result = EpisodeSpec[]
    for index in 1:max(length(left), length(right))
        index <= length(left) && push!(result, left[index])
        index <= length(right) && push!(result, right[index])
    end
    return result
end

"""Frozen base plan: validation first, then a train mixture and old-policy fallback.

The high seed ranges are disjoint from development 5742--5757, validation
8001--8008, and sealed 91001--91032. The generator stops only after the whole
validation prefix and at least `target_train_states` training states exist.
"""
function base_episode_plan()
    validation_old = [
        EpisodeSpec(:validation, :old_policy, seed) for seed in 120_001:120_024
    ]
    epsilons = (0.05, 0.10, 0.20)
    validation_epsilon = [
        EpisodeSpec(
            :validation,
            :epsilon,
            seed;
            epsilon=epsilons[mod1(index, length(epsilons))],
        )
        for (index, seed) in enumerate(121_001:121_024)
    ]
    train_old = [
        EpisodeSpec(:train, :old_policy, seed) for seed in 100_001:100_320
    ]
    train_epsilon = [
        EpisodeSpec(
            :train,
            :epsilon,
            seed;
            epsilon=epsilons[mod1(index, length(epsilons))],
        )
        for (index, seed) in enumerate(110_001:110_200)
    ]
    # These are only consumed if epsilon trajectories terminate so early that
    # the 320+200 mixture has not reached 100k training states.
    fallback_old = [
        EpisodeSpec(:train, :old_policy, seed) for seed in 105_001:105_120
    ]
    return vcat(
        _interleave(validation_old, validation_epsilon),
        _interleave(train_old, train_epsilon),
        fallback_old,
    )
end

"""Later student-visited DAgger phase; old teacher still labels every candidate."""
function dagger_episode_plan()
    isempty(STUDENT_CHECKPOINT) && error(
        "BEAT_DATASET_STUDENT_CHECKPOINT is required for the dagger plan",
    )
    return [EpisodeSpec(:train, :dagger, seed) for seed in 130_001:130_240]
end

function _validate_plan(plan)
    keys = [episode_key(spec) for spec in plan]
    length(unique(keys)) == length(keys) || error("episode plan contains duplicate keys")
    train_seeds = Set(spec.seed for spec in plan if spec.split === :train)
    validation_seeds = Set(spec.seed for spec in plan if spec.split === :validation)
    isempty(intersect(train_seeds, validation_seeds)) || error(
        "train and validation plans share an environment seed",
    )
    return plan
end

function _load_orphan_summary(output_root::AbstractString, spec, student_sha::String)
    basename = part_basename(spec; student_checkpoint_sha256=student_sha)
    summary_path = joinpath(output_root, "parts", basename * ".json")
    part_path = joinpath(output_root, "parts", basename * ".jld2")
    if isfile(summary_path) && isfile(part_path)
        return JSON3.read(read(summary_path, String))
    elseif isfile(part_path)
        return recover_part_summary!(output_root, part_path, summary_path)
    elseif isfile(summary_path)
        error("orphan summary has no JLD2 part for $basename")
    end
    return nothing
end

function _episode_already_complete(manifest, spec, student_sha::String)
    key = episode_key(spec; student_checkpoint_sha256=student_sha)
    return any(part -> String(part.episode_key) == key, manifest.parts)
end

function _select_behavior(spec, behavior_rng, old_scores, student, state, nodes)
    old_action = argmax(old_scores)
    if spec.role === :old_policy
        return old_action, false
    elseif spec.role === :epsilon
        exploratory = rand(behavior_rng) < spec.epsilon
        return exploratory ? rand(behavior_rng, eachindex(nodes)) : old_action,
               exploratory
    elseif spec.role === :dagger
        result = Main.BeatFirstCandidateAdapter.score_candidate_set(
            student, state, nodes,
        )
        scores = Float32.(result.scores)
        length(scores) == length(nodes) || error(
            "student returned the wrong candidate count",
        )
        all(isfinite, scores) || error("student returned non-finite values")
        return argmax(scores), false
    end
    error("unsupported role $(spec.role)")
end

function _boolean_environment(name::AbstractString, default::Bool=false)
    value = lowercase(strip(get(ENV, name, string(default))))
    value in ("true", "1", "yes") && return true
    value in ("false", "0", "no") && return false
    error("$name must be true or false, observed $(repr(value))")
end

function _teacher_scores(inference, input; overlap_tail::Bool)
    !overlap_tail && return openvino_scores(inference, input)
    # PythonCall's embedded CPython remains on this generator's main Julia
    # thread.  Only the CPU InferRequest runs asynchronously inside OpenVINO.
    return lock(PYTHON_INFERENCE_LOCK) do
        pyconvert(
            Vector{Float32},
            inference.predict_overlap_tail(
                input[1], input[2], input[3], input[4], input[5], input[6],
            ),
        )
    end
end

function _generate_episode(
    spec,
    inference,
    student;
    max_steps::Int,
    next_count::Int,
    overlap_tail::Bool,
)
    data = allocate_episode(max_steps)
    state = GameState(Xoshiro(spec.seed))
    # Exploration must not perturb the piece RNG owned by GameState.
    behavior_seed = UInt64(spec.seed) ⊻ 0x626561745f646174
    behavior_rng = Xoshiro(behavior_seed)
    row = 0
    inference_seconds = 0.0
    generation_seconds = 0.0
    started = time()
    while !state.game_over_flag && row < max_steps
        generation_seconds += @elapsed nodes = stable_node_list(state)
        isempty(nodes) && break
        length(nodes) <= MAX_CANDIDATES || error(
            "seed $(spec.seed) produced $(length(nodes)) candidates, above proven storage capacity $MAX_CANDIDATES",
        )
        input = legacy_candidate_batch(state, nodes; next_count)
        inference_seconds += @elapsed old_scores = _teacher_scores(
            inference, input; overlap_tail,
        )
        selected, exploratory = _select_behavior(
            spec, behavior_rng, old_scores, student, state, nodes,
        )
        row += 1
        append_state!(
            data,
            row,
            state,
            input,
            old_scores,
            nodes,
            selected,
            spec.seed,
            exploratory,
            BeatFirstTrainingCore._geometry,
            apply_node!,
        )
    end
    return data, row, (;
        generated_at=string(now()),
        final_score=state.score,
        game_over=state.game_over_flag,
        generation_seconds,
        inference_seconds,
        wall_seconds=time() - started,
        exploratory_actions=count(data.behavior_exploratory[1:row]),
    )
end

function main()
    output_root = abspath(get(
        ENV,
        "BEAT_DATASET_ROOT",
        raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3",
    ))
    plan_name = lowercase(get(ENV, "BEAT_DATASET_PLAN", "base"))
    plan_name in ("base", "dagger") || error("BEAT_DATASET_PLAN must be base or dagger")
    plan_name == "base" && !isempty(STUDENT_CHECKPOINT) && error(
        "clear BEAT_DATASET_STUDENT_CHECKPOINT for the base plan",
    )
    target_train_states = parse(Int, get(ENV, "BEAT_DATASET_TARGET_TRAIN_STATES", "100000"))
    max_steps = parse(Int, get(ENV, "BEAT_DATASET_MAX_STEPS", "250"))
    next_count = parse(Int, get(ENV, "BEAT_DATASET_NEXT", "5"))
    next_count == 5 || error("beat-first teacher data freezes NEXT at 5")
    device = get(ENV, "OPENVINO_DEVICE", "NPU")
    teacher_batch = parse(Int, get(ENV, "OPENVINO_BATCH", "16"))
    teacher_batch == 16 || error("historical teacher semantics require OPENVINO_BATCH=16")
    overlap_tail = _boolean_environment("BEAT_DATASET_OVERLAP_TAIL", false)
    overlap_tail && device != "NPU" && error(
        "BEAT_DATASET_OVERLAP_TAIL is gated only for the NPU plus dynamic CPU-tail path",
    )
    max_new_episodes = parse(Int, get(ENV, "BEAT_DATASET_MAX_NEW_EPISODES", "1000000"))

    old_checkpoint = _fingerprint(joinpath(
        REPOSITORY_ROOT, "1313", "mainmodel copy 3.jld2",
    ))
    old_openvino_weights = _fingerprint(joinpath(
        REPOSITORY_ROOT,
        "artifacts",
        "legacy_openvino",
        "legacy_1313_weights.npz",
    ))
    student_fingerprint = isempty(STUDENT_CHECKPOINT) ? nothing :
                          _fingerprint(STUDENT_CHECKPOINT)
    student_sha = isnothing(student_fingerprint) ? "" : student_fingerprint.sha256
    plan = _validate_plan(
        plan_name == "base" ? base_episode_plan() : dagger_episode_plan(),
    )

    sys = pyimport("sys")
    sys.path.insert(0, joinpath(REPOSITORY_ROOT, "tools"))
    legacy_openvino = pyimport("legacy_openvino")

    mkpath(joinpath(output_root, "parts"))
    manifest = load_manifest(output_root)
    for part in manifest.parts
        path = joinpath(output_root, String(part.relative_path))
        isfile(path) || error("manifest references a missing part: $path")
    end
    # A crash after atomic part+summary creation but before manifest replacement
    # is recovered without replaying or duplicating the episode.
    for spec in plan
        orphan = _load_orphan_summary(output_root, spec, student_sha)
        isnothing(orphan) || reconcile_part!(manifest, output_root, orphan)
    end
    run_metadata = (;
        repository_root=REPOSITORY_ROOT,
        git_commit=readchomp(`git -C $REPOSITORY_ROOT rev-parse HEAD`),
        julia_version=string(VERSION),
        openvino_version=pyconvert(String, pyimport("openvino").__version__),
        old_checkpoint,
        old_openvino_weights,
        student_checkpoint=student_fingerprint,
        student_adapter=(isnothing(student_fingerprint) ? nothing : _fingerprint(STUDENT_ADAPTER_SOURCE)),
        plan=plan_name,
        target_train_states,
        max_steps,
        next_count,
        device,
        teacher_batch,
        overlap_tail,
        teacher_execution_schedule=(overlap_tail ?
            "static NPU batch-16 calls in canonical order with the unchanged actual-size dynamic CPU tail executing concurrently" :
            "serial static NPU batch-16 calls followed by the unchanged actual-size dynamic CPU tail"),
        held_out_development_validation_sealed_seeds_used=false,
    )
    manifest = write_manifest!(output_root, manifest; run_metadata)

    initial_counts = Dict{String,Int}(
        String(key) => Int(value) for (key, value) in pairs(manifest.counts)
    )
    base_train_states = get(initial_counts, "states.train.old_policy", 0) +
                        get(initial_counts, "states.train.epsilon", 0)
    validation_states = get(initial_counts, "states.validation", 0)
    if plan_name == "dagger"
        base_train_states >= target_train_states || error(
            "DAgger requires the completed base dataset: $base_train_states / $target_train_states training states",
        )
        validation_states > 0 || error("DAgger requires a frozen validation split")
    end
    all_complete = all(
        _episode_already_complete(manifest, spec, student_sha) for spec in plan
    )
    if all_complete || (plan_name == "base" && base_train_states >= target_train_states &&
                        all(spec.split !== :validation ||
                            _episode_already_complete(manifest, spec, student_sha) for spec in plan))
        @info "Dataset plan is already complete; no inference backend compiled" counts=manifest.counts
        return
    end

    @info "Compiling frozen old teacher" device teacher_batch overlap_tail
    inference = legacy_openvino.LegacyOpenVINOInference(device, teacher_batch)
    student = isnothing(student_fingerprint) ? nothing :
              Main.BeatFirstCandidateAdapter.build_candidate_policy(STUDENT_CHECKPOINT)

    new_episodes = 0
    for spec in plan
        counts = Dict{String,Int}(String(key) => Int(value) for (key, value) in pairs(manifest.counts))
        train_states = get(counts, "states.train.old_policy", 0) +
                       get(counts, "states.train.epsilon", 0)
        validation_complete = all(
            _episode_already_complete(manifest, candidate, student_sha)
            for candidate in plan if candidate.split === :validation
        )
        if plan_name == "base" && validation_complete &&
           train_states >= target_train_states
            @info "Teacher dataset target reached" train_states target_train_states
            break
        end
        new_episodes >= max_new_episodes && break
        _episode_already_complete(manifest, spec, student_sha) && continue

        data, rows, episode_metadata = _generate_episode(
            spec, inference, student; max_steps, next_count, overlap_tail,
        )
        rows > 0 || error("seed $(spec.seed) produced an empty episode")
        summary = save_episode_part!(
            output_root,
            spec,
            data,
            rows,
            merge(run_metadata, episode_metadata);
            student_checkpoint_sha256=student_sha,
        )
        reconcile_part!(manifest, output_root, summary)
        manifest = write_manifest!(output_root, manifest; run_metadata)
        new_episodes += 1
        @info "Saved bounded teacher episode part" split=spec.split role=spec.role seed=spec.seed rows final_score=episode_metadata.final_score counts=manifest.counts
    end
    @info "Streaming teacher generation stopped cleanly" output_root new_episodes counts=manifest.counts
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
