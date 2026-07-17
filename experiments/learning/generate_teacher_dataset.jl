ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = get(
    ENV,
    "JULIA_PYTHONCALL_EXE",
    raw"D:\tetris-paper-plus\python-env\Scripts\python.exe",
)

const EXPERIMENT_DIR = @__DIR__
const REPOSITORY_ROOT = normpath(joinpath(EXPERIMENT_DIR, "..", ".."))
include(joinpath(REPOSITORY_ROOT, "scripts", "evaluate_openvino_checkpoint.jl"))
include(joinpath(EXPERIMENT_DIR, "compact_model.jl"))
include(joinpath(EXPERIMENT_DIR, "provenance.jl"))

using Dates
using JLD2
using JSON3
using PythonCall

function allocate_teacher_dataset(max_actions::Int, capacity::Int)
    return (;
        boards=zeros(UInt8, 24, 10, 1, capacity),
        placements=zeros(UInt8, 24, 10, 1, max_actions, capacity),
        ren=zeros(Float32, 1, capacity),
        back_to_back=zeros(Float32, 1, capacity),
        tspin=zeros(Float32, max_actions, capacity),
        queues=zeros(UInt8, 7, 6, capacity),
        teacher_q=fill(Float32(NaN), max_actions, capacity),
        action_counts=zeros(Int16, capacity),
        selected_actions=zeros(Int16, capacity),
        rewards=zeros(Float32, capacity),
        episode_ids=zeros(Int32, capacity),
        episode_steps=zeros(Int16, capacity),
        terminal=falses(capacity),
        scores_after=zeros(Int32, capacity),
    )
end

function record_state!(
    data, row, state, input, scores, nodes, episode_id, step, selected::Int
)
    count = length(nodes)
    count <= size(data.placements, 4) || error(
        "candidate count $count exceeds fixed max_actions=$(size(data.placements, 4))"
    )
    data.boards[:, :, :, row] .= UInt8.(input[1][:, :, :, 1])
    data.placements[:, :, :, 1:count, row] .= UInt8.(input[2])
    data.ren[1, row] = input[3][1, 1]
    data.back_to_back[1, row] = input[4][1, 1]
    data.tspin[1:count, row] .= vec(input[5])
    data.queues[:, :, row] .= UInt8.(input[6][:, :, 1])
    data.teacher_q[1:count, row] .= scores
    data.action_counts[row] = count
    data.selected_actions[row] = selected
    data.episode_ids[row] = episode_id
    data.episode_steps[row] = step
    previous_score = state.score
    apply_node!(state, nodes[selected])
    data.rewards[row] = Float32(state.score - previous_score) / 600.0f0
    data.terminal[row] = state.game_over_flag
    data.scores_after[row] = state.score
    return selected
end

function load_compact_rollout(path::AbstractString)
    return jldopen(path, "r") do file
        config = file["model_config"]
        model = CompactCandidateQ(;
            channels=Int(config.channels),
            blocks=Int(config.blocks),
            spatial_channels=Int(config.spatial_channels),
        )
        return (;
            model,
            parameters=file["ps"],
            state=Lux.testmode(file["st"]),
        )
    end
end

function main()
    seed_first = parse(Int, get(ENV, "TEACHER_SEED_FIRST", "5742"))
    episodes = parse(Int, get(ENV, "TEACHER_EPISODES", "3"))
    max_steps = parse(Int, get(ENV, "TEACHER_MAX_STEPS", "40"))
    max_actions = parse(Int, get(ENV, "TEACHER_MAX_ACTIONS", "128"))
    next_count = parse(Int, get(ENV, "TEACHER_NEXT_COUNT", "5"))
    device = get(ENV, "OPENVINO_DEVICE", "NPU")
    teacher_batch = parse(Int, get(ENV, "OPENVINO_BATCH", "16"))
    rollout_policy = lowercase(get(ENV, "TEACHER_ROLLOUT_POLICY", "teacher"))
    rollout_policy in ("teacher", "compact") || error(
        "TEACHER_ROLLOUT_POLICY must be teacher or compact"
    )
    rollout_checkpoint_path = get(ENV, "COMPACT_ROLLOUT_CHECKPOINT", "")
    rollout = if rollout_policy == "compact"
        isempty(rollout_checkpoint_path) && error(
            "COMPACT_ROLLOUT_CHECKPOINT is required for compact rollout"
        )
        load_compact_rollout(rollout_checkpoint_path)
    else
        nothing
    end
    dataset_root = get(
        ENV, "LEARNING_DATASET_ROOT", raw"D:\tetris-paper-plus\datasets\learning"
    )
    mkpath(dataset_root)
    output_path = get(
        ENV,
        "TEACHER_DATASET_PATH",
        joinpath(dataset_root, "teacher_dev_smoke.jld2"),
    )
    provenance = learning_provenance(REPOSITORY_ROOT)
    teacher_checkpoint_path = joinpath(
        REPOSITORY_ROOT, "1313", "mainmodel copy 3.jld2"
    )
    teacher_checkpoint = require_file_fingerprint(teacher_checkpoint_path)
    rollout_checkpoint = rollout_policy == "compact" ?
                         require_file_fingerprint(rollout_checkpoint_path) : nothing

    sys = pyimport("sys")
    sys.path.insert(0, joinpath(REPOSITORY_ROOT, "tools"))
    legacy_openvino = pyimport("legacy_openvino")
    @info "Compiling read-only 1313 teacher" device teacher_batch
    inference = legacy_openvino.LegacyOpenVINOInference(device, teacher_batch)
    data = allocate_teacher_dataset(max_actions, episodes * max_steps)
    row = 0
    candidate_total = 0
    inference_seconds = 0.0
    generation_seconds = 0.0
    started = time()

    for episode_offset in 0:(episodes - 1)
        seed = seed_first + episode_offset
        state = GameState(Xoshiro(seed))
        for step in 1:max_steps
            state.game_over_flag && break
            generation_seconds += @elapsed nodes = stable_node_list(state)
            isempty(nodes) && break
            input = legacy_candidate_batch(state, nodes; next_count)
            inference_seconds += @elapsed scores = openvino_scores(inference, input)
            all(isfinite, scores) || error("teacher returned non-finite values")
            selected = if rollout_policy == "teacher"
                argmax(scores)
            else
                rollout_raw, _ = rollout.model(
                    input, rollout.parameters, rollout.state
                )
                rollout_scores = vec(Array(rollout_raw))
                all(isfinite, rollout_scores) || error(
                    "compact rollout returned non-finite values"
                )
                argmax(rollout_scores)
            end
            row += 1
            record_state!(
                data,
                row,
                state,
                input,
                scores,
                nodes,
                episode_offset + 1,
                step,
                selected,
            )
            candidate_total += length(nodes)
        end
        @info "Recorded teacher development episode" seed score=state.score rows=row elapsed=time()-started
    end

    row > 0 || error("teacher dataset is empty")
    metadata = (;
        format_version=1,
        generated_at=string(now()),
        provenance,
        teacher_checkpoint,
        rollout_checkpoint,
        julia_version=string(VERSION),
        lux_version=string(Base.pkgversion(Lux)),
        openvino_version=pyconvert(String, pyimport("openvino").__version__),
        device,
        teacher_batch,
        next_count,
        seed_first,
        episodes,
        max_steps,
        max_actions,
        rollout_policy,
        rollout_checkpoint_path=(rollout_policy == "compact" ? abspath(rollout_checkpoint_path) : nothing),
        row_count=row,
        candidate_total,
        generation_seconds,
        inference_seconds,
        wall_seconds=time() - started,
        held_out_test_seeds_used=false,
        complete_config=(;
            seed_first,
            episodes,
            max_steps,
            max_actions,
            next_count,
            device,
            teacher_batch,
            rollout_policy,
            output_path=abspath(output_path),
        ),
    )
    jldsave(
        output_path;
        boards=data.boards[:, :, :, 1:row],
        placements=data.placements[:, :, :, :, 1:row],
        ren=data.ren[:, 1:row],
        back_to_back=data.back_to_back[:, 1:row],
        tspin=data.tspin[:, 1:row],
        queues=data.queues[:, :, 1:row],
        teacher_q=data.teacher_q[:, 1:row],
        action_counts=data.action_counts[1:row],
        selected_actions=data.selected_actions[1:row],
        rewards=data.rewards[1:row],
        episode_ids=data.episode_ids[1:row],
        episode_steps=data.episode_steps[1:row],
        terminal=data.terminal[1:row],
        scores_after=data.scores_after[1:row],
        metadata,
    )
    open(replace(output_path, r"\.jld2$" => ".json"), "w") do io
        JSON3.pretty(io, metadata)
    end
    @info "Saved fixed-shape teacher dataset" output_path metadata
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
