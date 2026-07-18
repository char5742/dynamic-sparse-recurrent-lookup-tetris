module BeatFirstTeacherDatasetV2

using Dates
using JLD2
using JSON3
using Printf
using SHA

export MAX_CANDIDATES,
       EpisodeSpec,
       allocate_episode,
       append_state!,
       episode_key,
       part_basename,
       save_episode_part!,
       recover_part_summary!,
       load_manifest,
       reconcile_part!,
       manifest_counts,
       write_manifest!

# Canonical `get_node_list` concatenates current and HOLD candidates without a
# cross-piece deduplication. Each piece has at most 34 direct landings and each
# can contribute the base plus two rotation-tuck results: 2 * 34 * 3 = 204.
# Round storage to the next NPU batch-16 boundary. Learners trim this physical
# capacity to the dataset's observed maximum before compiling.
const MAX_CANDIDATES = 208
const FORMAT_VERSION = 2

struct EpisodeSpec
    split::Symbol
    role::Symbol
    seed::Int
    epsilon::Float64
end

function EpisodeSpec(split::Symbol, role::Symbol, seed::Integer; epsilon::Real=0.0)
    split in (:train, :validation) || throw(ArgumentError("unknown split: $split"))
    role in (:old_policy, :epsilon, :dagger) ||
        throw(ArgumentError("unknown rollout role: $role"))
    0.0 <= epsilon <= 1.0 || throw(ArgumentError("epsilon must be in [0,1]"))
    role === :epsilon || epsilon == 0 || throw(
        ArgumentError("epsilon is only valid for epsilon rollouts"),
    )
    return EpisodeSpec(split, role, Int(seed), Float64(epsilon))
end

function allocate_episode(max_steps::Int; max_candidates::Int=MAX_CANDIDATES)
    max_steps > 0 || throw(ArgumentError("max_steps must be positive"))
    max_candidates == MAX_CANDIDATES || error(
        "beat-first v1 freezes storage capacity at $MAX_CANDIDATES",
    )
    return (;
        boards=zeros(UInt8, 24, 10, 1, max_steps),
        placements=zeros(UInt8, 24, 10, 1, max_candidates, max_steps),
        ren=zeros(Float32, 1, max_steps),
        back_to_back=zeros(Float32, 1, max_steps),
        tspin=zeros(Float32, max_candidates, max_steps),
        queues=zeros(UInt8, 7, 6, max_steps),
        teacher_q=fill(Float32(NaN), max_candidates, max_steps),
        teacher_rank=zeros(Int16, max_candidates, max_steps),
        action_counts=zeros(Int16, max_steps),
        selected_actions=zeros(Int16, max_steps),
        top1_actions=zeros(Int16, max_steps),
        top2_actions=zeros(Int16, max_steps),
        top1_top2_margin=zeros(Float32, max_steps),
        line_clear=zeros(Int8, max_candidates, max_steps),
        death=falses(max_candidates, max_steps),
        max_height=zeros(Int8, max_candidates, max_steps),
        holes=zeros(Int16, max_candidates, max_steps),
        cavities=zeros(Int16, max_candidates, max_steps),
        rewards=zeros(Float32, max_steps),
        seed_ids=zeros(Int64, max_steps),
        episode_ids=zeros(Int32, max_steps),
        episode_steps=zeros(Int16, max_steps),
        terminal=falses(max_steps),
        scores_after=zeros(Int32, max_steps),
        behavior_exploratory=falses(max_steps),
    )
end

function append_state!(
    data,
    row::Int,
    state,
    input,
    teacher_scores,
    ordered_nodes,
    selected::Int,
    episode_id::Int,
    exploratory::Bool,
    geometry_function::Function,
    apply_node_function::Function,
)
    1 <= row <= size(data.boards, 4) || throw(BoundsError(data.boards, row))
    count = length(ordered_nodes)
    1 <= count <= MAX_CANDIDATES || error(
        "candidate count $count is outside fixed width 1:$MAX_CANDIDATES",
    )
    length(teacher_scores) == count || throw(DimensionMismatch(
        "teacher returned $(length(teacher_scores)) values for $count candidates",
    ))
    all(isfinite, teacher_scores) || error("teacher returned non-finite values")
    1 <= selected <= count || throw(BoundsError(ordered_nodes, selected))

    data.boards[:, :, :, row] .= UInt8.(input[1][:, :, :, 1])
    data.placements[:, :, :, 1:count, row] .= UInt8.(input[2])
    data.ren[1, row] = input[3][1, 1]
    data.back_to_back[1, row] = input[4][1, 1]
    data.tspin[1:count, row] .= vec(input[5])
    data.queues[:, :, row] .= UInt8.(input[6][:, :, 1])
    data.teacher_q[1:count, row] .= teacher_scores
    data.action_counts[row] = count
    data.selected_actions[row] = selected
    data.seed_ids[row] = episode_id
    data.episode_ids[row] = episode_id
    data.episode_steps[row] = row
    data.behavior_exploratory[row] = exploratory

    # MergeSort makes equal-Q ranks follow the unchanged canonical candidate
    # order. Candidate arrays themselves are never sorted or deduplicated.
    ordering = sortperm(teacher_scores; rev=true, alg=MergeSort)
    for (rank, action) in enumerate(ordering)
        data.teacher_rank[action, row] = rank
    end
    data.top1_actions[row] = ordering[1]
    data.top2_actions[row] = ordering[min(2, count)]
    data.top1_top2_margin[row] = Float32(
        teacher_scores[ordering[1]] - teacher_scores[ordering[min(2, count)]],
    )

    root_blocks = sum(state.current_game_board.binary)
    for action in 1:count
        node = ordered_nodes[action]
        after_board = node.game_state.current_game_board.binary
        cleared = (root_blocks + 4 - sum(after_board)) ÷ 10
        0 <= cleared <= 4 || error("invalid line-clear target $cleared")
        geometry = geometry_function(after_board)
        data.line_clear[action, row] = cleared
        data.death[action, row] = node.game_state.game_over_flag
        data.max_height[action, row] = geometry.max_height
        data.holes[action, row] = geometry.holes
        data.cavities[action, row] = geometry.cavities
    end

    previous_score = state.score
    apply_node_function(state, ordered_nodes[selected])
    data.rewards[row] = Float32(state.score - previous_score) / 600.0f0
    data.terminal[row] = state.game_over_flag
    data.scores_after[row] = state.score
    return selected
end

function episode_key(spec::EpisodeSpec; student_checkpoint_sha256::AbstractString="")
    epsilon = reinterpret(UInt64, spec.epsilon)
    return join((
        "v$(FORMAT_VERSION)", spec.split, spec.role, spec.seed,
        string(epsilon; base=16, pad=16), student_checkpoint_sha256,
    ), '|')
end

function part_basename(spec::EpisodeSpec; student_checkpoint_sha256::AbstractString="")
    epsilon_tag = spec.role === :epsilon ?
                  "__eps" * replace(@sprintf("%.3f", spec.epsilon), "." => "p") : ""
    student_tag = isempty(student_checkpoint_sha256) ? "" :
                  "__student" * first(student_checkpoint_sha256, 12)
    return "part__$(spec.split)__$(spec.role)__seed$(spec.seed)$(epsilon_tag)$(student_tag)"
end

function _subset_last(array, rows)
    return copy(selectdim(array, ndims(array), rows))
end

function _atomic_write(writer::Function, path::AbstractString)
    mkpath(dirname(path))
    temporary = path * ".tmp"
    isfile(temporary) && rm(temporary; force=true)
    writer(temporary)
    mv(temporary, path; force=true)
    return path
end

hex_sha256_file(path::AbstractString) = bytes2hex(open(sha256, path))

function save_episode_part!(
    output_root::AbstractString,
    spec::EpisodeSpec,
    data,
    rows::Int,
    metadata;
    student_checkpoint_sha256::AbstractString="",
)
    rows > 0 || error("refusing to save an empty episode")
    basename = part_basename(spec; student_checkpoint_sha256)
    part_path = joinpath(output_root, "parts", basename * ".jld2")
    summary_path = joinpath(output_root, "parts", basename * ".json")
    isfile(part_path) && error("part already exists: $part_path")
    row_range = 1:rows
    part_metadata = merge(metadata, (;
        format_version=FORMAT_VERSION,
        episode_key=episode_key(spec; student_checkpoint_sha256),
        split=String(spec.split),
        role=String(spec.role),
        seed=spec.seed,
        epsilon=spec.epsilon,
        row_count=rows,
        candidate_total=sum(Int, data.action_counts[row_range]),
        max_candidates=MAX_CANDIDATES,
        preserves_candidate_order=true,
        preserves_candidate_multiplicity=true,
        teacher_tail_semantics="static NPU batches of 16 plus actual-size dynamic CPU tail",
        short_empirical_return_saved=false,
    ))
    _atomic_write(part_path) do temporary
        jldsave(
            temporary;
            boards=_subset_last(data.boards, row_range),
            placements=_subset_last(data.placements, row_range),
            ren=_subset_last(data.ren, row_range),
            back_to_back=_subset_last(data.back_to_back, row_range),
            tspin=_subset_last(data.tspin, row_range),
            queues=_subset_last(data.queues, row_range),
            teacher_q=_subset_last(data.teacher_q, row_range),
            teacher_rank=_subset_last(data.teacher_rank, row_range),
            action_counts=data.action_counts[row_range],
            selected_actions=data.selected_actions[row_range],
            top1_actions=data.top1_actions[row_range],
            top2_actions=data.top2_actions[row_range],
            top1_top2_margin=data.top1_top2_margin[row_range],
            line_clear=_subset_last(data.line_clear, row_range),
            death=_subset_last(data.death, row_range),
            max_height=_subset_last(data.max_height, row_range),
            holes=_subset_last(data.holes, row_range),
            cavities=_subset_last(data.cavities, row_range),
            rewards=data.rewards[row_range],
            seed_ids=data.seed_ids[row_range],
            episode_ids=data.episode_ids[row_range],
            episode_steps=data.episode_steps[row_range],
            terminal=data.terminal[row_range],
            scores_after=data.scores_after[row_range],
            behavior_exploratory=data.behavior_exploratory[row_range],
            metadata=part_metadata,
        )
    end
    part_sha256 = hex_sha256_file(part_path)
    summary = merge(part_metadata, (;
        relative_path=relpath(part_path, output_root),
        bytes=filesize(part_path),
        sha256=part_sha256,
    ))
    _atomic_write(summary_path) do temporary
        open(temporary, "w") do io
            JSON3.pretty(io, summary)
        end
    end
    return summary
end

"""Recover the small sidecar if interruption occurred after the JLD2 rename."""
function recover_part_summary!(
    output_root::AbstractString,
    part_path::AbstractString,
    summary_path::AbstractString,
)
    metadata = jldopen(part_path, "r") do file
        file["metadata"]
    end
    summary = merge(metadata, (;
        relative_path=relpath(part_path, output_root),
        bytes=filesize(part_path),
        sha256=hex_sha256_file(part_path),
    ))
    _atomic_write(summary_path) do temporary
        open(temporary, "w") do io
            JSON3.pretty(io, summary)
        end
    end
    return summary
end

function load_manifest(output_root::AbstractString)
    path = joinpath(output_root, "manifest.json")
    if !isfile(path)
        return (;
            format_version=FORMAT_VERSION,
            created_at=string(now()),
            updated_at=string(now()),
            parts=Any[],
        )
    end
    manifest = JSON3.read(read(path, String))
    Int(manifest.format_version) == FORMAT_VERSION || error(
        "unsupported manifest format $(manifest.format_version)",
    )
    return (;
        format_version=FORMAT_VERSION,
        created_at=String(manifest.created_at),
        updated_at=String(manifest.updated_at),
        parts=Any[part for part in manifest.parts],
    )
end

function _part_key(part)
    return String(part.episode_key)
end

function reconcile_part!(manifest, output_root::AbstractString, summary)
    key = String(summary.episode_key)
    existing = findfirst(part -> _part_key(part) == key, manifest.parts)
    if !isnothing(existing)
        String(manifest.parts[existing].sha256) == String(summary.sha256) || error(
            "episode key $key maps to two different part hashes",
        )
        return false
    end
    path = joinpath(output_root, String(summary.relative_path))
    isfile(path) || error("manifest part does not exist: $path")
    hex_sha256_file(path) == String(summary.sha256) || error(
        "part hash mismatch while reconciling $path",
    )
    push!(manifest.parts, summary)
    return true
end

function manifest_counts(manifest)
    counts = Dict{String,Int}()
    candidate_total = 0
    for part in manifest.parts
        split = String(part.split)
        role = String(part.role)
        rows = Int(part.row_count)
        counts["states.$split"] = get(counts, "states.$split", 0) + rows
        counts["states.$split.$role"] = get(counts, "states.$split.$role", 0) + rows
        counts["episodes.$split"] = get(counts, "episodes.$split", 0) + 1
        counts["episodes.$split.$role"] = get(counts, "episodes.$split.$role", 0) + 1
        candidate_total += Int(part.candidate_total)
    end
    counts["states.total"] = sum(
        (
            value for (key, value) in counts if startswith(key, "states.") &&
            count(==('.'), key) == 1
        );
        init=0,
    )
    counts["episodes.total"] = length(manifest.parts)
    counts["candidates.total"] = candidate_total
    return counts
end

function write_manifest!(output_root::AbstractString, manifest; run_metadata=(;))
    counts = manifest_counts(manifest)
    ordered_parts = sort(
        manifest.parts;
        by=part -> (String(part.split), String(part.role), Int(part.seed)),
    )
    payload = (;
        format_version=FORMAT_VERSION,
        created_at=manifest.created_at,
        updated_at=string(now()),
        counts,
        run_metadata,
        parts=ordered_parts,
    )
    path = joinpath(output_root, "manifest.json")
    _atomic_write(path) do temporary
        open(temporary, "w") do io
            JSON3.pretty(io, payload)
        end
    end
    return payload
end

end # module
