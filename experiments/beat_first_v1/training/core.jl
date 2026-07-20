module BeatFirstTrainingCore

using JLD2
using JSON3
using LinearAlgebra
using Lux
using Random
using SHA
using Statistics

export MAX_CANDIDATES,
       AUX_FEATURE_NAMES,
       EpochSampler,
       next_batch!,
       sampler_snapshot,
       restore_sampler,
       sampler_consumed_states,
       atomic_jldsave,
       load_teacher_dataset,
       allocate_host_batch,
       pack_batch!,
       STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
       RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE,
       normalize_objective_mode,
       FIXED_TEACHER_TOP2_MARGIN_MODE,
       STUDENT_HARD_NEGATIVE_MARGIN_MODE,
       normalize_margin_mode,
       hard_negative_selection,
       supervised_components,
       supervised_objective,
       teacher_metrics,
       evaluation_metrics,
       promotion_key,
       optional_target_coverage

const MAX_CANDIDATES = 208

# Keep this order synchronized with BeatFirstModels. Values are deliberately
# scale-bounded so that the three small candidate networks can consume them
# without dataset-wide normalization before the first useful training run.
const AUX_FEATURE_NAMES = (
    (Symbol("column_height_$column") for column in 1:10)...,
    (Symbol("column_holes_$column") for column in 1:10)...,
    (Symbol("column_well_depth_$column") for column in 1:10)...,
    :unreachable_cavities,
    :aggregate_height,
    :skyline_bumpiness,
    :max_height,
    :ren,
    :back_to_back,
    :tspin,
)

const LISTNET_WEIGHT = 1.0f0
const OLD_Q_WEIGHT = 0.25f0
const MARGIN_WEIGHT = 0.15f0
const DEATH_WEIGHT = 0.10f0
const QUANTILE_TEACHER_WEIGHT = 0.05f0
const GEOMETRY_WEIGHT = 0.10f0
const LISTNET_TEMPERATURE = 0.50f0
const HUBER_DELTA = 1.0f0
const RAW_TEACHER_TOP_GAP_WEIGHT = 1.0f0

# Objective composition and margin-witness selection are deliberately separate.
# The raw-gap arm removes the scale-invariant standardized ListNet term and uses
# exactly one copy of the existing teacher-fixed raw top-gap Huber term.
const STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE =
    :standardized_listnet_plus_margin
const RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE = :raw_teacher_top_gap_huber

function normalize_objective_mode(mode::Symbol)
    mode in (
        STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
        RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE,
    ) || throw(ArgumentError(
        "objective mode must be standardized_listnet_plus_margin or " *
        "raw_teacher_top_gap_huber; got $mode",
    ))
    return mode
end

normalize_objective_mode(mode::AbstractString) =
    normalize_objective_mode(Symbol(strip(mode)))

const FIXED_TEACHER_TOP2_MARGIN_MODE = :fixed_teacher_top2
const STUDENT_HARD_NEGATIVE_MARGIN_MODE = :student_hard_negative

function normalize_margin_mode(mode::Symbol)
    mode in (
        FIXED_TEACHER_TOP2_MARGIN_MODE,
        STUDENT_HARD_NEGATIVE_MARGIN_MODE,
    ) || throw(ArgumentError(
        "margin mode must be fixed_teacher_top2 or student_hard_negative; got $mode",
    ))
    return mode
end

normalize_margin_mode(mode::AbstractString) =
    normalize_margin_mode(Symbol(strip(mode)))

const SHARDED_FORMAT_VERSION = 3
const REQUIRED_TRAIN_STATES = 100_000

function _environment_flag(name::AbstractString, default::Bool=false)
    raw = lowercase(strip(get(ENV, name, string(default))))
    raw in ("true", "1", "yes") && return true
    raw in ("false", "0", "no") && return false
    error("$name must be true or false; got $(repr(raw))")
end

"""Shuffle-without-replacement state-row sampler with resumable epoch position."""
mutable struct EpochSampler{R<:AbstractRNG}
    source_rows::Vector{Int}
    permutation::Vector{Int}
    cursor::Int
    completed_epochs::Int
    rng::R
end

function EpochSampler(source_rows::AbstractVector{<:Integer}, rng::AbstractRNG)
    rows = Int.(source_rows)
    isempty(rows) && throw(ArgumentError("sampler source rows cannot be empty"))
    allunique(rows) || throw(ArgumentError("sampler source rows must be unique"))
    permutation = copy(rows)
    shuffle!(rng, permutation)
    return EpochSampler(rows, permutation, 1, 0, rng)
end

function _validate_sampler!(sampler::EpochSampler)
    isempty(sampler.source_rows) && error("sampler source rows cannot be empty")
    allunique(sampler.source_rows) || error("sampler source rows are not unique")
    length(sampler.permutation) == length(sampler.source_rows) || error(
        "sampler permutation length does not match source rows",
    )
    sort(sampler.permutation) == sort(sampler.source_rows) || error(
        "sampler permutation is not a permutation of the training rows",
    )
    1 <= sampler.cursor <= length(sampler.permutation) + 1 || error(
        "sampler cursor is outside the epoch permutation",
    )
    sampler.completed_epochs >= 0 || error("sampler completed epoch count is negative")
    return sampler
end

function _begin_next_epoch!(sampler::EpochSampler)
    sampler.permutation .= sampler.source_rows
    shuffle!(sampler.rng, sampler.permutation)
    sampler.cursor = 1
    return sampler
end

"""Return one fixed batch, crossing an epoch boundary without dropping rows.

Each epoch permutation contains every source row exactly once. If a batch
straddles a boundary, its tail is taken from the next independently shuffled
epoch rather than discarding the previous epoch's remainder.
"""
function next_batch!(sampler::EpochSampler, batch_size::Int)
    batch_size > 0 || throw(ArgumentError("batch size must be positive"))
    rows = Vector{Int}(undef, batch_size)
    filled = 0
    while filled < batch_size
        sampler.cursor > length(sampler.permutation) && _begin_next_epoch!(sampler)
        available = length(sampler.permutation) - sampler.cursor + 1
        taken = min(batch_size - filled, available)
        source = sampler.cursor:(sampler.cursor + taken - 1)
        destination = (filled + 1):(filled + taken)
        rows[destination] .= @view sampler.permutation[source]
        sampler.cursor += taken
        filled += taken
        if sampler.cursor > length(sampler.permutation)
            sampler.completed_epochs += 1
        end
    end
    return rows
end

sampler_snapshot(sampler::EpochSampler) = (;
    format_version=1,
    source_rows=copy(sampler.source_rows),
    permutation=copy(sampler.permutation),
    cursor=sampler.cursor,
    completed_epochs=sampler.completed_epochs,
    rng=deepcopy(sampler.rng),
)

function restore_sampler(source_rows::AbstractVector{<:Integer}, snapshot)
    hasproperty(snapshot, :format_version) && Int(snapshot.format_version) == 1 || error(
        "unsupported or missing sampler snapshot format",
    )
    expected_rows = Int.(source_rows)
    Int.(snapshot.source_rows) == expected_rows || error(
        "checkpoint sampler rows do not match the current training split",
    )
    sampler = EpochSampler(
        expected_rows,
        Int.(snapshot.permutation),
        Int(snapshot.cursor),
        Int(snapshot.completed_epochs),
        deepcopy(snapshot.rng),
    )
    return _validate_sampler!(sampler)
end

function sampler_consumed_states(sampler::EpochSampler)
    width = length(sampler.source_rows)
    return sampler.cursor == width + 1 ?
           sampler.completed_epochs * width :
           sampler.completed_epochs * width + sampler.cursor - 1
end

"""Write a JLD2 file through a same-directory `.tmp` and atomic rename."""
function atomic_jldsave(path::AbstractString; kwargs...)
    destination = abspath(path)
    mkpath(dirname(destination))
    temporary = destination * ".tmp"
    isfile(temporary) && rm(temporary; force=true)
    try
        jldsave(temporary; kwargs...)
        mv(temporary, destination; force=true)
    catch
        isfile(temporary) && rm(temporary; force=true)
        rethrow()
    end
    return destination
end

function _load_sharded_teacher_dataset(
    root::AbstractString;
    max_candidates::Int,
    allow_partial_dataset::Bool,
)
    manifest_path = joinpath(root, "manifest.json")
    isfile(manifest_path) || error("sharded dataset manifest does not exist: $manifest_path")
    manifest = JSON3.read(read(manifest_path, String))
    hasproperty(manifest, :format_version) || error(
        "sharded dataset manifest has no format_version",
    )
    Int(manifest.format_version) == SHARDED_FORMAT_VERSION || error(
        "sharded dataset format must be $SHARDED_FORMAT_VERSION; got $(manifest.format_version)",
    )
    hasproperty(manifest, :counts) || error("sharded dataset manifest has no counts")
    manifest_counts = Dict{String,Int}(
        String(key) => Int(value) for (key, value) in pairs(manifest.counts)
    )
    training_states = get(manifest_counts, "states.train", 0)
    validation_states = get(manifest_counts, "states.validation", 0)
    if !allow_partial_dataset
        training_states >= REQUIRED_TRAIN_STATES || error(
            "directory dataset has $training_states training states; require >= $REQUIRED_TRAIN_STATES " *
            "(set BEAT_ALLOW_PARTIAL_DATASET=true only for bounded smoke tests)",
        )
        validation_states > 0 || error(
            "directory dataset has no validation states " *
            "(set BEAT_ALLOW_PARTIAL_DATASET=true only for bounded smoke tests)",
        )
    end
    parts = collect(manifest.parts)
    isempty(parts) && error("sharded teacher dataset manifest has no parts")
    states = sum(Int(part.row_count) for part in parts)
    get(manifest_counts, "states.total", states) == states || error(
        "manifest states.total does not match the sum of part row counts",
    )

    boards = zeros(UInt8, 24, 10, 1, states)
    placements = zeros(UInt8, 24, 10, 1, max_candidates, states)
    ren = zeros(Float32, 1, states)
    back_to_back = zeros(Float32, 1, states)
    tspin = zeros(Float32, max_candidates, states)
    queues = zeros(UInt8, 7, 6, states)
    teacher_q = fill(Float32(NaN), max_candidates, states)
    action_counts = zeros(Int, states)
    selected_actions = zeros(Int, states)
    rewards = zeros(Float32, states)
    seed_ids = zeros(Int, states)
    episode_ids = zeros(Int, states)
    episode_steps = zeros(Int, states)
    terminal = falses(states)
    candidate_death = falses(max_candidates, states)
    candidate_death_available = falses(states)
    line_clear = zeros(Int8, max_candidates, states)
    max_height = zeros(Int8, max_candidates, states)
    holes = zeros(Int16, max_candidates, states)
    cavities = zeros(Int16, max_candidates, states)
    predefined_split = fill(:unspecified, states)

    resolved_root = realpath(root)
    resolved_part_paths = Set{String}()
    cursor = 1
    for part in parts
        count = Int(part.row_count)
        rows = cursor:(cursor + count - 1)
        part_path = normpath(joinpath(root, String(part.relative_path)))
        isfile(part_path) || error("manifest references missing part: $part_path")
        resolved_part = realpath(part_path)
        relative_part = relpath(resolved_part, resolved_root)
        relative_components = splitpath(relative_part)
        (!isabspath(relative_part) && !isempty(relative_components) &&
         first(relative_components) != "..") || error(
            "manifest part resolves outside the dataset root: $part_path",
        )
        resolved_key = Sys.iswindows() ?
            lowercase(normpath(resolved_part)) : normpath(resolved_part)
        resolved_key in resolved_part_paths && error(
            "manifest aliases the same resolved part more than once: $part_path",
        )
        push!(resolved_part_paths, resolved_key)
        expected_bytes = Int(part.bytes)
        filesize(resolved_part) == expected_bytes || error(
            "manifest part byte count mismatch: $part_path",
        )
        expected_sha256 = lowercase(String(part.sha256))
        actual_sha256 = bytes2hex(open(sha256, resolved_part))
        actual_sha256 == expected_sha256 || error(
            "manifest part SHA-256 mismatch: $part_path",
        )
        jldopen(resolved_part, "r") do file
            part_counts = Int.(file["action_counts"])
            length(part_counts) == count || error("manifest row count mismatch: $part_path")
            maximum(part_counts) <= max_candidates || error(
                "part exceeds fixed candidate width $max_candidates: $part_path",
            )
            boards[:, :, :, rows] .= file["boards"]
            placements[:, :, :, :, rows] .= file["placements"]
            ren[:, rows] .= Float32.(file["ren"])
            back_to_back[:, rows] .= Float32.(file["back_to_back"])
            tspin[:, rows] .= Float32.(file["tspin"])
            queues[:, :, rows] .= file["queues"]
            teacher_q[:, rows] .= Float32.(file["teacher_q"])
            action_counts[rows] .= part_counts
            selected_actions[rows] .= Int.(file["selected_actions"])
            rewards[rows] .= Float32.(file["rewards"])
            seed_ids[rows] .= haskey(file, "seed_ids") ?
                Int.(file["seed_ids"]) : fill(Int(part.seed), count)
            episode_ids[rows] .= Int.(file["episode_ids"])
            episode_steps[rows] .= Int.(file["episode_steps"])
            terminal[rows] .= Bool.(file["terminal"])
            required_geometry = ("line_clear", "max_height", "holes", "cavities")
            all(name -> haskey(file, name), required_geometry) || error(
                "format-3 part is missing stored geometry targets: $part_path",
            )
            line_clear[:, rows] .= Int8.(file["line_clear"])
            max_height[:, rows] .= Int8.(file["max_height"])
            holes[:, rows] .= Int16.(file["holes"])
            cavities[:, rows] .= Int16.(file["cavities"])
            if haskey(file, "death")
                candidate_death[:, rows] .= Bool.(file["death"])
                candidate_death_available[rows] .= true
            end
        end
        predefined_split[rows] .= Symbol(String(part.split))
        cursor += count
    end
    cursor == states + 1 || error("sharded dataset materialization ended at wrong row")
    return (;
        boards,
        placements,
        ren,
        back_to_back,
        tspin,
        queues,
        teacher_q,
        action_counts,
        selected_actions,
        rewards,
        seed_ids,
        episode_ids,
        split_group_ids=seed_ids,
        predefined_split,
        episode_steps,
        terminal,
        candidate_death,
        candidate_death_available,
        line_clear,
        max_height,
        holes,
        cavities,
        source_path=abspath(root),
        manifest_path=abspath(manifest_path),
        manifest_format_version=Int(manifest.format_version),
        manifest_counts,
        part_integrity_verified=true,
        verified_part_count=length(resolved_part_paths),
        partial_dataset_allowed=allow_partial_dataset,
        geometry_cache=nothing,
    )
end

function load_teacher_dataset(
    path::AbstractString;
    max_candidates::Int=MAX_CANDIDATES,
    geometry_cache_max_states::Int=2_048,
    allow_partial_dataset::Bool=_environment_flag("BEAT_ALLOW_PARTIAL_DATASET", false),
)
    dataset = if isdir(path)
        _load_sharded_teacher_dataset(path; max_candidates, allow_partial_dataset)
    else
        jldopen(path, "r") do file
        action_counts = Int.(file["action_counts"])
        maximum(action_counts) <= max_candidates || error(
            "dataset has $(maximum(action_counts)) candidates; fixed learner width is $max_candidates",
        )
        required = (
            "boards", "placements", "ren", "back_to_back", "tspin", "queues",
            "teacher_q", "selected_actions", "rewards", "episode_ids",
            "episode_steps", "terminal",
        )
        missing = filter(name -> !haskey(file, name), required)
        isempty(missing) || error("teacher dataset is missing required fields: $(join(missing, ", "))")
        # Prefer an explicit environment/training seed grouping when supplied.
        # Older datasets only have episode_ids; an episode is still a leakage-
        # safe split group because no state from it can cross train/validation.
        split_group_ids = if haskey(file, "seed_ids")
            Int.(file["seed_ids"])
        elseif haskey(file, "episode_seeds")
            Int.(file["episode_seeds"])
        else
            Int.(file["episode_ids"])
        end
        return (;
            boards=file["boards"],
            placements=file["placements"],
            ren=Float32.(file["ren"]),
            back_to_back=Float32.(file["back_to_back"]),
            tspin=Float32.(file["tspin"]),
            queues=file["queues"],
            teacher_q=Float32.(file["teacher_q"]),
            action_counts,
            selected_actions=Int.(file["selected_actions"]),
            rewards=Float32.(file["rewards"]),
            episode_ids=Int.(file["episode_ids"]),
            split_group_ids,
            predefined_split=fill(:unspecified, length(action_counts)),
            episode_steps=Int.(file["episode_steps"]),
            terminal=Bool.(file["terminal"]),
            candidate_death=haskey(file, "death") ?
                Bool.(file["death"]) : falses(max_candidates, length(action_counts)),
            candidate_death_available=fill(haskey(file, "death"), length(action_counts)),
            line_clear=haskey(file, "line_clear") ? Int8.(file["line_clear"]) : nothing,
            max_height=haskey(file, "max_height") ? Int8.(file["max_height"]) : nothing,
            holes=haskey(file, "holes") ? Int16.(file["holes"]) : nothing,
            cavities=haskey(file, "cavities") ? Int16.(file["cavities"]) : nothing,
            source_path=abspath(path),
            manifest_path=nothing,
            manifest_format_version=nothing,
            manifest_counts=nothing,
            part_integrity_verified=nothing,
            verified_part_count=nothing,
            partial_dataset_allowed=false,
            # A cache of every candidate in a 100k-state dataset would consume
            # more memory than the model. Disable it above the configured cap.
            geometry_cache=length(action_counts) <= geometry_cache_max_states ?
                Dict{Tuple{Int,Int},Any}() : nothing,
            )
        end
    end
    states = length(dataset.action_counts)
    states > 0 || error("teacher dataset is empty")
    length(dataset.split_group_ids) == states || error("split group length mismatch")
    length(dataset.predefined_split) == states || error("predefined split length mismatch")
    all((1 .<= dataset.selected_actions) .&
        (dataset.selected_actions .<= dataset.action_counts)) || error(
        "teacher dataset contains an invalid selected action",
    )
    for row in eachindex(dataset.action_counts)
        count = dataset.action_counts[row]
        all(isfinite, @view(dataset.teacher_q[1:count, row])) || error(
            "non-finite teacher Q in state $row",
        )
        if hasproperty(dataset, :line_clear) && dataset.line_clear !== nothing
            all(value -> 0 <= value <= 4, @view(dataset.line_clear[1:count, row])) || error(
                "line-clear target outside 0:4 in state $row",
            )
            all(value -> 0 <= value <= 24, @view(dataset.max_height[1:count, row])) || error(
                "max-height target outside 0:24 in state $row",
            )
            all(value -> 0 <= value <= 240, @view(dataset.holes[1:count, row])) || error(
                "holes target outside 0:240 in state $row",
            )
            all(value -> 0 <= value <= 240, @view(dataset.cavities[1:count, row])) || error(
                "cavities target outside 0:240 in state $row",
            )
        end
    end
    return dataset
end

"""Allocate one fixed-shape host batch, reusable by native or Reactant backends."""
function allocate_host_batch(
    state_batch::Int; max_candidates::Int=MAX_CANDIDATES,
)
    state_batch > 0 || throw(ArgumentError("state_batch must be positive"))
    n = max_candidates * state_batch
    inputs = (;
        board=zeros(Float32, 24, 10, 1, n),
        candidate=zeros(Float32, 24, 10, 1, n),
        difference=zeros(Float32, 24, 10, 1, n),
        aux=zeros(Float32, length(AUX_FEATURE_NAMES), n),
        next_hold=zeros(Float32, 7, 6, n),
        local_mask=zeros(Float32, 24, 10, 1, n),
    )
    targets = (;
        teacher_q=zeros(Float32, max_candidates, state_batch),
        teacher_z=zeros(Float32, max_candidates, state_batch),
        top1_mask=zeros(Float32, max_candidates, state_batch),
        top2_mask=zeros(Float32, max_candidates, state_batch),
        margin=zeros(Float32, 1, state_batch),
        line_clear=zeros(Float32, max_candidates, state_batch),
        death=zeros(Float32, max_candidates, state_batch),
        death_mask=zeros(Float32, max_candidates, state_batch),
        max_height=zeros(Float32, max_candidates, state_batch),
        holes=zeros(Float32, max_candidates, state_batch),
        cavities=zeros(Float32, max_candidates, state_batch),
    )
    return (; inputs, targets, mask=zeros(Float32, max_candidates, state_batch))
end

@inline _flat_index(action::Int, slot::Int, width::Int) = action + (slot - 1) * width

function _clear_full_rows(board::AbstractMatrix{<:Real})
    full = vec(sum(board; dims=2) .>= size(board, 2))
    kept = findall(!, full)
    result = zeros(Float32, size(board))
    if !isempty(kept)
        first_output = size(board, 1) - length(kept) + 1
        result[first_output:end, :] .= Float32.(@view board[kept, :])
    end
    return result, count(full)
end

function _geometry(board::AbstractMatrix{<:Real})
    rows, columns = size(board)
    heights = zeros(Int, columns)
    holes_per_column = zeros(Int, columns)
    for column in 1:columns
        first_filled = findfirst(>(0), @view board[:, column])
        isnothing(first_filled) && continue
        heights[column] = rows - first_filled + 1
        holes_per_column[column] = count(==(0), @view board[first_filled:end, column])
    end

    # Empty cells reachable from the open sky are not cavities. Flood fill is
    # host-side preprocessing and stays outside the compiled learner step.
    reachable = falses(rows, columns)
    queue_r = Vector{Int}(undef, rows * columns)
    queue_c = Vector{Int}(undef, rows * columns)
    head = 1
    tail = 0
    for column in 1:columns
        if board[1, column] == 0
            tail += 1
            queue_r[tail] = 1
            queue_c[tail] = column
            reachable[1, column] = true
        end
    end
    while head <= tail
        row = queue_r[head]
        column = queue_c[head]
        head += 1
        for (next_row, next_column) in (
            (row - 1, column), (row + 1, column),
            (row, column - 1), (row, column + 1),
        )
            if 1 <= next_row <= rows && 1 <= next_column <= columns &&
               board[next_row, next_column] == 0 && !reachable[next_row, next_column]
                tail += 1
                queue_r[tail] = next_row
                queue_c[tail] = next_column
                reachable[next_row, next_column] = true
            end
        end
    end
    cavities = count((board .== 0) .& .!reachable)
    aggregate_height = sum(heights)
    bumpiness = sum(abs.(diff(heights)))
    well_depths = zeros(Int, columns)
    for column in 1:columns
        left = column == 1 ? rows : heights[column - 1]
        right = column == columns ? rows : heights[column + 1]
        well_depths[column] = max(min(left, right) - heights[column], 0)
    end
    return (;
        heights,
        max_height=maximum(heights; init=0),
        holes_per_column,
        holes=sum(holes_per_column),
        cavities,
        aggregate_height,
        bumpiness,
        well_depths,
        well_depth=sum(well_depths),
        hidden_occupancy=count(>(0), @view board[1:4, :]),
    )
end

function _fill_local_mask!(destination, difference)
    destination .= 0.0f0
    changed = findall(value -> !iszero(value), difference)
    for index in changed
        row, column = Tuple(index)
        destination[
            max(row - 1, 1):min(row + 1, size(difference, 1)),
            max(column - 1, 1):min(column + 1, size(difference, 2)),
        ] .= 1.0f0
    end
    return destination
end

function _candidate_geometry_entry(dataset, board, row::Int, action::Int)
    build = function ()
        placement = Float32.(@view dataset.placements[:, :, 1, action, row])
        combined = min.(1.0f0, board .+ placement)
        after, line_clear = _clear_full_rows(combined)
        geometry = _geometry(after)
        difference = after .- board
        local_mask = zeros(Float32, size(board))
        _fill_local_mask!(local_mask, difference)
        aux = Float32[
            geometry.heights ./ 24;
            geometry.holes_per_column ./ 24;
            geometry.well_depths ./ 24;
            geometry.cavities / 240;
            geometry.aggregate_height / 240;
            geometry.bumpiness / 216;
            geometry.max_height / 24;
            dataset.ren[1, row] / 30.0f0;
            dataset.back_to_back[1, row];
            dataset.tspin[action, row];
        ]
        return (; after, difference, local_mask, aux, line_clear, geometry)
    end
    if hasproperty(dataset, :geometry_cache) && dataset.geometry_cache !== nothing
        return get!(build, dataset.geometry_cache, (row, action))
    end
    return build()
end

"""Pack complete candidate sets without dropping or reordering any candidate."""
function pack_batch!(batch, dataset, rows::AbstractVector{Int})
    width, state_batch = size(batch.mask)
    length(rows) == state_batch || throw(DimensionMismatch(
        "row count $(length(rows)) != fixed state batch $state_batch",
    ))
    fill!(batch.mask, 0.0f0)
    for array in values(batch.inputs)
        fill!(array, 0.0f0)
    end
    for array in values(batch.targets)
        fill!(array, 0.0f0)
    end
    stored_geometry = hasproperty(dataset, :line_clear) && dataset.line_clear !== nothing &&
                      hasproperty(dataset, :max_height) && dataset.max_height !== nothing &&
                      hasproperty(dataset, :holes) && dataset.holes !== nothing &&
                      hasproperty(dataset, :cavities) && dataset.cavities !== nothing

    for (slot, row) in enumerate(rows)
        1 <= row <= length(dataset.action_counts) || throw(BoundsError(dataset.action_counts, row))
        count_actions = dataset.action_counts[row]
        count_actions <= width || error("state $row exceeds fixed candidate width")
        teacher = @view dataset.teacher_q[1:count_actions, row]
        teacher_mean = mean(teacher)
        teacher_scale = max(std(teacher; corrected=false), 1.0f-4)
        ordering = sortperm(teacher; rev=true, alg=MergeSort)
        top1 = ordering[1]
        top2 = length(ordering) >= 2 ? ordering[2] : ordering[1]
        board = Float32.(@view dataset.boards[:, :, 1, row])
        queue = Float32.(@view dataset.queues[:, :, row])

        batch.mask[1:count_actions, slot] .= 1.0f0
        batch.targets.teacher_q[1:count_actions, slot] .= teacher
        batch.targets.teacher_z[1:count_actions, slot] .= (teacher .- teacher_mean) ./ teacher_scale
        batch.targets.top1_mask[top1, slot] = 1.0f0
        batch.targets.top2_mask[top2, slot] = 1.0f0
        batch.targets.margin[1, slot] = teacher[top1] - teacher[top2]
        if hasproperty(dataset, :candidate_death_available) &&
           dataset.candidate_death_available[row]
            batch.targets.death[1:count_actions, slot] .= Float32.(
                @view dataset.candidate_death[1:count_actions, row]
            )
            batch.targets.death_mask[1:count_actions, slot] .= 1.0f0
        else
            selected = dataset.selected_actions[row]
            batch.targets.death[selected, slot] = Float32(dataset.terminal[row])
            batch.targets.death_mask[selected, slot] = 1.0f0
        end

        for action in 1:width
            flat = _flat_index(action, slot, width)
            batch.inputs.board[:, :, 1, flat] .= board
            # Invalid padded candidates are deterministic no-ops. Their mask is
            # zero, but preserving candidate-board==difference avoids feeding
            # arbitrary out-of-contract values through the compiled model.
            batch.inputs.candidate[:, :, 1, flat] .= board
            batch.inputs.next_hold[:, :, flat] .= queue
            action <= count_actions || continue
            entry = _candidate_geometry_entry(dataset, board, row, action)
            batch.inputs.candidate[:, :, 1, flat] .= entry.after
            batch.inputs.difference[:, :, 1, flat] .= entry.difference
            batch.inputs.local_mask[:, :, 1, flat] .= entry.local_mask
            batch.inputs.aux[:, flat] .= entry.aux
            if stored_geometry
                expected_line_clear = Int(dataset.line_clear[action, row])
                expected_max_height = Int(dataset.max_height[action, row])
                expected_holes = Int(dataset.holes[action, row])
                expected_cavities = Int(dataset.cavities[action, row])
                expected_line_clear == Int(entry.line_clear) &&
                    expected_max_height == Int(entry.geometry.max_height) &&
                    expected_holes == Int(entry.geometry.holes) &&
                    expected_cavities == Int(entry.geometry.cavities) || error(
                    "stored geometry target mismatch at state $row candidate $action",
                )
                batch.targets.line_clear[action, slot] = Float32(expected_line_clear)
                batch.targets.max_height[action, slot] = Float32(expected_max_height)
                batch.targets.holes[action, slot] = Float32(expected_holes)
                batch.targets.cavities[action, slot] = Float32(expected_cavities)
            else
                batch.targets.line_clear[action, slot] = Float32(entry.line_clear)
                batch.targets.max_height[action, slot] = Float32(entry.geometry.max_height)
                batch.targets.holes[action, slot] = Float32(entry.geometry.holes)
                batch.targets.cavities[action, slot] = Float32(entry.geometry.cavities)
            end
        end
    end
    return batch
end

@inline function _masked_mean(values, mask)
    return sum(values .* mask) / max(sum(mask), 1.0f0)
end

function _masked_standardize(values, mask)
    counts = max.(sum(mask; dims=1), 1.0f0)
    center = sum(values .* mask; dims=1) ./ counts
    centered = (values .- center) .* mask
    scale = sqrt.(sum(centered .^ 2; dims=1) ./ counts .+ 1.0f-4)
    return centered ./ scale
end

function _listnet_loss(predictions, teacher_z, mask)
    student_z = _masked_standardize(predictions, mask)
    invalid = (1.0f0 .- mask) .* -1.0f4
    teacher_logits = teacher_z ./ LISTNET_TEMPERATURE .+ invalid
    student_logits = student_z ./ LISTNET_TEMPERATURE .+ invalid
    teacher_logits = teacher_logits .- maximum(teacher_logits; dims=1)
    student_logits = student_logits .- maximum(student_logits; dims=1)
    teacher_exp = exp.(teacher_logits) .* mask
    teacher_probability = teacher_exp ./ max.(sum(teacher_exp; dims=1), 1.0f-12)
    student_log_probability = student_logits .-
                              log.(max.(sum(exp.(student_logits) .* mask; dims=1), 1.0f-12))
    return -sum(teacher_probability .* student_log_probability .* mask) /
           size(predictions, 2)
end

function _huber(values; delta::Float32=HUBER_DELTA)
    absolute = abs.(values)
    return ifelse.(
        absolute .<= delta,
        0.5f0 .* values .^ 2,
        delta .* (absolute .- 0.5f0 * delta),
    )
end

@inline function _stable_argmax(values, valid_indices, excluded::Int=0)
    best = 0
    for index in valid_indices
        index == excluded && continue
        if best == 0 || values[index] > values[best]
            best = index
        end
    end
    return best
end

"""Select a fixed hard-negative witness for each state.

Teacher and student ties are resolved by the lowest candidate ID. The returned
masks are deliberately computed before an AD pullback and must be passed back
to `supervised_components`; hard routing is therefore stop-gradient while the
selected `q[y]` and `q[h]` remain differentiable.
"""
function hard_negative_selection(q::AbstractMatrix, batch)
    width, state_batch = size(batch.mask)
    size(q) == (width, state_batch) || throw(DimensionMismatch(
        "hard-negative Q shape $(size(q)) != batch shape $((width, state_batch))",
    ))
    teacher = batch.targets.teacher_q
    size(teacher) == (width, state_batch) || throw(DimensionMismatch(
        "teacher-Q shape differs from the hard-negative batch",
    ))
    value_type = promote_type(eltype(q), eltype(teacher), Float32)
    teacher_top1_mask = zeros(value_type, width, state_batch)
    hard_negative_mask = zeros(value_type, width, state_batch)
    teacher_delta = zeros(value_type, 1, state_batch)
    valid_state_mask = zeros(value_type, 1, state_batch)
    teacher_top1_indices = zeros(Int, state_batch)
    hard_negative_indices = zeros(Int, state_batch)
    teacher_top2_indices = zeros(Int, state_batch)
    valid_selections = 0
    differs_from_teacher_top2 = 0

    valid_indices = Int[]
    sizehint!(valid_indices, width)
    for slot in 1:state_batch
        empty!(valid_indices)
        for candidate in 1:width
            !iszero(batch.mask[candidate, slot]) && push!(valid_indices, candidate)
        end
        isempty(valid_indices) && continue
        teacher_column = @view teacher[:, slot]
        student_column = @view q[:, slot]
        y = _stable_argmax(teacher_column, valid_indices)
        teacher_top1_indices[slot] = y
        teacher_top1_mask[y, slot] = one(value_type)
        length(valid_indices) >= 2 || continue

        h = _stable_argmax(student_column, valid_indices, y)
        teacher_top2 = _stable_argmax(teacher_column, valid_indices, y)
        h != 0 && teacher_top2 != 0 || error(
            "hard-negative selection failed for a multi-candidate state",
        )
        hard_negative_indices[slot] = h
        teacher_top2_indices[slot] = teacher_top2
        hard_negative_mask[h, slot] = one(value_type)
        teacher_delta[1, slot] = teacher_column[y] - teacher_column[h]
        valid_state_mask[1, slot] = one(value_type)
        valid_selections += 1
        differs_from_teacher_top2 += h != teacher_top2
    end
    return (;
        teacher_top1_mask,
        hard_negative_mask,
        teacher_delta,
        valid_state_mask,
        teacher_top1_indices,
        hard_negative_indices,
        teacher_top2_indices,
        valid_selections,
        differs_from_teacher_top2,
    )
end

function _ranking_margin_components(
    q,
    batch;
    margin_mode::Union{Symbol,AbstractString}=FIXED_TEACHER_TOP2_MARGIN_MODE,
    hard_negative=nothing,
    hard_negative_margin_floor::Real=0.0f0,
)
    mode = normalize_margin_mode(margin_mode)
    if mode === FIXED_TEACHER_TOP2_MARGIN_MODE
        # Preserve the original dense/default operation order exactly.
        predicted_top1 = sum(q .* batch.targets.top1_mask; dims=1)
        predicted_top2 = sum(q .* batch.targets.top2_mask; dims=1)
        margin = mean(_huber(
            predicted_top1 .- predicted_top2 .- batch.targets.margin,
        ))
        return (;
            margin_loss=margin,
            raw_top_gap_loss=margin,
            raw_student_top_gap_mean=mean(predicted_top1 .- predicted_top2),
            raw_teacher_top_gap_mean=mean(batch.targets.margin),
            hard_negative_valid_selections=0,
            hard_negative_differs_from_teacher_top2=0,
        )
    end

    hard_negative === nothing && throw(ArgumentError(
        "student_hard_negative mode requires a stop-gradient selection",
    ))
    size(hard_negative.hard_negative_mask) == size(q) || throw(DimensionMismatch(
        "hard-negative mask shape differs from Q",
    ))
    predicted_top1 = sum(q .* hard_negative.teacher_top1_mask; dims=1)
    predicted_hard_negative = sum(q .* hard_negative.hard_negative_mask; dims=1)
    required_gap = iszero(hard_negative_margin_floor) ?
        hard_negative.teacher_delta :
        max.(hard_negative.teacher_delta, convert(eltype(q), hard_negative_margin_floor))
    violation = max.(
        zero(eltype(q)),
        required_gap .-
            (predicted_top1 .- predicted_hard_negative),
    )
    margin = _masked_mean(
        _huber(violation; delta=1.0f0),
        hard_negative.valid_state_mask,
    )
    teacher_top1 = sum(q .* batch.targets.top1_mask; dims=1)
    teacher_top2 = sum(q .* batch.targets.top2_mask; dims=1)
    raw_top_gap = mean(_huber(
        teacher_top1 .- teacher_top2 .- batch.targets.margin,
    ))
    return (;
        margin_loss=margin,
        raw_top_gap_loss=raw_top_gap,
        raw_student_top_gap_mean=mean(teacher_top1 .- teacher_top2),
        raw_teacher_top_gap_mean=mean(batch.targets.margin),
        hard_negative_valid_selections=hard_negative.valid_selections,
        hard_negative_differs_from_teacher_top2=
            hard_negative.differs_from_teacher_top2,
    )
end

function _binary_cross_entropy_with_logits(logits, labels)
    # max(x,0)-xy+log1p(exp(-abs(x))) without requiring a softplus rule.
    return max.(logits, 0.0f0) .- logits .* labels .+ log1p.(exp.(-abs.(logits)))
end

function _reshape_q(output, width::Int, state_batch::Int)
    q = output.q
    length(q) == width * state_batch || error(
        "model q has $(length(q)) elements; expected $(width * state_batch)",
    )
    return reshape(q, width, state_batch)
end

function _quantile_teacher_loss(quantiles, teacher_q, mask)
    isempty(quantiles) && return zero(eltype(teacher_q))
    k = size(quantiles, 1)
    n = length(teacher_q)
    size(quantiles, 2) == n || error("quantile output has the wrong candidate count")
    target = reshape(vec(teacher_q), 1, n)
    valid = reshape(vec(mask), 1, n)
    error = target .- quantiles
    tau = reshape((Float32.(1:k) .- 0.5f0) ./ Float32(k), k, 1)
    negative = ifelse.(error .< 0.0f0, 1.0f0, 0.0f0)
    weight = abs.(tau .- negative)
    return sum(weight .* _huber(error) .* valid) / max(sum(valid) * k, 1.0f0)
end

function _geometry_losses(output, batch, width::Int, state_batch::Int)
    if !hasproperty(output, :geometry)
        zero_loss = zero(eltype(batch.targets.teacher_q))
        return (;
            geometry_loss=zero_loss,
            line_clear_loss=zero_loss,
            max_height_loss=zero_loss,
            holes_loss=zero_loss,
            cavities_loss=zero_loss,
        )
    end

    geometry = output.geometry
    size(geometry) == (4, width * state_batch) || error(
        "model geometry has size $(size(geometry)); expected $((4, width * state_batch))",
    )
    line_clear_prediction = reshape(geometry[1, :], width, state_batch)
    max_height_prediction = reshape(geometry[2, :], width, state_batch)
    holes_prediction = reshape(geometry[3, :], width, state_batch)
    cavities_prediction = reshape(geometry[4, :], width, state_batch)
    line_clear_loss = _masked_mean(
        _huber(line_clear_prediction .- batch.targets.line_clear ./ 4.0f0), batch.mask,
    )
    max_height_loss = _masked_mean(
        _huber(max_height_prediction .- batch.targets.max_height ./ 24.0f0), batch.mask,
    )
    holes_loss = _masked_mean(
        _huber(holes_prediction .- batch.targets.holes ./ 240.0f0), batch.mask,
    )
    cavities_loss = _masked_mean(
        _huber(cavities_prediction .- batch.targets.cavities ./ 240.0f0), batch.mask,
    )
    geometry_loss = (
        line_clear_loss + max_height_loss + holes_loss + cavities_loss
    ) / 4.0f0
    return (;
        geometry_loss,
        line_clear_loss,
        max_height_loss,
        holes_loss,
        cavities_loss,
    )
end

"""Return every supervised loss component from one already-computed output.

`objective_mode` controls composition independently of `margin_mode`, which
only selects the ranking witness.  The legacy/default composition is exactly
`1.0*standardized ListNet + margin_weight*margin`.  The scale-identifiable
one-shot composition is exactly `0.0*ListNet + 1.0*raw teacher top-gap Huber`
with Huber delta 1.0; old-Q and auxiliary terms are unchanged.
"""
function supervised_components(
    output,
    batch;
    margin_weight::Real=MARGIN_WEIGHT,
    margin_mode::Union{Symbol,AbstractString}=FIXED_TEACHER_TOP2_MARGIN_MODE,
    objective_mode::Union{Symbol,AbstractString}=
        STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
    hard_negative=nothing,
    hard_negative_margin_floor::Real=0.0f0,
)
    width, state_batch = size(batch.mask)
    q = _reshape_q(output, width, state_batch)
    normalized_objective_mode = normalize_objective_mode(objective_mode)
    normalized_margin_mode = normalize_margin_mode(margin_mode)
    if normalized_objective_mode === RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE
        normalized_margin_mode === FIXED_TEACHER_TOP2_MARGIN_MODE || throw(
            ArgumentError(
                "raw_teacher_top_gap_huber requires fixed_teacher_top2 margin mode",
            ),
        )
    end
    listnet = normalized_objective_mode === RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE ?
        zero(eltype(q)) : _listnet_loss(q, batch.targets.teacher_z, batch.mask)
    old_q = _masked_mean(_huber(q .- batch.targets.teacher_q), batch.mask)
    ranking_margin = _ranking_margin_components(
        q,
        batch;
        margin_mode=normalized_margin_mode,
        hard_negative,
        hard_negative_margin_floor,
    )
    margin = ranking_margin.margin_loss
    death_logits = reshape(output.death_logit, width, state_batch)
    death = _masked_mean(
        _binary_cross_entropy_with_logits(death_logits, batch.targets.death),
        batch.targets.death_mask,
    )
    quantile = _quantile_teacher_loss(output.quantiles, batch.targets.teacher_q, batch.mask)
    geometry = _geometry_losses(output, batch, width, state_batch)
    effective_listnet_weight =
        normalized_objective_mode === RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE ?
            0.0f0 : LISTNET_WEIGHT
    effective_margin_weight =
        normalized_objective_mode === RAW_TEACHER_TOP_GAP_OBJECTIVE_MODE ?
            RAW_TEACHER_TOP_GAP_WEIGHT : Float32(margin_weight)
    effective_raw_top_gap_weight =
        normalized_margin_mode === FIXED_TEACHER_TOP2_MARGIN_MODE ?
            effective_margin_weight : 0.0f0
    loss = effective_listnet_weight * listnet + OLD_Q_WEIGHT * old_q +
           effective_margin_weight * margin + DEATH_WEIGHT * death +
           QUANTILE_TEACHER_WEIGHT * quantile +
           GEOMETRY_WEIGHT * geometry.geometry_loss
    return (;
        composite_loss=loss,
        listnet_loss=listnet,
        old_q_loss=old_q,
        q_huber_loss=old_q,
        margin_loss=margin,
        raw_top_gap_loss=ranking_margin.raw_top_gap_loss,
        raw_student_top_gap_mean=ranking_margin.raw_student_top_gap_mean,
        raw_teacher_top_gap_mean=ranking_margin.raw_teacher_top_gap_mean,
        effective_listnet_weight,
        effective_margin_weight,
        effective_raw_top_gap_weight,
        hard_negative_valid_selections=
            ranking_margin.hard_negative_valid_selections,
        hard_negative_differs_from_teacher_top2=
            ranking_margin.hard_negative_differs_from_teacher_top2,
        death_loss=death,
        quantile_teacher_loss=quantile,
        geometry...,
        valid_candidates=sum(batch.mask),
    )
end

"""Shared fixed-shape objective for Native Zygote and Reactant/EnzymeMLIR."""
function supervised_objective(
    model,
    ps,
    st,
    batch;
    margin_weight::Real=MARGIN_WEIGHT,
    margin_mode::Union{Symbol,AbstractString}=FIXED_TEACHER_TOP2_MARGIN_MODE,
    objective_mode::Union{Symbol,AbstractString}=
        STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
    hard_negative=nothing,
    hard_negative_margin_floor::Real=0.0f0,
)
    output, next_state = model(batch.inputs, ps, st)
    statistics = supervised_components(
        output,
        batch;
        margin_weight,
        margin_mode,
        objective_mode,
        hard_negative,
        hard_negative_margin_floor,
    )
    loss = statistics.composite_loss
    return loss, next_state, statistics
end

function _ndcg(prediction, teacher)
    count_actions = length(teacher)
    teacher_order = sortperm(teacher; rev=true, alg=MergeSort)
    relevance = zeros(Float64, count_actions)
    for (rank, action) in enumerate(teacher_order)
        relevance[action] = count_actions - rank
    end
    prediction_order = sortperm(prediction; rev=true, alg=MergeSort)
    discount(rank) = 1.0 / log2(rank + 1.0)
    dcg = sum(relevance[action] * discount(rank) for (rank, action) in enumerate(prediction_order))
    idcg = sum((count_actions - rank) * discount(rank) for rank in 1:count_actions)
    return idcg == 0 ? 1.0 : dcg / idcg
end

function _pairwise_accuracy(prediction, teacher)
    correct = 0
    compared = 0
    for left in 1:(length(teacher) - 1), right in (left + 1):length(teacher)
        teacher_difference = teacher[left] - teacher[right]
        teacher_difference == 0 && continue
        prediction_difference = prediction[left] - prediction[right]
        correct += prediction_difference * teacher_difference > 0
        compared += 1
    end
    return compared == 0 ? 1.0 : correct / compared
end

"""Held-out teacher metrics. `predict_batch` returns a model output NamedTuple."""
function teacher_metrics(
    dataset,
    rows::AbstractVector{Int},
    host_batch,
    predict_batch::Function,
)
    top1 = Bool[]
    ndcg = Float64[]
    pairwise = Float64[]
    q_huber = Float64[]
    batch_size = size(host_batch.mask, 2)
    padded_rows = Vector{Int}(undef, batch_size)
    for partition in Iterators.partition(rows, batch_size)
        actual = collect(partition)
        for slot in 1:batch_size
            padded_rows[slot] = actual[min(slot, length(actual))]
        end
        pack_batch!(host_batch, dataset, padded_rows)
        output = predict_batch(host_batch)
        q = Array(_reshape_q(output, size(host_batch.mask)...))
        for slot in eachindex(actual)
            row = actual[slot]
            count_actions = dataset.action_counts[row]
            prediction = @view q[1:count_actions, slot]
            teacher = @view dataset.teacher_q[1:count_actions, row]
            push!(top1, argmax(prediction) == argmax(teacher))
            push!(ndcg, _ndcg(prediction, teacher))
            push!(pairwise, _pairwise_accuracy(prediction, teacher))
            push!(q_huber, mean(_huber(prediction .- teacher)))
        end
    end
    return (;
        states=length(rows),
        top1_agreement=mean(top1),
        ndcg=mean(ndcg),
        pairwise_accuracy=mean(pairwise),
        old_q_huber=mean(q_huber),
    )
end

"""Full convergence metrics for a fixed, leakage-free row set.

Rows are evaluated only in complete fixed-shape batches. Callers should pass a
deterministic subset whose length is a multiple of the state batch size.
"""
function evaluation_metrics(
    dataset,
    rows::AbstractVector{Int},
    host_batch,
    predict_batch::Function,
    ;
    margin_weight::Real=MARGIN_WEIGHT,
    margin_mode::Union{Symbol,AbstractString}=FIXED_TEACHER_TOP2_MARGIN_MODE,
    objective_mode::Union{Symbol,AbstractString}=
        STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
    hard_negative_margin_floor::Real=0.0f0,
)
    normalized_margin_mode = normalize_margin_mode(margin_mode)
    normalized_objective_mode = normalize_objective_mode(objective_mode)
    batch_size = size(host_batch.mask, 2)
    length(rows) >= batch_size || error("evaluation rows are smaller than the fixed batch")
    usable = length(rows) - mod(length(rows), batch_size)
    usable > 0 || error("no complete evaluation batch")
    selected_rows = @view rows[1:usable]

    objective_component_names = (
        :composite_loss, :listnet_loss, :old_q_loss, :q_huber_loss, :margin_loss,
        :raw_top_gap_loss, :raw_student_top_gap_mean, :raw_teacher_top_gap_mean,
        :effective_listnet_weight, :effective_margin_weight,
        :effective_raw_top_gap_weight,
        :death_loss, :quantile_teacher_loss, :geometry_loss, :line_clear_loss,
        :max_height_loss, :holes_loss, :cavities_loss,
    )
    component_names = (objective_component_names..., :diagnostic_listnet_loss)
    component_sums = Dict(name => 0.0 for name in component_names)
    top1 = Bool[]
    ndcg = Float64[]
    pairwise = Float64[]
    q_values = Float64[]
    action_margins = Float64[]
    batches = 0
    hard_negative_valid_selections = 0
    hard_negative_differs_from_teacher_top2 = 0

    for partition in Iterators.partition(selected_rows, batch_size)
        packed_rows = collect(partition)
        pack_batch!(host_batch, dataset, packed_rows)
        output = predict_batch(host_batch)
        q = Array(_reshape_q(output, size(host_batch.mask)...))
        hard_negative = normalized_margin_mode === STUDENT_HARD_NEGATIVE_MARGIN_MODE ?
            hard_negative_selection(q, host_batch) : nothing
        components = supervised_components(
            output,
            host_batch;
            margin_weight,
            margin_mode=normalized_margin_mode,
            objective_mode=normalized_objective_mode,
            hard_negative,
            hard_negative_margin_floor,
        )
        for name in objective_component_names
            component_sums[name] += Float64(getproperty(components, name))
        end
        # In raw-gap training mode ListNet has coefficient zero and is not part
        # of the VJP.  Evaluation still reports this unweighted diagnostic so
        # ranking-shape changes remain observable without reintroducing its
        # scale-invariant gradient.
        component_sums[:diagnostic_listnet_loss] += Float64(
            _listnet_loss(q, host_batch.targets.teacher_z, host_batch.mask),
        )
        hard_negative_valid_selections +=
            Int(components.hard_negative_valid_selections)
        hard_negative_differs_from_teacher_top2 +=
            Int(components.hard_negative_differs_from_teacher_top2)
        for (slot, row) in enumerate(packed_rows)
            count_actions = dataset.action_counts[row]
            prediction = @view q[1:count_actions, slot]
            teacher = @view dataset.teacher_q[1:count_actions, row]
            push!(top1, argmax(prediction) == argmax(teacher))
            push!(ndcg, _ndcg(prediction, teacher))
            push!(pairwise, _pairwise_accuracy(prediction, teacher))
            append!(q_values, prediction)
            ordering = partialsortperm(prediction, 1:min(2, count_actions); rev=true)
            push!(action_margins, count_actions >= 2 ?
                prediction[ordering[1]] - prediction[ordering[2]] : 0.0)
        end
        batches += 1
    end

    losses = NamedTuple{component_names}(
        Tuple(component_sums[name] / batches for name in component_names),
    )
    return merge(losses, (;
        states=usable,
        top1_agreement=mean(top1),
        ndcg=mean(ndcg),
        pairwise_accuracy=mean(pairwise),
        q_mean=mean(q_values),
        q_std=std(q_values; corrected=false),
        action_margin=mean(action_margins),
        hard_negative_valid_selections,
        hard_negative_differs_from_teacher_top2,
    ))
end

promotion_key(metrics) = (
    metrics.top1_agreement,
    metrics.ndcg,
    metrics.pairwise_accuracy,
    -metrics.old_q_huber,
)

optional_target_coverage(dataset) = (;
    line_clear="stored or derived for every candidate, consistency-checked when stored; trained by the normalized geometry head",
    max_height="stored or derived for every candidate, consistency-checked when stored; used in aux input and geometry head",
    holes="stored or derived for every candidate, consistency-checked when stored; used in aux input and geometry head",
    cavities="stored or derived for every candidate, consistency-checked when stored; used in aux input and geometry head",
    death=hasproperty(dataset, :candidate_death_available) && any(dataset.candidate_death_available) ?
        "all-candidate death labels where supplied; trained through death_logit" :
        "legacy selected-action terminal labels only; trained through death_logit",
)

end # module
