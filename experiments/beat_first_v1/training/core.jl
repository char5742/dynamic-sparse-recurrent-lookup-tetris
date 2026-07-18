module BeatFirstTrainingCore

using JLD2
using LinearAlgebra
using Lux
using Random
using Statistics

export MAX_CANDIDATES,
       AUX_FEATURE_NAMES,
       load_teacher_dataset,
       allocate_host_batch,
       pack_batch!,
       supervised_objective,
       teacher_metrics,
       promotion_key,
       optional_target_coverage

const MAX_CANDIDATES = 74

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
const LISTNET_TEMPERATURE = 0.50f0
const HUBER_DELTA = 1.0f0

function load_teacher_dataset(path::AbstractString; max_candidates::Int=MAX_CANDIDATES)
    dataset = jldopen(path, "r") do file
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
            episode_steps=Int.(file["episode_steps"]),
            terminal=Bool.(file["terminal"]),
            source_path=abspath(path),
            geometry_cache=Dict{Tuple{Int,Int},Any}(),
        )
    end
    states = length(dataset.action_counts)
    states > 0 || error("teacher dataset is empty")
    all((1 .<= dataset.selected_actions) .&
        (dataset.selected_actions .<= dataset.action_counts)) || error(
        "teacher dataset contains an invalid selected action",
    )
    for row in eachindex(dataset.action_counts)
        count = dataset.action_counts[row]
        all(isfinite, @view(dataset.teacher_q[1:count, row])) || error(
            "non-finite teacher Q in state $row",
        )
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
    if hasproperty(dataset, :geometry_cache)
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
        selected = dataset.selected_actions[row]
        batch.targets.death[selected, slot] = Float32(dataset.terminal[row])
        batch.targets.death_mask[selected, slot] = 1.0f0

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
            batch.targets.line_clear[action, slot] = Float32(entry.line_clear)
            batch.targets.max_height[action, slot] = Float32(entry.geometry.max_height)
            batch.targets.holes[action, slot] = Float32(entry.geometry.holes)
            batch.targets.cavities[action, slot] = Float32(entry.geometry.cavities)
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

"""Shared fixed-shape objective for Native Zygote and Reactant/EnzymeMLIR."""
function supervised_objective(model, ps, st, batch)
    output, next_state = model(batch.inputs, ps, st)
    width, state_batch = size(batch.mask)
    q = _reshape_q(output, width, state_batch)
    listnet = _listnet_loss(q, batch.targets.teacher_z, batch.mask)
    old_q = _masked_mean(_huber(q .- batch.targets.teacher_q), batch.mask)
    predicted_top1 = sum(q .* batch.targets.top1_mask; dims=1)
    predicted_top2 = sum(q .* batch.targets.top2_mask; dims=1)
    margin = mean(_huber(predicted_top1 .- predicted_top2 .- batch.targets.margin))
    death_logits = reshape(output.death_logit, width, state_batch)
    death = _masked_mean(
        _binary_cross_entropy_with_logits(death_logits, batch.targets.death),
        batch.targets.death_mask,
    )
    quantile = _quantile_teacher_loss(output.quantiles, batch.targets.teacher_q, batch.mask)
    loss = LISTNET_WEIGHT * listnet + OLD_Q_WEIGHT * old_q +
           MARGIN_WEIGHT * margin + DEATH_WEIGHT * death +
           QUANTILE_TEACHER_WEIGHT * quantile
    statistics = (;
        listnet_loss=listnet,
        old_q_loss=old_q,
        margin_loss=margin,
        death_loss=death,
        quantile_teacher_loss=quantile,
        valid_candidates=sum(batch.mask),
    )
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

promotion_key(metrics) = (
    metrics.top1_agreement,
    metrics.ndcg,
    metrics.pairwise_accuracy,
    -metrics.old_q_huber,
)

optional_target_coverage(dataset) = (;
    line_clear="derived for every candidate; canonical model has no separate output head, so used as geometry only",
    max_height="derived for every candidate; canonical model has no separate output head, so used in aux input",
    holes="derived for every candidate; canonical model has no separate output head, so used in aux input",
    cavities="derived for every candidate; canonical model has no separate output head, so used in aux input",
    death="observed only for selected teacher actions and trained through death_logit",
)

end # module
