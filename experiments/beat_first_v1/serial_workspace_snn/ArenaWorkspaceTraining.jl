module ArenaWorkspaceTraining

using LinearAlgebra
using Lux
using Random
using Statistics

if !isdefined(Main, :BoundedMPMCRing)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "..",
            "episodic_vit_recurrent_lookup",
            "bounded_mpmc_queue.jl",
        ),
    )
end
if !isdefined(Main, :WinCpuSets)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "..",
            "episodic_vit_recurrent_lookup",
            "windows_cpu_sets.jl",
        ),
    )
end
if !isdefined(Main, :SerialWorkspaceSNN)
    Base.include(Main, joinpath(@__DIR__, "SerialWorkspaceSNN.jl"))
end
if !isdefined(Main, :BeatFirstTrainingCore)
    Base.include(Main, joinpath(@__DIR__, "..", "training", "core.jl"))
end

const Queue = Main.BoundedMPMCRing
const CpuSets = Main.WinCpuSets
const Model = Main.SerialWorkspaceSNN
const TrainingCore = Main.BeatFirstTrainingCore

const OUTPUT_DIM = 22
const QUANTILES = 16
const BOARD_ROWS = 24
const BOARD_COLUMNS = 10
const BOARD_CELLS = BOARD_ROWS * BOARD_COLUMNS
const QUEUE_PIECES = 7
const QUEUE_TOKENS = 6
const AUX_FEATURES = 37

export ArenaTrainer,
    ArenaExecutor,
    arena_gradient!,
    arena_update!,
    arena_output,
    arena_parameter_max_abs_difference,
    arena_tree_norm,
    copy_parameters,
    fill_next_rows!,
    loss_and_raw_gradient!,
    pack_arena_batch!,
    run_with_arena_team!,
    training_arena

mutable struct ArenaTargets
    teacher_q::Matrix{Float32}
    teacher_z::Matrix{Float32}
    top1::Vector{Int16}
    top2::Vector{Int16}
    margin::Vector{Float32}
    death::Matrix{Float32}
    death_mask::Matrix{Float32}
    line_clear::Matrix{Float32}
    max_height::Matrix{Float32}
    holes::Matrix{Float32}
    cavities::Matrix{Float32}
end

function ArenaTargets(width::Int, state_batch::Int)
    return ArenaTargets(
        zeros(Float32, width, state_batch),
        zeros(Float32, width, state_batch),
        zeros(Int16, state_batch),
        zeros(Int16, state_batch),
        zeros(Float32, state_batch),
        zeros(Float32, width, state_batch),
        zeros(Float32, width, state_batch),
        zeros(Float32, width, state_batch),
        zeros(Float32, width, state_batch),
        zeros(Float32, width, state_batch),
        zeros(Float32, width, state_batch),
    )
end

"""
All candidate-dependent storage needed by the manual forward/VJP.

The first dimension is contiguous for every hot vector.  `flat` retains the
same `candidate + (state-1)*width` convention as the shared teacher contract,
so loss and correctness comparisons never reorder candidates.
"""
mutable struct TrainingArena
    state_batch::Int
    width::Int
    capacity::Int
    rows::Vector{Int}
    counts::Vector{Int16}
    valid_flats::Vector{Int32}
    valid_count::Int
    targets::ArenaTargets
    rails::Matrix{Float32}
    raw::Matrix{Float32}
    raw_gradient::Matrix{Float32}
    membrane::Array{Float32,3}
    active_spikes::Array{Float32,3}
    workspace::Array{Float32,3}
    query::Matrix{Float32}
    hidden::Matrix{Float32}
    block_mask::Array{Float32,3}
end

function TrainingArena(model, state_batch::Int, width::Int)
    state_batch >= 1 || throw(ArgumentError("state_batch must be positive"))
    width >= 1 || throw(ArgumentError("candidate width must be positive"))
    nodes = model.blocks * model.node_dim
    capacity = state_batch * width
    return TrainingArena(
        state_batch,
        width,
        capacity,
        zeros(Int, state_batch),
        zeros(Int16, state_batch),
        zeros(Int32, capacity),
        0,
        ArenaTargets(width, state_batch),
        zeros(Float32, Model.INPUT_RAILS, capacity),
        zeros(Float32, OUTPUT_DIM, capacity),
        zeros(Float32, OUTPUT_DIM, capacity),
        zeros(Float32, nodes, model.cycles + 1, capacity),
        zeros(Float32, nodes, model.cycles, capacity),
        zeros(Float32, model.node_dim, model.cycles + 1, capacity),
        zeros(Float32, model.node_dim, capacity),
        zeros(Float32, model.hidden, capacity),
        zeros(Float32, model.blocks, model.cycles, capacity),
    )
end

mutable struct PackScratch
    board::Matrix{Float32}
    combined::Matrix{Float32}
    after::Matrix{Float32}
    full_rows::Vector{Bool}
    reachable::Matrix{Bool}
    flood_queue::Vector{Int16}
    heights::Vector{Int16}
    holes::Vector{Int16}
    wells::Vector{Int16}
end

PackScratch() = PackScratch(
    zeros(Float32, BOARD_ROWS, BOARD_COLUMNS),
    zeros(Float32, BOARD_ROWS, BOARD_COLUMNS),
    zeros(Float32, BOARD_ROWS, BOARD_COLUMNS),
    falses(BOARD_ROWS),
    falses(BOARD_ROWS, BOARD_COLUMNS),
    zeros(Int16, BOARD_CELLS),
    zeros(Int16, BOARD_COLUMNS),
    zeros(Int16, BOARD_COLUMNS),
    zeros(Int16, BOARD_COLUMNS),
)

@inline _flat_index(candidate::Int, state_slot::Int, width::Int) =
    candidate + (state_slot - 1) * width

@inline function _state_candidate(flat::Int, width::Int)
    state_slot = div(flat - 1, width) + 1
    candidate = flat - (state_slot - 1) * width
    return state_slot, candidate
end

@inline function _stable_top_two(dataset, row::Int, count::Int)
    top1 = 1
    @inbounds for candidate in 2:count
        dataset.teacher_q[candidate, row] > dataset.teacher_q[top1, row] &&
            (top1 = candidate)
    end
    top2 = count == 1 ? top1 : (top1 == 1 ? 2 : 1)
    @inbounds for candidate in 1:count
        candidate == top1 && continue
        dataset.teacher_q[candidate, row] > dataset.teacher_q[top2, row] &&
            (top2 = candidate)
    end
    return top1, top2
end

"""
Prepare targets and the canonical list of valid flat candidate IDs.

This is deliberately loop-only: no `sortperm`, views, temporary masks, or
standard-deviation arrays are created at an update boundary.
"""
function prepare_batch_metadata!(
    arena::TrainingArena,
    dataset,
)
    arena.valid_count = 0
    fill!(arena.raw, 0.0f0)
    fill!(arena.raw_gradient, 0.0f0)
    targets = arena.targets
    width = arena.width
    @inbounds for state_slot in 1:arena.state_batch
        row = arena.rows[state_slot]
        1 <= row <= length(dataset.action_counts) ||
            throw(BoundsError(dataset.action_counts, row))
        count = Int(dataset.action_counts[row])
        1 <= count <= width || error(
            "state $row has $count candidates outside width $width",
        )
        arena.counts[state_slot] = Int16(count)

        teacher_sum = 0.0f0
        for candidate in 1:count
            teacher_sum += Float32(dataset.teacher_q[candidate, row])
        end
        teacher_mean = teacher_sum / Float32(count)
        teacher_variance = 0.0f0
        for candidate in 1:count
            centered = Float32(dataset.teacher_q[candidate, row]) - teacher_mean
            teacher_variance = muladd(centered, centered, teacher_variance)
        end
        teacher_scale = max(
            sqrt(teacher_variance / Float32(count)),
            1.0f-4,
        )
        top1, top2 = _stable_top_two(dataset, row, count)
        targets.top1[state_slot] = Int16(top1)
        targets.top2[state_slot] = Int16(top2)
        targets.margin[state_slot] =
            Float32(dataset.teacher_q[top1, row]) -
            Float32(dataset.teacher_q[top2, row])
        candidate_death_available =
            hasproperty(dataset, :candidate_death_available) &&
            dataset.candidate_death_available[row]
        selected = Int(dataset.selected_actions[row])
        for candidate in 1:count
            teacher = Float32(dataset.teacher_q[candidate, row])
            targets.teacher_q[candidate, state_slot] = teacher
            targets.teacher_z[candidate, state_slot] =
                (teacher - teacher_mean) / teacher_scale
            if candidate_death_available
                targets.death[candidate, state_slot] =
                    Float32(dataset.candidate_death[candidate, row])
                targets.death_mask[candidate, state_slot] = 1.0f0
            else
                targets.death[candidate, state_slot] =
                    candidate == selected ? Float32(dataset.terminal[row]) : 0.0f0
                targets.death_mask[candidate, state_slot] =
                    candidate == selected ? 1.0f0 : 0.0f0
            end
            targets.line_clear[candidate, state_slot] =
                Float32(dataset.line_clear[candidate, row])
            targets.max_height[candidate, state_slot] =
                Float32(dataset.max_height[candidate, row])
            targets.holes[candidate, state_slot] =
                Float32(dataset.holes[candidate, row])
            targets.cavities[candidate, state_slot] =
                Float32(dataset.cavities[candidate, row])
            arena.valid_count += 1
            arena.valid_flats[arena.valid_count] =
                Int32(_flat_index(candidate, state_slot, width))
        end
    end
    return arena.valid_count
end

@inline function _fill_after_board!(
    scratch::PackScratch,
    dataset,
    row::Int,
    candidate::Int,
)
    @inbounds for column in 1:BOARD_COLUMNS, board_row in 1:BOARD_ROWS
        board = Float32(dataset.boards[board_row, column, 1, row])
        placement =
            Float32(dataset.placements[board_row, column, 1, candidate, row])
        scratch.board[board_row, column] = board
        scratch.combined[board_row, column] = min(1.0f0, board + placement)
    end
    full_count = 0
    @inbounds for board_row in 1:BOARD_ROWS
        occupied = 0
        for column in 1:BOARD_COLUMNS
            occupied += scratch.combined[board_row, column] >= 0.5f0
        end
        full = occupied == BOARD_COLUMNS
        scratch.full_rows[board_row] = full
        full_count += full
    end
    fill!(scratch.after, 0.0f0)
    output_row = full_count + 1
    @inbounds for board_row in 1:BOARD_ROWS
        scratch.full_rows[board_row] && continue
        for column in 1:BOARD_COLUMNS
            scratch.after[output_row, column] =
                scratch.combined[board_row, column]
        end
        output_row += 1
    end
    output_row == BOARD_ROWS + 1 || error("line-clear packing drift")
    return full_count
end

@inline function _fill_geometry!(scratch::PackScratch)
    fill!(scratch.heights, 0)
    fill!(scratch.holes, 0)
    @inbounds for column in 1:BOARD_COLUMNS
        first_filled = 0
        for board_row in 1:BOARD_ROWS
            if scratch.after[board_row, column] > 0.0f0
                first_filled = board_row
                break
            end
        end
        first_filled == 0 && continue
        scratch.heights[column] = Int16(BOARD_ROWS - first_filled + 1)
        holes = 0
        for board_row in first_filled:BOARD_ROWS
            holes += iszero(scratch.after[board_row, column])
        end
        scratch.holes[column] = Int16(holes)
    end

    fill!(scratch.reachable, false)
    head = 1
    tail = 0
    @inbounds for column in 1:BOARD_COLUMNS
        if iszero(scratch.after[1, column])
            tail += 1
            linear = 1 + (column - 1) * BOARD_ROWS
            scratch.flood_queue[tail] = Int16(linear)
            scratch.reachable[1, column] = true
        end
    end
    @inbounds while head <= tail
        linear = Int(scratch.flood_queue[head])
        head += 1
        column = div(linear - 1, BOARD_ROWS) + 1
        board_row = linear - (column - 1) * BOARD_ROWS
        if board_row > 1 &&
           iszero(scratch.after[board_row - 1, column]) &&
           !scratch.reachable[board_row - 1, column]
            tail += 1
            scratch.flood_queue[tail] = Int16(linear - 1)
            scratch.reachable[board_row - 1, column] = true
        end
        if board_row < BOARD_ROWS &&
           iszero(scratch.after[board_row + 1, column]) &&
           !scratch.reachable[board_row + 1, column]
            tail += 1
            scratch.flood_queue[tail] = Int16(linear + 1)
            scratch.reachable[board_row + 1, column] = true
        end
        if column > 1 &&
           iszero(scratch.after[board_row, column - 1]) &&
           !scratch.reachable[board_row, column - 1]
            tail += 1
            scratch.flood_queue[tail] = Int16(linear - BOARD_ROWS)
            scratch.reachable[board_row, column - 1] = true
        end
        if column < BOARD_COLUMNS &&
           iszero(scratch.after[board_row, column + 1]) &&
           !scratch.reachable[board_row, column + 1]
            tail += 1
            scratch.flood_queue[tail] = Int16(linear + BOARD_ROWS)
            scratch.reachable[board_row, column + 1] = true
        end
    end

    cavities = 0
    aggregate_height = 0
    bumpiness = 0
    max_height = 0
    @inbounds for column in 1:BOARD_COLUMNS
        height = Int(scratch.heights[column])
        aggregate_height += height
        max_height = max(max_height, height)
        column > 1 &&
            (bumpiness += abs(height - Int(scratch.heights[column - 1])))
        left = column == 1 ? BOARD_ROWS : Int(scratch.heights[column - 1])
        right = column == BOARD_COLUMNS ?
            BOARD_ROWS : Int(scratch.heights[column + 1])
        scratch.wells[column] = Int16(max(min(left, right) - height, 0))
        for board_row in 1:BOARD_ROWS
            cavities += iszero(scratch.after[board_row, column]) &&
                !scratch.reachable[board_row, column]
        end
    end
    return cavities, aggregate_height, bumpiness, max_height
end

@inline function _aux_value(
    scratch::PackScratch,
    dataset,
    row::Int,
    candidate::Int,
    index::Int,
    cavities::Int,
    aggregate_height::Int,
    bumpiness::Int,
    max_height::Int,
)
    index <= 10 && return Float32(scratch.heights[index]) / 24.0f0
    index <= 20 && return Float32(scratch.holes[index - 10]) / 24.0f0
    index <= 30 && return Float32(scratch.wells[index - 20]) / 24.0f0
    index == 31 && return Float32(cavities) / 240.0f0
    index == 32 && return Float32(aggregate_height) / 240.0f0
    index == 33 && return Float32(bumpiness) / 216.0f0
    index == 34 && return Float32(max_height) / 24.0f0
    index == 35 && return Float32(dataset.ren[1, row]) / 30.0f0
    index == 36 && return Float32(dataset.back_to_back[1, row])
    index == 37 && return Float32(dataset.tspin[candidate, row])
    error("invalid auxiliary feature $index")
end

"""Pack exactly one valid candidate directly into its 1,298 binary rails."""
function pack_candidate_rails!(
    arena::TrainingArena,
    dataset,
    scratch::PackScratch,
    flat::Int,
)
    state_slot, candidate = _state_candidate(flat, arena.width)
    row = arena.rows[state_slot]
    expected_line_clear =
        Int(arena.targets.line_clear[candidate, state_slot])
    actual_line_clear =
        _fill_after_board!(scratch, dataset, row, candidate)
    actual_line_clear == expected_line_clear || error(
        "stored line-clear target differs at row $row candidate $candidate",
    )
    cavities, aggregate_height, bumpiness, max_height =
        _fill_geometry!(scratch)
    max_height == Int(arena.targets.max_height[candidate, state_slot]) ||
        error("stored max-height target differs")
    holes_total = 0
    @inbounds for value in scratch.holes
        holes_total += Int(value)
    end
    holes_total == Int(arena.targets.holes[candidate, state_slot]) ||
        error("stored hole target differs")
    cavities == Int(arena.targets.cavities[candidate, state_slot]) ||
        error("stored cavity target differs")

    rail = 0
    @inbounds for column in 1:BOARD_COLUMNS, board_row in 1:BOARD_ROWS
        rail += 1
        arena.rails[rail, flat] =
            scratch.board[board_row, column] > 0.5f0
    end
    @inbounds for column in 1:BOARD_COLUMNS, board_row in 1:BOARD_ROWS
        rail += 1
        arena.rails[rail, flat] =
            scratch.after[board_row, column] > 0.5f0
    end
    @inbounds for column in 1:BOARD_COLUMNS, board_row in 1:BOARD_ROWS
        rail += 1
        arena.rails[rail, flat] =
            scratch.after[board_row, column] >
            scratch.board[board_row, column]
    end
    @inbounds for column in 1:BOARD_COLUMNS, board_row in 1:BOARD_ROWS
        rail += 1
        arena.rails[rail, flat] =
            scratch.after[board_row, column] <
            scratch.board[board_row, column]
    end
    @inbounds for token in 1:QUEUE_TOKENS, piece in 1:QUEUE_PIECES
        rail += 1
        arena.rails[rail, flat] =
            Float32(dataset.queues[piece, token, row]) > 0.5f0
    end
    @inbounds for level in 1:Model.AUX_LEVELS
        threshold = Float32(level) / Float32(Model.AUX_LEVELS)
        for index in 1:AUX_FEATURES
            rail += 1
            arena.rails[rail, flat] = _aux_value(
                scratch,
                dataset,
                row,
                candidate,
                index,
                cavities,
                aggregate_height,
                bumpiness,
                max_height,
            ) >= threshold
        end
    end
    rail == Model.INPUT_RAILS || error("binary rail packing drift")
    return nothing
end

function _zero_parameter_tree(parameters)
    return NamedTuple{keys(parameters)}(
        map(array -> zeros(Float32, size(array)), values(parameters)),
    )
end

function copy_parameters(parameters)
    return NamedTuple{keys(parameters)}(
        map(copy, values(parameters)),
    )
end

function _fill_parameter_tree!(tree, value::Float32=0.0f0)
    @inbounds for array in values(tree)
        fill!(array, value)
    end
    return tree
end

function arena_tree_norm(tree)
    total = 0.0
    @inbounds for array in values(tree)
        for value in array
            total = muladd(Float64(value), Float64(value), total)
        end
    end
    return sqrt(total)
end

function arena_parameter_max_abs_difference(left, right)
    maximum_difference = 0.0
    keys(left) == keys(right) || return Inf
    @inbounds for key in keys(left)
        left_array = getproperty(left, key)
        right_array = getproperty(right, key)
        size(left_array) == size(right_array) || return Inf
        for index in eachindex(left_array, right_array)
            maximum_difference = max(
                maximum_difference,
                abs(Float64(left_array[index]) - Float64(right_array[index])),
            )
        end
    end
    return maximum_difference
end

mutable struct ParameterCache
    gate_probability::Matrix{Float32}
    gate_hard::Matrix{Float32}
    gate_derivative::Matrix{Float32}
    delay::Matrix{Float32}
    delay_derivative::Matrix{Float32}
    leak::Vector{Float32}
    leak_derivative::Vector{Float32}
    threshold::Vector{Float32}
    threshold_derivative::Vector{Float32}
    workspace_decay::Float32
    workspace_decay_derivative::Float32
end

function ParameterCache(parameters)
    cache = ParameterCache(
        similar(parameters.gate_logits),
        similar(parameters.gate_logits),
        similar(parameters.gate_logits),
        similar(parameters.delay_logits),
        similar(parameters.delay_logits),
        similar(parameters.leak_logits),
        similar(parameters.leak_logits),
        similar(parameters.threshold_logits),
        similar(parameters.threshold_logits),
        0.0f0,
        0.0f0,
    )
    refresh_parameter_cache!(cache, parameters)
    return cache
end

function refresh_parameter_cache!(cache::ParameterCache, parameters)
    @inbounds for index in eachindex(parameters.gate_logits)
        probability = sigmoid(parameters.gate_logits[index])
        cache.gate_probability[index] = probability
        cache.gate_hard[index] =
            parameters.gate_logits[index] >= 0.0f0 ? 1.0f0 : 0.0f0
        cache.gate_derivative[index] = probability * (1.0f0 - probability)
    end
    @inbounds for index in eachindex(parameters.delay_logits)
        probability = sigmoid(parameters.delay_logits[index])
        cache.delay[index] = probability
        cache.delay_derivative[index] = probability * (1.0f0 - probability)
    end
    @inbounds for index in eachindex(parameters.leak_logits)
        probability = sigmoid(parameters.leak_logits[index])
        cache.leak[index] = 0.45f0 + 0.50f0 * probability
        cache.leak_derivative[index] =
            0.50f0 * probability * (1.0f0 - probability)
    end
    @inbounds for index in eachindex(parameters.threshold_logits)
        probability = sigmoid(parameters.threshold_logits[index])
        cache.threshold[index] = 0.25f0 + 0.75f0 * probability
        cache.threshold_derivative[index] =
            0.75f0 * probability * (1.0f0 - probability)
    end
    probability = sigmoid(parameters.workspace_decay_logit[1])
    cache.workspace_decay = probability
    cache.workspace_decay_derivative =
        probability * (1.0f0 - probability)
    return cache
end

mutable struct LossScratch
    student_z::Vector{Float32}
    teacher_probability::Vector{Float32}
    student_probability::Vector{Float32}
    z_cotangent::Vector{Float32}
end

LossScratch(width::Int) = LossScratch(
    zeros(Float32, width),
    zeros(Float32, width),
    zeros(Float32, width),
    zeros(Float32, width),
)

struct LossRecord
    composite_loss::Float32
    listnet_loss::Float32
    old_q_loss::Float32
    margin_loss::Float32
    death_loss::Float32
    quantile_teacher_loss::Float32
    geometry_loss::Float32
    structure_loss::Float32
    gate_density::Float32
    valid_candidates::Int
end

@inline _huber(value::Float32) =
    abs(value) <= 1.0f0 ?
        0.5f0 * value * value : abs(value) - 0.5f0

@inline _huber_derivative(value::Float32) =
    clamp(value, -1.0f0, 1.0f0)

"""
Exact allocation-free VJP of the shared default supervised objective.

The formula follows `BeatFirstTrainingCore.supervised_components`: per-state
standardized ListNet, fixed teacher top-2 margin, old-Q Huber, death BCE,
16 quantile Huber heads, and four normalized geometry heads.
"""
function loss_and_raw_gradient!(
    arena::TrainingArena,
    scratch::LossScratch,
    gate_density::Float32,
    structure_weight::Float32,
)
    fill!(arena.raw_gradient, 0.0f0)
    raw = arena.raw
    draw = arena.raw_gradient
    targets = arena.targets
    width = arena.width
    state_batch = arena.state_batch
    valid_total = arena.valid_count
    listnet_loss = 0.0f0
    old_q_loss = 0.0f0
    margin_loss = 0.0f0
    death_loss = 0.0f0
    quantile_loss = 0.0f0
    line_clear_loss = 0.0f0
    max_height_loss = 0.0f0
    holes_loss = 0.0f0
    cavities_loss = 0.0f0
    death_count = 0.0f0
    @inbounds for state_slot in 1:state_batch
        count = Int(arena.counts[state_slot])
        offset = (state_slot - 1) * width
        q_mean = 0.0f0
        for candidate in 1:count
            q_mean += raw[1, offset + candidate]
        end
        q_mean /= Float32(count)
        variance = 0.0f0
        for candidate in 1:count
            centered = raw[1, offset + candidate] - q_mean
            scratch.student_z[candidate] = centered
            variance = muladd(centered, centered, variance)
        end
        scale = sqrt(variance / Float32(count) + 1.0f-4)
        inverse_scale = inv(scale)
        teacher_max = -Inf32
        student_max = -Inf32
        for candidate in 1:count
            z = scratch.student_z[candidate] * inverse_scale
            scratch.student_z[candidate] = z
            teacher_logit =
                targets.teacher_z[candidate, state_slot] / 0.50f0
            student_logit = z / 0.50f0
            teacher_max = max(teacher_max, teacher_logit)
            student_max = max(student_max, student_logit)
        end
        teacher_sum = 0.0f0
        student_sum = 0.0f0
        for candidate in 1:count
            teacher_probability = exp(
                targets.teacher_z[candidate, state_slot] / 0.50f0 -
                teacher_max,
            )
            student_probability = exp(
                scratch.student_z[candidate] / 0.50f0 - student_max,
            )
            scratch.teacher_probability[candidate] = teacher_probability
            scratch.student_probability[candidate] = student_probability
            teacher_sum += teacher_probability
            student_sum += student_probability
        end
        inverse_teacher_sum = inv(max(teacher_sum, 1.0f-12))
        inverse_student_sum = inv(max(student_sum, 1.0f-12))
        mean_g = 0.0f0
        mean_gz = 0.0f0
        for candidate in 1:count
            teacher_probability =
                scratch.teacher_probability[candidate] * inverse_teacher_sum
            student_probability =
                scratch.student_probability[candidate] * inverse_student_sum
            scratch.teacher_probability[candidate] = teacher_probability
            scratch.student_probability[candidate] = student_probability
            listnet_loss -= teacher_probability *
                log(max(student_probability, 1.0f-12)) /
                Float32(state_batch)
            cotangent = (
                student_probability - teacher_probability
            ) / (0.50f0 * Float32(state_batch))
            scratch.z_cotangent[candidate] = cotangent
            mean_g += cotangent
            mean_gz = muladd(
                cotangent,
                scratch.student_z[candidate],
                mean_gz,
            )
        end
        mean_g /= Float32(count)
        mean_gz /= Float32(count)
        for candidate in 1:count
            flat = offset + candidate
            draw[1, flat] += (
                scratch.z_cotangent[candidate] -
                mean_g -
                scratch.student_z[candidate] * mean_gz
            ) * inverse_scale
        end

        top1 = Int(targets.top1[state_slot])
        top2 = Int(targets.top2[state_slot])
        margin_error =
            raw[1, offset + top1] -
            raw[1, offset + top2] -
            targets.margin[state_slot]
        margin_loss += _huber(margin_error) / Float32(state_batch)
        margin_gradient =
            0.15f0 * _huber_derivative(margin_error) /
            Float32(state_batch)
        draw[1, offset + top1] += margin_gradient
        draw[1, offset + top2] -= margin_gradient

        for candidate in 1:count
            flat = offset + candidate
            q_error =
                raw[1, flat] - targets.teacher_q[candidate, state_slot]
            old_q_loss += _huber(q_error) / Float32(valid_total)
            draw[1, flat] +=
                0.25f0 * _huber_derivative(q_error) /
                Float32(valid_total)

            if targets.death_mask[candidate, state_slot] != 0.0f0
                logit = raw[2, flat]
                label = targets.death[candidate, state_slot]
                death_loss +=
                    max(logit, 0.0f0) -
                    logit * label +
                    log1p(exp(-abs(logit)))
                death_count += 1.0f0
            end

            teacher_q = targets.teacher_q[candidate, state_slot]
            for quantile in 1:QUANTILES
                prediction = raw[2 + quantile, flat]
                error = teacher_q - prediction
                tau = (Float32(quantile) - 0.5f0) / Float32(QUANTILES)
                negative = error < 0.0f0 ? 1.0f0 : 0.0f0
                weight = abs(tau - negative)
                quantile_loss +=
                    weight * _huber(error) /
                    Float32(valid_total * QUANTILES)
                draw[2 + quantile, flat] -=
                    0.05f0 * weight * _huber_derivative(error) /
                    Float32(valid_total * QUANTILES)
            end

            line_error =
                raw[19, flat] -
                targets.line_clear[candidate, state_slot] / 4.0f0
            height_error =
                raw[20, flat] -
                targets.max_height[candidate, state_slot] / 24.0f0
            holes_error =
                raw[21, flat] -
                targets.holes[candidate, state_slot] / 240.0f0
            cavities_error =
                raw[22, flat] -
                targets.cavities[candidate, state_slot] / 240.0f0
            line_clear_loss += _huber(line_error) / Float32(valid_total)
            max_height_loss += _huber(height_error) / Float32(valid_total)
            holes_loss += _huber(holes_error) / Float32(valid_total)
            cavities_loss += _huber(cavities_error) / Float32(valid_total)
            geometry_gradient_scale = 0.025f0 / Float32(valid_total)
            draw[19, flat] +=
                geometry_gradient_scale * _huber_derivative(line_error)
            draw[20, flat] +=
                geometry_gradient_scale * _huber_derivative(height_error)
            draw[21, flat] +=
                geometry_gradient_scale * _huber_derivative(holes_error)
            draw[22, flat] +=
                geometry_gradient_scale * _huber_derivative(cavities_error)
        end
    end
    death_denominator = max(death_count, 1.0f0)
    @inbounds for state_slot in 1:state_batch
        count = Int(arena.counts[state_slot])
        offset = (state_slot - 1) * width
        for candidate in 1:count
            targets.death_mask[candidate, state_slot] == 0.0f0 && continue
            flat = offset + candidate
            draw[2, flat] +=
                0.10f0 * (
                    sigmoid(raw[2, flat]) -
                    targets.death[candidate, state_slot]
                ) / death_denominator
        end
    end
    death_loss /= death_denominator
    geometry_loss = (
        line_clear_loss +
        max_height_loss +
        holes_loss +
        cavities_loss
    ) / 4.0f0
    structure_loss =
        structure_weight * (gate_density - 0.50f0)^2
    composite_loss =
        listnet_loss +
        0.25f0 * old_q_loss +
        0.15f0 * margin_loss +
        0.10f0 * death_loss +
        0.05f0 * quantile_loss +
        0.10f0 * geometry_loss +
        structure_loss
    return LossRecord(
        composite_loss,
        listnet_loss,
        old_q_loss,
        margin_loss,
        death_loss,
        quantile_loss,
        geometry_loss,
        structure_loss,
        gate_density,
        valid_total,
    )
end

mutable struct CandidateScratch
    pack::PackScratch
    scores::Vector{Float32}
    selected::Vector{Bool}
    soft_route::Vector{Float32}
    dmembrane_a::Vector{Float32}
    dmembrane_b::Vector{Float32}
    dactive::Vector{Float32}
    dprevious_active::Vector{Float32}
    dworkspace_a::Vector{Float32}
    dworkspace_b::Vector{Float32}
    dquery::Vector{Float32}
    dseed::Vector{Float32}
    dblock_mask::Vector{Float32}
    features::Vector{Float32}
    dfeatures::Vector{Float32}
    dhidden::Vector{Float32}
    consolidation_evidence::Vector{Float32}
    consolidation_selected::Vector{Bool}
end

function CandidateScratch(model)
    nodes = model.blocks * model.node_dim
    return CandidateScratch(
        PackScratch(),
        zeros(Float32, model.blocks),
        falses(model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, nodes),
        zeros(Float32, nodes),
        zeros(Float32, nodes),
        zeros(Float32, nodes),
        zeros(Float32, model.node_dim),
        zeros(Float32, model.node_dim),
        zeros(Float32, model.node_dim),
        zeros(Float32, nodes),
        zeros(Float32, model.blocks),
        zeros(Float32, 3 * model.node_dim),
        zeros(Float32, 3 * model.node_dim),
        zeros(Float32, model.hidden),
        zeros(Float32, model.fanout),
        falses(model.fanout),
    )
end

@inline function _compute_scores!(
    scratch::CandidateScratch,
    arena::TrainingArena,
    model,
    parameters,
    cycle::Int,
    flat::Int,
)
    node_dim = model.node_dim
    @inbounds for block in 1:model.blocks
        score = 0.0f0
        magnitude = 0.0f0
        offset = (block - 1) * node_dim
        for coordinate in 1:node_dim
            node = offset + coordinate
            membrane = arena.membrane[node, cycle, flat]
            score = muladd(
                membrane * parameters.workspace_key[coordinate, block],
                arena.query[coordinate, flat],
                score,
            )
            magnitude += abs(membrane)
        end
        scratch.scores[block] = score + 0.05f0 * magnitude
    end
    return scratch.scores
end

@inline function _select_workspace!(
    scratch::CandidateScratch,
    arena::TrainingArena,
    model,
    cycle::Int,
    flat::Int,
)
    fill!(scratch.selected, false)
    @inbounds for _ in 1:model.workspace_k
        best = 0
        best_score = -Inf32
        for block in 1:model.blocks
            scratch.selected[block] && continue
            score = scratch.scores[block]
            if best == 0 || score > best_score
                best = block
                best_score = score
            end
        end
        best != 0 || error("workspace top-k selection failed")
        scratch.selected[best] = true
    end
    @inbounds for block in 1:model.blocks
        arena.block_mask[block, cycle, flat] =
            scratch.selected[block] ? 1.0f0 : 0.0f0
    end
    return nothing
end

"""
Manual one-candidate forward with all backward-live tensors stored in `arena`.

No array expression is used in this path.  Every destination is written by
one native worker and the synapse accumulation order is relation-major, the
same order as the auditable printer scan.
"""
function forward_candidate!(
    arena::TrainingArena,
    model,
    parameters,
    cache::ParameterCache,
    scratch::CandidateScratch,
    flat::Int,
)
    nodes = model.blocks * model.node_dim
    node_dim = model.node_dim
    @inbounds for node in 1:nodes
        rail = arena.rails[model.feature_for_node[node], flat]
        seed = muladd(parameters.input_gain[node], rail, parameters.input_bias[node])
        arena.membrane[node, 1, flat] = seed
    end
    @inbounds for coordinate in 1:node_dim
        value = 0.0f0
        for rail in 1:Model.INPUT_RAILS
            value = muladd(
                parameters.query_weight[coordinate, rail],
                arena.rails[rail, flat],
                value,
            )
        end
        arena.query[coordinate, flat] = tanh(value)
        arena.workspace[coordinate, 1, flat] = 0.0f0
    end

    @inbounds for cycle in 1:model.cycles
        _compute_scores!(
            scratch, arena, model, parameters, cycle, flat,
        )
        _select_workspace!(scratch, arena, model, cycle, flat)

        for node in 1:nodes
            block = div(node - 1, node_dim) + 1
            membrane = arena.membrane[node, cycle, flat]
            spike = membrane >= cache.threshold[node] ? 1.0f0 : 0.0f0
            arena.active_spikes[node, cycle, flat] =
                spike * arena.block_mask[block, cycle, flat]
            arena.membrane[node, cycle + 1, flat] =
                cache.leak[node] * membrane +
                0.18f0 * arena.membrane[node, 1, flat] -
                spike * cache.threshold[node]
        end

        for coordinate in 1:node_dim
            write = 0.0f0
            for block in 1:model.blocks
                node = coordinate + (block - 1) * node_dim
                write = muladd(
                    arena.membrane[node, cycle, flat],
                    arena.block_mask[block, cycle, flat],
                    write,
                )
            end
            write /= Float32(model.workspace_k)
            previous_workspace =
                arena.workspace[coordinate, cycle, flat]
            next_workspace = tanh(
                cache.workspace_decay * previous_workspace + write,
            )
            arena.workspace[coordinate, cycle + 1, flat] = next_workspace
            for block in 1:model.blocks
                node = coordinate + (block - 1) * node_dim
                arena.membrane[node, cycle + 1, flat] +=
                    parameters.feedback_gain[coordinate, block] *
                    next_workspace
            end
        end

        for relation in 1:model.fanout
            for source in 1:nodes
                cache.gate_hard[source, relation] == 0.0f0 && continue
                destination =
                    model.destination_for_source[source, relation]
                delay = cache.delay[source, relation]
                current = arena.active_spikes[source, cycle, flat]
                previous = cycle == 1 ? 0.0f0 :
                    arena.active_spikes[source, cycle - 1, flat]
                signal = muladd(1.0f0 - delay, current, delay * previous)
                arena.membrane[destination, cycle + 1, flat] +=
                    parameters.synapse_weight[source, relation] * signal
            end
        end
    end

    @inbounds for coordinate in 1:node_dim
        pooled = 0.0f0
        for block in 1:model.blocks
            node = coordinate + (block - 1) * node_dim
            pooled += arena.membrane[node, model.cycles + 1, flat]
        end
        scratch.features[coordinate] =
            arena.workspace[coordinate, model.cycles + 1, flat]
        scratch.features[node_dim + coordinate] =
            arena.query[coordinate, flat]
        scratch.features[2 * node_dim + coordinate] =
            pooled / Float32(model.blocks)
    end
    @inbounds for hidden in 1:model.hidden
        activation = parameters.head_bias[hidden]
        for coordinate in 1:node_dim
            activation = muladd(
                parameters.head_weight[hidden, coordinate],
                scratch.features[coordinate],
                activation,
            )
            activation = muladd(
                parameters.head_weight[hidden, node_dim + coordinate],
                scratch.features[node_dim + coordinate],
                activation,
            )
            activation = muladd(
                parameters.head_weight[hidden, 2 * node_dim + coordinate],
                scratch.features[2 * node_dim + coordinate],
                activation,
            )
        end
        arena.hidden[hidden, flat] = tanh(activation)
    end
    @inbounds for output in 1:OUTPUT_DIM
        value = parameters.output_bias[output]
        for hidden in 1:model.hidden
            value = muladd(
                parameters.output_weight[output, hidden],
                arena.hidden[hidden, flat],
                value,
            )
        end
        arena.raw[output, flat] = value
    end
    return nothing
end

@inline function _feature_value(
    arena::TrainingArena,
    model,
    flat::Int,
    feature::Int,
)
    node_dim = model.node_dim
    feature <= node_dim && return arena.workspace[
        feature, model.cycles + 1, flat,
    ]
    feature <= 2 * node_dim &&
        return arena.query[feature - node_dim, flat]
    coordinate = feature - 2 * node_dim
    pooled = 0.0f0
    @inbounds for block in 1:model.blocks
        node = coordinate + (block - 1) * node_dim
        pooled += arena.membrane[node, model.cycles + 1, flat]
    end
    return pooled / Float32(model.blocks)
end

"""
Analytic reverse pass for one candidate.

`gradient` is worker-owned for the entire update.  Consequently every `+=`
below is lock-free, while the parameters and forward arena are read-only.
"""
function backward_candidate!(
    gradient,
    arena::TrainingArena,
    model,
    parameters,
    cache::ParameterCache,
    scratch::CandidateScratch,
    flat::Int,
)
    nodes = model.blocks * model.node_dim
    node_dim = model.node_dim
    fill!(scratch.dfeatures, 0.0f0)
    fill!(scratch.dhidden, 0.0f0)
    fill!(scratch.dquery, 0.0f0)
    fill!(scratch.dseed, 0.0f0)
    @inbounds for coordinate in 1:node_dim
        pooled = 0.0f0
        for block in 1:model.blocks
            node = coordinate + (block - 1) * node_dim
            pooled += arena.membrane[node, model.cycles + 1, flat]
        end
        scratch.features[coordinate] =
            arena.workspace[coordinate, model.cycles + 1, flat]
        scratch.features[node_dim + coordinate] =
            arena.query[coordinate, flat]
        scratch.features[2 * node_dim + coordinate] =
            pooled / Float32(model.blocks)
    end

    @inbounds for output in 1:OUTPUT_DIM
        cotangent = arena.raw_gradient[output, flat]
        gradient.output_bias[output] += cotangent
        for hidden in 1:model.hidden
            gradient.output_weight[output, hidden] = muladd(
                cotangent,
                arena.hidden[hidden, flat],
                gradient.output_weight[output, hidden],
            )
            scratch.dhidden[hidden] = muladd(
                parameters.output_weight[output, hidden],
                cotangent,
                scratch.dhidden[hidden],
            )
        end
    end
    @inbounds for hidden in 1:model.hidden
        hidden_value = arena.hidden[hidden, flat]
        cotangent =
            scratch.dhidden[hidden] *
            (1.0f0 - hidden_value * hidden_value)
        gradient.head_bias[hidden] += cotangent
        for feature in 1:(3 * node_dim)
            value = scratch.features[feature]
            gradient.head_weight[hidden, feature] = muladd(
                cotangent,
                value,
                gradient.head_weight[hidden, feature],
            )
            scratch.dfeatures[feature] = muladd(
                parameters.head_weight[hidden, feature],
                cotangent,
                scratch.dfeatures[feature],
            )
        end
    end

    dmembrane_next = scratch.dmembrane_a
    dmembrane_previous = scratch.dmembrane_b
    fill!(dmembrane_next, 0.0f0)
    @inbounds for block in 1:model.blocks, coordinate in 1:node_dim
        node = coordinate + (block - 1) * node_dim
        dmembrane_next[node] =
            scratch.dfeatures[2 * node_dim + coordinate] /
            Float32(model.blocks)
    end
    dworkspace_next = scratch.dworkspace_a
    dworkspace_previous = scratch.dworkspace_b
    @inbounds for coordinate in 1:node_dim
        dworkspace_next[coordinate] = scratch.dfeatures[coordinate]
        scratch.dquery[coordinate] =
            scratch.dfeatures[node_dim + coordinate]
    end
    fill!(scratch.dactive, 0.0f0)

    @inbounds for cycle in model.cycles:-1:1
        fill!(dmembrane_previous, 0.0f0)
        fill!(dworkspace_previous, 0.0f0)
        fill!(scratch.dprevious_active, 0.0f0)
        fill!(scratch.dblock_mask, 0.0f0)

        # Synapse VJP. OFF edges contribute only to the straight-through gate.
        for relation in 1:model.fanout
            for source in 1:nodes
                destination =
                    model.destination_for_source[source, relation]
                output_cotangent = dmembrane_next[destination]
                delay = cache.delay[source, relation]
                current = arena.active_spikes[source, cycle, flat]
                previous = cycle == 1 ? 0.0f0 :
                    arena.active_spikes[source, cycle - 1, flat]
                signal = muladd(1.0f0 - delay, current, delay * previous)
                weight = parameters.synapse_weight[source, relation]
                gate = cache.gate_hard[source, relation]
                gradient.gate_logits[source, relation] = muladd(
                    output_cotangent * signal * weight,
                    cache.gate_derivative[source, relation],
                    gradient.gate_logits[source, relation],
                )
                gate == 0.0f0 && continue
                gradient.synapse_weight[source, relation] = muladd(
                    output_cotangent,
                    signal,
                    gradient.synapse_weight[source, relation],
                )
                signal_cotangent = output_cotangent * weight
                gradient.delay_logits[source, relation] = muladd(
                    signal_cotangent * (previous - current),
                    cache.delay_derivative[source, relation],
                    gradient.delay_logits[source, relation],
                )
                scratch.dactive[source] = muladd(
                    signal_cotangent,
                    1.0f0 - delay,
                    scratch.dactive[source],
                )
                scratch.dprevious_active[source] = muladd(
                    signal_cotangent,
                    delay,
                    scratch.dprevious_active[source],
                )
            end
        end

        # Direct membrane, seed, feedback, and reset paths.
        for block in 1:model.blocks, coordinate in 1:node_dim
            node = coordinate + (block - 1) * node_dim
            cotangent = dmembrane_next[node]
            membrane = arena.membrane[node, cycle, flat]
            dmembrane_previous[node] = muladd(
                cotangent,
                cache.leak[node],
                dmembrane_previous[node],
            )
            gradient.leak_logits[node] = muladd(
                cotangent * membrane,
                cache.leak_derivative[node],
                gradient.leak_logits[node],
            )
            scratch.dseed[node] = muladd(
                0.18f0,
                cotangent,
                scratch.dseed[node],
            )
            workspace_value =
                arena.workspace[coordinate, cycle + 1, flat]
            gradient.feedback_gain[coordinate, block] = muladd(
                cotangent,
                workspace_value,
                gradient.feedback_gain[coordinate, block],
            )
            dworkspace_next[coordinate] = muladd(
                cotangent,
                parameters.feedback_gain[coordinate, block],
                dworkspace_next[coordinate],
            )
        end

        # Workspace recurrence and write.
        for coordinate in 1:node_dim
            workspace_value =
                arena.workspace[coordinate, cycle + 1, flat]
            dz = dworkspace_next[coordinate] *
                (1.0f0 - workspace_value * workspace_value)
            previous_workspace =
                arena.workspace[coordinate, cycle, flat]
            gradient.workspace_decay_logit[1] = muladd(
                dz * previous_workspace,
                cache.workspace_decay_derivative,
                gradient.workspace_decay_logit[1],
            )
            dworkspace_previous[coordinate] +=
                dz * cache.workspace_decay
            write_cotangent = dz / Float32(model.workspace_k)
            for block in 1:model.blocks
                node = coordinate + (block - 1) * node_dim
                mask = arena.block_mask[block, cycle, flat]
                dmembrane_previous[node] = muladd(
                    write_cotangent,
                    mask,
                    dmembrane_previous[node],
                )
                scratch.dblock_mask[block] = muladd(
                    write_cotangent,
                    arena.membrane[node, cycle, flat],
                    scratch.dblock_mask[block],
                )
            end
        end

        # Active spike and surrogate spike VJP.
        for block in 1:model.blocks, coordinate in 1:node_dim
            node = coordinate + (block - 1) * node_dim
            membrane = arena.membrane[node, cycle, flat]
            spike = membrane >= cache.threshold[node] ? 1.0f0 : 0.0f0
            mask = arena.block_mask[block, cycle, flat]
            active_cotangent = scratch.dactive[node]
            scratch.dblock_mask[block] = muladd(
                active_cotangent,
                spike,
                scratch.dblock_mask[block],
            )
            spike_cotangent =
                active_cotangent * mask -
                dmembrane_next[node] * cache.threshold[node]
            soft = sigmoid(
                (membrane - cache.threshold[node]) /
                model.spike_temperature,
            )
            surrogate = soft * (1.0f0 - soft) / model.spike_temperature
            normalized_cotangent = spike_cotangent * surrogate
            dmembrane_previous[node] += normalized_cotangent
            threshold_cotangent =
                -dmembrane_next[node] * spike - normalized_cotangent
            gradient.threshold_logits[node] = muladd(
                threshold_cotangent,
                cache.threshold_derivative[node],
                gradient.threshold_logits[node],
            )
        end

        # Normalized sigmoid routing VJP and score VJP.
        _compute_scores!(
            scratch, arena, model, parameters, cycle, flat,
        )
        route_sum = 0.0f0
        route_projection = 0.0f0
        for block in 1:model.blocks
            soft = sigmoid(
                scratch.scores[block] / model.route_temperature,
            )
            scratch.soft_route[block] = soft
            route_sum += soft
            route_projection = muladd(
                scratch.dblock_mask[block],
                soft,
                route_projection,
            )
        end
        inverse_route_sum = inv(max(route_sum, eps(Float32)))
        for block in 1:model.blocks
            soft = scratch.soft_route[block]
            score_cotangent =
                Float32(model.workspace_k) *
                inverse_route_sum *
                (
                    scratch.dblock_mask[block] -
                    route_projection * inverse_route_sum
                ) *
                soft *
                (1.0f0 - soft) /
                model.route_temperature
            offset = (block - 1) * node_dim
            for coordinate in 1:node_dim
                node = offset + coordinate
                membrane = arena.membrane[node, cycle, flat]
                query = arena.query[coordinate, flat]
                key = parameters.workspace_key[coordinate, block]
                gradient.workspace_key[coordinate, block] = muladd(
                    score_cotangent * membrane,
                    query,
                    gradient.workspace_key[coordinate, block],
                )
                scratch.dquery[coordinate] = muladd(
                    score_cotangent * membrane,
                    key,
                    scratch.dquery[coordinate],
                )
                sign_membrane = membrane > 0.0f0 ? 1.0f0 :
                    (membrane < 0.0f0 ? -1.0f0 : 0.0f0)
                dmembrane_previous[node] += score_cotangent * (
                    key * query + 0.05f0 * sign_membrane
                )
            end
        end

        copyto!(scratch.dactive, scratch.dprevious_active)
        dmembrane_next, dmembrane_previous =
            dmembrane_previous, dmembrane_next
        dworkspace_next, dworkspace_previous =
            dworkspace_previous, dworkspace_next
    end

    # Initial membrane is the sensory seed. It receives both recurrent and
    # explicit 0.18*seed cotangents from every cycle.
    @inbounds for node in 1:nodes
        scratch.dseed[node] += dmembrane_next[node]
        rail = arena.rails[model.feature_for_node[node], flat]
        gradient.input_gain[node] = muladd(
            scratch.dseed[node],
            rail,
            gradient.input_gain[node],
        )
        gradient.input_bias[node] += scratch.dseed[node]
    end
    @inbounds for coordinate in 1:node_dim
        query = arena.query[coordinate, flat]
        query_cotangent =
            scratch.dquery[coordinate] * (1.0f0 - query * query)
        for rail in 1:Model.INPUT_RAILS
            gradient.query_weight[coordinate, rail] = muladd(
                query_cotangent,
                arena.rails[rail, flat],
                gradient.query_weight[coordinate, rail],
            )
        end
    end
    return nothing
end

function arena_output(arena::TrainingArena)
    return (;
        q=vec(@view(arena.raw[1:1, :])),
        death_logit=vec(@view(arena.raw[2:2, :])),
        quantiles=@view(arena.raw[3:18, :]),
        geometry=@view(arena.raw[19:22, :]),
    )
end

const PARAMETER_FIELDS = (
    :input_gain,
    :input_bias,
    :query_weight,
    :workspace_key,
    :feedback_gain,
    :leak_logits,
    :threshold_logits,
    :synapse_weight,
    :gate_logits,
    :delay_logits,
    :workspace_decay_logit,
    :head_weight,
    :head_bias,
    :output_weight,
    :output_bias,
)

struct ParameterShard
    field::UInt8
    first::Int32
    last::Int32
end

function _parameter_shards(parameters; elements_per_shard::Int=4096)
    keys(parameters) == PARAMETER_FIELDS || error(
        "SerialWorkspaceSNN parameter registry changed",
    )
    elements_per_shard >= 256 || throw(ArgumentError(
        "parameter shard size must be at least 256",
    ))
    shards = ParameterShard[]
    for (field, name) in enumerate(PARAMETER_FIELDS)
        length_array = length(getproperty(parameters, name))
        first_index = 1
        while first_index <= length_array
            last_index = min(
                first_index + elements_per_shard - 1,
                length_array,
            )
            push!(shards, ParameterShard(
                UInt8(field),
                Int32(first_index),
                Int32(last_index),
            ))
            first_index = last_index + 1
        end
    end
    length(shards) <= typemax(UInt16) || error("too many parameter shards")
    return shards
end

mutable struct ArenaAdamW{M,V}
    first_moment::M
    second_moment::V
    learning_rate::Float32
    beta1::Float32
    beta2::Float32
    beta1_power::Float32
    beta2_power::Float32
    epsilon::Float32
    weight_decay::Float32
    step::Int
end

function ArenaAdamW(
    parameters;
    learning_rate::Real=5.0f-4,
    beta1::Real=0.9,
    beta2::Real=0.999,
    epsilon::Real=1.0f-8,
    weight_decay::Real=1.0f-5,
)
    b1 = Float32(beta1)
    b2 = Float32(beta2)
    return ArenaAdamW(
        _zero_parameter_tree(parameters),
        _zero_parameter_tree(parameters),
        Float32(learning_rate),
        b1,
        b2,
        b1,
        b2,
        Float32(epsilon),
        Float32(weight_decay),
        0,
    )
end

mutable struct ArenaMetrics
    wall_seconds::Float64
    cpu_seconds::Float64
    allocation_bytes::Int128
    gc_seconds::Float64
    pack_seconds::Float64
    forward_seconds::Float64
    loss_seconds::Float64
    backward_seconds::Float64
    optimizer_seconds::Float64
    consolidation_seconds::Float64
    whole_machine_cpu_percent::Float64
    active_worker_cpu_percent::Float64
    states_per_second::Float64
end

ArenaMetrics() = ArenaMetrics(
    0.0, 0.0, Int128(0), 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
)

mutable struct ArenaTrainer{M,P,O,G}
    model::M
    parameters::P
    cache::ParameterCache
    optimizer::O
    arena::TrainingArena
    loss_scratch::LossScratch
    gradient::G
    parameter_shards::Vector{ParameterShard}
    gradient_norm_squares::Vector{Float64}
    consolidation_flips::Vector{Int}
    structure_weight::Float32
    structure_gradient_coefficient::Float32
    last_loss::LossRecord
    last_gradient_norm::Float64
    total_structural_flips::Int
    metrics::ArenaMetrics
end

function ArenaTrainer(
    model,
    parameters;
    state_batch::Int=4,
    width::Int=80,
    learning_rate::Real=5.0f-4,
    weight_decay::Real=1.0f-5,
    structure_weight::Real=1.0f-2,
    parameter_shard_size::Int=4096,
)
    arena = TrainingArena(model, state_batch, width)
    shards = _parameter_shards(
        parameters;
        elements_per_shard=parameter_shard_size,
    )
    empty_loss = LossRecord(
        NaN32, NaN32, NaN32, NaN32, NaN32,
        NaN32, NaN32, NaN32, NaN32, 0,
    )
    return ArenaTrainer(
        model,
        parameters,
        ParameterCache(parameters),
        ArenaAdamW(
            parameters;
            learning_rate,
            weight_decay,
        ),
        arena,
        LossScratch(width),
        _zero_parameter_tree(parameters),
        shards,
        zeros(Float64, length(shards)),
        zeros(Int, cld(model.blocks * model.node_dim, 32)),
        Float32(structure_weight),
        0.0f0,
        empty_loss,
        NaN,
        0,
        ArenaMetrics(),
    )
end

training_arena(trainer::ArenaTrainer) = trainer.arena

mutable struct ArenaWorkerRuntime{G}
    gradient::G
    scratch::CandidateScratch
    jobs::UInt64
    cpu_ticks::UInt64
end

ArenaWorkerRuntime(model, parameters) = ArenaWorkerRuntime(
    _zero_parameter_tree(parameters),
    CandidateScratch(model),
    UInt64(0),
    UInt64(0),
)

@enum ArenaWorkKind::UInt8 begin
    ARENA_NO_WORK = 0
    ARENA_PACK = 1
    ARENA_FORWARD = 2
    ARENA_BACKWARD = 3
    ARENA_OPTIMIZER = 4
    ARENA_CONSOLIDATE = 5
end

struct ArenaWorkItem
    kind::UInt8
    target::UInt16
    generation::UInt32
end

ArenaWorkItem(
    kind::ArenaWorkKind,
    target::Integer,
    generation::UInt32,
) = ArenaWorkItem(UInt8(kind), UInt16(target), generation)

Base.zero(::Type{ArenaWorkItem}) =
    ArenaWorkItem(UInt8(ARENA_NO_WORK), UInt16(0), UInt32(0))
isbitstype(ArenaWorkItem) || error("ArenaWorkItem must remain isbits")

mutable struct ArenaExecutor{W,T,D}
    queue::Queue.BoundedMPMCQueue{ArenaWorkItem}
    active_workers::Int
    julia_workers::Int
    cpuset_mode::Symbol
    workers::W
    trainer::T
    dataset::D
    generation::Base.Threads.Atomic{UInt32}
    remaining::Base.Threads.Atomic{Int}
    shutdown_requested::Base.Threads.Atomic{UInt32}
    ready_workers::Base.Threads.Atomic{Int}
    booted_workers::Base.Threads.Atomic{Int}
    failure_worker::Base.Threads.Atomic{Int}
    failures::Vector{Any}
    bindings::Vector{Any}
    startup_event::Base.Event
    started::Bool
end

function ArenaExecutor(
    trainer::ArenaTrainer,
    dataset;
    active_workers::Int=Base.Threads.nthreads(:default),
    cpuset_mode::Symbol=:none,
    queue_capacity::Int=2048,
)
    julia_workers = Base.Threads.nthreads(:default)
    Base.Threads.nthreads(:interactive) == 0 || error(
        "launch Julia with --threads=N,0",
    )
    2 <= active_workers <= julia_workers || throw(ArgumentError(
        "active_workers must be in 2:$julia_workers",
    ))
    cpuset_mode in (:none, :all, :p_only) || throw(ArgumentError(
        "cpuset_mode must be none, all, or p_only",
    ))
    ispow2(queue_capacity) || throw(ArgumentError(
        "queue_capacity must be a power of two",
    ))
    workers = [
        ArenaWorkerRuntime(trainer.model, trainer.parameters)
        for _ in 1:active_workers
    ]
    return ArenaExecutor(
        Queue.BoundedMPMCQueue{ArenaWorkItem}(
            queue_capacity,
            zero(ArenaWorkItem),
        ),
        active_workers,
        julia_workers,
        cpuset_mode,
        workers,
        trainer,
        dataset,
        Base.Threads.Atomic{UInt32}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{UInt32}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Any[nothing for _ in 1:julia_workers],
        Any[nothing for _ in 1:julia_workers],
        Base.Event(true),
        false,
    )
end

@inline function _adam_range!(
    trainer::ArenaTrainer,
    executor::ArenaExecutor,
    field::Val{F},
    first_index::Int,
    last_index::Int,
) where {F}
    parameter = getproperty(trainer.parameters, F)
    global_gradient = getproperty(trainer.gradient, F)
    first_moment = getproperty(trainer.optimizer.first_moment, F)
    second_moment = getproperty(trainer.optimizer.second_moment, F)
    optimizer = trainer.optimizer
    inverse_first_bias = inv(1.0f0 - optimizer.beta1_power)
    inverse_second_bias = inv(1.0f0 - optimizer.beta2_power)
    beta1_complement = 1.0f0 - optimizer.beta1
    beta2_complement = 1.0f0 - optimizer.beta2
    norm_square = 0.0
    @inbounds for index in first_index:last_index
        value = 0.0f0
        for worker in executor.workers
            source = getproperty(worker.gradient, F)
            value += source[index]
            source[index] = 0.0f0
        end
        if F === :gate_logits
            value += trainer.structure_gradient_coefficient *
                trainer.cache.gate_derivative[index]
        end
        global_gradient[index] = value
        norm_square = muladd(Float64(value), Float64(value), norm_square)
        moment = muladd(
            optimizer.beta1,
            first_moment[index],
            beta1_complement * value,
        )
        variance = muladd(
            optimizer.beta2,
            second_moment[index],
            beta2_complement * value * value,
        )
        first_moment[index] = moment
        second_moment[index] = variance
        adam = (moment * inverse_first_bias) / (
            sqrt(variance * inverse_second_bias) + optimizer.epsilon
        )
        old_parameter = parameter[index]
        parameter[index] = old_parameter - optimizer.learning_rate * (
            adam + optimizer.weight_decay * old_parameter
        )
    end
    return norm_square
end

@inline function _refresh_cache_range!(
    cache::ParameterCache,
    parameters,
    field::UInt8,
    first_index::Int,
    last_index::Int,
)
    if field == 6
        @inbounds for index in first_index:last_index
            probability = sigmoid(parameters.leak_logits[index])
            cache.leak[index] = 0.45f0 + 0.50f0 * probability
            cache.leak_derivative[index] =
                0.50f0 * probability * (1.0f0 - probability)
        end
    elseif field == 7
        @inbounds for index in first_index:last_index
            probability = sigmoid(parameters.threshold_logits[index])
            cache.threshold[index] = 0.25f0 + 0.75f0 * probability
            cache.threshold_derivative[index] =
                0.75f0 * probability * (1.0f0 - probability)
        end
    elseif field == 9
        @inbounds for index in first_index:last_index
            probability = sigmoid(parameters.gate_logits[index])
            cache.gate_probability[index] = probability
            cache.gate_hard[index] =
                parameters.gate_logits[index] >= 0.0f0 ? 1.0f0 : 0.0f0
            cache.gate_derivative[index] =
                probability * (1.0f0 - probability)
        end
    elseif field == 10
        @inbounds for index in first_index:last_index
            probability = sigmoid(parameters.delay_logits[index])
            cache.delay[index] = probability
            cache.delay_derivative[index] =
                probability * (1.0f0 - probability)
        end
    elseif field == 11
        probability = sigmoid(parameters.workspace_decay_logit[1])
        cache.workspace_decay = probability
        cache.workspace_decay_derivative =
            probability * (1.0f0 - probability)
    end
    return nothing
end

function _optimizer_shard!(
    trainer::ArenaTrainer,
    executor::ArenaExecutor,
    target::Int,
)
    shard = @inbounds trainer.parameter_shards[target]
    field = Int(shard.field)
    first_index = Int(shard.first)
    last_index = Int(shard.last)
    norm_square = if field == 1
        _adam_range!(trainer, executor, Val(:input_gain), first_index, last_index)
    elseif field == 2
        _adam_range!(trainer, executor, Val(:input_bias), first_index, last_index)
    elseif field == 3
        _adam_range!(trainer, executor, Val(:query_weight), first_index, last_index)
    elseif field == 4
        _adam_range!(trainer, executor, Val(:workspace_key), first_index, last_index)
    elseif field == 5
        _adam_range!(trainer, executor, Val(:feedback_gain), first_index, last_index)
    elseif field == 6
        _adam_range!(trainer, executor, Val(:leak_logits), first_index, last_index)
    elseif field == 7
        _adam_range!(trainer, executor, Val(:threshold_logits), first_index, last_index)
    elseif field == 8
        _adam_range!(trainer, executor, Val(:synapse_weight), first_index, last_index)
    elseif field == 9
        _adam_range!(trainer, executor, Val(:gate_logits), first_index, last_index)
    elseif field == 10
        _adam_range!(trainer, executor, Val(:delay_logits), first_index, last_index)
    elseif field == 11
        _adam_range!(
            trainer,
            executor,
            Val(:workspace_decay_logit),
            first_index,
            last_index,
        )
    elseif field == 12
        _adam_range!(trainer, executor, Val(:head_weight), first_index, last_index)
    elseif field == 13
        _adam_range!(trainer, executor, Val(:head_bias), first_index, last_index)
    elseif field == 14
        _adam_range!(trainer, executor, Val(:output_weight), first_index, last_index)
    elseif field == 15
        _adam_range!(trainer, executor, Val(:output_bias), first_index, last_index)
    else
        error("unknown parameter field $field")
    end
    trainer.gradient_norm_squares[target] = norm_square
    _refresh_cache_range!(
        trainer.cache,
        trainer.parameters,
        shard.field,
        first_index,
        last_index,
    )
    return nothing
end

function _consolidate_node_range!(
    trainer::ArenaTrainer,
    scratch::CandidateScratch,
    target::Int,
)
    model = trainer.model
    parameters = trainer.parameters
    cache = trainer.cache
    first_node = (target - 1) * 32 + 1
    last_node = min(target * 32, model.blocks * model.node_dim)
    flips = 0
    keep = clamp(round(Int, 0.50 * model.fanout), 1, model.fanout - 1)
    @inbounds for node in first_node:last_node
        fill!(scratch.consolidation_selected, false)
        for relation in 1:model.fanout
            scratch.consolidation_evidence[relation] =
                abs(parameters.synapse_weight[node, relation]) *
                cache.gate_probability[node, relation]
        end
        for _ in 1:keep
            best = 0
            best_evidence = -Inf32
            for relation in 1:model.fanout
                scratch.consolidation_selected[relation] && continue
                evidence = scratch.consolidation_evidence[relation]
                if best == 0 || evidence > best_evidence
                    best = relation
                    best_evidence = evidence
                end
            end
            scratch.consolidation_selected[best] = true
        end
        for relation in 1:model.fanout
            old_hard = cache.gate_hard[node, relation]
            magnitude = max(
                abs(parameters.gate_logits[node, relation]),
                0.02f0,
            )
            selected = scratch.consolidation_selected[relation]
            parameters.gate_logits[node, relation] =
                selected ? magnitude : -magnitude
            probability = sigmoid(parameters.gate_logits[node, relation])
            cache.gate_probability[node, relation] = probability
            cache.gate_hard[node, relation] = selected ? 1.0f0 : 0.0f0
            cache.gate_derivative[node, relation] =
                probability * (1.0f0 - probability)
            flips += old_hard != cache.gate_hard[node, relation]
        end
    end
    trainer.consolidation_flips[target] = flips
    return nothing
end

function _gate_density(cache::ParameterCache)
    total = 0.0f0
    @inbounds for probability in cache.gate_probability
        total += probability
    end
    return total / Float32(length(cache.gate_probability))
end

function _clear_worker_gradients!(executor::ArenaExecutor)
    @inbounds for worker in executor.workers
        _fill_parameter_tree!(worker.gradient)
    end
    return nothing
end

function _reduce_worker_gradients!(
    trainer::ArenaTrainer,
    executor::ArenaExecutor,
)
    _fill_parameter_tree!(trainer.gradient)
    @inbounds for name in PARAMETER_FIELDS
        destination = getproperty(trainer.gradient, name)
        for index in eachindex(destination)
            value = 0.0f0
            for worker in executor.workers
                value += getproperty(worker.gradient, name)[index]
            end
            destination[index] = value
        end
    end
    coefficient = trainer.structure_gradient_coefficient
    @inbounds for index in eachindex(trainer.gradient.gate_logits)
        trainer.gradient.gate_logits[index] +=
            coefficient * trainer.cache.gate_derivative[index]
    end
    trainer.last_gradient_norm = arena_tree_norm(trainer.gradient)
    return trainer.gradient
end

function fill_next_rows!(destination::Vector{Int}, sampler)
    batch_size = length(destination)
    batch_size > 0 || throw(ArgumentError("destination cannot be empty"))
    filled = 0
    while filled < batch_size
        if sampler.cursor > length(sampler.permutation)
            sampler.permutation .= sampler.source_rows
            shuffle!(sampler.rng, sampler.permutation)
            sampler.cursor = 1
        end
        available = length(sampler.permutation) - sampler.cursor + 1
        taken = min(batch_size - filled, available)
        @inbounds for offset in 0:(taken - 1)
            destination[filled + offset + 1] =
                sampler.permutation[sampler.cursor + offset]
        end
        sampler.cursor += taken
        filled += taken
        if sampler.cursor > length(sampler.permutation)
            sampler.completed_epochs += 1
        end
    end
    return destination
end

@inline function _complete_work!(executor::ArenaExecutor)
    previous = Base.Threads.atomic_add!(executor.remaining, -1)
    previous >= 1 || error("arena work counter underflow")
    previous == 1 && Queue.wake_consumers!(executor.queue)
    return nothing
end

function _dispatch!(
    executor::ArenaExecutor,
    worker_slot::Int,
    work::ArenaWorkItem,
)
    work.generation == executor.generation[] || error(
        "stale arena work generation",
    )
    trainer = executor.trainer
    worker = @inbounds executor.workers[worker_slot]
    target = Int(work.target)
    cpu_started = CpuSets.thread_cpu_ticks_100ns()
    if work.kind == UInt8(ARENA_PACK)
        flat = Int(trainer.arena.valid_flats[target])
        pack_candidate_rails!(
            trainer.arena,
            executor.dataset,
            worker.scratch.pack,
            flat,
        )
    elseif work.kind == UInt8(ARENA_FORWARD)
        flat = Int(trainer.arena.valid_flats[target])
        forward_candidate!(
            trainer.arena,
            trainer.model,
            trainer.parameters,
            trainer.cache,
            worker.scratch,
            flat,
        )
    elseif work.kind == UInt8(ARENA_BACKWARD)
        flat = Int(trainer.arena.valid_flats[target])
        backward_candidate!(
            worker.gradient,
            trainer.arena,
            trainer.model,
            trainer.parameters,
            trainer.cache,
            worker.scratch,
            flat,
        )
    elseif work.kind == UInt8(ARENA_OPTIMIZER)
        _optimizer_shard!(trainer, executor, target)
    elseif work.kind == UInt8(ARENA_CONSOLIDATE)
        _consolidate_node_range!(trainer, worker.scratch, target)
    else
        error("unknown arena work kind $(work.kind)")
    end
    worker.jobs += UInt64(1)
    worker.cpu_ticks +=
        CpuSets.thread_cpu_ticks_100ns() - cpu_started
    _complete_work!(executor)
    return nothing
end

function _record_failure!(
    executor::ArenaExecutor,
    worker_slot::Int,
    exception,
    backtrace,
)
    executor.failures[worker_slot] = (exception, backtrace)
    Base.Threads.atomic_cas!(executor.failure_worker, 0, worker_slot)
    Base.Threads.atomic_xchg!(executor.shutdown_requested, UInt32(1))
    Queue.close!(executor.queue)
    notify(executor.startup_event)
    return nothing
end

function _throw_failure(executor::ArenaExecutor)
    worker = executor.failure_worker[]
    worker == 0 && return nothing
    payload = executor.failures[worker]
    payload === nothing && error("arena worker $worker failed without payload")
    exception, backtrace = payload
    throw(Base.CapturedException(exception, backtrace))
end

function _worker_loop!(executor::ArenaExecutor, worker_slot::Int)
    while executor.shutdown_requested[] == 0
        available, work = Queue.dequeue_wait!(
            executor.queue;
            timeout_ms=100,
        )
        if !available
            Queue.isclosed(executor.queue) && return nothing
            continue
        end
        _dispatch!(executor, worker_slot, work)
    end
    return nothing
end

function _coordinator_drain!(executor::ArenaExecutor)
    while executor.remaining[] > 0
        _throw_failure(executor)
        available, work = Queue.try_dequeue!(executor.queue)
        if available
            _dispatch!(executor, 1, work)
            continue
        end
        expected = Queue.item_epoch(executor.queue)
        executor.remaining[] == 0 && break
        Queue.wait_for_item_change!(
            executor.queue,
            expected;
            timeout_ms=10,
        )
    end
    _throw_failure(executor)
    executor.remaining[] == 0 || error("arena phase ended early")
    return nothing
end

function _run_phase!(
    executor::ArenaExecutor,
    kind::ArenaWorkKind,
    count::Int,
    generation::UInt32,
)
    count >= 0 || throw(ArgumentError("phase count cannot be negative"))
    count == 0 && return 0.0
    count <= typemax(UInt16) || error("arena phase has too many jobs")
    started = time_ns()
    executor.remaining[] = count
    @inbounds for target in 1:count
        Queue.enqueue_wait!(
            executor.queue,
            ArenaWorkItem(kind, target, generation);
            timeout_ms=10_000,
        ) || error("arena queue closed during publication")
    end
    _coordinator_drain!(executor)
    Queue.approx_length(executor.queue) == 0 || error(
        "arena queue is not empty at phase boundary",
    )
    return (time_ns() - started) * 1.0e-9
end

function pack_arena_batch!(
    arena::TrainingArena,
    dataset,
)
    prepare_batch_metadata!(arena, dataset)
    scratch = PackScratch()
    @inbounds for target in 1:arena.valid_count
        pack_candidate_rails!(
            arena,
            dataset,
            scratch,
            Int(arena.valid_flats[target]),
        )
    end
    return arena
end

function _run_forward_backward!(
    trainer::ArenaTrainer,
    executor::ArenaExecutor,
    generation::UInt32,
)
    pack_started = time_ns()
    prepare_batch_metadata!(trainer.arena, executor.dataset)
    pack_parallel_seconds = _run_phase!(
        executor,
        ARENA_PACK,
        trainer.arena.valid_count,
        generation,
    )
    pack_seconds =
        (time_ns() - pack_started) * 1.0e-9
    forward_seconds = _run_phase!(
        executor,
        ARENA_FORWARD,
        trainer.arena.valid_count,
        generation,
    )
    gate_density = _gate_density(trainer.cache)
    loss_started = time_ns()
    trainer.last_loss = loss_and_raw_gradient!(
        trainer.arena,
        trainer.loss_scratch,
        gate_density,
        trainer.structure_weight,
    )
    loss_seconds = (time_ns() - loss_started) * 1.0e-9
    trainer.structure_gradient_coefficient =
        2.0f0 *
        trainer.structure_weight *
        (gate_density - 0.50f0) /
        Float32(length(trainer.cache.gate_probability))
    backward_seconds = _run_phase!(
        executor,
        ARENA_BACKWARD,
        trainer.arena.valid_count,
        generation,
    )
    return (;
        pack_seconds,
        pack_parallel_seconds,
        forward_seconds,
        loss_seconds,
        backward_seconds,
    )
end

function arena_gradient!(
    executor::ArenaExecutor,
)
    executor.started || error("arena team is not running")
    Queue.approx_length(executor.queue) == 0 || error("arena queue is not empty")
    _clear_worker_gradients!(executor)
    generation =
        Base.Threads.atomic_add!(executor.generation, UInt32(1)) +
        UInt32(1)
    trainer = executor.trainer
    phases = _run_forward_backward!(trainer, executor, generation)
    gradient = _reduce_worker_gradients!(trainer, executor)
    return (;
        loss=trainer.last_loss,
        gradient,
        raw=trainer.arena.raw,
        raw_gradient=trainer.arena.raw_gradient,
        phases,
    )
end

function arena_update!(
    executor::ArenaExecutor;
    structural_interval::Int=25,
)
    executor.started || error("arena team is not running")
    structural_interval >= 1 || throw(ArgumentError(
        "structural_interval must be positive",
    ))
    Queue.approx_length(executor.queue) == 0 || error("arena queue is not empty")
    trainer = executor.trainer
    for worker in executor.workers
        worker.jobs = 0
        worker.cpu_ticks = 0
    end
    wall_started = time_ns()
    cpu_started = CpuSets.process_cpu_ticks_100ns()
    gc_started = Base.gc_num()
    generation =
        Base.Threads.atomic_add!(executor.generation, UInt32(1)) +
        UInt32(1)
    phases = _run_forward_backward!(trainer, executor, generation)

    optimizer_seconds = _run_phase!(
        executor,
        ARENA_OPTIMIZER,
        length(trainer.parameter_shards),
        generation,
    )
    gradient_norm_square = 0.0
    @inbounds for value in trainer.gradient_norm_squares
        gradient_norm_square += value
    end
    trainer.last_gradient_norm = sqrt(gradient_norm_square)
    trainer.optimizer.step += 1
    trainer.optimizer.beta1_power *= trainer.optimizer.beta1
    trainer.optimizer.beta2_power *= trainer.optimizer.beta2

    consolidation_seconds = 0.0
    if trainer.optimizer.step % structural_interval == 0
        consolidation_seconds = _run_phase!(
            executor,
            ARENA_CONSOLIDATE,
            length(trainer.consolidation_flips),
            generation,
        )
        flips = 0
        @inbounds for value in trainer.consolidation_flips
            flips += value
        end
        trainer.total_structural_flips += flips
    else
        fill!(trainer.consolidation_flips, 0)
    end

    wall_seconds = (time_ns() - wall_started) * 1.0e-9
    cpu_seconds =
        (CpuSets.process_cpu_ticks_100ns() - cpu_started) * 1.0e-7
    gc_difference = Base.GC_Diff(Base.gc_num(), gc_started)
    metrics = trainer.metrics
    metrics.wall_seconds = wall_seconds
    metrics.cpu_seconds = cpu_seconds
    metrics.allocation_bytes = Int128(gc_difference.allocd)
    metrics.gc_seconds = Float64(gc_difference.total_time) * 1.0e-9
    metrics.pack_seconds = phases.pack_seconds
    metrics.forward_seconds = phases.forward_seconds
    metrics.loss_seconds = phases.loss_seconds
    metrics.backward_seconds = phases.backward_seconds
    metrics.optimizer_seconds = optimizer_seconds
    metrics.consolidation_seconds = consolidation_seconds
    metrics.whole_machine_cpu_percent =
        100.0 * cpu_seconds /
        max(wall_seconds * executor.julia_workers, eps(Float64))
    metrics.active_worker_cpu_percent =
        100.0 * cpu_seconds /
        max(wall_seconds * executor.active_workers, eps(Float64))
    metrics.states_per_second =
        trainer.arena.state_batch / max(wall_seconds, eps(Float64))
    isfinite(trainer.last_loss.composite_loss) || error(
        "non-finite arena loss",
    )
    isfinite(trainer.last_gradient_norm) || error(
        "non-finite arena gradient",
    )
    return trainer
end

function run_with_arena_team!(
    body::F,
    executor::ArenaExecutor,
) where {F}
    executor.started && error("arena team is already running")
    Queue.isclosed(executor.queue) && error("cannot restart a closed arena queue")
    topology = CpuSets.discover_topology()
    binding_plan = CpuSets.configure_worker_bindings(
        executor.cpuset_mode,
        executor.active_workers,
        topology,
    )
    executor.ready_workers[] = 0
    executor.booted_workers[] = 0
    executor.failure_worker[] = 0
    executor.shutdown_requested[] = 0
    reset(executor.startup_event)
    executor.started = true
    result = Ref{Any}(nothing)
    try
        Base.Threads.threading_run(worker_slot -> begin
            try
                binding = CpuSets.bind_current_worker!(worker_slot)
                executor.bindings[worker_slot] = binding
                booted =
                    Base.Threads.atomic_add!(executor.booted_workers, 1) + 1
                booted == executor.julia_workers &&
                    notify(executor.startup_event)
                worker_slot <= executor.active_workers || return nothing
                ready =
                    Base.Threads.atomic_add!(executor.ready_workers, 1) + 1
                ready == executor.active_workers &&
                    notify(executor.startup_event)
                if worker_slot == 1
                    while executor.booted_workers[] < executor.julia_workers ||
                          executor.ready_workers[] < executor.active_workers
                        _throw_failure(executor)
                        wait(executor.startup_event)
                    end
                    result[] = body(executor)
                    Queue.approx_length(executor.queue) == 0 || error(
                        "coordinator returned with queued work",
                    )
                    Base.Threads.atomic_xchg!(
                        executor.shutdown_requested,
                        UInt32(1),
                    )
                    Queue.close!(executor.queue)
                else
                    _worker_loop!(executor, worker_slot)
                end
                return nothing
            catch exception
                _record_failure!(
                    executor,
                    min(worker_slot, length(executor.failures)),
                    exception,
                    catch_backtrace(),
                )
                return nothing
            end
        end, true)
    finally
        executor.started = false
    end
    _throw_failure(executor)
    return (;
        result=result[],
        binding_plan,
        bindings=copy(executor.bindings),
    )
end

run_with_arena_team!(
    executor::ArenaExecutor,
    body::F,
) where {F} = run_with_arena_team!(body, executor)

end # module
