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
if !isdefined(Main, :WorkspaceRoutingPolicy)
    Base.include(Main, joinpath(@__DIR__, "WorkspaceRoutingPolicy.jl"))
end
if !isdefined(Main, :BeatFirstTrainingCore)
    Base.include(Main, joinpath(@__DIR__, "..", "training", "core.jl"))
end

const Queue = Main.BoundedMPMCRing
const CpuSets = Main.WinCpuSets
const Model = Main.SerialWorkspaceSNN
const Routing = Main.WorkspaceRoutingPolicy
const TrainingCore = Main.BeatFirstTrainingCore

const OUTPUT_DIM = 22
const QUANTILES = 16
const BOARD_ROWS = 24
const BOARD_COLUMNS = 10
const BOARD_CELLS = BOARD_ROWS * BOARD_COLUMNS
const QUEUE_PIECES = 7
const QUEUE_TOKENS = 6
const AUX_FEATURES = 37
const ARENA_TEAM_ACTIVE = Base.Threads.Atomic{UInt32}(0)
const GATE_SIGN_EPSILON = 1.0f-6
const ROUTING_GUMBEL_LUT_BITS = 12
const ROUTING_GUMBEL_LUT = Float32[
    -log(-log(
        (index - 0.5) / (1 << ROUTING_GUMBEL_LUT_BITS),
    ))
    for index in 1:(1 << ROUTING_GUMBEL_LUT_BITS)
]

export ArenaTrainer,
    ArenaExecutor,
    EPropShadowConfig,
    EPropShadowReport,
    arena_gradient!,
    arena_shadow_report,
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
    listnet_q_gradient::Vector{Float32}
    membrane::Array{Float32,3}
    active_spikes::Array{Float32,3}
    workspace::Array{Float32,3}
    workspace_inv_rms::Vector{Float32}
    selected_pool_inv_rms::Vector{Float32}
    query_pre::Matrix{Float32}
    query::Matrix{Float32}
    query_inv_rms::Vector{Float32}
    hidden_pre::Matrix{Float32}
    hidden::Matrix{Float32}
    hidden_inv_rms::Vector{Float32}
    block_mask::Array{Float32,3}
    route_probability::Array{Float32,3}
    route_policy_probability::Array{Float32,3}
    route_base_probability::Array{Float32,3}
    route_score::Array{Float32,3}
    route_eligibility::Array{Float32,3}
    route_regularizer_gradient::Array{Float32,3}
    route_order::Array{Int16,3}
    route_selection_gap_value::Matrix{Float32}
    route_score_square_sum::Matrix{Float64}
    route_normalized_entropy::Matrix{Float32}
    route_mask_fingerprint::Matrix{UInt64}
    route_cycle_churn_count::Matrix{Int16}
end

function TrainingArena(model, state_batch::Int, width::Int)
    state_batch >= 1 || throw(ArgumentError("state_batch must be positive"))
    width >= 1 || throw(ArgumentError("candidate width must be positive"))
    nodes = model.blocks * model.node_dim
    capacity = state_batch * width
    route_policy_probability =
        zeros(Float32, model.blocks, model.cycles, capacity)
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
        zeros(Float32, capacity),
        zeros(Float32, nodes, model.cycles + 1, capacity),
        zeros(Float32, nodes, model.cycles, capacity),
        zeros(Float32, model.node_dim, model.cycles + 1, capacity),
        zeros(Float32, capacity),
        zeros(Float32, capacity),
        zeros(Float32, model.node_dim, capacity),
        zeros(Float32, model.node_dim, capacity),
        zeros(Float32, capacity),
        zeros(Float32, model.hidden, capacity),
        zeros(Float32, model.hidden, capacity),
        zeros(Float32, capacity),
        zeros(Float32, model.blocks, model.cycles, capacity),
        route_policy_probability,
        route_policy_probability,
        zeros(Float32, model.blocks, model.cycles, capacity),
        zeros(Float32, model.blocks, model.cycles, capacity),
        zeros(Float32, model.blocks, model.cycles, capacity),
        zeros(Float32, model.blocks, model.cycles, capacity),
        zeros(Int16, model.workspace_k, model.cycles, capacity),
        zeros(Float32, model.cycles, capacity),
        zeros(Float64, model.cycles, capacity),
        zeros(Float32, model.cycles, capacity),
        zeros(UInt64, model.cycles, capacity),
        zeros(Int16, model.cycles, capacity),
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

function _zero_head_gradient_tree(parameters)
    return (;
        head_weight=zeros(Float32, size(parameters.head_weight)),
        head_bias=zeros(Float32, size(parameters.head_bias)),
        output_weight=zeros(Float32, size(parameters.output_weight)),
        output_bias=zeros(Float32, size(parameters.output_bias)),
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
    cache.workspace_decay =
        Model.bounded_workspace_decay(parameters.workspace_decay_logit[1])
    cache.workspace_decay_derivative =
        Model.bounded_workspace_decay_derivative(
            parameters.workspace_decay_logit[1],
        )
    return cache
end

mutable struct LossScratch
    student_z::Vector{Float32}
    teacher_probability::Vector{Float32}
    student_probability::Vector{Float32}
    z_cotangent::Vector{Float32}
    state_composite::Vector{Float32}
    state_teacher_entropy::Vector{Float32}
end

LossScratch(width::Int, state_capacity::Int=width) = LossScratch(
    zeros(Float32, width),
    zeros(Float32, width),
    zeros(Float32, width),
    zeros(Float32, width),
    zeros(Float32, state_capacity),
    zeros(Float32, state_capacity),
)

struct LossRecord
    composite_loss::Float32
    listnet_loss::Float32
    teacher_entropy::Float32
    listnet_kl::Float32
    old_q_loss::Float32
    q_huber_loss::Float32
    margin_loss::Float32
    raw_top_gap_loss::Float32
    death_loss::Float32
    quantile_teacher_loss::Float32
    geometry_loss::Float32
    line_clear_loss::Float32
    max_height_loss::Float32
    holes_loss::Float32
    cavities_loss::Float32
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
    fill!(arena.listnet_q_gradient, 0.0f0)
    raw = arena.raw
    draw = arena.raw_gradient
    targets = arena.targets
    width = arena.width
    state_batch = arena.state_batch
    valid_total = arena.valid_count
    state_batch <= length(scratch.state_composite) ||
        throw(DimensionMismatch("state loss scratch"))
    fill!(scratch.state_composite, 0.0f0)
    fill!(scratch.state_teacher_entropy, 0.0f0)
    listnet_loss = 0.0f0
    teacher_entropy = 0.0f0
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
        for candidate in 1:count
            death_count +=
                targets.death_mask[candidate, state_slot] != 0.0f0
        end
    end
    death_denominator = max(death_count, 1.0f0)
    @inbounds for state_slot in 1:state_batch
        listnet_before = listnet_loss
        teacher_entropy_before = teacher_entropy
        old_q_before = old_q_loss
        margin_before = margin_loss
        death_before = death_loss
        quantile_before = quantile_loss
        line_before = line_clear_loss
        height_before = max_height_loss
        holes_before = holes_loss
        cavities_before = cavities_loss
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
            teacher_entropy -= teacher_probability *
                log(max(teacher_probability, 1.0f-12)) /
                Float32(state_batch)
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
            listnet_q_cotangent = (
                scratch.z_cotangent[candidate] -
                mean_g -
                scratch.student_z[candidate] * mean_gz
            ) * inverse_scale
            draw[1, flat] += listnet_q_cotangent
            arena.listnet_q_gradient[flat] = listnet_q_cotangent
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
        scratch.state_teacher_entropy[state_slot] =
            teacher_entropy - teacher_entropy_before
        scratch.state_composite[state_slot] =
            (listnet_loss - listnet_before) +
            0.25f0 * (old_q_loss - old_q_before) +
            0.15f0 * (margin_loss - margin_before) +
            0.10f0 * (death_loss - death_before) /
                death_denominator +
            0.05f0 * (quantile_loss - quantile_before) +
            0.025f0 * (
                line_clear_loss - line_before +
                max_height_loss - height_before +
                holes_loss - holes_before +
                cavities_loss - cavities_before
            )
    end
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
    # Compatibility names have an explicit identity contract. `old_q_loss`
    # is the canonical unweighted valid-candidate Q Huber and `margin_loss`
    # is the canonical unweighted per-state raw teacher-top1/top2 gap Huber.
    # Their newer spellings must remain bit-identical, not independently
    # recomputed approximations with potentially different reduction order.
    q_huber_loss = old_q_loss
    raw_top_gap_loss = margin_loss
    structure_loss =
        structure_weight * (gate_density - 0.50f0)^2
    structure_per_state = structure_loss / Float32(state_batch)
    @inbounds for state_slot in 1:state_batch
        scratch.state_composite[state_slot] += structure_per_state
    end
    composite_loss =
        listnet_loss +
        0.25f0 * old_q_loss +
        0.15f0 * margin_loss +
        0.10f0 * death_loss +
        0.05f0 * quantile_loss +
        0.10f0 * geometry_loss +
        structure_loss
    listnet_kl = max(listnet_loss - teacher_entropy, 0.0f0)
    return LossRecord(
        composite_loss,
        listnet_loss,
        teacher_entropy,
        listnet_kl,
        old_q_loss,
        q_huber_loss,
        margin_loss,
        raw_top_gap_loss,
        death_loss,
        quantile_loss,
        geometry_loss,
        line_clear_loss,
        max_height_loss,
        holes_loss,
        cavities_loss,
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
    base_route::Vector{Float32}
    route_eligibility::Vector{Float32}
    route_standardized::Vector{Float32}
    route_logweight::Vector{Float32}
    route_alpha::Vector{Float32}
    route_key::Vector{Float32}
    route_order::Vector{Int16}
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
    head_feature_dim = hasproperty(model, :head_readout) &&
        model.head_readout === :ordered_topk ?
        model.node_dim * (model.workspace_k + 1) :
        hasproperty(model, :head_readout) &&
        model.head_readout === :anchored_temporal ?
        3 * model.node_dim : 2 * model.node_dim
    return CandidateScratch(
        PackScratch(),
        zeros(Float32, model.blocks),
        falses(model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Int16, model.workspace_k),
        zeros(Float32, nodes),
        zeros(Float32, nodes),
        zeros(Float32, nodes),
        zeros(Float32, nodes),
        zeros(Float32, model.node_dim),
        zeros(Float32, model.node_dim),
        zeros(Float32, model.node_dim),
        zeros(Float32, nodes),
        zeros(Float32, model.blocks),
        zeros(Float32, head_feature_dim),
        zeros(Float32, head_feature_dim),
        zeros(Float32, model.hidden),
        zeros(Float32, model.fanout),
        falses(model.fanout),
    )
end

"""
Configuration for the shadow and updating hybrid eligibility learner.

The edge trace follows the LIF e-prop factorization

    epsilon(t + 1) = leak(post) * epsilon(t) + delayed_pre_spike(t)
    eligibility(t + 1) = surrogate(post, t + 1) * epsilon(t + 1)

The fixed-random and symmetric-head modes are retained as falsification and
teacher controls.  Production `:block_local` mode never reads `head_weight` or
`output_weight` while constructing recurrent learning signals.  Instead, each
block owns a seed-fixed, non-trainable 22-to-node-dimension projection and
forms local Q, death, quantile, and geometry prediction errors from its own
membrane/spike trajectory.  `synapse_learning_mode=:vjp` records a comparison;
`:local_eligibility` passes all enabled recurrent groups to AdamW while the
supervised output head keeps its exact analytic four-parameter-group update.
"""
struct EPropShadowConfig
    trace_decay_scale::Float32
    feedback_seed::UInt64
    feedback_scale::Float32
    feedback_mode::Symbol
    eligibility_mode::Symbol
    error_signal_mode::Symbol
    edge_parameter_mode::Symbol
    node_parameter_mode::Symbol
    routing_parameter_mode::Symbol
    signal_schedule::Symbol
    third_factor_mode::Symbol
    time_order::Symbol
    routing_entropy_weight::Float32
    routing_entropy_floor::Float32
    routing_load_weight::Float32
end

function EPropShadowConfig(;
    trace_decay_scale::Real=1.0f0,
    feedback_seed::Integer=0x4550524f50534844,
    feedback_scale::Real=1.0f0,
    feedback_mode::Symbol=:fixed_random,
    eligibility_mode::Symbol=:spike,
    error_signal_mode::Symbol=:listnet_q,
    edge_parameter_mode::Symbol=:weight_only,
    node_parameter_mode::Symbol=:none,
    routing_parameter_mode::Symbol=:none,
    signal_schedule::Symbol=:terminal,
    third_factor_mode::Symbol=:aligned,
    time_order::Symbol=:forward,
    routing_entropy_weight::Real=0.0f0,
    routing_entropy_floor::Real=0.70f0,
    routing_load_weight::Real=0.0f0,
)
    decay = Float32(trace_decay_scale)
    scale = Float32(feedback_scale)
    isfinite(decay) && 0.0f0 <= decay <= 1.0f0 || throw(ArgumentError(
        "trace_decay_scale must be finite and in [0, 1]",
    ))
    isfinite(scale) && scale >= 0.0f0 || throw(ArgumentError(
        "feedback_scale must be finite and nonnegative",
    ))
    feedback_mode in (:fixed_random, :symmetric_head, :block_local) ||
        throw(ArgumentError(
            "feedback_mode must be fixed_random, symmetric_head, or block_local",
        ))
    eligibility_mode in (:spike, :membrane) ||
        throw(ArgumentError(
            "eligibility_mode must be spike or membrane",
        ))
    error_signal_mode in (:listnet_q, :full_raw) ||
        throw(ArgumentError(
            "error_signal_mode must be listnet_q or full_raw",
        ))
    error_signal_mode === :full_raw &&
       feedback_mode ∉ (:symmetric_head, :block_local) &&
        throw(ArgumentError(
            "full_raw requires feedback_mode=:symmetric_head or :block_local",
        ))
    edge_parameter_mode in (:weight_only, :weight_gate_delay) ||
        throw(ArgumentError(
            "edge_parameter_mode must be weight_only or weight_gate_delay",
        ))
    node_parameter_mode in (:none, :lif_feedback, :full_state) ||
        throw(ArgumentError(
            "node_parameter_mode must be none, lif_feedback, or full_state",
        ))
    routing_parameter_mode in (:none, :three_factor, :local_soft) ||
        throw(ArgumentError(
            "routing_parameter_mode must be none, three_factor, or local_soft",
        ))
    signal_schedule in (:terminal, :all_cycles) ||
        throw(ArgumentError(
            "signal_schedule must be terminal or all_cycles",
        ))
    third_factor_mode in (:aligned, :zero, :candidate_shuffle) ||
        throw(ArgumentError(
            "third_factor_mode must be aligned, zero, or candidate_shuffle",
        ))
    time_order in (:forward, :reverse) || throw(ArgumentError(
        "time_order must be forward or reverse",
    ))
    entropy_weight = Float32(routing_entropy_weight)
    entropy_floor = Float32(routing_entropy_floor)
    load_weight = Float32(routing_load_weight)
    isfinite(entropy_weight) && entropy_weight >= 0.0f0 ||
        throw(ArgumentError(
            "routing_entropy_weight must be finite and nonnegative",
        ))
    isfinite(entropy_floor) && 0.0f0 <= entropy_floor <= 1.0f0 ||
        throw(ArgumentError(
            "routing_entropy_floor must be in [0, 1]",
        ))
    isfinite(load_weight) && load_weight >= 0.0f0 ||
        throw(ArgumentError(
            "routing_load_weight must be finite and nonnegative",
        ))
    return EPropShadowConfig(
        decay,
        UInt64(feedback_seed),
        scale,
        feedback_mode,
        eligibility_mode,
        error_signal_mode,
        edge_parameter_mode,
        node_parameter_mode,
        routing_parameter_mode,
        signal_schedule,
        third_factor_mode,
        time_order,
        entropy_weight,
        entropy_floor,
        load_weight,
    )
end

struct EPropParameterReport
    enabled::Bool
    local_gradient_norm::Float64
    full_vjp_gradient_norm::Float64
    dot_with_full_vjp::Float64
    cosine_with_full_vjp::Float64
    local_nonzero_fraction::Float64
    full_vjp_nonzero_fraction::Float64
end

function _disabled_eprop_parameter_report()
    return EPropParameterReport(
        false,
        0.0,
        NaN,
        0.0,
        NaN,
        0.0,
        NaN,
    )
end

struct EPropShadowReport
    feedback_mode::Symbol
    eligibility_mode::Symbol
    third_factor_mode::Symbol
    time_order::Symbol
    reference_vjp_available::Bool
    used_for_update::Bool
    candidates::Int
    wall_seconds::Float64
    local_gradient_norm::Float64
    full_vjp_gradient_norm::Float64
    dot_with_full_vjp::Float64
    cosine_with_full_vjp::Float64
    local_nonzero_fraction::Float64
    full_vjp_nonzero_fraction::Float64
    worker_trace_bytes::Int
    worker_gradient_bytes::Int
    worker_signal_bytes::Int
    fixed_feedback_bytes::Int
    local_q_loss::Float64
    local_death_loss::Float64
    local_quantile_loss::Float64
    local_geometry_loss::Float64
    parameter_reports::NamedTuple
end

function _empty_eprop_shadow_report(config::EPropShadowConfig)
    return EPropShadowReport(
        config.feedback_mode,
        config.eligibility_mode,
        config.third_factor_mode,
        config.time_order,
        false,
        false,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        NaN,
        0.0,
        0.0,
        0,
        0,
        0,
        0,
        NaN,
        NaN,
        NaN,
        NaN,
        (;
            synapse_weight=_disabled_eprop_parameter_report(),
            input_gain=_disabled_eprop_parameter_report(),
            input_bias=_disabled_eprop_parameter_report(),
            gate_logits=_disabled_eprop_parameter_report(),
            delay_logits=_disabled_eprop_parameter_report(),
            leak_logits=_disabled_eprop_parameter_report(),
            threshold_logits=_disabled_eprop_parameter_report(),
            feedback_gain=_disabled_eprop_parameter_report(),
            workspace_key=_disabled_eprop_parameter_report(),
            query_weight=_disabled_eprop_parameter_report(),
            workspace_decay_logit=_disabled_eprop_parameter_report(),
        ),
    )
end

mutable struct EPropWorkerShadow
    gradient::Matrix{Float32}
    input_gain_gradient::Union{Nothing,Vector{Float32}}
    input_bias_gradient::Union{Nothing,Vector{Float32}}
    gate_gradient::Union{Nothing,Matrix{Float32}}
    delay_gradient::Union{Nothing,Matrix{Float32}}
    utility_evidence::Union{Nothing,Matrix{Float64}}
    leak_gradient::Union{Nothing,Vector{Float32}}
    threshold_gradient::Union{Nothing,Vector{Float32}}
    feedback_gradient::Union{Nothing,Matrix{Float32}}
    workspace_key_gradient::Union{Nothing,Matrix{Float32}}
    query_weight_gradient::Union{Nothing,Matrix{Float32}}
    workspace_decay_gradient::Union{Nothing,Vector{Float32}}
    post_sensitivity::Matrix{Float32}
    state_recurrence::Matrix{Float32}
    node_signal::Matrix{Float32}
    hidden_signal::Vector{Float32}
    local_prediction::Vector{Float32}
    local_error::Vector{Float32}
    route_query_signal::Vector{Float32}
    route_workspace_signal::Vector{Float32}
    route_block_signal::Vector{Float32}
    route_block_soft::Vector{Float32}
    route_state_signal::Matrix{Float32}
    route_score_signal::Matrix{Float32}
    route_write_signal::Matrix{Float32}
    eligibility_square_sum::Float64
    eligibility_count::Int64
    membrane_margin_sum::Float64
    membrane_margin_square_sum::Float64
    surrogate_sensitivity_sum::Float64
    surrogate_sensitivity_square_sum::Float64
    membrane_sample_count::Int64
    local_q_loss_sum::Float64
    local_q_loss_count::Int64
    local_death_loss_sum::Float64
    local_death_loss_count::Int64
    local_quantile_loss_sum::Float64
    local_quantile_loss_count::Int64
    local_geometry_loss_sum::Float64
    local_geometry_loss_count::Int64
end

function EPropWorkerShadow(
    model,
    parameters,
    config::EPropShadowConfig;
    gradient_storage::Bool=true,
)
    edge_all = config.edge_parameter_mode === :weight_gate_delay
    node_all = config.node_parameter_mode !== :none
    full_state = config.node_parameter_mode === :full_state
    routing_all = config.routing_parameter_mode !== :none
    nodes = size(parameters.synapse_weight, 1)
    return EPropWorkerShadow(
        gradient_storage ?
            zeros(Float32, size(parameters.synapse_weight)) :
            zeros(Float32, 0, 0),
        full_state ?
            (
                gradient_storage ?
                zeros(Float32, size(parameters.input_gain)) :
                zeros(Float32, 0)
            ) : nothing,
        full_state ?
            (
                gradient_storage ?
                zeros(Float32, size(parameters.input_bias)) :
                zeros(Float32, 0)
            ) : nothing,
        edge_all ?
            (
                gradient_storage ?
                zeros(Float32, size(parameters.gate_logits)) :
                zeros(Float32, 0, 0)
            ) : nothing,
        edge_all ?
            (
                gradient_storage ?
                zeros(Float32, size(parameters.delay_logits)) :
                zeros(Float32, 0, 0)
            ) : nothing,
        edge_all && gradient_storage ?
            zeros(Float64, size(parameters.gate_logits)) :
            nothing,
        node_all ?
            (
                gradient_storage ?
                zeros(Float32, size(parameters.leak_logits)) :
                zeros(Float32, 0)
            ) : nothing,
        node_all ?
            (
                gradient_storage ?
                zeros(Float32, size(parameters.threshold_logits)) :
                zeros(Float32, 0)
            ) : nothing,
        node_all ?
            (
                gradient_storage ?
                zeros(Float32, size(parameters.feedback_gain)) :
                zeros(Float32, 0, 0)
            ) : nothing,
        routing_all ?
            (
                gradient_storage ?
                zeros(Float32, size(parameters.workspace_key)) :
                zeros(Float32, 0, 0)
            ) : nothing,
        routing_all ?
            (
                gradient_storage ?
                zeros(Float32, size(parameters.query_weight)) :
                zeros(Float32, 0, 0)
            ) : nothing,
        full_state ?
            (
                gradient_storage ?
                zeros(Float32, size(parameters.workspace_decay_logit)) :
                zeros(Float32, 0)
            ) :
            nothing,
        zeros(Float32, nodes, model.cycles),
        zeros(Float32, nodes, model.cycles),
        zeros(Float32, nodes, model.cycles),
        zeros(Float32, length(parameters.head_bias)),
        zeros(Float32, OUTPUT_DIM),
        zeros(Float32, OUTPUT_DIM),
        zeros(Float32, model.node_dim),
        zeros(Float32, model.node_dim),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, nodes, model.cycles + 1),
        zeros(Float32, model.blocks, model.cycles),
        zeros(Float32, model.node_dim, model.cycles),
        0.0,
        Int64(0),
        0.0,
        0.0,
        0.0,
        0.0,
        Int64(0),
        0.0,
        Int64(0),
        0.0,
        Int64(0),
        0.0,
        Int64(0),
        0.0,
        Int64(0),
    )
end

mutable struct EPropShadowState
    config::EPropShadowConfig
    q_feedback::Vector{Float32}
    block_projection::Array{Float32,3}
    block_learning_signal::Array{Float32,3}
    block_signal_inv_rms::Array{Float32,3}
    block_supervised_reward::Array{Float32,3}
    block_advantage::Array{Float32,3}
    signal_flat::Vector{Int32}
    trajectory_gate_hard::Matrix{UInt8}
    gradient::Matrix{Float32}
    input_gain_gradient::Union{Nothing,Vector{Float32}}
    input_bias_gradient::Union{Nothing,Vector{Float32}}
    gate_gradient::Union{Nothing,Matrix{Float32}}
    delay_gradient::Union{Nothing,Matrix{Float32}}
    leak_gradient::Union{Nothing,Vector{Float32}}
    threshold_gradient::Union{Nothing,Vector{Float32}}
    feedback_gradient::Union{Nothing,Matrix{Float32}}
    workspace_key_gradient::Union{Nothing,Matrix{Float32}}
    query_weight_gradient::Union{Nothing,Matrix{Float32}}
    workspace_decay_gradient::Union{Nothing,Vector{Float32}}
    route_load::Matrix{Float32}
    last_report::EPropShadowReport
end

function _fixed_q_feedback(model, config::EPropShadowConfig)
    nodes = model.blocks * model.node_dim
    rng = Xoshiro(config.feedback_seed)
    block_normalizer = inv(sqrt(Float32(model.node_dim)))
    feedback = Vector{Float32}(undef, nodes)
    @inbounds for node in 1:nodes
        sign = rand(rng, Bool) ? 1.0f0 : -1.0f0
        feedback[node] =
            config.feedback_scale * block_normalizer * sign
    end
    return feedback
end

function _fixed_block_projection(model, config::EPropShadowConfig)
    rng = Xoshiro(config.feedback_seed ⊻ UInt64(0x424c4f434b4c4f43))
    normalizer =
        config.feedback_scale / sqrt(Float32(model.node_dim))
    projection = Array{Float32}(
        undef,
        model.node_dim,
        OUTPUT_DIM,
        model.blocks,
    )
    @inbounds for block in 1:model.blocks
        for output in 1:OUTPUT_DIM
            for coordinate in 1:model.node_dim
                projection[coordinate, output, block] =
                    rand(rng, Bool) ? normalizer : -normalizer
            end
        end
    end
    return projection
end

function EPropShadowState(model, parameters, capacity::Int, config::EPropShadowConfig)
    edge_all = config.edge_parameter_mode === :weight_gate_delay
    node_all = config.node_parameter_mode !== :none
    full_state = config.node_parameter_mode === :full_state
    routing_all = config.routing_parameter_mode !== :none
    return EPropShadowState(
        config,
        _fixed_q_feedback(model, config),
        _fixed_block_projection(model, config),
        zeros(
            Float32,
            model.blocks * model.node_dim,
            model.cycles,
            capacity,
        ),
        zeros(Float32, model.blocks, model.cycles, capacity),
        zeros(Float32, model.blocks, model.cycles, capacity),
        zeros(Float32, model.blocks, model.cycles, capacity),
        zeros(Int32, capacity),
        zeros(UInt8, size(parameters.gate_logits)),
        zeros(Float32, size(parameters.synapse_weight)),
        full_state ? zeros(Float32, size(parameters.input_gain)) : nothing,
        full_state ? zeros(Float32, size(parameters.input_bias)) : nothing,
        edge_all ? zeros(Float32, size(parameters.gate_logits)) : nothing,
        edge_all ? zeros(Float32, size(parameters.delay_logits)) : nothing,
        node_all ? zeros(Float32, size(parameters.leak_logits)) : nothing,
        node_all ? zeros(Float32, size(parameters.threshold_logits)) : nothing,
        node_all ? zeros(Float32, size(parameters.feedback_gain)) : nothing,
        routing_all ? zeros(Float32, size(parameters.workspace_key)) : nothing,
        routing_all ? zeros(Float32, size(parameters.query_weight)) : nothing,
        full_state ?
            zeros(Float32, size(parameters.workspace_decay_logit)) :
            nothing,
        zeros(Float32, model.blocks, model.cycles),
        _empty_eprop_shadow_report(config),
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

@inline function _record_workspace_selection!(
    scratch::CandidateScratch,
    arena::TrainingArena,
    model,
    cycle::Int,
    flat::Int,
)
    selected_cutoff = Inf32
    best_unselected = -Inf32
    score_square_sum = 0.0
    entropy = 0.0f0
    mask_hash = UInt64(0xcbf29ce484222325)
    cycle_churn = 0
    @inbounds for block in 1:model.blocks
        selected = scratch.selected[block]
        score = scratch.scores[block]
        base_probability = scratch.base_route[block]
        arena.block_mask[block, cycle, flat] =
            selected ? 1.0f0 : 0.0f0
        arena.route_policy_probability[block, cycle, flat] =
            scratch.soft_route[block]
        arena.route_base_probability[block, cycle, flat] =
            base_probability
        arena.route_score[block, cycle, flat] = score
        arena.route_eligibility[block, cycle, flat] =
            scratch.route_eligibility[block]
        score_square_sum = muladd(
            Float64(score),
            Float64(score),
            score_square_sum,
        )
        entropy -=
            base_probability * log(max(base_probability, 1.0f-12))
        if selected
            selected_cutoff = min(selected_cutoff, score)
            mask_hash = xor(mask_hash, UInt64(block))
            mask_hash *= UInt64(0x100000001b3)
        else
            best_unselected = max(best_unselected, score)
        end
        if cycle > 1
            previous_selected =
                arena.block_mask[block, cycle - 1, flat] != 0.0f0
            cycle_churn += selected != previous_selected
        end
    end
    @inbounds for rank in 1:model.workspace_k
        arena.route_order[rank, cycle, flat] =
            scratch.route_order[rank]
    end
    arena.route_selection_gap_value[cycle, flat] =
        model.workspace_k == model.blocks ?
        0.0f0 : selected_cutoff - best_unselected
    arena.route_score_square_sum[cycle, flat] = score_square_sum
    arena.route_normalized_entropy[cycle, flat] =
        model.blocks == 1 ? 1.0f0 :
        entropy / log(Float32(model.blocks))
    arena.route_mask_fingerprint[cycle, flat] = mask_hash
    arena.route_cycle_churn_count[cycle, flat] = Int16(cycle_churn)
    return nothing
end

@inline function _select_workspace!(
    scratch::CandidateScratch,
    arena::TrainingArena,
    model,
    cycle::Int,
    flat::Int,
)
    Routing.prepare_policy!(
        scratch.route_standardized,
        scratch.base_route,
        scratch.soft_route,
        scratch.scores;
        temperature=model.route_temperature,
    )
    Routing.deterministic_topk!(
        scratch.selected,
        scratch.route_order,
        scratch.scores,
        model.workspace_k,
    )
    Routing.ordered_score_eligibility!(
        scratch.route_eligibility,
        scratch.route_logweight,
        scratch.route_alpha,
        scratch.route_standardized,
        scratch.base_route,
        scratch.soft_route,
        scratch.scores,
        scratch.route_order,
        model.workspace_k;
        temperature=model.route_temperature,
    )
    _record_workspace_selection!(scratch, arena, model, cycle, flat)
    return nothing
end

@inline _routing_mix64(value::UInt64) =
    Routing.routing_mix64(value)

@inline function _select_workspace_stochastic!(
    scratch::CandidateScratch,
    arena::TrainingArena,
    model,
    cycle::Int,
    flat::Int,
    nonce::UInt64,
)
    Routing.prepare_policy!(
        scratch.route_standardized,
        scratch.base_route,
        scratch.soft_route,
        scratch.scores;
        temperature=model.route_temperature,
    )
    Routing.sample_plackett_luce_topk!(
        scratch.selected,
        scratch.route_order,
        scratch.route_key,
        scratch.soft_route,
        model.workspace_k,
        nonce,
        cycle,
    )
    Routing.ordered_score_eligibility!(
        scratch.route_eligibility,
        scratch.route_logweight,
        scratch.route_alpha,
        scratch.route_standardized,
        scratch.base_route,
        scratch.soft_route,
        scratch.scores,
        scratch.route_order,
        model.workspace_k;
        temperature=model.route_temperature,
    )
    _record_workspace_selection!(scratch, arena, model, cycle, flat)
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
    routing_nonce::UInt64=UInt64(0),
)
    nodes = model.blocks * model.node_dim
    node_dim = model.node_dim
    @inbounds for node in 1:nodes
        rail = arena.rails[model.feature_for_node[node], flat]
        seed = muladd(parameters.input_gain[node], rail, parameters.input_bias[node])
        arena.membrane[node, 1, flat] = seed
    end
    query_square_sum = 0.0f0
    @inbounds for coordinate in 1:node_dim
        value = 0.0f0
        for rail in 1:Model.INPUT_RAILS
            value = muladd(
                parameters.query_weight[coordinate, rail],
                arena.rails[rail, flat],
                value,
            )
        end
        arena.query_pre[coordinate, flat] = value
        query_square_sum = muladd(value, value, query_square_sum)
        arena.workspace[coordinate, 1, flat] = 0.0f0
    end
    query_inv_rms = inv(sqrt(
        query_square_sum / Float32(node_dim) + Model.RMS_NORM_EPS,
    ))
    arena.query_inv_rms[flat] = query_inv_rms
    @inbounds for coordinate in 1:node_dim
        arena.query[coordinate, flat] = tanh(
            Model.QUERY_NORM_SCALE *
            arena.query_pre[coordinate, flat] *
            query_inv_rms,
        )
    end

    @inbounds for cycle in 1:model.cycles
        _compute_scores!(
            scratch, arena, model, parameters, cycle, flat,
        )
        if routing_nonce == 0
            _select_workspace!(scratch, arena, model, cycle, flat)
        else
            _select_workspace_stochastic!(
                scratch,
                arena,
                model,
                cycle,
                flat,
                routing_nonce,
            )
        end

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

    workspace_square_sum = 0.0f0
    pool_square_sum = 0.0f0
    @inbounds for coordinate in 1:node_dim
        selected_pool = 0.0f0
        for block in 1:model.blocks
            node = coordinate + (block - 1) * node_dim
            selected_pool = muladd(
                arena.membrane[node, model.cycles + 1, flat],
                arena.block_mask[block, model.cycles, flat],
                selected_pool,
            )
        end
        selected_pool /= Float32(model.workspace_k)
        workspace_value =
            arena.workspace[coordinate, model.cycles + 1, flat]
        scratch.features[coordinate] = workspace_value
        scratch.features[node_dim + coordinate] =
            selected_pool
        workspace_square_sum = muladd(
            workspace_value,
            workspace_value,
            workspace_square_sum,
        )
        pool_square_sum = muladd(
            selected_pool,
            selected_pool,
            pool_square_sum,
        )
    end
    workspace_inv_rms = inv(sqrt(
        workspace_square_sum / Float32(node_dim) +
        Model.RMS_NORM_EPS,
    ))
    pool_inv_rms = inv(sqrt(
        pool_square_sum / Float32(node_dim) +
        Model.RMS_NORM_EPS,
    ))
    arena.workspace_inv_rms[flat] = workspace_inv_rms
    arena.selected_pool_inv_rms[flat] = pool_inv_rms
    @inbounds for coordinate in 1:node_dim
        scratch.features[coordinate] *= workspace_inv_rms
        scratch.features[node_dim + coordinate] *= pool_inv_rms
    end
    hidden_square_sum = 0.0f0
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
        end
        arena.hidden_pre[hidden, flat] = activation
        hidden_square_sum = muladd(
            activation,
            activation,
            hidden_square_sum,
        )
    end
    hidden_inv_rms = inv(sqrt(
        hidden_square_sum / Float32(model.hidden) +
        Model.RMS_NORM_EPS,
    ))
    arena.hidden_inv_rms[flat] = hidden_inv_rms
    @inbounds for hidden in 1:model.hidden
        arena.hidden[hidden, flat] = tanh(
            Model.HIDDEN_NORM_SCALE *
            arena.hidden_pre[hidden, flat] *
            hidden_inv_rms,
        )
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

@inline function _eprop_surrogate(
    membrane::Float32,
    threshold::Float32,
    temperature::Float32,
)
    soft = sigmoid((membrane - threshold) / temperature)
    return soft * (1.0f0 - soft) / temperature
end

const LOCAL_PREDICTOR_SPIKE_SCALE = 0.25f0
const LOCAL_SIGNAL_RMS_EPSILON = 1.0f-4

@inline function _local_predictor_state(
    arena::TrainingArena,
    node::Int,
    cycle::Int,
    flat::Int,
)
    return tanh(arena.membrane[node, cycle, flat]) +
        LOCAL_PREDICTOR_SPIKE_SCALE *
        arena.active_spikes[node, cycle, flat]
end

@inline function _local_predictor_state_derivative(
    arena::TrainingArena,
    cache::ParameterCache,
    model,
    node::Int,
    cycle::Int,
    flat::Int,
)
    membrane = arena.membrane[node, cycle, flat]
    bounded = tanh(membrane)
    return (
        1.0f0 - bounded * bounded +
        LOCAL_PREDICTOR_SPIKE_SCALE *
        _eprop_surrogate(
            membrane,
            cache.threshold[node],
            model.spike_temperature,
        )
    )
end

@inline function _prepare_local_predictor_error!(
    worker_shadow::EPropWorkerShadow,
    arena::TrainingArena,
    signal_flat::Int,
)
    fill!(worker_shadow.local_error, 0.0f0)
    state_slot = div(signal_flat - 1, arena.width) + 1
    candidate = signal_flat - (state_slot - 1) * arena.width
    targets = arena.targets
    prediction = worker_shadow.local_prediction
    error = worker_shadow.local_error
    inverse_valid = inv(Float32(max(arena.valid_count, 1)))

    q_error =
        prediction[1] - targets.teacher_q[candidate, state_slot]
    error[1] =
        0.25f0 * _huber_derivative(q_error) * inverse_valid
    worker_shadow.local_q_loss_sum += Float64(_huber(q_error))
    worker_shadow.local_q_loss_count += Int64(1)

    if targets.death_mask[candidate, state_slot] != 0.0f0
        label = targets.death[candidate, state_slot]
        logit = prediction[2]
        error[2] =
            0.10f0 * (sigmoid(logit) - label) * inverse_valid
        worker_shadow.local_death_loss_sum += Float64(
            max(logit, 0.0f0) -
            logit * label +
            log1p(exp(-abs(logit))),
        )
        worker_shadow.local_death_loss_count += Int64(1)
    end

    teacher_q = targets.teacher_q[candidate, state_slot]
    @inbounds for quantile in 1:QUANTILES
        output = 2 + quantile
        quantile_error = teacher_q - prediction[output]
        tau =
            (Float32(quantile) - 0.5f0) / Float32(QUANTILES)
        negative = quantile_error < 0.0f0 ? 1.0f0 : 0.0f0
        weight = abs(tau - negative)
        error[output] =
            -0.05f0 *
            weight *
            _huber_derivative(quantile_error) *
            inverse_valid /
            Float32(QUANTILES)
        worker_shadow.local_quantile_loss_sum +=
            Float64(weight * _huber(quantile_error))
        worker_shadow.local_quantile_loss_count += Int64(1)
    end

    geometry_targets = (
        targets.line_clear[candidate, state_slot] / 4.0f0,
        targets.max_height[candidate, state_slot] / 24.0f0,
        targets.holes[candidate, state_slot] / 240.0f0,
        targets.cavities[candidate, state_slot] / 240.0f0,
    )
    @inbounds for local_index in 1:4
        output = 18 + local_index
        geometry_error =
            prediction[output] - geometry_targets[local_index]
        error[output] =
            0.025f0 *
            _huber_derivative(geometry_error) *
            inverse_valid
        worker_shadow.local_geometry_loss_sum +=
            Float64(_huber(geometry_error))
        worker_shadow.local_geometry_loss_count += Int64(1)
    end
    return nothing
end

"""
Generate the production DECOLLE-style signal for one candidate.

This function deliberately has no parameter-tree argument: recurrent credit
assignment cannot observe `head_weight` or `output_weight`.  The fixed
`block_projection` is used both as a local readout and as the seed-fixed
22-to-node-dimension projection of the already-computed supervised raw
cotangent.  Every written cell belongs exclusively to `flat`.
"""
@inline function _prepare_block_learning_signal_candidate!(
    worker_shadow::EPropWorkerShadow,
    shadow::EPropShadowState,
    arena::TrainingArena,
    model,
    cache::ParameterCache,
    flat::Int,
)
    if shadow.config.third_factor_mode === :zero
        @inbounds for cycle in 1:model.cycles
            for block in 1:model.blocks
                shadow.block_signal_inv_rms[block, cycle, flat] = 0.0f0
                shadow.block_supervised_reward[block, cycle, flat] = 0.0f0
                offset = (block - 1) * model.node_dim
                for coordinate in 1:model.node_dim
                    shadow.block_learning_signal[
                        offset + coordinate,
                        cycle,
                        flat,
                    ] = 0.0f0
                end
            end
        end
        return nothing
    end

    projection = shadow.block_projection
    inverse_cycles = inv(Float32(model.cycles))
    inverse_node_dim = inv(Float32(model.node_dim))
    @inbounds for cycle in 1:model.cycles
        for block in 1:model.blocks
            fill!(worker_shadow.local_prediction, 0.0f0)
            offset = (block - 1) * model.node_dim
            for output in 1:OUTPUT_DIM
                value = 0.0f0
                for coordinate in 1:model.node_dim
                    value = muladd(
                        projection[coordinate, output, block],
                        _local_predictor_state(
                            arena,
                            offset + coordinate,
                            cycle,
                            flat,
                        ),
                        value,
                    )
                end
                worker_shadow.local_prediction[output] = value
            end
            _prepare_local_predictor_error!(
                worker_shadow,
                arena,
                flat,
            )

            signal_square_sum = 0.0f0
            reward_projection = 0.0f0
            for coordinate in 1:model.node_dim
                global_projection = 0.0f0
                local_projection = 0.0f0
                for output in 1:OUTPUT_DIM
                    block_value =
                        projection[coordinate, output, block]
                    global_projection = muladd(
                        block_value,
                        arena.raw_gradient[output, flat],
                        global_projection,
                    )
                    local_projection = muladd(
                        block_value,
                        worker_shadow.local_error[output],
                        local_projection,
                    )
                end
                node = offset + coordinate
                signal = inverse_cycles * (
                    global_projection +
                    _local_predictor_state_derivative(
                        arena,
                        cache,
                        model,
                        node,
                        cycle,
                        flat,
                    ) * local_projection
                )
                shadow.block_learning_signal[
                    node,
                    cycle,
                    flat,
                ] = signal
                signal_square_sum =
                    muladd(signal, signal, signal_square_sum)
                reward_projection = muladd(
                    signal,
                    _local_predictor_state(
                        arena,
                        node,
                        cycle,
                        flat,
                    ),
                    reward_projection,
                )
            end
            signal_rms = sqrt(signal_square_sum * inverse_node_dim)
            shadow.block_signal_inv_rms[block, cycle, flat] =
                inv(max(signal_rms, LOCAL_SIGNAL_RMS_EPSILON))
            # This is not an environment return.  It is a supervised reward
            # surrogate: negative local loss-change proxy, centered across
            # candidates below before the policy-gradient sign conversion.
            shadow.block_supervised_reward[block, cycle, flat] =
                -reward_projection * inverse_node_dim
        end
    end
    return nothing
end

function _center_block_supervised_rewards!(executor)
    shadow = executor.eprop_shadow
    shadow === nothing && return nothing
    shadow.config.feedback_mode === :block_local || return nothing
    arena = executor.trainer.arena
    model = executor.trainer.model
    if shadow.config.third_factor_mode === :zero
        fill!(shadow.block_advantage, 0.0f0)
        return nothing
    end
    @inbounds for state_slot in 1:arena.state_batch
        count = Int(arena.counts[state_slot])
        offset = (state_slot - 1) * arena.width
        inverse_count = inv(Float32(count))
        for cycle in 1:model.cycles
            for block in 1:model.blocks
                reward_mean = 0.0f0
                for candidate in 1:count
                    reward_mean += shadow.block_supervised_reward[
                        block,
                        cycle,
                        offset + candidate,
                    ]
                end
                reward_mean *= inverse_count
                for candidate in 1:count
                    flat = offset + candidate
                    shadow.block_advantage[block, cycle, flat] =
                        shadow.block_supervised_reward[
                            block,
                            cycle,
                            flat,
                        ] - reward_mean
                end
            end
        end
    end
    return nothing
end

@inline function _replay_signal_flat(
    shadow::EPropShadowState,
    flat::Int,
)
    signal_flat = Int(@inbounds shadow.signal_flat[flat])
    1 <= signal_flat <= length(shadow.signal_flat) || error(
        "e-prop signal map was not prepared",
    )
    return signal_flat
end

@inline function _node_learning_signal(
    worker_shadow::EPropWorkerShadow,
    shadow::EPropShadowState,
    flat::Int,
    node::Int,
    cycle::Int,
)
    if shadow.config.feedback_mode === :block_local
        return @inbounds shadow.block_learning_signal[
            node,
            cycle,
            _replay_signal_flat(shadow, flat),
        ]
    end
    return @inbounds worker_shadow.node_signal[node, cycle]
end

@inline function _routing_block_advantage(
    shadow::EPropShadowState,
    flat::Int,
    block::Int,
    cycle::Int,
)
    shadow.config.feedback_mode === :block_local || return 0.0f0
    return @inbounds shadow.block_advantage[
        block,
        cycle,
        _replay_signal_flat(shadow, flat),
    ]
end

@inline function _prepare_eprop_cycle_sensitivity!(
    worker_shadow::EPropWorkerShadow,
    accumulator::EPropWorkerShadow,
    shadow::EPropShadowState,
    arena::TrainingArena,
    model,
    cache::ParameterCache,
    flat::Int,
    cycle::Int,
)
    nodes = model.blocks * model.node_dim
    decay_scale = shadow.config.trace_decay_scale
    if shadow.config.eligibility_mode === :spike
        @inbounds for node in 1:nodes
            membrane = arena.membrane[node, cycle, flat]
            threshold = cache.threshold[node]
            margin = Float64(membrane - threshold)
            spike_surrogate = _eprop_surrogate(
                membrane,
                threshold,
                model.spike_temperature,
            )
            accumulator.membrane_margin_sum += margin
            accumulator.membrane_margin_square_sum =
                muladd(margin, margin, accumulator.membrane_margin_square_sum)
            accumulator.surrogate_sensitivity_sum +=
                Float64(spike_surrogate)
            accumulator.surrogate_sensitivity_square_sum = muladd(
                Float64(spike_surrogate),
                Float64(spike_surrogate),
                accumulator.surrogate_sensitivity_square_sum,
            )
            accumulator.membrane_sample_count += Int64(1)
            worker_shadow.state_recurrence[node, cycle] =
                decay_scale * cache.leak[node]
            worker_shadow.post_sensitivity[node, cycle] = _eprop_surrogate(
                arena.membrane[node, cycle + 1, flat],
                cache.threshold[node],
                model.spike_temperature,
            )
        end
    else
        @inbounds for node in 1:nodes
            membrane = arena.membrane[node, cycle, flat]
            threshold = cache.threshold[node]
            reset_surrogate = _eprop_surrogate(
                membrane,
                threshold,
                model.spike_temperature,
            )
            margin = Float64(membrane - threshold)
            accumulator.membrane_margin_sum += margin
            accumulator.membrane_margin_square_sum =
                muladd(margin, margin, accumulator.membrane_margin_square_sum)
            accumulator.surrogate_sensitivity_sum +=
                Float64(reset_surrogate)
            accumulator.surrogate_sensitivity_square_sum = muladd(
                Float64(reset_surrogate),
                Float64(reset_surrogate),
                accumulator.surrogate_sensitivity_square_sum,
            )
            accumulator.membrane_sample_count += Int64(1)
            worker_shadow.state_recurrence[node, cycle] =
                decay_scale * (
                    cache.leak[node] -
                    cache.threshold[node] * reset_surrogate
                )
            worker_shadow.post_sensitivity[node, cycle] = 1.0f0
        end
    end
    return nothing
end

@inline function _prepare_eprop_node_signal!(
    worker_shadow::EPropWorkerShadow,
    shadow::EPropShadowState,
    arena::TrainingArena,
    model,
    parameters,
    flat::Int,
    signal_flat::Int,
    listnet_delta_q::Float32,
    zero_signal::Bool,
)
    nodes = model.blocks * model.node_dim
    fill!(worker_shadow.node_signal, 0.0f0)
    shadow.config.feedback_mode === :block_local && error(
        "block-local recurrent signals must use the head-independent signal phase",
    )
    if shadow.config.feedback_mode === :fixed_random
        zero_signal && return nothing
        @inbounds for cycle in 1:model.cycles
            for node in 1:nodes
                worker_shadow.node_signal[node, cycle] =
                    listnet_delta_q * shadow.q_feedback[node]
            end
        end
        return nothing
    end

    hidden_projection_mean = 0.0f0
    @inbounds for hidden in 1:model.hidden
        hidden_value = arena.hidden[hidden, flat]
        output_cotangent = if zero_signal
            0.0f0
        elseif shadow.config.error_signal_mode === :full_raw
            value = 0.0f0
            for output in 1:OUTPUT_DIM
                value = muladd(
                    parameters.output_weight[output, hidden],
                    arena.raw_gradient[output, signal_flat],
                    value,
                )
            end
            value
        else
            parameters.output_weight[1, hidden] * listnet_delta_q
        end
        worker_shadow.hidden_signal[hidden] =
            shadow.config.feedback_scale *
            output_cotangent *
            Model.HIDDEN_NORM_SCALE *
            (1.0f0 - hidden_value * hidden_value)
        hidden_projection_mean = muladd(
            worker_shadow.hidden_signal[hidden],
            arena.hidden_pre[hidden, flat],
            hidden_projection_mean,
        )
    end
    hidden_projection_mean /= Float32(model.hidden)
    hidden_inv_rms = arena.hidden_inv_rms[flat]
    hidden_inv_rms_squared = hidden_inv_rms * hidden_inv_rms
    @inbounds for hidden in 1:model.hidden
        worker_shadow.hidden_signal[hidden] = hidden_inv_rms * (
            worker_shadow.hidden_signal[hidden] -
            arena.hidden_pre[hidden, flat] *
            hidden_inv_rms_squared *
            hidden_projection_mean
        )
    end
    pool_projection_mean = 0.0f0
    @inbounds for coordinate in 1:model.node_dim
        pooled_cotangent = 0.0f0
        feature = model.node_dim + coordinate
        for hidden in 1:model.hidden
            pooled_cotangent = muladd(
                parameters.head_weight[hidden, feature],
                worker_shadow.hidden_signal[hidden],
                pooled_cotangent,
            )
        end
        worker_shadow.route_query_signal[coordinate] = pooled_cotangent
        selected_pool = 0.0f0
        for block in 1:model.blocks
            node = coordinate + (block - 1) * model.node_dim
            selected_pool = muladd(
                arena.membrane[node, model.cycles + 1, flat],
                arena.block_mask[block, model.cycles, flat],
                selected_pool,
            )
        end
        selected_pool /= Float32(model.workspace_k)
        pool_projection_mean = muladd(
            pooled_cotangent,
            selected_pool,
            pool_projection_mean,
        )
    end
    pool_projection_mean /= Float32(model.node_dim)
    pool_inv_rms = arena.selected_pool_inv_rms[flat]
    pool_inv_rms_squared = pool_inv_rms * pool_inv_rms
    inverse_k = inv(Float32(model.workspace_k))
    @inbounds for coordinate in 1:model.node_dim
        selected_pool = 0.0f0
        for block in 1:model.blocks
            node = coordinate + (block - 1) * model.node_dim
            selected_pool = muladd(
                arena.membrane[node, model.cycles + 1, flat],
                arena.block_mask[block, model.cycles, flat],
                selected_pool,
            )
        end
        selected_pool *= inverse_k
        membrane_cotangent = pool_inv_rms * (
            worker_shadow.route_query_signal[coordinate] -
            selected_pool *
            pool_inv_rms_squared *
            pool_projection_mean
        ) * inverse_k
        for block in 1:model.blocks
            node = coordinate + (block - 1) * model.node_dim
            terminal_signal =
                membrane_cotangent *
                arena.block_mask[block, model.cycles, flat]
            if shadow.config.signal_schedule === :all_cycles
                for cycle in 1:model.cycles
                    worker_shadow.node_signal[node, cycle] =
                        terminal_signal
                end
            else
                worker_shadow.node_signal[node, model.cycles] =
                    terminal_signal
            end
        end
    end
    return nothing
end

@inline function _accumulate_candidate_edge_utility!(
    utility_evidence::Matrix{Float64},
    active::Bool,
    source::Int,
    relation::Int,
    weight_increment::Float32,
    gate_increment::Float32,
    delay_increment::Float32,
    normalization::Float32=1.0f0,
)
    isfinite(normalization) && normalization >= 0.0f0 || error(
        "utility normalization must be finite and nonnegative",
    )
    responsibility = Float64(normalization) * if active
        abs(Float64(weight_increment)) +
        abs(Float64(gate_increment)) +
        abs(Float64(delay_increment))
    else
        max(-Float64(gate_increment), 0.0)
    end
    utility_evidence[source, relation] += responsibility
    return nothing
end

@inline function _shadow_eprop_edges!(
    worker_shadow::EPropWorkerShadow,
    accumulator::EPropWorkerShadow,
    shadow::EPropShadowState,
    arena::TrainingArena,
    model,
    parameters,
    cache::ParameterCache,
    flat::Int,
)
    nodes = model.blocks * model.node_dim
    @inbounds for cycle in 1:model.cycles
        _prepare_eprop_cycle_sensitivity!(
            worker_shadow,
            accumulator,
            shadow,
            arena,
            model,
            cache,
            flat,
            cycle,
        )
    end
    cycle_range = shadow.config.time_order === :forward ?
        (1:model.cycles) : (model.cycles:-1:1)
    edge_all =
        shadow.config.edge_parameter_mode === :weight_gate_delay
    @inbounds for relation in 1:model.fanout
        for source in 1:nodes
            destination =
                model.destination_for_source[source, relation]
            weight_trace = 0.0f0
            gate_trace = 0.0f0
            delay_trace = 0.0f0
            for cycle in cycle_range
                current =
                    arena.active_spikes[source, cycle, flat]
                previous = cycle == 1 ? 0.0f0 :
                    arena.active_spikes[source, cycle - 1, flat]
                delay = cache.delay[source, relation]
                delayed_pre = muladd(
                    1.0f0 - delay,
                    current,
                    delay * previous,
                )
                recurrence =
                    worker_shadow.state_recurrence[destination, cycle]
                weight_trace = muladd(
                    recurrence,
                    weight_trace,
                    cache.gate_hard[source, relation] * delayed_pre,
                )
                if edge_all
                    weight =
                        parameters.synapse_weight[source, relation]
                    gate_trace = muladd(
                        recurrence,
                        gate_trace,
                        weight *
                        delayed_pre *
                        cache.gate_derivative[source, relation],
                    )
                    delay_trace = muladd(
                        recurrence,
                        delay_trace,
                        cache.gate_hard[source, relation] *
                        weight *
                        (previous - current) *
                        cache.delay_derivative[source, relation],
                    )
                end
                post_sensitivity =
                    worker_shadow.post_sensitivity[destination, cycle]
                third_factor = _node_learning_signal(
                    worker_shadow,
                    shadow,
                    flat,
                    destination,
                    cycle,
                )
                weight_eligibility =
                    weight_trace * post_sensitivity
                weight_eligibility64 = Float64(weight_eligibility)
                accumulator.eligibility_square_sum = muladd(
                    weight_eligibility64,
                    weight_eligibility64,
                    accumulator.eligibility_square_sum,
                )
                accumulator.eligibility_count += Int64(1)
                weight_increment =
                    third_factor * weight_eligibility
                accumulator.gradient[source, relation] +=
                    weight_increment
                gate_increment = 0.0f0
                delay_increment = 0.0f0
                if edge_all
                    gate_increment =
                        third_factor * gate_trace * post_sensitivity
                    delay_increment =
                        third_factor * delay_trace * post_sensitivity
                    accumulator.gate_gradient[source, relation] +=
                        gate_increment
                    accumulator.delay_gradient[source, relation] +=
                        delay_increment
                end
                utility_evidence = accumulator.utility_evidence
                if utility_evidence !== nothing
                    destination_block =
                        div(destination - 1, model.node_dim) + 1
                    normalization =
                        shadow.config.feedback_mode === :block_local ?
                        @inbounds(shadow.block_signal_inv_rms[
                            destination_block,
                            cycle,
                            _replay_signal_flat(shadow, flat),
                        ]) : 1.0f0
                    _accumulate_candidate_edge_utility!(
                        utility_evidence,
                        shadow.trajectory_gate_hard[
                            source,
                            relation,
                        ] != 0x00,
                        source,
                        relation,
                        weight_increment,
                        gate_increment,
                        delay_increment,
                        normalization,
                    )
                end
            end
        end
    end
    return nothing
end

@inline function _shadow_eprop_nodes!(
    worker_shadow::EPropWorkerShadow,
    accumulator::EPropWorkerShadow,
    shadow::EPropShadowState,
    arena::TrainingArena,
    model,
    cache::ParameterCache,
    flat::Int,
)
    shadow.config.node_parameter_mode !== :none ||
        return nothing
    cycle_range = shadow.config.time_order === :forward ?
        (1:model.cycles) : (model.cycles:-1:1)
    nodes = model.blocks * model.node_dim
    @inbounds for node in 1:nodes
        coordinate = mod1(node, model.node_dim)
        block = div(node - 1, model.node_dim) + 1
        leak_trace = 0.0f0
        threshold_trace = 0.0f0
        feedback_trace = 0.0f0
        for cycle in cycle_range
            recurrence =
                worker_shadow.state_recurrence[node, cycle]
            membrane = arena.membrane[node, cycle, flat]
            reset_surrogate = _eprop_surrogate(
                membrane,
                cache.threshold[node],
                model.spike_temperature,
            )
            spike =
                membrane >= cache.threshold[node] ? 1.0f0 : 0.0f0
            leak_trace = muladd(
                recurrence,
                leak_trace,
                membrane * cache.leak_derivative[node],
            )
            threshold_trace = muladd(
                recurrence,
                threshold_trace,
                (
                    cache.threshold[node] * reset_surrogate -
                    spike
                ) * cache.threshold_derivative[node],
            )
            feedback_trace = muladd(
                recurrence,
                feedback_trace,
                arena.workspace[coordinate, cycle + 1, flat],
            )
            post_sensitivity =
                worker_shadow.post_sensitivity[node, cycle]
            third_factor = _node_learning_signal(
                worker_shadow,
                shadow,
                flat,
                node,
                cycle,
            )
            accumulator.leak_gradient[node] = muladd(
                third_factor,
                leak_trace * post_sensitivity,
                accumulator.leak_gradient[node],
            )
            accumulator.threshold_gradient[node] = muladd(
                third_factor,
                threshold_trace * post_sensitivity,
                accumulator.threshold_gradient[node],
            )
            accumulator.feedback_gradient[coordinate, block] = muladd(
                third_factor,
                feedback_trace * post_sensitivity,
                accumulator.feedback_gradient[coordinate, block],
            )
        end
    end
    return nothing
end

@inline function _shadow_eprop_inputs!(
    worker_shadow::EPropWorkerShadow,
    accumulator::EPropWorkerShadow,
    shadow::EPropShadowState,
    arena::TrainingArena,
    model,
    parameters,
    cache::ParameterCache,
    flat::Int,
)
    shadow.config.node_parameter_mode === :full_state ||
        return nothing
    nodes = model.blocks * model.node_dim
    @inbounds for source in 1:nodes
        seed_trace = 1.0f0
        block = div(source - 1, model.node_dim) + 1
        coordinate = mod1(source, model.node_dim)
        route_seed_cotangent = 0.0f0
        local_seed_cotangent = 0.0f0
        for cycle in 1:model.cycles
            membrane = arena.membrane[source, cycle, flat]
            magnitude_derivative =
                membrane > 0.0f0 ? 1.0f0 :
                (membrane < 0.0f0 ? -1.0f0 : 0.0f0)
            score_derivative = muladd(
                parameters.workspace_key[coordinate, block],
                arena.query[coordinate, flat],
                0.05f0 * magnitude_derivative,
            )
            route_seed_cotangent = muladd(
                worker_shadow.route_score_signal[block, cycle] *
                score_derivative,
                seed_trace,
                route_seed_cotangent,
            )
            route_seed_cotangent = muladd(
                worker_shadow.route_write_signal[coordinate, cycle] *
                arena.block_mask[block, cycle, flat],
                seed_trace,
                route_seed_cotangent,
            )
            local_seed_cotangent = muladd(
                _node_learning_signal(
                    worker_shadow,
                    shadow,
                    flat,
                    source,
                    cycle,
                ),
                seed_trace,
                local_seed_cotangent,
            )
            seed_trace = muladd(
                worker_shadow.state_recurrence[source, cycle],
                seed_trace,
                0.18f0,
            )
        end
        seed_cotangent =
            local_seed_cotangent + route_seed_cotangent
        rail = arena.rails[
            model.feature_for_node[source],
            flat,
        ]
        accumulator.input_gain_gradient[source] = muladd(
            seed_cotangent,
            rail,
            accumulator.input_gain_gradient[source],
        )
        accumulator.input_bias_gradient[source] +=
            seed_cotangent
    end
    return nothing
end

@inline function _shadow_workspace_decay!(
    worker_shadow::EPropWorkerShadow,
    accumulator::EPropWorkerShadow,
    shadow::EPropShadowState,
    arena::TrainingArena,
    model,
    parameters,
    cache::ParameterCache,
    flat::Int,
)
    shadow.config.node_parameter_mode === :full_state ||
        return nothing
    fill!(worker_shadow.route_workspace_signal, 0.0f0)
    if shadow.config.feedback_mode === :symmetric_head
        workspace_projection_mean = 0.0f0
        @inbounds for coordinate in 1:model.node_dim
            normalized_signal = 0.0f0
            for hidden in 1:model.hidden
                normalized_signal = muladd(
                    parameters.head_weight[hidden, coordinate],
                    worker_shadow.hidden_signal[hidden],
                    normalized_signal,
                )
            end
            worker_shadow.route_workspace_signal[coordinate] =
                normalized_signal
            workspace_projection_mean = muladd(
                normalized_signal,
                arena.workspace[
                    coordinate,
                    model.cycles + 1,
                    flat,
                ],
                workspace_projection_mean,
            )
        end
        workspace_projection_mean /= Float32(model.node_dim)
        workspace_inv_rms = arena.workspace_inv_rms[flat]
        workspace_inv_rms_squared =
            workspace_inv_rms * workspace_inv_rms
        @inbounds for coordinate in 1:model.node_dim
            workspace_value = arena.workspace[
                coordinate,
                model.cycles + 1,
                flat,
            ]
            worker_shadow.route_workspace_signal[coordinate] =
                workspace_inv_rms * (
                    worker_shadow.route_workspace_signal[coordinate] -
                    workspace_value *
                    workspace_inv_rms_squared *
                    workspace_projection_mean
                )
        end
    end
    @inbounds for coordinate in 1:model.node_dim
        direct_workspace_signal =
            worker_shadow.route_workspace_signal[coordinate]
        if shadow.config.feedback_mode === :block_local
            workspace_trace = 0.0f0
            local_cotangent = 0.0f0
            for cycle in 1:model.cycles
                previous_workspace =
                    arena.workspace[coordinate, cycle, flat]
                next_workspace =
                    arena.workspace[coordinate, cycle + 1, flat]
                workspace_trace =
                    (1.0f0 - next_workspace * next_workspace) *
                    muladd(
                        cache.workspace_decay,
                        workspace_trace,
                        previous_workspace *
                        cache.workspace_decay_derivative,
                    )
                for block in 1:model.blocks
                    node =
                        coordinate +
                        (block - 1) * model.node_dim
                    local_cotangent = muladd(
                        _node_learning_signal(
                            worker_shadow,
                            shadow,
                            flat,
                            node,
                            cycle,
                        ) *
                        parameters.feedback_gain[coordinate, block],
                        workspace_trace,
                        local_cotangent,
                    )
                end
            end
            accumulator.workspace_decay_gradient[1] +=
                local_cotangent
            continue
        end
        fill!(worker_shadow.route_block_signal, 0.0f0)
        workspace_trace = 0.0f0
        for cycle in 1:model.cycles
            previous_workspace =
                arena.workspace[coordinate, cycle, flat]
            next_workspace =
                arena.workspace[coordinate, cycle + 1, flat]
            workspace_trace =
                (1.0f0 - next_workspace * next_workspace) *
                muladd(
                    cache.workspace_decay,
                    workspace_trace,
                    previous_workspace *
                    cache.workspace_decay_derivative,
                )
            for block in 1:model.blocks
                node =
                    coordinate +
                    (block - 1) * model.node_dim
                worker_shadow.route_block_signal[block] = muladd(
                    worker_shadow.state_recurrence[node, cycle],
                    worker_shadow.route_block_signal[block],
                    parameters.feedback_gain[coordinate, block] *
                    workspace_trace,
                )
            end
        end
        local_cotangent =
            direct_workspace_signal * workspace_trace
        for block in 1:model.blocks
            node =
                coordinate +
                (block - 1) * model.node_dim
            local_cotangent = muladd(
                worker_shadow.node_signal[node, model.cycles],
                worker_shadow.route_block_signal[block],
                local_cotangent,
            )
        end
        accumulator.workspace_decay_gradient[1] +=
            local_cotangent
    end
    return nothing
end

@inline function _shadow_routing_candidate!(
    worker_shadow::EPropWorkerShadow,
    accumulator::EPropWorkerShadow,
    shadow::EPropShadowState,
    arena::TrainingArena,
    model,
    parameters,
    cache::ParameterCache,
    flat::Int,
    listnet_delta_q::Float32,
)
    mode = shadow.config.routing_parameter_mode
    fill!(worker_shadow.route_score_signal, 0.0f0)
    fill!(worker_shadow.route_write_signal, 0.0f0)
    mode === :none &&
        return nothing
    shadow.config.third_factor_mode === :zero &&
        return nothing
    fill!(worker_shadow.route_query_signal, 0.0f0)
    fill!(worker_shadow.route_workspace_signal, 0.0f0)
    workspace_projection_mean = 0.0f0
    @inbounds for coordinate in 1:model.node_dim
        if shadow.config.feedback_mode === :symmetric_head
            normalized_workspace_signal = 0.0f0
            for hidden in 1:model.hidden
                normalized_workspace_signal = muladd(
                    parameters.head_weight[hidden, coordinate],
                    worker_shadow.hidden_signal[hidden],
                    normalized_workspace_signal,
                )
            end
            worker_shadow.route_workspace_signal[coordinate] =
                normalized_workspace_signal
            workspace_projection_mean = muladd(
                normalized_workspace_signal,
                arena.workspace[
                    coordinate,
                    model.cycles + 1,
                    flat,
                ],
                workspace_projection_mean,
            )
        end
    end
    if shadow.config.feedback_mode === :symmetric_head
        workspace_projection_mean /= Float32(model.node_dim)
        workspace_inv_rms = arena.workspace_inv_rms[flat]
        workspace_inv_rms_squared =
            workspace_inv_rms * workspace_inv_rms
        @inbounds for coordinate in 1:model.node_dim
            workspace_value = arena.workspace[
                coordinate,
                model.cycles + 1,
                flat,
            ]
            worker_shadow.route_workspace_signal[coordinate] =
                workspace_inv_rms * (
                    worker_shadow.route_workspace_signal[coordinate] -
                    workspace_value *
                    workspace_inv_rms_squared *
                    workspace_projection_mean
                )
        end
    end

    if mode === :three_factor
        @inbounds for cycle in 1:model.cycles
            local_loss_proxy = 0.0f0
            if shadow.config.feedback_mode !== :block_local
                for block in 1:model.blocks
                    arena.block_mask[block, cycle, flat] == 0.0f0 &&
                        continue
                    offset = (block - 1) * model.node_dim
                    for coordinate in 1:model.node_dim
                        node = offset + coordinate
                        local_loss_proxy = muladd(
                            _node_learning_signal(
                                worker_shadow,
                                shadow,
                                flat,
                                node,
                                cycle,
                            ),
                            arena.membrane[node, cycle, flat],
                            local_loss_proxy,
                        )
                    end
                end
                local_loss_proxy /= Float32(model.workspace_k)
            end
            for block in 1:model.blocks
                routing_factor = if shadow.config.feedback_mode ===
                                    :block_local
                    # ordered_route_eligibility is d(log policy)/d(score).
                    # AdamW consumes a loss gradient, hence the minus sign
                    # converts the centered supervised reward surrogate into
                    # the requested +advantage*eligibility parameter update.
                    -_routing_block_advantage(
                        shadow,
                        flat,
                        block,
                        cycle,
                    ) *
                    arena.route_eligibility[block, cycle, flat] +
                    arena.route_regularizer_gradient[
                        block,
                        cycle,
                        flat,
                    ]
                else
                    local_loss_proxy *
                    arena.route_eligibility[block, cycle, flat] +
                    arena.route_regularizer_gradient[
                        block,
                        cycle,
                        flat,
                    ]
                end
                worker_shadow.route_score_signal[block, cycle] =
                    routing_factor
                offset = (block - 1) * model.node_dim
                for coordinate in 1:model.node_dim
                    node = offset + coordinate
                    membrane = arena.membrane[node, cycle, flat]
                    query = arena.query[coordinate, flat]
                    key = parameters.workspace_key[coordinate, block]
                    accumulator.workspace_key_gradient[
                        coordinate,
                        block,
                    ] = muladd(
                        routing_factor * membrane,
                        query,
                        accumulator.workspace_key_gradient[
                            coordinate,
                            block,
                        ],
                    )
                    worker_shadow.route_query_signal[coordinate] = muladd(
                        routing_factor * membrane,
                        key,
                        worker_shadow.route_query_signal[coordinate],
                    )
                end
            end
        end
    else
        @inbounds for node in 1:(model.blocks * model.node_dim)
            worker_shadow.route_state_signal[
                node,
                model.cycles + 1,
            ] = _node_learning_signal(
                worker_shadow,
                shadow,
                flat,
                node,
                model.cycles,
            )
        end
        @inbounds for cycle in model.cycles:-1:1
            for node in 1:(model.blocks * model.node_dim)
                worker_shadow.route_state_signal[node, cycle] =
                    worker_shadow.route_state_signal[node, cycle + 1] *
                    worker_shadow.state_recurrence[node, cycle]
            end
        end
        @inbounds for cycle in model.cycles:-1:1
            for coordinate in 1:model.node_dim
                for block in 1:model.blocks
                    node =
                        coordinate +
                        (block - 1) * model.node_dim
                    worker_shadow.route_workspace_signal[coordinate] =
                        muladd(
                            worker_shadow.route_state_signal[
                                node,
                                cycle + 1,
                            ],
                            parameters.feedback_gain[coordinate, block],
                            worker_shadow.route_workspace_signal[
                                coordinate,
                            ],
                        )
                end
                workspace_value =
                    arena.workspace[coordinate, cycle + 1, flat]
                dz =
                    worker_shadow.route_workspace_signal[coordinate] *
                    (1.0f0 - workspace_value * workspace_value)
                worker_shadow.route_write_signal[
                    coordinate,
                    cycle,
                ] = dz / Float32(model.workspace_k)
                worker_shadow.route_workspace_signal[coordinate] =
                    dz * cache.workspace_decay
                inverse_k = inv(Float32(model.workspace_k))
                for block in 1:model.blocks
                    node =
                        coordinate +
                        (block - 1) * model.node_dim
                    worker_shadow.route_block_signal[block] = muladd(
                        dz * inverse_k,
                        arena.membrane[node, cycle, flat],
                        coordinate == 1 ? 0.0f0 :
                            worker_shadow.route_block_signal[block],
                    )
                end
            end
            for block in 1:model.blocks
                offset = (block - 1) * model.node_dim
                active_route_signal =
                    worker_shadow.route_block_signal[block]
                for coordinate in 1:model.node_dim
                    source = offset + coordinate
                    membrane =
                        arena.membrane[source, cycle, flat]
                    spike =
                        membrane >= cache.threshold[source] ?
                        1.0f0 : 0.0f0
                    spike == 0.0f0 && continue
                    outgoing_signal = 0.0f0
                    for relation in 1:model.fanout
                        cache.gate_hard[source, relation] == 0.0f0 &&
                            continue
                        destination =
                            model.destination_for_source[
                                source,
                                relation,
                            ]
                        delay = cache.delay[source, relation]
                        state_signal =
                            (1.0f0 - delay) *
                            worker_shadow.route_state_signal[
                                destination,
                                cycle + 1,
                            ]
                        if cycle < model.cycles
                            state_signal = muladd(
                                delay,
                                worker_shadow.route_state_signal[
                                    destination,
                                    cycle + 2,
                                ],
                                state_signal,
                            )
                        end
                        outgoing_signal = muladd(
                            parameters.synapse_weight[source, relation],
                            state_signal,
                            outgoing_signal,
                        )
                    end
                    active_route_signal = muladd(
                        spike,
                        outgoing_signal,
                        active_route_signal,
                    )
                end
                worker_shadow.route_block_signal[block] =
                    active_route_signal
            end
            route_sum = 0.0f0
            route_projection = 0.0f0
            for block in 1:model.blocks
                score = 0.0f0
                magnitude = 0.0f0
                offset = (block - 1) * model.node_dim
                for coordinate in 1:model.node_dim
                    node = offset + coordinate
                    membrane = arena.membrane[node, cycle, flat]
                    score = muladd(
                        membrane *
                        parameters.workspace_key[coordinate, block],
                        arena.query[coordinate, flat],
                        score,
                    )
                    magnitude += abs(membrane)
                end
                soft = sigmoid(
                    (score + 0.05f0 * magnitude) /
                    model.route_temperature,
                )
                worker_shadow.route_block_soft[block] = soft
                route_sum += soft
                route_projection = muladd(
                    worker_shadow.route_block_signal[block],
                    soft,
                    route_projection,
                )
            end
            inverse_route_sum =
                inv(max(route_sum, eps(Float32)))
            for block in 1:model.blocks
                soft = worker_shadow.route_block_soft[block]
                score_cotangent =
                    Float32(model.workspace_k) *
                    inverse_route_sum *
                    (
                        worker_shadow.route_block_signal[block] -
                        route_projection * inverse_route_sum
                    ) *
                    soft *
                    (1.0f0 - soft) /
                    model.route_temperature
                worker_shadow.route_score_signal[block, cycle] =
                    score_cotangent
                offset = (block - 1) * model.node_dim
                for coordinate in 1:model.node_dim
                    node = offset + coordinate
                    membrane = arena.membrane[node, cycle, flat]
                    query = arena.query[coordinate, flat]
                    key = parameters.workspace_key[coordinate, block]
                    accumulator.workspace_key_gradient[
                        coordinate,
                        block,
                    ] = muladd(
                        score_cotangent * membrane,
                        query,
                        accumulator.workspace_key_gradient[
                            coordinate,
                            block,
                        ],
                    )
                    worker_shadow.route_query_signal[coordinate] = muladd(
                        score_cotangent * membrane,
                        key,
                        worker_shadow.route_query_signal[coordinate],
                    )
                end
            end
        end
    end
    query_projection_mean = 0.0f0
    @inbounds for coordinate in 1:model.node_dim
        query = arena.query[coordinate, flat]
        normalized_cotangent =
            worker_shadow.route_query_signal[coordinate] *
            Model.QUERY_NORM_SCALE *
            (1.0f0 - query * query)
        worker_shadow.route_query_signal[coordinate] =
            normalized_cotangent
        query_projection_mean = muladd(
            normalized_cotangent,
            arena.query_pre[coordinate, flat],
            query_projection_mean,
        )
    end
    query_projection_mean /= Float32(model.node_dim)
    query_inv_rms = arena.query_inv_rms[flat]
    query_inv_rms_squared = query_inv_rms * query_inv_rms
    @inbounds for coordinate in 1:model.node_dim
        query_cotangent = query_inv_rms * (
            worker_shadow.route_query_signal[coordinate] -
            arena.query_pre[coordinate, flat] *
            query_inv_rms_squared *
            query_projection_mean
        )
        for rail in 1:Model.INPUT_RAILS
            accumulator.query_weight_gradient[coordinate, rail] =
                muladd(
                    query_cotangent,
                    arena.rails[rail, flat],
                    accumulator.query_weight_gradient[coordinate, rail],
                )
        end
    end
    return nothing
end

"""
Replay one candidate's stored forward trajectory in forward time and accumulate
the shadow e-prop update.  Only worker-owned trace and gradient matrices are
written.  Reversed time is exposed solely as a falsification control.
"""
function shadow_eprop_candidate!(
    worker_shadow::EPropWorkerShadow,
    accumulator::EPropWorkerShadow,
    shadow::EPropShadowState,
    arena::TrainingArena,
    model,
    parameters,
    cache::ParameterCache,
    flat::Int,
)
    signal_flat = _replay_signal_flat(shadow, flat)
    listnet_delta_q = shadow.config.third_factor_mode === :zero ?
        0.0f0 : @inbounds(arena.listnet_q_gradient[signal_flat])
    zero_signal = shadow.config.third_factor_mode === :zero
    if shadow.config.feedback_mode !== :block_local
        _prepare_eprop_node_signal!(
            worker_shadow,
            shadow,
            arena,
            model,
            parameters,
            flat,
            signal_flat,
            listnet_delta_q,
            zero_signal,
        )
    end
    _shadow_eprop_edges!(
        worker_shadow,
        accumulator,
        shadow,
        arena,
        model,
        parameters,
        cache,
        flat,
    )
    _shadow_eprop_nodes!(
        worker_shadow,
        accumulator,
        shadow,
        arena,
        model,
        cache,
        flat,
    )
    _shadow_workspace_decay!(
        worker_shadow,
        accumulator,
        shadow,
        arena,
        model,
        parameters,
        cache,
        flat,
    )
    _shadow_routing_candidate!(
        worker_shadow,
        accumulator,
        shadow,
        arena,
        model,
        parameters,
        cache,
        flat,
        listnet_delta_q,
    )
    _shadow_eprop_inputs!(
        worker_shadow,
        accumulator,
        shadow,
        arena,
        model,
        parameters,
        cache,
        flat,
    )
    return nothing
end

function shadow_eprop_candidate!(
    worker_shadow::EPropWorkerShadow,
    shadow::EPropShadowState,
    arena::TrainingArena,
    model,
    parameters,
    cache::ParameterCache,
    flat::Int,
)
    return shadow_eprop_candidate!(
        worker_shadow,
        worker_shadow,
        shadow,
        arena,
        model,
        parameters,
        cache,
        flat,
    )
end

@inline function _feature_value(
    arena::TrainingArena,
    model,
    flat::Int,
    feature::Int,
)
    node_dim = model.node_dim
    feature <= node_dim && return (
        arena.workspace[feature, model.cycles + 1, flat] *
        arena.workspace_inv_rms[flat]
    )
    feature <= 2 * node_dim || throw(BoundsError(1:(2 * node_dim), feature))
    coordinate = feature - node_dim
    selected_pool = 0.0f0
    @inbounds for block in 1:model.blocks
        node = coordinate + (block - 1) * node_dim
        selected_pool = muladd(
            arena.membrane[node, model.cycles + 1, flat],
            arena.block_mask[block, model.cycles, flat],
            selected_pool,
        )
    end
    return (
        selected_pool /
        Float32(model.workspace_k) *
        arena.selected_pool_inv_rms[flat]
    )
end

"""
Analytic output-head reverse pass for one candidate.

`gradient` is worker-owned for the entire update.  Consequently every `+=`
below is lock-free, while the parameters and forward arena are read-only.
The recurrent graph is deliberately not traversed; `scratch.dfeatures`
retains the head cotangent so the full VJP can continue from the same helper.
"""
function backward_head_candidate!(
    gradient,
    arena::TrainingArena,
    model,
    parameters,
    cache::ParameterCache,
    scratch::CandidateScratch,
    flat::Int,
)
    node_dim = model.node_dim
    fill!(scratch.dfeatures, 0.0f0)
    fill!(scratch.dhidden, 0.0f0)
    @inbounds for coordinate in 1:node_dim
        selected_pool = 0.0f0
        for block in 1:model.blocks
            node = coordinate + (block - 1) * node_dim
            selected_pool = muladd(
                arena.membrane[node, model.cycles + 1, flat],
                arena.block_mask[block, model.cycles, flat],
                selected_pool,
            )
        end
        scratch.features[coordinate] =
            arena.workspace[coordinate, model.cycles + 1, flat] *
            arena.workspace_inv_rms[flat]
        scratch.features[node_dim + coordinate] =
            selected_pool /
            Float32(model.workspace_k) *
            arena.selected_pool_inv_rms[flat]
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
    hidden_projection_mean = 0.0f0
    @inbounds for hidden in 1:model.hidden
        hidden_value = arena.hidden[hidden, flat]
        normalized_cotangent =
            scratch.dhidden[hidden] *
            Model.HIDDEN_NORM_SCALE *
            (1.0f0 - hidden_value * hidden_value)
        scratch.dhidden[hidden] = normalized_cotangent
        hidden_projection_mean = muladd(
            normalized_cotangent,
            arena.hidden_pre[hidden, flat],
            hidden_projection_mean,
        )
    end
    hidden_projection_mean /= Float32(model.hidden)
    hidden_inv_rms = arena.hidden_inv_rms[flat]
    hidden_inv_rms_squared = hidden_inv_rms * hidden_inv_rms
    @inbounds for hidden in 1:model.hidden
        cotangent = hidden_inv_rms * (
            scratch.dhidden[hidden] -
            arena.hidden_pre[hidden, flat] *
            hidden_inv_rms_squared *
            hidden_projection_mean
        )
        gradient.head_bias[hidden] += cotangent
        for feature in 1:(2 * node_dim)
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
    return nothing
end

"""
Analytic full reverse pass for one candidate.

The output head is handled by `backward_head_candidate!`; this function then
walks the complete recurrent graph in reverse.
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
    backward_head_candidate!(
        gradient,
        arena,
        model,
        parameters,
        cache,
        scratch,
        flat,
    )
    fill!(scratch.dseed, 0.0f0)

    dmembrane_next = scratch.dmembrane_a
    dmembrane_previous = scratch.dmembrane_b
    fill!(dmembrane_next, 0.0f0)
    dworkspace_next = scratch.dworkspace_a
    dworkspace_previous = scratch.dworkspace_b
    workspace_projection_mean = 0.0f0
    pool_projection_mean = 0.0f0
    @inbounds for coordinate in 1:node_dim
        workspace_value =
            arena.workspace[coordinate, model.cycles + 1, flat]
        selected_pool = 0.0f0
        for block in 1:model.blocks
            node = coordinate + (block - 1) * node_dim
            selected_pool = muladd(
                arena.membrane[node, model.cycles + 1, flat],
                arena.block_mask[block, model.cycles, flat],
                selected_pool,
            )
        end
        selected_pool /= Float32(model.workspace_k)
        scratch.dquery[coordinate] = selected_pool
        workspace_projection_mean = muladd(
            scratch.dfeatures[coordinate],
            workspace_value,
            workspace_projection_mean,
        )
        pool_projection_mean = muladd(
            scratch.dfeatures[node_dim + coordinate],
            selected_pool,
            pool_projection_mean,
        )
    end
    workspace_projection_mean /= Float32(node_dim)
    pool_projection_mean /= Float32(node_dim)
    workspace_inv_rms = arena.workspace_inv_rms[flat]
    workspace_inv_rms_squared =
        workspace_inv_rms * workspace_inv_rms
    pool_inv_rms = arena.selected_pool_inv_rms[flat]
    pool_inv_rms_squared = pool_inv_rms * pool_inv_rms
    @inbounds for coordinate in 1:node_dim
        workspace_value =
            arena.workspace[coordinate, model.cycles + 1, flat]
        dworkspace_next[coordinate] = workspace_inv_rms * (
            scratch.dfeatures[coordinate] -
            workspace_value *
            workspace_inv_rms_squared *
            workspace_projection_mean
        )
        selected_pool = scratch.dquery[coordinate]
        pool_cotangent = pool_inv_rms * (
            scratch.dfeatures[node_dim + coordinate] -
            selected_pool *
            pool_inv_rms_squared *
            pool_projection_mean
        ) / Float32(model.workspace_k)
        for block in 1:model.blocks
            node = coordinate + (block - 1) * node_dim
            dmembrane_next[node] =
                pool_cotangent *
                arena.block_mask[block, model.cycles, flat]
        end
    end
    fill!(scratch.dquery, 0.0f0)
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

        # Candidate-local score standardization and softmax top-k surrogate
        # VJP. This exactly matches SerialWorkspaceSNN._workspace_mask.
        _compute_scores!(
            scratch, arena, model, parameters, cycle, flat,
        )
        score_mean = 0.0f0
        for block in 1:model.blocks
            score_mean += scratch.scores[block]
        end
        score_mean /= Float32(model.blocks)
        score_variance = 0.0f0
        for block in 1:model.blocks
            centered = scratch.scores[block] - score_mean
            scratch.soft_route[block] = centered
            score_variance = muladd(
                centered,
                centered,
                score_variance,
            )
        end
        score_inv_rms = inv(sqrt(
            score_variance / Float32(model.blocks) +
            Model.RMS_NORM_EPS,
        ))
        maximum_logit = -Inf32
        for block in 1:model.blocks
            standardized =
                scratch.soft_route[block] * score_inv_rms
            scratch.soft_route[block] = standardized
            maximum_logit = max(
                maximum_logit,
                standardized / model.route_temperature,
            )
        end
        route_sum = 0.0f0
        for block in 1:model.blocks
            probability = exp(
                scratch.soft_route[block] /
                model.route_temperature -
                maximum_logit,
            )
            scratch.base_route[block] = probability
            route_sum += probability
        end
        inverse_route_sum = inv(max(route_sum, eps(Float32)))
        route_projection = 0.0f0
        for block in 1:model.blocks
            probability =
                scratch.base_route[block] * inverse_route_sum
            scratch.base_route[block] = probability
            route_projection = muladd(
                scratch.dblock_mask[block],
                probability,
                route_projection,
            )
        end
        standardized_gradient_mean = 0.0f0
        standardized_projection_mean = 0.0f0
        for block in 1:model.blocks
            standardized_gradient =
                Float32(model.workspace_k) *
                scratch.base_route[block] *
                (scratch.dblock_mask[block] - route_projection) /
                model.route_temperature
            scratch.route_eligibility[block] =
                standardized_gradient
            standardized_gradient_mean +=
                standardized_gradient
            standardized_projection_mean = muladd(
                standardized_gradient,
                scratch.soft_route[block],
                standardized_projection_mean,
            )
        end
        standardized_gradient_mean /= Float32(model.blocks)
        standardized_projection_mean /= Float32(model.blocks)
        for block in 1:model.blocks
            score_cotangent = score_inv_rms * (
                scratch.route_eligibility[block] -
                standardized_gradient_mean -
                scratch.soft_route[block] *
                standardized_projection_mean
            )
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
    query_projection_mean = 0.0f0
    @inbounds for coordinate in 1:node_dim
        query = arena.query[coordinate, flat]
        normalized_cotangent =
            scratch.dquery[coordinate] *
            Model.QUERY_NORM_SCALE *
            (1.0f0 - query * query)
        scratch.dquery[coordinate] = normalized_cotangent
        query_projection_mean = muladd(
            normalized_cotangent,
            arena.query_pre[coordinate, flat],
            query_projection_mean,
        )
    end
    query_projection_mean /= Float32(node_dim)
    query_inv_rms = arena.query_inv_rms[flat]
    query_inv_rms_squared = query_inv_rms * query_inv_rms
    @inbounds for coordinate in 1:node_dim
        query_cotangent = query_inv_rms * (
            scratch.dquery[coordinate] -
            arena.query_pre[coordinate, flat] *
            query_inv_rms_squared *
            query_projection_mean
        )
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

"""
Allocation-free diagnostics for the most recently completed arena update.

Routing quantities are measured from the exact forward trajectory:

* `route_selection_gap` is the mean signed difference between the lowest
  selected score and highest unselected score (negative is possible under
  stochastic routing).
* `route_score_rms` is the RMS of every raw block score.
* `hard_mask_unique_fraction` is the exact number of distinct candidate-cycle
  hard masks divided by the number of candidate-cycle decisions. Hash
  collisions are resolved by comparing the masks themselves.
* `hard_mask_cycle_churn` is the fraction of block bits that change between
  adjacent cycles.
* `entropy_floor_violation_fraction` is the fraction of candidate-cycle base
  policies whose normalized entropy is below the configured floor.

`utility_swap_gap` is the mean signed
`best_inactive_utility - connection_cost - worst_active_utility` over
utility-scheduled, at-budget nodes; zero means that no comparable utility
consolidation was dispatched. `consolidation_scheduled` records the periodic
clock independently of whether structural learning is enabled, while
`consolidation_actual` records actual dispatch. `net_mask_flips` is the exact
Hamming distance from the hard mask used by the forward trajectory to the
post-update hard mask.

Transformed parameter fields are arithmetic means after AdamW and any
structural consolidation. Membrane margins are `membrane - threshold` at each
spike decision. Surrogate statistics use `_eprop_surrogate` at those same
decisions. `eligibility_rms` is the RMS of every recurrent weight eligibility
actually consumed by the configured e-prop signal schedule, before
multiplication by its third factor.
"""
mutable struct ArenaMetrics
    wall_seconds::Float64
    cpu_seconds::Float64
    allocation_bytes::Int128
    gc_seconds::Float64
    pack_seconds::Float64
    forward_seconds::Float64
    loss_seconds::Float64
    shadow_seconds::Float64
    backward_seconds::Float64
    optimizer_seconds::Float64
    consolidation_seconds::Float64
    whole_machine_cpu_percent::Float64
    active_worker_cpu_percent::Float64
    states_per_second::Float64
    route_selection_gap::Float64
    route_score_rms::Float64
    hard_mask_unique_fraction::Float64
    hard_mask_cycle_churn::Float64
    entropy_floor_violation_fraction::Float64
    utility_swap_gap::Float64
    consolidation_scheduled::Bool
    consolidation_actual::Bool
    net_mask_flips::Int
    gate_probability_mean::Float64
    gate_derivative_mean::Float64
    delay_mean::Float64
    delay_derivative_mean::Float64
    leak_mean::Float64
    leak_derivative_mean::Float64
    threshold_mean::Float64
    threshold_derivative_mean::Float64
    workspace_decay::Float64
    workspace_decay_derivative::Float64
    membrane_threshold_margin_mean::Float64
    membrane_threshold_margin_rms::Float64
    surrogate_sensitivity_mean::Float64
    surrogate_sensitivity_rms::Float64
    eligibility_rms::Float64
    local_q_loss::Float64
    local_death_loss::Float64
    local_quantile_loss::Float64
    local_geometry_loss::Float64
end

ArenaMetrics() = ArenaMetrics(
    0.0, 0.0, Int128(0), 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    false, false, 0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
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
    transformed_value_sums::Vector{Float64}
    transformed_derivative_sums::Vector{Float64}
    consolidation_gate_probability_delta_sums::Vector{Float64}
    consolidation_gate_derivative_delta_sums::Vector{Float64}
    consolidation_utility_swap_gap_sums::Vector{Float64}
    consolidation_utility_swap_gap_counts::Vector{Int}
    mask_before_update::Matrix{UInt8}
    route_mask_hash_table::Vector{UInt64}
    route_mask_index_table::Vector{Int32}
    synapse_utility::Matrix{Float32}
    utility_updates::Int
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
    mask_decisions = arena.capacity * model.cycles
    mask_hash_table_size = nextpow(2, max(2, 2 * mask_decisions))
    empty_loss = LossRecord(
        0.0f0, 0.0f0, 0.0f0, 0.0f0,
        0.0f0, 0.0f0, 0.0f0, 0.0f0,
        0.0f0, 0.0f0, 0.0f0, 0.0f0,
        0.0f0, 0.0f0, 0.0f0, 0.0f0,
        0.0f0, 0,
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
        LossScratch(width, state_batch),
        _zero_parameter_tree(parameters),
        shards,
        zeros(Float64, length(shards)),
        zeros(Int, cld(model.blocks * model.node_dim, 32)),
        zeros(Float64, length(shards)),
        zeros(Float64, length(shards)),
        zeros(Float64, cld(model.blocks * model.node_dim, 32)),
        zeros(Float64, cld(model.blocks * model.node_dim, 32)),
        zeros(Float64, cld(model.blocks * model.node_dim, 32)),
        zeros(Int, cld(model.blocks * model.node_dim, 32)),
        zeros(UInt8, size(parameters.gate_logits)),
        zeros(UInt64, mask_hash_table_size),
        zeros(Int32, mask_hash_table_size),
        zeros(Float32, size(parameters.synapse_weight)),
        0,
        Float32(structure_weight),
        0.0f0,
        empty_loss,
        NaN,
        0,
        ArenaMetrics(),
    )
end

training_arena(trainer::ArenaTrainer) = trainer.arena

mutable struct ArenaWorkerRuntime{G,S}
    gradient::G
    scratch::CandidateScratch
    eprop_shadow::S
    jobs::UInt64
    cpu_ticks::UInt64
end

ArenaWorkerRuntime(
    model,
    parameters,
    eprop_shadow,
    synapse_learning_mode::Symbol,
    eprop_gradient_storage::Bool=true,
) = ArenaWorkerRuntime(
    synapse_learning_mode === :local_eligibility ?
        _zero_head_gradient_tree(parameters) :
        _zero_parameter_tree(parameters),
    CandidateScratch(model),
    eprop_shadow === nothing ? nothing :
        EPropWorkerShadow(
            model,
            parameters,
            eprop_shadow.config;
            gradient_storage=eprop_gradient_storage,
        ),
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
    ARENA_EPROP_SHADOW = 6
    ARENA_HEAD_BACKWARD = 7
    ARENA_EPROP_SIGNAL = 8
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

mutable struct ArenaExecutor{W,T,D,S}
    queue::Queue.BoundedMPMCQueue{ArenaWorkItem}
    active_workers::Int
    julia_workers::Int
    cpuset_mode::Symbol
    workers::W
    trainer::T
    dataset::D
    eprop_shadow::S
    eprop_reducer_count::Int
    synapse_learning_mode::Symbol
    stochastic_routing::Bool
    routing_seed::UInt64
    structural_learning_mode::Symbol
    utility_decay::Float32
    utility_connection_cost::Float32
    utility_keep_fraction::Float32
    utility_turnover_period::Int
    consolidation_event_ordinal::Int
    generation::Base.Threads.Atomic{UInt32}
    remaining::Base.Threads.Atomic{Int}
    shutdown_requested::Base.Threads.Atomic{UInt32}
    ready_workers::Base.Threads.Atomic{Int}
    booted_workers::Base.Threads.Atomic{Int}
    failure_worker::Base.Threads.Atomic{Int}
    failures::Vector{Any}
    bindings::Vector{Any}
    bindings_released::Vector{Bool}
    startup_event::Base.Event
    started::Bool
end

function ArenaExecutor(
    trainer::ArenaTrainer,
    dataset;
    active_workers::Int=Base.Threads.nthreads(:default),
    cpuset_mode::Symbol=:none,
    queue_capacity::Int=2048,
    eprop_shadow_config::Union{Nothing,EPropShadowConfig}=nothing,
    eprop_reducer_count::Int=active_workers,
    synapse_learning_mode::Symbol=:vjp,
    stochastic_routing::Bool=false,
    routing_seed::Integer=0x524f555445534545,
    structural_learning_mode::Symbol=:auto,
    utility_decay::Real=0.99f0,
    utility_connection_cost::Real=0.0f0,
    utility_keep_fraction::Real=0.50f0,
    utility_turnover_period::Int=128,
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
    synapse_learning_mode in (:vjp, :local_eligibility) ||
        throw(ArgumentError(
            "synapse_learning_mode must be vjp or local_eligibility",
        ))
    if eprop_shadow_config === nothing
        eprop_reducer_count = 0
    else
        1 <= eprop_reducer_count <= active_workers ||
            throw(ArgumentError(
                "eprop_reducer_count must be in 1:active_workers",
            ))
    end
    resolved_structural_mode = structural_learning_mode === :auto ?
        (
            synapse_learning_mode === :local_eligibility ?
            :frozen : :legacy
        ) : structural_learning_mode
    resolved_structural_mode in (:legacy, :frozen, :utility) ||
        throw(ArgumentError(
            "structural_learning_mode must be auto, legacy, frozen, or utility",
        ))
    synapse_learning_mode === :local_eligibility &&
       resolved_structural_mode === :legacy &&
        throw(ArgumentError(
            "local_eligibility cannot use legacy magnitude consolidation",
        ))
    if resolved_structural_mode === :utility
        eprop_shadow_config === nothing && throw(ArgumentError(
            "utility structure learning requires eprop_shadow_config",
        ))
        eprop_shadow_config.edge_parameter_mode === :weight_gate_delay ||
            throw(ArgumentError(
                "utility structure learning requires " *
                "edge_parameter_mode=:weight_gate_delay",
            ))
    end
    decay = Float32(utility_decay)
    connection_cost = Float32(utility_connection_cost)
    keep_fraction = Float32(utility_keep_fraction)
    isfinite(decay) && 0.0f0 <= decay < 1.0f0 ||
        throw(ArgumentError("utility_decay must be in [0, 1)"))
    isfinite(connection_cost) && connection_cost >= 0.0f0 ||
        throw(ArgumentError(
            "utility_connection_cost must be finite and nonnegative",
        ))
    isfinite(keep_fraction) && 0.0f0 < keep_fraction < 1.0f0 ||
        throw(ArgumentError("utility_keep_fraction must be in (0, 1)"))
    utility_turnover_period >= 1 || throw(ArgumentError(
        "utility_turnover_period must be positive",
    ))
    if synapse_learning_mode === :local_eligibility
        eprop_shadow_config === nothing && throw(ArgumentError(
            "local_eligibility requires eprop_shadow_config",
        ))
        config = eprop_shadow_config
        config.feedback_mode in (:symmetric_head, :block_local) ||
            throw(ArgumentError(
                "local_eligibility requires feedback_mode=:symmetric_head " *
                "or :block_local",
            ))
        config.eligibility_mode === :membrane || throw(ArgumentError(
            "local_eligibility requires eligibility_mode=:membrane",
        ))
        if config.feedback_mode === :symmetric_head
            config.signal_schedule === :terminal || throw(ArgumentError(
                "symmetric-head local_eligibility requires " *
                "signal_schedule=:terminal",
            ))
        else
            config.signal_schedule === :all_cycles || throw(ArgumentError(
                "block-local local_eligibility requires " *
                "signal_schedule=:all_cycles",
            ))
        end
        config.third_factor_mode in (
            :aligned,
            :zero,
            :candidate_shuffle,
        ) || throw(ArgumentError(
            "local_eligibility third_factor_mode must be aligned, zero, " *
            "or candidate_shuffle",
        ))
        config.time_order === :forward || throw(ArgumentError(
            "local_eligibility requires time_order=:forward",
        ))
        stochastic_routing &&
           config.routing_parameter_mode === :local_soft &&
            throw(ArgumentError(
                "stochastic local_eligibility cannot use routing_parameter_mode=:local_soft; use :three_factor or :none",
            ))
    end
    eprop_shadow = eprop_shadow_config === nothing ? nothing :
        EPropShadowState(
            trainer.model,
            trainer.parameters,
            trainer.arena.capacity,
            eprop_shadow_config,
        )
    workers = [
        ArenaWorkerRuntime(
            trainer.model,
            trainer.parameters,
            eprop_shadow,
            synapse_learning_mode,
            worker_slot <= eprop_reducer_count,
        )
        for worker_slot in 1:active_workers
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
        eprop_shadow,
        eprop_reducer_count,
        synapse_learning_mode,
        stochastic_routing,
        UInt64(routing_seed),
        resolved_structural_mode,
        decay,
        connection_cost,
        keep_fraction,
        utility_turnover_period,
        1,
        Base.Threads.Atomic{UInt32}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{UInt32}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Any[nothing for _ in 1:julia_workers],
        Any[nothing for _ in 1:julia_workers],
        fill(false, julia_workers),
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
    local_edge_all =
        executor.eprop_shadow !== nothing &&
        executor.eprop_shadow.config.edge_parameter_mode ===
            :weight_gate_delay
    local_node_all =
        executor.eprop_shadow !== nothing &&
        executor.eprop_shadow.config.node_parameter_mode !== :none
    local_full_state =
        executor.eprop_shadow !== nothing &&
        executor.eprop_shadow.config.node_parameter_mode === :full_state
    local_routing_all =
        executor.eprop_shadow !== nothing &&
        executor.eprop_shadow.config.routing_parameter_mode !== :none
    use_local_parameter =
        executor.synapse_learning_mode === :local_eligibility &&
        (
            F === :synapse_weight ||
            (local_full_state && F === :input_gain) ||
            (local_full_state && F === :input_bias) ||
            (local_edge_all && F === :gate_logits) ||
            (local_edge_all && F === :delay_logits) ||
            (local_node_all && F === :feedback_gain) ||
            (local_node_all && F === :leak_logits) ||
            (local_node_all && F === :threshold_logits) ||
            (local_routing_all && F === :workspace_key) ||
            (local_routing_all && F === :query_weight) ||
            (local_full_state && F === :workspace_decay_logit)
        )
    if use_local_parameter &&
       executor.eprop_shadow.config.third_factor_mode === :zero
        @inbounds for index in first_index:last_index
            global_gradient[index] = 0.0f0
        end
        return 0.0
    end
    @inbounds for index in first_index:last_index
        value = 0.0f0
        source_count = use_local_parameter ?
            executor.eprop_reducer_count :
            length(executor.workers)
        for source_index in 1:source_count
            worker = executor.workers[source_index]
            source = if use_local_parameter && F === :synapse_weight
                worker.eprop_shadow.gradient
            elseif use_local_parameter && F === :input_gain
                worker.eprop_shadow.input_gain_gradient
            elseif use_local_parameter && F === :input_bias
                worker.eprop_shadow.input_bias_gradient
            elseif use_local_parameter && F === :gate_logits
                worker.eprop_shadow.gate_gradient
            elseif use_local_parameter && F === :delay_logits
                worker.eprop_shadow.delay_gradient
            elseif use_local_parameter && F === :feedback_gain
                worker.eprop_shadow.feedback_gradient
            elseif use_local_parameter && F === :leak_logits
                worker.eprop_shadow.leak_gradient
            elseif use_local_parameter && F === :threshold_logits
                worker.eprop_shadow.threshold_gradient
            elseif use_local_parameter && F === :workspace_key
                worker.eprop_shadow.workspace_key_gradient
            elseif use_local_parameter && F === :query_weight
                worker.eprop_shadow.query_weight_gradient
            elseif use_local_parameter && F === :workspace_decay_logit
                worker.eprop_shadow.workspace_decay_gradient
            else
                getproperty(worker.gradient, F)
            end
            value += source[index]
            use_local_parameter || (source[index] = 0.0f0)
        end
        if F === :gate_logits &&
           !(
               executor.eprop_shadow !== nothing &&
               executor.eprop_shadow.config.third_factor_mode === :zero
           )
            # The density regularizer is part of the reported composite loss.
            # It is global rather than candidate-local, so add it exactly once
            # here for both VJP and local-hybrid learning.  It is deliberately
            # excluded from candidate responsibility/utility evidence.
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
        proposed_parameter =
            old_parameter - optimizer.learning_rate * (
            adam + optimizer.weight_decay * old_parameter
        )
        if F === :gate_logits &&
           executor.structural_learning_mode !== :legacy
            trajectory_active = if executor.eprop_shadow === nothing
                old_parameter >= 0.0f0
            else
                executor.eprop_shadow.trajectory_gate_hard[index] != 0x00
            end
            (old_parameter >= 0.0f0) == trajectory_active || error(
                "gate parameter crossed its trajectory hard-mask side " *
                "before Adam projection",
            )
            parameter[index] = trajectory_active ?
                max(proposed_parameter, GATE_SIGN_EPSILON) :
                min(proposed_parameter, -GATE_SIGN_EPSILON)
        else
            parameter[index] = proposed_parameter
        end
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
        cache.workspace_decay =
            Model.bounded_workspace_decay(
                parameters.workspace_decay_logit[1],
            )
        cache.workspace_decay_derivative =
            Model.bounded_workspace_decay_derivative(
                parameters.workspace_decay_logit[1],
            )
    end
    return nothing
end

@inline function _record_transformed_shard_telemetry!(
    trainer::ArenaTrainer,
    target::Int,
    field::Int,
    first_index::Int,
    last_index::Int,
)
    value_sum = 0.0
    derivative_sum = 0.0
    cache = trainer.cache
    if field == 6
        @inbounds for index in first_index:last_index
            value_sum += Float64(cache.leak[index])
            derivative_sum += Float64(cache.leak_derivative[index])
        end
    elseif field == 7
        @inbounds for index in first_index:last_index
            value_sum += Float64(cache.threshold[index])
            derivative_sum += Float64(cache.threshold_derivative[index])
        end
    elseif field == 9
        @inbounds for index in first_index:last_index
            value_sum += Float64(cache.gate_probability[index])
            derivative_sum += Float64(cache.gate_derivative[index])
        end
    elseif field == 10
        @inbounds for index in first_index:last_index
            value_sum += Float64(cache.delay[index])
            derivative_sum += Float64(cache.delay_derivative[index])
        end
    elseif field == 11
        value_sum = Float64(cache.workspace_decay)
        derivative_sum = Float64(cache.workspace_decay_derivative)
    end
    @inbounds trainer.transformed_value_sums[target] = value_sum
    @inbounds trainer.transformed_derivative_sums[target] = derivative_sum
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
    local_edge_all =
        executor.eprop_shadow !== nothing &&
        executor.eprop_shadow.config.edge_parameter_mode ===
            :weight_gate_delay
    local_node_all =
        executor.eprop_shadow !== nothing &&
        executor.eprop_shadow.config.node_parameter_mode !== :none
    local_full_state =
        executor.eprop_shadow !== nothing &&
        executor.eprop_shadow.config.node_parameter_mode === :full_state
    local_routing_all =
        executor.eprop_shadow !== nothing &&
        executor.eprop_shadow.config.routing_parameter_mode !== :none
    local_trainable =
        field == 8 ||
        (local_full_state && field == 1) ||
        (local_full_state && field == 2) ||
        (local_edge_all && field == 9) ||
        (local_edge_all && field == 10) ||
        (local_node_all && field == 5) ||
        (local_node_all && field == 6) ||
        (local_node_all && field == 7) ||
        (local_routing_all && field == 3) ||
        (local_routing_all && field == 4) ||
        (local_full_state && field == 11) ||
        12 <= field <= 15
    if executor.synapse_learning_mode === :local_eligibility &&
       !local_trainable
        destination = getproperty(
            trainer.gradient,
            PARAMETER_FIELDS[field],
        )
        @inbounds for index in first_index:last_index
            destination[index] = 0.0f0
        end
        trainer.gradient_norm_squares[target] = 0.0
        _record_transformed_shard_telemetry!(
            trainer,
            target,
            field,
            first_index,
            last_index,
        )
        return nothing
    end
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
    _record_transformed_shard_telemetry!(
        trainer,
        target,
        field,
        first_index,
        last_index,
    )
    return nothing
end

function _consolidate_node_range!(
    trainer::ArenaTrainer,
    executor::ArenaExecutor,
    scratch::CandidateScratch,
    target::Int,
)
    model = trainer.model
    parameters = trainer.parameters
    cache = trainer.cache
    structural_mode = executor.structural_learning_mode
    structural_mode in (:legacy, :utility) || error(
        "consolidation dispatched for structural mode $structural_mode",
    )
    first_node = (target - 1) * 32 + 1
    last_node = min(target * 32, model.blocks * model.node_dim)
    flips = 0
    gate_probability_delta_sum = 0.0
    gate_derivative_delta_sum = 0.0
    utility_swap_gap_sum = 0.0
    utility_swap_gap_count = 0
    keep_fraction = structural_mode === :utility ?
        executor.utility_keep_fraction : 0.50f0
    keep = clamp(
        round(Int, keep_fraction * model.fanout),
        1,
        model.fanout - 1,
    )
    @inbounds for node in first_node:last_node
        fill!(scratch.consolidation_selected, false)
        for relation in 1:model.fanout
            scratch.consolidation_evidence[relation] =
                structural_mode === :utility ?
                trainer.synapse_utility[node, relation] :
                (
                    abs(parameters.synapse_weight[node, relation]) *
                    cache.gate_probability[node, relation]
                )
        end
        if structural_mode === :legacy
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
                best == 0 && error(
                    "legacy consolidation could not fill its budget",
                )
                scratch.consolidation_selected[best] = true
            end
        else
            selected_count = 0
            for relation in 1:model.fanout
                selected =
                    cache.gate_hard[node, relation] != 0.0f0
                scratch.consolidation_selected[relation] = selected
                selected_count += selected
            end
            executor.consolidation_event_ordinal >= 1 || error(
                "utility consolidation requires a positive event ordinal",
            )
            scheduled = mod(
                node - 1,
                executor.utility_turnover_period,
            ) == mod(
                executor.consolidation_event_ordinal - 1,
                executor.utility_turnover_period,
            )
            if scheduled
                best_inactive = 0
                best_inactive_score = -Inf32
                worst_active = 0
                worst_active_score = Inf32
                for relation in 1:model.fanout
                    evidence =
                        scratch.consolidation_evidence[relation]
                    if scratch.consolidation_selected[relation]
                        if worst_active == 0 ||
                           evidence < worst_active_score
                            worst_active = relation
                            worst_active_score = evidence
                        end
                    else
                        score =
                            evidence -
                            executor.utility_connection_cost
                        if best_inactive == 0 ||
                           score > best_inactive_score
                            best_inactive = relation
                            best_inactive_score = score
                        end
                    end
                end
                if selected_count == keep &&
                   best_inactive != 0 &&
                   worst_active != 0
                    utility_swap_gap_sum +=
                        Float64(best_inactive_score) -
                        Float64(worst_active_score)
                    utility_swap_gap_count += 1
                end
                if selected_count > keep
                    worst_active != 0 || error(
                        "utility budget reduction found no active gate",
                    )
                    scratch.consolidation_selected[
                        worst_active
                    ] = false
                elseif selected_count < keep
                    best_inactive != 0 || error(
                        "utility budget growth found no inactive gate",
                    )
                    scratch.consolidation_selected[
                        best_inactive
                    ] = true
                elseif best_inactive != 0 &&
                       worst_active != 0 &&
                       best_inactive_score > worst_active_score
                    scratch.consolidation_selected[
                        worst_active
                    ] = false
                    scratch.consolidation_selected[
                        best_inactive
                    ] = true
                end
            end
        end
        for relation in 1:model.fanout
            old_hard = cache.gate_hard[node, relation]
            old_probability =
                Float64(cache.gate_probability[node, relation])
            old_derivative =
                Float64(cache.gate_derivative[node, relation])
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
            gate_probability_delta_sum +=
                Float64(cache.gate_probability[node, relation]) -
                old_probability
            gate_derivative_delta_sum +=
                Float64(cache.gate_derivative[node, relation]) -
                old_derivative
            changed = old_hard != cache.gate_hard[node, relation]
            flips += changed
            if changed && structural_mode === :utility
                trainer.optimizer.first_moment.gate_logits[
                    node,
                    relation,
                ] = 0.0f0
                trainer.optimizer.second_moment.gate_logits[
                    node,
                    relation,
                ] = 0.0f0
            end
        end
    end
    trainer.consolidation_flips[target] = flips
    trainer.consolidation_gate_probability_delta_sums[target] =
        gate_probability_delta_sum
    trainer.consolidation_gate_derivative_delta_sums[target] =
        gate_derivative_delta_sum
    trainer.consolidation_utility_swap_gap_sums[target] =
        utility_swap_gap_sum
    trainer.consolidation_utility_swap_gap_counts[target] =
        utility_swap_gap_count
    return nothing
end

function _update_synapse_utility!(executor::ArenaExecutor)
    executor.structural_learning_mode === :utility || return nothing
    shadow = executor.eprop_shadow
    shadow === nothing && error(
        "utility structure learning requires e-prop state",
    )
    shadow.config.third_factor_mode === :zero && return nothing
    shadow.config.edge_parameter_mode === :weight_gate_delay || error(
        "utility structure learning requires weight, gate, and delay gradients",
    )
    @inbounds for reducer in 1:executor.eprop_reducer_count
        worker_shadow = executor.workers[reducer].eprop_shadow
        worker_shadow === nothing && error(
            "utility reducer $reducer has no e-prop state",
        )
        worker_shadow.utility_evidence === nothing && error(
            "utility reducer $reducer has no candidate-local evidence",
        )
    end
    trainer = executor.trainer
    inverse_candidates =
        inv(Float64(max(trainer.arena.valid_count, 1)))
    decay = Float64(executor.utility_decay)
    @inbounds for index in eachindex(trainer.synapse_utility)
        trajectory_hard =
            shadow.trajectory_gate_hard[index] != 0x00
        current_hard =
            trainer.cache.gate_hard[index] != 0.0f0
        trajectory_hard == current_hard || error(
            "gate mask changed between trajectory replay and utility update",
        )
        responsibility = 0.0
        for reducer in 1:executor.eprop_reducer_count
            worker_shadow =
                executor.workers[reducer].eprop_shadow::EPropWorkerShadow
            utility_evidence =
                worker_shadow.utility_evidence::Matrix{Float64}
            responsibility += utility_evidence[index]
        end
        isfinite(responsibility) && responsibility >= 0.0 || error(
            "candidate-local utility evidence must be finite and nonnegative",
        )
        old_utility = Float64(trainer.synapse_utility[index])
        isfinite(old_utility) && old_utility >= 0.0 || error(
            "synapse utility must be finite and nonnegative",
        )
        trainer.synapse_utility[index] = Float32(muladd(
            decay,
            old_utility,
            responsibility * inverse_candidates,
        ))
    end
    trainer.utility_updates += 1
    return nothing
end

function _trajectory_gate_mask_flip_count(executor::ArenaExecutor)
    shadow = executor.eprop_shadow
    shadow === nothing && error(
        "trajectory gate-mask telemetry requires e-prop state",
    )
    cache = executor.trainer.cache
    flips = 0
    @inbounds for index in eachindex(shadow.trajectory_gate_hard)
        trajectory_active =
            shadow.trajectory_gate_hard[index] != 0x00
        current_active =
            cache.gate_hard[index] != 0.0f0
        flips += trajectory_active != current_active
    end
    return flips
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

function _prepare_routing_regularizer!(executor::ArenaExecutor)
    shadow = executor.eprop_shadow
    shadow === nothing && return nothing
    arena = executor.trainer.arena
    model = executor.trainer.model
    config = shadow.config
    fill!(shadow.route_load, 0.0f0)
    fill!(arena.route_regularizer_gradient, 0.0f0)
    config.routing_parameter_mode === :three_factor ||
        return nothing
    valid_count = arena.valid_count
    valid_count >= 1 || return nothing
    inverse_selection_count = inv(Float32(
        valid_count * model.workspace_k,
    ))
    @inbounds for target in 1:valid_count
        flat = Int(arena.valid_flats[target])
        for cycle in 1:model.cycles, block in 1:model.blocks
            shadow.route_load[block, cycle] = muladd(
                arena.block_mask[block, cycle, flat],
                inverse_selection_count,
                shadow.route_load[block, cycle],
            )
        end
    end
    entropy_weight = config.routing_entropy_weight
    load_weight = config.routing_load_weight
    entropy_weight == 0.0f0 && load_weight == 0.0f0 &&
        return nothing
    blocks_f = Float32(model.blocks)
    inverse_blocks = inv(blocks_f)
    inverse_valid = inv(Float32(valid_count))
    log_blocks = log(blocks_f)
    inverse_temperature = inv(model.route_temperature)
    @inbounds for target in 1:valid_count
        flat = Int(arena.valid_flats[target])
        for cycle in 1:model.cycles
            score_mean = 0.0f0
            for block in 1:model.blocks
                score_mean += arena.route_score[block, cycle, flat]
            end
            score_mean *= inverse_blocks
            score_square_sum = 0.0f0
            entropy = 0.0f0
            load_projection = 0.0f0
            for block in 1:model.blocks
                centered =
                    arena.route_score[block, cycle, flat] -
                    score_mean
                score_square_sum = muladd(
                    centered,
                    centered,
                    score_square_sum,
                )
                probability = arena.route_base_probability[
                    block,
                    cycle,
                    flat,
                ]
                entropy -= probability *
                    log(max(probability, 1.0f-12))
                load_projection = muladd(
                    probability,
                    shadow.route_load[block, cycle],
                    load_projection,
                )
            end
            score_inv_rms = inv(sqrt(
                score_square_sum * inverse_blocks +
                Model.RMS_NORM_EPS,
            ))
            normalized_entropy = entropy / log_blocks
            entropy_gap = max(
                config.routing_entropy_floor -
                normalized_entropy,
                0.0f0,
            )
            gradient_mean = 0.0f0
            gradient_score_projection_mean = 0.0f0
            for block in 1:model.blocks
                probability = arena.route_base_probability[
                    block,
                    cycle,
                    flat,
                ]
                log_probability =
                    log(max(probability, 1.0f-12))
                entropy_gradient =
                    2.0f0 *
                    entropy_weight *
                    entropy_gap *
                    probability *
                    (log_probability + entropy) /
                    log_blocks
                load_gradient =
                    load_weight *
                    blocks_f *
                    inverse_valid *
                    probability *
                    (
                        shadow.route_load[block, cycle] -
                        load_projection
                    )
                normalized_gradient =
                    (entropy_gradient + load_gradient) *
                    inverse_temperature
                arena.route_regularizer_gradient[
                    block,
                    cycle,
                    flat,
                ] = normalized_gradient
                standardized_score = (
                    arena.route_score[block, cycle, flat] -
                    score_mean
                ) * score_inv_rms
                gradient_mean += normalized_gradient
                gradient_score_projection_mean = muladd(
                    normalized_gradient,
                    standardized_score,
                    gradient_score_projection_mean,
                )
            end
            gradient_mean *= inverse_blocks
            gradient_score_projection_mean *= inverse_blocks
            for block in 1:model.blocks
                standardized_score = (
                    arena.route_score[block, cycle, flat] -
                    score_mean
                ) * score_inv_rms
                arena.route_regularizer_gradient[
                    block,
                    cycle,
                    flat,
                ] = score_inv_rms * (
                    arena.route_regularizer_gradient[
                        block,
                        cycle,
                        flat,
                    ] -
                    gradient_mean -
                    standardized_score *
                    gradient_score_projection_mean
                )
            end
        end
    end
    return nothing
end

function _prepare_eprop_shadow!(executor::ArenaExecutor)
    shadow = executor.eprop_shadow
    shadow === nothing && return nothing
    trainer = executor.trainer
    arena = trainer.arena
    @inbounds for index in eachindex(shadow.trajectory_gate_hard)
        hard = trainer.cache.gate_hard[index]
        parameter_hard =
            trainer.parameters.gate_logits[index] >= 0.0f0
        (hard != 0.0f0) == parameter_hard || error(
            "gate cache and parameter sign disagree before trajectory replay",
        )
        shadow.trajectory_gate_hard[index] =
            hard == 0.0f0 ? 0x00 : 0x01
    end
    fill!(shadow.signal_flat, Int32(0))
    mode = shadow.config.third_factor_mode
    @inbounds for state_slot in 1:arena.state_batch
        count = Int(arena.counts[state_slot])
        offset = (state_slot - 1) * arena.width
        for candidate in 1:count
            signal_candidate = mode === :candidate_shuffle ?
                mod1(candidate + 1, count) : candidate
            flat = offset + candidate
            shadow.signal_flat[flat] =
                Int32(offset + signal_candidate)
        end
    end
    _prepare_routing_regularizer!(executor)
    fill!(shadow.gradient, 0.0f0)
    if shadow.config.edge_parameter_mode === :weight_gate_delay
        fill!(shadow.gate_gradient, 0.0f0)
        fill!(shadow.delay_gradient, 0.0f0)
    end
    if shadow.config.node_parameter_mode !== :none
        fill!(shadow.leak_gradient, 0.0f0)
        fill!(shadow.threshold_gradient, 0.0f0)
        fill!(shadow.feedback_gradient, 0.0f0)
    end
    if shadow.config.node_parameter_mode === :full_state
        fill!(shadow.input_gain_gradient, 0.0f0)
        fill!(shadow.input_bias_gradient, 0.0f0)
        fill!(shadow.workspace_decay_gradient, 0.0f0)
    end
    if shadow.config.routing_parameter_mode !== :none
        fill!(shadow.workspace_key_gradient, 0.0f0)
        fill!(shadow.query_weight_gradient, 0.0f0)
    end
    @inbounds for worker in executor.workers
        worker_shadow = worker.eprop_shadow
        worker_shadow === nothing && error(
            "e-prop shadow executor has a worker without shadow storage",
        )
        worker_shadow.eligibility_square_sum = 0.0
        worker_shadow.eligibility_count = Int64(0)
        worker_shadow.membrane_margin_sum = 0.0
        worker_shadow.membrane_margin_square_sum = 0.0
        worker_shadow.surrogate_sensitivity_sum = 0.0
        worker_shadow.surrogate_sensitivity_square_sum = 0.0
        worker_shadow.membrane_sample_count = Int64(0)
        worker_shadow.local_q_loss_sum = 0.0
        worker_shadow.local_q_loss_count = Int64(0)
        worker_shadow.local_death_loss_sum = 0.0
        worker_shadow.local_death_loss_count = Int64(0)
        worker_shadow.local_quantile_loss_sum = 0.0
        worker_shadow.local_quantile_loss_count = Int64(0)
        worker_shadow.local_geometry_loss_sum = 0.0
        worker_shadow.local_geometry_loss_count = Int64(0)
        fill!(worker_shadow.gradient, 0.0f0)
        if shadow.config.edge_parameter_mode === :weight_gate_delay
            fill!(worker_shadow.gate_gradient, 0.0f0)
            fill!(worker_shadow.delay_gradient, 0.0f0)
            if worker_shadow.utility_evidence !== nothing
                fill!(worker_shadow.utility_evidence, 0.0)
            end
        end
        if shadow.config.node_parameter_mode !== :none
            fill!(worker_shadow.leak_gradient, 0.0f0)
            fill!(worker_shadow.threshold_gradient, 0.0f0)
            fill!(worker_shadow.feedback_gradient, 0.0f0)
        end
        if shadow.config.node_parameter_mode === :full_state
            fill!(worker_shadow.input_gain_gradient, 0.0f0)
            fill!(worker_shadow.input_bias_gradient, 0.0f0)
            fill!(worker_shadow.workspace_decay_gradient, 0.0f0)
        end
        if shadow.config.routing_parameter_mode !== :none
            fill!(worker_shadow.workspace_key_gradient, 0.0f0)
            fill!(worker_shadow.query_weight_gradient, 0.0f0)
        end
    end
    return nothing
end

function _aggregate_eprop_parameter!(
    local_gradient,
    executor::ArenaExecutor,
    worker_field::Symbol,
    exact,
    reference_available::Bool,
)
    local_gradient === nothing &&
        return _disabled_eprop_parameter_report()
    local_square = 0.0
    exact_square = 0.0
    dot = 0.0
    local_nonzero = 0
    exact_nonzero = 0
    @inbounds for index in eachindex(local_gradient)
        local_value = 0.0f0
        for reducer in 1:executor.eprop_reducer_count
            worker = executor.workers[reducer]
            worker_gradient =
                getproperty(worker.eprop_shadow, worker_field)
            local_value += worker_gradient[index]
        end
        local_gradient[index] = local_value
        exact_value = reference_available ? exact[index] : 0.0f0
        local64 = Float64(local_value)
        exact64 = Float64(exact_value)
        local_square = muladd(local64, local64, local_square)
        exact_square = muladd(exact64, exact64, exact_square)
        dot = muladd(local64, exact64, dot)
        local_nonzero += abs(local_value) > 1.0f-12
        exact_nonzero += abs(exact_value) > 1.0f-12
    end
    local_norm = sqrt(local_square)
    exact_norm = reference_available ? sqrt(exact_square) : NaN
    cosine = !reference_available ||
             local_norm == 0.0 ||
             exact_norm == 0.0 ?
        NaN : dot / (local_norm * exact_norm)
    return EPropParameterReport(
        true,
        local_norm,
        exact_norm,
        dot,
        cosine,
        local_nonzero / length(local_gradient),
        reference_available ? exact_nonzero / length(exact) : NaN,
    )
end

function _finalize_eprop_shadow!(
    executor::ArenaExecutor,
    wall_seconds::Float64,
)
    shadow = executor.eprop_shadow
    shadow === nothing && return nothing
    reference_available =
        executor.synapse_learning_mode === :vjp
    synapse_report = _aggregate_eprop_parameter!(
        shadow.gradient,
        executor,
        :gradient,
        executor.trainer.gradient.synapse_weight,
        reference_available,
    )
    full_state =
        shadow.config.node_parameter_mode === :full_state
    input_gain_report = _aggregate_eprop_parameter!(
        full_state ? shadow.input_gain_gradient : nothing,
        executor,
        :input_gain_gradient,
        executor.trainer.gradient.input_gain,
        reference_available,
    )
    input_bias_report = _aggregate_eprop_parameter!(
        full_state ? shadow.input_bias_gradient : nothing,
        executor,
        :input_bias_gradient,
        executor.trainer.gradient.input_bias,
        reference_available,
    )
    edge_all =
        shadow.config.edge_parameter_mode === :weight_gate_delay
    gate_report = _aggregate_eprop_parameter!(
        edge_all ? shadow.gate_gradient : nothing,
        executor,
        :gate_gradient,
        executor.trainer.gradient.gate_logits,
        reference_available,
    )
    delay_report = _aggregate_eprop_parameter!(
        edge_all ? shadow.delay_gradient : nothing,
        executor,
        :delay_gradient,
        executor.trainer.gradient.delay_logits,
        reference_available,
    )
    node_all =
        shadow.config.node_parameter_mode !== :none
    leak_report = _aggregate_eprop_parameter!(
        node_all ? shadow.leak_gradient : nothing,
        executor,
        :leak_gradient,
        executor.trainer.gradient.leak_logits,
        reference_available,
    )
    threshold_report = _aggregate_eprop_parameter!(
        node_all ? shadow.threshold_gradient : nothing,
        executor,
        :threshold_gradient,
        executor.trainer.gradient.threshold_logits,
        reference_available,
    )
    feedback_report = _aggregate_eprop_parameter!(
        node_all ? shadow.feedback_gradient : nothing,
        executor,
        :feedback_gradient,
        executor.trainer.gradient.feedback_gain,
        reference_available,
    )
    routing_all =
        shadow.config.routing_parameter_mode !== :none
    workspace_key_report = _aggregate_eprop_parameter!(
        routing_all ? shadow.workspace_key_gradient : nothing,
        executor,
        :workspace_key_gradient,
        executor.trainer.gradient.workspace_key,
        reference_available,
    )
    query_weight_report = _aggregate_eprop_parameter!(
        routing_all ? shadow.query_weight_gradient : nothing,
        executor,
        :query_weight_gradient,
        executor.trainer.gradient.query_weight,
        reference_available,
    )
    workspace_decay_report = _aggregate_eprop_parameter!(
        full_state ? shadow.workspace_decay_gradient : nothing,
        executor,
        :workspace_decay_gradient,
        executor.trainer.gradient.workspace_decay_logit,
        reference_available,
    )
    trace_bytes = 0
    gradient_bytes = 0
    signal_bytes = 0
    @inbounds for worker in executor.workers
        trace_bytes += (
            length(worker.eprop_shadow.post_sensitivity) +
            length(worker.eprop_shadow.state_recurrence)
        ) * sizeof(Float32)
        gradient_bytes +=
            length(worker.eprop_shadow.gradient) * sizeof(Float32)
        if worker.eprop_shadow.input_gain_gradient !== nothing
            gradient_bytes += (
                length(worker.eprop_shadow.input_gain_gradient) +
                length(worker.eprop_shadow.input_bias_gradient) +
                length(worker.eprop_shadow.workspace_decay_gradient)
            ) * sizeof(Float32)
        end
        if worker.eprop_shadow.gate_gradient !== nothing
            gradient_bytes += (
                length(worker.eprop_shadow.gate_gradient) +
                length(worker.eprop_shadow.delay_gradient)
            ) * sizeof(Float32)
        end
        if worker.eprop_shadow.leak_gradient !== nothing
            gradient_bytes += (
                length(worker.eprop_shadow.leak_gradient) +
                length(worker.eprop_shadow.threshold_gradient) +
                length(worker.eprop_shadow.feedback_gradient)
            ) * sizeof(Float32)
        end
        if worker.eprop_shadow.workspace_key_gradient !== nothing
            gradient_bytes += (
                length(worker.eprop_shadow.workspace_key_gradient) +
                length(worker.eprop_shadow.query_weight_gradient)
            ) * sizeof(Float32)
        end
        signal_bytes += (
            length(worker.eprop_shadow.node_signal) +
            length(worker.eprop_shadow.hidden_signal) +
            length(worker.eprop_shadow.route_query_signal) +
            length(worker.eprop_shadow.route_workspace_signal) +
            length(worker.eprop_shadow.route_block_signal) +
            length(worker.eprop_shadow.route_block_soft) +
            length(worker.eprop_shadow.route_state_signal) +
            length(worker.eprop_shadow.route_score_signal) +
            length(worker.eprop_shadow.route_write_signal)
        ) * sizeof(Float32)
    end
    local_q_sum = 0.0
    local_q_count = Int64(0)
    local_death_sum = 0.0
    local_death_count = Int64(0)
    local_quantile_sum = 0.0
    local_quantile_count = Int64(0)
    local_geometry_sum = 0.0
    local_geometry_count = Int64(0)
    @inbounds for worker in executor.workers
        worker_shadow = worker.eprop_shadow::EPropWorkerShadow
        local_q_sum += worker_shadow.local_q_loss_sum
        local_q_count += worker_shadow.local_q_loss_count
        local_death_sum += worker_shadow.local_death_loss_sum
        local_death_count += worker_shadow.local_death_loss_count
        local_quantile_sum += worker_shadow.local_quantile_loss_sum
        local_quantile_count +=
            worker_shadow.local_quantile_loss_count
        local_geometry_sum += worker_shadow.local_geometry_loss_sum
        local_geometry_count +=
            worker_shadow.local_geometry_loss_count
    end
    local_q_loss =
        local_q_count == 0 ? NaN : local_q_sum / local_q_count
    local_death_loss =
        local_death_count == 0 ? NaN :
        local_death_sum / local_death_count
    local_quantile_loss =
        local_quantile_count == 0 ? NaN :
        local_quantile_sum / local_quantile_count
    local_geometry_loss =
        local_geometry_count == 0 ? NaN :
        local_geometry_sum / local_geometry_count
    report = EPropShadowReport(
        shadow.config.feedback_mode,
        shadow.config.eligibility_mode,
        shadow.config.third_factor_mode,
        shadow.config.time_order,
        reference_available,
        executor.synapse_learning_mode === :local_eligibility,
        executor.trainer.arena.valid_count,
        wall_seconds,
        synapse_report.local_gradient_norm,
        synapse_report.full_vjp_gradient_norm,
        synapse_report.dot_with_full_vjp,
        synapse_report.cosine_with_full_vjp,
        synapse_report.local_nonzero_fraction,
        synapse_report.full_vjp_nonzero_fraction,
        trace_bytes,
        gradient_bytes,
        signal_bytes,
        (
            length(shadow.q_feedback) +
            length(shadow.block_projection)
        ) * sizeof(Float32),
        local_q_loss,
        local_death_loss,
        local_quantile_loss,
        local_geometry_loss,
        (;
            synapse_weight=synapse_report,
            input_gain=input_gain_report,
            input_bias=input_bias_report,
            gate_logits=gate_report,
            delay_logits=delay_report,
            leak_logits=leak_report,
            threshold_logits=threshold_report,
            feedback_gain=feedback_report,
            workspace_key=workspace_key_report,
            query_weight=query_weight_report,
            workspace_decay_logit=workspace_decay_report,
        ),
    )
    shadow.last_report = report
    return report
end

arena_shadow_report(executor::ArenaExecutor) =
    executor.eprop_shadow === nothing ? nothing :
    executor.eprop_shadow.last_report

function _reduce_worker_gradients!(
    trainer::ArenaTrainer,
    executor::ArenaExecutor,
)
    _fill_parameter_tree!(trainer.gradient)
    @inbounds for name in PARAMETER_FIELDS
        destination = getproperty(trainer.gradient, name)
        if executor.synapse_learning_mode === :local_eligibility &&
           name !== :head_weight &&
           name !== :head_bias &&
           name !== :output_weight &&
           name !== :output_bias
            continue
        end
        for index in eachindex(destination)
            value = 0.0f0
            for worker in executor.workers
                value += getproperty(worker.gradient, name)[index]
            end
            destination[index] = value
        end
    end
    if executor.synapse_learning_mode === :vjp
        coefficient = trainer.structure_gradient_coefficient
        @inbounds for index in eachindex(trainer.gradient.gate_logits)
            trainer.gradient.gate_logits[index] +=
                coefficient * trainer.cache.gate_derivative[index]
        end
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
        routing_nonce = executor.stochastic_routing ?
            _routing_mix64(
                executor.routing_seed ⊻
                UInt64(trainer.optimizer.step + 1) *
                0x9e3779b97f4a7c15 ⊻
                UInt64(flat) * 0xd1b54a32d192ed03,
            ) : UInt64(0)
        forward_candidate!(
            trainer.arena,
            trainer.model,
            trainer.parameters,
            trainer.cache,
            worker.scratch,
            flat,
            routing_nonce,
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
    elseif work.kind == UInt8(ARENA_HEAD_BACKWARD)
        flat = Int(trainer.arena.valid_flats[target])
        backward_head_candidate!(
            worker.gradient,
            trainer.arena,
            trainer.model,
            trainer.parameters,
            trainer.cache,
            worker.scratch,
            flat,
        )
    elseif work.kind == UInt8(ARENA_EPROP_SHADOW)
        shadow = executor.eprop_shadow
        shadow === nothing && error("e-prop shadow work has no shadow state")
        worker.eprop_shadow === nothing && error(
            "e-prop shadow work has no worker trace",
        )
        accumulator = executor.workers[target].eprop_shadow
        accumulator === nothing && error(
            "e-prop reducer has no gradient storage",
        )
        ordinal = target
        while ordinal <= trainer.arena.valid_count
            flat = Int(trainer.arena.valid_flats[ordinal])
            shadow_eprop_candidate!(
                worker.eprop_shadow,
                accumulator,
                shadow,
                trainer.arena,
                trainer.model,
                trainer.parameters,
                trainer.cache,
                flat,
            )
            ordinal += executor.eprop_reducer_count
        end
    elseif work.kind == UInt8(ARENA_EPROP_SIGNAL)
        shadow = executor.eprop_shadow
        shadow === nothing && error("e-prop signal work has no shadow state")
        shadow.config.feedback_mode === :block_local || error(
            "e-prop signal phase is only valid for block-local learning",
        )
        worker.eprop_shadow === nothing && error(
            "e-prop signal work has no worker scratch",
        )
        flat = Int(trainer.arena.valid_flats[target])
        _prepare_block_learning_signal_candidate!(
            worker.eprop_shadow,
            shadow,
            trainer.arena,
            trainer.model,
            trainer.cache,
            flat,
        )
    elseif work.kind == UInt8(ARENA_OPTIMIZER)
        _optimizer_shard!(trainer, executor, target)
    elseif work.kind == UInt8(ARENA_CONSOLIDATE)
        _consolidate_node_range!(
            trainer,
            executor,
            worker.scratch,
            target,
        )
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
    shadow_seconds = 0.0
    if executor.eprop_shadow !== nothing
        _prepare_eprop_shadow!(executor)
        signal_started = time_ns()
        if executor.eprop_shadow.config.feedback_mode === :block_local
            _run_phase!(
                executor,
                ARENA_EPROP_SIGNAL,
                trainer.arena.valid_count,
                generation,
            )
            _center_block_supervised_rewards!(executor)
        end
        signal_seconds = (time_ns() - signal_started) * 1.0e-9
        replay_seconds = _run_phase!(
            executor,
            ARENA_EPROP_SHADOW,
            executor.eprop_reducer_count,
            generation,
        )
        shadow_seconds = signal_seconds + replay_seconds
    end
    backward_kind =
        executor.synapse_learning_mode === :local_eligibility ?
        ARENA_HEAD_BACKWARD : ARENA_BACKWARD
    backward_seconds = _run_phase!(
        executor,
        backward_kind,
        trainer.arena.valid_count,
        generation,
    )
    return (;
        pack_seconds,
        pack_parallel_seconds,
        forward_seconds,
        loss_seconds,
        shadow_seconds,
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
    shadow = _finalize_eprop_shadow!(
        executor,
        phases.shadow_seconds,
    )
    if executor.synapse_learning_mode === :local_eligibility
        copyto!(
            trainer.gradient.synapse_weight,
            executor.eprop_shadow.gradient,
        )
        if executor.eprop_shadow.config.edge_parameter_mode ===
           :weight_gate_delay
            copyto!(
                trainer.gradient.gate_logits,
                executor.eprop_shadow.gate_gradient,
            )
            copyto!(
                trainer.gradient.delay_logits,
                executor.eprop_shadow.delay_gradient,
            )
        end
        if executor.eprop_shadow.config.node_parameter_mode !== :none
            copyto!(
                trainer.gradient.feedback_gain,
                executor.eprop_shadow.feedback_gradient,
            )
            copyto!(
                trainer.gradient.leak_logits,
                executor.eprop_shadow.leak_gradient,
            )
            copyto!(
                trainer.gradient.threshold_logits,
                executor.eprop_shadow.threshold_gradient,
            )
        end
        if executor.eprop_shadow.config.node_parameter_mode === :full_state
            copyto!(
                trainer.gradient.input_gain,
                executor.eprop_shadow.input_gain_gradient,
            )
            copyto!(
                trainer.gradient.input_bias,
                executor.eprop_shadow.input_bias_gradient,
            )
            copyto!(
                trainer.gradient.workspace_decay_logit,
                executor.eprop_shadow.workspace_decay_gradient,
            )
        end
        if executor.eprop_shadow.config.routing_parameter_mode !== :none
            copyto!(
                trainer.gradient.workspace_key,
                executor.eprop_shadow.workspace_key_gradient,
            )
            copyto!(
                trainer.gradient.query_weight,
                executor.eprop_shadow.query_weight_gradient,
            )
        end
    end
    gate_trainable =
        executor.synapse_learning_mode === :local_eligibility &&
        (
            executor.eprop_shadow !== nothing &&
            executor.eprop_shadow.config.edge_parameter_mode ===
                :weight_gate_delay &&
            executor.eprop_shadow.config.third_factor_mode !== :zero
        )
    if gate_trainable
        @inbounds for index in eachindex(trainer.gradient.gate_logits)
            trainer.gradient.gate_logits[index] +=
                trainer.structure_gradient_coefficient *
                trainer.cache.gate_derivative[index]
        end
    end
    trainer.last_gradient_norm = arena_tree_norm(trainer.gradient)
    return (;
        loss=trainer.last_loss,
        gradient,
        raw=trainer.arena.raw,
        raw_gradient=trainer.arena.raw_gradient,
        phases,
        shadow,
    )
end

@inline function _snapshot_update_gate_mask!(trainer::ArenaTrainer)
    @inbounds for index in eachindex(trainer.mask_before_update)
        trainer.mask_before_update[index] =
            trainer.cache.gate_hard[index] == 0.0f0 ? 0x00 : 0x01
    end
    return nothing
end

@inline function _same_route_mask(
    arena::TrainingArena,
    model,
    left_decision::Int,
    right_decision::Int,
)
    left_target = div(left_decision - 1, model.cycles) + 1
    left_cycle =
        left_decision - (left_target - 1) * model.cycles
    right_target = div(right_decision - 1, model.cycles) + 1
    right_cycle =
        right_decision - (right_target - 1) * model.cycles
    left_flat = Int(@inbounds arena.valid_flats[left_target])
    right_flat = Int(@inbounds arena.valid_flats[right_target])
    @inbounds for block in 1:model.blocks
        arena.block_mask[block, left_cycle, left_flat] ==
            arena.block_mask[block, right_cycle, right_flat] ||
            return false
    end
    return true
end

function _refresh_forward_telemetry!(executor::ArenaExecutor)
    trainer = executor.trainer
    metrics = trainer.metrics
    arena = trainer.arena
    model = trainer.model
    decisions = arena.valid_count * model.cycles
    decisions >= 1 || error("forward telemetry has no routing decisions")
    fill!(trainer.route_mask_index_table, Int32(0))
    route_gap_sum = 0.0
    route_score_square_sum = 0.0
    cycle_churn_count = 0
    entropy_violations = 0
    unique_masks = 0
    entropy_floor = executor.eprop_shadow === nothing ?
        0.0f0 : executor.eprop_shadow.config.routing_entropy_floor
    hash_table_length = length(trainer.route_mask_index_table)
    hash_table_mask = UInt64(hash_table_length - 1)
    ispow2(hash_table_length) || error(
        "route-mask telemetry table must have power-of-two capacity",
    )
    @inbounds for target in 1:arena.valid_count
        flat = Int(arena.valid_flats[target])
        for cycle in 1:model.cycles
            route_gap_sum +=
                Float64(arena.route_selection_gap_value[cycle, flat])
            route_score_square_sum +=
                arena.route_score_square_sum[cycle, flat]
            cycle_churn_count +=
                Int(arena.route_cycle_churn_count[cycle, flat])
            if executor.eprop_shadow !== nothing &&
               arena.route_normalized_entropy[cycle, flat] < entropy_floor
                entropy_violations += 1
            end
            decision = (target - 1) * model.cycles + cycle
            fingerprint =
                arena.route_mask_fingerprint[cycle, flat]
            slot = Int(fingerprint & hash_table_mask) + 1
            duplicate = false
            probes = 0
            while true
                previous = Int(trainer.route_mask_index_table[slot])
                if previous == 0
                    trainer.route_mask_hash_table[slot] = fingerprint
                    trainer.route_mask_index_table[slot] =
                        Int32(decision)
                    unique_masks += 1
                    break
                end
                if trainer.route_mask_hash_table[slot] == fingerprint &&
                   _same_route_mask(arena, model, decision, previous)
                    duplicate = true
                    break
                end
                slot = slot == hash_table_length ? 1 : slot + 1
                probes += 1
                probes < hash_table_length || error(
                    "route-mask telemetry table is full",
                )
            end
            duplicate && continue
        end
    end
    decision_denominator = Float64(decisions)
    metrics.route_selection_gap =
        route_gap_sum / decision_denominator
    metrics.route_score_rms = sqrt(
        route_score_square_sum /
        Float64(decisions * model.blocks),
    )
    metrics.hard_mask_unique_fraction =
        Float64(unique_masks) / decision_denominator
    churn_denominator =
        arena.valid_count * max(model.cycles - 1, 0) * model.blocks
    metrics.hard_mask_cycle_churn =
        churn_denominator == 0 ? 0.0 :
        Float64(cycle_churn_count) / Float64(churn_denominator)
    metrics.entropy_floor_violation_fraction =
        executor.eprop_shadow === nothing ? 0.0 :
        Float64(entropy_violations) / decision_denominator

    if executor.eprop_shadow === nothing
        metrics.membrane_threshold_margin_mean = 0.0
        metrics.membrane_threshold_margin_rms = 0.0
        metrics.surrogate_sensitivity_mean = 0.0
        metrics.surrogate_sensitivity_rms = 0.0
        metrics.eligibility_rms = 0.0
    else
        eligibility_square_sum = 0.0
        eligibility_count = Int64(0)
        margin_sum = 0.0
        margin_square_sum = 0.0
        surrogate_sum = 0.0
        surrogate_square_sum = 0.0
        membrane_count = Int64(0)
        @inbounds for reducer in 1:executor.eprop_reducer_count
            worker_shadow =
                executor.workers[reducer].eprop_shadow::EPropWorkerShadow
            eligibility_square_sum +=
                worker_shadow.eligibility_square_sum
            eligibility_count += worker_shadow.eligibility_count
            margin_sum += worker_shadow.membrane_margin_sum
            margin_square_sum +=
                worker_shadow.membrane_margin_square_sum
            surrogate_sum +=
                worker_shadow.surrogate_sensitivity_sum
            surrogate_square_sum +=
                worker_shadow.surrogate_sensitivity_square_sum
            membrane_count += worker_shadow.membrane_sample_count
        end
        membrane_count >= 1 || error(
            "e-prop membrane telemetry has no samples",
        )
        eligibility_count >= 1 || error(
            "e-prop eligibility telemetry has no samples",
        )
        inverse_membrane_count = inv(Float64(membrane_count))
        metrics.membrane_threshold_margin_mean =
            margin_sum * inverse_membrane_count
        metrics.membrane_threshold_margin_rms =
            sqrt(margin_square_sum * inverse_membrane_count)
        metrics.surrogate_sensitivity_mean =
            surrogate_sum * inverse_membrane_count
        metrics.surrogate_sensitivity_rms =
            sqrt(surrogate_square_sum * inverse_membrane_count)
        metrics.eligibility_rms =
            sqrt(eligibility_square_sum / Float64(eligibility_count))
    end
    if executor.eprop_shadow === nothing ||
       executor.eprop_shadow.config.feedback_mode !== :block_local
        metrics.local_q_loss = 0.0
        metrics.local_death_loss = 0.0
        metrics.local_quantile_loss = 0.0
        metrics.local_geometry_loss = 0.0
    else
        local_q_sum = 0.0
        local_q_count = Int64(0)
        local_death_sum = 0.0
        local_death_count = Int64(0)
        local_quantile_sum = 0.0
        local_quantile_count = Int64(0)
        local_geometry_sum = 0.0
        local_geometry_count = Int64(0)
        @inbounds for worker in executor.workers
            worker_shadow =
                worker.eprop_shadow::EPropWorkerShadow
            local_q_sum += worker_shadow.local_q_loss_sum
            local_q_count += worker_shadow.local_q_loss_count
            local_death_sum += worker_shadow.local_death_loss_sum
            local_death_count +=
                worker_shadow.local_death_loss_count
            local_quantile_sum +=
                worker_shadow.local_quantile_loss_sum
            local_quantile_count +=
                worker_shadow.local_quantile_loss_count
            local_geometry_sum +=
                worker_shadow.local_geometry_loss_sum
            local_geometry_count +=
                worker_shadow.local_geometry_loss_count
        end
        metrics.local_q_loss =
            local_q_count == 0 ? NaN :
            local_q_sum / Float64(local_q_count)
        metrics.local_death_loss =
            local_death_count == 0 ? NaN :
            local_death_sum / Float64(local_death_count)
        metrics.local_quantile_loss =
            local_quantile_count == 0 ? NaN :
            local_quantile_sum / Float64(local_quantile_count)
        metrics.local_geometry_loss =
            local_geometry_count == 0 ? NaN :
            local_geometry_sum / Float64(local_geometry_count)
    end
    return nothing
end

function _refresh_transformed_parameter_telemetry!(
    trainer::ArenaTrainer,
)
    gate_value_sum = 0.0
    gate_derivative_sum = 0.0
    delay_value_sum = 0.0
    delay_derivative_sum = 0.0
    leak_value_sum = 0.0
    leak_derivative_sum = 0.0
    threshold_value_sum = 0.0
    threshold_derivative_sum = 0.0
    workspace_decay_sum = 0.0
    workspace_decay_derivative_sum = 0.0
    @inbounds for target in eachindex(trainer.parameter_shards)
        field = Int(trainer.parameter_shards[target].field)
        value_sum = trainer.transformed_value_sums[target]
        derivative_sum =
            trainer.transformed_derivative_sums[target]
        if field == 6
            leak_value_sum += value_sum
            leak_derivative_sum += derivative_sum
        elseif field == 7
            threshold_value_sum += value_sum
            threshold_derivative_sum += derivative_sum
        elseif field == 9
            gate_value_sum += value_sum
            gate_derivative_sum += derivative_sum
        elseif field == 10
            delay_value_sum += value_sum
            delay_derivative_sum += derivative_sum
        elseif field == 11
            workspace_decay_sum += value_sum
            workspace_decay_derivative_sum += derivative_sum
        end
    end
    @inbounds for target in eachindex(
        trainer.consolidation_gate_probability_delta_sums,
    )
        gate_value_sum +=
            trainer.consolidation_gate_probability_delta_sums[target]
        gate_derivative_sum +=
            trainer.consolidation_gate_derivative_delta_sums[target]
    end
    metrics = trainer.metrics
    metrics.gate_probability_mean =
        gate_value_sum / Float64(length(trainer.cache.gate_probability))
    metrics.gate_derivative_mean =
        gate_derivative_sum / Float64(length(trainer.cache.gate_derivative))
    metrics.delay_mean =
        delay_value_sum / Float64(length(trainer.cache.delay))
    metrics.delay_derivative_mean =
        delay_derivative_sum / Float64(length(trainer.cache.delay_derivative))
    metrics.leak_mean =
        leak_value_sum / Float64(length(trainer.cache.leak))
    metrics.leak_derivative_mean =
        leak_derivative_sum / Float64(length(trainer.cache.leak_derivative))
    metrics.threshold_mean =
        threshold_value_sum / Float64(length(trainer.cache.threshold))
    metrics.threshold_derivative_mean =
        threshold_derivative_sum /
        Float64(length(trainer.cache.threshold_derivative))
    metrics.workspace_decay = workspace_decay_sum
    metrics.workspace_decay_derivative =
        workspace_decay_derivative_sum
    return nothing
end

@inline function _net_mask_flip_count(trainer::ArenaTrainer)
    flips = 0
    @inbounds for index in eachindex(trainer.mask_before_update)
        current =
            trainer.cache.gate_hard[index] == 0.0f0 ? 0x00 : 0x01
        flips += current != trainer.mask_before_update[index]
    end
    return flips
end

function _refresh_structure_telemetry!(trainer::ArenaTrainer)
    gap_sum = 0.0
    gap_count = 0
    @inbounds for target in eachindex(
        trainer.consolidation_utility_swap_gap_sums,
    )
        gap_sum +=
            trainer.consolidation_utility_swap_gap_sums[target]
        gap_count +=
            trainer.consolidation_utility_swap_gap_counts[target]
    end
    trainer.metrics.utility_swap_gap =
        trainer.metrics.consolidation_actual && gap_count > 0 ?
        gap_sum / Float64(gap_count) : 0.0
    trainer.metrics.net_mask_flips = _net_mask_flip_count(trainer)
    return nothing
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
    metrics = trainer.metrics
    _snapshot_update_gate_mask!(trainer)
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
    if executor.synapse_learning_mode === :vjp
        _finalize_eprop_shadow!(
            executor,
            phases.shadow_seconds,
        )
    end
    _update_synapse_utility!(executor)

    consolidation_seconds = 0.0
    recurrent_signal_enabled =
        executor.eprop_shadow === nothing ||
        executor.eprop_shadow.config.third_factor_mode !== :zero
    consolidation_scheduled =
        recurrent_signal_enabled &&
        trainer.optimizer.step % structural_interval == 0
    metrics.consolidation_scheduled = consolidation_scheduled
    metrics.consolidation_actual = false
    metrics.utility_swap_gap = 0.0
    fill!(
        trainer.consolidation_gate_probability_delta_sums,
        0.0,
    )
    fill!(
        trainer.consolidation_gate_derivative_delta_sums,
        0.0,
    )
    fill!(
        trainer.consolidation_utility_swap_gap_sums,
        0.0,
    )
    fill!(
        trainer.consolidation_utility_swap_gap_counts,
        0,
    )
    if executor.structural_learning_mode === :frozen
        fill!(trainer.consolidation_flips, 0)
    elseif consolidation_scheduled
        if executor.structural_learning_mode === :utility
            executor.consolidation_event_ordinal =
                div(trainer.optimizer.step, structural_interval)
        end
        metrics.consolidation_actual = true
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
        if executor.structural_learning_mode === :utility
            observed_flips =
                _trajectory_gate_mask_flip_count(executor)
            flips == observed_flips || error(
                "structural flip telemetry missed a gate-mask change",
            )
        end
        trainer.total_structural_flips += flips
    else
        fill!(trainer.consolidation_flips, 0)
        if executor.structural_learning_mode === :utility
            _trajectory_gate_mask_flip_count(executor) == 0 || error(
                "gate mask changed outside utility consolidation",
            )
        end
    end

    wall_seconds = (time_ns() - wall_started) * 1.0e-9
    cpu_seconds =
        (CpuSets.process_cpu_ticks_100ns() - cpu_started) * 1.0e-7
    gc_difference = Base.GC_Diff(Base.gc_num(), gc_started)
    metrics.wall_seconds = wall_seconds
    metrics.cpu_seconds = cpu_seconds
    metrics.allocation_bytes = Int128(gc_difference.allocd)
    metrics.gc_seconds = Float64(gc_difference.total_time) * 1.0e-9
    metrics.pack_seconds = phases.pack_seconds
    metrics.forward_seconds = phases.forward_seconds
    metrics.loss_seconds = phases.loss_seconds
    metrics.shadow_seconds = phases.shadow_seconds
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
    # Scalar aggregation and exact hash-set accounting are intentionally
    # outside the measured allocation-free update interval. Their primitive
    # observations were recorded in preallocated storage by the relevant
    # forward/e-prop/optimizer/consolidation phases.
    _refresh_forward_telemetry!(executor)
    _refresh_structure_telemetry!(trainer)
    _refresh_transformed_parameter_telemetry!(trainer)
    isfinite(trainer.last_loss.composite_loss) || error(
        "non-finite arena loss",
    )
    isfinite(trainer.last_gradient_norm) || error(
        "non-finite arena gradient",
    )
    return trainer
end

function _verify_arena_team_teardown!(executor::ArenaExecutor)
    violations = String[]
    executor.booted_workers[] == executor.julia_workers || push!(
        violations,
        "booted_workers=$(executor.booted_workers[]) expected " *
        "$(executor.julia_workers)",
    )
    executor.ready_workers[] == executor.active_workers || push!(
        violations,
        "ready_workers=$(executor.ready_workers[]) expected " *
        "$(executor.active_workers)",
    )
    executor.shutdown_requested[] == UInt32(1) || push!(
        violations,
        "shutdown_requested=$(executor.shutdown_requested[]) expected 1",
    )
    Queue.isclosed(executor.queue) || push!(
        violations,
        "arena queue is not closed",
    )
    queue_length = Queue.approx_length(executor.queue)
    iszero(queue_length) || push!(
        violations,
        "arena queue retains $queue_length work items",
    )
    executor.remaining[] == 0 || push!(
        violations,
        "remaining=$(executor.remaining[]) expected 0",
    )
    length(executor.bindings) == executor.julia_workers || push!(
        violations,
        "binding evidence has $(length(executor.bindings)) slots, expected " *
        "$(executor.julia_workers)",
    )
    length(executor.bindings_released) == executor.julia_workers || push!(
        violations,
        "binding release evidence has " *
        "$(length(executor.bindings_released)) slots, expected " *
        "$(executor.julia_workers)",
    )
    for worker_slot in 1:min(
        length(executor.bindings),
        executor.julia_workers,
    )
        binding = executor.bindings[worker_slot]
        if binding === nothing
            push!(
                violations,
                "worker slot $worker_slot has no startup binding evidence",
            )
            continue
        end
        hasproperty(binding, :worker_slot) &&
            binding.worker_slot == worker_slot || push!(
                violations,
                "worker slot $worker_slot has mismatched binding evidence",
            )
        hasproperty(binding, :julia_thread_id) &&
            binding.julia_thread_id == worker_slot || push!(
                violations,
                "worker slot $worker_slot has mismatched Julia thread evidence",
            )
        hasproperty(binding, :verified) && binding.verified || push!(
            violations,
            "worker slot $worker_slot startup binding was not verified",
        )
    end
    for worker_slot in 1:min(
        length(executor.bindings_released),
        executor.julia_workers,
    )
        executor.bindings_released[worker_slot] || push!(
            violations,
            "worker slot $worker_slot did not release its CPU Set binding",
        )
    end
    isempty(violations) || error(
        "arena team teardown invariant failed: " *
        join(violations, "; "),
    )
    return nothing
end

function _acquire_arena_team_guard!()
    observed = Base.Threads.atomic_cas!(
        ARENA_TEAM_ACTIVE,
        UInt32(0),
        UInt32(1),
    )
    iszero(observed) || error(
        "another arena worker team is already active in this process",
    )
    if ccall(:jl_in_threaded_region, Cint, ()) != 0
        Base.Threads.atomic_xchg!(ARENA_TEAM_ACTIVE, UInt32(0))
        error(
            "arena worker team cannot start inside an existing threaded region",
        )
    end
    return nothing
end

function _release_arena_team_guard!()
    previous = Base.Threads.atomic_xchg!(
        ARENA_TEAM_ACTIVE,
        UInt32(0),
    )
    previous == UInt32(1) || error(
        "arena worker team guard was not held during release",
    )
    return nothing
end

function _run_with_arena_team_guarded!(
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
    fill!(executor.bindings, nothing)
    fill!(executor.bindings_released, false)
    reset(executor.startup_event)
    executor.started = true
    result = Ref{Any}(nothing)
    threading_failure = nothing
    try
        Base.Threads.threading_run(worker_slot -> begin
            worker_exception = nothing
            worker_backtrace = nothing
            binding_attempted = false
            try
                binding_attempted = true
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
                worker_exception = exception
                worker_backtrace = catch_backtrace()
            finally
                if binding_attempted
                    try
                        CpuSets.clear_current_binding!()
                        executor.bindings_released[worker_slot] = true
                    catch teardown_exception
                        teardown_backtrace = catch_backtrace()
                        if worker_exception === nothing
                            worker_exception = teardown_exception
                            worker_backtrace = teardown_backtrace
                        else
                            worker_exception = CompositeException(Any[
                                Base.CapturedException(
                                    worker_exception,
                                    worker_backtrace,
                                ),
                                Base.CapturedException(
                                    teardown_exception,
                                    teardown_backtrace,
                                ),
                            ])
                        end
                    end
                end
                if worker_exception !== nothing
                    _record_failure!(
                        executor,
                        min(worker_slot, length(executor.failures)),
                        worker_exception,
                        worker_backtrace,
                    )
                end
            end
            return nothing
        end, true)
    catch exception
        threading_failure = (
            exception=exception,
            backtrace=catch_backtrace(),
        )
    finally
        executor.started = false
    end
    teardown_failure = try
        _verify_arena_team_teardown!(executor)
        nothing
    catch exception
        (
            exception=exception,
            backtrace=catch_backtrace(),
        )
    end
    captured_failures = Base.CapturedException[]
    failure_worker = executor.failure_worker[]
    if failure_worker != 0
        payload = executor.failures[failure_worker]
        if payload === nothing
            push!(
                captured_failures,
                Base.CapturedException(
                    ErrorException(
                        "arena worker $failure_worker failed without payload",
                    ),
                    Any[],
                ),
            )
        else
            exception, backtrace = payload
            push!(
                captured_failures,
                Base.CapturedException(exception, backtrace),
            )
        end
    end
    threading_failure === nothing || push!(
        captured_failures,
        Base.CapturedException(
            threading_failure.exception,
            threading_failure.backtrace,
        ),
    )
    teardown_failure === nothing || push!(
        captured_failures,
        Base.CapturedException(
            teardown_failure.exception,
            teardown_failure.backtrace,
        ),
    )
    if length(captured_failures) == 1
        throw(only(captured_failures))
    elseif length(captured_failures) > 1
        throw(CompositeException(Any[captured_failures...]))
    end
    return (;
        result=result[],
        binding_plan,
        bindings=copy(executor.bindings),
        bindings_released=copy(executor.bindings_released),
    )
end

function run_with_arena_team!(
    body::F,
    executor::ArenaExecutor,
) where {F}
    _acquire_arena_team_guard!()
    try
        return _run_with_arena_team_guarded!(body, executor)
    finally
        _release_arena_team_guard!()
    end
end

run_with_arena_team!(
    executor::ArenaExecutor,
    body::F,
) where {F} = run_with_arena_team!(body, executor)

end # module
