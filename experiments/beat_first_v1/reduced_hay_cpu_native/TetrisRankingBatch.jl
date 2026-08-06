module TetrisRankingBatch

"""
Model-neutral, fixed-arena Tetris ranking batch contract.

This module deliberately owns no neuron, routing, tape, optimizer, or
structural-learning state.  Its complete mutable surface is batch metadata,
1,298 binary sensory rails, supervised targets, the 22 raw outputs and their
supervised cotangent, plus fixed packing/loss scratch.
"""

export AUX_FEATURES,
    AUX_LEVELS,
    INPUT_RAILS,
    OUTPUT_DIM,
    QUANTILES,
    Batch,
    LossScratch,
    PackScratch,
    SupervisedLoss,
    Targets,
    ValidatedDataset,
    flat_index,
    pack_batch!,
    pack_candidate_rails!,
    prepare_batch_metadata!,
    state_candidate,
    q_state_objective!,
    supervised_loss_and_raw_gradient!,
    validate_dataset

const OUTPUT_DIM = 22
const QUANTILES = 16
const BOARD_ROWS = 24
const BOARD_COLUMNS = 10
const BOARD_CELLS = BOARD_ROWS * BOARD_COLUMNS
const QUEUE_PIECES = 7
const QUEUE_TOKENS = 6
const AUX_FEATURES = 37
const AUX_LEVELS = 8
const INPUT_RAILS =
    4 * BOARD_CELLS + QUEUE_PIECES * QUEUE_TOKENS +
    AUX_FEATURES * AUX_LEVELS

const LISTNET_TEMPERATURE = 0.50f0
const Q_HUBER_WEIGHT = 0.25f0
const MARGIN_WEIGHT = 0.15f0
const DEATH_WEIGHT = 0.10f0
const QUANTILE_WEIGHT = 0.05f0
const GEOMETRY_WEIGHT = 0.10f0

"""
Borrowed, schema-checked teacher dataset.

Construction is the sole boundary that accepts an untrusted dataset object.
The wrapper does not copy its arrays, so callers must treat the underlying
arrays as immutable for the wrapper's lifetime.  Hot packing functions accept
only this type and can consequently use `@inbounds` without repeating dynamic
rank/shape/type checks for every candidate.
"""
struct ValidatedDataset{
    B,
    P,
    Q,
    TQ,
    AC,
    SA,
    TM,
    CD,
    CDA,
    LC,
    MH,
    HO,
    CA,
    RE,
    BTB,
    TS,
}
    state_count::Int
    candidate_width::Int
    boards::B
    placements::P
    queues::Q
    teacher_q::TQ
    action_counts::AC
    selected_actions::SA
    terminal::TM
    candidate_death::CD
    candidate_death_available::CDA
    line_clear::LC
    max_height::MH
    holes::HO
    cavities::CA
    ren::RE
    back_to_back::BTB
    tspin::TS
end

@noinline function _required_property(dataset, name::Symbol)
    hasproperty(dataset, name) || throw(ArgumentError(
        "teacher dataset is missing required field `$name`",
    ))
    return getproperty(dataset, name)
end

@noinline function _checked_array(
    dataset,
    name::Symbol,
    ::Type{T},
    dimensions::Int,
    expected_shape::Tuple,
) where {T}
    value = _required_property(dataset, name)
    value isa AbstractArray || throw(ArgumentError(
        "teacher dataset `$name` must be an array",
    ))
    ndims(value) == dimensions || throw(DimensionMismatch(
        "teacher dataset `$name` rank $(ndims(value)); expected $dimensions",
    ))
    eltype(value) === T || throw(ArgumentError(
        "teacher dataset `$name` element type $(eltype(value)); expected $T",
    ))
    size(value) == expected_shape || throw(DimensionMismatch(
        "teacher dataset `$name` shape $(size(value)); expected $expected_shape",
    ))
    return value
end

"""
Validate the exact dense teacher contract once and return a borrowed wrapper.

The canonical element types are the materialized `BeatFirstTrainingCore`
types: binary tensors are `UInt8`, continuous values are `Float32`, candidate
indices/counts are native `Int`, death flags are `Bool`, and stored geometry is
`Int8`/`Int16`.  Candidate storage is exactly `candidate_width`; wider or
narrower arrays are rejected instead of silently truncated.
"""
function validate_dataset(dataset, candidate_width::Int)
    candidate_width >= 1 || throw(ArgumentError(
        "candidate_width must be positive",
    ))
    action_counts_value = _required_property(dataset, :action_counts)
    action_counts_value isa AbstractVector || throw(ArgumentError(
        "teacher dataset `action_counts` must be a vector",
    ))
    eltype(action_counts_value) === Int || throw(ArgumentError(
        "teacher dataset `action_counts` element type " *
        "$(eltype(action_counts_value)); expected Int",
    ))
    state_count = length(action_counts_value)
    state_count >= 1 || throw(ArgumentError("teacher dataset is empty"))

    boards = _checked_array(
        dataset,
        :boards,
        UInt8,
        4,
        (BOARD_ROWS, BOARD_COLUMNS, 1, state_count),
    )
    placements = _checked_array(
        dataset,
        :placements,
        UInt8,
        5,
        (BOARD_ROWS, BOARD_COLUMNS, 1, candidate_width, state_count),
    )
    queues = _checked_array(
        dataset,
        :queues,
        UInt8,
        3,
        (QUEUE_PIECES, QUEUE_TOKENS, state_count),
    )
    teacher_q = _checked_array(
        dataset,
        :teacher_q,
        Float32,
        2,
        (candidate_width, state_count),
    )
    selected_actions = _checked_array(
        dataset,
        :selected_actions,
        Int,
        1,
        (state_count,),
    )
    terminal = _checked_array(
        dataset,
        :terminal,
        Bool,
        1,
        (state_count,),
    )
    candidate_death = _checked_array(
        dataset,
        :candidate_death,
        Bool,
        2,
        (candidate_width, state_count),
    )
    candidate_death_available = _checked_array(
        dataset,
        :candidate_death_available,
        Bool,
        1,
        (state_count,),
    )
    line_clear = _checked_array(
        dataset,
        :line_clear,
        Int8,
        2,
        (candidate_width, state_count),
    )
    max_height = _checked_array(
        dataset,
        :max_height,
        Int8,
        2,
        (candidate_width, state_count),
    )
    holes = _checked_array(
        dataset,
        :holes,
        Int16,
        2,
        (candidate_width, state_count),
    )
    cavities = _checked_array(
        dataset,
        :cavities,
        Int16,
        2,
        (candidate_width, state_count),
    )
    ren = _checked_array(
        dataset,
        :ren,
        Float32,
        2,
        (1, state_count),
    )
    back_to_back = _checked_array(
        dataset,
        :back_to_back,
        Float32,
        2,
        (1, state_count),
    )
    tspin = _checked_array(
        dataset,
        :tspin,
        Float32,
        2,
        (candidate_width, state_count),
    )

    @inbounds for row in 1:state_count
        count = action_counts_value[row]
        1 <= count <= candidate_width || throw(ArgumentError(
            "state $row has $count candidates outside 1:$candidate_width",
        ))
        selected = selected_actions[row]
        1 <= selected <= count || throw(ArgumentError(
            "state $row selected action $selected outside 1:$count",
        ))
        for candidate in 1:count
            isfinite(teacher_q[candidate, row]) || throw(ArgumentError(
                "non-finite teacher Q at state $row candidate $candidate",
            ))
            0 <= line_clear[candidate, row] <= 4 || throw(ArgumentError(
                "line-clear target outside 0:4 at state $row candidate $candidate",
            ))
            0 <= max_height[candidate, row] <= BOARD_ROWS ||
                throw(ArgumentError(
                    "max-height target outside 0:$BOARD_ROWS at state $row " *
                    "candidate $candidate",
                ))
            0 <= holes[candidate, row] <= BOARD_CELLS || throw(ArgumentError(
                "holes target outside 0:$BOARD_CELLS at state $row candidate $candidate",
            ))
            0 <= cavities[candidate, row] <= BOARD_CELLS ||
                throw(ArgumentError(
                    "cavities target outside 0:$BOARD_CELLS at state $row " *
                    "candidate $candidate",
                ))
            isfinite(tspin[candidate, row]) || throw(ArgumentError(
                "non-finite tspin at state $row candidate $candidate",
            ))
        end
        isfinite(ren[1, row]) || throw(ArgumentError(
            "non-finite ren at state $row",
        ))
        isfinite(back_to_back[1, row]) || throw(ArgumentError(
            "non-finite back_to_back at state $row",
        ))
    end

    return ValidatedDataset(
        state_count,
        candidate_width,
        boards,
        placements,
        queues,
        teacher_q,
        action_counts_value,
        selected_actions,
        terminal,
        candidate_death,
        candidate_death_available,
        line_clear,
        max_height,
        holes,
        cavities,
        ren,
        back_to_back,
        tspin,
    )
end

mutable struct Targets
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

function Targets(width::Int, state_batch::Int)
    return Targets(
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
Fixed-capacity storage for `state_batch` groups of at most `width` candidates.

Candidate `c` in state slot `s` always occupies
`flat = c + (s - 1) * width`.  `counts` is therefore the only candidate mask;
valid candidates are also listed in stable state-major order in `valid_flats`.
"""
mutable struct Batch
    state_batch::Int
    width::Int
    capacity::Int
    rows::Vector{Int}
    counts::Vector{Int16}
    valid_flats::Vector{Int32}
    valid_count::Int
    targets::Targets
    rails::Matrix{Float32}
    raw::Matrix{Float32}
    raw_gradient::Matrix{Float32}
    listnet_q_gradient::Vector{Float32}
end

function Batch(state_batch::Int, width::Int)
    state_batch >= 1 || throw(ArgumentError("state_batch must be positive"))
    width >= 1 || throw(ArgumentError("candidate width must be positive"))
    capacity = Base.checked_mul(state_batch, width)
    return Batch(
        state_batch,
        width,
        capacity,
        zeros(Int, state_batch),
        zeros(Int16, state_batch),
        zeros(Int32, capacity),
        0,
        Targets(width, state_batch),
        zeros(Float32, INPUT_RAILS, capacity),
        zeros(Float32, OUTPUT_DIM, capacity),
        zeros(Float32, OUTPUT_DIM, capacity),
        zeros(Float32, capacity),
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

"""
Fixed-shape storage for the supervised loss kernel.

The backing arrays are matrices rather than vectors deliberately: Julia
vectors can be resized after construction, whereas a matrix has fixed storage
and dimensions.  The immutable wrapper also prevents replacing one buffer with
an incompatible array.  Linear indexing keeps the hot kernel unchanged.
"""
struct LossScratch
    student_z::Matrix{Float32}
    teacher_probability::Matrix{Float32}
    student_probability::Matrix{Float32}
    z_cotangent::Matrix{Float32}
    state_composite::Matrix{Float32}
    state_teacher_entropy::Matrix{Float32}

    function LossScratch(width::Int, state_batch::Int)
        width >= 1 || throw(ArgumentError("candidate width must be positive"))
        state_batch >= 1 || throw(ArgumentError("state_batch must be positive"))
        return new(
            zeros(Float32, width, 1),
            zeros(Float32, width, 1),
            zeros(Float32, width, 1),
            zeros(Float32, width, 1),
            zeros(Float32, state_batch, 1),
            zeros(Float32, state_batch, 1),
        )
    end
end

struct SupervisedLoss
    composite_loss::Float32
    listnet_loss::Float32
    teacher_entropy::Float32
    listnet_kl::Float32
    q_huber_loss::Float32
    margin_loss::Float32
    death_loss::Float32
    quantile_teacher_loss::Float32
    geometry_loss::Float32
    line_clear_loss::Float32
    max_height_loss::Float32
    holes_loss::Float32
    cavities_loss::Float32
    valid_candidates::Int
end

@inline flat_index(candidate::Int, state_slot::Int, width::Int) =
    candidate + (state_slot - 1) * width

@inline function state_candidate(flat::Int, width::Int)
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

"""Prepare candidate grouping and every supervised target without allocation."""
function prepare_batch_metadata!(
    batch::Batch,
    dataset::ValidatedDataset,
)
    batch.width == dataset.candidate_width || throw(DimensionMismatch(
        "batch width $(batch.width) differs from validated dataset width " *
        "$(dataset.candidate_width)",
    ))
    batch.valid_count = 0
    fill!(batch.raw, 0.0f0)
    fill!(batch.raw_gradient, 0.0f0)
    fill!(batch.listnet_q_gradient, 0.0f0)
    targets = batch.targets
    width = batch.width
    @inbounds for state_slot in 1:batch.state_batch
        row = batch.rows[state_slot]
        1 <= row <= dataset.state_count ||
            throw(BoundsError(dataset.action_counts, row))
        count = Int(dataset.action_counts[row])
        1 <= count <= width || error(
            "state $row has $count candidates outside width $width",
        )
        batch.counts[state_slot] = Int16(count)

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
        candidate_death_available = dataset.candidate_death_available[row]
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
            batch.valid_count += 1
            batch.valid_flats[batch.valid_count] =
                Int32(flat_index(candidate, state_slot, width))
        end
    end
    return batch.valid_count
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

"""Pack one valid candidate into the canonical 1,298 binary rails."""
function pack_candidate_rails!(
    batch::Batch,
    dataset::ValidatedDataset,
    scratch::PackScratch,
    flat::Int,
)
    batch.width == dataset.candidate_width || throw(DimensionMismatch(
        "batch and validated dataset candidate widths differ",
    ))
    1 <= flat <= batch.capacity || throw(BoundsError(batch.rails, (:, flat)))
    state_slot, candidate = state_candidate(flat, batch.width)
    1 <= state_slot <= batch.state_batch ||
        throw(BoundsError(batch.rows, state_slot))
    1 <= candidate <= Int(batch.counts[state_slot]) || throw(ArgumentError(
        "flat candidate $flat is not valid in its prepared state slot",
    ))
    row = batch.rows[state_slot]
    expected_line_clear = Int(batch.targets.line_clear[candidate, state_slot])
    actual_line_clear = _fill_after_board!(scratch, dataset, row, candidate)
    actual_line_clear == expected_line_clear || error(
        "stored line-clear target differs at row $row candidate $candidate",
    )
    cavities, aggregate_height, bumpiness, max_height =
        _fill_geometry!(scratch)
    max_height == Int(batch.targets.max_height[candidate, state_slot]) ||
        error("stored max-height target differs")
    holes_total = 0
    @inbounds for value in scratch.holes
        holes_total += Int(value)
    end
    holes_total == Int(batch.targets.holes[candidate, state_slot]) ||
        error("stored hole target differs")
    cavities == Int(batch.targets.cavities[candidate, state_slot]) ||
        error("stored cavity target differs")

    rail = 0
    @inbounds for column in 1:BOARD_COLUMNS, board_row in 1:BOARD_ROWS
        rail += 1
        batch.rails[rail, flat] = scratch.board[board_row, column] > 0.5f0
    end
    @inbounds for column in 1:BOARD_COLUMNS, board_row in 1:BOARD_ROWS
        rail += 1
        batch.rails[rail, flat] = scratch.after[board_row, column] > 0.5f0
    end
    @inbounds for column in 1:BOARD_COLUMNS, board_row in 1:BOARD_ROWS
        rail += 1
        batch.rails[rail, flat] =
            scratch.after[board_row, column] > scratch.board[board_row, column]
    end
    @inbounds for column in 1:BOARD_COLUMNS, board_row in 1:BOARD_ROWS
        rail += 1
        batch.rails[rail, flat] =
            scratch.after[board_row, column] < scratch.board[board_row, column]
    end
    @inbounds for token in 1:QUEUE_TOKENS, piece in 1:QUEUE_PIECES
        rail += 1
        batch.rails[rail, flat] =
            Float32(dataset.queues[piece, token, row]) > 0.5f0
    end
    @inbounds for level in 1:AUX_LEVELS
        threshold = Float32(level) / Float32(AUX_LEVELS)
        for index in 1:AUX_FEATURES
            rail += 1
            batch.rails[rail, flat] = _aux_value(
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
    rail == INPUT_RAILS || error("binary rail packing drift")
    return nothing
end

"""
Prepare and pack a batch using caller-owned scratch.

After `Batch` and `PackScratch` construction this routine performs no heap
allocation.  Parallel runtimes should own one `PackScratch` per worker and
call `prepare_batch_metadata!` once before candidate dispatch.
"""
function pack_batch!(
    batch::Batch,
    dataset::ValidatedDataset,
    scratch::PackScratch,
)
    prepare_batch_metadata!(batch, dataset)
    @inbounds for ordinal in 1:batch.valid_count
        pack_candidate_rails!(
            batch,
            dataset,
            scratch,
            Int(batch.valid_flats[ordinal]),
        )
    end
    return batch
end

@inline _huber(value::Float32) =
    abs(value) <= 1.0f0 ? 0.5f0 * value * value : abs(value) - 0.5f0

@inline _huber_derivative(value::Float32) = clamp(value, -1.0f0, 1.0f0)

@inline function _q_value(
    batch::Batch,
    offset::Int,
    candidate::Int,
    override_candidate::Int,
    override_value::Float32,
)
    return candidate == override_candidate ?
        override_value : batch.raw[1, offset + candidate]
end

"""
Compute the Q-dependent part of one state's supervised objective.

`override_candidate != 0` replaces exactly one candidate Q while every other
hard output and auxiliary channel remains fixed.  This is the cheap known-loss
counterfactual required by the terminal hard-bit credit rule; it neither writes
the model output nor produces a backward cotangent.
"""
function q_state_objective!(
    batch::Batch,
    scratch::LossScratch,
    state_slot::Int,
    override_candidate::Int=0,
    override_value::Float32=0.0f0,
)
    1 <= state_slot <= batch.state_batch || throw(BoundsError(
        batch.counts,
        state_slot,
    ))
    count = Int(batch.counts[state_slot])
    1 <= count <= batch.width || throw(ArgumentError(
        "state slot $state_slot has an invalid candidate count",
    ))
    0 <= override_candidate <= count || throw(ArgumentError(
        "counterfactual candidate is outside the state",
    ))
    isfinite(override_value) || throw(ArgumentError(
        "counterfactual Q must be finite",
    ))
    offset = (state_slot - 1) * batch.width
    q_mean = 0.0f0
    @inbounds for candidate in 1:count
        q_mean += _q_value(
            batch,
            offset,
            candidate,
            override_candidate,
            override_value,
        )
    end
    q_mean /= Float32(count)
    variance = 0.0f0
    @inbounds for candidate in 1:count
        centered = _q_value(
            batch,
            offset,
            candidate,
            override_candidate,
            override_value,
        ) - q_mean
        scratch.student_z[candidate] = centered
        variance = muladd(centered, centered, variance)
    end
    inverse_scale = inv(sqrt(variance / Float32(count) + 1.0f-4))
    teacher_max = -Inf32
    student_max = -Inf32
    @inbounds for candidate in 1:count
        student_z = scratch.student_z[candidate] * inverse_scale
        scratch.student_z[candidate] = student_z
        teacher_logit = batch.targets.teacher_z[candidate, state_slot] /
            LISTNET_TEMPERATURE
        student_logit = student_z / LISTNET_TEMPERATURE
        teacher_max = max(teacher_max, teacher_logit)
        student_max = max(student_max, student_logit)
    end
    teacher_sum = 0.0f0
    student_sum = 0.0f0
    @inbounds for candidate in 1:count
        teacher_probability = exp(
            batch.targets.teacher_z[candidate, state_slot] /
            LISTNET_TEMPERATURE - teacher_max,
        )
        student_probability = exp(
            scratch.student_z[candidate] /
            LISTNET_TEMPERATURE - student_max,
        )
        scratch.teacher_probability[candidate] = teacher_probability
        scratch.student_probability[candidate] = student_probability
        teacher_sum += teacher_probability
        student_sum += student_probability
    end
    inverse_teacher_sum = inv(max(teacher_sum, 1.0f-12))
    inverse_student_sum = inv(max(student_sum, 1.0f-12))
    listnet_loss = 0.0f0
    q_huber_loss = 0.0f0
    @inbounds for candidate in 1:count
        teacher_probability =
            scratch.teacher_probability[candidate] * inverse_teacher_sum
        student_probability =
            scratch.student_probability[candidate] * inverse_student_sum
        listnet_loss -= teacher_probability *
            log(max(student_probability, 1.0f-12)) /
            Float32(batch.state_batch)
        q_error = _q_value(
            batch,
            offset,
            candidate,
            override_candidate,
            override_value,
        ) - batch.targets.teacher_q[candidate, state_slot]
        q_huber_loss += _huber(q_error) / Float32(batch.valid_count)
    end
    top1 = Int(batch.targets.top1[state_slot])
    top2 = Int(batch.targets.top2[state_slot])
    margin_error =
        _q_value(batch, offset, top1, override_candidate, override_value) -
        _q_value(batch, offset, top2, override_candidate, override_value) -
        batch.targets.margin[state_slot]
    margin_loss = _huber(margin_error) / Float32(batch.state_batch)
    return listnet_loss + Q_HUBER_WEIGHT * q_huber_loss +
        MARGIN_WEIGHT * margin_loss
end

# Stable Float32 sigmoid used by the historical supervised contract.  The
# negative branch avoids overflow and also preserves its established rounding.
@inline function _sigmoid(value::Float32)
    if value >= 0.0f0
        return inv(1.0f0 + exp(-value))
    end
    exponential = exp(value)
    return exponential / (1.0f0 + exponential)
end

@inline function _require_matrix_shape(
    value::AbstractMatrix,
    rows::Int,
    columns::Int,
    name::Symbol,
)
    size(value, 1) == rows && size(value, 2) == columns ||
        throw(DimensionMismatch(
            "`$name` shape $(size(value)); expected ($rows, $columns)",
        ))
    return nothing
end

@inline function _require_vector_length(
    value::AbstractVector,
    expected::Int,
    name::Symbol,
)
    length(value) == expected || throw(DimensionMismatch(
        "`$name` length $(length(value)); expected $expected",
    ))
    return nothing
end

"""
Prove every shape and index invariant used by the bounds-elided loss kernel.

This check is constant in the number of arrays and linear only in the number
of state groups.  It deliberately runs before any output buffer is mutated, so
a corrupted arena fails closed under ordinary Julia execution as well as under
`--check-bounds=yes`.
"""
@noinline function _validate_loss_contract(
    batch::Batch,
    scratch::LossScratch,
)
    width = batch.width
    state_batch = batch.state_batch
    width >= 1 || throw(ArgumentError("batch width must be positive"))
    state_batch >= 1 || throw(ArgumentError("state_batch must be positive"))
    width <= typemax(Int) ÷ state_batch || throw(OverflowError(
        "batch width × state_batch overflows Int",
    ))
    capacity = width * state_batch
    batch.capacity == capacity || throw(DimensionMismatch(
        "batch capacity $(batch.capacity); expected $capacity",
    ))

    _require_vector_length(batch.rows, state_batch, :rows)
    _require_vector_length(batch.counts, state_batch, :counts)
    _require_vector_length(batch.valid_flats, capacity, :valid_flats)
    _require_matrix_shape(batch.rails, INPUT_RAILS, capacity, :rails)
    _require_matrix_shape(batch.raw, OUTPUT_DIM, capacity, :raw)
    _require_matrix_shape(
        batch.raw_gradient,
        OUTPUT_DIM,
        capacity,
        :raw_gradient,
    )
    _require_vector_length(
        batch.listnet_q_gradient,
        capacity,
        :listnet_q_gradient,
    )

    targets = batch.targets
    _require_matrix_shape(targets.teacher_q, width, state_batch, :teacher_q)
    _require_matrix_shape(targets.teacher_z, width, state_batch, :teacher_z)
    _require_vector_length(targets.top1, state_batch, :top1)
    _require_vector_length(targets.top2, state_batch, :top2)
    _require_vector_length(targets.margin, state_batch, :margin)
    _require_matrix_shape(targets.death, width, state_batch, :death)
    _require_matrix_shape(
        targets.death_mask,
        width,
        state_batch,
        :death_mask,
    )
    _require_matrix_shape(
        targets.line_clear,
        width,
        state_batch,
        :line_clear,
    )
    _require_matrix_shape(
        targets.max_height,
        width,
        state_batch,
        :max_height,
    )
    _require_matrix_shape(targets.holes, width, state_batch, :holes)
    _require_matrix_shape(targets.cavities, width, state_batch, :cavities)

    _require_matrix_shape(scratch.student_z, width, 1, :student_z)
    _require_matrix_shape(
        scratch.teacher_probability,
        width,
        1,
        :teacher_probability,
    )
    _require_matrix_shape(
        scratch.student_probability,
        width,
        1,
        :student_probability,
    )
    _require_matrix_shape(scratch.z_cotangent, width, 1, :z_cotangent)
    _require_matrix_shape(
        scratch.state_composite,
        state_batch,
        1,
        :state_composite,
    )
    _require_matrix_shape(
        scratch.state_teacher_entropy,
        state_batch,
        1,
        :state_teacher_entropy,
    )

    counted = 0
    @inbounds for state_slot in 1:state_batch
        count = Int(batch.counts[state_slot])
        1 <= count <= width || throw(ArgumentError(
            "state slot $state_slot has $count candidates outside 1:$width",
        ))
        top1 = Int(targets.top1[state_slot])
        top2 = Int(targets.top2[state_slot])
        1 <= top1 <= count || throw(ArgumentError(
            "state slot $state_slot top1 $top1 outside 1:$count",
        ))
        1 <= top2 <= count || throw(ArgumentError(
            "state slot $state_slot top2 $top2 outside 1:$count",
        ))
        counted += count
    end
    batch.valid_count == counted || throw(ArgumentError(
        "batch valid_count $(batch.valid_count); counts sum to $counted",
    ))
    return counted
end

"""
Compute the pure supervised ranking loss and its exact raw-output cotangent.

The objective is standardized ListNet plus raw-Q, top-gap, death, quantile and
geometry supervision.  Structural/gate regularization is intentionally absent
from this API and must be applied by the owning model/optimizer.
"""
function supervised_loss_and_raw_gradient!(
    batch::Batch,
    scratch::LossScratch,
)
    valid_total = _validate_loss_contract(batch, scratch)
    fill!(batch.raw_gradient, 0.0f0)
    fill!(batch.listnet_q_gradient, 0.0f0)
    raw = batch.raw
    draw = batch.raw_gradient
    targets = batch.targets
    width = batch.width
    state_batch = batch.state_batch
    fill!(scratch.state_composite, 0.0f0)
    fill!(scratch.state_teacher_entropy, 0.0f0)

    listnet_loss = 0.0f0
    teacher_entropy = 0.0f0
    q_huber_loss = 0.0f0
    margin_loss = 0.0f0
    death_loss = 0.0f0
    quantile_loss = 0.0f0
    line_clear_loss = 0.0f0
    max_height_loss = 0.0f0
    holes_loss = 0.0f0
    cavities_loss = 0.0f0
    death_count = 0.0f0
    @inbounds for state_slot in 1:state_batch
        count = Int(batch.counts[state_slot])
        for candidate in 1:count
            death_count +=
                targets.death_mask[candidate, state_slot] != 0.0f0
        end
    end
    death_denominator = max(death_count, 1.0f0)

    @inbounds for state_slot in 1:state_batch
        listnet_before = listnet_loss
        teacher_entropy_before = teacher_entropy
        q_before = q_huber_loss
        margin_before = margin_loss
        death_before = death_loss
        quantile_before = quantile_loss
        line_before = line_clear_loss
        height_before = max_height_loss
        holes_before = holes_loss
        cavities_before = cavities_loss
        count = Int(batch.counts[state_slot])
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
                targets.teacher_z[candidate, state_slot] /
                LISTNET_TEMPERATURE
            student_logit = z / LISTNET_TEMPERATURE
            teacher_max = max(teacher_max, teacher_logit)
            student_max = max(student_max, student_logit)
        end
        teacher_sum = 0.0f0
        student_sum = 0.0f0
        for candidate in 1:count
            teacher_probability = exp(
                targets.teacher_z[candidate, state_slot] /
                LISTNET_TEMPERATURE - teacher_max,
            )
            student_probability = exp(
                scratch.student_z[candidate] /
                LISTNET_TEMPERATURE - student_max,
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
            cotangent =
                (student_probability - teacher_probability) /
                (LISTNET_TEMPERATURE * Float32(state_batch))
            scratch.z_cotangent[candidate] = cotangent
            mean_g += cotangent
            mean_gz = muladd(cotangent, scratch.student_z[candidate], mean_gz)
        end
        mean_g /= Float32(count)
        mean_gz /= Float32(count)
        for candidate in 1:count
            flat = offset + candidate
            listnet_q_cotangent = (
                scratch.z_cotangent[candidate] - mean_g -
                scratch.student_z[candidate] * mean_gz
            ) * inverse_scale
            draw[1, flat] += listnet_q_cotangent
            batch.listnet_q_gradient[flat] = listnet_q_cotangent
        end

        top1 = Int(targets.top1[state_slot])
        top2 = Int(targets.top2[state_slot])
        margin_error =
            raw[1, offset + top1] - raw[1, offset + top2] -
            targets.margin[state_slot]
        margin_loss += _huber(margin_error) / Float32(state_batch)
        margin_gradient =
            MARGIN_WEIGHT * _huber_derivative(margin_error) /
            Float32(state_batch)
        draw[1, offset + top1] += margin_gradient
        draw[1, offset + top2] -= margin_gradient

        for candidate in 1:count
            flat = offset + candidate
            q_error = raw[1, flat] - targets.teacher_q[candidate, state_slot]
            q_huber_loss += _huber(q_error) / Float32(valid_total)
            draw[1, flat] +=
                Q_HUBER_WEIGHT * _huber_derivative(q_error) /
                Float32(valid_total)

            if targets.death_mask[candidate, state_slot] != 0.0f0
                logit = raw[2, flat]
                label = targets.death[candidate, state_slot]
                death_loss +=
                    max(logit, 0.0f0) - logit * label +
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
                    QUANTILE_WEIGHT * weight * _huber_derivative(error) /
                    Float32(valid_total * QUANTILES)
            end

            line_error = raw[19, flat] -
                targets.line_clear[candidate, state_slot] / 4.0f0
            height_error = raw[20, flat] -
                targets.max_height[candidate, state_slot] / 24.0f0
            holes_error = raw[21, flat] -
                targets.holes[candidate, state_slot] / 240.0f0
            cavities_error = raw[22, flat] -
                targets.cavities[candidate, state_slot] / 240.0f0
            line_clear_loss += _huber(line_error) / Float32(valid_total)
            max_height_loss += _huber(height_error) / Float32(valid_total)
            holes_loss += _huber(holes_error) / Float32(valid_total)
            cavities_loss += _huber(cavities_error) / Float32(valid_total)
            geometry_gradient_scale =
                (GEOMETRY_WEIGHT / 4.0f0) / Float32(valid_total)
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
            Q_HUBER_WEIGHT * (q_huber_loss - q_before) +
            MARGIN_WEIGHT * (margin_loss - margin_before) +
            DEATH_WEIGHT * (death_loss - death_before) / death_denominator +
            QUANTILE_WEIGHT * (quantile_loss - quantile_before) +
            (GEOMETRY_WEIGHT / 4.0f0) * (
                line_clear_loss - line_before +
                max_height_loss - height_before +
                holes_loss - holes_before +
                cavities_loss - cavities_before
            )
    end

    @inbounds for state_slot in 1:state_batch
        count = Int(batch.counts[state_slot])
        offset = (state_slot - 1) * width
        for candidate in 1:count
            targets.death_mask[candidate, state_slot] == 0.0f0 && continue
            flat = offset + candidate
            draw[2, flat] += DEATH_WEIGHT * (
                _sigmoid(raw[2, flat]) -
                targets.death[candidate, state_slot]
            ) / death_denominator
        end
    end
    death_loss /= death_denominator
    geometry_loss = (
        line_clear_loss + max_height_loss + holes_loss + cavities_loss
    ) / 4.0f0
    composite_loss =
        listnet_loss +
        Q_HUBER_WEIGHT * q_huber_loss +
        MARGIN_WEIGHT * margin_loss +
        DEATH_WEIGHT * death_loss +
        QUANTILE_WEIGHT * quantile_loss +
        GEOMETRY_WEIGHT * geometry_loss
    listnet_kl = max(listnet_loss - teacher_entropy, 0.0f0)
    return SupervisedLoss(
        composite_loss,
        listnet_loss,
        teacher_entropy,
        listnet_kl,
        q_huber_loss,
        margin_loss,
        death_loss,
        quantile_loss,
        geometry_loss,
        line_clear_loss,
        max_height_loss,
        holes_loss,
        cavities_loss,
        valid_total,
    )
end

end # module TetrisRankingBatch
