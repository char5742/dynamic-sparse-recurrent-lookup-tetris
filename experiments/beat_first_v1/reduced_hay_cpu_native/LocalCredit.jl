module CanonicalLocalLearner

using LinearAlgebra

using ..ActiveApicalCell
using ..StateCodec
using ..Float32NumericCore
using ..ReducedHayCPUNativeEventGraph
using ..Payload
using ..SensoryEncoder
using ..OutputCellBank
using ..Topology
using ..ReducedHayCPUNativeModel
using ..LearningPrimitives
using ..ReducedHayCPUNativeArena
using ..BarrierlessScheduler
using ..TetrisRankingBatch
using ..ActivityPlasticity
using ..LearningConfig
using ..CanonicalOptimizer

export FixedBlockFeedback,
    QOrdinalScratch,
    LocalCreditScratch,
    CanonicalTrainer,
    prepare_q_eprop_signal!,
    add_q_top_ordinal_signal!,
    local_candidate_gradient!,
    train_update!,
    batch_top1,
    batch_tie_aware_top1,
    mechanism_counts,
    assert_due_mechanisms!

const Cell = ActiveApicalCell
const Codec = StateCodec
const Numeric = Float32NumericCore
const Graph = ReducedHayCPUNativeEventGraph
const EventPayload = Payload
const Encoder = SensoryEncoder
const OutputBank = OutputCellBank
const NetworkTopology = Topology
const Model = ReducedHayCPUNativeModel
const Primitives = LearningPrimitives
const Arena = ReducedHayCPUNativeArena
const SchedulerRuntime = BarrierlessScheduler
const Ranking = TetrisRankingBatch
const Plasticity = ActivityPlasticity
const Config = LearningConfig
const Optimizer = CanonicalOptimizer

const RECURRENT_PARAMETER_FIELDS = (
    :cell_raw,
    :sensory_gain_raw,
    :edge_strength_raw,
    :payload_gain_raw,
)
const OUTPUT_PARAMETER_FIELDS = (
    :output_cell_raw,
    :output_edge_raw,
    :output_q_basal_bias_raw,
    :output_gain,
    :output_bias,
)
const FEEDBACK_RANK = 4
# Keep enough deterministic reducer partitions to expose one credit chunk per
# canonical CPU worker.  Reducer ownership is ordinal-based, so increasing the
# fixed count preserves serial/parallel numerical equivalence while avoiding a
# 12-way ceiling on a 20-core machine.
const GRADIENT_REDUCERS = 20

"""Build the posterior IEEE-bit signal for teacher-free Q eligibility.

`latch_probability` is produced without a target during forward. Only here,
after every candidate has completed, is the teacher Float32 word unpacked. The
signal is the bit-BCE derivative with respect to the max-margin logit and is not
candidate-centered, so constant sign and exponent bits remain learnable.
"""
function prepare_q_eprop_signal!(
    destination::AbstractMatrix{Float32},
    batch::Ranking.Batch,
    latch_probability::AbstractMatrix{Float32};
    label_smoothing::Float32=0.0f0,
)
    size(destination) == (OutputBank.Q_OUTPUT_CELLS, batch.capacity) || throw(
        DimensionMismatch("Q e-prop signal storage has the wrong shape"),
    )
    size(latch_probability) == size(destination) || throw(DimensionMismatch(
        "Q latch-probability storage has the wrong shape",
    ))
    0.0f0 <= label_smoothing < 0.5f0 || throw(ArgumentError(
        "Q label smoothing must be in [0, 0.5)",
    ))
    fill!(destination, 0.0f0)
    applied = 0
    @inbounds for ordinal in 1:batch.valid_count
        flat = Int(batch.valid_flats[ordinal])
        state_slot, candidate = Ranking.state_candidate(flat, batch.width)
        teacher_q = batch.targets.teacher_q[candidate, state_slot]
        isfinite(teacher_q) || throw(ArgumentError(
            "teacher Q must be finite for IEEE-bit supervision",
        ))
        target_word = reinterpret(UInt32, teacher_q)
        for bit in 0:(OutputBank.Q_OUTPUT_CELLS - 1)
            probability = latch_probability[bit + 1, flat]
            isfinite(probability) && 0.0f0 <= probability <= 1.0f0 || throw(
                ArgumentError("Q latch probability must be finite and in [0,1]"),
            )
            target = iszero(target_word & (UInt32(1) << bit)) ? 0.0f0 : 1.0f0
            smoothed_target = muladd(
                1.0f0 - 2.0f0 * label_smoothing,
                target,
                label_smoothing,
            )
            signal = (probability - smoothed_target) /
                OutputBank.Q_LATCH_TEMPERATURE
            destination[bit + 1, flat] = signal
            applied += !iszero(signal)
        end
    end
    return applied
end

"""Fixed scratch for the posterior IEEE sortable-key surrogate."""
struct QOrdinalScratch
    equality::Vector{Float64}
    decisive::Vector{Float64}
    tail::Vector{Float64}
    greater_a::Vector{Float64}
    greater_b::Vector{Float64}
    less_a::Vector{Float64}
    less_b::Vector{Float64}
    result_a::Vector{Float64}
    result_b::Vector{Float64}
end

QOrdinalScratch() = QOrdinalScratch(
    zeros(Float64, 31),
    zeros(Float64, 31),
    zeros(Float64, 32),
    zeros(Float64, 32),
    zeros(Float64, 32),
    zeros(Float64, 32),
    zeros(Float64, 32),
    zeros(Float64, 32),
    zeros(Float64, 32),
)

"""Probability and probability-gradient of a lower-31-bit lexicographic test."""
function _lexicographic_probability_gradient!(
    gradient_a::Vector{Float64},
    gradient_b::Vector{Float64},
    probability_a::AbstractVector{Float32},
    probability_b::AbstractVector{Float32},
    scratch::QOrdinalScratch,
    ::Val{greater},
) where {greater}
    fill!(gradient_a, 0.0)
    fill!(gradient_b, 0.0)
    @inbounds for position in 1:31
        bit = 31 - position
        left = Float64(probability_a[bit + 1])
        right = Float64(probability_b[bit + 1])
        scratch.equality[position] =
            left * right + (1.0 - left) * (1.0 - right)
        scratch.decisive[position] = greater ?
            left * (1.0 - right) :
            (1.0 - left) * right
    end
    scratch.tail[32] = 0.0
    @inbounds for position in 31:-1:1
        scratch.tail[position] = scratch.decisive[position] +
            scratch.equality[position] * scratch.tail[position + 1]
    end
    prefix = 1.0
    @inbounds for position in 1:31
        bit = 31 - position
        left = Float64(probability_a[bit + 1])
        right = Float64(probability_b[bit + 1])
        decisive_a = greater ? 1.0 - right : -right
        decisive_b = greater ? -left : 1.0 - left
        equality_a = 2.0 * right - 1.0
        equality_b = 2.0 * left - 1.0
        gradient_a[bit + 1] = prefix * (
            decisive_a + equality_a * scratch.tail[position + 1]
        )
        gradient_b[bit + 1] = prefix * (
            decisive_b + equality_b * scratch.tail[position + 1]
        )
        prefix *= scratch.equality[position]
    end
    return scratch.tail[1]
end

"""Soft sortable-key probability for finite IEEE-754 numeric order."""
function _float_greater_probability_gradient!(
    probability_a::AbstractVector{Float32},
    probability_b::AbstractVector{Float32},
    scratch::QOrdinalScratch,
)
    greater = _lexicographic_probability_gradient!(
        scratch.greater_a,
        scratch.greater_b,
        probability_a,
        probability_b,
        scratch,
        Val(true),
    )
    less = _lexicographic_probability_gradient!(
        scratch.less_a,
        scratch.less_b,
        probability_a,
        probability_b,
        scratch,
        Val(false),
    )
    sign_a = Float64(probability_a[32])
    sign_b = Float64(probability_b[32])
    both_positive = (1.0 - sign_a) * (1.0 - sign_b)
    both_negative = sign_a * sign_b
    probability = (1.0 - sign_a) * sign_b +
        both_positive * greater + both_negative * less
    @inbounds for bit in 1:31
        scratch.result_a[bit] =
            both_positive * scratch.greater_a[bit] +
            both_negative * scratch.less_a[bit]
        scratch.result_b[bit] =
            both_positive * scratch.greater_b[bit] +
            both_negative * scratch.less_b[bit]
    end
    scratch.result_a[32] = -sign_b -
        (1.0 - sign_b) * greater + sign_b * less
    scratch.result_b[32] = (1.0 - sign_a) -
        (1.0 - sign_a) * greater + sign_a * less
    return probability, scratch.result_a, scratch.result_b
end

"""
    add_q_top_ordinal_signal!(destination, batch, probability, scratch, weight)

Add the posterior top-vs-rest IEEE sortable-key signal after the independent
bit-BCE signal has been prepared.  `probability` was produced during the
teacher-free forward pass; teacher ranking enters only here.  This is a smooth
sortable-key surrogate for intermediate hard spikes, not an exact derivative
of the discrete hard register.

The optimizer averages the combined Q gradient over candidates, and the shared
cell additionally averages its 32 bit-local gradients.  To represent

`mean(bit BCE) + weight * mean(pair ordinal loss)`

before that common normalization, each comparison contributes
`weight * bit_count * valid_candidate_count / pair_count` to the raw signal.
The returned diagnostic loss remains `weight * mean(pair loss)`.
"""
function add_q_top_ordinal_signal!(
    destination::AbstractMatrix{Float32},
    batch::Ranking.Batch,
    probability::AbstractMatrix{Float32},
    scratch::QOrdinalScratch,
    weight::Float32,
)
    size(destination) == (OutputBank.Q_OUTPUT_CELLS, batch.capacity) || throw(
        DimensionMismatch("Q ordinal signal storage has the wrong shape"),
    )
    size(probability) == size(destination) || throw(DimensionMismatch(
        "Q latch-probability storage has the wrong shape",
    ))
    isfinite(weight) && weight >= 0.0f0 || throw(ArgumentError(
        "Q ordinal weight must be finite and nonnegative",
    ))
    iszero(weight) && return 0.0
    pair_count = 0
    @inbounds for state in 1:batch.state_batch
        candidate_count = Int(batch.counts[state])
        candidate_count > 0 || throw(ArgumentError(
            "Q ordinal state must contain at least one candidate",
        ))
        teacher_maximum = batch.targets.teacher_q[1, state]
        for candidate in 2:candidate_count
            teacher_maximum = max(
                teacher_maximum,
                batch.targets.teacher_q[candidate, state],
            )
        end
        for candidate in 1:candidate_count
            pair_count += batch.targets.teacher_q[candidate, state] <
                teacher_maximum - 1.0f-6
        end
    end
    pair_count > 0 || return 0.0
    # Preserve the validated capacity objective exactly.  The raw BCE signal
    # is later normalized by candidate count and, for the one shared cell,
    # bit count. Compensate the pair objective before that common reduction.
    loss_scale = Float64(weight) / pair_count
    signal_scale = loss_scale * OutputBank.Q_OUTPUT_CELLS * batch.valid_count
    ordinal_loss = 0.0
    @inbounds for state in 1:batch.state_batch
        candidate_count = Int(batch.counts[state])
        top = 1
        teacher_maximum = batch.targets.teacher_q[1, state]
        for candidate in 2:candidate_count
            teacher = batch.targets.teacher_q[candidate, state]
            if teacher > teacher_maximum
                teacher_maximum = teacher
                top = candidate
            end
        end
        top_flat = Ranking.flat_index(top, state, batch.width)
        top_probability = @view probability[:, top_flat]
        for candidate in 1:candidate_count
            batch.targets.teacher_q[candidate, state] <
                teacher_maximum - 1.0f-6 || continue
            other_flat = Ranking.flat_index(candidate, state, batch.width)
            other_probability = @view probability[:, other_flat]
            comparison, gradient_top, gradient_other =
                _float_greater_probability_gradient!(
                    top_probability,
                    other_probability,
                    scratch,
                )
            comparison = clamp(comparison, 1.0e-8, 1.0 - 1.0e-8)
            ordinal_loss -= loss_scale * log(comparison)
            loss_probability_bar = -signal_scale / comparison
            for bit in 1:OutputBank.Q_OUTPUT_CELLS
                top_p = top_probability[bit]
                other_p = other_probability[bit]
                destination[bit, top_flat] += Float32(
                    loss_probability_bar * gradient_top[bit] *
                    top_p * (1.0f0 - top_p) /
                    OutputBank.Q_LATCH_TEMPERATURE,
                )
                destination[bit, other_flat] += Float32(
                    loss_probability_bar * gradient_other[bit] *
                    other_p * (1.0f0 - other_p) /
                    OutputBank.Q_LATCH_TEMPERATURE,
                )
            end
        end
    end
    return ordinal_loss
end

"""
Fixed, seed-reproducible block-local projection of the 22-dimensional ranking
error. The factorization keeps storage proportional to cells and states while
remaining independent of every trainable output-cell parameter.
"""
struct FixedBlockFeedback
    left::Array{Float32,5}
    right::Array{Float32,5}
    time_coefficient::Matrix{Float32}
    utility_projection::Array{Float32,3}
end

@inline function _mix64(value::UInt64)
    value = xor(value, value >> 30)
    value *= UInt64(0xbf58476d1ce4e5b9)
    value = xor(value, value >> 27)
    value *= UInt64(0x94d049bb133111eb)
    return xor(value, value >> 31)
end

function FixedBlockFeedback(seed::Integer=0x4445434f4c4c4532)
    left = Array{Float32}(
        undef,
        Model.OUTPUT_DIM,
        FEEDBACK_RANK,
        Model.CELLS_PER_BLOCK,
        Model.BLOCKS,
        Model.RECURRENT_STEPS + 1,
    )
    right = Array{Float32}(
        undef,
        FEEDBACK_RANK,
        Model.STATE_DIM,
        Model.CELLS_PER_BLOCK,
        Model.BLOCKS,
        Model.RECURRENT_STEPS + 1,
    )
    left_scale = inv(sqrt(Float32(Model.OUTPUT_DIM)))
    right_scale = inv(sqrt(Float32(FEEDBACK_RANK)))
    linear = 0
    @inbounds for component in 1:(Model.RECURRENT_STEPS + 1)
        for block in 1:Model.BLOCKS
            for cell in 1:Model.CELLS_PER_BLOCK
                for output in 1:Model.OUTPUT_DIM
                    for latent in 1:FEEDBACK_RANK
                        linear += 1
                        mixed = _mix64(
                            xor(
                                UInt64(seed),
                                UInt64(linear) * UInt64(0x9e3779b97f4a7c15),
                            ),
                        )
                        left[output, latent, cell, block, component] =
                            ifelse(isodd(mixed), -left_scale, left_scale)
                    end
                end
                for latent in 1:FEEDBACK_RANK
                    for state in 1:Model.STATE_DIM
                        linear += 1
                        mixed = _mix64(
                            xor(
                                UInt64(seed),
                                UInt64(linear) * UInt64(0x9e3779b97f4a7c15),
                            ),
                        )
                        right[latent, state, cell, block, component] =
                            ifelse(isodd(mixed), right_scale, -right_scale)
                    end
                end
            end
        end
    end
    time_coefficient = fill(
        inv(sqrt(Float32(Model.RECURRENT_STEPS))),
        Model.OUTPUT_DIM,
        Model.RECURRENT_STEPS,
    )
    utility_projection = Array{Float32}(
        undef,
        Model.OUTPUT_DIM,
        Model.CELLS_PER_BLOCK,
        Model.BLOCKS,
    )
    @inbounds for block in 1:Model.BLOCKS
        for cell in 1:Model.CELLS_PER_BLOCK
            projection_square = 0.0f0
            for output in 1:Model.OUTPUT_DIM
                value = 0.0f0
                for component in 1:(Model.RECURRENT_STEPS + 1)
                    time_scale = component == 1 ?
                        time_coefficient[output, 1] :
                        time_coefficient[output, component - 1]
                    for state in 1:Model.STATE_DIM
                        for latent in 1:FEEDBACK_RANK
                            value = muladd(
                                left[output, latent, cell, block, component] *
                                    right[latent, state, cell, block, component],
                                time_scale,
                                value,
                            )
                        end
                    end
                end
                utility_projection[output, cell, block] = value
                projection_square = muladd(value, value, projection_square)
            end
            projection_square > 0.0f0 || error(
                "fixed block feedback collapsed to zero",
            )
            inverse_norm = inv(sqrt(projection_square))
            for output in 1:Model.OUTPUT_DIM
                utility_projection[output, cell, block] *= inverse_norm
            end
        end
    end
    return FixedBlockFeedback(
        left,
        right,
        time_coefficient,
        utility_projection,
    )
end

struct LocalCreditScratch{V<:AbstractVector{Float32}}
    output_trajectory::OutputBank.OutputTrajectory
    output_scratch::OutputBank.OutputScratch
    output_raw::Vector{Float32}
    output_anchor_bar::Array{Float32,3}
    output_recurrent_bar::Array{Float32,4}
    output_recurrent_bar_views::Vector{V}
    output_payload_gain_bar::Vector{Float32}
    encoded_anchor::Array{Float32,3}
    encoded_recurrent::Array{Float32,4}
    recurrent_encoded_views::Vector{V}
    anchor_signal::Array{Float32,3}
    recurrent_signal::Array{Float32,4}
    feedback_latent::Vector{Float32}
    encoded_bar::Vector{Float32}
    physical_bar::Vector{Float32}
    state_bar::Vector{Float32}
    input_bar::Vector{Float32}
    raw_cell_bar::Vector{Float32}
    payload_bar::Vector{Float32}
    payload_value::Vector{Float32}
    initial_state::Vector{Float32}
    recurrent_input_bar::Array{Float32,4}
    sensory_input::Array{Float32,3}
    sensory_input_bar::Array{Float32,3}
    rail_bar::Vector{Float32}
    sensory_gain_bar::Matrix{Float32}
    pending_bar::Graph.ConductanceInbox{Float32}
    ring::Graph.DelayedPayloadRing{Float32}
    ring_bar::Graph.DelayedPayloadRing{Float32}
    strength_bar::Vector{Float32}
    current_sources::Vector{Int}
    previous_sources::Vector{Int}
    active_sources::Vector{Int}
end

function _recurrent_views(array::Array{Float32,4})
    first_view = @view array[:, 1, 1, 1]
    views = Vector{typeof(first_view)}(
        undef,
        Model.CELLS_PER_BLOCK * Model.BLOCKS * Model.RECURRENT_STEPS,
    )
    @inbounds for step in 1:Model.RECURRENT_STEPS
        for block in 1:Model.BLOCKS
            for cell in 1:Model.CELLS_PER_BLOCK
                slot = cell + Model.CELLS_PER_BLOCK * (
                    (block - 1) + Model.BLOCKS * (step - 1)
                )
                views[slot] = @view array[:, cell, block, step]
            end
        end
    end
    return views
end

function LocalCreditScratch()
    output_anchor_bar = zeros(
        Float32,
        Model.STATE_DIM,
        Model.CELLS_PER_BLOCK,
        Model.BLOCKS,
    )
    output_recurrent_bar = zeros(
        Float32,
        Model.STATE_DIM,
        Model.CELLS_PER_BLOCK,
        Model.BLOCKS,
        Model.RECURRENT_STEPS,
    )
    encoded_anchor = similar(output_anchor_bar)
    encoded_recurrent = similar(output_recurrent_bar)
    return LocalCreditScratch(
        OutputBank.OutputTrajectory(),
        OutputBank.OutputScratch(),
        zeros(Float32, Model.OUTPUT_DIM),
        output_anchor_bar,
        output_recurrent_bar,
        _recurrent_views(output_recurrent_bar),
        zeros(Float32, EventPayload.ANALOG_GAIN_COUNT),
        encoded_anchor,
        encoded_recurrent,
        _recurrent_views(encoded_recurrent),
        zeros(Float32, size(encoded_anchor)),
        zeros(Float32, size(encoded_recurrent)),
        zeros(Float32, FEEDBACK_RANK),
        zeros(Float32, Model.STATE_DIM),
        zeros(Float32, Model.STATE_DIM),
        zeros(Float32, Model.STATE_DIM),
        zeros(Float32, Cell.INPUT_DIM),
        zeros(Float32, Cell.PARAM_DIM),
        zeros(Float32, EventPayload.PAYLOAD_DIM),
        zeros(Float32, EventPayload.PAYLOAD_DIM),
        zeros(Float32, Model.STATE_DIM),
        zeros(
            Float32,
            Cell.INPUT_DIM,
            Model.CELLS_PER_BLOCK,
            Model.BLOCKS,
            Model.RECURRENT_STEPS,
        ),
        zeros(Float32, Cell.INPUT_DIM, Model.CELLS_PER_BLOCK, Model.BLOCKS),
        zeros(Float32, Cell.INPUT_DIM, Model.CELLS_PER_BLOCK, Model.BLOCKS),
        zeros(Float32, Model.INPUT_RAILS),
        zeros(Float32, 2, Model.INPUT_RAILS),
        Graph.ConductanceInbox(Model.TOTAL_CELLS, Float32),
        Graph.DelayedPayloadRing(Model.TOTAL_CELLS, Float32),
        Graph.DelayedPayloadRing(Model.TOTAL_CELLS, Float32),
        zeros(Float32, Model.FANOUT * Model.TOTAL_CELLS),
        Vector{Int}(undef, Model.BLOCKS * Model.CELLS_PER_BLOCK),
        Vector{Int}(undef, Model.BLOCKS * Model.CELLS_PER_BLOCK),
        Vector{Int}(undef, 2 * Model.BLOCKS * Model.CELLS_PER_BLOCK),
    )
end

@inline _global_cell(block::Int, cell::Int) =
    (block - 1) * Model.CELLS_PER_BLOCK + cell

@inline function _accumulate_cell_raw!(destination, source)
    @inbounds for parameter in 1:Cell.PARAM_DIM
        destination[parameter] += source[parameter]
    end
    return nothing
end

function _encode_tape!(scratch, physical_anchor, physical_recurrent)
    @inbounds for block in 1:Model.BLOCKS
        for cell in 1:Model.CELLS_PER_BLOCK
            Codec.encode_state!(
                @view(scratch.encoded_anchor[:, cell, block]),
                @view(physical_anchor[:, cell, block]),
            )
        end
    end
    @inbounds for step in 1:Model.RECURRENT_STEPS
        for block in 1:Model.BLOCKS
            for cell in 1:Model.CELLS_PER_BLOCK
                Codec.encode_state!(
                    @view(scratch.encoded_recurrent[:, cell, block, step]),
                    @view(physical_recurrent[:, cell, block, step]),
                )
            end
        end
    end
    return nothing
end

function _prepare_local_signals!(
    scratch,
    feedback,
    output_cotangent,
    decolle_scale::Float32,
)
    # Canonical recurrent credit is independent of every trainable output-bank
    # Jacobian. The output reverse above updates only the four supervised
    # output parameter groups; these cell-local signals begin at zero and are
    # generated exclusively by the seed-fixed, block-specific DECOLLE map.
    fill!(scratch.anchor_signal, 0.0f0)
    fill!(scratch.recurrent_signal, 0.0f0)
    nonzero = 0
    @inbounds for block in 1:Model.BLOCKS
        for cell in 1:Model.CELLS_PER_BLOCK
            nonzero += _add_fixed_feedback!(
                @view(scratch.anchor_signal[:, cell, block]),
                scratch,
                feedback,
                output_cotangent,
                cell,
                block,
                1,
                decolle_scale,
            )
        end
    end
    @inbounds for step in 1:Model.RECURRENT_STEPS
        component = step + 1
        for block in 1:Model.BLOCKS
            for cell in 1:Model.CELLS_PER_BLOCK
                nonzero += _add_fixed_feedback!(
                    @view(scratch.recurrent_signal[:, cell, block, step]),
                    scratch,
                    feedback,
                    output_cotangent,
                    cell,
                    block,
                    component,
                    decolle_scale,
                )
            end
        end
    end
    return nonzero
end

function _add_fixed_feedback!(
    destination,
    scratch,
    feedback,
    output_cotangent,
    cell::Int,
    block::Int,
    component::Int,
    decolle_scale::Float32,
)
    fill!(scratch.feedback_latent, 0.0f0)
    output_square = 0.0f0
    @inbounds for output in 1:Model.OUTPUT_DIM
        cotangent = output_cotangent[output]
        output_square = muladd(cotangent, cotangent, output_square)
        time_scale = component == 1 ?
            feedback.time_coefficient[output, 1] :
            feedback.time_coefficient[output, component - 1]
        scaled = cotangent * time_scale
        for latent in 1:FEEDBACK_RANK
            scratch.feedback_latent[latent] = muladd(
                feedback.left[output, latent, cell, block, component],
                scaled,
                scratch.feedback_latent[latent],
            )
        end
    end
    projection_square = 0.0f0
    destination_square = 0.0f0
    @inbounds for state in 1:Model.STATE_DIM
        projected = 0.0f0
        for latent in 1:FEEDBACK_RANK
            projected = muladd(
                feedback.right[latent, state, cell, block, component],
                scratch.feedback_latent[latent],
                projected,
            )
        end
        scratch.encoded_bar[state] = projected
        projection_square = muladd(projected, projected, projection_square)
        destination_square = muladd(
            destination[state],
            destination[state],
            destination_square,
        )
    end
    projection_square > 0.0f0 || return 0
    output_rms = sqrt(output_square / Float32(Model.OUTPUT_DIM))
    destination_rms = sqrt(destination_square / Float32(Model.STATE_DIM))
    target_rms = decolle_scale * max(
        destination_rms,
        output_rms / sqrt(Float32(Model.STATE_DIM)),
    )
    scale = target_rms * sqrt(Float32(Model.STATE_DIM) / projection_square)
    @inbounds @simd for state in 1:Model.STATE_DIM
        destination[state] = muladd(
            scale,
            scratch.encoded_bar[state],
            destination[state],
        )
    end
    return target_rms > 0.0f0 ? 1 : 0
end

@inline function _local_physical_signal!(
    scratch,
    physical_state,
    signal,
)
    copyto!(scratch.encoded_bar, signal)
    event_bar = scratch.encoded_bar[Cell.SPIKE_INDEX]
    scratch.encoded_bar[Cell.SPIKE_INDEX] = 0.0f0
    Codec.state_codec_pullback!(
        scratch.physical_bar,
        physical_state,
        scratch.encoded_bar,
    )
    return event_bar
end

function _cell_local_credit!(
    gradient,
    scratch,
    cache,
    rails,
    physical_anchor,
    physical_recurrent,
    recurrent_inputs,
)
    fill!(scratch.recurrent_input_bar, 0.0f0)
    Encoder.encode_sensory_cached!(
        scratch.sensory_input,
        rails,
        cache.sensory_gain,
    )
    fill!(scratch.sensory_input_bar, 0.0f0)
    @inbounds for block in 1:Model.BLOCKS
        for cell in 1:Model.CELLS_PER_BLOCK
            # This is the forward-eligibility equivalent evaluated by a short
            # cell-local reverse replay. State credit crosses cycles inside
            # this one cell, but never traverses a recurrent graph edge or a
            # different cell.
            fill!(scratch.state_bar, 0.0f0)
            for step in Model.RECURRENT_STEPS:-1:1
                next_state = @view physical_recurrent[:, cell, block, step]
                event_bar = _local_physical_signal!(
                    scratch,
                    next_state,
                    @view(scratch.recurrent_signal[:, cell, block, step]),
                )
                for state in 1:Model.STATE_DIM
                    scratch.physical_bar[state] += scratch.state_bar[state]
                end
                if step == 1
                    Cell.cell_step_pullback!(
                        scratch.state_bar,
                        scratch.input_bar,
                        scratch.raw_cell_bar,
                        @view(physical_anchor[:, cell, block]),
                        @view(recurrent_inputs[:, cell, block, step]),
                        cache.cell[cell, block],
                        cache.cell_derivative[cell, block],
                        next_state,
                        scratch.physical_bar,
                        event_bar,
                        0.0f0,
                    )
                else
                    Cell.cell_step_pullback!(
                        scratch.state_bar,
                        scratch.input_bar,
                        scratch.raw_cell_bar,
                        @view(physical_recurrent[:, cell, block, step - 1]),
                        @view(recurrent_inputs[:, cell, block, step]),
                        cache.cell[cell, block],
                        cache.cell_derivative[cell, block],
                        next_state,
                        scratch.physical_bar,
                        event_bar,
                        0.0f0,
                    )
                end
                _accumulate_cell_raw!(
                    @view(gradient.cell_raw[:, cell, block]),
                    scratch.raw_cell_bar,
                )
                copyto!(
                    @view(scratch.recurrent_input_bar[:, cell, block, step]),
                    scratch.input_bar,
                )
            end
            anchor = @view physical_anchor[:, cell, block]
            event_bar = _local_physical_signal!(
                scratch,
                anchor,
                @view(scratch.anchor_signal[:, cell, block]),
            )
            for state in 1:Model.STATE_DIM
                scratch.physical_bar[state] += scratch.state_bar[state]
            end
            Cell.initial_state!(scratch.initial_state, cache.cell[cell, block])
            Cell.cell_step_pullback!(
                scratch.state_bar,
                scratch.input_bar,
                scratch.raw_cell_bar,
                scratch.initial_state,
                @view(scratch.sensory_input[:, cell, block]),
                cache.cell[cell, block],
                cache.cell_derivative[cell, block],
                anchor,
                scratch.physical_bar,
                event_bar,
                0.0f0,
            )
            Cell.initial_state_pullback!(
                scratch.raw_cell_bar,
                scratch.state_bar,
                cache.cell_derivative[cell, block],
            )
            _accumulate_cell_raw!(
                @view(gradient.cell_raw[:, cell, block]),
                scratch.raw_cell_bar,
            )
            copyto!(
                @view(scratch.sensory_input_bar[:, cell, block]),
                scratch.input_bar,
            )
        end
    end
    Encoder.sensory_cached_raw_vjp!(
        scratch.rail_bar,
        scratch.sensory_gain_bar,
        scratch.sensory_input_bar,
        rails,
        cache.sensory_gain,
        cache.sensory_gain_derivative,
    )
    @inbounds for index in eachindex(gradient.sensory_gain_raw)
        gradient.sensory_gain_raw[index] += scratch.sensory_gain_bar[index]
    end
    return nothing
end

@inline function _load_pending_bar!(scratch, step)
    Graph.clear_inbox!(scratch.pending_bar)
    @inbounds for block in 1:Model.BLOCKS
        for cell in 1:Model.CELLS_PER_BLOCK
            destination = _global_cell(block, cell)
            input_bar = @view scratch.recurrent_input_bar[:, cell, block, step]
            for compartment in 1:Cell.N_COMPARTMENTS
                scratch.pending_bar.ampa[compartment, destination] =
                    input_bar[Cell.input_index(compartment, Cell.INPUT_AMPA)]
                scratch.pending_bar.nmda[compartment, destination] =
                    input_bar[Cell.input_index(compartment, Cell.INPUT_NMDA)]
                scratch.pending_bar.gaba[compartment, destination] =
                    input_bar[Cell.input_index(compartment, Cell.INPUT_GABA)]
            end
        end
    end
    return nothing
end

function _edge_local_credit!(gradient, scratch, model, cache)
    fill!(scratch.strength_bar, 0.0f0)
    fill!(scratch.output_anchor_bar, 0.0f0)
    fill!(scratch.output_recurrent_bar, 0.0f0)
    @inbounds for step in 1:Model.RECURRENT_STEPS
        _load_pending_bar!(scratch, step)
        Graph.clear_payload_ring!(scratch.ring)
        Graph.clear_payload_ring!(scratch.ring_bar)
        if step == 1
            for block in 1:Model.BLOCKS
                for cell in 1:Model.CELLS_PER_BLOCK
                    source = _global_cell(block, cell)
                    EventPayload.payload_channels_cached_unchecked!(
                        @view(scratch.ring.current[:, source]),
                        @view(scratch.encoded_anchor[:, cell, block]),
                        cache.payload_gain,
                    )
                    copyto!(
                        @view(scratch.ring.previous[:, source]),
                        @view(scratch.ring.current[:, source]),
                    )
                end
            end
            Graph.deliver_payloads_vjp!(
                scratch.strength_bar,
                scratch.ring_bar,
                model.graph,
                cache.edge_strength,
                scratch.ring,
                1:Model.TOTAL_CELLS,
                scratch.pending_bar,
            )
            for source in 1:Model.TOTAL_CELLS
                for channel in axes(scratch.ring_bar.current, 1)
                    scratch.ring_bar.current[channel, source] +=
                        scratch.ring_bar.previous[channel, source]
                end
            end
            Primitives.accumulate_anchor_payload_bar!(
                scratch.output_anchor_bar,
                gradient.payload_gain_raw,
                scratch.encoded_anchor,
                scratch.ring_bar.current,
                cache.payload_gain,
                cache.payload_gain_derivative,
            )
        else
            Primitives.set_step_payloads!(
                scratch.ring,
                scratch.recurrent_encoded_views,
                step - 1,
                cache.payload_gain,
                0x00,
                0.0f0,
            )
            Primitives.set_step_payloads!(
                scratch.ring,
                scratch.recurrent_encoded_views,
                step - 2,
                cache.payload_gain,
                0x01,
                0.0f0,
            )
            current_count = Primitives.fill_sources!(
                scratch.current_sources,
                scratch.recurrent_encoded_views,
                step - 1,
                0.0f0,
            )
            previous_count = Primitives.fill_sources!(
                scratch.previous_sources,
                scratch.recurrent_encoded_views,
                step - 2,
                0.0f0,
            )
            active_count = Primitives.merge_sorted_sources!(
                scratch.active_sources,
                scratch.current_sources,
                current_count,
                scratch.previous_sources,
                previous_count,
            )
            Graph.deliver_payloads_vjp!(
                scratch.strength_bar,
                scratch.ring_bar,
                model.graph,
                cache.edge_strength,
                scratch.ring,
                @view(scratch.active_sources[1:active_count]),
                scratch.pending_bar,
            )
            Primitives.accumulate_step_payload_bar!(
                scratch.output_recurrent_bar_views,
                gradient.payload_gain_raw,
                scratch.recurrent_encoded_views,
                step - 1,
                scratch.ring_bar.current,
                cache.payload_gain,
                cache.payload_gain_derivative,
                0.0f0,
                scratch.payload_bar,
                scratch.payload_value,
            )
            Primitives.accumulate_step_payload_bar!(
                scratch.output_recurrent_bar_views,
                gradient.payload_gain_raw,
                scratch.recurrent_encoded_views,
                step - 2,
                scratch.ring_bar.previous,
                cache.payload_gain,
                cache.payload_gain_derivative,
                0.0f0,
                scratch.payload_bar,
                scratch.payload_value,
            )
        end
    end
    NetworkTopology.edge_strength_cached_raw_vjp!(
        gradient.edge_strength_raw,
        cache.edge_strength_derivative,
        scratch.strength_bar,
    )
    return nothing
end

function _output_parameter_gradient!(
    gradient,
    scratch,
    model,
    parameters,
    cache,
    output_cotangent,
    q_learning_signal::AbstractVector{Float32},
    output_subthreshold_scale::Float32,
    q_eprop_scale::Float32,
)
    fill!(scratch.output_anchor_bar, 0.0f0)
    fill!(scratch.output_recurrent_bar, 0.0f0)
    fill!(scratch.output_payload_gain_bar, 0.0f0)
    OutputBank.output_forward!(
        scratch.output_raw,
        scratch.output_trajectory,
        scratch.output_scratch,
        model.output_topology,
        Model._output_parameters(parameters),
        cache.output,
        model.numeric_core,
        cache.payload_gain,
        scratch.encoded_anchor,
        scratch.encoded_recurrent;
        event_floor=0.0f0,
        spike_smoothing=0.0f0,
        collect_q_eligibility=true,
    )
    OutputBank.output_pullback!(
        scratch.output_anchor_bar,
        scratch.output_recurrent_bar,
        Primitives.output_gradient(gradient),
        scratch.output_payload_gain_bar,
        scratch.output_trajectory,
        scratch.output_scratch,
        model.output_topology,
        Model._output_parameters(parameters),
        cache.output,
        cache.payload_gain,
        cache.payload_gain_derivative,
        scratch.encoded_anchor,
        scratch.encoded_recurrent,
        output_cotangent;
        event_floor=0.0f0,
        spike_smoothing=0.0f0,
        subthreshold_credit=output_subthreshold_scale,
    )
    q_afferent_updates = OutputBank.apply_q_error_eprop!(
        Primitives.q_output_gradient(gradient),
        scratch.output_scratch,
        model.output_topology,
        q_learning_signal;
        scale=q_eprop_scale,
    )
    q_cell_updates = OutputBank.apply_q_cell_error_vjp!(
        Primitives.output_gradient(gradient),
        scratch.output_scratch,
        q_learning_signal;
        scale=q_eprop_scale,
    )
    return q_afferent_updates, q_cell_updates
end

function _local_candidate_gradient!(
    gradient::Optimizer.ParameterGradient,
    scratch::LocalCreditScratch,
    feedback::FixedBlockFeedback,
    model::Model.CPUHayModel{Float32},
    prepared::Model.PreparedModelState{Float32},
    rails::AbstractVector{Float32},
    physical_anchor::AbstractArray{Float32,3},
    physical_recurrent::AbstractArray{Float32,4},
    recurrent_inputs::AbstractArray{Float32,4},
    output_cotangent::AbstractVector{Float32},
    q_learning_signal::AbstractVector{Float32},
    config::Config.LocalLearningConfig,
    expected_generation::UInt64,
    train_recurrent::Bool,
)
    slot = Model.assert_generation(prepared, expected_generation)
    parameters = slot.parameters
    cache = slot.cache
    _encode_tape!(scratch, physical_anchor, physical_recurrent)
    q_afferent_updates, q_cell_updates = _output_parameter_gradient!(
        gradient,
        scratch,
        model,
        parameters,
        cache,
        output_cotangent,
        q_learning_signal,
        config.output_subthreshold_scale,
        config.q_eprop_scale,
    )
    if train_recurrent
        decolle_nonzero = _prepare_local_signals!(
            scratch,
            feedback,
            output_cotangent,
            config.decolle_scale,
        )
        _cell_local_credit!(
            gradient,
            scratch,
            cache,
            rails,
            physical_anchor,
            physical_recurrent,
            recurrent_inputs,
        )
        _edge_local_credit!(
            gradient,
            scratch,
            model,
            cache,
        )
        Model.assert_generation(prepared, expected_generation)
        return decolle_nonzero, q_afferent_updates, q_cell_updates
    end
    Model.assert_generation(prepared, expected_generation)
    return 0, q_afferent_updates, q_cell_updates
end

function local_candidate_gradient!(
    gradient::Optimizer.ParameterGradient,
    scratch::LocalCreditScratch,
    feedback::FixedBlockFeedback,
    model::Model.CPUHayModel{Float32},
    prepared::Model.PreparedModelState{Float32},
    rails::AbstractVector{Float32},
    physical_anchor::AbstractArray{Float32,3},
    physical_recurrent::AbstractArray{Float32,4},
    recurrent_inputs::AbstractArray{Float32,4},
    output_cotangent::AbstractVector{Float32},
    q_learning_signal::AbstractVector{Float32};
    config::Config.LocalLearningConfig,
    expected_generation::UInt64=Model.prepared_generation(prepared),
    train_recurrent::Bool=true,
)
    return _local_candidate_gradient!(
        gradient,
        scratch,
        feedback,
        model,
        prepared,
        rails,
        physical_anchor,
        physical_recurrent,
        recurrent_inputs,
        output_cotangent,
        q_learning_signal,
        config,
        expected_generation,
        train_recurrent,
    )
end

mutable struct MechanismCounters
    decolle_signal_nonzero::Int
    q_eprop_updates::Int
    q_cell_updates::Int
end

MechanismCounters() = MechanismCounters(0, 0, 0)

"""The sole mutable owner of canonical route-free local training."""
mutable struct CanonicalTrainer
    model::Model.CPUHayModel{Float32}
    staging::Model.Parameters
    prepared::Model.PreparedModelState{Float32}
    arena::Arena.FixedBatchArena
    gradient::Optimizer.ParameterGradient
    optimizer::Optimizer.AdamWState
    plasticity::Plasticity.PlasticityState
    loss_scratch::Ranking.LossScratch
    config::Config.LocalLearningConfig
    updates::Int
    runtime::Any
    counters::MechanismCounters
end

mutable struct LocalCreditOwner
    trainer::CanonicalTrainer
    batch::Ranking.Batch
    feedback::FixedBlockFeedback
    recurrent_due::Bool
    forward_workers::Vector{Arena.ArenaWorker}
    scratches::Vector{LocalCreditScratch}
    gradients::Vector{Optimizer.ParameterGradient}
    decolle_counts::Vector{Int}
    q_afferent_update_counts::Vector{Int}
    q_cell_update_counts::Vector{Int}
    recurrent_activity::Vector{Matrix{UInt64}}
    output_activity::Vector{Vector{UInt64}}
    q_eprop_signal::Matrix{Float32}
    q_latch_probability::Matrix{Float32}
    q_ordinal_scratch::QOrdinalScratch
    generation::UInt64
    credit_chunk_size::Int
end

struct LocalCreditWorkspace
    owner::LocalCreditOwner
    scheduler::SchedulerRuntime.Scheduler{LocalCreditOwner}
    timing::Vector{Float64}
end

function LocalCreditWorkspace(
    trainer::CanonicalTrainer,
    batch::Ranking.Batch;
    workers::Integer=min(8, Base.Threads.nthreads(:default)),
    feedback::FixedBlockFeedback=FixedBlockFeedback(trainer.config.feedback_seed),
)
    count = Int(workers)
    count >= 1 || throw(ArgumentError("workers must be positive"))
    owner = LocalCreditOwner(
        trainer,
        batch,
        feedback,
        false,
        [Arena.ArenaWorker() for _ in 1:count],
        [LocalCreditScratch() for _ in 1:count],
        [Optimizer.ParameterGradient(trainer.staging) for _ in 1:GRADIENT_REDUCERS],
        zeros(Int, GRADIENT_REDUCERS),
        zeros(Int, GRADIENT_REDUCERS),
        zeros(Int, GRADIENT_REDUCERS),
        [zeros(UInt64, Model.CELLS_PER_BLOCK, Model.BLOCKS) for _ in 1:count],
        [zeros(UInt64, OutputBank.OUTPUT_CELLS) for _ in 1:count],
        zeros(Float32, OutputBank.Q_OUTPUT_CELLS, batch.capacity),
        zeros(Float32, OutputBank.Q_OUTPUT_CELLS, batch.capacity),
        QOrdinalScratch(),
        Arena.arena_generation(trainer.arena),
        1,
    )
    scheduler = SchedulerRuntime.Scheduler(
        owner;
        workers=count,
        queue_capacity=64,
        binding_mode=:none,
    )
    return LocalCreditWorkspace(owner, scheduler, zeros(Float64, 6))
end

function CanonicalTrainer(
    model::Model.CPUHayModel{Float32},
    staging::Model.Parameters,
    prepared::Model.PreparedModelState{Float32},
    batch::Ranking.Batch;
    config::Config.LocalLearningConfig=Config.LocalLearningConfig(),
    workers::Integer=min(8, Base.Threads.nthreads(:default)),
)
    batch.state_batch == Arena.STATE_BATCH || throw(DimensionMismatch(
        "canonical trainer requires state batch $(Arena.STATE_BATCH)",
    ))
    batch.width == Arena.CANDIDATE_WIDTH || throw(DimensionMismatch(
        "canonical trainer requires candidate width $(Arena.CANDIDATE_WIDTH)",
    ))
    trainer = CanonicalTrainer(
        model,
        staging,
        prepared,
        Arena.FixedBatchArena(),
        Optimizer.ParameterGradient(staging),
        Optimizer.AdamWState(staging),
        Plasticity.PlasticityState(config),
        Ranking.LossScratch(batch.width, batch.state_batch),
        config,
        0,
        nothing,
        MechanismCounters(),
    )
    Optimizer.assert_shared_q_state(trainer.optimizer, trainer.staging)
    trainer.runtime = LocalCreditWorkspace(trainer, batch; workers)
    return trainer
end

function _local_candidate_gradient_slot!(
    gradient::Optimizer.ParameterGradient,
    scratch::LocalCreditScratch,
    feedback::FixedBlockFeedback,
    model::Model.CPUHayModel{Float32},
    prepared::Model.PreparedModelState{Float32},
    batch::Ranking.Batch,
    arena::Arena.FixedBatchArena,
    q_eprop_signal::AbstractMatrix{Float32},
    flat::Int,
    config::Config.LocalLearningConfig,
    generation::UInt64,
    train_recurrent::Bool,
)
    state_slot, candidate = Ranking.state_candidate(flat, batch.width)
    return _local_candidate_gradient!(
        gradient,
        scratch,
        feedback,
        model,
        prepared,
        @view(batch.rails[:, flat]),
        @view(arena.physical_anchor[:, :, :, flat]),
        @view(arena.physical_recurrent[:, :, :, :, flat]),
        @view(arena.recurrent_inputs[:, :, :, :, flat]),
        @view(batch.raw_gradient[:, flat]),
        @view(q_eprop_signal[:, flat]),
        config,
        generation,
        train_recurrent,
    )
end

function SchedulerRuntime.dispatch_work!(
    owner::LocalCreditOwner,
    worker_slot::Int,
    item::SchedulerRuntime.WorkItem,
)
    phase = Int(item.phase)
    if phase == 1
        worker = @inbounds owner.forward_workers[worker_slot]
        @inbounds for ordinal in Int(item.first):Int(item.last)
            flat = Int(owner.batch.valid_flats[ordinal])
            Arena.forward_candidate!(
                owner.trainer.arena,
                owner.batch.raw,
                worker,
                flat,
                owner.trainer.model,
                owner.trainer.prepared,
                owner.batch.rails;
                event_floor=0.0f0,
                spike_smoothing=0.0f0,
            )
            copyto!(
                @view(owner.q_latch_probability[:, flat]),
                worker.scratch.output.q_latch_probability,
            )
            Plasticity.accumulate_activity!(
                owner.recurrent_activity[worker_slot],
                owner.output_activity[worker_slot],
                worker.buffers,
            )
        end
    elseif phase == 2
        scratch = @inbounds owner.scratches[worker_slot]
        # Gradient ownership follows a deterministic ordinal partition rather
        # than the worker that happened to dequeue the job.  Work stealing can
        # therefore change execution order without changing Float32 reduction
        # grouping (or first-step Adam sign near exact cancellations).
        reducer = div(Int(item.first) - 1, owner.credit_chunk_size) + 1
        1 <= reducer <= length(owner.gradients) || error(
            "credit reducer index is outside the fixed reducer arena",
        )
        gradient = @inbounds owner.gradients[reducer]
        @inbounds for ordinal in Int(item.first):Int(item.last)
            flat = Int(owner.batch.valid_flats[ordinal])
            decolle_nonzero, q_afferent_updates, q_cell_updates =
                _local_candidate_gradient_slot!(
                gradient,
                scratch,
                owner.feedback,
                owner.trainer.model,
                owner.trainer.prepared,
                owner.batch,
                owner.trainer.arena,
                owner.q_eprop_signal,
                flat,
                owner.trainer.config,
                owner.generation,
                owner.recurrent_due,
            )
            owner.decolle_counts[reducer] += decolle_nonzero
            owner.q_afferent_update_counts[reducer] += q_afferent_updates
            owner.q_cell_update_counts[reducer] += q_cell_updates
        end
    elseif phase == 3
        @inbounds for index in Int(item.first):Int(item.last)
            Optimizer.clear_gradient!(owner.gradients[index])
            owner.decolle_counts[index] = 0
            owner.q_afferent_update_counts[index] = 0
            owner.q_cell_update_counts[index] = 0
        end
    else
        error("unexpected local-credit phase")
    end
    return nothing
end

function with_local_credit_team(action::Function, workspace::LocalCreditWorkspace)
    return SchedulerRuntime.run_team!(workspace.scheduler) do active
        action(active)
    end
end

function _check_batch(trainer::CanonicalTrainer, batch::Ranking.Batch)
    batch.state_batch == Arena.STATE_BATCH || throw(DimensionMismatch(
        "canonical trainer requires state batch $(Arena.STATE_BATCH)",
    ))
    batch.width == Arena.CANDIDATE_WIDTH || throw(DimensionMismatch(
        "canonical trainer requires candidate width $(Arena.CANDIDATE_WIDTH)",
    ))
    workspace = trainer.runtime::LocalCreditWorkspace
    workspace.owner.batch === batch || throw(ArgumentError(
        "canonical trainer owns fixed batch storage; refill that batch in place",
    ))
    return nothing
end

function batch_top1(batch::Ranking.Batch)
    correct = 0
    @inbounds for state in 1:batch.state_batch
        count = Int(batch.counts[state])
        offset = (state - 1) * batch.width
        predicted = 1
        predicted_value = batch.raw[1, offset + 1]
        teacher = 1
        teacher_value = batch.targets.teacher_q[1, state]
        for candidate in 2:count
            candidate_raw = batch.raw[1, offset + candidate]
            if candidate_raw > predicted_value
                predicted = candidate
                predicted_value = candidate_raw
            end
            candidate_teacher = batch.targets.teacher_q[candidate, state]
            if candidate_teacher > teacher_value
                teacher = candidate
                teacher_value = candidate_teacher
            end
        end
        correct += predicted == teacher
    end
    return Float32(correct / batch.state_batch)
end

function batch_tie_aware_top1(
    batch::Ranking.Batch;
    teacher_tolerance::Float32=1.0f-6,
)
    correct = 0
    @inbounds for state in 1:batch.state_batch
        count = Int(batch.counts[state])
        offset = (state - 1) * batch.width
        predicted = 1
        predicted_value = batch.raw[1, offset + 1]
        teacher_value = batch.targets.teacher_q[1, state]
        for candidate in 2:count
            candidate_raw = batch.raw[1, offset + candidate]
            if candidate_raw > predicted_value
                predicted = candidate
                predicted_value = candidate_raw
            end
            teacher_value = max(
                teacher_value,
                batch.targets.teacher_q[candidate, state],
            )
        end
        correct += batch.targets.teacher_q[predicted, state] >=
            teacher_value - teacher_tolerance
    end
    return Float32(correct / batch.state_batch)
end

function mechanism_counts(trainer::CanonicalTrainer)
    plasticity = trainer.plasticity
    return (
        decolle_signal_nonzero=trainer.counters.decolle_signal_nonzero,
        q_eprop_updates=trainer.counters.q_eprop_updates,
        q_cell_updates=trainer.counters.q_cell_updates,
        subthreshold_updates=plasticity.subthreshold_updates,
        nonspiking_updates=plasticity.nonspiking_updates,
        homeostasis_events=plasticity.homeostatic_events,
        synaptic_scaling_events=plasticity.synaptic_scaling_events,
        utility_updates=plasticity.utility_updates,
        rewires=plasticity.rewires,
    )
end

"""Fail closed when a configured canonical mechanism was due but never fired."""
function assert_due_mechanisms!(trainer::CanonicalTrainer)
    counts = mechanism_counts(trainer)
    required = Symbol[:q_eprop_updates, :q_cell_updates]
    recurrent_due = trainer.updates >= trainer.config.recurrent_start_update
    if recurrent_due
        append!(required, (
            :decolle_signal_nonzero,
            :subthreshold_updates,
            :nonspiking_updates,
            :utility_updates,
        ))
    end
    homeostasis_due = trainer.updates > trainer.config.warmup_updates &&
        (trainer.config.recurrent_homeostasis_until >
            trainer.config.warmup_updates ||
         trainer.config.output_homeostasis_until >
            trainer.config.warmup_updates)
    if homeostasis_due
        append!(required, (:homeostasis_events, :synaptic_scaling_events))
    end
    rewiring_due = recurrent_due && trainer.config.maximum_rewires > 0 &&
        trainer.config.structure_until >= trainer.config.recurrent_start_update
    rewiring_due && push!(required, :rewires)
    for name in required
        getproperty(counts, name) > 0 || error(
            "due canonical mechanism `$name` never fired",
        )
    end
    return counts
end

function _train_update!(
    trainer::CanonicalTrainer,
    batch::Ranking.Batch,
    workspace::LocalCreditWorkspace,
    scheduler;
    chunk_size::Integer=0,
)
    _check_batch(trainer, batch)
    owner = workspace.owner
    trainer.updates += 1
    owner.recurrent_due = trainer.config.mode == Config.LEARNING_ACTIVE &&
        trainer.updates >= trainer.config.recurrent_start_update &&
        trainer.updates % trainer.config.recurrent_interval == 0

    started = time_ns()
    owner.generation = Arena.begin_batch!(trainer.arena, trainer.prepared)
    Plasticity.reset_activity_counts!(
        owner.recurrent_activity,
        owner.output_activity,
    )
    requested_chunk = Int(chunk_size)
    requested_chunk >= 0 || throw(ArgumentError(
        "forward chunk size must be nonnegative",
    ))
    forward_chunk_size = requested_chunk == 0 ?
        max(1, cld(batch.valid_count, scheduler.worker_count)) :
        requested_chunk
    SchedulerRuntime.run_phase!(
        scheduler,
        1,
        1,
        batch.valid_count;
        chunk_size=forward_chunk_size,
    )
    Plasticity.update_firing_rates!(
        trainer.plasticity,
        owner.recurrent_activity,
        owner.output_activity,
        batch.valid_count,
    )
    workspace.timing[1] = (time_ns() - started) * 1.0e-9

    started = time_ns()
    loss = Ranking.supervised_loss_and_raw_gradient!(
        batch,
        trainer.loss_scratch,
    )
    prepare_q_eprop_signal!(
        owner.q_eprop_signal,
        batch,
        owner.q_latch_probability;
        label_smoothing=trainer.config.q_label_smoothing,
    )
    add_q_top_ordinal_signal!(
        owner.q_eprop_signal,
        batch,
        owner.q_latch_probability,
        owner.q_ordinal_scratch,
        trainer.config.q_ordinal_weight,
    )
    top1 = batch_top1(batch)
    workspace.timing[2] = (time_ns() - started) * 1.0e-9

    started = time_ns()
    SchedulerRuntime.run_phase!(
        scheduler,
        3,
        1,
        length(owner.gradients);
        chunk_size=1,
    )
    workspace.timing[3] = (time_ns() - started) * 1.0e-9

    started = time_ns()
    owner.credit_chunk_size = max(
        1,
        cld(batch.valid_count, length(owner.gradients)),
    )
    SchedulerRuntime.run_phase!(
        scheduler,
        2,
        1,
        batch.valid_count;
        chunk_size=owner.credit_chunk_size,
    )
    workspace.timing[4] = (time_ns() - started) * 1.0e-9

    started = time_ns()
    Optimizer.clear_gradient!(trainer.gradient)
    for gradient in owner.gradients
        Optimizer.accumulate_gradient!(trainer.gradient, gradient)
    end
    trainer.counters.decolle_signal_nonzero += sum(owner.decolle_counts)
    trainer.counters.q_eprop_updates +=
        sum(owner.q_afferent_update_counts)
    trainer.counters.q_cell_updates += sum(owner.q_cell_update_counts)
    if owner.recurrent_due
        subthreshold_report = Plasticity.apply_subthreshold_eprop!(
            trainer.plasticity,
            trainer.arena,
            batch,
            trainer.model,
            trainer.prepared,
            trainer.gradient,
            owner.feedback.utility_projection,
        )
        subthreshold_report.local_norm > 0.0 &&
            Plasticity.update_structural_utility!(trainer.plasticity)
    end
    workspace.timing[5] = (time_ns() - started) * 1.0e-9

    started = time_ns()
    gradient_norm, clip_scale = Optimizer.apply_adamw!(
        trainer.optimizer,
        trainer.staging,
        trainer.gradient,
        trainer.config;
        phase=owner.recurrent_due ?
            Optimizer.RECURRENT_OPTIMIZATION :
            Optimizer.OUTPUT_OPTIMIZATION,
        q_candidate_count=batch.valid_count,
    )
    Plasticity.apply_intrinsic_homeostasis!(
        trainer.plasticity,
        trainer.staging,
        trainer.optimizer.first,
        trainer.optimizer.second,
        trainer.model,
    )
    Plasticity.apply_utility_rewiring!(
        trainer.plasticity,
        trainer.model,
        trainer.staging,
        trainer.optimizer.first,
        trainer.optimizer.second,
    )
    Optimizer.assert_shared_q_state(trainer.optimizer, trainer.staging)
    Model.publish!(trainer.prepared, trainer.staging)
    workspace.timing[6] = (time_ns() - started) * 1.0e-9
    return loss, top1, gradient_norm, clip_scale
end

"""Canonical public update: hard route-free forward plus local learning."""
function train_update!(trainer::CanonicalTrainer, batch::Ranking.Batch)
    workspace = trainer.runtime::LocalCreditWorkspace
    return with_local_credit_team(workspace) do scheduler
        _train_update!(trainer, batch, workspace, scheduler)
    end
end

end # module CanonicalLocalLearner
