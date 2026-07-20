module SlideSparseOptimizer

"""
Selected-only optimizer support for the SLIDE-style neuron bank.

This module intentionally does **not** implement standard dense AdamW semantics.
The wide bank uses event-time, row-wise AdaGrad with decoupled lazy weight decay:

* only rows present in the sparse gradient accumulator are updated;
* an inactive row's parameter bytes, accumulator, event count, and timestamps are
  untouched by the hot step;
* optional global-time decoupled weight decay is represented by one scalar log
  product and materialized only when a selected row is read or updated; and
* a positive scalar decay of a complete route-key row preserves WTA/SimHash bucket
  membership, so decay alone never makes a row dirty for rehashing.

The v1 correctness baseline fixes wide-bank `weight_decay=0`.  Non-zero bank decay
is rejected unless the caller explicitly promises that every bucket candidate is
materialized *before* the WTA collision-rank `key' * query` is evaluated.  Positive
scaling preserves a hash code, but stale per-row magnitudes do not preserve that
within-bucket dot-product ordering.

The small dense output head has an ordinary dense AdamW implementation.  Combining
that head update with the event-time wide-bank update still does not make the overall
optimizer equivalent to dense AdamW.
"""

export ROUTE_DIMS,
       VALUE_DIMS,
       OUTPUT_DIMS,
       BANK_ROW_DIMS,
       SparseAccessCounters,
       SparseRowGradientAccumulator,
       SparseAdaGradWState,
       TinyDenseAdamWState,
       SparseInvariantSnapshot,
       init_sparse_adagradw,
       init_tiny_dense_adamw,
       accumulate!,
       accumulate_columns!,
       reserve_gradient_records!,
       append_accumulator!,
       reduce_gradients!,
       reset!,
       prepare_selected_rows!,
       materialize_decay!,
       advance_global_decay!,
       sparse_adagradw_step!,
       tiny_dense_adamw_step!,
       dirty_rows,
       take_dirty_rows!,
       decay_requires_rehash,
       reset_counters!,
       snapshot_sparse_invariants,
       assert_inactive_rows_unchanged,
       assert_dirty_subset,
       assert_selected_rows_current,
       assert_sparse_layout

const ROUTE_DIMS = 64
const VALUE_DIMS = 496
const OUTPUT_DIMS = 48
const BANK_ROW_DIMS = ROUTE_DIMS + VALUE_DIMS + OUTPUT_DIMS

@assert BANK_ROW_DIMS == 608

"""Counters whose growth is proportional to selected rows, never bank size."""
Base.@kwdef mutable struct SparseAccessCounters
    rows_read::UInt64 = 0
    rows_written::UInt64 = 0
    theta_elements_read::UInt64 = 0
    theta_elements_written::UInt64 = 0
    decay_rows_materialized::UInt64 = 0
    optimizer_rows_updated::UInt64 = 0
    row_state_writes::UInt64 = 0
    gradient_records_seen::UInt64 = 0
    gradient_rows_reduced::UInt64 = 0
    dirty_route_rows::UInt64 = 0
    dense_parameters_read::UInt64 = 0
    dense_parameters_written::UInt64 = 0
end

function reset_counters!(counters::SparseAccessCounters)
    counters.rows_read = 0
    counters.rows_written = 0
    counters.theta_elements_read = 0
    counters.theta_elements_written = 0
    counters.decay_rows_materialized = 0
    counters.optimizer_rows_updated = 0
    counters.row_state_writes = 0
    counters.gradient_records_seen = 0
    counters.gradient_rows_reduced = 0
    counters.dirty_route_rows = 0
    counters.dense_parameters_read = 0
    counters.dense_parameters_written = 0
    return counters
end

"""
Compact append-only sparse gradient records.

`values` stores exactly 608 contiguous values per record in
`key64, value496, out48` order.  Neither the raw nor reduced representation has a
dimension proportional to the total neuron count.  Reduction uses a stable sort by
row ID, so equal-ID records are summed in insertion order.
"""
mutable struct SparseRowGradientAccumulator{T<:AbstractFloat}
    ids::Vector{Int32}
    values::Vector{T}
    reduced_ids::Vector{Int32}
    reduced_values::Vector{T}
    is_reduced::Bool
    is_consumed::Bool
end

SparseRowGradientAccumulator(; capacity::Integer=0) =
    SparseRowGradientAccumulator(Float32; capacity=capacity)

function SparseRowGradientAccumulator(
    ::Type{T};
    capacity::Integer=0,
) where {T<:AbstractFloat}
    capacity >= 0 || throw(ArgumentError("capacity must be non-negative"))
    ids = Int32[]
    values = T[]
    reduced_ids = Int32[]
    reduced_values = T[]
    sizehint!(ids, capacity)
    sizehint!(values, capacity * BANK_ROW_DIMS)
    sizehint!(reduced_ids, capacity)
    sizehint!(reduced_values, capacity * BANK_ROW_DIMS)
    return SparseRowGradientAccumulator{T}(
        ids,
        values,
        reduced_ids,
        reduced_values,
        false,
        false,
    )
end

function _invalidate_reduction!(accumulator::SparseRowGradientAccumulator)
    accumulator.is_reduced = false
    empty!(accumulator.reduced_ids)
    empty!(accumulator.reduced_values)
    return accumulator
end

function reset!(accumulator::SparseRowGradientAccumulator)
    empty!(accumulator.ids)
    empty!(accumulator.values)
    _invalidate_reduction!(accumulator)
    accumulator.is_consumed = false
    return accumulator
end

"""Reserve compact row-gradient records without scalar `push!` traffic.

The returned integer is the first position in `accumulator.values` belonging to
`row_ids[1]`.  Callers must fill all `length(row_ids) * BANK_ROW_DIMS` reserved
values before reduction.  Record order is exactly the order of `row_ids`, so the
stable reduction has the same floating-point accumulation order as repeated
`accumulate!` calls.
"""
function reserve_gradient_records!(
    accumulator::SparseRowGradientAccumulator,
    row_ids::AbstractVector{<:Integer},
)
    _assert_accumulator_open(accumulator)
    record_count = length(row_ids)
    first_record = length(accumulator.ids) + 1
    first_value = length(accumulator.values) + 1
    resize!(accumulator.ids, length(accumulator.ids) + record_count)
    resize!(accumulator.values, length(accumulator.values) + record_count * BANK_ROW_DIMS)
    @inbounds for offset in 1:record_count
        accumulator.ids[first_record + offset - 1] = _checked_row_id(row_ids[offset])
    end
    _invalidate_reduction!(accumulator)
    return first_value
end

@inline function _assert_accumulator_open(accumulator::SparseRowGradientAccumulator)
    accumulator.is_consumed && error(
        "sparse gradient accumulator was already applied; call reset! before reuse",
    )
    return nothing
end

@inline function _checked_row_id(row_id::Integer)
    1 <= row_id <= typemax(Int32) ||
        throw(ArgumentError("row ID must be in 1:$(typemax(Int32)); got $row_id"))
    return Int32(row_id)
end

function accumulate!(
    accumulator::SparseRowGradientAccumulator{T},
    row_id::Integer,
    gradient::AbstractVector,
) where {T}
    _assert_accumulator_open(accumulator)
    length(gradient) == BANK_ROW_DIMS || throw(DimensionMismatch(
        "a bank-row gradient must have $BANK_ROW_DIMS values; got $(length(gradient))",
    ))
    push!(accumulator.ids, _checked_row_id(row_id))
    @inbounds for value in gradient
        converted = T(value)
        isfinite(converted) || throw(ArgumentError("non-finite sparse gradient"))
        push!(accumulator.values, converted)
    end
    _invalidate_reduction!(accumulator)
    return accumulator
end

function accumulate!(
    accumulator::SparseRowGradientAccumulator{T},
    row_id::Integer,
    key64::AbstractVector,
    value496::AbstractVector,
    out48::AbstractVector,
) where {T}
    _assert_accumulator_open(accumulator)
    length(key64) == ROUTE_DIMS || throw(DimensionMismatch(
        "key gradient must have $ROUTE_DIMS values; got $(length(key64))",
    ))
    length(value496) == VALUE_DIMS || throw(DimensionMismatch(
        "value gradient must have $VALUE_DIMS values; got $(length(value496))",
    ))
    length(out48) == OUTPUT_DIMS || throw(DimensionMismatch(
        "output gradient must have $OUTPUT_DIMS values; got $(length(out48))",
    ))
    push!(accumulator.ids, _checked_row_id(row_id))
    @inbounds for chunk in (key64, value496, out48), value in chunk
        converted = T(value)
        isfinite(converted) || throw(ArgumentError("non-finite sparse gradient"))
        push!(accumulator.values, converted)
    end
    _invalidate_reduction!(accumulator)
    return accumulator
end

"""Append all active columns from a hand-written selected-only VJP."""
function accumulate_columns!(
    accumulator::SparseRowGradientAccumulator,
    row_ids::AbstractVector{<:Integer},
    gradients::AbstractMatrix,
)
    size(gradients, 1) == BANK_ROW_DIMS || throw(DimensionMismatch(
        "gradient matrix must be $BANK_ROW_DIMS x active_rows; got $(size(gradients))",
    ))
    size(gradients, 2) == length(row_ids) || throw(DimensionMismatch(
        "row ID count $(length(row_ids)) != gradient columns $(size(gradients, 2))",
    ))
    @inbounds for column in eachindex(row_ids)
        accumulate!(accumulator, row_ids[column], view(gradients, :, column))
    end
    return accumulator
end

"""
Deterministically append a thread-local accumulator to another accumulator.

Call this in a fixed thread order before `reduce_gradients!`; it never allocates a
bank-sized temporary.
"""
function append_accumulator!(
    destination::SparseRowGradientAccumulator{T},
    source::SparseRowGradientAccumulator,
) where {T}
    _assert_accumulator_open(destination)
    expected = length(source.ids) * BANK_ROW_DIMS
    length(source.values) == expected || error("corrupt source sparse accumulator")
    for record in eachindex(source.ids)
        first_value = (record - 1) * BANK_ROW_DIMS + 1
        last_value = first_value + BANK_ROW_DIMS - 1
        accumulate!(
            destination,
            source.ids[record],
            view(source.values, first_value:last_value),
        )
    end
    return destination
end

function reduce_gradients!(accumulator::SparseRowGradientAccumulator{T}) where {T}
    accumulator.is_reduced && return accumulator
    record_count = length(accumulator.ids)
    length(accumulator.values) == record_count * BANK_ROW_DIMS ||
        error("corrupt sparse accumulator: ID/value counts disagree")

    empty!(accumulator.reduced_ids)
    empty!(accumulator.reduced_values)
    if record_count == 0
        accumulator.is_reduced = true
        return accumulator
    end

    # MergeSort is stable: repeated row IDs are reduced in record insertion order.
    permutation = sortperm(accumulator.ids; alg=Base.Sort.MergeSort)
    previous_id = Int32(0)
    destination_first = 0

    @inbounds for record in permutation
        row_id = accumulator.ids[record]
        if row_id != previous_id
            push!(accumulator.reduced_ids, row_id)
            destination_first = length(accumulator.reduced_values) + 1
            resize!(
                accumulator.reduced_values,
                length(accumulator.reduced_values) + BANK_ROW_DIMS,
            )
            for offset in 0:(BANK_ROW_DIMS - 1)
                accumulator.reduced_values[destination_first + offset] = zero(T)
            end
            previous_id = row_id
        end

        source_first = (record - 1) * BANK_ROW_DIMS + 1
        for offset in 0:(BANK_ROW_DIMS - 1)
            accumulator.reduced_values[destination_first + offset] +=
                accumulator.values[source_first + offset]
        end
    end

    accumulator.is_reduced = true
    return accumulator
end

"""
Event-time state for the wide bank.

`accumulator_sq`, `event_count`, `last_event_step`, and `last_log_decay` are scalar
metadata per neuron.  The hot update indexes these arrays only by reduced active IDs.
There is no 608xN gradient or moment array.
"""
mutable struct SparseAdaGradWState{T<:AbstractFloat}
    accumulator_sq::Vector{T}
    event_count::Vector{UInt64}
    last_event_step::Vector{UInt64}
    last_log_decay::Vector{Float64}
    learning_rate::T
    weight_decay::T
    epsilon::T
    global_step::UInt64
    global_log_decay::Float64
    query_materializes_decay::Bool
    dirty_rows_buffer::Vector{Int32}
    counters::SparseAccessCounters
end

function init_sparse_adagradw(
    theta::AbstractMatrix{T};
    learning_rate::Real=1.0f-2,
    weight_decay::Real=0.0f0,
    epsilon::Real=1.0f-8,
    initial_accumulator::Real=0.0f0,
    query_materializes_decay::Bool=false,
) where {T<:AbstractFloat}
    size(theta, 1) == BANK_ROW_DIMS || throw(DimensionMismatch(
        "theta must be $BANK_ROW_DIMS x N; got $(size(theta))",
    ))
    nrows = size(theta, 2)
    nrows <= typemax(Int32) || throw(ArgumentError("too many neuron rows"))

    lr = T(learning_rate)
    decay = T(weight_decay)
    eps = T(epsilon)
    initial = T(initial_accumulator)
    isfinite(lr) && lr > zero(T) || throw(ArgumentError("learning rate must be finite and positive"))
    isfinite(decay) && decay >= zero(T) || throw(ArgumentError("weight decay must be finite and non-negative"))
    isfinite(eps) && eps > zero(T) || throw(ArgumentError("epsilon must be finite and positive"))
    isfinite(initial) && initial >= zero(T) || throw(ArgumentError(
        "initial accumulator must be finite and non-negative",
    ))
    Float64(lr) * Float64(decay) < 1.0 || throw(ArgumentError(
        "learning_rate * weight_decay must be less than one",
    ))
    decay == zero(T) || query_materializes_decay || throw(ArgumentError(
        "v1 bank weight_decay must be zero unless every retrieved row is " *
        "materialized before WTA dot-product ranking; set " *
        "query_materializes_decay=true only when that query contract is implemented",
    ))

    return SparseAdaGradWState{T}(
        fill(initial, nrows),
        zeros(UInt64, nrows),
        zeros(UInt64, nrows),
        zeros(Float64, nrows),
        lr,
        decay,
        eps,
        0,
        0.0,
        query_materializes_decay,
        Int32[],
        SparseAccessCounters(),
    )
end

function assert_sparse_layout(
    theta::AbstractMatrix,
    state::SparseAdaGradWState,
)
    size(theta, 1) == BANK_ROW_DIMS || throw(DimensionMismatch(
        "theta must have $BANK_ROW_DIMS rows; got $(size(theta, 1))",
    ))
    neurons = size(theta, 2)
    length(state.accumulator_sq) == neurons || throw(DimensionMismatch("accumulator size"))
    length(state.event_count) == neurons || throw(DimensionMismatch("event count size"))
    length(state.last_event_step) == neurons || throw(DimensionMismatch("event timestamp size"))
    length(state.last_log_decay) == neurons || throw(DimensionMismatch("decay timestamp size"))
    return true
end

@inline function _validate_row_id(theta::AbstractMatrix, row_id::Integer)
    1 <= row_id <= size(theta, 2) || throw(BoundsError(theta, (:, row_id)))
    return Int(row_id)
end

"""
Advance the single global decay clock without touching a neuron row.

This is O(1).  In particular, it does not scan `theta`, optimizer state, or row
timestamps.
"""
function advance_global_decay!(
    state::SparseAdaGradWState;
    learning_rate::Real=state.learning_rate,
)
    state.global_step == typemax(UInt64) && error("global optimizer step overflow")
    lr = Float64(learning_rate)
    decay = Float64(state.weight_decay)
    isfinite(lr) && lr > 0.0 || throw(ArgumentError("learning rate must be finite and positive"))
    product = lr * decay
    0.0 <= product < 1.0 || throw(ArgumentError(
        "learning_rate * weight_decay must be in [0, 1)",
    ))
    state.global_log_decay += log1p(-product)
    state.global_step += 1
    return state.global_log_decay
end

"""
Materialize accumulated decoupled decay on one selected row.

The factor is strictly positive.  Scaling all 64 route-key coordinates of a row by
that factor preserves coordinate order and sign, hence WTA/SimHash bucket membership.
It does *not* by itself preserve a bucket's magnitude-sensitive `key' * query` rank
when different rows are physically stale by different amounts.  Consequently v1
uses zero bank decay; an experimental non-zero-decay caller must materialize every
retrieved row before ranking.  This function deliberately does not add the row to
`dirty_rows_buffer`.
"""
function materialize_decay!(
    theta::AbstractMatrix{T},
    state::SparseAdaGradWState{T},
    row_id::Integer,
) where {T<:AbstractFloat}
    assert_sparse_layout(theta, state)
    row = _validate_row_id(theta, row_id)
    delta = state.global_log_decay - state.last_log_decay[row]
    delta <= 8eps(Float64) || error("row decay clock is ahead of global decay clock")
    delta == 0.0 && return false

    factor = exp(delta)
    isfinite(factor) && factor > 0.0 || error("lazy decay factor is not positive finite")
    @inbounds for dimension in 1:BANK_ROW_DIMS
        theta[dimension, row] = T(Float64(theta[dimension, row]) * factor)
    end
    state.last_log_decay[row] = state.global_log_decay

    state.counters.rows_read += 1
    state.counters.rows_written += 1
    state.counters.theta_elements_read += BANK_ROW_DIMS
    state.counters.theta_elements_written += BANK_ROW_DIMS
    state.counters.decay_rows_materialized += 1
    state.counters.row_state_writes += 1
    return true
end

"""
Prepare selected rows before forward computation.

This function records only physical lazy-decay materialization performed by
`materialize_decay!`; it does not pretend that the later neural forward's reads
occur here. Calling it before the hand-written `forward_selected` is part of the
optimizer contract. Duplicate IDs are allowed, but normal routing should supply
unique IDs.
"""
function prepare_selected_rows!(
    theta::AbstractMatrix{T},
    state::SparseAdaGradWState{T},
    row_ids::AbstractVector{<:Integer},
) where {T<:AbstractFloat}
    assert_sparse_layout(theta, state)
    @inbounds for raw_id in row_ids
        row = _validate_row_id(theta, raw_id)
        materialize_decay!(theta, state, row)
    end
    return theta
end

"""Assert that selected forward rows have all prior global decay materialized."""
function assert_selected_rows_current(
    state::SparseAdaGradWState,
    row_ids::AbstractVector{<:Integer},
)
    neurons = length(state.last_log_decay)
    @inbounds for raw_id in row_ids
        1 <= raw_id <= neurons || throw(BoundsError(state.last_log_decay, raw_id))
        state.last_log_decay[raw_id] == state.global_log_decay || error(
            "row $raw_id is stale; call prepare_selected_rows! before forward",
        )
    end
    return true
end

"""
Apply one selected-only wide-bank update.

AdaGrad's scalar row statistic is the cumulative mean squared gradient of that row.
Event counts and timestamps advance only for reduced active rows.  The current step's
decoupled decay is applied lazily before the gradient update.  Returned dirty IDs are
sorted and unique and include only rows whose 64-dimensional route key received a
non-zero gradient; positive scalar decay alone is not dirty.
"""
function sparse_adagradw_step!(
    theta::AbstractMatrix{T},
    state::SparseAdaGradWState{T},
    accumulator::SparseRowGradientAccumulator;
    require_prepared::Bool=true,
    learning_rate::Real=state.learning_rate,
) where {T<:AbstractFloat}
    assert_sparse_layout(theta, state)
    _assert_accumulator_open(accumulator)
    reduce_gradients!(accumulator)
    reduced_ids = accumulator.reduced_ids
    reduced_values = accumulator.reduced_values

    if require_prepared
        assert_selected_rows_current(state, reduced_ids)
    end

    advance_global_decay!(state; learning_rate=learning_rate)
    empty!(state.dirty_rows_buffer)
    state.counters.gradient_records_seen += length(accumulator.ids)
    state.counters.gradient_rows_reduced += length(reduced_ids)

    lr = Float64(learning_rate)
    epsilon = Float64(state.epsilon)
    @inbounds for reduced_column in eachindex(reduced_ids)
        row = _validate_row_id(theta, reduced_ids[reduced_column])
        gradient_first = (reduced_column - 1) * BANK_ROW_DIMS + 1

        # This applies exactly the current global decoupled-decay event to the row.
        # The row was required to be current before advance_global_decay!, so a VJP
        # cannot silently have been computed from an unmaterialized stale parameter.
        materialize_decay!(theta, state, row)

        squared_norm = 0.0
        route_key_changed = false
        for dimension in 1:BANK_ROW_DIMS
            gradient = Float64(reduced_values[gradient_first + dimension - 1])
            isfinite(gradient) || throw(ArgumentError("non-finite reduced gradient"))
            squared_norm += gradient * gradient
            if dimension <= ROUTE_DIMS && gradient != 0.0
                route_key_changed = true
            end
        end
        mean_square = squared_norm / BANK_ROW_DIMS
        new_accumulator = Float64(state.accumulator_sq[row]) + mean_square
        isfinite(new_accumulator) || error("row-wise AdaGrad accumulator overflow")
        state.accumulator_sq[row] = T(new_accumulator)

        state.event_count[row] == typemax(UInt64) && error("row event count overflow")
        state.event_count[row] += 1
        state.last_event_step[row] = state.global_step
        state.counters.row_state_writes += 1

        if squared_norm > 0.0
            inverse_scale = lr / (sqrt(new_accumulator) + epsilon)
            for dimension in 1:BANK_ROW_DIMS
                gradient = Float64(reduced_values[gradient_first + dimension - 1])
                theta[dimension, row] = T(
                    Float64(theta[dimension, row]) - inverse_scale * gradient,
                )
            end
            state.counters.rows_read += 1
            state.counters.rows_written += 1
            state.counters.theta_elements_read += BANK_ROW_DIMS
            state.counters.theta_elements_written += BANK_ROW_DIMS
            state.counters.optimizer_rows_updated += 1

            if route_key_changed
                # reduced_ids are sorted unique, so this is sorted unique as well.
                push!(state.dirty_rows_buffer, Int32(row))
            end
        end
    end

    state.counters.dirty_route_rows += length(state.dirty_rows_buffer)
    accumulator.is_consumed = true
    return state.dirty_rows_buffer
end

dirty_rows(state::SparseAdaGradWState) = state.dirty_rows_buffer

function take_dirty_rows!(state::SparseAdaGradWState)
    result = copy(state.dirty_rows_buffer)
    empty!(state.dirty_rows_buffer)
    return result
end

"""
Positive whole-row lazy decay preserves WTA bucket membership.

This guarantee covers only rank/sign-based WTA or SimHash membership, not the
magnitude-sensitive dot-product rank among retrieved IDs.  Non-zero bank decay is
therefore gated by `query_materializes_decay` at construction.
"""
decay_requires_rehash(::SparseAdaGradWState) = false

"""Dense AdamW state for the tiny 48-to-output head only."""
mutable struct TinyDenseAdamWState{T<:AbstractFloat}
    weight_first_moment::Matrix{T}
    weight_second_moment::Matrix{T}
    bias_first_moment::Vector{T}
    bias_second_moment::Vector{T}
    learning_rate::T
    beta1::T
    beta2::T
    weight_decay::T
    epsilon::T
    decay_bias::Bool
    step::UInt64
end

function init_tiny_dense_adamw(
    weight::AbstractMatrix{T},
    bias::AbstractVector{T};
    learning_rate::Real=1.0f-3,
    beta1::Real=0.9f0,
    beta2::Real=0.999f0,
    weight_decay::Real=1.0f-4,
    epsilon::Real=1.0f-8,
    decay_bias::Bool=false,
) where {T<:AbstractFloat}
    size(weight, 1) == length(bias) || throw(DimensionMismatch(
        "head output dimension and bias length differ",
    ))
    size(weight, 2) == OUTPUT_DIMS || throw(DimensionMismatch(
        "tiny head must consume $OUTPUT_DIMS latent features",
    ))
    lr = T(learning_rate)
    b1 = T(beta1)
    b2 = T(beta2)
    decay = T(weight_decay)
    eps = T(epsilon)
    isfinite(lr) && lr > zero(T) || throw(ArgumentError("learning rate"))
    zero(T) <= b1 < one(T) || throw(ArgumentError("beta1"))
    zero(T) <= b2 < one(T) || throw(ArgumentError("beta2"))
    isfinite(decay) && decay >= zero(T) || throw(ArgumentError("weight decay"))
    isfinite(eps) && eps > zero(T) || throw(ArgumentError("epsilon"))
    Float64(lr) * Float64(decay) < 1.0 || throw(ArgumentError(
        "learning_rate * weight_decay must be less than one",
    ))
    return TinyDenseAdamWState{T}(
        zeros(T, size(weight)),
        zeros(T, size(weight)),
        zeros(T, size(bias)),
        zeros(T, size(bias)),
        lr,
        b1,
        b2,
        decay,
        eps,
        decay_bias,
        0,
    )
end

function tiny_dense_adamw_step!(
    weight::AbstractMatrix{T},
    bias::AbstractVector{T},
    state::TinyDenseAdamWState{T},
    weight_gradient::AbstractMatrix,
    bias_gradient::AbstractVector;
    counters::Union{Nothing,SparseAccessCounters}=nothing,
) where {T<:AbstractFloat}
    (size(weight) == size(weight_gradient) &&
     size(weight) == size(state.weight_first_moment)) ||
        throw(DimensionMismatch("tiny-head weight shapes differ"))
    (length(bias) == length(bias_gradient) &&
     length(bias) == length(state.bias_first_moment)) ||
        throw(DimensionMismatch("tiny-head bias shapes differ"))
    state.step == typemax(UInt64) && error("dense head optimizer step overflow")
    state.step += 1

    lr = Float64(state.learning_rate)
    beta1 = Float64(state.beta1)
    beta2 = Float64(state.beta2)
    epsilon = Float64(state.epsilon)
    decay_factor = 1.0 - lr * Float64(state.weight_decay)
    correction1 = 1.0 - beta1^Float64(state.step)
    correction2 = 1.0 - beta2^Float64(state.step)

    @inbounds for index in eachindex(weight)
        gradient = Float64(weight_gradient[index])
        isfinite(gradient) || throw(ArgumentError("non-finite head gradient"))
        first = beta1 * Float64(state.weight_first_moment[index]) +
                (1.0 - beta1) * gradient
        second = beta2 * Float64(state.weight_second_moment[index]) +
                 (1.0 - beta2) * gradient * gradient
        state.weight_first_moment[index] = T(first)
        state.weight_second_moment[index] = T(second)
        update = (first / correction1) / (sqrt(second / correction2) + epsilon)
        weight[index] = T(decay_factor * Float64(weight[index]) - lr * update)
    end

    bias_decay_factor = state.decay_bias ? decay_factor : 1.0
    @inbounds for index in eachindex(bias)
        gradient = Float64(bias_gradient[index])
        isfinite(gradient) || throw(ArgumentError("non-finite head-bias gradient"))
        first = beta1 * Float64(state.bias_first_moment[index]) +
                (1.0 - beta1) * gradient
        second = beta2 * Float64(state.bias_second_moment[index]) +
                 (1.0 - beta2) * gradient * gradient
        state.bias_first_moment[index] = T(first)
        state.bias_second_moment[index] = T(second)
        update = (first / correction1) / (sqrt(second / correction2) + epsilon)
        bias[index] = T(bias_decay_factor * Float64(bias[index]) - lr * update)
    end

    if counters !== nothing
        parameter_count = length(weight) + length(bias)
        counters.dense_parameters_read += parameter_count
        counters.dense_parameters_written += parameter_count
    end
    return state
end

"""
Diagnostic-only full snapshot for selected-only invariant tests.

This deliberately copies the whole bank and must never be called in a timed or
production training step.  The production hot path has no full-bank scan.
"""
struct SparseInvariantSnapshot{T<:AbstractFloat}
    theta::Matrix{T}
    accumulator_sq::Vector{T}
    event_count::Vector{UInt64}
    last_event_step::Vector{UInt64}
    last_log_decay::Vector{Float64}
end

function snapshot_sparse_invariants(
    theta::AbstractMatrix{T},
    state::SparseAdaGradWState{T},
) where {T<:AbstractFloat}
    assert_sparse_layout(theta, state)
    return SparseInvariantSnapshot{T}(
        Matrix(theta),
        copy(state.accumulator_sq),
        copy(state.event_count),
        copy(state.last_event_step),
        copy(state.last_log_decay),
    )
end

@inline _same_bits(left::Float32, right::Float32) =
    reinterpret(UInt32, left) == reinterpret(UInt32, right)
@inline _same_bits(left::Float64, right::Float64) =
    reinterpret(UInt64, left) == reinterpret(UInt64, right)
@inline _same_bits(left, right) = isequal(left, right)

"""
Diagnostic-only proof that every inactive parameter/state/timestamp byte is unchanged.

The selected set is O(active_rows); the intentional full-bank scan is restricted to
tests and is not part of optimizer timing.
"""
function assert_inactive_rows_unchanged(
    snapshot::SparseInvariantSnapshot,
    theta::AbstractMatrix,
    state::SparseAdaGradWState,
    selected_ids::AbstractVector{<:Integer},
)
    assert_sparse_layout(theta, state)
    size(snapshot.theta) == size(theta) || throw(DimensionMismatch("snapshot theta"))
    selected = Set{Int}(Int(row_id) for row_id in selected_ids)
    neurons = size(theta, 2)
    for row in 1:neurons
        row in selected && continue
        @inbounds for dimension in 1:BANK_ROW_DIMS
            _same_bits(snapshot.theta[dimension, row], theta[dimension, row]) || error(
                "inactive theta row $row changed at dimension $dimension",
            )
        end
        _same_bits(snapshot.accumulator_sq[row], state.accumulator_sq[row]) ||
            error("inactive AdaGrad state changed for row $row")
        snapshot.event_count[row] == state.event_count[row] ||
            error("inactive event count changed for row $row")
        snapshot.last_event_step[row] == state.last_event_step[row] ||
            error("inactive event timestamp changed for row $row")
        _same_bits(snapshot.last_log_decay[row], state.last_log_decay[row]) ||
            error("inactive decay timestamp changed for row $row")
    end
    return true
end

"""Assert that every rehash request belongs to the selected gradient support."""
function assert_dirty_subset(
    state::SparseAdaGradWState,
    selected_ids::AbstractVector{<:Integer},
)
    selected = Set{Int}(Int(row_id) for row_id in selected_ids)
    previous = 0
    for dirty in state.dirty_rows_buffer
        Int(dirty) in selected || error("dirty row $dirty was not selected")
        Int(dirty) > previous || error("dirty rows must be sorted unique")
        previous = Int(dirty)
    end
    return true
end

end # module SlideSparseOptimizer
