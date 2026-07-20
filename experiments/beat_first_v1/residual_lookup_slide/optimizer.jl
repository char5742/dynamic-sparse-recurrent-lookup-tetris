const LOOKUP_OPTIMIZER_VERSION = 1

"""Active-event AdamW state for one table-major lookup bank.

Moments and bias-correction event time advance only for columns present in the
stably reduced selected support.  Decoupled weight decay follows learner-global
time, but is represented by a scalar log clock and materialized only when a
column is selected.  This is deliberately not dense-equivalent AdamW with
implicit zero-gradient updates.
"""
mutable struct LookupSparseAdamWState
    m::Matrix{Float32}
    v::Matrix{Float32}
    event_count::Vector{UInt64}
    last_event_step::Vector{UInt64}
    last_log_decay::Vector{Float64}
    global_step::UInt64
    global_log_decay::Float64
    beta1::Float32
    beta2::Float32
    epsilon::Float32
    learning_rate::Float32
    weight_decay::Float32
end

function _validated_adamw_hyperparameters(
    beta1::Real,
    beta2::Real,
    epsilon::Real,
    learning_rate::Real,
    weight_decay::Real,
)
    b1 = Float32(beta1)
    b2 = Float32(beta2)
    eps = Float32(epsilon)
    lr = Float32(learning_rate)
    wd = Float32(weight_decay)
    0.0f0 <= b1 < 1.0f0 || throw(ArgumentError("beta1 must be in [0, 1)"))
    0.0f0 <= b2 < 1.0f0 || throw(ArgumentError("beta2 must be in [0, 1)"))
    isfinite(eps) && eps > 0.0f0 || throw(ArgumentError(
        "epsilon must be finite and positive",
    ))
    isfinite(lr) && lr > 0.0f0 || throw(ArgumentError(
        "learning_rate must be finite and positive",
    ))
    isfinite(wd) && wd >= 0.0f0 || throw(ArgumentError(
        "weight_decay must be finite and nonnegative",
    ))
    Float64(lr) * Float64(wd) < 1.0 || throw(ArgumentError(
        "learning_rate * weight_decay must be less than one",
    ))
    return (; beta1=b1, beta2=b2, epsilon=eps, learning_rate=lr, weight_decay=wd)
end

function init_lookup_sparse_adamw(
    theta::Matrix{Float32};
    beta1::Real=0.9f0,
    beta2::Real=0.999f0,
    epsilon::Real=1.0f-8,
    learning_rate::Real=1.0f-4,
    weight_decay::Real=0.0f0,
)
    hyper = _validated_adamw_hyperparameters(
        beta1,
        beta2,
        epsilon,
        learning_rate,
        weight_decay,
    )
    columns = size(theta, 2)
    columns > 0 || throw(ArgumentError("lookup bank must contain columns"))
    return LookupSparseAdamWState(
        zeros(Float32, size(theta)),
        zeros(Float32, size(theta)),
        zeros(UInt64, columns),
        zeros(UInt64, columns),
        zeros(Float64, columns),
        UInt64(0),
        0.0,
        hyper.beta1,
        hyper.beta2,
        hyper.epsilon,
        hyper.learning_rate,
        hyper.weight_decay,
    )
end

function _assert_lookup_sparse_layout(
    theta::Matrix{Float32}, state::LookupSparseAdamWState
)
    size(state.m) == size(theta) || throw(DimensionMismatch(
        "lookup first-moment shape differs from bank",
    ))
    size(state.v) == size(theta) || throw(DimensionMismatch(
        "lookup second-moment shape differs from bank",
    ))
    columns = size(theta, 2)
    length(state.event_count) == columns || throw(DimensionMismatch(
        "lookup event-count width differs from bank",
    ))
    length(state.last_event_step) == columns || throw(DimensionMismatch(
        "lookup event-clock width differs from bank",
    ))
    length(state.last_log_decay) == columns || throw(DimensionMismatch(
        "lookup decay-clock width differs from bank",
    ))
    return nothing
end

@inline function _same_float64_bits(left::Float64, right::Float64)
    return reinterpret(UInt64, left) == reinterpret(UInt64, right)
end

@inline function _same_float32_bits(left::Float32, right::Float32)
    return reinterpret(UInt32, left) == reinterpret(UInt32, right)
end

@inline function logical_decay_scale(
    state::LookupSparseAdamWState, column::Integer
)
    1 <= column <= length(state.last_log_decay) || throw(BoundsError(
        state.last_log_decay,
        column,
    ))
    return exp(state.global_log_decay - state.last_log_decay[column])
end

"""Materialize pending decay for explicitly selected columns only.

All IDs and scales are validated before the first write.  Moments, event
counters, and inactive per-column clocks are untouched.

When bank weight decay is nonzero, this function is a mandatory *pre-gather*
barrier: call it immediately after a block's fixed router has produced its
columns and before reading those bank columns.  Materializing after a completed
forward is not equivalent, because an updated residual can change deeper-block
routes.  Weight-decay-zero callers may omit the barrier because every logical
scale is exactly one.
"""
function materialize_selected_columns!(
    theta::Matrix{Float32},
    state::LookupSparseAdamWState,
    columns::AbstractVector{<:Integer},
)
    _assert_lookup_sparse_layout(theta, state)
    selected = sort!(unique!(Int32[Int32(column) for column in columns]))
    scales = Vector{Float32}(undef, length(selected))
    for position in eachindex(selected)
        column = Int(selected[position])
        1 <= column <= size(theta, 2) || throw(BoundsError(
            theta,
            (Colon(), column),
        ))
        state.last_event_step[column] <= state.global_step || error(
            "lookup column $column has a future event clock",
        )
        state.last_log_decay[column] + 32 * eps(Float64) >=
            state.global_log_decay || error(
                "lookup column $column decay clock is ahead of global time",
            )
        scale = Float32(logical_decay_scale(state, column))
        isfinite(scale) && scale > 0.0f0 || throw(ArgumentError(
            "lookup column $column has invalid pending decay",
        ))
        scales[position] = scale
    end
    @inbounds for position in eachindex(selected)
        column = Int(selected[position])
        scale = scales[position]
        if scale != 1.0f0
            @simd for coordinate in axes(theta, 1)
                theta[coordinate, column] *= scale
            end
        end
        state.last_log_decay[column] = state.global_log_decay
    end
    return theta
end

function _validate_table_major_selection(
    columns::AbstractVector{<:Integer},
    bank_width::Int,
    tables_per_block::Int,
    rows_per_table::Int,
)
    tables_per_block > 0 || throw(ArgumentError("tables_per_block must be positive"))
    rows_per_table > 0 || throw(ArgumentError("rows_per_table must be positive"))
    tables_per_block * rows_per_table == bank_width || throw(DimensionMismatch(
        "table topology does not cover the lookup bank",
    ))
    !isempty(columns) || throw(ArgumentError("selected support must not be empty"))
    length(columns) % tables_per_block == 0 || throw(DimensionMismatch(
        "selected support must contain exactly one column per table per example",
    ))
    @inbounds for position in eachindex(columns)
        column = Int(columns[position])
        table = mod(position - 1, tables_per_block) + 1
        first_column = (table - 1) * rows_per_table + 1
        last_column = table * rows_per_table
        first_column <= column <= last_column || throw(ArgumentError(
            "selected column $column at position $position is outside table $table",
        ))
    end
    return div(length(columns), tables_per_block)
end

"""Stable active-only duplicate reduction.

The first occurrence owns the compact slot, duplicate gradients are summed in
caller order, and the returned support is sorted by flat table-major column ID.
No bank-wide mark array is cleared or scanned.
"""
function _reduce_selected_gradients(
    columns::AbstractVector{<:Integer},
    gradients::AbstractMatrix{Float32},
    bank_width::Int,
)
    size(gradients, 2) == length(columns) || throw(DimensionMismatch(
        "selected gradient count differs from column count",
    ))
    !isempty(columns) || throw(ArgumentError("selected support must not be empty"))
    slots = Dict{Int32,Int}()
    ids = Int32[]
    reduced = zeros(Float32, size(gradients, 1), length(columns))
    for position in eachindex(columns)
        column = Int(columns[position])
        1 <= column <= bank_width || throw(BoundsError(1:bank_width, column))
        id = Int32(column)
        slot = get(slots, id, 0)
        if slot == 0
            push!(ids, id)
            slot = length(ids)
            slots[id] = slot
        end
        @inbounds @simd for coordinate in axes(gradients, 1)
            value = gradients[coordinate, position]
            isfinite(value) || throw(ArgumentError(
                "selected lookup gradient is non-finite",
            ))
            reduced_value = reduced[coordinate, slot] + value
            isfinite(reduced_value) || throw(ArgumentError(
                "duplicate lookup gradient reduction became non-finite",
            ))
            reduced[coordinate, slot] = reduced_value
        end
    end
    permutation = sortperm(ids; alg=Base.Sort.MergeSort)
    sorted_ids = ids[permutation]
    sorted_gradients = copy(view(reduced, :, permutation))
    return sorted_ids, sorted_gradients
end

struct PreparedLookupSparseAdamWStep
    expected_global_step::UInt64
    expected_global_log_decay::Float64
    expected_beta1::Float32
    expected_beta2::Float32
    expected_epsilon::Float32
    expected_learning_rate::Float32
    expected_weight_decay::Float32
    next_global_step::UInt64
    next_global_log_decay::Float64
    columns::Vector{Int32}
    theta::Matrix{Float32}
    m::Matrix{Float32}
    v::Matrix{Float32}
    event_count::Vector{UInt64}
end

struct LookupSparseStepTelemetry
    global_step::UInt64
    input_records::Int
    active_columns::Int
    active_elements::Int
end

function prepare_lookup_sparse_adamw_step(
    theta::Matrix{Float32},
    state::LookupSparseAdamWState,
    columns::AbstractVector{<:Integer},
    gradients::AbstractMatrix{Float32};
    tables_per_block::Int,
    rows_per_table::Int,
    validate_table_groups::Bool=true,
)
    _assert_lookup_sparse_layout(theta, state)
    size(gradients, 1) == size(theta, 1) || throw(DimensionMismatch(
        "selected lookup gradient value dimension differs from bank",
    ))
    if validate_table_groups
        _validate_table_major_selection(
            columns,
            size(theta, 2),
            tables_per_block,
            rows_per_table,
        )
    elseif isempty(columns)
        throw(ArgumentError("selected support must not be empty"))
    end
    ids, reduced = _reduce_selected_gradients(columns, gradients, size(theta, 2))
    state.global_step == typemax(UInt64) && error("lookup optimizer step exhausted")
    next_step = state.global_step + UInt64(1)
    decay_increment = log1p(
        -Float64(state.learning_rate) * Float64(state.weight_decay),
    )
    next_log_decay = state.global_log_decay + decay_increment
    isfinite(next_log_decay) || throw(ArgumentError(
        "lookup global decay clock became non-finite",
    ))

    active = length(ids)
    value_dim = size(theta, 1)
    next_theta = Matrix{Float32}(undef, value_dim, active)
    next_m = Matrix{Float32}(undef, value_dim, active)
    next_v = Matrix{Float32}(undef, value_dim, active)
    next_events = Vector{UInt64}(undef, active)
    for compact_position in 1:active
        column = Int(ids[compact_position])
        state.event_count[column] == typemax(UInt64) && error(
            "lookup event counter exhausted for column $column",
        )
        state.last_event_step[column] <= state.global_step || error(
            "lookup column $column has a future event clock",
        )
        state.last_log_decay[column] + 32 * eps(Float64) >=
            state.global_log_decay || error(
                "lookup column $column decay clock is ahead of global time",
            )
        event = state.event_count[column] + UInt64(1)
        next_events[compact_position] = event
        scale = Float32(exp(next_log_decay - state.last_log_decay[column]))
        isfinite(scale) && scale > 0.0f0 || throw(ArgumentError(
            "lookup pending decay is invalid for column $column",
        ))
        correction1 = 1.0f0 - state.beta1^event
        correction2 = 1.0f0 - state.beta2^event
        correction1 > 0.0f0 && correction2 > 0.0f0 || throw(ArgumentError(
            "lookup Adam bias correction is invalid for column $column",
        ))
        @inbounds for coordinate in 1:value_dim
            gradient = reduced[coordinate, compact_position]
            old_m = state.m[coordinate, column]
            old_v = state.v[coordinate, column]
            updated_m = muladd(
                state.beta1,
                old_m,
                (1.0f0 - state.beta1) * gradient,
            )
            updated_v = muladd(
                state.beta2,
                old_v,
                (1.0f0 - state.beta2) * gradient * gradient,
            )
            decayed_theta = theta[coordinate, column] * scale
            adam = (updated_m / correction1) /
                (sqrt(updated_v / correction2) + state.epsilon)
            updated_theta = decayed_theta - state.learning_rate * adam
            isfinite(updated_m) && isfinite(updated_v) &&
                isfinite(updated_theta) || throw(ArgumentError(
                    "lookup optimizer result is non-finite for column $column",
                ))
            next_m[coordinate, compact_position] = updated_m
            next_v[coordinate, compact_position] = updated_v
            next_theta[coordinate, compact_position] = updated_theta
        end
    end
    return PreparedLookupSparseAdamWStep(
        state.global_step,
        state.global_log_decay,
        state.beta1,
        state.beta2,
        state.epsilon,
        state.learning_rate,
        state.weight_decay,
        next_step,
        next_log_decay,
        ids,
        next_theta,
        next_m,
        next_v,
        next_events,
    )
end

function _assert_lookup_sparse_commit_ready(
    theta::Matrix{Float32},
    state::LookupSparseAdamWState,
    prepared::PreparedLookupSparseAdamWStep,
)
    _assert_lookup_sparse_layout(theta, state)
    state.global_step == prepared.expected_global_step || error(
        "lookup optimizer state changed after prepare",
    )
    _same_float64_bits(
        state.global_log_decay,
        prepared.expected_global_log_decay,
    ) || error("lookup decay clock changed after prepare")
    _same_float32_bits(state.beta1, prepared.expected_beta1) &&
        _same_float32_bits(state.beta2, prepared.expected_beta2) &&
        _same_float32_bits(state.epsilon, prepared.expected_epsilon) &&
        _same_float32_bits(state.learning_rate, prepared.expected_learning_rate) &&
        _same_float32_bits(state.weight_decay, prepared.expected_weight_decay) ||
        error("lookup optimizer hyperparameters changed after prepare")
    prepared.next_global_step == prepared.expected_global_step + UInt64(1) ||
        error("prepared lookup step is not adjacent")
    expected_next_log_decay = prepared.expected_global_log_decay + log1p(
        -Float64(prepared.expected_learning_rate) *
        Float64(prepared.expected_weight_decay),
    )
    isfinite(prepared.next_global_log_decay) && _same_float64_bits(
        prepared.next_global_log_decay,
        expected_next_log_decay,
    ) || error("prepared lookup decay clock is invalid")
    active = length(prepared.columns)
    size(prepared.theta) == (size(theta, 1), active) || throw(DimensionMismatch(
        "prepared lookup theta shape changed",
    ))
    size(prepared.m) == size(prepared.theta) || throw(DimensionMismatch(
        "prepared lookup first-moment shape changed",
    ))
    size(prepared.v) == size(prepared.theta) || throw(DimensionMismatch(
        "prepared lookup second-moment shape changed",
    ))
    length(prepared.event_count) == active || throw(DimensionMismatch(
        "prepared lookup event vector changed",
    ))
    issorted(prepared.columns) || error("prepared lookup support is not sorted")
    allunique(prepared.columns) || error("prepared lookup support is not unique")
    for position in eachindex(prepared.columns)
        column = Int(prepared.columns[position])
        1 <= column <= size(theta, 2) || throw(BoundsError(theta, (Colon(), column)))
        prepared.event_count[position] == state.event_count[column] + UInt64(1) ||
            error("prepared lookup event count is stale")
    end
    all(isfinite, prepared.theta) && all(isfinite, prepared.m) &&
        all(isfinite, prepared.v) &&
        all(value -> value >= 0.0f0, prepared.v) || throw(ArgumentError(
            "prepared lookup update is non-finite or has a negative second moment",
        ))
    return nothing
end

function _commit_lookup_sparse_unchecked!(
    theta::Matrix{Float32},
    state::LookupSparseAdamWState,
    prepared::PreparedLookupSparseAdamWStep,
)
    state.global_step = prepared.next_global_step
    state.global_log_decay = prepared.next_global_log_decay
    for position in eachindex(prepared.columns)
        column = Int(prepared.columns[position])
        @inbounds begin
            copyto!(view(theta, :, column), view(prepared.theta, :, position))
            copyto!(view(state.m, :, column), view(prepared.m, :, position))
            copyto!(view(state.v, :, column), view(prepared.v, :, position))
            state.event_count[column] = prepared.event_count[position]
            state.last_event_step[column] = prepared.next_global_step
            state.last_log_decay[column] = prepared.next_global_log_decay
        end
    end
    return nothing
end

function commit_lookup_sparse_adamw_step!(
    theta::Matrix{Float32},
    state::LookupSparseAdamWState,
    prepared::PreparedLookupSparseAdamWStep;
    input_records::Int=length(prepared.columns),
)
    _assert_lookup_sparse_commit_ready(theta, state, prepared)
    _commit_lookup_sparse_unchecked!(theta, state, prepared)
    return LookupSparseStepTelemetry(
        prepared.next_global_step,
        input_records,
        length(prepared.columns),
        length(prepared.columns) * size(theta, 1),
    )
end

mutable struct DenseHeadAdamWState
    m_head::Matrix{Float32}
    v_head::Matrix{Float32}
    m_bias::Vector{Float32}
    v_bias::Vector{Float32}
    step::UInt64
    beta1::Float32
    beta2::Float32
    epsilon::Float32
    learning_rate::Float32
    weight_decay::Float32
end

mutable struct DenseAlphaAdamWState
    m::Vector{Float32}
    v::Vector{Float32}
    step::UInt64
    beta1::Float32
    beta2::Float32
    epsilon::Float32
    learning_rate::Float32
    weight_decay::Float32
end

function init_dense_head_adamw(
    head::Matrix{Float32},
    bias::Vector{Float32};
    beta1::Real=0.9f0,
    beta2::Real=0.999f0,
    epsilon::Real=1.0f-8,
    learning_rate::Real=1.0f-4,
    weight_decay::Real=0.0f0,
)
    length(bias) == size(head, 1) || throw(DimensionMismatch(
        "head bias width differs from head output dimension",
    ))
    hyper = _validated_adamw_hyperparameters(
        beta1,
        beta2,
        epsilon,
        learning_rate,
        weight_decay,
    )
    return DenseHeadAdamWState(
        zeros(Float32, size(head)),
        zeros(Float32, size(head)),
        zeros(Float32, length(bias)),
        zeros(Float32, length(bias)),
        UInt64(0),
        hyper.beta1,
        hyper.beta2,
        hyper.epsilon,
        hyper.learning_rate,
        hyper.weight_decay,
    )
end

function init_dense_alpha_adamw(
    alpha_logits::Vector{Float32};
    beta1::Real=0.9f0,
    beta2::Real=0.999f0,
    epsilon::Real=1.0f-8,
    learning_rate::Real=1.0f-4,
    weight_decay::Real=0.0f0,
)
    hyper = _validated_adamw_hyperparameters(
        beta1,
        beta2,
        epsilon,
        learning_rate,
        weight_decay,
    )
    return DenseAlphaAdamWState(
        zeros(Float32, length(alpha_logits)),
        zeros(Float32, length(alpha_logits)),
        UInt64(0),
        hyper.beta1,
        hyper.beta2,
        hyper.epsilon,
        hyper.learning_rate,
        hyper.weight_decay,
    )
end

struct PreparedDenseHeadAdamWStep
    expected_step::UInt64
    expected_beta1::Float32
    expected_beta2::Float32
    expected_epsilon::Float32
    expected_learning_rate::Float32
    expected_weight_decay::Float32
    next_step::UInt64
    head::Matrix{Float32}
    bias::Vector{Float32}
    m_head::Matrix{Float32}
    v_head::Matrix{Float32}
    m_bias::Vector{Float32}
    v_bias::Vector{Float32}
end

struct PreparedDenseAlphaAdamWStep
    expected_step::UInt64
    expected_beta1::Float32
    expected_beta2::Float32
    expected_epsilon::Float32
    expected_learning_rate::Float32
    expected_weight_decay::Float32
    next_step::UInt64
    alpha_logits::Vector{Float32}
    m::Vector{Float32}
    v::Vector{Float32}
end

function _prepare_dense_values(
    values::AbstractArray{Float32},
    m::AbstractArray{Float32},
    v::AbstractArray{Float32},
    gradients::AbstractArray{Float32},
    step::UInt64,
    beta1::Float32,
    beta2::Float32,
    epsilon::Float32,
    learning_rate::Float32,
    weight_decay::Float32,
)
    size(m) == size(values) && size(v) == size(values) &&
        size(gradients) == size(values) || throw(DimensionMismatch(
            "dense optimizer array shapes differ",
        ))
    all(isfinite, gradients) || throw(ArgumentError("dense gradient is non-finite"))
    step == typemax(UInt64) && error("dense optimizer step exhausted")
    next_step = step + UInt64(1)
    correction1 = 1.0f0 - beta1^next_step
    correction2 = 1.0f0 - beta2^next_step
    decay = 1.0f0 - learning_rate * weight_decay
    correction1 > 0.0f0 && correction2 > 0.0f0 || throw(ArgumentError(
        "dense Adam bias correction is invalid",
    ))
    isfinite(decay) && decay > 0.0f0 || throw(ArgumentError(
        "dense AdamW decay is invalid",
    ))
    next_values = similar(values)
    next_m = similar(m)
    next_v = similar(v)
    @inbounds for position in eachindex(values, m, v, gradients)
        gradient = gradients[position]
        updated_m = muladd(beta1, m[position], (1.0f0 - beta1) * gradient)
        updated_v = muladd(
            beta2,
            v[position],
            (1.0f0 - beta2) * gradient * gradient,
        )
        updated_value = values[position] * decay - learning_rate *
            (updated_m / correction1) /
            (sqrt(updated_v / correction2) + epsilon)
        isfinite(updated_m) && isfinite(updated_v) &&
            isfinite(updated_value) || throw(ArgumentError(
                "dense optimizer result is non-finite",
            ))
        next_m[position] = updated_m
        next_v[position] = updated_v
        next_values[position] = updated_value
    end
    return next_values, next_m, next_v, next_step
end

function prepare_dense_head_adamw_step(
    head::Matrix{Float32},
    bias::Vector{Float32},
    state::DenseHeadAdamWState,
    dhead::AbstractMatrix{Float32},
    dbias::AbstractVector{Float32},
)
    length(bias) == size(head, 1) || throw(DimensionMismatch(
        "head bias width differs from head output dimension",
    ))
    next_head, next_m_head, next_v_head, next_step = _prepare_dense_values(
        head,
        state.m_head,
        state.v_head,
        dhead,
        state.step,
        state.beta1,
        state.beta2,
        state.epsilon,
        state.learning_rate,
        state.weight_decay,
    )
    # Bias deliberately receives no decoupled decay.
    next_bias, next_m_bias, next_v_bias, bias_step = _prepare_dense_values(
        bias,
        state.m_bias,
        state.v_bias,
        dbias,
        state.step,
        state.beta1,
        state.beta2,
        state.epsilon,
        state.learning_rate,
        0.0f0,
    )
    bias_step == next_step || error("dense head and bias clocks diverged")
    return PreparedDenseHeadAdamWStep(
        state.step,
        state.beta1,
        state.beta2,
        state.epsilon,
        state.learning_rate,
        state.weight_decay,
        next_step,
        next_head,
        next_bias,
        next_m_head,
        next_v_head,
        next_m_bias,
        next_v_bias,
    )
end

function prepare_dense_alpha_adamw_step(
    alpha_logits::Vector{Float32},
    state::DenseAlphaAdamWState,
    gradients::AbstractVector{Float32},
)
    next_alpha, next_m, next_v, next_step = _prepare_dense_values(
        alpha_logits,
        state.m,
        state.v,
        gradients,
        state.step,
        state.beta1,
        state.beta2,
        state.epsilon,
        state.learning_rate,
        state.weight_decay,
    )
    return PreparedDenseAlphaAdamWStep(
        state.step,
        state.beta1,
        state.beta2,
        state.epsilon,
        state.learning_rate,
        state.weight_decay,
        next_step,
        next_alpha,
        next_m,
        next_v,
    )
end

function _assert_dense_head_commit_ready(
    head::Matrix{Float32},
    bias::Vector{Float32},
    state::DenseHeadAdamWState,
    prepared::PreparedDenseHeadAdamWStep,
)
    state.step == prepared.expected_step || error(
        "dense-head state changed after prepare",
    )
    _same_float32_bits(state.beta1, prepared.expected_beta1) &&
        _same_float32_bits(state.beta2, prepared.expected_beta2) &&
        _same_float32_bits(state.epsilon, prepared.expected_epsilon) &&
        _same_float32_bits(state.learning_rate, prepared.expected_learning_rate) &&
        _same_float32_bits(state.weight_decay, prepared.expected_weight_decay) ||
        error("dense-head hyperparameters changed after prepare")
    prepared.next_step == prepared.expected_step + UInt64(1) || error(
        "prepared dense-head step is not adjacent",
    )
    size(prepared.head) == size(head) || throw(DimensionMismatch(
        "prepared head shape changed",
    ))
    length(prepared.bias) == length(bias) || throw(DimensionMismatch(
        "prepared bias shape changed",
    ))
    size(prepared.m_head) == size(head) && size(prepared.v_head) == size(head) ||
        throw(DimensionMismatch("prepared head moment shape changed"))
    length(prepared.m_bias) == length(bias) &&
        length(prepared.v_bias) == length(bias) || throw(DimensionMismatch(
            "prepared bias moment shape changed",
        ))
    all(isfinite, prepared.head) && all(isfinite, prepared.bias) &&
        all(isfinite, prepared.m_head) && all(isfinite, prepared.v_head) &&
        all(isfinite, prepared.m_bias) && all(isfinite, prepared.v_bias) &&
        all(value -> value >= 0.0f0, prepared.v_head) &&
        all(value -> value >= 0.0f0, prepared.v_bias) ||
        throw(ArgumentError(
            "prepared dense-head update is non-finite or has a negative second moment",
        ))
    return nothing
end

function _assert_dense_alpha_commit_ready(
    alpha_logits::Vector{Float32},
    state::DenseAlphaAdamWState,
    prepared::PreparedDenseAlphaAdamWStep,
)
    state.step == prepared.expected_step || error(
        "dense-alpha state changed after prepare",
    )
    _same_float32_bits(state.beta1, prepared.expected_beta1) &&
        _same_float32_bits(state.beta2, prepared.expected_beta2) &&
        _same_float32_bits(state.epsilon, prepared.expected_epsilon) &&
        _same_float32_bits(state.learning_rate, prepared.expected_learning_rate) &&
        _same_float32_bits(state.weight_decay, prepared.expected_weight_decay) ||
        error("dense-alpha hyperparameters changed after prepare")
    prepared.next_step == prepared.expected_step + UInt64(1) || error(
        "prepared dense-alpha step is not adjacent",
    )
    length(prepared.alpha_logits) == length(alpha_logits) ||
        throw(DimensionMismatch("prepared alpha shape changed"))
    length(prepared.m) == length(alpha_logits) &&
        length(prepared.v) == length(alpha_logits) || throw(DimensionMismatch(
            "prepared alpha moment shape changed",
        ))
    all(isfinite, prepared.alpha_logits) && all(isfinite, prepared.m) &&
        all(isfinite, prepared.v) &&
        all(value -> value >= 0.0f0, prepared.v) || throw(ArgumentError(
            "prepared dense-alpha update is non-finite or has a negative second moment",
        ))
    return nothing
end

function _commit_dense_head_unchecked!(
    head::Matrix{Float32},
    bias::Vector{Float32},
    state::DenseHeadAdamWState,
    prepared::PreparedDenseHeadAdamWStep,
)
    copyto!(head, prepared.head)
    copyto!(bias, prepared.bias)
    copyto!(state.m_head, prepared.m_head)
    copyto!(state.v_head, prepared.v_head)
    copyto!(state.m_bias, prepared.m_bias)
    copyto!(state.v_bias, prepared.v_bias)
    state.step = prepared.next_step
    return nothing
end

function _commit_dense_alpha_unchecked!(
    alpha_logits::Vector{Float32},
    state::DenseAlphaAdamWState,
    prepared::PreparedDenseAlphaAdamWStep,
)
    copyto!(alpha_logits, prepared.alpha_logits)
    copyto!(state.m, prepared.m)
    copyto!(state.v, prepared.v)
    state.step = prepared.next_step
    return nothing
end

mutable struct ResidualLookupOptimizerState
    bank_states::NTuple{3,LookupSparseAdamWState}
    head_state::DenseHeadAdamWState
    alpha_state::DenseAlphaAdamWState
    step::UInt64
end

function init_residual_lookup_optimizer(
    model;
    beta1::Real=0.9f0,
    beta2::Real=0.999f0,
    epsilon::Real=1.0f-8,
    bank_learning_rate::Real=1.0f-4,
    bank_weight_decay::Real=0.0f0,
    head_learning_rate::Real=bank_learning_rate,
    head_weight_decay::Real=0.0f0,
    alpha_learning_rate::Real=head_learning_rate,
    alpha_weight_decay::Real=0.0f0,
)
    bank_states = ntuple(3) do layer
        init_lookup_sparse_adamw(
            model.banks[layer];
            beta1,
            beta2,
            epsilon,
            learning_rate=bank_learning_rate,
            weight_decay=bank_weight_decay,
        )
    end
    head_state = init_dense_head_adamw(
        model.head,
        model.bias;
        beta1,
        beta2,
        epsilon,
        learning_rate=head_learning_rate,
        weight_decay=head_weight_decay,
    )
    alpha_state = init_dense_alpha_adamw(
        model.alpha_logits;
        beta1,
        beta2,
        epsilon,
        learning_rate=alpha_learning_rate,
        weight_decay=alpha_weight_decay,
    )
    state = ResidualLookupOptimizerState(
        bank_states,
        head_state,
        alpha_state,
        UInt64(0),
    )
    _assert_residual_lookup_optimizer_layout(model, state)
    return state
end

function _assert_residual_lookup_optimizer_layout(
    model, state::ResidualLookupOptimizerState
)
    length(model.banks) == 3 || throw(DimensionMismatch(
        "Residual Lookup-SLIDE requires three banks",
    ))
    for layer in 1:3
        _assert_lookup_sparse_layout(model.banks[layer], state.bank_states[layer])
        state.bank_states[layer].global_step == state.step || error(
            "lookup bank optimizer clocks diverged",
        )
    end
    size(state.head_state.m_head) == size(model.head) &&
        size(state.head_state.v_head) == size(model.head) || throw(DimensionMismatch(
            "head optimizer shape differs from model",
        ))
    length(state.head_state.m_bias) == length(model.bias) &&
        length(state.head_state.v_bias) == length(model.bias) ||
        throw(DimensionMismatch("bias optimizer shape differs from model"))
    length(state.alpha_state.m) == length(model.alpha_logits) &&
        length(state.alpha_state.v) == length(model.alpha_logits) ||
        throw(DimensionMismatch("alpha optimizer shape differs from model"))
    state.head_state.step == state.step || error("head optimizer clock diverged")
    state.alpha_state.step == state.step || error("alpha optimizer clock diverged")
    return nothing
end

struct PreparedResidualLookupOptimizerStep
    expected_step::UInt64
    next_step::UInt64
    bank_steps::NTuple{3,PreparedLookupSparseAdamWStep}
    bank_input_records::NTuple{3,Int}
    head_step::PreparedDenseHeadAdamWStep
    alpha_step::PreparedDenseAlphaAdamWStep
end

function prepare_optimizer_step(
    model,
    state::ResidualLookupOptimizerState,
    vjp;
    validate_table_groups::Bool=true,
)
    _assert_residual_lookup_optimizer_layout(model, state)
    length(vjp.columns) == 3 && length(vjp.dbanks) == 3 || throw(DimensionMismatch(
        "selected VJP must contain three bank supports",
    ))
    tables = Int(model.tables_per_block)
    tables > 0 || throw(ArgumentError("model table count must be positive"))
    bank_steps = ntuple(3) do layer
        bank = model.banks[layer]
        size(bank, 2) % tables == 0 || throw(DimensionMismatch(
            "lookup bank width is not divisible by table count",
        ))
        rows = div(size(bank, 2), tables)
        prepare_lookup_sparse_adamw_step(
            bank,
            state.bank_states[layer],
            vjp.columns[layer],
            vjp.dbanks[layer];
            tables_per_block=tables,
            rows_per_table=rows,
            validate_table_groups,
        )
    end
    head_step = prepare_dense_head_adamw_step(
        model.head,
        model.bias,
        state.head_state,
        vjp.dhead,
        vjp.dbias,
    )
    alpha_step = prepare_dense_alpha_adamw_step(
        model.alpha_logits,
        state.alpha_state,
        vjp.dalpha_logits,
    )
    next_step = state.step + UInt64(1)
    all(step.next_global_step == next_step for step in bank_steps) || error(
        "prepared bank clocks diverged",
    )
    head_step.next_step == next_step && alpha_step.next_step == next_step || error(
        "prepared dense and sparse clocks diverged",
    )
    return PreparedResidualLookupOptimizerStep(
        state.step,
        next_step,
        bank_steps,
        ntuple(layer -> length(vjp.columns[layer]), 3),
        head_step,
        alpha_step,
    )
end

"""Commit a fully prepared whole-model step after one mutation-free barrier.

Every sparse bank, dense head, and dense alpha state is preflighted before the
first model byte is changed.  The subsequent copy-only commit contains no
shape, finite-value, or stale-state checks that could split the transaction.
"""
function commit_optimizer_step!(
    model,
    state::ResidualLookupOptimizerState,
    prepared::PreparedResidualLookupOptimizerStep,
)
    _assert_residual_lookup_optimizer_layout(model, state)
    state.step == prepared.expected_step || error(
        "whole optimizer state changed after prepare",
    )
    prepared.next_step == prepared.expected_step + UInt64(1) || error(
        "prepared whole-model step is not adjacent",
    )
    for layer in 1:3
        _assert_lookup_sparse_commit_ready(
            model.banks[layer],
            state.bank_states[layer],
            prepared.bank_steps[layer],
        )
    end
    _assert_dense_head_commit_ready(
        model.head,
        model.bias,
        state.head_state,
        prepared.head_step,
    )
    _assert_dense_alpha_commit_ready(
        model.alpha_logits,
        state.alpha_state,
        prepared.alpha_step,
    )

    bank_telemetry = ntuple(3) do layer
        bank_step = prepared.bank_steps[layer]
        _commit_lookup_sparse_unchecked!(
            model.banks[layer],
            state.bank_states[layer],
            bank_step,
        )
        LookupSparseStepTelemetry(
            bank_step.next_global_step,
            prepared.bank_input_records[layer],
            length(bank_step.columns),
            length(bank_step.columns) * size(model.banks[layer], 1),
        )
    end
    _commit_dense_head_unchecked!(
        model.head,
        model.bias,
        state.head_state,
        prepared.head_step,
    )
    _commit_dense_alpha_unchecked!(
        model.alpha_logits,
        state.alpha_state,
        prepared.alpha_step,
    )
    state.step = prepared.next_step
    return (;
        global_step=prepared.next_step,
        banks=bank_telemetry,
        dense_head_elements=length(model.head) + length(model.bias),
        dense_alpha_elements=length(model.alpha_logits),
    )
end

function optimizer_step!(
    model,
    state::ResidualLookupOptimizerState,
    vjp;
    validate_table_groups::Bool=true,
)
    prepared = prepare_optimizer_step(
        model,
        state,
        vjp;
        validate_table_groups,
    )
    return commit_optimizer_step!(model, state, prepared)
end
