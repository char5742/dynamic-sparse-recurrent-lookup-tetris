"""Per-layer active-event AdamW state.

This is intentionally *not* dense-equivalent AdamW.  A neuron's moments and
event counter advance only when that neuron belongs to the stable-reduced
active support.  During inactive gaps no implicit zero-gradient momentum step
occurs.  Decoupled weight decay alone follows learner-global time and is
materialized lazily for a row when it is next scored/selected.
"""
mutable struct EventTimeSparseAdamWState
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
    dirty_ids::Vector{Int32}
end

function init_eventtime_adamw(
    theta::Matrix{Float32};
    beta1::Real=0.9f0,
    beta2::Real=0.999f0,
    epsilon::Real=1.0f-8,
    learning_rate::Real=1.0f-4,
    weight_decay::Real=0.0f0,
)
    b1 = Float32(beta1)
    b2 = Float32(beta2)
    eps = Float32(epsilon)
    lr = Float32(learning_rate)
    wd = Float32(weight_decay)
    0.0f0 <= b1 < 1.0f0 || throw(ArgumentError("beta1 must be in [0,1)"))
    0.0f0 <= b2 < 1.0f0 || throw(ArgumentError("beta2 must be in [0,1)"))
    isfinite(eps) && eps > 0.0f0 || throw(ArgumentError("epsilon must be finite and positive"))
    isfinite(lr) && lr > 0.0f0 || throw(ArgumentError("learning_rate must be finite and positive"))
    isfinite(wd) && wd >= 0.0f0 || throw(ArgumentError("weight_decay must be finite and nonnegative"))
    Float64(lr) * Float64(wd) < 1.0 || throw(ArgumentError(
        "learning_rate * weight_decay must be less than one",
    ))
    neurons = size(theta, 2)
    return EventTimeSparseAdamWState(
        zeros(Float32, size(theta)),
        zeros(Float32, size(theta)),
        zeros(UInt64, neurons),
        zeros(UInt64, neurons),
        zeros(Float64, neurons),
        UInt64(0),
        0.0,
        b1,
        b2,
        eps,
        lr,
        wd,
        Int32[],
    )
end

function _assert_optimizer_layout(theta, state::EventTimeSparseAdamWState)
    size(state.m) == size(theta) || throw(DimensionMismatch("m shape differs from theta"))
    size(state.v) == size(theta) || throw(DimensionMismatch("v shape differs from theta"))
    size(theta, 2) == length(state.event_count) || throw(DimensionMismatch(
        "event_count length differs from bank width",
    ))
    length(state.last_event_step) == size(theta, 2) || throw(DimensionMismatch(
        "last_event_step length differs from bank width",
    ))
    length(state.last_log_decay) == size(theta, 2) || throw(DimensionMismatch(
        "last_log_decay length differs from bank width",
    ))
    return nothing
end

@inline function logical_decay_scale(
    state::EventTimeSparseAdamWState,
    neuron_id::Integer,
)
    1 <= neuron_id <= length(state.last_log_decay) || throw(BoundsError(
        state.last_log_decay,
        neuron_id,
    ))
    return exp(state.global_log_decay - state.last_log_decay[neuron_id])
end

"""Physically apply pending scalar decay to explicitly named rows only.

Positive whole-row scaling cannot change WTA hash codes.  Runtime routing uses
`logical_decay_scale` during exact-dot reranking, then calls this barrier on
the final selected IDs before the neural forward.  No inactive row or its
per-row clock is written.
"""
function materialize_rows!(
    theta::Matrix{Float32},
    state::EventTimeSparseAdamWState,
    ids::AbstractVector{<:Integer},
)
    _assert_optimizer_layout(theta, state)
    unique_ids = sort!(unique!(Int32[Int32(id) for id in ids]))
    for id32 in unique_ids
        id = Int(id32)
        1 <= id <= size(theta, 2) || throw(BoundsError(theta, (Colon(), id)))
        scale = logical_decay_scale(state, id)
        isfinite(scale) && scale > 0.0 || throw(ArgumentError(
            "invalid pending decay scale for neuron $id",
        ))
    end
    @inbounds for id32 in unique_ids
        id = Int(id32)
        scale = Float32(logical_decay_scale(state, id))
        if scale != 1.0f0
            @simd for row in axes(theta, 1)
                theta[row, id] *= scale
            end
        end
        state.last_log_decay[id] = state.global_log_decay
    end
    return theta
end

"""Generation-map compact gradient storage for one sparse layer.

`marks` and `slots` are never cleared across the bank.  Starting a learner
update increments one generation and clears only compact vectors.  Repeated
IDs accumulate into their first slot in caller order; the optimizer later
visits the unique support in ascending neuron-ID order.
"""
mutable struct EventTimeGradientAccumulator
    row_dim::Int
    neurons::Int
    generation::UInt64
    marks::Vector{UInt64}
    slots::Vector{Int32}
    ids::Vector{Int32}
    values::Matrix{Float32}
    used::Int
    order::Vector{Int32}
    sealed::Bool
end

function EventTimeGradientAccumulator(
    row_dim::Integer,
    neurons::Integer;
    initial_capacity::Integer=64,
)
    rows = Int(row_dim)
    count = Int(neurons)
    capacity = max(Int(initial_capacity), 1)
    rows >= 1 || throw(ArgumentError("row_dim must be positive"))
    count >= 1 || throw(ArgumentError("neurons must be positive"))
    return EventTimeGradientAccumulator(
        rows,
        count,
        UInt64(0),
        zeros(UInt64, count),
        zeros(Int32, count),
        Int32[],
        zeros(Float32, rows, capacity),
        0,
        Int32[],
        false,
    )
end

function begin_accumulation!(accumulator::EventTimeGradientAccumulator)
    accumulator.generation == typemax(UInt64) && error(
        "gradient generation exhausted; construct a fresh accumulator",
    )
    accumulator.generation += UInt64(1)
    empty!(accumulator.ids)
    empty!(accumulator.order)
    accumulator.used = 0
    accumulator.sealed = false
    return accumulator
end

function _ensure_accumulator_capacity!(
    accumulator::EventTimeGradientAccumulator,
    required::Int,
)
    required <= size(accumulator.values, 2) && return accumulator
    capacity = max(required, 2 * size(accumulator.values, 2))
    replacement = zeros(Float32, accumulator.row_dim, capacity)
    if accumulator.used > 0
        copyto!(
            view(replacement, :, 1:accumulator.used),
            view(accumulator.values, :, 1:accumulator.used),
        )
    end
    accumulator.values = replacement
    return accumulator
end

function accumulate_row!(
    accumulator::EventTimeGradientAccumulator,
    neuron_id::Integer,
    gradient::AbstractVector{Float32},
)
    accumulator.generation > 0 || error("call begin_accumulation! first")
    accumulator.sealed && error("cannot accumulate after support was sealed")
    id = Int(neuron_id)
    1 <= id <= accumulator.neurons || throw(BoundsError(accumulator.marks, id))
    length(gradient) == accumulator.row_dim || throw(DimensionMismatch(
        "gradient record must have length $(accumulator.row_dim)",
    ))
    all(isfinite, gradient) || throw(ArgumentError("gradient record is non-finite"))

    slot = if @inbounds accumulator.marks[id] == accumulator.generation
        Int(@inbounds accumulator.slots[id])
    else
        accumulator.used += 1
        _ensure_accumulator_capacity!(accumulator, accumulator.used)
        new_slot = accumulator.used
        @inbounds begin
            accumulator.marks[id] = accumulator.generation
            accumulator.slots[id] = Int32(new_slot)
        end
        push!(accumulator.ids, Int32(id))
        fill!(view(accumulator.values, :, new_slot), 0.0f0)
        new_slot
    end
    @inbounds @simd for row in 1:accumulator.row_dim
        accumulator.values[row, slot] += gradient[row]
    end
    all(isfinite, view(accumulator.values, :, slot)) || throw(ArgumentError(
        "duplicate gradient reduction became non-finite",
    ))
    return slot
end

function accumulate_layer_vjp!(
    accumulator::EventTimeGradientAccumulator,
    ids::AbstractVector{Int32},
    gradients::AbstractMatrix{Float32},
)
    size(gradients) == (accumulator.row_dim, length(ids)) || throw(DimensionMismatch(
        "selected gradient matrix shape differs from IDs/row dimension",
    ))
    for position in eachindex(ids)
        accumulate_row!(accumulator, ids[position], view(gradients, :, position))
    end
    return accumulator
end

function sorted_active_slots!(accumulator::EventTimeGradientAccumulator)
    if !accumulator.sealed
        empty!(accumulator.order)
        sizehint!(accumulator.order, accumulator.used)
        for slot in 1:accumulator.used
            push!(accumulator.order, Int32(slot))
        end
        sort!(accumulator.order; by=slot -> accumulator.ids[Int(slot)])
        accumulator.sealed = true
    end
    return accumulator.order
end

struct SparseStepTelemetry
    global_step::UInt64
    active_rows::Int
    active_elements::Int
    dirty_route_rows::Int
    theta_elements_read::Int
    theta_elements_written::Int
    moment_elements_read::Int
    moment_elements_written::Int
end

"""Fully validated compact sparse-bank update, not yet committed."""
struct PreparedSparseAdamWStep
    next_step::UInt64
    next_global_log_decay::Float64
    ids::Vector{Int32}
    theta::Matrix{Float32}
    m::Matrix{Float32}
    v::Matrix{Float32}
    event_count::Vector{UInt64}
    route_changed::BitVector
end

"""Commit one mutation-atomic active-event AdamW learner step.

All compact results are calculated and checked before `theta`, moments, or
clocks are mutated.  Duplicate records have already been stably reduced to
one event per row.  Decay is applied before the Adam update, matching
`theta=(1-lr*wd)*theta-lr*adam` for the row's logical global-time decay while
moments follow only its active-event subsequence.
"""
function prepare_eventtime_adamw_step(
    theta::Matrix{Float32},
    state::EventTimeSparseAdamWState,
    accumulator::EventTimeGradientAccumulator,
)
    _assert_optimizer_layout(theta, state)
    size(theta, 1) == accumulator.row_dim || throw(DimensionMismatch(
        "accumulator row dimension differs from theta",
    ))
    size(theta, 2) == accumulator.neurons || throw(DimensionMismatch(
        "accumulator bank width differs from theta",
    ))
    state.global_step == typemax(UInt64) && error("optimizer global step exhausted")
    order = sorted_active_slots!(accumulator)
    next_step = state.global_step + UInt64(1)
    decay_increment = log1p(-Float64(state.learning_rate) * Float64(state.weight_decay))
    next_global_log_decay = state.global_log_decay + decay_increment
    isfinite(next_global_log_decay) || throw(ArgumentError("decay clock became non-finite"))

    active = length(order)
    row_dim = size(theta, 1)
    next_theta = Matrix{Float32}(undef, row_dim, active)
    next_m = Matrix{Float32}(undef, row_dim, active)
    next_v = Matrix{Float32}(undef, row_dim, active)
    next_events = Vector{UInt64}(undef, active)
    ids = Vector{Int32}(undef, active)
    route_changed = falses(active)

    for compact_position in 1:active
        slot = Int(order[compact_position])
        id = Int(accumulator.ids[slot])
        ids[compact_position] = Int32(id)
        state.event_count[id] == typemax(UInt64) && error(
            "event counter exhausted for neuron $id",
        )
        event = state.event_count[id] + UInt64(1)
        next_events[compact_position] = event
        decay_scale = Float32(exp(
            next_global_log_decay - state.last_log_decay[id],
        ))
        isfinite(decay_scale) && decay_scale > 0.0f0 || throw(ArgumentError(
            "pending decay scale is invalid for neuron $id",
        ))
        correction1 = 1.0f0 - state.beta1^event
        correction2 = 1.0f0 - state.beta2^event
        correction1 > 0.0f0 && correction2 > 0.0f0 || throw(ArgumentError(
            "invalid Adam bias correction for neuron $id",
        ))

        @inbounds for row in 1:row_dim
            gradient = accumulator.values[row, slot]
            old_m = state.m[row, id]
            old_v = state.v[row, id]
            updated_m = muladd(state.beta1, old_m, (1.0f0 - state.beta1) * gradient)
            updated_v = muladd(
                state.beta2,
                old_v,
                (1.0f0 - state.beta2) * gradient * gradient,
            )
            decayed_theta = theta[row, id] * decay_scale
            adam = (updated_m / correction1) /
                (sqrt(updated_v / correction2) + state.epsilon)
            updated_theta = decayed_theta - state.learning_rate * adam
            isfinite(updated_m) && isfinite(updated_v) && isfinite(updated_theta) ||
                throw(ArgumentError("non-finite optimizer result for neuron $id"))
            next_m[row, compact_position] = updated_m
            next_v[row, compact_position] = updated_v
            next_theta[row, compact_position] = updated_theta
            if row <= ROUTE_DIM
                route_changed[compact_position] |=
                    reinterpret(UInt32, updated_theta) != reinterpret(UInt32, decayed_theta)
            end
        end
    end

    return PreparedSparseAdamWStep(
        next_step,
        next_global_log_decay,
        ids,
        next_theta,
        next_m,
        next_v,
        next_events,
        route_changed,
    )
end

function commit_eventtime_adamw_step!(
    theta::Matrix{Float32},
    state::EventTimeSparseAdamWState,
    prepared::PreparedSparseAdamWStep,
)
    state.global_step = prepared.next_step
    state.global_log_decay = prepared.next_global_log_decay
    empty!(state.dirty_ids)
    active = length(prepared.ids)
    row_dim = size(theta, 1)
    for compact_position in 1:active
        id32 = prepared.ids[compact_position]
        id = Int(id32)
        @inbounds begin
            copyto!(view(theta, :, id), view(prepared.theta, :, compact_position))
            copyto!(view(state.m, :, id), view(prepared.m, :, compact_position))
            copyto!(view(state.v, :, id), view(prepared.v, :, compact_position))
            state.event_count[id] = prepared.event_count[compact_position]
            state.last_event_step[id] = prepared.next_step
            state.last_log_decay[id] = prepared.next_global_log_decay
        end
        prepared.route_changed[compact_position] && push!(state.dirty_ids, id32)
    end
    return SparseStepTelemetry(
        prepared.next_step,
        active,
        active * row_dim,
        length(state.dirty_ids),
        active * row_dim,
        active * row_dim,
        2 * active * row_dim,
        2 * active * row_dim,
    )
end

function eventtime_adamw_step!(
    theta::Matrix{Float32},
    state::EventTimeSparseAdamWState,
    accumulator::EventTimeGradientAccumulator,
)
    prepared = prepare_eventtime_adamw_step(theta, state, accumulator)
    return commit_eventtime_adamw_step!(theta, state, prepared)
end

mutable struct DenseHeadAdamWState
    m_weight::Matrix{Float32}
    v_weight::Matrix{Float32}
    m_bias::Vector{Float32}
    v_bias::Vector{Float32}
    step::UInt64
    beta1::Float32
    beta2::Float32
    epsilon::Float32
    learning_rate::Float32
    weight_decay::Float32
end

"""Fully validated dense-head update, not yet committed."""
struct PreparedDenseHeadAdamWStep
    weight::Matrix{Float32}
    bias::Vector{Float32}
    m_weight::Matrix{Float32}
    v_weight::Matrix{Float32}
    m_bias::Vector{Float32}
    v_bias::Vector{Float32}
    next_step::UInt64
end

function init_dense_head_adamw(
    weight::Matrix{Float32},
    bias::Vector{Float32};
    beta1::Real=0.9f0,
    beta2::Real=0.999f0,
    epsilon::Real=1.0f-8,
    learning_rate::Real=1.0f-4,
    weight_decay::Real=0.0f0,
)
    length(bias) == size(weight, 1) || throw(DimensionMismatch(
        "bias length must equal head output width",
    ))
    return DenseHeadAdamWState(
        zeros(Float32, size(weight)),
        zeros(Float32, size(weight)),
        zeros(Float32, length(bias)),
        zeros(Float32, length(bias)),
        UInt64(0),
        Float32(beta1),
        Float32(beta2),
        Float32(epsilon),
        Float32(learning_rate),
        Float32(weight_decay),
    )
end

function prepare_dense_head_adamw_step(
    weight::Matrix{Float32},
    bias::Vector{Float32},
    state::DenseHeadAdamWState,
    dweight::AbstractMatrix{Float32},
    dbias::AbstractVector{Float32},
)
    size(dweight) == size(weight) || throw(DimensionMismatch("head gradient shape differs"))
    length(dbias) == length(bias) || throw(DimensionMismatch("bias gradient shape differs"))
    size(state.m_weight) == size(weight) || throw(DimensionMismatch(
        "head first-moment shape differs",
    ))
    size(state.v_weight) == size(weight) || throw(DimensionMismatch(
        "head second-moment shape differs",
    ))
    length(state.m_bias) == length(bias) || throw(DimensionMismatch(
        "bias first-moment shape differs",
    ))
    length(state.v_bias) == length(bias) || throw(DimensionMismatch(
        "bias second-moment shape differs",
    ))
    all(isfinite, dweight) && all(isfinite, dbias) || throw(ArgumentError(
        "dense-head gradient is non-finite",
    ))
    state.step == typemax(UInt64) && error("dense-head optimizer step exhausted")
    next_step = state.step + UInt64(1)
    correction1 = 1.0f0 - state.beta1^next_step
    correction2 = 1.0f0 - state.beta2^next_step
    decay = 1.0f0 - state.learning_rate * state.weight_decay
    isfinite(correction1) && correction1 > 0.0f0 || throw(ArgumentError(
        "invalid dense-head first-moment correction",
    ))
    isfinite(correction2) && correction2 > 0.0f0 || throw(ArgumentError(
        "invalid dense-head second-moment correction",
    ))
    isfinite(decay) && decay > 0.0f0 || throw(ArgumentError(
        "invalid dense-head decay factor",
    ))
    next_weight = similar(weight)
    next_bias = similar(bias)
    next_m_weight = similar(state.m_weight)
    next_v_weight = similar(state.v_weight)
    next_m_bias = similar(state.m_bias)
    next_v_bias = similar(state.v_bias)
    @inbounds for position in eachindex(weight, dweight)
        gradient = dweight[position]
        updated_m = muladd(
            state.beta1,
            state.m_weight[position],
            (1.0f0 - state.beta1) * gradient,
        )
        updated_v = muladd(
            state.beta2,
            state.v_weight[position],
            (1.0f0 - state.beta2) * gradient * gradient,
        )
        updated_weight = weight[position] * decay - state.learning_rate *
            (updated_m / correction1) /
            (sqrt(updated_v / correction2) + state.epsilon)
        isfinite(updated_m) && isfinite(updated_v) && isfinite(updated_weight) ||
            throw(ArgumentError("non-finite dense-head optimizer result"))
        next_m_weight[position] = updated_m
        next_v_weight[position] = updated_v
        next_weight[position] = updated_weight
    end
    @inbounds for position in eachindex(bias, dbias)
        gradient = dbias[position]
        updated_m = muladd(
            state.beta1,
            state.m_bias[position],
            (1.0f0 - state.beta1) * gradient,
        )
        updated_v = muladd(
            state.beta2,
            state.v_bias[position],
            (1.0f0 - state.beta2) * gradient * gradient,
        )
        # Bias deliberately has no decoupled weight decay.
        updated_bias = bias[position] - state.learning_rate *
            (updated_m / correction1) /
            (sqrt(updated_v / correction2) + state.epsilon)
        isfinite(updated_m) && isfinite(updated_v) && isfinite(updated_bias) ||
            throw(ArgumentError("non-finite dense-bias optimizer result"))
        next_m_bias[position] = updated_m
        next_v_bias[position] = updated_v
        next_bias[position] = updated_bias
    end
    return PreparedDenseHeadAdamWStep(
        next_weight,
        next_bias,
        next_m_weight,
        next_v_weight,
        next_m_bias,
        next_v_bias,
        next_step,
    )
end

function commit_dense_head_adamw_step!(
    weight::Matrix{Float32},
    bias::Vector{Float32},
    state::DenseHeadAdamWState,
    prepared::PreparedDenseHeadAdamWStep,
)
    copyto!(weight, prepared.weight)
    copyto!(bias, prepared.bias)
    copyto!(state.m_weight, prepared.m_weight)
    copyto!(state.v_weight, prepared.v_weight)
    copyto!(state.m_bias, prepared.m_bias)
    copyto!(state.v_bias, prepared.v_bias)
    state.step = prepared.next_step
    return state
end

function dense_head_adamw_step!(
    weight::Matrix{Float32},
    bias::Vector{Float32},
    state::DenseHeadAdamWState,
    dweight::AbstractMatrix{Float32},
    dbias::AbstractVector{Float32},
)
    prepared = prepare_dense_head_adamw_step(
        weight,
        bias,
        state,
        dweight,
        dbias,
    )
    return commit_dense_head_adamw_step!(weight, bias, state, prepared)
end
