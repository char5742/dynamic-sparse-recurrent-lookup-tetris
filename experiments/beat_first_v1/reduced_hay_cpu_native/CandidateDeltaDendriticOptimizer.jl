module CandidateDeltaDendriticOptimizer

using ..DendriticProgramBank
using ..CandidateDeltaDendriticGraph

const Bank = DendriticProgramBank
const Model = CandidateDeltaDendriticGraph

export AdamWConfig,
       AdamWState,
       AdamWStepCounters,
       AdamWStepStats,
       GroupLearningRateMultipliers,
       LearningRateSchedule,
       apply_adamw!,
       gradient_norm,
       learning_rate_at

"""
Pure update-indexed warmup and cosine-to-floor learning-rate schedule.

`update` in [`learning_rate_at`](@ref) is the number of the update being
applied, starting at zero.  The rate rises linearly from zero to
`base_learning_rate` over `warmup_updates`, then decays for exactly
`decay_updates` update intervals.  It remains at
`base_learning_rate * min_ratio` afterwards.  Because the schedule has no
mutable clock, restoring an optimizer at update `u` reproduces the same rate
bit-for-bit by evaluating `learning_rate_at(schedule, u)`.
"""
struct LearningRateSchedule
    base_learning_rate::Float32
    warmup_updates::Int
    decay_updates::Int
    min_ratio::Float32

    function LearningRateSchedule(
        base_learning_rate::Real,
        warmup_updates::Integer,
        decay_updates::Integer,
        min_ratio::Real,
    )
        base = Float32(base_learning_rate)
        floor_ratio = Float32(min_ratio)
        isfinite(base) && base >= 0.0f0 || throw(ArgumentError(
            "base_learning_rate must be finite and non-negative",
        ))
        warmup_updates >= 0 || throw(ArgumentError(
            "warmup_updates must be non-negative",
        ))
        decay_updates > 0 || throw(ArgumentError(
            "decay_updates must be positive",
        ))
        0.0f0 <= floor_ratio <= 1.0f0 || throw(ArgumentError(
            "min_ratio must lie in [0, 1]",
        ))
        warmup_updates <= typemax(Int) || throw(ArgumentError(
            "warmup_updates does not fit Int",
        ))
        decay_updates <= typemax(Int) || throw(ArgumentError(
            "decay_updates does not fit Int",
        ))
        return new(base, Int(warmup_updates), Int(decay_updates), floor_ratio)
    end
end

LearningRateSchedule(
    ;
    base_learning_rate::Real=1.0f-3,
    warmup_updates::Integer=100,
    decay_updates::Integer=100_000,
    min_ratio::Real=0.01f0,
) = LearningRateSchedule(
    base_learning_rate,
    warmup_updates,
    decay_updates,
    min_ratio,
)

"""Return the deterministic learning rate for zero-based `update`."""
@inline function learning_rate_at(
    schedule::LearningRateSchedule,
    update::Integer,
)
    update >= 0 || throw(ArgumentError("update must be non-negative"))
    warmup = schedule.warmup_updates
    if warmup > 0 && update < warmup
        return schedule.base_learning_rate *
               (Float32(update) / Float32(warmup))
    end

    elapsed = update - warmup
    if elapsed >= schedule.decay_updates
        return schedule.base_learning_rate * schedule.min_ratio
    end
    phase = Float32(elapsed) / Float32(schedule.decay_updates)
    cosine_fraction = 0.5f0 * (1.0f0 + cospi(phase))
    ratio = schedule.min_ratio +
            (1.0f0 - schedule.min_ratio) * cosine_fraction
    return schedule.base_learning_rate * ratio
end

"""Learning-rate multipliers for the eleven canonical DDF parameter groups."""
struct GroupLearningRateMultipliers
    leaf_cell::Float32
    forest_internal::Float32
    forest_contact::Float32
    program::Float32
    output_cell::Float32
    output_anchor::Float32
    output_context::Float32
    output_placement::Float32
    output_cascade::Float32
    output_gain::Float32
    output_bias::Float32
end

function GroupLearningRateMultipliers(
    ;
    leaf_cell::Real=0.1,
    forest_internal::Real=0.1,
    forest_contact::Real=1.0,
    program::Real=1.0,
    output_cell::Real=0.1,
    output_anchor::Real=1.0,
    output_context::Real=1.0,
    output_placement::Real=1.0,
    output_cascade::Real=1.0,
    output_gain::Real=1.0,
    output_bias::Real=1.0,
)
    values = Float32.((
        leaf_cell,
        forest_internal,
        forest_contact,
        program,
        output_cell,
        output_anchor,
        output_context,
        output_placement,
        output_cascade,
        output_gain,
        output_bias,
    ))
    @inbounds for index in eachindex(values)
        value = values[index]
        name = fieldnames(GroupLearningRateMultipliers)[index]
        isfinite(value) && value >= 0.0f0 || throw(ArgumentError(
            "$name learning-rate multiplier must be finite and non-negative",
        ))
    end
    return GroupLearningRateMultipliers(values...)
end

"""
Canonical DDF AdamW configuration.

Leaf, forest-internal and output-cell coordinates are bounded raw
biophysical parameters and therefore use `cell_weight_decay` (zero by
default).  Forest contacts, program payloads and all signed output contacts
use ordinary decoupled `weight_decay`.  Output bias is never decayed.
"""
struct AdamWConfig
    learning_rate::Float32
    beta1::Float32
    beta2::Float32
    epsilon::Float32
    clip_norm::Float32
    weight_decay::Float32
    cell_weight_decay::Float32
    multipliers::GroupLearningRateMultipliers
end

function AdamWConfig(
    ;
    learning_rate::Real=1.0f-3,
    beta1::Real=0.9f0,
    beta2::Real=0.999f0,
    epsilon::Real=1.0f-8,
    clip_norm::Real=1.0f0,
    weight_decay::Real=1.0f-4,
    cell_weight_decay::Real=0.0f0,
    multipliers::GroupLearningRateMultipliers=
        GroupLearningRateMultipliers(),
)
    learning_rate_value = Float32(learning_rate)
    beta1_value = Float32(beta1)
    beta2_value = Float32(beta2)
    epsilon_value = Float32(epsilon)
    clip_norm_value = Float32(clip_norm)
    weight_decay_value = Float32(weight_decay)
    cell_weight_decay_value = Float32(cell_weight_decay)

    isfinite(learning_rate_value) && learning_rate_value >= 0.0f0 ||
        throw(ArgumentError("learning_rate must be finite and non-negative"))
    0.0f0 <= beta1_value < 1.0f0 ||
        throw(ArgumentError("beta1 must lie in [0, 1)"))
    0.0f0 <= beta2_value < 1.0f0 ||
        throw(ArgumentError("beta2 must lie in [0, 1)"))
    isfinite(epsilon_value) && epsilon_value > 0.0f0 ||
        throw(ArgumentError("epsilon must be finite and positive"))
    (isfinite(clip_norm_value) || isinf(clip_norm_value)) &&
        clip_norm_value > 0.0f0 ||
        throw(ArgumentError("clip_norm must be positive"))
    for (name, value) in (
        (:weight_decay, weight_decay_value),
        (:cell_weight_decay, cell_weight_decay_value),
    )
        isfinite(value) && value >= 0.0f0 || throw(ArgumentError(
            "$name must be finite and non-negative",
        ))
    end
    return AdamWConfig(
        learning_rate_value,
        beta1_value,
        beta2_value,
        epsilon_value,
        clip_norm_value,
        weight_decay_value,
        cell_weight_decay_value,
        multipliers,
    )
end

"""Dense resident moments mirroring every non-program DDF parameter."""
struct DenseMoments
    leaf_shared_raw::Vector{Float32}
    forest_internal_raw::Matrix{Float32}
    forest_child_contact::Matrix{Float32}
    output_cell_raw::Matrix{Float32}
    output_anchor_weight::Array{Float32,4}
    output_context_weight::Matrix{Float32}
    output_placement_weight::Matrix{Float32}
    output_cascade_weight::Matrix{Float32}
    output_gain::Vector{Float32}
    output_bias::Vector{Float32}
end

function DenseMoments(parameters::Model.ModelParameters)
    return DenseMoments(
        zeros(Float32, size(parameters.leaf_shared_raw)),
        zeros(Float32, size(parameters.forest.internal_raw)),
        zeros(Float32, size(parameters.forest.child_contact)),
        zeros(Float32, size(parameters.output.cell_raw)),
        zeros(Float32, size(parameters.output.anchor_weight)),
        zeros(Float32, size(parameters.output.context_weight)),
        zeros(Float32, size(parameters.output.placement_weight)),
        zeros(Float32, size(parameters.output.cascade_weight)),
        zeros(Float32, size(parameters.output.gain)),
        zeros(Float32, size(parameters.output.bias)),
    )
end

"""
Visible optimizer clocks.

A zero multiplier is a true freeze: parameter, moments and group clock remain
bitwise fixed.  Program rows and sparse placement coordinates additionally
own local clocks for correct late-first-touch Adam bias correction.
"""
mutable struct AdamWStepCounters
    total::Int
    leaf_cell::Int
    forest_internal::Int
    forest_contact::Int
    program_batches::Int
    program_rows::Int
    output_cell::Int
    output_anchor::Int
    output_context::Int
    output_placement::Int
    output_placement_coordinates::Int
    output_cascade::Int
    output_gain::Int
    output_bias::Int
end

AdamWStepCounters() = AdamWStepCounters(
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)

"""
CPU-native optimizer state.

The 208,448-row program bank has resident direct-address moments, but only
rows present in `SparseProgramGradient` are visited.  Placement moments are
dense because the complete 240 x 22 array is small; updates remain sparse and
use one clock per actually touched coordinate.
"""
struct AdamWState
    first::DenseMoments
    second::DenseMoments
    program_first::Matrix{Float32}
    program_second::Matrix{Float32}
    program_step_by_row::Vector{UInt32}
    placement_step_by_coordinate::Matrix{UInt32}
    steps::AdamWStepCounters
end

function AdamWState(parameters::Model.ModelParameters)
    payload_shape = size(parameters.program_bank.payload)
    return AdamWState(
        DenseMoments(parameters),
        DenseMoments(parameters),
        zeros(Float32, payload_shape),
        zeros(Float32, payload_shape),
        zeros(UInt32, payload_shape[2]),
        zeros(UInt32, size(parameters.output.placement_weight)),
        AdamWStepCounters(),
    )
end

struct AdamWStepStats
    gradient_norm::Float64
    clip_scale::Float32
    active_program_rows::Int
end

@inline function _sum_squares(array, scale::Float64)
    total = 0.0
    @inbounds @simd for raw_value in array
        value = Float64(raw_value) * scale
        total = muladd(value, value, total)
    end
    return total
end

function _gradient_norm_squared(
    gradient::Model.ModelGradient,
    multipliers::GroupLearningRateMultipliers,
    scale::Float64,
)
    total = 0.0
    multipliers.leaf_cell > 0.0f0 &&
        (total += _sum_squares(gradient.leaf_shared_raw, scale))
    multipliers.forest_internal > 0.0f0 &&
        (total += _sum_squares(gradient.forest.internal_raw, scale))
    multipliers.forest_contact > 0.0f0 &&
        (total += _sum_squares(gradient.forest.child_contact, scale))
    multipliers.output_cell > 0.0f0 &&
        (total += _sum_squares(gradient.output.cell_raw, scale))
    multipliers.output_anchor > 0.0f0 &&
        (total += _sum_squares(gradient.output.anchor_weight, scale))
    multipliers.output_context > 0.0f0 &&
        (total += _sum_squares(gradient.output.context_weight, scale))
    multipliers.output_placement > 0.0f0 &&
        (total += _sum_squares(gradient.output.placement_weight, scale))
    multipliers.output_cascade > 0.0f0 &&
        (total += _sum_squares(gradient.output.cascade_weight, scale))
    multipliers.output_gain > 0.0f0 &&
        (total += _sum_squares(gradient.output.gain, scale))
    multipliers.output_bias > 0.0f0 &&
        (total += _sum_squares(gradient.output.bias, scale))
    if multipliers.program > 0.0f0
        @inbounds for slot in 1:Bank.active_gradient_count(gradient.program)
            for lane in 1:Bank.PAYLOAD_WIDTH
                value = Float64(gradient.program.values[lane, slot]) * scale
                total = muladd(value, value, total)
            end
        end
    end
    return total
end

"""L2 norm over every enabled canonical DDF gradient group."""
function gradient_norm(
    gradient::Model.ModelGradient;
    gradient_scale::Real=1.0,
    multipliers::GroupLearningRateMultipliers=
        GroupLearningRateMultipliers(),
)
    scale = Float64(gradient_scale)
    isfinite(scale) && scale >= 0.0 || throw(ArgumentError(
        "gradient_scale must be finite and non-negative",
    ))
    value = sqrt(_gradient_norm_squared(gradient, multipliers, scale))
    isfinite(value) || throw(DomainError(value, "gradient norm is not finite"))
    return value
end

@inline function _increment(counter::Int, label::AbstractString)
    counter == typemax(Int) && throw(OverflowError("$label step overflow"))
    return counter + 1
end

@inline function _adamw_dense!(
    parameter,
    gradient,
    first,
    second,
    learning_rate::Float32,
    gradient_scale::Float32,
    weight_decay::Float32,
    beta1::Float32,
    beta2::Float32,
    epsilon::Float32,
    step::Int,
)
    inverse_correction1 = inv(1.0f0 - beta1^step)
    inverse_correction2 = inv(1.0f0 - beta2^step)
    one_minus_beta1 = 1.0f0 - beta1
    one_minus_beta2 = 1.0f0 - beta2
    @inbounds @simd for index in eachindex(parameter)
        value = gradient[index] * gradient_scale
        momentum = muladd(beta1, first[index], one_minus_beta1 * value)
        variance = muladd(
            beta2,
            second[index],
            one_minus_beta2 * value * value,
        )
        first[index] = momentum
        second[index] = variance
        parameter[index] -= learning_rate * (
            momentum * inverse_correction1 /
                (sqrt(variance * inverse_correction2) + epsilon) +
            weight_decay * parameter[index]
        )
    end
    return nothing
end

@inline function _apply_dense_group!(
    parameter,
    gradient,
    first,
    second,
    config::AdamWConfig,
    multiplier::Float32,
    gradient_scale::Float32,
    weight_decay::Float32,
    step::Int,
)
    _adamw_dense!(
        parameter,
        gradient,
        first,
        second,
        config.learning_rate * multiplier,
        gradient_scale,
        weight_decay,
        config.beta1,
        config.beta2,
        config.epsilon,
        step,
    )
    return nothing
end

@inline function _next_program_step!(state::AdamWState, row::Int)
    @inbounds previous = state.program_step_by_row[row]
    previous == typemax(UInt32) && throw(OverflowError(
        "program-row Adam step overflow",
    ))
    next = previous + UInt32(1)
    @inbounds state.program_step_by_row[row] = next
    return Int(next)
end

function _adamw_sparse_program!(
    state::AdamWState,
    parameters::Model.ModelParameters,
    gradient::Model.ModelGradient,
    config::AdamWConfig,
    gradient_scale::Float32,
)
    active = Bank.active_gradient_count(gradient.program)
    active == 0 && return 0
    learning_rate = config.learning_rate * config.multipliers.program
    beta1 = config.beta1
    beta2 = config.beta2
    one_minus_beta1 = 1.0f0 - beta1
    one_minus_beta2 = 1.0f0 - beta2
    @inbounds for slot in 1:active
        row = Int(Bank.active_gradient_row(gradient.program, slot))
        step = _next_program_step!(state, row)
        inverse_correction1 = inv(1.0f0 - beta1^step)
        inverse_correction2 = inv(1.0f0 - beta2^step)
        for lane in 1:Bank.PAYLOAD_WIDTH
            value = gradient.program.values[lane, slot] * gradient_scale
            momentum = muladd(
                beta1,
                state.program_first[lane, row],
                one_minus_beta1 * value,
            )
            variance = muladd(
                beta2,
                state.program_second[lane, row],
                one_minus_beta2 * value * value,
            )
            state.program_first[lane, row] = momentum
            state.program_second[lane, row] = variance
            parameters.program_bank.payload[lane, row] -= learning_rate * (
                momentum * inverse_correction1 /
                    (sqrt(variance * inverse_correction2) + config.epsilon) +
                config.weight_decay *
                    parameters.program_bank.payload[lane, row]
            )
        end
    end
    state.steps.program_batches = _increment(
        state.steps.program_batches,
        "program batch",
    )
    state.steps.program_rows = Base.checked_add(state.steps.program_rows, active)
    return active
end

@inline function _next_placement_step!(state::AdamWState, index::Int)
    @inbounds previous = state.placement_step_by_coordinate[index]
    previous == typemax(UInt32) && throw(OverflowError(
        "placement-coordinate Adam step overflow",
    ))
    next = previous + UInt32(1)
    @inbounds state.placement_step_by_coordinate[index] = next
    return Int(next)
end

function _adamw_sparse_placement!(
    state::AdamWState,
    parameters::Model.ModelParameters,
    gradient::Model.ModelGradient,
    config::AdamWConfig,
    gradient_scale::Float32,
)
    parameter = parameters.output.placement_weight
    value_bar = gradient.output.placement_weight
    first = state.first.output_placement_weight
    second = state.second.output_placement_weight
    learning_rate = config.learning_rate * config.multipliers.output_placement
    beta1 = config.beta1
    beta2 = config.beta2
    one_minus_beta1 = 1.0f0 - beta1
    one_minus_beta2 = 1.0f0 - beta2
    active = 0
    @inbounds for index in eachindex(parameter)
        raw_gradient = value_bar[index]
        iszero(raw_gradient) && continue
        active += 1
        step = _next_placement_step!(state, index)
        inverse_correction1 = inv(1.0f0 - beta1^step)
        inverse_correction2 = inv(1.0f0 - beta2^step)
        value = raw_gradient * gradient_scale
        momentum = muladd(beta1, first[index], one_minus_beta1 * value)
        variance = muladd(
            beta2,
            second[index],
            one_minus_beta2 * value * value,
        )
        first[index] = momentum
        second[index] = variance
        parameter[index] -= learning_rate * (
            momentum * inverse_correction1 /
                (sqrt(variance * inverse_correction2) + config.epsilon) +
            config.weight_decay * parameter[index]
        )
    end
    if active > 0
        state.steps.output_placement = _increment(
            state.steps.output_placement,
            "output-placement batch",
        )
        state.steps.output_placement_coordinates = Base.checked_add(
            state.steps.output_placement_coordinates,
            active,
        )
    end
    return active
end

"""
Apply one globally clipped AdamW update to the canonical DDF tree.

The caller owns gradient averaging through `gradient_scale` and cache refresh.
No gradient is cleared.  Program rows and placement coordinates are updated
only when present in the current sparse gradient support.
"""
function apply_adamw!(
    state::AdamWState,
    parameters::Model.ModelParameters,
    gradient::Model.ModelGradient,
    config::AdamWConfig;
    gradient_scale::Real=1.0,
)
    scale64 = Float64(gradient_scale)
    isfinite(scale64) && scale64 >= 0.0 || throw(ArgumentError(
        "gradient_scale must be finite and non-negative",
    ))
    norm = gradient_norm(
        gradient;
        gradient_scale=scale64,
        multipliers=config.multipliers,
    )
    clip_scale = Float32(min(
        1.0,
        Float64(config.clip_norm) / max(norm, eps(Float64)),
    ))
    effective_scale = Float32(scale64) * clip_scale
    multipliers = config.multipliers
    steps = state.steps
    steps.total = _increment(steps.total, "total optimizer")

    if multipliers.leaf_cell > 0.0f0
        steps.leaf_cell = _increment(steps.leaf_cell, "leaf-cell")
        _apply_dense_group!(
            parameters.leaf_shared_raw,
            gradient.leaf_shared_raw,
            state.first.leaf_shared_raw,
            state.second.leaf_shared_raw,
            config,
            multipliers.leaf_cell,
            effective_scale,
            config.cell_weight_decay,
            steps.leaf_cell,
        )
    end
    if multipliers.forest_internal > 0.0f0
        steps.forest_internal = _increment(
            steps.forest_internal,
            "forest-internal",
        )
        _apply_dense_group!(
            parameters.forest.internal_raw,
            gradient.forest.internal_raw,
            state.first.forest_internal_raw,
            state.second.forest_internal_raw,
            config,
            multipliers.forest_internal,
            effective_scale,
            config.cell_weight_decay,
            steps.forest_internal,
        )
    end
    if multipliers.forest_contact > 0.0f0
        steps.forest_contact = _increment(
            steps.forest_contact,
            "forest-contact",
        )
        _apply_dense_group!(
            parameters.forest.child_contact,
            gradient.forest.child_contact,
            state.first.forest_child_contact,
            state.second.forest_child_contact,
            config,
            multipliers.forest_contact,
            effective_scale,
            config.weight_decay,
            steps.forest_contact,
        )
    end
    active_program_rows = multipliers.program > 0.0f0 ?
        _adamw_sparse_program!(
            state,
            parameters,
            gradient,
            config,
            effective_scale,
        ) : 0
    if multipliers.output_cell > 0.0f0
        steps.output_cell = _increment(steps.output_cell, "output-cell")
        _apply_dense_group!(
            parameters.output.cell_raw,
            gradient.output.cell_raw,
            state.first.output_cell_raw,
            state.second.output_cell_raw,
            config,
            multipliers.output_cell,
            effective_scale,
            config.cell_weight_decay,
            steps.output_cell,
        )
    end
    if multipliers.output_anchor > 0.0f0
        steps.output_anchor = _increment(steps.output_anchor, "output-anchor")
        _apply_dense_group!(
            parameters.output.anchor_weight,
            gradient.output.anchor_weight,
            state.first.output_anchor_weight,
            state.second.output_anchor_weight,
            config,
            multipliers.output_anchor,
            effective_scale,
            config.weight_decay,
            steps.output_anchor,
        )
    end
    if multipliers.output_context > 0.0f0
        steps.output_context = _increment(
            steps.output_context,
            "output-context",
        )
        _apply_dense_group!(
            parameters.output.context_weight,
            gradient.output.context_weight,
            state.first.output_context_weight,
            state.second.output_context_weight,
            config,
            multipliers.output_context,
            effective_scale,
            config.weight_decay,
            steps.output_context,
        )
    end
    multipliers.output_placement > 0.0f0 && _adamw_sparse_placement!(
        state,
        parameters,
        gradient,
        config,
        effective_scale,
    )
    if multipliers.output_cascade > 0.0f0
        steps.output_cascade = _increment(
            steps.output_cascade,
            "output-cascade",
        )
        _apply_dense_group!(
            parameters.output.cascade_weight,
            gradient.output.cascade_weight,
            state.first.output_cascade_weight,
            state.second.output_cascade_weight,
            config,
            multipliers.output_cascade,
            effective_scale,
            config.weight_decay,
            steps.output_cascade,
        )
    end
    if multipliers.output_gain > 0.0f0
        steps.output_gain = _increment(steps.output_gain, "output-gain")
        _apply_dense_group!(
            parameters.output.gain,
            gradient.output.gain,
            state.first.output_gain,
            state.second.output_gain,
            config,
            multipliers.output_gain,
            effective_scale,
            config.weight_decay,
            steps.output_gain,
        )
    end
    if multipliers.output_bias > 0.0f0
        steps.output_bias = _increment(steps.output_bias, "output-bias")
        _apply_dense_group!(
            parameters.output.bias,
            gradient.output.bias,
            state.first.output_bias,
            state.second.output_bias,
            config,
            multipliers.output_bias,
            effective_scale,
            0.0f0,
            steps.output_bias,
        )
    end
    return AdamWStepStats(norm, clip_scale, active_program_rows)
end

end # module CandidateDeltaDendriticOptimizer
