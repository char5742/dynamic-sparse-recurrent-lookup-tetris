module CanonicalOptimizer

using LinearAlgebra
using ..Architecture
using ..OutputCellBank
using ..ReducedHayCPUNativeModel
using ..LearningConfig

export ParameterGradient,
    AdamWState,
    OptimizationPhase,
    OUTPUT_OPTIMIZATION,
    RECURRENT_OPTIMIZATION,
    clear_gradient!,
    accumulate_gradient!,
    gradient_norm,
    assert_shared_q_state,
    apply_adamw!

const Model = ReducedHayCPUNativeModel
const Config = LearningConfig

struct ParameterGradient
    cell_raw::Array{Float32,3}
    sensory_gain_raw::Matrix{Float32}
    edge_strength_raw::Array{Float32,3}
    payload_gain_raw::Vector{Float32}
    output_cell_raw::Matrix{Float32}
    output_edge_raw::Matrix{Float32}
    output_q_edge_raw::Matrix{Float32}
    output_q_basal_bias_raw::Vector{Float32}
    output_gain::Matrix{Float32}
    output_bias::Vector{Float32}
end

function ParameterGradient(parameters::Model.Parameters)
    return ParameterGradient(
        zeros(Float32, size(parameters.cell_raw)),
        zeros(Float32, size(parameters.sensory_gain_raw)),
        zeros(Float32, size(parameters.edge_strength_raw)),
        zeros(Float32, size(parameters.payload_gain_raw)),
        zeros(Float32, size(parameters.output_cell_raw)),
        zeros(Float32, size(parameters.output_edge_raw)),
        zeros(Float32, size(parameters.output_edge_raw)),
        zeros(Float32, size(parameters.output_q_basal_bias_raw)),
        zeros(Float32, size(parameters.output_gain)),
        zeros(Float32, size(parameters.output_bias)),
    )
end

function clear_gradient!(gradient::ParameterGradient)
    for field in fieldnames(ParameterGradient)
        fill!(getfield(gradient, field), 0.0f0)
    end
    return gradient
end

function accumulate_gradient!(destination::ParameterGradient, source::ParameterGradient)
    for field in fieldnames(ParameterGradient)
        left = getfield(destination, field)
        right = getfield(source, field)
        @inbounds @simd for index in eachindex(left)
            left[index] += right[index]
        end
    end
    return destination
end

function gradient_norm(gradient::ParameterGradient, fields=fieldnames(ParameterGradient))
    total = 0.0
    for field in fields
        array = getfield(gradient, field)
        @inbounds @simd for index in eachindex(array)
            value = Float64(array[index])
            total = muladd(value, value, total)
        end
    end
    return sqrt(total)
end

"""Norm of the exact shared-Q gradient after candidate and bit averaging."""
function _q_mean_gradient_norm(
    gradient::ParameterGradient,
    candidate_count::Int,
)
    candidate_count > 0 || throw(ArgumentError(
        "Q candidate count must be positive",
    ))
    inverse_candidates = inv(Float64(candidate_count))
    total = 0.0
    @inbounds for source in axes(gradient.output_q_edge_raw, 2)
        for relation in 1:OutputCellBank.Q_FANOUT_PER_SOURCE
            value = Float64(
                gradient.output_q_edge_raw[relation, source],
            ) * inverse_candidates
            total = muladd(value, value, total)
        end
    end
    @inbounds for value_raw in gradient.output_q_basal_bias_raw
        value = Float64(value_raw) * inverse_candidates
        total = muladd(value, value, total)
    end
    inverse_cells = inv(Float64(Architecture.Q_OUTPUT_CELL_COUNT))
    @inbounds for parameter in axes(gradient.output_cell_raw, 1)
        shared = 0.0
        @simd for output in 1:Architecture.Q_OUTPUT_CELL_COUNT
            shared += Float64(gradient.output_cell_raw[parameter, output])
        end
        shared *= inverse_candidates * inverse_cells
        total = muladd(shared, shared, total)
    end
    return sqrt(total)
end

"""Norm of auxiliary output gradients, excluding every numeric-Q group."""
function _auxiliary_gradient_norm(gradient::ParameterGradient)
    total = 0.0
    first_auxiliary = Architecture.Q_OUTPUT_CELL_COUNT + 1
    @inbounds for output in first_auxiliary:size(gradient.output_cell_raw, 2)
        for parameter in axes(gradient.output_cell_raw, 1)
            value = Float64(gradient.output_cell_raw[parameter, output])
            total = muladd(value, value, total)
        end
    end
    for array in (
        gradient.output_edge_raw,
        gradient.output_gain,
        gradient.output_bias,
    )
        @inbounds @simd for index in eachindex(array)
            value = Float64(array[index])
            total = muladd(value, value, total)
        end
    end
    return sqrt(total)
end

mutable struct AdamWState
    first::ParameterGradient
    second::ParameterGradient
    total_step::Int
    output_step::Int
    recurrent_step::Int
end

AdamWState(parameters::Model.Parameters) = AdamWState(
    ParameterGradient(parameters), ParameterGradient(parameters),
    0, 0, 0,
)

"""Fail closed if a checkpoint or mutation introduced hidden per-bit Q state."""
function assert_shared_q_state(
    state::AdamWState,
    parameters::Model.Parameters,
)
    q_cells = Architecture.Q_OUTPUT_CELL_COUNT
    arrays = (
        parameters.output_cell_raw,
        state.first.output_cell_raw,
        state.second.output_cell_raw,
    )
    labels = ("parameter", "first moment", "second moment")
    @inbounds for (array, label) in zip(arrays, labels)
        for output in 2:q_cells
            for cell_parameter in axes(array, 1)
                array[cell_parameter, output] == array[cell_parameter, 1] ||
                    throw(ArgumentError(
                        "numeric Q-cell $label columns must be bitwise shared",
                    ))
            end
        end
    end
    return nothing
end

@enum OptimizationPhase::UInt8 begin
    OUTPUT_OPTIMIZATION = 1
    RECURRENT_OPTIMIZATION = 2
end

@inline function _adamw_array!(parameter, gradient, first, second, alpha,
                               scale, decay)
    beta1 = 0.9f0
    beta2 = 0.999f0
    epsilon = 1.0f-8
    @inbounds for index in eachindex(parameter)
        value = gradient[index] * scale
        m = muladd(beta1, first[index], (1.0f0 - beta1) * value)
        v = muladd(beta2, second[index], (1.0f0 - beta2) * value * value)
        first[index] = m
        second[index] = v
        parameter[index] -= alpha * (m / (sqrt(v) + epsilon) + decay * parameter[index])
    end
    return nothing
end

function _adamw_output_cells!(parameter, gradient, first, second, alpha,
                              scale, decay)
    beta1 = 0.9f0
    beta2 = 0.999f0
    epsilon = 1.0f-8
    @inbounds for output in (Architecture.Q_OUTPUT_CELL_COUNT + 1):size(parameter, 2)
        for cell_parameter in axes(parameter, 1)
            value = gradient[cell_parameter, output] * scale
            m = muladd(
                beta1,
                first[cell_parameter, output],
                (1.0f0 - beta1) * value,
            )
            v = muladd(
                beta2,
                second[cell_parameter, output],
                (1.0f0 - beta2) * value * value,
            )
            first[cell_parameter, output] = m
            second[cell_parameter, output] = v
            parameter[cell_parameter, output] -= alpha * (
                m / (sqrt(v) + epsilon) +
                decay * parameter[cell_parameter, output]
            )
        end
    end
    return nothing
end

"""Update one shared numeric-cell parameter vector from the mean bit gradient.

The 32 hard IEEE register cells are distinct dynamical states, but their
46-parameter Reduced Hay equation is one shared numeric primitive.  Only the
first column owns the Adam calculation.  Copying the resulting parameter and
moments to every Q column after each update makes that tying bitwise exact and
prevents optimizer state from becoming a hidden per-bit parameterization.
"""
function _adamw_shared_q_cell!(
    parameter,
    gradient,
    first,
    second,
    alpha,
    scale,
    decay,
)
    beta1 = 0.9f0
    beta2 = 0.999f0
    epsilon = 1.0f-8
    q_cells = Architecture.Q_OUTPUT_CELL_COUNT
    inverse_q_cells = inv(Float32(q_cells))
    @inbounds for cell_parameter in axes(parameter, 1)
        mean_gradient = 0.0f0
        @simd for output in 1:q_cells
            mean_gradient += gradient[cell_parameter, output]
        end
        value = mean_gradient * inverse_q_cells * scale
        momentum = muladd(
            beta1,
            first[cell_parameter, 1],
            (1.0f0 - beta1) * value,
        )
        variance = muladd(
            beta2,
            second[cell_parameter, 1],
            (1.0f0 - beta2) * value * value,
        )
        shared_parameter = parameter[cell_parameter, 1] - alpha * (
            momentum / (sqrt(variance) + epsilon) +
            decay * parameter[cell_parameter, 1]
        )
        for output in 1:q_cells
            parameter[cell_parameter, output] = shared_parameter
            first[cell_parameter, output] = momentum
            second[cell_parameter, output] = variance
        end
    end
    return nothing
end

"""Update auxiliary output edges without touching numeric-Q edge slots."""
function _adamw_auxiliary_output_edges!(
    parameter,
    gradient,
    first,
    second,
    alpha,
    scale,
    decay,
)
    beta1 = 0.9f0
    beta2 = 0.999f0
    epsilon = 1.0f-8
    @inbounds for source in axes(parameter, 2)
        for relation in (OutputCellBank.Q_FANOUT_PER_SOURCE + 1):size(parameter, 1)
            value = gradient[relation, source] * scale
            m = muladd(
                beta1,
                first[relation, source],
                (1.0f0 - beta1) * value,
            )
            v = muladd(
                beta2,
                second[relation, source],
                (1.0f0 - beta2) * value * value,
            )
            first[relation, source] = m
            second[relation, source] = v
            parameter[relation, source] -= alpha * (
                m / (sqrt(v) + epsilon) +
                decay * parameter[relation, source]
            )
        end
    end
    return nothing
end

"""Apply candidate-mean Adam to numeric-Q afferents and basal intercepts."""
function _adamw_q_adapter!(
    edge_parameter,
    edge_gradient,
    edge_first,
    edge_second,
    basal_parameter,
    basal_gradient,
    basal_first,
    basal_second,
    alpha,
    scale,
    decay,
)
    beta1 = 0.9f0
    beta2 = 0.999f0
    epsilon = 1.0f-8
    @inbounds for source in axes(edge_parameter, 2)
        for relation in 1:OutputCellBank.Q_FANOUT_PER_SOURCE
            value = edge_gradient[relation, source] * scale
            momentum = muladd(
                beta1,
                edge_first[relation, source],
                (1.0f0 - beta1) * value,
            )
            variance = muladd(
                beta2,
                edge_second[relation, source],
                (1.0f0 - beta2) * value * value,
            )
            edge_first[relation, source] = momentum
            edge_second[relation, source] = variance
            edge_parameter[relation, source] -= alpha * (
                momentum / (sqrt(variance) + epsilon) +
                decay * edge_parameter[relation, source]
            )
        end
    end
    @inbounds for output in eachindex(basal_parameter)
        value = basal_gradient[output] * scale
        momentum = muladd(
            beta1,
            basal_first[output],
            (1.0f0 - beta1) * value,
        )
        variance = muladd(
            beta2,
            basal_second[output],
            (1.0f0 - beta2) * value * value,
        )
        basal_first[output] = momentum
        basal_second[output] = variance
        basal_parameter[output] -= alpha * (
            momentum / (sqrt(variance) + epsilon) +
            decay * basal_parameter[output]
        )
    end
    return nothing
end

function apply_adamw!(state::AdamWState, parameters::Model.Parameters,
                      gradient::ParameterGradient,
                      config::Config.LocalLearningConfig;
                      phase::OptimizationPhase=OUTPUT_OPTIMIZATION,
                      q_candidate_count::Integer=1)
    state.total_step += 1
    recurrent_fields = (
        :cell_raw,
        :sensory_gain_raw,
        :edge_strength_raw,
        :payload_gain_raw,
    )
    if phase == RECURRENT_OPTIMIZATION
        norm = gradient_norm(gradient, recurrent_fields)
        clip = Float32(min(
            1.0,
            Float64(config.clip_norm) / max(norm, eps(Float64)),
        ))
        state.recurrent_step += 1
        correction = sqrt(1.0f0 - 0.999f0^state.recurrent_step) /
                     (1.0f0 - 0.9f0^state.recurrent_step)
        recurrent_ramp = min(
            Float32(state.recurrent_step) /
            Float32(config.recurrent_ramp_updates),
            1.0f0,
        )
        recurrent_alpha = config.learning_rate * correction *
            config.recurrent_multiplier * recurrent_ramp
        sensory_alpha = recurrent_alpha * config.sensory_multiplier
        _adamw_array!(parameters.cell_raw, gradient.cell_raw,
                      state.first.cell_raw, state.second.cell_raw,
                      recurrent_alpha, clip, config.weight_decay)
        _adamw_array!(parameters.sensory_gain_raw, gradient.sensory_gain_raw,
                      state.first.sensory_gain_raw, state.second.sensory_gain_raw,
                      sensory_alpha, clip, config.weight_decay)
        _adamw_array!(parameters.edge_strength_raw, gradient.edge_strength_raw,
                      state.first.edge_strength_raw, state.second.edge_strength_raw,
                      recurrent_alpha, clip, config.weight_decay)
        _adamw_array!(parameters.payload_gain_raw, gradient.payload_gain_raw,
                      state.first.payload_gain_raw, state.second.payload_gain_raw,
                      recurrent_alpha, clip, config.weight_decay)
        return norm, clip
    end

    candidates = Int(q_candidate_count)
    q_norm = _q_mean_gradient_norm(gradient, candidates)
    auxiliary_norm = _auxiliary_gradient_norm(gradient)
    q_clip = Float32(min(
        1.0,
        Float64(config.clip_norm) / max(q_norm, eps(Float64)),
    ))
    auxiliary_clip = Float32(min(
        1.0,
        Float64(config.clip_norm) / max(auxiliary_norm, eps(Float64)),
    ))
    q_scale = q_clip / Float32(candidates)
    norm = hypot(q_norm, auxiliary_norm)
    clip = min(q_clip, auxiliary_clip)
    state.output_step += 1
    correction = sqrt(1.0f0 - 0.999f0^state.output_step) /
                 (1.0f0 - 0.9f0^state.output_step)
    schedule = state.total_step > config.learning_rate_decay_start ?
        config.learning_rate_decay_multiplier : 1.0f0
    output_alpha = config.learning_rate * schedule * correction
    _adamw_shared_q_cell!(
        parameters.output_cell_raw,
        gradient.output_cell_raw,
        state.first.output_cell_raw,
        state.second.output_cell_raw,
        output_alpha * config.q_cell_multiplier,
        q_scale,
        config.q_weight_decay,
    )
    _adamw_output_cells!(parameters.output_cell_raw, gradient.output_cell_raw,
                         state.first.output_cell_raw, state.second.output_cell_raw,
                         output_alpha, auxiliary_clip, config.weight_decay)
    _adamw_auxiliary_output_edges!(
        parameters.output_edge_raw,
        gradient.output_edge_raw,
        state.first.output_edge_raw,
        state.second.output_edge_raw,
        output_alpha,
        auxiliary_clip,
        config.weight_decay,
    )
    _adamw_q_adapter!(
        parameters.output_edge_raw,
        gradient.output_q_edge_raw,
        state.first.output_q_edge_raw,
        state.second.output_q_edge_raw,
        parameters.output_q_basal_bias_raw,
        gradient.output_q_basal_bias_raw,
        state.first.output_q_basal_bias_raw,
        state.second.output_q_basal_bias_raw,
        output_alpha,
        q_scale,
        config.q_weight_decay,
    )
    _adamw_array!(parameters.output_gain, gradient.output_gain,
                  state.first.output_gain, state.second.output_gain,
                  output_alpha, auxiliary_clip, config.weight_decay)
    _adamw_array!(parameters.output_bias, gradient.output_bias,
                  state.first.output_bias, state.second.output_bias,
                  output_alpha, auxiliary_clip, config.weight_decay)
    return norm, clip
end

end
