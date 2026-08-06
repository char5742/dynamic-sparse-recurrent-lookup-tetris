module RelationGraphOptimizer

using ..DendriticProgramBank
using ..CandidateDeltaRelationGraph

const Bank = DendriticProgramBank
const Model = CandidateDeltaRelationGraph

export AdamWState,
       AdamWStepCounters,
       AdamWStepStats,
       OptimizerConfig,
       apply_adamw!,
       gradient_norm,
       with_learning_rate

"""
All optimizer policy for the canonical candidate-delta relation graph.

All thirteen learning-rate multipliers live beside clipping and AdamW policy so
there is no second, partially applied group configuration. A zero multiplier
is a strict freeze: parameter, moments and clock remain bit-identical. AdamW
decay applies only to signed program payloads and signed output readout
weights. Positive/transformed raw parameters use explicit physical-space
constraints instead of the mathematically inverted raw-space decay.
"""
struct OptimizerConfig
    learning_rate::Float32
    beta1::Float32
    beta2::Float32
    epsilon::Float32
    clip_norm::Float32
    weight_decay::Float32
    program_multiplier::Float32
    leaf_relation_multiplier::Float32
    relation_cell_multiplier::Float32
    relation_motif_multiplier::Float32
    motif_cell_multiplier::Float32
    common_relation_multiplier::Float32
    common_output_multiplier::Float32
    auxiliary_relation_multiplier::Float32
    placement_relation_multiplier::Float32
    motif_readout_multiplier::Float32
    output_cell_multiplier::Float32
    output_readout_weight_multiplier::Float32
    output_bias_multiplier::Float32
end

function OptimizerConfig(
    ;
    learning_rate::Real=1.0f-3,
    beta1::Real=0.9f0,
    beta2::Real=0.999f0,
    epsilon::Real=1.0f-8,
    clip_norm::Real=1.0f0,
    weight_decay::Real=1.0f-4,
    program_multiplier::Real=1.0,
    leaf_relation_multiplier::Real=1.0,
    relation_cell_multiplier::Real=0.1,
    relation_motif_multiplier::Real=1.0,
    motif_cell_multiplier::Real=0.1,
    common_relation_multiplier::Real=1.0,
    common_output_multiplier::Real=1.0,
    auxiliary_relation_multiplier::Real=1.0,
    placement_relation_multiplier::Real=1.0,
    motif_readout_multiplier::Real=1.0,
    output_cell_multiplier::Real=0.1,
    output_readout_weight_multiplier::Real=1.0,
    output_bias_multiplier::Real=1.0,
)
    values = Float32.((
        learning_rate,
        beta1,
        beta2,
        epsilon,
        clip_norm,
        weight_decay,
        program_multiplier,
        leaf_relation_multiplier,
        relation_cell_multiplier,
        relation_motif_multiplier,
        motif_cell_multiplier,
        common_relation_multiplier,
        common_output_multiplier,
        auxiliary_relation_multiplier,
        placement_relation_multiplier,
        motif_readout_multiplier,
        output_cell_multiplier,
        output_readout_weight_multiplier,
        output_bias_multiplier,
    ))
    learning_rate_value = values[1]
    beta1_value = values[2]
    beta2_value = values[3]
    epsilon_value = values[4]
    clip_norm_value = values[5]
    weight_decay_value = values[6]
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
    isfinite(weight_decay_value) && weight_decay_value >= 0.0f0 ||
        throw(ArgumentError("weight_decay must be finite and non-negative"))
    names = fieldnames(OptimizerConfig)
    @inbounds for index in 7:length(values)
        value = values[index]
        isfinite(value) && value >= 0.0f0 || throw(ArgumentError(
            "$(names[index]) must be finite and non-negative",
        ))
    end
    return OptimizerConfig(values...)
end

"""Return the same optimizer policy with only its scalar learning rate changed."""
function with_learning_rate(config::OptimizerConfig, learning_rate::Real)
    values = (;
        (name => (
            name === :learning_rate ? learning_rate : getfield(config, name)
        ) for name in fieldnames(OptimizerConfig))...,
    )
    return OptimizerConfig(; values...)
end

"""Dense resident moments for every non-program parameter group."""
struct DenseMoments
    leaf_relation::Vector{Float32}
    relation_cell::Matrix{Float32}
    relation_motif::Vector{Float32}
    motif_cell::Matrix{Float32}
    common_relation::Vector{Float32}
    common_output::Vector{Float32}
    auxiliary_relation::Vector{Float32}
    placement_relation::Vector{Float32}
    motif_readout::Matrix{Float32}
    output_cell::Matrix{Float32}
    output_readout_weight::Matrix{Float32}
    output_bias::Vector{Float32}
end

function DenseMoments(parameters::Model.ModelParameters)
    return DenseMoments(
        zeros(Float32, size(parameters.leaf_relation.raw_conductance)),
        zeros(Float32, size(parameters.relation.cell_raw)),
        zeros(Float32, size(parameters.relation_motif.raw_conductance)),
        zeros(Float32, size(parameters.motif.cell_raw)),
        zeros(Float32, size(parameters.context.common_relation.raw_conductance)),
        zeros(Float32, size(parameters.context.common_output.raw_conductance)),
        zeros(Float32, size(parameters.context.aux_relation.raw_conductance)),
        zeros(Float32, size(parameters.placement_relation.raw_conductance)),
        zeros(Float32, size(parameters.motif_readout.source_gain_raw)),
        zeros(Float32, size(parameters.output.cell_raw)),
        zeros(Float32, size(parameters.output.readout_weight)),
        zeros(Float32, size(parameters.output.bias)),
    )
end

"""Visible per-group clocks; program rows additionally own local clocks."""
mutable struct AdamWStepCounters
    total::Int
    program_batches::Int
    program_rows::Int
    leaf_relation::Int
    relation_cell::Int
    relation_motif::Int
    motif_cell::Int
    common_relation::Int
    common_output::Int
    auxiliary_relation::Int
    placement_relation::Int
    motif_readout::Int
    output_cell::Int
    output_readout_weight::Int
    output_bias::Int
end

AdamWStepCounters() = AdamWStepCounters(
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)

"""
AdamW state for the canonical relation graph.

Program moments use direct-address storage for constant-time row lookup, but
only rows named by `SparseProgramGradient` are ever read, written, decayed or
clocked.  The optimizer never scans program-bank capacity.
"""
struct AdamWState
    first::DenseMoments
    second::DenseMoments
    program_first::Matrix{Float32}
    program_second::Matrix{Float32}
    program_step_by_row::Vector{UInt32}
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
        AdamWStepCounters(),
    )
end

struct AdamWStepStats
    gradient_norm::Float64
    clip_scale::Float32
    active_program_rows::Int
end

@inline function _sum_squares_finite(array, label::Symbol)
    total = 0.0
    @inbounds @simd for raw_value in array
        value = Float64(raw_value)
        isfinite(value) || throw(DomainError(raw_value, "$label is not finite"))
        total = muladd(value, value, total)
    end
    return total
end

@inline function _validate_finite(array, label::Symbol)
    @inbounds for value in array
        isfinite(value) || throw(DomainError(value, "$label is not finite"))
    end
    return nothing
end

@inline _enabled(multiplier::Float32) = multiplier > 0.0f0

function _gradient_norm_squared(
    gradient::Model.ModelGradient,
    config::OptimizerConfig,
)
    total = 0.0
    _enabled(config.leaf_relation_multiplier) &&
        (total += _sum_squares_finite(
            gradient.leaf_relation,
            :leaf_relation_gradient,
        ))
    _enabled(config.relation_cell_multiplier) &&
        (total += _sum_squares_finite(
            gradient.relation.cell_raw,
            :relation_cell_gradient,
        ))
    _enabled(config.relation_motif_multiplier) &&
        (total += _sum_squares_finite(
            gradient.relation_motif,
            :relation_motif_gradient,
        ))
    _enabled(config.motif_cell_multiplier) &&
        (total += _sum_squares_finite(
            gradient.motif.cell_raw,
            :motif_cell_gradient,
        ))
    _enabled(config.common_relation_multiplier) &&
        (total += _sum_squares_finite(
            gradient.context.common_relation_raw,
            :common_relation_gradient,
        ))
    _enabled(config.common_output_multiplier) &&
        (total += _sum_squares_finite(
            gradient.context.common_output_raw,
            :common_output_gradient,
        ))
    _enabled(config.auxiliary_relation_multiplier) &&
        (total += _sum_squares_finite(
            gradient.context.aux_relation_raw,
            :auxiliary_relation_gradient,
        ))
    _enabled(config.placement_relation_multiplier) &&
        (total += _sum_squares_finite(
            gradient.placement_relation,
            :placement_relation_gradient,
        ))
    _enabled(config.motif_readout_multiplier) &&
        (total += _sum_squares_finite(
            gradient.motif_readout.source_gain_raw,
            :motif_readout_gradient,
        ))
    _enabled(config.output_cell_multiplier) &&
        (total += _sum_squares_finite(
            gradient.output.cell_raw,
            :output_cell_gradient,
        ))
    _enabled(config.output_readout_weight_multiplier) &&
        (total += _sum_squares_finite(
            gradient.output.readout_weight,
            :output_readout_weight_gradient,
        ))
    _enabled(config.output_bias_multiplier) &&
        (total += _sum_squares_finite(
            gradient.output.bias,
            :output_bias_gradient,
        ))
    if _enabled(config.program_multiplier)
        @inbounds for slot in 1:Bank.active_gradient_count(gradient.program)
            for lane in 1:Bank.PAYLOAD_WIDTH
                value = gradient.program.values[lane, slot]
                isfinite(value) || throw(DomainError(
                    value,
                    "program gradient is not finite",
                ))
                value64 = Float64(value)
                total = muladd(value64, value64, total)
            end
        end
    end
    return total
end

"""L2 norm over every non-frozen canonical gradient group."""
function gradient_norm(
    gradient::Model.ModelGradient,
    config::OptimizerConfig=OptimizerConfig();
    gradient_scale::Real=1.0,
)
    scale = Float64(gradient_scale)
    isfinite(scale) && scale >= 0.0 || throw(ArgumentError(
        "gradient_scale must be finite and non-negative",
    ))
    value = sqrt(_gradient_norm_squared(gradient, config)) * scale
    isfinite(value) || throw(DomainError(value, "gradient norm is not finite"))
    return value
end

@inline function _validate_triplet(parameter, first, second, label::Symbol)
    _validate_finite(parameter, label)
    _validate_finite(first, label)
    _validate_finite(second, label)
    return nothing
end

@inline function _check_increment(counter::Int, label::AbstractString)
    counter == typemax(Int) && throw(OverflowError("$label step overflow"))
    return nothing
end

function _preflight!(
    state::AdamWState,
    parameters::Model.ModelParameters,
    gradient::Model.ModelGradient,
    config::OptimizerConfig,
)
    _check_increment(state.steps.total, "total optimizer")
    if _enabled(config.program_multiplier)
        active = Bank.active_gradient_count(gradient.program)
        active > 0 && _check_increment(
            state.steps.program_batches,
            "program batch",
        )
        state.steps.program_rows <= typemax(Int) - active ||
            throw(OverflowError("program-row aggregate step overflow"))
        @inbounds for slot in 1:active
            row = Int(Bank.active_gradient_row(gradient.program, slot))
            1 <= row <= size(parameters.program_bank.payload, 2) ||
                throw(BoundsError(axes(parameters.program_bank.payload, 2), row))
            state.program_step_by_row[row] == typemax(UInt32) &&
                throw(OverflowError("program-row Adam step overflow"))
            for lane in 1:Bank.PAYLOAD_WIDTH
                parameter = parameters.program_bank.payload[lane, row]
                first = state.program_first[lane, row]
                second = state.program_second[lane, row]
                isfinite(parameter) || throw(DomainError(
                    parameter,
                    "active program parameter is not finite",
                ))
                isfinite(first) || throw(DomainError(
                    first,
                    "active program first moment is not finite",
                ))
                isfinite(second) || throw(DomainError(
                    second,
                    "active program second moment is not finite",
                ))
            end
        end
    end
    if _enabled(config.leaf_relation_multiplier)
        _check_increment(state.steps.leaf_relation, "leaf-relation")
        _validate_triplet(
            parameters.leaf_relation.raw_conductance,
            state.first.leaf_relation,
            state.second.leaf_relation,
            :leaf_relation,
        )
    end
    if _enabled(config.relation_cell_multiplier)
        _check_increment(state.steps.relation_cell, "relation-cell")
        _validate_triplet(
            parameters.relation.cell_raw,
            state.first.relation_cell,
            state.second.relation_cell,
            :relation_cell,
        )
    end
    if _enabled(config.relation_motif_multiplier)
        _check_increment(state.steps.relation_motif, "relation-motif")
        _validate_triplet(
            parameters.relation_motif.raw_conductance,
            state.first.relation_motif,
            state.second.relation_motif,
            :relation_motif,
        )
    end
    if _enabled(config.motif_cell_multiplier)
        _check_increment(state.steps.motif_cell, "motif-cell")
        _validate_triplet(
            parameters.motif.cell_raw,
            state.first.motif_cell,
            state.second.motif_cell,
            :motif_cell,
        )
    end
    if _enabled(config.common_relation_multiplier)
        _check_increment(state.steps.common_relation, "common-relation")
        _validate_triplet(
            parameters.context.common_relation.raw_conductance,
            state.first.common_relation,
            state.second.common_relation,
            :common_relation,
        )
    end
    if _enabled(config.common_output_multiplier)
        _check_increment(state.steps.common_output, "common-output")
        _validate_triplet(
            parameters.context.common_output.raw_conductance,
            state.first.common_output,
            state.second.common_output,
            :common_output,
        )
    end
    if _enabled(config.auxiliary_relation_multiplier)
        _check_increment(state.steps.auxiliary_relation, "auxiliary-relation")
        _validate_triplet(
            parameters.context.aux_relation.raw_conductance,
            state.first.auxiliary_relation,
            state.second.auxiliary_relation,
            :auxiliary_relation,
        )
    end
    if _enabled(config.placement_relation_multiplier)
        _check_increment(state.steps.placement_relation, "placement-relation")
        _validate_triplet(
            parameters.placement_relation.raw_conductance,
            state.first.placement_relation,
            state.second.placement_relation,
            :placement_relation,
        )
    end
    if _enabled(config.motif_readout_multiplier)
        _check_increment(state.steps.motif_readout, "motif-readout")
        _validate_triplet(
            parameters.motif_readout.source_gain_raw,
            state.first.motif_readout,
            state.second.motif_readout,
            :motif_readout,
        )
    end
    if _enabled(config.output_cell_multiplier)
        _check_increment(state.steps.output_cell, "output-cell")
        _validate_triplet(
            parameters.output.cell_raw,
            state.first.output_cell,
            state.second.output_cell,
            :output_cell,
        )
    end
    if _enabled(config.output_readout_weight_multiplier)
        _check_increment(
            state.steps.output_readout_weight,
            "output-readout-weight",
        )
        _validate_triplet(
            parameters.output.readout_weight,
            state.first.output_readout_weight,
            state.second.output_readout_weight,
            :output_readout_weight,
        )
    end
    if _enabled(config.output_bias_multiplier)
        _check_increment(state.steps.output_bias, "output-bias")
        _validate_triplet(
            parameters.output.bias,
            state.first.output_bias,
            state.second.output_bias,
            :output_bias,
        )
    end
    return nothing
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

@inline function _dense_group!(
    parameter,
    gradient,
    first,
    second,
    config::OptimizerConfig,
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

function _sparse_program!(
    state::AdamWState,
    parameters::Model.ModelParameters,
    gradient::Model.ModelGradient,
    config::OptimizerConfig,
    gradient_scale::Float32,
)
    active = Bank.active_gradient_count(gradient.program)
    active == 0 && return 0
    beta1 = config.beta1
    beta2 = config.beta2
    one_minus_beta1 = 1.0f0 - beta1
    one_minus_beta2 = 1.0f0 - beta2
    learning_rate = config.learning_rate * config.program_multiplier
    @inbounds for slot in 1:active
        row = Int(Bank.active_gradient_row(gradient.program, slot))
        next_step = state.program_step_by_row[row] + UInt32(1)
        state.program_step_by_row[row] = next_step
        step = Int(next_step)
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
    state.steps.program_batches += 1
    state.steps.program_rows += active
    return active
end

"""
Apply one globally clipped AdamW step.

All enabled gradients, parameters and moments are validated before the first
mutation.  A NaN/Inf or counter overflow therefore fails closed.  The caller
owns gradient averaging through `gradient_scale` and cache refresh.
"""
function apply_adamw!(
    state::AdamWState,
    parameters::Model.ModelParameters,
    gradient::Model.ModelGradient,
    config::OptimizerConfig;
    gradient_scale::Real=1.0,
)
    scale64 = Float64(gradient_scale)
    isfinite(scale64) && scale64 >= 0.0 || throw(ArgumentError(
        "gradient_scale must be finite and non-negative",
    ))
    norm = gradient_norm(gradient, config; gradient_scale=scale64)
    _preflight!(state, parameters, gradient, config)
    clip_scale = Float32(min(
        1.0,
        Float64(config.clip_norm) / max(norm, eps(Float64)),
    ))
    effective_scale = Float32(scale64) * clip_scale
    steps = state.steps
    steps.total += 1

    active_program_rows = _enabled(config.program_multiplier) ?
        _sparse_program!(state, parameters, gradient, config, effective_scale) :
        0
    if _enabled(config.leaf_relation_multiplier)
        steps.leaf_relation += 1
        _dense_group!(
            parameters.leaf_relation.raw_conductance,
            gradient.leaf_relation,
            state.first.leaf_relation,
            state.second.leaf_relation,
            config,
            config.leaf_relation_multiplier,
            effective_scale,
            0.0f0,
            steps.leaf_relation,
        )
    end
    if _enabled(config.relation_cell_multiplier)
        steps.relation_cell += 1
        _dense_group!(
            parameters.relation.cell_raw,
            gradient.relation.cell_raw,
            state.first.relation_cell,
            state.second.relation_cell,
            config,
            config.relation_cell_multiplier,
            effective_scale,
            0.0f0,
            steps.relation_cell,
        )
    end
    if _enabled(config.relation_motif_multiplier)
        steps.relation_motif += 1
        _dense_group!(
            parameters.relation_motif.raw_conductance,
            gradient.relation_motif,
            state.first.relation_motif,
            state.second.relation_motif,
            config,
            config.relation_motif_multiplier,
            effective_scale,
            0.0f0,
            steps.relation_motif,
        )
    end
    if _enabled(config.motif_cell_multiplier)
        steps.motif_cell += 1
        _dense_group!(
            parameters.motif.cell_raw,
            gradient.motif.cell_raw,
            state.first.motif_cell,
            state.second.motif_cell,
            config,
            config.motif_cell_multiplier,
            effective_scale,
            0.0f0,
            steps.motif_cell,
        )
    end
    if _enabled(config.common_relation_multiplier)
        steps.common_relation += 1
        _dense_group!(
            parameters.context.common_relation.raw_conductance,
            gradient.context.common_relation_raw,
            state.first.common_relation,
            state.second.common_relation,
            config,
            config.common_relation_multiplier,
            effective_scale,
            0.0f0,
            steps.common_relation,
        )
    end
    if _enabled(config.common_output_multiplier)
        steps.common_output += 1
        _dense_group!(
            parameters.context.common_output.raw_conductance,
            gradient.context.common_output_raw,
            state.first.common_output,
            state.second.common_output,
            config,
            config.common_output_multiplier,
            effective_scale,
            0.0f0,
            steps.common_output,
        )
    end
    if _enabled(config.auxiliary_relation_multiplier)
        steps.auxiliary_relation += 1
        _dense_group!(
            parameters.context.aux_relation.raw_conductance,
            gradient.context.aux_relation_raw,
            state.first.auxiliary_relation,
            state.second.auxiliary_relation,
            config,
            config.auxiliary_relation_multiplier,
            effective_scale,
            0.0f0,
            steps.auxiliary_relation,
        )
    end
    if _enabled(config.placement_relation_multiplier)
        steps.placement_relation += 1
        _dense_group!(
            parameters.placement_relation.raw_conductance,
            gradient.placement_relation,
            state.first.placement_relation,
            state.second.placement_relation,
            config,
            config.placement_relation_multiplier,
            effective_scale,
            0.0f0,
            steps.placement_relation,
        )
    end
    if _enabled(config.motif_readout_multiplier)
        steps.motif_readout += 1
        _dense_group!(
            parameters.motif_readout.source_gain_raw,
            gradient.motif_readout.source_gain_raw,
            state.first.motif_readout,
            state.second.motif_readout,
            config,
            config.motif_readout_multiplier,
            effective_scale,
            0.0f0,
            steps.motif_readout,
        )
    end
    if _enabled(config.output_cell_multiplier)
        steps.output_cell += 1
        _dense_group!(
            parameters.output.cell_raw,
            gradient.output.cell_raw,
            state.first.output_cell,
            state.second.output_cell,
            config,
            config.output_cell_multiplier,
            effective_scale,
            0.0f0,
            steps.output_cell,
        )
    end
    if _enabled(config.output_readout_weight_multiplier)
        steps.output_readout_weight += 1
        _dense_group!(
            parameters.output.readout_weight,
            gradient.output.readout_weight,
            state.first.output_readout_weight,
            state.second.output_readout_weight,
            config,
            config.output_readout_weight_multiplier,
            effective_scale,
            config.weight_decay,
            steps.output_readout_weight,
        )
    end
    if _enabled(config.output_bias_multiplier)
        steps.output_bias += 1
        _dense_group!(
            parameters.output.bias,
            gradient.output.bias,
            state.first.output_bias,
            state.second.output_bias,
            config,
            config.output_bias_multiplier,
            effective_scale,
            0.0f0,
            steps.output_bias,
        )
    end
    return AdamWStepStats(norm, clip_scale, active_program_rows)
end

end # module RelationGraphOptimizer
