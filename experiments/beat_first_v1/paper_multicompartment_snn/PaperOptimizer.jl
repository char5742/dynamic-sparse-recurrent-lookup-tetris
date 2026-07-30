module PaperOptimizer

export PaperAdamW,
    paper_adam_step!,
    parameter_copy,
    parameter_norm,
    zero_parameter_tree!,
    zero_parameter_tree

const PARAMETER_FIELDS = (
    :input_conductance,
    :recurrent_conductance,
    :workspace_conductance,
    :query_weight,
    :workspace_key,
    :workspace_decay_logit,
    :head_weight,
    :head_bias,
    :output_weight,
    :output_bias,
)

function zero_parameter_tree(parameters)
    keys(parameters) == PARAMETER_FIELDS ||
        error("paper parameter registry changed: $(keys(parameters))")
    return NamedTuple{keys(parameters)}(
        map(array -> zeros(Float32, size(array)), values(parameters)),
    )
end

@generated function zero_parameter_tree!(
    tree::NamedTuple{K},
) where {K}
    operations = [
        :(fill!(getfield(tree, $(QuoteNode(name))), 0.0f0))
        for name in K
    ]
    return quote
        $(operations...)
        tree
    end
end

function parameter_copy(parameters)
    return NamedTuple{keys(parameters)}(
        map(copy, values(parameters)),
    )
end

@generated function parameter_norm(tree::NamedTuple{K}) where {K}
    operations = [
        quote
            array = getfield(tree, $(QuoteNode(name)))
            @inbounds for value in array
                total = muladd(Float64(value), Float64(value), total)
            end
        end
        for name in K
    ]
    return quote
        total = 0.0
        $(operations...)
        sqrt(total)
    end
end

mutable struct PaperAdamW{T}
    first_moment::T
    second_moment::T
    learning_rate::Float32
    beta1::Float32
    beta2::Float32
    epsilon::Float32
    weight_decay::Float32
    beta1_power::Float32
    beta2_power::Float32
    step::Int
end

function PaperAdamW(
    parameters;
    learning_rate::Real=5.0f-4,
    beta1::Real=0.9f0,
    beta2::Real=0.999f0,
    epsilon::Real=1.0f-8,
    weight_decay::Real=1.0f-5,
)
    return PaperAdamW(
        zero_parameter_tree(parameters),
        zero_parameter_tree(parameters),
        Float32(learning_rate),
        Float32(beta1),
        Float32(beta2),
        Float32(epsilon),
        Float32(weight_decay),
        1.0f0,
        1.0f0,
        0,
    )
end

@inline function _adam_array!(
    parameter,
    gradient,
    first,
    second,
    optimizer::PaperAdamW,
    gradient_scale::Float32,
)
    inverse_first_bias = inv(1.0f0 - optimizer.beta1_power)
    inverse_second_bias = inv(1.0f0 - optimizer.beta2_power)
    one_minus_beta1 = 1.0f0 - optimizer.beta1
    one_minus_beta2 = 1.0f0 - optimizer.beta2
    @inbounds for index in eachindex(parameter)
        grad = gradient[index] * gradient_scale
        moment1 = muladd(
            optimizer.beta1,
            first[index],
            one_minus_beta1 * grad,
        )
        moment2 = muladd(
            optimizer.beta2,
            second[index],
            one_minus_beta2 * grad * grad,
        )
        first[index] = moment1
        second[index] = moment2
        parameter[index] -= optimizer.learning_rate * (
            moment1 * inverse_first_bias /
            (sqrt(moment2 * inverse_second_bias) + optimizer.epsilon) +
            optimizer.weight_decay * parameter[index]
        )
        gradient[index] = 0.0f0
    end
    return nothing
end

function paper_adam_step!(
    optimizer::PaperAdamW,
    parameters,
    gradient;
    maximum_norm::Real=5.0,
)
    norm = parameter_norm(gradient)
    scale = Float32(norm > maximum_norm ? maximum_norm / norm : 1.0)
    optimizer.step += 1
    optimizer.beta1_power *= optimizer.beta1
    optimizer.beta2_power *= optimizer.beta2
    @inbounds for name in PARAMETER_FIELDS
        _adam_array!(
            getproperty(parameters, name),
            getproperty(gradient, name),
            getproperty(optimizer.first_moment, name),
            getproperty(optimizer.second_moment, name),
            optimizer,
            scale,
        )
    end
    # Paper connectivity constraints.
    clamp!(parameters.input_conductance, 0.0f0, 1.0f0)
    clamp!(parameters.recurrent_conductance, 0.0f0, 1.0f0)
    clamp!(parameters.workspace_conductance, 0.0f0, 1.0f0)
    return norm
end

end
