module N1SmokeCore

using LinearAlgebra
using SHA

export candidate_pair,
       candidate_instance_digest,
       discounted_score,
       gate_decision,
       logistic_head_update,
       make_candidate_refs,
       make_decision_context,
       normalize_execution_devices,
       q_ordinal_binding_digest,
       stable_top_two

_text_digest(value::AbstractString) = bytes2hex(SHA.sha256(codeunits(value)))

"""First-max stable top-1/top-2 scan over an already stable candidate order."""
function stable_top_two(values::AbstractVector{<:Real})
    length(values) >= 2 || error("top-2 requires at least two candidates")
    all(isfinite, values) || error("candidate values must all be finite")
    first_index = 0
    second_index = 0
    @inbounds for index in eachindex(values)
        if first_index == 0 || values[index] > values[first_index]
            second_index = first_index
            first_index = index
        elseif second_index == 0 || values[index] > values[second_index]
            second_index = index
        end
    end
    first_index != 0 && second_index != 0 || error("stable top-2 scan failed")
    first_index != second_index || error("stable top-2 indices aliased")
    return first_index, second_index
end

"""Slice the candidate axis of each raw model input without changing its rank."""
function candidate_pair(input::Tuple, indices::NTuple{2,Int})
    return map(input) do value
        selectors = ntuple(
            dimension -> dimension == ndims(value) ? collect(indices) : Colon(),
            ndims(value),
        )
        value[selectors...]
    end
end

"""Preserve every ordered candidate, even when identity components collide."""
function make_candidate_refs(
    stable_key_digests::AbstractVector{<:AbstractString},
    action_digests::AbstractVector{<:AbstractString},
    afterstate_digests::AbstractVector{<:AbstractString},
)
    count = length(stable_key_digests)
    length(action_digests) == count == length(afterstate_digests) ||
        error("candidate identity vector lengths differ")
    return [
        (;
            ordinal=index,
            stable_key_digest=String(stable_key_digests[index]),
            action_digest=String(action_digests[index]),
            afterstate_digest=String(afterstate_digests[index]),
        )
        for index in 1:count
    ]
end

function candidate_instance_digest(reference)
    return _text_digest(join((
        string(reference.ordinal),
        String(reference.stable_key_digest),
        String(reference.action_digest),
        String(reference.afterstate_digest),
    ), "|"))
end

function make_decision_context(root_state_digest::AbstractString, references)
    ordered_digest = _text_digest(join(candidate_instance_digest.(references), "\n"))
    return (;
        root_state_digest=String(root_state_digest),
        count=length(references),
        ordered_candidate_vector_digest=ordered_digest,
    )
end

"""Bind every Q output and its historical chunk coordinate to one ordinal."""
function q_ordinal_binding_digest(
    references,
    values::AbstractVector{Float32};
    chunk_size::Int=16,
)
    length(references) == length(values) || error("Q/candidate count mismatch")
    chunk_size > 0 || error("chunk size must be positive")
    lines = String[]
    count = length(references)
    for (reference, value) in zip(references, values)
        ordinal = Int(reference.ordinal)
        1 <= ordinal <= count || error("candidate ordinal is outside decision context")
        chunk = cld(ordinal, chunk_size)
        within_chunk = mod1(ordinal, chunk_size)
        chunk_first = (chunk - 1) * chunk_size + 1
        actual_chunk_size = min(chunk_size, count - chunk_first + 1)
        push!(lines, join((
            string(ordinal),
            string(chunk),
            string(within_chunk),
            string(actual_chunk_size),
            candidate_instance_digest(reference),
            string(reinterpret(UInt32, value); base=16, pad=8),
        ), "|"))
    end
    return _text_digest(join(lines, "\n"))
end

"""Unbootstrapped discounted sum. `rewards[1]` has exponent zero."""
function discounted_score(rewards::AbstractVector{<:Real}; gamma::Float64=0.997)
    0.0 < gamma <= 1.0 || error("gamma must lie in (0, 1]")
    all(isfinite, rewards) || error("rewards must all be finite")
    value = 0.0
    discount = 1.0
    @inbounds for reward in rewards
        value += discount * Float64(reward)
        discount *= gamma
    end
    return value
end

"""One plain finite logistic-head update. Caller owns and may discard the result."""
function logistic_head_update(
    weights::AbstractVector{<:Real},
    bias::Real,
    features::AbstractVector{<:Real},
    label::Real;
    learning_rate::Float32=1.0f-3,
)
    length(weights) == 64 || error("N1 logistic head requires 64 weights")
    length(features) == 64 || error("N1 logistic head requires 64 features")
    label in (0, 1, 0.0f0, 1.0f0) || error("label must be binary")
    all(isfinite, weights) && all(isfinite, features) && isfinite(bias) ||
        error("logistic update inputs must be finite")
    logit = Float32(dot(Float32.(weights), Float32.(features)) + Float32(bias))
    probability = inv(1.0f0 + exp(-logit))
    residual = probability - Float32(label)
    updated_weights = Float32.(weights) .-
                      learning_rate .* residual .* Float32.(features)
    updated_bias = Float32(bias) - learning_rate * residual
    all(isfinite, updated_weights) && isfinite(updated_bias) ||
        error("logistic update produced a non-finite parameter")
    return updated_weights, updated_bias
end

"""Label-free deployment decision from a pre-existing 65-parameter head."""
function gate_decision(
    weights::AbstractVector{<:Real},
    bias::Real,
    features::AbstractVector{<:Real};
    threshold::Float32=0.6f0,
)
    length(weights) == 64 || error("N1 gate requires 64 weights")
    length(features) == 64 || error("N1 gate requires 64 features")
    all(isfinite, weights) && all(isfinite, features) && isfinite(bias) ||
        return :fallback
    logit = Float32(dot(Float32.(weights), Float32.(features)) + Float32(bias))
    probability = inv(1.0f0 + exp(-logit))
    return isfinite(probability) && probability >= threshold ? :top2 : :fallback
end

"""Normalize OpenVINO's scalar or sequence EXECUTION_DEVICES property."""
function normalize_execution_devices(value)
    devices = value isa AbstractString ? [String(value)] : String.(collect(value))
    isempty(devices) && error("OpenVINO reported no execution device")
    all(device -> !isempty(device), devices) ||
        error("OpenVINO reported an empty execution device")
    return devices
end

end # module
