module WorkspaceRoutingPolicy

export DEFAULT_EXPLORATION,
    DEFAULT_LOGIT_LIMIT,
    DEFAULT_NORM_EPSILON,
    counter_gumbel,
    deterministic_topk!,
    ordered_logpolicy,
    ordered_score_eligibility!,
    prepare_policy!,
    routing_mix64,
    sample_plackett_luce_topk!,
    self_test

const DEFAULT_EXPLORATION = 0.05f0
const DEFAULT_LOGIT_LIMIT = 3.0f0
const DEFAULT_NORM_EPSILON = 1.0f-4

@inline function bounded_standardized_score(
    value::T,
    limit::T=T(DEFAULT_LOGIT_LIMIT),
) where {T<:AbstractFloat}
    limit == T(Inf) && return value
    return limit * value / (limit + abs(value))
end

@inline function bounded_standardized_derivative(
    value::T,
    limit::T=T(DEFAULT_LOGIT_LIMIT),
) where {T<:AbstractFloat}
    limit == T(Inf) && return one(T)
    ratio = limit / (limit + abs(value))
    return ratio * ratio
end

@inline function _check_common_lengths(
    first::AbstractVector,
    second::AbstractVector,
    third::AbstractVector,
    scores::AbstractVector,
)
    blocks = length(scores)
    blocks >= 2 || throw(ArgumentError("routing requires at least two blocks"))
    length(first) == blocks || throw(DimensionMismatch("first routing buffer"))
    length(second) == blocks || throw(DimensionMismatch("second routing buffer"))
    length(third) == blocks || throw(DimensionMismatch("third routing buffer"))
    return blocks
end

@inline function _routing_parameters(
    ::Type{T},
    temperature::Real,
    exploration::Real,
    norm_epsilon::Real,
    logit_limit::Real,
) where {T<:AbstractFloat}
    tau = T(temperature)
    epsilon = T(exploration)
    normalization_epsilon = T(norm_epsilon)
    limit = T(logit_limit)
    isfinite(tau) && tau > zero(T) || throw(ArgumentError(
        "routing temperature must be finite and positive",
    ))
    isfinite(epsilon) && zero(T) <= epsilon < one(T) ||
        throw(ArgumentError("routing exploration must be in [0, 1)"))
    isfinite(normalization_epsilon) && normalization_epsilon > zero(T) ||
        throw(ArgumentError(
            "routing normalization epsilon must be finite and positive",
        ))
    (isfinite(limit) && limit > zero(T)) || limit == T(Inf) ||
        throw(ArgumentError(
            "routing logit limit must be positive or Inf",
        ))
    return tau, epsilon, normalization_epsilon, limit
end

"""
Standardize raw block scores and apply a monotone soft bound without changing
their deterministic ordering.

The destination is overwritten with

    z = (score - mean(score)) / sqrt(mean(abs2, centered_score) + norm_epsilon)
    bounded = limit * z / (limit + abs(z))

The soft bound prevents a single learned score outlier from making the
Plackett-Luce policy numerically deterministic.  Unlike clipping, its
derivative stays nonzero for every finite score.  The inverse RMS is returned.
The implementation is scalar and allocation free after compilation.
"""
@inline function _standardize_scores!(
    standardized::AbstractVector{T},
    scores::AbstractVector{T},
    norm_epsilon::T,
    logit_limit::T,
) where {T<:AbstractFloat}
    blocks = length(scores)
    length(standardized) == blocks ||
        throw(DimensionMismatch("standardized routing score buffer"))
    score_sum = zero(T)
    @inbounds for block in 1:blocks
        score = scores[block]
        isfinite(score) || throw(ArgumentError("routing score is not finite"))
        score_sum += score
    end
    mean_score = score_sum / T(blocks)
    square_sum = zero(T)
    @inbounds for block in 1:blocks
        centered = scores[block] - mean_score
        standardized[block] = centered
        square_sum = muladd(centered, centered, square_sum)
    end
    inverse_rms = inv(sqrt(square_sum / T(blocks) + norm_epsilon))
    @inbounds for block in 1:blocks
        standardized[block] = bounded_standardized_score(
            standardized[block] * inverse_rms,
            logit_limit,
        )
    end
    return inverse_rms
end

"""
Prepare the routing policy in preallocated vectors.

`base_probability` is the exploitation policy

    pi = softmax(softbound(standardize(scores)) / temperature)

and `policy_probability` is the policy actually sampled during training

    w = (1 - exploration) * pi + exploration / blocks.

Both probability vectors sum to one. In particular, neither vector is a
`workspace_k`-scaled inclusion-probability approximation and no clamp is
applied. The function returns the inverse RMS used by score normalization.
"""
function prepare_policy!(
    standardized::AbstractVector{T},
    base_probability::AbstractVector{T},
    policy_probability::AbstractVector{T},
    scores::AbstractVector{T};
    temperature::Real=one(T),
    exploration::Real=T(DEFAULT_EXPLORATION),
    norm_epsilon::Real=T(DEFAULT_NORM_EPSILON),
    logit_limit::Real=T(Inf),
) where {T<:AbstractFloat}
    blocks = _check_common_lengths(
        standardized,
        base_probability,
        policy_probability,
        scores,
    )
    tau, epsilon, normalization_epsilon, limit = _routing_parameters(
        T,
        temperature,
        exploration,
        norm_epsilon,
        logit_limit,
    )
    inverse_rms = _standardize_scores!(
        standardized,
        scores,
        normalization_epsilon,
        limit,
    )

    maximum_logit = T(-Inf)
    @inbounds for block in 1:blocks
        maximum_logit = max(maximum_logit, standardized[block] / tau)
    end
    exponential_sum = zero(T)
    @inbounds for block in 1:blocks
        value = exp(standardized[block] / tau - maximum_logit)
        base_probability[block] = value
        exponential_sum += value
    end
    isfinite(exponential_sum) && exponential_sum > zero(T) || error(
        "routing softmax normalization failed",
    )
    inverse_exponential_sum = inv(exponential_sum)
    uniform_floor = epsilon / T(blocks)
    exploit_scale = one(T) - epsilon
    policy_sum = zero(T)
    @inbounds for block in 1:blocks
        base = base_probability[block] * inverse_exponential_sum
        base_probability[block] = base
        policy = muladd(exploit_scale, base, uniform_floor)
        policy_probability[block] = policy
        policy_sum += policy
    end

    # The analytic mass is exactly one. This final normalization removes only
    # floating-point accumulation drift and is not a clamp or capped-simplex
    # approximation.
    isfinite(policy_sum) && policy_sum > zero(T) || error(
        "routing exploration normalization failed",
    )
    inverse_policy_sum = inv(policy_sum)
    @inbounds for block in 1:blocks
        policy_probability[block] *= inverse_policy_sum
    end
    return inverse_rms
end

"""
Write a deterministic hard top-k selection and its ordered ranking.

`selected` and `route_order` are overwritten. Ties are resolved by the lowest
block index, making the result bitwise deterministic.
"""
function deterministic_topk!(
    selected::AbstractVector{Bool},
    route_order::AbstractVector{I},
    scores::AbstractVector{T},
    workspace_k::Integer,
) where {I<:Integer,T<:Real}
    blocks = length(scores)
    length(selected) == blocks ||
        throw(DimensionMismatch("routing selection buffer"))
    1 <= workspace_k <= blocks || throw(ArgumentError(
        "workspace_k must be in 1:blocks",
    ))
    length(route_order) >= workspace_k ||
        throw(DimensionMismatch("routing order buffer"))
    fill!(selected, false)
    fill!(route_order, zero(I))
    @inbounds for rank in 1:workspace_k
        best = 0
        best_score = T(-Inf)
        for block in 1:blocks
            selected[block] && continue
            score = scores[block]
            isfinite(score) ||
                throw(ArgumentError("routing score is not finite"))
            if best == 0 || score > best_score
                best = block
                best_score = score
            end
        end
        best != 0 || error("deterministic routing top-k selection failed")
        selected[best] = true
        route_order[rank] = I(best)
    end
    return nothing
end

"""
SplitMix64 finalizer used to derive independent routing counters.
"""
@inline function routing_mix64(value::UInt64)
    value ⊻= value >> 30
    value *= 0xbf58476d1ce4e5b9
    value ⊻= value >> 27
    value *= 0x94d049bb133111eb
    return value ⊻ (value >> 31)
end

"""
Counter-based Gumbel(0, 1) sample for one cycle and block.

No mutable RNG is read. An optimizer-update-derived nonce therefore makes
sampling reproducible across worker schedules and checkpoint resume.
"""
@inline function counter_gumbel(
    nonce::UInt64,
    cycle::Integer,
    block::Integer,
)
    cycle >= 1 || throw(ArgumentError("routing cycle must be positive"))
    block >= 1 || throw(ArgumentError("routing block must be positive"))
    bits = routing_mix64(
        nonce ⊻
        UInt64(cycle) * 0x9e3779b97f4a7c15 ⊻
        UInt64(block) * 0xd1b54a32d192ed03,
    )
    # Midpoints of the 2^53 bins avoid both log(0) and log(1).
    uniform = (Float64(bits >> 11) + 0.5) * 0x1.0p-53
    return -log(-log(uniform))
end

"""
Sample an ordered hard top-k from `policy_probability`.

The keys `log(w[block]) + Gumbel(block)` produce a Plackett-Luce sample
without replacement. `selected`, `route_order`, and `key_scratch` are all
overwritten, and no mutable RNG or temporary allocation is used.
"""
function sample_plackett_luce_topk!(
    selected::AbstractVector{Bool},
    route_order::AbstractVector{I},
    key_scratch::AbstractVector{T},
    policy_probability::AbstractVector{T},
    workspace_k::Integer,
    nonce::UInt64,
    cycle::Integer,
) where {I<:Integer,T<:AbstractFloat}
    blocks = length(policy_probability)
    length(selected) == blocks ||
        throw(DimensionMismatch("routing selection buffer"))
    length(key_scratch) == blocks ||
        throw(DimensionMismatch("routing key scratch"))
    1 <= workspace_k <= blocks || throw(ArgumentError(
        "workspace_k must be in 1:blocks",
    ))
    length(route_order) >= workspace_k ||
        throw(DimensionMismatch("routing order buffer"))
    fill!(selected, false)
    fill!(route_order, zero(I))
    @inbounds for block in 1:blocks
        probability = policy_probability[block]
        isfinite(probability) && probability > zero(T) || throw(
            ArgumentError("routing policy probabilities must be positive"),
        )
        key_scratch[block] =
            log(probability) + T(counter_gumbel(nonce, cycle, block))
    end
    @inbounds for rank in 1:workspace_k
        best = 0
        best_key = T(-Inf)
        for block in 1:blocks
            selected[block] && continue
            key = key_scratch[block]
            if best == 0 || key > best_key
                best = block
                best_key = key
            end
        end
        best != 0 || error("stochastic routing top-k selection failed")
        selected[best] = true
        route_order[rank] = I(best)
    end
    return nothing
end

@inline function _selected_before(
    route_order::AbstractVector{<:Integer},
    rank::Int,
    block::Int,
)
    @inbounds for previous_rank in 1:(rank - 1)
        Int(route_order[previous_rank]) == block && return true
    end
    return false
end

"""
Log probability of one fixed ordered Plackett-Luce top-k sample.
"""
function ordered_logpolicy(
    policy_probability::AbstractVector{T},
    route_order::AbstractVector{<:Integer},
    workspace_k::Integer,
) where {T<:AbstractFloat}
    blocks = length(policy_probability)
    1 <= workspace_k <= blocks || throw(ArgumentError(
        "workspace_k must be in 1:blocks",
    ))
    length(route_order) >= workspace_k ||
        throw(DimensionMismatch("routing order buffer"))
    remaining_mass = zero(T)
    @inbounds for block in 1:blocks
        probability = policy_probability[block]
        isfinite(probability) && probability > zero(T) || throw(
            ArgumentError("routing policy probabilities must be positive"),
        )
        remaining_mass += probability
    end
    log_probability = zero(T)
    @inbounds for rank in 1:workspace_k
        selected = Int(route_order[rank])
        1 <= selected <= blocks ||
            throw(ArgumentError("routing order contains an invalid block"))
        _selected_before(route_order, rank, selected) && throw(
            ArgumentError("routing order contains a duplicate block"),
        )
        remaining_mass > zero(T) ||
            error("Plackett-Luce remaining mass is not positive")
        selected_probability = policy_probability[selected]
        log_probability += log(selected_probability) - log(remaining_mass)
        remaining_mass -= selected_probability
    end
    return log_probability
end

"""
Compute the exact raw-score eligibility of an ordered routing sample.

The destination is overwritten with

    d log P(route_order | scores) / d scores

for the same normalized-softmax, exploration mixture, and ordered
Plackett-Luce policy used by `prepare_policy!` and
`sample_plackett_luce_topk!`.

All four scratch vectors are caller owned and overwritten:

- `logweight_eligibility`: derivative with respect to `log(w)`
- `alpha_scratch`: exploration-mixture chain factor
- `standardized_scratch`: normalized raw scores
- `base_probability` and `policy_probability`: recomputed policy vectors

The returned vector does not include the third factor. A local learner should
multiply it by its scalar loss/advantage signal before accumulating parameter
gradients.
"""
function ordered_score_eligibility!(
    score_eligibility::AbstractVector{T},
    logweight_eligibility::AbstractVector{T},
    alpha_scratch::AbstractVector{T},
    standardized_scratch::AbstractVector{T},
    base_probability::AbstractVector{T},
    policy_probability::AbstractVector{T},
    scores::AbstractVector{T},
    route_order::AbstractVector{<:Integer},
    workspace_k::Integer;
    temperature::Real=one(T),
    exploration::Real=T(DEFAULT_EXPLORATION),
    norm_epsilon::Real=T(DEFAULT_NORM_EPSILON),
    logit_limit::Real=T(Inf),
) where {T<:AbstractFloat}
    blocks = _check_common_lengths(
        score_eligibility,
        logweight_eligibility,
        alpha_scratch,
        scores,
    )
    length(standardized_scratch) == blocks ||
        throw(DimensionMismatch("standardized routing scratch"))
    length(base_probability) == blocks ||
        throw(DimensionMismatch("base routing probability buffer"))
    length(policy_probability) == blocks ||
        throw(DimensionMismatch("sample routing probability buffer"))
    1 <= workspace_k <= blocks || throw(ArgumentError(
        "workspace_k must be in 1:blocks",
    ))
    length(route_order) >= workspace_k ||
        throw(DimensionMismatch("routing order buffer"))
    tau, epsilon, normalization_epsilon, limit = _routing_parameters(
        T,
        temperature,
        exploration,
        norm_epsilon,
        logit_limit,
    )
    inverse_rms = prepare_policy!(
        standardized_scratch,
        base_probability,
        policy_probability,
        scores;
        temperature=tau,
        exploration=epsilon,
        norm_epsilon=normalization_epsilon,
        logit_limit=limit,
    )

    fill!(logweight_eligibility, zero(T))
    remaining_mass = zero(T)
    @inbounds for block in 1:blocks
        remaining_mass += policy_probability[block]
    end
    @inbounds for rank in 1:workspace_k
        selected = Int(route_order[rank])
        1 <= selected <= blocks ||
            throw(ArgumentError("routing order contains an invalid block"))
        _selected_before(route_order, rank, selected) && throw(
            ArgumentError("routing order contains a duplicate block"),
        )
        remaining_mass > zero(T) ||
            error("Plackett-Luce remaining mass is not positive")
        inverse_remaining_mass = inv(remaining_mass)
        for block in 1:blocks
            _selected_before(route_order, rank, block) && continue
            logweight_eligibility[block] -=
                policy_probability[block] * inverse_remaining_mass
        end
        logweight_eligibility[selected] += one(T)
        remaining_mass -= policy_probability[selected]
    end

    exploit_scale = one(T) - epsilon
    weighted_alpha_eligibility = zero(T)
    @inbounds for block in 1:blocks
        alpha = exploit_scale * base_probability[block] /
            policy_probability[block]
        alpha_scratch[block] = alpha
        weighted_alpha_eligibility = muladd(
            alpha,
            logweight_eligibility[block],
            weighted_alpha_eligibility,
        )
    end

    inverse_temperature = inv(tau)
    score_mean = zero(T)
    @inbounds for block in 1:blocks
        score_mean += scores[block]
    end
    score_mean /= T(blocks)
    mean_normalized_gradient = zero(T)
    mean_gradient_times_score = zero(T)
    @inbounds for block in 1:blocks
        logit_gradient = (
            alpha_scratch[block] * logweight_eligibility[block] -
            base_probability[block] * weighted_alpha_eligibility
        )
        raw_standardized = (scores[block] - score_mean) * inverse_rms
        normalized_gradient =
            logit_gradient * inverse_temperature *
            bounded_standardized_derivative(raw_standardized, limit)
        score_eligibility[block] = normalized_gradient
        mean_normalized_gradient += normalized_gradient
        mean_gradient_times_score = muladd(
            normalized_gradient,
            raw_standardized,
            mean_gradient_times_score,
        )
    end
    mean_normalized_gradient /= T(blocks)
    mean_gradient_times_score /= T(blocks)
    @inbounds for block in 1:blocks
        raw_standardized = (scores[block] - score_mean) * inverse_rms
        score_eligibility[block] = inverse_rms * (
            score_eligibility[block] -
            mean_normalized_gradient -
            raw_standardized * mean_gradient_times_score
        )
    end
    return nothing
end

function _fixed_order_logpolicy(
    scores::Vector{Float64},
    route_order::Vector{Int},
    workspace_k::Int,
    temperature::Float64,
    exploration::Float64,
    norm_epsilon::Float64,
    logit_limit::Float64,
)
    blocks = length(scores)
    standardized = zeros(Float64, blocks)
    base_probability = zeros(Float64, blocks)
    policy_probability = zeros(Float64, blocks)
    prepare_policy!(
        standardized,
        base_probability,
        policy_probability,
        scores;
        temperature,
        exploration,
        norm_epsilon,
        logit_limit,
    )
    return ordered_logpolicy(
        policy_probability,
        route_order,
        workspace_k,
    )
end

"""
Run allocation-insensitive numerical checks for this module's policy algebra.

Hot-path allocation is intentionally measured by the caller after compilation;
this helper allocates only its test fixtures.
"""
function self_test()
    scores = Float64[-1.2, 0.4, 1.1, -0.3, 0.8, 0.05]
    blocks = length(scores)
    workspace_k = 3
    temperature = 0.9
    exploration = 0.05
    norm_epsilon = 1.0e-4
    logit_limit = 3.0
    standardized = zeros(Float64, blocks)
    base_probability = zeros(Float64, blocks)
    policy_probability = zeros(Float64, blocks)
    prepare_policy!(
        standardized,
        base_probability,
        policy_probability,
        scores;
        temperature,
        exploration,
        norm_epsilon,
        logit_limit,
    )
    base_mass_error = abs(sum(base_probability) - 1.0)
    policy_mass_error = abs(sum(policy_probability) - 1.0)
    minimum_floor_margin =
        minimum(policy_probability) - exploration / blocks

    selected = falses(blocks)
    repeated_selected = falses(blocks)
    route_order = zeros(Int, workspace_k)
    repeated_order = zeros(Int, workspace_k)
    key_scratch = zeros(Float64, blocks)
    repeated_key = zeros(Float64, blocks)
    nonce = UInt64(0x51a7c0de12345678)
    sample_plackett_luce_topk!(
        selected,
        route_order,
        key_scratch,
        policy_probability,
        workspace_k,
        nonce,
        2,
    )
    sample_plackett_luce_topk!(
        repeated_selected,
        repeated_order,
        repeated_key,
        policy_probability,
        workspace_k,
        nonce,
        2,
    )
    nonce_deterministic =
        selected == repeated_selected &&
        route_order == repeated_order &&
        key_scratch == repeated_key

    score_eligibility = zeros(Float64, blocks)
    logweight_eligibility = zeros(Float64, blocks)
    alpha_scratch = zeros(Float64, blocks)
    ordered_score_eligibility!(
        score_eligibility,
        logweight_eligibility,
        alpha_scratch,
        standardized,
        base_probability,
        policy_probability,
        scores,
        route_order,
        workspace_k;
        temperature,
        exploration,
        norm_epsilon,
        logit_limit,
    )
    finite_difference = zeros(Float64, blocks)
    step = 1.0e-6
    for block in 1:blocks
        plus_scores = copy(scores)
        minus_scores = copy(scores)
        plus_scores[block] += step
        minus_scores[block] -= step
        plus_value = _fixed_order_logpolicy(
            plus_scores,
            route_order,
            workspace_k,
            temperature,
            exploration,
            norm_epsilon,
            logit_limit,
        )
        minus_value = _fixed_order_logpolicy(
            minus_scores,
            route_order,
            workspace_k,
            temperature,
            exploration,
            norm_epsilon,
            logit_limit,
        )
        finite_difference[block] =
            (plus_value - minus_value) / (2.0 * step)
    end
    finite_difference_max_error =
        maximum(abs.(finite_difference .- score_eligibility))
    zero_sum_error = abs(sum(score_eligibility))

    # A single score outlier is the worst case for RMS normalization: its
    # unbounded standardized value approaches sqrt(blocks).  The monotone
    # soft bound must keep the training policy exploratory without changing
    # deterministic top-k ordering.
    outlier_scores = zeros(Float64, 96)
    outlier_scores[1] = 1000.0
    outlier_standardized = zeros(Float64, 96)
    outlier_base = zeros(Float64, 96)
    outlier_policy = zeros(Float64, 96)
    prepare_policy!(
        outlier_standardized,
        outlier_base,
        outlier_policy,
        outlier_scores;
        temperature=1.0,
        exploration=0.05,
        norm_epsilon,
        logit_limit,
    )
    outlier_max_probability = maximum(outlier_base)

    base_mass_error <= 64eps(Float64) || error(
        "base policy mass self-test failed: $base_mass_error",
    )
    policy_mass_error <= 64eps(Float64) || error(
        "sample policy mass self-test failed: $policy_mass_error",
    )
    minimum_floor_margin >= -64eps(Float64) || error(
        "exploration floor self-test failed: $minimum_floor_margin",
    )
    nonce_deterministic || error("routing nonce determinism self-test failed")
    finite_difference_max_error <= 1.0e-7 || error(
        "routing eligibility finite-difference self-test failed: " *
        "$finite_difference_max_error",
    )
    zero_sum_error <= 1.0e-10 || error(
        "routing eligibility zero-sum self-test failed: $zero_sum_error",
    )
    outlier_max_probability <= 0.15 || error(
        "routing outlier soft bound failed: $outlier_max_probability",
    )
    return (;
        base_mass_error,
        policy_mass_error,
        minimum_floor_margin,
        finite_difference_max_error,
        zero_sum_error,
        outlier_max_probability,
        nonce_deterministic,
        route_order=copy(route_order),
    )
end

end
