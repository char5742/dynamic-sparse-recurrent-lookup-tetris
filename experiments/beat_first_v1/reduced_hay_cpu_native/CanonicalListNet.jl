module CanonicalListNet

export ListNetConfig,
       ListNetScratch,
       ListNetResult,
       compose_dueling_q!,
       dueling_pullback!,
       listnet_loss_and_gradient!,
       dueling_listnet_loss_and_gradient!

"""
Configuration for the canonical scalar-Q ranking objective.

The ListNet temperature and scale floor are expressed in teacher-Q units.  A
single teacher-derived scale is used for both teacher and student logits, so
the ranking objective is invariant to a candidate-common shift but remains
sensitive to the magnitude of the student's candidate gaps.
"""
struct ListNetConfig{T<:AbstractFloat}
    temperature::T
    scale_floor::T
    q_huber_weight::T
    huber_delta::T

    function ListNetConfig{T}(
        temperature::T,
        scale_floor::T,
        q_huber_weight::T,
        huber_delta::T,
    ) where {T<:AbstractFloat}
        isfinite(temperature) && temperature > zero(T) || throw(
            ArgumentError("temperature must be finite and positive"),
        )
        isfinite(scale_floor) && scale_floor > zero(T) || throw(
            ArgumentError("scale_floor must be finite and positive"),
        )
        isfinite(q_huber_weight) && q_huber_weight >= zero(T) || throw(
            ArgumentError("q_huber_weight must be finite and nonnegative"),
        )
        isfinite(huber_delta) && huber_delta > zero(T) || throw(
            ArgumentError("huber_delta must be finite and positive"),
        )
        return new{T}(
            temperature,
            scale_floor,
            q_huber_weight,
            huber_delta,
        )
    end
end

function ListNetConfig(
    ::Type{T}=Float32;
    temperature::Real=1,
    scale_floor::Real=1.0e-2,
    q_huber_weight::Real=0.25,
    huber_delta::Real=1,
) where {T<:AbstractFloat}
    return ListNetConfig{T}(
        T(temperature),
        T(scale_floor),
        T(q_huber_weight),
        T(huber_delta),
    )
end

"""Fixed storage for the allocation-free per-state softmax kernel."""
struct ListNetScratch{T<:AbstractFloat}
    teacher_probability::Vector{T}
    student_probability::Vector{T}

    function ListNetScratch{T}(width::Integer) where {T<:AbstractFloat}
        width >= 1 || throw(ArgumentError("candidate width must be positive"))
        return new{T}(zeros(T, width), zeros(T, width))
    end
end

ListNetScratch(width::Integer, ::Type{T}=Float32) where {T<:AbstractFloat} =
    ListNetScratch{T}(width)

"""Loss diagnostics returned by [`listnet_loss_and_gradient!`](@ref)."""
struct ListNetResult{T<:AbstractFloat}
    total_loss::T
    listnet_kl::T
    q_huber_loss::T
    teacher_entropy::T
    valid_candidates::Int
end

@inline function _check_ranking_shapes(
    student_q::AbstractMatrix,
    teacher_q::AbstractMatrix,
    q_gradient::AbstractMatrix,
    counts::AbstractVector,
    scratch::ListNetScratch,
)
    size(student_q) == size(teacher_q) || throw(DimensionMismatch(
        "student_q and teacher_q must have identical shapes",
    ))
    size(q_gradient) == size(student_q) || throw(DimensionMismatch(
        "q_gradient must have the same shape as student_q",
    ))
    length(counts) == size(student_q, 2) || throw(DimensionMismatch(
        "counts length must equal the number of states",
    ))
    length(scratch.teacher_probability) >= size(student_q, 1) || throw(
        DimensionMismatch("ListNet scratch is narrower than student_q"),
    )
    length(scratch.student_probability) >= size(student_q, 1) || throw(
        DimensionMismatch("ListNet scratch is narrower than student_q"),
    )
    return nothing
end

@inline function _check_dueling_shapes(
    q::AbstractMatrix,
    value::AbstractVector,
    advantage::AbstractMatrix,
    counts::AbstractVector,
)
    size(q) == size(advantage) || throw(DimensionMismatch(
        "q and advantage must have identical shapes",
    ))
    length(value) == size(q, 2) || throw(DimensionMismatch(
        "value length must equal the number of states",
    ))
    length(counts) == size(q, 2) || throw(DimensionMismatch(
        "counts length must equal the number of states",
    ))
    return nothing
end

@inline function _checked_count(counts, state::Int, width::Int)
    count = Int(@inbounds counts[state])
    1 <= count <= width || throw(ArgumentError(
        "state $state has $count candidates outside 1:$width",
    ))
    return count
end

"""
    compose_dueling_q!(q, value, advantage, counts)

Compose one scalar Q per valid candidate using

`q[a,s] = value[s] + advantage[a,s] - mean(advantage[1:count,s])`.

Padded candidate slots are set to zero and never participate in the mean.
"""
function compose_dueling_q!(
    q::AbstractMatrix{T},
    value::AbstractVector{T},
    advantage::AbstractMatrix{T},
    counts::AbstractVector{<:Integer},
) where {T<:AbstractFloat}
    _check_dueling_shapes(q, value, advantage, counts)
    fill!(q, zero(T))
    width, states = size(q)
    @inbounds for state in 1:states
        count = _checked_count(counts, state, width)
        advantage_sum = zero(T)
        for candidate in 1:count
            advantage_sum += advantage[candidate, state]
        end
        advantage_mean = advantage_sum / T(count)
        state_value = value[state]
        for candidate in 1:count
            q[candidate, state] = state_value +
                advantage[candidate, state] - advantage_mean
        end
    end
    return q
end

"""
    dueling_pullback!(value_gradient, advantage_gradient, q_gradient, counts)

Exact pullback of [`compose_dueling_q!`](@ref).  Destination gradients are
overwritten; padded advantage gradients are zero.
"""
function dueling_pullback!(
    value_gradient::AbstractVector{T},
    advantage_gradient::AbstractMatrix{T},
    q_gradient::AbstractMatrix{T},
    counts::AbstractVector{<:Integer},
) where {T<:AbstractFloat}
    size(advantage_gradient) == size(q_gradient) || throw(DimensionMismatch(
        "advantage_gradient and q_gradient must have identical shapes",
    ))
    length(value_gradient) == size(q_gradient, 2) || throw(DimensionMismatch(
        "value_gradient length must equal the number of states",
    ))
    length(counts) == size(q_gradient, 2) || throw(DimensionMismatch(
        "counts length must equal the number of states",
    ))
    fill!(value_gradient, zero(T))
    fill!(advantage_gradient, zero(T))
    width, states = size(q_gradient)
    @inbounds for state in 1:states
        count = _checked_count(counts, state, width)
        state_sum = zero(T)
        for candidate in 1:count
            state_sum += q_gradient[candidate, state]
        end
        value_gradient[state] = state_sum
        state_mean = state_sum / T(count)
        for candidate in 1:count
            advantage_gradient[candidate, state] =
                q_gradient[candidate, state] - state_mean
        end
    end
    return value_gradient, advantage_gradient
end

@inline function _huber_loss(error::T, delta::T) where {T<:AbstractFloat}
    magnitude = abs(error)
    return magnitude <= delta ?
        T(0.5) * error * error :
        delta * (magnitude - T(0.5) * delta)
end

@inline function _huber_derivative(error::T, delta::T) where {T<:AbstractFloat}
    return clamp(error, -delta, delta)
end

"""
    listnet_loss_and_gradient!(q_gradient, scratch, student_q, teacher_q,
                               counts, config)

Compute the canonical ranking objective and its exact student-Q derivative.
For state `s`, the common logit scale is derived only from teacher Q:

```
a_s = sqrt(mean((teacher_q - mean(teacher_q))^2) + scale_floor^2)
p_t = softmax(teacher_q / (temperature * a_s))
p_s = softmax(student_q / (temperature * a_s))
```

The reported ranking term is `KL(p_t || p_s)`, not cross entropy with its
irreducible teacher entropy.  Raw-Q Huber supervision is averaged over all
valid candidates and added with `config.q_huber_weight`.
"""
function listnet_loss_and_gradient!(
    q_gradient::AbstractMatrix{T},
    scratch::ListNetScratch{T},
    student_q::AbstractMatrix{T},
    teacher_q::AbstractMatrix{T},
    counts::AbstractVector{<:Integer},
    config::ListNetConfig{T},
) where {T<:AbstractFloat}
    _check_ranking_shapes(
        student_q,
        teacher_q,
        q_gradient,
        counts,
        scratch,
    )
    fill!(q_gradient, zero(T))
    width, states = size(student_q)
    states >= 1 || throw(ArgumentError("at least one state is required"))

    valid_total = 0
    @inbounds for state in 1:states
        valid_total += _checked_count(counts, state, width)
    end
    inverse_states = inv(T(states))
    inverse_valid = inv(T(valid_total))
    listnet_kl = zero(T)
    q_huber_loss = zero(T)
    teacher_entropy = zero(T)

    @inbounds for state in 1:states
        count = _checked_count(counts, state, width)
        teacher_mean = zero(T)
        for candidate in 1:count
            teacher = teacher_q[candidate, state]
            student = student_q[candidate, state]
            isfinite(teacher) || throw(ArgumentError(
                "teacher_q contains a non-finite valid candidate",
            ))
            isfinite(student) || throw(ArgumentError(
                "student_q contains a non-finite valid candidate",
            ))
            teacher_mean += teacher
        end
        teacher_mean /= T(count)

        teacher_variance = zero(T)
        for candidate in 1:count
            centered = teacher_q[candidate, state] - teacher_mean
            teacher_variance = muladd(centered, centered, teacher_variance)
        end
        common_scale = sqrt(
            teacher_variance / T(count) + config.scale_floor^2,
        )
        inverse_logit_scale = inv(config.temperature * common_scale)

        teacher_max = -T(Inf)
        student_max = -T(Inf)
        for candidate in 1:count
            teacher_logit = teacher_q[candidate, state] * inverse_logit_scale
            student_logit = student_q[candidate, state] * inverse_logit_scale
            teacher_max = max(teacher_max, teacher_logit)
            student_max = max(student_max, student_logit)
        end

        teacher_sum = zero(T)
        student_sum = zero(T)
        for candidate in 1:count
            teacher_exponential = exp(
                teacher_q[candidate, state] * inverse_logit_scale - teacher_max,
            )
            student_exponential = exp(
                student_q[candidate, state] * inverse_logit_scale - student_max,
            )
            scratch.teacher_probability[candidate] = teacher_exponential
            scratch.student_probability[candidate] = student_exponential
            teacher_sum += teacher_exponential
            student_sum += student_exponential
        end
        inverse_teacher_sum = inv(teacher_sum)
        inverse_student_sum = inv(student_sum)
        teacher_log_partition = teacher_max + log(teacher_sum)
        student_log_partition = student_max + log(student_sum)

        for candidate in 1:count
            teacher_probability =
                scratch.teacher_probability[candidate] * inverse_teacher_sum
            student_probability =
                scratch.student_probability[candidate] * inverse_student_sum
            scratch.teacher_probability[candidate] = teacher_probability
            scratch.student_probability[candidate] = student_probability

            teacher_log_probability =
                teacher_q[candidate, state] * inverse_logit_scale -
                teacher_log_partition
            student_log_probability =
                student_q[candidate, state] * inverse_logit_scale -
                student_log_partition
            listnet_kl += inverse_states * teacher_probability *
                (teacher_log_probability - student_log_probability)
            teacher_entropy -= inverse_states * teacher_probability *
                teacher_log_probability

            q_gradient[candidate, state] = inverse_states *
                (student_probability - teacher_probability) *
                inverse_logit_scale

            q_error = student_q[candidate, state] -
                teacher_q[candidate, state]
            q_huber_loss += inverse_valid *
                _huber_loss(q_error, config.huber_delta)
            q_gradient[candidate, state] += config.q_huber_weight *
                inverse_valid *
                _huber_derivative(q_error, config.huber_delta)
        end
    end

    total_loss = listnet_kl + config.q_huber_weight * q_huber_loss
    return ListNetResult(
        total_loss,
        listnet_kl,
        q_huber_loss,
        teacher_entropy,
        valid_total,
    )
end

"""
Compose centered dueling Q, evaluate the canonical objective, and pull its
exact Q derivative back to state value and per-candidate advantage.
"""
function dueling_listnet_loss_and_gradient!(
    value_gradient::AbstractVector{T},
    advantage_gradient::AbstractMatrix{T},
    q::AbstractMatrix{T},
    q_gradient::AbstractMatrix{T},
    scratch::ListNetScratch{T},
    value::AbstractVector{T},
    advantage::AbstractMatrix{T},
    teacher_q::AbstractMatrix{T},
    counts::AbstractVector{<:Integer},
    config::ListNetConfig{T},
) where {T<:AbstractFloat}
    compose_dueling_q!(q, value, advantage, counts)
    result = listnet_loss_and_gradient!(
        q_gradient,
        scratch,
        q,
        teacher_q,
        counts,
        config,
    )
    dueling_pullback!(
        value_gradient,
        advantage_gradient,
        q_gradient,
        counts,
    )
    return result
end

end # module CanonicalListNet
