module SleepAlignmentDiagnostics

export Float64StatewiseLoss,
    float64_statewise_loss,
    practical_effect_floor

const QUANTILES = 16

struct Float64StatewiseLoss
    composite::Vector{Float64}
    teacher_entropy::Vector{Float64}
    excess::Vector{Float64}
    listnet::Vector{Float64}
    q_huber::Vector{Float64}
    margin::Vector{Float64}
    death::Vector{Float64}
    quantile::Vector{Float64}
    geometry::Vector{Float64}
    structure::Vector{Float64}
end

@inline function _huber64(value::Float64)
    absolute = abs(value)
    return absolute <= 1.0 ? 0.5 * value * value : absolute - 0.5
end

@inline function _logistic_loss64(logit::Float64, label::Float64)
    return max(logit, 0.0) - logit * label + log1p(exp(-abs(logit)))
end

"""
Evaluate the shared 22-output objective directly in Float64 and retain one
additive contribution per state.  This is evaluation-only code: it never
writes gradients or model state.
"""
function float64_statewise_loss(
    arena;
    gate_density::Real=0,
    structure_weight::Real=0,
)
    raw = arena.raw
    targets = arena.targets
    state_batch = arena.state_batch
    width = arena.width
    valid_total = arena.valid_count
    composite = zeros(Float64, state_batch)
    teacher_entropy = zeros(Float64, state_batch)
    listnet = zeros(Float64, state_batch)
    q_huber = zeros(Float64, state_batch)
    margin = zeros(Float64, state_batch)
    death = zeros(Float64, state_batch)
    quantile = zeros(Float64, state_batch)
    geometry = zeros(Float64, state_batch)
    structure = fill(
        Float64(structure_weight) *
        (Float64(gate_density) - 0.5)^2 /
        Float64(state_batch),
        state_batch,
    )
    death_count = 0
    @inbounds for state_slot in 1:state_batch
        count = Int(arena.counts[state_slot])
        for candidate in 1:count
            death_count +=
                targets.death_mask[candidate, state_slot] != 0.0f0
        end
    end
    death_denominator = Float64(max(death_count, 1))
    student_z = zeros(Float64, width)
    teacher_probability = zeros(Float64, width)
    student_probability = zeros(Float64, width)
    inverse_states = inv(Float64(state_batch))
    inverse_valid = inv(Float64(valid_total))
    @inbounds for state_slot in 1:state_batch
        count = Int(arena.counts[state_slot])
        offset = (state_slot - 1) * width
        q_mean = 0.0
        for candidate in 1:count
            q_mean += Float64(raw[1, offset + candidate])
        end
        q_mean /= Float64(count)
        variance = 0.0
        for candidate in 1:count
            centered = Float64(raw[1, offset + candidate]) - q_mean
            student_z[candidate] = centered
            variance = muladd(centered, centered, variance)
        end
        inverse_scale = inv(sqrt(variance / Float64(count) + 1.0e-4))
        teacher_max = -Inf
        student_max = -Inf
        for candidate in 1:count
            student_z[candidate] *= inverse_scale
            teacher_logit =
                Float64(targets.teacher_z[candidate, state_slot]) / 0.5
            student_logit = student_z[candidate] / 0.5
            teacher_max = max(teacher_max, teacher_logit)
            student_max = max(student_max, student_logit)
        end
        teacher_sum = 0.0
        student_sum = 0.0
        for candidate in 1:count
            teacher_probability[candidate] = exp(
                Float64(targets.teacher_z[candidate, state_slot]) / 0.5 -
                teacher_max,
            )
            student_probability[candidate] = exp(
                student_z[candidate] / 0.5 - student_max,
            )
            teacher_sum += teacher_probability[candidate]
            student_sum += student_probability[candidate]
        end
        for candidate in 1:count
            teacher_p = teacher_probability[candidate] / teacher_sum
            student_p = student_probability[candidate] / student_sum
            teacher_entropy[state_slot] -=
                teacher_p * log(max(teacher_p, 1.0e-300)) * inverse_states
            listnet[state_slot] -=
                teacher_p * log(max(student_p, 1.0e-300)) * inverse_states
        end

        top1 = Int(targets.top1[state_slot])
        top2 = Int(targets.top2[state_slot])
        margin_error =
            Float64(raw[1, offset + top1]) -
            Float64(raw[1, offset + top2]) -
            Float64(targets.margin[state_slot])
        margin[state_slot] = _huber64(margin_error) * inverse_states

        line_loss = 0.0
        height_loss = 0.0
        holes_loss = 0.0
        cavities_loss = 0.0
        for candidate in 1:count
            flat = offset + candidate
            q_error =
                Float64(raw[1, flat]) -
                Float64(targets.teacher_q[candidate, state_slot])
            q_huber[state_slot] += _huber64(q_error) * inverse_valid

            if targets.death_mask[candidate, state_slot] != 0.0f0
                death[state_slot] += _logistic_loss64(
                    Float64(raw[2, flat]),
                    Float64(targets.death[candidate, state_slot]),
                ) / death_denominator
            end

            teacher_q = Float64(targets.teacher_q[candidate, state_slot])
            for quantile_index in 1:QUANTILES
                prediction = Float64(raw[2 + quantile_index, flat])
                error = teacher_q - prediction
                tau = (Float64(quantile_index) - 0.5) / QUANTILES
                negative = error < 0.0 ? 1.0 : 0.0
                weight = abs(tau - negative)
                quantile[state_slot] +=
                    weight * _huber64(error) /
                    Float64(valid_total * QUANTILES)
            end

            line_loss += _huber64(
                Float64(raw[19, flat]) -
                Float64(targets.line_clear[candidate, state_slot]) / 4.0,
            ) * inverse_valid
            height_loss += _huber64(
                Float64(raw[20, flat]) -
                Float64(targets.max_height[candidate, state_slot]) / 24.0,
            ) * inverse_valid
            holes_loss += _huber64(
                Float64(raw[21, flat]) -
                Float64(targets.holes[candidate, state_slot]) / 240.0,
            ) * inverse_valid
            cavities_loss += _huber64(
                Float64(raw[22, flat]) -
                Float64(targets.cavities[candidate, state_slot]) / 240.0,
            ) * inverse_valid
        end
        geometry[state_slot] =
            (line_loss + height_loss + holes_loss + cavities_loss) / 4.0
        composite[state_slot] =
            listnet[state_slot] +
            0.25 * q_huber[state_slot] +
            0.15 * margin[state_slot] +
            0.10 * death[state_slot] +
            0.05 * quantile[state_slot] +
            0.10 * geometry[state_slot] +
            structure[state_slot]
    end
    excess = composite .- teacher_entropy
    return Float64StatewiseLoss(
        composite,
        teacher_entropy,
        excess,
        listnet,
        q_huber,
        margin,
        death,
        quantile,
        geometry,
        structure,
    )
end

function practical_effect_floor(
    baseline_excess::Real,
    repeated_measurement_width::Real,
)
    return max(
        0.01 * abs(Float64(baseline_excess)),
        10.0 * abs(Float64(repeated_measurement_width)),
    )
end

end
