module BeatFirstRLStage2

using Random
using Statistics

export PrioritizedReplay,
       initialize!,
       sample_indices,
       update_priorities!,
       nstep_transitions,
       ema_parameters,
       double_dqn_quantile_targets,
       quantile_td_objective,
       teacher_weight

mutable struct PrioritizedReplay
    priorities::Vector{Float64}
    alpha::Float64
    epsilon::Float64
    count::Int
    max_priority::Float64
end

function PrioritizedReplay(capacity::Int; alpha::Real=0.6, epsilon::Real=1.0e-3)
    capacity > 0 || throw(ArgumentError("capacity must be positive"))
    return PrioritizedReplay(zeros(capacity), Float64(alpha), Float64(epsilon), 0, 1.0)
end

function initialize!(replay::PrioritizedReplay, count::Int)
    0 < count <= length(replay.priorities) || throw(ArgumentError("invalid replay count"))
    replay.count = count
    replay.priorities[1:count] .= replay.max_priority^replay.alpha
    return replay
end

function sample_indices(
    rng::AbstractRNG,
    replay::PrioritizedReplay,
    batch_size::Int;
    beta::Real=0.4,
)
    replay.count > 0 || error("replay is empty")
    masses = @view replay.priorities[1:replay.count]
    total = sum(masses)
    total > 0 || error("replay priority mass is zero")
    cumulative = cumsum(masses)
    segment = total / batch_size
    indices = Vector{Int}(undef, batch_size)
    probabilities = Vector{Float64}(undef, batch_size)
    for slot in 1:batch_size
        mass = (slot - 1 + rand(rng)) * segment
        index = searchsortedfirst(cumulative, mass)
        indices[slot] = index
        probabilities[slot] = masses[index] / total
    end
    weights = (replay.count .* probabilities) .^ (-Float64(beta))
    weights ./= maximum(weights)
    return indices, Float32.(weights)
end

function update_priorities!(replay::PrioritizedReplay, indices, td_errors)
    length(indices) == length(td_errors) || throw(DimensionMismatch())
    for (index, error) in zip(indices, td_errors)
        priority = abs(Float64(error)) + replay.epsilon
        isfinite(priority) || error("non-finite replay priority")
        replay.max_priority = max(replay.max_priority, priority)
        replay.priorities[index] = priority^replay.alpha
    end
    return replay
end

function nstep_transitions(
    rewards,
    episode_ids,
    steps,
    terminal;
    n::Int=3,
    discount::Float32=0.997f0,
)
    count = length(rewards)
    count == length(episode_ids) == length(steps) == length(terminal) || throw(DimensionMismatch())
    returns = zeros(Float32, count)
    bootstrap_rows = zeros(Int, count)
    bootstrap_discounts = zeros(Float32, count)
    for row in 1:count
        factor = 1.0f0
        horizon = 0
        for offset in 0:(n - 1)
            cursor = row + offset
            if cursor > count || episode_ids[cursor] != episode_ids[row] ||
               steps[cursor] != steps[row] + offset
                break
            end
            returns[row] += factor * Float32(rewards[cursor])
            horizon += 1
            terminal[cursor] && break
            factor *= discount
        end
        next_row = row + horizon
        if horizon == n && next_row <= count &&
           episode_ids[next_row] == episode_ids[row] &&
           steps[next_row] == steps[row] + horizon && !terminal[row + horizon - 1]
            bootstrap_rows[row] = next_row
            bootstrap_discounts[row] = factor
        end
    end
    return (; returns, bootstrap_rows, bootstrap_discounts)
end

_ema(target::AbstractArray, online::AbstractArray, tau) =
    (1 - tau) .* target .+ tau .* online
_ema(target::Number, online::Number, tau) = (1 - tau) * target + tau * online
function _ema(target::NamedTuple, online::NamedTuple, tau)
    keys(target) == keys(online) || error("EMA parameter structures differ")
    return NamedTuple{keys(target)}(Tuple(
        _ema(getfield(target, key), getfield(online, key), tau) for key in keys(target)
    ))
end
_ema(target::Tuple, online::Tuple, tau) = map((left, right) -> _ema(left, right, tau), target, online)

ema_parameters(target, online; tau::Float32=0.005f0) = _ema(target, online, tau)

teacher_weight(update::Integer; initial::Real=0.25, decay_updates::Integer=50_000) =
    Float32(initial * max(1 - update / decay_updates, 0))

"""Build n-step Double-DQN distributional targets on the host.

The online mean quantiles select the masked next action; the target/EMA network
supplies that action's quantiles. `rewards` are already n-step returns and
`discounts` are zero for terminal/no-bootstrap transitions.
"""
function double_dqn_quantile_targets(
    online_output,
    target_output,
    next_mask,
    rewards,
    discounts,
)
    width, batch_size = size(next_mask)
    online_quantiles = reshape(
        online_output.quantiles, size(online_output.quantiles, 1), width, batch_size,
    )
    target_quantiles = reshape(
        target_output.quantiles, size(target_output.quantiles, 1), width, batch_size,
    )
    size(online_quantiles, 1) == size(target_quantiles, 1) || error(
        "online and target quantile counts differ",
    )
    k = size(online_quantiles, 1)
    result = zeros(eltype(target_quantiles), k, batch_size)
    for slot in 1:batch_size
        count_actions = count(>(0), @view next_mask[:, slot])
        if count_actions == 0 || discounts[slot] == 0
            result[:, slot] .= rewards[slot]
            continue
        end
        online_q = vec(mean(@view online_quantiles[:, 1:count_actions, slot]; dims=1))
        action = argmax(online_q)
        result[:, slot] .= rewards[slot] .+
                           discounts[slot] .* @view(target_quantiles[:, action, slot])
    end
    return result
end

function _quantile_huber(error, delta::Float32)
    absolute = abs.(error)
    return ifelse.(
        absolute .<= delta,
        0.5f0 .* error .^ 2,
        delta .* (absolute .- 0.5f0 * delta),
    )
end

"""QR-DQN learner objective with externally frozen Double-DQN targets.

`batch.target_quantiles` must be computed from the EMA/target network at the
online network's masked argmax next action. Keeping target construction outside
this function lets the fixed compiled update remain stable across target/EMA
refreshes. The current network and all backbone layers are differentiated.
"""
function quantile_td_objective(model, ps, st, batch)
    output, next_state = model(batch.inputs, ps, st)
    quantiles = output.quantiles
    k, n = size(quantiles)
    width, batch_size = size(batch.action_mask)
    width * batch_size == n || error("RL action mask shape mismatch")
    packed_quantiles = reshape(quantiles, k, width, batch_size)
    chosen = packed_quantiles .* reshape(batch.action_mask, 1, width, batch_size)
    # Packed RL batches carry one valid selected candidate per transition.
    selected_quantiles = dropdims(sum(chosen; dims=2); dims=2)
    prediction = reshape(selected_quantiles, k, 1, batch_size)
    target = reshape(batch.target_quantiles, 1, k, batch_size)
    error = target .- prediction
    tau = reshape((Float32.(1:k) .- 0.5f0) ./ Float32(k), k, 1, 1)
    negative = ifelse.(error .< 0.0f0, 1.0f0, 0.0f0)
    weight = abs.(tau .- negative)
    per_pair = weight .* _quantile_huber(error, 1.0f0)
    per_transition = dropdims(mean(per_pair; dims=(1, 2)); dims=(1, 2))
    qr_loss = sum(per_transition .* batch.importance_weights) /
              max(sum(batch.importance_weights), 1.0f0)
    q = reshape(output.q, width, batch_size)
    teacher_difference = q .- batch.teacher_q
    teacher_huber = ifelse.(
        abs.(teacher_difference) .<= 1.0f0,
        0.5f0 .* teacher_difference .^ 2,
        abs.(teacher_difference) .- 0.5f0,
    )
    teacher_loss = sum(teacher_huber .* batch.teacher_mask) /
                   max(sum(batch.teacher_mask), 1.0f0)
    loss = qr_loss + batch.teacher_weight * teacher_loss
    return loss, next_state, (;
        qr_loss,
        teacher_loss,
        teacher_weight=batch.teacher_weight,
        mean_q=mean(output.q),
        mean_absolute_td=mean(abs.(error)),
    )
end

end # module
