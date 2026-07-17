using Random

"""Fenwick tree storing positive proportional-prioritization masses."""
mutable struct PriorityTree
    tree::Vector{Float64}
    values::Vector{Float64}
end

PriorityTree(capacity::Int) = PriorityTree(zeros(capacity + 1), zeros(capacity))
Base.length(tree::PriorityTree) = length(tree.values)

function priority_total(tree::PriorityTree)
    result = 0.0
    index = length(tree.values)
    while index > 0
        result += tree.tree[index]
        index -= index & -index
    end
    return result
end

function set_priority!(tree::PriorityTree, index::Int, value::Real)
    1 <= index <= length(tree) || throw(BoundsError(tree.values, index))
    isfinite(value) && value >= 0 || throw(ArgumentError("priority must be finite and nonnegative"))
    delta = Float64(value) - tree.values[index]
    tree.values[index] = Float64(value)
    cursor = index
    while cursor <= length(tree)
        tree.tree[cursor] += delta
        cursor += cursor & -cursor
    end
    return tree
end

function priority_index(tree::PriorityTree, mass::Real)
    total_mass = priority_total(tree)
    total_mass > 0 || throw(ArgumentError("cannot sample an empty priority tree"))
    target = clamp(Float64(mass), 0.0, prevfloat(total_mass))
    index = 0
    bit = prevpow(2, length(tree))
    while bit != 0
        candidate = index + bit
        if candidate <= length(tree) && tree.tree[candidate] <= target
            index = candidate
            target -= tree.tree[candidate]
        end
        bit >>= 1
    end
    return index + 1
end

mutable struct PrioritizedIndexReplay
    tree::PriorityTree
    alpha::Float64
    epsilon::Float64
    count::Int
    max_priority::Float64
end

function PrioritizedIndexReplay(capacity::Int; alpha::Real=0.6, epsilon::Real=1.0e-3)
    return PrioritizedIndexReplay(
        PriorityTree(capacity), Float64(alpha), Float64(epsilon), 0, 1.0
    )
end

function initialize!(replay::PrioritizedIndexReplay, count::Int)
    0 <= count <= length(replay.tree) || throw(ArgumentError("count exceeds capacity"))
    replay.count = count
    for index in 1:count
        set_priority!(replay.tree, index, replay.max_priority^replay.alpha)
    end
    return replay
end

function sample_indices(
    rng::AbstractRNG, replay::PrioritizedIndexReplay, batch_size::Int; beta::Real=0.4
)
    replay.count > 0 || throw(ArgumentError("replay is empty"))
    total_mass = priority_total(replay.tree)
    segment = total_mass / batch_size
    indices = Vector{Int}(undef, batch_size)
    probabilities = Vector{Float64}(undef, batch_size)
    @inbounds for slot in 1:batch_size
        mass = (slot - 1 + rand(rng)) * segment
        index = priority_index(replay.tree, mass)
        indices[slot] = index
        probabilities[slot] = replay.tree.values[index] / total_mass
    end
    weights = (replay.count .* probabilities) .^ (-Float64(beta))
    weights ./= maximum(weights)
    return indices, Float32.(weights)
end

function update_priorities!(
    replay::PrioritizedIndexReplay, indices::AbstractVector{Int}, td_errors
)
    length(indices) == length(td_errors) || throw(DimensionMismatch())
    for (index, error) in zip(indices, td_errors)
        priority = abs(Float64(error)) + replay.epsilon
        replay.max_priority = max(replay.max_priority, priority)
        set_priority!(replay.tree, index, priority^replay.alpha)
    end
    return replay
end

"""Create n-step offline transitions from consecutive teacher trajectory rows."""
function nstep_transitions(
    rewards::AbstractVector,
    episode_ids::AbstractVector{<:Integer},
    steps::AbstractVector{<:Integer},
    terminal::AbstractVector{Bool};
    n::Int=3,
    discount::Float32=0.997f0,
)
    count = length(rewards)
    count == length(episode_ids) == length(steps) == length(terminal) ||
        throw(DimensionMismatch())
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
            if terminal[cursor]
                break
            end
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
