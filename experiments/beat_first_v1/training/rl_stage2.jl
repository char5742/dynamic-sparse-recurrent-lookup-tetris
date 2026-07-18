module BeatFirstRLStage2

using Random
using Statistics

export PrioritizedReplay,
       CompactCandidateSet,
       CompactTransition,
       compact_candidate_set,
       materialize_candidate_set,
       initialize!,
       push_transition!,
       sampled_transitions,
       sample_indices,
       update_priorities!,
       nstep_transitions,
       insertable_nstep_rows,
       ema_parameters,
       ema_parameters!,
       double_dqn_quantile_targets,
       quantile_td_objective,
       scaled_game_reward,
       teacher_weight

const PACKED_BOARD_WORDS = 4
const PackedBoard = NTuple{PACKED_BOARD_WORDS,UInt64}
const SCORE_NORMALIZER = 600.0f0
const TERMINAL_SCORE = -1000.0f0

"""Match the historical learner's reward scale and terminal replacement.

Ordinary score deltas are divided by 600.  A game-over transition is assigned
`-1000 / 600` instead of retaining its immediate line-clear reward, exactly as
the legacy learner's terminal branch does.  Keeping this policy in one tested
function prevents the QR target and old-Q teacher from silently using different
units.
"""
function scaled_game_reward(score_delta::Real, terminal::Bool)
    isfinite(score_delta) || error("score delta is non-finite")
    return terminal ? TERMINAL_SCORE / SCORE_NORMALIZER :
           Float32(score_delta) / SCORE_NORMALIZER
end

"""Lossless compact representation of one canonical candidate set.

The 24x10 root board and every candidate placement use four UInt64 words
instead of materialized Float32 tensors.  Candidate order and multiplicity are
unchanged.  `teacher_q` is either empty (teacher loss masked off) or contains
one old-policy label for every candidate.
"""
struct CompactCandidateSet
    board::PackedBoard
    placements::Vector{PackedBoard}
    ren::Float32
    back_to_back::Float32
    tspin::BitVector
    queue::NTuple{6,UInt8}
    teacher_q::Vector{Float32}
end

Base.length(set::CompactCandidateSet) = length(set.placements)

struct CompactTransition
    current::CompactCandidateSet
    action::Int16
    return_n::Float32
    bootstrap::Union{Nothing,CompactCandidateSet}
    bootstrap_discount::Float32
end

function _pack_board(board)::PackedBoard
    length(board) == 240 || throw(DimensionMismatch("a candidate board must have 240 cells"))
    words = zeros(UInt64, PACKED_BOARD_WORDS)
    for (index, value) in enumerate(board)
        iszero(value) && continue
        value == one(value) || error("compact boards must be binary")
        bit = index - 1
        words[(bit >>> 6) + 1] |= UInt64(1) << (bit & 63)
    end
    return Tuple(words)
end

function _unpack_board(board::PackedBoard)
    result = zeros(UInt8, 24, 10)
    for index in 1:240
        bit = index - 1
        result[index] = UInt8((board[(bit >>> 6) + 1] >>> (bit & 63)) & 0x01)
    end
    return result
end

function _queue_codes(queue)
    size(queue, 1) == 7 || throw(DimensionMismatch("queue must have seven mino rows"))
    size(queue, 2) == 6 || throw(DimensionMismatch("queue must contain HOLD plus five NEXT"))
    return ntuple(6) do column
        hot = findall(>(0), @view queue[:, column])
        length(hot) <= 1 || error("queue column $column is not one-hot")
        isempty(hot) ? UInt8(0) : UInt8(only(hot))
    end
end

"""Compact the canonical `legacy_candidate_batch` without changing order."""
function compact_candidate_set(input; teacher_q=Float32[])
    board, placements, ren, back_to_back, tspin, queue = input
    count = size(placements, 4)
    count > 0 || error("cannot compact an empty candidate set")
    count <= 208 || error("canonical candidate count $count exceeds physical capacity 208")
    size(board, 4) == count || throw(DimensionMismatch("board/candidate count differs"))
    labels = Float32.(collect(teacher_q))
    isempty(labels) || length(labels) == count || throw(
        DimensionMismatch("teacher labels must be empty or match the candidate count"),
    )
    all(isfinite, labels) || error("teacher labels contain a non-finite value")
    packed_placements = PackedBoard[
        _pack_board(@view placements[:, :, 1, action]) for action in 1:count
    ]
    tspin_bits = BitVector(vec(tspin) .> 0)
    length(tspin_bits) == count || throw(DimensionMismatch("T-spin count differs"))
    return CompactCandidateSet(
        _pack_board(@view board[:, :, 1, 1]),
        packed_placements,
        Float32(ren[1, 1]),
        Float32(back_to_back[1, 1]),
        tspin_bits,
        _queue_codes(@view queue[:, :, 1]),
        labels,
    )
end

"""Materialize a compact set into the canonical dataset-packer representation."""
function materialize_candidate_set(set::CompactCandidateSet)
    count = length(set)
    count > 0 || error("compact candidate set is empty")
    placements = zeros(UInt8, 24, 10, 1, count)
    for action in 1:count
        placements[:, :, 1, action] .= _unpack_board(set.placements[action])
    end
    queue = zeros(UInt8, 7, 6)
    for column in 1:6
        code = Int(set.queue[column])
        code == 0 || (queue[code, column] = 1)
    end
    return (;
        board=_unpack_board(set.board),
        placements,
        ren=set.ren,
        back_to_back=set.back_to_back,
        tspin=Float32.(set.tspin),
        queue,
        teacher_q=copy(set.teacher_q),
    )
end

mutable struct PrioritizedReplay
    priorities::Vector{Float64}
    entries::Vector{Union{Nothing,CompactTransition}}
    alpha::Float64
    epsilon::Float64
    count::Int
    next_index::Int
    insertions::Int
    max_priority::Float64
end

function PrioritizedReplay(capacity::Int; alpha::Real=0.6, epsilon::Real=1.0e-3)
    capacity > 0 || throw(ArgumentError("capacity must be positive"))
    return PrioritizedReplay(
        zeros(capacity),
        Union{Nothing,CompactTransition}[nothing for _ in 1:capacity],
        Float64(alpha),
        Float64(epsilon),
        0,
        1,
        0,
        1.0,
    )
end

function initialize!(replay::PrioritizedReplay, count::Int)
    0 < count <= length(replay.priorities) || throw(ArgumentError("invalid replay count"))
    replay.count = count
    replay.priorities[1:count] .= replay.max_priority^replay.alpha
    replay.next_index = mod1(count + 1, length(replay.priorities))
    return replay
end

"""Insert one transition into the fixed-capacity PER ring."""
function push_transition!(replay::PrioritizedReplay, transition::CompactTransition)
    1 <= transition.action <= length(transition.current) || error(
        "transition action is outside its candidate set",
    )
    transition.bootstrap === nothing && transition.bootstrap_discount != 0 && error(
        "a no-bootstrap transition must have zero discount",
    )
    isfinite(transition.return_n) || error("transition return is non-finite")
    isfinite(transition.bootstrap_discount) || error("transition discount is non-finite")
    index = replay.next_index
    replay.entries[index] = transition
    replay.priorities[index] = replay.max_priority^replay.alpha
    replay.count = min(replay.count + 1, length(replay.entries))
    replay.next_index = mod1(index + 1, length(replay.entries))
    replay.insertions += 1
    return index
end

function sampled_transitions(replay::PrioritizedReplay, indices)
    result = Vector{CompactTransition}(undef, length(indices))
    for (slot, index) in pairs(indices)
        1 <= index <= replay.count || throw(BoundsError(replay.entries, index))
        transition = replay.entries[index]
        transition === nothing && error("sampled replay slot $index is uninitialized")
        result[slot] = transition
    end
    return result
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

"""Rows whose n-step target is valid at an episode boundary.

A true terminal makes every preceding partial return final.  A nonterminal
time-limit truncation has no value for the unobserved boundary state, so its
last `n` rows are omitted instead of being mislabeled as terminal returns.
"""
function insertable_nstep_rows(terminal; n::Int=3)
    n >= 1 || throw(ArgumentError("n must be positive"))
    row_count = length(terminal)
    row_count == 0 && return 1:0
    terminal_count = count(identity, terminal)
    terminal_count <= 1 || error("episode contains multiple terminal markers")
    if terminal_count == 1
        terminal[end] || error("terminal marker must be the final episode row")
        return 1:row_count
    end
    return 1:max(row_count - n, 0)
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

function _ema!(target::AbstractArray, online::AbstractArray, tau)
    size(target) == size(online) || throw(DimensionMismatch("EMA array shapes differ"))
    @. target = (1 - tau) * target + tau * online
    return target
end
function _ema!(target::NamedTuple, online::NamedTuple, tau)
    keys(target) == keys(online) || error("EMA parameter structures differ")
    for key in keys(target)
        _ema!(getproperty(target, key), getproperty(online, key), tau)
    end
    return target
end
function _ema!(target::Tuple, online::Tuple, tau)
    length(target) == length(online) || error("EMA tuple lengths differ")
    for index in eachindex(target)
        _ema!(target[index], online[index], tau)
    end
    return target
end
_ema!(target, online, tau) = target == online ? target : error(
    "unsupported non-array EMA leaf $(typeof(target))",
)

"""Update a preallocated target parameter tree without allocating new leaves."""
function ema_parameters!(target, online; tau::Float32=0.005f0)
    0.0f0 < tau <= 1.0f0 || error("EMA tau must be in (0,1]")
    return _ema!(target, online, tau)
end

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
    targets = hasproperty(batch, :targets) ? batch.targets : batch
    quantiles = output.quantiles
    k, n = size(quantiles)
    width, batch_size = size(targets.action_mask)
    width * batch_size == n || error("RL action mask shape mismatch")
    packed_quantiles = reshape(quantiles, k, width, batch_size)
    chosen = packed_quantiles .* reshape(targets.action_mask, 1, width, batch_size)
    # Packed RL batches carry one valid selected candidate per transition.
    selected_quantiles = dropdims(sum(chosen; dims=2); dims=2)
    priority_errors = vec(mean(targets.target_quantiles; dims=1)) .-
                      vec(mean(selected_quantiles; dims=1))
    prediction = reshape(selected_quantiles, k, 1, batch_size)
    target = reshape(targets.target_quantiles, 1, k, batch_size)
    error = target .- prediction
    tau = reshape((Float32.(1:k) .- 0.5f0) ./ Float32(k), k, 1, 1)
    negative = ifelse.(error .< 0.0f0, 1.0f0, 0.0f0)
    weight = abs.(tau .- negative)
    per_pair = weight .* _quantile_huber(error, 1.0f0)
    per_transition = dropdims(mean(per_pair; dims=(1, 2)); dims=(1, 2))
    qr_loss = sum(per_transition .* targets.importance_weights) /
              max(sum(targets.importance_weights), 1.0f0)
    q = reshape(output.q, width, batch_size)
    teacher_difference = q .- targets.teacher_q
    teacher_huber = ifelse.(
        abs.(teacher_difference) .<= 1.0f0,
        0.5f0 .* teacher_difference .^ 2,
        abs.(teacher_difference) .- 0.5f0,
    )
    teacher_loss = sum(teacher_huber .* targets.teacher_mask) /
                   max(sum(targets.teacher_mask), 1.0f0)
    current_teacher_weight = targets.teacher_weight[1]
    loss = qr_loss + current_teacher_weight * teacher_loss
    return loss, next_state, (;
        qr_loss,
        teacher_loss,
        teacher_weight=current_teacher_weight,
        mean_q=mean(output.q),
        mean_absolute_td=mean(abs.(error)),
        priority_errors,
    )
end

end # module
