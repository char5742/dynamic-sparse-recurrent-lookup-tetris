module R1CounterfactualCollector

using Statistics
using SHA

export ACTION_METRIC_NAMES,
       CURRENT_METRIC_NAMES,
       FEATURE_SCHEMA,
       R1_TRAIN_SEEDS,
       R1_CALIBRATION_SEEDS,
       R1_SAMPLE_PIECES,
       AdapterContract,
       DecisionEvidence,
       BranchOutcome,
       CounterfactualRow,
       EpisodeCollection,
       ExcludedState,
       ExcludableCollectionError,
       FatalCollectionInvariant,
       validate_adapter_contract,
       validate_role_seed,
       stable_top_two,
       build_feature_vector,
       rollout_branch,
       collect_counterfactual,
       collect_episode,
       collect_role_episode,
       numeric_table

const R1_TRAIN_SEEDS = Tuple(73001:73012)
const R1_CALIBRATION_SEEDS = Tuple(73101:73106)
const R1_SAMPLE_PIECES = Tuple(10:10:240)

const ACTION_METRIC_NAMES = (
    :immediate_score,
    :cleared_lines,
    :holes,
    :covered_cells,
    :aggregate_height,
    :max_height,
    :bumpiness,
    :well_sum,
    :row_transitions,
    :column_transitions,
    :ren,
    :back_to_back,
    :tspin,
)

const CURRENT_METRIC_NAMES = (
    :holes,
    :covered_cells,
    :aggregate_height,
    :max_height,
    :bumpiness,
    :well_sum,
    :row_transitions,
    :column_transitions,
)

const _Q_FEATURE_NAMES = (
    :old_q_top1_raw,
    :old_q_top2_raw,
    :old_q_gap_top1_minus_top2,
    :old_q_top1_within_state_z,
    :old_q_top2_within_state_z,
    :old_q_gap_within_state_z,
    :valid_action_count,
)
const _ACTION_DIFF_FEATURE_NAMES = (
    :delta_immediate_score,
    :delta_cleared_lines,
    :delta_holes,
    :delta_covered_cells,
    :delta_aggregate_height,
    :delta_max_height,
    :delta_bumpiness,
    :delta_well_sum,
    :delta_row_transitions,
    :delta_column_transitions,
    :delta_ren,
    :delta_b2b,
    :delta_tspin,
)
const _CURRENT_FEATURE_NAMES = Tuple(Symbol("current_", name) for name in CURRENT_METRIC_NAMES)
const _PIECE_NAMES = (:I, :O, :S, :Z, :J, :L, :T)
const _SLOT_NAMES = (:hold, :next1, :next2, :next3, :next4, :next5)
const _QUEUE_FEATURE_NAMES = Tuple(
    Symbol(slot, "_", piece) for slot in _SLOT_NAMES for piece in _PIECE_NAMES
)
const FEATURE_SCHEMA = (
    _Q_FEATURE_NAMES...,
    _ACTION_DIFF_FEATURE_NAMES...,
    _CURRENT_FEATURE_NAMES...,
    _QUEUE_FEATURE_NAMES...,
)

@assert length(FEATURE_SCHEMA) == 70
@assert length(unique(FEATURE_SCHEMA)) == length(FEATURE_SCHEMA)
@assert length(FEATURE_SCHEMA) < 100

"""The evaluator semantics that R1 is allowed to use.

The production adapter must explicitly attest to these fields.  In particular,
`q_chunk_size=16` with `q_tail_mode=:actual` preserves the historical LayerNorm
candidate-batch semantics; padding the final chunk is not equivalent.
"""
Base.@kwdef struct AdapterContract
    candidate_order::Symbol
    q_chunk_size::Int
    q_tail_mode::Symbol
    next_count::Int
    hold_enabled::Bool
end

struct DecisionEvidence
    rollout_piece::Int
    kind::Symbol
    candidate_count::Int
    candidate_order_digest::String
    q_vector_digest::String
    selected_index::Int
    selected_action_digest::String
    selected_q::Float64
    max_q::Float64
end

struct ExcludableCollectionError <: Exception
    code::Symbol
    detail::String
end

struct FatalCollectionInvariant <: Exception
    code::Symbol
    detail::String
end

Base.showerror(io::IO, error::ExcludableCollectionError) =
    print(io, "R1 excludable state [", error.code, "]: ", error.detail)
Base.showerror(io::IO, error::FatalCollectionInvariant) =
    print(io, "R1 fatal invariant [", error.code, "]: ", error.detail)

struct BranchOutcome
    return_value::Float64
    terminal::Bool
    terminal_step::Int
    bootstrap_q::Float64
    score_deltas::Vector{Float64}
    pre_action_current_piece_tokens::Vector{String}
    placed_piece_tokens::Vector{String}
    rng_digests::Vector{String}
    selected_action_digests::Vector{String}
    candidate_counts::Vector{Int}
    future_stream_digest::String
    decision_evidence::Vector{DecisionEvidence}
end

struct CounterfactualRow
    seed::Int
    episode_id::Int
    piece_index::Int
    root_state_digest::String
    root_future_stream_digest::String
    top1_index::Int
    top2_index::Int
    top1_action_digest::String
    top2_action_digest::String
    q_top1::Float64
    q_top2::Float64
    q_gap::Float64
    valid_action_count::Int
    features::Vector{Float64}
    g6_top1::Float64
    g6_top2::Float64
    advantage_unclipped::Float64
    target_clipped::Float64
    top1_branch::BranchOutcome
    top2_branch::BranchOutcome
end

struct ExcludedState
    seed::Int
    episode_id::Int
    piece_index::Int
    code::Symbol
    detail::String
    root_state_digest::String
end

struct EpisodeCollection
    seed::Int
    episode_id::Int
    rows::Vector{CounterfactualRow}
    exclusions::Vector{ExcludedState}
    canonical_action_digests::Vector{String}
    canonical_terminal::Bool
    canonical_pieces::Int
    canonical_score::Float64
end

function validate_adapter_contract(contract::AdapterContract)
    contract.candidate_order === :stable_node_key || throw(FatalCollectionInvariant(
        :candidate_order,
        "candidate order must be :stable_node_key, got $(contract.candidate_order)",
    ))
    contract.q_chunk_size == 16 || throw(FatalCollectionInvariant(
        :q_chunk_size,
        "historical evaluator requires q_chunk_size=16, got $(contract.q_chunk_size)",
    ))
    contract.q_tail_mode === :actual || throw(FatalCollectionInvariant(
        :q_tail_mode,
        "historical evaluator requires the actual-size tail, got $(contract.q_tail_mode)",
    ))
    contract.next_count == 5 || throw(FatalCollectionInvariant(
        :next_count,
        "R1 is frozen at NEXT=5, got $(contract.next_count)",
    ))
    contract.hold_enabled || throw(FatalCollectionInvariant(
        :hold_disabled,
        "R1 requires HOLD to be enabled",
    ))
    return true
end

function validate_role_seed(seed::Integer, role::Symbol)
    allowed = if role === :train
        R1_TRAIN_SEEDS
    elseif role === :calibration
        R1_CALIBRATION_SEEDS
    else
        throw(ArgumentError("R1 collection role must be :train or :calibration"))
    end
    Int(seed) in allowed || throw(FatalCollectionInvariant(
        :seed_role,
        "seed $(Int(seed)) is not preregistered for role $role",
    ))
    return true
end

"""Return the two highest finite Q indices, preserving candidate order on ties."""
function stable_top_two(q_values::AbstractVector{<:Real})
    length(q_values) >= 2 || throw(ExcludableCollectionError(
        :candidate_count_lt2,
        "top-2 comparison requires at least two valid actions",
    ))
    all(isfinite, q_values) || throw(ExcludableCollectionError(
        :nonfinite_q,
        "old-Q produced a non-finite candidate value",
    ))
    top1 = 0
    top2 = 0
    @inbounds for index in eachindex(q_values)
        if top1 == 0 || q_values[index] > q_values[top1]
            top2 = top1
            top1 = index
        elseif top2 == 0 || q_values[index] > q_values[top2]
            top2 = index
        end
    end
    top2 != 0 || throw(ExcludableCollectionError(
        :candidate_count_lt2,
        "top-2 comparison did not produce two indices",
    ))
    return Int(top1), Int(top2)
end

function _finite_metric_vector(metrics, names, label::AbstractString)
    values = Vector{Float64}(undef, length(names))
    for (index, name) in pairs(names)
        hasproperty(metrics, name) || throw(ExcludableCollectionError(
            :feature_schema,
            "$label is missing metric $name",
        ))
        value = Float64(getproperty(metrics, name))
        isfinite(value) || throw(ExcludableCollectionError(
            :nonfinite_feature,
            "$label metric $name is non-finite",
        ))
        values[index] = value
    end
    return values
end

function _validate_queue_onehot(queue_onehot)
    values = Float64.(collect(queue_onehot))
    length(values) == 42 || throw(ExcludableCollectionError(
        :feature_schema,
        "HOLD/NEXT one-hot must contain 42 values, got $(length(values))",
    ))
    all(isfinite, values) || throw(ExcludableCollectionError(
        :nonfinite_feature,
        "HOLD/NEXT one-hot contains a non-finite value",
    ))
    all(value -> value == 0.0 || value == 1.0, values) || throw(
        ExcludableCollectionError(
            :feature_schema,
            "HOLD/NEXT values must be exactly zero or one",
        ),
    )
    for slot in 1:6
        slot_sum = sum(@view values[((slot - 1) * 7 + 1):(slot * 7)])
        (slot_sum == 0.0 || slot_sum == 1.0) || throw(ExcludableCollectionError(
            :feature_schema,
            "HOLD/NEXT slot $slot is not zero-or-one-hot",
        ))
    end
    return values
end

"""Build the frozen 70-column R1 feature vector.

`top1_metrics` and `top2_metrics` describe the two root afterstates.  The
difference is always encoded as top2 minus top1.  Population standardization is
within the complete candidate set.  A constant-Q set receives zero standardized
features.
"""
function build_feature_vector(
    q_values::AbstractVector{<:Real},
    top1_index::Integer,
    top2_index::Integer,
    top1_metrics,
    top2_metrics,
    current_metrics,
    queue_onehot,
)
    all(isfinite, q_values) || throw(ExcludableCollectionError(
        :nonfinite_q,
        "old-Q produced a non-finite candidate value",
    ))
    q = Float64.(q_values)
    q1 = q[top1_index]
    q2 = q[top2_index]
    q_mean = mean(q)
    q_scale = sqrt(sum(value -> abs2(value - q_mean), q) / length(q))
    if q_scale == 0.0
        z1 = 0.0
        z2 = 0.0
    else
        z1 = (q1 - q_mean) / q_scale
        z2 = (q2 - q_mean) / q_scale
    end

    action1 = _finite_metric_vector(top1_metrics, ACTION_METRIC_NAMES, "top1")
    action2 = _finite_metric_vector(top2_metrics, ACTION_METRIC_NAMES, "top2")
    current = _finite_metric_vector(current_metrics, CURRENT_METRIC_NAMES, "current")
    queue = _validate_queue_onehot(queue_onehot)
    result = vcat(
        Float64[q1, q2, q1 - q2, z1, z2, z1 - z2, length(q)],
        action2 .- action1,
        current,
        queue,
    )
    length(result) == length(FEATURE_SCHEMA) || error("internal R1 feature width error")
    all(isfinite, result) || throw(ExcludableCollectionError(
        :nonfinite_feature,
        "constructed feature vector is non-finite",
    ))
    return result
end

function _adapter_q(adapter, state, actions)
    raw = try
        adapter.q_values(state, actions)
    catch error
        error isa FatalCollectionInvariant && rethrow()
        throw(ExcludableCollectionError(:q_evaluation_failure, sprint(showerror, error)))
    end
    q = Float64.(collect(raw))
    length(q) == length(actions) || throw(ExcludableCollectionError(
        :q_shape,
        "old-Q returned $(length(q)) values for $(length(actions)) actions",
    ))
    all(isfinite, q) || throw(ExcludableCollectionError(
        :nonfinite_q,
        "old-Q produced a non-finite candidate value",
    ))
    return q
end

function _adapter_actions(adapter, state)
    actions = try
        collect(adapter.candidate_actions(state))
    catch error
        error isa FatalCollectionInvariant && rethrow()
        throw(ExcludableCollectionError(:branch_construction_failure, sprint(showerror, error)))
    end
    return actions
end


function _candidate_order_digest(adapter, actions)
    payload = join((String(adapter.action_digest(action)) for action in actions), "\n")
    return bytes2hex(sha256(payload))
end

function _q_vector_digest(q::AbstractVector{<:Real})
    # Digest exact Float64 bit patterns, not locale-sensitive decimal rendering.
    payload = join(
        (string(reinterpret(UInt64, Float64(value)); base=16, pad=16) for value in q),
        "\n",
    )
    return bytes2hex(sha256(payload))
end

function _decision_evidence(
    adapter,
    rollout_piece::Integer,
    kind::Symbol,
    actions,
    q::AbstractVector{<:Real},
    selected_index::Integer,
)
    1 <= selected_index <= length(actions) || throw(FatalCollectionInvariant(
        :selected_index,
        "selected index is outside the complete candidate list",
    ))
    return DecisionEvidence(
        Int(rollout_piece),
        kind,
        length(actions),
        _candidate_order_digest(adapter, actions),
        _q_vector_digest(q),
        Int(selected_index),
        String(adapter.action_digest(actions[selected_index])),
        Float64(q[selected_index]),
        Float64(maximum(q)),
    )
end

function _apply_branch_action!(adapter, state, action)
    try
        adapter.apply_action(state, action)
    catch error
        error isa FatalCollectionInvariant && rethrow()
        throw(ExcludableCollectionError(:branch_apply_failure, sprint(showerror, error)))
    end
    return state
end

"""Roll out one root action plus the canonical old policy for `horizon-1` pieces."""
function rollout_branch(
    adapter,
    root_state,
    root_action;
    gamma::Real=0.997,
    horizon::Integer=6,
    score_normalizer::Real=600.0,
    root_candidate_count::Integer,
    root_actions,
    root_q,
    root_selected_index::Integer,
)
    horizon == 6 || throw(FatalCollectionInvariant(
        :horizon,
        "R1 is frozen at a root-inclusive six-piece horizon",
    ))
    gamma == 0.997 || throw(FatalCollectionInvariant(
        :gamma,
        "R1 is frozen at gamma=0.997",
    ))
    score_normalizer == 600.0 || throw(FatalCollectionInvariant(
        :score_normalizer,
        "R1 is frozen at score_normalizer=600",
    ))

    # Production must implement this callback as `GameState(root_state)`.  A
    # generic `deepcopy` is deliberately not accepted by the engine adapter,
    # because its RNG/queue semantics have not been audited.
    state = try
        adapter.clone_state(root_state)
    catch error
        error isa FatalCollectionInvariant && rethrow()
        throw(ExcludableCollectionError(:branch_clone_failure, sprint(showerror, error)))
    end
    state === root_state && throw(FatalCollectionInvariant(
        :aliased_clone,
        "clone_state returned the canonical root object",
    ))
    branch_digest_before_oracle = String(adapter.state_digest(state))
    future_stream_digest = String(adapter.future_stream_digest(state, horizon))
    String(adapter.state_digest(state)) == branch_digest_before_oracle || throw(
        FatalCollectionInvariant(
            :future_oracle_mutated_branch,
            "future_stream_digest modified a branch-start clone",
        ),
    )
    score_deltas = Float64[]
    pre_action_current_piece_tokens = String[]
    placed_piece_tokens = String[]
    rng_digests = String[]
    selected_action_digests = String[]
    candidate_counts = Int[]
    decision_evidence = DecisionEvidence[]
    discounted_return = 0.0
    terminal_step = 0
    action = root_action

    for step in 1:horizon
        if step > 1
            actions = _adapter_actions(adapter, state)
            isempty(actions) && throw(ExcludableCollectionError(
                :branch_construction_failure,
                "non-terminal branch has no action at rollout step $step",
            ))
            q = _adapter_q(adapter, state, actions)
            action_index = if length(actions) == 1
                1
            else
                first(stable_top_two(q))
            end
            action = actions[action_index]
            push!(candidate_counts, length(actions))
            push!(decision_evidence, _decision_evidence(
                adapter, step, :action, actions, q, action_index
            ))
        else
            root_candidate_count >= 2 || throw(FatalCollectionInvariant(
                :root_candidate_count,
                "root candidate count must be at least two",
            ))
            length(root_actions) == root_candidate_count || throw(
                FatalCollectionInvariant(
                    :root_candidate_count,
                    "root action vector does not match its recorded count",
                ),
            )
            length(root_q) == root_candidate_count || throw(FatalCollectionInvariant(
                :root_q_shape,
                "root Q vector does not match its recorded count",
            ))
            String(adapter.action_digest(root_actions[root_selected_index])) ==
            String(adapter.action_digest(root_action)) || throw(FatalCollectionInvariant(
                :root_action_identity,
                "root selected index does not identify the supplied root action",
            ))
            push!(candidate_counts, Int(root_candidate_count))
            push!(decision_evidence, _decision_evidence(
                adapter, step, :action, root_actions, root_q, root_selected_index
            ))
        end

        push!(pre_action_current_piece_tokens, String(adapter.piece_token(state)))
        push!(placed_piece_tokens, String(adapter.placed_piece_token(action)))
        push!(selected_action_digests, String(adapter.action_digest(action)))
        score_before = Float64(adapter.score(state))
        _apply_branch_action!(adapter, state, action)
        score_after = Float64(adapter.score(state))
        delta = score_after - score_before
        isfinite(delta) || throw(ExcludableCollectionError(
            :nonfinite_return,
            "score delta is non-finite at rollout step $step",
        ))
        push!(score_deltas, delta)
        push!(rng_digests, String(adapter.rng_digest(state)))
        discounted_return += Float64(gamma)^(step - 1) * delta / Float64(score_normalizer)

        if adapter.terminal(state)
            terminal_step = step
            break
        end
    end

    bootstrap_q = 0.0
    if terminal_step == 0
        actions = _adapter_actions(adapter, state)
        isempty(actions) && throw(ExcludableCollectionError(
            :branch_construction_failure,
            "surviving s6 state has no bootstrap action",
        ))
        q = _adapter_q(adapter, state, actions)
        bootstrap_index = length(actions) == 1 ? 1 : first(stable_top_two(q))
        bootstrap_q = q[bootstrap_index]
        push!(decision_evidence, _decision_evidence(
            adapter, horizon + 1, :bootstrap, actions, q, bootstrap_index
        ))
        discounted_return += Float64(gamma)^horizon * bootstrap_q
    end
    isfinite(discounted_return) || throw(ExcludableCollectionError(
        :nonfinite_return,
        "G6 is non-finite",
    ))
    return BranchOutcome(
        discounted_return,
        terminal_step != 0,
        terminal_step,
        bootstrap_q,
        score_deltas,
        pre_action_current_piece_tokens,
        placed_piece_tokens,
        rng_digests,
        selected_action_digests,
        candidate_counts,
        future_stream_digest,
        decision_evidence,
    )
end

function _verify_common_future(top1::BranchOutcome, top2::BranchOutcome)
    top1.future_stream_digest == top2.future_stream_digest || throw(
        FatalCollectionInvariant(
            :future_stream_mismatch,
            "top-1 and top-2 branch future-stream digests differ",
        ),
    )
    # HOLD can consume the common source stream at different rates.  Therefore
    # consumed pieces and post-action RNG states are evidence to record, not an
    # equality invariant.  Equality is defined by the immutable branch-start
    # queue + serialized RNG + future-bag oracle digest above.
    return true
end

"""Collect one top-1/top-2 counterfactual row without mutating `root_state`."""
function collect_counterfactual(
    adapter,
    root_state;
    seed::Integer,
    episode_id::Integer,
    piece_index::Integer,
    root_actions=nothing,
    root_q=nothing,
)
    validate_adapter_contract(adapter.contract)
    root_digest_before = String(adapter.state_digest(root_state))
    actions = isnothing(root_actions) ? _adapter_actions(adapter, root_state) : root_actions
    length(actions) >= 2 || throw(ExcludableCollectionError(
        :candidate_count_lt2,
        "root state has fewer than two valid actions",
    ))
    q = isnothing(root_q) ? _adapter_q(adapter, root_state, actions) : Float64.(root_q)
    length(q) == length(actions) || throw(ExcludableCollectionError(
        :q_shape,
        "root Q/action lengths differ",
    ))
    top1_index, top2_index = stable_top_two(q)
    top1_action = actions[top1_index]
    top2_action = actions[top2_index]

    top1_metrics = try
        adapter.action_metrics(root_state, top1_action)
    catch error
        throw(ExcludableCollectionError(:feature_extraction_failure, sprint(showerror, error)))
    end
    top2_metrics = try
        adapter.action_metrics(root_state, top2_action)
    catch error
        throw(ExcludableCollectionError(:feature_extraction_failure, sprint(showerror, error)))
    end
    current_metrics = try
        adapter.current_metrics(root_state)
    catch error
        throw(ExcludableCollectionError(:feature_extraction_failure, sprint(showerror, error)))
    end
    queue_onehot = try
        adapter.queue_onehot(root_state)
    catch error
        throw(ExcludableCollectionError(:feature_extraction_failure, sprint(showerror, error)))
    end
    features = build_feature_vector(
        q,
        top1_index,
        top2_index,
        top1_metrics,
        top2_metrics,
        current_metrics,
        queue_onehot,
    )

    top1_branch = rollout_branch(
        adapter,
        root_state,
        top1_action;
        root_candidate_count=length(actions),
        root_actions=actions,
        root_q=q,
        root_selected_index=top1_index,
    )
    top2_branch = rollout_branch(
        adapter,
        root_state,
        top2_action;
        root_candidate_count=length(actions),
        root_actions=actions,
        root_q=q,
        root_selected_index=top2_index,
    )
    _verify_common_future(top1_branch, top2_branch)
    root_digest_after = String(adapter.state_digest(root_state))
    root_digest_after == root_digest_before || throw(FatalCollectionInvariant(
        :canonical_state_mutated,
        "counterfactual rollout modified the canonical trajectory root",
    ))
    root_future_stream_digest = String(adapter.future_stream_digest(root_state, 6))
    String(adapter.state_digest(root_state)) == root_digest_before || throw(
        FatalCollectionInvariant(
            :future_oracle_mutated_root,
            "future_stream_digest modified the canonical root",
        ),
    )
    top1_branch.future_stream_digest == root_future_stream_digest || throw(
        FatalCollectionInvariant(
            :root_future_stream_mismatch,
            "branch future stream differs from the canonical root",
        ),
    )

    advantage = top2_branch.return_value - top1_branch.return_value
    isfinite(advantage) || throw(ExcludableCollectionError(
        :nonfinite_return,
        "A6 is non-finite",
    ))
    return CounterfactualRow(
        Int(seed),
        Int(episode_id),
        Int(piece_index),
        root_digest_before,
        root_future_stream_digest,
        top1_index,
        top2_index,
        String(adapter.action_digest(top1_action)),
        String(adapter.action_digest(top2_action)),
        q[top1_index],
        q[top2_index],
        q[top1_index] - q[top2_index],
        length(actions),
        features,
        top1_branch.return_value,
        top2_branch.return_value,
        advantage,
        clamp(advantage, -2.0, 2.0),
        top1_branch,
        top2_branch,
    )
end

"""Follow the canonical old policy and sample only the frozen piece positions.

Excludable sample failures are recorded and do not change the canonical action.
Fatal invariants (notably future/RNG mismatch or canonical mutation) abort the
whole collection.  Counterfactual root actions are never applied to the
canonical state.
"""
function collect_episode(
    adapter,
    seed::Integer;
    episode_id::Integer=1,
    sample_pieces=R1_SAMPLE_PIECES,
    max_pieces::Integer=240,
    on_sample_complete=event -> nothing,
)
    validate_adapter_contract(adapter.contract)
    max_pieces == 240 || throw(FatalCollectionInvariant(
        :max_pieces,
        "R1 data collection is frozen at 240 canonical pieces",
    ))
    state = adapter.initial_state(Int(seed))
    rows = CounterfactualRow[]
    exclusions = ExcludedState[]
    canonical_action_digests = String[]
    sample_set = Set(Int.(sample_pieces))
    sample_set == Set(R1_SAMPLE_PIECES) || throw(FatalCollectionInvariant(
        :sample_schedule,
        "R1 sampling must be exactly pieces 10,20,...,240",
    ))
    pieces = 0

    for piece_index in 1:max_pieces
        adapter.terminal(state) && break
        actions = _adapter_actions(adapter, state)
        isempty(actions) && break
        q = _adapter_q(adapter, state, actions)
        top1_index = if length(actions) == 1
            1
        else
            first(stable_top_two(q))
        end

        if piece_index in sample_set
            root_digest = String(adapter.state_digest(state))
            if length(actions) < 2
                push!(exclusions, ExcludedState(
                    Int(seed), Int(episode_id), piece_index, :candidate_count_lt2,
                    "root state has fewer than two valid actions", root_digest,
                ))
                on_sample_complete((;
                    seed=Int(seed),
                    episode_id=Int(episode_id),
                    piece_index,
                    status=:excluded,
                    exclusion_code=:candidate_count_lt2,
                ))
            else
                try
                    row = collect_counterfactual(
                        adapter,
                        state;
                        seed,
                        episode_id,
                        piece_index,
                        root_actions=actions,
                        root_q=q,
                    )
                    push!(rows, row)
                    on_sample_complete((;
                        seed=Int(seed),
                        episode_id=Int(episode_id),
                        piece_index,
                        status=:row,
                        exclusion_code=:none,
                    ))
                catch error
                    if error isa ExcludableCollectionError
                        push!(exclusions, ExcludedState(
                            Int(seed),
                            Int(episode_id),
                            piece_index,
                            error.code,
                            error.detail,
                            root_digest,
                        ))
                        on_sample_complete((;
                            seed=Int(seed),
                            episode_id=Int(episode_id),
                            piece_index,
                            status=:excluded,
                            exclusion_code=error.code,
                        ))
                    else
                        rethrow()
                    end
                end
            end
            String(adapter.state_digest(state)) == root_digest || throw(
                FatalCollectionInvariant(
                    :canonical_state_mutated,
                    "sample collection modified the canonical state at piece $piece_index",
                ),
            )
        end

        chosen_action = actions[top1_index]
        push!(canonical_action_digests, String(adapter.action_digest(chosen_action)))
        adapter.apply_action(state, chosen_action)
        pieces += 1
    end
    return EpisodeCollection(
        Int(seed),
        Int(episode_id),
        rows,
        exclusions,
        canonical_action_digests,
        Bool(adapter.terminal(state)),
        pieces,
        Float64(adapter.score(state)),
    )
end

function collect_role_episode(
    adapter,
    seed::Integer,
    role::Symbol;
    episode_id::Integer,
    on_sample_complete=event -> nothing,
)
    validate_role_seed(seed, role)
    return collect_episode(adapter, seed; episode_id, on_sample_complete)
end

"""Return the fixed numeric table consumed by the later analytic fitter."""
function numeric_table(rows::AbstractVector{CounterfactualRow})
    features = Matrix{Float64}(undef, length(rows), length(FEATURE_SCHEMA))
    for (row_index, row) in pairs(rows)
        length(row.features) == length(FEATURE_SCHEMA) || throw(
            FatalCollectionInvariant(
                :feature_schema,
                "row $row_index has a non-canonical feature width",
            ),
        )
        features[row_index, :] .= row.features
    end
    return (;
        features,
        seed=Int[row.seed for row in rows],
        episode_id=Int[row.episode_id for row in rows],
        piece_index=Int[row.piece_index for row in rows],
        q_top1=Float64[row.q_top1 for row in rows],
        q_top2=Float64[row.q_top2 for row in rows],
        q_gap=Float64[row.q_gap for row in rows],
        valid_action_count=Int[row.valid_action_count for row in rows],
        g6_top1=Float64[row.g6_top1 for row in rows],
        g6_top2=Float64[row.g6_top2 for row in rows],
        advantage_unclipped=Float64[row.advantage_unclipped for row in rows],
        target_clipped=Float64[row.target_clipped for row in rows],
    )
end

end # module
