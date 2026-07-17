module R1ProductionEngineAdapter

using Random
using SHA

using ..R1CounterfactualCollector

export EngineAccounting,
       FUTURE_ORACLE_BAGS,
       build_production_adapter,
       make_production_adapter,
       covered_cells,
       engine_piece_token,
       node_action_digest,
       rng_digest,
       strong_state_digest

const FUTURE_ORACLE_BAGS = 4
const RIDGE_PIECE_TOKENS = ("I", "O", "S", "Z", "J", "L", "T")

mutable struct EngineAccounting
    logical_network_calls::Int
    physical_network_calls::Int
    candidate_evaluations::Int
end

EngineAccounting() = EngineAccounting(0, 0, 0)

_hexbytes(bytes) = bytes2hex(sha256(bytes))
_hextext(text::AbstractString) = _hexbytes(codeunits(text))

function _rng_words(rng)
    fieldnames(typeof(rng)) == (:s0, :s1, :s2, :s3, :s4) || error(
        "R1 requires Julia 1.12 Xoshiro with fields s0:s4",
    )
    return ntuple(index -> UInt64(getfield(rng, index)), 5)
end

function rng_digest(rng)
    payload = join((string(word; base=16, pad=16) for word in _rng_words(rng)), ":")
    return _hextext(payload)
end

function engine_piece_token(mino, mino_types)
    isnothing(mino) && return "none"
    index = findfirst(piece_type -> typeof(mino) === piece_type, mino_types)
    isnothing(index) && error("piece type is outside frozen Tetris.MINOS")
    return RIDGE_PIECE_TOKENS[index]
end

function _matrix_payload(matrix)
    values = join((string(Int(value)) for value in vec(matrix)), ",")
    return "$(size(matrix, 1))x$(size(matrix, 2))[$values]"
end

function _mino_payload(mino, mino_types)
    isnothing(mino) && return "none"
    return join((
        engine_piece_token(mino, mino_types),
        string(Int(mino.direction)),
        _matrix_payload(mino.block),
    ), "|")
end

function strong_state_digest(state, mino_types)
    queue = join((_mino_payload(mino, mino_types) for mino in state.mino_list), ";")
    payload = join((
        "binary=" * _matrix_payload(state.current_game_board.binary),
        "color=" * _matrix_payload(state.current_game_board.color),
        "current=" * _mino_payload(state.current_mino, mino_types),
        "position=$(Int(state.current_position.x)),$(Int(state.current_position.y))",
        "hold=" * _mino_payload(state.hold_mino, mino_types),
        "queue=$queue",
        "score=$(state.score)",
        "ren=$(Int(state.ren))",
        "ren_flag=$(state.ren_flag)",
        "b2b=$(state.back_to_back_flag)",
        "game_over=$(state.game_over_flag)",
        "hold_flag=$(state.hold_flag)",
        "hard_drop=$(state.hard_drop_flag)",
        "tspin_flag=$(state.t_spin_flag)",
        "srs=$(Int(state.srs_index))",
        "rng=" * join((string(word; base=16, pad=16) for word in _rng_words(state.rng)), ":"),
    ), "\n")
    return _hextext(payload)
end

function _action_payload(action)
    fields = fieldnames(typeof(action))
    values = isempty(fields) ? "" : join((
        "$(name)=$(getfield(action, name))" for name in fields
    ), ",")
    return "$(nameof(typeof(action)))($values)"
end

function node_action_digest(node, mino_types)
    action_flow = join((_action_payload(action) for action in node.action_list), ";")
    payload = join((
        action_flow,
        _mino_payload(node.mino, mino_types),
        "position=$(Int(node.position.x)),$(Int(node.position.y))",
        "tspin=$(Int(node.tspin))",
    ), "|")
    return _hextext(payload)
end

"""Count unique occupied cells above one or more holes in the same column."""
function covered_cells(board::AbstractMatrix)
    height, width = size(board)
    covered = 0
    @inbounds for column in 1:width
        hole_below = false
        for row in height:-1:1
            occupied = !iszero(board[row, column])
            if occupied
                covered += hole_below
            else
                hole_below = true
            end
        end
    end
    return covered
end

function _board_metrics(board, board_features_fn)
    features = board_features_fn(board)
    return (;
        holes=features.holes,
        covered_cells=covered_cells(board),
        aggregate_height=features.aggregate_height,
        max_height=features.max_height,
        bumpiness=features.bumpiness,
        well_sum=features.wells,
        row_transitions=features.row_transitions,
        column_transitions=features.column_transitions,
    )
end

function _future_source_digest(state, horizon, mino_types, generate_mino_list_fn)
    horizon == 6 || error("R1 future-source digest is frozen at horizon 6")
    before = strong_state_digest(state, mino_types)
    raw_pop_queue = join((
        engine_piece_token(state.mino_list[index], mino_types)
        for index in length(state.mino_list):-1:1
    ), ",")
    oracle_rng = copy(state.rng)
    oracle = String[]
    for _ in 1:FUTURE_ORACLE_BAGS
        bag = generate_mino_list_fn(oracle_rng)
        push!(oracle, join((
            engine_piece_token(bag[index], mino_types) for index in length(bag):-1:1
        ), ","))
    end
    payload = join((
        "queue_pop=$raw_pop_queue",
        "rng=" * join((string(word; base=16, pad=16) for word in _rng_words(state.rng)), ":"),
        "oracle=" * join(oracle, ";"),
    ), "\n")
    after = strong_state_digest(state, mino_types)
    before == after || throw(FatalCollectionInvariant(
        :future_oracle_mutated_state,
        "future stream oracle modified its input state",
    ))
    return _hextext(payload)
end

function _queue_onehot(state, mino_types)
    # Logical feature order is HOLD, immediate NEXT, ..., fifth NEXT.  The old
    # network input remains untouched and uses its historical end-4:end order.
    pieces = Any[state.hold_mino]
    append!(pieces, (state.mino_list[index] for index in length(state.mino_list):-1:(length(state.mino_list)-4)))
    output = zeros(Float64, 6 * 7)
    for (slot, mino) in pairs(pieces)
        isnothing(mino) && continue
        index = findfirst(piece_type -> typeof(mino) === piece_type, mino_types)
        isnothing(index) && error("queue contains an unknown piece type")
        output[(slot - 1) * 7 + index] = 1.0
    end
    return output
end

"""Construct the production adapter without changing any engine semantics.

Every engine function is passed explicitly by the production entrypoint.  This
keeps this file synthetically testable while letting the readiness manifest
bind the exact source hashes of `stable_node_key`, `stable_node_list`,
`legacy_candidate_batch`, and `apply_node!`.
"""
function build_production_adapter(
    inference;
    game_state_fn,
    stable_node_list_fn,
    stable_node_key_fn,
    legacy_candidate_batch_fn,
    openvino_scores_fn,
    apply_node_fn,
    board_features_fn,
    generate_mino_list_fn,
    mino_types,
    accounting::EngineAccounting=EngineAccounting(),
)
    Tuple(engine_piece_token(piece_type(), mino_types) for piece_type in mino_types) ==
        RIDGE_PIECE_TOKENS || error("Tetris.MINOS type order differs from frozen R1 order")

    state_digest_fn = state -> strong_state_digest(state, mino_types)
    action_digest_fn = node -> node_action_digest(node, mino_types)

    function clone_state_fn(state)
        clone = game_state_fn(state)
        clone === state && throw(FatalCollectionInvariant(:aliased_clone, "GameState(root) aliased root"))
        clone.current_game_board.binary === state.current_game_board.binary && throw(
            FatalCollectionInvariant(:aliased_board, "GameState(root) shares binary board"),
        )
        clone.current_game_board.color === state.current_game_board.color && throw(
            FatalCollectionInvariant(:aliased_board, "GameState(root) shares color board"),
        )
        clone.mino_list === state.mino_list && throw(
            FatalCollectionInvariant(:aliased_queue, "GameState(root) shares queue vector"),
        )
        clone.rng === state.rng && throw(
            FatalCollectionInvariant(:aliased_rng, "GameState(root) shares RNG object"),
        )
        state_digest_fn(clone) == state_digest_fn(state) || throw(
            FatalCollectionInvariant(:clone_mismatch, "GameState(root) changed canonical state"),
        )
        return clone
    end

    function candidate_actions_fn(state)
        before = state_digest_fn(state)
        nodes = stable_node_list_fn(state)
        state_digest_fn(state) == before || throw(FatalCollectionInvariant(
            :candidate_generation_mutated_state,
            "stable_node_list modified the scored state",
        ))
        keys = stable_node_key_fn.(nodes)
        length(unique(keys)) == length(keys) || throw(FatalCollectionInvariant(
            :duplicate_stable_node_key,
            "stable_node_key is not unique at a collected state",
        ))
        digests = action_digest_fn.(nodes)
        length(unique(digests)) == length(digests) || throw(FatalCollectionInvariant(
            :duplicate_action_digest,
            "candidate action digests are not unique",
        ))
        return nodes
    end

    function q_values_fn(state, nodes)
        before = state_digest_fn(state)
        input = legacy_candidate_batch_fn(state, nodes; next_count=5)
        scores = openvino_scores_fn(inference, input)
        state_digest_fn(state) == before || throw(FatalCollectionInvariant(
            :q_evaluation_mutated_state,
            "old-Q evaluation modified the scored state",
        ))
        length(scores) == length(nodes) || error("old-Q candidate count mismatch")
        all(isfinite, scores) || error("old-Q returned a non-finite value")
        accounting.logical_network_calls += 1
        accounting.physical_network_calls += cld(length(nodes), 16)
        accounting.candidate_evaluations += length(nodes)
        return scores
    end

    function apply_action_fn(state, node)
        expected = state_digest_fn(node.game_state)
        apply_node_fn(state, node)
        observed = state_digest_fn(state)
        observed == expected || throw(FatalCollectionInvariant(
            :node_replay_mismatch,
            "apply_node!(GameState(root), node) differs from node.game_state",
        ))
        return state
    end

    function action_metrics_fn(root, node)
        safety = _board_metrics(node.game_state.current_game_board.binary, board_features_fn)
        cleared = (sum(root.current_game_board.binary) + 4 -
                   sum(node.game_state.current_game_board.binary)) ÷ 10
        return (;
            immediate_score=node.game_state.score - root.score,
            cleared_lines=cleared,
            safety...,
            ren=Int(node.game_state.ren),
            back_to_back=Int(node.game_state.back_to_back_flag),
            tspin=Int(node.tspin > 1),
        )
    end

    contract = AdapterContract(
        candidate_order=:stable_node_key,
        q_chunk_size=16,
        q_tail_mode=:actual,
        next_count=5,
        hold_enabled=true,
    )
    adapter = (;
        contract,
        initial_state=seed -> game_state_fn(Xoshiro(seed)),
        clone_state=clone_state_fn,
        terminal=state -> state.game_over_flag,
        score=state -> state.score,
        candidate_actions=candidate_actions_fn,
        q_values=q_values_fn,
        apply_action=apply_action_fn,
        state_digest=state_digest_fn,
        future_stream_digest=(state, horizon) -> _future_source_digest(
            state, horizon, mino_types, generate_mino_list_fn,
        ),
        piece_token=state -> engine_piece_token(state.current_mino, mino_types),
        placed_piece_token=node -> engine_piece_token(node.mino, mino_types),
        rng_digest=state -> rng_digest(state.rng),
        action_digest=action_digest_fn,
        action_metrics=action_metrics_fn,
        current_metrics=state -> _board_metrics(
            state.current_game_board.binary, board_features_fn,
        ),
        queue_onehot=state -> _queue_onehot(state, mino_types),
        accounting,
    )
    validate_adapter_contract(adapter.contract)
    return adapter
end

"""Load and bind the exact historical engine/OpenVINO production path.

This function is intentionally called only after `collect_online.jl` has
published its pre-import milestones.  Merely including `engine_adapter.jl`
does not load the game, checkpoint weights, Python, or OpenVINO.
"""
function _construct_after_evaluator(
    repository_root::AbstractString,
    evaluator_path::AbstractString,
    engine_path::AbstractString,
    weights_sha::AbstractString,
    checkpoint_path::AbstractString,
    checkpoint_sha::AbstractString,
)
    required = (
        :GameState, :stable_node_list, :stable_node_key, :legacy_candidate_batch,
        :openvino_scores, :apply_node!, :board_features, :Tetris,
    )
    all(name -> isdefined(Main, name), required) || error(
        "canonical evaluator did not define the required engine bindings",
    )
    sys = Main.pyimport("sys")
    tools_path = joinpath(repository_root, "tools")
    sys.path.insert(0, tools_path)
    legacy_openvino = Main.pyimport("legacy_openvino")
    openvino = Main.pyimport("openvino")
    version = Main.pyconvert(String, openvino.__version__)
    startswith(version, "2026.2.1") || error("OpenVINO runtime version mismatch: $version")
    inference = legacy_openvino.LegacyOpenVINOInference("NPU", 16)

    mino_types = Tuple(typeof(piece) for piece in Main.Tetris.MINOS)
    adapter = build_production_adapter(
        inference;
        game_state_fn=Main.GameState,
        stable_node_list_fn=Main.stable_node_list,
        stable_node_key_fn=Main.stable_node_key,
        legacy_candidate_batch_fn=Main.legacy_candidate_batch,
        openvino_scores_fn=Main.openvino_scores,
        apply_node_fn=Main.apply_node!,
        board_features_fn=Main.board_features,
        generate_mino_list_fn=Main.Tetris.generate_mino_list,
        mino_types,
    )
    return merge(adapter, (;
        stable_node_key_source_sha256=bytes2hex(open(sha256, engine_path)),
        evaluator_source_sha256=bytes2hex(open(sha256, evaluator_path)),
        old_openvino_weight_npz_sha256=String(weights_sha),
        old_checkpoint_path=String(checkpoint_path),
        old_checkpoint_sha256=String(checkpoint_sha),
        old_openvino_weight_npz_path=joinpath(
            repository_root, "artifacts", "legacy_openvino", "legacy_1313_weights.npz",
        ),
        openvino_version=version,
        openvino_complete_device="NPU",
        openvino_tail_device="CPU",
        openvino_batch_size=16,
    ))
end

function make_production_adapter()
    repository_root = normpath(joinpath(@__DIR__, "..", ".."))
    evaluator_path = joinpath(repository_root, "scripts", "evaluate_openvino_checkpoint.jl")
    engine_path = joinpath(repository_root, "scripts", "benchmark_legacy_engine.jl")
    weights_path = joinpath(
        repository_root, "artifacts", "legacy_openvino", "legacy_1313_weights.npz",
    )
    checkpoint_path = joinpath(repository_root, "1313", "mainmodel copy 3.jld2")
    isfile(evaluator_path) || error("missing canonical OpenVINO evaluator")
    isfile(engine_path) || error("missing canonical stable-node engine")
    isfile(weights_path) || error("missing canonical OpenVINO weights")
    isfile(checkpoint_path) || error("missing canonical old checkpoint")
    weights_sha = bytes2hex(open(sha256, weights_path))
    weights_sha == "2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba" ||
        error("canonical OpenVINO weight hash mismatch")
    checkpoint_sha = bytes2hex(open(sha256, checkpoint_path))
    checkpoint_sha == "7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1" ||
        error("canonical old checkpoint hash mismatch")

    # The canonical script recursively includes the exact engine and imports
    # PythonCall.  Load into Main because all existing evaluation functions are
    # defined there and are already source-audited by the repository harness.
    isdefined(Main, :evaluate_openvino_episode) || Base.include(Main, evaluator_path)
    return Base.invokelatest(
        _construct_after_evaluator,
        repository_root,
        evaluator_path,
        engine_path,
        weights_sha,
        checkpoint_path,
        checkpoint_sha,
    )
end

end # module
