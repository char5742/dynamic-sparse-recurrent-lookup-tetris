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
    evaluator_module::Module,
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
    all(name -> isdefined(evaluator_module, name), required) || error(
        "canonical evaluator did not define the required engine bindings",
    )
    for name in (
        :stable_node_list, :stable_node_key, :legacy_candidate_batch,
        :openvino_scores, :apply_node!, :board_features,
    )
        parentmodule(getfield(evaluator_module, name)) === evaluator_module || error(
            "evaluator binding $name was imported from an ambient module",
        )
    end
    parentmodule(getfield(evaluator_module, :GameState)) ===
        getfield(evaluator_module, :Tetris) || error(
        "GameState was not imported from the frozen Tetris package",
    )
    pyimport = getfield(evaluator_module, :pyimport)
    pyconvert = getfield(evaluator_module, :pyconvert)
    legacy_openvino_path = joinpath(repository_root, "tools", "legacy_openvino.py")
    isfile(legacy_openvino_path) || error("missing frozen legacy_openvino.py")
    legacy_openvino_sha = bytes2hex(open(sha256, legacy_openvino_path))
    importlib = pyimport("importlib.util")
    module_name = "r1_legacy_openvino_$(legacy_openvino_sha[1:16])"
    spec = importlib.spec_from_file_location(module_name, legacy_openvino_path)
    isnothing(spec) && error("failed to create exact-path legacy_openvino module spec")
    legacy_openvino = importlib.module_from_spec(spec)
    spec.loader.exec_module(legacy_openvino)
    loaded_python_path = normpath(abspath(pyconvert(String, legacy_openvino.__file__)))
    loaded_python_path == normpath(abspath(legacy_openvino_path)) || error(
        "legacy_openvino loaded from an ambient path: $loaded_python_path",
    )
    openvino = pyimport("openvino")
    version = pyconvert(String, openvino.__version__)
    version == "2026.2.1-21919-ede283a88e3-releases/2026/2" || error(
        "OpenVINO runtime build mismatch: $version",
    )
    inference = legacy_openvino.LegacyOpenVINOInference("NPU", 16)

    engine = evaluator_module
    tetris = getfield(engine, :Tetris)
    mino_types = Tuple(typeof(piece) for piece in tetris.MINOS)
    adapter = build_production_adapter(
        inference;
        game_state_fn=getfield(engine, :GameState),
        stable_node_list_fn=getfield(engine, :stable_node_list),
        stable_node_key_fn=getfield(engine, :stable_node_key),
        legacy_candidate_batch_fn=getfield(engine, :legacy_candidate_batch),
        openvino_scores_fn=getfield(engine, :openvino_scores),
        apply_node_fn=getfield(engine, :apply_node!),
        board_features_fn=getfield(engine, :board_features),
        generate_mino_list_fn=tetris.generate_mino_list,
        mino_types,
    )
    return merge(adapter, (;
        stable_node_key_source_sha256=bytes2hex(open(sha256, engine_path)),
        evaluator_source_sha256=bytes2hex(open(sha256, evaluator_path)),
        legacy_openvino_source_sha256=legacy_openvino_sha,
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

const _FROZEN_NODE_SHA256 =
    "e98d2052f9248f5c08c1eb58adaace1bd01533f287e682bf35a2fefa1325fe82"
const _FROZEN_ANALYZER_SHA256 =
    "24152e2549dcc6c3c25d928454268e8baaa4d45fea31044603917cfbabbe02bc"

function _dependency_record(repository_root::AbstractString, relative_path::AbstractString)
    normalized_relative = replace(String(relative_path), '\\' => '/')
    path = joinpath(repository_root, split(normalized_relative, '/')...)
    isfile(path) || error("missing engine dependency: $normalized_relative")
    return (;
        path=normalized_relative,
        bytes=filesize(path),
        sha256=bytes2hex(open(sha256, path)),
    )
end

function _dependency_digest(records)
    payload = IOBuffer()
    for record in sort(collect(records); by=item -> item.path)
        write(payload, record.path, UInt8(0), lowercase(record.sha256), '\n')
    end
    return bytes2hex(sha256(take!(payload)))
end

function _engine_dependency_graph(repository_root::AbstractString)
    relative_paths = (
        "experiments/online_counterfactual_top2_r1/engine_adapter.jl",
        "experiments/online_counterfactual_top2_r1/vendor/TetrisAI/src/core/analyzer.jl",
        "experiments/online_counterfactual_top2_r1/vendor/TetrisAI/src/core/components/node.jl",
        "vendor/Tetris/lib/curses.jl",
        "vendor/Tetris/lib/game.so",
        "vendor/Tetris/lib/key_input.jl",
        "vendor/Tetris/lib/pdcurses.dll",
    )
    records = sort(
        [_dependency_record(repository_root, path) for path in relative_paths];
        by=item -> item.path,
    )
    node = only(filter(item -> endswith(item.path, "/components/node.jl"), records))
    analyzer = only(filter(item -> endswith(item.path, "/analyzer.jl"), records))
    node.sha256 == _FROZEN_NODE_SHA256 || error("vendored Node graph hash mismatch")
    analyzer.sha256 == _FROZEN_ANALYZER_SHA256 || error(
        "vendored analyzer graph hash mismatch",
    )

    upstream = joinpath(repository_root, "upstream", "TetrisAI")
    upstream_head = readchomp(`git -C $upstream rev-parse HEAD`)
    upstream_status = read(`git -C $upstream status --porcelain=v1 --untracked-files=all`, String)
    upstream_head == "6fdfb1d30197246fd862b716438e998f0315c830" || error(
        "nested upstream TetrisAI HEAD mismatch",
    )
    isempty(upstream_status) || error("nested upstream TetrisAI repository is not clean")
    runtime_records = filter(
        item -> item.path !=
                "experiments/online_counterfactual_top2_r1/engine_adapter.jl",
        records,
    )
    return (;
        schema_version="r1-engine-dependency-graph-v1",
        encoding="sorted relative_path + NUL + lowercase sha256 + newline",
        upstream_tetrisai=(; head=upstream_head, clean=true),
        records,
        graph_sha256=_dependency_digest(records),
        runtime_closure_sha256=_dependency_digest(runtime_records),
        node_source_sha256=node.sha256,
        analyzer_source_sha256=analyzer.sha256,
    )
end

function _load_frozen_evaluator(repository_root::AbstractString, evaluator_path::AbstractString)
    original_node = normpath(joinpath(
        repository_root, "upstream", "TetrisAI", "src", "core", "components", "node.jl",
    ))
    original_analyzer = normpath(joinpath(
        repository_root, "upstream", "TetrisAI", "src", "core", "analyzer.jl",
    ))
    vendored_node = normpath(joinpath(
        @__DIR__, "vendor", "TetrisAI", "src", "core", "components", "node.jl",
    ))
    vendored_analyzer = normpath(joinpath(
        @__DIR__, "vendor", "TetrisAI", "src", "core", "analyzer.jl",
    ))
    bytes2hex(open(sha256, vendored_node)) == _FROZEN_NODE_SHA256 || error(
        "vendored Node source hash mismatch",
    )
    bytes2hex(open(sha256, vendored_analyzer)) == _FROZEN_ANALYZER_SHA256 || error(
        "vendored analyzer source hash mismatch",
    )

    evaluate_legacy = normpath(joinpath(repository_root, "scripts", "evaluate_legacy_checkpoint.jl"))
    benchmark = normpath(joinpath(repository_root, "scripts", "benchmark_legacy_engine.jl"))
    artifact_helpers = normpath(joinpath(repository_root, "scripts", "evaluation_artifact_helpers.jl"))
    redirects = Dict(original_node => vendored_node, original_analyzer => vendored_analyzer)
    allowed = Set((evaluate_legacy, benchmark, original_node, original_analyzer, artifact_helpers))
    evaluator_module = Module(gensym(:R1FrozenLegacyEvaluator), true, false)
    Core.eval(evaluator_module, :(const _R1_INCLUDE_REDIRECTS = $redirects))
    Core.eval(evaluator_module, :(const _R1_ALLOWED_INCLUDES = $allowed))
    Core.eval(evaluator_module, :(const _R1_SEEN_INCLUDES = String[]))
    Core.eval(evaluator_module, quote
        function include(path::AbstractString)
            source = normpath(abspath(path))
            source in _R1_ALLOWED_INCLUDES || error(
                "unexpected recursive include in frozen legacy evaluator: $source",
            )
            push!(_R1_SEEN_INCLUDES, source)
            target = get(_R1_INCLUDE_REDIRECTS, source, source)
            return Base.include(@__MODULE__, target)
        end
    end)
    Base.include(evaluator_module, evaluator_path)
    seen_values = String.(Core.eval(evaluator_module, :(_R1_SEEN_INCLUDES)))
    length(seen_values) == length(allowed) && Set(seen_values) == allowed || error(
        "frozen legacy evaluator include closure differs from the audited graph",
    )
    return evaluator_module
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

    # Load the tracked evaluator chain into a fresh private module.  Its
    # module-local include function remaps only the two historical ignored
    # TetrisAI leaves to byte-identical, tracked R1 vendor copies and rejects
    # any unexpected recursive include.  Ambient Main bindings are never used.
    evaluator_module = _load_frozen_evaluator(repository_root, evaluator_path)
    adapter = Base.invokelatest(
        _construct_after_evaluator,
        evaluator_module,
        repository_root,
        evaluator_path,
        engine_path,
        weights_sha,
        checkpoint_path,
        checkpoint_sha,
    )
    return merge(adapter, (;
        engine_dependency_graph=_engine_dependency_graph(repository_root),
    ))
end

end # module
