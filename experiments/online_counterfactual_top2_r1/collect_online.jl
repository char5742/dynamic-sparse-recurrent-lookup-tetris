# This prefix deliberately uses Base only.  Production may import OpenVINO and
# the historical engine later, but memory telemetry must see script entry first.
const _R1_COLLECT_STARTED_AT = time()

function _early_json_string(value::AbstractString)
    escaped = replace(value, '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n", '\r' => "\\r")
    return "\"$escaped\""
end

function _append_milestone(path::AbstractString, event::AbstractString; detail::AbstractString="")
    line = "{\"event\":" * _early_json_string(event) *
           ",\"detail\":" * _early_json_string(detail) *
           ",\"pid\":" * string(getpid()) *
           ",\"unix_seconds\":" * string(time()) * "}\n"
    open(path, "a") do io
        write(io, line)
        flush(io)
    end
    return nothing
end

function _parse_early_args(args)
    (length(args) == 4 || (length(args) == 5 && args[5] == "--synthetic")) || error(
        "usage: collect_online.jl <train|calibration> <table.json> " *
        "<manifest.json> <milestones.jsonl> [--synthetic]"
    )
    role = Symbol(args[1])
    role in (:train, :calibration) || error("role must be train or calibration")
    table_path = abspath(args[2])
    manifest_path = abspath(args[3])
    milestone_path = abspath(args[4])
    length(unique((table_path, manifest_path, milestone_path))) == 3 || error(
        "table, manifest, and milestone paths must be distinct"
    )
    for path in (table_path, manifest_path, milestone_path)
        ispath(path) && error("refusing to overwrite R1 collection artifact: $path")
        mkpath(dirname(path))
    end
    return (;
        role,
        table_path,
        manifest_path,
        milestone_path,
        synthetic=length(args) == 5,
    )
end

function _atomic_json_create(path::AbstractString, value, JSON3)
    ispath(path) && error("refusing to overwrite R1 collection artifact: $path")
    temporary = "$path.tmp.$(getpid())"
    ispath(temporary) && error("stale R1 temporary artifact: $temporary")
    try
        open(temporary, "w") do io
            JSON3.pretty(io, value)
            write(io, '\n')
            flush(io)
        end
        # Refuse a last-moment collision instead of replacing an existing file.
        ispath(path) && error("R1 output appeared during write: $path")
        mv(temporary, path)
    finally
        ispath(temporary) && rm(temporary; force=true)
    end
    return path
end

function _branch_payload(branch)
    evidence = [(;
        rollout_piece=item.rollout_piece,
        kind=String(item.kind),
        candidate_count=item.candidate_count,
        candidate_order_digest=item.candidate_order_digest,
        q_vector_digest=item.q_vector_digest,
        selected_index=item.selected_index,
        selected_action_digest=item.selected_action_digest,
        selected_q=item.selected_q,
        max_q=item.max_q,
    ) for item in branch.decision_evidence]
    return (;
        return_G6=branch.return_value,
        terminal_within_horizon=branch.terminal,
        terminal_step=branch.terminal_step,
        bootstrap_q=branch.bootstrap_q,
        score_deltas=copy(branch.score_deltas),
        pre_action_current_piece_tokens=copy(branch.pre_action_current_piece_tokens),
        placed_piece_tokens=copy(branch.placed_piece_tokens),
        post_action_rng_digests=copy(branch.rng_digests),
        final_rng_digest=(isempty(branch.rng_digests) ? "" : branch.rng_digests[end]),
        selected_action_digests=copy(branch.selected_action_digests),
        candidate_counts=copy(branch.candidate_counts),
        branch_start_future_stream_digest=branch.future_stream_digest,
        decision_evidence=evidence,
    )
end

function _row_payload(row)
    return (;
        episode_id=row.episode_id,
        seed=row.seed,
        piece_index=row.piece_index,
        features=copy(row.features),
        advantage=row.advantage_unclipped,
        advantage_unclipped_A6=row.advantage_unclipped,
        clipped_target=row.target_clipped,
        g6_top1=row.g6_top1,
        g6_top2=row.g6_top2,
        a1_terminal_within_horizon=row.top1_branch.terminal,
        a2_terminal_within_horizon=row.top2_branch.terminal,
        root_state_digest=row.root_state_digest,
        root_future_stream_digest=row.root_future_stream_digest,
        canonical_top1_candidate_index=row.top1_index,
        canonical_top2_candidate_index=row.top2_index,
        canonical_top1_action_digest=row.top1_action_digest,
        canonical_top2_action_digest=row.top2_action_digest,
        q_top1=row.q_top1,
        q_top2=row.q_top2,
        q_gap=row.q_gap,
        valid_action_count=row.valid_action_count,
        top1_branch=_branch_payload(row.top1_branch),
        top2_branch=_branch_payload(row.top2_branch),
    )
end

mutable struct R1SyntheticState
    seed::Int
    piece_index::Int
    rng_position::Int
    score::Float64
    terminal::Bool
    future::Vector{Int}
    branch_root_pending::Bool
end

struct R1SyntheticAction
    id::Int
end

function _synthetic_adapter(R1)
    # This adapter exercises the exact production argv/serialization branch but
    # never imports a game, model, checkpoint, OpenVINO, or preregistered seed.
    synthetic_reward(state, action) = begin
        sample_ordinal = (state.piece_index + 1) ÷ 10
        favored = sample_ordinal > 0 && sample_ordinal % 4 == 0 ? 2 : 1
        action.id == favored ? 120.0 : 0.0
    end
    action_metrics(state, action) = (;
        immediate_score=synthetic_reward(state, action),
        cleared_lines=(synthetic_reward(state, action) > 0 ? 1.0 : 0.0),
        holes=Float64(action.id),
        covered_cells=2.0 * action.id,
        aggregate_height=3.0 * action.id,
        max_height=4.0 * action.id,
        bumpiness=5.0 * action.id,
        well_sum=6.0 * action.id,
        row_transitions=7.0 * action.id,
        column_transitions=8.0 * action.id,
        ren=9.0 * action.id,
        back_to_back=(action.id == 2 ? 1.0 : 0.0),
        tspin=(action.id == 2 ? 1.0 : 0.0),
    )
    state_digest(state) = bytes2hex(SHA.sha256(join((
        state.seed,
        state.piece_index,
        state.rng_position,
        state.score,
        state.terminal,
        state.branch_root_pending,
    ), '|')))
    return (;
        contract=R1.AdapterContract(
            candidate_order=:stable_node_key,
            q_chunk_size=16,
            q_tail_mode=:actual,
            next_count=5,
            hold_enabled=true,
        ),
        initial_state=seed -> R1SyntheticState(
            seed, 0, 0, 0.0, false, [mod(seed + index, 7) + 1 for index in 1:400], false
        ),
        clone_state=state -> begin
            branch = deepcopy(state)
            branch.branch_root_pending = true
            branch
        end,
        terminal=state -> state.terminal,
        score=state -> state.score,
        candidate_actions=state -> R1SyntheticAction[R1SyntheticAction(1), R1SyntheticAction(2), R1SyntheticAction(3)],
        q_values=(state, actions) -> Float64[action.id == 1 ? 10.0 : action.id == 2 ? 9.0 : 1.0 for action in actions],
        apply_action=(state, action) -> begin
            reward = state.branch_root_pending ? synthetic_reward(state, action) : 60.0
            state.piece_index += 1
            state.rng_position += 1
            state.score += reward
            state.branch_root_pending = false
            state
        end,
        state_digest,
        future_stream_digest=(state, horizon) -> bytes2hex(SHA.sha256(join(
            state.future[(state.piece_index + 1):(state.piece_index + horizon)], ','
        ))),
        piece_token=state -> string(state.future[state.piece_index + 1]),
        placed_piece_token=action -> "synthetic-placed-$(action.id)",
        rng_digest=state -> string(state.rng_position),
        action_digest=action -> "synthetic-action-$(action.id)",
        action_metrics,
        current_metrics=state -> (;
            holes=1.0,
            covered_cells=2.0,
            aggregate_height=3.0,
            max_height=4.0,
            bumpiness=5.0,
            well_sum=6.0,
            row_transitions=7.0,
            column_transitions=8.0,
        ),
        queue_onehot=state -> begin
            values = zeros(Float64, 42)
            for slot in 1:6
                values[(slot - 1) * 7 + state.future[state.piece_index + slot]] = 1.0
            end
            values
        end,
        stable_node_key_source_sha256="synthetic-no-engine",
    )
end

function _load_production_adapter(experiment_dir::AbstractString)
    adapter_path = joinpath(experiment_dir, "engine_adapter.jl")
    isfile(adapter_path) || error("missing production R1 engine adapter: $adapter_path")
    include(adapter_path)
    if isdefined(Main, :make_production_adapter)
        return Base.invokelatest(getfield(Main, :make_production_adapter))
    elseif isdefined(Main, :R1ProductionEngineAdapter) &&
           isdefined(
               getfield(Main, :R1ProductionEngineAdapter), :make_production_adapter,
           )
        return Base.invokelatest(getfield(
            getfield(Main, :R1ProductionEngineAdapter), :make_production_adapter,
        ))
    elseif isdefined(Main, :R1EngineAdapter) &&
           isdefined(getfield(Main, :R1EngineAdapter), :make_production_adapter)
        return Base.invokelatest(
            getfield(getfield(Main, :R1EngineAdapter), :make_production_adapter)
        )
    end
    error("engine_adapter.jl must define make_production_adapter()")
end

function main(args=ARGS)
    parsed = _parse_early_args(args)
    # The milestone path was proven fresh above; this first write is therefore
    # a create-new operation, while later records are durable append-only.
    _append_milestone(parsed.milestone_path, "script_enter")
    _append_milestone(
        parsed.milestone_path,
        "args_validated";
        detail="role=$(parsed.role);synthetic=$(parsed.synthetic)",
    )
    _append_milestone(parsed.milestone_path, "imports_begin")

    experiment_dir = @__DIR__
    include(joinpath(experiment_dir, "collector.jl"))
    include(joinpath(experiment_dir, "contract.jl"))
    @eval using JSON3
    @eval using SHA
    return Base.invokelatest(_collect_after_imports, parsed, experiment_dir)
end

function _collect_after_imports(parsed, experiment_dir)
    R1 = R1CounterfactualCollector
    contract = OnlineCounterfactualTop2R1Contract.load_contract()
    feature_names = collect(String.(R1.FEATURE_SCHEMA))
    feature_schema_digest = bytes2hex(SHA.sha256(join(feature_names, "\n")))
    contract_feature_names = String.(contract.feature_schema.names)
    feature_names == contract_feature_names || error(
        "collector feature order differs from frozen contract; collector=" *
        repr(feature_names) * "; contract=" * repr(contract_feature_names)
    )
    feature_schema_digest == String(contract.feature_schema.feature_names_sha256) || error(
        "collector feature digest differs from frozen contract"
    )
    _append_milestone(parsed.milestone_path, "imports_complete")

    adapter = parsed.synthetic ? _synthetic_adapter(R1) :
              _load_production_adapter(experiment_dir)
    R1.validate_adapter_contract(adapter.contract)
    _append_milestone(parsed.milestone_path, "collection_begin")

    role_seeds = if parsed.synthetic
        parsed.role === :train ? [101, 102] : [201, 202]
    elseif parsed.role === :train
        collect(R1.R1_TRAIN_SEEDS)
    else
        collect(R1.R1_CALIBRATION_SEEDS)
    end
    collections = R1.EpisodeCollection[]
    sample_count = Ref(0)
    first32_setup_seconds = Ref{Union{Nothing,Float64}}(nothing)
    first32_elapsed_seconds = Ref{Union{Nothing,Float64}}(nothing)
    first32_projected_seconds = Ref{Union{Nothing,Float64}}(nothing)
    collection_started = time()
    # The preregistered first-32 gate projects the complete online collection
    # workload: 12 training + 6 calibration episodes, each sampled 24 times.
    # It is first observed in the training phase and is not a per-role estimate.
    expected_sample_states = parsed.synthetic ?
        length(role_seeds) * length(R1.R1_SAMPLE_PIECES) : (12 + 6) * 24
    function on_sample_complete(event)
        sample_count[] += 1
        _append_milestone(
            parsed.milestone_path,
            "counterfactual_state_complete";
            detail="sample=$(sample_count[]);episode_id=$(event.episode_id);piece=$(event.piece_index);status=$(event.status)",
        )
        if parsed.role === :train && sample_count[] == 32
            elapsed = time() - collection_started
            setup_elapsed = collection_started - _R1_COLLECT_STARTED_AT
            # Training and calibration are separate collector processes, so
            # account for the observed one-time import/OpenVINO setup twice,
            # while extrapolating only steady collection across all 432 states.
            projected = 2 * setup_elapsed + elapsed / 32 * expected_sample_states
            first32_setup_seconds[] = setup_elapsed
            first32_elapsed_seconds[] = elapsed
            first32_projected_seconds[] = projected
            _append_milestone(
                parsed.milestone_path,
                "first32_projection";
                detail="formula=2*setup+first32_collection/32*basis;setup_seconds=$setup_elapsed;first32_collection_seconds=$elapsed;basis_states=$expected_sample_states;projected_seconds=$projected;limit_seconds=3300",
            )
            projected <= 3300.0 || error(
                "R1 first-32 projection $projected exceeds the frozen 3300 s limit"
            )
        end
        return nothing
    end
    for (episode_index, seed) in pairs(role_seeds)
        episode_id = parsed.synthetic ? episode_index : seed
        collection = if parsed.synthetic
            R1.collect_episode(adapter, seed; episode_id, on_sample_complete)
        else
            R1.collect_role_episode(
                adapter, seed, parsed.role; episode_id, on_sample_complete
            )
        end
        push!(collections, collection)
        _append_milestone(
            parsed.milestone_path,
            "episode_complete";
            detail="episode_id=$episode_id;rows=$(length(collection.rows));exclusions=$(length(collection.exclusions))",
        )
    end

    rows = reduce(vcat, (collection.rows for collection in collections); init=R1.CounterfactualRow[])
    exclusions = reduce(
        vcat,
        (collection.exclusions for collection in collections);
        init=R1.ExcludedState[],
    )
    all(length(row.features) == 70 for row in rows) || error("non-canonical feature width")
    positive_fraction = isempty(rows) ? NaN :
        count(row -> row.advantage_unclipped > 0.0, rows) / length(rows)
    _append_milestone(
        parsed.milestone_path,
        "collection_complete";
        detail="rows=$(length(rows));exclusions=$(length(exclusions))",
    )

    network_accounting = if parsed.synthetic || !hasproperty(adapter, :accounting)
        nothing
    else
        (;
            logical_network_calls=Int(adapter.accounting.logical_network_calls),
            physical_network_calls=Int(adapter.accounting.physical_network_calls),
            candidate_evaluations=Int(adapter.accounting.candidate_evaluations),
        )
    end
    backend_binding = parsed.synthetic ? nothing : (;
        old_openvino_weight_npz_sha256=String(adapter.old_openvino_weight_npz_sha256),
        old_checkpoint_sha256=String(adapter.old_checkpoint_sha256),
        openvino_version=String(adapter.openvino_version),
        complete_device=String(adapter.openvino_complete_device),
        tail_device=String(adapter.openvino_tail_device),
        complete_batch_size=Int(adapter.openvino_batch_size),
        evaluator_source_sha256=String(adapter.evaluator_source_sha256),
    )
    immutable_input_end_hashes = if parsed.synthetic
        nothing
    else
        checkpoint_end = bytes2hex(open(SHA.sha256, adapter.old_checkpoint_path))
        weights_end = bytes2hex(open(
            SHA.sha256, adapter.old_openvino_weight_npz_path,
        ))
        checkpoint_end == String(adapter.old_checkpoint_sha256) || error(
            "old checkpoint changed during R1 collection",
        )
        weights_end == String(adapter.old_openvino_weight_npz_sha256) || error(
            "old OpenVINO weights changed during R1 collection",
        )
        (;
            old_checkpoint_sha256=checkpoint_end,
            old_openvino_weight_npz_sha256=weights_end,
            unchanged=true,
        )
    end

    source_role = parsed.role === :train ? "training" : "calibration"
    table = (;
        schema_version=(parsed.role === :train ?
                        "r1-training-table-v1" : "r1-calibration-table-v1"),
        source_role,
        metadata=(;
            source_role,
            feature_names,
            feature_schema_digest,
            training_seeds=(parsed.role === :train && !parsed.synthetic ? role_seeds : Int[]),
            role_seeds,
            sample_pieces=collect(R1.R1_SAMPLE_PIECES),
            synthetic=parsed.synthetic,
            stable_node_key_source_sha256=String(adapter.stable_node_key_source_sha256),
            network_accounting,
            backend_binding,
            immutable_input_end_hashes,
            counterfactual_states_completed=sample_count[],
            first32_elapsed_seconds=first32_elapsed_seconds[],
            first32_setup_seconds=first32_setup_seconds[],
            first32_projected_seconds=first32_projected_seconds[],
            first32_projection_limit_seconds=3300.0,
            first32_projection_basis_states=expected_sample_states,
            first32_projection_formula="2*setup_seconds + first32_collection_seconds/32*$expected_sample_states",
            positive_advantage_fraction=positive_fraction,
            validation_seed_used=false,
            sealed_test_seed_used=false,
        ),
        feature_names,
        feature_schema_digest,
        training_seeds=(parsed.role === :train && !parsed.synthetic ? role_seeds : Int[]),
        synthetic=parsed.synthetic,
        validation_seed_used=false,
        sealed_test_seed_used=false,
        rows=[_row_payload(row) for row in rows],
    )
    _atomic_json_create(parsed.table_path, table, JSON3)
    table_sha256 = bytes2hex(open(SHA.sha256, parsed.table_path))
    manifest = (;
        schema_version="r1-collection-manifest-v1",
        source_role,
        synthetic=parsed.synthetic,
        table_path=parsed.table_path,
        table_sha256,
        feature_names,
        feature_schema_digest,
        stable_node_key_source_sha256=String(adapter.stable_node_key_source_sha256),
        network_accounting,
        backend_binding,
        immutable_input_end_hashes,
        episode_count=length(collections),
        row_count=length(rows),
        exclusion_count=length(exclusions),
        role_seeds,
        counterfactual_states_completed=sample_count[],
        first32_elapsed_seconds=first32_elapsed_seconds[],
        first32_setup_seconds=first32_setup_seconds[],
        first32_projected_seconds=first32_projected_seconds[],
        first32_projection_limit_seconds=3300.0,
        first32_projection_basis_states=expected_sample_states,
        first32_projection_formula="2*setup_seconds + first32_collection_seconds/32*$expected_sample_states",
        positive_advantage_fraction=positive_fraction,
        exclusions=[(;
            seed=item.seed,
            episode_id=item.episode_id,
            piece_index=item.piece_index,
            code=String(item.code),
            detail=item.detail,
            root_state_digest=item.root_state_digest,
        ) for item in exclusions],
        episodes=[(;
            seed=item.seed,
            episode_id=item.episode_id,
            canonical_pieces=item.canonical_pieces,
            canonical_score=item.canonical_score,
            canonical_terminal=item.canonical_terminal,
            canonical_action_digests=item.canonical_action_digests,
            rows=length(item.rows),
            exclusions=length(item.exclusions),
        ) for item in collections],
        real_model_or_game_loaded=!parsed.synthetic,
        validation_seed_used=false,
        sealed_test_seed_used=false,
        wall_seconds=time() - _R1_COLLECT_STARTED_AT,
    )
    _atomic_json_create(parsed.manifest_path, manifest, JSON3)
    _append_milestone(parsed.milestone_path, "artifacts_written")
    _append_milestone(parsed.milestone_path, "script_complete")
    return (; table=parsed.table_path, manifest=parsed.manifest_path)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
