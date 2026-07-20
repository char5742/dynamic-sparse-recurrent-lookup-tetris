using Dates
using JSON3
using Random
using Serialization
using SHA

include(joinpath(@__DIR__, "teacher_training.jl"))

const T = BeatFirstThreeLayerTeacherTraining
const C = T.BeatFirstTrainingCore
const S = Main.SparseDynamic3Layer
const O = S.MongooseSimHashOverlay

const DATASET_DEFAULT = raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const OUTPUT_PARENT_DEFAULT = raw"C:\tmp"
const ACTIVE_COUNTS = (48, 40, 40)
const TRAINING_PROBES = (6, 5, 5)
const LEARNER_WIDTH = 80
const MODEL_SEED = UInt64(0x4d56325245414c31)
const SAMPLER_SEED = UInt64(0x4d563253414d5031)
const PANEL_SEED = UInt64(0x4d563250414e454c)
const ROUTER_SEED = 0x4d4f4e47
const RESERVED_GAME_SEEDS = Set(vcat(collect(8001:8008), collect(91001:91032)))

@inline _sha256_file(path::AbstractString) = bytes2hex(open(sha256, path))

function _parse_arguments(arguments::Vector{String})
    dataset_root = DATASET_DEFAULT
    output_root = joinpath(
        OUTPUT_PARENT_DEFAULT,
        "tetris_mongoose_v2_real_teacher_smoke_" *
        Dates.format(now(), dateformat"yyyymmddTHHMMSS"),
    )
    pair_mining_mode = T.MONGOOSE_EASY_EXTREMA_PAIR_MINING_MODE
    seen = Set{String}()
    for argument in arguments
        startswith(argument, "--dataset-root=") && begin
            "dataset-root" in seen && error("--dataset-root was supplied twice")
            dataset_root = split(argument, '='; limit=2)[2]
            isempty(dataset_root) && error("--dataset-root is empty")
            push!(seen, "dataset-root")
            continue
        end
        startswith(argument, "--output-root=") && begin
            "output-root" in seen && error("--output-root was supplied twice")
            output_root = split(argument, '='; limit=2)[2]
            isempty(output_root) && error("--output-root is empty")
            push!(seen, "output-root")
            continue
        end
        startswith(argument, "--pair-mining-mode=") && begin
            "pair-mining-mode" in seen && error(
                "--pair-mining-mode was supplied twice",
            )
            value = split(argument, '='; limit=2)[2]
            pair_mining_mode = value == "cutoff" ?
                T.MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_MODE :
                T._normalize_mongoose_pair_mining_mode(value)
            push!(seen, "pair-mining-mode")
            continue
        end
        error("unsupported argument: $argument")
    end
    return (;
        dataset_root=abspath(dataset_root),
        output_root=abspath(output_root),
        pair_mining_mode,
    )
end

function _write_json(path::AbstractString, value)
    temporary = path * ".tmp"
    isfile(temporary) && error("temporary JSON path already exists: $temporary")
    open(temporary, "w") do io
        JSON3.pretty(io, value)
        write(io, '\n')
        flush(io)
    end
    mv(temporary, path; force=false)
    return path
end

function _serialized_sha256(value, parent::AbstractString)
    mktemp(parent) do path, io
        serialize(io, value)
        close(io)
        return _sha256_file(path)
    end
end

function _without_fields(value::NamedTuple, removed::Tuple)
    names = Tuple(name for name in propertynames(value) if !(name in removed))
    return NamedTuple{names}(Tuple(getproperty(value, name) for name in names))
end

function _deterministic_accounting(accounting::NamedTuple)
    pair = accounting.mongoose_pair
    deterministic_pair = pair === nothing ? nothing :
        _without_fields(pair, (:mining_nanoseconds, :pair_nanoseconds))
    return merge(
        accounting,
        (;
            routing_projection_nanoseconds=(0, 0, 0),
            mongoose_pair=deterministic_pair,
        ),
    )
end

function _assert_finite_tree(value, label::AbstractString)
    if value isa AbstractFloat
        isfinite(value) || error("$label contains a non-finite value")
    elseif value isa NamedTuple
        for name in propertynames(value)
            _assert_finite_tree(getproperty(value, name), "$label.$name")
        end
    elseif value isa Tuple || value isa AbstractArray
        for (index, item) in pairs(value)
            _assert_finite_tree(item, "$label[$index]")
        end
    end
    return true
end

function _inactive_witness_snapshot(runtime)
    return ntuple(3) do layer_id
        theta = runtime.model.layers[layer_id].theta
        optimizer = runtime.bank_optimizers[layer_id]
        neurons = size(theta, 2)
        ids = unique!(round.(Int, range(1, neurons; length=min(256, neurons))))
        records = Dict{Int,Any}()
        for id in ids
            records[id] = (;
                theta=copy(@view theta[:, id]),
                m=copy(@view optimizer.m[:, id]),
                v=copy(@view optimizer.v[:, id]),
                event_count=optimizer.event_count[id],
                last_event_step=optimizer.last_event_step[id],
                last_log_decay=optimizer.last_log_decay[id],
            )
        end
        records
    end
end

@inline _f32_bits(values) = collect(reinterpret(UInt32, vec(values)))

function _verify_inactive_witnesses!(trainer, snapshots)
    return ntuple(3) do layer_id
        accumulator = trainer.workspace.accumulators[layer_id]
        active = Set(Int.(accumulator.ids[1:accumulator.used]))
        theta = trainer.runtime.model.layers[layer_id].theta
        optimizer = trainer.runtime.bank_optimizers[layer_id]
        verified = 0
        for (id, before) in snapshots[layer_id]
            id in active && continue
            _f32_bits(@view(theta[:, id])) == _f32_bits(before.theta) || error(
                "inactive theta bytes changed in layer $layer_id row $id",
            )
            _f32_bits(@view(optimizer.m[:, id])) == _f32_bits(before.m) || error(
                "inactive first-moment bytes changed in layer $layer_id row $id",
            )
            _f32_bits(@view(optimizer.v[:, id])) == _f32_bits(before.v) || error(
                "inactive second-moment bytes changed in layer $layer_id row $id",
            )
            optimizer.event_count[id] == before.event_count || error(
                "inactive event counter changed in layer $layer_id row $id",
            )
            optimizer.last_event_step[id] == before.last_event_step || error(
                "inactive last-event clock changed in layer $layer_id row $id",
            )
            reinterpret(UInt64, optimizer.last_log_decay[id]) ==
                reinterpret(UInt64, before.last_log_decay) || error(
                    "inactive lazy-decay clock changed in layer $layer_id row $id",
                )
            verified += 1
        end
        verified >= 16 || error(
            "layer $layer_id retained only $verified inactive byte witnesses",
        )
        verified
    end
end

function _assert_v2_cap_closure(accounting)
    accounting.routing_mode === T.MONGOOSE_SIMHASH_V2_ROUTING_MODE || error(
        "training step did not use the MONGOOSE v2 routing mode",
    )
    count = accounting.valid_candidates
    count >= 1 || error("MONGOOSE v2 step has no valid candidate")
    for layer_id in 1:3
        available = accounting.mongoose_v2_bucket_entries_available[layer_id]
        visited = accounting.bucket_entries[layer_id]
        truncated = accounting.mongoose_v2_truncated_bucket_entries[layer_id]
        available == visited + truncated || error(
            "layer $layer_id bucket available/visited/truncated closure failed",
        )
        sum(accounting.mongoose_v2_table_entries_available[layer_id]) == available ||
            error("layer $layer_id table-available closure failed")
        sum(accounting.mongoose_v2_table_entries_visited[layer_id]) == visited ||
            error("layer $layer_id table-visited closure failed")
        sum(accounting.mongoose_v2_lane_entries_available[layer_id]) == available ||
            error("layer $layer_id lane-available closure failed")
        sum(accounting.mongoose_v2_lane_entries_visited[layer_id]) == visited ||
            error("layer $layer_id lane-visited closure failed")
        visited <= count * S.LAYER_MAX_BUCKET_ENTRIES[layer_id] || error(
            "layer $layer_id exceeded its aggregate bucket-entry cap",
        )
        accounting.scored_rows[layer_id] <=
            count * S.LAYER_MAX_SCORED_ROWS[layer_id] || error(
                "layer $layer_id exceeded its aggregate exact-score cap",
            )
        lane_visited = accounting.mongoose_v2_lane_entries_visited[layer_id]
        lane_available = accounting.mongoose_v2_lane_entries_available[layer_id]
        all(
            lane_visited[lane] <= lane_available[lane]
            for lane in eachindex(lane_visited)
        ) || error("layer $layer_id visited more entries than a lane contains")
        accounting.mongoose_v2_fill_probe_attempts[layer_id] <=
            count * ACTIVE_COUNTS[layer_id] || error(
                "layer $layer_id exceeded its bounded fill-probe budget",
            )
        accounting.mongoose_v2_training_probe_attempts[layer_id] <=
            count * ACTIVE_COUNTS[layer_id] || error(
                "layer $layer_id exceeded its bounded training-probe budget",
            )
        accounting.mongoose_v2_overloaded_routes[layer_id] <= count || error(
            "layer $layer_id overloaded-route count exceeds candidates",
        )
    end
    sum(accounting.routing_projection_macs) > 0 || error(
        "MONGOOSE v2 update executed no learned routing projections",
    )
    accounting.active_counts == ACTIVE_COUNTS || error(
        "MONGOOSE v2 changed fixed active widths",
    )
    accounting.training_probes == TRAINING_PROBES || error(
        "MONGOOSE v2 changed fixed training probes",
    )
    return true
end

function _assert_pair_mining_telemetry(accounting, pair_mining_mode::Symbol)
    pair = accounting.mongoose_pair
    pair === nothing && error("real-teacher smoke published no router pair telemetry")
    pair.pair_mining_mode === pair_mining_mode || error(
        "real-teacher smoke pair-mining identity changed",
    )
    pair_mining_mode === T.MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_MODE ||
        return true
    expected_targets = ntuple(
        i -> ACTIVE_COUNTS[i] - TRAINING_PROBES[i],
        3,
    )
    pair.exploitation_targets == expected_targets || error(
        "cutoff exploitation targets changed",
    )
    for layer_id in 1:3
        pair.positive_ranks[layer_id] == expected_targets[layer_id] || error(
            "cutoff positive rank changed at layer $layer_id",
        )
        pair.negative_ranks[layer_id] == expected_targets[layer_id] + 1 || error(
            "cutoff negative rank changed at layer $layer_id",
        )
        pair.positive_in_exploitation[layer_id] || error(
            "cutoff positive escaped exploitation at layer $layer_id",
        )
        !pair.negative_in_exploitation[layer_id] || error(
            "cutoff negative entered exploitation at layer $layer_id",
        )
        pair.positive_collisions[layer_id] >= 1 || error(
            "cutoff positive is not a natural positive-collision row at layer $layer_id",
        )
        pair.negative_collisions[layer_id] >= 1 || error(
            "cutoff negative is not a natural positive-collision row at layer $layer_id",
        )
        pair.eligible_witness_rows[layer_id] >= pair.negative_ranks[layer_id] ||
            error("cutoff bounded witness underflow at layer $layer_id")
        gap = pair.positive_scores[layer_id] - pair.negative_scores[layer_id]
        isfinite(gap) && gap >= 0.0 || error(
            "cutoff boundary score order failed at layer $layer_id",
        )
        isequal(gap, pair.cutoff_score_gaps[layer_id]) || error(
            "cutoff boundary score gap changed at layer $layer_id",
        )
        if isequal(pair.positive_scores[layer_id], pair.negative_scores[layer_id])
            pair.positive_ids[layer_id] < pair.negative_ids[layer_id] || error(
                "cutoff equal-score stable-ID order changed at layer $layer_id",
            )
        end
    end
    return true
end

function _checkpoint_config(
    dataset_root::AbstractString,
    state,
    pair_mining_mode::Symbol,
)
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    manifest = abspath(joinpath(@__DIR__, "..", "Manifest.toml"))
    base = (;
        bounded_test_contract=true,
        variant=:bounded_test,
        active_counts=ACTIVE_COUNTS,
        training_probes=TRAINING_PROBES,
        learner_width=LEARNER_WIDTH,
        routing_mode=T.MONGOOSE_SIMHASH_V2_ROUTING_MODE,
        routing_policy=T.MONGOOSE_V2_ROUTING_POLICY,
        mongoose_route_dim=O.ROUTE_DIM,
        mongoose_bits_per_table=O.BITS_PER_TABLE,
        mongoose_tables=O.TABLES,
        mongoose_router_parameters=O.ROUTER_PARAMETERS,
        mongoose_column_normalization=O.COLUMN_NORMALIZATION,
        mongoose_learning_rate=Float64(state.optimizers[1].learning_rate),
        mongoose_beta=Float64(state.beta),
        mongoose_seed=Int(state.seed),
        mongoose_warmup_updates=1,
        mongoose_refresh_interval=1,
        objective_margin_weight=Float64(C.MARGIN_WEIGHT),
        objective_margin_mode=C.FIXED_TEACHER_TOP2_MARGIN_MODE,
        objective_mode=C.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
        model_seed=Int(MODEL_SEED),
        sampler_seed=Int(SAMPLER_SEED),
        source_sha256=T._source_sha256(),
        environment_project_sha256=_sha256_file(project),
        environment_manifest_sha256=_sha256_file(manifest),
        dataset_manifest_sha256=T._dataset_manifest_sha256(dataset_root),
    )
    config = merge(base, T._mongoose_v2_policy_config())
    return pair_mining_mode === T.MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_MODE ?
        merge(config, T._mongoose_v2_cutoff_pair_policy_config()) : config
end

function _construct_trainer(pair_mining_mode::Symbol)
    model = S.initialize_model(
        Xoshiro(MODEL_SEED);
        neuron_counts=S.LAYER_NEURON_COUNTS,
        active_counts=ACTIVE_COUNTS,
    )
    runtime = S.initialize_runtime(
        model;
        learning_rate=1.0f-4,
        weight_decay=1.0f-4,
        beta1=0.9f0,
        beta2=0.999f0,
        epsilon=1.0f-8,
    )
    S.parameter_count(runtime.model) == S.TOTAL_PARAMETERS || error(
        "full-shape model parameter count changed",
    )
    positions = ntuple(i -> runtime.indexes[i].positions, 3)
    state = O.initialize_v2_overlay(
        positions;
        learning_rate=1.0f-4,
        beta1=0.9f0,
        beta2=0.999f0,
        epsilon=1.0f-8,
        beta=1.0f0,
        warmup_updates=1,
        refresh_interval=1,
        seed=ROUTER_SEED,
    )
    trainer = T._trainer_from_runtime(
        runtime;
        variant=:bounded_test,
        training_probes=TRAINING_PROBES,
        candidate_width=LEARNER_WIDTH,
        objective_margin_weight=C.MARGIN_WEIGHT,
        objective_margin_mode=C.FIXED_TEACHER_TOP2_MARGIN_MODE,
        objective_mode=C.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
        mongoose_state=state,
        routing_mode=T.MONGOOSE_SIMHASH_V2_ROUTING_MODE,
        mongoose_pair_mining_mode=pair_mining_mode,
    )
    trainer.variant === :bounded_test || error("test-only variant was not retained")
    trainer.routing_mode === T.MONGOOSE_SIMHASH_V2_ROUTING_MODE || error(
        "test-only trainer lost v2 routing identity",
    )
    trainer.mongoose_pair_mining_mode === pair_mining_mode || error(
        "test-only trainer lost pair-mining identity",
    )
    size(trainer.workspace.raw, 2) == LEARNER_WIDTH || error(
        "learner width differs from 80",
    )
    return trainer
end

function _advance_one!(trainer, sampler, dataset, host_batch, update::Int)
    row = only(C.next_batch!(sampler, 1))
    dataset.seed_ids[row] in RESERVED_GAME_SEEDS && error(
        "smoke selected reserved game seed $(dataset.seed_ids[row])",
    )
    C.pack_batch!(host_batch, dataset, [row])
    inactive_before = _inactive_witness_snapshot(trainer.runtime)
    result = T.teacher_train_step!(
        trainer,
        host_batch;
        row_id=row,
        training_step=update,
    )
    inactive_verified = _verify_inactive_witnesses!(trainer, inactive_before)
    refresh = T.maybe_refresh_mongoose!(trainer, update)
    refresh === nothing && error("warmup=1/refresh=1 did not refresh at update $update")
    _assert_finite_tree(result.components, "update-$update components")
    return (; row, result, refresh, inactive_verified)
end

function smoke_main(arguments::Vector{String}=ARGS)
    options = _parse_arguments(arguments)
    expected_project = realpath(joinpath(@__DIR__, "..", "Project.toml"))
    active_project = Base.active_project()
    active_project === nothing && error("Julia has no active project")
    realpath(active_project) == expected_project || error(
        "active Julia project differs from $expected_project",
    )
    isdir(options.dataset_root) || error(
        "teacher_v3 dataset root does not exist: $(options.dataset_root)",
    )
    isfile(joinpath(options.dataset_root, "manifest.json")) || error(
        "teacher_v3 manifest is missing",
    )
    ispath(options.output_root) && error(
        "fresh output root already exists: $(options.output_root)",
    )
    mkpath(options.output_root)

    dataset = C.load_teacher_dataset(
        options.dataset_root;
        max_candidates=C.MAX_CANDIDATES,
        allow_partial_dataset=false,
    )
    observed_max_candidates = maximum(dataset.action_counts)
    observed_max_candidates <= LEARNER_WIDTH || error(
        "teacher_v3 exceeds learner width $LEARNER_WIDTH",
    )
    split = T.episode_separated_split(dataset; seed=PANEL_SEED)
    isempty(split.training_rows) && error("teacher_v3 training split is empty")
    panel_rows = T.fixed_evaluation_subset(split.training_rows, 2, PANEL_SEED)
    all(row -> dataset.predefined_split[row] === :train, panel_rows) || error(
        "teacher panel escaped the predefined training split",
    )
    all(row -> !(dataset.seed_ids[row] in RESERVED_GAME_SEEDS), panel_rows) || error(
        "teacher panel includes a reserved game seed",
    )

    trainer = _construct_trainer(options.pair_mining_mode)
    state = trainer.mongoose_state
    state isa O.MongooseV2OverlayState || error("trainer has no v2 overlay state")
    !state.active || error("v2 overlay activated before its warmup update")
    sampler = C.EpochSampler(split.training_rows, Xoshiro(SAMPLER_SEED))
    host_batch = C.allocate_host_batch(1; max_candidates=LEARNER_WIDTH)

    first = _advance_one!(trainer, sampler, dataset, host_batch, 1)
    state.active || error("update 1 failed to activate MONGOOSE v2")
    state.last_refresh_update == 1 || error("update 1 refresh was not published")
    state.refresh_count == 1 || error("update 1 refresh count changed")

    second = _advance_one!(trainer, sampler, dataset, host_batch, 2)
    _assert_v2_cap_closure(second.result.accounting)
    _assert_pair_mining_telemetry(first.result.accounting, options.pair_mining_mode)
    _assert_pair_mining_telemetry(second.result.accounting, options.pair_mining_mode)
    state.active || error("MONGOOSE v2 became inactive at update 2")
    state.last_refresh_update == 2 || error("update 2 refresh was not published")
    state.refresh_count == 2 || error("update 2 refresh count changed")
    O.validate_v2_overlay!(
        state,
        ntuple(i -> trainer.runtime.model.layers[i].theta, 3),
        2,
    )

    panel_metrics = T.three_layer_evaluation_metrics(
        trainer,
        dataset,
        panel_rows,
        host_batch,
    )
    _assert_finite_tree(panel_metrics, "teacher panel")

    config = _checkpoint_config(
        options.dataset_root,
        state,
        options.pair_mining_mode,
    )
    split_metadata = T._split_checkpoint_metadata(split, panel_rows, Int[])
    history = Any[(;
        update=2,
        rows=copy(panel_rows),
        seed_ids=Int.(dataset.seed_ids[panel_rows]),
        metrics=panel_metrics,
    )]
    checkpoint_path = joinpath(options.output_root, "checkpoint_000000002.jls")
    checkpoint_staging = checkpoint_path * ".staging"
    artifact = T.save_three_layer_teacher_checkpoint(
        checkpoint_staging,
        trainer,
        sampler,
        config,
        split_metadata,
        history,
        2;
        _bounded_test_contract=true,
    )
    isfile(checkpoint_staging) || error("staged checkpoint was not written")
    artifact.sha256 == _sha256_file(checkpoint_staging) || error(
        "staged checkpoint hash changed after publication",
    )

    restored = T.restore_three_layer_teacher_checkpoint(
        checkpoint_staging,
        config,
        split,
        panel_rows,
        Int[];
        _bounded_test_contract=true,
    )
    restored.update == 2 || error("restored update differs from 2")
    restored.trainer.variant === :bounded_test || error(
        "restored trainer lost its test-only variant",
    )
    restored.trainer.routing_mode === T.MONGOOSE_SIMHASH_V2_ROUTING_MODE || error(
        "restored trainer lost MONGOOSE v2 routing identity",
    )
    restored.trainer.mongoose_pair_mining_mode === options.pair_mining_mode ||
        error("restored trainer lost pair-mining identity")
    restored.trainer.mongoose_state isa O.MongooseV2OverlayState || error(
        "restored trainer contains the wrong overlay type",
    )
    ntuple(i -> restored.trainer.runtime.model.layers[i].active_count, 3) ==
        ACTIVE_COUNTS || error("restored active widths changed")
    size(restored.trainer.workspace.raw, 2) == LEARNER_WIDTH || error(
        "restored learner width changed",
    )
    O.validate_v2_overlay!(
        restored.trainer.mongoose_state,
        ntuple(i -> restored.trainer.runtime.model.layers[i].theta, 3),
        2,
    )

    runtime_sha_before = _serialized_sha256(trainer.runtime, options.output_root)
    restored_runtime_sha_before = _serialized_sha256(
        restored.trainer.runtime,
        options.output_root,
    )
    runtime_sha_before == restored_runtime_sha_before || error(
        "checkpoint restore changed runtime bytes",
    )
    overlay_sha_before = _serialized_sha256(state, options.output_root)
    restored_overlay_sha_before = _serialized_sha256(
        restored.trainer.mongoose_state,
        options.output_root,
    )
    overlay_sha_before == restored_overlay_sha_before || error(
        "checkpoint restore changed MONGOOSE v2 state/index bytes",
    )
    sampler_sha_before = _serialized_sha256(C.sampler_snapshot(sampler), options.output_root)
    restored_sampler_sha_before = _serialized_sha256(
        C.sampler_snapshot(restored.sampler),
        options.output_root,
    )
    sampler_sha_before == restored_sampler_sha_before || error(
        "checkpoint restore changed sampler/Xoshiro bytes",
    )

    original_row = only(C.next_batch!(sampler, 1))
    restored_row = only(C.next_batch!(restored.sampler, 1))
    original_row == restored_row || error("restored sampler chose a different next row")
    dataset.seed_ids[original_row] in RESERVED_GAME_SEEDS && error(
        "continuation selected a reserved game seed",
    )
    C.pack_batch!(host_batch, dataset, [original_row])
    original_step = T.teacher_train_step!(
        trainer,
        host_batch;
        row_id=original_row,
        training_step=3,
    )
    original_refresh = T.maybe_refresh_mongoose!(trainer, 3)
    C.pack_batch!(host_batch, dataset, [restored_row])
    replay_step = T.teacher_train_step!(
        restored.trainer,
        host_batch;
        row_id=restored_row,
        training_step=3,
    )
    replay_refresh = T.maybe_refresh_mongoose!(restored.trainer, 3)
    original_refresh === nothing && error("original continuation did not refresh")
    replay_refresh === nothing && error("restored continuation did not refresh")
    isequal(original_step.loss, replay_step.loss) || error(
        "exact continuation loss differs",
    )
    original_step.components == replay_step.components || error(
        "exact continuation loss components differ",
    )
    _deterministic_accounting(original_step.accounting) ==
        _deterministic_accounting(replay_step.accounting) || error(
            "exact continuation deterministic accounting differs",
        )
    _assert_v2_cap_closure(original_step.accounting)
    _assert_v2_cap_closure(replay_step.accounting)
    _assert_pair_mining_telemetry(original_step.accounting, options.pair_mining_mode)
    _assert_pair_mining_telemetry(replay_step.accounting, options.pair_mining_mode)

    runtime_sha_after = _serialized_sha256(trainer.runtime, options.output_root)
    restored_runtime_sha_after = _serialized_sha256(
        restored.trainer.runtime,
        options.output_root,
    )
    runtime_sha_after == restored_runtime_sha_after || error(
        "exact next-step runtime bytes differ",
    )
    overlay_sha_after = _serialized_sha256(state, options.output_root)
    restored_overlay_sha_after = _serialized_sha256(
        restored.trainer.mongoose_state,
        options.output_root,
    )
    overlay_sha_after == restored_overlay_sha_after || error(
        "exact next-step MONGOOSE v2 state/index bytes differ",
    )
    sampler_sha_after = _serialized_sha256(C.sampler_snapshot(sampler), options.output_root)
    restored_sampler_sha_after = _serialized_sha256(
        C.sampler_snapshot(restored.sampler),
        options.output_root,
    )
    sampler_sha_after == restored_sampler_sha_after || error(
        "exact next-step sampler/Xoshiro bytes differ",
    )
    O.validate_v2_overlay!(
        state,
        ntuple(i -> trainer.runtime.model.layers[i].theta, 3),
        3,
    )

    mv(checkpoint_staging, checkpoint_path; force=false)
    isfile(checkpoint_path) || error("verified checkpoint was not published")
    !isfile(checkpoint_staging) || error("staged checkpoint remains after publish")
    artifact.sha256 == _sha256_file(checkpoint_path) || error(
        "verified checkpoint hash changed during publication",
    )
    O.validate_v2_overlay!(
        restored.trainer.mongoose_state,
        ntuple(i -> restored.trainer.runtime.model.layers[i].theta, 3),
        3,
    )

    driver_path = abspath(@__FILE__)
    summary = (;
        schema=options.pair_mining_mode ===
            T.MONGOOSE_V2_CUTOFF_BOUNDARY_PAIR_MINING_MODE ?
            "mongoose-v2-real-teacher-two-update-smoke-cutoff-v1" :
            "mongoose-v2-real-teacher-two-update-smoke-v1",
        status="PASS",
        bounded_test_contract=true,
        dataset_root=options.dataset_root,
        dataset_manifest_sha256=config.dataset_manifest_sha256,
        output_root=options.output_root,
        julia_version=string(VERSION),
        driver_path,
        driver_sha256=_sha256_file(driver_path),
        source_sha256=config.source_sha256,
        project_sha256=config.environment_project_sha256,
        manifest_sha256=config.environment_manifest_sha256,
        total_parameters=S.parameter_count(trainer.runtime.model),
        active_counts=ACTIVE_COUNTS,
        training_probes=TRAINING_PROBES,
        learner_width=LEARNER_WIDTH,
        observed_max_candidates,
        objective_mode=String(trainer.objective_mode),
        objective_margin_mode=String(trainer.objective_margin_mode),
        objective_margin_weight=Float64(trainer.objective_margin_weight),
        routing_mode=String(trainer.routing_mode),
        pair_mining_mode=String(trainer.mongoose_pair_mining_mode),
        overlay_type=string(typeof(state)),
        warmup_updates=state.warmup_updates,
        refresh_interval=state.refresh_interval,
        update_rows=(first.row, second.row),
        update_seed_ids=(dataset.seed_ids[first.row], dataset.seed_ids[second.row]),
        update2_loss=second.result.loss,
        update2_pair_telemetry=_deterministic_accounting(
            second.result.accounting,
        ).mongoose_pair,
        update2_cap_telemetry=(;
            valid_candidates=second.result.accounting.valid_candidates,
            bucket_entries_available=
                second.result.accounting.mongoose_v2_bucket_entries_available,
            bucket_entries_visited=second.result.accounting.bucket_entries,
            truncated_bucket_entries=
                second.result.accounting.mongoose_v2_truncated_bucket_entries,
            scored_rows=second.result.accounting.scored_rows,
            overloaded_routes=
                second.result.accounting.mongoose_v2_overloaded_routes,
        ),
        inactive_byte_witnesses=(;
            update1=first.inactive_verified,
            update2=second.inactive_verified,
        ),
        teacher_panel=(;
            rows=panel_rows,
            seed_ids=Int.(dataset.seed_ids[panel_rows]),
            metrics=panel_metrics,
        ),
        checkpoint=(;
            path=abspath(checkpoint_path),
            bytes=artifact.bytes,
            sha256=artifact.sha256,
            update=artifact.update,
        ),
        exact_continuation=(;
            next_row=original_row,
            next_seed_id=dataset.seed_ids[original_row],
            loss=original_step.loss,
            runtime_sha256_before=runtime_sha_before,
            overlay_sha256_before=overlay_sha_before,
            sampler_sha256_before=sampler_sha_before,
            runtime_sha256_after=runtime_sha_after,
            overlay_sha256_after=overlay_sha_after,
            sampler_sha256_after=sampler_sha_after,
        ),
    )
    summary_path = joinpath(options.output_root, "summary.json")
    _write_json(summary_path, summary)
    published = (;
        status="PASS",
        summary_path=abspath(summary_path),
        summary_sha256=_sha256_file(summary_path),
        checkpoint_path=abspath(checkpoint_path),
        checkpoint_sha256=artifact.sha256,
    )
    println("MONGOOSE_V2_REAL_SMOKE_SUMMARY ", JSON3.write(published))
    return published
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        smoke_main(ARGS)
    catch exception
        failure = (;
            status="FAIL",
            exception_type=string(typeof(exception)),
            message=sprint(showerror, exception),
        )
        println(stderr, "MONGOOSE_V2_REAL_SMOKE_FAILURE ", JSON3.write(failure))
        showerror(stderr, exception, catch_backtrace())
        println(stderr)
        exit(1)
    end
end
