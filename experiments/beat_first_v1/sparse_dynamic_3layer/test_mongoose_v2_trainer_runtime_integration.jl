using Test
using Random
using Serialization

include(joinpath(@__DIR__, "teacher_training.jl"))

const MV2T = BeatFirstThreeLayerTeacherTraining
const MV2S = Main.SparseDynamic3Layer
const MV2O = MV2S.MongooseSimHashOverlay

function _v2_runtime(; neurons=(640, 640, 640), active=(48, 40, 40))
    model = MV2S.initialize_model(
        Xoshiro(0x4d4f4e475632494e);
        neuron_counts=neurons,
        active_counts=active,
    )
    return MV2S.initialize_runtime(model; weight_decay=0.0f0)
end

function _empty_accumulators(runtime)
    return ntuple(3) do layer_id
        layer = runtime.model.layers[layer_id]
        accumulator = MV2S.EventTimeGradientAccumulator(
            size(layer.theta, 1),
            size(layer.theta, 2),
        )
        MV2S.begin_accumulation!(accumulator)
        accumulator
    end
end

function _serialized_v2(value)
    io = IOBuffer()
    serialize(io, value)
    return take!(io)
end

function _v2_checkpoint_config(state)
    digest = repeat("0", 64)
    base = (;
        variant=:k128,
        active_counts=(48, 40, 40),
        training_probes=(6, 5, 5),
        routing_mode=MV2T.MONGOOSE_SIMHASH_V2_ROUTING_MODE,
        routing_policy=MV2T.MONGOOSE_V2_ROUTING_POLICY,
        mongoose_route_dim=MV2O.ROUTE_DIM,
        mongoose_bits_per_table=MV2O.BITS_PER_TABLE,
        mongoose_tables=MV2O.TABLES,
        mongoose_router_parameters=MV2O.ROUTER_PARAMETERS,
        mongoose_column_normalization=MV2O.COLUMN_NORMALIZATION,
        mongoose_learning_rate=Float64(state.optimizers[1].learning_rate),
        mongoose_beta=Float64(state.beta),
        mongoose_seed=Int(state.seed),
        mongoose_warmup_updates=2_000,
        mongoose_refresh_interval=2_000,
        objective_margin_weight=Float64(MV2T.BeatFirstTrainingCore.MARGIN_WEIGHT),
        objective_margin_mode=MV2T.BeatFirstTrainingCore.FIXED_TEACHER_TOP2_MARGIN_MODE,
        objective_mode=
            MV2T.BeatFirstTrainingCore.STANDARDIZED_LISTNET_MARGIN_OBJECTIVE_MODE,
        source_sha256=digest,
        environment_project_sha256=digest,
        environment_manifest_sha256=digest,
        dataset_manifest_sha256=digest,
    )
    return merge(base, MV2T._mongoose_v2_policy_config())
end

function _bounded_v2_checkpoint_config(state; learner_width::Int=1)
    return merge(_v2_checkpoint_config(state), (;
        variant=:bounded_test,
        bounded_test_contract=true,
        learner_width,
        mongoose_warmup_updates=state.warmup_updates,
        mongoose_refresh_interval=state.refresh_interval,
    ))
end

@testset "base WTA transaction journal is bounded by dirty rows" begin
    neurons = (512, 64, 64)
    model = MV2S.initialize_model(
        Xoshiro(0x5754414a4f55524e);
        neuron_counts=neurons,
        active_counts=(48, 40, 40),
    )
    for layer in model.layers
        layer.theta[1:MV2S.ROUTE_DIM, :] .= 0.0f0
    end
    runtime = MV2S.initialize_runtime(model; weight_decay=0.0f0)
    layer = runtime.model.layers[1]
    index = runtime.indexes[1]
    ids = Int32[neurons[1], neurons[1] - 1]
    proposed = copy(layer.theta[:, Int.(ids)])
    sampled_second = Int(index.positions[2])
    proposed[sampled_second, :] .= 1.0f0
    prepared = MV2S.PreparedSparseAdamWStep(
        UInt64(1),
        0.0,
        ids,
        proposed,
        zeros(Float32, size(proposed)),
        zeros(Float32, size(proposed)),
        fill(UInt64(1), length(ids)),
        trues(length(ids)),
    )
    code_changes = false
    for position in eachindex(ids), table in 1:index.config.L
        slot = MV2S.WTALSHIndex._slot(index, ids[position], table)
        code_changes |= MV2S._prepared_wta_code(
            index,
            prepared,
            position,
            table,
        ) != Int(index.codes[slot])
    end
    @test code_changes
    before_index = _serialized_v2(index)
    before_theta = copy(layer.theta[:, Int.(ids)])
    snapshot = MV2S._snapshot_index_transaction(index, layer, prepared)
    @test length(snapshot.head_slots) <=
        2 * index.config.L * length(ids)
    @test length(snapshot.link_slots) <=
        4 * index.config.L * length(ids)
    layer.theta[:, Int.(ids)] .= proposed
    @test MV2S.WTALSHIndex.rehash!(index, layer.theta, ids) >= 1
    MV2S._restore_index_transaction!(index, snapshot)
    layer.theta[:, Int.(ids)] .= before_theta
    @test _serialized_v2(index) == before_index
end

@testset "MONGOOSE v2 trainer/runtime identity and bounded telemetry" begin
    @test MV2T.MONGOOSE_SIMHASH_ROUTING_MODE === :mongoose_simhash_k7_l2_v1
    @test MV2T.MONGOOSE_SIMHASH_V2_ROUTING_MODE ===
        :mongoose_simhash_k7_l2_bounded_lanes_fixed_k128_v2
    @test MV2T._is_mongoose_mode(MV2T.MONGOOSE_SIMHASH_ROUTING_MODE)
    @test MV2T._is_mongoose_mode(MV2T.MONGOOSE_SIMHASH_V2_ROUTING_MODE)
    @test !MV2T._is_mongoose_mode(MV2T.FIXED_WTA_ROUTING_MODE)
    policy = MV2T._mongoose_v2_policy_config()
    @test policy.mongoose_v2_query_version == 2
    @test policy.mongoose_v2_index_version == 3
    @test policy.mongoose_v2_load_balance_lanes == 16
    @test policy.mongoose_v2_lane_identity ==
        "splitmix-domain-separated-router-seed-layer-table-neuron-v1"
    @test policy.mongoose_v2_bucket_entry_caps == (1_536, 1_280, 1_280)
    @test policy.mongoose_v2_exact_score_caps == (384, 640, 640)
    @test policy.mongoose_v2_lane_entry_caps == (96, 80, 80)

    timing_totals = MV2T.ThreeLayerTimingTotals()
    timing_totals.mongoose_v2_bucket_entries_visited = (11, 12, 13)
    timing_totals.mongoose_v2_key_rows_scored = (7, 8, 9)
    timing = MV2T._timing_snapshot(
        timing_totals;
        routing_mode=MV2T.MONGOOSE_SIMHASH_V2_ROUTING_MODE,
    )
    @test timing.mongoose_v2_bucket_entries_visited == (11, 12, 13)
    @test timing.mongoose_v2_key_rows_scored == (7, 8, 9)
    @test timing.mongoose_v2_bucket_entry_caps == (1_536, 1_280, 1_280)
    @test timing.mongoose_v2_exact_score_caps == (384, 640, 640)
    @test timing.mongoose_v2_lane_entry_caps == (96, 80, 80)
    @test !hasproperty(
        MV2T._timing_snapshot(
            timing_totals;
            routing_mode=MV2T.FIXED_WTA_ROUTING_MODE,
        ),
        :mongoose_v2_bucket_entries_visited,
    )

    runtime = _v2_runtime()
    positions = ntuple(i -> runtime.indexes[i].positions, 3)
    state = MV2O.initialize_v2_overlay(
        positions;
        warmup_updates=1,
        refresh_interval=1,
    )
    @test_throws ErrorException MV2T._trainer_from_runtime(
        runtime;
        variant=:k128,
        training_probes=(6, 5, 5),
        candidate_width=1,
        mongoose_state=state,
        routing_mode=MV2T.MONGOOSE_SIMHASH_V2_ROUTING_MODE,
    )
    trainer = MV2T._trainer_from_runtime(
        runtime;
        variant=:bounded_test,
        training_probes=(6, 5, 5),
        candidate_width=1,
        mongoose_state=state,
        routing_mode=MV2T.MONGOOSE_SIMHASH_V2_ROUTING_MODE,
    )
    @test trainer.routing_mode === MV2T.MONGOOSE_SIMHASH_V2_ROUTING_MODE
    @test trainer.workspace.mongoose_route isa MV2O.V2OverlayQueryWorkspace
    @test_throws ErrorException MV2T._trainer_from_runtime(
        runtime;
        variant=:bounded_test,
        training_probes=(6, 5, 5),
        candidate_width=1,
        mongoose_state=state,
        routing_mode=MV2T.MONGOOSE_SIMHASH_ROUTING_MODE,
    )

    zero_router_gradients = ntuple(
        i -> zeros(Float32, size(state.pending[i])),
        3,
    )
    first = MV2S.apply_accumulated_step!(
        runtime,
        _empty_accumulators(runtime),
        zeros(Float32, size(runtime.model.head)),
        zeros(Float32, length(runtime.model.bias));
        mongoose_state=state,
        mongoose_gradients=zero_router_gradients,
    )
    @test all(iszero, first.mongoose_rehash)
    refresh = MV2T.maybe_refresh_mongoose!(trainer, 1)
    @test refresh !== nothing
    @test state.active
    @test ntuple(i -> state.indexes[i].router_seed, 3) ==
        ntuple(_ -> state.seed, 3)
    @test ntuple(i -> state.indexes[i].layer_id, 3) == (1, 2, 3)
    saved_indexes = state.indexes
    state.indexes = (saved_indexes[2], saved_indexes[1], saved_indexes[3])
    @test_throws ErrorException MV2O.validate_v2_overlay!(
        state,
        ntuple(i -> runtime.model.layers[i].theta, 3),
        1;
        full_index_validation=false,
    )
    state.indexes = saved_indexes
    saved_seed = state.indexes[1].router_seed
    state.indexes[1].router_seed = xor(saved_seed, UInt64(1))
    @test_throws ErrorException MV2O.validate_v2_overlay!(
        state,
        ntuple(i -> runtime.model.layers[i].theta, 3),
        1;
        full_index_validation=false,
    )
    state.indexes[1].router_seed = saved_seed
    @test MV2O.validate_v2_overlay!(
        state,
        ntuple(i -> runtime.model.layers[i].theta, 3),
        1;
        full_index_validation=false,
    ) === state

    routed = MV2S.route_forward!(
        runtime,
        trainer.workspace.route,
        zeros(Float32, MV2S.ROUTE_DIM),
        zeros(Float32, MV2S.RAW_VALUE_DIM);
        training_probes=(6, 5, 5),
        probe_token=0x5632494e,
        mongoose_state=state,
        mongoose_workspace=trainer.workspace.mongoose_route,
        mongoose_routing_mode=trainer.routing_mode,
    )
    @test ntuple(i -> length(routed.tape.ids[i]), 3) == (48, 40, 40)
    telemetry = routed.telemetry
    for layer_id in 1:3
        @test telemetry.bucket_entries[layer_id] <=
            MV2S.LAYER_MAX_BUCKET_ENTRIES[layer_id]
        @test telemetry.scored_rows[layer_id] <= MV2S.LAYER_MAX_SCORED_ROWS[layer_id]
        @test telemetry.mongoose_v2_bucket_entries_available[layer_id] ==
            telemetry.bucket_entries[layer_id] +
            telemetry.mongoose_v2_truncated_bucket_entries[layer_id]
        @test sum(telemetry.mongoose_v2_table_entries_available[layer_id]) ==
            telemetry.mongoose_v2_bucket_entries_available[layer_id]
        @test sum(telemetry.mongoose_v2_table_entries_visited[layer_id]) ==
            telemetry.bucket_entries[layer_id]
        @test sum(telemetry.mongoose_v2_lane_entries_available[layer_id]) ==
            telemetry.mongoose_v2_bucket_entries_available[layer_id]
        @test sum(telemetry.mongoose_v2_lane_entries_visited[layer_id]) ==
            telemetry.bucket_entries[layer_id]
    end
    @test_throws ErrorException MV2S.route_forward!(
        runtime,
        trainer.workspace.route,
        zeros(Float32, MV2S.ROUTE_DIM),
        zeros(Float32, MV2S.RAW_VALUE_DIM);
        mongoose_state=state,
        mongoose_workspace=trainer.workspace.mongoose_route,
        mongoose_routing_mode=MV2T.MONGOOSE_SIMHASH_ROUTING_MODE,
    )

    dy = ones(Float32, MV2S.OUTPUT_DIM)
    vjp = MV2S.vjp_selected_parameters(runtime.model, routed.tape, dy)
    accumulators = _empty_accumulators(runtime)
    for layer_id in 1:3
        MV2S.accumulate_layer_vjp!(
            accumulators[layer_id],
            vjp.ids[layer_id],
            vjp.dtheta[layer_id],
        )
    end
    inactive_ids = ntuple(3) do layer_id
        selected = Set(Int.(vjp.ids[layer_id]))
        findfirst(id -> !(id in selected), axes(runtime.model.layers[layer_id].theta, 2))
    end
    inactive_before = ntuple(3) do layer_id
        id = inactive_ids[layer_id]
        optimizer = runtime.bank_optimizers[layer_id]
        (;
            theta=copy(runtime.model.layers[layer_id].theta[:, id]),
            m=copy(optimizer.m[:, id]),
            v=copy(optimizer.v[:, id]),
            event=optimizer.event_count[id],
            last_event=optimizer.last_event_step[id],
            last_decay=optimizer.last_log_decay[id],
        )
    end
    second = MV2S.apply_accumulated_step!(
        runtime,
        accumulators,
        vjp.dhead,
        vjp.dbias;
        mongoose_state=state,
        mongoose_gradients=zero_router_gradients,
    )
    @test all(optimizer -> optimizer.global_step == UInt64(2), runtime.bank_optimizers)
    @test all(optimizer -> optimizer.step == UInt64(2), state.optimizers)
    @test all(item -> item.projected_rows >= 0, second.mongoose_rehash_telemetry)
    for layer_id in 1:3
        id = inactive_ids[layer_id]
        optimizer = runtime.bank_optimizers[layer_id]
        before = inactive_before[layer_id]
        @test runtime.model.layers[layer_id].theta[:, id] == before.theta
        @test optimizer.m[:, id] == before.m
        @test optimizer.v[:, id] == before.v
        @test optimizer.event_count[id] == before.event
        @test optimizer.last_event_step[id] == before.last_event
        @test optimizer.last_log_decay[id] == before.last_decay
    end
end

@testset "scheduled v2 refresh retains full-bank validation" begin
    runtime = _v2_runtime(neurons=(64, 64, 64), active=(48, 40, 40))
    positions = ntuple(i -> runtime.indexes[i].positions, 3)
    state = MV2O.initialize_v2_overlay(
        positions;
        warmup_updates=1,
        refresh_interval=1,
    )
    zero_router_gradients = ntuple(i -> zeros(Float32, size(state.pending[i])), 3)
    MV2S.apply_accumulated_step!(
        runtime,
        _empty_accumulators(runtime),
        zeros(Float32, size(runtime.model.head)),
        zeros(Float32, length(runtime.model.bias));
        mongoose_state=state,
        mongoose_gradients=zero_router_gradients,
    )
    thetas = ntuple(i -> runtime.model.layers[i].theta, 3)
    prepared = MV2O.prepare_v2_refresh(state, thetas, 1)
    saved_code = prepared.indexes[1].codes[1]
    prepared.indexes[1].codes[1] = Int16(
        mod(Int(saved_code) + 1, 1 << MV2O.BITS_PER_TABLE),
    )
    @test_throws ErrorException MV2O.commit_v2_refresh!(state, prepared, thetas)
    @test !state.active
    @test state.indexes === nothing
    @test state.refresh_count == 0
    @test state.last_refresh_update == 0
end

@testset "MONGOOSE v2 bounded checkpoint round trip is test-only" begin
    runtime = _v2_runtime(neurons=(64, 64, 64), active=(48, 40, 40))
    positions = ntuple(i -> runtime.indexes[i].positions, 3)
    state = MV2O.initialize_v2_overlay(
        positions;
        warmup_updates=2,
        refresh_interval=2,
    )
    trainer = MV2T._trainer_from_runtime(
        runtime;
        variant=:bounded_test,
        training_probes=(6, 5, 5),
        candidate_width=1,
        mongoose_state=state,
        routing_mode=MV2T.MONGOOSE_SIMHASH_V2_ROUTING_MODE,
    )
    zero_router_gradients = ntuple(
        i -> zeros(Float32, size(state.pending[i])),
        3,
    )
    for _ in 1:2
        MV2S.apply_accumulated_step!(
            runtime,
            _empty_accumulators(runtime),
            zeros(Float32, size(runtime.model.head)),
            zeros(Float32, length(runtime.model.bias));
            mongoose_state=state,
            mongoose_gradients=zero_router_gradients,
        )
    end
    @test MV2T.maybe_refresh_mongoose!(trainer, 2) !== nothing
    @test state.active
    MV2S.apply_accumulated_step!(
        runtime,
        _empty_accumulators(runtime),
        zeros(Float32, size(runtime.model.head)),
        zeros(Float32, length(runtime.model.bias));
        mongoose_state=state,
        mongoose_gradients=zero_router_gradients,
    )
    @test !MV2O.refresh_due(state, 3)

    split = (;
        training_rows=[1, 2],
        validation_rows=[3],
        training_groups=[101],
        validation_groups=[202],
        predefined=true,
    )
    sampler = MV2T.BeatFirstTrainingCore.EpochSampler(
        split.training_rows,
        Xoshiro(0x5632434845434b50),
    )
    consumed_rows = [
        only(MV2T.BeatFirstTrainingCore.next_batch!(sampler, 1))
        for _ in 1:3
    ]
    @test all(row -> row in split.training_rows, consumed_rows)
    training_eval_rows = [1]
    validation_eval_rows = [3]
    split_metadata = MV2T._split_checkpoint_metadata(
        split,
        training_eval_rows,
        validation_eval_rows,
    )
    config = _bounded_v2_checkpoint_config(state)
    history = Any[(update=3, contract=:bounded_v2_nonrefresh)]

    @test_throws ErrorException MV2T._assert_overlay_checkpoint_state!(
        trainer,
        config,
        3,
    )
    @test MV2T._assert_overlay_checkpoint_state!(
        trainer,
        config,
        3;
        _bounded_test_contract=true,
    )
    missing_contract = merge(config, (; bounded_test_contract=false))
    @test_throws ErrorException MV2T._assert_overlay_checkpoint_state!(
        trainer,
        missing_contract,
        3;
        _bounded_test_contract=true,
    )
    production_like = merge(config, (;
        variant=:k128,
        bounded_test_contract=false,
    ))
    @test_throws ErrorException MV2T._assert_overlay_checkpoint_state!(
        trainer,
        production_like,
        3,
    )

    mktempdir() do directory
        path = joinpath(directory, "bounded_v2_checkpoint.jls")
        @test_throws ArgumentError MV2T.save_three_layer_teacher_checkpoint(
            path,
            trainer,
            sampler,
            config,
            split_metadata,
            history,
            3,
        )
        artifact = MV2T.save_three_layer_teacher_checkpoint(
            path,
            trainer,
            sampler,
            config,
            split_metadata,
            history,
            3;
            _bounded_test_contract=true,
        )
        @test artifact.update == 3
        @test artifact.variant === :bounded_test
        @test artifact.bytes == filesize(path)
        @test_throws ArgumentError MV2T.restore_three_layer_teacher_checkpoint(
            path,
            config,
            split,
            training_eval_rows,
            validation_eval_rows,
        )
        restored = MV2T.restore_three_layer_teacher_checkpoint(
            path,
            config,
            split,
            training_eval_rows,
            validation_eval_rows;
            _bounded_test_contract=true,
        )
        @test restored.update == 3
        @test restored.history == history
        @test restored.trainer.variant === :bounded_test
        @test restored.trainer.routing_mode ===
            MV2T.MONGOOSE_SIMHASH_V2_ROUTING_MODE
        @test restored.trainer.mongoose_state isa MV2O.MongooseV2OverlayState
        @test restored.trainer.mongoose_state.active
        @test _serialized_v2(restored.trainer.runtime) == _serialized_v2(runtime)
        @test _serialized_v2(restored.trainer.mongoose_state) ==
            _serialized_v2(state)
        @test only(MV2T.BeatFirstTrainingCore.next_batch!(restored.sampler, 1)) ==
            only(MV2T.BeatFirstTrainingCore.next_batch!(sampler, 1))
    end
end

@testset "MONGOOSE v2 metadata and cross-version restore gates" begin
    runtime = _v2_runtime(neurons=(64, 64, 64), active=(48, 40, 40))
    positions = ntuple(i -> runtime.indexes[i].positions, 3)
    state = MV2O.initialize_v2_overlay(
        positions;
        warmup_updates=2_000,
        refresh_interval=2_000,
    )
    trainer = MV2T._trainer_from_runtime(
        runtime;
        variant=:k128,
        training_probes=(6, 5, 5),
        candidate_width=1,
        mongoose_state=state,
        routing_mode=MV2T.MONGOOSE_SIMHASH_V2_ROUTING_MODE,
    )
    config = _v2_checkpoint_config(state)
    @test MV2T._assert_overlay_checkpoint_state!(trainer, config, 0)
    metadata = MV2T._checkpoint_metadata(config)
    @test metadata["routing_mode"] ==
        "mongoose_simhash_k7_l2_bounded_lanes_fixed_k128_v2"
    @test metadata["mongoose_v2_load_balance_lanes"] == 16
    @test metadata["mongoose_v2_lane_entry_caps"] == (96, 80, 80)
    @test MV2T._validate_checkpoint_metadata(metadata, config)

    v1_config = merge(config, (;
        routing_mode=MV2T.MONGOOSE_SIMHASH_ROUTING_MODE,
        routing_policy=MV2T.MONGOOSE_ROUTING_POLICY,
    ))
    @test_throws ErrorException MV2T._assert_overlay_checkpoint_state!(
        trainer, v1_config, 0,
    )
    bad_caps = merge(config, (; mongoose_v2_lane_entry_caps=(97, 80, 80)))
    @test_throws ErrorException MV2T._assert_overlay_checkpoint_state!(
        trainer, bad_caps, 0,
    )
    bad_seed = merge(config, (; mongoose_seed=config.mongoose_seed + 1))
    @test_throws ErrorException MV2T._assert_overlay_checkpoint_state!(
        trainer, bad_seed, 0,
    )
    bad_lane_identity = merge(config, (;
        mongoose_v2_lane_identity="legacy-table-neuron-only",
    ))
    @test_throws ErrorException MV2T._assert_overlay_checkpoint_state!(
        trainer, bad_lane_identity, 0,
    )
    tampered_metadata = copy(metadata)
    tampered_metadata["mongoose_v2_load_balance_lanes"] = 15
    @test_throws ErrorException MV2T._validate_checkpoint_metadata(
        tampered_metadata, config,
    )

    current_timing = MV2T._timing_state(MV2T.ThreeLayerTimingTotals())
    legacy_names = Tuple(filter(
        name -> !startswith(String(name), "mongoose_v2_"),
        collect(propertynames(current_timing)),
    ))
    legacy_timing = NamedTuple{legacy_names}(
        Tuple(getproperty(current_timing, name) for name in legacy_names),
    )
    restored_timing = MV2T._restore_timing_state(legacy_timing)
    @test restored_timing.mongoose_v2_bucket_entries_visited == (0, 0, 0)
    @test restored_timing.mongoose_v2_key_rows_scored == (0, 0, 0)
    @test restored_timing.mongoose_v2_bucket_entries_available == (0, 0, 0)
    @test restored_timing.mongoose_v2_overloaded_routes == (0, 0, 0)
end
