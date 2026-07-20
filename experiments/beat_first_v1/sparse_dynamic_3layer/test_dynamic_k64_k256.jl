using Test
using Random
using Serialization
using Statistics

include(joinpath(@__DIR__, "teacher_training.jl"))

const DKT = BeatFirstThreeLayerTeacherTraining
const DKC = DKT.BeatFirstTrainingCore
const DKS = Main.SparseDynamic3Layer

function _dynamic_batch(;
    width::Int=DKT.LEARNER_WIDTH,
    valid::Int=3,
    seed::UInt64=0x44594e4b54455354,
)
    batch = DKC.allocate_host_batch(1; max_candidates=width)
    rng = Xoshiro(seed)
    for array in values(batch.inputs)
        rand!(rng, array)
    end
    batch.mask[1:valid, 1] .= 1.0f0
    teacher = Float32.(valid:-1:1)
    batch.targets.teacher_q[1:valid, 1] .= teacher
    batch.targets.teacher_z[1:valid, 1] .= valid == 1 ? 0.0f0 :
        (teacher .- sum(teacher) / valid) ./
            max(std(teacher; corrected=false), 1.0f-4)
    batch.targets.top1_mask[1, 1] = 1.0f0
    batch.targets.top2_mask[min(2, valid), 1] = 1.0f0
    batch.targets.margin[1, 1] = valid == 1 ? 0.0f0 : teacher[1] - teacher[2]
    batch.targets.death_mask[1:valid, 1] .= 1.0f0
    return batch
end

function _serialized_dynamic(value)
    io = IOBuffer()
    serialize(io, value)
    return take!(io)
end

function _serialized_dynamic_policy_without_timings(value)
    copy_value = deepcopy(value)
    copy_value.scout_forward_nanoseconds = UInt64(0)
    copy_value.expanded_forward_nanoseconds = UInt64(0)
    return _serialized_dynamic(copy_value)
end

function _runtime_arrays_equal(left, right)
    @test left.model.head == right.model.head
    @test left.model.bias == right.model.bias
    @test left.head_optimizer.m_weight == right.head_optimizer.m_weight
    @test left.head_optimizer.v_weight == right.head_optimizer.v_weight
    @test left.head_optimizer.m_bias == right.head_optimizer.m_bias
    @test left.head_optimizer.v_bias == right.head_optimizer.v_bias
    @test left.head_optimizer.step == right.head_optimizer.step
    for layer_id in 1:3
        @test left.model.layers[layer_id].theta ==
            right.model.layers[layer_id].theta
        lo = left.bank_optimizers[layer_id]
        ro = right.bank_optimizers[layer_id]
        @test lo.m == ro.m
        @test lo.v == ro.v
        @test lo.event_count == ro.event_count
        @test lo.last_event_step == ro.last_event_step
        @test lo.last_log_decay == ro.last_log_decay
        @test lo.global_step == ro.global_step
        @test lo.global_log_decay == ro.global_log_decay
        @test left.indexes[layer_id].head == right.indexes[layer_id].head
        @test left.indexes[layer_id].next == right.indexes[layer_id].next
        @test left.indexes[layer_id].prev == right.indexes[layer_id].prev
        @test left.indexes[layer_id].codes == right.indexes[layer_id].codes
    end
end

function _force_clear_q_head!(runtime, routes; target_margin::Float32=0.05f0)
    best_gap = -Inf
    best_hidden = 0
    best_sign = 0.0f0
    for hidden in 1:DKS.LATENT_DIM, direction in (-1.0f0, 1.0f0)
        scores = sort!(
            Float64[
                direction * route.tape.latent[hidden] for route in routes
            ];
            rev=true,
        )
        gap = scores[1] - scores[2]
        if gap > best_gap
            best_gap = gap
            best_hidden = hidden
            best_sign = direction
        end
    end
    best_gap > 0.0 || error("scout latents did not expose a separating coordinate")
    fill!(@view(runtime.model.head[1, :]), 0.0f0)
    runtime.model.head[1, best_hidden] =
        best_sign * Float32(Float64(target_margin) / best_gap)
    runtime.model.bias[1] = 0.0f0
    return nothing
end

function _route_support(routes)
    support = ntuple(_ -> Set{Int32}(), 3)
    for route in routes, layer_id in 1:3
        union!(support[layer_id], route.tape.ids[layer_id])
    end
    return support
end

@testset "dynamic-k stable state gate boundaries" begin
    threshold = DKT.DYNAMIC_K_MARGIN_THRESHOLD
    for (margin, expanded) in (
        (prevfloat(threshold), true),
        (threshold, false),
        (nextfloat(threshold), false),
    )
        raw = zeros(Float32, DKS.OUTPUT_DIM, 4)
        raw[1, 1] = margin
        raw[1, 2] = 0.0f0
        raw[1, 3] = -1.0f0
        raw[1, 4] = Float32(NaN) # padding is outside candidate_count
        decision = DKT.dynamic_k_stable_top2_margin(raw, 3)
        @test decision.top1_index == 1
        @test decision.top2_index == 2
        @test decision.expanded === expanded
    end
    tied = zeros(Float32, DKS.OUTPUT_DIM, 3)
    tie = DKT.dynamic_k_stable_top2_margin(tied, 3)
    @test tie.margin == 0.0f0
    @test tie.top1_index == 1
    @test tie.top2_index == 2
    @test tie.expanded
    singleton = DKT.dynamic_k_stable_top2_margin(tied, 1)
    @test singleton.margin == Float32(Inf)
    @test singleton.top1_index == 1
    @test singleton.top2_index == 0
    @test !singleton.expanded
    nonfinite = copy(tied)
    nonfinite[1, 2] = Float32(Inf)
    @test_throws ErrorException DKT.dynamic_k_stable_top2_margin(nonfinite, 3)

    counters = DKT.DynamicKPolicyState()
    DKT.record_dynamic_k_state!(
        counters;
        expanded=false,
        scout_margin=Float32(Inf),
        candidate_count=1,
        scout_forward_nanoseconds=1,
        expanded_forward_nanoseconds=0,
    )
    DKT.record_dynamic_k_state!(
        counters;
        expanded=true,
        scout_margin=0.0f0,
        candidate_count=3,
        scout_forward_nanoseconds=2,
        expanded_forward_nanoseconds=3,
    )
    snapshot = DKT.dynamic_k_policy_snapshot(counters)
    @test snapshot.candidates_expanded == 3
    @test snapshot.candidates_total == 4
    @test snapshot.candidate_weighted_expansion_fraction == 0.75
    @test snapshot.core_training_macs ==
        DKT.DYNAMIC_K_CLEAR_TRAINING_MACS_PER_CANDIDATE +
        3 * DKT.DYNAMIC_K_EXPANDED_TOTAL_MACS_PER_CANDIDATE
    @test_throws ErrorException DKT.assert_dynamic_k_panel!(counters)

    # The published cap is the exact candidate-weighted mixture that lands on
    # the fixed-k128 core-MAC budget.  Exercise the integer inequality itself,
    # not merely its decimal rendering.
    exact = DKT.DynamicKPolicyState()
    DKT.record_dynamic_k_state!(
        exact;
        expanded=false,
        scout_margin=0.03f0,
        candidate_count=158_052,
        scout_forward_nanoseconds=1,
        expanded_forward_nanoseconds=0,
    )
    DKT.record_dynamic_k_state!(
        exact;
        expanded=true,
        scout_margin=0.0f0,
        candidate_count=63_016,
        scout_forward_nanoseconds=1,
        expanded_forward_nanoseconds=1,
    )
    exact_snapshot = DKT.assert_dynamic_k_panel!(exact)
    @test exact_snapshot.core_training_macs ==
        DKT.DYNAMIC_K_K128_BUDGET_MACS_PER_CANDIDATE *
            exact_snapshot.candidates_total
end

@testset "dynamic-k shared views and chosen-only expanded update" begin
    trainer = DKT.initialize_three_layer_teacher_trainer(
        variant=:k128,
        model_seed=2026071902,
        candidate_width=DKT.LEARNER_WIDTH,
        routing_mode=DKT.DYNAMIC_K64_K256_ROUTING_MODE,
    )
    runtime = trainer.runtime
    views = trainer.dynamic_views
    @test ntuple(i -> runtime.model.layers[i].active_count, 3) == (48, 40, 40)
    @test ntuple(i -> views.scout.model.layers[i].active_count, 3) == (24, 20, 20)
    @test ntuple(i -> views.expanded.model.layers[i].active_count, 3) == (96, 80, 80)
    for view in (views.scout, views.expanded), layer_id in 1:3
        @test view.model.layers[layer_id].theta ===
            runtime.model.layers[layer_id].theta
        @test view.indexes[layer_id] === runtime.indexes[layer_id]
        @test view.bank_optimizers[layer_id] === runtime.bank_optimizers[layer_id]
    end
    @test views.scout.model.head === runtime.model.head
    @test views.expanded.model.bias === runtime.model.bias
    @test views.scout.head_optimizer === runtime.head_optimizer

    batch = _dynamic_batch(valid=3)
    token = DKT._probe_token(1, 1, 1)
    explicit = DKT.route_forward_dynamic!(
        views,
        trainer.workspace.dynamic_scout_route,
        batch.inputs,
        1;
        active_counts=DKT.DYNAMIC_K_SCOUT_COUNTS,
        training_probes=DKT.DYNAMIC_K_SCOUT_TRAINING_PROBES,
        probe_token=token,
    )
    native = DKS.route_forward!(
        views.scout,
        DKS.ThreeLayerWorkspace(views.scout),
        batch.inputs,
        1;
        training_probes=DKT.DYNAMIC_K_SCOUT_TRAINING_PROBES,
        probe_token=token,
    )
    @test _serialized_dynamic(explicit.output) == _serialized_dynamic(native.output)
    @test explicit.tape.ids == native.tape.ids
    @test all(
        explicit.tape.ids[i] !==
            trainer.workspace.dynamic_scout_route.selected_ids[i] for i in 1:3
    )
    @test_throws ArgumentError DKT.route_forward_dynamic!(
        views,
        trainer.workspace.dynamic_scout_route,
        batch.inputs,
        1;
        active_counts=(48, 40, 40),
        training_probes=(6, 5, 5),
        probe_token=token,
    )
    @test_throws ArgumentError DKT.route_forward_dynamic!(
        views,
        trainer.workspace.dynamic_scout_route,
        batch.inputs,
        1;
        active_counts=DKT.DYNAMIC_K_SCOUT_COUNTS,
        training_probes=(4, 2, 2),
        probe_token=token,
    )

    # Full clear-path training witness.  A one-coordinate Q projection is
    # chosen from the already-routed scout latents so the state margin is
    # deterministically above 0.02.  There must be no k256 pass or tape and the
    # committed sparse support must be exactly the k64 scout support.
    clear_routes = [
        DKT.route_forward_dynamic!(
            views,
            trainer.workspace.dynamic_scout_route,
            batch.inputs,
            candidate;
            active_counts=DKT.DYNAMIC_K_SCOUT_COUNTS,
            training_probes=DKT.DYNAMIC_K_SCOUT_TRAINING_PROBES,
            probe_token=DKT._probe_token(1, 1, candidate),
        )
        for candidate in 1:3
    ]
    clear_support = _route_support(clear_routes)
    _force_clear_q_head!(runtime, clear_routes)
    clear_oracle = DKT._dynamic_k_forward_state!(
        trainer,
        batch,
        3;
        row_id=1,
        training_step=1,
        training=true,
    )
    @test !clear_oracle.decision.expanded
    @test clear_oracle.expanded_pass === nothing
    @test clear_oracle.chosen_counts == DKT.DYNAMIC_K_SCOUT_COUNTS
    @test all(
        length(trainer.workspace.traces[candidate].ids[layer_id]) ==
            DKT.DYNAMIC_K_SCOUT_COUNTS[layer_id]
        for candidate in 1:3 for layer_id in 1:3
    )
    clear_result = DKT.teacher_train_step!(
        trainer, batch; row_id=1, training_step=1,
    )
    clear_dynamic = clear_result.accounting.dynamic_k
    @test !clear_dynamic.expanded
    @test clear_dynamic.scout_margin > Float64(DKT.DYNAMIC_K_MARGIN_THRESHOLD)
    @test clear_dynamic.chosen_active_counts == DKT.DYNAMIC_K_SCOUT_COUNTS
    @test clear_dynamic.expanded_pass_nanoseconds == 0
    @test clear_dynamic.core_training_macs ==
        3 * DKT.DYNAMIC_K_CLEAR_TRAINING_MACS_PER_CANDIDATE
    @test clear_dynamic.chosen_training_macs ==
        3 * (DKT.DYNAMIC_K_CLEAR_TRAINING_MACS_PER_CANDIDATE -
            DKT.DYNAMIC_K_SCOUT_FORWARD_MACS_PER_CANDIDATE)
    for layer_id in 1:3
        @test Set(trainer.workspace.accumulators[layer_id].ids) ==
            clear_support[layer_id]
    end

    # A zero Q head creates an exact stable tie and therefore forces expansion
    # on the following update.  Re-route after the clear update because the
    # sparse bank and its WTA index have legitimately changed.
    fill!(@view(runtime.model.head[1, :]), 0.0f0)
    runtime.model.bias[1] = 0.0f0
    scout_routes = [
        DKT.route_forward_dynamic!(
            views,
            trainer.workspace.dynamic_scout_route,
            batch.inputs,
            candidate;
            active_counts=DKT.DYNAMIC_K_SCOUT_COUNTS,
            training_probes=DKT.DYNAMIC_K_SCOUT_TRAINING_PROBES,
            probe_token=DKT._probe_token(2, 2, candidate),
        )
        for candidate in 1:3
    ]
    expanded_routes = [
        DKT.route_forward_dynamic!(
            views,
            trainer.workspace.dynamic_expanded_route,
            batch.inputs,
            candidate;
            active_counts=DKT.DYNAMIC_K_EXPANDED_COUNTS,
            training_probes=DKT.DYNAMIC_K_EXPANDED_TRAINING_PROBES,
            probe_token=DKT._probe_token(2, 2, candidate),
        )
        for candidate in 1:3
    ]
    scout_support = _route_support(scout_routes)
    expanded_support = _route_support(expanded_routes)
    event_before = ntuple(i -> copy(runtime.bank_optimizers[i].event_count), 3)
    last_before = ntuple(i -> copy(runtime.bank_optimizers[i].last_event_step), 3)
    scout_only = ntuple(i -> setdiff(scout_support[i], expanded_support[i]), 3)
    scout_only_m = ntuple(i -> Dict(
        id => copy(runtime.bank_optimizers[i].m[:, Int(id)]) for id in scout_only[i]
    ), 3)
    scout_only_v = ntuple(i -> Dict(
        id => copy(runtime.bank_optimizers[i].v[:, Int(id)]) for id in scout_only[i]
    ), 3)

    result = DKT.teacher_train_step!(trainer, batch; row_id=2, training_step=2)
    dynamic = result.accounting.dynamic_k
    @test dynamic.expanded
    @test dynamic.scout_margin == 0.0
    @test dynamic.scout_active_counts == (24, 20, 20)
    @test dynamic.chosen_active_counts == (96, 80, 80)
    @test dynamic.scout_forward_macs ==
        3 * DKT.DYNAMIC_K_SCOUT_FORWARD_MACS_PER_CANDIDATE
    @test dynamic.core_training_macs ==
        3 * DKT.DYNAMIC_K_EXPANDED_TOTAL_MACS_PER_CANDIDATE
    @test result.accounting.parameter_vjp_macs +
        result.accounting.parameter_vjp_sketch_accumulates ==
        3 * (DKT.DYNAMIC_K_EXPANDED_CHOSEN_TRAINING_MACS_PER_CANDIDATE - 111_184)
    for layer_id in 1:3
        @test Set(trainer.workspace.accumulators[layer_id].ids) ==
            expanded_support[layer_id]
        for neuron in 1:length(event_before[layer_id])
            if Int32(neuron) in expanded_support[layer_id]
                @test runtime.bank_optimizers[layer_id].event_count[neuron] ==
                    event_before[layer_id][neuron] + 1
                @test runtime.bank_optimizers[layer_id].last_event_step[neuron] == 2
            else
                @test runtime.bank_optimizers[layer_id].event_count[neuron] ==
                    event_before[layer_id][neuron]
                @test runtime.bank_optimizers[layer_id].last_event_step[neuron] ==
                    last_before[layer_id][neuron]
            end
        end
        for id in scout_only[layer_id]
            @test runtime.bank_optimizers[layer_id].m[:, Int(id)] ==
                scout_only_m[layer_id][id]
            @test runtime.bank_optimizers[layer_id].v[:, Int(id)] ==
                scout_only_v[layer_id][id]
        end
    end

    state_before_eval = _serialized_dynamic(trainer.dynamic_state)
    clocks_before_eval = (
        ntuple(i -> runtime.bank_optimizers[i].global_step, 3),
        runtime.head_optimizer.step,
    )
    DKT.predict_three_layer_raw!(trainer, batch)
    @test _serialized_dynamic(trainer.dynamic_state) == state_before_eval
    @test (
        ntuple(i -> runtime.bank_optimizers[i].global_step, 3),
        runtime.head_optimizer.step,
    ) == clocks_before_eval

    split = (;
        training_rows=[1, 2, 3],
        validation_rows=[4],
        training_groups=[101],
        validation_groups=[202],
        predefined=true,
    )
    sampler = DKC.EpochSampler(split.training_rows, Xoshiro(0x44594e4b53414d50))
    only(DKC.next_batch!(sampler, 1))
    only(DKC.next_batch!(sampler, 1))
    split_metadata = DKT._split_checkpoint_metadata(split, [1], [4])
    digest = repeat("0", 64)
    config = (;
        format_version=DKT.TRAINING_STATE_FORMAT_VERSION,
        variant=:k128,
        active_counts=(48, 40, 40),
        training_probes=(6, 5, 5),
        routing_mode=DKT.DYNAMIC_K64_K256_ROUTING_MODE,
        routing_policy=DKT.DYNAMIC_K64_K256_ROUTING_POLICY,
        dynamic_k_margin_threshold=Float64(DKT.DYNAMIC_K_MARGIN_THRESHOLD),
        dynamic_k_scout_counts=DKT.DYNAMIC_K_SCOUT_COUNTS,
        dynamic_k_expanded_counts=DKT.DYNAMIC_K_EXPANDED_COUNTS,
        dynamic_k_scout_training_probes=DKT.DYNAMIC_K_SCOUT_TRAINING_PROBES,
        dynamic_k_expanded_training_probes=
            DKT.DYNAMIC_K_EXPANDED_TRAINING_PROBES,
        dynamic_k_candidate_expansion_cap=DKT.DYNAMIC_K_CANDIDATE_EXPANSION_CAP,
        dynamic_k_mean_core_macs_cap=
            Float64(DKT.DYNAMIC_K_K128_BUDGET_MACS_PER_CANDIDATE),
        mongoose_route_dim=DKS.MongooseSimHashOverlay.ROUTE_DIM,
        mongoose_bits_per_table=DKS.MongooseSimHashOverlay.BITS_PER_TABLE,
        mongoose_tables=DKS.MongooseSimHashOverlay.TABLES,
        mongoose_router_parameters=0,
        mongoose_column_normalization=
            DKS.MongooseSimHashOverlay.COLUMN_NORMALIZATION,
        mongoose_learning_rate=1.0e-4,
        mongoose_beta=1.0,
        mongoose_seed=0x4d4f4e47,
        mongoose_warmup_updates=2_000,
        mongoose_refresh_interval=2_000,
        learner_width=DKT.LEARNER_WIDTH,
        objective_margin_weight=Float64(DKC.MARGIN_WEIGHT),
        objective_margin_mode=DKC.FIXED_TEACHER_TOP2_MARGIN_MODE,
        source_sha256=digest,
        environment_project_sha256=digest,
        environment_manifest_sha256=digest,
        dataset_manifest_sha256=digest,
        pairing_contract_sha256=digest,
    )
    mktempdir() do directory
        path = joinpath(directory, "dynamic_k_exact_continuation.jls")
        DKT.save_three_layer_teacher_checkpoint(
            path,
            trainer,
            sampler,
            config,
            split_metadata,
            Any[
                (update=1, loss=clear_result.loss),
                (update=2, loss=result.loss),
            ],
            2,
        )
        loaded = DKS.load_checkpoint(path)
        @test loaded.topology.active_counts == (48, 40, 40)
        @test ntuple(i -> loaded.runtime.model.layers[i].active_count, 3) ==
            (48, 40, 40)
        @test !hasproperty(loaded.training_state, :dynamic_views)
        @test hasproperty(loaded.training_state, :dynamic_k_state)
        restored = DKT.restore_three_layer_teacher_checkpoint(
            path,
            config,
            split,
            [1],
            [4];
            loaded,
        )
        @test _serialized_dynamic(restored.trainer.dynamic_state) ==
            _serialized_dynamic(trainer.dynamic_state)
        @test restored.trainer.dynamic_views.base === restored.trainer.runtime
        @test ntuple(
            i -> restored.trainer.runtime.model.layers[i].active_count,
            3,
        ) == (48, 40, 40)
        original_row = only(DKC.next_batch!(sampler, 1))
        restored_row = only(DKC.next_batch!(restored.sampler, 1))
        @test original_row == restored_row
        next_batch = _dynamic_batch(seed=0x44594e4b4e455854)
        original = DKT.teacher_train_step!(
            trainer, next_batch; row_id=original_row, training_step=3,
        )
        replay = DKT.teacher_train_step!(
            restored.trainer,
            next_batch;
            row_id=restored_row,
            training_step=3,
        )
        @test original.loss == replay.loss
        @test original.components == replay.components
        @test original.accounting.dynamic_k.expanded ==
            replay.accounting.dynamic_k.expanded
        @test original.accounting.dynamic_k.scout_margin ==
            replay.accounting.dynamic_k.scout_margin
        # Routing decisions/counters continue exactly; independently measured
        # wall-clock pass durations are intentionally not deterministic.
        @test _serialized_dynamic_policy_without_timings(trainer.dynamic_state) ==
            _serialized_dynamic_policy_without_timings(restored.trainer.dynamic_state)
        _runtime_arrays_equal(trainer.runtime, restored.trainer.runtime)
        @test _serialized_dynamic(DKC.sampler_snapshot(sampler)) ==
            _serialized_dynamic(DKC.sampler_snapshot(restored.sampler))
    end
end

@testset "legacy low-level checkpoint remains loadable" begin
    model = DKS.initialize_model(
        Xoshiro(0x4c4547414359444b);
        neuron_counts=(16, 16, 16),
        active_counts=(2, 2, 2),
    )
    runtime = DKS.initialize_runtime(model)
    mktempdir() do directory
        path = joinpath(directory, "legacy_fixed_wta.jls")
        DKS.save_checkpoint(path, runtime)
        restored = DKS.load_checkpoint(path)
        @test restored.training_state === nothing
        @test ntuple(i -> restored.runtime.model.layers[i].active_count, 3) ==
            (2, 2, 2)
    end
end
