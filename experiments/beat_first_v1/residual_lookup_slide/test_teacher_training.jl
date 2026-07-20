using Test
using Random
using Statistics

include(joinpath(@__DIR__, "teacher_training.jl"))

const RLS = ResidualLookupSlideR0TeacherTraining
const Core = RLS.TrainingCore
const Lookup = RLS.Lookup

function synthetic_batch(candidate_width::Int)
    batch = Core.allocate_host_batch(1; max_candidates=candidate_width)
    candidates = min(3, candidate_width)
    candidates >= 2 || error("synthetic objective test requires at least two candidates")
    batch.mask[1:candidates, 1] .= 1.0f0
    teacher = Float32[1.4, 0.3, -0.7][1:candidates]
    teacher_mean = mean(teacher)
    teacher_scale = max(std(teacher; corrected=false), 1.0f-4)
    batch.targets.teacher_q[1:candidates, 1] .= teacher
    batch.targets.teacher_z[1:candidates, 1] .=
        (teacher .- teacher_mean) ./ teacher_scale
    ordering = sortperm(teacher; rev=true, alg=MergeSort)
    batch.targets.top1_mask[ordering[1], 1] = 1.0f0
    batch.targets.top2_mask[ordering[2], 1] = 1.0f0
    batch.targets.margin[1, 1] = teacher[ordering[1]] - teacher[ordering[2]]
    batch.targets.death_mask[1:candidates, 1] .= 1.0f0
    batch.targets.death[candidates, 1] = 1.0f0
    batch.targets.line_clear[1:candidates, 1] .= Float32.(0:(candidates - 1))
    batch.targets.max_height[1:candidates, 1] .= Float32.(4:(3 + candidates))
    batch.targets.holes[1:candidates, 1] .= Float32.(1:candidates)
    batch.targets.cavities[1:candidates, 1] .= Float32.(2:(1 + candidates))
    for candidate in 1:candidates
        batch.inputs.candidate[24 - candidate, candidate, 1, candidate] = 1.0f0
        batch.inputs.candidate[24, candidate + 1, 1, candidate] = 1.0f0
        batch.inputs.difference[24 - candidate, candidate, 1, candidate] =
            Float32(candidate) / 3.0f0
        batch.inputs.next_hold[mod(candidate - 1, 7) + 1, 1, candidate] = 1.0f0
        batch.inputs.aux[:, candidate] .=
            Float32(candidate) .* Float32.(1:37) ./ 111.0f0
    end
    return batch
end

function selected_columns(addresses, block::Int, candidate_count::Int, tables::Int)
    columns = Int[]
    for candidate in 1:candidate_count, table in 1:tables
        push!(columns, Lookup.flat_row_column(
            table,
            Int(addresses[block, table, candidate]),
        ))
    end
    return sort(unique(columns))
end

function assert_optimizer_equal(left, right)
    @test left.step == right.step
    for block in 1:Lookup.BLOCKS
        a = left.bank_states[block]
        b = right.bank_states[block]
        @test a.m == b.m
        @test a.v == b.v
        @test a.event_count == b.event_count
        @test a.last_event_step == b.last_event_step
        @test a.last_log_decay == b.last_log_decay
        @test a.global_step == b.global_step
        @test a.global_log_decay == b.global_log_decay
    end
    @test left.head_state.m_head == right.head_state.m_head
    @test left.head_state.v_head == right.head_state.v_head
    @test left.head_state.m_bias == right.head_state.m_bias
    @test left.head_state.v_bias == right.head_state.v_bias
    @test left.head_state.step == right.head_state.step
    @test left.alpha_state.m == right.alpha_state.m
    @test left.alpha_state.v == right.alpha_state.v
    @test left.alpha_state.step == right.alpha_state.step
end

function assert_trainers_equal(left, right)
    @test left.update == right.update
    @test left.model.banks == right.model.banks
    @test left.model.alpha_logits == right.model.alpha_logits
    @test left.model.head == right.model.head
    @test left.model.bias == right.model.bias
    assert_optimizer_equal(left.optimizer, right.optimizer)
    @test left.usage.counts == right.usage.counts
    @test left.usage.calls == right.usage.calls
    @test left.block_selected_gradient_events == right.block_selected_gradient_events
    @test left.block_nonzero_selected_gradient_events ==
          right.block_nonzero_selected_gradient_events
    @test left.timed_updates == right.timed_updates
    @test left.timed_candidates == right.timed_candidates
end

@testset "Residual Lookup-SLIDE R0 frozen teacher contract" begin
    @test Lookup.CARRIER_DIM == 256
    @test Lookup.BLOCKS == 3
    @test Lookup.TABLES_PER_BLOCK == 76
    @test Lookup.ROWS_PER_TABLE == 343
    @test Lookup.OUTPUT_DIM == 22
    @test Lookup.TOTAL_PARAMETERS == 20_025_881
    @test Lookup.BANK_PARAMETERS == 20_020_224
    @test Lookup.EXACT_ACCOUNTING.selected_bank_parameters == 58_368
    @test RLS.LEARNER_WIDTH == 80
    @test RLS.MAXIMUM_UPDATES == 1_000
    @test RLS.EVALUATION_UPDATES == (0, 500, 1_000)
    @test RLS.CHECKPOINT_UPDATES == (500, 1_000)
    @test RLS.REVIVAL_ENABLED === false
    @test RLS.EXPLORATION_ENABLED === false
    @test_throws ErrorException RLS.initialize_trainer(;
        candidate_width=4,
        tables_per_block=2,
        model_seed=RLS.MODEL_SEED + UInt64(1),
        allow_test_geometry=true,
    )
    contract = RLS.objective_contract()
    @test contract.listnet_temperature == 0.50f0
    @test contract.listnet_weight == 1.0f0
    @test contract.q_huber_weight == 0.25f0
    @test contract.margin_huber_weight == 0.15f0
    @test contract.margin_mode == Core.FIXED_TEACHER_TOP2_MARGIN_MODE
    @test contract.death_weight == 0.10f0
    @test contract.quantile_teacher_weight == 0.05f0
    @test contract.geometry_weight == 0.10f0
    @test occursin(r"^[0-9a-f]{64}$", RLS.router_fingerprint())
end

@testset "fixed router, selected rows, dense head, and alpha" begin
    candidate_width = 4
    tables = 2
    trainer = RLS.initialize_trainer(;
        candidate_width,
        tables_per_block=tables,
        allow_test_geometry=true,
    )
    batch = synthetic_batch(candidate_width)
    raw_first = copy(RLS.predict_raw!(trainer, batch))
    addresses_first = copy(trainer.workspace.addresses)
    raw_second = copy(RLS.predict_raw!(trainer, batch))
    @test raw_first == raw_second
    @test addresses_first == trainer.workspace.addresses
    router_identity = RLS.router_fingerprint()
    candidate_count = Int(sum(batch.mask))
    supports = [selected_columns(addresses_first, block, candidate_count, tables)
                for block in 1:Lookup.BLOCKS]
    banks_before = deepcopy(trainer.model.banks)
    head_before = copy(trainer.model.head)
    alpha_before = copy(trainer.model.alpha_logits)
    step = RLS.train_step!(trainer, batch; expected_update=1)
    @test trainer.model.head != head_before
    @test trainer.model.alpha_logits != alpha_before
    @test RLS.router_fingerprint() == router_identity
    @test step.components.effective_listnet_weight == 1.0
    @test step.components.effective_margin_weight == Float64(RLS.MARGIN_WEIGHT)
    @test all(record -> record.nonzero_selected_gradient_rate == 1.0,
              step.block_gradient_rates)
    for block in 1:Lookup.BLOCKS
        state = trainer.optimizer.bank_states[block]
        event_rows = findall(==(UInt64(1)), state.event_count)
        @test event_rows == supports[block]
        inactive = setdiff(collect(axes(trainer.model.banks[block], 2)), supports[block])
        @test trainer.model.banks[block][:, inactive] == banks_before[block][:, inactive]
        @test all(iszero, state.m[:, inactive])
        @test all(iszero, state.v[:, inactive])
    end
    RLS.train_step!(trainer, batch; expected_update=2)
    table_metrics = RLS.per_table_route_metrics(trainer)
    @test length(table_metrics) == Lookup.BLOCKS * tables
    @test all(record -> record.selected == 2 * candidate_count, table_metrics)
    @test all(record -> record.unused_rows + record.used_rows == Lookup.ROWS_PER_TABLE,
              table_metrics)

    # Non-finite supervision must fail before the whole-model prepared commit.
    model_before = deepcopy(trainer.model)
    optimizer_before = deepcopy(trainer.optimizer)
    usage_before = deepcopy(trainer.usage)
    selected_before = copy(trainer.block_selected_gradient_events)
    batch.targets.teacher_q[1, 1] = Float32(NaN)
    @test_throws ErrorException RLS.train_step!(trainer, batch; expected_update=3)
    @test trainer.update == 2
    @test trainer.model.banks == model_before.banks
    @test trainer.model.head == model_before.head
    @test trainer.model.bias == model_before.bias
    @test trainer.model.alpha_logits == model_before.alpha_logits
    assert_optimizer_equal(trainer.optimizer, optimizer_before)
    @test trainer.usage.counts == usage_before.counts
    @test trainer.usage.calls == usage_before.calls
    @test trainer.block_selected_gradient_events == selected_before
end

@testset "checkpoint exact continuation and fail-closed identity" begin
    candidate_width = 4
    tables = 2
    trainer = RLS.initialize_trainer(;
        candidate_width,
        tables_per_block=tables,
        allow_test_geometry=true,
    )
    batch = synthetic_batch(candidate_width)
    RLS.train_step!(trainer, batch; expected_update=1)
    training_rows = [11, 12, 13, 14]
    sampler = Core.EpochSampler(training_rows, Xoshiro(UInt64(0x51525354)))
    Core.next_batch!(sampler, 2)
    config = RLS._config(
        "synthetic_teacher_v3",
        "checkpoint_test",
        candidate_width,
        tables;
        test_geometry=true,
        dataset_manifest_sha256_override=repeat("ab", 32),
    )
    split_metadata = (;
        training_rows,
        validation_rows=[21, 22],
        held_rows=[21, 22],
        training_groups=[1],
        validation_groups=[2],
        predefined=true,
    )
    state = RLS.TrainingRunState(
        trainer,
        sampler,
        config,
        split_metadata,
        Any[],
        nothing,
        nothing,
    )
    mktempdir() do directory
        path = joinpath(directory, "checkpoint.jls")
        receipt = RLS.save_training_checkpoint(path, state)
        @test receipt.update == 1
        @test occursin(r"^[0-9a-f]{64}$", receipt.sha256)
        restored = RLS.restore_training_checkpoint(
            path,
            config,
            split_metadata;
            expected_bytes=receipt.bytes,
            expected_sha256=receipt.sha256,
        )
        @test Core.next_batch!(state.sampler, 1) ==
              Core.next_batch!(restored.sampler, 1)
        RLS.train_step!(state.trainer, batch; expected_update=2)
        RLS.train_step!(restored.trainer, batch; expected_update=2)
        assert_trainers_equal(state.trainer, restored.trainer)
        @test rand(state.trainer.model_rng, UInt64, 8) ==
              rand(restored.trainer.model_rng, UInt64, 8)

        changed_config = merge(config, (; bank_learning_rate=2.0f-4))
        @test_throws ErrorException RLS.restore_training_checkpoint(
            path,
            changed_config,
            split_metadata;
            expected_bytes=receipt.bytes,
            expected_sha256=receipt.sha256,
        )
        changed_manifest = merge(config, (; dataset_manifest_sha256=repeat("cd", 32)))
        @test_throws ErrorException RLS.restore_training_checkpoint(
            path,
            changed_manifest,
            split_metadata;
            expected_bytes=receipt.bytes,
            expected_sha256=receipt.sha256,
        )
        changed_split = merge(split_metadata, (; held_rows=[22, 21]))
        @test_throws ErrorException RLS.restore_training_checkpoint(
            path,
            config,
            changed_split;
            expected_bytes=receipt.bytes,
            expected_sha256=receipt.sha256,
        )
    end
end
