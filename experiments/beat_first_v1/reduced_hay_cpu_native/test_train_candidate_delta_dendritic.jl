using Serialization
using Test

include(joinpath(@__DIR__, "train_candidate_delta_dendritic.jl"))

const Scratch = CandidateDeltaDendriticScratchTraining
const Ranking = Scratch.TetrisRankingBatch
const Bank = Scratch.DendriticProgramBank
const Model = Scratch.CandidateDeltaDendriticGraph
const Parallel = Scratch.CandidateDeltaDendriticBarrierless
const Optimizer = Scratch.CandidateDeltaDendriticOptimizer

function fixture_checkpoint(directory::AbstractString)
    options = Scratch.ScratchOptions(
        updates=2,
        workers=1,
        checkpoint=joinpath(directory, "latest.jls"),
        progress=joinpath(directory, "progress.jsonl"),
    )
    parameters = Model.initialize_model()
    optimizer = Optimizer.AdamWState(parameters)
    gradient = Model.ModelGradient(parameters; active_program_capacity=8)
    gradient.leaf_shared_raw[1] = 0.25f0
    gradient.forest.internal_raw[1] = -0.5f0
    gradient.forest.child_contact[2] = 0.75f0
    gradient.output.cell_raw[1] = -0.125f0
    gradient.output.anchor_weight[2] = 0.375f0
    gradient.output.context_weight[3] = 0.625f0
    gradient.output.placement_weight[4] = -0.875f0
    gradient.output.cascade_weight[5] = 0.3125f0
    gradient.output.gain[1] = -0.1875f0
    gradient.output.bias[2] = 0.4375f0
    Bank.accumulate_program_gradient!(
        gradient.program,
        3,
        Float32.(1:Bank.PAYLOAD_WIDTH),
        0.0625f0,
    )
    Optimizer.apply_adamw!(
        optimizer,
        parameters,
        gradient,
        Optimizer.AdamWConfig(),
    )
    training_rows = collect(1:16)
    validation_rows = collect(101:228)
    sampler = Scratch.RandomTrainingSampler(training_rows, options.sampler_seed)
    sampled = zeros(Int, Scratch.STATE_BATCH)
    Scratch.next_batch!(sampled, sampler)
    snapshot = Scratch.build_checkpoint(
        1,
        Scratch.STATE_BATCH,
        43,
        UInt128(123_456),
        parameters,
        optimizer,
        sampler,
        options,
        "fixture-source",
        "fixture-dataset",
        training_rows,
        validation_rows,
    )
    return (;
        options,
        parameters,
        optimizer,
        sampler,
        sampled,
        training_rows,
        validation_rows,
        snapshot,
    )
end

function model_trainables(parameters)
    return (
        leaf=parameters.leaf_shared_raw,
        program=parameters.program_bank.payload,
        forest_internal=parameters.forest.internal_raw,
        forest_contact=parameters.forest.child_contact,
        output_cell=parameters.output.cell_raw,
        output_anchor=parameters.output.anchor_weight,
        output_context=parameters.output.context_weight,
        output_placement=parameters.output.placement_weight,
        output_cascade=parameters.output.cascade_weight,
        output_gain=parameters.output.gain,
        output_bias=parameters.output.bias,
    )
end

function optimizer_arrays(optimizer)
    names = fieldnames(Optimizer.DenseMoments)
    return (
        first=NamedTuple{names}(Tuple(getfield(optimizer.first, name) for name in names)),
        second=NamedTuple{names}(Tuple(getfield(optimizer.second, name) for name in names)),
        optimizer.program_first,
        optimizer.program_second,
        optimizer.program_step_by_row,
        optimizer.placement_step_by_coordinate,
    )
end

@testset "DDF scratch CLI contract" begin
    options = Scratch.parse_options([
        "--updates=17",
        "--workers", "1",
        "--log-every", "5",
        "--evaluate-every", "10",
        "--checkpoint-every", "5",
        "--binding-mode", "none",
        "--sampler-seed", "99",
        "--learning-rate", "0.002",
        "--warmup-updates", "7",
        "--learning-rate-schedule-updates", "31",
        "--min-learning-rate-ratio", "0.2",
        "--leaf-cell-multiplier", "0.25",
        "--forest-contact-multiplier", "0.75",
        "--output-placement-multiplier", "0.5",
        "--resume", "some-checkpoint.jls",
    ])
    @test options.updates == 17
    @test options.workers == 1
    @test options.sampler_seed == UInt64(99)
    @test options.learning_rate == 0.002f0
    @test options.warmup_updates == 7
    @test options.learning_rate_schedule_updates == 31
    @test options.min_learning_rate_ratio == 0.2f0
    @test options.leaf_cell_multiplier == 0.25f0
    @test options.output_placement_multiplier == 0.5f0
    @test options.resume == "some-checkpoint.jls"
    semantic = Scratch.semantic_config(options)
    @test semantic.model_family == "candidate_delta_dendritic_forest_scratch"
    @test semantic.state_batch == 8
    @test semantic.candidate_width == 80
    @test semantic.address_scheme == Bank.ADDRESS_SCHEME
    @test semantic.forest_nodes == Model.Topology.NODE_COUNT
    @test semantic.base_learning_rate == 0.002f0
    @test semantic.warmup_updates == 7
    @test semantic.learning_rate_schedule_updates == 31
    @test semantic.learning_rate_decay_updates == 24
    @test semantic.min_learning_rate_ratio == 0.2f0
    optimizer_config = Scratch._optimizer_config(options)
    @test optimizer_config.learning_rate == 0.002f0
    @test optimizer_config.multipliers.leaf_cell == 0.25f0
    @test optimizer_config.multipliers.forest_contact == 0.75f0
    @test optimizer_config.multipliers.output_placement == 0.5f0
    schedule = Scratch._learning_rate_schedule(options)
    @test schedule.warmup_updates == 7
    @test schedule.decay_updates == 24
    scheduled_rate = Optimizer.learning_rate_at(schedule, 3)
    step_config = Scratch._with_learning_rate(optimizer_config, scheduled_rate)
    @test step_config.learning_rate === scheduled_rate
    @test step_config.multipliers === optimizer_config.multipliers
    @test step_config.weight_decay === optimizer_config.weight_decay
    @test_throws ErrorException Scratch.parse_options(["--factor-multiplier", "1"])
    @test_throws ErrorException Scratch.parse_options(["--workers", "0"])
    @test_throws ErrorException Scratch.parse_options(["--warmup-updates", "-1"])
    @test_throws ErrorException Scratch.parse_options([
        "--learning-rate-schedule-updates", "0",
    ])
    @test_throws ErrorException Scratch.parse_options([
        "--min-learning-rate-ratio", "1.1",
    ])
    @test_throws ErrorException Scratch.parse_options(["--unknown", "1"])
end

@testset "training-only random sampler is exactly resumable" begin
    rows = Int[2, 5, 11, 19]
    sampler = Scratch.RandomTrainingSampler(rows, UInt64(77))
    first = zeros(Int, 8)
    second = zeros(Int, 8)
    Scratch.next_batch!(first, sampler)
    @test all(row -> row in rows, first)
    replay = Scratch.RandomTrainingSampler(rows, UInt64(77))
    replay_first = zeros(Int, 8)
    Scratch.next_batch!(replay_first, replay)
    @test replay_first == first
    Scratch.next_batch!(second, sampler)
    replay_second = zeros(Int, 8)
    Scratch.next_batch!(replay_second, replay)
    @test replay_second == second
    @test sampler.draws == UInt64(16)
    @test_throws ErrorException Scratch.RandomTrainingSampler([1, 1], UInt64(1))
end

@testset "ranking and DDF execution metrics" begin
    batch = Ranking.Batch(2, 3)
    batch.counts .= Int16[3, 2]
    batch.valid_count = 5
    batch.raw[1, 1:3] .= Float32[0, 2, 1]
    batch.targets.teacher_q[1:3, 1] .= Float32[3, 3, 0]
    batch.raw[1, 4:5] .= Float32[-1, 1]
    batch.targets.teacher_q[1:2, 2] .= Float32[0, 2]
    loss = Ranking.SupervisedLoss(
        2.0f0, 1.3f0, 1.0f0, 0.3f0,
        0.1f0, 0.1f0, 0.1f0, 0.1f0, 0.1f0,
        0.025f0, 0.025f0, 0.025f0, 0.025f0,
        5,
    )
    diagnostics = Parallel.ForwardDiagnostics(2, 3, 1, 17, 12, 4, 5, 9)
    metrics = Scratch.batch_metrics(batch, loss, diagnostics)
    @test metrics.composite == 2.0
    @test metrics.excess == 1.0
    @test metrics.listnet_kl ≈ 0.3 atol=1.0e-7
    @test metrics.legacy_top1 == 0.5
    @test metrics.tie_top1 == 1.0
    @test metrics.forest_event_rate == 0.5
    @test metrics.output_event_rate ==
          17 / (5 * Scratch.Output.hard_event_denominator())
    @test metrics.dirty_leaves == 4
    @test metrics.compact_messages == 9
end

@testset "plain atomic DDF checkpoint roundtrip and fail-closed resume" begin
    mktempdir() do directory
        fixture = fixture_checkpoint(directory)
        snapshot = fixture.snapshot
        @test snapshot.magic == "candidate_delta_dendritic_forest_scratch"
        @test snapshot.schema == 3
        @test keys(snapshot.parameters) == (
            :leaf_shared_raw, :program_payload, :forest_internal_raw,
            :forest_child_contact, :output_cell_raw, :output_anchor_weight,
            :output_context_weight, :output_placement_weight,
            :output_cascade_weight, :output_gain, :output_bias,
        )
        @test hasproperty(snapshot.optimizer, :placement_step_by_coordinate)
        @test !hasproperty(snapshot, :cache)
        @test !hasproperty(snapshot, :gradient)
        @test !hasproperty(snapshot, :scheduler)

        Scratch.save_checkpoint_atomic(fixture.options.checkpoint, snapshot)
        loaded = Scratch.load_checkpoint(fixture.options.checkpoint)
        resume_options = Scratch.ScratchOptions(
            updates=10,
            workers=1,
            checkpoint=fixture.options.checkpoint,
            progress=fixture.options.progress,
        )
        restored = Scratch.restore_checkpoint(
            loaded,
            resume_options,
            "fixture-source",
            "fixture-dataset",
            fixture.training_rows,
            fixture.validation_rows,
        )
        @test model_trainables(restored.parameters) ==
              model_trainables(fixture.parameters)
        @test optimizer_arrays(restored.optimizer) ==
              optimizer_arrays(fixture.optimizer)
        step_names = fieldnames(Optimizer.AdamWStepCounters)
        @test Tuple(getfield(restored.optimizer.steps, name) for name in step_names) ==
              Tuple(getfield(fixture.optimizer.steps, name) for name in step_names)
        @test restored.update == 1
        @test restored.consumed_states == 8
        @test restored.consumed_candidates == 43

        # The schedule owns no mutable clock.  A resumed run derives the next
        # update's rate from the restored global update and reproduces the
        # uninterrupted bit pattern exactly.
        uninterrupted_rate = Optimizer.learning_rate_at(
            Scratch._learning_rate_schedule(fixture.options),
            2,
        )
        resumed_rate = Optimizer.learning_rate_at(
            Scratch._learning_rate_schedule(resume_options),
            restored.update + 1,
        )
        @test reinterpret(UInt32, resumed_rate) ==
              reinterpret(UInt32, uninterrupted_rate)

        expected_next = zeros(Int, 8)
        actual_next = zeros(Int, 8)
        Scratch.next_batch!(expected_next, fixture.sampler)
        Scratch.next_batch!(actual_next, restored.sampler)
        @test actual_next == expected_next

        @test_throws ErrorException Scratch.validate_checkpoint!(
            merge(loaded, (; schema=0)), resume_options,
            "fixture-source", "fixture-dataset",
            fixture.training_rows, fixture.validation_rows,
        )
        @test_throws ErrorException Scratch.validate_checkpoint!(
            loaded, resume_options, "wrong-source", "fixture-dataset",
            fixture.training_rows, fixture.validation_rows,
        )
        schedule_drift = Scratch.ScratchOptions(
            updates=10,
            workers=1,
            warmup_updates=999,
            checkpoint=fixture.options.checkpoint,
            progress=fixture.options.progress,
        )
        @test_throws ErrorException Scratch.validate_checkpoint!(
            loaded, schedule_drift, "fixture-source", "fixture-dataset",
            fixture.training_rows, fixture.validation_rows,
        )
        total_drift = Scratch.ScratchOptions(
            updates=10,
            workers=1,
            learning_rate_schedule_updates=99_999,
            checkpoint=fixture.options.checkpoint,
            progress=fixture.options.progress,
        )
        @test_throws ErrorException Scratch.validate_checkpoint!(
            loaded, total_drift, "fixture-source", "fixture-dataset",
            fixture.training_rows, fixture.validation_rows,
        )
        floor_drift = Scratch.ScratchOptions(
            updates=10,
            workers=1,
            min_learning_rate_ratio=0.02f0,
            checkpoint=fixture.options.checkpoint,
            progress=fixture.options.progress,
        )
        @test_throws ErrorException Scratch.validate_checkpoint!(
            loaded, floor_drift, "fixture-source", "fixture-dataset",
            fixture.training_rows, fixture.validation_rows,
        )
        malformed_parameters = merge(
            loaded.parameters,
            (; output_cascade_weight=fill(
                NaN32,
                size(loaded.parameters.output_cascade_weight),
            )),
        )
        @test_throws ErrorException Scratch.restore_checkpoint(
            merge(loaded, (; parameters=malformed_parameters)),
            resume_options, "fixture-source", "fixture-dataset",
            fixture.training_rows, fixture.validation_rows,
        )
        malformed_second = merge(
            loaded.optimizer.second,
            (; output_placement_weight=fill(
                -1.0f0,
                size(loaded.optimizer.second.output_placement_weight),
            )),
        )
        @test_throws ErrorException Scratch.restore_checkpoint(
            merge(loaded, (; optimizer=merge(
                loaded.optimizer,
                (; second=malformed_second),
            ))),
            resume_options, "fixture-source", "fixture-dataset",
            fixture.training_rows, fixture.validation_rows,
        )

        old_path = joinpath(directory, "old-candidate-delta.jls")
        open(old_path, "w") do io
            serialize(io, (; magic="candidate_delta_dendritic_scratch", schema=2))
        end
        @test_throws ErrorException Scratch.load_checkpoint(old_path)
        wrong_schema = joinpath(directory, "wrong-schema.jls")
        open(wrong_schema, "w") do io
            serialize(io, (; magic=snapshot.magic, schema=2))
        end
        @test_throws ErrorException Scratch.load_checkpoint(wrong_schema)
        truncated = joinpath(directory, "truncated.jls")
        write(truncated, UInt8[0x37, 0x4a, 0x4c])
        @test_throws ErrorException Scratch.load_checkpoint(truncated)

        Scratch.append_json!(
            fixture.options.progress,
            (; kind="training", update=1, excess=0.25),
        )
        @test occursin("\"update\":1", only(readlines(fixture.options.progress)))
    end
end

@testset "source fingerprints and DDF preflight gate" begin
    @test Scratch.rows_sha256([1, 2, 3]) != Scratch.rows_sha256([3, 2, 1])
    @test Scratch.source_fingerprint() == Scratch.source_fingerprint()
    healthy = (
        candidates=2,
        evaluated_nodes=10,
        compact_messages=5,
        forest_event_rate=0.2,
        output_event_rate=0.4,
    )
    @test Scratch._assert_preflight_gate(healthy) === healthy
    @test_throws ErrorException Scratch._assert_preflight_gate(merge(
        healthy,
        (; forest_event_rate=0.0),
    ))
    @test_throws ErrorException Scratch._assert_preflight_gate(merge(
        healthy,
        (; compact_messages=0),
    ))
end
