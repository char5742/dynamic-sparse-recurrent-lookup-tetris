using Test

include(joinpath(@__DIR__, "overfit_candidate_delta_dendritic.jl"))

@testset "DDF hard-overfit entrypoint owns only canonical modules" begin
    source = read(
        joinpath(@__DIR__, "overfit_candidate_delta_dendritic.jl"),
        String,
    )
    for retired in (
        "SharedDendriticFactor",
        "TypedSparseAfferents",
        "ContextAfferents",
        "ContinuousDendriticReadout",
        "SpatialDendriticFactors",
        "DendriticDecisionGraph",
    )
        @test !occursin(retired, source)
    end
    for canonical in (
        "CompactDendriticNode",
        "DendriticDeltaForestTopology",
        "DendriticDeltaForest",
        "DendriticForestOutput",
        "CandidateDeltaDendriticGraph",
        "CandidateDeltaDendriticOptimizer",
        "CandidateDeltaDendriticBarrierless",
    )
        @test occursin(canonical, source)
    end
end

@testset "DDF hard-overfit CLI exposes the eleven optimizer groups" begin
    parsed = parse_options([
        "--states=16",
        "--updates", "25",
        "--seed", "19",
        "--log-every", "5",
        "--workers", "1",
        "--candidate-chunk", "7",
        "--learning-rate", "0.003",
        "--clip-norm", "2",
        "--weight-decay", "0.0002",
        "--cell-weight-decay", "0.00003",
        "--leaf-cell-multiplier", "0.11",
        "--forest-internal-multiplier", "0.12",
        "--forest-contact-multiplier", "0.13",
        "--program-multiplier", "0.14",
        "--output-cell-multiplier", "0.15",
        "--output-anchor-multiplier", "0.16",
        "--output-context-multiplier", "0.17",
        "--output-placement-multiplier", "0.18",
        "--output-cascade-multiplier", "0.19",
        "--output-gain-multiplier", "0.20",
        "--output-bias-multiplier", "0.21",
    ])
    @test parsed.states == 16
    @test parsed.updates == 25
    @test parsed.seed == 19
    @test parsed.log_every == 5
    @test parsed.workers == 1
    @test parsed.candidate_chunk == 7
    @test parsed.learning_rate == 0.003f0
    @test parsed.clip_norm == 2.0f0
    @test parsed.weight_decay == 0.0002f0
    @test parsed.cell_weight_decay == 0.00003f0
    config = optimizer_config(parsed)
    @test config.multipliers.leaf_cell == 0.11f0
    @test config.multipliers.forest_internal == 0.12f0
    @test config.multipliers.forest_contact == 0.13f0
    @test config.multipliers.program == 0.14f0
    @test config.multipliers.output_cell == 0.15f0
    @test config.multipliers.output_anchor == 0.16f0
    @test config.multipliers.output_context == 0.17f0
    @test config.multipliers.output_placement == 0.18f0
    @test config.multipliers.output_cascade == 0.19f0
    @test config.multipliers.output_gain == 0.20f0
    @test config.multipliers.output_bias == 0.21f0
    @test config.weight_decay == 0.0002f0
    @test config.cell_weight_decay == 0.00003f0
    @test_throws ErrorException parse_options(["--states", "1"])
    @test_throws ErrorException parse_options(["--states", "2"])
    @test_throws ErrorException parse_options(["--workers", "0"])
    @test_throws ErrorException parse_options(["--candidate-chunk", "0"])
    @test_throws ErrorException parse_options(["--factor-multiplier", "1"])
end

@testset "training panel selection is seeded and training-only" begin
    source = (predefined_split=Symbol[
        :train, :validation, :train, :train, :test, :train,
    ],)
    first = select_train_rows(source, 3, 7)
    second = select_train_rows(source, 3, 7)
    @test first == second
    @test issorted(first)
    @test length(first) == 3
    @test all(row -> source.predefined_split[row] === :train, first)
    @test_throws ErrorException select_train_rows(source, 5, 7)
end

@testset "legacy and teacher-tie-aware top-1 remain distinct" begin
    batch = Ranking.Batch(2, 4)
    batch.counts .= Int16[3, 4]
    batch.targets.teacher_q[:, 1] .= Float32[1, 1, 0, 0]
    batch.targets.teacher_q[:, 2] .= Float32[0, 2, 1, 2]
    batch.raw[1, 1:4] .= Float32[0, 3, -1, 0]
    batch.raw[1, 5:8] .= Float32[0, 1, 0, 2]
    legacy, tied = top1_metrics(batch)
    @test legacy == 0.0f0
    @test tied == 1.0f0
end

function adapter_fixture_dataset()
    states = 2
    width = 4
    boards = zeros(UInt8, 24, 10, 1, states)
    boards[24, 1, 1, 1] = 0x01
    boards[24, 7, 1, 1] = 0x01
    boards[23, 3, 1, 2] = 0x01
    boards[24, 9, 1, 2] = 0x01
    placements = zeros(UInt8, 24, 10, 1, width, states)
    @inbounds for state in 1:states, candidate in 1:width
        column = mod1(2candidate + state, 10)
        placements[21 - mod(candidate + state, 3), column, 1,
                   candidate, state] = 0x01
        placements[22 - mod(candidate, 2), mod1(column + 1, 10), 1,
                   candidate, state] = 0x01
    end
    queues = zeros(UInt8, 7, 6, states)
    @inbounds for state in 1:states, token in 1:6
        queues[mod1(state + 2token, 7), token, state] = 0x01
    end
    return Ranking.validate_dataset((;
        boards,
        placements,
        queues,
        teacher_q=Float32[
            -0.8  1.2
             0.3 -0.4
             1.1  0.6
             0.0 -1.0
        ],
        action_counts=Int[3, 4],
        selected_actions=Int[3, 1],
        terminal=Bool[false, true],
        candidate_death=Bool[
            false true
            true false
            false false
            false true
        ],
        candidate_death_available=Bool[true, true],
        line_clear=Int8[0 1; 1 0; 0 2; 0 0],
        max_height=Int8[3 6; 4 5; 2 7; 0 4],
        holes=Int16[0 2; 1 1; 0 3; 0 1],
        cavities=Int16[0 1; 2 0; 1 2; 0 3],
        ren=reshape(Float32[2, 5], 1, :),
        back_to_back=reshape(Float32[0, 1], 1, :),
        tspin=Float32[0 1; 1 0; 0 0; 0 1],
    ), width)
end

function compare_adapter_parameters(first, second)
    @test first.leaf_shared_raw ≈ second.leaf_shared_raw rtol=2f-6 atol=2f-6
    @test first.program_bank.payload ≈
        second.program_bank.payload rtol=2f-6 atol=2f-6
    @test first.forest.internal_raw ≈
        second.forest.internal_raw rtol=2f-6 atol=2f-6
    @test first.forest.child_contact ≈
        second.forest.child_contact rtol=2f-6 atol=2f-6
    @test first.output.cell_raw ≈ second.output.cell_raw rtol=2f-6 atol=2f-6
    @test first.output.anchor_weight ≈
        second.output.anchor_weight rtol=2f-6 atol=2f-6
    @test first.output.context_weight ≈
        second.output.context_weight rtol=2f-6 atol=2f-6
    @test first.output.placement_weight ≈
        second.output.placement_weight rtol=2f-6 atol=2f-6
    @test first.output.cascade_weight ≈
        second.output.cascade_weight rtol=2f-6 atol=2f-6
    @test first.output.gain ≈ second.output.gain rtol=2f-6 atol=2f-6
    @test first.output.bias ≈ second.output.bias rtol=2f-6 atol=2f-6
end

@testset "serial and barrierless DDF adapters agree for diagnostics and update" begin
    if Threads.nthreads(:default) < 2 || Threads.nthreads(:interactive) != 0
        @test true
    else
        dataset = adapter_fixture_dataset()
        serial_parameters = Model.initialize_model()
        parallel_parameters = Model.initialize_model()
        serial_batch = Ranking.Batch(2, 4)
        parallel_batch = Ranking.Batch(2, 4)
        serial_batch.rows .= (1, 2)
        parallel_batch.rows .= (1, 2)
        serial_trainer = Training.ExactBatchTrainer(serial_parameters, 2, 4)
        executor = Barrierless.BarrierlessExactExecutor(
            parallel_parameters,
            parallel_batch,
            dataset;
            worker_capacity=2,
            candidate_chunk_size=1,
        )
        serial_optimizer = Optimizer.AdamWState(serial_parameters)
        parallel_optimizer = Optimizer.AdamWState(parallel_parameters)
        config = Optimizer.AdamWConfig(learning_rate=7.5f-4)

        _, _, _, serial_diagnostics, serial_event_rate =
            evaluate!(serial_trainer, serial_batch, dataset)
        parallel_stats, parallel_diagnostics = Barrierless.run_executor_team!(
            executor;
            workers=2,
            queue_capacity=16,
        ) do session
            Barrierless.forward_batch!(session)
            diagnostics = Barrierless.latest_forward_diagnostics(session)
            stats = _training_update!(
                BarrierlessOverfitExecution(session, executor),
                parallel_optimizer,
                parallel_parameters,
                config,
                parallel_batch,
                dataset,
            )
            return stats, diagnostics
        end
        serial_stats = _training_update!(
            SerialOverfitExecution(serial_trainer),
            serial_optimizer,
            serial_parameters,
            config,
            serial_batch,
            dataset,
        )

        @test serial_diagnostics == parallel_diagnostics
        @test serial_diagnostics.evaluated_nodes > 0
        @test serial_diagnostics.compact_messages > 0
        @test serial_diagnostics.dirty_leaves > 0
        @test serial_diagnostics.dirty_ancestors > 0
        @test 0.0f0 <= serial_event_rate <= 1.0f0
        @test serial_stats.gradient_norm ≈
            parallel_stats.gradient_norm rtol=2e-6
        @test serial_stats.clip_scale ≈
            parallel_stats.clip_scale rtol=2f-6
        @test serial_stats.active_program_rows ==
            parallel_stats.active_program_rows
        compare_adapter_parameters(serial_parameters, parallel_parameters)
    end
end
