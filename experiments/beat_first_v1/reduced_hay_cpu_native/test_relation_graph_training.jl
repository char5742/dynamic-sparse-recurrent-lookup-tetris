using Test

module RelationGraphTrainingTestHarness
for file in (
    "TetrisRankingBatch.jl",
    "ActiveApicalCell.jl",
    "CandidateDeltaInput.jl",
    "DendriticProgramBank.jl",
    "SpatialProgramPackets.jl",
    "DendriticRelationTopology.jl",
    "DendriticMotifTopology.jl",
    "TypedDendriticAfferents.jl",
    "HighDimensionalCellPacket.jl",
    "TypedRelationCellBank.jl",
    "TypedRelationContext.jl",
    "TypedOutputCellBank.jl",
    "StructuredMotifReadout.jl",
    "CandidateDeltaRelationGraph.jl",
    "RelationGraphOptimizer.jl",
    "RelationGraphTraining.jl",
)
    include(joinpath(@__DIR__, file))
end
end

const H = RelationGraphTrainingTestHarness
const Ranking = H.TetrisRankingBatch
const Model = H.CandidateDeltaRelationGraph
const Optimizer = H.RelationGraphOptimizer
const Training = H.RelationGraphTraining

function fixture_dataset(; teacher_scale::Float32=1.0f0)
    states = 2
    width = 4
    boards = zeros(UInt8, 24, 10, 1, states)
    boards[24, 1, 1, 1] = 0x01
    boards[24, 7, 1, 1] = 0x01
    boards[23, 3, 1, 2] = 0x01
    boards[24, 3, 1, 2] = 0x01
    boards[24, 9, 1, 2] = 0x01
    placements = zeros(UInt8, 24, 10, 1, width, states)
    for state in 1:states, candidate in 1:width
        column = mod1(2candidate + state, 10)
        placements[22 - mod(candidate + state, 3), column, 1, candidate, state] =
            0x01
        placements[23 - mod(candidate, 2), mod1(column + 1, 10), 1,
                   candidate, state] = 0x01
    end
    queues = zeros(UInt8, 7, 6, states)
    for state in 1:states, token in 1:6
        queues[mod1(state + 2token, 7), token, state] = 0x01
    end
    teacher_q = teacher_scale .* Float32[
        -0.8  1.2
         0.3 -0.4
         1.1  0.6
         0.0 -1.0
    ]
    return Ranking.validate_dataset((;
        boards,
        placements,
        queues,
        teacher_q,
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
        line_clear=Int8[
            0 1
            1 0
            0 2
            0 0
        ],
        max_height=Int8[
            3 6
            4 5
            2 7
            0 4
        ],
        holes=Int16[
            0 2
            1 1
            0 3
            0 1
        ],
        cavities=Int16[
            0 1
            2 0
            1 2
            0 3
        ],
        ren=reshape(Float32[2, 5], 1, :),
        back_to_back=reshape(Float32[0, 1], 1, :),
        tspin=Float32[
            0 1
            1 0
            0 0
            0 1
        ],
    ), width)
end

@testset "canonical trainer owns an input-only forward boundary" begin
    dataset = fixture_dataset()
    parameters = Model.initialize_model()
    trainer = Training.RelationGraphTrainer(
        parameters,
        dataset,
        2,
        4;
        optimizer_config=Optimizer.OptimizerConfig(
            learning_rate=5.0f-4,
            clip_norm=2.0f0,
        ),
    )
    @test fieldnames(typeof(trainer.forward_inputs)) ==
          (:boards, :placements, :queues, :ren, :back_to_back, :tspin)
    @test !hasproperty(trainer.forward_inputs, :teacher_q)
    @test !hasproperty(trainer.forward_inputs, :targets)

    @test_throws ArgumentError Training.RelationGraphTrainer(
        parameters,
        dataset,
        2,
        4;
        active_program_capacity=1,
    )

    @test_throws DimensionMismatch Training.train_update!(
        trainer,
        Ranking.Batch(1, 4),
    )
    @test_throws DimensionMismatch Training.train_update!(
        trainer,
        Ranking.Batch(2, 3),
    )
end


@testset "teacher and stale cotangent cannot influence forward output" begin
    original = fixture_dataset(teacher_scale=1.0f0)
    inverted = fixture_dataset(teacher_scale=-3.0f0)
    parameters = Model.initialize_model()
    frozen_optimizer = Optimizer.OptimizerConfig(
        learning_rate=0.0f0,
        clip_norm=2.0f0,
    )
    first = Training.RelationGraphTrainer(
        parameters,
        original,
        2,
        4;
        optimizer_config=frozen_optimizer,
    )
    first_batch = Ranking.Batch(2, 4)
    first_batch.rows .= (1, 2)
    fill!(first_batch.raw_gradient, NaN32)
    Training.train_update!(first, first_batch)
    raw_without_teacher_change = copy(first_batch.raw)

    # The first update had zero learning rate, so parameters are bit-identical.
    second = Training.RelationGraphTrainer(
        parameters,
        inverted,
        2,
        4;
        optimizer_config=frozen_optimizer,
    )
    second_batch = Ranking.Batch(2, 4)
    second_batch.rows .= (1, 2)
    fill!(second_batch.raw_gradient, Inf32)
    Training.train_update!(second, second_batch)
    @test second_batch.raw == raw_without_teacher_change
    @test second_batch.raw_gradient != first_batch.raw_gradient
end

@testset "one API performs grouped exact learning and reports work" begin
    dataset = fixture_dataset()
    parameters = Model.initialize_model()
    trainer = Training.RelationGraphTrainer(
        parameters,
        dataset,
        2,
        4;
        optimizer_config=Optimizer.OptimizerConfig(
            learning_rate=5.0f-4,
            clip_norm=2.0f0,
        ),
    )
    batch = Ranking.Batch(2, 4)
    batch.rows .= (1, 2)
    fill!(batch.rails, NaN32)
    old_bias = copy(parameters.output.bias)

    metrics = Training.train_update!(trainer, batch)
    @test all(isnan, batch.rails)
    @test isfinite(metrics.loss.composite_loss)
    @test isfinite(metrics.excess_loss)
    @test 0.0f0 <= metrics.tie_top1 <= 1.0f0
    @test metrics.loss.valid_candidates == 7
    @test metrics.changed_positions > 0
    @test metrics.affected_positions >= metrics.changed_positions
    @test metrics.affected_relations > 0
    @test metrics.affected_motifs > 0
    @test metrics.base_contact_visits > 0
    @test metrics.candidate_contact_visits > 0
    @test metrics.base_relation_events >= 0
    @test metrics.candidate_relation_events >= 0
    @test metrics.base_motif_events >= 0
    @test metrics.candidate_motif_events >= 0
    @test metrics.base_output_events >= 0
    @test metrics.candidate_output_events >= 0
    @test isfinite(metrics.gradient_norm)
    @test metrics.gradient_norm > 0.0
    @test 0.0f0 < metrics.clip_scale <= 1.0f0
    @test metrics.active_program_rows > 0
    @test trainer.optimizer_state.steps.total == 1
    @test parameters.output.bias != old_bias
    @test all(channel -> any(
        abs(batch.raw_gradient[channel, flat]) > 1.0f-8
        for flat in batch.valid_flats[1:batch.valid_count]
    ), 1:Ranking.OUTPUT_DIM)

    parameters_identity = trainer.parameters
    optimizer_identity = trainer.optimizer_state
    optimizer_steps = trainer.optimizer_state.steps.total
    Training.set_learning_rate!(trainer, 5.0f-5)
    @test trainer.parameters === parameters_identity
    @test trainer.optimizer_state === optimizer_identity
    @test trainer.optimizer_state.steps.total == optimizer_steps
    @test trainer.optimizer_config.learning_rate == 5.0f-5

    # The complete update, including metrics construction, remains fixed arena.
    Training.train_update!(trainer, batch)
    allocated = @allocated Training.train_update!(trainer, batch)
    @test allocated == 0
    @test trainer.optimizer_state.steps.total == 3
end
