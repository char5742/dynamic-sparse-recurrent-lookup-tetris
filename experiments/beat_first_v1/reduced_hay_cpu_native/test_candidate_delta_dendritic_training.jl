using Test

module CandidateDeltaDendriticTrainingTestHarness
for file in (
    "TetrisRankingBatch.jl",
    "ActiveApicalCell.jl",
    "CandidateDeltaInput.jl",
    "DendriticProgramBank.jl",
    "CompactDendriticNode.jl",
    "DendriticDeltaForestTopology.jl",
    "DendriticDeltaForest.jl",
    "DendriticForestOutput.jl",
    "CandidateDeltaDendriticGraph.jl",
    "CandidateDeltaDendriticTraining.jl",
)
    include(joinpath(@__DIR__, file))
end
end

const H = CandidateDeltaDendriticTrainingTestHarness
const Ranking = H.TetrisRankingBatch
const Model = H.CandidateDeltaDendriticGraph
const Training = H.CandidateDeltaDendriticTraining
const Bank = H.DendriticProgramBank

function fixture_dataset()
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
    teacher_q = Float32[
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

function dense_program_gradient(gradient, bank)
    dense = zeros(Float32, size(bank.payload))
    @inbounds for slot in 1:Bank.active_gradient_count(gradient)
        row = Int(Bank.active_gradient_row(gradient, slot))
        dense[:, row] .= @view gradient.values[:, slot]
    end
    return dense
end

function reverse_candidate_order!(trainer, batch, dataset)
    Training.clear_batch_gradient!(trainer)
    state = trainer.state
    worker = trainer.worker
    parameters = trainer.parameters
    cache = trainer.cache
    width = batch.width
    for state_slot in 1:batch.state_batch
        row = batch.rows[state_slot]
        Model.prepare_state!(state, worker, parameters, cache, dataset, row)
        offset = (state_slot - 1) * width
        for candidate in Int(batch.counts[state_slot]):-1:1
            flat = offset + candidate
            Model.prepare_candidate!(
                worker, state, parameters, cache, dataset, row, candidate,
            )
            Model.pullback_candidate!(
                trainer.gradient,
                @view(batch.raw[:, flat]),
                @view(batch.raw_gradient[:, flat]),
                worker,
                state,
                parameters,
                cache,
            )
        end
        Model.finish_state_pullback!(
            trainer.gradient, worker, state, parameters, cache,
        )
    end
    return trainer.gradient
end

@testset "real-ranking exact trainer bypasses packed rails" begin
    dataset = fixture_dataset()
    parameters = Model.initialize_model()
    trainer = Training.ExactBatchTrainer(parameters, 2, 4)
    batch = Ranking.Batch(2, 4)
    batch.rows .= (1, 2)

    fill!(batch.rails, NaN32)
    Training.forward_batch!(trainer, batch, dataset)
    raw_with_nan_rails = copy(batch.raw)
    @test all(isnan, batch.rails)
    @test batch.valid_count == 7
    @test batch.valid_flats[1:7] == Int32[1, 2, 3, 5, 6, 7, 8]

    fill!(batch.rails, 17.0f0)
    Training.forward_batch!(trainer, batch, dataset)
    @test batch.raw == raw_with_nan_rails
    @test all(==(17.0f0), batch.rails)

    loss = Training.loss_and_raw_gradient!(trainer, batch)
    @test isfinite(loss.composite_loss)
    @test loss.valid_candidates == 7
    @test all(channel -> any(
        abs(batch.raw_gradient[channel, flat]) > 1.0f-8
        for flat in batch.valid_flats[1:batch.valid_count]
    ), 1:Ranking.OUTPUT_DIM)
end

@testset "grouped reverse is candidate-order invariant" begin
    dataset = fixture_dataset()
    parameters = Model.initialize_model()
    forward_order = Training.ExactBatchTrainer(parameters, 2, 4)
    reverse_order = Training.ExactBatchTrainer(parameters, 2, 4)
    batch_forward = Ranking.Batch(2, 4)
    batch_reverse = Ranking.Batch(2, 4)
    batch_forward.rows .= (1, 2)
    batch_reverse.rows .= (1, 2)

    Training.forward_loss_backward!(forward_order, batch_forward, dataset)
    Training.forward_batch!(reverse_order, batch_reverse, dataset)
    Training.loss_and_raw_gradient!(reverse_order, batch_reverse)
    reverse_candidate_order!(reverse_order, batch_reverse, dataset)

    first = forward_order.gradient
    second = reverse_order.gradient
    @test first.leaf_shared_raw ≈ second.leaf_shared_raw rtol=2.0f-5 atol=2.0f-5
    @test first.forest.internal_raw ≈
          second.forest.internal_raw rtol=2.0f-5 atol=2.0f-5
    @test first.forest.child_contact ≈
          second.forest.child_contact rtol=2.0f-5 atol=2.0f-5
    for field in (
        :cell_raw, :anchor_weight, :context_weight, :placement_weight,
        :cascade_weight, :gain, :bias,
    )
        @test getfield(first.output, field) ≈
              getfield(second.output, field) rtol=2.0f-5 atol=2.0f-5
    end
    @test isapprox(
        dense_program_gradient(first.program, parameters.program_bank),
        dense_program_gradient(second.program, parameters.program_bank);
        rtol=3.0f-5,
        atol=3.0f-5,
    )
end

@testset "all 22 supervised channels reach exact model parameters" begin
    dataset = fixture_dataset()
    parameters = Model.initialize_model()
    trainer = Training.ExactBatchTrainer(parameters, 2, 4)
    batch = Ranking.Batch(2, 4)
    batch.rows .= (1, 2)
    Training.forward_loss_backward!(trainer, batch, dataset)
    analytic = copy(trainer.gradient.output.bias)
    finite = similar(analytic)
    epsilon = 2.0f-3
    for channel in 1:Ranking.OUTPUT_DIM
        original = parameters.output.bias[channel]
        parameters.output.bias[channel] = original + epsilon
        Training.forward_batch!(trainer, batch, dataset)
        plus = Training.loss_and_raw_gradient!(trainer, batch).composite_loss
        parameters.output.bias[channel] = original - epsilon
        Training.forward_batch!(trainer, batch, dataset)
        minus = Training.loss_and_raw_gradient!(trainer, batch).composite_loss
        parameters.output.bias[channel] = original
        finite[channel] = (plus - minus) / (2.0f0 * epsilon)
    end
    @test all(isfinite, analytic)
    @test all(!iszero, analytic)
    @test analytic ≈ finite rtol=1.5f-2 atol=1.5f-3
end
