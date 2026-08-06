using Test

module CandidateDeltaDendriticBarrierlessTestHarness
for file in (
    "TetrisRankingBatch.jl",
    "ActiveApicalCell.jl",
    "CandidateDeltaInput.jl",
    "CompactDendriticNode.jl",
    "DendriticProgramBank.jl",
    "DendriticDeltaForestTopology.jl",
    "DendriticDeltaForest.jl",
    "DendriticForestOutput.jl",
    "CandidateDeltaDendriticGraph.jl",
    "BarrierlessScheduler.jl",
    "CandidateDeltaDendriticBarrierless.jl",
)
    include(joinpath(@__DIR__, file))
end
end

const H = CandidateDeltaDendriticBarrierlessTestHarness
const Ranking = H.TetrisRankingBatch
const Bank = H.DendriticProgramBank
const Topology = H.DendriticDeltaForestTopology
const Output = H.DendriticForestOutput
const Model = H.CandidateDeltaDendriticGraph
const Parallel = H.CandidateDeltaDendriticBarrierless

function fixture_dataset()
    states = 4
    width = 6
    boards = zeros(UInt8, 24, 10, 1, states)
    placements = zeros(UInt8, 24, 10, 1, width, states)
    queues = zeros(UInt8, 7, 6, states)
    @inbounds for state in 1:states
        for column in 1:10
            height = mod(3state + 2column, 5)
            for delta in 0:(height - 1)
                boards[24 - delta, column, 1, state] = 0x01
            end
        end
        for token in 1:6
            queues[mod1(state + 2token, 7), token, state] = 0x01
        end
        for candidate in 1:width
            column = mod1(2candidate + state, 10)
            placements[20 - mod(candidate + state, 3), column, 1,
                       candidate, state] = 0x01
            placements[21 - mod(candidate, 2), mod1(column + 1, 10), 1,
                       candidate, state] = 0x01
        end
    end
    teacher_q = Matrix{Float32}(undef, width, states)
    line_clear = Matrix{Int8}(undef, width, states)
    max_height = Matrix{Int8}(undef, width, states)
    holes = Matrix{Int16}(undef, width, states)
    cavities = Matrix{Int16}(undef, width, states)
    tspin = Matrix{Float32}(undef, width, states)
    candidate_death = Matrix{Bool}(undef, width, states)
    @inbounds for state in 1:states, candidate in 1:width
        teacher_q[candidate, state] =
            0.7f0 * Float32(candidate) - 0.4f0 * Float32(state) +
            0.15f0 * Float32(mod(candidate * state, 3))
        line_clear[candidate, state] = Int8(mod(candidate + state, 3))
        max_height[candidate, state] = Int8(3 + mod(2candidate + state, 8))
        holes[candidate, state] = Int16(mod(candidate + 2state, 5))
        cavities[candidate, state] = Int16(mod(3candidate + state, 4))
        tspin[candidate, state] = Float32(iszero(mod(candidate + state, 5)))
        candidate_death[candidate, state] = iszero(mod(candidate + 2state, 7))
    end
    return Ranking.validate_dataset((;
        boards,
        placements,
        queues,
        teacher_q,
        action_counts=Int[3, 4, 5, 6],
        selected_actions=Int[3, 4, 5, 6],
        terminal=Bool[false, false, true, false],
        candidate_death,
        candidate_death_available=trues(states),
        line_clear,
        max_height,
        holes,
        cavities,
        ren=reshape(Float32[0, 2, 4, 7], 1, :),
        back_to_back=reshape(Float32[0, 1, 0, 1], 1, :),
        tspin,
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

function compare_gradient(first, second, bank)
    @test first.leaf_shared_raw ≈ second.leaf_shared_raw rtol=5.0f-5 atol=5.0f-5
    @test first.forest.internal_raw ≈ second.forest.internal_raw rtol=5.0f-5 atol=5.0f-5
    @test first.forest.child_contact ≈ second.forest.child_contact rtol=5.0f-5 atol=5.0f-5
    @test first.output.cell_raw ≈ second.output.cell_raw rtol=5.0f-5 atol=5.0f-5
    @test first.output.anchor_weight ≈ second.output.anchor_weight rtol=5.0f-5 atol=5.0f-5
    @test first.output.context_weight ≈ second.output.context_weight rtol=5.0f-5 atol=5.0f-5
    @test first.output.placement_weight ≈ second.output.placement_weight rtol=5.0f-5 atol=5.0f-5
    @test first.output.cascade_weight ≈ second.output.cascade_weight rtol=5.0f-5 atol=5.0f-5
    @test first.output.gain ≈ second.output.gain rtol=5.0f-5 atol=5.0f-5
    @test first.output.bias ≈ second.output.bias rtol=5.0f-5 atol=5.0f-5
    @test dense_program_gradient(first.program, bank) ≈
          dense_program_gradient(second.program, bank) rtol=6.0f-5 atol=6.0f-5
end

function gradient_snapshot(gradient, bank)
    return (
        leaf_shared_raw=copy(gradient.leaf_shared_raw),
        program=dense_program_gradient(gradient.program, bank),
        forest_internal_raw=copy(gradient.forest.internal_raw),
        forest_child_contact=copy(gradient.forest.child_contact),
        output_cell_raw=copy(gradient.output.cell_raw),
        output_anchor_weight=copy(gradient.output.anchor_weight),
        output_context_weight=copy(gradient.output.context_weight),
        output_placement_weight=copy(gradient.output.placement_weight),
        output_cascade_weight=copy(gradient.output.cascade_weight),
        output_gain=copy(gradient.output.gain),
        output_bias=copy(gradient.output.bias),
    )
end

function gradient_matches_snapshot(gradient, snapshot, bank)
    return gradient.leaf_shared_raw == snapshot.leaf_shared_raw &&
        dense_program_gradient(gradient.program, bank) == snapshot.program &&
        gradient.forest.internal_raw == snapshot.forest_internal_raw &&
        gradient.forest.child_contact == snapshot.forest_child_contact &&
        gradient.output.cell_raw == snapshot.output_cell_raw &&
        gradient.output.anchor_weight == snapshot.output_anchor_weight &&
        gradient.output.context_weight == snapshot.output_context_weight &&
        gradient.output.placement_weight == snapshot.output_placement_weight &&
        gradient.output.cascade_weight == snapshot.output_cascade_weight &&
        gradient.output.gain == snapshot.output_gain &&
        gradient.output.bias == snapshot.output_bias
end

function serial_forward_loss_backward!(parameters, batch, dataset)
    cache = Model.ModelCache(parameters)
    gradient = Model.ModelGradient(parameters)
    state = Model.ModelState()
    worker = Model.ModelWorker()
    scratch = Ranking.LossScratch(batch.width, batch.state_batch)

    Ranking.prepare_batch_metadata!(batch, dataset)
    Model.refresh_cache!(cache, parameters)
    @inbounds for state_slot in 1:batch.state_batch
        row = batch.rows[state_slot]
        Model.prepare_state!(state, worker, parameters, cache, dataset, row)
        count = Int(batch.counts[state_slot])
        offset = (state_slot - 1) * batch.width
        for candidate in 1:count
            Model.forward_candidate!(
                @view(batch.raw[:, offset + candidate]),
                worker,
                state,
                parameters,
                cache,
                dataset,
                row,
                candidate,
            )
        end
    end
    loss = Ranking.supervised_loss_and_raw_gradient!(batch, scratch)

    Model.clear_gradient!(gradient)
    @inbounds for state_slot in 1:batch.state_batch
        row = batch.rows[state_slot]
        Model.prepare_state!(state, worker, parameters, cache, dataset, row)
        count = Int(batch.counts[state_slot])
        offset = (state_slot - 1) * batch.width
        for candidate in 1:count
            flat = offset + candidate
            Model.prepare_candidate!(
                worker,
                state,
                parameters,
                cache,
                dataset,
                row,
                candidate,
            )
            Model.pullback_candidate!(
                gradient,
                @view(batch.raw[:, flat]),
                @view(batch.raw_gradient[:, flat]),
                worker,
                state,
                parameters,
                cache,
            )
        end
        Model.finish_state_pullback!(
            gradient,
            worker,
            state,
            parameters,
            cache,
        )
    end
    return loss, gradient
end

@testset "DDF serial/barrierless exact equivalence" begin
    @test fieldnames(Model.ModelGradient) ==
          (:leaf_shared_raw, :program, :forest, :output)

    dataset = fixture_dataset()
    parameters = Model.initialize_model()
    serial_batch = Ranking.Batch(4, 6)
    parallel_batch = Ranking.Batch(4, 6)
    serial_batch.rows .= 1:4
    parallel_batch.rows .= 1:4
    serial_loss, serial_gradient = serial_forward_loss_backward!(
        parameters,
        serial_batch,
        dataset,
    )

    executor = Parallel.BarrierlessExactExecutor(
        parameters,
        parallel_batch,
        dataset,
    )
    result = Parallel.run_executor_team!(
        executor;
        workers=min(4, Threads.nthreads(:default)),
        queue_capacity=16,
    ) do session
        loss = Parallel.forward_loss_backward!(session)
        diagnostics = Parallel.latest_forward_diagnostics(session)
        raw_snapshot = copy(parallel_batch.raw)
        gradient_copy = gradient_snapshot(
            executor.gradient,
            parameters.program_bank,
        )

        Parallel.forward_loss_backward!(session, executor.loss_sink)
        hot_allocated = @allocated Parallel.forward_loss_backward!(
            session,
            executor.loss_sink,
        )
        repeated_diagnostics = Parallel.latest_forward_diagnostics(session)

        @test parallel_batch.raw == raw_snapshot
        @test gradient_matches_snapshot(
            executor.gradient,
            gradient_copy,
            parameters.program_bank,
        )
        @test repeated_diagnostics == diagnostics
        return (; loss, diagnostics, hot_allocated)
    end

    diagnostics = result.diagnostics
    @test diagnostics.evaluated_nodes ==
          2 * parallel_batch.state_batch * Topology.NODE_COUNT +
          diagnostics.dirty_leaves + diagnostics.dirty_ancestors
    @test 0 <= diagnostics.output_events <=
          parallel_batch.valid_count * Output.hard_event_denominator()
    @test diagnostics.leaf_events >= 0
    @test diagnostics.internal_events >= 0
    @test diagnostics.root_events >= 0
    @test diagnostics.dirty_leaves > 0
    @test diagnostics.dirty_ancestors > 0
    @test diagnostics.compact_messages > 0

    @test serial_loss.composite_loss == result.loss.composite_loss
    @test serial_batch.raw == parallel_batch.raw
    compare_gradient(
        serial_gradient,
        executor.gradient,
        parameters.program_bank,
    )
    @test result.hot_allocated == 0
end
