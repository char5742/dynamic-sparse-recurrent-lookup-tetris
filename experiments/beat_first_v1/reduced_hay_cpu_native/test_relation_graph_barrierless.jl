using Test

module RelationGraphBarrierlessTestHarness
for file in (
    "BarrierlessScheduler.jl",
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
    "RelationGraphBarrierless.jl",
)
    include(joinpath(@__DIR__, file))
end
end

const H = RelationGraphBarrierlessTestHarness
const SchedulerCore = H.BarrierlessScheduler
const Ranking = H.TetrisRankingBatch
const Model = H.CandidateDeltaRelationGraph
const Optimizer = H.RelationGraphOptimizer
const Serial = H.RelationGraphTraining
const Parallel = H.RelationGraphBarrierless
const Bank = H.DendriticProgramBank
const Afferents = H.TypedDendriticAfferents

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

function parameter_arrays(parameters::Model.ModelParameters)
    return (
        parameters.program_bank.payload,
        parameters.leaf_relation.raw_conductance,
        parameters.relation.cell_raw,
        parameters.relation_motif.raw_conductance,
        parameters.motif.cell_raw,
        parameters.context.common_relation.raw_conductance,
        parameters.context.common_output.raw_conductance,
        parameters.context.aux_relation.raw_conductance,
        parameters.placement_relation.raw_conductance,
        parameters.motif_readout.source_gain_raw,
        parameters.output.cell_raw,
        parameters.output.readout_weight,
        parameters.output.bias,
    )
end

function dense_gradient_arrays(gradient::Model.ModelGradient)
    return (
        gradient.leaf_relation,
        gradient.relation.cell_raw,
        gradient.relation_motif,
        gradient.motif.cell_raw,
        gradient.context.common_relation_raw,
        gradient.context.common_output_raw,
        gradient.context.aux_relation_raw,
        gradient.placement_relation,
        gradient.motif_readout.source_gain_raw,
        gradient.output.cell_raw,
        gradient.output.readout_weight,
        gradient.output.bias,
    )
end

function active_program_snapshot(gradient::Model.ModelGradient)
    count = Bank.active_gradient_count(gradient.program)
    rows = Vector{Int32}(undef, count)
    values = Matrix{Float32}(undef, Bank.PAYLOAD_WIDTH, count)
    @inbounds for slot in 1:count
        rows[slot] = Bank.active_gradient_row(gradient.program, slot)
        for lane in 1:Bank.PAYLOAD_WIDTH
            values[lane, slot] = gradient.program.values[lane, slot]
        end
    end
    return rows, values
end

function assert_loss_equal(left::Ranking.SupervisedLoss, right::Ranking.SupervisedLoss)
    @test left.valid_candidates == right.valid_candidates
    for field in fieldnames(Ranking.SupervisedLoss)
        field == :valid_candidates && continue
        @test getfield(left, field) == getfield(right, field)
    end
end

function assert_allclose(left, right; rtol=2.0f-5, atol=2.0f-6)
    @test size(left) == size(right)
    @test isapprox(left, right; rtol, atol)
end

@inline raw_conductance(physical::Float32) =
    physical + log(-expm1(-physical))

function afferent_groups(trainer)
    parameters = trainer.parameters
    cache = trainer.cache
    return (
        (cache.leaf_relation, parameters.leaf_relation, 0.1f0),
        (cache.relation_motif, parameters.relation_motif, 0.1f0),
        (
            cache.common_relation,
            parameters.context.common_relation,
            0.1f0,
        ),
        (cache.common_output, parameters.context.common_output, 0.1f0),
        (cache.aux_relation, parameters.context.aux_relation, 0.05f0),
        (
            cache.placement_relation,
            parameters.placement_relation,
            0.25f0,
        ),
    )
end

function project_serial_reference!(trainer, config)
    parameters = trainer.parameters
    cache = trainer.cache
    for (multiplier, typed_cache, graph, target) in (
        (
            config.leaf_relation_multiplier,
            cache.leaf_relation,
            parameters.leaf_relation,
            0.1f0,
        ),
        (
            config.relation_motif_multiplier,
            cache.relation_motif,
            parameters.relation_motif,
            0.1f0,
        ),
        (
            config.common_relation_multiplier,
            cache.common_relation,
            parameters.context.common_relation,
            0.1f0,
        ),
        (
            config.common_output_multiplier,
            cache.common_output,
            parameters.context.common_output,
            0.1f0,
        ),
        (
            config.auxiliary_relation_multiplier,
            cache.aux_relation,
            parameters.context.aux_relation,
            0.05f0,
        ),
        (
            config.placement_relation_multiplier,
            cache.placement_relation,
            parameters.placement_relation,
            0.25f0,
        ),
    )
        Parallel._project_afferent_group!(
            multiplier,
            typed_cache,
            graph,
            target,
        )
    end
    Model.refresh_cache!(cache, parameters)
    return trainer
end

function assert_projected_homeostasis(cache, graph, target)
    lower = 0.25f0 * target
    upper = 4.0f0 * target
    tolerance = 16.0f0 * eps(Float32) * upper
    @test length(cache.physical) == Afferents.contact_count(graph)
    @test all(
        value -> lower - tolerance <= value <= upper + tolerance,
        cache.physical,
    )
    means_match = true
    @inbounds for group in 1:(length(cache.group_offsets) - 1)
        first = cache.group_offsets[group]
        stop = cache.group_offsets[group + 1] - 1
        size = stop - first + 1
        size > 1 || continue
        total = 0.0f0
        for position in first:stop
            total += cache.physical[cache.group_slots[position]]
        end
        means_match &= isapprox(
            total / Float32(size),
            target;
            atol=2.0f-6,
        )
    end
    @test means_match
    return nothing
end

@testset "barrierless optimizer boundary projects every active afferent group" begin
    dataset = fixture_dataset()
    trainer = Parallel.BarrierlessRelationGraphTrainer(
        Model.initialize_model(),
        Ranking.Batch(2, 4),
        dataset;
        worker_capacity=1,
    )
    for (_, graph, target) in afferent_groups(trainer)
        fill!(graph.raw_conductance, raw_conductance(8.0f0 * target))
    end
    Model.refresh_cache!(trainer.cache, trainer.parameters)
    Parallel._project_conductance_homeostasis!(trainer)
    for (cache, graph, target) in afferent_groups(trainer)
        assert_projected_homeostasis(cache, graph, target)
    end

    frozen = Parallel.BarrierlessRelationGraphTrainer(
        Model.initialize_model(),
        Ranking.Batch(2, 4),
        dataset;
        optimizer_config=Optimizer.OptimizerConfig(
            leaf_relation_multiplier=0.0,
            relation_motif_multiplier=0.0,
            common_relation_multiplier=0.0,
            common_output_multiplier=0.0,
            auxiliary_relation_multiplier=0.0,
            placement_relation_multiplier=0.0,
        ),
        worker_capacity=1,
    )
    for (_, graph, target) in afferent_groups(frozen)
        fill!(graph.raw_conductance, raw_conductance(8.0f0 * target))
    end
    frozen_before = map(
        group -> copy(group[2].raw_conductance),
        afferent_groups(frozen),
    )
    Parallel._project_conductance_homeostasis!(frozen)
    for (before, (_, graph, _)) in zip(
        frozen_before,
        afferent_groups(frozen),
    )
        @test graph.raw_conductance == before
    end
end

@testset "three-phase exact barrierless matches the serial oracle" begin
    dataset = fixture_dataset()
    base = Model.initialize_model()
    serial_parameters = deepcopy(base)
    parallel_parameters = deepcopy(base)
    config = Optimizer.OptimizerConfig(
        learning_rate=5.0f-4,
        clip_norm=2.0f0,
    )
    serial = Serial.RelationGraphTrainer(
        serial_parameters,
        dataset,
        2,
        4;
        optimizer_config=config,
    )
    serial_batch = Ranking.Batch(2, 4)
    serial_batch.rows .= (1, 2)
    serial_metrics = Serial.train_update!(serial, serial_batch)
    # Normalize the reference at the same post-Adam boundary. This remains
    # idempotent once the serial oracle owns the identical canonical policy.
    project_serial_reference!(serial, config)

    parallel_batch = Ranking.Batch(2, 4)
    parallel_batch.rows .= (1, 2)
    workers = min(4, Base.Threads.nthreads(:default))
    parallel = Parallel.BarrierlessRelationGraphTrainer(
        parallel_parameters,
        parallel_batch,
        dataset;
        optimizer_config=config,
        worker_capacity=workers,
        candidate_chunk_size=1,
    )
    session_sink = Ref{Any}(nothing)
    parallel_metrics = Parallel.run_trainer_team!(
        parallel;
        workers,
        queue_capacity=16,
    ) do session
        session_sink[] = session
        Parallel.train_update!(session)
    end

    assert_loss_equal(serial_metrics.loss, parallel_metrics.loss)
    @test serial_batch.raw == parallel_batch.raw
    @test serial_batch.raw_gradient == parallel_batch.raw_gradient
    @test serial_metrics.tie_top1 == parallel_metrics.tie_top1
    @test serial_metrics.changed_positions == parallel_metrics.changed_positions
    @test serial_metrics.affected_positions == parallel_metrics.affected_positions
    @test serial_metrics.affected_relations == parallel_metrics.affected_relations
    @test serial_metrics.affected_motifs == parallel_metrics.affected_motifs
    @test serial_metrics.base_relation_events == parallel_metrics.base_relation_events
    @test serial_metrics.candidate_relation_events ==
          parallel_metrics.candidate_relation_events
    @test serial_metrics.base_motif_events == parallel_metrics.base_motif_events
    @test serial_metrics.candidate_motif_events ==
          parallel_metrics.candidate_motif_events
    @test serial_metrics.base_output_events == parallel_metrics.base_output_events
    @test serial_metrics.candidate_output_events ==
          parallel_metrics.candidate_output_events
    @test serial_metrics.base_contact_visits == parallel_metrics.base_contact_visits
    @test serial_metrics.candidate_contact_visits ==
          parallel_metrics.candidate_contact_visits
    @test serial_metrics.active_program_rows ==
          parallel_metrics.active_program_rows

    serial_rows, serial_program = active_program_snapshot(serial.gradient)
    parallel_rows, parallel_program = active_program_snapshot(parallel.gradient)
    @test serial_rows == parallel_rows
    assert_allclose(serial_program, parallel_program)
    for (left, right) in zip(
        dense_gradient_arrays(serial.gradient),
        dense_gradient_arrays(parallel.gradient),
    )
        assert_allclose(left, right)
    end
    for (left, right) in zip(
        parameter_arrays(serial_parameters),
        parameter_arrays(parallel_parameters),
    )
        assert_allclose(left, right)
    end

    report = Parallel.scheduler_report(session_sink[])
    # The persistent team enters every native Julia thread; `workers` limits
    # dispatch ownership, not scheduler teardown participation.
    @test report.entered_threads == Base.Threads.nthreads(:default)
    @test report.exited_threads == Base.Threads.nthreads(:default)
    @test report.queue_empty
    @test report.queue_closed
    @test report.remaining == 0
    @test report.active_dispatches == 0
    @test report.phase_idle
end

@testset "scheduling order cannot change updates or active program rows" begin
    dataset = fixture_dataset()
    base = Model.initialize_model()
    config = Optimizer.OptimizerConfig(
        learning_rate=3.0f-4,
        clip_norm=2.0f0,
    )
    workers = min(4, Base.Threads.nthreads(:default))

    function run_copy(chunk)
        parameters = deepcopy(base)
        batch = Ranking.Batch(2, 4)
        batch.rows .= (1, 2)
        trainer = Parallel.BarrierlessRelationGraphTrainer(
            parameters,
            batch,
            dataset;
            optimizer_config=config,
            worker_capacity=workers,
            candidate_chunk_size=chunk,
        )
        metrics = Parallel.run_trainer_team!(
            trainer;
            workers,
            queue_capacity=16,
        ) do session
            latest = Parallel.train_update!(session)
            latest = Parallel.train_update!(session)
            latest = Parallel.train_update!(session)
            latest
        end
        rows, values = active_program_snapshot(trainer.gradient)
        return parameters, copy(batch.raw), metrics, rows, values
    end

    first = run_copy(1)
    second = run_copy(2)
    @test first[2] == second[2]
    assert_loss_equal(first[3].loss, second[3].loss)
    @test first[4] == second[4]
    @test first[5] == second[5]
    for (left, right) in zip(parameter_arrays(first[1]), parameter_arrays(second[1]))
        @test left == right
    end
end

@testset "hot update allocation is zero" begin
    dataset = fixture_dataset()
    parameters = Model.initialize_model()
    batch = Ranking.Batch(2, 4)
    batch.rows .= (1, 2)
    workers = min(4, Base.Threads.nthreads(:default))
    trainer = Parallel.BarrierlessRelationGraphTrainer(
        parameters,
        batch,
        dataset;
        optimizer_config=Optimizer.OptimizerConfig(
            learning_rate=1.0f-4,
            clip_norm=2.0f0,
        ),
        worker_capacity=workers,
        candidate_chunk_size=1,
    )
    allocated = Ref{Int}(-1)
    Parallel.run_trainer_team!(
        trainer;
        workers,
        queue_capacity=16,
    ) do session
        Parallel.train_update!(session)
        allocated[] = @allocated Parallel.train_update!(session)
        nothing
    end
    @test allocated[] == 0
end

@testset "coordinator failure tears the persistent team down cleanly" begin
    dataset = fixture_dataset()
    parameters = Model.initialize_model()
    batch = Ranking.Batch(2, 4)
    batch.rows .= (dataset.state_count + 1, 2)
    workers = min(4, Base.Threads.nthreads(:default))
    trainer = Parallel.BarrierlessRelationGraphTrainer(
        parameters,
        batch,
        dataset;
        worker_capacity=workers,
        candidate_chunk_size=1,
    )
    failure = try
        Parallel.run_trainer_team!(
            trainer;
            workers,
            queue_capacity=16,
        ) do session
            Parallel.train_update!(session)
        end
        nothing
    catch exception
        exception
    end
    @test failure isa SchedulerCore.SchedulerFailure
    report = failure.report
    @test report.entered_threads == Base.Threads.nthreads(:default)
    @test report.exited_threads == Base.Threads.nthreads(:default)
    @test report.queue_empty
    @test report.queue_closed
    @test report.remaining == 0
    @test report.active_dispatches == 0
    @test report.phase_idle
end
