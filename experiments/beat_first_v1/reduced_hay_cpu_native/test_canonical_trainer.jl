using Test

module CanonicalTrainerTestHarness
include(joinpath(@__DIR__, "ReducedHayCPU.jl"))
end

const Root = CanonicalTrainerTestHarness.ReducedHayCPU
const Ranking = Root.TetrisRankingBatch
const Arena = Root.ReducedHayCPUNativeArena

function synthetic_batch()
    batch = Ranking.Batch(Arena.STATE_BATCH, Arena.CANDIDATE_WIDTH)
    ordinal = 0
    for state in 1:batch.state_batch
        batch.rows[state] = state
        batch.counts[state] = 2
        batch.targets.teacher_q[1, state] = -0.5f0
        batch.targets.teacher_q[2, state] = 0.5f0
        batch.targets.teacher_z[1, state] = -1.0f0
        batch.targets.teacher_z[2, state] = 1.0f0
        batch.targets.top1[state] = 2
        batch.targets.top2[state] = 1
        batch.targets.margin[state] = 1.0f0
        for candidate in 1:2
            flat = candidate + (state - 1) * batch.width
            ordinal += 1
            batch.valid_flats[ordinal] = flat
            batch.rails[candidate:2:end, flat] .= 1.0f0
        end
    end
    batch.valid_count = ordinal
    return batch
end

function build_trainer(batch, workers)
    model, staging, prepared = Root.build_model(0x606)
    staging.cell_raw[
        Root.ActiveApicalCell.P_SOMA_THRESHOLD_GAP,
        :,
        :,
    ] .= 8.0f0
    staging.cell_raw[
        Root.ActiveApicalCell.P_SOMA_THRESHOLD_GAP,
        1,
        :,
    ] .= -8.0f0
    Root.publish!(prepared, staging)
    config = Root.LocalLearningConfig(
        recurrent_interval=2,
        recurrent_start_update=0,
        recurrent_ramp_updates=1,
        subthreshold_interval=1,
        warmup_updates=10_000,
        maximum_rewires=0,
    )
    return Root.CanonicalTrainer(
        model,
        staging,
        prepared,
        batch;
        config,
        workers,
    )
end

function run_two_phases!(trainer, batch)
    workspace = trainer.runtime
    return Root.CanonicalLocalLearner.with_local_credit_team(workspace) do scheduler
        Root.CanonicalLocalLearner._train_update!(
            trainer, batch, workspace, scheduler,
        )
        Root.CanonicalLocalLearner._train_update!(
            trainer, batch, workspace, scheduler,
        )
    end
end

@testset "canonical serial and barrierless update equivalence" begin
    Base.Threads.nthreads(:interactive) == 0 || error(
        "run with --threads=N,0",
    )
    serial_batch = synthetic_batch()
    parallel_batch = synthetic_batch()
    serial = build_trainer(serial_batch, 1)
    parallel = build_trainer(parallel_batch, min(4, Base.Threads.nthreads()))
    q_cells = Root.Architecture.Q_OUTPUT_CELL_COUNT
    serial_numeric_cells_before = copy(view(
        serial.staging.output_cell_raw,
        :,
        1:q_cells,
    ))
    parallel_numeric_cells_before = copy(view(
        parallel.staging.output_cell_raw,
        :,
        1:q_cells,
    ))
    serial_q_bias_before = copy(serial.staging.output_q_basal_bias_raw)
    parallel_q_bias_before = copy(parallel.staging.output_q_basal_bias_raw)
    serial_q_edge_before = copy(serial.staging.output_edge_raw)
    parallel_q_edge_before = copy(parallel.staging.output_edge_raw)
    # The canonical optimizer is block-coordinate: output-only first, then a
    # recurrent-only step.  Two updates exercise both phases without allowing
    # either parameter group to move in the same optimizer call.
    serial_result = run_two_phases!(serial, serial_batch)
    parallel_result = run_two_phases!(parallel, parallel_batch)
    @test isapprox(serial_result[1].composite_loss,
                   parallel_result[1].composite_loss; rtol=0, atol=0)
    @test serial_result[2] == parallel_result[2]
    @test serial_batch.raw == parallel_batch.raw
    for field in fieldnames(Root.ReducedHayCPUNativeModel.Parameters)
        @test isapprox(
            getfield(serial.staging, field),
            getfield(parallel.staging, field);
            rtol=2.0f-5,
            atol=2.0f-6,
        )
    end
    @test Root.mechanism_counts(serial) == Root.mechanism_counts(parallel)
    @test Root.mechanism_counts(serial).decolle_signal_nonzero > 0
    @test Root.mechanism_counts(serial).q_eprop_updates > 0
    @test Root.mechanism_counts(serial).q_cell_updates > 0
    @test Root.mechanism_counts(serial).subthreshold_updates > 0
    @test Root.mechanism_counts(serial).nonspiking_updates > 0
    @test view(serial.staging.output_cell_raw, :, 1:q_cells) !=
          serial_numeric_cells_before
    @test view(parallel.staging.output_cell_raw, :, 1:q_cells) !=
          parallel_numeric_cells_before
    @test any(!iszero, view(
        serial.optimizer.first.output_cell_raw,
        :,
        1:q_cells,
    ))
    @test any(!iszero, view(
        serial.optimizer.second.output_cell_raw,
        :,
        1:q_cells,
    ))
    for output in 2:q_cells
        @test view(serial.staging.output_cell_raw, :, output) ==
            view(serial.staging.output_cell_raw, :, 1)
        @test view(serial.optimizer.first.output_cell_raw, :, output) ==
            view(serial.optimizer.first.output_cell_raw, :, 1)
        @test view(serial.optimizer.second.output_cell_raw, :, output) ==
            view(serial.optimizer.second.output_cell_raw, :, 1)
    end
    @test serial.staging.output_q_basal_bias_raw != serial_q_bias_before
    @test parallel.staging.output_q_basal_bias_raw != parallel_q_bias_before
    @test serial.staging.output_edge_raw != serial_q_edge_before
    @test parallel.staging.output_edge_raw != parallel_q_edge_before
    @test any(!iszero, serial.optimizer.first.output_q_basal_bias_raw)
    @test any(!iszero, serial.optimizer.second.output_q_basal_bias_raw)
    @test any(!iszero, serial.optimizer.first.output_q_edge_raw)
    @test any(!iszero, serial.optimizer.second.output_q_edge_raw)
    @test serial.optimizer.first.output_q_basal_bias_raw ==
        parallel.optimizer.first.output_q_basal_bias_raw
    @test serial.optimizer.second.output_q_basal_bias_raw ==
        parallel.optimizer.second.output_q_basal_bias_raw
    @test serial.optimizer.first.output_q_edge_raw ==
        parallel.optimizer.first.output_q_edge_raw
    @test serial.optimizer.second.output_q_edge_raw ==
        parallel.optimizer.second.output_q_edge_raw
end
