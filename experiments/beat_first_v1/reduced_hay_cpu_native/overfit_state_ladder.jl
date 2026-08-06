using Printf
using Statistics

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ReducedHayCPU.jl"))
include(joinpath(@__DIR__, "ExperimentData.jl"))

using .BeatFirstTrainingCore
using .ReducedHayCPU
using .ReducedHayCPUExperimentData

const Ranking = ReducedHayCPU.TetrisRankingBatch
const Arena = ReducedHayCPU.ReducedHayCPUNativeArena
const LocalLearner = ReducedHayCPU.CanonicalLocalLearner
const LADDER_SIZES = (1, 2, 4, 8, 16, 32, 64)

function stable_hard_pass(history)
    length(history) >= 5 || return false
    tail = history[(end - 4):end]
    all(record -> record.tie_top1 == 1.0f0, tail) || return false
    tail[end].excess < tail[1].excess || return false
    maximum(record.excess for record in tail) <= 1.10 * tail[1].excess || return false
    return true
end

function report_hard_q_bits!(trainer, batch, worker)
    q_cells = ReducedHayCPU.Architecture.Q_OUTPUT_CELL_COUNT
    bits = ReducedHayCPU.Architecture.NUMERIC_OPERAND_BITS
    exact_registers = 0
    correct_bits = 0
    target_zeros = zeros(Int, bits)
    target_ones = zeros(Int, bits)
    observed_zeros = zeros(Int, bits)
    observed_ones = zeros(Int, bits)
    bit_group_correct = zeros(Int, 5)
    bit_group_total = zeros(Int, 5)
    @inbounds for state in 1:batch.state_batch
        count = Int(batch.counts[state])
        for candidate in 1:count
            flat = Ranking.flat_index(candidate, state, batch.width)
            Arena.replay_candidate!(
                trainer.arena,
                batch.raw,
                worker,
                flat,
                trainer.model,
                trainer.prepared,
                batch.rails;
                event_floor=0.0f0,
                spike_smoothing=0.0f0,
            )
            output_trajectory = worker.buffers.output_trajectory
            observed_word = ReducedHayCPU.OutputCellBank.q_code_word(
                output_trajectory,
                trainer.model.numeric_core,
            )
            teacher = batch.targets.teacher_q[candidate, state]
            target_word = ReducedHayCPU.OutputCellBank.q_code_from_value(
                teacher,
            )
            mismatch = 0
            for bit in 0:(bits - 1)
                target_one = Bool((target_word >> bit) & UInt32(1))
                target_ones[bit + 1] += target_one
                target_zeros[bit + 1] += !target_one
                observed_one = Bool((observed_word >> bit) & UInt32(1))
                observed_ones[bit + 1] += observed_one
                observed_zeros[bit + 1] += !observed_one
                correct = observed_one == target_one
                correct_bits += correct
                mismatch += !correct
                group = bit >= 23 ? 1 : bit >= 19 ? 2 :
                    bit >= 15 ? 3 : bit >= 7 ? 4 : 5
                bit_group_total[group] += 1
                bit_group_correct[group] += correct
            end
            exact_registers += iszero(mismatch)
            @printf(
                "hard_q_register row=%d candidate=%d teacher=%.6f raw=%.6f word=%08x target=%08x mismatches=%d\n",
                batch.rows[state], candidate, teacher, batch.raw[1, flat],
                observed_word, target_word, mismatch,
            )
        end
    end
    @printf(
        "hard_q_register_summary candidates=%d exact_registers=%d bit_accuracy=%.6f sign_exponent_accuracy=%.6f mantissa_19_22_accuracy=%.6f mantissa_15_18_accuracy=%.6f mantissa_7_14_accuracy=%.6f mantissa_0_6_accuracy=%.6f recurrent_rate_mean=%.6f output_q_rate_mean=%.6f output_q_rate_min=%.6f output_q_rate_max=%.6f\n",
        batch.valid_count,
        exact_registers,
        correct_bits / Float64(bits * batch.valid_count),
        (bit_group_correct ./ bit_group_total)...,
        mean(trainer.plasticity.recurrent_rate),
        mean(@view trainer.plasticity.output_rate[1:q_cells]),
        minimum(@view trainer.plasticity.output_rate[1:q_cells]),
        maximum(@view trainer.plasticity.output_rate[1:q_cells]),
    )
    basal_bias_raw = trainer.staging.output_q_basal_bias_raw
    basal_bias_group_mean = zeros(Float64, 5)
    basal_bias_group_count = zeros(Int, 5)
    for bit in 0:31
        group = bit >= 23 ? 1 : bit >= 19 ? 2 :
            bit >= 15 ? 3 : bit >= 7 ? 4 : 5
        output = ReducedHayCPU.OutputCellBank.q_bit_output(bit)
        basal_bias_group_mean[group] += basal_bias_raw[output]
        basal_bias_group_count[group] += 1
    end
    basal_bias_group_mean ./= basal_bias_group_count
    println(
        "hard_q_bit_counts target_zeros=", join(target_zeros, ','),
        " target_ones=", join(target_ones, ','),
        " observed_zeros=", join(observed_zeros, ','),
        " observed_ones=", join(observed_ones, ','),
        " basal_bias_raw_group_mean=", join(basal_bias_group_mean, ','),
    )
    return exact_registers
end

function run_size!(source, dataset, state_count::Int, max_updates::Int, workers::Int)
    state_count in LADDER_SIZES || error("state count must be one of $LADDER_SIZES")
    state_count <= Arena.STATE_BATCH || error(
        "state count $state_count requires rotating multi-batch training; " *
        "the canonical 8-state gate must pass before that path is enabled",
    )
    rows = fixed_panel_rows(source, :train, state_count)
    batch = Ranking.Batch(Arena.STATE_BATCH, Arena.CANDIDATE_WIDTH)
    fill_state_batch!(batch, rows)
    pack_scratch = Ranking.PackScratch()
    Ranking.pack_batch!(batch, dataset, pack_scratch)
    evaluation_worker = Arena.ArenaWorker()

    config = ReducedHayCPU.LocalLearningConfig(
        # Establish the hard numeric readout before changing its recurrent
        # input representation. Recurrent learning remains independently
        # clocked and starts only after this canonical capacity gate.
        recurrent_start_update=4_096,
        recurrent_ramp_updates=512,
        # Keep the discovery rate through 3k updates, then consolidate without
        # changing the objective or optimizer family.
        learning_rate_decay_start=3_000,
        learning_rate_decay_multiplier=0.5,
        warmup_updates=4_096,
        homeostasis_interval=512,
        maximum_recurrent_adjustments=2,
        maximum_output_adjustments=2,
        structure_interval=512,
        maximum_rewires=1,
        recurrent_homeostasis_until=max_updates,
        output_homeostasis_until=max_updates,
        structure_until=max_updates,
    )
    # Capacity steps must start from the same parameter distribution; varying
    # the model seed with state count confounds memorization with initialization.
    model, staging, prepared = ReducedHayCPU.build_model(0x48415939)
    trainer = ReducedHayCPU.CanonicalTrainer(
        model,
        staging,
        prepared,
        batch;
        config,
        workers,
    )
    println("canonical_local_learning_config ", ReducedHayCPU.config_summary(config))
    println("canonical_local_learning_fingerprint=", ReducedHayCPU.config_fingerprint(config))
    println(
        "ladder_start states=", state_count,
        " rows=", join(rows, ','),
        " candidates=", batch.valid_count,
        " cycles=", ReducedHayCPU.Architecture.CYCLES,
        " cells=", ReducedHayCPU.Architecture.TOTAL_CELLS,
        " basal=", ReducedHayCPU.ActiveApicalCell.N_BASAL,
        " route_free=true hard_forward=true",
    )

    history = NamedTuple[]
    report_interval = max(25, min(100, max_updates ÷ 20))
    initial, top1, tie_top1 = evaluate_batch!(trainer, batch, evaluation_worker)
    initial_excess = initial.composite_loss - initial.teacher_entropy
    push!(history, (; update=0, excess=initial_excess, top1, tie_top1))
    @printf(
        "ladder_progress states=%d update=0 excess=%.6f top1=%.6f tie_top1=%.6f\n",
        state_count, initial_excess, top1, tie_top1,
    )

    workspace = trainer.runtime
    started = time()
    LocalLearner.with_local_credit_team(workspace) do scheduler
        while trainer.updates < max_updates
            LocalLearner._train_update!(
                trainer,
                batch,
                workspace,
                scheduler,
            )
            if trainer.updates % report_interval == 0 ||
               trainer.updates == max_updates
                loss, current_top1, current_tie_top1 = evaluate_batch!(
                    trainer,
                    batch,
                    evaluation_worker,
                )
                excess = loss.composite_loss - loss.teacher_entropy
                push!(history, (;
                    update=trainer.updates,
                    excess,
                    top1=current_top1,
                    tie_top1=current_tie_top1,
                ))
                @printf(
                    "ladder_progress states=%d update=%d excess=%.6f top1=%.6f tie_top1=%.6f updates_per_s=%.3f counts=%s\n",
                    state_count,
                    trainer.updates,
                    excess,
                    current_top1,
                    current_tie_top1,
                    trainer.updates / max(time() - started, eps()),
                    string(ReducedHayCPU.mechanism_counts(trainer)),
                )
                @printf(
                    "local_phase_seconds forward=%.6f loss=%.6f clear=%.6f credit=%.6f reduce=%.6f optimizer=%.6f\n",
                    workspace.timing...,
                )
                flush(stdout)
            end
        end
    end

    report_hard_q_bits!(trainer, batch, evaluation_worker)
    counts = LocalLearner.assert_due_mechanisms!(trainer)
    passed = stable_hard_pass(history)
    final = history[end]
    println(
        "ladder_result states=", state_count,
        " passed=", passed,
        " final_excess=", final.excess,
        " final_top1=", final.top1,
        " final_tie_top1=", final.tie_top1,
        " counts=", counts,
    )
    passed || error(
        "hard state ladder failed sustained loss/top-1 criteria at $state_count states",
    )
    return trainer, history
end

function main()
    state_count = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
    max_updates = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5_000
    workers = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : min(20, Threads.nthreads())
    dataset_path = length(ARGS) >= 4 ? abspath(ARGS[4]) :
        ReducedHayCPUExperimentData.DEFAULT_DATASET
    max_updates >= 1 || error("max updates must be positive")
    workers >= 1 || error("workers must be positive")
    source, dataset = load_width80_dataset(dataset_path)
    run_size!(source, dataset, state_count, max_updates, workers)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
