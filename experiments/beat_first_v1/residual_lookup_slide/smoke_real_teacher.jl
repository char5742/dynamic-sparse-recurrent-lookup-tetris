#!/usr/bin/env julia

using Dates
using JSON3
using Random

include(joinpath(@__DIR__, "teacher_training.jl"))

const RLS = ResidualLookupSlideR0TeacherTraining
const Core = RLS.TrainingCore
const Lookup = RLS.Lookup

function assert_exact_scientific_state(left, right)
    left.update == right.update || error("trainer updates differ")
    left.model.banks == right.model.banks || error("lookup bank values differ")
    left.model.head == right.model.head || error("head values differ")
    left.model.bias == right.model.bias || error("bias values differ")
    left.model.alpha_logits == right.model.alpha_logits || error(
        "residual logits differ",
    )
    left.optimizer.step == right.optimizer.step || error("optimizer clocks differ")
    for layer in 1:Lookup.BLOCKS
        a = left.optimizer.bank_states[layer]
        b = right.optimizer.bank_states[layer]
        a.m == b.m || error("bank first moments differ at layer $layer")
        a.v == b.v || error("bank second moments differ at layer $layer")
        a.event_count == b.event_count || error("bank event counts differ")
        a.last_event_step == b.last_event_step || error("bank event clocks differ")
        a.last_log_decay == b.last_log_decay || error("bank decay clocks differ")
        a.global_step == b.global_step || error("bank global steps differ")
        a.global_log_decay == b.global_log_decay || error(
            "bank global decay differs",
        )
    end
    ah = left.optimizer.head_state
    bh = right.optimizer.head_state
    ah.m_head == bh.m_head && ah.v_head == bh.v_head || error(
        "head moments differ",
    )
    ah.m_bias == bh.m_bias && ah.v_bias == bh.v_bias || error(
        "bias moments differ",
    )
    ah.step == bh.step || error("head optimizer steps differ")
    aa = left.optimizer.alpha_state
    ba = right.optimizer.alpha_state
    aa.m == ba.m && aa.v == ba.v || error("alpha moments differ")
    aa.step == ba.step || error("alpha optimizer steps differ")
    left.usage.counts == right.usage.counts || error("route usage differs")
    left.usage.calls == right.usage.calls || error("route usage clocks differ")
    left.block_selected_gradient_events == right.block_selected_gradient_events ||
        error("selected-gradient counters differ")
    left.block_nonzero_selected_gradient_events ==
        right.block_nonzero_selected_gradient_events || error(
            "nonzero-gradient counters differ",
        )
    left.timed_updates == right.timed_updates || error("timed update counts differ")
    left.timed_candidates == right.timed_candidates || error(
        "timed candidate counts differ",
    )
    return true
end

function main()
    dataset_path = abspath(get(ENV, "BEAT_RLS_R0_SMOKE_DATASET", RLS.DEFAULT_DATASET))
    output_root = abspath(get(
        ENV,
        "BEAT_RLS_R0_SMOKE_OUTPUT",
        raw"D:\tetris-paper-plus\runs\beat_first_v1\residual_lookup_slide_smoke",
    ))
    run_id = "real_teacher_smoke_" * Dates.format(now(), "yyyymmddTHHMMSS")
    run_dir = joinpath(output_root, run_id)
    ispath(run_dir) && error("fresh smoke directory already exists: $run_dir")
    mkpath(run_dir)

    dataset = Core.load_teacher_dataset(
        dataset_path;
        max_candidates=Core.MAX_CANDIDATES,
        allow_partial_dataset=false,
    )
    observed_max_candidates = maximum(dataset.action_counts)
    observed_max_candidates <= RLS.LEARNER_WIDTH || error(
        "teacher data exceeds learner width",
    )
    split = RLS.ParentTraining.episode_separated_split(
        dataset;
        seed=RLS.SPLIT_SEED,
        validation_fraction=0.20,
    )
    held_rows = RLS.ParentTraining.fixed_evaluation_subset(
        split.validation_rows,
        RLS.HELD_EVALUATION_STATES,
        RLS.HELD_SEED,
    )
    split_metadata = RLS._split_metadata(split, held_rows)
    trainer = RLS.initialize_trainer()
    config = RLS.default_r0_config(dataset_path, run_id)
    sampler = Core.EpochSampler(split.training_rows, Xoshiro(RLS.SAMPLER_SEED))
    state = RLS.TrainingRunState(
        trainer,
        sampler,
        config,
        split_metadata,
        Any[],
        nothing,
        nothing,
    )
    RLS._validate_run_state(state)
    batch = Core.allocate_host_batch(1; max_candidates=RLS.LEARNER_WIDTH)
    initial = RLS._evaluation_record!(state, dataset, batch, nothing)

    row1 = only(Core.next_batch!(state.sampler, 1))
    Core.pack_batch!(batch, dataset, [row1])
    step1 = RLS.train_step!(trainer, batch; expected_update=1)
    checkpoint_path = joinpath(run_dir, "checkpoint_000000001.jls")
    receipt = RLS.save_training_checkpoint(checkpoint_path, state)
    restored = RLS.restore_training_checkpoint(
        checkpoint_path,
        config,
        split_metadata;
        expected_bytes=receipt.bytes,
        expected_sha256=receipt.sha256,
    )

    row2_left = only(Core.next_batch!(state.sampler, 1))
    row2_right = only(Core.next_batch!(restored.sampler, 1))
    row2_left == row2_right || error("restored sampler selected a different row")
    left_batch = Core.allocate_host_batch(1; max_candidates=RLS.LEARNER_WIDTH)
    right_batch = Core.allocate_host_batch(1; max_candidates=RLS.LEARNER_WIDTH)
    Core.pack_batch!(left_batch, dataset, [row2_left])
    Core.pack_batch!(right_batch, dataset, [row2_right])
    step2_left = RLS.train_step!(state.trainer, left_batch; expected_update=2)
    step2_right = RLS.train_step!(restored.trainer, right_batch; expected_update=2)
    assert_exact_scientific_state(state.trainer, restored.trainer)
    rng_witness_left = rand(state.trainer.model_rng, UInt64, 16)
    rng_witness_right = rand(restored.trainer.model_rng, UInt64, 16)
    rng_witness_left == rng_witness_right || error("restored Xoshiro diverged")

    summary = (
        schema="residual-lookup-slide-r0-real-teacher-smoke-v1",
        status="PASS_NOT_STRENGTH_EVIDENCE",
        run_dir=abspath(run_dir),
        dataset_path,
        dataset_manifest_sha256=config.dataset_manifest_sha256,
        source_fingerprint=config.source_fingerprint,
        julia_version=string(VERSION),
        total_parameters=Lookup.parameter_count(trainer.model),
        active_parameters_per_candidate=Lookup.active_parameter_count(trainer.model),
        observed_max_candidates,
        held_states=length(held_rows),
        initial_held_signal=initial.held_signal,
        first_row=row1,
        second_row=row2_left,
        first_step_loss=step1.loss,
        second_step_loss=step2_left.loss,
        restored_second_step_loss=step2_right.loss,
        checkpoint=(;
            path=abspath(checkpoint_path),
            bytes=receipt.bytes,
            sha256=receipt.sha256,
        ),
        exact_continuation=true,
        protected_game_seed_sets_used=false,
        game_evaluation=false,
    )
    summary_path = joinpath(run_dir, "summary.json")
    open(summary_path, "w") do io
        write(io, JSON3.write(summary))
        write(io, '\n')
    end
    println(JSON3.write(merge(summary, (; summary_path=abspath(summary_path)))))
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
