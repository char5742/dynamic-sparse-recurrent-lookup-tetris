# Canonical executable entrypoint. The included file supplies only shared CLI,
# trace, checkpoint, and reporting helpers; its `main` guard is false here.
include(joinpath(@__DIR__, "train_hd_swsnn_twinprop.jl"))

function final_usage()
    return """
    Canonical scratch-only HD-SWSNN-TwinProp trainer.

    julia --threads=N,0 train_hd_swsnn_twinprop_final.jl [options]

      --updates N                 target updates (default: 10000)
      --dataset PATH              teacher_v3 dataset directory
      --output-root PATH          parent for a timestamped run
      --output-dir PATH           exact new run directory; must not exist
      --state-batch N             states per update (default: 1)
      --width N                   candidate arena width; must be 80
      --workers N                 barrierless workers
      --learning-rate X           AdamW learning rate (default: 5e-4)
      --weight-decay X            AdamW weight decay (default: 1e-5)
      --location-interval N        anatomical relocation cadence
      --checkpoint-interval N      checkpoint cadence (default: 1000)
      --log-interval N             console report cadence (default: 100)
      --cpuset-mode MODE           none, all, or p_only
      --cell-mode MODE             distilled (default) or detailed
      --cell-artifact PATH         frozen distilled-11 artifact
      --deterministic-routing      disable stochastic training routing
      --help                       show this text

    The production 10k path is always full-mechanism, distilled-frozen.
    Detailed mode is limited to 64 updates. Ablation and resume are separate
    experiment families and are rejected by this entrypoint.
    """
end

function parse_final_options(arguments)
    for argument in arguments
        argument == "--structural-interval" && error(
            "--structural-interval is not a PaperTrainer option",
        )
        argument == "--ablation-mode" && error(
            "the canonical trainer is full-mechanism only; " *
            "use a dedicated ablation runner",
        )
    end
    options = validate_options(parse_options(arguments))
    options.help && return options
    options.ablation_mode == :full ||
        error("the canonical trainer requires full mechanism mode")
    return options
end

function final_main(arguments=ARGS)
    options = parse_final_options(arguments)
    if options.help
        println(final_usage())
        return nothing
    end
    Threads.nthreads(:interactive) == 0 ||
        error("launch with --threads=N,0")
    BLAS.set_num_threads(1)

    artifact_before =
        cell_artifact_metadata(options.cell_artifact)
    artifact_before.dt_ms == 1.0 ||
        error(
            "HD-SWSNN-TwinProp requires dt_ms=1.0; artifact has " *
            string(artifact_before.dt_ms),
        )
    run_dir = reserve_run_directory(options)
    checkpoint_dir = joinpath(run_dir, "checkpoints")
    mkpath(checkpoint_dir)
    trace_path = joinpath(run_dir, "training_trace.tsv")
    manifest_path = joinpath(run_dir, "checkpoint_manifest.tsv")

    dataset = load_teacher_dataset(
        options.dataset;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    rows = training_rows(dataset)
    maximum(dataset.action_counts) <= options.width ||
        error("arena width is too small for teacher candidates")

    model = build_paper_model(PRESET)
    parameters, _ = Lux.setup(Xoshiro(MODEL_SEED), model)
    trainer_cell_mode = options.cell_mode == :distilled ?
        :distilled_frozen : :detailed
    trainer = PaperTrainer(
        model,
        parameters;
        state_batch=options.state_batch,
        width=options.width,
        learning_rate=options.learning_rate,
        weight_decay=options.weight_decay,
        location_interval=options.location_interval,
        cell_mode=trainer_cell_mode,
        cell_artifact=options.cell_artifact,
    )
    cell_mode = options.cell_mode == :distilled ?
        "distilled-frozen" : "detailed-control"
    frozen_internal = options.cell_mode == :distilled
    initial_internal = internal_audit(trainer)
    if frozen_internal
        initial_internal.artifact_sha256 ==
            lowercase(
                artifact_before.distilled_artifact_hash,
            ) || error(
                "loaded cell artifact hash differs from disk artifact",
            )
        initial_internal.parameter_sha256 ==
            lowercase(
                artifact_before.internal_parameter_sha256,
            ) || error(
                "loaded in-memory cell parameters differ from artifact",
            )
    end
    initial_internal.max_delta == 0.0 ||
        error("internal cell is changed before training")

    run_config = (;
        model_family=MODEL_FAMILY,
        architecture="hd-swsnn-twinprop",
        preset=String(PRESET),
        start_mode="scratch",
        target_updates=options.updates,
        state_batch=options.state_batch,
        width=options.width,
        workers=options.workers,
        cpuset_mode=String(options.cpuset_mode),
        stochastic_routing=options.stochastic_routing,
        dataset=options.dataset,
        dataset_manifest_sha256=manifest_sha256(options.dataset),
        model_seed=string(MODEL_SEED),
        sampler_seed=string(SAMPLER_SEED),
        routing_seed=string(ROUTING_SEED),
        learning_rate=options.learning_rate,
        weight_decay=options.weight_decay,
        location_interval=options.location_interval,
        cell_mode,
        cell_artifact=artifact_before.path,
        cell_artifact_schema=artifact_before.artifact_schema,
        teacher_hash=artifact_before.teacher_hash,
        digital_twin_hash=artifact_before.digital_twin_hash,
        distilled_artifact_hash=
            artifact_before.distilled_artifact_hash,
        internal_parameter_sha256=
            artifact_before.internal_parameter_sha256,
        loaded_internal_artifact_sha256=
            initial_internal.artifact_sha256,
        loaded_internal_parameter_sha256=
            initial_internal.parameter_sha256,
        frozen_internal,
        cell_mechanism_sha256=
            artifact_before.cell_mechanism_sha256,
        morphology_sha256=artifact_before.morphology_sha256,
        dt_ms=artifact_before.dt_ms,
        substeps_per_cycle=model.substeps_per_cycle,
        cycles=model.cycles,
        decision_window_ms=
            artifact_before.dt_ms *
            model.substeps_per_cycle *
            model.cycles,
        ablation_mode="full",
        mechanism_scope=(
            "detailed Hay-style teacher to frozen digital twin to " *
            "frozen distilled-11 internal cell; unpublished exact " *
            "TwinProp implementation is not claimed"
        ),
        source_revision=source_revision(),
        julia_version=string(VERSION),
        julia_threads=Threads.nthreads(:default),
        blas_threads=BLAS.get_num_threads(),
    )
    config_path = joinpath(run_dir, "run_config.json")
    open(config_path, "w") do io
        JSON3.pretty(io, run_config)
        println(io)
    end

    sampler = EpochSampler(rows, Xoshiro(SAMPLER_SEED))
    started_at = now(UTC)
    total_wall_started = time_ns()
    last_checkpoint = nothing
    total_allocations = Int128(0)
    total_gc_seconds = 0.0
    initial_loss = NaN
    final_loss = NaN
    executor = PaperExecutor(
        trainer,
        dataset;
        active_workers=options.workers,
        cpuset_mode=options.cpuset_mode,
        stochastic_routing=options.stochastic_routing,
        routing_seed=ROUTING_SEED,
    )

    open(trace_path, "w") do trace_io
        println(trace_io, join(TRACE_COLUMNS, '\t'))
        open(manifest_path, "w") do manifest_io
            println(manifest_io, "update\tsha256\tpath")
            last_checkpoint = save_training_checkpoint!(
                trainer,
                sampler,
                run_config,
                checkpoint_dir,
                manifest_io,
                initial_internal,
                artifact_before,
            )
            window = WindowMetrics()
            run_with_paper_team!(executor) do running
                for update in 1:options.updates
                    arena = paper_training_arena(trainer)
                    PaperArenaTraining.Point.fill_next_rows!(
                        arena.rows,
                        sampler,
                    )
                    paper_arena_update!(running)
                    trainer.optimizer.step == update ||
                        error("optimizer update drift")
                    write_trace_row(trace_io, update, trainer)
                    observed_loss = Float64(
                        _loss(trainer, (:composite_loss, :loss)),
                    )
                    initial_loss = update == 1 ?
                        observed_loss : initial_loss
                    final_loss = observed_loss
                    total_allocations += Int128(
                        _metric(
                            trainer,
                            (:allocation_bytes,),
                            0,
                        ),
                    )
                    total_gc_seconds += Float64(
                        _metric(
                            trainer,
                            (:gc_seconds,),
                            0.0,
                        ),
                    )
                    add!(window, trainer)
                    if update % options.log_interval == 0 ||
                       update == options.updates
                        flush(trace_io)
                        report!(window, update, trainer)
                    end
                    if update % options.checkpoint_interval == 0 ||
                       update == options.updates
                        last_checkpoint =
                            save_training_checkpoint!(
                                trainer,
                                sampler,
                                run_config,
                                checkpoint_dir,
                                manifest_io,
                                initial_internal,
                                artifact_before,
                            )
                    end
                end
            end
        end
    end

    total_wall_seconds =
        (time_ns() - total_wall_started) * 1.0e-9
    last_checkpoint === nothing &&
        error("training ended without a checkpoint")
    artifact_after =
        cell_artifact_metadata(options.cell_artifact)
    artifact_after == artifact_before ||
        error("distilled cell artifact changed during training")
    final_internal = verify_frozen_internal!(
        trainer,
        initial_internal,
        artifact_before,
        cell_mode,
        "completed training",
    )
    deltas = paper_parameter_deltas(trainer)
    result = (;
        schema="hd-swsnn-twinprop-result-final-v1",
        completed=true,
        model_family=MODEL_FAMILY,
        started_at_utc=Dates.format(
            started_at,
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
        completed_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
        run_dir,
        updates=trainer.optimizer.step,
        trained_states=trainer.optimizer.step *
            options.state_batch,
        initial_loss,
        final_loss,
        total_wall_seconds,
        updates_per_second=
            trainer.optimizer.step /
            max(total_wall_seconds, eps(Float64)),
        total_allocation_bytes=string(total_allocations),
        total_gc_seconds,
        final_metrics=final_metrics(trainer),
        parameter_max_deltas=deltas,
        frozen_internal_audit=(;
            cell_mode,
            frozen_internal,
            initial_artifact_sha256=
                initial_internal.artifact_sha256,
            final_artifact_sha256=
                final_internal.artifact_sha256,
            initial_parameter_sha256=
                initial_internal.parameter_sha256,
            final_parameter_sha256=
                final_internal.parameter_sha256,
            distilled_artifact_hash_before=
                artifact_before.distilled_artifact_hash,
            distilled_artifact_hash_after=
                artifact_after.distilled_artifact_hash,
            internal_max_delta=final_internal.max_delta,
            passed=final_internal == initial_internal &&
                final_internal.max_delta == 0.0 &&
                artifact_after == artifact_before,
        ),
        checkpoint=last_checkpoint,
        run_config,
    )
    result.frozen_internal_audit.passed ||
        error("frozen internal audit failed")
    result_path = joinpath(run_dir, "result.json")
    open(result_path, "w") do io
        JSON3.pretty(io, result)
        println(io)
    end
    println(
        "completed updates=$(result.updates) " *
        "initial_loss=$(result.initial_loss) " *
        "final_loss=$(result.final_loss) " *
        "internal_max_delta=" *
        string(result.frozen_internal_audit.internal_max_delta) *
        " result=$(result_path)",
    )
    flush(stdout)
    return result
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && final_main()
