using Dates
using JSON3
using LinearAlgebra
using Lux
using Random
using SHA

include(joinpath(@__DIR__, "PinnedV5ScratchRunnerCLI.jl"))
using .PinnedV5ScratchRunnerCLI

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropPinnedV5CanonicalFinal.jl",
))
include(joinpath(@__DIR__, "..", "training", "core.jl"))

const PinnedV5Production =
    Main.HD_SWSNN_TWINPROP_PINNED_V5_CANONICAL_FINAL
const PinnedV5Arena = PinnedV5Production.Training
const PinnedV5Model = Main.PaperModelCanonical

if !isdefined(Main, :PaperArenaTraining)
    Core.eval(
        Main,
        :(const PaperArenaTraining = Main.PaperArenaTrainingFinalProduction),
    )
elseif Main.PaperArenaTraining !== PinnedV5Arena
    error("a non-FinalProduction PaperArenaTraining is already loaded")
end
include(joinpath(
    @__DIR__,
    "HDSWSNNTwinPropCheckpointFinal.jl",
))

using .BeatFirstTrainingCore
using .HDSWSNNTwinPropCheckpointFinal

const PINNED_RUNNER_SCHEMA =
    "hd-swsnn-twinprop-pinned-v5-checkpoint-lineage-v1"
const PINNED_MODEL_SEED = UInt64(0x48445357534e4e4d)
const PINNED_SAMPLER_SEED = UInt64(0x48445357534e4e53)
const PINNED_ROUTING_SEED = UInt64(0x48445357534e4e52)
const PINNED_WIDTH = 208
const PINNED_OPTIMIZER_GROUPS = Set((
    :input_conductance,
    :recurrent_conductance,
    :workspace_conductance,
    :query_weight,
    :workspace_key,
    :workspace_decay_logit,
    :head_weight,
    :head_bias,
    :output_weight,
    :output_bias,
))

function _pinned_file_sha256(path)
    isfile(path) || error("file is absent: $path")
    return bytes2hex(SHA.sha256(read(path)))
end

function _pinned_dataset_manifest_sha256(path)
    manifest = joinpath(path, "manifest.json")
    isfile(manifest) ||
        error("Tetris dataset manifest is absent: $manifest")
    return _pinned_file_sha256(manifest)
end

function _pinned_source_revision()
    try
        return strip(readchomp(`git rev-parse HEAD`))
    catch
        return "unknown"
    end
end

function _pinned_write_json(path, value)
    open(path, "w") do io
        JSON3.pretty(io, value)
        println(io)
    end
    return path
end

function _pinned_sidecar_path(checkpoint_path, suffix)
    stem, _ = splitext(checkpoint_path)
    return stem * suffix
end

function _pinned_periodic_path(checkpoint_path, update)
    stem, extension = splitext(checkpoint_path)
    isempty(extension) && (extension = ".jld2")
    return stem * ".update_" *
        lpad(string(update), 9, '0') * extension
end

function _pinned_lineage_id(
    options,
    artifact,
    dataset_manifest_sha256,
)
    fields = (
        PINNED_RUNNER_SCHEMA,
        MODEL_FAMILY,
        artifact.teacher_hash,
        artifact.digital_twin_hash,
        artifact.distilled_artifact_hash,
        artifact.internal_parameter_sha256,
        artifact.cell_mechanism_sha256,
        artifact.morphology_sha256,
        string(artifact.dt_ms),
        dataset_manifest_sha256,
        string(PINNED_MODEL_SEED),
        string(PINNED_SAMPLER_SEED),
        string(PINNED_ROUTING_SEED),
        string(PINNED_WIDTH),
        string(options.require_production),
        string(options.learning_rate),
        string(options.weight_decay),
        string(options.location_interval),
    )
    return bytes2hex(SHA.sha256(codeunits(join(fields, '\n'))))
end

function _pinned_validate_paths!(options)
    for (label, path, predicate) in (
        ("sealed release", options.sealed_release, isfile),
        ("teacher manifest", options.teacher_manifest, isfile),
        ("teacher shards", options.teacher_shards, isdir),
        ("distilled cell", options.distilled_cell, isfile),
        ("Tetris dataset", options.dataset, isdir),
    )
        predicate(path) || error("$label is absent: $path")
    end
    options.checkpoint_in === nothing ||
        isfile(options.checkpoint_in) ||
        error("checkpoint input is absent: $(options.checkpoint_in)")
    actual = _pinned_file_sha256(options.distilled_cell)
    actual == options.distilled_sha256 ||
        error("external distilled artifact SHA-256 differs")
    for path in (
        options.checkpoint_out,
        _pinned_sidecar_path(options.checkpoint_out, ".trace.tsv"),
        _pinned_sidecar_path(options.checkpoint_out, ".manifest.tsv"),
        _pinned_sidecar_path(options.checkpoint_out, ".result.json"),
    )
        ispath(path) && error("output already exists: $path")
    end
    options.workers <= Threads.nthreads(:default) ||
        error(
            "workers exceed default Julia threads: " *
            "$(options.workers) > $(Threads.nthreads(:default))",
        )
    return options
end

function _pinned_run_config(
    options,
    artifact,
    dataset_manifest_sha256,
    lineage_id,
    start_update,
    input_checkpoint_sha256,
    model,
    aux,
)
    return (;
        runner_schema=PINNED_RUNNER_SCHEMA,
        lineage_id,
        model_family=MODEL_FAMILY,
        architecture="HD-SWSNN-TwinProp-PinnedV5",
        start_mode=options.scratch ? "scratch" : "checkpoint",
        start_update,
        target_updates=options.updates,
        checkpoint_in=options.checkpoint_in,
        checkpoint_in_sha256=input_checkpoint_sha256,
        state_batch=1,
        width=PINNED_WIDTH,
        workers=options.workers,
        cpuset_mode=String(options.cpuset_mode),
        stochastic_routing=options.stochastic_routing,
        require_production=options.require_production,
        release_scale_mode=options.require_production ?
            "production" : "development",
        dataset=options.dataset,
        dataset_manifest_sha256,
        model_seed=string(PINNED_MODEL_SEED),
        sampler_seed=string(PINNED_SAMPLER_SEED),
        routing_seed=string(PINNED_ROUTING_SEED),
        learning_rate=options.learning_rate,
        weight_decay=options.weight_decay,
        location_interval=options.location_interval,
        cell_mode="distilled-frozen",
        frozen_internal=true,
        teacher_hash=artifact.teacher_hash,
        digital_twin_hash=artifact.digital_twin_hash,
        distilled_artifact_hash=
            artifact.distilled_artifact_hash,
        internal_parameter_sha256=
            artifact.internal_parameter_sha256,
        cell_mechanism_sha256=
            artifact.cell_mechanism_sha256,
        morphology_sha256=artifact.morphology_sha256,
        dt_ms=artifact.dt_ms,
        substeps_per_cycle=model.substeps_per_cycle,
        cycles=model.cycles,
        decision_window_ms=
            artifact.dt_ms *
            model.substeps_per_cycle *
            model.cycles,
        ablation_mode="full",
        official_segment_count=642,
        location_index_type="UInt16",
        semantic_state_scale="normalized_unit_interval",
        location_mapping_sha256=aux.location_mapping_sha256,
        final_executor="PaperExecutorFinal",
        source_revision=_pinned_source_revision(),
        julia_version=string(VERSION),
        julia_threads=Threads.nthreads(:default),
        blas_threads=BLAS.get_num_threads(),
    )
end

function _pinned_assert_input_lineage!(
    payload,
    lineage_id,
    target_updates,
)
    hasproperty(payload.run_config, :runner_schema) &&
        payload.run_config.runner_schema == PINNED_RUNNER_SCHEMA ||
        error("checkpoint was not created by the pinned-V5 lineage runner")
    hasproperty(payload.run_config, :lineage_id) &&
        payload.run_config.lineage_id == lineage_id ||
        error("checkpoint lineage differs from current immutable inputs")
    start_update = Int(payload.update)
    start_update in (64, 1_000) ||
        (1_000 < start_update < 10_000) ||
        error("checkpoint is not on the 64 -> 1k -> 10k lineage")
    start_update < target_updates ||
        error(
            "checkpoint update $start_update is not before " *
            "target $target_updates",
        )
    target_updates == 1_000 && start_update != 64 &&
        error("the 1k stage must continue the 64-update checkpoint")
    target_updates == 64 &&
        error("the 64-update stage must start with --scratch")
    return start_update
end

function _pinned_save_checkpoint!(
    path,
    trainer,
    sampler,
    run_config,
    manifest_io,
)
    ispath(path) && error("checkpoint output already exists: $path")
    PinnedV5Arena.paper_checkpoint_integrity!(trainer)
    record = save_checkpoint(
        path,
        trainer,
        sampler,
        run_config;
        update=trainer.optimizer.step,
    )
    println(
        manifest_io,
        join((record.update, record.sha256, record.path), '\t'),
    )
    flush(manifest_io)
    return record
end

function pinned_v5_main(arguments=ARGS)
    options = parse_pinned_v5_options(arguments)
    if options.help
        print(pinned_v5_usage())
        return nothing
    end
    _pinned_validate_paths!(options)
    Threads.nthreads(:interactive) == 0 ||
        error("launch with --threads=N,0")
    BLAS.set_num_threads(1)

    bundle = PinnedV5Production.load_pinned_v5_production_bundle(
        options.sealed_release,
        options.teacher_manifest,
        options.teacher_shards,
        options.distilled_cell;
        expected_distilled_artifact_sha256=
            options.distilled_sha256,
        require_production=options.require_production,
        scratch_root=options.scratch_root,
    )
    PinnedV5Production.assert_pinned_v5_bundle_unchanged!(
        bundle,
    )
    artifact = cell_artifact_metadata(options.distilled_cell)
    artifact.distilled_artifact_hash ==
        bundle.expected_distilled_artifact_sha256 ||
        error("checkpoint artifact digest differs from pinned bundle")
    artifact.internal_parameter_sha256 ==
        bundle.distilled_parameter_sha256 ||
        error("cell parameter digest differs from pinned bundle")

    dataset = load_teacher_dataset(
        options.dataset;
        max_candidates=PINNED_WIDTH,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    maximum(dataset.action_counts) <= PINNED_WIDTH ||
        error("teacher candidates exceed width $PINNED_WIDTH")
    rows = Int.(findall(==(:train), dataset.predefined_split))
    isempty(rows) && error("Tetris dataset has no train split")
    dataset_manifest_sha256 =
        _pinned_dataset_manifest_sha256(options.dataset)

    model = PinnedV5Model.build_paper_model(:paper_scaled_v1)
    parameters, _ = Lux.setup(
        Xoshiro(PINNED_MODEL_SEED),
        model,
    )
    builder_options = (;
        state_batch=1,
        width=PINNED_WIDTH,
        learning_rate=options.learning_rate,
        weight_decay=options.weight_decay,
        location_interval=options.location_interval,
    )
    trainer = if options.require_production
        PinnedV5Production.build_production_trainer(
            bundle,
            model,
            parameters;
            builder_options...,
        )
    else
        PinnedV5Production.build_development_trainer(
            bundle,
            model,
            parameters;
            development_scale_chain=true,
            builder_options...,
        )
    end
    Set(keys(trainer.parameters)) == PINNED_OPTIMIZER_GROUPS ||
        error("optimizer parameter groups differ from pinned contract")
    Set(keys(trainer.optimizer.first_moment)) ==
        PINNED_OPTIMIZER_GROUPS ||
        error("frozen internal arrays leaked into optimizer moments")
    aux = PinnedV5Arena.register_paper_trainer_aux!(trainer)
    aux isa PinnedV5Arena.PaperReleaseAux ||
        error("pinned builder did not install PaperReleaseAux")
    eltype(aux.input_location) === UInt16 ||
        error("input locations are not UInt16")
    aux.location_catalog == UInt16.(1:642) ||
        error("arena does not expose official locations 1:642")
    PinnedV5Arena.paper_preflight_integrity!(trainer)
    initial_parameter_hash =
        PinnedV5Arena.paper_internal_parameter_sha256(trainer)
    initial_artifact_hash =
        PinnedV5Arena.paper_internal_sha256(trainer)
    initial_parameter_hash ==
        bundle.distilled_parameter_sha256 ||
        error("loaded frozen parameter hash differs")
    initial_artifact_hash ==
        bundle.expected_distilled_artifact_sha256 ||
        error("loaded frozen artifact hash differs")
    PinnedV5Arena.paper_internal_max_delta(trainer) == 0.0f0 ||
        error("frozen internal cell changed before training")

    lineage_id = _pinned_lineage_id(
        options,
        artifact,
        dataset_manifest_sha256,
    )
    payload = options.checkpoint_in === nothing ?
        nothing : load_checkpoint(options.checkpoint_in)
    start_update = options.scratch ? 0 :
        _pinned_assert_input_lineage!(
            payload,
            lineage_id,
            options.updates,
        )
    input_checkpoint_sha256 =
        options.checkpoint_in === nothing ?
        nothing : checkpoint_sha256(options.checkpoint_in)
    run_config = _pinned_run_config(
        options,
        artifact,
        dataset_manifest_sha256,
        lineage_id,
        start_update,
        input_checkpoint_sha256,
        model,
        aux,
    )
    sampler = if options.scratch
        EpochSampler(rows, Xoshiro(PINNED_SAMPLER_SEED))
    else
        restore_checkpoint!(
            trainer,
            payload,
            rows;
            current_run_config=run_config,
        )
    end
    trainer.optimizer.step == start_update ||
        error("restored optimizer step differs from checkpoint")
    PinnedV5Arena.paper_checkpoint_integrity!(trainer)

    executor = PinnedV5Arena.PaperExecutorFinal(
        trainer,
        dataset;
        active_workers=options.workers,
        stochastic_routing=options.stochastic_routing,
        routing_seed=PINNED_ROUTING_SEED,
        cpuset_mode=options.cpuset_mode,
    )
    all(
        worker -> worker.runtime isa
            PinnedV5Arena.ReleaseCellRuntime,
        executor.workers,
    ) || error("final executor contains a non-release runtime")

    mkpath(dirname(options.checkpoint_out))
    trace_path =
        _pinned_sidecar_path(options.checkpoint_out, ".trace.tsv")
    manifest_path =
        _pinned_sidecar_path(options.checkpoint_out, ".manifest.tsv")
    result_path =
        _pinned_sidecar_path(options.checkpoint_out, ".result.json")
    _pinned_write_json(
        _pinned_sidecar_path(options.checkpoint_out, ".config.json"),
        run_config,
    )

    first_loss = NaN
    final_loss = NaN
    last_checkpoint = nothing
    wall_started = time_ns()
    open(trace_path, "w") do trace_io
        println(
            trace_io,
            "update\tcomposite_loss\tlistnet_loss\t" *
            "soma_spike_rate\tnmda_current_mean\t" *
            "calcium_event_rate\trouting_entropy\t" *
            "gradient_norm\tstates_per_second\twall_seconds\t" *
            "allocation_bytes\tgc_seconds\tlocation_moves",
        )
        open(manifest_path, "w") do manifest_io
            println(manifest_io, "update\tsha256\tpath")
            PinnedV5Arena.run_with_paper_team!(executor) do running
                for update in (start_update + 1):options.updates
                    PinnedV5Arena.Point.fill_next_rows!(
                        PinnedV5Arena.paper_training_arena(
                            trainer,
                        ).rows,
                        sampler,
                    )
                    PinnedV5Arena.paper_arena_update!(running)
                    trainer.optimizer.step == update ||
                        error("optimizer update drift")
                    loss = trainer.last_loss
                    metrics = trainer.metrics
                    observed = Float64(loss.composite_loss)
                    isnan(first_loss) && (first_loss = observed)
                    final_loss = observed
                    println(
                        trace_io,
                        join((
                            update,
                            loss.composite_loss,
                            loss.listnet_loss,
                            metrics.firing_rate,
                            metrics.nmda_current_mean,
                            metrics.calcium_event_rate,
                            metrics.routing_entropy,
                            metrics.gradient_norm,
                            metrics.states_per_second,
                            metrics.wall_seconds,
                            metrics.allocation_bytes,
                            metrics.gc_seconds,
                            metrics.location_moves,
                        ), '\t'),
                    )
                    if update % options.log_interval == 0 ||
                       update == options.updates
                        flush(trace_io)
                        println(
                            "update=$update " *
                            "loss=$(loss.composite_loss) " *
                            "states_s=$(metrics.states_per_second) " *
                            "spike=$(metrics.firing_rate) " *
                            "route_H=$(metrics.routing_entropy)",
                        )
                        flush(stdout)
                    end
                    if update < options.updates &&
                       update % options.checkpoint_interval == 0
                        last_checkpoint = _pinned_save_checkpoint!(
                            _pinned_periodic_path(
                                options.checkpoint_out,
                                update,
                            ),
                            trainer,
                            sampler,
                            run_config,
                            manifest_io,
                        )
                    end
                end
            end
            last_checkpoint = _pinned_save_checkpoint!(
                options.checkpoint_out,
                trainer,
                sampler,
                run_config,
                manifest_io,
            )
        end
    end

    final_parameter_hash =
        PinnedV5Arena.paper_end_run_integrity!(trainer)
    final_parameter_hash == initial_parameter_hash ||
        error("frozen internal parameter hash changed")
    PinnedV5Arena.paper_internal_sha256(trainer) ==
        initial_artifact_hash ||
        error("frozen internal artifact hash changed")
    PinnedV5Arena.paper_internal_max_delta(trainer) == 0.0f0 ||
        error("frozen internal arrays changed")
    PinnedV5Production.assert_pinned_v5_bundle_unchanged!(
        bundle,
    )
    wall_seconds = (time_ns() - wall_started) * 1.0e-9
    completed_updates = options.updates - start_update
    result = (;
        schema="hd-swsnn-twinprop-pinned-v5-segment-result-v1",
        completed=true,
        lineage_id,
        start_update,
        updates=options.updates,
        segment_updates=completed_updates,
        first_loss,
        final_loss,
        segment_wall_seconds=wall_seconds,
        segment_updates_per_second=
            completed_updates / max(wall_seconds, eps(Float64)),
        checkpoint_in=options.checkpoint_in,
        checkpoint_out=last_checkpoint,
        frozen_internal_audit=(;
            initial_artifact_sha256=initial_artifact_hash,
            final_artifact_sha256=
                PinnedV5Arena.paper_internal_sha256(trainer),
            initial_parameter_sha256=initial_parameter_hash,
            final_parameter_sha256=final_parameter_hash,
            internal_max_delta=
                PinnedV5Arena.paper_internal_max_delta(trainer),
            frozen_internal_outside_optimizer=true,
            passed=true,
        ),
        run_config,
        completed_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
    )
    _pinned_write_json(result_path, result)
    println(
        "completed start=$start_update target=$(options.updates) " *
        "lineage=$lineage_id checkpoint=$(options.checkpoint_out)",
    )
    return result
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) &&
    pinned_v5_main()
