using Dates
using JSON3
using LinearAlgebra
using Lux
using Random
using SHA

include(joinpath(
    @__DIR__,
    "LoadHDSWSNNTwinPropReleaseProductionFinalV3.jl",
))
include(joinpath(@__DIR__, "..", "training", "core.jl"))

const Production = Main.HDSWSNNTwinPropProduction
const Arena = Production.Training
const PaperModel = Main.PaperModelCanonical

# The final checkpoint module intentionally resolves this Main-level name.
if !isdefined(Main, :PaperArenaTraining)
    Core.eval(
        Main,
        :(const PaperArenaTraining =
            Main.PaperArenaTrainingFinalProduction),
    )
elseif Main.PaperArenaTraining !== Arena
    error("a non-FinalProduction PaperArenaTraining is already loaded")
end
include(joinpath(
    @__DIR__,
    "HDSWSNNTwinPropCheckpointFinal.jl",
))

using .BeatFirstTrainingCore
using .HDSWSNNTwinPropCheckpointFinal

const FINAL_MODEL_SEED = UInt64(0x48445357534e4e4d)
const FINAL_SAMPLER_SEED = UInt64(0x48445357534e4e53)
const FINAL_ROUTING_SEED = UInt64(0x48445357534e4e52)
const FINAL_WIDTH = 208
const FINAL_OPTIMIZER_GROUPS = (
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
)

function final_release_usage()
    return """
    Scratch-only HD-SWSNN-TwinProp FinalProduction trainer.

    julia --project=experiments/beat_first_v1 --threads=N,0 \\
      train_hd_swsnn_release_finalproduction.jl \\
      --stage 64|1k|10k \\
      --teacher-manifest PATH \\
      --frozen-twin PATH \\
      --distilled-cell PATH \\
      --dataset PATH \\
      --output-dir NEW_PATH [options]

      --workers N
      --learning-rate X          default 5e-4
      --weight-decay X           default 1e-5
      --location-interval N      default 128
      --checkpoint-interval N    stage-dependent
      --log-interval N           default 10/50/100
      --cpuset-mode MODE         none, all, or p_only
      --deterministic-routing

    There is no resume mode and existing output directories are rejected.
    The only accepted chain is detailed Hay -> official ELM twin -> semantic
    11-state release-v2 frozen cell -> UInt16/642 arena -> Final MPMC executor.
    """
end

function _final_release_options(arguments)
    values = Dict{String,String}()
    flags = Set{String}()
    value_names = Set((
        "stage",
        "teacher-manifest",
        "frozen-twin",
        "distilled-cell",
        "dataset",
        "output-dir",
        "workers",
        "learning-rate",
        "weight-decay",
        "location-interval",
        "checkpoint-interval",
        "log-interval",
        "cpuset-mode",
    ))
    index = 1
    while index <= length(arguments)
        token = arguments[index]
        startswith(token, "--") ||
            error("unexpected positional argument: $token")
        name = token[3:end]
        name in ("help", "deterministic-routing") &&
            (push!(flags, name); index += 1; continue)
        name in ("resume", "cell-mode", "ablation-mode") &&
            error("$token is forbidden in the scratch FinalProduction driver")
        name in value_names ||
            error("unknown option: $token")
        index < length(arguments) ||
            error("missing value for $token")
        haskey(values, name) &&
            error("option repeated: $token")
        values[name] = arguments[index + 1]
        index += 2
    end
    "help" in flags && return (; help=true)
    for name in (
        "stage",
        "teacher-manifest",
        "frozen-twin",
        "distilled-cell",
        "dataset",
        "output-dir",
    )
        haskey(values, name) || error("--$name is required")
    end
    stage = lowercase(values["stage"])
    updates = stage == "64" ? 64 :
        stage == "1k" ? 1_000 :
        stage == "10k" ? 10_000 :
        error("--stage must be 64, 1k, or 10k")
    default_checkpoint =
        updates == 64 ? 64 :
        updates == 1_000 ? 100 : 1_000
    default_log =
        updates == 64 ? 10 :
        updates == 1_000 ? 50 : 100
    integer(name, default) =
        parse(Int, get(values, name, string(default)))
    real(name, default) =
        parse(Float64, get(values, name, string(default)))
    options = (;
        help=false,
        stage,
        updates,
        teacher_manifest=
            abspath(values["teacher-manifest"]),
        frozen_twin=abspath(values["frozen-twin"]),
        distilled_cell=abspath(values["distilled-cell"]),
        dataset=abspath(values["dataset"]),
        output_dir=abspath(values["output-dir"]),
        workers=integer(
            "workers",
            min(20, Threads.nthreads(:default)),
        ),
        learning_rate=real("learning-rate", 5.0e-4),
        weight_decay=real("weight-decay", 1.0e-5),
        location_interval=integer("location-interval", 128),
        checkpoint_interval=integer(
            "checkpoint-interval",
            default_checkpoint,
        ),
        log_interval=integer("log-interval", default_log),
        cpuset_mode=Symbol(get(values, "cpuset-mode", "none")),
        stochastic_routing=
            !("deterministic-routing" in flags),
    )
    for (label, path, predicate) in (
        ("teacher manifest", options.teacher_manifest, isfile),
        ("frozen twin", options.frozen_twin, isfile),
        ("distilled cell", options.distilled_cell, isfile),
        ("Tetris dataset", options.dataset, isdir),
    )
        predicate(path) || error("$label is absent: $path")
    end
    ispath(options.output_dir) &&
        error("scratch output directory already exists: $(options.output_dir)")
    2 <= options.workers <= Threads.nthreads(:default) ||
        error("workers must be in 2:$(Threads.nthreads(:default))")
    options.learning_rate > 0 ||
        error("learning-rate must be positive")
    options.weight_decay >= 0 ||
        error("weight-decay must be nonnegative")
    options.location_interval > 0 ||
        error("location-interval must be positive")
    options.checkpoint_interval > 0 ||
        error("checkpoint-interval must be positive")
    options.log_interval > 0 ||
        error("log-interval must be positive")
    options.cpuset_mode in (:none, :all, :p_only) ||
        error("cpuset-mode must be none, all, or p_only")
    return options
end

function _final_dataset_manifest_sha256(path)
    manifest = joinpath(path, "manifest.json")
    isfile(manifest) ||
        error("Tetris dataset manifest is absent: $manifest")
    return bytes2hex(SHA.sha256(read(manifest)))
end

function _final_source_revision()
    try
        return strip(readchomp(`git rev-parse HEAD`))
    catch
        return "unknown"
    end
end

function _write_json(path, value)
    open(path, "w") do io
        JSON3.pretty(io, value)
        println(io)
    end
    return path
end

function _release_status(
    path;
    training_started::Bool,
    completed::Bool,
    update::Int,
    stage::String,
)
    return _write_json(path, (;
        schema="hd-swsnn-release-status-v1",
        training_started,
        completed,
        update,
        stage,
        updated_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
    ))
end

function _parameter_tree_finite(parameters)
    return all(
        array -> all(isfinite, array),
        values(parameters),
    )
end

function final_release_main(arguments=ARGS)
    options = _final_release_options(arguments)
    if options.help
        print(final_release_usage())
        return nothing
    end
    Threads.nthreads(:interactive) == 0 ||
        error("launch with --threads=N,0")
    BLAS.set_num_threads(1)

    # Complete every immutable-source and semantic gate before creating a run.
    bundle = Production.load_production_bundle(
        options.teacher_manifest,
        options.frozen_twin,
        options.distilled_cell;
        verify_teacher_shards=true,
    )
    Production.assert_production_bundle_unchanged!(bundle)
    artifact = cell_artifact_metadata(options.distilled_cell)
    artifact.distilled_artifact_hash ==
        bundle.distilled_file_sha256 ||
        error("checkpoint artifact digest differs from production bundle")
    artifact.internal_parameter_sha256 ==
        bundle.distilled_parameter_sha256 ||
        error("cell parameter digest differs from production bundle")
    artifact.digital_twin_hash ==
        bundle.twin_artifact_sha256 ||
        error("cell lineage differs from frozen official twin")

    dataset = load_teacher_dataset(
        options.dataset;
        max_candidates=FINAL_WIDTH,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    maximum(dataset.action_counts) <= FINAL_WIDTH ||
        error("teacher candidates exceed width $FINAL_WIDTH")
    rows = Int.(findall(
        ==(:train),
        dataset.predefined_split,
    ))
    isempty(rows) && error("Tetris dataset has no train split")

    model = PaperModel.build_paper_model(:paper_scaled_v1)
    parameters, _ = Lux.setup(
        Xoshiro(FINAL_MODEL_SEED),
        model,
    )
    trainer = Production.build_production_trainer(
        bundle,
        model,
        parameters;
        state_batch=1,
        width=FINAL_WIDTH,
        learning_rate=options.learning_rate,
        weight_decay=options.weight_decay,
        location_interval=options.location_interval,
    )
    Set(keys(trainer.parameters)) ==
        Set(FINAL_OPTIMIZER_GROUPS) ||
        error("optimizer parameter groups differ from final contract")
    Set(keys(trainer.optimizer.first_moment)) ==
        Set(FINAL_OPTIMIZER_GROUPS) ||
        error("frozen internal arrays leaked into optimizer moments")
    _parameter_tree_finite(trainer.parameters) ||
        error("trainable parameter tree is non-finite")
    aux = Arena.register_paper_trainer_aux!(trainer)
    aux isa Arena.PaperReleaseAux ||
        error("production trainer did not enable release-v2")
    eltype(aux.input_location) === UInt16 ||
        error("input locations are not UInt16")
    aux.location_catalog == UInt16.(1:642) ||
        error("production arena does not expose official locations 1:642")
    Arena.paper_preflight_integrity!(trainer)
    initial_parameter_hash =
        Arena.paper_internal_parameter_sha256(trainer)
    initial_artifact_hash =
        Arena.paper_internal_sha256(trainer)
    initial_parameter_hash ==
        bundle.distilled_parameter_sha256 ||
        error("loaded frozen parameter hash differs")
    initial_artifact_hash ==
        bundle.distilled_file_sha256 ||
        error("loaded frozen artifact hash differs")
    Arena.paper_internal_max_delta(trainer) == 0.0f0 ||
        error("frozen internal cell changed before training")

    mkpath(options.output_dir)
    checkpoint_dir =
        joinpath(options.output_dir, "checkpoints")
    mkpath(checkpoint_dir)
    status_path = joinpath(options.output_dir, "status.json")
    trace_path =
        joinpath(options.output_dir, "training_trace.tsv")
    manifest_path =
        joinpath(options.output_dir, "checkpoint_manifest.tsv")
    _release_status(
        status_path;
        training_started=false,
        completed=false,
        update=0,
        stage=options.stage,
    )

    run_config = (;
        model_family=Production.MODEL_FAMILY,
        architecture="HD-SWSNN-TwinProp-FinalProduction",
        start_mode="scratch",
        stage=options.stage,
        target_updates=options.updates,
        state_batch=1,
        width=FINAL_WIDTH,
        workers=options.workers,
        cpuset_mode=String(options.cpuset_mode),
        stochastic_routing=options.stochastic_routing,
        dataset=options.dataset,
        dataset_manifest_sha256=
            _final_dataset_manifest_sha256(options.dataset),
        model_seed=string(FINAL_MODEL_SEED),
        sampler_seed=string(FINAL_SAMPLER_SEED),
        routing_seed=string(FINAL_ROUTING_SEED),
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
        location_mapping_sha256=
            aux.location_mapping_sha256,
        final_executor="PaperExecutorFinal",
        source_revision=_final_source_revision(),
        julia_version=string(VERSION),
        julia_threads=Threads.nthreads(:default),
        blas_threads=BLAS.get_num_threads(),
    )
    _write_json(
        joinpath(options.output_dir, "run_config.json"),
        run_config,
    )

    sampler = EpochSampler(
        rows,
        Xoshiro(FINAL_SAMPLER_SEED),
    )
    executor = Arena.PaperExecutorFinal(
        trainer,
        dataset;
        active_workers=options.workers,
        stochastic_routing=options.stochastic_routing,
        routing_seed=FINAL_ROUTING_SEED,
        cpuset_mode=options.cpuset_mode,
    )
    all(
        worker -> worker.runtime isa Arena.ReleaseCellRuntime,
        executor.workers,
    ) || error("final executor contains a non-release runtime")

    initial_loss = NaN
    final_loss = NaN
    total_allocation_bytes = Int128(0)
    total_gc_seconds = 0.0
    last_checkpoint = nothing
    started_at = now(UTC)
    wall_started = time_ns()
    open(trace_path, "w") do trace_io
        println(
            trace_io,
            "update\tcomposite_loss\tlistnet_loss\t" *
            "soma_spike_rate\tnmda_current_mean\t" *
            "calcium_event_rate\trouting_entropy\t" *
            "gradient_norm\tstates_per_second\t" *
            "wall_seconds\tallocation_bytes\tgc_seconds\t" *
            "location_moves",
        )
        open(manifest_path, "w") do manifest_io
            println(manifest_io, "update\tsha256\tpath")
            Arena.paper_checkpoint_integrity!(trainer)
            record = save_checkpoint(
                joinpath(
                    checkpoint_dir,
                    "checkpoint_000000000.jld2",
                ),
                trainer,
                sampler,
                run_config;
                update=0,
            )
            println(
                manifest_io,
                join(
                    (record.update, record.sha256, record.path),
                    '\t',
                ),
            )
            flush(manifest_io)
            last_checkpoint = record

            Arena.run_with_paper_team!(executor) do running
                _release_status(
                    status_path;
                    training_started=true,
                    completed=false,
                    update=0,
                    stage=options.stage,
                )
                for update in 1:options.updates
                    Arena.Point.fill_next_rows!(
                        Arena.paper_training_arena(
                            trainer,
                        ).rows,
                        sampler,
                    )
                    Arena.paper_arena_update!(running)
                    trainer.optimizer.step == update ||
                        error("optimizer update drift")
                    loss = trainer.last_loss
                    metrics = trainer.metrics
                    observed =
                        Float64(loss.composite_loss)
                    update == 1 &&
                        (initial_loss = observed)
                    final_loss = observed
                    total_allocation_bytes +=
                        Int128(metrics.allocation_bytes)
                    total_gc_seconds += metrics.gc_seconds
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
                            "nmda=$(metrics.nmda_current_mean) " *
                            "route_H=$(metrics.routing_entropy)",
                        )
                        flush(stdout)
                    end
                    if update % options.checkpoint_interval == 0 ||
                       update == options.updates
                        Arena.paper_checkpoint_integrity!(
                            trainer,
                        )
                        record = save_checkpoint(
                            joinpath(
                                checkpoint_dir,
                                "checkpoint_" *
                                lpad(string(update), 9, '0') *
                                ".jld2",
                            ),
                            trainer,
                            sampler,
                            run_config;
                            update,
                        )
                        println(
                            manifest_io,
                            join((
                                record.update,
                                record.sha256,
                                record.path,
                            ), '\t'),
                        )
                        flush(manifest_io)
                        last_checkpoint = record
                        _release_status(
                            status_path;
                            training_started=true,
                            completed=false,
                            update,
                            stage=options.stage,
                        )
                    end
                end
            end
        end
    end

    final_parameter_hash =
        Arena.paper_end_run_integrity!(trainer)
    final_parameter_hash == initial_parameter_hash ||
        error("frozen internal parameter hash changed")
    Arena.paper_internal_sha256(trainer) ==
        initial_artifact_hash ||
        error("frozen internal artifact hash changed")
    Arena.paper_internal_max_delta(trainer) == 0.0f0 ||
        error("frozen internal arrays changed")
    Production.assert_production_bundle_unchanged!(bundle)
    total_wall_seconds =
        (time_ns() - wall_started) * 1.0e-9
    _release_status(
        status_path;
        training_started=true,
        completed=true,
        update=trainer.optimizer.step,
        stage=options.stage,
    )
    result = (;
        schema="hd-swsnn-release-result-final-v1",
        completed=true,
        training_started=true,
        stage=options.stage,
        updates=trainer.optimizer.step,
        initial_loss,
        final_loss,
        total_wall_seconds,
        updates_per_second=
            trainer.optimizer.step /
            max(total_wall_seconds, eps(Float64)),
        total_allocation_bytes=string(total_allocation_bytes),
        total_gc_seconds,
        final_metrics=(;
            firing_rate=trainer.metrics.firing_rate,
            nmda_current_mean=
                trainer.metrics.nmda_current_mean,
            calcium_event_rate=
                trainer.metrics.calcium_event_rate,
            routing_entropy=trainer.metrics.routing_entropy,
            states_per_second=
                trainer.metrics.states_per_second,
        ),
        frozen_internal_audit=(;
            initial_artifact_sha256=initial_artifact_hash,
            final_artifact_sha256=
                Arena.paper_internal_sha256(trainer),
            initial_parameter_sha256=
                initial_parameter_hash,
            final_parameter_sha256=
                final_parameter_hash,
            internal_max_delta=
                Arena.paper_internal_max_delta(trainer),
            optimizer_groups=FINAL_OPTIMIZER_GROUPS,
            frozen_internal_outside_optimizer=true,
            passed=true,
        ),
        official_location_audit=(;
            location_index_type="UInt16",
            official_segment_count=642,
            location_mapping_sha256=
                aux.location_mapping_sha256,
        ),
        final_executor="PaperExecutorFinal",
        checkpoint=last_checkpoint,
        started_at_utc=Dates.format(
            started_at,
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
        completed_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
        run_config,
    )
    result_path =
        joinpath(options.output_dir, "result.json")
    _write_json(result_path, result)
    println(
        "completed stage=$(options.stage) " *
        "updates=$(result.updates) " *
        "initial_loss=$(result.initial_loss) " *
        "final_loss=$(result.final_loss) " *
        "result=$result_path",
    )
    return result
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) &&
    final_release_main()
