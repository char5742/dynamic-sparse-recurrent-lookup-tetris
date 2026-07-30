using Dates
using JSON3
using LinearAlgebra
using Lux
using Printf
using Random
using SHA

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "PaperWorkspaceSNN.jl"))
include(joinpath(@__DIR__, "PaperArenaTraining.jl"))
include(joinpath(@__DIR__, "PaperCheckpoint.jl"))

using .BeatFirstTrainingCore
using .PaperWorkspaceSNN
using .PaperArenaTraining
using .PaperCheckpoint

const DEFAULT_DATASET =
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const DEFAULT_OUTPUT_ROOT =
    raw"D:\tetris-paper-plus\runs\paper_multicompartment_snn"
const MODEL_SEED = UInt64(0x50415045524d4f44)
const SAMPLER_SEED = UInt64(0x504150455253414d)
const ROUTING_SEED = UInt64(0x5041504552524f55)
const PRESET = :paper_scaled_v1

const TRACE_COLUMNS = (
    "update",
    "composite_loss",
    "listnet_loss",
    "q_huber_loss",
    "margin_loss",
    "death_loss",
    "quantile_loss",
    "geometry_loss",
    "soma_spike_rate",
    "nmda_current_mean",
    "calcium_event_rate",
    "routing_entropy",
    "local_q_loss",
    "local_death_loss",
    "local_quantile_loss",
    "local_geometry_loss",
    "gradient_norm",
    "states_per_second",
    "wall_seconds",
    "cpu_seconds",
    "allocation_bytes",
    "gc_seconds",
    "structural_flips",
    "location_moves",
)

const VALUE_OPTIONS = Set((
    "dataset",
    "output-root",
    "output-dir",
    "updates",
    "state-batch",
    "width",
    "workers",
    "learning-rate",
    "weight-decay",
    "structural-interval",
    "location-interval",
    "checkpoint-interval",
    "log-interval",
    "cpuset-mode",
    "ablation-mode",
))

function usage()
    return """
    Scratch-only paper-mechanism SNN trainer.

    julia --threads=N,0 train_paper_10k.jl [options]

      --updates N                 Target updates (default: 10000; use 64 for smoke)
      --dataset PATH              teacher_v3 dataset directory
      --output-root PATH          parent directory for a timestamped run
      --output-dir PATH           exact new run directory (must not exist)
      --state-batch N             states per update (default: 1)
      --width N                   candidate arena width (must be 80)
      --workers N                 barrierless workers (default: min(20,N))
      --learning-rate X           AdamW learning rate (default: 5e-4)
      --weight-decay X            AdamW weight decay (default: 1e-5)
      --structural-interval N      gate consolidation interval (default: 25)
      --location-interval N        contact relocation interval (default: 128)
      --checkpoint-interval N      checkpoint cadence (default: 1000)
      --log-interval N             console report cadence (default: 100)
      --cpuset-mode MODE           none, all, or p_only
      --ablation-mode MODE         full by default; recorded in all artifacts
      --deterministic-routing      disable stochastic training routing
      --help                       show this text

    This command deliberately has no --resume mode. Existing output
    directories are rejected so every invocation is an auditable scratch run.
    """
end

function parse_options(arguments)
    values = Dict{String,String}()
    flags = Set{String}()
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        startswith(argument, "--") ||
            error("unexpected positional argument: $argument")
        name = argument[3:end]
        if name in ("deterministic-routing", "help")
            push!(flags, name)
            index += 1
            continue
        end
        name == "resume" && error(
            "train_paper_10k.jl is scratch-only and does not accept --resume",
        )
        name in VALUE_OPTIONS ||
            error("unknown option: $argument")
        index < length(arguments) ||
            error("missing value for $argument")
        haskey(values, name) &&
            error("option repeated: $argument")
        values[name] = arguments[index + 1]
        index += 2
    end
    integer(name, default) =
        parse(Int, get(values, name, string(default)))
    real(name, default) =
        parse(Float64, get(values, name, string(default)))
    workers_default = min(20, Threads.nthreads(:default))
    output_dir = get(values, "output-dir", "")
    return (;
        help="help" in flags,
        dataset=abspath(get(values, "dataset", DEFAULT_DATASET)),
        output_root=abspath(get(
            values,
            "output-root",
            DEFAULT_OUTPUT_ROOT,
        )),
        output_dir=isempty(output_dir) ?
            nothing : abspath(output_dir),
        updates=integer("updates", 10_000),
        state_batch=integer("state-batch", 1),
        width=integer("width", 80),
        workers=integer("workers", workers_default),
        learning_rate=real("learning-rate", 5.0e-4),
        weight_decay=real("weight-decay", 1.0e-5),
        structural_interval=integer("structural-interval", 25),
        location_interval=integer("location-interval", 128),
        checkpoint_interval=integer("checkpoint-interval", 1_000),
        log_interval=integer("log-interval", 100),
        stochastic_routing=
            !("deterministic-routing" in flags),
        cpuset_mode=Symbol(get(values, "cpuset-mode", "none")),
        ablation_mode=Symbol(get(values, "ablation-mode", "full")),
    )
end

function validate_options(options)
    options.help && return options
    isdir(options.dataset) ||
        error("teacher dataset directory is absent: $(options.dataset)")
    options.updates > 0 ||
        error("updates must be positive")
    options.state_batch > 0 ||
        error("state-batch must be positive")
    options.width == 80 ||
        error("teacher_v3 production width must be 80")
    1 <= options.workers <= Threads.nthreads(:default) ||
        error("workers must be in 1:$(Threads.nthreads(:default))")
    options.learning_rate > 0.0 ||
        error("learning-rate must be positive")
    options.weight_decay >= 0.0 ||
        error("weight-decay must be nonnegative")
    options.structural_interval > 0 ||
        error("structural-interval must be positive")
    options.location_interval > 0 ||
        error("location-interval must be positive")
    options.checkpoint_interval > 0 ||
        error("checkpoint-interval must be positive")
    options.log_interval > 0 ||
        error("log-interval must be positive")
    options.cpuset_mode in (:none, :all, :p_only) ||
        error("cpuset-mode must be none, all, or p_only")
    options.ablation_mode in (
        :full,
        :passive,
        :soma_only,
        :no_nmda,
        :lif,
    ) || error(
        "ablation-mode must be full, passive, soma_only, no_nmda, or lif",
    )
    options.output_dir === nothing ||
        !ispath(options.output_dir) ||
        error(
            "scratch output directory already exists: " *
            string(options.output_dir),
        )
    return options
end

function training_rows(dataset)
    rows = findall(==(:train), dataset.predefined_split)
    isempty(rows) && error("teacher dataset has no train split")
    return Int.(rows)
end

function manifest_sha256(dataset_path)
    path = joinpath(dataset_path, "manifest.json")
    isfile(path) ||
        error("teacher dataset manifest is absent: $path")
    return bytes2hex(SHA.sha256(read(path)))
end

function source_revision()
    try
        return strip(readchomp(`git rev-parse HEAD`))
    catch
        return "unknown"
    end
end

function reserve_run_directory(options)
    directory = if options.output_dir !== nothing
        options.output_dir
    else
        stamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
        run_id =
            "paper_scaled_v1_scratch_" *
            stamp *
            "_u" *
            string(options.updates)
        joinpath(options.output_root, run_id)
    end
    ispath(directory) &&
        error("scratch output directory already exists: $directory")
    mkpath(directory)
    return directory
end

function _property(source, names, default=NaN)
    source === nothing && return default
    for name in names
        hasproperty(source, name) &&
            return getproperty(source, name)
    end
    return default
end

function _loss(trainer, names)
    return _property(trainer.last_loss, names)
end

function _metric(trainer, names)
    return _property(trainer.metrics, names)
end

function trace_values(update, trainer)
    return (
        update,
        _loss(trainer, (:composite_loss, :loss)),
        _loss(trainer, (:listnet_loss, :listnet)),
        _loss(trainer, (:q_huber_loss, :q_huber)),
        _loss(trainer, (:margin_loss, :margin)),
        _loss(trainer, (:death_loss, :death)),
        _loss(
            trainer,
            (:quantile_teacher_loss, :quantile_loss, :quantile),
        ),
        _loss(trainer, (:geometry_loss, :geometry)),
        _metric(trainer, (:soma_spike_rate, :firing_rate)),
        _metric(
            trainer,
            (:nmda_current_mean, :mean_nmda_current),
        ),
        _metric(
            trainer,
            (:calcium_event_rate, :local_ca_spike_rate),
        ),
        _metric(trainer, (:routing_entropy,)),
        _metric(trainer, (:local_q_loss,)),
        _metric(trainer, (:local_death_loss,)),
        _metric(trainer, (:local_quantile_loss,)),
        _metric(trainer, (:local_geometry_loss,)),
        _metric(trainer, (:gradient_norm,)),
        _metric(trainer, (:states_per_second,)),
        _metric(trainer, (:wall_seconds,)),
        _metric(trainer, (:cpu_seconds,)),
        _metric(trainer, (:allocation_bytes,), 0),
        _metric(trainer, (:gc_seconds,), 0.0),
        _metric(trainer, (:structural_flips,), 0),
        _metric(trainer, (:location_moves,), 0),
    )
end

function write_trace_row(io, update, trainer)
    println(io, join(trace_values(update, trainer), '\t'))
    return nothing
end

mutable struct WindowMetrics
    updates::Int
    loss::Float64
    states_per_second::Float64
    wall_seconds::Float64
    cpu_seconds::Float64
    allocation_bytes::Int128
    gc_seconds::Float64
end

WindowMetrics() =
    WindowMetrics(0, 0.0, 0.0, 0.0, 0.0, Int128(0), 0.0)

function add!(window::WindowMetrics, trainer)
    window.updates += 1
    window.loss += Float64(
        _loss(trainer, (:composite_loss, :loss)),
    )
    window.states_per_second += Float64(
        _metric(trainer, (:states_per_second,), 0.0),
    )
    window.wall_seconds += Float64(
        _metric(trainer, (:wall_seconds,), 0.0),
    )
    window.cpu_seconds += Float64(
        _metric(trainer, (:cpu_seconds,), 0.0),
    )
    window.allocation_bytes += Int128(
        _metric(trainer, (:allocation_bytes,), 0),
    )
    window.gc_seconds += Float64(
        _metric(trainer, (:gc_seconds,), 0.0),
    )
    return window
end

function report!(window::WindowMetrics, update, trainer)
    inverse = inv(max(window.updates, 1))
    cpu_percent = 100.0 * window.cpu_seconds /
        max(
            window.wall_seconds * Threads.nthreads(:default),
            eps(Float64),
        )
    @printf(
        "update=%d loss=%.6f window_loss=%.6f states_s=%.3f " *
        "cpu=%.1f%% spike=%.6f nmda=%.6f ca=%.6f route_H=%.6f " *
        "alloc=%d gc=%.6f\n",
        update,
        Float64(_loss(trainer, (:composite_loss, :loss))),
        window.loss * inverse,
        window.states_per_second * inverse,
        cpu_percent,
        Float64(_metric(
            trainer,
            (:soma_spike_rate, :firing_rate),
        )),
        Float64(_metric(
            trainer,
            (:nmda_current_mean, :mean_nmda_current),
        )),
        Float64(_metric(
            trainer,
            (:calcium_event_rate, :local_ca_spike_rate),
        )),
        Float64(_metric(trainer, (:routing_entropy,))),
        window.allocation_bytes,
        window.gc_seconds,
    )
    flush(stdout)
    window.updates = 0
    window.loss = 0.0
    window.states_per_second = 0.0
    window.wall_seconds = 0.0
    window.cpu_seconds = 0.0
    window.allocation_bytes = Int128(0)
    window.gc_seconds = 0.0
    return window
end

function save_checkpoint!(
    trainer,
    sampler,
    run_config,
    checkpoint_dir,
    manifest_io,
)
    update = trainer.optimizer.step
    record = save_paper_checkpoint(
        joinpath(
            checkpoint_dir,
            @sprintf("checkpoint_%09d.jld2", update),
        ),
        trainer,
        sampler,
        run_config;
        update,
    )
    println(
        manifest_io,
        join((record.update, record.sha256, record.path), '\t'),
    )
    flush(manifest_io)
    println(
        "checkpoint update=$(record.update) sha256=$(record.sha256) " *
        "path=$(record.path)",
    )
    flush(stdout)
    return record
end

function _json_metric(value)
    value isa AbstractFloat && !isfinite(value) &&
        return nothing
    return value
end

function final_metrics(trainer)
    values = trace_values(trainer.optimizer.step, trainer)
    return Dict(
        TRACE_COLUMNS[index] => _json_metric(values[index])
        for index in 2:length(TRACE_COLUMNS)
    )
end

function main(arguments=ARGS)
    options = validate_options(parse_options(arguments))
    if options.help
        println(usage())
        return nothing
    end
    Threads.nthreads(:interactive) == 0 ||
        error("launch with --threads=N,0")
    BLAS.set_num_threads(1)

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
    trainer = PaperTrainer(
        model,
        parameters;
        state_batch=options.state_batch,
        width=options.width,
        learning_rate=options.learning_rate,
        weight_decay=options.weight_decay,
        structural_interval=options.structural_interval,
        location_interval=options.location_interval,
        ablation_mode=options.ablation_mode,
    )
    mechanism = paper_mechanism_metadata(trainer)
    mechanism.dt_ms == 1.0 ||
        error(
            "paper production runner requires dt_ms=1.0; observed " *
            string(mechanism.dt_ms),
        )
    mechanism.ablation_mode == String(options.ablation_mode) ||
        error("trainer ablation mode differs from requested mode")
    run_config = (;
        architecture="paper-mechanism-multicompartment-workspace-snn",
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
        structural_interval=options.structural_interval,
        location_interval=options.location_interval,
        cell_mechanism_sha256=mechanism.cell_mechanism_sha256,
        cell_mechanism_source=mechanism.cell_mechanism_source,
        morphology_sha256=mechanism.morphology_sha256,
        dt_ms=mechanism.dt_ms,
        substeps_per_cycle=mechanism.substeps_per_cycle,
        cycles=mechanism.cycles,
        decision_window_ms=mechanism.decision_window_ms,
        ablation_mode=mechanism.ablation_mode,
        mechanism_scope=
            "paper-described reduced multicompartment mechanism; " *
            "not the unpublished exact Hay/TwinProp implementation",
        intrinsic_biophysics="fixed",
        trainable_cell_interface=(
            "nonnegative synaptic conductance and discrete anatomical " *
            "contact location"
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
                        last_checkpoint = save_checkpoint!(
                            trainer,
                            sampler,
                            run_config,
                            checkpoint_dir,
                            manifest_io,
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
    deltas = paper_parameter_deltas(trainer)
    result = (;
        schema="paper-mechanism-multicompartment-workspace-snn-result-v1",
        completed=true,
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
        checkpoint=last_checkpoint,
        run_config,
    )
    result_path = joinpath(run_dir, "result.json")
    open(result_path, "w") do io
        JSON3.pretty(io, result)
        println(io)
    end
    println(
        "completed updates=$(result.updates) " *
        "initial_loss=$(result.initial_loss) " *
        "final_loss=$(result.final_loss) " *
        "result=$(result_path)",
    )
    flush(stdout)
    return result
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
