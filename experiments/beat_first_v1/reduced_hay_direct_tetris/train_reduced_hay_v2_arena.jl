using Dates
using JSON3
using LinearAlgebra
using Lux
using Printf
using Random
using SHA

include(joinpath(@__DIR__, "..", "training", "core.jl"))
include(joinpath(@__DIR__, "ReducedHayV2ArenaTraining.jl"))
include(joinpath(@__DIR__, "ReducedHayV2TrainingCheckpoint.jl"))

using .BeatFirstTrainingCore
using .ReducedHayWorkspaceSNN
using .ReducedHayV2ArenaTraining
using .ReducedHayV2TrainingCheckpoint

const DEFAULT_DATASET =
    raw"D:\tetris-paper-plus\datasets\beat_first_v1\teacher_v3"
const DEFAULT_OUTPUT_ROOT =
    raw"D:\tetris-paper-plus\runs\reduced_hay_v2_arena"
const MODEL_SEED = UInt64(0x44454e4453435241)
const SAMPLER_SEED = UInt64(0x44454e4453414d50)
const ROUTING_SEED = UInt64(0x44454e44524f5554)
const OVERFIT_PANEL_SEED = UInt64(0x4f56455246495438)
const TRACE_HEADER = join((
    "update",
    "loss",
    "listnet",
    "q_huber",
    "margin",
    "death",
    "quantile",
    "geometry",
    "firing_rate",
    "plateau_mean",
    "routing_entropy",
    "local_q",
    "local_death",
    "local_quantile",
    "local_geometry",
    "gradient_norm",
    "states_per_second",
    "wall_seconds",
    "cpu_seconds",
    "allocation_bytes",
    "gc_seconds",
    "structural_flips",
    "branch_moves",
), '\t')

function parse_options(arguments)
    values = Dict{String,String}()
    flags = Set{String}()
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        startswith(argument, "--") ||
            error("unexpected positional argument: $argument")
        name = argument[3:end]
        if name in ("deterministic-routing",)
            push!(flags, name)
            index += 1
            continue
        end
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
    resume = get(values, "resume", "")
    return (;
        dataset=abspath(get(values, "dataset", DEFAULT_DATASET)),
        output_root=abspath(get(
            values,
            "output-root",
            DEFAULT_OUTPUT_ROOT,
        )),
        output_dir=isempty(output_dir) ?
            nothing : abspath(output_dir),
        resume=isempty(resume) ? nothing : abspath(resume),
        preset=Symbol(get(
            values,
            "preset",
            "reduced_hay_scaled_v2",
        )),
        updates=integer("updates", 100_000),
        state_batch=integer("state-batch", 8),
        width=integer("width", 80),
        workers=integer("workers", workers_default),
        learning_rate=real("learning-rate", 5.0e-4),
        weight_decay=real("weight-decay", 1.0e-5),
        structural_interval=integer("structural-interval", 25),
        branch_interval=integer("branch-interval", 128),
        credit_warmup_updates=
            integer("credit-warmup-updates", 128),
        credit_ramp_updates=
            integer("credit-ramp-updates", 512),
        recurrent_learning_rate_multiplier=
            real("recurrent-learning-rate-multiplier", 0.001),
        overfit_states=integer("overfit-states", 0),
        checkpoint_interval=integer("checkpoint-interval", 1_000),
        log_interval=integer("log-interval", 100),
        stochastic_routing=
            !("deterministic-routing" in flags),
        cpuset_mode=Symbol(get(values, "cpuset-mode", "none")),
    )
end

function validate_options(options)
    isdir(options.dataset) ||
        error("teacher dataset directory is absent: $(options.dataset)")
    options.updates > 0 ||
        error("updates must be positive")
    options.state_batch > 0 ||
        error("state-batch must be positive")
    options.width == 80 ||
        error("teacher_v3 production width must be 80")
    options.preset in (
        :tiny_recurrent_v2,
        :small_recurrent_v2,
        :reduced_hay_scaled_v2,
    ) || error(
        "preset must be tiny_recurrent_v2, small_recurrent_v2, " *
        "or reduced_hay_scaled_v2",
    )
    2 <= options.workers <= Threads.nthreads(:default) ||
        error("workers must be in 2:$(Threads.nthreads(:default))")
    options.checkpoint_interval > 0 ||
        error("checkpoint-interval must be positive")
    options.log_interval > 0 ||
        error("log-interval must be positive")
    options.credit_warmup_updates >= 0 ||
        error("credit-warmup-updates must be nonnegative")
    options.credit_ramp_updates >= 0 ||
        error("credit-ramp-updates must be nonnegative")
    isfinite(options.recurrent_learning_rate_multiplier) &&
        options.recurrent_learning_rate_multiplier > 0 ||
        error(
            "recurrent-learning-rate-multiplier must be finite and positive",
        )
    options.overfit_states >= 0 ||
        error("overfit-states must be nonnegative")
    options.overfit_states == 0 ||
        options.overfit_states >= options.state_batch ||
        error("overfit-states must be zero or at least state-batch")
    options.overfit_states == 0 ||
        mod(options.overfit_states, options.state_batch) == 0 ||
        error("overfit-states must be a multiple of state-batch")
    options.cpuset_mode in (:none, :all, :p_only) ||
        error("cpuset-mode must be none, all, or p_only")
    options.resume === nothing ||
        isfile(options.resume) ||
        error("resume checkpoint is absent: $(options.resume)")
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
    if options.output_dir !== nothing
        directory = options.output_dir
        if options.resume === nothing
            ispath(directory) &&
                error("scratch output directory already exists: $directory")
            mkpath(directory)
        else
            mkpath(directory)
        end
        return directory
    end
    if options.resume !== nothing
        return dirname(dirname(options.resume))
    end
    stamp = Dates.format(
        now(),
        dateformat"yyyymmdd_HHMMSS",
    )
    run_id =
        (options.overfit_states == 0 ?
         "reduced_hay_v2_decolle_eprop_scratch_" :
         "reduced_hay_v2_decolle_eprop_overfit_" *
         string(options.overfit_states) * "states_") *
        stamp *
        "_u" *
        string(options.updates)
    directory = joinpath(options.output_root, run_id)
    ispath(directory) &&
        error("run directory already exists: $directory")
    mkpath(directory)
    return directory
end

function write_trace_row(io, update, trainer)
    loss = trainer.last_loss
    metrics = trainer.metrics
    values = (
        update,
        loss.composite_loss,
        loss.listnet_loss,
        loss.q_huber_loss,
        loss.margin_loss,
        loss.death_loss,
        loss.quantile_teacher_loss,
        loss.geometry_loss,
        metrics.firing_rate,
        metrics.plateau_mean,
        metrics.routing_entropy,
        metrics.local_q_loss,
        metrics.local_death_loss,
        metrics.local_quantile_loss,
        metrics.local_geometry_loss,
        metrics.gradient_norm,
        metrics.states_per_second,
        metrics.wall_seconds,
        metrics.cpu_seconds,
        metrics.allocation_bytes,
        metrics.gc_seconds,
        metrics.structural_flips,
        metrics.branch_moves,
    )
    println(io, join(values, '\t'))
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
    window.loss += trainer.last_loss.composite_loss
    window.states_per_second +=
        trainer.metrics.states_per_second
    window.wall_seconds += trainer.metrics.wall_seconds
    window.cpu_seconds += trainer.metrics.cpu_seconds
    window.allocation_bytes +=
        trainer.metrics.allocation_bytes
    window.gc_seconds += trainer.metrics.gc_seconds
    return window
end

function report!(window::WindowMetrics, update, trainer)
    inverse = inv(max(window.updates, 1))
    @printf(
        "update=%d loss=%.6f excess=%.6f listnet_kl=%.6f window_loss=%.6f states_s=%.3f cpu=%.1f%% fire=%.6f plateau=%.6f route_H=%.6f local=(%.5f,%.5f,%.5f,%.5f) alloc=%d gc=%.6f\n",
        update,
        trainer.last_loss.composite_loss,
        trainer.last_loss.composite_loss -
        trainer.last_loss.teacher_entropy,
        trainer.last_loss.listnet_kl,
        window.loss * inverse,
        window.states_per_second * inverse,
        100.0 * window.cpu_seconds /
        max(
            window.wall_seconds *
            Threads.nthreads(:default),
            eps(Float64),
        ),
        trainer.metrics.firing_rate,
        trainer.metrics.plateau_mean,
        trainer.metrics.routing_entropy,
        trainer.metrics.local_q_loss,
        trainer.metrics.local_death_loss,
        trainer.metrics.local_quantile_loss,
        trainer.metrics.local_geometry_loss,
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
    record = save_reduced_hay_v2_checkpoint(
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

function main(arguments=ARGS)
    Threads.nthreads(:interactive) == 0 ||
        error("launch with --threads=N,0")
    BLAS.set_num_threads(1)
    options = validate_options(parse_options(arguments))
    run_dir = reserve_run_directory(options)
    checkpoint_dir = joinpath(run_dir, "checkpoints")
    mkpath(checkpoint_dir)
    trace_path = joinpath(run_dir, "training_trace.tsv")
    checkpoint_manifest =
        joinpath(run_dir, "checkpoint_manifest.tsv")

    dataset = load_teacher_dataset(
        options.dataset;
        max_candidates=MAX_CANDIDATES,
        allow_partial_dataset=false,
        geometry_cache_max_states=1,
    )
    all_training_rows = training_rows(dataset)
    options.overfit_states <= length(all_training_rows) ||
        error("overfit-states exceeds the training split")
    rows = if options.overfit_states == 0
        all_training_rows
    else
        selected = copy(all_training_rows)
        shuffle!(Xoshiro(OVERFIT_PANEL_SEED), selected)
        sort!(selected[1:options.overfit_states])
    end
    maximum(dataset.action_counts) <= options.width ||
        error("arena width is too small for teacher candidates")
    model = build_reduced_hay_model(options.preset)
    parameters, _ = Lux.setup(Xoshiro(MODEL_SEED), model)
    trainer = ReducedHayV2ArenaTrainer(
        model,
        parameters;
        state_batch=options.state_batch,
        width=options.width,
        learning_rate=options.learning_rate,
        weight_decay=options.weight_decay,
        structural_interval=options.structural_interval,
        branch_interval=options.branch_interval,
        recurrent_learning_rate_multiplier=
            options.recurrent_learning_rate_multiplier,
    )
    run_config = (;
        architecture="cpu-native-reduced-hay-v2-workspace-snn",
        training_backend="fixed-arena-barrierless-decolle-eprop",
        preset=String(options.preset),
        start_mode=options.resume === nothing ?
            "scratch" : "resume",
        target_updates=options.updates,
        state_batch=options.state_batch,
        width=options.width,
        workers=options.workers,
        cpuset_mode=String(options.cpuset_mode),
        stochastic_routing=options.stochastic_routing,
        routing_reward_semantics=String(
            ReducedHayV2ArenaTraining.ROUTING_REWARD_SEMANTICS,
        ),
        dataset=options.dataset,
        dataset_manifest_sha256=
            manifest_sha256(options.dataset),
        model_seed=string(MODEL_SEED),
        sampler_seed=string(SAMPLER_SEED),
        routing_seed=string(ROUTING_SEED),
        learning_rate=options.learning_rate,
        weight_decay=options.weight_decay,
        structural_interval=options.structural_interval,
        branch_interval=options.branch_interval,
        credit_warmup_updates=options.credit_warmup_updates,
        credit_ramp_updates=options.credit_ramp_updates,
        recurrent_learning_rate_multiplier=
            trainer.recurrent_learning_rate_multiplier,
        overfit_states=options.overfit_states,
        overfit_rows=options.overfit_states == 0 ? Int[] : rows,
        overfit_panel_seed=string(OVERFIT_PANEL_SEED),
        global_signal_scale=trainer.global_signal_scale,
        local_signal_scale=trainer.local_signal_scale,
        routing_entropy_weight=
            trainer.routing_entropy_weight,
        routing_entropy_floor=
            trainer.routing_entropy_floor,
        routing_load_weight=trainer.routing_load_weight,
        source_revision=source_revision(),
        julia_version=string(VERSION),
        julia_threads=Threads.nthreads(:default),
        blas_threads=BLAS.get_num_threads(),
    )
    sampler = if options.resume === nothing
        EpochSampler(rows, Xoshiro(SAMPLER_SEED))
    else
        payload = load_reduced_hay_v2_checkpoint(options.resume)
        payload.run_config.dataset_manifest_sha256 ==
            run_config.dataset_manifest_sha256 ||
            error("resume dataset manifest differs")
        restore_reduced_hay_v2_checkpoint!(
            trainer,
            payload,
            rows,
        )
    end
    trainer.optimizer.step <= options.updates ||
        error("checkpoint already exceeds target updates")

    config_path = joinpath(run_dir, "run_config.json")
    if options.resume === nothing
        open(config_path, "w") do io
            JSON3.pretty(io, run_config)
            println(io)
        end
    elseif !isfile(config_path)
        open(config_path, "w") do io
            JSON3.pretty(io, run_config)
            println(io)
        end
    end

    trace_mode =
        trainer.optimizer.step == 0 ? "w" : "a"
    manifest_mode =
        trainer.optimizer.step == 0 ? "w" : "a"
    started_at = now(UTC)
    total_wall_started = time_ns()
    last_checkpoint = nothing
    total_allocations = Int128(0)
    total_gc_seconds = 0.0
    initial_loss = NaN
    final_loss = NaN
    executor = ReducedHayV2ArenaExecutor(
        trainer,
        dataset;
        active_workers=options.workers,
        cpuset_mode=options.cpuset_mode,
        stochastic_routing=options.stochastic_routing,
        routing_seed=ROUTING_SEED,
    )

    open(trace_path, trace_mode) do trace_io
        trace_mode == "w" && println(trace_io, TRACE_HEADER)
        open(checkpoint_manifest, manifest_mode) do manifest_io
            manifest_mode == "w" &&
                println(manifest_io, "update\tsha256\tpath")
            window = WindowMetrics()
            run_with_dendritic_team!(executor) do running
                for update in (
                    trainer.optimizer.step + 1
                ):options.updates
                    Point = ReducedHayV2ArenaTraining.Point
                    Point.fill_next_rows!(
                        trainer.tape.base.rows,
                        sampler,
                    )
                    completed = trainer.optimizer.step
                    running.recurrent_signal_scale = if completed <
                                                         options.credit_warmup_updates
                        0.0f0
                    elseif options.credit_ramp_updates == 0
                        1.0f0
                    else
                        min(
                            1.0f0,
                            Float32(
                                completed -
                                options.credit_warmup_updates +
                                1,
                            ) /
                            Float32(options.credit_ramp_updates),
                        )
                    end
                    reduced_hay_v2_arena_update!(running)
                    trainer.optimizer.step == update ||
                        error("optimizer update drift")
                    write_trace_row(trace_io, update, trainer)
                    initial_loss = isnan(initial_loss) ?
                        trainer.last_loss.composite_loss :
                        initial_loss
                    final_loss =
                        trainer.last_loss.composite_loss
                    total_allocations +=
                        trainer.metrics.allocation_bytes
                    total_gc_seconds +=
                        trainer.metrics.gc_seconds
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
    deltas = reduced_hay_v2_parameter_deltas(trainer)
    results = (;
        schema="reduced-hay-v2-decolle-eprop-arena-results-v1",
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
        final_teacher_entropy=trainer.last_loss.teacher_entropy,
        final_listnet_kl=trainer.last_loss.listnet_kl,
        final_excess_loss=
            trainer.last_loss.composite_loss -
            trainer.last_loss.teacher_entropy,
        total_wall_seconds,
        updates_per_second=
            trainer.optimizer.step /
            max(total_wall_seconds, eps(Float64)),
        total_allocation_bytes=string(total_allocations),
        total_gc_seconds,
        final_metrics=(;
            firing_rate=trainer.metrics.firing_rate,
            plateau_mean=trainer.metrics.plateau_mean,
            routing_entropy=trainer.metrics.routing_entropy,
            local_q_loss=trainer.metrics.local_q_loss,
            local_death_loss=trainer.metrics.local_death_loss,
            local_quantile_loss=
                trainer.metrics.local_quantile_loss,
            local_geometry_loss=
                trainer.metrics.local_geometry_loss,
            gradient_norm=trainer.metrics.gradient_norm,
        ),
        parameter_max_deltas=deltas,
        checkpoint=last_checkpoint,
        run_config,
    )
    results_path = joinpath(run_dir, "results.json")
    open(results_path, "w") do io
        JSON3.pretty(io, results)
        println(io)
    end
    println(
        "completed updates=$(results.updates) " *
        "initial_loss=$(results.initial_loss) " *
        "final_loss=$(results.final_loss) " *
        "run_dir=$(results.run_dir)",
    )
    return results
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
