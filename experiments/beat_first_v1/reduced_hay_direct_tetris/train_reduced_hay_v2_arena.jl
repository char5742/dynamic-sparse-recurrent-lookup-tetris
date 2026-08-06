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
const MIN_APICAL_FEEDBACK_WARMUP = 128
const MIN_APICAL_CREDIT_RAMP = 128
const OVERFIT_PANEL_SEED = UInt64(0x4f56455246495438)
const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const SOURCE_SNAPSHOT_SCHEMA = "reduced-hay-v2-source-snapshot-v1"
const REDUCED_HAY_TRAIN_SOURCE_PATHS = (
    "Project.toml",
    "Manifest.toml",
    "experiments/beat_first_v1/reduced_hay_direct_tetris/README.md",
    "experiments/beat_first_v1/reduced_hay_direct_tetris/train_reduced_hay_v2_arena.jl",
    "experiments/beat_first_v1/reduced_hay_direct_tetris/ReducedHayV2TrainingCheckpoint.jl",
    "experiments/beat_first_v1/reduced_hay_direct_tetris/ReducedHayV2ArenaTraining.jl",
    "experiments/beat_first_v1/reduced_hay_direct_tetris/ReducedHayV2IntrinsicAdjoint.jl",
    "experiments/beat_first_v1/reduced_hay_direct_tetris/ReducedHayWorkspaceSNN.jl",
    "experiments/beat_first_v1/dendritic_workspace_snn/DendriticWorkspaceSNN.jl",
    "experiments/beat_first_v1/serial_workspace_snn/ArenaWorkspaceTraining.jl",
    "experiments/beat_first_v1/serial_workspace_snn/SerialWorkspaceSNN.jl",
    "experiments/beat_first_v1/serial_workspace_snn/WorkspaceRoutingPolicy.jl",
    "experiments/beat_first_v1/episodic_vit_recurrent_lookup/bounded_mpmc_queue.jl",
    "experiments/beat_first_v1/episodic_vit_recurrent_lookup/windows_cpu_sets.jl",
    "experiments/beat_first_v1/training/core.jl",
)
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
    "apical_predictor",
    "feedback_alignment",
    "burst_gate",
    "gradient_norm",
    "states_per_second",
    "wall_seconds",
    "cpu_seconds",
    "allocation_bytes",
    "gc_seconds",
    "structural_flips",
    "branch_moves",
), '\t')

function usage()
    return """
Usage: julia --project=. --threads=N,0 train_reduced_hay_v2_arena.jl [OPTIONS]

Core options:
  --preset NAME             required; choose the model architecture explicitly
  --dataset PATH            teacher_v3 dataset directory
  --output-root PATH        parent directory for a new run
  --output-dir PATH         explicit run directory
  --resume PATH             checkpoint to resume
  --updates N               target optimizer update (default: 100000)
  --state-batch N           states per update (default: 8)
  --width N                 candidate arena width (must be 80)
  --workers N               barrierless worker count
  --credit-mode NAME        apical_predictive_online or exact_bptt, plus controls
  --allow-source-drift-on-resume
                            UNSAFE: bypass resume source-snapshot mismatch
  --help                    show this text
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
        if name in (
            "deterministic-routing",
            "stochastic-routing",
            "allow-source-drift-on-resume",
        )
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
    seed(name, default) =
        parse(UInt64, get(values, name, string(default)))
    workers_default = min(20, Threads.nthreads(:default))
    output_dir = get(values, "output-dir", "")
    resume = get(values, "resume", "")
    "deterministic-routing" in flags &&
        "stochastic-routing" in flags &&
        error("routing mode flags are mutually exclusive")
    haskey(values, "preset") || error(
        "--preset is required; choose the architecture explicitly",
    )
    preset = Symbol(values["preset"])
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
        preset,
        updates=integer("updates", 100_000),
        state_batch=integer("state-batch", 8),
        width=integer("width", 80),
        workers=integer("workers", workers_default),
        learning_rate=real("learning-rate", 5.0e-4),
        weight_decay=real("weight-decay", 1.0e-5),
        # Wake may learn the continuous gate/delay/location utilities, but it
        # must not change the discrete graph while the supervised head is
        # stabilising.  Structural integration belongs to the slower sleep
        # phase; finite intervals remain explicit research controls.
        structural_interval=integer("structural-interval", typemax(Int)),
        branch_interval=integer("branch-interval", typemax(Int)),
        credit_warmup_updates=
            integer("credit-warmup-updates", 128),
        credit_ramp_updates=
            integer("credit-ramp-updates", 512),
        recurrent_learning_rate_multiplier=
            # The recurrent graph is a much more sensitive, non-stationary
            # feature generator than the supervised head.  Keep the CLI
            # default aligned with DendriticArenaTrainer instead of silently
            # overriding its two-timescale default by 1,000x.
            real("recurrent-learning-rate-multiplier", 0.001),
        sensory_learning_rate_multiplier=
            real("sensory-learning-rate-multiplier", 0.001),
        routing_learning_rate_multiplier=
            real("routing-learning-rate-multiplier", 0.001),
        communication_learning_rate_multiplier=
            real("communication-learning-rate-multiplier", 0.001),
        update_schedule=Symbol(get(
            values,
            "update-schedule",
            "joint",
        )),
        alternating_head_only_updates=
            integer("alternating-head-only-updates", 512),
        alternating_head_updates=
            integer("alternating-head-updates", 128),
        alternating_recurrent_updates=
            integer("alternating-recurrent-updates", 128),
        recurrent_phase_learning_rate_multiplier=
            real(
                "recurrent-phase-learning-rate-multiplier",
                0.01,
            ),
        recurrent_accumulation_steps=
            integer("recurrent-accumulation-steps", 1),
        routing_temperature_final=
            real("routing-temperature-final", 1.0),
        routing_temperature_anneal_updates=
            integer("routing-temperature-anneal-updates", 5_000),
        routing_entropy_weight=
            real("routing-entropy-weight", 4.0),
        routing_entropy_floor=
            real("routing-entropy-floor", 0.85),
        routing_load_weight=
            real("routing-load-weight", 0.10),
        credit_mode=Symbol(get(
            values,
            "credit-mode",
            values["preset"] in (
                "reduced_hay_fullstate_bound_v10",
                "reduced_hay_exact_slots_v11",
                "reduced_hay_exact_slots_fullrank_v12",
                "reduced_hay_exact_slots_direct_v13",
            ) ?
                "exact_bptt" : "apical_predictive_online",
        )),
        overfit_states=integer("overfit-states", 0),
        checkpoint_interval=integer("checkpoint-interval", 1_000),
        log_interval=integer("log-interval", 100),
        # Wake uses the stable hard route.  Stochastic Plackett-Luce is an
        # explicit sleep/exploration mode because coupling a high-entropy
        # route distribution to the supervised head destroyed the feature
        # stationarity that the two-timescale repair established.
        stochastic_routing="stochastic-routing" in flags,
        allow_source_drift_on_resume=
            "allow-source-drift-on-resume" in flags,
        cpuset_mode=Symbol(get(values, "cpuset-mode", "none")),
        model_seed=seed("model-seed", MODEL_SEED),
        sampler_seed=seed("sampler-seed", SAMPLER_SEED),
        routing_seed=seed("routing-seed", ROUTING_SEED),
        overfit_panel_seed=seed(
            "overfit-panel-seed",
            OVERFIT_PANEL_SEED,
        ),
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
        :tiny_ordered_v3,
        :tiny_structured_v4,
        :tiny_structured_persistent_v5,
        :tiny_structured_quiet_v7,
        :small_recurrent_v2,
        :small_ordered_v3,
        :small_structured_v4,
        :small_structured_coverage_v6,
        :small_structured_quiet_v7,
        :reduced_hay_scaled_v2,
        :reduced_hay_scaled_v3,
        :reduced_hay_structured_v4,
        :reduced_hay_scaled_persistent_v5,
        :reduced_hay_structured_persistent_v5,
        :reduced_hay_scaled_coverage_v6,
        :reduced_hay_scaled_fullcoverage_v8,
        :reduced_hay_tetris_tiles_v9,
        :reduced_hay_fullstate_bound_v10,
        :reduced_hay_exact_slots_v11,
        :reduced_hay_exact_slots_fullrank_v12,
        :reduced_hay_exact_slots_direct_v13,
        :reduced_hay_structured_coverage_v6,
        :reduced_hay_structured_quiet_v7,
    ) || error(
        "preset must be tiny_recurrent_v2, tiny_ordered_v3, " *
        "tiny_structured_v4, tiny_structured_persistent_v5, " *
        "tiny_structured_quiet_v7, " *
        "small_recurrent_v2, " *
        "small_ordered_v3, small_structured_v4, " *
        "small_structured_coverage_v6, " *
        "small_structured_quiet_v7, " *
        "reduced_hay_scaled_v2, reduced_hay_scaled_v3, or " *
        "reduced_hay_structured_v4, reduced_hay_scaled_persistent_v5, " *
        "reduced_hay_structured_persistent_v5, " *
        "reduced_hay_scaled_coverage_v6, or " *
        "reduced_hay_scaled_fullcoverage_v8, or " *
        "reduced_hay_tetris_tiles_v9, or " *
        "reduced_hay_fullstate_bound_v10, or " *
        "reduced_hay_exact_slots_v11, or " *
        "reduced_hay_exact_slots_fullrank_v12, or " *
        "reduced_hay_exact_slots_direct_v13, or " *
        "reduced_hay_structured_coverage_v6, or " *
        "reduced_hay_structured_quiet_v7",
    )
    2 <= options.workers <= Threads.nthreads(:default) ||
        error("workers must be in 2:$(Threads.nthreads(:default))")
    options.checkpoint_interval > 0 ||
        error("checkpoint-interval must be positive")
    options.log_interval > 0 ||
        error("log-interval must be positive")
    options.structural_interval > 0 ||
        error("structural-interval must be positive")
    options.branch_interval > 0 ||
        error("branch-interval must be positive")
    options.credit_warmup_updates >= 0 ||
        error("credit-warmup-updates must be nonnegative")
    options.credit_ramp_updates >= 0 ||
        error("credit-ramp-updates must be nonnegative")
    isfinite(options.recurrent_learning_rate_multiplier) &&
        options.recurrent_learning_rate_multiplier > 0 ||
        error(
            "recurrent-learning-rate-multiplier must be finite and positive",
        )
    isfinite(options.sensory_learning_rate_multiplier) &&
        options.sensory_learning_rate_multiplier >= 0 ||
        error(
            "sensory-learning-rate-multiplier must be finite and nonnegative",
        )
    isfinite(options.routing_learning_rate_multiplier) &&
        options.routing_learning_rate_multiplier > 0 ||
        error(
            "routing-learning-rate-multiplier must be finite and positive",
        )
    isfinite(options.communication_learning_rate_multiplier) &&
        options.communication_learning_rate_multiplier >= 0 ||
        error(
            "communication-learning-rate-multiplier must be finite and nonnegative",
        )
    options.update_schedule in (:joint, :alternating, :accumulated) ||
        error(
            "update-schedule must be joint, alternating, or accumulated",
        )
    options.alternating_head_only_updates >= 0 ||
        error("alternating-head-only-updates must be nonnegative")
    options.alternating_head_updates > 0 ||
        error("alternating-head-updates must be positive")
    options.alternating_recurrent_updates > 0 ||
        error("alternating-recurrent-updates must be positive")
    isfinite(options.recurrent_phase_learning_rate_multiplier) &&
        options.recurrent_phase_learning_rate_multiplier > 0 ||
        error(
            "recurrent-phase-learning-rate-multiplier must be finite and positive",
        )
    options.recurrent_accumulation_steps > 0 ||
        error("recurrent-accumulation-steps must be positive")
    options.update_schedule === :accumulated &&
        options.recurrent_accumulation_steps < 2 &&
        error("accumulated schedule requires at least two accumulation steps")
    options.update_schedule !== :accumulated &&
        options.recurrent_accumulation_steps != 1 &&
        error(
            "recurrent accumulation above one requires accumulated schedule",
        )
    isfinite(options.routing_temperature_final) &&
        options.routing_temperature_final > 0 ||
        error("routing-temperature-final must be finite and positive")
    options.routing_temperature_anneal_updates > 0 ||
        error("routing-temperature-anneal-updates must be positive")
    isfinite(options.routing_entropy_weight) &&
        options.routing_entropy_weight >= 0 ||
        error("routing-entropy-weight must be finite and nonnegative")
    0 <= options.routing_entropy_floor <= 1 ||
        error("routing-entropy-floor must be in [0, 1]")
    isfinite(options.routing_load_weight) &&
        options.routing_load_weight >= 0 ||
        error("routing-load-weight must be finite and nonnegative")
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
    options.credit_mode in (
        :block_teacher,
        :workspace_root_control,
        :workspace_root_reciprocal_control,
        :workspace_root_adaptive_control,
        :apical_predictive_online,
        :exact_bptt,
    ) || error("unsupported credit-mode $(options.credit_mode)")
    if options.preset === :reduced_hay_fullstate_bound_v10
        options.credit_mode in (:exact_bptt, :block_teacher) ||
            error(
                "reduced_hay_fullstate_bound_v10 currently supports " *
                "exact_bptt or block_teacher credit",
            )
    end
    if options.preset in (
        :reduced_hay_exact_slots_v11,
        :reduced_hay_exact_slots_fullrank_v12,
        :reduced_hay_exact_slots_direct_v13,
    )
        options.credit_mode === :exact_bptt ||
            error(
                "exact-slot v11/v12/v13 currently requires " *
                "exact_bptt credit",
            )
    end
    if options.credit_mode === :apical_predictive_online
        options.credit_warmup_updates >= MIN_APICAL_FEEDBACK_WARMUP ||
            error(
                "apical_predictive_online requires at least " *
                "$(MIN_APICAL_FEEDBACK_WARMUP) feedback warm-up updates; " *
                "shorter calibration can reverse cancellation-sensitive " *
                "workspace_decay credit",
            )
        options.credit_ramp_updates >= MIN_APICAL_CREDIT_RAMP ||
            error(
                "apical_predictive_online requires at least " *
                "$(MIN_APICAL_CREDIT_RAMP) recurrent ramp updates",
            )
    end
    options.credit_mode === :exact_bptt &&
        options.stochastic_routing &&
        error("exact_bptt does not support --stochastic-routing")
    options.update_schedule !== :joint &&
        options.credit_mode !== :exact_bptt &&
        error("non-joint update schedules require exact_bptt")
    options.resume === nothing ||
        isfile(options.resume) ||
        error("resume checkpoint is absent: $(options.resume)")
    options.resume !== nothing ||
        !options.allow_source_drift_on_resume ||
        error("allow-source-drift-on-resume requires --resume")
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

file_sha256(path::AbstractString) = bytes2hex(SHA.sha256(read(path)))

function _resolved_source_path(path::AbstractString)
    resolved = realpath(abspath(path))
    return Sys.iswindows() ? lowercase(resolved) : resolved
end

function validate_reduced_hay_active_project(
    repository_root::AbstractString=REPOSITORY_ROOT;
    active_project=Base.active_project(),
)
    active_project === nothing &&
        error("Reduced Hay training requires --project=.")
    expected = joinpath(abspath(repository_root), "Project.toml")
    isfile(expected) || error("repository Project.toml is absent: $expected")
    isfile(active_project) ||
        error("active Julia project is absent: $active_project")
    _resolved_source_path(active_project) ==
        _resolved_source_path(expected) || error(
        "Reduced Hay training requires repository --project=.; " *
        "active project is $(abspath(active_project))",
    )
    return (;
        path=realpath(abspath(active_project)),
        sha256=file_sha256(active_project),
    )
end

function capture_reduced_hay_source_snapshot!(
    run_dir::AbstractString;
    new_run::Bool,
    repository_root::AbstractString=REPOSITORY_ROOT,
)
    new_run || return nothing
    root = abspath(repository_root)
    active_project = validate_reduced_hay_active_project(root)
    directory = joinpath(abspath(run_dir), "source_snapshot")
    ispath(directory) &&
        error("source snapshot already exists: $directory")

    git_head = strip(readchomp(`git -C $root rev-parse HEAD`))
    git_status = read(
        `git -C $root status --porcelain=v1 --untracked-files=all`,
        String,
    )
    git_dirty = !isempty(strip(git_status))

    files_root = joinpath(directory, "files")
    mkpath(files_root)
    records = NamedTuple[]
    for relative_path in REDUCED_HAY_TRAIN_SOURCE_PATHS
        source = normpath(joinpath(root, relative_path))
        isfile(source) ||
            error("source snapshot input is absent: $source")
        destination = normpath(joinpath(files_root, relative_path))
        mkpath(dirname(destination))
        source_hash = file_sha256(source)
        cp(source, destination)
        file_sha256(destination) == source_hash ||
            error("source changed while snapshotting: $source")
        push!(records, (;
            path=replace(relative_path, '\\' => '/'),
            copy=replace(
                relpath(destination, directory),
                '\\' => '/',
            ),
            bytes=filesize(destination),
            sha256=source_hash,
        ))
    end
    closure_payload = join(
        (record.path * "\0" * record.sha256 for record in records),
        '\n',
    )
    closure_hash = bytes2hex(SHA.sha256(codeunits(closure_payload)))
    status_path = joinpath(directory, "git_status_porcelain.txt")
    write(status_path, git_status)
    manifest = (;
        schema=SOURCE_SNAPSHOT_SCHEMA,
        captured_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
        git_head,
        git_dirty,
        active_project_path=replace(active_project.path, '\\' => '/'),
        active_project_sha256=active_project.sha256,
        git_status_path="git_status_porcelain.txt",
        git_status_sha256=file_sha256(status_path),
        source_closure_sha256=closure_hash,
        files=records,
    )
    manifest_path = joinpath(directory, "manifest.json")
    open(manifest_path, "w") do io
        JSON3.pretty(io, manifest)
        println(io)
    end
    manifest_hash = file_sha256(manifest_path)
    write(
        joinpath(directory, "manifest.sha256"),
        manifest_hash * "  manifest.json\n",
    )
    return (;
        created=true,
        verified=true,
        unsafe_override=false,
        directory,
        manifest_path,
        manifest_sha256=manifest_hash,
        git_head,
        git_dirty,
        source_closure_sha256=closure_hash,
        file_count=length(records),
    )
end

function _verify_reduced_hay_source_snapshot(
    run_dir::AbstractString,
    root::AbstractString,
    active_project,
)
    directory = joinpath(abspath(run_dir), "source_snapshot")
    manifest_path = joinpath(directory, "manifest.json")
    manifest_hash_path = joinpath(directory, "manifest.sha256")
    isfile(manifest_path) ||
        error("resume source snapshot is absent: $manifest_path")
    isfile(manifest_hash_path) ||
        error("resume source snapshot hash is absent: $manifest_hash_path")
    manifest_hash = file_sha256(manifest_path)
    recorded_manifest_hash = first(split(strip(
        read(manifest_hash_path, String),
    )))
    manifest_hash == recorded_manifest_hash ||
        error("resume source snapshot manifest hash differs")
    manifest = JSON3.read(read(manifest_path, String))
    String(manifest.schema) == SOURCE_SNAPSHOT_SCHEMA ||
        error("unsupported resume source snapshot schema")
    hasproperty(manifest, :active_project_path) &&
        hasproperty(manifest, :active_project_sha256) ||
        error("resume source snapshot omits active project identity")
    _resolved_source_path(String(manifest.active_project_path)) ==
        _resolved_source_path(active_project.path) ||
        error("resume active Julia project path differs")
    String(manifest.active_project_sha256) == active_project.sha256 ||
        error("resume active Julia project SHA-256 differs")

    expected_paths = Tuple(REDUCED_HAY_TRAIN_SOURCE_PATHS)
    recorded_paths = Tuple(String(entry.path) for entry in manifest.files)
    recorded_paths == expected_paths ||
        error("resume source snapshot closure paths differ")
    current_records = NamedTuple[]
    for (relative_path, entry) in zip(expected_paths, manifest.files)
        source = normpath(joinpath(root, relative_path))
        isfile(source) || error("resume source is absent: $source")
        current_hash = file_sha256(source)
        current_hash == String(entry.sha256) ||
            error("resume source SHA-256 differs: $relative_path")
        expected_copy = replace(
            joinpath("files", relative_path),
            '\\' => '/',
        )
        String(entry.copy) == expected_copy ||
            error("resume source snapshot copy path differs: $relative_path")
        copied = normpath(joinpath(directory, expected_copy))
        isfile(copied) ||
            error("resume source snapshot copy is absent: $relative_path")
        file_sha256(copied) == String(entry.sha256) ||
            error("resume source snapshot copy SHA-256 differs: $relative_path")
        push!(current_records, (path=relative_path, sha256=current_hash))
    end
    closure_payload = join(
        (
            record.path * "\0" * record.sha256
            for record in current_records
        ),
        '\n',
    )
    closure_hash = bytes2hex(SHA.sha256(codeunits(closure_payload)))
    closure_hash == String(manifest.source_closure_sha256) ||
        error("resume source closure SHA-256 differs")
    return (;
        created=false,
        verified=true,
        unsafe_override=false,
        directory,
        manifest_path,
        manifest_sha256=manifest_hash,
        git_head=String(manifest.git_head),
        git_dirty=Bool(manifest.git_dirty),
        source_closure_sha256=closure_hash,
        file_count=length(current_records),
    )
end

function verify_reduced_hay_source_snapshot(
    run_dir::AbstractString;
    repository_root::AbstractString=REPOSITORY_ROOT,
    allow_source_drift::Bool=false,
)
    root = abspath(repository_root)
    active_project = validate_reduced_hay_active_project(root)
    try
        return _verify_reduced_hay_source_snapshot(
            run_dir,
            root,
            active_project,
        )
    catch exception
        allow_source_drift || rethrow()
        reason = sprint(showerror, exception)
        @warn "UNSAFE resume source drift override" reason
        return (;
            created=false,
            verified=false,
            unsafe_override=true,
            directory=joinpath(abspath(run_dir), "source_snapshot"),
            reason,
        )
    end
end

function source_revision()
    try
        return strip(readchomp(`git -C $REPOSITORY_ROOT rev-parse HEAD`))
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
    credit_label = options.credit_mode === :exact_bptt ?
        "exact_bptt" :
        "decolle_eprop"
    run_id =
        (options.overfit_states == 0 ?
         "reduced_hay_v2_$(credit_label)_scratch_" :
         "reduced_hay_v2_$(credit_label)_overfit_" *
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
        metrics.apical_predictor_loss,
        metrics.feedback_alignment_loss,
        metrics.burst_gate_mean,
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
        "update=%d loss=%.6f excess=%.6f listnet_kl=%.6f window_loss=%.6f states_s=%.3f cpu=%.1f%% fire=%.6f plateau=%.6f route_H=%.6f apical=(%.6f,%.6f,%.4f) alloc=%d gc=%.6f\n",
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
        trainer.metrics.apical_predictor_loss,
        trainer.metrics.feedback_alignment_loss,
        trainer.metrics.burst_gate_mean,
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
    if any(argument -> argument in ("--help", "-h"), arguments)
        print(usage())
        return nothing
    end
    Threads.nthreads(:interactive) == 0 ||
        error("launch with --threads=N,0")
    BLAS.set_num_threads(1)
    options = validate_options(parse_options(arguments))
    run_dir = reserve_run_directory(options)
    source_snapshot = if options.resume === nothing
        capture_reduced_hay_source_snapshot!(run_dir; new_run=true)
    else
        verify_reduced_hay_source_snapshot(
            run_dir;
            allow_source_drift=options.allow_source_drift_on_resume,
        )
    end
    if source_snapshot.verified
        println(
            "source_snapshot sha256=$(source_snapshot.manifest_sha256) " *
            "verified=true dirty=$(source_snapshot.git_dirty) " *
            "path=$(source_snapshot.directory)",
        )
    else
        println(
            "source_snapshot verified=false unsafe_override=true " *
            "path=$(source_snapshot.directory)",
        )
    end
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
        shuffle!(Xoshiro(options.overfit_panel_seed), selected)
        sort!(selected[1:options.overfit_states])
    end
    maximum(dataset.action_counts) <= options.width ||
        error("arena width is too small for teacher candidates")
    model = build_reduced_hay_model(options.preset)
    model_topology = reduced_hay_topology(model)
    parameters, _ = Lux.setup(Xoshiro(options.model_seed), model)
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
        sensory_learning_rate_multiplier=
            options.sensory_learning_rate_multiplier,
        routing_learning_rate_multiplier=
            options.routing_learning_rate_multiplier,
        communication_learning_rate_multiplier=
            options.communication_learning_rate_multiplier,
        recurrent_accumulation_steps=
            options.recurrent_accumulation_steps,
        routing_entropy_weight=options.routing_entropy_weight,
        routing_entropy_floor=options.routing_entropy_floor,
        routing_load_weight=options.routing_load_weight,
    )
    run_config = (;
        architecture="cpu-native-reduced-hay-v2-workspace-snn",
        training_backend=options.credit_mode === :exact_bptt ?
            "fixed-arena-barrierless-analytic-exact-bptt" :
            "fixed-arena-barrierless-decolle-eprop",
        preset=String(options.preset),
        cell_export=String(model.cell_export),
        readout_per_cell=model.readout_per_cell,
        block_interface_dim=model.node_dim,
        head_readout=String(model.head_readout),
        head_feature_dim=model_topology.head_feature_dim,
        workspace_binding=String(model.workspace_binding),
        workspace_layout=String(model.workspace_layout),
        workspace_slot_shape=model.workspace_layout ===
            ReducedHayWorkspaceSNN.EXACT_BLOCK_SLOTS ?
            [model.blocks, model.cells_per_block, model.readout_per_cell] :
            nothing,
        route_dim=model.route_dim,
        route_state_rank=div(model.route_dim, model.cells_per_block),
        head_layout=String(model.head_layout),
        head_state_rank=model.head_state_rank,
        branch_bias_mode=String(model.branch_bias_mode),
        workspace_binding_version=String(
            model_topology.workspace_binding,
        ),
        spatial_binding_seed=
            model_topology.spatial_binding_seed === nothing ?
            nothing : string(model_topology.spatial_binding_seed),
        temporal_binding_seed=
            model_topology.temporal_binding_seed === nothing ?
            nothing : string(model_topology.temporal_binding_seed),
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
        credit_mode=String(options.credit_mode),
        dataset=options.dataset,
        dataset_manifest_sha256=
            manifest_sha256(options.dataset),
        model_seed=string(options.model_seed),
        sampler_seed=string(options.sampler_seed),
        routing_seed=string(options.routing_seed),
        learning_rate=options.learning_rate,
        weight_decay=options.weight_decay,
        structural_interval=options.structural_interval,
        branch_interval=options.branch_interval,
        credit_warmup_updates=options.credit_warmup_updates,
        credit_ramp_updates=options.credit_ramp_updates,
        recurrent_learning_rate_multiplier=
            options.recurrent_learning_rate_multiplier,
        sensory_learning_rate_multiplier=
            options.sensory_learning_rate_multiplier,
        routing_learning_rate_multiplier=
            options.routing_learning_rate_multiplier,
        communication_learning_rate_multiplier=
            options.communication_learning_rate_multiplier,
        update_schedule=String(options.update_schedule),
        alternating_head_only_updates=
            options.alternating_head_only_updates,
        alternating_head_updates=options.alternating_head_updates,
        alternating_recurrent_updates=
            options.alternating_recurrent_updates,
        recurrent_phase_learning_rate_multiplier=
            options.recurrent_phase_learning_rate_multiplier,
        recurrent_accumulation_steps=
            options.recurrent_accumulation_steps,
        routing_temperature_initial=model.route_temperature,
        routing_temperature_final=options.routing_temperature_final,
        routing_temperature_anneal_updates=
            options.routing_temperature_anneal_updates,
        overfit_states=options.overfit_states,
        overfit_rows=options.overfit_states == 0 ? Int[] : rows,
        overfit_panel_seed=string(options.overfit_panel_seed),
        checkpoint_interval=options.checkpoint_interval,
        log_interval=options.log_interval,
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
        EpochSampler(rows, Xoshiro(options.sampler_seed))
    else
        payload = load_reduced_hay_v2_checkpoint(options.resume)
        payload.run_config.dataset_manifest_sha256 ==
            run_config.dataset_manifest_sha256 ||
            error("resume dataset manifest differs")
        restored_sampler = restore_reduced_hay_v2_checkpoint!(
            trainer,
            payload,
            rows,
        )
        trainer.recurrent_learning_rate_multiplier == Float32(
            options.recurrent_learning_rate_multiplier,
        ) || error(
            "resume recurrent-learning-rate-multiplier differs from checkpoint",
        )
        trainer.sensory_learning_rate_multiplier == Float32(
            options.sensory_learning_rate_multiplier,
        ) || error(
            "resume sensory-learning-rate-multiplier differs from checkpoint",
        )
        trainer.routing_learning_rate_multiplier == Float32(
            options.routing_learning_rate_multiplier,
        ) || error(
            "resume routing-learning-rate-multiplier differs from checkpoint",
        )
        trainer.communication_learning_rate_multiplier == Float32(
            options.communication_learning_rate_multiplier,
        ) || error(
            "resume communication-learning-rate-multiplier differs from checkpoint",
        )
        restored_sampler
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
        isfile(trace_path) && filesize(trace_path) > 0 ? "a" : "w"
    manifest_mode =
        isfile(checkpoint_manifest) &&
        filesize(checkpoint_manifest) > 0 ? "a" : "w"
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
        routing_seed=options.routing_seed,
        credit_mode=options.credit_mode,
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
                    if options.update_schedule === :accumulated
                        in_warmup = completed <
                            options.alternating_head_only_updates
                        running.recurrent_signal_scale =
                            in_warmup ? 0.0f0 : 1.0f0
                        # The head sees every accumulation batch except the
                        # one on which the averaged recurrent proposal is
                        # committed.  This is a true local block-coordinate
                        # boundary, not simultaneous feature/readout motion.
                        recurrent_commit = !in_warmup &&
                            trainer.recurrent_accumulation_count + 1 >=
                            trainer.recurrent_accumulation_steps
                        running.head_updates_enabled = !recurrent_commit
                        trainer.recurrent_learning_rate_multiplier = Float32(
                            options.recurrent_phase_learning_rate_multiplier,
                        )
                        trainer.routing_learning_rate_multiplier = Float32(
                            options.routing_learning_rate_multiplier,
                        )
                    elseif options.update_schedule === :alternating
                        if completed <
                           options.alternating_head_only_updates
                            running.recurrent_signal_scale = 0.0f0
                            running.head_updates_enabled = true
                        else
                            alternating_offset = completed -
                                options.alternating_head_only_updates
                            alternating_period =
                                options.alternating_head_updates +
                                options.alternating_recurrent_updates
                            in_head_phase = mod(
                                alternating_offset,
                                alternating_period,
                            ) < options.alternating_head_updates
                            running.recurrent_signal_scale =
                                in_head_phase ? 0.0f0 : 1.0f0
                            running.head_updates_enabled = in_head_phase
                        end
                        trainer.recurrent_learning_rate_multiplier = Float32(
                            options.recurrent_phase_learning_rate_multiplier,
                        )
                        trainer.routing_learning_rate_multiplier = Float32(
                            options.routing_learning_rate_multiplier,
                        )
                    else
                        running.head_updates_enabled = true
                        trainer.recurrent_learning_rate_multiplier = Float32(
                            options.recurrent_learning_rate_multiplier,
                        )
                        trainer.routing_learning_rate_multiplier = Float32(
                            options.routing_learning_rate_multiplier,
                        )
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
                    end
                    routing_progress = min(
                        Float32(completed) /
                        Float32(options.routing_temperature_anneal_updates),
                        1.0f0,
                    )
                    initial_temperature =
                        Float32(model.route_temperature)
                    final_temperature = Float32(
                        options.routing_temperature_final,
                    )
                    running.routing_temperature =
                        initial_temperature *
                        (final_temperature / initial_temperature) ^
                        routing_progress
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
        schema="reduced-hay-v2-arena-results-v2",
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
            apical_predictor_loss=
                trainer.metrics.apical_predictor_loss,
            feedback_alignment_loss=
                trainer.metrics.feedback_alignment_loss,
            burst_gate_mean=trainer.metrics.burst_gate_mean,
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
