module PinnedV5ScratchRunnerCLI

export parse_pinned_v5_options, pinned_v5_usage

const _VALUE_OPTIONS = Set((
    "sealed-release",
    "teacher-manifest",
    "teacher-shards",
    "distilled-cell",
    "distilled-sha256",
    "dataset",
    "scratch-root",
    "updates",
    "checkpoint-in",
    "checkpoint-out",
    "workers",
    "learning-rate",
    "weight-decay",
    "location-interval",
    "checkpoint-interval",
    "log-interval",
    "cpuset-mode",
))

const _FLAG_OPTIONS = Set((
    "scratch",
    "require-production",
    "deterministic-routing",
    "help",
))

function pinned_v5_usage()
    return """
    Pinned-V5 HD-SWSNN-TwinProp checkpoint-lineage trainer.

    Scratch 64:
      julia --project=experiments/beat_first_v1 --threads=N,0 \\
        train_hd_swsnn_pinned_v5_checkpoint_lineage.jl \\
        --scratch --updates 64 --checkpoint-out PATH [inputs]

    Continue the same lineage:
      ... --updates 1000 --checkpoint-in CHECKPOINT_64 \\
          --checkpoint-out CHECKPOINT_1K [same inputs]
      ... --updates 10000 --checkpoint-in CHECKPOINT_1K \\
          --checkpoint-out CHECKPOINT_10K [same inputs]

    Required immutable inputs:
      --sealed-release PATH
      --teacher-manifest PATH
      --teacher-shards PATH
      --distilled-cell PATH
      --distilled-sha256 HEX
      --dataset PATH

    Required run controls:
      --updates 64|1000|10000   cumulative target optimizer step
      --checkpoint-out PATH     new final checkpoint
      --scratch                 only for the update-0 lineage root
      --checkpoint-in PATH      required when --scratch is absent
      --require-production      require paper-scale promotable lineage

    Optional:
      --scratch-root PATH       verifier temporary directory
      --workers N
      --learning-rate X         default 5e-4
      --weight-decay X          default 1e-5
      --location-interval N     default 128
      --checkpoint-interval N   default 64/100/1000 by target
      --log-interval N          default 10/50/100 by target
      --cpuset-mode MODE        none, all, or p_only
      --deterministic-routing

    With --require-production omitted, the exact pinned development-scale
    builder is used with development_scale_chain=true.
    """
end

function parse_pinned_v5_options(arguments)
    values = Dict{String,String}()
    flags = Set{String}()
    index = 1
    while index <= length(arguments)
        token = arguments[index]
        startswith(token, "--") ||
            error("unexpected positional argument: $token")
        name = token[3:end]
        if name in _FLAG_OPTIONS
            name in flags && error("option repeated: $token")
            push!(flags, name)
            index += 1
            continue
        end
        name in _VALUE_OPTIONS || error("unknown option: $token")
        index < length(arguments) ||
            error("missing value for $token")
        haskey(values, name) &&
            error("option repeated: $token")
        values[name] = arguments[index + 1]
        index += 2
    end

    "help" in flags && return (; help=true)
    for name in (
        "sealed-release",
        "teacher-manifest",
        "teacher-shards",
        "distilled-cell",
        "distilled-sha256",
        "dataset",
        "updates",
        "checkpoint-out",
    )
        haskey(values, name) || error("--$name is required")
    end

    scratch = "scratch" in flags
    checkpoint_in = get(values, "checkpoint-in", nothing)
    scratch && checkpoint_in !== nothing &&
        error("--scratch and --checkpoint-in are mutually exclusive")
    !scratch && checkpoint_in === nothing &&
        error("--checkpoint-in is required unless --scratch is explicit")

    updates = parse(Int, values["updates"])
    updates in (64, 1_000, 10_000) ||
        error("--updates must be 64, 1000, or 10000")
    scratch && updates != 64 &&
        error("--scratch must create the 64-update lineage root")

    checkpoint_out = abspath(values["checkpoint-out"])
    checkpoint_in = checkpoint_in === nothing ?
        nothing : abspath(checkpoint_in)
    checkpoint_in == checkpoint_out &&
        error("--checkpoint-in and --checkpoint-out must differ")

    integer(name, default) =
        parse(Int, get(values, name, string(default)))
    real(name, default) =
        parse(Float64, get(values, name, string(default)))
    default_checkpoint =
        updates == 64 ? 64 : updates == 1_000 ? 100 : 1_000
    default_log =
        updates == 64 ? 10 : updates == 1_000 ? 50 : 100

    options = (;
        help=false,
        sealed_release=abspath(values["sealed-release"]),
        teacher_manifest=abspath(values["teacher-manifest"]),
        teacher_shards=abspath(values["teacher-shards"]),
        distilled_cell=abspath(values["distilled-cell"]),
        distilled_sha256=lowercase(values["distilled-sha256"]),
        dataset=abspath(values["dataset"]),
        scratch_root=haskey(values, "scratch-root") ?
            abspath(values["scratch-root"]) : nothing,
        updates,
        checkpoint_in,
        checkpoint_out,
        scratch,
        require_production="require-production" in flags,
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

    occursin(r"^[0-9a-f]{64}$", options.distilled_sha256) ||
        error("--distilled-sha256 must be 64 hexadecimal digits")
    options.workers >= 1 ||
        error("--workers must be positive")
    options.learning_rate > 0 ||
        error("--learning-rate must be positive")
    options.weight_decay >= 0 ||
        error("--weight-decay must be nonnegative")
    options.location_interval > 0 ||
        error("--location-interval must be positive")
    options.checkpoint_interval > 0 ||
        error("--checkpoint-interval must be positive")
    options.log_interval > 0 ||
        error("--log-interval must be positive")
    options.cpuset_mode in (:none, :all, :p_only) ||
        error("--cpuset-mode must be none, all, or p_only")
    return options
end

end # module
