using Dates
using JSON3
using SHA
using TOML

if !isdefined(@__MODULE__, :validate_freeze_registry)
    include(joinpath(@__DIR__, "create_evaluation_freeze.jl"))
end

const EXPORT_G2_SCHEMA_VERSION = "g2-submission-v1"

export_sha256_file(path::AbstractString) = bytes2hex(open(sha256, path))

function export_property(value, names::Tuple; required::Bool=true, default=nothing)
    for name in names
        if hasproperty(value, name)
            return getproperty(value, name)
        end
    end
    required && error("episode is missing one of $(join(String.(names), ", "))")
    return default
end

function normalize_g2_episode(episode)
    return (;
        seed=Int(export_property(episode, (:seed,))),
        score=Int(export_property(episode, (:score,))),
        steps=Int(export_property(episode, (:steps,))),
        game_over=Bool(export_property(episode, (:game_over,))),
        candidate_evaluations=Int(
            export_property(episode, (:candidate_evaluations, :candidate_count))
        ),
        logical_network_calls=Int(
            export_property(
                episode,
                (:logical_network_calls, :logical_model_passes);
                required=false,
                default=export_property(episode, (:steps,)),
            )
        ),
        physical_network_calls=Int(
            export_property(
                episode, (:physical_network_calls, :physical_backend_requests)
            )
        ),
        generation_seconds=Float64(export_property(episode, (:generation_seconds,))),
        inference_seconds=Float64(export_property(episode, (:inference_seconds,))),
        wall_seconds=Float64(export_property(episode, (:wall_seconds,))),
    )
end

function read_episode_source(path::AbstractString)
    absolute_path = normpath(abspath(path))
    isfile(absolute_path) || error("episode source does not exist: $absolute_path")
    document = JSON3.read(read(absolute_path, String))
    episodes = if document isa AbstractVector
        document
    elseif hasproperty(document, :episodes)
        document.episodes
    else
        [document]
    end
    return [normalize_g2_episode(item) for item in episodes], (;
        path=absolute_path,
        sha256=export_sha256_file(absolute_path),
        episode_count=length(episodes),
    )
end

function export_g2_submission(
    registry_path::AbstractString,
    role_name::Union{Symbol,AbstractString},
    episode_paths::AbstractVector{<:AbstractString},
    output_path::AbstractString;
    protocol_path::AbstractString=DEFAULT_FREEZE_PROTOCOL,
)
    role_symbol = Symbol(role_name)
    role_symbol in (:baseline, :candidate) || error("role must be baseline or candidate")
    registry, registry_errors = validate_freeze_registry(registry_path; protocol_path)
    isempty(registry_errors) || error(
        "freeze registry is invalid:\n" * join(registry_errors, '\n')
    )
    hasproperty(registry.roles, role_symbol) || error("registry has no $role_symbol role")
    role = getproperty(registry.roles, role_symbol)
    isempty(episode_paths) && error("at least one episode JSON is required")
    episodes = NamedTuple[]
    sources = NamedTuple[]
    for path in episode_paths
        source_episodes, source = read_episode_source(path)
        append!(episodes, source_episodes)
        push!(sources, source)
    end
    protocol = TOML.parsefile(protocol_path)
    expected_seeds = sort!(Int.(protocol["seed_sets"]["test"]))
    observed_seeds = sort!([episode.seed for episode in episodes])
    observed_seeds == expected_seeds || error(
        "episode inputs must contain exactly the 32 frozen test seeds; observed=$observed_seeds"
    )
    length(observed_seeds) == length(unique(observed_seeds)) || error(
        "episode inputs contain duplicate seeds"
    )
    absolute_registry = normpath(abspath(registry_path))
    submission = (;
        schema_version=EXPORT_G2_SCHEMA_VERSION,
        generated_at=string(now()),
        role=String(role_symbol),
        protocol_sha256=freeze_sha256_file(protocol_path),
        evaluation_freeze_id=String(registry.freeze_id),
        freeze_registry_path=absolute_registry,
        freeze_registry_sha256=export_sha256_file(absolute_registry),
        primary_statistic=String(protocol["g2_success"]["primary_statistic"]),
        checkpoint_path=String(role.checkpoint_path),
        checkpoint_sha256=String(role.checkpoint_sha256),
        config_path=String(role.config_path),
        config_sha256=String(role.config_sha256),
        source_sha256=String(registry.source.sha256),
        manifest_sha256=String(registry.source.manifest_sha256),
        runtime=role.runtime,
        budget=role.budget,
        episode_sources=sources,
        episodes=sort!(episodes; by=episode -> episode.seed),
    )
    absolute_output = normpath(abspath(output_path))
    mkpath(dirname(absolute_output))
    open(absolute_output, "w") do io
        JSON3.pretty(io, submission)
    end
    @info "Exported G2 submission" absolute_output role=role_symbol episodes=length(episodes)
    return submission
end

function export_g2_main(args=ARGS)
    length(args) >= 4 || error(
        "usage: julia --project=. scripts/export_g2_submission.jl " *
        "REGISTRY.json ROLE OUTPUT.json EPISODE.json [EPISODE2.json ...] " *
        "[--protocol=PROTOCOL.toml]"
    )
    registry_path, role_name, output_path = args[1:3]
    episode_paths = String[]
    protocol_path = DEFAULT_FREEZE_PROTOCOL
    for argument in args[4:end]
        if startswith(argument, "--protocol=")
            protocol_path = split(argument, '='; limit=2)[2]
        else
            push!(episode_paths, argument)
        end
    end
    return export_g2_submission(
        registry_path, role_name, episode_paths, output_path; protocol_path
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    export_g2_main()
end
