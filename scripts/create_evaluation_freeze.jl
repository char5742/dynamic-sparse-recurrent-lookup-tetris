using Dates
using JSON3
using SHA
using TOML

module FreezeSourceFingerprint
include(joinpath(@__DIR__, "source_fingerprint.jl"))
end

const FREEZE_ROOT = normpath(joinpath(@__DIR__, ".."))
const FREEZE_SCHEMA_VERSION = "evaluation-freeze-v1"
const DEFAULT_FREEZE_PROTOCOL =
    joinpath(FREEZE_ROOT, "configs", "evaluation_protocol.toml")

freeze_sha256_file(path::AbstractString) = bytes2hex(open(sha256, path))

function freeze_property(value, name::Symbol, default=nothing)
    if value isa AbstractDict
        return get(value, String(name), get(value, name, default))
    end
    return hasproperty(value, name) ? getproperty(value, name) : default
end

function freeze_sorted_pairs(value)
    names = if value isa AbstractDict
        sort!(String.(collect(keys(value))))
    else
        sort!(String.(propertynames(value)))
    end
    return [name => freeze_property(value, Symbol(name)) for name in names]
end

function freeze_material(registry)
    lines = String[
        "schema=$(freeze_property(registry, :schema_version))",
        "protocol=$(freeze_property(registry.protocol, :sha256))",
        "protocol_version=$(freeze_property(registry.protocol, :version))",
        "source=$(freeze_property(registry.source, :sha256))",
        "manifest=$(freeze_property(registry.source, :manifest_sha256))",
    ]
    for role_name in (:baseline, :candidate)
        role = freeze_property(registry.roles, role_name)
        push!(lines, "role=$(role_name)")
        push!(lines, "config=$(freeze_property(role, :config_sha256))")
        push!(lines, "checkpoint=$(freeze_property(role, :checkpoint_sha256))")
        for (name, value) in freeze_sorted_pairs(role.runtime)
            push!(lines, "$(role_name).runtime.$name=$(repr(value))")
        end
        for (name, value) in freeze_sorted_pairs(role.budget)
            push!(lines, "$(role_name).budget.$name=$(repr(value))")
        end
    end
    return join(lines, '\n')
end

freeze_id(registry) = "eval-" * bytes2hex(sha256(freeze_material(registry)))

function toml_role_record(config_path::AbstractString, expected_role::String)
    absolute_config = normpath(abspath(config_path))
    isfile(absolute_config) || error("missing $expected_role config: $absolute_config")
    config = TOML.parsefile(absolute_config)
    get(config, "role", nothing) == expected_role || error(
        "$(absolute_config) role must be $(repr(expected_role))"
    )
    for table in ("runtime", "budget")
        haskey(config, table) || error("$absolute_config missing [$table]")
    end
    checkpoint_value = get(config, "checkpoint", nothing)
    checkpoint_value isa AbstractString || error("$absolute_config missing checkpoint")
    checkpoint_path = normpath(abspath(String(checkpoint_value)))
    isfile(checkpoint_path) || error("missing $expected_role checkpoint: $checkpoint_path")
    checkpoint_hash = freeze_sha256_file(checkpoint_path)
    declared_hash = get(config, "checkpoint_sha256", nothing)
    declared_hash == checkpoint_hash || error(
        "$absolute_config checkpoint_sha256 does not match $checkpoint_path"
    )
    return (;
        role=expected_role,
        config_path=absolute_config,
        config_sha256=freeze_sha256_file(absolute_config),
        checkpoint_path,
        checkpoint_sha256=checkpoint_hash,
        runtime=config["runtime"],
        budget=config["budget"],
    )
end

function freeze_expected_budget(protocol)
    environment = protocol["environment"]
    budget = protocol["model_only_budget"]
    return Dict(
        "episode_piece_limit" => Int(environment["episode_piece_limit"]),
        "next_count" => Int(environment["default_next_count"]),
        "hold_enabled" => Bool(environment["hold_enabled"]),
        "candidate_order" => String(environment["candidate_order"]),
        "lookahead_expansions" => Int(budget["lookahead_expansions"]),
        "logical_network_calls_per_decision" => Int(
            budget["network_calls_per_decision"]
        ),
        "candidate_generation" => String(budget["candidate_generation"]),
        "selection" => String(budget["selection"]),
    )
end

function require_frozen_budget(role, expected)
    actual = role.budget
    sort!(collect(keys(actual))) == sort!(collect(keys(expected))) || error(
        "$(role.role) budget fields do not exactly match protocol v1.1.0"
    )
    for (field, value) in expected
        actual[field] == value || error(
            "$(role.role) budget.$field=$(repr(actual[field])); protocol requires $(repr(value))"
        )
    end
end

function create_evaluation_freeze(
    baseline_config_path::AbstractString,
    candidate_config_path::AbstractString,
    output_path::AbstractString;
    protocol_path::AbstractString=DEFAULT_FREEZE_PROTOCOL,
)
    absolute_protocol = normpath(abspath(protocol_path))
    isfile(absolute_protocol) || error("missing protocol: $absolute_protocol")
    protocol = TOML.parsefile(absolute_protocol)
    baseline = toml_role_record(baseline_config_path, "baseline")
    candidate = toml_role_record(candidate_config_path, "candidate")
    frozen_baseline_hash = protocol["baseline_policy_semantics"]["baseline_checkpoint_sha256"]
    baseline.checkpoint_sha256 == frozen_baseline_hash || error(
        "baseline checkpoint does not match protocol baseline_policy_semantics"
    )
    frozen_budget = freeze_expected_budget(protocol)
    require_frozen_budget(baseline, frozen_budget)
    require_frozen_budget(candidate, frozen_budget)
    fingerprint = FreezeSourceFingerprint.fingerprint()
    body = (;
        schema_version=FREEZE_SCHEMA_VERSION,
        generated_at=string(now()),
        protocol=(;
            path=absolute_protocol,
            sha256=freeze_sha256_file(absolute_protocol),
            version=String(protocol["protocol_version"]),
        ),
        source=(;
            repository_root=FREEZE_ROOT,
            sha256=fingerprint.source_sha256,
            manifest_sha256=fingerprint.manifest_sha256,
            file_count=fingerprint.file_count,
        ),
        roles=(; baseline, candidate),
    )
    registry = merge(body, (; freeze_id=freeze_id(body)))
    absolute_output = normpath(abspath(output_path))
    mkpath(dirname(absolute_output))
    open(absolute_output, "w") do io
        JSON3.pretty(io, registry)
    end
    @info "Created evaluation freeze" absolute_output registry.freeze_id
    return registry
end

function compare_freeze_table!(errors, registered, actual, label)
    registered_names = sort!(String.(propertynames(registered)))
    actual_names = sort!(collect(keys(actual)))
    registered_names == actual_names || push!(
        errors, "$label fields differ from the registered config"
    )
    for name in intersect(registered_names, actual_names)
        freeze_property(registered, Symbol(name)) == actual[name] || push!(
            errors, "$label.$name differs from the registered config"
        )
    end
end

"""Validate a registry against current protocol, source, Manifest and role files."""
function validate_freeze_registry(
    registry_path::AbstractString;
    protocol_path::AbstractString=DEFAULT_FREEZE_PROTOCOL,
)
    errors = String[]
    absolute_registry = normpath(abspath(registry_path))
    if !isfile(absolute_registry)
        return nothing, ["freeze registry does not exist: $absolute_registry"]
    end
    registry = try
        JSON3.read(read(absolute_registry, String))
    catch exception
        return nothing, ["freeze registry is not valid JSON: $(sprint(showerror, exception))"]
    end
    freeze_property(registry, :schema_version) == FREEZE_SCHEMA_VERSION || push!(
        errors, "freeze registry schema_version must be $FREEZE_SCHEMA_VERSION"
    )
    for field in (:protocol, :source, :roles, :freeze_id)
        hasproperty(registry, field) || push!(errors, "freeze registry missing $field")
    end
    !isempty(errors) && return registry, errors

    absolute_protocol = normpath(abspath(protocol_path))
    if !isfile(absolute_protocol)
        push!(errors, "protocol does not exist: $absolute_protocol")
    else
        protocol_hash = freeze_sha256_file(absolute_protocol)
        freeze_property(registry.protocol, :sha256) == protocol_hash || push!(
            errors, "freeze registry protocol hash does not match the on-disk protocol"
        )
        protocol = TOML.parsefile(absolute_protocol)
        freeze_property(registry.protocol, :version) == protocol["protocol_version"] || push!(
            errors, "freeze registry protocol version does not match the on-disk protocol"
        )
    end

    fingerprint = FreezeSourceFingerprint.fingerprint()
    freeze_property(registry.source, :sha256) == fingerprint.source_sha256 || push!(
        errors, "freeze registry source hash does not match the current source tree"
    )
    freeze_property(registry.source, :manifest_sha256) == fingerprint.manifest_sha256 || push!(
        errors, "freeze registry Manifest hash does not match the current Manifest"
    )
    freeze_property(registry, :freeze_id) == freeze_id(registry) || push!(
        errors, "freeze registry freeze_id does not match its canonical contents"
    )

    for role_name in (:baseline, :candidate)
        if !hasproperty(registry.roles, role_name)
            push!(errors, "freeze registry missing roles.$role_name")
            continue
        end
        role = getproperty(registry.roles, role_name)
        config_path = String(freeze_property(role, :config_path, ""))
        checkpoint_path = String(freeze_property(role, :checkpoint_path, ""))
        if !isfile(config_path)
            push!(errors, "registered $role_name config does not exist: $config_path")
            continue
        end
        if freeze_sha256_file(config_path) != freeze_property(role, :config_sha256)
            push!(errors, "registered $role_name config file was modified")
            continue
        end
        config = TOML.parsefile(config_path)
        get(config, "role", nothing) == String(role_name) || push!(
            errors, "registered $role_name config has the wrong role"
        )
        config_checkpoint = haskey(config, "checkpoint") ?
                            normpath(abspath(String(config["checkpoint"]))) : ""
        config_checkpoint == checkpoint_path || push!(
            errors, "registered $role_name checkpoint path differs from its config"
        )
        get(config, "checkpoint_sha256", nothing) ==
            freeze_property(role, :checkpoint_sha256) || push!(
            errors, "registered $role_name checkpoint hash differs from its config"
        )
        if !isfile(checkpoint_path)
            push!(errors, "registered $role_name checkpoint does not exist: $checkpoint_path")
        elseif freeze_sha256_file(checkpoint_path) != freeze_property(role, :checkpoint_sha256)
            push!(errors, "registered $role_name checkpoint file was modified")
        end
        if haskey(config, "runtime")
            compare_freeze_table!(errors, role.runtime, config["runtime"], "$role_name runtime")
        else
            push!(errors, "registered $role_name config is missing runtime")
        end
        if haskey(config, "budget")
            compare_freeze_table!(errors, role.budget, config["budget"], "$role_name budget")
        else
            push!(errors, "registered $role_name config is missing budget")
        end
    end
    if isfile(absolute_protocol) && hasproperty(registry.roles, :baseline)
        protocol = TOML.parsefile(absolute_protocol)
        baseline_hash = freeze_property(registry.roles.baseline, :checkpoint_sha256)
        baseline_hash ==
            protocol["baseline_policy_semantics"]["baseline_checkpoint_sha256"] || push!(
            errors, "registered baseline checkpoint differs from the protocol"
        )
    end
    return registry, unique(errors)
end

function create_freeze_main(args=ARGS)
    length(args) in (3, 4) || error(
        "usage: julia --project=. scripts/create_evaluation_freeze.jl " *
        "BASELINE.toml CANDIDATE.toml OUTPUT.json [PROTOCOL.toml]"
    )
    protocol_path = length(args) == 4 ? args[4] : DEFAULT_FREEZE_PROTOCOL
    return create_evaluation_freeze(args[1], args[2], args[3]; protocol_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    create_freeze_main()
end
