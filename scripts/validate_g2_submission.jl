using Dates
using JSON3
using Random
using SHA
using Statistics
using TOML

if !isdefined(@__MODULE__, :validate_freeze_registry)
    include(joinpath(@__DIR__, "create_evaluation_freeze.jl"))
end

const G2_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_PROTOCOL_PATH = joinpath(G2_ROOT, "configs", "evaluation_protocol.toml")
const G2_SCHEMA_VERSION = "g2-submission-v1"
const BOOTSTRAP_SAMPLES = 20_000
const BOOTSTRAP_SEED = 0x5eed_1313
const REQUIRED_HASH_FIELDS = (
    :checkpoint_sha256,
    :config_sha256,
    :source_sha256,
    :manifest_sha256,
    :freeze_registry_sha256,
)
const REQUIRED_RUNTIME_FIELDS = (
    :backend,
    :dtype,
    :inference_batch_size,
    :tail_backend,
    :tie_break,
)
const REQUIRED_BUDGET_FIELDS = (
    :episode_piece_limit,
    :next_count,
    :hold_enabled,
    :candidate_order,
    :lookahead_expansions,
    :logical_network_calls_per_decision,
    :candidate_generation,
    :selection,
)
const REQUIRED_EPISODE_FIELDS = (
    :seed,
    :score,
    :steps,
    :game_over,
    :candidate_evaluations,
    :logical_network_calls,
    :physical_network_calls,
    :generation_seconds,
    :inference_seconds,
    :wall_seconds,
)

sha256_file(path::AbstractString) = bytes2hex(open(sha256, path))

property_or_nothing(value, name::Symbol) =
    hasproperty(value, name) ? getproperty(value, name) : nothing

function is_sha256(value)
    value isa AbstractString || return false
    return occursin(r"^[0-9a-f]{64}$", String(value))
end

finite_nonnegative(value) = value isa Real && isfinite(value) && value >= 0

function expected_budget(protocol)
    environment = protocol["environment"]
    model_budget = protocol["model_only_budget"]
    return Dict(
        "episode_piece_limit" => Int(environment["episode_piece_limit"]),
        "next_count" => Int(environment["default_next_count"]),
        "hold_enabled" => Bool(environment["hold_enabled"]),
        "candidate_order" => String(environment["candidate_order"]),
        "lookahead_expansions" => Int(model_budget["lookahead_expansions"]),
        "logical_network_calls_per_decision" => Int(
            model_budget["network_calls_per_decision"]
        ),
        "candidate_generation" => String(model_budget["candidate_generation"]),
        "selection" => String(model_budget["selection"]),
    )
end

function validate_episode!(errors, episode, expected_seeds, piece_limit, path_label)
    for field in REQUIRED_EPISODE_FIELDS
        hasproperty(episode, field) || push!(errors, "$path_label missing episode.$field")
    end
    any(field -> !hasproperty(episode, field), REQUIRED_EPISODE_FIELDS) && return
    seed = episode.seed
    seed isa Integer || push!(errors, "$path_label episode seed must be an integer")
    seed in expected_seeds || push!(errors, "$path_label contains non-test seed $seed")
    steps = episode.steps
    steps isa Integer || push!(errors, "$path_label seed $seed steps must be an integer")
    if steps isa Integer
        0 <= steps <= piece_limit || push!(
            errors, "$path_label seed $seed steps outside 0:$piece_limit"
        )
        steps < piece_limit && episode.game_over !== true && push!(
            errors,
            "$path_label seed $seed stopped before $piece_limit without game_over=true",
        )
    end
    episode.game_over isa Bool || push!(errors, "$path_label seed $seed game_over must be Bool")
    episode.score isa Integer || push!(errors, "$path_label seed $seed score must be integer")
    for field in (:candidate_evaluations, :logical_network_calls, :physical_network_calls)
        value = getproperty(episode, field)
        (value isa Integer && value >= 0) || push!(
            errors, "$path_label seed $seed $field must be a non-negative integer"
        )
    end
    if steps isa Integer && episode.logical_network_calls isa Integer
        episode.logical_network_calls == steps || push!(
            errors,
            "$path_label seed $seed logical_network_calls must equal one pass per decision/step",
        )
    end
    if steps isa Integer && steps > 0 && episode.candidate_evaluations isa Integer
        episode.candidate_evaluations > 0 || push!(
            errors, "$path_label seed $seed has zero candidate evaluations"
        )
    end
    if episode.physical_network_calls isa Integer && episode.logical_network_calls isa Integer
        episode.physical_network_calls >= episode.logical_network_calls || push!(
            errors, "$path_label seed $seed physical calls cannot be below logical calls"
        )
    end
    for field in (:generation_seconds, :inference_seconds, :wall_seconds)
        finite_nonnegative(getproperty(episode, field)) || push!(
            errors, "$path_label seed $seed $field must be finite and non-negative"
        )
    end
    if finite_nonnegative(episode.wall_seconds) &&
       finite_nonnegative(episode.inference_seconds) &&
       episode.wall_seconds < episode.inference_seconds
        push!(errors, "$path_label seed $seed wall_seconds < inference_seconds")
    end
end

function compare_registered_table!(errors, submitted, registered, fields, label)
    for field in fields
        if !hasproperty(submitted, field)
            push!(errors, "$label missing $field")
        elseif !hasproperty(registered, field)
            push!(errors, "freeze registry $label missing $field")
        elseif getproperty(submitted, field) != getproperty(registered, field)
            push!(errors, "$label.$field differs from the freeze registry")
        end
    end
end

function validate_episode_sources!(errors, document, label)
    sources = property_or_nothing(document, :episode_sources)
    if !(sources isa AbstractVector) || isempty(sources)
        push!(errors, "$label episode_sources must be a non-empty array")
        return
    end
    declared_count = 0
    for (index, source) in enumerate(sources)
        path = property_or_nothing(source, :path)
        digest = property_or_nothing(source, :sha256)
        count = property_or_nothing(source, :episode_count)
        if !(path isa AbstractString) || !isfile(path)
            push!(errors, "$label episode_sources[$index] file does not exist")
        elseif !is_sha256(digest) || sha256_file(path) != digest
            push!(errors, "$label episode_sources[$index] hash mismatch")
        end
        if count isa Integer && count >= 0
            declared_count += count
        else
            push!(errors, "$label episode_sources[$index] invalid episode_count")
        end
    end
    episodes = property_or_nothing(document, :episodes)
    episodes isa AbstractVector && declared_count != length(episodes) && push!(
        errors, "$label episode_sources count does not match episodes"
    )
end

function validate_document(
    document,
    role_name::Symbol,
    registry,
    registry_path,
    protocol,
    protocol_hash,
    path_label,
)
    errors = String[]
    role = getproperty(registry.roles, role_name)
    property_or_nothing(document, :schema_version) == G2_SCHEMA_VERSION || push!(
        errors, "$path_label schema_version must be $G2_SCHEMA_VERSION"
    )
    property_or_nothing(document, :role) == String(role_name) || push!(
        errors, "$path_label role must be $(String(role_name))"
    )
    property_or_nothing(document, :protocol_sha256) == protocol_hash || push!(
        errors, "$path_label protocol_sha256 does not match the on-disk frozen protocol"
    )
    property_or_nothing(document, :evaluation_freeze_id) == registry.freeze_id || push!(
        errors, "$path_label evaluation_freeze_id is not registered"
    )
    absolute_registry = normpath(abspath(registry_path))
    property_or_nothing(document, :freeze_registry_path) == absolute_registry || push!(
        errors, "$path_label freeze_registry_path does not match the validator registry"
    )
    property_or_nothing(document, :freeze_registry_sha256) == sha256_file(absolute_registry) || push!(
        errors, "$path_label freeze_registry_sha256 does not match the registry file"
    )
    frozen_primary = String(protocol["g2_success"]["primary_statistic"])
    property_or_nothing(document, :primary_statistic) == frozen_primary || push!(
        errors, "$path_label primary_statistic must equal $(repr(frozen_primary))"
    )
    for field in REQUIRED_HASH_FIELDS
        is_sha256(property_or_nothing(document, field)) || push!(
            errors, "$path_label $field must be a lowercase 64-character SHA-256"
        )
    end
    expected_hashes = (;
        checkpoint_sha256=role.checkpoint_sha256,
        config_sha256=role.config_sha256,
        source_sha256=registry.source.sha256,
        manifest_sha256=registry.source.manifest_sha256,
    )
    for field in propertynames(expected_hashes)
        property_or_nothing(document, field) == getproperty(expected_hashes, field) || push!(
            errors, "$path_label $field differs from the freeze registry/actual file"
        )
    end
    property_or_nothing(document, :checkpoint_path) == role.checkpoint_path || push!(
        errors, "$path_label checkpoint_path differs from the freeze registry"
    )
    property_or_nothing(document, :config_path) == role.config_path || push!(
        errors, "$path_label config_path differs from the freeze registry"
    )

    runtime = property_or_nothing(document, :runtime)
    if isnothing(runtime)
        push!(errors, "$path_label missing runtime")
    else
        compare_registered_table!(
            errors, runtime, role.runtime, REQUIRED_RUNTIME_FIELDS, "$path_label runtime"
        )
    end
    required_budget = expected_budget(protocol)
    budget = property_or_nothing(document, :budget)
    if isnothing(budget)
        push!(errors, "$path_label missing budget")
    else
        compare_registered_table!(
            errors, budget, role.budget, REQUIRED_BUDGET_FIELDS, "$path_label budget"
        )
        for field in REQUIRED_BUDGET_FIELDS
            hasproperty(budget, field) || continue
            actual = getproperty(budget, field)
            expected = required_budget[String(field)]
            actual == expected || push!(
                errors,
                "$path_label budget.$field=$(repr(actual)); protocol value is $(repr(expected))",
            )
        end
    end
    validate_episode_sources!(errors, document, path_label)

    expected_seeds = Set(Int.(protocol["seed_sets"]["test"]))
    piece_limit = Int(protocol["environment"]["episode_piece_limit"])
    episodes = property_or_nothing(document, :episodes)
    if !(episodes isa AbstractVector)
        push!(errors, "$path_label episodes must be an array")
        return errors
    end
    observed_seeds = Int[]
    for (index, episode) in enumerate(episodes)
        validate_episode!(
            errors, episode, expected_seeds, piece_limit, "$path_label episodes[$index]"
        )
        hasproperty(episode, :seed) && episode.seed isa Integer && push!(observed_seeds, episode.seed)
    end
    length(observed_seeds) == length(unique(observed_seeds)) || push!(
        errors, "$path_label has duplicate episode seeds"
    )
    sort(observed_seeds) == sort!(collect(expected_seeds)) || push!(
        errors,
        "$path_label must contain exactly the 32 sealed test seeds; observed=$(sort(observed_seeds))",
    )
    return errors
end

function bootstrap_location_ci(differences; samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)
    values = Float64.(differences)
    n = length(values)
    rng = Xoshiro(seed)
    means = Vector{Float64}(undef, samples)
    medians = similar(means)
    sample = similar(values)
    for iteration in eachindex(means)
        for index in eachindex(sample)
            sample[index] = values[rand(rng, eachindex(values))]
        end
        means[iteration] = mean(sample)
        medians[iteration] = median(sample)
    end
    return (;
        mean=mean(values),
        mean_ci95=(lower=quantile(means, 0.025), upper=quantile(means, 0.975)),
        median=median(values),
        median_ci95=(lower=quantile(medians, 0.025), upper=quantile(medians, 0.975)),
        bootstrap_samples=samples,
        bootstrap_seed=seed,
    )
end

function metric_summary(episodes, piece_limit)
    scores = Float64[item.score for item in episodes]
    completions = [item.steps == piece_limit for item in episodes]
    function resource_summary(field)
        values = Float64[getproperty(item, field) for item in episodes]
        return (; total=sum(values), mean=mean(values), median=median(values))
    end
    return (;
        episode_count=length(episodes),
        score=(;
            mean=mean(scores),
            median=median(scores),
            maximum=maximum(scores),
            p10=quantile(scores, 0.10),
            p25=quantile(scores, 0.25),
            p75=quantile(scores, 0.75),
            p90=quantile(scores, 0.90),
        ),
        completion_rate=mean(completions),
        candidate_evaluations=resource_summary(:candidate_evaluations),
        logical_network_calls=resource_summary(:logical_network_calls),
        physical_network_calls=resource_summary(:physical_network_calls),
        generation_seconds=resource_summary(:generation_seconds),
        inference_seconds=resource_summary(:inference_seconds),
        wall_seconds=resource_summary(:wall_seconds),
    )
end

function ineligible_result(errors)
    return (;
        generated_at=string(now()),
        schema_version="g2-validation-v2",
        eligibility="ineligible",
        errors=unique(errors),
        note="No G2 decision was computed because the freeze/submission contract failed.",
    )
end

function validate_g2_pair(
    baseline_path::AbstractString,
    candidate_path::AbstractString;
    registry_path::Union{Nothing,AbstractString}=nothing,
    protocol_path::AbstractString=DEFAULT_PROTOCOL_PATH,
)
    isnothing(registry_path) && return ineligible_result([
        "a registered evaluation freeze registry JSON is required"
    ])
    registry, registry_errors = validate_freeze_registry(registry_path; protocol_path)
    !isempty(registry_errors) && return ineligible_result(registry_errors)
    protocol = TOML.parsefile(protocol_path)
    protocol_hash = sha256_file(protocol_path)
    baseline = JSON3.read(read(baseline_path, String))
    candidate = JSON3.read(read(candidate_path, String))
    errors = vcat(
        validate_document(
            baseline,
            :baseline,
            registry,
            registry_path,
            protocol,
            protocol_hash,
            "baseline",
        ),
        validate_document(
            candidate,
            :candidate,
            registry,
            registry_path,
            protocol,
            protocol_hash,
            "candidate",
        ),
    )
    for field in (:evaluation_freeze_id, :primary_statistic)
        property_or_nothing(baseline, field) == property_or_nothing(candidate, field) || push!(
            errors, "baseline and candidate must share $field"
        )
    end
    !isempty(errors) && return ineligible_result(errors)

    baseline_by_seed = Dict(Int(item.seed) => item for item in baseline.episodes)
    candidate_by_seed = Dict(Int(item.seed) => item for item in candidate.episodes)
    seeds = Int.(protocol["seed_sets"]["test"])
    paired_rows = [
        (;
            seed,
            baseline_score=Int(baseline_by_seed[seed].score),
            candidate_score=Int(candidate_by_seed[seed].score),
            difference=Int(candidate_by_seed[seed].score - baseline_by_seed[seed].score),
        ) for seed in seeds
    ]
    differences = [row.difference for row in paired_rows]
    paired = bootstrap_location_ci(differences)
    primary = String(baseline.primary_statistic)
    primary_ci = primary == "mean" ? paired.mean_ci95 : paired.median_ci95
    piece_limit = Int(protocol["environment"]["episode_piece_limit"])
    baseline_summary = metric_summary(baseline.episodes, piece_limit)
    candidate_summary = metric_summary(candidate.episodes, piece_limit)
    wins = count(>(0), differences)
    ties = count(==(0), differences)
    losses = count(<(0), differences)
    return (;
        generated_at=string(now()),
        schema_version="g2-validation-v2",
        eligibility="eligible",
        protocol_sha256=protocol_hash,
        evaluation_freeze_id=String(registry.freeze_id),
        freeze_registry_sha256=sha256_file(registry_path),
        primary_statistic=primary,
        seeds,
        baseline=baseline_summary,
        candidate=candidate_summary,
        paired_differences=paired_rows,
        paired_difference=paired,
        win_tie_loss=(; wins, ties, losses),
        completion_rate_difference=
            candidate_summary.completion_rate - baseline_summary.completion_rate,
        g2_decision=(primary_ci.lower > 0 ? "pass" : "fail"),
        rule="pre-frozen paired-mean bootstrap 95% CI lower bound > 0",
    )
end

function validate_g2_main(args=ARGS)
    length(args) in (3, 4, 5) || error(
        "usage: julia --project=. scripts/validate_g2_submission.jl " *
        "BASELINE.json CANDIDATE.json REGISTRY.json [OUTPUT.json] [PROTOCOL.toml]"
    )
    baseline_path, candidate_path, registry_path = abspath.(args[1:3])
    output_path = length(args) >= 4 ? abspath(args[4]) :
                  joinpath(dirname(candidate_path), "g2_validation.json")
    protocol_path = length(args) == 5 ? abspath(args[5]) : DEFAULT_PROTOCOL_PATH
    result = validate_g2_pair(
        baseline_path, candidate_path; registry_path, protocol_path
    )
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        JSON3.pretty(io, result)
    end
    @info "G2 submission validation" output_path result.eligibility
    result.eligibility == "eligible" || exit(2)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    validate_g2_main()
end
