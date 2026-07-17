# These first milestones intentionally use Base only and run before JSON3 or
# any other package import. The wrapper supplies a fresh per-child JSONL path.
let path = get(ENV, "R1_CHILD_MILESTONE_PATH", "")
    if !isempty(path)
        mkpath(dirname(abspath(path)))
        open(abspath(path), "a") do io
            println(io, "{\"phase\":\"calibration\",\"stage\":\"script_enter\",\"pid\":$(getpid()),\"time_ns\":$(time_ns())}")
            println(io, "{\"phase\":\"calibration\",\"stage\":\"imports_begin\",\"pid\":$(getpid()),\"time_ns\":$(time_ns())}")
            flush(io)
        end
    end
end

module R1CalibrationGate

using JSON3
using Random
using SHA
using Statistics

export CALIBRATION_EPISODES,
       CalibrationRow,
       GateArtifactEvidence,
       SourceHashEvidence,
       calibration_row,
       calibration_rows_from_gate,
       gate_artifact_evidence,
       source_hash_evidence,
       calibration_cluster_schedules,
       schedule_digest,
       cluster_bootstrap_override_metrics,
       evaluate_calibration,
       load_calibration_document,
       main

const CALIBRATION_EPISODES = collect(73101:73106)
const MINIMUM_STATES = 120
const MINIMUM_OVERRIDES = 12
const MINIMUM_OVERRIDE_EPISODES = 4
const MINIMUM_OVERRIDE_RATE = 0.01
const MAXIMUM_OVERRIDE_RATE = 0.15
const MINIMUM_PRECISION = 0.70
const MINIMUM_PRECISION_LOWER_BOUND = 0.50
const MINIMUM_MEAN_ADVANTAGE = 0.10
const MAXIMUM_MEDIAN_OVERHEAD_MS = 0.10
const BOOTSTRAP_REPLICATES = 2_000
const BOOTSTRAP_SEED = UInt64(0x5231_73106)
const ONE_SIDED_LOWER_QUANTILE = 0.10
const EXPECTED_FEATURE_COUNT = 70
const EXPECTED_COEFFICIENT_COUNT = 71
const EXPECTED_ENSEMBLE_SIZE = 256
const EXPECTED_FEATURE_NAMES_SHA256 =
    "7e89c16b57dcebac56e3ab4c5be161d5e5430c682e60f3b565dd23ab3b04ac44"

function child_milestone(stage::AbstractString)
    path = get(ENV, "R1_CHILD_MILESTONE_PATH", "")
    isempty(path) && return nothing
    open(abspath(path), "a") do io
        JSON3.write(io, (;
            phase="calibration",
            stage=String(stage),
            pid=getpid(),
            time_ns=time_ns(),
        ))
        write(io, '\n')
        flush(io)
    end
    return nothing
end

child_milestone("imports_complete")

"""
One calibration state. `production_decision=true` means that the production
gate selected old top-2. `reference_decision` must be computed independently
by the scalar reference implementation. Advantages are the *unclipped* A6
values; clipped fit targets must never be passed here.
"""
struct CalibrationRow
    episode_id::Int
    advantage::Float64
    a1_terminal::Bool
    a2_terminal::Bool
    production_decision::Bool
    reference_decision::Bool
    production_selected_candidate_index::Int
    canonical_top1_candidate_index::Int
    canonical_top2_candidate_index::Int
    production_selected_action_digest::String
    canonical_top1_action_digest::String
    canonical_top2_action_digest::String
    overhead_ms::Float64
end

struct GateArtifactEvidence
    finite::Bool
    feature_schema_exact::Bool
    coefficient_shape_exact::Bool
    hyperparameters_exact::Bool
    numeric_value_count::Int
end

struct SourceHashEvidence
    path::String
    actual_sha256::String
    expected_sha256::String
    verified::Bool
end

function calibration_row(;
    episode_id,
    advantage,
    a1_terminal,
    a2_terminal,
    production_decision,
    reference_decision,
    production_selected_candidate_index,
    canonical_top1_candidate_index,
    canonical_top2_candidate_index,
    production_selected_action_digest,
    canonical_top1_action_digest,
    canonical_top2_action_digest,
    overhead_ms,
)
    row = CalibrationRow(
        Int(episode_id),
        Float64(advantage),
        Bool(a1_terminal),
        Bool(a2_terminal),
        Bool(production_decision),
        Bool(reference_decision),
        Int(production_selected_candidate_index),
        Int(canonical_top1_candidate_index),
        Int(canonical_top2_candidate_index),
        String(production_selected_action_digest),
        String(canonical_top1_action_digest),
        String(canonical_top2_action_digest),
        Float64(overhead_ms),
    )
    isfinite(row.advantage) || error("non-finite unclipped A6")
    isfinite(row.overhead_ms) || error("non-finite decision overhead")
    row.overhead_ms >= 0 || error("negative decision overhead")
    row.production_selected_candidate_index > 0 || error("invalid selected candidate index")
    row.canonical_top1_candidate_index > 0 || error("invalid canonical top-1 index")
    row.canonical_top2_candidate_index > 0 || error("invalid canonical top-2 index")
    row.canonical_top1_candidate_index != row.canonical_top2_candidate_index ||
        error("canonical top-1 and top-2 indices are identical")
    all(value -> !isempty(value), (
        row.production_selected_action_digest,
        row.canonical_top1_action_digest,
        row.canonical_top2_action_digest,
    )) || error("empty action digest")
    row.canonical_top1_action_digest != row.canonical_top2_action_digest ||
        error("canonical top-1 and top-2 action digests are identical")
    row.episode_id in CALIBRATION_EPISODES || error(
        "non-calibration episode $(row.episode_id) supplied to R1 calibration gate"
    )
    return row
end

function feature_names_digest(feature_names)
    return bytes2hex(sha256(join(String.(feature_names), "\n")))
end

function gate_artifact_evidence(payload)
    feature_names = String.(required_property(payload, :feature_names))
    means = Float64.(required_property(payload, :feature_mean))
    scales = Float64.(required_property(payload, :feature_scale))
    coefficients = Float64.(required_property(payload, :coefficients))
    constant_feature = Bool.(required_property(payload, :constant_feature))
    numeric = vcat(means, scales, coefficients)
    finite = all(isfinite, numeric) && all(>(0), scales)
    feature_schema_exact = length(feature_names) == EXPECTED_FEATURE_COUNT &&
        length(unique(feature_names)) == EXPECTED_FEATURE_COUNT &&
        feature_names_digest(feature_names) == EXPECTED_FEATURE_NAMES_SHA256 &&
        length(means) == EXPECTED_FEATURE_COUNT &&
        length(scales) == EXPECTED_FEATURE_COUNT &&
        length(constant_feature) == EXPECTED_FEATURE_COUNT
    coefficient_shape_exact =
        Int(required_property(payload, :coefficient_count)) ==
            EXPECTED_COEFFICIENT_COUNT &&
        Int(required_property(payload, :ensemble_size)) == EXPECTED_ENSEMBLE_SIZE &&
        length(coefficients) == EXPECTED_COEFFICIENT_COUNT * EXPECTED_ENSEMBLE_SIZE
    hyperparameters_exact =
        Float64(required_property(payload, :lambda)) == 1.0 &&
        UInt64(required_property(payload, :bootstrap_seed)) == UInt64(0x5231_2026) &&
        Float64(required_property(payload, :lower_quantile)) == 0.10 &&
        Float64(required_property(payload, :override_threshold)) == 0.05
    return GateArtifactEvidence(
        finite,
        feature_schema_exact,
        coefficient_shape_exact,
        hyperparameters_exact,
        length(numeric),
    )
end

function source_hash_evidence(path::AbstractString, expected_sha256::AbstractString)
    source_path = abspath(path)
    isfile(source_path) || error("missing calibration source: $source_path")
    expected = lowercase(String(expected_sha256))
    occursin(r"^[0-9a-f]{64}$", expected) || error("invalid expected source SHA-256")
    actual = bytes2hex(open(sha256, source_path))
    return SourceHashEvidence(source_path, actual, expected, actual == expected)
end

function calibration_cluster_schedules()
    rng = Xoshiro(BOOTSTRAP_SEED)
    return [[
        CALIBRATION_EPISODES[rand(rng, eachindex(CALIBRATION_EPISODES))]
        for _ in eachindex(CALIBRATION_EPISODES)
    ] for _ in 1:BOOTSTRAP_REPLICATES]
end

function schedule_digest(schedules)
    payload = join((join(schedule, ",") for schedule in schedules), ";")
    return bytes2hex(sha256(payload))
end

"""Build rows while keeping the production batch and scalar reference paths separate.

`production_lower_bounds` must execute the production vectorized evaluator and
`reference_lower_bound` the independently implemented scalar evaluator. The
strict `> 0.05` comparison is made here for both paths so equality is a
fallback. This helper intentionally accepts callbacks and does not duplicate
the fixed feature schema owned by `ridge_gate.jl`.
"""
function calibration_rows_from_gate(
    features::AbstractMatrix,
    episode_ids,
    advantages,
    a1_terminal,
    a2_terminal,
    production_selected_candidate_index,
    canonical_top1_candidate_index,
    canonical_top2_candidate_index,
    production_selected_action_digest,
    canonical_top1_action_digest,
    canonical_top2_action_digest,
    overhead_ms;
    production_lower_bounds::Function,
    reference_lower_bound::Function,
    override_threshold::Float64=0.05,
)
    row_count = size(features, 2)
    all(length(values) == row_count for values in (
        episode_ids, advantages, a1_terminal, a2_terminal,
        production_selected_candidate_index, canonical_top1_candidate_index,
        canonical_top2_candidate_index, production_selected_action_digest,
        canonical_top1_action_digest, canonical_top2_action_digest, overhead_ms,
    )) || error("calibration row count mismatch")
    override_threshold == 0.05 || error("R1 override threshold mismatch")
    production = Float64.(production_lower_bounds(features))
    length(production) == row_count || error("production prediction row mismatch")
    all(isfinite, production) || error("non-finite production lower bound")
    rows = Vector{CalibrationRow}(undef, row_count)
    for row in 1:row_count
        reference = Float64(reference_lower_bound(@view features[:, row]))
        isfinite(reference) || error("non-finite scalar reference lower bound")
        production_decision = production[row] > override_threshold
        reference_decision = reference > override_threshold
        rows[row] = calibration_row(;
            episode_id=episode_ids[row],
            advantage=advantages[row],
            a1_terminal=a1_terminal[row],
            a2_terminal=a2_terminal[row],
            production_decision,
            reference_decision,
            production_selected_candidate_index=
                production_selected_candidate_index[row],
            canonical_top1_candidate_index=canonical_top1_candidate_index[row],
            canonical_top2_candidate_index=canonical_top2_candidate_index[row],
            production_selected_action_digest=production_selected_action_digest[row],
            canonical_top1_action_digest=canonical_top1_action_digest[row],
            canonical_top2_action_digest=canonical_top2_action_digest[row],
            overhead_ms=overhead_ms[row],
        )
    end
    return rows
end

"""
Episode-cluster bootstrap of the two override-only statistics. Each replicate
samples all six calibration episodes with replacement and includes every
override row from each sampled episode. A replicate containing no override is
assigned zero for both statistics. This is deliberately fail-closed: sparse,
episode-concentrated overrides cannot gain evidence by dropping empty draws.
"""
function cluster_bootstrap_override_metrics(
    rows::AbstractVector{CalibrationRow};
    schedules,
    expected_schedule_sha256::AbstractString,
)
    regenerated = calibration_cluster_schedules()
    normalized_schedules = [Int.(schedule) for schedule in schedules]
    normalized_schedules == regenerated || error(
        "calibration bootstrap schedules differ from frozen Xoshiro schedule"
    )
    digest = schedule_digest(normalized_schedules)
    digest == lowercase(String(expected_schedule_sha256)) || error(
        "calibration bootstrap schedule digest mismatch"
    )

    override_by_episode = Dict(
        episode => [
            row for row in rows
            if row.episode_id == episode && row.production_decision
        ] for episode in CALIBRATION_EPISODES
    )
    precision_samples = Vector{Float64}(undef, BOOTSTRAP_REPLICATES)
    advantage_samples = Vector{Float64}(undef, BOOTSTRAP_REPLICATES)
    empty_override_replicate_count = 0
    sampled_advantages = Float64[]
    sizehint!(sampled_advantages, length(rows))
    for replicate in 1:BOOTSTRAP_REPLICATES
        empty!(sampled_advantages)
        for episode in normalized_schedules[replicate]
            append!(sampled_advantages, getproperty.(override_by_episode[episode], :advantage))
        end
        if isempty(sampled_advantages)
            empty_override_replicate_count += 1
            precision_samples[replicate] = 0.0
            advantage_samples[replicate] = 0.0
        else
            precision_samples[replicate] = mean(>(0), sampled_advantages)
            advantage_samples[replicate] = mean(sampled_advantages)
        end
    end
    return (;
        precision_lower90=quantile(precision_samples, ONE_SIDED_LOWER_QUANTILE),
        mean_advantage_lower90=quantile(
            advantage_samples, ONE_SIDED_LOWER_QUANTILE
        ),
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        schedule_sha256=digest,
        schedule_matches_regenerated=true,
        lower_quantile=ONE_SIDED_LOWER_QUANTILE,
        empty_override_replicate_count,
    )
end

function evaluate_calibration(
    rows::AbstractVector{CalibrationRow};
    artifact_evidence::GateArtifactEvidence,
    source_evidence::SourceHashEvidence,
    bootstrap_schedules,
    expected_bootstrap_schedule_sha256::AbstractString,
    forbidden_seed_used::Bool=false,
)
    isempty(rows) && error("empty R1 calibration rows")
    episode_ids = sort!(unique(getproperty.(rows, :episode_id)))
    override_rows = filter(row -> row.production_decision, rows)
    override_count = length(override_rows)
    state_count = length(rows)
    override_rate = override_count / state_count
    override_episodes = sort!(unique(getproperty.(override_rows, :episode_id)))
    precision = override_count == 0 ? 0.0 : mean(row.advantage > 0 for row in override_rows)
    mean_advantage = override_count == 0 ? 0.0 : mean(getproperty.(override_rows, :advantage))
    unsafe_terminal_count = count(
        row -> row.production_decision && row.a2_terminal && !row.a1_terminal,
        rows,
    )
    decision_mismatch_count = count(
        row -> row.production_decision != row.reference_decision,
        rows,
    )
    selected_action_mismatch_count = count(
        row -> begin
            expected_index = row.production_decision ?
                row.canonical_top2_candidate_index : row.canonical_top1_candidate_index
            expected_digest = row.production_decision ?
                row.canonical_top2_action_digest : row.canonical_top1_action_digest
            row.production_selected_candidate_index != expected_index ||
                row.production_selected_action_digest != expected_digest
        end,
        rows,
    )
    fallback_mismatch_count = count(row -> begin
        !row.production_decision && (
            row.production_selected_candidate_index != row.canonical_top1_candidate_index ||
            row.production_selected_action_digest != row.canonical_top1_action_digest
        )
    end, rows)
    median_overhead_ms = median(getproperty.(rows, :overhead_ms))
    bootstrap = cluster_bootstrap_override_metrics(rows;
        schedules=bootstrap_schedules,
        expected_schedule_sha256=expected_bootstrap_schedule_sha256,
    )

    checks = (;
        minimum_state_count=state_count >= MINIMUM_STATES,
        exact_calibration_episodes=episode_ids == CALIBRATION_EPISODES,
        override_rate_in_range=MINIMUM_OVERRIDE_RATE <= override_rate <=
                               MAXIMUM_OVERRIDE_RATE,
        minimum_override_count=override_count >= MINIMUM_OVERRIDES,
        override_episode_distribution=length(override_episodes) >=
                                      MINIMUM_OVERRIDE_EPISODES,
        precision_point=precision >= MINIMUM_PRECISION,
        precision_lower_bound=bootstrap.precision_lower90 >
                              MINIMUM_PRECISION_LOWER_BOUND,
        mean_advantage_point=mean_advantage >= MINIMUM_MEAN_ADVANTAGE,
        mean_advantage_lower_bound=bootstrap.mean_advantage_lower90 > 0.0,
        no_top2_only_terminal=unsafe_terminal_count == 0,
        production_reference_exact=decision_mismatch_count == 0,
        selected_action_matches_decision=selected_action_mismatch_count == 0,
        fallback_top1_exact=fallback_mismatch_count == 0,
        overhead_within_budget=median_overhead_ms <= MAXIMUM_MEDIAN_OVERHEAD_MS,
        artifact_finite=artifact_evidence.finite,
        feature_schema_exact=artifact_evidence.feature_schema_exact,
        coefficient_shape_exact=artifact_evidence.coefficient_shape_exact,
        hyperparameters_exact=artifact_evidence.hyperparameters_exact,
        calibration_source_hash_verified=source_evidence.verified,
        no_forbidden_seed_used=!forbidden_seed_used,
    )
    promoted = all(values(checks))
    return (;
        experiment="R1_online_counterfactual_top2_safety_gate",
        status=promoted ? "R1-calibration-promoted" : "R1-calibration-rejected",
        promoted,
        scope="calibration_only_not_game_strength_not_model_improvement",
        state_count,
        calibration_episodes=episode_ids,
        expected_calibration_episodes=CALIBRATION_EPISODES,
        override_count,
        override_rate,
        override_episode_count=length(override_episodes),
        override_episodes,
        override_precision=precision,
        override_mean_unclipped_A6=mean_advantage,
        unsafe_top2_terminal_count=unsafe_terminal_count,
        production_reference_mismatch_count=decision_mismatch_count,
        selected_action_mismatch_count,
        fallback_top1_mismatch_count=fallback_mismatch_count,
        median_decision_overhead_ms=median_overhead_ms,
        bootstrap,
        gate_artifact_evidence=(;
            finite=artifact_evidence.finite,
            feature_schema_exact=artifact_evidence.feature_schema_exact,
            coefficient_shape_exact=artifact_evidence.coefficient_shape_exact,
            hyperparameters_exact=artifact_evidence.hyperparameters_exact,
            numeric_value_count=artifact_evidence.numeric_value_count,
        ),
        calibration_source_evidence=(;
            path=source_evidence.path,
            actual_sha256=source_evidence.actual_sha256,
            expected_sha256=source_evidence.expected_sha256,
            verified=source_evidence.verified,
        ),
        thresholds=(;
            minimum_states=MINIMUM_STATES,
            override_rate=(minimum=MINIMUM_OVERRIDE_RATE, maximum=MAXIMUM_OVERRIDE_RATE),
            minimum_overrides=MINIMUM_OVERRIDES,
            minimum_override_episodes=MINIMUM_OVERRIDE_EPISODES,
            minimum_precision=MINIMUM_PRECISION,
            minimum_precision_lower90=MINIMUM_PRECISION_LOWER_BOUND,
            minimum_mean_unclipped_A6=MINIMUM_MEAN_ADVANTAGE,
            minimum_mean_A6_lower90=0.0,
            maximum_median_overhead_ms=MAXIMUM_MEDIAN_OVERHEAD_MS,
        ),
        checks,
        validation_seed_used=false,
        sealed_test_seed_used=false,
        game_strength_evidence=false,
        model_improvement_evidence=false,
    )
end

required_property(object, name::Symbol) = hasproperty(object, name) ?
    getproperty(object, name) : error("missing required property: $name")

function load_calibration_document(path::AbstractString)
    document = JSON3.read(read(path, String))
    source_role = String(required_property(document, :source_role))
    source_role == "calibration" || error("calibration source_role mismatch")
    rows = CalibrationRow[
        calibration_row(;
            episode_id=required_property(item, :episode_id),
            advantage=required_property(item, :advantage_unclipped_A6),
            a1_terminal=required_property(item, :a1_terminal_within_horizon),
            a2_terminal=required_property(item, :a2_terminal_within_horizon),
            production_decision=required_property(item, :production_decision),
            reference_decision=required_property(item, :reference_decision),
            production_selected_candidate_index=required_property(
                item, :production_selected_candidate_index
            ),
            canonical_top1_candidate_index=required_property(
                item, :canonical_top1_candidate_index
            ),
            canonical_top2_candidate_index=required_property(
                item, :canonical_top2_candidate_index
            ),
            production_selected_action_digest=required_property(
                item, :production_selected_action_digest
            ),
            canonical_top1_action_digest=required_property(
                item, :canonical_top1_action_digest
            ),
            canonical_top2_action_digest=required_property(
                item, :canonical_top2_action_digest
            ),
            overhead_ms=required_property(item, :decision_overhead_ms),
        ) for item in required_property(document, :rows)
    ]
    return (;
        rows,
        artifact_evidence=gate_artifact_evidence(required_property(
            document, :gate_artifact
        )),
        bootstrap_schedules=required_property(document, :calibration_bootstrap_schedules),
        expected_bootstrap_schedule_sha256=String(required_property(
            document, :calibration_bootstrap_schedule_sha256
        )),
        forbidden_seed_used=Bool(required_property(document, :forbidden_seed_used)),
    )
end

function write_fresh_json(path::AbstractString, value)
    output_path = abspath(path)
    ispath(output_path) && error("refusing to overwrite calibration result: $output_path")
    mkpath(dirname(output_path))
    temporary_path = output_path * ".tmp.$(getpid())"
    ispath(temporary_path) && rm(temporary_path; force=true)
    try
        open(temporary_path, "w") do io
            JSON3.pretty(io, value)
            println(io)
            flush(io)
        end
        mv(temporary_path, output_path)
    finally
        ispath(temporary_path) && rm(temporary_path; force=true)
    end
    return output_path
end

function main(args=ARGS)
    length(args) == 3 || error(
        "usage: julia --project=. calibration_gate.jl " *
        "CALIBRATION.json OUTPUT.json EXPECTED_CALIBRATION_SHA256"
    )
    child_milestone("input_load_begin")
    input_path = abspath(args[1])
    input = load_calibration_document(input_path)
    child_milestone("input_load_complete")
    child_milestone("calibration_begin")
    source_evidence = source_hash_evidence(input_path, args[3])
    result = evaluate_calibration(input.rows;
        artifact_evidence=input.artifact_evidence,
        source_evidence,
        bootstrap_schedules=input.bootstrap_schedules,
        expected_bootstrap_schedule_sha256=
            input.expected_bootstrap_schedule_sha256,
        forbidden_seed_used=input.forbidden_seed_used,
    )
    child_milestone("calibration_complete")
    child_milestone("artifact_write_begin")
    output_path = write_fresh_json(args[2], result)
    child_milestone("artifact_write_complete")
    println("R1_CALIBRATION_STATUS=$(result.status)")
    println("R1_CALIBRATION_RESULT=$(output_path)")
    child_milestone("phase_complete")
    return result
end

end # module

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    using .R1CalibrationGate
    R1CalibrationGate.main()
end
