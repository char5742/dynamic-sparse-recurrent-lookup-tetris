module R1DevelopmentGate

using JSON3
using Statistics

export DevelopmentEpisode,
       development_episode,
       evaluate_development_stage,
       main

const FIRST_DEVELOPMENT_SEED = 5756
const SECOND_DEVELOPMENT_SEED = 5757
const EPISODE_PIECES = 250
const MINIMUM_MEAN_SCORE_IMPROVEMENT = 500.0
const MAXIMUM_MEDIAN_DECISION_WALL_OVERHEAD = 0.05

struct DevelopmentEpisode
    seed::Int
    score::Int
    pieces::Int
    game_over::Bool
    candidate_count::Int
    network_candidate_evaluations::Int
    logical_model_passes::Int
    physical_backend_requests::Int
    lookahead_expansions::Int
    median_decision_wall_ms::Float64
end

function development_episode(;
    seed,
    score,
    pieces,
    game_over,
    candidate_count,
    network_candidate_evaluations,
    logical_model_passes,
    physical_backend_requests,
    lookahead_expansions,
    median_decision_wall_ms,
)
    episode = DevelopmentEpisode(
        Int(seed),
        Int(score),
        Int(pieces),
        Bool(game_over),
        Int(candidate_count),
        Int(network_candidate_evaluations),
        Int(logical_model_passes),
        Int(physical_backend_requests),
        Int(lookahead_expansions),
        Float64(median_decision_wall_ms),
    )
    episode.seed in (FIRST_DEVELOPMENT_SEED, SECOND_DEVELOPMENT_SEED) ||
        error("unregistered R1 development seed $(episode.seed)")
    episode.pieces <= EPISODE_PIECES || error("development episode exceeds piece limit")
    episode.score >= 0 || error("negative development score")
    episode.candidate_count >= 0 || error("negative candidate count")
    episode.network_candidate_evaluations >= 0 || error(
        "negative network candidate evaluations"
    )
    episode.logical_model_passes >= 0 || error("negative logical model passes")
    episode.physical_backend_requests >= 0 || error("negative backend requests")
    episode.lookahead_expansions >= 0 || error("negative lookahead expansions")
    isfinite(episode.median_decision_wall_ms) || error("non-finite decision wall")
    episode.median_decision_wall_ms > 0 || error("non-positive decision wall")
    return episode
end

completed(episode::DevelopmentEpisode) =
    episode.pieces == EPISODE_PIECES && !episode.game_over

function paired_budget_exact(baseline::DevelopmentEpisode, candidate::DevelopmentEpisode)
    return baseline.candidate_count == candidate.candidate_count &&
           baseline.network_candidate_evaluations ==
               candidate.network_candidate_evaluations &&
           baseline.logical_model_passes == candidate.logical_model_passes &&
           baseline.physical_backend_requests == candidate.physical_backend_requests &&
           baseline.lookahead_expansions == 0 && candidate.lookahead_expansions == 0
end

"""Evaluate the sequential two-seed R1 development screen.

The function is disabled unless `enabled=true`. One positive 5756 pair only
authorizes collecting 5757; it is not a promotion. Supplying 5757 after a
non-positive 5756 difference is a contract violation, because the preregistered
early stop should have prevented that seed from being run.
"""
function evaluate_development_stage(
    baseline::AbstractVector{DevelopmentEpisode},
    candidate::AbstractVector{DevelopmentEpisode};
    enabled::Bool=false,
    calibration_status::AbstractString="",
    frozen_artifacts_verified::Bool=false,
)
    if !enabled
        return (;
            status="R1-development-disabled",
            promoted=false,
            authorize_second_pair=false,
            input_consumed=false,
            reason="explicit enable is required after R1-calibration-promoted",
            validation_seed_used=false,
            sealed_test_seed_used=false,
        )
    end
    calibration_status == "R1-calibration-promoted" || error(
        "development screen requires R1-calibration-promoted"
    )
    frozen_artifacts_verified || error("development artifacts are not frozen")
    length(baseline) == length(candidate) || error("development pair count mismatch")
    length(baseline) in (1, 2) || error("development stage requires one or two pairs")
    baseline_seeds = getproperty.(baseline, :seed)
    candidate_seeds = getproperty.(candidate, :seed)
    expected_seeds = length(baseline) == 1 ? [FIRST_DEVELOPMENT_SEED] :
                     [FIRST_DEVELOPMENT_SEED, SECOND_DEVELOPMENT_SEED]
    baseline_seeds == expected_seeds || error("baseline development seed/order mismatch")
    candidate_seeds == expected_seeds || error("candidate development seed/order mismatch")

    differences = [candidate[i].score - baseline[i].score for i in eachindex(baseline)]
    if differences[1] <= 0
        length(differences) == 1 || error(
            "5757 was consumed despite the mandatory non-positive 5756 early stop"
        )
        return (;
            status="R1-development-rejected",
            promoted=false,
            authorize_second_pair=false,
            input_consumed=true,
            seeds=expected_seeds,
            score_differences=differences,
            reason="first paired difference was non-positive",
            validation_seed_used=false,
            sealed_test_seed_used=false,
        )
    end

    budget_checks = [paired_budget_exact(baseline[i], candidate[i]) for i in eachindex(baseline)]
    all(budget_checks) || error("old network/candidate/request budget mismatch")
    overheads = [
        candidate[i].median_decision_wall_ms / baseline[i].median_decision_wall_ms - 1
        for i in eachindex(baseline)
    ]
    median_overhead = median(overheads)
    if length(differences) == 1
        return (;
            status="R1-development-second-pair-authorized",
            promoted=false,
            authorize_second_pair=true,
            input_consumed=true,
            seeds=expected_seeds,
            score_differences=differences,
            budget_checks,
            median_decision_wall_overhead_fraction=median_overhead,
            reason="positive first pair; no strength claim before 5757",
            validation_seed_used=false,
            sealed_test_seed_used=false,
        )
    end

    mean_difference = mean(differences)
    completion_nonregression = count(completed, candidate) >= count(completed, baseline)
    checks = (;
        both_differences_positive=all(>(0), differences),
        minimum_mean_improvement=mean_difference >= MINIMUM_MEAN_SCORE_IMPROVEMENT,
        completion_nonregression,
        exact_compute_budget=all(budget_checks),
        decision_wall_overhead=median_overhead <=
                               MAXIMUM_MEDIAN_DECISION_WALL_OVERHEAD,
    )
    promoted = all(values(checks))
    return (;
        status=promoted ? "R1-development-promoted-system-candidate" :
                          "R1-development-rejected",
        promoted,
        authorize_second_pair=false,
        input_consumed=true,
        scope="two_seed_development_screen_only_not_G2_not_model_improvement",
        seeds=expected_seeds,
        baseline_scores=getproperty.(baseline, :score),
        candidate_scores=getproperty.(candidate, :score),
        score_differences=differences,
        mean_score_difference=mean_difference,
        baseline_completion_count=count(completed, baseline),
        candidate_completion_count=count(completed, candidate),
        median_decision_wall_overhead_fraction=median_overhead,
        checks,
        validation_seed_used=false,
        sealed_test_seed_used=false,
        model_improvement_evidence=false,
        g2_evidence=false,
    )
end

required_property(object, name::Symbol) = hasproperty(object, name) ?
    getproperty(object, name) : error("missing required property: $name")

function episode_from_json(item)
    return development_episode(;
        seed=required_property(item, :seed),
        score=required_property(item, :score),
        pieces=required_property(item, :pieces),
        game_over=required_property(item, :game_over),
        candidate_count=required_property(item, :candidate_count),
        network_candidate_evaluations=required_property(
            item, :network_candidate_evaluations
        ),
        logical_model_passes=required_property(item, :logical_model_passes),
        physical_backend_requests=required_property(item, :physical_backend_requests),
        lookahead_expansions=required_property(item, :lookahead_expansions),
        median_decision_wall_ms=required_property(item, :median_decision_wall_ms),
    )
end

function main(args=ARGS)
    enabled = "--enable-development-screen" in args
    if !enabled
        # Crucially, no result path is parsed or opened in the default mode.
        result = evaluate_development_stage(DevelopmentEpisode[], DevelopmentEpisode[])
        println(JSON3.write(result))
        return result
    end
    positional = filter(!=("--enable-development-screen"), args)
    length(positional) == 1 || error(
        "usage: development_gate.jl [--enable-development-screen INPUT.json]"
    )
    document = JSON3.read(read(abspath(only(positional)), String))
    baseline = DevelopmentEpisode[
        episode_from_json(item) for item in required_property(document, :baseline)
    ]
    candidate = DevelopmentEpisode[
        episode_from_json(item) for item in required_property(document, :candidate)
    ]
    result = evaluate_development_stage(baseline, candidate;
        enabled=true,
        calibration_status=String(required_property(document, :calibration_status)),
        frozen_artifacts_verified=Bool(required_property(
            document, :frozen_artifacts_verified
        )),
    )
    println(JSON3.write(result))
    return result
end

end # module

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    using .R1DevelopmentGate
    R1DevelopmentGate.main()
end
