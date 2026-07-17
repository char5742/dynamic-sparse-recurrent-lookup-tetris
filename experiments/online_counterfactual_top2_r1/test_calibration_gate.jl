using Test
using JSON3
using SHA

include(joinpath(@__DIR__, "calibration_gate.jl"))
using .R1CalibrationGate
include(joinpath(@__DIR__, "ridge_gate.jl"))
using .R1RidgeGate
include(joinpath(@__DIR__, "development_gate.jl"))
using .R1DevelopmentGate

function passing_rows()
    rows = CalibrationRow[]
    for (episode_slot, episode) in enumerate(CALIBRATION_EPISODES)
        for state in 1:20
            override = state <= 2 # 12 / 120 = 10%, two in every episode
            push!(rows, calibration_row(;
                episode_id=episode,
                advantage=override ? 0.8 + 0.01 * episode_slot : -0.2,
                a1_terminal=false,
                a2_terminal=false,
                production_decision=override,
                reference_decision=override,
                production_selected_candidate_index=override ? 2 : 1,
                canonical_top1_candidate_index=1,
                canonical_top2_candidate_index=2,
                production_selected_action_digest=override ?
                    "top2-$(episode)-$(state)" : "top1-$(episode)-$(state)",
                canonical_top1_action_digest="top1-$(episode)-$(state)",
                canonical_top2_action_digest="top2-$(episode)-$(state)",
                overhead_ms=0.05,
            ))
        end
    end
    return rows
end

function replace_row(row::CalibrationRow; kwargs...)
    values = Dict{Symbol,Any}(
        :episode_id => row.episode_id,
        :advantage => row.advantage,
        :a1_terminal => row.a1_terminal,
        :a2_terminal => row.a2_terminal,
        :production_decision => row.production_decision,
        :reference_decision => row.reference_decision,
        :production_selected_candidate_index => row.production_selected_candidate_index,
        :canonical_top1_candidate_index => row.canonical_top1_candidate_index,
        :canonical_top2_candidate_index => row.canonical_top2_candidate_index,
        :production_selected_action_digest => row.production_selected_action_digest,
        :canonical_top1_action_digest => row.canonical_top1_action_digest,
        :canonical_top2_action_digest => row.canonical_top2_action_digest,
        :overhead_ms => row.overhead_ms,
    )
    merge!(values, Dict(kwargs))
    return calibration_row(; values...)
end

function gate_result(rows; kwargs...)
    artifact = gate_artifact_evidence((;
        feature_names=collect(FEATURE_NAMES),
        feature_mean=zeros(Float64, FEATURE_COUNT),
        feature_scale=ones(Float64, FEATURE_COUNT),
        constant_feature=falses(FEATURE_COUNT),
        coefficients=zeros(Float64, COEFFICIENT_COUNT * ENSEMBLE_SIZE),
        coefficient_count=COEFFICIENT_COUNT,
        ensemble_size=ENSEMBLE_SIZE,
        lambda=RIDGE_LAMBDA,
        bootstrap_seed=UInt64(R1RidgeGate.BOOTSTRAP_SEED),
        lower_quantile=LOWER_QUANTILE,
        override_threshold=OVERRIDE_THRESHOLD,
    ))
    actual_test_sha = bytes2hex(open(sha256, @__FILE__))
    options = Dict{Symbol,Any}(
        :artifact_evidence => artifact,
        :source_evidence => source_hash_evidence(@__FILE__, actual_test_sha),
        :bootstrap_schedules => calibration_cluster_schedules(),
        :expected_bootstrap_schedule_sha256 => schedule_digest(
            calibration_cluster_schedules()
        ),
        :forbidden_seed_used => false,
    )
    merge!(options, Dict(kwargs))
    return evaluate_calibration(rows; options...)
end

function episode(seed, score; pieces=250, game_over=false, wall=1.0)
    return development_episode(;
        seed,
        score,
        pieces,
        game_over,
        candidate_count=1_000,
        network_candidate_evaluations=1_000,
        logical_model_passes=250,
        physical_backend_requests=250,
        lookahead_expansions=0,
        median_decision_wall_ms=wall,
    )
end

function synthetic_deployment_decision(; use_top2::Bool)
    top1_index = 1
    top2_index = 3
    selected_index = use_top2 ? top2_index : top1_index
    top1_action = "action-1"
    top2_action = "action-3"
    selected_action = use_top2 ? top2_action : top1_action
    top1_node = "node-1"
    top2_node = "node-3"
    selected_node = use_top2 ? top2_node : top1_node
    binding = bytes2hex(sha256(join((
        "r1-live-ridge-decision-v1",
        string(selected_index),
        selected_action,
        selected_node,
    ), '\n')))
    feature_vector = zeros(Float64, FEATURE_COUNT)
    feature_digest = bytes2hex(sha256(join((
        FEATURE_SCHEMA_DIGEST,
        (bitstring(value) for value in feature_vector)...,
    ), '\n')))
    return DeploymentDecision(
        "r1-live-ridge-decision-v1",
        FEATURE_SCHEMA_DIGEST,
        feature_digest,
        "feature_schema_sha256 newline Float64-bitstring-per-feature newline joined",
        feature_vector,
        top1_index,
        top2_index,
        selected_index,
        top1_action,
        top2_action,
        selected_action,
        top1_node,
        top2_node,
        selected_node,
        binding,
        selected_action,
        selected_node,
        use_top2,
        use_top2 ? nextfloat(OVERRIDE_THRESHOLD) : OVERRIDE_THRESHOLD,
        use_top2 ? :none : :lower_bound_not_above_threshold,
        "root-digest",
        "root-digest",
        "root-digest",
        "clone-digest",
        UInt64(10_000),
        UInt64(10_375),
        UInt64(375),
        "feature_build+ridge_eval+selection_binding;" *
        "excludes_candidate_enumeration+old_q_evaluation+artifact_load+clone_apply_verification",
    )
end

function decision_namedtuple(decision::DeploymentDecision)
    names = fieldnames(DeploymentDecision)
    return NamedTuple{names}(Tuple(getfield(decision, name) for name in names))
end

@testset "R1 exact calibration promotion gate" begin
    rows = passing_rows()
    result = gate_result(rows)
    @test result.promoted
    @test result.status == "R1-calibration-promoted"
    @test result.override_count == 12
    @test result.override_rate == 0.10
    @test result.override_episode_count == 6
    @test result.override_precision == 1.0
    @test result.bootstrap.precision_lower90 == 1.0
    @test result.bootstrap.mean_advantage_lower90 > 0.0
    @test result.bootstrap.replicates == 2_000
    @test result.bootstrap.seed == UInt64(0x5231_73106)
    @test !result.game_strength_evidence
    @test !result.model_improvement_evidence

    mismatch = copy(rows)
    index = findfirst(row -> !row.production_decision, mismatch)
    mismatch[index] = replace_row(mismatch[index]; reference_decision=true)
    rejected = gate_result(mismatch)
    @test !rejected.promoted
    @test !rejected.checks.production_reference_exact

    unsafe = copy(rows)
    unsafe[1] = replace_row(unsafe[1]; a2_terminal=true)
    rejected = gate_result(unsafe)
    @test !rejected.checks.no_top2_only_terminal

    bad_fallback = copy(rows)
    index = findfirst(row -> !row.production_decision, bad_fallback)
    bad_fallback[index] = replace_row(bad_fallback[index];
        production_selected_candidate_index=2,
        production_selected_action_digest=bad_fallback[index].canonical_top2_action_digest,
    )
    @test !gate_result(bad_fallback).checks.fallback_top1_exact

    slow = [replace_row(row; overhead_ms=0.101) for row in rows]
    @test !gate_result(slow).checks.overhead_within_budget
    @test !gate_result(rows; forbidden_seed_used=true).checks.no_forbidden_seed_used
    @test !gate_result(rows;
        artifact_evidence=GateArtifactEvidence(false, true, true, true, 1),
    ).checks.artifact_finite
    @test !gate_result(rows;
        source_evidence=source_hash_evidence(@__FILE__, repeat("0", 64)),
    ).checks.calibration_source_hash_verified

    # Keep 12 overrides but concentrate them in three episodes. The point
    # metrics look strong; the episode-distribution requirement still rejects.
    concentrated = CalibrationRow[]
    for row in rows
        state_in_episode = count(item -> item.episode_id == row.episode_id, concentrated) + 1
        episode_slot = findfirst(==(row.episode_id), CALIBRATION_EPISODES)
        override = episode_slot <= 3 && state_in_episode <= 4
        push!(concentrated, replace_row(row;
            production_decision=override,
            reference_decision=override,
            production_selected_candidate_index=override ?
                row.canonical_top2_candidate_index : row.canonical_top1_candidate_index,
            production_selected_action_digest=override ?
                row.canonical_top2_action_digest : row.canonical_top1_action_digest,
            advantage=override ? 0.8 : -0.2,
        ))
    end
    rejected = gate_result(concentrated)
    @test rejected.override_count == 12
    @test !rejected.checks.override_episode_distribution
end

@testset "R1 independent scalar/production decision comparison" begin
    features = zeros(Float64, 3, 120)
    features[1, :] .= 1:120
    episode_ids = repeat(CALIBRATION_EPISODES; inner=20)
    advantages = fill(-0.2, 120)
    advantages[1:20:120] .= 0.8
    advantages[2:20:120] .= 0.8
    terminals = falses(120)
    overhead = fill(0.03, 120)
    top1_indices = ones(Int, 120)
    top2_indices = fill(2, 120)
    production_indices = [((row - 1) % 20 < 2) ? 2 : 1 for row in 1:120]
    top1_digests = ["top1-$row" for row in 1:120]
    top2_digests = ["top2-$row" for row in 1:120]
    production_digests = [production_indices[row] == 2 ?
        top2_digests[row] : top1_digests[row] for row in 1:120]
    production(x) = [((row - 1) % 20 < 2) ? 0.051 : 0.05 for row in axes(x, 2)]
    reference(x) = ((Int(x[1]) - 1) % 20 < 2) ? 0.051 : 0.05
    rows = calibration_rows_from_gate(
        features,
        episode_ids,
        advantages,
        terminals,
        terminals,
        production_indices,
        top1_indices,
        top2_indices,
        production_digests,
        top1_digests,
        top2_digests,
        overhead;
        production_lower_bounds=production,
        reference_lower_bound=reference,
    )
    @test count(row -> row.production_decision, rows) == 12
    @test all(row.production_decision == row.reference_decision for row in rows)
    # Equality at 0.05 is deliberately a fallback, not an override.
    @test !rows[3].production_decision
end

@testset "R1 live deployment producer fields are measured and self-consistent" begin
    fallback = deployment_calibration_fields(synthetic_deployment_decision(use_top2=false))
    @test !fallback.production_decision
    @test fallback.production_selected_candidate_index == 1
    @test fallback.production_selected_action_digest == "action-1"
    @test fallback.production_selected_node_identity == "node-1"
    @test length(fallback.production_feature_vector) == FEATURE_COUNT
    @test fallback.production_gate_incremental_elapsed_ns == 375
    @test fallback.production_gate_incremental_finished_ns -
          fallback.production_gate_incremental_started_ns ==
          fallback.production_gate_incremental_elapsed_ns

    override_decision = synthetic_deployment_decision(use_top2=true)
    override = deployment_calibration_fields(override_decision)
    @test override.production_decision
    @test override.production_selected_candidate_index == 3
    @test override.production_selected_action_digest == "action-3"
    @test override.production_selected_node_identity == "node-3"
    @test override.production_applied_action_digest == override.production_selected_action_digest
    @test override.canonical_state_digest_before == override.canonical_state_digest_after
    @test override.production_clone_state_digest_before ==
          override.canonical_state_digest_before

    raw = decision_namedtuple(override_decision)
    @test_throws ErrorException deployment_calibration_fields(merge(
        raw, (; gate_incremental_elapsed_ns=UInt64(374)),
    ))
    @test_throws ErrorException deployment_calibration_fields(merge(
        raw, (; applied_node_identity="node-1"),
    ))
    @test_throws ErrorException deployment_calibration_fields(merge(
        raw, (; canonical_state_digest_after="mutated-root"),
    ))
    @test_throws ErrorException deployment_calibration_fields(merge(
        raw, (; selection_binding_sha256=repeat("0", 64)),
    ))
    @test_throws ErrorException deployment_calibration_fields(merge(
        raw, (; feature_vector=zeros(Float64, FEATURE_COUNT - 1)),
    ))
    @test_throws ErrorException deployment_calibration_fields(merge(
        raw, (; lower_bound=OVERRIDE_THRESHOLD),
    ))
end

@testset "R1 development screen is opt-in and sequential" begin
    disabled = evaluate_development_stage(DevelopmentEpisode[], DevelopmentEpisode[])
    @test disabled.status == "R1-development-disabled"
    @test !disabled.input_consumed

    baseline_first = [episode(5756, 10_000)]
    losing_first = [episode(5756, 10_000)]
    rejected = evaluate_development_stage(baseline_first, losing_first;
        enabled=true,
        calibration_status="R1-calibration-promoted",
        frozen_artifacts_verified=true,
    )
    @test rejected.status == "R1-development-rejected"
    @test !rejected.authorize_second_pair

    winning_first = [episode(5756, 10_600; wall=1.04)]
    continued = evaluate_development_stage(baseline_first, winning_first;
        enabled=true,
        calibration_status="R1-calibration-promoted",
        frozen_artifacts_verified=true,
    )
    @test continued.status == "R1-development-second-pair-authorized"
    @test continued.authorize_second_pair
    @test !continued.promoted

    baseline = [episode(5756, 10_000), episode(5757, 9_000)]
    candidate = [episode(5756, 10_600; wall=1.04), episode(5757, 9_500; wall=1.04)]
    promoted = evaluate_development_stage(baseline, candidate;
        enabled=true,
        calibration_status="R1-calibration-promoted",
        frozen_artifacts_verified=true,
    )
    @test promoted.status == "R1-development-promoted-system-candidate"
    @test promoted.promoted
    @test promoted.mean_score_difference == 550
    @test !promoted.model_improvement_evidence
    @test !promoted.g2_evidence

    @test_throws ErrorException evaluate_development_stage(
        baseline,
        [episode(5756, 10_000), episode(5757, 9_500)];
        enabled=true,
        calibration_status="R1-calibration-promoted",
        frozen_artifacts_verified=true,
    )
end

@testset "R1 calibration production CLI and early milestones" begin
    mktempdir() do temporary
        input_path = joinpath(temporary, "calibration.json")
        output_path = joinpath(temporary, "result.json")
        milestone_path = joinpath(temporary, "calibration.child.jsonl")
        rows = passing_rows()
        document = (;
            source_role="calibration",
            forbidden_seed_used=false,
            gate_artifact=(;
                feature_names=collect(FEATURE_NAMES),
                feature_mean=zeros(Float64, FEATURE_COUNT),
                feature_scale=ones(Float64, FEATURE_COUNT),
                constant_feature=falses(FEATURE_COUNT),
                coefficients=zeros(Float64, COEFFICIENT_COUNT * ENSEMBLE_SIZE),
                coefficient_count=COEFFICIENT_COUNT,
                ensemble_size=ENSEMBLE_SIZE,
                lambda=RIDGE_LAMBDA,
                bootstrap_seed=UInt64(R1RidgeGate.BOOTSTRAP_SEED),
                lower_quantile=LOWER_QUANTILE,
                override_threshold=OVERRIDE_THRESHOLD,
            ),
            calibration_bootstrap_schedules=calibration_cluster_schedules(),
            calibration_bootstrap_schedule_sha256=schedule_digest(
                calibration_cluster_schedules()
            ),
            rows=[(;
                episode_id=row.episode_id,
                advantage_unclipped_A6=row.advantage,
                a1_terminal_within_horizon=row.a1_terminal,
                a2_terminal_within_horizon=row.a2_terminal,
                production_decision=row.production_decision,
                reference_decision=row.reference_decision,
                production_selected_candidate_index=
                    row.production_selected_candidate_index,
                canonical_top1_candidate_index=row.canonical_top1_candidate_index,
                canonical_top2_candidate_index=row.canonical_top2_candidate_index,
                production_selected_action_digest=
                    row.production_selected_action_digest,
                canonical_top1_action_digest=row.canonical_top1_action_digest,
                canonical_top2_action_digest=row.canonical_top2_action_digest,
                decision_overhead_ms=row.overhead_ms,
            ) for row in rows],
        )
        open(input_path, "w") do io
            JSON3.write(io, document)
        end
        project_root = normpath(joinpath(@__DIR__, "..", ".."))
        input_sha = bytes2hex(open(sha256, input_path))
        command = `$(Base.julia_cmd()) --project=$(project_root) $(joinpath(@__DIR__, "calibration_gate.jl")) $(input_path) $(output_path) $(input_sha)`
        run(addenv(command, "R1_CHILD_MILESTONE_PATH" => milestone_path))
        result = JSON3.read(read(output_path, String))
        @test result.status == "R1-calibration-promoted"
        milestones = [JSON3.read(line) for line in readlines(milestone_path)]
        stages = String.(getproperty.(milestones, :stage))
        @test stages == [
            "script_enter",
            "imports_begin",
            "imports_complete",
            "input_load_begin",
            "input_load_complete",
            "calibration_begin",
            "calibration_complete",
            "artifact_write_begin",
            "artifact_write_complete",
            "phase_complete",
        ]
        @test length(unique(Int.(getproperty.(milestones, :pid)))) == 1
    end

    # Without the explicit enable switch, even a positional nonexistent path
    # is ignored rather than loaded.
    project_root = normpath(joinpath(@__DIR__, "..", ".."))
    missing_path = joinpath(tempdir(), "R1-must-not-be-read-$(time_ns()).json")
    command = `$(Base.julia_cmd()) --project=$(project_root) $(joinpath(@__DIR__, "development_gate.jl")) $(missing_path)`
    output = read(command, String)
    @test occursin("R1-development-disabled", output)
    @test !ispath(missing_path)
end

println("R1 calibration/development synthetic tests passed")
