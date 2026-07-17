using Test
using JSON3
using Random
using SHA

include(joinpath(@__DIR__, "ridge_gate.jl"))
using .R1RidgeGate

function synthetic_training_data()
    episodes = repeat(73001:73012; inner=4)
    rows = length(episodes)
    x = zeros(Float64, FEATURE_COUNT, rows)
    for row in 1:rows
        episode = episodes[row] - 73000
        phase = mod(row - 1, 4) + 1
        for feature in 1:FEATURE_COUNT
            x[feature, row] = sin(0.071 * feature * phase) + 0.03 * episode +
                              cos(0.017 * feature * episode)
        end
    end
    x[7, :] .= 4.0 # valid-action-count feature is semantically integral
    x[end, :] .= 0.1 # non-binary constant-feature witness
    advantage = [
        0.45 * x[1, row] - 0.30 * x[8, row] + 0.12 * x[24, row] +
        0.04 * (episodes[row] - 73006) for row in 1:rows
    ]
    advantage[1] = 99.0
    advantage[2] = -99.0
    return x, advantage, collect(episodes)
end

@testset "R1 fixed feature contract" begin
    @test FEATURE_COUNT == 70
    @test COEFFICIENT_COUNT == 71
    @test COEFFICIENT_COUNT < 100
    @test ENSEMBLE_SIZE == 256
    @test RIDGE_LAMBDA == 1.0
    @test UInt64(BOOTSTRAP_SEED) == UInt64(0x5231_2026)
    @test TRAINING_SCHEDULE_DIGEST ==
          "5b60a1e340b542dc8654a5c80777c254d8336aa086e51be6c2ba1251be20e5f7"
    @test LOWER_QUANTILE == 0.10
    @test OVERRIDE_THRESHOLD == 0.05
    @test QUANTILE_METHOD == "linear_type7_position_1_plus_n_minus_1_p"
    @test validate_feature_schema(FEATURE_NAMES)
    @test !validate_feature_schema(reverse(FEATURE_NAMES))
    @test length(unique(FEATURE_NAMES)) == FEATURE_COUNT
    @test length(FEATURE_SCHEMA_DIGEST) == 64
    @test count(startswith("hold_"), FEATURE_NAMES) == 7
    @test count(startswith("next"), FEATURE_NAMES) == 35
end

@testset "deterministic episode-cluster bootstrap ridge" begin
    x, advantage, episodes = synthetic_training_data()
    gate1 = fit_ridge_gate(x, advantage, episodes)
    gate2 = fit_ridge_gate(x, advantage, episodes)
    ids = sort!(unique(episodes))
    schedule_rng = Random.Xoshiro(BOOTSTRAP_SEED)
    frozen_schedules = [
        [ids[rand(schedule_rng, eachindex(ids))] for _ in eachindex(ids)] for
        _ in 1:ENSEMBLE_SIZE
    ]
    gate_from_schedule = fit_ridge_gate(
        x, advantage, episodes; bootstrap_schedules=frozen_schedules,
    )

    @test gate1.feature_mean == gate2.feature_mean
    @test gate1.feature_scale == gate2.feature_scale
    @test gate1.constant_feature == gate2.constant_feature
    @test gate1.coefficients == gate2.coefficients
    @test gate1.coefficients == gate_from_schedule.coefficients
    @test size(gate1.coefficients) == (COEFFICIENT_COUNT, ENSEMBLE_SIZE)
    @test all(isfinite, gate1.coefficients)
    @test gate1.constant_feature[end]
    @test all(standardized_features(gate1, x)[end, :] .== 0.0)
    @test any(gate1.coefficients[:, 1] .!= gate1.coefficients[:, 2])

    # Artifact round-trip keeps every decision-relevant numeric value exact.
    restored = gate_from_payload(gate_payload(gate1))
    @test restored.coefficients == gate1.coefficients
    @test restored.feature_mean == gate1.feature_mean
    @test restored.constant_feature == gate1.constant_feature
    json_restored = gate_from_payload(JSON3.read(JSON3.write(gate_payload(gate1))))
    @test json_restored.coefficients == gate1.coefficients
    @test json_restored.feature_mean == gate1.feature_mean
    bad_schedules = deepcopy(frozen_schedules)
    bad_schedules[1][1] = 91001
    @test_throws ErrorException fit_ridge_gate(
        x, advantage, episodes; bootstrap_schedules=bad_schedules,
    )
    production_artifact = (;
        gate_payload(gate1)...,
        experiment_id="online_counterfactual_top2_R1",
        fit_role="training_only",
        fit_backend="python_numpy_analytic_ridge",
        source_table_synthetic=false,
        training_bootstrap_schedule_consumed=true,
        all_finite=true,
        validation_seed_used=false,
        sealed_test_seed_used=false,
        source_table_sha256=repeat("a", 64),
        source_table_path="unit-test-does-not-exist.json",
        source_collection_manifest_sha256=repeat("d", 64),
        source_collection_manifest_path="unit-test-does-not-exist-manifest.json",
        training_row_order_sha256=repeat("e", 64),
        training_row_order_encoding=TRAINING_ROW_ORDER_ENCODING,
        design_freeze_sha256=repeat("b", 64),
        design_freeze_path="unit-test-does-not-exist-freeze.json",
        training_bootstrap_schedule_sha256=TRAINING_SCHEDULE_DIGEST,
        training_bootstrap_schedule_source_anchor_sha256=TRAINING_SCHEDULE_DIGEST,
    )
    @test gate_from_production_artifact(
        production_artifact; verify_source_files=false,
    ).coefficients ==
          gate1.coefficients
    synthetic_artifact = merge(production_artifact, (source_table_synthetic=true,))
    @test_throws ErrorException gate_from_production_artifact(
        synthetic_artifact; verify_source_files=false,
    )
    wrong_schedule_artifact = merge(
        production_artifact, (training_bootstrap_schedule_sha256=repeat("c", 64),),
    )
    @test_throws ErrorException gate_from_production_artifact(
        wrong_schedule_artifact; verify_source_files=false,
    )
    @test_throws ErrorException gate_from_production_artifact(
        merge(production_artifact, (fit_backend="julia_reference",));
        verify_source_files=false,
    )
    @test_throws ErrorException gate_from_production_artifact(
        merge(production_artifact, (training_row_order_encoding="unspecified",));
        verify_source_files=false,
    )
    @test_throws ErrorException gate_from_production_artifact(
        merge(production_artifact, (
            training_bootstrap_schedule_source_anchor_sha256=repeat("f", 64),
        ));
        verify_source_files=false,
    )
end

@testset "linked production provenance and tamper rejection" begin
    x, advantage, episodes = synthetic_training_data()
    gate = fit_ridge_gate(x, advantage, episodes)
    mktempdir() do directory
        table_path = joinpath(directory, "training.json")
        manifest_path = joinpath(directory, "manifest.json")
        freeze_path = joinpath(directory, "freeze.json")
        rows = [
            (;
                episode_id=episode,
                piece_index=piece,
                root_state_digest=bytes2hex(sha256("root-$(episode)-$(piece)")),
            ) for episode in 73001:73012 for piece in 10:10:240
        ]
        table = (;
            schema_version="r1-training-table-v1",
            source_role="training",
            synthetic=false,
            validation_seed_used=false,
            sealed_test_seed_used=false,
            rows,
        )
        open(table_path, "w") do io
            JSON3.write(io, table)
            write(io, '\n')
        end
        table_sha = bytes2hex(open(sha256, table_path))
        manifest = (;
            schema_version="r1-collection-manifest-v1",
            source_role="training",
            synthetic=false,
            real_model_or_game_loaded=true,
            validation_seed_used=false,
            sealed_test_seed_used=false,
            table_path=abspath(table_path),
            table_sha256=table_sha,
        )
        open(manifest_path, "w") do io
            JSON3.write(io, manifest)
            write(io, '\n')
        end
        write(freeze_path, "{}\n")
        manifest_sha = bytes2hex(open(sha256, manifest_path))
        freeze_sha = bytes2hex(open(sha256, freeze_path))
        row_order_sha = bytes2hex(sha256(codeunits(join((
            "$(row.episode_id),$(row.piece_index),$(row.root_state_digest)" for row in rows
        ), '\n'))))
        artifact = (;
            gate_payload(gate)...,
            experiment_id="online_counterfactual_top2_R1",
            fit_role="training_only",
            fit_backend="python_numpy_analytic_ridge",
            source_table_synthetic=false,
            source_table_path=abspath(table_path),
            source_table_sha256=table_sha,
            source_collection_manifest_path=abspath(manifest_path),
            source_collection_manifest_sha256=manifest_sha,
            training_row_order_sha256=row_order_sha,
            training_row_order_encoding=TRAINING_ROW_ORDER_ENCODING,
            design_freeze_path=abspath(freeze_path),
            design_freeze_sha256=freeze_sha,
            training_bootstrap_schedule_sha256=TRAINING_SCHEDULE_DIGEST,
            training_bootstrap_schedule_source_anchor_sha256=TRAINING_SCHEDULE_DIGEST,
            training_bootstrap_schedule_consumed=true,
            all_finite=true,
            validation_seed_used=false,
            sealed_test_seed_used=false,
        )
        @test gate_from_production_artifact(artifact).coefficients == gate.coefficients
        @test_throws ErrorException gate_from_production_artifact(
            merge(artifact, (training_row_order_sha256=repeat("0", 64),)),
        )
        @test_throws ErrorException gate_from_production_artifact(
            merge(artifact, (source_collection_manifest_sha256=repeat("1", 64),)),
        )
        @test_throws ErrorException gate_from_production_artifact(
            merge(artifact, (source_table_sha256=repeat("2", 64),)),
        )
    end
end

@testset "batch/scalar conformance and strict safe decision" begin
    x, advantage, episodes = synthetic_training_data()
    gate = fit_ridge_gate(x, advantage, episodes)
    probes = x[:, 5:12]
    batch_lower = predict_lower_bounds(gate, probes)
    scalar_lower = [
        predict_lower_bound_scalar_reference(gate, @view probes[:, row]) for
        row in axes(probes, 2)
    ]
    @test batch_lower ≈ scalar_lower rtol=2e-13 atol=2e-13
    @test decide_top2(gate, probes[:, 1]; valid_action_count=4).lower_bound == batch_lower[1]
    @test R1RidgeGate._linear_quantile_sorted(collect(Float64, 0:255), 0.10) == 25.5

    # The threshold is strict: equality remains old top-1.
    zero_coefficients = zeros(Float64, COEFFICIENT_COUNT, ENSEMBLE_SIZE)
    zero_coefficients[1, :] .= OVERRIDE_THRESHOLD
    threshold_gate = RidgeGate(
        collect(FEATURE_NAMES), zeros(FEATURE_COUNT), ones(FEATURE_COUNT), falses(FEATURE_COUNT),
        zero_coefficients, RIDGE_LAMBDA, UInt64(BOOTSTRAP_SEED), LOWER_QUANTILE,
        OVERRIDE_THRESHOLD,
    )
    threshold_features = zeros(FEATURE_COUNT)
    threshold_features[7] = 2.0
    equal_decision = decide_top2(
        threshold_gate, threshold_features; valid_action_count=2,
    )
    @test !equal_decision.use_top2
    @test equal_decision.fallback_reason == :lower_bound_not_above_threshold

    positive_coefficients = copy(zero_coefficients)
    positive_coefficients[1, :] .= nextfloat(OVERRIDE_THRESHOLD)
    positive_gate = RidgeGate(
        collect(FEATURE_NAMES), zeros(FEATURE_COUNT), ones(FEATURE_COUNT), falses(FEATURE_COUNT),
        positive_coefficients, RIDGE_LAMBDA, UInt64(BOOTSTRAP_SEED), LOWER_QUANTILE,
        OVERRIDE_THRESHOLD,
    )
    @test decide_top2(
        positive_gate, threshold_features; valid_action_count=2,
    ).use_top2

    @test decide_top2(gate, probes[:, 1]).fallback_reason == :fewer_than_two_candidates
    @test decide_top2(gate, probes[:, 1]; valid_action_count=1).fallback_reason ==
          :fewer_than_two_candidates
    @test decide_top2(gate, probes[:, 1]; valid_action_count=3).fallback_reason ==
          :candidate_count_mismatch
    @test decide_top2(gate, probes[:, 1]; feature_names=reverse(FEATURE_NAMES)).fallback_reason ==
          :schema_mismatch
    invalid = copy(probes[:, 1])
    invalid[3] = NaN
    @test decide_top2(gate, invalid; valid_action_count=4).fallback_reason == :invalid_features
    @test decide_top2(
        gate, fill("not numeric", FEATURE_COUNT); valid_action_count=4,
    ).fallback_reason ==
          :invalid_features
end

@testset "invalid data fail closed" begin
    x, advantage, episodes = synthetic_training_data()
    @test_throws ErrorException fit_ridge_gate(x[1:end-1, :], advantage, episodes)
    @test_throws ErrorException fit_ridge_gate(x, advantage, episodes[1:end-1])
    bad = copy(x)
    bad[1, 1] = Inf
    @test_throws ErrorException fit_ridge_gate(bad, advantage, episodes)
    @test_throws ErrorException fit_ridge_gate(
        x, advantage, episodes; feature_names=reverse(FEATURE_NAMES),
    )
end
