using Test

include(joinpath(@__DIR__, "n1_logistic_gate.jl"))
using .N1LogisticGate

function training_rows(; positives=40, rows=160, variable=false)
    result = NamedTuple[]
    for index in 1:rows
        z1 = zeros(Float64, FEATURE_COUNT)
        z2 = zeros(Float64, FEATURE_COUNT)
        if variable
            z2[1] = (index - (rows + 1) / 2) / rows
            z2[2] = sin(index)
        end
        push!(result, (; z1, z2, y=index <= positives))
    end
    return result
end

function calibration_rows(;
    overrides=Set((episode, piece) for episode in 1:4 for piece in 1:2),
    override_advantage=0.5,
    fallback_identical=true,
    gate_ns=500_000,
)
    rows = NamedTuple[]
    for episode in 1:6, piece in 1:10
        override = (episode, piece) in overrides
        advantage = override ? override_advantage : -0.25
        push!(rows, (;
            episode_id=episode,
            sample_piece=piece * 10,
            p=override ? 0.95 : 0.10,
            selected_top2=override,
            y=advantage > 0.0,
            advantage,
            a1_terminal=false,
            a2_terminal=false,
            fallback_identical=override ? false : fallback_identical,
            gate_ns,
        ))
    end
    return rows
end

@testset "N1 frozen logistic Newton/Cholesky fit" begin
    # All 64 columns are constant: population scales and standardized values
    # must be exact zero.  The unpenalized intercept has the closed-form MLE.
    fit = fit_training_gate(training_rows())
    @test fit.iterations <= 100
    @test fit.gradient_infinity_norm <= 1.0e-10
    @test fit.positive_fraction == 0.25
    @test all(iszero, fit.model.standardizer.scale)
    @test all(iszero, standardize(fit.model.standardizer, fill(123.0, 64)))
    @test all(iszero, fit.model.weights)
    @test fit.model.lambda == 1.0
    @test fit.model.intercept ≈ log(0.25 / 0.75) atol=1.0e-11
    @test predict_probability(fit.model, zeros(64)) ≈ 0.25 atol=1.0e-12

    # A nonconstant fixture exercises standardized slopes and is bitwise
    # deterministic across two independent zero-start fits.
    variable_rows = training_rows(; variable=true)
    first = fit_training_gate(variable_rows)
    second = fit_training_gate(variable_rows)
    @test first.model.weights == second.model.weights
    @test first.model.intercept == second.model.intercept
    @test first.objective == second.objective
    @test first.gradient_infinity_norm <= 1.0e-10
    @test first.objective < 160 * log(2)

    @test_throws ErrorException fit_training_gate(training_rows(; rows=149))
    @test_throws ErrorException fit_training_gate(
        training_rows(; positives=4), # 2.5%, below inclusive 3% minimum.
    )
    @test_throws ErrorException fit_training_gate(
        training_rows(; positives=65), # 40.625%, above maximum.
    )
end

@testset "N1 inclusive deployment gate" begin
    @test select_top2(0.90)
    @test !select_top2(prevfloat(0.90))
    @test !select_top2(NaN)
    standardizer = N1LogisticGate.FeatureStandardizer(zeros(64), ones(64))
    model = N1LogisticGate.LogisticGateModel(
        standardizer, zeros(64), log(9.0), 1.0,
    )
    decision = deployment_decision(model, zeros(64), zeros(64))
    @test decision.probability ≈ 0.9
    # Exercise the probability rule itself above; libm rounding of log(9) is
    # not used as the definition of equality at the frozen threshold.
    @test decision.selected_top2 == select_top2(decision.probability)
end

@testset "N1 Type-7 quantile and episode bootstrap" begin
    @test type7_quantile(collect(1:5), 0.10) == 1.4
    @test type7_quantile(fill(-Inf, 5), 0.10) == -Inf

    rows = calibration_rows()
    first = assess_calibration(rows)
    second = assess_calibration(rows)
    @test first == second
    @test first.promoted
    @test first.status == "N1-calibration-promoted"
    @test first.row_count == 60
    @test first.episode_count == 6
    @test first.override_count == 8
    @test first.override_rate ≈ 8 / 60
    @test first.override_episode_count == 4
    @test first.override_precision == 1.0
    @test first.override_mean_G12_advantage == 0.5
    @test first.bootstrap.replicate_count == 10_000
    @test first.bootstrap.seed == UInt64(0x4e31_2026)
    @test first.bootstrap.precision_lower90 == 1.0
    @test first.bootstrap.mean_advantage_lower90 == 0.5
    @test first.bootstrap.empty_override_replicates > 0
    @test first.median_gate_ns == 500_000

    # A fixed no-override prediction set must never be refit in a bootstrap
    # replicate and receives fail-closed -Inf override-only statistics.
    no_overrides = calibration_rows(; overrides=Set{Tuple{Int,Int}}())
    rejected = assess_calibration(no_overrides)
    @test !rejected.promoted
    @test rejected.override_precision == -Inf
    @test rejected.override_mean_G12_advantage == -Inf
    @test rejected.bootstrap.empty_override_replicates == 10_000
    @test rejected.bootstrap.precision_lower90 == -Inf
    @test rejected.bootstrap.mean_advantage_lower90 == -Inf
end

@testset "N1 calibration fail-closed checks" begin
    base = calibration_rows(; gate_ns=1_000_000)
    @test assess_calibration(base).checks.median_gate_time # inclusive 1 ms

    seven = Set((episode, piece) for episode in 1:3 for piece in 1:2)
    push!(seven, (4, 1))
    too_few = assess_calibration(calibration_rows(; overrides=seven))
    @test !too_few.checks.minimum_override_count

    unsafe = copy(base)
    unsafe[1] = merge(unsafe[1], (; a2_terminal=true, a1_terminal=false))
    unsafe_result = assess_calibration(unsafe)
    @test !unsafe_result.checks.no_unsafe_top2_terminal

    nonidentical = calibration_rows(; fallback_identical=false)
    @test !assess_calibration(nonidentical).checks.fallback_bit_identical

    slow = calibration_rows(; gate_ns=1_000_001)
    @test !assess_calibration(slow).checks.median_gate_time

    inconsistent = copy(base)
    inconsistent[1] = merge(inconsistent[1], (; selected_top2=false))
    @test_throws ErrorException assess_calibration(inconsistent)

    duplicate = copy(base)
    duplicate[2] = merge(
        duplicate[2],
        (; episode_id=duplicate[1].episode_id,
           sample_piece=duplicate[1].sample_piece),
    )
    @test_throws ErrorException assess_calibration(duplicate)
end
