using Test
using Random
using LinearAlgebra
using Statistics

include(joinpath(@__DIR__, "CanonicalListNet.jl"))
using .CanonicalListNet

function canonical_objective(student_q, teacher_q, counts, config)
    scratch = ListNetScratch(size(student_q, 1), eltype(student_q))
    gradient = similar(student_q)
    return listnet_loss_and_gradient!(
        gradient,
        scratch,
        student_q,
        teacher_q,
        counts,
        config,
    ).total_loss
end

@testset "canonical teacher-scale ListNet invariants" begin
    teacher = reshape(Float64[3.0, 0.5, -2.0, 17.0], 4, 1)
    student = reshape(Float64[0.7, -0.2, 0.1, -41.0], 4, 1)
    counts = Int[3]
    rank_only = ListNetConfig(
        Float64;
        temperature=0.7,
        scale_floor=0.05,
        q_huber_weight=0.0,
    )
    scratch = ListNetScratch(4, Float64)
    gradient = fill(NaN, 4, 1)
    baseline = listnet_loss_and_gradient!(
        gradient,
        scratch,
        student,
        teacher,
        counts,
        rank_only,
    )
    baseline_gradient = copy(gradient)

    shifted = copy(student)
    shifted[1:3, 1] .+= 37.0
    shifted_result = listnet_loss_and_gradient!(
        gradient,
        scratch,
        shifted,
        teacher,
        counts,
        rank_only,
    )
    @test shifted_result.listnet_kl ≈ baseline.listnet_kl atol=2.0e-14
    @test gradient ≈ baseline_gradient atol=2.0e-14
    @test gradient[4, 1] == 0.0
    @test abs(sum(@view gradient[1:3, 1])) < 2.0e-15

    matched = copy(teacher)
    matched_result = listnet_loss_and_gradient!(
        gradient,
        scratch,
        matched,
        teacher,
        counts,
        rank_only,
    )
    scaled = copy(teacher)
    scaled[1:3, 1] .*= 1.8
    scaled_result = listnet_loss_and_gradient!(
        gradient,
        scratch,
        scaled,
        teacher,
        counts,
        rank_only,
    )
    @test abs(matched_result.listnet_kl) < 2.0e-15
    @test scaled_result.listnet_kl > matched_result.listnet_kl + 1.0e-3
end

@testset "candidate permutation and teacher ties" begin
    teacher = Float64[
        2.0  1.5;
       -0.5  1.5;
        1.0  1.5;
        0.25 1.5;
    ]
    student = Float64[
        0.2  1.5;
        0.8  1.5;
       -0.4  1.5;
        1.1  1.5;
    ]
    counts = Int[4, 4]
    config = ListNetConfig(
        Float64;
        temperature=0.9,
        scale_floor=0.1,
        q_huber_weight=0.3,
        huber_delta=0.75,
    )
    scratch = ListNetScratch(4, Float64)
    gradient = zeros(4, 2)
    original = listnet_loss_and_gradient!(
        gradient,
        scratch,
        student,
        teacher,
        counts,
        config,
    )
    original_gradient = copy(gradient)

    permutation = Int[3, 1, 4, 2]
    permuted_student = student[permutation, :]
    permuted_teacher = teacher[permutation, :]
    permuted_gradient = similar(gradient)
    permuted = listnet_loss_and_gradient!(
        permuted_gradient,
        scratch,
        permuted_student,
        permuted_teacher,
        counts,
        config,
    )
    @test permuted.total_loss ≈ original.total_loss atol=2.0e-14
    @test permuted.listnet_kl ≈ original.listnet_kl atol=2.0e-14
    @test permuted.q_huber_loss ≈ original.q_huber_loss atol=2.0e-14
    @test permuted_gradient ≈ original_gradient[permutation, :] atol=2.0e-14

    tie_teacher = fill(2.25, 4, 1)
    tie_student = copy(tie_teacher)
    tie_counts = Int[4]
    tie_gradient = zeros(4, 1)
    tie = listnet_loss_and_gradient!(
        tie_gradient,
        ListNetScratch(4, Float64),
        tie_student,
        tie_teacher,
        tie_counts,
        config,
    )
    @test abs(tie.listnet_kl) < 2.0e-15
    @test tie.q_huber_loss == 0.0
    @test tie.total_loss ≈ 0.0 atol=2.0e-15
    @test tie_gradient == zeros(4, 1)
end

@testset "dueling composition and exact pullback" begin
    value = Float64[1.25, -0.75]
    advantage = Float64[
        2.0  -1.0;
       -1.0   3.0;
        0.5   0.25;
       99.0  -0.5;
    ]
    counts = Int[3, 4]
    q = fill(NaN, 4, 2)
    compose_dueling_q!(q, value, advantage, counts)
    @test mean(@view q[1:3, 1]) ≈ value[1] atol=2.0e-15
    @test mean(@view q[1:4, 2]) ≈ value[2] atol=2.0e-15
    @test q[4, 1] == 0.0

    q_gradient = Float64[
        0.2  -0.4;
       -0.7   0.5;
        1.1   0.8;
       17.0  -0.2;
    ]
    value_gradient = zeros(2)
    advantage_gradient = fill(NaN, 4, 2)
    dueling_pullback!(
        value_gradient,
        advantage_gradient,
        q_gradient,
        counts,
    )
    @test value_gradient[1] ≈ sum(@view q_gradient[1:3, 1])
    @test value_gradient[2] ≈ sum(@view q_gradient[1:4, 2])
    @test abs(sum(@view advantage_gradient[1:3, 1])) < 2.0e-15
    @test abs(sum(@view advantage_gradient[1:4, 2])) < 2.0e-15
    @test advantage_gradient[4, 1] == 0.0

    epsilon = 1.0e-6
    for state in eachindex(value)
        plus = copy(value)
        minus = copy(value)
        plus[state] += epsilon
        minus[state] -= epsilon
        q_plus = similar(q)
        q_minus = similar(q)
        compose_dueling_q!(q_plus, plus, advantage, counts)
        compose_dueling_q!(q_minus, minus, advantage, counts)
        numerical = sum(q_gradient .* (q_plus .- q_minus)) / (2epsilon)
        @test numerical ≈ value_gradient[state] rtol=2.0e-9 atol=2.0e-9
    end
    for state in 1:2, candidate in 1:Int(counts[state])
        plus = copy(advantage)
        minus = copy(advantage)
        plus[candidate, state] += epsilon
        minus[candidate, state] -= epsilon
        q_plus = similar(q)
        q_minus = similar(q)
        compose_dueling_q!(q_plus, value, plus, counts)
        compose_dueling_q!(q_minus, value, minus, counts)
        numerical = sum(q_gradient .* (q_plus .- q_minus)) / (2epsilon)
        @test numerical ≈ advantage_gradient[candidate, state] rtol=2.0e-9 atol=2.0e-9
    end
end

@testset "canonical loss and complete dueling derivative finite difference" begin
    rng = MersenneTwister(0xc41157)
    width = 5
    states = 3
    counts = Int[5, 3, 4]
    teacher = randn(rng, Float64, width, states)
    value = 0.3 .* randn(rng, Float64, states)
    advantage = 0.4 .* randn(rng, Float64, width, states)
    # Exercise both the quadratic and linear Huber regions without landing on
    # the nondifferentiable boundary.
    teacher[1, 1] += 2.5
    config = ListNetConfig(
        Float64;
        temperature=0.8,
        scale_floor=0.07,
        q_huber_weight=0.35,
        huber_delta=0.6,
    )
    q = zeros(Float64, width, states)
    q_gradient = similar(q)
    value_gradient = similar(value)
    advantage_gradient = similar(advantage)
    scratch = ListNetScratch(width, Float64)
    result = dueling_listnet_loss_and_gradient!(
        value_gradient,
        advantage_gradient,
        q,
        q_gradient,
        scratch,
        value,
        advantage,
        teacher,
        counts,
        config,
    )
    @test isfinite(result.total_loss)
    @test result.valid_candidates == sum(counts)

    epsilon = 2.0e-6
    objective(v, a) = begin
        local_q = zeros(Float64, width, states)
        compose_dueling_q!(local_q, v, a, counts)
        canonical_objective(local_q, teacher, counts, config)
    end
    for state in eachindex(value)
        plus = copy(value)
        minus = copy(value)
        plus[state] += epsilon
        minus[state] -= epsilon
        numerical = (objective(plus, advantage) -
                     objective(minus, advantage)) / (2epsilon)
        @test numerical ≈ value_gradient[state] rtol=2.0e-7 atol=2.0e-8
    end
    for state in 1:states, candidate in 1:Int(counts[state])
        plus = copy(advantage)
        minus = copy(advantage)
        plus[candidate, state] += epsilon
        minus[candidate, state] -= epsilon
        numerical = (objective(value, plus) -
                     objective(value, minus)) / (2epsilon)
        @test numerical ≈ advantage_gradient[candidate, state] rtol=3.0e-7 atol=3.0e-8
    end
end

@testset "Float32 hot path allocates zero bytes" begin
    rng = MersenneTwister(0xa110c)
    width = 8
    states = 4
    counts = Int32[8, 5, 7, 6]
    teacher = randn(rng, Float32, width, states)
    value = randn(rng, Float32, states)
    advantage = randn(rng, Float32, width, states)
    q = zeros(Float32, width, states)
    q_gradient = similar(q)
    value_gradient = similar(value)
    advantage_gradient = similar(advantage)
    scratch = ListNetScratch(width, Float32)
    config = ListNetConfig(Float32)

    for _ in 1:4
        dueling_listnet_loss_and_gradient!(
            value_gradient,
            advantage_gradient,
            q,
            q_gradient,
            scratch,
            value,
            advantage,
            teacher,
            counts,
            config,
        )
    end
    allocated = @allocated dueling_listnet_loss_and_gradient!(
        value_gradient,
        advantage_gradient,
        q,
        q_gradient,
        scratch,
        value,
        advantage,
        teacher,
        counts,
        config,
    )
    @test allocated == 0
end
