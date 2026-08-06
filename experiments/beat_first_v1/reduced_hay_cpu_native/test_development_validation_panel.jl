using Test

include(joinpath(@__DIR__, "DevelopmentValidationPanel.jl"))
using .DevelopmentValidationPanel

@testset "stable and tie-aware ranking metrics remain distinct" begin
    dataset = (;
        action_counts=Int[3, 2],
        teacher_q=Float32[2 0; 2 1; 1 0],
    )
    contract = PanelContract(
        :development_validation,
        "fixture",
        "fixture",
        Int[1, 2],
        "fixture",
        2,
        5,
        2,
        3,
        1,
        false,
        false,
    )
    scores = [Float32[1, 3, 0], Float32[-1, 2]]
    metrics = evaluate_rankings(scores, dataset, contract)
    @test metrics.legacy_stable_top1 == 0.5
    @test metrics.tie_aware_top1 == 1.0
    @test metrics.states == 2
    @test metrics.candidates == 5
    @test metrics.listnet_excess >= 0
    @test 0 <= metrics.ndcg <= 1
    @test 0 <= metrics.pairwise_accuracy <= 1
end

@testset "perfect affine ranking has zero-ordering error" begin
    dataset = (;
        action_counts=Int[3],
        teacher_q=reshape(Float32[3, 2, 1], 3, 1),
    )
    contract = PanelContract(
        :development_validation,
        "fixture",
        "fixture",
        Int[1],
        "fixture",
        1,
        3,
        3,
        3,
        0,
        false,
        false,
    )
    metrics = evaluate_rankings([Float32[6, 4, 2]], dataset, contract)
    @test metrics.legacy_stable_top1 == 1
    @test metrics.tie_aware_top1 == 1
    @test metrics.ndcg == 1
    @test metrics.pairwise_accuracy == 1
    @test metrics.listnet_excess < 1.0e-6
end

@testset "development-only contract fields cannot masquerade as held" begin
    contract = PanelContract(
        :development_validation,
        "fixture",
        "fixture",
        Int[1],
        "fixture",
        1,
        1,
        1,
        1,
        0,
        false,
        false,
    )
    @test contract.stage === :development_validation
    @test !contract.held_test_touched
    @test !contract.sealed_game_seed_touched
end
