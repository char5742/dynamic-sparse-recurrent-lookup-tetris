using Test

include(joinpath(@__DIR__, "evaluate_development_validation.jl"))

@testset "development evaluator CLI fails closed" begin
    temporary = tempname()
    preact = parse_options(["--model", "preact", "--output", temporary])
    @test preact.model === :preact
    @test preact.checkpoint == abspath(DEFAULT_PREACT_CHECKPOINT)
    @test preact.expected_checkpoint_sha256 == DEFAULT_PREACT_SHA256

    @test_throws ErrorException parse_options([
        "--model", "candidate-delta",
        "--checkpoint", tempname(),
        "--output", tempname(),
    ])
    @test_throws ErrorException parse_options([
        "--model", "preact",
        "--stage", "held",
        "--output", tempname(),
    ])

    checkpoint = tempname()
    all_models = parse_options([
        "--model", "all",
        "--checkpoint", checkpoint,
        "--expected-checkpoint-sha256", repeat("a", 64),
        "--output", tempname(),
        "--repeats", "2",
    ])
    @test all_models.model === :all
    @test all_models.checkpoint == abspath(checkpoint)
    @test all_models.repeats == 2
end
