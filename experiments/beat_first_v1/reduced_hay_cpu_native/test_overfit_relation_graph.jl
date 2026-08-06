using Test

include(joinpath(@__DIR__, "overfit_relation_graph.jl"))

@testset "minimal canonical overfit CLI" begin
    options = parse_options([
        "--states=64",
        "--updates", "321",
        "--log-every=17",
        "--eval-every", "19",
        "--batch-size", "8",
        "--learning-rate", "0.0007",
        "--finish-learning-rate", "0.00007",
        "--finish-at", "200",
        "--seed", "9",
        "--dataset", raw"D:\fixture",
        "--trainer", "barrierless",
        "--workers", "7",
        "--candidate-chunk", "3",
        "--diagnose-final", "true",
    ])
    @test options.states == 64
    @test options.updates == 321
    @test options.log_every == 17
    @test options.eval_every == 19
    @test options.batch_size == 8
    @test options.learning_rate == 7.0f-4
    @test options.finish_learning_rate == 7.0f-5
    @test options.finish_at == 200
    @test options.seed == 9
    @test options.dataset == raw"D:\fixture"
    @test options.trainer === :barrierless
    @test options.workers == 7
    @test options.candidate_chunk_size == 3
    @test options.diagnose_final

    @test_throws ErrorException parse_options(["--states", "16"])
    @test_throws ErrorException parse_options(["--states", "0"])
    @test_throws ErrorException parse_options(["--updates", "0"])
    @test_throws ErrorException parse_options(["--log-every", "0"])
    @test_throws ErrorException parse_options(["--eval-every", "0"])
    @test_throws ErrorException parse_options([
        "--states", "64", "--batch-size", "16",
    ])
    @test_throws ErrorException parse_options([
        "--states", "8", "--batch-size", "3",
    ])
    @test_throws ErrorException parse_options(["--learning-rate", "0"])
    @test_throws ErrorException parse_options([
        "--states", "8", "--updates", "10",
        "--finish-learning-rate", "0.0001",
    ])
    @test_throws ErrorException parse_options([
        "--states", "8", "--updates", "10", "--finish-at", "11",
        "--finish-learning-rate", "0.0001",
    ])
    @test_throws ErrorException parse_options(["--trainer", "legacy"])
    @test_throws ErrorException parse_options(["--workers", "0"])
    @test_throws ErrorException parse_options(["--candidate-chunk", "0"])

    @test parse_options(["--states", "1"]).batch_size == 1
    @test parse_options(["--states", "8"]).batch_size == 8
    @test parse_options(["--states", "64"]).batch_size == 8
end

@testset "deterministic fixed-panel minibatch cycle" begin
    panel = collect(1001:1064)
    batch = Ranking.Batch(8, CANDIDATE_WIDTH)
    visited = Int[]
    for update in 1:8
        select_cyclic_minibatch!(batch, panel, update)
        append!(visited, batch.rows)
    end
    @test visited == panel

    select_cyclic_minibatch!(batch, panel, 9)
    @test batch.rows == panel[1:8]
    @test_throws ArgumentError select_cyclic_minibatch!(batch, panel, 0)
    @test_throws DimensionMismatch select_cyclic_minibatch!(
        Ranking.Batch(7, CANDIDATE_WIDTH),
        panel,
        1,
    )
end

@testset "complete-panel late-collapse gate" begin
    passing = EvaluationTail()
    for excess in (0.041f0, 0.030f0, 0.021f0, 0.018f0, 0.015f0)
        record_evaluation!(passing, excess, 1.0f0)
    end
    summary = summarize_tail(passing)
    @test summary.count == TAIL_EVALUATIONS
    @test summary.excess_max == 0.041f0
    @test summary.top1_min == 1.0f0
    @test summary.stable

    collapse = EvaluationTail()
    for (excess, top1) in zip(
        (0.030f0, 0.025f0, 0.080f0, 0.020f0, 0.019f0),
        (1.0f0, 1.0f0, 0.984375f0, 1.0f0, 1.0f0),
    )
        record_evaluation!(collapse, excess, top1)
    end
    failed = summarize_tail(collapse)
    @test failed.excess_max == 0.080f0
    @test failed.top1_min == 0.984375f0
    @test !failed.stable

    insufficient = EvaluationTail()
    record_evaluation!(insufficient, 0.01f0, 1.0f0)
    record_evaluation!(insufficient, 0.01f0, 1.0f0)
    @test !summarize_tail(insufficient).stable
end
