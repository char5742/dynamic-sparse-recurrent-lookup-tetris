using Test

include(joinpath(@__DIR__, "PinnedV5ScratchRunnerCLI.jl"))
using .PinnedV5ScratchRunnerCLI

const BASE_ARGS = [
    "--sealed-release", "sealed.jld2",
    "--teacher-manifest", "manifest.json",
    "--teacher-shards", "shards",
    "--distilled-cell", "cell.jld2",
    "--distilled-sha256", repeat("a", 64),
    "--dataset", "teacher_v3",
]

@testset "pinned V5 checkpoint-lineage CLI" begin
    scratch = parse_pinned_v5_options(vcat(
        BASE_ARGS,
        [
            "--scratch",
            "--updates", "64",
            "--checkpoint-out", "checkpoint_64.jld2",
        ],
    ))
    @test scratch.scratch
    @test scratch.checkpoint_in === nothing
    @test scratch.updates == 64
    @test !scratch.require_production

    resumed = parse_pinned_v5_options(vcat(
        BASE_ARGS,
        [
            "--require-production",
            "--updates", "1000",
            "--checkpoint-in", "checkpoint_64.jld2",
            "--checkpoint-out", "checkpoint_1k.jld2",
        ],
    ))
    @test !resumed.scratch
    @test resumed.require_production
    @test resumed.updates == 1_000
    @test endswith(resumed.checkpoint_in, "checkpoint_64.jld2")

    @test_throws ErrorException parse_pinned_v5_options(vcat(
        BASE_ARGS,
        [
            "--scratch",
            "--updates", "1000",
            "--checkpoint-out", "bad.jld2",
        ],
    ))
    @test_throws ErrorException parse_pinned_v5_options(vcat(
        BASE_ARGS,
        [
            "--updates", "1000",
            "--checkpoint-out", "missing_input.jld2",
        ],
    ))
    @test_throws ErrorException parse_pinned_v5_options(vcat(
        BASE_ARGS,
        [
            "--scratch",
            "--updates", "64",
            "--checkpoint-in", "unexpected.jld2",
            "--checkpoint-out", "bad.jld2",
        ],
    ))
    @test_throws ErrorException parse_pinned_v5_options(vcat(
        BASE_ARGS,
        [
            "--scratch",
            "--updates", "42",
            "--checkpoint-out", "bad.jld2",
        ],
    ))
    @test parse_pinned_v5_options(["--help"]).help
end
