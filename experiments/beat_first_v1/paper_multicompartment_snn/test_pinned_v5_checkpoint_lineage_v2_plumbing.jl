using Test

include(joinpath(@__DIR__, "PinnedV5ScratchRunnerCLI.jl"))
using .PinnedV5ScratchRunnerCLI

const V2_RUNNER = joinpath(
    @__DIR__,
    "train_hd_swsnn_pinned_v5_checkpoint_lineage_v2.jl",
)
const V2_SOURCE = read(V2_RUNNER, String)

@testset "pinned V5 V2 runner plumbing" begin
    @test Meta.parseall(V2_SOURCE) isa Expr
    @test occursin(
        "load_pinned_v5_production_bundle",
        V2_SOURCE,
    )
    @test occursin("build_development_trainer", V2_SOURCE)
    @test occursin("build_production_trainer", V2_SOURCE)
    @test occursin("PaperExecutorFinal", V2_SOURCE)
    @test occursin("restore_checkpoint!", V2_SOURCE)
    @test occursin("save_checkpoint", V2_SOURCE)
    @test occursin("lineage_origin=\"scratch\"", V2_SOURCE)

    root = parse_pinned_v5_options([
        "--sealed-release", "sealed.jld2",
        "--teacher-manifest", "manifest.json",
        "--teacher-shards", "shards",
        "--distilled-cell", "cell.jld2",
        "--distilled-sha256", repeat("a", 64),
        "--dataset", "teacher_v3",
        "--scratch",
        "--updates", "64",
        "--checkpoint-out", "checkpoint_64.jld2",
    ])
    @test root.scratch
    @test root.updates == 64
end
