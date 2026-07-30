using Test

const FINAL_RUNNER = joinpath(
    @__DIR__,
    "train_hd_swsnn_pinned_v5_checkpoint_lineage_final.jl",
)
const FINAL_RUNNER_SOURCE = read(FINAL_RUNNER, String)
const V2_RUNNER_SOURCE = read(
    joinpath(
        @__DIR__,
        "train_hd_swsnn_pinned_v5_checkpoint_lineage_v2.jl",
    ),
    String,
)

@testset "pinned V5 final checkpoint-lineage plumbing" begin
    @test Meta.parseall(FINAL_RUNNER_SOURCE) isa Expr
    @test Meta.parseall(V2_RUNNER_SOURCE) isa Expr
    @test occursin(
        "final barrierless executor requires at least 2 workers",
        FINAL_RUNNER_SOURCE,
    )
    @test occursin(
        "train_hd_swsnn_pinned_v5_checkpoint_lineage_v2.jl",
        FINAL_RUNNER_SOURCE,
    )
    @test occursin(
        "load_pinned_v5_production_bundle",
        V2_RUNNER_SOURCE,
    )
    @test occursin("build_development_trainer", V2_RUNNER_SOURCE)
    @test occursin("build_production_trainer", V2_RUNNER_SOURCE)
    @test occursin("PaperExecutorFinal", V2_RUNNER_SOURCE)
    @test occursin("restore_checkpoint!", V2_RUNNER_SOURCE)
    @test occursin("lineage_origin=\"scratch\"", V2_RUNNER_SOURCE)
end
