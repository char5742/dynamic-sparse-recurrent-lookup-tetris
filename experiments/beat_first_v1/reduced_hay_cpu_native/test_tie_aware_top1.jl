using Test

include(joinpath(@__DIR__, "ReducedHayCPU.jl"))
using .ReducedHayCPU

const Ranking = ReducedHayCPU.TetrisRankingBatch

@testset "tie-aware top-1" begin
    batch = Ranking.Batch(1, 3)
    batch.counts[1] = 3
    batch.targets.teacher_q[1:3, 1] .= Float32[2, 2, 1]
    batch.raw[1, 1:3] .= Float32[1, 3, 0]
    @test ReducedHayCPU.batch_top1(batch) == 0.0f0
    @test ReducedHayCPU.batch_tie_aware_top1(batch) == 1.0f0

    batch.raw[1, 1:3] .= Float32[1, 0, 3]
    @test ReducedHayCPU.batch_tie_aware_top1(batch) == 0.0f0
end
