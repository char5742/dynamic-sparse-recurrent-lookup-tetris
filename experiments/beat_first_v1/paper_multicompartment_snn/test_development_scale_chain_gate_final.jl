using Test

include(joinpath(@__DIR__, "LoadDevelopmentScaleChainGateFinal.jl"))
using .DevelopmentScaleChainGate

@testset "beat_first_v1 canonical development gate loader" begin
    report = verify_development_teacher_manifest()
    @test report.verified === true
    @test report.counts.train == 40
    @test report.counts.held_out_test == 8
    @test report.duration_ms == 100
    @test report.sample_dt_ms == 1.0
    @test report.total_segments == 642
    @test report.paper_scale === false
    @test report.promotable_production === false
    @test report.all_hashes_verified === true
end

