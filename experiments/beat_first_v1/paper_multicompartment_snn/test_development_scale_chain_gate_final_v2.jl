using Test

include(joinpath(@__DIR__, "LoadDevelopmentScaleChainGateFinalV2.jl"))
using .DevelopmentScaleChainGate

@testset "beat_first_v1 canonical development gate final.v2 loader" begin
    report = verify_development_teacher_manifest()
    @test report.verified === true
    @test report.schema_name ==
        "hd_swsnn_twinprop.neuron_teacher.final.v2"
    @test report.schema_version == 2
    @test report.teacher_contract_sha256 ==
        DEVELOPMENT_TEACHER_CONTRACT_SHA256_FINAL_V2
    @test report.manifest_sha256 ==
        DEVELOPMENT_TEACHER_MANIFEST_SHA256_FINAL_V2
    @test report.counts.train == 40
    @test report.counts.validation_from_train == 8
    @test report.counts.fit == 32
    @test report.counts.held_out_test == 8
    @test report.counts.completed == 48
    @test report.duration_ms == 500
    @test report.sample_dt_ms == 1.0
    @test report.time_steps == 500
    @test report.total_segments == 642
    @test report.paper_scale === false
    @test report.promotable_production === false
    @test report.all_hashes_verified === true
end
