using Test

include(joinpath(@__DIR__, "DevelopmentScaleChainGate.jl"))
using .DevelopmentScaleChainGate

const DEVELOPMENT_MANIFEST_FOR_TEST = get(
    ENV,
    "HD_SWSNN_DEVELOPMENT_TEACHER_MANIFEST",
    DEVELOPMENT_TEACHER_MANIFEST,
)

@testset "rich64 development-scale teacher is explicit and hash-complete" begin
    report = verify_development_teacher_manifest(
        DEVELOPMENT_MANIFEST_FOR_TEST,
    )

    @test report.verified === true
    @test report.gate_name === :development_scale_teacher_gate
    @test report.schema_name == DEVELOPMENT_TEACHER_SCHEMA
    @test report.teacher_contract_sha256 ==
        DEVELOPMENT_TEACHER_CONTRACT_SHA256
    @test report.manifest_sha256 == DEVELOPMENT_TEACHER_MANIFEST_SHA256
    @test report.official_hay_neuron_teacher === true
    @test report.source_kind === :real_hay_modeldb_neuron
    @test report.provisional === false
    @test report.synthetic === false
    @test report.reconstructed === false

    @test report.counts == (
        train=40,
        validation_from_train=8,
        fit=32,
        held_out_test=8,
        completed=48,
    )
    @test report.train_trials == 40
    @test report.held_out_test_trials == 8
    @test report.duration_ms == 100
    @test report.sample_dt_ms == 1.0
    @test report.time_steps == 100
    @test report.total_segments == 642
    @test report.paper_scale === false
    @test report.promotable_production === false
    @test report.chain_complete === false
    @test report.downstream_artifact_gate === :not_bound

    @test report.verified_split_counts ==
        (train=40, validation=0, held_out_test=8)
    @test report.shard_count == 24
    @test report.verified_shard_count == 24
    @test report.all_shard_hashes_verified === true
    @test report.all_sidecars_verified === true
    @test report.all_source_hashes_verified === true
    @test report.all_hashes_verified === true
    @test report.source_hashes.local_modeldb_tree_verified === true
    @test report.source_hashes.local_morphology_verified === true
    @test report.source_hashes.local_mechanism_sources_verified === true
    @test report.source_hashes.local_mechanism_library_verified === true
    @test report.source_hashes.local_generators_verified === true
end

@testset "development cannot enter paper-scale production path" begin
    @test_throws ErrorException verify_paper_scale_teacher_manifest(
        DEVELOPMENT_MANIFEST_FOR_TEST,
    )
end

@testset "downstream artifact API is fail-closed and type-owner neutral" begin
    @test_throws UndefKeywordError verify_development_scale_chain(
        DEVELOPMENT_MANIFEST_FOR_TEST,
    )
end

