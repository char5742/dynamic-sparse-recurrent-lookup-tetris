using Test

include(joinpath(@__DIR__, "LoadDevelopmentScaleChainExactFinal.jl"))
using .DevelopmentScaleChainGate

const _CHAIN_HASH_TWIN = repeat("1", 64)
const _CHAIN_HASH_DISTILL = repeat("2", 64)
const _CHAIN_HASH_FREEZE = repeat("3", 64)

@testset "development chain accepts only detailed-twin-distill-freeze" begin
    calls = Symbol[]
    twin_gate = detailed -> begin
        push!(calls, :twin)
        @test detailed.gate_name === :development_scale_teacher_gate
        return (verified=true, artifact_sha256=_CHAIN_HASH_TWIN)
    end
    distill_gate = (detailed, twin) -> begin
        push!(calls, :distill)
        @test detailed.duration_ms == 1_500
        @test twin.artifact_sha256 == _CHAIN_HASH_TWIN
        return (verified=true, artifact_sha256=_CHAIN_HASH_DISTILL)
    end
    freeze_gate = (detailed, twin, distill) -> begin
        push!(calls, :freeze)
        @test detailed.paper_scale === false
        @test twin.artifact_sha256 == _CHAIN_HASH_TWIN
        @test distill.artifact_sha256 == _CHAIN_HASH_DISTILL
        return (verified=true, artifact_sha256=_CHAIN_HASH_FREEZE)
    end

    chain = verify_development_scale_chain(
        ;
        official_v2_gate=twin_gate,
        distilled_cell_gate=distill_gate,
        frozen_runtime_gate=freeze_gate,
    )

    @test calls == [:twin, :distill, :freeze]
    @test chain.chain_complete === true
    @test chain.chain_stages ==
        (:detailed, :twin, :distill, :freeze)
    @test chain.downstream_artifact_gate === :verified
    @test chain.detailed_teacher_artifact.manifest_sha256 ==
        DEVELOPMENT_TEACHER_MANIFEST_SHA256_DEV1500
    @test chain.official_v2_artifact.artifact_sha256 ==
        _CHAIN_HASH_TWIN
    @test chain.distilled_cell_artifact.artifact_sha256 ==
        _CHAIN_HASH_DISTILL
    @test chain.frozen_runtime_artifact.artifact_sha256 ==
        _CHAIN_HASH_FREEZE

    @test_throws UndefKeywordError verify_development_scale_chain(
        ;
        official_v2_gate=twin_gate,
        distilled_cell_gate=distill_gate,
    )
end
