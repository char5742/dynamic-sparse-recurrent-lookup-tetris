using SHA
using Test

const _RUNNER_BODY = joinpath(
    @__DIR__,
    "run_twinprop_parity_sealed_final.jl",
)
const _RUNNER_LOADER = joinpath(
    @__DIR__,
    "run_twinprop_parity_sealed_v2_canonical.jl",
)

@testset "exact V2 parity runner is immutable and full-cell only" begin
    body = read(_RUNNER_BODY, String)
    @test bytes2hex(SHA.sha256(codeunits(body))) ==
        "2ce9cb907a9d72581a7b54da0122d9b499e50286ad4ba62b5d7e0e0bbd3cea4f"
    @test length(findall(
        "TwinPropParityOfficialSealedCanonical.jl",
        body,
    )) == 1
    @test length(findall("SealedELMRelease", body)) == 2
    @test length(findall("variant=:full", body)) == 2
    @test !occursin(":passive", body)
    @test !occursin(":no_nmda", body)
    @test !occursin(":soma_only", body)

    loader = read(_RUNNER_LOADER, String)
    @test occursin(
        "TwinPropParityOfficialSealedV2Canonical.jl",
        loader,
    )
    @test occursin("SealedELMReleaseV2", loader)
    @test occursin("expected_sha256", loader)
    @test occursin("exact-v2-canonical-transform", loader)
end

include(_RUNNER_LOADER)

@testset "exact V2 transformed runner fails closed without evidence" begin
    @test TwinPropParityOfficial.SealedELMReleaseV2 ===
        Main.PaperELMTwinOfficialV2SealedReleaseV2
    @test TwinPropParityOfficial.SealedELMReleaseV2.SEALED_RELEASE_SCHEMA ==
        "hd_swsnn.paper_elm_v2.sealed_release.final.v2"
    @test _dimensions() == [2, 4]
    withenv("TWINPROP_DIMENSIONS" => "2,6") do
        @test_throws ErrorException _dimensions()
    end
    withenv(
        "TWINPROP_SEALED_ARTIFACT" => "",
        "TWINPROP_TEACHER_MANIFEST" => "",
        "TWINPROP_TEACHER_SHARDS" => "",
    ) do
        @test_throws ErrorException main()
    end
end
