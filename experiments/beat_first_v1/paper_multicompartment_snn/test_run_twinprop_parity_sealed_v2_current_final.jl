"""Test the exact-current-V2 full-cell production runner."""

using SHA
using Test

const _CURRENT_RUNNER_BODY = joinpath(
    @__DIR__,
    "run_twinprop_parity_sealed_final.jl",
)
const _CURRENT_RUNNER_ENTRY = joinpath(
    @__DIR__,
    "run_twinprop_parity_sealed_v2_current_final.jl",
)

@testset "exact current V2 runner source boundary" begin
    body = read(_CURRENT_RUNNER_BODY, String)
    @test bytes2hex(SHA.sha256(codeunits(body))) ==
        "2ce9cb907a9d72581a7b54da0122d9b499e50286ad4ba62b5d7e0e0bbd3cea4f"
    @test length(findall("variant=:full", body)) == 2
    @test !occursin(":passive", body)
    @test !occursin(":no_nmda", body)
    @test !occursin(":soma_only", body)
    entry = read(_CURRENT_RUNNER_ENTRY, String)
    @test occursin(
        "LoadTwinPropParityOfficialSealedV2CanonicalCurrentFinal.jl",
        entry,
    )
    @test occursin("SealedELMReleaseV2", entry)
    @test occursin("exact-current-v2-transform", entry)
end

include(_CURRENT_RUNNER_ENTRY)

@testset "exact current V2 runner fails closed" begin
    @test TwinPropParityOfficial.SealedELMReleaseV2 ===
        Main.PaperELMTwinOfficialV2SealedReleaseV2
    @test TwinPropParityOfficial.
        _SEALED_V2_EXPECTED_SOURCE_SHA256.evaluator_source_sha256 ==
        TwinPropParityOfficial._file_sha256(joinpath(
            @__DIR__,
            "PaperELMTwinOfficialV2SealedReleaseV2.jl",
        ))
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
