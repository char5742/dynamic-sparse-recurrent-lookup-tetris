using SHA
using Test

include(joinpath(
    @__DIR__,
    "prepare_distillation_dataset_official1278_sealed_contract_fixed_v2.jl",
))
include(joinpath(
    @__DIR__,
    "LoadTwinPropParityOfficialSealedV2CanonicalContractFixedV2.jl",
))

const CriticalBridge =
    Main.DistillationDatasetBridgeOfficial1278SealedContractFixedV2
const CriticalParity =
    Main.TWINPROP_PARITY_OFFICIAL_SEALED_V2_CONTRACT_FIXED_CANONICAL

@testset "contract-fixed V2 parity/bridge identity" begin
    @test CriticalBridge.Sealed ===
        Main.PAPER_ELM_OFFICIAL_V2_SEALED_RELEASE_CONTRACT_FIXED_V2
    @test CriticalBridge.Writer.Sealed === CriticalBridge.Sealed
    @test CriticalBridge.Writer.Bridge.Sealed === CriticalBridge.Sealed
    @test CriticalParity.SealedELMReleaseV2 === CriticalBridge.Sealed
    @test CriticalParity.
        _SEALED_V2_EXPECTED_SOURCE_SHA256.evaluator_source_sha256 ==
        CriticalBridge.Sealed.corrected_evaluator_source_sha256_v2()
    @test CriticalBridge.SEALED_RELEASE_SCHEMA ==
        "hd_swsnn.paper_elm_v2.sealed_release.final.v2"
    @test CriticalBridge.OFFICIAL_INPUT_DIM == 1_278
end

@testset "contract-fixed V2 add-only CLI boundary" begin
    sealed, passthrough = CriticalBridge._split_wrapper_arguments([
        "--sealed-artifact",
        "sealed.jld2",
        "--dataset",
        "teacher",
        "--frozen-twin",
        "frozen.jld2",
    ])
    @test sealed == abspath("sealed.jld2")
    @test passthrough == [
        "--dataset",
        "teacher",
        "--frozen-twin",
        "frozen.jld2",
    ]
    withenv("HD_TWINPROP_SEALED_ARTIFACT" => "") do
        @test_throws ErrorException
            CriticalBridge._split_wrapper_arguments(String[])
    end
    @test_throws ErrorException CriticalBridge.extract_attested_frozen(
        "missing-sealed.jld2",
        "unused-frozen.jld2",
        "missing-teacher",
    )
end

@testset "contract-fixed V2 runner transform boundary" begin
    body_path = joinpath(
        @__DIR__,
        "run_twinprop_parity_sealed_final.jl",
    )
    entry_path = joinpath(
        @__DIR__,
        "run_twinprop_parity_sealed_v2_contract_fixed_v2.jl",
    )
    @test bytes2hex(SHA.sha256(read(body_path))) ==
        "2ce9cb907a9d72581a7b54da0122d9b499e50286ad4ba62b5d7e0e0bbd3cea4f"
    entry = read(entry_path, String)
    @test occursin(
        "LoadTwinPropParityOfficialSealedV2CanonicalContractFixedV2.jl",
        entry,
    )
    @test occursin("contract-fixed-v2-transform", entry)
    @test occursin("SealedELMReleaseV2", entry)
    @test !occursin(":passive", entry)
    @test !occursin(":no_nmda", entry)
    @test !occursin(":soma_only", entry)
end
