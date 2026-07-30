using Test

include(joinpath(
    @__DIR__,
    "LoadPaperELMTwinOfficialV2SealedReleaseV2ContractFixedV2.jl",
))

const FixedSealV2 =
    Main.PAPER_ELM_OFFICIAL_V2_SEALED_RELEASE_CONTRACT_FIXED_V2

@testset "sealed V2 held-out contract fix" begin
    @test FixedSealV2._clip_official_soma_target_v2(
        Float32[-70, -55, -50, 20],
    ) == Float32[-70, -55, -55, -55]

    zero_variance = FixedSealV2._PairMoments()
    FixedSealV2._update!(
        zero_variance,
        Float64[1, 3],
        Float64[0, 0],
    )
    @test isapprox(
        FixedSealV2._contract_normalized_rmse_v2(
            zero_variance,
            2.0,
        ),
        sqrt(5.0) / 2.0;
        rtol=0,
        atol=1e-15,
    )

    nonzero_variance = FixedSealV2._PairMoments()
    FixedSealV2._update!(
        nonzero_variance,
        Float64[1, 4],
        Float64[0, 2],
    )
    @test FixedSealV2._contract_normalized_rmse_v2(
        nonzero_variance,
        999.0,
    ) == FixedSealV2._normalized_rmse(nonzero_variance)

    evaluator = FixedSealV2._contract_fixed_evaluator_v2((;
        id="old",
        source_sha256="0"^64,
    ))
    @test evaluator.source_sha256 ==
        FixedSealV2.corrected_evaluator_source_sha256_v2()
    @test evaluator.heldout_target_voltage_clip_mv == -55.0
    @test evaluator.nonzero_variance_nmda_normalization_unchanged
end
