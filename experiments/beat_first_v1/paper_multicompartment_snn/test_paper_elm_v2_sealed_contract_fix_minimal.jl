using Test

include(joinpath(
    @__DIR__,
    "LoadPaperELMTwinOfficialV2SealedReleaseV2ContractFixed.jl",
))

const FixedSeal =
    Main.PAPER_ELM_OFFICIAL_V2_SEALED_RELEASE_CONTRACT_FIXED

@testset "sealed V2 held-out contract fix" begin
    clipped = FixedSeal._clip_official_soma_target(
        Float32[-70, -55, -50, 20],
    )
    @test clipped == Float32[-70, -55, -55, -55]

    zero_variance = FixedSeal._PairMoments()
    FixedSeal._update!(
        zero_variance,
        Float64[1, 3],
        Float64[0, 0],
    )
    @test isapprox(
        FixedSeal._contract_normalized_rmse(
            zero_variance,
            2.0,
        ),
        sqrt(5.0) / 2.0;
        rtol=0,
        atol=1e-15,
    )

    nonzero_variance = FixedSeal._PairMoments()
    FixedSeal._update!(
        nonzero_variance,
        Float64[1, 4],
        Float64[0, 2],
    )
    @test FixedSeal._contract_normalized_rmse(
        nonzero_variance,
        999.0,
    ) == FixedSeal._normalized_rmse(nonzero_variance)

    evaluator = FixedSeal._contract_fixed_evaluator((;
        id="old",
        source_sha256="0"^64,
    ))
    @test evaluator.source_sha256 ==
        FixedSeal.corrected_evaluator_source_sha256()
    @test evaluator.heldout_target_voltage_clip_mv == -55.0
    @test evaluator.nonzero_variance_nmda_normalization_unchanged
end
