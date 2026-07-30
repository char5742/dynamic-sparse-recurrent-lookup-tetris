using Test

include(joinpath(
    @__DIR__,
    "prepare_distillation_dataset_official1278_sealed_final_v2.jl",
))

const FinalV2 =
    Main.DistillationDatasetBridgeOfficial1278SealedFinalV2
const BridgeV2 = FinalV2.Bridge

@testset "final.v2 bridge and post-write replay promotion" begin
    @test BridgeV2.Sealed ===
          Main.PaperELMTwinOfficialV2SealedReleaseV2
    @test BridgeV2.OfficialTwin === BridgeV2.Sealed.Twin
    @test FinalV2.SEALED_RELEASE_SCHEMA ==
          "hd_swsnn.paper_elm_v2.sealed_release.final.v2"
    @test FinalV2.PRIMARY_REPLAY_SCHEMA ==
          "hd_swsnn.distillation." *
          "primary_cache_live_replay.final.v2"

    verification_method = which(
        BridgeV2.BaseBridge._verify_twin,
        (
            BridgeV2.BaseBridge.PrepareDistillationConfig,
            BridgeV2.BaseBridge._Source,
        ),
    )
    @test endswith(
        String(verification_method.file),
        "prepare_distillation_dataset_official1278_sealed_v2.jl",
    )

    time_contract = BridgeV2._neuronio_contract(1_500, 1.0)
    @test time_contract.training.
          valid_window_start_indices_one_based == (501, 1_000)
    @test time_contract.heldout.
          evaluated_time_indices_one_based == (501, 1_500)
    @test time_contract.heldout.
          evaluated_steps_per_trial == 1_000

    replay = (;
        cache_verified_all_samples=true,
        bit_exact=true,
        samples_verified=48,
        time_points_verified=72_000,
        soma_voltage_max_delta=0.0,
        spike_probability_max_delta=0.0,
        spike_logit_max_delta=0.0,
        nmda_max_delta=0.0,
    )
    result = FinalV2._replay_result(replay)
    @test result.all_samples
    @test result.bit_exact
    @test all(iszero, values(result.max_absolute_delta))

    bad_replay = merge(
        replay,
        (; nmda_max_delta=eps(Float32)),
    )
    @test_throws ErrorException
        FinalV2._replay_result(bad_replay)

    measurement = (;
        schema=FinalV2.PRIMARY_REPLAY_SCHEMA,
        measurement="postwrite_live_replay",
        sealed_attestation_sha256=repeat("a", 64),
        result,
    )
    promoted = FinalV2._promoted_manifest(
        Dict("schema" => FinalV2.RELEASE_DATASET_SCHEMA),
        measurement,
    )
    @test FinalV2.Stream._assert_primary_claim(promoted)
    claim = promoted["primary_cache_live_equality"]
    @test claim.bit_exact
    @test claim.all_samples
    @test claim.measurement == "postwrite_live_replay"
    @test claim.measurement_sha256 ==
          FinalV2.Sealed.canonical_sha256(measurement)

    failed_measurement = merge(
        measurement,
        (; result=merge(
            result,
            (; bit_exact=false),
        )),
    )
    @test_throws ErrorException FinalV2._promoted_manifest(
        Dict("schema" => FinalV2.RELEASE_DATASET_SCHEMA),
        failed_measurement,
    )
end
