using Test

include(joinpath(
    @__DIR__,
    "StreamingOfficialELMReleaseDatasetFinal.jl",
))

const StreamV3 = Main.StreamingOfficialELMReleaseDatasetV3
const StreamFinal = Main.StreamingOfficialELMReleaseDatasetFinal

@testset "exact final.v2 sealed streaming contract" begin
    @test StreamV3.Sealed.SEALED_RELEASE_SCHEMA ==
          "hd_swsnn.paper_elm_v2.sealed_release.final.v2"
    @test StreamV3.SEALED_EXECUTION_TYPE ==
          "PaperELMTwinOfficialV2SealedReleaseV2." *
          "SealedOfficialELMRelease"
    @test StreamFinal.PRIMARY_REPLAY_SCHEMA ==
          "hd_swsnn.distillation." *
          "primary_cache_live_replay.final.v2"
    @test StreamV3.OFFICIAL_ELM_INPUT_DIM == 1_278
    @test StreamV3.Signed.OFFICIAL_ELM_INPUT_DIM == 1_278

    old_bundle =
        StreamV3.Compat.Sealed.SealedOfficialELMRelease(
            nothing,
            nothing,
        )
    @test !applicable(
        StreamV3.open_sealed_stream_dataset,
        "unused",
        old_bundle,
        "unused",
        "unused",
    )

    v2_method = only(methods(
        StreamV3.open_sealed_stream_dataset,
    ))
    signature = Base.unwrap_unionall(v2_method.sig)
    @test signature.parameters[3] <:
          StreamV3.Sealed.SealedOfficialELMRelease

    valid_claim = Dict(
        "required" => true,
        "all_samples" => true,
        "bit_exact" => true,
        "targets" => [
            "soma_voltage",
            "spike_probability",
            "spike_logit",
            "regional_nmda_current",
        ],
        "detailed_auxiliary_excluded" => [
            "calcium_event_sparse",
            "dendritic_voltage_sparse",
        ],
    )
    @test StreamV3._assert_primary_claim(Dict(
        "primary_cache_live_equality" => valid_claim,
    ))
    invalid_claim = deepcopy(valid_claim)
    invalid_claim["bit_exact"] = false
    @test_throws ErrorException StreamV3._assert_primary_claim(Dict(
        "primary_cache_live_equality" => invalid_claim,
    ))

    @test StreamFinal._zero_deltas(Dict(
        "soma_voltage" => 0.0,
        "spike_probability" => 0.0,
        "spike_logit" => 0.0,
        "regional_nmda_current" => 0.0,
    )) == (0.0, 0.0, 0.0, 0.0)
    @test_throws ErrorException StreamFinal._zero_deltas(Dict(
        "soma_voltage" => 0.0,
        "spike_probability" => 0.0,
        "spike_logit" => 1.0f-7,
        "regional_nmda_current" => 0.0,
    ))
end
