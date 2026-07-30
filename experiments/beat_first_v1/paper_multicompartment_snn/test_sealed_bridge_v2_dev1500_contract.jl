using Test

include(joinpath(
    @__DIR__,
    "prepare_distillation_dataset_official1278_sealed_v2.jl",
))

const BridgeV2 =
    Main.DistillationDatasetBridgeOfficial1278SealedV2

@testset "final.v2 dev1500 source and outcome contract" begin
    payload = (;
        schema=BridgeV2.Sealed.SEALED_RELEASE_SCHEMA,
        outcome=(;
            gate_passed=true,
            caller_metrics_accepted=false,
            caller_targets_accepted=false,
            caller_manifest_digest_accepted=false,
            development_scale=true,
            paper_scale=false,
            promotable_production=false,
        ),
        split=(;
            duration_ms=1_500.0,
            fit_count=32,
            validation_count=8,
            heldout_count=8,
        ),
        teacher=(;
            manifest_sha256=
                BridgeV2.DEV1500_MANIFEST_SHA256,
            teacher_contract_sha256=
                BridgeV2.DEV1500_CONTRACT_SHA256,
        ),
    )
    @test BridgeV2._assert_v2_outcome(
        payload;
        require_production=false,
    )
    @test_throws ErrorException BridgeV2._assert_v2_outcome(
        merge(
            payload,
            (; outcome=merge(
                payload.outcome,
                (; promotable_production=true),
            )),
        );
        require_production=false,
    )
    @test_throws ErrorException BridgeV2._assert_v2_outcome(
        merge(
            payload,
            (; split=merge(
                payload.split,
                (; duration_ms=500.0),
            )),
        );
        require_production=false,
    )
    @test_throws ErrorException BridgeV2._assert_v2_outcome(
        merge(
            payload,
            (; teacher=merge(
                payload.teacher,
                (; teacher_contract_sha256=repeat("0", 64)),
            )),
        );
        require_production=false,
    )
end
