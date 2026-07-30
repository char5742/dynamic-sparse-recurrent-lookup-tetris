using Test

include(joinpath(@__DIR__, "LoadDevelopmentScaleChainGateFinalV2.jl"))

const Gate = DevelopmentScaleChainGate

function development_profile_fixture(;
    train_trials=12,
    held_out_test_trials=3,
    validation_from_train_trials=2,
    duration_ms=750,
    completed_trials=train_trials + held_out_test_trials,
)
    manifest = (;
        config=(;
            train_trials,
            test_trials=held_out_test_trials,
            validation_trials_from_train=validation_from_train_trials,
            duration_ms,
            sample_dt_ms=1.0,
            axons=64,
            preset="smoke",
            connectivity_interpretation_acknowledged=false,
        ),
        completed_trials,
        validation_from_train_indices=collect(
            (train_trials - validation_from_train_trials + 1):train_trials,
        ),
    )
    contract = (;
        connectivity_scale_conflict=(;
            fully_paper_scale_claim=false,
            interpretation_explicitly_acknowledged=false,
        ),
    )
    return manifest, contract
end

@testset "development final.v2 scale metadata is manifest-derived" begin
    manifest, contract = development_profile_fixture()
    scale = Gate._scale_profile(manifest, contract, :development)

    @test scale.train_trials == 12
    @test scale.validation_from_train_trials == 2
    @test scale.fit_trials == 10
    @test scale.held_out_test_trials == 3
    @test scale.completed_trials == 15
    @test scale.duration_ms == 750
    @test scale.sample_dt_ms == 1.0
    @test scale.time_steps == 750
    @test scale.paper_scale === false
    @test scale.promotable_production === false

    bad_manifest, bad_contract =
        development_profile_fixture(completed_trials=14)
    @test_throws ErrorException Gate._scale_profile(
        bad_manifest,
        bad_contract,
        :development,
    )
end
