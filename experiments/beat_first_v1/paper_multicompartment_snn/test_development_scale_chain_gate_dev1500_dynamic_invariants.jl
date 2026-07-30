using Test

include(
    joinpath(
        @__DIR__,
        "LoadDevelopmentScaleChainGateDev1500Canonical.jl",
    ),
)

const GateDev1500 = DevelopmentScaleChainGate

function dev1500_profile_fixture(; duration_ms=1_250)
    manifest = (;
        config=(;
            train_trials=12,
            test_trials=3,
            validation_trials_from_train=2,
            duration_ms,
            sample_dt_ms=1.0,
            axons=64,
            preset="smoke",
            connectivity_interpretation_acknowledged=false,
        ),
        completed_trials=15,
        validation_from_train_indices=[11, 12],
    )
    contract = (;
        connectivity_scale_conflict=(;
            fully_paper_scale_claim=false,
            interpretation_explicitly_acknowledged=false,
        ),
    )
    return manifest, contract
end

@testset "dev scale is dynamic but NeuronIO window is mandatory" begin
    manifest, contract = dev1500_profile_fixture()
    scale = GateDev1500._scale_profile(
        manifest,
        contract,
        :development,
    )
    accepted = GateDev1500._assert_neuronio_development_window(scale)

    @test accepted.train_trials == 12
    @test accepted.validation_from_train_trials == 2
    @test accepted.fit_trials == 10
    @test accepted.held_out_test_trials == 3
    @test accepted.completed_trials == 15
    @test accepted.duration_ms == 1_250
    @test accepted.time_steps == 1_250
    @test accepted.paper_scale === false
    @test accepted.promotable_production === false

    too_short_manifest, too_short_contract =
        dev1500_profile_fixture(duration_ms=500)
    too_short = GateDev1500._scale_profile(
        too_short_manifest,
        too_short_contract,
        :development,
    )
    @test_throws ErrorException GateDev1500._assert_neuronio_development_window(
        too_short,
    )
end
