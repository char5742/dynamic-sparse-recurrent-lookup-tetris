using Test

include(joinpath(
    @__DIR__,
    "prepare_distillation_dataset_official1278_sealed.jl",
))

const Bridge = Main.DistillationDatasetBridgeOfficial1278Sealed
const Diagnostic = Bridge.Diagnostic
const Sealed = Bridge.Sealed

@testset "sealed official-1278 bridge contract" begin
    @test Bridge.SEALED_RELEASE_SCHEMA ==
          "hd_swsnn.paper_elm_v2.sealed_release.final.v1"
    @test Bridge.OFFICIAL_INPUT_DIM == 1_278
    @test Bridge.NEURONIO_REFERENCE_COMMIT ==
          "52e68a6d39523ac6613a586699b116e8e606dda3"

    contract = Bridge._neuronio_contract(1_500, 1.0)
    @test contract.training.full_time_steps == 1_500
    @test contract.training.ignore_time_from_start_ms == 500
    @test contract.training.input_window_steps == 500
    @test contract.training.valid_window_start_indices_one_based ==
          (501, 1_000)
    @test contract.training.sampling == "uniform_with_replacement"
    @test contract.heldout.evaluated_time_indices_one_based ==
          (501, 1_500)
    @test contract.heldout.evaluated_steps_per_trial == 1_000
    @test_throws ErrorException Bridge._neuronio_contract(500, 1.0)
    @test_throws ErrorException Bridge._neuronio_contract(1_500, 0.5)

    # The bridge proxy is tied to the independently source-recomputed sealed
    # type.  Legacy caller-metric and diagnostic verified objects cannot enter
    # its constructor dispatch.
    @test fieldtype(Bridge.SealedOfficialBridgeTwin, 1) isa TypeVar
    @test !applicable(
        Bridge._bridge_twin,
        Diagnostic.OfficialTwin.VerifiedOfficialELMTwin,
        nothing,
    )

    verification_method = which(
        Bridge.BaseBridge._verify_twin,
        (
            Bridge.BaseBridge.PrepareDistillationConfig,
            Bridge.BaseBridge._Source,
        ),
    )
    @test endswith(
        String(verification_method.file),
        "prepare_distillation_dataset_official1278_sealed.jl",
    )

    @test Sealed.MINIMUM_SPIKE_AUROC == 0.985
    @test Sealed.MAXIMUM_VOLTAGE_RMSE_MV == 1.0
    @test Sealed.MAXIMUM_REGIONAL_NMDA_NORMALIZED_RMSE == 1.0
    @test length(methods(
        Sealed.attest_sealed_official_elm_release,
    )) == 1
    @test Set(Base.kwarg_decl(only(methods(
        Sealed.attest_sealed_official_elm_release,
    )))) == Set((:scratch_root,))
end
