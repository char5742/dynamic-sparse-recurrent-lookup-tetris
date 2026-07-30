using Lux
using Random
using Test
using Zygote

include(joinpath(@__DIR__, "PaperELMTwinOfficialV2Final.jl"))
using .PaperELMTwinOfficialV2Final

@testset "Official ELM v2 final trust boundary" begin
    @testset "source-derived fail-closed contract" begin
        config = OfficialELMConfig(;
            num_memory=4,
            hidden_size=8,
            nmda_regions=2,
        )
        model = build_official_elm_twin(config)
        @test config.num_input == 1_278
        @test config.num_branch == 45
        @test config.num_synapse_per_branch == 100
        @test length(model.input_indices) == 4_500
        @test count(!iszero, model.valid_indices_mask) == 4_282
        @test official_contact_channel(2, :E) == 1
        @test official_contact_channel(640, :I) == 1_278
        @test_throws ArgumentError official_contact_channel(1, :E)
        @test_throws ArgumentError official_contact_channel(641, :I)

        @test preprocess_soma_voltage(
            Float32[-80, -67.7, -55, -40],
        ) ≈ Float32[-1.23, 0, 1.27, 1.27]
        @test soma_voltage_from_coordinate(
            Float32[-1.23, 0, 1.27],
        ) ≈ Float32[-80, -67.7, -55]
        metadata = official_elm_source_metadata()
        @test metadata.source_derived_input_dim == 1_278
        @test metadata.identity_input_normalization
        @test metadata.input_normalization == "none"
    end

    @testset "identity input and explicit NMDA extension" begin
        rng = Xoshiro(1)
        input = randn(rng, Float32, 1_278, 3, 4)
        voltage = randn(rng, Float32, 3, 4)
        nmda = randn(rng, Float32, 2, 3, 4)
        normalizer = fit_official_elm_normalizer(
            input,
            voltage,
            nmda,
            1:3,
        )
        @test normalize_official_elm_input(normalizer, input) === input
        @test length(normalizer.nmda_mean) == 2
        @test all(>(0), normalizer.nmda_scale)
        @test !hasproperty(normalizer, :input_mean)
        @test !hasproperty(normalizer, :voltage_mean)
    end

    @testset "strict frozen and attested artifacts" begin
        config = OfficialELMConfig(;
            num_memory=4,
            hidden_size=8,
            nmda_regions=2,
        )
        model = build_official_elm_twin(config)
        ps, _ = Lux.setup(Xoshiro(2), model)
        normalizer = OfficialELMNormalizer(
            zeros(Float32, 2),
            ones(Float32, 2),
        )
        @test_throws ArgumentError freeze_official_elm_twin(
            model,
            ps,
            normalizer;
            metadata=(verification_passed=true,),
        )
        frozen = freeze_official_elm_twin(
            model,
            ps,
            normalizer;
            metadata=(
                teacher_dataset_id="rich64_current",
                run_id="test",
            ),
        )
        @test assert_frozen_official_elm_unchanged(frozen)
        @test frozen.metadata.verification_status === :unverified

        frozen_other_metadata = freeze_official_elm_twin(
            model,
            ps,
            normalizer;
            metadata=(
                teacher_dataset_id="rich64_current",
                run_id="different",
            ),
        )
        @test frozen.artifact_sha256 !=
              frozen_other_metadata.artifact_sha256

        passing_metrics = (;
            voltage_rmse=0.45,
            spike_auroc=0.96,
            nmda_rmse=0.30,
        )
        thresholds = (;
            max_voltage_rmse=0.50,
            min_spike_auroc=0.95,
            max_nmda_rmse=0.35,
        )
        manifest_hash = repeat("a", 64)
        contract_hash = repeat("b", 64)
        @test_throws ErrorException attest_official_elm_twin(
            frozen;
            metrics=merge(
                passing_metrics,
                (spike_auroc=0.80,),
            ),
            thresholds,
            teacher_manifest_sha256=manifest_hash,
            teacher_contract_sha256=contract_hash,
            evaluator_id="heldout-v1",
        )
        verified = attest_official_elm_twin(
            frozen;
            metrics=passing_metrics,
            thresholds,
            teacher_manifest_sha256=manifest_hash,
            teacher_contract_sha256=contract_hash,
            evaluator_id="heldout-v1",
        )
        @test assert_verified_official_elm(verified)
        @test verified.attestation.passed
        @test verified.attestation.metrics == passing_metrics
        @test verified.attestation_sha256 ==
              attestation_sha256(verified.attestation)

        input = 0.01f0 .* randn(
            Xoshiro(3),
            Float32,
            1_278,
            3,
            2,
        )
        coordinate = official_elm_forward(model, ps, input)
        physical = twin_forward(verified, input)
        @test physical.voltage_coordinate ≈ coordinate.voltage
        @test physical.voltage ≈
              soma_voltage_from_coordinate(coordinate.voltage)
        input_gradient = only(Zygote.gradient(input) do candidate
            prediction = twin_forward(
                verified,
                candidate;
                normalized=true,
            )
            return sum(prediction.spike_probability) +
                   0.01f0 * sum(prediction.voltage) +
                   0.01f0 * sum(prediction.nmda)
        end)
        @test all(isfinite, input_gradient)
        @test maximum(abs, input_gradient) > 0

        tampered_parameters = deepcopy(verified)
        tampered_parameters.parameters.output_bias[1] += 0.25f0
        @test_throws ErrorException assert_verified_official_elm(
            tampered_parameters,
        )

        bad_attestation = OfficialHeldOutAttestation(
            merge(
                verified.attestation.metrics,
                (spike_auroc=0.50,),
            ),
            verified.attestation.thresholds,
            verified.attestation.teacher_manifest_sha256,
            verified.attestation.teacher_contract_sha256,
            verified.attestation.evaluator_id,
            verified.attestation.source_metadata_sha256,
            verified.attestation.model_config_sha256,
            verified.attestation.parameter_sha256,
            verified.attestation.base_artifact_sha256,
            true,
        )
        forged = VerifiedOfficialELMTwin(
            verified.model,
            verified.parameters,
            verified.normalizer,
            verified.metadata,
            verified.parameter_sha256,
            verified.artifact_sha256,
            bad_attestation,
            verified.attestation_sha256,
        )
        @test_throws ErrorException assert_verified_official_elm(forged)

        mktempdir() do directory
            unverified_path = joinpath(directory, "unverified.jld2")
            verified_path = joinpath(directory, "verified.jld2")
            save_frozen_official_elm(unverified_path, frozen)
            @test_throws ErrorException load_verified_official_elm(
                unverified_path,
            )
            save_verified_official_elm(verified_path, verified)
            loaded = load_verified_official_elm(verified_path)
            @test assert_verified_official_elm(loaded)
            @test loaded.attestation_sha256 ==
                  verified.attestation_sha256
        end
    end
end
