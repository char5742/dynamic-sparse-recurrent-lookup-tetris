using JSON3
using NPZ
using SHA
using Test

include(joinpath(@__DIR__, "TorchM1000ContractFixedHandoffV1.jl"))
const Handoff = TorchM1000ContractFixedHandoffV1
const Twin = Handoff.Twin

function fixture_arrays(model, parameters)
    decay = Twin.Core.memory_decay_factors(model, parameters)
    input = zeros(Float32, 1_278, 4, 1)
    input[4, 1, 1] = 0.25f0
    input[647, 2, 1] = -0.25f0
    output = Twin.Core.twin_forward(model, parameters, input)
    raw = cat(
        reshape(output.spike_logit, 1, 4, 1),
        reshape(output.voltage, 1, 4, 1),
        output.nmda;
        dims=1,
    )
    return Dict{String,Any}(
        "state_proto_w_s" => parameters.proto_w_s,
        "state_input_weight" => parameters.input_weight,
        "state_input_bias" => parameters.input_bias,
        "state_memory_weight" => parameters.memory_weight,
        "state_memory_bias" => parameters.memory_bias,
        "state_output_weight" => parameters.output_weight,
        "state_output_bias" => parameters.output_bias,
        "state_route_indices" => Int64.(model.input_indices .- 1),
        "state_valid_indices_mask" => model.valid_indices_mask,
        "state_kappa_b" => model.kappa_b,
        "state_kappa_m" => decay.kappa_m,
        "state_kappa_lambda" => decay.kappa_lambda,
        "state_nmda_mean" =>
            Float32[-0.01, 0.02, 0.03, -0.04],
        "state_nmda_scale" =>
            Float32[0.1, 0.2, 0.3, 0.4],
        "oracle_input_signed1278" => input,
        "oracle_raw" => raw,
    )
end

@testset "Torch M1000 corrected handoff" begin
    model = Twin.build_profiled_official_elm_twin(
        Twin.OfficialELMConfig();
        mlp_activation=:silu,
        compatibility_profile=:twinprop_paper_reconstruction,
    )
    parameters = (;
        proto_w_s=fill(0.5f0, 4_500),
        input_weight=zeros(Float32, 2_000, 1_045),
        input_bias=zeros(Float32, 2_000),
        memory_weight=zeros(Float32, 1_000, 2_000),
        memory_bias=zeros(Float32, 1_000),
        output_weight=zeros(Float32, 6, 1_000),
        output_bias=Float32[0.5, -0.25, 0.1, 0.2, 0.3, 0.4],
    )
    arrays = fixture_arrays(model, parameters)

    mktempdir() do root
        checkpoint = joinpath(root, "checkpoint_update_256.pt")
        write(checkpoint, "synthetic-checkpoint-bytes")
        weights = joinpath(root, "weights.npz")
        NPZ.npzwrite(weights, arrays)
        checkpoint_sha =
            bytes2hex(SHA.sha256(read(checkpoint)))
        weights_sha = bytes2hex(SHA.sha256(read(weights)))
        base_sha = repeat("1", 64)
        augmentation_sha = repeat("2", 64)
        composite_sha = repeat("3", 64)
        details = (;
            schema=Handoff.COMPOSITE_CONTRACT_SCHEMA,
            base_manifest_sha256=base_sha,
            augmentation_manifest_sha256=augmentation_sha,
            composite_sha256=composite_sha,
            initial_update_index=0,
            updates=256,
            batch_size=32,
            heldout_opened=false,
            augmentation_heldout_trials=0,
            fit_trials=4_096,
        )
        checkpoint_metadata = (;
            schema=Handoff.TORCH_EXPORT_SCHEMA,
            checkpoint_path=abspath(checkpoint),
            checkpoint_sha256=checkpoint_sha,
            weights_npz_sha256=weights_sha,
            checkpoint_schema=Handoff.TORCH_CHECKPOINT_SCHEMA,
            memory=1_000,
            hidden=2_000,
            update_index=256,
            heldout_opened=false,
            event_count=256,
            source_bridge_metadata=(;
                manifest_sha256=composite_sha,
                m1000_fit4096_mainline=details,
            ),
            checkpoint_metadata=(;
                schema=Handoff.TORCH_CHECKPOINT_SCHEMA,
                heldout_opened=false,
            ),
        )
        metadata_path = joinpath(root, "metadata.json")
        write(metadata_path, JSON3.write(checkpoint_metadata))

        validation = (;
            schema=Handoff.TORCH_FIT4096_VALIDATION_SCHEMA,
            checkpoint=abspath(checkpoint),
            update_index=256,
            memory=1_000,
            hidden=2_000,
            base_manifest_sha256=base_sha,
            augmentation_manifest_sha256=augmentation_sha,
            metrics=(;
                validation_trials=8,
                observations=8_000,
                exact_spike_auroc=0.99,
                clip_voltage_rmse_mv=0.9,
                nmda_normalized_rmse=[0.5, 0.6, 0.7, 0.8],
                validation_source="base-only",
                heldout_opened=false,
            ),
        )
        validation_path = joinpath(root, "validation.json")
        write(validation_path, JSON3.write(validation))

        imported = Handoff.import_exported_torch_m1000_handoff(
            checkpoint,
            weights,
            metadata_path,
            validation_path,
        )
        @test imported.validation_gate.passed
        @test imported.provenance.batch_size == 32
        @test imported.provenance.update_count == 256
        @test imported.provenance.composite_contract_sha256 ==
            composite_sha
        @test imported.provenance.numeric_oracle_max_abs == 0.0
        @test imported.parameters.output_bias ==
            parameters.output_bias
        @test Handoff.assert_imported_torch_m1000_unchanged(
            imported,
        )

        frozen = Handoff.freeze_torch_m1000_handoff(imported)
        @test frozen isa Twin.FrozenOfficialELMTwin
        @test frozen.metadata.verification_status === :unverified
        @test frozen.metadata.torch_batch_size == 32
        @test frozen.metadata.torch_update_count == 256
        @test frozen.metadata.heldout_targets_opened === false
        @test frozen.metadata.
            legacy_three_by_thirty_five_evidence_claimed === false
        @test !hasproperty(frozen.metadata, :training_protocol)

        failing = merge(
            validation,
            (;
                metrics=merge(
                    validation.metrics,
                    (; exact_spike_auroc=0.984),
                ),
            ),
        )
        failing_gate =
            Handoff.corrected_v2_validation_gate(failing)
        @test !failing_gate.passed
        failing_import = Handoff.ImportedTorchM1000Handoff(
            imported.model,
            imported.parameters,
            imported.normalizer,
            imported.provenance,
            failing_gate,
            imported.parameter_sha256,
            Handoff._import_digest(
                imported.parameter_sha256,
                imported.provenance,
                failing_gate,
            ),
        )
        @test_throws ErrorException Handoff.freeze_torch_m1000_handoff(
            failing_import,
        )
    end
end
