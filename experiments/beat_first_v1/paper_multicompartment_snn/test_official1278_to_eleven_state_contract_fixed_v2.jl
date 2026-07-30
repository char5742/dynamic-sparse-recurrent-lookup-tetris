using Test

include(joinpath(
    @__DIR__,
    "run_official1278_to_eleven_state_contract_fixed_v2.jl",
))

const Chain = Main.Official1278ToElevenStateContractFixedV2
const FrozenCell = Chain.Cell

function _fixture_parameters(
    twin_parameter_sha256,
    twin_artifact_sha256,
)
    projection = zeros(Float32, 4, 642)
    projection[:, 2:640] .= 0.25f0
    return FrozenCell.DistilledParameters(
        dt_ms=1.0f0,
        transition_decay=fill(0.5f0, 11),
        recurrent_weight=zeros(Float32, 11, 11),
        input_weight=zeros(Float32, 11, 16),
        transition_bias=zeros(Float32, 11),
        readout_weight=zeros(Float32, 11, 11),
        readout_bias=zeros(Float32, 11),
        target_mean=zeros(Float32, 11),
        target_scale=ones(Float32, 11),
        initial_state=zeros(Float32, 11),
        compartment_projection=projection,
        region_projection=fill(0.25f0, 4, 4),
        teacher_schema="sealed-official-ELM-v2-primary",
        detailed_kernel_hash="d"^64,
        morphology_hash="e"^64,
        frozen_twin_parameter_hash=twin_parameter_sha256,
        frozen_twin_artifact_hash=twin_artifact_sha256,
        distillation_dataset_hash="f"^64,
        distillation_config_hash="1"^64,
    )
end

@testset "contract-fixed official1278 -> frozen 11-state entrypoint" begin
    loader = Chain.assert_contract_fixed_loader!()
    @test loader.evaluator_source_sha256 ==
        Chain.CORRECTED_EVALUATOR_SOURCE_SHA256
    @test Chain.Bridge.Bridge.Sealed === Chain.Canonical
    @test Main.Sealed === Chain.Canonical

    source_artifact = "2"^64
    source_manifest = "3"^64
    teacher_contract = "4"^64
    twin_parameter = "5"^64
    twin_artifact = "6"^64
    attestation = "7"^64
    identity = (;
        source_artifact_sha256=source_artifact,
        source_manifest_sha256=source_manifest,
        teacher_contract_sha256=teacher_contract,
        frozen_twin_parameter_sha256=twin_parameter,
        frozen_twin_artifact_sha256=twin_artifact,
        sealed_attestation_sha256=attestation,
        corrected_evaluator_source_sha256=
            Chain.CORRECTED_EVALUATOR_SOURCE_SHA256,
    )
    parameters =
        _fixture_parameters(twin_parameter, twin_artifact)
    payload = (;
        frozen_internal=true,
        ablation_mode=:full,
        provisional=false,
        frozen_twin_integrity_before=true,
        frozen_twin_integrity_after=true,
        parameters,
        parameter_sha256=
            FrozenCell.parameter_sha256(parameters),
        source_manifest_sha256=source_manifest,
        source_teacher_contract_sha256=teacher_contract,
        frozen_twin_artifact_hash=twin_artifact,
        frozen_twin_parameter_hash=twin_parameter,
        source_bound_sealed_elm=(;
            sealed_artifact_sha256=source_artifact,
            source_manifest_sha256=source_manifest,
            source_teacher_contract_sha256=teacher_contract,
            sealed_attestation_sha256=attestation,
            base_artifact_sha256=twin_artifact,
            parameter_sha256=twin_parameter,
        ),
    )
    verified =
        Chain.verify_distilled_payload_provenance(
            payload,
            identity,
        )
    @test verified.frozen_internal
    @test verified.optimizer_visible_parameter_count == 0
    @test verified.soma_contacts == 0
    @test verified.axon_contacts == 0
    @test verified.source_artifact_sha256 == source_artifact
    @test verified.source_manifest_sha256 == source_manifest
    @test verified.teacher_contract_sha256 == teacher_contract

    @test_throws ErrorException Chain.verify_distilled_payload_provenance(
        merge(payload, (; frozen_internal=false)),
        identity,
    )
    @test_throws ErrorException Chain.verify_distilled_payload_provenance(
        merge(payload, (; source_manifest_sha256="8"^64)),
        identity,
    )
    @test_throws ErrorException Chain.ContractFixedV2SourcePins(
        source_artifact_sha256="short",
        source_manifest_sha256=source_manifest,
        teacher_contract_sha256=teacher_contract,
    ) |> Chain._normalized_pins
end
