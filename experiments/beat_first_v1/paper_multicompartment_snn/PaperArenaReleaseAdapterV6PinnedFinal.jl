# Pinned verifier overlay for the frozen V5 11-state artifact.
#
# RuntimeV6 is the security boundary for the RuntimeV5-format artifact: it
# requires an external byte SHA and the independently verified sealed-V2
# attestation SHA.  RuntimeV3 remains only its frozen numerical ABI.

include(joinpath(
    @__DIR__,
    "DistilledElevenStateCellReleaseRuntimeV6.jl",
))

const ReleasePinnedCell =
    DistilledElevenStateCellReleaseRuntimeV6
const RELEASE_PINNED_VERIFIER_CONTRACT_VERSION = 6
const RELEASE_REQUIRED_ARTIFACT_RUNTIME =
    :pinned_v5_artifact_v6_verifier

ReleasePinnedCell.TrustedReleaseRuntime ===
    ReleaseCell.TrustedReleaseRuntime ||
    error("RuntimeV6 does not share the frozen 11-state ABI")
ReleasePinnedCell.SEALED_RELEASE_SCHEMA ==
    SealedV2.SEALED_RELEASE_SCHEMA ||
    error("RuntimeV6 sealed schema differs from exact V2")
ReleasePinnedCell.SEALED_RELEASE_ARTIFACT_KIND ==
    SealedV2.SEALED_RELEASE_ARTIFACT_KIND ||
    error("RuntimeV6 sealed artifact kind differs from exact V2")

paper_release_adapter_contract_version() =
    RELEASE_PINNED_VERIFIER_CONTRACT_VERSION

const _TRAINER_DISTILLED_ARTIFACT_PIN =
    IdDict{Any,String}()

function _assert_runtime_v6_source_anchor(
    path::AbstractString,
    anchor::SealedV2LineageAnchor,
)
    payload = JLD2.load(path)["payload"]
    source = _payload_release_value(
        payload,
        :source_bound_sealed_elm,
        nothing,
    )
    source === nothing &&
        error("RuntimeV6 artifact lacks source-bound sealed ELM")
    sealed_payload = anchor.bundle.attestation.payload
    String(_payload_release_value(
        source,
        :source_manifest_sha256,
        "",
    )) == anchor.teacher_manifest_sha256 ||
        error("RuntimeV6 source manifest differs from anchor")
    String(_payload_release_value(
        source,
        :source_teacher_contract_sha256,
        "",
    )) == anchor.teacher_contract_sha256 ||
        error("RuntimeV6 teacher contract differs from anchor")
    String(_payload_release_value(
        source,
        :sealed_attestation_sha256,
        "",
    )) == anchor.attestation_sha256 ||
        error("RuntimeV6 sealed attestation differs from anchor")
    String(_payload_release_value(
        source,
        :parameter_sha256,
        "",
    )) == String(sealed_payload.model.parameter_sha256) ||
        error("RuntimeV6 sealed model parameter hash differs")
    String(_payload_release_value(
        source,
        :base_artifact_sha256,
        "",
    )) == String(sealed_payload.model.base_artifact_sha256) ||
        error("RuntimeV6 sealed model artifact hash differs")
    source_dataset = _payload_release_value(
        payload,
        :source_dataset_sha256,
        "",
    )
    String(source_dataset) == anchor.source_dataset_sha256 ||
        error("RuntimeV6 source dataset differs from anchor")
    Bool(_payload_release_value(
        payload,
        :paper_scale,
        !anchor.paper_scale,
    )) == anchor.paper_scale ||
        error("RuntimeV6 scale differs from sealed anchor")
    return true
end

function _load_trusted_release_runtime_v6(
    trainer::PaperTrainer,
    artifact_path::AbstractString,
    sealed_lineage::SealedV2LineageAnchor,
    expected_distilled_artifact_sha256::AbstractString,
)
    assert_sealed_v5_lineage_anchor_unchanged!(sealed_lineage)
    path = abspath(artifact_path)
    trainer.cell_mode === :distilled_frozen ||
        error("release runtime requires cell_mode=:distilled_frozen")
    trainer.cell_artifact == path ||
        error("trainer artifact path differs from pinned V5 artifact")
    _assert_runtime_v6_source_anchor(path, sealed_lineage)
    trusted = ReleasePinnedCell.load_release_runtime(
        path;
        expected_artifact_sha256=
            expected_distilled_artifact_sha256,
        expected_sealed_attestation_sha256=
            sealed_lineage.attestation_sha256,
    )
    ReleasePinnedCell.preflight_integrity!(trusted)
    return path, trusted
end

function enable_release_runtime!(
    trainer::PaperTrainer,
    artifact_path::AbstractString=something(trainer.cell_artifact);
    sealed_lineage::SealedV2LineageAnchor,
    expected_distilled_artifact_sha256::AbstractString,
)
    path, trusted = _load_trusted_release_runtime_v6(
        trainer,
        artifact_path,
        sealed_lineage,
        expected_distilled_artifact_sha256,
    )
    sealed_lineage.promotable_production === true ||
        error("production sealed anchor is not promotable")
    trusted.paper_scale === true ||
        error("production frozen V5 artifact is not paper_scale")
    aux = _register_trusted_release_runtime_v5!(
        trainer,
        trusted,
        path,
        :production,
    )
    _TRAINER_SEALED_V2_LINEAGE[trainer] = sealed_lineage
    _TRAINER_DISTILLED_ARTIFACT_PIN[trainer] =
        lowercase(String(expected_distilled_artifact_sha256))
    return aux
end

function enable_development_release_runtime!(
    trainer::PaperTrainer,
    artifact_path::AbstractString=something(trainer.cell_artifact);
    development_scale_chain::Bool,
    sealed_lineage::SealedV2LineageAnchor,
    expected_distilled_artifact_sha256::AbstractString,
)
    development_scale_chain === true ||
        error("development_scale_chain=true is required")
    path, trusted = _load_trusted_release_runtime_v6(
        trainer,
        artifact_path,
        sealed_lineage,
        expected_distilled_artifact_sha256,
    )
    sealed_lineage.paper_scale === false ||
        error("development sealed anchor claims paper scale")
    sealed_lineage.promotable_production === false ||
        error("development sealed anchor is promotable")
    trusted.paper_scale === false ||
        error("development frozen V5 artifact claims paper scale")
    aux = _register_trusted_release_runtime_v5!(
        trainer,
        trusted,
        path,
        :development,
    )
    _TRAINER_SEALED_V2_LINEAGE[trainer] = sealed_lineage
    _TRAINER_DISTILLED_ARTIFACT_PIN[trainer] =
        lowercase(String(expected_distilled_artifact_sha256))
    return aux
end

function _assert_runtime_v5_file_unchanged!(
    trainer::PaperTrainer,
    aux::PaperReleaseAux,
)
    anchor = get(
        _TRAINER_SEALED_V2_LINEAGE,
        trainer,
        nothing,
    )
    anchor isa SealedV2LineageAnchor ||
        error("trainer has no verified sealed V2 anchor")
    expected = get(
        _TRAINER_DISTILLED_ARTIFACT_PIN,
        trainer,
        "",
    )
    occursin(r"^[0-9a-f]{64}$", expected) ||
        error("trainer has no external frozen V5 artifact pin")
    assert_sealed_v5_lineage_anchor_unchanged!(anchor)
    _assert_release_scale_mode!(trainer, aux)
    _assert_release_legal_contact_state!(aux)
    path = something(trainer.cell_artifact)
    _assert_runtime_v6_source_anchor(path, anchor)
    fresh = ReleasePinnedCell.load_release_runtime(
        path;
        expected_artifact_sha256=expected,
        expected_sealed_attestation_sha256=
            anchor.attestation_sha256,
    )
    fresh.artifact_sha256 == aux.trusted.artifact_sha256 ||
        error("pinned frozen V5 artifact changed")
    fresh.expected_parameter_sha256 ==
        aux.trusted.expected_parameter_sha256 ||
        error("pinned frozen V5 parameters changed")
    ReleasePinnedCell.preflight_integrity!(aux.trusted)
    ReleasePinnedCell.preflight_integrity!(fresh)
    return fresh.artifact_sha256
end
