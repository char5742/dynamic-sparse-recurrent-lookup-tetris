# External authenticity anchor for RuntimeV5Final.
#
# A distilled JLD2 payload is never trusted to prove its own sealed lineage.
# Registration requires a SealedV2LineageAnchor produced by independently
# reloading and raw-evidence-verifying the exact final.v2 sealed ELM bundle.

using SHA

if !isdefined(Main, :PaperELMTwinOfficialV2SealedReleaseV2)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "PaperELMTwinOfficialV2SealedReleaseV2.jl",
        ),
    )
end
const SealedV2 =
    Main.PaperELMTwinOfficialV2SealedReleaseV2

export SealedV2LineageAnchor,
    assert_sealed_v5_lineage_anchor_unchanged!,
    verify_sealed_v5_lineage_anchor

struct _SealedV2LineageToken end
const _SEALED_V2_LINEAGE_TOKEN =
    _SealedV2LineageToken()

struct SealedV2LineageAnchor{B}
    bundle::B
    sealed_release_path::String
    sealed_release_file_sha256::String
    teacher_manifest_path::String
    teacher_manifest_sha256::String
    teacher_shard_directory::String
    teacher_inventory_sha256::String
    attestation_sha256::String
    teacher_contract_sha256::String
    source_dataset_sha256::String
    paper_scale::Bool
    promotable_production::Bool

    function SealedV2LineageAnchor(
        ::_SealedV2LineageToken,
        bundle::B,
        sealed_release_path,
        sealed_release_file_sha256,
        teacher_manifest_path,
        teacher_manifest_sha256,
        teacher_shard_directory,
        teacher_inventory_sha256,
        attestation_sha256,
        teacher_contract_sha256,
        source_dataset_sha256,
        paper_scale,
        promotable_production,
    ) where {B}
        bundle isa SealedV2.SealedOfficialELMRelease ||
            error("lineage anchor requires exact sealed release V2 type")
        return new{B}(
            bundle,
            String(sealed_release_path),
            String(sealed_release_file_sha256),
            String(teacher_manifest_path),
            String(teacher_manifest_sha256),
            String(teacher_shard_directory),
            String(teacher_inventory_sha256),
            String(attestation_sha256),
            String(teacher_contract_sha256),
            String(source_dataset_sha256),
            Bool(paper_scale),
            Bool(promotable_production),
        )
    end
end

function _v5_file_sha256(path::AbstractString)
    source = abspath(path)
    isfile(source) || error("lineage file is absent: $source")
    return bytes2hex(SHA.sha256(read(source)))
end

function _v5_teacher_inventory_sha256(
    directory::AbstractString,
)
    root = abspath(directory)
    isdir(root) || error("teacher shard directory is absent: $root")
    files = String[]
    for (parent, _, names) in walkdir(root)
        for name in names
            lower = lowercase(name)
            if lower == "manifest.json" ||
                    endswith(lower, ".npz") ||
                    endswith(lower, ".done.json")
                push!(files, joinpath(parent, name))
            end
        end
    end
    isempty(files) && error("teacher shard inventory is empty")
    sort!(files; by=path -> lowercase(relpath(path, root)))
    context = SHA.SHA256_CTX()
    for path in files
        relative = replace(relpath(path, root), '\\' => '/')
        SHA.update!(context, codeunits(relative))
        SHA.update!(context, UInt8[0x00])
        SHA.update!(context, codeunits(_v5_file_sha256(path)))
        SHA.update!(context, UInt8[0x0a])
    end
    return bytes2hex(SHA.digest!(context))
end

function verify_sealed_v5_lineage_anchor(
    sealed_release_path::AbstractString,
    teacher_manifest_path::AbstractString,
    teacher_shard_directory::AbstractString;
    require_production::Bool,
    scratch_root=nothing,
)
    sealed_path = abspath(sealed_release_path)
    manifest_path = abspath(teacher_manifest_path)
    shard_directory = abspath(teacher_shard_directory)
    bundle = SealedV2.load_verified_sealed_official_elm_release(
        sealed_path,
        manifest_path,
        shard_directory;
        require_production,
        scratch_root,
    )
    bundle isa SealedV2.SealedOfficialELMRelease ||
        error("verified sealed loader returned the wrong exact type")
    payload = bundle.attestation.payload
    payload.schema == SealedV2.SEALED_RELEASE_SCHEMA ||
        error("verified sealed release schema differs")
    payload.artifact_kind ==
        SealedV2.SEALED_RELEASE_ARTIFACT_KIND ||
        error("verified sealed release kind differs")
    payload.outcome.gate_passed === true ||
        error("verified sealed release held-out gate failed")
    paper_scale = Bool(payload.outcome.paper_scale)
    promotable = Bool(payload.outcome.promotable_production)
    require_production && !promotable &&
        error("production anchor is not promotable")
    !require_production && paper_scale &&
        error("development anchor unexpectedly claims paper scale")

    return SealedV2LineageAnchor(
        _SEALED_V2_LINEAGE_TOKEN,
        bundle,
        sealed_path,
        _v5_file_sha256(sealed_path),
        manifest_path,
        _v5_file_sha256(manifest_path),
        shard_directory,
        _v5_teacher_inventory_sha256(shard_directory),
        bundle.attestation.attestation_sha256,
        payload.teacher.teacher_contract_sha256,
        payload.teacher.source_dataset_sha256,
        paper_scale,
        promotable,
    )
end

function assert_sealed_v5_lineage_anchor_unchanged!(
    anchor::SealedV2LineageAnchor,
)
    _v5_file_sha256(anchor.sealed_release_path) ==
        anchor.sealed_release_file_sha256 ||
        error("sealed release V2 file changed")
    _v5_file_sha256(anchor.teacher_manifest_path) ==
        anchor.teacher_manifest_sha256 ||
        error("sealed teacher manifest changed")
    _v5_teacher_inventory_sha256(
        anchor.teacher_shard_directory,
    ) == anchor.teacher_inventory_sha256 ||
        error("sealed teacher shard inventory changed")
    SealedV2.canonical_sha256(
        anchor.bundle.attestation.payload,
    ) == anchor.attestation_sha256 ||
        error("in-memory sealed V2 attestation changed")
    SealedV2.Twin.assert_frozen_official_elm_unchanged(
        anchor.bundle.frozen,
    )
    return anchor
end

const _TRAINER_SEALED_V2_LINEAGE =
    IdDict{Any,SealedV2LineageAnchor}()

function _assert_distilled_v5_anchor(
    path::AbstractString,
    anchor::SealedV2LineageAnchor,
)
    data = JLD2.load(path)
    payload = data["payload"]
    String(_payload_release_value(
        payload,
        :sealed_attestation_sha256,
        "",
    )) == anchor.attestation_sha256 ||
        error("distilled artifact/sealed V2 attestation differs")
    String(_payload_release_value(
        payload,
        :sealed_release_file_sha256,
        "",
    )) == anchor.sealed_release_file_sha256 ||
        error("distilled artifact lacks exact sealed V2 file identity")
    String(_payload_release_value(
        payload,
        :source_manifest_sha256,
        "",
    )) == anchor.teacher_manifest_sha256 ||
        error("distilled artifact/teacher manifest differs")
    String(_payload_release_value(
        payload,
        :source_teacher_contract_sha256,
        "",
    )) == anchor.teacher_contract_sha256 ||
        error("distilled artifact/teacher contract differs")
    String(_payload_release_value(
        payload,
        :source_dataset_sha256,
        "",
    )) == anchor.source_dataset_sha256 ||
        error("distilled artifact/source dataset differs")
    Bool(_payload_release_value(
        payload,
        :paper_scale,
        !anchor.paper_scale,
    )) == anchor.paper_scale ||
        error("distilled artifact/sealed V2 scale differs")
    return true
end

function _load_trusted_release_runtime_v5(
    trainer::PaperTrainer,
    artifact_path::AbstractString,
    sealed_lineage::SealedV2LineageAnchor,
)
    assert_sealed_v5_lineage_anchor_unchanged!(sealed_lineage)
    path = abspath(artifact_path)
    _assert_distilled_v5_anchor(path, sealed_lineage)
    trainer.cell_mode === :distilled_frozen ||
        error("release runtime requires cell_mode=:distilled_frozen")
    trainer.cell_artifact == path ||
        error("trainer artifact path differs from V5 artifact")
    trusted = ReleaseSecurityCell.load_release_runtime(path)
    ReleaseSecurityCell.preflight_integrity!(trusted)
    return path, trusted
end

function enable_release_runtime!(
    trainer::PaperTrainer,
    artifact_path::AbstractString=something(trainer.cell_artifact);
    sealed_lineage::SealedV2LineageAnchor,
)
    path, trusted = _load_trusted_release_runtime_v5(
        trainer,
        artifact_path,
        sealed_lineage,
    )
    sealed_lineage.promotable_production === true ||
        error("production V5 anchor is not promotable")
    trusted.paper_scale === true ||
        error("production V5 artifact is not paper_scale")
    aux = _register_trusted_release_runtime_v5!(
        trainer,
        trusted,
        path,
        :production,
    )
    _TRAINER_SEALED_V2_LINEAGE[trainer] = sealed_lineage
    return aux
end

function enable_development_release_runtime!(
    trainer::PaperTrainer,
    artifact_path::AbstractString=something(trainer.cell_artifact);
    development_scale_chain::Bool,
    sealed_lineage::SealedV2LineageAnchor,
)
    development_scale_chain === true ||
        error("development V5 chain flag is required")
    path, trusted = _load_trusted_release_runtime_v5(
        trainer,
        artifact_path,
        sealed_lineage,
    )
    sealed_lineage.paper_scale === false ||
        error("development V5 anchor claims paper scale")
    sealed_lineage.promotable_production === false ||
        error("development V5 anchor is unexpectedly promotable")
    trusted.paper_scale === false ||
        error("development V5 artifact claims paper scale")
    aux = _register_trusted_release_runtime_v5!(
        trainer,
        trusted,
        path,
        :development,
    )
    _TRAINER_SEALED_V2_LINEAGE[trainer] = sealed_lineage
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
        error("trainer has no externally verified sealed V2 anchor")
    assert_sealed_v5_lineage_anchor_unchanged!(anchor)
    _assert_distilled_v5_anchor(
        something(trainer.cell_artifact),
        anchor,
    )
    _assert_release_scale_mode!(trainer, aux)
    _assert_release_legal_contact_state!(aux)
    fresh = ReleaseSecurityCell.load_release_runtime(
        something(trainer.cell_artifact),
    )
    fresh.artifact_sha256 == aux.trusted.artifact_sha256 ||
        error("RuntimeV5 frozen artifact changed")
    fresh.expected_parameter_sha256 ==
        aux.trusted.expected_parameter_sha256 ||
        error("RuntimeV5 frozen parameters changed")
    ReleaseSecurityCell.preflight_integrity!(aux.trusted)
    ReleaseSecurityCell.preflight_integrity!(fresh)
    return fresh.artifact_sha256
end
