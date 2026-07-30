# Canonical typed bundle owning the verified sealed-V2 twin and an externally
# pinned RuntimeV5-format frozen 11-state artifact.

export PinnedV5ProductionBundle,
    assert_pinned_v5_bundle_unchanged!,
    load_pinned_v5_production_bundle

struct PinnedV5ProductionBundle{A,P}
    sealed_lineage::A
    distilled_parameters::P
    distilled_path::String
    expected_distilled_artifact_sha256::String
    distilled_parameter_sha256::String
    paper_scale::Bool
    promotable_production::Bool
end

function load_pinned_v5_production_bundle(
    sealed_release_path::AbstractString,
    teacher_manifest_path::AbstractString,
    teacher_shard_directory::AbstractString,
    distilled_path::AbstractString;
    expected_distilled_artifact_sha256::AbstractString,
    require_production::Bool,
    scratch_root=nothing,
)
    lineage = Training.verify_sealed_v5_lineage_anchor(
        sealed_release_path,
        teacher_manifest_path,
        teacher_shard_directory;
        require_production,
        scratch_root,
    )
    path = abspath(distilled_path)
    expected =
        lowercase(String(expected_distilled_artifact_sha256))
    occursin(r"^[0-9a-f]{64}$", expected) ||
        error("external frozen V5 pin is not a SHA-256")
    Training._assert_runtime_v6_source_anchor(path, lineage)
    trusted = Training.ReleasePinnedCell.load_release_runtime(
        path;
        expected_artifact_sha256=expected,
        expected_sealed_attestation_sha256=
            lineage.attestation_sha256,
    )
    trusted.paper_scale == lineage.paper_scale ||
        error("sealed V2/frozen V5 scale differs")
    bundle = PinnedV5ProductionBundle(
        lineage,
        trusted.parameters,
        path,
        expected,
        trusted.expected_parameter_sha256,
        lineage.paper_scale,
        lineage.promotable_production,
    )
    assert_pinned_v5_bundle_unchanged!(bundle)
    return bundle
end

function assert_pinned_v5_bundle_unchanged!(
    bundle::PinnedV5ProductionBundle,
)
    Training.assert_sealed_v5_lineage_anchor_unchanged!(
        bundle.sealed_lineage,
    )
    Training._assert_runtime_v6_source_anchor(
        bundle.distilled_path,
        bundle.sealed_lineage,
    )
    trusted = Training.ReleasePinnedCell.load_release_runtime(
        bundle.distilled_path;
        expected_artifact_sha256=
            bundle.expected_distilled_artifact_sha256,
        expected_sealed_attestation_sha256=
            bundle.sealed_lineage.attestation_sha256,
    )
    Cell.assert_parameter_sha256(
        bundle.distilled_parameters,
        bundle.distilled_parameter_sha256,
    )
    trusted.expected_parameter_sha256 ==
        bundle.distilled_parameter_sha256 ||
        error("pinned frozen V5 parameter identity changed")
    trusted.paper_scale == bundle.paper_scale ||
        error("pinned frozen V5 scale changed")
    return bundle
end

function build_production_trainer(
    ::ProductionBundle,
    model,
    parameters;
    kwargs...,
)
    error("legacy PaperDigitalTwin ProductionBundle is forbidden")
end

function build_production_trainer(
    ::SealedV5ProductionBundle,
    model,
    parameters;
    kwargs...,
)
    error("unpinned SealedV5ProductionBundle is forbidden")
end

function build_production_trainer(
    bundle::PinnedV5ProductionBundle,
    model,
    parameters;
    kwargs...,
)
    assert_pinned_v5_bundle_unchanged!(bundle)
    bundle.paper_scale === true ||
        error("production builder requires paper_scale=true")
    bundle.promotable_production === true ||
        error("production builder requires promotable sealed V2")
    trainer = Training.PaperTrainer(
        model,
        parameters;
        cell_mode=:distilled_frozen,
        cell_artifact=bundle.distilled_path,
        kwargs...,
    )
    Training.enable_release_runtime!(
        trainer,
        bundle.distilled_path;
        sealed_lineage=bundle.sealed_lineage,
        expected_distilled_artifact_sha256=
            bundle.expected_distilled_artifact_sha256,
    )
    Training.paper_preflight_integrity!(trainer)
    Training.register_paper_trainer_aux!(trainer) isa
        Training.PaperReleaseAux ||
        error("pinned production builder did not install PaperReleaseAux")
    return trainer
end

function build_development_trainer(
    ::SealedV5ProductionBundle,
    model,
    parameters;
    kwargs...,
)
    error("unpinned SealedV5ProductionBundle is forbidden")
end

function build_development_trainer(
    bundle::PinnedV5ProductionBundle,
    model,
    parameters;
    development_scale_chain::Bool,
    kwargs...,
)
    development_scale_chain === true ||
        error("development_scale_chain=true is required")
    assert_pinned_v5_bundle_unchanged!(bundle)
    bundle.paper_scale === false ||
        error("development builder received paper-scale bundle")
    bundle.promotable_production === false ||
        error("development bundle is unexpectedly promotable")
    trainer = Training.PaperTrainer(
        model,
        parameters;
        cell_mode=:distilled_frozen,
        cell_artifact=bundle.distilled_path,
        kwargs...,
    )
    Training.enable_development_release_runtime!(
        trainer,
        bundle.distilled_path;
        development_scale_chain,
        sealed_lineage=bundle.sealed_lineage,
        expected_distilled_artifact_sha256=
            bundle.expected_distilled_artifact_sha256,
    )
    Training.paper_preflight_integrity!(trainer)
    Training.register_paper_trainer_aux!(trainer) isa
        Training.PaperReleaseAux ||
        error("pinned development builder did not install PaperReleaseAux")
    return trainer
end
