# Exact typed bundle for:
# verified detailed teacher -> SealedOfficialELMReleaseV2
# -> anchored frozen RuntimeV5 11-state artifact.
#
# Included into HDSWSNNTwinPropProduction after the V5 anchored arena adapter.

export SealedV5ProductionBundle,
    assert_sealed_v5_bundle_unchanged!,
    build_development_trainer,
    load_sealed_v5_production_bundle

struct SealedV5ProductionBundle{A,P}
    sealed_lineage::A
    distilled_parameters::P
    distilled_path::String
    distilled_file_sha256::String
    distilled_parameter_sha256::String
    distilled_artifact_sha256::String
    paper_scale::Bool
    promotable_production::Bool
end

function load_sealed_v5_production_bundle(
    sealed_release_path::AbstractString,
    teacher_manifest_path::AbstractString,
    teacher_shard_directory::AbstractString,
    distilled_path::AbstractString;
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
    Training._assert_distilled_v5_anchor(path, lineage)
    trusted = Training.ReleaseSecurityCell.load_release_runtime(path)
    trusted.paper_scale == lineage.paper_scale ||
        error("sealed twin/distilled runtime scale differs")
    parameters = trusted.parameters
    parameter_sha256 = Cell.parameter_sha256(parameters)
    parameter_sha256 == trusted.expected_parameter_sha256 ||
        error("trusted RuntimeV5 parameter hash differs")
    bundle = SealedV5ProductionBundle(
        lineage,
        parameters,
        path,
        _sha256_file(path),
        parameter_sha256,
        trusted.artifact_sha256,
        lineage.paper_scale,
        lineage.promotable_production,
    )
    assert_sealed_v5_bundle_unchanged!(bundle)
    return bundle
end

function assert_sealed_v5_bundle_unchanged!(
    bundle::SealedV5ProductionBundle,
)
    Training.assert_sealed_v5_lineage_anchor_unchanged!(
        bundle.sealed_lineage,
    )
    _sha256_file(bundle.distilled_path) ==
        bundle.distilled_file_sha256 ||
        error("frozen RuntimeV5 artifact file changed")
    Training._assert_distilled_v5_anchor(
        bundle.distilled_path,
        bundle.sealed_lineage,
    )
    Cell.assert_parameter_sha256(
        bundle.distilled_parameters,
        bundle.distilled_parameter_sha256,
    )
    trusted = Training.ReleaseSecurityCell.load_release_runtime(
        bundle.distilled_path,
    )
    trusted.artifact_sha256 ==
        bundle.distilled_artifact_sha256 ||
        error("RuntimeV5 artifact identity changed")
    trusted.expected_parameter_sha256 ==
        bundle.distilled_parameter_sha256 ||
        error("RuntimeV5 parameter identity changed")
    trusted.paper_scale == bundle.paper_scale ||
        error("RuntimeV5 scale declaration changed")
    return bundle
end

# The legacy ProductionBundle owns PaperDigitalTwin rather than the exact
# SealedOfficialELMReleaseV2 and can never enter the V5 canonical builder.
function build_production_trainer(
    ::ProductionBundle,
    model,
    parameters;
    kwargs...,
)
    error(
        "legacy ProductionBundle is forbidden; use " *
        "load_sealed_v5_production_bundle",
    )
end

function build_production_trainer(
    bundle::SealedV5ProductionBundle,
    model,
    parameters;
    kwargs...,
)
    assert_sealed_v5_bundle_unchanged!(bundle)
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
    )
    Training.paper_preflight_integrity!(trainer)
    Training.register_paper_trainer_aux!(trainer) isa
        Training.PaperReleaseAux ||
        error("V5 production builder did not install PaperReleaseAux")
    return trainer
end

function build_development_trainer(
    bundle::SealedV5ProductionBundle,
    model,
    parameters;
    development_scale_chain::Bool,
    kwargs...,
)
    development_scale_chain === true ||
        error("development_scale_chain=true is required")
    assert_sealed_v5_bundle_unchanged!(bundle)
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
    )
    Training.paper_preflight_integrity!(trainer)
    Training.register_paper_trainer_aux!(trainer) isa
        Training.PaperReleaseAux ||
        error("V5 development builder did not install PaperReleaseAux")
    return trainer
end
