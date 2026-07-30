# Replace the generic ProductionBundle builder with a V5Final registration
# boundary.  Construction is not complete until the sealed runtime has loaded,
# registered PaperReleaseAux, and passed lifecycle preflight.

function build_production_trainer(
    bundle::ProductionBundle,
    model,
    parameters;
    kwargs...,
)
    assert_production_bundle_unchanged!(bundle)
    size(
        bundle.distilled_parameters.compartment_projection,
    ) == (4, OFFICIAL_HAY_SEGMENTS) ||
        error("production bundle does not expose 642 locations")
    trainer = Training.PaperTrainer(
        model,
        parameters;
        cell_mode=:distilled_frozen,
        cell_artifact=bundle.distilled_path,
        kwargs...,
    )
    Training.enable_release_runtime!(
        trainer,
        bundle.distilled_path,
    )
    Training.paper_preflight_integrity!(trainer)
    Training.register_paper_trainer_aux!(trainer) isa
        Training.PaperReleaseAux ||
        error("production builder did not install V5 PaperReleaseAux")
    return trainer
end
