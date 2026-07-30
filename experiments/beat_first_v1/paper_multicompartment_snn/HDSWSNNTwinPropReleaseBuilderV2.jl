# Replace the obsolete UInt8 fail-closed boundary after the exact arena module
# has acquired the UInt16 official-segment release overlay.
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
    return trainer
end
