# Fail-close the inherited generic distilled-cell registration path.
#
# A frozen distilled trainer must already have a PaperReleaseAux installed by
# the V5Final security adapter.  Direct PaperWorker/make_cell_runtime calls can
# no longer deserialize an arbitrary DistilledElevenStateCellFinal artifact.

function register_paper_trainer_aux!(trainer::PaperTrainer)
    haskey(_TRAINER_AUX, trainer) && return _TRAINER_AUX[trainer]
    trainer.cell_mode === :distilled_frozen &&
        error(
            "frozen distilled trainer is not registered through " *
            "RuntimeV5Final sealed lineage",
        )

    # Preserve the explicit detailed-Hay control arm.  It has no frozen
    # distilled artifact and therefore does not cross the release boundary.
    tree = Hay.paper_hay_tree()
    catalog = paper_location_catalog()
    capacity = _location_capacity(tree)
    workspace_location = Matrix{UInt8}(
        undef,
        trainer.model.workspace_contacts,
        trainer.model.blocks,
    )
    tuft = isempty(tree.tuft_terminals) ?
        last(catalog) : UInt8(first(tree.tuft_terminals))
    @inbounds for block in 1:trainer.model.blocks
        for contact in 1:trainer.model.workspace_contacts
            workspace_location[contact, block] =
                catalog[mod1(
                    Int(tuft) + contact + 3block,
                    length(catalog),
                )]
        end
    end
    parameters = Hay.HayParameters(tree; ablation=:full)
    detailed_hash = _source_sha256(
        joinpath(@__DIR__, "PaperHayCell.jl"),
    )
    lineage = PaperLineage(
        detailed_hash,
        "detailed-control-no-digital-twin",
        "detailed-control-no-distilled-artifact",
        detailed_hash,
        "paper-hay-detailed-control-v1",
    )
    aux = PaperTrainerAux(
        parameters,
        deepcopy(parameters),
        lineage,
        catalog,
        copy(capacity),
        copy(capacity),
        workspace_location,
        zeros(
            Float32,
            length(catalog),
            trainer.model.workspace_contacts,
            trainer.model.blocks,
        ),
        _regional_projection(trainer.model.blocks),
    )
    _TRAINER_AUX[trainer] = aux
    return aux
end
