const _TRAINER_ABLATION = IdDict{Any,Symbol}()
const MINIMUM_PRODUCTION_SPIKE_AUROC = 0.985

@inline function _payload_value(payload, name::Symbol, default=nothing)
    if payload isa AbstractDict
        return get(payload, name, get(payload, String(name), default))
    end
    return hasproperty(payload, name) ?
        getproperty(payload, name) : default
end

function _load_lineage_final(
    path::AbstractString,
    parameters::Distilled.DistilledParameters,
)
    data = JLD2.load(path)
    payload = data["payload"]
    gate = _payload_value(payload, :gate)
    gate === nothing &&
        error("final distilled artifact lacks held-out gate")
    _payload_value(gate, :passed, false) === true ||
        error("final distilled artifact gate did not pass")
    auroc = Float64(
        _payload_value(gate, :held_out_spike_auroc, NaN),
    )
    isfinite(auroc) && auroc >= MINIMUM_PRODUCTION_SPIKE_AUROC ||
        error(
            "production distilled AUROC $auroc is below " *
            "$(MINIMUM_PRODUCTION_SPIKE_AUROC)",
        )
    String(_payload_value(payload, :detailed_kernel_hash, "")) ==
        parameters.detailed_kernel_hash ||
        error("detailed-kernel lineage mismatch")
    String(_payload_value(payload, :morphology_hash, "")) ==
        parameters.morphology_hash ||
        error("morphology lineage mismatch")
    String(_payload_value(payload, :digital_twin_hash, "")) ==
        parameters.frozen_twin_artifact_hash ||
        error("digital-twin lineage mismatch")
    return PaperLineage(
        parameters.detailed_kernel_hash,
        parameters.frozen_twin_artifact_hash,
        Distilled.artifact_sha256(path),
        Distilled.parameter_sha256(parameters),
        Distilled.DISTILLED_ARTIFACT_SCHEMA,
    )
end

function set_paper_ablation!(
    trainer::PaperTrainer,
    ablation::Symbol,
)
    ablation in (:full, :passive, :no_nmda, :soma_only) ||
        throw(ArgumentError("unsupported paper ablation $ablation"))
    trainer.cell_mode === :distilled_frozen && ablation !== :full &&
        error(
            "distilled ablation requires a separately trained final " *
            "variant artifact; canonical production accepts :full only",
        )
    _TRAINER_ABLATION[trainer] = ablation
    return trainer
end

paper_ablation(trainer::PaperTrainer) =
    get(_TRAINER_ABLATION, trainer, :full)

function make_cell_runtime_final(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    blocks = trainer.model.blocks
    if trainer.cell_mode === :distilled_frozen
        parameters = aux.internal_parameters
        return DistilledCellRuntime(
            parameters,
            deepcopy(aux.initial_internal_parameters),
            [Distilled.DistilledState(parameters) for _ in 1:blocks],
            [Distilled.DistilledDrive(parameters) for _ in 1:blocks],
            [Distilled.DistilledDiagnostics() for _ in 1:blocks],
        )
    end
    tree = Hay.paper_hay_tree()
    parameters = Hay.HayParameters(
        tree;
        ablation=paper_ablation(trainer),
    )
    return DetailedCellRuntime(
        tree,
        parameters,
        deepcopy(parameters),
        [Hay.HayState(tree, parameters) for _ in 1:blocks],
        [Hay.HaySynapticDrive(tree) for _ in 1:blocks],
        [Hay.HayDiagnostics(tree) for _ in 1:blocks],
    )
end
