const _TRAINER_ABLATION = IdDict{Any,Symbol}()

function _load_lineage_v2(
    path::AbstractString,
    parameters::Distilled.DistilledParameters,
)
    data = JLD2.load(path)
    haskey(data, "payload") ||
        error("distilled artifact has no payload")
    payload = data["payload"]
    detailed = _payload_field(
        payload,
        (
            :detailed_cell_sha256,
            :cell_mechanism_sha256,
            :detailed_sha256,
            :mechanism_sha256,
        ),
    )
    twin = _payload_field(
        payload,
        (:digital_twin_sha256, :twin_sha256, :teacher_sha256),
    )
    if hasproperty(payload, :lineage)
        isempty(detailed) && (detailed = _payload_field(
            payload.lineage,
            (
                :detailed_cell_sha256,
                :cell_mechanism_sha256,
                :detailed_sha256,
            ),
        ))
        isempty(twin) && (twin = _payload_field(
            payload.lineage,
            (:digital_twin_sha256, :twin_sha256),
        ))
    end
    isempty(twin) && (twin = parameters.teacher_sha256)
    isempty(detailed) &&
        error("artifact lineage lacks detailed-cell SHA-256")
    isempty(twin) &&
        error("artifact lineage lacks frozen digital-twin SHA-256")
    twin == parameters.teacher_sha256 ||
        error("distilled teacher hash differs from lineage twin hash")
    occursin("twin", lowercase(parameters.teacher_schema)) ||
        error("distilled teacher schema is not a digital-twin schema")
    return PaperLineage(
        detailed,
        twin,
        Distilled.artifact_sha256(path),
        Distilled.parameter_sha256(parameters),
        Distilled.DISTILLED_ARTIFACT_SCHEMA,
    )
end

function _artifact_ablation(path::AbstractString)
    data = JLD2.load(path)
    payload = data["payload"]
    value = if hasproperty(payload, :ablation_mode)
        getproperty(payload, :ablation_mode)
    elseif hasproperty(payload, :lineage) &&
           hasproperty(payload.lineage, :ablation_mode)
        getproperty(payload.lineage, :ablation_mode)
    else
        :full
    end
    return Symbol(value)
end

function set_paper_ablation!(
    trainer::PaperTrainer,
    ablation::Symbol,
)
    ablation in (:full, :passive, :no_nmda, :soma_only) ||
        throw(ArgumentError("unsupported paper ablation $ablation"))
    if trainer.cell_mode === :distilled_frozen
        artifact_mode = _artifact_ablation(something(trainer.cell_artifact))
        artifact_mode == ablation || error(
            "distilled ablation requires its own detailed→twin→distilled " *
            "artifact: requested=$ablation artifact=$artifact_mode",
        )
    end
    _TRAINER_ABLATION[trainer] = ablation
    return trainer
end

paper_ablation(trainer::PaperTrainer) =
    get(_TRAINER_ABLATION, trainer, :full)

# Canonical runtime constructor. This method intentionally supersedes the
# initial adapter constructor so the detailed control receives the requested
# Hay ablation while distilled runs can only use a matching variant artifact.
function make_cell_runtime_v2(trainer::PaperTrainer)
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

paper_internal_parameter_sha256(trainer::PaperTrainer) =
    register_paper_trainer_aux!(trainer).lineage.distilled_parameter_sha256

paper_internal_artifact_sha256(trainer::PaperTrainer) =
    register_paper_trainer_aux!(trainer).lineage.distilled_artifact_sha256

# Checkpoint compatibility: this name is the artifact/source digest, while
# `paper_internal_parameter_sha256` is the loaded in-memory array digest.
paper_internal_sha256_v2(trainer::PaperTrainer) =
    paper_internal_artifact_sha256(trainer)

paper_internal_lineage(trainer::PaperTrainer) =
    register_paper_trainer_aux!(trainer).lineage
