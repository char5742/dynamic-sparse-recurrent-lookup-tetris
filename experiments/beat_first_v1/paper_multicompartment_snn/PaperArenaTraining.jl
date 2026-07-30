module PaperArenaTraining

using LinearAlgebra
using Random
using SHA
using Statistics

if !isdefined(Main, :ArenaWorkspaceTraining)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "..",
            "serial_workspace_snn",
            "ArenaWorkspaceTraining.jl",
        ),
    )
end
if !isdefined(Main, :PaperModelCanonical)
    Base.include(
        Main,
        joinpath(@__DIR__, "PaperModelCanonical.jl"),
    )
end

include(joinpath(@__DIR__, "PaperHayCell.jl"))
include(joinpath(@__DIR__, "DistilledElevenStateCell.jl"))
include(joinpath(@__DIR__, "PaperOptimizer.jl"))

const Point = Main.ArenaWorkspaceTraining
const InputModel = Main.SerialWorkspaceSNN
const Routing = Main.WorkspaceRoutingPolicy
const Model = Main.PaperModelCanonical
const Hay = PaperHayCell
const Distilled = DistilledElevenStateCell
const Optim = PaperOptimizer

include(joinpath(@__DIR__, "PaperArenaCore.jl"))
include(joinpath(@__DIR__, "PaperCellAdapter.jl"))
include(joinpath(@__DIR__, "PaperCompatibility.jl"))

# Supersede the early loader with the strict detailed→twin→distilled lineage
# audit before any trainer auxiliary state can be registered.
_load_lineage(
    path::AbstractString,
    parameters::Distilled.DistilledParameters,
) = _load_lineage_v2(path, parameters)

include(joinpath(@__DIR__, "PaperArenaLearning.jl"))
include(joinpath(@__DIR__, "PaperArenaForward.jl"))
include(joinpath(@__DIR__, "PaperArenaReplayV2.jl"))

# The v2 replay carries receptor-specific eligibility, local NMDA/V
# sensitivity and 11-state block-local predictor errors. The initial scalar
# replay remains source-auditable but is not dispatched by production.
paper_local_replay_candidate!(
    worker::PaperWorker,
    trainer::PaperTrainer,
    flat::Int,
) = paper_local_replay_candidate_v2!(worker, trainer, flat)

include(joinpath(@__DIR__, "PaperArenaExecutor.jl"))

function paper_internal_sha256(trainer::PaperTrainer)
    return paper_internal_artifact_sha256(trainer)
end

function paper_internal_parameter_sha256(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    if trainer.cell_mode === :distilled_frozen
        return Distilled.parameter_sha256(aux.internal_parameters)
    end
    return aux.lineage.detailed_cell_sha256
end

function paper_aux_snapshot(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    return (;
        input_location=copy(trainer.input_location),
        recurrent_location=copy(trainer.recurrent_location),
        workspace_location=copy(aux.workspace_location),
        input_location_utility=
            copy(trainer.input_location_utility),
        recurrent_location_utility=
            copy(trainer.recurrent_location_utility),
        workspace_location_utility=
            copy(aux.workspace_location_utility),
        contact_capacity=(
            excitatory=copy(aux.excitatory_capacity),
            inhibitory=copy(aux.inhibitory_capacity),
        ),
        regional_projection=copy(aux.regional_projection),
        lineage=aux.lineage,
    )
end

function restore_paper_aux_snapshot!(
    trainer::PaperTrainer,
    snapshot,
)
    aux = register_paper_trainer_aux!(trainer)
    copyto!(trainer.input_location, snapshot.input_location)
    copyto!(
        trainer.recurrent_location,
        snapshot.recurrent_location,
    )
    copyto!(aux.workspace_location, snapshot.workspace_location)
    copyto!(
        trainer.input_location_utility,
        snapshot.input_location_utility,
    )
    copyto!(
        trainer.recurrent_location_utility,
        snapshot.recurrent_location_utility,
    )
    copyto!(
        aux.workspace_location_utility,
        snapshot.workspace_location_utility,
    )
    copyto!(
        aux.excitatory_capacity,
        snapshot.contact_capacity.excitatory,
    )
    copyto!(
        aux.inhibitory_capacity,
        snapshot.contact_capacity.inhibitory,
    )
    copyto!(aux.regional_projection, snapshot.regional_projection)
    snapshot.lineage == aux.lineage ||
        error("checkpoint cell lineage differs")
    return trainer
end

export PaperExecutor,
    PaperMetrics,
    PaperTrainer,
    ROUTING_REWARD_SEMANTICS,
    internal_cell_max_delta,
    paper_ablation,
    paper_arena_output,
    paper_arena_update!,
    paper_aux_snapshot,
    paper_internal_artifact_sha256,
    paper_internal_lineage,
    paper_internal_max_delta,
    paper_internal_parameter_sha256,
    paper_internal_sha256,
    paper_parameter_deltas,
    paper_training_arena,
    restore_paper_aux_snapshot!,
    run_with_paper_team!,
    set_paper_ablation!

end
