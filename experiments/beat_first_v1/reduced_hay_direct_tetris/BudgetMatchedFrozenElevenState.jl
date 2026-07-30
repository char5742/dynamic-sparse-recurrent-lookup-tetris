module BudgetMatchedFrozenElevenState

using Lux
using Random

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(_PARENT_MODULE, :PaperArenaTraining)
    Base.include(
        _PARENT_MODULE,
        joinpath(
            @__DIR__,
            "..",
            "paper_multicompartment_snn",
            "PaperArenaTraining.jl",
        ),
    )
end

const Training = getfield(_PARENT_MODULE, :PaperArenaTraining)
const Canonical = getfield(_PARENT_MODULE, :PaperModelCanonical)
const BaseModel = Canonical.BaseModel

export build_budget_frozen_model,
    build_budget_frozen_trainer,
    budget_frozen_topology

"""
Build the frozen 11-state comparison topology.

Thirty-four frozen cells expose 374 internal state scalars, within 1.7% of the
Reduced Hay arm. The artifact remains mandatory and fail-closed when a trainer
is constructed; no distillation parameters enter the Tetris optimizer tree.
"""
function build_budget_frozen_model()
    base = BaseModel.PaperModel(
        blocks=34,
        cycles=3,
        substeps_per_cycle=1,
        workspace_k=4,
        hidden=32,
        sensory_contacts=8,
        recurrent_contacts=4,
    )
    return Canonical.PaperModel(base)
end

function build_budget_frozen_trainer(
    artifact::AbstractString;
    rng::AbstractRNG=MersenneTwister(0x46524f5a),
    state_batch::Int=1,
    width::Int=80,
    learning_rate::Real=3.0f-4,
    weight_decay::Real=1.0f-5,
)
    source = abspath(artifact)
    isfile(source) || error("frozen 11-state artifact is absent: $source")
    model = build_budget_frozen_model()
    parameters = Lux.initialparameters(rng, model)
    trainer = Training.PaperTrainer(
        model,
        parameters;
        state_batch,
        width,
        learning_rate,
        weight_decay,
        cell_mode=:distilled_frozen,
        cell_artifact=source,
    )
    return (; model, trainer)
end

function budget_frozen_topology(model=build_budget_frozen_model())
    return (;
        family=:budget_matched_frozen_eleven_state,
        cells=model.blocks,
        persistent_states_per_cell=11,
        persistent_state_scalars=11model.blocks,
        cycles=model.cycles,
        substeps_per_cycle=model.substeps_per_cycle,
        dynamic_sparse=true,
        internal_credit=:frozen,
    )
end

end # module BudgetMatchedFrozenElevenState
