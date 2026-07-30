module ReducedHayDirectTraining

using Lux
using Optimisers
using Random
using Zygote

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(_PARENT_MODULE, :ArenaWorkspaceTraining)
    Base.include(
        _PARENT_MODULE,
        joinpath(
            @__DIR__,
            "..",
            "serial_workspace_snn",
            "ArenaWorkspaceTraining.jl",
        ),
    )
end
if !isdefined(_PARENT_MODULE, :ReducedHayWorkspaceSNN)
    Base.include(
        _PARENT_MODULE,
        joinpath(@__DIR__, "ReducedHayWorkspaceSNN.jl"),
    )
end

const Point = getfield(_PARENT_MODULE, :ArenaWorkspaceTraining)
const Reduced = getfield(_PARENT_MODULE, :ReducedHayWorkspaceSNN)

export ReducedHayDirectTrainer,
    CanonicalDirectTrainer,
    direct_gradient!,
    direct_objective!,
    direct_update!,
    gradient_group_norms,
    pack_rows!,
    parameter_max_delta,
    tree_norm

"""
Minimal direct-BPTT trainer.

The shared Point-SNN arena remains the canonical source of 1,298 binary rails,
candidate grouping and the 22-output Tetris ranking objective. Its exact raw
cotangent is injected into a reverse-mode pullback through every Reduced Hay
cycle. Thus all continuous compartment parameters receive the global Tetris
teacher signal; there is no DECOLLE/e-prop recurrent update in this path.

The prototype intentionally uses Zygote as the correctness/reference BPTT.
The CPU production path can replace this pullback with an analytic VJP while
keeping the same arena, raw cotangent and parameter contract.
"""
mutable struct ReducedHayDirectTrainer{M,F,P,S,O,A,L,G}
    model::M
    raw_function::F
    parameters::P
    model_state::S
    optimizer_state::O
    arena::A
    loss_scratch::L
    gradient::G
    last_loss::Any
    last_gradient_norm::Float64
    updates::Int
end

function _zero_like(parameters)
    return NamedTuple{keys(parameters)}(
        map(array -> zeros(Float32, size(array)), values(parameters)),
    )
end

function CanonicalDirectTrainer(
    model,
    raw_function;
    rng::AbstractRNG=MersenneTwister(0x52484454),
    state_batch::Int=1,
    width::Int=80,
    learning_rate::Real=3.0f-4,
    weight_decay::Real=1.0f-5,
)
    parameters, model_state = Lux.setup(rng, model)
    optimizer = Optimisers.AdamW(
        Float32(learning_rate),
        (0.9, 0.999),
        Float32(weight_decay),
    )
    optimizer_state = Optimisers.setup(optimizer, parameters)
    arena = Point.TrainingArena(model, state_batch, width)
    return ReducedHayDirectTrainer(
        model,
        raw_function,
        parameters,
        model_state,
        optimizer_state,
        arena,
        Point.LossScratch(width),
        _zero_like(parameters),
        nothing,
        0.0,
        0,
    )
end

function ReducedHayDirectTrainer(
    model::Reduced.ReducedHayWorkspaceModel;
    kwargs...,
)
    return CanonicalDirectTrainer(
        model,
        Reduced.reduced_hay_raw;
        kwargs...,
    )
end

function tree_norm(value)
    value === nothing && return 0.0
    value isa AbstractArray &&
        return sqrt(sum(abs2, value; init=0.0))
    value isa NamedTuple &&
        return sqrt(sum(tree_norm(child)^2 for child in values(value)))
    value isa Tuple &&
        return sqrt(sum(tree_norm(child)^2 for child in value))
    return 0.0
end

function parameter_max_delta(left, right)
    keys(left) == keys(right) || return Inf
    result = 0.0
    for name in keys(left)
        lvalue = getproperty(left, name)
        rvalue = getproperty(right, name)
        size(lvalue) == size(rvalue) || return Inf
        result = max(
            result,
            maximum(abs.(Float64.(lvalue) .- Float64.(rvalue))),
        )
    end
    return result
end

function pack_rows!(
    trainer::ReducedHayDirectTrainer,
    dataset,
    rows::AbstractVector{<:Integer},
)
    length(rows) == trainer.arena.state_batch ||
        throw(DimensionMismatch("row count differs from state batch"))
    copyto!(trainer.arena.rows, rows)
    Point.pack_arena_batch!(trainer.arena, dataset)
    return trainer
end

function _forward_pullback(trainer::ReducedHayDirectTrainer)
    rails = trainer.arena.rails
    return Zygote.pullback(
        parameters -> trainer.raw_function(
            trainer.model,
            rails,
            parameters,
        ),
        trainer.parameters,
    )
end

function direct_gradient!(
    trainer::ReducedHayDirectTrainer;
    gate_density::Real=0.5f0,
    structure_weight::Real=0.0f0,
)
    raw, pullback = _forward_pullback(trainer)
    size(raw) == size(trainer.arena.raw) ||
        throw(DimensionMismatch("Reduced Hay raw output"))
    copyto!(trainer.arena.raw, raw)
    loss = Point.loss_and_raw_gradient!(
        trainer.arena,
        trainer.loss_scratch,
        Float32(gate_density),
        Float32(structure_weight),
    )
    gradient = only(pullback(trainer.arena.raw_gradient))
    gradient === nothing &&
        error("Reduced Hay direct BPTT returned no gradient")
    norm = tree_norm(gradient)
    isfinite(norm) || error("non-finite Reduced Hay direct gradient")
    trainer.gradient = gradient
    trainer.last_gradient_norm = norm
    trainer.last_loss = loss
    return loss, gradient
end

function direct_objective!(
    trainer::ReducedHayDirectTrainer;
    gate_density::Real=0.5f0,
    structure_weight::Real=0.0f0,
)
    loss, _ = direct_gradient!(
        trainer;
        gate_density,
        structure_weight,
    )
    return loss
end

function direct_update!(
    trainer::ReducedHayDirectTrainer;
    gate_density::Real=0.5f0,
    structure_weight::Real=0.0f0,
)
    loss, gradient = direct_gradient!(
        trainer;
        gate_density,
        structure_weight,
    )
    optimizer_state, parameters = Optimisers.update(
        trainer.optimizer_state,
        trainer.parameters,
        gradient,
    )
    trainer.optimizer_state = optimizer_state
    trainer.parameters = parameters
    trainer.updates += 1
    return loss
end

function gradient_group_norms(gradient)
    compartment_names = (
        :input_exc_gain,
        :input_inh_gain,
        :branch_bias,
        :branch_leak_logits,
        :ampa_decay_logits,
        :nmda_decay_logits,
        :gaba_decay_logits,
        :current_gain_logits,
        :axial_gain_logits,
        :nmda_slope_logits,
        :nmda_half_logits,
        :plateau_decay_logits,
        :plateau_threshold_logits,
        :plateau_slope_logits,
        :plateau_gain_logits,
        :plateau_feedback_logits,
        :soma_coupling,
        :apical_leak_logits,
        :soma_leak_logits,
        :adaptation_decay_logits,
        :apical_gain_logits,
        :soma_threshold_logits,
        :adaptation_gain_logits,
    )
    graph_names = (
        :synapse_weight,
        :gate_logits,
        :delay_logits,
    )
    routing_names = (
        :query_weight,
        :workspace_key,
        :feedback_gain,
        :workspace_decay_logit,
    )
    head_names = (
        :head_weight,
        :head_bias,
        :output_weight,
        :output_bias,
    )
    group(names) = sqrt(sum(
        tree_norm(getproperty(gradient, name))^2
        for name in names
    ))
    return (;
        compartment=group(compartment_names),
        graph=group(graph_names),
        routing=group(routing_names),
        head=group(head_names),
    )
end

end # module ReducedHayDirectTraining
