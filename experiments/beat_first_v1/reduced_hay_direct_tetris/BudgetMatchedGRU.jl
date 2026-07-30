module BudgetMatchedGRU

using Lux
using Random

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(_PARENT_MODULE, :SerialWorkspaceSNN)
    Base.include(
        _PARENT_MODULE,
        joinpath(
            @__DIR__,
            "..",
            "serial_workspace_snn",
            "SerialWorkspaceSNN.jl",
        ),
    )
end
const PointModel = getfield(_PARENT_MODULE, :SerialWorkspaceSNN)

export DiagonalGRUBaseline,
    budget_gru_raw,
    budget_gru_topology

"""
State-matched conventional recurrent control.

The recurrent matrices are diagonal so state count and CPU work stay in the
same order as the sparse Reduced Hay arm. This is a GRU, not an SNN: it has no
hard spike, event routing, compartment placement or plateau. A dense GRU with
the same 360-state width would exceed the target CPU budget by orders of
magnitude, so it is a separate, explicitly non-budget-matched control.
"""
struct DiagonalGRUBaseline <: Lux.AbstractLuxLayer
    blocks::Int
    node_dim::Int
    cycles::Int
    workspace_k::Int
    hidden::Int
    state_dim::Int
    readout_dim::Int
    input_feature::Vector{Int32}
end

function DiagonalGRUBaseline(;
    state_dim::Int=360,
    readout_dim::Int=24,
    cycles::Int=3,
    hidden::Int=32,
)
    state_dim > 0 || throw(ArgumentError("state_dim must be positive"))
    readout_dim > 0 || throw(ArgumentError("readout_dim must be positive"))
    state_dim % readout_dim == 0 ||
        throw(ArgumentError("state_dim must be divisible by readout_dim"))
    input_feature = Int32[
        mod1(7919index + 104729, PointModel.INPUT_RAILS)
        for index in 1:state_dim
    ]
    # The first five fields retain the shared arena shape contract. They are
    # bookkeeping only; this baseline has one recurrent vector.
    return DiagonalGRUBaseline(
        8,
        div(state_dim, 8),
        cycles,
        8,
        hidden,
        state_dim,
        readout_dim,
        input_feature,
    )
end

function Lux.initialparameters(
    rng::AbstractRNG,
    model::DiagonalGRUBaseline,
)
    state = model.state_dim
    return (;
        input_gain=0.35f0 .+
            0.05f0 .* randn(rng, Float32, state),
        update_input=0.45f0 .+
            0.05f0 .* randn(rng, Float32, state),
        update_recurrent=0.25f0 .+
            0.05f0 .* randn(rng, Float32, state),
        update_bias=fill(-0.20f0, state),
        reset_input=0.40f0 .+
            0.05f0 .* randn(rng, Float32, state),
        reset_recurrent=0.25f0 .+
            0.05f0 .* randn(rng, Float32, state),
        reset_bias=zeros(Float32, state),
        candidate_input=0.60f0 .+
            0.05f0 .* randn(rng, Float32, state),
        candidate_recurrent=0.45f0 .+
            0.05f0 .* randn(rng, Float32, state),
        candidate_bias=zeros(Float32, state),
        head_weight=0.12f0 .* randn(
            rng,
            Float32,
            model.hidden,
            2model.readout_dim,
        ) ./ sqrt(Float32(2model.readout_dim)),
        head_bias=zeros(Float32, model.hidden),
        output_weight=0.08f0 .* randn(
            rng,
            Float32,
            22,
            model.hidden,
        ) ./ sqrt(Float32(model.hidden)),
        output_bias=zeros(Float32, 22),
    )
end

Lux.initialstates(::AbstractRNG, ::DiagonalGRUBaseline) = NamedTuple()

function budget_gru_raw(
    model::DiagonalGRUBaseline,
    rails::AbstractMatrix,
    ps,
)
    size(rails, 1) == PointModel.INPUT_RAILS ||
        throw(DimensionMismatch("GRU input rails"))
    candidates = size(rails, 2)
    input =
        reshape(ps.input_gain, model.state_dim, 1) .*
        rails[model.input_feature, :]
    state = zeros(Float32, model.state_dim, candidates)
    for _cycle in 1:model.cycles
        update = sigmoid.(
            reshape(ps.update_input, :, 1) .* input .+
            reshape(ps.update_recurrent, :, 1) .* state .+
            reshape(ps.update_bias, :, 1),
        )
        reset = sigmoid.(
            reshape(ps.reset_input, :, 1) .* input .+
            reshape(ps.reset_recurrent, :, 1) .* state .+
            reshape(ps.reset_bias, :, 1),
        )
        proposal = tanh.(
            reshape(ps.candidate_input, :, 1) .* input .+
            reshape(ps.candidate_recurrent, :, 1) .* reset .* state .+
            reshape(ps.candidate_bias, :, 1),
        )
        state = update .* state .+ (1.0f0 .- update) .* proposal
    end
    groups = div(model.state_dim, model.readout_dim)
    grouped = reshape(state, model.readout_dim, groups, candidates)
    pooled = dropdims(sum(grouped; dims=2); dims=2) ./ Float32(groups)
    features = vcat(
        PointModel.rms_normalize(pooled),
        PointModel.rms_normalize(abs.(pooled)),
    )
    hidden_pre = ps.head_weight * features .+ ps.head_bias
    hidden = tanh.(PointModel.rms_normalize(
        hidden_pre,
        PointModel.HIDDEN_NORM_SCALE,
    ))
    return ps.output_weight * hidden .+ ps.output_bias
end

function (model::DiagonalGRUBaseline)(rails, ps, st)
    raw = budget_gru_raw(model, rails, ps)
    return (;
        q=vec(raw[1:1, :]),
        death_logit=vec(raw[2:2, :]),
        quantiles=raw[3:18, :],
        geometry=raw[19:22, :],
        raw,
    ), st
end

function budget_gru_topology(model::DiagonalGRUBaseline)
    return (;
        family=:diagonal_gru_cpu_control,
        persistent_state_scalars=model.state_dim,
        cycles=model.cycles,
        dynamic_sparse=false,
        recurrent_matrix=:diagonal,
        credit=:direct_bptt,
    )
end

end # module BudgetMatchedGRU
