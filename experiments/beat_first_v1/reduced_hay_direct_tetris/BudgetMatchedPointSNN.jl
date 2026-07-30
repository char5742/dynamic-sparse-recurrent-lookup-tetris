module BudgetMatchedPointSNN

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
const Point = getfield(_PARENT_MODULE, :SerialWorkspaceSNN)

export build_budget_point_snn,
    budget_point_raw,
    budget_point_topology

"""
Point-SNN control with 372 persistent membrane scalars, matching the tiny
Reduced Hay arm's 16 * 23 continuous-state budget within 1.1%.

The public block width is 12, identical to the Reduced Hay arm. Extra point
states are represented as more blocks rather than a wider query/head, so the
comparison does not charge Point-SNN for a 46-dimensional public interface.
"""
function build_budget_point_snn()
    return Point.SerialWorkspaceModel(
        blocks=31,
        node_dim=12,
        fanout=2,
        cycles=3,
        workspace_k=2,
        hidden=32,
    )
end

function budget_point_raw(model, rails::AbstractMatrix, ps)
    dynamics = Point._dynamics(model, rails, ps)
    features = Point.head_features(dynamics)
    hidden_pre = ps.head_weight * features .+ ps.head_bias
    hidden = tanh.(Point.rms_normalize(
        hidden_pre,
        Point.HIDDEN_NORM_SCALE,
    ))
    return ps.output_weight * hidden .+ ps.output_bias
end

function budget_point_topology(model)
    return (;
        family=:budget_matched_point_snn,
        persistent_state_scalars=model.blocks * model.node_dim,
        cycles=model.cycles,
        candidate_synapses=
            model.blocks * model.node_dim * model.fanout,
        dynamic_sparse=true,
        credit=:analytic_vjp,
    )
end

end # module BudgetMatchedPointSNN
