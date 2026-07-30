module PaperModelCanonical

using Lux
using Random

if !isdefined(@__MODULE__, :PaperWorkspaceModel)
    include(joinpath(@__DIR__, "PaperWorkspaceModel.jl"))
end
const BaseModel = PaperWorkspaceModel

export PaperModel,
    build_paper_model,
    paper_parameter_count,
    EXCITATORY,
    INHIBITORY

const EXCITATORY = BaseModel.EXCITATORY
const INHIBITORY = BaseModel.INHIBITORY

"""
Canonical wrapper adding a global Dale type for every recurrent source block.

`base.rail_kind` is already global per sensory rail. `block_kind` performs the
same role for recurrent/workspace axons, so a source block can never be
excitatory at one contact and inhibitory at another.
"""
struct PaperModel <: Lux.AbstractLuxLayer
    base::BaseModel.PaperModel
    block_kind::Vector{UInt8}
end

function PaperModel(base::BaseModel.PaperModel)
    kind = Vector{UInt8}(undef, base.blocks)
    @inbounds for block in 1:base.blocks
        kind[block] =
            mod(block + 7, 5) == 0 ? INHIBITORY : EXCITATORY
    end
    return PaperModel(base, kind)
end

function Base.getproperty(model::PaperModel, name::Symbol)
    name === :base && return getfield(model, :base)
    name === :block_kind && return getfield(model, :block_kind)
    return getproperty(getfield(model, :base), name)
end

function Base.propertynames(model::PaperModel, private::Bool=false)
    base_names = propertynames(getfield(model, :base), private)
    return (:base, :block_kind, base_names...)
end

function build_paper_model(preset::Symbol=:paper_scaled_v1)
    return PaperModel(BaseModel.build_paper_model(preset))
end

function Lux.initialparameters(rng::AbstractRNG, model::PaperModel)
    return Lux.initialparameters(rng, getfield(model, :base))
end

Lux.initialstates(rng::AbstractRNG, model::PaperModel) =
    Lux.initialstates(rng, getfield(model, :base))

paper_parameter_count(model::PaperModel) =
    BaseModel.paper_parameter_count(getfield(model, :base))

end
