module PaperWorkspaceModel

using Lux
using Random

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(_PARENT_MODULE, :SerialWorkspaceSNN)
    Base.include(
        _PARENT_MODULE,
        joinpath(@__DIR__, "..", "serial_workspace_snn", "SerialWorkspaceSNN.jl"),
    )
end
using ..SerialWorkspaceSNN: INPUT_RAILS

export PaperModel,
    build_paper_model,
    paper_parameter_count,
    EXCITATORY,
    INHIBITORY

const TRACE_DIM = 48
const EXCITATORY = UInt8(1)
const INHIBITORY = UInt8(2)

"""
Paper-mechanism outer graph.

Each block owns exactly one canonical PaperHayCell. Its public 48-dimensional
state is a fixed bank of causal filters of the soma spike train. No
compartment voltage, calcium value, conductance or channel gate is readable by
the global workspace or supervised head.

`rail_kind` makes Dale type a property of the presynaptic sensory axon, never
of an individual contact. Recurrent type is likewise a property of the source
block. A normalized excitatory strength emits paired AMPA/NMDA conductances;
an inhibitory strength emits GABA_A. Intrinsic morphology/channels are not in
the trainable parameter tree.
"""
struct PaperModel <: Lux.AbstractLuxLayer
    blocks::Int
    cells_per_block::Int
    node_dim::Int
    cycles::Int
    substeps_per_cycle::Int
    workspace_k::Int
    hidden::Int
    fanout::Int
    sensory_contacts::Int
    recurrent_contacts::Int
    workspace_contacts::Int
    route_temperature::Float32
    trace_decay::Vector{Float32}
    rail_kind::Vector{UInt8}
    input_rail::Matrix{Int32}
    recurrent_source::Matrix{Int16}
end

function _trace_decay_bank(dim::Int)
    result = Vector{Float32}(undef, dim)
    @inbounds for index in 1:dim
        fraction = Float32(index - 1) / Float32(max(dim - 1, 1))
        tau_ms = exp(muladd(fraction, log(300.0f0), log(1.0f0)))
        result[index] = exp(-1.0f0 / tau_ms)
    end
    return result
end

function PaperModel(;
    blocks::Int=96,
    cycles::Int=4,
    substeps_per_cycle::Int=8,
    workspace_k::Int=8,
    hidden::Int=192,
    sensory_contacts::Int=24,
    recurrent_contacts::Int=16,
    route_temperature::Real=0.35,
    topology_seed::Integer=0x5041504552544f50,
)
    blocks >= 2 || throw(ArgumentError("blocks must be at least two"))
    cycles >= 1 || throw(ArgumentError("cycles must be positive"))
    substeps_per_cycle >= 1 ||
        throw(ArgumentError("substeps_per_cycle must be positive"))
    1 <= workspace_k <= blocks ||
        throw(ArgumentError("workspace_k must be in 1:blocks"))
    sensory_contacts >= 2 ||
        throw(ArgumentError("sensory_contacts must be at least two"))
    recurrent_contacts >= 2 ||
        throw(ArgumentError("recurrent_contacts must be at least two"))

    rng = Xoshiro(UInt64(topology_seed))
    rail_kind = Vector{UInt8}(undef, INPUT_RAILS)
    @inbounds for rail in 1:INPUT_RAILS
        # Fixed 80/20 E/I population split; all contacts of one rail retain it.
        rail_kind[rail] =
            mod(rail + 17, 5) == 0 ? INHIBITORY : EXCITATORY
    end
    input_rail = Matrix{Int32}(undef, sensory_contacts, blocks)
    recurrent_source = Matrix{Int16}(undef, recurrent_contacts, blocks)
    @inbounds for block in 1:blocks
        for contact in 1:sensory_contacts
            input_rail[contact, block] = Int32(rand(rng, 1:INPUT_RAILS))
        end
        for contact in 1:recurrent_contacts
            recurrent_source[contact, block] = Int16(mod1(
                block +
                contact +
                7contact * contact +
                11block * contact,
                blocks,
            ))
        end
    end
    return PaperModel(
        blocks,
        1,
        TRACE_DIM,
        cycles,
        substeps_per_cycle,
        workspace_k,
        hidden,
        recurrent_contacts,
        sensory_contacts,
        recurrent_contacts,
        workspace_k,
        Float32(route_temperature),
        _trace_decay_bank(TRACE_DIM),
        rail_kind,
        input_rail,
        recurrent_source,
    )
end

function build_paper_model(preset::Symbol=:paper_scaled_v1)
    preset === :tiny && return PaperModel(
        blocks=8,
        cycles=2,
        substeps_per_cycle=4,
        workspace_k=2,
        hidden=32,
        sensory_contacts=8,
        recurrent_contacts=4,
    )
    preset === :smoke && return PaperModel(
        blocks=24,
        cycles=3,
        substeps_per_cycle=6,
        workspace_k=4,
        hidden=96,
        sensory_contacts=16,
        recurrent_contacts=8,
    )
    preset in (:paper_scaled_v1, :paper_mechanism_v1) &&
        return PaperModel()
    throw(ArgumentError(
        "unknown paper preset $preset; use :tiny, :smoke or :paper_scaled_v1",
    ))
end

function Lux.initialparameters(rng::AbstractRNG, model::PaperModel)
    input_shape = (model.sensory_contacts, model.blocks)
    recurrent_shape = (model.recurrent_contacts, model.blocks)
    return (;
        input_conductance=
            0.08f0 .+ 0.04f0 .* rand(rng, Float32, input_shape),
        recurrent_conductance=
            0.04f0 .+ 0.03f0 .* rand(rng, Float32, recurrent_shape),
        workspace_conductance=
            0.02f0 .+
            0.02f0 .* rand(
                rng,
                Float32,
                model.workspace_contacts,
                model.blocks,
            ),
        query_weight=
            0.08f0 .* randn(
                rng,
                Float32,
                model.node_dim,
                INPUT_RAILS,
            ) ./ sqrt(Float32(INPUT_RAILS)),
        workspace_key=
            0.16f0 .* randn(
                rng,
                Float32,
                model.node_dim,
                model.blocks,
            ) ./ sqrt(Float32(model.node_dim)),
        workspace_decay_logit=Float32[1.2f0],
        head_weight=
            0.10f0 .* randn(
                rng,
                Float32,
                model.hidden,
                2model.node_dim,
            ) ./ sqrt(Float32(2model.node_dim)),
        head_bias=zeros(Float32, model.hidden),
        output_weight=
            0.08f0 .* randn(
                rng,
                Float32,
                22,
                model.hidden,
            ) ./ sqrt(Float32(model.hidden)),
        output_bias=zeros(Float32, 22),
    )
end

Lux.initialstates(::AbstractRNG, ::PaperModel) = (;)

function paper_parameter_count(model::PaperModel)
    parameters, _ = Lux.setup(Xoshiro(0), model)
    return sum(length, values(parameters))
end

end
