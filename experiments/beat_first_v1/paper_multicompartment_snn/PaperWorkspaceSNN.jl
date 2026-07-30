module PaperWorkspaceSNN

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

export PaperWorkspaceModel,
    build_paper_model,
    paper_parameter_count,
    input_contact_kind,
    recurrent_contact_kind

const TRACE_DIM = 48
const DEFAULT_BLOCKS = 96
const EXCITATORY = UInt8(1)
const INHIBITORY = UInt8(2)

"""
Outer Tetris graph for the paper-mechanism multicompartment cell.

Every cognitive block contains one canonical `PaperHayCell`. The block's
48-dimensional public state is a fixed bank of causal filters of the soma
spike train. Compartment voltages, channel gates and calcium never enter the
workspace or supervised head directly.

The static contact tables obey Dale's law. An excitatory contact always emits
both AMPA and NMDA events; an inhibitory contact emits GABA_A. Strength is one
nonnegative normalized conductance per anatomical contact and is clipped by
the trainer before it is converted to the receptor-specific paper maximum.
"""
struct PaperWorkspaceModel <: Lux.AbstractLuxLayer
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
    route_temperature::Float32
    trace_decay::Vector{Float32}
    input_rail::Matrix{Int32}
    input_kind::Matrix{UInt8}
    recurrent_source::Matrix{Int16}
    recurrent_kind::Matrix{UInt8}
end

@inline input_contact_kind(model::PaperWorkspaceModel, contact, block) =
    model.input_kind[contact, block]

@inline recurrent_contact_kind(model::PaperWorkspaceModel, contact, block) =
    model.recurrent_kind[contact, block]

function _trace_decay_bank(dim::Int)
    decay = Vector{Float32}(undef, dim)
    # 1--300 ms, as a deterministic logarithmic memory basis. These constants
    # are fixed and are not an analog-voltage bypass.
    log_min = log(1.0f0)
    log_max = log(300.0f0)
    @inbounds for index in 1:dim
        fraction = Float32(index - 1) / Float32(max(dim - 1, 1))
        tau_ms = exp(muladd(fraction, log_max - log_min, log_min))
        decay[index] = exp(-1.0f0 / tau_ms)
    end
    return decay
end

function PaperWorkspaceModel(;
    blocks::Int=DEFAULT_BLOCKS,
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
    input_rail = Matrix{Int32}(undef, sensory_contacts, blocks)
    input_kind = Matrix{UInt8}(undef, sensory_contacts, blocks)
    recurrent_source = Matrix{Int16}(undef, recurrent_contacts, blocks)
    recurrent_kind = Matrix{UInt8}(undef, recurrent_contacts, blocks)

    @inbounds for block in 1:blocks
        # Every block sees both excitation and inhibition. The source axon type
        # is immutable, so relocating a contact cannot violate Dale's law.
        for contact in 1:sensory_contacts
            input_rail[contact, block] =
                Int32(rand(rng, 1:INPUT_RAILS))
            input_kind[contact, block] =
                mod(contact + 3block, 5) == 0 ? INHIBITORY : EXCITATORY
        end
        for contact in 1:recurrent_contacts
            source = mod1(
                block +
                contact +
                7contact * contact +
                11block * contact,
                blocks,
            )
            recurrent_source[contact, block] = Int16(source)
            recurrent_kind[contact, block] =
                mod(source, 5) == 0 ? INHIBITORY : EXCITATORY
        end
    end

    return PaperWorkspaceModel(
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
        Float32(route_temperature),
        _trace_decay_bank(TRACE_DIM),
        input_rail,
        input_kind,
        recurrent_source,
        recurrent_kind,
    )
end

function build_paper_model(preset::Symbol=:paper_scaled_v1)
    preset === :tiny && return PaperWorkspaceModel(
        blocks=8,
        cycles=2,
        substeps_per_cycle=4,
        workspace_k=2,
        hidden=32,
        sensory_contacts=8,
        recurrent_contacts=4,
    )
    preset === :smoke && return PaperWorkspaceModel(
        blocks=24,
        cycles=3,
        substeps_per_cycle=6,
        workspace_k=4,
        hidden=96,
        sensory_contacts=16,
        recurrent_contacts=8,
    )
    preset === :paper_scaled_v1 && return PaperWorkspaceModel()
    throw(ArgumentError(
        "unknown paper preset $preset; use :tiny, :smoke or :paper_scaled_v1",
    ))
end

function Lux.initialparameters(
    rng::AbstractRNG,
    model::PaperWorkspaceModel,
)
    input_shape = (model.sensory_contacts, model.blocks)
    recurrent_shape = (model.recurrent_contacts, model.blocks)
    return (;
        # Normalized anatomical conductance in [0,1]. Forward multiplies by
        # AMPA/NMDA/GABA_A maxima from PaperHayCell.
        input_conductance=
            0.08f0 .+ 0.04f0 .* rand(rng, Float32, input_shape),
        recurrent_conductance=
            0.04f0 .+ 0.03f0 .* rand(rng, Float32, recurrent_shape),
        apical_feedback_conductance=
            0.02f0 .+
            0.02f0 .* rand(rng, Float32, model.node_dim, model.blocks),
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

Lux.initialstates(::AbstractRNG, ::PaperWorkspaceModel) = (;)

function paper_parameter_count(model::PaperWorkspaceModel)
    parameters, _ = Lux.setup(Xoshiro(0), model)
    total = 0
    for array in values(parameters)
        total += length(array)
    end
    return total
end

end
