module ReducedHayCellKernel

export ReducedHayEventGraph,
    ReducedHayKernelCache,
    ReducedHaySoA,
    active_event_count,
    deliver_events!,
    reduced_hay_step!

"""
Structure-of-arrays state for the production CPU kernel.

The functional BPTT implementation is the reference trajectory. This type is
the allocation-free forward representation that the analytic-VJP/barrierless
executor will use.
"""
mutable struct ReducedHaySoA
    branch_voltage::Matrix{Float32}
    ampa::Matrix{Float32}
    nmda::Matrix{Float32}
    gaba::Matrix{Float32}
    plateau::Matrix{Float32}
    apical::Vector{Float32}
    soma::Vector{Float32}
    adaptation::Vector{Float32}
    spike::Vector{Float32}
end

function ReducedHaySoA(branches::Int, cells::Int)
    branches > 0 || throw(ArgumentError("branches must be positive"))
    cells > 0 || throw(ArgumentError("cells must be positive"))
    return ReducedHaySoA(
        zeros(Float32, branches, cells),
        zeros(Float32, branches, cells),
        zeros(Float32, branches, cells),
        zeros(Float32, branches, cells),
        zeros(Float32, branches, cells),
        zeros(Float32, cells),
        zeros(Float32, cells),
        zeros(Float32, cells),
        zeros(Float32, cells),
    )
end

"""
Materialized bounded parameters. Construct or refresh this cache outside the
hot candidate loop; every field is read-only during `reduced_hay_step!`.
"""
struct ReducedHayKernelCache
    branch_leak::Matrix{Float32}
    ampa_decay::Matrix{Float32}
    nmda_decay::Matrix{Float32}
    gaba_decay::Matrix{Float32}
    current_gain::Matrix{Float32}
    axial_gain::Matrix{Float32}
    nmda_slope::Matrix{Float32}
    nmda_half::Matrix{Float32}
    plateau_decay::Matrix{Float32}
    plateau_threshold::Matrix{Float32}
    plateau_slope::Matrix{Float32}
    plateau_gain::Matrix{Float32}
    plateau_feedback::Matrix{Float32}
    soma_coupling::Matrix{Float32}
    apical_leak::Vector{Float32}
    soma_leak::Vector{Float32}
    adaptation_decay::Vector{Float32}
    apical_gain::Vector{Float32}
    soma_threshold::Vector{Float32}
    adaptation_gain::Vector{Float32}
end

@inline _sigmoid(value::Float32) =
    inv(1.0f0 + exp(-value))

@inline _hard_sigmoid(value::Float32) =
    clamp(muladd(0.2f0, value, 0.5f0), 0.0f0, 1.0f0)

@inline function _assert_step_shapes(
    state::ReducedHaySoA,
    cache::ReducedHayKernelCache,
    excitatory::AbstractMatrix{Float32},
    inhibitory::AbstractMatrix{Float32},
    apical_drive::AbstractVector{Float32},
    active::AbstractVector{Bool},
)
    shape = size(state.branch_voltage)
    size(state.ampa) == shape || throw(DimensionMismatch("AMPA state"))
    size(state.nmda) == shape || throw(DimensionMismatch("NMDA state"))
    size(state.gaba) == shape || throw(DimensionMismatch("GABA state"))
    size(state.plateau) == shape ||
        throw(DimensionMismatch("plateau state"))
    size(excitatory) == shape ||
        throw(DimensionMismatch("excitatory inbox"))
    size(inhibitory) == shape ||
        throw(DimensionMismatch("inhibitory inbox"))
    size(cache.branch_leak) == shape ||
        throw(DimensionMismatch("kernel cache"))
    cells = shape[2]
    length(apical_drive) == cells ||
        throw(DimensionMismatch("apical drive"))
    length(active) == cells || throw(DimensionMismatch("active mask"))
    return nothing
end

"""
Allocation-free reduced-Hay state transition.

`active` gates only the external event bit. Continuous compartment state is
still advanced for every cell in the correctness kernel. Lazy decay and
inactive-cell skipping can be added after analytic-VJP equivalence is closed.
"""
function reduced_hay_step!(
    state::ReducedHaySoA,
    cache::ReducedHayKernelCache,
    excitatory::AbstractMatrix{Float32},
    inhibitory::AbstractMatrix{Float32},
    apical_drive::AbstractVector{Float32},
    active::AbstractVector{Bool},
)
    _assert_step_shapes(
        state,
        cache,
        excitatory,
        inhibitory,
        apical_drive,
        active,
    )
    branches, cells = size(state.branch_voltage)
    @inbounds for cell in 1:cells
        old_soma = state.soma[cell]
        basal = 0.0f0
        for branch in 1:branches
            old_voltage = state.branch_voltage[branch, cell]
            next_ampa = muladd(
                cache.ampa_decay[branch, cell],
                state.ampa[branch, cell],
                excitatory[branch, cell],
            )
            next_nmda = muladd(
                cache.nmda_decay[branch, cell],
                state.nmda[branch, cell],
                0.72f0 * excitatory[branch, cell],
            )
            next_gaba = muladd(
                cache.gaba_decay[branch, cell],
                state.gaba[branch, cell],
                inhibitory[branch, cell],
            )
            unblock = _sigmoid(
                cache.nmda_slope[branch, cell] *
                (old_voltage - cache.nmda_half[branch, cell]),
            )
            excitatory_current =
                (next_ampa + next_nmda * unblock) *
                (1.0f0 - old_voltage)
            inhibitory_current =
                next_gaba * (-1.0f0 - old_voltage)
            voltage = clamp(
                cache.branch_leak[branch, cell] * old_voltage +
                cache.current_gain[branch, cell] *
                (excitatory_current + inhibitory_current) +
                cache.axial_gain[branch, cell] *
                (old_soma - old_voltage) +
                cache.plateau_feedback[branch, cell] *
                state.plateau[branch, cell],
                -2.0f0,
                3.0f0,
            )
            coincidence = _hard_sigmoid(
                cache.plateau_slope[branch, cell] *
                (voltage - cache.plateau_threshold[branch, cell]),
            )
            plateau = clamp(
                cache.plateau_decay[branch, cell] *
                state.plateau[branch, cell] +
                cache.plateau_gain[branch, cell] *
                next_nmda * coincidence,
                0.0f0,
                4.0f0,
            )
            state.ampa[branch, cell] = next_ampa
            state.nmda[branch, cell] = next_nmda
            state.gaba[branch, cell] = next_gaba
            state.branch_voltage[branch, cell] = voltage
            state.plateau[branch, cell] = plateau
            basal = muladd(
                cache.soma_coupling[branch, cell],
                voltage + plateau,
                basal,
            )
        end
        next_apical = muladd(
            cache.apical_leak[cell],
            state.apical[cell],
            apical_drive[cell],
        )
        modulation =
            1.0f0 +
            cache.apical_gain[cell] * _hard_sigmoid(next_apical)
        soma_pre =
            cache.soma_leak[cell] * old_soma +
            basal * modulation -
            state.adaptation[cell]
        spike = soma_pre >= cache.soma_threshold[cell] ? 1.0f0 : 0.0f0
        state.apical[cell] = next_apical
        state.soma[cell] =
            soma_pre - spike * cache.soma_threshold[cell]
        state.adaptation[cell] = muladd(
            cache.adaptation_decay[cell],
            state.adaptation[cell],
            cache.adaptation_gain[cell] * spike,
        )
        state.spike[cell] = active[cell] ? spike : 0.0f0
    end
    return state
end

"""
Source-major fixed-fanout event graph. Disabled contacts stay in-place so
structure consolidation never reallocates adjacency storage.
"""
struct ReducedHayEventGraph
    destination::Matrix{Int32}
    branch::Matrix{UInt8}
    weight::Matrix{Float32}
    enabled::BitMatrix
end

function ReducedHayEventGraph(
    destination::AbstractMatrix{<:Integer},
    branch::AbstractMatrix{<:Integer},
    weight::AbstractMatrix{<:Real},
    enabled::AbstractMatrix{Bool},
)
    size(destination) == size(branch) == size(weight) == size(enabled) ||
        throw(DimensionMismatch("event graph arrays"))
    return ReducedHayEventGraph(
        Int32.(destination),
        UInt8.(branch),
        Float32.(weight),
        BitMatrix(enabled),
    )
end

active_event_count(spikes::AbstractVector{Float32}) =
    count(!iszero, spikes)

"""
Allocation-free sparse event delivery. Only fired sources scan their fixed
fanout; positive contacts enter the excitatory inbox and negative contacts
enter the inhibitory inbox.
"""
function deliver_events!(
    excitatory::AbstractMatrix{Float32},
    inhibitory::AbstractMatrix{Float32},
    spikes::AbstractVector{Float32},
    graph::ReducedHayEventGraph,
)
    fill!(excitatory, 0.0f0)
    fill!(inhibitory, 0.0f0)
    cells, fanout = size(graph.destination)
    length(spikes) == cells || throw(DimensionMismatch("spike vector"))
    branches = size(excitatory, 1)
    size(excitatory) == size(inhibitory) ||
        throw(DimensionMismatch("event inboxes"))
    size(excitatory, 2) == cells ||
        throw(DimensionMismatch("event inbox cells"))
    deliveries = 0
    @inbounds for source in 1:cells
        signal = spikes[source]
        iszero(signal) && continue
        for relation in 1:fanout
            graph.enabled[source, relation] || continue
            destination = Int(graph.destination[source, relation])
            branch = Int(graph.branch[source, relation])
            1 <= destination <= cells ||
                throw(BoundsError(1:cells, destination))
            1 <= branch <= branches ||
                throw(BoundsError(1:branches, branch))
            payload = graph.weight[source, relation] * signal
            if payload >= 0.0f0
                excitatory[branch, destination] += payload
            else
                inhibitory[branch, destination] -= payload
            end
            deliveries += 1
        end
    end
    return deliveries
end

end # module ReducedHayCellKernel
