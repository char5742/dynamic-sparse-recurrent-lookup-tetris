module DendriticCellKernel

export ActiveDendriticCellParameters,
    ActiveDendriticCellState,
    DendriticCellArena,
    DendriticEligibilityTrace,
    analog_readout!,
    compartment_count,
    dendritic_arena_step!,
    dendritic_cell_step!,
    hard_sigmoid,
    reset_arena!,
    reset_eligibility!,
    reset_state!,
    three_factor_update,
    update_eligibility!,
    utility_contribution

"""
Reduced active-dendrite cell used by the CPU implementation.

For `B` basal branches, one cell owns `2B + 3` persistent scalar states:

  * `B` branch voltages,
  * `B` slow plateau states,
  * one apical/context voltage,
  * one soma voltage,
  * one adaptation state.

The vectors are allocated once. `dendritic_cell_step!` mutates the state in
place and performs no allocation after compilation.
"""
struct ActiveDendriticCellParameters
    branches::Int
    branch_leak::Vector{Float32}
    plateau_decay::Vector{Float32}
    plateau_threshold::Vector{Float32}
    plateau_slope::Vector{Float32}
    plateau_gain::Vector{Float32}
    plateau_feedback::Vector{Float32}
    soma_coupling::Vector{Float32}
    apical_leak::Float32
    soma_leak::Float32
    adaptation_decay::Float32
    apical_gain::Float32
    soma_threshold::Float32
    adaptation_gain::Float32
end

function ActiveDendriticCellParameters(
    branches::Integer=4;
    branch_leak::Real=0.72f0,
    plateau_decay::Real=0.88f0,
    plateau_threshold::Real=0.45f0,
    plateau_slope::Real=4.0f0,
    plateau_gain::Real=0.80f0,
    plateau_feedback::Real=0.20f0,
    soma_coupling::Real=0.55f0,
    apical_leak::Real=0.80f0,
    soma_leak::Real=0.72f0,
    adaptation_decay::Real=0.85f0,
    apical_gain::Real=0.35f0,
    soma_threshold::Real=0.75f0,
    adaptation_gain::Real=0.15f0,
)
    branches >= 1 || throw(ArgumentError("branches must be positive"))
    vector(value) = fill(Float32(value), branches)
    return ActiveDendriticCellParameters(
        Int(branches),
        vector(branch_leak),
        vector(plateau_decay),
        vector(plateau_threshold),
        vector(plateau_slope),
        vector(plateau_gain),
        vector(plateau_feedback),
        vector(soma_coupling),
        Float32(apical_leak),
        Float32(soma_leak),
        Float32(adaptation_decay),
        Float32(apical_gain),
        Float32(soma_threshold),
        Float32(adaptation_gain),
    )
end

mutable struct ActiveDendriticCellState
    branch_voltage::Vector{Float32}
    plateau::Vector{Float32}
    apical::Float32
    soma::Float32
    adaptation::Float32
    spike::Float32
end

"""
Contiguous structure-of-arrays storage for a population of dendritic cells.

The cell index is the first matrix dimension, so one branch compartment across
the whole population is contiguous and SIMD-friendly. This is the CPU
execution representation; the scalar state type above is retained for
capability and finite-difference tests.
"""
struct DendriticCellArena
    branch_voltage::Matrix{Float32}
    plateau::Matrix{Float32}
    apical::Vector{Float32}
    soma::Vector{Float32}
    adaptation::Vector{Float32}
    spike::Vector{Float32}
    basal_scratch::Vector{Float32}
end

function DendriticCellArena(cells::Integer, branches::Integer=4)
    cells >= 1 || throw(ArgumentError("cells must be positive"))
    branches >= 1 || throw(ArgumentError("branches must be positive"))
    return DendriticCellArena(
        zeros(Float32, cells, branches),
        zeros(Float32, cells, branches),
        zeros(Float32, cells),
        zeros(Float32, cells),
        zeros(Float32, cells),
        zeros(Float32, cells),
        zeros(Float32, cells),
    )
end

"""
Factorized forward eligibility for one synapse.

An edge terminates on exactly one basal branch, so it does not need an
11-by-11 Jacobian. The locally causal path is represented by three scalars:

`edge -> branch voltage -> plateau -> soma`.
"""
mutable struct DendriticEligibilityTrace
    branch::Int
    branch_voltage::Float32
    plateau::Float32
    soma::Float32
end

function DendriticEligibilityTrace(branch::Integer)
    branch >= 1 || throw(ArgumentError("branch must be positive"))
    return DendriticEligibilityTrace(Int(branch), 0.0f0, 0.0f0, 0.0f0)
end

function ActiveDendriticCellState(branches::Integer=4)
    branches >= 1 || throw(ArgumentError("branches must be positive"))
    return ActiveDendriticCellState(
        zeros(Float32, branches),
        zeros(Float32, branches),
        0.0f0,
        0.0f0,
        0.0f0,
        0.0f0,
    )
end

@inline compartment_count(branches::Integer) = 2Int(branches) + 3

@inline function hard_sigmoid(value::Float32)
    return clamp(muladd(0.2f0, value, 0.5f0), 0.0f0, 1.0f0)
end

function reset_state!(state::ActiveDendriticCellState)
    fill!(state.branch_voltage, 0.0f0)
    fill!(state.plateau, 0.0f0)
    state.apical = 0.0f0
    state.soma = 0.0f0
    state.adaptation = 0.0f0
    state.spike = 0.0f0
    return state
end

function reset_arena!(arena::DendriticCellArena)
    fill!(arena.branch_voltage, 0.0f0)
    fill!(arena.plateau, 0.0f0)
    fill!(arena.apical, 0.0f0)
    fill!(arena.soma, 0.0f0)
    fill!(arena.adaptation, 0.0f0)
    fill!(arena.spike, 0.0f0)
    fill!(arena.basal_scratch, 0.0f0)
    return arena
end

function reset_eligibility!(trace::DendriticEligibilityTrace)
    trace.branch_voltage = 0.0f0
    trace.plateau = 0.0f0
    trace.soma = 0.0f0
    return trace
end

@inline function _hard_sigmoid_derivative(value::Float32)
    return (-2.5f0 < value < 2.5f0) ? 0.2f0 : 0.0f0
end

"""
Advance one edge's e-prop-style local eligibility in the forward direction.

Only locally available quantities are required: presynaptic activity, current
on the target branch, its voltage, apical modulation, the postsynaptic
surrogate sensitivity, and the cell's own parameters. No downstream graph or
head Jacobian is traversed.
"""
function update_eligibility!(
    trace::DendriticEligibilityTrace,
    presynaptic_activity::Float32,
    branch_current::Float32,
    branch_voltage::Float32,
    apical_modulation::Float32,
    post_surrogate::Float32,
    parameters::ActiveDendriticCellParameters,
)
    branch = trace.branch
    1 <= branch <= parameters.branches ||
        throw(BoundsError(1:parameters.branches, branch))

    epsilon_branch = muladd(
        parameters.branch_leak[branch],
        trace.branch_voltage,
        presynaptic_activity,
    )
    coincidence_argument =
        parameters.plateau_slope[branch] *
        (branch_voltage - parameters.plateau_threshold[branch])
    coincidence = hard_sigmoid(coincidence_argument)
    coincidence_derivative =
        _hard_sigmoid_derivative(coincidence_argument) *
        parameters.plateau_slope[branch] *
        epsilon_branch
    recruited_derivative = if branch_current > 0.0f0
        presynaptic_activity * coincidence +
        branch_current * coincidence_derivative
    else
        0.0f0
    end
    epsilon_plateau = muladd(
        parameters.plateau_decay[branch],
        trace.plateau,
        parameters.plateau_gain[branch] * recruited_derivative,
    )
    soma_before_reset = muladd(
        parameters.soma_leak,
        trace.soma,
        parameters.soma_coupling[branch] *
        (epsilon_branch + epsilon_plateau) *
        apical_modulation,
    )
    reset_sensitivity =
        1.0f0 - parameters.soma_threshold * post_surrogate

    trace.branch_voltage = epsilon_branch
    trace.plateau = epsilon_plateau
    trace.soma = reset_sensitivity * soma_before_reset
    return trace.soma
end

@inline three_factor_update(
    block_learning_signal::Float32,
    trace::DendriticEligibilityTrace,
) = block_learning_signal * trace.soma

@inline utility_contribution(
    block_learning_signal::Float32,
    trace::DendriticEligibilityTrace,
) = abs(block_learning_signal * trace.soma)

"""
Advance one high-dimensional neuron by one cycle.

`excitatory` and `inhibitory` are branch-local currents. A plateau is recruited
only when positive net current coincides with a depolarized branch. Branch and
plateau states are never reset by a soma spike; only the soma loses one
threshold unit. This preserves the cell's analog state while exposing a hard
event for graph communication.
"""
function dendritic_cell_step!(
    state::ActiveDendriticCellState,
    excitatory::AbstractVector{Float32},
    inhibitory::AbstractVector{Float32},
    apical_drive::Float32,
    parameters::ActiveDendriticCellParameters,
)
    branches = parameters.branches
    length(state.branch_voltage) == branches ||
        throw(DimensionMismatch("branch voltage count"))
    length(state.plateau) == branches ||
        throw(DimensionMismatch("plateau count"))
    length(excitatory) == branches ||
        throw(DimensionMismatch("excitatory branch count"))
    length(inhibitory) == branches ||
        throw(DimensionMismatch("inhibitory branch count"))

    basal_sum = 0.0f0
    @inbounds for branch in 1:branches
        old_plateau = state.plateau[branch]
        net_current = excitatory[branch] - inhibitory[branch]
        branch_voltage = muladd(
            parameters.branch_leak[branch],
            state.branch_voltage[branch],
            net_current + parameters.plateau_feedback[branch] * old_plateau,
        )
        coincidence = hard_sigmoid(
            parameters.plateau_slope[branch] *
            (branch_voltage - parameters.plateau_threshold[branch]),
        )
        recruited = max(net_current, 0.0f0) * coincidence
        plateau = muladd(
            parameters.plateau_decay[branch],
            old_plateau,
            parameters.plateau_gain[branch] * recruited,
        )
        state.branch_voltage[branch] = branch_voltage
        state.plateau[branch] = plateau
        # Passive branch currents are summed linearly. The plateau contributes
        # a branch-local nonlinear term, which cannot be reproduced by merely
        # summing all input at the soma.
        basal_sum = muladd(
            parameters.soma_coupling[branch],
            branch_voltage + plateau,
            basal_sum,
        )
    end

    apical = muladd(parameters.apical_leak, state.apical, apical_drive)
    apical_modulation =
        1.0f0 + parameters.apical_gain * hard_sigmoid(apical)
    soma_pre = muladd(
        parameters.soma_leak,
        state.soma,
        basal_sum * apical_modulation - state.adaptation,
    )
    spike = soma_pre >= parameters.soma_threshold ? 1.0f0 : 0.0f0
    state.apical = apical
    state.soma = soma_pre - spike * parameters.soma_threshold
    state.adaptation = muladd(
        parameters.adaptation_decay,
        state.adaptation,
        parameters.adaptation_gain * spike,
    )
    state.spike = spike
    return spike
end

"""
Advance a contiguous population arena by one cycle without allocation.

`excitatory` and `inhibitory` have shape `cells × branches`; `apical_drive`
contains one scalar per cell.
"""
function dendritic_arena_step!(
    arena::DendriticCellArena,
    excitatory::AbstractMatrix{Float32},
    inhibitory::AbstractMatrix{Float32},
    apical_drive::AbstractVector{Float32},
    parameters::ActiveDendriticCellParameters,
)
    cells, branches = size(arena.branch_voltage)
    branches == parameters.branches ||
        throw(DimensionMismatch("arena branch count"))
    size(arena.plateau) == (cells, branches) ||
        throw(DimensionMismatch("arena plateau shape"))
    size(excitatory) == (cells, branches) ||
        throw(DimensionMismatch("excitatory arena shape"))
    size(inhibitory) == (cells, branches) ||
        throw(DimensionMismatch("inhibitory arena shape"))
    length(apical_drive) == cells ||
        throw(DimensionMismatch("apical drive count"))
    length(arena.apical) == cells ||
        throw(DimensionMismatch("arena cell count"))
    length(arena.basal_scratch) == cells ||
        throw(DimensionMismatch("arena scratch count"))

    fill!(arena.basal_scratch, 0.0f0)
    @inbounds for branch in 1:branches
        branch_leak = parameters.branch_leak[branch]
        plateau_feedback = parameters.plateau_feedback[branch]
        plateau_slope = parameters.plateau_slope[branch]
        plateau_threshold = parameters.plateau_threshold[branch]
        plateau_decay = parameters.plateau_decay[branch]
        plateau_gain = parameters.plateau_gain[branch]
        soma_coupling = parameters.soma_coupling[branch]
        @simd for cell in 1:cells
            old_plateau = arena.plateau[cell, branch]
            net_current =
                excitatory[cell, branch] - inhibitory[cell, branch]
            branch_voltage = muladd(
                branch_leak,
                arena.branch_voltage[cell, branch],
                net_current +
                plateau_feedback * old_plateau,
            )
            coincidence = hard_sigmoid(
                plateau_slope *
                (branch_voltage - plateau_threshold),
            )
            recruited = max(net_current, 0.0f0) * coincidence
            plateau = muladd(
                plateau_decay,
                old_plateau,
                plateau_gain * recruited,
            )
            arena.branch_voltage[cell, branch] = branch_voltage
            arena.plateau[cell, branch] = plateau
            arena.basal_scratch[cell] = muladd(
                soma_coupling,
                branch_voltage + plateau,
                arena.basal_scratch[cell],
            )
        end
    end

    @inbounds @simd for cell in 1:cells
        apical = muladd(
            parameters.apical_leak,
            arena.apical[cell],
            apical_drive[cell],
        )
        modulation =
            1.0f0 + parameters.apical_gain * hard_sigmoid(apical)
        soma_pre = muladd(
            parameters.soma_leak,
            arena.soma[cell],
            arena.basal_scratch[cell] * modulation -
            arena.adaptation[cell],
        )
        spike =
            soma_pre >= parameters.soma_threshold ? 1.0f0 : 0.0f0
        arena.apical[cell] = apical
        arena.soma[cell] =
            soma_pre - spike * parameters.soma_threshold
        arena.adaptation[cell] = muladd(
            parameters.adaptation_decay,
            arena.adaptation[cell],
            parameters.adaptation_gain * spike,
        )
        arena.spike[cell] = spike
    end
    return nothing
end

"""
Write the analog information plane:

`[tanh(soma), tanh(apical), tanh(branch_voltage[1:B])...]`.
"""
function analog_readout!(
    destination::AbstractVector{Float32},
    state::ActiveDendriticCellState,
)
    branches = length(state.branch_voltage)
    length(destination) == branches + 2 ||
        throw(DimensionMismatch("analog readout length"))
    destination[1] = tanh(state.soma)
    destination[2] = tanh(state.apical)
    @inbounds for branch in 1:branches
        destination[branch + 2] = tanh(state.branch_voltage[branch])
    end
    return destination
end

end # module
