module PaperBiophysicalCell

export APICAL_TRUNK,
    APICAL_TUFT,
    BASAL_DENDRITE,
    SOMA,
    PaperBiophysicalParameters,
    PaperBiophysicalState,
    PaperCompartmentTree,
    PaperSynapticDrive,
    compartment_count,
    layer5_reduced_tree,
    nmda_magnesium_block,
    paper_cell_step!,
    receptor_conductance,
    reset_drive!,
    reset_state!

# Region identifiers are bytes so that the hot compartment loop can dispatch
# without Symbols or heap objects.
const SOMA = UInt8(0x01)
const BASAL_DENDRITE = UInt8(0x02)
const APICAL_TRUNK = UInt8(0x03)
const APICAL_TUFT = UInt8(0x04)

"""
Flattened, explicit basal/apical compartment tree.

Compartment 1 is the soma. Every other compartment stores its parent and the
bidirectional axial conductance of that cable edge. The default reduced L5
tree has four two-compartment basal branches, a three-compartment apical
trunk, and four two-compartment tuft branches. This is intentionally a
reduced cable tree, not a claim to reproduce every segment of the Hay L5PC.
"""
struct PaperCompartmentTree
    parent::Vector{Int32}
    region::Vector{UInt8}
    axial_coupling::Vector{Float32}
    basal_terminals::Vector{Int32}
    apical_trunk::Vector{Int32}
    apical_terminals::Vector{Int32}
    soma::Int32
end

@inline compartment_count(tree::PaperCompartmentTree) = length(tree.parent)

function layer5_reduced_tree(;
    basal_branches::Integer=4,
    basal_depth::Integer=2,
    apical_trunk_depth::Integer=3,
    tuft_branches::Integer=4,
    tuft_depth::Integer=2,
    basal_coupling::Real=0.42f0,
    apical_coupling::Real=0.32f0,
    tuft_coupling::Real=0.24f0,
)
    basal_branches >= 1 ||
        throw(ArgumentError("basal_branches must be positive"))
    basal_depth >= 1 ||
        throw(ArgumentError("basal_depth must be positive"))
    apical_trunk_depth >= 1 ||
        throw(ArgumentError("apical_trunk_depth must be positive"))
    tuft_branches >= 1 ||
        throw(ArgumentError("tuft_branches must be positive"))
    tuft_depth >= 1 ||
        throw(ArgumentError("tuft_depth must be positive"))

    parent = Int32[0]
    region = UInt8[SOMA]
    axial_coupling = Float32[0.0f0]
    basal_terminals = Int32[]
    apical_trunk = Int32[]
    apical_terminals = Int32[]

    for _ in 1:Int(basal_branches)
        proximal = Int32(1)
        for depth in 1:Int(basal_depth)
            push!(parent, proximal)
            push!(region, BASAL_DENDRITE)
            # Distal segments are thinner and therefore slightly more weakly
            # coupled in this reduced cable representation.
            push!(
                axial_coupling,
                Float32(basal_coupling) / (1.0f0 + 0.18f0 * (depth - 1)),
            )
            proximal = Int32(length(parent))
        end
        push!(basal_terminals, proximal)
    end

    proximal = Int32(1)
    for depth in 1:Int(apical_trunk_depth)
        push!(parent, proximal)
        push!(region, APICAL_TRUNK)
        push!(
            axial_coupling,
            Float32(apical_coupling) / (1.0f0 + 0.12f0 * (depth - 1)),
        )
        proximal = Int32(length(parent))
        push!(apical_trunk, proximal)
    end
    tuft_root = proximal

    for _ in 1:Int(tuft_branches)
        proximal = tuft_root
        for depth in 1:Int(tuft_depth)
            push!(parent, proximal)
            push!(region, APICAL_TUFT)
            push!(
                axial_coupling,
                Float32(tuft_coupling) / (1.0f0 + 0.20f0 * (depth - 1)),
            )
            proximal = Int32(length(parent))
        end
        push!(apical_terminals, proximal)
    end

    return PaperCompartmentTree(
        parent,
        region,
        axial_coupling,
        basal_terminals,
        apical_trunk,
        apical_terminals,
        Int32(1),
    )
end

@inline function _double_exponential_scale(
    tau_rise_ms::Float32,
    tau_decay_ms::Float32,
)
    tau_decay_ms > tau_rise_ms > 0.0f0 ||
        throw(ArgumentError("double-exponential taus require decay > rise > 0"))
    peak_time =
        (tau_rise_ms * tau_decay_ms) /
        (tau_decay_ms - tau_rise_ms) *
        log(tau_decay_ms / tau_rise_ms)
    peak =
        exp(-peak_time / tau_decay_ms) -
        exp(-peak_time / tau_rise_ms)
    return inv(peak)
end

"""
Biophysical parameters for the reduced compartment tree.

Receptor states use normalized double exponentials. Active conductances are
region-dependent: fast Na and delayed-rectifier K support dendritic/somatic
spikes, high-threshold Ca supports local dendritic Ca spikes, and KCa
terminates the plateau. Setting `active_channels=false` retains the identical
cable/receptor tree while zeroing Na/Ca/K/KCa conductances.
"""
struct PaperBiophysicalParameters
    dt_ms::Float32
    resting_voltage_mv::Float32
    capacitance::Vector{Float32}
    leak_conductance::Vector{Float32}
    axial_conductance::Vector{Float32}
    ampa_rise_decay::Float32
    ampa_decay_decay::Float32
    ampa_scale::Float32
    nmda_rise_decay::Float32
    nmda_decay_decay::Float32
    nmda_scale::Float32
    gaba_rise_decay::Float32
    gaba_decay_decay::Float32
    gaba_scale::Float32
    ampa_reversal_mv::Float32
    nmda_reversal_mv::Float32
    gaba_reversal_mv::Float32
    extracellular_magnesium_mm::Float32
    gbar_na::Vector{Float32}
    gbar_ca::Vector{Float32}
    gbar_k::Vector{Float32}
    gbar_kca::Vector{Float32}
    sodium_reversal_mv::Float32
    calcium_reversal_mv::Float32
    potassium_reversal_mv::Float32
    calcium_decay::Float32
    calcium_influx_scale::Float32
    kca_half_activation::Float32
    soma_spike_threshold_mv::Float32
    local_ca_threshold_mv::Float32
end

function PaperBiophysicalParameters(
    tree::PaperCompartmentTree;
    dt_ms::Real=0.05f0,
    resting_voltage_mv::Real=-65.0f0,
    capacitance::Real=1.0f0,
    leak_conductance::Real=0.08f0,
    axial_scale::Real=1.0f0,
    ampa_tau_rise_ms::Real=0.20f0,
    ampa_tau_decay_ms::Real=2.0f0,
    nmda_tau_rise_ms::Real=2.0f0,
    nmda_tau_decay_ms::Real=60.0f0,
    gaba_tau_rise_ms::Real=0.50f0,
    gaba_tau_decay_ms::Real=8.0f0,
    ampa_reversal_mv::Real=0.0f0,
    nmda_reversal_mv::Real=0.0f0,
    gaba_reversal_mv::Real=-75.0f0,
    extracellular_magnesium_mm::Real=1.0f0,
    active_channels::Bool=true,
    sodium_scale::Real=1.0f0,
    calcium_scale::Real=1.0f0,
    potassium_scale::Real=1.0f0,
    kca_scale::Real=1.0f0,
    sodium_reversal_mv::Real=55.0f0,
    calcium_reversal_mv::Real=120.0f0,
    potassium_reversal_mv::Real=-90.0f0,
    calcium_tau_ms::Real=45.0f0,
    calcium_influx_scale::Real=0.0035f0,
    kca_half_activation::Real=0.20f0,
    soma_spike_threshold_mv::Real=-20.0f0,
    local_ca_threshold_mv::Real=-25.0f0,
)
    dt = Float32(dt_ms)
    dt > 0.0f0 || throw(ArgumentError("dt_ms must be positive"))
    count = compartment_count(tree)
    cap = fill(Float32(capacitance), count)
    leak = fill(Float32(leak_conductance), count)
    axial = Vector{Float32}(undef, count)
    @inbounds for compartment in 1:count
        axial[compartment] =
            Float32(axial_scale) * tree.axial_coupling[compartment]
    end

    gbar_na = zeros(Float32, count)
    gbar_ca = zeros(Float32, count)
    gbar_k = zeros(Float32, count)
    gbar_kca = zeros(Float32, count)
    if active_channels
        @inbounds for compartment in 1:count
            region = tree.region[compartment]
            if region == SOMA
                gbar_na[compartment] = 24.0f0 * Float32(sodium_scale)
                gbar_ca[compartment] = 0.25f0 * Float32(calcium_scale)
                gbar_k[compartment] = 7.0f0 * Float32(potassium_scale)
                gbar_kca[compartment] = 0.6f0 * Float32(kca_scale)
            elseif region == BASAL_DENDRITE
                gbar_na[compartment] = 3.0f0 * Float32(sodium_scale)
                gbar_ca[compartment] = 1.5f0 * Float32(calcium_scale)
                gbar_k[compartment] = 2.5f0 * Float32(potassium_scale)
                gbar_kca[compartment] = 2.0f0 * Float32(kca_scale)
            elseif region == APICAL_TRUNK
                gbar_na[compartment] = 1.5f0 * Float32(sodium_scale)
                gbar_ca[compartment] = 3.5f0 * Float32(calcium_scale)
                gbar_k[compartment] = 2.0f0 * Float32(potassium_scale)
                gbar_kca[compartment] = 3.0f0 * Float32(kca_scale)
            else
                gbar_na[compartment] = 0.8f0 * Float32(sodium_scale)
                gbar_ca[compartment] = 5.0f0 * Float32(calcium_scale)
                gbar_k[compartment] = 1.5f0 * Float32(potassium_scale)
                gbar_kca[compartment] = 3.5f0 * Float32(kca_scale)
            end
        end
    end

    ampa_rise = Float32(ampa_tau_rise_ms)
    ampa_decay = Float32(ampa_tau_decay_ms)
    nmda_rise = Float32(nmda_tau_rise_ms)
    nmda_decay = Float32(nmda_tau_decay_ms)
    gaba_rise = Float32(gaba_tau_rise_ms)
    gaba_decay = Float32(gaba_tau_decay_ms)
    return PaperBiophysicalParameters(
        dt,
        Float32(resting_voltage_mv),
        cap,
        leak,
        axial,
        exp(-dt / ampa_rise),
        exp(-dt / ampa_decay),
        _double_exponential_scale(ampa_rise, ampa_decay),
        exp(-dt / nmda_rise),
        exp(-dt / nmda_decay),
        _double_exponential_scale(nmda_rise, nmda_decay),
        exp(-dt / gaba_rise),
        exp(-dt / gaba_decay),
        _double_exponential_scale(gaba_rise, gaba_decay),
        Float32(ampa_reversal_mv),
        Float32(nmda_reversal_mv),
        Float32(gaba_reversal_mv),
        Float32(extracellular_magnesium_mm),
        gbar_na,
        gbar_ca,
        gbar_k,
        gbar_kca,
        Float32(sodium_reversal_mv),
        Float32(calcium_reversal_mv),
        Float32(potassium_reversal_mv),
        exp(-dt / Float32(calcium_tau_ms)),
        Float32(calcium_influx_scale),
        Float32(kca_half_activation),
        Float32(soma_spike_threshold_mv),
        Float32(local_ca_threshold_mv),
    )
end

"""
Preallocated external drive for a cell.

AMPA, NMDA, and GABA entries are event amplitudes added to both kinetic
states. `injected_current` is a compartment-local current used for current
clamp, sensory transduction, and deterministic capability tests.
"""
struct PaperSynapticDrive
    ampa_event::Vector{Float32}
    nmda_event::Vector{Float32}
    gaba_event::Vector{Float32}
    injected_current::Vector{Float32}
end

function PaperSynapticDrive(tree::PaperCompartmentTree)
    count = compartment_count(tree)
    return PaperSynapticDrive(
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
    )
end

function reset_drive!(drive::PaperSynapticDrive)
    fill!(drive.ampa_event, 0.0f0)
    fill!(drive.nmda_event, 0.0f0)
    fill!(drive.gaba_event, 0.0f0)
    fill!(drive.injected_current, 0.0f0)
    return drive
end

"""
Persistent state of one reduced L5 pyramidal cell.

All compartment voltages, receptor kinetics, voltage-gated channel gates and
intracellular Ca states survive a soma event. `soma_spike` is the only event
exported by the cell; `local_ca_spike` is an internal diagnostic/eligibility
signal and never resets a compartment.
"""
mutable struct PaperBiophysicalState
    voltage_mv::Vector{Float32}
    next_voltage_mv::Vector{Float32}
    axial_current::Vector{Float32}
    ampa_rise::Vector{Float32}
    ampa_decay::Vector{Float32}
    ampa_conductance::Vector{Float32}
    nmda_rise::Vector{Float32}
    nmda_decay::Vector{Float32}
    nmda_conductance::Vector{Float32}
    gaba_rise::Vector{Float32}
    gaba_decay::Vector{Float32}
    gaba_conductance::Vector{Float32}
    sodium_inactivation::Vector{Float32}
    potassium_activation::Vector{Float32}
    calcium_activation::Vector{Float32}
    calcium_inactivation::Vector{Float32}
    intracellular_calcium::Vector{Float32}
    local_ca_spike::Vector{Float32}
    soma_spike::Float32
end

@inline _sigmoid(value::Float32) = inv(1.0f0 + exp(-value))
@inline _sodium_activation_inf(voltage::Float32) =
    _sigmoid((voltage + 35.0f0) / 5.0f0)
@inline _sodium_inactivation_inf(voltage::Float32) =
    _sigmoid(-(voltage + 55.0f0) / 6.0f0)
@inline _potassium_activation_inf(voltage::Float32) =
    _sigmoid((voltage + 30.0f0) / 5.0f0)
@inline _calcium_activation_inf(voltage::Float32) =
    _sigmoid((voltage + 30.0f0) / 6.0f0)
@inline _calcium_inactivation_inf(voltage::Float32) =
    _sigmoid(-(voltage + 45.0f0) / 7.0f0)

function PaperBiophysicalState(
    tree::PaperCompartmentTree,
    parameters::PaperBiophysicalParameters,
)
    count = compartment_count(tree)
    state = PaperBiophysicalState(
        fill(parameters.resting_voltage_mv, count),
        fill(parameters.resting_voltage_mv, count),
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
        0.0f0,
    )
    return reset_state!(state, parameters)
end

function reset_state!(
    state::PaperBiophysicalState,
    parameters::PaperBiophysicalParameters,
)
    voltage = parameters.resting_voltage_mv
    fill!(state.voltage_mv, voltage)
    fill!(state.next_voltage_mv, voltage)
    fill!(state.axial_current, 0.0f0)
    fill!(state.ampa_rise, 0.0f0)
    fill!(state.ampa_decay, 0.0f0)
    fill!(state.ampa_conductance, 0.0f0)
    fill!(state.nmda_rise, 0.0f0)
    fill!(state.nmda_decay, 0.0f0)
    fill!(state.nmda_conductance, 0.0f0)
    fill!(state.gaba_rise, 0.0f0)
    fill!(state.gaba_decay, 0.0f0)
    fill!(state.gaba_conductance, 0.0f0)
    fill!(
        state.sodium_inactivation,
        _sodium_inactivation_inf(voltage),
    )
    fill!(
        state.potassium_activation,
        _potassium_activation_inf(voltage),
    )
    fill!(
        state.calcium_activation,
        _calcium_activation_inf(voltage),
    )
    fill!(
        state.calcium_inactivation,
        _calcium_inactivation_inf(voltage),
    )
    fill!(state.intracellular_calcium, 0.0f0)
    fill!(state.local_ca_spike, 0.0f0)
    state.soma_spike = 0.0f0
    return state
end

@inline function nmda_magnesium_block(
    voltage_mv::Float32,
    extracellular_magnesium_mm::Float32=1.0f0,
)
    # Jahr-Stevens form used by conductance-based NMDA models.
    return inv(
        1.0f0 +
        extracellular_magnesium_mm * exp(-0.062f0 * voltage_mv) / 3.57f0,
    )
end

@inline receptor_conductance(
    rise_state::Float32,
    decay_state::Float32,
    scale::Float32,
) = max(0.0f0, scale * (decay_state - rise_state))

@inline function _relax_gate(
    value::Float32,
    target::Float32,
    dt_ms::Float32,
    tau_ms::Float32,
)
    decay = exp(-dt_ms / tau_ms)
    return muladd(decay, value - target, target)
end

"""
Advance the complete multi-compartment cell by one integration step.

The hot path performs no allocation. Receptor kinetics, cable currents, active
channel currents and voltages are evaluated from the old voltage state; new
voltages are committed only after every compartment has been evaluated.
"""
function paper_cell_step!(
    state::PaperBiophysicalState,
    drive::PaperSynapticDrive,
    tree::PaperCompartmentTree,
    parameters::PaperBiophysicalParameters,
)
    count = compartment_count(tree)
    length(state.voltage_mv) == count ||
        throw(DimensionMismatch("state/tree compartment count"))
    length(drive.ampa_event) == count ||
        throw(DimensionMismatch("drive/tree compartment count"))

    fill!(state.axial_current, 0.0f0)
    fill!(state.local_ca_spike, 0.0f0)

    # Bidirectional cable coupling on the explicit parent tree.
    @inbounds for compartment in 2:count
        parent = Int(tree.parent[compartment])
        current =
            parameters.axial_conductance[compartment] *
            (state.voltage_mv[parent] - state.voltage_mv[compartment])
        state.axial_current[compartment] += current
        state.axial_current[parent] -= current
    end

    dt = parameters.dt_ms
    @inbounds for compartment in 1:count
        voltage = state.voltage_mv[compartment]

        ampa_rise = muladd(
            parameters.ampa_rise_decay,
            state.ampa_rise[compartment],
            drive.ampa_event[compartment],
        )
        ampa_decay = muladd(
            parameters.ampa_decay_decay,
            state.ampa_decay[compartment],
            drive.ampa_event[compartment],
        )
        ampa_g = receptor_conductance(
            ampa_rise,
            ampa_decay,
            parameters.ampa_scale,
        )
        state.ampa_rise[compartment] = ampa_rise
        state.ampa_decay[compartment] = ampa_decay
        state.ampa_conductance[compartment] = ampa_g

        nmda_rise = muladd(
            parameters.nmda_rise_decay,
            state.nmda_rise[compartment],
            drive.nmda_event[compartment],
        )
        nmda_decay = muladd(
            parameters.nmda_decay_decay,
            state.nmda_decay[compartment],
            drive.nmda_event[compartment],
        )
        nmda_g = receptor_conductance(
            nmda_rise,
            nmda_decay,
            parameters.nmda_scale,
        )
        state.nmda_rise[compartment] = nmda_rise
        state.nmda_decay[compartment] = nmda_decay
        state.nmda_conductance[compartment] = nmda_g

        gaba_rise = muladd(
            parameters.gaba_rise_decay,
            state.gaba_rise[compartment],
            drive.gaba_event[compartment],
        )
        gaba_decay = muladd(
            parameters.gaba_decay_decay,
            state.gaba_decay[compartment],
            drive.gaba_event[compartment],
        )
        gaba_g = receptor_conductance(
            gaba_rise,
            gaba_decay,
            parameters.gaba_scale,
        )
        state.gaba_rise[compartment] = gaba_rise
        state.gaba_decay[compartment] = gaba_decay
        state.gaba_conductance[compartment] = gaba_g

        h_na = _relax_gate(
            state.sodium_inactivation[compartment],
            _sodium_inactivation_inf(voltage),
            dt,
            1.2f0,
        )
        n_k = _relax_gate(
            state.potassium_activation[compartment],
            _potassium_activation_inf(voltage),
            dt,
            2.5f0,
        )
        m_ca = _relax_gate(
            state.calcium_activation[compartment],
            _calcium_activation_inf(voltage),
            dt,
            0.8f0,
        )
        h_ca = _relax_gate(
            state.calcium_inactivation[compartment],
            _calcium_inactivation_inf(voltage),
            dt,
            18.0f0,
        )
        state.sodium_inactivation[compartment] = h_na
        state.potassium_activation[compartment] = n_k
        state.calcium_activation[compartment] = m_ca
        state.calcium_inactivation[compartment] = h_ca

        m_na = _sodium_activation_inf(voltage)
        sodium_current =
            parameters.gbar_na[compartment] *
            m_na * m_na * m_na * h_na *
            (parameters.sodium_reversal_mv - voltage)
        calcium_current =
            parameters.gbar_ca[compartment] *
            m_ca * m_ca * h_ca *
            (parameters.calcium_reversal_mv - voltage)
        calcium = muladd(
            parameters.calcium_decay,
            state.intracellular_calcium[compartment],
            parameters.calcium_influx_scale * max(calcium_current, 0.0f0) * dt,
        )
        state.intracellular_calcium[compartment] = calcium
        potassium_current =
            parameters.gbar_k[compartment] *
            n_k * n_k * n_k * n_k *
            (parameters.potassium_reversal_mv - voltage)
        kca_activation =
            calcium /
            (calcium + parameters.kca_half_activation + eps(Float32))
        kca_current =
            parameters.gbar_kca[compartment] *
            kca_activation *
            (parameters.potassium_reversal_mv - voltage)

        ampa_current =
            ampa_g * (parameters.ampa_reversal_mv - voltage)
        nmda_current =
            nmda_g *
            nmda_magnesium_block(
                voltage,
                parameters.extracellular_magnesium_mm,
            ) *
            (parameters.nmda_reversal_mv - voltage)
        gaba_current =
            gaba_g * (parameters.gaba_reversal_mv - voltage)
        leak_current =
            parameters.leak_conductance[compartment] *
            (parameters.resting_voltage_mv - voltage)

        total_current =
            leak_current +
            state.axial_current[compartment] +
            ampa_current +
            nmda_current +
            gaba_current +
            sodium_current +
            calcium_current +
            potassium_current +
            kca_current +
            drive.injected_current[compartment]
        next_voltage = clamp(
            voltage +
            dt * total_current / parameters.capacitance[compartment],
            -120.0f0,
            80.0f0,
        )
        state.next_voltage_mv[compartment] = next_voltage

        if tree.region[compartment] != SOMA &&
           voltage < parameters.local_ca_threshold_mv <= next_voltage &&
           m_ca > 0.10f0
            state.local_ca_spike[compartment] = 1.0f0
        end
    end

    soma = Int(tree.soma)
    old_soma_voltage = state.voltage_mv[soma]
    new_soma_voltage = state.next_voltage_mv[soma]
    state.soma_spike = if old_soma_voltage <
                          parameters.soma_spike_threshold_mv <=
                          new_soma_voltage
        1.0f0
    else
        0.0f0
    end

    copyto!(state.voltage_mv, state.next_voltage_mv)
    return state.soma_spike
end

end # module PaperBiophysicalCell
