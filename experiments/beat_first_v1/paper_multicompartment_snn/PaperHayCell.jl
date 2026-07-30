module PaperHayCell

"""
Mechanism-faithful, CPU-oriented reference cell for HD-SWSNN-TwinProp.

This is a reduced cable discretisation of the Hay et al. layer-5 pyramidal
cell, not a segment-for-segment NEURON clone.  It retains the mechanisms that
the TwinProp paper ablates: basal/apical morphology, axial cable coupling,
AMPA/NMDA/GABAA conductances, voltage-dependent NMDA magnesium block, the
separate Hay voltage-gated channel families, intracellular Ca dynamics and a
distal apical Ca hot zone.  Intrinsic biophysics is fixed; downstream
TwinProp-style code learns non-negative synaptic strength and compartment
location.

Paper-specified receptor constants are exact in this implementation:

* AMPA: rise 0.20 ms, decay 1.70 ms, maximum 0.40 nS/contact
* NMDA: rise 0.29 ms, decay 43.0 ms, maximum 0.30 nS/contact
* GABAA: rise 0.20 ms, decay 8.0 ms, maximum 0.70 nS/contact
* NMDA magnesium voltage coefficient: 0.062 / mV

The channel-density distributions and gate-rate equations use the public Hay
ModelDB 139653 values.  The compact morphology, axial conductances, fixed Ca
reversal potential and semi-implicit voltage solve are explicitly reduced
numerical assumptions required for a small allocation-free CPU kernel.
"""

export SOMA,
    BASAL,
    APICAL_TRUNK,
    APICAL_TUFT,
    ABLATION_FULL,
    ABLATION_PASSIVE,
    ABLATION_NO_NMDA,
    ABLATION_SOMA_ONLY,
    OUTER_DT_MS,
    SUBSTEP_DT_MS,
    SUBSTEPS,
    HayTree,
    HayParameters,
    HayState,
    HaySynapticDrive,
    HayDiagnostics,
    paper_hay_tree,
    compartment_count,
    nmda_magnesium_block,
    receptor_conductance,
    add_synaptic_event!,
    reset_state!,
    reset_drive!,
    reset_diagnostics!,
    hay_cell_step!

const SOMA = UInt8(0x01)
const BASAL = UInt8(0x02)
const APICAL_TRUNK = UInt8(0x03)
const APICAL_TUFT = UInt8(0x04)

const ABLATION_FULL = UInt8(0x01)
const ABLATION_PASSIVE = UInt8(0x02)
const ABLATION_NO_NMDA = UInt8(0x03)
const ABLATION_SOMA_ONLY = UInt8(0x04)

# One workspace cycle represents 1 ms.  Twenty fixed substeps preserve fast
# somatic NaT/SKv3 dynamics without exposing a tunable integration shortcut.
const OUTER_DT_MS = 1.0f0
const SUBSTEPS = Int32(20)
const SUBSTEP_DT_MS = OUTER_DT_MS / Float32(SUBSTEPS)

const AMPA_TAU_RISE_MS = 0.20f0
const AMPA_TAU_DECAY_MS = 1.70f0
const AMPA_MAX_NS = 0.40f0
const NMDA_TAU_RISE_MS = 0.29f0
const NMDA_TAU_DECAY_MS = 43.0f0
const NMDA_MAX_NS = 0.30f0
const GABAA_TAU_RISE_MS = 0.20f0
const GABAA_TAU_DECAY_MS = 8.0f0
const GABAA_MAX_NS = 0.70f0

"""
Explicit reduced layer-5 cable tree.

Compartment one is the soma.  The default topology contains four two-segment
basal branches, three apical-trunk segments, and three two-segment tuft
branches (18 compartments total).  `parent[i] < i` gives a single flattened
tree suitable for a branch-free axial-current loop.
"""
struct HayTree
    parent::Vector{Int16}
    region::Vector{UInt8}
    distance_um::Vector{Float32}
    area_um2::Vector{Float32}
    axial_conductance_ns::Vector{Float32}
    soma::Int16
    basal_terminals::Vector{Int16}
    apical_trunk::Vector{Int16}
    apical_hot_zone::Vector{Int16}
    tuft_terminals::Vector{Int16}
end

@inline compartment_count(tree::HayTree) = length(tree.parent)

function paper_hay_tree()
    parent = Int16[0]
    region = UInt8[SOMA]
    distance = Float32[0.0f0]
    area = Float32[1_256.0f0]
    axial = Float32[0.0f0]
    basal_terminals = Int16[]
    apical_trunk = Int16[]
    apical_hot_zone = Int16[]
    tuft_terminals = Int16[]

    # Four independent basal arms, each with proximal and distal cable.
    for _ in 1:4
        proximal = Int16(1)
        push!(parent, proximal)
        push!(region, BASAL)
        push!(distance, 75.0f0)
        push!(area, 420.0f0)
        push!(axial, 28.0f0)
        proximal = Int16(length(parent))

        push!(parent, proximal)
        push!(region, BASAL)
        push!(distance, 175.0f0)
        push!(area, 260.0f0)
        push!(axial, 18.0f0)
        push!(basal_terminals, Int16(length(parent)))
    end

    # Apical trunk explicitly traverses the 685--885 um Hay Ca hot zone.
    proximal = Int16(1)
    for (distance_um, area_um2, coupling_ns) in (
        (250.0f0, 900.0f0, 42.0f0),
        (700.0f0, 680.0f0, 34.0f0),
        (825.0f0, 480.0f0, 26.0f0),
    )
        push!(parent, proximal)
        push!(region, APICAL_TRUNK)
        push!(distance, distance_um)
        push!(area, area_um2)
        push!(axial, coupling_ns)
        proximal = Int16(length(parent))
        push!(apical_trunk, proximal)
        if 685.0f0 <= distance_um <= 885.0f0
            push!(apical_hot_zone, proximal)
        end
    end

    # Three bifurcated tuft arms represented by two serial segments each.
    tuft_root = proximal
    for _ in 1:3
        proximal = tuft_root
        push!(parent, proximal)
        push!(region, APICAL_TUFT)
        push!(distance, 975.0f0)
        push!(area, 260.0f0)
        push!(axial, 17.0f0)
        proximal = Int16(length(parent))

        push!(parent, proximal)
        push!(region, APICAL_TUFT)
        push!(distance, 1_125.0f0)
        push!(area, 170.0f0)
        push!(axial, 11.0f0)
        push!(tuft_terminals, Int16(length(parent)))
    end

    return HayTree(
        parent,
        region,
        distance,
        area,
        axial,
        Int16(1),
        basal_terminals,
        apical_trunk,
        apical_hot_zone,
        tuft_terminals,
    )
end

@inline function _ablation_code(ablation::Symbol)
    ablation === :full && return ABLATION_FULL
    ablation === :passive && return ABLATION_PASSIVE
    ablation === :no_nmda && return ABLATION_NO_NMDA
    ablation === :soma_only && return ABLATION_SOMA_ONLY
    throw(
        ArgumentError(
            "ablation must be :full, :passive, :no_nmda, or :soma_only",
        ),
    )
end

@inline function _double_exponential_scale(
    tau_rise_ms::Float32,
    tau_decay_ms::Float32,
)
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
Fixed intrinsic and receptor parameters.

Channel densities are mS/cm².  ModelDB values expressed in S/cm² are
multiplied by 1000.  The default public soma values are NaT 2.04, NaP
0.00172, K_P 0.00223, K_T 0.0812, SKv3.1 0.693, SK(Ca) 0.0441,
Ca_HVA 0.000992 and Ca_LVA 0.00343 S/cm².  Dendritic NaT, SKv3.1,
Im and SK(Ca) likewise use 0.0213, 0.000261, 0.0000675 and 0.0012
S/cm².  The apical Ca zone and Ih distance law follow
`L5PCbiophys3.hoc`.
"""
struct HayParameters
    ablation::UInt8
    outer_dt_ms::Float32
    substep_dt_ms::Float32
    substeps::Int32
    celsius::Float32
    cm_uf_cm2::Vector{Float32}
    g_pas::Vector{Float32}
    e_pas_mv::Vector{Float32}
    axial_conductance_ns::Vector{Float32}
    gbar_nat::Vector{Float32}
    gbar_nap::Vector{Float32}
    gbar_kp::Vector{Float32}
    gbar_kt::Vector{Float32}
    gbar_skv3::Vector{Float32}
    gbar_im::Vector{Float32}
    gbar_ih::Vector{Float32}
    gbar_cahva::Vector{Float32}
    gbar_calva::Vector{Float32}
    gbar_skca::Vector{Float32}
    ca_gamma::Vector{Float32}
    ca_decay_ms::Vector{Float32}
    ena_mv::Float32
    ek_mv::Float32
    eh_mv::Float32
    eca_mv::Float32
    e_exc_mv::Float32
    e_gaba_mv::Float32
    ca_rest_mm::Float32
    ampa_rise_decay::Float32
    ampa_decay_decay::Float32
    ampa_scale::Float32
    ampa_max_ns::Float32
    nmda_rise_decay::Float32
    nmda_decay_decay::Float32
    nmda_scale::Float32
    nmda_max_ns::Float32
    gaba_rise_decay::Float32
    gaba_decay_decay::Float32
    gaba_scale::Float32
    gaba_max_ns::Float32
    extracellular_magnesium_mm::Float32
    soma_spike_threshold_mv::Float32
    local_ca_threshold_mv::Float32
end

function HayParameters(tree::HayTree; ablation::Symbol=:full)
    mode = _ablation_code(ablation)
    count = compartment_count(tree)
    cm = fill(2.0f0, count)
    g_pas = fill(0.0467f0, count)
    e_pas = fill(-90.0f0, count)
    axial = copy(tree.axial_conductance_ns)

    gbar_nat = zeros(Float32, count)
    gbar_nap = zeros(Float32, count)
    gbar_kp = zeros(Float32, count)
    gbar_kt = zeros(Float32, count)
    gbar_skv3 = zeros(Float32, count)
    gbar_im = zeros(Float32, count)
    gbar_ih = zeros(Float32, count)
    gbar_cahva = zeros(Float32, count)
    gbar_calva = zeros(Float32, count)
    gbar_skca = zeros(Float32, count)
    ca_gamma = fill(0.000509f0, count)
    ca_decay = fill(460.0f0, count)

    @inbounds for compartment in 1:count
        region = tree.region[compartment]
        distance = tree.distance_um[compartment]
        if region == SOMA
            cm[compartment] = 1.0f0
            g_pas[compartment] = 0.0338f0
            gbar_nat[compartment] = 2_040.0f0
            gbar_nap[compartment] = 1.72f0
            gbar_kp[compartment] = 2.23f0
            gbar_kt[compartment] = 81.2f0
            gbar_skv3[compartment] = 693.0f0
            gbar_im[compartment] = 0.0788f0
            gbar_ih[compartment] = 0.20f0
            gbar_cahva[compartment] = 0.992f0
            gbar_calva[compartment] = 3.43f0
            gbar_skca[compartment] = 44.1f0
            ca_gamma[compartment] = 0.000501f0
            ca_decay[compartment] = 460.0f0
        elseif region == BASAL
            # The public Hay basal tree is passive except for Ih.
            gbar_ih[compartment] = 0.20f0
            ca_decay[compartment] = 122.0f0
        else
            g_pas[compartment] = 0.0589f0
            gbar_nat[compartment] = 21.3f0
            gbar_skv3[compartment] = 0.261f0
            gbar_im[compartment] = 0.0675f0
            # Type-2 distribution in L5PCtemplate.hoc.  The original uses
            # distance normalised by the longest apical branch.
            normalized_distance = distance / 1_125.0f0
            gbar_ih[compartment] =
                (
                    -0.8696f0 +
                    2.0870f0 *
                    exp(3.6161f0 * normalized_distance)
                ) * 0.20f0
            in_hot_zone = 685.0f0 <= distance <= 885.0f0
            gbar_cahva[compartment] =
                in_hot_zone ? 0.555f0 : 0.0555f0
            gbar_calva[compartment] =
                in_hot_zone ? 18.700f0 : 0.187f0
            gbar_skca[compartment] = 1.20f0
            ca_decay[compartment] = 122.0f0
        end
    end

    if mode == ABLATION_PASSIVE
        for values in (
            gbar_nat,
            gbar_nap,
            gbar_kp,
            gbar_kt,
            gbar_skv3,
            gbar_im,
            gbar_ih,
            gbar_cahva,
            gbar_calva,
            gbar_skca,
        )
            fill!(values, 0.0f0)
        end
    elseif mode == ABLATION_SOMA_ONLY
        @inbounds for compartment in 2:count
            axial[compartment] = 0.0f0
            g_pas[compartment] = 0.0f0
            gbar_nat[compartment] = 0.0f0
            gbar_nap[compartment] = 0.0f0
            gbar_kp[compartment] = 0.0f0
            gbar_kt[compartment] = 0.0f0
            gbar_skv3[compartment] = 0.0f0
            gbar_im[compartment] = 0.0f0
            gbar_ih[compartment] = 0.0f0
            gbar_cahva[compartment] = 0.0f0
            gbar_calva[compartment] = 0.0f0
            gbar_skca[compartment] = 0.0f0
        end
    end

    dt = SUBSTEP_DT_MS
    return HayParameters(
        mode,
        OUTER_DT_MS,
        dt,
        SUBSTEPS,
        34.0f0,
        cm,
        g_pas,
        e_pas,
        axial,
        gbar_nat,
        gbar_nap,
        gbar_kp,
        gbar_kt,
        gbar_skv3,
        gbar_im,
        gbar_ih,
        gbar_cahva,
        gbar_calva,
        gbar_skca,
        ca_gamma,
        ca_decay,
        50.0f0,
        -85.0f0,
        -45.0f0,
        120.0f0, # reduced fixed-E_Ca assumption
        0.0f0,
        -80.0f0,
        1.0f-4,
        exp(-dt / AMPA_TAU_RISE_MS),
        exp(-dt / AMPA_TAU_DECAY_MS),
        _double_exponential_scale(
            AMPA_TAU_RISE_MS,
            AMPA_TAU_DECAY_MS,
        ),
        AMPA_MAX_NS,
        exp(-dt / NMDA_TAU_RISE_MS),
        exp(-dt / NMDA_TAU_DECAY_MS),
        _double_exponential_scale(
            NMDA_TAU_RISE_MS,
            NMDA_TAU_DECAY_MS,
        ),
        mode == ABLATION_NO_NMDA ? 0.0f0 : NMDA_MAX_NS,
        exp(-dt / GABAA_TAU_RISE_MS),
        exp(-dt / GABAA_TAU_DECAY_MS),
        _double_exponential_scale(
            GABAA_TAU_RISE_MS,
            GABAA_TAU_DECAY_MS,
        ),
        GABAA_MAX_NS,
        1.0f0,
        -20.0f0,
        -25.0f0,
    )
end

"""
Per-cycle event buffer.  AMPA/NMDA/GABA entries are non-negative contact
amplitudes, not nS.  The kernel applies 0.4/0.3/0.7 nS peak conductance
internally.  `injected_current` is positive-inward µA/cm² and is reserved for
calibration tests and sensory transduction.
"""
struct HaySynapticDrive
    ampa_event::Vector{Float32}
    nmda_event::Vector{Float32}
    gaba_event::Vector{Float32}
    injected_current::Vector{Float32}
end

function HaySynapticDrive(tree::HayTree)
    count = compartment_count(tree)
    return HaySynapticDrive(
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
    )
end

function reset_drive!(drive::HaySynapticDrive)
    fill!(drive.ampa_event, 0.0f0)
    fill!(drive.nmda_event, 0.0f0)
    fill!(drive.gaba_event, 0.0f0)
    fill!(drive.injected_current, 0.0f0)
    return drive
end

function add_synaptic_event!(
    drive::HaySynapticDrive,
    compartment::Integer;
    ampa::Real=0.0f0,
    nmda::Real=0.0f0,
    gaba::Real=0.0f0,
)
    ampa >= 0 || throw(ArgumentError("AMPA drive must be non-negative"))
    nmda >= 0 || throw(ArgumentError("NMDA drive must be non-negative"))
    gaba >= 0 || throw(ArgumentError("GABAA drive must be non-negative"))
    index = Int(compartment)
    @boundscheck checkbounds(drive.ampa_event, index)
    @inbounds begin
        drive.ampa_event[index] += Float32(ampa)
        drive.nmda_event[index] += Float32(nmda)
        drive.gaba_event[index] += Float32(gaba)
    end
    return drive
end

"""
Persistent state for all compartments and all Hay mechanism gates.

No voltage or intracellular state is reset on a soma spike.  The scalar
`soma_spike` is the sole event exported to the SNN; `local_ca_event` is an
internal diagnostic/eligibility feature.
"""
mutable struct HayState
    voltage_mv::Vector{Float32}
    next_voltage_mv::Vector{Float32}
    intracellular_calcium::Vector{Float32}
    local_ca_event::Vector{Float32}
    soma_spike::Float32
    ampa_rise::Vector{Float32}
    ampa_decay::Vector{Float32}
    nmda_rise::Vector{Float32}
    nmda_decay::Vector{Float32}
    gaba_rise::Vector{Float32}
    gaba_decay::Vector{Float32}
    nat_m::Vector{Float32}
    nat_h::Vector{Float32}
    nap_m::Vector{Float32}
    nap_h::Vector{Float32}
    kp_m::Vector{Float32}
    kp_h::Vector{Float32}
    kt_m::Vector{Float32}
    kt_h::Vector{Float32}
    skv3_m::Vector{Float32}
    im_m::Vector{Float32}
    ih_m::Vector{Float32}
    cahva_m::Vector{Float32}
    cahva_h::Vector{Float32}
    calva_m::Vector{Float32}
    calva_h::Vector{Float32}
    skca_m::Vector{Float32}
end

@inline _sigmoid(value::Float32) = inv(1.0f0 + exp(-value))
@inline _rl(old::Float32, target::Float32, tau_ms::Float32, dt::Float32) =
    muladd(exp(-dt / max(tau_ms, 1.0f-4)), old - target, target)

const _HAY_QT = 2.3f0^1.3f0 # q10 at 34 C relative to 21 C

@inline function _x_over_one_minus_exp_neg(
    x::Float32,
    scale::Float32,
)
    if abs(x) < 1.0f-4
        return scale + 0.5f0 * x + x * x / (12.0f0 * scale)
    end
    return x / (-expm1(-x / scale))
end

@inline function _x_over_exp_minus_one(
    x::Float32,
    scale::Float32,
)
    if abs(x) < 1.0f-4
        return scale - 0.5f0 * x + x * x / (12.0f0 * scale)
    end
    return x / expm1(x / scale)
end

# Reimplementations of the public ModelDB 139653 MOD rates.  The integration
# is Rush-Larsen (`cnexp` in NEURON); each mechanism retains independent
# state/current and no generic plateau gate replaces these channel families.
@inline _nat_m_alpha(v::Float32) =
    0.182f0 * _x_over_one_minus_exp_neg(v + 38.0f0, 6.0f0)
@inline _nat_m_beta(v::Float32) =
    0.124f0 * _x_over_one_minus_exp_neg(-v - 38.0f0, 6.0f0)
@inline _nat_h_alpha(v::Float32) =
    0.015f0 * _x_over_one_minus_exp_neg(-v - 66.0f0, 6.0f0)
@inline _nat_h_beta(v::Float32) =
    0.015f0 * _x_over_one_minus_exp_neg(v + 66.0f0, 6.0f0)
@inline _nat_m_inf(v::Float32) =
    _nat_m_alpha(v) / (_nat_m_alpha(v) + _nat_m_beta(v))
@inline _nat_h_inf(v::Float32) =
    _nat_h_alpha(v) / (_nat_h_alpha(v) + _nat_h_beta(v))
@inline _nat_m_tau(v::Float32) =
    inv((_nat_m_alpha(v) + _nat_m_beta(v)) * _HAY_QT)
@inline _nat_h_tau(v::Float32) =
    inv((_nat_h_alpha(v) + _nat_h_beta(v)) * _HAY_QT)

@inline _nap_m_inf(v::Float32) = _sigmoid((v + 52.6f0) / 4.6f0)
@inline _nap_h_inf(v::Float32) = _sigmoid(-(v + 48.8f0) / 10.0f0)
@inline _nap_m_tau(v::Float32) =
    6.0f0 / ((_nat_m_alpha(v) + _nat_m_beta(v)) * _HAY_QT)
@inline _nap_h_alpha(v::Float32) =
    2.88f-6 * _x_over_one_minus_exp_neg(-v - 17.0f0, 4.63f0)
@inline _nap_h_beta(v::Float32) =
    6.94f-6 * _x_over_one_minus_exp_neg(v + 64.4f0, 2.63f0)
@inline _nap_h_tau(v::Float32) =
    inv((_nap_h_alpha(v) + _nap_h_beta(v)) * _HAY_QT)

@inline function _kp_m_inf(v::Float32)
    shifted = v + 10.0f0
    return _sigmoid((shifted + 1.0f0) / 12.0f0)
end
@inline function _kp_h_inf(v::Float32)
    shifted = v + 10.0f0
    return _sigmoid(-(shifted + 54.0f0) / 11.0f0)
end
@inline function _kp_m_tau(v::Float32)
    shifted = v + 10.0f0
    value = if shifted < -50.0f0
        1.25f0 + 175.03f0 * exp(0.026f0 * shifted)
    else
        1.25f0 + 13.0f0 * exp(-0.026f0 * shifted)
    end
    return value / _HAY_QT
end
@inline function _kp_h_tau(v::Float32)
    shifted = v + 10.0f0
    return (
        360.0f0 +
        (1010.0f0 + 24.0f0 * (shifted + 55.0f0)) *
        exp(-((shifted + 75.0f0) / 48.0f0)^2)
    ) / _HAY_QT
end

@inline function _kt_m_inf(v::Float32)
    shifted = v + 10.0f0
    return _sigmoid(shifted / 19.0f0)
end
@inline function _kt_h_inf(v::Float32)
    shifted = v + 10.0f0
    return _sigmoid(-(shifted + 66.0f0) / 10.0f0)
end
@inline function _kt_m_tau(v::Float32)
    shifted = v + 10.0f0
    return (
        0.34f0 + 0.92f0 * exp(-((shifted + 71.0f0) / 59.0f0)^2)
    ) / _HAY_QT
end
@inline function _kt_h_tau(v::Float32)
    shifted = v + 10.0f0
    return (
        8.0f0 + 49.0f0 * exp(-((shifted + 73.0f0) / 23.0f0)^2)
    ) / _HAY_QT
end

@inline _skv3_m_inf(v::Float32) = _sigmoid((v - 18.7f0) / 9.7f0)
@inline _skv3_m_tau(v::Float32) =
    4.0f0 / (1.0f0 + exp(-(v + 46.56f0) / 44.14f0))

@inline function _im_m_inf(v::Float32)
    alpha = 3.3f-3 * exp(0.1f0 * (v + 35.0f0))
    beta = 3.3f-3 * exp(-0.1f0 * (v + 35.0f0))
    return alpha / (alpha + beta)
end
@inline function _im_m_tau(v::Float32)
    alpha = 3.3f-3 * exp(0.1f0 * (v + 35.0f0))
    beta = 3.3f-3 * exp(-0.1f0 * (v + 35.0f0))
    return inv((alpha + beta) * _HAY_QT)
end

@inline _ih_alpha(v::Float32) =
    0.00643f0 * _x_over_exp_minus_one(v + 154.9f0, 11.9f0)
@inline _ih_beta(v::Float32) = 0.193f0 * exp(v / 33.1f0)
@inline _ih_m_inf(v::Float32) =
    _ih_alpha(v) / (_ih_alpha(v) + _ih_beta(v))
@inline _ih_m_tau(v::Float32) = inv(_ih_alpha(v) + _ih_beta(v))

@inline _cahva_m_alpha(v::Float32) =
    0.055f0 * _x_over_exp_minus_one(-27.0f0 - v, 3.8f0)
@inline _cahva_m_beta(v::Float32) = 0.94f0 * exp((-75.0f0 - v) / 17.0f0)
@inline _cahva_h_alpha(v::Float32) = 0.000457f0 * exp((-13.0f0 - v) / 50.0f0)
@inline _cahva_h_beta(v::Float32) =
    0.0065f0 / (exp((-v - 15.0f0) / 28.0f0) + 1.0f0)
@inline _cahva_m_inf(v::Float32) =
    _cahva_m_alpha(v) / (_cahva_m_alpha(v) + _cahva_m_beta(v))
@inline _cahva_h_inf(v::Float32) =
    _cahva_h_alpha(v) / (_cahva_h_alpha(v) + _cahva_h_beta(v))
@inline _cahva_m_tau(v::Float32) =
    inv(_cahva_m_alpha(v) + _cahva_m_beta(v))
@inline _cahva_h_tau(v::Float32) =
    inv(_cahva_h_alpha(v) + _cahva_h_beta(v))

@inline function _calva_m_inf(v::Float32)
    shifted = v + 10.0f0
    return _sigmoid((shifted + 30.0f0) / 6.0f0)
end
@inline function _calva_h_inf(v::Float32)
    shifted = v + 10.0f0
    return inv(1.0f0 + exp((shifted + 80.0f0) / 6.4f0))
end
@inline function _calva_m_tau(v::Float32)
    shifted = v + 10.0f0
    return (
        5.0f0 + 20.0f0 / (1.0f0 + exp((shifted + 25.0f0) / 5.0f0))
    ) / _HAY_QT
end
@inline function _calva_h_tau(v::Float32)
    shifted = v + 10.0f0
    return (
        20.0f0 + 50.0f0 / (1.0f0 + exp((shifted + 40.0f0) / 7.0f0))
    ) / _HAY_QT
end

@inline function _skca_inf(calcium_mm::Float32)
    calcium = max(calcium_mm, 1.0f-7)
    return inv(1.0f0 + (4.3f-4 / calcium)^4.8f0)
end

function HayState(tree::HayTree, parameters::HayParameters)
    count = compartment_count(tree)
    voltage = copy(parameters.e_pas_mv)
    state = HayState(
        voltage,
        copy(voltage),
        fill(parameters.ca_rest_mm, count),
        zeros(Float32, count),
        0.0f0,
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
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
        zeros(Float32, count),
    )
    return reset_state!(state, parameters)
end

function reset_state!(state::HayState, parameters::HayParameters)
    copyto!(state.voltage_mv, parameters.e_pas_mv)
    copyto!(state.next_voltage_mv, parameters.e_pas_mv)
    fill!(state.intracellular_calcium, parameters.ca_rest_mm)
    fill!(state.local_ca_event, 0.0f0)
    state.soma_spike = 0.0f0
    fill!(state.ampa_rise, 0.0f0)
    fill!(state.ampa_decay, 0.0f0)
    fill!(state.nmda_rise, 0.0f0)
    fill!(state.nmda_decay, 0.0f0)
    fill!(state.gaba_rise, 0.0f0)
    fill!(state.gaba_decay, 0.0f0)
    @inbounds for i in eachindex(state.voltage_mv)
        v = state.voltage_mv[i]
        state.nat_m[i] = _nat_m_inf(v)
        state.nat_h[i] = _nat_h_inf(v)
        state.nap_m[i] = _nap_m_inf(v)
        state.nap_h[i] = _nap_h_inf(v)
        state.kp_m[i] = _kp_m_inf(v)
        state.kp_h[i] = _kp_h_inf(v)
        state.kt_m[i] = _kt_m_inf(v)
        state.kt_h[i] = _kt_h_inf(v)
        state.skv3_m[i] = _skv3_m_inf(v)
        state.im_m[i] = _im_m_inf(v)
        state.ih_m[i] = _ih_m_inf(v)
        state.cahva_m[i] = _cahva_m_inf(v)
        state.cahva_h[i] = _cahva_h_inf(v)
        state.calva_m[i] = _calva_m_inf(v)
        state.calva_h[i] = _calva_h_inf(v)
        state.skca_m[i] =
            _skca_inf(state.intracellular_calcium[i])
    end
    return state
end

"""
Preallocated current diagnostics.  Channel and receptor currents use the
outward-current convention (negative is inward).  `axial_current` and
`injected_current` are positive inward.  Values are from the final internal
substep of the most recent 1-ms cell step.
"""
struct HayDiagnostics
    passive_current::Vector{Float32}
    axial_current::Vector{Float32}
    ampa_current::Vector{Float32}
    nmda_current::Vector{Float32}
    gaba_current::Vector{Float32}
    nat_current::Vector{Float32}
    nap_current::Vector{Float32}
    kp_current::Vector{Float32}
    kt_current::Vector{Float32}
    skv3_current::Vector{Float32}
    im_current::Vector{Float32}
    ih_current::Vector{Float32}
    cahva_current::Vector{Float32}
    calva_current::Vector{Float32}
    skca_current::Vector{Float32}
    calcium_total::Vector{Float32}
    injected_current::Vector{Float32}
end

function HayDiagnostics(tree::HayTree)
    count = compartment_count(tree)
    return HayDiagnostics(
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
        zeros(Float32, count),
    )
end

function reset_diagnostics!(diagnostics::HayDiagnostics)
    for values in (
        diagnostics.passive_current,
        diagnostics.axial_current,
        diagnostics.ampa_current,
        diagnostics.nmda_current,
        diagnostics.gaba_current,
        diagnostics.nat_current,
        diagnostics.nap_current,
        diagnostics.kp_current,
        diagnostics.kt_current,
        diagnostics.skv3_current,
        diagnostics.im_current,
        diagnostics.ih_current,
        diagnostics.cahva_current,
        diagnostics.calva_current,
        diagnostics.skca_current,
        diagnostics.calcium_total,
        diagnostics.injected_current,
    )
        fill!(values, 0.0f0)
    end
    return diagnostics
end

@inline function nmda_magnesium_block(
    voltage_mv::Float32,
    extracellular_magnesium_mm::Float32=1.0f0,
)
    return inv(
        1.0f0 +
        extracellular_magnesium_mm *
        exp(-0.062f0 * voltage_mv) / 3.57f0,
    )
end

@inline receptor_conductance(
    rise_state::Float32,
    decay_state::Float32,
    peak_scale::Float32,
    maximum_ns::Float32,
) = maximum_ns * max(0.0f0, peak_scale * (decay_state - rise_state))

@inline function _density_from_ns(
    conductance_ns::Float32,
    area_um2::Float32,
)
    # 1 nS / 1 µm² = 100 mS/cm².
    return 100.0f0 * conductance_ns / area_um2
end

@inline function _advance_receptor(
    old_rise::Float32,
    old_decay::Float32,
    event::Float32,
    rise_decay::Float32,
    decay_decay::Float32,
)
    return (
        muladd(rise_decay, old_rise, event),
        muladd(decay_decay, old_decay, event),
    )
end

"""
Advance one detailed cell by exactly 1 ms using fixed 0.05-ms substeps.

Event amplitudes are added only on the first internal substep.  Reusing a
nonzero event buffer on a later call therefore represents another presynaptic
event in that millisecond.  Receptor drives must be non-negative.  The hot
path is allocation-free after state/drive/diagnostics construction.
"""
function hay_cell_step!(
    state::HayState,
    drive::HaySynapticDrive,
    diagnostics::HayDiagnostics,
    tree::HayTree,
    parameters::HayParameters,
)
    count = compartment_count(tree)
    fill!(state.local_ca_event, 0.0f0)
    reset_diagnostics!(diagnostics)
    state.soma_spike = 0.0f0
    dt = parameters.substep_dt_ms
    mode = parameters.ablation
    soma = Int(tree.soma)

    @inbounds for substep in 1:Int(parameters.substeps)
        fill!(diagnostics.axial_current, 0.0f0)

        if mode != ABLATION_SOMA_ONLY
            for compartment in 2:count
                parent = Int(tree.parent[compartment])
                conductance_ns =
                    parameters.axial_conductance_ns[compartment]
                delta =
                    state.voltage_mv[parent] -
                    state.voltage_mv[compartment]
                child_current = _density_from_ns(
                    conductance_ns,
                    tree.area_um2[compartment],
                ) * delta
                parent_current = _density_from_ns(
                    conductance_ns,
                    tree.area_um2[parent],
                ) * (-delta)
                diagnostics.axial_current[compartment] += child_current
                diagnostics.axial_current[parent] += parent_current
            end
        end

        for compartment in 1:count
            if mode == ABLATION_SOMA_ONLY && compartment != soma
                state.next_voltage_mv[compartment] =
                    state.voltage_mv[compartment]
                continue
            end

            voltage = state.voltage_mv[compartment]
            event_scale = substep == 1 ? 1.0f0 : 0.0f0
            ampa_event = drive.ampa_event[compartment]
            nmda_event = drive.nmda_event[compartment]
            gaba_event = drive.gaba_event[compartment]
            if ampa_event < 0.0f0 ||
               nmda_event < 0.0f0 ||
               gaba_event < 0.0f0
                throw(
                    DomainError(
                        min(ampa_event, nmda_event, gaba_event),
                        "receptor event amplitudes must be non-negative",
                    ),
                )
            end

            ampa_rise, ampa_decay = _advance_receptor(
                state.ampa_rise[compartment],
                state.ampa_decay[compartment],
                event_scale * ampa_event,
                parameters.ampa_rise_decay,
                parameters.ampa_decay_decay,
            )
            nmda_rise, nmda_decay = _advance_receptor(
                state.nmda_rise[compartment],
                state.nmda_decay[compartment],
                event_scale * nmda_event,
                parameters.nmda_rise_decay,
                parameters.nmda_decay_decay,
            )
            gaba_rise, gaba_decay = _advance_receptor(
                state.gaba_rise[compartment],
                state.gaba_decay[compartment],
                event_scale * gaba_event,
                parameters.gaba_rise_decay,
                parameters.gaba_decay_decay,
            )
            state.ampa_rise[compartment] = ampa_rise
            state.ampa_decay[compartment] = ampa_decay
            state.nmda_rise[compartment] = nmda_rise
            state.nmda_decay[compartment] = nmda_decay
            state.gaba_rise[compartment] = gaba_rise
            state.gaba_decay[compartment] = gaba_decay

            state.nat_m[compartment] = _rl(
                state.nat_m[compartment],
                _nat_m_inf(voltage),
                _nat_m_tau(voltage),
                dt,
            )
            state.nat_h[compartment] = _rl(
                state.nat_h[compartment],
                _nat_h_inf(voltage),
                _nat_h_tau(voltage),
                dt,
            )
            state.nap_m[compartment] = _rl(
                state.nap_m[compartment],
                _nap_m_inf(voltage),
                _nap_m_tau(voltage),
                dt,
            )
            state.nap_h[compartment] = _rl(
                state.nap_h[compartment],
                _nap_h_inf(voltage),
                _nap_h_tau(voltage),
                dt,
            )
            state.kp_m[compartment] = _rl(
                state.kp_m[compartment],
                _kp_m_inf(voltage),
                _kp_m_tau(voltage),
                dt,
            )
            state.kp_h[compartment] = _rl(
                state.kp_h[compartment],
                _kp_h_inf(voltage),
                _kp_h_tau(voltage),
                dt,
            )
            state.kt_m[compartment] = _rl(
                state.kt_m[compartment],
                _kt_m_inf(voltage),
                _kt_m_tau(voltage),
                dt,
            )
            state.kt_h[compartment] = _rl(
                state.kt_h[compartment],
                _kt_h_inf(voltage),
                _kt_h_tau(voltage),
                dt,
            )
            state.skv3_m[compartment] = _rl(
                state.skv3_m[compartment],
                _skv3_m_inf(voltage),
                _skv3_m_tau(voltage),
                dt,
            )
            state.im_m[compartment] = _rl(
                state.im_m[compartment],
                _im_m_inf(voltage),
                _im_m_tau(voltage),
                dt,
            )
            state.ih_m[compartment] = _rl(
                state.ih_m[compartment],
                _ih_m_inf(voltage),
                _ih_m_tau(voltage),
                dt,
            )
            state.cahva_m[compartment] = _rl(
                state.cahva_m[compartment],
                _cahva_m_inf(voltage),
                _cahva_m_tau(voltage),
                dt,
            )
            state.cahva_h[compartment] = _rl(
                state.cahva_h[compartment],
                _cahva_h_inf(voltage),
                _cahva_h_tau(voltage),
                dt,
            )
            state.calva_m[compartment] = _rl(
                state.calva_m[compartment],
                _calva_m_inf(voltage),
                _calva_m_tau(voltage),
                dt,
            )
            state.calva_h[compartment] = _rl(
                state.calva_h[compartment],
                _calva_h_inf(voltage),
                _calva_h_tau(voltage),
                dt,
            )
            state.skca_m[compartment] = _rl(
                state.skca_m[compartment],
                _skca_inf(
                    state.intracellular_calcium[compartment],
                ),
                1.0f0,
                dt,
            )

            nat_g =
                parameters.gbar_nat[compartment] *
                state.nat_m[compartment]^3 *
                state.nat_h[compartment]
            nap_g =
                parameters.gbar_nap[compartment] *
                state.nap_m[compartment]^3 *
                state.nap_h[compartment]
            kp_g =
                parameters.gbar_kp[compartment] *
                state.kp_m[compartment]^2 *
                state.kp_h[compartment]
            kt_g =
                parameters.gbar_kt[compartment] *
                state.kt_m[compartment]^4 *
                state.kt_h[compartment]
            skv3_g =
                parameters.gbar_skv3[compartment] *
                state.skv3_m[compartment]
            im_g =
                parameters.gbar_im[compartment] *
                state.im_m[compartment]
            ih_g =
                parameters.gbar_ih[compartment] *
                state.ih_m[compartment]
            cahva_g =
                parameters.gbar_cahva[compartment] *
                state.cahva_m[compartment]^2 *
                state.cahva_h[compartment]
            calva_g =
                parameters.gbar_calva[compartment] *
                state.calva_m[compartment]^2 *
                state.calva_h[compartment]
            skca_g =
                parameters.gbar_skca[compartment] *
                state.skca_m[compartment]

            ampa_ns = receptor_conductance(
                ampa_rise,
                ampa_decay,
                parameters.ampa_scale,
                parameters.ampa_max_ns,
            )
            nmda_ns = receptor_conductance(
                nmda_rise,
                nmda_decay,
                parameters.nmda_scale,
                parameters.nmda_max_ns,
            )
            gaba_ns = receptor_conductance(
                gaba_rise,
                gaba_decay,
                parameters.gaba_scale,
                parameters.gaba_max_ns,
            )
            ampa_g =
                _density_from_ns(ampa_ns, tree.area_um2[compartment])
            nmda_g =
                _density_from_ns(nmda_ns, tree.area_um2[compartment]) *
                nmda_magnesium_block(
                    voltage,
                    parameters.extracellular_magnesium_mm,
                )
            gaba_g =
                _density_from_ns(gaba_ns, tree.area_um2[compartment])
            passive_g = parameters.g_pas[compartment]

            diagnostics.passive_current[compartment] =
                passive_g * (voltage - parameters.e_pas_mv[compartment])
            diagnostics.ampa_current[compartment] =
                ampa_g * (voltage - parameters.e_exc_mv)
            diagnostics.nmda_current[compartment] =
                nmda_g * (voltage - parameters.e_exc_mv)
            diagnostics.gaba_current[compartment] =
                gaba_g * (voltage - parameters.e_gaba_mv)
            diagnostics.nat_current[compartment] =
                nat_g * (voltage - parameters.ena_mv)
            diagnostics.nap_current[compartment] =
                nap_g * (voltage - parameters.ena_mv)
            diagnostics.kp_current[compartment] =
                kp_g * (voltage - parameters.ek_mv)
            diagnostics.kt_current[compartment] =
                kt_g * (voltage - parameters.ek_mv)
            diagnostics.skv3_current[compartment] =
                skv3_g * (voltage - parameters.ek_mv)
            diagnostics.im_current[compartment] =
                im_g * (voltage - parameters.ek_mv)
            diagnostics.ih_current[compartment] =
                ih_g * (voltage - parameters.eh_mv)
            diagnostics.cahva_current[compartment] =
                cahva_g * (voltage - parameters.eca_mv)
            diagnostics.calva_current[compartment] =
                calva_g * (voltage - parameters.eca_mv)
            diagnostics.skca_current[compartment] =
                skca_g * (voltage - parameters.ek_mv)
            diagnostics.calcium_total[compartment] =
                diagnostics.cahva_current[compartment] +
                diagnostics.calva_current[compartment]
            diagnostics.injected_current[compartment] =
                drive.injected_current[compartment]

            total_g =
                passive_g +
                ampa_g +
                nmda_g +
                gaba_g +
                nat_g +
                nap_g +
                kp_g +
                kt_g +
                skv3_g +
                im_g +
                ih_g +
                cahva_g +
                calva_g +
                skca_g
            reversal_drive =
                passive_g * parameters.e_pas_mv[compartment] +
                (ampa_g + nmda_g) * parameters.e_exc_mv +
                gaba_g * parameters.e_gaba_mv +
                (nat_g + nap_g) * parameters.ena_mv +
                (kp_g + kt_g + skv3_g + im_g + skca_g) *
                parameters.ek_mv +
                ih_g * parameters.eh_mv +
                (cahva_g + calva_g) * parameters.eca_mv
            inward_drive =
                diagnostics.axial_current[compartment] +
                drive.injected_current[compartment]
            scale = dt / parameters.cm_uf_cm2[compartment]
            next_voltage = clamp(
                (
                    voltage +
                    scale * (inward_drive + reversal_drive)
                ) / (1.0f0 + scale * total_g),
                -120.0f0,
                80.0f0,
            )
            state.next_voltage_mv[compartment] = next_voltage

            calcium_current =
                diagnostics.calcium_total[compartment]
            calcium = state.intracellular_calcium[compartment]
            calcium +=
                dt *
                (
                    # CaDynamics_E2: current here is µA/cm², hence the
                    # 1/1000 conversion from the MOD file's mA/cm² before
                    # applying its 10000/(2 F depth) factor.  Shell depth is
                    # the public 0.1 µm default.
                    10.0f0 * parameters.ca_gamma[compartment] *
                    max(-calcium_current, 0.0f0) /
                    (2.0f0 * 96_485.332f0 * 0.1f0) -
                    (calcium - parameters.ca_rest_mm) /
                    parameters.ca_decay_ms[compartment]
                )
            state.intracellular_calcium[compartment] =
                max(parameters.ca_rest_mm, calcium)

            if compartment != soma &&
               voltage < parameters.local_ca_threshold_mv <= next_voltage &&
               calcium_current < -1.0f-5
                state.local_ca_event[compartment] = 1.0f0
            end
        end

        old_soma = state.voltage_mv[soma]
        new_soma = state.next_voltage_mv[soma]
        if old_soma < parameters.soma_spike_threshold_mv <= new_soma
            state.soma_spike = 1.0f0
        end
        copyto!(state.voltage_mv, state.next_voltage_mv)
    end

    return state.soma_spike
end

end # module PaperHayCell
