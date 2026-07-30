module ReducedHayWorkspaceSNN

using Lux
using Random
using Zygote

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(_PARENT_MODULE, :DendriticWorkspaceSNN)
    Base.include(
        _PARENT_MODULE,
        joinpath(
            @__DIR__,
            "..",
            "dendritic_workspace_snn",
            "DendriticWorkspaceSNN.jl",
        ),
    )
end

const Dendritic = getfield(_PARENT_MODULE, :DendriticWorkspaceSNN)
const OUTPUT_DIM = 22

export ReducedHayWorkspaceModel,
    build_reduced_hay_model,
    reduced_hay_dynamics,
    reduced_hay_parameter_count,
    reduced_hay_raw,
    reduced_hay_topology

"""
CPU-oriented reduced Hay cell embedded in the Serial Workspace graph.

Every cell owns, for each basal compartment, a voltage, AMPA conductance,
NMDA conductance, GABA conductance and slow plateau state. Apical voltage,
soma voltage and adaptation add three cell-wide states:

    5 * branches + 3

The default four-branch cell therefore has 23 persistent continuous states.
The detailed Hay model remains an oracle in `paper_multicompartment_snn`;
this model is deliberately reduced and is trained directly from the Tetris
teacher objective.
"""
struct ReducedHayWorkspaceModel <: Lux.AbstractLuxLayer
    base::Dendritic.DendriticWorkspaceModel
end

function Base.getproperty(model::ReducedHayWorkspaceModel, name::Symbol)
    name === :base && return getfield(model, :base)
    return getproperty(getfield(model, :base), name)
end

function Base.propertynames(
    model::ReducedHayWorkspaceModel,
    private::Bool=false,
)
    return (:base, propertynames(getfield(model, :base), private)...)
end

function ReducedHayWorkspaceModel(;
    blocks::Int=96,
    cells_per_block::Int=8,
    branches::Int=4,
    fanout::Int=48,
    cycles::Int=4,
    workspace_k::Int=8,
    hidden::Int=192,
    spike_temperature::Real=0.20,
    route_temperature::Real=0.35,
)
    return ReducedHayWorkspaceModel(Dendritic.DendriticWorkspaceModel(
        ;
        blocks,
        cells_per_block,
        branches,
        fanout,
        cycles,
        workspace_k,
        hidden,
        spike_temperature,
        route_temperature,
    ))
end

function build_reduced_hay_model(preset::Symbol=:reduced_hay_scaled_v1)
    preset === :tiny && return ReducedHayWorkspaceModel(
        blocks=8,
        cells_per_block=2,
        branches=4,
        fanout=8,
        cycles=3,
        workspace_k=2,
        hidden=32,
    )
    preset === :small && return ReducedHayWorkspaceModel(
        blocks=32,
        cells_per_block=4,
        branches=4,
        fanout=24,
        cycles=4,
        workspace_k=4,
        hidden=96,
    )
    preset === :reduced_hay_scaled_v1 &&
        return ReducedHayWorkspaceModel()
    throw(ArgumentError(
        "unknown Reduced Hay preset $preset; use :tiny, :small, or " *
        ":reduced_hay_scaled_v1",
    ))
end

@inline _logit(probability::Float32) =
    log(probability / (1.0f0 - probability))

function Lux.initialparameters(
    rng::AbstractRNG,
    model::ReducedHayWorkspaceModel,
)
    base = model.base
    inherited = Lux.initialparameters(rng, base)
    cells = base.blocks * base.cells_per_block
    branch_shape = (base.branches, cells)
    jitter(probability) =
        fill(_logit(Float32(probability)), branch_shape) .+
        0.04f0 .* randn(rng, Float32, branch_shape)
    cell_jitter(probability) =
        fill(_logit(Float32(probability)), cells) .+
        0.04f0 .* randn(rng, Float32, cells)

    return (;
        input_exc_gain=
            0.72f0 .+ 0.05f0 .* randn(rng, Float32, branch_shape),
        input_inh_gain=
            0.48f0 .+ 0.05f0 .* randn(rng, Float32, branch_shape),
        branch_bias=
            0.025f0 .+ 0.01f0 .* randn(rng, Float32, branch_shape),
        branch_leak_logits=jitter(0.58),
        ampa_decay_logits=jitter(0.42),
        nmda_decay_logits=jitter(0.86),
        gaba_decay_logits=jitter(0.66),
        current_gain_logits=jitter(0.56),
        axial_gain_logits=jitter(0.22),
        nmda_slope_logits=jitter(0.55),
        nmda_half_logits=jitter(0.48),
        plateau_decay_logits=jitter(0.80),
        plateau_threshold_logits=jitter(0.42),
        plateau_slope_logits=jitter(0.55),
        plateau_gain_logits=jitter(0.48),
        plateau_feedback_logits=jitter(0.25),
        soma_coupling=
            0.58f0 .+ 0.04f0 .* randn(rng, Float32, branch_shape),
        apical_leak_logits=cell_jitter(0.64),
        soma_leak_logits=cell_jitter(0.54),
        adaptation_decay_logits=cell_jitter(0.72),
        apical_gain_logits=cell_jitter(0.36),
        soma_threshold_logits=cell_jitter(0.12),
        adaptation_gain_logits=cell_jitter(0.24),
        query_weight=inherited.query_weight,
        workspace_key=inherited.workspace_key,
        feedback_gain=inherited.feedback_gain,
        synapse_weight=inherited.synapse_weight,
        gate_logits=inherited.gate_logits,
        delay_logits=inherited.delay_logits,
        workspace_decay_logit=inherited.workspace_decay_logit,
        head_weight=inherited.head_weight,
        head_bias=inherited.head_bias,
        output_weight=inherited.output_weight,
        output_bias=inherited.output_bias,
    )
end

Lux.initialstates(::AbstractRNG, ::ReducedHayWorkspaceModel) = NamedTuple()

function reduced_hay_parameter_count(value)
    value isa AbstractArray && return length(value)
    value isa NamedTuple &&
        return sum(reduced_hay_parameter_count, values(value); init=0)
    value isa Tuple &&
        return sum(reduced_hay_parameter_count, value; init=0)
    return 0
end

function reduced_hay_topology(
    model::ReducedHayWorkspaceModel,
    ps=nothing,
)
    base = model.base
    cells = base.blocks * base.cells_per_block
    return (;
        family=:reduced_hay_direct_tetris,
        blocks=base.blocks,
        cells,
        cells_per_block=base.cells_per_block,
        branches_per_cell=base.branches,
        persistent_states_per_cell=5base.branches + 3,
        persistent_state_scalars=cells * (5base.branches + 3),
        analog_readout_per_cell=base.readout_per_cell,
        block_interface_dim=base.node_dim,
        candidate_synapses=cells * base.fanout,
        enabled_synapses=ps === nothing ? nothing :
            count(>=(0.0f0), ps.gate_logits),
        fanout=base.fanout,
        cycles=base.cycles,
        workspace_capacity=base.workspace_k,
        input_rails=Dendritic.INPUT_RAILS,
        continuous_credit=:direct_bptt,
        discrete_credit=(:spike_surrogate, :route_ste, :gate_ste),
    )
end

@inline _bounded_decay(logits, low::Float32, high::Float32) =
    low .+ (high - low) .* sigmoid.(logits)

"""
Voltage-dependent magnesium unblock in normalized voltage coordinates.

This preserves the functional dependency of Hay NMDA current on local
compartment voltage without pretending that normalized Tetris-time units are
millivolts or that the reduced cell is a biophysical reproduction.
"""
@inline function _nmda_unblock(voltage, slope_logits, half_logits)
    slope = 2.0f0 .+ 8.0f0 .* sigmoid.(slope_logits)
    half = -0.45f0 .+ 0.90f0 .* sigmoid.(half_logits)
    return sigmoid.(slope .* (voltage .- half))
end

function reduced_hay_dynamics(
    model::ReducedHayWorkspaceModel,
    rails::AbstractMatrix,
    ps,
    ;
    plateau_scale::Real=1.0f0,
    apical_scale::Real=1.0f0,
    recurrent_scale::Real=1.0f0,
)
    base = model.base
    cells = base.blocks * base.cells_per_block
    candidates = size(rails, 2)
    size(rails, 1) == Dendritic.INPUT_RAILS ||
        throw(DimensionMismatch("Reduced Hay input rails"))

    excitatory_rails =
        rails[base.excitatory_feature_for_branch, :]
    inhibitory_rails =
        rails[base.inhibitory_feature_for_branch, :]
    sensory_exc =
        reshape(ps.input_exc_gain, base.branches, cells, 1) .*
        excitatory_rails .+
        reshape(ps.branch_bias, base.branches, cells, 1)
    sensory_inh =
        reshape(ps.input_inh_gain, base.branches, cells, 1) .*
        inhibitory_rails

    query_pre = ps.query_weight * rails
    query = tanh.(Dendritic.rms_normalize(
        query_pre,
        Dendritic.QUERY_NORM_SCALE,
    ))

    branch_voltage =
        zeros(Float32, base.branches, cells, candidates)
    ampa = zeros(Float32, base.branches, cells, candidates)
    nmda = zeros(Float32, base.branches, cells, candidates)
    gaba = zeros(Float32, base.branches, cells, candidates)
    plateau = zeros(Float32, base.branches, cells, candidates)
    apical = zeros(Float32, cells, candidates)
    soma = zeros(Float32, cells, candidates)
    adaptation = zeros(Float32, cells, candidates)
    active_spikes = zeros(Float32, cells, candidates)
    previous_active_spikes = zeros(Float32, cells, candidates)
    active_spike_sum = 0.0f0
    soma_spike_sum = 0.0f0
    workspace = zeros(Float32, base.node_dim, candidates)
    final_block_mask = zeros(Float32, base.blocks, candidates)

    branch_shape = (base.branches, cells, 1)
    cell_shape = (cells, 1)
    branch_leak = reshape(
        _bounded_decay(ps.branch_leak_logits, 0.35f0, 0.96f0),
        branch_shape,
    )
    ampa_decay = reshape(
        _bounded_decay(ps.ampa_decay_logits, 0.05f0, 0.78f0),
        branch_shape,
    )
    nmda_decay = reshape(
        _bounded_decay(ps.nmda_decay_logits, 0.55f0, 0.995f0),
        branch_shape,
    )
    gaba_decay = reshape(
        _bounded_decay(ps.gaba_decay_logits, 0.20f0, 0.94f0),
        branch_shape,
    )
    current_gain = reshape(
        0.02f0 .+ 0.34f0 .* sigmoid.(ps.current_gain_logits),
        branch_shape,
    )
    axial_gain = reshape(
        0.18f0 .* sigmoid.(ps.axial_gain_logits),
        branch_shape,
    )
    plateau_decay = reshape(
        _bounded_decay(ps.plateau_decay_logits, 0.45f0, 0.995f0),
        branch_shape,
    )
    plateau_threshold = reshape(
        -0.10f0 .+ 0.85f0 .* sigmoid.(ps.plateau_threshold_logits),
        branch_shape,
    )
    plateau_slope = reshape(
        2.0f0 .+ 10.0f0 .* sigmoid.(ps.plateau_slope_logits),
        branch_shape,
    )
    plateau_gain = reshape(
        0.02f0 .+ 0.48f0 .* sigmoid.(ps.plateau_gain_logits),
        branch_shape,
    )
    plateau_feedback = reshape(
        0.30f0 .* sigmoid.(ps.plateau_feedback_logits),
        branch_shape,
    )
    soma_coupling = reshape(ps.soma_coupling, branch_shape)
    apical_leak = reshape(
        _bounded_decay(ps.apical_leak_logits, 0.35f0, 0.97f0),
        cell_shape,
    )
    soma_leak = reshape(
        _bounded_decay(ps.soma_leak_logits, 0.35f0, 0.96f0),
        cell_shape,
    )
    adaptation_decay = reshape(
        _bounded_decay(ps.adaptation_decay_logits, 0.35f0, 0.98f0),
        cell_shape,
    )
    apical_gain = reshape(
        0.85f0 .* sigmoid.(ps.apical_gain_logits),
        cell_shape,
    )
    soma_threshold = reshape(
        0.12f0 .+ 0.70f0 .* sigmoid.(ps.soma_threshold_logits),
        cell_shape,
    )
    adaptation_gain = reshape(
        0.45f0 .* sigmoid.(ps.adaptation_gain_logits),
        cell_shape,
    )
    feedback_gain = reshape(
        ps.feedback_gain,
        base.readout_per_cell,
        base.cells_per_block,
        base.blocks,
        1,
    )
    workspace_key = reshape(
        ps.workspace_key,
        base.node_dim,
        base.blocks,
        1,
    )
    nmda_slope = reshape(ps.nmda_slope_logits, branch_shape)
    nmda_half = reshape(ps.nmda_half_logits, branch_shape)

    for _cycle in 1:base.cycles
        block_state = Dendritic.exported_state(
            base,
            branch_voltage,
            apical,
            soma,
        )
        query3 = reshape(query, base.node_dim, 1, candidates)
        scores = dropdims(
            sum(block_state .* workspace_key .* query3; dims=1);
            dims=1,
        )
        scores = scores .+ 0.05f0 .* dropdims(
            sum(abs.(block_state); dims=1);
            dims=1,
        )
        block_mask = Dendritic._workspace_mask(scores, base)
        cell_mask = reshape(
            repeat(
                reshape(block_mask, 1, base.blocks, candidates),
                base.cells_per_block,
                1,
                1,
            ),
            cells,
            candidates,
        )

        recurrent_inbox =
            Float32(recurrent_scale) .*
            Dendritic.vectorized_dendritic_synapse_scan(
                base,
                active_spikes,
                previous_active_spikes,
                ps,
            )
        recurrent_exc = max.(recurrent_inbox, 0.0f0)
        recurrent_inh = max.(-recurrent_inbox, 0.0f0)
        exc_drive = recurrent_exc .+ 0.18f0 .* sensory_exc
        inh_drive = recurrent_inh .+ 0.18f0 .* sensory_inh

        next_ampa = ampa_decay .* ampa .+ exc_drive
        next_nmda = nmda_decay .* nmda .+ 0.72f0 .* exc_drive
        next_gaba = gaba_decay .* gaba .+ inh_drive
        unblock = _nmda_unblock(
            branch_voltage,
            nmda_slope,
            nmda_half,
        )
        excitatory_current =
            (next_ampa .+ next_nmda .* unblock) .*
            (1.0f0 .- branch_voltage)
        inhibitory_current =
            next_gaba .* (-1.0f0 .- branch_voltage)
        axial_current =
            axial_gain .* (
                reshape(soma, 1, cells, candidates) .-
                branch_voltage
            )
        next_branch_voltage = clamp.(
            branch_leak .* branch_voltage .+
            current_gain .* (excitatory_current .+ inhibitory_current) .+
            axial_current .+
            plateau_feedback .* plateau,
            -2.0f0,
            3.0f0,
        )
        coincidence = Dendritic._hard_sigmoid(
            plateau_slope .*
            (next_branch_voltage .- plateau_threshold),
        )
        next_plateau =
            Float32(plateau_scale) .*
            clamp.(
                plateau_decay .* plateau .+
                plateau_gain .* next_nmda .* coincidence,
                0.0f0,
                4.0f0,
            )

        selected = reshape(block_mask, 1, base.blocks, candidates)
        write = dropdims(
            sum(block_state .* selected; dims=2);
            dims=2,
        ) ./ Float32(base.workspace_k)
        decay = Dendritic.bounded_workspace_decay(
            ps.workspace_decay_logit[1],
        )
        next_workspace = tanh.(decay .* workspace .+ write)
        workspace_cells = reshape(
            next_workspace,
            base.readout_per_cell,
            base.cells_per_block,
            1,
            candidates,
        )
        apical_drive = reshape(
            dropdims(
                sum(feedback_gain .* workspace_cells; dims=1);
                dims=1,
            ),
            cells,
            candidates,
        ) ./ Float32(base.readout_per_cell)
        next_apical =
            Float32(apical_scale) .*
            (apical_leak .* apical .+ apical_drive)
        basal = dropdims(
            sum(
                soma_coupling .*
                (next_branch_voltage .+ next_plateau);
                dims=1,
            );
            dims=1,
        )
        apical_modulation =
            1.0f0 .+
            apical_gain .* Dendritic._hard_sigmoid(next_apical)
        soma_pre =
            soma_leak .* soma .+
            basal .* apical_modulation .-
            adaptation
        spikes = Dendritic._surrogate_spike(
            soma_pre,
            soma_threshold,
            base.spike_temperature,
        )
        next_active_spikes = spikes .* cell_mask
        next_soma = soma_pre .- spikes .* soma_threshold
        next_adaptation =
            adaptation_decay .* adaptation .+
            adaptation_gain .* spikes
        active_spike_sum += sum(next_active_spikes)
        soma_spike_sum += sum(spikes)

        previous_active_spikes = active_spikes
        active_spikes = next_active_spikes
        branch_voltage = next_branch_voltage
        ampa = next_ampa
        nmda = next_nmda
        gaba = next_gaba
        plateau = next_plateau
        apical = next_apical
        soma = next_soma
        adaptation = next_adaptation
        workspace = next_workspace
        final_block_mask = block_mask
    end

    final_blocks = Dendritic.exported_state(
        base,
        branch_voltage,
        apical,
        soma,
    )
    final_hard_mask = Zygote.dropgrad(final_block_mask)
    pooled = dropdims(
        sum(
            final_blocks .* reshape(
                final_hard_mask,
                1,
                base.blocks,
                candidates,
            );
            dims=2,
        );
        dims=2,
    ) ./ Float32(base.workspace_k)
    return (;
        workspace,
        query,
        pooled,
        branch_voltage,
        ampa,
        nmda,
        gaba,
        plateau,
        apical,
        soma,
        adaptation,
        active_spikes,
        active_spike_rate=
            active_spike_sum /
            Float32(cells * candidates * base.cycles),
        soma_spike_rate=
            soma_spike_sum /
            Float32(cells * candidates * base.cycles),
        final_block_mask,
    )
end

function reduced_hay_raw(
    model::ReducedHayWorkspaceModel,
    rails::AbstractMatrix,
    ps,
    ;
    plateau_scale::Real=1.0f0,
    apical_scale::Real=1.0f0,
    recurrent_scale::Real=1.0f0,
)
    dynamics = reduced_hay_dynamics(
        model,
        rails,
        ps;
        plateau_scale,
        apical_scale,
        recurrent_scale,
    )
    features = vcat(
        Dendritic.rms_normalize(dynamics.workspace),
        Dendritic.rms_normalize(dynamics.pooled),
    )
    hidden_pre = ps.head_weight * features .+ ps.head_bias
    hidden = tanh.(Dendritic.rms_normalize(
        hidden_pre,
        Dendritic.HIDDEN_NORM_SCALE,
    ))
    return ps.output_weight * hidden .+ ps.output_bias
end

function (model::ReducedHayWorkspaceModel)(rails, ps, st)
    raw = reduced_hay_raw(model, rails, ps)
    output = (;
        q=vec(raw[1:1, :]),
        death_logit=vec(raw[2:2, :]),
        quantiles=raw[3:18, :],
        geometry=raw[19:22, :],
        raw,
    )
    return output, st
end

end # module ReducedHayWorkspaceSNN
