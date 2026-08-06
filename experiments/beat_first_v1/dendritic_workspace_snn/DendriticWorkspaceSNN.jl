module DendriticWorkspaceSNN

using Lux
using Random
using Statistics
using Zygote

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(_PARENT_MODULE, :SerialWorkspaceSNN)
    Base.include(
        _PARENT_MODULE,
        joinpath(@__DIR__, "..", "serial_workspace_snn", "SerialWorkspaceSNN.jl"),
    )
end
using ..SerialWorkspaceSNN: HIDDEN_NORM_SCALE,
    INPUT_RAILS,
    QUERY_NORM_SCALE,
    RMS_NORM_EPS,
    binary_rails,
    bounded_workspace_decay,
    rms_normalize

export DendriticWorkspaceModel,
    build_dendritic_model,
    dendritic_graph_topology,
    dendritic_parameter_count,
    dendritic_structural_mask,
    dendritic_trace_candidate,
    exported_state,
    serial_dendritic_synapse_scan,
    vectorized_dendritic_synapse_scan

const OUTPUT_DIM = 22
const DEFAULT_BRANCHES = 4

"""
High-dimensional Serial Workspace SNN.

Each cell owns `2 * branches + 3` persistent states:

  * branch voltage and plateau for every basal branch,
  * apical/context voltage,
  * soma voltage,
  * adaptation.

The block interface remains a regular vector.  The default exporter exposes
`soma`, `apical`, and every branch voltage, but specialised cell models may
request a wider block contract with `readout_per_cell`.  This lets a reduced
multi-conductance cell preserve its internal state instead of being forced
back into the historical six-values-per-cell interface.
"""
struct DendriticWorkspaceModel <: Lux.AbstractLuxLayer
    blocks::Int
    cells_per_block::Int
    branches::Int
    readout_per_cell::Int
    node_dim::Int
    fanout::Int
    cycles::Int
    workspace_k::Int
    hidden::Int
    spike_temperature::Float32
    route_temperature::Float32
    source_for_destination::Matrix{Int}
    destination_for_source::Matrix{Int}
    branch_for_relation::Vector{UInt8}
    relations_for_branch::Vector{Vector{Int}}
    excitatory_feature_for_branch::Matrix{Int}
    inhibitory_feature_for_branch::Matrix{Int}
end

function _cyclic_cell_tape(
    cells::Int,
    fanout::Int,
    cells_per_block::Int,
)
    destination = Matrix{Int}(undef, cells, fanout)
    source = Matrix{Int}(undef, cells, fanout)
    block_count = div(cells, cells_per_block)
    for relation in 1:fanout
        shift = mod(
            (relation % cells_per_block) +
            cells_per_block *
            (1 + mod(relation * relation + 3relation, block_count)),
            cells,
        )
        shift == 0 && (shift = relation)
        for source_cell in 1:cells
            destination_cell = mod1(source_cell + shift, cells)
            destination[source_cell, relation] = destination_cell
            source[destination_cell, relation] = source_cell
        end
    end
    return source, destination
end

function DendriticWorkspaceModel(;
    blocks::Int=96,
    cells_per_block::Int=8,
    branches::Int=DEFAULT_BRANCHES,
    fanout::Int=48,
    cycles::Int=4,
    workspace_k::Int=8,
    hidden::Int=192,
    spike_temperature::Real=0.20,
    route_temperature::Real=0.35,
    readout_per_cell::Union{Nothing,Int}=nothing,
)
    blocks >= 2 || throw(ArgumentError("blocks must be at least two"))
    cells_per_block >= 1 ||
        throw(ArgumentError("cells_per_block must be positive"))
    branches >= 2 || throw(ArgumentError("branches must be at least two"))
    fanout >= branches ||
        throw(ArgumentError("fanout must cover every branch"))
    cycles >= 1 || throw(ArgumentError("cycles must be positive"))
    1 <= workspace_k <= blocks ||
        throw(ArgumentError("workspace_k must be in 1:blocks"))

    resolved_readout = something(readout_per_cell, branches + 2)
    resolved_readout >= 1 ||
        throw(ArgumentError("readout_per_cell must be positive"))
    cells = blocks * cells_per_block
    node_dim = cells_per_block * resolved_readout
    source, destination =
        _cyclic_cell_tape(cells, fanout, cells_per_block)
    branch_for_relation =
        UInt8[mod1(relation, branches) for relation in 1:fanout]
    relations_for_branch = [
        findall(==(UInt8(branch)), branch_for_relation)
        for branch in 1:branches
    ]
    excitatory_feature_for_branch = Matrix{Int}(undef, branches, cells)
    inhibitory_feature_for_branch = Matrix{Int}(undef, branches, cells)
    for cell in 1:cells, branch in 1:branches
        excitatory_feature_for_branch[branch, cell] =
            mod1(7919cell + 104729branch + 17, INPUT_RAILS)
        inhibitory_feature_for_branch[branch, cell] =
            mod1(1543cell + 65537branch + 97, INPUT_RAILS)
    end
    return DendriticWorkspaceModel(
        blocks,
        cells_per_block,
        branches,
        resolved_readout,
        node_dim,
        fanout,
        cycles,
        workspace_k,
        hidden,
        Float32(spike_temperature),
        Float32(route_temperature),
        source,
        destination,
        branch_for_relation,
        relations_for_branch,
        excitatory_feature_for_branch,
        inhibitory_feature_for_branch,
    )
end

function build_dendritic_model(preset::Symbol=:dendritic_scaled_v1)
    preset === :tiny && return DendriticWorkspaceModel(
        blocks=8,
        cells_per_block=2,
        branches=4,
        fanout=8,
        cycles=3,
        workspace_k=2,
        hidden=32,
    )
    preset === :small && return DendriticWorkspaceModel(
        blocks=32,
        cells_per_block=4,
        branches=4,
        fanout=24,
        cycles=4,
        workspace_k=4,
        hidden=96,
    )
    preset === :dendritic_scaled_v1 && return DendriticWorkspaceModel()
    throw(ArgumentError(
        "unknown dendritic preset $preset; use :tiny, :small, or " *
        ":dendritic_scaled_v1",
    ))
end

@inline _logit(probability::Float32) =
    log(probability / (1.0f0 - probability))

function _normalize_initial_gate_logits!(
    logits::AbstractMatrix{Float32};
    margin::Float32=0.01f0,
)
    cells, fanout = size(logits)
    keep = round(Int, fanout / 2)
    @inbounds for cell in 1:cells
        order = sortperm(
            @view(logits[cell, :]);
            rev=true,
            alg=Base.Sort.MergeSort,
        )
        for rank in 1:fanout
            relation = order[rank]
            magnitude = max(abs(logits[cell, relation]), margin)
            logits[cell, relation] =
                rank <= keep ? magnitude : -magnitude
        end
    end
    return logits
end

function Lux.initialparameters(
    rng::AbstractRNG,
    model::DendriticWorkspaceModel,
)
    cells = model.blocks * model.cells_per_block
    edge_scale = 0.65f0 / sqrt(Float32(model.fanout))
    gate_logits = 0.08f0 .* randn(rng, Float32, cells, model.fanout)
    _normalize_initial_gate_logits!(gate_logits)
    branch_shape = (model.branches, cells)
    return (;
        input_exc_gain=
            0.75f0 .+ 0.08f0 .* randn(rng, Float32, branch_shape),
        input_inh_gain=
            0.55f0 .+ 0.08f0 .* randn(rng, Float32, branch_shape),
        # A small tonic drive keeps the soma/event plane alive at scratch
        # initialization. Input-specific excitation and inhibition still
        # dominate the branch trajectories.
        branch_bias=0.03f0 .+
            0.03f0 .* randn(rng, Float32, branch_shape),
        branch_leak_logits=fill(_logit(0.54f0), branch_shape) .+
            0.05f0 .* randn(rng, Float32, branch_shape),
        plateau_decay_logits=fill(_logit(0.76f0), branch_shape) .+
            0.05f0 .* randn(rng, Float32, branch_shape),
        plateau_threshold_logits=fill(_logit(0.35f0), branch_shape) .+
            0.05f0 .* randn(rng, Float32, branch_shape),
        plateau_slope_logits=fill(_logit(0.50f0), branch_shape) .+
            0.05f0 .* randn(rng, Float32, branch_shape),
        plateau_gain_logits=fill(_logit(0.50f0), branch_shape) .+
            0.05f0 .* randn(rng, Float32, branch_shape),
        plateau_feedback_logits=fill(_logit(0.25f0), branch_shape) .+
            0.05f0 .* randn(rng, Float32, branch_shape),
        soma_coupling=0.75f0 .+
            0.05f0 .* randn(rng, Float32, branch_shape),
        apical_leak_logits=fill(_logit(0.62f0), cells) .+
            0.05f0 .* randn(rng, Float32, cells),
        soma_leak_logits=fill(_logit(0.54f0), cells) .+
            0.05f0 .* randn(rng, Float32, cells),
        adaptation_decay_logits=fill(_logit(0.70f0), cells) .+
            0.05f0 .* randn(rng, Float32, cells),
        apical_gain_logits=fill(_logit(0.35f0), cells) .+
            0.05f0 .* randn(rng, Float32, cells),
        soma_threshold_logits=fill(_logit(0.05f0), cells) .+
            0.05f0 .* randn(rng, Float32, cells),
        adaptation_gain_logits=fill(_logit(0.25f0), cells) .+
            0.05f0 .* randn(rng, Float32, cells),
        query_weight=0.12f0 .* randn(
            rng,
            Float32,
            model.node_dim,
            INPUT_RAILS,
        ) ./ sqrt(Float32(INPUT_RAILS)),
        workspace_key=0.20f0 .* randn(
            rng,
            Float32,
            model.node_dim,
            model.blocks,
        ),
        feedback_gain=0.10f0 .* randn(
            rng,
            Float32,
            model.node_dim,
            model.blocks,
        ),
        synapse_weight=edge_scale .* randn(
            rng,
            Float32,
            cells,
            model.fanout,
        ),
        gate_logits,
        delay_logits=fill(_logit(0.20f0), cells, model.fanout) .+
            0.10f0 .* randn(rng, Float32, cells, model.fanout),
        workspace_decay_logit=Float32[_logit(
            (0.75f0 - 0.60f0) / (0.95f0 - 0.60f0),
        )],
        head_weight=0.12f0 .* randn(
            rng,
            Float32,
            model.hidden,
            2model.node_dim,
        ) ./ sqrt(Float32(2model.node_dim)),
        head_bias=zeros(Float32, model.hidden),
        output_weight=0.08f0 .* randn(
            rng,
            Float32,
            OUTPUT_DIM,
            model.hidden,
        ) ./ sqrt(Float32(model.hidden)),
        output_bias=zeros(Float32, OUTPUT_DIM),
    )
end

Lux.initialstates(::AbstractRNG, ::DendriticWorkspaceModel) = NamedTuple()

function dendritic_parameter_count(value)
    value isa AbstractArray && return length(value)
    value isa NamedTuple &&
        return sum(dendritic_parameter_count, values(value); init=0)
    value isa Tuple &&
        return sum(dendritic_parameter_count, value; init=0)
    return 0
end

dendritic_structural_mask(ps) =
    Float32.(ps.gate_logits .>= 0.0f0)

function dendritic_graph_topology(
    model::DendriticWorkspaceModel,
    ps=nothing,
)
    cells = model.blocks * model.cells_per_block
    return (;
        blocks=model.blocks,
        cells,
        cells_per_block=model.cells_per_block,
        branches_per_cell=model.branches,
        persistent_states_per_cell=2model.branches + 3,
        analog_readout_per_cell=model.readout_per_cell,
        block_interface_dim=model.node_dim,
        candidate_synapses=cells * model.fanout,
        enabled_synapses=ps === nothing ? nothing :
            count(>=(0.0f0), ps.gate_logits),
        fanout=model.fanout,
        cycles=model.cycles,
        workspace_capacity=model.workspace_k,
        input_rails=INPUT_RAILS,
    )
end

function _hard_topk_mask(scores::AbstractMatrix, k::Int)
    blocks, candidates = size(scores)
    result = zeros(Float32, blocks, candidates)
    for candidate in 1:candidates
        selected = partialsortperm(
            @view(scores[:, candidate]),
            1:k;
            rev=true,
        )
        result[selected, candidate] .= 1.0f0
    end
    return result
end
Zygote.@nograd _hard_topk_mask

@inline function _straight_through_binary(logits)
    soft = sigmoid.(logits)
    hard = Float32.(logits .>= 0.0f0)
    return hard .+ soft .- Zygote.dropgrad(soft)
end

@inline function _surrogate_spike(
    soma,
    threshold,
    temperature::Float32,
)
    normalized = (soma .- threshold) ./ temperature
    soft = sigmoid.(normalized)
    hard = Float32.(soma .>= threshold)
    return hard .+ soft .- Zygote.dropgrad(soft)
end

@inline _hard_sigmoid(values) =
    clamp.(muladd.(0.2f0, values, 0.5f0), 0.0f0, 1.0f0)

function _route_probabilities(
    scores::AbstractMatrix,
    model::DendriticWorkspaceModel,
)
    block_count = Float32(size(scores, 1))
    score_mean = sum(scores; dims=1) ./ block_count
    centered = scores .- score_mean
    inverse_std = inv.(sqrt.(
        sum(abs2, centered; dims=1) ./ block_count .+
        RMS_NORM_EPS,
    ))
    logits = centered .* inverse_std ./ model.route_temperature
    shifted = logits .- maximum(logits; dims=1)
    weights = exp.(shifted)
    return Float32(model.workspace_k) .* weights ./ max.(
        sum(weights; dims=1),
        eps(Float32),
    )
end

function _workspace_mask(
    scores::AbstractMatrix,
    model::DendriticWorkspaceModel,
)
    soft = _route_probabilities(scores, model)
    hard = _hard_topk_mask(scores, model.workspace_k)
    return hard .+ soft .- Zygote.dropgrad(soft)
end

"""
Differentiable batched edge scan. Every relation targets one fixed basal
branch. The relation-to-branch mapping is fixed in v1 so the experiment tests
the compartment dynamics before structural branch relocation is introduced.
"""
function vectorized_dendritic_synapse_scan(
    model::DendriticWorkspaceModel,
    current_spikes::AbstractMatrix,
    previous_spikes::AbstractMatrix,
    ps,
)
    cells, candidates = size(current_spikes)
    expected_cells = model.blocks * model.cells_per_block
    cells == expected_cells || throw(DimensionMismatch("cell count"))
    gate = _straight_through_binary(ps.gate_logits)
    delay = sigmoid.(ps.delay_logits)
    mixed = reshape(current_spikes, cells, 1, candidates) .*
            reshape(1.0f0 .- delay, cells, model.fanout, 1) .+
            reshape(previous_spikes, cells, 1, candidates) .*
            reshape(delay, cells, model.fanout, 1)
    payload = mixed .* reshape(
        ps.synapse_weight .* gate,
        cells,
        model.fanout,
        1,
    )
    linear_source = vec(
        model.source_for_destination .+
        reshape((0:(model.fanout - 1)) .* cells, 1, model.fanout),
    )
    gathered = reshape(
        reshape(payload, cells * model.fanout, candidates)[linear_source, :],
        cells,
        model.fanout,
        candidates,
    )
    per_branch = map(1:model.branches) do branch
        relations = model.relations_for_branch[branch]
        dropdims(
            sum(@view(gathered[:, relations, :]); dims=2);
            dims=2,
        )
    end
    reshaped = map(
        values -> reshape(values, 1, cells, candidates),
        per_branch,
    )
    return cat(reshaped...; dims=1)
end

function serial_dendritic_synapse_scan(
    model::DendriticWorkspaceModel,
    current_spikes::AbstractVector,
    previous_spikes::AbstractVector,
    ps;
    record_path::Bool=false,
    path_limit::Int=256,
)
    cells = model.blocks * model.cells_per_block
    length(current_spikes) == cells ||
        throw(DimensionMismatch("current spikes"))
    length(previous_spikes) == cells ||
        throw(DimensionMismatch("previous spikes"))
    inbox = zeros(Float32, model.branches, cells)
    path = NTuple{4,Int}[]
    for relation in 1:model.fanout
        branch = Int(model.branch_for_relation[relation])
        for source in 1:cells
            ps.gate_logits[source, relation] >= 0.0f0 || continue
            delay = sigmoid(ps.delay_logits[source, relation])
            signal = (1.0f0 - delay) * current_spikes[source] +
                     delay * previous_spikes[source]
            iszero(signal) && continue
            destination =
                model.destination_for_source[source, relation]
            inbox[branch, destination] +=
                ps.synapse_weight[source, relation] * signal
            if record_path && length(path) < path_limit
                push!(path, (source, destination, branch, relation))
            end
        end
    end
    return (; inbox, path)
end

"""
Export the analog information plane as a regular block vector.
"""
function exported_state(
    model::DendriticWorkspaceModel,
    branch_voltage,
    apical,
    soma,
)
    cells = model.blocks * model.cells_per_block
    candidates = size(soma, 2)
    size(branch_voltage) ==
        (model.branches, cells, candidates) ||
        throw(DimensionMismatch("branch voltage shape"))
    values = tanh.(cat(
        reshape(soma, 1, cells, candidates),
        reshape(apical, 1, cells, candidates),
        branch_voltage;
        dims=1,
    ))
    return reshape(
        values,
        model.node_dim,
        model.blocks,
        candidates,
    )
end

function _dynamics(
    model::DendriticWorkspaceModel,
    rails::AbstractMatrix,
    ps;
    record_trace::Bool=false,
    serial::Bool=false,
)
    cells = model.blocks * model.cells_per_block
    candidates = size(rails, 2)
    serial && candidates != 1 &&
        error("serial dendritic dynamics accepts exactly one candidate")

    excitatory_rails =
        rails[model.excitatory_feature_for_branch, :]
    inhibitory_rails =
        rails[model.inhibitory_feature_for_branch, :]
    seed = reshape(ps.input_exc_gain, model.branches, cells, 1) .*
           excitatory_rails .-
           reshape(ps.input_inh_gain, model.branches, cells, 1) .*
           inhibitory_rails .+
           reshape(ps.branch_bias, model.branches, cells, 1)

    query_pre = ps.query_weight * rails
    query = tanh.(rms_normalize(query_pre, QUERY_NORM_SCALE))
    branch_voltage = seed
    plateau = zeros(Float32, model.branches, cells, candidates)
    apical = zeros(Float32, cells, candidates)
    soma = zeros(Float32, cells, candidates)
    adaptation = zeros(Float32, cells, candidates)
    active_spikes = zeros(Float32, cells, candidates)
    previous_active_spikes = zeros(Float32, cells, candidates)
    workspace = zeros(Float32, model.node_dim, candidates)
    final_block_mask = zeros(Float32, model.blocks, candidates)
    cycle_trace = Any[]

    branch_leak = reshape(
        0.45f0 .+ 0.50f0 .* sigmoid.(ps.branch_leak_logits),
        model.branches,
        cells,
        1,
    )
    plateau_decay = reshape(
        0.50f0 .+ 0.49f0 .* sigmoid.(ps.plateau_decay_logits),
        model.branches,
        cells,
        1,
    )
    plateau_threshold = reshape(
        0.10f0 .+ sigmoid.(ps.plateau_threshold_logits),
        model.branches,
        cells,
        1,
    )
    plateau_slope = reshape(
        1.0f0 .+ 7.0f0 .* sigmoid.(ps.plateau_slope_logits),
        model.branches,
        cells,
        1,
    )
    plateau_gain = reshape(
        0.05f0 .+ 0.95f0 .* sigmoid.(ps.plateau_gain_logits),
        model.branches,
        cells,
        1,
    )
    plateau_feedback = reshape(
        0.40f0 .* sigmoid.(ps.plateau_feedback_logits),
        model.branches,
        cells,
        1,
    )
    soma_coupling = reshape(
        ps.soma_coupling,
        model.branches,
        cells,
        1,
    )
    apical_leak = reshape(
        0.45f0 .+ 0.50f0 .* sigmoid.(ps.apical_leak_logits),
        cells,
        1,
    )
    soma_leak = reshape(
        0.45f0 .+ 0.50f0 .* sigmoid.(ps.soma_leak_logits),
        cells,
        1,
    )
    adaptation_decay = reshape(
        0.45f0 .+ 0.50f0 .* sigmoid.(ps.adaptation_decay_logits),
        cells,
        1,
    )
    apical_gain = reshape(
        0.75f0 .* sigmoid.(ps.apical_gain_logits),
        cells,
        1,
    )
    soma_threshold = reshape(
        0.25f0 .+ 0.75f0 .* sigmoid.(ps.soma_threshold_logits),
        cells,
        1,
    )
    adaptation_gain = reshape(
        0.50f0 .* sigmoid.(ps.adaptation_gain_logits),
        cells,
        1,
    )
    feedback_gain = reshape(
        ps.feedback_gain,
        model.readout_per_cell,
        model.cells_per_block,
        model.blocks,
        1,
    )
    workspace_key = reshape(
        ps.workspace_key,
        model.node_dim,
        model.blocks,
        1,
    )

    for cycle in 1:model.cycles
        block_state = exported_state(
            model,
            branch_voltage,
            apical,
            soma,
        )
        query3 = reshape(query, model.node_dim, 1, candidates)
        scores = dropdims(
            sum(block_state .* workspace_key .* query3; dims=1);
            dims=1,
        )
        scores = scores .+ 0.05f0 .* dropdims(
            sum(abs.(block_state); dims=1);
            dims=1,
        )
        block_mask = _workspace_mask(scores, model)
        cell_mask = reshape(
            repeat(
                reshape(block_mask, 1, model.blocks, candidates),
                model.cells_per_block,
                1,
                1,
            ),
            cells,
            candidates,
        )

        path_for_trace = nothing
        recurrent_inbox = if serial
            result = serial_dendritic_synapse_scan(
                model,
                vec(active_spikes),
                vec(previous_active_spikes),
                ps;
                record_path=record_trace,
                path_limit=cells * model.fanout,
            )
            path_for_trace = result.path
            reshape(result.inbox, model.branches, cells, 1)
        else
            vectorized_dendritic_synapse_scan(
                model,
                active_spikes,
                previous_active_spikes,
                ps,
            )
        end

        final_block_mask = block_mask
        selected = reshape(block_mask, 1, model.blocks, candidates)
        write = dropdims(sum(block_state .* selected; dims=2); dims=2) ./
            Float32(model.workspace_k)
        decay = bounded_workspace_decay(ps.workspace_decay_logit[1])
        workspace = tanh.(decay .* workspace .+ write)

        workspace_cells = reshape(
            workspace,
            model.readout_per_cell,
            model.cells_per_block,
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
        ) ./ Float32(model.readout_per_cell)

        branch_drive = recurrent_inbox .+ 0.18f0 .* seed
        next_branch_voltage =
            branch_leak .* branch_voltage .+
            branch_drive .+
            plateau_feedback .* plateau
        coincidence = _hard_sigmoid(
            plateau_slope .* (next_branch_voltage .- plateau_threshold),
        )
        recruited = max.(branch_drive, 0.0f0) .* coincidence
        next_plateau = clamp.(
            plateau_decay .* plateau .+ plateau_gain .* recruited,
            0.0f0,
            4.0f0,
        )
        next_apical = apical_leak .* apical .+ apical_drive
        basal = dropdims(
            sum(
                soma_coupling .*
                (next_branch_voltage .+ next_plateau);
                dims=1,
            );
            dims=1,
        )
        apical_modulation =
            1.0f0 .+ apical_gain .* _hard_sigmoid(next_apical)
        soma_pre =
            soma_leak .* soma .+
            basal .* apical_modulation .-
            adaptation
        spikes = _surrogate_spike(
            soma_pre,
            soma_threshold,
            model.spike_temperature,
        )
        next_active_spikes = spikes .* cell_mask
        next_soma = soma_pre .- spikes .* soma_threshold
        next_adaptation =
            adaptation_decay .* adaptation .+
            adaptation_gain .* spikes

        if record_trace
            if path_for_trace === nothing
                result = serial_dendritic_synapse_scan(
                    model,
                    vec(active_spikes[:, 1]),
                    vec(previous_active_spikes[:, 1]),
                    ps;
                    record_path=true,
                    path_limit=cells * model.fanout,
                )
                path_for_trace = result.path
            end
            push!(cycle_trace, (;
                cycle,
                active_blocks=findall(
                    >(0.5f0),
                    vec(block_mask[:, 1]),
                ),
                fired_cells=count(>(0.5f0), @view(spikes[:, 1])),
                active_fired_cells=count(
                    >(0.5f0),
                    @view(next_active_spikes[:, 1]),
                ),
                firing_path=path_for_trace,
                branch_voltage_min=minimum(
                    @view(next_branch_voltage[:, :, 1]),
                ),
                branch_voltage_max=maximum(
                    @view(next_branch_voltage[:, :, 1]),
                ),
                plateau_mean=mean(@view(next_plateau[:, :, 1])),
                plateau_active_fraction=mean(
                    @view(next_plateau[:, :, 1]) .> 0.05f0,
                ),
                apical_mean=mean(@view(next_apical[:, 1])),
                soma_mean=mean(@view(next_soma[:, 1])),
                workspace=copy(@view(workspace[:, 1])),
            ))
        end

        previous_active_spikes = active_spikes
        active_spikes = next_active_spikes
        branch_voltage = next_branch_voltage
        plateau = next_plateau
        apical = next_apical
        soma = next_soma
        adaptation = next_adaptation
    end

    final_blocks = exported_state(
        model,
        branch_voltage,
        apical,
        soma,
    )
    final_hard_mask = Zygote.dropgrad(final_block_mask)
    pooled = dropdims(sum(
        final_blocks .* reshape(
            final_hard_mask,
            1,
            model.blocks,
            candidates,
        );
        dims=2,
    ); dims=2) ./ Float32(model.workspace_k)
    return (;
        workspace,
        query,
        pooled,
        branch_voltage,
        plateau,
        apical,
        soma,
        adaptation,
        active_spikes,
        cycle_trace,
    )
end

function _head_features(dynamics)
    return vcat(
        rms_normalize(dynamics.workspace),
        rms_normalize(dynamics.pooled),
    )
end

function (model::DendriticWorkspaceModel)(input, ps, st)
    rails = binary_rails(input)
    dynamics = _dynamics(model, rails, ps)
    features = _head_features(dynamics)
    hidden_pre = ps.head_weight * features .+ ps.head_bias
    hidden = tanh.(rms_normalize(hidden_pre, HIDDEN_NORM_SCALE))
    raw = ps.output_weight * hidden .+ ps.output_bias
    output = (;
        q=vec(raw[1:1, :]),
        death_logit=vec(raw[2:2, :]),
        quantiles=raw[3:18, :],
        geometry=raw[19:22, :],
    )
    return output, st
end

function dendritic_trace_candidate(
    model::DendriticWorkspaceModel,
    input,
    ps;
    candidate::Int=1,
)
    count = size(input.board, 4)
    1 <= candidate <= count ||
        throw(BoundsError(1:count, candidate))
    sliced = (;
        board=input.board[:, :, :, candidate:candidate],
        candidate=input.candidate[:, :, :, candidate:candidate],
        difference=input.difference[:, :, :, candidate:candidate],
        aux=input.aux[:, candidate:candidate],
        next_hold=input.next_hold[:, :, candidate:candidate],
        local_mask=input.local_mask[:, :, :, candidate:candidate],
    )
    rails = binary_rails(sliced)
    dynamics = _dynamics(
        model,
        rails,
        ps;
        record_trace=true,
        serial=true,
    )
    features = _head_features(dynamics)
    hidden_pre = ps.head_weight * features .+ ps.head_bias
    hidden = tanh.(rms_normalize(hidden_pre, HIDDEN_NORM_SCALE))
    raw = ps.output_weight * hidden .+ ps.output_bias
    return (;
        raw=vec(raw),
        rails=vec(rails),
        cycles=dynamics.cycle_trace,
        final_branch_voltage=copy(dynamics.branch_voltage[:, :, 1]),
        final_plateau=copy(dynamics.plateau[:, :, 1]),
        final_apical=vec(copy(dynamics.apical[:, 1])),
        final_soma=vec(copy(dynamics.soma[:, 1])),
        final_adaptation=vec(copy(dynamics.adaptation[:, 1])),
        final_spikes=vec(copy(dynamics.active_spikes[:, 1])),
        topology=dendritic_graph_topology(model, ps),
    )
end

end # module
