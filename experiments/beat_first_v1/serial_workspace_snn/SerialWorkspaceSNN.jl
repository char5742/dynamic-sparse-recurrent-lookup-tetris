module SerialWorkspaceSNN

using Lux
using Random
using Statistics
using Zygote

export AUX_LEVELS,
    HIDDEN_RMS_SCALE,
    HIDDEN_NORM_SCALE,
    INPUT_RAILS,
    QUERY_NORM_SCALE,
    QUERY_RMS_SCALE,
    RMS_EPS,
    RMS_NORM_EPS,
    SerialWorkspaceModel,
    WORKSPACE_DECAY_MAX,
    WORKSPACE_DECAY_MIN,
    WORKSPACE_DECAY_SPAN,
    binary_rails,
    bounded_workspace_decay,
    bounded_workspace_decay_derivative,
    build_model,
    consolidate_structure,
    enabled_synapse_count,
    graph_topology,
    head_features,
    parameter_count,
    rms_inverse,
    rms_normalize,
    serial_synapse_scan,
    standardized_route_probabilities,
    structural_mask,
    trace_candidate,
    vectorized_synapse_scan

const BOARD_CELLS = 24 * 10
const QUEUE_BITS = 7 * 6
const AUX_FEATURES = 37
const AUX_LEVELS = 8
const INPUT_RAILS =
    2 * BOARD_CELLS + 2 * BOARD_CELLS + QUEUE_BITS + AUX_FEATURES * AUX_LEVELS
const OUTPUT_DIM = 22
const QUANTILES = 16
const RMS_NORM_EPS = 1.0f-4
const QUERY_NORM_SCALE = 0.50f0
const HIDDEN_NORM_SCALE = 0.75f0
const WORKSPACE_DECAY_MIN = 0.60f0
const WORKSPACE_DECAY_MAX = 0.95f0
const WORKSPACE_DECAY_SPAN =
    WORKSPACE_DECAY_MAX - WORKSPACE_DECAY_MIN
# Backward-compatible descriptive aliases for diagnostics written against the
# first normalized-readout prototype.
const RMS_EPS = RMS_NORM_EPS
const QUERY_RMS_SCALE = QUERY_NORM_SCALE
const HIDDEN_RMS_SCALE = HIDDEN_NORM_SCALE
const WORKSPACE_DECAY_RANGE = WORKSPACE_DECAY_SPAN

"""
Third beat-first model: a recurrent graph of continuous LIF-like nodes.

The fixed tape contains `nodes * fanout` candidate synapses. Every synapse has
an independently learned weight, delay, and hard ON/OFF gate. A cycle first
chooses `workspace_k` blocks by content-addressed WTA, then scans the complete
edge tape and delivers enabled spikes. The normal forward pass batches this
commutative scan; `serial_synapse_scan` executes the identical semantics one
edge at a time for auditing.
"""
struct SerialWorkspaceModel <: Lux.AbstractLuxLayer
    blocks::Int
    node_dim::Int
    fanout::Int
    cycles::Int
    workspace_k::Int
    hidden::Int
    spike_temperature::Float32
    route_temperature::Float32
    source_for_destination::Matrix{Int}
    destination_for_source::Matrix{Int}
    feature_for_node::Vector{Int}
end

function _cyclic_tape(nodes::Int, fanout::Int, node_dim::Int)
    destination = Matrix{Int}(undef, nodes, fanout)
    source = Matrix{Int}(undef, nodes, fanout)
    for relation in 1:fanout
        # Every relation is a permutation. Small and large offsets mix within
        # a block, adjacent blocks, and distant blocks while retaining an exact
        # inverse needed by the allocation-efficient training scan.
        shift = mod(
            (relation % node_dim) +
            node_dim * (1 + mod(relation * relation + 3relation, div(nodes, node_dim))),
            nodes,
        )
        shift == 0 && (shift = relation)
        for source_node in 1:nodes
            destination_node = mod1(source_node + shift, nodes)
            destination[source_node, relation] = destination_node
            source[destination_node, relation] = source_node
        end
    end
    return source, destination
end

function SerialWorkspaceModel(;
    blocks::Int=96,
    node_dim::Int=48,
    fanout::Int=24,
    cycles::Int=4,
    workspace_k::Int=8,
    hidden::Int=192,
    spike_temperature::Real=0.20,
    route_temperature::Real=0.35,
)
    blocks >= 2 || throw(ArgumentError("blocks must be at least two"))
    node_dim >= 2 || throw(ArgumentError("node_dim must be at least two"))
    1 <= workspace_k <= blocks || throw(ArgumentError("workspace_k must be in 1:blocks"))
    fanout >= 1 || throw(ArgumentError("fanout must be positive"))
    cycles >= 1 || throw(ArgumentError("cycles must be positive"))
    nodes = blocks * node_dim
    source, destination = _cyclic_tape(nodes, fanout, node_dim)
    # Deterministic coprime hashing gives every sensory rail repeated access to
    # the graph without a dense DNN input projection.
    feature_for_node = [mod1(7919 * node + 104729, INPUT_RAILS) for node in 1:nodes]
    return SerialWorkspaceModel(
        blocks,
        node_dim,
        fanout,
        cycles,
        workspace_k,
        hidden,
        Float32(spike_temperature),
        Float32(route_temperature),
        source,
        destination,
        feature_for_node,
    )
end

function build_model(preset::Symbol=:scaled)
    preset === :tiny && return SerialWorkspaceModel(
        blocks=8, node_dim=8, fanout=4, cycles=3, workspace_k=2, hidden=32,
    )
    preset === :small && return SerialWorkspaceModel(
        blocks=32, node_dim=24, fanout=12, cycles=4, workspace_k=4, hidden=96,
    )
    preset === :scaled && return SerialWorkspaceModel()
    preset === :scaled_v2 && return SerialWorkspaceModel()
    preset === :large && return SerialWorkspaceModel(
        blocks=128, node_dim=64, fanout=32, cycles=4, workspace_k=8, hidden=256,
    )
    throw(ArgumentError(
        "unknown preset $preset; use :tiny, :small, :scaled, :scaled_v2, or :large",
    ))
end

@inline _logit(probability::Float32) = log(probability / (1.0f0 - probability))

"""
Candidate-local inverse RMS shared by the differentiable reference and the
allocation-free arena implementation.
"""
@inline function rms_inverse(
    sum_squares::T,
    count::Integer,
) where {T<:AbstractFloat}
    count >= 1 || throw(ArgumentError("RMS count must be positive"))
    return inv(sqrt(sum_squares / T(count) + T(RMS_EPS)))
end

"""
Normalize every candidate column independently.  Unlike batch normalization,
this operation has no cross-candidate state and remains deterministic for
single-candidate Loihi-style inference.
"""
function rms_normalize(
    values::AbstractMatrix,
    scale::Real=1.0f0,
)
    inverse_rms = inv.(sqrt.(
        sum(abs2, values; dims=1) ./ Float32(size(values, 1)) .+
        RMS_EPS,
    ))
    return Float32(scale) .* values .* inverse_rms
end

@inline function bounded_workspace_decay(logit::Real)
    return WORKSPACE_DECAY_MIN +
        WORKSPACE_DECAY_RANGE * sigmoid(logit)
end

@inline function bounded_workspace_decay_derivative(logit::Real)
    probability = sigmoid(logit)
    return WORKSPACE_DECAY_RANGE *
        probability *
        (1.0f0 - probability)
end

"""
Project the random gate prior onto an exact per-node hard fanout budget.

Only the signs are chosen by the budget: every logit's sampled magnitude is
retained (up to `margin`).  Stable sorting makes exact ties prefer the lower
relation index.  This is an initialization invariant only; subsequent gradient
and utility-based structural learning remain free to change every gate.
"""
function _normalize_initial_gate_logits!(
    logits::AbstractMatrix{Float32};
    margin::Float32=0.01f0,
)
    margin > 0.0f0 || throw(ArgumentError("gate margin must be positive"))
    nodes, fanout = size(logits)
    keep = round(Int, fanout / 2)
    1 <= keep <= fanout || error("invalid initial gate budget")
    @inbounds for node in 1:nodes
        order = sortperm(
            @view(logits[node, :]);
            rev=true,
            alg=Base.Sort.MergeSort,
        )
        for rank in 1:fanout
            relation = order[rank]
            magnitude = max(abs(logits[node, relation]), margin)
            logits[node, relation] =
                rank <= keep ? magnitude : -magnitude
        end
    end
    return logits
end

function Lux.initialparameters(rng::AbstractRNG, model::SerialWorkspaceModel)
    nodes = model.blocks * model.node_dim
    edge_scale = 0.65f0 / sqrt(Float32(model.fanout))
    gate_logits = 0.08f0 .* randn(rng, Float32, nodes, model.fanout)
    _normalize_initial_gate_logits!(gate_logits)
    return (;
        input_gain=1.0f0 .+ 0.10f0 .* randn(rng, Float32, nodes),
        input_bias=-0.15f0 .+ 0.05f0 .* randn(rng, Float32, nodes),
        query_weight=0.12f0 .* randn(rng, Float32, model.node_dim, INPUT_RAILS) ./
            sqrt(Float32(INPUT_RAILS)),
        workspace_key=0.20f0 .* randn(rng, Float32, model.node_dim, model.blocks),
        feedback_gain=0.10f0 .* randn(rng, Float32, model.node_dim, model.blocks),
        leak_logits=fill(_logit(0.72f0), nodes) .+
            0.05f0 .* randn(rng, Float32, nodes),
        threshold_logits=fill(_logit(1.0f0 / 3.0f0), nodes) .+
            0.05f0 .* randn(rng, Float32, nodes),
        synapse_weight=edge_scale .* randn(rng, Float32, nodes, model.fanout),
        gate_logits,
        delay_logits=fill(_logit(0.20f0), nodes, model.fanout) .+
            0.10f0 .* randn(rng, Float32, nodes, model.fanout),
        workspace_decay_logit=Float32[_logit(
            (0.75f0 - WORKSPACE_DECAY_MIN) /
            WORKSPACE_DECAY_RANGE,
        )],
        head_weight=0.12f0 .* randn(
            rng, Float32, model.hidden, 2 * model.node_dim,
        ) ./ sqrt(Float32(2 * model.node_dim)),
        head_bias=zeros(Float32, model.hidden),
        output_weight=0.08f0 .* randn(rng, Float32, OUTPUT_DIM, model.hidden) ./
            sqrt(Float32(model.hidden)),
        output_bias=zeros(Float32, OUTPUT_DIM),
    )
end

Lux.initialstates(::AbstractRNG, ::SerialWorkspaceModel) = NamedTuple()

function parameter_count(value)
    value isa AbstractArray && return length(value)
    value isa NamedTuple && return sum(parameter_count, values(value); init=0)
    value isa Tuple && return sum(parameter_count, value; init=0)
    return 0
end

"""Convert the complete candidate contract to deterministic 0/1 rails."""
function binary_rails(input)
    candidates = size(input.board, 4)
    board = reshape(input.board, BOARD_CELLS, candidates)
    candidate = reshape(input.candidate, BOARD_CELLS, candidates)
    difference = reshape(input.difference, BOARD_CELLS, candidates)
    queue = reshape(input.next_hold, QUEUE_BITS, candidates)
    thresholds = reshape(
        Float32.(1:AUX_LEVELS) ./ Float32(AUX_LEVELS),
        1,
        AUX_LEVELS,
        1,
    )
    aux = reshape(input.aux, AUX_FEATURES, 1, candidates)
    thermometer = reshape(Float32.(aux .>= thresholds), AUX_FEATURES * AUX_LEVELS, candidates)
    rails = vcat(
        Float32.(board .> 0.5f0),
        Float32.(candidate .> 0.5f0),
        Float32.(difference .> 0.0f0),
        Float32.(difference .< 0.0f0),
        Float32.(queue .> 0.5f0),
        thermometer,
    )
    size(rails) == (INPUT_RAILS, candidates) || error("binary rail shape drift")
    return rails
end

"""Actual binary topology implied by the learned gate logits."""
structural_mask(ps) = Float32.(ps.gate_logits .>= 0.0f0)
enabled_synapse_count(ps) = count(>=(0.0f0), ps.gate_logits)

"""
Discrete structural-learning step with a fixed per-node fanout budget.

Gradient learning continuously changes edge weights and gate evidence. At a
consolidation boundary, the strongest `density` fraction for every source node
are switched ON and the rest OFF. The returned parameter tree is new; optimizer
moments remain valid because shapes and edge identities never change.
"""
function consolidate_structure(
    ps;
    density::Real=0.50,
    evidence_margin::Real=0.02,
)
    0.0 < density < 1.0 || throw(ArgumentError("density must be in (0,1)"))
    fanout = size(ps.gate_logits, 2)
    keep = clamp(round(Int, density * fanout), 1, fanout - 1)
    old_mask = structural_mask(ps)
    evidence = abs.(ps.synapse_weight) .* sigmoid.(ps.gate_logits)
    logits = copy(ps.gate_logits)
    margin = Float32(evidence_margin)
    for node in axes(logits, 1)
        selected = partialsortperm(@view(evidence[node, :]), 1:keep; rev=true)
        selected_mask = falses(fanout)
        selected_mask[selected] .= true
        for relation in 1:fanout
            magnitude = max(abs(logits[node, relation]), margin)
            logits[node, relation] = selected_mask[relation] ? magnitude : -magnitude
        end
    end
    next_ps = merge(ps, (; gate_logits=logits))
    new_mask = structural_mask(next_ps)
    return (;
        parameters=next_ps,
        flips=count(old_mask .!= new_mask),
        turned_on=count((old_mask .== 0.0f0) .& (new_mask .== 1.0f0)),
        turned_off=count((old_mask .== 1.0f0) .& (new_mask .== 0.0f0)),
        enabled=count(==(1.0f0), new_mask),
    )
end

function graph_topology(model::SerialWorkspaceModel, ps=nothing)
    nodes = model.blocks * model.node_dim
    return (;
        blocks=model.blocks,
        nodes,
        candidate_synapses=nodes * model.fanout,
        enabled_synapses=ps === nothing ? nothing : enabled_synapse_count(ps),
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
        selected = partialsortperm(@view(scores[:, candidate]), 1:k; rev=true)
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

@inline function _surrogate_spike(membrane, threshold, temperature::Float32)
    normalized = (membrane .- threshold) ./ temperature
    soft = sigmoid.(normalized)
    hard = Float32.(membrane .>= threshold)
    return hard .+ soft .- Zygote.dropgrad(soft)
end

"""
Stable, candidate-local soft routing surrogate.

Scores are standardized by a positive candidate-specific scale before the
softmax, so this transformation cannot change the hard top-k ordering.
Returned columns sum to `workspace_k`, matching the mass of the hard mask.
"""
function standardized_route_probabilities(
    scores::AbstractMatrix,
    model::SerialWorkspaceModel,
)
    block_count = Float32(size(scores, 1))
    score_mean = sum(scores; dims=1) ./ block_count
    centered = scores .- score_mean
    inverse_std = inv.(sqrt.(
        sum(abs2, centered; dims=1) ./ block_count .+
        RMS_NORM_EPS,
    ))
    logits =
        centered .* inverse_std ./ model.route_temperature
    shifted = logits .- maximum(logits; dims=1)
    weights = exp.(shifted)
    return Float32(model.workspace_k) .* weights ./ max.(
        sum(weights; dims=1),
        eps(Float32),
    )
end

function _workspace_mask(scores, model::SerialWorkspaceModel)
    soft = standardized_route_probabilities(scores, model)
    hard = _hard_topk_mask(scores, model.workspace_k)
    return hard .+ soft .- Zygote.dropgrad(soft)
end

"""
All tape edges in a single differentiable gather/sum. It is numerically
equivalent to accumulating the serial scan into an inbox before node updates.
"""
function vectorized_synapse_scan(
    model::SerialWorkspaceModel,
    current_spikes::AbstractMatrix,
    previous_spikes::AbstractMatrix,
    ps,
)
    nodes, candidates = size(current_spikes)
    nodes == model.blocks * model.node_dim || throw(DimensionMismatch("node count"))
    gate = _straight_through_binary(ps.gate_logits)
    delay = sigmoid.(ps.delay_logits)
    mixed = reshape(current_spikes, nodes, 1, candidates) .*
            reshape(1.0f0 .- delay, nodes, model.fanout, 1) .+
            reshape(previous_spikes, nodes, 1, candidates) .*
            reshape(delay, nodes, model.fanout, 1)
    payload = mixed .* reshape(
        ps.synapse_weight .* gate,
        nodes,
        model.fanout,
        1,
    )
    linear_source = vec(
        model.source_for_destination .+
        reshape((0:(model.fanout - 1)) .* nodes, 1, model.fanout),
    )
    gathered = reshape(payload, nodes * model.fanout, candidates)[linear_source, :]
    return dropdims(sum(reshape(gathered, nodes, model.fanout, candidates); dims=2); dims=2)
end

"""
Reference printer scan. Every candidate synapse is inspected in tape order.
Disabled or non-firing synapses do no work, but remain observable in topology.
"""
function serial_synapse_scan(
    model::SerialWorkspaceModel,
    current_spikes::AbstractVector,
    previous_spikes::AbstractVector,
    ps;
    record_path::Bool=false,
    path_limit::Int=256,
)
    nodes = model.blocks * model.node_dim
    length(current_spikes) == nodes || throw(DimensionMismatch("current spikes"))
    length(previous_spikes) == nodes || throw(DimensionMismatch("previous spikes"))
    inbox = zeros(Float32, nodes)
    path = Tuple{Int,Int,Int}[]
    for relation in 1:model.fanout
        for source in 1:nodes
            ps.gate_logits[source, relation] >= 0.0f0 || continue
            delay = sigmoid(ps.delay_logits[source, relation])
            signal = (1.0f0 - delay) * current_spikes[source] +
                     delay * previous_spikes[source]
            iszero(signal) && continue
            destination = model.destination_for_source[source, relation]
            inbox[destination] += ps.synapse_weight[source, relation] * signal
            if record_path && length(path) < path_limit
                push!(path, (source, destination, relation))
            end
        end
    end
    return (; inbox, path)
end

function _dynamics(
    model::SerialWorkspaceModel,
    rails::AbstractMatrix,
    ps;
    record_trace::Bool=false,
    serial::Bool=false,
)
    nodes = model.blocks * model.node_dim
    candidates = size(rails, 2)
    seed = ps.input_gain .* rails[model.feature_for_node, :] .+ ps.input_bias
    query_pre = ps.query_weight * rails
    query = tanh.(rms_normalize(query_pre, QUERY_NORM_SCALE))
    membrane = seed
    previous_active_spikes = zeros(Float32, nodes, candidates)
    workspace = zeros(Float32, model.node_dim, candidates)
    final_block_mask = zeros(Float32, model.blocks, candidates)
    cycle_trace = Any[]
    threshold = 0.25f0 .+ 0.75f0 .* sigmoid.(ps.threshold_logits)
    leak = 0.45f0 .+ 0.50f0 .* sigmoid.(ps.leak_logits)
    threshold_matrix = reshape(threshold, nodes, 1)
    leak_matrix = reshape(leak, nodes, 1)
    feedback_gain = reshape(ps.feedback_gain, model.node_dim, model.blocks, 1)
    workspace_key = reshape(ps.workspace_key, model.node_dim, model.blocks, 1)

    for cycle in 1:model.cycles
        block_state = reshape(membrane, model.node_dim, model.blocks, candidates)
        query3 = reshape(query, model.node_dim, 1, candidates)
        scores = dropdims(sum(block_state .* workspace_key .* query3; dims=1); dims=1)
        scores = scores .+ 0.05f0 .* dropdims(sum(abs.(block_state); dims=1); dims=1)
        block_mask = _workspace_mask(scores, model)
        spikes = _surrogate_spike(
            membrane,
            threshold_matrix,
            model.spike_temperature,
        )
        node_mask = reshape(
            repeat(reshape(block_mask, 1, model.blocks, candidates), model.node_dim, 1, 1),
            nodes,
            candidates,
        )
        active_spikes = spikes .* node_mask
        path_for_trace = nothing
        message = if serial
            candidates == 1 || error("serial dynamics accepts exactly one candidate")
            serial_result = serial_synapse_scan(
                model,
                vec(active_spikes),
                vec(previous_active_spikes),
                ps;
                record_path=record_trace,
                path_limit=nodes * model.fanout,
            )
            path_for_trace = serial_result.path
            reshape(serial_result.inbox, nodes, 1)
        else
            vectorized_synapse_scan(model, active_spikes, previous_active_spikes, ps)
        end
        final_block_mask = block_mask
        selected = reshape(block_mask, 1, model.blocks, candidates)
        write = dropdims(sum(block_state .* selected; dims=2); dims=2) ./
            Float32(model.workspace_k)
        decay = bounded_workspace_decay(ps.workspace_decay_logit[1])
        workspace = tanh.(decay .* workspace .+ write)
        feedback = reshape(
            feedback_gain .* reshape(workspace, model.node_dim, 1, candidates),
            nodes,
            candidates,
        )
        membrane = leak_matrix .* membrane .+ message .+ 0.18f0 .* seed .+
                   feedback .- spikes .* threshold_matrix

        if record_trace
            if path_for_trace === nothing
                path_for_trace = serial_synapse_scan(
                    model,
                    vec(active_spikes[:, 1]),
                    vec(previous_active_spikes[:, 1]),
                    ps;
                    record_path=true,
                    path_limit=nodes * model.fanout,
                ).path
            end
            push!(cycle_trace, (;
                cycle,
                active_blocks=findall(>(0.5f0), vec(block_mask[:, 1])),
                fired_nodes=count(>(0.5f0), @view(spikes[:, 1])),
                active_fired_nodes=count(>(0.5f0), @view(active_spikes[:, 1])),
                firing_path=path_for_trace,
                membrane_min=minimum(@view(membrane[:, 1])),
                membrane_max=maximum(@view(membrane[:, 1])),
                membrane_mean=mean(@view(membrane[:, 1])),
                workspace=copy(@view(workspace[:, 1])),
            ))
        end
        previous_active_spikes = active_spikes
    end

    final_blocks = reshape(membrane, model.node_dim, model.blocks, candidates)
    # The output may only observe the graph states selected by the last hard
    # workspace route.  Routing receives its own local three-factor credit, so
    # the readout does not inject a second soft-attention shortcut.
    final_hard_mask = Zygote.dropgrad(final_block_mask)
    pooled = dropdims(sum(
        final_blocks .*
        reshape(final_hard_mask, 1, model.blocks, candidates);
        dims=2,
    ); dims=2) ./ Float32(model.workspace_k)
    return (; workspace, query, pooled, membrane, cycle_trace)
end

"""Normalized workspace/selected-pool readout; the routing query is never exposed."""
function head_features(dynamics)
    return vcat(
        rms_normalize(dynamics.workspace),
        rms_normalize(dynamics.pooled),
    )
end

function (model::SerialWorkspaceModel)(input, ps, st)
    rails = binary_rails(input)
    dynamics = _dynamics(model, rails, ps)
    features = head_features(dynamics)
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

"""Run one candidate through the exact edge-at-a-time implementation."""
function trace_candidate(model::SerialWorkspaceModel, input, ps; candidate::Int=1)
    count = size(input.board, 4)
    1 <= candidate <= count || throw(BoundsError(1:count, candidate))
    sliced = (;
        board=input.board[:, :, :, candidate:candidate],
        candidate=input.candidate[:, :, :, candidate:candidate],
        difference=input.difference[:, :, :, candidate:candidate],
        aux=input.aux[:, candidate:candidate],
        next_hold=input.next_hold[:, :, candidate:candidate],
        local_mask=input.local_mask[:, :, :, candidate:candidate],
    )
    rails = binary_rails(sliced)
    dynamics = _dynamics(model, rails, ps; record_trace=true, serial=true)
    features = head_features(dynamics)
    hidden_pre = ps.head_weight * features .+ ps.head_bias
    hidden = tanh.(rms_normalize(hidden_pre, HIDDEN_NORM_SCALE))
    raw = ps.output_weight * hidden .+ ps.output_bias
    return (;
        raw=vec(raw),
        rails=vec(rails),
        cycles=dynamics.cycle_trace,
        final_membrane=vec(dynamics.membrane),
        topology=graph_topology(model, ps),
    )
end

end # module
