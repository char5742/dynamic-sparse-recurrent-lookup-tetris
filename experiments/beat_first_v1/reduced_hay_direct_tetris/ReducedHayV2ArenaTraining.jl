module ReducedHayV2ArenaTraining

using LinearAlgebra
using Lux
using Random
using Statistics

if !isdefined(Main, :ReducedHayWorkspaceSNN)
    Base.include(
        Main,
        joinpath(@__DIR__, "ReducedHayWorkspaceSNN.jl"),
    )
end
if !isdefined(Main, :ArenaWorkspaceTraining)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "..",
            "serial_workspace_snn",
            "ArenaWorkspaceTraining.jl",
        ),
    )
end

const Model = Main.ReducedHayWorkspaceSNN
const Point = Main.ArenaWorkspaceTraining
const Routing = Main.WorkspaceRoutingPolicy
const Queue = Point.Queue
const CpuSets = Point.CpuSets
const InputModel = Main.SerialWorkspaceSNN

const OUTPUT_DIM = 22
const QUANTILES = 16
const LOCAL_PREDICTOR_SPIKE_SCALE = 0.25f0
const LOCAL_SIGNAL_RMS_EPSILON = 1.0f-4
const LOCAL_GEOMETRY_WEIGHT = 0.10f0
const GATE_SIGN_EPSILON = 1.0f-6
const ROUTING_REWARD_SEMANTICS = :supervised_reward_surrogate

export DendriticArenaExecutor,
    DendriticArenaMetrics,
    DendriticArenaTrainer,
    DendriticParameterCache,
    ReducedHayV2ArenaExecutor,
    ReducedHayV2ArenaTrainer,
    dendritic_arena_output,
    dendritic_arena_update!,
    dendritic_forward_candidate!,
    dendritic_parameter_deltas,
    dendritic_training_arena,
    reduced_hay_v2_arena_output,
    reduced_hay_v2_arena_forward!,
    reduced_hay_v2_arena_update!,
    reduced_hay_v2_parameter_deltas,
    reduced_hay_v2_training_arena,
    run_with_dendritic_team!

const DENDRITIC_PARAMETER_FIELDS = (
    :input_exc_logits,
    :input_inh_logits,
    :state_query_weight,
    :branch_bias,
    :branch_leak_logits,
    :ampa_decay_logits,
    :nmda_decay_logits,
    :gaba_decay_logits,
    :current_gain_logits,
    :axial_gain_logits,
    :nmda_slope_logits,
    :nmda_half_logits,
    :plateau_decay_logits,
    :plateau_threshold_logits,
    :plateau_slope_logits,
    :plateau_gain_logits,
    :plateau_feedback_logits,
    :soma_coupling,
    :apical_leak_logits,
    :soma_leak_logits,
    :adaptation_decay_logits,
    :apical_gain_logits,
    :soma_threshold_logits,
    :adaptation_gain_logits,
    :workspace_key,
    :feedback_gain,
    :synapse_weight,
    :gate_logits,
    :delay_logits,
    :workspace_decay_logit,
    :head_weight,
    :head_bias,
    :output_weight,
    :output_bias,
)

const HEAD_PARAMETER_FIELDS = (
    :head_weight,
    :head_bias,
    :output_weight,
    :output_bias,
)

@inline _logistic_derivative(value::Float32) =
    value * (1.0f0 - value)

@inline function _hard_sigmoid(value::Float32)
    return clamp(muladd(0.2f0, value, 0.5f0), 0.0f0, 1.0f0)
end

@inline function _hard_sigmoid_derivative(value::Float32)
    return (-2.5f0 < value < 2.5f0) ? 0.2f0 : 0.0f0
end

@inline function _spike_surrogate(
    voltage::Float32,
    threshold::Float32,
    temperature::Float32,
)
    soft = sigmoid((voltage - threshold) / temperature)
    return soft * (1.0f0 - soft) / temperature
end

function _zero_parameter_tree(parameters)
    keys(parameters) == DENDRITIC_PARAMETER_FIELDS || error(
        "dendritic parameter registry changed",
    )
    return NamedTuple{keys(parameters)}(
        map(array -> zeros(Float32, size(array)), values(parameters)),
    )
end

@generated function _fill_parameter_tree!(
    tree::NamedTuple{K},
    value::Float32=0.0f0,
) where {K}
    operations = [
        :(fill!(getfield(tree, $(QuoteNode(name))), value))
        for name in K
    ]
    return quote
        $(operations...)
        tree
    end
end

@generated function _tree_norm(tree::NamedTuple{K}) where {K}
    operations = [
        quote
            array = getfield(tree, $(QuoteNode(name)))
            @inbounds for element in array
                square_sum = muladd(
                    Float64(element),
                    Float64(element),
                    square_sum,
                )
            end
        end
        for name in K
    ]
    return quote
        square_sum = 0.0
        $(operations...)
        sqrt(square_sum)
    end
end

function _copy_parameters(parameters)
    return NamedTuple{keys(parameters)}(
        map(copy, values(parameters)),
    )
end

mutable struct DendriticParameterCache
    input_exc_gain::Array{Float32,3}
    input_exc_derivative::Array{Float32,3}
    input_inh_gain::Array{Float32,3}
    input_inh_derivative::Array{Float32,3}
    gate_probability::Matrix{Float32}
    gate_hard::Matrix{Float32}
    gate_derivative::Matrix{Float32}
    delay::Matrix{Float32}
    delay_derivative::Matrix{Float32}
    branch_leak::Matrix{Float32}
    branch_leak_derivative::Matrix{Float32}
    ampa_decay::Matrix{Float32}
    ampa_decay_derivative::Matrix{Float32}
    nmda_decay::Matrix{Float32}
    nmda_decay_derivative::Matrix{Float32}
    gaba_decay::Matrix{Float32}
    gaba_decay_derivative::Matrix{Float32}
    current_gain::Matrix{Float32}
    current_gain_derivative::Matrix{Float32}
    axial_gain::Matrix{Float32}
    axial_gain_derivative::Matrix{Float32}
    nmda_slope::Matrix{Float32}
    nmda_slope_derivative::Matrix{Float32}
    nmda_half::Matrix{Float32}
    nmda_half_derivative::Matrix{Float32}
    plateau_decay::Matrix{Float32}
    plateau_decay_derivative::Matrix{Float32}
    plateau_threshold::Matrix{Float32}
    plateau_threshold_derivative::Matrix{Float32}
    plateau_slope::Matrix{Float32}
    plateau_slope_derivative::Matrix{Float32}
    plateau_gain::Matrix{Float32}
    plateau_gain_derivative::Matrix{Float32}
    plateau_feedback::Matrix{Float32}
    plateau_feedback_derivative::Matrix{Float32}
    apical_leak::Vector{Float32}
    apical_leak_derivative::Vector{Float32}
    soma_leak::Vector{Float32}
    soma_leak_derivative::Vector{Float32}
    adaptation_decay::Vector{Float32}
    adaptation_decay_derivative::Vector{Float32}
    apical_gain::Vector{Float32}
    apical_gain_derivative::Vector{Float32}
    soma_threshold::Vector{Float32}
    soma_threshold_derivative::Vector{Float32}
    adaptation_gain::Vector{Float32}
    adaptation_gain_derivative::Vector{Float32}
    workspace_decay::Float32
    workspace_decay_derivative::Float32
end

function DendriticParameterCache(parameters)
    cache = DendriticParameterCache(
        similar(parameters.input_exc_logits),
        similar(parameters.input_exc_logits),
        similar(parameters.input_inh_logits),
        similar(parameters.input_inh_logits),
        similar(parameters.gate_logits),
        similar(parameters.gate_logits),
        similar(parameters.gate_logits),
        similar(parameters.delay_logits),
        similar(parameters.delay_logits),
        similar(parameters.branch_leak_logits),
        similar(parameters.branch_leak_logits),
        similar(parameters.ampa_decay_logits),
        similar(parameters.ampa_decay_logits),
        similar(parameters.nmda_decay_logits),
        similar(parameters.nmda_decay_logits),
        similar(parameters.gaba_decay_logits),
        similar(parameters.gaba_decay_logits),
        similar(parameters.current_gain_logits),
        similar(parameters.current_gain_logits),
        similar(parameters.axial_gain_logits),
        similar(parameters.axial_gain_logits),
        similar(parameters.nmda_slope_logits),
        similar(parameters.nmda_slope_logits),
        similar(parameters.nmda_half_logits),
        similar(parameters.nmda_half_logits),
        similar(parameters.plateau_decay_logits),
        similar(parameters.plateau_decay_logits),
        similar(parameters.plateau_threshold_logits),
        similar(parameters.plateau_threshold_logits),
        similar(parameters.plateau_slope_logits),
        similar(parameters.plateau_slope_logits),
        similar(parameters.plateau_gain_logits),
        similar(parameters.plateau_gain_logits),
        similar(parameters.plateau_feedback_logits),
        similar(parameters.plateau_feedback_logits),
        similar(parameters.apical_leak_logits),
        similar(parameters.apical_leak_logits),
        similar(parameters.soma_leak_logits),
        similar(parameters.soma_leak_logits),
        similar(parameters.adaptation_decay_logits),
        similar(parameters.adaptation_decay_logits),
        similar(parameters.apical_gain_logits),
        similar(parameters.apical_gain_logits),
        similar(parameters.soma_threshold_logits),
        similar(parameters.soma_threshold_logits),
        similar(parameters.adaptation_gain_logits),
        similar(parameters.adaptation_gain_logits),
        0.0f0,
        0.0f0,
    )
    refresh_dendritic_cache!(cache, parameters)
    return cache
end

function refresh_dendritic_cache!(
    cache::DendriticParameterCache,
    parameters,
    gate_mask=nothing,
)
    @inbounds for index in eachindex(parameters.input_exc_logits)
        probability = sigmoid(parameters.input_exc_logits[index])
        cache.input_exc_gain[index] =
            0.002f0 + 0.198f0 * probability
        cache.input_exc_derivative[index] =
            0.198f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(parameters.input_inh_logits)
        probability = sigmoid(parameters.input_inh_logits[index])
        cache.input_inh_gain[index] =
            0.002f0 + 0.198f0 * probability
        cache.input_inh_derivative[index] =
            0.198f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(parameters.gate_logits)
        probability = sigmoid(parameters.gate_logits[index])
        cache.gate_probability[index] = probability
        cache.gate_hard[index] = if gate_mask === nothing
            parameters.gate_logits[index] >= 0.0f0 ?
                1.0f0 : 0.0f0
        else
            gate_mask[index] ? 1.0f0 : 0.0f0
        end
        cache.gate_derivative[index] =
            _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(parameters.delay_logits)
        probability = sigmoid(parameters.delay_logits[index])
        cache.delay[index] = probability
        cache.delay_derivative[index] =
            _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(parameters.branch_leak_logits)
        probability = sigmoid(parameters.branch_leak_logits[index])
        cache.branch_leak[index] = 0.35f0 + 0.61f0 * probability
        cache.branch_leak_derivative[index] =
            0.61f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(parameters.ampa_decay_logits)
        probability = sigmoid(parameters.ampa_decay_logits[index])
        cache.ampa_decay[index] = 0.05f0 + 0.73f0 * probability
        cache.ampa_decay_derivative[index] =
            0.73f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(parameters.nmda_decay_logits)
        probability = sigmoid(parameters.nmda_decay_logits[index])
        cache.nmda_decay[index] = 0.55f0 + 0.445f0 * probability
        cache.nmda_decay_derivative[index] =
            0.445f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(parameters.gaba_decay_logits)
        probability = sigmoid(parameters.gaba_decay_logits[index])
        cache.gaba_decay[index] = 0.20f0 + 0.74f0 * probability
        cache.gaba_decay_derivative[index] =
            0.74f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(parameters.current_gain_logits)
        probability = sigmoid(parameters.current_gain_logits[index])
        cache.current_gain[index] = 0.02f0 + 0.34f0 * probability
        cache.current_gain_derivative[index] =
            0.34f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(parameters.axial_gain_logits)
        probability = sigmoid(parameters.axial_gain_logits[index])
        cache.axial_gain[index] = 0.18f0 * probability
        cache.axial_gain_derivative[index] =
            0.18f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(parameters.nmda_slope_logits)
        probability = sigmoid(parameters.nmda_slope_logits[index])
        cache.nmda_slope[index] = 2.0f0 + 8.0f0 * probability
        cache.nmda_slope_derivative[index] =
            8.0f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(parameters.nmda_half_logits)
        probability = sigmoid(parameters.nmda_half_logits[index])
        cache.nmda_half[index] = -0.45f0 + 0.90f0 * probability
        cache.nmda_half_derivative[index] =
            0.90f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(parameters.plateau_decay_logits)
        probability = sigmoid(parameters.plateau_decay_logits[index])
        cache.plateau_decay[index] = 0.45f0 + 0.545f0 * probability
        cache.plateau_decay_derivative[index] =
            0.545f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(
        parameters.plateau_threshold_logits,
    )
        probability =
            sigmoid(parameters.plateau_threshold_logits[index])
        cache.plateau_threshold[index] =
            -0.10f0 + 0.85f0 * probability
        cache.plateau_threshold_derivative[index] =
            0.85f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(parameters.plateau_slope_logits)
        probability = sigmoid(parameters.plateau_slope_logits[index])
        cache.plateau_slope[index] = 2.0f0 + 10.0f0 * probability
        cache.plateau_slope_derivative[index] =
            10.0f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(parameters.plateau_gain_logits)
        probability = sigmoid(parameters.plateau_gain_logits[index])
        cache.plateau_gain[index] = 0.02f0 + 0.48f0 * probability
        cache.plateau_gain_derivative[index] =
            0.48f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(
        parameters.plateau_feedback_logits,
    )
        probability = sigmoid(parameters.plateau_feedback_logits[index])
        cache.plateau_feedback[index] = 0.30f0 * probability
        cache.plateau_feedback_derivative[index] =
            0.30f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(parameters.apical_leak_logits)
        probability = sigmoid(parameters.apical_leak_logits[index])
        cache.apical_leak[index] = 0.35f0 + 0.62f0 * probability
        cache.apical_leak_derivative[index] =
            0.62f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(parameters.soma_leak_logits)
        probability = sigmoid(parameters.soma_leak_logits[index])
        cache.soma_leak[index] = 0.35f0 + 0.61f0 * probability
        cache.soma_leak_derivative[index] =
            0.61f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(
        parameters.adaptation_decay_logits,
    )
        probability =
            sigmoid(parameters.adaptation_decay_logits[index])
        cache.adaptation_decay[index] =
            0.35f0 + 0.63f0 * probability
        cache.adaptation_decay_derivative[index] =
            0.63f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(parameters.apical_gain_logits)
        probability = sigmoid(parameters.apical_gain_logits[index])
        cache.apical_gain[index] = 0.85f0 * probability
        cache.apical_gain_derivative[index] =
            0.85f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(parameters.soma_threshold_logits)
        probability = sigmoid(parameters.soma_threshold_logits[index])
        cache.soma_threshold[index] =
            0.12f0 + 0.70f0 * probability
        cache.soma_threshold_derivative[index] =
            0.70f0 * _logistic_derivative(probability)
    end
    @inbounds for index in eachindex(parameters.adaptation_gain_logits)
        probability = sigmoid(parameters.adaptation_gain_logits[index])
        cache.adaptation_gain[index] = 0.45f0 * probability
        cache.adaptation_gain_derivative[index] =
            0.45f0 * _logistic_derivative(probability)
    end
    cache.workspace_decay =
        InputModel.bounded_workspace_decay(
            parameters.workspace_decay_logit[1],
        )
    cache.workspace_decay_derivative =
        InputModel.bounded_workspace_decay_derivative(
            parameters.workspace_decay_logit[1],
        )
    return cache
end

mutable struct DendriticTape
    base::Point.TrainingArena
    branch_voltage::Array{Float32,4}
    ampa::Array{Float32,4}
    nmda::Array{Float32,4}
    gaba::Array{Float32,4}
    plateau::Array{Float32,4}
    apical::Array{Float32,3}
    soma::Array{Float32,3}
    adaptation::Array{Float32,3}
    soma_spikes::Array{Float32,3}
    cell_spikes::Array{Float32,3}
    state_query_pre::Array{Float32,3}
    state_query::Array{Float32,3}
    state_query_inv_rms::Matrix{Float32}
    block_supervised_reward::Array{Float32,3}
    block_advantage::Array{Float32,3}
end

function DendriticTape(model, state_batch::Int, width::Int)
    base = Point.TrainingArena(model, state_batch, width)
    cells = model.blocks * model.cells_per_block
    capacity = state_batch * width
    times = model.cycles + 1
    return DendriticTape(
        base,
        zeros(
            Float32,
            cells,
            model.branches,
            times,
            capacity,
        ),
        zeros(
            Float32,
            cells,
            model.branches,
            times,
            capacity,
        ),
        zeros(
            Float32,
            cells,
            model.branches,
            times,
            capacity,
        ),
        zeros(
            Float32,
            cells,
            model.branches,
            times,
            capacity,
        ),
        zeros(
            Float32,
            cells,
            model.branches,
            times,
            capacity,
        ),
        zeros(Float32, cells, times, capacity),
        zeros(Float32, cells, times, capacity),
        zeros(Float32, cells, times, capacity),
        zeros(Float32, cells, model.cycles, capacity),
        zeros(Float32, cells, model.cycles, capacity),
        zeros(
            Float32,
            model.node_dim,
            model.cycles,
            capacity,
        ),
        zeros(
            Float32,
            model.node_dim,
            model.cycles,
            capacity,
        ),
        zeros(Float32, model.cycles, capacity),
        zeros(
            Float32,
            model.blocks,
            model.cycles,
            capacity,
        ),
        zeros(
            Float32,
            model.blocks,
            model.cycles,
            capacity,
        ),
    )
end

@inline function _cell_for_coordinate(model, coordinate::Int, block::Int)
    local_cell = div(coordinate - 1, model.readout_per_cell) + 1
    return local_cell + (block - 1) * model.cells_per_block
end

@inline _channel_for_coordinate(model, coordinate::Int) =
    mod(coordinate - 1, model.readout_per_cell) + 1

@inline function _analog_value(
    tape::DendriticTape,
    model,
    coordinate::Int,
    block::Int,
    time::Int,
    flat::Int,
)
    cell = _cell_for_coordinate(model, coordinate, block)
    channel = _channel_for_coordinate(model, coordinate)
    channel == 1 &&
        return tanh(tape.soma[cell, time, flat])
    channel == 2 &&
        return tanh(tape.apical[cell, time, flat])
    branch = channel - 2
    return tanh(tape.branch_voltage[cell, branch, time, flat])
end

@inline function _write_exported_state!(
    tape::DendriticTape,
    model,
    time::Int,
    flat::Int,
)
    node_dim = model.node_dim
    @inbounds for block in 1:model.blocks
        offset = (block - 1) * node_dim
        for coordinate in 1:node_dim
            tape.base.membrane[offset + coordinate, time, flat] =
                _analog_value(
                    tape,
                    model,
                    coordinate,
                    block,
                    time,
                    flat,
                )
        end
    end
    return nothing
end

mutable struct DendriticWorkerScratch{G}
    gradient::G
    pack::Point.PackScratch
    point_scratch::Point.CandidateScratch
    scores::Vector{Float32}
    selected::Vector{Bool}
    soft_route::Vector{Float32}
    base_route::Vector{Float32}
    route_eligibility::Vector{Float32}
    route_standardized::Vector{Float32}
    route_logweight::Vector{Float32}
    route_alpha::Vector{Float32}
    route_key::Vector{Float32}
    route_order::Vector{Int16}
    branch_inbox::Matrix{Float32}
    local_prediction::Vector{Float32}
    local_error::Vector{Float32}
    block_signal::Vector{Float32}
    soma_signal::Matrix{Float32}
    apical_signal::Matrix{Float32}
    branch_signal::Array{Float32,3}
    eligibility_weight_a::Matrix{Float32}
    eligibility_weight_n::Matrix{Float32}
    eligibility_weight_g::Matrix{Float32}
    eligibility_weight_u::Matrix{Float32}
    eligibility_weight_p::Matrix{Float32}
    eligibility_weight_s::Matrix{Float32}
    eligibility_weight_q::Matrix{Float32}
    eligibility_gate_a::Matrix{Float32}
    eligibility_gate_n::Matrix{Float32}
    eligibility_gate_g::Matrix{Float32}
    eligibility_gate_u::Matrix{Float32}
    eligibility_gate_p::Matrix{Float32}
    eligibility_gate_s::Matrix{Float32}
    eligibility_gate_q::Matrix{Float32}
    eligibility_delay_a::Matrix{Float32}
    eligibility_delay_n::Matrix{Float32}
    eligibility_delay_g::Matrix{Float32}
    eligibility_delay_u::Matrix{Float32}
    eligibility_delay_p::Matrix{Float32}
    eligibility_delay_s::Matrix{Float32}
    eligibility_delay_q::Matrix{Float32}
    utility::Matrix{Float32}
    branch_utility::Array{Float32,3}
    active_edges::Vector{Int32}
    active_edge_mask::Vector{Bool}
    active_edge_count::Int
    local_q_loss::Float64
    local_death_loss::Float64
    local_quantile_loss::Float64
    local_geometry_loss::Float64
    jobs::UInt64
    cpu_ticks::UInt64
end

function DendriticWorkerScratch(model, parameters)
    cells = model.blocks * model.cells_per_block
    edge_shape = size(parameters.synapse_weight)
    return DendriticWorkerScratch(
        _zero_parameter_tree(parameters),
        Point.PackScratch(),
        Point.CandidateScratch(model),
        zeros(Float32, model.blocks),
        fill(false, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Int16, model.workspace_k),
        zeros(Float32, cells, model.branches),
        zeros(Float32, OUTPUT_DIM),
        zeros(Float32, OUTPUT_DIM),
        zeros(Float32, model.node_dim),
        zeros(Float32, cells, model.cycles),
        zeros(Float32, cells, model.cycles),
        zeros(Float32, cells, model.branches, model.cycles),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(Float32, edge_shape),
        zeros(
            Float32,
            model.branches,
            edge_shape[1],
            edge_shape[2],
        ),
        zeros(Int32, prod(edge_shape)),
        fill(false, prod(edge_shape)),
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        UInt64(0),
        UInt64(0),
    )
end

@inline function _routing_nonce(
    seed::UInt64,
    update::Int,
    flat::Int,
)
    return Routing.routing_mix64(
        seed ⊻
        UInt64(update + 1) * UInt64(0x9e3779b97f4a7c15) ⊻
        UInt64(flat) * UInt64(0xd1b54a32d192ed03),
    )
end

@inline function _prepare_route!(
    scratch::DendriticWorkerScratch,
    model,
    stochastic::Bool,
    nonce::UInt64,
    cycle::Int,
)
    Routing.prepare_policy!(
        scratch.route_standardized,
        scratch.base_route,
        scratch.soft_route,
        scratch.scores;
        temperature=model.route_temperature,
    )
    if stochastic
        Routing.sample_plackett_luce_topk!(
            scratch.selected,
            scratch.route_order,
            scratch.route_key,
            scratch.soft_route,
            model.workspace_k,
            nonce,
            cycle,
        )
    else
        Routing.deterministic_topk!(
            scratch.selected,
            scratch.route_order,
            scratch.scores,
            model.workspace_k,
        )
    end
    Routing.ordered_score_eligibility!(
        scratch.route_eligibility,
        scratch.route_logweight,
        scratch.route_alpha,
        scratch.route_standardized,
        scratch.base_route,
        scratch.soft_route,
        scratch.scores,
        scratch.route_order,
        model.workspace_k;
        temperature=model.route_temperature,
    )
    return nothing
end

@inline function _record_route!(
    tape::DendriticTape,
    scratch::DendriticWorkerScratch,
    model,
    cycle::Int,
    flat::Int,
)
    base = tape.base
    score_square_sum = 0.0
    entropy = 0.0f0
    mask_hash = UInt64(0xcbf29ce484222325)
    churn = 0
    selected_cutoff = Inf32
    best_unselected = -Inf32
    @inbounds for block in 1:model.blocks
        selected = scratch.selected[block]
        score = scratch.scores[block]
        probability = scratch.base_route[block]
        base.block_mask[block, cycle, flat] =
            selected ? 1.0f0 : 0.0f0
        base.route_policy_probability[block, cycle, flat] =
            scratch.soft_route[block]
        base.route_base_probability[block, cycle, flat] =
            probability
        base.route_score[block, cycle, flat] = score
        base.route_eligibility[block, cycle, flat] =
            scratch.route_eligibility[block]
        score_square_sum = muladd(
            Float64(score),
            Float64(score),
            score_square_sum,
        )
        entropy -= probability * log(max(probability, 1.0f-12))
        if selected
            selected_cutoff = min(selected_cutoff, score)
            mask_hash = xor(mask_hash, UInt64(block))
            mask_hash *= UInt64(0x100000001b3)
        else
            best_unselected = max(best_unselected, score)
        end
        if cycle > 1
            previous =
                base.block_mask[block, cycle - 1, flat] != 0.0f0
            churn += selected != previous
        end
    end
    @inbounds for rank in 1:model.workspace_k
        base.route_order[rank, cycle, flat] =
            scratch.route_order[rank]
    end
    base.route_selection_gap_value[cycle, flat] =
        model.workspace_k == model.blocks ? 0.0f0 :
        selected_cutoff - best_unselected
    base.route_score_square_sum[cycle, flat] = score_square_sum
    base.route_normalized_entropy[cycle, flat] =
        model.blocks == 1 ? 1.0f0 :
        entropy / log(Float32(model.blocks))
    base.route_mask_fingerprint[cycle, flat] = mask_hash
    base.route_cycle_churn_count[cycle, flat] = Int16(churn)
    return nothing
end

function dendritic_forward_candidate!(
    tape::DendriticTape,
    model,
    parameters,
    cache::DendriticParameterCache,
    scratch::DendriticWorkerScratch,
    branch_for_edge::Matrix{UInt8},
    flat::Int;
    stochastic_routing::Bool=false,
    routing_nonce::UInt64=UInt64(0),
)
    base = tape.base
    cells = model.blocks * model.cells_per_block
    node_dim = model.node_dim
    readout = model.readout_per_cell
    @inbounds for cell in 1:cells
        for branch in 1:model.branches
            tape.branch_voltage[cell, branch, 1, flat] = 0.0f0
            tape.ampa[cell, branch, 1, flat] = 0.0f0
            tape.nmda[cell, branch, 1, flat] = 0.0f0
            tape.gaba[cell, branch, 1, flat] = 0.0f0
            tape.plateau[cell, branch, 1, flat] = 0.0f0
        end
        tape.apical[cell, 1, flat] = 0.0f0
        tape.soma[cell, 1, flat] = 0.0f0
        tape.adaptation[cell, 1, flat] = 0.0f0
    end
    _write_exported_state!(tape, model, 1, flat)

    @inbounds for coordinate in 1:node_dim
        base.workspace[coordinate, 1, flat] = 0.0f0
        base.query_pre[coordinate, flat] = 0.0f0
        base.query[coordinate, flat] = 0.0f0
    end
    sensory_normalization =
        inv(sqrt(Float32(model.sensory_fanin)))
    @inbounds for cycle in 1:model.cycles
        fill!(scratch.branch_inbox, 0.0f0)
        for source in 1:cells
            current = cycle == 1 ? 0.0f0 :
                tape.cell_spikes[source, cycle - 1, flat]
            previous = cycle <= 2 ? 0.0f0 :
                tape.cell_spikes[source, cycle - 2, flat]
            (current == 0.0f0 && previous == 0.0f0) && continue
            for relation in 1:model.fanout
                cache.gate_hard[source, relation] == 0.0f0 &&
                    continue
                delay = cache.delay[source, relation]
                signal = muladd(
                    1.0f0 - delay,
                    current,
                    delay * previous,
                )
                signal == 0.0f0 && continue
                destination =
                    model.destination_for_source[source, relation]
                branch = Int(branch_for_edge[source, relation])
                scratch.branch_inbox[destination, branch] +=
                    parameters.synapse_weight[source, relation] *
                    signal
            end
        end

        for cell in 1:cells
            block = div(cell - 1, model.cells_per_block) + 1
            local_cell =
                cell - (block - 1) * model.cells_per_block
            apical_drive = 0.0f0
            for channel in 1:readout
                coordinate =
                    channel +
                    (local_cell - 1) * readout
                apical_drive = muladd(
                    parameters.feedback_gain[coordinate, block],
                    base.workspace[coordinate, cycle, flat],
                    apical_drive,
                )
            end
            apical_drive /= Float32(readout)
            next_apical = muladd(
                cache.apical_leak[cell],
                tape.apical[cell, cycle, flat],
                apical_drive,
            )
            basal = 0.0f0
            for branch in 1:model.branches
                sensory_exc = parameters.branch_bias[branch, cell]
                sensory_inh = 0.0f0
                if cycle <= model.sensory_cycles
                    for contact in 1:model.sensory_fanin
                        exc_rail = model.excitatory_feature[
                            contact,
                            branch,
                            cell,
                        ]
                        inh_rail = model.inhibitory_feature[
                            contact,
                            branch,
                            cell,
                        ]
                        sensory_exc = muladd(
                            cache.input_exc_gain[
                                contact,
                                branch,
                                cell,
                            ],
                            base.rails[exc_rail, flat] *
                            sensory_normalization,
                            sensory_exc,
                        )
                        sensory_inh = muladd(
                            cache.input_inh_gain[
                                contact,
                                branch,
                                cell,
                            ],
                            base.rails[inh_rail, flat] *
                            sensory_normalization,
                            sensory_inh,
                        )
                    end
                else
                    sensory_exc = 0.0f0
                end
                recurrent = scratch.branch_inbox[cell, branch]
                recurrent_exc = max(recurrent, 0.0f0)
                recurrent_inh = max(-recurrent, 0.0f0)
                exc_drive = recurrent_exc + sensory_exc
                inh_drive = recurrent_inh + sensory_inh
                old_branch = tape.branch_voltage[
                    cell,
                    branch,
                    cycle,
                    flat,
                ]
                old_ampa = tape.ampa[cell, branch, cycle, flat]
                old_nmda = tape.nmda[cell, branch, cycle, flat]
                old_gaba = tape.gaba[cell, branch, cycle, flat]
                old_plateau =
                    tape.plateau[cell, branch, cycle, flat]
                next_ampa = muladd(
                    cache.ampa_decay[branch, cell],
                    old_ampa,
                    exc_drive,
                )
                next_nmda = muladd(
                    cache.nmda_decay[branch, cell],
                    old_nmda,
                    0.72f0 * exc_drive,
                )
                next_gaba = muladd(
                    cache.gaba_decay[branch, cell],
                    old_gaba,
                    inh_drive,
                )
                unblock = sigmoid(
                    cache.nmda_slope[branch, cell] *
                    (
                        old_branch -
                        cache.nmda_half[branch, cell]
                    ),
                )
                excitatory_current =
                    (next_ampa + next_nmda * unblock) *
                    (1.0f0 - old_branch)
                inhibitory_current =
                    next_gaba * (-1.0f0 - old_branch)
                axial_current =
                    cache.axial_gain[branch, cell] *
                    (
                        tape.soma[cell, cycle, flat] -
                        old_branch
                    )
                next_branch = clamp(
                    cache.branch_leak[branch, cell] * old_branch +
                    cache.current_gain[branch, cell] *
                    (excitatory_current + inhibitory_current) +
                    axial_current +
                    cache.plateau_feedback[branch, cell] *
                    old_plateau,
                    -2.0f0,
                    3.0f0,
                )
                argument =
                    cache.plateau_slope[branch, cell] *
                    (
                        next_branch -
                        cache.plateau_threshold[branch, cell]
                    )
                coincidence = _hard_sigmoid(argument)
                next_plateau = clamp(
                    muladd(
                        cache.plateau_decay[branch, cell],
                        old_plateau,
                        cache.plateau_gain[branch, cell] *
                        next_nmda *
                        coincidence,
                    ),
                    0.0f0,
                    4.0f0,
                )
                tape.branch_voltage[
                    cell,
                    branch,
                    cycle + 1,
                    flat,
                ] = next_branch
                tape.ampa[
                    cell,
                    branch,
                    cycle + 1,
                    flat,
                ] = next_ampa
                tape.nmda[
                    cell,
                    branch,
                    cycle + 1,
                    flat,
                ] = next_nmda
                tape.gaba[
                    cell,
                    branch,
                    cycle + 1,
                    flat,
                ] = next_gaba
                tape.plateau[
                    cell,
                    branch,
                    cycle + 1,
                    flat,
                ] = next_plateau
                basal = muladd(
                    parameters.soma_coupling[branch, cell],
                    next_branch + next_plateau,
                    basal,
                )
            end
            modulation =
                1.0f0 +
                cache.apical_gain[cell] *
                _hard_sigmoid(next_apical)
            soma_pre = muladd(
                cache.soma_leak[cell],
                tape.soma[cell, cycle, flat],
                basal * modulation -
                tape.adaptation[cell, cycle, flat],
            )
            spike =
                soma_pre >= cache.soma_threshold[cell] ?
                1.0f0 : 0.0f0
            tape.soma_spikes[cell, cycle, flat] = spike
            tape.apical[cell, cycle + 1, flat] = next_apical
            tape.soma[cell, cycle + 1, flat] =
                soma_pre - spike * cache.soma_threshold[cell]
            tape.adaptation[cell, cycle + 1, flat] = muladd(
                cache.adaptation_decay[cell],
                tape.adaptation[cell, cycle, flat],
                cache.adaptation_gain[cell] * spike,
            )
        end
        _write_exported_state!(tape, model, cycle + 1, flat)

        query_square_sum = 0.0f0
        for coordinate in 1:node_dim
            global_state = 0.0f0
            for block in 1:model.blocks
                node = coordinate + (block - 1) * node_dim
                global_state +=
                    base.membrane[node, cycle + 1, flat]
            end
            global_state /= Float32(model.blocks)
            value = 0.0f0
            for input_coordinate in 1:node_dim
                input_state = 0.0f0
                for block in 1:model.blocks
                    node =
                        input_coordinate +
                        (block - 1) * node_dim
                    input_state +=
                        base.membrane[
                            node,
                            cycle + 1,
                            flat,
                        ]
                end
                input_state /= Float32(model.blocks)
                value = muladd(
                    parameters.state_query_weight[
                        coordinate,
                        input_coordinate,
                    ],
                    input_state,
                    value,
                )
            end
            tape.state_query_pre[coordinate, cycle, flat] =
                value
            query_square_sum = muladd(
                value,
                value,
                query_square_sum,
            )
        end
        query_inv_rms = inv(sqrt(
            query_square_sum / Float32(node_dim) +
            InputModel.RMS_NORM_EPS,
        ))
        tape.state_query_inv_rms[cycle, flat] =
            query_inv_rms
        for coordinate in 1:node_dim
            query = tanh(
                InputModel.QUERY_NORM_SCALE *
                tape.state_query_pre[
                    coordinate,
                    cycle,
                    flat,
                ] *
                query_inv_rms,
            )
            tape.state_query[coordinate, cycle, flat] = query
            base.query_pre[coordinate, flat] =
                tape.state_query_pre[
                    coordinate,
                    cycle,
                    flat,
                ]
            base.query[coordinate, flat] = query
        end
        base.query_inv_rms[flat] = query_inv_rms

        for block in 1:model.blocks
            score = 0.0f0
            magnitude = 0.0f0
            offset = (block - 1) * node_dim
            for coordinate in 1:node_dim
                state = base.membrane[
                    offset + coordinate,
                    cycle + 1,
                    flat,
                ]
                score = muladd(
                    state *
                    parameters.workspace_key[coordinate, block],
                    tape.state_query[
                        coordinate,
                        cycle,
                        flat,
                    ],
                    score,
                )
                magnitude += abs(state)
            end
            scratch.scores[block] = score + 0.05f0 * magnitude
        end
        _prepare_route!(
            scratch,
            model,
            stochastic_routing,
            routing_nonce,
            cycle,
        )
        _record_route!(tape, scratch, model, cycle, flat)

        for cell in 1:cells
            block = div(cell - 1, model.cells_per_block) + 1
            tape.cell_spikes[cell, cycle, flat] =
                tape.soma_spikes[cell, cycle, flat] *
                base.block_mask[block, cycle, flat]
        end
        for coordinate in 1:node_dim
            write = 0.0f0
            for block in 1:model.blocks
                node = coordinate + (block - 1) * node_dim
                write = muladd(
                    base.membrane[
                        node,
                        cycle + 1,
                        flat,
                    ],
                    base.block_mask[block, cycle, flat],
                    write,
                )
            end
            write /= Float32(model.workspace_k)
            base.workspace[coordinate, cycle + 1, flat] = tanh(
                cache.workspace_decay *
                base.workspace[coordinate, cycle, flat] +
                write,
            )
        end
    end

    workspace_square_sum = 0.0f0
    pool_square_sum = 0.0f0
    @inbounds for coordinate in 1:node_dim
        selected_pool = 0.0f0
        for block in 1:model.blocks
            node = coordinate + (block - 1) * node_dim
            selected_pool = muladd(
                base.membrane[
                    node,
                    model.cycles + 1,
                    flat,
                ],
                base.block_mask[block, model.cycles, flat],
                selected_pool,
            )
        end
        selected_pool /= Float32(model.workspace_k)
        workspace_value =
            base.workspace[coordinate, model.cycles + 1, flat]
        scratch.point_scratch.features[coordinate] =
            workspace_value
        scratch.point_scratch.features[node_dim + coordinate] =
            selected_pool
        workspace_square_sum = muladd(
            workspace_value,
            workspace_value,
            workspace_square_sum,
        )
        pool_square_sum = muladd(
            selected_pool,
            selected_pool,
            pool_square_sum,
        )
    end
    workspace_inv_rms = inv(sqrt(
        workspace_square_sum / Float32(node_dim) +
        InputModel.RMS_NORM_EPS,
    ))
    pool_inv_rms = inv(sqrt(
        pool_square_sum / Float32(node_dim) +
        InputModel.RMS_NORM_EPS,
    ))
    base.workspace_inv_rms[flat] = workspace_inv_rms
    base.selected_pool_inv_rms[flat] = pool_inv_rms
    @inbounds for coordinate in 1:node_dim
        scratch.point_scratch.features[coordinate] *=
            workspace_inv_rms
        scratch.point_scratch.features[node_dim + coordinate] *=
            pool_inv_rms
    end
    hidden_square_sum = 0.0f0
    @inbounds for hidden in 1:model.hidden
        activation = parameters.head_bias[hidden]
        for feature in 1:(2 * node_dim)
            activation = muladd(
                parameters.head_weight[hidden, feature],
                scratch.point_scratch.features[feature],
                activation,
            )
        end
        base.hidden_pre[hidden, flat] = activation
        hidden_square_sum = muladd(
            activation,
            activation,
            hidden_square_sum,
        )
    end
    hidden_inv_rms = inv(sqrt(
        hidden_square_sum / Float32(model.hidden) +
        InputModel.RMS_NORM_EPS,
    ))
    base.hidden_inv_rms[flat] = hidden_inv_rms
    @inbounds for hidden in 1:model.hidden
        base.hidden[hidden, flat] = tanh(
            InputModel.HIDDEN_NORM_SCALE *
            base.hidden_pre[hidden, flat] *
            hidden_inv_rms,
        )
    end
    @inbounds for output in 1:OUTPUT_DIM
        value = parameters.output_bias[output]
        for hidden in 1:model.hidden
            value = muladd(
                parameters.output_weight[output, hidden],
                base.hidden[hidden, flat],
                value,
            )
        end
        base.raw[output, flat] = value
    end
    return nothing
end

@inline function _huber_derivative(value::Float32)
    return clamp(value, -1.0f0, 1.0f0)
end

@inline function _huber_loss(value::Float32)
    magnitude = abs(value)
    return magnitude <= 1.0f0 ?
        0.5f0 * value * value :
        magnitude - 0.5f0
end

@inline function _prepare_local_error!(
    scratch::DendriticWorkerScratch,
    base::Point.TrainingArena,
    flat::Int,
    record_metrics::Bool,
)
    fill!(scratch.local_error, 0.0f0)
    state_slot = div(flat - 1, base.width) + 1
    candidate = flat - (state_slot - 1) * base.width
    targets = base.targets
    prediction = scratch.local_prediction
    error = scratch.local_error
    inverse_valid = inv(Float32(max(base.valid_count, 1)))

    q_residual =
        prediction[1] -
        targets.teacher_q[candidate, state_slot]
    error[1] =
        0.25f0 *
        _huber_derivative(q_residual) *
        inverse_valid
    record_metrics &&
        (scratch.local_q_loss += _huber_loss(q_residual))
    if targets.death_mask[candidate, state_slot] != 0.0f0
        death_target =
            targets.death[candidate, state_slot]
        error[2] =
            0.10f0 *
            (
                sigmoid(prediction[2]) -
                death_target
            ) *
            inverse_valid
        record_metrics &&
            (scratch.local_death_loss +=
                max(prediction[2], 0.0f0) -
                death_target * prediction[2] +
                log1p(exp(-abs(prediction[2]))))
    end
    teacher_q = targets.teacher_q[candidate, state_slot]
    @inbounds for quantile in 1:QUANTILES
        output = 2 + quantile
        quantile_error = teacher_q - prediction[output]
        tau =
            (Float32(quantile) - 0.5f0) /
            Float32(QUANTILES)
        negative = quantile_error < 0.0f0 ? 1.0f0 : 0.0f0
        quantile_weight = abs(tau - negative)
        error[output] =
            -0.05f0 *
            quantile_weight *
            _huber_derivative(quantile_error) *
            inverse_valid /
            Float32(QUANTILES)
        record_metrics &&
            (scratch.local_quantile_loss +=
                quantile_weight *
                _huber_loss(quantile_error) /
                Float32(QUANTILES))
    end
    geometry_targets = (
        targets.line_clear[candidate, state_slot] / 4.0f0,
        targets.max_height[candidate, state_slot] / 24.0f0,
        targets.holes[candidate, state_slot] / 240.0f0,
        targets.cavities[candidate, state_slot] / 240.0f0,
    )
    @inbounds for local_index in 1:4
        output = 18 + local_index
        residual =
            prediction[output] -
            geometry_targets[local_index]
        error[output] =
            LOCAL_GEOMETRY_WEIGHT *
            _huber_derivative(residual) *
            inverse_valid
        record_metrics &&
            (scratch.local_geometry_loss +=
                _huber_loss(residual) / 4.0f0)
    end
    return nothing
end

@inline function _signal_coordinate!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    projection::Array{Float32,3},
    model,
    block::Int,
    cycle::Int,
    flat::Int,
    global_signal_scale::Float32,
    local_signal_scale::Float32,
)
    @inbounds for coordinate in 1:model.node_dim
        signal = 0.0f0
        for output in 1:OUTPUT_DIM
            signal = muladd(
                projection[coordinate, output, block],
                global_signal_scale *
                tape.base.raw_gradient[output, flat] +
                local_signal_scale *
                scratch.local_error[output],
                signal,
            )
        end
        signal /= Float32(model.cycles)
        scratch.block_signal[coordinate] = signal
    end
    return nothing
end

function _prepare_block_signals!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    projection::Array{Float32,3},
    model,
    flat::Int,
    record_metrics::Bool,
    global_signal_scale::Float32,
    local_signal_scale::Float32,
)
    base = tape.base
    fill!(scratch.soma_signal, 0.0f0)
    fill!(scratch.apical_signal, 0.0f0)
    fill!(scratch.branch_signal, 0.0f0)
    inverse_node_dim = inv(Float32(model.node_dim))

    @inbounds for cycle in 1:model.cycles
        for block in 1:model.blocks
            fill!(scratch.local_prediction, 0.0f0)
            offset = (block - 1) * model.node_dim
            for output in 1:OUTPUT_DIM
                prediction = 0.0f0
                for coordinate in 1:model.node_dim
                    state = base.membrane[
                        offset + coordinate,
                        cycle + 1,
                        flat,
                    ]
                    cell = _cell_for_coordinate(
                        model,
                        coordinate,
                        block,
                    )
                    spike =
                        tape.cell_spikes[cell, cycle, flat]
                    predictor_state =
                        state +
                        LOCAL_PREDICTOR_SPIKE_SCALE * spike
                    prediction = muladd(
                        projection[coordinate, output, block],
                        predictor_state,
                        prediction,
                    )
                end
                scratch.local_prediction[output] = prediction
            end
            _prepare_local_error!(
                scratch,
                base,
                flat,
                record_metrics,
            )
            _signal_coordinate!(
                scratch,
                tape,
                projection,
                model,
                block,
                cycle,
                flat,
                global_signal_scale,
                local_signal_scale,
            )

            advantage = 0.0f0
            for coordinate in 1:model.node_dim
                state = base.membrane[
                    offset + coordinate,
                    cycle + 1,
                    flat,
                ]
                signal = scratch.block_signal[coordinate]
                advantage = muladd(
                    signal,
                    state,
                    advantage,
                )
                cell = _cell_for_coordinate(
                    model,
                    coordinate,
                    block,
                )
                channel =
                    _channel_for_coordinate(model, coordinate)
                derivative = 1.0f0 - state * state
                local_signal = signal * derivative
                if channel == 1
                    scratch.soma_signal[cell, cycle] =
                        local_signal
                elseif channel == 2
                    scratch.apical_signal[cell, cycle] =
                        local_signal
                else
                    scratch.branch_signal[
                        cell,
                        channel - 2,
                        cycle,
                    ] = local_signal
                end
            end
            tape.block_supervised_reward[
                block,
                cycle,
                flat,
            ] = -advantage * inverse_node_dim
        end
    end
    return nothing
end

@inline function _accumulate_routing_gradients!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    parameters,
    flat::Int,
)
    base = tape.base
    @inbounds for cycle in 1:model.cycles
        fill!(scratch.point_scratch.dquery, 0.0f0)
        for block in 1:model.blocks
            # block_advantage is a candidate-centered supervised reward
            # surrogate, not an environment return.  AdamW consumes a loss
            # gradient, so the policy-gradient contribution carries a minus.
            route_factor =
                -tape.block_advantage[
                    block,
                    cycle,
                    flat,
                ] *
                base.route_eligibility[
                    block,
                    cycle,
                    flat,
                ] +
                base.route_regularizer_gradient[
                    block,
                    cycle,
                    flat,
                ]
            offset = (block - 1) * model.node_dim
            for coordinate in 1:model.node_dim
                state = base.membrane[
                    offset + coordinate,
                    cycle + 1,
                    flat,
                ]
                query = tape.state_query[
                    coordinate,
                    cycle,
                    flat,
                ]
                key = parameters.workspace_key[
                    coordinate,
                    block,
                ]
                scratch.gradient.workspace_key[
                    coordinate,
                    block,
                ] = muladd(
                    route_factor * state,
                    query,
                    scratch.gradient.workspace_key[
                        coordinate,
                        block,
                    ],
                )
                scratch.point_scratch.dquery[coordinate] = muladd(
                    route_factor * state,
                    key,
                    scratch.point_scratch.dquery[coordinate],
                )
            end
        end

        query_projection_mean = 0.0f0
        for coordinate in 1:model.node_dim
            query = tape.state_query[
                coordinate,
                cycle,
                flat,
            ]
            normalized =
                scratch.point_scratch.dquery[coordinate] *
                InputModel.QUERY_NORM_SCALE *
                (1.0f0 - query * query)
            scratch.point_scratch.dquery[coordinate] = normalized
            query_projection_mean = muladd(
                normalized,
                tape.state_query_pre[
                    coordinate,
                    cycle,
                    flat,
                ],
                query_projection_mean,
            )
        end
        query_projection_mean /= Float32(model.node_dim)
        inverse_rms = tape.state_query_inv_rms[cycle, flat]
        inverse_rms_squared = inverse_rms * inverse_rms
        for coordinate in 1:model.node_dim
            cotangent = inverse_rms * (
                scratch.point_scratch.dquery[coordinate] -
                tape.state_query_pre[
                    coordinate,
                    cycle,
                    flat,
                ] *
                inverse_rms_squared *
                query_projection_mean
            )
            for input_coordinate in 1:model.node_dim
                global_state = 0.0f0
                for block in 1:model.blocks
                    node =
                        input_coordinate +
                        (block - 1) * model.node_dim
                    global_state += base.membrane[
                        node,
                        cycle + 1,
                        flat,
                    ]
                end
                global_state /= Float32(model.blocks)
                scratch.gradient.state_query_weight[
                    coordinate,
                    input_coordinate,
                ] = muladd(
                    cotangent,
                    global_state,
                    scratch.gradient.state_query_weight[
                        coordinate,
                        input_coordinate,
                    ],
                )
            end
        end
    end
    return nothing
end

@inline function _accumulate_cell_parameter_gradients!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    parameters,
    cache::DendriticParameterCache,
    flat::Int,
)
    base = tape.base
    readout = model.readout_per_cell
    cells = model.blocks * model.cells_per_block
    sensory_normalization =
        inv(sqrt(Float32(model.sensory_fanin)))
    @inbounds for cycle in 1:model.cycles
        fill!(scratch.point_scratch.dworkspace_a, 0.0f0)
        for cell in 1:cells
            block = div(cell - 1, model.cells_per_block) + 1
            local_cell =
                cell - (block - 1) * model.cells_per_block
            next_apical =
                tape.apical[cell, cycle + 1, flat]
            modulation =
                1.0f0 +
                cache.apical_gain[cell] *
                _hard_sigmoid(next_apical)
            basal = 0.0f0
            for branch in 1:model.branches
                basal = muladd(
                    parameters.soma_coupling[branch, cell],
                    tape.branch_voltage[
                        cell,
                        branch,
                        cycle + 1,
                        flat,
                    ] +
                    tape.plateau[
                        cell,
                        branch,
                        cycle + 1,
                        flat,
                    ],
                    basal,
                )
            end
            soma_before_reset =
                tape.soma[cell, cycle + 1, flat] +
                tape.soma_spikes[cell, cycle, flat] *
                cache.soma_threshold[cell]
            post_surrogate = _spike_surrogate(
                soma_before_reset,
                cache.soma_threshold[cell],
                model.spike_temperature,
            )
            reset_factor =
                1.0f0 -
                cache.soma_threshold[cell] * post_surrogate
            soma_pre_signal =
                scratch.soma_signal[cell, cycle] *
                reset_factor
            apical_argument = next_apical
            apical_signal =
                scratch.apical_signal[cell, cycle] +
                soma_pre_signal *
                basal *
                cache.apical_gain[cell] *
                _hard_sigmoid_derivative(apical_argument)

            scratch.gradient.apical_leak_logits[cell] +=
                apical_signal *
                tape.apical[cell, cycle, flat] *
                cache.apical_leak_derivative[cell]
            scratch.gradient.soma_leak_logits[cell] +=
                soma_pre_signal *
                tape.soma[cell, cycle, flat] *
                cache.soma_leak_derivative[cell]
            scratch.gradient.apical_gain_logits[cell] +=
                soma_pre_signal *
                basal *
                _hard_sigmoid(next_apical) *
                cache.apical_gain_derivative[cell]
            scratch.gradient.soma_threshold_logits[cell] +=
                scratch.soma_signal[cell, cycle] *
                (
                    -tape.soma_spikes[cell, cycle, flat] +
                    cache.soma_threshold[cell] * post_surrogate
                ) *
                cache.soma_threshold_derivative[cell]

            if cycle >= 2
                # q(t) is produced on the preceding cycle and subtracts
                # directly from the present soma.  This one-step local trace
                # gives both adaptation parameters a causal learning signal
                # without traversing the global graph backwards.
                adaptation_effect = -soma_pre_signal
                scratch.gradient.adaptation_decay_logits[cell] +=
                    adaptation_effect *
                    tape.adaptation[cell, cycle - 1, flat] *
                    cache.adaptation_decay_derivative[cell]
                scratch.gradient.adaptation_gain_logits[cell] +=
                    adaptation_effect *
                    tape.soma_spikes[cell, cycle - 1, flat] *
                    cache.adaptation_gain_derivative[cell]
            end

            for channel in 1:readout
                coordinate =
                    channel +
                    (local_cell - 1) * readout
                feedback =
                    parameters.feedback_gain[coordinate, block]
                scratch.gradient.feedback_gain[
                    coordinate,
                    block,
                ] = muladd(
                    apical_signal / Float32(readout),
                    base.workspace[
                        coordinate,
                        cycle,
                        flat,
                    ],
                    scratch.gradient.feedback_gain[
                        coordinate,
                        block,
                    ],
                )
                scratch.point_scratch.dworkspace_a[coordinate] =
                    muladd(
                        apical_signal *
                        feedback /
                        Float32(readout),
                        1.0f0,
                        scratch.point_scratch.dworkspace_a[coordinate],
                    )
            end

            for branch in 1:model.branches
                old_branch = tape.branch_voltage[
                    cell,
                    branch,
                    cycle,
                    flat,
                ]
                next_branch = tape.branch_voltage[
                    cell,
                    branch,
                    cycle + 1,
                    flat,
                ]
                old_plateau = tape.plateau[
                    cell,
                    branch,
                    cycle,
                    flat,
                ]
                next_plateau = tape.plateau[
                    cell,
                    branch,
                    cycle + 1,
                    flat,
                ]
                old_ampa = tape.ampa[
                    cell,
                    branch,
                    cycle,
                    flat,
                ]
                old_nmda = tape.nmda[
                    cell,
                    branch,
                    cycle,
                    flat,
                ]
                old_gaba = tape.gaba[
                    cell,
                    branch,
                    cycle,
                    flat,
                ]
                next_ampa = tape.ampa[
                    cell,
                    branch,
                    cycle + 1,
                    flat,
                ]
                next_nmda = tape.nmda[
                    cell,
                    branch,
                    cycle + 1,
                    flat,
                ]
                next_gaba = tape.gaba[
                    cell,
                    branch,
                    cycle + 1,
                    flat,
                ]
                unblock = sigmoid(
                    cache.nmda_slope[branch, cell] *
                    (
                        old_branch -
                        cache.nmda_half[branch, cell]
                    ),
                )
                excitatory_current =
                    (next_ampa + next_nmda * unblock) *
                    (1.0f0 - old_branch)
                inhibitory_current =
                    next_gaba * (-1.0f0 - old_branch)
                raw_branch =
                    cache.branch_leak[branch, cell] * old_branch +
                    cache.current_gain[branch, cell] *
                    (excitatory_current + inhibitory_current) +
                    cache.axial_gain[branch, cell] *
                    (
                        tape.soma[cell, cycle, flat] -
                        old_branch
                    ) +
                    cache.plateau_feedback[branch, cell] *
                    old_plateau
                branch_clamp_derivative =
                    -2.0f0 < raw_branch < 3.0f0 ? 1.0f0 : 0.0f0
                argument =
                    cache.plateau_slope[branch, cell] *
                    (
                        next_branch -
                        cache.plateau_threshold[branch, cell]
                    )
                coincidence = _hard_sigmoid(argument)
                hard_derivative =
                    _hard_sigmoid_derivative(argument)
                recruited = next_nmda * coincidence
                raw_plateau =
                    cache.plateau_decay[branch, cell] *
                    old_plateau +
                    cache.plateau_gain[branch, cell] *
                    recruited
                plateau_clamp_derivative =
                    0.0f0 < raw_plateau < 4.0f0 ?
                    1.0f0 : 0.0f0
                basal_effect =
                    soma_pre_signal *
                    parameters.soma_coupling[branch, cell] *
                    modulation
                plateau_effect =
                    basal_effect * plateau_clamp_derivative
                branch_effect =
                    scratch.branch_signal[cell, branch, cycle] +
                    basal_effect +
                    plateau_effect *
                    cache.plateau_gain[branch, cell] *
                    next_nmda *
                    hard_derivative *
                    cache.plateau_slope[branch, cell]
                branch_pre_effect =
                    branch_effect * branch_clamp_derivative
                nmda_effect =
                    plateau_effect *
                    cache.plateau_gain[branch, cell] *
                    coincidence
                current_effect =
                    branch_pre_effect *
                    cache.current_gain[branch, cell]
                ampa_effect =
                    current_effect *
                    (1.0f0 - old_branch)
                nmda_effect +=
                    current_effect *
                    unblock *
                    (1.0f0 - old_branch)
                gaba_effect =
                    current_effect *
                    (-1.0f0 - old_branch)

                scratch.gradient.branch_leak_logits[branch, cell] +=
                    branch_pre_effect *
                    old_branch *
                    cache.branch_leak_derivative[branch, cell]
                scratch.gradient.ampa_decay_logits[
                    branch,
                    cell,
                ] +=
                    ampa_effect *
                    old_ampa *
                    cache.ampa_decay_derivative[branch, cell]
                scratch.gradient.nmda_decay_logits[
                    branch,
                    cell,
                ] +=
                    nmda_effect *
                    old_nmda *
                    cache.nmda_decay_derivative[branch, cell]
                scratch.gradient.gaba_decay_logits[
                    branch,
                    cell,
                ] +=
                    gaba_effect *
                    old_gaba *
                    cache.gaba_decay_derivative[branch, cell]
                scratch.gradient.current_gain_logits[
                    branch,
                    cell,
                ] +=
                    branch_pre_effect *
                    (excitatory_current + inhibitory_current) *
                    cache.current_gain_derivative[branch, cell]
                scratch.gradient.axial_gain_logits[
                    branch,
                    cell,
                ] +=
                    branch_pre_effect *
                    (
                        tape.soma[cell, cycle, flat] -
                        old_branch
                    ) *
                    cache.axial_gain_derivative[branch, cell]
                unblock_effect =
                    current_effect *
                    next_nmda *
                    (1.0f0 - old_branch)
                unblock_derivative =
                    unblock * (1.0f0 - unblock)
                scratch.gradient.nmda_slope_logits[
                    branch,
                    cell,
                ] +=
                    unblock_effect *
                    unblock_derivative *
                    (
                        old_branch -
                        cache.nmda_half[branch, cell]
                    ) *
                    cache.nmda_slope_derivative[branch, cell]
                scratch.gradient.nmda_half_logits[
                    branch,
                    cell,
                ] +=
                    -unblock_effect *
                    unblock_derivative *
                    cache.nmda_slope[branch, cell] *
                    cache.nmda_half_derivative[branch, cell]
                scratch.gradient.plateau_feedback_logits[
                    branch,
                    cell,
                ] +=
                    branch_pre_effect *
                    old_plateau *
                    cache.plateau_feedback_derivative[branch, cell]
                scratch.gradient.plateau_decay_logits[
                    branch,
                    cell,
                ] +=
                    plateau_effect *
                    old_plateau *
                    cache.plateau_decay_derivative[branch, cell]
                scratch.gradient.plateau_gain_logits[
                    branch,
                    cell,
                ] +=
                    plateau_effect *
                    recruited *
                    cache.plateau_gain_derivative[branch, cell]
                scratch.gradient.plateau_threshold_logits[
                    branch,
                    cell,
                ] +=
                    plateau_effect *
                    cache.plateau_gain[branch, cell] *
                    next_nmda *
                    hard_derivative *
                    (-cache.plateau_slope[branch, cell]) *
                    cache.plateau_threshold_derivative[branch, cell]
                scratch.gradient.plateau_slope_logits[
                    branch,
                    cell,
                ] +=
                    plateau_effect *
                    cache.plateau_gain[branch, cell] *
                    next_nmda *
                    hard_derivative *
                    (
                        next_branch -
                        cache.plateau_threshold[branch, cell]
                    ) *
                    cache.plateau_slope_derivative[branch, cell]
                scratch.gradient.soma_coupling[branch, cell] +=
                    soma_pre_signal *
                    (next_branch + next_plateau) *
                    modulation

                if cycle <= model.sensory_cycles
                    exc_drive_effect =
                        ampa_effect + 0.72f0 * nmda_effect
                    scratch.gradient.branch_bias[branch, cell] +=
                        exc_drive_effect
                    for contact in 1:model.sensory_fanin
                        exc_rail = model.excitatory_feature[
                            contact,
                            branch,
                            cell,
                        ]
                        inh_rail = model.inhibitory_feature[
                            contact,
                            branch,
                            cell,
                        ]
                        scratch.gradient.input_exc_logits[
                            contact,
                            branch,
                            cell,
                        ] = muladd(
                            exc_drive_effect *
                            cache.input_exc_derivative[
                                contact,
                                branch,
                                cell,
                            ],
                            base.rails[exc_rail, flat] *
                            sensory_normalization,
                            scratch.gradient.input_exc_logits[
                                contact,
                                branch,
                                cell,
                            ],
                        )
                        scratch.gradient.input_inh_logits[
                            contact,
                            branch,
                            cell,
                        ] = muladd(
                            gaba_effect *
                            cache.input_inh_derivative[
                                contact,
                                branch,
                                cell,
                            ],
                            base.rails[inh_rail, flat] *
                            sensory_normalization,
                            scratch.gradient.input_inh_logits[
                                contact,
                                branch,
                                cell,
                            ],
                        )
                    end
                end
            end
        end
        if cycle >= 2
            for coordinate in 1:model.node_dim
                workspace =
                    base.workspace[coordinate, cycle, flat]
                workspace_decay_signal = (
                    scratch.point_scratch.dworkspace_a[coordinate] *
                    (1.0f0 - workspace * workspace) *
                    base.workspace[coordinate, cycle - 1, flat]
                )
                scratch.gradient.workspace_decay_logit[1] +=
                    workspace_decay_signal *
                    cache.workspace_decay_derivative
            end
        end
    end
    return nothing
end

@inline function _update_edge_trace(
    epsilon_a::Float32,
    epsilon_n::Float32,
    epsilon_g::Float32,
    epsilon_u::Float32,
    epsilon_p::Float32,
    epsilon_s::Float32,
    epsilon_q::Float32,
    forcing::Float32,
    recurrent_drive::Float32,
    old_branch::Float32,
    next_branch::Float32,
    old_ampa::Float32,
    next_ampa::Float32,
    old_nmda::Float32,
    next_nmda::Float32,
    old_gaba::Float32,
    next_gaba::Float32,
    old_plateau::Float32,
    next_plateau::Float32,
    branch::Int,
    destination::Int,
    apical_modulation::Float32,
    post_surrogate::Float32,
    parameters,
    cache::DendriticParameterCache,
)
    excitatory_forcing =
        recurrent_drive > 0.0f0 ? forcing : 0.0f0
    inhibitory_forcing =
        recurrent_drive < 0.0f0 ? -forcing : 0.0f0
    next_a = muladd(
        cache.ampa_decay[branch, destination],
        epsilon_a,
        excitatory_forcing,
    )
    next_n = muladd(
        cache.nmda_decay[branch, destination],
        epsilon_n,
        0.72f0 * excitatory_forcing,
    )
    next_g = muladd(
        cache.gaba_decay[branch, destination],
        epsilon_g,
        inhibitory_forcing,
    )
    unblock = sigmoid(
        cache.nmda_slope[branch, destination] *
        (
            old_branch -
            cache.nmda_half[branch, destination]
        ),
    )
    unblock_trace =
        unblock *
        (1.0f0 - unblock) *
        cache.nmda_slope[branch, destination] *
        epsilon_u
    excitatory_trace = (
        next_a +
        next_n * unblock +
        next_nmda * unblock_trace
    ) * (1.0f0 - old_branch) -
        (next_ampa + next_nmda * unblock) * epsilon_u
    inhibitory_trace =
        next_g * (-1.0f0 - old_branch) -
        next_gaba * epsilon_u
    raw_branch_trace =
        cache.branch_leak[branch, destination] * epsilon_u +
        cache.current_gain[branch, destination] *
        (excitatory_trace + inhibitory_trace) +
        cache.axial_gain[branch, destination] *
        (epsilon_s - epsilon_u) +
        cache.plateau_feedback[branch, destination] * epsilon_p
    branch_clamp_derivative =
        -2.0f0 < next_branch < 3.0f0 ? 1.0f0 : 0.0f0
    next_u = raw_branch_trace * branch_clamp_derivative
    argument =
        cache.plateau_slope[branch, destination] *
        (
            next_branch -
            cache.plateau_threshold[branch, destination]
        )
    coincidence = _hard_sigmoid(argument)
    coincidence_derivative =
        _hard_sigmoid_derivative(argument) *
        cache.plateau_slope[branch, destination] *
        next_u
    raw_plateau_trace = muladd(
        cache.plateau_decay[branch, destination],
        epsilon_p,
        cache.plateau_gain[branch, destination] *
        (
            next_n * coincidence +
            next_nmda * coincidence_derivative
        ),
    )
    plateau_clamp_derivative =
        0.0f0 < next_plateau < 4.0f0 ? 1.0f0 : 0.0f0
    next_p = raw_plateau_trace * plateau_clamp_derivative
    soma_pre_trace = muladd(
        cache.soma_leak[destination],
        epsilon_s,
        parameters.soma_coupling[branch, destination] *
        (next_u + next_p) *
        apical_modulation -
        epsilon_q,
    )
    spike_trace = post_surrogate * soma_pre_trace
    next_s = soma_pre_trace *
        1.0f0 -
        cache.soma_threshold[destination] * spike_trace
    next_q = muladd(
        cache.adaptation_decay[destination],
        epsilon_q,
        cache.adaptation_gain[destination] * spike_trace,
    )
    return next_a, next_n, next_g, next_u, next_p, next_s, next_q
end

function _accumulate_edge_eligibility!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    parameters,
    cache::DendriticParameterCache,
    branch_for_edge::Matrix{UInt8},
    flat::Int,
)
    @inbounds for active_index in 1:scratch.active_edge_count
        scratch.active_edge_mask[
            scratch.active_edges[active_index]
        ] = false
    end
    scratch.active_edge_count = 0
    fill!(scratch.eligibility_weight_a, 0.0f0)
    fill!(scratch.eligibility_weight_n, 0.0f0)
    fill!(scratch.eligibility_weight_g, 0.0f0)
    fill!(scratch.eligibility_weight_u, 0.0f0)
    fill!(scratch.eligibility_weight_p, 0.0f0)
    fill!(scratch.eligibility_weight_s, 0.0f0)
    fill!(scratch.eligibility_weight_q, 0.0f0)
    fill!(scratch.eligibility_gate_a, 0.0f0)
    fill!(scratch.eligibility_gate_n, 0.0f0)
    fill!(scratch.eligibility_gate_g, 0.0f0)
    fill!(scratch.eligibility_gate_u, 0.0f0)
    fill!(scratch.eligibility_gate_p, 0.0f0)
    fill!(scratch.eligibility_gate_s, 0.0f0)
    fill!(scratch.eligibility_gate_q, 0.0f0)
    fill!(scratch.eligibility_delay_a, 0.0f0)
    fill!(scratch.eligibility_delay_n, 0.0f0)
    fill!(scratch.eligibility_delay_g, 0.0f0)
    fill!(scratch.eligibility_delay_u, 0.0f0)
    fill!(scratch.eligibility_delay_p, 0.0f0)
    fill!(scratch.eligibility_delay_s, 0.0f0)
    fill!(scratch.eligibility_delay_q, 0.0f0)
    cells = model.blocks * model.cells_per_block
    @inbounds for cycle in 1:model.cycles
        fill!(scratch.branch_inbox, 0.0f0)
        for source in 1:cells
            current = cycle == 1 ? 0.0f0 :
                tape.cell_spikes[source, cycle - 1, flat]
            previous = cycle <= 2 ? 0.0f0 :
                tape.cell_spikes[source, cycle - 2, flat]
            if current != 0.0f0 || previous != 0.0f0
                for relation in 1:model.fanout
                    edge = source + (relation - 1) * cells
                    if !scratch.active_edge_mask[edge]
                        scratch.active_edge_count += 1
                        scratch.active_edges[
                            scratch.active_edge_count
                        ] = Int32(edge)
                        scratch.active_edge_mask[edge] = true
                    end
                    gate = cache.gate_hard[source, relation]
                    gate == 0.0f0 && continue
                    delay = cache.delay[source, relation]
                    pre = muladd(
                        1.0f0 - delay,
                        current,
                        delay * previous,
                    )
                    pre == 0.0f0 && continue
                    destination =
                        model.destination_for_source[
                            source,
                            relation,
                        ]
                    branch =
                        Int(branch_for_edge[source, relation])
                    scratch.branch_inbox[destination, branch] +=
                        parameters.synapse_weight[
                            source,
                            relation,
                        ] *
                        pre
                end
            end
        end
        for active_index in 1:scratch.active_edge_count
            edge = Int(scratch.active_edges[active_index])
            source = mod1(edge, cells)
            relation = (edge - 1) ÷ cells + 1
            current = cycle == 1 ? 0.0f0 :
                tape.cell_spikes[source, cycle - 1, flat]
            previous = cycle <= 2 ? 0.0f0 :
                tape.cell_spikes[source, cycle - 2, flat]
            destination =
                model.destination_for_source[source, relation]
            branch = Int(branch_for_edge[source, relation])
            delay = cache.delay[source, relation]
            pre = muladd(
                1.0f0 - delay,
                current,
                delay * previous,
            )
            weight = parameters.synapse_weight[source, relation]
            gate = cache.gate_hard[source, relation]
            force_weight = gate * pre
            force_gate =
                weight *
                cache.gate_derivative[source, relation] *
                pre
            force_delay =
                weight *
                gate *
                (previous - current) *
                cache.delay_derivative[source, relation]
            recurrent_drive =
                scratch.branch_inbox[destination, branch]
            old_branch = tape.branch_voltage[
                destination,
                branch,
                cycle,
                flat,
            ]
            next_branch = tape.branch_voltage[
                destination,
                branch,
                cycle + 1,
                flat,
            ]
            old_ampa = tape.ampa[
                destination,
                branch,
                cycle,
                flat,
            ]
            next_ampa = tape.ampa[
                destination,
                branch,
                cycle + 1,
                flat,
            ]
            old_nmda = tape.nmda[
                destination,
                branch,
                cycle,
                flat,
            ]
            next_nmda = tape.nmda[
                destination,
                branch,
                cycle + 1,
                flat,
            ]
            old_gaba = tape.gaba[
                destination,
                branch,
                cycle,
                flat,
            ]
            next_gaba = tape.gaba[
                destination,
                branch,
                cycle + 1,
                flat,
            ]
            old_plateau = tape.plateau[
                destination,
                branch,
                cycle,
                flat,
            ]
            next_plateau = tape.plateau[
                destination,
                branch,
                cycle + 1,
                flat,
            ]
            next_apical =
                tape.apical[destination, cycle + 1, flat]
            modulation =
                1.0f0 +
                cache.apical_gain[destination] *
                _hard_sigmoid(next_apical)
            soma_before_reset =
                tape.soma[destination, cycle + 1, flat] +
                tape.soma_spikes[destination, cycle, flat] *
                cache.soma_threshold[destination]
            post_surrogate = _spike_surrogate(
                soma_before_reset,
                cache.soma_threshold[destination],
                model.spike_temperature,
            )
            wa, wn, wg, wu, wp, ws, wq = _update_edge_trace(
                scratch.eligibility_weight_a[source, relation],
                scratch.eligibility_weight_n[source, relation],
                scratch.eligibility_weight_g[source, relation],
                scratch.eligibility_weight_u[source, relation],
                scratch.eligibility_weight_p[source, relation],
                scratch.eligibility_weight_s[source, relation],
                scratch.eligibility_weight_q[source, relation],
                force_weight,
                recurrent_drive,
                old_branch,
                next_branch,
                old_ampa,
                next_ampa,
                old_nmda,
                next_nmda,
                old_gaba,
                next_gaba,
                old_plateau,
                next_plateau,
                branch,
                destination,
                modulation,
                post_surrogate,
                parameters,
                cache,
            )
            ga, gn, gg, gu, gp, gs, gq = _update_edge_trace(
                scratch.eligibility_gate_a[source, relation],
                scratch.eligibility_gate_n[source, relation],
                scratch.eligibility_gate_g[source, relation],
                scratch.eligibility_gate_u[source, relation],
                scratch.eligibility_gate_p[source, relation],
                scratch.eligibility_gate_s[source, relation],
                scratch.eligibility_gate_q[source, relation],
                force_gate,
                recurrent_drive,
                old_branch,
                next_branch,
                old_ampa,
                next_ampa,
                old_nmda,
                next_nmda,
                old_gaba,
                next_gaba,
                old_plateau,
                next_plateau,
                branch,
                destination,
                modulation,
                post_surrogate,
                parameters,
                cache,
            )
            da, dn, dg, du, dp, ds, dq = _update_edge_trace(
                scratch.eligibility_delay_a[source, relation],
                scratch.eligibility_delay_n[source, relation],
                scratch.eligibility_delay_g[source, relation],
                scratch.eligibility_delay_u[source, relation],
                scratch.eligibility_delay_p[source, relation],
                scratch.eligibility_delay_s[source, relation],
                scratch.eligibility_delay_q[source, relation],
                force_delay,
                recurrent_drive,
                old_branch,
                next_branch,
                old_ampa,
                next_ampa,
                old_nmda,
                next_nmda,
                old_gaba,
                next_gaba,
                old_plateau,
                next_plateau,
                branch,
                destination,
                modulation,
                post_surrogate,
                parameters,
                cache,
            )
            scratch.eligibility_weight_a[source, relation] = wa
            scratch.eligibility_weight_n[source, relation] = wn
            scratch.eligibility_weight_g[source, relation] = wg
            scratch.eligibility_weight_u[source, relation] = wu
            scratch.eligibility_weight_p[source, relation] = wp
            scratch.eligibility_weight_s[source, relation] = ws
            scratch.eligibility_weight_q[source, relation] = wq
            scratch.eligibility_gate_a[source, relation] = ga
            scratch.eligibility_gate_n[source, relation] = gn
            scratch.eligibility_gate_g[source, relation] = gg
            scratch.eligibility_gate_u[source, relation] = gu
            scratch.eligibility_gate_p[source, relation] = gp
            scratch.eligibility_gate_s[source, relation] = gs
            scratch.eligibility_gate_q[source, relation] = gq
            scratch.eligibility_delay_a[source, relation] = da
            scratch.eligibility_delay_n[source, relation] = dn
            scratch.eligibility_delay_g[source, relation] = dg
            scratch.eligibility_delay_u[source, relation] = du
            scratch.eligibility_delay_p[source, relation] = dp
            scratch.eligibility_delay_s[source, relation] = ds
            scratch.eligibility_delay_q[source, relation] = dq
            soma_signal =
                scratch.soma_signal[destination, cycle]
            branch_signal =
                scratch.branch_signal[
                    destination,
                    branch,
                    cycle,
                ]
            weight_update =
                branch_signal * wu + soma_signal * ws
            gate_update =
                branch_signal * gu + soma_signal * gs
            delay_update =
                branch_signal * du + soma_signal * ds
            scratch.gradient.synapse_weight[source, relation] +=
                weight_update
            scratch.gradient.gate_logits[source, relation] +=
                gate_update
            scratch.gradient.delay_logits[source, relation] +=
                delay_update
            scratch.utility[source, relation] += if gate != 0.0f0
                abs(weight_update) +
                abs(gate_update) +
                abs(delay_update)
            else
                # For an OFF edge, a negative gate loss-gradient means that
                # increasing its gate would improve the supervised objective.
                max(-gate_update, 0.0f0)
            end
            for counterfactual_branch in 1:model.branches
                scratch.branch_utility[
                    counterfactual_branch,
                    source,
                    relation,
                ] += abs(
                    pre *
                    scratch.branch_signal[
                        destination,
                        counterfactual_branch,
                        cycle,
                    ],
                )
            end
        end
    end
    return nothing
end

function _backward_head_candidate!(
    gradient,
    base::Point.TrainingArena,
    model,
    parameters,
    scratch::Point.CandidateScratch,
    flat::Int,
)
    node_dim = model.node_dim
    fill!(scratch.dfeatures, 0.0f0)
    fill!(scratch.dhidden, 0.0f0)
    @inbounds for coordinate in 1:node_dim
        selected_pool = 0.0f0
        for block in 1:model.blocks
            node = coordinate + (block - 1) * node_dim
            selected_pool = muladd(
                base.membrane[
                    node,
                    model.cycles + 1,
                    flat,
                ],
                base.block_mask[block, model.cycles, flat],
                selected_pool,
            )
        end
        scratch.features[coordinate] =
            base.workspace[
                coordinate,
                model.cycles + 1,
                flat,
            ] * base.workspace_inv_rms[flat]
        scratch.features[node_dim + coordinate] =
            selected_pool /
            Float32(model.workspace_k) *
            base.selected_pool_inv_rms[flat]
    end
    @inbounds for output in 1:OUTPUT_DIM
        cotangent = base.raw_gradient[output, flat]
        gradient.output_bias[output] += cotangent
        for hidden in 1:model.hidden
            gradient.output_weight[output, hidden] = muladd(
                cotangent,
                base.hidden[hidden, flat],
                gradient.output_weight[output, hidden],
            )
            scratch.dhidden[hidden] = muladd(
                parameters.output_weight[output, hidden],
                cotangent,
                scratch.dhidden[hidden],
            )
        end
    end
    projection_mean = 0.0f0
    @inbounds for hidden in 1:model.hidden
        hidden_value = base.hidden[hidden, flat]
        normalized =
            scratch.dhidden[hidden] *
            InputModel.HIDDEN_NORM_SCALE *
            (1.0f0 - hidden_value * hidden_value)
        scratch.dhidden[hidden] = normalized
        projection_mean = muladd(
            normalized,
            base.hidden_pre[hidden, flat],
            projection_mean,
        )
    end
    projection_mean /= Float32(model.hidden)
    inverse_rms = base.hidden_inv_rms[flat]
    inverse_rms_squared = inverse_rms * inverse_rms
    @inbounds for hidden in 1:model.hidden
        cotangent = inverse_rms * (
            scratch.dhidden[hidden] -
            base.hidden_pre[hidden, flat] *
            inverse_rms_squared *
            projection_mean
        )
        gradient.head_bias[hidden] += cotangent
        for feature in 1:(2 * node_dim)
            value = scratch.features[feature]
            gradient.head_weight[hidden, feature] = muladd(
                cotangent,
                value,
                gradient.head_weight[hidden, feature],
            )
            scratch.dfeatures[feature] = muladd(
                parameters.head_weight[hidden, feature],
                cotangent,
                scratch.dfeatures[feature],
            )
        end
    end
    return nothing
end

function dendritic_prepare_signal_candidate!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    projection::Array{Float32,3},
    model,
    flat::Int,
    global_signal_scale::Float32,
    local_signal_scale::Float32,
)
    _prepare_block_signals!(
        scratch,
        tape,
        projection,
        model,
        flat,
        true,
        global_signal_scale,
        local_signal_scale,
    )
    return nothing
end

function dendritic_local_candidate!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    projection::Array{Float32,3},
    model,
    parameters,
    cache::DendriticParameterCache,
    branch_for_edge::Matrix{UInt8},
    flat::Int,
    global_signal_scale::Float32,
    local_signal_scale::Float32,
)
    _prepare_block_signals!(
        scratch,
        tape,
        projection,
        model,
        flat,
        false,
        global_signal_scale,
        local_signal_scale,
    )
    _accumulate_routing_gradients!(
        scratch,
        tape,
        model,
        parameters,
        flat,
    )
    _accumulate_cell_parameter_gradients!(
        scratch,
        tape,
        model,
        parameters,
        cache,
        flat,
    )
    _accumulate_edge_eligibility!(
        scratch,
        tape,
        model,
        parameters,
        cache,
        branch_for_edge,
        flat,
    )
    _backward_head_candidate!(
        scratch.gradient,
        tape.base,
        model,
        parameters,
        scratch.point_scratch,
        flat,
    )
    return nothing
end

struct DendriticParameterShard
    field::UInt8
    first::Int32
    last::Int32
end

function _parameter_shards(
    parameters;
    elements_per_shard::Int=4096,
)
    shards = DendriticParameterShard[]
    for (field, name) in enumerate(DENDRITIC_PARAMETER_FIELDS)
        length_array = length(getproperty(parameters, name))
        first_index = 1
        while first_index <= length_array
            last_index = min(
                first_index + elements_per_shard - 1,
                length_array,
            )
            push!(
                shards,
                DendriticParameterShard(
                    UInt8(field),
                    Int32(first_index),
                    Int32(last_index),
                ),
            )
            first_index = last_index + 1
        end
    end
    length(shards) <= typemax(UInt16) ||
        error("too many dendritic parameter shards")
    return shards
end

mutable struct DendriticAdamW{M,V}
    first_moment::M
    second_moment::V
    learning_rate::Float32
    beta1::Float32
    beta2::Float32
    beta1_power::Float32
    beta2_power::Float32
    epsilon::Float32
    weight_decay::Float32
    step::Int
end

function DendriticAdamW(
    parameters;
    learning_rate::Real=5.0f-4,
    beta1::Real=0.9,
    beta2::Real=0.999,
    epsilon::Real=1.0f-8,
    weight_decay::Real=1.0f-5,
)
    b1 = Float32(beta1)
    b2 = Float32(beta2)
    return DendriticAdamW(
        _zero_parameter_tree(parameters),
        _zero_parameter_tree(parameters),
        Float32(learning_rate),
        b1,
        b2,
        1.0f0,
        1.0f0,
        Float32(epsilon),
        Float32(weight_decay),
        0,
    )
end

mutable struct DendriticArenaMetrics
    wall_seconds::Float64
    cpu_seconds::Float64
    allocation_bytes::Int128
    gc_seconds::Float64
    pack_seconds::Float64
    forward_seconds::Float64
    loss_seconds::Float64
    local_seconds::Float64
    optimizer_seconds::Float64
    states_per_second::Float64
    firing_rate::Float64
    plateau_mean::Float64
    routing_entropy::Float64
    local_q_loss::Float64
    local_death_loss::Float64
    local_quantile_loss::Float64
    local_geometry_loss::Float64
    gradient_norm::Float64
    structural_flips::Int
    branch_moves::Int
end

DendriticArenaMetrics() = DendriticArenaMetrics(
    0.0,
    0.0,
    Int128(0),
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0,
    0,
)

mutable struct DendriticArenaTrainer{M,P,O,G}
    model::M
    parameters::P
    initial_parameters::P
    cache::DendriticParameterCache
    optimizer::O
    tape::DendriticTape
    loss_scratch::Point.LossScratch
    gradient::G
    projection::Array{Float32,3}
    branch_for_edge::Matrix{UInt8}
    gate_mask::BitMatrix
    synapse_utility::Matrix{Float32}
    branch_utility::Array{Float32,3}
    parameter_shards::Vector{DendriticParameterShard}
    gradient_norm_squares::Vector{Float64}
    optimizer_scale::Float32
    recurrent_updates_enabled::Bool
    utility_decay::Float32
    utility_connection_cost::Float32
    structural_interval::Int
    branch_interval::Int
    global_signal_scale::Float32
    local_signal_scale::Float32
    routing_entropy_weight::Float32
    routing_entropy_floor::Float32
    routing_load_weight::Float32
    route_load::Matrix{Float32}
    last_loss::Point.LossRecord
    metrics::DendriticArenaMetrics
end

function DendriticArenaTrainer(
    model,
    parameters;
    state_batch::Int=4,
    width::Int=80,
    learning_rate::Real=5.0f-4,
    weight_decay::Real=1.0f-5,
    utility_decay::Real=0.99f0,
    utility_connection_cost::Real=1.0f-6,
    structural_interval::Int=25,
    branch_interval::Int=128,
    global_signal_scale::Real=0.25f0,
    local_signal_scale::Real=4.0f0,
    routing_entropy_weight::Real=0.002f0,
    routing_entropy_floor::Real=0.70f0,
    routing_load_weight::Real=0.002f0,
    projection_seed::Integer=0x44454e4450524f4a,
)
    model.variant === :causal_recurrent_v2 ||
        throw(ArgumentError(
            "ReducedHayV2ArenaTrainer requires causal_recurrent_v2",
        ))
    isfinite(global_signal_scale) &&
        global_signal_scale >= 0 ||
        throw(ArgumentError(
            "global_signal_scale must be finite and nonnegative",
        ))
    isfinite(local_signal_scale) &&
        local_signal_scale >= 0 ||
        throw(ArgumentError(
            "local_signal_scale must be finite and nonnegative",
        ))
    isfinite(routing_entropy_weight) &&
        routing_entropy_weight >= 0 ||
        throw(ArgumentError(
            "routing_entropy_weight must be finite and nonnegative",
        ))
    0 <= routing_entropy_floor <= 1 ||
        throw(ArgumentError(
            "routing_entropy_floor must be in [0, 1]",
        ))
    isfinite(routing_load_weight) &&
        routing_load_weight >= 0 ||
        throw(ArgumentError(
            "routing_load_weight must be finite and nonnegative",
        ))
    tape = DendriticTape(model, state_batch, width)
    rng = Xoshiro(UInt64(projection_seed))
    projection = randn(
        rng,
        Float32,
        model.node_dim,
        OUTPUT_DIM,
        model.blocks,
    ) ./ sqrt(Float32(model.node_dim))
    branch_for_edge = Matrix{UInt8}(
        undef,
        size(parameters.synapse_weight),
    )
    @inbounds for relation in 1:model.fanout
        for source in axes(branch_for_edge, 1)
            branch_for_edge[source, relation] =
                UInt8(model.branch_for_relation[relation])
        end
    end
    gate_mask = falses(size(parameters.gate_logits))
    @inbounds for source in axes(gate_mask, 1)
        selected = partialsortperm(
            view(parameters.gate_logits, source, :),
            1:model.fixed_recurrent_fanout;
            rev=true,
        )
        for relation in selected
            gate_mask[source, relation] = true
        end
    end
    empty_loss = Point.LossRecord(
        ntuple(_ -> 0.0f0, 17)...,
        0,
    )
    shards = _parameter_shards(parameters)
    cache = DendriticParameterCache(parameters)
    refresh_dendritic_cache!(
        cache,
        parameters,
        gate_mask,
    )
    trainer = DendriticArenaTrainer(
        model,
        parameters,
        _copy_parameters(parameters),
        cache,
        DendriticAdamW(
            parameters;
            learning_rate,
            weight_decay,
        ),
        tape,
        Point.LossScratch(width),
        _zero_parameter_tree(parameters),
        projection,
        branch_for_edge,
        gate_mask,
        zeros(Float32, size(parameters.synapse_weight)),
        zeros(
            Float32,
            model.branches,
            size(parameters.synapse_weight, 1),
            size(parameters.synapse_weight, 2),
        ),
        shards,
        zeros(Float64, length(shards)),
        1.0f0,
        true,
        Float32(utility_decay),
        Float32(utility_connection_cost),
        structural_interval,
        branch_interval,
        Float32(global_signal_scale),
        Float32(local_signal_scale),
        Float32(routing_entropy_weight),
        Float32(routing_entropy_floor),
        Float32(routing_load_weight),
        zeros(Float32, model.blocks, model.cycles),
        empty_loss,
        DendriticArenaMetrics(),
    )
    return trainer
end

dendritic_training_arena(trainer::DendriticArenaTrainer) =
    trainer.tape.base

dendritic_arena_output(trainer::DendriticArenaTrainer) =
    Point.arena_output(trainer.tape.base)

function dendritic_parameter_deltas(
    trainer::DendriticArenaTrainer,
)
    return NamedTuple{keys(trainer.parameters)}(
        map(
            (current, initial) -> begin
                maximum_difference = 0.0f0
                @inbounds for index in eachindex(current, initial)
                    maximum_difference = max(
                        maximum_difference,
                        abs(current[index] - initial[index]),
                    )
                end
                maximum_difference
            end,
            values(trainer.parameters),
            values(trainer.initial_parameters),
        ),
    )
end

@enum DendriticWorkKind::UInt8 begin
    DENDRITIC_NO_WORK = 0
    DENDRITIC_PACK = 1
    DENDRITIC_FORWARD = 2
    DENDRITIC_SIGNAL = 3
    DENDRITIC_LOCAL = 4
    DENDRITIC_REDUCE = 5
    DENDRITIC_OPTIMIZER = 6
end

struct DendriticWorkItem
    kind::UInt8
    target::UInt16
    generation::UInt32
end

DendriticWorkItem(
    kind::DendriticWorkKind,
    target::Integer,
    generation::UInt32,
) = DendriticWorkItem(UInt8(kind), UInt16(target), generation)

Base.zero(::Type{DendriticWorkItem}) = DendriticWorkItem(
    UInt8(DENDRITIC_NO_WORK),
    UInt16(0),
    UInt32(0),
)

isbitstype(DendriticWorkItem) ||
    error("DendriticWorkItem must remain isbits")

mutable struct DendriticArenaExecutor{W,T,D}
    queue::Queue.BoundedMPMCQueue{DendriticWorkItem}
    active_workers::Int
    julia_workers::Int
    cpuset_mode::Symbol
    workers::W
    trainer::T
    dataset::D
    stochastic_routing::Bool
    routing_seed::UInt64
    recurrent_signal_scale::Float32
    generation::Base.Threads.Atomic{UInt32}
    remaining::Base.Threads.Atomic{Int}
    shutdown_requested::Base.Threads.Atomic{UInt32}
    ready_workers::Base.Threads.Atomic{Int}
    booted_workers::Base.Threads.Atomic{Int}
    failure_worker::Base.Threads.Atomic{Int}
    failures::Vector{Any}
    bindings::Vector{Any}
    bindings_released::Vector{Bool}
    startup_event::Base.Event
    started::Bool
end

function DendriticArenaExecutor(
    trainer::DendriticArenaTrainer,
    dataset;
    active_workers::Int=Base.Threads.nthreads(:default),
    cpuset_mode::Symbol=:none,
    queue_capacity::Int=2048,
    stochastic_routing::Bool=true,
    routing_seed::Integer=0x44454e44524f5554,
    recurrent_signal_scale::Real=1.0f0,
)
    julia_workers = Base.Threads.nthreads(:default)
    2 <= active_workers <= julia_workers || throw(ArgumentError(
        "active_workers must be in 2:$julia_workers",
    ))
    Base.Threads.nthreads(:interactive) == 0 || error(
        "launch Julia with --threads=N,0",
    )
    cpuset_mode in (:none, :all, :p_only) || throw(ArgumentError(
        "cpuset_mode must be none, all, or p_only",
    ))
    ispow2(queue_capacity) || throw(ArgumentError(
        "queue_capacity must be a power of two",
    ))
    scale = Float32(recurrent_signal_scale)
    isfinite(scale) && scale >= 0.0f0 || throw(ArgumentError(
        "recurrent_signal_scale must be finite and nonnegative",
    ))
    workers = [
        DendriticWorkerScratch(trainer.model, trainer.parameters)
        for _ in 1:active_workers
    ]
    return DendriticArenaExecutor(
        Queue.BoundedMPMCQueue{DendriticWorkItem}(
            queue_capacity,
            zero(DendriticWorkItem),
        ),
        active_workers,
        julia_workers,
        cpuset_mode,
        workers,
        trainer,
        dataset,
        stochastic_routing,
        UInt64(routing_seed),
        scale,
        Base.Threads.Atomic{UInt32}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{UInt32}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Base.Threads.Atomic{Int}(0),
        Any[nothing for _ in 1:julia_workers],
        Any[nothing for _ in 1:julia_workers],
        fill(false, julia_workers),
        Base.Event(true),
        false,
    )
end

function _clear_worker_accumulators!(
    executor::DendriticArenaExecutor,
)
    @inbounds for worker in executor.workers
        _fill_parameter_tree!(worker.gradient)
        fill!(worker.utility, 0.0f0)
        fill!(worker.branch_utility, 0.0f0)
        worker.local_q_loss = 0.0
        worker.local_death_loss = 0.0
        worker.local_quantile_loss = 0.0
        worker.local_geometry_loss = 0.0
        worker.jobs = UInt64(0)
        worker.cpu_ticks = UInt64(0)
    end
    return nothing
end

@inline function _reduce_gradient_field!(
    trainer::DendriticArenaTrainer,
    workers,
    ::Val{F},
    scale::Float32,
) where {F}
    destination = getproperty(trainer.gradient, F)
    @inbounds for index in eachindex(destination)
        value = 0.0f0
        for worker in workers
            value += getproperty(worker.gradient, F)[index]
        end
        destination[index] = scale * value
    end
    return nothing
end

@inline function _reduce_gradient_range!(
    trainer::DendriticArenaTrainer,
    workers,
    ::Val{F},
    first_index::Int,
    last_index::Int,
    scale::Float32,
) where {F}
    destination = getproperty(trainer.gradient, F)
    square_sum = 0.0
    @inbounds for index in first_index:last_index
        value = 0.0f0
        for worker in workers
            value += getproperty(worker.gradient, F)[index]
        end
        value *= scale
        destination[index] = value
        square_sum = muladd(
            Float64(value),
            Float64(value),
            square_sum,
        )
    end
    return square_sum
end

function _reduce_shard!(
    executor::DendriticArenaExecutor,
    target::Int,
)
    trainer = executor.trainer
    shard = @inbounds trainer.parameter_shards[target]
    field = Int(shard.field)
    first_index = Int(shard.first)
    last_index = Int(shard.last)
    recurrent = executor.recurrent_signal_scale
    head = 1.0f0
    square_sum = if field == 1
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:input_exc_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 2
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:input_inh_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 3
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:state_query_weight),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 4
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:branch_bias),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 5
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:branch_leak_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 6
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:ampa_decay_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 7
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:nmda_decay_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 8
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:gaba_decay_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 9
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:current_gain_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 10
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:axial_gain_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 11
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:nmda_slope_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 12
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:nmda_half_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 13
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:plateau_decay_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 14
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:plateau_threshold_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 15
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:plateau_slope_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 16
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:plateau_gain_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 17
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:plateau_feedback_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 18
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:soma_coupling),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 19
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:apical_leak_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 20
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:soma_leak_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 21
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:adaptation_decay_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 22
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:apical_gain_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 23
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:soma_threshold_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 24
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:adaptation_gain_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 25
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:workspace_key),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 26
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:feedback_gain),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 27
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:synapse_weight),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 28
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:gate_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 29
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:delay_logits),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 30
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:workspace_decay_logit),
            first_index,
            last_index,
            recurrent,
        )
    elseif field == 31
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:head_weight),
            first_index,
            last_index,
            head,
        )
    elseif field == 32
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:head_bias),
            first_index,
            last_index,
            head,
        )
    elseif field == 33
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:output_weight),
            first_index,
            last_index,
            head,
        )
    elseif field == 34
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:output_bias),
            first_index,
            last_index,
            head,
        )
    else
        error("unknown Reduced Hay reduction field $field")
    end
    trainer.gradient_norm_squares[target] = square_sum
    return nothing
end

function _finish_gradient_reduction!(
    trainer::DendriticArenaTrainer,
)
    square_sum = 0.0
    @inbounds for value in trainer.gradient_norm_squares
        square_sum += value
    end
    trainer.metrics.gradient_norm = sqrt(square_sum)
    maximum_norm = 5.0
    trainer.optimizer_scale = Float32(
        trainer.metrics.gradient_norm > maximum_norm ?
        maximum_norm / trainer.metrics.gradient_norm : 1.0,
    )
    return nothing
end

function _reduce_worker_accumulators!(
    executor::DendriticArenaExecutor,
)
    trainer = executor.trainer
    recurrent = executor.recurrent_signal_scale
    recurrent == 0.0f0 && return nothing
    cells, fanout = size(trainer.synapse_utility)
    inverse_candidates = inv(Float32(max(
        trainer.tape.base.valid_count,
        1,
    )))
    @inbounds for source in 1:cells
        square_sum = 0.0f0
        for relation in 1:fanout
            value = 0.0f0
            for worker in executor.workers
                value += worker.utility[source, relation]
            end
            value *= inverse_candidates
            executor.workers[1].utility[source, relation] = value
            square_sum = muladd(value, value, square_sum)
        end
        inverse_rms = inv(sqrt(
            square_sum / Float32(fanout) + 1.0f-12,
        ))
        for relation in 1:fanout
            observed =
                executor.workers[1].utility[source, relation] *
                inverse_rms
            trainer.synapse_utility[source, relation] =
                trainer.utility_decay *
                trainer.synapse_utility[source, relation] +
                (1.0f0 - trainer.utility_decay) * observed
        end
        for relation in 1:fanout
            branch_square_sum = 0.0f0
            for branch in 1:trainer.model.branches
                value = 0.0f0
                for worker in executor.workers
                    value += worker.branch_utility[
                        branch,
                        source,
                        relation,
                    ]
                end
                value *= inverse_candidates
                executor.workers[1].branch_utility[
                    branch,
                    source,
                    relation,
                ] = value
                branch_square_sum = muladd(
                    value,
                    value,
                    branch_square_sum,
                )
            end
            branch_inverse_rms = inv(sqrt(
                branch_square_sum /
                Float32(trainer.model.branches) +
                1.0f-12,
            ))
            for branch in 1:trainer.model.branches
                observed = executor.workers[1].branch_utility[
                    branch,
                    source,
                    relation,
                ] * branch_inverse_rms
                trainer.branch_utility[
                    branch,
                    source,
                    relation,
                ] =
                    trainer.utility_decay *
                    trainer.branch_utility[
                        branch,
                        source,
                        relation,
                    ] +
                    (1.0f0 - trainer.utility_decay) * observed
            end
        end
    end
    return nothing
end

@inline function _adam_range!(
    trainer::DendriticArenaTrainer,
    ::Val{F},
    first_index::Int,
    last_index::Int,
) where {F}
    parameter = getproperty(trainer.parameters, F)
    gradient = getproperty(trainer.gradient, F)
    if !trainer.recurrent_updates_enabled &&
       !(F in HEAD_PARAMETER_FIELDS)
        @inbounds for index in first_index:last_index
            gradient[index] = 0.0f0
        end
        return 0.0
    end
    first_moment =
        getproperty(trainer.optimizer.first_moment, F)
    second_moment =
        getproperty(trainer.optimizer.second_moment, F)
    optimizer = trainer.optimizer
    inverse_first_bias =
        inv(1.0f0 - optimizer.beta1_power)
    inverse_second_bias =
        inv(1.0f0 - optimizer.beta2_power)
    beta1_complement = 1.0f0 - optimizer.beta1
    beta2_complement = 1.0f0 - optimizer.beta2
    scale = trainer.optimizer_scale
    norm_square = 0.0
    @inbounds for index in first_index:last_index
        grad = gradient[index] * scale
        norm_square = muladd(
            Float64(grad),
            Float64(grad),
            norm_square,
        )
        moment1 = muladd(
            optimizer.beta1,
            first_moment[index],
            beta1_complement * grad,
        )
        moment2 = muladd(
            optimizer.beta2,
            second_moment[index],
            beta2_complement * grad * grad,
        )
        first_moment[index] = moment1
        second_moment[index] = moment2
        corrected1 = moment1 * inverse_first_bias
        corrected2 = moment2 * inverse_second_bias
        parameter[index] -= optimizer.learning_rate * (
            corrected1 /
            (sqrt(corrected2) + optimizer.epsilon) +
            optimizer.weight_decay * parameter[index]
        )
        gradient[index] = 0.0f0
    end
    return norm_square
end

function _adam_shard!(
    trainer::DendriticArenaTrainer,
    target::Int,
)
    shard = @inbounds trainer.parameter_shards[target]
    field = Int(shard.field)
    first_index = Int(shard.first)
    last_index = Int(shard.last)
    norm_square = if field == 1
        _adam_range!(trainer, Val(:input_exc_logits), first_index, last_index)
    elseif field == 2
        _adam_range!(trainer, Val(:input_inh_logits), first_index, last_index)
    elseif field == 3
        _adam_range!(trainer, Val(:state_query_weight), first_index, last_index)
    elseif field == 4
        _adam_range!(trainer, Val(:branch_bias), first_index, last_index)
    elseif field == 5
        _adam_range!(trainer, Val(:branch_leak_logits), first_index, last_index)
    elseif field == 6
        _adam_range!(trainer, Val(:ampa_decay_logits), first_index, last_index)
    elseif field == 7
        _adam_range!(trainer, Val(:nmda_decay_logits), first_index, last_index)
    elseif field == 8
        _adam_range!(trainer, Val(:gaba_decay_logits), first_index, last_index)
    elseif field == 9
        _adam_range!(trainer, Val(:current_gain_logits), first_index, last_index)
    elseif field == 10
        _adam_range!(trainer, Val(:axial_gain_logits), first_index, last_index)
    elseif field == 11
        _adam_range!(trainer, Val(:nmda_slope_logits), first_index, last_index)
    elseif field == 12
        _adam_range!(trainer, Val(:nmda_half_logits), first_index, last_index)
    elseif field == 13
        _adam_range!(trainer, Val(:plateau_decay_logits), first_index, last_index)
    elseif field == 14
        _adam_range!(trainer, Val(:plateau_threshold_logits), first_index, last_index)
    elseif field == 15
        _adam_range!(trainer, Val(:plateau_slope_logits), first_index, last_index)
    elseif field == 16
        _adam_range!(trainer, Val(:plateau_gain_logits), first_index, last_index)
    elseif field == 17
        _adam_range!(trainer, Val(:plateau_feedback_logits), first_index, last_index)
    elseif field == 18
        _adam_range!(trainer, Val(:soma_coupling), first_index, last_index)
    elseif field == 19
        _adam_range!(trainer, Val(:apical_leak_logits), first_index, last_index)
    elseif field == 20
        _adam_range!(trainer, Val(:soma_leak_logits), first_index, last_index)
    elseif field == 21
        _adam_range!(trainer, Val(:adaptation_decay_logits), first_index, last_index)
    elseif field == 22
        _adam_range!(trainer, Val(:apical_gain_logits), first_index, last_index)
    elseif field == 23
        _adam_range!(trainer, Val(:soma_threshold_logits), first_index, last_index)
    elseif field == 24
        _adam_range!(trainer, Val(:adaptation_gain_logits), first_index, last_index)
    elseif field == 25
        _adam_range!(trainer, Val(:workspace_key), first_index, last_index)
    elseif field == 26
        _adam_range!(trainer, Val(:feedback_gain), first_index, last_index)
    elseif field == 27
        _adam_range!(trainer, Val(:synapse_weight), first_index, last_index)
    elseif field == 28
        _adam_range!(trainer, Val(:gate_logits), first_index, last_index)
    elseif field == 29
        _adam_range!(trainer, Val(:delay_logits), first_index, last_index)
    elseif field == 30
        _adam_range!(trainer, Val(:workspace_decay_logit), first_index, last_index)
    elseif field == 31
        _adam_range!(trainer, Val(:head_weight), first_index, last_index)
    elseif field == 32
        _adam_range!(trainer, Val(:head_bias), first_index, last_index)
    elseif field == 33
        _adam_range!(trainer, Val(:output_weight), first_index, last_index)
    elseif field == 34
        _adam_range!(trainer, Val(:output_bias), first_index, last_index)
    else
        error("unknown dendritic parameter field $field")
    end
    trainer.gradient_norm_squares[target] = norm_square
    return nothing
end

@inline function _zero_gradient_shard!(
    trainer::DendriticArenaTrainer,
    target::Int,
)
    shard = @inbounds trainer.parameter_shards[target]
    field = Int(shard.field)
    name = DENDRITIC_PARAMETER_FIELDS[field]
    gradient = getproperty(trainer.gradient, name)
    @inbounds for index in Int(shard.first):Int(shard.last)
        gradient[index] = 0.0f0
    end
    trainer.gradient_norm_squares[target] = 0.0
    return nothing
end

@inline function _reset_edge_moments!(
    trainer::DendriticArenaTrainer,
    source::Int,
    relation::Int,
)
    first = trainer.optimizer.first_moment
    second = trainer.optimizer.second_moment
    first.synapse_weight[source, relation] = 0.0f0
    second.synapse_weight[source, relation] = 0.0f0
    first.gate_logits[source, relation] = 0.0f0
    second.gate_logits[source, relation] = 0.0f0
    first.delay_logits[source, relation] = 0.0f0
    second.delay_logits[source, relation] = 0.0f0
    return nothing
end

function _consolidate_structure!(
    trainer::DendriticArenaTrainer,
)
    flips = 0
    moves = 0
    step = trainer.optimizer.step
    cells, fanout = size(trainer.parameters.gate_logits)
    if step % trainer.structural_interval == 0
        @inbounds for source in 1:cells
            worst_active = 0
            best_inactive = 0
            worst_utility = Inf32
            best_utility = -Inf32
            for relation in 1:fanout
                utility =
                    trainer.synapse_utility[source, relation]
                if trainer.gate_mask[source, relation]
                    if utility < worst_utility
                        worst_utility = utility
                        worst_active = relation
                    end
                elseif utility > best_utility
                    best_utility = utility
                    best_inactive = relation
                end
            end
            if worst_active != 0 &&
               best_inactive != 0 &&
               best_utility -
               trainer.utility_connection_cost >
               worst_utility
                on_magnitude = max(
                    abs(
                        trainer.parameters.gate_logits[
                            source,
                            best_inactive,
                        ],
                    ),
                    GATE_SIGN_EPSILON,
                )
                off_magnitude = max(
                    abs(
                        trainer.parameters.gate_logits[
                            source,
                            worst_active,
                        ],
                    ),
                    GATE_SIGN_EPSILON,
                )
                trainer.parameters.gate_logits[
                    source,
                    best_inactive,
                ] = on_magnitude
                trainer.parameters.gate_logits[
                    source,
                    worst_active,
                ] = -off_magnitude
                trainer.gate_mask[source, best_inactive] = true
                trainer.gate_mask[source, worst_active] = false
                _reset_edge_moments!(
                    trainer,
                    source,
                    best_inactive,
                )
                _reset_edge_moments!(
                    trainer,
                    source,
                    worst_active,
                )
                flips += 2
            end
        end
    end
    if step % trainer.branch_interval == 0
        @inbounds for source in 1:cells
            best_relation = 0
            best_branch = 0
            best_gain = 0.0f0
            for relation in 1:fanout
                current =
                    Int(trainer.branch_for_edge[source, relation])
                current_utility = trainer.branch_utility[
                    current,
                    source,
                    relation,
                ]
                for branch in 1:trainer.model.branches
                    branch == current && continue
                    gain = trainer.branch_utility[
                        branch,
                        source,
                        relation,
                    ] - current_utility
                    if gain > best_gain
                        best_gain = gain
                        best_relation = relation
                        best_branch = branch
                    end
                end
            end
            if best_relation != 0
                trainer.branch_for_edge[
                    source,
                    best_relation,
                ] = UInt8(best_branch)
                _reset_edge_moments!(
                    trainer,
                    source,
                    best_relation,
                )
                moves += 1
            end
        end
    end
    trainer.metrics.structural_flips = flips
    trainer.metrics.branch_moves = moves
    return nothing
end

function _center_block_supervised_rewards!(
    trainer::DendriticArenaTrainer,
)
    tape = trainer.tape
    base = tape.base
    model = trainer.model
    @inbounds for state_slot in 1:base.state_batch
        count = Int(base.counts[state_slot])
        offset = (state_slot - 1) * base.width
        inverse_count = inv(Float32(count))
        for cycle in 1:model.cycles
            for block in 1:model.blocks
                reward_mean = 0.0f0
                for candidate in 1:count
                    reward_mean +=
                        tape.block_supervised_reward[
                            block,
                            cycle,
                            offset + candidate,
                        ]
                end
                reward_mean *= inverse_count
                for candidate in 1:count
                    flat = offset + candidate
                    tape.block_advantage[
                        block,
                        cycle,
                        flat,
                    ] =
                        tape.block_supervised_reward[
                            block,
                            cycle,
                            flat,
                        ] -
                        reward_mean
                end
            end
        end
    end
    return nothing
end

function _prepare_routing_regularizer!(
    trainer::DendriticArenaTrainer,
)
    base = trainer.tape.base
    model = trainer.model
    fill!(trainer.route_load, 0.0f0)
    fill!(base.route_regularizer_gradient, 0.0f0)
    valid_count = base.valid_count
    valid_count >= 1 || return nothing
    inverse_selection_count = inv(Float32(
        valid_count * model.workspace_k,
    ))
    @inbounds for target in 1:valid_count
        flat = Int(base.valid_flats[target])
        for cycle in 1:model.cycles
            for block in 1:model.blocks
                trainer.route_load[block, cycle] = muladd(
                    base.block_mask[block, cycle, flat],
                    inverse_selection_count,
                    trainer.route_load[block, cycle],
                )
            end
        end
    end
    entropy_weight = trainer.routing_entropy_weight
    load_weight = trainer.routing_load_weight
    entropy_weight == 0.0f0 && load_weight == 0.0f0 &&
        return nothing
    blocks_f = Float32(model.blocks)
    inverse_blocks = inv(blocks_f)
    inverse_valid = inv(Float32(valid_count))
    log_blocks = log(blocks_f)
    inverse_temperature = inv(model.route_temperature)
    @inbounds for target in 1:valid_count
        flat = Int(base.valid_flats[target])
        for cycle in 1:model.cycles
            score_mean = 0.0f0
            for block in 1:model.blocks
                score_mean +=
                    base.route_score[block, cycle, flat]
            end
            score_mean *= inverse_blocks
            score_square_sum = 0.0f0
            entropy = 0.0f0
            load_projection = 0.0f0
            for block in 1:model.blocks
                centered =
                    base.route_score[block, cycle, flat] -
                    score_mean
                score_square_sum = muladd(
                    centered,
                    centered,
                    score_square_sum,
                )
                probability =
                    base.route_base_probability[
                        block,
                        cycle,
                        flat,
                    ]
                entropy -=
                    probability *
                    log(max(probability, 1.0f-12))
                load_projection = muladd(
                    probability,
                    trainer.route_load[block, cycle],
                    load_projection,
                )
            end
            score_inv_rms = inv(sqrt(
                score_square_sum * inverse_blocks +
                InputModel.RMS_NORM_EPS,
            ))
            normalized_entropy = entropy / log_blocks
            entropy_gap = max(
                trainer.routing_entropy_floor -
                normalized_entropy,
                0.0f0,
            )
            gradient_mean = 0.0f0
            gradient_score_projection_mean = 0.0f0
            for block in 1:model.blocks
                probability =
                    base.route_base_probability[
                        block,
                        cycle,
                        flat,
                    ]
                log_probability =
                    log(max(probability, 1.0f-12))
                entropy_gradient =
                    2.0f0 *
                    entropy_weight *
                    entropy_gap *
                    probability *
                    (log_probability + entropy) /
                    log_blocks
                load_gradient =
                    load_weight *
                    blocks_f *
                    inverse_valid *
                    probability *
                    (
                        trainer.route_load[block, cycle] -
                        load_projection
                    )
                normalized_gradient =
                    (entropy_gradient + load_gradient) *
                    inverse_temperature
                base.route_regularizer_gradient[
                    block,
                    cycle,
                    flat,
                ] = normalized_gradient
                standardized_score =
                    (
                        base.route_score[block, cycle, flat] -
                        score_mean
                    ) * score_inv_rms
                gradient_mean += normalized_gradient
                gradient_score_projection_mean = muladd(
                    normalized_gradient,
                    standardized_score,
                    gradient_score_projection_mean,
                )
            end
            gradient_mean *= inverse_blocks
            gradient_score_projection_mean *= inverse_blocks
            for block in 1:model.blocks
                standardized_score =
                    (
                        base.route_score[block, cycle, flat] -
                        score_mean
                    ) * score_inv_rms
                base.route_regularizer_gradient[
                    block,
                    cycle,
                    flat,
                ] = score_inv_rms * (
                    base.route_regularizer_gradient[
                        block,
                        cycle,
                        flat,
                    ] -
                    gradient_mean -
                    standardized_score *
                    gradient_score_projection_mean
                )
            end
        end
    end
    return nothing
end

@inline function _complete_work!(
    executor::DendriticArenaExecutor,
)
    previous = Base.Threads.atomic_add!(executor.remaining, -1)
    previous >= 1 || error("dendritic work counter underflow")
    previous == 1 && Queue.wake_consumers!(executor.queue)
    return nothing
end

function _dispatch!(
    executor::DendriticArenaExecutor,
    worker_slot::Int,
    work::DendriticWorkItem,
)
    work.generation == executor.generation[] ||
        error("stale dendritic work generation")
    trainer = executor.trainer
    worker = @inbounds executor.workers[worker_slot]
    target = Int(work.target)
    cpu_started = CpuSets.thread_cpu_ticks_100ns()
    if work.kind == UInt8(DENDRITIC_PACK)
        flat = Int(trainer.tape.base.valid_flats[target])
        Point.pack_candidate_rails!(
            trainer.tape.base,
            executor.dataset,
            worker.pack,
            flat,
        )
    elseif work.kind == UInt8(DENDRITIC_FORWARD)
        flat = Int(trainer.tape.base.valid_flats[target])
        nonce = executor.stochastic_routing ?
            _routing_nonce(
                executor.routing_seed,
                trainer.optimizer.step,
                flat,
            ) : UInt64(0)
        dendritic_forward_candidate!(
            trainer.tape,
            trainer.model,
            trainer.parameters,
            trainer.cache,
            worker,
            trainer.branch_for_edge,
            flat;
            stochastic_routing=executor.stochastic_routing,
            routing_nonce=nonce,
        )
    elseif work.kind == UInt8(DENDRITIC_SIGNAL)
        flat = Int(trainer.tape.base.valid_flats[target])
        dendritic_prepare_signal_candidate!(
            worker,
            trainer.tape,
            trainer.projection,
            trainer.model,
            flat,
            trainer.global_signal_scale,
            trainer.local_signal_scale,
        )
    elseif work.kind == UInt8(DENDRITIC_LOCAL)
        flat = Int(trainer.tape.base.valid_flats[target])
        dendritic_local_candidate!(
            worker,
            trainer.tape,
            trainer.projection,
            trainer.model,
            trainer.parameters,
            trainer.cache,
            trainer.branch_for_edge,
            flat,
            trainer.global_signal_scale,
            trainer.local_signal_scale,
        )
    elseif work.kind == UInt8(DENDRITIC_REDUCE)
        _reduce_shard!(executor, target)
    elseif work.kind == UInt8(DENDRITIC_OPTIMIZER)
        shard = @inbounds trainer.parameter_shards[target]
        if executor.recurrent_signal_scale == 0.0f0 &&
           Int(shard.field) <= length(DENDRITIC_PARAMETER_FIELDS) -
           length(HEAD_PARAMETER_FIELDS)
            _zero_gradient_shard!(trainer, target)
        else
            _adam_shard!(trainer, target)
        end
    else
        error("unknown dendritic work kind $(work.kind)")
    end
    worker.jobs += UInt64(1)
    worker.cpu_ticks +=
        CpuSets.thread_cpu_ticks_100ns() - cpu_started
    _complete_work!(executor)
    return nothing
end

function _record_failure!(
    executor::DendriticArenaExecutor,
    worker_slot::Int,
    exception,
    backtrace,
)
    executor.failures[worker_slot] = (exception, backtrace)
    Base.Threads.atomic_cas!(
        executor.failure_worker,
        0,
        worker_slot,
    )
    Base.Threads.atomic_xchg!(
        executor.shutdown_requested,
        UInt32(1),
    )
    Queue.close!(executor.queue)
    notify(executor.startup_event)
    return nothing
end

function _throw_failure(executor::DendriticArenaExecutor)
    worker = executor.failure_worker[]
    worker == 0 && return nothing
    payload = executor.failures[worker]
    payload === nothing &&
        error("dendritic worker $worker failed without payload")
    exception, backtrace = payload
    throw(Base.CapturedException(exception, backtrace))
end

function _worker_loop!(
    executor::DendriticArenaExecutor,
    worker_slot::Int,
)
    while executor.shutdown_requested[] == 0
        available, work = Queue.dequeue_wait!(
            executor.queue;
            timeout_ms=100,
        )
        if !available
            Queue.isclosed(executor.queue) && return nothing
            continue
        end
        _dispatch!(executor, worker_slot, work)
    end
    return nothing
end

function _coordinator_drain!(
    executor::DendriticArenaExecutor,
)
    while executor.remaining[] > 0
        _throw_failure(executor)
        available, work = Queue.try_dequeue!(executor.queue)
        if available
            _dispatch!(executor, 1, work)
            continue
        end
        expected = Queue.item_epoch(executor.queue)
        executor.remaining[] == 0 && break
        Queue.wait_for_item_change!(
            executor.queue,
            expected;
            timeout_ms=10,
        )
    end
    _throw_failure(executor)
    executor.remaining[] == 0 ||
        error("dendritic phase ended early")
    return nothing
end

function _run_phase!(
    executor::DendriticArenaExecutor,
    kind::DendriticWorkKind,
    count::Int,
    generation::UInt32,
)
    count == 0 && return 0.0
    count <= typemax(UInt16) ||
        error("dendritic phase has too many jobs")
    started = time_ns()
    executor.remaining[] = count
    @inbounds for target in 1:count
        Queue.enqueue_wait!(
            executor.queue,
            DendriticWorkItem(kind, target, generation);
            timeout_ms=10_000,
        ) || error("dendritic queue closed")
    end
    _coordinator_drain!(executor)
    Queue.approx_length(executor.queue) == 0 ||
        error("dendritic queue not empty at phase boundary")
    return (time_ns() - started) * 1.0e-9
end

function _refresh_dendritic_metrics!(
    executor::DendriticArenaExecutor,
)
    trainer = executor.trainer
    tape = trainer.tape
    base = tape.base
    model = trainer.model
    spike_sum = 0.0
    plateau_sum = 0.0
    entropy_sum = 0.0
    spike_count = 0
    plateau_count = 0
    decision_count = 0
    @inbounds for target in 1:base.valid_count
        flat = Int(base.valid_flats[target])
        for cycle in 1:model.cycles
            for cell in axes(tape.cell_spikes, 1)
                spike_sum += tape.cell_spikes[cell, cycle, flat]
                spike_count += 1
                for branch in 1:model.branches
                    plateau_sum += tape.plateau[
                        cell,
                        branch,
                        cycle + 1,
                        flat,
                    ]
                    plateau_count += 1
                end
            end
            entropy_sum +=
                base.route_normalized_entropy[cycle, flat]
            decision_count += 1
        end
    end
    trainer.metrics.firing_rate =
        spike_sum / max(spike_count, 1)
    trainer.metrics.plateau_mean =
        plateau_sum / max(plateau_count, 1)
    trainer.metrics.routing_entropy =
        entropy_sum / max(decision_count, 1)
    local_count = max(
        base.valid_count * model.blocks * model.cycles,
        1,
    )
    local_q_loss = 0.0
    local_death_loss = 0.0
    local_quantile_loss = 0.0
    local_geometry_loss = 0.0
    @inbounds for worker in executor.workers
        local_q_loss += worker.local_q_loss
        local_death_loss += worker.local_death_loss
        local_quantile_loss += worker.local_quantile_loss
        local_geometry_loss += worker.local_geometry_loss
    end
    trainer.metrics.local_q_loss =
        local_q_loss / local_count
    trainer.metrics.local_death_loss =
        local_death_loss / local_count
    trainer.metrics.local_quantile_loss =
        local_quantile_loss / local_count
    trainer.metrics.local_geometry_loss =
        local_geometry_loss / local_count
    return nothing
end

function reduced_hay_v2_arena_forward!(
    executor::DendriticArenaExecutor,
)
    executor.started ||
        error("Reduced Hay v2 team is not running")
    trainer = executor.trainer
    wall_started = time_ns()
    cpu_started = CpuSets.process_cpu_ticks_100ns()
    gc_started = Base.gc_num()
    generation =
        Base.Threads.atomic_add!(
            executor.generation,
            UInt32(1),
        ) + UInt32(1)
    pack_started = time_ns()
    Point.prepare_batch_metadata!(
        trainer.tape.base,
        executor.dataset,
    )
    _run_phase!(
        executor,
        DENDRITIC_PACK,
        trainer.tape.base.valid_count,
        generation,
    )
    pack_seconds =
        (time_ns() - pack_started) * 1.0e-9
    forward_seconds = _run_phase!(
        executor,
        DENDRITIC_FORWARD,
        trainer.tape.base.valid_count,
        generation,
    )
    wall_seconds =
        (time_ns() - wall_started) * 1.0e-9
    cpu_seconds =
        (
            CpuSets.process_cpu_ticks_100ns() -
            cpu_started
        ) * 1.0e-7
    gc_difference = Base.GC_Diff(Base.gc_num(), gc_started)
    trainer.metrics.wall_seconds = wall_seconds
    trainer.metrics.cpu_seconds = cpu_seconds
    trainer.metrics.allocation_bytes =
        Int128(gc_difference.allocd)
    trainer.metrics.gc_seconds =
        Float64(gc_difference.total_time) * 1.0e-9
    trainer.metrics.pack_seconds = pack_seconds
    trainer.metrics.forward_seconds = forward_seconds
    trainer.metrics.states_per_second =
        trainer.tape.base.state_batch /
        max(wall_seconds, eps(Float64))
    _refresh_dendritic_metrics!(executor)
    return trainer
end

function dendritic_arena_update!(
    executor::DendriticArenaExecutor,
)
    executor.started ||
        error("dendritic team is not running")
    trainer = executor.trainer
    _clear_worker_accumulators!(executor)
    wall_started = time_ns()
    cpu_started = CpuSets.process_cpu_ticks_100ns()
    gc_started = Base.gc_num()
    generation =
        Base.Threads.atomic_add!(
            executor.generation,
            UInt32(1),
        ) + UInt32(1)

    pack_started = time_ns()
    Point.prepare_batch_metadata!(
        trainer.tape.base,
        executor.dataset,
    )
    _run_phase!(
        executor,
        DENDRITIC_PACK,
        trainer.tape.base.valid_count,
        generation,
    )
    pack_seconds =
        (time_ns() - pack_started) * 1.0e-9
    forward_seconds = _run_phase!(
        executor,
        DENDRITIC_FORWARD,
        trainer.tape.base.valid_count,
        generation,
    )
    gate_sum = 0.0f0
    @inbounds for value in trainer.cache.gate_probability
        gate_sum += value
    end
    gate_density =
        gate_sum /
        Float32(length(trainer.cache.gate_probability))
    loss_started = time_ns()
    trainer.last_loss = Point.loss_and_raw_gradient!(
        trainer.tape.base,
        trainer.loss_scratch,
        gate_density,
        0.0f0,
    )
    loss_seconds =
        (time_ns() - loss_started) * 1.0e-9
    signal_seconds = _run_phase!(
        executor,
        DENDRITIC_SIGNAL,
        trainer.tape.base.valid_count,
        generation,
    )
    _center_block_supervised_rewards!(trainer)
    _prepare_routing_regularizer!(trainer)
    local_replay_seconds = _run_phase!(
        executor,
        DENDRITIC_LOCAL,
        trainer.tape.base.valid_count,
        generation,
    )
    trainer.recurrent_updates_enabled =
        executor.recurrent_signal_scale != 0.0f0
    reduction_seconds = _run_phase!(
        executor,
        DENDRITIC_REDUCE,
        length(trainer.parameter_shards),
        generation,
    )
    _finish_gradient_reduction!(trainer)
    local_seconds =
        signal_seconds +
        local_replay_seconds +
        reduction_seconds
    _reduce_worker_accumulators!(executor)

    trainer.optimizer.step += 1
    trainer.optimizer.beta1_power *= trainer.optimizer.beta1
    trainer.optimizer.beta2_power *= trainer.optimizer.beta2
    optimizer_seconds = _run_phase!(
        executor,
        DENDRITIC_OPTIMIZER,
        length(trainer.parameter_shards),
        generation,
    )
    if executor.recurrent_signal_scale == 0.0f0
        trainer.metrics.structural_flips = 0
        trainer.metrics.branch_moves = 0
    else
        _consolidate_structure!(trainer)
    end
    refresh_dendritic_cache!(
        trainer.cache,
        trainer.parameters,
        trainer.gate_mask,
    )
    wall_seconds =
        (time_ns() - wall_started) * 1.0e-9
    cpu_seconds =
        (
            CpuSets.process_cpu_ticks_100ns() -
            cpu_started
        ) * 1.0e-7
    gc_difference = Base.GC_Diff(Base.gc_num(), gc_started)
    trainer.metrics.wall_seconds = wall_seconds
    trainer.metrics.cpu_seconds = cpu_seconds
    trainer.metrics.allocation_bytes =
        Int128(gc_difference.allocd)
    trainer.metrics.gc_seconds =
        Float64(gc_difference.total_time) * 1.0e-9
    trainer.metrics.pack_seconds = pack_seconds
    trainer.metrics.forward_seconds = forward_seconds
    trainer.metrics.loss_seconds = loss_seconds
    trainer.metrics.local_seconds = local_seconds
    trainer.metrics.optimizer_seconds = optimizer_seconds
    trainer.metrics.states_per_second =
        trainer.tape.base.state_batch /
        max(wall_seconds, eps(Float64))
    _refresh_dendritic_metrics!(executor)
    isfinite(trainer.last_loss.composite_loss) ||
        error("non-finite dendritic loss")
    isfinite(trainer.metrics.gradient_norm) ||
        error("non-finite dendritic gradient")
    return trainer
end

function _run_with_dendritic_team_guarded!(
    body::F,
    executor::DendriticArenaExecutor,
) where {F}
    executor.started &&
        error("dendritic team is already running")
    Queue.isclosed(executor.queue) &&
        error("cannot restart closed dendritic queue")
    topology = CpuSets.discover_topology()
    binding_plan = CpuSets.configure_worker_bindings(
        executor.cpuset_mode,
        executor.active_workers,
        topology,
    )
    executor.ready_workers[] = 0
    executor.booted_workers[] = 0
    executor.failure_worker[] = 0
    executor.shutdown_requested[] = 0
    fill!(executor.bindings, nothing)
    fill!(executor.bindings_released, false)
    reset(executor.startup_event)
    executor.started = true
    result = Ref{Any}(nothing)
    failure = nothing
    try
        Base.Threads.threading_run(worker_slot -> begin
            local_failure = nothing
            local_backtrace = nothing
            binding_attempted = false
            try
                binding_attempted = true
                binding =
                    CpuSets.bind_current_worker!(worker_slot)
                executor.bindings[worker_slot] = binding
                booted = Base.Threads.atomic_add!(
                    executor.booted_workers,
                    1,
                ) + 1
                booted == executor.julia_workers &&
                    notify(executor.startup_event)
                worker_slot <= executor.active_workers ||
                    return nothing
                ready = Base.Threads.atomic_add!(
                    executor.ready_workers,
                    1,
                ) + 1
                ready == executor.active_workers &&
                    notify(executor.startup_event)
                if worker_slot == 1
                    while executor.booted_workers[] <
                          executor.julia_workers ||
                          executor.ready_workers[] <
                          executor.active_workers
                        _throw_failure(executor)
                        wait(executor.startup_event)
                    end
                    result[] = body(executor)
                    Base.Threads.atomic_xchg!(
                        executor.shutdown_requested,
                        UInt32(1),
                    )
                    Queue.close!(executor.queue)
                else
                    _worker_loop!(executor, worker_slot)
                end
            catch exception
                local_failure = exception
                local_backtrace = catch_backtrace()
            finally
                if binding_attempted
                    try
                        CpuSets.clear_current_binding!()
                        executor.bindings_released[worker_slot] =
                            true
                    catch exception
                        local_failure === nothing && begin
                            local_failure = exception
                            local_backtrace = catch_backtrace()
                        end
                    end
                end
                local_failure === nothing || _record_failure!(
                    executor,
                    min(worker_slot, length(executor.failures)),
                    local_failure,
                    local_backtrace,
                )
            end
            return nothing
        end, true)
    catch exception
        failure = Base.CapturedException(
            exception,
            catch_backtrace(),
        )
    finally
        executor.started = false
    end
    failure === nothing || throw(failure)
    _throw_failure(executor)
    return (;
        result=result[],
        binding_plan,
        bindings=copy(executor.bindings),
        bindings_released=copy(executor.bindings_released),
    )
end

function run_with_dendritic_team!(
    body::F,
    executor::DendriticArenaExecutor,
) where {F}
    return _run_with_dendritic_team_guarded!(
        body,
        executor,
    )
end

run_with_dendritic_team!(
    executor::DendriticArenaExecutor,
    body::F,
) where {F} = run_with_dendritic_team!(body, executor)

const ReducedHayV2ArenaExecutor = DendriticArenaExecutor
const ReducedHayV2ArenaTrainer = DendriticArenaTrainer
const reduced_hay_v2_arena_output = dendritic_arena_output
const reduced_hay_v2_arena_update! = dendritic_arena_update!
const reduced_hay_v2_parameter_deltas =
    dendritic_parameter_deltas
const reduced_hay_v2_training_arena =
    dendritic_training_arena

end # module
