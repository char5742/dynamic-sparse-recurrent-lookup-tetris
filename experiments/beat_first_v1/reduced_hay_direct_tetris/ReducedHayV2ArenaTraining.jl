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
const FEEDBACK_NOISE_SCALE = 0.025f0
const FEEDBACK_ALIGNMENT_DECAY = 0.02f0
const FEEDBACK_LEARNING_RATE_MULTIPLIER = 5.0f0
const APICAL_RESIDUAL_SCALE = 0.50f0
const BRANCH_MOVE_MARGIN = 0.25f0
const BRANCH_CONSOLIDATION_PARTITIONS = 64
const MAX_PARAMETER_FAMILY_GRADIENT_NORM = 5.0

export DendriticArenaExecutor,
    DendriticArenaMetrics,
    DendriticArenaTrainer,
    DendriticParameterCache,
    ReducedHayV2ArenaExecutor,
    ReducedHayV2ArenaTrainer,
    dendritic_arena_gradient!,
    dendritic_arena_output,
    dendritic_arena_update!,
    dendritic_forward_candidate!,
    dendritic_prepare_workspace_root_signal_candidate!,
    dendritic_parameter_deltas,
    dendritic_training_arena,
    reduced_hay_v2_arena_output,
    reduced_hay_v2_arena_forward!,
    reduced_hay_v2_arena_gradient!,
    reduced_hay_v2_arena_update!,
    reduced_hay_v2_parameter_deltas,
    reduced_hay_v2_training_arena,
    run_with_dendritic_team!

const RECURRENT_PARAMETER_FIELDS = (
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
)

# These two families move the hard top-k boundary directly.  They require a
# slower trust-region timescale than the continuous cell and edge dynamics:
# even a sub-micro parameter displacement can change the selected workspace
# block discontinuously.
const DIRECT_ROUTING_PARAMETER_FIELDS = (
    :state_query_weight,
    :workspace_key,
)

# Sensory contacts are an open-loop encoder.  They do not move the hard route
# boundary or recurrent transition directly, so coupling them to the very slow
# closed-loop recurrent trust region leaves the model on fixed random features.
const SENSORY_PARAMETER_FIELDS = (
    :input_exc_logits,
    :input_inh_logits,
)

# Zero-initialized inter-cell and workspace-feedback paths must grow faster
# than the sensitive intrinsic cell dynamics.  Keeping this family separate
# lets communication emerge without moving membrane time constants and
# thresholds on the same optimizer timescale.
const COMMUNICATION_PARAMETER_FIELDS = (
    :feedback_gain,
    :synapse_weight,
    :gate_logits,
    :delay_logits,
)

const LOCAL_PREDICTOR_PARAMETER_FIELDS = (
    :local_readout,
    :local_readout_bias,
)

const APICAL_CREDIT_PARAMETER_FIELDS = (
    :apical_predictor_weight,
    :apical_predictor_bias,
    :root_feedback,
    :feature_feedback,
    :output_feedback,
)

const LAYERED_FEEDBACK_PARAMETER_FIELDS = (
    :feature_feedback,
    :output_feedback,
)

const HEAD_PARAMETER_FIELDS = (
    :head_weight,
    :head_bias,
    :output_weight,
    :output_bias,
)

const MODEL_PARAMETER_FIELDS = (
    RECURRENT_PARAMETER_FIELDS...,
    HEAD_PARAMETER_FIELDS...,
)

const DENDRITIC_PARAMETER_FIELDS = (
    RECURRENT_PARAMETER_FIELDS...,
    LOCAL_PREDICTOR_PARAMETER_FIELDS...,
    APICAL_CREDIT_PARAMETER_FIELDS...,
    HEAD_PARAMETER_FIELDS...,
)

# v11 intentionally has a different, exact-BPTT-only parameter tree.  Keep it
# separate from the historical v2-v10 registry: adding zero-sized local or
# apical fields to the new tree would both hide accidental dependencies and
# invalidate old checkpoint signatures.
const V11_RECURRENT_PARAMETER_FIELDS = (
    :input_exc_logits,
    :input_inh_logits,
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
    :route_state_projection,
    :state_query_weight,
    :workspace_key,
    :feedback_gain,
    :global_feedback_gain,
    :synapse_weight,
    :gate_logits,
    :delay_logits,
    :workspace_decay_logit,
)

const V11_DIRECT_ROUTING_PARAMETER_FIELDS = (
    :route_state_projection,
    :state_query_weight,
    :workspace_key,
)

const V11_COMMUNICATION_PARAMETER_FIELDS = (
    :feedback_gain,
    :global_feedback_gain,
    :synapse_weight,
    :gate_logits,
    :delay_logits,
)

const V11_HEAD_PARAMETER_FIELDS = (
    :head_state_projection,
    :head_anchor_mix,
    :head_history_mix,
    :head_delta_mix,
    :output_bias,
)

const V11_MODEL_PARAMETER_FIELDS = (
    V11_RECURRENT_PARAMETER_FIELDS...,
    V11_HEAD_PARAMETER_FIELDS...,
)

# No local predictor or apical-credit arrays are legal in the v11 arena.
const V11_ARENA_PARAMETER_FIELDS = V11_MODEL_PARAMETER_FIELDS

# v13 reads the canonical 24 exported coordinates directly.  Its recurrent
# tree is intentionally identical to v11/v12, but there is no learned
# head-state projection and therefore no zero-sized compatibility parameter.
const V13_HEAD_PARAMETER_FIELDS = (
    :head_anchor_mix,
    :head_history_mix,
    :head_delta_mix,
    :output_bias,
)

const V13_MODEL_PARAMETER_FIELDS = (
    V11_RECURRENT_PARAMETER_FIELDS...,
    V13_HEAD_PARAMETER_FIELDS...,
)

const V13_ARENA_PARAMETER_FIELDS = V13_MODEL_PARAMETER_FIELDS

@generated function _is_v11_parameter_tree(
    ::NamedTuple{K},
) where {K}
    return K == V11_ARENA_PARAMETER_FIELDS ? :(true) : :(false)
end

@generated function _is_v13_parameter_tree(
    ::NamedTuple{K},
) where {K}
    return K == V13_ARENA_PARAMETER_FIELDS ? :(true) : :(false)
end

@generated function _is_exact_slot_parameter_tree(
    ::NamedTuple{K},
) where {K}
    return K in (
        V11_ARENA_PARAMETER_FIELDS,
        V13_ARENA_PARAMETER_FIELDS,
    ) ? :(true) : :(false)
end

@inline _arena_parameter_fields(parameters) =
    _is_v13_parameter_tree(parameters) ?
    V13_ARENA_PARAMETER_FIELDS :
    _is_v11_parameter_tree(parameters) ?
    V11_ARENA_PARAMETER_FIELDS :
    DENDRITIC_PARAMETER_FIELDS

@inline _recurrent_parameter_fields(parameters) =
    _is_exact_slot_parameter_tree(parameters) ?
    V11_RECURRENT_PARAMETER_FIELDS : RECURRENT_PARAMETER_FIELDS

@inline _head_parameter_fields(parameters) =
    _is_v13_parameter_tree(parameters) ?
    V13_HEAD_PARAMETER_FIELDS :
    _is_v11_parameter_tree(parameters) ?
    V11_HEAD_PARAMETER_FIELDS :
    HEAD_PARAMETER_FIELDS

@inline _local_predictor_parameter_fields(parameters) =
    _is_exact_slot_parameter_tree(parameters) ? () :
    LOCAL_PREDICTOR_PARAMETER_FIELDS

@inline _apical_credit_parameter_fields(parameters) =
    _is_exact_slot_parameter_tree(parameters) ? () :
    APICAL_CREDIT_PARAMETER_FIELDS

@generated function _is_recurrent_parameter(
    ::NamedTuple{K},
    ::Val{F},
) where {K,F}
    fields = K in (
        V11_ARENA_PARAMETER_FIELDS,
        V13_ARENA_PARAMETER_FIELDS,
    ) ?
        V11_RECURRENT_PARAMETER_FIELDS : RECURRENT_PARAMETER_FIELDS
    return F in fields ? :(true) : :(false)
end

@generated function _is_head_parameter(
    ::NamedTuple{K},
    ::Val{F},
) where {K,F}
    fields = K == V13_ARENA_PARAMETER_FIELDS ?
        V13_HEAD_PARAMETER_FIELDS :
        K == V11_ARENA_PARAMETER_FIELDS ?
        V11_HEAD_PARAMETER_FIELDS : HEAD_PARAMETER_FIELDS
    return F in fields ? :(true) : :(false)
end

@generated function _is_local_predictor_parameter(
    ::NamedTuple{K},
    ::Val{F},
) where {K,F}
    return K in (
        V11_ARENA_PARAMETER_FIELDS,
        V13_ARENA_PARAMETER_FIELDS,
    ) ?
        :(false) : (F in LOCAL_PREDICTOR_PARAMETER_FIELDS ? :(true) : :(false))
end

@generated function _is_apical_credit_parameter(
    ::NamedTuple{K},
    ::Val{F},
) where {K,F}
    return K in (
        V11_ARENA_PARAMETER_FIELDS,
        V13_ARENA_PARAMETER_FIELDS,
    ) ?
        :(false) : (F in APICAL_CREDIT_PARAMETER_FIELDS ? :(true) : :(false))
end

@generated function _is_direct_routing_parameter(
    ::NamedTuple{K},
    ::Val{F},
) where {K,F}
    fields = K in (
        V11_ARENA_PARAMETER_FIELDS,
        V13_ARENA_PARAMETER_FIELDS,
    ) ?
        V11_DIRECT_ROUTING_PARAMETER_FIELDS :
        DIRECT_ROUTING_PARAMETER_FIELDS
    return F in fields ? :(true) : :(false)
end

@generated function _is_communication_parameter(
    ::NamedTuple{K},
    ::Val{F},
) where {K,F}
    fields = K in (
        V11_ARENA_PARAMETER_FIELDS,
        V13_ARENA_PARAMETER_FIELDS,
    ) ?
        V11_COMMUNICATION_PARAMETER_FIELDS :
        COMMUNICATION_PARAMETER_FIELDS
    return F in fields ? :(true) : :(false)
end

@generated function _recurrent_field_index(
    ::NamedTuple{K},
    ::Val{F},
) where {K,F}
    fields = K in (
        V11_ARENA_PARAMETER_FIELDS,
        V13_ARENA_PARAMETER_FIELDS,
    ) ?
        V11_RECURRENT_PARAMETER_FIELDS : RECURRENT_PARAMETER_FIELDS
    index = findfirst(==(F), fields)
    index === nothing && return :(0)
    return :($index)
end

@generated function _recurrent_field_index(::Val{F}) where {F}
    index = findfirst(==(F), RECURRENT_PARAMETER_FIELDS)
    index === nothing && return :(0)
    return :($index)
end

@inline _logistic_derivative(value::Float32) =
    value * (1.0f0 - value)

@inline function _hard_sigmoid(value::Float32)
    return clamp(muladd(0.2f0, value, 0.5f0), 0.0f0, 1.0f0)
end

@inline function _hard_sigmoid_derivative(value::Float32)
    return (-2.5f0 < value < 2.5f0) ? 0.2f0 : 0.0f0
end

@inline function _apical_activation(model, value::Float32)
    baseline = model.apical_response === :centered_v2 ? 0.5f0 : 0.0f0
    return _hard_sigmoid(value) - baseline
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
    parameter_keys = keys(parameters)
    parameter_keys == DENDRITIC_PARAMETER_FIELDS ||
        parameter_keys == V11_ARENA_PARAMETER_FIELDS ||
        parameter_keys == V13_ARENA_PARAMETER_FIELDS ||
        error("dendritic parameter registry changed")
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

function _with_local_predictor(parameters, projection, model)
    if keys(parameters) in (
        V11_MODEL_PARAMETER_FIELDS,
        V13_MODEL_PARAMETER_FIELDS,
    )
        _uses_exact_block_slots(model) || error(
            "exact-slot parameter tree requires exact block slots",
        )
        return parameters
    end
    keys(parameters) == MODEL_PARAMETER_FIELDS || error(
        "Reduced Hay v2 model parameter registry changed",
    )
    recurrent = map(
        name -> getproperty(parameters, name),
        RECURRENT_PARAMETER_FIELDS,
    )
    head = map(
        name -> getproperty(parameters, name),
        HEAD_PARAMETER_FIELDS,
    )
    feature_dim = Model.reduced_hay_head_feature_dim(model)
    root_feedback = zeros(Float32, feature_dim, OUTPUT_DIM)
    return NamedTuple{DENDRITIC_PARAMETER_FIELDS}((
        recurrent...,
        copy(projection),
        zeros(Float32, OUTPUT_DIM, model.blocks),
        zeros(
            Float32,
            model.readout_per_cell,
            model.readout_per_cell,
            model.blocks * model.cells_per_block,
        ),
        zeros(
            Float32,
            model.readout_per_cell,
            model.blocks * model.cells_per_block,
        ),
        root_feedback,
        zeros(Float32, feature_dim, model.hidden),
        zeros(Float32, model.hidden, OUTPUT_DIM),
        head...,
    ))
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
    spatial_bound_coordinate::Matrix{Int32}
    spatial_bound_sign::Matrix{Float32}
    spatial_inverse_coordinate::Matrix{Int32}
    spatial_inverse_sign::Matrix{Float32}
    temporal_bound_coordinate::Matrix{Int32}
    temporal_bound_sign::Matrix{Float32}
    sensory_anchor::Matrix{Float32}
    temporal_workspace::Matrix{Float32}
    anchor_delta::Matrix{Float32}
    sensory_anchor_inv_rms::Vector{Float32}
    temporal_workspace_inv_rms::Vector{Float32}
    anchor_delta_inv_rms::Vector{Float32}
    block_supervised_reward::Array{Float32,3}
    block_advantage::Array{Float32,3}
end

@inline _uses_exact_block_slots(model) =
    hasproperty(model, :workspace_layout) &&
    model.workspace_layout === :exact_block_slots

@inline _arena_route_dim(model) =
    _uses_exact_block_slots(model) ? Int(model.route_dim) : model.node_dim

@inline _uses_direct_axis_head(model) =
    hasproperty(model, :head_layout) &&
    model.head_layout === :axis_direct

function DendriticTape(model, state_batch::Int, width::Int)
    base = Point.TrainingArena(model, state_batch, width)
    cells = model.blocks * model.cells_per_block
    capacity = state_batch * width
    times = model.cycles + 1
    route_dim = _arena_route_dim(model)
    spatial_coordinate = Matrix{Int32}(
        undef,
        model.node_dim,
        model.blocks,
    )
    spatial_sign = Matrix{Float32}(
        undef,
        model.node_dim,
        model.blocks,
    )
    spatial_inverse_coordinate = similar(spatial_coordinate)
    spatial_inverse_sign = similar(spatial_sign)
    @inbounds for block in 1:model.blocks
        for bound_coordinate in 1:model.node_dim
            raw_coordinate = Model.spatial_bound_coordinate(
                model,
                block,
                bound_coordinate,
            )
            sign = Model.spatial_bound_sign(
                model,
                block,
                bound_coordinate,
            )
            spatial_coordinate[bound_coordinate, block] =
                Int32(raw_coordinate)
            spatial_sign[bound_coordinate, block] = sign
            spatial_inverse_coordinate[raw_coordinate, block] =
                Int32(bound_coordinate)
            spatial_inverse_sign[raw_coordinate, block] = sign
        end
    end
    temporal_coordinate = Matrix{Int32}(
        undef,
        model.node_dim,
        model.cycles,
    )
    temporal_sign = Matrix{Float32}(
        undef,
        model.node_dim,
        model.cycles,
    )
    @inbounds for cycle in 1:model.cycles
        for bound_coordinate in 1:model.node_dim
            temporal_coordinate[bound_coordinate, cycle] = Int32(
                Model.temporal_bound_coordinate(
                    model,
                    cycle,
                    bound_coordinate,
                ),
            )
            temporal_sign[bound_coordinate, cycle] =
                Model.temporal_bound_sign(
                    model,
                    cycle,
                    bound_coordinate,
                )
        end
    end
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
            route_dim,
            model.cycles,
            capacity,
        ),
        zeros(
            Float32,
            route_dim,
            model.cycles,
            capacity,
        ),
        zeros(Float32, model.cycles, capacity),
        spatial_coordinate,
        spatial_sign,
        spatial_inverse_coordinate,
        spatial_inverse_sign,
        temporal_coordinate,
        temporal_sign,
        zeros(Float32, model.node_dim, capacity),
        zeros(Float32, model.node_dim, capacity),
        zeros(Float32, model.node_dim, capacity),
        zeros(Float32, capacity),
        zeros(Float32, capacity),
        zeros(Float32, capacity),
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

@inline _positive_state_readout(value::Float32) =
    value / (1.0f0 + value)

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
    if model.cell_export === :full24
        channel == 1 &&
            return tanh(tape.soma[cell, time, flat])
        channel == 2 &&
            return time == 1 ? 0.0f0 :
                tape.soma_spikes[cell, time - 1, flat]
        channel == 3 &&
            return tanh(tape.apical[cell, time, flat])
        channel == 4 &&
            return _positive_state_readout(
                tape.adaptation[cell, time, flat],
            )
        branch = div(channel - 5, 5) + 1
        branch_channel = mod(channel - 5, 5) + 1
        branch_channel == 1 &&
            return tanh(
                tape.branch_voltage[cell, branch, time, flat],
            )
        branch_channel == 2 &&
            return _positive_state_readout(
                tape.ampa[cell, branch, time, flat],
            )
        branch_channel == 3 &&
            return _positive_state_readout(
                tape.nmda[cell, branch, time, flat],
            )
        branch_channel == 4 &&
            return _positive_state_readout(
                tape.gaba[cell, branch, time, flat],
            )
        return _positive_state_readout(
            tape.plateau[cell, branch, time, flat],
        )
    end
    channel == 1 &&
        return tanh(tape.soma[cell, time, flat])
    channel == 2 &&
        return tanh(tape.apical[cell, time, flat])
    branch = channel - 2
    return tanh(tape.branch_voltage[cell, branch, time, flat])
end

@inline function _scatter_export_cotangent!(
    scratch,
    tape::DendriticTape,
    model,
    coordinate::Int,
    block::Int,
    cycle::Int,
    flat::Int,
    signal::Float32,
)
    signal == 0.0f0 && return nothing
    cell = _cell_for_coordinate(model, coordinate, block)
    channel = _channel_for_coordinate(model, coordinate)
    exported = tape.base.membrane[
        coordinate + (block - 1) * model.node_dim,
        cycle + 1,
        flat,
    ]
    if model.cell_export === :full24
        if channel == 1
            scratch.soma_signal[cell, cycle] +=
                signal * (1.0f0 - exported * exported)
        elseif channel == 2
            # The exported event is raw.  The cell adjoint applies the soma
            # spike surrogate exactly once when this cotangent is consumed.
            scratch.spike_signal[cell, cycle] += signal
        elseif channel == 3
            scratch.apical_signal[cell, cycle] +=
                signal * (1.0f0 - exported * exported)
        elseif channel == 4
            scratch.adaptation_signal[cell, cycle] +=
                signal * (1.0f0 - exported) * (1.0f0 - exported)
        else
            branch = div(channel - 5, 5) + 1
            branch_channel = mod(channel - 5, 5) + 1
            if branch_channel == 1
                scratch.branch_signal[cell, branch, cycle] +=
                    signal * (1.0f0 - exported * exported)
            elseif branch_channel == 2
                scratch.ampa_signal[cell, branch, cycle] +=
                    signal * (1.0f0 - exported) * (1.0f0 - exported)
            elseif branch_channel == 3
                scratch.nmda_signal[cell, branch, cycle] +=
                    signal * (1.0f0 - exported) * (1.0f0 - exported)
            elseif branch_channel == 4
                scratch.gaba_signal[cell, branch, cycle] +=
                    signal * (1.0f0 - exported) * (1.0f0 - exported)
            else
                scratch.plateau_signal[cell, branch, cycle] +=
                    signal * (1.0f0 - exported) * (1.0f0 - exported)
            end
        end
        return nothing
    end

    if channel == 1
        scratch.soma_signal[cell, cycle] +=
            signal * (1.0f0 - exported * exported)
    elseif channel == 2
        scratch.apical_signal[cell, cycle] +=
            signal * (1.0f0 - exported * exported)
    else
        scratch.branch_signal[cell, channel - 2, cycle] +=
            signal * (1.0f0 - exported * exported)
    end
    return nothing
end

@inline function _reset_export_signals!(scratch)
    fill!(scratch.workspace_root_signal, 0.0f0)
    fill!(scratch.soma_signal, 0.0f0)
    fill!(scratch.apical_signal, 0.0f0)
    fill!(scratch.adaptation_signal, 0.0f0)
    fill!(scratch.branch_signal, 0.0f0)
    fill!(scratch.ampa_signal, 0.0f0)
    fill!(scratch.nmda_signal, 0.0f0)
    fill!(scratch.gaba_signal, 0.0f0)
    fill!(scratch.plateau_signal, 0.0f0)
    fill!(scratch.spike_signal, 0.0f0)
    fill!(scratch.route_mask_signal, 0.0f0)
    return nothing
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

@inline function _v11_route_block_sign(model, block::Int, route::Int)
    return Model.reduced_hay_route_block_sign(route, block)
end

@inline function _v11_axis_project_cell!(
    destination::Vector{Float32},
    tape::DendriticTape,
    model,
    parameters,
    block::Int,
    local_cell::Int,
    time::Int,
    flat::Int,
    subtract_time::Int=0,
)
    rank_count = model.head_state_rank
    state_count = model.readout_per_cell
    block_offset = (block - 1) * model.node_dim
    cell_offset = block_offset + (local_cell - 1) * state_count
    @inbounds for rank in 1:rank_count
        destination[rank] = 0.0f0
    end
    @inbounds for state in 1:state_count
        value = tape.base.membrane[cell_offset + state, time, flat]
        if subtract_time != 0
            value -= tape.base.membrane[
                cell_offset + state,
                subtract_time,
                flat,
            ]
        end
        for rank in 1:rank_count
            destination[rank] = muladd(
                parameters.head_state_projection[
                    rank,
                    state,
                    local_cell,
                ],
                value,
                destination[rank],
            )
        end
    end
    @inbounds for rank in 1:rank_count
        destination[rank] = tanh(destination[rank])
    end
    return nothing
end

@inline function _v11_accumulate_axis_head!(
    raw::Matrix{Float32},
    flat::Int,
    projection::Vector{Float32},
    mix,
    block::Int,
    local_cell::Int,
)
    @inbounds for rank in eachindex(projection)
        value = projection[rank]
        for output in 1:OUTPUT_DIM
            raw[output, flat] = muladd(
                mix[output, block, local_cell, rank],
                value,
                raw[output, flat],
            )
        end
    end
    return nothing
end

@inline function _v11_accumulate_history_head!(
    raw::Matrix{Float32},
    flat::Int,
    projection::Vector{Float32},
    mix,
    cycle::Int,
    block::Int,
    local_cell::Int,
)
    @inbounds for rank in eachindex(projection)
        value = projection[rank]
        for output in 1:OUTPUT_DIM
            raw[output, flat] = muladd(
                mix[output, cycle, block, local_cell, rank],
                value,
                raw[output, flat],
            )
        end
    end
    return nothing
end

@inline function _v13_accumulate_direct_axis_head!(
    raw::Matrix{Float32},
    flat::Int,
    tape::DendriticTape,
    model,
    mix,
    block::Int,
    local_cell::Int,
    time::Int,
    subtract_time::Int=0,
)
    state_count = model.readout_per_cell
    block_offset = (block - 1) * model.node_dim
    cell_offset = block_offset + (local_cell - 1) * state_count
    @inbounds for state in 1:state_count
        value = tape.base.membrane[cell_offset + state, time, flat]
        if subtract_time != 0
            value -= tape.base.membrane[
                cell_offset + state,
                subtract_time,
                flat,
            ]
        end
        for output in 1:OUTPUT_DIM
            raw[output, flat] = muladd(
                mix[output, block, local_cell, state],
                value,
                raw[output, flat],
            )
        end
    end
    return nothing
end

@inline function _v13_accumulate_direct_history_head!(
    raw::Matrix{Float32},
    flat::Int,
    tape::DendriticTape,
    model,
    mix,
    cycle::Int,
    block::Int,
    local_cell::Int,
)
    state_count = model.readout_per_cell
    block_offset = (block - 1) * model.node_dim
    cell_offset = block_offset + (local_cell - 1) * state_count
    @inbounds for state in 1:state_count
        value = tape.base.membrane[
            cell_offset + state,
            cycle + 1,
            flat,
        ]
        for output in 1:OUTPUT_DIM
            raw[output, flat] = muladd(
                mix[output, cycle, block, local_cell, state],
                value,
                raw[output, flat],
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
    global_block_signal::Vector{Float32}
    query_input_state::Vector{Float32}
    exact_workspace::Array{Float32,3}
    route_state::Matrix{Float32}
    route_context::Matrix{Float32}
    axis_projection::Vector{Float32}
    slot_adjoint::Matrix{Float32}
    route_state_signal::Matrix{Float32}
    route_query_signal::Vector{Float32}
    global_context_signal::Vector{Float32}
    local_block_signal::Vector{Float32}
    root_block_signal::Matrix{Float32}
    feedback_error::Matrix{Float32}
    feedback_next::Matrix{Float32}
    feedback_norm::Vector{Float32}
    workspace_root_signal::Matrix{Float32}
    soma_signal::Matrix{Float32}
    apical_signal::Matrix{Float32}
    adaptation_signal::Matrix{Float32}
    branch_signal::Array{Float32,3}
    ampa_signal::Array{Float32,3}
    nmda_signal::Array{Float32,3}
    gaba_signal::Array{Float32,3}
    plateau_signal::Array{Float32,3}
    branch_exc_drive_signal::Array{Float32,3}
    branch_inh_drive_signal::Array{Float32,3}
    spike_signal::Matrix{Float32}
    route_mask_signal::Matrix{Float32}
    gate_cotangent::Matrix{Float32}
    adjoint_ampa::Matrix{Float32}
    adjoint_nmda::Matrix{Float32}
    adjoint_gaba::Matrix{Float32}
    adjoint_branch::Matrix{Float32}
    adjoint_plateau::Matrix{Float32}
    adjoint_apical::Vector{Float32}
    adjoint_soma::Vector{Float32}
    adjoint_adaptation::Vector{Float32}
    utility::Matrix{Float32}
    branch_utility::Array{Float32,3}
    active_edges::Vector{Int32}
    active_edge_mask::Vector{Bool}
    active_edge_count::Int
    local_q_loss::Float64
    local_death_loss::Float64
    local_quantile_loss::Float64
    local_geometry_loss::Float64
    apical_predictor_loss::Float64
    feedback_alignment_loss::Float64
    burst_gate_sum::Float64
    burst_gate_count::Int
    jobs::UInt64
    cpu_ticks::UInt64
end

function DendriticWorkerScratch(model, parameters)
    cells = model.blocks * model.cells_per_block
    edge_shape = size(parameters.synapse_weight)
    exact_block_slots = _uses_exact_block_slots(model)
    route_dim = _arena_route_dim(model)
    head_state_rank =
        exact_block_slots && !_uses_direct_axis_head(model) ?
        Int(model.head_state_rank) : 0
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
        zeros(Float32, model.node_dim),
        zeros(Float32, model.node_dim),
        exact_block_slots ?
            zeros(
                Float32,
                model.node_dim,
                model.blocks,
                model.cycles + 1,
            ) : zeros(Float32, 0, 0, 0),
        exact_block_slots ?
            zeros(Float32, route_dim, model.blocks) :
            zeros(Float32, 0, 0),
        exact_block_slots ?
            zeros(Float32, route_dim, model.cycles + 1) :
            zeros(Float32, 0, 0),
        exact_block_slots && head_state_rank > 0 ?
            zeros(Float32, head_state_rank) : Float32[],
        exact_block_slots ?
            zeros(Float32, model.node_dim, model.blocks) :
            zeros(Float32, 0, 0),
        exact_block_slots ?
            zeros(Float32, route_dim, model.blocks) :
            zeros(Float32, 0, 0),
        exact_block_slots ? zeros(Float32, route_dim) : Float32[],
        exact_block_slots ? zeros(Float32, route_dim) : Float32[],
        zeros(Float32, model.node_dim),
        zeros(Float32, model.node_dim, model.blocks),
        zeros(Float32, model.node_dim, model.blocks),
        zeros(Float32, model.node_dim, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.node_dim, model.cycles),
        zeros(Float32, cells, model.cycles),
        zeros(Float32, cells, model.cycles),
        zeros(Float32, cells, model.cycles),
        zeros(Float32, cells, model.branches, model.cycles),
        zeros(Float32, cells, model.branches, model.cycles),
        zeros(Float32, cells, model.branches, model.cycles),
        zeros(Float32, cells, model.branches, model.cycles),
        zeros(Float32, cells, model.branches, model.cycles),
        zeros(Float32, cells, model.branches, model.cycles),
        zeros(Float32, cells, model.branches, model.cycles),
        zeros(Float32, cells, model.cycles),
        zeros(Float32, model.blocks, model.cycles),
        zeros(Float32, edge_shape),
        zeros(Float32, cells, model.branches),
        zeros(Float32, cells, model.branches),
        zeros(Float32, cells, model.branches),
        zeros(Float32, cells, model.branches),
        zeros(Float32, cells, model.branches),
        zeros(Float32, cells),
        zeros(Float32, cells),
        zeros(Float32, cells),
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
        0.0,
        0.0,
        0.0,
        0,
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
    routing_temperature::Float32,
    routing_logit_limit::Float32=Routing.DEFAULT_LOGIT_LIMIT,
)
    Routing.prepare_policy!(
        scratch.route_standardized,
        scratch.base_route,
        scratch.soft_route,
        scratch.scores;
        temperature=routing_temperature,
        logit_limit=routing_logit_limit,
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
        temperature=routing_temperature,
        logit_limit=routing_logit_limit,
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

@inline function _force_recorded_route!(
    scratch::DendriticWorkerScratch,
    model,
    forced_route_order,
    cycle::Int,
    routing_temperature::Float32,
    routing_logit_limit::Float32,
)
    fill!(scratch.selected, false)
    @inbounds for rank in 1:model.workspace_k
        block = Int(forced_route_order[rank, cycle])
        1 <= block <= model.blocks || throw(ArgumentError(
            "forced route contains an invalid block",
        ))
        scratch.selected[block] && throw(ArgumentError(
            "forced route contains a duplicate block",
        ))
        scratch.selected[block] = true
        scratch.route_order[rank] = Int16(block)
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
        temperature=routing_temperature,
        logit_limit=routing_logit_limit,
    )
    return nothing
end

@inline function _apply_route_revisit_policy!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    cycle::Int,
    flat::Int,
)
    model.route_revisit_policy === :allow && return nothing
    cycle == 1 && return nothing
    minimum_score = Inf32
    maximum_score = -Inf32
    @inbounds for block in 1:model.blocks
        score = scratch.scores[block]
        minimum_score = min(minimum_score, score)
        maximum_score = max(maximum_score, score)
    end
    penalty = maximum_score - minimum_score + 1.0f0
    @inbounds for block in 1:model.blocks
        visited = false
        for previous_cycle in 1:(cycle - 1)
            if tape.base.block_mask[block, previous_cycle, flat] != 0.0f0
                visited = true
                break
            end
        end
        visited && (scratch.scores[block] -= penalty)
    end
    return nothing
end

@inline function _internal_sleep_noise(
    seed::UInt64,
    flat::Int,
    coordinate::Int,
)
    mixed = Routing.routing_mix64(
        seed ⊻
        UInt64(flat) * UInt64(0x9e3779b97f4a7c15) ⊻
        UInt64(coordinate) * UInt64(0xd6e8feb86659fd93),
    )
    unit = Float32(mixed >> 40) / Float32(0x01000000)
    return muladd(2.0f0, unit, -1.0f0)
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
    routing_temperature::Real=model.route_temperature,
    routing_logit_limit::Real=Routing.DEFAULT_LOGIT_LIMIT,
    internal_noise_seed::Union{Nothing,UInt64}=nothing,
    internal_noise_scale::Real=0.0f0,
    internal_noise_block::Int=0,
    require_zero_rails::Bool=false,
    silenced_block::Int=0,
    internal_state_key=nothing,
    internal_key_fraction::Real=0.0f0,
    internal_key_gain::Real=0.0f0,
    forced_route_order=nothing,
)
    base = tape.base
    cells = model.blocks * model.cells_per_block
    node_dim = model.node_dim
    readout = model.readout_per_cell
    exact_block_slots = _uses_exact_block_slots(model)
    direct_axis_head = _uses_direct_axis_head(model)
    route_dim = _arena_route_dim(model)
    route_state_rank = exact_block_slots ?
        div(route_dim, model.cells_per_block) : 0
    bound_workspace = model.workspace_binding !== :none
    anchored_temporal = model.head_readout === :anchored_temporal
    noise_scale = Float32(internal_noise_scale)
    isfinite(noise_scale) && noise_scale >= 0.0f0 ||
        throw(ArgumentError(
            "internal_noise_scale must be finite and nonnegative",
        ))
    0 <= internal_noise_block <= model.blocks ||
        throw(ArgumentError("internal_noise_block is outside the model"))
    0 <= silenced_block <= model.blocks ||
        throw(ArgumentError("silenced_block is outside the model"))
    key_fraction = Float32(internal_key_fraction)
    key_gain = Float32(internal_key_gain)
    0.0f0 <= key_fraction <= 1.0f0 ||
        throw(ArgumentError("internal_key_fraction must be in [0, 1]"))
    0.0f0 <= key_gain <= 1.0f0 ||
        throw(ArgumentError("internal_key_gain must be in [0, 1]"))
    internal_state_key === nothing ||
        length(internal_state_key) == node_dim ||
        throw(DimensionMismatch("internal_state_key"))
    if forced_route_order !== nothing
        size(forced_route_order, 1) == model.workspace_k ||
            throw(DimensionMismatch("forced route rank axis"))
        size(forced_route_order, 2) == model.cycles ||
            throw(DimensionMismatch("forced route cycle axis"))
    end
    if require_zero_rails
        @inbounds for rail in axes(base.rails, 1)
            base.rails[rail, flat] == 0.0f0 || error(
                "sleep forward observed a nonzero external rail",
            )
        end
    end
    @inbounds for cell in 1:cells
        block = div(cell - 1, model.cells_per_block) + 1
        inject =
            internal_noise_seed !== nothing &&
            (internal_noise_block == 0 || block == internal_noise_block)
        for branch in 1:model.branches
            coordinate = branch + model.branches * (cell - 1)
            initial = inject ?
                noise_scale * _internal_sleep_noise(
                    internal_noise_seed::UInt64,
                    flat,
                    coordinate,
                ) :
                0.0f0
            if inject && internal_state_key !== nothing
                local_cell = mod1(cell, model.cells_per_block)
                branch_channel = model.cell_export === :full24 ?
                    5 + 5 * (branch - 1) : 2 + branch
                key_coordinate =
                    (local_cell - 1) * readout + branch_channel
                selector = 0.5f0 * (
                    _internal_sleep_noise(
                        (internal_noise_seed::UInt64) ⊻
                        UInt64(0x4355454b45593131),
                        flat,
                        key_coordinate,
                    ) + 1.0f0
                )
                if selector < key_fraction
                    cue = atanh(clamp(
                        Float32(internal_state_key[key_coordinate]),
                        -0.95f0,
                        0.95f0,
                    ))
                    initial = muladd(key_gain, cue - initial, initial)
                end
            end
            tape.branch_voltage[cell, branch, 1, flat] = initial
            tape.ampa[cell, branch, 1, flat] = 0.0f0
            tape.nmda[cell, branch, 1, flat] = 0.0f0
            tape.gaba[cell, branch, 1, flat] = 0.0f0
            tape.plateau[cell, branch, 1, flat] = 0.0f0
        end
        tape.apical[cell, 1, flat] = 0.0f0
        soma_coordinate = model.branches * cells + cell
        soma_initial = inject ?
            noise_scale * _internal_sleep_noise(
                internal_noise_seed::UInt64,
                flat,
                soma_coordinate,
            ) : 0.0f0
        if inject && internal_state_key !== nothing
            local_cell = mod1(cell, model.cells_per_block)
            key_coordinate = (local_cell - 1) * readout + 1
            selector = 0.5f0 * (
                _internal_sleep_noise(
                    (internal_noise_seed::UInt64) ⊻
                    UInt64(0x4355454b45593232),
                    flat,
                    key_coordinate,
                ) + 1.0f0
            )
            if selector < key_fraction
                cue = atanh(clamp(
                    Float32(internal_state_key[key_coordinate]),
                    -0.95f0,
                    0.95f0,
                ))
                soma_initial = muladd(
                    key_gain,
                    cue - soma_initial,
                    soma_initial,
                )
            end
        end
        tape.soma[cell, 1, flat] = soma_initial
        tape.adaptation[cell, 1, flat] = 0.0f0
    end
    _write_exported_state!(tape, model, 1, flat)

    @inbounds for coordinate in 1:node_dim
        base.workspace[coordinate, 1, flat] = 0.0f0
        base.query_pre[coordinate, flat] = 0.0f0
        base.query[coordinate, flat] = 0.0f0
        tape.sensory_anchor[coordinate, flat] = 0.0f0
        tape.temporal_workspace[coordinate, flat] = 0.0f0
        tape.anchor_delta[coordinate, flat] = 0.0f0
    end
    tape.sensory_anchor_inv_rms[flat] = 0.0f0
    tape.temporal_workspace_inv_rms[flat] = 0.0f0
    tape.anchor_delta_inv_rms[flat] = 0.0f0
    if exact_block_slots
        fill!(scratch.exact_workspace, 0.0f0)
        fill!(scratch.route_state, 0.0f0)
        fill!(scratch.route_context, 0.0f0)
        @inbounds for route in 1:route_dim
            for cycle in 1:model.cycles
                tape.state_query_pre[route, cycle, flat] = 0.0f0
                tape.state_query[route, cycle, flat] = 0.0f0
            end
        end
    end
    sensory_cycle_scale =
        Model.reduced_hay_sensory_cycle_scale(model)
    sensory_normalization =
        sensory_cycle_scale *
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
            if exact_block_slots
                cell_offset = (local_cell - 1) * readout
                for state in 1:readout
                    apical_drive = muladd(
                        parameters.feedback_gain[
                            state,
                            local_cell,
                            block,
                        ],
                        scratch.exact_workspace[
                            cell_offset + state,
                            block,
                            cycle,
                        ],
                        apical_drive,
                    )
                end
                apical_drive /= sqrt(Float32(readout))
                global_drive = 0.0f0
                for route in 1:route_dim
                    global_drive = muladd(
                        parameters.global_feedback_gain[
                            route,
                            local_cell,
                            block,
                        ],
                        scratch.route_context[route, cycle],
                        global_drive,
                    )
                end
                apical_drive +=
                    global_drive / sqrt(Float32(route_dim))
            else
                for channel in 1:readout
                    raw_coordinate =
                        channel +
                        (local_cell - 1) * readout
                    workspace_coordinate = bound_workspace ?
                        Int(tape.spatial_inverse_coordinate[
                            raw_coordinate,
                            block,
                        ]) : raw_coordinate
                    workspace_sign = bound_workspace ?
                        tape.spatial_inverse_sign[
                            raw_coordinate,
                            block,
                        ] : 1.0f0
                    apical_drive = muladd(
                        parameters.feedback_gain[
                            raw_coordinate,
                            block,
                        ],
                        workspace_sign * base.workspace[
                            workspace_coordinate,
                            cycle,
                            flat,
                        ],
                        apical_drive,
                    )
                end
                apical_drive /= bound_workspace ?
                    sqrt(Float32(readout)) : Float32(readout)
            end
            next_apical = muladd(
                cache.apical_leak[cell],
                tape.apical[cell, cycle, flat],
                apical_drive,
            )
            basal = 0.0f0
            for branch in 1:model.branches
                sensory_exc =
                    sensory_cycle_scale *
                    Model.reduced_hay_branch_bias(
                        model,
                        parameters.branch_bias[branch, cell],
                    )
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
                _apical_activation(model, next_apical)
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
        if silenced_block != 0
            offset = (silenced_block - 1) * node_dim
            for coordinate in 1:node_dim
                base.membrane[offset + coordinate, cycle + 1, flat] = 0.0f0
            end
        end

        if exact_block_slots
            fill!(scratch.route_state, 0.0f0)
            @inbounds for block in 1:model.blocks
                block_offset = (block - 1) * node_dim
                for local_cell in 1:model.cells_per_block
                    cell_offset =
                        block_offset + (local_cell - 1) * readout
                    route_offset =
                        (local_cell - 1) * route_state_rank
                    for state in 1:readout
                        value = base.membrane[
                            cell_offset + state,
                            cycle + 1,
                            flat,
                        ]
                        for rank in 1:route_state_rank
                            route = route_offset + rank
                            scratch.route_state[route, block] = muladd(
                                parameters.route_state_projection[
                                    rank,
                                    state,
                                    local_cell,
                                ],
                                value,
                                scratch.route_state[route, block],
                            )
                        end
                    end
                end
            end
            inverse_blocks = inv(sqrt(Float32(model.blocks)))
            @inbounds for route in 1:route_dim
                query_input = 0.0f0
                for block in 1:model.blocks
                    query_input +=
                        _v11_route_block_sign(model, block, route) *
                        scratch.route_state[route, block]
                end
                scratch.query_input_state[route] =
                    query_input * inverse_blocks
                tape.state_query_pre[route, cycle, flat] = 0.0f0
            end
            @inbounds for input_route in 1:route_dim
                input_value = scratch.query_input_state[input_route]
                for route in 1:route_dim
                    tape.state_query_pre[route, cycle, flat] = muladd(
                        parameters.state_query_weight[
                            route,
                            input_route,
                        ],
                        input_value,
                        tape.state_query_pre[route, cycle, flat],
                    )
                end
            end
            query_square_sum = 0.0f0
            @inbounds for route in 1:route_dim
                value = tape.state_query_pre[route, cycle, flat]
                query_square_sum =
                    muladd(value, value, query_square_sum)
            end
            query_inv_rms = inv(sqrt(
                query_square_sum / Float32(route_dim) +
                InputModel.RMS_NORM_EPS,
            ))
            tape.state_query_inv_rms[cycle, flat] = query_inv_rms
            @inbounds for route in 1:route_dim
                query = tanh(
                    InputModel.QUERY_NORM_SCALE *
                    tape.state_query_pre[route, cycle, flat] *
                    query_inv_rms,
                )
                tape.state_query[route, cycle, flat] = query
                base.query_pre[route, flat] =
                    tape.state_query_pre[route, cycle, flat]
                base.query[route, flat] = query
            end
            base.query_inv_rms[flat] = query_inv_rms
            inverse_route = inv(sqrt(Float32(route_dim)))
            @inbounds for block in 1:model.blocks
                score = 0.0f0
                magnitude = 0.0f0
                for route in 1:route_dim
                    coded_state =
                        _v11_route_block_sign(model, block, route) *
                        scratch.route_state[route, block]
                    score = muladd(
                        coded_state *
                        parameters.workspace_key[route, block],
                        tape.state_query[route, cycle, flat],
                        score,
                    )
                end
                block_offset = (block - 1) * node_dim
                for coordinate in 1:node_dim
                    magnitude += abs(base.membrane[
                        block_offset + coordinate,
                        cycle + 1,
                        flat,
                    ])
                end
                scratch.scores[block] =
                    score * inverse_route +
                    0.05f0 * magnitude / Float32(node_dim)
            end
        else
            block_summary_normalization = bound_workspace ?
                sqrt(Float32(model.blocks)) : Float32(model.blocks)
            for coordinate in 1:node_dim
                global_state = 0.0f0
                for block in 1:model.blocks
                    state_coordinate = bound_workspace ?
                        Int(tape.spatial_bound_coordinate[
                            coordinate,
                            block,
                        ]) : coordinate
                    state_sign = bound_workspace ?
                        tape.spatial_bound_sign[coordinate, block] :
                        1.0f0
                    node = state_coordinate +
                        (block - 1) * node_dim
                    global_state = muladd(
                        state_sign,
                        base.membrane[node, cycle + 1, flat],
                        global_state,
                    )
                end
                scratch.global_block_signal[coordinate] =
                    global_state / block_summary_normalization
            end
            # `state_query_weight` is column-major (`output, input`).  Keep
            # each output's accumulation order unchanged while streaming each
            # contiguous input column.
            for coordinate in 1:node_dim
                tape.state_query_pre[coordinate, cycle, flat] = 0.0f0
            end
            for input_coordinate in 1:node_dim
                input_value = scratch.global_block_signal[input_coordinate]
                for coordinate in 1:node_dim
                    tape.state_query_pre[coordinate, cycle, flat] = muladd(
                        parameters.state_query_weight[
                            coordinate,
                            input_coordinate,
                        ],
                        input_value,
                        tape.state_query_pre[coordinate, cycle, flat],
                    )
                end
            end
            query_square_sum = 0.0f0
            for coordinate in 1:node_dim
                value = tape.state_query_pre[coordinate, cycle, flat]
                query_square_sum = muladd(value, value, query_square_sum)
            end
            query_inv_rms = inv(sqrt(
                query_square_sum / Float32(node_dim) +
                InputModel.RMS_NORM_EPS,
            ))
            tape.state_query_inv_rms[cycle, flat] = query_inv_rms
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
                    state_coordinate = bound_workspace ?
                        Int(tape.spatial_bound_coordinate[
                            coordinate,
                            block,
                        ]) : coordinate
                    state_sign = bound_workspace ?
                        tape.spatial_bound_sign[coordinate, block] :
                        1.0f0
                    state = base.membrane[
                        offset + state_coordinate,
                        cycle + 1,
                        flat,
                    ] * state_sign
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
                scratch.scores[block] = bound_workspace ?
                    score / sqrt(Float32(node_dim)) +
                        0.05f0 * magnitude / Float32(node_dim) :
                    score + 0.05f0 * magnitude
            end
        end
        _apply_route_revisit_policy!(
            scratch,
            tape,
            model,
            cycle,
            flat,
        )
        _prepare_route!(
            scratch,
            model,
            stochastic_routing,
            routing_nonce,
            cycle,
            Float32(routing_temperature),
            Float32(routing_logit_limit),
        )
        if forced_route_order !== nothing &&
           forced_route_order[1, cycle] != 0
            _force_recorded_route!(
                scratch,
                model,
                forced_route_order,
                cycle,
                Float32(routing_temperature),
                Float32(routing_logit_limit),
            )
        end
        _record_route!(tape, scratch, model, cycle, flat)

        if exact_block_slots
            inverse_selected = inv(sqrt(Float32(model.workspace_k)))
            @inbounds for route in 1:route_dim
                context = 0.0f0
                for rank in 1:model.workspace_k
                    block = Int(base.route_order[rank, cycle, flat])
                    context +=
                        _v11_route_block_sign(model, block, route) *
                        scratch.route_state[route, block]
                end
                scratch.route_context[route, cycle + 1] =
                    context * inverse_selected
            end
        end

        for cell in 1:cells
            block = div(cell - 1, model.cells_per_block) + 1
            tape.cell_spikes[cell, cycle, flat] =
                block == silenced_block ? 0.0f0 :
                tape.soma_spikes[cell, cycle, flat] *
                base.block_mask[block, cycle, flat]
        end
        if exact_block_slots
            @inbounds for block in 1:model.blocks
                selected = base.block_mask[block, cycle, flat] != 0.0f0
                block_offset = (block - 1) * node_dim
                for coordinate in 1:node_dim
                    scratch.exact_workspace[
                        coordinate,
                        block,
                        cycle + 1,
                    ] = selected ?
                        base.membrane[
                            block_offset + coordinate,
                            cycle + 1,
                            flat,
                        ] :
                        cache.workspace_decay *
                        scratch.exact_workspace[
                            coordinate,
                            block,
                            cycle,
                        ]
                end
            end
        else
            for coordinate in 1:node_dim
                write = 0.0f0
                for block in 1:model.blocks
                    state_coordinate = bound_workspace ?
                        Int(tape.spatial_bound_coordinate[
                            coordinate,
                            block,
                        ]) : coordinate
                    state_sign = bound_workspace ?
                        tape.spatial_bound_sign[coordinate, block] :
                        1.0f0
                    node = state_coordinate +
                        (block - 1) * node_dim
                    write = muladd(
                        state_sign * base.membrane[
                            node,
                            cycle + 1,
                            flat,
                        ],
                        base.block_mask[block, cycle, flat],
                        write,
                    )
                end
                if bound_workspace
                    write /= sqrt(Float32(model.workspace_k))
                    base.workspace[coordinate, cycle + 1, flat] =
                        cache.workspace_decay *
                        base.workspace[coordinate, cycle, flat] +
                        (1.0f0 - cache.workspace_decay) * write
                else
                    write /= Float32(model.workspace_k)
                    base.workspace[coordinate, cycle + 1, flat] = tanh(
                        cache.workspace_decay *
                        base.workspace[coordinate, cycle, flat] +
                        write,
                    )
                end
            end
        end
        if anchored_temporal && !exact_block_slots
            inverse_cycle_scale = inv(sqrt(Float32(model.cycles)))
            for coordinate in 1:node_dim
                cycle == 1 && (
                    tape.sensory_anchor[coordinate, flat] =
                        scratch.global_block_signal[coordinate]
                )
                source_coordinate = Int(
                    tape.temporal_bound_coordinate[
                        coordinate,
                        cycle,
                    ],
                )
                tape.temporal_workspace[coordinate, flat] +=
                    tape.temporal_bound_sign[coordinate, cycle] *
                    base.workspace[
                        source_coordinate,
                        cycle + 1,
                        flat,
                    ] * inverse_cycle_scale
            end
        end
    end

    if exact_block_slots
        anchor_time = 2
        final_time = model.cycles + 1
        @inbounds for output in 1:OUTPUT_DIM
            base.raw[output, flat] = parameters.output_bias[output]
        end
        @inbounds for block in 1:model.blocks
            for local_cell in 1:model.cells_per_block
                if direct_axis_head
                    _v13_accumulate_direct_axis_head!(
                        base.raw,
                        flat,
                        tape,
                        model,
                        parameters.head_anchor_mix,
                        block,
                        local_cell,
                        anchor_time,
                    )
                    _v13_accumulate_direct_axis_head!(
                        base.raw,
                        flat,
                        tape,
                        model,
                        parameters.head_delta_mix,
                        block,
                        local_cell,
                        final_time,
                        anchor_time,
                    )
                else
                    _v11_axis_project_cell!(
                        scratch.axis_projection,
                        tape,
                        model,
                        parameters,
                        block,
                        local_cell,
                        anchor_time,
                        flat,
                    )
                    _v11_accumulate_axis_head!(
                        base.raw,
                        flat,
                        scratch.axis_projection,
                        parameters.head_anchor_mix,
                        block,
                        local_cell,
                    )
                    _v11_axis_project_cell!(
                        scratch.axis_projection,
                        tape,
                        model,
                        parameters,
                        block,
                        local_cell,
                        final_time,
                        flat,
                        anchor_time,
                    )
                    _v11_accumulate_axis_head!(
                        base.raw,
                        flat,
                        scratch.axis_projection,
                        parameters.head_delta_mix,
                        block,
                        local_cell,
                    )
                end
            end
        end
        # The trajectory term is sparse by construction: only the T*K route
        # events are visited, while block identity and cycle identity remain
        # explicit axes in the mixing tensor.
        @inbounds for cycle in 1:model.cycles
            for route_rank in 1:model.workspace_k
                block = Int(base.route_order[route_rank, cycle, flat])
                for local_cell in 1:model.cells_per_block
                    if direct_axis_head
                        _v13_accumulate_direct_history_head!(
                            base.raw,
                            flat,
                            tape,
                            model,
                            parameters.head_history_mix,
                            cycle,
                            block,
                            local_cell,
                        )
                    else
                        _v11_axis_project_cell!(
                            scratch.axis_projection,
                            tape,
                            model,
                            parameters,
                            block,
                            local_cell,
                            cycle + 1,
                            flat,
                        )
                        _v11_accumulate_history_head!(
                            base.raw,
                            flat,
                            scratch.axis_projection,
                            parameters.head_history_mix,
                            cycle,
                            block,
                            local_cell,
                        )
                    end
                end
            end
        end
        return nothing
    end

    feature_dim = Model.reduced_hay_head_feature_dim(model)
    if anchored_temporal
        anchor_square_sum = 0.0f0
        temporal_square_sum = 0.0f0
        delta_square_sum = 0.0f0
        block_normalization = sqrt(Float32(model.blocks))
        @inbounds for coordinate in 1:node_dim
            final_summary = 0.0f0
            for block in 1:model.blocks
                state_coordinate = Int(
                    tape.spatial_bound_coordinate[
                        coordinate,
                        block,
                    ],
                )
                final_summary = muladd(
                    tape.spatial_bound_sign[coordinate, block],
                    base.membrane[
                        state_coordinate +
                            (block - 1) * node_dim,
                        model.cycles + 1,
                        flat,
                    ],
                    final_summary,
                )
            end
            final_summary /= block_normalization
            anchor = tape.sensory_anchor[coordinate, flat]
            temporal = tape.temporal_workspace[coordinate, flat]
            delta = final_summary - anchor
            tape.anchor_delta[coordinate, flat] = delta
            anchor_square_sum = muladd(
                anchor,
                anchor,
                anchor_square_sum,
            )
            temporal_square_sum = muladd(
                temporal,
                temporal,
                temporal_square_sum,
            )
            delta_square_sum = muladd(
                delta,
                delta,
                delta_square_sum,
            )
        end
        anchor_inv_rms = inv(sqrt(
            anchor_square_sum / Float32(node_dim) +
            InputModel.RMS_NORM_EPS,
        ))
        temporal_inv_rms = inv(sqrt(
            temporal_square_sum / Float32(node_dim) +
            InputModel.RMS_NORM_EPS,
        ))
        delta_inv_rms = inv(sqrt(
            delta_square_sum / Float32(node_dim) +
            InputModel.RMS_NORM_EPS,
        ))
        tape.sensory_anchor_inv_rms[flat] = anchor_inv_rms
        tape.temporal_workspace_inv_rms[flat] = temporal_inv_rms
        tape.anchor_delta_inv_rms[flat] = delta_inv_rms
        base.workspace_inv_rms[flat] = anchor_inv_rms
        base.selected_pool_inv_rms[flat] = temporal_inv_rms
        @inbounds for coordinate in 1:node_dim
            scratch.point_scratch.features[coordinate] =
                tape.sensory_anchor[coordinate, flat] *
                anchor_inv_rms
            scratch.point_scratch.features[
                node_dim + coordinate
            ] = tape.temporal_workspace[coordinate, flat] *
                temporal_inv_rms
            scratch.point_scratch.features[
                2node_dim + coordinate
            ] = tape.anchor_delta[coordinate, flat] *
                delta_inv_rms
        end
    else
        local_feature_dim = feature_dim - node_dim
        workspace_square_sum = 0.0f0
        local_square_sum = 0.0f0
        @inbounds for coordinate in 1:node_dim
            workspace_value =
                base.workspace[coordinate, model.cycles + 1, flat]
            scratch.point_scratch.features[coordinate] =
                workspace_value
            workspace_square_sum = muladd(
                workspace_value,
                workspace_value,
                workspace_square_sum,
            )
        end
        if model.head_readout === :pooled
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
                        base.block_mask[
                            block,
                            model.cycles,
                            flat,
                        ],
                        selected_pool,
                    )
                end
                selected_pool /= Float32(model.workspace_k)
                scratch.point_scratch.features[
                    node_dim + coordinate
                ] = selected_pool
                local_square_sum = muladd(
                    selected_pool,
                    selected_pool,
                    local_square_sum,
                )
            end
        else
            @inbounds for rank in 1:model.workspace_k
                block = Int(
                    base.route_order[rank, model.cycles, flat],
                )
                block_offset = (block - 1) * node_dim
                feature_offset = node_dim * rank
                for coordinate in 1:node_dim
                    value = base.membrane[
                        block_offset + coordinate,
                        model.cycles + 1,
                        flat,
                    ]
                    scratch.point_scratch.features[
                        feature_offset + coordinate
                    ] = value
                    local_square_sum = muladd(
                        value,
                        value,
                        local_square_sum,
                    )
                end
            end
        end
        workspace_inv_rms = inv(sqrt(
            workspace_square_sum / Float32(node_dim) +
            InputModel.RMS_NORM_EPS,
        ))
        local_inv_rms = inv(sqrt(
            local_square_sum / Float32(local_feature_dim) +
            InputModel.RMS_NORM_EPS,
        ))
        base.workspace_inv_rms[flat] = workspace_inv_rms
        base.selected_pool_inv_rms[flat] = local_inv_rms
        @inbounds for coordinate in 1:node_dim
            scratch.point_scratch.features[coordinate] *=
                workspace_inv_rms
        end
        @inbounds for feature in (node_dim + 1):feature_dim
            scratch.point_scratch.features[feature] *= local_inv_rms
        end
    end
    @inbounds for hidden in 1:model.hidden
        base.hidden_pre[hidden, flat] = parameters.head_bias[hidden]
    end
    # `head_weight` is (`hidden, feature`), so feature-major traversal streams
    # contiguous columns.  A fixed hidden unit still accumulates features in
    # exactly the original order.
    @inbounds for feature in 1:feature_dim
        feature_value = scratch.point_scratch.features[feature]
        for hidden in 1:model.hidden
            base.hidden_pre[hidden, flat] = muladd(
                parameters.head_weight[hidden, feature],
                feature_value,
                base.hidden_pre[hidden, flat],
            )
        end
    end
    hidden_square_sum = 0.0f0
    @inbounds for hidden in 1:model.hidden
        activation = base.hidden_pre[hidden, flat]
        hidden_square_sum = muladd(activation, activation, hidden_square_sum)
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
        base.raw[output, flat] = parameters.output_bias[output]
    end
    @inbounds for hidden in 1:model.hidden
        hidden_value = base.hidden[hidden, flat]
        for output in 1:OUTPUT_DIM
            base.raw[output, flat] = muladd(
                parameters.output_weight[output, hidden],
                hidden_value,
                base.raw[output, flat],
            )
        end
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
    local_loss = 0.0f0

    q_residual =
        prediction[1] -
        targets.teacher_q[candidate, state_slot]
    error[1] =
        0.25f0 *
        _huber_derivative(q_residual) *
        inverse_valid
    local_loss +=
        0.25f0 * _huber_loss(q_residual) * inverse_valid
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
        local_loss += 0.10f0 * (
            max(prediction[2], 0.0f0) -
            death_target * prediction[2] +
            log1p(exp(-abs(prediction[2])))
        ) * inverse_valid
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
        local_loss +=
            0.05f0 *
            quantile_weight *
            _huber_loss(quantile_error) *
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
        local_loss +=
            LOCAL_GEOMETRY_WEIGHT *
            _huber_loss(residual) *
            inverse_valid
        record_metrics &&
            (scratch.local_geometry_loss +=
                _huber_loss(residual) / 4.0f0)
    end
    return local_loss
end

function _prepare_block_signals!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    projection::Array{Float32,3},
    model,
    parameters,
    flat::Int,
    record_metrics::Bool,
    global_signal_scale::Float32,
    local_signal_scale::Float32,
)
    base = tape.base
    _reset_export_signals!(scratch)
    inverse_node_dim = inv(Float32(model.node_dim))
    inverse_cycles = inv(Float32(model.cycles))

    @inbounds for cycle in 1:model.cycles
        for block in 1:model.blocks
            offset = (block - 1) * model.node_dim
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
                spike = tape.cell_spikes[cell, cycle, flat]
                scratch.block_signal[coordinate] =
                    state +
                    LOCAL_PREDICTOR_SPIKE_SCALE * spike
            end
            for output in 1:OUTPUT_DIM
                prediction =
                    parameters.local_readout_bias[output, block]
                for coordinate in 1:model.node_dim
                    prediction = muladd(
                        parameters.local_readout[
                            coordinate,
                            output,
                            block,
                        ],
                        scratch.block_signal[coordinate],
                        prediction,
                    )
                end
                scratch.local_prediction[output] = prediction
            end
            local_loss = _prepare_local_error!(
                scratch,
                base,
                flat,
                record_metrics,
            )
            for output in 1:OUTPUT_DIM
                predictor_error =
                    scratch.local_error[output] * inverse_cycles
                scratch.gradient.local_readout_bias[output, block] +=
                    predictor_error
                for coordinate in 1:model.node_dim
                    scratch.gradient.local_readout[
                        coordinate,
                        output,
                        block,
                    ] = muladd(
                        predictor_error,
                        scratch.block_signal[coordinate],
                        scratch.gradient.local_readout[
                            coordinate,
                            output,
                            block,
                        ],
                    )
                end
            end
            fill!(scratch.global_block_signal, 0.0f0)
            fill!(scratch.local_block_signal, 0.0f0)
            global_first_order = 0.0f0
            for output in 1:OUTPUT_DIM
                global_error = base.raw_gradient[output, flat]
                local_error = scratch.local_error[output]
                fixed_prediction = 0.0f0
                for coordinate in 1:model.node_dim
                    fixed_prediction = muladd(
                        projection[coordinate, output, block],
                        scratch.block_signal[coordinate],
                        fixed_prediction,
                    )
                end
                global_first_order = muladd(
                    global_error,
                    fixed_prediction,
                    global_first_order,
                )
                for coordinate in 1:model.node_dim
                    fixed_readout = projection[
                        coordinate,
                        output,
                        block,
                    ]
                    local_readout = parameters.local_readout[
                        coordinate,
                        output,
                        block,
                    ]
                    scratch.global_block_signal[coordinate] = muladd(
                        fixed_readout,
                        global_error,
                        scratch.global_block_signal[coordinate],
                    )
                    scratch.local_block_signal[coordinate] = muladd(
                        local_readout,
                        local_error,
                        scratch.local_block_signal[coordinate],
                    )
                end
            end

            global_square_sum = 0.0f0
            local_square_sum = 0.0f0
            for coordinate in 1:model.node_dim
                global_value = scratch.global_block_signal[coordinate]
                local_value = scratch.local_block_signal[coordinate]
                global_square_sum = muladd(
                    global_value,
                    global_value,
                    global_square_sum,
                )
                local_square_sum = muladd(
                    local_value,
                    local_value,
                    local_square_sum,
                )
            end
            # Clip unusually large third factors without amplifying small
            # errors.  Unit-RMS normalization kept e-prop updates finite even
            # as the supervised error approached zero, which prevented exact
            # memorization and continuously displaced a converged head.
            global_inverse_rms = min(
                1.0f0,
                inv(sqrt(
                    global_square_sum * inverse_node_dim +
                    LOCAL_SIGNAL_RMS_EPSILON,
                )),
            )
            local_inverse_rms = min(
                1.0f0,
                inv(sqrt(
                    local_square_sum * inverse_node_dim +
                    LOCAL_SIGNAL_RMS_EPSILON,
                )),
            )
            for coordinate in 1:model.node_dim
                scratch.block_signal[coordinate] =
                    global_signal_scale *
                    scratch.global_block_signal[coordinate] *
                    global_inverse_rms +
                    local_signal_scale *
                    scratch.local_block_signal[coordinate] *
                    local_inverse_rms
                signal = scratch.block_signal[coordinate] * inverse_cycles
                _scatter_export_cotangent!(
                    scratch,
                    tape,
                    model,
                    coordinate,
                    block,
                    cycle,
                    flat,
                    signal,
                )
            end
            tape.block_supervised_reward[
                block,
                cycle,
                flat,
            ] = -(
                global_signal_scale * global_first_order +
                local_signal_scale * local_loss
            )
        end
    end
    return nothing
end

@inline function _v11_accumulate_route_score_vjp!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    parameters,
    flat::Int,
    cycle::Int;
    accumulate_parameter_gradient::Bool=true,
    accumulate_state_cotangent::Bool=true,
)
    base = tape.base
    route_dim = _arena_route_dim(model)
    route_rank = div(route_dim, model.cells_per_block)
    inverse_route = inv(sqrt(Float32(route_dim)))
    inverse_blocks = inv(sqrt(Float32(model.blocks)))
    inverse_node_dim = inv(Float32(model.node_dim))

    # In stochastic mode route-score parameter gradients are supplied by the
    # ordered Plackett-Luce estimator.  The selected route context is still a
    # continuous payload, however, so its already accumulated cotangent must
    # update P_route even when score/query parameter gradients are suppressed.
    if !accumulate_parameter_gradient && accumulate_state_cotangent
        @inbounds for block in 1:model.blocks
            block_offset = (block - 1) * model.node_dim
            for local_cell in 1:model.cells_per_block
                cell_offset = block_offset +
                    (local_cell - 1) * model.readout_per_cell
                route_offset = (local_cell - 1) * route_rank
                for rank in 1:route_rank
                    signal = scratch.route_state_signal[
                        route_offset + rank,
                        block,
                    ]
                    for state in 1:model.readout_per_cell
                        scratch.gradient.route_state_projection[
                            rank,
                            state,
                            local_cell,
                        ] = muladd(
                            signal,
                            base.membrane[
                                cell_offset + state,
                                cycle + 1,
                                flat,
                            ],
                            scratch.gradient.route_state_projection[
                                rank,
                                state,
                                local_cell,
                            ],
                        )
                    end
                end
            end
        end
    end

    fill!(scratch.route_query_signal, 0.0f0)
    @inbounds for block in 1:model.blocks
        route_factor = scratch.route_eligibility[block]
        block_offset = (block - 1) * model.node_dim
        for route in 1:route_dim
            role = _v11_route_block_sign(model, block, route)
            route_state = scratch.route_state[route, block]
            coded_state = role * route_state
            query = tape.state_query[route, cycle, flat]
            key = parameters.workspace_key[route, block]
            score_scale = route_factor * inverse_route
            if accumulate_parameter_gradient
                scratch.gradient.workspace_key[route, block] = muladd(
                    score_scale * coded_state,
                    query,
                    scratch.gradient.workspace_key[route, block],
                )
            end
            scratch.route_query_signal[route] = muladd(
                score_scale * coded_state,
                key,
                scratch.route_query_signal[route],
            )
            scratch.route_state_signal[route, block] = muladd(
                score_scale * role * key,
                query,
                scratch.route_state_signal[route, block],
            )
        end
        if accumulate_state_cotangent && route_factor != 0.0f0
            magnitude_scale = route_factor * 0.05f0 * inverse_node_dim
            for coordinate in 1:model.node_dim
                value = base.membrane[
                    block_offset + coordinate,
                    cycle + 1,
                    flat,
                ]
                value == 0.0f0 && continue
                scratch.feedback_error[coordinate, block] +=
                    magnitude_scale * sign(value)
            end
        end
    end

    projection_mean = 0.0f0
    @inbounds for route in 1:route_dim
        query = tape.state_query[route, cycle, flat]
        normalized = scratch.route_query_signal[route] *
            InputModel.QUERY_NORM_SCALE * (1.0f0 - query * query)
        scratch.route_query_signal[route] = normalized
        projection_mean = muladd(
            normalized,
            tape.state_query_pre[route, cycle, flat],
            projection_mean,
        )
    end
    projection_mean /= Float32(route_dim)
    inverse_rms = tape.state_query_inv_rms[cycle, flat]
    inverse_rms_squared = inverse_rms * inverse_rms
    @inbounds for route in 1:route_dim
        scratch.route_query_signal[route] = inverse_rms * (
            scratch.route_query_signal[route] -
            tape.state_query_pre[route, cycle, flat] *
            inverse_rms_squared * projection_mean
        )
    end

    # Reconstruct the exact query input from the current route projection.
    @inbounds for input_route in 1:route_dim
        value = 0.0f0
        for block in 1:model.blocks
            value = muladd(
                _v11_route_block_sign(model, block, input_route),
                scratch.route_state[input_route, block],
                value,
            )
        end
        scratch.query_input_state[input_route] = value * inverse_blocks
    end
    @inbounds for input_route in 1:route_dim
        input_value = scratch.query_input_state[input_route]
        input_signal = 0.0f0
        for route in 1:route_dim
            signal = scratch.route_query_signal[route]
            if accumulate_parameter_gradient
                scratch.gradient.state_query_weight[
                    route,
                    input_route,
                ] = muladd(
                    signal,
                    input_value,
                    scratch.gradient.state_query_weight[
                        route,
                        input_route,
                    ],
                )
            end
            input_signal = muladd(
                parameters.state_query_weight[route, input_route],
                signal,
                input_signal,
            )
        end
        scratch.global_context_signal[input_route] = input_signal
    end
    @inbounds for block in 1:model.blocks
        for route in 1:route_dim
            scratch.route_state_signal[route, block] = muladd(
                _v11_route_block_sign(model, block, route) * inverse_blocks,
                scratch.global_context_signal[route],
                scratch.route_state_signal[route, block],
            )
        end
    end

    # Return the complete route-state cotangent through the shared per-cell
    # projection.  It includes global-context, score and query paths exactly
    # once.  The caller scatters feedback_error into the full24 cell adjoint.
    @inbounds for block in 1:model.blocks
        block_offset = (block - 1) * model.node_dim
        for local_cell in 1:model.cells_per_block
            cell_offset = block_offset +
                (local_cell - 1) * model.readout_per_cell
            route_offset = (local_cell - 1) * route_rank
            for rank in 1:route_rank
                signal = scratch.route_state_signal[
                    route_offset + rank,
                    block,
                ]
                for state in 1:model.readout_per_cell
                    value = base.membrane[
                        cell_offset + state,
                        cycle + 1,
                        flat,
                    ]
                    if accumulate_parameter_gradient
                        scratch.gradient.route_state_projection[
                            rank,
                            state,
                            local_cell,
                        ] = muladd(
                            signal,
                            value,
                            scratch.gradient.route_state_projection[
                                rank,
                                state,
                                local_cell,
                            ],
                        )
                    end
                    if accumulate_state_cotangent
                        coordinate =
                            (local_cell - 1) * model.readout_per_cell + state
                        scratch.feedback_error[coordinate, block] = muladd(
                            parameters.route_state_projection[
                                rank,
                                state,
                                local_cell,
                            ],
                            signal,
                            scratch.feedback_error[coordinate, block],
                        )
                    end
                end
            end
        end
    end
    return nothing
end

@inline function _v11_accumulate_pathwise_route_cycle!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    parameters,
    flat::Int,
    cycle::Int,
    routing_temperature::Float32,
    routing_logit_limit::Float32,
    accumulate_parameter_gradient::Bool,
)
    base = tape.base
    inverse_blocks = inv(Float32(model.blocks))
    score_mean = 0.0f0
    expected_mask_cotangent = 0.0f0
    @inbounds for block in 1:model.blocks
        score_mean += base.route_score[block, cycle, flat]
        expected_mask_cotangent = muladd(
            base.route_base_probability[block, cycle, flat],
            scratch.route_alpha[block],
            expected_mask_cotangent,
        )
    end
    score_mean *= inverse_blocks
    score_square_sum = 0.0f0
    @inbounds for block in 1:model.blocks
        centered = base.route_score[block, cycle, flat] - score_mean
        scratch.route_standardized[block] = centered
        score_square_sum = muladd(centered, centered, score_square_sum)
    end
    score_inv_rms = inv(sqrt(
        score_square_sum * inverse_blocks + InputModel.RMS_NORM_EPS,
    ))
    inverse_temperature = inv(routing_temperature)
    normalized_mean = 0.0f0
    normalized_projection_mean = 0.0f0
    @inbounds for block in 1:model.blocks
        raw_standardized =
            scratch.route_standardized[block] * score_inv_rms
        scratch.route_standardized[block] = raw_standardized
        normalized = Routing.bounded_standardized_derivative(
            raw_standardized,
            routing_logit_limit,
        ) * Float32(model.workspace_k) *
            base.route_base_probability[block, cycle, flat] *
            (scratch.route_alpha[block] - expected_mask_cotangent) *
            inverse_temperature
        scratch.route_eligibility[block] = normalized
        normalized_mean += normalized
        normalized_projection_mean = muladd(
            normalized,
            raw_standardized,
            normalized_projection_mean,
        )
    end
    normalized_mean *= inverse_blocks
    normalized_projection_mean *= inverse_blocks
    @inbounds for block in 1:model.blocks
        scratch.route_eligibility[block] = score_inv_rms * (
            scratch.route_eligibility[block] - normalized_mean -
            scratch.route_standardized[block] * normalized_projection_mean
        )
    end
    _v11_accumulate_route_score_vjp!(
        scratch,
        tape,
        model,
        parameters,
        flat,
        cycle;
        accumulate_parameter_gradient,
        accumulate_state_cotangent=true,
    )
    return nothing
end

@inline function _accumulate_pathwise_route_cycle!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    parameters,
    flat::Int,
    cycle::Int,
    routing_temperature::Float32,
    routing_logit_limit::Float32=Routing.DEFAULT_LOGIT_LIMIT,
    accumulate_parameter_gradient::Bool=true,
)
    if _uses_exact_block_slots(model)
        return _v11_accumulate_pathwise_route_cycle!(
            scratch,
            tape,
            model,
            parameters,
            flat,
            cycle,
            routing_temperature,
            routing_logit_limit,
            accumulate_parameter_gradient,
        )
    end
    base = tape.base
    blocks_f = Float32(model.blocks)
    inverse_blocks = inv(blocks_f)
    bound_workspace = model.workspace_binding !== :none
    score_dot_scale = bound_workspace ?
        inv(sqrt(Float32(model.node_dim))) : 1.0f0
    score_magnitude_scale = bound_workspace ?
        0.05f0 / Float32(model.node_dim) : 0.05f0
    query_input_scale = bound_workspace ?
        inv(sqrt(blocks_f)) : inverse_blocks

    # Straight-through fixed-mass softmax VJP.  The hard (possibly sampled)
    # mask remains the forward route, while this path supplies a low-variance
    # local derivative of the actual workspace write.  route_alpha enters as
    # dL/d(mask_b); route_eligibility leaves as dL/d(raw_score_b).
    score_mean = 0.0f0
    expected_mask_cotangent = 0.0f0
    @inbounds for block in 1:model.blocks
        score_mean += base.route_score[block, cycle, flat]
        expected_mask_cotangent = muladd(
            base.route_base_probability[block, cycle, flat],
            scratch.route_alpha[block],
            expected_mask_cotangent,
        )
    end
    score_mean *= inverse_blocks
    score_square_sum = 0.0f0
    @inbounds for block in 1:model.blocks
        centered = base.route_score[block, cycle, flat] - score_mean
        scratch.route_standardized[block] = centered
        score_square_sum = muladd(centered, centered, score_square_sum)
    end
    score_inv_rms = inv(sqrt(
        score_square_sum * inverse_blocks + InputModel.RMS_NORM_EPS,
    ))
    inverse_temperature = inv(routing_temperature)
    normalized_mean = 0.0f0
    normalized_projection_mean = 0.0f0
    @inbounds for block in 1:model.blocks
        raw_standardized =
            scratch.route_standardized[block] * score_inv_rms
        scratch.route_standardized[block] = raw_standardized
        bounded_derivative =
            Routing.bounded_standardized_derivative(
                raw_standardized,
                routing_logit_limit,
            )
        normalized = bounded_derivative * Float32(model.workspace_k) *
            base.route_base_probability[block, cycle, flat] *
            (scratch.route_alpha[block] - expected_mask_cotangent) *
            inverse_temperature
        scratch.route_eligibility[block] = normalized
        normalized_mean += normalized
        normalized_projection_mean = muladd(
            normalized,
            raw_standardized,
            normalized_projection_mean,
        )
    end
    normalized_mean *= inverse_blocks
    normalized_projection_mean *= inverse_blocks
    @inbounds for block in 1:model.blocks
        scratch.route_eligibility[block] = score_inv_rms * (
            scratch.route_eligibility[block] -
            normalized_mean -
            scratch.route_standardized[block] *
            normalized_projection_mean
        )
    end

    fill!(scratch.point_scratch.dquery, 0.0f0)
    fill!(scratch.block_signal, 0.0f0)
    @inbounds for block in 1:model.blocks
        route_factor = scratch.route_eligibility[block]
        offset = (block - 1) * model.node_dim
        for coordinate in 1:model.node_dim
            state_coordinate = bound_workspace ?
                Int(tape.spatial_bound_coordinate[
                    coordinate,
                    block,
                ]) : coordinate
            state_sign = bound_workspace ?
                tape.spatial_bound_sign[coordinate, block] : 1.0f0
            state = state_sign * base.membrane[
                offset + state_coordinate, cycle + 1, flat,
            ]
            query = tape.state_query[coordinate, cycle, flat]
            key = parameters.workspace_key[coordinate, block]
            if accumulate_parameter_gradient
                scratch.gradient.workspace_key[coordinate, block] = muladd(
                    route_factor * state * score_dot_scale,
                    query,
                    scratch.gradient.workspace_key[coordinate, block],
                )
            end
            scratch.point_scratch.dquery[coordinate] = muladd(
                route_factor * state * score_dot_scale,
                key,
                scratch.point_scratch.dquery[coordinate],
            )
            score_state_derivative =
                key * query * score_dot_scale
            state != 0.0f0 &&
                (score_state_derivative +=
                    score_magnitude_scale * sign(state))
            scratch.feedback_error[state_coordinate, block] = muladd(
                route_factor,
                state_sign * score_state_derivative,
                scratch.feedback_error[state_coordinate, block],
            )
        end
    end

    query_projection_mean = 0.0f0
    @inbounds for coordinate in 1:model.node_dim
        query = tape.state_query[coordinate, cycle, flat]
        normalized = scratch.point_scratch.dquery[coordinate] *
            InputModel.QUERY_NORM_SCALE * (1.0f0 - query * query)
        scratch.point_scratch.dquery[coordinate] = normalized
        query_projection_mean = muladd(
            normalized,
            tape.state_query_pre[coordinate, cycle, flat],
            query_projection_mean,
        )
    end
    query_projection_mean *= inv(Float32(model.node_dim))
    inverse_rms = tape.state_query_inv_rms[cycle, flat]
    inverse_rms_squared = inverse_rms * inverse_rms
    # The query input is independent of the output coordinate.  Cache the
    # bound all-block summary once per cycle instead of rebuilding it inside
    # every row of the dense query VJP (O(D*B + D^2), not O(D^2*B)).  A
    # dedicated buffer avoids aliasing `global_block_signal`, which carries
    # the legacy workspace root between reverse cycles.
    @inbounds for input_coordinate in 1:model.node_dim
        global_state = 0.0f0
        for block in 1:model.blocks
            state_coordinate = bound_workspace ?
                Int(tape.spatial_bound_coordinate[
                    input_coordinate,
                    block,
                ]) : input_coordinate
            state_sign = bound_workspace ?
                tape.spatial_bound_sign[input_coordinate, block] :
                1.0f0
            node = state_coordinate +
                (block - 1) * model.node_dim
            global_state = muladd(
                state_sign,
                base.membrane[node, cycle + 1, flat],
                global_state,
            )
        end
        scratch.query_input_state[input_coordinate] =
            global_state * query_input_scale
    end
    @inbounds for coordinate in 1:model.node_dim
        scratch.point_scratch.dquery[coordinate] = inverse_rms * (
            scratch.point_scratch.dquery[coordinate] -
            tape.state_query_pre[coordinate, cycle, flat] *
            inverse_rms_squared * query_projection_mean
        )
    end
    # Julia matrices are column-major.  Keeping input_coordinate (the second
    # matrix index) outside makes both the parameter and gradient scans
    # contiguous while producing exactly the same outer product.
    @inbounds for input_coordinate in 1:model.node_dim
        global_state = scratch.query_input_state[input_coordinate]
        state_cotangent = 0.0f0
        for coordinate in 1:model.node_dim
            cotangent = scratch.point_scratch.dquery[coordinate]
            if accumulate_parameter_gradient
                scratch.gradient.state_query_weight[
                    coordinate, input_coordinate,
                ] = muladd(
                    cotangent,
                    global_state,
                    scratch.gradient.state_query_weight[
                        coordinate, input_coordinate,
                    ],
                )
            end
            state_cotangent = muladd(
                parameters.state_query_weight[
                    coordinate, input_coordinate,
                ],
                cotangent,
                state_cotangent,
            )
        end
        scratch.block_signal[input_coordinate] = state_cotangent
    end
    @inbounds for block in 1:model.blocks
        for coordinate in 1:model.node_dim
            state_coordinate = bound_workspace ?
                Int(tape.spatial_bound_coordinate[
                    coordinate,
                    block,
                ]) : coordinate
            state_sign = bound_workspace ?
                tape.spatial_bound_sign[coordinate, block] : 1.0f0
            scratch.feedback_error[state_coordinate, block] +=
                state_sign * scratch.block_signal[coordinate] *
                query_input_scale
        end
    end
    return nothing
end

@inline function _v11_accumulate_pl_routing_gradients!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    parameters,
    flat::Int,
    use_supervised_advantage::Bool,
)
    base = tape.base
    @inbounds for cycle in 1:model.cycles
        # Local/score-function replay can run on a different worker from the
        # forward pass, so reconstruct the route projection from the tape.
        _replay_v11_route_state_cycle!(
            scratch,
            tape,
            model,
            parameters,
            flat,
            cycle,
        )
        fill!(scratch.route_state_signal, 0.0f0)
        for block in 1:model.blocks
            # `route_eligibility` is the derivative of the complete ordered
            # Plackett-Luce log probability with respect to this raw score.
            # The reward is the supervised state-level surrogate, never an
            # environment return or a private block target.
            scratch.route_eligibility[block] =
                (use_supervised_advantage ?
                 -tape.block_advantage[block, cycle, flat] *
                 base.route_eligibility[block, cycle, flat] : 0.0f0) +
                base.route_regularizer_gradient[block, cycle, flat]
        end
        _v11_accumulate_route_score_vjp!(
            scratch,
            tape,
            model,
            parameters,
            flat,
            cycle;
            accumulate_parameter_gradient=true,
            accumulate_state_cotangent=false,
        )
    end
    return nothing
end

@inline function _accumulate_routing_gradients!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    parameters,
    flat::Int,
    use_supervised_advantage::Bool=true,
)
    if _uses_exact_block_slots(model)
        return _v11_accumulate_pl_routing_gradients!(
            scratch,
            tape,
            model,
            parameters,
            flat,
            use_supervised_advantage,
        )
    end
    base = tape.base
    bound_workspace = model.workspace_binding !== :none
    score_dot_scale = bound_workspace ?
        inv(sqrt(Float32(model.node_dim))) : 1.0f0
    query_input_scale = bound_workspace ?
        inv(sqrt(Float32(model.blocks))) :
        inv(Float32(model.blocks))
    @inbounds for cycle in 1:model.cycles
        fill!(scratch.point_scratch.dquery, 0.0f0)
        for block in 1:model.blocks
            # In stochastic mode block_advantage contains the same scalar
            # state-level supervised reward surrogate for every score
            # component of this ordered route.  It is not an environment
            # return.  AdamW consumes a loss gradient, so the score-function
            # contribution carries a minus.
            route_factor = (use_supervised_advantage ?
                -tape.block_advantage[
                    block,
                    cycle,
                    flat,
                ] *
                base.route_eligibility[
                    block,
                    cycle,
                    flat,
                ] : 0.0f0) +
                base.route_regularizer_gradient[
                    block,
                    cycle,
                    flat,
                ]
            offset = (block - 1) * model.node_dim
            for coordinate in 1:model.node_dim
                state_coordinate = bound_workspace ?
                    Int(tape.spatial_bound_coordinate[
                        coordinate,
                        block,
                    ]) : coordinate
                state_sign = bound_workspace ?
                    tape.spatial_bound_sign[coordinate, block] : 1.0f0
                state = state_sign * base.membrane[
                    offset + state_coordinate,
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
                    route_factor * state * score_dot_scale,
                    query,
                    scratch.gradient.workspace_key[
                        coordinate,
                        block,
                    ],
                )
                scratch.point_scratch.dquery[coordinate] = muladd(
                    route_factor * state * score_dot_scale,
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
        @inbounds for input_coordinate in 1:model.node_dim
            global_state = 0.0f0
            for block in 1:model.blocks
                state_coordinate = bound_workspace ?
                    Int(tape.spatial_bound_coordinate[
                        input_coordinate,
                        block,
                    ]) : input_coordinate
                state_sign = bound_workspace ?
                    tape.spatial_bound_sign[
                        input_coordinate,
                        block,
                    ] : 1.0f0
                node = state_coordinate +
                    (block - 1) * model.node_dim
                global_state = muladd(
                    state_sign,
                    base.membrane[
                        node,
                        cycle + 1,
                        flat,
                    ],
                    global_state,
                )
            end
            scratch.query_input_state[input_coordinate] =
                global_state * query_input_scale
        end
        for coordinate in 1:model.node_dim
            scratch.point_scratch.dquery[coordinate] = inverse_rms * (
                scratch.point_scratch.dquery[coordinate] -
                tape.state_query_pre[
                    coordinate,
                    cycle,
                    flat,
                ] *
                inverse_rms_squared *
                query_projection_mean
            )
        end
        for input_coordinate in 1:model.node_dim
            global_state = scratch.query_input_state[input_coordinate]
            for coordinate in 1:model.node_dim
                cotangent = scratch.point_scratch.dquery[coordinate]
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
    sensory_cycle_scale =
        Model.reduced_hay_sensory_cycle_scale(model)
    sensory_normalization =
        sensory_cycle_scale *
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
                _apical_activation(model, next_apical)
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
                _apical_activation(model, next_apical) *
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
                        sensory_cycle_scale * exc_drive_effect
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

function _accumulate_edge_eligibility!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    parameters,
    cache::DendriticParameterCache,
    branch_for_edge::Matrix{UInt8},
    flat::Int,
)
    # The cell-local temporal adjoint has already converted every exported
    # learning signal into dL/d(E-drive) and dL/d(I-drive) for each branch and
    # cycle.  Replaying source-major edges now gives the exact local
    # eligibility factor without walking the inter-cell graph backwards.  The
    # previous seven-scalar trace followed only the destination branch and
    # omitted paths through soma -> sibling branches -> soma.
    @inbounds for active_index in 1:scratch.active_edge_count
        edge = Int(scratch.active_edges[active_index])
        scratch.active_edge_mask[edge] = false
    end
    scratch.active_edge_count = 0
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
            drive_signal = if recurrent_drive > 0.0f0
                scratch.branch_exc_drive_signal[
                    destination, branch, cycle,
                ]
            elseif recurrent_drive < 0.0f0
                -scratch.branch_inh_drive_signal[
                    destination, branch, cycle,
                ]
            else
                0.5f0 * (
                    scratch.branch_exc_drive_signal[
                        destination, branch, cycle,
                    ] -
                    scratch.branch_inh_drive_signal[
                        destination, branch, cycle,
                    ]
                )
            end
            weight_update = drive_signal * force_weight
            gate_update = drive_signal * force_gate
            delay_update = drive_signal * force_delay
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
                counterfactual_signal = max(
                    abs(scratch.branch_exc_drive_signal[
                        destination, counterfactual_branch, cycle,
                    ]),
                    abs(scratch.branch_inh_drive_signal[
                        destination, counterfactual_branch, cycle,
                    ]),
                )
                scratch.branch_utility[
                    counterfactual_branch,
                    source,
                    relation,
                ] += abs(
                    pre * counterfactual_signal,
                )
            end
        end
    end
    return nothing
end

function _backward_head_candidate!(
    gradient,
    source,
    model,
    parameters,
    scratch::Point.CandidateScratch,
    flat::Int,
)
    base = source isa DendriticTape ? source.base : source
    node_dim = model.node_dim
    feature_dim = Model.reduced_hay_head_feature_dim(model)
    fill!(scratch.dfeatures, 0.0f0)
    fill!(scratch.dhidden, 0.0f0)
    _prepare_root_features!(scratch, source, model, flat)
    @inbounds for output in 1:OUTPUT_DIM
        gradient.output_bias[output] += base.raw_gradient[output, flat]
    end
    # Stream each contiguous `output_weight[:, hidden]` column.  For every
    # hidden unit the output cotangents are still reduced in output order.
    @inbounds for hidden in 1:model.hidden
        hidden_value = base.hidden[hidden, flat]
        hidden_cotangent = 0.0f0
        for output in 1:OUTPUT_DIM
            output_cotangent = base.raw_gradient[output, flat]
            gradient.output_weight[output, hidden] = muladd(
                output_cotangent,
                hidden_value,
                gradient.output_weight[output, hidden],
            )
            hidden_cotangent = muladd(
                parameters.output_weight[output, hidden],
                output_cotangent,
                hidden_cotangent,
            )
        end
        scratch.dhidden[hidden] = hidden_cotangent
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
        scratch.dhidden[hidden] = cotangent
        gradient.head_bias[hidden] += cotangent
    end
    # `head_weight[:, feature]` is contiguous.  Feature-major traversal also
    # preserves the original hidden-order reduction into each feature root.
    @inbounds for feature in 1:feature_dim
        value = scratch.features[feature]
        feature_cotangent = 0.0f0
        for hidden in 1:model.hidden
            hidden_cotangent = scratch.dhidden[hidden]
            gradient.head_weight[hidden, feature] = muladd(
                hidden_cotangent,
                value,
                gradient.head_weight[hidden, feature],
            )
            feature_cotangent = muladd(
                parameters.head_weight[hidden, feature],
                hidden_cotangent,
                feature_cotangent,
            )
        end
        scratch.dfeatures[feature] = feature_cotangent
    end
    return nothing
end

function dendritic_prepare_signal_candidate!(
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
        parameters,
        flat,
        true,
        global_signal_scale,
        local_signal_scale,
    )
    _accumulate_cell_temporal_gradients!(
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
        tape,
        model,
        parameters,
        scratch.point_scratch,
        flat,
    )
    return nothing
end

@inline function _alignment_sign(feature::Int, flat::Int)
    value = UInt64(feature) * UInt64(0x9e3779b97f4a7c15) ⊻
            UInt64(flat) * UInt64(0xbf58476d1ce4e5b9)
    value ⊻= value >> 30
    value *= UInt64(0xbf58476d1ce4e5b9)
    value ⊻= value >> 27
    value *= UInt64(0x94d049bb133111eb)
    value ⊻= value >> 31
    return isodd(value) ? 1.0f0 : -1.0f0
end

function _prepare_root_features!(scratch, source, model, flat::Int)
    tape = source isa DendriticTape ? source : nothing
    base = tape === nothing ? source : tape.base
    node_dim = model.node_dim
    final_time = model.cycles + 1
    if model.head_readout === :anchored_temporal
        tape === nothing && error(
            "anchored-temporal features require the dendritic tape",
        )
        @inbounds for coordinate in 1:node_dim
            scratch.features[coordinate] =
                tape.sensory_anchor[coordinate, flat] *
                tape.sensory_anchor_inv_rms[flat]
            scratch.features[node_dim + coordinate] =
                tape.temporal_workspace[coordinate, flat] *
                tape.temporal_workspace_inv_rms[flat]
            scratch.features[2node_dim + coordinate] =
                tape.anchor_delta[coordinate, flat] *
                tape.anchor_delta_inv_rms[flat]
        end
        return nothing
    end
    @inbounds for coordinate in 1:node_dim
        scratch.features[coordinate] =
            base.workspace[coordinate, final_time, flat] *
            base.workspace_inv_rms[flat]
    end
    if model.head_readout === :pooled
        @inbounds for coordinate in 1:node_dim
            selected_pool = 0.0f0
            for block in 1:model.blocks
                node = coordinate + (block - 1) * node_dim
                selected_pool = muladd(
                    base.membrane[node, final_time, flat],
                    base.block_mask[block, model.cycles, flat],
                    selected_pool,
                )
            end
            scratch.features[node_dim + coordinate] =
                selected_pool / Float32(model.workspace_k) *
                base.selected_pool_inv_rms[flat]
        end
    else
        @inbounds for rank in 1:model.workspace_k
            block = Int(base.route_order[rank, model.cycles, flat])
            block_offset = (block - 1) * node_dim
            feature_offset = rank * node_dim
            for coordinate in 1:node_dim
                scratch.features[feature_offset + coordinate] =
                    base.membrane[
                        block_offset + coordinate,
                        final_time,
                        flat,
                    ] * base.selected_pool_inv_rms[flat]
            end
        end
    end
    return nothing
end

function _head_perturbed_raw!(
    destination::Vector{Float32},
    scratch,
    model,
    parameters,
    flat::Int,
    perturbation::Float32,
)
    feature_count = Model.reduced_hay_head_feature_dim(model)
    @inbounds for hidden in 1:model.hidden
        scratch.dhidden[hidden] = parameters.head_bias[hidden]
    end
    @inbounds for feature in 1:feature_count
        perturbed = scratch.features[feature] +
            perturbation * _alignment_sign(feature, flat)
        for hidden in 1:model.hidden
            scratch.dhidden[hidden] = muladd(
                parameters.head_weight[hidden, feature],
                perturbed,
                scratch.dhidden[hidden],
            )
        end
    end
    square_sum = 0.0f0
    @inbounds for hidden in 1:model.hidden
        value = scratch.dhidden[hidden]
        square_sum = muladd(value, value, square_sum)
    end
    inverse_rms = inv(sqrt(
        square_sum / Float32(model.hidden) +
        InputModel.RMS_NORM_EPS,
    ))
    @inbounds for hidden in 1:model.hidden
        scratch.dhidden[hidden] = tanh(
            InputModel.HIDDEN_NORM_SCALE *
            scratch.dhidden[hidden] * inverse_rms,
        )
    end
    @inbounds for output in 1:OUTPUT_DIM
        destination[output] = parameters.output_bias[output]
    end
    @inbounds for hidden in 1:model.hidden
        hidden_value = scratch.dhidden[hidden]
        for output in 1:OUTPUT_DIM
            destination[output] = muladd(
                parameters.output_weight[output, hidden],
                hidden_value,
                destination[output],
            )
        end
    end
    return nothing
end

"""
Learn the output-to-root feedback map from forward noise correlations.

A zero-mean Rademacher perturbation is applied only to the real global head
features.  Its central-difference output response estimates `J * noise`, so
`noise * response'` is an unbiased one-sample estimate of `J'`.  No head
weight is traversed backwards and the functional recurrent signal only reads
the independently stored `root_feedback` matrix.
"""
function _accumulate_root_feedback_alignment!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    parameters,
    flat::Int,
)
    base = tape.base
    _prepare_root_features!(scratch.point_scratch, tape, model, flat)
    _head_perturbed_raw!(
        scratch.local_prediction,
        scratch.point_scratch,
        model,
        parameters,
        flat,
        FEEDBACK_NOISE_SCALE,
    )
    _head_perturbed_raw!(
        scratch.local_error,
        scratch.point_scratch,
        model,
        parameters,
        flat,
        -FEEDBACK_NOISE_SCALE,
    )
    inverse_difference = inv(2.0f0 * FEEDBACK_NOISE_SCALE)
    inverse_valid = inv(Float32(max(base.valid_count, 1)))
    @inbounds for output in 1:OUTPUT_DIM
        response = (
            scratch.local_prediction[output] -
            scratch.local_error[output]
        ) * inverse_difference
        for feature in 1:Model.reduced_hay_head_feature_dim(model)
            estimate = _alignment_sign(feature, flat) * response
            residual = parameters.root_feedback[feature, output] - estimate
            scratch.gradient.root_feedback[feature, output] +=
                inverse_valid * (
                    residual +
                    FEEDBACK_ALIGNMENT_DECAY *
                    parameters.root_feedback[feature, output]
                )
            scratch.feedback_alignment_loss +=
                0.5 * Float64(residual) * Float64(residual) *
                Float64(inverse_valid)
        end
    end
    return nothing
end

"""
Align a two-stage feedback pathway without a reverse traversal of the
supervised head.  Each candidate perturbs one feature coordinate and one
hidden coordinate in the forward direction.  Across a packed arena the
coordinates are covered round-robin, producing low-variance local targets for
the two independently stored feedback matrices.
"""
function _accumulate_layered_feedback_alignment!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    parameters,
    flat::Int,
)
    base = tape.base
    _prepare_root_features!(scratch.point_scratch, tape, model, flat)
    feature_count = Model.reduced_hay_head_feature_dim(model)
    feature_probe = mod(flat - 1, feature_count) + 1
    hidden_probe = mod(flat - 1, model.hidden) + 1
    inverse_valid = inv(Float32(max(base.valid_count, 1)))
    feature_scale = Float32(feature_count) * inverse_valid
    hidden_scale = Float32(model.hidden) * inverse_valid
    inverse_difference = inv(2.0f0 * FEEDBACK_NOISE_SCALE)

    @inbounds for hidden in 1:model.hidden
        positive = parameters.head_bias[hidden]
        negative = parameters.head_bias[hidden]
        for feature in 1:feature_count
            value = scratch.point_scratch.features[feature]
            if feature == feature_probe
                positive = muladd(
                    parameters.head_weight[hidden, feature],
                    value + FEEDBACK_NOISE_SCALE,
                    positive,
                )
                negative = muladd(
                    parameters.head_weight[hidden, feature],
                    value - FEEDBACK_NOISE_SCALE,
                    negative,
                )
            else
                positive = muladd(
                    parameters.head_weight[hidden, feature],
                    value,
                    positive,
                )
                negative = muladd(
                    parameters.head_weight[hidden, feature],
                    value,
                    negative,
                )
            end
        end
        response = (positive - negative) * inverse_difference
        residual = parameters.feature_feedback[
            feature_probe,
            hidden,
        ] - response
        scratch.gradient.feature_feedback[
            feature_probe,
            hidden,
        ] += feature_scale * (
            residual +
            FEEDBACK_ALIGNMENT_DECAY * parameters.feature_feedback[
                feature_probe,
                hidden,
            ]
        )
        scratch.feedback_alignment_loss +=
            0.5 * Float64(residual) * Float64(residual) *
            Float64(feature_scale)
    end

    @inbounds for output in 1:OUTPUT_DIM
        positive = parameters.output_bias[output]
        negative = parameters.output_bias[output]
        for hidden in 1:model.hidden
            value = base.hidden[hidden, flat]
            if hidden == hidden_probe
                positive = muladd(
                    parameters.output_weight[output, hidden],
                    value + FEEDBACK_NOISE_SCALE,
                    positive,
                )
                negative = muladd(
                    parameters.output_weight[output, hidden],
                    value - FEEDBACK_NOISE_SCALE,
                    negative,
                )
            else
                positive = muladd(
                    parameters.output_weight[output, hidden],
                    value,
                    positive,
                )
                negative = muladd(
                    parameters.output_weight[output, hidden],
                    value,
                    negative,
                )
            end
        end
        response = (positive - negative) * inverse_difference
        residual = parameters.output_feedback[
            hidden_probe,
            output,
        ] - response
        scratch.gradient.output_feedback[
            hidden_probe,
            output,
        ] += hidden_scale * (
            residual +
            FEEDBACK_ALIGNMENT_DECAY * parameters.output_feedback[
                hidden_probe,
                output,
            ]
        )
        scratch.feedback_alignment_loss +=
            0.5 * Float64(residual) * Float64(residual) *
            Float64(hidden_scale)
    end
    return nothing
end

function _layered_feedback_features!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    parameters,
    flat::Int,
)
    base = tape.base
    fill!(scratch.point_scratch.dhidden, 0.0f0)
    fill!(scratch.point_scratch.dfeatures, 0.0f0)
    @inbounds for output in 1:OUTPUT_DIM
        cotangent = base.raw_gradient[output, flat]
        for hidden in 1:model.hidden
            scratch.point_scratch.dhidden[hidden] = muladd(
                parameters.output_feedback[hidden, output],
                cotangent,
                scratch.point_scratch.dhidden[hidden],
            )
        end
    end
    projection_mean = 0.0f0
    @inbounds for hidden in 1:model.hidden
        hidden_value = base.hidden[hidden, flat]
        normalized = scratch.point_scratch.dhidden[hidden] *
            InputModel.HIDDEN_NORM_SCALE *
            (1.0f0 - hidden_value * hidden_value)
        scratch.point_scratch.dhidden[hidden] = normalized
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
        hidden_cotangent = inverse_rms * (
            scratch.point_scratch.dhidden[hidden] -
            base.hidden_pre[hidden, flat] *
            inverse_rms_squared * projection_mean
        )
        for feature in 1:Model.reduced_hay_head_feature_dim(model)
            scratch.point_scratch.dfeatures[feature] = muladd(
                parameters.feature_feedback[feature, hidden],
                hidden_cotangent,
                scratch.point_scratch.dfeatures[feature],
            )
        end
    end
    return nothing
end

function _apply_apical_predictive_residual!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    parameters,
    flat::Int,
    cycle::Int,
)
    base = tape.base
    readout = model.readout_per_cell
    predictor_scale = inv(Float32(readout * model.cycles))
    @inbounds for cell in 1:(model.blocks * model.cells_per_block)
        block = div(cell - 1, model.cells_per_block) + 1
        local_cell = cell - (block - 1) * model.cells_per_block
        coordinate_offset = (local_cell - 1) * readout
        plateau_mean = 0.0f0
        for branch in 1:model.branches
            plateau_mean += tape.plateau[
                cell,
                branch,
                cycle + 1,
                flat,
            ]
        end
        plateau_mean /= Float32(model.branches)
        plateau_event = plateau_mean / (1.0f0 + plateau_mean)
        burst_gate = clamp(
            0.25f0 +
            0.50f0 * tape.soma_spikes[cell, cycle, flat] +
            0.25f0 * plateau_event,
            0.25f0,
            1.0f0,
        )
        scratch.burst_gate_sum += Float64(burst_gate)
        scratch.burst_gate_count += 1
        for target_channel in 1:readout
            target_coordinate = coordinate_offset + target_channel
            prediction = parameters.apical_predictor_bias[
                target_channel,
                cell,
            ]
            for source_channel in 1:readout
                source_coordinate = coordinate_offset + source_channel
                state = base.membrane[
                    source_coordinate + (block - 1) * model.node_dim,
                    cycle + 1,
                    flat,
                ]
                prediction = muladd(
                    parameters.apical_predictor_weight[
                        source_channel,
                        target_channel,
                        cell,
                    ],
                    state,
                    prediction,
                )
            end
            target = scratch.feedback_error[target_coordinate, block]
            residual = target - prediction
            positive = max(residual, 0.0f0)
            negative = max(-residual, 0.0f0)
            signed_burst_residual = burst_gate * (positive - negative)
            scratch.feedback_error[target_coordinate, block] =
                target + APICAL_RESIDUAL_SCALE * signed_burst_residual
            predictor_gradient = -residual * predictor_scale
            scratch.gradient.apical_predictor_bias[
                target_channel,
                cell,
            ] += predictor_gradient
            for source_channel in 1:readout
                source_coordinate = coordinate_offset + source_channel
                state = base.membrane[
                    source_coordinate + (block - 1) * model.node_dim,
                    cycle + 1,
                    flat,
                ]
                scratch.gradient.apical_predictor_weight[
                    source_channel,
                    target_channel,
                    cell,
                ] = muladd(
                    predictor_gradient,
                    state,
                    scratch.gradient.apical_predictor_weight[
                        source_channel,
                        target_channel,
                        cell,
                    ],
                )
            end
            scratch.apical_predictor_loss +=
                0.5 * Float64(residual) * Float64(residual)
        end
    end
    return nothing
end

"""
Teacher-control credit path with a single supervised root.

Unlike the DECOLLE control, this path does not attach a Tetris predictor to
every block.  The analytic head VJP is evaluated once at the global output,
then its cotangent is transported only through the actual normalized
workspace and selected-pool interfaces.  Earlier workspace writes receive the
same root cotangent through the workspace decay recurrence.  The resulting
block-state signals are combined with the existing forward eligibility traces.

This deliberately remains a symmetric-head control: it establishes whether
removing block-local labels fixes the spatial learning signal before replacing
the root transport with an adaptive apical feedback graph.
"""
@inline function _propagate_reciprocal_block_credit!(
    scratch::DendriticWorkerScratch,
    model,
    parameters,
    cache::DendriticParameterCache,
    scale::Float32,
)
    fill!(scratch.feedback_next, 0.0f0)
    fill!(scratch.feedback_norm, 0.0f0)
    cells = model.blocks * model.cells_per_block
    @inbounds for source in 1:cells
        source_block = div(source - 1, model.cells_per_block) + 1
        for relation in 1:model.fanout
            destination = model.destination_for_source[source, relation]
            destination_block =
                div(destination - 1, model.cells_per_block) + 1
            source_block == destination_block && continue
            gain = cache.gate_hard[source, relation] *
                parameters.synapse_weight[source, relation]
            scratch.feedback_norm[source_block] += abs(gain)
        end
    end
    @inbounds for source in 1:cells
        source_block = div(source - 1, model.cells_per_block) + 1
        inverse_norm = inv(max(
            scratch.feedback_norm[source_block],
            LOCAL_SIGNAL_RMS_EPSILON,
        ))
        for relation in 1:model.fanout
            destination = model.destination_for_source[source, relation]
            destination_block =
                div(destination - 1, model.cells_per_block) + 1
            source_block == destination_block && continue
            gain = scale * cache.gate_hard[source, relation] *
                parameters.synapse_weight[source, relation] * inverse_norm
            gain == 0.0f0 && continue
            for coordinate in 1:model.node_dim
                scratch.feedback_next[coordinate, source_block] = muladd(
                    gain,
                    scratch.feedback_error[coordinate, destination_block],
                    scratch.feedback_next[coordinate, source_block],
                )
            end
        end
    end
    @inbounds for index in eachindex(
        scratch.feedback_error,
        scratch.feedback_next,
    )
        scratch.feedback_error[index] += scratch.feedback_next[index]
    end
    return nothing
end

const _V11_HEAD_ANCHOR = UInt8(1)
const _V11_HEAD_DELTA = UInt8(2)
const _V11_HEAD_HISTORY = UInt8(3)

@inline function _v11_axis_head_cell_vjp!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    parameters,
    flat::Int,
    block::Int,
    local_cell::Int,
    segment::UInt8,
    cycle::Int=0,
)
    base = tape.base
    state_count = model.readout_per_cell
    rank_count = model.head_state_rank
    anchor_time = 2
    final_time = model.cycles + 1
    block_offset = (block - 1) * model.node_dim
    cell_offset = block_offset + (local_cell - 1) * state_count
    selected = segment == _V11_HEAD_HISTORY ?
        base.block_mask[block, cycle, flat] : 1.0f0

    # The history head is evaluated on m * x.  For an unselected block m=0,
    # so the projected value, head-mix gradient, projection-parameter
    # gradient, and direct state cotangent are all exactly zero.  The only
    # surviving derivative is the counterfactual route-mask cotangent
    #
    #   dL/dm = sum_r dL/du_r * sum_s P[r,s,c] * x_s,
    #
    # because tanh'(0)=1.  Handle that case directly instead of executing the
    # general path's zero FMAs/writes and full24 scatter for every unselected
    # block.  This preserves the exact hard-mask straight-through derivative;
    # it only removes algebraic zeros from the hot history reverse.
    if segment == _V11_HEAD_HISTORY && selected == 0.0f0
        @inbounds for rank in 1:rank_count
            scratch.axis_projection[rank] = 0.0f0
        end
        # `rank` is the first (contiguous) projection axis.  Read every state
        # once and update all four ranks, matching Julia's column-major
        # storage instead of striding through P once per rank.
        @inbounds for state in 1:state_count
            current = base.membrane[
                cell_offset + state,
                cycle + 1,
                flat,
            ]
            for rank in 1:rank_count
                scratch.axis_projection[rank] = muladd(
                    parameters.head_state_projection[
                        rank,
                        state,
                        local_cell,
                    ],
                    current,
                    scratch.axis_projection[rank],
                )
            end
        end
        @inbounds for rank in 1:rank_count
            projected_signal = 0.0f0
            for output in 1:OUTPUT_DIM
                projected_signal = muladd(
                    parameters.head_history_mix[
                        output,
                        cycle,
                        block,
                        local_cell,
                        rank,
                    ],
                    base.raw_gradient[output, flat],
                    projected_signal,
                )
            end
            scratch.route_mask_signal[block, cycle] = muladd(
                projected_signal,
                scratch.axis_projection[rank],
                scratch.route_mask_signal[block, cycle],
            )
        end
        return nothing
    end

    @inbounds for rank in 1:rank_count
        scratch.axis_projection[rank] = 0.0f0
    end
    @inbounds for state in 1:state_count
        anchor = base.membrane[cell_offset + state, anchor_time, flat]
        value = if segment == _V11_HEAD_ANCHOR
            anchor
        elseif segment == _V11_HEAD_DELTA
            base.membrane[cell_offset + state, final_time, flat] - anchor
        else
            selected * base.membrane[
                cell_offset + state,
                cycle + 1,
                flat,
            ]
        end
        for rank in 1:rank_count
            scratch.axis_projection[rank] = muladd(
                parameters.head_state_projection[rank, state, local_cell],
                value,
                scratch.axis_projection[rank],
            )
        end
    end
    @inbounds for rank in 1:rank_count
        scratch.axis_projection[rank] = tanh(scratch.axis_projection[rank])
    end

    @inbounds for rank in 1:rank_count
        projected = scratch.axis_projection[rank]
        projected_signal = 0.0f0
        for output in 1:OUTPUT_DIM
            output_signal = base.raw_gradient[output, flat]
            if segment == _V11_HEAD_ANCHOR
                scratch.gradient.head_anchor_mix[
                    output,
                    block,
                    local_cell,
                    rank,
                ] = muladd(
                    output_signal,
                    projected,
                    scratch.gradient.head_anchor_mix[
                        output,
                        block,
                        local_cell,
                        rank,
                    ],
                )
                projected_signal = muladd(
                    parameters.head_anchor_mix[
                        output,
                        block,
                        local_cell,
                        rank,
                    ],
                    output_signal,
                    projected_signal,
                )
            elseif segment == _V11_HEAD_DELTA
                scratch.gradient.head_delta_mix[
                    output,
                    block,
                    local_cell,
                    rank,
                ] = muladd(
                    output_signal,
                    projected,
                    scratch.gradient.head_delta_mix[
                        output,
                        block,
                        local_cell,
                        rank,
                    ],
                )
                projected_signal = muladd(
                    parameters.head_delta_mix[
                        output,
                        block,
                        local_cell,
                        rank,
                    ],
                    output_signal,
                    projected_signal,
                )
            else
                scratch.gradient.head_history_mix[
                    output,
                    cycle,
                    block,
                    local_cell,
                    rank,
                ] = muladd(
                    output_signal,
                    projected,
                    scratch.gradient.head_history_mix[
                        output,
                        cycle,
                        block,
                        local_cell,
                        rank,
                    ],
                )
                projected_signal = muladd(
                    parameters.head_history_mix[
                        output,
                        cycle,
                        block,
                        local_cell,
                        rank,
                    ],
                    output_signal,
                    projected_signal,
                )
            end
        end
        scratch.route_query_signal[rank] =
            projected_signal * (1.0f0 - projected * projected)
    end

    @inbounds for state in 1:state_count
        anchor = base.membrane[cell_offset + state, anchor_time, flat]
        current = segment == _V11_HEAD_HISTORY ?
            base.membrane[cell_offset + state, cycle + 1, flat] : 0.0f0
        projection_input = if segment == _V11_HEAD_ANCHOR
            anchor
        elseif segment == _V11_HEAD_DELTA
            base.membrane[cell_offset + state, final_time, flat] - anchor
        else
            selected * current
        end
        state_signal = 0.0f0
        for rank in 1:rank_count
            projection_signal = scratch.route_query_signal[rank]
            scratch.gradient.head_state_projection[
                rank,
                state,
                local_cell,
            ] = muladd(
                projection_signal,
                projection_input,
                scratch.gradient.head_state_projection[
                    rank,
                    state,
                    local_cell,
                ],
            )
            state_signal = muladd(
                parameters.head_state_projection[rank, state, local_cell],
                projection_signal,
                state_signal,
            )
        end
        coordinate = (local_cell - 1) * state_count + state
        if segment == _V11_HEAD_ANCHOR
            _scatter_export_cotangent!(
                scratch,
                tape,
                model,
                coordinate,
                block,
                1,
                flat,
                state_signal,
            )
        elseif segment == _V11_HEAD_DELTA
            _scatter_export_cotangent!(
                scratch,
                tape,
                model,
                coordinate,
                block,
                model.cycles,
                flat,
                state_signal,
            )
            _scatter_export_cotangent!(
                scratch,
                tape,
                model,
                coordinate,
                block,
                1,
                flat,
                -state_signal,
            )
        else
            _scatter_export_cotangent!(
                scratch,
                tape,
                model,
                coordinate,
                block,
                cycle,
                flat,
                selected * state_signal,
            )
            # Counterfactual route credit must be computed for every block,
            # including blocks that were not selected in this trajectory.
            scratch.route_mask_signal[block, cycle] = muladd(
                state_signal,
                current,
                scratch.route_mask_signal[block, cycle],
            )
        end
    end
    return nothing
end

function _backward_v11_axis_head_candidate!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    parameters,
    flat::Int,
)
    base = tape.base
    @inbounds for output in 1:OUTPUT_DIM
        scratch.gradient.output_bias[output] +=
            base.raw_gradient[output, flat]
    end
    @inbounds for block in 1:model.blocks
        for local_cell in 1:model.cells_per_block
            _v11_axis_head_cell_vjp!(
                scratch,
                tape,
                model,
                parameters,
                flat,
                block,
                local_cell,
                _V11_HEAD_ANCHOR,
            )
            _v11_axis_head_cell_vjp!(
                scratch,
                tape,
                model,
                parameters,
                flat,
                block,
                local_cell,
                _V11_HEAD_DELTA,
            )
        end
    end
    @inbounds for cycle in 1:model.cycles
        for block in 1:model.blocks
            for local_cell in 1:model.cells_per_block
                _v11_axis_head_cell_vjp!(
                    scratch,
                    tape,
                    model,
                    parameters,
                    flat,
                    block,
                    local_cell,
                    _V11_HEAD_HISTORY,
                    cycle,
                )
            end
        end
    end
    return nothing
end

@inline function _v13_direct_head_cell_vjp!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    parameters,
    flat::Int,
    block::Int,
    local_cell::Int,
    segment::UInt8,
    cycle::Int=0,
)
    base = tape.base
    state_count = model.readout_per_cell
    anchor_time = 2
    final_time = model.cycles + 1
    block_offset = (block - 1) * model.node_dim
    cell_offset = block_offset + (local_cell - 1) * state_count
    history = segment == _V11_HEAD_HISTORY
    selected = history ?
        base.block_mask[block, cycle, flat] : 1.0f0
    mask_cotangent = 0.0f0

    @inbounds for state in 1:state_count
        anchor = base.membrane[cell_offset + state, anchor_time, flat]
        current = history ?
            base.membrane[cell_offset + state, cycle + 1, flat] : 0.0f0
        value = if segment == _V11_HEAD_ANCHOR
            anchor
        elseif segment == _V11_HEAD_DELTA
            base.membrane[cell_offset + state, final_time, flat] - anchor
        else
            current
        end
        state_signal = 0.0f0
        for output in 1:OUTPUT_DIM
            output_signal = base.raw_gradient[output, flat]
            if segment == _V11_HEAD_ANCHOR
                scratch.gradient.head_anchor_mix[
                    output,
                    block,
                    local_cell,
                    state,
                ] = muladd(
                    output_signal,
                    value,
                    scratch.gradient.head_anchor_mix[
                        output,
                        block,
                        local_cell,
                        state,
                    ],
                )
                state_signal = muladd(
                    parameters.head_anchor_mix[
                        output,
                        block,
                        local_cell,
                        state,
                    ],
                    output_signal,
                    state_signal,
                )
            elseif segment == _V11_HEAD_DELTA
                scratch.gradient.head_delta_mix[
                    output,
                    block,
                    local_cell,
                    state,
                ] = muladd(
                    output_signal,
                    value,
                    scratch.gradient.head_delta_mix[
                        output,
                        block,
                        local_cell,
                        state,
                    ],
                )
                state_signal = muladd(
                    parameters.head_delta_mix[
                        output,
                        block,
                        local_cell,
                        state,
                    ],
                    output_signal,
                    state_signal,
                )
            else
                # history = mask * state.  An unselected block receives only
                # the exact counterfactual dmask below: neither its state nor
                # its history-mix parameter participated in the hard forward.
                if selected != 0.0f0
                    scratch.gradient.head_history_mix[
                        output,
                        cycle,
                        block,
                        local_cell,
                        state,
                    ] = muladd(
                        output_signal,
                        value,
                        scratch.gradient.head_history_mix[
                            output,
                            cycle,
                            block,
                            local_cell,
                            state,
                        ],
                    )
                end
                state_signal = muladd(
                    parameters.head_history_mix[
                        output,
                        cycle,
                        block,
                        local_cell,
                        state,
                    ],
                    output_signal,
                    state_signal,
                )
            end
        end

        coordinate = (local_cell - 1) * state_count + state
        if segment == _V11_HEAD_ANCHOR
            _scatter_export_cotangent!(
                scratch,
                tape,
                model,
                coordinate,
                block,
                1,
                flat,
                state_signal,
            )
        elseif segment == _V11_HEAD_DELTA
            _scatter_export_cotangent!(
                scratch,
                tape,
                model,
                coordinate,
                block,
                model.cycles,
                flat,
                state_signal,
            )
            _scatter_export_cotangent!(
                scratch,
                tape,
                model,
                coordinate,
                block,
                1,
                flat,
                -state_signal,
            )
        else
            mask_cotangent = muladd(
                state_signal,
                current,
                mask_cotangent,
            )
            selected == 0.0f0 && continue
            _scatter_export_cotangent!(
                scratch,
                tape,
                model,
                coordinate,
                block,
                cycle,
                flat,
                state_signal,
            )
        end
    end
    if history
        scratch.route_mask_signal[block, cycle] += mask_cotangent
    end
    return nothing
end

function _backward_v13_direct_head_candidate!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    parameters,
    flat::Int,
)
    base = tape.base
    @inbounds for output in 1:OUTPUT_DIM
        scratch.gradient.output_bias[output] +=
            base.raw_gradient[output, flat]
    end
    @inbounds for block in 1:model.blocks
        for local_cell in 1:model.cells_per_block
            _v13_direct_head_cell_vjp!(
                scratch,
                tape,
                model,
                parameters,
                flat,
                block,
                local_cell,
                _V11_HEAD_ANCHOR,
            )
            _v13_direct_head_cell_vjp!(
                scratch,
                tape,
                model,
                parameters,
                flat,
                block,
                local_cell,
                _V11_HEAD_DELTA,
            )
        end
    end
    @inbounds for cycle in 1:model.cycles
        for block in 1:model.blocks
            for local_cell in 1:model.cells_per_block
                _v13_direct_head_cell_vjp!(
                    scratch,
                    tape,
                    model,
                    parameters,
                    flat,
                    block,
                    local_cell,
                    _V11_HEAD_HISTORY,
                    cycle,
                )
            end
        end
    end
    return nothing
end

@inline function _rms_input_cotangent(
    output_cotangent::Float32,
    input::Float32,
    inverse_rms::Float32,
    projection_mean::Float32,
)
    return inverse_rms * (
        output_cotangent -
        input * inverse_rms * inverse_rms * projection_mean
    )
end

function dendritic_prepare_workspace_root_signal_candidate!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    parameters,
    cache::DendriticParameterCache,
    branch_for_edge::Matrix{UInt8},
    flat::Int,
    feedback_hops::Int=0,
    feedback_scale::Float32=0.5f0,
    root_feedback::Union{Nothing,Matrix{Float32}}=nothing,
    online_feedback_alignment::Bool=false,
    apical_predictive_credit::Bool=false,
    routing_temperature::Float32=model.route_temperature,
    exact_graph_bptt::Bool=false,
    pathwise_route_parameters::Bool=true,
)
    base = tape.base
    node_dim = model.node_dim
    routing_logit_limit = exact_graph_bptt ?
        Inf32 : Routing.DEFAULT_LOGIT_LIMIT
    _reset_export_signals!(scratch)
    anchored_temporal =
        model.head_readout === :anchored_temporal
    if anchored_temporal
        exact_graph_bptt || error(
            "anchored-temporal recurrent credit requires exact graph BPTT",
        )
        feedback_hops == 0 || error(
            "reciprocal feedback is not defined for anchored-temporal credit",
        )
        online_feedback_alignment && error(
            "online feedback alignment is not defined for anchored-temporal credit",
        )
        apical_predictive_credit && error(
            "apical-predictive credit is not defined for anchored-temporal credit",
        )
    end
    @inbounds for cycle in 1:model.cycles
        for block in 1:model.blocks
            tape.block_supervised_reward[block, cycle, flat] = 0.0f0
        end
    end

    # Exact-slot models have no legacy pooled/anchored head representation.
    # Their supervised roots are the three raw22 readouts over full24 state:
    # sensory anchor, sparse temporal history and final-minus-anchor delta.
    # Keep this branch ahead of every legacy feedback/local-credit path so an
    # exact-slot model cannot silently fall back to the old node_dim contract.
    if _uses_exact_block_slots(model)
        exact_graph_bptt || error(
            "exact block slots require exact graph BPTT",
        )
        feedback_hops == 0 || error(
            "reciprocal feedback is not defined for exact block slots",
        )
        root_feedback === nothing || error(
            "independent root feedback is not defined for exact block slots",
        )
        online_feedback_alignment && error(
            "online feedback alignment is not defined for exact block slots",
        )
        apical_predictive_credit && error(
            "apical-predictive credit is not defined for exact block slots",
        )
        if _uses_direct_axis_head(model)
            _backward_v13_direct_head_candidate!(
                scratch,
                tape,
                model,
                parameters,
                flat,
            )
        elseif model.head_layout === :axis_factorized
            _backward_v11_axis_head_candidate!(
                scratch,
                tape,
                model,
                parameters,
                flat,
            )
        else
            error(
                "unsupported exact-slot head layout $(model.head_layout)",
            )
        end
        _accumulate_cell_temporal_gradients!(
            scratch,
            tape,
            model,
            parameters,
            cache,
            flat;
            routing_temperature,
            routing_logit_limit,
            branch_for_edge,
            exact_graph_bptt,
            pathwise_route_parameters,
        )
        return nothing
    end

    if online_feedback_alignment
        _accumulate_layered_feedback_alignment!(
            scratch,
            tape,
            model,
            parameters,
            flat,
        )
    end

    if apical_predictive_credit
        _layered_feedback_features!(
            scratch,
            tape,
            model,
            parameters,
            flat,
        )
    elseif root_feedback === nothing
        # This leaves dfeatures as dL/d(normalized head features).  It is the
        # only place in the symmetric control where supervised head weights
        # are consulted.
        _backward_head_candidate!(
            scratch.gradient,
            tape,
            model,
            parameters,
            scratch.point_scratch,
            flat,
        )
    else
        feature_dim = Model.reduced_hay_head_feature_dim(model)
        size(root_feedback) == (feature_dim, OUTPUT_DIM) ||
            throw(DimensionMismatch("root feedback"))
        fill!(scratch.point_scratch.dfeatures, 0.0f0)
        @inbounds for output in 1:OUTPUT_DIM
            error = base.raw_gradient[output, flat]
            for feature in 1:feature_dim
                scratch.point_scratch.dfeatures[feature] = muladd(
                    root_feedback[feature, output],
                    error,
                    scratch.point_scratch.dfeatures[feature],
                )
            end
        end
    end

    if anchored_temporal
        # The three head segments are normalized independently.  Reverse each
        # normalization before separating the delta into its final-summary and
        # sensory-anchor parents.
        anchor_projection_mean = 0.0f0
        temporal_projection_mean = 0.0f0
        delta_projection_mean = 0.0f0
        @inbounds for coordinate in 1:node_dim
            anchor_projection_mean = muladd(
                scratch.point_scratch.dfeatures[coordinate],
                tape.sensory_anchor[coordinate, flat],
                anchor_projection_mean,
            )
            temporal_projection_mean = muladd(
                scratch.point_scratch.dfeatures[node_dim + coordinate],
                tape.temporal_workspace[coordinate, flat],
                temporal_projection_mean,
            )
            delta_projection_mean = muladd(
                scratch.point_scratch.dfeatures[2node_dim + coordinate],
                tape.anchor_delta[coordinate, flat],
                delta_projection_mean,
            )
        end
        inverse_node_dim = inv(Float32(node_dim))
        anchor_projection_mean *= inverse_node_dim
        temporal_projection_mean *= inverse_node_dim
        delta_projection_mean *= inverse_node_dim
        inverse_blocks = inv(sqrt(Float32(model.blocks)))
        inverse_cycles = inv(sqrt(Float32(model.cycles)))
        anchor_inverse_rms = tape.sensory_anchor_inv_rms[flat]
        temporal_inverse_rms = tape.temporal_workspace_inv_rms[flat]
        delta_inverse_rms = tape.anchor_delta_inv_rms[flat]

        @inbounds for bound_coordinate in 1:node_dim
            anchor_cotangent = _rms_input_cotangent(
                scratch.point_scratch.dfeatures[bound_coordinate],
                tape.sensory_anchor[bound_coordinate, flat],
                anchor_inverse_rms,
                anchor_projection_mean,
            )
            temporal_cotangent = _rms_input_cotangent(
                scratch.point_scratch.dfeatures[
                    node_dim + bound_coordinate
                ],
                tape.temporal_workspace[bound_coordinate, flat],
                temporal_inverse_rms,
                temporal_projection_mean,
            )
            delta_cotangent = _rms_input_cotangent(
                scratch.point_scratch.dfeatures[
                    2node_dim + bound_coordinate
                ],
                tape.anchor_delta[bound_coordinate, flat],
                delta_inverse_rms,
                delta_projection_mean,
            )

            # anchor = all-block bound summary at cycle 1, while
            # delta = final all-block bound summary - anchor.
            anchor_state_cotangent =
                (anchor_cotangent - delta_cotangent) * inverse_blocks
            final_state_cotangent = delta_cotangent * inverse_blocks
            for block in 1:model.blocks
                raw_coordinate = Int(tape.spatial_bound_coordinate[
                    bound_coordinate,
                    block,
                ])
                sign_value = tape.spatial_bound_sign[
                    bound_coordinate,
                    block,
                ]
                _scatter_export_cotangent!(
                    scratch,
                    tape,
                    model,
                    raw_coordinate,
                    block,
                    1,
                    flat,
                    sign_value * anchor_state_cotangent,
                )
                _scatter_export_cotangent!(
                    scratch,
                    tape,
                    model,
                    raw_coordinate,
                    block,
                    model.cycles,
                    flat,
                    sign_value * final_state_cotangent,
                )
            end

            # temporal[bound] = sum_t sign * workspace[source,t] / sqrt(T).
            # Store only the direct head roots here; the intrinsic adjoint
            # combines them with later recurrence and apical-feedback roots.
            for cycle in 1:model.cycles
                source_coordinate = Int(tape.temporal_bound_coordinate[
                    bound_coordinate,
                    cycle,
                ])
                scratch.workspace_root_signal[
                    source_coordinate,
                    cycle,
                ] += tape.temporal_bound_sign[
                    bound_coordinate,
                    cycle,
                ] * temporal_cotangent * inverse_cycles
            end
        end

        _accumulate_cell_temporal_gradients!(
            scratch,
            tape,
            model,
            parameters,
            cache,
            flat;
            routing_temperature,
            routing_logit_limit,
            branch_for_edge,
            exact_graph_bptt,
        )
        if root_feedback !== nothing
            # Recurrent roots above came from the independent feedback matrix;
            # the supervised head itself always retains its analytic VJP.
            _backward_head_candidate!(
                scratch.gradient,
                tape,
                model,
                parameters,
                scratch.point_scratch,
                flat,
            )
        end
        return nothing
    end

    feature_dim = Model.reduced_hay_head_feature_dim(model)
    local_feature_dim = feature_dim - node_dim
    workspace_projection_mean = 0.0f0
    local_projection_mean = 0.0f0
    final_time = model.cycles + 1
    fill!(scratch.root_block_signal, 0.0f0)
    fill!(scratch.local_block_signal, 0.0f0)
    @inbounds for coordinate in 1:node_dim
        workspace_value = base.workspace[coordinate, final_time, flat]
        workspace_projection_mean = muladd(
            scratch.point_scratch.dfeatures[coordinate],
            workspace_value,
            workspace_projection_mean,
        )
    end
    inverse_node_dim = inv(Float32(node_dim))
    workspace_projection_mean *= inverse_node_dim
    workspace_inv_rms = base.workspace_inv_rms[flat]
    workspace_inv_rms_squared = workspace_inv_rms * workspace_inv_rms
    local_inv_rms = base.selected_pool_inv_rms[flat]
    local_inv_rms_squared = local_inv_rms * local_inv_rms
    inverse_workspace_k = inv(Float32(model.workspace_k))
    @inbounds for coordinate in 1:node_dim
        workspace_value = base.workspace[coordinate, final_time, flat]
        scratch.global_block_signal[coordinate] = workspace_inv_rms * (
            scratch.point_scratch.dfeatures[coordinate] -
            workspace_value * workspace_inv_rms_squared *
            workspace_projection_mean
        )
    end
    if model.head_readout === :pooled
        @inbounds for coordinate in 1:node_dim
            selected_pool = 0.0f0
            for block in 1:model.blocks
                node = coordinate + (block - 1) * node_dim
                selected_pool = muladd(
                    base.membrane[node, final_time, flat],
                    base.block_mask[block, model.cycles, flat],
                    selected_pool,
                )
            end
            selected_pool *= inverse_workspace_k
            scratch.block_signal[coordinate] = selected_pool
            local_projection_mean = muladd(
                scratch.point_scratch.dfeatures[node_dim + coordinate],
                selected_pool,
                local_projection_mean,
            )
        end
        local_projection_mean *= inverse_node_dim
        @inbounds for coordinate in 1:node_dim
            selected_pool = scratch.block_signal[coordinate]
            signal = local_inv_rms * (
                scratch.point_scratch.dfeatures[node_dim + coordinate] -
                selected_pool * local_inv_rms_squared *
                local_projection_mean
            ) * inverse_workspace_k
            scratch.local_block_signal[coordinate] = signal
            for block in 1:model.blocks
                base.block_mask[block, model.cycles, flat] == 0.0f0 &&
                    continue
                scratch.root_block_signal[coordinate, block] = signal
            end
        end
    else
        @inbounds for rank in 1:model.workspace_k
            block = Int(base.route_order[rank, model.cycles, flat])
            block_offset = (block - 1) * node_dim
            feature_offset = rank * node_dim
            for coordinate in 1:node_dim
                value = base.membrane[
                    block_offset + coordinate,
                    final_time,
                    flat,
                ]
                local_projection_mean = muladd(
                    scratch.point_scratch.dfeatures[
                        feature_offset + coordinate
                    ],
                    value,
                    local_projection_mean,
                )
            end
        end
        local_projection_mean /= Float32(local_feature_dim)
        @inbounds for rank in 1:model.workspace_k
            block = Int(base.route_order[rank, model.cycles, flat])
            block_offset = (block - 1) * node_dim
            feature_offset = rank * node_dim
            for coordinate in 1:node_dim
                value = base.membrane[
                    block_offset + coordinate,
                    final_time,
                    flat,
                ]
                scratch.root_block_signal[coordinate, block] +=
                    local_inv_rms * (
                        scratch.point_scratch.dfeatures[
                            feature_offset + coordinate
                        ] -
                        value * local_inv_rms_squared *
                        local_projection_mean
                    )
            end
        end
    end

    # The root cotangent reaches earlier blocks only through the actual
    # workspace recurrence.  No block receives a private Tetris target.
    @inbounds for cycle in model.cycles:-1:1
        fill!(scratch.feedback_error, 0.0f0)
        fill!(scratch.route_alpha, 0.0f0)
        for coordinate in 1:node_dim
            workspace_value = base.workspace[coordinate, cycle + 1, flat]
            workspace_write_signal =
                scratch.global_block_signal[coordinate] *
                (1.0f0 - workspace_value * workspace_value)
            scratch.gradient.workspace_decay_logit[1] +=
                workspace_write_signal *
                base.workspace[coordinate, cycle, flat] *
                cache.workspace_decay_derivative
            for block in 1:model.blocks
                state = base.membrane[
                    coordinate + (block - 1) * node_dim,
                    cycle + 1,
                    flat,
                ]
                scratch.route_alpha[block] = muladd(
                    workspace_write_signal * inverse_workspace_k,
                    state,
                    scratch.route_alpha[block],
                )
                if cycle == model.cycles
                    route_signal = model.head_readout === :pooled ?
                        scratch.local_block_signal[coordinate] :
                        scratch.root_block_signal[coordinate, block]
                    scratch.route_alpha[block] = muladd(
                        route_signal,
                        state,
                        scratch.route_alpha[block],
                    )
                end
                selected = base.block_mask[block, cycle, flat]
                selected == 0.0f0 && continue
                root_signal = cycle == model.cycles ?
                    scratch.root_block_signal[coordinate, block] : 0.0f0
                scratch.feedback_error[coordinate, block] += root_signal +
                    workspace_write_signal * inverse_workspace_k
            end
            scratch.global_block_signal[coordinate] =
                workspace_write_signal * cache.workspace_decay
        end
        for block in 1:model.blocks
            # Logged as a supervised reward surrogate for diagnostics only.
            # The actual key/query update below is the pathwise loss VJP.
            tape.block_supervised_reward[block, cycle, flat] =
                -scratch.route_alpha[block]
        end
        _accumulate_pathwise_route_cycle!(
            scratch,
            tape,
            model,
            parameters,
            flat,
            cycle,
            routing_temperature,
            routing_logit_limit,
            pathwise_route_parameters,
        )
        for _ in 1:feedback_hops
            _propagate_reciprocal_block_credit!(
                scratch,
                model,
                parameters,
                cache,
                feedback_scale,
            )
        end
        apical_predictive_credit && _apply_apical_predictive_residual!(
            scratch,
            tape,
            model,
            parameters,
            flat,
            cycle,
        )
        for block in 1:model.blocks
            for coordinate in 1:node_dim
                signal = scratch.feedback_error[coordinate, block]
                signal == 0.0f0 && continue
                _scatter_export_cotangent!(
                    scratch,
                    tape,
                    model,
                    coordinate,
                    block,
                    cycle,
                    flat,
                    signal,
                )
            end
        end
    end

    _accumulate_cell_temporal_gradients!(
        scratch,
        tape,
        model,
        parameters,
        cache,
        flat;
        routing_temperature,
        routing_logit_limit,
        branch_for_edge,
        exact_graph_bptt,
    )
    exact_graph_bptt || _accumulate_edge_eligibility!(
        scratch,
        tape,
        model,
        parameters,
        cache,
        branch_for_edge,
        flat,
    )
    if root_feedback !== nothing
        # The supervised head retains its exact analytic VJP, but this call is
        # deliberately delayed until all recurrent signals are materialized.
        _backward_head_candidate!(
            scratch.gradient,
            tape,
            model,
            parameters,
            scratch.point_scratch,
            flat,
        )
    end
    return nothing
end

function dendritic_local_candidate!(
    scratch::DendriticWorkerScratch,
    tape::DendriticTape,
    model,
    parameters,
    flat::Int,
    use_supervised_advantage::Bool=true,
)
    _accumulate_routing_gradients!(
        scratch,
        tape,
        model,
        parameters,
        flat,
        use_supervised_advantage,
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
    parameter_fields = _arena_parameter_fields(parameters)
    keys(parameters) == parameter_fields ||
        error("arena parameter registry changed")
    for (field, name) in enumerate(parameter_fields)
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
    apical_predictor_loss::Float64
    feedback_alignment_loss::Float64
    burst_gate_mean::Float64
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
    recurrent_gradient_accumulator::G
    projection::Array{Float32,3}
    branch_for_edge::Matrix{UInt8}
    gate_mask::BitMatrix
    synapse_utility::Matrix{Float32}
    branch_utility::Array{Float32,3}
    parameter_shards::Vector{DendriticParameterShard}
    gradient_norm_squares::Vector{Float64}
    recurrent_field_norm_squares::Vector{Float64}
    recurrent_optimizer_scales::Vector{Float32}
    optimizer_scale::Float32
    recurrent_optimizer_scale::Float32
    local_predictor_optimizer_scale::Float32
    apical_credit_optimizer_scale::Float32
    head_optimizer_scale::Float32
    recurrent_updates_enabled::Bool
    utility_decay::Float32
    utility_connection_cost::Float32
    structural_interval::Int
    branch_interval::Int
    global_signal_scale::Float32
    local_signal_scale::Float32
    recurrent_learning_rate_multiplier::Float32
    sensory_learning_rate_multiplier::Float32
    routing_learning_rate_multiplier::Float32
    communication_learning_rate_multiplier::Float32
    recurrent_accumulation_steps::Int
    recurrent_accumulation_count::Int
    recurrent_beta1_power::Float32
    recurrent_beta2_power::Float32
    recurrent_optimizer_due::Bool
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
    # The wake trainer learns continuous gate and branch utilities, while
    # discrete mask/location integration is opt-in.  The production schedule
    # performs those non-stationary changes during the slower sleep phase.
    structural_interval::Int=typemax(Int),
    branch_interval::Int=typemax(Int),
    global_signal_scale::Real=1.0f0,
    local_signal_scale::Real=1.0f0,
    recurrent_learning_rate_multiplier::Real=0.001f0,
    sensory_learning_rate_multiplier::Real=
        recurrent_learning_rate_multiplier,
    routing_learning_rate_multiplier::Real=
        recurrent_learning_rate_multiplier,
    communication_learning_rate_multiplier::Real=
        recurrent_learning_rate_multiplier,
    recurrent_accumulation_steps::Int=1,
    routing_entropy_weight::Real=4.0f0,
    routing_entropy_floor::Real=0.85f0,
    routing_load_weight::Real=0.10f0,
    projection_seed::Integer=0x44454e4450524f4a,
)
    model.variant === :causal_recurrent_v2 ||
        throw(ArgumentError(
            "ReducedHayV2ArenaTrainer requires causal_recurrent_v2",
        ))
    structural_interval > 0 || throw(ArgumentError(
        "structural_interval must be positive",
    ))
    branch_interval > 0 || throw(ArgumentError(
        "branch_interval must be positive",
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
    isfinite(sensory_learning_rate_multiplier) &&
        sensory_learning_rate_multiplier >= 0 ||
        throw(ArgumentError(
            "sensory learning-rate multiplier must be finite and nonnegative",
        ))
    isfinite(recurrent_learning_rate_multiplier) &&
        recurrent_learning_rate_multiplier > 0 ||
        throw(ArgumentError(
            "recurrent_learning_rate_multiplier must be finite and positive",
        ))
    isfinite(routing_learning_rate_multiplier) &&
        routing_learning_rate_multiplier > 0 ||
        throw(ArgumentError(
            "routing_learning_rate_multiplier must be finite and positive",
        ))
    isfinite(communication_learning_rate_multiplier) &&
        communication_learning_rate_multiplier >= 0 ||
        throw(ArgumentError(
            "communication_learning_rate_multiplier must be finite and nonnegative",
        ))
    recurrent_accumulation_steps > 0 ||
        throw(ArgumentError(
            "recurrent_accumulation_steps must be positive",
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
    exact_block_slots = _uses_exact_block_slots(model)
    projection = exact_block_slots ?
        zeros(Float32, 0, 0, 0) :
        randn(
            rng,
            Float32,
            model.node_dim,
            OUTPUT_DIM,
            model.blocks,
        ) ./ sqrt(Float32(model.node_dim))
    parameters = _with_local_predictor(parameters, projection, model)
    recurrent_parameter_fields =
        _recurrent_parameter_fields(parameters)
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
        Point.LossScratch(width, state_batch),
        _zero_parameter_tree(parameters),
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
        zeros(Float64, length(recurrent_parameter_fields)),
        ones(Float32, length(recurrent_parameter_fields)),
        1.0f0,
        1.0f0,
        1.0f0,
        1.0f0,
        1.0f0,
        true,
        Float32(utility_decay),
        Float32(utility_connection_cost),
        structural_interval,
        branch_interval,
        Float32(global_signal_scale),
        Float32(local_signal_scale),
        Float32(recurrent_learning_rate_multiplier),
        Float32(sensory_learning_rate_multiplier),
        Float32(routing_learning_rate_multiplier),
        Float32(communication_learning_rate_multiplier),
        recurrent_accumulation_steps,
        0,
        1.0f0,
        1.0f0,
        false,
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
    routing_temperature::Float32
    credit_mode::Symbol
    root_feedback::Matrix{Float32}
    head_updates_enabled::Bool
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
    routing_temperature::Real=trainer.model.route_temperature,
    credit_mode::Symbol=
        _is_exact_slot_parameter_tree(trainer.parameters) ?
        :exact_bptt : :block_teacher,
    root_feedback=nothing,
    head_updates_enabled::Bool=true,
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
    temperature = Float32(routing_temperature)
    isfinite(temperature) && temperature > 0.0f0 ||
        throw(ArgumentError(
            "routing_temperature must be finite and positive",
        ))
    credit_mode in (
        :block_teacher,
        :workspace_root_control,
        :workspace_root_reciprocal_control,
        :workspace_root_adaptive_control,
        :apical_predictive_online,
        :exact_bptt,
    ) ||
        throw(ArgumentError(
            "unsupported credit_mode $credit_mode",
        ))
    exact_slot = _is_exact_slot_parameter_tree(trainer.parameters)
    exact_slot && credit_mode !== :exact_bptt &&
        throw(ArgumentError(
            "exact-slot arena supports only credit_mode=:exact_bptt",
        ))
    if exact_slot && root_feedback !== nothing
        throw(ArgumentError(
            "exact-slot arena has no apical root_feedback parameter",
        ))
    elseif root_feedback !== nothing
        supplied_feedback = Matrix{Float32}(root_feedback)
        size(supplied_feedback) ==
            size(trainer.parameters.root_feedback) ||
            throw(DimensionMismatch("root_feedback"))
        copyto!(trainer.parameters.root_feedback, supplied_feedback)
    end
    feedback = exact_slot ?
        zeros(Float32, 0, 0) : trainer.parameters.root_feedback
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
        temperature,
        credit_mode,
        feedback,
        head_updates_enabled,
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
        worker.apical_predictor_loss = 0.0
        worker.feedback_alignment_loss = 0.0
        worker.burst_gate_sum = 0.0
        worker.burst_gate_count = 0
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

# Compile one direct Val-specialized branch per v11 field.  The shard index is
# runtime data, but every leaf call remains statically dispatched; no Symbol
# lookup or dynamic getproperty is introduced in the reduction hot path.
@generated function _reduce_parameter_field!(
    trainer::DendriticArenaTrainer,
    workers,
    ::Val{FIELDS},
    field::Int,
    first_index::Int,
    last_index::Int,
    scale::Float32,
) where {FIELDS}
    branches = [
        quote
            field == $index && return _reduce_gradient_range!(
                trainer,
                workers,
                Val($(QuoteNode(name))),
                first_index,
                last_index,
                scale,
            )
        end
        for (index, name) in enumerate(FIELDS)
    ]
    return quote
        $(branches...)
        error("unknown static parameter field $field")
    end
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
    if _is_v13_parameter_tree(trainer.parameters)
        scale = field <= length(V11_RECURRENT_PARAMETER_FIELDS) ?
            recurrent : head
        trainer.gradient_norm_squares[target] =
            _reduce_parameter_field!(
                trainer,
                executor.workers,
                Val(V13_ARENA_PARAMETER_FIELDS),
                field,
                first_index,
                last_index,
                scale,
            )
        return nothing
    elseif _is_v11_parameter_tree(trainer.parameters)
        scale = field <= length(V11_RECURRENT_PARAMETER_FIELDS) ?
            recurrent : head
        trainer.gradient_norm_squares[target] =
            _reduce_parameter_field!(
                trainer,
                executor.workers,
                Val(V11_ARENA_PARAMETER_FIELDS),
                field,
                first_index,
                last_index,
                scale,
            )
        return nothing
    end
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
            Val(:local_readout),
            first_index,
            last_index,
            head,
        )
    elseif field == 32
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:local_readout_bias),
            first_index,
            last_index,
            head,
        )
    elseif field == 33
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:apical_predictor_weight),
            first_index,
            last_index,
            head,
        )
    elseif field == 34
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:apical_predictor_bias),
            first_index,
            last_index,
            head,
        )
    elseif field == 35
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:root_feedback),
            first_index,
            last_index,
            head,
        )
    elseif field == 36
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:feature_feedback),
            first_index,
            last_index,
            head,
        )
    elseif field == 37
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:output_feedback),
            first_index,
            last_index,
            head,
        )
    elseif field == 38
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:head_weight),
            first_index,
            last_index,
            head,
        )
    elseif field == 39
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:head_bias),
            first_index,
            last_index,
            head,
        )
    elseif field == 40
        _reduce_gradient_range!(
            trainer,
            executor.workers,
            Val(:output_weight),
            first_index,
            last_index,
            head,
        )
    elseif field == 41
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
    recurrent_fields = _recurrent_parameter_fields(trainer.parameters)
    local_predictor_fields =
        _local_predictor_parameter_fields(trainer.parameters)
    apical_credit_fields =
        _apical_credit_parameter_fields(trainer.parameters)
    fill!(trainer.recurrent_field_norm_squares, 0.0)
    recurrent_square_sum = 0.0
    local_predictor_square_sum = 0.0
    apical_credit_square_sum = 0.0
    head_square_sum = 0.0
    @inbounds for target in eachindex(
        trainer.gradient_norm_squares,
        trainer.parameter_shards,
    )
        field = Int(trainer.parameter_shards[target].field)
        value = trainer.gradient_norm_squares[target]
        if field <= length(recurrent_fields)
            recurrent_square_sum += value
            trainer.recurrent_field_norm_squares[field] += value
        elseif field <= length(recurrent_fields) +
                        length(local_predictor_fields)
            local_predictor_square_sum += value
        elseif field <= length(recurrent_fields) +
                        length(local_predictor_fields) +
                        length(apical_credit_fields)
            apical_credit_square_sum += value
        else
            head_square_sum += value
        end
    end
    total_square_sum =
        recurrent_square_sum +
        local_predictor_square_sum +
        apical_credit_square_sum +
        head_square_sum
    trainer.metrics.gradient_norm = sqrt(total_square_sum)
    maximum_norm = MAX_PARAMETER_FAMILY_GRADIENT_NORM
    scale(square_sum) = Float32(
        square_sum > maximum_norm^2 ?
        maximum_norm / sqrt(square_sum) : 1.0,
    )
    @inbounds for field in eachindex(
        trainer.recurrent_optimizer_scales,
        trainer.recurrent_field_norm_squares,
    )
        trainer.recurrent_optimizer_scales[field] =
            scale(trainer.recurrent_field_norm_squares[field])
    end
    # Retain the scalar as a conservative diagnostic for older callers.  The
    # optimizer itself uses the per-family values so a large branch-bias
    # adjoint cannot starve routing, edge, leak, or workspace updates.
    trainer.recurrent_optimizer_scale =
        minimum(trainer.recurrent_optimizer_scales)
    trainer.local_predictor_optimizer_scale =
        scale(local_predictor_square_sum)
    trainer.apical_credit_optimizer_scale =
        scale(apical_credit_square_sum)
    trainer.head_optimizer_scale = scale(head_square_sum)
    trainer.optimizer_scale = 1.0f0
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
    recurrent_field =
        _is_recurrent_parameter(trainer.parameters, Val(F))
    head_field = _is_head_parameter(trainer.parameters, Val(F))
    local_predictor_field =
        _is_local_predictor_parameter(trainer.parameters, Val(F))
    apical_credit_field =
        _is_apical_credit_parameter(trainer.parameters, Val(F))
    if !trainer.recurrent_updates_enabled &&
       !head_field &&
       !local_predictor_field &&
       !apical_credit_field
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
    inverse_first_bias = inv(
        1.0f0 - (
            recurrent_field ?
            trainer.recurrent_beta1_power : optimizer.beta1_power
        ),
    )
    inverse_second_bias = inv(
        1.0f0 - (
            recurrent_field ?
            trainer.recurrent_beta2_power : optimizer.beta2_power
        ),
    )
    beta1_complement = 1.0f0 - optimizer.beta1
    beta2_complement = 1.0f0 - optimizer.beta2
    scale = if recurrent_field
        field_index = _recurrent_field_index(
            trainer.parameters,
            Val(F),
        )
        @inbounds trainer.recurrent_optimizer_scales[field_index]
    elseif local_predictor_field
        trainer.local_predictor_optimizer_scale
    elseif apical_credit_field
        trainer.apical_credit_optimizer_scale
    else
        trainer.head_optimizer_scale
    end
    learning_rate_multiplier = if _is_direct_routing_parameter(
        trainer.parameters,
        Val(F),
    )
        trainer.routing_learning_rate_multiplier
    elseif F in SENSORY_PARAMETER_FIELDS
        trainer.sensory_learning_rate_multiplier
    elseif _is_communication_parameter(
        trainer.parameters,
        Val(F),
    )
        trainer.communication_learning_rate_multiplier
    elseif recurrent_field
        trainer.recurrent_learning_rate_multiplier
    elseif F in LAYERED_FEEDBACK_PARAMETER_FIELDS
        FEEDBACK_LEARNING_RATE_MULTIPLIER
    else
        1.0f0
    end
    norm_square = 0.0
    accumulator = recurrent_field ? getproperty(
        trainer.recurrent_gradient_accumulator,
        F,
    ) : gradient
    accumulation_inverse = recurrent_field &&
        trainer.recurrent_optimizer_due ?
        inv(Float32(trainer.recurrent_accumulation_count)) : 1.0f0
    @inbounds for index in first_index:last_index
        grad = gradient[index] * scale
        norm_square = muladd(
            Float64(grad),
            Float64(grad),
            norm_square,
        )
        if recurrent_field
            accumulated = accumulator[index] + grad
            if trainer.recurrent_optimizer_due
                grad = accumulated * accumulation_inverse
                accumulator[index] = 0.0f0
            else
                accumulator[index] = accumulated
                gradient[index] = 0.0f0
                continue
            end
        end
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
        parameter[index] -=
            optimizer.learning_rate * learning_rate_multiplier * (
                corrected1 /
                (sqrt(corrected2) + optimizer.epsilon) +
                optimizer.weight_decay * parameter[index]
            )
        gradient[index] = 0.0f0
    end
    return norm_square
end

@generated function _adam_parameter_field!(
    trainer::DendriticArenaTrainer,
    ::Val{FIELDS},
    field::Int,
    first_index::Int,
    last_index::Int,
) where {FIELDS}
    branches = [
        quote
            field == $index && return _adam_range!(
                trainer,
                Val($(QuoteNode(name))),
                first_index,
                last_index,
            )
        end
        for (index, name) in enumerate(FIELDS)
    ]
    return quote
        $(branches...)
        error("unknown static Adam parameter field $field")
    end
end

function _adam_shard!(
    trainer::DendriticArenaTrainer,
    target::Int,
)
    shard = @inbounds trainer.parameter_shards[target]
    field = Int(shard.field)
    first_index = Int(shard.first)
    last_index = Int(shard.last)
    if _is_v13_parameter_tree(trainer.parameters)
        trainer.gradient_norm_squares[target] =
            _adam_parameter_field!(
                trainer,
                Val(V13_ARENA_PARAMETER_FIELDS),
                field,
                first_index,
                last_index,
            )
        return nothing
    elseif _is_v11_parameter_tree(trainer.parameters)
        trainer.gradient_norm_squares[target] =
            _adam_parameter_field!(
                trainer,
                Val(V11_ARENA_PARAMETER_FIELDS),
                field,
                first_index,
                last_index,
            )
        return nothing
    end
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
        _adam_range!(trainer, Val(:local_readout), first_index, last_index)
    elseif field == 32
        _adam_range!(trainer, Val(:local_readout_bias), first_index, last_index)
    elseif field == 33
        _adam_range!(trainer, Val(:apical_predictor_weight), first_index, last_index)
    elseif field == 34
        _adam_range!(trainer, Val(:apical_predictor_bias), first_index, last_index)
    elseif field == 35
        _adam_range!(trainer, Val(:root_feedback), first_index, last_index)
    elseif field == 36
        _adam_range!(trainer, Val(:feature_feedback), first_index, last_index)
    elseif field == 37
        _adam_range!(trainer, Val(:output_feedback), first_index, last_index)
    elseif field == 38
        _adam_range!(trainer, Val(:head_weight), first_index, last_index)
    elseif field == 39
        _adam_range!(trainer, Val(:head_bias), first_index, last_index)
    elseif field == 40
        _adam_range!(trainer, Val(:output_weight), first_index, last_index)
    elseif field == 41
        _adam_range!(trainer, Val(:output_bias), first_index, last_index)
    else
        error("unknown dendritic parameter field $field")
    end
    trainer.gradient_norm_squares[target] = norm_square
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
        partitions = min(BRANCH_CONSOLIDATION_PARTITIONS, cells)
        event = div(step, trainer.branch_interval)
        partition = mod(event - 1, partitions) + 1
        @inbounds for source in partition:partitions:cells
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
            if best_relation != 0 && best_gain > BRANCH_MOVE_MARGIN
                trainer.branch_for_edge[
                    source,
                    best_relation,
                ] = UInt8(best_branch)
                for branch in 1:trainer.model.branches
                    trainer.branch_utility[
                        branch,
                        source,
                        best_relation,
                    ] = 0.0f0
                end
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
    inverse_valid = inv(Float32(max(base.valid_count, 1)))
    @inbounds for state_slot in 1:base.state_batch
        count = Int(base.counts[state_slot])
        offset = (state_slot - 1) * base.width
        inverse_count = inv(Float32(count))
        for cycle in 1:model.cycles
            reward_square_sum = 0.0f0
            for block in 1:model.blocks
                reward_mean = 0.0f0
                for candidate in 1:count
                    flat = offset + candidate
                    reward_mean += tape.block_supervised_reward[
                        block,
                        cycle,
                        flat,
                    ]
                end
                reward_mean *= inverse_count
                for candidate in 1:count
                    flat = offset + candidate
                    centered = tape.block_supervised_reward[
                        block,
                        cycle,
                        flat,
                    ] - reward_mean
                    tape.block_advantage[block, cycle, flat] = centered
                    reward_square_sum = muladd(
                        centered,
                        centered,
                        reward_square_sum,
                    )
                end
            end
            inverse_reward_rms = inv(sqrt(
                reward_square_sum * inverse_count /
                Float32(model.blocks) +
                LOCAL_SIGNAL_RMS_EPSILON,
            ))
            for block in 1:model.blocks
                for candidate in 1:count
                    flat = offset + candidate
                    tape.block_advantage[
                        block,
                        cycle,
                        flat,
                    ] *= inverse_reward_rms * inverse_valid
                end
            end
        end
    end
    return nothing
end

"""
Prepare the scalar third factor required by an ordered Plackett-Luce route.

`route_eligibility[:, cycle, flat]` is the gradient of the log probability of
the complete ordered top-k sample, not a collection of independent per-block
log probabilities.  It must therefore be multiplied by one scalar reward for
that joint route.  The reward here is the negative supervised state excess
loss.  It is a model-level supervised reward surrogate, never an environment
return and never a private Tetris target attached to a block.

The leave-one-state-out baseline is independent of the current state's route
sample, so it reduces variance without changing the expected score-function
gradient.  A one-state diagnostic batch falls back to the raw negative excess.
"""
function _prepare_state_supervised_route_advantages!(
    trainer::DendriticArenaTrainer,
)
    tape = trainer.tape
    base = tape.base
    model = trainer.model
    scratch = trainer.loss_scratch
    states = base.state_batch
    total_excess = 0.0f0
    @inbounds for state_slot in 1:states
        total_excess +=
            scratch.state_composite[state_slot] -
            scratch.state_teacher_entropy[state_slot]
    end
    inverse_other = states > 1 ? inv(Float32(states - 1)) : 0.0f0
    @inbounds for state_slot in 1:states
        excess =
            scratch.state_composite[state_slot] -
            scratch.state_teacher_entropy[state_slot]
        baseline = states > 1 ?
            (total_excess - excess) * inverse_other : 0.0f0
        supervised_reward_surrogate = -(excess - baseline)
        count = Int(base.counts[state_slot])
        offset = (state_slot - 1) * base.width
        for candidate in 1:count
            flat = offset + candidate
            for cycle in 1:model.cycles
                for block in 1:model.blocks
                    tape.block_advantage[block, cycle, flat] =
                        supervised_reward_surrogate
                end
            end
        end
    end
    return nothing
end

function _prepare_routing_regularizer!(
    trainer::DendriticArenaTrainer,
    routing_temperature::Float32=trainer.model.route_temperature,
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
    inverse_temperature = inv(routing_temperature)
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
                    log_blocks *
                    inverse_valid
                load_gradient =
                    load_weight *
                    blocks_f *
                    inverse_valid *
                    probability *
                    (
                        trainer.route_load[block, cycle] -
                        load_projection
                    )
                raw_standardized =
                    (
                        base.route_score[block, cycle, flat] -
                        score_mean
                    ) * score_inv_rms
                normalized_gradient =
                    (entropy_gradient + load_gradient) *
                    inverse_temperature *
                    Routing.bounded_standardized_derivative(
                        raw_standardized,
                        Routing.DEFAULT_LOGIT_LIMIT,
                    )
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
            routing_temperature=executor.routing_temperature,
            routing_logit_limit=
                executor.credit_mode === :exact_bptt ?
                Inf32 : Routing.DEFAULT_LOGIT_LIMIT,
        )
    elseif work.kind == UInt8(DENDRITIC_SIGNAL)
        flat = Int(trainer.tape.base.valid_flats[target])
        if executor.credit_mode in (
            :workspace_root_control,
            :workspace_root_reciprocal_control,
            :workspace_root_adaptive_control,
            :apical_predictive_online,
            :exact_bptt,
        )
            dendritic_prepare_workspace_root_signal_candidate!(
                worker,
                trainer.tape,
                trainer.model,
                trainer.parameters,
                trainer.cache,
                trainer.branch_for_edge,
                flat,
                executor.credit_mode ===
                    :workspace_root_reciprocal_control ? 2 : 0,
                0.5f0,
                (
                    executor.credit_mode ===
                        :workspace_root_adaptive_control ||
                    executor.credit_mode ===
                        :apical_predictive_online
                ) ?
                    executor.root_feedback : nothing,
                executor.credit_mode === :apical_predictive_online,
                executor.credit_mode === :apical_predictive_online,
                executor.routing_temperature,
                executor.credit_mode === :exact_bptt,
                !executor.stochastic_routing,
            )
        else
            dendritic_prepare_signal_candidate!(
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
        end
    elseif work.kind == UInt8(DENDRITIC_LOCAL)
        flat = Int(trainer.tape.base.valid_flats[target])
        dendritic_local_candidate!(
            worker,
            trainer.tape,
            trainer.model,
            trainer.parameters,
            flat,
            executor.stochastic_routing,
        )
    elseif work.kind == UInt8(DENDRITIC_REDUCE)
        _reduce_shard!(executor, target)
    elseif work.kind == UInt8(DENDRITIC_OPTIMIZER)
        shard = @inbounds trainer.parameter_shards[target]
        field = Int(shard.field)
        recurrent_count =
            length(_recurrent_parameter_fields(trainer.parameters))
        local_count = length(
            _local_predictor_parameter_fields(trainer.parameters),
        )
        apical_count = length(
            _apical_credit_parameter_fields(trainer.parameters),
        )
        recurrent_field = field <= recurrent_count
        local_predictor_field =
            recurrent_count < field <= recurrent_count + local_count
        apical_credit_field =
            recurrent_count + local_count < field <=
            recurrent_count + local_count + apical_count
        head_field = field >
            recurrent_count + local_count + apical_count
        if executor.recurrent_signal_scale == 0.0f0 &&
           recurrent_field
            # Keep dispatch type-stable.  The older Symbol/getproperty zeroing
            # helper allocated once per recurrent shard during head-only
            # phases; `_adam_shard!` reaches the generated Val-specialized
            # early-zero path without touching moments or parameters.
            _adam_shard!(trainer, target)
        elseif executor.credit_mode !== :block_teacher &&
               local_predictor_field
            # Root-feedback modes do not use the block-local Tetris
            # predictors.  Freeze both their gradients and AdamW decay so a
            # control run cannot silently depend on them.
            nothing
        elseif executor.credit_mode !== :apical_predictive_online &&
               apical_credit_field
            nothing
        elseif !executor.head_updates_enabled &&
               head_field
            nothing
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
    apical_predictor_loss = 0.0
    feedback_alignment_loss = 0.0
    burst_gate_sum = 0.0
    burst_gate_count = 0
    @inbounds for worker in executor.workers
        local_q_loss += worker.local_q_loss
        local_death_loss += worker.local_death_loss
        local_quantile_loss += worker.local_quantile_loss
        local_geometry_loss += worker.local_geometry_loss
        apical_predictor_loss += worker.apical_predictor_loss
        feedback_alignment_loss += worker.feedback_alignment_loss
        burst_gate_sum += worker.burst_gate_sum
        burst_gate_count += worker.burst_gate_count
    end
    trainer.metrics.local_q_loss =
        local_q_loss / local_count
    trainer.metrics.local_death_loss =
        local_death_loss / local_count
    trainer.metrics.local_quantile_loss =
        local_quantile_loss / local_count
    trainer.metrics.local_geometry_loss =
        local_geometry_loss / local_count
    trainer.metrics.apical_predictor_loss =
        apical_predictor_loss /
        max(base.valid_count * model.cycles * model.blocks, 1)
    feedback_parameter_count =
        _is_exact_slot_parameter_tree(trainer.parameters) ? 0 :
        length(trainer.parameters.feature_feedback) +
        length(trainer.parameters.output_feedback)
    trainer.metrics.feedback_alignment_loss =
        feedback_alignment_loss /
        max(
            base.valid_count * feedback_parameter_count,
            1,
        )
    trainer.metrics.burst_gate_mean =
        burst_gate_sum / max(burst_gate_count, 1)
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

function _refresh_exact_gate_mask!(
    trainer::DendriticArenaTrainer,
)
    mask = trainer.gate_mask
    logits = trainer.parameters.gate_logits
    keep = trainer.model.fixed_recurrent_fanout
    fill!(mask, false)
    @inbounds for source in axes(logits, 1)
        for _ in 1:keep
            best_relation = 0
            best_logit = -Inf32
            for relation in axes(logits, 2)
                mask[source, relation] && continue
                value = logits[source, relation]
                if value > best_logit
                    best_logit = value
                    best_relation = relation
                end
            end
            best_relation != 0 || error("exact gate top-k underflow")
            mask[source, best_relation] = true
        end
    end
    trainer.metrics.structural_flips = 0
    trainer.metrics.branch_moves = 0
    return trainer
end

"""
Populate `trainer.gradient` for the current batch without advancing Adam or
changing any trainable parameter.  This is the diagnostic counterpart of
`dendritic_arena_update!`: it executes the identical pack, forward, loss,
signal, local replay, and deterministic reduction phases, then stops before
the optimizer and structural consolidation phases.

The returned gradient is the raw loss derivative before family clipping,
learning-rate multipliers, Adam moments, and weight decay.  Callers that need
an update direction must therefore use `-gradient`.
"""
function dendritic_arena_gradient!(
    executor::DendriticArenaExecutor,
)
    executor.started ||
        error("dendritic team is not running")
    trainer = executor.trainer
    _clear_worker_accumulators!(executor)
    generation =
        Base.Threads.atomic_add!(
            executor.generation,
            UInt32(1),
        ) + UInt32(1)

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
    _run_phase!(
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
    trainer.last_loss = Point.loss_and_raw_gradient!(
        trainer.tape.base,
        trainer.loss_scratch,
        gate_density,
        0.0f0,
    )
    _run_phase!(
        executor,
        DENDRITIC_SIGNAL,
        trainer.tape.base.valid_count,
        generation,
    )
    if executor.stochastic_routing
        _prepare_state_supervised_route_advantages!(trainer)
    else
        _center_block_supervised_rewards!(trainer)
    end
    if executor.stochastic_routing
        _prepare_routing_regularizer!(
            trainer,
            executor.routing_temperature,
        )
    else
        fill!(trainer.tape.base.route_regularizer_gradient, 0.0f0)
    end
    if executor.credit_mode !== :exact_bptt ||
       executor.stochastic_routing
        _run_phase!(
            executor,
            DENDRITIC_LOCAL,
            trainer.tape.base.valid_count,
            generation,
        )
    end
    trainer.recurrent_updates_enabled =
        executor.recurrent_signal_scale != 0.0f0
    _run_phase!(
        executor,
        DENDRITIC_REDUCE,
        length(trainer.parameter_shards),
        generation,
    )
    _finish_gradient_reduction!(trainer)
    isfinite(trainer.last_loss.composite_loss) ||
        error("non-finite dendritic loss")
    isfinite(trainer.metrics.gradient_norm) ||
        error("non-finite dendritic gradient")
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
    if executor.stochastic_routing
        _prepare_state_supervised_route_advantages!(trainer)
    else
        _center_block_supervised_rewards!(trainer)
    end
    if executor.stochastic_routing
        _prepare_routing_regularizer!(
            trainer,
            executor.routing_temperature,
        )
    else
        fill!(trainer.tape.base.route_regularizer_gradient, 0.0f0)
    end
    # Exact BPTT already accumulates the complete pathwise route VJP during
    # DENDRITIC_SIGNAL.  With deterministic routing the supervised
    # score-function term is disabled and the route regularizer above is
    # exactly zero, so DENDRITIC_LOCAL would only rescan the dense query
    # matrices and add zeros.
    local_replay_seconds =
        if executor.credit_mode === :exact_bptt &&
           !executor.stochastic_routing
            0.0
        else
            _run_phase!(
                executor,
                DENDRITIC_LOCAL,
                trainer.tape.base.valid_count,
                generation,
            )
        end
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
    if trainer.recurrent_updates_enabled
        trainer.recurrent_accumulation_count += 1
        trainer.recurrent_optimizer_due =
            trainer.recurrent_accumulation_count >=
            trainer.recurrent_accumulation_steps
        if trainer.recurrent_optimizer_due
            trainer.recurrent_beta1_power *= trainer.optimizer.beta1
            trainer.recurrent_beta2_power *= trainer.optimizer.beta2
        end
    else
        trainer.recurrent_optimizer_due = false
    end
    optimizer_seconds = _run_phase!(
        executor,
        DENDRITIC_OPTIMIZER,
        length(trainer.parameter_shards),
        generation,
    )
    if trainer.recurrent_optimizer_due
        trainer.recurrent_accumulation_count = 0
        trainer.recurrent_optimizer_due = false
    end
    if executor.recurrent_signal_scale == 0.0f0
        trainer.metrics.structural_flips = 0
        trainer.metrics.branch_moves = 0
    elseif executor.credit_mode === :exact_bptt
        # Exact wake credit must obey the same two-timescale structural
        # contract as local credit.  Re-selecting hard fanout every optimizer
        # step silently bypassed the wake structural freeze.
        if trainer.optimizer.step % trainer.structural_interval == 0
            _refresh_exact_gate_mask!(trainer)
        else
            trainer.metrics.structural_flips = 0
            trainer.metrics.branch_moves = 0
        end
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
const reduced_hay_v2_arena_gradient! = dendritic_arena_gradient!
const reduced_hay_v2_arena_output = dendritic_arena_output
const reduced_hay_v2_arena_update! = dendritic_arena_update!
const reduced_hay_v2_parameter_deltas =
    dendritic_parameter_deltas
const reduced_hay_v2_training_arena =
    dendritic_training_arena

include(joinpath(@__DIR__, "ReducedHayV2IntrinsicAdjoint.jl"))

end # module
