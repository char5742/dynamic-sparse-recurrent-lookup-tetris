module PaperELMTwinFinal

# Equation-faithful Julia/Lux port of the ELM v2 dynamics used as the
# digital-twin class for the paper-mechanism pipeline.
#
# Upstream:
#   https://github.com/AaronSpieler/elmneuron
#   Copyright (c) 2023 Aaron Spieler
#   MIT License
#
# This file is a clean Julia implementation of the published equations.  It
# does not copy model weights and does not claim numerical identity to an
# unpublished TwinProp checkpoint.

using JLD2
using Lux
using Random
using SHA
using Statistics

export ELM_SOURCE_REPOSITORY,
    ELM_SOURCE_COMMIT_SHA1,
    ELM_SOURCE_ROOT_TREE_SHA1,
    ELM_SOURCE_SRC_TREE_SHA1,
    ELM_V2_SOURCE_BLOB_SHA1,
    ELM_V2_SOURCE_SHA256,
    ELM_SOURCE_LICENSE,
    ELM_SOURCE_COPYRIGHT,
    ELM_RECEPTORS,
    ELM_INPUT_PLANES,
    ELMTwinConfig,
    ELMTwinState,
    ELMTwinNormalizer,
    PaperELMTwin,
    FrozenELMTwin,
    build_elm_twin,
    elm_source_metadata,
    elm_input_layout,
    elm_feature_index,
    flatten_elm_input,
    effective_synapse_weight,
    memory_time_constants,
    memory_decay_factors,
    elm_route_input,
    initial_elm_state,
    elm_readout,
    elm_step,
    elm_forward,
    twin_step,
    twin_forward,
    fit_elm_normalizer,
    normalize_elm_input,
    denormalize_elm_output,
    elm_parameter_sha256,
    elm_artifact_sha256,
    freeze_elm_twin,
    assert_frozen_elm_unchanged,
    save_frozen_elm,
    load_frozen_elm,
    load_verified_frozen_elm

const ELM_SOURCE_REPOSITORY =
    "https://github.com/AaronSpieler/elmneuron"
const ELM_SOURCE_COMMIT_SHA1 =
    "52e68a6d39523ac6613a586699b116e8e606dda3"
const ELM_SOURCE_ROOT_TREE_SHA1 =
    "afc22caee862e25d87da70208190b87a0d37591e"
const ELM_SOURCE_SRC_TREE_SHA1 =
    "9294efbcdd0e861cb2059420fc2fa6421ffb44da"
const ELM_V2_SOURCE_BLOB_SHA1 =
    "99bb47798647570760162baac26d917c92b66f1b"
const ELM_V2_SOURCE_SHA256 =
    "fea3d77cc64824e59d0fd6ad4523d60fd948b90fc8860566595124329518bf5e"
const ELM_SOURCE_LICENSE = "MIT"
const ELM_SOURCE_COPYRIGHT = "Copyright (c) 2023 Aaron Spieler"

const ELM_RECEPTORS = (:AMPA, :NMDA, :GABAA)
const ELM_INPUT_PLANES = (:event_conductance, :strength_location)

"""
Configuration for the equation-faithful ELM v2 TwinProp surrogate.

The recurrence and initialization transforms match
`expressive_leaky_memory_neuron_v2.py`.  The default capacity is the
TwinProp paper configuration: `d_m=1000`, one ReLU hidden layer of width
`2d_m=2000`, and log-spaced memory time constants from 0.1 to 300 ms.

The official standalone ELM repository defaults to `d_m=100` and permits
other time-constant ranges; those are ordinary constructor overrides.
`learn_memory_tau=false` matches the official default and keeps the
inverse-scaled-sigmoid time constants outside Lux's parameter tree.
"""
struct ELMTwinConfig
    segments::Int
    receptors::Int
    input_planes::Int
    input_dim::Int
    num_branch::Int
    num_synapse_per_branch::Int
    num_synapse::Int
    num_memory::Int
    hidden_size::Int
    nmda_regions::Int
    lambda_value::Float32
    tau_s_ms::Float32
    memory_tau_min_ms::Float32
    memory_tau_max_ms::Float32
    learn_memory_tau::Bool
    initial_synapse_weight::Float32
    delta_t_ms::Float32
end

function ELMTwinConfig(;
    segments::Integer,
    num_memory::Integer=1_000,
    hidden_size::Union{Nothing,Integer}=nothing,
    nmda_regions::Integer=4,
    lambda_value::Real=5.0,
    tau_s_ms::Real=5.0,
    memory_tau_min_ms::Real=0.1,
    memory_tau_max_ms::Real=300.0,
    learn_memory_tau::Bool=false,
    initial_synapse_weight::Real=0.5,
    delta_t_ms::Real=1.0,
)
    segments >= 1 || throw(ArgumentError("segments must be positive"))
    num_memory >= 1 ||
        throw(ArgumentError("num_memory must be positive"))
    nmda_regions >= 1 ||
        throw(ArgumentError("nmda_regions must be positive"))
    lambda_value > 0 ||
        throw(ArgumentError("lambda_value must be positive"))
    tau_s_ms > 0 || throw(ArgumentError("tau_s_ms must be positive"))
    0 < memory_tau_min_ms < memory_tau_max_ms || throw(ArgumentError(
        "require 0 < memory_tau_min_ms < memory_tau_max_ms",
    ))
    initial_synapse_weight >= 0 || throw(ArgumentError(
        "initial_synapse_weight must be non-negative",
    ))
    delta_t_ms > 0 ||
        throw(ArgumentError("delta_t_ms must be positive"))
    resolved_hidden =
        hidden_size === nothing ? 2 * Int(num_memory) : Int(hidden_size)
    resolved_hidden >= 1 ||
        throw(ArgumentError("hidden_size must be positive"))
    receptors = length(ELM_RECEPTORS)
    input_planes = length(ELM_INPUT_PLANES)
    branches = Int(segments) * receptors
    synapses_per_branch = input_planes
    return ELMTwinConfig(
        Int(segments),
        receptors,
        input_planes,
        branches * input_planes,
        branches,
        synapses_per_branch,
        branches * synapses_per_branch,
        Int(num_memory),
        resolved_hidden,
        Int(nmda_regions),
        Float32(lambda_value),
        Float32(tau_s_ms),
        Float32(memory_tau_min_ms),
        Float32(memory_tau_max_ms),
        learn_memory_tau,
        Float32(initial_synapse_weight),
        Float32(delta_t_ms),
    )
end

"""
Pure recurrent state.  `branch` is `num_branch × batch`, and `memory` is
`num_memory × batch`.
"""
struct ELMTwinState{B<:AbstractMatrix,M<:AbstractMatrix}
    branch::B
    memory::M
end

"""
Training-split normalization statistics.

Spike targets/logits are deliberately not normalized.  Voltage and the
regional NMDA-current targets use independent affine coordinates.
"""
struct ELMTwinNormalizer
    input_mean::Vector{Float32}
    input_scale::Vector{Float32}
    voltage_mean::Float32
    voltage_scale::Float32
    nmda_mean::Vector{Float32}
    nmda_scale::Vector{Float32}
end

"""
ELM v2 Lux layer.

`initial_proto_tau_m` is fixed model metadata unless
`config.learn_memory_tau=true`.  `kappa_s` is the official fixed synapse
trace decay.  The trainable Lux tree contains positive-by-ReLU synapse
prototypes, the one-hidden-layer ReLU MLP, and voltage/spike/NMDA readout.
"""
struct PaperELMTwin <: Lux.AbstractLuxLayer
    config::ELMTwinConfig
    initial_proto_tau_m::Vector{Float32}
    kappa_s::Float32
end

@inline _scaled_sigmoid(x, lower, upper) =
    (upper - lower) / (one(x) + exp(-x)) + lower

@inline function _inverse_scaled_sigmoid(x, lower, upper)
    clamped = clamp(x, lower + 1.0f-6, upper - 1.0f-6)
    return log((clamped - lower) / (upper - clamped))
end

@inline _custom_tanh(x) = 1.7159f0 * tanh((2.0f0 / 3.0f0) * x)
@inline _sigmoid(x) = inv(one(x) + exp(-x))

function _initial_tau(config::ELMTwinConfig)
    result = Vector{Float32}(undef, config.num_memory)
    log_min = log10(config.memory_tau_min_ms + 1.0f-6)
    log_max = log10(config.memory_tau_max_ms - 1.0f-6)
    denominator = Float32(max(config.num_memory - 1, 1))
    @inbounds for index in eachindex(result)
        fraction = Float32(index - 1) / denominator
        tau = 10.0f0 ^ muladd(fraction, log_max - log_min, log_min)
        result[index] = _inverse_scaled_sigmoid(
            tau,
            config.memory_tau_min_ms,
            config.memory_tau_max_ms,
        )
    end
    return result
end

function build_elm_twin(config::ELMTwinConfig)
    return PaperELMTwin(
        config,
        _initial_tau(config),
        exp(-config.delta_t_ms / max(config.tau_s_ms, 1.0f-6)),
    )
end

function elm_source_metadata()
    return (;
        repository=ELM_SOURCE_REPOSITORY,
        commit_sha1=ELM_SOURCE_COMMIT_SHA1,
        root_tree_sha1=ELM_SOURCE_ROOT_TREE_SHA1,
        src_tree_sha1=ELM_SOURCE_SRC_TREE_SHA1,
        v2_blob_sha1=ELM_V2_SOURCE_BLOB_SHA1,
        v2_sha256=ELM_V2_SOURCE_SHA256,
        license=ELM_SOURCE_LICENSE,
        copyright=ELM_SOURCE_COPYRIGHT,
        source_equation_profile="expressive_leaky_memory_neuron_v2.py",
        twinprop_capacity_profile=(
            num_memory=1_000,
            hidden_size=2_000,
            memory_tau_min_ms=0.1,
            memory_tau_max_ms=300.0,
        ),
    )
end

function _linear_parameters(
    rng::AbstractRNG,
    output_size::Int,
    input_size::Int,
)
    bound = inv(sqrt(Float32(input_size)))
    weight =
        (2.0f0 * bound) .*
        rand(rng, Float32, output_size, input_size) .-
        bound
    bias =
        (2.0f0 * bound) .*
        rand(rng, Float32, output_size) .-
        bound
    return weight, bias
end

function Lux.initialparameters(
    rng::AbstractRNG,
    model::PaperELMTwin,
)
    config = model.config
    input_weight, input_bias = _linear_parameters(
        rng,
        config.hidden_size,
        config.num_branch + config.num_memory,
    )
    memory_weight, memory_bias = _linear_parameters(
        rng,
        config.num_memory,
        config.hidden_size,
    )
    output_weight, output_bias = _linear_parameters(
        rng,
        2 + config.nmda_regions,
        config.num_memory,
    )
    base = (;
        proto_w_s=fill(
            config.initial_synapse_weight,
            config.num_synapse,
        ),
        input_weight,
        input_bias,
        memory_weight,
        memory_bias,
        output_weight,
        output_bias,
    )
    return config.learn_memory_tau ?
        merge(base, (; proto_tau_m=copy(model.initial_proto_tau_m))) :
        base
end

Lux.initialstates(::AbstractRNG, ::PaperELMTwin) = (;)

function elm_input_layout(config::ELMTwinConfig)
    return (;
        segments=config.segments,
        receptors=ELM_RECEPTORS,
        input_planes=ELM_INPUT_PLANES,
        input_dim=config.input_dim,
        num_branch=config.num_branch,
        num_synapse_per_branch=config.num_synapse_per_branch,
        flatten_order="segment_fastest_then_receptor_then_plane",
        routing=(
            "event(branch_1), strength(branch_1), " *
            "event(branch_2), strength(branch_2), ...",
        ),
    )
end

elm_input_layout(model::PaperELMTwin) = elm_input_layout(model.config)

@inline function elm_feature_index(
    config::ELMTwinConfig,
    segment::Integer,
    receptor::Integer,
    plane::Integer,
)
    1 <= segment <= config.segments ||
        throw(BoundsError(1:config.segments, segment))
    1 <= receptor <= config.receptors ||
        throw(BoundsError(1:config.receptors, receptor))
    1 <= plane <= config.input_planes ||
        throw(BoundsError(1:config.input_planes, plane))
    return Int(segment) +
           config.segments *
           ((Int(receptor) - 1) +
            config.receptors * (Int(plane) - 1))
end

"""
Flatten `segment × receptor × time × batch` event and strength/location
planes into the ELM's plane-major `feature × time × batch` input.
"""
function flatten_elm_input(
    event_conductance::AbstractArray{<:Real,4},
    strength_location::AbstractArray{<:Real,4},
)
    size(event_conductance) == size(strength_location) ||
        throw(DimensionMismatch("event and strength/location shapes differ"))
    size(event_conductance, 2) == length(ELM_RECEPTORS) ||
        throw(DimensionMismatch("expected three receptor channels"))
    combined = cat(event_conductance, strength_location; dims=2)
    return reshape(
        combined,
        size(combined, 1) * size(combined, 2),
        size(combined, 3),
        size(combined, 4),
    )
end

effective_synapse_weight(ps) = max.(ps.proto_w_s, zero(eltype(ps.proto_w_s)))

function memory_time_constants(model::PaperELMTwin, ps)
    proto = model.config.learn_memory_tau ?
        ps.proto_tau_m :
        model.initial_proto_tau_m
    lower = model.config.memory_tau_min_ms
    upper = model.config.memory_tau_max_ms
    return _scaled_sigmoid.(proto, lower, upper)
end

function memory_decay_factors(model::PaperELMTwin, ps)
    tau = max.(memory_time_constants(model, ps), 1.0f-6)
    kappa_m = exp.(-model.config.delta_t_ms ./ tau)
    kappa_lambda = exp.(
        -model.config.delta_t_ms * model.config.lambda_value ./ tau,
    )
    return (; kappa_m, kappa_lambda)
end

"""
Apply the fixed anatomy-preserving adapter used before the official ELM
branch reshape.

The raw project tensor is plane-major.  For anatomical branch `b`, this
explicitly pairs feature `b` (event) with feature `B+b`
(strength/location), multiplies them by adjacent nonnegative ELM synapse
weights, and sums the pair.  A raw two-at-a-time reshape would incorrectly
mix anatomical branches.
"""
function elm_route_input(
    model::PaperELMTwin,
    ps,
    input::AbstractMatrix,
)
    config = model.config
    size(input, 1) == config.input_dim ||
        throw(DimensionMismatch("invalid ELM input feature dimension"))
    weight = effective_synapse_weight(ps)
    event = @view input[1:config.num_branch, :]
    strength = @view input[(config.num_branch + 1):config.input_dim, :]
    event_weight = reshape(@view(weight[1:2:end]), :, 1)
    strength_weight = reshape(@view(weight[2:2:end]), :, 1)
    return event_weight .* event .+ strength_weight .* strength
end

function initial_elm_state(
    model::PaperELMTwin,
    batch_size::Integer;
    element_type::Type{<:Real}=Float32,
)
    batch_size >= 1 || throw(ArgumentError("batch_size must be positive"))
    return ELMTwinState(
        zeros(element_type, model.config.num_branch, Int(batch_size)),
        zeros(element_type, model.config.num_memory, Int(batch_size)),
    )
end

"""
Official ELM `w_y(m_t)` readout, extended with regional NMDA-current rows.

Rows are ordered `[spike_logit, soma_voltage, nmda_1, ...]`.  Every readout
is trainable; only spike receives a sigmoid postprocessing.
"""
function elm_readout(
    model::PaperELMTwin,
    ps,
    memory::AbstractMatrix,
)
    size(memory, 1) == model.config.num_memory ||
        throw(DimensionMismatch("invalid memory dimension"))
    raw = ps.output_weight * memory .+ ps.output_bias
    spike_logit = vec(raw[1, :])
    voltage = vec(raw[2, :])
    nmda = raw[3:end, :]
    return (;
        voltage,
        spike_logit,
        spike_probability=_sigmoid.(spike_logit),
        nmda,
        raw,
    )
end

"""
Exact ELM v2 single-step recurrence.

```text
b_t       = kappa_s * b_prev + sum_s(relu(w_s) * x_s)
h_t       = relu(W1 * [b_t; kappa_m * m_prev] + c1)
delta_m_t = 1.7159 * tanh((2/3) * (W2 * h_t + c2))
m_t       = kappa_m * m_prev + (1-kappa_lambda) * delta_m_t
y_t       = W_y * m_t + c_y
```
"""
function elm_step(
    model::PaperELMTwin,
    ps,
    state::ELMTwinState,
    input::AbstractMatrix,
)
    config = model.config
    size(state.branch, 1) == config.num_branch ||
        throw(DimensionMismatch("invalid branch-state dimension"))
    size(state.memory, 1) == config.num_memory ||
        throw(DimensionMismatch("invalid memory-state dimension"))
    size(state.branch, 2) == size(state.memory, 2) ==
        size(input, 2) ||
        throw(DimensionMismatch("state/input batch mismatch"))
    branch_input = elm_route_input(model, ps, input)
    branch =
        model.kappa_s .* state.branch .+
        branch_input
    decay = memory_decay_factors(model, ps)
    decayed_memory = decay.kappa_m .* state.memory
    mlp_input = vcat(branch, decayed_memory)
    hidden = max.(
        ps.input_weight * mlp_input .+ ps.input_bias,
        0.0f0,
    )
    delta_memory = _custom_tanh.(
        ps.memory_weight * hidden .+ ps.memory_bias,
    )
    memory =
        decayed_memory .+
        (1.0f0 .- decay.kappa_lambda) .* delta_memory
    next_state = ELMTwinState(branch, memory)
    output = elm_readout(model, ps, memory)
    return merge(
        output,
        (;
            state=next_state,
            branch,
            memory,
            hidden,
            delta_memory,
        ),
    )
end

twin_step(model::PaperELMTwin, ps, state::ELMTwinState, input) =
    elm_step(model, ps, state, input)

function _single_step_trajectory(result)
    batch = length(result.voltage)
    regions = size(result.nmda, 1)
    return (;
        voltage=reshape(result.voltage, 1, batch),
        spike_logit=reshape(result.spike_logit, 1, batch),
        spike_probability=reshape(
            result.spike_probability,
            1,
            batch,
        ),
        nmda=reshape(result.nmda, regions, 1, batch),
    )
end

function _concatenate_trajectory(left, right)
    return (;
        voltage=vcat(left.voltage, right.voltage),
        spike_logit=vcat(left.spike_logit, right.spike_logit),
        spike_probability=vcat(
            left.spike_probability,
            right.spike_probability,
        ),
        nmda=cat(left.nmda, right.nmda; dims=2),
    )
end

# Divide-and-conquer scan keeps the public training path mutation-free while
# avoiding a linearly deep tuple type and the quadratic copying of a naive
# left fold.  State flow remains strictly chronological.
function _elm_scan(
    model::PaperELMTwin,
    ps,
    input::AbstractArray{<:Real,3},
    state::ELMTwinState,
    first_time::Int,
    last_time::Int,
)
    if first_time == last_time
        result = elm_step(
            model,
            ps,
            state,
            @view(input[:, first_time, :]),
        )
        return _single_step_trajectory(result), result.state
    end
    middle = (first_time + last_time) >>> 1
    left, middle_state = _elm_scan(
        model,
        ps,
        input,
        state,
        first_time,
        middle,
    )
    right, final_state = _elm_scan(
        model,
        ps,
        input,
        middle_state,
        middle + 1,
        last_time,
    )
    return _concatenate_trajectory(left, right), final_state
end

"""
Pure, Zygote-compatible ELM trajectory.

Input shape is `feature × time × batch`.  Gradients flow both to the Lux
parameter tree and to every event/strength/location input feature.
"""
function elm_forward(
    model::PaperELMTwin,
    ps,
    input::AbstractArray{<:Real,3};
    initial_state=nothing,
)
    size(input, 1) == model.config.input_dim ||
        throw(DimensionMismatch("invalid ELM input feature dimension"))
    size(input, 2) >= 1 ||
        throw(ArgumentError("ELM trajectory must contain at least one step"))
    batch = size(input, 3)
    state = initial_state === nothing ?
        initial_elm_state(
            model,
            batch;
            element_type=eltype(input),
        ) :
        initial_state
    size(state.branch, 2) == batch ||
        throw(DimensionMismatch("initial-state batch mismatch"))
    trajectory, final_state = _elm_scan(
        model,
        ps,
        input,
        state,
        1,
        size(input, 2),
    )
    return merge(
        trajectory,
        (;
            final_state,
            final_branch=final_state.branch,
            final_memory=final_state.memory,
        ),
    )
end

twin_forward(model::PaperELMTwin, ps, input; kwargs...) =
    elm_forward(model, ps, input; kwargs...)

function (model::PaperELMTwin)(input, ps, st)
    return elm_forward(model, ps, input), st
end

function fit_elm_normalizer(
    input::AbstractArray{<:Real,3},
    target_voltage::AbstractMatrix,
    target_nmda::AbstractArray{<:Real,3},
    indices;
    epsilon::Real=1.0f-5,
)
    isempty(indices) && throw(ArgumentError("normalizer split is empty"))
    input_view = @view input[:, :, indices]
    voltage_view = @view target_voltage[:, indices]
    nmda_view = @view target_nmda[:, :, indices]
    input_mean = vec(Float32.(mean(input_view; dims=(2, 3))))
    input_scale = vec(Float32.(std(
        input_view;
        dims=(2, 3),
        corrected=false,
    )))
    input_scale .= max.(input_scale, Float32(epsilon))
    voltage_mean = Float32(mean(voltage_view))
    voltage_scale = max(
        Float32(std(voltage_view; corrected=false)),
        Float32(epsilon),
    )
    nmda_mean = vec(Float32.(mean(nmda_view; dims=(2, 3))))
    nmda_scale = vec(Float32.(std(
        nmda_view;
        dims=(2, 3),
        corrected=false,
    )))
    nmda_scale .= max.(nmda_scale, Float32(epsilon))
    return ELMTwinNormalizer(
        input_mean,
        input_scale,
        voltage_mean,
        voltage_scale,
        nmda_mean,
        nmda_scale,
    )
end

function normalize_elm_input(
    normalizer::ELMTwinNormalizer,
    input::AbstractArray{<:Real,3},
)
    size(input, 1) == length(normalizer.input_mean) ||
        throw(DimensionMismatch("normalizer/input feature mismatch"))
    return (
        input .-
        reshape(normalizer.input_mean, :, 1, 1)
    ) ./ reshape(normalizer.input_scale, :, 1, 1)
end

function denormalize_elm_output(
    normalizer::ELMTwinNormalizer,
    output,
)
    voltage =
        output.voltage .* normalizer.voltage_scale .+
        normalizer.voltage_mean
    nmda =
        output.nmda .*
        reshape(normalizer.nmda_scale, :, 1, 1) .+
        reshape(normalizer.nmda_mean, :, 1, 1)
    return merge(output, (; voltage, nmda))
end

"""
Frozen, verified-digital-twin-compatible wrapper.

TwinProp optimizes only the input tensor (synapse strength/location).  Calling
`twin_forward(frozen, input)` therefore leaves all ELM arrays outside the
gradient argument while preserving the complete input gradient path.
"""
struct FrozenELMTwin{M,P,N,D}
    model::M
    parameters::P
    normalizer::N
    metadata::D
    parameter_sha256::String
    artifact_sha256::String
end

function _update_digest!(context, value)
    if value isa NamedTuple
        for (name, child) in pairs(value)
            SHA.update!(context, codeunits(String(name)))
            _update_digest!(context, child)
        end
    elseif value isa AbstractArray
        SHA.update!(context, codeunits(string(eltype(value))))
        SHA.update!(context, codeunits(join(size(value), ",")))
        SHA.update!(context, reinterpret(UInt8, vec(Array(value))))
    elseif value isa ELMTwinConfig ||
           value isa ELMTwinNormalizer
        for field in fieldnames(typeof(value))
            SHA.update!(context, codeunits(String(field)))
            _update_digest!(context, getfield(value, field))
        end
    else
        SHA.update!(context, codeunits(repr(value)))
    end
    return context
end

function elm_parameter_sha256(parameters)
    context = SHA.SHA2_256_CTX()
    _update_digest!(context, parameters)
    return bytes2hex(SHA.digest!(context))
end

function elm_artifact_sha256(
    model::PaperELMTwin,
    parameters,
    normalizer::ELMTwinNormalizer,
)
    context = SHA.SHA2_256_CTX()
    _update_digest!(context, model.config)
    _update_digest!(context, model.initial_proto_tau_m)
    _update_digest!(context, model.kappa_s)
    _update_digest!(context, parameters)
    _update_digest!(context, normalizer)
    _update_digest!(context, elm_source_metadata())
    return bytes2hex(SHA.digest!(context))
end

function freeze_elm_twin(
    model::PaperELMTwin,
    parameters,
    normalizer::ELMTwinNormalizer;
    metadata=(;),
)
    frozen_parameters = deepcopy(parameters)
    parameter_digest = elm_parameter_sha256(frozen_parameters)
    artifact_digest = elm_artifact_sha256(
        model,
        frozen_parameters,
        normalizer,
    )
    complete_metadata = merge(
        (;
            model_name="Paper-ELM-v2-Twin",
            stage="detailed_cell_to_digital_twin",
            frozen=true,
            verification_passed=false,
            source=elm_source_metadata(),
            equation_faithful_elm_v2=true,
            unpublished_twinprop_checkpoint_identity_claimed=false,
            fixed_memory_tau=!model.config.learn_memory_tau,
            parameter_sha256=parameter_digest,
            artifact_sha256=artifact_digest,
        ),
        metadata,
    )
    return FrozenELMTwin(
        model,
        frozen_parameters,
        normalizer,
        complete_metadata,
        parameter_digest,
        artifact_digest,
    )
end

function assert_frozen_elm_unchanged(frozen::FrozenELMTwin)
    current_parameter =
        elm_parameter_sha256(frozen.parameters)
    current_parameter == frozen.parameter_sha256 || error(
        "frozen ELM parameter digest changed",
    )
    current_artifact = elm_artifact_sha256(
        frozen.model,
        frozen.parameters,
        frozen.normalizer,
    )
    current_artifact == frozen.artifact_sha256 || error(
        "frozen ELM artifact digest changed",
    )
    return true
end

function twin_forward(
    frozen::FrozenELMTwin,
    input::AbstractArray{<:Real,3};
    normalized::Bool=false,
    initial_state=nothing,
)
    normalized_input = normalized ?
        input :
        normalize_elm_input(frozen.normalizer, input)
    output = elm_forward(
        frozen.model,
        frozen.parameters,
        normalized_input;
        initial_state,
    )
    return normalized ?
        output :
        denormalize_elm_output(frozen.normalizer, output)
end

function twin_step(
    frozen::FrozenELMTwin,
    state::ELMTwinState,
    input::AbstractMatrix;
    normalized::Bool=false,
)
    normalized_input = if normalized
        input
    else
        (
            input .- frozen.normalizer.input_mean
        ) ./ frozen.normalizer.input_scale
    end
    output = elm_step(
        frozen.model,
        frozen.parameters,
        state,
        normalized_input,
    )
    if normalized
        return output
    end
    voltage =
        output.voltage .* frozen.normalizer.voltage_scale .+
        frozen.normalizer.voltage_mean
    nmda =
        output.nmda .* frozen.normalizer.nmda_scale
    nmda = nmda .+ frozen.normalizer.nmda_mean
    return merge(output, (; voltage, nmda))
end

function save_frozen_elm(path::AbstractString, frozen::FrozenELMTwin)
    assert_frozen_elm_unchanged(frozen)
    parent = dirname(abspath(path))
    isdir(parent) || mkpath(parent)
    jldsave(
        path;
        artifact_kind="FrozenELMTwin",
        format_version=1,
        frozen,
    )
    return abspath(path)
end

function load_frozen_elm(path::AbstractString)
    isfile(path) || throw(ArgumentError("ELM artifact not found: $path"))
    payload = JLD2.load(path)
    get(payload, "artifact_kind", nothing) == "FrozenELMTwin" ||
        error("not a FrozenELMTwin artifact")
    get(payload, "format_version", nothing) == 1 ||
        error("unsupported FrozenELMTwin artifact version")
    frozen = payload["frozen"]
    frozen isa FrozenELMTwin ||
        error("artifact payload has the wrong type")
    assert_frozen_elm_unchanged(frozen)
    return frozen
end

function _verification_passed(metadata)
    return hasproperty(metadata, :verification_passed) &&
           getproperty(metadata, :verification_passed) === true
end

function load_verified_frozen_elm(path::AbstractString)
    frozen = load_frozen_elm(path)
    _verification_passed(frozen.metadata) || error(
        "ELM artifact has not passed the held-out verification gate",
    )
    return frozen
end

end # module PaperELMTwinFinal
