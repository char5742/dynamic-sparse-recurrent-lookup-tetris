module PaperELMTwinOfficialV2

# Canonical Spieler ELM-v2 routing and recurrence for the TwinProp pipeline.
#
# Upstream:
#   https://github.com/AaronSpieler/elmneuron
#   Copyright (c) 2023 Aaron Spieler
#   MIT License
#
# The ELM equations, NeuronIO E/I encoding, interlocking indices, overlapping
# windows, and valid-mask semantics below follow the pinned official source.
# The 1,000-memory 0.1--300 ms profile is the TwinProp paper configuration.

using JLD2
using Lux
using Random
using SHA
using Statistics

export OFFICIAL_ELM_REPOSITORY,
    OFFICIAL_ELM_COMMIT_SHA1,
    OFFICIAL_ELM_ROOT_TREE_SHA1,
    OFFICIAL_ELM_SRC_TREE_SHA1,
    OFFICIAL_ELM_V2_BLOB_SHA1,
    OFFICIAL_ELM_V2_SHA256,
    OFFICIAL_MODELING_UTILS_SHA256,
    OFFICIAL_NEURONIO_LOADER_SHA256,
    OFFICIAL_NEURONIO_DATA_UTILS_SHA256,
    OFFICIAL_NEURONIO_TRAIN_UTILS_SHA256,
    OFFICIAL_NEURONIO_TRAIN_SCRIPT_SHA256,
    HAY_TOTAL_SEGMENTS,
    HAY_SOMA_SEGMENT,
    HAY_FIRST_DENDRITIC_SEGMENT,
    HAY_LAST_DENDRITIC_SEGMENT,
    HAY_FIRST_AXON_SEGMENT,
    HAY_LAST_AXON_SEGMENT,
    OFFICIAL_DENDRITIC_LOCATIONS,
    OFFICIAL_ELM_INPUT_DIM,
    OFFICIAL_ELM_BRANCHES,
    OFFICIAL_ELM_SYNAPSES_PER_BRANCH,
    OfficialELMConfig,
    OfficialELMState,
    OfficialELMNormalizer,
    OfficialPaperELMTwin,
    FrozenOfficialELMTwin,
    official_elm_source_metadata,
    official_elm_input_layout,
    official_input_type,
    create_interlocking_indices,
    create_overlapping_window_indices,
    create_neuronio_routing,
    official_contact_channel,
    signed_presynaptic_input,
    build_official_elm_twin,
    effective_synapse_weight,
    memory_time_constants,
    memory_decay_factors,
    route_official_input,
    initial_official_elm_state,
    official_elm_readout,
    official_elm_step,
    official_elm_forward,
    twin_step,
    twin_forward,
    fit_official_elm_normalizer,
    normalize_official_elm_input,
    denormalize_official_elm_output,
    official_parameter_sha256,
    official_artifact_sha256,
    freeze_official_elm_twin,
    assert_frozen_official_elm_unchanged,
    save_frozen_official_elm,
    load_frozen_official_elm,
    load_verified_official_elm

const OFFICIAL_ELM_REPOSITORY =
    "https://github.com/AaronSpieler/elmneuron"
const OFFICIAL_ELM_COMMIT_SHA1 =
    "52e68a6d39523ac6613a586699b116e8e606dda3"
const OFFICIAL_ELM_ROOT_TREE_SHA1 =
    "afc22caee862e25d87da70208190b87a0d37591e"
const OFFICIAL_ELM_SRC_TREE_SHA1 =
    "9294efbcdd0e861cb2059420fc2fa6421ffb44da"
const OFFICIAL_ELM_V2_BLOB_SHA1 =
    "99bb47798647570760162baac26d917c92b66f1b"
const OFFICIAL_ELM_V2_SHA256 =
    "fea3d77cc64824e59d0fd6ad4523d60fd948b90fc8860566595124329518bf5e"
const OFFICIAL_MODELING_UTILS_SHA256 =
    "78306cfb7cc5e8898fa688783130eaa21308f9f0ef6078beee1a75a1b61cab4c"
const OFFICIAL_NEURONIO_LOADER_SHA256 =
    "db83c96060f211ee4889dc0bd6a2a4a8584cc637307df9484e4dfcb77ef6bac8"
const OFFICIAL_NEURONIO_DATA_UTILS_SHA256 =
    "b311010b803f889636a9ecde94ee1bf2aa5f5218b21848e6cf8736f10839fa2d"
const OFFICIAL_NEURONIO_TRAIN_UTILS_SHA256 =
    "99b48d10070654b712963786bf024b5ce3621bbe1da1f3cf3fe8768359018375"
const OFFICIAL_NEURONIO_TRAIN_SCRIPT_SHA256 =
    "f1a8e75af38738530175680517931d917dee7f8b479f436131e068dd766e1bcc"

# The final.v2 Hay manifest has 642 segments:
#   1       soma
#   2:640   dendrites (639 legal contact locations)
#   641:642 axon
const HAY_TOTAL_SEGMENTS = 642
const HAY_SOMA_SEGMENT = 1
const HAY_FIRST_DENDRITIC_SEGMENT = 2
const HAY_LAST_DENDRITIC_SEGMENT = 640
const HAY_FIRST_AXON_SEGMENT = 641
const HAY_LAST_AXON_SEGMENT = 642
const OFFICIAL_DENDRITIC_LOCATIONS =
    HAY_LAST_DENDRITIC_SEGMENT - HAY_FIRST_DENDRITIC_SEGMENT + 1
const OFFICIAL_ELM_INPUT_DIM = 2 * OFFICIAL_DENDRITIC_LOCATIONS
const OFFICIAL_ELM_BRANCHES = 45
const OFFICIAL_ELM_SYNAPSES_PER_BRANCH = 100

"""
Canonical ELM-v2/TwinProp configuration.

The anatomical input and routing dimensions are intentionally fixed:

- 639 legal dendritic locations (`Hay segment 2:640`)
- 639 excitatory followed by 639 inhibitory channels
- 45 branches with 100 routed synapses per branch
- official `neuronio_routing`

Inputs are strength-weighted presynaptic event counts.  Excitatory values are
positive and inhibitory values are negative.  AMPA and NMDA are the paired
consequence of one excitatory presynaptic channel; they are not duplicated as
independent ELM input planes.
"""
struct OfficialELMConfig
    num_input::Int
    num_output::Int
    num_memory::Int
    hidden_size::Int
    nmda_regions::Int
    num_branch::Int
    num_synapse_per_branch::Int
    num_synapse::Int
    lambda_value::Float32
    tau_b_ms::Float32
    memory_tau_min_ms::Float32
    memory_tau_max_ms::Float32
    learn_memory_tau::Bool
    initial_synapse_weight::Float32
    delta_t_ms::Float32
    input_to_synapse_routing::Symbol
end

function OfficialELMConfig(;
    num_memory::Integer=1_000,
    hidden_size::Union{Nothing,Integer}=nothing,
    nmda_regions::Integer=4,
    lambda_value::Real=5.0,
    tau_b_ms::Real=5.0,
    memory_tau_min_ms::Real=0.1,
    memory_tau_max_ms::Real=300.0,
    learn_memory_tau::Bool=false,
    initial_synapse_weight::Real=0.5,
    delta_t_ms::Real=1.0,
)
    num_memory >= 1 ||
        throw(ArgumentError("num_memory must be positive"))
    nmda_regions >= 1 ||
        throw(ArgumentError("nmda_regions must be positive"))
    lambda_value > 0 ||
        throw(ArgumentError("lambda_value must be positive"))
    tau_b_ms > 0 ||
        throw(ArgumentError("tau_b_ms must be positive"))
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
    return OfficialELMConfig(
        OFFICIAL_ELM_INPUT_DIM,
        2 + Int(nmda_regions),
        Int(num_memory),
        resolved_hidden,
        Int(nmda_regions),
        OFFICIAL_ELM_BRANCHES,
        OFFICIAL_ELM_SYNAPSES_PER_BRANCH,
        OFFICIAL_ELM_BRANCHES * OFFICIAL_ELM_SYNAPSES_PER_BRANCH,
        Float32(lambda_value),
        Float32(tau_b_ms),
        Float32(memory_tau_min_ms),
        Float32(memory_tau_max_ms),
        learn_memory_tau,
        Float32(initial_synapse_weight),
        Float32(delta_t_ms),
        :neuronio_routing,
    )
end

struct OfficialELMState{B<:AbstractMatrix,M<:AbstractMatrix}
    branch::B
    memory::M
end

struct OfficialELMNormalizer
    input_mean::Vector{Float32}
    input_scale::Vector{Float32}
    voltage_mean::Float32
    voltage_scale::Float32
    nmda_mean::Vector{Float32}
    nmda_scale::Vector{Float32}
end

"""
Fixed-routing official ELM v2 Lux layer.

`input_indices` and `valid_indices_mask` are the exact non-trainable routing
artifacts created by the upstream `modeling_utils.py` algorithms.
"""
struct OfficialPaperELMTwin <: Lux.AbstractLuxLayer
    config::OfficialELMConfig
    input_indices::Vector{Int}
    valid_indices_mask::Vector{Float32}
    initial_proto_tau_m::Vector{Float32}
    kappa_b::Vector{Float32}
end

function official_elm_source_metadata()
    return (;
        repository=OFFICIAL_ELM_REPOSITORY,
        commit_sha1=OFFICIAL_ELM_COMMIT_SHA1,
        root_tree_sha1=OFFICIAL_ELM_ROOT_TREE_SHA1,
        src_tree_sha1=OFFICIAL_ELM_SRC_TREE_SHA1,
        elm_v2_blob_sha1=OFFICIAL_ELM_V2_BLOB_SHA1,
        elm_v2_sha256=OFFICIAL_ELM_V2_SHA256,
        modeling_utils_sha256=OFFICIAL_MODELING_UTILS_SHA256,
        neuronio_loader_sha256=OFFICIAL_NEURONIO_LOADER_SHA256,
        neuronio_data_utils_sha256=OFFICIAL_NEURONIO_DATA_UTILS_SHA256,
        neuronio_train_utils_sha256=OFFICIAL_NEURONIO_TRAIN_UTILS_SHA256,
        neuronio_train_script_sha256=OFFICIAL_NEURONIO_TRAIN_SCRIPT_SHA256,
        license="MIT",
        copyright="Copyright (c) 2023 Aaron Spieler",
        recurrence="ELM v2",
        routing="neuronio_routing",
        input_semantics="signed strength * presynaptic event_count",
        hay_segment_mapping=(
            soma=HAY_SOMA_SEGMENT,
            dendrites=(
                HAY_FIRST_DENDRITIC_SEGMENT,
                HAY_LAST_DENDRITIC_SEGMENT,
            ),
            axon=(HAY_FIRST_AXON_SEGMENT, HAY_LAST_AXON_SEGMENT),
        ),
    )
end

function official_elm_input_layout(
    config::OfficialELMConfig=OfficialELMConfig(),
)
    return (;
        total_hay_segments=HAY_TOTAL_SEGMENTS,
        legal_contact_segments=(
            HAY_FIRST_DENDRITIC_SEGMENT,
            HAY_LAST_DENDRITIC_SEGMENT,
        ),
        excluded_soma_segment=HAY_SOMA_SEGMENT,
        excluded_axon_segments=(
            HAY_FIRST_AXON_SEGMENT,
            HAY_LAST_AXON_SEGMENT,
        ),
        dendritic_locations=OFFICIAL_DENDRITIC_LOCATIONS,
        excitatory_channels=1:OFFICIAL_DENDRITIC_LOCATIONS,
        inhibitory_channels=(
            OFFICIAL_DENDRITIC_LOCATIONS + 1
        ):OFFICIAL_ELM_INPUT_DIM,
        input_dim=config.num_input,
        input_semantics="E:+strength*event_count, I:-strength*event_count",
        num_branch=config.num_branch,
        num_synapse_per_branch=config.num_synapse_per_branch,
        input_to_synapse_routing=config.input_to_synapse_routing,
    )
end

"""
Official NeuronIO input-type vector: first half `+1`, second half `-1`.
"""
function official_input_type()
    return vcat(
        ones(Float32, OFFICIAL_DENDRITIC_LOCATIONS),
        -ones(Float32, OFFICIAL_DENDRITIC_LOCATIONS),
    )
end

"""
Exact 1-based Julia translation of upstream `create_interlocking_indices`.

For 639 E followed by 639 I inputs the result begins
`E1,I1,E2,I2,...`.
"""
function create_interlocking_indices(num_input::Integer)
    num_input >= 2 ||
        throw(ArgumentError("num_input must be at least two"))
    iseven(num_input) ||
        throw(ArgumentError("E/I interlocking requires an even input count"))
    half = Int(num_input) ÷ 2
    result = Vector{Int}(undef, Int(num_input))
    @inbounds for source_zero in 0:(Int(num_input) - 1)
        index_zero =
            (source_zero % 2) * half +
            source_zero ÷ 2
        result[source_zero + 1] = index_zero + 1
    end
    return result
end

"""
Exact upstream overlapping-window indices and mask, converted to 1-based
indices after applying the upstream clamp-to-last-input operation.
"""
function create_overlapping_window_indices(
    num_input::Integer,
    num_windows::Integer,
    num_elements_per_window::Integer,
)
    num_input >= 1 ||
        throw(ArgumentError("num_input must be positive"))
    num_windows >= 1 ||
        throw(ArgumentError("num_windows must be positive"))
    num_elements_per_window >= 1 ||
        throw(ArgumentError("num_elements_per_window must be positive"))
    stride = cld(Int(num_input), Int(num_windows))
    count = Int(num_windows) * Int(num_elements_per_window)
    indices = Vector{Int}(undef, count)
    valid = Vector{Bool}(undef, count)
    slot = 1
    @inbounds for window_zero in 0:(Int(num_windows) - 1)
        for element_zero in 0:(Int(num_elements_per_window) - 1)
            index_zero = window_zero * stride + element_zero
            valid[slot] = index_zero < num_input
            indices[slot] = min(index_zero, Int(num_input) - 1) + 1
            slot += 1
        end
    end
    return indices, valid
end

function create_neuronio_routing(config::OfficialELMConfig)
    cld(config.num_input, config.num_branch) <=
        config.num_synapse_per_branch || throw(ArgumentError(
        "num_synapse_per_branch is too small for neuronio_routing",
    ))
    interlocking = create_interlocking_indices(config.num_input)
    overlapping, valid = create_overlapping_window_indices(
        config.num_input,
        config.num_branch,
        config.num_synapse_per_branch,
    )
    return interlocking[overlapping], valid
end

"""
Map a legal Hay dendritic segment and Dale type to the canonical input
channel.  Soma and axon contacts fail rather than silently entering the twin.
"""
function official_contact_channel(segment::Integer, kind)
    HAY_FIRST_DENDRITIC_SEGMENT <= segment <=
        HAY_LAST_DENDRITIC_SEGMENT || throw(ArgumentError(
        "ELM contacts must target dendritic Hay segments 2:640; " *
        "soma and axon are forbidden",
    ))
    location = Int(segment) - HAY_FIRST_DENDRITIC_SEGMENT + 1
    if kind === :E || kind === :excitatory ||
       kind == 1 || kind == UInt8(1)
        return location
    elseif kind === :I || kind === :inhibitory ||
           kind == -1 || kind == UInt8(2)
        return OFFICIAL_DENDRITIC_LOCATIONS + location
    end
    throw(ArgumentError("kind must be excitatory/E or inhibitory/I"))
end

"""
Construct canonical signed E/I input from nonnegative, already
strength-weighted event-count tensors.

Both inputs must be `639 × ...`.  The result is `1278 × ...`; inhibitory
events receive the official `-1` input type.
"""
function signed_presynaptic_input(
    excitatory::AbstractArray{<:Real},
    inhibitory::AbstractArray{<:Real},
)
    size(excitatory) == size(inhibitory) ||
        throw(DimensionMismatch("E/I input shapes differ"))
    size(excitatory, 1) == OFFICIAL_DENDRITIC_LOCATIONS ||
        throw(DimensionMismatch("expected 639 dendritic locations"))
    return cat(excitatory, -inhibitory; dims=1)
end

@inline _scaled_sigmoid(x, lower, upper) =
    (upper - lower) / (one(x) + exp(-x)) + lower

@inline function _inverse_scaled_sigmoid(x, lower, upper)
    clamped = clamp(x, lower + 1.0f-6, upper - 1.0f-6)
    return log((clamped - lower) / (upper - clamped))
end

@inline _custom_tanh(x) = 1.7159f0 * tanh((2.0f0 / 3.0f0) * x)
@inline _sigmoid(x) = inv(one(x) + exp(-x))

function _initial_tau(config::OfficialELMConfig)
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

function build_official_elm_twin(config::OfficialELMConfig)
    indices, valid = create_neuronio_routing(config)
    kappa = exp(
        -config.delta_t_ms / max(config.tau_b_ms, 1.0f-6),
    )
    return OfficialPaperELMTwin(
        config,
        indices,
        Float32.(valid),
        _initial_tau(config),
        fill(kappa, config.num_branch),
    )
end

function _linear_parameters(
    rng::AbstractRNG,
    output_size::Int,
    input_size::Int,
)
    # PyTorch nn.Linear default: U(-1/sqrt(fan_in), +1/sqrt(fan_in)).
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
    model::OfficialPaperELMTwin,
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
        config.num_output,
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

Lux.initialstates(::AbstractRNG, ::OfficialPaperELMTwin) = (;)

effective_synapse_weight(ps) =
    max.(ps.proto_w_s, zero(eltype(ps.proto_w_s)))

function memory_time_constants(model::OfficialPaperELMTwin, ps)
    proto = model.config.learn_memory_tau ?
        ps.proto_tau_m :
        model.initial_proto_tau_m
    return _scaled_sigmoid.(
        proto,
        model.config.memory_tau_min_ms,
        model.config.memory_tau_max_ms,
    )
end

function memory_decay_factors(model::OfficialPaperELMTwin, ps)
    tau = max.(memory_time_constants(model, ps), 1.0f-6)
    kappa_m = exp.(-model.config.delta_t_ms ./ tau)
    kappa_lambda = exp.(
        -model.config.delta_t_ms * model.config.lambda_value ./ tau,
    )
    return (; kappa_m, kappa_lambda)
end

"""
Apply exact official `route_input_to_synapses` semantics.

Input is already signed and has shape `1278 × batch`.  Output is
`4500 × batch`, in branch-major/synapse-minor order.
"""
function route_official_input(
    model::OfficialPaperELMTwin,
    input::AbstractMatrix,
)
    size(input, 1) == model.config.num_input ||
        throw(DimensionMismatch("official ELM input must have 1278 rows"))
    routed = input[model.input_indices, :]
    return routed .* model.valid_indices_mask
end

function initial_official_elm_state(
    model::OfficialPaperELMTwin,
    batch_size::Integer;
    element_type::Type{<:Real}=Float32,
)
    batch_size >= 1 || throw(ArgumentError("batch_size must be positive"))
    return OfficialELMState(
        zeros(element_type, model.config.num_branch, Int(batch_size)),
        zeros(element_type, model.config.num_memory, Int(batch_size)),
    )
end

function official_elm_readout(
    model::OfficialPaperELMTwin,
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
Exact official ELM v2 recurrence after fixed NeuronIO routing.
"""
function official_elm_step(
    model::OfficialPaperELMTwin,
    ps,
    state::OfficialELMState,
    input::AbstractMatrix,
)
    config = model.config
    size(state.branch) == (config.num_branch, size(input, 2)) ||
        throw(DimensionMismatch("branch/input batch mismatch"))
    size(state.memory) == (config.num_memory, size(input, 2)) ||
        throw(DimensionMismatch("memory/input batch mismatch"))
    routed = route_official_input(model, input)
    weighted = effective_synapse_weight(ps) .* routed
    branch_input = reshape(
        sum(
            reshape(
                weighted,
                config.num_synapse_per_branch,
                config.num_branch,
                size(input, 2),
            );
            dims=1,
        ),
        config.num_branch,
        size(input, 2),
    )
    branch = model.kappa_b .* state.branch .+ branch_input
    decay = memory_decay_factors(model, ps)
    decayed_memory = decay.kappa_m .* state.memory
    hidden = max.(
        ps.input_weight * vcat(branch, decayed_memory) .+
        ps.input_bias,
        0.0f0,
    )
    delta_memory = _custom_tanh.(
        ps.memory_weight * hidden .+ ps.memory_bias,
    )
    memory =
        decayed_memory .+
        (1.0f0 .- decay.kappa_lambda) .* delta_memory
    next_state = OfficialELMState(branch, memory)
    output = official_elm_readout(model, ps, memory)
    return merge(
        output,
        (;
            state=next_state,
            routed_input=routed,
            branch,
            branch_input,
            memory,
            hidden,
            delta_memory,
        ),
    )
end

twin_step(
    model::OfficialPaperELMTwin,
    ps,
    state::OfficialELMState,
    input,
) = official_elm_step(model, ps, state, input)

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

function _official_scan(
    model::OfficialPaperELMTwin,
    ps,
    input::AbstractArray{<:Real,3},
    state::OfficialELMState,
    first_time::Int,
    last_time::Int,
)
    if first_time == last_time
        result = official_elm_step(
            model,
            ps,
            state,
            @view(input[:, first_time, :]),
        )
        return _single_step_trajectory(result), result.state
    end
    middle = (first_time + last_time) >>> 1
    left, middle_state = _official_scan(
        model,
        ps,
        input,
        state,
        first_time,
        middle,
    )
    right, final_state = _official_scan(
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
Pure, mutation-free, Zygote-compatible trajectory.

Input is signed `1278 × time × batch`; gradients flow to every
strength/location-generated presynaptic input and all trainable ELM arrays.
"""
function official_elm_forward(
    model::OfficialPaperELMTwin,
    ps,
    input::AbstractArray{<:Real,3};
    initial_state=nothing,
)
    size(input, 1) == model.config.num_input ||
        throw(DimensionMismatch("official ELM input must have 1278 rows"))
    size(input, 2) >= 1 ||
        throw(ArgumentError("trajectory must contain at least one step"))
    batch = size(input, 3)
    state = initial_state === nothing ?
        initial_official_elm_state(
            model,
            batch;
            element_type=eltype(input),
        ) :
        initial_state
    trajectory, final_state = _official_scan(
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

twin_forward(model::OfficialPaperELMTwin, ps, input; kwargs...) =
    official_elm_forward(model, ps, input; kwargs...)

function (model::OfficialPaperELMTwin)(input, ps, st)
    return official_elm_forward(model, ps, input), st
end

function fit_official_elm_normalizer(
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
    return OfficialELMNormalizer(
        input_mean,
        input_scale,
        voltage_mean,
        voltage_scale,
        nmda_mean,
        nmda_scale,
    )
end

function normalize_official_elm_input(
    normalizer::OfficialELMNormalizer,
    input::AbstractArray{<:Real,3},
)
    size(input, 1) == length(normalizer.input_mean) ||
        throw(DimensionMismatch("normalizer/input feature mismatch"))
    return (
        input .-
        reshape(normalizer.input_mean, :, 1, 1)
    ) ./ reshape(normalizer.input_scale, :, 1, 1)
end

function denormalize_official_elm_output(
    normalizer::OfficialELMNormalizer,
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

struct FrozenOfficialELMTwin{M,P,N,D}
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
    elseif value isa OfficialELMConfig ||
           value isa OfficialELMNormalizer
        for field in fieldnames(typeof(value))
            SHA.update!(context, codeunits(String(field)))
            _update_digest!(context, getfield(value, field))
        end
    else
        SHA.update!(context, codeunits(repr(value)))
    end
    return context
end

function official_parameter_sha256(parameters)
    context = SHA.SHA2_256_CTX()
    _update_digest!(context, parameters)
    return bytes2hex(SHA.digest!(context))
end

function official_artifact_sha256(
    model::OfficialPaperELMTwin,
    parameters,
    normalizer::OfficialELMNormalizer,
)
    context = SHA.SHA2_256_CTX()
    _update_digest!(context, model.config)
    _update_digest!(context, model.input_indices)
    _update_digest!(context, model.valid_indices_mask)
    _update_digest!(context, model.initial_proto_tau_m)
    _update_digest!(context, model.kappa_b)
    _update_digest!(context, parameters)
    _update_digest!(context, normalizer)
    _update_digest!(context, official_elm_source_metadata())
    return bytes2hex(SHA.digest!(context))
end

function freeze_official_elm_twin(
    model::OfficialPaperELMTwin,
    parameters,
    normalizer::OfficialELMNormalizer;
    metadata=(;),
)
    frozen_parameters = deepcopy(parameters)
    parameter_digest = official_parameter_sha256(frozen_parameters)
    artifact_digest = official_artifact_sha256(
        model,
        frozen_parameters,
        normalizer,
    )
    complete_metadata = merge(
        (;
            model_name="Paper-ELM-v2-OfficialRouting-Twin",
            stage="detailed_cell_to_digital_twin",
            canonical_official_routing=true,
            frozen=true,
            verification_passed=false,
            source=official_elm_source_metadata(),
            input_dim=OFFICIAL_ELM_INPUT_DIM,
            dendritic_locations=OFFICIAL_DENDRITIC_LOCATIONS,
            branches=OFFICIAL_ELM_BRANCHES,
            synapses_per_branch=OFFICIAL_ELM_SYNAPSES_PER_BRANCH,
            fixed_memory_tau=!model.config.learn_memory_tau,
            unpublished_twinprop_checkpoint_identity_claimed=false,
            parameter_sha256=parameter_digest,
            artifact_sha256=artifact_digest,
        ),
        metadata,
    )
    return FrozenOfficialELMTwin(
        model,
        frozen_parameters,
        normalizer,
        complete_metadata,
        parameter_digest,
        artifact_digest,
    )
end

function assert_frozen_official_elm_unchanged(
    frozen::FrozenOfficialELMTwin,
)
    official_parameter_sha256(frozen.parameters) ==
        frozen.parameter_sha256 ||
        error("frozen official ELM parameter digest changed")
    official_artifact_sha256(
        frozen.model,
        frozen.parameters,
        frozen.normalizer,
    ) == frozen.artifact_sha256 ||
        error("frozen official ELM artifact digest changed")
    return true
end

function twin_forward(
    frozen::FrozenOfficialELMTwin,
    input::AbstractArray{<:Real,3};
    normalized::Bool=false,
    initial_state=nothing,
)
    normalized_input = normalized ?
        input :
        normalize_official_elm_input(frozen.normalizer, input)
    output = official_elm_forward(
        frozen.model,
        frozen.parameters,
        normalized_input;
        initial_state,
    )
    return normalized ?
        output :
        denormalize_official_elm_output(frozen.normalizer, output)
end

function twin_step(
    frozen::FrozenOfficialELMTwin,
    state::OfficialELMState,
    input::AbstractMatrix;
    normalized::Bool=false,
)
    normalized_input = normalized ?
        input :
        (
            input .- frozen.normalizer.input_mean
        ) ./ frozen.normalizer.input_scale
    output = official_elm_step(
        frozen.model,
        frozen.parameters,
        state,
        normalized_input,
    )
    normalized && return output
    voltage =
        output.voltage .* frozen.normalizer.voltage_scale .+
        frozen.normalizer.voltage_mean
    nmda =
        output.nmda .* frozen.normalizer.nmda_scale .+
        frozen.normalizer.nmda_mean
    return merge(output, (; voltage, nmda))
end

function save_frozen_official_elm(
    path::AbstractString,
    frozen::FrozenOfficialELMTwin,
)
    assert_frozen_official_elm_unchanged(frozen)
    parent = dirname(abspath(path))
    isdir(parent) || mkpath(parent)
    jldsave(
        path;
        artifact_kind="FrozenOfficialELMTwin",
        format_version=2,
        frozen,
    )
    return abspath(path)
end

function load_frozen_official_elm(path::AbstractString)
    isfile(path) ||
        throw(ArgumentError("official ELM artifact not found: $path"))
    payload = JLD2.load(path)
    get(payload, "artifact_kind", nothing) ==
        "FrozenOfficialELMTwin" ||
        error("not a FrozenOfficialELMTwin artifact")
    get(payload, "format_version", nothing) == 2 ||
        error("unsupported FrozenOfficialELMTwin artifact version")
    frozen = payload["frozen"]
    frozen isa FrozenOfficialELMTwin ||
        error("artifact payload has the wrong type")
    assert_frozen_official_elm_unchanged(frozen)
    return frozen
end

function load_verified_official_elm(path::AbstractString)
    frozen = load_frozen_official_elm(path)
    hasproperty(frozen.metadata, :verification_passed) &&
        frozen.metadata.verification_passed === true ||
        error("official ELM artifact has not passed held-out verification")
    return frozen
end

end # module PaperELMTwinOfficialV2
