module TwinPropParity

using LinearAlgebra
using Random
using Statistics
using Zygote

export MODEL_FAMILY,
    EXCITATORY,
    INHIBITORY,
    PAPER_REFERENCE,
    AfferentCode,
    ParityConfig,
    ParityDataset,
    SynapseCapacity,
    SynapseParameters,
    TwinPropRun,
    build_afferent_code,
    constraint_report,
    decision_probability,
    evaluate_hay_transfer,
    generate_parity_dataset,
    hard_contact_mapping,
    initialize_synapses,
    paper_parity_config,
    parity_loss,
    receptor_event_tensor,
    run_benchmark,
    run_variant,
    soft_contact_distribution,
    train_twinprop,
    twin_predict

const MODEL_FAMILY = "HD-SWSNN-TwinProp"
const EXCITATORY = UInt8(0x01)
const INHIBITORY = UInt8(0x02)

# These are reference values reported by the preprint.  They are never copied
# into a measured-result field.
const PAPER_REFERENCE = (
    source="Aizenbud et al., What can a neuron compute, bioRxiv 2026.06.08.730984v1",
    xor_accuracy=1.0,
    parity_4_full_accuracy=0.994,
    parity_10_full_accuracy=0.9124,
    parity_4_passive_accuracy=0.781,
    parity_4_soma_only_accuracy=0.769,
    parity_4_no_nmda_accuracy=0.738,
    parity_4_lif_accuracy=0.688,
    random_boolean_4_mean_accuracy=0.9912,
    random_boolean_4_std_accuracy=0.0134,
    spike_auroc=0.98576,
)

const _PARENT_MODULE = parentmodule(@__MODULE__)

"""
Configuration of one independently trained XOR/parity arm.

`paper_parity_config(d; scale=:paper)` instantiates the protocol disclosed in
the preprint: 4,000 E plus 4,000 I axons, mixed ON/OFF coding, twenty contacts
per axon, 100-ms patterns, 2.5-ms Gaussian jitter, batch size 32, Adam in the
reported learning-rate range, and fifty epochs.  `scale=:smoke` changes only
the computational scale and is marked as such in result metadata.
"""
Base.@kwdef struct ParityConfig
    dimension::Int = 4
    protocol_scale::Symbol = :paper
    total_excitatory_axons::Int = 4_000
    total_inhibitory_axons::Int = 4_000
    contacts_per_axon::Int = 20
    pattern_ms::Float32 = 100.0f0
    dt_ms::Float32 = 1.0f0
    jitter_sigma_ms::Float32 = 2.5f0
    test_jitter_sigma_ms::Float32 = 2.5f0
    decision_window_ms::Float32 = 50.0f0
    burst_rate_hz::Float32 = 20.0f0
    spikes_per_burst::Int = 2
    train_trials_per_pattern::Int = 4
    test_trials_per_pattern::Int = 8
    batch_size::Int = 32
    epochs::Int = 50
    restarts::Int = 3
    learning_rate::Float32 = 1.5f-3
    adam_beta1::Float32 = 0.9f0
    adam_beta2::Float32 = 0.999f0
    adam_epsilon::Float32 = 1.0f-8
    location_temperature_start::Float32 = 1.0f0
    location_temperature_end::Float32 = 0.15f0
    capacity_penalty::Float32 = 0.05f0
    location_entropy_penalty::Float32 = 1.0f-4
    strength_logit_clip::Float32 = 9.0f0
    location_logit_clip::Float32 = 12.0f0
    seed::UInt64 = 0x5457494e50524f50
    transfer_trace_trials::Int = 4
end

function paper_parity_config(
    dimension::Integer;
    scale::Symbol=:paper,
    kwargs...,
)
    d = Int(dimension)
    d in (2, 4, 6, 8, 10) ||
        throw(ArgumentError("paper parity dimension must be 2, 4, 6, 8, or 10"))
    window = d == 10 ? 25.0f0 : 50.0f0
    if scale === :paper
        return ParityConfig(;
            dimension=d,
            protocol_scale=:paper,
            decision_window_ms=window,
            kwargs...,
        )
    elseif scale === :smoke
        return ParityConfig(;
            dimension=d,
            protocol_scale=:smoke,
            total_excitatory_axons=max(4d, 16),
            total_inhibitory_axons=max(4d, 16),
            contacts_per_axon=2,
            decision_window_ms=window,
            spikes_per_burst=1,
            train_trials_per_pattern=2,
            test_trials_per_pattern=2,
            batch_size=min(16, 2^d),
            epochs=3,
            restarts=1,
            kwargs...,
        )
    end
    throw(ArgumentError("scale must be :paper or :smoke"))
end

function _validate(config::ParityConfig)
    config.dimension >= 2 || throw(ArgumentError("dimension must be >= 2"))
    config.total_excitatory_axons >= 2config.dimension ||
        throw(ArgumentError("too few excitatory axons for ON/OFF coding"))
    config.total_inhibitory_axons >= 2config.dimension ||
        throw(ArgumentError("too few inhibitory axons for ON/OFF coding"))
    config.contacts_per_axon >= 1 ||
        throw(ArgumentError("contacts_per_axon must be positive"))
    config.pattern_ms > 0.0f0 || throw(ArgumentError("pattern_ms must be positive"))
    config.dt_ms > 0.0f0 || throw(ArgumentError("dt_ms must be positive"))
    0.0f0 < config.decision_window_ms <= config.pattern_ms ||
        throw(ArgumentError("invalid decision window"))
    config.jitter_sigma_ms >= 0.0f0 ||
        throw(ArgumentError("jitter sigma cannot be negative"))
    config.batch_size >= 1 || throw(ArgumentError("batch size must be positive"))
    config.epochs >= 1 || throw(ArgumentError("epochs must be positive"))
    1 <= config.restarts <= 100 ||
        throw(ArgumentError("restarts must be in 1:100"))
    1.0f-3 <= config.learning_rate <= 2.0f-3 ||
        throw(ArgumentError("paper protocol Adam learning rate must be in [0.001,0.002]"))
    return config
end

"""
Fixed presynaptic population.  Axon type and ON/OFF selectivity never change,
which enforces Dale's law independently of the optimized conductance.
"""
struct AfferentCode
    dimension::Int
    axon_bit::Vector{Int16}
    active_value::Vector{UInt8}
    kind::Vector{UInt8}
    target_time_ms::Vector{Float32}
end

@inline axon_count(code::AfferentCode) = length(code.kind)

function _assign_population!(
    axon_bit::Vector{Int16},
    active_value::Vector{UInt8},
    kind::Vector{UInt8},
    target_time_ms::Vector{Float32},
    first_index::Int,
    count::Int,
    kind_value::UInt8,
    dimension::Int,
    pattern_ms::Float32,
)
    @inbounds for local_index in 1:count
        index = first_index + local_index - 1
        axon_bit[index] = Int16(mod1(local_index, dimension))
        active_value[index] = UInt8((div(local_index - 1, dimension) & 1))
        kind[index] = kind_value
        fraction = Float32(local_index - 0.5f0) / Float32(count)
        target_time_ms[index] = fraction * pattern_ms
    end
    return nothing
end

function build_afferent_code(config::ParityConfig)
    _validate(config)
    excitatory = config.total_excitatory_axons
    inhibitory = config.total_inhibitory_axons
    total = excitatory + inhibitory
    axon_bit = Vector{Int16}(undef, total)
    active_value = Vector{UInt8}(undef, total)
    kind = Vector{UInt8}(undef, total)
    target_time_ms = Vector{Float32}(undef, total)
    _assign_population!(
        axon_bit,
        active_value,
        kind,
        target_time_ms,
        1,
        excitatory,
        EXCITATORY,
        config.dimension,
        config.pattern_ms,
    )
    _assign_population!(
        axon_bit,
        active_value,
        kind,
        target_time_ms,
        excitatory + 1,
        inhibitory,
        INHIBITORY,
        config.dimension,
        config.pattern_ms,
    )
    return AfferentCode(
        config.dimension,
        axon_bit,
        active_value,
        kind,
        target_time_ms,
    )
end

"""
An exhaustive truth-table dataset with independent temporal jitter trials.

`spikes` has shape axon × millisecond × trial.  Labels are determined only by
odd parity.  Clean and held-out jitter sets are generated with independent
seeds while retaining the same fixed presynaptic axons.
"""
struct ParityDataset
    dimension::Int
    bits::Matrix{UInt8}
    target::Vector{Float32}
    spikes::Array{Float32,3}
    dt_ms::Float32
    decision_first_step::Int
    jitter_sigma_ms::Float32
    split::Symbol
end

@inline trial_count(dataset::ParityDataset) = length(dataset.target)
@inline time_steps(dataset::ParityDataset) = size(dataset.spikes, 2)

function _truth_bits!(bits, target, pattern::Int, trial::Int, dimension::Int)
    ones_count = 0
    @inbounds for bit in 1:dimension
        value = UInt8((pattern >> (bit - 1)) & 1)
        bits[bit, trial] = value
        ones_count += Int(value)
    end
    target[trial] = Float32(isodd(ones_count))
    return nothing
end

function generate_parity_dataset(
    code::AfferentCode,
    config::ParityConfig;
    split::Symbol=:train,
    jitter_sigma_ms::Real=split === :clean ? 0.0f0 :
                         split === :test ? config.test_jitter_sigma_ms :
                         config.jitter_sigma_ms,
    trials_per_pattern::Integer=split === :train ?
                                config.train_trials_per_pattern :
                                config.test_trials_per_pattern,
    seed::Integer=config.seed + (split === :train ? 0x101 : split === :test ? 0x202 : 0x303),
)
    code.dimension == config.dimension ||
        throw(DimensionMismatch("afferent code/config dimension mismatch"))
    repetitions = Int(trials_per_pattern)
    repetitions >= 1 || throw(ArgumentError("trials_per_pattern must be positive"))
    patterns = 1 << config.dimension
    trials = patterns * repetitions
    steps = round(Int, config.pattern_ms / config.dt_ms)
    window_steps = round(Int, config.decision_window_ms / config.dt_ms)
    decision_first = steps - window_steps + 1
    bits = Matrix{UInt8}(undef, config.dimension, trials)
    target = Vector{Float32}(undef, trials)
    spikes = zeros(Float32, axon_count(code), steps, trials)
    rng = Xoshiro(UInt64(seed))
    sigma = Float32(jitter_sigma_ms)
    interval_ms = 1_000.0f0 / config.burst_rate_hz

    trial = 0
    @inbounds for repeat in 1:repetitions
        for pattern in 0:(patterns - 1)
            trial += 1
            _truth_bits!(bits, target, pattern, trial, config.dimension)
            for axon in 1:axon_count(code)
                bit = Int(code.axon_bit[axon])
                bits[bit, trial] == code.active_value[axon] || continue
                for burst_index in 1:config.spikes_per_burst
                    centered_offset =
                        (
                            Float32(burst_index) -
                            0.5f0 * Float32(config.spikes_per_burst + 1)
                        ) * interval_ms
                    jitter = sigma == 0.0f0 ? 0.0f0 : sigma * randn(rng, Float32)
                    spike_time =
                        code.target_time_ms[axon] + centered_offset + jitter
                    step = clamp(
                        round(Int, spike_time / config.dt_ms) + 1,
                        1,
                        steps,
                    )
                    spikes[axon, step, trial] += 1.0f0
                end
            end
        end
    end
    return ParityDataset(
        config.dimension,
        bits,
        target,
        spikes,
        config.dt_ms,
        decision_first,
        sigma,
        split,
    )
end

"""
Per-receptor anatomical capacity for the reduced segment representation.

The paper's density bound is one E and one I contact per micrometre.  A
reduced compartment represents many micrometre sites, so the capacity vector
stores their aggregate count separately for E and I.
"""
struct SynapseCapacity
    excitatory::Vector{Int32}
    inhibitory::Vector{Int32}
    allowed::BitVector
end

function _balanced_capacity(
    segment_weight::AbstractVector{<:Real},
    total::Int,
    allowed::AbstractVector{Bool},
)
    total >= 0 || throw(ArgumentError("capacity total cannot be negative"))
    weight = Float64[
        allowed[index] ? max(Float64(segment_weight[index]), eps(Float64)) : 0.0
        for index in eachindex(segment_weight)
    ]
    sum(weight) > 0.0 || throw(ArgumentError("no allowed synaptic segment"))
    raw = total .* weight ./ sum(weight)
    capacity = Int32.(floor.(raw))
    remainder = total - sum(capacity)
    order = sortperm(raw .- capacity; rev=true)
    @inbounds for index in 1:remainder
        capacity[order[index]] += Int32(1)
    end
    return capacity
end

function SynapseCapacity(
    segment_weight::AbstractVector{<:Real},
    code::AfferentCode,
    config::ParityConfig;
    allowed::AbstractVector{Bool}=trues(length(segment_weight)),
    reserve_fraction::Real=0.05,
)
    length(allowed) == length(segment_weight) ||
        throw(DimensionMismatch("allowed/segment length mismatch"))
    excitatory_axons = count(==(EXCITATORY), code.kind)
    inhibitory_axons = count(==(INHIBITORY), code.kind)
    e_total = ceil(
        Int,
        excitatory_axons *
        config.contacts_per_axon *
        (1 + Float64(reserve_fraction)),
    )
    i_total = ceil(
        Int,
        inhibitory_axons *
        config.contacts_per_axon *
        (1 + Float64(reserve_fraction)),
    )
    return SynapseCapacity(
        _balanced_capacity(segment_weight, e_total, allowed),
        _balanced_capacity(segment_weight, i_total, allowed),
        BitVector(allowed),
    )
end

"""
Only these arrays are optimized in TwinProp step 2.  The digital twin,
morphology, channel kinetics, axon identity, and output rule remain frozen.
"""
struct SynapseParameters
    strength_logit::Matrix{Float32}
    location_logit::Matrix{Float32}
end

function initialize_synapses(
    rng::AbstractRNG,
    segments::Integer,
    code::AfferentCode,
    capacity::SynapseCapacity,
)
    count = Int(segments)
    count == length(capacity.allowed) ||
        throw(DimensionMismatch("segment/capacity mismatch"))
    axons = axon_count(code)
    # Uniform normalized conductance, as disclosed by the paper.
    normalized = clamp.(
        rand(rng, Float32, count, axons),
        1.0f-4,
        1.0f0 - 1.0f-4,
    )
    strength = log.(normalized ./ (1.0f0 .- normalized))
    location = 0.15f0 .* randn(rng, Float32, count, axons)
    @inbounds for segment in 1:count
        if !capacity.allowed[segment]
            location[segment, :] .= -20.0f0
        end
    end
    return SynapseParameters(strength, location)
end

@inline _logistic(value) = inv(one(value) + exp(-value))

function soft_contact_distribution(
    location_logit::AbstractMatrix,
    allowed::AbstractVector{Bool},
    temperature::Real,
)
    size(location_logit, 1) == length(allowed) ||
        throw(DimensionMismatch("location/allowed mismatch"))
    temperature > 0 || throw(ArgumentError("temperature must be positive"))
    mask = reshape(
        Float32[flag ? 0.0f0 : -1.0f9 for flag in allowed],
        :,
        1,
    )
    scaled = (location_logit .+ mask) ./ Float32(temperature)
    maximum_per_axon = maximum(scaled; dims=1)
    unnormalized = exp.(scaled .- maximum_per_axon)
    return unnormalized ./ sum(unnormalized; dims=1)
end

function _kind_masks(code::AfferentCode)
    excitatory = reshape(Float32.(code.kind .== EXCITATORY), 1, :)
    inhibitory = reshape(Float32.(code.kind .== INHIBITORY), 1, :)
    return excitatory, inhibitory
end

function _effective_contact_matrix(
    parameters::SynapseParameters,
    code::AfferentCode,
    capacity::SynapseCapacity,
    config::ParityConfig,
    temperature::Real,
)
    distribution = soft_contact_distribution(
        parameters.location_logit,
        capacity.allowed,
        temperature,
    )
    contact_mass = Float32(config.contacts_per_axon) .* distribution
    normalized_strength = _logistic.(parameters.strength_logit)
    effective = contact_mass .* normalized_strength
    excitatory, inhibitory = _kind_masks(code)
    return effective .* excitatory, effective .* inhibitory, distribution
end

"""
Build the differentiable receptor-event input for the frozen digital twin.

The result has shape `(3 * segments, time, batch)` with AMPA, NMDA and GABAA
blocks in that order.  Values are non-negative normalized contact amplitudes;
the canonical detailed kernel applies the paper maxima 0.4/0.3/0.7 nS.
Excitatory axons emit paired AMPA+NMDA events, while inhibitory axons emit
GABAA only.
"""
function receptor_event_tensor(
    parameters::SynapseParameters,
    code::AfferentCode,
    capacity::SynapseCapacity,
    config::ParityConfig,
    spikes::AbstractArray{<:Real,3};
    temperature::Real=config.location_temperature_end,
)
    axon_count(code) == size(spikes, 1) ||
        throw(DimensionMismatch("spike/axon mismatch"))
    excitatory, inhibitory, _ = _effective_contact_matrix(
        parameters,
        code,
        capacity,
        config,
        temperature,
    )
    flattened_spikes = reshape(spikes, size(spikes, 1), :)
    excitatory_events = excitatory * flattened_spikes
    inhibitory_events = inhibitory * flattened_spikes
    combined = vcat(
        excitatory_events,
        excitatory_events,
        inhibitory_events,
    )
    return reshape(
        combined,
        3size(parameters.location_logit, 1),
        size(spikes, 2),
        size(spikes, 3),
    )
end

"""
Interface point for a frozen digital twin.

External tests or alternative twins may add a specialized `twin_predict`
method.  The fallback calls `PaperDigitalTwin.twin_forward`, preferring the
`FrozenTwin` convenience method and otherwise using its frozen model and
parameter fields.
"""
function twin_predict(frozen_twin, input)
    if isdefined(_PARENT_MODULE, :PaperDigitalTwin)
        module_value = getfield(_PARENT_MODULE, :PaperDigitalTwin)
        forward = getfield(module_value, :twin_forward)
        if applicable(forward, frozen_twin, input)
            return forward(frozen_twin, input)
        elseif hasproperty(frozen_twin, :model) &&
               hasproperty(frozen_twin, :parameters)
            return forward(
                getproperty(frozen_twin, :model),
                getproperty(frozen_twin, :parameters),
                input,
            )
        elseif hasproperty(frozen_twin, :model) &&
               hasproperty(frozen_twin, :ps)
            return forward(
                getproperty(frozen_twin, :model),
                getproperty(frozen_twin, :ps),
                input,
            )
        end
    end
    throw(
        ArgumentError(
            "no twin_predict method for $(typeof(frozen_twin)); load PaperDigitalTwin or extend TwinPropParity.twin_predict",
        ),
    )
end

function _spike_probability(output)
    value = if hasproperty(output, :spike_probability)
        getproperty(output, :spike_probability)
    elseif output isa Tuple && !isempty(output)
        _spike_probability(first(output))
    else
        throw(ArgumentError("digital twin output lacks spike_probability"))
    end
    if ndims(value) == 3
        size(value, 1) == 1 ||
            throw(DimensionMismatch("spike probability must have singleton channel"))
        return dropdims(value; dims=1)
    elseif ndims(value) == 2
        return value
    end
    throw(DimensionMismatch("spike probability must be time × batch"))
end

function decision_probability(
    spike_probability::AbstractMatrix,
    decision_first_step::Integer,
)
    first_step = Int(decision_first_step)
    1 <= first_step <= size(spike_probability, 1) ||
        throw(BoundsError(spike_probability, first_step))
    probability = clamp.(
        @view(spike_probability[first_step:end, :]),
        1.0f-6,
        1.0f0 - 1.0f-6,
    )
    # Probability of one or more soma spikes; no voltage or dendritic state is
    # exposed to the classifier.
    no_spike_log_probability = sum(log1p.(-probability); dims=1)
    return vec(1.0f0 .- exp.(no_spike_log_probability))
end

@inline function _binary_cross_entropy(predicted, target)
    probability = clamp.(predicted, 1.0f-6, 1.0f0 - 1.0f-6)
    return -mean(
        target .* log.(probability) .+
        (1.0f0 .- target) .* log1p.(-probability),
    )
end

function _capacity_loss(
    distribution,
    code::AfferentCode,
    capacity::SynapseCapacity,
    config::ParityConfig,
)
    excitatory, inhibitory = _kind_masks(code)
    mass = Float32(config.contacts_per_axon) .* distribution
    expected_e = vec(sum(mass .* excitatory; dims=2))
    expected_i = vec(sum(mass .* inhibitory; dims=2))
    e_capacity = Float32.(capacity.excitatory)
    i_capacity = Float32.(capacity.inhibitory)
    e_violation = max.(expected_e .- e_capacity, 0.0f0)
    i_violation = max.(expected_i .- i_capacity, 0.0f0)
    return mean((e_violation ./ max.(e_capacity, 1.0f0)).^2) +
           mean((i_violation ./ max.(i_capacity, 1.0f0)).^2)
end

function _location_entropy(distribution)
    return -mean(sum(distribution .* log.(distribution .+ 1.0f-8); dims=1))
end

function _loss_components(
    parameters::SynapseParameters,
    frozen_twin,
    code::AfferentCode,
    dataset::ParityDataset,
    capacity::SynapseCapacity,
    config::ParityConfig,
    indices;
    temperature::Real,
)
    spikes = @view dataset.spikes[:, :, indices]
    target = @view dataset.target[indices]
    input = receptor_event_tensor(
        parameters,
        code,
        capacity,
        config,
        spikes;
        temperature,
    )
    output = twin_predict(frozen_twin, input)
    probability = decision_probability(
        _spike_probability(output),
        dataset.decision_first_step,
    )
    distribution = soft_contact_distribution(
        parameters.location_logit,
        capacity.allowed,
        temperature,
    )
    bce = _binary_cross_entropy(probability, target)
    capacity_value = _capacity_loss(distribution, code, capacity, config)
    entropy = _location_entropy(distribution)
    total =
        bce +
        config.capacity_penalty * capacity_value +
        config.location_entropy_penalty * entropy
    return (;
        total,
        bce,
        capacity=capacity_value,
        entropy,
        probability,
    )
end

function parity_loss(
    parameters::SynapseParameters,
    frozen_twin,
    code::AfferentCode,
    dataset::ParityDataset,
    capacity::SynapseCapacity,
    config::ParityConfig;
    indices=1:trial_count(dataset),
    temperature::Real=config.location_temperature_end,
)
    return _loss_components(
        parameters,
        frozen_twin,
        code,
        dataset,
        capacity,
        config,
        indices;
        temperature,
    ).total
end

function _temperature(config::ParityConfig, epoch::Int)
    config.epochs == 1 && return config.location_temperature_end
    fraction = Float32(epoch - 1) / Float32(config.epochs - 1)
    log_value = muladd(
        fraction,
        log(config.location_temperature_end) -
        log(config.location_temperature_start),
        log(config.location_temperature_start),
    )
    return exp(log_value)
end

mutable struct _AdamState
    strength_m::Matrix{Float32}
    strength_v::Matrix{Float32}
    location_m::Matrix{Float32}
    location_v::Matrix{Float32}
    step::Int
end

function _AdamState(parameters::SynapseParameters)
    return _AdamState(
        zeros(Float32, size(parameters.strength_logit)),
        zeros(Float32, size(parameters.strength_logit)),
        zeros(Float32, size(parameters.location_logit)),
        zeros(Float32, size(parameters.location_logit)),
        0,
    )
end

function _adam_array_step!(
    value,
    gradient,
    first,
    second,
    step::Int,
    config::ParityConfig,
    clip_value::Float32,
)
    gradient === nothing && return value
    beta1 = config.adam_beta1
    beta2 = config.adam_beta2
    correction1 = 1.0f0 - beta1^step
    correction2 = 1.0f0 - beta2^step
    @inbounds for index in eachindex(value)
        grad = Float32(gradient[index])
        first[index] = muladd(beta1, first[index], (1.0f0 - beta1) * grad)
        second[index] = muladd(
            beta2,
            second[index],
            (1.0f0 - beta2) * grad * grad,
        )
        corrected_first = first[index] / correction1
        corrected_second = second[index] / correction2
        value[index] = clamp(
            value[index] -
            config.learning_rate * corrected_first /
            (sqrt(corrected_second) + config.adam_epsilon),
            -clip_value,
            clip_value,
        )
    end
    return value
end

function _adam_step!(
    parameters::SynapseParameters,
    gradient,
    state::_AdamState,
    config::ParityConfig,
)
    state.step += 1
    _adam_array_step!(
        parameters.strength_logit,
        gradient.strength_logit,
        state.strength_m,
        state.strength_v,
        state.step,
        config,
        config.strength_logit_clip,
    )
    _adam_array_step!(
        parameters.location_logit,
        gradient.location_logit,
        state.location_m,
        state.location_v,
        state.step,
        config,
        config.location_logit_clip,
    )
    return parameters
end

function _dataset_accuracy(
    parameters,
    frozen_twin,
    code,
    dataset,
    capacity,
    config;
    temperature=config.location_temperature_end,
)
    components = _loss_components(
        parameters,
        frozen_twin,
        code,
        dataset,
        capacity,
        config,
        1:trial_count(dataset);
        temperature,
    )
    predicted = components.probability .>= 0.5f0
    target = dataset.target .>= 0.5f0
    return (
        accuracy=count(predicted .== target) / length(target),
        loss=Float64(components.total),
        bce=Float64(components.bce),
        capacity_loss=Float64(components.capacity),
        location_entropy=Float64(components.entropy),
    )
end

"""
Exact hard projection used for detailed-cell transfer.

Each axon receives exactly `contacts_per_axon` integer contacts.  Placement is
greedy in descending learned logit while respecting independent E/I segment
capacities.  Repeated contacts in a reduced compartment represent distinct
one-micrometre anatomical sites inside that aggregate compartment.
"""
function hard_contact_mapping(
    parameters::SynapseParameters,
    code::AfferentCode,
    capacity::SynapseCapacity,
    config::ParityConfig,
)
    segments, axons = size(parameters.location_logit)
    axons == axon_count(code) || throw(DimensionMismatch("location/axon mismatch"))
    counts = zeros(Int16, segments, axons)
    used_e = zeros(Int32, segments)
    used_i = zeros(Int32, segments)
    allowed_indices = findall(capacity.allowed)
    isempty(allowed_indices) && throw(ArgumentError("no allowed location"))

    @inbounds for axon in 1:axons
        used = code.kind[axon] == EXCITATORY ? used_e : used_i
        limit = code.kind[axon] == EXCITATORY ?
                capacity.excitatory : capacity.inhibitory
        for _ in 1:config.contacts_per_axon
            best_segment = 0
            best_score = -Inf32
            for segment in allowed_indices
                used[segment] < limit[segment] || continue
                own_count = Float32(counts[segment, axon])
                load = Float32(used[segment]) / Float32(max(limit[segment], 1))
                score =
                    parameters.location_logit[segment, axon] -
                    log1p(own_count) -
                    0.25f0 * load
                if score > best_score
                    best_score = score
                    best_segment = segment
                end
            end
            best_segment != 0 ||
                throw(ArgumentError("insufficient $(code.kind[axon] == EXCITATORY ? "E" : "I") contact capacity"))
            counts[best_segment, axon] += Int16(1)
            used[best_segment] += Int32(1)
        end
    end
    return counts
end

function constraint_report(
    parameters::SynapseParameters,
    hard_mapping::AbstractMatrix{<:Integer},
    code::AfferentCode,
    capacity::SynapseCapacity,
    config::ParityConfig,
)
    strength = _logistic.(parameters.strength_logit)
    e_mask = reshape(code.kind .== EXCITATORY, 1, :)
    i_mask = reshape(code.kind .== INHIBITORY, 1, :)
    e_count = vec(sum(hard_mapping .* e_mask; dims=2))
    i_count = vec(sum(hard_mapping .* i_mask; dims=2))
    contacts = vec(sum(hard_mapping; dims=1))
    return (
        dale_law_fixed=all(
            kind == EXCITATORY || kind == INHIBITORY for kind in code.kind
        ),
        normalized_conductance_min=Float64(minimum(strength)),
        normalized_conductance_max=Float64(maximum(strength)),
        nonnegative_conductance=minimum(strength) >= 0.0f0,
        receptor_maxima_ns=(ampa=0.4, nmda=0.3, gabaa=0.7),
        exact_contacts_per_axon=all(==(config.contacts_per_axon), contacts),
        excitatory_capacity_respected=all(e_count .<= capacity.excitatory),
        inhibitory_capacity_respected=all(i_count .<= capacity.inhibitory),
        all_locations_allowed=all(
            hard_mapping[segment, axon] == 0 || capacity.allowed[segment]
            for segment in axes(hard_mapping, 1),
                axon in axes(hard_mapping, 2)
        ),
    )
end

function _hard_event_tensor(
    parameters::SynapseParameters,
    mapping::AbstractMatrix{<:Integer},
    code::AfferentCode,
    spikes::AbstractArray{<:Real,3},
)
    strength = _logistic.(parameters.strength_logit)
    effective = Float32.(mapping) .* strength
    excitatory, inhibitory = _kind_masks(code)
    flattened_spikes = reshape(spikes, size(spikes, 1), :)
    e = (effective .* excitatory) * flattened_spikes
    i = (effective .* inhibitory) * flattened_spikes
    return reshape(
        vcat(e, e, i),
        3size(mapping, 1),
        size(spikes, 2),
        size(spikes, 3),
    )
end

function _paper_hay_module()
    if !isdefined(_PARENT_MODULE, :PaperHayCell)
        path = joinpath(@__DIR__, "PaperHayCell.jl")
        isfile(path) ||
            throw(ArgumentError("canonical PaperHayCell.jl is unavailable"))
        Base.include(_PARENT_MODULE, path)
    end
    return getfield(_PARENT_MODULE, :PaperHayCell)
end

function _variant_allowed_segments(tree, variant::Symbol)
    hay = _paper_hay_module()
    if variant === :soma_only
        return BitVector([
            index == Int(tree.soma) for index in 1:hay.compartment_count(tree)
        ])
    end
    return BitVector([
        tree.region[index] != hay.SOMA for index in 1:hay.compartment_count(tree)
    ])
end

function _variant_capacity(tree, code, config, variant)
    allowed = _variant_allowed_segments(tree, variant)
    # Area is used as the number of represented one-micrometre sites.  The
    # capacity constructor rescales this reduced morphology to the requested
    # total while preserving relative anatomical area.
    return SynapseCapacity(tree.area_um2, code, config; allowed)
end

"""
Map learned hard locations/strengths back to the canonical detailed Hay cell.

Classification uses only whether the soma emitted at least one spike in the
decision window.  Dendritic voltage and NMDA current are returned strictly as
diagnostics.  Each ablation must call `train_twinprop` separately before this
function; this function never applies a post-hoc ablation to a full-model
solution.
"""
function evaluate_hay_transfer(
    parameters::SynapseParameters,
    mapping::AbstractMatrix{<:Integer},
    code::AfferentCode,
    dataset::ParityDataset,
    config::ParityConfig;
    variant::Symbol=:full,
    trace_trials::Integer=config.transfer_trace_trials,
)
    variant in (:full, :passive, :no_nmda, :soma_only) ||
        throw(ArgumentError("unknown Hay ablation $variant"))
    hay = _paper_hay_module()
    tree = hay.paper_hay_tree()
    size(mapping, 1) == hay.compartment_count(tree) ||
        throw(DimensionMismatch("hard mapping does not match canonical Hay tree"))
    hay_parameters = hay.HayParameters(tree; ablation=variant)
    state = hay.HayState(tree, hay_parameters)
    drive = hay.HaySynapticDrive(tree)
    diagnostics = hay.HayDiagnostics(tree)
    events = _hard_event_tensor(parameters, mapping, code, dataset.spikes)
    segments = hay.compartment_count(tree)
    trials = trial_count(dataset)
    predicted = falses(trials)
    spike_count = zeros(Int32, trials)
    trace_count = min(Int(trace_trials), trials)
    voltage_trace = zeros(Float32, segments, time_steps(dataset), trace_count)
    nmda_trace = zeros(Float32, segments, time_steps(dataset), trace_count)

    @inbounds for trial in 1:trials
        hay.reset_state!(state, hay_parameters)
        hay.reset_drive!(drive)
        hay.reset_diagnostics!(diagnostics)
        for step in 1:time_steps(dataset)
            for segment in 1:segments
                drive.ampa_event[segment] = events[segment, step, trial]
                drive.nmda_event[segment] =
                    events[segments + segment, step, trial]
                drive.gaba_event[segment] =
                    events[2segments + segment, step, trial]
            end
            spike = hay.hay_cell_step!(
                state,
                drive,
                diagnostics,
                tree,
                hay_parameters,
            )
            hay.reset_drive!(drive)
            if step >= dataset.decision_first_step && spike > 0.5f0
                predicted[trial] = true
                spike_count[trial] += Int32(1)
            end
            if trial <= trace_count
                voltage_trace[:, step, trial] .= state.voltage_mv
                nmda_trace[:, step, trial] .= diagnostics.nmda_current
            end
        end
    end
    target = dataset.target .>= 0.5f0
    return (
        accuracy=count(predicted .== target) / trials,
        predictions=predicted,
        target,
        spike_count,
        voltage_trace,
        nmda_current_trace=nmda_trace,
        mean_abs_nmda_current=mean(abs, nmda_trace),
        readout="at_least_one_soma_spike_in_decision_window",
        analog_bypass=false,
        canonical_kernel="PaperHayCell",
        variant=String(variant),
    )
end

struct TwinPropRun
    parameters::SynapseParameters
    hard_mapping::Matrix{Int16}
    restart::Int
    loss_history::Vector{NamedTuple}
    train_metrics::NamedTuple
    test_metrics::NamedTuple
    clean_metrics::NamedTuple
    constraints::NamedTuple
end

function train_twinprop(
    frozen_twin,
    code::AfferentCode,
    train_dataset::ParityDataset,
    test_dataset::ParityDataset,
    clean_dataset::ParityDataset,
    capacity::SynapseCapacity,
    config::ParityConfig,
)
    _validate(config)
    segments = length(capacity.allowed)
    best = nothing
    best_score = -Inf
    for restart in 1:config.restarts
        rng = Xoshiro(config.seed + UInt64(0x10000 * restart))
        parameters = initialize_synapses(rng, segments, code, capacity)
        optimizer = _AdamState(parameters)
        history = NamedTuple[]
        order = collect(1:trial_count(train_dataset))
        for epoch in 1:config.epochs
            shuffle!(rng, order)
            temperature = _temperature(config, epoch)
            for first_index in 1:config.batch_size:length(order)
                last_index = min(
                    first_index + config.batch_size - 1,
                    length(order),
                )
                indices = @view order[first_index:last_index]
                gradient = only(Zygote.gradient(parameters) do current
                    parity_loss(
                        current,
                        frozen_twin,
                        code,
                        train_dataset,
                        capacity,
                        config;
                        indices,
                        temperature,
                    )
                end)
                _adam_step!(parameters, gradient, optimizer, config)
            end
            train_metrics = _dataset_accuracy(
                parameters,
                frozen_twin,
                code,
                train_dataset,
                capacity,
                config;
                temperature,
            )
            push!(
                history,
                (
                    epoch=epoch,
                    temperature=Float64(temperature),
                    loss=train_metrics.loss,
                    bce=train_metrics.bce,
                    accuracy=train_metrics.accuracy,
                    capacity_loss=train_metrics.capacity_loss,
                    location_entropy=train_metrics.location_entropy,
                ),
            )
        end
        train_metrics = _dataset_accuracy(
            parameters,
            frozen_twin,
            code,
            train_dataset,
            capacity,
            config,
        )
        test_metrics = _dataset_accuracy(
            parameters,
            frozen_twin,
            code,
            test_dataset,
            capacity,
            config,
        )
        clean_metrics = _dataset_accuracy(
            parameters,
            frozen_twin,
            code,
            clean_dataset,
            capacity,
            config,
        )
        mapping = hard_contact_mapping(parameters, code, capacity, config)
        constraints = constraint_report(
            parameters,
            mapping,
            code,
            capacity,
            config,
        )
        score = test_metrics.accuracy - 1.0f-3 * test_metrics.bce
        if score > best_score
            best_score = score
            best = TwinPropRun(
                SynapseParameters(
                    copy(parameters.strength_logit),
                    copy(parameters.location_logit),
                ),
                mapping,
                restart,
                history,
                train_metrics,
                test_metrics,
                clean_metrics,
                constraints,
            )
        end
    end
    return something(best)
end

function _reference_accuracy(dimension::Int, variant::Symbol)
    dimension == 2 && variant === :full && return PAPER_REFERENCE.xor_accuracy
    if dimension == 4
        variant === :full && return PAPER_REFERENCE.parity_4_full_accuracy
        variant === :passive && return PAPER_REFERENCE.parity_4_passive_accuracy
        variant === :soma_only && return PAPER_REFERENCE.parity_4_soma_only_accuracy
        variant === :no_nmda && return PAPER_REFERENCE.parity_4_no_nmda_accuracy
    end
    dimension == 10 && variant === :full &&
        return PAPER_REFERENCE.parity_10_full_accuracy
    return nothing
end

function run_variant(
    frozen_twin,
    config::ParityConfig;
    variant::Symbol=:full,
    transfer::Bool=true,
)
    tree_module = _paper_hay_module()
    tree = tree_module.paper_hay_tree()
    code = build_afferent_code(config)
    capacity = _variant_capacity(tree, code, config, variant)
    train_dataset = generate_parity_dataset(code, config; split=:train)
    test_dataset = generate_parity_dataset(code, config; split=:test)
    clean_dataset = generate_parity_dataset(code, config; split=:clean)
    run = train_twinprop(
        frozen_twin,
        code,
        train_dataset,
        test_dataset,
        clean_dataset,
        capacity,
        config,
    )
    transfer_test = transfer ? evaluate_hay_transfer(
        run.parameters,
        run.hard_mapping,
        code,
        test_dataset,
        config;
        variant,
    ) : nothing
    transfer_clean = transfer ? evaluate_hay_transfer(
        run.parameters,
        run.hard_mapping,
        code,
        clean_dataset,
        config;
        variant,
    ) : nothing
    reference = _reference_accuracy(config.dimension, variant)
    measured = transfer ? transfer_test.accuracy : run.test_metrics.accuracy
    return (
        model_family=MODEL_FAMILY,
        task=config.dimension == 2 ? "xor" : "parity",
        dimension=config.dimension,
        variant=String(variant),
        independently_retrained=true,
        protocol_scale=String(config.protocol_scale),
        restart=run.restart,
        twin_train=run.train_metrics,
        twin_heldout_jitter=run.test_metrics,
        twin_clean=run.clean_metrics,
        transfer_heldout_jitter_accuracy=transfer ? transfer_test.accuracy : nothing,
        transfer_clean_accuracy=transfer ? transfer_clean.accuracy : nothing,
        transfer_mean_abs_nmda_current=transfer ?
            transfer_test.mean_abs_nmda_current : nothing,
        detailed_readout=transfer ? transfer_test.readout :
                         "not_run_twin_metric_is_not_transfer_accuracy",
        paper_reported_accuracy=reference,
        absolute_gap_to_paper=reference === nothing ? nothing :
                              abs(measured - reference),
        reproduction_within_2pp=reference === nothing ? nothing :
                                abs(measured - reference) <= 0.02,
        constraints=run.constraints,
        loss_history=run.loss_history,
        learned=(
            normalized_conductance=_logistic.(run.parameters.strength_logit),
            hard_location=run.hard_mapping,
        ),
    )
end

"""
Run all requested parity dimensions and independently retrained ablations.

`twins` must provide a separately fitted/frozen digital twin for every
ablation key.  Passing the full twin for all arms is rejected by default at
the CLI layer because it would not reproduce the paper's retrained ablations.
"""
function run_benchmark(
    twins::AbstractDict;
    dimensions=(2, 4, 6),
    variants=(:full, :passive, :no_nmda, :soma_only),
    scale::Symbol=:paper,
    transfer::Bool=true,
    config_overrides=NamedTuple(),
)
    results = NamedTuple[]
    for dimension in dimensions
        for variant in variants
            haskey(twins, variant) ||
                throw(KeyError("missing independently fitted twin for $variant"))
            config = paper_parity_config(
                dimension;
                scale,
                config_overrides...,
            )
            push!(
                results,
                run_variant(
                    twins[variant],
                    config;
                    variant,
                    transfer,
                ),
            )
        end
    end
    return (
        schema="hd_swsnn_twinprop_parity_v1",
        model_family=MODEL_FAMILY,
        generated_at=string(Dates.now()),
        paper_reference=PAPER_REFERENCE,
        measured=results,
        disclosure=(
            canonical_cell="mechanism-faithful reduced Hay cable tree, not segment-identical NEURON",
            public_author_code_available=false,
            twin_only_accuracy_is_not_counted_as_reproduction=true,
            all_ablations_independently_retrained=true,
        ),
    )
end

end # module
