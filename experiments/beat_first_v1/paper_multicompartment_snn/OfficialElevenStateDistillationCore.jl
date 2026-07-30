module OfficialElevenStateDistillationCore

using LinearAlgebra
using Random
using Serialization
using SHA

const _PARENT = parentmodule(@__MODULE__)
if !isdefined(_PARENT, :DistilledElevenStateCellFinal)
    Base.include(
        _PARENT,
        joinpath(@__DIR__, "DistilledElevenStateCellFinal.jl"),
    )
end
const Cell = getfield(_PARENT, :DistilledElevenStateCellFinal)

export SEMANTIC_COORDINATE_NAMES,
    RECURRENT_MASK,
    INPUT_MASK,
    RECURRENT_MASK_SHA256,
    INPUT_MASK_SHA256,
    all_finite,
    freeze_parameters,
    initial_parameters,
    normalize_target,
    physical_output,
    project_official_input,
    semantic_target,
    sequence_loss,
    softmax_columns,
    structured_readout,
    transition

const OFFICIAL_SEGMENTS = 642
const OFFICIAL_DENDRITIC_LOCATIONS = 639
const OFFICIAL_ELM_INPUT_DIM = 1_278
const SEMANTIC_COORDINATE_NAMES = (
    "distal_basal_dendritic_voltage",
    "proximal_apical_trunk_voltage",
    "apical_calcium_hot_zone_voltage",
    "distal_apical_tuft_voltage",
    "soma_nmda_current",
    "basal_nmda_current",
    "apical_trunk_nmda_current",
    "apical_tuft_nmda_current",
    "apical_calcium_context",
    "soma_voltage",
    "calcium_adaptation",
)

@inline sigmoid(value) = inv(one(value) + exp(-value))

function _sha256(value)
    stream = IOBuffer()
    Serialization.serialize(stream, value)
    return bytes2hex(SHA.sha256(take!(stream)))
end

function _recurrent_mask()
    mask = zeros(Float32, 11, 11)
    @inbounds for branch in 1:4
        mask[branch, branch] = 1.0f0
        mask[branch, 4 + branch] = 1.0f0
        mask[branch, 9] = branch >= 3 ? 1.0f0 : 0.25f0
        for neighbour in 1:4
            branch != neighbour &&
                (mask[branch, neighbour] = 0.25f0)
        end
    end
    @inbounds for branch in 1:4
        mask[4 + branch, branch] = 1.0f0
        mask[4 + branch, 4 + branch] = 1.0f0
    end
    for source in (3, 4, 7, 8, 9, 11)
        mask[9, source] = 1.0f0
    end
    mask[10, 1:11] .= 1.0f0
    mask[11, 9] = 1.0f0
    mask[11, 10] = 1.0f0
    mask[11, 11] = 1.0f0
    return mask
end

function _input_mask()
    mask = zeros(Float32, 11, 16)
    @inbounds for branch in 1:4
        mask[branch, branch] = 1.0f0
        mask[branch, 4 + branch] = 1.0f0
        mask[branch, 8 + branch] = 1.0f0
        mask[branch, 12 + branch] = 1.0f0
        mask[4 + branch, branch] = 0.5f0
        mask[4 + branch, 4 + branch] = 1.0f0
        mask[4 + branch, 8 + branch] = 0.25f0
    end
    for branch in 3:4
        for receptor_offset in (0, 4, 8, 12)
            mask[9, receptor_offset + branch] = 1.0f0
        end
    end
    mask[10, :] .= 0.25f0
    return mask
end

const RECURRENT_MASK = _recurrent_mask()
const INPUT_MASK = _input_mask()
const RECURRENT_MASK_SHA256 = _sha256(RECURRENT_MASK)
const INPUT_MASK_SHA256 = _sha256(INPUT_MASK)

function softmax_columns(logits)
    shifted = logits .- maximum(logits; dims=1)
    exponent = exp.(shifted)
    return exponent ./ sum(exponent; dims=1)
end

"""
Project the exact official signed E/I input onto four reduced compartments.

The official ELM input has one excitatory channel per legal location.  AMPA
and NMDA are paired consequences of that same presynaptic event, therefore the
same projected excitatory drive enters both reduced receptor coordinates.
The negative inhibitory half is converted back to nonnegative GABA_A drive.
"""
function project_official_input(raw, location_logits)
    size(raw, 1) == OFFICIAL_ELM_INPUT_DIM ||
        throw(DimensionMismatch("official input must have 1278 rows"))
    size(location_logits) ==
        (4, OFFICIAL_DENDRITIC_LOCATIONS) ||
        throw(DimensionMismatch("location logits must be 4 x 639"))
    location = softmax_columns(location_logits)
    excitatory = @view raw[1:639, :]
    inhibitory = .-(@view raw[640:1278, :])
    all(value -> value >= 0, excitatory) ||
        error("official excitatory input contains a negative value")
    all(value -> value >= 0, inhibitory) ||
        error("official inhibitory input contains a positive value")
    projected_excitatory = location * excitatory
    projected_inhibitory = location * inhibitory
    auxiliary = zeros(eltype(raw), 4, size(raw, 2))
    return vcat(
        projected_excitatory,
        projected_excitatory,
        projected_inhibitory,
        auxiliary,
    )
end

function _state_activation(value)
    return vcat(
        tanh.(value[1:10, :]),
        sigmoid.(value[11:11, :]),
    )
end

function transition(parameters, state, input)
    recurrent = parameters.recurrent_weight .* RECURRENT_MASK
    input_weight = parameters.input_weight .* INPUT_MASK
    proposal = _state_activation(
        recurrent * state .+
        input_weight * input .+
        parameters.transition_bias,
    )
    decay = sigmoid.(parameters.transition_decay_logit)
    return decay .* state .+ (1.0f0 .- decay) .* proposal
end

function normalize_target(target, target_mean, target_scale)
    normalized = (target .- reshape(target_mean, :, 1)) ./
        reshape(target_scale, :, 1)
    return vcat(
        normalized[1:1, :],
        target[2:2, :],
        normalized[3:6, :],
        target[7:7, :],
        normalized[8:11, :],
    )
end

function semantic_target(normalized)
    apical_context = tanh.(
        0.25f0 .* normalized[10:10, :] .+
        0.50f0 .* normalized[11:11, :] .+
        0.25f0 .* normalized[7:7, :],
    )
    return vcat(
        tanh.(normalized[8:11, :]),
        tanh.(normalized[3:6, :]),
        apical_context,
        tanh.(normalized[1:1, :]),
        normalized[7:7, :],
    )
end

function structured_readout(parameters, state)
    gain = parameters.readout_gain
    bias = parameters.readout_bias
    soma = gain[1] .* state[10:10, :] .+ bias[1]
    spike =
        gain[2] .* state[10:10, :] .+
        parameters.spike_adaptation_gain[1] .* state[11:11, :] .+
        bias[2]
    nmda = gain[3:6] .* state[5:8, :] .+ bias[3:6]
    calcium = gain[7] .* state[11:11, :] .+ bias[7]
    dendritic = gain[8:11] .* state[1:4, :] .+ bias[8:11]
    return vcat(soma, spike, nmda, calcium, dendritic)
end

@inline _bce(logit, target) =
    max(logit, zero(logit)) - logit * target +
    log1p(exp(-abs(logit)))

function sequence_loss(
    parameters,
    raw_input,
    target,
    target_mean,
    target_scale,
    free_fraction::Float32,
)
    size(raw_input, 1) == OFFICIAL_ELM_INPUT_DIM ||
        throw(DimensionMismatch("sequence input must have 1278 rows"))
    time_steps = size(raw_input, 2)
    batch = size(raw_input, 3)
    state = repeat(parameters.initial_state, 1, batch)
    continuous_loss = 0.0f0
    spike_loss = 0.0f0
    calcium_loss = 0.0f0
    semantic_loss = 0.0f0
    for time in 1:time_steps
        input = project_official_input(
            raw_input[:, time, :],
            parameters.location_logits,
        )
        predicted_state = transition(parameters, state, input)
        output = structured_readout(parameters, predicted_state)
        normalized = normalize_target(
            target[:, time, :],
            target_mean,
            target_scale,
        )
        semantic = semantic_target(normalized)
        continuous_loss +=
            sum(abs2, output[1:1, :] .- normalized[1:1, :])
        continuous_loss +=
            0.75f0 *
            sum(abs2, output[3:6, :] .- normalized[3:6, :])
        continuous_loss +=
            0.50f0 *
            sum(abs2, output[8:11, :] .- normalized[8:11, :])
        spike_loss += sum(_bce.(
            output[2:2, :],
            normalized[2:2, :],
        ))
        calcium_loss += sum(_bce.(
            output[7:7, :],
            normalized[7:7, :],
        ))
        semantic_loss += sum(abs2, predicted_state .- semantic)
        state =
            free_fraction .* predicted_state .+
            (1.0f0 - free_fraction) .* semantic
    end
    count = Float32(time_steps * batch)
    return continuous_loss / (9.0f0 * count) +
           4.0f0 * spike_loss / count +
           1.5f0 * calcium_loss / count +
           2.0f0 * semantic_loss / (11.0f0 * count) +
           1.0f-5 * (
               sum(abs2, parameters.recurrent_weight) +
               sum(abs2, parameters.input_weight) +
               sum(abs2, parameters.readout_gain)
           )
end

function initial_parameters(rng)
    location_logits =
        -2.0f0 .+
        0.02f0 .* randn(
            rng,
            Float32,
            4,
            OFFICIAL_DENDRITIC_LOCATIONS,
        )
    @inbounds for location in 1:OFFICIAL_DENDRITIC_LOCATIONS
        location_logits[mod1(location, 4), location] += 4.0f0
    end
    recurrent_weight =
        0.08f0 .* randn(rng, Float32, 11, 11) .+
        0.55f0 .* Matrix{Float32}(I, 11, 11)
    recurrent_weight .*= RECURRENT_MASK
    input_weight =
        0.10f0 .* randn(rng, Float32, 11, 16) .* INPUT_MASK
    return (;
        transition_decay_logit=fill(1.4f0, 11),
        recurrent_weight,
        input_weight,
        transition_bias=zeros(Float32, 11, 1),
        readout_gain=ones(Float32, 11, 1),
        readout_bias=zeros(Float32, 11, 1),
        spike_adaptation_gain=Float32[-0.25f0],
        initial_state=zeros(Float32, 11, 1),
        location_logits,
    )
end

function all_finite(parameters)
    return all(
        values -> all(isfinite, values),
        values(parameters),
    )
end

function physical_output(raw, target_mean, target_scale)
    output = vec(raw) .* target_scale .+ target_mean
    output[2] = Float32(sigmoid(raw[2]))
    output[7] = Float32(sigmoid(raw[7]))
    return output
end

function _readout_matrix(parameters)
    matrix = zeros(Float32, 11, 11)
    gain = vec(parameters.readout_gain)
    matrix[1, 10] = gain[1]
    matrix[2, 10] = gain[2]
    matrix[2, 11] = parameters.spike_adaptation_gain[1]
    @inbounds for region in 1:4
        matrix[2 + region, 4 + region] = gain[2 + region]
        matrix[7 + region, region] = gain[7 + region]
    end
    matrix[7, 11] = gain[7]
    return matrix
end

function _official_projection(location)
    size(location) == (4, OFFICIAL_DENDRITIC_LOCATIONS) ||
        throw(DimensionMismatch("learned projection must be 4 x 639"))
    projection = zeros(Float32, 4, OFFICIAL_SEGMENTS)
    projection[:, 2:640] .= location
    return projection
end

function _region_projection(segment_region, projection)
    length(segment_region) == OFFICIAL_SEGMENTS ||
        error("official segment-region map must contain 642 entries")
    result = zeros(Float32, 4, 4)
    count = zeros(Int, 4)
    @inbounds for segment in 1:OFFICIAL_SEGMENTS
        label = lowercase(String(segment_region[segment]))
        region =
            occursin("soma", label) ? 1 :
            occursin("basal", label) ? 2 :
            occursin("tuft", label) ? 4 : 3
        result[:, region] .+= projection[:, segment]
        count[region] += 1
    end
    @inbounds for region in 1:4
        count[region] > 0 ||
            error("official segment map lacks region $region")
        result[:, region] ./= count[region]
    end
    return result
end

function freeze_parameters(
    trained,
    segment_region,
    frozen,
    provenance,
    target_mean,
    target_scale,
    dataset_sha256,
    config_sha256,
)
    location = Float32.(softmax_columns(trained.location_logits))
    projection = _official_projection(location)
    parameters = Cell.DistilledParameters(
        dt_ms=frozen.model.config.delta_t_ms,
        transition_decay=Float32.(
            sigmoid.(trained.transition_decay_logit),
        ),
        recurrent_weight=trained.recurrent_weight .* RECURRENT_MASK,
        input_weight=trained.input_weight .* INPUT_MASK,
        transition_bias=vec(trained.transition_bias),
        readout_weight=_readout_matrix(trained),
        readout_bias=vec(trained.readout_bias),
        target_mean,
        target_scale,
        initial_state=vec(trained.initial_state),
        compartment_projection=projection,
        region_projection=
            _region_projection(segment_region, projection),
        spike_threshold=0.5f0,
        teacher_schema=
            "sealed-official-ELM-v2-primary+Hay-NEURON-final-v2-auxiliary",
        detailed_kernel_hash=provenance.detailed_kernel_hash,
        morphology_hash=provenance.morphology_hash,
        frozen_twin_parameter_hash=frozen.parameter_sha256,
        frozen_twin_artifact_hash=frozen.artifact_sha256,
        distillation_dataset_hash=dataset_sha256,
        distillation_config_hash=config_sha256,
    )
    all_finite(parameters) ||
        error("frozen official 11-state core contains non-finite values")
    return parameters
end

end # module OfficialElevenStateDistillationCore
