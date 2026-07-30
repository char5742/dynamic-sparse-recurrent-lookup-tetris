using Dates
using JLD2
using JSON3
using LinearAlgebra
using Optimisers
using Printf
using Random
using Serialization
using SHA
using Statistics
using Zygote

if !isdefined(Main, :PaperDigitalTwin)
    include(joinpath(@__DIR__, "PaperDigitalTwin.jl"))
end
if !isdefined(Main, :DistilledElevenStateCellFinal)
    include(joinpath(@__DIR__, "DistilledElevenStateCellFinal.jl"))
end
if !isdefined(Main, :DistilledElevenStateCellProduction)
    include(joinpath(@__DIR__, "DistilledElevenStateCellProduction.jl"))
end

const ReleaseTwin = Main.PaperDigitalTwin
const ReleaseCell = Main.DistilledElevenStateCellFinal
const ReleaseRuntime = Main.DistilledElevenStateCellProduction

const RELEASE_DATASET_SCHEMA =
    "hd-swsnn-twinprop-distillation-dataset-v1"
const RELEASE_OFFICIAL_TEACHER_SCHEMA =
    "hd_swsnn_twinprop.neuron_teacher.v1"
const RELEASE_SEGMENTS = 642
const RELEASE_COORDINATE_NAMES = (
    "basal_dendritic_voltage_1",
    "basal_dendritic_voltage_2",
    "apical_dendritic_voltage_1",
    "apical_dendritic_voltage_2",
    "basal_nmda_current_1",
    "basal_nmda_current_2",
    "apical_nmda_current_1",
    "apical_nmda_current_2",
    "apical_calcium_context",
    "soma_voltage",
    "calcium_adaptation",
)

const RELEASE_MINIMUM_SPIKE_AUROC = 0.985
const RELEASE_MAXIMUM_VOLTAGE_RMSE_MV = 5.0
const RELEASE_MINIMUM_VOLTAGE_CORRELATION = 0.90
const RELEASE_MINIMUM_NMDA_CORRELATION = 0.80
const RELEASE_MINIMUM_CALCIUM_AUROC = 0.80
const RELEASE_MAXIMUM_DENDRITIC_RMSE_MV = 8.0
const RELEASE_MAXIMUM_COORDINATE_RMSE = 0.70
const RELEASE_MINIMUM_COORDINATE_CORRELATION = 0.70

function _release_recurrent_mask()
    mask = zeros(Float32, 11, 11)
    # Dendritic voltage coordinates exchange cable/context information but do
    # not form an unrestricted rotationally symmetric hidden basis.
    @inbounds for branch in 1:4
        mask[branch, branch] = 1.0f0
        mask[branch, 4 + branch] = 1.0f0
        mask[branch, 9] = branch >= 3 ? 1.0f0 : 0.25f0
        for neighbour in 1:4
            branch != neighbour &&
                (mask[branch, neighbour] = 0.25f0)
        end
    end
    # Each NMDA coordinate is identified with the matching local voltage and
    # its own slow history.
    @inbounds for branch in 1:4
        mask[4 + branch, branch] = 1.0f0
        mask[4 + branch, 4 + branch] = 1.0f0
    end
    # Apical context is built from apical u/n plus Ca/adaptation.
    for source in (3, 4, 7, 8, 9, 11)
        mask[9, source] = 1.0f0
    end
    # Soma integrates all dendritic coordinates, apical context and its own
    # state; adaptation supplies negative-feedback capacity.
    mask[10, 1:11] .= 1.0f0
    # Adaptation/Ca summary depends on soma, apical context and itself.
    mask[11, 9] = 1.0f0
    mask[11, 10] = 1.0f0
    mask[11, 11] = 1.0f0
    return mask
end

function _release_input_mask()
    mask = zeros(Float32, 11, 16)
    # Input ordering is receptor-major: AMPA[1:4], NMDA[1:4],
    # GABA_A[1:4], auxiliary current[1:4].
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

const RELEASE_RECURRENT_MASK = _release_recurrent_mask()
const RELEASE_INPUT_MASK = _release_input_mask()
const RELEASE_RECURRENT_MASK_SHA256 = let
    stream = IOBuffer()
    Serialization.serialize(stream, RELEASE_RECURRENT_MASK)
    bytes2hex(SHA.sha256(take!(stream)))
end
const RELEASE_INPUT_MASK_SHA256 = let
    stream = IOBuffer()
    Serialization.serialize(stream, RELEASE_INPUT_MASK)
    bytes2hex(SHA.sha256(take!(stream)))
end

Base.@kwdef struct ReleaseDistillationConfig
    dataset::String
    frozen_twin::String
    output::String
    metrics::String
    epochs::Int = 45
    steps_per_epoch::Int = 64
    batch::Int = 32
    window::Int = 256
    learning_rate::Float32 = 0.001f0
    free_rollout_epochs::Int = 15
    minimum_spike_auroc::Float64 = RELEASE_MINIMUM_SPIKE_AUROC
    seed::UInt64 = 0x005b5a19
end

@inline _release_get(object, name::Symbol, default=nothing) =
    if object isa AbstractDict
        get(object, name, get(object, String(name), default))
    elseif hasproperty(object, name)
        getproperty(object, name)
    else
        default
    end

function _release_metadata_get(dataset, name::Symbol, default=nothing)
    direct = _release_get(dataset, name, nothing)
    direct !== nothing && return direct
    metadata = _release_get(dataset, :metadata, nothing)
    metadata === nothing && return default
    direct = _release_get(metadata, name, nothing)
    direct !== nothing && return direct
    hashes = _release_get(metadata, :hashes, nothing)
    hashes === nothing && return default
    return _release_get(hashes, name, default)
end

function _release_required(dataset, names::Tuple)
    for name in names
        value = _release_metadata_get(dataset, name, nothing)
        if value !== nothing && !isempty(String(value))
            return String(value)
        end
    end
    error("prepared dataset lacks required lineage $(join(names, '/'))")
end

function _release_file_sha256(path::AbstractString)
    source = abspath(path)
    isfile(source) || error("file is absent: $source")
    return bytes2hex(SHA.sha256(read(source)))
end

function _release_value_sha256(value)
    stream = IOBuffer()
    Serialization.serialize(stream, value)
    return bytes2hex(SHA.sha256(take!(stream)))
end

function _release_parse_arguments(arguments)
    values = Dict{String,String}(
        "dataset" => "",
        "frozen-twin" => "",
        "output" => joinpath(
            @__DIR__,
            "artifacts",
            "distilled_eleven_state_cell_release.jld2",
        ),
        "metrics" => "",
        "epochs" => "45",
        "steps-per-epoch" => "64",
        "batch" => "32",
        "window" => "256",
        "learning-rate" => "0.001",
        "free-rollout-epochs" => "15",
        "minimum-spike-auroc" => "0.985",
        "seed" => "5987321",
    )
    index = 1
    while index <= length(arguments)
        option = arguments[index]
        startswith(option, "--") ||
            error("unexpected positional argument $option")
        key = option[3:end]
        haskey(values, key) || error("unknown option --$key")
        index == length(arguments) &&
            error("missing value for --$key")
        values[key] = arguments[index + 1]
        index += 2
    end
    isempty(values["dataset"]) && error("--dataset is required")
    isempty(values["frozen-twin"]) &&
        error("--frozen-twin is required")
    output = abspath(values["output"])
    metrics = isempty(values["metrics"]) ?
        output * ".metrics.json" :
        abspath(values["metrics"])
    return ReleaseDistillationConfig(
        dataset=abspath(values["dataset"]),
        frozen_twin=abspath(values["frozen-twin"]),
        output,
        metrics,
        epochs=parse(Int, values["epochs"]),
        steps_per_epoch=parse(Int, values["steps-per-epoch"]),
        batch=parse(Int, values["batch"]),
        window=parse(Int, values["window"]),
        learning_rate=parse(Float32, values["learning-rate"]),
        free_rollout_epochs=parse(
            Int,
            values["free-rollout-epochs"],
        ),
        minimum_spike_auroc=parse(
            Float64,
            values["minimum-spike-auroc"],
        ),
        seed=parse(UInt64, values["seed"]),
    )
end

function _release_validate_config(config::ReleaseDistillationConfig)
    config.epochs >= 1 || error("epochs must be positive")
    config.steps_per_epoch >= 1 ||
        error("steps-per-epoch must be positive")
    config.batch >= 1 || error("batch must be positive")
    config.window >= 2 || error("window must be at least two")
    config.learning_rate > 0 ||
        error("learning-rate must be positive")
    0 <= config.free_rollout_epochs <= config.epochs ||
        error("invalid free-rollout-epochs")
    config.minimum_spike_auroc >= RELEASE_MINIMUM_SPIKE_AUROC ||
        error("release spike AUROC gate cannot be below 0.985")
    config.minimum_spike_auroc <= 1.0 ||
        error("spike AUROC gate cannot exceed one")
    return config
end

function _release_load_dataset(path, frozen)
    isfile(path) || error("prepared dataset is absent: $path")
    loaded = JLD2.load(path)
    haskey(loaded, "dataset") ||
        error("only prepare_distillation_dataset.jl output is accepted")
    raw = loaded["dataset"]
    String(_release_get(raw, :schema, "")) ==
        RELEASE_DATASET_SCHEMA ||
        error("prepared dataset schema mismatch")
    _release_metadata_get(raw, :mixed_supervision, false) === true ||
        error("prepared dataset is not mixed_supervision")
    metadata = _release_get(raw, :metadata, NamedTuple())
    lowercase(String(_release_get(
        metadata,
        :source_kind,
        "",
    ))) == "official_neuron" ||
        error("release requires official NEURON source_kind")
    _release_get(metadata, :official_neuron_source, false) === true ||
        error("release dataset is not marked official_neuron_source")
    String(_release_get(
        metadata,
        :official_neuron_schema,
        _release_get(metadata, :source_teacher_schema, ""),
    )) == RELEASE_OFFICIAL_TEACHER_SCHEMA ||
        error("official NEURON teacher schema mismatch")
    lowercase(String(_release_get(
        metadata,
        :source_completion_state,
        _release_get(metadata, :completion_state, ""),
    ))) == "complete" ||
        error("official teacher is not completion_state=complete")

    twin_gate = _release_get(metadata, :twin_gate, nothing)
    twin_gate !== nothing &&
        _release_get(twin_gate, :passed, false) === true ||
        error("prepared dataset lacks a passed frozen-twin gate")
    twin_metrics = _release_get(
        metadata,
        :twin_held_out_metrics,
        nothing,
    )
    twin_metrics === nothing &&
        error("prepared dataset lacks frozen-twin held-out metrics")
    twin_auroc = Float64(_release_get(
        twin_metrics,
        :spike_auroc,
        NaN,
    ))
    isfinite(twin_auroc) &&
        twin_auroc >= RELEASE_MINIMUM_SPIKE_AUROC ||
        error("frozen digital twin held-out AUROC is below 0.985")

    input = _release_get(raw, :input)
    input isa AbstractArray && ndims(input) == 3 ||
        error("input must be input_dim x time x sample")
    frozen.model.config.segments == RELEASE_SEGMENTS ||
        error("frozen twin is not the official 642-segment model")
    size(input, 1) == frozen.model.config.input_dim ||
        throw(DimensionMismatch("input/frozen-twin dimension mismatch"))
    time_steps = size(input, 2)
    samples = size(input, 3)
    cached_voltage = _release_get(raw, :target_voltage)
    cached_spike = _release_get(raw, :target_spike)
    cached_nmda = _release_get(raw, :target_nmda)
    detailed_calcium = _release_get(raw, :target_calcium_event)
    detailed_dendritic =
        _release_get(raw, :target_dendritic_voltage)
    size(cached_voltage) == (time_steps, samples) ||
        throw(DimensionMismatch("cached voltage target shape differs"))
    size(cached_spike) == (time_steps, samples) ||
        throw(DimensionMismatch("cached spike target shape differs"))
    size(cached_nmda) == (4, time_steps, samples) ||
        throw(DimensionMismatch("cached NMDA target shape differs"))
    size(detailed_calcium) == (time_steps, samples) ||
        throw(DimensionMismatch("detailed Ca-event target shape differs"))
    size(detailed_dendritic) == (4, time_steps, samples) ||
        throw(DimensionMismatch(
            "detailed dendritic-voltage target shape differs",
        ))

    twin_parameter_hash = _release_required(
        raw,
        (:frozen_twin_parameter_hash,),
    )
    twin_artifact_hash = _release_required(
        raw,
        (:frozen_twin_artifact_hash, :digital_twin_hash),
    )
    twin_parameter_hash == frozen.parameter_sha256 ||
        error("frozen-twin parameter hash mismatch")
    twin_artifact_hash == frozen.artifact_sha256 ||
        error("frozen-twin artifact hash mismatch")
    raw_twin_file_hash = _release_required(
        raw,
        (:frozen_twin_file_sha256, :raw_twin_file_sha256),
    )

    segment_region = _release_metadata_get(
        raw,
        :segment_region,
        _release_metadata_get(
            raw,
            :official_segment_region,
            nothing,
        ),
    )
    segment_region !== nothing &&
        length(segment_region) == RELEASE_SEGMENTS ||
        error("prepared dataset lacks the official 642 segment-region map")
    mapping_source_hash = _release_required(
        raw,
        (
            :segment_catalog_sha256,
            :official_segment_catalog_sha256,
        ),
    )
    train_indices = Int.(_release_get(raw, :train_indices))
    validation_indices = Int.(_release_get(raw, :validation_indices))
    test_indices = Int.(_release_get(raw, :test_indices))
    all(!isempty, (train_indices, validation_indices, test_indices)) ||
        error("prepared dataset contains an empty split")
    all(index -> 1 <= index <= samples,
        vcat(train_indices, validation_indices, test_indices)) ||
        error("prepared dataset split index is out of bounds")

    provenance = (;
        detailed_teacher_hash=_release_required(
            raw,
            (:detailed_teacher_hash, :teacher_contract_sha256),
        ),
        detailed_kernel_hash=_release_required(
            raw,
            (:detailed_kernel_hash, :cell_mechanism_sha256),
        ),
        morphology_hash=_release_required(
            raw,
            (:morphology_hash, :morphology_sha256),
        ),
        official_modeldb_source_hash=_release_required(
            raw,
            (:official_modeldb_source_hash,),
        ),
        official_teacher_file_hash=_release_required(
            raw,
            (:official_teacher_file_hash, :source_manifest_sha256),
        ),
        official_source_dataset_hash=_release_required(
            raw,
            (:original_dataset_sha256, :dataset_sha256),
        ),
        segment_catalog_sha256=mapping_source_hash,
        frozen_twin_parameter_hash=twin_parameter_hash,
        frozen_twin_artifact_hash=twin_artifact_hash,
        raw_twin_file_sha256=raw_twin_file_hash,
        twin_held_out_spike_auroc=twin_auroc,
    )
    return (;
        input=Float32.(input),
        cached_twin_voltage=Float32.(cached_voltage),
        cached_twin_spike=Float32.(cached_spike),
        cached_twin_nmda=Float32.(cached_nmda),
        detailed_calcium_event=Float32.(detailed_calcium),
        detailed_dendritic_voltage=Float32.(detailed_dendritic),
        train_indices,
        validation_indices,
        test_indices,
        segment_region,
        provenance,
        prepared_dataset_file_sha256=_release_file_sha256(path),
    )
end

function _release_twin_inference(frozen, input; batch_size=8)
    time_steps = size(input, 2)
    samples = size(input, 3)
    voltage = Matrix{Float32}(undef, time_steps, samples)
    spike = Matrix{Float32}(undef, time_steps, samples)
    nmda = Array{Float32}(undef, 4, time_steps, samples)
    for first in 1:batch_size:samples
        last = min(first + batch_size - 1, samples)
        prediction = ReleaseTwin.twin_forward(
            frozen,
            @view(input[:, :, first:last]),
        )
        voltage[:, first:last] .= prediction.voltage
        spike[:, first:last] .= prediction.spike_probability
        nmda[:, :, first:last] .= prediction.nmda
    end
    return (; voltage, spike, nmda)
end

function _release_cache_check(dataset, prediction)
    voltage_error = maximum(
        abs,
        Float64.(dataset.cached_twin_voltage) .-
        Float64.(prediction.voltage),
    )
    spike_error = maximum(
        abs,
        Float64.(dataset.cached_twin_spike) .-
        Float64.(prediction.spike),
    )
    nmda_error = maximum(
        abs,
        Float64.(dataset.cached_twin_nmda) .-
        Float64.(prediction.nmda),
    )
    tolerance = 2.0e-5
    maximum((voltage_error, spike_error, nmda_error)) <= tolerance ||
        error(
            "prepared twin cache differs from live frozen inference: " *
            "voltage=$voltage_error spike=$spike_error nmda=$nmda_error",
        )
    return (; voltage_error, spike_error, nmda_error, tolerance)
end

function _release_targets(dataset, twin)
    time_steps = size(dataset.input, 2)
    samples = size(dataset.input, 3)
    target = Array{Float32}(undef, 11, time_steps, samples)
    target[1, :, :] = twin.voltage
    target[2, :, :] = twin.spike
    target[3:6, :, :] = twin.nmda
    target[7, :, :] = dataset.detailed_calcium_event
    target[8:11, :, :] = dataset.detailed_dendritic_voltage
    return target
end

function _release_target_statistics(target, train_indices)
    target_mean = zeros(Float32, 11)
    target_scale = ones(Float32, 11)
    for index in (1, 3, 4, 5, 6, 8, 9, 10, 11)
        values = vec(@view target[index, :, train_indices])
        target_mean[index] = Float32(mean(values))
        target_scale[index] =
            max(Float32(std(values; corrected=false)), 1.0f-4)
    end
    return target_mean, target_scale
end

@inline _release_sigmoid(value) =
    ifelse(
        value >= zero(value),
        inv(one(value) + exp(-value)),
        exp(value) / (one(value) + exp(value)),
    )

function _release_softmax_columns(logits)
    shifted = logits .- maximum(logits; dims=1)
    exponent = exp.(shifted)
    return exponent ./ sum(exponent; dims=1)
end

function _release_project_input(raw, location_logits, segments)
    size(raw, 1) == 6segments ||
        throw(DimensionMismatch("frozen-twin input layout differs"))
    location = _release_softmax_columns(location_logits)
    event = reshape(
        raw[1:(3segments), :],
        segments,
        3,
        size(raw, 2),
    )
    ampa = location * event[:, 1, :]
    nmda = location * event[:, 2, :]
    gaba = location * event[:, 3, :]
    auxiliary = zeros(eltype(raw), 4, size(raw, 2))
    return vcat(ampa, nmda, gaba, auxiliary)
end

function _release_state_activation(value)
    return vcat(
        tanh.(value[1:10, :]),
        _release_sigmoid.(value[11:11, :]),
    )
end

function _release_transition(parameters, state, input)
    recurrent =
        parameters.recurrent_weight .* RELEASE_RECURRENT_MASK
    input_weight = parameters.input_weight .* RELEASE_INPUT_MASK
    proposal = _release_state_activation(
        recurrent * state .+
        input_weight * input .+
        parameters.transition_bias,
    )
    decay = _release_sigmoid.(parameters.transition_decay_logit)
    return decay .* state .+ (1.0f0 .- decay) .* proposal
end

function _release_normalize_target(target, target_mean, target_scale)
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

function _release_semantic_target(normalized)
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

function _release_structured_readout(parameters, state)
    gain = parameters.readout_gain
    bias = parameters.readout_bias
    soma = gain[1] .* state[10:10, :] .+ bias[1]
    spike = gain[2] .* state[10:10, :] .+
        parameters.spike_adaptation_gain[1] .* state[11:11, :] .+
        bias[2]
    nmda = gain[3:6] .* state[5:8, :] .+ bias[3:6]
    calcium = gain[7] .* state[11:11, :] .+ bias[7]
    dendritic = gain[8:11] .* state[1:4, :] .+ bias[8:11]
    return vcat(soma, spike, nmda, calcium, dendritic)
end

@inline _release_bce(logit, target) =
    max(logit, zero(logit)) - logit * target +
    log1p(exp(-abs(logit)))

function _release_sequence_loss(
    parameters,
    raw_input,
    target,
    target_mean,
    target_scale,
    segments,
    free_fraction::Float32,
)
    time_steps = size(raw_input, 2)
    batch = size(raw_input, 3)
    state = repeat(parameters.initial_state, 1, batch)
    continuous_loss = 0.0f0
    spike_loss = 0.0f0
    calcium_loss = 0.0f0
    semantic_loss = 0.0f0
    for time in 1:time_steps
        input = _release_project_input(
            raw_input[:, time, :],
            parameters.location_logits,
            segments,
        )
        predicted_state =
            _release_transition(parameters, state, input)
        output =
            _release_structured_readout(parameters, predicted_state)
        normalized = _release_normalize_target(
            target[:, time, :],
            target_mean,
            target_scale,
        )
        semantic = _release_semantic_target(normalized)
        continuous_loss +=
            sum(abs2, output[1:1, :] .- normalized[1:1, :])
        continuous_loss += 0.75f0 *
            sum(abs2, output[3:6, :] .- normalized[3:6, :])
        continuous_loss += 0.50f0 *
            sum(abs2, output[8:11, :] .- normalized[8:11, :])
        spike_loss += sum(_release_bce.(
            output[2:2, :],
            normalized[2:2, :],
        ))
        calcium_loss += sum(_release_bce.(
            output[7:7, :],
            normalized[7:7, :],
        ))
        semantic_loss += sum(abs2, predicted_state .- semantic)
        state = free_fraction .* predicted_state .+
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

function _release_initial_parameters(rng, segments)
    location_logits =
        -2.0f0 .+ 0.02f0 .* randn(rng, Float32, 4, segments)
    @inbounds for segment in 1:segments
        location_logits[mod1(segment, 4), segment] += 4.0f0
    end
    recurrent_weight =
        0.08f0 .* randn(rng, Float32, 11, 11) .+
        0.55f0 .* Matrix{Float32}(I, 11, 11)
    recurrent_weight .*= RELEASE_RECURRENT_MASK
    input_weight =
        0.10f0 .* randn(rng, Float32, 11, 16) .*
        RELEASE_INPUT_MASK
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

function _release_fraction(epoch, config)
    teacher_epochs =
        max(config.epochs - config.free_rollout_epochs, 1)
    epoch > teacher_epochs && return 1.0f0
    return Float32(epoch - 1) / Float32(teacher_epochs)
end

function _release_sample_window(
    rng,
    dataset,
    target,
    config,
)
    window = min(config.window, size(dataset.input, 2))
    first = rand(rng, 1:(size(dataset.input, 2) - window + 1))
    samples = rand(rng, dataset.train_indices, config.batch)
    time_range = first:(first + window - 1)
    return (
        dataset.input[:, time_range, samples],
        target[:, time_range, samples],
    )
end

function _release_all_finite(parameters)
    for value in values(parameters)
        value isa AbstractArray && all(isfinite, value) || return false
    end
    return true
end

function _release_train(
    rng,
    dataset,
    target,
    target_mean,
    target_scale,
    segments,
    config,
)
    parameters = _release_initial_parameters(rng, segments)
    optimizer_state = Optimisers.setup(
        Optimisers.Adam(config.learning_rate),
        parameters,
    )
    history = NamedTuple[]
    for epoch in 1:config.epochs
        free_fraction = _release_fraction(epoch, config)
        total_loss = 0.0
        started = time()
        for _ in 1:config.steps_per_epoch
            input_window, target_window =
                _release_sample_window(
                    rng,
                    dataset,
                    target,
                    config,
                )
            loss, gradient = Zygote.withgradient(parameters) do candidate
                _release_sequence_loss(
                    candidate,
                    input_window,
                    target_window,
                    target_mean,
                    target_scale,
                    segments,
                    free_fraction,
                )
            end
            isfinite(loss) || error("distillation loss is non-finite")
            optimizer_state, parameters = Optimisers.update(
                optimizer_state,
                parameters,
                only(gradient),
            )
            _release_all_finite(parameters) ||
                error("distillation parameter became non-finite")
            total_loss += Float64(loss)
        end
        record = (;
            epoch,
            loss=total_loss / config.steps_per_epoch,
            free_rollout_fraction=free_fraction,
            elapsed_seconds=time() - started,
        )
        push!(history, record)
        @printf(
            "release distill epoch %d/%d loss=%.6f free=%.3f time=%.2fs\n",
            epoch,
            config.epochs,
            record.loss,
            free_fraction,
            record.elapsed_seconds,
        )
        flush(stdout)
    end
    return parameters, history
end

function _release_rollout(
    parameters,
    raw_input,
    target,
    target_mean,
    target_scale,
    segments,
)
    time_steps = size(raw_input, 2)
    batch = size(raw_input, 3)
    state = repeat(parameters.initial_state, 1, batch)
    output = Array{Float32}(undef, 11, time_steps, batch)
    states = Array{Float32}(undef, 11, time_steps, batch)
    semantic = Array{Float32}(undef, 11, time_steps, batch)
    for time in 1:time_steps
        input = _release_project_input(
            raw_input[:, time, :],
            parameters.location_logits,
            segments,
        )
        state = _release_transition(parameters, state, input)
        output[:, time, :] =
            _release_structured_readout(parameters, state)
        normalized = _release_normalize_target(
            target[:, time, :],
            target_mean,
            target_scale,
        )
        states[:, time, :] = state
        semantic[:, time, :] =
            _release_semantic_target(normalized)
    end
    return (; output, states, semantic)
end

function _release_physical_output(raw, target_mean, target_scale)
    output = raw .* reshape(target_scale, :, 1, 1) .+
        reshape(target_mean, :, 1, 1)
    output[2, :, :] .= _release_sigmoid.(raw[2, :, :])
    output[7, :, :] .= _release_sigmoid.(raw[7, :, :])
    return output
end

function _release_auroc(score, target)
    score = vec(Float64.(score))
    positive = vec(target .>= 0.5f0)
    positive_count = count(identity, positive)
    negative_count = length(positive) - positive_count
    (positive_count == 0 || negative_count == 0) && return NaN
    order = sortperm(score)
    rank_sum = 0.0
    index = 1
    while index <= length(order)
        last = index
        while last < length(order) &&
              score[order[last + 1]] == score[order[index]]
            last += 1
        end
        rank = 0.5 * (index + last)
        for position in index:last
            positive[order[position]] && (rank_sum += rank)
        end
        index = last + 1
    end
    return (
        rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)
end

_release_rmse(prediction, target) =
    sqrt(mean(abs2, Float64.(prediction) .- Float64.(target)))

function _release_correlation(prediction, target)
    x = vec(Float64.(prediction))
    y = vec(Float64.(target))
    x .-= mean(x)
    y .-= mean(y)
    denominator = sqrt(sum(abs2, x) * sum(abs2, y))
    denominator <= eps(Float64) && return NaN
    return sum(x .* y) / denominator
end

function _release_metrics(
    parameters,
    dataset,
    target,
    indices,
    target_mean,
    target_scale,
    segments,
)
    rollout = _release_rollout(
        parameters,
        dataset.input[:, :, indices],
        target[:, :, indices],
        target_mean,
        target_scale,
        segments,
    )
    prediction = _release_physical_output(
        rollout.output,
        target_mean,
        target_scale,
    )
    truth = target[:, :, indices]
    coordinate_rmse = [
        _release_rmse(
            rollout.states[coordinate, :, :],
            rollout.semantic[coordinate, :, :],
        )
        for coordinate in 1:11
    ]
    coordinate_correlation = [
        _release_correlation(
            rollout.states[coordinate, :, :],
            rollout.semantic[coordinate, :, :],
        )
        for coordinate in 1:11
    ]
    coordinate_passed = [
        isfinite(coordinate_rmse[coordinate]) &&
        isfinite(coordinate_correlation[coordinate]) &&
        coordinate_rmse[coordinate] <=
            RELEASE_MAXIMUM_COORDINATE_RMSE &&
        coordinate_correlation[coordinate] >=
            RELEASE_MINIMUM_COORDINATE_CORRELATION
        for coordinate in 1:11
    ]
    return (;
        samples=length(indices),
        free_rollout_horizon=size(truth, 2),
        soma_voltage_rmse_mv=_release_rmse(
            prediction[1, :, :],
            truth[1, :, :],
        ),
        soma_voltage_correlation=_release_correlation(
            prediction[1, :, :],
            truth[1, :, :],
        ),
        spike_auroc=_release_auroc(
            prediction[2, :, :],
            truth[2, :, :],
        ),
        nmda_rmse_by_region=[
            _release_rmse(
                prediction[2 + region, :, :],
                truth[2 + region, :, :],
            )
            for region in 1:4
        ],
        nmda_correlation_by_region=[
            _release_correlation(
                prediction[2 + region, :, :],
                truth[2 + region, :, :],
            )
            for region in 1:4
        ],
        calcium_event_auroc=_release_auroc(
            prediction[7, :, :],
            truth[7, :, :],
        ),
        dendritic_voltage_rmse_mv=[
            _release_rmse(
                prediction[7 + region, :, :],
                truth[7 + region, :, :],
            )
            for region in 1:4
        ],
        semantic_coordinate_names=RELEASE_COORDINATE_NAMES,
        semantic_coordinate_rmse=coordinate_rmse,
        semantic_coordinate_correlation=coordinate_correlation,
        semantic_coordinate_passed=coordinate_passed,
    )
end

function _release_gate(test_metrics, minimum_spike_auroc)
    spike_passed =
        isfinite(test_metrics.spike_auroc) &&
        test_metrics.spike_auroc >= minimum_spike_auroc
    voltage_passed =
        isfinite(test_metrics.soma_voltage_rmse_mv) &&
        isfinite(test_metrics.soma_voltage_correlation) &&
        test_metrics.soma_voltage_rmse_mv <=
            RELEASE_MAXIMUM_VOLTAGE_RMSE_MV &&
        test_metrics.soma_voltage_correlation >=
            RELEASE_MINIMUM_VOLTAGE_CORRELATION
    nmda_passed =
        all(isfinite, test_metrics.nmda_rmse_by_region) &&
        all(isfinite, test_metrics.nmda_correlation_by_region) &&
        mean(test_metrics.nmda_correlation_by_region) >=
            RELEASE_MINIMUM_NMDA_CORRELATION
    calcium_passed =
        isfinite(test_metrics.calcium_event_auroc) &&
        test_metrics.calcium_event_auroc >=
            RELEASE_MINIMUM_CALCIUM_AUROC
    dendritic_voltage_passed =
        all(isfinite, test_metrics.dendritic_voltage_rmse_mv) &&
        mean(test_metrics.dendritic_voltage_rmse_mv) <=
            RELEASE_MAXIMUM_DENDRITIC_RMSE_MV
    semantic_coordinate_passed =
        all(test_metrics.semantic_coordinate_passed)
    multi_target_passed =
        voltage_passed &&
        nmda_passed &&
        calcium_passed &&
        dendritic_voltage_passed &&
        semantic_coordinate_passed
    return (;
        passed=spike_passed && multi_target_passed,
        minimum_spike_auroc,
        held_out_spike_auroc=test_metrics.spike_auroc,
        spike_passed,
        multi_target_passed,
        voltage_passed,
        nmda_passed,
        calcium_passed,
        dendritic_voltage_passed,
        semantic_state_passed=semantic_coordinate_passed,
        semantic_coordinate_passed,
    )
end

function _release_region_projection(dataset, location)
    projection = zeros(Float32, 4, 4)
    count = zeros(Int, 4)
    @inbounds for segment in 1:RELEASE_SEGMENTS
        label = lowercase(String(dataset.segment_region[segment]))
        region = occursin("soma", label) ? 1 :
            occursin("basal", label) ? 2 :
            occursin("tuft", label) ? 4 : 3
        projection[:, region] .+= location[:, segment]
        count[region] += 1
    end
    @inbounds for region in 1:4
        count[region] > 0 ||
            error("official segment map lacks region $region")
        projection[:, region] ./= count[region]
    end
    return projection
end

function _release_readout_matrix(parameters)
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

function _release_frozen_parameters(
    trained,
    dataset,
    frozen_twin,
    target_mean,
    target_scale,
    config_hash,
)
    location = Float32.(
        _release_softmax_columns(trained.location_logits),
    )
    parameters = ReleaseCell.DistilledParameters(
        dt_ms=frozen_twin.model.config.dt_ms,
        transition_decay=Float32.(
            _release_sigmoid.(trained.transition_decay_logit),
        ),
        recurrent_weight=
            trained.recurrent_weight .* RELEASE_RECURRENT_MASK,
        input_weight=trained.input_weight .* RELEASE_INPUT_MASK,
        transition_bias=vec(trained.transition_bias),
        readout_weight=_release_readout_matrix(trained),
        readout_bias=vec(trained.readout_bias),
        target_mean,
        target_scale,
        initial_state=vec(trained.initial_state),
        compartment_projection=location,
        region_projection=
            _release_region_projection(dataset, location),
        spike_threshold=0.5f0,
        teacher_schema=
            "PaperDigitalTwin+official-NEURON-mixed-release-v1",
        detailed_kernel_hash=
            dataset.provenance.detailed_kernel_hash,
        morphology_hash=dataset.provenance.morphology_hash,
        frozen_twin_parameter_hash=
            frozen_twin.parameter_sha256,
        frozen_twin_artifact_hash=
            frozen_twin.artifact_sha256,
        distillation_dataset_hash=
            dataset.prepared_dataset_file_sha256,
        distillation_config_hash=config_hash,
    )
    arrays = (
        parameters.transition_decay,
        parameters.recurrent_weight,
        parameters.input_weight,
        parameters.transition_bias,
        parameters.readout_weight,
        parameters.readout_bias,
        parameters.target_mean,
        parameters.target_scale,
        parameters.initial_state,
        parameters.compartment_projection,
        parameters.region_projection,
    )
    all(array -> all(isfinite, array), arrays) ||
        error("frozen distilled core contains non-finite values")
    return parameters
end

function _release_atomic_jldsave(path; payload)
    destination = abspath(path)
    mkpath(dirname(destination))
    temporary = destination * ".tmp." * string(getpid())
    JLD2.jldsave(temporary; payload)
    mv(temporary, destination; force=true)
    return destination
end

function _release_atomic_json(path, value)
    destination = abspath(path)
    mkpath(dirname(destination))
    temporary = destination * ".tmp." * string(getpid())
    open(temporary, "w") do stream
        JSON3.pretty(stream, value)
        write(stream, '\n')
    end
    mv(temporary, destination; force=true)
    return destination
end

function run_release_distillation(config::ReleaseDistillationConfig)
    _release_validate_config(config)
    frozen_twin = ReleaseTwin.load_frozen_twin(config.frozen_twin)
    twin_before = ReleaseTwin.assert_frozen_unchanged(frozen_twin)
    raw_twin_file_sha256 =
        _release_file_sha256(config.frozen_twin)
    dataset = _release_load_dataset(
        config.dataset,
        frozen_twin,
    )
    raw_twin_file_sha256 ==
        dataset.provenance.raw_twin_file_sha256 ||
        error("raw frozen-twin file SHA-256 mismatch")
    live_twin =
        _release_twin_inference(frozen_twin, dataset.input)
    cache_check = _release_cache_check(dataset, live_twin)
    target = _release_targets(dataset, live_twin)
    target_mean, target_scale =
        _release_target_statistics(target, dataset.train_indices)
    config_record = (;
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        batch=config.batch,
        window=min(config.window, size(dataset.input, 2)),
        learning_rate=config.learning_rate,
        free_rollout_epochs=config.free_rollout_epochs,
        minimum_spike_auroc=config.minimum_spike_auroc,
        seed=config.seed,
        official_segment_count=RELEASE_SEGMENTS,
        semantic_coordinate_names=RELEASE_COORDINATE_NAMES,
        recurrent_mask_sha256=RELEASE_RECURRENT_MASK_SHA256,
        input_mask_sha256=RELEASE_INPUT_MASK_SHA256,
        structured_transition=true,
        structured_readout=true,
        coordinate_wise_semantic_supervision=true,
        soma_spike_is_sole_external_event=true,
        mixed_supervision=(
            frozen_twin=(:soma_voltage, :soma_spike, :nmda_current),
            official_neuron=(:calcium_event, :dendritic_voltage),
        ),
    )
    config_hash = _release_value_sha256(config_record)
    trained, history = _release_train(
        Xoshiro(config.seed),
        dataset,
        target,
        target_mean,
        target_scale,
        frozen_twin.model.config.segments,
        config,
    )
    metrics = (;
        train=_release_metrics(
            trained,
            dataset,
            target,
            dataset.train_indices,
            target_mean,
            target_scale,
            frozen_twin.model.config.segments,
        ),
        validation=_release_metrics(
            trained,
            dataset,
            target,
            dataset.validation_indices,
            target_mean,
            target_scale,
            frozen_twin.model.config.segments,
        ),
        test=_release_metrics(
            trained,
            dataset,
            target,
            dataset.test_indices,
            target_mean,
            target_scale,
            frozen_twin.model.config.segments,
        ),
    )
    gate = _release_gate(
        metrics.test,
        config.minimum_spike_auroc,
    )
    parameters = _release_frozen_parameters(
        trained,
        dataset,
        frozen_twin,
        target_mean,
        target_scale,
        config_hash,
    )
    parameter_hash = ReleaseCell.parameter_sha256(parameters)
    location_mapping_sha256 = _release_value_sha256((
        dataset.provenance.segment_catalog_sha256,
        dataset.segment_region,
        parameters.compartment_projection,
    ))
    twin_after =
        ReleaseTwin.assert_frozen_unchanged(frozen_twin)
    twin_before == twin_after ||
        error("frozen digital twin changed during distillation")
    semantic_coordinate_gate = (;
        passed=all(metrics.test.semantic_coordinate_passed),
        coordinate_names=RELEASE_COORDINATE_NAMES,
        rmse=metrics.test.semantic_coordinate_rmse,
        correlation=metrics.test.semantic_coordinate_correlation,
        per_coordinate_passed=
            metrics.test.semantic_coordinate_passed,
        maximum_rmse=RELEASE_MAXIMUM_COORDINATE_RMSE,
        minimum_correlation=
            RELEASE_MINIMUM_COORDINATE_CORRELATION,
    )
    structured_transition_contract = (;
        recurrent_mask_sha256=RELEASE_RECURRENT_MASK_SHA256,
        input_mask_sha256=RELEASE_INPUT_MASK_SHA256,
        structured_readout=true,
        dense_rotational_hidden_basis=false,
        coordinate_wise_semantic_supervision=true,
    )
    payload = (;
        schema=ReleaseCell.DISTILLED_ARTIFACT_SCHEMA,
        parameters,
        parameter_sha256=parameter_hash,
        frozen_internal=true,
        ablation_mode=:full,
        official_segment_count=RELEASE_SEGMENTS,
        location_mapping_sha256,
        semantic_coordinate_gate,
        structured_transition_contract,
        teacher_hash=dataset.provenance.detailed_teacher_hash,
        cell_mechanism_sha256=
            dataset.provenance.detailed_kernel_hash,
        detailed_kernel_hash=
            dataset.provenance.detailed_kernel_hash,
        morphology_hash=dataset.provenance.morphology_hash,
        digital_twin_sha256=frozen_twin.artifact_sha256,
        digital_twin_hash=frozen_twin.artifact_sha256,
        frozen_twin_parameter_hash=
            frozen_twin.parameter_sha256,
        frozen_twin_artifact_hash=
            frozen_twin.artifact_sha256,
        raw_twin_file_sha256,
        official_modeldb_source_hash=
            dataset.provenance.official_modeldb_source_hash,
        official_teacher_file_hash=
            dataset.provenance.official_teacher_file_hash,
        official_source_dataset_hash=
            dataset.provenance.official_source_dataset_hash,
        prepared_dataset_file_sha256=
            dataset.prepared_dataset_file_sha256,
        distillation_dataset_hash=
            dataset.prepared_dataset_file_sha256,
        distillation_config_hash=config_hash,
        source_segment_catalog_sha256=
            dataset.provenance.segment_catalog_sha256,
        mixed_supervision=(;
            twin_targets=(:soma_voltage, :soma_spike, :nmda_current),
            official_neuron_targets=
                (:calcium_event, :dendritic_voltage),
            live_frozen_twin_inference=true,
            cache_check,
        ),
        frozen_twin_integrity_before=twin_before,
        frozen_twin_integrity_after=twin_after,
        metrics,
        gate,
        config=config_record,
        history,
        created_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
    )
    timestamp = Dates.format(now(UTC), dateformat"yyyymmddTHHMMSS")
    destination = gate.passed ?
        config.output :
        config.output * ".failed." * timestamp * ".jld2"
    _release_atomic_jldsave(destination; payload)
    if gate.passed
        runtime =
            ReleaseRuntime.load_production_distilled_artifact(destination)
        ReleaseRuntime.checkpoint_frozen_digest(runtime)
    end
    report = (;
        accepted=gate.passed,
        artifact_path=destination,
        artifact_sha256=
            ReleaseCell.artifact_sha256(destination),
        parameter_sha256=parameter_hash,
        location_mapping_sha256,
        semantic_coordinate_gate,
        structured_transition_contract,
        metrics,
        gate,
        cache_check,
        history,
    )
    _release_atomic_json(config.metrics, report)
    @info "release distillation finished" report.accepted report.artifact_path report.gate report.semantic_coordinate_gate
    gate.passed || error(
        "release gates failed; candidate saved separately at " *
        destination * "; accepted artifact path was not modified",
    )
    return report
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_release_distillation(_release_parse_arguments(ARGS))
end
