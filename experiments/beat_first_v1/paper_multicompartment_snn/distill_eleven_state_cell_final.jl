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

const Twin = Main.PaperDigitalTwin
const Cell = Main.DistilledElevenStateCellFinal
const PREPARED_DATASET_SCHEMA =
    "hd-swsnn-twinprop-distillation-dataset-v1"

@inline _value(object, name::Symbol, default=nothing) =
    if object isa AbstractDict
        get(object, name, get(object, String(name), default))
    elseif hasproperty(object, name)
        getproperty(object, name)
    else
        default
    end

function _nested_value(payload, name::Symbol, default=nothing)
    direct = _value(payload, name, nothing)
    direct !== nothing && return direct
    metadata = _value(payload, :metadata, nothing)
    metadata === nothing && return default
    return _value(metadata, name, default)
end

function _sha256_file(path::AbstractString)
    source = abspath(path)
    isfile(source) || error("file is absent: $source")
    return bytes2hex(SHA.sha256(read(source)))
end

function _sha256_value(value)
    stream = IOBuffer()
    Serialization.serialize(stream, value)
    return bytes2hex(SHA.sha256(take!(stream)))
end

function _required_string(payload, names::Tuple)
    for name in names
        value = _nested_value(payload, name, nothing)
        if value !== nothing && !isempty(String(value))
            return String(value)
        end
    end
    error("prepared dataset lacks required provenance: $(join(names, '/'))")
end

function _parse_arguments(arguments)
    options = Dict{String,String}(
        "dataset" => "",
        "frozen-twin" => "",
        "output" => joinpath(
            @__DIR__,
            "artifacts",
            "distilled_eleven_state_cell_final.jld2",
        ),
        "metrics" => "",
        "epochs" => "45",
        "steps-per-epoch" => "64",
        "batch" => "32",
        "window" => "256",
        "learning-rate" => "0.001",
        "free-rollout-epochs" => "15",
        "minimum-spike-auroc" => "0.95",
        "seed" => "5987321",
        "allow-below-gate" => "false",
    )
    index = 1
    while index <= length(arguments)
        token = arguments[index]
        startswith(token, "--") ||
            error("unexpected positional argument $token")
        key = token[3:end]
        haskey(options, key) || error("unknown option --$key")
        index == length(arguments) &&
            error("missing value for --$key")
        options[key] = arguments[index + 1]
        index += 2
    end
    isempty(options["dataset"]) && error("--dataset is required")
    isempty(options["frozen-twin"]) &&
        error("--frozen-twin is required")
    output = abspath(options["output"])
    return (;
        dataset=abspath(options["dataset"]),
        frozen_twin=abspath(options["frozen-twin"]),
        output,
        metrics=isempty(options["metrics"]) ?
            output * ".metrics.json" :
            abspath(options["metrics"]),
        epochs=parse(Int, options["epochs"]),
        steps_per_epoch=parse(Int, options["steps-per-epoch"]),
        batch=parse(Int, options["batch"]),
        window=parse(Int, options["window"]),
        learning_rate=parse(Float32, options["learning-rate"]),
        free_rollout_epochs=parse(Int, options["free-rollout-epochs"]),
        minimum_spike_auroc=parse(
            Float64,
            options["minimum-spike-auroc"],
        ),
        seed=parse(UInt64, options["seed"]),
        allow_below_gate=
            lowercase(options["allow-below-gate"]) == "true",
    )
end

function _validate_config(config)
    config.epochs >= 1 || error("epochs must be positive")
    config.steps_per_epoch >= 1 ||
        error("steps-per-epoch must be positive")
    config.batch >= 1 || error("batch must be positive")
    config.window >= 2 || error("window must be at least two")
    config.learning_rate > 0 ||
        error("learning-rate must be positive")
    0 <= config.free_rollout_epochs <= config.epochs ||
        error("invalid free-rollout-epochs")
    0.5 <= config.minimum_spike_auroc <= 1.0 ||
        error("minimum-spike-auroc must be in [0.5,1]")
    return config
end

function _load_prepared_dataset(path, frozen)
    isfile(path) || error("prepared distillation dataset is absent: $path")
    data = JLD2.load(path)
    haskey(data, "dataset") || error(
        "only prepare_distillation_dataset.jl output is accepted; " *
        "top-level `dataset` is absent",
    )
    dataset = data["dataset"]
    String(_value(dataset, :schema, "")) == PREPARED_DATASET_SCHEMA ||
        error(
            "unsupported/provisional distillation dataset schema; " *
            "expected $PREPARED_DATASET_SCHEMA",
        )
    _nested_value(dataset, :mixed_supervision, false) === true ||
        error("prepared dataset is not marked mixed_supervision")
    _nested_value(dataset, :digital_twin_gate_passed, false) === true ||
        error("prepared dataset was built from an ungated digital twin")

    input = _value(dataset, :input)
    input isa AbstractArray && ndims(input) == 3 ||
        error("prepared input must be input_dim x time x sample")
    size(input, 1) == frozen.model.config.input_dim ||
        throw(DimensionMismatch(
            "prepared input dimension differs from frozen twin",
        ))
    time_steps = size(input, 2)
    samples = size(input, 3)
    target_voltage = _value(dataset, :target_voltage)
    target_spike = _value(dataset, :target_spike)
    target_nmda = _value(dataset, :target_nmda)
    target_ca = _value(dataset, :target_calcium_event)
    target_dendritic = _value(dataset, :target_dendritic_voltage)
    size(target_voltage) == (time_steps, samples) ||
        throw(DimensionMismatch("cached twin voltage shape differs"))
    size(target_spike) == (time_steps, samples) ||
        throw(DimensionMismatch("cached twin spike shape differs"))
    size(target_nmda) == (4, time_steps, samples) ||
        throw(DimensionMismatch("cached twin NMDA shape differs"))
    size(target_ca) == (time_steps, samples) ||
        throw(DimensionMismatch("detailed Ca-event shape differs"))
    size(target_dendritic) == (4, time_steps, samples) ||
        throw(DimensionMismatch(
            "detailed dendritic-voltage shape differs",
        ))

    twin_parameter_hash = _required_string(
        dataset,
        (:frozen_twin_parameter_hash, :digital_twin_parameter_sha256),
    )
    twin_artifact_hash = _required_string(
        dataset,
        (:frozen_twin_artifact_hash, :digital_twin_hash),
    )
    twin_parameter_hash == frozen.parameter_sha256 ||
        error("prepared dataset frozen-twin parameter hash mismatch")
    twin_artifact_hash == frozen.artifact_sha256 ||
        error("prepared dataset frozen-twin artifact hash mismatch")

    official_schema = _required_string(
        dataset,
        (:official_neuron_schema, :source_teacher_schema),
    )
    official_schema == "hd_swsnn_twinprop.neuron_teacher.v1" ||
        error("dataset source is not the official NEURON teacher schema")
    provenance = (;
        detailed_teacher_hash=_required_string(
            dataset,
            (:detailed_teacher_hash, :teacher_contract_sha256),
        ),
        detailed_kernel_hash=_required_string(
            dataset,
            (:detailed_kernel_hash, :mechanism_library_sha256),
        ),
        morphology_hash=_required_string(
            dataset,
            (:morphology_hash, :morphology_sha256),
        ),
        official_modeldb_source_hash=_required_string(
            dataset,
            (:official_modeldb_source_hash, :modeldb_tree_sha256),
        ),
        official_neuron_schema=official_schema,
        frozen_twin_parameter_hash=twin_parameter_hash,
        frozen_twin_artifact_hash=twin_artifact_hash,
        source_dataset_hash=_required_string(
            dataset,
            (:source_dataset_hash, :source_manifest_sha256),
        ),
    )

    train_indices = Int.(_value(dataset, :train_indices))
    validation_indices = Int.(_value(dataset, :validation_indices))
    test_indices = Int.(_value(dataset, :test_indices))
    isempty(train_indices) && error("prepared training split is empty")
    isempty(validation_indices) &&
        error("prepared validation split is empty")
    isempty(test_indices) && error("prepared test split is empty")
    all(index -> 1 <= index <= samples,
        vcat(train_indices, validation_indices, test_indices)) ||
        error("prepared split index is out of bounds")

    return (;
        schema=PREPARED_DATASET_SCHEMA,
        input=Float32.(input),
        cached_twin_voltage=Float32.(target_voltage),
        cached_twin_spike=Float32.(target_spike),
        cached_twin_nmda=Float32.(target_nmda),
        detailed_calcium_event=Float32.(target_ca),
        detailed_dendritic_voltage=Float32.(target_dendritic),
        train_indices,
        validation_indices,
        test_indices,
        segment_region=_value(
            dataset,
            :segment_region,
            _nested_value(dataset, :segment_region, nothing),
        ),
        provenance,
        file_sha256=_sha256_file(path),
    )
end

function _run_frozen_twin(frozen, input; chunk_size=32)
    time_steps = size(input, 2)
    samples = size(input, 3)
    voltage = Matrix{Float32}(undef, time_steps, samples)
    spike_probability = Matrix{Float32}(undef, time_steps, samples)
    nmda = Array{Float32}(undef, 4, time_steps, samples)
    for first in 1:chunk_size:samples
        last = min(first + chunk_size - 1, samples)
        prediction = Twin.twin_forward(
            frozen,
            @view(input[:, :, first:last]),
        )
        voltage[:, first:last] .= prediction.voltage
        spike_probability[:, first:last] .=
            prediction.spike_probability
        nmda[:, :, first:last] .= prediction.nmda
    end
    return (; voltage, spike_probability, nmda)
end

_maximum_error(left, right) =
    maximum(abs, Float64.(left) .- Float64.(right))

function _verify_cached_twin!(dataset, prediction)
    voltage_error = _maximum_error(
        dataset.cached_twin_voltage,
        prediction.voltage,
    )
    spike_error = _maximum_error(
        dataset.cached_twin_spike,
        prediction.spike_probability,
    )
    nmda_error = _maximum_error(
        dataset.cached_twin_nmda,
        prediction.nmda,
    )
    tolerance = 2.0e-5
    maximum((voltage_error, spike_error, nmda_error)) <= tolerance ||
        error(
            "prepared cached twin targets differ from live frozen-twin " *
            "inference: voltage=$voltage_error spike=$spike_error " *
            "nmda=$nmda_error",
        )
    return (; voltage_error, spike_error, nmda_error, tolerance)
end

function _target_tensor(dataset, twin_prediction)
    time_steps = size(dataset.input, 2)
    samples = size(dataset.input, 3)
    target = Array{Float32}(undef, 11, time_steps, samples)
    target[1, :, :] = twin_prediction.voltage
    target[2, :, :] = twin_prediction.spike_probability
    target[3:6, :, :] = twin_prediction.nmda
    target[7, :, :] = dataset.detailed_calcium_event
    target[8:11, :, :] = dataset.detailed_dendritic_voltage
    return target
end

function _target_statistics(target, train_indices)
    mean_vector = zeros(Float32, 11)
    scale_vector = ones(Float32, 11)
    for target_index in (1, 3, 4, 5, 6, 8, 9, 10, 11)
        values = vec(@view target[target_index, :, train_indices])
        mean_vector[target_index] = Float32(mean(values))
        scale_vector[target_index] =
            max(Float32(std(values; corrected=false)), 1.0f-4)
    end
    return mean_vector, scale_vector
end

@inline _sigmoid(value) =
    ifelse(
        value >= zero(value),
        inv(one(value) + exp(-value)),
        exp(value) / (one(value) + exp(value)),
    )

function _softmax_columns(logits)
    shifted = logits .- maximum(logits; dims=1)
    exponent = exp.(shifted)
    return exponent ./ sum(exponent; dims=1)
end

function _project_twin_input(raw, spatial_logits, segments)
    size(raw, 1) == 6segments ||
        throw(DimensionMismatch("unexpected frozen-twin input layout"))
    spatial = _softmax_columns(spatial_logits)
    # Runtime receives conductance events whose amplitudes already include
    # nonnegative contact strength. The static strength/location plane is a
    # frozen-twin conditioning input, not an additional runtime current.
    event_plane = reshape(
        raw[1:(3segments), :],
        segments,
        3,
        size(raw, 2),
    )
    reduced = map(1:3) do receptor
        spatial * event_plane[:, receptor, :]
    end
    return vcat(
        reduced[1],
        reduced[2],
        reduced[3],
        zeros(eltype(raw), 4, size(raw, 2)),
    )
end

function _state_activation(value)
    return vcat(tanh.(value[1:10, :]), _sigmoid.(value[11:11, :]))
end

function _transition(parameters, state, input)
    proposal = _state_activation(
        parameters.recurrent_weight * state .+
        parameters.input_weight * input .+
        parameters.transition_bias,
    )
    decay = _sigmoid.(parameters.transition_decay_logit)
    return decay .* state .+ (1.0f0 .- decay) .* proposal
end

function _normalized_target(target, target_mean, target_scale)
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

@inline _bce_logit(logit, target) =
    max(logit, zero(logit)) - logit * target +
    log1p(exp(-abs(logit)))

function _sequence_loss(
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
    continuous = 0.0f0
    spike = 0.0f0
    calcium = 0.0f0
    regularization = 0.0f0
    for time in 1:time_steps
        input = _project_twin_input(
            raw_input[:, time, :],
            parameters.spatial_logits,
            segments,
        )
        predicted_state = _transition(parameters, state, input)
        output = parameters.readout_weight * predicted_state .+
            parameters.readout_bias
        normalized = _normalized_target(
            target[:, time, :],
            target_mean,
            target_scale,
        )
        continuous += sum(abs2, output[1:1, :] .- normalized[1:1, :])
        continuous += 0.75f0 *
            sum(abs2, output[3:6, :] .- normalized[3:6, :])
        continuous += 0.50f0 *
            sum(abs2, output[8:11, :] .- normalized[8:11, :])
        spike += sum(_bce_logit.(
            output[2:2, :],
            normalized[2:2, :],
        ))
        calcium += sum(_bce_logit.(
            output[7:7, :],
            normalized[7:7, :],
        ))
        regularization += sum(abs2, predicted_state)
        teacher_state = _state_activation(
            parameters.teacher_encoder * normalized .+
            parameters.teacher_encoder_bias,
        )
        state = free_fraction .* predicted_state .+
            (1.0f0 - free_fraction) .* teacher_state
    end
    count = Float32(time_steps * batch)
    return continuous / (9.0f0 * count) +
           4.0f0 * spike / count +
           1.5f0 * calcium / count +
           1.0f-4 * regularization / (11.0f0 * count) +
           1.0f-5 * (
               sum(abs2, parameters.recurrent_weight) +
               sum(abs2, parameters.input_weight) +
               sum(abs2, parameters.readout_weight)
           )
end

function _initial_parameters(rng, segments)
    spatial_logits = -2.0f0 .+
        0.02f0 .* randn(rng, Float32, 4, segments)
    @inbounds for segment in 1:segments
        spatial_logits[mod1(segment, 4), segment] += 4.0f0
    end
    return (;
        transition_decay_logit=fill(1.4f0, 11),
        recurrent_weight=
            0.08f0 .* randn(rng, Float32, 11, 11) .+
            0.55f0 .* Matrix{Float32}(I, 11, 11),
        input_weight=0.10f0 .* randn(rng, Float32, 11, 16),
        transition_bias=zeros(Float32, 11, 1),
        readout_weight=0.10f0 .* randn(rng, Float32, 11, 11),
        readout_bias=zeros(Float32, 11, 1),
        initial_state=zeros(Float32, 11, 1),
        spatial_logits,
        teacher_encoder=0.10f0 .* randn(rng, Float32, 11, 11),
        teacher_encoder_bias=zeros(Float32, 11, 1),
    )
end

function _sample_window(rng, dataset, target, indices, config)
    window = min(config.window, size(dataset.input, 2))
    time_start = rand(rng, 1:(size(dataset.input, 2) - window + 1))
    samples = rand(rng, indices, config.batch)
    time_range = time_start:(time_start + window - 1)
    return (
        dataset.input[:, time_range, samples],
        target[:, time_range, samples],
    )
end

function _free_fraction(epoch, config)
    teacher_epochs = max(config.epochs - config.free_rollout_epochs, 1)
    epoch > teacher_epochs && return 1.0f0
    return Float32(epoch - 1) / Float32(teacher_epochs)
end

function _train(
    rng,
    dataset,
    target,
    target_mean,
    target_scale,
    segments,
    config,
)
    parameters = _initial_parameters(rng, segments)
    optimizer_state = Optimisers.setup(
        Optimisers.Adam(config.learning_rate),
        parameters,
    )
    history = NamedTuple[]
    for epoch in 1:config.epochs
        fraction = _free_fraction(epoch, config)
        total_loss = 0.0
        started = time()
        for _ in 1:config.steps_per_epoch
            input_window, target_window = _sample_window(
                rng,
                dataset,
                target,
                dataset.train_indices,
                config,
            )
            loss, gradients = Zygote.withgradient(parameters) do candidate
                _sequence_loss(
                    candidate,
                    input_window,
                    target_window,
                    target_mean,
                    target_scale,
                    segments,
                    fraction,
                )
            end
            isfinite(loss) || error("distillation loss is non-finite")
            optimizer_state, parameters = Optimisers.update(
                optimizer_state,
                parameters,
                only(gradients),
            )
            total_loss += Float64(loss)
        end
        record = (;
            epoch,
            loss=total_loss / config.steps_per_epoch,
            free_rollout_fraction=fraction,
            elapsed_seconds=time() - started,
        )
        push!(history, record)
        @printf(
            "final distill epoch %d/%d loss=%.6f free=%.3f time=%.2fs\n",
            epoch,
            config.epochs,
            record.loss,
            fraction,
            record.elapsed_seconds,
        )
        flush(stdout)
    end
    return parameters, history
end

function _rollout(parameters, raw_input, segments)
    time_steps = size(raw_input, 2)
    batch = size(raw_input, 3)
    state = repeat(parameters.initial_state, 1, batch)
    output = Array{Float32}(undef, 11, time_steps, batch)
    for time in 1:time_steps
        input = _project_twin_input(
            raw_input[:, time, :],
            parameters.spatial_logits,
            segments,
        )
        state = _transition(parameters, state, input)
        output[:, time, :] =
            parameters.readout_weight * state .+
            parameters.readout_bias
    end
    return output
end

function _physical_output(raw, target_mean, target_scale)
    output = raw .* reshape(target_scale, :, 1, 1) .+
        reshape(target_mean, :, 1, 1)
    output[2, :, :] .= _sigmoid.(raw[2, :, :])
    output[7, :, :] .= _sigmoid.(raw[7, :, :])
    return output
end

function _auroc(score, target)
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

_rmse(prediction, target) =
    sqrt(mean(abs2, Float64.(prediction) .- Float64.(target)))

function _correlation(prediction, target)
    x = vec(Float64.(prediction))
    y = vec(Float64.(target))
    x .-= mean(x)
    y .-= mean(y)
    denominator = sqrt(sum(abs2, x) * sum(abs2, y))
    denominator <= eps(Float64) && return NaN
    return sum(x .* y) / denominator
end

function _metrics(
    parameters,
    dataset,
    target,
    indices,
    segments,
    target_mean,
    target_scale,
)
    raw = _rollout(parameters, dataset.input[:, :, indices], segments)
    prediction = _physical_output(raw, target_mean, target_scale)
    truth = target[:, :, indices]
    return (;
        samples=length(indices),
        free_rollout_horizon=size(truth, 2),
        soma_voltage_rmse_mv=_rmse(prediction[1, :, :], truth[1, :, :]),
        soma_voltage_correlation=
            _correlation(prediction[1, :, :], truth[1, :, :]),
        spike_auroc=_auroc(prediction[2, :, :], truth[2, :, :]),
        nmda_rmse_by_region=[
            _rmse(prediction[2 + region, :, :], truth[2 + region, :, :])
            for region in 1:4
        ],
        nmda_correlation_by_region=[
            _correlation(
                prediction[2 + region, :, :],
                truth[2 + region, :, :],
            )
            for region in 1:4
        ],
        calcium_event_auroc=
            _auroc(prediction[7, :, :], truth[7, :, :]),
        dendritic_voltage_rmse_mv=[
            _rmse(prediction[7 + region, :, :], truth[7 + region, :, :])
            for region in 1:4
        ],
    )
end

function _region_projection(dataset, spatial)
    projection = zeros(Float32, 4, 4)
    labels = dataset.segment_region
    if labels === nothing || length(labels) != size(spatial, 2)
        projection .= Matrix{Float32}(I, 4, 4)
        return projection
    end
    count = zeros(Int, 4)
    @inbounds for segment in axes(spatial, 2)
        raw = labels[segment]
        label = lowercase(String(raw))
        region = occursin("soma", label) ? 1 :
            occursin("basal", label) ? 2 :
            occursin("tuft", label) ? 4 : 3
        projection[:, region] .+= spatial[:, segment]
        count[region] += 1
    end
    @inbounds for region in 1:4
        if count[region] == 0
            projection[region, region] = 1.0f0
        else
            projection[:, region] ./= count[region]
        end
    end
    return projection
end

function _freeze(
    trained,
    dataset,
    frozen,
    target_mean,
    target_scale,
    config_hash,
)
    spatial = Float32.(_softmax_columns(trained.spatial_logits))
    return Cell.DistilledParameters(
        dt_ms=frozen.model.config.dt_ms,
        transition_decay=Float32.(
            _sigmoid.(trained.transition_decay_logit),
        ),
        recurrent_weight=trained.recurrent_weight,
        input_weight=trained.input_weight,
        transition_bias=vec(trained.transition_bias),
        readout_weight=trained.readout_weight,
        readout_bias=vec(trained.readout_bias),
        target_mean,
        target_scale,
        initial_state=vec(trained.initial_state),
        compartment_projection=spatial,
        region_projection=_region_projection(dataset, spatial),
        spike_threshold=0.5f0,
        teacher_schema="PaperDigitalTwin+official-NEURON-mixed-v1",
        detailed_kernel_hash=
            dataset.provenance.detailed_kernel_hash,
        morphology_hash=dataset.provenance.morphology_hash,
        frozen_twin_parameter_hash=frozen.parameter_sha256,
        frozen_twin_artifact_hash=frozen.artifact_sha256,
        distillation_dataset_hash=dataset.file_sha256,
        distillation_config_hash=config_hash,
    )
end

function _atomic_jldsave(path; payload)
    destination = abspath(path)
    mkpath(dirname(destination))
    temporary = destination * ".tmp." * string(getpid())
    JLD2.jldsave(temporary; payload)
    mv(temporary, destination; force=true)
    return destination
end

function _atomic_json(path, value)
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

function run_final_distillation(config)
    _validate_config(config)
    frozen = Twin.load_frozen_twin(config.frozen_twin)
    frozen_before = Twin.assert_frozen_unchanged(frozen)
    dataset = _load_prepared_dataset(config.dataset, frozen)
    prediction = _run_frozen_twin(frozen, dataset.input)
    cache_check = _verify_cached_twin!(dataset, prediction)
    target = _target_tensor(dataset, prediction)
    target_mean, target_scale =
        _target_statistics(target, dataset.train_indices)
    config_record = (;
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        batch=config.batch,
        window=min(config.window, size(dataset.input, 2)),
        learning_rate=config.learning_rate,
        free_rollout_epochs=config.free_rollout_epochs,
        seed=config.seed,
        dataset_schema=dataset.schema,
        mixed_supervision=(
            twin=(:soma_voltage, :soma_spike, :nmda_current),
            official_neuron=(:calcium_event, :dendritic_voltage),
        ),
        state_semantics=(
            "dendritic_voltage_latent_1",
            "dendritic_voltage_latent_2",
            "dendritic_voltage_latent_3",
            "dendritic_voltage_latent_4",
            "nmda_current_latent_1",
            "nmda_current_latent_2",
            "nmda_current_latent_3",
            "nmda_current_latent_4",
            "apical_context_latent",
            "soma_voltage_latent",
            "adaptation_calcium_summary",
        ),
        soma_spike_is_sole_external_event=true,
    )
    config_hash = _sha256_value(config_record)
    rng = Xoshiro(config.seed)
    trained, history = _train(
        rng,
        dataset,
        target,
        target_mean,
        target_scale,
        frozen.model.config.segments,
        config,
    )
    metrics = (;
        train=_metrics(
            trained,
            dataset,
            target,
            dataset.train_indices,
            frozen.model.config.segments,
            target_mean,
            target_scale,
        ),
        validation=_metrics(
            trained,
            dataset,
            target,
            dataset.validation_indices,
            frozen.model.config.segments,
            target_mean,
            target_scale,
        ),
        test=_metrics(
            trained,
            dataset,
            target,
            dataset.test_indices,
            frozen.model.config.segments,
            target_mean,
            target_scale,
        ),
    )
    gate_passed = isfinite(metrics.test.spike_auroc) &&
        metrics.test.spike_auroc >= config.minimum_spike_auroc
    parameters = _freeze(
        trained,
        dataset,
        frozen,
        target_mean,
        target_scale,
        config_hash,
    )
    parameter_hash = Cell.parameter_sha256(parameters)
    frozen_after = Twin.assert_frozen_unchanged(frozen)
    frozen_before == frozen_after ||
        error("frozen-twin integrity record changed during distillation")
    gate = (;
        passed=gate_passed,
        held_out_spike_auroc=metrics.test.spike_auroc,
        minimum_spike_auroc=config.minimum_spike_auroc,
    )
    payload = (;
        schema=Cell.DISTILLED_ARTIFACT_SCHEMA,
        parameters,
        parameter_sha256=parameter_hash,
        frozen_internal=true,
        teacher_hash=dataset.provenance.detailed_teacher_hash,
        digital_twin_hash=frozen.artifact_sha256,
        detailed_kernel_hash=parameters.detailed_kernel_hash,
        morphology_hash=parameters.morphology_hash,
        official_modeldb_source_hash=
            dataset.provenance.official_modeldb_source_hash,
        frozen_twin_parameter_hash=
            parameters.frozen_twin_parameter_hash,
        frozen_twin_artifact_hash=
            parameters.frozen_twin_artifact_hash,
        distillation_dataset_hash=
            parameters.distillation_dataset_hash,
        distillation_config_hash=
            parameters.distillation_config_hash,
        source_dataset_hash=dataset.provenance.source_dataset_hash,
        dt_ms=parameters.dt_ms,
        mixed_supervision=(;
            twin_targets=(:soma_voltage, :soma_spike, :nmda_current),
            detailed_targets=(:calcium_event, :dendritic_voltage),
            live_frozen_twin_inference=true,
            cached_twin_target_check=cache_check,
        ),
        frozen_twin_integrity_before=frozen_before,
        frozen_twin_integrity_after=frozen_after,
        metrics,
        gate,
        config=config_record,
        history,
        created_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
    )
    _atomic_jldsave(config.output; payload)
    # Re-load through the production loader before publishing the report.
    reloaded = Cell.load_distilled_artifact(config.output)
    Cell.assert_parameter_sha256(reloaded, parameter_hash)
    report = (;
        schema=payload.schema,
        artifact_path=config.output,
        artifact_sha256=Cell.artifact_sha256(config.output),
        parameter_sha256=parameter_hash,
        teacher_hash=payload.teacher_hash,
        digital_twin_hash=payload.digital_twin_hash,
        lineage=(;
            detailed_kernel_hash=parameters.detailed_kernel_hash,
            morphology_hash=parameters.morphology_hash,
            official_modeldb_source_hash=
                dataset.provenance.official_modeldb_source_hash,
            frozen_twin_parameter_hash=
                parameters.frozen_twin_parameter_hash,
            frozen_twin_artifact_hash=
                parameters.frozen_twin_artifact_hash,
            distillation_dataset_hash=
                parameters.distillation_dataset_hash,
            distillation_config_hash=
                parameters.distillation_config_hash,
        ),
        state_count=Cell.DISTILLED_STATE_DIM,
        frozen_internal=true,
        metrics,
        gate,
        cache_check,
        history,
    )
    _atomic_json(config.metrics, report)
    @info "Final 11-state distillation complete" report.artifact_path report.artifact_sha256 gate metrics.test
    if !gate_passed && !config.allow_below_gate
        error(
            @sprintf(
                "held-out spike AUROC %.6f is below required %.6f; " *
                "candidate artifact retained but production loader rejects it",
                metrics.test.spike_auroc,
                config.minimum_spike_auroc,
            ),
        )
    end
    return report
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_final_distillation(_parse_arguments(ARGS))
end
