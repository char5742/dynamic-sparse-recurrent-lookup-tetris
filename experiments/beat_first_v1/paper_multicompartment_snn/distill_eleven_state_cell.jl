using Dates
using JLD2
using JSON3
using Optimisers
using Printf
using Random
using SHA
using Statistics
using Zygote

if !isdefined(Main, :DistilledElevenStateCell)
    include(joinpath(@__DIR__, "DistilledElevenStateCell.jl"))
end
using .DistilledElevenStateCell

const DEFAULT_OUTPUT =
    joinpath(@__DIR__, "artifacts", "distilled_eleven_state_cell.jld2")

@inline _payload_get(payload, name::Symbol, default=nothing) =
    if payload isa AbstractDict
        get(payload, name, get(payload, String(name), default))
    elseif hasproperty(payload, name)
        getproperty(payload, name)
    else
        default
    end

function _file_sha256(path::AbstractString)
    source = abspath(path)
    isfile(source) || error("file is absent: $source")
    return bytes2hex(SHA.sha256(read(source)))
end

function _parse_arguments(arguments)
    options = Dict{String,String}(
        "dataset" => "",
        "teacher" => "",
        "output" => DEFAULT_OUTPUT,
        "metrics" => "",
        "epochs" => "45",
        "batch" => "32",
        "window" => "256",
        "steps-per-epoch" => "64",
        "learning-rate" => "0.001",
        "seed" => "5987321",
        "minimum-spike-auroc" => "0.95",
        "free-rollout-epochs" => "12",
        "morphology-sha256" => "",
        "cell-mechanism-sha256" => "",
        "allow-below-gate" => "false",
        "allow-missing-diagnostics" => "false",
    )
    index = 1
    while index <= length(arguments)
        token = arguments[index]
        startswith(token, "--") ||
            error("unexpected positional argument: $token")
        key = token[3:end]
        haskey(options, key) || error("unknown option --$key")
        index == length(arguments) &&
            error("missing value after --$key")
        options[key] = arguments[index + 1]
        index += 2
    end
    isempty(options["dataset"]) &&
        error("--dataset is required")
    isempty(options["teacher"]) &&
        error("--teacher frozen digital-twin artifact is required")
    return (;
        dataset=abspath(options["dataset"]),
        teacher=abspath(options["teacher"]),
        output=abspath(options["output"]),
        metrics=isempty(options["metrics"]) ?
            abspath(options["output"] * ".metrics.json") :
            abspath(options["metrics"]),
        epochs=parse(Int, options["epochs"]),
        batch=parse(Int, options["batch"]),
        window=parse(Int, options["window"]),
        steps_per_epoch=parse(Int, options["steps-per-epoch"]),
        learning_rate=parse(Float32, options["learning-rate"]),
        seed=parse(UInt64, options["seed"]),
        minimum_spike_auroc=parse(Float64, options["minimum-spike-auroc"]),
        free_rollout_epochs=parse(Int, options["free-rollout-epochs"]),
        morphology_sha256=options["morphology-sha256"],
        cell_mechanism_sha256=options["cell-mechanism-sha256"],
        allow_below_gate=lowercase(options["allow-below-gate"]) == "true",
        allow_missing_diagnostics=
            lowercase(options["allow-missing-diagnostics"]) == "true",
    )
end

function _load_dataset(path::AbstractString; allow_missing::Bool=false)
    isfile(path) || error("distillation dataset is absent: $path")
    data = JLD2.load(path)
    payload = haskey(data, "dataset") ? data["dataset"] :
        haskey(data, "payload") ? data["payload"] : data

    input = _payload_get(payload, :input)
    voltage = _payload_get(payload, :target_voltage)
    spike = _payload_get(payload, :target_spike)
    nmda = _payload_get(payload, :target_nmda)
    calcium = _payload_get(payload, :target_calcium_event)
    dendritic = _payload_get(payload, :target_dendritic_voltage)
    metadata = _payload_get(payload, :metadata, NamedTuple())

    input === nothing && error("dataset has no input")
    voltage === nothing && error("dataset has no target_voltage")
    spike === nothing && error("dataset has no target_spike")
    nmda === nothing && error("dataset has no target_nmda")
    ndims(input) == 3 ||
        throw(DimensionMismatch("input must be input_dim x time x sample"))
    time = size(input, 2)
    samples = size(input, 3)
    size(voltage) == (time, samples) ||
        throw(DimensionMismatch("target_voltage shape differs"))
    size(spike) == (time, samples) ||
        throw(DimensionMismatch("target_spike shape differs"))
    size(nmda) == (4, time, samples) ||
        throw(DimensionMismatch("target_nmda must be 4 x time x sample"))

    if calcium === nothing
        allow_missing || error(
            "dataset has no target_calcium_event; paper-mechanism " *
            "distillation requires Ca-event supervision",
        )
        calcium = zeros(Float32, time, samples)
    end
    if dendritic === nothing
        allow_missing || error(
            "dataset has no target_dendritic_voltage; paper-mechanism " *
            "distillation requires selected dendritic voltages",
        )
        dendritic = zeros(Float32, 4, time, samples)
    end
    size(calcium) == (time, samples) ||
        throw(DimensionMismatch("target_calcium_event shape differs"))
    size(dendritic) == (4, time, samples) ||
        throw(DimensionMismatch(
            "target_dendritic_voltage must be 4 x time x sample",
        ))

    train_indices = Int.(_payload_get(
        payload,
        :train_indices,
        collect(1:max(1, floor(Int, 0.8samples))),
    ))
    validation_indices = Int.(_payload_get(
        payload,
        :validation_indices,
        collect(
            max(1, floor(Int, 0.8samples) + 1):
            max(1, floor(Int, 0.9samples)),
        ),
    ))
    test_indices = Int.(_payload_get(
        payload,
        :test_indices,
        collect(max(1, floor(Int, 0.9samples) + 1):samples),
    ))
    isempty(train_indices) && error("training split is empty")
    isempty(validation_indices) &&
        (validation_indices = copy(train_indices[1:min(end, 8)]))
    isempty(test_indices) &&
        (test_indices = copy(validation_indices))

    return (;
        input=Float32.(input),
        target_voltage=Float32.(voltage),
        target_spike=Float32.(spike),
        target_nmda=Float32.(nmda),
        target_calcium_event=Float32.(calcium),
        target_dendritic_voltage=Float32.(dendritic),
        train_indices,
        validation_indices,
        test_indices,
        metadata,
        dataset_sha256=_file_sha256(path),
    )
end

function _input_anatomy(dataset)
    input_dim = size(dataset.input, 1)
    metadata = dataset.metadata
    raw_compartment = _payload_get(metadata, :input_compartment)
    raw_receptor = _payload_get(metadata, :input_receptor)
    if raw_compartment !== nothing && raw_receptor !== nothing
        compartment = Int.(raw_compartment)
        receptor = Int.(raw_receptor)
        length(compartment) == input_dim ||
            throw(DimensionMismatch("input_compartment length differs"))
        length(receptor) == input_dim ||
            throw(DimensionMismatch("input_receptor length differs"))
        all(value -> 1 <= value <= 4, receptor) ||
            error("input_receptor must use indices 1:4")
        compartment_count = maximum(compartment)
        compartment_count >= 1 || error("invalid input compartment")
        return (; compartment, receptor, compartment_count)
    end
    input_dim == DISTILLED_INPUT_DIM || error(
        "dataset needs input_compartment/input_receptor metadata " *
        "unless input_dim is exactly 16",
    )
    # Direct format is receptor-major, with four reduced spatial channels.
    compartment = repeat(collect(1:4), 4)
    receptor = repeat(collect(1:4), inner=4)
    return (; compartment, receptor, compartment_count=4)
end

function _target_statistics(dataset)
    indices = dataset.train_indices
    continuous = (
        vec(dataset.target_voltage[:, indices]),
        vec(dataset.target_nmda[1, :, indices]),
        vec(dataset.target_nmda[2, :, indices]),
        vec(dataset.target_nmda[3, :, indices]),
        vec(dataset.target_nmda[4, :, indices]),
        vec(dataset.target_dendritic_voltage[1, :, indices]),
        vec(dataset.target_dendritic_voltage[2, :, indices]),
        vec(dataset.target_dendritic_voltage[3, :, indices]),
        vec(dataset.target_dendritic_voltage[4, :, indices]),
    )
    mean_vector = zeros(Float32, DISTILLED_TARGET_DIM)
    scale_vector = ones(Float32, DISTILLED_TARGET_DIM)
    target_indices = (1, 3, 4, 5, 6, 8, 9, 10, 11)
    for (values, target) in zip(continuous, target_indices)
        mean_vector[target] = Float32(mean(values))
        scale_vector[target] = max(Float32(std(values)), 1.0f-4)
    end
    return mean_vector, scale_vector
end

function _target_window(dataset, starts, samples, window)
    batch = length(samples)
    target = Array{Float32}(undef, DISTILLED_TARGET_DIM, window, batch)
    @inbounds for batch_index in 1:batch
        sample = samples[batch_index]
        start = starts[batch_index]
        for offset in 1:window
            time = start + offset - 1
            target[1, offset, batch_index] =
                dataset.target_voltage[time, sample]
            target[2, offset, batch_index] =
                dataset.target_spike[time, sample]
            for region in 1:4
                target[2 + region, offset, batch_index] =
                    dataset.target_nmda[region, time, sample]
            end
            target[7, offset, batch_index] =
                dataset.target_calcium_event[time, sample]
            for region in 1:4
                target[7 + region, offset, batch_index] =
                    dataset.target_dendritic_voltage[region, time, sample]
            end
        end
    end
    return target
end

function _input_window(dataset, starts, samples, window)
    batch = length(samples)
    input = Array{Float32}(
        undef,
        size(dataset.input, 1),
        window,
        batch,
    )
    @inbounds for batch_index in 1:batch
        sample = samples[batch_index]
        start = starts[batch_index]
        copyto!(
            view(input, :, :, batch_index),
            view(dataset.input, :, start:start + window - 1, sample),
        )
    end
    return input
end

function _sample_window(rng, dataset, indices, batch, window)
    time = size(dataset.input, 2)
    window <= time ||
        throw(ArgumentError("window exceeds trajectory length"))
    samples = rand(rng, indices, batch)
    starts = rand(rng, 1:(time - window + 1), batch)
    return (
        _input_window(dataset, starts, samples, window),
        _target_window(dataset, starts, samples, window),
    )
end

function _softmax_columns(logits)
    shifted = logits .- maximum(logits; dims=1)
    exponent = exp.(shifted)
    return exponent ./ sum(exponent; dims=1)
end

function _project_input(raw, spatial_logits, anatomy)
    spatial = _softmax_columns(spatial_logits)
    contact_projection = spatial[:, anatomy.compartment]
    projected = map(1:4) do receptor
        mask = reshape(
            Float32.(anatomy.receptor .== receptor),
            :,
            1,
        )
        (contact_projection .* transpose(mask)) * raw
    end
    return reduce(vcat, projected)
end

@inline _stable_sigmoid(value) =
    ifelse(
        value >= zero(value),
        inv(one(value) + exp(-value)),
        exp(value) / (one(value) + exp(value)),
    )

function _state_activation(value)
    return vcat(tanh.(value[1:10, :]), _stable_sigmoid.(value[11:11, :]))
end

function _transition(parameters, state, input)
    proposal = _state_activation(
        parameters.recurrent_weight * state .+
        parameters.input_weight * input .+
        parameters.transition_bias,
    )
    decay = _stable_sigmoid.(parameters.transition_decay_logit)
    return decay .* state .+ (1.0f0 .- decay) .* proposal
end

function _normalize_target(target, mean_vector, scale_vector)
    normalized = (target .- reshape(mean_vector, :, 1)) ./
        reshape(scale_vector, :, 1)
    return vcat(
        normalized[1:1, :],
        target[2:2, :],
        normalized[3:6, :],
        target[7:7, :],
        normalized[8:11, :],
    )
end

@inline function _binary_cross_entropy_logit(logit, target)
    return max(logit, zero(logit)) - logit * target +
        log1p(exp(-abs(logit)))
end

function _sequence_loss(
    parameters,
    raw_input,
    target,
    mean_vector,
    scale_vector,
    anatomy,
    free_rollout_fraction::Float32,
)
    batch = size(raw_input, 3)
    time = size(raw_input, 2)
    state = repeat(parameters.initial_state, 1, batch)
    continuous_loss = 0.0f0
    spike_loss = 0.0f0
    calcium_loss = 0.0f0
    latent_loss = 0.0f0
    target_count = Float32(time * batch)

    for step in 1:time
        raw = raw_input[:, step, :]
        input = _project_input(
            raw,
            parameters.spatial_logits,
            anatomy,
        )
        predicted_state = _transition(parameters, state, input)
        output = parameters.readout_weight * predicted_state .+
            parameters.readout_bias
        normalized_target = _normalize_target(
            target[:, step, :],
            mean_vector,
            scale_vector,
        )

        continuous_loss += sum(abs2, output[1:1, :] .-
            normalized_target[1:1, :])
        continuous_loss += 0.75f0 * sum(abs2, output[3:6, :] .-
            normalized_target[3:6, :])
        continuous_loss += 0.50f0 * sum(abs2, output[8:11, :] .-
            normalized_target[8:11, :])
        spike_loss += sum(
            _binary_cross_entropy_logit.(
                output[2:2, :],
                normalized_target[2:2, :],
            ),
        )
        calcium_loss += sum(
            _binary_cross_entropy_logit.(
                output[7:7, :],
                normalized_target[7:7, :],
            ),
        )
        latent_loss += sum(abs2, predicted_state) /
            Float32(DISTILLED_STATE_DIM)

        teacher_state = _state_activation(
            parameters.teacher_encoder * normalized_target .+
            parameters.teacher_encoder_bias,
        )
        state =
            free_rollout_fraction .* predicted_state .+
            (1.0f0 - free_rollout_fraction) .* teacher_state
    end

    loss =
        continuous_loss / (9.0f0 * target_count) +
        4.0f0 * spike_loss / target_count +
        1.5f0 * calcium_loss / target_count +
        1.0f-4 * latent_loss / target_count +
        1.0f-5 * (
            sum(abs2, parameters.recurrent_weight) +
            sum(abs2, parameters.input_weight) +
            sum(abs2, parameters.readout_weight)
        )
    return loss
end

function _initial_training_parameters(rng, anatomy)
    state_dim = DISTILLED_STATE_DIM
    input_dim = DISTILLED_INPUT_DIM
    target_dim = DISTILLED_TARGET_DIM
    spatial_logits = -2.0f0 .+
        0.02f0 .* randn(
            rng,
            Float32,
            4,
            anatomy.compartment_count,
        )
    @inbounds for compartment in 1:anatomy.compartment_count
        spatial_logits[mod1(compartment, 4), compartment] += 4.0f0
    end
    return (;
        transition_decay_logit=fill(1.4f0, state_dim),
        recurrent_weight=
            0.10f0 .* randn(rng, Float32, state_dim, state_dim) .+
            Matrix{Float32}(I, state_dim, state_dim) .* 0.55f0,
        input_weight=
            0.12f0 .* randn(rng, Float32, state_dim, input_dim),
        transition_bias=zeros(Float32, state_dim, 1),
        readout_weight=
            0.10f0 .* randn(rng, Float32, target_dim, state_dim),
        readout_bias=zeros(Float32, target_dim, 1),
        initial_state=zeros(Float32, state_dim, 1),
        spatial_logits,
        teacher_encoder=
            0.10f0 .* randn(rng, Float32, state_dim, target_dim),
        teacher_encoder_bias=zeros(Float32, state_dim, 1),
    )
end

function _free_rollout_fraction(epoch, epochs, free_rollout_epochs)
    teacher_epochs = max(1, epochs - free_rollout_epochs)
    epoch <= teacher_epochs &&
        return Float32(epoch - 1) / Float32(max(teacher_epochs, 1))
    return 1.0f0
end

function _train(
    rng,
    dataset,
    anatomy,
    mean_vector,
    scale_vector,
    config,
)
    parameters = _initial_training_parameters(rng, anatomy)
    optimizer_state = Optimisers.setup(
        Optimisers.Adam(config.learning_rate),
        parameters,
    )
    history = NamedTuple[]
    for epoch in 1:config.epochs
        rollout = _free_rollout_fraction(
            epoch,
            config.epochs,
            config.free_rollout_epochs,
        )
        epoch_loss = 0.0
        start_time = time()
        for _ in 1:config.steps_per_epoch
            raw_input, target = _sample_window(
                rng,
                dataset,
                dataset.train_indices,
                config.batch,
                min(config.window, size(dataset.input, 2)),
            )
            loss, pullback = Zygote.withgradient(parameters) do candidate
                _sequence_loss(
                    candidate,
                    raw_input,
                    target,
                    mean_vector,
                    scale_vector,
                    anatomy,
                    rollout,
                )
            end
            gradient = only(pullback)
            isfinite(loss) || error("distillation loss is non-finite")
            optimizer_state, parameters = Optimisers.update(
                optimizer_state,
                parameters,
                gradient,
            )
            epoch_loss += Float64(loss)
        end
        elapsed = time() - start_time
        record = (;
            epoch,
            loss=epoch_loss / config.steps_per_epoch,
            free_rollout_fraction=rollout,
            elapsed_seconds=elapsed,
        )
        push!(history, record)
        @printf(
            "distill epoch %d/%d loss=%.6f free_rollout=%.3f time=%.2fs\n",
            epoch,
            config.epochs,
            record.loss,
            rollout,
            elapsed,
        )
        flush(stdout)
    end
    return parameters, history
end

function _physical_output(output, mean_vector, scale_vector)
    physical = output .* reshape(scale_vector, :, 1, 1) .+
        reshape(mean_vector, :, 1, 1)
    physical[2, :, :] .= 1.0f0 ./
        (1.0f0 .+ exp.(-output[2, :, :]))
    physical[7, :, :] .= 1.0f0 ./
        (1.0f0 .+ exp.(-output[7, :, :]))
    return physical
end

function _rollout(parameters, dataset, indices, anatomy)
    time = size(dataset.input, 2)
    batch = length(indices)
    state = repeat(parameters.initial_state, 1, batch)
    output = Array{Float32}(
        undef,
        DISTILLED_TARGET_DIM,
        time,
        batch,
    )
    for step in 1:time
        raw = dataset.input[:, step, indices]
        input = _project_input(
            raw,
            parameters.spatial_logits,
            anatomy,
        )
        state = _transition(parameters, state, input)
        output[:, step, :] =
            parameters.readout_weight * state .+
            parameters.readout_bias
    end
    return output
end

function _target_tensor(dataset, indices)
    time = size(dataset.input, 2)
    target = Array{Float32}(
        undef,
        DISTILLED_TARGET_DIM,
        time,
        length(indices),
    )
    target[1, :, :] = dataset.target_voltage[:, indices]
    target[2, :, :] = dataset.target_spike[:, indices]
    target[3:6, :, :] = dataset.target_nmda[:, :, indices]
    target[7, :, :] = dataset.target_calcium_event[:, indices]
    target[8:11, :, :] =
        dataset.target_dendritic_voltage[:, :, indices]
    return target
end

function _auroc(score, target)
    score_vector = vec(Float64.(score))
    target_vector = vec(target .>= 0.5f0)
    positive = count(identity, target_vector)
    negative = length(target_vector) - positive
    (positive == 0 || negative == 0) && return NaN
    order = sortperm(score_vector)
    rank_sum = 0.0
    index = 1
    while index <= length(order)
        last = index
        while last < length(order) &&
              score_vector[order[last + 1]] == score_vector[order[index]]
            last += 1
        end
        average_rank = 0.5 * (index + last)
        for position in index:last
            target_vector[order[position]] &&
                (rank_sum += average_rank)
        end
        index = last + 1
    end
    return (
        rank_sum - positive * (positive + 1) / 2
    ) / (positive * negative)
end

function _correlation(predicted, target)
    x = vec(Float64.(predicted))
    y = vec(Float64.(target))
    x_centered = x .- mean(x)
    y_centered = y .- mean(y)
    denominator = sqrt(sum(abs2, x_centered) * sum(abs2, y_centered))
    denominator <= eps(Float64) && return NaN
    return sum(x_centered .* y_centered) / denominator
end

_rmse(predicted, target) =
    sqrt(mean(abs2, Float64.(predicted) .- Float64.(target)))

function _split_metrics(parameters, dataset, indices, anatomy, mean_vector, scale_vector)
    raw_output = _rollout(parameters, dataset, indices, anatomy)
    predicted = _physical_output(raw_output, mean_vector, scale_vector)
    target = _target_tensor(dataset, indices)
    predicted_spike = predicted[2, :, :] .>= 0.5f0
    target_spike = target[2, :, :] .>= 0.5f0
    true_positive = count(predicted_spike .& target_spike)
    false_positive = count(predicted_spike .& .!target_spike)
    false_negative = count(.!predicted_spike .& target_spike)
    precision = true_positive /
        max(true_positive + false_positive, 1)
    recall = true_positive /
        max(true_positive + false_negative, 1)
    f1 = 2precision * recall / max(precision + recall, eps())
    return (;
        samples=length(indices),
        trajectory_steps=size(dataset.input, 2),
        soma_voltage_rmse_mv=_rmse(predicted[1, :, :], target[1, :, :]),
        soma_voltage_correlation=
            _correlation(predicted[1, :, :], target[1, :, :]),
        spike_auroc=_auroc(predicted[2, :, :], target[2, :, :]),
        spike_f1=f1,
        nmda_rmse_by_region=[
            _rmse(predicted[2 + region, :, :], target[2 + region, :, :])
            for region in 1:4
        ],
        nmda_correlation_by_region=[
            _correlation(
                predicted[2 + region, :, :],
                target[2 + region, :, :],
            )
            for region in 1:4
        ],
        calcium_event_auroc=_auroc(
            predicted[7, :, :],
            target[7, :, :],
        ),
        dendritic_voltage_rmse_mv=[
            _rmse(predicted[7 + region, :, :], target[7 + region, :, :])
            for region in 1:4
        ],
        free_rollout_horizon=size(dataset.input, 2),
    )
end

function _frozen_parameters(
    trained,
    anatomy,
    mean_vector,
    scale_vector,
    dataset,
    teacher_sha256,
)
    spatial = Float32.(_softmax_columns(trained.spatial_logits))
    # The runtime projection is compartment based. If multiple input channels
    # share a compartment they necessarily share the same distilled location.
    region_projection = zeros(Float32, 4, 4)
    compartment_region = _payload_get(
        dataset.metadata,
        :compartment_region,
        nothing,
    )
    if compartment_region === nothing
        @inbounds for region in 1:4
            region_projection[region, region] = 1.0f0
        end
    else
        counts = zeros(Int, 4)
        @inbounds for compartment in 1:anatomy.compartment_count
            region = clamp(Int(compartment_region[compartment]), 1, 4)
            region_projection[:, region] .+= spatial[:, compartment]
            counts[region] += 1
        end
        @inbounds for region in 1:4
            if counts[region] == 0
                region_projection[region, region] = 1.0f0
            else
                region_projection[:, region] ./= counts[region]
            end
        end
    end
    return DistilledParameters(
        dt_ms=Float32(_payload_get(dataset.metadata, :dt_ms, 1.0f0)),
        transition_decay=Float32.(
            _stable_sigmoid.(trained.transition_decay_logit),
        ),
        recurrent_weight=trained.recurrent_weight,
        input_weight=trained.input_weight,
        transition_bias=vec(trained.transition_bias),
        readout_weight=trained.readout_weight,
        readout_bias=vec(trained.readout_bias),
        target_mean=mean_vector,
        target_scale=scale_vector,
        initial_state=vec(trained.initial_state),
        compartment_projection=spatial,
        region_projection,
        spike_threshold=0.5f0,
        teacher_sha256,
        teacher_schema=String(_payload_get(
            dataset.metadata,
            :teacher_schema,
            "PaperDigitalTwin-frozen",
        )),
    )
end

function _required_provenance(config, dataset, teacher_hash)
    metadata = dataset.metadata
    morphology_sha256 = isempty(config.morphology_sha256) ?
        String(_payload_get(metadata, :morphology_sha256, "")) :
        config.morphology_sha256
    mechanism_sha256 = isempty(config.cell_mechanism_sha256) ?
        String(_payload_get(metadata, :cell_mechanism_sha256, "")) :
        config.cell_mechanism_sha256
    isempty(morphology_sha256) && error(
        "morphology_sha256 is required in dataset metadata or CLI",
    )
    isempty(mechanism_sha256) && error(
        "cell_mechanism_sha256 is required in dataset metadata or CLI",
    )
    return (;
        teacher_hash,
        digital_twin_hash=teacher_hash,
        morphology_sha256,
        cell_mechanism_sha256=mechanism_sha256,
    )
end

function _atomic_jldsave(path::AbstractString; payload)
    destination = abspath(path)
    mkpath(dirname(destination))
    temporary = destination * ".tmp." * string(getpid())
    JLD2.jldsave(temporary; payload)
    mv(temporary, destination; force=true)
    return destination
end

function _atomic_json(path::AbstractString, value)
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

function run_distillation(config)
    config.epochs >= 1 || error("epochs must be positive")
    config.batch >= 1 || error("batch must be positive")
    config.steps_per_epoch >= 1 || error("steps-per-epoch must be positive")
    isfile(config.teacher) ||
        error("frozen digital twin is absent: $(config.teacher)")

    dataset = _load_dataset(
        config.dataset;
        allow_missing=config.allow_missing_diagnostics,
    )
    anatomy = _input_anatomy(dataset)
    mean_vector, scale_vector = _target_statistics(dataset)
    teacher_hash = _file_sha256(config.teacher)
    provenance = _required_provenance(config, dataset, teacher_hash)
    rng = Xoshiro(config.seed)
    trained, history = _train(
        rng,
        dataset,
        anatomy,
        mean_vector,
        scale_vector,
        config,
    )
    frozen = _frozen_parameters(
        trained,
        anatomy,
        mean_vector,
        scale_vector,
        dataset,
        teacher_hash,
    )
    metrics = (;
        train=_split_metrics(
            trained,
            dataset,
            dataset.train_indices,
            anatomy,
            mean_vector,
            scale_vector,
        ),
        validation=_split_metrics(
            trained,
            dataset,
            dataset.validation_indices,
            anatomy,
            mean_vector,
            scale_vector,
        ),
        test=_split_metrics(
            trained,
            dataset,
            dataset.test_indices,
            anatomy,
            mean_vector,
            scale_vector,
        ),
    )
    gate_pass = isfinite(metrics.test.spike_auroc) &&
        metrics.test.spike_auroc >= config.minimum_spike_auroc
    payload = (;
        schema=DISTILLED_ARTIFACT_SCHEMA,
        parameters=frozen,
        parameter_sha256=parameter_sha256(frozen),
        teacher_sha256=teacher_hash,
        teacher_hash=provenance.teacher_hash,
        digital_twin_hash=provenance.digital_twin_hash,
        morphology_sha256=provenance.morphology_sha256,
        cell_mechanism_sha256=provenance.cell_mechanism_sha256,
        dt_ms=frozen.dt_ms,
        frozen_internal=true,
        dataset_sha256=dataset.dataset_sha256,
        metrics,
        gate=(;
            minimum_spike_auroc=config.minimum_spike_auroc,
            passed=gate_pass,
        ),
        config=(;
            epochs=config.epochs,
            batch=config.batch,
            window=min(config.window, size(dataset.input, 2)),
            steps_per_epoch=config.steps_per_epoch,
            learning_rate=config.learning_rate,
            seed=config.seed,
            free_rollout_epochs=config.free_rollout_epochs,
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
            teacher_forcing_to_free_rollout=true,
            soma_spike_is_sole_external_event=true,
        ),
        history,
        created_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
    )
    _atomic_jldsave(config.output; payload)
    artifact_hash = artifact_sha256(config.output)
    report = (;
        schema=payload.schema,
        output=config.output,
        artifact_sha256=artifact_hash,
        parameter_sha256=payload.parameter_sha256,
        teacher_sha256=teacher_hash,
        provenance,
        frozen_internal=true,
        state_count=DISTILLED_STATE_DIM,
        metrics,
        gate=payload.gate,
        history,
    )
    _atomic_json(config.metrics, report)
    @info "distilled eleven-state artifact saved" output=config.output artifact_hash gate_pass metrics.test

    if !gate_pass && !config.allow_below_gate
        error(
            @sprintf(
                "held-out spike AUROC %.6f is below required %.6f; " *
                "candidate artifact and metrics were retained",
                metrics.test.spike_auroc,
                config.minimum_spike_auroc,
            ),
        )
    end
    return report
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_distillation(_parse_arguments(ARGS))
end
