module DigitalTwinTraining

using Dates
using JLD2
using JSON3
using Lux
using Optimisers
using Random
using Statistics
using Zygote

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(_PARENT_MODULE, :PaperDigitalTwin)
    Base.include(_PARENT_MODULE, joinpath(@__DIR__, "PaperDigitalTwin.jl"))
end
if !isdefined(_PARENT_MODULE, :TwinDatasetGeneration)
    Base.include(_PARENT_MODULE, joinpath(@__DIR__, "generate_twin_dataset.jl"))
end

using ..PaperDigitalTwin
using ..TwinDatasetGeneration: expand_compact_twin_input

export TwinLossWeights,
    twin_objective,
    fit_streaming_normalizer,
    evaluate_twin_split,
    train_digital_twin,
    main

struct TwinLossWeights
    voltage::Float32
    spike::Float32
    nmda::Float32
    huber_delta::Float32
end

TwinLossWeights(;
    voltage::Real=1,
    spike::Real=1,
    nmda::Real=1,
    huber_delta::Real=1,
) = TwinLossWeights(
    Float32(voltage),
    Float32(spike),
    Float32(nmda),
    Float32(huber_delta),
)

@inline _softplus(x) = max(x, zero(x)) + log1p(exp(-abs(x)))

function _huber_mean(error, delta::Float32)
    absolute = abs.(error)
    return mean(ifelse.(
        absolute .<= delta,
        0.5f0 .* error .* error,
        delta .* (absolute .- 0.5f0 * delta),
    ))
end

function _normalized_targets(normalizer::TwinNormalizer, batch)
    voltage =
        (batch.target_voltage .- normalizer.voltage_mean) ./
        normalizer.voltage_scale
    nmda =
        (
            batch.target_nmda .-
            reshape(normalizer.nmda_mean, :, 1, 1)
        ) ./ reshape(normalizer.nmda_scale, :, 1, 1)
    return voltage, nmda
end

"""
Joint voltage/spike/NMDA Step-1 objective.

Voltage uses Huber loss, spike timing uses numerically stable BCE with a
training-split positive-class weight, and compartment-region NMDA current
uses normalized MSE.  No task label or parity label is visible here.
"""
function twin_objective(
    model::PaperTwin,
    parameters,
    normalizer::TwinNormalizer,
    batch,
    weights::TwinLossWeights,
    positive_weight::Float32,
)
    input = normalize_twin_input(normalizer, batch.input)
    target_voltage, target_nmda =
        _normalized_targets(normalizer, batch)
    prediction = twin_forward(model, parameters, input)
    voltage_loss = _huber_mean(
        prediction.voltage .- target_voltage,
        weights.huber_delta,
    )
    target_spike = batch.target_spike
    spike_element =
        max.(prediction.spike_logit, 0.0f0) .-
        prediction.spike_logit .* target_spike .+
        log1p.(exp.(-abs.(prediction.spike_logit)))
    spike_weight = ifelse.(
        target_spike .>= 0.5f0,
        positive_weight,
        1.0f0,
    )
    spike_loss = sum(spike_weight .* spike_element) / sum(spike_weight)
    nmda_loss = mean(abs2, prediction.nmda .- target_nmda)
    total =
        weights.voltage * voltage_loss +
        weights.spike * spike_loss +
        weights.nmda * nmda_loss
    return total, (;
        voltage=voltage_loss,
        spike=spike_loss,
        nmda=nmda_loss,
    )
end

function _read_manifest(dataset_root::AbstractString)
    path = joinpath(abspath(dataset_root), "manifest.json")
    isfile(path) || error("missing twin dataset manifest: $path")
    return JSON3.read(read(path, String)), path
end

function _shard_paths(dataset_root::AbstractString, manifest)
    root = abspath(dataset_root)
    return [joinpath(root, String(record.path)) for record in manifest.shards]
end

function _load_dense_input(data)
    stored = get(data, "input", nothing)
    stored !== nothing && return Float32.(stored)
    return expand_compact_twin_input(
        data["contact_segment"],
        data["contact_kind"],
        data["contact_strength"],
        data["event_spike"],
        data["twin_config"],
    )
end

@inline function _split_code(split::Symbol)
    split === :train && return UInt8(1)
    split === :validation && return UInt8(2)
    split === :test && return UInt8(3)
    throw(ArgumentError("split must be :train, :validation or :test"))
end

function _split_indices(data, split::Symbol)
    code = _split_code(split)
    return findall(==(code), data["split_code"])
end

mutable struct _StreamingMoments
    count::Int64
    sum::Vector{Float64}
    sumsq::Vector{Float64}
end

_StreamingMoments(dim::Int) =
    _StreamingMoments(0, zeros(Float64, dim), zeros(Float64, dim))

function _update_moments!(
    moments::_StreamingMoments,
    values::AbstractArray,
)
    size(values, 1) == length(moments.sum) ||
        throw(DimensionMismatch("streaming moment dimension mismatch"))
    flattened = reshape(values, size(values, 1), :)
    moments.count += size(flattened, 2)
    moments.sum .+= vec(sum(Float64.(flattened); dims=2))
    moments.sumsq .+= vec(sum(abs2, Float64.(flattened); dims=2))
    return moments
end

function _finish_moments(moments::_StreamingMoments; epsilon=1.0f-5)
    moments.count > 0 || error("cannot finish empty streaming moments")
    mean_value = moments.sum ./ moments.count
    variance = max.(
        moments.sumsq ./ moments.count .- mean_value .* mean_value,
        0.0,
    )
    scale = max.(sqrt.(variance), Float64(epsilon))
    return Float32.(mean_value), Float32.(scale)
end

"""
Fit all normalizers and the spike class weight by streaming one shard at a
time.  Test split is never read.
"""
function fit_streaming_normalizer(
    dataset_root::AbstractString,
    manifest,
)
    paths = _shard_paths(dataset_root, manifest)
    isempty(paths) && error("dataset has no shards")
    first_data = JLD2.load(first(paths))
    twin_config = first_data["twin_config"]
    input_moments = _StreamingMoments(twin_config.input_dim)
    voltage_moments = _StreamingMoments(1)
    nmda_moments = _StreamingMoments(twin_config.nmda_regions)
    spike_positive = 0
    spike_total = 0
    for path in paths
        data = JLD2.load(path)
        indices = _split_indices(data, :train)
        isempty(indices) && continue
        input = _load_dense_input(data)
        _update_moments!(input_moments, @view(input[:, :, indices]))
        voltage = reshape(
            @view(data["target_voltage"][:, indices]),
            1,
            :,
        )
        _update_moments!(voltage_moments, voltage)
        _update_moments!(
            nmda_moments,
            @view(data["target_nmda"][:, :, indices]),
        )
        spike = @view data["target_spike"][:, indices]
        spike_positive += count(>=(0.5f0), spike)
        spike_total += length(spike)
    end
    input_mean, input_scale = _finish_moments(input_moments)
    voltage_mean_vector, voltage_scale_vector =
        _finish_moments(voltage_moments)
    nmda_mean, nmda_scale = _finish_moments(nmda_moments)
    spike_positive > 0 ||
        error("training split contains no soma spikes; regenerate protocol")
    spike_negative = spike_total - spike_positive
    positive_weight = Float32(clamp(
        spike_negative / spike_positive,
        1.0,
        100.0,
    ))
    normalizer = TwinNormalizer(
        input_mean,
        input_scale,
        only(voltage_mean_vector),
        only(voltage_scale_vector),
        nmda_mean,
        nmda_scale,
    )
    return normalizer, positive_weight, (;
        spike_positive,
        spike_negative,
        spike_positive_fraction=spike_positive / spike_total,
    )
end

function _batch_from_data(data, input, indices)
    return (;
        input=Float32.(@view(input[:, :, indices])),
        target_voltage=
            Float32.(@view(data["target_voltage"][:, indices])),
        target_spike=
            Float32.(@view(data["target_spike"][:, indices])),
        target_nmda=
            Float32.(@view(data["target_nmda"][:, :, indices])),
    )
end

function _predict_physical(model, parameters, normalizer, input)
    normalized = normalize_twin_input(normalizer, input)
    raw = twin_forward(model, parameters, normalized)
    return denormalize_twin_output(normalizer, raw)
end

function evaluate_twin_split(
    dataset_root::AbstractString,
    manifest,
    model::PaperTwin,
    parameters,
    normalizer::TwinNormalizer,
    split::Symbol;
    maximum_samples::Integer=256,
)
    voltage_predictions = Matrix{Float32}[]
    voltage_targets = Matrix{Float32}[]
    spike_predictions = Matrix{Float32}[]
    spike_targets = Matrix{Float32}[]
    nmda_predictions = Array{Float32,3}[]
    nmda_targets = Array{Float32,3}[]
    samples = 0
    for path in _shard_paths(dataset_root, manifest)
        samples >= maximum_samples && break
        data = JLD2.load(path)
        available = _split_indices(data, split)
        isempty(available) && continue
        take_count = min(length(available), maximum_samples - samples)
        indices = first(available, take_count)
        input = _load_dense_input(data)
        batch = _batch_from_data(data, input, indices)
        prediction = _predict_physical(
            model,
            parameters,
            normalizer,
            batch.input,
        )
        push!(voltage_predictions, Float32.(prediction.voltage))
        push!(voltage_targets, batch.target_voltage)
        push!(
            spike_predictions,
            Float32.(prediction.spike_probability),
        )
        push!(spike_targets, batch.target_spike)
        push!(nmda_predictions, Float32.(prediction.nmda))
        push!(nmda_targets, batch.target_nmda)
        samples += take_count
    end
    samples > 0 || error("split $split has no samples")
    prediction = (;
        voltage=reduce(hcat, voltage_predictions),
        spike_probability=reduce(hcat, spike_predictions),
        nmda=cat(nmda_predictions...; dims=3),
    )
    target_voltage = reduce(hcat, voltage_targets)
    target_spike = reduce(hcat, spike_targets)
    target_nmda = cat(nmda_targets...; dims=3)
    metrics = twin_metrics(
        prediction,
        target_voltage,
        target_spike,
        target_nmda;
        normalizer,
    )
    return merge(metrics, (; split=String(split), samples))
end

function _training_shards(dataset_root, manifest)
    result = String[]
    for path in _shard_paths(dataset_root, manifest)
        data = JLD2.load(path)
        isempty(_split_indices(data, :train)) || push!(result, path)
    end
    isempty(result) && error("dataset has no training shards")
    return result
end

function train_digital_twin(
    dataset_root::AbstractString,
    checkpoint_path::AbstractString;
    updates::Integer=200,
    batch_size::Integer=8,
    learning_rate::Real=3.0f-4,
    weight_decay::Real=1.0f-5,
    weights::TwinLossWeights=TwinLossWeights(),
    seed::Integer=0x5457494e54524149,
    log_interval::Integer=25,
    evaluation_samples::Integer=256,
)
    updates >= 1 || throw(ArgumentError("updates must be positive"))
    batch_size >= 1 || throw(ArgumentError("batch_size must be positive"))
    manifest, manifest_path = _read_manifest(dataset_root)
    shard_paths = _shard_paths(dataset_root, manifest)
    first_data = JLD2.load(first(shard_paths))
    dataset_config = first_data["twin_config"]
    config = TwinConfig(
        segments=dataset_config.segments,
        nmda_regions=dataset_config.nmda_regions,
        memory_units=1_000,
        core_dim=128,
        dt_ms=dataset_config.dt_ms,
        tau_min_ms=0.1,
        tau_max_ms=300,
        bank_seed=dataset_config.bank_seed,
    )
    model = build_paper_twin(config)
    normalizer, positive_weight, spike_statistics =
        fit_streaming_normalizer(dataset_root, manifest)
    rng = Xoshiro(UInt64(seed))
    parameters, _ = Lux.setup(rng, model)
    optimizer = Optimisers.AdamW(
        Float32(learning_rate),
        (0.9, 0.999),
        Float32(weight_decay),
    )
    optimizer_state = Optimisers.setup(optimizer, parameters)
    train_shards = _training_shards(dataset_root, manifest)

    initial_validation = evaluate_twin_split(
        dataset_root,
        manifest,
        model,
        parameters,
        normalizer,
        :validation;
        maximum_samples=evaluation_samples,
    )
    losses = Float64[]
    component_history = NamedTuple[]
    started = time()
    shard_cursor = 0
    current_path = ""
    current_data = nothing
    current_input = nothing
    current_train_indices = Int[]
    for update in 1:Int(updates)
        # Keep one shard resident for several consecutive minibatches.  This
        # is fixed-memory and avoids production-scale dataset materialization.
        if current_data === nothing ||
           isempty(current_train_indices) ||
           mod(update - 1, 4) == 0
            shard_cursor = mod1(shard_cursor + 1, length(train_shards))
            current_path = train_shards[shard_cursor]
            current_data = JLD2.load(current_path)
            current_input = _load_dense_input(current_data)
            current_train_indices =
                _split_indices(current_data, :train)
        end
        local_batch = min(Int(batch_size), length(current_train_indices))
        chosen = rand(rng, current_train_indices, local_batch)
        batch = _batch_from_data(
            current_data,
            current_input,
            chosen,
        )
        loss, pullback = Zygote.pullback(parameters) do candidate
            objective, _ = twin_objective(
                model,
                candidate,
                normalizer,
                batch,
                weights,
                positive_weight,
            )
            return objective
        end
        gradient = only(pullback(one(loss)))
        isfinite(loss) ||
            error("non-finite digital twin loss at update $update")
        optimizer_state, parameters = Optimisers.update(
            optimizer_state,
            parameters,
            gradient,
        )
        _, components = twin_objective(
            model,
            parameters,
            normalizer,
            batch,
            weights,
            positive_weight,
        )
        push!(losses, Float64(loss))
        push!(component_history, components)
        if update == 1 ||
           update % Int(log_interval) == 0 ||
           update == updates
            @info "HD-SWSNN-TwinProp Step-1 update" update loss components elapsed_seconds=time()-started shard=current_path
        end
    end

    final_validation = evaluate_twin_split(
        dataset_root,
        manifest,
        model,
        parameters,
        normalizer,
        :validation;
        maximum_samples=evaluation_samples,
    )
    held_out_test = evaluate_twin_split(
        dataset_root,
        manifest,
        model,
        parameters,
        normalizer,
        :test;
        maximum_samples=evaluation_samples,
    )
    teacher_hash = String(manifest.teacher_hash)
    cell_mechanism_sha256 = String(manifest.cell_mechanism_sha256)
    morphology_sha256 = String(manifest.morphology_sha256)
    frozen = freeze_twin(
        model,
        parameters,
        normalizer;
        metadata=(;
            generated_at=string(now()),
            official_model_name=HD_SWSNN_TWINPROP_NAME,
            dataset_manifest=manifest_path,
            teacher_hash,
            detailed_teacher_hash=teacher_hash,
            cell_mechanism_sha256,
            morphology_sha256,
            dt_ms=config.dt_ms,
            training_updates=Int(updates),
            training_backend="Lux+Zygote+Optimisers",
            loss_targets=(
                "soma_voltage",
                "soma_spike",
                "region_nmda_current",
            ),
            held_out_test,
            source_implementation=
                "deterministic CPU reconstruction; not author code",
            frozen_internal=true,
        ),
    )
    save_frozen_twin(checkpoint_path, frozen)
    integrity = assert_frozen_unchanged(frozen)
    component_first = first(component_history)
    component_last = last(component_history)
    summary = (;
        schema_version=1,
        model_name=HD_SWSNN_TWINPROP_NAME,
        stage="detailed_to_digital_twin",
        generated_at=string(now()),
        dataset_manifest=manifest_path,
        checkpoint_path=abspath(checkpoint_path),
        teacher_hash,
        detailed_teacher_hash=teacher_hash,
        digital_twin_hash=frozen.artifact_sha256,
        frozen_artifact_sha256=frozen.artifact_sha256,
        frozen_parameter_sha256=frozen.parameter_sha256,
        frozen_internal=true,
        frozen_integrity=integrity,
        cell_mechanism_sha256,
        morphology_sha256,
        dt_ms=config.dt_ms,
        fixed_memory_units=config.memory_units,
        fixed_memory_tau_ms=(
            config.tau_min_ms,
            config.tau_max_ms,
        ),
        trainable_parameters=Lux.parameterlength(parameters),
        backend="Lux+Zygote+Optimisers",
        updates=Int(updates),
        batch_size=Int(batch_size),
        learning_rate=Float32(learning_rate),
        weight_decay=Float32(weight_decay),
        loss_weights=(;
            voltage=weights.voltage,
            spike=weights.spike,
            nmda=weights.nmda,
            huber_delta=weights.huber_delta,
        ),
        spike_positive_weight=positive_weight,
        spike_statistics,
        initial_validation,
        final_validation,
        held_out_test,
        loss_first=first(losses),
        loss_last=last(losses),
        loss_component_first=component_first,
        loss_component_last=component_last,
        wall_seconds=time() - started,
        public_paper_reference=(;
            memory_units=1_000,
            tau_ms=(0.1, 300.0),
            random_initializations=3,
            approximate_epochs=35,
            held_out_spike_auroc=0.98576,
        ),
        reconstruction_disclosure=
            "The authors' implementation is unavailable; fixed random map, optimizer schedule, and NMDA auxiliary target are this repository's CPU reconstruction.",
        public_paper_values_separated=true,
    )
    summary_path = replace(abspath(checkpoint_path), r"\.jld2$" => ".json")
    open(summary_path, "w") do io
        JSON3.pretty(io, summary)
    end
    return merge(summary, (; summary_path))
end

function main()
    preset = Symbol(lowercase(get(ENV, "TWIN_TRAIN_PRESET", "smoke")))
    default_updates = preset === :production ? 227_500 : 200
    default_batch = preset === :production ? 8 : 4
    dataset_root = abspath(get(
        ENV,
        "TWIN_DATASET_PATH",
        joinpath(
            @__DIR__,
            "artifacts",
            "twin_dataset_$(String(preset))",
        ),
    ))
    checkpoint_path = abspath(get(
        ENV,
        "TWIN_CHECKPOINT_PATH",
        joinpath(
            @__DIR__,
            "artifacts",
            "hd_swsnn_twinprop_$(String(preset)).jld2",
        ),
    ))
    result = train_digital_twin(
        dataset_root,
        checkpoint_path;
        updates=parse(Int, get(
            ENV,
            "TWIN_TRAIN_UPDATES",
            string(default_updates),
        )),
        batch_size=parse(Int, get(
            ENV,
            "TWIN_TRAIN_BATCH",
            string(default_batch),
        )),
        learning_rate=parse(
            Float32,
            get(ENV, "TWIN_TRAIN_LEARNING_RATE", "3e-4"),
        ),
        weight_decay=parse(
            Float32,
            get(ENV, "TWIN_TRAIN_WEIGHT_DECAY", "1e-5"),
        ),
        evaluation_samples=parse(
            Int,
            get(ENV, "TWIN_EVALUATION_SAMPLES", "256"),
        ),
    )
    println(JSON3.write(result))
    return result
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    DigitalTwinTraining.main()
end
