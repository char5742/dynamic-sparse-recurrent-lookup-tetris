module DistillationDatasetBridgeStreamingV5

using Dates
using JLD2
using JSON3
using NPZ
using SHA

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(_PARENT_MODULE, :DistillationDatasetBridgeProductionV4)
    Base.include(
        _PARENT_MODULE,
        joinpath(
            @__DIR__,
            "prepare_distillation_dataset_production_v4.jl",
        ),
    )
end

using ..DistillationDatasetBridgeProductionV4
using ..PaperDigitalTwin

const BaseBridge = DistillationDatasetBridgeProductionV4

export SHARDED_DATASET_SCHEMA,
    SHARD_SCHEMA,
    StreamingPrepareConfig,
    prepare_distillation_dataset_streaming,
    main

const SHARDED_DATASET_SCHEMA =
    "hd-swsnn-twinprop-distillation-dataset-sharded-v1"
const SHARD_SCHEMA =
    "hd-swsnn-twinprop-distillation-shard-v1"
const _NUMERIC_NPZ_FIELDS = [
    "sample_indices",
    "split_code",
    "contact_axon",
    "contact_segment",
    "contact_kind",
    "contact_strength",
    "axon_kind",
    "event_trial_offset",
    "event_axon",
    "event_time_bin",
    "diagnostic_segment_indices",
    "time_ms",
    "event_spike",
    "axon_event_spike",
    "target_voltage",
    "target_spike",
    "target_nmda",
    "target_compartment_voltage",
    "target_compartment_nmda",
    "target_dendritic_cai",
    "target_dendritic_ica",
    "target_ca_event",
]

Base.@kwdef struct StreamingPrepareConfig
    dataset_path::String
    frozen_twin_path::String
    output_directory::String
    source_kind::Symbol = :official_neuron
    maximum_train_samples::Int = typemax(Int)
    maximum_validation_samples::Int = typemax(Int)
    maximum_test_samples::Int = typemax(Int)
    time_chunk::Int = 256
    output_shard_samples::Int = 1
    minimum_twin_spike_auroc::Float64 = 0.985
    auroc_histogram_bins::Int = 16_384
    expected_source_dataset_sha256::String = ""
    expected_modeldb_source_sha256::String = ""
    expected_detailed_teacher_sha256::String = ""
    expected_detailed_kernel_sha256::String = ""
    expected_morphology_sha256::String = ""
    expected_twin_parameter_sha256::String = ""
    expected_twin_artifact_sha256::String = ""
    selected_dendritic_segments::Vector{Int} = Int[]
end

function _base_config(config::StreamingPrepareConfig)
    return BaseBridge.PrepareDistillationConfig(
        dataset_path=config.dataset_path,
        frozen_twin_path=config.frozen_twin_path,
        output_path="unused-by-streaming-bridge",
        source_kind=config.source_kind,
        maximum_train_samples=config.maximum_train_samples,
        maximum_validation_samples=config.maximum_validation_samples,
        maximum_test_samples=config.maximum_test_samples,
        twin_batch_size=1,
        minimum_twin_spike_auroc=config.minimum_twin_spike_auroc,
        expected_source_dataset_sha256=
            config.expected_source_dataset_sha256,
        expected_modeldb_source_sha256=
            config.expected_modeldb_source_sha256,
        expected_detailed_teacher_sha256=
            config.expected_detailed_teacher_sha256,
        expected_detailed_kernel_sha256=
            config.expected_detailed_kernel_sha256,
        expected_morphology_sha256=
            config.expected_morphology_sha256,
        expected_twin_parameter_sha256=
            config.expected_twin_parameter_sha256,
        expected_twin_artifact_sha256=
            config.expected_twin_artifact_sha256,
        selected_dendritic_segments=
            config.selected_dendritic_segments,
    )
end

function _read_numeric_shard(path::AbstractString)
    extension = lowercase(splitext(path)[2])
    extension == ".npz" &&
        return NPZ.npzread(path, _NUMERIC_NPZ_FIELDS)
    extension == ".jld2" &&
        return BaseBridge._unwrap(JLD2.load(path))
    error("unsupported teacher shard extension $extension")
end

function _stream_plan(config, source)
    limits = Dict(
        BaseBridge.TRAIN_SPLIT => config.maximum_train_samples,
        BaseBridge.VALIDATION_SPLIT => config.maximum_validation_samples,
        BaseBridge.TEST_SPLIT => config.maximum_test_samples,
    )
    used = Dict(
        BaseBridge.TRAIN_SPLIT => 0,
        BaseBridge.VALIDATION_SPLIT => 0,
        BaseBridge.TEST_SPLIT => 0,
    )
    plan = NamedTuple[]
    common_time = 0
    for path in source.shard_paths
        shard = _read_numeric_shard(path)
        split_code = UInt8.(vec(BaseBridge._value(
            shard,
            :split_code,
        )))
        source_ids = Int32.(vec(BaseBridge._value(
            shard,
            :sample_indices,
            collect(1:length(split_code)),
        )))
        selected = Int[]
        selected_code = UInt8[]
        selected_id = Int32[]
        for (local_index, code) in enumerate(split_code)
            haskey(limits, code) ||
                error("teacher shard has an unknown split code")
            used[code] >= limits[code] && continue
            used[code] += 1
            push!(selected, local_index)
            push!(selected_code, code)
            push!(selected_id, source_ids[local_index])
        end
        isempty(selected) && continue
        voltage = BaseBridge._value(shard, :target_voltage)
        voltage === nothing &&
            error("teacher shard has no target_voltage")
        time_steps = size(voltage, 1)
        common_time == 0 && (common_time = time_steps)
        time_steps == common_time ||
            error("teacher shards have different time lengths")
        push!(
            plan,
            (;
                path,
                selected,
                split_code=selected_code,
                source_ids=selected_id,
            ),
        )
    end
    isempty(plan) && error("sample limits selected no teacher trials")
    for code in (
        BaseBridge.TRAIN_SPLIT,
        BaseBridge.VALIDATION_SPLIT,
        BaseBridge.TEST_SPLIT,
    )
        used[code] > 0 ||
            error("streaming dataset needs train/validation/test trials")
    end
    return plan, common_time, used
end

struct _SparseSample
    contact_axon::Vector{Int32}
    contact_segment::Vector{Int32}
    contact_kind::Vector{UInt8}
    contact_strength::Vector{Float32}
    event_axon::Vector{Int32}
    event_time_bin::Vector{Int32}
    static_strength::Vector{Float32}
    contacts_by_axon::Dict{Int,Vector{Int}}
    axons_by_time::Vector{Vector{Int}}
end

@inline function _feature_index(
    segments::Int,
    segment::Int,
    receptor::Int,
    plane::Int,
)
    return segment +
        segments * ((receptor - 1) + 3 * (plane - 1))
end

function _sparse_sample(shard, trial::Int, twin_config)
    contact_axon, contact_segment, contact_kind, contact_strength =
        BaseBridge._contact_arrays(shard)
    axon = Int32.(vec(@view contact_axon[:, trial]))
    segment = Int32.(vec(@view contact_segment[:, trial]))
    kind = UInt8.(vec(@view contact_kind[:, trial]))
    strength = Float32.(vec(@view contact_strength[:, trial]))
    offsets = BaseBridge._value(shard, :event_trial_offset)
    all_event_axon = BaseBridge._value(shard, :event_axon)
    all_event_time = BaseBridge._value(shard, :event_time_bin)
    if offsets === nothing ||
       all_event_axon === nothing ||
       all_event_time === nothing
        error(
            "streaming schema requires compact event_trial_offset, " *
            "event_axon and event_time_bin",
        )
    end
    first_event = Int(offsets[trial]) + 1
    last_event = Int(offsets[trial + 1])
    event_axon = Int32.(all_event_axon[first_event:last_event])
    event_time = Int32.(all_event_time[first_event:last_event])
    time_steps = size(BaseBridge._value(shard, :target_voltage), 1)
    static_strength = zeros(Float32, twin_config.input_dim)
    contacts_by_axon = Dict{Int,Vector{Int}}()
    @inbounds for contact in eachindex(axon)
        destination = Int(segment[contact])
        1 <= destination <= twin_config.segments ||
            error("contact targets a segment absent from the frozen twin")
        receptors = kind[contact] == UInt8(1) ? (1, 2) : (3,)
        for receptor in receptors
            feature = _feature_index(
                twin_config.segments,
                destination,
                receptor,
                2,
            )
            static_strength[feature] += strength[contact]
        end
        push!(
            get!(contacts_by_axon, Int(axon[contact]), Int[]),
            contact,
        )
    end
    static_strength .= min.(static_strength, 1.0f0)
    axons_by_time = [Int[] for _ in 1:time_steps]
    @inbounds for event in eachindex(event_axon)
        time = Int(event_time[event]) + 1
        1 <= time <= time_steps ||
            error("compact event time lies outside trajectory")
        push!(axons_by_time[time], Int(event_axon[event]))
    end
    return _SparseSample(
        axon,
        segment,
        kind,
        strength,
        event_axon,
        event_time,
        static_strength,
        contacts_by_axon,
        axons_by_time,
    )
end

function _dense_time_chunk(
    sample::_SparseSample,
    twin_config,
    first_time::Int,
    last_time::Int,
)
    count = last_time - first_time + 1
    input = zeros(Float32, twin_config.input_dim, count, 1)
    event_features = twin_config.segments * 3
    @inbounds for local_time in 1:count
        global_time = first_time + local_time - 1
        copyto!(
            @view(input[(event_features + 1):end, local_time, 1]),
            @view(sample.static_strength[(event_features + 1):end]),
        )
        for axon in sample.axons_by_time[global_time]
            contacts = get(sample.contacts_by_axon, axon, Int[])
            for contact in contacts
                receptors = sample.contact_kind[contact] == UInt8(1) ?
                    (1, 2) : (3,)
                for receptor in receptors
                    feature = _feature_index(
                        twin_config.segments,
                        Int(sample.contact_segment[contact]),
                        receptor,
                        1,
                    )
                    input[feature, local_time, 1] +=
                        sample.contact_strength[contact]
                end
            end
        end
    end
    @views input[1:event_features, :, :] .=
        min.(input[1:event_features, :, :], 1.0f0)
    all(isfinite, input) ||
        error("stream-expanded twin input is non-finite")
    return input
end

function _infer_sample(frozen, sample, time_steps, chunk)
    voltage = Vector{Float32}(undef, time_steps)
    spike = Vector{Float32}(undef, time_steps)
    spike_logit = Vector{Float32}(undef, time_steps)
    nmda = Matrix{Float32}(
        undef,
        frozen.model.config.nmda_regions,
        time_steps,
    )
    memory = nothing
    for first_time in 1:chunk:time_steps
        last_time = min(first_time + chunk - 1, time_steps)
        input = _dense_time_chunk(
            sample,
            frozen.model.config,
            first_time,
            last_time,
        )
        prediction = twin_forward(
            frozen,
            input;
            initial_memory=memory,
        )
        all(isfinite, prediction.voltage) ||
            error("frozen twin produced non-finite voltage")
        all(isfinite, prediction.spike_probability) ||
            error("frozen twin produced non-finite spike probability")
        all(isfinite, prediction.nmda) ||
            error("frozen twin produced non-finite NMDA")
        voltage[first_time:last_time] .=
            vec(prediction.voltage)
        spike[first_time:last_time] .=
            vec(prediction.spike_probability)
        spike_logit[first_time:last_time] .=
            vec(prediction.spike_logit)
        nmda[:, first_time:last_time] .=
            reshape(
                prediction.nmda,
                frozen.model.config.nmda_regions,
                :,
            )
        memory = prediction.final_memory
    end
    return (; voltage, spike, spike_logit, nmda)
end

mutable struct _StreamingGate
    voltage_count::Int64
    voltage_error2::Float64
    voltage_sum_x::Float64
    voltage_sum_y::Float64
    voltage_sum_x2::Float64
    voltage_sum_y2::Float64
    voltage_sum_xy::Float64
    nmda_count::Int64
    nmda_error2::Float64
    nmda_sum_x::Float64
    nmda_sum_y::Float64
    nmda_sum_x2::Float64
    nmda_sum_y2::Float64
    nmda_sum_xy::Float64
    spike_bce::Float64
    spike_correct::Int64
    spike_count::Int64
    positive_histogram::Vector{Int64}
    negative_histogram::Vector{Int64}
    held_out_samples::Int
end

function _StreamingGate(bins::Int)
    bins >= 256 || throw(ArgumentError("AUROC bins must be >= 256"))
    return _StreamingGate(
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        zeros(Int64, bins),
        zeros(Int64, bins),
        0,
    )
end

function _update_moments!(
    gate,
    prefix::Symbol,
    prediction,
    target,
)
    count_field = Symbol(prefix, :_count)
    error_field = Symbol(prefix, :_error2)
    sum_x_field = Symbol(prefix, :_sum_x)
    sum_y_field = Symbol(prefix, :_sum_y)
    sum_x2_field = Symbol(prefix, :_sum_x2)
    sum_y2_field = Symbol(prefix, :_sum_y2)
    sum_xy_field = Symbol(prefix, :_sum_xy)
    count = getfield(gate, count_field)
    error2 = getfield(gate, error_field)
    sum_x = getfield(gate, sum_x_field)
    sum_y = getfield(gate, sum_y_field)
    sum_x2 = getfield(gate, sum_x2_field)
    sum_y2 = getfield(gate, sum_y2_field)
    sum_xy = getfield(gate, sum_xy_field)
    for (x_raw, y_raw) in zip(prediction, target)
        x = Float64(x_raw)
        y = Float64(y_raw)
        isfinite(x) && isfinite(y) ||
            error("held-out metric contains non-finite values")
        difference = x - y
        count += 1
        error2 += difference * difference
        sum_x += x
        sum_y += y
        sum_x2 += x * x
        sum_y2 += y * y
        sum_xy += x * y
    end
    setfield!(gate, count_field, count)
    setfield!(gate, error_field, error2)
    setfield!(gate, sum_x_field, sum_x)
    setfield!(gate, sum_y_field, sum_y)
    setfield!(gate, sum_x2_field, sum_x2)
    setfield!(gate, sum_y2_field, sum_y2)
    setfield!(gate, sum_xy_field, sum_xy)
    return gate
end

function _update_gate!(
    gate::_StreamingGate,
    prediction,
    detailed_voltage,
    detailed_spike,
    detailed_nmda,
)
    _update_moments!(
        gate,
        :voltage,
        prediction.voltage,
        detailed_voltage,
    )
    _update_moments!(
        gate,
        :nmda,
        prediction.nmda,
        detailed_nmda,
    )
    bins = length(gate.positive_histogram)
    for (probability_raw, target_raw) in
        zip(prediction.spike, detailed_spike)
        probability = clamp(Float64(probability_raw), 1.0e-7, 1 - 1.0e-7)
        target = Float64(target_raw)
        bin = clamp(floor(Int, probability * bins) + 1, 1, bins)
        if target >= 0.5
            gate.positive_histogram[bin] += 1
        else
            gate.negative_histogram[bin] += 1
        end
        gate.spike_bce +=
            -target * log(probability) -
            (1 - target) * log1p(-probability)
        gate.spike_correct +=
            (probability >= 0.5) == (target >= 0.5)
        gate.spike_count += 1
    end
    gate.held_out_samples += 1
    return gate
end

function _correlation(
    count,
    sum_x,
    sum_y,
    sum_x2,
    sum_y2,
    sum_xy,
)
    count > 0 || return NaN
    covariance = sum_xy - sum_x * sum_y / count
    variance_x = sum_x2 - sum_x * sum_x / count
    variance_y = sum_y2 - sum_y * sum_y / count
    variance_x > 0 && variance_y > 0 || return NaN
    return covariance / sqrt(variance_x * variance_y)
end

function _finish_gate(gate::_StreamingGate)
    positives = sum(gate.positive_histogram)
    negatives = sum(gate.negative_histogram)
    positives > 0 ||
        error("held-out detailed target has no spike positives")
    negatives > 0 ||
        error("held-out detailed target has no spike negatives")
    lower_negative = Int64(0)
    wins = 0.0
    for bin in eachindex(gate.positive_histogram)
        positive = gate.positive_histogram[bin]
        negative = gate.negative_histogram[bin]
        wins += positive * (lower_negative + 0.5 * negative)
        lower_negative += negative
    end
    return (;
        source="stream-recomputed against official detailed teacher",
        self_report_trusted=false,
        auroc_histogram_bins=length(gate.positive_histogram),
        held_out_samples=gate.held_out_samples,
        held_out_bins=gate.spike_count,
        voltage_rmse=sqrt(
            gate.voltage_error2 / gate.voltage_count,
        ),
        voltage_correlation=_correlation(
            gate.voltage_count,
            gate.voltage_sum_x,
            gate.voltage_sum_y,
            gate.voltage_sum_x2,
            gate.voltage_sum_y2,
            gate.voltage_sum_xy,
        ),
        spike_auroc=wins / (positives * negatives),
        spike_bce=gate.spike_bce / gate.spike_count,
        spike_accuracy=gate.spike_correct / gate.spike_count,
        nmda_rmse=sqrt(gate.nmda_error2 / gate.nmda_count),
        nmda_correlation=_correlation(
            gate.nmda_count,
            gate.nmda_sum_x,
            gate.nmda_sum_y,
            gate.nmda_sum_x2,
            gate.nmda_sum_y2,
            gate.nmda_sum_xy,
        ),
    )
end

function _compact_slice(shard, selected)
    axon, segment, kind, strength =
        BaseBridge._contact_arrays(shard)
    contact_axon = Int32.(@view axon[:, selected])
    contact_segment = Int32.(@view segment[:, selected])
    contact_kind = UInt8.(@view kind[:, selected])
    contact_strength = Float32.(@view strength[:, selected])
    source_offsets = BaseBridge._value(shard, :event_trial_offset)
    source_event_axon = BaseBridge._value(shard, :event_axon)
    source_event_time = BaseBridge._value(shard, :event_time_bin)
    offsets = Int64[0]
    event_axon = Int32[]
    event_time_bin = Int32[]
    for trial in selected
        first_event = Int(source_offsets[trial]) + 1
        last_event = Int(source_offsets[trial + 1])
        append!(
            event_axon,
            Int32.(source_event_axon[first_event:last_event]),
        )
        append!(
            event_time_bin,
            Int32.(source_event_time[first_event:last_event]),
        )
        push!(offsets, length(event_axon))
    end
    return (;
        contact_axon,
        contact_segment,
        contact_kind,
        contact_strength,
        event_trial_offset=offsets,
        event_axon,
        event_time_bin,
    )
end

function _atomic_shard(path, dataset)
    temporary = path * ".tmp." * string(getpid())
    JLD2.jldsave(temporary; dataset)
    mv(temporary, path; force=true)
    return path
end

function _json_write(path, value)
    temporary = path * ".tmp." * string(getpid())
    open(temporary, "w") do stream
        JSON3.pretty(stream, value)
        write(stream, '\n')
    end
    mv(temporary, path; force=true)
    return path
end

function _config_hash(config)
    logical = (;
        schema=SHARDED_DATASET_SCHEMA,
        source_kind=String(config.source_kind),
        maximum_train_samples=config.maximum_train_samples,
        maximum_validation_samples=config.maximum_validation_samples,
        maximum_test_samples=config.maximum_test_samples,
        time_chunk=config.time_chunk,
        output_shard_samples=config.output_shard_samples,
        minimum_twin_spike_auroc=config.minimum_twin_spike_auroc,
        auroc_histogram_bins=config.auroc_histogram_bins,
        selected_dendritic_segments=config.selected_dendritic_segments,
    )
    return bytes2hex(SHA.sha256(codeunits(JSON3.write(logical))))
end

function prepare_distillation_dataset_streaming(
    config::StreamingPrepareConfig,
)
    config.time_chunk >= 1 ||
        throw(ArgumentError("time_chunk must be positive"))
    config.output_shard_samples >= 1 ||
        throw(ArgumentError("output_shard_samples must be positive"))
    destination = abspath(config.output_directory)
    ispath(destination) &&
        error("output directory already exists: $destination")
    staging = destination * ".staging." * string(getpid())
    ispath(staging) &&
        error("staging directory already exists: $staging")
    mkpath(staging)

    base_config = _base_config(config)
    source = BaseBridge._load_source(base_config)
    frozen, integrity_before, reported_metrics, twin_file_hash =
        BaseBridge._verify_twin(base_config, source)
    plan, time_steps, split_counts = _stream_plan(config, source)
    total_samples = sum(length(item.selected) for item in plan)
    config_hash = _config_hash(config)
    gate = _StreamingGate(config.auroc_histogram_bins)
    shard_records = NamedTuple[]
    output_shard_index = 0
    selected_segments = Int[]
    diagnostic_segments = Int[]
    global_output_index = 0

    for source_plan in plan
        source_shard = _read_numeric_shard(source_plan.path)
        for first_selected in
            1:config.output_shard_samples:length(source_plan.selected)
            last_selected = min(
                first_selected + config.output_shard_samples - 1,
                length(source_plan.selected),
            )
            group_range = first_selected:last_selected
            selected_trials = source_plan.selected[group_range]
            local_split = source_plan.split_code[group_range]
            local_source_ids = source_plan.source_ids[group_range]
            local_count = length(selected_trials)
            output_shard_index += 1
            voltage = Matrix{Float32}(undef, time_steps, local_count)
            spike = Matrix{Float32}(undef, time_steps, local_count)
            spike_logit =
                Matrix{Float32}(undef, time_steps, local_count)
            nmda = Array{Float32,3}(undef, 4, time_steps, local_count)
            local_selected, selected_rows, local_diagnostic =
                BaseBridge._diagnostic_selection(
                    base_config,
                    source,
                    source_shard,
                )
            if isempty(selected_segments)
                selected_segments = local_selected
                diagnostic_segments = local_diagnostic
            elseif selected_segments != local_selected ||
                   diagnostic_segments != local_diagnostic
                error("diagnostic segment mapping changed across shards")
            end
            calcium, dendritic =
                BaseBridge._detailed_internal_targets(
                    base_config,
                    source_shard,
                    selected_trials,
                    selected_segments,
                    selected_rows,
                )
            for (local_output, source_trial) in
                enumerate(selected_trials)
                sparse = _sparse_sample(
                    source_shard,
                    source_trial,
                    frozen.model.config,
                )
                prediction = _infer_sample(
                    frozen,
                    sparse,
                    time_steps,
                    config.time_chunk,
                )
                voltage[:, local_output] .= prediction.voltage
                spike[:, local_output] .= prediction.spike
                spike_logit[:, local_output] .=
                    prediction.spike_logit
                nmda[:, :, local_output] .= prediction.nmda
                if local_split[local_output] == BaseBridge.TEST_SPLIT
                    detailed_voltage = @view(
                        BaseBridge._value(
                            source_shard,
                            :target_voltage,
                        )[:, source_trial]
                    )
                    detailed_spike = @view(
                        BaseBridge._value(
                            source_shard,
                            :target_spike,
                        )[:, source_trial]
                    )
                    detailed_nmda = @view(
                        BaseBridge._value(
                            source_shard,
                            :target_nmda,
                        )[:, :, source_trial]
                    )
                    _update_gate!(
                        gate,
                        prediction,
                        detailed_voltage,
                        detailed_spike,
                        detailed_nmda,
                    )
                end
            end
            compact = _compact_slice(source_shard, selected_trials)
            first_global = global_output_index + 1
            global_output_index += local_count
            last_global = global_output_index
            shard_name = "distillation_shard_" *
                lpad(output_shard_index, 6, '0') * ".jld2"
            shard_path = joinpath(staging, shard_name)
            dataset = (;
                schema=SHARD_SCHEMA,
                input_representation=
                    "compact_contact_event_location_v1",
                input=nothing,
                compact...,
                target_voltage=voltage,
                target_spike=spike,
                target_spike_logit=spike_logit,
                target_nmda=nmda,
                target_calcium_event=calcium,
                target_dendritic_voltage=dendritic,
                split_code=local_split,
                source_sample_indices=local_source_ids,
                global_output_indices=collect(
                    Int32(first_global):Int32(last_global),
                ),
                diagnostic_segment_indices=diagnostic_segments,
                selected_dendritic_segments=selected_segments,
                mixed_supervision=true,
                digital_twin_gate_pending=true,
                frozen_twin_parameter_hash=
                    frozen.parameter_sha256,
                frozen_twin_artifact_hash=
                    frozen.artifact_sha256,
                detailed_teacher_hash=
                    source.detailed_teacher_hash,
                detailed_kernel_hash=
                    source.detailed_kernel_hash,
                morphology_hash=source.morphology_hash,
                official_modeldb_source_hash=source.modeldb_hash,
                dt_ms=frozen.model.config.dt_ms,
                time_steps,
                source_teacher_schema=source.source_schema,
                config_sha256=config_hash,
            )
            _atomic_shard(shard_path, dataset)
            shard_hash = BaseBridge._sha256_file(shard_path)
            push!(
                shard_records,
                (;
                    path=shard_name,
                    sha256=shard_hash,
                    bytes=filesize(shard_path),
                    samples=local_count,
                    global_first=first_global,
                    global_last=last_global,
                    split_counts=(;
                        train=count(
                            ==(BaseBridge.TRAIN_SPLIT),
                            local_split,
                        ),
                        validation=count(
                            ==(BaseBridge.VALIDATION_SPLIT),
                            local_split,
                        ),
                        test=count(
                            ==(BaseBridge.TEST_SPLIT),
                            local_split,
                        ),
                    ),
                ),
            )
        end
    end
    global_output_index == total_samples ||
        error("streaming sample count mismatch")
    recomputed_gate = _finish_gate(gate)
    recomputed_gate.spike_auroc >=
        config.minimum_twin_spike_auroc || error(
        "frozen twin failed stream-recomputed held-out gate: " *
        "spike AUROC $(recomputed_gate.spike_auroc) < " *
        "$(config.minimum_twin_spike_auroc). Staging directory was not " *
        "published.",
    )
    integrity_after = assert_frozen_unchanged(
        frozen;
        expected_artifact_sha256=integrity_before.artifact_sha256,
    )
    integrity_after.max_delta == 0 ||
        error("frozen twin changed during streaming preparation")
    segment_region =
        BaseBridge._segment_regions(source, config.source_kind)
    input_compartment, input_receptor, input_plane =
        BaseBridge._input_anatomy(frozen.model.config)
    hashes = (;
        official_modeldb_source_hash=source.modeldb_hash,
        detailed_teacher_hash=source.detailed_teacher_hash,
        detailed_kernel_hash=source.detailed_kernel_hash,
        morphology_hash=source.morphology_hash,
        frozen_twin_parameter_hash=frozen.parameter_sha256,
        frozen_twin_artifact_hash=frozen.artifact_sha256,
        frozen_twin_file_sha256=twin_file_hash,
        source_dataset_hash=source.source_dataset_hash,
        source_manifest_sha256=source.source_manifest_hash,
        config_sha256=config_hash,
    )
    manifest = (;
        schema=SHARDED_DATASET_SCHEMA,
        shard_schema=SHARD_SCHEMA,
        prepared_dataset_schema=
            BaseBridge.PREPARED_DATASET_SCHEMA,
        model_name=HD_SWSNN_TWINPROP_NAME,
        completion_state="complete",
        created_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
        source_kind=String(config.source_kind),
        source_teacher_schema=source.source_schema,
        official_neuron_schema=
            config.source_kind === :official_neuron ?
            BaseBridge.OFFICIAL_NEURON_SCHEMA : "",
        source_completion_state=source.completion_state,
        input_representation="compact_contact_event_location_v1",
        input_layout=twin_input_layout(frozen),
        input_compartment,
        input_receptor,
        input_plane,
        time_steps,
        total_samples,
        split_counts=(;
            train=split_counts[BaseBridge.TRAIN_SPLIT],
            validation=split_counts[BaseBridge.VALIDATION_SPLIT],
            test=split_counts[BaseBridge.TEST_SPLIT],
        ),
        time_chunk=config.time_chunk,
        output_shard_samples=config.output_shard_samples,
        peak_dense_chunk_bytes=
            frozen.model.config.input_dim *
            min(config.time_chunk, time_steps) *
            sizeof(Float32),
        dense_memory_scales_with_total_samples=false,
        mixed_supervision=true,
        mixed_supervision_provenance=(;
            target_voltage="actual frozen PaperDigitalTwin inference",
            target_spike="actual frozen PaperDigitalTwin probability",
            target_spike_logit="actual frozen PaperDigitalTwin logit",
            target_nmda="actual frozen PaperDigitalTwin inference",
            target_calcium_event="detailed teacher only",
            target_dendritic_voltage="detailed teacher only",
        ),
        digital_twin_gate_passed=true,
        twin_self_report_trusted=false,
        twin_artifact_reported_metrics=reported_metrics,
        recomputed_twin_gate,
        integrity_before,
        integrity_after,
        diagnostic_segment_indices=diagnostic_segments,
        selected_dendritic_segments=selected_segments,
        selected_dendritic_semantics=(
            "distal_basal",
            "proximal_apical_trunk",
            "apical_calcium_hot_zone",
            "distal_apical_tuft",
        ),
        segment_region,
        teacher_hash=source.detailed_teacher_hash,
        detailed_teacher_hash=source.detailed_teacher_hash,
        detailed_kernel_hash=source.detailed_kernel_hash,
        cell_mechanism_sha256=source.detailed_kernel_hash,
        morphology_hash=source.morphology_hash,
        official_modeldb_source_hash=source.modeldb_hash,
        frozen_twin_parameter_hash=frozen.parameter_sha256,
        frozen_twin_artifact_hash=frozen.artifact_sha256,
        digital_twin_hash=frozen.artifact_sha256,
        source_dataset_hash=source.source_dataset_hash,
        source_manifest_sha256=source.source_manifest_hash,
        config_sha256=config_hash,
        hashes,
        shards=shard_records,
    )
    manifest_path = joinpath(staging, "manifest.json")
    _json_write(manifest_path, manifest)
    manifest_hash = BaseBridge._sha256_file(manifest_path)
    mv(staging, destination)
    return (;
        schema=SHARDED_DATASET_SCHEMA,
        output_directory=destination,
        manifest_path=joinpath(destination, "manifest.json"),
        manifest_sha256=manifest_hash,
        total_samples,
        shard_count=length(shard_records),
        split_counts=manifest.split_counts,
        peak_dense_chunk_bytes=manifest.peak_dense_chunk_bytes,
        dense_memory_scales_with_total_samples=false,
        recomputed_twin_gate,
        digital_twin_gate_passed=true,
        frozen_max_delta_before=integrity_before.max_delta,
        frozen_max_delta_after=integrity_after.max_delta,
    )
end

function _parse_arguments(arguments)
    options = Dict{String,String}(
        "dataset" => get(ENV, "HD_TWINPROP_TEACHER_PATH", ""),
        "frozen-twin" => get(ENV, "HD_TWINPROP_TWIN_PATH", ""),
        "output-directory" => get(
            ENV,
            "HD_TWINPROP_DISTILL_DATASET_DIR",
            joinpath(@__DIR__, "artifacts", "distillation_dataset_shards"),
        ),
        "source-kind" => get(
            ENV,
            "HD_TWINPROP_SOURCE_KIND",
            "official-neuron",
        ),
        "max-train" => get(ENV, "HD_TWINPROP_MAX_TRAIN", string(typemax(Int))),
        "max-validation" => get(
            ENV,
            "HD_TWINPROP_MAX_VALIDATION",
            string(typemax(Int)),
        ),
        "max-test" => get(ENV, "HD_TWINPROP_MAX_TEST", string(typemax(Int))),
        "time-chunk" => get(ENV, "HD_TWINPROP_TIME_CHUNK", "256"),
        "output-shard-samples" => get(
            ENV,
            "HD_TWINPROP_OUTPUT_SHARD_SAMPLES",
            "1",
        ),
        "minimum-twin-spike-auroc" => get(
            ENV,
            "HD_TWINPROP_MIN_TWIN_AUROC",
            "0.985",
        ),
        "auroc-histogram-bins" => get(
            ENV,
            "HD_TWINPROP_AUROC_BINS",
            "16384",
        ),
    )
    index = 1
    while index <= length(arguments)
        key = replace(arguments[index], r"^--" => "")
        haskey(options, key) || error("unknown option --$key")
        index == length(arguments) &&
            error("missing value for --$key")
        options[key] = arguments[index + 1]
        index += 2
    end
    isempty(options["dataset"]) &&
        error("--dataset or HD_TWINPROP_TEACHER_PATH is required")
    isempty(options["frozen-twin"]) &&
        error("--frozen-twin or HD_TWINPROP_TWIN_PATH is required")
    return StreamingPrepareConfig(
        dataset_path=abspath(options["dataset"]),
        frozen_twin_path=abspath(options["frozen-twin"]),
        output_directory=abspath(options["output-directory"]),
        source_kind=Symbol(replace(
            lowercase(options["source-kind"]),
            "-" => "_",
        )),
        maximum_train_samples=parse(Int, options["max-train"]),
        maximum_validation_samples=
            parse(Int, options["max-validation"]),
        maximum_test_samples=parse(Int, options["max-test"]),
        time_chunk=parse(Int, options["time-chunk"]),
        output_shard_samples=parse(
            Int,
            options["output-shard-samples"],
        ),
        minimum_twin_spike_auroc=parse(
            Float64,
            options["minimum-twin-spike-auroc"],
        ),
        auroc_histogram_bins=parse(
            Int,
            options["auroc-histogram-bins"],
        ),
    )
end

function main(arguments=ARGS)
    report = prepare_distillation_dataset_streaming(
        _parse_arguments(arguments),
    )
    println(JSON3.write(report))
    return report
end

end # module DistillationDatasetBridgeStreamingV5

if abspath(PROGRAM_FILE) == @__FILE__
    DistillationDatasetBridgeStreamingV5.main()
end
