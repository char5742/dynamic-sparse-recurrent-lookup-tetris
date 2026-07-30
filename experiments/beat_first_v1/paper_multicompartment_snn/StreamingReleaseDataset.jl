module StreamingReleaseDataset

using JLD2
using JSON3
using Serialization
using SHA
using Statistics

export RELEASE_DATASET_SCHEMA,
    RELEASE_SHARD_SCHEMA,
    StreamDataset,
    StreamMemoryTracker,
    StreamShardRecord,
    open_stream_dataset,
    stream_dataset_integrity!,
    stream_materialize_window,
    stream_target_statistics

const RELEASE_DATASET_SCHEMA =
    "hd-swsnn-twinprop-distillation-dataset-sharded-release-v2"
const RELEASE_SHARD_SCHEMA =
    "hd-swsnn-twinprop-distillation-shard-release-v2"
const OFFICIAL_TEACHER_SCHEMA =
    "hd_swsnn_twinprop.neuron_teacher.final.v2"
const OFFICIAL_SEGMENTS = 642
const TRAIN_SPLIT = UInt8(1)
const VALIDATION_SPLIT = UInt8(2)
const TEST_SPLIT = UInt8(3)

@inline _get(object, name::Symbol, default=nothing) =
    if object isa AbstractDict
        get(object, name, get(object, String(name), default))
    elseif hasproperty(object, name)
        getproperty(object, name)
    else
        default
    end

function _required(object, name::Symbol)
    value = _get(object, name, nothing)
    value === nothing &&
        error("required streaming field $(String(name)) is absent")
    return value
end

function _file_sha256(path::AbstractString)
    source = abspath(path)
    isfile(source) || error("streaming file is absent: $source")
    return open(source, "r") do stream
        bytes2hex(SHA.sha256(stream))
    end
end

function _value_sha256(value)
    stream = IOBuffer()
    Serialization.serialize(stream, value)
    return bytes2hex(SHA.sha256(take!(stream)))
end

function _require_sha256(value, label)
    text = lowercase(String(value))
    length(text) == 64 &&
        all(character -> isdigit(character) ||
            ('a' <= character <= 'f'), text) ||
        error("$label is not a SHA-256 digest")
    return text
end

function _manifest_path(path::AbstractString)
    source = abspath(path)
    if isdir(source)
        source = joinpath(source, "manifest.json")
    end
    isfile(source) || error("streaming manifest is absent: $source")
    return source
end

function _safe_shard_path(root::AbstractString, relative::AbstractString)
    isabspath(relative) &&
        error("stream shard paths must be relative")
    candidate = abspath(normpath(joinpath(root, relative)))
    root_absolute = abspath(root)
    prefix = lowercase(root_absolute * Base.Filesystem.path_separator)
    candidate_lower = lowercase(candidate)
    (
        candidate_lower == lowercase(root_absolute) ||
        startswith(candidate_lower, prefix)
    ) || error("stream shard escapes its manifest directory")
    isfile(candidate) || error("stream shard is absent: $candidate")
    return candidate
end

struct StreamShardRecord
    path::String
    relative_path::String
    sha256::String
    bytes::Int
    samples::Int
    global_first::Int
    global_last::Int
end

mutable struct StreamMemoryTracker
    peak_loaded_shard_bytes::Int
    peak_dense_window_bytes::Int
    peak_combined_bytes::Int
    windows_materialized::Int
    samples_materialized::Int
end

StreamMemoryTracker() = StreamMemoryTracker(0, 0, 0, 0, 0)

struct StreamDataset
    manifest_path::String
    root::String
    manifest_sha256::String
    dataset_sha256::String
    manifest::Any
    records::Vector{StreamShardRecord}
    total_samples::Int
    time_steps::Int
    input_dim::Int
    train_indices::Vector{Int}
    validation_indices::Vector{Int}
    test_indices::Vector{Int}
    split_code::Vector{UInt8}
    global_to_shard::Vector{Int32}
    diagnostic_time_indices::Vector{Int}
    segment_region::Vector{String}
    segment_catalog_sha256::String
    provenance::Any
    frozen_twin_file_sha256::String
    verified_shard_hashes::Bool
    tracker::StreamMemoryTracker
end

function _split_indices(manifest, name, total)
    values = Int.(collect(_required(manifest, name)))
    all(index -> 1 <= index <= total, values) ||
        error("$(String(name)) contains an out-of-range global index")
    length(unique(values)) == length(values) ||
        error("$(String(name)) contains duplicate global indices")
    return values
end

function _read_manifest(path)
    return JSON3.read(read(path, String), Dict{String,Any})
end

function _manifest_hash_value(manifest, names::Tuple)
    for name in names
        direct = _get(manifest, name, nothing)
        direct !== nothing &&
            !isempty(String(direct)) &&
            return _require_sha256(direct, String(name))
        hashes = _get(manifest, :hashes, nothing)
        hashes === nothing && continue
        nested = _get(hashes, name, nothing)
        nested !== nothing &&
            !isempty(String(nested)) &&
            return _require_sha256(nested, String(name))
    end
    error("manifest lacks lineage $(join(String.(names), '/'))")
end

function _maximum_delta(record)
    record === nothing && return Inf
    value = _get(record, :max_delta, nothing)
    value === nothing && return Inf
    return Float64(value)
end

function open_stream_dataset(
    path::AbstractString,
    frozen;
    minimum_spike_auroc::Real=0.985,
    verify_shard_hashes::Bool=true,
    require_promotion_eligible::Bool=true,
)
    manifest_path = _manifest_path(path)
    root = dirname(manifest_path)
    manifest_sha256 = _file_sha256(manifest_path)
    manifest = _read_manifest(manifest_path)
    String(_required(manifest, :schema)) == RELEASE_DATASET_SCHEMA ||
        error("streaming release dataset schema mismatch")
    String(_required(manifest, :shard_schema)) == RELEASE_SHARD_SCHEMA ||
        error("streaming release shard schema mismatch")
    lowercase(String(_required(manifest, :completion_state))) ==
        "complete" ||
        error("streaming release dataset is incomplete")
    String(_required(manifest, :official_neuron_schema)) ==
        OFFICIAL_TEACHER_SCHEMA ||
        error("streaming release official teacher schema mismatch")
    if require_promotion_eligible
        _get(manifest, :promotion_eligible, false) === true ||
            error("streaming dataset is not promotion eligible")
    end
    _get(manifest, :mixed_supervision, false) === true ||
        error("streaming dataset is not mixed supervision")
    _get(manifest, :digital_twin_gate_passed, false) === true ||
        error("upstream frozen-twin gate did not pass")
    _get(manifest, :twin_self_report_trusted, true) === false ||
        error("streaming dataset trusted a twin self-report")
    recomputed_gate = _required(manifest, :recomputed_twin_gate)
    recomputed_auroc =
        Float64(_required(recomputed_gate, :spike_auroc))
    isfinite(recomputed_auroc) &&
        recomputed_auroc >= Float64(minimum_spike_auroc) ||
        error(
            "stream-recomputed frozen-twin spike AUROC " *
            "$recomputed_auroc is below $minimum_spike_auroc",
        )
    _maximum_delta(_get(manifest, :integrity_before, nothing)) == 0 ||
        error("upstream twin integrity-before delta is nonzero")
    _maximum_delta(_get(manifest, :integrity_after, nothing)) == 0 ||
        error("upstream twin integrity-after delta is nonzero")

    frozen.model.config.segments == OFFICIAL_SEGMENTS ||
        error("frozen twin is not the official 642-segment model")
    twin_parameter_hash = _manifest_hash_value(
        manifest,
        (:frozen_twin_parameter_hash,),
    )
    twin_artifact_hash = _manifest_hash_value(
        manifest,
        (:frozen_twin_artifact_hash, :digital_twin_hash),
    )
    twin_file_hash = _manifest_hash_value(
        manifest,
        (:frozen_twin_file_sha256,),
    )
    twin_parameter_hash == lowercase(String(frozen.parameter_sha256)) ||
        error("manifest/frozen-twin parameter hash mismatch")
    twin_artifact_hash == lowercase(String(frozen.artifact_sha256)) ||
        error("manifest/frozen-twin artifact hash mismatch")

    total_samples = Int(_required(manifest, :total_samples))
    time_steps = Int(_required(manifest, :time_steps))
    total_samples >= 1 || error("streaming dataset has no samples")
    time_steps >= 1 || error("streaming dataset has no time steps")
    input_dim = Int(frozen.model.config.input_dim)
    input_dim == 6OFFICIAL_SEGMENTS ||
        error("frozen twin input is not the official six-plane layout")
    String(_required(manifest, :input_representation)) ==
        "compact_ragged_contact_event_location_v2" ||
        error("streaming input representation mismatch")
    _get(manifest, :dense_memory_scales_with_total_samples, true) ===
        false ||
        error("streaming manifest permits total-sample dense memory")

    raw_records = collect(_required(manifest, :shards))
    isempty(raw_records) && error("streaming manifest has no shards")
    records = StreamShardRecord[]
    global_to_shard = zeros(Int32, total_samples)
    next_global = 1
    digest_records = NamedTuple[]
    for (shard_index, raw) in enumerate(raw_records)
        relative = String(_required(raw, :path))
        source = _safe_shard_path(root, relative)
        sha256 = _require_sha256(
            _required(raw, :sha256),
            "shard sha256",
        )
        bytes = Int(_required(raw, :bytes))
        samples = Int(_required(raw, :samples))
        global_first = Int(_required(raw, :global_first))
        global_last = Int(_required(raw, :global_last))
        bytes == filesize(source) ||
            error("stream shard byte count changed: $relative")
        global_first == next_global ||
            error("stream shard global ranges contain a gap/reordering")
        global_last - global_first + 1 == samples ||
            error("stream shard sample/range count differs")
        global_last <= total_samples ||
            error("stream shard range exceeds total samples")
        if verify_shard_hashes
            _file_sha256(source) == sha256 ||
                error("stream shard SHA-256 mismatch: $relative")
        end
        push!(
            records,
            StreamShardRecord(
                source,
                relative,
                sha256,
                bytes,
                samples,
                global_first,
                global_last,
            ),
        )
        global_to_shard[global_first:global_last] .= Int32(shard_index)
        push!(digest_records, (; path=relative, sha256))
        next_global = global_last + 1
    end
    next_global == total_samples + 1 ||
        error("stream shard ranges do not cover every sample")

    train_indices =
        _split_indices(manifest, :train_indices, total_samples)
    validation_indices =
        _split_indices(manifest, :validation_indices, total_samples)
    test_indices =
        _split_indices(manifest, :test_indices, total_samples)
    all_indices = vcat(
        train_indices,
        validation_indices,
        test_indices,
    )
    sort(all_indices) == collect(1:total_samples) ||
        error("train/validation/test indices do not partition samples")
    split_code = zeros(UInt8, total_samples)
    split_code[train_indices] .= TRAIN_SPLIT
    split_code[validation_indices] .= VALIDATION_SPLIT
    split_code[test_indices] .= TEST_SPLIT

    diagnostic_zero =
        Int.(collect(_required(manifest, :diagnostic_time_indices)))
    issorted(diagnostic_zero) ||
        error("diagnostic time indices are not ordered")
    length(unique(diagnostic_zero)) == length(diagnostic_zero) ||
        error("diagnostic time indices repeat")
    all(time -> 0 <= time < time_steps, diagnostic_zero) ||
        error("diagnostic time index lies outside the trajectory")
    diagnostic_time_indices = diagnostic_zero .+ 1

    segment_region =
        String.(collect(_required(manifest, :segment_region)))
    length(segment_region) == OFFICIAL_SEGMENTS ||
        error("manifest lacks the official 642 segment-region map")
    segment_catalog_sha256 = _manifest_hash_value(
        manifest,
        (:segment_catalog_sha256,),
    )
    provenance = (;
        detailed_teacher_hash=_manifest_hash_value(
            manifest,
            (:detailed_teacher_hash, :teacher_hash),
        ),
        detailed_kernel_hash=_manifest_hash_value(
            manifest,
            (:detailed_kernel_hash, :cell_mechanism_sha256),
        ),
        morphology_hash=_manifest_hash_value(
            manifest,
            (:morphology_hash,),
        ),
        official_modeldb_source_hash=_manifest_hash_value(
            manifest,
            (:official_modeldb_source_hash,),
        ),
        official_source_dataset_hash=_manifest_hash_value(
            manifest,
            (:source_dataset_hash,),
        ),
        official_teacher_file_hash=_manifest_hash_value(
            manifest,
            (:source_manifest_sha256,),
        ),
        segment_catalog_sha256,
        frozen_twin_parameter_hash=twin_parameter_hash,
        frozen_twin_artifact_hash=twin_artifact_hash,
        raw_twin_file_sha256=twin_file_hash,
        twin_held_out_spike_auroc=recomputed_auroc,
    )
    dataset_sha256 = _value_sha256((
        manifest_sha256,
        Tuple(digest_records),
    ))
    return StreamDataset(
        manifest_path,
        root,
        manifest_sha256,
        dataset_sha256,
        manifest,
        records,
        total_samples,
        time_steps,
        input_dim,
        train_indices,
        validation_indices,
        test_indices,
        split_code,
        global_to_shard,
        diagnostic_time_indices,
        segment_region,
        segment_catalog_sha256,
        provenance,
        twin_file_hash,
        verify_shard_hashes,
        StreamMemoryTracker(),
    )
end

function _offset_range(offsets, local_index, total, label)
    length(offsets) >= local_index + 1 ||
        error("$label offsets are too short")
    first_item = Int(offsets[local_index]) + 1
    last_item = Int(offsets[local_index + 1])
    0 <= first_item - 1 <= last_item <= total ||
        error("$label offsets are invalid")
    return first_item:last_item
end

function _load_shard(dataset::StreamDataset, shard_index::Int)
    record = dataset.records[shard_index]
    loaded = JLD2.load(record.path)
    haskey(loaded, "dataset") ||
        error("stream shard has no dataset payload")
    shard = loaded["dataset"]
    String(_required(shard, :schema)) == RELEASE_SHARD_SCHEMA ||
        error("stream shard schema mismatch")
    String(_required(shard, :input_representation)) ==
        "compact_ragged_contact_event_location_v2" ||
        error("stream shard input representation mismatch")
    _get(shard, :input, :missing) === nothing ||
        error("stream shard unexpectedly stores a dense input")
    Int(_required(shard, :time_steps)) == dataset.time_steps ||
        error("stream shard time length differs from manifest")
    Int.(collect(_required(shard, :global_output_indices))) ==
        collect(record.global_first:record.global_last) ||
        error("stream shard global indices differ from manifest")
    local_split = UInt8.(collect(_required(shard, :split_code)))
    length(local_split) == record.samples ||
        error("stream shard split length differs")
    local_split ==
        dataset.split_code[record.global_first:record.global_last] ||
        error("stream shard split assignment differs from manifest")
    Int.(collect(_required(shard, :diagnostic_time_indices))) .+ 1 ==
        dataset.diagnostic_time_indices ||
        error("stream shard diagnostic time grid differs")
    for (name, expected) in (
        (
            :frozen_twin_parameter_hash,
            dataset.provenance.frozen_twin_parameter_hash,
        ),
        (
            :frozen_twin_artifact_hash,
            dataset.provenance.frozen_twin_artifact_hash,
        ),
        (
            :frozen_twin_file_sha256,
            dataset.provenance.raw_twin_file_sha256,
        ),
        (
            :detailed_teacher_hash,
            dataset.provenance.detailed_teacher_hash,
        ),
        (
            :detailed_kernel_hash,
            dataset.provenance.detailed_kernel_hash,
        ),
        (:morphology_hash, dataset.provenance.morphology_hash),
        (
            :segment_catalog_sha256,
            dataset.provenance.segment_catalog_sha256,
        ),
    )
        lowercase(String(_required(shard, name))) == expected ||
            error("stream shard lineage mismatch for $(String(name))")
    end
    time_steps = dataset.time_steps
    samples = record.samples
    size(_required(shard, :target_voltage)) ==
        (time_steps, samples) ||
        error("stream target_voltage shape differs")
    size(_required(shard, :target_spike)) ==
        (time_steps, samples) ||
        error("stream target_spike shape differs")
    size(_required(shard, :target_nmda)) ==
        (4, time_steps, samples) ||
        error("stream target_nmda shape differs")
    diagnostics = length(dataset.diagnostic_time_indices)
    size(_required(shard, :target_calcium_event)) ==
        (diagnostics, samples) ||
        error("stream target_calcium_event shape differs")
    size(_required(shard, :target_dendritic_voltage)) ==
        (4, diagnostics, samples) ||
        error("stream target_dendritic_voltage shape differs")
    contact_axon = vec(_required(shard, :contact_axon))
    contact_segment = vec(_required(shard, :contact_segment))
    contact_kind = vec(_required(shard, :contact_kind))
    contact_strength = vec(_required(shard, :contact_strength))
    all(length(contact_axon) == length(values) for values in (
        contact_segment,
        contact_kind,
        contact_strength,
    )) || error("stream contact ragged fields differ")
    contact_offsets = vec(_required(shard, :contact_trial_offset))
    length(contact_offsets) == samples + 1 ||
        error("stream contact offsets differ")
    Int(first(contact_offsets)) == 0 &&
        Int(last(contact_offsets)) == length(contact_axon) ||
        error("stream contact offsets do not cover contacts")
    event_axon = vec(_required(shard, :event_axon))
    event_time = vec(_required(shard, :event_time_bin))
    event_count = vec(_required(shard, :event_count))
    length(event_axon) == length(event_time) ==
        length(event_count) ||
        error("stream event ragged fields differ")
    event_offsets = vec(_required(shard, :event_trial_offset))
    length(event_offsets) == samples + 1 ||
        error("stream event offsets differ")
    Int(first(event_offsets)) == 0 &&
        Int(last(event_offsets)) == length(event_axon) ||
        error("stream event offsets do not cover events")
    all(isfinite, _required(shard, :target_voltage)) ||
        error("stream voltage target is non-finite")
    all(isfinite, _required(shard, :target_spike)) ||
        error("stream spike target is non-finite")
    all(isfinite, _required(shard, :target_nmda)) ||
        error("stream NMDA target is non-finite")
    all(isfinite, _required(shard, :target_calcium_event)) ||
        error("stream calcium target is non-finite")
    all(isfinite, _required(shard, :target_dendritic_voltage)) ||
        error("stream dendritic target is non-finite")
    dataset.tracker.peak_loaded_shard_bytes = max(
        dataset.tracker.peak_loaded_shard_bytes,
        record.bytes,
    )
    return shard
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

function _fill_raw_window!(
    destination::AbstractMatrix{Float32},
    shard,
    local_index::Int,
    first_time::Int,
    last_time::Int,
)
    segments = OFFICIAL_SEGMENTS
    size(destination) ==
        (6segments, last_time - first_time + 1) ||
        throw(DimensionMismatch("stream raw destination shape differs"))
    fill!(destination, 0.0f0)
    contact_axon = Int.(vec(_required(shard, :contact_axon)))
    contact_segment = Int.(vec(_required(shard, :contact_segment)))
    contact_kind = UInt8.(vec(_required(shard, :contact_kind)))
    contact_strength =
        Float32.(vec(_required(shard, :contact_strength)))
    contact_range = _offset_range(
        vec(_required(shard, :contact_trial_offset)),
        local_index,
        length(contact_axon),
        "contact",
    )
    contacts_by_axon = Dict{Int,Vector{Int}}()
    static_strength = zeros(Float32, 3segments)
    for contact in contact_range
        segment = contact_segment[contact]
        1 <= segment <= segments ||
            error("stream contact targets a nonofficial segment")
        kind = contact_kind[contact]
        kind in (UInt8(1), UInt8(2)) ||
            error("stream contact violates Dale E/I coding")
        strength = contact_strength[contact]
        isfinite(strength) && strength >= 0.0f0 ||
            error("stream contact strength is invalid")
        receptors = kind == UInt8(1) ? (1, 2) : (3,)
        for receptor in receptors
            feature = _feature_index(
                segments,
                segment,
                receptor,
                1,
            )
            static_strength[feature] += strength
        end
        push!(
            get!(contacts_by_axon, contact_axon[contact], Int[]),
            contact,
        )
    end
    static_strength .= min.(static_strength, 1.0f0)
    event_plane_offset = 3segments
    for local_time in axes(destination, 2)
        copyto!(
            @view(destination[
                (event_plane_offset + 1):(6segments),
                local_time,
            ]),
            static_strength,
        )
    end

    event_axon = Int.(vec(_required(shard, :event_axon)))
    event_time = Int.(vec(_required(shard, :event_time_bin)))
    event_count = UInt8.(vec(_required(shard, :event_count)))
    event_range = _offset_range(
        vec(_required(shard, :event_trial_offset)),
        local_index,
        length(event_axon),
        "event",
    )
    if !isempty(event_range)
        local_times = @view event_time[event_range]
        event_indices = if issorted(local_times)
            first_event = searchsortedfirst(local_times, first_time - 1)
            last_event = searchsortedlast(local_times, last_time - 1)
            first_event > last_event ?
                Int[] :
                collect(
                    first(event_range) + first_event - 1:
                    first(event_range) + last_event - 1,
                )
        else
            [
                event for event in event_range
                if first_time - 1 <= event_time[event] <= last_time - 1
            ]
        end
        for event in event_indices
            global_time = event_time[event] + 1
            local_time = global_time - first_time + 1
            multiplicity = Float32(event_count[event])
            multiplicity > 0.0f0 ||
                error("stream event multiplicity is not positive")
            for contact in get(
                contacts_by_axon,
                event_axon[event],
                Int[],
            )
                receptors = contact_kind[contact] == UInt8(1) ?
                    (1, 2) : (3,)
                for receptor in receptors
                    feature = _feature_index(
                        segments,
                        contact_segment[contact],
                        receptor,
                        1,
                    )
                    destination[feature, local_time] +=
                        contact_strength[contact] * multiplicity
                end
            end
        end
    end
    @views destination[1:(3segments), :] .=
        min.(destination[1:(3segments), :], 1.0f0)
    all(isfinite, destination) ||
        error("stream-expanded input is non-finite")
    return destination
end

@inline function _interpolate(
    values,
    diagnostic_times::Vector{Int},
    time::Int,
)
    upper = searchsortedfirst(diagnostic_times, time)
    upper == 1 && return Float32(values[1])
    upper > length(diagnostic_times) &&
        return Float32(values[end])
    diagnostic_times[upper] == time &&
        return Float32(values[upper])
    lower = upper - 1
    lower_time = diagnostic_times[lower]
    upper_time = diagnostic_times[upper]
    fraction =
        Float32(time - lower_time) /
        Float32(upper_time - lower_time)
    return muladd(
        fraction,
        Float32(values[upper]) - Float32(values[lower]),
        Float32(values[lower]),
    )
end

function _fill_target_window!(
    target::AbstractMatrix{Float32},
    observed::AbstractMatrix{Bool},
    dataset::StreamDataset,
    shard,
    local_index::Int,
    first_time::Int,
    last_time::Int,
)
    window = last_time - first_time + 1
    size(target) == (11, window) ||
        throw(DimensionMismatch("stream target destination differs"))
    size(observed) == (11, window) ||
        throw(DimensionMismatch("stream observed mask differs"))
    target[1, :] .= @view(
        _required(shard, :target_voltage)[
            first_time:last_time,
            local_index,
        ]
    )
    target[2, :] .= @view(
        _required(shard, :target_spike)[
            first_time:last_time,
            local_index,
        ]
    )
    target[3:6, :] .= @view(
        _required(shard, :target_nmda)[
            :,
            first_time:last_time,
            local_index,
        ]
    )
    observed[1:6, :] .= true
    calcium = @view(
        _required(shard, :target_calcium_event)[:, local_index]
    )
    dendritic = @view(
        _required(shard, :target_dendritic_voltage)[
            :,
            :,
            local_index,
        ]
    )
    diagnostic_times = dataset.diagnostic_time_indices
    for (local_time, global_time) in
        enumerate(first_time:last_time)
        target[7, local_time] =
            _interpolate(calcium, diagnostic_times, global_time)
        for branch in 1:4
            target[7 + branch, local_time] = _interpolate(
                @view(dendritic[branch, :]),
                diagnostic_times,
                global_time,
            )
        end
        diagnostic_position =
            searchsortedfirst(diagnostic_times, global_time)
        if diagnostic_position <= length(diagnostic_times) &&
           diagnostic_times[diagnostic_position] == global_time
            observed[7:11, local_time] .= true
        end
    end
    all(isfinite, target) ||
        error("stream materialized target is non-finite")
    return target, observed
end

function stream_materialize_window(
    dataset::StreamDataset,
    global_indices,
    first_time::Integer,
    window::Integer,
)
    samples = Int.(collect(global_indices))
    isempty(samples) &&
        throw(ArgumentError("stream window needs at least one sample"))
    first = Int(first_time)
    count = Int(window)
    count >= 1 || throw(ArgumentError("stream window must be positive"))
    last = first + count - 1
    1 <= first <= last <= dataset.time_steps ||
        throw(BoundsError(1:dataset.time_steps, first:last))
    all(index -> 1 <= index <= dataset.total_samples, samples) ||
        throw(BoundsError(1:dataset.total_samples, samples))
    batch = length(samples)
    raw_input =
        zeros(Float32, dataset.input_dim, count, batch)
    target = zeros(Float32, 11, count, batch)
    observed = falses(11, count, batch)
    shard_indices = Int.(
        dataset.global_to_shard[samples]
    )
    order = sortperm(shard_indices)
    current_shard_index = 0
    current_shard = nothing
    current_shard_bytes = 0
    for ordered_position in order
        global_index = samples[ordered_position]
        shard_index = shard_indices[ordered_position]
        if shard_index != current_shard_index
            current_shard = _load_shard(dataset, shard_index)
            current_shard_index = shard_index
            current_shard_bytes =
                dataset.records[shard_index].bytes
        end
        record = dataset.records[shard_index]
        local_index = global_index - record.global_first + 1
        _fill_raw_window!(
            @view(raw_input[:, :, ordered_position]),
            current_shard,
            local_index,
            first,
            last,
        )
        _fill_target_window!(
            @view(target[:, :, ordered_position]),
            @view(observed[:, :, ordered_position]),
            dataset,
            current_shard,
            local_index,
            first,
            last,
        )
        window_bytes =
            sizeof(Float32) * (length(raw_input) + length(target)) +
            sizeof(Bool) * length(observed)
        dataset.tracker.peak_dense_window_bytes = max(
            dataset.tracker.peak_dense_window_bytes,
            window_bytes,
        )
        dataset.tracker.peak_combined_bytes = max(
            dataset.tracker.peak_combined_bytes,
            current_shard_bytes + window_bytes,
        )
    end
    dataset.tracker.windows_materialized += 1
    dataset.tracker.samples_materialized += batch
    return (; raw_input, target, observed)
end

mutable struct _RunningStatistic
    count::Int64
    sum::Float64
    sum2::Float64
end

_RunningStatistic() = _RunningStatistic(0, 0.0, 0.0)

function _update!(statistic::_RunningStatistic, values)
    for raw in values
        value = Float64(raw)
        isfinite(value) ||
            error("target statistics contain non-finite data")
        statistic.count += 1
        statistic.sum += value
        statistic.sum2 += value * value
    end
    return statistic
end

function stream_target_statistics(dataset::StreamDataset)
    statistics = [_RunningStatistic() for _ in 1:11]
    for (shard_index, record) in enumerate(dataset.records)
        shard = _load_shard(dataset, shard_index)
        for local_index in 1:record.samples
            global_index = record.global_first + local_index - 1
            dataset.split_code[global_index] == TRAIN_SPLIT ||
                continue
            _update!(
                statistics[1],
                @view(_required(shard, :target_voltage)[:, local_index]),
            )
            for region in 1:4
                _update!(
                    statistics[2 + region],
                    @view(
                        _required(shard, :target_nmda)[
                            region,
                            :,
                            local_index,
                        ]
                    ),
                )
                _update!(
                    statistics[7 + region],
                    @view(
                        _required(shard, :target_dendritic_voltage)[
                            region,
                            :,
                            local_index,
                        ]
                    ),
                )
            end
        end
    end
    target_mean = zeros(Float32, 11)
    target_scale = ones(Float32, 11)
    for coordinate in (1, 3, 4, 5, 6, 8, 9, 10, 11)
        statistic = statistics[coordinate]
        statistic.count > 0 ||
            error("stream target coordinate $coordinate has no train data")
        mean_value = statistic.sum / statistic.count
        variance = max(
            statistic.sum2 / statistic.count -
            mean_value * mean_value,
            0.0,
        )
        target_mean[coordinate] = Float32(mean_value)
        target_scale[coordinate] =
            max(Float32(sqrt(variance)), 1.0f-4)
    end
    return target_mean, target_scale
end

function stream_dataset_integrity!(dataset::StreamDataset)
    observed_manifest = _file_sha256(dataset.manifest_path)
    observed_manifest == dataset.manifest_sha256 ||
        error("streaming manifest changed during distillation")
    if dataset.verified_shard_hashes
        for record in dataset.records
            filesize(record.path) == record.bytes ||
                error("stream shard byte count changed during distillation")
            _file_sha256(record.path) == record.sha256 ||
                error("stream shard changed during distillation")
        end
    end
    observed_dataset = _value_sha256((
        observed_manifest,
        Tuple((;
            path=record.relative_path,
            sha256=record.sha256,
        ) for record in dataset.records),
    ))
    observed_dataset == dataset.dataset_sha256 ||
        error("streaming dataset lineage digest changed")
    return observed_dataset
end

end # module StreamingReleaseDataset
