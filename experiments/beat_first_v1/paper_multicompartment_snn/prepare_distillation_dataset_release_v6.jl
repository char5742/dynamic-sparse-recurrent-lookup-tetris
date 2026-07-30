module DistillationDatasetBridgeReleaseV6

using Dates
using JLD2
using JSON3
using NPZ
using SHA

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(_PARENT_MODULE, :DistillationDatasetBridgeStreamingV5)
    Base.include(
        _PARENT_MODULE,
        joinpath(
            @__DIR__,
            "prepare_distillation_dataset_streaming_v5.jl",
        ),
    )
end

using ..DistillationDatasetBridgeStreamingV5
using ..PaperDigitalTwin

const Legacy = DistillationDatasetBridgeStreamingV5
const BaseBridge = Legacy.BaseBridge

export FINAL_NEURON_SCHEMA,
    RELEASE_DATASET_SCHEMA,
    RELEASE_SHARD_SCHEMA,
    ReleaseStreamingPrepareConfig,
    prepare_distillation_dataset_release,
    main

const FINAL_NEURON_SCHEMA =
    "hd_swsnn_twinprop.neuron_teacher.final.v2"
const RELEASE_DATASET_SCHEMA =
    "hd-swsnn-twinprop-distillation-dataset-sharded-release-v2"
const RELEASE_SHARD_SCHEMA =
    "hd-swsnn-twinprop-distillation-shard-release-v2"
const PAPER_TRAIN_POOL = 50_000
const PAPER_HELD_OUT_TEST = 2_000
const PAPER_VALIDATION = 1_000
const PAPER_DURATION_MS = 10_000

const _FINAL_NUMERIC_FIELDS = [
    "sample_indices",
    "split_code",
    "axon_kind",
    "contact_count_per_axon",
    "contact_trial_offset",
    "contact_axon",
    "contact_segment",
    "contact_location_slot",
    "contact_section",
    "contact_x",
    "contact_path_distance_um",
    "contact_kind",
    "contact_strength",
    "event_trial_offset",
    "event_axon",
    "event_time_bin",
    "event_count",
    "rate_window_ms",
    "rate_sigma_ms",
    "rate_mean_hz",
    "rate_std_hz",
    "diagnostic_segment_indices",
    "diagnostic_time_indices",
    "time_ms",
    "diagnostic_time_ms",
    "target_voltage",
    "target_spike",
    "target_nmda",
    "target_compartment_voltage",
    "target_compartment_nmda",
    "target_dendritic_cai",
    "target_dendritic_ica",
    "target_ca_event",
]

Base.@kwdef struct ReleaseStreamingPrepareConfig
    dataset_path::String
    frozen_twin_path::String
    output_directory::String
    validation_samples::Int = PAPER_VALIDATION
    validation_hash_seed::String =
        "HD-SWSNN-TwinProp/final-v2/validation-v1"
    maximum_train_samples::Int = typemax(Int)
    maximum_validation_samples::Int = typemax(Int)
    maximum_test_samples::Int = typemax(Int)
    time_chunk::Int = 256
    output_shard_samples::Int = 2
    minimum_twin_spike_auroc::Float64 = 0.985
    auroc_histogram_bins::Int = 16_384
    require_full_public_counts::Bool = true
    expected_source_dataset_sha256::String = ""
    expected_modeldb_source_sha256::String = ""
    expected_detailed_teacher_sha256::String = ""
    expected_detailed_kernel_sha256::String = ""
    expected_morphology_sha256::String = ""
    expected_twin_parameter_sha256::String = ""
    expected_twin_artifact_sha256::String = ""
    selected_dendritic_segments::Vector{Int} = Int[]
end

function _base_config(config::ReleaseStreamingPrepareConfig)
    return BaseBridge.PrepareDistillationConfig(
        dataset_path=config.dataset_path,
        frozen_twin_path=config.frozen_twin_path,
        output_path="unused-by-release-streaming-bridge",
        source_kind=:official_neuron,
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

@inline _value(object, name::Symbol, default=nothing) =
    BaseBridge._value(object, name, default)

function _required(object, name::Symbol)
    value = _value(object, name)
    value === nothing && error("required field $(String(name)) is absent")
    return value
end

function _canonical_json(value; drop_contract_digest::Bool=false)
    if value isa AbstractDict ||
       (!(value isa AbstractArray) &&
        !(value isa AbstractString) &&
        !(value isa Number) &&
        !(value isa Bool) &&
        value !== nothing &&
        propertynames(value) != ())
        names = if value isa AbstractDict
            sort!(String.(collect(keys(value))))
        else
            sort!(String.(collect(propertynames(value))))
        end
        drop_contract_digest &&
            filter!(!=("teacher_contract_sha256"), names)
        body = String[]
        for text in names
            push!(
                body,
                JSON3.write(text) * ":" *
                _canonical_json(
                    _value(value, Symbol(text));
                    drop_contract_digest=false,
                ),
            )
        end
        return "{" * join(body, ",") * "}"
    elseif value isa AbstractArray
        return "[" *
            join((_canonical_json(child) for child in value), ",") *
            "]"
    end
    return JSON3.write(value)
end

_sha256_text(value::AbstractString) =
    bytes2hex(SHA.sha256(codeunits(value)))

function _inside_root(path::AbstractString, root::AbstractString)
    absolute = lowercase(abspath(path))
    prefix =
        lowercase(abspath(root)) *
        string(Base.Filesystem.path_separator)
    return startswith(absolute, prefix)
end

function _validate_public_contract(manifest)
    paper = _required(manifest, :paper_production_contract)
    Int(_required(paper, :train_trials)) == PAPER_TRAIN_POOL ||
        error("paper train-pool contract is not 50,000")
    Int(_required(paper, :held_out_test_trials)) ==
        PAPER_HELD_OUT_TEST ||
        error("paper held-out contract is not 2,000")
    Int(_required(paper, :duration_ms)) == PAPER_DURATION_MS ||
        error("paper duration contract is not 10,000 ms")
    contract = _required(manifest, :teacher_contract)
    paper_protocol = _required(contract, :paper_protocol)
    Int(_required(paper_protocol, :train_simulations)) ==
        PAPER_TRAIN_POOL ||
        error("embedded teacher contract lost the 50,000 train pool")
    Int(_required(paper_protocol, :held_out_test_simulations)) ==
        PAPER_HELD_OUT_TEST ||
        error("embedded teacher contract lost the 2,000 held-out set")
    Int(_required(paper_protocol, :duration_ms)) ==
        PAPER_DURATION_MS ||
        error("embedded teacher contract lost the 10-second duration")
    return nothing
end

function _load_release_source(config::ReleaseStreamingPrepareConfig)
    root = abspath(config.dataset_path)
    isdir(root) ||
        error("teacher dataset must be a manifest directory: $root")
    manifest_path = joinpath(root, "manifest.json")
    isfile(manifest_path) ||
        error("teacher dataset has no manifest.json")
    manifest = JSON3.read(read(manifest_path, String))
    String(_required(manifest, :schema_name)) ==
        FINAL_NEURON_SCHEMA || error(
        "release promotion accepts only $FINAL_NEURON_SCHEMA",
    )
    String(_required(manifest, :model_name)) ==
        HD_SWSNN_TWINPROP_NAME ||
        error("teacher model family is not HD-SWSNN-TwinProp")
    String(_required(manifest, :stage)) ==
        "official_hay_neuron_teacher_final" ||
        error("teacher is not the final official Hay/NEURON stage")
    String(_required(manifest, :completion_state)) == "complete" ||
        error("official final-v2 generation is incomplete")
    _required(manifest, :modeldb_source_modified_by_generator) ===
        false ||
        error("official ModelDB checkout was modified by the generator")
    _validate_public_contract(manifest)

    contract = _required(manifest, :teacher_contract)
    String(_required(contract, :schema_name)) ==
        FINAL_NEURON_SCHEMA ||
        error("embedded teacher contract uses another schema")
    declared_contract = BaseBridge._require_hash(
        "teacher contract",
        _required(manifest, :teacher_contract_sha256),
    )
    embedded_contract = BaseBridge._require_hash(
        "embedded teacher contract",
        _required(contract, :teacher_contract_sha256),
    )
    declared_contract == embedded_contract ||
        error("embedded teacher-contract hash differs")
    computed_contract = _sha256_text(
        _canonical_json(contract; drop_contract_digest=true),
    )
    computed_contract == declared_contract ||
        error("teacher-contract canonical SHA-256 mismatch")
    _canonical_json(_required(contract, :config)) ==
        _canonical_json(_required(manifest, :config)) ||
        error("teacher contract/config differs from manifest")
    _canonical_json(_required(contract, :source_hashes)) ==
        _canonical_json(_required(manifest, :source_hashes)) ||
        error("teacher contract/source hashes differ from manifest")

    detailed_teacher,
    detailed_kernel,
    morphology,
    modeldb = BaseBridge._official_hashes(manifest)
    detailed_teacher == declared_contract ||
        error("official teacher lineage is not the final contract")
    BaseBridge._expected_hash(
        "detailed teacher",
        detailed_teacher,
        config.expected_detailed_teacher_sha256,
    )
    BaseBridge._expected_hash(
        "detailed kernel",
        detailed_kernel,
        config.expected_detailed_kernel_sha256,
    )
    BaseBridge._expected_hash(
        "morphology",
        morphology,
        config.expected_morphology_sha256,
    )
    BaseBridge._expected_hash(
        "ModelDB source",
        modeldb,
        config.expected_modeldb_source_sha256,
    )

    records = collect(_required(manifest, :shards))
    isempty(records) && error("teacher manifest has no shards")
    shard_paths = String[]
    shard_hashes = String[]
    expected_first = 1
    train_count = 0
    test_count = 0
    for record in records
        relative = String(_required(record, :path))
        path = abspath(joinpath(root, relative))
        _inside_root(path, root) ||
            error("teacher shard escapes dataset root: $relative")
        isfile(path) || error("teacher shard is absent: $path")
        Int(_required(record, :global_first)) == expected_first ||
            error("teacher shard ranges are not contiguous")
        global_last = Int(_required(record, :global_last))
        global_last >= expected_first ||
            error("teacher shard range is empty")
        samples = Int(_required(record, :samples))
        samples == global_last - expected_first + 1 ||
            error("teacher shard sample count differs from its range")
        expected_first = global_last + 1
        filesize(path) == Int(_required(record, :bytes)) ||
            error("teacher shard byte count differs: $relative")
        declared = BaseBridge._require_hash(
            "declared teacher shard",
            _required(record, :sha256),
        )
        actual = BaseBridge._sha256_file(path)
        actual == declared || error(
            "teacher shard hash mismatch for $relative: " *
            "expected $declared, got $actual",
        )
        BaseBridge._require_hash(
            "shard teacher contract",
            _required(record, :teacher_contract_sha256),
        ) == declared_contract ||
            error("teacher shard uses another teacher contract")
        counts = _required(record, :split_counts)
        train_count += Int(_required(counts, :train))
        test_count += Int(_required(counts, :held_out_test))
        push!(shard_paths, path)
        push!(shard_hashes, actual)
    end
    completed = Int(_required(manifest, :completed_trials))
    expected_first == completed + 1 ||
        error("teacher shard ranges do not cover completed trials")
    completed == train_count + test_count ||
        error("teacher split counts do not cover completed trials")

    source_config = _required(manifest, :config)
    configured_train = Int(_required(source_config, :train_trials))
    configured_test = Int(_required(source_config, :test_trials))
    configured_duration = Int(_required(source_config, :duration_ms))
    configured_train == train_count ||
        error("manifest train count differs from shard records")
    configured_test == test_count ||
        error("manifest test count differs from shard records")
    if config.require_full_public_counts
        (train_count, test_count, configured_duration) ==
            (
                PAPER_TRAIN_POOL,
                PAPER_HELD_OUT_TEST,
                PAPER_DURATION_MS,
            ) || error(
            "promotable release requires complete 50k/2k 10-second source",
        )
        completed == PAPER_TRAIN_POOL + PAPER_HELD_OUT_TEST ||
            error("promotable release source is not all 52,000 trials")
        Int(_required(manifest, :total_segments)) == 642 ||
            error("promotable release source must expose 642 segments")
    end

    manifest_hash = BaseBridge._sha256_file(manifest_path)
    dataset_hash =
        BaseBridge._sha256_strings(manifest_hash, shard_hashes...)
    BaseBridge._expected_hash(
        "source dataset",
        dataset_hash,
        config.expected_source_dataset_sha256,
    )
    source = BaseBridge._Source(
        root,
        manifest_path,
        manifest,
        shard_paths,
        dataset_hash,
        manifest_hash,
        FINAL_NEURON_SCHEMA,
        "complete",
        detailed_teacher,
        detailed_kernel,
        morphology,
        modeldb,
    )
    segment_catalog_sha256 =
        _sha256_text(_canonical_json(_required(manifest, :segments)))
    source_counts = (;
        train_pool=train_count,
        held_out_test=test_count,
        duration_ms=configured_duration,
        completed,
    )
    return source, source_counts, segment_catalog_sha256
end

function _read_final_shard(path::AbstractString)
    extension = lowercase(splitext(path)[2])
    if extension == ".npz"
        return NPZ.npzread(path, _FINAL_NUMERIC_FIELDS)
    elseif extension == ".jld2"
        return BaseBridge._unwrap(JLD2.load(path))
    end
    error("unsupported final teacher shard extension $extension")
end

function _validate_offsets(offsets, count, samples, label)
    values = Int64.(vec(offsets))
    length(values) == samples + 1 ||
        error("$label offsets must have trial_count + 1 entries")
    first(values) == 0 ||
        error("$label offsets must be zero based")
    issorted(values) ||
        error("$label offsets are not monotone")
    last(values) == count ||
        error("$label offsets do not cover the ragged payload")
    return values
end

function _validate_final_shard(shard)
    split_code = UInt8.(vec(_required(shard, :split_code)))
    samples = length(split_code)
    samples > 0 || error("teacher shard is empty")
    all(code -> code in (
        BaseBridge.TRAIN_SPLIT,
        BaseBridge.TEST_SPLIT,
    ), split_code) ||
        error("final-v2 source may contain only train/test codes")
    source_ids = Int32.(vec(_required(shard, :sample_indices)))
    length(source_ids) == samples ||
        error("sample_indices/split_code length mismatch")
    length(unique(source_ids)) == samples ||
        error("teacher shard repeats sample_indices")

    contact_axon = vec(_required(shard, :contact_axon))
    for name in (
        :contact_segment,
        :contact_location_slot,
        :contact_section,
        :contact_x,
        :contact_path_distance_um,
        :contact_kind,
        :contact_strength,
    )
        length(vec(_required(shard, name))) == length(contact_axon) ||
            error("ragged contact field $(String(name)) differs")
    end
    contact_offsets = _validate_offsets(
        _required(shard, :contact_trial_offset),
        length(contact_axon),
        samples,
        "contact",
    )
    event_axon = vec(_required(shard, :event_axon))
    for name in (:event_time_bin, :event_count)
        length(vec(_required(shard, name))) == length(event_axon) ||
            error("ragged event field $(String(name)) differs")
    end
    event_offsets = _validate_offsets(
        _required(shard, :event_trial_offset),
        length(event_axon),
        samples,
        "event",
    )
    all(>(0), vec(_required(shard, :event_count))) ||
        error("event multiplicity must be positive")
    voltage = _required(shard, :target_voltage)
    ndims(voltage) == 2 && size(voltage, 2) == samples ||
        error("target_voltage must be time x trial")
    time_steps = size(voltage, 1)
    size(_required(shard, :target_spike)) ==
        (time_steps, samples) ||
        error("target_spike shape differs")
    size(_required(shard, :target_nmda)) ==
        (4, time_steps, samples) ||
        error("target_nmda shape differs")
    diagnostic_segments =
        Int32.(vec(_required(shard, :diagnostic_segment_indices)))
    diagnostic_times =
        Int32.(vec(_required(shard, :diagnostic_time_indices)))
    for name in (
        :target_compartment_voltage,
        :target_compartment_nmda,
        :target_dendritic_cai,
        :target_dendritic_ica,
        :target_ca_event,
    )
        size(_required(shard, name)) ==
            (
                length(diagnostic_segments),
                length(diagnostic_times),
                samples,
            ) || error("$(String(name)) diagnostic shape differs")
    end
    all(time -> 0 <= time < time_steps, diagnostic_times) ||
        error("diagnostic time index lies outside trajectory")
    return (;
        split_code,
        source_ids,
        contact_offsets,
        event_offsets,
        time_steps,
        diagnostic_segments,
        diagnostic_times,
    )
end

function _validation_key(seed::AbstractString, source_id::Integer)
    return _sha256_text(String(seed) * ":" * string(source_id))
end

function _stream_plan(config, source)
    train_ids = Int32[]
    test_ids = Int32[]
    seen_ids = Set{Int32}()
    time_steps = 0
    diagnostic_segments = Int32[]
    diagnostic_times = Int32[]
    for path in source.shard_paths
        shard = _read_final_shard(path)
        layout = _validate_final_shard(shard)
        for (source_id, code) in
            zip(layout.source_ids, layout.split_code)
            source_id in seen_ids &&
                error("sample_indices repeat across teacher shards")
            push!(seen_ids, source_id)
            if code == BaseBridge.TRAIN_SPLIT
                push!(train_ids, source_id)
            else
                push!(test_ids, source_id)
            end
        end
        if time_steps == 0
            time_steps = layout.time_steps
            diagnostic_segments = layout.diagnostic_segments
            diagnostic_times = layout.diagnostic_times
        elseif time_steps != layout.time_steps ||
               diagnostic_segments != layout.diagnostic_segments ||
               diagnostic_times != layout.diagnostic_times
            error("teacher shard trajectory/diagnostic layout changed")
        end
    end
    0 < config.validation_samples < length(train_ids) ||
        error("validation_samples must leave at least one train trial")
    ordered_train = sort(
        copy(train_ids);
        by=id -> (_validation_key(
            config.validation_hash_seed,
            id,
        ), id),
    )
    validation_ids =
        Set(ordered_train[1:config.validation_samples])
    validation_id_vector = sort!(collect(validation_ids))
    validation_digest = _sha256_text(
        config.validation_hash_seed * ":" *
        join(validation_id_vector, ","),
    )

    limits = Dict(
        BaseBridge.TRAIN_SPLIT => config.maximum_train_samples,
        BaseBridge.VALIDATION_SPLIT =>
            config.maximum_validation_samples,
        BaseBridge.TEST_SPLIT => config.maximum_test_samples,
    )
    used = Dict(
        BaseBridge.TRAIN_SPLIT => 0,
        BaseBridge.VALIDATION_SPLIT => 0,
        BaseBridge.TEST_SPLIT => 0,
    )
    plan = NamedTuple[]
    for path in source.shard_paths
        shard = _read_final_shard(path)
        layout = _validate_final_shard(shard)
        selected = Int[]
        selected_code = UInt8[]
        selected_id = Int32[]
        for (trial, (source_id, source_code)) in
            enumerate(zip(layout.source_ids, layout.split_code))
            output_code = if source_code == BaseBridge.TEST_SPLIT
                BaseBridge.TEST_SPLIT
            elseif source_id in validation_ids
                BaseBridge.VALIDATION_SPLIT
            else
                BaseBridge.TRAIN_SPLIT
            end
            used[output_code] >= limits[output_code] && continue
            used[output_code] += 1
            push!(selected, trial)
            push!(selected_code, output_code)
            push!(selected_id, source_id)
        end
        isempty(selected) || push!(
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
            error("release dataset needs train/validation/test trials")
    end
    return (;
        plan,
        time_steps,
        diagnostic_segments,
        diagnostic_times,
        split_counts=used,
        source_train_pool=length(train_ids),
        source_test=length(test_ids),
        validation_ids=validation_id_vector,
        validation_digest,
    )
end

struct _SparseSample
    contact_axon::Vector{Int32}
    contact_segment::Vector{Int32}
    contact_kind::Vector{UInt8}
    contact_strength::Vector{Float32}
    event_axon::Vector{Int32}
    event_time_bin::Vector{Int32}
    event_count::Vector{UInt8}
    static_strength::Vector{Float32}
    contacts_by_axon::Dict{Int,Vector{Int}}
    axons_by_time::Vector{Vector{Tuple{Int,UInt8}}}
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

function _ragged_range(offsets, trial)
    first_item = Int(offsets[trial]) + 1
    last_item = Int(offsets[trial + 1])
    return first_item:last_item
end

function _sparse_sample(shard, trial::Int, twin_config)
    layout = _validate_final_shard(shard)
    contact_range = _ragged_range(layout.contact_offsets, trial)
    event_range = _ragged_range(layout.event_offsets, trial)
    contact_axon =
        Int32.(vec(_required(shard, :contact_axon))[contact_range])
    contact_segment =
        Int32.(vec(_required(shard, :contact_segment))[contact_range])
    contact_kind =
        UInt8.(vec(_required(shard, :contact_kind))[contact_range])
    contact_strength =
        Float32.(vec(_required(shard, :contact_strength))[contact_range])
    event_axon =
        Int32.(vec(_required(shard, :event_axon))[event_range])
    event_time =
        Int32.(vec(_required(shard, :event_time_bin))[event_range])
    event_count =
        UInt8.(vec(_required(shard, :event_count))[event_range])
    static_strength = zeros(Float32, twin_config.input_dim)
    contacts_by_axon = Dict{Int,Vector{Int}}()
    for contact in eachindex(contact_axon)
        destination = Int(contact_segment[contact])
        1 <= destination <= twin_config.segments ||
            error("contact targets a segment absent from the frozen twin")
        kind = contact_kind[contact]
        kind in (UInt8(1), UInt8(2)) ||
            error("contact kind is not Dale-fixed E/I")
        receptors = kind == UInt8(1) ? (1, 2) : (3,)
        for receptor in receptors
            feature = _feature_index(
                twin_config.segments,
                destination,
                receptor,
                2,
            )
            static_strength[feature] += contact_strength[contact]
        end
        push!(
            get!(contacts_by_axon, Int(contact_axon[contact]), Int[]),
            contact,
        )
    end
    static_strength .= min.(static_strength, 1.0f0)
    axons_by_time =
        [Tuple{Int,UInt8}[] for _ in 1:layout.time_steps]
    for event in eachindex(event_axon)
        time = Int(event_time[event]) + 1
        1 <= time <= layout.time_steps ||
            error("compact event time lies outside trajectory")
        push!(
            axons_by_time[time],
            (Int(event_axon[event]), event_count[event]),
        )
    end
    return _SparseSample(
        contact_axon,
        contact_segment,
        contact_kind,
        contact_strength,
        event_axon,
        event_time,
        event_count,
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
    for chunk_time in 1:count
        global_time = first_time + chunk_time - 1
        copyto!(
            @view(input[(event_features + 1):end, chunk_time, 1]),
            @view(sample.static_strength[(event_features + 1):end]),
        )
        for (axon, multiplicity) in
            sample.axons_by_time[global_time]
            contacts = get(sample.contacts_by_axon, axon, Int[])
            for contact in contacts
                receptors =
                    sample.contact_kind[contact] == UInt8(1) ?
                    (1, 2) : (3,)
                for receptor in receptors
                    feature = _feature_index(
                        twin_config.segments,
                        Int(sample.contact_segment[contact]),
                        receptor,
                        1,
                    )
                    input[feature, chunk_time, 1] +=
                        sample.contact_strength[contact] *
                        Float32(multiplicity)
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
        for (label, values) in (
            ("voltage", prediction.voltage),
            ("spike probability", prediction.spike_probability),
            ("NMDA", prediction.nmda),
        )
            all(isfinite, values) ||
                error("frozen twin produced non-finite $label")
        end
        voltage[first_time:last_time] .= vec(prediction.voltage)
        spike[first_time:last_time] .=
            vec(prediction.spike_probability)
        spike_logit[first_time:last_time] .=
            vec(prediction.spike_logit)
        nmda[:, first_time:last_time] .= reshape(
            prediction.nmda,
            frozen.model.config.nmda_regions,
            :,
        )
        memory = prediction.final_memory
    end
    return (; voltage, spike, spike_logit, nmda)
end

function _new_gate(bins::Int)
    bins >= 256 || throw(ArgumentError("AUROC bins must be >= 256"))
    return Legacy._StreamingGate(
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        0,
        zeros(Int64, bins),
        zeros(Int64, bins),
        0,
    )
end

function _slice_ragged!(
    destination::Dict{Symbol,Any},
    shard,
    selected,
    prefix::Symbol,
    fields::Tuple,
)
    offsets = Int64.(vec(_required(
        shard,
        Symbol(prefix, :_trial_offset),
    )))
    destination[Symbol(prefix, :_trial_offset)] = Int64[0]
    for field in fields
        source = vec(_required(shard, field))
        destination[field] = similar(source, 0)
    end
    for trial in selected
        source_range = _ragged_range(offsets, trial)
        for field in fields
            append!(
                destination[field],
                vec(_required(shard, field))[source_range],
            )
        end
        first_field = destination[first(fields)]
        push!(
            destination[Symbol(prefix, :_trial_offset)],
            length(first_field),
        )
    end
    return destination
end

function _compact_slice(shard, selected)
    payload = Dict{Symbol,Any}()
    _slice_ragged!(
        payload,
        shard,
        selected,
        :contact,
        (
            :contact_axon,
            :contact_segment,
            :contact_location_slot,
            :contact_section,
            :contact_x,
            :contact_path_distance_um,
            :contact_kind,
            :contact_strength,
        ),
    )
    _slice_ragged!(
        payload,
        shard,
        selected,
        :event,
        (:event_axon, :event_time_bin, :event_count),
    )
    for name in (
        :axon_kind,
        :contact_count_per_axon,
    )
        array = _required(shard, name)
        payload[name] = copy(@view array[:, selected])
    end
    for name in (
        :rate_window_ms,
        :rate_sigma_ms,
        :rate_mean_hz,
        :rate_std_hz,
    )
        payload[name] = copy(vec(_required(shard, name))[selected])
    end
    return (;
        contact_trial_offset=payload[:contact_trial_offset],
        contact_axon=Int32.(payload[:contact_axon]),
        contact_segment=Int32.(payload[:contact_segment]),
        contact_location_slot=
            Int32.(payload[:contact_location_slot]),
        contact_section=Int32.(payload[:contact_section]),
        contact_x=Float32.(payload[:contact_x]),
        contact_path_distance_um=
            Float32.(payload[:contact_path_distance_um]),
        contact_kind=UInt8.(payload[:contact_kind]),
        contact_strength=Float32.(payload[:contact_strength]),
        event_trial_offset=payload[:event_trial_offset],
        event_axon=Int32.(payload[:event_axon]),
        event_time_bin=Int32.(payload[:event_time_bin]),
        event_count=UInt8.(payload[:event_count]),
        axon_kind=UInt8.(payload[:axon_kind]),
        contact_count_per_axon=
            Int32.(payload[:contact_count_per_axon]),
        rate_window_ms=Float32.(payload[:rate_window_ms]),
        rate_sigma_ms=Float32.(payload[:rate_sigma_ms]),
        rate_mean_hz=Float32.(payload[:rate_mean_hz]),
        rate_std_hz=Float32.(payload[:rate_std_hz]),
    )
end

function _config_hash(config)
    logical = (;
        schema=RELEASE_DATASET_SCHEMA,
        validation_samples=config.validation_samples,
        validation_hash_seed=config.validation_hash_seed,
        maximum_train_samples=config.maximum_train_samples,
        maximum_validation_samples=config.maximum_validation_samples,
        maximum_test_samples=config.maximum_test_samples,
        time_chunk=config.time_chunk,
        output_shard_samples=config.output_shard_samples,
        minimum_twin_spike_auroc=config.minimum_twin_spike_auroc,
        auroc_histogram_bins=config.auroc_histogram_bins,
        require_full_public_counts=config.require_full_public_counts,
        selected_dendritic_segments=
            config.selected_dendritic_segments,
    )
    return _sha256_text(JSON3.write(logical))
end

function _target_schema(time_steps, diagnostic_times)
    return (;
        target_voltage=(;
            provenance="actual frozen PaperDigitalTwin inference",
            axes=("time_1ms", "trial"),
            size_time=time_steps,
        ),
        target_spike=(;
            provenance="actual frozen PaperDigitalTwin probability",
            axes=("time_1ms", "trial"),
            size_time=time_steps,
        ),
        target_spike_logit=(;
            provenance="actual frozen PaperDigitalTwin logit",
            axes=("time_1ms", "trial"),
            size_time=time_steps,
        ),
        target_nmda=(;
            provenance="actual frozen PaperDigitalTwin inference",
            axes=("region", "time_1ms", "trial"),
            regions=4,
            size_time=time_steps,
        ),
        target_calcium_event=(;
            provenance="official detailed Hay/NEURON teacher only",
            axes=("diagnostic_time", "trial"),
            diagnostic_time_indices_zero_based=diagnostic_times,
        ),
        target_dendritic_voltage=(;
            provenance="official detailed Hay/NEURON teacher only",
            axes=("selected_dendrite", "diagnostic_time", "trial"),
            selected_dendrites=4,
            diagnostic_time_indices_zero_based=diagnostic_times,
        ),
    )
end

function prepare_distillation_dataset_release(
    config::ReleaseStreamingPrepareConfig,
)
    config.time_chunk >= 1 ||
        throw(ArgumentError("time_chunk must be positive"))
    config.output_shard_samples >= 1 ||
        throw(ArgumentError("output_shard_samples must be positive"))
    config.auroc_histogram_bins >= 256 ||
        throw(ArgumentError("auroc_histogram_bins must be >= 256"))
    for (name, value) in (
        ("maximum_train_samples", config.maximum_train_samples),
        (
            "maximum_validation_samples",
            config.maximum_validation_samples,
        ),
        ("maximum_test_samples", config.maximum_test_samples),
    )
        value >= 1 || throw(ArgumentError("$name must be positive"))
    end
    destination = abspath(config.output_directory)
    ispath(destination) &&
        error("output directory already exists: $destination")
    staging = destination * ".staging." * string(getpid())
    ispath(staging) &&
        error("staging directory already exists: $staging")

    base_config = _base_config(config)
    source, source_counts, segment_catalog_sha256 =
        _load_release_source(config)
    frozen, integrity_before, reported_metrics, twin_file_hash =
        BaseBridge._verify_twin(base_config, source)
    plan_data = _stream_plan(config, source)
    frozen.model.config.nmda_regions == 4 ||
        error("release distillation requires four NMDA regions")
    frozen.model.config.segments ==
        Int(_required(source.manifest, :total_segments)) ||
        error("frozen twin/official segment count differs")
    total_samples = sum(
        length(item.selected) for item in plan_data.plan
    )
    full_output_counts = (
        plan_data.split_counts[BaseBridge.TRAIN_SPLIT],
        plan_data.split_counts[BaseBridge.VALIDATION_SPLIT],
        plan_data.split_counts[BaseBridge.TEST_SPLIT],
    )
    promotion_eligible =
        config.require_full_public_counts &&
        source_counts.train_pool == PAPER_TRAIN_POOL &&
        source_counts.held_out_test == PAPER_HELD_OUT_TEST &&
        full_output_counts ==
            (
                PAPER_TRAIN_POOL - config.validation_samples,
                config.validation_samples,
                PAPER_HELD_OUT_TEST,
            )
    if config.require_full_public_counts && !promotion_eligible
        error(
            "promotable release must preserve all 50k train-pool and " *
            "2k held-out trials after deterministic validation derivation",
        )
    end

    config_sha256 = _config_hash(config)
    gate = _new_gate(config.auroc_histogram_bins)
    shard_records = NamedTuple[]
    train_indices = Int32[]
    validation_indices = Int32[]
    test_indices = Int32[]
    output_shard_index = 0
    global_output_index = 0
    selected_segments = Int[]
    diagnostic_segments = Int[]
    diagnostic_times = Int32[]
    mkpath(staging)
    try
        for source_plan in plan_data.plan
            source_shard = _read_final_shard(source_plan.path)
            layout = _validate_final_shard(source_shard)
            for first_selected in
                1:config.output_shard_samples:length(source_plan.selected)
                last_selected = min(
                    first_selected +
                    config.output_shard_samples - 1,
                    length(source_plan.selected),
                )
                group_range = first_selected:last_selected
                selected_trials =
                    source_plan.selected[group_range]
                split_code =
                    source_plan.split_code[group_range]
                source_ids =
                    source_plan.source_ids[group_range]
                trial_count = length(selected_trials)
                output_shard_index += 1
                voltage = Matrix{Float32}(
                    undef,
                    plan_data.time_steps,
                    trial_count,
                )
                spike = similar(voltage)
                spike_logit = similar(voltage)
                nmda = Array{Float32,3}(
                    undef,
                    4,
                    plan_data.time_steps,
                    trial_count,
                )
                chosen_segments,
                selected_rows,
                shard_diagnostic_segments =
                    BaseBridge._diagnostic_selection(
                        base_config,
                        source,
                        source_shard,
                    )
                if isempty(selected_segments)
                    selected_segments = chosen_segments
                    diagnostic_segments =
                        shard_diagnostic_segments
                    diagnostic_times = layout.diagnostic_times
                elseif selected_segments != chosen_segments ||
                       diagnostic_segments !=
                           shard_diagnostic_segments ||
                       diagnostic_times != layout.diagnostic_times
                    error("diagnostic mapping changed across shards")
                end
                calcium, dendritic =
                    BaseBridge._detailed_internal_targets(
                        base_config,
                        source_shard,
                        selected_trials,
                        selected_segments,
                        selected_rows,
                    )
                for (output_trial, source_trial) in
                    enumerate(selected_trials)
                    sparse = _sparse_sample(
                        source_shard,
                        source_trial,
                        frozen.model.config,
                    )
                    prediction = _infer_sample(
                        frozen,
                        sparse,
                        plan_data.time_steps,
                        config.time_chunk,
                    )
                    voltage[:, output_trial] .= prediction.voltage
                    spike[:, output_trial] .= prediction.spike
                    spike_logit[:, output_trial] .=
                        prediction.spike_logit
                    nmda[:, :, output_trial] .= prediction.nmda
                    if split_code[output_trial] ==
                       BaseBridge.TEST_SPLIT
                        Legacy._update_gate!(
                            gate,
                            prediction,
                            @view(
                                _required(
                                    source_shard,
                                    :target_voltage,
                                )[:, source_trial]
                            ),
                            @view(
                                _required(
                                    source_shard,
                                    :target_spike,
                                )[:, source_trial]
                            ),
                            @view(
                                _required(
                                    source_shard,
                                    :target_nmda,
                                )[:, :, source_trial]
                            ),
                        )
                    end
                end
                compact =
                    _compact_slice(source_shard, selected_trials)
                first_global = global_output_index + 1
                global_output_index += trial_count
                last_global = global_output_index
                global_indices =
                    Int32.(first_global:last_global)
                for (index, code) in
                    zip(global_indices, split_code)
                    if code == BaseBridge.TRAIN_SPLIT
                        push!(train_indices, index)
                    elseif code == BaseBridge.VALIDATION_SPLIT
                        push!(validation_indices, index)
                    else
                        push!(test_indices, index)
                    end
                end
                shard_name =
                    "distillation_release_" *
                    lpad(output_shard_index, 6, '0') *
                    ".jld2"
                shard_path = joinpath(staging, shard_name)
                dataset = (;
                    schema=RELEASE_SHARD_SCHEMA,
                    input_representation=
                        "compact_ragged_contact_event_location_v2",
                    input=nothing,
                    compact...,
                    target_voltage=voltage,
                    target_spike=spike,
                    target_spike_logit=spike_logit,
                    target_nmda=nmda,
                    target_calcium_event=calcium,
                    target_dendritic_voltage=dendritic,
                    split_code,
                    source_split_code=
                        layout.split_code[selected_trials],
                    source_sample_indices=source_ids,
                    global_output_indices=global_indices,
                    diagnostic_segment_indices=
                        Int32.(diagnostic_segments),
                    diagnostic_time_indices=
                        diagnostic_times,
                    selected_dendritic_segments=
                        Int32.(selected_segments),
                    mixed_supervision=true,
                    digital_twin_gate_passed=true,
                    frozen_twin_parameter_hash=
                        frozen.parameter_sha256,
                    frozen_twin_artifact_hash=
                        frozen.artifact_sha256,
                    frozen_twin_file_sha256=twin_file_hash,
                    detailed_teacher_hash=
                        source.detailed_teacher_hash,
                    detailed_kernel_hash=
                        source.detailed_kernel_hash,
                    morphology_hash=source.morphology_hash,
                    official_modeldb_source_hash=
                        source.modeldb_hash,
                    segment_catalog_sha256,
                    dt_ms=frozen.model.config.dt_ms,
                    time_steps=plan_data.time_steps,
                    source_teacher_schema=FINAL_NEURON_SCHEMA,
                    config_sha256,
                )
                Legacy._atomic_shard(shard_path, dataset)
                shard_hash = BaseBridge._sha256_file(shard_path)
                push!(
                    shard_records,
                    (;
                        path=shard_name,
                        sha256=shard_hash,
                        bytes=filesize(shard_path),
                        samples=trial_count,
                        global_first=first_global,
                        global_last=last_global,
                        split_counts=(;
                            train=count(
                                ==(BaseBridge.TRAIN_SPLIT),
                                split_code,
                            ),
                            validation=count(
                                ==(BaseBridge.VALIDATION_SPLIT),
                                split_code,
                            ),
                            test=count(
                                ==(BaseBridge.TEST_SPLIT),
                                split_code,
                            ),
                        ),
                    ),
                )
            end
        end
        global_output_index == total_samples ||
            error("streaming sample count mismatch")
        recomputed_gate = Legacy._finish_gate(gate)
        recomputed_gate.spike_auroc >=
            config.minimum_twin_spike_auroc || error(
            "frozen twin failed stream-recomputed held-out gate: " *
            "spike AUROC $(recomputed_gate.spike_auroc) < " *
            "$(config.minimum_twin_spike_auroc)",
        )
        integrity_after = assert_frozen_unchanged(
            frozen;
            expected_artifact_sha256=
                integrity_before.artifact_sha256,
        )
        integrity_after.max_delta == 0 ||
            error("frozen twin changed during dataset preparation")
        segment_region =
            BaseBridge._segment_regions(source, :official_neuron)
        input_compartment, input_receptor, input_plane =
            BaseBridge._input_anatomy(frozen.model.config)
        split_counts = (;
            train=length(train_indices),
            validation=length(validation_indices),
            test=length(test_indices),
        )
        source_hashes = _required(source.manifest, :source_hashes)
        final_generator_sha256 = String(_value(
            source_hashes,
            :final_generator_source_sha256,
            "",
        ))
        hashes = (;
            official_modeldb_source_hash=source.modeldb_hash,
            detailed_teacher_hash=source.detailed_teacher_hash,
            detailed_kernel_hash=source.detailed_kernel_hash,
            morphology_hash=source.morphology_hash,
            final_generator_sha256,
            segment_catalog_sha256,
            frozen_twin_parameter_hash=frozen.parameter_sha256,
            frozen_twin_artifact_hash=frozen.artifact_sha256,
            frozen_twin_file_sha256=twin_file_hash,
            source_dataset_hash=source.source_dataset_hash,
            source_manifest_sha256=source.source_manifest_hash,
            config_sha256,
        )
        target_schema =
            _target_schema(plan_data.time_steps, diagnostic_times)
        manifest = (;
            schema=RELEASE_DATASET_SCHEMA,
            shard_schema=RELEASE_SHARD_SCHEMA,
            model_name=HD_SWSNN_TWINPROP_NAME,
            completion_state="complete",
            promotion_eligible,
            created_at_utc=Dates.format(
                now(UTC),
                dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
            ),
            source_kind="official_neuron_final_v2",
            source_teacher_schema=FINAL_NEURON_SCHEMA,
            official_neuron_schema=FINAL_NEURON_SCHEMA,
            source_completion_state=source.completion_state,
            source_public_counts=(;
                train_pool=PAPER_TRAIN_POOL,
                held_out_test=PAPER_HELD_OUT_TEST,
                duration_ms=PAPER_DURATION_MS,
            ),
            observed_source_counts=source_counts,
            validation_derivation=(;
                algorithm="lowest SHA-256(seed:source_sample_id)",
                seed=config.validation_hash_seed,
                requested=config.validation_samples,
                selected=length(plan_data.validation_ids),
                source_train_pool=plan_data.source_train_pool,
                selected_source_sample_ids=
                    plan_data.validation_ids,
                selected_ids_sha256=
                    plan_data.validation_digest,
            ),
            input_representation=
                "compact_ragged_contact_event_location_v2",
            input_layout=twin_input_layout(frozen),
            input_compartment,
            input_receptor,
            input_plane,
            time_steps=plan_data.time_steps,
            diagnostic_time_indices,
            total_samples,
            split_counts,
            train_indices,
            validation_indices,
            test_indices,
            time_chunk=config.time_chunk,
            output_shard_samples=config.output_shard_samples,
            peak_dense_chunk_bytes=
                frozen.model.config.input_dim *
                min(config.time_chunk, plan_data.time_steps) *
                sizeof(Float32),
            dense_memory_scales_with_total_samples=false,
            mixed_supervision=true,
            mixed_supervision_provenance=(;
                target_voltage=
                    "actual frozen PaperDigitalTwin inference",
                target_spike=
                    "actual frozen PaperDigitalTwin probability",
                target_spike_logit=
                    "actual frozen PaperDigitalTwin logit",
                target_nmda=
                    "actual frozen PaperDigitalTwin inference",
                target_calcium_event=
                    "official detailed Hay/NEURON teacher only",
                target_dendritic_voltage=
                    "official detailed Hay/NEURON teacher only",
            ),
            target_schema,
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
            segment_catalog_sha256,
            teacher_hash=source.detailed_teacher_hash,
            detailed_teacher_hash=source.detailed_teacher_hash,
            detailed_kernel_hash=source.detailed_kernel_hash,
            cell_mechanism_sha256=source.detailed_kernel_hash,
            morphology_hash=source.morphology_hash,
            official_modeldb_source_hash=source.modeldb_hash,
            frozen_twin_parameter_hash=frozen.parameter_sha256,
            frozen_twin_artifact_hash=frozen.artifact_sha256,
            frozen_twin_file_sha256=twin_file_hash,
            digital_twin_hash=frozen.artifact_sha256,
            source_dataset_hash=source.source_dataset_hash,
            source_manifest_sha256=source.source_manifest_hash,
            config_sha256,
            hashes,
            shards=shard_records,
        )
        manifest_path = joinpath(staging, "manifest.json")
        Legacy._json_write(manifest_path, manifest)
        manifest_sha256 =
            BaseBridge._sha256_file(manifest_path)
        mv(staging, destination)
        return (;
            schema=RELEASE_DATASET_SCHEMA,
            output_directory=destination,
            manifest_path=joinpath(destination, "manifest.json"),
            manifest_sha256,
            total_samples,
            shard_count=length(shard_records),
            split_counts,
            promotion_eligible,
            peak_dense_chunk_bytes=
                manifest.peak_dense_chunk_bytes,
            dense_memory_scales_with_total_samples=false,
            recomputed_twin_gate,
            digital_twin_gate_passed=true,
            frozen_max_delta_before=integrity_before.max_delta,
            frozen_max_delta_after=integrity_after.max_delta,
        )
    catch
        ispath(staging) && rm(staging; recursive=true, force=true)
        rethrow()
    end
end

function _parse_arguments(arguments)
    options = Dict{String,String}(
        "dataset" => get(ENV, "HD_TWINPROP_TEACHER_PATH", ""),
        "frozen-twin" => get(ENV, "HD_TWINPROP_TWIN_PATH", ""),
        "output-directory" => get(
            ENV,
            "HD_TWINPROP_DISTILL_DATASET_DIR",
            joinpath(
                @__DIR__,
                "artifacts",
                "distillation_dataset_release",
            ),
        ),
        "validation-samples" => get(
            ENV,
            "HD_TWINPROP_VALIDATION_SAMPLES",
            string(PAPER_VALIDATION),
        ),
        "validation-seed" => get(
            ENV,
            "HD_TWINPROP_VALIDATION_SEED",
            "HD-SWSNN-TwinProp/final-v2/validation-v1",
        ),
        "max-train" => get(
            ENV,
            "HD_TWINPROP_MAX_TRAIN",
            string(typemax(Int)),
        ),
        "max-validation" => get(
            ENV,
            "HD_TWINPROP_MAX_VALIDATION",
            string(typemax(Int)),
        ),
        "max-test" => get(
            ENV,
            "HD_TWINPROP_MAX_TEST",
            string(typemax(Int)),
        ),
        "time-chunk" => get(ENV, "HD_TWINPROP_TIME_CHUNK", "256"),
        "output-shard-samples" => get(
            ENV,
            "HD_TWINPROP_OUTPUT_SHARD_SAMPLES",
            "2",
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
        "require-full-public-counts" => get(
            ENV,
            "HD_TWINPROP_REQUIRE_FULL_PUBLIC_COUNTS",
            "true",
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
    require_full = lowercase(
        options["require-full-public-counts"],
    ) in ("true", "1", "yes")
    return ReleaseStreamingPrepareConfig(
        dataset_path=abspath(options["dataset"]),
        frozen_twin_path=abspath(options["frozen-twin"]),
        output_directory=abspath(options["output-directory"]),
        validation_samples=parse(
            Int,
            options["validation-samples"],
        ),
        validation_hash_seed=options["validation-seed"],
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
        require_full_public_counts=require_full,
    )
end

function main(arguments=ARGS)
    report = prepare_distillation_dataset_release(
        _parse_arguments(arguments),
    )
    println(JSON3.write(report))
    return report
end

end # module DistillationDatasetBridgeReleaseV6

if abspath(PROGRAM_FILE) == @__FILE__
    DistillationDatasetBridgeReleaseV6.main()
end
