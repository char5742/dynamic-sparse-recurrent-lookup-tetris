module DistillationDatasetBridgeFinal

using Dates
using JLD2
using JSON3
using NPZ
using SHA

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(_PARENT_MODULE, :PaperHayCell)
    Base.include(_PARENT_MODULE, joinpath(@__DIR__, "PaperHayCell.jl"))
end
if !isdefined(_PARENT_MODULE, :PaperDigitalTwin)
    Base.include(_PARENT_MODULE, joinpath(@__DIR__, "PaperDigitalTwin.jl"))
end
if !isdefined(_PARENT_MODULE, :TwinDatasetGeneration)
    Base.include(_PARENT_MODULE, joinpath(@__DIR__, "generate_twin_dataset.jl"))
end

using ..PaperHayCell
using ..PaperDigitalTwin
using ..TwinDatasetGeneration: expand_compact_twin_input

export PREPARED_DATASET_SCHEMA,
    OFFICIAL_NEURON_SCHEMA,
    PrepareDistillationConfig,
    canonical_morphology_sha256,
    prepare_distillation_dataset,
    main

const PREPARED_DATASET_SCHEMA =
    "hd-swsnn-twinprop-distillation-dataset-v1"
const OFFICIAL_NEURON_SCHEMA =
    "hd_swsnn_twinprop.neuron_teacher.v1"
const TRAIN_SPLIT = UInt8(1)
const VALIDATION_SPLIT = UInt8(2)
const TEST_SPLIT = UInt8(3)

"""
Verified bridge from the detailed teacher and a *frozen* PaperDigitalTwin to
the final 11-state-cell supervision schema.

The official path accepts only `hd_swsnn_twinprop.neuron_teacher.v1`.
`:canonical_julia` remains an explicitly labelled fallback for development;
it is never relabelled as the official NEURON source.
"""
Base.@kwdef struct PrepareDistillationConfig
    dataset_path::String
    frozen_twin_path::String
    output_path::String
    source_kind::Symbol = :official_neuron
    maximum_train_samples::Int = typemax(Int)
    maximum_validation_samples::Int = typemax(Int)
    maximum_test_samples::Int = typemax(Int)
    twin_batch_size::Int = 8
    minimum_twin_spike_auroc::Float64 = 0.985
    expected_source_dataset_sha256::String = ""
    expected_modeldb_source_sha256::String = ""
    expected_detailed_teacher_sha256::String = ""
    expected_detailed_kernel_sha256::String = ""
    expected_morphology_sha256::String = ""
    expected_twin_parameter_sha256::String = ""
    expected_twin_artifact_sha256::String = ""
    selected_dendritic_segments::Vector{Int} = Int[]
end

@inline function _value(object, name::Symbol, default=nothing)
    if object isa AbstractDict
        return get(object, name, get(object, String(name), default))
    elseif object !== nothing && hasproperty(object, name)
        return getproperty(object, name)
    end
    return default
end

function _first_value(object, names::Tuple, default=nothing)
    for name in names
        result = _value(object, name)
        result === nothing || return result
    end
    return default
end

function _sha256_file(path::AbstractString)
    source = abspath(path)
    isfile(source) || error("file is absent: $source")
    return bytes2hex(SHA.sha256(read(source)))
end

function _sha256_strings(values...)
    digest = SHA.SHA2_256_CTX()
    for value in values
        bytes = codeunits(String(value))
        SHA.update!(digest, bytes)
        SHA.update!(digest, UInt8[0x00])
    end
    return bytes2hex(SHA.digest!(digest))
end

function _require_digest(label, value)
    result = lowercase(String(value))
    occursin(r"^[0-9a-f]{64}$", result) ||
        error("$label is not a SHA-256 digest: $value")
    return result
end

function _match_expected(label, actual, expected)
    isempty(expected) && return actual
    lowercase(String(actual)) == lowercase(String(expected)) ||
        error("$label hash mismatch: expected $expected, got $actual")
    return actual
end

function _tree_sha256(tree::HayTree)
    digest = SHA.SHA2_256_CTX()
    for array in (
        tree.parent,
        tree.region,
        tree.distance_um,
        tree.area_um2,
        tree.axial_conductance_ns,
    )
        SHA.update!(digest, reinterpret(UInt8, vec(array)))
    end
    return bytes2hex(SHA.digest!(digest))
end

canonical_morphology_sha256() = _tree_sha256(paper_hay_tree())

struct _SourceLineage
    root::String
    manifest_path::String
    manifest::Any
    shard_paths::Vector{String}
    shard_sha256::Vector{String}
    source_dataset_sha256::String
    source_manifest_sha256::String
    source_teacher_schema::String
    detailed_teacher_sha256::String
    detailed_kernel_sha256::String
    morphology_sha256::String
    modeldb_source_sha256::String
end

function _source_hashes(manifest, source_kind::Symbol)
    if source_kind === :official_neuron
        source_hashes = _value(manifest, :source_hashes)
        source_hashes === nothing &&
            error("official manifest has no source_hashes")
        detailed_teacher = _require_digest(
            "teacher contract",
            _value(manifest, :teacher_contract_sha256, ""),
        )
        detailed_kernel = _require_digest(
            "mechanism library",
            _value(source_hashes, :mechanism_library_sha256, ""),
        )
        morphology = _require_digest(
            "official morphology",
            _value(source_hashes, :morphology_sha256, ""),
        )
        modeldb = _require_digest(
            "ModelDB tracked tree",
            _value(source_hashes, :modeldb_tree_sha256, ""),
        )
        return detailed_teacher, detailed_kernel, morphology, modeldb
    elseif source_kind === :canonical_julia
        canonical_source =
            _sha256_file(joinpath(@__DIR__, "PaperHayCell.jl"))
        detailed_teacher = _require_digest(
            "canonical detailed teacher",
            _first_value(
                manifest,
                (
                    :teacher_hash,
                    :detailed_teacher_hash,
                    :cell_mechanism_sha256,
                ),
                canonical_source,
            ),
        )
        detailed_kernel = _require_digest(
            "canonical detailed kernel",
            _first_value(
                manifest,
                (:cell_mechanism_sha256, :detailed_kernel_hash),
                canonical_source,
            ),
        )
        morphology = _require_digest(
            "canonical morphology",
            _first_value(
                manifest,
                (:morphology_sha256, :morphology_hash),
                canonical_morphology_sha256(),
            ),
        )
        modeldb = _require_digest(
            "canonical ModelDB-derived source",
            _first_value(
                manifest,
                (
                    :official_modeldb_source_hash,
                    :modeldb_source_sha256,
                ),
                canonical_source,
            ),
        )
        return detailed_teacher, detailed_kernel, morphology, modeldb
    end
    error("source_kind must be :official_neuron or :canonical_julia")
end

function _load_source(config::PrepareDistillationConfig)
    root = abspath(config.dataset_path)
    isdir(root) ||
        error("detailed teacher dataset must be a manifest directory: $root")
    manifest_path = joinpath(root, "manifest.json")
    isfile(manifest_path) ||
        error("detailed teacher has no manifest.json: $root")
    manifest = JSON3.read(read(manifest_path, String))
    source_schema = String(_first_value(
        manifest,
        (:schema_name, :source_teacher_schema),
        "",
    ))
    if config.source_kind === :official_neuron
        source_schema == OFFICIAL_NEURON_SCHEMA || error(
            "official source schema mismatch: expected " *
            "$OFFICIAL_NEURON_SCHEMA, got $(repr(source_schema))",
        )
        _value(manifest, :completion_state, "complete") == "complete" ||
            error("official NEURON teacher manifest is not complete")
        _value(manifest, :modeldb_source_modified_by_generator, false) ===
            false ||
            error("official ModelDB source was modified by the generator")
    elseif config.source_kind === :canonical_julia
        source_schema == OFFICIAL_NEURON_SCHEMA && error(
            "an official NEURON manifest cannot be relabelled canonical",
        )
        isempty(source_schema) &&
            (source_schema = "canonical_julia.PaperHayCell.v1")
    else
        error("unsupported source_kind $(config.source_kind)")
    end

    records = _value(manifest, :shards)
    records === nothing && error("teacher manifest has no shards")
    shard_paths = String[]
    shard_hashes = String[]
    for record in records
        relative = String(_value(record, :path, ""))
        isempty(relative) && error("teacher shard record has no path")
        path = abspath(joinpath(root, relative))
        isfile(path) || error("teacher shard is absent: $path")
        declared = _require_digest(
            "declared shard",
            _value(record, :sha256, ""),
        )
        actual = _sha256_file(path)
        actual == declared || error(
            "teacher shard hash mismatch for $relative: " *
            "expected $declared, got $actual",
        )
        push!(shard_paths, path)
        push!(shard_hashes, actual)
    end
    isempty(shard_paths) && error("teacher manifest contains no shards")
    source_manifest_sha256 = _sha256_file(manifest_path)
    source_dataset_sha256 = _sha256_strings(
        source_manifest_sha256,
        shard_hashes...,
    )
    _match_expected(
        "source dataset",
        source_dataset_sha256,
        config.expected_source_dataset_sha256,
    )
    detailed_teacher, detailed_kernel, morphology, modeldb =
        _source_hashes(manifest, config.source_kind)
    _match_expected(
        "detailed teacher",
        detailed_teacher,
        config.expected_detailed_teacher_sha256,
    )
    _match_expected(
        "detailed kernel",
        detailed_kernel,
        config.expected_detailed_kernel_sha256,
    )
    _match_expected(
        "morphology",
        morphology,
        config.expected_morphology_sha256,
    )
    _match_expected(
        "ModelDB source",
        modeldb,
        config.expected_modeldb_source_sha256,
    )
    return _SourceLineage(
        root,
        manifest_path,
        manifest,
        shard_paths,
        shard_hashes,
        source_dataset_sha256,
        source_manifest_sha256,
        source_schema,
        detailed_teacher,
        detailed_kernel,
        morphology,
        modeldb,
    )
end

function _held_out_twin_metrics(frozen)
    metrics = _first_value(
        frozen.metadata,
        (:held_out_test, :test_metrics, :held_out_metrics),
    )
    metrics === nothing && error(
        "frozen twin lacks held-out metrics; its fidelity gate " *
        "cannot be verified",
    )
    return metrics
end

function _verify_frozen_twin(
    config::PrepareDistillationConfig,
    source::_SourceLineage,
)
    path = abspath(config.frozen_twin_path)
    isfile(path) || error("frozen PaperDigitalTwin is absent: $path")
    frozen = load_frozen_twin(path)
    before = assert_frozen_unchanged(frozen)
    before.max_delta == 0.0f0 ||
        error("digital twin was not frozen before inference")
    _match_expected(
        "digital twin parameter",
        frozen.parameter_sha256,
        config.expected_twin_parameter_sha256,
    )
    _match_expected(
        "digital twin artifact",
        frozen.artifact_sha256,
        config.expected_twin_artifact_sha256,
    )
    metadata = frozen.metadata
    twin_teacher = String(_first_value(
        metadata,
        (:detailed_teacher_hash, :teacher_hash),
        "",
    ))
    twin_kernel = String(_first_value(
        metadata,
        (
            :cell_mechanism_sha256,
            :detailed_kernel_hash,
            :mechanism_library_sha256,
        ),
        "",
    ))
    twin_morphology = String(_first_value(
        metadata,
        (:morphology_sha256, :morphology_hash),
        "",
    ))
    lowercase(twin_teacher) == source.detailed_teacher_sha256 || error(
        "digital twin teacher hash differs from detailed dataset",
    )
    lowercase(twin_kernel) == source.detailed_kernel_sha256 || error(
        "digital twin detailed-kernel hash differs from dataset",
    )
    lowercase(twin_morphology) == source.morphology_sha256 || error(
        "digital twin morphology hash differs from dataset",
    )
    metrics = _held_out_twin_metrics(frozen)
    spike_auroc = Float64(_value(metrics, :spike_auroc, NaN))
    isfinite(spike_auroc) ||
        error("digital twin held-out spike AUROC is missing")
    spike_auroc >= config.minimum_twin_spike_auroc || error(
        "digital twin is below gate: spike AUROC $spike_auroc < " *
        "$(config.minimum_twin_spike_auroc)",
    )
    explicit_gate = _value(metadata, :gate)
    if explicit_gate !== nothing &&
       _value(explicit_gate, :passed, false) !== true
        error("digital twin metadata records a failed gate")
    end
    return frozen, before, metrics, _sha256_file(path)
end

function _read_shard(path::AbstractString)
    extension = lowercase(splitext(path)[2])
    extension == ".npz" && return NPZ.npzread(path)
    extension == ".jld2" && return JLD2.load(path)
    error("unsupported teacher shard extension $extension")
end

function _unwrap_shard(shard)
    for key in (:dataset, :payload)
        child = _value(shard, key)
        child === nothing || return child
    end
    return shard
end

function _split_codes(shard)
    split = _value(shard, :split_code)
    split === nothing &&
        error("teacher shard has no split_code")
    result = UInt8.(vec(split))
    all(code -> code in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT),
        result) || error("teacher shard has an unknown split code")
    return result
end

function _sample_indices(shard, count::Int)
    indices = _value(shard, :sample_indices)
    indices === nothing && return collect(Int32(1):Int32(count))
    length(indices) == count ||
        throw(DimensionMismatch("source sample_indices length differs"))
    return Int32.(vec(indices))
end

function _time_steps(shard)
    voltage = _value(shard, :target_voltage)
    voltage === nothing &&
        error("teacher shard has no target_voltage")
    ndims(voltage) == 2 ||
        throw(DimensionMismatch("target_voltage must be time x trial"))
    return size(voltage, 1)
end

function _sample_plan(
    source::_SourceLineage,
    config::PrepareDistillationConfig,
)
    limit = Dict(
        TRAIN_SPLIT => config.maximum_train_samples,
        VALIDATION_SPLIT => config.maximum_validation_samples,
        TEST_SPLIT => config.maximum_test_samples,
    )
    used = Dict(
        TRAIN_SPLIT => 0,
        VALIDATION_SPLIT => 0,
        TEST_SPLIT => 0,
    )
    plan = NamedTuple[]
    common_time = 0
    for path in source.shard_paths
        shard = _unwrap_shard(_read_shard(path))
        codes = _split_codes(shard)
        source_ids = _sample_indices(shard, length(codes))
        chosen = Int[]
        chosen_codes = UInt8[]
        chosen_source_ids = Int32[]
        for (local, code) in enumerate(codes)
            used[code] >= limit[code] && continue
            used[code] += 1
            push!(chosen, local)
            push!(chosen_codes, code)
            push!(chosen_source_ids, source_ids[local])
        end
        isempty(chosen) && continue
        steps = _time_steps(shard)
        common_time == 0 && (common_time = steps)
        steps == common_time ||
            error("teacher shards have different trajectory lengths")
        push!(
            plan,
            (;
                path,
                indices=chosen,
                split_code=chosen_codes,
                source_sample_indices=chosen_source_ids,
            ),
        )
    end
    isempty(plan) && error("sample limits selected no teacher trajectory")
    for code in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
        used[code] > 0 ||
            error("prepared dataset requires all three non-empty splits")
    end
    return plan, common_time, used
end

function _validate_contact_arrays(shard)
    contact_axon = _value(shard, :contact_axon)
    contact_segment = _first_value(
        shard,
        (:contact_segment, :segment),
    )
    contact_kind = _first_value(shard, (:contact_kind, :kind))
    contact_strength = _first_value(
        shard,
        (:contact_strength, :strength),
    )
    for (name, array) in (
        ("contact_axon", contact_axon),
        ("contact_segment", contact_segment),
        ("contact_kind", contact_kind),
        ("contact_strength", contact_strength),
    )
        array === nothing && error("official shard has no $name")
        ndims(array) == 2 ||
            throw(DimensionMismatch("$name must be contact x trial"))
    end
    size(contact_axon) == size(contact_segment) ==
        size(contact_kind) == size(contact_strength) ||
        throw(DimensionMismatch("official contact arrays differ"))
    all(>=(1), contact_axon) ||
        error("contact_axon must be one-based and positive")
    all(>=(1), contact_segment) ||
        error("contact_segment must be one-based and positive")
    all(kind -> kind in (1, 2), contact_kind) ||
        error("contact_kind violates fixed E/I types")
    all(value -> 0 <= value <= 1, contact_strength) ||
        error("contact strengths violate nonnegative bounded conductance")
    return (
        contact_axon,
        contact_segment,
        contact_kind,
        contact_strength,
    )
end

function _event_times_by_axon(shard, trial::Int, time_steps::Int)
    offsets = _value(shard, :event_trial_offset)
    axons = _value(shard, :event_axon)
    bins = _value(shard, :event_time_bin)
    for (name, value) in (
        ("event_trial_offset", offsets),
        ("event_axon", axons),
        ("event_time_bin", bins),
    )
        value === nothing && error("official sparse shard has no $name")
    end
    length(offsets) >= trial + 1 ||
        throw(DimensionMismatch("event_trial_offset is too short"))
    first_event = Int(offsets[trial]) + 1
    last_event = Int(offsets[trial + 1])
    result = Dict{Int,Vector{Int}}()
    for event in first_event:last_event
        axon = Int(axons[event])
        time = Int(bins[event]) + 1
        1 <= time <= time_steps ||
            error("zero-based event_time_bin is outside trajectory")
        push!(get!(result, axon, Int[]), time)
    end
    return result
end

function _official_dense_input(
    shard,
    indices,
    twin_config,
    total_segments::Int,
)
    twin_config.segments == total_segments || error(
        "official full-segment input has $total_segments segments, " *
        "but frozen twin expects $(twin_config.segments). A diagnostic-only " *
        "remap is forbidden unless an explicit manifest mapping is supplied.",
    )
    contact_axon, contact_segment, contact_kind, contact_strength =
        _validate_contact_arrays(shard)
    time_steps = _time_steps(shard)
    batch = length(indices)
    event_plane =
        zeros(Float32, total_segments, 3, time_steps, batch)
    strength_plane = similar(event_plane)
    dense_contact_events = _value(shard, :event_spike)
    if dense_contact_events !== nothing
        size(dense_contact_events, 1) == size(contact_axon, 1) ||
            throw(DimensionMismatch("dense event/contact count differs"))
        size(dense_contact_events, 2) == time_steps ||
            throw(DimensionMismatch("dense event time differs"))
    end

    @inbounds for (output_trial, source_trial) in enumerate(indices)
        event_times = dense_contact_events === nothing ?
            _event_times_by_axon(shard, source_trial, time_steps) :
            nothing
        for contact in axes(contact_axon, 1)
            segment = Int(contact_segment[contact, source_trial])
            1 <= segment <= total_segments ||
                error("contact segment is outside official morphology")
            kind = UInt8(contact_kind[contact, source_trial])
            strength = Float32(contact_strength[contact, source_trial])
            receptors = kind == UInt8(1) ? (1, 2) : (3,)
            for receptor in receptors, time in 1:time_steps
                strength_plane[
                    segment,
                    receptor,
                    time,
                    output_trial,
                ] += strength
            end
            if dense_contact_events === nothing
                times = get(
                    event_times,
                    Int(contact_axon[contact, source_trial]),
                    Int[],
                )
                for receptor in receptors, time in times
                    event_plane[
                        segment,
                        receptor,
                        time,
                        output_trial,
                    ] += strength
                end
            else
                for time in 1:time_steps
                    dense_contact_events[
                        contact,
                        time,
                        source_trial,
                    ] == 0 && continue
                    for receptor in receptors
                        event_plane[
                            segment,
                            receptor,
                            time,
                            output_trial,
                        ] += strength
                    end
                end
            end
        end
    end
    event_plane .= min.(event_plane, 1.0f0)
    strength_plane .= min.(strength_plane, 1.0f0)
    return flatten_twin_input(event_plane, strength_plane)
end

function _canonical_dense_input(shard, indices, twin_config)
    stored = _value(shard, :input)
    if stored !== nothing
        dense = Float32.(@view stored[:, :, indices])
    else
        for required in (
            :contact_segment,
            :contact_kind,
            :contact_strength,
            :event_spike,
        )
            _value(shard, required) === nothing &&
                error("canonical compact shard has no $required")
        end
        all_dense = expand_compact_twin_input(
            _value(shard, :contact_segment),
            _value(shard, :contact_kind),
            _value(shard, :contact_strength),
            Bool.(_value(shard, :event_spike)),
            twin_config,
        )
        dense = Float32.(@view all_dense[:, :, indices])
    end
    size(dense, 1) == twin_config.input_dim ||
        throw(DimensionMismatch("canonical dense input/twin mismatch"))
    return dense
end

function _dense_input(
    source::_SourceLineage,
    config::PrepareDistillationConfig,
    shard,
    indices,
    twin_config,
)
    if config.source_kind === :official_neuron
        total_segments = Int(_value(source.manifest, :total_segments, 0))
        total_segments >= 1 ||
            error("official manifest has no total_segments")
        return _official_dense_input(
            shard,
            indices,
            twin_config,
            total_segments,
        )
    end
    return _canonical_dense_input(shard, indices, twin_config)
end

function _catalog_record(source::_SourceLineage, segment::Int)
    records = _value(source.manifest, :segments)
    records === nothing && return nothing
    for record in records
        Int(_value(record, :index, -1)) == segment && return record
    end
    return nothing
end

function _select_by_region(records, region_name, mode::Symbol)
    candidates = filter(records) do item
        lowercase(String(item.region)) == region_name
    end
    isempty(candidates) &&
        error("diagnostic catalog has no $region_name compartment")
    if mode === :minimum
        return candidates[argmin(getproperty.(candidates, :distance))]
    elseif mode === :maximum
        return candidates[argmax(getproperty.(candidates, :distance))]
    elseif mode === :hot
        return candidates[
            argmin(abs.(getproperty.(candidates, :distance) .- 785.0))
        ]
    end
    error("unknown diagnostic selection mode")
end

function _diagnostic_selection(
    source::_SourceLineage,
    config::PrepareDistillationConfig,
    shard,
)
    diagnostic = _first_value(
        shard,
        (:diagnostic_segment_indices, :target_compartment),
    )
    diagnostic === nothing && (diagnostic = _value(
        source.manifest,
        :diagnostic_segment_indices,
    ))
    diagnostic === nothing &&
        error("detailed teacher has no diagnostic segment indices")
    segment_ids = Int.(vec(diagnostic))
    if !isempty(config.selected_dendritic_segments)
        selected_ids = copy(config.selected_dendritic_segments)
    elseif config.source_kind === :official_neuron
        records = NamedTuple[]
        for segment in segment_ids
            record = _catalog_record(source, segment)
            record === nothing &&
                error("diagnostic segment is absent from manifest catalog")
            push!(
                records,
                (;
                    segment,
                    region=String(_value(record, :region_name, "")),
                    distance=Float64(_value(record, :distance_um, NaN)),
                ),
            )
        end
        selected_ids = Int[
            _select_by_region(records, "basal", :maximum).segment,
            _select_by_region(records, "apical_trunk", :minimum).segment,
            _select_by_region(records, "apical_trunk", :hot).segment,
            _select_by_region(records, "apical_tuft", :maximum).segment,
        ]
    else
        metadata = _value(shard, :metadata, source.manifest)
        region = _value(metadata, :compartment_region)
        distance = _value(metadata, :compartment_distance_um)
        if region === nothing || distance === nothing
            tree = paper_hay_tree()
            region = tree.region
            distance = tree.distance_um
        end
        candidates = NamedTuple[]
        for segment in segment_ids
            region_name = UInt8(region[segment]) == BASAL ? "basal" :
                UInt8(region[segment]) == APICAL_TRUNK ?
                "apical_trunk" :
                UInt8(region[segment]) == APICAL_TUFT ?
                "apical_tuft" : "soma"
            push!(
                candidates,
                (;
                    segment,
                    region=region_name,
                    distance=Float64(distance[segment]),
                ),
            )
        end
        selected_ids = Int[
            _select_by_region(candidates, "basal", :maximum).segment,
            _select_by_region(
                candidates,
                "apical_trunk",
                :minimum,
            ).segment,
            _select_by_region(candidates, "apical_trunk", :hot).segment,
            _select_by_region(
                candidates,
                "apical_tuft",
                :maximum,
            ).segment,
        ]
    end
    length(selected_ids) == 4 ||
        error("exactly four dendritic segments must be selected")
    length(unique(selected_ids)) == 4 ||
        error("selected dendritic segments must be distinct")
    rows = Int[]
    for segment in selected_ids
        row = findfirst(==(segment), segment_ids)
        row === nothing &&
            error("selected segment $segment is not a recorded diagnostic")
        push!(rows, row)
    end
    return selected_ids, rows, segment_ids
end

function _canonical_replay_targets(
    shard,
    indices,
    selected_segments,
)
    segment = _value(shard, :contact_segment)
    kind = _value(shard, :contact_kind)
    strength = _value(shard, :contact_strength)
    event = _value(shard, :event_spike)
    for (name, array) in (
        ("contact_segment", segment),
        ("contact_kind", kind),
        ("contact_strength", strength),
        ("event_spike", event),
    )
        array === nothing &&
            error("canonical Ca replay requires $name")
    end
    tree = paper_hay_tree()
    parameters = HayParameters(tree; ablation=:full)
    state = HayState(tree, parameters)
    drive = HaySynapticDrive(tree)
    diagnostics = HayDiagnostics(tree)
    time_steps = size(event, 2)
    batch = length(indices)
    calcium_event = zeros(Float32, time_steps, batch)
    voltage = Array{Float32,3}(undef, 4, time_steps, batch)
    @inbounds for (output_trial, source_trial) in enumerate(indices)
        reset_state!(state, parameters)
        for time in 1:time_steps
            reset_drive!(drive)
            for contact in axes(segment, 1)
                event[contact, time, source_trial] || continue
                target = Int(segment[contact, source_trial])
                amplitude = Float32(strength[contact, source_trial])
                if UInt8(kind[contact, source_trial]) == UInt8(1)
                    add_synaptic_event!(
                        drive,
                        target;
                        ampa=amplitude,
                        nmda=amplitude,
                    )
                else
                    add_synaptic_event!(
                        drive,
                        target;
                        gaba=amplitude,
                    )
                end
            end
            hay_cell_step!(
                state,
                drive,
                diagnostics,
                tree,
                parameters,
            )
            calcium_event[time, output_trial] =
                maximum(@view state.local_ca_event[2:end])
            for selected in 1:4
                voltage[selected, time, output_trial] =
                    state.voltage_mv[selected_segments[selected]]
            end
        end
    end
    return calcium_event, voltage
end

function _detailed_only_targets(
    source::_SourceLineage,
    config::PrepareDistillationConfig,
    shard,
    indices,
    selected_segments,
    selected_rows,
)
    compartment_voltage = _first_value(
        shard,
        (
            :target_compartment_voltage,
            :target_dendritic_voltage,
        ),
    )
    calcium = _first_value(
        shard,
        (
            :target_ca_event,
            :target_dendritic_ca_event,
            :target_calcium_event,
        ),
    )
    if compartment_voltage !== nothing
        if size(compartment_voltage, 1) == 4 &&
           _value(shard, :diagnostic_segment_indices) === nothing
            dendritic =
                Float32.(@view compartment_voltage[:, :, indices])
        else
            dendritic = Float32.(@view(
                compartment_voltage[selected_rows, :, indices],
            ))
        end
    else
        dendritic = nothing
    end

    if calcium !== nothing
        if ndims(calcium) == 2
            calcium_event = Float32.(@view calcium[:, indices])
        elseif ndims(calcium) == 3
            calcium_event = Float32.(dropdims(
                maximum(@view(calcium[:, :, indices]); dims=1);
                dims=1,
            ))
        else
            throw(DimensionMismatch("detailed Ca event has invalid rank"))
        end
    else
        calcium_event = nothing
    end

    if calcium_event === nothing || dendritic === nothing
        config.source_kind === :canonical_julia || error(
            "official NEURON shard is missing detailed-only Ca/voltage targets",
        )
        replay_ca, replay_voltage = _canonical_replay_targets(
            shard,
            indices,
            selected_segments,
        )
        calcium_event === nothing && (calcium_event = replay_ca)
        dendritic === nothing && (dendritic = replay_voltage)
    end
    return calcium_event, dendritic
end

function _run_twin!(
    voltage,
    spike_probability,
    spike_logit,
    nmda,
    frozen,
    input,
    destination,
    batch_size,
)
    batch = size(input, 3)
    for first_local in 1:batch_size:batch
        last_local = min(first_local + batch_size - 1, batch)
        local = first_local:last_local
        first_destination = first(destination) + first_local - 1
        last_destination = first(destination) + last_local - 1
        target = first_destination:last_destination
        prediction = twin_forward(frozen, @view(input[:, :, local]))
        voltage[:, target] .= prediction.voltage
        spike_probability[:, target] .= prediction.spike_probability
        spike_logit[:, target] .= prediction.spike_logit
        nmda[:, :, target] .= prediction.nmda
    end
    return nothing
end

function _input_anatomy(twin_config)
    compartment = Int[]
    receptor = Int[]
    plane = Int[]
    for input_plane in 1:twin_config.input_planes
        for input_receptor in 1:twin_config.receptors
            for segment in 1:twin_config.segments
                push!(compartment, segment)
                push!(receptor, input_receptor)
                push!(plane, input_plane)
            end
        end
    end
    length(compartment) == twin_config.input_dim ||
        error("internal twin input anatomy mismatch")
    return compartment, receptor, plane
end

function _config_hash(config::PrepareDistillationConfig)
    logical = (;
        schema=PREPARED_DATASET_SCHEMA,
        source_kind=String(config.source_kind),
        maximum_train_samples=config.maximum_train_samples,
        maximum_validation_samples=config.maximum_validation_samples,
        maximum_test_samples=config.maximum_test_samples,
        twin_batch_size=config.twin_batch_size,
        minimum_twin_spike_auroc=config.minimum_twin_spike_auroc,
        selected_dendritic_segments=config.selected_dendritic_segments,
        mixed_supervision=true,
    )
    return bytes2hex(SHA.sha256(codeunits(JSON3.write(logical))))
end

function _atomic_save(path::AbstractString, dataset)
    destination = abspath(path)
    mkpath(dirname(destination))
    temporary = destination * ".tmp." * string(getpid())
    JLD2.jldsave(temporary; dataset)
    mv(temporary, destination; force=true)
    return destination
end

"""
Prepare the final mixed-supervision dataset.

Cached voltage, spike probability/logit and NMDA targets are generated by
actual calls to the frozen twin.  Ca events and selected dendritic voltages
come only from the detailed teacher.  The twin is hash-checked before and
after inference and must have a held-out spike AUROC at or above the gate.
"""
function prepare_distillation_dataset(
    config::PrepareDistillationConfig,
)
    config.twin_batch_size >= 1 ||
        throw(ArgumentError("twin_batch_size must be positive"))
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
    source = _load_source(config)
    frozen, integrity_before, twin_metrics, twin_file_sha256 =
        _verify_frozen_twin(config, source)
    plan, time_steps, split_counts = _sample_plan(source, config)
    samples = sum(length(item.indices) for item in plan)
    input = Array{Float32}(
        undef,
        frozen.model.config.input_dim,
        time_steps,
        samples,
    )
    target_voltage = Matrix{Float32}(undef, time_steps, samples)
    target_spike = Matrix{Float32}(undef, time_steps, samples)
    target_spike_logit = Matrix{Float32}(undef, time_steps, samples)
    frozen.model.config.nmda_regions == 4 ||
        error("final distillation requires four NMDA regions")
    target_nmda = Array{Float32,3}(undef, 4, time_steps, samples)
    target_calcium_event =
        Matrix{Float32}(undef, time_steps, samples)
    target_dendritic_voltage =
        Array{Float32,3}(undef, 4, time_steps, samples)
    split_code = Vector{UInt8}(undef, samples)
    source_sample_indices = Vector{Int32}(undef, samples)
    selected_segments = Int[]
    diagnostic_segments = Int[]

    cursor = 1
    for item in plan
        shard = _unwrap_shard(_read_shard(item.path))
        local_input = _dense_input(
            source,
            config,
            shard,
            item.indices,
            frozen.model.config,
        )
        local_selected, selected_rows, local_diagnostic =
            _diagnostic_selection(source, config, shard)
        if isempty(selected_segments)
            selected_segments = local_selected
            diagnostic_segments = local_diagnostic
        elseif selected_segments != local_selected ||
               diagnostic_segments != local_diagnostic
            error("detailed diagnostic segment mapping changed across shards")
        end
        calcium_event, dendritic_voltage =
            _detailed_only_targets(
                source,
                config,
                shard,
                item.indices,
                selected_segments,
                selected_rows,
            )
        local_count = length(item.indices)
        destination = cursor:(cursor + local_count - 1)
        input[:, :, destination] .= local_input
        _run_twin!(
            target_voltage,
            target_spike,
            target_spike_logit,
            target_nmda,
            frozen,
            local_input,
            destination,
            config.twin_batch_size,
        )
        target_calcium_event[:, destination] .= calcium_event
        target_dendritic_voltage[:, :, destination] .= dendritic_voltage
        split_code[destination] .= item.split_code
        source_sample_indices[destination] .= item.source_sample_indices
        cursor += local_count
    end

    integrity_after = assert_frozen_unchanged(
        frozen;
        expected_artifact_sha256=integrity_before.artifact_sha256,
    )
    integrity_after.max_delta == 0.0f0 ||
        error("frozen twin changed while preparing distillation data")
    integrity_after.parameter_sha256 ==
        integrity_before.parameter_sha256 ||
        error("frozen twin parameter hash changed during inference")
    integrity_after.artifact_sha256 ==
        integrity_before.artifact_sha256 ||
        error("frozen twin artifact hash changed during inference")

    train_indices = findall(==(TRAIN_SPLIT), split_code)
    validation_indices = findall(==(VALIDATION_SPLIT), split_code)
    test_indices = findall(==(TEST_SPLIT), split_code)
    input_compartment, input_receptor, input_plane =
        _input_anatomy(frozen.model.config)
    config_sha256 = _config_hash(config)
    detailed_source_label =
        config.source_kind === :official_neuron ?
        "official NEURON Hay L5PC teacher" :
        "canonical Julia ModelDB-derived PaperHayCell fallback"
    mixed_supervision_provenance = (;
        input="verified detailed-teacher event/location protocol",
        target_voltage="actual frozen PaperDigitalTwin inference",
        target_spike=
            "actual frozen PaperDigitalTwin per-step probability",
        target_spike_logit=
            "actual frozen PaperDigitalTwin per-step logit",
        target_nmda="actual frozen PaperDigitalTwin inference",
        target_calcium_event=detailed_source_label,
        target_dendritic_voltage=detailed_source_label,
        twin_is_frozen=true,
        twin_checked_before_and_after=true,
        detailed_only_targets=(
            "target_calcium_event",
            "target_dendritic_voltage",
        ),
    )
    hashes = (;
        official_modeldb_source_hash=source.modeldb_source_sha256,
        detailed_teacher_hash=source.detailed_teacher_sha256,
        detailed_kernel_hash=source.detailed_kernel_sha256,
        morphology_hash=source.morphology_sha256,
        frozen_twin_parameter_hash=frozen.parameter_sha256,
        frozen_twin_artifact_hash=frozen.artifact_sha256,
        frozen_twin_file_sha256=twin_file_sha256,
        source_dataset_hash=source.source_dataset_sha256,
        source_manifest_sha256=source.source_manifest_sha256,
        config_sha256,
    )
    metadata = (;
        schema=PREPARED_DATASET_SCHEMA,
        model_name=HD_SWSNN_TWINPROP_NAME,
        stage="detailed_to_twin_to_eleven_state_dataset",
        created_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
        source_kind=String(config.source_kind),
        source_teacher_schema=source.source_teacher_schema,
        official_neuron_schema=
            config.source_kind === :official_neuron ?
            OFFICIAL_NEURON_SCHEMA : "",
        official_neuron_source=
            config.source_kind === :official_neuron,
        canonical_julia_fallback=
            config.source_kind === :canonical_julia,
        canonical_fallback_disclosure=
            config.source_kind === :canonical_julia ?
            "Not the authors' unpublished data; detailed targets use " *
            "the checked-in Julia reconstruction." :
            "",
        mixed_supervision=true,
        mixed_supervision_provenance,
        digital_twin_gate_passed=true,
        twin_held_out_metrics=twin_metrics,
        twin_gate=(;
            minimum_spike_auroc=config.minimum_twin_spike_auroc,
            passed=true,
        ),
        integrity_before,
        integrity_after,
        input_layout=twin_input_layout(frozen),
        input_compartment,
        input_receptor,
        input_plane,
        diagnostic_segment_indices=diagnostic_segments,
        selected_dendritic_segments=selected_segments,
        selected_dendritic_semantics=(
            "distal_basal",
            "proximal_apical_trunk",
            "apical_calcium_hot_zone",
            "distal_apical_tuft",
        ),
        dt_ms=frozen.model.config.dt_ms,
        target_spike_semantics="soft per-step probability",
        hashes,
        teacher_hash=source.detailed_teacher_sha256,
        detailed_teacher_hash=source.detailed_teacher_sha256,
        detailed_kernel_hash=source.detailed_kernel_sha256,
        cell_mechanism_sha256=source.detailed_kernel_sha256,
        morphology_hash=source.morphology_sha256,
        morphology_sha256=source.morphology_sha256,
        official_modeldb_source_hash=source.modeldb_source_sha256,
        frozen_twin_parameter_hash=frozen.parameter_sha256,
        frozen_twin_artifact_hash=frozen.artifact_sha256,
        digital_twin_hash=frozen.artifact_sha256,
        source_dataset_hash=source.source_dataset_sha256,
        source_manifest_sha256=source.source_manifest_sha256,
        dataset_sha256=source.source_dataset_sha256,
        config_sha256,
    )
    dataset = (;
        schema=PREPARED_DATASET_SCHEMA,
        input,
        target_voltage,
        target_spike,
        target_spike_logit,
        target_nmda,
        target_calcium_event,
        target_dendritic_voltage,
        train_indices,
        validation_indices,
        test_indices,
        split_code,
        source_sample_indices,
        diagnostic_segment_indices=diagnostic_segments,
        selected_dendritic_segments=selected_segments,
        segment_region=_value(source.manifest, :regions, nothing),
        mixed_supervision=true,
        mixed_supervision_provenance,
        digital_twin_gate_passed=true,
        source_teacher_schema=source.source_teacher_schema,
        official_neuron_schema=
            config.source_kind === :official_neuron ?
            OFFICIAL_NEURON_SCHEMA : "",
        teacher_hash=source.detailed_teacher_sha256,
        detailed_teacher_hash=source.detailed_teacher_sha256,
        detailed_kernel_hash=source.detailed_kernel_sha256,
        cell_mechanism_sha256=source.detailed_kernel_sha256,
        morphology_hash=source.morphology_sha256,
        morphology_sha256=source.morphology_sha256,
        official_modeldb_source_hash=source.modeldb_source_sha256,
        frozen_twin_parameter_hash=frozen.parameter_sha256,
        frozen_twin_artifact_hash=frozen.artifact_sha256,
        digital_twin_hash=frozen.artifact_sha256,
        source_dataset_hash=source.source_dataset_sha256,
        source_manifest_sha256=source.source_manifest_sha256,
        dataset_sha256=source.source_dataset_sha256,
        config_sha256,
        dt_ms=frozen.model.config.dt_ms,
        metadata,
    )
    output_path = _atomic_save(config.output_path, dataset)
    output_file_sha256 = _sha256_file(output_path)
    reloaded = JLD2.load(output_path)
    haskey(reloaded, "dataset") ||
        error("prepared dataset failed reload verification")
    saved = reloaded["dataset"]
    _value(saved, :schema) == PREPARED_DATASET_SCHEMA ||
        error("prepared dataset schema changed during save")
    _value(saved, :frozen_twin_artifact_hash) ==
        frozen.artifact_sha256 ||
        error("prepared dataset lost frozen-twin lineage")
    return (;
        schema=PREPARED_DATASET_SCHEMA,
        output_path,
        output_file_sha256,
        samples,
        time_steps,
        input_dim=size(input, 1),
        split_counts=(;
            train=split_counts[TRAIN_SPLIT],
            validation=split_counts[VALIDATION_SPLIT],
            test=split_counts[TEST_SPLIT],
        ),
        selected_dendritic_segments=selected_segments,
        hashes,
        mixed_supervision=true,
        digital_twin_gate_passed=true,
        frozen_max_delta_before=integrity_before.max_delta,
        frozen_max_delta_after=integrity_after.max_delta,
    )
end

function _parse_arguments(arguments)
    options = Dict{String,String}(
        "dataset" => get(ENV, "HD_TWINPROP_TEACHER_PATH", ""),
        "frozen-twin" => get(ENV, "HD_TWINPROP_TWIN_PATH", ""),
        "output" => get(
            ENV,
            "HD_TWINPROP_DISTILL_DATASET",
            joinpath(@__DIR__, "artifacts", "distillation_dataset.jld2"),
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
        "twin-batch" => get(ENV, "HD_TWINPROP_TWIN_BATCH", "8"),
        "minimum-twin-spike-auroc" => get(
            ENV,
            "HD_TWINPROP_MIN_TWIN_AUROC",
            "0.985",
        ),
        "expected-source-dataset-sha256" => "",
        "expected-modeldb-source-sha256" => "",
        "expected-detailed-teacher-sha256" => "",
        "expected-detailed-kernel-sha256" => "",
        "expected-morphology-sha256" => "",
        "expected-twin-parameter-sha256" => "",
        "expected-twin-artifact-sha256" => "",
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
    isempty(options["dataset"]) && error(
        "--dataset or HD_TWINPROP_TEACHER_PATH is required",
    )
    isempty(options["frozen-twin"]) && error(
        "--frozen-twin or HD_TWINPROP_TWIN_PATH is required",
    )
    source_kind = Symbol(replace(
        lowercase(options["source-kind"]),
        "-" => "_",
    ))
    return PrepareDistillationConfig(
        dataset_path=abspath(options["dataset"]),
        frozen_twin_path=abspath(options["frozen-twin"]),
        output_path=abspath(options["output"]),
        source_kind,
        maximum_train_samples=parse(Int, options["max-train"]),
        maximum_validation_samples=
            parse(Int, options["max-validation"]),
        maximum_test_samples=parse(Int, options["max-test"]),
        twin_batch_size=parse(Int, options["twin-batch"]),
        minimum_twin_spike_auroc=parse(
            Float64,
            options["minimum-twin-spike-auroc"],
        ),
        expected_source_dataset_sha256=
            options["expected-source-dataset-sha256"],
        expected_modeldb_source_sha256=
            options["expected-modeldb-source-sha256"],
        expected_detailed_teacher_sha256=
            options["expected-detailed-teacher-sha256"],
        expected_detailed_kernel_sha256=
            options["expected-detailed-kernel-sha256"],
        expected_morphology_sha256=
            options["expected-morphology-sha256"],
        expected_twin_parameter_sha256=
            options["expected-twin-parameter-sha256"],
        expected_twin_artifact_sha256=
            options["expected-twin-artifact-sha256"],
    )
end

function main(arguments=ARGS)
    report = prepare_distillation_dataset(_parse_arguments(arguments))
    println(JSON3.write(report))
    return report
end

end # module DistillationDatasetBridgeFinal

if abspath(PROGRAM_FILE) == @__FILE__
    DistillationDatasetBridgeFinal.main()
end
