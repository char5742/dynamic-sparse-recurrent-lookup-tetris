module DistillationDatasetBridgeProductionV4

using Dates
using JLD2
using JSON3
using NPZ
using SHA

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(_PARENT_MODULE, :PaperDigitalTwin)
    Base.include(_PARENT_MODULE, joinpath(@__DIR__, "PaperDigitalTwin.jl"))
end
if !isdefined(_PARENT_MODULE, :PaperHayCell)
    Base.include(_PARENT_MODULE, joinpath(@__DIR__, "PaperHayCell.jl"))
end

using ..PaperDigitalTwin
using ..PaperHayCell

export PREPARED_DATASET_SCHEMA,
    OFFICIAL_NEURON_SCHEMA,
    PrepareDistillationConfig,
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
Standalone production bridge for HD-SWSNN-TwinProp.

The output targets have deliberately mixed provenance:

* soma voltage, soma spike probability/logit and regional NMDA current are
  recomputed by actual inference through a frozen `PaperDigitalTwin`;
* local Ca events and four selected dendritic voltages come only from the
  detailed teacher.

Official input is accepted only from the exact NEURON teacher schema.  A
canonical Julia fallback remains available but is explicitly labelled and
cannot be passed off as official NEURON data.
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
        SHA.update!(digest, codeunits(String(value)))
        SHA.update!(digest, UInt8[0])
    end
    return bytes2hex(SHA.digest!(digest))
end

function _require_hash(label, value)
    result = lowercase(String(value))
    occursin(r"^[0-9a-f]{64}$", result) ||
        error("$label is not a SHA-256 digest: $value")
    return result
end

function _expected_hash(label, actual, expected)
    isempty(expected) && return actual
    lowercase(String(actual)) == lowercase(String(expected)) ||
        error("$label hash mismatch: expected $expected, got $actual")
    return actual
end

function _canonical_tree_hash()
    tree = paper_hay_tree()
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

struct _Source
    root::String
    manifest_path::String
    manifest::Any
    shard_paths::Vector{String}
    source_dataset_hash::String
    source_manifest_hash::String
    source_schema::String
    completion_state::String
    detailed_teacher_hash::String
    detailed_kernel_hash::String
    morphology_hash::String
    modeldb_hash::String
end

function _official_hashes(manifest)
    source_hashes = _value(manifest, :source_hashes)
    source_hashes === nothing &&
        error("official teacher manifest has no source_hashes")
    return (
        _require_hash(
            "teacher contract",
            _value(manifest, :teacher_contract_sha256, ""),
        ),
        _require_hash(
            "NEURON mechanism library",
            _value(source_hashes, :mechanism_library_sha256, ""),
        ),
        _require_hash(
            "official morphology",
            _value(source_hashes, :morphology_sha256, ""),
        ),
        _require_hash(
            "ModelDB tracked tree",
            _value(source_hashes, :modeldb_tree_sha256, ""),
        ),
    )
end

function _canonical_hashes(manifest)
    source_hash = _sha256_file(joinpath(@__DIR__, "PaperHayCell.jl"))
    return (
        _require_hash(
            "canonical teacher",
            _first_value(
                manifest,
                (
                    :teacher_hash,
                    :detailed_teacher_hash,
                    :cell_mechanism_sha256,
                ),
                source_hash,
            ),
        ),
        _require_hash(
            "canonical kernel",
            _first_value(
                manifest,
                (:cell_mechanism_sha256, :detailed_kernel_hash),
                source_hash,
            ),
        ),
        _require_hash(
            "canonical morphology",
            _first_value(
                manifest,
                (:morphology_sha256, :morphology_hash),
                _canonical_tree_hash(),
            ),
        ),
        _require_hash(
            "canonical ModelDB-derived source",
            _first_value(
                manifest,
                (
                    :modeldb_source_sha256,
                    :official_modeldb_source_hash,
                ),
                source_hash,
            ),
        ),
    )
end

function _load_source(config::PrepareDistillationConfig)
    root = abspath(config.dataset_path)
    isdir(root) ||
        error("teacher dataset must be a manifest directory: $root")
    manifest_path = joinpath(root, "manifest.json")
    isfile(manifest_path) ||
        error("teacher dataset has no manifest.json")
    manifest = JSON3.read(read(manifest_path, String))
    source_schema = String(_first_value(
        manifest,
        (:schema_name, :source_teacher_schema),
        "",
    ))
    completion = String(_value(manifest, :completion_state, "complete"))
    hashes = if config.source_kind === :official_neuron
        source_schema == OFFICIAL_NEURON_SCHEMA || error(
            "official teacher schema mismatch: expected " *
            "$OFFICIAL_NEURON_SCHEMA, got $(repr(source_schema))",
        )
        completion == "complete" ||
            error("official teacher generation is not complete")
        _value(manifest, :modeldb_source_modified_by_generator, false) ===
            false ||
            error("official ModelDB checkout was modified by the generator")
        _official_hashes(manifest)
    elseif config.source_kind === :canonical_julia
        source_schema == OFFICIAL_NEURON_SCHEMA && error(
            "official NEURON data cannot be relabelled canonical Julia",
        )
        isempty(source_schema) &&
            (source_schema = "canonical_julia.PaperHayCell.v1")
        _canonical_hashes(manifest)
    else
        error("source_kind must be :official_neuron or :canonical_julia")
    end
    detailed_teacher, detailed_kernel, morphology, modeldb = hashes
    _expected_hash(
        "detailed teacher",
        detailed_teacher,
        config.expected_detailed_teacher_sha256,
    )
    _expected_hash(
        "detailed kernel",
        detailed_kernel,
        config.expected_detailed_kernel_sha256,
    )
    _expected_hash(
        "morphology",
        morphology,
        config.expected_morphology_sha256,
    )
    _expected_hash(
        "ModelDB source",
        modeldb,
        config.expected_modeldb_source_sha256,
    )

    records = _value(manifest, :shards)
    records === nothing && error("teacher manifest has no shards")
    shard_paths = String[]
    shard_hashes = String[]
    for record in records
        relative = String(_value(record, :path, ""))
        isempty(relative) && error("teacher shard record has no path")
        path = abspath(joinpath(root, relative))
        isfile(path) || error("teacher shard is absent: $path")
        declared = _require_hash(
            "declared teacher shard",
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
    isempty(shard_paths) && error("teacher manifest has no shard files")
    manifest_hash = _sha256_file(manifest_path)
    dataset_hash = _sha256_strings(manifest_hash, shard_hashes...)
    _expected_hash(
        "source dataset",
        dataset_hash,
        config.expected_source_dataset_sha256,
    )
    return _Source(
        root,
        manifest_path,
        manifest,
        shard_paths,
        dataset_hash,
        manifest_hash,
        source_schema,
        completion,
        detailed_teacher,
        detailed_kernel,
        morphology,
        modeldb,
    )
end

function _verify_twin(config, source)
    path = abspath(config.frozen_twin_path)
    isfile(path) || error("frozen PaperDigitalTwin is absent: $path")
    frozen = load_frozen_twin(path)
    integrity = assert_frozen_unchanged(frozen)
    integrity.max_delta == 0 ||
        error("digital twin is not frozen")
    _expected_hash(
        "twin parameter",
        frozen.parameter_sha256,
        config.expected_twin_parameter_sha256,
    )
    _expected_hash(
        "twin artifact",
        frozen.artifact_sha256,
        config.expected_twin_artifact_sha256,
    )
    metadata = frozen.metadata
    twin_teacher = lowercase(String(_first_value(
        metadata,
        (:detailed_teacher_hash, :teacher_hash),
        "",
    )))
    twin_kernel = lowercase(String(_first_value(
        metadata,
        (
            :cell_mechanism_sha256,
            :detailed_kernel_hash,
            :mechanism_library_sha256,
        ),
        "",
    )))
    twin_morphology = lowercase(String(_first_value(
        metadata,
        (:morphology_sha256, :morphology_hash),
        "",
    )))
    twin_teacher == source.detailed_teacher_hash ||
        error("twin/detailed-teacher lineage mismatch")
    twin_kernel == source.detailed_kernel_hash ||
        error("twin/detailed-kernel lineage mismatch")
    twin_morphology == source.morphology_hash ||
        error("twin/morphology lineage mismatch")
    self_report = _first_value(
        metadata,
        (:held_out_test, :test_metrics, :held_out_metrics),
        nothing,
    )
    return frozen, integrity, self_report, _sha256_file(path)
end

function _read_shard(path)
    extension = lowercase(splitext(path)[2])
    extension == ".npz" && return NPZ.npzread(path)
    extension == ".jld2" && return JLD2.load(path)
    error("unsupported teacher shard extension $extension")
end

function _unwrap(shard)
    for name in (:dataset, :payload)
        child = _value(shard, name)
        child === nothing || return child
    end
    return shard
end

function _split_codes(shard)
    values = _value(shard, :split_code)
    values === nothing && error("teacher shard has no split_code")
    result = UInt8.(vec(values))
    all(code -> code in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT),
        result) || error("teacher shard has an unknown split code")
    return result
end

function _source_sample_indices(shard, samples)
    values = _value(shard, :sample_indices)
    values === nothing && return Int32.(1:samples)
    length(values) == samples ||
        throw(DimensionMismatch("sample_indices length differs"))
    return Int32.(vec(values))
end

function _plan(config, source)
    limits = Dict(
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
        shard = _unwrap(_read_shard(path))
        codes = _split_codes(shard)
        sample_ids = _source_sample_indices(shard, length(codes))
        chosen = Int[]
        chosen_codes = UInt8[]
        chosen_ids = Int32[]
        for (local_index, code) in enumerate(codes)
            used[code] >= limits[code] && continue
            used[code] += 1
            push!(chosen, local_index)
            push!(chosen_codes, code)
            push!(chosen_ids, sample_ids[local_index])
        end
        isempty(chosen) && continue
        voltage = _value(shard, :target_voltage)
        voltage === nothing && error("teacher shard has no soma voltage")
        ndims(voltage) == 2 ||
            throw(DimensionMismatch("target_voltage must be time x trial"))
        time_steps = size(voltage, 1)
        common_time == 0 && (common_time = time_steps)
        common_time == time_steps ||
            error("teacher shards have different trajectory lengths")
        push!(
            plan,
            (;
                path,
                indices=chosen,
                split_code=chosen_codes,
                sample_ids=chosen_ids,
            ),
        )
    end
    isempty(plan) && error("sample limits selected no trajectories")
    for code in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
        used[code] > 0 ||
            error("prepared dataset requires all three splits")
    end
    return plan, common_time, used
end

function _contact_arrays(shard)
    axon = _value(shard, :contact_axon)
    segment = _first_value(shard, (:contact_segment, :segment))
    kind = _first_value(shard, (:contact_kind, :kind))
    strength = _first_value(shard, (:contact_strength, :strength))
    for (name, array) in (
        ("contact_axon", axon),
        ("contact_segment", segment),
        ("contact_kind", kind),
        ("contact_strength", strength),
    )
        array === nothing && error("teacher shard has no $name")
        ndims(array) == 2 ||
            throw(DimensionMismatch("$name must be contact x trial"))
    end
    size(axon) == size(segment) == size(kind) == size(strength) ||
        throw(DimensionMismatch("teacher contact arrays differ"))
    all(>=(1), axon) || error("contact axon must be one-based")
    all(>=(1), segment) || error("contact segment must be one-based")
    all(value -> value in (1, 2), kind) ||
        error("contact kind violates Dale-fixed E/I type")
    all(value -> 0 <= value <= 1, strength) ||
        error("contact strength violates nonnegative bounded conductance")
    return axon, segment, kind, strength
end

function _trial_event_times(shard, trial, time_steps)
    offsets = _value(shard, :event_trial_offset)
    event_axon = _value(shard, :event_axon)
    event_time = _value(shard, :event_time_bin)
    for (name, array) in (
        ("event_trial_offset", offsets),
        ("event_axon", event_axon),
        ("event_time_bin", event_time),
    )
        array === nothing && error("sparse official shard has no $name")
    end
    first_event = Int(offsets[trial]) + 1
    last_event = Int(offsets[trial + 1])
    result = Dict{Int,Vector{Int}}()
    for event in first_event:last_event
        axon = Int(event_axon[event])
        time = Int(event_time[event]) + 1
        1 <= time <= time_steps ||
            error("zero-based event time is outside trajectory")
        push!(get!(result, axon, Int[]), time)
    end
    return result
end

function _flatten_input(event_plane, strength_plane)
    # `strength_plane` must be zero-initialised by the caller.  Using
    # `similar(event_plane)` followed by `+=` is invalid and was the source
    # of a prior NaN-producing implementation.
    combined = cat(event_plane, strength_plane; dims=2)
    return reshape(
        combined,
        size(combined, 1) * size(combined, 2),
        size(combined, 3),
        size(combined, 4),
    )
end

function _dense_input(config, source, shard, selected, twin_config)
    stored = _value(shard, :input)
    if stored !== nothing
        result = Float32.(@view stored[:, :, selected])
        size(result, 1) == twin_config.input_dim ||
            throw(DimensionMismatch("stored input/twin dimension differs"))
        all(isfinite, result) ||
            error("stored twin input contains non-finite values")
        return result
    end
    axon, segment, kind, strength = _contact_arrays(shard)
    time_steps = size(_value(shard, :target_voltage), 1)
    batch = length(selected)
    total_segments = config.source_kind === :official_neuron ?
        Int(_value(source.manifest, :total_segments, 0)) :
        twin_config.segments
    total_segments == twin_config.segments || error(
        "teacher has $total_segments segments but frozen twin expects " *
        "$(twin_config.segments); implicit diagnostic remapping is forbidden",
    )
    event_plane =
        zeros(Float32, total_segments, 3, time_steps, batch)
    strength_plane =
        zeros(Float32, total_segments, 3, time_steps, batch)
    dense_events = _value(shard, :event_spike)
    if dense_events !== nothing
        size(dense_events, 1) == size(axon, 1) ||
            throw(DimensionMismatch("dense event/contact count differs"))
        size(dense_events, 2) == time_steps ||
            throw(DimensionMismatch("dense event time differs"))
    end
    @inbounds for (output_trial, source_trial) in enumerate(selected)
        sparse_times = dense_events === nothing ?
            _trial_event_times(shard, source_trial, time_steps) :
            nothing
        for contact in axes(axon, 1)
            destination = Int(segment[contact, source_trial])
            1 <= destination <= total_segments ||
                error("contact targets an absent morphology segment")
            amplitude = Float32(strength[contact, source_trial])
            receptors = UInt8(kind[contact, source_trial]) == UInt8(1) ?
                (1, 2) : (3,)
            for receptor in receptors, time in 1:time_steps
                strength_plane[
                    destination,
                    receptor,
                    time,
                    output_trial,
                ] += amplitude
            end
            if dense_events === nothing
                times = get(
                    sparse_times,
                    Int(axon[contact, source_trial]),
                    Int[],
                )
                for receptor in receptors, time in times
                    event_plane[
                        destination,
                        receptor,
                        time,
                        output_trial,
                    ] += amplitude
                end
            else
                for time in 1:time_steps
                    dense_events[contact, time, source_trial] == 0 &&
                        continue
                    for receptor in receptors
                        event_plane[
                            destination,
                            receptor,
                            time,
                            output_trial,
                        ] += amplitude
                    end
                end
            end
        end
    end
    event_plane .= min.(event_plane, 1.0f0)
    strength_plane .= min.(strength_plane, 1.0f0)
    result = _flatten_input(event_plane, strength_plane)
    all(isfinite, result) ||
        error("expanded twin input contains non-finite values")
    return result
end

function _catalog(source, segment)
    records = _value(source.manifest, :segments)
    records === nothing && return nothing
    for record in records
        Int(_value(record, :index, -1)) == segment && return record
    end
    return nothing
end

function _pick(records, region, mode)
    candidates = filter(
        record -> lowercase(String(record.region)) == region,
        records,
    )
    isempty(candidates) &&
        error("diagnostic recordings have no $region site")
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

function _diagnostic_selection(config, source, shard)
    diagnostic = _first_value(
        shard,
        (:diagnostic_segment_indices, :target_compartment),
        _value(source.manifest, :diagnostic_segment_indices),
    )
    if diagnostic === nothing
        config.source_kind === :canonical_julia ||
            error("official shard has no diagnostic segment mapping")
        compartment_voltage = _value(
            shard,
            :target_compartment_voltage,
        )
        diagnostic = collect(1:size(compartment_voltage, 1))
    end
    diagnostic = Int.(vec(diagnostic))
    selected_segments = if !isempty(config.selected_dendritic_segments)
        copy(config.selected_dendritic_segments)
    else
        records = NamedTuple[]
        if config.source_kind === :official_neuron
            for segment in diagnostic
                record = _catalog(source, segment)
                record === nothing &&
                    error("diagnostic site missing from segment catalog")
                push!(
                    records,
                    (;
                        segment,
                        region=String(
                            _value(record, :region_name, ""),
                        ),
                        distance=Float64(
                            _value(record, :distance_um, NaN),
                        ),
                    ),
                )
            end
        else
            metadata = _value(shard, :metadata, source.manifest)
            region = _value(metadata, :compartment_region)
            distance = _value(metadata, :compartment_distance_um)
            if region === nothing || distance === nothing
                tree = paper_hay_tree()
                region = tree.region
                distance = tree.distance_um
            end
            for segment in diagnostic
                code = UInt8(region[segment])
                name = code == BASAL ? "basal" :
                    code == APICAL_TRUNK ? "apical_trunk" :
                    code == APICAL_TUFT ? "apical_tuft" : "soma"
                push!(
                    records,
                    (;
                        segment,
                        region=name,
                        distance=Float64(distance[segment]),
                    ),
                )
            end
        end
        Int[
            _pick(records, "basal", :maximum).segment,
            _pick(records, "apical_trunk", :minimum).segment,
            _pick(records, "apical_trunk", :hot).segment,
            _pick(records, "apical_tuft", :maximum).segment,
        ]
    end
    length(selected_segments) == 4 ||
        error("exactly four dendritic sites are required")
    length(unique(selected_segments)) == 4 ||
        error("selected dendritic sites must be distinct")
    rows = Int[]
    for segment in selected_segments
        row = findfirst(==(segment), diagnostic)
        row === nothing &&
            error("selected dendritic site is not recorded")
        push!(rows, row)
    end
    return selected_segments, rows, diagnostic
end

function _replay_canonical(shard, selected, segments)
    axon, contact_segment, kind, strength = _contact_arrays(shard)
    event = _value(shard, :event_spike)
    event === nothing &&
        error("canonical Ca replay needs dense contact events")
    tree = paper_hay_tree()
    parameters = HayParameters(tree; ablation=:full)
    state = HayState(tree, parameters)
    drive = HaySynapticDrive(tree)
    diagnostics = HayDiagnostics(tree)
    time_steps = size(event, 2)
    calcium = zeros(Float32, time_steps, length(selected))
    voltage =
        Array{Float32,3}(undef, 4, time_steps, length(selected))
    @inbounds for (output_trial, source_trial) in enumerate(selected)
        reset_state!(state, parameters)
        for time in 1:time_steps
            reset_drive!(drive)
            for contact in axes(contact_segment, 1)
                event[contact, time, source_trial] || continue
                destination = Int(contact_segment[contact, source_trial])
                amplitude = Float32(strength[contact, source_trial])
                if UInt8(kind[contact, source_trial]) == UInt8(1)
                    add_synaptic_event!(
                        drive,
                        destination;
                        ampa=amplitude,
                        nmda=amplitude,
                    )
                else
                    add_synaptic_event!(
                        drive,
                        destination;
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
            calcium[time, output_trial] =
                maximum(@view state.local_ca_event[2:end])
            for site in 1:4
                voltage[site, time, output_trial] =
                    state.voltage_mv[segments[site]]
            end
        end
    end
    return calcium, voltage
end

function _detailed_internal_targets(
    config,
    shard,
    selected_trials,
    selected_segments,
    selected_rows,
)
    voltage_source = _first_value(
        shard,
        (:target_compartment_voltage, :target_dendritic_voltage),
    )
    calcium_source = _first_value(
        shard,
        (
            :target_ca_event,
            :target_dendritic_ca_event,
            :target_calcium_event,
        ),
    )
    voltage = if voltage_source === nothing
        nothing
    elseif size(voltage_source, 1) == 4 &&
           _value(shard, :diagnostic_segment_indices) === nothing
        Float32.(@view voltage_source[:, :, selected_trials])
    else
        Float32.(@view(
            voltage_source[selected_rows, :, selected_trials],
        ))
    end
    calcium = if calcium_source === nothing
        nothing
    elseif ndims(calcium_source) == 2
        Float32.(@view calcium_source[:, selected_trials])
    elseif ndims(calcium_source) == 3
        Float32.(dropdims(
            maximum(
                @view(calcium_source[:, :, selected_trials]);
                dims=1,
            );
            dims=1,
        ))
    else
        throw(DimensionMismatch("Ca-event target has invalid rank"))
    end
    if voltage === nothing || calcium === nothing
        config.source_kind === :canonical_julia || error(
            "official teacher lacks detailed Ca/voltage diagnostics",
        )
        replay_calcium, replay_voltage = _replay_canonical(
            shard,
            selected_trials,
            selected_segments,
        )
        calcium === nothing && (calcium = replay_calcium)
        voltage === nothing && (voltage = replay_voltage)
    end
    all(isfinite, calcium) ||
        error("detailed Ca-event target is non-finite")
    all(isfinite, voltage) ||
        error("detailed dendritic voltage target is non-finite")
    return calcium, voltage
end

function _run_twin!(
    voltage,
    spike,
    spike_logit,
    nmda,
    frozen,
    input,
    destination,
    batch_size,
)
    batch = size(input, 3)
    for first_item in 1:batch_size:batch
        last_item = min(first_item + batch_size - 1, batch)
        local_range = first_item:last_item
        target_range = (
            first(destination) + first_item - 1
        ):(
            first(destination) + last_item - 1
        )
        prediction = twin_forward(
            frozen,
            @view(input[:, :, local_range]),
        )
        all(isfinite, prediction.voltage) ||
            error("frozen twin produced non-finite soma voltage")
        all(isfinite, prediction.spike_probability) ||
            error("frozen twin produced non-finite spike probability")
        all(isfinite, prediction.nmda) ||
            error("frozen twin produced non-finite NMDA current")
        voltage[:, target_range] .= prediction.voltage
        spike[:, target_range] .= prediction.spike_probability
        spike_logit[:, target_range] .= prediction.spike_logit
        nmda[:, :, target_range] .= prediction.nmda
    end
    return nothing
end

function _binary_auroc(score, target)
    score_vector = vec(Float64.(score))
    target_vector = vec(Float64.(target))
    positive = findall(>=(0.5), target_vector)
    negative = findall(<(0.5), target_vector)
    isempty(positive) &&
        error("held-out detailed target has no spike positives")
    isempty(negative) &&
        error("held-out detailed target has no spike negatives")
    wins = 0.0
    for positive_index in positive, negative_index in negative
        wins += score_vector[positive_index] >
            score_vector[negative_index] ? 1.0 :
            score_vector[positive_index] ==
            score_vector[negative_index] ? 0.5 : 0.0
    end
    return wins / (length(positive) * length(negative))
end

function _correlation(left, right)
    x = vec(Float64.(left))
    y = vec(Float64.(right))
    x .-= sum(x) / length(x)
    y .-= sum(y) / length(y)
    denominator = sqrt(sum(abs2, x) * sum(abs2, y))
    denominator > 0 || return NaN
    return sum(x .* y) / denominator
end

function _recomputed_gate(
    predicted_voltage,
    predicted_spike,
    predicted_nmda,
    detailed_voltage,
    detailed_spike,
    detailed_nmda,
    test_indices,
    minimum_auroc,
)
    isempty(test_indices) && error("held-out test split is empty")
    prediction_v = @view predicted_voltage[:, test_indices]
    target_v = @view detailed_voltage[:, test_indices]
    prediction_s = @view predicted_spike[:, test_indices]
    target_s = @view detailed_spike[:, test_indices]
    prediction_n = @view predicted_nmda[:, :, test_indices]
    target_n = @view detailed_nmda[:, :, test_indices]
    for (name, values) in (
        ("predicted voltage", prediction_v),
        ("detailed voltage", target_v),
        ("predicted spike", prediction_s),
        ("detailed spike", target_s),
        ("predicted NMDA", prediction_n),
        ("detailed NMDA", target_n),
    )
        all(isfinite, values) ||
            error("$name contains non-finite values")
    end
    voltage_error = prediction_v .- target_v
    nmda_error = prediction_n .- target_n
    spike_auroc = _binary_auroc(prediction_s, target_s)
    probability = clamp.(prediction_s, 1.0f-7, 1.0f0 - 1.0f-7)
    spike_bce = -sum(
        target_s .* log.(probability) .+
        (1.0f0 .- target_s) .* log1p.(-probability),
    ) / length(target_s)
    result = (;
        source="recomputed against held-out detailed teacher",
        self_report_trusted=false,
        held_out_samples=length(test_indices),
        held_out_bins=length(target_s),
        voltage_rmse=sqrt(sum(abs2, voltage_error) / length(voltage_error)),
        voltage_correlation=_correlation(prediction_v, target_v),
        spike_auroc,
        spike_bce,
        spike_accuracy=sum(
            (prediction_s .>= 0.5f0) .==
            (target_s .>= 0.5f0),
        ) / length(target_s),
        nmda_rmse=sqrt(sum(abs2, nmda_error) / length(nmda_error)),
        nmda_correlation=_correlation(prediction_n, target_n),
    )
    isfinite(result.spike_auroc) ||
        error("recomputed held-out spike AUROC is non-finite")
    result.spike_auroc >= minimum_auroc || error(
        "frozen twin failed recomputed held-out gate: spike AUROC " *
        "$(result.spike_auroc) < $minimum_auroc",
    )
    return result
end

function _input_anatomy(config)
    compartment = Int[]
    receptor = Int[]
    plane = Int[]
    for input_plane in 1:config.input_planes
        for input_receptor in 1:config.receptors
            for segment in 1:config.segments
                push!(compartment, segment)
                push!(receptor, input_receptor)
                push!(plane, input_plane)
            end
        end
    end
    return compartment, receptor, plane
end

function _segment_regions(source, source_kind)
    if source_kind !== :official_neuron
        return String[]
    end
    total = Int(_value(source.manifest, :total_segments, 0))
    result = fill("", total)
    records = _value(source.manifest, :segments)
    records === nothing && error("official manifest has no segment catalog")
    for record in records
        index = Int(_value(record, :index, 0))
        1 <= index <= total ||
            error("official segment catalog index is invalid")
        result[index] = String(_value(record, :region_name, ""))
    end
    all(!isempty, result) ||
        error("official segment catalog has missing regions")
    return result
end

function _configuration_hash(config)
    logical = (;
        schema=PREPARED_DATASET_SCHEMA,
        source_kind=String(config.source_kind),
        maximum_train_samples=config.maximum_train_samples,
        maximum_validation_samples=config.maximum_validation_samples,
        maximum_test_samples=config.maximum_test_samples,
        twin_batch_size=config.twin_batch_size,
        minimum_twin_spike_auroc=config.minimum_twin_spike_auroc,
        selected_dendritic_segments=config.selected_dendritic_segments,
    )
    return bytes2hex(SHA.sha256(codeunits(JSON3.write(logical))))
end

function _atomic_save(path, dataset)
    destination = abspath(path)
    mkpath(dirname(destination))
    temporary = destination * ".publishing." * string(getpid())
    JLD2.jldsave(temporary; dataset)
    mv(temporary, destination; force=true)
    return destination
end

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
    frozen, integrity_before, reported_metrics, twin_file_hash =
        _verify_twin(config, source)
    plan, time_steps, split_counts = _plan(config, source)
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
    detailed_voltage = Matrix{Float32}(undef, time_steps, samples)
    detailed_spike = Matrix{Float32}(undef, time_steps, samples)
    detailed_nmda = Array{Float32,3}(undef, 4, time_steps, samples)
    split_code = Vector{UInt8}(undef, samples)
    source_sample_indices = Vector{Int32}(undef, samples)
    selected_segments = Int[]
    diagnostic_segments = Int[]
    cursor = 1

    for item in plan
        shard = _unwrap(_read_shard(item.path))
        local_input = _dense_input(
            config,
            source,
            shard,
            item.indices,
            frozen.model.config,
        )
        local_selected, selected_rows, local_diagnostic =
            _diagnostic_selection(config, source, shard)
        if isempty(selected_segments)
            selected_segments = local_selected
            diagnostic_segments = local_diagnostic
        elseif selected_segments != local_selected ||
               diagnostic_segments != local_diagnostic
            error("diagnostic segment mapping changed across shards")
        end
        calcium, dendritic = _detailed_internal_targets(
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
        target_calcium_event[:, destination] .= calcium
        target_dendritic_voltage[:, :, destination] .= dendritic
        detailed_voltage[:, destination] .= Float32.(
            @view _value(shard, :target_voltage)[:, item.indices]
        )
        detailed_spike[:, destination] .= Float32.(
            @view _value(shard, :target_spike)[:, item.indices]
        )
        detailed_nmda[:, :, destination] .= Float32.(
            @view _value(shard, :target_nmda)[:, :, item.indices]
        )
        split_code[destination] .= item.split_code
        source_sample_indices[destination] .= item.sample_ids
        cursor += local_count
    end
    cursor == samples + 1 || error("sample cursor mismatch")

    integrity_after = assert_frozen_unchanged(
        frozen;
        expected_artifact_sha256=integrity_before.artifact_sha256,
    )
    integrity_after.max_delta == 0 ||
        error("frozen twin changed during dataset preparation")
    integrity_after.parameter_sha256 ==
        integrity_before.parameter_sha256 ||
        error("frozen twin parameter hash changed")
    integrity_after.artifact_sha256 ==
        integrity_before.artifact_sha256 ||
        error("frozen twin artifact hash changed")

    train_indices = findall(==(TRAIN_SPLIT), split_code)
    validation_indices = findall(==(VALIDATION_SPLIT), split_code)
    test_indices = findall(==(TEST_SPLIT), split_code)
    recomputed_twin_gate = _recomputed_gate(
        target_voltage,
        target_spike,
        target_nmda,
        detailed_voltage,
        detailed_spike,
        detailed_nmda,
        test_indices,
        config.minimum_twin_spike_auroc,
    )
    input_compartment, input_receptor, input_plane =
        _input_anatomy(frozen.model.config)
    segment_region = _segment_regions(source, config.source_kind)
    config_hash = _configuration_hash(config)
    detailed_label = config.source_kind === :official_neuron ?
        "official NEURON Hay L5PC teacher" :
        "canonical Julia PaperHayCell fallback"
    mixed_provenance = (;
        input="verified detailed-teacher event/location protocol",
        target_voltage="actual frozen PaperDigitalTwin inference",
        target_spike="actual frozen PaperDigitalTwin probability",
        target_spike_logit="actual frozen PaperDigitalTwin logit",
        target_nmda="actual frozen PaperDigitalTwin inference",
        target_calcium_event=detailed_label,
        target_dendritic_voltage=detailed_label,
        twin_is_frozen=true,
        twin_checked_before_and_after=true,
    )
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
    metadata = (;
        schema=PREPARED_DATASET_SCHEMA,
        model_name=HD_SWSNN_TWINPROP_NAME,
        stage="detailed_to_twin_to_eleven_state_dataset",
        created_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
        source_kind=String(config.source_kind),
        source_teacher_schema=source.source_schema,
        official_neuron_schema=
            config.source_kind === :official_neuron ?
            OFFICIAL_NEURON_SCHEMA : "",
        source_completion_state=source.completion_state,
        official_neuron_source=
            config.source_kind === :official_neuron,
        canonical_julia_fallback=
            config.source_kind === :canonical_julia,
        mixed_supervision=true,
        mixed_supervision_provenance=mixed_provenance,
        digital_twin_gate_passed=true,
        twin_self_report_trusted=false,
        twin_artifact_reported_metrics=reported_metrics,
        recomputed_twin_gate,
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
        segment_region,
        dt_ms=frozen.model.config.dt_ms,
        hashes,
        teacher_hash=source.detailed_teacher_hash,
        detailed_teacher_hash=source.detailed_teacher_hash,
        detailed_kernel_hash=source.detailed_kernel_hash,
        cell_mechanism_sha256=source.detailed_kernel_hash,
        morphology_hash=source.morphology_hash,
        morphology_sha256=source.morphology_hash,
        official_modeldb_source_hash=source.modeldb_hash,
        frozen_twin_parameter_hash=frozen.parameter_sha256,
        frozen_twin_artifact_hash=frozen.artifact_sha256,
        digital_twin_hash=frozen.artifact_sha256,
        source_dataset_hash=source.source_dataset_hash,
        source_manifest_sha256=source.source_manifest_hash,
        dataset_sha256=source.source_dataset_hash,
        config_sha256=config_hash,
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
        segment_region,
        mixed_supervision=true,
        mixed_supervision_provenance=mixed_provenance,
        digital_twin_gate_passed=true,
        recomputed_twin_gate,
        source_teacher_schema=source.source_schema,
        official_neuron_schema=
            config.source_kind === :official_neuron ?
            OFFICIAL_NEURON_SCHEMA : "",
        source_completion_state=source.completion_state,
        teacher_hash=source.detailed_teacher_hash,
        detailed_teacher_hash=source.detailed_teacher_hash,
        detailed_kernel_hash=source.detailed_kernel_hash,
        cell_mechanism_sha256=source.detailed_kernel_hash,
        morphology_hash=source.morphology_hash,
        morphology_sha256=source.morphology_hash,
        official_modeldb_source_hash=source.modeldb_hash,
        frozen_twin_parameter_hash=frozen.parameter_sha256,
        frozen_twin_artifact_hash=frozen.artifact_sha256,
        digital_twin_hash=frozen.artifact_sha256,
        source_dataset_hash=source.source_dataset_hash,
        source_manifest_sha256=source.source_manifest_hash,
        dataset_sha256=source.source_dataset_hash,
        config_sha256=config_hash,
        dt_ms=frozen.model.config.dt_ms,
        metadata,
    )
    output_path = _atomic_save(config.output_path, dataset)
    saved = JLD2.load(output_path)
    haskey(saved, "dataset") ||
        error("prepared dataset failed reload verification")
    saved_dataset = saved["dataset"]
    _value(saved_dataset, :schema) == PREPARED_DATASET_SCHEMA ||
        error("prepared dataset schema changed during save")
    _value(saved_dataset, :frozen_twin_artifact_hash) ==
        frozen.artifact_sha256 ||
        error("prepared dataset lost twin lineage")
    return (;
        schema=PREPARED_DATASET_SCHEMA,
        output_path,
        output_file_sha256=_sha256_file(output_path),
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
        recomputed_twin_gate,
        mixed_supervision=true,
        digital_twin_gate_passed=true,
        twin_self_report_trusted=false,
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
    isempty(options["dataset"]) &&
        error("--dataset or HD_TWINPROP_TEACHER_PATH is required")
    isempty(options["frozen-twin"]) &&
        error("--frozen-twin or HD_TWINPROP_TWIN_PATH is required")
    return PrepareDistillationConfig(
        dataset_path=abspath(options["dataset"]),
        frozen_twin_path=abspath(options["frozen-twin"]),
        output_path=abspath(options["output"]),
        source_kind=Symbol(replace(
            lowercase(options["source-kind"]),
            "-" => "_",
        )),
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

end # module DistillationDatasetBridgeProductionV4

if abspath(PROGRAM_FILE) == @__FILE__
    DistillationDatasetBridgeProductionV4.main()
end
