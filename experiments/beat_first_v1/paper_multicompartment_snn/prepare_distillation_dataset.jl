module DistillationDatasetBridge

using Dates
using JLD2
using JSON3
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

export DISTILLATION_DATASET_SCHEMA,
    DistillationDatasetConfig,
    canonical_morphology_sha256,
    prepare_distillation_dataset,
    main

const DISTILLATION_DATASET_SCHEMA =
    "hd-swsnn-twinprop-distillation-dataset-v1"
const TRAIN_SPLIT = UInt8(1)
const VALIDATION_SPLIT = UInt8(2)
const TEST_SPLIT = UInt8(3)

"""
Configuration for the detailed-teacher/frozen-twin distillation bridge.

`source_kind` is deliberately mandatory.  Use `:official_neuron` only for
shards whose manifest explicitly identifies a NEURON/ModelDB source.  The
repository's deterministic Julia reconstruction must be labelled
`:canonical_julia`; the resulting artifact records that it is not the
authors' unpublished dataset.
"""
Base.@kwdef struct DistillationDatasetConfig
    dataset_path::String
    twin_artifact::String
    output_path::String
    source_kind::Symbol
    maximum_train_samples::Int = typemax(Int)
    maximum_validation_samples::Int = typemax(Int)
    maximum_test_samples::Int = typemax(Int)
    twin_batch_size::Int = 8
    minimum_twin_spike_auroc::Float64 = 0.985
    expected_original_dataset_sha256::String = ""
    expected_modeldb_source_sha256::String = ""
    expected_detailed_kernel_sha256::String = ""
    expected_morphology_sha256::String = ""
    expected_twin_parameter_sha256::String = ""
    expected_twin_artifact_sha256::String = ""
    selected_dendritic_compartments::Vector{Int} = Int[]
end

@inline function _get(value, name::Symbol, default=nothing)
    if value isa AbstractDict
        return get(value, name, get(value, String(name), default))
    elseif value !== nothing && hasproperty(value, name)
        return getproperty(value, name)
    end
    return default
end

function _file_sha256(path::AbstractString)
    source = abspath(path)
    isfile(source) || error("file is absent: $source")
    return bytes2hex(SHA.sha256(read(source)))
end

function _digest_values(values...)
    context = SHA.SHA2_256_CTX()
    for value in values
        bytes = codeunits(String(value))
        SHA.update!(context, reinterpret(UInt8, [length(bytes)]))
        SHA.update!(context, bytes)
    end
    return bytes2hex(SHA.digest!(context))
end

function _tree_sha256(tree::HayTree)
    context = SHA.SHA2_256_CTX()
    for value in (
        tree.parent,
        tree.region,
        tree.distance_um,
        tree.area_um2,
        tree.axial_conductance_ns,
    )
        SHA.update!(context, reinterpret(UInt8, vec(value)))
    end
    return bytes2hex(SHA.digest!(context))
end

canonical_morphology_sha256() = _tree_sha256(paper_hay_tree())

function _require_sha256(name::AbstractString, value::AbstractString)
    occursin(r"^[0-9a-fA-F]{64}$", value) ||
        error("$name is not a 64-character SHA-256 digest: $value")
    return lowercase(value)
end

function _assert_expected(
    name::AbstractString,
    actual::AbstractString,
    expected::AbstractString,
)
    isempty(expected) && return actual
    lowercase(actual) == lowercase(expected) ||
        error("$name hash mismatch: expected $expected, got $actual")
    return actual
end

function _config_payload(config::DistillationDatasetConfig)
    return (;
        schema=DISTILLATION_DATASET_SCHEMA,
        source_kind=String(config.source_kind),
        maximum_train_samples=config.maximum_train_samples,
        maximum_validation_samples=config.maximum_validation_samples,
        maximum_test_samples=config.maximum_test_samples,
        twin_batch_size=config.twin_batch_size,
        minimum_twin_spike_auroc=config.minimum_twin_spike_auroc,
        selected_dendritic_compartments=
            config.selected_dendritic_compartments,
        target_semantics=(;
            voltage="frozen PaperDigitalTwin physical soma voltage",
            spike="frozen PaperDigitalTwin per-step soma spike probability",
            nmda="frozen PaperDigitalTwin physical regional NMDA current",
            calcium_event="detailed teacher only",
            dendritic_voltage="detailed teacher only",
        ),
    )
end

function _config_sha256(config::DistillationDatasetConfig)
    return bytes2hex(SHA.sha256(codeunits(JSON3.write(
        _config_payload(config),
    ))))
end

function _manifest_value(manifest, names::Tuple, default=nothing)
    for name in names
        value = _get(manifest, name)
        value === nothing || return value
    end
    return default
end

struct _DatasetSource
    root::String
    manifest_path::String
    manifest::Any
    shard_paths::Vector{String}
    manifest_sha256::String
    original_dataset_sha256::String
    detailed_kernel_sha256::String
    morphology_sha256::String
    modeldb_source_sha256::String
    source_backend::String
end

function _declared_backend(manifest)
    direct = _manifest_value(
        manifest,
        (:source_kind, :teacher_backend, :simulation_backend),
        "",
    )
    !isempty(String(direct)) && return String(direct)
    reconstruction = _get(manifest, :cpu_reconstruction)
    reconstruction === nothing || return "canonical_julia"
    return ""
end

function _verify_source_kind(
    source_kind::Symbol,
    manifest,
    backend::AbstractString,
)
    source_kind in (:official_neuron, :canonical_julia) ||
        error("source_kind must be :official_neuron or :canonical_julia")
    normalized = lowercase(String(backend))
    if source_kind === :official_neuron
        (
            occursin("neuron", normalized) ||
            occursin("modeldb", normalized)
        ) || error(
            "official_neuron requires a manifest explicitly tagged " *
            "as a NEURON/ModelDB teacher source",
        )
        reconstruction = _get(manifest, :cpu_reconstruction)
        if reconstruction !== nothing
            exact = _get(reconstruction, :exact_neuron_morphology, false)
            Bool(exact) || error(
                "manifest identifies a reduced CPU reconstruction; " *
                "it cannot be relabelled official_neuron",
            )
        end
    else
        (
            isempty(normalized) ||
            occursin("julia", normalized) ||
            occursin("canonical", normalized)
        ) || error(
            "canonical_julia source_kind conflicts with manifest backend " *
            repr(backend),
        )
    end
    return nothing
end

function _source_hashes(
    manifest,
    source_kind::Symbol,
)
    detailed = String(_manifest_value(
        manifest,
        (
            :cell_mechanism_sha256,
            :detailed_kernel_sha256,
            :detailed_teacher_hash,
            :teacher_hash,
        ),
        "",
    ))
    morphology = String(_manifest_value(
        manifest,
        (:morphology_sha256, :morphology_hash),
        "",
    ))
    modeldb = String(_manifest_value(
        manifest,
        (
            :official_modeldb_source_hash,
            :modeldb_source_sha256,
            :modeldb_source_hash,
        ),
        "",
    ))
    if source_kind === :canonical_julia
        canonical_path = joinpath(@__DIR__, "PaperHayCell.jl")
        canonical_hash = _file_sha256(canonical_path)
        isempty(detailed) && (detailed = canonical_hash)
        isempty(morphology) &&
            (morphology = canonical_morphology_sha256())
        # This is the hash of the checked-in ModelDB-derived Julia source,
        # not a claim that the original ModelDB archive bytes are embedded.
        isempty(modeldb) && (modeldb = canonical_hash)
    end
    isempty(detailed) &&
        error("dataset manifest has no detailed-kernel hash")
    isempty(morphology) &&
        error("dataset manifest has no morphology hash")
    isempty(modeldb) && error(
        "official NEURON dataset must declare its ModelDB source hash",
    )
    return (
        _require_sha256("detailed kernel", detailed),
        _require_sha256("morphology", morphology),
        _require_sha256("ModelDB source", modeldb),
    )
end

function _load_dataset_source(config::DistillationDatasetConfig)
    source_path = abspath(config.dataset_path)
    manifest = nothing
    manifest_path = ""
    root = ""
    shard_paths = String[]
    declared_hashes = String[]

    if isdir(source_path)
        root = source_path
        manifest_path = joinpath(root, "manifest.json")
        isfile(manifest_path) ||
            error("dataset directory has no manifest.json: $root")
        manifest = JSON3.read(read(manifest_path, String))
        records = _get(manifest, :shards)
        records === nothing &&
            error("dataset manifest has no shard records")
        for record in records
            relative = String(_get(record, :path, ""))
            isempty(relative) && error("shard record has no path")
            path = abspath(joinpath(root, relative))
            isfile(path) || error("dataset shard is absent: $path")
            declared = String(_get(record, :sha256, ""))
            isempty(declared) &&
                error("dataset shard has no declared SHA-256: $relative")
            actual = _file_sha256(path)
            lowercase(actual) == lowercase(declared) || error(
                "dataset shard hash mismatch for $relative: " *
                "expected $declared, got $actual",
            )
            push!(shard_paths, path)
            push!(declared_hashes, lowercase(actual))
        end
    elseif isfile(source_path)
        root = dirname(source_path)
        manifest_path = source_path
        manifest = JLD2.load(source_path)
        push!(shard_paths, source_path)
        push!(declared_hashes, _file_sha256(source_path))
    else
        error("dataset source is absent: $source_path")
    end
    isempty(shard_paths) && error("dataset source contains no shards")

    manifest_sha256 = _file_sha256(manifest_path)
    original_dataset_sha256 = _digest_values(
        manifest_sha256,
        declared_hashes...,
    )
    _assert_expected(
        "original dataset",
        original_dataset_sha256,
        config.expected_original_dataset_sha256,
    )
    backend = _declared_backend(manifest)
    _verify_source_kind(config.source_kind, manifest, backend)
    detailed, morphology, modeldb =
        _source_hashes(manifest, config.source_kind)
    _assert_expected(
        "detailed kernel",
        detailed,
        config.expected_detailed_kernel_sha256,
    )
    _assert_expected(
        "morphology",
        morphology,
        config.expected_morphology_sha256,
    )
    _assert_expected(
        "ModelDB source",
        modeldb,
        config.expected_modeldb_source_sha256,
    )

    return _DatasetSource(
        root,
        manifest_path,
        manifest,
        shard_paths,
        manifest_sha256,
        original_dataset_sha256,
        detailed,
        morphology,
        modeldb,
        isempty(backend) ? String(config.source_kind) : String(backend),
    )
end

function _held_out_metrics(frozen)
    metadata = frozen.metadata
    metrics = _get(metadata, :held_out_test)
    metrics === nothing &&
        (metrics = _get(metadata, :test_metrics))
    metrics === nothing && error(
        "frozen twin has no held-out test metrics; below-gate status " *
        "cannot be ruled out",
    )
    return metrics
end

function _verify_twin(
    config::DistillationDatasetConfig,
    source::_DatasetSource,
)
    artifact_path = abspath(config.twin_artifact)
    isfile(artifact_path) ||
        error("frozen digital twin is absent: $artifact_path")
    frozen = load_frozen_twin(artifact_path)
    integrity = assert_frozen_unchanged(frozen)
    lowercase(integrity.parameter_sha256) ==
        lowercase(frozen.parameter_sha256) ||
        error("frozen twin parameter integrity mismatch")
    lowercase(integrity.artifact_sha256) ==
        lowercase(frozen.artifact_sha256) ||
        error("frozen twin artifact integrity mismatch")
    _assert_expected(
        "frozen twin parameter",
        frozen.parameter_sha256,
        config.expected_twin_parameter_sha256,
    )
    _assert_expected(
        "frozen twin artifact",
        frozen.artifact_sha256,
        config.expected_twin_artifact_sha256,
    )

    metadata = frozen.metadata
    twin_detailed = String(_manifest_value(
        metadata,
        (
            :cell_mechanism_sha256,
            :detailed_teacher_hash,
            :teacher_hash,
        ),
        "",
    ))
    twin_morphology = String(_manifest_value(
        metadata,
        (:morphology_sha256, :morphology_hash),
        "",
    ))
    isempty(twin_detailed) &&
        error("frozen twin has no detailed-teacher hash")
    isempty(twin_morphology) &&
        error("frozen twin has no morphology hash")
    lowercase(twin_detailed) == lowercase(source.detailed_kernel_sha256) ||
        error(
            "frozen twin detailed-teacher hash does not match dataset",
        )
    lowercase(twin_morphology) == lowercase(source.morphology_sha256) ||
        error("frozen twin morphology hash does not match dataset")

    metrics = _held_out_metrics(frozen)
    spike_auroc = Float64(_get(metrics, :spike_auroc, NaN))
    isfinite(spike_auroc) || error(
        "frozen twin held-out spike AUROC is absent or non-finite",
    )
    spike_auroc >= config.minimum_twin_spike_auroc || error(
        "frozen twin is below gate: held-out spike AUROC " *
        "$spike_auroc < $(config.minimum_twin_spike_auroc)",
    )
    gate = _get(metadata, :gate)
    if gate !== nothing && !Bool(_get(gate, :passed, false))
        error("frozen twin metadata records a failed fidelity gate")
    end
    return frozen, integrity, metrics, _file_sha256(artifact_path)
end

function _unwrap_shard(data)
    for key in ("dataset", "payload", :dataset, :payload)
        if data isa AbstractDict && haskey(data, key)
            candidate = data[key]
            _get(candidate, :input) !== nothing ||
                _get(candidate, :contact_segment) !== nothing ||
                continue
            return candidate
        end
    end
    return data
end

function _split_code_for_shard(shard)
    split = _get(shard, :split_code)
    split !== nothing && return UInt8.(split)
    input = _get(shard, :input)
    samples = input === nothing ?
        size(_get(shard, :contact_segment), 2) :
        size(input, 3)
    result = fill(UInt8(0), samples)
    for (name, code) in (
        (:train_indices, TRAIN_SPLIT),
        (:validation_indices, VALIDATION_SPLIT),
        (:test_indices, TEST_SPLIT),
    )
        indices = _get(shard, name, Int[])
        for index in indices
            result[Int(index)] == 0 ||
                error("sample belongs to multiple dataset splits")
            result[Int(index)] = code
        end
    end
    all(!=(0x00), result) ||
        error("dataset shard does not assign every sample to a split")
    return result
end

function _dense_input(shard, frozen)
    stored = _get(shard, :input)
    if stored !== nothing
        dense = Float32.(stored)
    else
        for required in (
            :contact_segment,
            :contact_kind,
            :contact_strength,
            :event_spike,
        )
            _get(shard, required) === nothing &&
                error("compact shard has no $required")
        end
        dense = expand_compact_twin_input(
            _get(shard, :contact_segment),
            _get(shard, :contact_kind),
            _get(shard, :contact_strength),
            _get(shard, :event_spike),
            frozen.model.config,
        )
    end
    ndims(dense) == 3 ||
        throw(DimensionMismatch("dense input must be feature x time x sample"))
    size(dense, 1) == frozen.model.config.input_dim ||
        throw(DimensionMismatch(
            "dataset/twin input dimension mismatch: " *
            "$(size(dense, 1)) != $(frozen.model.config.input_dim)",
        ))
    all(isfinite, dense) || error("dense twin input contains non-finite values")
    return dense
end

function _metadata_for_shard(shard, source::_DatasetSource)
    metadata = _get(shard, :metadata)
    metadata === nothing && (metadata = source.manifest)
    return metadata
end

function _choose_region_compartment(
    region::AbstractVector,
    distance::AbstractVector,
    code::Integer;
    mode::Symbol,
)
    indices = findall(==(UInt8(code)), UInt8.(region))
    isempty(indices) &&
        error("detailed teacher has no compartment in region $code")
    if mode === :minimum
        return indices[argmin(Float32.(distance[indices]))]
    elseif mode === :maximum
        return indices[argmax(Float32.(distance[indices]))]
    elseif mode === :hot_zone
        return indices[argmin(abs.(Float32.(distance[indices]) .- 785.0f0))]
    end
    error("unsupported compartment-selection mode")
end

function _selected_compartments(
    config::DistillationDatasetConfig,
    metadata,
    segment_count::Int,
)
    if !isempty(config.selected_dendritic_compartments)
        result = copy(config.selected_dendritic_compartments)
    else
        declared = _get(metadata, :selected_dendritic_compartments)
        if declared !== nothing
            result = Int.(declared)
        else
            region = _get(metadata, :compartment_region)
            distance = _get(metadata, :compartment_distance_um)
            if region === nothing || distance === nothing
                tree = paper_hay_tree()
                compartment_count(tree) == segment_count || error(
                    "shard has no compartment geometry and does not match " *
                    "the canonical Julia detailed tree",
                )
                region = tree.region
                distance = tree.distance_um
            end
            result = Int[
                _choose_region_compartment(
                    region,
                    distance,
                    BASAL;
                    mode=:maximum,
                ),
                _choose_region_compartment(
                    region,
                    distance,
                    APICAL_TRUNK;
                    mode=:minimum,
                ),
                _choose_region_compartment(
                    region,
                    distance,
                    APICAL_TRUNK;
                    mode=:hot_zone,
                ),
                _choose_region_compartment(
                    region,
                    distance,
                    APICAL_TUFT;
                    mode=:maximum,
                ),
            ]
        end
    end
    length(result) == 4 ||
        error("exactly four dendritic compartments must be selected")
    length(unique(result)) == 4 ||
        error("selected dendritic compartments must be distinct")
    all(index -> 1 <= index <= segment_count, result) ||
        error("selected dendritic compartment is out of range")
    return result
end

function _stored_detailed_targets(
    shard,
    indices,
    selected_compartments,
)
    calcium = _get(shard, :target_calcium_event)
    dendritic = _get(shard, :target_dendritic_voltage)
    if dendritic === nothing
        compartment_voltage =
            _get(shard, :target_compartment_voltage)
        if compartment_voltage !== nothing
            dendritic = Float32.(@view(
                compartment_voltage[selected_compartments, :, indices],
            ))
        end
    else
        dendritic = Float32.(@view(dendritic[:, :, indices]))
    end
    if calcium !== nothing
        calcium = Float32.(@view(calcium[:, indices]))
    else
        compartment_event =
            _get(shard, :target_compartment_calcium_event)
        if compartment_event !== nothing
            calcium = Float32.(dropdims(
                maximum(
                    @view(compartment_event[:, :, indices]);
                    dims=1,
                );
                dims=1,
            ))
        end
    end
    return calcium, dendritic
end

function _replay_canonical_detailed(
    shard,
    indices,
    selected_compartments,
)
    segment = _get(shard, :contact_segment)
    kind = _get(shard, :contact_kind)
    strength = _get(shard, :contact_strength)
    event = _get(shard, :event_spike)
    for (name, value) in (
        ("contact_segment", segment),
        ("contact_kind", kind),
        ("contact_strength", strength),
        ("event_spike", event),
    )
        value === nothing && error(
            "canonical detailed replay needs compact $name",
        )
    end
    tree = paper_hay_tree()
    parameters = HayParameters(tree; ablation=:full)
    compartment_count(tree) == maximum(Int.(segment)) ||
        maximum(Int.(segment)) <= compartment_count(tree) ||
        error("compact contact location exceeds canonical morphology")
    time_steps = size(event, 2)
    batch = length(indices)
    calcium = zeros(Float32, time_steps, batch)
    dendritic =
        Array{Float32}(undef, 4, time_steps, batch)
    state = HayState(tree, parameters)
    drive = HaySynapticDrive(tree)
    diagnostics = HayDiagnostics(tree)
    soma = Int(tree.soma)
    @inbounds for (output_item, source_item) in enumerate(indices)
        reset_state!(state, parameters)
        reset_drive!(drive)
        reset_diagnostics!(diagnostics)
        for time in 1:time_steps
            reset_drive!(drive)
            for contact in axes(segment, 1)
                event[contact, time, source_item] || continue
                destination = Int(segment[contact, source_item])
                amplitude = Float32(strength[contact, source_item])
                if UInt8(kind[contact, source_item]) == UInt8(1)
                    add_synaptic_event!(
                        drive,
                        destination;
                        ampa=amplitude,
                        nmda=amplitude,
                    )
                elseif UInt8(kind[contact, source_item]) == UInt8(2)
                    add_synaptic_event!(
                        drive,
                        destination;
                        gaba=amplitude,
                    )
                else
                    error("unknown compact contact kind")
                end
            end
            hay_cell_step!(
                state,
                drive,
                diagnostics,
                tree,
                parameters,
            )
            calcium[time, output_item] =
                maximum(@view state.local_ca_event[2:end])
            for selected in 1:4
                dendritic[selected, time, output_item] =
                    state.voltage_mv[selected_compartments[selected]]
            end
            state.local_ca_event[soma] = 0.0f0
        end
    end
    return calcium, dendritic
end

function _detailed_targets(
    config::DistillationDatasetConfig,
    shard,
    indices,
    selected_compartments,
)
    calcium, dendritic = _stored_detailed_targets(
        shard,
        indices,
        selected_compartments,
    )
    if calcium === nothing
        config.source_kind === :canonical_julia || error(
            "official NEURON shard has no detailed Ca-event target",
        )
        replay_calcium, replay_dendritic =
            _replay_canonical_detailed(
                shard,
                indices,
                selected_compartments,
            )
        calcium = replay_calcium
        dendritic === nothing && (dendritic = replay_dendritic)
    end
    dendritic === nothing && error(
        "detailed teacher has no selected or compartment voltage target",
    )
    time_steps = size(calcium, 1)
    batch = length(indices)
    size(calcium) == (time_steps, batch) ||
        throw(DimensionMismatch("detailed Ca-event target shape differs"))
    size(dendritic) == (4, time_steps, batch) ||
        throw(DimensionMismatch(
            "detailed dendritic voltage target must be 4 x time x sample",
        ))
    all(isfinite, calcium) ||
        error("detailed Ca-event target contains non-finite values")
    all(isfinite, dendritic) ||
        error("detailed dendritic voltage contains non-finite values")
    return Float32.(calcium), Float32.(dendritic)
end

function _input_metadata(config)
    segments = config.segments
    compartment = Int[]
    receptor = Int[]
    plane = Int[]
    for input_plane in 1:config.input_planes
        for input_receptor in 1:config.receptors
            for segment in 1:segments
                push!(compartment, segment)
                push!(receptor, input_receptor)
                push!(plane, input_plane)
            end
        end
    end
    length(compartment) == config.input_dim ||
        error("internal twin input-layout mismatch")
    return compartment, receptor, plane
end

function _sample_plan(
    source::_DatasetSource,
    config::DistillationDatasetConfig,
)
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
    time_steps = 0
    for path in source.shard_paths
        shard = _unwrap_shard(JLD2.load(path))
        split = _split_code_for_shard(shard)
        local_indices = Int[]
        local_codes = UInt8[]
        for (index, code) in enumerate(split)
            haskey(limits, code) ||
                error("unknown split code $code")
            used[code] >= limits[code] && continue
            used[code] += 1
            push!(local_indices, index)
            push!(local_codes, code)
        end
        isempty(local_indices) && continue
        stored_input = _get(shard, :input)
        shard_time = stored_input === nothing ?
            size(_get(shard, :event_spike), 2) :
            size(stored_input, 2)
        time_steps == 0 && (time_steps = shard_time)
        time_steps == shard_time ||
            error("all distillation shards must have the same time length")
        push!(
            plan,
            (;
                path,
                indices=local_indices,
                split_codes=local_codes,
            ),
        )
    end
    isempty(plan) && error("sample limits selected no trajectories")
    all(code -> used[code] > 0, keys(used)) ||
        error("distillation dataset requires non-empty train/validation/test")
    return plan, time_steps, used
end

function _run_frozen_twin!(
    target_voltage,
    target_spike,
    target_nmda,
    frozen,
    input,
    output_range,
    batch_size::Int,
)
    source_count = size(input, 3)
    length(output_range) == source_count ||
        throw(DimensionMismatch("twin output range differs"))
    for first_item in 1:batch_size:source_count
        last_item = min(first_item + batch_size - 1, source_count)
        source_indices = first_item:last_item
        destination = first(output_range) + first_item - 1:
            first(output_range) + last_item - 1
        prediction = twin_forward(
            frozen,
            Float32.(@view input[:, :, source_indices]),
        )
        target_voltage[:, destination] .= prediction.voltage
        target_spike[:, destination] .= prediction.spike_probability
        target_nmda[:, :, destination] .= prediction.nmda
    end
    return nothing
end

function _atomic_jldsave(path::AbstractString; dataset)
    destination = abspath(path)
    mkpath(dirname(destination))
    temporary = destination * ".tmp." * string(getpid())
    JLD2.jldsave(temporary; dataset)
    mv(temporary, destination; force=true)
    return destination
end

"""
Build the exact mixed-supervision dataset consumed by final 11-state
distillation.

The three public neuronal outputs (soma voltage, soma spike probability and
regional NMDA current) are *always recomputed* by actual inference through
the supplied frozen `PaperDigitalTwin`.  Ca events and selected dendritic
voltages are never synthesized by the twin: they remain detailed-teacher
targets.  This boundary is recorded per target in provenance.
"""
function prepare_distillation_dataset(
    config::DistillationDatasetConfig,
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
    source = _load_dataset_source(config)
    frozen, twin_integrity, held_out_metrics, artifact_file_sha256 =
        _verify_twin(config, source)
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
    target_nmda = Array{Float32,3}(
        undef,
        frozen.model.config.nmda_regions,
        time_steps,
        samples,
    )
    frozen.model.config.nmda_regions == 4 ||
        error("final 11-state distillation requires four NMDA regions")
    target_calcium_event =
        Matrix{Float32}(undef, time_steps, samples)
    target_dendritic_voltage =
        Array{Float32,3}(undef, 4, time_steps, samples)
    split_code = Vector{UInt8}(undef, samples)
    selected_compartments = Int[]

    cursor = 1
    for item in plan
        shard = _unwrap_shard(JLD2.load(item.path))
        dense = _dense_input(shard, frozen)
        selected_dense =
            Float32.(@view dense[:, :, item.indices])
        metadata = _metadata_for_shard(shard, source)
        local_selected = _selected_compartments(
            config,
            metadata,
            frozen.model.config.segments,
        )
        if isempty(selected_compartments)
            selected_compartments = local_selected
        elseif selected_compartments != local_selected
            error(
                "selected dendritic compartments differ between shards",
            )
        end
        calcium, dendritic = _detailed_targets(
            config,
            shard,
            item.indices,
            selected_compartments,
        )
        local_count = length(item.indices)
        destination = cursor:(cursor + local_count - 1)
        input[:, :, destination] .= selected_dense
        _run_frozen_twin!(
            target_voltage,
            target_spike,
            target_nmda,
            frozen,
            selected_dense,
            destination,
            config.twin_batch_size,
        )
        target_calcium_event[:, destination] .= calcium
        target_dendritic_voltage[:, :, destination] .= dendritic
        split_code[destination] .= item.split_codes
        cursor += local_count
    end
    cursor == samples + 1 || error("internal sample cursor mismatch")

    train_indices = findall(==(TRAIN_SPLIT), split_code)
    validation_indices = findall(==(VALIDATION_SPLIT), split_code)
    test_indices = findall(==(TEST_SPLIT), split_code)
    input_compartment, input_receptor, input_plane =
        _input_metadata(frozen.model.config)
    config_sha256 = _config_sha256(config)
    detailed_target_source =
        config.source_kind === :official_neuron ?
        "official NEURON detailed teacher" :
        "canonical Julia PaperHayCell replay/stored detailed trajectory"
    mixed_supervision_provenance = (;
        input="verified original detailed-teacher shard protocol",
        target_voltage="actual frozen PaperDigitalTwin inference",
        target_spike=
            "actual frozen PaperDigitalTwin per-step spike probability",
        target_nmda="actual frozen PaperDigitalTwin inference",
        target_calcium_event=detailed_target_source,
        target_dendritic_voltage=detailed_target_source,
        twin_is_frozen=true,
        detailed_only_targets=(
            "target_calcium_event",
            "target_dendritic_voltage",
        ),
        twin_targets=(
            "target_voltage",
            "target_spike",
            "target_nmda",
        ),
    )
    hashes = (;
        official_modeldb_source_hash=source.modeldb_source_sha256,
        detailed_kernel_hash=source.detailed_kernel_sha256,
        morphology_hash=source.morphology_sha256,
        frozen_twin_parameter_hash=frozen.parameter_sha256,
        frozen_twin_artifact_hash=frozen.artifact_sha256,
        frozen_twin_file_sha256=artifact_file_sha256,
        original_dataset_sha256=source.original_dataset_sha256,
        source_manifest_sha256=source.manifest_sha256,
        config_sha256,
    )
    metadata = (;
        schema=DISTILLATION_DATASET_SCHEMA,
        model_name=HD_SWSNN_TWINPROP_NAME,
        stage="digital_twin_to_eleven_state_dataset",
        created_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
        source_kind=String(config.source_kind),
        source_backend=source.source_backend,
        official_neuron_source=
            config.source_kind === :official_neuron,
        canonical_julia_fallback=
            config.source_kind === :canonical_julia,
        canonical_fallback_disclosure=
            config.source_kind === :canonical_julia ?
            "The authors' unpublished NEURON shards were not used; " *
            "detailed-only targets come from the checked-in canonical " *
            "Julia ModelDB-derived reconstruction." :
            "",
        mixed_supervision=true,
        mixed_supervision_provenance,
        teacher_schema="PaperDigitalTwin-frozen-v1",
        target_spike_semantics="soft per-step probability",
        input_layout=twin_input_layout(frozen),
        input_compartment,
        input_receptor,
        input_plane,
        selected_dendritic_compartments=selected_compartments,
        selected_dendritic_semantics=(
            "distal_basal",
            "proximal_apical_trunk",
            "apical_calcium_hot_zone",
            "distal_apical_tuft",
        ),
        dt_ms=frozen.model.config.dt_ms,
        twin_held_out_metrics=held_out_metrics,
        twin_gate=(;
            minimum_spike_auroc=config.minimum_twin_spike_auroc,
            passed=true,
        ),
        twin_integrity,
        hashes,
        # Compatibility aliases consumed by the existing final distiller.
        morphology_sha256=source.morphology_sha256,
        morphology_hash=source.morphology_sha256,
        cell_mechanism_sha256=source.detailed_kernel_sha256,
        detailed_kernel_hash=source.detailed_kernel_sha256,
        official_modeldb_source_hash=source.modeldb_source_sha256,
        frozen_twin_parameter_hash=frozen.parameter_sha256,
        frozen_twin_artifact_hash=frozen.artifact_sha256,
        digital_twin_hash=frozen.artifact_sha256,
        teacher_hash=frozen.artifact_sha256,
        original_dataset_sha256=source.original_dataset_sha256,
        dataset_sha256=source.original_dataset_sha256,
        source_manifest_sha256=source.manifest_sha256,
        config_sha256,
    )
    dataset = (;
        schema=DISTILLATION_DATASET_SCHEMA,
        input,
        target_voltage,
        target_spike,
        target_nmda,
        target_calcium_event,
        target_dendritic_voltage,
        train_indices,
        validation_indices,
        test_indices,
        split_code,
        dataset_sha256=source.original_dataset_sha256,
        original_dataset_sha256=source.original_dataset_sha256,
        source_manifest_sha256=source.manifest_sha256,
        frozen_twin_parameter_hash=frozen.parameter_sha256,
        frozen_twin_artifact_hash=frozen.artifact_sha256,
        detailed_kernel_hash=source.detailed_kernel_sha256,
        morphology_hash=source.morphology_sha256,
        official_modeldb_source_hash=source.modeldb_source_sha256,
        cell_mechanism_sha256=source.detailed_kernel_sha256,
        config_sha256,
        dt_ms=frozen.model.config.dt_ms,
        mixed_supervision=true,
        mixed_supervision_provenance,
        metadata,
    )
    output_path = _atomic_jldsave(config.output_path; dataset)
    output_file_sha256 = _file_sha256(output_path)
    reloaded = JLD2.load(output_path)
    haskey(reloaded, "dataset") ||
        error("saved distillation dataset cannot be reloaded")
    reloaded_dataset = reloaded["dataset"]
    _get(reloaded_dataset, :schema) == DISTILLATION_DATASET_SCHEMA ||
        error("saved distillation dataset schema changed")
    _get(reloaded_dataset, :frozen_twin_artifact_hash) ==
        frozen.artifact_sha256 ||
        error("saved distillation dataset lost twin lineage")

    return (;
        schema=DISTILLATION_DATASET_SCHEMA,
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
        selected_dendritic_compartments=selected_compartments,
        hashes,
        mixed_supervision=true,
        twin_gate_passed=true,
    )
end

function _parse_arguments(arguments)
    values = Dict{String,String}(
        "dataset" => "",
        "twin" => "",
        "output" => joinpath(
            @__DIR__,
            "artifacts",
            "distillation_dataset.jld2",
        ),
        "source-kind" => "canonical-julia",
        "max-train" => string(typemax(Int)),
        "max-validation" => string(typemax(Int)),
        "max-test" => string(typemax(Int)),
        "twin-batch" => "8",
        "minimum-twin-spike-auroc" => "0.985",
        "expected-original-dataset-sha256" => "",
        "expected-modeldb-source-sha256" => "",
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
        haskey(values, key) || error("unknown option --$key")
        index == length(arguments) &&
            error("missing value after --$key")
        values[key] = arguments[index + 1]
        index += 2
    end
    isempty(values["dataset"]) && error("--dataset is required")
    isempty(values["twin"]) && error("--twin is required")
    source_kind = Symbol(replace(
        lowercase(values["source-kind"]),
        "-" => "_",
    ))
    return DistillationDatasetConfig(
        dataset_path=abspath(values["dataset"]),
        twin_artifact=abspath(values["twin"]),
        output_path=abspath(values["output"]),
        source_kind,
        maximum_train_samples=parse(Int, values["max-train"]),
        maximum_validation_samples=
            parse(Int, values["max-validation"]),
        maximum_test_samples=parse(Int, values["max-test"]),
        twin_batch_size=parse(Int, values["twin-batch"]),
        minimum_twin_spike_auroc=parse(
            Float64,
            values["minimum-twin-spike-auroc"],
        ),
        expected_original_dataset_sha256=
            values["expected-original-dataset-sha256"],
        expected_modeldb_source_sha256=
            values["expected-modeldb-source-sha256"],
        expected_detailed_kernel_sha256=
            values["expected-detailed-kernel-sha256"],
        expected_morphology_sha256=
            values["expected-morphology-sha256"],
        expected_twin_parameter_sha256=
            values["expected-twin-parameter-sha256"],
        expected_twin_artifact_sha256=
            values["expected-twin-artifact-sha256"],
    )
end

function main(arguments=ARGS)
    report = prepare_distillation_dataset(_parse_arguments(arguments))
    println(JSON3.write(report))
    return report
end

end # module DistillationDatasetBridge

if abspath(PROGRAM_FILE) == @__FILE__
    DistillationDatasetBridge.main()
end
