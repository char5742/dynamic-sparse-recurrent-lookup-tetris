module DigitalTwinTrainingFinal

using Dates
using JLD2
using JSON3
using Lux
using NPZ
using Optimisers
using Random
using SHA
using Statistics
using Zygote

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(_PARENT_MODULE, :PaperDigitalTwin)
    Base.include(
        _PARENT_MODULE,
        joinpath(@__DIR__, "PaperDigitalTwin.jl"),
    )
end
if !isdefined(_PARENT_MODULE, :OfficialTeacherContract)
    Base.include(
        _PARENT_MODULE,
        joinpath(@__DIR__, "OfficialTeacherContract.jl"),
    )
end

using ..PaperDigitalTwin
import ..OfficialTeacherContract

export OFFICIAL_TEACHER_SCHEMA,
    FINAL_LINEAGE_SCHEMA,
    OfficialTeacherDataset,
    OfficialValidationPartition,
    FinalTwinTrainingConfig,
    TwinLossWeightsFinal,
    verify_official_teacher_dataset,
    derive_validation_partition,
    load_official_shard,
    expand_official_input_chunk,
    fit_official_normalizer,
    evaluate_official_split,
    adapter_smoke,
    train_digital_twin_final,
    verify_frozen_twin_final,
    load_frozen_twin_final,
    main

const OFFICIAL_TEACHER_SCHEMA =
    "hd_swsnn_twinprop.neuron_teacher.v1"
const OFFICIAL_TEACHER_FINAL_SCHEMA =
    "hd_swsnn_twinprop.neuron_teacher.final.v2"
const OFFICIAL_TEACHER_SCHEMAS = (
    OFFICIAL_TEACHER_SCHEMA,
    OFFICIAL_TEACHER_FINAL_SCHEMA,
)
const FINAL_LINEAGE_SCHEMA =
    "hd_swsnn_twinprop.digital_twin.lineage.v1"
const EXPECTED_SEGMENTS = 642
const FIRST_DENDRITIC_SEGMENT = 2
const LAST_DENDRITIC_SEGMENT = 640
const DENDRITIC_SEGMENTS = 639
const OFFICIAL_ELM_INPUT_DIM = 2 * DENDRITIC_SEGMENTS
const TRAIN_SPLIT = UInt8(1)
const VALIDATION_SPLIT = UInt8(2)
const TEST_SPLIT = UInt8(3)
const EXCITATORY = UInt8(1)
const INHIBITORY = UInt8(2)
const REQUIRED_SOURCE_HASHES = (
    :modeldb_tree_sha256,
    :morphology_sha256,
    :biophysics_sha256,
    :mechanism_sources_sha256,
    :mechanism_library_sha256,
    :template_sha256,
    :generator_source_sha256,
)
const BASE_NPZ_KEYS = [
    "sample_indices",
    "split_code",
    "contact_axon",
    "contact_segment",
    "contact_kind",
    "contact_strength",
    "axon_kind",
    "target_voltage",
    "target_spike",
    "target_nmda",
]
const DENSE_EVENT_KEYS = ["axon_event_spike"]
const COMPACT_EVENT_KEYS = [
    "event_trial_offset",
    "event_axon",
    "event_time_bin",
]

@inline _get(value, name::Symbol, default=nothing) =
    hasproperty(value, name) ? getproperty(value, name) :
    value isa AbstractDict ? get(value, String(name), get(value, name, default)) :
    default

_file_sha256(path::AbstractString) =
    bytes2hex(SHA.sha256(read(path)))

function _require_hex_digest(value, label)
    digest = lowercase(String(value))
    occursin(r"^[0-9a-f]{64}$", digest) ||
        error("$label is not a SHA-256 digest")
    return digest
end

function _json_object_pairs(value)
    pairs_out = Pair{String,Any}[]
    for (key, child) in pairs(value)
        push!(pairs_out, String(key) => child)
    end
    sort!(pairs_out; by=first)
    return pairs_out
end

"""
Python-compatible canonical JSON for cryptographic contracts.

The official generator uses `json.dumps(sort_keys=true, separators=(",",":"))`.
This implementation deliberately handles only JSON values; model arrays use a
separate binary digest.
"""
function _canonical_json(value)
    if value isa JSON3.Object || value isa AbstractDict ||
       value isa NamedTuple
        entries = String[]
        for (key, child) in _json_object_pairs(value)
            push!(
                entries,
                string(JSON3.write(key), ":", _canonical_json(child)),
            )
        end
        return string("{", join(entries, ","), "}")
    elseif value isa JSON3.Array || value isa AbstractVector ||
           value isa Tuple
        return string(
            "[",
            join((_canonical_json(child) for child in value), ","),
            "]",
        )
    elseif value === nothing || value === missing
        return "null"
    else
        return String(JSON3.write(value))
    end
end

_canonical_sha256(value) =
    bytes2hex(SHA.sha256(codeunits(_canonical_json(value))))

function _compact_json_lexemes(raw::AbstractString)
    output = IOBuffer()
    quoted = false
    escaped = false
    for character in raw
        if quoted
            write(output, character)
            if escaped
                escaped = false
            elseif character == '\\'
                escaped = true
            elseif character == '"'
                quoted = false
            end
        elseif character == '"'
            quoted = true
            write(output, character)
        elseif !isspace(character)
            write(output, character)
        end
    end
    quoted && error("unterminated JSON string in teacher contract")
    return String(take!(output))
end

function _raw_teacher_contract_sha256(
    manifest_text::AbstractString,
    declared::AbstractString,
)
    marker = "\"teacher_contract\": {"
    marker_range = findlast(marker, manifest_text)
    marker_range === nothing &&
        error("manifest text has no top-level teacher_contract")
    object_first = findnext('{', manifest_text, last(marker_range))
    object_first === nothing && error("teacher_contract object is malformed")
    quoted = false
    escaped = false
    depth = 0
    object_last = nothing
    for index in object_first:lastindex(manifest_text)
        character = manifest_text[index]
        if quoted
            escaped ? (escaped = false) :
            character == '\\' ? (escaped = true) :
            character == '"' ? (quoted = false) : nothing
        elseif character == '"'
            quoted = true
        elseif character == '{'
            depth += 1
        elseif character == '}'
            depth -= 1
            depth == 0 && (object_last = index; break)
        end
    end
    object_last === nothing && error("teacher_contract object is unbalanced")
    raw = manifest_text[object_first:object_last]
    without_digest = replace(
        raw,
        r",\s*\"teacher_contract_sha256\"\s*:\s*\"[0-9a-fA-F]{64}\"\s*}$" => "}",
    )
    without_digest == raw &&
        error("teacher_contract digest field was not the canonical final key")
    calculated = bytes2hex(
        SHA.sha256(codeunits(_compact_json_lexemes(without_digest))),
    )
    calculated == lowercase(String(declared)) ||
        error("teacher contract canonical lexeme digest mismatch")
    return calculated
end

function _contract_without_digest(contract)
    result = Dict{String,Any}()
    for (key, value) in pairs(contract)
        String(key) == "teacher_contract_sha256" && continue
        result[String(key)] = value
    end
    return result
end

function _source_hashes(manifest)
    hashes = _get(manifest, :source_hashes)
    hashes === nothing && error("official manifest has no source_hashes")
    values = Pair{Symbol,String}[]
    for name in REQUIRED_SOURCE_HASHES
        value = _get(hashes, name)
        value === nothing && error("source_hashes.$name is missing")
        push!(
            values,
            name => _require_hex_digest(value, "source_hashes.$name"),
        )
    end
    modeldb_commit = String(_get(hashes, :modeldb_git_commit, ""))
    modeldb_tree = String(_get(hashes, :modeldb_git_tree, ""))
    modeldb_url = String(_get(hashes, :modeldb_repository_url, ""))
    return (;
        values...,
        modeldb_git_commit=modeldb_commit,
        modeldb_git_tree=modeldb_tree,
        modeldb_repository_url=modeldb_url,
    )
end

function _segment_mapping(manifest)
    records = _get(manifest, :segments)
    records === nothing && error("official manifest has no segment catalog")
    length(records) == EXPECTED_SEGMENTS ||
        error(
            "official segment catalog has $(length(records)) entries; " *
            "expected $EXPECTED_SEGMENTS",
        )
    result = NamedTuple[]
    for (position, record) in enumerate(records)
        index = Int(_get(record, :index, 0))
        index == position ||
            error("segment catalog is not contiguous at position $position")
        push!(
            result,
            (;
                index,
                region_code=Int(_get(record, :region_code, -1)),
                region_name=String(_get(record, :region_name, "")),
                section_name=String(_get(record, :section_name, "")),
                section_region=String(_get(record, :section_region, "")),
                x=Float64(_get(record, :x, NaN)),
                distance_um=Float64(_get(record, :distance_um, NaN)),
                length_um=Float64(_get(record, :length_um, NaN)),
                diameter_um=Float64(_get(record, :diameter_um, NaN)),
                area_um2=Float64(_get(record, :area_um2, NaN)),
                has_calcium=Bool(_get(record, :has_calcium, false)),
                eligible_basal_apical=(
                    FIRST_DENDRITIC_SEGMENT <= index <=
                    LAST_DENDRITIC_SEGMENT &&
                    String(_get(record, :section_region, "")) in
                    ("basal", "apical")
                ),
                location_slot_rule=(
                    FIRST_DENDRITIC_SEGMENT <= index <=
                    LAST_DENDRITIC_SEGMENT ?
                    "one_based_hay_segment_index; density_is_per_contact" :
                    "ineligible_soma_or_axon"
                ),
            ),
        )
    end
    all(record -> all(
        isfinite,
        (
            record.x,
            record.distance_um,
            record.length_um,
            record.diameter_um,
            record.area_um2,
        ),
    ), result) || error("segment catalog contains non-finite geometry")
    return result
end

function _safe_child(root::AbstractString, relative::AbstractString)
    root_path = normpath(abspath(root))
    candidate = normpath(abspath(joinpath(root_path, relative)))
    prefix = lowercase(string(root_path, Base.Filesystem.path_separator))
    lowercase(candidate) == lowercase(root_path) ||
        startswith(lowercase(candidate), prefix) ||
        error("shard path escapes dataset root: $relative")
    return candidate
end

struct OfficialTeacherDataset
    root::String
    schema_name::String
    manifest_path::String
    manifest::Any
    manifest_sha256::String
    teacher_contract_sha256::String
    source_hashes::NamedTuple
    segment_mapping::Vector{NamedTuple}
    segment_mapping_sha256::String
    shard_paths::Vector{String}
    shard_sha256::Vector{String}
    shard_split_counts::Vector{NamedTuple}
    store_dense_events::Bool
    sample_dt_ms::Float32
    time_steps::Int
    axons::Int
    ragged_contacts::Bool
    compact_event_count::Bool
    dense_event_key::String
    paper_scale::Bool
    promotable_production::Bool
    connectivity_policy::NamedTuple
    official::Bool
end

function _npz_read(path::AbstractString, keys::Vector{String})
    # Loading the whole archive is forbidden: NumPy's Unicode metadata_json
    # has no NPZ.jl reader.  Selective reads also bound the adapter surface.
    result = NPZ.npzread(path, keys)
    result isa AbstractDict ||
        error("NPZ selective read returned an unexpected object")
    for key in keys
        haskey(result, key) || error("NPZ shard lacks required array $key")
    end
    return result
end

function _split_counts(codes)
    all(code -> code in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT), codes) ||
        error("split_code contains a value outside 1=train,2=validation,3=test")
    return (;
        train=count(==(TRAIN_SPLIT), codes),
        validation=count(==(VALIDATION_SPLIT), codes),
        test=count(==(TEST_SPLIT), codes),
    )
end

"""
Verify an official NEURON teacher dataset before reading any target.

This is fail-closed by default.  Julia reconstruction/random teachers are
never silently promoted to official data; callers must opt into a separate
control path with `allow_reconstruction=true`.
"""
function verify_official_teacher_dataset(
    dataset_root::AbstractString;
    allow_reconstruction::Bool=false,
    require_all_splits::Bool=false,
)
    root = abspath(dataset_root)
    manifest_path = joinpath(root, "manifest.json")
    isfile(manifest_path) ||
        error("teacher dataset has no manifest.json: $manifest_path")
    manifest_text = read(manifest_path, String)
    manifest = JSON3.read(manifest_text)
    schema = String(_get(
        manifest,
        :schema_name,
        _get(manifest, :schema, ""),
    ))
    if schema ∉ OFFICIAL_TEACHER_SCHEMAS
        allow_reconstruction ||
            error(
                "refusing non-official teacher schema '$schema'; " *
                "set allow_reconstruction=true only for an explicitly " *
                "non-promotable control",
            )
        error(
            "schema '$schema' is an allowed reconstruction control but is " *
            "not NPZ official data; use the reconstruction trainer path",
        )
    end
    String(_get(manifest, :completion_state, "")) == "complete" ||
        error("official teacher manifest is not completion_state=complete")
    stage = String(_get(manifest, :stage, ""))
    stage in (
        "official_hay_neuron_teacher",
        "official_hay_neuron_teacher_final",
    ) || error("manifest stage is not an official Hay NEURON stage")
    Int(_get(manifest, :total_segments, 0)) == EXPECTED_SEGMENTS ||
        error("official teacher must expose all $EXPECTED_SEGMENTS segments")
    array_contract = _get(manifest, :array_contract)
    contact_contract =
        array_contract === nothing ?
        nothing : _get(array_contract, :contact_segment)
    contact_index_base = Int(_get(
        manifest,
        :contact_segment_index_base,
        _get(contact_contract, :index_base, 0),
    ))
    contact_index_base == 1 ||
        error("contact_segment must use one-based Hay segment indices")

    declared_contract = _require_hex_digest(
        _get(manifest, :teacher_contract_sha256, ""),
        "teacher_contract_sha256",
    )
    contract = _get(manifest, :teacher_contract)
    contract === nothing && error("manifest has no teacher_contract")
    String(_get(contract, :schema_name, "")) == schema ||
        error("teacher_contract schema differs from manifest schema")
    contract_declared = _require_hex_digest(
        _get(contract, :teacher_contract_sha256, ""),
        "teacher_contract.teacher_contract_sha256",
    )
    contract_declared == declared_contract ||
        error("teacher contract digest differs between manifest fields")
    contract_verification =
        OfficialTeacherContract.verify_teacher_contract_manifest(
            manifest_text,
        )
    contract_verification.digest == declared_contract ||
        error("shared teacher-contract verification digest differs")

    source_hashes = _source_hashes(manifest)
    contract_hashes = _get(contract, :source_hashes)
    contract_hashes === nothing &&
        error("teacher_contract has no source_hashes")
    for name in REQUIRED_SOURCE_HASHES
        lowercase(String(_get(contract_hashes, name, ""))) ==
            getproperty(source_hashes, name) ||
            error("manifest and teacher_contract differ at source hash $name")
    end

    segment_mapping = _segment_mapping(manifest)
    segment_mapping_sha256 = _canonical_sha256(segment_mapping)
    config = _get(manifest, :config)
    config === nothing && error("official manifest has no config")
    sample_dt_ms = Float32(_get(config, :sample_dt_ms, NaN))
    duration_ms = Float64(_get(config, :duration_ms, NaN))
    isfinite(sample_dt_ms) && sample_dt_ms > 0 ||
        error("invalid official sample_dt_ms")
    isfinite(duration_ms) && duration_ms > 0 ||
        error("invalid official duration_ms")
    time_steps = round(Int, duration_ms / sample_dt_ms)
    axons = Int(_get(config, :axons, 0))
    axons > 0 || error("official config axons must be positive")
    store_dense_events = Bool(_get(config, :store_dense_events, false))
    ragged_contacts =
        array_contract !== nothing &&
        _get(array_contract, :contact_trial_offset) !== nothing
    compact_event_count =
        array_contract !== nothing &&
        _get(array_contract, :event_count) !== nothing
    dense_event_key = Bool(_get(
        config,
        :store_dense_axon_events,
        store_dense_events,
    )) ?
        (schema == OFFICIAL_TEACHER_FINAL_SCHEMA ?
            "axon_event_count" : "axon_event_spike") :
        ""

    conflict = _get(contract, :connectivity_scale_conflict)
    fully_paper_scale = conflict !== nothing && Bool(_get(
        conflict,
        :fully_paper_scale_claim,
        false,
    ))
    acknowledged = conflict !== nothing && Bool(_get(
        conflict,
        :interpretation_explicitly_acknowledged,
        false,
    ))
    paper_contract = _get(manifest, :paper_production_contract)
    actual_paper_counts =
        paper_contract !== nothing &&
        Int(_get(config, :train_trials, -1)) ==
            Int(_get(paper_contract, :train_trials, -2)) &&
        Int(_get(config, :test_trials, -1)) ==
            Int(_get(paper_contract, :held_out_test_trials, -2)) &&
        Int(_get(config, :duration_ms, -1)) ==
            Int(_get(paper_contract, :duration_ms, -2))
    paper_scale =
        fully_paper_scale && acknowledged && actual_paper_counts
    connectivity_policy = (;
        fully_paper_scale_claim=fully_paper_scale,
        interpretation_explicitly_acknowledged=acknowledged,
        actual_paper_counts,
        paper_scale,
        source=conflict === nothing ?
            "legacy_contract_without_scale_policy" :
            "teacher_contract.connectivity_scale_conflict",
    )
    promotable_production =
        schema == OFFICIAL_TEACHER_FINAL_SCHEMA && paper_scale

    records = _get(manifest, :shards)
    records === nothing && error("official manifest has no shards")
    isempty(records) && error("official manifest has an empty shard list")
    paths = String[]
    digests = String[]
    verified_split_counts = NamedTuple[]
    total_samples = 0
    aggregate = (train=0, validation=0, test=0)
    for (ordinal, record) in enumerate(records)
        String(_get(record, :schema_name, "")) == schema ||
            error("shard $ordinal schema differs from official manifest")
        lowercase(String(_get(
            record,
            :teacher_contract_sha256,
            "",
        ))) == declared_contract ||
            error("shard $ordinal teacher contract digest differs")
        path = _safe_child(root, String(_get(record, :path, "")))
        lowercase(splitext(path)[2]) == ".npz" ||
            error("official shard is not NPZ: $path")
        isfile(path) || error("official shard does not exist: $path")
        declared_hash = _require_hex_digest(
            _get(record, :sha256, ""),
            "shard[$ordinal].sha256",
        )
        actual_hash = _file_sha256(path)
        actual_hash == declared_hash ||
            error("official shard hash mismatch at $path")
        declared_bytes = Int(_get(record, :bytes, filesize(path)))
        filesize(path) == declared_bytes ||
            error("official shard byte length mismatch at $path")

        split_payload = _npz_read(path, ["sample_indices", "split_code"])
        codes = vec(UInt8.(split_payload["split_code"]))
        samples = length(codes)
        length(vec(split_payload["sample_indices"])) == samples ||
            error("sample_indices/split_code length mismatch in $path")
        Int(_get(record, :samples, -1)) == samples ||
            error("shard record sample count mismatch in $path")
        counts = _split_counts(codes)
        declared_counts = _get(record, :split_counts)
        declared_counts === nothing &&
            error("shard record has no split_counts: $path")
        for name in (:train, :validation, :test)
            declared_name =
                name === :test &&
                _get(declared_counts, :test) === nothing ?
                :held_out_test : name
            declared_value = _get(
                declared_counts,
                declared_name,
                nothing,
            )
            actual_value = getproperty(counts, name)
            (
                declared_value === nothing ?
                actual_value == 0 :
                Int(declared_value) == actual_value
            ) ||
                error("shard split count mismatch for $name in $path")
        end
        aggregate = (;
            train=aggregate.train + counts.train,
            validation=aggregate.validation + counts.validation,
            test=aggregate.test + counts.test,
        )
        counts.test > 0 &&
            (counts.train > 0 || counts.validation > 0) &&
            error(
                "held-out test samples share a shard with fit samples; " *
                "test targets could not remain unread until final gate",
            )
        total_samples += samples
        push!(paths, path)
        push!(digests, declared_hash)
        push!(verified_split_counts, counts)
    end
    Int(_get(manifest, :completed_trials, -1)) == total_samples ||
        error("completed_trials does not equal verified shard samples")
    require_all_splits &&
        (aggregate.train < 2 || aggregate.test == 0) &&
        error(
            "production fitting requires at least two code-1 samples and " *
            "a nonempty untouched code-3 test; verified counts=$aggregate",
        )

    return OfficialTeacherDataset(
        root,
        schema,
        manifest_path,
        manifest,
        _file_sha256(manifest_path),
        declared_contract,
        source_hashes,
        segment_mapping,
        segment_mapping_sha256,
        paths,
        digests,
        verified_split_counts,
        store_dense_events,
        sample_dt_ms,
        time_steps,
        axons,
        ragged_contacts,
        compact_event_count,
        dense_event_key,
        paper_scale,
        promotable_production,
        connectivity_policy,
        true,
    )
end

struct OfficialValidationPartition
    fit_sample_ids::Vector{Int32}
    validation_sample_ids::Vector{Int32}
    test_sample_ids::Vector{Int32}
    validation_fraction::Float64
    derivation::String
    validation_ids_sha256::String
    partition_sha256::String
end

@inline function _validation_score(
    teacher_contract_sha256::AbstractString,
    sample_id::Integer,
)
    digest = bytes2hex(SHA.sha256(codeunits(
        string(
            teacher_contract_sha256,
            ":validation-from-train:",
            sample_id,
        ),
    )))
    return parse(UInt64, digest[1:16]; base=16)
end

"""
Derive validation exclusively from split-code 1 without reading any target.

The lowest hash scores are selected, giving an exact, deterministic count.
Code-3 IDs remain in pure test shards and are not opened with target keys
until the single final held-out evaluation.
"""
function derive_validation_partition(
    dataset::OfficialTeacherDataset;
    validation_fraction::Real=0.02,
    validation_count::Union{Nothing,Integer}=nothing,
)
    0 < validation_fraction < 0.5 ||
        throw(ArgumentError("validation_fraction must be in (0,0.5)"))
    train_ids = Int32[]
    test_ids = Int32[]
    explicit_validation = Int32[]
    for shard_index in eachindex(dataset.shard_paths)
        payload = _npz_read(
            dataset.shard_paths[shard_index],
            ["sample_indices", "split_code"],
        )
        ids = vec(Int32.(payload["sample_indices"]))
        codes = vec(UInt8.(payload["split_code"]))
        for (sample_id, code) in zip(ids, codes)
            code == TRAIN_SPLIT ? push!(train_ids, sample_id) :
            code == TEST_SPLIT ? push!(test_ids, sample_id) :
            code == VALIDATION_SPLIT ?
                push!(explicit_validation, sample_id) :
                error("invalid split code")
        end
    end
    isempty(explicit_validation) ||
        error(
            "paper-final contract derives validation from code-1; " *
            "unexpected code-2 IDs were present",
        )
    length(train_ids) >= 2 ||
        error("validation derivation requires at least two code-1 samples")
    isempty(test_ids) &&
        error("paper-final contract requires untouched code-3 test samples")
    all_ids = vcat(train_ids, test_ids)
    length(unique(all_ids)) == length(all_ids) ||
        error("official sample_indices are not globally unique")
    manifest_config = _get(dataset.manifest, :config)
    manifest_validation_ids = _get(
        dataset.manifest,
        :validation_from_train_indices,
        nothing,
    )
    declared_count = _get(
        manifest_config,
        :validation_trials_from_train,
        nothing,
    )
    requested_count = validation_count === nothing ?
        (
            declared_count === nothing ?
            round(Int, length(train_ids) * Float64(validation_fraction)) :
            Int(declared_count)
        ) :
        Int(validation_count)
    resolved_validation_count = clamp(
        requested_count,
        1,
        length(train_ids) - 1,
    )
    requested_count == resolved_validation_count ||
        error("declared validation-from-train count is out of range")
    if manifest_validation_ids === nothing
        ordered = sort(
            train_ids;
            by=id -> (
                _validation_score(dataset.teacher_contract_sha256, id),
                id,
            ),
        )
        validation_ids =
            sort(copy(ordered[1:resolved_validation_count]))
        derivation_source = declared_count === nothing ?
            "caller_fraction_hash_selection" :
            "manifest_count_hash_selection"
    else
        validation_ids =
            sort(Int32.(collect(manifest_validation_ids)))
        length(validation_ids) == resolved_validation_count ||
            error(
                "manifest validation ID count differs from " *
                "validation_trials_from_train",
            )
        all(id -> id in train_ids, validation_ids) ||
            error("manifest validation IDs are not all split-code 1")
        length(unique(validation_ids)) == length(validation_ids) ||
            error("manifest validation IDs contain duplicates")
        derivation_source =
            "manifest.validation_from_train_indices"
    end
    validation_set = Set(validation_ids)
    fit_ids = sort(Int32[
        id for id in train_ids if id ∉ validation_set
    ])
    sort!(test_ids)
    validation_ids_sha256 = _canonical_sha256(validation_ids)
    resolved_fraction =
        resolved_validation_count / length(train_ids)
    derivation =
        derivation_source == "manifest.validation_from_train_indices" ?
        "exact_manifest_validation_from_train_indices" :
        "sort_by_sha256(contract + ':validation-from-train:' + sample_id);" *
        "take_exact_count_from_" * derivation_source
    partition_payload = (;
        teacher_contract_sha256=dataset.teacher_contract_sha256,
        validation_fraction=resolved_fraction,
        validation_count=resolved_validation_count,
        derivation_source,
        derivation,
        fit_sample_ids=fit_ids,
        validation_sample_ids=validation_ids,
        test_sample_ids=test_ids,
        validation_ids_sha256,
    )
    return OfficialValidationPartition(
        fit_ids,
        validation_ids,
        test_ids,
        resolved_fraction,
        derivation,
        validation_ids_sha256,
        _canonical_sha256(partition_payload),
    )
end

function _partition_indices(
    data,
    split::Symbol,
    partition::OfficialValidationPartition,
)
    wanted = split === :train ?
        Set(partition.fit_sample_ids) :
        split === :validation ?
            Set(partition.validation_sample_ids) :
        split === :test ?
            Set(partition.test_sample_ids) :
            throw(ArgumentError("invalid partition split"))
    ids = vec(Int32.(data["sample_indices"]))
    return findall(id -> id in wanted, ids)
end

function load_official_shard(
    dataset::OfficialTeacherDataset,
    shard_index::Integer;
    include_targets::Bool=true,
)
    1 <= shard_index <= length(dataset.shard_paths) ||
        throw(BoundsError(dataset.shard_paths, shard_index))
    path = dataset.shard_paths[Int(shard_index)]
    _file_sha256(path) == dataset.shard_sha256[Int(shard_index)] ||
        error("official shard changed after dataset verification: $path")
    event_keys = !isempty(dataset.dense_event_key) ?
        [dataset.dense_event_key] :
        vcat(
            COMPACT_EVENT_KEYS,
            dataset.compact_event_count ? ["event_count"] : String[],
        )
    keys = vcat(
        [
            "sample_indices",
            "split_code",
            "contact_axon",
            "contact_segment",
            "contact_kind",
            "contact_strength",
            "axon_kind",
        ],
        dataset.ragged_contacts ?
            ["contact_trial_offset"] : String[],
        event_keys,
        include_targets ?
            ["target_voltage", "target_spike", "target_nmda"] :
            String[],
    )
    data = _npz_read(path, keys)
    samples = length(vec(data["split_code"]))
    if dataset.ragged_contacts
        offsets = vec(Int64.(data["contact_trial_offset"]))
        length(offsets) == samples + 1 ||
            error("contact_trial_offset must have trial_count+1 entries")
        first(offsets) == 0 ||
            error("contact_trial_offset must start at zero")
        issorted(offsets) ||
            error("contact_trial_offset must be monotonic")
        contacts = Int(last(offsets))
        for name in (
            "contact_axon",
            "contact_segment",
            "contact_kind",
            "contact_strength",
        )
            length(vec(data[name])) == contacts ||
                error("$name ragged payload length differs")
        end
    else
        contacts = size(data["contact_segment"], 1)
        for name in (
            "contact_axon",
            "contact_segment",
            "contact_kind",
            "contact_strength",
        )
            size(data[name]) == (contacts, samples) ||
                error("$name has an invalid shape in $path")
        end
    end
    size(data["axon_kind"]) == (dataset.axons, samples) ||
        error("axon_kind has an invalid shape in $path")
    if include_targets
        size(data["target_voltage"]) ==
            (dataset.time_steps, samples) ||
            error("target_voltage has an invalid shape in $path")
        size(data["target_spike"]) ==
            (dataset.time_steps, samples) ||
            error("target_spike has an invalid shape in $path")
        size(data["target_nmda"]) ==
            (4, dataset.time_steps, samples) ||
            error("target_nmda has an invalid shape in $path")
    end
    return data
end

@inline function _contact_range(
    dataset::OfficialTeacherDataset,
    data,
    item::Int,
)
    if dataset.ragged_contacts
        offsets = vec(data["contact_trial_offset"])
        return (Int(offsets[item]) + 1):Int(offsets[item + 1])
    end
    return axes(data["contact_segment"], 1)
end

@inline function _contact_value(
    dataset::OfficialTeacherDataset,
    data,
    name::String,
    contact::Int,
    item::Int,
)
    return dataset.ragged_contacts ?
        vec(data[name])[contact] :
        data[name][contact, item]
end

function _validate_contacts!(
    dataset::OfficialTeacherDataset,
    data,
    indices,
)
    for item in indices, contact in _contact_range(dataset, data, item)
        axon = Int(_contact_value(
            dataset,
            data,
            "contact_axon",
            contact,
            item,
        ))
        1 <= axon <= dataset.axons ||
            error("contact_axon is outside the declared axon catalog")
        segment = Int(_contact_value(
            dataset,
            data,
            "contact_segment",
            contact,
            item,
        ))
        FIRST_DENDRITIC_SEGMENT <= segment <=
            LAST_DENDRITIC_SEGMENT ||
            error(
                "official ELM accepts only Hay dendritic segments 2:640; " *
                "received segment $segment",
            )
        kind = UInt8(_contact_value(
            dataset,
            data,
            "contact_kind",
            contact,
            item,
        ))
        kind in (EXCITATORY, INHIBITORY) ||
            error("contact_kind must be 1=E or 2=I")
        UInt8(data["axon_kind"][axon, item]) == kind ||
            error("Dale-law violation between contact_kind and axon_kind")
        strength = Float32(_contact_value(
            dataset,
            data,
            "contact_strength",
            contact,
            item,
        ))
        isfinite(strength) && 0.0f0 <= strength <= 1.0f0 ||
            error("contact strength must be finite and nonnegative")
    end
    return nothing
end

function _axon_event_counts(
    dataset::OfficialTeacherDataset,
    data,
    indices,
    time_range::UnitRange{Int},
)
    batch = length(indices)
    chunk = length(time_range)
    result = zeros(UInt16, dataset.axons, chunk, batch)
    if haskey(data, "axon_event_spike") ||
       haskey(data, "axon_event_count")
        key = haskey(data, "axon_event_count") ?
            "axon_event_count" : "axon_event_spike"
        source = data[key]
        size(source, 1) == dataset.axons ||
            error("axon_event_spike axon dimension differs from manifest")
        size(source, 2) == dataset.time_steps ||
            error("axon_event_spike time dimension differs from manifest")
        @inbounds for (local_item, item) in enumerate(indices)
            for (local_time, time) in enumerate(time_range)
                for axon in 1:dataset.axons
                    count_value = Int(source[axon, time, item])
                    count_value >= 0 ||
                        error("dense axon event count must be nonnegative")
                    result[axon, local_time, local_item] =
                        UInt16(count_value)
                end
            end
        end
        return result
    end

    offsets = vec(Int64.(data["event_trial_offset"]))
    event_axon = vec(Int.(data["event_axon"]))
    event_time = vec(Int.(data["event_time_bin"]))
    event_count = dataset.compact_event_count ?
        vec(Int.(data["event_count"])) :
        ones(Int, length(event_axon))
    length(event_count) == length(event_axon) ||
        error("compact event_count length differs from event arrays")
    length(offsets) == length(vec(data["split_code"])) + 1 ||
        error("event_trial_offset must have trial_count+1 entries")
    first_time = first(time_range) - 1 # compact event bins are zero-based
    last_time = last(time_range) - 1
    @inbounds for (local_item, item) in enumerate(indices)
        first_event = Int(offsets[item]) + 1
        last_event = Int(offsets[item + 1])
        1 <= first_event <= length(event_axon) + 1 ||
            error("compact event offset is invalid")
        0 <= last_event <= length(event_axon) ||
            error("compact event offset is invalid")
        for event in first_event:last_event
            axon = event_axon[event]
            time_bin = event_time[event]
            1 <= axon <= dataset.axons ||
                error("compact event axon is invalid")
            if first_time <= time_bin <= last_time
                count_value = event_count[event]
                count_value >= 1 ||
                    error("compact event_count must be positive")
                result[
                    axon,
                    time_bin - first_time + 1,
                    local_item,
                ] += UInt16(count_value)
            end
        end
    end
    return result
end

"""
Expand official axon events into the exact PaperDigitalTwin input layout.

Every axon event is mapped through `contact_axon`; each contact retains its
full 1:642 Hay segment.  E contacts create paired AMPA/NMDA channels and I
contacts create GABA_A.  There is no receptor-total or region-total collapse.
"""
function expand_official_input_chunk(
    dataset::OfficialTeacherDataset,
    data,
    sample_indices,
    time_range::UnitRange{Int},
    config=nothing,
)
    if config !== nothing
        hasproperty(config, :input_dim) ||
            error("official ELM config has no input_dim")
        Int(getproperty(config, :input_dim)) == OFFICIAL_ELM_INPUT_DIM ||
            error(
                "official Spieler NeuronIO routing requires input_dim=" *
                "$OFFICIAL_ELM_INPUT_DIM",
            )
    end
    first(time_range) >= 1 &&
        last(time_range) <= dataset.time_steps ||
        throw(BoundsError(1:dataset.time_steps, time_range))
    indices = Int.(collect(sample_indices))
    isempty(indices) && error("cannot expand an empty sample batch")
    samples = length(vec(data["split_code"]))
    all(item -> 1 <= item <= samples, indices) ||
        throw(BoundsError(1:samples, indices))
    _validate_contacts!(dataset, data, indices)
    events = _axon_event_counts(
        dataset,
        data,
        indices,
        time_range,
    )
    output = zeros(
        Float32,
        OFFICIAL_ELM_INPUT_DIM,
        length(time_range),
        length(indices),
    )
    @inbounds for (local_item, item) in enumerate(indices)
        for contact in _contact_range(dataset, data, item)
            axon = Int(_contact_value(
                dataset, data, "contact_axon", contact, item,
            ))
            segment = Int(_contact_value(
                dataset, data, "contact_segment", contact, item,
            ))
            kind = UInt8(_contact_value(
                dataset, data, "contact_kind", contact, item,
            ))
            strength = Float32(_contact_value(
                dataset, data, "contact_strength", contact, item,
            ))
            local_segment = segment - 1
            feature = kind == EXCITATORY ?
                local_segment :
                DENDRITIC_SEGMENTS + local_segment
            sign_value = kind == EXCITATORY ? 1.0f0 : -1.0f0
            for local_time in eachindex(time_range)
                count_value = events[axon, local_time, local_item]
                count_value == 0 && continue
                output[feature, local_time, local_item] +=
                    sign_value * strength * Float32(count_value)
            end
        end
    end
    all(isfinite, output) ||
        error("official input expansion produced non-finite values")
    return output
end

struct TwinLossWeightsFinal
    voltage::Float32
    spike::Float32
    nmda::Float32
    huber_delta::Float32
end

TwinLossWeightsFinal(;
    voltage::Real=1,
    spike::Real=1,
    nmda::Real=1,
    huber_delta::Real=1,
) = TwinLossWeightsFinal(
    Float32(voltage),
    Float32(spike),
    Float32(nmda),
    Float32(huber_delta),
)

struct FinalTwinTrainingConfig
    preset::Symbol
    updates::Int
    batch_size::Int
    time_chunk::Int
    learning_rate::Float32
    weight_decay::Float32
    seed::UInt64
    log_interval::Int
    evaluation_samples::Int
    spike_auroc_gate::Float64
    allow_smoke_gate_override::Bool
    allow_reconstruction::Bool
end

function FinalTwinTrainingConfig(;
    preset::Symbol=:production,
    updates::Integer=227_500,
    batch_size::Integer=8,
    time_chunk::Integer=250,
    learning_rate::Real=3.0f-4,
    weight_decay::Real=1.0f-5,
    seed::Integer=0x5457494e46494e41,
    log_interval::Integer=100,
    evaluation_samples::Integer=0,
    spike_auroc_gate::Real=0.985,
    allow_smoke_gate_override::Bool=false,
    allow_reconstruction::Bool=false,
)
    updates >= 1 || throw(ArgumentError("updates must be positive"))
    batch_size >= 1 || throw(ArgumentError("batch_size must be positive"))
    time_chunk >= 1 || throw(ArgumentError("time_chunk must be positive"))
    evaluation_samples >= 0 ||
        throw(ArgumentError("evaluation_samples must be nonnegative"))
    gate = Float64(spike_auroc_gate)
    0 <= gate <= 1 || throw(ArgumentError("invalid spike AUROC gate"))
    if gate != 0.985
        preset === :smoke && allow_smoke_gate_override ||
            error(
                "the production spike gate is fixed at AUROC>=0.985; " *
                "an override is allowed only with preset=:smoke and " *
                "allow_smoke_gate_override=true",
            )
    end
    return FinalTwinTrainingConfig(
        preset,
        Int(updates),
        Int(batch_size),
        Int(time_chunk),
        Float32(learning_rate),
        Float32(weight_decay),
        UInt64(seed),
        Int(log_interval),
        Int(evaluation_samples),
        gate,
        allow_smoke_gate_override,
        allow_reconstruction,
    )
end

mutable struct _StreamingMoments
    count::Int64
    sum::Vector{Float64}
    sumsq::Vector{Float64}
end

_StreamingMoments(dimension::Int) =
    _StreamingMoments(0, zeros(dimension), zeros(dimension))

function _update_moments!(moments::_StreamingMoments, values)
    size(values, 1) == length(moments.sum) ||
        throw(DimensionMismatch("streaming moment dimension mismatch"))
    flat = reshape(values, size(values, 1), :)
    moments.count += size(flat, 2)
    moments.sum .+= vec(sum(Float64.(flat); dims=2))
    moments.sumsq .+= vec(sum(abs2, Float64.(flat); dims=2))
    return moments
end

function _finish_moments(moments::_StreamingMoments; epsilon=1.0f-5)
    moments.count > 0 || error("cannot finish empty streaming moments")
    mean_value = moments.sum ./ moments.count
    variance = max.(
        moments.sumsq ./ moments.count .- mean_value .* mean_value,
        0.0,
    )
    return Float32.(mean_value), Float32.(
        max.(sqrt.(variance), Float64(epsilon)),
    )
end

@inline _split_code(split::Symbol) =
    split === :train ? TRAIN_SPLIT :
    split === :validation ? VALIDATION_SPLIT :
    split === :test ? TEST_SPLIT :
    throw(ArgumentError("split must be :train, :validation or :test"))

function _split_indices(data, split::Symbol)
    return findall(==(_split_code(split)), vec(UInt8.(data["split_code"])))
end

function _time_ranges(time_steps::Int, chunk::Int)
    return (
        first:min(first + chunk - 1, time_steps)
        for first in 1:chunk:time_steps
    )
end

"""
Fit input/output normalizers from split_code==1 only.

Only one verified shard and one bounded time chunk are expanded at a time.
Validation and test samples present in the same NPZ are never accumulated.
"""
function fit_official_normalizer(
    dataset::OfficialTeacherDataset,
    config::TwinConfig;
    time_chunk::Integer=250,
    partition::Union{Nothing,OfficialValidationPartition}=nothing,
)
    input_moments = _StreamingMoments(config.input_dim)
    voltage_moments = _StreamingMoments(1)
    nmda_moments = _StreamingMoments(config.nmda_regions)
    spike_positive = 0
    spike_total = 0
    for shard_index in eachindex(dataset.shard_paths)
        dataset.shard_split_counts[shard_index].train == 0 && continue
        data = load_official_shard(dataset, shard_index)
        indices = partition === nothing ?
            _split_indices(data, :train) :
            _partition_indices(data, :train, partition)
        isempty(indices) && continue
        for range in _time_ranges(dataset.time_steps, Int(time_chunk))
            input = expand_official_input_chunk(
                dataset,
                data,
                indices,
                range,
                config,
            )
            _update_moments!(input_moments, input)
            voltage = reshape(
                Float32.(@view(data["target_voltage"][range, indices])),
                1,
                :,
            )
            _update_moments!(voltage_moments, voltage)
            nmda = Float32.(
                @view(data["target_nmda"][:, range, indices])
            )
            _update_moments!(nmda_moments, nmda)
            spike = @view data["target_spike"][range, indices]
            spike_positive += count(>=(0.5f0), spike)
            spike_total += length(spike)
        end
    end
    spike_total > 0 || error("official training split is empty")
    spike_positive > 0 ||
        error("official training split contains no soma spikes")
    input_mean, input_scale = _finish_moments(input_moments)
    voltage_mean, voltage_scale = _finish_moments(voltage_moments)
    nmda_mean, nmda_scale = _finish_moments(nmda_moments)
    spike_negative = spike_total - spike_positive
    positive_weight = Float32(clamp(
        spike_negative / spike_positive,
        1.0,
        100.0,
    ))
    normalizer = TwinNormalizer(
        input_mean,
        input_scale,
        only(voltage_mean),
        only(voltage_scale),
        nmda_mean,
        nmda_scale,
    )
    return normalizer, positive_weight, (;
        split="train",
        spike_positive,
        spike_negative,
        spike_positive_fraction=spike_positive / spike_total,
    )
end

@inline function _huber_mean(error, delta::Float32)
    absolute = abs.(error)
    return mean(ifelse.(
        absolute .<= delta,
        0.5f0 .* error .* error,
        delta .* (absolute .- 0.5f0 * delta),
    ))
end

function _objective(
    model::PaperTwin,
    parameters,
    normalizer::TwinNormalizer,
    input,
    target_voltage,
    target_spike,
    target_nmda,
    weights::TwinLossWeightsFinal,
    positive_weight::Float32,
    initial_memory,
)
    normalized_input = normalize_twin_input(normalizer, input)
    # Pure PaperDigitalTwin trajectory is the canonical correctness path.
    prediction = twin_forward(
        model,
        parameters,
        normalized_input;
        initial_memory,
    )
    voltage_target =
        (target_voltage .- normalizer.voltage_mean) ./
        normalizer.voltage_scale
    nmda_target =
        (
            target_nmda .-
            reshape(normalizer.nmda_mean, :, 1, 1)
        ) ./ reshape(normalizer.nmda_scale, :, 1, 1)
    voltage_loss = _huber_mean(
        prediction.voltage .- voltage_target,
        weights.huber_delta,
    )
    spike_element =
        max.(prediction.spike_logit, 0.0f0) .-
        prediction.spike_logit .* target_spike .+
        log1p.(exp.(-abs.(prediction.spike_logit)))
    class_weight = ifelse.(
        target_spike .>= 0.5f0,
        positive_weight,
        1.0f0,
    )
    spike_loss = sum(class_weight .* spike_element) / sum(class_weight)
    nmda_loss = mean(abs2, prediction.nmda .- nmda_target)
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

function _advance_fixed_memory(
    model::PaperTwin,
    normalized_input,
    initial_memory,
)
    memory = copy(initial_memory)
    for time in axes(normalized_input, 2)
        drive = tanh.(
            model.input_weight *
            @view(normalized_input[:, time, :]) .+
            model.input_bias
        )
        memory =
            model.decay .* memory .+
            model.injection .* drive
    end
    return memory
end

function _physical_prediction(
    model,
    parameters,
    normalizer,
    input,
    initial_memory,
)
    normalized_input = normalize_twin_input(normalizer, input)
    raw = twin_forward(
        model,
        parameters,
        normalized_input;
        initial_memory,
    )
    return denormalize_twin_output(normalizer, raw)
end

function _take_split_indices(indices, remaining::Int)
    remaining < 0 && return indices
    return first(indices, min(length(indices), remaining))
end

"""
Recompute one split directly from verified official NPZ targets.

The final test call uses `maximum_samples=0`, meaning every held-out sample.
Predictions are generated in bounded time chunks and passed to `twin_metrics`;
no manifest- or training-reported metric is trusted.
"""
function evaluate_official_split(
    dataset::OfficialTeacherDataset,
    model::PaperTwin,
    parameters,
    normalizer::TwinNormalizer,
    split::Symbol;
    maximum_samples::Integer=0,
    batch_size::Integer=8,
    time_chunk::Integer=250,
    partition::Union{Nothing,OfficialValidationPartition}=nothing,
)
    limit = maximum_samples == 0 ? -1 : Int(maximum_samples)
    prediction_voltage = Matrix{Float32}[]
    prediction_spike = Matrix{Float32}[]
    prediction_nmda = Array{Float32,3}[]
    target_voltage_all = Matrix{Float32}[]
    target_spike_all = Matrix{Float32}[]
    target_nmda_all = Array{Float32,3}[]
    samples = 0
    for shard_index in eachindex(dataset.shard_paths)
        limit >= 0 && samples >= limit && break
        counts = dataset.shard_split_counts[shard_index]
        split === :test && counts.test == 0 && continue
        split !== :test && counts.train == 0 && continue
        data = load_official_shard(dataset, shard_index)
        available = partition === nothing ?
            _split_indices(data, split) :
            _partition_indices(data, split, partition)
        isempty(available) && continue
        remaining = limit < 0 ? -1 : limit - samples
        selected = _take_split_indices(available, remaining)
        for batch_first in 1:Int(batch_size):length(selected)
            batch_last = min(
                batch_first + Int(batch_size) - 1,
                length(selected),
            )
            indices = selected[batch_first:batch_last]
            count_batch = length(indices)
            voltage_pred =
                Matrix{Float32}(undef, dataset.time_steps, count_batch)
            spike_pred =
                Matrix{Float32}(undef, dataset.time_steps, count_batch)
            nmda_pred = Array{Float32,3}(
                undef,
                model.config.nmda_regions,
                dataset.time_steps,
                count_batch,
            )
            memory = zeros(
                Float32,
                model.config.memory_units,
                count_batch,
            )
            for range in _time_ranges(
                dataset.time_steps,
                Int(time_chunk),
            )
                input = expand_official_input_chunk(
                    dataset,
                    data,
                    indices,
                    range,
                    model.config,
                )
                prediction = _physical_prediction(
                    model,
                    parameters,
                    normalizer,
                    input,
                    memory,
                )
                voltage_pred[range, :] .= prediction.voltage
                spike_pred[range, :] .= prediction.spike_probability
                nmda_pred[:, range, :] .= prediction.nmda
                normalized =
                    normalize_twin_input(normalizer, input)
                memory = _advance_fixed_memory(
                    model,
                    normalized,
                    memory,
                )
            end
            push!(prediction_voltage, voltage_pred)
            push!(prediction_spike, spike_pred)
            push!(prediction_nmda, nmda_pred)
            push!(
                target_voltage_all,
                Float32.(@view(data["target_voltage"][:, indices])),
            )
            push!(
                target_spike_all,
                Float32.(@view(data["target_spike"][:, indices])),
            )
            push!(
                target_nmda_all,
                Float32.(@view(data["target_nmda"][:, :, indices])),
            )
            samples += count_batch
        end
    end
    samples > 0 || error("official split $split has no samples")
    prediction = (;
        voltage=reduce(hcat, prediction_voltage),
        spike_probability=reduce(hcat, prediction_spike),
        nmda=cat(prediction_nmda...; dims=3),
    )
    metrics = twin_metrics(
        prediction,
        reduce(hcat, target_voltage_all),
        reduce(hcat, target_spike_all),
        cat(target_nmda_all...; dims=3);
        normalizer,
    )
    return merge(metrics, (;
        split=String(split),
        samples,
        source="recomputed_from_verified_official_npz",
    ))
end

function _update_binary_digest!(context, value)
    if value isa NamedTuple
        for (name, child) in pairs(value)
            SHA.update!(context, codeunits(String(name)))
            _update_binary_digest!(context, child)
        end
    elseif value isa AbstractArray
        SHA.update!(context, codeunits(string(eltype(value))))
        SHA.update!(context, codeunits(join(size(value), ",")))
        array = vec(Array(value))
        SHA.update!(context, reinterpret(UInt8, array))
    elseif value isa TwinConfig || value isa TwinNormalizer
        for name in fieldnames(typeof(value))
            SHA.update!(context, codeunits(String(name)))
            _update_binary_digest!(context, getfield(value, name))
        end
    else
        SHA.update!(context, codeunits(repr(value)))
    end
    return context
end

function _bank_config_normalizer_sha256(model, normalizer)
    context = SHA.SHA2_256_CTX()
    _update_binary_digest!(context, model.config)
    _update_binary_digest!(context, model.input_weight)
    _update_binary_digest!(context, model.input_bias)
    _update_binary_digest!(context, model.decay)
    _update_binary_digest!(context, model.injection)
    _update_binary_digest!(context, normalizer)
    return bytes2hex(SHA.digest!(context))
end

function _shard_lineage(dataset::OfficialTeacherDataset)
    return [
        (;
            path=basename(path),
            sha256=digest,
        )
        for (path, digest) in
            zip(dataset.shard_paths, dataset.shard_sha256)
    ]
end

function _save_final_artifact(
    path,
    frozen,
    lineage_manifest,
    lineage_manifest_sha256,
)
    mkpath(dirname(abspath(path)))
    integrity = assert_frozen_unchanged(frozen)
    integrity.max_delta == 0.0f0 ||
        error("cannot save a changed digital twin")
    jldsave(
        path;
        frozen,
        integrity,
        lineage_manifest,
        lineage_manifest_sha256,
    )
    return abspath(path)
end

function verify_frozen_twin_final(
    path::AbstractString;
    require_gate_pass::Bool=true,
)
    data = JLD2.load(path)
    for key in (
        "frozen",
        "integrity",
        "lineage_manifest",
        "lineage_manifest_sha256",
    )
        haskey(data, key) ||
            error("Final frozen artifact lacks $key: $path")
    end
    frozen = data["frozen"]
    frozen isa FrozenTwin ||
        error("Final artifact has an incompatible frozen twin")
    integrity = assert_frozen_unchanged(frozen)
    integrity.max_delta == 0.0f0 ||
        error("Final frozen twin max_delta is not zero")
    lineage = data["lineage_manifest"]
    String(_get(lineage, :schema_name, "")) == FINAL_LINEAGE_SCHEMA ||
        error("Final artifact lineage schema mismatch")
    declared = _require_hex_digest(
        data["lineage_manifest_sha256"],
        "lineage_manifest_sha256",
    )
    calculated = _canonical_sha256(lineage)
    calculated == declared ||
        error("Final artifact lineage manifest digest mismatch")
    String(_get(lineage, :parameter_sha256, "")) ==
        frozen.parameter_sha256 ||
        error("lineage parameter digest differs from frozen twin")
    String(_get(lineage, :digital_twin_sha256, "")) ==
        frozen.artifact_sha256 ||
        error("lineage digital-twin digest differs from frozen twin")
    String(_get(
        lineage,
        :bank_config_normalizer_sha256,
        "",
    )) == _bank_config_normalizer_sha256(
        frozen.model,
        frozen.normalizer,
    ) || error("lineage bank/config/normalizer digest mismatch")
    Float64(_get(lineage, :frozen_max_delta, Inf)) == 0.0 ||
        error("lineage does not certify frozen_max_delta=0")
    gate = _get(lineage, :promotion_gate)
    gate === nothing && error("lineage has no promotion_gate")
    require_gate_pass && !Bool(_get(gate, :passed, false)) &&
        error("Final artifact did not pass its held-out promotion gate")
    return (;
        frozen,
        lineage_manifest=lineage,
        lineage_manifest_sha256=declared,
        integrity,
    )
end

load_frozen_twin_final(path::AbstractString; kwargs...) =
    verify_frozen_twin_final(path; kwargs...).frozen

"""
Schema/hash/location adapter smoke.  It intentionally does not fit a model
and therefore also works on the one-trial direct NEURON fixture.
"""
function adapter_smoke(dataset_root::AbstractString)
    dataset = verify_official_teacher_dataset(
        dataset_root;
        require_all_splits=false,
    )
    data = load_official_shard(dataset, 1)
    indices = 1:min(2, length(vec(data["split_code"])))
    range = 1:min(16, dataset.time_steps)
    input = expand_official_input_chunk(
        dataset,
        data,
        indices,
        range,
        nothing,
    )
    all(isfinite, input) || error("adapter smoke generated non-finite input")
    return (;
        schema_name=dataset.schema_name,
        official=true,
        teacher_contract_sha256=dataset.teacher_contract_sha256,
        manifest_sha256=dataset.manifest_sha256,
        segment_mapping_sha256=dataset.segment_mapping_sha256,
        segments=EXPECTED_SEGMENTS,
        dendritic_segments=DENDRITIC_SEGMENTS,
        input_dim=OFFICIAL_ELM_INPUT_DIM,
        memory_units=1_000,
        core_dim=2_000,
        input_shape=size(input),
        active_features=count(!iszero, input),
        split_counts=_split_counts(vec(UInt8.(data["split_code"]))),
        paper_scale=dataset.paper_scale,
        promotable_production=dataset.promotable_production,
        connectivity_policy=dataset.connectivity_policy,
    )
end

function train_digital_twin_final(
    dataset_root::AbstractString,
    checkpoint_path::AbstractString;
    training::FinalTwinTrainingConfig=FinalTwinTrainingConfig(),
    weights::TwinLossWeightsFinal=TwinLossWeightsFinal(),
)
    training.allow_reconstruction &&
        error(
            "this Final entry is the official NPZ promotion path; " *
            "reconstruction controls must use train_digital_twin.jl and " *
            "cannot produce a Final promotable artifact",
        )
    dataset = verify_official_teacher_dataset(
        dataset_root;
        allow_reconstruction=false,
        require_all_splits=true,
    )
    config = TwinConfig(
        segments=EXPECTED_SEGMENTS,
        nmda_regions=4,
        memory_units=1_000,
        core_dim=2_000,
        dt_ms=dataset.sample_dt_ms,
        tau_min_ms=0.1,
        tau_max_ms=300.0,
        bank_seed=0x5457494e50524f50,
    )
    model = build_paper_twin(config)
    normalizer, positive_weight, spike_statistics =
        fit_official_normalizer(
            dataset,
            config;
            time_chunk=training.time_chunk,
        )
    rng = Xoshiro(training.seed)
    parameters, _ = Lux.setup(rng, model)
    optimizer = Optimisers.AdamW(
        training.learning_rate,
        (0.9, 0.999),
        training.weight_decay,
    )
    optimizer_state = Optimisers.setup(optimizer, parameters)

    initial_validation = evaluate_official_split(
        dataset,
        model,
        parameters,
        normalizer,
        :validation;
        maximum_samples=training.evaluation_samples,
        batch_size=training.batch_size,
        time_chunk=training.time_chunk,
    )

    train_shards = Int[]
    for shard_index in eachindex(dataset.shard_paths)
        codes = _npz_read(
            dataset.shard_paths[shard_index],
            ["split_code"],
        )["split_code"]
        isempty(findall(==(TRAIN_SPLIT), vec(UInt8.(codes)))) ||
            push!(train_shards, shard_index)
    end
    isempty(train_shards) && error("verified official dataset has no train shard")

    losses = Float64[]
    components = NamedTuple[]
    shard_cursor = 0
    data = nothing
    batch_indices = Int[]
    ranges = UnitRange{Int}[]
    range_cursor = 0
    memory = Matrix{Float32}(undef, config.memory_units, 0)
    started = time()
    for update in 1:training.updates
        if data === nothing || range_cursor >= length(ranges)
            shard_cursor = mod1(shard_cursor + 1, length(train_shards))
            data = load_official_shard(
                dataset,
                train_shards[shard_cursor],
            )
            available = _split_indices(data, :train)
            count_batch = min(training.batch_size, length(available))
            batch_indices = rand(rng, available, count_batch)
            ranges = collect(_time_ranges(
                dataset.time_steps,
                training.time_chunk,
            ))
            range_cursor = 0
            memory = zeros(Float32, config.memory_units, count_batch)
        end
        range_cursor += 1
        range = ranges[range_cursor]
        input = expand_official_input_chunk(
            dataset,
            data,
            batch_indices,
            range,
            config,
        )
        target_voltage =
            Float32.(@view(data["target_voltage"][range, batch_indices]))
        target_spike =
            Float32.(@view(data["target_spike"][range, batch_indices]))
        target_nmda =
            Float32.(@view(data["target_nmda"][:, range, batch_indices]))
        loss, pullback = Zygote.pullback(parameters) do candidate
            value, _ = _objective(
                model,
                candidate,
                normalizer,
                input,
                target_voltage,
                target_spike,
                target_nmda,
                weights,
                positive_weight,
                memory,
            )
            return value
        end
        isfinite(loss) ||
            error("non-finite Final digital-twin loss at update $update")
        gradient = only(pullback(one(loss)))
        optimizer_state, parameters = Optimisers.update(
            optimizer_state,
            parameters,
            gradient,
        )
        _, current_components = _objective(
            model,
            parameters,
            normalizer,
            input,
            target_voltage,
            target_spike,
            target_nmda,
            weights,
            positive_weight,
            memory,
        )
        normalized = normalize_twin_input(normalizer, input)
        memory = _advance_fixed_memory(model, normalized, memory)
        push!(losses, Float64(loss))
        push!(components, current_components)
        if update == 1 ||
           update % training.log_interval == 0 ||
           update == training.updates
            @info "official NEURON -> PaperDigitalTwin update" update loss current_components shard=train_shards[shard_cursor] time_range=range elapsed_seconds=time()-started
        end
    end

    final_validation = evaluate_official_split(
        dataset,
        model,
        parameters,
        normalizer,
        :validation;
        maximum_samples=training.evaluation_samples,
        batch_size=training.batch_size,
        time_chunk=training.time_chunk,
    )

    # Re-verify all bytes immediately before the held-out promotion metric.
    verified_again = verify_official_teacher_dataset(
        dataset.root;
        require_all_splits=true,
    )
    verified_again.manifest_sha256 == dataset.manifest_sha256 ||
        error("official teacher manifest changed during twin fitting")
    held_out_test = evaluate_official_split(
        verified_again,
        model,
        parameters,
        normalizer,
        :test;
        maximum_samples=training.evaluation_samples,
        batch_size=training.batch_size,
        time_chunk=training.time_chunk,
    )
    gate_passed =
        isfinite(held_out_test.spike_auroc) &&
        held_out_test.spike_auroc >= training.spike_auroc_gate
    gate_passed ||
        error(
            "refusing to freeze: recomputed official held-out spike AUROC " *
            "$(held_out_test.spike_auroc) is below " *
            "$(training.spike_auroc_gate)",
        )

    base_metadata = (;
        generated_at=string(now()),
        official_neuron_teacher=true,
        official_teacher_schema=OFFICIAL_TEACHER_SCHEMA,
        teacher_contract_sha256=dataset.teacher_contract_sha256,
        source_manifest_sha256=dataset.manifest_sha256,
        source_hashes=dataset.source_hashes,
        source_shards=_shard_lineage(dataset),
        total_segments=EXPECTED_SEGMENTS,
        segment_mapping=dataset.segment_mapping,
        segment_mapping_sha256=dataset.segment_mapping_sha256,
        memory_units=1_000,
        core_dim=2_000,
        tau_min_ms=0.1,
        tau_max_ms=300.0,
        loss_targets=(
            "soma_voltage_huber",
            "soma_spike_weighted_bce",
            "region_nmda_normalized_mse",
        ),
        train_split_code=Int(TRAIN_SPLIT),
        validation_split_code=Int(VALIDATION_SPLIT),
        test_split_code=Int(TEST_SPLIT),
        validation_metrics=final_validation,
        held_out_test_metrics=held_out_test,
        promotion_gate=(;
            metric="spike_auroc",
            threshold=training.spike_auroc_gate,
            passed=gate_passed,
            override=training.spike_auroc_gate != 0.985,
            override_scope=training.preset === :smoke ?
                "explicit_smoke_only" : "none",
        ),
        reconstruction_control=false,
    )
    provisional = freeze_twin(
        model,
        parameters,
        normalizer;
        metadata=base_metadata,
    )
    frozen_delta = frozen_max_delta(provisional, parameters)
    frozen_delta == 0.0f0 ||
        error("newly frozen twin differs from trained parameters")
    code_sha256 = _file_sha256(@__FILE__)
    lineage_manifest = (;
        schema_name=FINAL_LINEAGE_SCHEMA,
        generated_at=base_metadata.generated_at,
        official_neuron_teacher=true,
        teacher_schema=OFFICIAL_TEACHER_SCHEMA,
        teacher_contract_sha256=dataset.teacher_contract_sha256,
        source_manifest_sha256=dataset.manifest_sha256,
        source_hashes=dataset.source_hashes,
        source_shards=_shard_lineage(dataset),
        segment_mapping=dataset.segment_mapping,
        segment_mapping_sha256=dataset.segment_mapping_sha256,
        total_segments=EXPECTED_SEGMENTS,
        parameter_sha256=provisional.parameter_sha256,
        bank_config_normalizer_sha256=
            _bank_config_normalizer_sha256(model, normalizer),
        digital_twin_sha256=provisional.artifact_sha256,
        code_sha256,
        training=(;
            preset=String(training.preset),
            updates=training.updates,
            batch_size=training.batch_size,
            time_chunk=training.time_chunk,
            learning_rate=training.learning_rate,
            weight_decay=training.weight_decay,
            seed=string(training.seed),
            fixed_memory_units=1_000,
            fixed_core_dim=2_000,
            fixed_tau_ms=(0.1, 300.0),
            train_split_only=true,
            normalizer_fit_split="train",
        ),
        metrics=(;
            initial_validation,
            final_validation,
            held_out_test,
        ),
        promotion_gate=base_metadata.promotion_gate,
        frozen_max_delta=Float64(frozen_delta),
        reconstruction_control=false,
    )
    lineage_manifest_sha256 = _canonical_sha256(lineage_manifest)
    frozen = freeze_twin(
        model,
        parameters,
        normalizer;
        metadata=merge(
            base_metadata,
            (;
                lineage_manifest,
                lineage_manifest_sha256,
                code_sha256,
                frozen_max_delta=Float64(frozen_delta),
            ),
        ),
    )
    frozen.artifact_sha256 == provisional.artifact_sha256 ||
        error("metadata unexpectedly changed the digital-twin digest")
    artifact_path = _save_final_artifact(
        checkpoint_path,
        frozen,
        lineage_manifest,
        lineage_manifest_sha256,
    )
    verified = verify_frozen_twin_final(artifact_path)

    summary = (;
        schema_name=FINAL_LINEAGE_SCHEMA,
        generated_at=base_metadata.generated_at,
        artifact_path,
        official_neuron_teacher=true,
        teacher_contract_sha256=dataset.teacher_contract_sha256,
        source_manifest_sha256=dataset.manifest_sha256,
        segment_mapping_sha256=dataset.segment_mapping_sha256,
        parameter_sha256=frozen.parameter_sha256,
        digital_twin_sha256=frozen.artifact_sha256,
        lineage_manifest_sha256,
        frozen_max_delta=verified.integrity.max_delta,
        training_updates=training.updates,
        loss_first=first(losses),
        loss_last=last(losses),
        component_first=first(components),
        component_last=last(components),
        spike_statistics,
        initial_validation,
        final_validation,
        held_out_test,
        promotion_gate=base_metadata.promotion_gate,
        wall_seconds=time() - started,
    )
    summary_path = replace(
        artifact_path,
        r"\.jld2$" => ".json",
    )
    open(summary_path, "w") do stream
        JSON3.pretty(stream, summary)
    end
    return merge(summary, (; summary_path))
end

@inline function _env_bool(name::AbstractString, default::Bool=false)
    value = lowercase(strip(get(ENV, name, string(default))))
    value in ("1", "true", "yes", "on") && return true
    value in ("0", "false", "no", "off") && return false
    error("$name must be a boolean")
end

function main()
    mode = Symbol(lowercase(get(ENV, "TWIN_FINAL_MODE", "train")))
    dataset_root = abspath(get(
        ENV,
        "TWIN_DATASET_PATH",
        joinpath(@__DIR__, "artifacts", "neuron_teacher"),
    ))
    if mode === :adapter_smoke
        result = adapter_smoke(dataset_root)
        println(JSON3.write(result))
        return result
    elseif mode !== :train
        error("TWIN_FINAL_MODE must be train or adapter_smoke")
    end
    preset = Symbol(lowercase(get(
        ENV,
        "TWIN_TRAIN_PRESET",
        "production",
    )))
    smoke_override = _env_bool("TWIN_SMOKE_GATE_OVERRIDE", false)
    default_updates = preset === :production ? 227_500 : 200
    default_batch = preset === :production ? 8 : 2
    default_chunk = preset === :production ? 250 : 25
    gate = parse(
        Float64,
        get(ENV, "TWIN_SPIKE_AUROC_GATE", "0.985"),
    )
    training = FinalTwinTrainingConfig(
        preset,
        updates=parse(
            Int,
            get(ENV, "TWIN_TRAIN_UPDATES", string(default_updates)),
        ),
        batch_size=parse(
            Int,
            get(ENV, "TWIN_TRAIN_BATCH", string(default_batch)),
        ),
        time_chunk=parse(
            Int,
            get(ENV, "TWIN_TIME_CHUNK", string(default_chunk)),
        ),
        learning_rate=parse(
            Float32,
            get(ENV, "TWIN_TRAIN_LEARNING_RATE", "3e-4"),
        ),
        weight_decay=parse(
            Float32,
            get(ENV, "TWIN_TRAIN_WEIGHT_DECAY", "1e-5"),
        ),
        seed=parse(
            UInt64,
            get(ENV, "TWIN_TRAIN_SEED", "6077406352078495297"),
        ),
        log_interval=parse(
            Int,
            get(ENV, "TWIN_LOG_INTERVAL", "100"),
        ),
        evaluation_samples=parse(
            Int,
            get(ENV, "TWIN_EVALUATION_SAMPLES", "0"),
        ),
        spike_auroc_gate=gate,
        allow_smoke_gate_override=smoke_override,
        allow_reconstruction=_env_bool(
            "TWIN_ALLOW_RECONSTRUCTION",
            false,
        ),
    )
    checkpoint_path = abspath(get(
        ENV,
        "TWIN_CHECKPOINT_PATH",
        joinpath(
            @__DIR__,
            "artifacts",
            "hd_swsnn_official_digital_twin_final.jld2",
        ),
    ))
    result = train_digital_twin_final(
        dataset_root,
        checkpoint_path;
        training,
    )
    println(JSON3.write(result))
    return result
end

end # module DigitalTwinTrainingFinal

if abspath(PROGRAM_FILE) == @__FILE__
    DigitalTwinTrainingFinal.main()
end
