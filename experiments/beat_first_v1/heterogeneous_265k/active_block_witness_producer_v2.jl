module ActiveBlockWitnessProducerV2

"""Deterministic, fail-closed producer for the active-block v2 NPZ witness.

This file is an unexecuted implementation.  It accepts only an externally
hashed, already-trained k256 three-layer sparse checkpoint and manifest-bound
teacher_v3 *train* parts.  It never initializes weights and has no synthetic
fallback.  The emitted largest-support witness is shared verbatim by the CPU,
NPU, and iGPU component cells; k64 and k128 consume ordered prefixes.
"""

using JLD2
using JSON3
using SHA

if !isdefined(Main, :SparseDynamic3Layer)
    Base.include(
        Main,
        joinpath(@__DIR__, "..", "sparse_dynamic_3layer", "SparseDynamic3Layer.jl"),
    )
end
using Main.SparseDynamic3Layer

module CanonicalTeacherPacking
include(joinpath(@__DIR__, "..", "training", "core.jl"))
using .BeatFirstTrainingCore: allocate_host_batch, pack_batch!
end

const PRODUCER_SCHEMA = "heterogeneous-265k-active-block-witness-producer-v2"
const WITNESS_SCHEMA = "heterogeneous-265k-active-block-witness-v2"
const CONTRACT_SCHEMA =
    "heterogeneous-265k-active-block-accelerator-microbench-contract-v2"
const SOURCE_STATUS = "UNEXECUTED_STATIC_IMPLEMENTATION"
const SAMPLING_DOMAIN = "teacher_v3_train_active_block_v2"

const BATCH = 16
const MAX_WITNESS_CANDIDATES = 4_096
const LEARNER_WIDTH = 80
const LAYER_DIMS = (560, 321, 321)
const MAX_COUNTS = (96, 80, 80)
const NAMED_PREFIXES = (
    k64=(24, 20, 20),
    k128=(48, 40, 40),
    k256=(96, 80, 80),
)
const ALLOWED_TRAIN_ROLES = Set(("old_policy", "epsilon", "dagger"))
const FORBIDDEN_VALIDATION_SEEDS = Set(8001:8008)
const FORBIDDEN_SEALED_SEEDS = Set(91001:91032)
const FILE_ATTRIBUTE_REPARSE_POINT = UInt32(0x00000400)
const INVALID_FILE_ATTRIBUTES = typemax(UInt32)
const FIXED_ZIP_MTIME = 315_532_800.0 # 1980-01-01, the DOS ZIP epoch.
const ZIP_LOCAL_FILE_HEADER_SIGNATURE = UInt32(0x04034b50)
const ZIP_CENTRAL_DIRECTORY_SIGNATURE = UInt32(0x02014b50)
const ZIP_END_DIRECTORY_SIGNATURE = UInt32(0x06054b50)
const ZIP_VERSION_20 = UInt16(20)
const ZIP_DOS_TIME = UInt16(0)
const ZIP_DOS_DATE = UInt16(33) # 1980-01-01.

const CRC32_TABLE = ntuple(256) do index
    value = UInt32(index - 1)
    for _ in 1:8
        value = isodd(value) ? xor(value >> 1, UInt32(0xedb88320)) : value >> 1
    end
    value
end

const REQUIRED_FLAGS = Set((
    "--dataset-root",
    "--dataset-manifest",
    "--dataset-manifest-sha256",
    "--checkpoint",
    "--checkpoint-sha256",
    "--contract",
    "--contract-sha256",
    "--producer-sha256",
    "--candidate-count",
    "--sampling-domain",
    "--output-npz",
    "--output-metadata",
))

struct TrainPart
    episode_key::String
    role::String
    seed::Int
    relative_path::String
    path::String
    bytes::Int
    sha256::String
    states::Int
    candidates::Int
end

struct CandidateSelection
    output_index::Int
    global_ordinal::Int
    part_index::Int
    local_ordinal::Int
end

_sha256_file(path::AbstractString) = bytes2hex(open(sha256, path))
_valid_sha256(value) = value isa AbstractString &&
    occursin(r"^[0-9a-f]{64}$", String(value))

function _parse_cli(arguments::Vector{String})
    values = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        flag = arguments[index]
        flag in REQUIRED_FLAGS || throw(ArgumentError("unknown argument: $flag"))
        haskey(values, flag) && throw(ArgumentError("duplicate argument: $flag"))
        index == length(arguments) && throw(ArgumentError("missing value for $flag"))
        value = arguments[index + 1]
        startswith(value, "--") && throw(ArgumentError("missing value for $flag"))
        values[flag] = value
        index += 2
    end
    Set(keys(values)) == REQUIRED_FLAGS || throw(ArgumentError(
        "all producer arguments are mandatory; missing=$(sort!(collect(setdiff(REQUIRED_FLAGS, Set(keys(values))))))",
    ))
    return values
end

function _absolute_path(raw::AbstractString, label::AbstractString)
    isabspath(raw) || throw(ArgumentError("$label must be an explicit absolute path"))
    return normpath(abspath(raw))
end

function _windows_attributes(path::AbstractString)
    Sys.iswindows() || return UInt32(0)
    attributes = ccall(
        (:GetFileAttributesW, "kernel32"),
        stdcall,
        UInt32,
        (Cwstring,),
        path,
    )
    attributes == INVALID_FILE_ATTRIBUTES && error(
        "GetFileAttributesW failed for $path",
    )
    return attributes
end

function _path_prefixes(path::AbstractString)
    parts = splitpath(normpath(abspath(path)))
    isempty(parts) && error("cannot enumerate an empty path")
    prefixes = String[]
    current = first(parts)
    push!(prefixes, current)
    for component in Iterators.drop(parts, 1)
        current = joinpath(current, component)
        push!(prefixes, current)
    end
    return prefixes
end

"""Reject symlink/junction traversal for every existing path component."""
function _reject_reparse_chain(path::AbstractString, label::AbstractString)
    for prefix in _path_prefixes(path)
        ispath(prefix) || error("$label path component does not exist: $prefix")
        if Sys.iswindows()
            iszero(_windows_attributes(prefix) & FILE_ATTRIBUTE_REPARSE_POINT) ||
                error("$label traverses a Windows reparse point: $prefix")
        else
            !islink(prefix) || error("$label traverses a symbolic link: $prefix")
        end
    end
    return path
end

function _canonical_existing_file(raw::AbstractString, label::AbstractString)
    path = _absolute_path(raw, label)
    isfile(path) || throw(ArgumentError("$label is not an existing file: $path"))
    _reject_reparse_chain(path, label)
    return realpath(path)
end

function _canonical_existing_directory(raw::AbstractString, label::AbstractString)
    path = _absolute_path(raw, label)
    isdir(path) || throw(ArgumentError("$label is not an existing directory: $path"))
    _reject_reparse_chain(path, label)
    return realpath(path)
end

_path_key(path::AbstractString) = Sys.iswindows() ?
    lowercase(normpath(path)) : normpath(path)

function _is_strict_descendant(path::AbstractString, root::AbstractString)
    relative = relpath(path, root)
    parts = splitpath(relative)
    return !isabspath(relative) && relative != "." && !isempty(parts) &&
        first(parts) != ".."
end

function _fresh_output(raw::AbstractString, label::AbstractString)
    path = _absolute_path(raw, label)
    ispath(path) && error("refusing to overwrite $label: $path")
    parent = dirname(path)
    mkpath(parent)
    _reject_reparse_chain(parent, "$label parent")
    return path
end

function _json_property(value, name::Symbol, label::AbstractString)
    if value isa AbstractDict
        haskey(value, name) && return value[name]
        text = String(name)
        haskey(value, text) && return value[text]
    end
    hasproperty(value, name) || error("$label is missing $(String(name))")
    return getproperty(value, name)
end

function _strict_int(value, label::AbstractString)
    value isa Bool && error("$label must be an integer, not Bool")
    value isa Integer || error("$label must be an integer")
    return Int(value)
end

function _verified_input(path::String, expected::AbstractString, label::AbstractString)
    _valid_sha256(expected) || throw(ArgumentError(
        "$label expected SHA-256 must be lowercase hexadecimal",
    ))
    observed = _sha256_file(path)
    observed == expected || error("$label SHA-256 mismatch")
    return observed
end

function _validate_contract(path::String, expected_sha256::String)
    observed = _verified_input(path, expected_sha256, "active-block v2 contract")
    contract = JSON3.read(read(path, String))
    String(_json_property(contract, :schema, "contract")) == CONTRACT_SCHEMA ||
        error("active-block contract schema mismatch")
    String(_json_property(contract, :status, "contract")) ==
        "UNEXECUTED_STATIC_CONTRACT" || error("active-block contract status mismatch")
    geometry = _json_property(contract, :fixed_geometry, "contract")
    Int.(_json_property(geometry, :layer_dimensions, "contract geometry")) ==
        collect(LAYER_DIMS) || error("contract layer dimensions changed")
    variants = _json_property(geometry, :variants, "contract geometry")
    for name in propertynames(NAMED_PREFIXES)
        observed_variant = _json_property(variants, name, "contract variants")
        Int.(_json_property(observed_variant, :active_counts, "contract variant")) ==
            collect(getproperty(NAMED_PREFIXES, name)) || error(
            "contract active counts changed for $name",
        )
    end
    return observed
end

function _validate_manifest_header(manifest, manifest_sha256::String)
    _strict_int(_json_property(manifest, :format_version, "dataset manifest"),
        "dataset format version") == 3 || error("teacher_v3 manifest format mismatch")
    run_metadata = _json_property(manifest, :run_metadata, "dataset manifest")
    _json_property(
        run_metadata,
        :held_out_development_validation_sealed_seeds_used,
        "dataset run metadata",
    ) === false || error("teacher_v3 manifest does not attest reserved-seed exclusion")
    counts = _json_property(manifest, :counts, "dataset manifest")
    _strict_int(_json_property(counts, Symbol("states.train"), "dataset counts"),
        "training state count") >= 100_000 || error(
        "teacher_v3 manifest has fewer than 100,000 train states",
    )
    _valid_sha256(manifest_sha256) || error("dataset manifest digest is not canonical")
    return nothing
end

function _validate_relative_part_path(relative::String, role::String, seed::Int)
    isempty(relative) && error("teacher_v3 train part has an empty path")
    isabspath(relative) && error("teacher_v3 train part path must be relative")
    normalized = normpath(relative)
    components = splitpath(normalized)
    (normalized != "." && !isempty(components) && first(components) != "..") ||
        error("teacher_v3 train part path escapes the dataset root")
    name = basename(normalized)
    startswith(name, "part__train__$(role)__seed$(seed)") ||
        error("teacher_v3 train part filename namespace mismatch")
    lowercase(splitext(normalized)[2]) == ".jld2" || error(
        "teacher_v3 train part must be a JLD2 file",
    )
    return normalized
end

function _validate_train_part_index(
    path::String,
    episode_key::String,
    role::String,
    seed::Int,
    states::Int,
    candidates::Int,
)
    jldopen(path, "r") do file
        haskey(file, "metadata") || error("train part lacks metadata")
        haskey(file, "action_counts") || error("train part lacks action_counts")
        metadata = file["metadata"]
        String(_json_property(metadata, :episode_key, "part metadata")) == episode_key ||
            error("JLD2 episode key differs from manifest")
        String(_json_property(metadata, :split, "part metadata")) == "train" ||
            error("JLD2 part is not train-only")
        String(_json_property(metadata, :role, "part metadata")) == role ||
            error("JLD2 role differs from manifest")
        _strict_int(_json_property(metadata, :seed, "part metadata"), "JLD2 seed") ==
            seed || error("JLD2 seed differs from manifest")
        action_counts = Int.(file["action_counts"])
        length(action_counts) == states || error("JLD2 row count differs from manifest")
        all(count -> 1 <= count <= LEARNER_WIDTH, action_counts) || error(
            "train part action count exceeds sparse learner width",
        )
        sum(action_counts) == candidates || error(
            "JLD2 candidate total differs from manifest",
        )
    end
    return nothing
end

function _load_verified_train_parts(dataset_root::String, manifest, manifest_sha256::String)
    _validate_manifest_header(manifest, manifest_sha256)
    raw_parts = collect(_json_property(manifest, :parts, "dataset manifest"))
    isempty(raw_parts) && error("teacher_v3 manifest has no parts")
    records = TrainPart[]
    episode_keys = Set{String}()
    canonical_paths = Set{String}()
    physical_files = Set{Tuple{UInt,UInt}}()
    for raw in raw_parts
        split = String(_json_property(raw, :split, "manifest part"))
        role = String(_json_property(raw, :role, "train part"))
        split in ("train", "validation") || error(
            "forbidden teacher_v3 split namespace: $split",
        )
        role in ALLOWED_TRAIN_ROLES || error("forbidden teacher_v3 rollout role: $role")
        seed = _strict_int(_json_property(raw, :seed, "train part"), "train seed")
        episode_key = String(_json_property(raw, :episode_key, "train part"))
        episode_key in episode_keys && error("duplicate teacher_v3 episode key")
        expected_prefix = "v3|$(split)|$(role)|$(seed)|"
        startswith(episode_key, expected_prefix) || error(
            "teacher_v3 episode-key namespace mismatch",
        )
        push!(episode_keys, episode_key)
        split == "train" || continue # Validation part bytes are never opened here.
        seed in FORBIDDEN_VALIDATION_SEEDS && error("validation seed in train namespace")
        seed in FORBIDDEN_SEALED_SEEDS && error("sealed seed in train namespace")

        relative = _validate_relative_part_path(
            String(_json_property(raw, :relative_path, "train part")), role, seed,
        )
        joined = normpath(joinpath(dataset_root, relative))
        isfile(joined) || error("teacher_v3 train part is missing: $joined")
        _reject_reparse_chain(joined, "teacher_v3 train part")
        canonical = realpath(joined)
        _is_strict_descendant(canonical, dataset_root) || error(
            "teacher_v3 train part resolves outside the dataset root",
        )
        key = _path_key(canonical)
        key in canonical_paths && error("teacher_v3 train manifest has a canonical alias")
        push!(canonical_paths, key)
        status = stat(canonical)
        physical_id = (status.device, status.inode)
        physical_id in physical_files && error(
            "teacher_v3 train manifest aliases one physical file through multiple paths",
        )
        push!(physical_files, physical_id)

        expected_bytes = _strict_int(_json_property(raw, :bytes, "train part"),
            "train part byte count")
        expected_bytes > 0 || error("teacher_v3 train part byte count must be positive")
        filesize(canonical) == expected_bytes || error("teacher_v3 train part byte mismatch")
        expected_sha = String(_json_property(raw, :sha256, "train part"))
        _valid_sha256(expected_sha) || error("teacher_v3 train part SHA-256 is invalid")
        _sha256_file(canonical) == expected_sha || error(
            "teacher_v3 train part SHA-256 mismatch",
        )
        states = _strict_int(_json_property(raw, :row_count, "train part"),
            "train part state count")
        candidates = _strict_int(_json_property(raw, :candidate_total, "train part"),
            "train part candidate count")
        states > 0 && candidates >= states || error("invalid train part counts")
        _validate_train_part_index(
            canonical,
            episode_key,
            role,
            seed,
            states,
            candidates,
        )
        push!(records, TrainPart(
            episode_key, role, seed, relative, canonical, expected_bytes,
            expected_sha, states, candidates,
        ))
    end
    isempty(records) && error("teacher_v3 manifest has no eligible train parts")
    sort!(records; by=part -> (part.seed, part.role, part.episode_key, part.relative_path))
    counts = _json_property(manifest, :counts, "dataset manifest")
    expected_states = _strict_int(
        _json_property(counts, Symbol("states.train"), "dataset counts"),
        "manifest train states",
    )
    expected_episodes = _strict_int(
        _json_property(counts, Symbol("episodes.train"), "dataset counts"),
        "manifest train episodes",
    )
    sum(part.states for part in records) == expected_states || error(
        "verified train-part states differ from manifest counts",
    )
    length(records) == expected_episodes || error(
        "verified train-part episodes differ from manifest counts",
    )
    return records
end

function _digest_fields(fields)
    context = SHA.SHA256_CTX()
    for field in fields
        bytes = Vector{UInt8}(codeunits(string(field)))
        length_bytes = reinterpret(UInt8, [htol(UInt64(length(bytes)))])
        SHA.update!(context, length_bytes)
        SHA.update!(context, bytes)
    end
    return bytes2hex(SHA.digest!(context))
end

function _train_corpus_digest(parts::Vector{TrainPart})
    fields = Any["teacher_v3_train_corpus_v1"]
    for part in parts
        append!(fields, (
            part.episode_key, part.role, part.seed, part.relative_path,
            part.bytes, part.sha256, part.states, part.candidates,
        ))
    end
    return _digest_fields(fields)
end

function _derive_positions(
    total::Int,
    count::Int,
    manifest_sha256::String,
    checkpoint_sha256::String,
    contract_sha256::String,
    producer_sha256::String,
    corpus_sha256::String,
    sampling_domain::String,
)
    0 < count <= total || error("candidate count exceeds the train candidate corpus")
    count % BATCH == 0 || error("candidate count must be a positive multiple of 16")
    sampling_domain == SAMPLING_DOMAIN || error("sampling domain mismatch")
    seed_digest = _digest_fields((
        "active_block_witness_sample_v2", sampling_domain, manifest_sha256,
        checkpoint_sha256, contract_sha256, producer_sha256, corpus_sha256,
        total, count,
    ))
    start_word = parse(UInt64, seed_digest[1:16]; base=16)
    stride_word = parse(UInt64, seed_digest[17:32]; base=16)
    offset = Int(mod(start_word, UInt64(total))) + 1
    stride = Int(mod(stride_word, UInt64(total - 1))) + 1
    while gcd(stride, total) != 1
        stride = stride == total - 1 ? 1 : stride + 1
    end
    positions = [mod(offset - 1 + index * stride, total) + 1 for index in 0:(count - 1)]
    allunique(positions) || error("deterministic corpus sampler produced a duplicate")
    sort!(positions)
    return positions, offset, stride, seed_digest
end

function _map_positions(parts::Vector{TrainPart}, positions::Vector{Int})
    selections = CandidateSelection[]
    sizehint!(selections, length(positions))
    part_index = 1
    preceding = 0
    for (output_index, global_ordinal) in enumerate(positions)
        while global_ordinal > preceding + parts[part_index].candidates
            preceding += parts[part_index].candidates
            part_index += 1
            part_index <= length(parts) || error("candidate ordinal exceeds corpus")
        end
        push!(selections, CandidateSelection(
            output_index,
            global_ordinal,
            part_index,
            global_ordinal - preceding,
        ))
    end
    return selections
end

function _required_jld2_keys()
    return (
        "boards", "placements", "ren", "back_to_back", "tspin", "queues",
        "teacher_q", "action_counts", "selected_actions", "rewards", "seed_ids",
        "episode_ids", "episode_steps", "terminal", "death", "line_clear",
        "max_height", "holes", "cavities", "metadata",
    )
end

function _load_train_part_dataset(part::TrainPart)
    return jldopen(part.path, "r") do file
        missing = filter(name -> !haskey(file, name), _required_jld2_keys())
        isempty(missing) || error("train part lacks required fields: $(join(missing, ','))")
        metadata = file["metadata"]
        String(_json_property(metadata, :episode_key, "part metadata")) ==
            part.episode_key || error("JLD2 episode key differs from manifest")
        String(_json_property(metadata, :split, "part metadata")) == "train" ||
            error("JLD2 part is not train-only")
        String(_json_property(metadata, :role, "part metadata")) == part.role ||
            error("JLD2 role differs from manifest")
        _strict_int(_json_property(metadata, :seed, "part metadata"), "JLD2 seed") ==
            part.seed || error("JLD2 seed differs from manifest")
        _strict_int(_json_property(metadata, :format_version, "part metadata"),
            "JLD2 format version") == 3 || error("JLD2 format version mismatch")
        _json_property(metadata, :preserves_candidate_order, "part metadata") === true ||
            error("JLD2 part does not preserve candidate order")
        _json_property(metadata, :preserves_candidate_multiplicity, "part metadata") === true ||
            error("JLD2 part does not preserve candidate multiplicity")

        action_counts = Int.(file["action_counts"])
        length(action_counts) == part.states || error("JLD2 row count differs from manifest")
        all(count -> 1 <= count <= LEARNER_WIDTH, action_counts) || error(
            "train part action count exceeds sparse learner width",
        )
        sum(action_counts) == part.candidates || error(
            "JLD2 candidate total differs from manifest",
        )
        death = Bool.(file["death"])
        return (;
            boards=file["boards"],
            placements=file["placements"],
            ren=Float32.(file["ren"]),
            back_to_back=Float32.(file["back_to_back"]),
            tspin=Float32.(file["tspin"]),
            queues=file["queues"],
            teacher_q=Float32.(file["teacher_q"]),
            action_counts,
            selected_actions=Int.(file["selected_actions"]),
            rewards=Float32.(file["rewards"]),
            seed_ids=Int.(file["seed_ids"]),
            episode_ids=Int.(file["episode_ids"]),
            split_group_ids=Int.(file["seed_ids"]),
            predefined_split=fill(:train, part.states),
            episode_steps=Int.(file["episode_steps"]),
            terminal=Bool.(file["terminal"]),
            candidate_death=death,
            candidate_death_available=trues(part.states),
            line_clear=Int8.(file["line_clear"]),
            max_height=Int8.(file["max_height"]),
            holes=Int16.(file["holes"]),
            cavities=Int16.(file["cavities"]),
            source_path=part.path,
            geometry_cache=nothing,
        )
    end
end

function _local_candidate(action_counts::Vector{Int}, local_ordinal::Int)
    remaining = local_ordinal
    for row in eachindex(action_counts)
        count = action_counts[row]
        if remaining <= count
            return row, remaining
        end
        remaining -= count
    end
    error("local candidate ordinal exceeds its part")
end

function _validate_sparse_checkpoint(
    checkpoint_path::String,
    checkpoint_sha256::String,
    dataset_root::String,
    dataset_manifest_sha256::String,
)
    _verified_input(checkpoint_path, checkpoint_sha256, "sparse checkpoint")
    loaded = SparseDynamic3Layer.load_checkpoint(checkpoint_path)
    topology = loaded.topology
    topology.parameter_count == SparseDynamic3Layer.TOTAL_PARAMETERS == 19_924_022 ||
        error("sparse checkpoint is not the exact 19.924M model")
    Tuple(Int.(topology.row_dims)) == LAYER_DIMS || error("checkpoint row geometry mismatch")
    Tuple(Int.(topology.neuron_counts)) ==
        SparseDynamic3Layer.LAYER_NEURON_COUNTS || error("checkpoint bank geometry mismatch")
    Tuple(Int.(topology.active_counts)) == MAX_COUNTS || error(
        "active-block witness requires a trained k256 checkpoint",
    )
    loaded.training_state === nothing && error(
        "checkpoint has no training state; initialized/untrained banks are forbidden",
    )
    state = loaded.training_state
    hasproperty(state, :update) || error("checkpoint training state has no update")
    update = _strict_int(state.update, "checkpoint update")
    update > 0 || error("checkpoint must contain at least one completed sparse update")
    hasproperty(state, :config) || error("checkpoint training state has no config")
    config = state.config
    String(config.variant) == "k256" || error("checkpoint config is not k256")
    String(config.dataset_manifest_sha256) == dataset_manifest_sha256 || error(
        "checkpoint config is not bound to the supplied teacher_v3 manifest",
    )
    configured_root = _canonical_existing_directory(
        String(config.dataset_path), "checkpoint-configured dataset root",
    )
    _path_key(configured_root) == _path_key(dataset_root) || error(
        "checkpoint was trained against a different dataset root",
    )
    metadata = loaded.metadata
    for key in (
        "source_sha256", "project_sha256", "manifest_sha256",
        "dataset_manifest_sha256",
    )
        haskey(metadata, key) || error("checkpoint metadata lacks $key")
        _valid_sha256(metadata[key]) || error("checkpoint metadata $key is not a SHA-256")
    end
    String(metadata["dataset_manifest_sha256"]) == dataset_manifest_sha256 || error(
        "checkpoint metadata is not bound to the supplied teacher_v3 manifest",
    )
    String(metadata["variant"]) == "k256" || error("checkpoint metadata is not k256")
    String(metadata["ranking_source"]) == "q" || error(
        "checkpoint ranking source is not q",
    )
    Int(metadata["ranking_output_index"]) == 1 || error(
        "checkpoint ranking output index is not one",
    )
    clocks = ntuple(i -> loaded.runtime.bank_optimizers[i].global_step, 3)
    all(clock -> clock == UInt64(update), clocks) || error(
        "checkpoint sparse optimizer clocks differ from its update",
    )
    loaded.runtime.head_optimizer.step == UInt64(update) || error(
        "checkpoint head optimizer clock differs from its update",
    )
    return loaded.runtime, update, Dict{String,Any}(metadata)
end

function _allocate_witness(count::Int)
    # Storage axes are deliberately reversed.  Julia column-major bytes then
    # equal NumPy C-order bytes for logical [N,K,D], [N,K], and [N,D] arrays.
    rows = ntuple(
        layer -> Array{Float32}(undef, LAYER_DIMS[layer], MAX_COUNTS[layer], count),
        3,
    )
    ids = ntuple(layer -> Array{Int32}(undef, MAX_COUNTS[layer], count), 3)
    inputs = ntuple(layer -> Array{Float32}(undef, LAYER_DIMS[layer], count), 3)
    return rows, ids, inputs
end

function _capture_candidate!(
    rows,
    ids,
    inputs,
    output_index::Int,
    runtime::SparseDynamic3Layer.ThreeLayerRuntime,
    workspace::SparseDynamic3Layer.ThreeLayerWorkspace,
    candidate_input,
    action::Int,
)
    result = SparseDynamic3Layer.route_forward!(
        runtime,
        workspace,
        candidate_input,
        action;
        training_probes=(0, 0, 0),
        probe_token=0,
    )
    all(isfinite, result.output) || error("sparse checkpoint produced non-finite output")
    maximum_error = 0.0
    for layer in 1:3
        tape_ids = result.tape.ids[layer]
        length(tape_ids) == MAX_COUNTS[layer] || error("k256 route width changed")
        allunique(tape_ids) || error("k256 route contains duplicate IDs")
        query = result.tape.queries[layer]
        value = result.tape.values[layer]
        row_dim = LAYER_DIMS[layer]
        route_scale = inv(sqrt(Float32(SparseDynamic3Layer.ROUTE_DIM)))
        value_scale = inv(sqrt(Float32(length(value)))
        for coordinate in 1:SparseDynamic3Layer.ROUTE_DIM
            inputs[layer][coordinate, output_index] = query[coordinate] * route_scale
        end
        for coordinate in eachindex(value)
            inputs[layer][SparseDynamic3Layer.ROUTE_DIM + coordinate, output_index] =
                value[coordinate] * value_scale
        end
        for position in eachindex(tape_ids)
            neuron_id = Int(tape_ids[position])
            1 <= neuron_id <= size(runtime.model.layers[layer].theta, 2) || error(
                "selected neuron ID is outside its bank",
            )
            ids[layer][position, output_index] = Int32(neuron_id)
            accumulator = 0.0f0
            for coordinate in 1:row_dim
                weight = runtime.model.layers[layer].theta[coordinate, neuron_id]
                isfinite(weight) || error("selected sparse weight is non-finite")
                rows[layer][coordinate, position, output_index] = weight
                accumulator = muladd(
                    weight,
                    inputs[layer][coordinate, output_index],
                    accumulator,
                )
            end
            maximum_error = max(
                maximum_error,
                abs(Float64(accumulator) -
                    Float64(result.tape.preactivation[layer][position])),
            )
        end
        all(isfinite, @view(inputs[layer][:, output_index])) || error(
            "scaled sparse input is non-finite",
        )
    end
    maximum_error <= 1.0e-4 || error(
        "scaled row-dot witness differs from the sparse preactivation",
    )
    return maximum_error
end

function _write_u16_le(io::IO, value::Integer)
    0 <= value <= typemax(UInt16) || error("NPY v1 header is too large")
    word = UInt16(value)
    write(io, UInt8(word & 0x00ff))
    write(io, UInt8((word >> 8) & 0x00ff))
    return nothing
end

function _write_u32_le(io::IO, value::Integer)
    0 <= value <= typemax(UInt32) || error("ZIP32 field is too large")
    word = UInt32(value)
    write(io, UInt8(word & 0x000000ff))
    write(io, UInt8((word >> 8) & 0x000000ff))
    write(io, UInt8((word >> 16) & 0x000000ff))
    write(io, UInt8((word >> 24) & 0x000000ff))
    return nothing
end

function _shape_literal(shape::Tuple)
    body = join(shape, ", ")
    return length(shape) == 1 ? "($body,)" : "($body)"
end

function _npy_header(logical_shape::Tuple, descriptor::String)
    header_core = "{'descr': '$descriptor', 'fortran_order': False, " *
        "'shape': $(_shape_literal(logical_shape)), }"
    header_length_without_padding = ncodeunits(header_core) + 1
    padding = mod(-10 - header_length_without_padding, 64)
    header = header_core * repeat(" ", padding) * "\n"
    io = IOBuffer()
    write(io, UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01, 0x00])
    _write_u16_le(io, ncodeunits(header))
    write(io, codeunits(header))
    return take!(io)
end

function _array_digest(storage::Array)
    return bytes2hex(sha256(reinterpret(UInt8, vec(storage))))
end

function _npz_members(rows, ids, inputs, count::Int)
    members = Tuple{String,Array,Tuple,String}[]
    for layer in 1:3
        push!(members, (
            "selected_rows_l$layer", rows[layer],
            (count, MAX_COUNTS[layer], LAYER_DIMS[layer]), "<f4",
        ))
    end
    for layer in 1:3
        push!(members, (
            "selected_ids_l$layer", ids[layer],
            (count, MAX_COUNTS[layer]), "<i4",
        ))
    end
    for layer in 1:3
        push!(members, (
            "scaled_input_l$layer", inputs[layer],
            (count, LAYER_DIMS[layer]), "<f4",
        ))
    end
    return members
end

function _crc32(chunks...)
    state = typemax(UInt32)
    for chunk in chunks
        for byte in chunk
            slot = Int((state ⊻ UInt32(byte)) & UInt32(0xff)) + 1
            state = CRC32_TABLE[slot] ⊻ (state >> 8)
        end
    end
    return state ⊻ typemax(UInt32)
end

function _write_zip_local_header(
    io::IO,
    filename::Vector{UInt8},
    crc32::UInt32,
    size::UInt32,
)
    _write_u32_le(io, ZIP_LOCAL_FILE_HEADER_SIGNATURE)
    _write_u16_le(io, ZIP_VERSION_20)
    _write_u16_le(io, 0) # general-purpose flag
    _write_u16_le(io, 0) # STORE
    _write_u16_le(io, ZIP_DOS_TIME)
    _write_u16_le(io, ZIP_DOS_DATE)
    _write_u32_le(io, crc32)
    _write_u32_le(io, size)
    _write_u32_le(io, size)
    _write_u16_le(io, length(filename))
    _write_u16_le(io, 0) # extra length
    write(io, filename)
    return nothing
end

function _write_zip_central_entry(io::IO, entry)
    _write_u32_le(io, ZIP_CENTRAL_DIRECTORY_SIGNATURE)
    _write_u16_le(io, ZIP_VERSION_20) # made by
    _write_u16_le(io, ZIP_VERSION_20) # needed
    _write_u16_le(io, 0) # flags
    _write_u16_le(io, 0) # STORE
    _write_u16_le(io, ZIP_DOS_TIME)
    _write_u16_le(io, ZIP_DOS_DATE)
    _write_u32_le(io, entry.crc32)
    _write_u32_le(io, entry.size)
    _write_u32_le(io, entry.size)
    _write_u16_le(io, length(entry.filename))
    _write_u16_le(io, 0) # extra
    _write_u16_le(io, 0) # comment
    _write_u16_le(io, 0) # disk
    _write_u16_le(io, 0) # internal attributes
    _write_u32_le(io, 0) # external attributes
    _write_u32_le(io, entry.offset)
    write(io, entry.filename)
    return nothing
end

function _write_zip_end(io::IO, count::Int, central_size::UInt32, central_offset::UInt32)
    count <= typemax(UInt16) || error("too many NPZ members for ZIP32")
    _write_u32_le(io, ZIP_END_DIRECTORY_SIGNATURE)
    _write_u16_le(io, 0)
    _write_u16_le(io, 0)
    _write_u16_le(io, count)
    _write_u16_le(io, count)
    _write_u32_le(io, central_size)
    _write_u32_le(io, central_offset)
    _write_u16_le(io, 0)
    return nothing
end

function _write_deterministic_npz(path::String, members)
    ENDIAN_BOM == 0x04030201 || error(
        "active-block NPZ writer requires little-endian x86",
    )
    file = Base.Filesystem.open(
        path,
        Base.Filesystem.JL_O_WRONLY |
            Base.Filesystem.JL_O_CREAT |
            Base.Filesystem.JL_O_EXCL,
        0o600,
    )
    entries = NamedTuple[]
    completed = false
    try
        for (name, storage, shape, descriptor) in members
            prod(shape) == length(storage) || error("NPY logical/storage size mismatch")
            header = _npy_header(shape, descriptor)
            raw = reinterpret(UInt8, vec(storage))
            data_size = length(header) + length(raw)
            data_size <= typemax(UInt32) || error("NPY member exceeds ZIP32")
            position(file) <= typemax(UInt32) || error("NPZ archive exceeds ZIP32")
            filename = Vector{UInt8}(codeunits(name * ".npy"))
            crc32 = _crc32(header, raw)
            entry = (;
                filename,
                crc32,
                size=UInt32(data_size),
                offset=UInt32(position(file)),
            )
            _write_zip_local_header(file, filename, crc32, entry.size)
            write(file, header)
            write(file, raw)
            push!(entries, entry)
        end
        position(file) <= typemax(UInt32) || error("NPZ archive exceeds ZIP32")
        central_offset = UInt32(position(file))
        for entry in entries
            _write_zip_central_entry(file, entry)
        end
        central_length = position(file) - Int(central_offset)
        central_length <= typemax(UInt32) || error("NPZ central directory exceeds ZIP32")
        _write_zip_end(file, length(entries), UInt32(central_length), central_offset)
        flush(file)
        completed = true
    finally
        close(file)
        !completed && isfile(path) && rm(path; force=true)
    end
    return path
end

function _write_metadata(path::String, metadata)
    io = Base.Filesystem.open(
        path,
        Base.Filesystem.JL_O_WRONLY |
            Base.Filesystem.JL_O_CREAT |
            Base.Filesystem.JL_O_EXCL,
        0o600,
    )
    try
        JSON3.pretty(io, metadata)
        write(io, '\n')
        flush(io)
    finally
        close(io)
    end
    return path
end

function _source_part_metadata(parts::Vector{TrainPart}, used_indices::Vector{Int})
    return [
        (;
            episode_key=parts[index].episode_key,
            role=parts[index].role,
            environment_seed=parts[index].seed,
            relative_path=parts[index].relative_path,
            bytes=parts[index].bytes,
            sha256=parts[index].sha256,
        ) for index in used_indices
    ]
end

function produce(arguments::Vector{String}=ARGS)
    cli = _parse_cli(arguments)
    sampling_domain = cli["--sampling-domain"]
    sampling_domain == SAMPLING_DOMAIN || throw(ArgumentError(
        "--sampling-domain must be exactly $SAMPLING_DOMAIN",
    ))
    candidate_count = tryparse(Int, cli["--candidate-count"])
    candidate_count === nothing && throw(ArgumentError("candidate count is not an integer"))
    candidate_count > 0 && candidate_count % BATCH == 0 || throw(ArgumentError(
        "candidate count must be a positive multiple of 16",
    ))
    candidate_count <= MAX_WITNESS_CANDIDATES || throw(ArgumentError(
        "candidate count exceeds deterministic ZIP32 witness limit $MAX_WITNESS_CANDIDATES",
    ))

    dataset_root = _canonical_existing_directory(cli["--dataset-root"], "dataset root")
    manifest_path = _canonical_existing_file(
        cli["--dataset-manifest"], "dataset manifest",
    )
    expected_manifest_path = realpath(joinpath(dataset_root, "manifest.json"))
    _path_key(manifest_path) == _path_key(expected_manifest_path) || error(
        "dataset manifest must be the dataset root's exact manifest.json",
    )
    manifest_sha256 = _verified_input(
        manifest_path,
        cli["--dataset-manifest-sha256"],
        "dataset manifest",
    )

    contract_path = _canonical_existing_file(cli["--contract"], "active-block contract")
    contract_sha256 = _validate_contract(contract_path, cli["--contract-sha256"])
    producer_path = _canonical_existing_file(abspath(@__FILE__), "witness producer source")
    producer_sha256 = _verified_input(
        producer_path,
        cli["--producer-sha256"],
        "witness producer source",
    )
    checkpoint_path = _canonical_existing_file(cli["--checkpoint"], "sparse checkpoint")
    checkpoint_sha256 = String(cli["--checkpoint-sha256"])

    output_npz = _fresh_output(cli["--output-npz"], "witness NPZ")
    output_metadata = _fresh_output(cli["--output-metadata"], "witness metadata")
    _path_key(output_npz) != _path_key(output_metadata) || error(
        "witness NPZ and metadata outputs must differ",
    )

    manifest = JSON3.read(read(manifest_path, String))
    _validate_manifest_header(manifest, manifest_sha256)
    runtime, checkpoint_update, checkpoint_metadata = _validate_sparse_checkpoint(
        checkpoint_path,
        checkpoint_sha256,
        dataset_root,
        manifest_sha256,
    )
    parts = _load_verified_train_parts(dataset_root, manifest, manifest_sha256)
    corpus_sha256 = _train_corpus_digest(parts)
    total_candidates = sum(part.candidates for part in parts)
    positions, offset, stride, sampling_sha256 = _derive_positions(
        total_candidates,
        candidate_count,
        manifest_sha256,
        checkpoint_sha256,
        contract_sha256,
        producer_sha256,
        corpus_sha256,
        sampling_domain,
    )
    selections = _map_positions(parts, positions)

    workspace = SparseDynamic3Layer.ThreeLayerWorkspace(runtime)
    rows, ids, inputs = _allocate_witness(candidate_count)
    host_batch = CanonicalTeacherPacking.allocate_host_batch(
        1;
        max_candidates=LEARNER_WIDTH,
    )
    maximum_preactivation_error = 0.0
    used_part_indices = sort!(unique(selection.part_index for selection in selections))
    for part_index in used_part_indices
        part = parts[part_index]
        dataset = _load_train_part_dataset(part)
        for selection in selections
            selection.part_index == part_index || continue
            row, action = _local_candidate(dataset.action_counts, selection.local_ordinal)
            CanonicalTeacherPacking.pack_batch!(host_batch, dataset, [row])
            maximum_preactivation_error = max(
                maximum_preactivation_error,
                _capture_candidate!(
                    rows,
                    ids,
                    inputs,
                    selection.output_index,
                    runtime,
                    workspace,
                    host_batch.inputs,
                    action,
                ),
            )
        end
    end

    members = _npz_members(rows, ids, inputs, candidate_count)
    array_receipts = [
        (;
            name,
            dtype=descriptor,
            shape=collect(shape),
            c_contiguous=true,
            sha256=_array_digest(storage),
        ) for (name, storage, shape, descriptor) in members
    ]
    temporary_npz = output_npz * ".tmp.$(getpid())"
    temporary_metadata = output_metadata * ".tmp.$(getpid())"
    ispath(temporary_npz) && error("refusing to overwrite temporary NPZ")
    ispath(temporary_metadata) && error("refusing to overwrite temporary metadata")
    moved_npz = false
    try
        _write_deterministic_npz(temporary_npz, members)
        npz_sha256 = _sha256_file(temporary_npz)
        source_parts = _source_part_metadata(parts, used_part_indices)
        environment_seeds = sort!(unique(part.environment_seed for part in source_parts))
        selection_source_sha256 = _digest_fields((
            PRODUCER_SCHEMA,
            producer_sha256,
            contract_sha256,
            manifest_sha256,
            corpus_sha256,
            checkpoint_sha256,
            checkpoint_update,
            sampling_sha256,
            join(positions, ','),
        ))
        metadata = (;
            schema=WITNESS_SCHEMA,
            producer_schema=PRODUCER_SCHEMA,
            producer_status=SOURCE_STATUS,
            split="teacher_v3_train",
            reserved_seed_free=true,
            development_validation_sealed_seeds_used=false,
            candidate_count,
            npz_sha256,
            contract_sha256,
            selection_source_sha256,
            producer_source_sha256=producer_sha256,
            dataset_manifest_sha256=manifest_sha256,
            verified_train_corpus_sha256=corpus_sha256,
            verified_train_part_count=length(parts),
            verified_train_candidate_count=total_candidates,
            source_parts,
            environment_seeds,
            sampling=(;
                domain=sampling_domain,
                algorithm="sha256_coprime_stride_without_replacement_v1",
                digest=sampling_sha256,
                offset_1_based=offset,
                stride,
                global_candidate_ordinals=positions,
            ),
            sparse_checkpoint=(;
                sha256=checkpoint_sha256,
                format=SparseDynamic3Layer.CHECKPOINT_FORMAT,
                version=SparseDynamic3Layer.CHECKPOINT_VERSION,
                update=checkpoint_update,
                variant="k256",
                total_parameters=SparseDynamic3Layer.TOTAL_PARAMETERS,
                active_counts=collect(MAX_COUNTS),
                source_sha256=String(checkpoint_metadata["source_sha256"]),
                project_sha256=String(checkpoint_metadata["project_sha256"]),
                manifest_sha256=String(checkpoint_metadata["manifest_sha256"]),
            ),
            prefix_variants=(;
                k64=collect(NAMED_PREFIXES.k64),
                k128=collect(NAMED_PREFIXES.k128),
                k256=collect(NAMED_PREFIXES.k256),
            ),
            routing=(;
                hard_router="production WTA/LSH plus exact reranking",
                training_probes=[0, 0, 0],
                lazy_decay_materialized=true,
                largest_support_selected_once=true,
                smaller_variants_use_ordered_prefixes=true,
            ),
            maximum_scaled_dot_vs_sparse_preactivation_abs_error=
                maximum_preactivation_error,
            arrays=array_receipts,
            deterministic_zip=(;
                compression="STORE",
                member_mtime_unix=Int(FIXED_ZIP_MTIME),
                npy_version="1.0",
                fortran_order=false,
            ),
        )
        _write_metadata(temporary_metadata, metadata)
        mv(temporary_npz, output_npz; force=false)
        moved_npz = true
        mv(temporary_metadata, output_metadata; force=false)
        return (;
            npz=output_npz,
            metadata=output_metadata,
            npz_sha256,
            metadata_sha256=_sha256_file(output_metadata),
            candidate_count,
            checkpoint_update,
        )
    finally
        isfile(temporary_npz) && rm(temporary_npz; force=true)
        isfile(temporary_metadata) && rm(temporary_metadata; force=true)
        if moved_npz && !isfile(output_metadata)
            # Preserve the immutable NPZ for diagnosis; its missing sidecar
            # makes it unusable by the fail-closed consumer.
            @error "NPZ committed but metadata commit failed" output_npz
        end
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    artifact = produce(ARGS)
    println(JSON3.write(artifact))
end

end # module ActiveBlockWitnessProducerV2
