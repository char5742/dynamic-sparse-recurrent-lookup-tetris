module CanonicalDatasetIO

"""
Independent, fail-closed reader for the canonical beat-first teacher-v3 data.

This module owns only the on-disk boundary.  It verifies the format-3 JSON
manifest and every JLD2 part before exposing a concrete, owning source object.
The physical 208-candidate tensors are intentionally preserved here;
`CanonicalExperimentData.width80_dataset` remains the sole authority that
creates the canonical width-80 training view.

Neither the historical `training/core.jl` module nor any legacy model,
relation, sampler, or trainer API is loaded by this reader.
"""

using JLD2
using JSON3
using SHA

using ..CanonicalExperimentData

const Data = CanonicalExperimentData

export FORMAT_VERSION,
       STORAGE_MAX_CANDIDATES,
       COMPLETE_DATASET_REQUIREMENTS,
       DatasetRequirements,
       DatasetIdentity,
       CanonicalDatasetSource,
       CanonicalSourceLoader,
       LoadedCanonicalDataset,
       load_dataset_source,
       load_canonical_dataset,
       portable_fingerprint,
       ordered_training_rows_sha256

const FORMAT_VERSION = 3
const STORAGE_MAX_CANDIDATES = 208
const _BOARD_ROWS = 24
const _BOARD_COLUMNS = 10
const _QUEUE_PIECES = 7
const _QUEUE_TOKENS = 6
const _ROW_IDENTITY_ENCODING =
    "canonical-teacher-v3-ordered-training-row-identity-u64be-v1"
const _PORTABLE_ENCODING =
    "canonical-teacher-v3-portable-dataset-fingerprint-u64be-v1"
const _SEMANTIC_PART_ENCODING =
    "canonical-teacher-v3-semantic-part-payload-le-v1"
const _FORBIDDEN_DEVELOPMENT_SEEDS = 5742:5757
const _FORBIDDEN_VALIDATION_SEEDS = 8001:8008
const _FORBIDDEN_SEALED_SEEDS = 91_001:91_032
const _EPSILON_TAGS = (
    "3fa999999999999a", # 0.05
    "3fb999999999999a", # 0.10
    "3fc999999999999a", # 0.20
)

"""Minimum manifest counts accepted by one explicit loading contract."""
struct DatasetRequirements
    minimum_training_states::Int
    minimum_validation_states::Int

    function DatasetRequirements(
        minimum_training_states::Integer,
        minimum_validation_states::Integer,
    )
        minimum_training_states >= 1 || throw(ArgumentError(
            "minimum training state count must be positive",
        ))
        minimum_validation_states >= 1 || throw(ArgumentError(
            "minimum validation state count must be positive",
        ))
        return new(
            Int(minimum_training_states),
            Int(minimum_validation_states),
        )
    end
end

const COMPLETE_DATASET_REQUIREMENTS = DatasetRequirements(100_000, 10_000)

"""
Portable provenance for one fully verified source.

`manifest_sha256` authenticates the exact manifest bytes supplied by the
caller.  `portable_sha256` and `ordered_training_rows_sha256` never encode the
dataset root, manifest path, part path, or operational paths inside manifest
or JLD2 metadata.  Moving or regenerating the same typed payload under a
different operational path does not change either portable digest.
"""
struct DatasetIdentity
    format_version::Int
    manifest_sha256::String
    ordered_training_rows_sha256::String
    portable_sha256::String
    state_count::Int
    training_state_count::Int
    validation_state_count::Int
    candidate_count::Int
    part_count::Int
end

portable_fingerprint(identity::DatasetIdentity) = identity.portable_sha256
ordered_training_rows_sha256(identity::DatasetIdentity) =
    identity.ordered_training_rows_sha256

"""
Owning, concrete source representation accepted by `CanonicalExperimentData`.

The operational absolute paths are retained for diagnostics only.  They are
not inputs to either portable identity digest.
"""
struct CanonicalDatasetSource
    state_count::Int
    candidate_width::Int
    boards::Array{UInt8,4}
    placements::Array{UInt8,5}
    queues::Array{UInt8,3}
    teacher_q::Matrix{Float32}
    teacher_rank::Matrix{Int16}
    action_counts::Vector{Int}
    selected_actions::Vector{Int}
    top1_actions::Vector{Int}
    top2_actions::Vector{Int}
    top1_top2_margin::Vector{Float32}
    terminal::BitVector
    candidate_death::BitMatrix
    candidate_death_available::BitVector
    line_clear::Matrix{Int8}
    max_height::Matrix{Int8}
    holes::Matrix{Int16}
    cavities::Matrix{Int16}
    ren::Matrix{Float32}
    back_to_back::Matrix{Float32}
    tspin::Matrix{Float32}
    rewards::Vector{Float32}
    seed_ids::Vector{Int}
    episode_ids::Vector{Int}
    split_group_ids::Vector{Int}
    predefined_split::Vector{Symbol}
    episode_steps::Vector{Int}
    scores_after::Vector{Int}
    behavior_exploratory::BitVector
    source_path::String
    manifest_path::String
    manifest_format_version::Int
    manifest_counts::Dict{String,Int}
    part_integrity_verified::Bool
    verified_part_count::Int
    identity::DatasetIdentity
end

"""Callable dependency-injection object for `Data.load_width80_dataset`."""
struct CanonicalSourceLoader
    expected_manifest_sha256::String
    expected_ordered_training_rows_sha256::Union{Nothing,String}
    requirements::DatasetRequirements
end

function CanonicalSourceLoader(
    expected_manifest_sha256::AbstractString;
    expected_ordered_training_rows_sha256::Union{Nothing,AbstractString}=
        nothing,
    requirements::DatasetRequirements=COMPLETE_DATASET_REQUIREMENTS,
)
    expected_manifest = _normalize_sha256(
        expected_manifest_sha256,
        "expected manifest SHA-256",
    )
    expected_rows = expected_ordered_training_rows_sha256 === nothing ?
        nothing : _normalize_sha256(
            expected_ordered_training_rows_sha256,
            "expected ordered-training-row SHA-256",
        )
    return CanonicalSourceLoader(expected_manifest, expected_rows, requirements)
end

function (loader::CanonicalSourceLoader)(root::AbstractString)
    return load_dataset_source(
        root,
        loader.expected_manifest_sha256;
        expected_ordered_training_rows_sha256=
            loader.expected_ordered_training_rows_sha256,
        requirements=loader.requirements,
    )
end

"""Verified source, canonical width-80 view, and immutable training row list."""
struct LoadedCanonicalDataset{D}
    source::CanonicalDatasetSource
    dataset::D
    training_rows::Vector{Int}
    identity::DatasetIdentity
end

struct _PartSpec
    relative_path::String
    sha256::String
    split::Symbol
    role::Symbol
    seed::Int
    row_count::Int
    candidate_count::Int
    byte_count::Int64
    episode_key::String
end

@noinline _fail(message::AbstractString) = throw(ArgumentError(String(message)))

@noinline function _required_property(object, name::Symbol, context::String)
    hasproperty(object, name) || _fail("$context has no `$name`")
    return getproperty(object, name)
end

function _integer_property(object, name::Symbol, context::String)
    value = _required_property(object, name, context)
    value isa Bool && _fail("$context `$name` must be an integer, not Bool")
    value isa Integer || _fail("$context `$name` must be an integer")
    try
        return Int(value)
    catch error
        error isa InexactError || rethrow()
        _fail("$context `$name` is outside native Int")
    end
end

function _string_property(object, name::Symbol, context::String)
    value = _required_property(object, name, context)
    value isa AbstractString || _fail("$context `$name` must be a string")
    return String(value)
end

function _bool_property(object, name::Symbol, context::String)
    value = _required_property(object, name, context)
    value isa Bool || _fail("$context `$name` must be Bool")
    return value
end

@inline function _is_hex_digit(byte::UInt8)
    return UInt8('0') <= byte <= UInt8('9') ||
           UInt8('a') <= byte <= UInt8('f') ||
           UInt8('A') <= byte <= UInt8('F')
end

function _normalize_sha256(value::AbstractString, context::AbstractString)
    digest = String(value)
    ncodeunits(digest) == 64 || _fail("$context must contain 64 hex digits")
    all(_is_hex_digit, codeunits(digest)) || _fail(
        "$context contains a non-hex character",
    )
    return lowercase(digest)
end

function _parse_part(item, ordinal::Int)
    context = "manifest part $ordinal"
    relative_path = _string_property(item, :relative_path, context)
    isempty(relative_path) && _fail("$context has an empty relative path")
    isabspath(relative_path) && _fail("$context path must be relative")
    any(==(".."), splitpath(normpath(relative_path))) && _fail(
        "$context path contains a parent traversal",
    )
    split = Symbol(_string_property(item, :split, context))
    split in (:train, :validation) || _fail(
        "$context split must be train or validation",
    )
    role = Symbol(_string_property(item, :role, context))
    role in (:dagger, :epsilon, :old_policy) || _fail(
        "$context has unsupported rollout role `$role`",
    )
    seed = _integer_property(item, :seed, context)
    row_count = _integer_property(item, :row_count, context)
    row_count >= 1 || _fail("$context row count must be positive")
    candidate_count = _integer_property(item, :candidate_total, context)
    row_count <= candidate_count <= Base.checked_mul(
        STORAGE_MAX_CANDIDATES,
        row_count,
    ) || _fail("$context candidate total is outside its physical capacity")
    _integer_property(item, :max_candidates, context) ==
        STORAGE_MAX_CANDIDATES || _fail(
            "$context physical width must be $STORAGE_MAX_CANDIDATES",
        )
    _bool_property(item, :preserves_candidate_order, context) || _fail(
        "$context does not preserve canonical candidate order",
    )
    _bool_property(item, :preserves_candidate_multiplicity, context) || _fail(
        "$context does not preserve canonical candidate multiplicity",
    )
    byte_count_value = _integer_property(item, :bytes, context)
    byte_count_value >= 1 || _fail("$context byte count must be positive")
    sha256 = _normalize_sha256(
        _string_property(item, :sha256, context),
        "$context SHA-256",
    )
    episode_key = _string_property(item, :episode_key, context)
    isempty(episode_key) && _fail("$context has an empty episode key")
    part = _PartSpec(
        relative_path,
        sha256,
        split,
        role,
        seed,
        row_count,
        candidate_count,
        Int64(byte_count_value),
        episode_key,
    )
    _validate_episode_contract!(part, context)
    return part
end

function _validate_episode_contract!(part::_PartSpec, context::String)
    fields = split(part.episode_key, '|'; keepempty=true)
    length(fields) == 6 || _fail("$context episode key has the wrong field count")
    fields[1] == "v3" || _fail("$context episode key is not in the v3 namespace")
    fields[2] == String(part.split) || _fail(
        "$context episode-key split differs from manifest",
    )
    fields[3] == String(part.role) || _fail(
        "$context episode-key role differs from manifest",
    )
    fields[4] == string(part.seed) || _fail(
        "$context episode-key seed differs from manifest",
    )
    epsilon_tag = lowercase(fields[5])
    ncodeunits(epsilon_tag) == 16 && all(_is_hex_digit, codeunits(epsilon_tag)) ||
        _fail("$context episode key has an invalid epsilon tag")
    student_sha = fields[6]
    if part.role === :epsilon
        epsilon_tag in _EPSILON_TAGS || _fail(
            "$context epsilon episode key is outside the frozen schedule",
        )
        isempty(student_sha) || _fail(
            "$context epsilon episode key unexpectedly names a student",
        )
    elseif part.role === :old_policy
        epsilon_tag == "0000000000000000" || _fail(
            "$context old-policy episode key has nonzero epsilon",
        )
        isempty(student_sha) || _fail(
            "$context old-policy episode key unexpectedly names a student",
        )
    else
        epsilon_tag == "0000000000000000" || _fail(
            "$context DAgger episode key has nonzero epsilon",
        )
        _normalize_sha256(student_sha, "$context DAgger student SHA-256")
    end

    seed = part.seed
    seed in _FORBIDDEN_DEVELOPMENT_SEEDS && _fail(
        "$context uses a reserved development seed",
    )
    seed in _FORBIDDEN_VALIDATION_SEEDS && _fail(
        "$context uses a reserved validation seed",
    )
    seed in _FORBIDDEN_SEALED_SEEDS && _fail(
        "$context uses a sealed seed",
    )
    allowed = if part.split === :train && part.role === :old_policy
        seed in 100_001:100_320 || seed in 105_001:105_120
    elseif part.split === :train && part.role === :epsilon
        seed in 110_001:110_200
    elseif part.split === :train && part.role === :dagger
        seed in 130_001:130_240
    elseif part.split === :validation && part.role === :old_policy
        seed in 120_001:120_024
    elseif part.split === :validation && part.role === :epsilon
        seed in 121_001:121_024
    else
        false
    end
    allowed || _fail("$context seed/role/split is outside the frozen v3 schedule")
    return nothing
end

function _validate_manifest_attestation!(manifest)
    run_metadata = _required_property(
        manifest,
        :run_metadata,
        "teacher manifest",
    )
    attestation = _required_property(
        run_metadata,
        :held_out_development_validation_sealed_seeds_used,
        "teacher manifest run metadata",
    )
    attestation === false || _fail(
        "teacher manifest does not attest held-out/reserved seed exclusion",
    )
    return nothing
end

function _parse_parts(manifest)
    raw_parts = _required_property(manifest, :parts, "teacher manifest")
    raw_parts isa AbstractVector || _fail("teacher manifest `parts` must be an array")
    isempty(raw_parts) && _fail("teacher manifest has no parts")
    parts = Vector{_PartSpec}(undef, length(raw_parts))
    episode_keys = Set{String}()
    relative_paths = Set{String}()
    train_seeds = Set{Int}()
    validation_seeds = Set{Int}()
    previous_order = nothing
    for ordinal in eachindex(raw_parts)
        part = _parse_part(raw_parts[ordinal], ordinal)
        part.episode_key in episode_keys && _fail(
            "manifest duplicates episode key $(repr(part.episode_key))",
        )
        push!(episode_keys, part.episode_key)
        path_key = Sys.iswindows() ?
            lowercase(normpath(part.relative_path)) : normpath(part.relative_path)
        path_key in relative_paths && _fail(
            "manifest aliases relative part path $(repr(part.relative_path))",
        )
        push!(relative_paths, path_key)
        push!(part.split === :train ? train_seeds : validation_seeds, part.seed)
        current_order = (String(part.split), String(part.role), part.seed)
        if previous_order !== nothing && current_order < previous_order
            _fail("manifest parts are not in canonical split/role/seed order")
        end
        previous_order = current_order
        parts[ordinal] = part
    end
    isempty(intersect(train_seeds, validation_seeds)) || _fail(
        "teacher manifest shares an environment seed across train and validation",
    )
    return parts
end

function _derived_counts(parts::Vector{_PartSpec})
    counts = Dict{String,Int}()
    candidate_total = 0
    for part in parts
        split = String(part.split)
        role = String(part.role)
        counts["states.$split"] = get(counts, "states.$split", 0) +
                                   part.row_count
        counts["states.$split.$role"] =
            get(counts, "states.$split.$role", 0) + part.row_count
        counts["episodes.$split"] = get(counts, "episodes.$split", 0) + 1
        counts["episodes.$split.$role"] =
            get(counts, "episodes.$split.$role", 0) + 1
        candidate_total = Base.checked_add(candidate_total, part.candidate_count)
    end
    counts["states.total"] = sum(part.row_count for part in parts)
    counts["episodes.total"] = length(parts)
    counts["candidates.total"] = candidate_total
    return counts
end

function _manifest_counts(manifest, expected::Dict{String,Int})
    raw_counts = _required_property(manifest, :counts, "teacher manifest")
    raw_counts isa AbstractDict || _fail("teacher manifest `counts` must be an object")
    observed = Dict{String,Int}()
    for (raw_key, raw_value) in pairs(raw_counts)
        key = String(raw_key)
        haskey(observed, key) && _fail("teacher manifest duplicates count `$key`")
        raw_value isa Bool && _fail("teacher manifest count `$key` is Bool")
        raw_value isa Integer || _fail(
            "teacher manifest count `$key` must be an integer",
        )
        value = try
            Int(raw_value)
        catch error
            error isa InexactError || rethrow()
            _fail("teacher manifest count `$key` is outside native Int")
        end
        value >= 0 || _fail("teacher manifest count `$key` is negative")
        observed[key] = value
    end
    Set(keys(observed)) == Set(keys(expected)) || _fail(
        "teacher manifest count keys do not exactly match its declared parts",
    )
    for key in keys(expected)
        observed[key] == expected[key] || _fail(
            "teacher manifest count `$key` is $(observed[key]); " *
            "declared parts require $(expected[key])",
        )
    end
    return observed
end

function _check_completeness!(
    counts::Dict{String,Int},
    requirements::DatasetRequirements,
)
    training = get(counts, "states.train", 0)
    validation = get(counts, "states.validation", 0)
    training >= requirements.minimum_training_states || _fail(
        "teacher dataset is partial: $training training states; require at least " *
        "$(requirements.minimum_training_states)",
    )
    validation >= requirements.minimum_validation_states || _fail(
        "teacher dataset is partial: $validation validation states; require at least " *
        "$(requirements.minimum_validation_states)",
    )
    return nothing
end

function _resolved_part_paths(root::String, parts::Vector{_PartSpec})
    resolved_root = realpath(root)
    paths = Vector{String}(undef, length(parts))
    resolved_keys = Set{String}()
    for ordinal in eachindex(parts)
        part = parts[ordinal]
        candidate = normpath(joinpath(resolved_root, part.relative_path))
        isfile(candidate) || _fail("manifest references missing part: $candidate")
        resolved = realpath(candidate)
        relative = relpath(resolved, resolved_root)
        components = splitpath(relative)
        (!isabspath(relative) && !isempty(components) && first(components) != "..") ||
            _fail("manifest part resolves outside the dataset root: $candidate")
        key = Sys.iswindows() ? lowercase(normpath(resolved)) : normpath(resolved)
        key in resolved_keys && _fail(
            "manifest aliases one resolved part more than once: $candidate",
        )
        push!(resolved_keys, key)
        filesize(resolved) == part.byte_count || _fail(
            "manifest byte count mismatch for part $(part.relative_path)",
        )
        actual_sha = bytes2hex(open(SHA.sha256, resolved))
        actual_sha == part.sha256 || _fail(
            "manifest SHA-256 mismatch for part $(part.relative_path)",
        )
        paths[ordinal] = resolved
    end
    return resolved_root, paths
end

function _checked_array(
    file,
    name::String,
    ::Type{T},
    shape::Tuple,
    context::String,
) where {T}
    haskey(file, name) || _fail("$context is missing `$name`")
    value = file[name]
    value isa AbstractArray || _fail("$context `$name` must be an array")
    eltype(value) === T || _fail(
        "$context `$name` has element type $(eltype(value)); expected $T",
    )
    size(value) == shape || throw(DimensionMismatch(
        "$context `$name` has shape $(size(value)); expected $shape",
    ))
    return value
end

function _check_metadata!(metadata, part::_PartSpec, context::String)
    _integer_property(metadata, :format_version, context) == FORMAT_VERSION ||
        _fail("$context format version differs from manifest")
    _string_property(metadata, :episode_key, context) == part.episode_key ||
        _fail("$context episode key differs from manifest")
    Symbol(_string_property(metadata, :split, context)) === part.split ||
        _fail("$context split differs from manifest")
    Symbol(_string_property(metadata, :role, context)) === part.role ||
        _fail("$context role differs from manifest")
    _integer_property(metadata, :seed, context) == part.seed ||
        _fail("$context seed differs from manifest")
    _integer_property(metadata, :row_count, context) == part.row_count ||
        _fail("$context row count differs from manifest")
    _integer_property(metadata, :candidate_total, context) ==
        part.candidate_count || _fail(
            "$context candidate total differs from manifest",
        )
    _integer_property(metadata, :max_candidates, context) ==
        STORAGE_MAX_CANDIDATES || _fail(
            "$context physical width must be $STORAGE_MAX_CANDIDATES",
        )
    _bool_property(metadata, :preserves_candidate_order, context) || _fail(
        "$context does not preserve candidate order",
    )
    _bool_property(metadata, :preserves_candidate_multiplicity, context) ||
        _fail("$context does not preserve candidate multiplicity")
    return nothing
end

@inline _binary(value::UInt8) = value == 0x00 || value == 0x01

function _validate_part_payload!(
    file,
    part::_PartSpec,
    path::String,
)
    count = part.row_count
    width = STORAGE_MAX_CANDIDATES
    context = "teacher part $(part.relative_path)"
    boards = _checked_array(
        file, "boards", UInt8, (_BOARD_ROWS, _BOARD_COLUMNS, 1, count), context,
    )
    placements = _checked_array(
        file,
        "placements",
        UInt8,
        (_BOARD_ROWS, _BOARD_COLUMNS, 1, width, count),
        context,
    )
    ren = _checked_array(file, "ren", Float32, (1, count), context)
    back_to_back = _checked_array(
        file, "back_to_back", Float32, (1, count), context,
    )
    tspin = _checked_array(file, "tspin", Float32, (width, count), context)
    queues = _checked_array(
        file, "queues", UInt8, (_QUEUE_PIECES, _QUEUE_TOKENS, count), context,
    )
    teacher_q = _checked_array(
        file, "teacher_q", Float32, (width, count), context,
    )
    teacher_rank = _checked_array(
        file, "teacher_rank", Int16, (width, count), context,
    )
    action_counts = _checked_array(
        file, "action_counts", Int16, (count,), context,
    )
    selected_actions = _checked_array(
        file, "selected_actions", Int16, (count,), context,
    )
    top1_actions = _checked_array(
        file, "top1_actions", Int16, (count,), context,
    )
    top2_actions = _checked_array(
        file, "top2_actions", Int16, (count,), context,
    )
    top1_top2_margin = _checked_array(
        file, "top1_top2_margin", Float32, (count,), context,
    )
    line_clear = _checked_array(
        file, "line_clear", Int8, (width, count), context,
    )
    death = _checked_array(file, "death", Bool, (width, count), context)
    max_height = _checked_array(
        file, "max_height", Int8, (width, count), context,
    )
    holes = _checked_array(file, "holes", Int16, (width, count), context)
    cavities = _checked_array(
        file, "cavities", Int16, (width, count), context,
    )
    rewards = _checked_array(file, "rewards", Float32, (count,), context)
    seed_ids = _checked_array(file, "seed_ids", Int64, (count,), context)
    episode_ids = _checked_array(file, "episode_ids", Int32, (count,), context)
    episode_steps = _checked_array(
        file, "episode_steps", Int16, (count,), context,
    )
    terminal = _checked_array(file, "terminal", Bool, (count,), context)
    scores_after = _checked_array(
        file, "scores_after", Int32, (count,), context,
    )
    behavior_exploratory = _checked_array(
        file, "behavior_exploratory", Bool, (count,), context,
    )
    haskey(file, "metadata") || _fail("$context is missing `metadata`")
    _check_metadata!(file["metadata"], part, "$context metadata")

    all(_binary, boards) || _fail("$context boards contain a non-binary value")
    all(_binary, placements) || _fail(
        "$context placements contain a non-binary value",
    )
    all(_binary, queues) || _fail("$context queues contain a non-binary value")
    all(isfinite, ren) || _fail("$context REN contains a non-finite value")
    all(isfinite, back_to_back) || _fail(
        "$context back-to-back contains a non-finite value",
    )
    all(isfinite, tspin) || _fail("$context T-spin contains a non-finite value")
    all(isfinite, rewards) || _fail("$context rewards contain a non-finite value")
    all(isfinite, top1_top2_margin) || _fail(
        "$context top-1/top-2 margins contain a non-finite value",
    )

    observed_candidate_count = 0
    @inbounds for row in 1:count
        candidates = Int(action_counts[row])
        1 <= candidates <= width || _fail(
            "$context row $row has $candidates candidates outside 1:$width",
        )
        observed_candidate_count += candidates
        selected = Int(selected_actions[row])
        1 <= selected <= candidates || _fail(
            "$context row $row selected action $selected outside 1:$candidates",
        )
        Int(seed_ids[row]) == part.seed || _fail(
            "$context row $row seed id differs from manifest seed",
        )
        Int(episode_ids[row]) == part.seed || _fail(
            "$context row $row episode id differs from manifest seed",
        )
        Int(episode_steps[row]) == row || _fail(
            "$context row $row has a noncanonical episode step",
        )
        ren_value = ren[1, row]
        ren_value >= 0.0f0 && ren_value == trunc(ren_value) || _fail(
            "$context row $row REN is not a nonnegative integer",
        )
        back_to_back[1, row] in (0.0f0, 1.0f0) || _fail(
            "$context row $row back-to-back is not binary",
        )
        for role in 1:_QUEUE_TOKENS
            ones = 0
            for piece in 1:_QUEUE_PIECES
                ones += queues[piece, role, row]
            end
            if role == 1
                ones in (0, 1) || _fail(
                    "$context row $row HOLD queue role is not zero/one-hot",
                )
            else
                ones == 1 || _fail(
                    "$context row $row NEXT queue role $role is not one-hot",
                )
            end
        end
        for candidate in 1:candidates
            isfinite(teacher_q[candidate, row]) || _fail(
                "$context row $row candidate $candidate has non-finite teacher Q",
            )
            tspin[candidate, row] in (0.0f0, 1.0f0) || _fail(
                "$context row $row candidate $candidate T-spin is not binary",
            )
            0 <= line_clear[candidate, row] <= 4 || _fail(
                "$context row $row candidate $candidate line clear is outside 0:4",
            )
            0 <= max_height[candidate, row] <= _BOARD_ROWS || _fail(
                "$context row $row candidate $candidate height is outside 0:$_BOARD_ROWS",
            )
            0 <= holes[candidate, row] <= _BOARD_ROWS * _BOARD_COLUMNS || _fail(
                "$context row $row candidate $candidate holes are out of range",
            )
            0 <= cavities[candidate, row] <= _BOARD_ROWS * _BOARD_COLUMNS ||
                _fail(
                    "$context row $row candidate $candidate cavities are out of range",
                )
        end
        if candidates < width
            all(isnan, @view teacher_q[(candidates + 1):width, row]) || _fail(
                "$context row $row teacher-Q padding is not canonical NaN",
            )
            all(iszero, @view teacher_rank[(candidates + 1):width, row]) ||
                _fail("$context row $row teacher-rank padding is not zero")
        end
        ordering = sortperm(
            @view(teacher_q[1:candidates, row]);
            rev=true,
            alg=MergeSort,
        )
        for (rank, candidate) in enumerate(ordering)
            Int(teacher_rank[candidate, row]) == rank || _fail(
                "$context row $row teacher rank differs from stable Q order",
            )
        end
        expected_top1 = ordering[1]
        expected_top2 = ordering[min(2, candidates)]
        Int(top1_actions[row]) == expected_top1 || _fail(
            "$context row $row top-1 action differs from teacher Q",
        )
        Int(top2_actions[row]) == expected_top2 || _fail(
            "$context row $row top-2 action differs from teacher Q",
        )
        expected_margin = Float32(
            teacher_q[expected_top1, row] - teacher_q[expected_top2, row],
        )
        top1_top2_margin[row] == expected_margin || _fail(
            "$context row $row top-1/top-2 margin differs from teacher Q",
        )
    end
    observed_candidate_count == part.candidate_count || _fail(
        "$context candidate total differs from manifest",
    )

    return (;
        boards,
        placements,
        queues,
        teacher_q,
        teacher_rank,
        action_counts,
        selected_actions,
        top1_actions,
        top2_actions,
        top1_top2_margin,
        terminal,
        death,
        line_clear,
        max_height,
        holes,
        cavities,
        ren,
        back_to_back,
        tspin,
        rewards,
        seed_ids,
        episode_ids,
        episode_steps,
        scores_after,
        behavior_exploratory,
    )
end

@inline function _write_u64(io::IO, value::UInt64)
    @inbounds for shift in 56:-8:0
        write(io, UInt8((value >> shift) & 0xff))
    end
    return io
end

@inline _write_i64(io::IO, value::Integer) =
    _write_u64(io, reinterpret(UInt64, Int64(value)))

function _write_string(io::IO, value::AbstractString)
    bytes = codeunits(value)
    _write_u64(io, UInt64(length(bytes)))
    write(io, bytes)
    return io
end

function _ordered_row_digest(
    predefined_split::Vector{Symbol},
    seed_ids::Vector{Int},
    episode_ids::Vector{Int},
    episode_steps::Vector{Int},
    action_counts::Vector{Int},
)
    training_count = count(==(:train), predefined_split)
    io = IOBuffer()
    _write_string(io, _ROW_IDENTITY_ENCODING)
    _write_u64(io, UInt64(training_count))
    @inbounds for row in eachindex(predefined_split)
        predefined_split[row] === :train || continue
        _write_u64(io, UInt64(row))
        _write_i64(io, seed_ids[row])
        _write_i64(io, episode_ids[row])
        _write_i64(io, episode_steps[row])
        _write_u64(io, UInt64(action_counts[row]))
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function _portable_digest(
    parts::Vector{_PartSpec},
    counts::Dict{String,Int},
    ordered_rows_sha256::String,
    semantic_part_sha256s::Vector{String},
)
    length(semantic_part_sha256s) == length(parts) || throw(DimensionMismatch(
        "semantic part digest count differs from manifest part count",
    ))
    io = IOBuffer()
    _write_string(io, _PORTABLE_ENCODING)
    _write_u64(io, UInt64(FORMAT_VERSION))
    count_keys = sort!(collect(keys(counts)))
    _write_u64(io, UInt64(length(count_keys)))
    for key in count_keys
        _write_string(io, key)
        _write_u64(io, UInt64(counts[key]))
    end
    _write_u64(io, UInt64(length(parts)))
    for (ordinal, part) in enumerate(parts)
        # Deliberately exclude root, manifest path, relative part path, exact
        # JLD2 byte hash, and all operational metadata paths.
        _write_string(io, String(part.split))
        _write_string(io, String(part.role))
        _write_i64(io, part.seed)
        _write_string(io, part.episode_key)
        _write_u64(io, UInt64(part.row_count))
        _write_u64(io, UInt64(part.candidate_count))
        write(io, hex2bytes(semantic_part_sha256s[ordinal]))
    end
    write(io, hex2bytes(ordered_rows_sha256))
    return bytes2hex(SHA.sha256(take!(io)))
end

@inline function _sha_update_u64!(context, value::UInt64)
    bytes = Vector{UInt8}(undef, 8)
    @inbounds for index in 1:8
        shift = 8 * (index - 1)
        bytes[index] = UInt8((value >> shift) & 0xff)
    end
    SHA.update!(context, bytes)
    return context
end


@inline _sha_update_i64!(context, value::Integer) =
    _sha_update_u64!(context, reinterpret(UInt64, Int64(value)))

function _sha_update_string!(context, value::AbstractString)
    bytes = Vector{UInt8}(codeunits(value))
    _sha_update_u64!(context, UInt64(length(bytes)))
    SHA.update!(context, bytes)
    return context
end

function _sha_update_array!(context, name::Symbol, value::AbstractArray)
    _sha_update_string!(context, String(name))
    _sha_update_string!(context, string(eltype(value)))
    _sha_update_u64!(context, UInt64(ndims(value)))
    for dimension in size(value)
        _sha_update_u64!(context, UInt64(dimension))
    end
    if eltype(value) === Bool
        # BitArray storage is an implementation detail; encode one semantic
        # byte per value in Julia's canonical column-major iteration order.
        bytes = Vector{UInt8}(undef, length(value))
        @inbounds for ordinal in eachindex(value)
            bytes[ordinal] = value[ordinal] ? 0x01 : 0x00
        end
        SHA.update!(context, bytes)
        return context
    end

    isbitstype(eltype(value)) || _fail(
        "semantic payload `$name` must have an isbits element type",
    )
    raw = reinterpret(UInt8, vec(value))
    if Base.ENDIAN_BOM == 0x04030201 || sizeof(eltype(value)) == 1
        SHA.update!(context, raw)
        return context
    end

    # The portable encoding is fixed little-endian even on a big-endian host.
    width = sizeof(eltype(value))
    bytes = Vector{UInt8}(raw)
    @inbounds for first_byte in 1:width:length(bytes)
        reverse!(@view bytes[first_byte:(first_byte + width - 1)])
    end
    SHA.update!(context, bytes)
    return context
end

function _semantic_part_digest(payload, part::_PartSpec)
    context = SHA.SHA2_256_CTX()
    _sha_update_string!(context, _SEMANTIC_PART_ENCODING)
    _sha_update_string!(context, String(part.split))
    _sha_update_string!(context, String(part.role))
    _sha_update_i64!(context, part.seed)
    _sha_update_string!(context, part.episode_key)
    _sha_update_u64!(context, UInt64(part.row_count))
    _sha_update_u64!(context, UInt64(part.candidate_count))
    fields = (
        :boards,
        :placements,
        :queues,
        :teacher_q,
        :teacher_rank,
        :action_counts,
        :selected_actions,
        :top1_actions,
        :top2_actions,
        :top1_top2_margin,
        :terminal,
        :death,
        :line_clear,
        :max_height,
        :holes,
        :cavities,
        :ren,
        :back_to_back,
        :tspin,
        :rewards,
        :seed_ids,
        :episode_ids,
        :episode_steps,
        :scores_after,
        :behavior_exploratory,
    )
    for name in fields
        _sha_update_array!(context, name, getproperty(payload, name))
    end
    return bytes2hex(SHA.digest!(context))
end

function _allocate_source_arrays(states::Int)
    width = STORAGE_MAX_CANDIDATES
    return (;
        boards=zeros(UInt8, _BOARD_ROWS, _BOARD_COLUMNS, 1, states),
        placements=zeros(
            UInt8,
            _BOARD_ROWS,
            _BOARD_COLUMNS,
            1,
            width,
            states,
        ),
        queues=zeros(UInt8, _QUEUE_PIECES, _QUEUE_TOKENS, states),
        teacher_q=fill(Float32(NaN), width, states),
        teacher_rank=zeros(Int16, width, states),
        action_counts=zeros(Int, states),
        selected_actions=zeros(Int, states),
        top1_actions=zeros(Int, states),
        top2_actions=zeros(Int, states),
        top1_top2_margin=zeros(Float32, states),
        terminal=falses(states),
        candidate_death=falses(width, states),
        candidate_death_available=falses(states),
        line_clear=zeros(Int8, width, states),
        max_height=zeros(Int8, width, states),
        holes=zeros(Int16, width, states),
        cavities=zeros(Int16, width, states),
        ren=zeros(Float32, 1, states),
        back_to_back=zeros(Float32, 1, states),
        tspin=zeros(Float32, width, states),
        rewards=zeros(Float32, states),
        seed_ids=zeros(Int, states),
        episode_ids=zeros(Int, states),
        predefined_split=fill(:unspecified, states),
        episode_steps=zeros(Int, states),
        scores_after=zeros(Int, states),
        behavior_exploratory=falses(states),
    )
end

function _copy_part!(storage, payload, rows::UnitRange{Int}, split::Symbol)
    storage.boards[:, :, :, rows] .= payload.boards
    storage.placements[:, :, :, :, rows] .= payload.placements
    storage.queues[:, :, rows] .= payload.queues
    storage.teacher_q[:, rows] .= payload.teacher_q
    storage.teacher_rank[:, rows] .= payload.teacher_rank
    storage.action_counts[rows] .= Int.(payload.action_counts)
    storage.selected_actions[rows] .= Int.(payload.selected_actions)
    storage.top1_actions[rows] .= Int.(payload.top1_actions)
    storage.top2_actions[rows] .= Int.(payload.top2_actions)
    storage.top1_top2_margin[rows] .= payload.top1_top2_margin
    storage.terminal[rows] .= payload.terminal
    storage.candidate_death[:, rows] .= payload.death
    storage.candidate_death_available[rows] .= true
    storage.line_clear[:, rows] .= payload.line_clear
    storage.max_height[:, rows] .= payload.max_height
    storage.holes[:, rows] .= payload.holes
    storage.cavities[:, rows] .= payload.cavities
    storage.ren[:, rows] .= payload.ren
    storage.back_to_back[:, rows] .= payload.back_to_back
    storage.tspin[:, rows] .= payload.tspin
    storage.rewards[rows] .= payload.rewards
    storage.seed_ids[rows] .= Int.(payload.seed_ids)
    storage.episode_ids[rows] .= Int.(payload.episode_ids)
    storage.predefined_split[rows] .= split
    storage.episode_steps[rows] .= Int.(payload.episode_steps)
    storage.scores_after[rows] .= Int.(payload.scores_after)
    storage.behavior_exploratory[rows] .= payload.behavior_exploratory
    return nothing
end

"""
Load one complete teacher-v3 directory after checking an explicit manifest SHA.

There is deliberately no environment-variable or `allow_partial` escape hatch.
Focused fixtures may pass a smaller, typed `DatasetRequirements` value; the
production default rejects anything below 100,000 train and 10,000 validation
states.
"""
function load_dataset_source(
    root::AbstractString,
    expected_manifest_sha256::AbstractString;
    expected_ordered_training_rows_sha256::Union{Nothing,AbstractString}=
        nothing,
    requirements::DatasetRequirements=COMPLETE_DATASET_REQUIREMENTS,
)
    isdir(root) || _fail("teacher dataset root does not exist: $(abspath(root))")
    absolute_root = abspath(root)
    manifest_path = joinpath(absolute_root, "manifest.json")
    isfile(manifest_path) || _fail(
        "teacher dataset manifest does not exist: $manifest_path",
    )
    manifest_bytes = read(manifest_path)
    manifest_sha256 = bytes2hex(SHA.sha256(manifest_bytes))
    expected_manifest = _normalize_sha256(
        expected_manifest_sha256,
        "expected manifest SHA-256",
    )
    manifest_sha256 == expected_manifest || _fail(
        "teacher manifest changed: expected $expected_manifest, got $manifest_sha256",
    )
    manifest = try
        JSON3.read(String(manifest_bytes))
    catch error
        _fail("teacher manifest is not valid UTF-8 JSON: $(sprint(showerror, error))")
    end
    _integer_property(manifest, :format_version, "teacher manifest") ==
        FORMAT_VERSION || _fail(
            "teacher manifest format must be $FORMAT_VERSION",
        )
    _validate_manifest_attestation!(manifest)
    parts = _parse_parts(manifest)
    derived_counts = _derived_counts(parts)
    counts = _manifest_counts(manifest, derived_counts)
    _check_completeness!(counts, requirements)
    resolved_root, part_paths = _resolved_part_paths(absolute_root, parts)

    states = counts["states.total"]
    storage = _allocate_source_arrays(states)
    semantic_part_sha256s = Vector{String}(undef, length(parts))
    cursor = 1
    for ordinal in eachindex(parts)
        part = parts[ordinal]
        rows = cursor:(cursor + part.row_count - 1)
        # Decode the same immutable bytes whose size and SHA are checked here;
        # a path replacement between verification and JLD2 reads cannot enter
        # the published source.
        part_bytes = read(part_paths[ordinal])
        length(part_bytes) == part.byte_count || _fail(
            "part changed before decode: $(part.relative_path)",
        )
        bytes2hex(SHA.sha256(part_bytes)) == part.sha256 || _fail(
            "part SHA-256 changed before decode: $(part.relative_path)",
        )
        JLD2.jldopen(part_bytes, "r") do file
            payload = _validate_part_payload!(file, part, part_paths[ordinal])
            semantic_part_sha256s[ordinal] =
                _semantic_part_digest(payload, part)
            _copy_part!(storage, payload, rows, part.split)
        end
        cursor += part.row_count
    end
    cursor == states + 1 || _fail(
        "teacher dataset materialization ended at the wrong row",
    )
    count(==(:train), storage.predefined_split) == counts["states.train"] ||
        _fail("materialized training-row count differs from manifest")
    count(==(:validation), storage.predefined_split) ==
        counts["states.validation"] || _fail(
            "materialized validation-row count differs from manifest",
        )
    sum(storage.action_counts) == counts["candidates.total"] || _fail(
        "materialized candidate count differs from manifest",
    )

    ordered_rows_sha256 = _ordered_row_digest(
        storage.predefined_split,
        storage.seed_ids,
        storage.episode_ids,
        storage.episode_steps,
        storage.action_counts,
    )
    if expected_ordered_training_rows_sha256 !== nothing
        expected_rows = _normalize_sha256(
            expected_ordered_training_rows_sha256,
            "expected ordered-training-row SHA-256",
        )
        ordered_rows_sha256 == expected_rows || _fail(
            "ordered training rows changed: expected $expected_rows, " *
            "got $ordered_rows_sha256",
        )
    end
    portable_sha256 = _portable_digest(
        parts,
        counts,
        ordered_rows_sha256,
        semantic_part_sha256s,
    )
    identity = DatasetIdentity(
        FORMAT_VERSION,
        manifest_sha256,
        ordered_rows_sha256,
        portable_sha256,
        states,
        counts["states.train"],
        counts["states.validation"],
        counts["candidates.total"],
        length(parts),
    )
    split_group_ids = storage.seed_ids
    return CanonicalDatasetSource(
        states,
        STORAGE_MAX_CANDIDATES,
        storage.boards,
        storage.placements,
        storage.queues,
        storage.teacher_q,
        storage.teacher_rank,
        storage.action_counts,
        storage.selected_actions,
        storage.top1_actions,
        storage.top2_actions,
        storage.top1_top2_margin,
        storage.terminal,
        storage.candidate_death,
        storage.candidate_death_available,
        storage.line_clear,
        storage.max_height,
        storage.holes,
        storage.cavities,
        storage.ren,
        storage.back_to_back,
        storage.tspin,
        storage.rewards,
        storage.seed_ids,
        storage.episode_ids,
        split_group_ids,
        storage.predefined_split,
        storage.episode_steps,
        storage.scores_after,
        storage.behavior_exploratory,
        resolved_root,
        realpath(manifest_path),
        FORMAT_VERSION,
        counts,
        true,
        length(parts),
        identity,
    )
end

function load_canonical_dataset(
    loader::CanonicalSourceLoader,
    root::AbstractString,
)
    source = loader(root)
    dataset = Data.width80_dataset(source)
    rows = Data.training_rows(source)
    length(rows) == source.identity.training_state_count || _fail(
        "canonical training rows differ from the verified source identity",
    )
    return LoadedCanonicalDataset(source, dataset, rows, source.identity)
end

function load_canonical_dataset(
    root::AbstractString,
    expected_manifest_sha256::AbstractString;
    expected_ordered_training_rows_sha256::Union{Nothing,AbstractString}=
        nothing,
    requirements::DatasetRequirements=COMPLETE_DATASET_REQUIREMENTS,
)
    loader = CanonicalSourceLoader(
        expected_manifest_sha256;
        expected_ordered_training_rows_sha256,
        requirements,
    )
    return load_canonical_dataset(loader, root)
end

end # module
