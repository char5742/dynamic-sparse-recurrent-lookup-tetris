module OfficialNeuronTeacherMetadataProduction

using JSON3
using SHA

export MODEL_FAMILY,
    OFFICIAL_TEACHER_SCHEMA,
    OfficialTeacherMetadata,
    load_official_teacher_metadata

const MODEL_FAMILY = "HD-SWSNN-TwinProp"
const OFFICIAL_TEACHER_SCHEMA =
    "hd_swsnn_twinprop.neuron_teacher.v1"

struct OfficialTeacherMetadata
    manifest_path::String
    manifest_sha256::String
    teacher_contract_sha256::String
    mechanism_library_sha256::String
    mechanism_sources_sha256::String
    morphology_sha256::String
    modeldb_tree_sha256::String
    generator_source_sha256::String
    total_segments::Int
    completed_trials::Int
    train_trials::Int
    validation_trials::Int
    test_trials::Int
    dt_ms::Float64
    sample_dt_ms::Float64
end

@inline function _value(object, name::Symbol, default=nothing)
    if object isa AbstractDict
        return get(object, name, get(object, String(name), default))
    elseif hasproperty(object, name)
        return getproperty(object, name)
    end
    return default
end

function _required(object, name::Symbol)
    value = _value(object, name, nothing)
    value === nothing &&
        error("official NEURON manifest lacks `$name`")
    return value
end

function _required_sha256(object, name::Symbol)
    value = lowercase(String(_required(object, name)))
    occursin(r"^[0-9a-f]{64}$", value) ||
        error("official NEURON `$name` is not SHA-256")
    return value
end

function _file_sha256(path::AbstractString)
    isfile(path) || error("file is absent: $path")
    return bytes2hex(SHA.sha256(read(path)))
end

function _manifest_path(path::AbstractString)
    source = abspath(path)
    return isdir(source) ? joinpath(source, "manifest.json") : source
end

"""
Load and verify metadata produced by `neuron_hay_teacher.py`.

This accepts only a completed, unmodified ModelDB/NEURON teacher.  With
`verify_shards=true` every listed NPZ is hashed before the metadata is
published to the production chain.
"""
function load_official_teacher_metadata(
    path::AbstractString;
    verify_shards::Bool=true,
)
    manifest_path = _manifest_path(path)
    isfile(manifest_path) ||
        error("official NEURON manifest is absent: $manifest_path")
    manifest = JSON3.read(read(manifest_path, String))

    String(_required(manifest, :schema_name)) ==
        OFFICIAL_TEACHER_SCHEMA ||
        error("unsupported official NEURON teacher schema")
    String(_required(manifest, :model_name)) == MODEL_FAMILY ||
        error("official teacher model family differs")
    String(_required(manifest, :stage)) ==
        "official_hay_neuron_teacher" ||
        error("manifest is not the official Hay/NEURON stage")
    String(_required(manifest, :completion_state)) == "complete" ||
        error("official NEURON teacher generation is incomplete")
    _required(manifest, :modeldb_source_modified_by_generator) === false ||
        error("generator reports a modified ModelDB source")

    contract = _required(manifest, :teacher_contract)
    contract_hash =
        _required_sha256(manifest, :teacher_contract_sha256)
    _required_sha256(contract, :teacher_contract_sha256) ==
        contract_hash ||
        error("teacher-contract digest differs between manifest fields")
    String(_required(contract, :schema_name)) ==
        OFFICIAL_TEACHER_SCHEMA ||
        error("teacher contract schema differs")
    String(_required(contract, :model_name)) == MODEL_FAMILY ||
        error("teacher contract model family differs")

    hashes = _required(manifest, :source_hashes)
    String(_required(hashes, :modeldb_repository_url)) ==
        "https://github.com/ModelDBRepository/139653.git" ||
        error("official teacher is not ModelDB accession 139653")
    String(_required(hashes, :modeldb_tracked_status)) == "" ||
        error("ModelDB tracked source was dirty during teacher generation")
    mechanism_library =
        _required_sha256(hashes, :mechanism_library_sha256)
    mechanism_sources =
        _required_sha256(hashes, :mechanism_sources_sha256)
    morphology = _required_sha256(hashes, :morphology_sha256)
    modeldb_tree = _required_sha256(hashes, :modeldb_tree_sha256)
    generator = _required_sha256(hashes, :generator_source_sha256)

    config = _required(manifest, :config)
    train_trials = Int(_required(config, :train_trials))
    validation_trials = Int(_required(config, :validation_trials))
    test_trials = Int(_required(config, :test_trials))
    train_trials > 0 || error("official teacher train split is empty")
    validation_trials > 0 ||
        error("official teacher validation split is empty")
    test_trials > 0 || error("official teacher held-out split is empty")
    completed_trials = Int(_required(manifest, :completed_trials))
    completed_trials ==
        train_trials + validation_trials + test_trials ||
        error("official teacher split counts do not cover all trials")
    total_segments = Int(_required(manifest, :total_segments))
    total_segments > 0 || error("official teacher has no segments")

    shards = _required(manifest, :shards)
    isempty(shards) && error("official teacher has no data shards")
    split_sum = zeros(Int, 3)
    root = dirname(manifest_path)
    for record in shards
        relative = String(_required(record, :path))
        declared = _required_sha256(record, :sha256)
        shard_path = abspath(joinpath(root, relative))
        isfile(shard_path) ||
            error("official teacher shard is absent: $shard_path")
        if verify_shards
            _file_sha256(shard_path) == declared ||
                error("official teacher shard hash mismatch: $relative")
        end
        counts = _required(record, :split_counts)
        split_sum[1] += Int(_required(counts, :train))
        split_sum[2] += Int(_required(counts, :validation))
        split_sum[3] += Int(_required(counts, :test))
    end
    Tuple(split_sum) ==
        (train_trials, validation_trials, test_trials) ||
        error("official teacher shard split counts differ from config")

    dt_ms = Float64(_required(config, :dt_ms))
    sample_dt_ms = Float64(_required(config, :sample_dt_ms))
    dt_ms == 0.025 ||
        error("official detailed teacher must use dt=0.025 ms")
    sample_dt_ms == 1.0 ||
        error("official teacher must publish 1 ms targets")

    return OfficialTeacherMetadata(
        manifest_path,
        _file_sha256(manifest_path),
        contract_hash,
        mechanism_library,
        mechanism_sources,
        morphology,
        modeldb_tree,
        generator,
        total_segments,
        completed_trials,
        train_trials,
        validation_trials,
        test_trials,
        dt_ms,
        sample_dt_ms,
    )
end

end # module OfficialNeuronTeacherMetadataProduction
