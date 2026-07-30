# Load this file into `OfficialNeuronTeacherMetadataProduction`.

const EXPECTED_MODELDB_COMMIT =
    "50a4aab3ce5c295ad16a134c5d9261b7cc3fbe58"
const EXPECTED_MODELDB_GIT_TREE =
    "ffdabdeaa0b6f0d358d5d56ac0f0d046e14f534a"
const EXPECTED_GENERATOR_SHA256 =
    "5de5096f0d292e841e9115b72cfd378433dfdd1d884901e6e04b765c1f092f71"

struct OfficialTeacherMetadataFinal
    manifest_path::String
    manifest_sha256::String
    source_dataset_sha256::String
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
    segment_region::Vector{UInt8}
end

function _canonical_json(value; drop_contract_digest::Bool=false)
    if value isa AbstractDict ||
       !(value isa AbstractArray) &&
       !(value isa AbstractString) &&
       propertynames(value) != ()
        names = sort!(String.(collect(propertynames(value))))
        if drop_contract_digest
            filter!(
                !=("teacher_contract_sha256"),
                names,
            )
        end
        body = String[]
        for text in names
            name = Symbol(text)
            child = _value(value, name, nothing)
            push!(
                body,
                JSON3.write(text) * ":" *
                _canonical_json(child),
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

function _sha256_text(value::AbstractString)
    return bytes2hex(SHA.sha256(codeunits(value)))
end

function _same_canonical(left, right)
    return _canonical_json(left) == _canonical_json(right)
end

function _inside_root(path::AbstractString, root::AbstractString)
    absolute = lowercase(abspath(path))
    prefix = lowercase(abspath(root)) * string(Base.Filesystem.path_separator)
    return startswith(absolute, prefix)
end

function _dataset_digest(
    manifest_sha256::AbstractString,
    records,
)
    context = SHA.SHA2_256_CTX()
    SHA.update!(context, codeunits(manifest_sha256))
    for record in records
        SHA.update!(context, codeunits(String(_required(record, :path))))
        SHA.update!(context, codeunits(
            lowercase(String(_required(record, :sha256))),
        ))
        SHA.update!(context, codeunits(string(Int(
            _required(record, :bytes),
        ))))
    end
    return bytes2hex(SHA.digest!(context))
end

function load_official_teacher_metadata(
    path::AbstractString;
    verify_shards::Bool=true,
)
    verify_shards || error(
        "production cannot skip official teacher shard hashing",
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
    declared_contract =
        _required_sha256(manifest, :teacher_contract_sha256)
    computed_contract = _sha256_text(
        _canonical_json(contract; drop_contract_digest=true),
    )
    computed_contract == declared_contract ||
        error("teacher-contract canonical SHA-256 mismatch")
    _required_sha256(contract, :teacher_contract_sha256) ==
        declared_contract ||
        error("embedded teacher-contract SHA-256 differs")
    _same_canonical(
        _required(contract, :config),
        _required(manifest, :config),
    ) || error("teacher contract/config differs from manifest")
    _same_canonical(
        _required(contract, :source_hashes),
        _required(manifest, :source_hashes),
    ) || error("teacher contract/source hashes differ from manifest")

    hashes = _required(manifest, :source_hashes)
    String(_required(hashes, :modeldb_repository_url)) ==
        "https://github.com/ModelDBRepository/139653.git" ||
        error("official teacher is not ModelDB accession 139653")
    String(_required(hashes, :modeldb_git_commit)) ==
        EXPECTED_MODELDB_COMMIT ||
        error("ModelDB commit differs from the pinned production commit")
    String(_required(hashes, :modeldb_git_tree)) ==
        EXPECTED_MODELDB_GIT_TREE ||
        error("ModelDB git tree differs from the pinned production tree")
    String(_required(hashes, :modeldb_tracked_status)) == "" ||
        error("ModelDB tracked source was dirty during generation")
    mechanism_library =
        _required_sha256(hashes, :mechanism_library_sha256)
    mechanism_sources =
        _required_sha256(hashes, :mechanism_sources_sha256)
    morphology = _required_sha256(hashes, :morphology_sha256)
    modeldb_tree = _required_sha256(hashes, :modeldb_tree_sha256)
    generator = _required_sha256(hashes, :generator_source_sha256)
    generator == EXPECTED_GENERATOR_SHA256 ||
        error("official teacher generator is not the pinned source")
    _file_sha256(joinpath(@__DIR__, "neuron_hay_teacher.py")) ==
        EXPECTED_GENERATOR_SHA256 ||
        error("local official teacher generator changed after pinning")

    config = _required(manifest, :config)
    String(_required(config, :preset)) == "production" ||
        error("production requires the official production teacher preset")
    train_trials = Int(_required(config, :train_trials))
    validation_trials = Int(_required(config, :validation_trials))
    test_trials = Int(_required(config, :test_trials))
    (train_trials, validation_trials, test_trials) ==
        (49_000, 1_000, 2_000) ||
        error("official production split must be 49k/1k/2k")
    Int(_required(config, :duration_ms)) == 10_000 ||
        error("official production trials must be 10 seconds")
    Int(_required(config, :contacts_per_axon)) == 20 ||
        error("official production teacher needs 20 contacts per axon")
    Float64(_required(config, :dt_ms)) == 0.025 ||
        error("official detailed teacher must use dt=0.025 ms")
    Float64(_required(config, :sample_dt_ms)) == 1.0 ||
        error("official teacher must publish 1 ms targets")
    completed_trials = Int(_required(manifest, :completed_trials))
    completed_trials == 52_000 ||
        error("official production teacher is not complete")
    total_segments = Int(_required(manifest, :total_segments))
    total_segments == 642 ||
        error("official Hay teacher must expose 642 segments")

    array_contract = _required(manifest, :array_contract)
    for name in (
        :target_voltage,
        :target_spike,
        :target_nmda,
        :target_compartment_voltage,
        :target_compartment_nmda,
        :target_dendritic_cai,
        :target_dendritic_ica,
        :target_ca_event,
    )
        _required(array_contract, name)
    end
    nmda_contract = _required(array_contract, :target_nmda)
    String(_required(nmda_contract, :sign)) ==
        "outward_positive" ||
        error("official NMDA sign convention differs")

    segments = _required(manifest, :segments)
    length(segments) == total_segments ||
        error("official segment catalog length differs")
    segment_region = UInt8[
        UInt8(Int(_required(segment, :region_code)) + 1)
        for segment in segments
    ]
    all(region -> 1 <= region <= 4, segment_region) ||
        error("official segment region code is invalid")

    records = collect(_required(manifest, :shards))
    isempty(records) && error("official teacher has no shards")
    root = dirname(manifest_path)
    seen = Set{String}()
    split_sum = zeros(Int, 3)
    expected_first = 1
    for record in records
        relative = String(_required(record, :path))
        relative in seen &&
            error("official teacher repeats shard path $relative")
        push!(seen, relative)
        shard_path = abspath(joinpath(root, relative))
        _inside_root(shard_path, root) ||
            error("official shard escapes dataset root: $relative")
        isfile(shard_path) ||
            error("official teacher shard is absent: $shard_path")
        Int(_required(record, :global_first)) == expected_first ||
            error("official shard ranges are not contiguous")
        global_last = Int(_required(record, :global_last))
        global_last >= expected_first ||
            error("official shard range is empty")
        expected_first = global_last + 1
        filesize(shard_path) == Int(_required(record, :bytes)) ||
            error("official shard byte count differs: $relative")
        _file_sha256(shard_path) ==
            _required_sha256(record, :sha256) ||
            error("official shard hash mismatch: $relative")
        _required_sha256(record, :teacher_contract_sha256) ==
            declared_contract ||
            error("official shard uses another teacher contract")
        counts = _required(record, :split_counts)
        split_sum[1] += Int(_required(counts, :train))
        split_sum[2] += Int(_required(counts, :validation))
        split_sum[3] += Int(_required(counts, :test))
    end
    expected_first == completed_trials + 1 ||
        error("official shard ranges do not cover every trial")
    Tuple(split_sum) ==
        (train_trials, validation_trials, test_trials) ||
        error("official shard split counts differ")

    manifest_sha256 = _file_sha256(manifest_path)
    return OfficialTeacherMetadataFinal(
        manifest_path,
        manifest_sha256,
        _dataset_digest(manifest_sha256, records),
        declared_contract,
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
        0.025,
        1.0,
        segment_region,
    )
end

