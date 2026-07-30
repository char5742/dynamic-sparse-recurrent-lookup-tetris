module PaperELMTwinOfficialV2SealedRelease

using JLD2
using JSON3
using NPZ
using SHA

if !isdefined(Main, :PAPER_ELM_OFFICIAL_V2_PROFILED_CANONICAL)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "LoadPaperELMTwinOfficialV2ProfiledCanonical.jl",
        ),
    )
end
const Twin = Main.PAPER_ELM_OFFICIAL_V2_PROFILED_CANONICAL
if !isdefined(Main, :OfficialTeacherContract)
    Base.include(
        Main,
        joinpath(@__DIR__, "OfficialTeacherContract.jl"),
    )
end
const Contract = Main.OfficialTeacherContract

export SealedOfficialELMReleaseAttestation,
    SealedOfficialELMRelease,
    SEALED_RELEASE_SCHEMA,
    SEALED_RELEASE_ARTIFACT_KIND,
    MINIMUM_SPIKE_AUROC,
    MAXIMUM_VOLTAGE_RMSE_MV,
    MAXIMUM_REGIONAL_NMDA_NORMALIZED_RMSE,
    canonical_sha256,
    attest_sealed_official_elm_release,
    verify_sealed_official_elm_release,
    save_sealed_official_elm_release,
    load_checked_sealed_official_elm_release,
    load_verified_sealed_official_elm_release

const SEALED_RELEASE_SCHEMA =
    "hd_swsnn.paper_elm_v2.sealed_release.final.v1"
const SEALED_RELEASE_ARTIFACT_KIND =
    "SealedOfficialELMRelease"
const SEALED_RELEASE_FORMAT_VERSION = 1
const CANONICAL_ENCODING =
    "sha256-tagged-column-major-little-endian-v1"
const EVALUATOR_ID =
    "official-final-v2-signed1278-state-carry-exact-auroc-v1"

# Immutable release gates.
const MINIMUM_SPIKE_AUROC = 0.985
const MAXIMUM_VOLTAGE_RMSE_MV = 1.0
const MAXIMUM_REGIONAL_NMDA_NORMALIZED_RMSE = 1.0

const INPUT_DIM = 1_278
const DENDRITIC_LOCATIONS = 639
const NMDA_REGIONS = 4
const EVALUATOR_CHUNK_STEPS = 100
const AUROC_RUN_RECORDS = 1_000_000

const PAPER_TRAIN_POOL_TRIALS = 50_000
const PAPER_HELDOUT_TRIALS = 2_000
const PAPER_DURATION_MS = 10_000.0

const EXPECTED_SCHEMA =
    "hd_swsnn_twinprop.neuron_teacher.final.v2"
const EXPECTED_MODEL =
    "HD-SWSNN-TwinProp"
const EXPECTED_MODELDB_URL =
    "https://github.com/ModelDBRepository/139653.git"
const EXPECTED_MODELDB_COMMIT =
    "50a4aab3ce5c295ad16a134c5d9261b7cc3fbe58"
const EXPECTED_MODELDB_TREE =
    "ffdabdeaa0b6f0d358d5d56ac0f0d046e14f534a"
const EXPECTED_GENERATOR_SHA256 =
    "5de5096f0d292e841e9115b72cfd378433dfdd1d884901e6e04b765c1f092f71"
const EXPECTED_FINAL_GENERATOR_SHA256 =
    "0d0fbfaeb326b8a6a91822c5c80717b84490cb2db8bd6e1dca253a01e8ae34ba"

const EXPECTED_DEV1500_MANIFEST_SHA256 =
    "5c0efd11a7c807235bd27601769e47447114616c32f135b7687513251de9e968"
const KNOWN_DEVELOPMENT_CONTRACTS = Set([
    # current 1,500 ms, 40 train + 8 held-out source
    "4ee32b8070c361084e5334f1d131e99680e2c53f1ac9234b6ea4810f78d5b320",
])

const _NUMERIC_KEYS = [
    "sample_indices",
    "split_code",
    "target_voltage",
    "target_spike",
    "target_nmda",
    "axon_kind",
    "contact_trial_offset",
    "contact_axon",
    "contact_segment",
    "contact_kind",
    "contact_strength",
    "contact_location_slot",
    "event_trial_offset",
    "event_axon",
    "event_time_bin",
    "event_count",
]

const _MODULE_SOURCE = @__FILE__
const _FINAL_BASE_SOURCE =
    joinpath(@__DIR__, "PaperELMTwinOfficialV2FinalBase.jl")
const _FINAL_DIFFERENTIABLE_SOURCE =
    joinpath(@__DIR__, "PaperELMTwinOfficialV2FinalDifferentiable.jl")
const _ACTIVATION_PROFILE_SOURCE =
    joinpath(@__DIR__, "PaperELMTwinOfficialV2ActivationProfiles.jl")
const _ACTIVATION_HOTFIX_SOURCE =
    joinpath(@__DIR__, "PaperELMTwinOfficialV2ActivationProfilesHotfixV2.jl")
const _PROFILED_LOADER_SOURCE =
    joinpath(@__DIR__, "LoadPaperELMTwinOfficialV2ProfiledCanonical.jl")
const _FINAL_LOADER_SOURCE =
    joinpath(@__DIR__, "LoadPaperELMTwinOfficialV2FinalCanonical.jl")
const _MODEL_SOURCE =
    joinpath(@__DIR__, "PaperELMTwinOfficialV2Final.jl")
const _CORE_SOURCE =
    joinpath(@__DIR__, "PaperELMTwinOfficialV2.jl")
const _CONTRACT_SOURCE =
    joinpath(@__DIR__, "OfficialTeacherContract.jl")
const _GENERATOR_SOURCE =
    joinpath(@__DIR__, "neuron_hay_teacher.py")
const _FINAL_GENERATOR_SOURCE =
    joinpath(@__DIR__, "neuron_hay_teacher_final.py")

struct SealedOfficialELMReleaseAttestation{P}
    payload::P
    attestation_sha256::String
end

struct SealedOfficialELMRelease{F,A}
    frozen::F
    attestation::A
end

struct _ShardRecord
    path::String
    sha256::String
    bytes::Int64
    global_first::Int
    global_last::Int
    samples::Int
end

struct _SealedDataset
    manifest_path::String
    root::String
    manifest_sha256::String
    teacher_contract_sha256::String
    source_dataset_sha256::String
    source_hashes_sha256::String
    records::Vector{_ShardRecord}
    validation_ids::Vector{Int32}
    fit_ids::Vector{Int32}
    heldout_ids::Vector{Int32}
    duration_ms::Float64
    sample_dt_ms::Float64
    train_trials::Int
    heldout_trials::Int
    paper_scale_claim::Bool
    connectivity_acknowledged::Bool
    paper_scale_uncertainty::String
end

# -- canonical hashing ------------------------------------------------------

function _text!(context, value)
    bytes = codeunits(String(value))
    SHA.update!(context, codeunits(string(length(bytes))))
    SHA.update!(context, UInt8[0x3a])
    SHA.update!(context, bytes)
end

function _canon!(context, value)
    if value === nothing
        _text!(context, "nothing")
    elseif value isa Bool
        _text!(context, value ? "bool:1" : "bool:0")
    elseif value isa Integer
        _text!(context, "int:" * string(typeof(value)) * ":" * string(value))
    elseif value isa AbstractFloat
        isfinite(value) || error("non-finite canonical payload value")
        _text!(
            context,
            "float:" * string(typeof(value)) * ":" * bitstring(value),
        )
    elseif value isa AbstractString
        _text!(context, "string")
        _text!(context, value)
    elseif value isa Symbol
        _text!(context, "symbol")
        _text!(context, String(value))
    elseif value isa JSON3.Object
        names = sort!(collect(propertynames(value)); by=String)
        _text!(context, "jsonobject:" * string(length(names)))
        for name in names
            _text!(context, String(name))
            _canon!(context, getproperty(value, name))
        end
    elseif value isa NamedTuple
        _text!(context, "namedtuple:" * string(length(value)))
        for name in keys(value)
            _text!(context, String(name))
            _canon!(context, getproperty(value, name))
        end
    elseif value isa Tuple
        _text!(context, "tuple:" * string(length(value)))
        for child in value
            _canon!(context, child)
        end
    elseif value isa AbstractArray
        _text!(context, "array:" * string(eltype(value)))
        _canon!(context, Tuple(size(value)))
        if isbitstype(eltype(value))
            ENDIAN_BOM == 0x04030201 ||
                error("canonical encoder requires a little-endian host")
            bytes = reinterpret(UInt8, vec(Array(value)))
            _text!(context, "bytes:" * string(length(bytes)))
            SHA.update!(context, bytes)
        else
            for child in value
                _canon!(context, child)
            end
        end
    elseif isstructtype(typeof(value))
        _text!(
            context,
            "struct:" * string(parentmodule(typeof(value))) * "." *
            string(nameof(typeof(value))),
        )
        for name in fieldnames(typeof(value))
            _text!(context, String(name))
            _canon!(context, getfield(value, name))
        end
    else
        error("unsupported canonical payload type $(typeof(value))")
    end
    return context
end

function canonical_sha256(value)
    context = SHA.SHA2_256_CTX()
    _canon!(context, value)
    return bytes2hex(SHA.digest!(context))
end

_file_sha256(path) = bytes2hex(SHA.sha256(read(path)))

function _artifact_payload_sha256(frozen)
    return canonical_sha256((;
        model=frozen.model,
        parameters=frozen.parameters,
        normalizer=frozen.normalizer,
        metadata=frozen.metadata,
        parameter_sha256=frozen.parameter_sha256,
        artifact_sha256=frozen.artifact_sha256,
    ))
end

# -- manifest/shard sealing -------------------------------------------------

_required(object, name::Symbol) =
    hasproperty(object, name) ?
    getproperty(object, name) :
    error("manifest lacks required field `$name`")

function _required_sha(object, name::Symbol)
    value = lowercase(String(_required(object, name)))
    occursin(r"^[0-9a-f]{64}$", value) ||
        error("$name is not a lowercase SHA-256")
    return value
end

function _inside(path, root)
    absolute = lowercase(abspath(path))
    prefix =
        lowercase(abspath(root)) *
        string(Base.Filesystem.path_separator)
    return startswith(absolute, prefix)
end

function _dataset_digest(manifest_sha, records)
    return canonical_sha256((;
        manifest_sha256=manifest_sha,
        shards=[(
            path=record.path,
            sha256=record.sha256,
            bytes=record.bytes,
            global_first=record.global_first,
            global_last=record.global_last,
        ) for record in records],
    ))
end

function _verify_source_hashes!(manifest)
    hashes = _required(manifest, :source_hashes)
    String(_required(hashes, :modeldb_repository_url)) ==
        EXPECTED_MODELDB_URL ||
        error("teacher is not pinned ModelDB accession 139653")
    String(_required(hashes, :modeldb_git_commit)) ==
        EXPECTED_MODELDB_COMMIT ||
        error("teacher ModelDB commit differs")
    String(_required(hashes, :modeldb_git_tree)) ==
        EXPECTED_MODELDB_TREE ||
        error("teacher ModelDB tree differs")
    String(_required(hashes, :modeldb_tracked_status)) == "" ||
        error("teacher reports dirty ModelDB tracked files")
    _required_sha(hashes, :generator_source_sha256) ==
        EXPECTED_GENERATOR_SHA256 ||
        error("teacher base generator source differs")
    _required_sha(hashes, :final_generator_source_sha256) ==
        EXPECTED_FINAL_GENERATOR_SHA256 ||
        error("teacher final generator source differs")
    _file_sha256(_GENERATOR_SOURCE) ==
        EXPECTED_GENERATOR_SHA256 ||
        error("local base teacher generator changed")
    _file_sha256(_FINAL_GENERATOR_SOURCE) ==
        EXPECTED_FINAL_GENERATOR_SHA256 ||
        error("local final teacher generator changed")
    for name in (
        :mechanism_library_sha256,
        :mechanism_sources_sha256,
        :morphology_sha256,
        :modeldb_tree_sha256,
        :biophysics_sha256,
        :template_sha256,
    )
        _required_sha(hashes, name)
    end
    contract_hashes =
        _required(_required(manifest, :teacher_contract), :source_hashes)
    canonical_sha256(hashes) == canonical_sha256(contract_hashes) ||
        error("manifest and teacher-contract source hashes differ")
    return canonical_sha256(hashes)
end

function _verify_manifest_and_shards(manifest_path, shard_directory)
    manifest_absolute = abspath(String(manifest_path))
    root = abspath(String(shard_directory))
    isfile(manifest_absolute) ||
        error("teacher manifest is absent: $manifest_absolute")
    isdir(root) || error("teacher shard directory is absent: $root")
    dirname(manifest_absolute) == root ||
        error("sealed manifest must reside in the verified shard directory")

    raw = read(manifest_absolute, String)
    contract = Contract.verify_teacher_contract_manifest(raw)
    manifest = JSON3.read(raw)
    String(_required(manifest, :schema_name)) == EXPECTED_SCHEMA ||
        error("teacher schema is not final.v2")
    Int(_required(manifest, :schema_version)) == 2 ||
        error("teacher schema version differs")
    String(_required(manifest, :model_name)) == EXPECTED_MODEL ||
        error("teacher model family differs")
    String(_required(manifest, :stage)) ==
        "official_hay_neuron_teacher_final" ||
        error("manifest is not the official Hay teacher stage")
    String(_required(manifest, :completion_state)) == "complete" ||
        error("teacher generation is incomplete")
    _required(manifest, :modeldb_source_modified_by_generator) === false ||
        error("teacher generator modified ModelDB source")
    Int(_required(manifest, :total_segments)) == 642 ||
        error("teacher does not expose 642 Hay segments")
    declared_contract =
        _required_sha(manifest, :teacher_contract_sha256)
    contract.digest == declared_contract ||
        error("strict teacher-contract digest differs")
    declared_contract in KNOWN_DEVELOPMENT_CONTRACTS ||
        error("sealed development release accepts only canonical dev1500 contract")
    source_hashes_sha256 = _verify_source_hashes!(manifest)

    config = _required(manifest, :config)
    sample_dt = Float64(_required(config, :sample_dt_ms))
    sample_dt == 1.0 ||
        error("teacher publication interval must be 1 ms")
    Float64(_required(config, :dt_ms)) == 0.025 ||
        error("detailed teacher integration interval must be 0.025 ms")
    duration = Float64(_required(config, :duration_ms))
    train_trials = Int(_required(config, :train_trials))
    heldout_trials = Int(_required(config, :test_trials))
    validation_ids =
        Int32.(collect(_required(
            manifest,
            :validation_from_train_indices,
        )))
    length(validation_ids) ==
        Int(_required(config, :validation_trials_from_train)) ||
        error("manifest validation count differs")
    length(unique(validation_ids)) == length(validation_ids) ||
        error("manifest validation IDs repeat")

    records = _ShardRecord[]
    expected_first = 1
    inventory_paths = Set{String}()
    for raw_record in _required(manifest, :shards)
        relative = String(_required(raw_record, :path))
        basename(relative) == relative ||
            error("shard path is not a plain basename")
        relative in inventory_paths &&
            error("manifest repeats shard $relative")
        push!(inventory_paths, relative)
        path = abspath(joinpath(root, relative))
        _inside(path, root) ||
            error("shard escapes verified directory")
        isfile(path) || error("teacher shard is absent: $path")
        first = Int(_required(raw_record, :global_first))
        last = Int(_required(raw_record, :global_last))
        samples = Int(_required(raw_record, :samples))
        first == expected_first ||
            error("shard global ranges are not contiguous")
        last - first + 1 == samples ||
            error("shard sample count/range differs")
        expected_first = last + 1
        bytes = Int64(_required(raw_record, :bytes))
        filesize(path) == bytes ||
            error("shard byte count differs: $relative")
        shard_sha = _required_sha(raw_record, :sha256)
        _file_sha256(path) == shard_sha ||
            error("shard SHA-256 mismatch: $relative")
        _required_sha(raw_record, :teacher_contract_sha256) ==
            declared_contract ||
            error("shard uses another teacher contract")
        String(_required(raw_record, :schema_name)) == EXPECTED_SCHEMA ||
            error("shard schema differs")
        push!(
            records,
            _ShardRecord(
                relative,
                shard_sha,
                bytes,
                first,
                last,
                samples,
            ),
        )
    end
    completed = Int(_required(manifest, :completed_trials))
    expected_first == completed + 1 ||
        error("shards do not cover all completed trials")
    completed == train_trials + heldout_trials ||
        error("completed trial count differs from config")
    extra_npz = Set(
        file for file in readdir(root)
        if endswith(lowercase(file), ".npz")
    )
    extra_npz == inventory_paths ||
        error("verified directory NPZ inventory differs from manifest")

    # Read only IDs/codes after every shard byte has been verified.
    train_ids = Int32[]
    test_ids = Int32[]
    for record in records
        path = joinpath(root, record.path)
        ids_and_codes = NPZ.npzread(
            path,
            ["sample_indices", "split_code"],
        )
        ids = Int32.(vec(ids_and_codes["sample_indices"]))
        codes = UInt8.(vec(ids_and_codes["split_code"]))
        ids == Int32.(record.global_first:record.global_last) ||
            error("shard sample_indices differ from manifest range")
        length(codes) == length(ids) ||
            error("split_code length differs")
        for (id, code) in zip(ids, codes)
            code == 1 ? push!(train_ids, id) :
            code == 3 ? push!(test_ids, id) :
            error("final.v2 split_code must be 1=train or 3=test")
        end
    end
    length(train_ids) == train_trials ||
        error("actual train IDs differ from config")
    length(test_ids) == heldout_trials ||
        error("actual held-out IDs differ from config")
    all(id -> id in Set(train_ids), validation_ids) ||
        error("validation IDs are not a subset of split-code 1")
    validation_set = Set(validation_ids)
    fit_ids = Int32[id for id in train_ids if id ∉ validation_set]

    conflict = _required(
        _required(manifest, :teacher_contract),
        :connectivity_scale_conflict,
    )
    paper_scale_claim = Bool(
        _required(conflict, :fully_paper_scale_claim),
    )
    acknowledged = Bool(
        _required(
            conflict,
            :interpretation_explicitly_acknowledged,
        ),
    )
    uncertainty =
        paper_scale_claim && acknowledged ?
        "none" :
        String(_required(conflict, :production_interpretation))
    manifest_sha = _file_sha256(manifest_absolute)
    manifest_sha == EXPECTED_DEV1500_MANIFEST_SHA256 ||
        error("manifest bytes are not canonical dev1500")
    return _SealedDataset(
        manifest_absolute,
        root,
        manifest_sha,
        declared_contract,
        _dataset_digest(manifest_sha, records),
        source_hashes_sha256,
        records,
        validation_ids,
        fit_ids,
        test_ids,
        duration,
        sample_dt,
        train_trials,
        heldout_trials,
        paper_scale_claim,
        acknowledged,
        uncertainty,
    )
end

# -- canonical model/protocol ----------------------------------------------

function _metadata_forbidden(metadata)
    for name in propertynames(metadata)
        text = lowercase(String(name))
        if startswith(text, "held_out") ||
           text in (
               "verification_passed",
               "verified",
               "verification",
               "attestation",
               "attestation_sha256",
           )
            error("caller-injected verification metadata `$name` is forbidden")
        end
    end
end

function _validate_model!(frozen)
    frozen isa Twin.FrozenOfficialELMTwin ||
        error("sealed release requires PaperELMTwinOfficialV2Final frozen type")
    Twin.assert_frozen_official_elm_unchanged(frozen)
    _metadata_forbidden(frozen.metadata)
    config = frozen.model.config
    config.num_input == 1_278 ||
        error("official ELM input_dim differs")
    config.num_branch == 45 ||
        error("official ELM branch count differs")
    config.num_synapse_per_branch == 100 ||
        error("official ELM branch fan-in differs")
    config.num_memory == 1_000 ||
        error("official ELM must use 1,000 memory units")
    config.hidden_size == 2_000 ||
        error("official ELM hidden width must be 2,000")
    config.nmda_regions == 4 ||
        error("project NMDA extension must have four regions")
    config.lambda_value == 5.0f0 ||
        error("official ELM lambda differs")
    config.tau_b_ms == 5.0f0 ||
        error("official ELM branch tau differs")
    config.memory_tau_min_ms == 0.1f0 ||
        error("official ELM minimum memory tau differs")
    config.memory_tau_max_ms == 300.0f0 ||
        error("official ELM maximum memory tau differs")
    config.learn_memory_tau === false ||
        error("official ELM memory taus must remain fixed")
    config.delta_t_ms == 1.0f0 ||
        error("official ELM time step differs")
    config.input_to_synapse_routing == :neuronio_routing ||
        error("official NeuronIO routing differs")
    frozen.model isa Twin.ProfiledOfficialPaperELMTwin ||
        error("sealed release requires activation-profiled Final model")
    frozen.model.mlp_activation === :silu ||
        error("official Final executable activation must be SiLU")
    frozen.model.compatibility_profile === :twinprop_paper_reconstruction ||
        error("sealed twin must use TwinProp paper reconstruction profile")
    Twin.assert_profiled_official_elm_contract(frozen.model)
    frozen.normalizer isa Twin.OfficialELMNormalizer ||
        error("base/core fitted input normalizer is forbidden")

    regenerated = Twin.build_profiled_official_elm_twin(
        config;
        mlp_activation=:silu,
        compatibility_profile=:twinprop_paper_reconstruction,
    )
    frozen.model.input_indices == regenerated.input_indices ||
        error("official routing indices differ from regeneration")
    frozen.model.valid_indices_mask ==
        regenerated.valid_indices_mask ||
        error("official routing mask differs from regeneration")
    frozen.model.initial_proto_tau_m ==
        regenerated.initial_proto_tau_m ||
        error("official fixed memory taus differ from regeneration")
    frozen.model.kappa_b == regenerated.kappa_b ||
        error("official branch decay differs from regeneration")
    return nothing
end

function _training_protocol(frozen)
    metadata = frozen.metadata
    hasproperty(metadata, :training_protocol) ||
        error("frozen model lacks sealed training_protocol metadata")
    protocol = metadata.training_protocol
    Int(_required(protocol, :restarts)) == 3 ||
        error("release training requires three restarts")
    run_ids = String.(collect(_required(protocol, :run_ids)))
    seeds = UInt64.(collect(_required(protocol, :seeds)))
    length(run_ids) == 3 && length(unique(run_ids)) == 3 ||
        error("release needs three unique run IDs")
    length(seeds) == 3 && length(unique(seeds)) == 3 ||
        error("release needs three unique seeds")
    Int(_required(protocol, :epochs)) == 35 ||
        error("release training requires 35 epochs")
    Int(_required(protocol, :batch_size)) == 8 ||
        error("release training batch size must be 8")
    Float64(_required(protocol, :window_ms)) == 500.0 ||
        error("release training window must be 500 ms")
    Float64(_required(protocol, :training_loss_burn_in_ms)) == 0.0 ||
        error("training loss must consume all 500 window bins")
    Float64(_required(protocol, :evaluation_window_overlap_burn_in_ms)) == 150.0 ||
        error("evaluation window overlap burn-in must be 150 ms")
    Float64(_required(protocol, :heldout_global_ignore_start_ms)) == 500.0 ||
        error("held-out global ignore time must be 500 ms")
    Tuple(Int.(collect(_required(protocol, :random_window_start_indices_julia)))) == (501, 1000) ||
        error("training random window starts must be Julia 501:1000")
    lowercase(String(_required(protocol, :random_window_sampling))) == "uniform_with_replacement" ||
        error("training random window sampling must be uniform with replacement")
    Int(_required(protocol, :upstream_reference_batches_per_epoch)) == 10_000 ||
        error("upstream Spieler reference must record 10k batches/epoch")
    Int(_required(protocol, :batches_per_epoch)) == 4 ||
        error("dev1500 schedule must record four batches/epoch")
    Bool(_required(protocol, :development_schedule_choice)) === true ||
        error("dev1500 four-batch schedule must be labeled a project choice")
    lowercase(String(_required(protocol, :optimizer))) == "adam" ||
        error("release optimizer must be Adam")
    Float64(_required(protocol, :learning_rate)) == 5e-4 ||
        error("release learning rate must be 5e-4")
    Float64(_required(protocol, :weight_decay)) == 0.0 ||
        error("release training must not use weight decay")
    lowercase(String(_required(protocol, :schedule))) in (
        "cosine",
        "cosineannealing",
        "cosine_annealing",
    ) || error("release schedule must be cosine annealing")
    lowercase(String(_required(protocol, :selection_split))) in (
        "derived_validation",
        "validation_from_train",
    ) || error("selection must use train-derived validation")
    lowercase(String(_required(protocol, :selection_metric))) in (
        "physical_voltage_rmse",
        "physical_voltage_rmse_mv",
    ) || error("selection metric must be physical voltage RMSE")
    String(_required(protocol, :selected_run_id)) in run_ids ||
        error("selected run ID is not one of the three restarts")
    Bool(_required(protocol, :heldout_used_for_selection)) === false ||
        error("held-out data was used for model selection")
    Int(_required(protocol, :heldout_reads_after_selection)) == 1 ||
        error("held-out must be read once after selection")
    for name in (:training_log_sha256, :selection_record_sha256)
        value = lowercase(String(_required(protocol, name)))
        occursin(r"^[0-9a-f]{64}$", value) ||
            error("$name is not a complete SHA-256")
    end
    return protocol
end

# -- final.v2 signed-1278 adapter ------------------------------------------

function _load_numeric(dataset, record)
    data = NPZ.npzread(joinpath(dataset.root, record.path), _NUMERIC_KEYS)
    ids = Int32.(vec(data["sample_indices"]))
    ids == Int32.(record.global_first:record.global_last) ||
        error("numeric shard IDs changed after sealing")
    return data
end

function _contact_range(data, item)
    offsets = Int64.(vec(data["contact_trial_offset"]))
    return (Int(offsets[item]) + 1):Int(offsets[item + 1])
end

function _event_range(data, item)
    offsets = Int64.(vec(data["event_trial_offset"]))
    return (Int(offsets[item]) + 1):Int(offsets[item + 1])
end

function _validated_offsets(data, key, samples, total)
    offsets = Int64.(vec(data[key]))
    length(offsets) == samples + 1 ||
        error("$key must have samples+1 entries")
    first(offsets) == 0 || error("$key must start at zero")
    last(offsets) == total || error("$key terminal offset differs")
    all(diff(offsets) .>= 0) || error("$key is not monotone")
    all(value -> 0 <= value <= total, offsets) ||
        error("$key is outside its ragged array bounds")
    return offsets
end

function _validate_numeric!(data)
    ids = vec(data["sample_indices"])
    samples = length(ids)
    steps = size(data["target_voltage"], 1)
    size(data["target_voltage"], 2) == samples ||
        error("voltage target sample dimension differs")
    size(data["target_spike"]) == size(data["target_voltage"]) ||
        error("spike/voltage target shapes differ")
    size(data["target_nmda"]) ==
        (NMDA_REGIONS, steps, samples) ||
        error("regional NMDA target shape differs")
    all(isfinite, data["target_voltage"]) ||
        error("voltage target is non-finite")
    all(isfinite, data["target_nmda"]) ||
        error("NMDA target is non-finite")
    all(value -> value == 0 || value == 1, data["target_spike"]) ||
        error("spike target is not binary")

    axon_kind = data["axon_kind"]
    size(axon_kind, 2) == samples ||
        error("axon_kind sample dimension differs")
    all(value -> value == 1 || value == 2, axon_kind) ||
        error("axon_kind must contain only E/I codes")
    axons = size(axon_kind, 1)

    contact_axon = vec(data["contact_axon"])
    contact_segment = vec(data["contact_segment"])
    contact_kind = vec(data["contact_kind"])
    contact_strength = vec(data["contact_strength"])
    contact_slot = vec(data["contact_location_slot"])
    contacts = length(contact_axon)
    all(length(values) == contacts for values in (
        contact_segment,
        contact_kind,
        contact_strength,
        contact_slot,
    )) || error("ragged contact array lengths differ")
    contact_offsets = _validated_offsets(
        data,
        "contact_trial_offset",
        samples,
        contacts,
    )
    for item in 1:samples
        used_slots = Set{Tuple{UInt8,Int}}()
        for contact in (Int(contact_offsets[item]) + 1):Int(contact_offsets[item + 1])
            axon = Int(contact_axon[contact])
            segment = Int(contact_segment[contact])
            kind = UInt8(contact_kind[contact])
            strength = Float32(contact_strength[contact])
            slot = Int(contact_slot[contact])
            1 <= axon <= axons || error("contact axon is out of bounds")
            2 <= segment <= 640 ||
                error("contact is not on a dendritic segment")
            kind in (UInt8(1), UInt8(2)) ||
                error("contact kind is not E/I")
            UInt8(axon_kind[axon, item]) == kind ||
                error("contact violates axon Dale identity")
            isfinite(strength) && 0.0f0 <= strength <= 1.0f0 ||
                error("contact strength is outside [0,1]")
            slot >= 1 || error("contact location slot is not positive")
            key = (kind, slot)
            key in used_slots &&
                error("contact location slot repeats within E/I kind")
            push!(used_slots, key)
        end
    end

    event_axon = vec(data["event_axon"])
    event_time = vec(data["event_time_bin"])
    event_count = vec(data["event_count"])
    events = length(event_axon)
    length(event_time) == events && length(event_count) == events ||
        error("ragged event array lengths differ")
    event_offsets = _validated_offsets(
        data,
        "event_trial_offset",
        samples,
        events,
    )
    for item in 1:samples
        previous = (0, -1)
        for event in (Int(event_offsets[item]) + 1):Int(event_offsets[item + 1])
            axon = Int(event_axon[event])
            time = Int(event_time[event])
            count = Int(event_count[event])
            1 <= axon <= axons || error("event axon is out of bounds")
            0 <= time < steps || error("event time bin is out of bounds")
            1 <= count <= typemax(UInt16) ||
                error("event multiplicity cannot fit UInt16")
            pair = (axon, time)
            pair > previous ||
                error("events must be unique axon-major/time-minor order")
            previous = pair
        end
    end
    return nothing
end

function _expand_input(data, item, time_range)
    axon_kind = data["axon_kind"]
    axons = size(axon_kind, 1)
    output = zeros(Float32, INPUT_DIM, length(time_range), 1)
    event_counts = zeros(UInt16, axons, length(time_range))
    first_bin = first(time_range) - 1
    last_bin = last(time_range) - 1
    event_axon = vec(data["event_axon"])
    event_time = vec(data["event_time_bin"])
    multiplicity = vec(data["event_count"])
    for event in _event_range(data, item)
        axon = Int(event_axon[event])
        time_bin = Int(event_time[event])
        1 <= axon <= axons || error("event axon is outside catalog")
        if first_bin <= time_bin <= last_bin
            count = Int(multiplicity[event])
            count >= 1 || error("event multiplicity must be positive")
            local_time = time_bin - first_bin + 1
            updated = Int(event_counts[axon, local_time]) + count
            updated <= typemax(UInt16) ||
                error("event-count accumulation overflows UInt16")
            event_counts[axon, local_time] = UInt16(updated)
        end
    end

    contact_axon = vec(data["contact_axon"])
    contact_segment = vec(data["contact_segment"])
    contact_kind = vec(data["contact_kind"])
    contact_strength = vec(data["contact_strength"])
    for contact in _contact_range(data, item)
        axon = Int(contact_axon[contact])
        segment = Int(contact_segment[contact])
        kind = UInt8(contact_kind[contact])
        strength = Float32(contact_strength[contact])
        1 <= axon <= axons || error("contact axon is outside catalog")
        2 <= segment <= 640 ||
            error("ELM contact is not on a dendritic segment")
        kind in (UInt8(1), UInt8(2)) ||
            error("contact kind is not E/I")
        UInt8(axon_kind[axon, item]) == kind ||
            error("contact violates axon Dale identity")
        isfinite(strength) && 0.0f0 <= strength <= 1.0f0 ||
            error("contact strength is invalid")
        feature =
            kind == UInt8(1) ?
            segment - 1 :
            DENDRITIC_LOCATIONS + segment - 1
        sign = kind == UInt8(1) ? 1.0f0 : -1.0f0
        for local_time in eachindex(time_range)
            count = event_counts[axon, local_time]
            count == 0 && continue
            output[feature, local_time, 1] +=
                sign * strength * Float32(count)
        end
    end
    return output
end

# -- online metrics and exact external AUROC -------------------------------

mutable struct _PairMoments
    n::Int64
    sum_x::Float64
    sum_y::Float64
    sum_x2::Float64
    sum_y2::Float64
    sum_xy::Float64
    error2::Float64
end

_PairMoments() = _PairMoments(0, 0, 0, 0, 0, 0, 0)

function _update!(moments::_PairMoments, predicted, target)
    for index in eachindex(predicted, target)
        x = Float64(predicted[index])
        y = Float64(target[index])
        isfinite(x) && isfinite(y) ||
            error("metric observation is non-finite")
        moments.n += 1
        moments.sum_x += x
        moments.sum_y += y
        moments.sum_x2 += x * x
        moments.sum_y2 += y * y
        moments.sum_xy += x * y
        difference = x - y
        moments.error2 += difference * difference
    end
end

function _rmse(moments::_PairMoments)
    moments.n > 0 || return NaN
    return sqrt(moments.error2 / moments.n)
end

function _normalized_rmse(moments::_PairMoments)
    moments.n > 0 || return NaN
    centered =
        moments.sum_y2 -
        moments.sum_y * moments.sum_y / moments.n
    centered > eps(Float64) || return Inf
    return sqrt(moments.error2 / centered)
end

function _correlation(moments::_PairMoments)
    moments.n > 0 || return NaN
    covariance =
        moments.sum_xy -
        moments.sum_x * moments.sum_y / moments.n
    xvar =
        moments.sum_x2 -
        moments.sum_x * moments.sum_x / moments.n
    yvar =
        moments.sum_y2 -
        moments.sum_y * moments.sum_y / moments.n
    denominator = sqrt(max(xvar, 0) * max(yvar, 0))
    denominator > eps(Float64) || return NaN
    return covariance / denominator
end

mutable struct _AUROCSpool
    root::String
    capacity::Int
    scores::Vector{Float32}
    labels::Vector{UInt8}
    run_paths::Vector{String}
    observations::Int64
    positives::Int64
end

function _AUROCSpool(root, capacity)
    return _AUROCSpool(
        root,
        capacity,
        Float32[],
        UInt8[],
        String[],
        0,
        0,
    )
end

function _flush!(spool::_AUROCSpool)
    isempty(spool.scores) && return
    order = sortperm(spool.scores; alg=MergeSort)
    path = joinpath(
        spool.root,
        "auroc_run_" *
        lpad(string(length(spool.run_paths) + 1), 6, '0') *
        ".bin",
    )
    open(path, "w") do io
        write(io, UInt64(length(order)))
        for index in order
            write(io, spool.scores[index])
            write(io, spool.labels[index])
        end
    end
    push!(spool.run_paths, path)
    empty!(spool.scores)
    empty!(spool.labels)
end

function _push!(spool::_AUROCSpool, scores, labels)
    for index in eachindex(scores, labels)
        score = Float32(scores[index])
        label_value = labels[index]
        isfinite(score) || error("spike score is non-finite")
        (label_value == 0 || label_value == 1) ||
            error("spike label is not binary")
        label = UInt8(label_value)
        push!(spool.scores, score)
        push!(spool.labels, label)
        spool.observations += 1
        spool.positives += label
        length(spool.scores) >= spool.capacity && _flush!(spool)
    end
end

@inline _heap_less(left, right) =
    left[1] < right[1] ||
    (left[1] == right[1] && left[2] < right[2])

function _heap_push!(heap, entry)
    push!(heap, entry)
    index = length(heap)
    while index > 1
        parent = index >>> 1
        _heap_less(heap[index], heap[parent]) || break
        heap[index], heap[parent] = heap[parent], heap[index]
        index = parent
    end
end

function _heap_pop!(heap)
    result = heap[1]
    last = pop!(heap)
    if !isempty(heap)
        heap[1] = last
        index = 1
        while true
            left = index << 1
            left > length(heap) && break
            right = left + 1
            child =
                right <= length(heap) &&
                _heap_less(heap[right], heap[left]) ?
                right : left
            _heap_less(heap[child], heap[index]) || break
            heap[index], heap[child] = heap[child], heap[index]
            index = child
        end
    end
    return result
end

function _exact_auroc!(spool::_AUROCSpool)
    _flush!(spool)
    negatives = spool.observations - spool.positives
    spool.positives > 0 && negatives > 0 || return NaN
    readers = IO[]
    remaining = UInt64[]
    heap = Tuple{Float32,Int,UInt8}[]
    try
        for (run, path) in enumerate(spool.run_paths)
            io = open(path, "r")
            push!(readers, io)
            count = read(io, UInt64)
            push!(remaining, count)
            if count > 0
                score = read(io, Float32)
                label = read(io, UInt8)
                remaining[run] -= 1
                _heap_push!(heap, (score, run, label))
            end
        end
        position = Int64(0)
        group_first = Int64(1)
        group_score = 0.0f0
        group_positives = Int64(0)
        group_count = Int64(0)
        rank_sum = 0.0
        initialized = false
        while !isempty(heap)
            score, run, label = _heap_pop!(heap)
            position += 1
            if !initialized || score != group_score
                if initialized
                    group_last = position - 1
                    rank_sum +=
                        group_positives *
                        (group_first + group_last) / 2
                end
                initialized = true
                group_score = score
                group_first = position
                group_positives = 0
                group_count = 0
            end
            group_count += 1
            group_positives += label
            if remaining[run] > 0
                next_score = read(readers[run], Float32)
                next_label = read(readers[run], UInt8)
                remaining[run] -= 1
                _heap_push!(heap, (next_score, run, next_label))
            end
        end
        initialized || return NaN
        rank_sum +=
            group_positives *
            (group_first + position) / 2
        return (
            rank_sum -
            spool.positives * (spool.positives + 1) / 2
        ) / (spool.positives * negatives)
    finally
        foreach(close, readers)
    end
end

# -- normalizer provenance and sealed evaluation --------------------------

function _fit_nmda_statistics(dataset)
    fit_set = Set(dataset.fit_ids)
    moments = [_PairMoments() for _ in 1:NMDA_REGIONS]
    # Use x=target and y=0 only to accumulate x moments.
    for record in dataset.records
        data = _load_numeric(dataset, record)
        _validate_numeric!(data)
        ids = Int32.(vec(data["sample_indices"]))
        target = data["target_nmda"]
        for (item, id) in enumerate(ids)
            id in fit_set || continue
            for region in 1:NMDA_REGIONS
                values = @view target[region, :, item]
                for value in values
                    x = Float64(value)
                    moment = moments[region]
                    moment.n += 1
                    moment.sum_x += x
                    moment.sum_x2 += x * x
                end
            end
        end
    end
    means = Float32[
        moment.sum_x / moment.n for moment in moments
    ]
    scales = Float32[
        sqrt(max(
            moment.sum_x2 / moment.n -
            (moment.sum_x / moment.n)^2,
            0.0,
        )) for moment in moments
    ]
    all(>(0), scales) ||
        error("fit-split NMDA scale is non-positive")
    return (; mean=means, scale=scales)
end

function _connectivity_statistics(dataset)
    total_contacts = Int64(0)
    excitatory_contacts = Int64(0)
    inhibitory_contacts = Int64(0)
    total_events = Int64(0)
    multiplicity_events = Int64(0)
    maximum_event_count = 0
    minimum_contacts_per_axon = typemax(Int)
    maximum_contacts_per_axon = 0
    minimum_strength = Inf
    maximum_strength = -Inf
    trials = 0
    axon_count = 0
    for record in dataset.records
        data = _load_numeric(dataset, record)
        _validate_numeric!(data)
        samples = length(vec(data["sample_indices"]))
        axons = size(data["axon_kind"], 1)
        axon_count == 0 ? (axon_count = axons) :
            axon_count == axons || error("axon count differs across shards")
        contact_offsets = Int64.(vec(data["contact_trial_offset"]))
        event_offsets = Int64.(vec(data["event_trial_offset"]))
        contact_axons = Int.(vec(data["contact_axon"]))
        contact_kinds = UInt8.(vec(data["contact_kind"]))
        strengths = Float64.(vec(data["contact_strength"]))
        event_counts = Int.(vec(data["event_count"]))
        for item in 1:samples
            trials += 1
            counts_by_axon = zeros(Int, axons)
            for contact in (Int(contact_offsets[item]) + 1):Int(contact_offsets[item + 1])
                total_contacts += 1
                kind = contact_kinds[contact]
                kind == UInt8(1) ?
                    (excitatory_contacts += 1) :
                    (inhibitory_contacts += 1)
                counts_by_axon[contact_axons[contact]] += 1
                minimum_strength = min(minimum_strength, strengths[contact])
                maximum_strength = max(maximum_strength, strengths[contact])
            end
            minimum_contacts_per_axon = min(
                minimum_contacts_per_axon,
                minimum(counts_by_axon),
            )
            maximum_contacts_per_axon = max(
                maximum_contacts_per_axon,
                maximum(counts_by_axon),
            )
            for event in (Int(event_offsets[item]) + 1):Int(event_offsets[item + 1])
                total_events += 1
                count = event_counts[event]
                count > 1 && (multiplicity_events += 1)
                maximum_event_count = max(maximum_event_count, count)
            end
        end
    end
    trials > 0 && total_contacts > 0 ||
        error("sealed dataset has no connectivity observations")
    return (;
        trials,
        axons_per_trial=axon_count,
        total_contacts,
        excitatory_contacts,
        inhibitory_contacts,
        mean_contacts_per_trial=total_contacts / trials,
        mean_contacts_per_axon=
            total_contacts / (trials * axon_count),
        minimum_contacts_per_axon,
        maximum_contacts_per_axon,
        minimum_strength,
        maximum_strength,
        total_events,
        multiplicity_events,
        maximum_event_count,
        paper_contact_interpretation_promotable=false,
    )
end

function _verify_normalizer!(frozen, statistics)
    normalizer = frozen.normalizer
    all(isapprox.(
        normalizer.nmda_mean,
        statistics.mean;
        rtol=2.0f-5,
        atol=2.0f-6,
    )) || error("NMDA normalizer mean is not fit-split derived")
    all(isapprox.(
        normalizer.nmda_scale,
        statistics.scale;
        rtol=2.0f-5,
        atol=2.0f-6,
    )) || error("NMDA normalizer scale is not fit-split derived")
end

function _evaluate(dataset, frozen, scratch)
    burn_in_ms = 500.0
    burn_in_steps = Int(round(burn_in_ms / dataset.sample_dt_ms))
    burn_in_steps < Int(round(dataset.duration_ms / dataset.sample_dt_ms)) ||
        error("teacher duration is not longer than the 500 ms fidelity discard")
    voltage = _PairMoments()
    nmda = [_PairMoments() for _ in 1:NMDA_REGIONS]
    spool = _AUROCSpool(scratch, AUROC_RUN_RECORDS)
    heldout_set = Set(dataset.heldout_ids)
    seen = Int32[]
    total_bins = 0
    evaluated_bins = 0

    for record in dataset.records
        data = _load_numeric(dataset, record)
        _validate_numeric!(data)
        ids = Int32.(vec(data["sample_indices"]))
        steps = size(data["target_voltage"], 1)
        expected_steps =
            Int(round(dataset.duration_ms / dataset.sample_dt_ms))
        steps == expected_steps ||
            error("target duration differs from manifest")
        for (item, id) in enumerate(ids)
            id in heldout_set || continue
            push!(seen, id)
            total_bins += steps
            state = nothing
            for first_step in 1:EVALUATOR_CHUNK_STEPS:steps
                last_step = min(
                    first_step + EVALUATOR_CHUNK_STEPS - 1,
                    steps,
                )
                time_range = first_step:last_step
                input = _expand_input(data, item, time_range)
                prediction = Twin.twin_forward(
                    frozen,
                    input;
                    initial_state=state,
                )
                state = prediction.final_state
                first_metric = max(first_step, burn_in_steps + 1)
                first_metric > last_step && continue
                local_first = first_metric - first_step + 1
                local_range = local_first:length(time_range)
                target_voltage = @view data[
                    "target_voltage"
                ][first_metric:last_step, item:item]
                target_spike = @view data[
                    "target_spike"
                ][first_metric:last_step, item:item]
                _update!(
                    voltage,
                    @view(prediction.voltage[local_range, :]),
                    target_voltage,
                )
                _push!(
                    spool,
                    @view(prediction.spike_logit[local_range, :]),
                    target_spike,
                )
                for region in 1:NMDA_REGIONS
                    _update!(
                        nmda[region],
                        @view(prediction.nmda[
                            region,
                            local_range,
                            :,
                        ]),
                        @view(data["target_nmda"][
                            region,
                            first_metric:last_step,
                            item:item,
                        ]),
                    )
                end
                evaluated_bins += length(local_range)
            end
        end
    end
    seen == dataset.heldout_ids ||
        error("held-out trials were not evaluated in manifest order")
    auroc = _exact_auroc!(spool)
    return (;
        spike_auroc=auroc,
        voltage_rmse_mv=_rmse(voltage),
        voltage_correlation=_correlation(voltage),
        nmda_raw_rmse_by_region=[_rmse(value) for value in nmda],
        nmda_normalized_rmse_by_region=[
            _normalized_rmse(value) for value in nmda
        ],
        nmda_correlation_by_region=[
            _correlation(value) for value in nmda
        ],
        raw_heldout_bins=total_bins,
        evaluated_bins,
        burn_in_ms,
        spike_positives=spool.positives,
        spike_negatives=
            spool.observations - spool.positives,
        auroc_external_runs=length(spool.run_paths),
        peak_auroc_records=AUROC_RUN_RECORDS,
        evaluator_chunk_steps=EVALUATOR_CHUNK_STEPS,
    )
end

function _gate(metrics)
    finite =
        isfinite(metrics.spike_auroc) &&
        isfinite(metrics.voltage_rmse_mv) &&
        isfinite(metrics.voltage_correlation) &&
        all(isfinite, metrics.nmda_raw_rmse_by_region) &&
        all(isfinite, metrics.nmda_normalized_rmse_by_region) &&
        all(isfinite, metrics.nmda_correlation_by_region)
    passed =
        finite &&
        metrics.spike_auroc >= MINIMUM_SPIKE_AUROC &&
        metrics.voltage_rmse_mv <= MAXIMUM_VOLTAGE_RMSE_MV &&
        all(
            <=(MAXIMUM_REGIONAL_NMDA_NORMALIZED_RMSE),
            metrics.nmda_normalized_rmse_by_region,
        )
    return (;
        minimum_spike_auroc=MINIMUM_SPIKE_AUROC,
        maximum_voltage_rmse_mv=MAXIMUM_VOLTAGE_RMSE_MV,
        maximum_regional_nmda_normalized_rmse=
            MAXIMUM_REGIONAL_NMDA_NORMALIZED_RMSE,
        correlations_are_report_only=true,
        passed,
    )
end

function _paper_scale(dataset)
    known_development =
        dataset.teacher_contract_sha256 in
        KNOWN_DEVELOPMENT_CONTRACTS ||
        startswith(dataset.teacher_contract_sha256, "d661")
    return (
        !known_development &&
        dataset.train_trials == PAPER_TRAIN_POOL_TRIALS &&
        dataset.heldout_trials == PAPER_HELDOUT_TRIALS &&
        dataset.duration_ms == PAPER_DURATION_MS &&
        dataset.paper_scale_claim &&
        dataset.connectivity_acknowledged
    )
end

function _build_payload(dataset, frozen, protocol, statistics, connectivity, metrics)
    fixed_gate = _gate(metrics)
    paper_scale = _paper_scale(dataset)
    promotable = paper_scale && fixed_gate.passed
    config = frozen.model.config
    return (;
        schema=SEALED_RELEASE_SCHEMA,
        artifact_kind=SEALED_RELEASE_ARTIFACT_KIND,
        canonical_encoding=CANONICAL_ENCODING,
        evaluator=(;
            id=EVALUATOR_ID,
            source_sha256=_file_sha256(_MODULE_SOURCE),
            bounded_dense_input=true,
            chunk_steps=EVALUATOR_CHUNK_STEPS,
            recurrent_state_carry=true,
            heldout_ignore_time_from_start_ms=500.0,
            heldout_metric_interval_julia="501:end",
            training_random_window_start_indices_julia=(501, 1000),
            training_random_window_sampling="uniform_with_replacement",
            online_voltage_nmda_moments=true,
            exact_spike_auroc_external_sort=true,
            auroc_run_record_bound=AUROC_RUN_RECORDS,
        ),
        teacher=(;
            manifest_sha256=dataset.manifest_sha256,
            teacher_contract_sha256=
                dataset.teacher_contract_sha256,
            source_dataset_sha256=dataset.source_dataset_sha256,
            source_hashes_sha256=dataset.source_hashes_sha256,
            shard_count=length(dataset.records),
            shard_inventory_sha256=canonical_sha256(dataset.records),
            connectivity_statistics=connectivity,
            connectivity_statistics_sha256=canonical_sha256(connectivity),
        ),
        split=(;
            fit_count=length(dataset.fit_ids),
            validation_count=length(dataset.validation_ids),
            heldout_count=length(dataset.heldout_ids),
            fit_ids_sha256=canonical_sha256(dataset.fit_ids),
            validation_from_train_indices=
                copy(dataset.validation_ids),
            validation_ids_sha256=
                canonical_sha256(dataset.validation_ids),
            heldout_ids_sha256=
                canonical_sha256(dataset.heldout_ids),
            duration_ms=dataset.duration_ms,
            sample_dt_ms=dataset.sample_dt_ms,
            raw_heldout_bins=
                dataset.heldout_trials *
                Int(round(
                    dataset.duration_ms /
                    dataset.sample_dt_ms,
                )),
            paper_scale_uncertainty=
                dataset.paper_scale_uncertainty,
        ),
        model=(;
            family="Paper-ELM-v2-OfficialRouting-Twin-Final",
            input_dim=config.num_input,
            branches=config.num_branch,
            synapses_per_branch=config.num_synapse_per_branch,
            memory_units=config.num_memory,
            hidden_size=config.hidden_size,
            nmda_regions=config.nmda_regions,
            identity_input_normalization=true,
            fixed_soma_transform=true,
            source_metadata_sha256=
                canonical_sha256(Twin.official_elm_source_metadata()),
            final_source_sha256=_file_sha256(_MODEL_SOURCE),
            final_base_source_sha256=
                _file_sha256(_FINAL_BASE_SOURCE),
            final_differentiable_source_sha256=
                _file_sha256(_FINAL_DIFFERENTIABLE_SOURCE),
            activation_profile_source_sha256=
                _file_sha256(_ACTIVATION_PROFILE_SOURCE),
            activation_hotfix_source_sha256=
                _file_sha256(_ACTIVATION_HOTFIX_SOURCE),
            profiled_loader_source_sha256=
                _file_sha256(_PROFILED_LOADER_SOURCE),
            final_loader_source_sha256=
                _file_sha256(_FINAL_LOADER_SOURCE),
            executable_mlp_activation=frozen.model.mlp_activation,
            compatibility_profile=
                frozen.model.compatibility_profile,
            core_source_sha256=_file_sha256(_CORE_SOURCE),
            contract_verifier_source_sha256=
                _file_sha256(_CONTRACT_SOURCE),
            config_sha256=canonical_sha256(config),
            routing_sha256=canonical_sha256((;
                indices=frozen.model.input_indices,
                mask=frozen.model.valid_indices_mask,
            )),
            parameter_sha256=frozen.parameter_sha256,
            normalizer_sha256=
                canonical_sha256(frozen.normalizer),
            base_artifact_sha256=frozen.artifact_sha256,
            artifact_payload_sha256=
                _artifact_payload_sha256(frozen),
        ),
        normalizer_provenance=(;
            source_split="fit_only",
            fit_ids_sha256=canonical_sha256(dataset.fit_ids),
            recomputed_nmda_mean=statistics.mean,
            recomputed_nmda_scale=statistics.scale,
            statistics_sha256=canonical_sha256(statistics),
        ),
        training_protocol=protocol,
        training_protocol_sha256=canonical_sha256(protocol),
        target_contract=(;
            original_paper_targets=(
                "soma_voltage",
                "soma_spike",
            ),
            project_required_extension_targets=(
                "regional_nmda_current",
            ),
            regional_nmda_is_project_extension=true,
            nmda_gate_is_project_defined=true,
            paper_identical_training_claimed=false,
            unpublished_checkpoint_identity_claimed=false,
        ),
        metrics,
        fixed_gate,
        outcome=(;
            gate_passed=fixed_gate.passed,
            paper_scale,
            development_scale=!paper_scale,
            promotable_production=promotable,
            metrics_recomputed_from_verified_shards=true,
            caller_metrics_accepted=false,
            caller_targets_accepted=false,
            caller_manifest_digest_accepted=false,
        ),
    )
end

function _with_scratch(callback, scratch_root)
    if scratch_root === nothing
        return mktempdir() do directory
            callback(directory)
        end
    end
    root = abspath(String(scratch_root))
    isdir(root) || error("scratch root is absent: $root")
    return mktempdir(root) do directory
        callback(directory)
    end
end

"""
The only release-attestation constructor.

It accepts a manifest path, the directory containing the manifest's verified
shards, and the exact canonical Final frozen twin.  It accepts no identity,
raw array, target, metric, threshold, pass flag, or paper-scale argument.
"""
function attest_sealed_official_elm_release(
    manifest_path::AbstractString,
    shard_directory::AbstractString,
    frozen::Twin.FrozenOfficialELMTwin;
    scratch_root=nothing,
)
    dataset = _verify_manifest_and_shards(
        manifest_path,
        shard_directory,
    )
    _validate_model!(frozen)
    protocol = _training_protocol(frozen)
    statistics = _fit_nmda_statistics(dataset)
    _verify_normalizer!(frozen, statistics)
    connectivity = _connectivity_statistics(dataset)
    metrics = _with_scratch(scratch_root) do scratch
        _evaluate(dataset, frozen, scratch)
    end
    payload = _build_payload(
        dataset,
        frozen,
        protocol,
        statistics,
        connectivity,
        metrics,
    )
    attestation = SealedOfficialELMReleaseAttestation(
        payload,
        canonical_sha256(payload),
    )
    return SealedOfficialELMRelease(frozen, attestation)
end

function verify_sealed_official_elm_release(
    bundle::SealedOfficialELMRelease,
    manifest_path::AbstractString,
    shard_directory::AbstractString;
    require_gate::Bool=false,
    require_production::Bool=false,
    scratch_root=nothing,
)
    canonical_sha256(bundle.attestation.payload) ==
        bundle.attestation.attestation_sha256 ||
        error("sealed attestation payload digest mismatch")
    expected = attest_sealed_official_elm_release(
        manifest_path,
        shard_directory,
        bundle.frozen;
        scratch_root,
    )
    expected.attestation.attestation_sha256 ==
        bundle.attestation.attestation_sha256 ||
        error("sealed attestation differs from recomputed evidence")
    require_gate &&
        bundle.attestation.payload.outcome.gate_passed !== true &&
        error("sealed held-out release gate failed")
    require_production &&
        bundle.attestation.payload.outcome.promotable_production !== true &&
        error("development-scale artifact is not production/promotable")
    return bundle
end

function save_sealed_official_elm_release(path, bundle)
    bundle isa SealedOfficialELMRelease ||
        error("only SealedOfficialELMRelease can be saved")
    canonical_sha256(bundle.attestation.payload) ==
        bundle.attestation.attestation_sha256 ||
        error("refusing to save a corrupt sealed attestation")
    Twin.assert_frozen_official_elm_unchanged(bundle.frozen)
    parent = dirname(abspath(path))
    isdir(parent) || mkpath(parent)
    jldsave(
        path;
        artifact_kind=SEALED_RELEASE_ARTIFACT_KIND,
        format_version=SEALED_RELEASE_FORMAT_VERSION,
        bundle,
    )
    return abspath(path)
end

function _load(path)
    isfile(path) || error("sealed release artifact is absent: $path")
    data = JLD2.load(path)
    get(data, "artifact_kind", nothing) ==
        SEALED_RELEASE_ARTIFACT_KIND ||
        error("artifact kind is not SealedOfficialELMRelease")
    get(data, "format_version", nothing) ==
        SEALED_RELEASE_FORMAT_VERSION ||
        error("sealed release artifact version differs")
    bundle = get(data, "bundle", nothing)
    bundle isa SealedOfficialELMRelease ||
        error("artifact payload is not the exact sealed release type")
    return bundle
end

function load_checked_sealed_official_elm_release(
    artifact_path,
    manifest_path,
    shard_directory;
    scratch_root=nothing,
)
    return verify_sealed_official_elm_release(
        _load(artifact_path),
        manifest_path,
        shard_directory;
        scratch_root,
    )
end

function load_verified_sealed_official_elm_release(
    artifact_path,
    manifest_path,
    shard_directory;
    require_production::Bool=true,
    scratch_root=nothing,
)
    return verify_sealed_official_elm_release(
        _load(artifact_path),
        manifest_path,
        shard_directory;
        require_gate=true,
        require_production,
        scratch_root,
    )
end

end # module PaperELMTwinOfficialV2SealedRelease
