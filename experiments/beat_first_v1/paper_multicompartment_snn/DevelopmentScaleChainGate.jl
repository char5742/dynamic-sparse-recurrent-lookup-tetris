module DevelopmentScaleChainGate

using JSON3
using NPZ
using SHA

const _PARENT_MODULE = parentmodule(@__MODULE__)
if !isdefined(_PARENT_MODULE, :OfficialTeacherContract)
    Base.include(
        _PARENT_MODULE,
        joinpath(@__DIR__, "OfficialTeacherContract.jl"),
    )
end
import ..OfficialTeacherContract

export DEVELOPMENT_TEACHER_MANIFEST,
    DEVELOPMENT_TEACHER_SCHEMA,
    DEVELOPMENT_TEACHER_CONTRACT_SHA256,
    DEVELOPMENT_TEACHER_MANIFEST_SHA256,
    DEFAULT_HAY_MODELDB_ROOT,
    verify_development_teacher_manifest,
    verify_paper_scale_teacher_manifest,
    verify_development_scale_chain

const DEVELOPMENT_TEACHER_MANIFEST =
    raw"C:\tmp\hd_swsnn_neuron_teacher_final_smoke_rich64_release\manifest.json"
const DEVELOPMENT_TEACHER_SCHEMA =
    "hd_swsnn_twinprop.neuron_teacher.final.v2"
const DEVELOPMENT_TEACHER_CONTRACT_SHA256 =
    "d66177cb43612947c5fbdc9e65c55f963a1800b2c70b57bba094a32a7ab890a5"
const DEVELOPMENT_TEACHER_MANIFEST_SHA256 =
    "4f49975bf60bd27db1db121671749384b53941e8ad2c8b73ea648e2874198f5d"
const DEFAULT_HAY_MODELDB_ROOT = raw"C:\tmp\hay_modeldb_139653"

const _MODEL_NAME = "HD-SWSNN-TwinProp"
const _OFFICIAL_STAGE = "official_hay_neuron_teacher_final"
const _MODELDB_URL = "https://github.com/ModelDBRepository/139653.git"
const _MODELDB_COMMIT =
    "50a4aab3ce5c295ad16a134c5d9261b7cc3fbe58"
const _MODELDB_GIT_TREE =
    "ffdabdeaa0b6f0d358d5d56ac0f0d046e14f534a"

const _EXPECTED_SOURCE_HASHES = (
    modeldb_tree_sha256=
        "953b96fb36374e79e9b32ffd675828bf18ba89f625efe09f38207e47d74703b8",
    morphology_sha256=
        "293d0aa92af8d03dbdcc40a711ba3923522615c3a65df485906538a1de986e23",
    biophysics_sha256=
        "67b528fe8446e4d76461cd339983caabd986efc92bf023eb5ec4515a55d817cc",
    mechanism_sources_sha256=
        "8bb6edc972a83c83a6ecab3654ef92f44e706b986a1366d6a5c4b97bc03a89f0",
    mechanism_library_sha256=
        "a33cfc0d3a3bf75667fc3f159b01d745cce414bc6d19a288680fe846098a95f9",
    template_sha256=
        "85b57479d246d0122cfc57830d9466151a5dbe7deae4b648fbc1308bbab1643d",
    generator_source_sha256=
        "5de5096f0d292e841e9115b72cfd378433dfdd1d884901e6e04b765c1f092f71",
    final_generator_source_sha256=
        "0d0fbfaeb326b8a6a91822c5c80717b84490cb2db8bd6e1dca253a01e8ae34ba",
)

const _REQUIRED_ARRAYS = (
    :sample_indices,
    :split_code,
    :target_voltage,
    :target_spike,
    :target_nmda,
    :target_compartment_voltage,
    :target_compartment_nmda,
    :target_dendritic_cai,
    :target_dendritic_ica,
    :target_ca_event,
    :contact_trial_offset,
    :contact_axon,
    :contact_segment,
    :contact_location_slot,
    :contact_section,
    :contact_x,
    :contact_path_distance_um,
    :contact_kind,
    :contact_strength,
    :event_trial_offset,
    :event_axon,
    :event_time_bin,
    :event_count,
)

const _NONOFFICIAL_FLAGS = (
    :provisional,
    :is_provisional,
    :synthetic,
    :is_synthetic,
    :reconstructed,
    :is_reconstructed,
    :reconstruction_control,
    :allow_reconstruction,
)

@inline function _get(value, name::Symbol, default=nothing)
    value === nothing && return default
    hasproperty(value, name) && return getproperty(value, name)
    if value isa AbstractDict
        haskey(value, name) && return value[name]
        text = String(name)
        haskey(value, text) && return value[text]
    end
    return default
end

@inline function _required(value, name::Symbol)
    result = _get(value, name, nothing)
    result === nothing && error("required field '$name' is absent")
    return result
end

function _require_sha256(value, label::AbstractString)
    digest = lowercase(String(value))
    occursin(r"^[0-9a-f]{64}$", digest) ||
        error("$label is not a lower-case SHA-256 digest")
    return digest
end

_file_sha256(path::AbstractString) =
    bytes2hex(SHA.sha256(read(path)))

function _canonical_json(value)
    if value isa AbstractDict ||
       (
        !(value isa AbstractArray) &&
        !(value isa AbstractString) &&
        !(value isa Number) &&
        !(value isa Bool) &&
        value !== nothing &&
        propertynames(value) != ()
       )
        names = sort!(String.(collect(propertynames(value))))
        if value isa AbstractDict
            names = sort!(String.(collect(keys(value))))
        end
        return "{" * join(
            (
                JSON3.write(name) * ":" *
                _canonical_json(_required(value, Symbol(name)))
                for name in names
            ),
            ",",
        ) * "}"
    elseif value isa AbstractArray
        return "[" *
            join((_canonical_json(child) for child in value), ",") *
            "]"
    end
    return String(JSON3.write(value))
end

_canonical_equal(left, right) =
    _canonical_json(left) == _canonical_json(right)

function _safe_child(root::AbstractString, relative::AbstractString)
    isempty(relative) && error("empty child path")
    root_path = normpath(abspath(root))
    child = normpath(abspath(joinpath(root_path, relative)))
    prefix = lowercase(root_path) *
        string(Base.Filesystem.path_separator)
    (
        lowercase(child) == lowercase(root_path) ||
        startswith(lowercase(child), prefix)
    ) || error("path escapes verified dataset root: $relative")
    return child
end

function _git_bytes(root::AbstractString, arguments::AbstractString...)
    command = Cmd(vcat(["git", "-C", String(root)], collect(arguments)))
    try
        return read(command)
    catch exception
        error(
            "cannot verify ModelDB git metadata: " *
            sprint(showerror, exception),
        )
    end
end

_git_text(root::AbstractString, arguments::AbstractString...) =
    strip(String(_git_bytes(root, arguments...)))

function _sha256_named_files(
    paths::AbstractVector{<:AbstractString},
    root::AbstractString,
)
    ordered = sort!(
        collect(String.(paths));
        by=path -> replace(relpath(path, root), '\\' => '/'),
    )
    context = SHA.SHA2_256_CTX()
    for path in ordered
        isfile(path) || error("hashed source file is absent: $path")
        relative = replace(relpath(path, root), '\\' => '/')
        relative_bytes = collect(codeunits(relative))
        length32 = UInt32(length(relative_bytes))
        length_bytes = UInt8[
            UInt8(length32 & 0xff),
            UInt8((length32 >> 8) & 0xff),
            UInt8((length32 >> 16) & 0xff),
            UInt8((length32 >> 24) & 0xff),
        ]
        SHA.update!(context, length_bytes)
        SHA.update!(context, relative_bytes)
        SHA.update!(context, hex2bytes(_file_sha256(path)))
    end
    return bytes2hex(SHA.digest!(context))
end

function _tracked_paths(root::AbstractString)
    raw = String(_git_bytes(root, "ls-files", "-z"))
    entries = split(raw, '\0'; keepempty=false)
    isempty(entries) && error("ModelDB checkout exposes no tracked files")
    return String[
        joinpath(root, split(entry, '/')...)
        for entry in entries
    ]
end

function _verify_real_hay_sources(
    source_hashes;
    modeldb_root::AbstractString,
)
    root = abspath(modeldb_root)
    isdir(root) ||
        error("pinned Hay ModelDB checkout is absent: $root")

    declared = Pair{Symbol,String}[]
    for name in propertynames(_EXPECTED_SOURCE_HASHES)
        expected = getproperty(_EXPECTED_SOURCE_HASHES, name)
        actual = _require_sha256(
            _required(source_hashes, name),
            "source_hashes.$name",
        )
        actual == expected ||
            error("source_hashes.$name differs from the pinned Hay source")
        push!(declared, name => actual)
    end

    String(_required(source_hashes, :modeldb_repository_url)) ==
        _MODELDB_URL ||
        error("teacher is not ModelDB accession 139653")
    String(_required(source_hashes, :modeldb_git_commit)) ==
        _MODELDB_COMMIT ||
        error("teacher ModelDB commit differs from the pinned Hay commit")
    String(_required(source_hashes, :modeldb_git_tree)) ==
        _MODELDB_GIT_TREE ||
        error("teacher ModelDB git tree differs from the pinned Hay tree")
    String(_required(source_hashes, :modeldb_tracked_status)) == "" ||
        error("teacher reports a dirty ModelDB tracked source")

    _git_text(root, "rev-parse", "HEAD") == _MODELDB_COMMIT ||
        error("local Hay checkout commit differs from teacher lineage")
    _git_text(root, "rev-parse", "HEAD^{tree}") == _MODELDB_GIT_TREE ||
        error("local Hay checkout tree differs from teacher lineage")
    _git_text(root, "status", "--porcelain", "--untracked-files=no") == "" ||
        error("local Hay checkout has modified tracked files")

    morphology = joinpath(root, "morphologies", "cell1.asc")
    biophysics = joinpath(root, "models", "L5PCbiophys3.hoc")
    template = joinpath(root, "models", "L5PCtemplate.hoc")
    mechanism_library = joinpath(root, "x86_64", "libnrnmech.so")
    for path in (morphology, biophysics, template, mechanism_library)
        isfile(path) || error("required Hay source artifact is absent: $path")
    end

    _file_sha256(morphology) ==
        _EXPECTED_SOURCE_HASHES.morphology_sha256 ||
        error("real Hay morphology SHA-256 differs")
    _file_sha256(biophysics) ==
        _EXPECTED_SOURCE_HASHES.biophysics_sha256 ||
        error("real Hay biophysics SHA-256 differs")
    _file_sha256(template) ==
        _EXPECTED_SOURCE_HASHES.template_sha256 ||
        error("real Hay template SHA-256 differs")
    _file_sha256(mechanism_library) ==
        _EXPECTED_SOURCE_HASHES.mechanism_library_sha256 ||
        error("compiled NEURON mechanism library SHA-256 differs")

    mod_sources = String[
        path for path in readdir(joinpath(root, "mod"); join=true)
        if endswith(lowercase(path), ".mod")
    ]
    isempty(mod_sources) &&
        error("real Hay ModelDB checkout has no NMODL mechanisms")
    _sha256_named_files(mod_sources, root) ==
        _EXPECTED_SOURCE_HASHES.mechanism_sources_sha256 ||
        error("NMODL mechanism source-set SHA-256 differs")
    _sha256_named_files(_tracked_paths(root), root) ==
        _EXPECTED_SOURCE_HASHES.modeldb_tree_sha256 ||
        error("tracked ModelDB source-tree SHA-256 differs")

    base_generator = joinpath(@__DIR__, "neuron_hay_teacher.py")
    final_generator = joinpath(@__DIR__, "neuron_hay_teacher_final.py")
    _file_sha256(base_generator) ==
        _EXPECTED_SOURCE_HASHES.generator_source_sha256 ||
        error("base official Hay generator changed after teacher creation")
    _file_sha256(final_generator) ==
        _EXPECTED_SOURCE_HASHES.final_generator_source_sha256 ||
        error("final.v2 Hay generator changed after teacher creation")

    return (;
        modeldb_root=root,
        modeldb_repository_url=_MODELDB_URL,
        modeldb_git_commit=_MODELDB_COMMIT,
        modeldb_git_tree=_MODELDB_GIT_TREE,
        declared...,
        local_modeldb_tree_verified=true,
        local_morphology_verified=true,
        local_mechanism_sources_verified=true,
        local_mechanism_library_verified=true,
        local_generators_verified=true,
    )
end

function _assert_official_only(manifest, contract)
    String(_required(manifest, :schema_name)) ==
        DEVELOPMENT_TEACHER_SCHEMA ||
        error("teacher schema is not final.v2")
    Int(_required(manifest, :schema_version)) == 2 ||
        error("teacher schema_version is not 2")
    String(_required(manifest, :model_name)) == _MODEL_NAME ||
        error("teacher model family differs")
    String(_required(manifest, :stage)) == _OFFICIAL_STAGE ||
        error("teacher stage is not the final official Hay/NEURON stage")
    String(_required(manifest, :completion_state)) == "complete" ||
        error("teacher generation is incomplete")
    _required(manifest, :modeldb_source_modified_by_generator) === false ||
        error("generator reports modified ModelDB source files")
    _required(manifest, :resumable_sidecars_verified) === true ||
        error("teacher completion sidecars were not verified")
    startswith(
        String(_required(manifest, :neuron_version)),
        "NEURON -- VERSION ",
    ) || error("teacher has no real NEURON version lineage")

    for object in (manifest, contract)
        for name in _NONOFFICIAL_FLAGS
            value = _get(object, name, nothing)
            (
                value === nothing ||
                value === false
            ) || error(
                "non-official flag '$name' is present; provisional, " *
                "synthetic, and reconstructed teachers are forbidden",
            )
        end
    end
    return nothing
end

function _verify_manifest_contract(
    manifest_path::AbstractString,
    manifest_text::AbstractString,
    manifest,
    expected_contract_sha256::Union{Nothing,AbstractString},
)
    declared = _require_sha256(
        _required(manifest, :teacher_contract_sha256),
        "teacher_contract_sha256",
    )
    if expected_contract_sha256 !== nothing
        declared == lowercase(String(expected_contract_sha256)) ||
            error("teacher contract is not the approved development contract")
    end
    contract = _required(manifest, :teacher_contract)
    _require_sha256(
        _required(contract, :teacher_contract_sha256),
        "teacher_contract.teacher_contract_sha256",
    ) == declared ||
        error("manifest and embedded teacher contract digests differ")
    String(_required(contract, :schema_name)) ==
        String(_required(manifest, :schema_name)) ||
        error("manifest and teacher contract schemas differ")
    String(_required(contract, :model_name)) == _MODEL_NAME ||
        error("embedded teacher contract model family differs")

    verification =
        OfficialTeacherContract.verify_teacher_contract_manifest(
            manifest_text,
        )
    verification.digest == declared ||
        error("shared contract verifier returned another digest")
    _canonical_equal(
        _required(contract, :config),
        _required(manifest, :config),
    ) || error("manifest config differs from the signed teacher contract")
    _canonical_equal(
        _required(contract, :source_hashes),
        _required(manifest, :source_hashes),
    ) || error("manifest source hashes differ from the signed contract")
    _canonical_equal(
        _required(contract, :section_catalog),
        _required(manifest, :section_catalog),
    ) || error("manifest section catalog differs from the signed contract")
    _require_sha256(
        _required(contract, :location_slot_sha256),
        "teacher_contract.location_slot_sha256",
    )

    return contract, declared, verification.canonical_source
end

function _verify_protocol_contract(manifest, contract)
    paper = _required(manifest, :paper_production_contract)
    (
        Int(_required(paper, :train_trials)),
        Int(_required(paper, :held_out_test_trials)),
        Int(_required(paper, :duration_ms)),
    ) == (50_000, 2_000, 10_000) ||
        error("paper production contract must remain 50k/2k/10s")

    protocol = _required(contract, :paper_protocol)
    (
        Int(_required(protocol, :train_simulations)),
        Int(_required(protocol, :held_out_test_simulations)),
        Int(_required(protocol, :duration_ms)),
    ) == (50_000, 2_000, 10_000) ||
        error("signed paper protocol must remain 50k/2k/10s")
    Int(_required(protocol, :mean_contacts_per_axon)) == 20 ||
        error("signed paper protocol changed contact count")
    String(_required(protocol, :contact_density)) ==
        "<=1 E and <=1 I per dendrite micron" ||
        error("signed paper protocol changed contact-density rule")
    return nothing
end

function _validate_catalogs(manifest)
    Int(_required(manifest, :total_segments)) == 642 ||
        error("official Hay teacher must expose exactly 642 segments")
    segments = _required(manifest, :segments)
    length(segments) == 642 ||
        error("segment catalog does not contain 642 entries")
    for (position, segment) in enumerate(segments)
        Int(_required(segment, :index)) == position ||
            error("segment catalog is not contiguous at $position")
        region = Int(_required(segment, :region_code))
        0 <= region <= 3 ||
            error("invalid Hay region code at segment $position")
        for name in (:x, :distance_um, :length_um, :diameter_um, :area_um2)
            isfinite(Float64(_required(segment, name))) ||
                error("non-finite segment geometry at $position.$name")
        end
    end

    sections = _required(manifest, :section_catalog)
    isempty(sections) && error("section catalog is empty")
    for (position, section) in enumerate(sections)
        Int(_required(section, :index)) == position ||
            error("section catalog is not contiguous at $position")
        Float64(_required(section, :length_um)) > 0 ||
            error("section $position has non-positive length")
        Int(_required(section, :micron_slots)) > 0 ||
            error("section $position has no legal location slots")
        String(_required(section, :region)) in ("basal", "apical") ||
            error("section $position is not basal/apical")
    end
    Int(_required(manifest, :location_slots)) > 0 ||
        error("official teacher has no dendritic location slots")

    array_contract = _required(manifest, :array_contract)
    for name in _REQUIRED_ARRAYS
        _required(array_contract, name)
    end
    Int(_required(
        _required(array_contract, :contact_segment),
        :index_base,
    )) == 1 ||
        error("contact_segment is not a one-based Hay segment index")
    String(_required(
        _required(array_contract, :target_nmda),
        :sign,
    )) == "outward_positive" ||
        error("teacher NMDA current sign convention differs")

    return (;
        total_segments=642,
        section_count=length(sections),
        location_slots=Int(_required(manifest, :location_slots)),
        segment_catalog_sha256=
            bytes2hex(SHA.sha256(codeunits(_canonical_json(segments)))),
        section_catalog_sha256=
            bytes2hex(SHA.sha256(codeunits(_canonical_json(sections)))),
    )
end

function _scale_profile(manifest, contract, profile::Symbol)
    config = _required(manifest, :config)
    train_trials = Int(_required(config, :train_trials))
    held_out_test_trials = Int(_required(config, :test_trials))
    validation_from_train_trials =
        Int(_required(config, :validation_trials_from_train))
    duration_ms = Int(_required(config, :duration_ms))
    sample_dt_ms = Float64(_required(config, :sample_dt_ms))
    completed_trials = Int(_required(manifest, :completed_trials))
    conflict = _required(contract, :connectivity_scale_conflict)
    fully_paper_scale_claim =
        _required(conflict, :fully_paper_scale_claim)
    interpretation_acknowledged =
        _required(conflict, :interpretation_explicitly_acknowledged)

    if profile === :development
        (
            train_trials,
            held_out_test_trials,
            duration_ms,
            sample_dt_ms,
            completed_trials,
        ) == (40, 8, 100, 1.0, 48) ||
            error(
                "development rich64 teacher must be 40 train, 8 held-out, " *
                "100 ms at 1 ms sampling, and 48 complete trials",
            )
        validation_from_train_trials == 8 ||
            error("development rich64 must derive 8 validation trials")
        Int(_required(config, :axons)) == 64 ||
            error("development rich64 teacher must use 64 axons")
        String(_required(config, :preset)) == "smoke" ||
            error("development rich64 teacher must use the smoke preset")
        fully_paper_scale_claim === false ||
            error("development teacher falsely claims paper scale")
        interpretation_acknowledged === false ||
            error("development teacher unexpectedly acknowledges production")
        Bool(_required(
            config,
            :connectivity_interpretation_acknowledged,
        )) === false ||
            error("development teacher must not claim production connectivity")
        paper_scale = false
        promotable_production = false
    elseif profile === :paper_production
        (
            train_trials,
            held_out_test_trials,
            duration_ms,
            sample_dt_ms,
            completed_trials,
        ) == (50_000, 2_000, 10_000, 1.0, 52_000) ||
            error(
                "paper-scale production accepts only 50k train, 2k held-out, " *
                "10 s at 1 ms sampling, and 52k complete trials",
            )
        fully_paper_scale_claim === true ||
            error("paper-scale contract does not claim full paper scale")
        interpretation_acknowledged === true ||
            error("paper-scale connectivity interpretation is unacknowledged")
        Bool(_required(
            config,
            :connectivity_interpretation_acknowledged,
        )) === true ||
            error("paper-scale config has no connectivity acknowledgement")
        String(_required(config, :preset)) == "production" ||
            error("paper-scale teacher must use the production preset")
        paper_scale = true
        promotable_production = true
    else
        error("unknown scale profile $profile")
    end

    0 <= validation_from_train_trials < train_trials ||
        error("validation-from-train count is invalid")
    validation_indices = Int.(
        collect(_required(manifest, :validation_from_train_indices)),
    )
    expected_validation = collect(
        (train_trials - validation_from_train_trials + 1):train_trials,
    )
    validation_indices == expected_validation ||
        error("validation_from_train_indices are not the final train trials")

    return (;
        scale_profile=profile,
        train_trials,
        validation_from_train_trials,
        fit_trials=train_trials - validation_from_train_trials,
        held_out_test_trials,
        completed_trials,
        duration_ms,
        sample_dt_ms,
        time_steps=round(Int, duration_ms / sample_dt_ms),
        paper_scale,
        promotable_production,
    )
end

function _split_counts(codes)
    all(code -> code in (UInt8(1), UInt8(3)), codes) ||
        error("development/final teacher split_code must be 1=train or 3=test")
    return (;
        train=count(==(UInt8(1)), codes),
        validation=0,
        held_out_test=count(==(UInt8(3)), codes),
    )
end

function _declared_split_counts(record)
    counts = _required(record, :split_counts)
    return (;
        train=Int(_get(counts, :train, 0)),
        validation=Int(_get(counts, :validation, 0)),
        held_out_test=Int(_get(
            counts,
            :held_out_test,
            _get(counts, :test, 0),
        )),
    )
end

function _verify_shards(
    root::AbstractString,
    manifest,
    contract_sha256::AbstractString,
    scale,
)
    records = collect(_required(manifest, :shards))
    isempty(records) && error("official final.v2 teacher has no shards")
    context = SHA.SHA2_256_CTX()
    verified_bytes = 0
    expected_first = 1
    aggregate = (train=0, validation=0, held_out_test=0)
    sidecars = Set{String}()

    for (ordinal, record) in enumerate(records)
        Int(_required(record, :shard_index)) == ordinal ||
            error("shard indices are not contiguous at $ordinal")
        String(_required(record, :schema_name)) ==
            DEVELOPMENT_TEACHER_SCHEMA ||
            error("shard $ordinal schema differs")
        _require_sha256(
            _required(record, :teacher_contract_sha256),
            "shard[$ordinal].teacher_contract_sha256",
        ) == contract_sha256 ||
            error("shard $ordinal teacher contract differs")

        first = Int(_required(record, :global_first))
        last = Int(_required(record, :global_last))
        first == expected_first ||
            error("shard ranges are not contiguous at shard $ordinal")
        last >= first || error("shard $ordinal has an empty range")
        samples = last - first + 1
        Int(_required(record, :samples)) == samples ||
            error("shard $ordinal sample count differs from its range")
        expected_first = last + 1

        relative = String(_required(record, :path))
        path = _safe_child(root, relative)
        lowercase(splitext(path)[2]) == ".npz" ||
            error("official shard is not NPZ: $relative")
        isfile(path) || error("official shard is absent: $path")
        declared_bytes = Int(_required(record, :bytes))
        filesize(path) == declared_bytes ||
            error("shard byte length differs: $relative")
        declared_sha = _require_sha256(
            _required(record, :sha256),
            "shard[$ordinal].sha256",
        )
        _file_sha256(path) == declared_sha ||
            error("shard SHA-256 differs: $relative")

        payload = NPZ.npzread(path, ["sample_indices", "split_code"])
        payload isa AbstractDict ||
            error("selective NPZ read failed for $relative")
        haskey(payload, "sample_indices") ||
            error("shard lacks sample_indices: $relative")
        haskey(payload, "split_code") ||
            error("shard lacks split_code: $relative")
        sample_indices = Int.(vec(payload["sample_indices"]))
        sample_indices == collect(first:last) ||
            error("shard sample_indices differ from the manifest range")
        codes = UInt8.(vec(payload["split_code"]))
        length(codes) == samples ||
            error("shard split_code length differs")
        actual_counts = _split_counts(codes)
        actual_counts == _declared_split_counts(record) ||
            error("shard split counts differ: $relative")
        aggregate = (;
            train=aggregate.train + actual_counts.train,
            validation=aggregate.validation + actual_counts.validation,
            held_out_test=
                aggregate.held_out_test + actual_counts.held_out_test,
        )

        sidecar = splitext(path)[1] * ".done.json"
        isfile(sidecar) ||
            error("verified completion sidecar is absent: $sidecar")
        sidecar_record = JSON3.read(read(sidecar, String))
        _canonical_equal(sidecar_record, record) ||
            error("completion sidecar differs from manifest: $sidecar")
        push!(sidecars, lowercase(abspath(sidecar)))

        SHA.update!(context, hex2bytes(declared_sha))
        SHA.update!(context, codeunits(string(declared_bytes)))
        verified_bytes += declared_bytes
    end

    expected_first == scale.completed_trials + 1 ||
        error("shard ranges do not cover every completed trial")
    aggregate == (
        train=scale.train_trials,
        validation=0,
        held_out_test=scale.held_out_test_trials,
    ) || error("verified NPZ split totals differ from the scale contract")

    discovered_sidecars = Set(
        lowercase(abspath(path))
        for path in readdir(root; join=true)
        if endswith(lowercase(path), ".done.json")
    )
    discovered_sidecars == sidecars ||
        error("completion sidecar set differs from the shard ledger")

    return (;
        shard_count=length(records),
        verified_shard_count=length(records),
        verified_shard_bytes=verified_bytes,
        verified_split_counts=aggregate,
        shard_ledger_sha256=bytes2hex(SHA.digest!(context)),
        all_shard_hashes_verified=true,
        all_sidecars_verified=true,
    )
end

function _verify_teacher(
    manifest_path::AbstractString,
    profile::Symbol;
    expected_contract_sha256::Union{Nothing,AbstractString}=nothing,
    expected_manifest_sha256::Union{Nothing,AbstractString}=nothing,
    modeldb_root::AbstractString=DEFAULT_HAY_MODELDB_ROOT,
)
    absolute_manifest = abspath(manifest_path)
    isfile(absolute_manifest) ||
        error("official teacher manifest is absent: $absolute_manifest")
    manifest_text = read(absolute_manifest, String)
    manifest_sha256 = _file_sha256(absolute_manifest)
    if expected_manifest_sha256 !== nothing
        manifest_sha256 == lowercase(String(expected_manifest_sha256)) ||
            error("development manifest SHA-256 differs from the approved artifact")
    end
    manifest = JSON3.read(manifest_text)
    contract, contract_sha256, canonical_source =
        _verify_manifest_contract(
            absolute_manifest,
            manifest_text,
            manifest,
            expected_contract_sha256,
        )
    _assert_official_only(manifest, contract)
    _verify_protocol_contract(manifest, contract)

    # Scale is classified before expensive source/shard verification.  A
    # development artifact can therefore never fall through into production.
    scale = _scale_profile(manifest, contract, profile)
    catalogs = _validate_catalogs(manifest)
    sources = _verify_real_hay_sources(
        _required(manifest, :source_hashes);
        modeldb_root,
    )
    root = dirname(absolute_manifest)
    shards = _verify_shards(
        root,
        manifest,
        contract_sha256,
        scale,
    )

    # Detect a concurrent manifest replacement after the lengthy source/shard
    # pass.  Frozen downstream artifacts must bind this returned digest.
    _file_sha256(absolute_manifest) == manifest_sha256 ||
        error("teacher manifest changed during gate verification")

    counts = (;
        train=scale.train_trials,
        validation_from_train=scale.validation_from_train_trials,
        fit=scale.fit_trials,
        held_out_test=scale.held_out_test_trials,
        completed=scale.completed_trials,
    )
    return (;
        gate_name=profile === :development ?
            :development_scale_teacher_gate :
            :paper_scale_teacher_gate,
        gate_version=1,
        verified=true,
        official_hay_neuron_teacher=true,
        provisional=false,
        synthetic=false,
        reconstructed=false,
        manifest_path=absolute_manifest,
        manifest_sha256,
        schema_name=DEVELOPMENT_TEACHER_SCHEMA,
        schema_version=2,
        teacher_contract_sha256=contract_sha256,
        teacher_contract_canonical_source=canonical_source,
        source_kind=:real_hay_modeldb_neuron,
        source_hashes=sources,
        counts,
        scale...,
        catalogs...,
        shards...,
        all_source_hashes_verified=true,
        all_hashes_verified=true,
        chain_complete=false,
        downstream_artifact_gate=:not_bound,
    )
end

"""
Verify the exact, non-promotable rich64 development teacher.

This gate intentionally returns `paper_scale=false` and
`promotable_production=false`.  It cannot be used as a paper-scale production
gate.  Its NamedTuple includes flat counts plus `counts` for driver/result
serialization.
"""
function verify_development_teacher_manifest(
    manifest_path::AbstractString=DEVELOPMENT_TEACHER_MANIFEST;
    modeldb_root::AbstractString=get(
        ENV,
        "HD_SWSNN_MODELDB_ROOT",
        DEFAULT_HAY_MODELDB_ROOT,
    ),
)
    return _verify_teacher(
        manifest_path,
        :development;
        expected_contract_sha256=DEVELOPMENT_TEACHER_CONTRACT_SHA256,
        expected_manifest_sha256=DEVELOPMENT_TEACHER_MANIFEST_SHA256,
        modeldb_root,
    )
end

"""
Verify only the separate paper-scale production teacher path.

Unlike the development gate, this requires 50,000 train simulations, 2,000
held-out simulations, 10-second trials, an explicit full-paper-scale claim,
and an acknowledged connectivity interpretation.  No development override is
accepted.
"""
function verify_paper_scale_teacher_manifest(
    manifest_path::AbstractString;
    modeldb_root::AbstractString=get(
        ENV,
        "HD_SWSNN_MODELDB_ROOT",
        DEFAULT_HAY_MODELDB_ROOT,
    ),
)
    return _verify_teacher(
        manifest_path,
        :paper_production;
        expected_contract_sha256=nothing,
        expected_manifest_sha256=nothing,
        modeldb_root,
    )
end

function _verified_downstream_gate(value, label::AbstractString)
    value isa NamedTuple ||
        error("$label must return a lineage NamedTuple")
    hasproperty(value, :verified) ||
        error("$label result has no verified field")
    getproperty(value, :verified) === true ||
        error("$label did not verify its artifact")
    hasproperty(value, :artifact_sha256) ||
        error("$label result has no artifact_sha256")
    _require_sha256(
        getproperty(value, :artifact_sha256),
        "$label.artifact_sha256",
    )
    return value
end

"""
Compose the development teacher gate with downstream artifact gates.

`official_v2_gate` and `distilled_cell_gate` are required keywords on purpose:
there is no permissive placeholder.  The OfficialV2 owner can bind its exact
module/type without this module importing an interim twin type.  Both callbacks
must return `(verified=true, artifact_sha256=..., ...)`.
"""
function verify_development_scale_chain(
    manifest_path::AbstractString=DEVELOPMENT_TEACHER_MANIFEST;
    modeldb_root::AbstractString=get(
        ENV,
        "HD_SWSNN_MODELDB_ROOT",
        DEFAULT_HAY_MODELDB_ROOT,
    ),
    official_v2_gate,
    distilled_cell_gate,
)
    teacher = verify_development_teacher_manifest(
        manifest_path;
        modeldb_root,
    )
    twin = _verified_downstream_gate(
        official_v2_gate(teacher),
        "official_v2_gate",
    )
    cell = _verified_downstream_gate(
        distilled_cell_gate(teacher, twin),
        "distilled_cell_gate",
    )
    return merge(
        teacher,
        (;
            chain_complete=true,
            downstream_artifact_gate=:verified,
            official_v2_artifact=twin,
            distilled_cell_artifact=cell,
        ),
    )
end

end # module DevelopmentScaleChainGate
