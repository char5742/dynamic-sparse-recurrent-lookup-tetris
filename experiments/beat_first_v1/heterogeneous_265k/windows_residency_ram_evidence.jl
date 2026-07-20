module WindowsResidencyRAMEvidence

using JSON3
using SHA

export EVIDENCE_IMPLEMENTATION_STATUS,
       EvidenceBindings,
       ExpectedProcessIdentity,
       ExpectedThreadIdentity,
       ExpectedStageInterval,
       ExpectedIMCProvenance,
       validate_etw_residency_evidence,
       validate_imc_ram_evidence,
       unavailable_evidence_artifact,
       validate_unavailable_evidence_artifact

const EVIDENCE_IMPLEMENTATION_STATUS = "UNEXECUTED_STATIC_ONLY"

const ETW_PROVIDER_NAME = "NT Kernel Logger/SystemTraceProvider"
const ETW_PROVIDER_GUID = "{9e814aad-3204-11d2-9a82-006008a86939}"
const ETW_CAPTURE_TOOL = "Microsoft Windows Performance Toolkit xperf.exe"
const ETW_EXTRACTOR_CAPABILITY = "RAW_QPC_CSWITCH_BOUNDARY_RECONSTRUCTION_V1"

const IMC_PROVIDER_NAME = "Intel Processor Counter Monitor (Intel PCM)"
const IMC_CAPTURE_TOOL = "pcm-memory.exe"
const IMC_PROJECT = "https://github.com/intel/pcm"
const IMC_ADAPTER_CAPABILITY = "INTEL_PCM_RAW_IMC_RD_WR_COUNTERS_V1"

const _HASH_PATTERN = r"^[0-9a-f]{64}$"
const _DECIMAL_PATTERN = r"^(0|[1-9][0-9]*)$"
const _SUPPORTED_CELLS = Set((
    "H0_cpu_hashstem_cpu_sparse",
    "H1_npu_hashstem_cpu_sparse",
))

"""Immutable provenance which every ETW/IMC artifact must repeat exactly.

`runner_result_sha256` is the SHA-256 of the runner `summary.json`, while
`raw_qpc_artifact_sha256` is the bound `records.jsonl` digest.  The evidence
producer is independent of both files and may not rewrite either one.
"""
Base.@kwdef struct EvidenceBindings
    cpu_set_topology_sha256::String
    system_contract_sha256::String
    benchmark_contract_sha256::String
    runner_source_sha256::String
    provider_source_sha256::String
    substrate_source_sha256::String
    source_manifest_sha256::String
    runner_result_sha256::String
    raw_qpc_artifact_sha256::String
    evidence_contract_sha256::String
    producer_source_sha256::String
end

"""Externally hashed probe/hook/raw artifacts which an IMC capture must match."""
Base.@kwdef struct ExpectedIMCProvenance
    availability_probe_sha256::String
    pcm_tool_sha256::String
    pcm_driver_sha256::String
    raw_adapter_sha256::String
    role_manifest_sha256::String
    timed_region_hook_source_sha256::String
    timed_region_witness_sha256::String
    before_snapshot_raw_sha256::String
    after_snapshot_raw_sha256::String
    counter_inventory_sha256::String
    counter_configuration_sha256::String
    program_generation::UInt64
end

"""ETW identity which prevents PID reuse or post-capture thread cherry-picking."""
Base.@kwdef struct ExpectedProcessIdentity
    process_id::UInt32
    runner_identity_qpc::Int64
    run_nonce::String
    role_manifest_sha256::String
    assignment_readback_sha256::String
end

"""Pre-bound OS thread ID plus role-registration QPC.

The independent ETW artifact must add the earlier ETW thread-start QPC because
Windows thread IDs are reused.
"""
Base.@kwdef struct ExpectedThreadIdentity
    thread_id::UInt32
    role_registration_qpc::Int64
end

"""One exact runner stage whose scheduled CPU residence must be reconstructed.

The interval is in raw QueryPerformanceCounter ticks. `ordinal` distinguishes
repeated intervals bearing the same stage name within one candidate-set record.
Owner thread IDs must come from the bound runner/role manifest, not from an
after-the-fact selection of ETW threads.
"""
Base.@kwdef struct ExpectedStageInterval
    cell_id::String
    repetition::Int
    sequence::Int
    set_id::String
    stage::String
    ordinal::Int
    begin_qpc::Int64
    end_qpc::Int64
    role::Symbol
    owner_threads::Vector{ExpectedThreadIdentity}
    allowed_cpu_set_ids::Vector{UInt32}
end

_isdict(value) = value isa AbstractDict

function _dictionary(value, label::AbstractString)
    _isdict(value) || throw(ArgumentError("$label must be an object"))
    return value
end

function _required(object, key::AbstractString, label::AbstractString=key)
    haskey(object, key) || throw(ArgumentError("missing $label"))
    return object[key]
end

function _string(value, label::AbstractString)
    value isa AbstractString || throw(ArgumentError("$label must be a string"))
    isempty(value) && throw(ArgumentError("$label must not be empty"))
    return String(value)
end

function _boolean(value, label::AbstractString)
    value isa Bool || throw(ArgumentError("$label must be boolean"))
    return value
end

function _integer(value, label::AbstractString)
    value isa Bool && throw(ArgumentError("$label must be an integer, not boolean"))
    value isa Integer || throw(ArgumentError("$label must be an integer"))
    return Int128(value)
end

function _nonnegative_tick(value, label::AbstractString)
    parsed = _integer(value, label)
    0 <= parsed <= typemax(Int64) || throw(ArgumentError("$label is outside Int64 QPC range"))
    return Int64(parsed)
end

function _positive_tick(value, label::AbstractString)
    parsed = _nonnegative_tick(value, label)
    parsed > 0 || throw(ArgumentError("$label must be positive"))
    return parsed
end

function _u32(value, label::AbstractString)
    parsed = _integer(value, label)
    0 <= parsed <= typemax(UInt32) || throw(ArgumentError("$label is outside UInt32"))
    return UInt32(parsed)
end

function _decimal_bigint(value, label::AbstractString)
    text = _string(value, label)
    occursin(_DECIMAL_PATTERN, text) || throw(ArgumentError(
        "$label must be a canonical unsigned decimal string",
    ))
    return parse(BigInt, text)
end

function _hash(value, label::AbstractString)
    normalized = lowercase(_string(value, label))
    occursin(_HASH_PATTERN, normalized) || throw(ArgumentError(
        "$label must be a lowercase SHA-256 hex digest",
    ))
    return normalized
end

function _binding_pairs(bindings::EvidenceBindings)
    return (
        "cpu_set_topology_sha256" => bindings.cpu_set_topology_sha256,
        "system_contract_sha256" => bindings.system_contract_sha256,
        "benchmark_contract_sha256" => bindings.benchmark_contract_sha256,
        "runner_source_sha256" => bindings.runner_source_sha256,
        "provider_source_sha256" => bindings.provider_source_sha256,
        "substrate_source_sha256" => bindings.substrate_source_sha256,
        "source_manifest_sha256" => bindings.source_manifest_sha256,
        "runner_result_sha256" => bindings.runner_result_sha256,
        "raw_qpc_artifact_sha256" => bindings.raw_qpc_artifact_sha256,
        "evidence_contract_sha256" => bindings.evidence_contract_sha256,
        "producer_source_sha256" => bindings.producer_source_sha256,
    )
end

function _validate_binding_values(bindings::EvidenceBindings)
    for (name, value) in _binding_pairs(bindings)
        _hash(value, name) == value || throw(ArgumentError("$name is not canonical lowercase"))
    end
    return nothing
end

function _validate_bindings(document, bindings::EvidenceBindings)
    _validate_binding_values(bindings)
    observed = _dictionary(_required(document, "bindings"), "bindings")
    for (name, expected) in _binding_pairs(bindings)
        _hash(_required(observed, name, "bindings.$name"), "bindings.$name") == expected ||
            throw(ArgumentError("bindings.$name mismatch"))
    end
    return nothing
end

function _record_property(record, name::Symbol)
    hasproperty(record, name) || throw(ArgumentError("topology record lacks $name"))
    return getproperty(record, name)
end

function _topology_maps(records)
    isempty(records) && throw(ArgumentError("CPU Set topology is empty"))
    by_id = Dict{UInt32,Any}()
    by_processor = Dict{Tuple{UInt16,UInt8},Any}()
    classes = Set{UInt8}()
    for record in records
        id = UInt32(_record_property(record, :id))
        group = UInt16(_record_property(record, :group))
        logical = UInt8(_record_property(record, :logical_processor_index))
        haskey(by_id, id) && throw(ArgumentError("duplicate CPU Set ID in topology"))
        haskey(by_processor, (group, logical)) && throw(ArgumentError(
            "duplicate processor group/logical index in topology",
        ))
        by_id[id] = record
        by_processor[(group, logical)] = record
        push!(classes, UInt8(_record_property(record, :efficiency_class)))
    end
    length(classes) == 2 || throw(ArgumentError(
        "P/E residency validation requires exactly two intrinsic efficiency classes",
    ))
    return by_id, by_processor, minimum(classes), maximum(classes)
end

function _validate_topology_digest(records, bindings::EvidenceBindings)
    owner = parentmodule(@__MODULE__)
    isdefined(owner, :windows_cpu_set_topology_sha256) || throw(ArgumentError(
        "windows_cpu_set_topology_sha256 must be loaded before evidence validation",
    ))
    observed = getfield(owner, :windows_cpu_set_topology_sha256)(records)
    observed == bindings.cpu_set_topology_sha256 || throw(ArgumentError(
        "topology records do not match the bound CPU Set topology SHA-256",
    ))
    return nothing
end

function _validate_role_manifest(role_manifest_bytes::AbstractVector{UInt8},
                                 bindings::EvidenceBindings,
                                 process_identity::ExpectedProcessIdentity,
                                 qpc_frequency::Int64)
    isempty(role_manifest_bytes) && throw(ArgumentError("role manifest bytes are empty"))
    observed_sha = bytes2hex(SHA.sha256(role_manifest_bytes))
    observed_sha == process_identity.role_manifest_sha256 || throw(ArgumentError(
        "role manifest bytes do not match the bound SHA-256",
    ))
    manifest = try
        JSON3.read(String(role_manifest_bytes), Dict{String,Any})
    catch error_value
        throw(ArgumentError("role manifest JSON is invalid: $(sprint(showerror, error_value))"))
    end
    _string(_required(manifest, "schema"), "role_manifest.schema") ==
        "heterogeneous-265k-thread-role-manifest-v1" || throw(ArgumentError(
            "role manifest schema mismatch",
        ))
    _string(_required(manifest, "implementation_status"),
            "role_manifest.implementation_status") == EVIDENCE_IMPLEMENTATION_STATUS ||
        throw(ArgumentError("role manifest implementation status mismatch"))
    _string(_required(manifest, "run_nonce"), "role_manifest.run_nonce") ==
        process_identity.run_nonce || throw(ArgumentError("role manifest run nonce mismatch"))
    _u32(_required(manifest, "process_id"), "role_manifest.process_id") ==
        process_identity.process_id || throw(ArgumentError("role manifest PID mismatch"))
    _nonnegative_tick(_required(manifest, "runner_identity_qpc"),
                      "role_manifest.runner_identity_qpc") ==
        process_identity.runner_identity_qpc || throw(ArgumentError(
            "role manifest runner identity QPC mismatch",
        ))
    _positive_tick(_required(manifest, "qpc_frequency"), "role_manifest.qpc_frequency") ==
        qpc_frequency || throw(ArgumentError("role manifest QPC frequency mismatch"))
    _hash(_required(manifest, "cpu_set_topology_sha256"),
          "role_manifest.cpu_set_topology_sha256") == bindings.cpu_set_topology_sha256 ||
        throw(ArgumentError("role manifest topology digest mismatch"))
    _hash(_required(manifest, "system_contract_sha256"),
          "role_manifest.system_contract_sha256") == bindings.system_contract_sha256 ||
        throw(ArgumentError("role manifest system-contract digest mismatch"))
    registrations = _required(manifest, "registrations")
    registrations isa AbstractVector || throw(ArgumentError(
        "role manifest registrations must be an array",
    ))
    isempty(registrations) && throw(ArgumentError("role manifest is empty"))
    indexed = Dict{UInt32,NamedTuple}()
    for raw_registration in registrations
        registration = _dictionary(raw_registration, "role manifest registration")
        thread_id = _u32(_required(registration, "thread_id"), "registration.thread_id")
        haskey(indexed, thread_id) && throw(ArgumentError(
            "role manifest repeats an OS thread ID",
        ))
        role_text = _string(_required(registration, "role"), "registration.role")
        role = if role_text == "p_sparse"
            :p_sparse
        elseif role_text == "e_background"
            :e_background
        else
            throw(ArgumentError("unknown role manifest role"))
        end
        registration_qpc = _nonnegative_tick(
            _required(registration, "role_registration_qpc"),
            "registration.role_registration_qpc",
        )
        ids = _sorted_u32(_required(registration, "allowed_cpu_set_ids"),
                          "registration.allowed_cpu_set_ids")
        isempty(ids) && throw(ArgumentError("role registration has no CPU Set"))
        assignment_sha = _hash(
            _required(registration, "assignment_readback_sha256"),
            "registration.assignment_readback_sha256",
        )
        indexed[thread_id] = (
            role=role,
            role_registration_qpc=registration_qpc,
            allowed_cpu_set_ids=ids,
            assignment_readback_sha256=assignment_sha,
        )
    end
    return indexed
end

function _stage_key(cell::AbstractString, repetition::Integer, sequence::Integer,
                    set_id::AbstractString, stage::AbstractString, ordinal::Integer)
    return (
        String(cell), Int(repetition), Int(sequence), String(set_id), String(stage), Int(ordinal),
    )
end

function _validated_expected_stages(expected_stages::AbstractVector{ExpectedStageInterval},
                                    by_id, low_class::UInt8, high_class::UInt8,
                                    role_manifest)
    isempty(expected_stages) && throw(ArgumentError("no expected ETW stage intervals"))
    expected = Dict{Tuple{String,Int,Int,String,String,Int},ExpectedStageInterval}()
    for interval in expected_stages
        interval.cell_id in _SUPPORTED_CELLS || throw(ArgumentError(
            "ETW evidence is restricted to post-idle H0/H1",
        ))
        interval.repetition > 0 || throw(ArgumentError("repetition must be positive"))
        interval.sequence > 0 || throw(ArgumentError("sequence must be positive"))
        isempty(interval.set_id) && throw(ArgumentError("set_id must not be empty"))
        interval.ordinal > 0 || throw(ArgumentError("stage ordinal must be positive"))
        0 <= interval.begin_qpc < interval.end_qpc || throw(ArgumentError(
            "expected stage interval must have positive raw-QPC duration",
        ))
        interval.role in (:p_sparse, :e_background) || throw(ArgumentError(
            "stage role must be :p_sparse or :e_background",
        ))
        isempty(interval.owner_threads) && throw(ArgumentError(
            "stage owner threads must be fixed before ETW extraction",
        ))
        owner_ids = UInt32[owner.thread_id for owner in interval.owner_threads]
        length(unique(owner_ids)) == length(owner_ids) ||
            throw(ArgumentError("duplicate stage owner thread ID"))
        manifest_role_owner_ids = sort!(UInt32[
            thread_id for (thread_id, registration) in role_manifest
            if registration.role === interval.role
        ])
        sort(copy(owner_ids)) == manifest_role_owner_ids || throw(ArgumentError(
            "stage owner set must equal every pre-bound manifest thread for that role",
        ))
        for owner in interval.owner_threads
            owner.thread_id > 0 || throw(ArgumentError("owner thread ID must be nonzero"))
            0 <= owner.role_registration_qpc <= interval.begin_qpc || throw(ArgumentError(
                "owner thread role was not registered before the stage",
            ))
            haskey(role_manifest, owner.thread_id) || throw(ArgumentError(
                "expected stage owner is absent from the pre-bound role manifest",
            ))
            registration = role_manifest[owner.thread_id]
            registration.role === interval.role || throw(ArgumentError(
                "expected stage role differs from the pre-bound role manifest",
            ))
            registration.role_registration_qpc == owner.role_registration_qpc ||
                throw(ArgumentError("expected owner registration QPC differs from manifest"))
            registration.allowed_cpu_set_ids == sort(copy(interval.allowed_cpu_set_ids)) ||
                throw(ArgumentError("expected allowed CPU Sets differ from role manifest"))
        end
        isempty(interval.allowed_cpu_set_ids) && throw(ArgumentError(
            "stage allowed_cpu_set_ids must not be empty",
        ))
        length(unique(interval.allowed_cpu_set_ids)) == length(interval.allowed_cpu_set_ids) ||
            throw(ArgumentError("duplicate allowed CPU Set ID"))
        required_class = interval.role === :p_sparse ? high_class : low_class
        for id in interval.allowed_cpu_set_ids
            haskey(by_id, id) || throw(ArgumentError("allowed CPU Set ID is absent from topology"))
            record = by_id[id]
            UInt8(_record_property(record, :efficiency_class)) == required_class ||
                throw(ArgumentError("allowed CPU Set ID is in the wrong intrinsic class"))
        end
        key = _stage_key(interval.cell_id, interval.repetition, interval.sequence,
                         interval.set_id, interval.stage, interval.ordinal)
        haskey(expected, key) && throw(ArgumentError("duplicate expected stage key"))
        expected[key] = interval
    end
    return expected
end

function _validate_etw_header(document, qpc_frequency::Int64, expected)
    _string(_required(document, "schema"), "schema") ==
        "heterogeneous-265k-etw-residency-evidence-v1" ||
        throw(ArgumentError("ETW evidence schema mismatch"))
    _string(_required(document, "producer_status"), "producer_status") ==
        "CAPTURED_PENDING_VALIDATION" || throw(ArgumentError(
            "ETW producer status is not CAPTURED_PENDING_VALIDATION",
        ))
    !_boolean(_required(document, "adoption_allowed"), "adoption_allowed") ||
        throw(ArgumentError("an evidence producer may not authorize adoption"))

    provider = _dictionary(_required(document, "provider"), "provider")
    _string(_required(provider, "name"), "provider.name") == ETW_PROVIDER_NAME ||
        throw(ArgumentError("unexpected ETW kernel provider"))
    lowercase(_string(_required(provider, "guid"), "provider.guid")) == ETW_PROVIDER_GUID ||
        throw(ArgumentError("unexpected ETW kernel provider GUID"))
    _string(_required(provider, "capture_tool"), "provider.capture_tool") == ETW_CAPTURE_TOOL ||
        throw(ArgumentError("ETW capture tool must be Microsoft WPT xperf.exe"))
    _hash(_required(provider, "capture_tool_sha256"), "provider.capture_tool_sha256")
    _string(_required(provider, "capture_tool_version"), "provider.capture_tool_version")
    _hash(_required(provider, "extractor_sha256"), "provider.extractor_sha256")
    _string(_required(provider, "extractor_version"), "provider.extractor_version")
    _string(_required(provider, "extractor_capability"), "provider.extractor_capability") ==
        ETW_EXTRACTOR_CAPABILITY || throw(ArgumentError(
            "extractor cannot prove raw-QPC CSwitch boundary reconstruction",
        ))
    _string(_required(provider, "trace_clock"), "provider.trace_clock") == "QPC" ||
        throw(ArgumentError("ETW trace clock must be QPC"))
    keywords = Set(String.(_required(provider, "kernel_keywords")))
    all(keyword -> keyword in keywords, ("PROC_THREAD", "CSWITCH")) ||
        throw(ArgumentError("ETW capture lacks PROC_THREAD or CSWITCH"))

    trace = _dictionary(_required(document, "trace"), "trace")
    _hash(_required(trace, "etl_sha256"), "trace.etl_sha256")
    _hash(_required(trace, "decoded_events_sha256"), "trace.decoded_events_sha256")
    _string(_required(trace, "os_boot_identity"), "trace.os_boot_identity")
    _string(_required(trace, "etw_session_identity"), "trace.etw_session_identity")
    _positive_tick(_required(trace, "qpc_frequency"), "trace.qpc_frequency") == qpc_frequency ||
        throw(ArgumentError("ETW/runner QPC frequency mismatch"))
    trace_begin = _nonnegative_tick(_required(trace, "begin_qpc"), "trace.begin_qpc")
    trace_end = _positive_tick(_required(trace, "end_qpc"), "trace.end_qpc")
    trace_begin < trace_end || throw(ArgumentError("ETW trace interval is empty"))
    for field in (
        "events_lost",
        "buffers_lost",
        "realtime_buffers_lost",
        "decode_failure_count",
        "unknown_cswitch_version_count",
    )
        _integer(_required(trace, field), "trace.$field") == 0 || throw(ArgumentError(
            "ETW $field is nonzero",
        ))
    end
    _integer(_required(trace, "cswitch_event_count"), "trace.cswitch_event_count") > 0 ||
        throw(ArgumentError("ETW trace has no CSwitch events"))
    _integer(_required(trace, "process_thread_event_count"),
             "trace.process_thread_event_count") > 0 ||
        throw(ArgumentError("ETW trace has no process/thread events"))
    for field in (
        "boundary_state_reconstructed",
        "all_target_thread_lifetimes_resolved",
        "all_scheduled_benchmark_threads_classified",
        "processor_group_mapping_complete",
    )
        _boolean(_required(trace, field), "trace.$field") || throw(ArgumentError(
            "ETW trace does not prove $field",
        ))
    end
    _string(_required(trace, "interval_semantics"), "trace.interval_semantics") ==
        "HALF_OPEN_QPC_WITH_ETW_RECORD_ORDER_TIEBREAK" || throw(ArgumentError(
            "ETW extractor interval semantics are not the audited half-open rule",
        ))
    !_boolean(_required(trace, "circular_overwrite_detected"),
              "trace.circular_overwrite_detected") || throw(ArgumentError(
        "ETW trace used an overwritten circular region",
    ))
    minimum_stage_qpc = minimum(value.begin_qpc for value in values(expected))
    maximum_stage_qpc = maximum(value.end_qpc for value in values(expected))
    trace_begin <= minimum_stage_qpc && maximum_stage_qpc <= trace_end ||
        throw(ArgumentError("ETW trace does not enclose every expected stage"))
    return nothing
end

function _stage_entry_key(entry)
    key = _dictionary(_required(entry, "key"), "stage.key")
    return _stage_key(
        _string(_required(key, "cell_id"), "stage.key.cell_id"),
        Int(_integer(_required(key, "repetition"), "stage.key.repetition")),
        Int(_integer(_required(key, "sequence"), "stage.key.sequence")),
        _string(_required(key, "set_id"), "stage.key.set_id"),
        _string(_required(key, "stage"), "stage.key.stage"),
        Int(_integer(_required(key, "ordinal"), "stage.key.ordinal")),
    )
end

function _sorted_u32(values, label)
    parsed = UInt32[_u32(value, label) for value in values]
    length(unique(parsed)) == length(parsed) || throw(ArgumentError("$label has duplicates"))
    return sort!(parsed)
end

function _validate_residency_stage(entry, expected::ExpectedStageInterval,
                                   target_process_id::UInt32, by_id, by_processor)
    _nonnegative_tick(_required(entry, "begin_qpc"), "stage.begin_qpc") ==
        expected.begin_qpc || throw(ArgumentError("stage begin QPC mismatch"))
    _positive_tick(_required(entry, "end_qpc"), "stage.end_qpc") ==
        expected.end_qpc || throw(ArgumentError("stage end QPC mismatch"))
    _string(_required(entry, "role"), "stage.role") == String(expected.role) ||
        throw(ArgumentError("stage role mismatch"))
    _string(_required(entry, "coverage_method"), "stage.coverage_method") ==
        "ALL_CSWITCH_SCHEDULED_SLICES_CLIPPED_WITH_BOUNDARY_STATE" ||
        throw(ArgumentError("ETW stage is not a whole-stage CSwitch reconstruction"))
    _boolean(_required(entry, "all_owner_slices_present"),
             "stage.all_owner_slices_present") || throw(ArgumentError(
        "ETW stage omits owner slices",
    ))
    raw_owners = _required(entry, "owner_threads")
    raw_owners isa AbstractVector || throw(ArgumentError("stage.owner_threads must be an array"))
    observed_owner_starts = Dict{UInt32,Int64}()
    for raw_owner in raw_owners
        owner = _dictionary(raw_owner, "stage.owner_threads entry")
        thread_id = _u32(_required(owner, "thread_id"), "owner.thread_id")
        thread_start_qpc = _nonnegative_tick(
            _required(owner, "thread_start_qpc"), "owner.thread_start_qpc",
        )
        haskey(observed_owner_starts, thread_id) && throw(ArgumentError(
            "stage.owner_threads has duplicate IDs",
        ))
        observed_owner_starts[thread_id] = thread_start_qpc
    end
    expected_owner_registrations = Dict(
        owner.thread_id => owner.role_registration_qpc for owner in expected.owner_threads
    )
    Set(keys(observed_owner_starts)) == Set(keys(expected_owner_registrations)) ||
        throw(ArgumentError("ETW owner thread ID manifest mismatch"))
    for (thread_id, thread_start_qpc) in observed_owner_starts
        thread_start_qpc <= expected_owner_registrations[thread_id] <= expected.begin_qpc ||
            throw(ArgumentError("ETW thread lifetime does not contain role registration/stage"))
    end
    observed_owners = sort!(collect(keys(observed_owner_starts)))
    observed_allowed = _sorted_u32(_required(entry, "allowed_cpu_set_ids"),
                                    "stage.allowed_cpu_set_ids")
    observed_allowed == sort(copy(expected.allowed_cpu_set_ids)) ||
        throw(ArgumentError("ETW allowed CPU Set manifest mismatch"))
    allowed = Set(observed_allowed)
    owners = Set(observed_owners)

    for field in (
        "unattributed_scheduled_ticks",
        "disallowed_scheduled_ticks",
        "same_thread_overlap_ticks",
    )
        _decimal_bigint(_required(entry, field), "stage.$field") == 0 ||
            throw(ArgumentError("ETW stage reports nonzero $field"))
    end

    slices = _required(entry, "scheduled_slices")
    slices isa AbstractVector || throw(ArgumentError("scheduled_slices must be an array"))
    isempty(slices) && throw(ArgumentError("ETW stage has no scheduled slice"))
    intervals_by_thread = Dict{UInt32,Vector{Tuple{Int64,Int64}}}(
        thread_id => Tuple{Int64,Int64}[] for thread_id in observed_owners
    )
    total_ticks = BigInt(0)
    for (index, raw_slice) in enumerate(slices)
        slice = _dictionary(raw_slice, "scheduled_slices[$index]")
        _u32(_required(slice, "process_id"), "slice.process_id") == target_process_id ||
            throw(ArgumentError("scheduled slice belongs to another process"))
        thread_id = _u32(_required(slice, "thread_id"), "slice.thread_id")
        thread_id in owners || throw(ArgumentError("scheduled slice has an undeclared owner"))
        _nonnegative_tick(_required(slice, "thread_start_qpc"),
                          "slice.thread_start_qpc") ==
            observed_owner_starts[thread_id] || throw(ArgumentError(
                "scheduled slice thread lifetime does not match the role manifest",
            ))
        begin_qpc = _nonnegative_tick(_required(slice, "begin_qpc"), "slice.begin_qpc")
        end_qpc = _positive_tick(_required(slice, "end_qpc"), "slice.end_qpc")
        expected.begin_qpc <= begin_qpc < end_qpc <= expected.end_qpc ||
            throw(ArgumentError("scheduled slice is not clipped to the stage interval"))
        group_value = _u32(_required(slice, "processor_group"), "slice.processor_group")
        group_value <= typemax(UInt16) || throw(ArgumentError("processor group exceeds UInt16"))
        logical_value = _u32(_required(slice, "logical_processor_index"),
                             "slice.logical_processor_index")
        logical_value <= typemax(UInt8) || throw(ArgumentError(
            "logical processor index exceeds UInt8",
        ))
        processor_key = (UInt16(group_value), UInt8(logical_value))
        haskey(by_processor, processor_key) || throw(ArgumentError(
            "ETW processor is absent from bound CPU Set topology",
        ))
        topology_record = by_processor[processor_key]
        cpu_set_id = _u32(_required(slice, "cpu_set_id"), "slice.cpu_set_id")
        cpu_set_id == UInt32(_record_property(topology_record, :id)) ||
            throw(ArgumentError("ETW processor/CPU Set mapping mismatch"))
        haskey(by_id, cpu_set_id) || throw(ArgumentError("unknown CPU Set ID"))
        cpu_set_id in allowed || throw(ArgumentError("scheduled work left its role CPU Sets"))
        _u32(_required(slice, "core_index"), "slice.core_index") ==
            UInt32(_record_property(topology_record, :core_index)) ||
            throw(ArgumentError("ETW core index/topology mismatch"))
        _u32(_required(slice, "efficiency_class"), "slice.efficiency_class") ==
            UInt32(_record_property(topology_record, :efficiency_class)) ||
            throw(ArgumentError("ETW efficiency class/topology mismatch"))
        push!(intervals_by_thread[thread_id], (begin_qpc, end_qpc))
        total_ticks += end_qpc - begin_qpc
    end
    Int(_integer(_required(entry, "slice_count"), "stage.slice_count")) == length(slices) ||
        throw(ArgumentError("stage slice_count mismatch"))
    _decimal_bigint(_required(entry, "scheduled_ticks"), "stage.scheduled_ticks") ==
        total_ticks || throw(ArgumentError("stage scheduled_ticks mismatch"))
    total_ticks > 0 || throw(ArgumentError("stage has zero scheduled CPU time"))
    for (thread_id, intervals) in intervals_by_thread
        isempty(intervals) && throw(ArgumentError("owner thread $thread_id has no scheduled slice"))
        sort!(intervals)
        for index in 2:length(intervals)
            intervals[index - 1][2] <= intervals[index][1] || throw(ArgumentError(
                "same ETW thread has overlapping reconstructed slices",
            ))
        end
    end
    return total_ticks
end

"""Validate independent ETW whole-stage residency evidence.

The validator accepts no point-sampling substitute. It requires loss-free
PROC_THREAD+CSWITCH capture, raw-QPC boundary-state reconstruction, exact stage
keys from the immutable runner artifact, and processor-group/logical-index to
CPU-Set mapping through the bound topology.
"""
function validate_etw_residency_evidence(document, bindings::EvidenceBindings,
                                         topology_records,
                                         process_identity::ExpectedProcessIdentity,
                                         role_manifest_bytes::AbstractVector{UInt8},
                                         expected_stages::AbstractVector{ExpectedStageInterval},
                                         qpc_frequency::Integer)
    qpc = _positive_tick(qpc_frequency, "qpc_frequency")
    _validate_bindings(document, bindings)
    _validate_topology_digest(topology_records, bindings)
    by_id, by_processor, low_class, high_class = _topology_maps(topology_records)
    role_manifest = _validate_role_manifest(
        role_manifest_bytes, bindings, process_identity, qpc,
    )
    expected = _validated_expected_stages(
        expected_stages, by_id, low_class, high_class, role_manifest,
    )
    _validate_etw_header(document, qpc, expected)
    process_identity.process_id > 0 || throw(ArgumentError("process ID must be nonzero"))
    process_identity.runner_identity_qpc >= 0 || throw(ArgumentError(
        "runner identity QPC must be nonnegative",
    ))
    isempty(process_identity.run_nonce) && throw(ArgumentError("run nonce must not be empty"))
    _hash(process_identity.role_manifest_sha256, "role_manifest_sha256") ==
        process_identity.role_manifest_sha256 || throw(ArgumentError(
            "role manifest digest is not canonical",
        ))
    _hash(process_identity.assignment_readback_sha256, "assignment_readback_sha256") ==
        process_identity.assignment_readback_sha256 || throw(ArgumentError(
            "assignment/readback digest is not canonical",
        ))
    process_document = _dictionary(_required(document, "target_process"), "target_process")
    _u32(_required(process_document, "process_id"), "target_process.process_id") ==
        process_identity.process_id || throw(ArgumentError("ETW process ID mismatch"))
    process_start_qpc = _nonnegative_tick(
        _required(process_document, "process_start_qpc"), "target_process.process_start_qpc",
    )
    trace_document = _dictionary(_required(document, "trace"), "trace")
    _nonnegative_tick(_required(trace_document, "begin_qpc"), "trace.begin_qpc") <=
        process_start_qpc || throw(ArgumentError(
            "ETW trace did not start before the benchmark process",
        ))
    process_start_qpc <= process_identity.runner_identity_qpc || throw(ArgumentError(
        "ETW process lifetime begins after the runner identity witness",
    ))
    _nonnegative_tick(_required(process_document, "runner_identity_qpc"),
                      "target_process.runner_identity_qpc") ==
        process_identity.runner_identity_qpc || throw(ArgumentError(
            "ETW runner identity QPC mismatch",
        ))
    _string(_required(process_document, "run_nonce"), "target_process.run_nonce") ==
        process_identity.run_nonce || throw(ArgumentError("ETW run nonce mismatch"))
    _hash(_required(process_document, "role_manifest_sha256"),
          "target_process.role_manifest_sha256") == process_identity.role_manifest_sha256 ||
        throw(ArgumentError("ETW role manifest digest mismatch"))
    _hash(_required(process_document, "assignment_readback_sha256"),
          "target_process.assignment_readback_sha256") ==
        process_identity.assignment_readback_sha256 || throw(ArgumentError(
            "ETW assignment/readback digest mismatch",
        ))
    entries = _required(document, "stage_residency")
    entries isa AbstractVector || throw(ArgumentError("stage_residency must be an array"))
    length(entries) == length(expected) || throw(ArgumentError(
        "ETW artifact must contain exactly the expected stage set",
    ))
    observed = Set{Tuple{String,Int,Int,String,String,Int}}()
    total_scheduled_ticks = BigInt(0)
    for raw_entry in entries
        entry = _dictionary(raw_entry, "stage_residency entry")
        key = _stage_entry_key(entry)
        haskey(expected, key) || throw(ArgumentError("unexpected ETW stage key"))
        key in observed && throw(ArgumentError("duplicate ETW stage key"))
        push!(observed, key)
        total_scheduled_ticks += _validate_residency_stage(
            entry, expected[key], process_identity.process_id, by_id, by_processor,
        )
    end
    observed == Set(keys(expected)) || throw(ArgumentError("ETW stage set is incomplete"))
    return (
        status="VALIDATED_ETW_RESIDENCY",
        adoption_allowed=false,
        stage_count=length(entries),
        scheduled_ticks=string(total_scheduled_ticks),
        note="ETW evidence is one mandatory gate and cannot authorize adoption alone",
    )
end

function _validate_imc_provenance(provenance::ExpectedIMCProvenance)
    for (name, value) in (
        "availability_probe_sha256" => provenance.availability_probe_sha256,
        "pcm_tool_sha256" => provenance.pcm_tool_sha256,
        "pcm_driver_sha256" => provenance.pcm_driver_sha256,
        "raw_adapter_sha256" => provenance.raw_adapter_sha256,
        "role_manifest_sha256" => provenance.role_manifest_sha256,
        "timed_region_hook_source_sha256" => provenance.timed_region_hook_source_sha256,
        "timed_region_witness_sha256" => provenance.timed_region_witness_sha256,
        "before_snapshot_raw_sha256" => provenance.before_snapshot_raw_sha256,
        "after_snapshot_raw_sha256" => provenance.after_snapshot_raw_sha256,
        "counter_inventory_sha256" => provenance.counter_inventory_sha256,
        "counter_configuration_sha256" => provenance.counter_configuration_sha256,
    )
        _hash(value, name) == value || throw(ArgumentError("$name is not canonical"))
    end
    return provenance
end

function _validate_imc_header(document, bindings::EvidenceBindings,
                              provenance::ExpectedIMCProvenance,
                              qpc_frequency::Int64, expected_begin_qpc::Int64,
                              expected_end_qpc::Int64)
    _validate_bindings(document, bindings)
    _validate_imc_provenance(provenance)
    _string(_required(document, "schema"), "schema") ==
        "heterogeneous-265k-imc-ram-evidence-v1" ||
        throw(ArgumentError("IMC evidence schema mismatch"))
    _string(_required(document, "producer_status"), "producer_status") ==
        "CAPTURED_PENDING_VALIDATION" || throw(ArgumentError(
            "IMC producer status is not CAPTURED_PENDING_VALIDATION",
        ))
    !_boolean(_required(document, "adoption_allowed"), "adoption_allowed") ||
        throw(ArgumentError("an evidence producer may not authorize adoption"))

    provider = _dictionary(_required(document, "provider"), "provider")
    _string(_required(provider, "name"), "provider.name") == IMC_PROVIDER_NAME ||
        throw(ArgumentError("unexpected IMC provider"))
    _string(_required(provider, "tool"), "provider.tool") == IMC_CAPTURE_TOOL ||
        throw(ArgumentError("IMC tool must be Intel PCM pcm-memory.exe"))
    _string(_required(provider, "project"), "provider.project") == IMC_PROJECT ||
        throw(ArgumentError("unexpected Intel PCM project identity"))
    _hash(_required(provider, "availability_probe_sha256"),
          "provider.availability_probe_sha256") == provenance.availability_probe_sha256 ||
        throw(ArgumentError("IMC availability probe digest mismatch"))
    _string(_required(provider, "availability_probe_status"),
            "provider.availability_probe_status") ==
        "AVAILABLE_PENDING_EXCLUSIVE_CAPTURE" || throw(ArgumentError(
            "IMC availability probe did not pass",
        ))
    _hash(_required(provider, "tool_sha256"), "provider.tool_sha256") ==
        provenance.pcm_tool_sha256 || throw(ArgumentError("PCM tool digest mismatch"))
    _string(_required(provider, "tool_version"), "provider.tool_version")
    _hash(_required(provider, "driver_sha256"), "provider.driver_sha256") ==
        provenance.pcm_driver_sha256 || throw(ArgumentError("PCM driver digest mismatch"))
    _string(_required(provider, "driver_version"), "provider.driver_version")
    _hash(_required(provider, "raw_adapter_sha256"), "provider.raw_adapter_sha256") ==
        provenance.raw_adapter_sha256 || throw(ArgumentError("PCM adapter digest mismatch"))
    _string(_required(provider, "raw_adapter_version"), "provider.raw_adapter_version")
    _string(_required(provider, "raw_adapter_capability"),
            "provider.raw_adapter_capability") == IMC_ADAPTER_CAPABILITY ||
        throw(ArgumentError("Intel PCM adapter lacks exact raw-counter capability"))
    _boolean(_required(provider, "driver_loaded"), "provider.driver_loaded") ||
        throw(ArgumentError("Intel PCM Windows driver was not loaded"))
    _boolean(_required(provider, "adapter_probe_passed"),
             "provider.adapter_probe_passed") || throw(ArgumentError(
        "Intel PCM raw adapter availability was not proven",
    ))

    capture = _dictionary(_required(document, "capture"), "capture")
    for (name, expected_value) in (
        "role_manifest_sha256" => provenance.role_manifest_sha256,
        "timed_region_hook_source_sha256" => provenance.timed_region_hook_source_sha256,
        "timed_region_witness_sha256" => provenance.timed_region_witness_sha256,
        "before_snapshot_raw_sha256" => provenance.before_snapshot_raw_sha256,
        "after_snapshot_raw_sha256" => provenance.after_snapshot_raw_sha256,
        "counter_inventory_sha256" => provenance.counter_inventory_sha256,
        "counter_configuration_sha256" => provenance.counter_configuration_sha256,
    )
        _hash(_required(capture, name), "capture.$name") == expected_value ||
            throw(ArgumentError("capture.$name mismatch"))
    end
    _integer(_required(capture, "program_generation"),
             "capture.program_generation") == provenance.program_generation ||
        throw(ArgumentError("IMC program generation mismatch"))
    _string(_required(capture, "scope"), "capture.scope") ==
        "SYSTEM_WIDE_EXCLUSIVE_WINDOW" || throw(ArgumentError(
            "IMC evidence is not an isolated system-wide window",
        ))
    _string(_required(capture, "attribution"), "capture.attribution") ==
        "EXCLUSIVE_HOST_WINDOW_ONLY_NOT_PROCESS_COUNTER" || throw(ArgumentError(
            "IMC attribution must state its system-wide exclusive-window scope",
        ))
    for field in (
        "exclusive_window_verified",
        "all_populated_channels_covered",
        "counter_programming_continuous",
    )
        _boolean(_required(capture, field), "capture.$field") || throw(ArgumentError(
            "IMC capture does not prove $field",
        ))
    end
    _boolean(_required(capture, "interfering_activity_detected"),
             "capture.interfering_activity_detected") && throw(ArgumentError(
        "system-wide IMC counters are unattributable with interfering activity",
    ))
    for field in (
        "unattributed_counter_count",
        "lost_sample_count",
        "counter_reset_count",
    )
        _integer(_required(capture, field), "capture.$field") == 0 ||
            throw(ArgumentError("IMC capture reports nonzero $field"))
    end
    _positive_tick(_required(capture, "qpc_frequency"), "capture.qpc_frequency") ==
        qpc_frequency || throw(ArgumentError("IMC/runner QPC frequency mismatch"))
    _nonnegative_tick(_required(capture, "timed_begin_qpc"), "capture.timed_begin_qpc") ==
        expected_begin_qpc || throw(ArgumentError("IMC begin QPC mismatch"))
    _positive_tick(_required(capture, "timed_end_qpc"), "capture.timed_end_qpc") ==
        expected_end_qpc || throw(ArgumentError("IMC end QPC mismatch"))
    before_begin = _nonnegative_tick(
        _required(capture, "before_snapshot_qpc_begin"),
        "capture.before_snapshot_qpc_begin",
    )
    before_end = _nonnegative_tick(
        _required(capture, "before_snapshot_qpc_end"),
        "capture.before_snapshot_qpc_end",
    )
    after_begin = _positive_tick(
        _required(capture, "after_snapshot_qpc_begin"),
        "capture.after_snapshot_qpc_begin",
    )
    after_end = _positive_tick(
        _required(capture, "after_snapshot_qpc_end"),
        "capture.after_snapshot_qpc_end",
    )
    counter_begin = _nonnegative_tick(
        _required(capture, "counter_latch_begin_qpc"),
        "capture.counter_latch_begin_qpc",
    )
    counter_end = _positive_tick(
        _required(capture, "counter_latch_end_qpc"),
        "capture.counter_latch_end_qpc",
    )
    before_begin <= counter_begin <= before_end <= expected_begin_qpc ||
        throw(ArgumentError("pre-counter snapshot does not precede timed work"))
    expected_end_qpc <= after_begin <= counter_end <= after_end ||
        throw(ArgumentError("post-counter snapshot does not follow timed work"))
    counter_begin < counter_end || throw(ArgumentError("IMC counter window is empty"))
    _boolean(_required(capture, "process_io_bytes_used"),
             "capture.process_io_bytes_used") && throw(ArgumentError(
        "process I/O bytes may not substitute for IMC traffic",
    ))
    _boolean(_required(capture, "theoretical_bandwidth_used"),
             "capture.theoretical_bandwidth_used") && throw(ArgumentError(
        "theoretical DIMM bandwidth may not substitute for IMC traffic",
    ))
    _boolean(_required(capture, "uncore_counter_bytes_used"),
             "capture.uncore_counter_bytes_used") || throw(ArgumentError(
        "IMC evidence did not derive bytes from uncore counters",
    ))
    !_boolean(_required(capture, "process_attribution_claimed"),
              "capture.process_attribution_claimed") || throw(ArgumentError(
        "system-wide IMC counters may not be claimed as process-attributed",
    ))
    !_boolean(_required(capture, "stock_pcm_csv_used"),
              "capture.stock_pcm_csv_used") || throw(ArgumentError(
        "stock pcm-memory CSV cannot satisfy the raw-counter gate",
    ))
    return capture, counter_begin, counter_end
end

function _counter_value_in_width(value::BigInt, width::Int, label::AbstractString)
    1 <= width <= 64 || throw(ArgumentError("$label counter width must be 1:64"))
    value < (BigInt(1) << width) || throw(ArgumentError("$label exceeds counter width"))
    return value
end

function _validate_imc_channel(raw_channel, duration_ticks::Int64)
    channel = _dictionary(raw_channel, "IMC channel")
    socket_id = Int(_integer(_required(channel, "socket_id"), "channel.socket_id"))
    controller_id = Int(_integer(_required(channel, "controller_id"),
                                 "channel.controller_id"))
    channel_id = Int(_integer(_required(channel, "channel_id"), "channel.channel_id"))
    minimum((socket_id, controller_id, channel_id)) >= 0 || throw(ArgumentError(
        "IMC channel coordinates must be nonnegative",
    ))
    _string(_required(channel, "read_semantic"), "channel.read_semantic") ==
        "DRAM_READ_CAS" || throw(ArgumentError("unknown IMC read counter semantic"))
    _string(_required(channel, "write_semantic"), "channel.write_semantic") ==
        "DRAM_WRITE_CAS" || throw(ArgumentError("unknown IMC write counter semantic"))
    width = Int(_integer(_required(channel, "counter_width_bits"),
                         "channel.counter_width_bits"))
    Int(_integer(_required(channel, "bytes_per_count"), "channel.bytes_per_count")) == 64 ||
        throw(ArgumentError("Intel IMC CAS counters must scale by 64 bytes"))
    for field in ("overflow_detected", "multiplexed", "counter_reset_detected")
        _boolean(_required(channel, field), "channel.$field") && throw(ArgumentError(
            "IMC channel reports $field",
        ))
    end
    _string(_required(channel, "timebase"), "channel.timebase") == "QPC" ||
        throw(ArgumentError("IMC enabled/running time must use QPC ticks"))
    enabled = _decimal_bigint(_required(channel, "time_enabled_ticks"),
                              "channel.time_enabled_ticks")
    running = _decimal_bigint(_required(channel, "time_running_ticks"),
                              "channel.time_running_ticks")
    enabled == running == duration_ticks || throw(ArgumentError(
        "IMC counter was multiplexed, paused, or measured over another window",
    ))

    read_before = _counter_value_in_width(
        _decimal_bigint(_required(channel, "read_before"), "channel.read_before"),
        width, "read_before",
    )
    read_after = _counter_value_in_width(
        _decimal_bigint(_required(channel, "read_after"), "channel.read_after"),
        width, "read_after",
    )
    write_before = _counter_value_in_width(
        _decimal_bigint(_required(channel, "write_before"), "channel.write_before"),
        width, "write_before",
    )
    write_after = _counter_value_in_width(
        _decimal_bigint(_required(channel, "write_after"), "channel.write_after"),
        width, "write_after",
    )
    read_after >= read_before || throw(ArgumentError(
        "IMC read counter wrapped; modulo inference is forbidden",
    ))
    write_after >= write_before || throw(ArgumentError(
        "IMC write counter wrapped; modulo inference is forbidden",
    ))
    read_delta = read_after - read_before
    write_delta = write_after - write_before
    _decimal_bigint(_required(channel, "read_delta_counts"),
                    "channel.read_delta_counts") == read_delta ||
        throw(ArgumentError("IMC read delta mismatch"))
    _decimal_bigint(_required(channel, "write_delta_counts"),
                    "channel.write_delta_counts") == write_delta ||
        throw(ArgumentError("IMC write delta mismatch"))
    return (socket_id, controller_id, channel_id), read_delta * 64, write_delta * 64
end

function _rounded_bytes_per_second(bytes::BigInt, frequency::Int64, duration::Int64)
    numerator = bytes * frequency
    return (numerator + (duration >> 1)) ÷ duration
end

"""Validate exact Intel PCM IMC read/write evidence for an H0/H1 timed window.

Stock `pcm-memory.exe` CSV bandwidth is deliberately insufficient: a passing
artifact must come from a probed adapter exposing raw before/after counts,
width, continuous enabled/running time, reset, overflow, and multiplex status
for every populated IMC channel. Bytes and bandwidth are recomputed here.
"""
function validate_imc_ram_evidence(document, bindings::EvidenceBindings,
                                   provenance::ExpectedIMCProvenance,
                                   expected_cell_id::AbstractString,
                                   expected_repetition::Integer,
                                   expected_run_nonce::AbstractString,
                                   qpc_frequency::Integer,
                                   expected_begin_qpc::Integer,
                                   expected_end_qpc::Integer)
    qpc = _positive_tick(qpc_frequency, "qpc_frequency")
    String(expected_cell_id) in _SUPPORTED_CELLS || throw(ArgumentError(
        "IMC evidence is restricted to post-idle H0/H1",
    ))
    expected_repetition > 0 || throw(ArgumentError("repetition must be positive"))
    isempty(expected_run_nonce) && throw(ArgumentError("run nonce must not be empty"))
    _string(_required(document, "matrix_cell_id"), "matrix_cell_id") ==
        expected_cell_id || throw(ArgumentError("IMC matrix cell mismatch"))
    Int(_integer(_required(document, "repetition"), "repetition")) ==
        expected_repetition || throw(ArgumentError("IMC repetition mismatch"))
    _string(_required(document, "run_nonce"), "run_nonce") == expected_run_nonce ||
        throw(ArgumentError("IMC run nonce mismatch"))
    begin_qpc = _nonnegative_tick(expected_begin_qpc, "expected_begin_qpc")
    end_qpc = _positive_tick(expected_end_qpc, "expected_end_qpc")
    begin_qpc < end_qpc || throw(ArgumentError("expected IMC interval is empty"))
    capture, counter_begin, counter_end = _validate_imc_header(
        document, bindings, provenance, qpc, begin_qpc, end_qpc,
    )
    duration = counter_end - counter_begin
    timed_duration = end_qpc - begin_qpc
    guard_ticks = (begin_qpc - counter_begin) + (counter_end - end_qpc)
    guard_ticks >= 0 || error("IMC counter guard computation underflowed")
    Int128(guard_ticks) * 1000 <= timed_duration || throw(ArgumentError(
        "IMC counter snapshot guard exceeds 0.1 percent of the timed region",
    ))
    channels = _required(document, "channels")
    channels isa AbstractVector || throw(ArgumentError("channels must be an array"))
    isempty(channels) && throw(ArgumentError("IMC artifact has no channel counters"))
    expected_count = Int(_integer(_required(capture, "enumerated_populated_channel_count"),
                                  "capture.enumerated_populated_channel_count"))
    expected_count == length(channels) || throw(ArgumentError(
        "IMC artifact does not cover every enumerated populated channel",
    ))
    keys = Set{Tuple{Int,Int,Int}}()
    read_bytes = BigInt(0)
    write_bytes = BigInt(0)
    for channel in channels
        key, channel_read, channel_write = _validate_imc_channel(channel, duration)
        key in keys && throw(ArgumentError("duplicate IMC channel counter"))
        push!(keys, key)
        read_bytes += channel_read
        write_bytes += channel_write
    end

    aggregate = _dictionary(_required(document, "aggregate"), "aggregate")
    total_bytes = read_bytes + write_bytes
    _decimal_bigint(_required(aggregate, "read_bytes"), "aggregate.read_bytes") ==
        read_bytes || throw(ArgumentError("aggregate IMC read bytes mismatch"))
    _decimal_bigint(_required(aggregate, "write_bytes"), "aggregate.write_bytes") ==
        write_bytes || throw(ArgumentError("aggregate IMC write bytes mismatch"))
    _decimal_bigint(_required(aggregate, "total_bytes"), "aggregate.total_bytes") ==
        total_bytes || throw(ArgumentError("aggregate IMC total bytes mismatch"))
    _decimal_bigint(_required(aggregate, "duration_qpc_ticks"),
                    "aggregate.duration_qpc_ticks") == duration ||
        throw(ArgumentError("aggregate IMC duration mismatch"))
    _decimal_bigint(_required(aggregate, "timed_duration_qpc_ticks"),
                    "aggregate.timed_duration_qpc_ticks") == timed_duration ||
        throw(ArgumentError("aggregate timed duration mismatch"))
    _decimal_bigint(_required(aggregate, "counter_guard_qpc_ticks"),
                    "aggregate.counter_guard_qpc_ticks") == guard_ticks ||
        throw(ArgumentError("aggregate counter guard mismatch"))
    read_bps = _rounded_bytes_per_second(read_bytes, qpc, duration)
    write_bps = _rounded_bytes_per_second(write_bytes, qpc, duration)
    total_bps = _rounded_bytes_per_second(total_bytes, qpc, duration)
    _decimal_bigint(_required(aggregate, "read_bytes_per_second"),
                    "aggregate.read_bytes_per_second") == read_bps ||
        throw(ArgumentError("aggregate IMC read bandwidth mismatch"))
    _decimal_bigint(_required(aggregate, "write_bytes_per_second"),
                    "aggregate.write_bytes_per_second") == write_bps ||
        throw(ArgumentError("aggregate IMC write bandwidth mismatch"))
    _decimal_bigint(_required(aggregate, "total_bytes_per_second"),
                    "aggregate.total_bytes_per_second") == total_bps ||
        throw(ArgumentError("aggregate IMC total bandwidth mismatch"))
    _string(_required(aggregate, "rounding"), "aggregate.rounding") ==
        "nearest_integer_bytes_per_second_half_up" || throw(ArgumentError(
            "unknown IMC bandwidth rounding rule",
        ))
    return (
        status="VALIDATED_IMC_RAM_TRAFFIC",
        adoption_allowed=false,
        channel_count=length(channels),
        read_bytes=string(read_bytes),
        write_bytes=string(write_bytes),
        total_bytes=string(total_bytes),
        read_bytes_per_second=string(read_bps),
        write_bytes_per_second=string(write_bps),
        total_bytes_per_second=string(total_bps),
        counter_guard_qpc_ticks=string(guard_ticks),
        note="system-wide exclusive-window IMC evidence is not a process I/O counter",
    )
end

"""Create the only legal artifact when exact ETW/IMC tooling is unavailable."""
function unavailable_evidence_artifact(bindings::EvidenceBindings;
                                       matrix_cell_id::AbstractString,
                                       missing_capabilities,
                                       tool_inventory=Dict{String,Any}(),
                                       timestamp_utc::AbstractString)
    _validate_binding_values(bindings)
    String(matrix_cell_id) in _SUPPORTED_CELLS || throw(ArgumentError(
        "unavailable evidence artifact is restricted to H0/H1",
    ))
    capabilities = sort!(unique(String.(collect(missing_capabilities))))
    isempty(capabilities) && throw(ArgumentError("missing_capabilities must not be empty"))
    isempty(timestamp_utc) && throw(ArgumentError("timestamp_utc must not be empty"))
    return Dict{String,Any}(
        "schema" => "heterogeneous-265k-residency-ram-unavailable-v1",
        "implementation_status" => EVIDENCE_IMPLEMENTATION_STATUS,
        "status" => "UNAVAILABLE_FAIL_CLOSED",
        "decision" => "INCONCLUSIVE_FAIL_CLOSED",
        "adoption_allowed" => false,
        "matrix_cell_id" => String(matrix_cell_id),
        "missing_capabilities" => capabilities,
        "tool_inventory" => tool_inventory,
        "bindings" => Dict(name => value for (name, value) in _binding_pairs(bindings)),
        "substitutions_forbidden" => [
            "point CPU samples for ETW whole-stage CSwitch residency",
            "process I/O bytes for IMC DRAM traffic",
            "theoretical DIMM bandwidth for measured IMC bandwidth",
            "pcm-memory CSV without raw counter/width/overflow/multiplex evidence",
        ],
        "timestamp_utc" => String(timestamp_utc),
    )
end

function validate_unavailable_evidence_artifact(document, bindings::EvidenceBindings)
    _validate_bindings(document, bindings)
    _string(_required(document, "schema"), "schema") ==
        "heterogeneous-265k-residency-ram-unavailable-v1" ||
        throw(ArgumentError("unavailable evidence schema mismatch"))
    _string(_required(document, "implementation_status"), "implementation_status") ==
        EVIDENCE_IMPLEMENTATION_STATUS || throw(ArgumentError(
            "unavailable artifact implementation status mismatch",
        ))
    _string(_required(document, "status"), "status") == "UNAVAILABLE_FAIL_CLOSED" ||
        throw(ArgumentError("unavailable artifact status mismatch"))
    _string(_required(document, "decision"), "decision") ==
        "INCONCLUSIVE_FAIL_CLOSED" || throw(ArgumentError(
            "unavailable artifact decision mismatch",
        ))
    !_boolean(_required(document, "adoption_allowed"), "adoption_allowed") ||
        throw(ArgumentError("unavailable evidence may not authorize adoption"))
    missing = _required(document, "missing_capabilities")
    missing isa AbstractVector && !isempty(missing) || throw(ArgumentError(
        "unavailable artifact has no missing capability",
    ))
    return true
end

end # module WindowsResidencyRAMEvidence
