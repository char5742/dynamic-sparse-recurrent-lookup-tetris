using JSON3
using SHA

export WindowsCPUSetRecord,
       CoreUltra265KCPUSetClassification,
       enumerate_windows_cpu_sets,
       classify_core_ultra_7_265k_cpu_sets,
       cpu_set_ids_for_role,
       windows_cpu_set_topology_json,
       windows_cpu_set_topology_sha256,
       windows_cpu_set_topology_contract,
       read_process_default_cpu_sets,
       apply_process_default_cpu_sets!,
       clear_process_default_cpu_sets!,
       read_thread_selected_cpu_sets,
       apply_thread_selected_cpu_sets!,
       clear_thread_selected_cpu_sets!,
       current_windows_thread_witness

const _CPU_SET_INFORMATION_TYPE = UInt32(0)
const _CPU_SET_MINIMUM_RECORD_SIZE = UInt32(32)
const _ERROR_INSUFFICIENT_BUFFER = UInt32(122)

"""One `SYSTEM_CPU_SET_INFORMATION::CpuSet` record copied into Julia values.

CPU Set IDs are opaque identifiers. `logical_processor_index` and `core_index`
are relative to `group`; none of these fields may be inferred from another.
"""
Base.@kwdef struct WindowsCPUSetRecord
    record_size::UInt32 = _CPU_SET_MINIMUM_RECORD_SIZE
    id::UInt32
    group::UInt16
    logical_processor_index::UInt8
    core_index::UInt8
    last_level_cache_index::UInt8
    numa_node_index::UInt8
    efficiency_class::UInt8
    scheduling_class::UInt8
    all_flags::UInt8
    parked::Bool
    allocated::Bool
    allocated_to_target_process::Bool
    realtime::Bool
    reserved_flags::UInt8
    allocation_tag::UInt64
end

"""Fail-closed interpretation of the discovered CPU Sets for this exact host.

The names `performance` and `efficiency` are emitted only when discovery shows
two efficiency classes, twenty usable one-thread cores, and the target 8+12
cardinality. A failed classification authorizes no P/E-specific assignment.
"""
struct CoreUltra265KCPUSetClassification
    adopted::Bool
    status::String
    reason::String
    performance_efficiency_class::Union{Nothing,UInt8}
    efficiency_efficiency_class::Union{Nothing,UInt8}
    performance_cpu_set_ids::Vector{UInt32}
    efficiency_cpu_set_ids::Vector{UInt32}
end

_read_u16_le(bytes::Vector{UInt8}, offset::Int) =
    UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)

_read_u32_le(bytes::Vector{UInt8}, offset::Int) =
    UInt32(bytes[offset]) |
    (UInt32(bytes[offset + 1]) << 8) |
    (UInt32(bytes[offset + 2]) << 16) |
    (UInt32(bytes[offset + 3]) << 24)

_read_u64_le(bytes::Vector{UInt8}, offset::Int) =
    UInt64(_read_u32_le(bytes, offset)) |
    (UInt64(_read_u32_le(bytes, offset + 4)) << 32)

function _require_supported_windows_cpu_set_runtime()
    Sys.iswindows() || throw(ArgumentError("Windows CPU Sets require Windows"))
    Sys.WORD_SIZE == 64 || throw(ArgumentError(
        "the audited SYSTEM_CPU_SET_INFORMATION layout requires 64-bit Windows",
    ))
    return nothing
end

_last_win32_error() = ccall((:GetLastError, "Kernel32"), UInt32, ())

function _throw_win32_error(api::AbstractString, code::UInt32=_last_win32_error())
    error("$api failed with Win32 error $code")
end

_current_process_handle() =
    ccall((:GetCurrentProcess, "Kernel32"), Ptr{Cvoid}, ())

_current_thread_handle() =
    ccall((:GetCurrentThread, "Kernel32"), Ptr{Cvoid}, ())

function _parse_cpu_set_buffer(buffer::Vector{UInt8})
    records = WindowsCPUSetRecord[]
    offset = 1
    while offset <= length(buffer)
        length(buffer) - offset + 1 >= 8 ||
            error("truncated SYSTEM_CPU_SET_INFORMATION header")
        record_size = _read_u32_le(buffer, offset)
        record_size >= 8 || error("invalid SYSTEM_CPU_SET_INFORMATION record size")
        record_end = offset + Int(record_size) - 1
        record_end <= length(buffer) ||
            error("SYSTEM_CPU_SET_INFORMATION record exceeds returned buffer")
        information_type = _read_u32_le(buffer, offset + 4)
        if information_type == _CPU_SET_INFORMATION_TYPE
            record_size >= _CPU_SET_MINIMUM_RECORD_SIZE || error(
                "CpuSet record is smaller than the audited Windows 10+ layout",
            )
            flags = buffer[offset + 19]
            push!(records, WindowsCPUSetRecord(
                record_size=record_size,
                id=_read_u32_le(buffer, offset + 8),
                group=_read_u16_le(buffer, offset + 12),
                logical_processor_index=buffer[offset + 14],
                core_index=buffer[offset + 15],
                last_level_cache_index=buffer[offset + 16],
                numa_node_index=buffer[offset + 17],
                efficiency_class=buffer[offset + 18],
                scheduling_class=buffer[offset + 20],
                all_flags=flags,
                parked=(flags & 0x01) != 0,
                allocated=(flags & 0x02) != 0,
                allocated_to_target_process=(flags & 0x04) != 0,
                realtime=(flags & 0x08) != 0,
                reserved_flags=flags >> 4,
                allocation_tag=_read_u64_le(buffer, offset + 24),
            ))
        end
        offset = record_end + 1
    end
    offset == length(buffer) + 1 || error("CPU Set record iteration did not end exactly")
    return records
end

"""Dynamically enumerate CPU Sets for the current process.

The variable-sized Win32 records are walked by their `Size` fields. Discovery
is repeated after an insufficient-buffer race and never assumes CPU Set IDs or
logical-processor numbering.
"""
function enumerate_windows_cpu_sets(; process_handle=_current_process_handle())
    _require_supported_windows_cpu_set_runtime()
    for _ in 1:4
        required = Ref{UInt32}(0)
        success = ccall(
            (:GetSystemCpuSetInformation, "Kernel32"),
            Int32,
            (Ptr{Cvoid}, UInt32, Ref{UInt32}, Ptr{Cvoid}, UInt32),
            C_NULL,
            UInt32(0),
            required,
            process_handle,
            UInt32(0),
        )
        if success == 0
            code = _last_win32_error()
            code == _ERROR_INSUFFICIENT_BUFFER || _throw_win32_error(
                "GetSystemCpuSetInformation(size)",
                code,
            )
        end
        required[] > 0 || error("GetSystemCpuSetInformation returned no CPU Sets")
        buffer = Vector{UInt8}(undef, Int(required[]))
        returned = Ref{UInt32}(required[])
        success = GC.@preserve buffer ccall(
            (:GetSystemCpuSetInformation, "Kernel32"),
            Int32,
            (Ptr{Cvoid}, UInt32, Ref{UInt32}, Ptr{Cvoid}, UInt32),
            pointer(buffer),
            UInt32(length(buffer)),
            returned,
            process_handle,
            UInt32(0),
        )
        if success != 0
            returned[] <= length(buffer) || error("Win32 returned an oversized CPU Set buffer")
            resize!(buffer, Int(returned[]))
            records = _parse_cpu_set_buffer(buffer)
            isempty(records) && error("CPU Set enumeration contained no CpuSet records")
            return records
        end
        code = _last_win32_error()
        code == _ERROR_INSUFFICIENT_BUFFER || _throw_win32_error(
            "GetSystemCpuSetInformation(data)",
            code,
        )
    end
    error("CPU Set topology changed repeatedly during enumeration")
end

function _classification_failure(reason::AbstractString)
    return CoreUltra265KCPUSetClassification(
        false,
        "fail_closed",
        String(reason),
        nothing,
        nothing,
        UInt32[],
        UInt32[],
    )
end

_cpu_set_order(record::WindowsCPUSetRecord) = (
    record.group,
    record.logical_processor_index,
    record.core_index,
    record.id,
)

_cpu_set_usable(record::WindowsCPUSetRecord) =
    !record.parked &&
    !record.realtime &&
    (!record.allocated || record.allocated_to_target_process)

"""Classify only the exact discovered 8P+12E, one-thread-per-core topology."""
function classify_core_ultra_7_265k_cpu_sets(
    records::AbstractVector{WindowsCPUSetRecord},
)
    isempty(records) && return _classification_failure("no CPU Sets were discovered")
    length(unique(record.id for record in records)) == length(records) ||
        return _classification_failure("duplicate CPU Set IDs")
    all(record -> !record.allocated_to_target_process || record.allocated, records) ||
        return _classification_failure("inconsistent CPU Set allocation flags")
    all(_cpu_set_usable, records) || return _classification_failure(
        "a CPU Set is parked, real-time, or allocated to another process",
    )
    length(records) == 20 || return _classification_failure(
        "target requires exactly 20 usable logical processors",
    )
    length(unique(record.group for record in records)) == 1 ||
        return _classification_failure("target discovery unexpectedly spans processor groups")
    length(unique((record.group, record.logical_processor_index) for record in records)) == 20 ||
        return _classification_failure("duplicate group-relative logical processor")
    length(unique((record.group, record.core_index) for record in records)) == 20 ||
        return _classification_failure("SMT or duplicate group-relative core was discovered")

    classes = sort!(unique(record.efficiency_class for record in records))
    length(classes) == 2 || return _classification_failure(
        "target requires exactly two intrinsic efficiency classes",
    )
    efficiency_class, performance_class = classes
    performance = sort!(
        [record for record in records if record.efficiency_class == performance_class];
        by=_cpu_set_order,
    )
    efficiency = sort!(
        [record for record in records if record.efficiency_class == efficiency_class];
        by=_cpu_set_order,
    )
    length(performance) == 8 || return _classification_failure(
        "higher efficiency-class cardinality is not the target eight P-cores",
    )
    length(efficiency) == 12 || return _classification_failure(
        "lower efficiency-class cardinality is not the target twelve E-cores",
    )
    return CoreUltra265KCPUSetClassification(
        true,
        "target_8p12e",
        "two-class 8P+12E topology discovered without numbering assumptions",
        performance_class,
        efficiency_class,
        UInt32[record.id for record in performance],
        UInt32[record.id for record in efficiency],
    )
end

"""Return dynamically discovered IDs for a declared role after classification."""
function cpu_set_ids_for_role(
    classification::CoreUltra265KCPUSetClassification,
    role::Symbol;
    leave_efficiency_unassigned::Integer=0,
)
    classification.adopted || throw(ArgumentError(
        "P/E role assignment is forbidden after fail-closed topology classification",
    ))
    leave_efficiency_unassigned >= 0 ||
        throw(ArgumentError("leave_efficiency_unassigned must be nonnegative"))
    if role == :p_sparse
        leave_efficiency_unassigned == 0 || throw(ArgumentError(
            "leave_efficiency_unassigned applies only to :e_background",
        ))
        return copy(classification.performance_cpu_set_ids)
    elseif role == :e_background
        ids = classification.efficiency_cpu_set_ids
        leave_efficiency_unassigned < length(ids) || throw(ArgumentError(
            "at least one E-core must remain assigned when the E role is enabled",
        ))
        return copy(ids[1:(end - leave_efficiency_unassigned)])
    end
    throw(ArgumentError("unknown CPU Set role $role"))
end

function _record_contract(record::WindowsCPUSetRecord)
    return (
        record_size=record.record_size,
        id=record.id,
        group=record.group,
        logical_processor_index=record.logical_processor_index,
        core_index=record.core_index,
        last_level_cache_index=record.last_level_cache_index,
        numa_node_index=record.numa_node_index,
        efficiency_class=record.efficiency_class,
        scheduling_class=record.scheduling_class,
        flags=(
            all_flags=record.all_flags,
            parked=record.parked,
            allocated=record.allocated,
            allocated_to_target_process=record.allocated_to_target_process,
            realtime=record.realtime,
            reserved_flags=record.reserved_flags,
        ),
        allocation_tag=record.allocation_tag,
    )
end

function _topology_payload(records::AbstractVector{WindowsCPUSetRecord})
    ordered = sort!(collect(records); by=_cpu_set_order)
    classification = classify_core_ultra_7_265k_cpu_sets(ordered)
    return (
        schema="windows-cpu-set-topology-v1",
        discovery_api="GetSystemCpuSetInformation",
        cpu_set_count=length(ordered),
        classification=(
            adopted=classification.adopted,
            status=classification.status,
            reason=classification.reason,
            performance_efficiency_class=classification.performance_efficiency_class,
            efficiency_efficiency_class=classification.efficiency_efficiency_class,
            performance_cpu_set_ids=classification.performance_cpu_set_ids,
            efficiency_cpu_set_ids=classification.efficiency_cpu_set_ids,
        ),
        cpu_sets=[_record_contract(record) for record in ordered],
    )
end

windows_cpu_set_topology_json(records::AbstractVector{WindowsCPUSetRecord}) =
    String(JSON3.write(_topology_payload(records)))

windows_cpu_set_topology_sha256(records::AbstractVector{WindowsCPUSetRecord}) =
    bytes2hex(SHA.sha256(Vector{UInt8}(codeunits(windows_cpu_set_topology_json(records)))))

"""JSON/hash fields that must be bound into a later measured system witness."""
function windows_cpu_set_topology_contract(records::AbstractVector{WindowsCPUSetRecord})
    classification = classify_core_ultra_7_265k_cpu_sets(records)
    topology_json = windows_cpu_set_topology_json(records)
    return (
        schema="windows-cpu-set-assignment-contract-v1",
        topology_json=topology_json,
        topology_sha256=bytes2hex(SHA.sha256(Vector{UInt8}(codeunits(topology_json)))),
        classification_status=classification.status,
        classification_adopted=classification.adopted,
        performance_cpu_set_ids=classification.performance_cpu_set_ids,
        efficiency_cpu_set_ids=classification.efficiency_cpu_set_ids,
        assignment_status="unapplied",
        required_application_apis=(
            "SetProcessDefaultCpuSets",
            "SetThreadSelectedCpuSets",
        ),
        required_readback_apis=(
            "GetProcessDefaultCpuSets",
            "GetThreadSelectedCpuSets",
        ),
        residency_probe_api="GetCurrentProcessorNumberEx",
        note="a topology contract is not evidence that any worker was assigned or resident",
    )
end

function _normalized_cpu_set_ids(ids::AbstractVector{<:Integer})
    normalized = UInt32[]
    sizehint!(normalized, length(ids))
    for id in ids
        0 <= id <= typemax(UInt32) || throw(ArgumentError("CPU Set ID is outside UInt32"))
        push!(normalized, UInt32(id))
    end
    length(unique(normalized)) == length(normalized) ||
        throw(ArgumentError("CPU Set assignment contains duplicate IDs"))
    sort!(normalized)
    return normalized
end

function read_process_default_cpu_sets(; process_handle=_current_process_handle())
    _require_supported_windows_cpu_set_runtime()
    for _ in 1:4
        required = Ref{UInt32}(0)
        success = ccall(
            (:GetProcessDefaultCpuSets, "Kernel32"),
            Int32,
            (Ptr{Cvoid}, Ptr{UInt32}, UInt32, Ref{UInt32}),
            process_handle,
            C_NULL,
            UInt32(0),
            required,
        )
        if success != 0 && required[] == 0
            return UInt32[]
        elseif success == 0
            code = _last_win32_error()
            code == _ERROR_INSUFFICIENT_BUFFER || _throw_win32_error(
                "GetProcessDefaultCpuSets(size)",
                code,
            )
        end
        ids = Vector{UInt32}(undef, Int(required[]))
        success = GC.@preserve ids ccall(
            (:GetProcessDefaultCpuSets, "Kernel32"),
            Int32,
            (Ptr{Cvoid}, Ptr{UInt32}, UInt32, Ref{UInt32}),
            process_handle,
            pointer(ids),
            UInt32(length(ids)),
            required,
        )
        if success != 0
            resize!(ids, Int(required[]))
            sort!(ids)
            return ids
        end
        code = _last_win32_error()
        code == _ERROR_INSUFFICIENT_BUFFER || _throw_win32_error(
            "GetProcessDefaultCpuSets(data)",
            code,
        )
    end
    error("process CPU Set assignment changed repeatedly during readback")
end

function _set_process_default_cpu_set_ids!(ids::Vector{UInt32}, process_handle)
    _require_supported_windows_cpu_set_runtime()
    success = GC.@preserve ids begin
        pointer_or_null = isempty(ids) ? Ptr{UInt32}(C_NULL) : pointer(ids)
        ccall(
            (:SetProcessDefaultCpuSets, "Kernel32"),
            Int32,
            (Ptr{Cvoid}, Ptr{UInt32}, UInt32),
            process_handle,
            pointer_or_null,
            UInt32(length(ids)),
        )
    end
    success != 0 || _throw_win32_error("SetProcessDefaultCpuSets")
    return nothing
end

"""Apply a discovered P/E role transactionally and bind it to live topology.

The function intentionally accepts a role, not caller-supplied CPU Set IDs. It
enumerates and classifies immediately before assignment, performs exact
readback, re-enumerates immediately afterward, and restores the prior assignment
if any check or topology hash fails.
"""
function apply_process_default_cpu_sets!(
    role::Symbol;
    leave_efficiency_unassigned::Union{Nothing,Integer}=nothing,
)
    process_handle = _current_process_handle()
    leave_count = isnothing(leave_efficiency_unassigned) ?
                  (role == :e_background ? 1 : 0) :
                  leave_efficiency_unassigned
    before_records = enumerate_windows_cpu_sets(; process_handle)
    before_hash = windows_cpu_set_topology_sha256(before_records)
    classification = classify_core_ultra_7_265k_cpu_sets(before_records)
    requested = _normalized_cpu_set_ids(cpu_set_ids_for_role(
        classification,
        role;
        leave_efficiency_unassigned=leave_count,
    ))
    previous = read_process_default_cpu_sets(; process_handle)
    try
        _set_process_default_cpu_set_ids!(requested, process_handle)
        observed = read_process_default_cpu_sets(; process_handle)
        observed == requested || error("process CPU Set readback differs from requested IDs")
        after_records = enumerate_windows_cpu_sets(; process_handle)
        after_hash = windows_cpu_set_topology_sha256(after_records)
        after_hash == before_hash || error("CPU Set topology changed during process assignment")
        return (
            schema="windows-process-cpu-set-assignment-v1",
            topology_sha256=before_hash,
            role=role,
            requested_cpu_set_ids=requested,
            readback_cpu_set_ids=observed,
            assignment_status="applied_readback_verified_residency_unverified",
        )
    catch
        _set_process_default_cpu_set_ids!(previous, process_handle)
        read_process_default_cpu_sets(; process_handle) == previous || error(
            "process CPU Set assignment failed and prior assignment could not be restored",
        )
        rethrow()
    end
end

function clear_process_default_cpu_sets!(; process_handle=_current_process_handle())
    _require_supported_windows_cpu_set_runtime()
    _set_process_default_cpu_set_ids!(UInt32[], process_handle)
    isempty(read_process_default_cpu_sets(; process_handle)) ||
        error("process CPU Set assignment did not clear")
    return UInt32[]
end

function read_thread_selected_cpu_sets(; thread_handle=_current_thread_handle())
    _require_supported_windows_cpu_set_runtime()
    for _ in 1:4
        required = Ref{UInt32}(0)
        success = ccall(
            (:GetThreadSelectedCpuSets, "Kernel32"),
            Int32,
            (Ptr{Cvoid}, Ptr{UInt32}, UInt32, Ref{UInt32}),
            thread_handle,
            C_NULL,
            UInt32(0),
            required,
        )
        if success != 0 && required[] == 0
            return UInt32[]
        elseif success == 0
            code = _last_win32_error()
            code == _ERROR_INSUFFICIENT_BUFFER || _throw_win32_error(
                "GetThreadSelectedCpuSets(size)",
                code,
            )
        end
        ids = Vector{UInt32}(undef, Int(required[]))
        success = GC.@preserve ids ccall(
            (:GetThreadSelectedCpuSets, "Kernel32"),
            Int32,
            (Ptr{Cvoid}, Ptr{UInt32}, UInt32, Ref{UInt32}),
            thread_handle,
            pointer(ids),
            UInt32(length(ids)),
            required,
        )
        if success != 0
            resize!(ids, Int(required[]))
            sort!(ids)
            return ids
        end
        code = _last_win32_error()
        code == _ERROR_INSUFFICIENT_BUFFER || _throw_win32_error(
            "GetThreadSelectedCpuSets(data)",
            code,
        )
    end
    error("thread CPU Set assignment changed repeatedly during readback")
end

function _set_thread_selected_cpu_set_ids!(ids::Vector{UInt32}, thread_handle)
    _require_supported_windows_cpu_set_runtime()
    success = GC.@preserve ids begin
        pointer_or_null = isempty(ids) ? Ptr{UInt32}(C_NULL) : pointer(ids)
        ccall(
            (:SetThreadSelectedCpuSets, "Kernel32"),
            Int32,
            (Ptr{Cvoid}, Ptr{UInt32}, UInt32),
            thread_handle,
            pointer_or_null,
            UInt32(length(ids)),
        )
    end
    success != 0 || _throw_win32_error("SetThreadSelectedCpuSets")
    return nothing
end

"""Apply a live-classified role to one dedicated OS thread transactionally."""
function apply_thread_selected_cpu_sets!(
    role::Symbol;
    leave_efficiency_unassigned::Union{Nothing,Integer}=nothing,
)
    process_handle = _current_process_handle()
    thread_handle = _current_thread_handle()
    leave_count = isnothing(leave_efficiency_unassigned) ?
                  (role == :e_background ? 1 : 0) :
                  leave_efficiency_unassigned
    before_records = enumerate_windows_cpu_sets(; process_handle)
    before_hash = windows_cpu_set_topology_sha256(before_records)
    classification = classify_core_ultra_7_265k_cpu_sets(before_records)
    requested = _normalized_cpu_set_ids(cpu_set_ids_for_role(
        classification,
        role;
        leave_efficiency_unassigned=leave_count,
    ))
    previous = read_thread_selected_cpu_sets(; thread_handle)
    try
        _set_thread_selected_cpu_set_ids!(requested, thread_handle)
        observed = read_thread_selected_cpu_sets(; thread_handle)
        observed == requested || error("thread CPU Set readback differs from requested IDs")
        after_records = enumerate_windows_cpu_sets(; process_handle)
        after_hash = windows_cpu_set_topology_sha256(after_records)
        after_hash == before_hash || error("CPU Set topology changed during thread assignment")
        return (
            schema="windows-thread-cpu-set-assignment-v1",
            topology_sha256=before_hash,
            role=role,
            requested_cpu_set_ids=requested,
            readback_cpu_set_ids=observed,
            assignment_status="applied_readback_verified_residency_unverified",
        )
    catch
        _set_thread_selected_cpu_set_ids!(previous, thread_handle)
        read_thread_selected_cpu_sets(; thread_handle) == previous || error(
            "thread CPU Set assignment failed and prior assignment could not be restored",
        )
        rethrow()
    end
end

function clear_thread_selected_cpu_sets!(; thread_handle=_current_thread_handle())
    _require_supported_windows_cpu_set_runtime()
    _set_thread_selected_cpu_set_ids!(UInt32[], thread_handle)
    isempty(read_thread_selected_cpu_sets(; thread_handle)) ||
        error("thread CPU Set assignment did not clear")
    return UInt32[]
end

struct _PROCESSOR_NUMBER
    group::UInt16
    number::UInt8
    reserved::UInt8
end

"""Capture current OS thread and processor plus effective CPU Set readback.

This is a point witness, not proof of whole-stage residency; ETW context-switch
evidence remains required before a performance claim.
"""
function current_windows_thread_witness(
    records::AbstractVector{WindowsCPUSetRecord};
    process_handle=_current_process_handle(),
    thread_handle=_current_thread_handle(),
)
    _require_supported_windows_cpu_set_runtime()
    processor = Ref(_PROCESSOR_NUMBER(UInt16(0), UInt8(0), UInt8(0)))
    ccall(
        (:GetCurrentProcessorNumberEx, "Kernel32"),
        Cvoid,
        (Ref{_PROCESSOR_NUMBER},),
        processor,
    )
    thread_id = ccall((:GetCurrentThreadId, "Kernel32"), UInt32, ())
    selected = read_thread_selected_cpu_sets(; thread_handle)
    defaults = read_process_default_cpu_sets(; process_handle)
    effective = isempty(selected) ? defaults : selected
    matching = sort!(UInt32[
        record.id for record in records
        if record.group == processor[].group &&
           record.logical_processor_index == processor[].number
    ])
    return (
        schema="windows-cpu-set-thread-point-witness-v1",
        topology_sha256=windows_cpu_set_topology_sha256(records),
        thread_id=thread_id,
        processor_group=processor[].group,
        logical_processor_index=processor[].number,
        matching_cpu_set_ids=matching,
        thread_selected_cpu_set_ids=selected,
        process_default_cpu_set_ids=defaults,
        effective_cpu_set_ids=effective,
        current_in_effective_set=(
            isempty(effective) ? nothing : any(id -> id in effective, matching)
        ),
        note="point sample only; ETW residency evidence is deferred",
    )
end
