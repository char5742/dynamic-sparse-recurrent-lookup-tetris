module WinCpuSets

export discover_topology, topology_summary, configure_worker_bindings,
       bind_current_worker!, clear_current_binding!, process_cpu_ticks_100ns,
       thread_cpu_ticks_100ns

const KERNEL32 = "Kernel32.dll"
const ERROR_INSUFFICIENT_BUFFER = UInt32(122)
const CPU_SET_INFORMATION = UInt32(0)
const CPU_SET_PREFIX_BYTES = 32

const FLAG_PARKED = UInt8(0x01)
const FLAG_ALLOCATED = UInt8(0x02)
const FLAG_ALLOCATED_TO_TARGET_PROCESS = UInt8(0x04)
const FLAG_REAL_TIME = UInt8(0x08)

"""The stable 32-byte x64 prefix of SYSTEM_CPU_SET_INFORMATION/CpuSet."""
struct WinCpuSetInformationV1
    size::UInt32
    kind::UInt32
    id::UInt32
    group::UInt16
    logical_processor_index::UInt8
    core_index::UInt8
    last_level_cache_index::UInt8
    numa_node_index::UInt8
    efficiency_class::UInt8
    all_flags::UInt8
    scheduling_reserved::UInt32
    allocation_tag::UInt64
end

struct WinFileTime
    low::UInt32
    high::UInt32
end

struct CpuSetInfo
    id::UInt32
    group::UInt16
    logical_processor_index::UInt8
    core_index::UInt8
    last_level_cache_index::UInt8
    numa_node_index::UInt8
    efficiency_class::UInt8
    all_flags::UInt8
    scheduling_class::UInt8
    allocation_tag::UInt64
end

struct CpuTopology
    logical_cpu_sets::Vector{CpuSetInfo}
    physical_cores::Vector{CpuSetInfo}
    p_cores::Vector{CpuSetInfo}
    e_cores::Vector{CpuSetInfo}
    efficiency_classes::Vector{UInt8}
end

struct WorkerBindingPlan
    generation::UInt64
    mode::Symbol
    requested_workers::Int
    julia_default_workers::Int
    assignments::Vector{Union{Nothing,UInt32}}
    topology::CpuTopology
end

const ACTIVE_PLAN = Ref{Union{Nothing,WorkerBindingPlan}}(nothing)
const PLAN_GENERATION = Ref{UInt64}(0)
const BOUND_GENERATION = zeros(UInt64, Base.Threads.maxthreadid())

@inline _is_parked(info::CpuSetInfo) = !iszero(info.all_flags & FLAG_PARKED)
@inline _is_allocated(info::CpuSetInfo) = !iszero(info.all_flags & FLAG_ALLOCATED)
@inline _is_allocated_to_target(info::CpuSetInfo) =
    !iszero(info.all_flags & FLAG_ALLOCATED_TO_TARGET_PROCESS)
@inline _is_real_time(info::CpuSetInfo) = !iszero(info.all_flags & FLAG_REAL_TIME)
@inline _is_available(info::CpuSetInfo) =
    !_is_allocated(info) || _is_allocated_to_target(info)

function _require_windows()
    Sys.iswindows() || error("WinCpuSets is supported only on Windows")
    Sys.WORD_SIZE == 64 || error("WinCpuSets requires 64-bit Windows")
    return nothing
end

@inline _last_error() =
    ccall((:GetLastError, KERNEL32), UInt32, ())

@inline _current_process() =
    ccall((:GetCurrentProcess, KERNEL32), Ptr{Cvoid}, ())

@inline _current_thread() =
    ccall((:GetCurrentThread, KERNEL32), Ptr{Cvoid}, ())

function _assert_abi_layouts()
    sizeof(WinCpuSetInformationV1) == CPU_SET_PREFIX_BYTES || error(
        "unexpected SYSTEM_CPU_SET_INFORMATION prefix size",
    )
    expected_offsets = (0, 4, 8, 12, 14, 15, 16, 17, 18, 19, 20, 24)
    actual_offsets = ntuple(
        field -> Int(fieldoffset(WinCpuSetInformationV1, field)),
        fieldcount(WinCpuSetInformationV1),
    )
    actual_offsets == expected_offsets || error(
        "unexpected SYSTEM_CPU_SET_INFORMATION prefix layout: $actual_offsets",
    )
    sizeof(WinFileTime) == 8 || error("unexpected FILETIME layout")
    return nothing
end

function _query_cpu_set_buffer()
    process = _current_process()
    required = Ref{UInt32}(0)
    ok = ccall(
        (:GetSystemCpuSetInformation, KERNEL32),
        Cint,
        (Ptr{UInt8}, UInt32, Ref{UInt32}, Ptr{Cvoid}, UInt32),
        Ptr{UInt8}(0), UInt32(0), required, process, UInt32(0),
    )
    if iszero(ok)
        code = _last_error()
        code == ERROR_INSUFFICIENT_BUFFER || error(
            "GetSystemCpuSetInformation size query failed; GetLastError=$code",
        )
    end
    required[] > 0 || error("Windows returned no CPU Set information")

    # The topology may change between the size query and the data query.
    # Retry only the documented insufficient-buffer race, and fail otherwise.
    for _ in 1:4
        bytes = Int(required[])
        UInt32(bytes) == required[] || error("CPU Set buffer length overflow")
        buffer = Vector{UInt8}(undef, bytes)
        returned = Ref{UInt32}(0)
        ok = GC.@preserve buffer begin
            ccall(
                (:GetSystemCpuSetInformation, KERNEL32),
                Cint,
                (Ptr{UInt8}, UInt32, Ref{UInt32}, Ptr{Cvoid}, UInt32),
                pointer(buffer), UInt32(length(buffer)), returned, process, UInt32(0),
            )
        end
        if !iszero(ok)
            used = Int(returned[])
            0 < used <= length(buffer) || error(
                "GetSystemCpuSetInformation returned an invalid length: $used",
            )
            resize!(buffer, used)
            return buffer
        end
        code = _last_error()
        code == ERROR_INSUFFICIENT_BUFFER || error(
            "GetSystemCpuSetInformation failed; GetLastError=$code",
        )
        returned[] > 0 || error("CPU Set retry returned an empty required length")
        required[] = returned[]
    end
    error("CPU Set topology changed during every enumeration attempt")
end

@inline function _read_at(::Type{T}, base::Ptr{UInt8}, offset::Int) where {T}
    return unsafe_load(Ptr{T}(base + offset))
end

function _parse_cpu_sets(buffer::Vector{UInt8})
    records = CpuSetInfo[]
    offset = 0
    GC.@preserve buffer begin
        base = pointer(buffer)
        while offset < length(buffer)
            length(buffer) - offset >= 8 || error(
                "truncated SYSTEM_CPU_SET_INFORMATION header at byte $offset",
            )
            record_size = Int(_read_at(UInt32, base, offset))
            record_size >= 8 || error("invalid CPU Set record size: $record_size")
            next_offset = offset + record_size
            next_offset <= length(buffer) || error(
                "CPU Set record extends past the returned buffer",
            )
            kind = _read_at(UInt32, base, offset + 4)
            if kind == CPU_SET_INFORMATION
                record_size >= CPU_SET_PREFIX_BYTES || error(
                    "CPU Set record is smaller than the supported prefix: $record_size",
                )
                scheduling_reserved = _read_at(UInt32, base, offset + 20)
                push!(records, CpuSetInfo(
                    _read_at(UInt32, base, offset + 8),
                    _read_at(UInt16, base, offset + 12),
                    _read_at(UInt8, base, offset + 14),
                    _read_at(UInt8, base, offset + 15),
                    _read_at(UInt8, base, offset + 16),
                    _read_at(UInt8, base, offset + 17),
                    _read_at(UInt8, base, offset + 18),
                    _read_at(UInt8, base, offset + 19),
                    UInt8(scheduling_reserved & UInt32(0xff)),
                    _read_at(UInt64, base, offset + 24),
                ))
            end
            offset = next_offset
        end
    end
    isempty(records) && error("Windows returned no CpuSetInformation records")
    ids = Set(info.id for info in records)
    length(ids) == length(records) || error("Windows returned duplicate CPU Set IDs")
    return records
end

@inline _core_key(info::CpuSetInfo) = (info.group, info.core_index)
@inline _core_order(info::CpuSetInfo) = (
    info.group,
    info.core_index,
    info.logical_processor_index,
    info.id,
)

function _physical_core_representatives(logical_cpu_sets::Vector{CpuSetInfo})
    by_core = Dict{Tuple{UInt16,UInt8},Vector{CpuSetInfo}}()
    for info in logical_cpu_sets
        _is_available(info) || continue
        push!(get!(by_core, _core_key(info), CpuSetInfo[]), info)
    end
    isempty(by_core) && error("no CPU Sets are available to this process")

    representatives = CpuSetInfo[]
    for key in sort!(collect(keys(by_core)))
        siblings = by_core[key]
        # One CPU Set per physical core. Prefer a non-parked SMT sibling, then
        # use the API-provided logical index and ID only as deterministic ties.
        sort!(siblings; by=info -> (
            _is_parked(info), info.logical_processor_index, info.id,
        ))
        push!(representatives, first(siblings))
    end
    sort!(representatives; by=_core_order)
    return representatives
end

"""
    discover_topology() -> CpuTopology

Enumerate Windows CPU Sets at runtime. SMT siblings are grouped by the
API-provided `(Group, CoreIndex)` and exactly one available CPU Set represents
each physical core. The maximum observed EfficiencyClass is the P-core class;
all lower classes are E-core classes. No CPU Set ID is inferred or hardcoded.
"""
function discover_topology()
    _require_windows()
    _assert_abi_layouts()
    logical_cpu_sets = _parse_cpu_sets(_query_cpu_set_buffer())
    physical_cores = _physical_core_representatives(logical_cpu_sets)
    efficiency_classes = sort!(unique(
        info.efficiency_class for info in physical_cores
    ))
    isempty(efficiency_classes) && error("no physical-core efficiency classes found")
    p_class = maximum(efficiency_classes)
    p_cores = sort!(
        [info for info in physical_cores if info.efficiency_class == p_class];
        by=_core_order,
    )
    e_cores = sort!(
        [info for info in physical_cores if info.efficiency_class != p_class];
        by=_core_order,
    )
    length(p_cores) + length(e_cores) == length(physical_cores) || error(
        "P/E core classification lost a physical core",
    )
    return CpuTopology(
        logical_cpu_sets,
        physical_cores,
        p_cores,
        e_cores,
        collect(efficiency_classes),
    )
end

@inline function _cpu_set_summary(info::CpuSetInfo)
    return (;
        cpu_set_id=Int(info.id),
        group=Int(info.group),
        logical_processor_index=Int(info.logical_processor_index),
        core_index=Int(info.core_index),
        efficiency_class=Int(info.efficiency_class),
        parked=_is_parked(info),
        allocated=_is_allocated(info),
        allocated_to_target_process=_is_allocated_to_target(info),
        real_time=_is_real_time(info),
    )
end

"""Return a serialization-friendly summary of a discovered topology."""
function topology_summary(topology::CpuTopology=discover_topology())
    classes = [
        (;
            efficiency_class=Int(class),
            logical_cpu_sets=count(
                info -> info.efficiency_class == class,
                topology.logical_cpu_sets,
            ),
            physical_cores=count(
                info -> info.efficiency_class == class,
                topology.physical_cores,
            ),
        )
        for class in topology.efficiency_classes
    ]
    return (;
        logical_cpu_set_count=length(topology.logical_cpu_sets),
        available_logical_cpu_set_count=count(_is_available, topology.logical_cpu_sets),
        physical_core_count=length(topology.physical_cores),
        p_core_count=length(topology.p_cores),
        e_core_count=length(topology.e_cores),
        efficiency_classes=Int.(topology.efficiency_classes),
        efficiency_class_counts=classes,
        processor_groups=sort!(unique(Int(info.group) for info in topology.logical_cpu_sets)),
        p_core_cpu_sets=_cpu_set_summary.(topology.p_cores),
        e_core_cpu_sets=_cpu_set_summary.(topology.e_cores),
    )
end

function _validate_configure_context(requested_workers::Int)
    _require_windows()
    requested_workers >= 1 || error("requested_workers must be positive")
    Base.Threads.threadpool() === :default || error(
        "configure_worker_bindings must run on the Julia default thread pool",
    )
    Base.Threads.nthreads(:interactive) == 0 || error(
        "WinCpuSets requires JULIA_NUM_THREADS=N,0 (interactive pool must be zero)",
    )
    default_workers = Base.Threads.nthreads(:default)
    requested_workers <= default_workers || error(
        "requested $requested_workers workers but Julia has $default_workers default workers",
    )
    return default_workers
end

"""
    configure_worker_bindings(mode, requested_workers)

Create the fail-closed binding plan used by `bind_current_worker!`. Supported
modes are `:all`, `:p_only`, and `:none`. `:all` requires one Julia worker per
discovered physical core. `:p_only` selects only runtime-discovered maximum
EfficiencyClass cores. Call this before entering the native worker team.
"""
function configure_worker_bindings(
    mode::Symbol,
    requested_workers::Integer,
    topology::CpuTopology=discover_topology(),
)
    requested = Int(requested_workers)
    default_workers = _validate_configure_context(requested)
    mode in (:all, :p_only, :none) || error(
        "worker binding mode must be :all, :p_only, or :none",
    )
    assignments = Union{Nothing,UInt32}[nothing for _ in 1:default_workers]

    selected = CpuSetInfo[]
    if mode === :all
        requested == length(topology.physical_cores) || error(
            ":all requires exactly $(length(topology.physical_cores)) workers; requested $requested",
        )
        # Keep the main Julia worker and serial optimizer on a P core.
        selected = vcat(topology.p_cores, topology.e_cores)
    elseif mode === :p_only
        length(topology.efficiency_classes) >= 2 || error(
            ":p_only requires at least two runtime EfficiencyClass values",
        )
        requested <= length(topology.p_cores) || error(
            "requested $requested P workers but only $(length(topology.p_cores)) P cores were detected",
        )
        selected = topology.p_cores[1:requested]
    end

    ids = UInt32[info.id for info in selected]
    length(ids) == length(Set(ids)) || error("binding plan contains duplicate CPU Set IDs")
    for worker_slot in eachindex(ids)
        assignments[worker_slot] = ids[worker_slot]
    end
    PLAN_GENERATION[] == typemax(UInt64) && error("CPU Set plan generation overflow")
    PLAN_GENERATION[] += UInt64(1)
    generation = PLAN_GENERATION[]
    fill!(BOUND_GENERATION, UInt64(0))
    plan = WorkerBindingPlan(
        generation, mode, requested, default_workers, assignments, topology,
    )
    ACTIVE_PLAN[] = plan
    return (;
        mode,
        generation,
        requested_workers=requested,
        julia_default_workers=default_workers,
        active_workers=requested,
        topology=topology_summary(topology),
        worker_assignments=[
            (;
                worker_slot,
                active=worker_slot <= requested,
                cpu_set_id=assignments[worker_slot] === nothing ? nothing :
                    Int(assignments[worker_slot]),
            )
            for worker_slot in 1:default_workers
        ],
    )
end

function _selected_cpu_sets_current_thread()
    thread = _current_thread()
    required = Ref{UInt32}(0)
    ok = ccall(
        (:GetThreadSelectedCpuSets, KERNEL32),
        Cint,
        (Ptr{Cvoid}, Ptr{UInt32}, UInt32, Ref{UInt32}),
        thread, Ptr{UInt32}(0), UInt32(0), required,
    )
    if !iszero(ok)
        iszero(required[]) || error(
            "GetThreadSelectedCpuSets succeeded without returning required IDs",
        )
        return UInt32[]
    end
    code = _last_error()
    code == ERROR_INSUFFICIENT_BUFFER || error(
        "GetThreadSelectedCpuSets size query failed; GetLastError=$code",
    )
    required[] > 0 || error("GetThreadSelectedCpuSets returned an empty required count")

    for _ in 1:4
        ids = Vector{UInt32}(undef, Int(required[]))
        returned = Ref{UInt32}(0)
        ok = GC.@preserve ids begin
            ccall(
                (:GetThreadSelectedCpuSets, KERNEL32),
                Cint,
                (Ptr{Cvoid}, Ptr{UInt32}, UInt32, Ref{UInt32}),
                thread, pointer(ids), UInt32(length(ids)), returned,
            )
        end
        if !iszero(ok)
            used = Int(returned[])
            0 <= used <= length(ids) || error(
                "GetThreadSelectedCpuSets returned an invalid count: $used",
            )
            resize!(ids, used)
            return ids
        end
        code = _last_error()
        code == ERROR_INSUFFICIENT_BUFFER || error(
            "GetThreadSelectedCpuSets failed; GetLastError=$code",
        )
        returned[] > 0 || error("thread CPU Set retry returned an empty count")
        required[] = returned[]
    end
    error("thread CPU Set assignment changed during every query attempt")
end

function _set_current_cpu_set!(cpu_set_id::UInt32)
    id = Ref{UInt32}(cpu_set_id)
    ok = ccall(
        (:SetThreadSelectedCpuSets, KERNEL32),
        Cint,
        (Ptr{Cvoid}, Ref{UInt32}, UInt32),
        _current_thread(), id, UInt32(1),
    )
    iszero(ok) && error(
        "SetThreadSelectedCpuSets failed; GetLastError=$(_last_error())",
    )
    actual = _selected_cpu_sets_current_thread()
    actual == UInt32[cpu_set_id] || error(
        "CPU Set verification failed: expected $cpu_set_id, received $actual",
    )
    return nothing
end

"""Clear and verify the explicit CPU Set selection of the current native thread."""
function clear_current_binding!()
    _require_windows()
    ok = ccall(
        (:SetThreadSelectedCpuSets, KERNEL32),
        Cint,
        (Ptr{Cvoid}, Ptr{UInt32}, UInt32),
        _current_thread(), Ptr{UInt32}(0), UInt32(0),
    )
    iszero(ok) && error(
        "clearing thread CPU Sets failed; GetLastError=$(_last_error())",
    )
    actual = _selected_cpu_sets_current_thread()
    isempty(actual) || error("CPU Set clear verification failed: received $actual")
    return nothing
end

"""
    bind_current_worker!(worker_slot)

Apply and immediately verify the configured assignment on the current Julia
native worker. `worker_slot` must be the stable slot supplied by a pinned
native-worker bootstrap. Slots beyond `requested_workers` are cleared and
reported inactive; candidate work must not be submitted to those slots.
"""
function bind_current_worker!(worker_slot::Integer)
    _require_windows()
    plan = ACTIVE_PLAN[]
    plan === nothing && error("configure_worker_bindings must be called first")
    slot = Int(worker_slot)
    1 <= slot <= plan.julia_default_workers || error("invalid worker slot: $slot")
    Base.Threads.threadpool() === :default || error(
        "worker slot $slot is not running on the Julia default thread pool",
    )
    expected_thread_id = slot # configure rejects a nonempty interactive pool.
    Base.Threads.threadid() == expected_thread_id || error(
        "worker slot $slot is running on Julia thread $(Base.Threads.threadid())",
    )

    expected = plan.assignments[slot]
    thread_id = Base.Threads.threadid()
    if BOUND_GENERATION[thread_id] == plan.generation
        return (;
            worker_slot=slot,
            julia_thread_id=thread_id,
            active=slot <= plan.requested_workers,
            cpu_set_id=expected === nothing ? nothing : Int(expected),
            verified=true,
            newly_bound=false,
        )
    end
    actual_before = _selected_cpu_sets_current_thread()
    if expected === nothing
        isempty(actual_before) || clear_current_binding!()
    elseif actual_before != UInt32[expected]
        _set_current_cpu_set!(expected)
    end
    actual_after = _selected_cpu_sets_current_thread()
    expected_ids = expected === nothing ? UInt32[] : UInt32[expected]
    actual_after == expected_ids || error(
        "worker $slot CPU Set verification failed: expected $expected_ids, received $actual_after",
    )
    BOUND_GENERATION[thread_id] = plan.generation
    return (;
        worker_slot=slot,
        julia_thread_id=Base.Threads.threadid(),
        active=slot <= plan.requested_workers,
        cpu_set_id=expected === nothing ? nothing : Int(expected),
        verified=true,
        newly_bound=true,
    )
end

@inline function _filetime_ticks(value::WinFileTime)
    return (UInt64(value.high) << 32) | UInt64(value.low)
end

@inline function _kernel_user_ticks(kernel::WinFileTime, user::WinFileTime)
    kernel_ticks = _filetime_ticks(kernel)
    user_ticks = _filetime_ticks(user)
    user_ticks <= typemax(UInt64) - kernel_ticks || error("CPU time overflow")
    return kernel_ticks + user_ticks
end

"""Return process kernel+user CPU time in native Windows 100 ns ticks."""
function process_cpu_ticks_100ns()
    _require_windows()
    creation = Ref(WinFileTime(UInt32(0), UInt32(0)))
    exit_time = Ref(WinFileTime(UInt32(0), UInt32(0)))
    kernel = Ref(WinFileTime(UInt32(0), UInt32(0)))
    user = Ref(WinFileTime(UInt32(0), UInt32(0)))
    ok = ccall(
        (:GetProcessTimes, KERNEL32),
        Cint,
        (
            Ptr{Cvoid}, Ref{WinFileTime}, Ref{WinFileTime},
            Ref{WinFileTime}, Ref{WinFileTime},
        ),
        _current_process(), creation, exit_time, kernel, user,
    )
    iszero(ok) && error("GetProcessTimes failed; GetLastError=$(_last_error())")
    return _kernel_user_ticks(kernel[], user[])
end

"""Return current native worker kernel+user CPU time in Windows 100 ns ticks."""
function thread_cpu_ticks_100ns()
    _require_windows()
    creation = Ref(WinFileTime(UInt32(0), UInt32(0)))
    exit_time = Ref(WinFileTime(UInt32(0), UInt32(0)))
    kernel = Ref(WinFileTime(UInt32(0), UInt32(0)))
    user = Ref(WinFileTime(UInt32(0), UInt32(0)))
    ok = ccall(
        (:GetThreadTimes, KERNEL32),
        Cint,
        (
            Ptr{Cvoid}, Ref{WinFileTime}, Ref{WinFileTime},
            Ref{WinFileTime}, Ref{WinFileTime},
        ),
        _current_thread(), creation, exit_time, kernel, user,
    )
    iszero(ok) && error("GetThreadTimes failed; GetLastError=$(_last_error())")
    return _kernel_user_ticks(kernel[], user[])
end

end # module WinCpuSets
