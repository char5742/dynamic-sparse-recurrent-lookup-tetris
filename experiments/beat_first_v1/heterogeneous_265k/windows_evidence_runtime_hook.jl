module WindowsEvidenceRuntimeHook

using JSON3
using SHA

export RoleThreadRegistration,
       EvidenceRoleManifest,
       SnapshotAck,
       TimedRegionEvidenceWitness,
       capture_role_thread_registration,
       role_manifest_json,
       role_manifest_sha256,
       run_with_counter_snapshot_handshake

const HOOK_IMPLEMENTATION_STATUS = "UNEXECUTED_STATIC_ONLY"
const _HASH_PATTERN = r"^[0-9a-f]{64}$"
const _IMC_CAPABILITY = "INTEL_PCM_RAW_IMC_RD_WR_COUNTERS_V1"

_qpc_now() = begin
    Sys.iswindows() || error("Windows evidence hook requires QueryPerformanceCounter")
    value = Ref{Int64}(0)
    ok = ccall((:QueryPerformanceCounter, "Kernel32"), Int32, (Ref{Int64},), value)
    ok != 0 || error("QueryPerformanceCounter failed")
    value[]
end

_current_thread_id() = begin
    Sys.iswindows() || error("Windows evidence hook requires GetCurrentThreadId")
    ccall((:GetCurrentThreadId, "Kernel32"), UInt32, ())
end

function _hash(value::AbstractString, label::AbstractString)
    normalized = lowercase(String(value))
    occursin(_HASH_PATTERN, normalized) || throw(ArgumentError(
        "$label must be a lowercase SHA-256 digest",
    ))
    normalized == value || throw(ArgumentError("$label must be canonical lowercase"))
    return normalized
end

"""Pre-bound role identity captured by the worker itself before timed work."""
Base.@kwdef struct RoleThreadRegistration
    thread_id::UInt32
    role_registration_qpc::Int64
    role::Symbol
    allowed_cpu_set_ids::Vector{UInt32}
    assignment_readback_sha256::String
end

"""Immutable runner/process identity plus every OS thread that owns timed work."""
Base.@kwdef struct EvidenceRoleManifest
    run_nonce::String
    process_id::UInt32
    runner_identity_qpc::Int64
    qpc_frequency::Int64
    cpu_set_topology_sha256::String
    system_contract_sha256::String
    registrations::Vector{RoleThreadRegistration}
end

"""Acknowledgement returned by the independent raw Intel PCM adapter.

The callback implementation is outside the benchmark runner. It must return a
fresh raw artifact digest and QPC bracket for the actual hardware snapshot.
"""
Base.@kwdef struct SnapshotAck
    phase::Symbol
    capability::String
    qpc_begin::Int64
    qpc_end::Int64
    raw_artifact_sha256::String
    counter_inventory_sha256::String
    counter_configuration_sha256::String
    program_generation::UInt64
    fallback_used::Bool
end

Base.@kwdef struct TimedRegionEvidenceWitness
    schema::String = "heterogeneous-265k-timed-region-hook-v1"
    implementation_status::String = HOOK_IMPLEMENTATION_STATUS
    role_manifest_sha256::String
    before_snapshot::SnapshotAck
    timed_begin_qpc::Int64
    timed_end_qpc::Int64
    after_snapshot::SnapshotAck
end

"""Capture one P-sparse or E-background OS thread registration.

`assignment_readback_sha256` is the immutable artifact produced from the
Set*CpuSets/Get*CpuSets transaction. Passing raw IDs without that readback is
not accepted.
"""
function capture_role_thread_registration(role::Symbol,
                                          allowed_cpu_set_ids::AbstractVector{<:Integer},
                                          assignment_readback_sha256::AbstractString)
    role in (:p_sparse, :e_background) || throw(ArgumentError(
        "role must be :p_sparse or :e_background",
    ))
    ids = UInt32[]
    for raw_id in allowed_cpu_set_ids
        0 <= raw_id <= typemax(UInt32) || throw(ArgumentError("CPU Set ID outside UInt32"))
        push!(ids, UInt32(raw_id))
    end
    isempty(ids) && throw(ArgumentError("allowed CPU Set IDs must not be empty"))
    length(unique(ids)) == length(ids) || throw(ArgumentError("duplicate CPU Set ID"))
    sort!(ids)
    return RoleThreadRegistration(
        thread_id=_current_thread_id(),
        role_registration_qpc=_qpc_now(),
        role=role,
        allowed_cpu_set_ids=ids,
        assignment_readback_sha256=_hash(
            assignment_readback_sha256, "assignment_readback_sha256",
        ),
    )
end

function _validated_manifest_payload(manifest::EvidenceRoleManifest)
    isempty(manifest.run_nonce) && throw(ArgumentError("run nonce must not be empty"))
    manifest.process_id > 0 || throw(ArgumentError("process ID must be nonzero"))
    manifest.runner_identity_qpc >= 0 || throw(ArgumentError(
        "runner identity QPC must be nonnegative",
    ))
    manifest.qpc_frequency > 0 || throw(ArgumentError("QPC frequency must be positive"))
    topology_sha = _hash(manifest.cpu_set_topology_sha256, "cpu_set_topology_sha256")
    system_sha = _hash(manifest.system_contract_sha256, "system_contract_sha256")
    isempty(manifest.registrations) && throw(ArgumentError("role manifest is empty"))
    ordered = sort!(copy(manifest.registrations); by=registration -> (
        registration.thread_id,
        String(registration.role),
        registration.role_registration_qpc,
    ))
    thread_ids = UInt32[]
    rows = NamedTuple[]
    for registration in ordered
        registration.thread_id > 0 || throw(ArgumentError("thread ID must be nonzero"))
        registration.role_registration_qpc >= manifest.runner_identity_qpc ||
            throw(ArgumentError("thread role was registered before runner identity"))
        registration.role in (:p_sparse, :e_background) || throw(ArgumentError(
            "unknown thread role",
        ))
        isempty(registration.allowed_cpu_set_ids) && throw(ArgumentError(
            "thread registration has no allowed CPU Set",
        ))
        length(unique(registration.allowed_cpu_set_ids)) ==
            length(registration.allowed_cpu_set_ids) || throw(ArgumentError(
                "thread registration has duplicate CPU Set IDs",
            ))
        push!(thread_ids, registration.thread_id)
        push!(rows, (
            thread_id=registration.thread_id,
            role_registration_qpc=registration.role_registration_qpc,
            role=String(registration.role),
            allowed_cpu_set_ids=sort(copy(registration.allowed_cpu_set_ids)),
            assignment_readback_sha256=_hash(
                registration.assignment_readback_sha256,
                "assignment_readback_sha256",
            ),
        ))
    end
    length(unique(thread_ids)) == length(thread_ids) || throw(ArgumentError(
        "one OS thread may have only one role in a timed run",
    ))
    return (
        schema="heterogeneous-265k-thread-role-manifest-v1",
        implementation_status=HOOK_IMPLEMENTATION_STATUS,
        run_nonce=manifest.run_nonce,
        process_id=manifest.process_id,
        runner_identity_qpc=manifest.runner_identity_qpc,
        qpc_frequency=manifest.qpc_frequency,
        cpu_set_topology_sha256=topology_sha,
        system_contract_sha256=system_sha,
        registrations=rows,
    )
end

role_manifest_json(manifest::EvidenceRoleManifest) =
    String(JSON3.write(_validated_manifest_payload(manifest)))

role_manifest_sha256(manifest::EvidenceRoleManifest) = bytes2hex(SHA.sha256(
    Vector{UInt8}(codeunits(role_manifest_json(manifest))),
))

function _validate_snapshot_ack(ack::SnapshotAck, phase::Symbol)
    ack.phase === phase || throw(ArgumentError("counter snapshot phase mismatch"))
    ack.capability == _IMC_CAPABILITY || throw(ArgumentError(
        "counter callback is not the audited Intel PCM raw adapter",
    ))
    0 <= ack.qpc_begin <= ack.qpc_end || throw(ArgumentError(
        "counter snapshot QPC bracket is invalid",
    ))
    _hash(ack.raw_artifact_sha256, "raw_artifact_sha256")
    _hash(ack.counter_inventory_sha256, "counter_inventory_sha256")
    _hash(ack.counter_configuration_sha256, "counter_configuration_sha256")
    ack.fallback_used && throw(ArgumentError("counter snapshot callback used a fallback"))
    return ack
end

"""Place an exact timed region between independent raw-counter snapshots.

`before_snapshot(manifest_sha)` and `after_snapshot(manifest_sha)` are supplied
by the separately hashed Intel PCM adapter. The runner does not implement a
no-op controller: missing callbacks throw before timed work. The returned QPC
witness proves that the pre-snapshot completed before the region and the
post-snapshot began after it. The raw adapter still must pass the independent
overflow/multiplex/full-channel validator; this hook alone authorizes nothing.
"""
function run_with_counter_snapshot_handshake(work::Function,
                                             manifest::EvidenceRoleManifest,
                                             before_snapshot::Function,
                                             after_snapshot::Function)
    manifest_sha = role_manifest_sha256(manifest)
    before = _validate_snapshot_ack(before_snapshot(manifest_sha), :before)
    timed_begin = _qpc_now()
    before.qpc_end <= timed_begin || throw(ArgumentError(
        "pre-counter snapshot did not complete before timed work",
    ))
    result = work()
    timed_end = _qpc_now()
    timed_begin < timed_end || error("timed work has no positive QPC duration")
    after = _validate_snapshot_ack(after_snapshot(manifest_sha), :after)
    timed_end <= after.qpc_begin || throw(ArgumentError(
        "post-counter snapshot began before timed work completed",
    ))
    before.counter_inventory_sha256 == after.counter_inventory_sha256 ||
        throw(ArgumentError("IMC counter inventory changed across timed work"))
    before.counter_configuration_sha256 == after.counter_configuration_sha256 ||
        throw(ArgumentError("IMC counter configuration changed across timed work"))
    before.program_generation == after.program_generation || throw(ArgumentError(
        "IMC counters were reprogrammed across timed work",
    ))
    witness = TimedRegionEvidenceWitness(
        role_manifest_sha256=manifest_sha,
        before_snapshot=before,
        timed_begin_qpc=timed_begin,
        timed_end_qpc=timed_end,
        after_snapshot=after,
    )
    return result, witness
end

end # module WindowsEvidenceRuntimeHook
