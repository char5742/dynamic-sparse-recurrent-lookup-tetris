export PipelineSlotState,
       FREE,
       PACKED,
       HASH_INFLIGHT,
       HASH_READY,
       SPARSE_DONE,
       STEM_TRAIN_DONE,
       PipelineSlot,
       PipelineTicket,
       PipelineRing,
       QPCStageTiming,
       qpc_now,
       qpc_frequency,
       qpc_seconds,
       transition_slot!,
       reset_slot!,
       try_acquire_free_slot!,
       try_acquire_cpu_slot!,
       try_acquire_npu_slot!,
       try_acquire_probation_npu_slot!,
       cancel_slot_reservation!,
       cancel_npu_slot_reservation!,
       cancel_hash_slot_reservation!,
       begin_sparse_stage!,
       mark_sparse_substage!,
       begin_stem_train_stage!,
       stage_timing_record,
       release_slot_with_record!,
       pipeline_role_contract,
       pipeline_fallback_contract,
       VersionCoordinator,
       finish_hash_request!,
       begin_first_adoption_probation!,
       complete_first_adoption_probation!,
       abort_first_adoption_probation!,
       begin_snapshot_refresh!,
       require_old_lineage_drained!,
       fallback_to_cpu!,
       restore_authorized_snapshot!,
       publish_snapshot!

@enum PipelineSlotState::UInt8 begin
    FREE = 0
    PACKED = 1
    HASH_INFLIGHT = 2
    HASH_READY = 3
    SPARSE_DONE = 4
    STEM_TRAIN_DONE = 5
end

const _NEXT_PIPELINE_STATE = Dict(
    FREE => PACKED,
    PACKED => HASH_INFLIGHT,
    HASH_INFLIGHT => HASH_READY,
    HASH_READY => SPARSE_DONE,
    SPARSE_DONE => STEM_TRAIN_DONE,
    STEM_TRAIN_DONE => FREE,
)

"""All stage boundaries use QueryPerformanceCounter ticks on Windows."""
Base.@kwdef mutable struct QPCStageTiming
    frequency::Int64 = 0
    acquired::Int64 = 0
    pack_begin::Int64 = 0
    pack_end::Int64 = 0
    hash_submit::Int64 = 0
    hash_complete::Int64 = 0
    sparse_begin::Int64 = 0
    route_begin::Int64 = 0
    route_end::Int64 = 0
    gather_begin::Int64 = 0
    gather_end::Int64 = 0
    sparse_compute_begin::Int64 = 0
    sparse_compute_end::Int64 = 0
    sparse_update_begin::Int64 = 0
    sparse_update_end::Int64 = 0
    sparse_end::Int64 = 0
    stem_train_begin::Int64 = 0
    stem_train_end::Int64 = 0
    released::Int64 = 0
end

function qpc_now()
    Sys.iswindows() || throw(ArgumentError("QPC timing is Windows-only"))
    value = Ref{Int64}(0)
    success = ccall((:QueryPerformanceCounter, "Kernel32"), Int32, (Ref{Int64},), value)
    success == 0 && error("QueryPerformanceCounter failed")
    return value[]
end

function qpc_frequency()
    Sys.iswindows() || throw(ArgumentError("QPC timing is Windows-only"))
    value = Ref{Int64}(0)
    success = ccall((:QueryPerformanceFrequency, "Kernel32"), Int32, (Ref{Int64},), value)
    success == 0 && error("QueryPerformanceFrequency failed")
    value[] > 0 || error("invalid QPC frequency")
    return value[]
end

qpc_seconds(start_tick::Int64, end_tick::Int64, frequency::Int64) =
    (end_tick - start_tick) / frequency

"""One bounded pipeline slot; payload buffers are allocated once and reused."""
mutable struct PipelineSlot
    slot_id::Int
    generation::UInt64
    state::PipelineSlotState
    sequence::UInt64
    candidate_count::Int
    snapshot_version::UInt64
    master_superstep::UInt64
    sparse_bank_version::UInt64
    sparse_index_version::UInt64
    reserved::Bool
    hash_backend::Symbol
    stem_train_backend::Symbol
    packed_input::Matrix{Float32}
    hash_output::Matrix{Float32}
    hash_output_gradient::Matrix{Float32}
    timing::QPCStageTiming
    lock::ReentrantLock
end

struct PipelineTicket
    slot_id::Int
    generation::UInt64
    sequence::UInt64
    snapshot_version::UInt64
end

_ticket_key(ticket::PipelineTicket) = (
    ticket.slot_id,
    ticket.generation,
    ticket.sequence,
    ticket.snapshot_version,
)

function _validate_ticket_locked(slot::PipelineSlot, ticket::PipelineTicket)
    slot.slot_id == ticket.slot_id || throw(ArgumentError("pipeline ticket slot mismatch"))
    slot.generation == ticket.generation || throw(ArgumentError("stale pipeline generation"))
    slot.sequence == ticket.sequence || throw(ArgumentError("pipeline ticket sequence mismatch"))
    slot.snapshot_version == ticket.snapshot_version ||
        throw(ArgumentError("pipeline ticket snapshot mismatch"))
    return nothing
end

function PipelineSlot(slot_id::Integer)
    slot_id > 0 || throw(ArgumentError("slot_id must be positive"))
    return PipelineSlot(
        Int(slot_id),
        UInt64(0),
        FREE,
        UInt64(0),
        0,
        UInt64(0),
        UInt64(0),
        UInt64(0),
        UInt64(0),
        false,
        :UNSET,
        :UNSET,
        Matrix{Float32}(undef, HASHSTEM_BATCH, HASHSTEM_INPUT_FEATURES),
        Matrix{Float32}(undef, HASHSTEM_BATCH, HASHSTEM_OUTPUT_FEATURES),
        Matrix{Float32}(undef, HASHSTEM_BATCH, HASHSTEM_OUTPUT_FEATURES),
        QPCStageTiming(),
        ReentrantLock(),
    )
end

mutable struct PipelineRing
    slots::Vector{PipelineSlot}
    next_sequence::UInt64
end

function PipelineRing(slot_count::Integer=4; next_sequence::Integer=1)
    slot_count >= 3 || throw(ArgumentError("pipeline requires at least three slots"))
    next_sequence > 0 || throw(ArgumentError("next_sequence must be positive"))
    return PipelineRing(
        [PipelineSlot(index) for index in 1:slot_count],
        UInt64(next_sequence),
    )
end

function _stamp_transition!(timing::QPCStageTiming, destination::PipelineSlotState, tick::Int64)
    if destination == PACKED
        timing.pack_end = tick
    elseif destination == HASH_INFLIGHT
        timing.hash_submit = tick
    elseif destination == HASH_READY
        timing.hash_complete = tick
    elseif destination == SPARSE_DONE
        timing.sparse_end = tick
    elseif destination == STEM_TRAIN_DONE
        timing.stem_train_end = tick
    elseif destination == FREE
        timing.released = tick
    end
    return nothing
end

"""Perform the only legal next state transition under the slot lock."""
function transition_slot!(
    slot::PipelineSlot,
    ticket::PipelineTicket,
    destination::PipelineSlotState;
    tick=qpc_now(),
)
    lock(slot.lock) do
        _validate_ticket_locked(slot, ticket)
        destination == FREE && throw(ArgumentError(
            "use release_slot_with_record! so the final QPC record is captured atomically",
        ))
        expected = _NEXT_PIPELINE_STATE[slot.state]
        destination == expected || throw(ArgumentError(
            "illegal pipeline transition $(slot.state) -> $destination; expected $expected",
        ))
        if destination == PACKED
            slot.reserved || throw(ArgumentError("FREE slot was not reserved by a packer"))
            0 < slot.timing.pack_begin <= Int64(tick) ||
                throw(ArgumentError("pack timing is not monotonic"))
            slot.reserved = false
        elseif destination == HASH_INFLIGHT
            0 < slot.timing.pack_end <= Int64(tick) ||
                throw(ArgumentError("HashStem submit timing is not monotonic"))
        elseif destination == HASH_READY
            0 < slot.timing.hash_submit <= Int64(tick) ||
                throw(ArgumentError("HashStem completion timing is not monotonic"))
        elseif destination == SPARSE_DONE
            t = slot.timing
            all(value -> value > 0, (
                t.sparse_begin,
                t.route_begin,
                t.route_end,
                t.gather_begin,
                t.gather_end,
                t.sparse_compute_begin,
                t.sparse_compute_end,
                t.sparse_update_begin,
                t.sparse_update_end,
            )) || throw(ArgumentError("sparse substage timing is incomplete"))
            issorted((
                t.sparse_begin,
                t.route_begin,
                t.route_end,
                t.gather_begin,
                t.gather_end,
                t.sparse_compute_begin,
                t.sparse_compute_end,
                t.sparse_update_begin,
                t.sparse_update_end,
                Int64(tick),
            )) || throw(ArgumentError("sparse substage timing is not monotonic"))
        elseif destination == STEM_TRAIN_DONE
            0 < slot.timing.stem_train_begin <= Int64(tick) ||
                throw(ArgumentError("stem training timing is not monotonic"))
        end
        _stamp_transition!(slot.timing, destination, Int64(tick))
        slot.state = destination
    end
    return slot
end

function reset_slot!(slot::PipelineSlot, ticket::PipelineTicket)
    lock(slot.lock) do
        _validate_ticket_locked(slot, ticket)
        slot.state == FREE || throw(ArgumentError("only a FREE slot may be reset"))
        slot.reserved && throw(ArgumentError("cannot reset a slot reserved by a packer"))
        slot.sequence = UInt64(0)
        slot.candidate_count = 0
        slot.snapshot_version = UInt64(0)
        slot.master_superstep = UInt64(0)
        slot.sparse_bank_version = UInt64(0)
        slot.sparse_index_version = UInt64(0)
        slot.hash_backend = :UNSET
        slot.stem_train_backend = :UNSET
        fill!(slot.packed_input, 0.0f0)
        fill!(slot.hash_output, 0.0f0)
        fill!(slot.hash_output_gradient, 0.0f0)
        slot.timing = QPCStageTiming()
    end
    return slot
end

"""Nonblocking E-core acquisition. The caller packs before committing PACKED."""
function _try_acquire_slot!(
    ring::PipelineRing;
    sequence::Integer,
    candidate_count::Integer,
    snapshot_version::Integer,
    master_superstep::Integer,
    sparse_bank_version::Integer,
    sparse_index_version::Integer,
    hash_backend::Symbol,
)
    1 <= candidate_count <= HASHSTEM_BATCH ||
        throw(ArgumentError("candidate_count must be in 1:16"))
    hash_backend in (:CPU, :NPU) || throw(ArgumentError("hash backend must be CPU or NPU"))
    sequence_u64 = UInt64(sequence)
    snapshot_version_u64 = UInt64(snapshot_version)
    master_superstep_u64 = UInt64(master_superstep)
    sparse_bank_version_u64 = UInt64(sparse_bank_version)
    sparse_index_version_u64 = UInt64(sparse_index_version)
    frequency = qpc_frequency()
    tick = qpc_now()
    for slot in ring.slots
        ticket = lock(slot.lock) do
            slot.state == FREE && !slot.reserved || return nothing
            slot.reserved = true
            slot.generation += UInt64(1)
            slot.sequence = sequence_u64
            slot.candidate_count = Int(candidate_count)
            slot.snapshot_version = snapshot_version_u64
            slot.master_superstep = master_superstep_u64
            slot.sparse_bank_version = sparse_bank_version_u64
            slot.sparse_index_version = sparse_index_version_u64
            slot.hash_backend = hash_backend
            slot.stem_train_backend = :UNSET
            slot.timing = QPCStageTiming(frequency=frequency, acquired=tick, pack_begin=tick)
            return PipelineTicket(
                slot.slot_id,
                slot.generation,
                slot.sequence,
                slot.snapshot_version,
            )
        end
        ticket === nothing || return (slot=slot, ticket=ticket)
    end
    return nothing
end

function try_acquire_free_slot!(
    ring::PipelineRing;
    sequence::Integer,
    candidate_count::Integer,
    snapshot_version::Integer=0,
    master_superstep::Integer,
    sparse_bank_version::Integer,
    sparse_index_version::Integer=sparse_bank_version,
)
    throw(ArgumentError(
        "uncoordinated HashStem admission is disabled; use try_acquire_cpu_slot! " *
        "or a coordinator-bound NPU admission path",
    )
end

"""Release a packer reservation after packing fails, without publishing data."""
function cancel_slot_reservation!(slot::PipelineSlot, ticket::PipelineTicket)
    throw(ArgumentError(
        "uncoordinated reservation cancellation is disabled; use " *
        "cancel_hash_slot_reservation!",
    ))
end

function begin_sparse_stage!(slot::PipelineSlot, ticket::PipelineTicket; tick=qpc_now())
    lock(slot.lock) do
        _validate_ticket_locked(slot, ticket)
        slot.state == HASH_READY || throw(ArgumentError("sparse stage requires HASH_READY"))
        slot.timing.sparse_begin == 0 || throw(ArgumentError("sparse stage already began"))
        slot.timing.hash_complete <= Int64(tick) ||
            throw(ArgumentError("sparse stage precedes HashStem completion"))
        slot.timing.sparse_begin = Int64(tick)
    end
    return slot
end

const _SPARSE_SUBSTAGE_FIELDS = (
    :route_begin,
    :route_end,
    :gather_begin,
    :gather_end,
    :sparse_compute_begin,
    :sparse_compute_end,
    :sparse_update_begin,
    :sparse_update_end,
)

function mark_sparse_substage!(
    slot::PipelineSlot,
    ticket::PipelineTicket,
    field::Symbol;
    tick=qpc_now(),
)
    field in _SPARSE_SUBSTAGE_FIELDS || throw(ArgumentError("unknown sparse substage $field"))
    lock(slot.lock) do
        _validate_ticket_locked(slot, ticket)
        slot.state == HASH_READY || throw(ArgumentError("sparse substages require HASH_READY"))
        slot.timing.sparse_begin > 0 || throw(ArgumentError("sparse stage did not begin"))
        getfield(slot.timing, field) == 0 || throw(ArgumentError("$field was already recorded"))
        setfield!(slot.timing, field, Int64(tick))
    end
    return slot
end

function begin_stem_train_stage!(
    slot::PipelineSlot,
    ticket::PipelineTicket;
    backend::Symbol=:CPU,
    tick=qpc_now(),
)
    backend in (:CPU, :IGPU) || throw(ArgumentError("stem backend must be CPU or IGPU"))
    lock(slot.lock) do
        _validate_ticket_locked(slot, ticket)
        slot.state == SPARSE_DONE || throw(ArgumentError("stem training requires SPARSE_DONE"))
        slot.timing.stem_train_begin == 0 || throw(ArgumentError("stem training already began"))
        slot.timing.sparse_end <= Int64(tick) ||
            throw(ArgumentError("stem training precedes sparse completion"))
        slot.stem_train_backend = backend
        slot.timing.stem_train_begin = Int64(tick)
    end
    return slot
end

function _duration_ns(start_tick::Int64, end_tick::Int64, frequency::Int64)
    start_tick > 0 && end_tick >= start_tick && frequency > 0 || return nothing
    return round(Int64, (end_tick - start_tick) * 1.0e9 / frequency)
end

"""JSON-ready timing record. Queueing, packing, submit/wait, and copies remain visible."""
function _stage_timing_record_unlocked(slot::PipelineSlot)
    t = slot.timing
    return (
        schema="heterogeneous-265k-qpc-stage-v1",
        slot_id=slot.slot_id,
        slot_generation=slot.generation,
        sequence=slot.sequence,
        candidate_count=slot.candidate_count,
        snapshot_version=slot.snapshot_version,
        master_superstep=slot.master_superstep,
        sparse_bank_version=slot.sparse_bank_version,
        sparse_index_version=slot.sparse_index_version,
        hash_backend=slot.hash_backend,
        stem_train_backend=slot.stem_train_backend,
        qpc_frequency=t.frequency,
        acquired=t.acquired,
        pack_begin=t.pack_begin,
        pack_end=t.pack_end,
        hash_submit=t.hash_submit,
        hash_complete=t.hash_complete,
        sparse_begin=t.sparse_begin,
        route_begin=t.route_begin,
        route_end=t.route_end,
        gather_begin=t.gather_begin,
        gather_end=t.gather_end,
        sparse_compute_begin=t.sparse_compute_begin,
        sparse_compute_end=t.sparse_compute_end,
        sparse_update_begin=t.sparse_update_begin,
        sparse_update_end=t.sparse_update_end,
        sparse_end=t.sparse_end,
        stem_train_begin=t.stem_train_begin,
        stem_train_end=t.stem_train_end,
        released=t.released,
        pack_ns=_duration_ns(t.pack_begin, t.pack_end, t.frequency),
        hash_queue_ns=_duration_ns(t.pack_end, t.hash_submit, t.frequency),
        hash_submit_wait_copy_ns=_duration_ns(t.hash_submit, t.hash_complete, t.frequency),
        sparse_queue_ns=_duration_ns(t.hash_complete, t.sparse_begin, t.frequency),
        sparse_ns=_duration_ns(t.sparse_begin, t.sparse_end, t.frequency),
        route_ns=_duration_ns(t.route_begin, t.route_end, t.frequency),
        gather_ns=_duration_ns(t.gather_begin, t.gather_end, t.frequency),
        sparse_compute_ns=_duration_ns(
            t.sparse_compute_begin,
            t.sparse_compute_end,
            t.frequency,
        ),
        sparse_update_ns=_duration_ns(
            t.sparse_update_begin,
            t.sparse_update_end,
            t.frequency,
        ),
        stem_train_queue_ns=_duration_ns(t.sparse_end, t.stem_train_begin, t.frequency),
        stem_train_ns=_duration_ns(t.stem_train_begin, t.stem_train_end, t.frequency),
        end_to_end_ns=_duration_ns(t.acquired, t.released, t.frequency),
    )
end

function stage_timing_record(slot::PipelineSlot, ticket::PipelineTicket)
    return lock(slot.lock) do
        _validate_ticket_locked(slot, ticket)
        return _stage_timing_record_unlocked(slot)
    end
end

"""Atomically stamp release, snapshot all timing fields, and expose the FREE slot."""
function release_slot_with_record!(slot::PipelineSlot, ticket::PipelineTicket; tick=qpc_now())
    return lock(slot.lock) do
        _validate_ticket_locked(slot, ticket)
        slot.state == STEM_TRAIN_DONE ||
            throw(ArgumentError("release requires STEM_TRAIN_DONE"))
        0 < slot.timing.stem_train_end <= Int64(tick) ||
            throw(ArgumentError("release timing is not monotonic"))
        _stamp_transition!(slot.timing, FREE, Int64(tick))
        record = _stage_timing_record_unlocked(slot)
        slot.state = FREE
        return record
    end
end

pipeline_role_contract() = (
    windows_cpu_sets=(
        status=:UNAPPLIED_UNVERIFIED,
        discovery="GetSystemCpuSetInformation at every process launch",
        classification="fail closed unless exactly two classes resolve to 8 higher-class and 12 lower-class one-thread cores",
        process_assignment="SetProcessDefaultCpuSets plus exact GetProcessDefaultCpuSets readback",
        thread_assignment="SetThreadSelectedCpuSets plus exact GetThreadSelectedCpuSets readback",
        point_witness="GetCurrentProcessorNumberEx; ETW whole-stage residency is still required",
        prohibition="CPU Set IDs are opaque; never derive them from Julia thread IDs or logical CPU numbers",
    ),
    e_cores=(
        ownership=(
            :environment,
            :candidate_generation,
            :replay_packing,
            :index_rebuild,
            :prefetch,
            :logging,
        ),
        rule="use dynamically discovered lower-EfficiencyClass CPU Sets; leave one unassigned to application workers, without claiming an OS/driver reservation",
    ),
    npu=(
        ownership=(:fixed_batch_16_hashstem, :legacy_teacher, :dense_actor),
        rule="one inference broker; no sparse backward or optimizer work",
    ),
    p_cores=(
        ownership=(
            :wta_lsh,
            :deduplication,
            :exact_rerank,
            :irregular_gather,
            :sparse_forward,
            :sparse_vjp,
            :lazy_sparse_optimizer,
            :final_head,
        ),
        rule="latency-critical active-only work on dynamically discovered higher-EfficiencyClass CPU Sets; no whole-bank traversal",
    ),
    igpu=(
        ownership=(:optional_hashstem_training,),
        rule="disabled until >=1.15x end-to-end and <=1.10x P-core sparse slowdown gates pass",
    ),
    ram=(
        ownership=(:neuron_banks, :sparse_optimizer, :replay, :lsh_indices, :checkpoints),
        rule="fixed buffers and bounded rings; report gathered bytes and shared-bandwidth contention",
    ),
)

pipeline_fallback_contract() = (
    hashstem=(
        primary="NPU fixed batch 16",
        tail="deterministic CPU reference at actual length",
        fallback="CPU HashStem for all rows",
        fallback_if=(
            :npu_unavailable,
            :snapshot_binding_failure,
            :numeric_gate_failure,
            :route_id_mismatch,
            :top1_mismatch,
            :speedup_below_1_15,
            :p95_regression,
        ),
    ),
    stem_training=(
        primary="CPU master",
        optional="iGPU only after isolated and concurrent systems gates",
        fallback="P-core CPU master training",
    ),
    active_block_offload=(
        adopted=false,
        reason="three dependent sparse layers would require repeated pack/submit/wait/scatter",
        experiment_only=true,
        adoption_gate="full boundary >=1.15x, p95 no worse, route IDs and outputs unchanged",
    ),
)

"""Coordinates an inference snapshot frozen for a complete training superstep.

Stem gradients are accumulated against the same frozen weights that produced
the NPU outputs. A refresh first closes admission, drains old requests and all
slots that reference the old snapshot, applies the CPU/iGPU master update,
exports and verifies a new snapshot, then atomically publishes its version.
"""
mutable struct VersionCoordinator
    model_id::String
    master_version::UInt64
    published_snapshot_version::UInt64
    npu_adopted::Bool
    ever_adopted::Bool
    accepting_npu::Bool
    accepting_cpu::Bool
    probation_open::Bool
    probation_snapshot_version::UInt64
    refresh_pending::Bool
    inflight::Dict{UInt64,Int}
    active_tickets::Set{Tuple{Int,UInt64,UInt64,UInt64}}
    lock::ReentrantLock
end

function VersionCoordinator(model_id::AbstractString; master_version=0, snapshot_version=0)
    isempty(model_id) && throw(ArgumentError("model_id must not be empty"))
    return VersionCoordinator(
        String(model_id),
        UInt64(master_version),
        UInt64(snapshot_version),
        false,
        false,
        false,
        false,
        false,
        UInt64(0),
        false,
        Dict{UInt64,Int}(),
        Set{Tuple{Int,UInt64,UInt64,UInt64}}(),
        ReentrantLock(),
    )
end

function _register_hash_reservation_locked!(
    coordinator::VersionCoordinator,
    ring::PipelineRing,
    reservation,
)
    reservation === nothing && return nothing
    ticket_key = _ticket_key(reservation.ticket)
    ticket_key in coordinator.active_tickets && error("duplicate active pipeline ticket")
    push!(coordinator.active_tickets, ticket_key)
    version = reservation.ticket.snapshot_version
    coordinator.inflight[version] = get(coordinator.inflight, version, 0) + 1
    ring.next_sequence == typemax(UInt64) && error("pipeline sequence overflow")
    ring.next_sequence += UInt64(1)
    return reservation
end

function _admit_hash_slot_locked!(
    ring::PipelineRing,
    coordinator::VersionCoordinator;
    sequence::Integer,
    candidate_count::Integer,
    snapshot_version::UInt64,
    master_superstep::Integer,
    sparse_bank_version::Integer,
    sparse_index_version::Integer,
    hash_backend::Symbol,
)
    UInt64(sequence) == ring.next_sequence || throw(ArgumentError(
        "pipeline sequence must equal the coordinator-bound ring cursor",
    ))
    reservation = _try_acquire_slot!(
        ring;
        sequence,
        candidate_count,
        snapshot_version,
        master_superstep,
        sparse_bank_version,
        sparse_index_version,
        hash_backend,
    )
    return _register_hash_reservation_locked!(coordinator, ring, reservation)
end

"""Coordinator-bound actual-length CPU tail/fallback admission.

CPU and NPU chunks carry the same nonzero published snapshot lineage. This
path participates in the same in-flight/ticket drain and cannot cross refresh.
"""
function try_acquire_cpu_slot!(
    ring::PipelineRing,
    coordinator::VersionCoordinator;
    sequence::Integer,
    candidate_count::Integer,
    master_superstep::Integer,
    sparse_bank_version::Integer,
    sparse_index_version::Integer=sparse_bank_version,
)
    return lock(coordinator.lock) do
        coordinator.accepting_cpu || return nothing
        coordinator.refresh_pending && return nothing
        version = coordinator.published_snapshot_version
        version > 0 || throw(ArgumentError("CPU tail requires a published snapshot lineage"))
        return _admit_hash_slot_locked!(
            ring,
            coordinator;
            sequence,
            candidate_count,
            snapshot_version=version,
            master_superstep,
            sparse_bank_version,
            sparse_index_version,
            hash_backend=:CPU,
        )
    end
end

"""Atomically admit and reserve an NPU-bound slot under coordinator->slot lock order.

This is the required NPU path. `try_acquire_free_slot!` alone is reserved for
CPU-tail/fallback work because it cannot close the refresh-admission race.
"""
function try_acquire_npu_slot!(
    ring::PipelineRing,
    coordinator::VersionCoordinator;
    sequence::Integer,
    candidate_count::Integer=HASHSTEM_BATCH,
    master_superstep::Integer,
    sparse_bank_version::Integer,
    sparse_index_version::Integer=sparse_bank_version,
)
    candidate_count == HASHSTEM_BATCH ||
        throw(ArgumentError("NPU HashStem admission requires exactly 16 candidates"))
    return lock(coordinator.lock) do
        coordinator.npu_adopted || return nothing
        coordinator.ever_adopted || error("adopted NPU state lost its permanent lineage bit")
        coordinator.probation_open && error("normal NPU admission cannot overlap probation")
        coordinator.accepting_npu || return nothing
        coordinator.refresh_pending && return nothing
        version = coordinator.published_snapshot_version
        return _admit_hash_slot_locked!(
            ring,
            coordinator;
            sequence,
            candidate_count,
            snapshot_version=version,
            master_superstep,
            sparse_bank_version,
            sparse_index_version,
            hash_backend=:NPU,
        )
    end
end

"""First-adoption-only NPU admission; this never flips an adoption bit."""
function try_acquire_probation_npu_slot!(
    ring::PipelineRing,
    coordinator::VersionCoordinator;
    sequence::Integer,
    candidate_count::Integer=HASHSTEM_BATCH,
    master_superstep::Integer,
    sparse_bank_version::Integer,
    sparse_index_version::Integer=sparse_bank_version,
)
    candidate_count == HASHSTEM_BATCH || throw(ArgumentError(
        "probation NPU HashStem admission requires exactly 16 candidates",
    ))
    return lock(coordinator.lock) do
        coordinator.ever_adopted && throw(ArgumentError(
            "first-adoption probation is permanently unavailable after adoption",
        ))
        coordinator.npu_adopted && error("probation cannot run in adopted state")
        coordinator.probation_open || return nothing
        coordinator.accepting_npu || return nothing
        coordinator.refresh_pending && return nothing
        version = coordinator.probation_snapshot_version
        version == coordinator.published_snapshot_version || error(
            "probation snapshot diverged from the published CPU comparator lineage",
        )
        return _admit_hash_slot_locked!(
            ring,
            coordinator;
            sequence,
            candidate_count,
            snapshot_version=version,
            master_superstep,
            sparse_bank_version,
            sparse_index_version,
            hash_backend=:NPU,
        )
    end
end

function cancel_hash_slot_reservation!(
    coordinator::VersionCoordinator,
    slot::PipelineSlot,
    ticket::PipelineTicket,
)
    lock(coordinator.lock) do
        key = ticket.snapshot_version
        ticket_key = _ticket_key(ticket)
        ticket_key in coordinator.active_tickets ||
            throw(ArgumentError("unknown or already completed NPU ticket"))
        count = get(coordinator.inflight, key, 0)
        count > 0 || throw(ArgumentError("missing NPU admission count"))
        lock(slot.lock) do
            _validate_ticket_locked(slot, ticket)
            slot.state == FREE && slot.reserved ||
                throw(ArgumentError("HashStem slot is no longer a packing reservation"))
            slot.reserved = false
            slot.sequence = UInt64(0)
            slot.candidate_count = 0
            slot.snapshot_version = UInt64(0)
            slot.master_superstep = UInt64(0)
            slot.sparse_bank_version = UInt64(0)
            slot.sparse_index_version = UInt64(0)
            slot.hash_backend = :UNSET
            slot.stem_train_backend = :UNSET
            slot.timing = QPCStageTiming()
        end
        delete!(coordinator.active_tickets, ticket_key)
        if count == 1
            delete!(coordinator.inflight, key)
        else
            coordinator.inflight[key] = count - 1
        end
    end
    return slot
end

cancel_npu_slot_reservation!(
    coordinator::VersionCoordinator,
    slot::PipelineSlot,
    ticket::PipelineTicket,
) = cancel_hash_slot_reservation!(coordinator, slot, ticket)

"""Fail closed to CPU HashStem after an unavailable device or failed gate."""
function fallback_to_cpu!(coordinator::VersionCoordinator)
    lock(coordinator.lock) do
        isempty(coordinator.inflight) ||
            throw(ArgumentError("cannot enter CPU fallback with NPU requests in flight"))
        isempty(coordinator.active_tickets) ||
            throw(ArgumentError("cannot enter CPU fallback with active NPU tickets"))
        coordinator.npu_adopted = false
        coordinator.accepting_npu = false
        coordinator.accepting_cpu = coordinator.published_snapshot_version > 0
        coordinator.probation_open = false
        coordinator.probation_snapshot_version = UInt64(0)
        coordinator.refresh_pending = false
    end
    return coordinator
end

function finish_hash_request!(
    coordinator::VersionCoordinator,
    slot::PipelineSlot,
    ticket::PipelineTicket,
)
    lock(coordinator.lock) do
        key = ticket.snapshot_version
        ticket_key = _ticket_key(ticket)
        ticket_key in coordinator.active_tickets ||
            throw(ArgumentError("unknown or already completed NPU ticket"))
        count = get(coordinator.inflight, key, 0)
        count > 0 || throw(ArgumentError("unmatched HashStem request completion"))
        lock(slot.lock) do
            _validate_ticket_locked(slot, ticket)
            slot.state == HASH_READY ||
                throw(ArgumentError("HashStem completion requires HASH_READY"))
        end
        delete!(coordinator.active_tickets, ticket_key)
        if count == 1
            delete!(coordinator.inflight, key)
        else
            coordinator.inflight[key] = count - 1
        end
    end
    return coordinator
end

function begin_snapshot_refresh!(coordinator::VersionCoordinator)
    return lock(coordinator.lock) do
        coordinator.refresh_pending && throw(ArgumentError("refresh already pending"))
        coordinator.probation_open && throw(ArgumentError(
            "snapshot refresh cannot begin during first-adoption probation",
        ))
        coordinator.accepting_npu = false
        coordinator.accepting_cpu = false
        coordinator.refresh_pending = true
        return coordinator.published_snapshot_version
    end
end

function _ring_references_snapshot(ring::PipelineRing, snapshot_version::UInt64)
    for slot in ring.slots
        referenced = lock(slot.lock) do
            (slot.state != FREE || slot.reserved) && slot.snapshot_version == snapshot_version
        end
        referenced && return true
    end
    return false
end

function _ring_has_live_payload(ring::PipelineRing)
    for slot in ring.slots
        live = lock(slot.lock) do
            slot.state != FREE || slot.reserved
        end
        live && return true
    end
    return false
end

function _require_old_lineage_drained_locked!(
    coordinator::VersionCoordinator,
    ring::PipelineRing,
    old_version::UInt64,
)
    coordinator.accepting_npu && throw(ArgumentError("NPU admission is still open"))
    coordinator.accepting_cpu && throw(ArgumentError("CPU-tail admission is still open"))
    get(coordinator.inflight, old_version, 0) == 0 || throw(ArgumentError(
        "old snapshot still has in-flight HashStem requests",
    ))
    isempty(coordinator.inflight) || throw(ArgumentError(
        "in-flight HashStem requests exist outside the old snapshot lineage",
    ))
    isempty(coordinator.active_tickets) || throw(ArgumentError(
        "active HashStem tickets remain at the barrier",
    ))
    _ring_references_snapshot(ring, old_version) && throw(ArgumentError(
        "ring still contains old-snapshot payloads",
    ))
    _ring_has_live_payload(ring) && throw(ArgumentError(
        "whole ring must be drained before a version barrier",
    ))
    return old_version
end

"""Fail-closed whole-ring drain proof for a frozen snapshot lineage."""
function require_old_lineage_drained!(
    coordinator::VersionCoordinator,
    ring::PipelineRing,
    old_version::Integer=coordinator.published_snapshot_version,
)
    return lock(coordinator.lock) do
        UInt64(old_version) == coordinator.published_snapshot_version || throw(
            ArgumentError("drain request does not name the published old snapshot"),
        )
        return _require_old_lineage_drained_locked!(
            coordinator,
            ring,
            UInt64(old_version),
        )
    end
end

function _validate_npu_evidence_binding(
    snapshot::HashStemInferenceSnapshot,
    evidence::NPUAdoptionEvidence,
)
    npu_adoption_passes(evidence) || throw(ArgumentError("NPU adoption gates did not pass"))
    evidence.model_id == snapshot.model_id || throw(ArgumentError("NPU evidence model mismatch"))
    evidence.snapshot_version == snapshot.snapshot_version ||
        throw(ArgumentError("NPU evidence snapshot version mismatch"))
    evidence.weights_sha256 == snapshot.weights_sha256 ||
        throw(ArgumentError("NPU evidence weight digest mismatch"))
    evidence.xml_sha256 == snapshot.xml_sha256 ||
        throw(ArgumentError("NPU evidence XML digest mismatch"))
    evidence.bin_sha256 == snapshot.bin_sha256 ||
        throw(ArgumentError("NPU evidence BIN digest mismatch"))
    evidence.snapshot_metadata_sha256 == snapshot.metadata_sha256 ||
        throw(ArgumentError("NPU evidence metadata digest mismatch"))
    return evidence
end

"""Open the only pre-adoption NPU path without recording adoption.

The supplied snapshot is already the CPU comparator lineage. Both actual-size
CPU tails and probation NPU batches are coordinator-bound to that exact version.
"""
function begin_first_adoption_probation!(
    coordinator::VersionCoordinator,
    ring::PipelineRing,
    master::HashStemMasterMetadata,
    snapshot::HashStemInferenceSnapshot,
)
    validate_snapshot_binding(master, snapshot)
    return lock(coordinator.lock) do
        coordinator.ever_adopted && throw(ArgumentError(
            "first-adoption probation is permanently unavailable after adoption",
        ))
        coordinator.npu_adopted && error("adopted coordinator cannot enter probation")
        coordinator.probation_open && throw(ArgumentError("probation is already open"))
        coordinator.refresh_pending && throw(ArgumentError("refresh is pending"))
        isempty(coordinator.inflight) || throw(ArgumentError("probation has in-flight work"))
        isempty(coordinator.active_tickets) || throw(ArgumentError("probation has active tickets"))
        _ring_has_live_payload(ring) && throw(ArgumentError("probation ring is not drained"))
        coordinator.model_id == master.model_id || throw(ArgumentError("model ID mismatch"))
        coordinator.master_version == master.master_version || throw(ArgumentError(
            "probation master version mismatch",
        ))
        coordinator.published_snapshot_version == snapshot.snapshot_version || throw(
            ArgumentError("probation snapshot is not the coordinator comparator lineage"),
        )
        coordinator.probation_open = true
        coordinator.probation_snapshot_version = snapshot.snapshot_version
        coordinator.accepting_cpu = true
        coordinator.accepting_npu = true
        return coordinator
    end
end

"""Adopt only after bound evidence passes and all probation work is drained."""
function complete_first_adoption_probation!(
    coordinator::VersionCoordinator,
    ring::PipelineRing,
    snapshot::HashStemInferenceSnapshot,
    evidence::NPUAdoptionEvidence,
)
    _validate_npu_evidence_binding(snapshot, evidence)
    return lock(coordinator.lock) do
        coordinator.ever_adopted && throw(ArgumentError(
            "first-adoption probation cannot be reused after adoption",
        ))
        coordinator.probation_open || throw(ArgumentError("probation is not open"))
        coordinator.probation_snapshot_version == snapshot.snapshot_version || throw(
            ArgumentError("probation completion snapshot mismatch"),
        )
        # Close both admission paths before testing the drain. A failed drain
        # leaves probation open but closed so callers may finish outstanding work.
        coordinator.accepting_npu = false
        coordinator.accepting_cpu = false
        _require_old_lineage_drained_locked!(
            coordinator,
            ring,
            coordinator.probation_snapshot_version,
        )
        coordinator.npu_adopted = true
        coordinator.ever_adopted = true
        coordinator.probation_open = false
        coordinator.probation_snapshot_version = UInt64(0)
        coordinator.accepting_cpu = true
        coordinator.accepting_npu = true
        return coordinator
    end
end

function abort_first_adoption_probation!(
    coordinator::VersionCoordinator,
    ring::PipelineRing,
)
    return lock(coordinator.lock) do
        coordinator.ever_adopted && throw(ArgumentError(
            "probation abort is unavailable after adoption",
        ))
        coordinator.probation_open || throw(ArgumentError("probation is not open"))
        coordinator.accepting_npu = false
        coordinator.accepting_cpu = false
        _require_old_lineage_drained_locked!(
            coordinator,
            ring,
            coordinator.probation_snapshot_version,
        )
        coordinator.probation_open = false
        coordinator.probation_snapshot_version = UInt64(0)
        coordinator.npu_adopted = false
        coordinator.accepting_cpu = coordinator.published_snapshot_version > 0
        return coordinator
    end
end

"""Reauthorize an exact checkpoint-restored snapshot; never trusts version alone."""
function restore_authorized_snapshot!(
    coordinator::VersionCoordinator,
    ring::PipelineRing,
    master::HashStemMasterMetadata,
    snapshot::HashStemInferenceSnapshot,
    evidence::NPUAdoptionEvidence,
)
    validate_snapshot_binding(master, snapshot)
    _validate_npu_evidence_binding(snapshot, evidence)
    lock(coordinator.lock) do
        coordinator.model_id == master.model_id || throw(ArgumentError("model ID mismatch"))
        coordinator.master_version == master.master_version ||
            throw(ArgumentError("restored master version mismatch"))
        coordinator.published_snapshot_version == snapshot.snapshot_version ||
            throw(ArgumentError("restored snapshot version mismatch"))
        isempty(coordinator.inflight) || throw(ArgumentError("restore has in-flight requests"))
        isempty(coordinator.active_tickets) || throw(ArgumentError("restore has active tickets"))
        _ring_has_live_payload(ring) &&
            throw(ArgumentError("restore ring is not drained"))
        coordinator.refresh_pending && throw(ArgumentError("restore has pending refresh"))
        coordinator.npu_adopted = true
        coordinator.ever_adopted = true
        coordinator.probation_open = false
        coordinator.probation_snapshot_version = UInt64(0)
        coordinator.accepting_npu = true
        coordinator.accepting_cpu = true
    end
    return coordinator
end

function publish_snapshot!(
    coordinator::VersionCoordinator,
    ring::PipelineRing,
    master::HashStemMasterMetadata,
    snapshot::HashStemInferenceSnapshot,
    evidence::NPUAdoptionEvidence,
)
    validate_snapshot_binding(master, snapshot)
    _validate_npu_evidence_binding(snapshot, evidence)
    lock(coordinator.lock) do
        coordinator.refresh_pending || throw(ArgumentError("refresh was not begun"))
        old_version = coordinator.published_snapshot_version
        _require_old_lineage_drained_locked!(coordinator, ring, old_version)
        master.model_id == coordinator.model_id || throw(ArgumentError("model ID mismatch"))
        master.master_version > coordinator.master_version ||
            throw(ArgumentError("master version must increase"))
        snapshot.snapshot_version > old_version ||
            throw(ArgumentError("snapshot version must increase"))
        coordinator.master_version = master.master_version
        coordinator.published_snapshot_version = snapshot.snapshot_version
        coordinator.npu_adopted = true
        coordinator.ever_adopted = true
        coordinator.probation_open = false
        coordinator.probation_snapshot_version = UInt64(0)
        coordinator.refresh_pending = false
        coordinator.accepting_npu = true
        coordinator.accepting_cpu = true
    end
    return coordinator
end
