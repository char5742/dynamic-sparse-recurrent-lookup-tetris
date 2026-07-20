using JLD2
using JSON3
using SHA

export FrozenStemGradientAccumulator,
       SuperstepPublicationBundle,
       begin_frozen_superstep,
       accumulate_stem_gradient!,
       apply_superstep_barrier_update!,
       publish_updated_superstep!,
       run_superstep_barrier!,
       save_heterogeneous_boundary_checkpoint!,
       load_heterogeneous_boundary_checkpoint

const HETEROGENEOUS_BOUNDARY_CHECKPOINT_SCHEMA =
    "heterogeneous-265k-boundary-checkpoint-v1"

@enum FrozenSuperstepPhase::UInt8 begin
    SUPERSTEP_COLLECTING = 1
    SUPERSTEP_DRAINING = 2
    SUPERSTEP_UPDATED_UNPUBLISHED = 3
    SUPERSTEP_PUBLISHED = 4
end

"""A fixed-shape, frozen-master stem-gradient collection buffer.

Rows are accepted in coordinator sequence order. The buffer stores the exact
packed input and selected-only dL/dHashStem-output that were produced under one
snapshot/master/sparse-bank/index lineage. It performs no optimizer update.
"""
mutable struct FrozenStemGradientAccumulator
    model_id::String
    superstep::UInt64
    source_master_version::UInt64
    source_snapshot_version::UInt64
    sparse_bank_version::UInt64
    sparse_index_version::UInt64
    capacity::Int
    used_rows::Int
    next_sequence::UInt64
    seen_sequences::Set{UInt64}
    packed::Matrix{Float32}
    output_cotangent::Matrix{Float32}
    auxiliary_target::Matrix{Float32}
    auxiliary_mask::Matrix{Float32}
    phase::FrozenSuperstepPhase
    barrier_update_count::Int
    destination_master_version::UInt64
    destination_snapshot_version::UInt64
    export_attempts::Int
    export_manifest_sha256::String
end

Base.@kwdef struct SuperstepPublicationBundle
    master::HashStemMasterMetadata
    snapshot::HashStemInferenceSnapshot
    evidence::NPUAdoptionEvidence
    export_manifest_sha256::String
    runtime_validation_status::String = UNEXECUTED_STATIC_ONLY
end

function begin_frozen_superstep(
    trainer::HashStemMasterTrainer,
    coordinator::VersionCoordinator,
    ring::PipelineRing;
    superstep::Integer,
    sparse_bank_version::Integer,
    sparse_index_version::Integer,
)
    superstep > 0 || throw(ArgumentError("superstep must be positive"))
    sparse_bank_version > 0 || throw(ArgumentError("sparse bank version must be positive"))
    sparse_index_version > 0 || throw(ArgumentError("sparse index version must be positive"))
    return lock(coordinator.lock) do
        coordinator.refresh_pending && throw(ArgumentError("refresh is pending"))
        coordinator.probation_open && throw(ArgumentError(
            "training supersteps cannot begin during first-adoption probation",
        ))
        coordinator.published_snapshot_version > 0 || throw(ArgumentError(
            "a frozen superstep requires a published snapshot lineage",
        ))
        trainer.model_id == coordinator.model_id || throw(ArgumentError("model ID mismatch"))
        trainer.master_version == coordinator.master_version || throw(ArgumentError(
            "trainer/coordinator master versions differ",
        ))
        trainer.last_snapshot_version == coordinator.published_snapshot_version || throw(
            ArgumentError("trainer/coordinator snapshot versions differ"),
        )
        trainer.last_snapshot_master_version == trainer.master_version || throw(
            ArgumentError("published snapshot was not exported from the frozen master"),
        )
        _ring_has_live_payload(ring) && throw(ArgumentError(
            "a superstep must start at a drained ring boundary",
        ))
        capacity = trainer.batch_size
        return FrozenStemGradientAccumulator(
            coordinator.model_id,
            UInt64(superstep),
            trainer.master_version,
            coordinator.published_snapshot_version,
            UInt64(sparse_bank_version),
            UInt64(sparse_index_version),
            capacity,
            0,
            ring.next_sequence,
            Set{UInt64}(),
            zeros(Float32, capacity, HASHSTEM_INPUT_FEATURES),
            zeros(Float32, capacity, HASHSTEM_OUTPUT_FEATURES),
            zeros(Float32, capacity, 10),
            zeros(Float32, capacity, 10),
            SUPERSTEP_COLLECTING,
            0,
            UInt64(0),
            UInt64(0),
            0,
            "",
        )
    end
end

"""Copy one completed sparse-VJP chunk into the frozen superstep buffer.

The coordinator lock is always acquired before the slot lock. Completion order
must equal admission sequence, making accumulation and exact resume deterministic.
"""
function accumulate_stem_gradient!(
    accumulator::FrozenStemGradientAccumulator,
    coordinator::VersionCoordinator,
    slot::PipelineSlot,
    ticket::PipelineTicket;
    auxiliary_target::AbstractMatrix{<:Real}=zeros(Float32, slot.candidate_count, 10),
    auxiliary_mask::AbstractMatrix{<:Real}=zeros(Float32, slot.candidate_count, 10),
)
    return lock(coordinator.lock) do
        accumulator.phase == SUPERSTEP_COLLECTING || throw(ArgumentError(
            "stem gradients may only be added while the superstep is collecting",
        ))
        coordinator.refresh_pending && throw(ArgumentError("refresh began before collection ended"))
        lock(slot.lock) do
            _validate_ticket_locked(slot, ticket)
            slot.state == SPARSE_DONE || throw(ArgumentError(
                "stem-gradient collection requires SPARSE_DONE",
            ))
            slot.snapshot_version == accumulator.source_snapshot_version || throw(
                ArgumentError("slot snapshot differs from frozen superstep"),
            )
            slot.master_superstep == accumulator.superstep || throw(ArgumentError(
                "slot master superstep differs from frozen accumulator",
            ))
            slot.sparse_bank_version == accumulator.sparse_bank_version || throw(
                ArgumentError("slot sparse-bank version differs from frozen accumulator"),
            )
            slot.sparse_index_version == accumulator.sparse_index_version || throw(
                ArgumentError("slot sparse-index version differs from frozen accumulator"),
            )
            slot.sequence == accumulator.next_sequence || throw(ArgumentError(
                "stem-gradient chunks must be accumulated in exact admission order",
            ))
            slot.sequence in accumulator.seen_sequences && throw(ArgumentError(
                "pipeline sequence was accumulated twice",
            ))
            rows = slot.candidate_count
            accumulator.used_rows + rows <= accumulator.capacity || throw(ArgumentError(
                "frozen superstep exceeds its fixed master batch capacity",
            ))
            size(auxiliary_target) == (rows, 10) || throw(DimensionMismatch(
                "auxiliary_target must be candidate_count x 10",
            ))
            size(auxiliary_mask) == (rows, 10) || throw(DimensionMismatch(
                "auxiliary_mask must be candidate_count x 10",
            ))
            all(isfinite, view(slot.packed_input, 1:rows, :)) || throw(ArgumentError(
                "slot packed input is non-finite",
            ))
            all(isfinite, view(slot.hash_output_gradient, 1:rows, :)) || throw(
                ArgumentError("slot HashStem output cotangent is non-finite"),
            )
            all(isfinite, auxiliary_target) || throw(ArgumentError(
                "auxiliary target is non-finite",
            ))
            all(value -> value == 0 || value == 1, auxiliary_mask) || throw(
                ArgumentError("auxiliary mask must contain only zero/one"),
            )
            destination = (accumulator.used_rows + 1):(accumulator.used_rows + rows)
            accumulator.packed[destination, :] .= @view slot.packed_input[1:rows, :]
            accumulator.output_cotangent[destination, :] .=
                @view slot.hash_output_gradient[1:rows, :]
            accumulator.auxiliary_target[destination, :] .= Float32.(auxiliary_target)
            accumulator.auxiliary_mask[destination, :] .= Float32.(auxiliary_mask)
            accumulator.used_rows += rows
            push!(accumulator.seen_sequences, slot.sequence)
            accumulator.next_sequence += UInt64(1)
        end
        return accumulator
    end
end

function _frozen_master_batch(accumulator::FrozenStemGradientAccumulator)
    accumulator.used_rows > 0 || throw(ArgumentError("superstep accumulated no rows"))
    mask = zeros(Float32, 1, accumulator.capacity)
    mask[1, 1:accumulator.used_rows] .= 1.0f0
    return (;
        inputs=(; packed=copy(accumulator.packed)),
        targets=(;
            output_cotangent=copy(accumulator.output_cotangent),
            auxiliary_target=copy(accumulator.auxiliary_target),
            auxiliary_mask=copy(accumulator.auxiliary_mask),
        ),
        mask,
    )
end

"""Close both admissions, prove a whole-ring drain, then apply exactly one update."""
function apply_superstep_barrier_update!(
    accumulator::FrozenStemGradientAccumulator,
    trainer::HashStemMasterTrainer,
    coordinator::VersionCoordinator,
    ring::PipelineRing,
)
    accumulator.phase == SUPERSTEP_COLLECTING || throw(ArgumentError(
        "superstep is not in collecting phase",
    ))
    old_version = begin_snapshot_refresh!(coordinator)
    accumulator.phase = SUPERSTEP_DRAINING
    try
        require_old_lineage_drained!(coordinator, ring, old_version)
        accumulator.next_sequence == ring.next_sequence || throw(ArgumentError(
            "ring cursor and accumulated sequence frontier differ",
        ))
        trainer.master_version == accumulator.source_master_version || throw(
            ArgumentError("master changed while the superstep was frozen"),
        )
        coordinator.master_version == accumulator.source_master_version || throw(
            ArgumentError("coordinator master changed while the superstep was frozen"),
        )
        accumulator.barrier_update_count == 0 || error("barrier update already applied")
        result = _apply_hashstem_master_batch!(trainer, _frozen_master_batch(accumulator))
        trainer.master_version == accumulator.source_master_version + UInt64(1) || error(
            "barrier update did not advance the master exactly once",
        )
        accumulator.barrier_update_count = 1
        accumulator.destination_master_version = trainer.master_version
        accumulator.phase = SUPERSTEP_UPDATED_UNPUBLISHED
        return result
    catch
        # Admission stays closed and refresh_pending stays true. If the master
        # update happened, its count/phase prevents a second update.
        rethrow()
    end
end

function publish_updated_superstep!(
    accumulator::FrozenStemGradientAccumulator,
    trainer::HashStemMasterTrainer,
    coordinator::VersionCoordinator,
    ring::PipelineRing,
    bundle::SuperstepPublicationBundle,
)
    accumulator.phase == SUPERSTEP_UPDATED_UNPUBLISHED || throw(ArgumentError(
        "superstep has no single updated master awaiting publication",
    ))
    accumulator.barrier_update_count == 1 || error("barrier update count is not exactly one")
    bundle.runtime_validation_status == VALIDATED_RUNTIME || throw(ArgumentError(
        "publication bundle lacks runtime validation",
    ))
    _valid_sha256(bundle.export_manifest_sha256) || throw(ArgumentError(
        "publication bundle has an invalid export-manifest SHA-256",
    ))
    bundle.master.master_version == accumulator.destination_master_version || throw(
        ArgumentError("publication master is not the one barrier-updated master"),
    )
    trainer.master_version == bundle.master.master_version || throw(ArgumentError(
        "trainer/publication master versions differ",
    ))
    accumulator.export_attempts += 1
    publish_snapshot!(
        coordinator,
        ring,
        bundle.master,
        bundle.snapshot,
        bundle.evidence,
    )
    trainer.last_snapshot_version = bundle.snapshot.snapshot_version
    trainer.last_snapshot_master_version = bundle.master.master_version
    accumulator.destination_snapshot_version = bundle.snapshot.snapshot_version
    accumulator.export_manifest_sha256 = bundle.export_manifest_sha256
    accumulator.phase = SUPERSTEP_PUBLISHED
    return accumulator
end

"""One synchronous barrier update, one export/validation callback, one publication.

No worker is created. If export fails, the master remains in
`UPDATED_UNPUBLISHED` and a caller may retry only `publish_updated_superstep!`;
the optimizer update cannot be repeated.
"""
function run_superstep_barrier!(
    accumulator::FrozenStemGradientAccumulator,
    trainer::HashStemMasterTrainer,
    coordinator::VersionCoordinator,
    ring::PipelineRing,
    export_and_validate::Function,
)
    update = apply_superstep_barrier_update!(accumulator, trainer, coordinator, ring)
    accumulator.export_attempts += 1
    bundle = export_and_validate(trainer, accumulator)
    bundle isa SuperstepPublicationBundle || throw(ArgumentError(
        "export callback must return SuperstepPublicationBundle",
    ))
    # Count the callback once; publication itself should not count another export.
    accumulator.export_attempts -= 1
    publish_updated_superstep!(accumulator, trainer, coordinator, ring, bundle)
    return (; update, bundle, accumulator)
end

function _coordinator_boundary_record(
    coordinator::VersionCoordinator;
    resume_accepting_npu::Bool=coordinator.accepting_npu,
    resume_accepting_cpu::Bool=coordinator.accepting_cpu,
)
    return (;
        model_id=coordinator.model_id,
        master_version=coordinator.master_version,
        published_snapshot_version=coordinator.published_snapshot_version,
        npu_adopted=coordinator.npu_adopted,
        ever_adopted=coordinator.ever_adopted,
        accepting_npu=resume_accepting_npu,
        accepting_cpu=resume_accepting_cpu,
        probation_open=coordinator.probation_open,
        probation_snapshot_version=coordinator.probation_snapshot_version,
        refresh_pending=coordinator.refresh_pending,
    )
end

function _completed_superstep_record(accumulator::FrozenStemGradientAccumulator)
    return (;
        model_id=accumulator.model_id,
        superstep=accumulator.superstep,
        source_master_version=accumulator.source_master_version,
        source_snapshot_version=accumulator.source_snapshot_version,
        sparse_bank_version=accumulator.sparse_bank_version,
        sparse_index_version=accumulator.sparse_index_version,
        used_rows=accumulator.used_rows,
        next_sequence=accumulator.next_sequence,
        phase=UInt8(accumulator.phase),
        barrier_update_count=accumulator.barrier_update_count,
        destination_master_version=accumulator.destination_master_version,
        destination_snapshot_version=accumulator.destination_snapshot_version,
        export_attempts=accumulator.export_attempts,
        export_manifest_sha256=accumulator.export_manifest_sha256,
    )
end

function _master_binding_from_checkpoint(loaded)
    metadata = loaded.metadata
    return HashStemMasterMetadata(
        model_id=String(metadata.model_id),
        master_version=UInt64(metadata.master_version),
        optimizer_step=UInt64(metadata.optimizer_step),
        weights_sha256=String(metadata.weights_sha256),
        normalization_sha256=String(metadata.normalization_sha256),
        optimizer_sha256=String(metadata.optimizer_sha256),
        source_checkpoint_sha256=String(metadata.source_checkpoint_sha256),
        created_utc=String(metadata.created_utc),
    )
end

function _validate_published_boundary_binding_locked(
    accumulator::FrozenStemGradientAccumulator,
    trainer::HashStemMasterTrainer,
    coordinator::VersionCoordinator,
    ring::PipelineRing,
    snapshot::HashStemInferenceSnapshot,
    evidence::NPUAdoptionEvidence,
)
    accumulator.phase == SUPERSTEP_PUBLISHED || throw(ArgumentError(
        "boundary checkpoint requires a published superstep",
    ))
    accumulator.barrier_update_count == 1 || error(
        "boundary lacks exactly one master update",
    )
    _valid_sha256(accumulator.export_manifest_sha256) || error(
        "boundary lacks a bound export manifest",
    )
    accumulator.model_id == trainer.model_id == coordinator.model_id || error(
        "boundary model lineage differs",
    )
    accumulator.destination_master_version == trainer.master_version ==
        coordinator.master_version || error("boundary master lineage differs")
    accumulator.source_master_version < accumulator.destination_master_version || error(
        "boundary source master must precede its destination",
    )
    accumulator.source_master_version + UInt64(1) ==
        accumulator.destination_master_version || error(
        "boundary master must contain exactly one superstep update",
    )
    accumulator.destination_snapshot_version == snapshot.snapshot_version ==
        trainer.last_snapshot_version == coordinator.published_snapshot_version || error(
        "boundary snapshot lineage differs",
    )
    trainer.last_snapshot_master_version == trainer.master_version || error(
        "boundary snapshot was not exported from the saved master",
    )
    accumulator.source_snapshot_version < accumulator.destination_snapshot_version || error(
        "boundary source snapshot must precede its destination",
    )
    accumulator.next_sequence == ring.next_sequence || error(
        "boundary ring cursor differs from the completed superstep frontier",
    )
    accumulator.sparse_bank_version > 0 || error("boundary sparse-bank version is zero")
    accumulator.sparse_index_version > 0 || error("boundary sparse-index version is zero")
    _validate_npu_evidence_binding(snapshot, evidence)
    return nothing
end

"""Atomically bind master, sparse bank/index, coordinator, ring, and superstep."""
function save_heterogeneous_boundary_checkpoint!(
    destination::AbstractString,
    trainer::HashStemMasterTrainer,
    coordinator::VersionCoordinator,
    ring::PipelineRing,
    accumulator::FrozenStemGradientAccumulator,
    snapshot::HashStemInferenceSnapshot,
    evidence::NPUAdoptionEvidence;
    sparse_bank,
    sparse_indexes,
    sparse_optimizer_state,
    source_checkpoint_sha256::AbstractString=repeat("0", 64),
)
    _valid_sha256(source_checkpoint_sha256) || throw(ArgumentError(
        "invalid source checkpoint SHA-256",
    ))
    final = abspath(destination)
    temporary = final * ".tmp"
    (ispath(final) || ispath(temporary)) && throw(ArgumentError(
        "boundary checkpoint and .tmp destinations must both be fresh",
    ))
    lock(coordinator.lock) do
        resume_accepting_npu = coordinator.accepting_npu
        resume_accepting_cpu = coordinator.accepting_cpu
        resume_accepting_npu && resume_accepting_cpu || error(
            "boundary checkpoint requires both adopted admission paths open",
        )
        coordinator.accepting_npu = false
        coordinator.accepting_cpu = false
        _require_old_lineage_drained_locked!(
            coordinator,
            ring,
            coordinator.published_snapshot_version,
        )
        _validate_published_boundary_binding_locked(
            accumulator,
            trainer,
            coordinator,
            ring,
            snapshot,
            evidence,
        )
        coordinator.refresh_pending && error("cannot checkpoint a pending refresh")
        coordinator.probation_open && error("cannot checkpoint probation")
        coordinator.npu_adopted && coordinator.ever_adopted || error(
            "published boundary checkpoint requires an adopted snapshot lineage",
        )
        coordinator.master_version == trainer.master_version || error(
            "checkpoint trainer/coordinator master mismatch",
        )
        coordinator.published_snapshot_version == trainer.last_snapshot_version || error(
            "checkpoint trainer/coordinator snapshot mismatch",
        )
        accumulator.destination_snapshot_version == snapshot.snapshot_version || error(
            "checkpoint snapshot differs from completed superstep",
        )
        mkpath(temporary)
        try
            master_root = joinpath(temporary, "master")
            save_hashstem_master_checkpoint!(
                trainer,
                master_root;
                source_checkpoint_sha256,
            )
            saved_master = load_hashstem_master_checkpoint(master_root)
            validate_snapshot_binding(
                _master_binding_from_checkpoint(saved_master),
                snapshot,
            )
            sparse_path = joinpath(temporary, "sparse_state.jld2")
            JLD2.jldsave(
                sparse_path;
                sparse_bank,
                sparse_indexes,
                sparse_optimizer_state,
                sparse_bank_version=accumulator.sparse_bank_version,
                sparse_index_version=accumulator.sparse_index_version,
            )
            lifecycle_path = joinpath(temporary, "lifecycle.jld2")
            JLD2.jldsave(
                lifecycle_path;
                coordinator=_coordinator_boundary_record(
                    coordinator;
                    resume_accepting_npu,
                    resume_accepting_cpu,
                ),
                ring=(;
                    next_sequence=ring.next_sequence,
                    slot_count=length(ring.slots),
                    slot_generations=UInt64[slot.generation for slot in ring.slots],
                ),
                completed_superstep=_completed_superstep_record(accumulator),
                snapshot,
                evidence,
            )
            master_manifest = joinpath(master_root, "checkpoint_manifest.json")
            manifest = (;
                schema=HETEROGENEOUS_BOUNDARY_CHECKPOINT_SCHEMA,
                implementation_provenance=UNEXECUTED_STATIC_ONLY,
                master_manifest="master/checkpoint_manifest.json",
                master_manifest_sha256=_sha256_file(master_manifest),
                sparse_file="sparse_state.jld2",
                sparse_sha256=_sha256_file(sparse_path),
                lifecycle_file="lifecycle.jld2",
                lifecycle_sha256=_sha256_file(lifecycle_path),
                pipeline_source_sha256=_sha256_file(joinpath(@__DIR__, "pipeline.jl")),
                lifecycle_source_sha256=_sha256_file(@__FILE__),
                master_source_sha256=_sha256_file(joinpath(@__DIR__, "hashstem_master.jl")),
            )
            _write_json(joinpath(temporary, "boundary_manifest.json"), manifest)
            mv(temporary, final)
            coordinator.accepting_cpu = resume_accepting_cpu
            coordinator.accepting_npu = resume_accepting_npu
            return final
        catch
            # A failed transaction remains diagnosable and admission stays closed.
            rethrow()
        end
    end
end

function load_heterogeneous_boundary_checkpoint(path::AbstractString)
    root = abspath(path)
    manifest_path = joinpath(root, "boundary_manifest.json")
    isfile(manifest_path) || error("boundary manifest is missing")
    manifest = JSON3.read(read(manifest_path, String))
    String(manifest.schema) == HETEROGENEOUS_BOUNDARY_CHECKPOINT_SCHEMA || error(
        "boundary checkpoint schema mismatch",
    )
    String(manifest.implementation_provenance) == UNEXECUTED_STATIC_ONLY || error(
        "boundary checkpoint provenance changed",
    )
    String(manifest.master_manifest) == "master/checkpoint_manifest.json" || error(
        "boundary master-manifest path changed",
    )
    String(manifest.sparse_file) == "sparse_state.jld2" || error(
        "boundary sparse-state path changed",
    )
    String(manifest.lifecycle_file) == "lifecycle.jld2" || error(
        "boundary lifecycle path changed",
    )
    for (relative, expected) in (
        (String(manifest.master_manifest), String(manifest.master_manifest_sha256)),
        (String(manifest.sparse_file), String(manifest.sparse_sha256)),
        (String(manifest.lifecycle_file), String(manifest.lifecycle_sha256)),
    )
        member = joinpath(root, relative)
        isfile(member) || error("boundary checkpoint member is missing: $relative")
        _sha256_file(member) == expected || error("boundary member hash mismatch: $relative")
    end
    for (file, expected) in (
        (joinpath(@__DIR__, "pipeline.jl"), String(manifest.pipeline_source_sha256)),
        (@__FILE__, String(manifest.lifecycle_source_sha256)),
        (joinpath(@__DIR__, "hashstem_master.jl"), String(manifest.master_source_sha256)),
    )
        _sha256_file(file) == expected || error("boundary source closure changed: $file")
    end
    loaded_master = load_hashstem_master_checkpoint(joinpath(root, "master"))
    master_binding = _master_binding_from_checkpoint(loaded_master)
    trainer = resume_hashstem_master(joinpath(root, "master"))
    sparse = JLD2.load(joinpath(root, String(manifest.sparse_file)))
    lifecycle = JLD2.load(joinpath(root, String(manifest.lifecycle_file)))
    coordinator_record = lifecycle["coordinator"]
    ring_record = lifecycle["ring"]
    completed = lifecycle["completed_superstep"]
    snapshot = lifecycle["snapshot"]
    evidence = lifecycle["evidence"]
    String(coordinator_record.model_id) == trainer.model_id || error(
        "checkpoint coordinator/trainer model IDs differ",
    )
    Int(coordinator_record.master_version) == Int(trainer.master_version) || error(
        "checkpoint coordinator/trainer master versions differ",
    )
    Int(coordinator_record.published_snapshot_version) ==
        Int(trainer.last_snapshot_version) || error(
        "checkpoint coordinator/trainer snapshot versions differ",
    )
    Bool(coordinator_record.npu_adopted) || error("checkpoint lost NPU adoption state")
    Bool(coordinator_record.ever_adopted) || error("checkpoint lost permanent adoption lineage")
    Bool(coordinator_record.accepting_npu) || error("checkpoint NPU resume admission is closed")
    Bool(coordinator_record.accepting_cpu) || error("checkpoint CPU resume admission is closed")
    !Bool(coordinator_record.probation_open) || error("checkpoint was taken during probation")
    Int(coordinator_record.probation_snapshot_version) == 0 || error(
        "checkpoint has a probation snapshot",
    )
    !Bool(coordinator_record.refresh_pending) || error("checkpoint was taken during refresh")
    Int(sparse["sparse_bank_version"]) == Int(completed.sparse_bank_version) || error(
        "sparse-bank version differs from completed superstep",
    )
    Int(sparse["sparse_index_version"]) == Int(completed.sparse_index_version) || error(
        "sparse-index version differs from completed superstep",
    )
    Int(completed.barrier_update_count) == 1 || error("checkpoint barrier count is not one")
    Int(completed.phase) == Int(UInt8(SUPERSTEP_PUBLISHED)) || error(
        "checkpoint is not at a published superstep boundary",
    )
    Int(completed.destination_master_version) == Int(trainer.master_version) || error(
        "checkpoint master differs from completed superstep",
    )
    Int(completed.destination_snapshot_version) == Int(trainer.last_snapshot_version) || error(
        "checkpoint snapshot differs from completed superstep",
    )
    String(completed.model_id) == trainer.model_id || error(
        "checkpoint completed-superstep model ID differs",
    )
    Int(completed.source_master_version) + 1 ==
        Int(completed.destination_master_version) || error(
        "checkpoint does not bind exactly one master update",
    )
    Int(completed.source_snapshot_version) <
        Int(completed.destination_snapshot_version) || error(
        "checkpoint source snapshot does not precede its destination",
    )
    Int(completed.sparse_bank_version) > 0 || error(
        "checkpoint sparse-bank version is zero",
    )
    Int(completed.sparse_index_version) > 0 || error(
        "checkpoint sparse-index version is zero",
    )
    _valid_sha256(String(completed.export_manifest_sha256)) || error(
        "checkpoint export-manifest binding is invalid",
    )
    coordinator = VersionCoordinator(
        String(coordinator_record.model_id);
        master_version=Int(coordinator_record.master_version),
        snapshot_version=Int(coordinator_record.published_snapshot_version),
    )
    ring = PipelineRing(
        Int(ring_record.slot_count);
        next_sequence=Int(ring_record.next_sequence),
    )
    generations = UInt64.(ring_record.slot_generations)
    length(generations) == length(ring.slots) || error("ring generation count changed")
    for (slot, generation) in zip(ring.slots, generations)
        slot.generation = generation
    end
    validate_snapshot_binding(master_binding, snapshot)
    restore_authorized_snapshot!(
        coordinator,
        ring,
        master_binding,
        snapshot,
        evidence,
    )
    Int(completed.next_sequence) == Int(ring.next_sequence) || error(
        "ring cursor differs from completed superstep frontier",
    )
    return (;
        trainer,
        coordinator,
        ring,
        sparse_bank=sparse["sparse_bank"],
        sparse_indexes=sparse["sparse_indexes"],
        sparse_optimizer_state=sparse["sparse_optimizer_state"],
        sparse_bank_version=UInt64(sparse["sparse_bank_version"]),
        sparse_index_version=UInt64(sparse["sparse_index_version"]),
        completed_superstep=completed,
        snapshot,
        evidence,
        next_superstep=UInt64(completed.superstep) + UInt64(1),
        manifest_sha256=_sha256_file(manifest_path),
        status=UNEXECUTED_STATIC_ONLY,
    )
end
