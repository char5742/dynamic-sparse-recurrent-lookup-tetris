# Final HD-SWSNN release integration.
#
# This file intentionally overlays only the distilled production arm.  The
# detailed Hay arm remains a bounded teacher/control path.  Production uses the
# release-v2 contract:
#   * UInt16 official segment IDs 1:642;
#   * coordinate-wise semantic 11-state representation;
#   * trusted, allocation-free frozen cell step;
#   * integrity hashes only at preflight/checkpoint/run-end boundaries.

include(joinpath(@__DIR__, "DistilledElevenStateCellReleaseRuntimeV2.jl"))
const ReleaseCell = DistilledElevenStateCellReleaseRuntimeV2

struct ReleaseCellRuntime <: AbstractPaperCellRuntime
    trusted::ReleaseCell.TrustedReleaseRuntime
    states::Vector{ReleaseCell.Final.DistilledState}
    drives::Vector{ReleaseCell.Final.DistilledDrive}
    diagnostics::Vector{ReleaseCell.Final.DistilledDiagnostics}
end

mutable struct PaperReleaseAux
    trusted::ReleaseCell.TrustedReleaseRuntime
    internal_parameters::ReleaseCell.Final.DistilledParameters
    initial_internal_parameters::ReleaseCell.Final.DistilledParameters
    lineage::PaperLineage
    location_catalog::Vector{UInt16}
    official_segment_region::Vector{String}
    excitatory_capacity::Vector{Int16}
    inhibitory_capacity::Vector{Int16}
    input_location::Matrix{UInt16}
    recurrent_location::Matrix{UInt16}
    workspace_location::Matrix{UInt16}
    input_location_utility::Array{Float32,3}
    recurrent_location_utility::Array{Float32,3}
    workspace_location_utility::Array{Float32,3}
    regional_projection::Array{Float32,3}
    location_mapping_sha256::String
end

mutable struct SparseReleaseLocationUtility
    input_current::Matrix{Float32}
    input_best_value::Matrix{Float32}
    input_best_location::Matrix{UInt16}
    recurrent_current::Matrix{Float32}
    recurrent_best_value::Matrix{Float32}
    recurrent_best_location::Matrix{UInt16}
    workspace_current::Matrix{Float32}
    workspace_best_value::Matrix{Float32}
    workspace_best_location::Matrix{UInt16}
end

const _RELEASE_WORKER_UTILITY = IdDict{Any,SparseReleaseLocationUtility}()

@inline function _payload_release_value(
    payload,
    name::Symbol,
    default=nothing,
)
    if payload isa AbstractDict
        return get(payload, name, get(payload, String(name), default))
    end
    return hasproperty(payload, name) ?
        getproperty(payload, name) : default
end

function _release_initial_locations(model)
    inputs = Matrix{UInt16}(
        undef,
        model.sensory_contacts,
        model.blocks,
    )
    recurrent = Matrix{UInt16}(
        undef,
        model.recurrent_contacts,
        model.blocks,
    )
    workspace = Matrix{UInt16}(
        undef,
        model.workspace_contacts,
        model.blocks,
    )
    count = ReleaseCell.OFFICIAL_LOCATION_COUNT
    @inbounds for block in 1:model.blocks
        base = 53 * (block - 1)
        for contact in 1:model.sensory_contacts
            inputs[contact, block] =
                UInt16(mod1(base + contact, count))
        end
        for contact in 1:model.recurrent_contacts
            recurrent[contact, block] = UInt16(mod1(
                base + model.sensory_contacts + contact,
                count,
            ))
        end
        for contact in 1:model.workspace_contacts
            workspace[contact, block] = UInt16(mod1(
                base +
                model.sensory_contacts +
                model.recurrent_contacts +
                contact,
                count,
            ))
        end
    end
    return inputs, recurrent, workspace
end

"""
Load and register the only production cell contract.

This is fail-closed: a provisional/final-v1 artifact, missing semantic gate,
rotationally unidentified state basis, non-642 mapping, failed multi-target
gate, or hash mismatch is rejected by `load_release_runtime`.
"""
function enable_release_runtime!(
    trainer::PaperTrainer,
    artifact_path::AbstractString=something(trainer.cell_artifact),
)
    trainer.cell_mode === :distilled_frozen ||
        error("release runtime requires cell_mode=:distilled_frozen")
    path = abspath(artifact_path)
    trainer.cell_artifact == path ||
        error("trainer artifact path differs from release artifact path")
    trusted = ReleaseCell.load_release_runtime(path)
    ReleaseCell.preflight_integrity!(trusted)
    data = JLD2.load(path)
    payload = data["payload"]
    regions = String.(collect(_payload_release_value(
        payload,
        :official_segment_region,
        (),
    )))
    length(regions) == ReleaseCell.OFFICIAL_LOCATION_COUNT ||
        error("release artifact has no complete 642-segment region map")
    String(_payload_release_value(
        payload,
        :semantic_state_scale,
        "",
    )) == "normalized_unit_interval" ||
        error("release semantic states are not normalized_unit_interval")
    String(_payload_release_value(
        payload,
        :location_index_type,
        "",
    )) == "UInt16" ||
        error("release location index type is not UInt16")
    input_location, recurrent_location, workspace_location =
        _release_initial_locations(trainer.model)
    catalog = UInt16.(1:ReleaseCell.OFFICIAL_LOCATION_COUNT)
    capacity = fill(
        Int16(1),
        ReleaseCell.OFFICIAL_LOCATION_COUNT,
    )
    parameters = trusted.parameters
    lineage = PaperLineage(
        parameters.detailed_kernel_hash,
        parameters.frozen_twin_artifact_hash,
        trusted.artifact_sha256,
        trusted.expected_parameter_sha256,
        ReleaseCell.Final.DISTILLED_ARTIFACT_SCHEMA,
    )
    aux = PaperReleaseAux(
        trusted,
        parameters,
        deepcopy(parameters),
        lineage,
        catalog,
        regions,
        copy(capacity),
        copy(capacity),
        input_location,
        recurrent_location,
        workspace_location,
        zeros(
            Float32,
            length(catalog),
            trainer.model.sensory_contacts,
            trainer.model.blocks,
        ),
        zeros(
            Float32,
            length(catalog),
            trainer.model.recurrent_contacts,
            trainer.model.blocks,
        ),
        zeros(
            Float32,
            length(catalog),
            trainer.model.workspace_contacts,
            trainer.model.blocks,
        ),
        _regional_projection(trainer.model.blocks),
        trusted.location_mapping_sha256,
    )
    _TRAINER_AUX[trainer] = aux
    return aux
end

function _release_runtime(aux::PaperReleaseAux, blocks::Int)
    trusted = aux.trusted
    return ReleaseCellRuntime(
        trusted,
        [ReleaseCell.release_new_state(trusted) for _ in 1:blocks],
        [ReleaseCell.release_new_drive(trusted) for _ in 1:blocks],
        [
            ReleaseCell.release_new_diagnostics(trusted)
            for _ in 1:blocks
        ],
    )
end

function _release_sparse_utility(model)
    return SparseReleaseLocationUtility(
        zeros(Float32, model.sensory_contacts, model.blocks),
        zeros(Float32, model.sensory_contacts, model.blocks),
        zeros(UInt16, model.sensory_contacts, model.blocks),
        zeros(Float32, model.recurrent_contacts, model.blocks),
        zeros(Float32, model.recurrent_contacts, model.blocks),
        zeros(UInt16, model.recurrent_contacts, model.blocks),
        zeros(Float32, model.workspace_contacts, model.blocks),
        zeros(Float32, model.workspace_contacts, model.blocks),
        zeros(UInt16, model.workspace_contacts, model.blocks),
    )
end

function _paper_release_worker(
    trainer::PaperTrainer,
    aux::PaperReleaseAux,
)
    model = trainer.model
    worker = PaperWorker(
        _release_runtime(aux, model.blocks),
        Optim.zero_parameter_tree(trainer.parameters),
        Point.PackScratch(),
        zeros(Float32, model.blocks),
        fill(false, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Int16, model.workspace_k),
        zeros(Float32, model.node_dim, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, model.blocks),
        zeros(Float32, 2model.node_dim),
        zeros(Float32, 2model.node_dim),
        zeros(Float32, model.hidden),
        zeros(Float32, 11, model.blocks),
        zeros(Float32, 11),
        zeros(Float32, 1, 1, 1),
        zeros(Float32, 1, 1, 1),
        zeros(Float32, 1, 1, 1),
        UInt64(0),
    )
    _RELEASE_WORKER_UTILITY[worker] =
        _release_sparse_utility(model)
    return worker
end

function PaperWorker(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    return aux isa PaperReleaseAux ?
        _paper_release_worker(trainer, aux) :
        paper_worker_final(trainer)
end

function reset_worker_accumulator!(
    worker::PaperWorker{ReleaseCellRuntime},
)
    Optim.zero_parameter_tree!(worker.gradient)
    utility = _RELEASE_WORKER_UTILITY[worker]
    fill!(utility.input_current, 0.0f0)
    fill!(utility.input_best_value, 0.0f0)
    fill!(utility.input_best_location, UInt16(0))
    fill!(utility.recurrent_current, 0.0f0)
    fill!(utility.recurrent_best_value, 0.0f0)
    fill!(utility.recurrent_best_location, UInt16(0))
    fill!(utility.workspace_current, 0.0f0)
    fill!(utility.workspace_best_value, 0.0f0)
    fill!(utility.workspace_best_location, UInt16(0))
    worker.jobs = UInt64(0)
    return worker
end

function reset_runtime!(runtime::ReleaseCellRuntime)
    parameters = runtime.trusted.parameters
    @inbounds for block in eachindex(runtime.states)
        ReleaseCell.Final.reset_state!(
            runtime.states[block],
            parameters,
        )
        ReleaseCell.Final.reset_drive!(runtime.drives[block])
        ReleaseCell.Final.reset_diagnostics!(
            runtime.diagnostics[block],
        )
    end
    return runtime
end

@inline reset_cell_drive!(
    runtime::ReleaseCellRuntime,
    block::Int,
) = ReleaseCell.Final.reset_drive!(runtime.drives[block])

@inline function add_cell_event!(
    runtime::ReleaseCellRuntime,
    block::Int,
    location::Int,
    kind::UInt8,
    amplitude::Float32,
)
    if kind == Model.EXCITATORY
        ReleaseCell.release_add_synaptic_event!(
            runtime.drives[block],
            UInt16(location),
            :ampa,
            amplitude,
        )
        ReleaseCell.release_add_synaptic_event!(
            runtime.drives[block],
            UInt16(location),
            :nmda,
            amplitude,
        )
    else
        ReleaseCell.release_add_synaptic_event!(
            runtime.drives[block],
            UInt16(location),
            :gaba,
            amplitude,
        )
    end
    return nothing
end

@inline step_cell!(
    runtime::ReleaseCellRuntime,
    block::Int,
) = ReleaseCell.trusted_cell_step!(
    runtime.trusted,
    runtime.states[block],
    runtime.drives[block],
    runtime.diagnostics[block],
)

@inline function cell_nmda_sum(
    runtime::ReleaseCellRuntime,
    block::Int,
)
    total = 0.0f0
    @inbounds for value in runtime.diagnostics[block].nmda_current
        total += abs(value)
    end
    return total
end

@inline cell_calcium_event(
    runtime::ReleaseCellRuntime,
    block::Int,
) = runtime.diagnostics[block].calcium_event

@inline function cell_surrogate(
    runtime::ReleaseCellRuntime,
    block::Int,
)
    probability =
        runtime.diagnostics[block].spike_probability
    return probability * (1.0f0 - probability)
end

function cell_local_state!(
    destination::AbstractVector{Float32},
    runtime::ReleaseCellRuntime,
    block::Int,
)
    copyto!(destination, runtime.states[block].value)
    return destination
end

@inline function _compartment_voltage(
    runtime::ReleaseCellRuntime,
    block::Int,
    location::Int,
)
    projection =
        runtime.trusted.parameters.compartment_projection
    diagnostics = runtime.diagnostics[block]
    value = 0.0f0
    @inbounds for branch in 1:4
        value = muladd(
            projection[branch, location],
            diagnostics.dendritic_voltage_mv[branch],
            value,
        )
    end
    return value
end

@inline function _compartment_nmda(
    runtime::ReleaseCellRuntime,
    block::Int,
    location::Int,
)
    projection =
        runtime.trusted.parameters.compartment_projection
    diagnostics = runtime.diagnostics[block]
    value = 0.0f0
    @inbounds for branch in 1:4
        value = muladd(
            projection[branch, location],
            abs(diagnostics.nmda_current[branch]),
            value,
        )
    end
    return value
end

@inline function _add_input_events!(
    worker::PaperWorker{ReleaseCellRuntime},
    trainer::PaperTrainer,
    block::Int,
    flat::Int,
    millisecond::Int,
)
    model = trainer.model
    arena = trainer.tape.base
    aux = register_paper_trainer_aux!(trainer)::PaperReleaseAux
    @inbounds for contact in 1:model.sensory_contacts
        rail = Int(model.input_rail[contact, block])
        _sensory_spike(
            arena.rails[rail, flat],
            rail,
            millisecond,
            flat,
        ) || continue
        add_cell_event!(
            worker.runtime,
            block,
            Int(aux.input_location[contact, block]),
            _contact_kind(model, rail),
            trainer.parameters.input_conductance[contact, block],
        )
    end
    return nothing
end

@inline function _add_recurrent_events!(
    worker::PaperWorker{ReleaseCellRuntime},
    trainer::PaperTrainer,
    block::Int,
)
    model = trainer.model
    aux = register_paper_trainer_aux!(trainer)::PaperReleaseAux
    @inbounds for contact in 1:model.recurrent_contacts
        source = Int(model.recurrent_source[contact, block])
        worker.previous_spike[source] == 0.0f0 && continue
        add_cell_event!(
            worker.runtime,
            block,
            Int(aux.recurrent_location[contact, block]),
            _source_block_kind(model, source),
            trainer.parameters.recurrent_conductance[contact, block],
        )
    end
    return nothing
end

@inline function _add_workspace_events!(
    worker::PaperWorker{ReleaseCellRuntime},
    trainer::PaperTrainer,
    aux::PaperReleaseAux,
    block::Int,
)
    model = trainer.model
    @inbounds for rank in 1:model.workspace_contacts
        source = Int(worker.route_order[rank])
        worker.previous_spike[source] == 0.0f0 && continue
        add_cell_event!(
            worker.runtime,
            block,
            Int(aux.workspace_location[rank, block]),
            _source_block_kind(model, source),
            trainer.parameters.workspace_conductance[rank, block],
        )
    end
    return nothing
end

function paper_preflight_integrity!(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    aux isa PaperReleaseAux ||
        error("production preflight has no release runtime")
    return ReleaseCell.preflight_integrity!(aux.trusted)
end

function paper_checkpoint_integrity!(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    aux isa PaperReleaseAux ||
        error("production checkpoint has no release runtime")
    return ReleaseCell.checkpoint_integrity!(aux.trusted)
end

function paper_end_run_integrity!(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    aux isa PaperReleaseAux ||
        error("production run has no release runtime")
    return ReleaseCell.end_run_integrity!(aux.trusted)
end

function paper_internal_max_delta(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    return aux isa PaperReleaseAux ? 0.0f0 :
        _max_parameter_delta(
            aux.initial_internal_parameters,
            aux.internal_parameters,
        )
end

function paper_internal_sha256(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    return aux isa PaperReleaseAux ?
        aux.trusted.artifact_sha256 :
        aux.lineage.distilled_artifact_sha256
end

paper_internal_artifact_sha256(trainer::PaperTrainer) =
    paper_internal_sha256(trainer)

function paper_internal_parameter_sha256(
    trainer::PaperTrainer,
)
    aux = register_paper_trainer_aux!(trainer)
    return aux isa PaperReleaseAux ?
        aux.trusted.expected_parameter_sha256 :
        Distilled.parameter_sha256(aux.internal_parameters)
end

function paper_internal_lineage(trainer::PaperTrainer)
    return register_paper_trainer_aux!(trainer).lineage
end

