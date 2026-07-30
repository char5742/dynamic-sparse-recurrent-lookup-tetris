using JLD2
using SHA

abstract type AbstractPaperCellRuntime end

struct DetailedCellRuntime <: AbstractPaperCellRuntime
    tree::Hay.HayTree
    parameters::Hay.HayParameters
    initial_parameters::Hay.HayParameters
    states::Vector{Hay.HayState}
    drives::Vector{Hay.HaySynapticDrive}
    diagnostics::Vector{Hay.HayDiagnostics}
end

struct DistilledCellRuntime <: AbstractPaperCellRuntime
    parameters::Distilled.DistilledParameters
    initial_parameters::Distilled.DistilledParameters
    states::Vector{Distilled.DistilledState}
    drives::Vector{Distilled.DistilledDrive}
    diagnostics::Vector{Distilled.DistilledDiagnostics}
end

struct PaperLineage
    detailed_cell_sha256::String
    digital_twin_sha256::String
    distilled_artifact_sha256::String
    distilled_parameter_sha256::String
    schema::String
end

mutable struct PaperTrainerAux{P}
    internal_parameters::P
    initial_internal_parameters::P
    lineage::PaperLineage
    location_catalog::Vector{UInt8}
    excitatory_capacity::Vector{Int16}
    inhibitory_capacity::Vector{Int16}
    workspace_location::Matrix{UInt8}
    workspace_location_utility::Array{Float32,3}
    regional_projection::Array{Float32,3}
end

const _TRAINER_AUX = IdDict{Any,Any}()

function _source_sha256(path::AbstractString)
    return bytes2hex(SHA.sha256(read(path)))
end

artifact_sha256(path::AbstractString) =
    Distilled.artifact_sha256(path)

function paper_location_catalog()
    tree = Hay.paper_hay_tree()
    return UInt8[compartment for compartment in 2:Hay.compartment_count(tree)]
end

function _location_capacity(tree::Hay.HayTree)
    count = Hay.compartment_count(tree)
    capacity = zeros(Int16, count)
    @inbounds for compartment in 2:count
        parent = Int(tree.parent[compartment])
        length_um = max(
            tree.distance_um[compartment] -
            tree.distance_um[parent],
            1.0f0,
        )
        # Paper constraint: at most one E and one I contact per dendritic µm.
        capacity[compartment] =
            Int16(clamp(floor(Int, length_um), 1, typemax(Int16)))
    end
    return capacity
end

function _payload_field(payload, names::Tuple)
    for name in names
        hasproperty(payload, name) &&
            return String(getproperty(payload, name))
    end
    return ""
end

function _load_lineage(
    path::AbstractString,
    parameters::Distilled.DistilledParameters,
)
    data = JLD2.load(path)
    haskey(data, "payload") ||
        error("distilled artifact has no payload")
    payload = data["payload"]
    detailed = _payload_field(
        payload,
        (:detailed_cell_sha256, :detailed_sha256, :mechanism_sha256),
    )
    twin = _payload_field(
        payload,
        (:digital_twin_sha256, :twin_sha256, :teacher_sha256),
    )
    isempty(twin) && (twin = parameters.teacher_sha256)
    isempty(detailed) && hasproperty(payload, :lineage) &&
        (detailed = _payload_field(
            payload.lineage,
            (:detailed_cell_sha256, :detailed_sha256),
        ))
    isempty(twin) && hasproperty(payload, :lineage) &&
        (twin = _payload_field(
            payload.lineage,
            (:digital_twin_sha256, :twin_sha256),
        ))
    isempty(detailed) &&
        error("artifact lineage lacks detailed-cell SHA-256")
    isempty(twin) &&
        error("artifact lineage lacks frozen digital-twin SHA-256")
    twin == parameters.teacher_sha256 ||
        error("distilled teacher hash differs from lineage twin hash")
    occursin("twin", lowercase(parameters.teacher_schema)) ||
        error("distilled teacher schema is not a digital-twin schema")
    return PaperLineage(
        detailed,
        twin,
        artifact_sha256(path),
        Distilled.parameter_sha256(parameters),
        Distilled.DISTILLED_ARTIFACT_SCHEMA,
    )
end

function _regional_projection(
    blocks::Int,
    seed::Integer=0x524547494f4e4d31,
)
    rng = Xoshiro(UInt64(seed))
    return randn(rng, Float32, 11, OUTPUT_DIM, blocks) ./
        sqrt(Float32(OUTPUT_DIM))
end

function register_paper_trainer_aux!(trainer::PaperTrainer)
    haskey(_TRAINER_AUX, trainer) && return _TRAINER_AUX[trainer]
    tree = Hay.paper_hay_tree()
    catalog = paper_location_catalog()
    capacity = _location_capacity(tree)
    workspace_location = Matrix{UInt8}(
        undef,
        trainer.model.workspace_contacts,
        trainer.model.blocks,
    )
    tuft = isempty(tree.tuft_terminals) ?
        last(catalog) : UInt8(first(tree.tuft_terminals))
    @inbounds for block in 1:trainer.model.blocks
        for contact in 1:trainer.model.workspace_contacts
            workspace_location[contact, block] =
                catalog[mod1(
                    Int(tuft) + contact + 3block,
                    length(catalog),
                )]
        end
    end
    if trainer.cell_mode === :distilled_frozen
        path = something(trainer.cell_artifact)
        parameters = Distilled.load_distilled_artifact(path)
        Distilled.is_frozen(parameters) ||
            error("distilled production cell is not frozen")
        isempty(keys(Distilled.trainable_parameters(parameters))) ||
            error("distilled internals leaked into a trainable tree")
        lineage = _load_lineage(path, parameters)
        aux = PaperTrainerAux(
            parameters,
            deepcopy(parameters),
            lineage,
            catalog,
            copy(capacity),
            copy(capacity),
            workspace_location,
            zeros(
                Float32,
                length(catalog),
                trainer.model.workspace_contacts,
                trainer.model.blocks,
            ),
            _regional_projection(trainer.model.blocks),
        )
    else
        parameters = Hay.HayParameters(tree; ablation=:full)
        detailed_hash = _source_sha256(
            joinpath(@__DIR__, "PaperHayCell.jl"),
        )
        lineage = PaperLineage(
            detailed_hash,
            "detailed-control-no-digital-twin",
            "detailed-control-no-distilled-artifact",
            detailed_hash,
            "paper-hay-detailed-control-v1",
        )
        aux = PaperTrainerAux(
            parameters,
            deepcopy(parameters),
            lineage,
            catalog,
            copy(capacity),
            copy(capacity),
            workspace_location,
            zeros(
                Float32,
                length(catalog),
                trainer.model.workspace_contacts,
                trainer.model.blocks,
            ),
            _regional_projection(trainer.model.blocks),
        )
    end
    _TRAINER_AUX[trainer] = aux
    return aux
end

paper_trainer_aux(trainer::PaperTrainer) =
    register_paper_trainer_aux!(trainer)

function make_cell_runtime(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    blocks = trainer.model.blocks
    if trainer.cell_mode === :distilled_frozen
        parameters = aux.internal_parameters
        return DistilledCellRuntime(
            parameters,
            deepcopy(aux.initial_internal_parameters),
            [Distilled.DistilledState(parameters) for _ in 1:blocks],
            [Distilled.DistilledDrive(parameters) for _ in 1:blocks],
            [Distilled.DistilledDiagnostics() for _ in 1:blocks],
        )
    end
    tree = Hay.paper_hay_tree()
    parameters = aux.internal_parameters
    return DetailedCellRuntime(
        tree,
        parameters,
        deepcopy(aux.initial_internal_parameters),
        [Hay.HayState(tree, parameters) for _ in 1:blocks],
        [Hay.HaySynapticDrive(tree) for _ in 1:blocks],
        [Hay.HayDiagnostics(tree) for _ in 1:blocks],
    )
end

function reset_runtime!(runtime::DetailedCellRuntime)
    @inbounds for block in eachindex(runtime.states)
        Hay.reset_state!(runtime.states[block], runtime.parameters)
        Hay.reset_drive!(runtime.drives[block])
        Hay.reset_diagnostics!(runtime.diagnostics[block])
    end
    return runtime
end

function reset_runtime!(runtime::DistilledCellRuntime)
    @inbounds for block in eachindex(runtime.states)
        Distilled.reset_state!(runtime.states[block], runtime.parameters)
        Distilled.reset_drive!(runtime.drives[block])
        Distilled.reset_diagnostics!(runtime.diagnostics[block])
    end
    return runtime
end

@inline function reset_cell_drive!(
    runtime::DetailedCellRuntime,
    block::Int,
)
    Hay.reset_drive!(runtime.drives[block])
end

@inline function reset_cell_drive!(
    runtime::DistilledCellRuntime,
    block::Int,
)
    Distilled.reset_drive!(runtime.drives[block])
end

@inline function add_cell_event!(
    runtime::DetailedCellRuntime,
    block::Int,
    compartment::Int,
    kind::UInt8,
    amplitude::Float32,
)
    if kind == Model.EXCITATORY
        Hay.add_synaptic_event!(
            runtime.drives[block],
            compartment;
            ampa=amplitude,
            nmda=amplitude,
        )
    else
        Hay.add_synaptic_event!(
            runtime.drives[block],
            compartment;
            gaba=amplitude,
        )
    end
    return nothing
end

@inline function add_cell_event!(
    runtime::DistilledCellRuntime,
    block::Int,
    compartment::Int,
    kind::UInt8,
    amplitude::Float32,
)
    if kind == Model.EXCITATORY
        Distilled.add_synaptic_event!(
            runtime.drives[block],
            compartment,
            :ampa,
            amplitude,
        )
        Distilled.add_synaptic_event!(
            runtime.drives[block],
            compartment,
            :nmda,
            amplitude,
        )
    else
        Distilled.add_synaptic_event!(
            runtime.drives[block],
            compartment,
            :gaba,
            amplitude,
        )
    end
    return nothing
end

@inline function step_cell!(
    runtime::DetailedCellRuntime,
    block::Int,
)
    return Hay.hay_cell_step!(
        runtime.states[block],
        runtime.drives[block],
        runtime.diagnostics[block],
        runtime.tree,
        runtime.parameters,
    )
end

@inline function step_cell!(
    runtime::DistilledCellRuntime,
    block::Int,
)
    return Distilled.distilled_cell_step!(
        runtime.states[block],
        runtime.drives[block],
        runtime.diagnostics[block],
        runtime.parameters,
    )
end

@inline function cell_nmda_sum(
    runtime::DetailedCellRuntime,
    block::Int,
)
    total = 0.0f0
    @inbounds for value in runtime.diagnostics[block].nmda_current
        total += abs(value)
    end
    return total
end

@inline function cell_nmda_sum(
    runtime::DistilledCellRuntime,
    block::Int,
)
    total = 0.0f0
    @inbounds for value in runtime.diagnostics[block].nmda_current
        total += abs(value)
    end
    return total
end

@inline function cell_calcium_event(
    runtime::DetailedCellRuntime,
    block::Int,
)
    @inbounds for value in runtime.states[block].local_ca_event
        value != 0.0f0 && return 1.0f0
    end
    return 0.0f0
end

@inline cell_calcium_event(
    runtime::DistilledCellRuntime,
    block::Int,
) = runtime.diagnostics[block].calcium_event

@inline function cell_surrogate(
    runtime::DetailedCellRuntime,
    block::Int,
)
    state = runtime.states[block]
    threshold = runtime.parameters.soma_spike_threshold_mv
    voltage = state.voltage_mv[Int(runtime.tree.soma)]
    return 0.10f0 * exp(-0.10f0 * abs(voltage - threshold))
end

@inline function cell_surrogate(
    runtime::DistilledCellRuntime,
    block::Int,
)
    probability = runtime.diagnostics[block].spike_probability
    return probability * (1.0f0 - probability)
end

function cell_local_state!(
    destination::AbstractVector{Float32},
    runtime::DistilledCellRuntime,
    block::Int,
)
    copyto!(destination, runtime.states[block].value)
    return destination
end

@inline function _region_mean(
    values,
    tree::Hay.HayTree,
    region::UInt8,
)
    total = 0.0f0
    count = 0
    @inbounds for index in eachindex(values)
        tree.region[index] == region || continue
        total += values[index]
        count += 1
    end
    return total / Float32(max(count, 1))
end

function cell_local_state!(
    destination::AbstractVector{Float32},
    runtime::DetailedCellRuntime,
    block::Int,
)
    state = runtime.states[block]
    diagnostics = runtime.diagnostics[block]
    tree = runtime.tree
    destination[1] = _region_mean(state.voltage_mv, tree, Hay.BASAL)
    destination[2] =
        _region_mean(state.voltage_mv, tree, Hay.APICAL_TRUNK)
    hot = 0.0f0
    @inbounds for compartment in tree.apical_hot_zone
        hot += state.voltage_mv[Int(compartment)]
    end
    destination[3] =
        hot / Float32(max(length(tree.apical_hot_zone), 1))
    destination[4] =
        _region_mean(state.voltage_mv, tree, Hay.APICAL_TUFT)
    destination[5] =
        abs(diagnostics.nmda_current[Int(tree.soma)])
    destination[6] =
        abs(_region_mean(diagnostics.nmda_current, tree, Hay.BASAL))
    destination[7] =
        abs(_region_mean(
            diagnostics.nmda_current,
            tree,
            Hay.APICAL_TRUNK,
        ))
    destination[8] =
        abs(_region_mean(
            diagnostics.nmda_current,
            tree,
            Hay.APICAL_TUFT,
        ))
    destination[9] = 0.5f0 * (destination[2] + destination[4])
    destination[10] = state.voltage_mv[Int(tree.soma)]
    destination[11] = cell_calcium_event(runtime, block)
    return destination
end

function _max_parameter_delta(reference, current)
    typeof(reference) === typeof(current) || return Inf32
    maximum_delta = 0.0f0
    for name in fieldnames(typeof(reference))
        first = getfield(reference, name)
        second = getfield(current, name)
        if first isa AbstractArray
            size(first) == size(second) || return Inf32
            @inbounds for index in eachindex(first, second)
                maximum_delta = max(
                    maximum_delta,
                    Float32(abs(first[index] - second[index])),
                )
            end
        elseif first isa Number
            maximum_delta = max(
                maximum_delta,
                Float32(abs(first - second)),
            )
        elseif first != second
            return Inf32
        end
    end
    return maximum_delta
end

function paper_internal_max_delta(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    return _max_parameter_delta(
        aux.initial_internal_parameters,
        aux.internal_parameters,
    )
end

function paper_internal_sha256(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    return aux.lineage.distilled_parameter_sha256
end

function paper_aux_snapshot(trainer::PaperTrainer)
    aux = register_paper_trainer_aux!(trainer)
    return (;
        workspace_location=copy(aux.workspace_location),
        workspace_location_utility=
            copy(aux.workspace_location_utility),
        excitatory_capacity=copy(aux.excitatory_capacity),
        inhibitory_capacity=copy(aux.inhibitory_capacity),
        regional_projection=copy(aux.regional_projection),
        lineage=aux.lineage,
    )
end

function restore_paper_aux_snapshot!(trainer::PaperTrainer, snapshot)
    aux = register_paper_trainer_aux!(trainer)
    copyto!(aux.workspace_location, snapshot.workspace_location)
    copyto!(
        aux.workspace_location_utility,
        snapshot.workspace_location_utility,
    )
    copyto!(aux.excitatory_capacity, snapshot.excitatory_capacity)
    copyto!(aux.inhibitory_capacity, snapshot.inhibitory_capacity)
    copyto!(aux.regional_projection, snapshot.regional_projection)
    snapshot.lineage == aux.lineage ||
        error("checkpoint cell lineage differs")
    return trainer
end
