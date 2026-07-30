module DistilledElevenStateCellFinal

using JLD2
using Serialization
using SHA

export DISTILLED_ARTIFACT_SCHEMA,
    DISTILLED_INPUT_DIM,
    DISTILLED_STATE_DIM,
    DISTILLED_TARGET_DIM,
    DistilledDiagnostics,
    DistilledDrive,
    DistilledParameters,
    DistilledSoADiagnostics,
    DistilledState,
    add_synaptic_event!,
    artifact_sha256,
    assert_parameter_sha256,
    distilled_cell_step!,
    distilled_cell_step_soa!,
    is_frozen,
    load_distilled_artifact,
    parameter_sha256,
    reset_diagnostics!,
    reset_drive!,
    reset_state!,
    reset_soa_diagnostics!,
    trainable_parameters

const DISTILLED_ARTIFACT_SCHEMA =
    "paper-hay-digital-twin-eleven-state-distillation-v2"
const DISTILLED_STATE_DIM = 11
const DISTILLED_INPUT_DIM = 16
const DISTILLED_TARGET_DIM = 11

"""
Immutable container for the frozen CPU surrogate of the detailed Hay cell.

The persistent coordinates are four dendritic-voltage latents, four
NMDA/current latents, one apical/context latent, one soma-voltage latent, and
one adaptation/Ca-summary latent. Arrays are optimized only by the standalone
distiller. The Tetris model must hold this object outside its optimizer tree.
"""
struct DistilledParameters
    dt_ms::Float32
    transition_decay::Vector{Float32}
    recurrent_weight::Matrix{Float32}
    input_weight::Matrix{Float32}
    transition_bias::Vector{Float32}
    readout_weight::Matrix{Float32}
    readout_bias::Vector{Float32}
    target_mean::Vector{Float32}
    target_scale::Vector{Float32}
    initial_state::Vector{Float32}
    compartment_projection::Matrix{Float32}
    region_projection::Matrix{Float32}
    spike_threshold::Float32
    teacher_schema::String
    detailed_kernel_hash::String
    morphology_hash::String
    frozen_twin_parameter_hash::String
    frozen_twin_artifact_hash::String
    distillation_dataset_hash::String
    distillation_config_hash::String
end

function DistilledParameters(;
    dt_ms::Real,
    transition_decay,
    recurrent_weight,
    input_weight,
    transition_bias,
    readout_weight,
    readout_bias,
    target_mean,
    target_scale,
    initial_state,
    compartment_projection,
    region_projection,
    spike_threshold::Real=0.5f0,
    teacher_schema::AbstractString,
    detailed_kernel_hash::AbstractString,
    morphology_hash::AbstractString,
    frozen_twin_parameter_hash::AbstractString,
    frozen_twin_artifact_hash::AbstractString,
    distillation_dataset_hash::AbstractString,
    distillation_config_hash::AbstractString,
)
    parameters = DistilledParameters(
        Float32(dt_ms),
        Float32.(transition_decay),
        Float32.(recurrent_weight),
        Float32.(input_weight),
        Float32.(transition_bias),
        Float32.(readout_weight),
        Float32.(readout_bias),
        Float32.(target_mean),
        Float32.(target_scale),
        Float32.(initial_state),
        Float32.(compartment_projection),
        Float32.(region_projection),
        Float32(spike_threshold),
        String(teacher_schema),
        String(detailed_kernel_hash),
        String(morphology_hash),
        String(frozen_twin_parameter_hash),
        String(frozen_twin_artifact_hash),
        String(distillation_dataset_hash),
        String(distillation_config_hash),
    )
    return _validate_parameters(parameters)
end

function _validate_parameters(parameters::DistilledParameters)
    parameters.dt_ms > 0.0f0 ||
        throw(ArgumentError("dt_ms must be positive"))
    size(parameters.recurrent_weight) == (11, 11) ||
        throw(DimensionMismatch("recurrent_weight must be 11 x 11"))
    size(parameters.input_weight) == (11, 16) ||
        throw(DimensionMismatch("input_weight must be 11 x 16"))
    size(parameters.readout_weight) == (11, 11) ||
        throw(DimensionMismatch("readout_weight must be 11 x 11"))
    length(parameters.transition_decay) == 11 ||
        throw(DimensionMismatch("transition_decay must have length 11"))
    length(parameters.transition_bias) == 11 ||
        throw(DimensionMismatch("transition_bias must have length 11"))
    length(parameters.readout_bias) == 11 ||
        throw(DimensionMismatch("readout_bias must have length 11"))
    length(parameters.target_mean) == 11 ||
        throw(DimensionMismatch("target_mean must have length 11"))
    length(parameters.target_scale) == 11 ||
        throw(DimensionMismatch("target_scale must have length 11"))
    length(parameters.initial_state) == 11 ||
        throw(DimensionMismatch("initial_state must have length 11"))
    size(parameters.compartment_projection, 1) == 4 ||
        throw(DimensionMismatch(
            "compartment_projection must have four rows",
        ))
    size(parameters.region_projection) == (4, 4) ||
        throw(DimensionMismatch("region_projection must be 4 x 4"))
    all(isfinite, parameters.transition_decay) ||
        throw(ArgumentError("transition decay is non-finite"))
    all(value -> 0.0f0 <= value <= 1.0f0,
        parameters.transition_decay) ||
        throw(ArgumentError("transition decay must lie in [0,1]"))
    all(value -> isfinite(value) && value > 0.0f0,
        parameters.target_scale) ||
        throw(ArgumentError("target scale must be finite and positive"))
    all(value -> isfinite(value) && value >= 0.0f0,
        parameters.compartment_projection) ||
        throw(ArgumentError(
            "compartment projection must be finite and nonnegative",
        ))
    all(value -> isfinite(value) && value >= 0.0f0,
        parameters.region_projection) ||
        throw(ArgumentError(
            "region projection must be finite and nonnegative",
        ))
    for name in (
        :teacher_schema,
        :detailed_kernel_hash,
        :morphology_hash,
        :frozen_twin_parameter_hash,
        :frozen_twin_artifact_hash,
        :distillation_dataset_hash,
        :distillation_config_hash,
    )
        isempty(getproperty(parameters, name)) &&
            throw(ArgumentError("$name must be recorded"))
    end
    return parameters
end

@inline is_frozen(::DistilledParameters) = true
@inline trainable_parameters(::DistilledParameters) = NamedTuple()

mutable struct DistilledState
    value::Vector{Float32}
    next_value::Vector{Float32}
end

function DistilledState(parameters::DistilledParameters)
    return reset_state!(
        DistilledState(zeros(Float32, 11), zeros(Float32, 11)),
        parameters,
    )
end

mutable struct DistilledDrive
    event::Vector{Float32}
    compartment_projection::Matrix{Float32}
    region_projection::Matrix{Float32}
end

DistilledDrive(parameters::DistilledParameters) = DistilledDrive(
    zeros(Float32, 16),
    parameters.compartment_projection,
    parameters.region_projection,
)

mutable struct DistilledDiagnostics
    soma_voltage_mv::Float32
    soma_spike::Float32
    spike_probability::Float32
    nmda_current::Vector{Float32}
    calcium_event_probability::Float32
    calcium_event::Float32
    dendritic_voltage_mv::Vector{Float32}
end

DistilledDiagnostics() = reset_diagnostics!(DistilledDiagnostics(
    0.0f0,
    0.0f0,
    0.0f0,
    zeros(Float32, 4),
    0.0f0,
    0.0f0,
    zeros(Float32, 4),
))

struct DistilledSoADiagnostics
    soma_voltage_mv::Vector{Float32}
    soma_spike::Vector{Float32}
    spike_probability::Vector{Float32}
    nmda_current::Matrix{Float32}
    calcium_event_probability::Vector{Float32}
    calcium_event::Vector{Float32}
    dendritic_voltage_mv::Matrix{Float32}
end

function DistilledSoADiagnostics(batch::Integer)
    batch >= 1 || throw(ArgumentError("batch must be positive"))
    return reset_soa_diagnostics!(DistilledSoADiagnostics(
        zeros(Float32, batch),
        zeros(Float32, batch),
        zeros(Float32, batch),
        zeros(Float32, 4, batch),
        zeros(Float32, batch),
        zeros(Float32, batch),
        zeros(Float32, 4, batch),
    ))
end

function reset_state!(
    state::DistilledState,
    parameters::DistilledParameters,
)
    length(state.value) == 11 ||
        throw(DimensionMismatch("state must have eleven coordinates"))
    copyto!(state.value, parameters.initial_state)
    copyto!(state.next_value, parameters.initial_state)
    return state
end

reset_drive!(drive::DistilledDrive) = (fill!(drive.event, 0.0f0); drive)

function reset_diagnostics!(diagnostics::DistilledDiagnostics)
    diagnostics.soma_voltage_mv = 0.0f0
    diagnostics.soma_spike = 0.0f0
    diagnostics.spike_probability = 0.0f0
    fill!(diagnostics.nmda_current, 0.0f0)
    diagnostics.calcium_event_probability = 0.0f0
    diagnostics.calcium_event = 0.0f0
    fill!(diagnostics.dendritic_voltage_mv, 0.0f0)
    return diagnostics
end

function reset_soa_diagnostics!(diagnostics::DistilledSoADiagnostics)
    fill!(diagnostics.soma_voltage_mv, 0.0f0)
    fill!(diagnostics.soma_spike, 0.0f0)
    fill!(diagnostics.spike_probability, 0.0f0)
    fill!(diagnostics.nmda_current, 0.0f0)
    fill!(diagnostics.calcium_event_probability, 0.0f0)
    fill!(diagnostics.calcium_event, 0.0f0)
    fill!(diagnostics.dendritic_voltage_mv, 0.0f0)
    return diagnostics
end

@inline function _receptor_index(receptor)
    receptor === :ampa && return 1
    receptor === :nmda && return 2
    (receptor === :gaba || receptor === :gabaa) && return 3
    receptor === :current && return 4
    receptor isa Integer ||
        throw(ArgumentError("unsupported receptor"))
    1 <= receptor <= 4 ||
        throw(ArgumentError("receptor index must be in 1:4"))
    return Int(receptor)
end

@inline function _region_index(region::Symbol)
    region === :soma && return 1
    region === :basal && return 2
    region === :apical && return 3
    region === :tuft && return 4
    throw(ArgumentError("unsupported anatomical region"))
end

function add_synaptic_event!(
    drive::DistilledDrive,
    compartment_or_region,
    receptor,
    amplitude::Real,
)
    value = Float32(amplitude)
    value >= 0.0f0 ||
        throw(ArgumentError(
            "conductance/current event amplitude must be nonnegative",
        ))
    receptor_index = _receptor_index(receptor)
    offset = 4(receptor_index - 1)
    if compartment_or_region isa Integer
        compartment = Int(compartment_or_region)
        1 <= compartment <= size(drive.compartment_projection, 2) ||
            throw(BoundsError(
                drive.compartment_projection,
                (:, compartment),
            ))
        @inbounds for branch in 1:4
            drive.event[offset + branch] = muladd(
                value,
                drive.compartment_projection[branch, compartment],
                drive.event[offset + branch],
            )
        end
    elseif compartment_or_region isa Symbol
        region = _region_index(compartment_or_region)
        @inbounds for branch in 1:4
            drive.event[offset + branch] = muladd(
                value,
                drive.region_projection[branch, region],
                drive.event[offset + branch],
            )
        end
    else
        throw(ArgumentError(
            "target must be a compartment integer or region Symbol",
        ))
    end
    return drive
end

@inline _sigmoid(value::Float32) =
    ifelse(
        value >= 0.0f0,
        inv(1.0f0 + exp(-value)),
        exp(value) / (1.0f0 + exp(value)),
    )

@inline _activation(index::Int, value::Float32) =
    index == 11 ? _sigmoid(value) : tanh(value)

@inline function _transition_coordinate(
    state,
    drive,
    parameters,
    coordinate::Int,
)
    value = parameters.transition_bias[coordinate]
    @inbounds for source in 1:11
        value = muladd(
            parameters.recurrent_weight[coordinate, source],
            state[source],
            value,
        )
    end
    @inbounds for source in 1:16
        value = muladd(
            parameters.input_weight[coordinate, source],
            drive[source],
            value,
        )
    end
    proposal = _activation(coordinate, value)
    decay = parameters.transition_decay[coordinate]
    return muladd(decay, state[coordinate] - proposal, proposal)
end

@inline function _readout(
    state,
    parameters::DistilledParameters,
    target::Int,
)
    value = parameters.readout_bias[target]
    @inbounds for coordinate in 1:11
        value = muladd(
            parameters.readout_weight[target, coordinate],
            state[coordinate],
            value,
        )
    end
    return target == 2 || target == 7 ? value :
        muladd(
            parameters.target_scale[target],
            value,
            parameters.target_mean[target],
        )
end

function _write_diagnostics!(
    diagnostics::DistilledDiagnostics,
    state,
    parameters,
)
    diagnostics.soma_voltage_mv = _readout(state, parameters, 1)
    diagnostics.spike_probability =
        _sigmoid(_readout(state, parameters, 2))
    diagnostics.soma_spike = ifelse(
        diagnostics.spike_probability >= parameters.spike_threshold,
        1.0f0,
        0.0f0,
    )
    @inbounds for region in 1:4
        diagnostics.nmda_current[region] =
            _readout(state, parameters, 2 + region)
        diagnostics.dendritic_voltage_mv[region] =
            _readout(state, parameters, 7 + region)
    end
    diagnostics.calcium_event_probability =
        _sigmoid(_readout(state, parameters, 7))
    diagnostics.calcium_event = ifelse(
        diagnostics.calcium_event_probability >= 0.5f0,
        1.0f0,
        0.0f0,
    )
    return diagnostics
end

"""
Allocation-free scalar step. The return value is the hard soma spike only.
Analog predictions remain diagnostics and are forbidden as task-head input.
"""
function distilled_cell_step!(
    state::DistilledState,
    drive::DistilledDrive,
    diagnostics::DistilledDiagnostics,
    parameters::DistilledParameters,
)::Float32
    @inbounds for coordinate in 1:11
        state.next_value[coordinate] = _transition_coordinate(
            state.value,
            drive.event,
            parameters,
            coordinate,
        )
    end
    state.value, state.next_value = state.next_value, state.value
    _write_diagnostics!(diagnostics, state.value, parameters)
    return diagnostics.soma_spike
end

"""
Allocation-free structure-of-arrays batch step.

The return value aliases `diagnostics.soma_spike` and therefore contains only
hard `0/1` events. It never returns the spike probability or an analog state.
"""
function distilled_cell_step_soa!(
    next_state::AbstractMatrix{Float32},
    state::AbstractMatrix{Float32},
    drive::AbstractMatrix{Float32},
    diagnostics::DistilledSoADiagnostics,
    parameters::DistilledParameters,
)
    batch = size(state, 2)
    size(state) == (11, batch) ||
        throw(DimensionMismatch("state must be 11 x batch"))
    size(next_state) == size(state) ||
        throw(DimensionMismatch("next_state must match state"))
    size(drive) == (16, batch) ||
        throw(DimensionMismatch("drive must be 16 x batch"))
    length(diagnostics.soma_spike) == batch ||
        throw(DimensionMismatch("diagnostics batch differs"))

    @inbounds for sample in 1:batch
        for coordinate in 1:11
            value = parameters.transition_bias[coordinate]
            for source in 1:11
                value = muladd(
                    parameters.recurrent_weight[coordinate, source],
                    state[source, sample],
                    value,
                )
            end
            for source in 1:16
                value = muladd(
                    parameters.input_weight[coordinate, source],
                    drive[source, sample],
                    value,
                )
            end
            proposal = _activation(coordinate, value)
            decay = parameters.transition_decay[coordinate]
            next_state[coordinate, sample] = muladd(
                decay,
                state[coordinate, sample] - proposal,
                proposal,
            )
        end
        for target in 1:11
            value = parameters.readout_bias[target]
            for coordinate in 1:11
                value = muladd(
                    parameters.readout_weight[target, coordinate],
                    next_state[coordinate, sample],
                    value,
                )
            end
            physical = if target == 2 || target == 7
                value
            else
                muladd(
                    parameters.target_scale[target],
                    value,
                    parameters.target_mean[target],
                )
            end
            if target == 1
                diagnostics.soma_voltage_mv[sample] = physical
            elseif target == 2
                probability = _sigmoid(physical)
                diagnostics.spike_probability[sample] = probability
                diagnostics.soma_spike[sample] = ifelse(
                    probability >= parameters.spike_threshold,
                    1.0f0,
                    0.0f0,
                )
            elseif 3 <= target <= 6
                diagnostics.nmda_current[target - 2, sample] = physical
            elseif target == 7
                probability = _sigmoid(physical)
                diagnostics.calcium_event_probability[sample] = probability
                diagnostics.calcium_event[sample] =
                    ifelse(probability >= 0.5f0, 1.0f0, 0.0f0)
            else
                diagnostics.dendritic_voltage_mv[target - 7, sample] =
                    physical
            end
        end
    end
    return diagnostics.soma_spike
end

function _serialized_sha256(value)
    stream = IOBuffer()
    Serialization.serialize(stream, value)
    return bytes2hex(SHA.sha256(take!(stream)))
end

parameter_sha256(parameters::DistilledParameters) =
    _serialized_sha256(parameters)

function assert_parameter_sha256(
    parameters::DistilledParameters,
    expected_sha256::AbstractString,
)
    observed = parameter_sha256(parameters)
    observed == expected_sha256 || error(
        "frozen distilled parameters changed: expected " *
        "$expected_sha256, observed $observed",
    )
    return observed
end

function artifact_sha256(path::AbstractString)
    source = abspath(path)
    isfile(source) || error("distilled artifact is absent: $source")
    return bytes2hex(SHA.sha256(read(source)))
end

@inline _payload_get(payload, name::Symbol, default=nothing) =
    if payload isa AbstractDict
        get(payload, name, get(payload, String(name), default))
    elseif hasproperty(payload, name)
        getproperty(payload, name)
    else
        default
    end

function load_distilled_artifact(path::AbstractString)
    source = abspath(path)
    isfile(source) || error("distilled artifact is absent: $source")
    data = JLD2.load(source)
    haskey(data, "payload") ||
        error("distilled artifact has no payload")
    payload = data["payload"]
    _payload_get(payload, :schema) == DISTILLED_ARTIFACT_SCHEMA ||
        error("unsupported distilled artifact schema")
    parameters = _payload_get(payload, :parameters)
    parameters isa DistilledParameters ||
        error("artifact has no final DistilledParameters")
    _validate_parameters(parameters)
    _payload_get(payload, :frozen_internal, false) === true ||
        error("artifact is not marked frozen_internal")
    gate = _payload_get(payload, :gate)
    gate === nothing && error("artifact has no validation gate")
    _payload_get(gate, :passed, false) === true ||
        error("distilled artifact did not pass its held-out gate")
    expected = String(_payload_get(payload, :parameter_sha256, ""))
    isempty(expected) && error("artifact has no parameter SHA-256")
    assert_parameter_sha256(parameters, expected)

    lineage = (
        (:detailed_kernel_hash, parameters.detailed_kernel_hash),
        (:morphology_hash, parameters.morphology_hash),
        (:frozen_twin_parameter_hash,
            parameters.frozen_twin_parameter_hash),
        (:frozen_twin_artifact_hash,
            parameters.frozen_twin_artifact_hash),
        (:distillation_dataset_hash,
            parameters.distillation_dataset_hash),
        (:distillation_config_hash,
            parameters.distillation_config_hash),
    )
    for (name, value) in lineage
        String(_payload_get(payload, name, "")) == value ||
            error("artifact lineage mismatch for $name")
    end
    is_frozen(parameters) ||
        error("distilled parameters are not frozen")
    isempty(keys(trainable_parameters(parameters))) ||
        error("distilled parameters expose trainable fields")
    return parameters
end

end # module DistilledElevenStateCellFinal
