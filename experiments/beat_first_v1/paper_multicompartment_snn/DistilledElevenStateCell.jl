module DistilledElevenStateCell

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
    "paper-hay-digital-twin-eleven-state-distillation-v1"
const DISTILLED_STATE_DIM = 11
const DISTILLED_INPUT_DIM = 16
const DISTILLED_TARGET_DIM = 11

const AMPA_RECEPTOR = UInt8(1)
const NMDA_RECEPTOR = UInt8(2)
const GABAA_RECEPTOR = UInt8(3)
const CURRENT_RECEPTOR = UInt8(4)

"""
Frozen parameters of the CPU cell distilled from the PaperHayCell digital twin.

The eleven persistent coordinates are, in order:

1. four dendritic-voltage latents,
2. four NMDA/current latents,
3. one apical/context latent,
4. one soma-voltage latent, and
5. one adaptation/Ca-summary latent.

The arrays are intentionally kept outside every Tetris optimizer tree.  The
public `trainable_parameters` contract returns an empty NamedTuple and
`is_frozen` is always true.  Training this object is only allowed in the
standalone distillation program before the artifact is loaded by Tetris.
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
    teacher_sha256::String
    teacher_schema::String
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
    teacher_sha256::AbstractString,
    teacher_schema::AbstractString,
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
        String(teacher_sha256),
        String(teacher_schema),
    )
    return _validate_parameters(parameters)
end

function _validate_parameters(parameters::DistilledParameters)
    parameters.dt_ms > 0.0f0 ||
        throw(ArgumentError("dt_ms must be positive"))
    size(parameters.recurrent_weight) ==
        (DISTILLED_STATE_DIM, DISTILLED_STATE_DIM) ||
        throw(DimensionMismatch("recurrent_weight must be 11 x 11"))
    size(parameters.input_weight) ==
        (DISTILLED_STATE_DIM, DISTILLED_INPUT_DIM) ||
        throw(DimensionMismatch("input_weight must be 11 x 16"))
    size(parameters.readout_weight) ==
        (DISTILLED_TARGET_DIM, DISTILLED_STATE_DIM) ||
        throw(DimensionMismatch("readout_weight must be 11 x 11"))
    length(parameters.transition_decay) == DISTILLED_STATE_DIM ||
        throw(DimensionMismatch("transition_decay must have length 11"))
    length(parameters.transition_bias) == DISTILLED_STATE_DIM ||
        throw(DimensionMismatch("transition_bias must have length 11"))
    length(parameters.readout_bias) == DISTILLED_TARGET_DIM ||
        throw(DimensionMismatch("readout_bias must have length 11"))
    length(parameters.target_mean) == DISTILLED_TARGET_DIM ||
        throw(DimensionMismatch("target_mean must have length 11"))
    length(parameters.target_scale) == DISTILLED_TARGET_DIM ||
        throw(DimensionMismatch("target_scale must have length 11"))
    length(parameters.initial_state) == DISTILLED_STATE_DIM ||
        throw(DimensionMismatch("initial_state must have length 11"))
    size(parameters.compartment_projection, 1) == 4 ||
        throw(DimensionMismatch(
            "compartment_projection must have four dendritic rows",
        ))
    size(parameters.region_projection) == (4, 4) ||
        throw(DimensionMismatch("region_projection must be 4 x 4"))
    all(isfinite, parameters.transition_decay) ||
        throw(ArgumentError("transition decay is not finite"))
    all(value -> 0.0f0 <= value <= 1.0f0,
        parameters.transition_decay) ||
        throw(ArgumentError("transition decay must lie in [0,1]"))
    all(isfinite, parameters.target_scale) ||
        throw(ArgumentError("target scale is not finite"))
    all(value -> value > 0.0f0, parameters.target_scale) ||
        throw(ArgumentError("target scale must be positive"))
    all(isfinite, parameters.compartment_projection) ||
        throw(ArgumentError("compartment projection is not finite"))
    all(value -> value >= 0.0f0, parameters.compartment_projection) ||
        throw(ArgumentError("compartment projection must be nonnegative"))
    all(isfinite, parameters.region_projection) ||
        throw(ArgumentError("region projection is not finite"))
    all(value -> value >= 0.0f0, parameters.region_projection) ||
        throw(ArgumentError("region projection must be nonnegative"))
    isempty(parameters.teacher_sha256) &&
        throw(ArgumentError("teacher_sha256 must be recorded"))
    return parameters
end

"""
The mutable eleven-coordinate state. `next_value` is preallocated scratch and
is not a twelfth persistent coordinate.
"""
mutable struct DistilledState
    value::Vector{Float32}
    next_value::Vector{Float32}
end

function DistilledState(parameters::DistilledParameters)
    state = DistilledState(
        Vector{Float32}(undef, DISTILLED_STATE_DIM),
        Vector{Float32}(undef, DISTILLED_STATE_DIM),
    )
    return reset_state!(state, parameters)
end

"""
Preallocated receptor-event drive.

`event[(receptor - 1) * 4 + branch]` contains the spatially projected
AMPA/NMDA/GABA_A/current-clamp event for one of four reduced dendritic
coordinates. The projection is learned during distillation and frozen in the
artifact.
"""
mutable struct DistilledDrive
    event::Vector{Float32}
    compartment_projection::Matrix{Float32}
    region_projection::Matrix{Float32}
end

function DistilledDrive(parameters::DistilledParameters)
    return DistilledDrive(
        zeros(Float32, DISTILLED_INPUT_DIM),
        parameters.compartment_projection,
        parameters.region_projection,
    )
end

mutable struct DistilledDiagnostics
    soma_voltage_mv::Float32
    spike_probability::Float32
    nmda_current::Vector{Float32}
    calcium_event_probability::Float32
    calcium_event::Float32
    dendritic_voltage_mv::Vector{Float32}
end

function DistilledDiagnostics()
    diagnostics = DistilledDiagnostics(
        0.0f0,
        0.0f0,
        zeros(Float32, 4),
        0.0f0,
        0.0f0,
        zeros(Float32, 4),
    )
    return reset_diagnostics!(diagnostics)
end

"""
Allocation-free batch diagnostics for the structure-of-arrays kernel.
"""
struct DistilledSoADiagnostics
    soma_voltage_mv::Vector{Float32}
    spike_probability::Vector{Float32}
    nmda_current::Matrix{Float32}
    calcium_event_probability::Vector{Float32}
    calcium_event::Vector{Float32}
    dendritic_voltage_mv::Matrix{Float32}
end

function DistilledSoADiagnostics(batch::Integer)
    batch >= 1 || throw(ArgumentError("batch must be positive"))
    diagnostics = DistilledSoADiagnostics(
        zeros(Float32, batch),
        zeros(Float32, batch),
        zeros(Float32, 4, batch),
        zeros(Float32, batch),
        zeros(Float32, batch),
        zeros(Float32, 4, batch),
    )
    return reset_soa_diagnostics!(diagnostics)
end

@inline is_frozen(::DistilledParameters) = true
@inline trainable_parameters(::DistilledParameters) = NamedTuple()

function reset_state!(
    state::DistilledState,
    parameters::DistilledParameters,
)
    length(state.value) == DISTILLED_STATE_DIM ||
        throw(DimensionMismatch("state must have eleven coordinates"))
    copyto!(state.value, parameters.initial_state)
    copyto!(state.next_value, parameters.initial_state)
    return state
end

function reset_drive!(drive::DistilledDrive)
    fill!(drive.event, 0.0f0)
    return drive
end

function reset_diagnostics!(diagnostics::DistilledDiagnostics)
    diagnostics.soma_voltage_mv = 0.0f0
    diagnostics.spike_probability = 0.0f0
    fill!(diagnostics.nmda_current, 0.0f0)
    diagnostics.calcium_event_probability = 0.0f0
    diagnostics.calcium_event = 0.0f0
    fill!(diagnostics.dendritic_voltage_mv, 0.0f0)
    return diagnostics
end

function reset_soa_diagnostics!(diagnostics::DistilledSoADiagnostics)
    fill!(diagnostics.soma_voltage_mv, 0.0f0)
    fill!(diagnostics.spike_probability, 0.0f0)
    fill!(diagnostics.nmda_current, 0.0f0)
    fill!(diagnostics.calcium_event_probability, 0.0f0)
    fill!(diagnostics.calcium_event, 0.0f0)
    fill!(diagnostics.dendritic_voltage_mv, 0.0f0)
    return diagnostics
end

@inline function _receptor_index(receptor)
    receptor === :ampa && return Int(AMPA_RECEPTOR)
    receptor === :nmda && return Int(NMDA_RECEPTOR)
    receptor === :gaba && return Int(GABAA_RECEPTOR)
    receptor === :gabaa && return Int(GABAA_RECEPTOR)
    receptor === :current && return Int(CURRENT_RECEPTOR)
    receptor isa Integer || throw(ArgumentError(
        "receptor must be :ampa, :nmda, :gaba, :current, or 1:4",
    ))
    index = Int(receptor)
    1 <= index <= 4 ||
        throw(ArgumentError("receptor index must be in 1:4"))
    return index
end

@inline function _region_index(region::Symbol)
    region === :soma && return 1
    region === :basal && return 2
    region === :apical && return 3
    region === :tuft && return 4
    throw(ArgumentError(
        "region must be :soma, :basal, :apical, or :tuft",
    ))
end

"""
Add one anatomical receptor event through the frozen spatial projection.

An integer target denotes a PaperHayCell compartment. A Symbol denotes a
coarse region (`:soma`, `:basal`, `:apical`, or `:tuft`). This function
performs no allocation in the hot path.
"""
function add_synaptic_event!(
    drive::DistilledDrive,
    compartment_or_region,
    receptor,
    amplitude::Real,
)
    receptor_index = _receptor_index(receptor)
    offset = 4 * (receptor_index - 1)
    value = Float32(amplitude)
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
            "compartment_or_region must be an integer or Symbol",
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

@inline function _state_activation(index::Int, value::Float32)
    # Smooth latents learned from trajectories. This deliberately does not use
    # the old hand-designed hard-sigmoid plateau mechanism.
    if index == DISTILLED_STATE_DIM
        return _sigmoid(value)
    end
    return tanh(value)
end

@inline function _target_value(
    state::AbstractVector{Float32},
    parameters::DistilledParameters,
    target::Int,
)
    value = parameters.readout_bias[target]
    @inbounds for coordinate in 1:DISTILLED_STATE_DIM
        value = muladd(
            parameters.readout_weight[target, coordinate],
            state[coordinate],
            value,
        )
    end
    if target == 2 || target == 7
        return value
    end
    return muladd(
        parameters.target_scale[target],
        value,
        parameters.target_mean[target],
    )
end

@inline function _transition_coordinate(
    old_state::AbstractVector{Float32},
    drive::AbstractVector{Float32},
    parameters::DistilledParameters,
    coordinate::Int,
)
    value = parameters.transition_bias[coordinate]
    @inbounds for source in 1:DISTILLED_STATE_DIM
        value = muladd(
            parameters.recurrent_weight[coordinate, source],
            old_state[source],
            value,
        )
    end
    @inbounds for source in 1:DISTILLED_INPUT_DIM
        value = muladd(
            parameters.input_weight[coordinate, source],
            drive[source],
            value,
        )
    end
    proposal = _state_activation(coordinate, value)
    decay = parameters.transition_decay[coordinate]
    return muladd(decay, old_state[coordinate] - proposal, proposal)
end

@inline function _write_diagnostics!(
    diagnostics::DistilledDiagnostics,
    state::AbstractVector{Float32},
    parameters::DistilledParameters,
)
    diagnostics.soma_voltage_mv = _target_value(state, parameters, 1)
    spike_logit = _target_value(state, parameters, 2)
    diagnostics.spike_probability = _sigmoid(spike_logit)
    @inbounds for region in 1:4
        diagnostics.nmda_current[region] =
            _target_value(state, parameters, 2 + region)
        diagnostics.dendritic_voltage_mv[region] =
            _target_value(state, parameters, 7 + region)
    end
    calcium_logit = _target_value(state, parameters, 7)
    diagnostics.calcium_event_probability = _sigmoid(calcium_logit)
    diagnostics.calcium_event =
        ifelse(diagnostics.calcium_event_probability >= 0.5f0, 1.0f0, 0.0f0)
    return diagnostics
end

"""
Advance the frozen distilled cell by one step and return its sole external
event: a hard soma spike (`0.0f0` or `1.0f0`).

Voltages, NMDA currents and Ca events are diagnostic outputs only. They are
never returned as the task event and must not be wired to a supervised head.
"""
function distilled_cell_step!(
    state::DistilledState,
    drive::DistilledDrive,
    diagnostics::DistilledDiagnostics,
    parameters::DistilledParameters,
)::Float32
    @inbounds for coordinate in 1:DISTILLED_STATE_DIM
        state.next_value[coordinate] = _transition_coordinate(
            state.value,
            drive.event,
            parameters,
            coordinate,
        )
    end
    state.value, state.next_value = state.next_value, state.value
    _write_diagnostics!(diagnostics, state.value, parameters)
    spike = ifelse(
        diagnostics.spike_probability >= parameters.spike_threshold,
        1.0f0,
        0.0f0,
    )
    return spike
end

"""
Structure-of-arrays batch step.

All workspaces are supplied by the caller:

- `state` and `next_state` are `11 x batch`;
- `drive` is `16 x batch`;
- diagnostics matrices are `4 x batch`.

The function is allocation-free after compilation and swaps no storage.
`next_state` becomes the new state; the caller may swap its two matrices.
"""
function distilled_cell_step_soa!(
    next_state::AbstractMatrix{Float32},
    state::AbstractMatrix{Float32},
    drive::AbstractMatrix{Float32},
    diagnostics::DistilledSoADiagnostics,
    parameters::DistilledParameters,
)
    batch = size(state, 2)
    size(state) == (DISTILLED_STATE_DIM, batch) ||
        throw(DimensionMismatch("state must be 11 x batch"))
    size(next_state) == size(state) ||
        throw(DimensionMismatch("next_state must match state"))
    size(drive) == (DISTILLED_INPUT_DIM, batch) ||
        throw(DimensionMismatch("drive must be 16 x batch"))
    length(diagnostics.soma_voltage_mv) == batch ||
        throw(DimensionMismatch("diagnostics batch differs"))

    @inbounds for sample in 1:batch
        for coordinate in 1:DISTILLED_STATE_DIM
            value = parameters.transition_bias[coordinate]
            for source in 1:DISTILLED_STATE_DIM
                value = muladd(
                    parameters.recurrent_weight[coordinate, source],
                    state[source, sample],
                    value,
                )
            end
            for source in 1:DISTILLED_INPUT_DIM
                value = muladd(
                    parameters.input_weight[coordinate, source],
                    drive[source, sample],
                    value,
                )
            end
            proposal = _state_activation(coordinate, value)
            decay = parameters.transition_decay[coordinate]
            next_state[coordinate, sample] = muladd(
                decay,
                state[coordinate, sample] - proposal,
                proposal,
            )
        end

        for target in 1:DISTILLED_TARGET_DIM
            value = parameters.readout_bias[target]
            for coordinate in 1:DISTILLED_STATE_DIM
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
                diagnostics.spike_probability[sample] = _sigmoid(physical)
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
    return diagnostics.spike_probability
end

function _serialized_sha256(value)
    stream = IOBuffer()
    Serialization.serialize(stream, value)
    return bytes2hex(SHA.sha256(take!(stream)))
end

parameter_sha256(parameters::DistilledParameters) =
    _serialized_sha256(parameters)

function artifact_sha256(path::AbstractString)
    source = abspath(path)
    isfile(source) || error("distilled artifact is absent: $source")
    return bytes2hex(SHA.sha256(read(source)))
end

"""
Load and integrity-check a frozen distillation artifact.

The return value is only the frozen `DistilledParameters`; training metrics
and configuration remain in the JLD2 payload for audit tools. Both the teacher
hash and the canonical parameter hash are checked before returning.
"""
function load_distilled_artifact(path::AbstractString)
    source = abspath(path)
    isfile(source) || error("distilled artifact is absent: $source")
    data = JLD2.load(source)
    haskey(data, "payload") ||
        error("distilled artifact has no payload")
    payload = data["payload"]
    hasproperty(payload, :schema) &&
        payload.schema == DISTILLED_ARTIFACT_SCHEMA ||
        error("unsupported distilled artifact schema")
    hasproperty(payload, :parameters) ||
        error("distilled artifact has no parameters")
    parameters = payload.parameters
    parameters isa DistilledParameters ||
        error("artifact parameters have the wrong type")
    _validate_parameters(parameters)
    hasproperty(payload, :parameter_sha256) ||
        error("artifact has no parameter SHA-256")
    String(payload.parameter_sha256) == parameter_sha256(parameters) ||
        error("distilled parameter SHA-256 mismatch")
    hasproperty(payload, :teacher_sha256) ||
        error("artifact has no teacher SHA-256")
    String(payload.teacher_sha256) == parameters.teacher_sha256 ||
        error("teacher SHA-256 differs from frozen parameters")
    is_frozen(parameters) ||
        error("distilled parameters are not frozen")
    isempty(keys(trainable_parameters(parameters))) ||
        error("distilled parameters expose trainable fields")
    return parameters
end

end # module DistilledElevenStateCell
