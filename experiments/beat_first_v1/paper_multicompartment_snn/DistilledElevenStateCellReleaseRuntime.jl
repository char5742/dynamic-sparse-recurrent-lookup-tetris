module DistilledElevenStateCellReleaseRuntime

using JLD2

const _PARENT = parentmodule(@__MODULE__)
if !isdefined(_PARENT, :DistilledElevenStateCellFinal)
    Base.include(
        _PARENT,
        joinpath(@__DIR__, "DistilledElevenStateCellFinal.jl"),
    )
end
if !isdefined(_PARENT, :DistilledElevenStateCellProduction)
    Base.include(
        _PARENT,
        joinpath(@__DIR__, "DistilledElevenStateCellProduction.jl"),
    )
end
const Final = getfield(_PARENT, :DistilledElevenStateCellFinal)
const Production =
    getfield(_PARENT, :DistilledElevenStateCellProduction)

export LOCATION_INDEX_TYPE,
    OFFICIAL_LOCATION_COUNT,
    SEMANTIC_COORDINATE_NAMES,
    SEMANTIC_STATE_SCALE,
    TrustedReleaseRuntime,
    checkpoint_integrity!,
    end_run_integrity!,
    load_release_runtime,
    preflight_integrity!,
    release_add_synaptic_event!,
    release_new_diagnostics,
    release_new_drive,
    release_new_soa_diagnostics,
    release_new_state,
    trusted_cell_step!,
    trusted_cell_step_soa!

const LOCATION_INDEX_TYPE = UInt16
const OFFICIAL_LOCATION_COUNT = 642
const SEMANTIC_STATE_SCALE = :normalized_unit_interval
const SEMANTIC_COORDINATE_NAMES = (
    :basal_dendritic_voltage_1,
    :basal_dendritic_voltage_2,
    :apical_dendritic_voltage_1,
    :apical_dendritic_voltage_2,
    :basal_nmda_current_1,
    :basal_nmda_current_2,
    :apical_nmda_current_1,
    :apical_nmda_current_2,
    :apical_calcium_context,
    :soma_voltage,
    :calcium_adaptation,
)

"""
Trusted hot-loop handle.

The full parameter digest is checked at preflight, batch/checkpoint boundary,
and run end. It is deliberately not serialized in every cell step. The hot
step is safe only while the caller respects those explicit lifecycle guards.
"""
struct TrustedReleaseRuntime
    parameters::Final.DistilledParameters
    expected_parameter_sha256::String
    artifact_sha256::String
    location_mapping_sha256::String
    semantic_coordinate_names::NTuple{11,Symbol}
    semantic_state_scale::Symbol
end

@inline _get(object, name::Symbol, default=nothing) =
    if object isa AbstractDict
        get(object, name, get(object, String(name), default))
    elseif hasproperty(object, name)
        getproperty(object, name)
    else
        default
    end

function load_release_runtime(path::AbstractString)
    guarded =
        Production.load_production_distilled_artifact(path)
    data = JLD2.load(path)
    payload = data["payload"]
    semantic_gate = _get(payload, :semantic_coordinate_gate, nothing)
    semantic_gate === nothing &&
        error("release artifact has no semantic_coordinate_gate")
    _get(semantic_gate, :passed, false) === true ||
        error("semantic coordinate gate did not pass")
    names = Tuple(Symbol.(collect(_get(
        semantic_gate,
        :coordinate_names,
        (),
    ))))
    names == SEMANTIC_COORDINATE_NAMES ||
        error("semantic coordinate names/order differ")
    per_coordinate = collect(_get(
        semantic_gate,
        :per_coordinate_passed,
        Bool[],
    ))
    length(per_coordinate) == 11 && all(per_coordinate) ||
        error("one or more semantic coordinates failed")
    contract = _get(
        payload,
        :structured_transition_contract,
        nothing,
    )
    contract === nothing &&
        error("artifact has no structured transition contract")
    _get(contract, :structured_readout, false) === true ||
        error("artifact readout is not coordinate-structured")
    _get(
        contract,
        :coordinate_wise_semantic_supervision,
        false,
    ) === true ||
        error("artifact lacks coordinate-wise semantic supervision")
    _get(contract, :dense_rotational_hidden_basis, true) === false ||
        error("artifact permits a rotationally unidentified dense basis")

    parameters = guarded.parameters
    all(array -> all(isfinite, array), (
        parameters.transition_decay,
        parameters.recurrent_weight,
        parameters.input_weight,
        parameters.transition_bias,
        parameters.readout_weight,
        parameters.readout_bias,
        parameters.target_mean,
        parameters.target_scale,
        parameters.initial_state,
        parameters.compartment_projection,
        parameters.region_projection,
    )) || error("release core contains a non-finite value")
    runtime = TrustedReleaseRuntime(
        parameters,
        guarded.expected_parameter_sha256,
        guarded.artifact_sha256,
        guarded.location_mapping_sha256,
        SEMANTIC_COORDINATE_NAMES,
        SEMANTIC_STATE_SCALE,
    )
    preflight_integrity!(runtime)
    return runtime
end

function preflight_integrity!(runtime::TrustedReleaseRuntime)
    return Final.assert_parameter_sha256(
        runtime.parameters,
        runtime.expected_parameter_sha256,
    )
end

checkpoint_integrity!(runtime::TrustedReleaseRuntime) =
    preflight_integrity!(runtime)
end_run_integrity!(runtime::TrustedReleaseRuntime) =
    preflight_integrity!(runtime)

release_new_state(runtime::TrustedReleaseRuntime) =
    Final.DistilledState(runtime.parameters)

function release_new_drive(runtime::TrustedReleaseRuntime)
    # Copy projections into the mutable drive. External code can clear or
    # inspect a drive without acquiring a mutable alias to frozen parameters.
    return Final.DistilledDrive(
        zeros(Float32, 16),
        copy(runtime.parameters.compartment_projection),
        copy(runtime.parameters.region_projection),
    )
end

release_new_diagnostics(::TrustedReleaseRuntime) =
    Final.DistilledDiagnostics()
release_new_soa_diagnostics(
    ::TrustedReleaseRuntime,
    batch::Integer,
) = Final.DistilledSoADiagnostics(batch)

function release_add_synaptic_event!(
    drive::Final.DistilledDrive,
    location::Integer,
    receptor,
    amplitude::Real,
)
    location_id = LOCATION_INDEX_TYPE(location)
    1 <= location_id <= OFFICIAL_LOCATION_COUNT ||
        throw(BoundsError(1:OFFICIAL_LOCATION_COUNT, location))
    return Final.add_synaptic_event!(
        drive,
        Int(location_id),
        receptor,
        amplitude,
    )
end

function release_add_synaptic_event!(
    drive::Final.DistilledDrive,
    region::Symbol,
    receptor,
    amplitude::Real,
)
    return Final.add_synaptic_event!(
        drive,
        region,
        receptor,
        amplitude,
    )
end

"""
Trusted allocation-free hot step.

No hash/serialization occurs here. Call `preflight_integrity!` before a batch,
`checkpoint_integrity!` at every checkpoint/batch boundary, and
`end_run_integrity!` after the final update.
"""
function trusted_cell_step!(
    runtime::TrustedReleaseRuntime,
    state::Final.DistilledState,
    drive::Final.DistilledDrive,
    diagnostics::Final.DistilledDiagnostics,
)::Float32
    return Final.distilled_cell_step!(
        state,
        drive,
        diagnostics,
        runtime.parameters,
    )
end

function trusted_cell_step_soa!(
    runtime::TrustedReleaseRuntime,
    next_state::AbstractMatrix{Float32},
    state::AbstractMatrix{Float32},
    drive::AbstractMatrix{Float32},
    diagnostics::Final.DistilledSoADiagnostics,
)
    return Final.distilled_cell_step_soa!(
        next_state,
        state,
        drive,
        diagnostics,
        runtime.parameters,
    )
end

end # module DistilledElevenStateCellReleaseRuntime
