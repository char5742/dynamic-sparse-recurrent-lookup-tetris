module DistilledElevenStateCellReleaseRuntimeV2

using JLD2
using Serialization
using SHA

const _PARENT = parentmodule(@__MODULE__)
if !isdefined(_PARENT, :DistilledElevenStateCellFinal)
    Base.include(
        _PARENT,
        joinpath(@__DIR__, "DistilledElevenStateCellFinal.jl"),
    )
end
const Final = getfield(_PARENT, :DistilledElevenStateCellFinal)

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
const MINIMUM_SPIKE_AUROC = 0.985

struct TrustedReleaseRuntime
    parameters::Final.DistilledParameters
    expected_parameter_sha256::String
    artifact_sha256::String
    location_mapping_sha256::String
    source_segment_catalog_sha256::String
end

@inline _get(object, name::Symbol, default=nothing) =
    if object isa AbstractDict
        get(object, name, get(object, String(name), default))
    elseif hasproperty(object, name)
        getproperty(object, name)
    else
        default
    end

function _sha256(value)
    stream = IOBuffer()
    Serialization.serialize(stream, value)
    return bytes2hex(SHA.sha256(take!(stream)))
end

function _required(payload, name::Symbol)
    value = _get(payload, name, nothing)
    value === nothing && error("release artifact lacks $name")
    return value
end

function load_release_runtime(path::AbstractString)
    source = abspath(path)
    data = JLD2.load(source)
    haskey(data, "payload") || error("release artifact has no payload")
    payload = data["payload"]
    parameters = Final.load_distilled_artifact(source)
    _get(payload, :ablation_mode, :missing) === :full ||
        error("release artifact is not the :full arm")
    Int(_required(payload, :official_segment_count)) ==
        OFFICIAL_LOCATION_COUNT ||
        error("release artifact does not expose 642 official segments")
    size(parameters.compartment_projection) == (4, 642) ||
        error("release projection is not 4 x 642")

    metrics = _required(payload, :metrics)
    test = _required(metrics, :test)
    gate = _required(payload, :gate)
    _get(gate, :passed, false) === true ||
        error("release multi-target gate is not passed")
    Float64(_get(gate, :minimum_spike_auroc, 0.0)) >=
        MINIMUM_SPIKE_AUROC ||
        error("release used a spike gate below 0.985")
    Float64(_get(test, :spike_auroc, NaN)) >=
        MINIMUM_SPIKE_AUROC ||
        error("release held-out spike AUROC is below 0.985")
    _get(gate, :multi_target_passed, false) === true ||
        error("release multi-target gate failed")
    for name in (
        :voltage_passed,
        :nmda_passed,
        :calcium_passed,
        :dendritic_voltage_passed,
        :semantic_state_passed,
    )
        _get(gate, name, false) === true ||
            error("release gate failed: $name")
    end

    semantic = _required(payload, :semantic_coordinate_gate)
    _get(semantic, :passed, false) === true ||
        error("semantic coordinate gate failed")
    Tuple(Symbol.(collect(_get(
        semantic,
        :coordinate_names,
        (),
    )))) == SEMANTIC_COORDINATE_NAMES ||
        error("semantic coordinate identity/order differs")
    per_coordinate = collect(_get(
        semantic,
        :per_coordinate_passed,
        Bool[],
    ))
    length(per_coordinate) == 11 && all(per_coordinate) ||
        error("not every semantic coordinate passed")
    contract = _required(payload, :structured_transition_contract)
    _get(contract, :structured_readout, false) === true ||
        error("release readout is not structured")
    _get(
        contract,
        :coordinate_wise_semantic_supervision,
        false,
    ) === true ||
        error("release lacks coordinate-wise semantic supervision")
    _get(contract, :dense_rotational_hidden_basis, true) === false ||
        error("release permits a rotationally unidentified basis")

    source_catalog =
        String(_required(payload, :source_segment_catalog_sha256))
    expected_mapping =
        String(_required(payload, :location_mapping_sha256))
    observed_mapping = _sha256((
        source_catalog,
        _required(payload, :official_segment_region),
        parameters.compartment_projection,
    ))
    observed_mapping == expected_mapping ||
        error("official 642-segment mapping hash mismatch")
    expected_parameter =
        String(_required(payload, :parameter_sha256))
    Final.assert_parameter_sha256(parameters, expected_parameter)
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
    )) || error("release parameters contain a non-finite value")
    return TrustedReleaseRuntime(
        parameters,
        expected_parameter,
        Final.artifact_sha256(source),
        expected_mapping,
        source_catalog,
    )
end

preflight_integrity!(runtime::TrustedReleaseRuntime) =
    Final.assert_parameter_sha256(
        runtime.parameters,
        runtime.expected_parameter_sha256,
    )
checkpoint_integrity!(runtime::TrustedReleaseRuntime) =
    preflight_integrity!(runtime)
end_run_integrity!(runtime::TrustedReleaseRuntime) =
    preflight_integrity!(runtime)

release_new_state(runtime::TrustedReleaseRuntime) =
    Final.DistilledState(runtime.parameters)

function release_new_drive(runtime::TrustedReleaseRuntime)
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
    1 <= location <= OFFICIAL_LOCATION_COUNT ||
        throw(BoundsError(1:OFFICIAL_LOCATION_COUNT, location))
    location_id = LOCATION_INDEX_TYPE(location)
    return Final.add_synaptic_event!(
        drive,
        Int(location_id),
        receptor,
        amplitude,
    )
end

release_add_synaptic_event!(
    drive::Final.DistilledDrive,
    region::Symbol,
    receptor,
    amplitude::Real,
) = Final.add_synaptic_event!(
    drive,
    region,
    receptor,
    amplitude,
)

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

end # module DistilledElevenStateCellReleaseRuntimeV2
