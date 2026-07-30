module DistilledElevenStateCellProduction

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

export OFFICIAL_HAY_SEGMENTS,
    PRODUCTION_MINIMUM_SPIKE_AUROC,
    FrozenDistilledRuntime,
    assert_frozen_runtime!,
    checkpoint_frozen_digest,
    load_production_distilled_artifact,
    new_diagnostics,
    new_drive,
    new_soa_diagnostics,
    new_state,
    production_cell_step!,
    production_cell_step_soa!

const OFFICIAL_HAY_SEGMENTS = 642
const PRODUCTION_MINIMUM_SPIKE_AUROC = 0.985

"""
Runtime guard for a production-qualified frozen 11-state artifact.

Julia arrays are mutable even inside an immutable struct. Therefore the
parameter digest is checked at every scalar/SoA use and every checkpoint
boundary, rather than treating a type annotation as physical immutability.
"""
struct FrozenDistilledRuntime
    parameters::Final.DistilledParameters
    expected_parameter_sha256::String
    artifact_sha256::String
    location_mapping_sha256::String
    official_segment_count::Int
end

function _serialized_sha256(value)
    stream = IOBuffer()
    Serialization.serialize(stream, value)
    return bytes2hex(SHA.sha256(take!(stream)))
end

@inline _get(object, name::Symbol, default=nothing) =
    if object isa AbstractDict
        get(object, name, get(object, String(name), default))
    elseif hasproperty(object, name)
        getproperty(object, name)
    else
        default
    end

function _required(payload, name::Symbol)
    value = _get(payload, name, nothing)
    value === nothing && error("artifact lacks required field $name")
    return value
end

function _test_metrics(payload)
    metrics = _required(payload, :metrics)
    test = _get(metrics, :test, nothing)
    test === nothing && error("artifact has no held-out test metrics")
    return test
end

function _validate_multitarget_gate(payload)
    gate = _required(payload, :gate)
    _get(gate, :passed, false) === true ||
        error("artifact gate is not passed")
    Float64(_get(
        gate,
        :minimum_spike_auroc,
        PRODUCTION_MINIMUM_SPIKE_AUROC,
    )) >= PRODUCTION_MINIMUM_SPIKE_AUROC ||
        error("artifact used a weaker spike AUROC gate than production")
    metrics = _test_metrics(payload)
    spike_auroc = Float64(_get(metrics, :spike_auroc, NaN))
    isfinite(spike_auroc) &&
        spike_auroc >= PRODUCTION_MINIMUM_SPIKE_AUROC ||
        error(
            "distilled held-out spike AUROC $spike_auroc is below " *
            "$PRODUCTION_MINIMUM_SPIKE_AUROC",
        )
    _get(gate, :multi_target_passed, false) === true ||
        error("artifact lacks a passed multi-target fidelity gate")
    for name in (
        :voltage_passed,
        :nmda_passed,
        :calcium_passed,
        :dendritic_voltage_passed,
    )
        _get(gate, name, false) === true ||
            error("artifact multi-target gate failed: $name")
    end
    all(isfinite, (
        Float64(_get(metrics, :soma_voltage_rmse_mv, NaN)),
        Float64(_get(metrics, :soma_voltage_correlation, NaN)),
        Float64(_get(metrics, :calcium_event_auroc, NaN)),
    )) || error("artifact multi-target metrics are non-finite")
    all(isfinite, Float64.(_get(
        metrics,
        :nmda_rmse_by_region,
        Float64[],
    ))) || error("artifact NMDA metrics are non-finite")
    length(_get(metrics, :nmda_rmse_by_region, ())) == 4 ||
        error("artifact needs four NMDA-region metrics")
    length(_get(metrics, :dendritic_voltage_rmse_mv, ())) == 4 ||
        error("artifact needs four dendritic-voltage metrics")
    return gate
end

function load_production_distilled_artifact(path::AbstractString)
    source = abspath(path)
    data = JLD2.load(source)
    haskey(data, "payload") || error("artifact has no payload")
    payload = data["payload"]
    parameters = Final.load_distilled_artifact(source)
    _validate_multitarget_gate(payload)
    Symbol(_get(payload, :ablation_mode, :missing)) === :full ||
        error("production Tetris requires a :full ablation artifact")
    _get(payload, :frozen_internal, false) === true ||
        error("artifact is not frozen_internal")

    segment_count = Int(_get(
        payload,
        :official_segment_count,
        0,
    ))
    segment_count == OFFICIAL_HAY_SEGMENTS ||
        error(
            "production requires the official $OFFICIAL_HAY_SEGMENTS-" *
            "segment mapping, artifact has $segment_count",
        )
    size(parameters.compartment_projection) ==
        (4, OFFICIAL_HAY_SEGMENTS) ||
        error("artifact location projection is not 4 x 642")
    observed_mapping_hash =
        _serialized_sha256(parameters.compartment_projection)
    expected_mapping_hash =
        String(_required(payload, :location_mapping_sha256))
    observed_mapping_hash == expected_mapping_hash ||
        error("artifact location-mapping SHA-256 mismatch")

    for name in (
        :cell_mechanism_sha256,
        :digital_twin_sha256,
        :official_modeldb_source_hash,
        :teacher_hash,
    )
        isempty(String(_required(payload, name))) &&
            error("artifact lineage field $name is empty")
    end
    expected_parameter_hash =
        String(_required(payload, :parameter_sha256))
    Final.assert_parameter_sha256(parameters, expected_parameter_hash)
    return FrozenDistilledRuntime(
        parameters,
        expected_parameter_hash,
        Final.artifact_sha256(source),
        expected_mapping_hash,
        segment_count,
    )
end

function assert_frozen_runtime!(runtime::FrozenDistilledRuntime)
    Final.assert_parameter_sha256(
        runtime.parameters,
        runtime.expected_parameter_sha256,
    )
    _serialized_sha256(
        runtime.parameters.compartment_projection,
    ) == runtime.location_mapping_sha256 ||
        error("frozen location projection changed")
    return runtime.expected_parameter_sha256
end

checkpoint_frozen_digest(runtime::FrozenDistilledRuntime) =
    assert_frozen_runtime!(runtime)

new_state(runtime::FrozenDistilledRuntime) =
    Final.DistilledState(runtime.parameters)
new_drive(runtime::FrozenDistilledRuntime) =
    Final.DistilledDrive(runtime.parameters)
new_diagnostics(::FrozenDistilledRuntime) =
    Final.DistilledDiagnostics()
new_soa_diagnostics(::FrozenDistilledRuntime, batch::Integer) =
    Final.DistilledSoADiagnostics(batch)

function production_cell_step!(
    runtime::FrozenDistilledRuntime,
    state::Final.DistilledState,
    drive::Final.DistilledDrive,
    diagnostics::Final.DistilledDiagnostics,
)::Float32
    assert_frozen_runtime!(runtime)
    return Final.distilled_cell_step!(
        state,
        drive,
        diagnostics,
        runtime.parameters,
    )
end

function production_cell_step_soa!(
    runtime::FrozenDistilledRuntime,
    next_state::AbstractMatrix{Float32},
    state::AbstractMatrix{Float32},
    drive::AbstractMatrix{Float32},
    diagnostics::Final.DistilledSoADiagnostics,
)
    assert_frozen_runtime!(runtime)
    return Final.distilled_cell_step_soa!(
        next_state,
        state,
        drive,
        diagnostics,
        runtime.parameters,
    )
end

end # module DistilledElevenStateCellProduction
