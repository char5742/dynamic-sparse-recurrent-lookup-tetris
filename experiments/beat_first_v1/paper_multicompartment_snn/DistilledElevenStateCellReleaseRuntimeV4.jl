module DistilledElevenStateCellReleaseRuntimeV4

using JLD2

const _PARENT = parentmodule(@__MODULE__)
if !isdefined(_PARENT, :DistilledElevenStateCellReleaseRuntimeV3)
    Base.include(
        _PARENT,
        joinpath(
            @__DIR__,
            "DistilledElevenStateCellReleaseRuntimeV3.jl",
        ),
    )
end
const V3 =
    getfield(_PARENT, :DistilledElevenStateCellReleaseRuntimeV3)

export LOCATION_INDEX_TYPE,
    OFFICIAL_LOCATION_COUNT,
    OFFICIAL_ELM_INPUT_DIM,
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

const LOCATION_INDEX_TYPE = V3.LOCATION_INDEX_TYPE
const OFFICIAL_LOCATION_COUNT = V3.OFFICIAL_LOCATION_COUNT
const OFFICIAL_ELM_INPUT_DIM = 1_278
const SEMANTIC_COORDINATE_NAMES = V3.SEMANTIC_COORDINATE_NAMES
const SEMANTIC_STATE_SCALE = V3.SEMANTIC_STATE_SCALE
const TrustedReleaseRuntime = V3.TrustedReleaseRuntime

@inline _get(object, name::Symbol, default=nothing) =
    if object isa AbstractDict
        get(object, name, get(object, String(name), default))
    elseif hasproperty(object, name)
        getproperty(object, name)
    else
        default
    end

function _required(object, name::Symbol)
    value = _get(object, name, nothing)
    value === nothing &&
        error("official release artifact lacks $(String(name))")
    return value
end

function _require_zero_delta(record, name::Symbol)
    Float64(_required(record, name)) == 0.0 ||
        error("official live/cache $(String(name)) is nonzero")
end

"""
Load only an accepted 11-state artifact distilled from the sealed 1278-input
official ELM release.

V3 performs all numerical, semantic, location-map, and held-out gates.  V4
adds the teacher identity boundary and requires bit-exact all-sample live-cache
verification, with detailed Hay voltage/Ca recorded separately as auxiliary
state targets.
"""
function load_release_runtime(path::AbstractString)
    source = abspath(path)
    data = JLD2.load(source)
    haskey(data, "payload") ||
        error("official release artifact has no payload")
    payload = data["payload"]
    _get(payload, :training_input_mode, "") ==
        "sharded_streaming_official_1278_v2" ||
        error("artifact was not trained on the official 1278 stream")
    Int(_required(payload, :official_elm_input_dim)) ==
        OFFICIAL_ELM_INPUT_DIM ||
        error("artifact official ELM input_dim differs")
    String(_required(payload, :official_elm_execution_type)) ==
        "PaperELMTwinOfficialV2ReleaseExecution.VerifiedOfficialELMExecution" ||
        error("artifact did not use the sealed official execution type")
    _get(payload, :legacy_3852_twin_fallback, true) === false ||
        error("artifact permits the legacy 3852-input twin")

    primary = _required(payload, :primary_frozen_twin_targets)
    _get(primary, :cache_verified_all_samples, false) === true ||
        error("not every primary target cache was verified live")
    _get(primary, :bit_exact, false) === true ||
        error("primary target cache was not bit exact")
    Int(_required(primary, :official_elm_input_dim)) ==
        OFFICIAL_ELM_INPUT_DIM ||
        error("primary target teacher input_dim differs")
    String(_required(primary, :official_elm_execution_type)) ==
        String(_required(payload, :official_elm_execution_type)) ||
        error("primary target execution identity differs")
    for name in (
        :soma_voltage_max_delta,
        :spike_probability_max_delta,
        :spike_logit_max_delta,
        :nmda_max_delta,
    )
        _require_zero_delta(primary, name)
    end
    Tuple(String.(collect(_required(primary, :primary_targets)))) == (
        "soma_voltage",
        "spike_probability",
        "spike_logit",
        "nmda_soma",
        "nmda_basal",
        "nmda_apical_trunk",
        "nmda_apical_tuft",
    ) || error("primary official ELM target identity/order differs")

    auxiliary =
        _required(payload, :detailed_model_auxiliary_state_targets)
    _get(auxiliary, :primary_teacher, true) === false ||
        error("detailed sparse states were mislabeled as primary targets")
    _get(auxiliary, :sparse_observation_grid, false) === true ||
        error("detailed auxiliary state sparsity is undisclosed")
    Tuple(String.(collect(_required(auxiliary, :targets)))) == (
        "calcium_event_sparse",
        "dendritic_voltage_sparse",
    ) || error("detailed auxiliary target identity/order differs")
    String(_required(auxiliary, :teacher_hash)) ==
        String(_required(payload, :teacher_hash)) ||
        error("detailed auxiliary teacher hash differs")

    String(_required(payload, :verified_attestation_sha256)) ==
        String(_required(primary, :verified_attestation_sha256)) ||
        error("artifact/primary attestation digest differs")
    return V3.load_release_runtime(source)
end

const preflight_integrity! = V3.preflight_integrity!
const checkpoint_integrity! = V3.checkpoint_integrity!
const end_run_integrity! = V3.end_run_integrity!
const release_new_state = V3.release_new_state
const release_new_drive = V3.release_new_drive
const release_new_diagnostics = V3.release_new_diagnostics
const release_new_soa_diagnostics = V3.release_new_soa_diagnostics
const release_add_synaptic_event! = V3.release_add_synaptic_event!
const trusted_cell_step! = V3.trusted_cell_step!
const trusted_cell_step_soa! = V3.trusted_cell_step_soa!

end # module DistilledElevenStateCellReleaseRuntimeV4
