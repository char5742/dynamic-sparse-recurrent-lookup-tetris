module DistilledElevenStateCellReleaseRuntimeV5Final

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
    SEALED_RELEASE_ARTIFACT_KIND,
    SEALED_RELEASE_SCHEMA,
    SEALED_RELEASE_TYPE,
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
const SEALED_RELEASE_TYPE =
    "PaperELMTwinOfficialV2SealedReleaseV2.SealedOfficialELMRelease"
const SEALED_RELEASE_SCHEMA =
    "hd_swsnn.paper_elm_v2.sealed_release.final.v2"
const SEALED_RELEASE_ARTIFACT_KIND =
    "SealedOfficialELMReleaseV2"

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
        error("sealed-v2 distilled artifact lacks $(String(name))")
    return value
end

function _require_zero(record, name)
    Float64(_required(record, name)) == 0.0 ||
        error("sealed-v2 live/cache $(String(name)) is nonzero")
end

function _verify_sealed_v2_contract(payload)
    _get(payload, :training_input_mode, "") ==
        "sharded_streaming_v2" ||
        error("artifact did not use the bounded final-v2 shard consumer")
    _get(payload, :official_training_input_mode, "") ==
        "signed_1278_sealed_neuronio_windows_v1" ||
        error("artifact did not use sealed signed1278 NeuronIO windows")
    Int(_required(payload, :official_elm_input_dim)) ==
        OFFICIAL_ELM_INPUT_DIM ||
        error("artifact official ELM input_dim differs")
    String(_required(payload, :sealed_execution_type)) ==
        SEALED_RELEASE_TYPE ||
        error("artifact did not use exact sealed release V2 type")
    String(_required(payload, :sealed_release_schema)) ==
        SEALED_RELEASE_SCHEMA ||
        error("artifact sealed release is not final.v2")
    declared_kind = _get(
        payload,
        :sealed_release_artifact_kind,
        SEALED_RELEASE_ARTIFACT_KIND,
    )
    String(declared_kind) == SEALED_RELEASE_ARTIFACT_KIND ||
        error("artifact sealed release kind differs from V2")
    _get(payload, :legacy_3852_twin_fallback, true) === false ||
        error("artifact permits the legacy 3852-input twin")

    primary = _required(payload, :primary_frozen_twin_targets)
    _get(primary, :cache_verified_all_samples, false) === true ||
        error("not every primary cache sample was verified live")
    _get(primary, :bit_exact, false) === true ||
        error("primary target cache was not bit exact")
    Int(_required(primary, :official_elm_input_dim)) ==
        OFFICIAL_ELM_INPUT_DIM ||
        error("primary target teacher input_dim differs")
    String(_required(primary, :sealed_execution_type)) ==
        SEALED_RELEASE_TYPE ||
        error("primary target did not use sealed release V2")
    primary_schema = _get(
        primary,
        :sealed_release_schema,
        SEALED_RELEASE_SCHEMA,
    )
    String(primary_schema) == SEALED_RELEASE_SCHEMA ||
        error("primary target sealed schema differs")
    for name in (
        :soma_voltage_max_delta,
        :spike_probability_max_delta,
        :spike_logit_max_delta,
        :nmda_max_delta,
    )
        _require_zero(primary, name)
    end
    Tuple(String.(collect(_required(primary, :primary_targets)))) == (
        "soma_voltage",
        "spike_probability",
        "spike_logit",
        "nmda_soma",
        "nmda_basal",
        "nmda_apical_trunk",
        "nmda_apical_tuft",
    ) || error("primary sealed-v2 target identity/order differs")

    auxiliary =
        _required(payload, :detailed_model_auxiliary_state_targets)
    _get(auxiliary, :primary_teacher, true) === false ||
        error("detailed sparse states were mislabeled as primary")
    _get(auxiliary, :sparse_observation_grid, false) === true ||
        error("detailed auxiliary sparsity is undisclosed")
    Tuple(String.(collect(_required(auxiliary, :targets)))) == (
        "calcium_event_sparse",
        "dendritic_voltage_sparse",
    ) || error("detailed auxiliary target identity/order differs")
    String(_required(auxiliary, :teacher_hash)) ==
        String(_required(payload, :teacher_hash)) ||
        error("detailed auxiliary teacher hash differs")

    String(_required(payload, :sealed_attestation_sha256)) ==
        String(_required(primary, :sealed_attestation_sha256)) ||
        error("artifact/primary sealed attestation digest differs")

    windows = _required(payload, :neuronio_window_contract)
    full_time_steps = Int(_required(windows, :full_time_steps))
    full_time_steps >= 1_000 ||
        error("NeuronIO trajectory is too short for burn-in and window")
    Int(_required(windows, :training_ignore_steps)) == 500 ||
        error("NeuronIO training ignore interval differs")
    Int(_required(windows, :training_window_steps)) == 500 ||
        error("NeuronIO training window differs")
    Tuple(Int.(collect(_required(
        windows,
        :training_start_range,
    )))) == (
        501,
        full_time_steps - 500,
    ) || error("NeuronIO training random-window bounds differ")
    String(_required(windows, :training_sampling)) ==
        "uniform_with_replacement" ||
        error("NeuronIO training window sampling differs")
    Int(_required(windows, :heldout_burnin_steps)) == 500 ||
        error("held-out burn-in differs")
    Tuple(Int.(collect(_required(
        windows,
        :heldout_evaluation_range,
    )))) == (
        501,
        full_time_steps,
    ) || error("held-out post-burn-in evaluation range differs")
    return true
end

function load_release_runtime(path::AbstractString)
    source = abspath(path)
    data = JLD2.load(source)
    haskey(data, "payload") ||
        error("sealed-v2 distilled artifact has no payload")
    _verify_sealed_v2_contract(data["payload"])
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

end # module DistilledElevenStateCellReleaseRuntimeV5Final
