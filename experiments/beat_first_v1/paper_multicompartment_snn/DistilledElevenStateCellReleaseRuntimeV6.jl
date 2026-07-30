module DistilledElevenStateCellReleaseRuntimeV6

# Canonical runtime verifier for distillation from the exact final.v2 sealed
# ELM.  The caller must pin both this artifact's bytes and the upstream sealed
# attestation; self-labelled payload provenance is not an external identity.

using JLD2
using SHA

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
const STRICT_GATE_SCHEMA =
    "hd_swsnn.eleven_state.strict_gate.final.v2"
const PRIMARY_REPLAY_SCHEMA =
    "hd_swsnn.distillation.primary_cache_live_replay.final.v2"

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

function _sha(value, label)
    digest = lowercase(String(value))
    occursin(r"^[0-9a-f]{64}$", digest) ||
        error("$label is not a complete SHA-256")
    return digest
end

function _file_sha(path)
    return bytes2hex(SHA.sha256(read(abspath(path))))
end

function _require_zero(record, name)
    Float64(_required(record, name)) == 0.0 ||
        error("sealed-v2 live/cache $(String(name)) is nonzero")
end

function _require_all_true(record, name, count)
    values = Bool.(collect(_required(record, name)))
    length(values) == count && all(values) ||
        error("sealed-v2 gate did not pass every $(String(name))")
    return values
end

function _verify_split_identity(payload)
    splits = _required(payload, :split_identity)
    for name in (:train, :validation, :test)
        split = _required(splits, name)
        Int(_required(split, :count)) > 0 ||
            error("$(String(name)) split is empty")
        _sha(
            _required(split, :indices_sha256),
            "$(String(name)) split indices",
        )
    end
    protocol = _required(payload, :evaluation_protocol)
    Int(_required(protocol, :candidate_restarts)) >= 1 ||
        error("distillation did not train any restart")
    Int(_required(protocol, :validation_evaluations)) ==
        Int(_required(protocol, :candidate_restarts)) ||
        error("every restart must have exactly one validation evaluation")
    Int(_required(protocol, :test_evaluations)) == 1 ||
        error("held-out test was not evaluated exactly once")
    String(_required(protocol, :selection_split)) == "validation" ||
        error("candidate selection used a non-validation split")
    String(_required(protocol, :final_gate_split)) == "test" ||
        error("final release gate used a non-test split")
    return splits
end

function _verify_strict_gate(payload, splits)
    metrics = _required(payload, :metrics)
    test = _required(metrics, :test)
    String(_required(test, :evaluation_split)) == "test" ||
        error("stored final metrics are not the exact test split")
    _required(test, :exact_dataset_split) === true ||
        error("stored final metrics used caller-selected indices")
    String(_required(test, :evaluated_indices_sha256)) ==
        String(_required(_required(splits, :test), :indices_sha256)) ||
        error("stored final metric test identity differs")
    Int(_required(test, :evaluated_index_count)) ==
        Int(_required(_required(splits, :test), :count)) ||
        error("stored final metric test count differs")
    _required(test, :sparse_auxiliary_metrics_use_observed_times_only) ===
        true ||
        error("sparse auxiliary gate included interpolated targets")
    _required(test, :interpolated_auxiliary_values_excluded_from_gate) ===
        true ||
        error("interpolated auxiliary values entered the gate")

    gate = _required(payload, :gate)
    String(_required(gate, :gate_schema)) == STRICT_GATE_SCHEMA ||
        error("artifact did not use the strict per-coordinate gate")
    _required(gate, :passed) === true ||
        error("strict release gate failed")
    _required(gate, :per_region_and_branch_gating) === true ||
        error("artifact permits mean-over-region gating")
    _required(gate, :spike_passed) === true ||
        error("strict spike gate failed")
    _required(gate, :multi_target_passed) === true ||
        error("strict multi-target gate failed")
    _required(gate, :voltage_passed) === true ||
        error("strict voltage gate failed")
    _required(gate, :nmda_passed) === true ||
        error("strict NMDA gate failed")
    _require_all_true(gate, :nmda_region_passed, 4)
    _required(gate, :calcium_passed) === true ||
        error("strict calcium gate failed")
    _required(gate, :dendritic_voltage_passed) === true ||
        error("strict dendritic gate failed")
    _require_all_true(gate, :dendritic_branch_passed, 4)
    _required(gate, :semantic_state_passed) === true ||
        error("strict semantic-state gate failed")
    _require_all_true(gate, :semantic_coordinate_passed, 11)
    _required(gate, :observation_counts_passed) === true ||
        error("strict gate has missing observations")
    Float64(_required(gate, :minimum_spike_auroc)) >= 0.985 ||
        error("strict spike gate is weaker than 0.985")
    return true
end

function _verify_sealed_v2_contract(
    payload,
    expected_sealed_attestation_sha256,
)
    String(_required(payload, :training_input_mode)) ==
        "sharded_streaming_v2" ||
        error("artifact did not use bounded shard streaming")
    String(_required(payload, :official_training_input_mode)) ==
        "signed_1278_sealed_v2_neuronio_windows_v1" ||
        error("artifact did not use sealed-v2 NeuronIO windows")
    Int(_required(payload, :official_elm_input_dim)) ==
        OFFICIAL_ELM_INPUT_DIM ||
        error("artifact official ELM input_dim differs")
    String(_required(payload, :sealed_execution_type)) ==
        SEALED_RELEASE_TYPE ||
        error("artifact did not use exact sealed release V2 type")
    String(_required(payload, :sealed_release_schema)) ==
        SEALED_RELEASE_SCHEMA ||
        error("artifact sealed release is not final.v2")
    String(_required(payload, :sealed_release_artifact_kind)) ==
        SEALED_RELEASE_ARTIFACT_KIND ||
        error("artifact sealed release kind differs from V2")
    _get(payload, :legacy_3852_twin_fallback, true) === false ||
        error("artifact permits the legacy 3852-input twin")
    attestation =
        _sha(_required(payload, :sealed_attestation_sha256), "attestation")
    attestation ==
        _sha(expected_sealed_attestation_sha256, "expected attestation") ||
        error("artifact belongs to another externally pinned sealed ELM")

    source = _required(payload, :source_bound_sealed_elm)
    for name in (
        :source_manifest_sha256,
        :source_teacher_contract_sha256,
        :parameter_sha256,
        :base_artifact_sha256,
        :sealed_attestation_sha256,
    )
        _sha(_required(source, name), "source-bound $(String(name))")
    end
    String(_required(source, :sealed_attestation_sha256)) == attestation ||
        error("source-bound ELM attestation differs")
    String(_required(source, :sealed_release_schema)) ==
        SEALED_RELEASE_SCHEMA ||
        error("source-bound ELM schema differs")
    String(_required(source, :sealed_execution_type)) ==
        SEALED_RELEASE_TYPE ||
        error("source-bound ELM type differs")
    isempty(String(_required(source, :executable_mlp_activation))) &&
        error("ELM executable activation is absent")
    isempty(String(_required(source, :compatibility_profile))) &&
        error("ELM compatibility profile is absent")

    primary = _required(payload, :primary_frozen_twin_targets)
    _required(primary, :cache_verified_all_samples) === true ||
        error("not every primary cache sample was verified live")
    _required(primary, :bit_exact) === true ||
        error("primary target cache was not bit exact")
    String(_required(primary, :measurement_schema)) ==
        PRIMARY_REPLAY_SCHEMA ||
        error("primary cache measurement schema differs")
    _sha(
        _required(primary, :measurement_sha256),
        "primary replay measurement",
    )
    Int(_required(primary, :official_elm_input_dim)) ==
        OFFICIAL_ELM_INPUT_DIM ||
        error("primary teacher input_dim differs")
    String(_required(primary, :sealed_execution_type)) ==
        SEALED_RELEASE_TYPE ||
        error("primary targets did not use sealed release V2")
    String(_required(primary, :sealed_release_schema)) ==
        SEALED_RELEASE_SCHEMA ||
        error("primary target sealed schema differs")
    String(_required(primary, :sealed_release_artifact_kind)) ==
        SEALED_RELEASE_ARTIFACT_KIND ||
        error("primary target sealed kind differs")
    String(_required(primary, :sealed_attestation_sha256)) == attestation ||
        error("primary target attestation differs")
    for name in (
        :soma_voltage_max_delta,
        :spike_probability_max_delta,
        :spike_logit_max_delta,
        :nmda_max_delta,
    )
        _require_zero(primary, name)
    end

    auxiliary =
        _required(payload, :detailed_model_auxiliary_state_targets)
    _required(auxiliary, :primary_teacher) === false ||
        error("detailed sparse states were mislabeled as primary")
    _required(auxiliary, :sparse_observation_grid) === true ||
        error("detailed auxiliary sparsity is undisclosed")

    windows = _required(payload, :neuronio_window_contract)
    full_time_steps = Int(_required(windows, :full_time_steps))
    full_time_steps > 1_000 ||
        error("trajectory has no valid Python-exclusive training window")
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
    ) || error("held-out evaluation range differs")

    splits = _verify_split_identity(payload)
    _verify_strict_gate(payload, splits)
    return true
end

function load_release_runtime(
    path::AbstractString;
    expected_artifact_sha256::AbstractString,
    expected_sealed_attestation_sha256::AbstractString,
)
    source = abspath(path)
    observed_artifact = _file_sha(source)
    observed_artifact ==
        _sha(expected_artifact_sha256, "expected distilled artifact") ||
        error("distilled artifact bytes differ from the external pin")
    data = JLD2.load(source)
    haskey(data, "payload") ||
        error("sealed-v2 distilled artifact has no payload")
    _verify_sealed_v2_contract(
        data["payload"],
        expected_sealed_attestation_sha256,
    )
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

end # module DistilledElevenStateCellReleaseRuntimeV6
