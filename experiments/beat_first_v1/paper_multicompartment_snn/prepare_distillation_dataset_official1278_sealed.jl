module DistillationDatasetBridgeOfficial1278Sealed

# Canonical official-1278 bridge.
#
# The older `prepare_distillation_dataset_official1278.jl` remains a diagnostic
# comparison path.  This entry point replaces its model-verification hook and
# accepts only a raw Final frozen ELM which is independently sealed against the
# hash-verified final.v2 teacher manifest and every source shard.

using JSON3

const _PARENT = parentmodule(@__MODULE__)
if !isdefined(_PARENT, :DistillationDatasetBridgeOfficial1278)
    Base.include(
        _PARENT,
        joinpath(
            @__DIR__,
            "prepare_distillation_dataset_official1278.jl",
        ),
    )
end
if !isdefined(Main, :PaperELMTwinOfficialV2SealedRelease)
    Base.include(
        Main,
        joinpath(
            @__DIR__,
            "PaperELMTwinOfficialV2SealedRelease.jl",
        ),
    )
end

const Diagnostic =
    getfield(_PARENT, :DistillationDatasetBridgeOfficial1278)
const Sealed = Main.PaperELMTwinOfficialV2SealedRelease
const OfficialTwin = Diagnostic.OfficialTwin
const ReleaseBridge = Diagnostic.ReleaseBridge
const Production = Diagnostic.Production
const OrderedBridge = Diagnostic.OrderedBridge
const FinalBridge = Diagnostic.FinalBridge
const V6 = Diagnostic.V6
const Legacy = Diagnostic.Legacy
const BaseBridge = Diagnostic.BaseBridge
const LegacyTwinRuntime = Diagnostic.LegacyTwinRuntime
const ReleaseStreamingPrepareConfig =
    Diagnostic.ReleaseStreamingPrepareConfig

export FINAL_NEURON_SCHEMA,
    RELEASE_DATASET_SCHEMA,
    RELEASE_SHARD_SCHEMA,
    SEALED_RELEASE_SCHEMA,
    OFFICIAL_INPUT_DIM,
    ReleaseStreamingPrepareConfig,
    SealedOfficialBridgeTwin,
    prepare_distillation_dataset_release,
    main

const FINAL_NEURON_SCHEMA = Diagnostic.FINAL_NEURON_SCHEMA
const RELEASE_DATASET_SCHEMA = Diagnostic.RELEASE_DATASET_SCHEMA
const RELEASE_SHARD_SCHEMA = Diagnostic.RELEASE_SHARD_SCHEMA
const SEALED_RELEASE_SCHEMA = Sealed.SEALED_RELEASE_SCHEMA
const OFFICIAL_INPUT_DIM = Diagnostic.OFFICIAL_INPUT_DIM

const NEURONIO_REFERENCE_COMMIT =
    "52e68a6d39523ac6613a586699b116e8e606dda3"
const NEURONIO_LOADER_SHA256 =
    "db83c96060f211ee4889dc0bd6a2a4a8584cc637307df9484e4dfcb77ef6bac8"
const NEURONIO_EVALUATOR_SHA256 =
    "1b1516e33790ce0183b83d1b419a323dd93f79c312392041da1438cb14a43ece"
const NEURONIO_IGNORE_MS = 500
const NEURONIO_WINDOW_MS = 500

"""
Runtime capability used by the sharded bridge after sealed source verification.

Construction is intentionally confined to `_verify_twin`: callers cannot feed
metrics, targets, pass flags, or identity objects into this bridge.
"""
struct SealedOfficialBridgeTwin{R,M,D,P}
    release::R
    model::M
    metadata::D
    payload::P
    parameter_sha256::String
    artifact_sha256::String
    attestation_sha256::String
    source_manifest_path::String
    source_root::String
end

function _validate_official_frozen(frozen)
    config = frozen.model.config
    config.num_input == OFFICIAL_INPUT_DIM ||
        error("sealed official ELM does not have 1278 inputs")
    config.nmda_regions == 4 ||
        error("sealed official ELM does not have four NMDA outputs")
    config.num_branch == OfficialTwin.OFFICIAL_ELM_BRANCHES ||
        error("sealed official ELM branch count differs")
    config.num_synapse_per_branch ==
        OfficialTwin.OFFICIAL_ELM_SYNAPSES_PER_BRANCH ||
        error("sealed official ELM synapses/branch differs")
    config.input_to_synapse_routing === :neuronio_routing ||
        error("sealed official ELM routing is not neuronio_routing")
    config.num_memory == 1_000 ||
        error("sealed TwinProp ELM must have 1,000 memory units")
    config.hidden_size == 2_000 ||
        error("sealed TwinProp ELM must have 2,000 hidden units")
    config.memory_tau_min_ms == 0.1f0 ||
        error("sealed TwinProp ELM minimum tau differs")
    config.memory_tau_max_ms == 300.0f0 ||
        error("sealed TwinProp ELM maximum tau differs")
    config.delta_t_ms == 1.0f0 ||
        error("sealed TwinProp ELM must run at 1 ms")
    return frozen
end

function _bridge_twin(
    release::Sealed.SealedOfficialELMRelease,
    source::BaseBridge._Source,
)
    frozen = _validate_official_frozen(release.frozen)
    payload = release.attestation.payload
    payload.schema == Sealed.SEALED_RELEASE_SCHEMA ||
        error("sealed official ELM schema differs")
    payload.outcome.gate_passed === true ||
        error("sealed official ELM fixed held-out gate failed")
    payload.outcome.caller_metrics_accepted === false ||
        error("sealed official ELM accepted caller metrics")
    payload.outcome.caller_targets_accepted === false ||
        error("sealed official ELM accepted caller targets")
    payload.teacher.manifest_sha256 == source.source_manifest_hash ||
        error("sealed ELM/teacher manifest mismatch")
    payload.teacher.teacher_contract_sha256 ==
        source.detailed_teacher_hash ||
        error("sealed ELM/teacher contract mismatch")
    payload.teacher.source_dataset_sha256 ==
        source.source_dataset_hash ||
        error("sealed ELM/source dataset mismatch")
    return SealedOfficialBridgeTwin(
        release,
        Diagnostic.OfficialBridgeModel(
            Diagnostic.OfficialBridgeConfig(
                OFFICIAL_INPUT_DIM,
                OfficialTwin.Core.HAY_TOTAL_SEGMENTS,
                4,
                frozen.model.config.delta_t_ms,
            ),
        ),
        frozen.metadata,
        payload,
        frozen.parameter_sha256,
        frozen.artifact_sha256,
        release.attestation.attestation_sha256,
        source.manifest_path,
        source.root,
    )
end

# This method intentionally replaces the diagnostic VerifiedOfficialELMTwin
# loader in this standalone canonical entry point.  A caller-metric
# VerifiedOfficialELMTwin or legacy ReleaseExecution artifact is rejected by
# `load_frozen_official_elm` before any dataset output can be staged.
function BaseBridge._verify_twin(
    config::BaseBridge.PrepareDistillationConfig,
    source::BaseBridge._Source,
)
    path = abspath(config.frozen_twin_path)
    isfile(path) ||
        error("raw official Final ELM artifact is absent: $path")
    frozen = OfficialTwin.load_frozen_official_elm(path)
    OfficialTwin.assert_frozen_official_elm_unchanged(frozen)
    BaseBridge._expected_hash(
        "twin parameter",
        frozen.parameter_sha256,
        config.expected_twin_parameter_sha256,
    )
    BaseBridge._expected_hash(
        "twin artifact",
        frozen.artifact_sha256,
        config.expected_twin_artifact_sha256,
    )
    release = Sealed.attest_sealed_official_elm_release(
        source.manifest_path,
        source.root,
        frozen,
    )
    bridge = _bridge_twin(release, source)
    integrity = (;
        frozen=true,
        verified=true,
        sealed_source_verified=true,
        max_delta=0.0f0,
        parameter_sha256=bridge.parameter_sha256,
        artifact_sha256=bridge.artifact_sha256,
        attestation_sha256=bridge.attestation_sha256,
        sealed_release_schema=Sealed.SEALED_RELEASE_SCHEMA,
    )
    return (
        bridge,
        integrity,
        release.attestation.payload.metrics,
        BaseBridge._sha256_file(path),
    )
end

function LegacyTwinRuntime.assert_frozen_unchanged(
    frozen::SealedOfficialBridgeTwin;
    expected_artifact_sha256::AbstractString="",
)
    OfficialTwin.assert_frozen_official_elm_unchanged(
        frozen.release.frozen,
    )
    isempty(expected_artifact_sha256) ||
        frozen.artifact_sha256 ==
            lowercase(String(expected_artifact_sha256)) ||
        error("sealed official ELM artifact changed")
    Sealed.verify_sealed_official_elm_release(
        frozen.release,
        frozen.source_manifest_path,
        frozen.source_root;
        require_gate=true,
        require_production=false,
    )
    return (;
        frozen=true,
        verified=true,
        sealed_source_verified=true,
        max_delta=0.0f0,
        parameter_sha256=frozen.parameter_sha256,
        artifact_sha256=frozen.artifact_sha256,
        attestation_sha256=frozen.attestation_sha256,
        sealed_release_schema=Sealed.SEALED_RELEASE_SCHEMA,
    )
end

LegacyTwinRuntime.twin_input_layout(
    ::SealedOfficialBridgeTwin,
) = OfficialTwin.official_elm_input_layout()

function V6._infer_sample(
    frozen::SealedOfficialBridgeTwin,
    sample::Diagnostic.OfficialSparseSample,
    time_steps,
    chunk,
)
    voltage = Vector{Float32}(undef, time_steps)
    spike = Vector{Float32}(undef, time_steps)
    spike_logit = Vector{Float32}(undef, time_steps)
    nmda = Matrix{Float32}(undef, 4, time_steps)
    state = nothing
    for first_time in 1:chunk:time_steps
        last_time = min(first_time + chunk - 1, time_steps)
        input = V6._dense_time_chunk(
            sample,
            frozen.model.config,
            first_time,
            last_time,
        )
        prediction = OfficialTwin.twin_forward(
            frozen.release.frozen,
            input;
            initial_state=state,
        )
        for (label, values) in (
            ("voltage", prediction.voltage),
            ("spike probability", prediction.spike_probability),
            ("spike logit", prediction.spike_logit),
            ("NMDA", prediction.nmda),
        )
            all(isfinite, values) ||
                error("sealed official ELM produced non-finite $label")
        end
        voltage[first_time:last_time] .= vec(prediction.voltage)
        spike[first_time:last_time] .=
            vec(prediction.spike_probability)
        spike_logit[first_time:last_time] .=
            vec(prediction.spike_logit)
        nmda[:, first_time:last_time] .=
            reshape(prediction.nmda, 4, :)
        state = prediction.final_state
    end
    return (; voltage, spike, spike_logit, nmda)
end

function _neuronio_contract(time_steps::Int, sample_dt_ms::Real)
    sample_dt_ms == 1.0 ||
        error("official NeuronIO bridge requires 1 ms samples")
    ignore_steps = Int(round(NEURONIO_IGNORE_MS / sample_dt_ms))
    window_steps = Int(round(NEURONIO_WINDOW_MS / sample_dt_ms))
    time_steps > ignore_steps + window_steps ||
        error(
            "official NeuronIO source must leave multiple 500 ms " *
            "windows after the first 500 ms",
        )
    last_start = time_steps - window_steps
    return (
        training=(;
            full_time_steps=time_steps,
            sample_dt_ms=Float64(sample_dt_ms),
            ignore_time_from_start_ms=NEURONIO_IGNORE_MS,
            ignore_steps,
            input_window_size_ms=NEURONIO_WINDOW_MS,
            input_window_steps=window_steps,
            valid_window_start_indices_one_based=(
                ignore_steps + 1,
                last_start,
            ),
            sampling="uniform_with_replacement",
            source_indexing="Python start in ignore:(N-window-1)",
            reference_commit=NEURONIO_REFERENCE_COMMIT,
            loader_sha256=NEURONIO_LOADER_SHA256,
        ),
        heldout=(;
            ignore_time_at_start_ms=NEURONIO_IGNORE_MS,
            evaluated_time_indices_one_based=(
                ignore_steps + 1,
                time_steps,
            ),
            evaluated_steps_per_trial=time_steps - ignore_steps,
            evaluator_sha256=NEURONIO_EVALUATOR_SHA256,
        ),
    )
end

function FinalBridge._release_manifest(
    config::ReleaseStreamingPrepareConfig,
    source::BaseBridge._Source,
    source_counts,
    plan_data,
    frozen::SealedOfficialBridgeTwin,
    integrity_before,
    integrity_after,
    reported_metrics,
    recomputed_gate,
    twin_file_hash,
    config_sha256,
    segment_catalog_sha256,
    selected_segments,
    diagnostic_segments,
    diagnostic_times,
    train_indices,
    validation_indices,
    test_indices,
    shard_records,
    promotion_eligible,
)
    arguments = (
        config,
        source,
        source_counts,
        plan_data,
        frozen,
        integrity_before,
        integrity_after,
        reported_metrics,
        recomputed_gate,
        twin_file_hash,
        config_sha256,
        segment_catalog_sha256,
        selected_segments,
        diagnostic_segments,
        diagnostic_times,
        train_indices,
        validation_indices,
        test_indices,
        shard_records,
        promotion_eligible,
    )
    base = invoke(
        FinalBridge._release_manifest,
        Tuple{
            ReleaseStreamingPrepareConfig,
            BaseBridge._Source,
            Vararg{Any,18},
        },
        arguments...,
    )
    time_contract = _neuronio_contract(
        plan_data.time_steps,
        frozen.model.config.dt_ms,
    )
    payload = frozen.payload
    sealed_lineage = (;
        digital_twin_type=
            "PaperELMTwinOfficialV2SealedRelease." *
            "SealedOfficialELMRelease",
        digital_twin_schema=Sealed.SEALED_RELEASE_SCHEMA,
        sealed_attestation_sha256=frozen.attestation_sha256,
        source_teacher_contract_sha256=
            payload.teacher.teacher_contract_sha256,
        neuronio_training_window=time_contract.training,
        heldout_evaluation_window=time_contract.heldout,
        input_layout=merge(
            base.input_layout,
            (;
                input_static_plane=false,
                input_semantics=
                    "E:+strength*event_count, " *
                    "I:-strength*event_count",
            ),
        ),
        primary_cache_live_equality=(;
            required=true,
            construction_path=
                "written directly from live sealed ELM forward",
            all_samples=true,
            construction_bit_exact=true,
            postwrite_live_replay_required=true,
            targets=(
                "soma_voltage",
                "spike_probability",
                "spike_logit",
                "regional_nmda_current",
            ),
            detailed_auxiliary_excluded=(
                "calcium_event_sparse",
                "dendritic_voltage_sparse",
            ),
        ),
        sealed_release_metrics=payload.metrics,
        sealed_release_fixed_gate=payload.fixed_gate,
        sealed_release_outcome=payload.outcome,
    )
    return merge(
        base,
        sealed_lineage,
        (;
            hashes=merge(
                base.hashes,
                (;
                    sealed_attestation_sha256=
                        frozen.attestation_sha256,
                ),
            ),
        ),
    )
end

prepare_distillation_dataset_release(
    config::ReleaseStreamingPrepareConfig,
) = ReleaseBridge.prepare_distillation_dataset_release(config)

function main(arguments=ARGS)
    report = prepare_distillation_dataset_release(
        V6._parse_arguments(arguments),
    )
    println(JSON3.write(report))
    return report
end

end # module DistillationDatasetBridgeOfficial1278Sealed

if abspath(PROGRAM_FILE) == @__FILE__
    DistillationDatasetBridgeOfficial1278Sealed.main()
end
