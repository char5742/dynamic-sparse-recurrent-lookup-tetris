"""
Canonical final.v2 sealed-ELM -> lightweight 11-state-cell distillation.

Order is fail-closed:

1. load/recompute the source-bound sealed final.v2 ELM attestation,
2. verify bridge/shard hashes,
3. replay every cached primary sample/time against the live frozen ELM,
4. compute normalization only from train split, NeuronIO-accessible times,
5. train restarts and select using validation only,
6. evaluate the exact test split once,
7. apply per-region/per-branch strict gates,
8. freeze a 4 x 642 mapping with soma/axon columns structurally zero.
"""

using Dates
using JLD2
using JSON3
using Random
using Serialization
using SHA

include(joinpath(
    @__DIR__,
    "OfficialElevenStateDistillationCoreV2.jl",
))
include(joinpath(
    @__DIR__,
    "StreamingOfficialELMReleaseDatasetFinalV2.jl",
))
include(joinpath(
    @__DIR__,
    "OfficialElevenStateStreamingTrainingV3.jl",
))
include(joinpath(
    @__DIR__,
    "OfficialElevenStateReleaseGateV2.jl",
))
include(joinpath(
    @__DIR__,
    "DistilledElevenStateCellReleaseRuntimeV6.jl",
))

const DistillCore = Main.OfficialElevenStateDistillationCore
const StreamFinal =
    Main.StreamingOfficialELMReleaseDatasetFinalV2
const Stream = Main.StreamingOfficialELMReleaseDatasetV4
const Sealed = Stream.Sealed
const Training = Main.OfficialElevenStateStreamingTrainingV3
const StrictGate = Main.OfficialElevenStateReleaseGateV2
const RuntimeV6 = Main.DistilledElevenStateCellReleaseRuntimeV6
const Cell = DistillCore.Cell

const DEV1500_SOURCE_MANIFEST_SHA256 =
    "5c0efd11a7c807235bd27601769e47447114616c32f135b7687513251de9e968"
const DEV1500_TEACHER_CONTRACT_SHA256 =
    "4ee32b8070c361084e5334f1d131e99680e2c53f1ac9234b6ea4810f78d5b320"

Base.@kwdef struct SealedV2ElevenStateDistillationConfig
    bridge_dataset::String
    sealed_artifact::String
    source_manifest::String
    source_shards::String
    output::String
    metrics::String
    scratch_root::Union{Nothing,String}=nothing
    epochs::Int=45
    steps_per_epoch::Int=64
    batch::Int=32
    window::Int=500
    learning_rate::Float32=0.001f0
    free_rollout_epochs::Int=15
    restarts::Int=3
    metric_time_chunk::Int=250
    metric_auroc_bins::Int=65_536
    cache_replay_time_chunk::Int=250
    statistics_time_chunk::Int=250
    minimum_spike_auroc::Float64=0.985
    seed::UInt64=0x005b5a19
    paper_scale::Bool=false
    expected_source_manifest_sha256::String=
        DEV1500_SOURCE_MANIFEST_SHA256
    expected_teacher_contract_sha256::String=
        DEV1500_TEACHER_CONTRACT_SHA256
    overwrite_accepted::Bool=false
end

mutable struct _Statistic
    count::Int64
    sum::Float64
    sum2::Float64
end

_Statistic() = _Statistic(0, 0.0, 0.0)

function _update!(statistic::_Statistic, value)
    number = Float64(value)
    isfinite(number) ||
        error("training-domain target statistic is non-finite")
    statistic.count += 1
    statistic.sum += number
    statistic.sum2 += number * number
    return statistic
end

function _sha256_file(path)
    source = abspath(path)
    isfile(source) || error("required file is absent: $source")
    return bytes2hex(SHA.sha256(read(source)))
end

function _sha256_value(value)
    stream = IOBuffer()
    Serialization.serialize(stream, value)
    return bytes2hex(SHA.sha256(take!(stream)))
end

function _assert_sha(value, label)
    digest = lowercase(String(value))
    occursin(r"^[0-9a-f]{64}$", digest) ||
        error("$label is not a complete SHA-256")
    return digest
end

function _validate(config)
    for name in (
        :bridge_dataset,
        :sealed_artifact,
        :source_manifest,
        :source_shards,
        :output,
        :metrics,
    )
        isempty(getproperty(config, name)) &&
            throw(ArgumentError("$(String(name)) is required"))
    end
    isdir(config.bridge_dataset) ||
        error("bridge dataset directory is absent")
    isfile(config.sealed_artifact) ||
        error("sealed final.v2 artifact is absent")
    isfile(config.source_manifest) ||
        error("source teacher manifest is absent")
    isdir(config.source_shards) ||
        error("source teacher shard directory is absent")
    config.epochs >= 1 ||
        throw(ArgumentError("epochs must be positive"))
    config.steps_per_epoch >= 1 ||
        throw(ArgumentError("steps_per_epoch must be positive"))
    config.batch >= 1 ||
        throw(ArgumentError("batch must be positive"))
    config.window == 500 ||
        throw(ArgumentError("NeuronIO distillation window must be 500"))
    config.learning_rate > 0.0f0 ||
        throw(ArgumentError("learning_rate must be positive"))
    0 <= config.free_rollout_epochs <= config.epochs ||
        throw(ArgumentError("free_rollout_epochs is invalid"))
    config.restarts >= 1 ||
        throw(ArgumentError("at least one restart is required"))
    config.metric_time_chunk >= 1 ||
        throw(ArgumentError("metric_time_chunk must be positive"))
    config.metric_auroc_bins >= 256 ||
        throw(ArgumentError("metric_auroc_bins must be at least 256"))
    config.cache_replay_time_chunk >= 1 ||
        throw(ArgumentError("cache replay chunk must be positive"))
    config.statistics_time_chunk >= 1 ||
        throw(ArgumentError("statistics chunk must be positive"))
    config.minimum_spike_auroc >= 0.985 ||
        throw(ArgumentError("spike AUROC gate cannot be weaker than 0.985"))
    _assert_sha(
        config.expected_source_manifest_sha256,
        "expected source manifest",
    )
    _assert_sha(
        config.expected_teacher_contract_sha256,
        "expected teacher contract",
    )
    return config
end

function _training_domain_statistics(dataset, materialize, time_chunk)
    # The union of all legal 500-step Python-choice windows is 501:1499.
    first_time = 501
    last_time = dataset.time_steps - 1
    last_time >= first_time ||
        error("training-domain trajectory is too short")
    statistics = [_Statistic() for _ in 1:11]
    for global_index in dataset.train_indices
        for first in first_time:time_chunk:last_time
            count = min(time_chunk, last_time - first + 1)
            batch = materialize(dataset, [global_index], first, count)
            target = @view batch.target[:, :, 1]
            observed = @view batch.observed[:, :, 1]
            @inbounds for time in 1:count
                for coordinate in 1:11
                    observed[coordinate, time] || continue
                    _update!(
                        statistics[coordinate],
                        target[coordinate, time],
                    )
                end
            end
        end
    end
    dense_expected =
        length(dataset.train_indices) * (last_time - first_time + 1)
    for coordinate in 1:6
        statistics[coordinate].count == dense_expected ||
            error(
                "primary coordinate $coordinate is not dense over the " *
                "training-domain interval",
            )
    end
    target_mean = zeros(Float32, 11)
    target_scale = ones(Float32, 11)
    for coordinate in (1, 3, 4, 5, 6, 8, 9, 10, 11)
        statistic = statistics[coordinate]
        statistic.count > 0 ||
            error("training coordinate $coordinate has no observations")
        mean_value = statistic.sum / statistic.count
        variance = max(
            statistic.sum2 / statistic.count -
            mean_value * mean_value,
            0.0,
        )
        target_mean[coordinate] = Float32(mean_value)
        target_scale[coordinate] =
            max(Float32(sqrt(variance)), 1.0f-4)
    end
    report = (;
        source_split="train_only",
        time_indices_one_based=(first_time, last_time),
        excludes_python_choice_final_sample=true,
        interpolated_sparse_auxiliary_excluded=true,
        observation_count_by_coordinate=
            Tuple(statistic.count for statistic in statistics),
        target_mean=Tuple(target_mean),
        target_scale=Tuple(target_scale),
        statistics_sha256=_sha256_value((
            Tuple(statistic.count for statistic in statistics),
            target_mean,
            target_scale,
            first_time,
            last_time,
        )),
    )
    return target_mean, target_scale, report
end

@inline _finite_or(value, fallback) =
    isfinite(Float64(value)) ? Float64(value) : Float64(fallback)

function _validation_score(metrics, target_scale)
    nmda_normalized =
        Float64.(metrics.nmda_rmse_by_region) ./
        Float64.(target_scale[3:6])
    return (
        _finite_or(metrics.spike_auroc, -Inf),
        count(identity, metrics.semantic_coordinate_passed),
        _finite_or(minimum(metrics.nmda_correlation_by_region), -Inf),
        -_finite_or(maximum(nmda_normalized), Inf),
        _finite_or(metrics.soma_voltage_correlation, -Inf),
        -_finite_or(metrics.soma_voltage_rmse_mv, Inf),
        _finite_or(metrics.calcium_event_auroc, -Inf),
        -_finite_or(maximum(metrics.dendritic_voltage_rmse_mv), Inf),
    )
end

function _seed_for_restart(seed::UInt64, restart::Int)
    return seed + UInt64(restart - 1) * UInt64(0x9e3779b97f4a7c15)
end

function _atomic_jldsave(path; payload)
    destination = abspath(path)
    mkpath(dirname(destination))
    temporary = tempname(dirname(destination)) * ".jld2"
    try
        jldsave(temporary; payload)
        mv(temporary, destination; force=false)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return destination
end

function _atomic_json(path, value)
    destination = abspath(path)
    mkpath(dirname(destination))
    temporary = tempname(dirname(destination)) * ".json"
    try
        open(temporary, "w") do io
            JSON3.write(io, value)
        end
        mv(temporary, destination; force=true)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return destination
end

function _memory_report(dataset)
    tracker = dataset.tracker
    return (;
        peak_loaded_shard_bytes=tracker.peak_loaded_shard_bytes,
        peak_dense_window_bytes=tracker.peak_dense_window_bytes,
        peak_combined_bytes=tracker.peak_combined_bytes,
        windows_materialized=tracker.windows_materialized,
        samples_materialized=tracker.samples_materialized,
        dense_memory_scales_with_total_samples=false,
        full_dataset_dense_materialization=false,
    )
end

function run_sealed_v2_eleven_state_distillation(
    config::SealedV2ElevenStateDistillationConfig,
)
    _validate(config)
    source_manifest_sha256 = _sha256_file(config.source_manifest)
    source_manifest_sha256 ==
        lowercase(config.expected_source_manifest_sha256) ||
        error("source teacher manifest differs from the external pin")
    sealed_artifact_sha256 = _sha256_file(config.sealed_artifact)

    bundle = Sealed.load_verified_sealed_official_elm_release(
        config.sealed_artifact,
        config.source_manifest,
        config.source_shards;
        require_production=config.paper_scale,
        scratch_root=config.scratch_root,
    )
    attestation = bundle.attestation.payload
    attestation.teacher.manifest_sha256 ==
        source_manifest_sha256 ||
        error("sealed attestation/source manifest mismatch")
    attestation.teacher.teacher_contract_sha256 ==
        lowercase(config.expected_teacher_contract_sha256) ||
        error("sealed teacher contract differs from the external pin")
    frozen_before =
        Stream.Twin.assert_frozen_official_elm_unchanged(bundle.frozen)

    # This canonical open call performs the mandatory independent all-sample,
    # all-time live replay before any statistic or optimizer is constructed.
    opened =
        StreamFinal.open_live_verified_sealed_stream_dataset(
            config.bridge_dataset,
            bundle,
            config.source_manifest,
            config.source_shards;
            time_chunk=config.cache_replay_time_chunk,
            minimum_spike_auroc=config.minimum_spike_auroc,
            verify_shard_hashes=true,
            require_promotion_eligible=config.paper_scale,
            require_production=config.paper_scale,
            scratch_root=config.scratch_root,
        )
    dataset = opened.dataset
    Stream.stream_dataset_integrity!(dataset)
    dataset.provenance.source_manifest_sha256 ==
        source_manifest_sha256 ||
        error("bridge dataset/source manifest mismatch")
    dataset.provenance.source_teacher_contract_sha256 ==
        lowercase(config.expected_teacher_contract_sha256) ||
        error("bridge dataset/source teacher contract mismatch")

    target_mean, target_scale, statistics_report =
        _training_domain_statistics(
            dataset,
            Stream.sealed_stream_materialize_window,
            config.statistics_time_chunk,
        )
    training_config = (;
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        batch=config.batch,
        window=config.window,
        learning_rate=config.learning_rate,
        free_rollout_epochs=config.free_rollout_epochs,
    )
    neuronio_contract =
        Training.neuronio_window_contract(dataset, training_config)
    config_record = (;
        training_config...,
        restarts=config.restarts,
        metric_time_chunk=config.metric_time_chunk,
        metric_auroc_bins=config.metric_auroc_bins,
        cache_replay_time_chunk=config.cache_replay_time_chunk,
        statistics_time_chunk=config.statistics_time_chunk,
        minimum_spike_auroc=config.minimum_spike_auroc,
        seed=config.seed,
        paper_scale=config.paper_scale,
        training_input_mode="sharded_streaming_v2",
        official_training_input_mode=
            "signed_1278_sealed_v2_neuronio_windows_v1",
        dense_full_memory_path=false,
        validation_only_restart_selection=true,
        final_test_evaluations=1,
        semantic_coordinate_names=
            DistillCore.SEMANTIC_COORDINATE_NAMES,
        recurrent_mask_sha256=
            DistillCore.RECURRENT_MASK_SHA256,
        input_mask_sha256=DistillCore.INPUT_MASK_SHA256,
    )
    config_sha256 = _sha256_value(config_record)

    candidates = NamedTuple[]
    best_parameters = nothing
    best_history = nothing
    best_validation = nothing
    best_sampling = nothing
    best_score = nothing
    best_restart = 0
    for restart in 1:config.restarts
        restart_seed = _seed_for_restart(config.seed, restart)
        trained, history, sampling =
            Training.train_streaming_neuronio_windows(
                Xoshiro(restart_seed),
                dataset,
                Stream.sealed_stream_materialize_window,
                target_mean,
                target_scale,
                training_config,
            )
        validation =
            Training.evaluate_streaming_split_post_burnin(
                trained,
                dataset,
                Stream.sealed_stream_materialize_window,
                :validation,
                target_mean,
                target_scale,
                training_config;
                time_chunk=config.metric_time_chunk,
                auroc_bins=config.metric_auroc_bins,
            )
        score = _validation_score(validation, target_scale)
        push!(candidates, (;
            restart,
            seed=restart_seed,
            final_training_loss=history[end].loss,
            validation,
            validation_score=score,
            sampling,
        ))
        if best_score === nothing || score > best_score
            best_parameters = trained
            best_history = history
            best_validation = validation
            best_sampling = sampling
            best_score = score
            best_restart = restart
        end
    end
    best_parameters === nothing &&
        error("validation selection produced no candidate")

    # The only call that evaluates :test in this driver.
    test_metrics =
        Training.evaluate_streaming_split_post_burnin(
            best_parameters,
            dataset,
            Stream.sealed_stream_materialize_window,
            :test,
            target_mean,
            target_scale,
            training_config;
            time_chunk=config.metric_time_chunk,
            auroc_bins=config.metric_auroc_bins,
        )
    test_indices_sha256 =
        Training.split_indices_sha256(dataset.test_indices)
    gate = StrictGate.strict_release_gate(
        test_metrics,
        target_scale;
        minimum_spike_auroc=config.minimum_spike_auroc,
        expected_test_indices_sha256=test_indices_sha256,
    )
    metrics = (;
        validation=best_validation,
        test=test_metrics,
        validation_candidates=candidates,
    )

    parameters = DistillCore.freeze_parameters(
        best_parameters,
        dataset.segment_region,
        bundle.frozen,
        dataset.provenance,
        target_mean,
        target_scale,
        dataset.dataset_sha256,
        config_sha256,
    )
    all(iszero, @view(parameters.compartment_projection[:, 1])) ||
        error("frozen projection permits a soma contact")
    all(iszero, @view(parameters.compartment_projection[:, 641:642])) ||
        error("frozen projection permits an axon contact")
    parameter_sha256 = Cell.parameter_sha256(parameters)
    official_segment_region =
        Tuple(String.(dataset.segment_region))
    location_mapping_sha256 = _sha256_value((
        dataset.segment_catalog_sha256,
        official_segment_region,
        parameters.compartment_projection,
    ))
    train_indices_sha256 =
        Training.split_indices_sha256(dataset.train_indices)
    validation_indices_sha256 =
        Training.split_indices_sha256(dataset.validation_indices)
    split_identity = (;
        train=(;
            count=length(dataset.train_indices),
            indices_sha256=train_indices_sha256,
        ),
        validation=(;
            count=length(dataset.validation_indices),
            indices_sha256=validation_indices_sha256,
        ),
        test=(;
            count=length(dataset.test_indices),
            indices_sha256=test_indices_sha256,
        ),
    )
    evaluation_protocol = (;
        candidate_restarts=config.restarts,
        validation_evaluations=config.restarts,
        test_evaluations=1,
        selection_split="validation",
        final_gate_split="test",
        selected_restart=best_restart,
        validation_score_order=(
            "spike_auroc_desc",
            "semantic_pass_count_desc",
            "minimum_nmda_correlation_desc",
            "maximum_normalized_nmda_rmse_asc",
            "voltage_correlation_desc",
            "voltage_rmse_asc",
            "calcium_auroc_desc",
            "maximum_dendritic_rmse_asc",
        ),
    )
    primary_targets = merge(
        opened.live_replay,
        (;
            measurement_schema=
                opened.measurement.measurement_schema,
            measurement_sha256=
                opened.measurement.measurement_sha256,
        ),
    )
    source_bound_sealed_elm = (;
        sealed_execution_type=Stream.SEALED_EXECUTION_TYPE,
        sealed_release_schema=Sealed.SEALED_RELEASE_SCHEMA,
        sealed_release_artifact_kind=
            Sealed.SEALED_RELEASE_ARTIFACT_KIND,
        sealed_artifact_sha256,
        sealed_attestation_sha256=
            bundle.attestation.attestation_sha256,
        source_manifest_sha256=
            attestation.teacher.manifest_sha256,
        source_teacher_contract_sha256=
            attestation.teacher.teacher_contract_sha256,
        parameter_sha256=attestation.model.parameter_sha256,
        base_artifact_sha256=
            attestation.model.base_artifact_sha256,
        executable_mlp_activation=
            attestation.model.executable_mlp_activation,
        compatibility_profile=
            attestation.model.compatibility_profile,
        gate_passed=attestation.outcome.gate_passed,
        paper_scale=attestation.outcome.paper_scale,
        development_scale=
            attestation.outcome.development_scale,
        promotable_production=
            attestation.outcome.promotable_production,
    )
    detailed_auxiliary = (;
        primary_teacher=false,
        sparse_observation_grid=true,
        training_interpolation_for_fit=true,
        release_gate_observed_times_only=true,
        targets=(
            "calcium_event_sparse",
            "dendritic_voltage_sparse",
        ),
        teacher_hash=dataset.provenance.detailed_teacher_hash,
    )
    semantic_coordinate_gate = (;
        passed=gate.semantic_state_passed,
        coordinate_names=
            Tuple(Symbol.(DistillCore.SEMANTIC_COORDINATE_NAMES)),
        rmse=test_metrics.semantic_coordinate_rmse,
        correlation=test_metrics.semantic_coordinate_correlation,
        per_coordinate_passed=
            test_metrics.semantic_coordinate_passed,
        maximum_rmse=0.70,
        minimum_correlation=0.70,
    )
    structured_transition_contract = (;
        recurrent_mask_sha256=
            DistillCore.RECURRENT_MASK_SHA256,
        input_mask_sha256=DistillCore.INPUT_MASK_SHA256,
        structured_readout=true,
        dense_rotational_hidden_basis=false,
        coordinate_wise_semantic_supervision=true,
        semantic_state_scale="normalized_unit_interval",
        location_index_type="UInt16",
    )
    frozen_after =
        Stream.Twin.assert_frozen_official_elm_unchanged(bundle.frozen)
    frozen_before == frozen_after ||
        error("sealed frozen ELM changed during distillation")
    Stream.stream_dataset_integrity!(dataset)

    payload = (;
        schema=Cell.DISTILLED_ARTIFACT_SCHEMA,
        parameters,
        parameter_sha256,
        frozen_internal=true,
        ablation_mode=:full,
        provisional=!gate.passed,
        training_input_mode="sharded_streaming_v2",
        official_training_input_mode=
            "signed_1278_sealed_v2_neuronio_windows_v1",
        prepared_dataset_schema=
            Stream.BaseStream.RELEASE_DATASET_SCHEMA,
        dense_full_memory_path=false,
        paper_scale=config.paper_scale,
        development_scale=!config.paper_scale,
        promotable_production=
            config.paper_scale && gate.passed,
        official_elm_input_dim=Stream.OFFICIAL_ELM_INPUT_DIM,
        official_segment_count=642,
        official_segment_region,
        location_index_type="UInt16",
        semantic_state_scale="normalized_unit_interval",
        source_segment_catalog_sha256=
            dataset.segment_catalog_sha256,
        location_mapping_sha256,
        semantic_coordinate_gate,
        structured_transition_contract,
        split_identity,
        evaluation_protocol,
        neuronio_window_contract=neuronio_contract,
        training_domain_statistics=statistics_report,
        sealed_execution_type=Stream.SEALED_EXECUTION_TYPE,
        sealed_release_schema=Sealed.SEALED_RELEASE_SCHEMA,
        sealed_release_artifact_kind=
            Sealed.SEALED_RELEASE_ARTIFACT_KIND,
        sealed_attestation_sha256=
            bundle.attestation.attestation_sha256,
        source_bound_sealed_elm,
        legacy_3852_twin_fallback=false,
        primary_frozen_twin_targets=primary_targets,
        detailed_model_auxiliary_state_targets=
            detailed_auxiliary,
        teacher_hash=dataset.provenance.detailed_teacher_hash,
        detailed_kernel_hash=
            dataset.provenance.detailed_kernel_hash,
        cell_mechanism_sha256=
            dataset.provenance.detailed_kernel_hash,
        morphology_hash=dataset.provenance.morphology_hash,
        frozen_twin_parameter_hash=
            bundle.frozen.parameter_sha256,
        frozen_twin_artifact_hash=
            bundle.frozen.artifact_sha256,
        digital_twin_sha256=bundle.frozen.artifact_sha256,
        distillation_dataset_hash=dataset.dataset_sha256,
        distillation_config_hash=config_sha256,
        prepared_dataset_manifest_sha256=
            dataset.manifest_sha256,
        prepared_dataset_file_sha256=
            dataset.dataset_sha256,
        source_manifest_sha256,
        source_teacher_contract_sha256=
            attestation.teacher.teacher_contract_sha256,
        metrics,
        gate,
        config=config_record,
        selected_restart=best_restart,
        selected_history=best_history,
        selected_sampling_report=best_sampling,
        memory=_memory_report(dataset),
        frozen_twin_integrity_before=frozen_before,
        frozen_twin_integrity_after=frozen_after,
        created_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
    )
    timestamp =
        Dates.format(now(UTC), dateformat"yyyymmddTHHMMSS")
    destination = if gate.passed
        if isfile(config.output)
            config.overwrite_accepted ||
                error(
                    "accepted artifact already exists; set " *
                    "overwrite_accepted=true explicitly",
                )
            backup = config.output * ".previous." * timestamp
            mv(config.output, backup; force=false)
        end
        config.output
    else
        config.output * ".failed." * timestamp * ".jld2"
    end
    artifact_path = _atomic_jldsave(destination; payload)
    artifact_sha256 = Cell.artifact_sha256(artifact_path)
    if gate.passed
        runtime = RuntimeV6.load_release_runtime(
            artifact_path;
            expected_artifact_sha256=artifact_sha256,
            expected_sealed_attestation_sha256=
                bundle.attestation.attestation_sha256,
        )
        RuntimeV6.preflight_integrity!(runtime)
        RuntimeV6.checkpoint_integrity!(runtime)
        RuntimeV6.end_run_integrity!(runtime)
    end
    report = (;
        accepted=gate.passed,
        artifact_path,
        artifact_sha256,
        parameter_sha256,
        selected_restart=best_restart,
        paper_scale=config.paper_scale,
        development_scale=!config.paper_scale,
        non_promotable_development=
            !config.paper_scale,
        source_bound_sealed_elm,
        split_identity,
        evaluation_protocol,
        neuronio_window_contract=neuronio_contract,
        training_domain_statistics=statistics_report,
        primary_frozen_twin_targets=primary_targets,
        detailed_model_auxiliary_state_targets=
            detailed_auxiliary,
        metrics,
        gate,
        memory=_memory_report(dataset),
    )
    _atomic_json(config.metrics, report)
    gate.passed || error(
        "strict sealed-v2 distillation gates failed; candidate was saved " *
        "separately at $artifact_path and accepted output was untouched",
    )
    return report
end

function _parse_arguments(arguments)
    options = Dict{String,String}(
        "bridge-dataset" => get(
            ENV,
            "HD_TWINPROP_DISTILL_DATASET_DIR",
            "",
        ),
        "sealed-artifact" => get(
            ENV,
            "HD_TWINPROP_SEALED_V2_PATH",
            "",
        ),
        "source-manifest" => get(
            ENV,
            "HD_TWINPROP_TEACHER_MANIFEST",
            "",
        ),
        "source-shards" => get(
            ENV,
            "HD_TWINPROP_TEACHER_SHARDS",
            "",
        ),
        "output" => get(
            ENV,
            "HD_TWINPROP_DISTILLED_PATH",
            joinpath(
                @__DIR__,
                "artifacts",
                "distilled_eleven_state_sealed_v2.jld2",
            ),
        ),
        "metrics" => get(
            ENV,
            "HD_TWINPROP_DISTILL_METRICS",
            joinpath(
                @__DIR__,
                "artifacts",
                "distilled_eleven_state_sealed_v2.json",
            ),
        ),
        "scratch-root" => "",
        "epochs" => "45",
        "steps-per-epoch" => "64",
        "batch" => "32",
        "window" => "500",
        "learning-rate" => "0.001",
        "free-rollout-epochs" => "15",
        "restarts" => "3",
        "metric-time-chunk" => "250",
        "metric-auroc-bins" => "65536",
        "cache-replay-time-chunk" => "250",
        "statistics-time-chunk" => "250",
        "minimum-spike-auroc" => "0.985",
        "seed" => string(UInt64(0x005b5a19)),
        "paper-scale" => "false",
        "expected-source-manifest-sha256" =>
            DEV1500_SOURCE_MANIFEST_SHA256,
        "expected-teacher-contract-sha256" =>
            DEV1500_TEACHER_CONTRACT_SHA256,
        "overwrite-accepted" => "false",
    )
    index = 1
    while index <= length(arguments)
        token = arguments[index]
        startswith(token, "--") ||
            error("unexpected positional argument: $token")
        key = token[3:end]
        haskey(options, key) || error("unknown option --$key")
        index += 1
        index <= length(arguments) ||
            error("missing value for --$key")
        options[key] = arguments[index]
        index += 1
    end
    scratch = options["scratch-root"]
    return SealedV2ElevenStateDistillationConfig(
        bridge_dataset=options["bridge-dataset"],
        sealed_artifact=options["sealed-artifact"],
        source_manifest=options["source-manifest"],
        source_shards=options["source-shards"],
        output=options["output"],
        metrics=options["metrics"],
        scratch_root=isempty(scratch) ? nothing : scratch,
        epochs=parse(Int, options["epochs"]),
        steps_per_epoch=parse(Int, options["steps-per-epoch"]),
        batch=parse(Int, options["batch"]),
        window=parse(Int, options["window"]),
        learning_rate=parse(Float32, options["learning-rate"]),
        free_rollout_epochs=
            parse(Int, options["free-rollout-epochs"]),
        restarts=parse(Int, options["restarts"]),
        metric_time_chunk=
            parse(Int, options["metric-time-chunk"]),
        metric_auroc_bins=
            parse(Int, options["metric-auroc-bins"]),
        cache_replay_time_chunk=
            parse(Int, options["cache-replay-time-chunk"]),
        statistics_time_chunk=
            parse(Int, options["statistics-time-chunk"]),
        minimum_spike_auroc=
            parse(Float64, options["minimum-spike-auroc"]),
        seed=parse(UInt64, options["seed"]),
        paper_scale=parse(Bool, options["paper-scale"]),
        expected_source_manifest_sha256=
            options["expected-source-manifest-sha256"],
        expected_teacher_contract_sha256=
            options["expected-teacher-contract-sha256"],
        overwrite_accepted=
            parse(Bool, options["overwrite-accepted"]),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_sealed_v2_eleven_state_distillation(
        _parse_arguments(ARGS),
    )
end
