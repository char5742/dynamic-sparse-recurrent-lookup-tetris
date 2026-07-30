"""
Canonical release entry point.

It reuses the standalone, directly parsed training implementation and adds
the final runtime contract fields: UInt16 official location IDs, the complete
642-segment region map, normalized semantic-state scale, and the matching
location-map digest consumed by `DistilledElevenStateCellReleaseRuntimeV2`.
No source rewriting or `include_string` is used.
"""

include(joinpath(
    @__DIR__,
    "distill_eleven_state_cell_release.jl",
))
if !isdefined(Main, :DistilledElevenStateCellReleaseRuntimeV2)
    include(joinpath(
        @__DIR__,
        "DistilledElevenStateCellReleaseRuntimeV2.jl",
    ))
end
const ReleaseRuntimeV2 =
    Main.DistilledElevenStateCellReleaseRuntimeV2

function run_release_distillation_v2(
    config::ReleaseDistillationConfig,
)
    _release_validate_config(config)
    frozen_twin = ReleaseTwin.load_frozen_twin(config.frozen_twin)
    twin_before = ReleaseTwin.assert_frozen_unchanged(frozen_twin)
    raw_twin_file_sha256 =
        _release_file_sha256(config.frozen_twin)
    dataset = _release_load_dataset(config.dataset, frozen_twin)
    raw_twin_file_sha256 ==
        dataset.provenance.raw_twin_file_sha256 ||
        error("raw frozen-twin file SHA-256 mismatch")
    live_twin =
        _release_twin_inference(frozen_twin, dataset.input)
    cache_check = _release_cache_check(dataset, live_twin)
    target = _release_targets(dataset, live_twin)
    target_mean, target_scale =
        _release_target_statistics(target, dataset.train_indices)
    config_record = (;
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        batch=config.batch,
        window=min(config.window, size(dataset.input, 2)),
        learning_rate=config.learning_rate,
        free_rollout_epochs=config.free_rollout_epochs,
        minimum_spike_auroc=config.minimum_spike_auroc,
        seed=config.seed,
        official_segment_count=RELEASE_SEGMENTS,
        location_index_type="UInt16",
        semantic_state_scale="normalized_unit_interval",
        semantic_coordinate_names=RELEASE_COORDINATE_NAMES,
        recurrent_mask_sha256=RELEASE_RECURRENT_MASK_SHA256,
        input_mask_sha256=RELEASE_INPUT_MASK_SHA256,
        structured_transition=true,
        structured_readout=true,
        coordinate_wise_semantic_supervision=true,
        soma_spike_is_sole_external_event=true,
        mixed_supervision=(
            frozen_twin=(:soma_voltage, :soma_spike, :nmda_current),
            official_neuron=(:calcium_event, :dendritic_voltage),
        ),
    )
    config_hash = _release_value_sha256(config_record)
    trained, history = _release_train(
        Xoshiro(config.seed),
        dataset,
        target,
        target_mean,
        target_scale,
        frozen_twin.model.config.segments,
        config,
    )
    metrics = (;
        train=_release_metrics(
            trained,
            dataset,
            target,
            dataset.train_indices,
            target_mean,
            target_scale,
            frozen_twin.model.config.segments,
        ),
        validation=_release_metrics(
            trained,
            dataset,
            target,
            dataset.validation_indices,
            target_mean,
            target_scale,
            frozen_twin.model.config.segments,
        ),
        test=_release_metrics(
            trained,
            dataset,
            target,
            dataset.test_indices,
            target_mean,
            target_scale,
            frozen_twin.model.config.segments,
        ),
    )
    gate = _release_gate(
        metrics.test,
        config.minimum_spike_auroc,
    )
    parameters = _release_frozen_parameters(
        trained,
        dataset,
        frozen_twin,
        target_mean,
        target_scale,
        config_hash,
    )
    parameter_hash = ReleaseCell.parameter_sha256(parameters)
    official_segment_region = Tuple(String.(dataset.segment_region))
    location_mapping_sha256 = _release_value_sha256((
        dataset.provenance.segment_catalog_sha256,
        official_segment_region,
        parameters.compartment_projection,
    ))
    twin_after =
        ReleaseTwin.assert_frozen_unchanged(frozen_twin)
    twin_before == twin_after ||
        error("frozen digital twin changed during distillation")
    semantic_coordinate_gate = (;
        passed=all(metrics.test.semantic_coordinate_passed),
        coordinate_names=RELEASE_COORDINATE_NAMES,
        rmse=metrics.test.semantic_coordinate_rmse,
        correlation=metrics.test.semantic_coordinate_correlation,
        per_coordinate_passed=
            metrics.test.semantic_coordinate_passed,
        maximum_rmse=RELEASE_MAXIMUM_COORDINATE_RMSE,
        minimum_correlation=
            RELEASE_MINIMUM_COORDINATE_CORRELATION,
    )
    structured_transition_contract = (;
        recurrent_mask_sha256=RELEASE_RECURRENT_MASK_SHA256,
        input_mask_sha256=RELEASE_INPUT_MASK_SHA256,
        structured_readout=true,
        dense_rotational_hidden_basis=false,
        coordinate_wise_semantic_supervision=true,
        semantic_state_scale="normalized_unit_interval",
        location_index_type="UInt16",
    )
    payload = (;
        schema=ReleaseCell.DISTILLED_ARTIFACT_SCHEMA,
        parameters,
        parameter_sha256=parameter_hash,
        frozen_internal=true,
        ablation_mode=:full,
        official_segment_count=RELEASE_SEGMENTS,
        official_segment_region,
        location_index_type="UInt16",
        semantic_state_scale="normalized_unit_interval",
        location_mapping_sha256,
        semantic_coordinate_gate,
        structured_transition_contract,
        teacher_hash=dataset.provenance.detailed_teacher_hash,
        cell_mechanism_sha256=
            dataset.provenance.detailed_kernel_hash,
        detailed_kernel_hash=
            dataset.provenance.detailed_kernel_hash,
        morphology_hash=dataset.provenance.morphology_hash,
        digital_twin_sha256=frozen_twin.artifact_sha256,
        digital_twin_hash=frozen_twin.artifact_sha256,
        frozen_twin_parameter_hash=
            frozen_twin.parameter_sha256,
        frozen_twin_artifact_hash=
            frozen_twin.artifact_sha256,
        raw_twin_file_sha256,
        official_modeldb_source_hash=
            dataset.provenance.official_modeldb_source_hash,
        official_teacher_file_hash=
            dataset.provenance.official_teacher_file_hash,
        official_source_dataset_hash=
            dataset.provenance.official_source_dataset_hash,
        prepared_dataset_file_sha256=
            dataset.prepared_dataset_file_sha256,
        distillation_dataset_hash=
            dataset.prepared_dataset_file_sha256,
        distillation_config_hash=config_hash,
        source_segment_catalog_sha256=
            dataset.provenance.segment_catalog_sha256,
        mixed_supervision=(;
            twin_targets=(:soma_voltage, :soma_spike, :nmda_current),
            official_neuron_targets=
                (:calcium_event, :dendritic_voltage),
            live_frozen_twin_inference=true,
            cache_check,
        ),
        frozen_twin_integrity_before=twin_before,
        frozen_twin_integrity_after=twin_after,
        metrics,
        gate,
        config=config_record,
        history,
        created_at_utc=Dates.format(
            now(UTC),
            dateformat"yyyy-mm-ddTHH:MM:SS.sssZ",
        ),
    )
    timestamp = Dates.format(now(UTC), dateformat"yyyymmddTHHMMSS")
    destination = gate.passed ?
        config.output :
        config.output * ".failed." * timestamp * ".jld2"
    _release_atomic_jldsave(destination; payload)
    if gate.passed
        runtime = ReleaseRuntimeV2.load_release_runtime(destination)
        ReleaseRuntimeV2.preflight_integrity!(runtime)
        ReleaseRuntimeV2.checkpoint_integrity!(runtime)
        ReleaseRuntimeV2.end_run_integrity!(runtime)
    end
    report = (;
        accepted=gate.passed,
        artifact_path=destination,
        artifact_sha256=
            ReleaseCell.artifact_sha256(destination),
        parameter_sha256=parameter_hash,
        location_mapping_sha256,
        semantic_coordinate_gate,
        structured_transition_contract,
        metrics,
        gate,
        cache_check,
        history,
    )
    _release_atomic_json(config.metrics, report)
    @info "release v2 distillation finished" report.accepted report.artifact_path report.gate report.semantic_coordinate_gate
    gate.passed || error(
        "release gates failed; candidate saved separately at " *
        destination * "; accepted artifact path was not modified",
    )
    return report
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_release_distillation_v2(
        _release_parse_arguments(ARGS),
    )
end
