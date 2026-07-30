# This file is included into `PaperELMTwinOfficialV2SealedReleaseV2`.
#
# It corrects only the held-out measurement contract:
#   1. voltage targets use the same fixed Spieler clip as training/validation;
#   2. a genuinely zero-variance NMDA target is normalized by the frozen
#      fit-split scale (whose contract floor is 1e-5).
#
# Gate thresholds and the one-shot held-out audit are intentionally unchanged.

const SEALED_V2_CONTRACT_FIX_APPLIED = true
const _SEALED_V2_CONTRACT_FIX_SOURCE = @__FILE__
const _SEALED_V2_CONTRACT_FIX_EVALUATOR_ID =
    "official-final-v2-signed1278-paper-window-reset-exact-auroc-v4-contract-fix"
const _SEALED_V2_NMDA_SCALE_FLOOR = 1.0e-5

corrected_evaluator_source_sha256() =
    _file_sha256(_SEALED_V2_CONTRACT_FIX_SOURCE)

_clip_official_soma_target(target) =
    min.(target, Twin.OFFICIAL_SOMA_CLIP_MV)

function _contract_normalized_rmse(
    moments::_PairMoments,
    frozen_nmda_scale,
)
    moments.n > 0 || return NaN
    centered =
        moments.sum_y2 -
        moments.sum_y * moments.sum_y / moments.n
    if centered > eps(Float64)
        # Preserve the original non-zero-variance definition exactly.
        return sqrt(moments.error2 / centered)
    end
    scale = max(
        abs(Float64(frozen_nmda_scale)),
        _SEALED_V2_NMDA_SCALE_FLOOR,
    )
    return _rmse(moments) / scale
end

function _evaluate(
    dataset,
    frozen,
    scratch,
    audit::_HeldoutEvaluationAudit,
)
    audit.metric_evaluations_after_selection += 1
    audit.metric_evaluations_after_selection == 1 ||
        error("held-out metrics may be evaluated only once after selection")
    burn_in_ms = 500.0
    burn_in_steps = Int(round(burn_in_ms / dataset.sample_dt_ms))
    burn_in_steps < Int(round(dataset.duration_ms / dataset.sample_dt_ms)) ||
        error("teacher duration is not longer than the 500 ms fidelity discard")
    voltage = _PairMoments()
    nmda = [_PairMoments() for _ in 1:NMDA_REGIONS]
    spool = _AUROCSpool(scratch, AUROC_RUN_RECORDS)
    heldout_set = Set(dataset.heldout_ids)
    seen = Int32[]
    total_bins = 0
    evaluated_bins = 0
    stitched_bins = 0
    window_evaluations = 0
    connectivity = _ConnectivityAccumulator()

    for record in dataset.records
        _record_overlaps_ids(record, dataset.heldout_ids) || continue
        audit.shard_opens_during_metric_evaluation += 1
        data = _load_numeric(dataset, record)
        _validate_numeric!(data)
        ids = Int32.(vec(data["sample_indices"]))
        steps = size(data["target_voltage"], 1)
        expected_steps =
            Int(round(dataset.duration_ms / dataset.sample_dt_ms))
        steps == expected_steps ||
            error("target duration differs from manifest")
        for (item, id) in enumerate(ids)
            id in heldout_set || continue
            push!(seen, id)
            _update_connectivity!(connectivity, data, item)
            total_bins += steps
            trial_stitched_bins = 0
            for (window_index, start_step) in
                enumerate(_paper_window_starts(steps))
                input, actual_steps = _paper_window_input(
                    data,
                    item,
                    start_step,
                    steps,
                )
                prediction = Twin.twin_forward(frozen, input)
                window_evaluations += 1
                local_keep_first =
                    window_index == 1 ?
                    1 :
                    PAPER_EVALUATION_OVERLAP_STEPS + 1
                local_keep_first <= actual_steps || continue
                global_keep_first =
                    start_step + local_keep_first - 1
                global_keep_last = start_step + actual_steps - 1
                retained = global_keep_last - global_keep_first + 1
                trial_stitched_bins += retained
                metric_global_first = max(
                    global_keep_first,
                    burn_in_steps + 1,
                )
                metric_global_first > global_keep_last && continue
                local_metric_first =
                    local_keep_first +
                    metric_global_first - global_keep_first
                local_range = local_metric_first:actual_steps
                target_range = metric_global_first:global_keep_last
                target_voltage = _clip_official_soma_target(
                    @view data["target_voltage"][
                        target_range,
                        item:item,
                    ],
                )
                target_spike = @view data[
                    "target_spike"
                ][target_range, item:item]
                _update!(
                    voltage,
                    @view(prediction.voltage[local_range, :]),
                    target_voltage,
                )
                _push!(
                    spool,
                    @view(prediction.spike_logit[local_range, :]),
                    target_spike,
                )
                for region in 1:NMDA_REGIONS
                    _update!(
                        nmda[region],
                        @view(prediction.nmda[
                            region,
                            local_range,
                            :,
                        ]),
                        @view(data["target_nmda"][
                            region,
                            target_range,
                            item:item,
                        ]),
                    )
                end
                evaluated_bins += length(local_range)
            end
            trial_stitched_bins == steps ||
                error("paper overlap/reset windows did not stitch exactly")
            stitched_bins += trial_stitched_bins
        end
    end
    seen == dataset.heldout_ids ||
        error("held-out trials were not evaluated in manifest order")
    auroc = _exact_auroc!(spool)
    metrics = (;
        spike_auroc=auroc,
        voltage_rmse_mv=_rmse(voltage),
        voltage_correlation=_correlation(voltage),
        paper_compatibility_calibrated_voltage_rmse_mv=
            _heldout_calibrated_rmse(voltage),
        paper_compatibility_voltage_calibration_uses_heldout_targets=true,
        paper_compatibility_calibrated_voltage_rmse_is_report_only=true,
        nmda_raw_rmse_by_region=[_rmse(value) for value in nmda],
        nmda_normalized_rmse_by_region=[
            _contract_normalized_rmse(
                nmda[region],
                frozen.normalizer.nmda_scale[region],
            ) for region in 1:NMDA_REGIONS
        ],
        nmda_correlation_by_region=[
            _correlation(value) for value in nmda
        ],
        raw_heldout_bins=total_bins,
        stitched_heldout_bins=stitched_bins,
        evaluated_bins,
        burn_in_ms,
        spike_positives=spool.positives,
        spike_negatives=
            spool.observations - spool.positives,
        auroc_external_runs=length(spool.run_paths),
        peak_auroc_records=AUROC_RUN_RECORDS,
        evaluation_window_steps=PAPER_EVALUATION_WINDOW_STEPS,
        evaluation_overlap_steps=PAPER_EVALUATION_OVERLAP_STEPS,
        evaluation_stride_steps=PAPER_EVALUATION_STRIDE_STEPS,
        evaluation_window_count=window_evaluations,
        recurrent_state_reset_each_window=true,
        continuous_state_carry_metrics_retained=false,
        heldout_metric_evaluations_after_selection=
            audit.metric_evaluations_after_selection,
        heldout_shards_opened_during_metric_evaluation=
            audit.shard_opens_during_metric_evaluation,
    )
    return (;
        metrics,
        connectivity=_finish_connectivity(connectivity),
    )
end

function _contract_fixed_evaluator(base)
    return merge(
        base,
        (;
            id=_SEALED_V2_CONTRACT_FIX_EVALUATOR_ID,
            source_sha256=corrected_evaluator_source_sha256(),
            heldout_target_voltage_clip_applied=true,
            heldout_target_voltage_clip_mv=
                Float64(Twin.OFFICIAL_SOMA_CLIP_MV),
            heldout_target_voltage_clip_expression=
                "min(raw_target_mv, OFFICIAL_SOMA_CLIP_MV)",
            zero_variance_nmda_normalization=
                "raw_rmse / max(abs(frozen.normalizer.nmda_scale[region]), 1e-5)",
            nonzero_variance_nmda_normalization_unchanged=true,
        ),
    )
end

function attest_sealed_official_elm_release(
    manifest_path::AbstractString,
    shard_directory::AbstractString,
    frozen::Twin.FrozenOfficialELMTwin;
    scratch_root=nothing,
)
    dataset = _verify_manifest_and_shards(
        manifest_path,
        shard_directory,
    )
    _validate_model!(frozen)
    protocol = _training_protocol(frozen)
    training_evidence =
        _verify_training_evidence(dataset, frozen, protocol)
    statistics = _fit_nmda_statistics(dataset)
    _verify_normalizer!(frozen, statistics)
    audit = _HeldoutEvaluationAudit()
    evaluation = _with_scratch(scratch_root) do scratch
        _evaluate(dataset, frozen, scratch, audit)
    end
    audit.metric_evaluations_after_selection == 1 ||
        error("held-out metric evaluation count differs")
    base_payload = _build_payload(
        dataset,
        frozen,
        protocol,
        training_evidence,
        statistics,
        evaluation.connectivity,
        evaluation.metrics,
    )
    payload = merge(
        base_payload,
        (; evaluator=_contract_fixed_evaluator(base_payload.evaluator)),
    )
    attestation = SealedOfficialELMReleaseAttestation(
        payload,
        canonical_sha256(payload),
    )
    return SealedOfficialELMRelease(frozen, attestation)
end
