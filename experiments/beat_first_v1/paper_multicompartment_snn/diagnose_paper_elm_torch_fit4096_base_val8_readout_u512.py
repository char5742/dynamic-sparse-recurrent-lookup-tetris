"""Fit-only readout calibration and base-val8 channel diagnostics.

This is an add-only wrapper around the corrected fit4096/base-val8 evaluator.
It never addresses held-out IDs.  All calibration parameters are estimated
from the composite fit4096 inventory before the fixed base validation split is
opened.  Every validation score is derived from one shared model forward.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

import evaluate_paper_elm_torch_fit4096_base_val8_m1000 as evaluator


entry = evaluator.entry
core = evaluator.core
runner = evaluator.runner

_STARTS_JULIA = (1, 351, 701, 1_051)
_VOLTAGE_LOGIT_SCALE_MV = 2.0
_FIXED_BLEND_WEIGHTS = (0.25, 0.50, 0.75)


def _sigmoid(values):
    values = np.asarray(values, np.float64)
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _probability_rmse(probability, target) -> float:
    probability = np.asarray(probability, np.float64)
    target = np.asarray(target, np.float64)
    return math.sqrt(float(np.mean(np.square(probability - target))))


def _selected_window(
    raw,
    target_voltage,
    target_spike,
    start_julia,
    window_index,
):
    actual_steps = min(
        core.TRAIN_WINDOW,
        1_500 - start_julia + 1,
    )
    local_keep_first = 0 if window_index == 0 else 150
    global_keep_first = start_julia - 1 + local_keep_first
    metric_global_first = max(global_keep_first, 500)
    global_keep_last = start_julia - 1 + actual_steps - 1
    if metric_global_first > global_keep_last:
        return None
    local_metric_first = (
        local_keep_first
        + metric_global_first
        - global_keep_first
    )
    selected = raw[local_metric_first:actual_steps]
    target_slice = slice(
        metric_global_first,
        global_keep_last + 1,
    )
    predicted_mv = (
        selected[:, 1].astype(np.float64, copy=False)
        / core.SOMA_TRAIN_SCALE
        + core.SOMA_BIAS_MV
    )
    clipped_target_mv = np.minimum(
        target_voltage[target_slice],
        core.SOMA_CLIP_MV,
    ).astype(np.float64)
    return (
        selected[:, 0].astype(np.float64, copy=False),
        predicted_mv,
        target_spike[target_slice].astype(np.float64),
        clipped_target_mv,
    )


def _collect_validation_once(model, dataset):
    logits = []
    predicted_mv = []
    spikes = []
    target_mv = []
    with torch.no_grad():
        for trial_id in entry._BASE_VALIDATION_IDS:
            data, item = dataset.trial_arrays(int(trial_id))
            trial_voltage = np.asarray(
                data["target_voltage"][:, item],
                np.float32,
            )
            trial_spike = np.asarray(
                data["target_spike"][:, item],
                np.float32,
            )
            for window_index, start_julia in enumerate(_STARTS_JULIA):
                input_array, _, _, _ = dataset.window(
                    int(trial_id),
                    start_julia,
                    core.TRAIN_WINDOW,
                    pad_to=core.TRAIN_WINDOW,
                )
                raw = (
                    model(
                        torch.from_numpy(
                            input_array.T[:, None, :].copy()
                        )
                    )[:, 0, :]
                    .cpu()
                    .numpy()
                )
                selected = _selected_window(
                    raw,
                    trial_voltage,
                    trial_spike,
                    start_julia,
                    window_index,
                )
                if selected is None:
                    continue
                logit, voltage, spike, voltage_target = selected
                logits.append(logit)
                predicted_mv.append(voltage)
                spikes.append(spike)
                target_mv.append(voltage_target)
    result = tuple(
        np.concatenate(parts)
        for parts in (logits, predicted_mv, spikes, target_mv)
    )
    expected = len(entry._BASE_VALIDATION_IDS) * 1_000
    if any(values.size != expected for values in result):
        raise AssertionError("base-val8 observation count differs")
    return result


def _run_fit_batch(model, dataset, pending):
    inputs = []
    targets = []
    for handle, window_index, start_julia in pending:
        data, item = dataset.trial_arrays(int(handle))
        target_voltage = np.asarray(
            data["target_voltage"][:, item],
            np.float32,
        )
        target_spike = np.asarray(
            data["target_spike"][:, item],
            np.float32,
        )
        input_array, _, _, _ = dataset.window(
            int(handle),
            start_julia,
            core.TRAIN_WINDOW,
            pad_to=core.TRAIN_WINDOW,
        )
        inputs.append(input_array.T)
        targets.append((
            target_voltage,
            target_spike,
            start_julia,
            window_index,
        ))
    batch = np.stack(inputs, axis=1).astype(np.float32, copy=False)
    with torch.no_grad():
        raw_batch = model(torch.from_numpy(batch)).cpu().numpy()
    selected = []
    for slot, (
        target_voltage,
        target_spike,
        start_julia,
        window_index,
    ) in enumerate(targets):
        values = _selected_window(
            raw_batch[:, slot, :],
            target_voltage,
            target_spike,
            start_julia,
            window_index,
        )
        if values is not None:
            selected.append(values)
    return selected


def _collect_fit(model, dataset, batch_size):
    logits = []
    predicted_mv = []
    spikes = []
    target_mv = []
    pending = []
    completed = 0
    handles = tuple(int(value) for value in entry._FIT_HANDLES)
    for handle in handles:
        # The first window contributes no post-burn-in observations and the
        # model state is reset per evaluator window, so it is safely omitted.
        for window_index, start_julia in enumerate(_STARTS_JULIA):
            if window_index == 0:
                continue
            pending.append((handle, window_index, start_julia))
            if len(pending) < batch_size:
                continue
            for values in _run_fit_batch(model, dataset, pending):
                logit, voltage, spike, voltage_target = values
                logits.append(logit)
                predicted_mv.append(voltage)
                spikes.append(spike)
                target_mv.append(voltage_target)
            completed += len(pending)
            pending.clear()
            if completed % (batch_size * 32) == 0:
                print(
                    f"fit calibration windows {completed}/"
                    f"{len(handles) * 3}",
                    file=sys.stderr,
                    flush=True,
                )
    if pending:
        for values in _run_fit_batch(model, dataset, pending):
            logit, voltage, spike, voltage_target = values
            logits.append(logit)
            predicted_mv.append(voltage)
            spikes.append(spike)
            target_mv.append(voltage_target)
        completed += len(pending)
    result = tuple(
        np.concatenate(parts)
        for parts in (logits, predicted_mv, spikes, target_mv)
    )
    expected = len(handles) * 1_000
    if any(values.size != expected for values in result):
        raise AssertionError("fit4096 calibration observation count differs")
    return result


def _fit_affine(source, target):
    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    source_mean = float(source.mean())
    target_mean = float(target.mean())
    centered = source - source_mean
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        raise ValueError("affine calibration source is constant")
    slope = float(
        np.dot(centered, target - target_mean) / denominator
    )
    intercept = target_mean - slope * source_mean
    return slope, intercept


def _logistic_loss(score, target):
    return float(
        np.mean(
            np.maximum(score, 0.0)
            - score * target
            + np.log1p(np.exp(-np.abs(score)))
        )
    )


def _fit_logistic_2p(feature, target):
    feature = np.asarray(feature, np.float64)
    target = np.asarray(target, np.float64)
    theta = np.array(
        [
            1.0,
            math.log(
                max(float(target.mean()), 1.0e-8)
                / max(1.0 - float(target.mean()), 1.0e-8)
            ),
        ],
        np.float64,
    )
    for _ in range(40):
        score = theta[0] * feature + theta[1]
        probability = _sigmoid(score)
        residual = probability - target
        weight = probability * (1.0 - probability)
        gradient = np.array(
            [
                np.mean(residual * feature),
                np.mean(residual),
            ],
            np.float64,
        )
        hessian = np.array(
            [
                [
                    np.mean(weight * feature * feature),
                    np.mean(weight * feature),
                ],
                [
                    np.mean(weight * feature),
                    np.mean(weight),
                ],
            ],
            np.float64,
        )
        hessian.flat[::3] += 1.0e-10
        step = np.linalg.solve(hessian, gradient)
        old_loss = _logistic_loss(score, target)
        scale = 1.0
        accepted = False
        for _ in range(24):
            candidate = theta - scale * step
            if candidate[0] <= 0.0:
                scale *= 0.5
                continue
            candidate_loss = _logistic_loss(
                candidate[0] * feature + candidate[1],
                target,
            )
            if candidate_loss <= old_loss:
                theta = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted or float(np.max(np.abs(scale * step))) < 1.0e-9:
            break
    return float(theta[0]), float(theta[1])


def _fit_monotone_blend_2p(logit, voltage_score, target):
    logit = np.asarray(logit, np.float64)
    voltage_score = np.asarray(voltage_score, np.float64)
    target = np.asarray(target, np.float64)
    difference = logit - voltage_score
    theta = np.array(
        [
            0.5,
            math.log(
                max(float(target.mean()), 1.0e-8)
                / max(1.0 - float(target.mean()), 1.0e-8)
            ),
        ],
        np.float64,
    )
    for _ in range(40):
        score = voltage_score + theta[0] * difference + theta[1]
        probability = _sigmoid(score)
        residual = probability - target
        weight = probability * (1.0 - probability)
        gradient = np.array(
            [
                np.mean(residual * difference),
                np.mean(residual),
            ],
            np.float64,
        )
        hessian = np.array(
            [
                [
                    np.mean(weight * difference * difference),
                    np.mean(weight * difference),
                ],
                [
                    np.mean(weight * difference),
                    np.mean(weight),
                ],
            ],
            np.float64,
        )
        hessian.flat[::3] += 1.0e-10
        step = np.linalg.solve(hessian, gradient)
        old_loss = _logistic_loss(score, target)
        scale = 1.0
        accepted = False
        for _ in range(24):
            candidate = theta - scale * step
            candidate[0] = np.clip(candidate[0], 0.0, 1.0)
            candidate_score = (
                voltage_score
                + candidate[0] * difference
                + candidate[1]
            )
            if _logistic_loss(candidate_score, target) <= old_loss:
                theta = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted or float(np.max(np.abs(scale * step))) < 1.0e-9:
            break
    return float(theta[0]), float(theta[1])


def _score_metrics(score, target, probability=None):
    score = np.asarray(score, np.float64)
    target = np.asarray(target, np.float64)
    if probability is None:
        probability = _sigmoid(score)
    return {
        "exact_auroc": evaluator._exact_auc(score, target),
        "probability_rmse": _probability_rmse(
            probability,
            target,
        ),
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--augmentation-dataset",
        type=Path,
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    # The imported fit4096 bridge adapter requires the historical warmstart
    # argument while reconstructing its immutable composite metadata.
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--fit-batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    options = _parse_args()
    if options.threads != entry.EXPECTED_THREADS:
        raise ValueError("fit4096 diagnostic requires threads=6")
    if options.fit_batch_size < 1:
        raise ValueError("fit batch size must be positive")
    torch.set_num_threads(options.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    bridge, metadata = core._load_bridge(options.bridge.resolve())
    model = runner._build(bridge, core.FULL_MEMORY, "torchscript")
    payload = torch.load(
        options.checkpoint.resolve(),
        map_location="cpu",
    )
    if payload.get("schema") != "paper_elm_torch_cpu_variable.v1":
        raise ValueError("input checkpoint schema differs")
    if int(payload["memory"]) != core.FULL_MEMORY:
        raise ValueError("input checkpoint is not M1000")
    if int(payload["update_index"]) != 512:
        raise ValueError("diagnostic requires cumulative update 512")
    if bool(payload.get("heldout_opened", True)):
        raise ValueError("checkpoint does not assert heldout exclusion")
    source = payload.get("source_bridge_metadata", {})
    if source.get("manifest_sha256") != metadata["manifest_sha256"]:
        raise ValueError("checkpoint/composite contract differs")
    model.load_state_dict(payload["model"])

    # Calibration is completed and frozen before validation is instantiated.
    fit_dataset = entry._CompositeDataset(
        options.dataset.resolve(),
        entry._COMPOSITE_SHA256,
    )
    fit_logit, fit_mv, fit_spike, fit_target_mv = _collect_fit(
        model,
        fit_dataset,
        options.fit_batch_size,
    )
    fit_dataset.assert_no_heldout_opened()

    voltage_affine = _fit_affine(fit_mv, fit_target_mv)
    logit_platt = _fit_logistic_2p(fit_logit, fit_spike)
    fit_voltage_score = (
        fit_mv - core.SOMA_CLIP_MV
    ) / _VOLTAGE_LOGIT_SCALE_MV
    voltage_logistic = _fit_logistic_2p(
        fit_voltage_score,
        fit_spike,
    )
    monotone_blend = _fit_monotone_blend_2p(
        fit_logit,
        fit_voltage_score,
        fit_spike,
    )
    del fit_logit, fit_mv, fit_spike, fit_target_mv

    validation = entry._BaseDataset(
        options.dataset.resolve(),
        entry._BASE_MANIFEST_SHA256,
    )
    val_logit, val_mv, val_spike, val_target_mv = (
        _collect_validation_once(model, validation)
    )
    validation.assert_no_heldout_opened()
    val_voltage_score = (
        val_mv - core.SOMA_CLIP_MV
    ) / _VOLTAGE_LOGIT_SCALE_MV
    logit_probability = _sigmoid(val_logit)
    voltage_probability = _sigmoid(val_voltage_score)

    fixed_blends = {}
    for weight in _FIXED_BLEND_WEIGHTS:
        probability = (
            weight * logit_probability
            + (1.0 - weight) * voltage_probability
        )
        fixed_blends[str(weight)] = _score_metrics(
            probability,
            val_spike,
            probability,
        )

    calibrated_voltage = (
        voltage_affine[0] * val_mv + voltage_affine[1]
    )
    platt_score = (
        logit_platt[0] * val_logit + logit_platt[1]
    )
    voltage_calibrated_score = (
        voltage_logistic[0] * val_voltage_score
        + voltage_logistic[1]
    )
    blend_score = (
        val_voltage_score
        + monotone_blend[0]
        * (val_logit - val_voltage_score)
        + monotone_blend[1]
    )

    result = {
        "schema": (
            "paper_elm_torch_fit4096_u512_"
            "base_val8_readout_diagnostic.v1"
        ),
        "checkpoint": str(options.checkpoint.resolve()),
        "update_index": 512,
        "memory": core.FULL_MEMORY,
        "hidden": core.FULL_HIDDEN,
        "fit_calibration": {
            "source": "composite-fit4096-only",
            "trials": len(entry._FIT_HANDLES),
            "observations": len(entry._FIT_HANDLES) * 1_000,
            "validation_used_for_fit": False,
            "voltage_affine": {
                "slope": voltage_affine[0],
                "intercept_mv": voltage_affine[1],
            },
            "logit_platt": {
                "slope": logit_platt[0],
                "intercept": logit_platt[1],
            },
            "voltage_logistic": {
                "slope": voltage_logistic[0],
                "intercept": voltage_logistic[1],
                "input": "(predicted_mv + 55) / 2",
            },
            "monotone_blend_logistic": {
                "logit_weight": monotone_blend[0],
                "voltage_weight": 1.0 - monotone_blend[0],
                "intercept": monotone_blend[1],
            },
        },
        "validation": {
            "source": "base-only",
            "trials": len(entry._BASE_VALIDATION_IDS),
            "observations": int(val_spike.size),
            "spike_positive_bins": int((val_spike > 0.5).sum()),
            "spike_negative_bins": int((val_spike <= 0.5).sum()),
            "same_forward_for_all_scores": True,
            "heldout_opened": False,
            "raw_spike_logit": _score_metrics(
                val_logit,
                val_spike,
            ),
            "raw_predicted_clipped_voltage": {
                **_score_metrics(
                    val_voltage_score,
                    val_spike,
                ),
                "physical_rmse_mv": math.sqrt(
                    float(np.mean(np.square(val_mv - val_target_mv)))
                ),
            },
            "fixed_monotone_probability_blends": fixed_blends,
            "fit_only_calibration": {
                "voltage_affine": {
                    "spike_exact_auroc": evaluator._exact_auc(
                        calibrated_voltage,
                        val_spike,
                    ),
                    "physical_rmse_mv": math.sqrt(
                        float(
                            np.mean(
                                np.square(
                                    calibrated_voltage - val_target_mv
                                )
                            )
                        )
                    ),
                },
                "logit_platt": _score_metrics(
                    platt_score,
                    val_spike,
                ),
                "voltage_logistic": _score_metrics(
                    voltage_calibrated_score,
                    val_spike,
                ),
                "monotone_blend_logistic": _score_metrics(
                    blend_score,
                    val_spike,
                ),
            },
        },
        "base_manifest_sha256": entry._BASE_MANIFEST_SHA256,
        "augmentation_manifest_sha256": (
            entry._AUGMENTATION_MANIFEST_SHA256
        ),
        "heldout_opened": False,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
