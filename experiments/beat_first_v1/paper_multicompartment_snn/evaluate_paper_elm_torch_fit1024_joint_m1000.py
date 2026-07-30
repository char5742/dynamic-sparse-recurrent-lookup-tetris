"""Corrected joint val8 metrics for the single-root fit1024 M1000 run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import run_paper_elm_torch_cpu_fit1024_balanced_m1000 as entry


core = entry.core
runner = entry.runner


def _exact_auc(scores: np.ndarray, targets: np.ndarray) -> float:
    scores = np.asarray(scores, np.float32).reshape(-1)
    targets = np.asarray(targets, np.uint8).reshape(-1)
    positives = int(targets.sum())
    negatives = int(targets.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both classes")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    negatives_before = 0
    concordant = 0.0
    first = 0
    while first < sorted_scores.size:
        last = first + 1
        while (
            last < sorted_scores.size
            and sorted_scores[last] == sorted_scores[first]
        ):
            last += 1
        group = sorted_targets[first:last]
        group_positive = int(group.sum())
        group_negative = int(group.size - group_positive)
        concordant += (
            group_positive * negatives_before
            + 0.5 * group_positive * group_negative
        )
        negatives_before += group_negative
        first = last
    return concordant / (positives * negatives)


def _evaluate(model, dataset) -> dict:
    all_logits = []
    all_spikes = []
    voltage_error2 = 0.0
    nmda_error2 = np.zeros((4,), np.float64)
    observations = 0
    starts_julia = (1, 351, 701, 1_051)
    with torch.no_grad():
        for trial_id in core.VALIDATION_IDS:
            data, item = dataset.trial_arrays(int(trial_id))
            target_voltage = np.asarray(
                data["target_voltage"][:, item],
                np.float32,
            )
            target_spike = np.asarray(
                data["target_spike"][:, item],
                np.float32,
            )
            target_nmda = np.asarray(
                data["target_nmda"][:, :, item],
                np.float32,
            )
            for window_index, start_julia in enumerate(starts_julia):
                input_array, _, _, _ = dataset.window(
                    int(trial_id),
                    start_julia,
                    core.TRAIN_WINDOW,
                    pad_to=core.TRAIN_WINDOW,
                )
                raw = model(
                    torch.from_numpy(
                        input_array.T[:, None, :].copy()
                    )
                )[:, 0, :]
                actual_steps = min(
                    core.TRAIN_WINDOW,
                    1_500 - start_julia + 1,
                )
                local_keep_first = 0 if window_index == 0 else 150
                global_keep_first = start_julia - 1 + local_keep_first
                metric_global_first = max(global_keep_first, 500)
                global_keep_last = start_julia - 1 + actual_steps - 1
                if metric_global_first > global_keep_last:
                    continue
                local_metric_first = (
                    local_keep_first
                    + metric_global_first
                    - global_keep_first
                )
                local_slice = slice(local_metric_first, actual_steps)
                target_slice = slice(
                    metric_global_first,
                    global_keep_last + 1,
                )
                selected = raw[local_slice]
                logits = selected[:, 0].cpu().numpy()
                spikes = target_spike[target_slice]
                all_logits.append(logits)
                all_spikes.append(spikes)

                predicted_mv = (
                    selected[:, 1].double().cpu().numpy()
                    / core.SOMA_TRAIN_SCALE
                    + core.SOMA_BIAS_MV
                )
                clipped_target_mv = np.minimum(
                    target_voltage[target_slice],
                    core.SOMA_CLIP_MV,
                ).astype(np.float64)
                voltage_error2 += float(
                    np.square(
                        predicted_mv - clipped_target_mv
                    ).sum()
                )

                predicted_nmda = (
                    selected[:, 2:].double().cpu().numpy()
                )
                target_coordinate = (
                    target_nmda[:, target_slice].T
                    - model.nmda_mean.cpu().numpy()[None, :]
                ) / model.nmda_scale.cpu().numpy()[None, :]
                nmda_error2 += np.square(
                    predicted_nmda - target_coordinate
                ).sum(axis=0)
                observations += selected.shape[0]
    expected = len(core.VALIDATION_IDS) * 1_000
    if observations != expected:
        raise AssertionError(f"observations {observations} != {expected}")
    logits = np.concatenate(all_logits)
    spikes = np.concatenate(all_spikes)
    element_bce = (
        np.maximum(logits, 0.0)
        - logits * spikes
        + np.log1p(np.exp(-np.abs(logits)))
    )
    positives = spikes > 0.5
    negatives = ~positives
    positive_bce = float(element_bce[positives].mean())
    negative_bce = float(element_bce[negatives].mean())
    return {
        "validation_trials": len(core.VALIDATION_IDS),
        "observations": observations,
        "exact_spike_auroc": _exact_auc(logits, spikes),
        "clip_voltage_rmse_mv": math.sqrt(
            voltage_error2 / observations
        ),
        "nmda_normalized_rmse": np.sqrt(
            nmda_error2 / observations
        ).tolist(),
        "spike_balanced_bce": 0.5 * (
            positive_bce + negative_bce
        ),
        "positive_bce": positive_bce,
        "negative_bce": negative_bce,
        "spike_positive_bins": int(positives.sum()),
        "spike_negative_bins": int(negatives.sum()),
        "heldout_opened": False,
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    options = _parse_args()
    if options.threads != entry.EXPECTED_THREADS:
        raise ValueError("fit1024 validator requires threads=6")
    torch.set_num_threads(options.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    full_bridge, metadata = core._load_bridge(
        options.bridge.resolve()
    )
    model = runner._build(
        full_bridge,
        core.FULL_MEMORY,
        "torchscript",
    )
    payload = torch.load(
        options.checkpoint.resolve(),
        map_location="cpu",
    )
    if payload.get("schema") != "paper_elm_torch_cpu_variable.v1":
        raise ValueError("input checkpoint schema differs")
    if int(payload["memory"]) != core.FULL_MEMORY:
        raise ValueError("input checkpoint is not M1000")
    if bool(payload.get("heldout_opened", True)):
        raise ValueError("checkpoint does not assert heldout exclusion")
    source_metadata = payload.get("source_bridge_metadata", {})
    if (
        source_metadata.get("manifest_sha256")
        != metadata["manifest_sha256"]
    ):
        raise ValueError("checkpoint/dataset manifest digest differs")
    model.load_state_dict(payload["model"])
    dataset = core.SafeFitValidationDataset(
        options.dataset,
        metadata["manifest_sha256"],
    )
    result = {
        "schema": "paper_elm_torch_fit1024_joint_validation.v1",
        "checkpoint": str(options.checkpoint.resolve()),
        "update_index": int(payload["update_index"]),
        "memory": core.FULL_MEMORY,
        "hidden": core.FULL_HIDDEN,
        "metrics": _evaluate(model, dataset),
    }
    dataset.assert_no_heldout_opened()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
