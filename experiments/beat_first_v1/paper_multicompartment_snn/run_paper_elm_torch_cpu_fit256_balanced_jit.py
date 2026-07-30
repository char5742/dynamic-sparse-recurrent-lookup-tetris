"""Fit256 TorchScript runner with positive-aware balanced spike training."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.nn import functional as F

import run_paper_elm_torch_cpu_fit256_jit as _entry


core = _entry._core
runner = _entry._fit256._runner
_original_materialize_batch = core._materialize_batch
_positive_catalog: Dict[int, List[int]] | None = None


def _catalog(dataset) -> Dict[int, List[int]]:
    global _positive_catalog
    if _positive_catalog is not None:
        return _positive_catalog
    result: Dict[int, List[int]] = {}
    for trial_id in core.FIT_IDS:
        data, item = dataset.trial_arrays(int(trial_id))
        spike = data["target_spike"][:, item]
        prefix = spike.astype("int64").cumsum()
        prefix = core.np.concatenate(
            (core.np.zeros((1,), dtype=core.np.int64), prefix)
        )
        starts = []
        for start_julia in range(
            core.FIRST_RANDOM_START_JULIA,
            core.LAST_RANDOM_START_JULIA + 1,
        ):
            first = start_julia - 1
            last = first + core.TRAIN_WINDOW
            if int(prefix[last] - prefix[first]) > 0:
                starts.append(start_julia)
        result[int(trial_id)] = starts
    if not any(result.values()):
        raise ValueError("fit split has no spike-positive training window")
    _positive_catalog = result
    return result


def _positive_materialize_batch(
    dataset,
    ids,
    starts_julia,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = _original_materialize_batch(dataset, ids, starts_julia)
    if float(batch[2].sum()) > 0:
        return batch
    catalog = _catalog(dataset)
    selected_ids = [int(value) for value in ids]
    selected_starts = [int(value) for value in starts_julia]
    eligible_slots = [
        slot
        for slot, trial_id in enumerate(selected_ids)
        if catalog[trial_id]
    ]
    if eligible_slots:
        slot = eligible_slots[0]
        selected_starts[slot] = catalog[selected_ids[slot]][0]
    else:
        replacements = [
            trial_id
            for trial_id in core.FIT_IDS
            if catalog[int(trial_id)]
            and int(trial_id) not in selected_ids
        ]
        if not replacements:
            raise ValueError("cannot choose a distinct positive anchor")
        selected_ids[0] = int(replacements[0])
        selected_starts[0] = catalog[selected_ids[0]][0]
    batch = _original_materialize_batch(
        dataset,
        selected_ids,
        selected_starts,
    )
    if float(batch[2].sum()) <= 0:
        raise AssertionError("positive-aware batch has no spike")
    return batch


def _balanced_objective(model, batch):
    inputs, target_voltage, target_spike, target_nmda = batch
    raw = model(inputs)
    spike_logit = raw[:, :, 0]
    voltage = raw[:, :, 1]
    nmda = raw[:, :, 2:]
    target_voltage_coordinate = (
        torch.minimum(
            target_voltage,
            target_voltage.new_tensor(core.SOMA_CLIP_MV),
        )
        - core.SOMA_BIAS_MV
    ) * core.SOMA_TRAIN_SCALE
    voltage_mse = torch.mean(
        torch.square(voltage - target_voltage_coordinate)
    )
    element_bce = F.binary_cross_entropy_with_logits(
        spike_logit,
        target_spike,
        reduction="none",
    )
    positives = torch.sum(target_spike)
    negatives = target_spike.numel() - positives
    if float(positives) <= 0 or float(negatives) <= 0:
        raise ValueError("balanced spike batch lacks one class")
    positive_bce = torch.sum(
        element_bce * target_spike
    ) / positives
    negative_bce = torch.sum(
        element_bce * (1.0 - target_spike)
    ) / negatives
    spike_balanced_bce = 0.5 * positive_bce + 0.5 * negative_bce
    target_nmda_coordinate = (
        target_nmda
        - model.nmda_mean.view(1, 1, 4)
    ) / model.nmda_scale.view(1, 1, 4)
    nmda_extension_loss = torch.mean(
        torch.square(nmda - target_nmda_coordinate)
    )
    paper_loss = (
        0.5 * voltage_mse
        + 0.5 * spike_balanced_bce
    )
    total = paper_loss + nmda_extension_loss
    return total, {
        "total": total,
        "paper_loss": paper_loss,
        "voltage_mse": voltage_mse,
        "spike_balanced_bce": spike_balanced_bce,
        "positive_bce": positive_bce,
        "negative_bce": negative_bce,
        "nmda_extension_loss": nmda_extension_loss,
        "spike_positives": positives,
        "spike_negatives": negatives,
    }


core._materialize_batch = _positive_materialize_batch
core._objective = _balanced_objective


if __name__ == "__main__":
    runner.main()
