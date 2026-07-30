"""Critical M1000 objective concentrating capacity on voltage and spike rank."""

from __future__ import annotations

import torch
from torch.nn import functional as F

import run_paper_elm_torch_cpu_fit256_balanced_m1000_jit as _mainline


core = _mainline.core
runner = _mainline.runner


def _critical_learning_rate(update_index: int) -> float:
    if update_index < 257:
        raise ValueError(
            "critical objective continuation starts after adopted u256"
        )
    return 1.0e-4


def _critical_objective(model, batch):
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
    positive_mask = target_spike > 0.5
    negative_mask = ~positive_mask
    positive_count = torch.sum(positive_mask)
    negative_count = torch.sum(negative_mask)
    if int(positive_count) < 1 or int(negative_count) < 1:
        raise ValueError("critical batch lacks one spike class")
    positive_bce = torch.mean(element_bce[positive_mask])
    negative_bce = torch.mean(element_bce[negative_mask])
    spike_balanced_bce = 0.5 * (
        positive_bce + negative_bce
    )
    positive_logits = spike_logit[positive_mask]
    negative_logits = spike_logit[negative_mask]
    pairwise_rank = torch.mean(
        F.softplus(
            negative_logits.unsqueeze(0)
            - positive_logits.unsqueeze(1)
        )
    )

    target_nmda_coordinate = (
        target_nmda
        - model.nmda_mean.view(1, 1, 4)
    ) / model.nmda_scale.view(1, 1, 4)
    nmda_extension_loss = torch.mean(
        torch.square(nmda - target_nmda_coordinate)
    )
    total = (
        voltage_mse
        + 0.5 * spike_balanced_bce
        + 0.5 * pairwise_rank
        + 0.25 * nmda_extension_loss
    )
    return total, {
        "total": total,
        "voltage_mse": voltage_mse,
        "spike_balanced_bce": spike_balanced_bce,
        "positive_bce": positive_bce,
        "negative_bce": negative_bce,
        "pairwise_rank": pairwise_rank,
        "nmda_extension_loss": nmda_extension_loss,
        "spike_positives": positive_count,
        "spike_negatives": negative_count,
    }


core._learning_rate = _critical_learning_rate
core._objective = _critical_objective


if __name__ == "__main__":
    runner.main()
