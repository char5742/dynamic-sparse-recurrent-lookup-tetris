"""Hardened staged orchestrator using strict cosine resume workers."""

from __future__ import annotations

import math
from pathlib import Path

import run_paper_elm_torch_fit4096_staged_to_u4480 as staged


staged.STAGE_RUNNER = Path(__file__).with_name(
    "run_paper_elm_torch_cpu_fit4096_batch32_cosine_stage_strict.py"
)
_BASE_GATE_ACHIEVED = staged._gate_achieved
_BASE_BEST_KEY = staged._best_key


def _validate_metrics(metrics: dict) -> None:
    if int(metrics.get("validation_trials", -1)) != 8:
        raise ValueError("base validation trial count differs")
    if int(metrics.get("observations", -1)) != 8_000:
        raise ValueError("base validation observations differ")
    if bool(metrics.get("heldout_opened", True)):
        raise ValueError("validation does not assert heldout exclusion")
    scalar_names = (
        "exact_spike_auroc",
        "clip_voltage_rmse_mv",
        "spike_balanced_bce",
        "positive_bce",
        "negative_bce",
    )
    if any(
        not math.isfinite(float(metrics[name]))
        for name in scalar_names
    ):
        raise ValueError("validation contains non-finite scalar")
    nmda = metrics.get("nmda_normalized_rmse", ())
    if len(nmda) != 4 or any(
        not math.isfinite(float(value))
        for value in nmda
    ):
        raise ValueError("validation contains invalid NMDA metrics")


def _strict_gate_achieved(metrics: dict, gates: dict) -> bool:
    _validate_metrics(metrics)
    return _BASE_GATE_ACHIEVED(metrics, gates)


def _strict_best_key(record: dict):
    _validate_metrics(record["metrics"])
    return _BASE_BEST_KEY(record)


staged._gate_achieved = _strict_gate_achieved
staged._best_key = _strict_best_key


if __name__ == "__main__":
    staged.main()
