"""Corrected joint val8 metrics for the M1000 fit256 mainline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_paper_elm_torch_cpu_fit256_balanced_m1000_jit as entry
import evaluate_paper_elm_torch_fit256_joint as joint


core = entry.core
runner = entry.runner


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    options = _parse_args()
    torch.set_num_threads(options.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    full_bridge, metadata = core._load_bridge(options.bridge.resolve())
    model = runner._build(full_bridge, 1_000, "torchscript")
    optimizer = core.JuliaAdam(model, full_bridge)
    update_index = runner._restore_checkpoint(
        options.checkpoint.resolve(),
        model,
        optimizer,
        1_000,
    )
    dataset = core.SafeFitValidationDataset(
        options.dataset,
        metadata["manifest_sha256"],
    )
    result = {
        "schema": "paper_elm_torch_fit256_joint_validation.v1",
        "checkpoint": str(options.checkpoint.resolve()),
        "update_index": update_index,
        "memory": 1_000,
        "hidden": 2_000,
        "metrics": joint._evaluate(model, dataset),
    }
    dataset.assert_no_heldout_opened()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
