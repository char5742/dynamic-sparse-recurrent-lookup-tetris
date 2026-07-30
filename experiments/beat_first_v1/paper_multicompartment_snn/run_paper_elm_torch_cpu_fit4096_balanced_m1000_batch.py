"""Batch-configurable fit4096 M1000 benchmark/training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import run_paper_elm_torch_cpu_fit4096_balanced_m1000 as mainline


core = mainline.core
runner = mainline.runner


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--augmentation-dataset",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--mode",
        choices=("benchmark", "train"),
        required=True,
    )
    parser.add_argument("--memory", type=int, choices=(1_000,), required=True)
    parser.add_argument(
        "--backend",
        choices=("torchscript",),
        required=True,
    )
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=(8, 32),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=6077687918186389330)
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument(
        "--force",
        type=core._parse_bool,
        default=False,
    )
    options = parser.parse_args()
    if options.dataset.resolve() != mainline._BASE_ROOT:
        raise ValueError("base dataset changed after audit")
    if (
        options.augmentation_dataset.resolve()
        != mainline._AUGMENTATION_ROOT
    ):
        raise ValueError("augmentation dataset changed after audit")
    if options.threads != mainline.EXPECTED_THREADS:
        raise ValueError("fit4096 requires threads=6")
    if options.mode == "benchmark":
        if options.updates != 1:
            raise ValueError("benchmark requires one update")
        if options.checkpoint_out is not None:
            raise ValueError("benchmark must not write a checkpoint")
    else:
        if options.updates != mainline.EXPECTED_UPDATES:
            raise ValueError("fit4096 stage requires 256 updates")
        if options.checkpoint_out is None:
            raise ValueError("training requires checkpoint output")
    core.BATCH_SIZE = options.batch_size
    return options


_ORIGINAL_RESTORE = runner._restore_checkpoint


def _strict_weights_only_restore(
    path,
    model,
    optimizer,
    expected_memory,
) -> int:
    payload = torch.load(path, map_location="cpu")
    source = payload.get("source_bridge_metadata", {})
    if (
        source.get("manifest_sha256")
        != mainline._BASE_MANIFEST_SHA256
    ):
        raise ValueError("warmstart is not bound to base fit1024")
    if bool(payload.get("heldout_opened", True)):
        raise ValueError("warmstart does not assert heldout exclusion")
    return _ORIGINAL_RESTORE(
        path,
        model,
        optimizer,
        expected_memory,
    )


runner._parse_args = _parse_args
runner._restore_checkpoint = _strict_weights_only_restore


if __name__ == "__main__":
    runner.main()
