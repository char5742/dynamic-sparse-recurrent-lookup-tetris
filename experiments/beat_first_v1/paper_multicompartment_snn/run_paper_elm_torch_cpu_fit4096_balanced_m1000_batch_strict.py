"""Strict complete-contract entry for batch-configurable fit4096 M1000."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _argument(name: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError as error:
        raise ValueError(f"missing required argument {name}") from error
    if index + 1 >= len(sys.argv):
        raise ValueError(f"missing value for {name}")
    return sys.argv[index + 1]


_augmentation_root = Path(
    _argument("--augmentation-dataset")
).resolve()
_manifest = json.loads(
    (_augmentation_root / "manifest.json").read_text(encoding="utf-8")
)
if _manifest.get("completion_state") != "complete":
    raise ValueError("augmentation completion_state is not complete")
if int(_manifest.get("completed_trials", -1)) != 3_072:
    raise ValueError("augmentation completed_trials is not 3072")
_config = _manifest.get("config", {})
if int(_config.get("duration_ms", -1)) != 1_500:
    raise ValueError("augmentation duration_ms is not 1500")
if int(_config.get("train_trials", -1)) != 3_072:
    raise ValueError("augmentation train_trials is not 3072")
if int(_config.get("test_trials", 0)) != 0:
    raise ValueError("augmentation config contains test trials")
if int(_config.get("validation_trials_from_train", 0)) != 0:
    raise ValueError("augmentation config contains validation trials")

import run_paper_elm_torch_cpu_fit4096_balanced_m1000_batch as batch


core = batch.core
runner = batch.runner
_PREVIOUS_LOAD_BRIDGE = core._load_bridge


def _batch_metadata_bridge(path):
    bridge, metadata = _PREVIOUS_LOAD_BRIDGE(path)
    metadata = dict(metadata)
    details = dict(metadata["m1000_fit4096_mainline"])
    details["batch_size"] = int(core.BATCH_SIZE)
    details["window_sampler_batch_size"] = int(core.BATCH_SIZE)
    details["objective_batch_size"] = int(core.BATCH_SIZE)
    details["checkpoint_batch_size"] = int(core.BATCH_SIZE)
    metadata["m1000_fit4096_mainline"] = details
    return bridge, metadata


core._load_bridge = _batch_metadata_bridge


if __name__ == "__main__":
    runner.main()
