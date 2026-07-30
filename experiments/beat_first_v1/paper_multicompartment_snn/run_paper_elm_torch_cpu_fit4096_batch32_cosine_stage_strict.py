"""Hardened add-only adapter for future fit4096 cosine stages."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

import run_paper_elm_torch_cpu_fit4096_batch32_cosine_stage as stage


core = stage.core
runner = stage.runner
_BASE_RESTORE = runner._restore_checkpoint
_BASE_LOAD_BRIDGE = core._load_bridge


def _strict_continuation_bridge(path):
    bridge, metadata = _BASE_LOAD_BRIDGE(path)
    metadata = dict(metadata)
    details = dict(metadata["m1000_fit4096_mainline"])
    details["validation_source"] = "base-only"
    details["validation_order"] = (
        "atomic checkpoint, SHA/schema verification, then base val8"
    )
    metadata["m1000_fit4096_mainline"] = details
    return bridge, metadata


def _strict_restore(
    path,
    model,
    optimizer,
    expected_memory,
) -> int:
    path = Path(path).resolve()
    expected_sha256 = stage._INPUT_SHA256
    if expected_sha256 is None:
        raise RuntimeError("arguments must be parsed before restore")
    if stage.mainline.sha256_file(path) != expected_sha256:
        raise ValueError("input checkpoint changed before restore")
    payload = torch.load(path, map_location="cpu")
    prior_update = int(payload["update_index"])
    details = payload["source_bridge_metadata"][
        "m1000_fit4096_mainline"
    ]
    events = list(payload.get("events", ()))
    initial = int(details["initial_update_index"])
    updates = int(details["updates"])
    final = int(details["final_update_index"])
    if final != prior_update:
        raise ValueError("checkpoint final_update_index differs")
    if len(events) != updates:
        raise ValueError("checkpoint event count differs")
    if not events:
        raise ValueError("checkpoint has no events")
    if int(events[0]["update_index"]) != initial + 1:
        raise ValueError("checkpoint first event differs")
    if int(events[-1]["update_index"]) != final:
        raise ValueError("checkpoint last event differs")
    validation_source = str(details.get("validation_source", ""))
    if not validation_source.startswith("base-only"):
        raise ValueError("checkpoint validation source differs")
    epsilon = np.float32(payload["julia_adam"]["epsilon"])
    restored_update = _BASE_RESTORE(
        path,
        model,
        optimizer,
        expected_memory,
    )
    optimizer.epsilon = epsilon
    if not math.isclose(
        float(optimizer.epsilon),
        float(epsilon),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("Adam epsilon restoration differs")
    return restored_update


core._load_bridge = _strict_continuation_bridge
runner._restore_checkpoint = _strict_restore


if __name__ == "__main__":
    runner.main()
