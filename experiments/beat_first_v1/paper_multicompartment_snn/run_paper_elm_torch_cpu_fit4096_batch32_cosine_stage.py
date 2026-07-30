"""One exact fit4096 batch32 cosine stage toward cumulative u4480."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

import run_paper_elm_torch_cpu_fit4096_balanced_m1000_batch_strict as strict


batch = strict.batch
mainline = batch.mainline
core = strict.core
runner = strict.runner

EXPECTED_SEED = 6077687918186389330
BATCH_SIZE = 32
FIXED_THROUGH_UPDATE = 512
TARGET_UPDATE = 4_480
MAX_LEARNING_RATE = 2.0e-4
FLOOR_LEARNING_RATE = 2.0e-5

_PREVIOUS_LOAD_BRIDGE = core._load_bridge
_ORIGINAL_DEFAULT_RNG = runner.np.random.default_rng
_RESUME_EVENTS = None
_PRIOR_UPDATE = None
_INPUT_SHA256 = None


def _cosine_learning_rate(update_index: int) -> float:
    if update_index <= FIXED_THROUGH_UPDATE:
        return MAX_LEARNING_RATE
    if update_index > TARGET_UPDATE:
        raise ValueError("update exceeds paper-equivalent target")
    progress = (
        (update_index - FIXED_THROUGH_UPDATE)
        / (TARGET_UPDATE - FIXED_THROUGH_UPDATE)
    )
    return FLOOR_LEARNING_RATE + 0.5 * (
        MAX_LEARNING_RATE - FLOOR_LEARNING_RATE
    ) * (1.0 + math.cos(math.pi * progress))


def _parse_args():
    global _PRIOR_UPDATE, _INPUT_SHA256
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--augmentation-dataset",
        type=Path,
        required=True,
    )
    parser.add_argument("--mode", choices=("train",), required=True)
    parser.add_argument("--memory", type=int, choices=(1_000,), required=True)
    parser.add_argument(
        "--backend",
        choices=("torchscript",),
        required=True,
    )
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--updates", type=int, choices=(128, 256), required=True)
    parser.add_argument("--batch-size", type=int, choices=(32,), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-sha256",
        type=str,
        required=True,
    )
    parser.add_argument("--checkpoint-out", type=Path, required=True)
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
        raise ValueError("fit4096 cosine stage requires threads=6")
    if options.seed != EXPECTED_SEED:
        raise ValueError("fit4096 cosine stage requires original seed")
    actual_sha256 = mainline.sha256_file(
        options.checkpoint_in.resolve()
    )
    expected_sha256 = options.checkpoint_sha256.lower()
    if actual_sha256 != expected_sha256:
        raise ValueError("input checkpoint SHA256 differs")
    payload = torch.load(
        options.checkpoint_in.resolve(),
        map_location="cpu",
    )
    prior_update = int(payload["update_index"])
    if prior_update < FIXED_THROUGH_UPDATE:
        raise ValueError("cosine stage requires cumulative u512 or later")
    if prior_update >= TARGET_UPDATE:
        raise ValueError("paper-equivalent target already reached")
    expected_updates = min(256, TARGET_UPDATE - prior_update)
    if options.updates != expected_updates:
        raise ValueError(
            f"stage from u{prior_update} requires "
            f"{expected_updates} updates"
        )
    core.BATCH_SIZE = BATCH_SIZE
    _PRIOR_UPDATE = prior_update
    _INPUT_SHA256 = actual_sha256
    return options


def _continuation_bridge(path):
    bridge, metadata = _PREVIOUS_LOAD_BRIDGE(path)
    if _PRIOR_UPDATE is None or _INPUT_SHA256 is None:
        raise RuntimeError("arguments must be parsed before bridge load")
    metadata = dict(metadata)
    details = dict(metadata["m1000_fit4096_mainline"])
    stage_updates = min(256, TARGET_UPDATE - _PRIOR_UPDATE)
    stage_end = _PRIOR_UPDATE + stage_updates
    details.update({
        "weight_initialization": "resumed fit4096 batch32 checkpoint",
        "resume_sha256": _INPUT_SHA256,
        "optimizer": "preserved Julia-formula Adam",
        "optimizer_state_from_warmstart": True,
        "initial_update_index": _PRIOR_UPDATE,
        "updates": stage_updates,
        "final_update_index": stage_end,
        "batch_size": BATCH_SIZE,
        "window_sampler_batch_size": BATCH_SIZE,
        "objective_batch_size": BATCH_SIZE,
        "checkpoint_batch_size": BATCH_SIZE,
        "schedule": "paper-like cosine decay",
        "schedule_formula": (
            "floor + 0.5*(max-floor)*(1+cos(pi*"
            "(update-512)/(4480-512)))"
        ),
        "fixed_learning_rate_through_update": FIXED_THROUGH_UPDATE,
        "cosine_decay_first_update": FIXED_THROUGH_UPDATE + 1,
        "cosine_decay_final_update": TARGET_UPDATE,
        "cosine_max_learning_rate": MAX_LEARNING_RATE,
        "cosine_floor_learning_rate": FLOOR_LEARNING_RATE,
        "stage_first_learning_rate": _cosine_learning_rate(
            _PRIOR_UPDATE + 1
        ),
        "stage_last_learning_rate": _cosine_learning_rate(stage_end),
        "paper_equivalent_epochs": 35,
        "updates_per_epoch": 128,
        "paper_equivalent_target_update": TARGET_UPDATE,
        "rng_resume": "seed replay verified against checkpoint events",
        "rng_seed": EXPECTED_SEED,
        "validation_source": "base-only after atomic checkpoint",
        "heldout_opened": False,
    })
    metadata["m1000_fit4096_mainline"] = details
    return bridge, metadata


def _restore_full_checkpoint(
    path,
    model,
    optimizer,
    expected_memory,
) -> int:
    global _RESUME_EVENTS
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != "paper_elm_torch_cpu_variable.v1":
        raise ValueError("input checkpoint schema differs")
    if int(payload["memory"]) != expected_memory:
        raise ValueError("input checkpoint memory differs")
    prior_update = int(payload["update_index"])
    if prior_update != _PRIOR_UPDATE:
        raise ValueError("input checkpoint update changed")
    if bool(payload.get("heldout_opened", True)):
        raise ValueError("checkpoint does not assert heldout exclusion")
    source = payload.get("source_bridge_metadata", {})
    if source.get("manifest_sha256") != mainline._COMPOSITE_SHA256:
        raise ValueError("checkpoint/composite contract differs")
    details = source.get("m1000_fit4096_mainline", {})
    required = {
        "base_manifest_sha256": mainline._BASE_MANIFEST_SHA256,
        "augmentation_manifest_sha256": (
            mainline._AUGMENTATION_MANIFEST_SHA256
        ),
        "composite_sha256": mainline._COMPOSITE_SHA256,
        "batch_size": BATCH_SIZE,
        "checkpoint_batch_size": BATCH_SIZE,
        "fit_trials": 4_096,
    }
    for key, expected in required.items():
        if details.get(key) != expected:
            raise ValueError(f"checkpoint contract differs: {key}")
    events = list(payload.get("events", ()))
    if not events:
        raise ValueError("checkpoint has no stage events")
    indices = [int(event["update_index"]) for event in events]
    if indices != list(range(indices[0], prior_update + 1)):
        raise ValueError("checkpoint stage event sequence differs")
    for event in events:
        update_index = int(event["update_index"])
        expected_lr = _cosine_learning_rate(update_index)
        if not math.isclose(
            float(event["learning_rate"]),
            expected_lr,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError("checkpoint learning-rate history differs")
        if not bool(event["gradient"]["finite"]):
            raise ValueError("checkpoint contains non-finite gradient")

    model.load_state_dict(payload["model"])
    state = payload["julia_adam"]
    for name in core.PARAMETER_NAMES:
        optimizer.first[name].copy_(state["first"][name])
        optimizer.second[name].copy_(state["second"][name])
    optimizer.beta.copy_(state["beta"])
    optimizer.beta_power.copy_(state["beta_power"])
    _RESUME_EVENTS = events
    return prior_update


def _replayed_default_rng(seed=None):
    if seed != EXPECTED_SEED:
        raise ValueError("RNG seed differs during cosine resume")
    if _PRIOR_UPDATE is None or _RESUME_EVENTS is None:
        raise RuntimeError("checkpoint must be restored before RNG replay")
    events_by_update = {
        int(event["update_index"]): event
        for event in _RESUME_EVENTS
    }
    rng = _ORIGINAL_DEFAULT_RNG(seed)
    fit_handles = np.asarray(core.FIT_IDS)
    for update_index in range(1, _PRIOR_UPDATE + 1):
        ids = rng.choice(
            fit_handles,
            size=BATCH_SIZE,
            replace=False,
        ).tolist()
        starts = rng.integers(
            core.FIRST_RANDOM_START_JULIA,
            core.LAST_RANDOM_START_JULIA + 1,
            size=BATCH_SIZE,
        ).tolist()
        expected = events_by_update.get(update_index)
        if expected is None:
            continue
        if ids != [int(value) for value in expected["ids"]]:
            raise ValueError("RNG replay trial IDs differ")
        if starts != [
            int(value)
            for value in expected["crop_starts_julia"]
        ]:
            raise ValueError("RNG replay crop starts differ")
    return rng


core._load_bridge = _continuation_bridge
core._learning_rate = _cosine_learning_rate
runner._parse_args = _parse_args
runner._restore_checkpoint = _restore_full_checkpoint
runner.np.random.default_rng = _replayed_default_rng


if __name__ == "__main__":
    runner.main()
