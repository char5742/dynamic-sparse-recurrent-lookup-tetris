"""Strict fit4096 batch32 resume from cumulative u256 to u512."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

import run_paper_elm_torch_cpu_fit4096_balanced_m1000_batch_strict as strict


batch = strict.batch
mainline = batch.mainline
core = strict.core
runner = strict.runner

EXPECTED_INPUT_SHA256 = (
    "5220c93213efbf196a145572a497e8a7f1cc100237aa1bab9f87736dcacf4754"
)
EXPECTED_SEED = 6077687918186389330
EXPECTED_PRIOR_UPDATES = 256
EXPECTED_ADDITIONAL_UPDATES = 256
EXPECTED_FINAL_UPDATE = 512
EXPECTED_BATCH_SIZE = 32
FIXED_LEARNING_RATE = 2.0e-4

_PREVIOUS_LOAD_BRIDGE = core._load_bridge
_PREVIOUS_PARSE_ARGS = runner._parse_args
_ORIGINAL_DEFAULT_RNG = runner.np.random.default_rng
_RESUME_EVENTS = None


def _argument(name: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError as error:
        raise ValueError(f"missing required argument {name}") from error
    if index + 1 >= len(sys.argv):
        raise ValueError(f"missing value for {name}")
    return sys.argv[index + 1]


def _continuation_bridge(path):
    bridge, metadata = _PREVIOUS_LOAD_BRIDGE(path)
    metadata = dict(metadata)
    details = dict(metadata["m1000_fit4096_mainline"])
    resume_path = Path(_argument("--checkpoint-in")).resolve()
    resume_sha256 = mainline.sha256_file(resume_path)
    if resume_sha256 != EXPECTED_INPUT_SHA256:
        raise ValueError("fit4096 resume checkpoint SHA256 differs")
    details.update({
        "weight_initialization": "resumed fit4096 batch32 u256",
        "resume_checkpoint": str(resume_path),
        "resume_sha256": resume_sha256,
        "optimizer": "preserved Julia-formula Adam",
        "optimizer_state_from_warmstart": True,
        "initial_update_index": EXPECTED_PRIOR_UPDATES,
        "updates": EXPECTED_ADDITIONAL_UPDATES,
        "final_update_index": EXPECTED_FINAL_UPDATE,
        "batch_size": EXPECTED_BATCH_SIZE,
        "window_sampler_batch_size": EXPECTED_BATCH_SIZE,
        "objective_batch_size": EXPECTED_BATCH_SIZE,
        "checkpoint_batch_size": EXPECTED_BATCH_SIZE,
        "learning_rate": FIXED_LEARNING_RATE,
        "schedule": "fixed continuation",
        "rng_resume": "seed replay verified against prior events",
        "rng_seed": EXPECTED_SEED,
        "validation_source": "base-only",
        "heldout_opened": False,
    })
    metadata["m1000_fit4096_mainline"] = details
    return bridge, metadata


def _parse_resume_args():
    options = _PREVIOUS_PARSE_ARGS()
    if options.mode != "train":
        raise ValueError("fit4096 resume supports train mode only")
    if options.batch_size != EXPECTED_BATCH_SIZE:
        raise ValueError("fit4096 resume requires batch32")
    if options.updates != EXPECTED_ADDITIONAL_UPDATES:
        raise ValueError("fit4096 resume requires 256 updates")
    if options.seed != EXPECTED_SEED:
        raise ValueError("fit4096 resume requires the original RNG seed")
    if options.checkpoint_out is None:
        raise ValueError("fit4096 resume requires checkpoint output")
    return options


def _restore_full_checkpoint(
    path,
    model,
    optimizer,
    expected_memory,
) -> int:
    global _RESUME_EVENTS
    path = Path(path).resolve()
    if mainline.sha256_file(path) != EXPECTED_INPUT_SHA256:
        raise ValueError("fit4096 resume checkpoint SHA256 differs")
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != "paper_elm_torch_cpu_variable.v1":
        raise ValueError("input checkpoint schema differs")
    if int(payload["memory"]) != expected_memory:
        raise ValueError("input checkpoint memory size differs")
    if int(payload["update_index"]) != EXPECTED_PRIOR_UPDATES:
        raise ValueError("fit4096 resume requires cumulative u256")
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
        "batch_size": EXPECTED_BATCH_SIZE,
        "checkpoint_batch_size": EXPECTED_BATCH_SIZE,
        "learning_rate": FIXED_LEARNING_RATE,
        "fit_trials": 4_096,
        "validation_source": "base-only",
    }
    for key, expected in required.items():
        if details.get(key) != expected:
            raise ValueError(f"checkpoint contract differs: {key}")
    events = list(payload.get("events", ()))
    if len(events) != EXPECTED_PRIOR_UPDATES:
        raise ValueError("checkpoint prior event count differs")
    if [int(event["update_index"]) for event in events] != list(
        range(1, EXPECTED_PRIOR_UPDATES + 1)
    ):
        raise ValueError("checkpoint update sequence differs")
    if any(
        float(event["learning_rate"]) != FIXED_LEARNING_RATE
        for event in events
    ):
        raise ValueError("checkpoint learning-rate history differs")
    if any(not bool(event["gradient"]["finite"]) for event in events):
        raise ValueError("checkpoint contains non-finite gradient")

    model.load_state_dict(payload["model"])
    state = payload["julia_adam"]
    for name in core.PARAMETER_NAMES:
        optimizer.first[name].copy_(state["first"][name])
        optimizer.second[name].copy_(state["second"][name])
    optimizer.beta.copy_(state["beta"])
    optimizer.beta_power.copy_(state["beta_power"])
    _RESUME_EVENTS = events
    return EXPECTED_PRIOR_UPDATES


def _replayed_default_rng(seed=None):
    if seed != EXPECTED_SEED:
        raise ValueError("RNG seed differs during fit4096 resume")
    if _RESUME_EVENTS is None:
        raise RuntimeError("checkpoint must be restored before RNG replay")
    rng = _ORIGINAL_DEFAULT_RNG(seed)
    fit_handles = np.asarray(core.FIT_IDS)
    for expected in _RESUME_EVENTS:
        ids = rng.choice(
            fit_handles,
            size=EXPECTED_BATCH_SIZE,
            replace=False,
        ).tolist()
        starts = rng.integers(
            core.FIRST_RANDOM_START_JULIA,
            core.LAST_RANDOM_START_JULIA + 1,
            size=EXPECTED_BATCH_SIZE,
        ).tolist()
        if ids != [int(value) for value in expected["ids"]]:
            raise ValueError("RNG replay trial IDs differ")
        if starts != [
            int(value)
            for value in expected["crop_starts_julia"]
        ]:
            raise ValueError("RNG replay crop starts differ")
    return rng


core._load_bridge = _continuation_bridge
runner._parse_args = _parse_resume_args
runner._restore_checkpoint = _restore_full_checkpoint
runner.np.random.default_rng = _replayed_default_rng


if __name__ == "__main__":
    runner.main()
