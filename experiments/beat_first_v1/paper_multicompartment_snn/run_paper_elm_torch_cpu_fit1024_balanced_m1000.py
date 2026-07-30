"""Single-root fit1024 M1000 training with a safe weights-only warmstart."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import numpy as np
import torch

import train_paper_elm_torch_cpu as core
from paper_elm_torch_fitn_support import (
    fit_inventory,
    fit_nmda_normalizer,
    install_positive_balanced_objective,
    make_manifest_safe_dataset,
    sha256_file,
)
from paper_elm_torchscript_model import make_model


core.FAST_MEMORY = 256
core.FAST_HIDDEN = 512
core.FULL_MEMORY = 1_000
core.FULL_HIDDEN = 2_000
core._make_model = make_model

import train_paper_elm_torch_cpu_variable as runner


EXPECTED_FIT_IDS = tuple(range(1, 1_025))
EXPECTED_VALIDATION_IDS = tuple(range(1_025, 1_033))
EXPECTED_HELDOUT_TRIALS = 8
EXPECTED_UPDATES = 256
EXPECTED_THREADS = 6
FIXED_LEARNING_RATE = 2.0e-4
DEFAULT_DATASET = Path(
    r"C:\tmp\hd_swsnn_neuron_teacher_final_fit1024_dev1500"
)


def _argument(name: str, default=None):
    try:
        index = sys.argv.index(name)
    except ValueError:
        return default
    if index + 1 >= len(sys.argv):
        raise ValueError(f"missing value for {name}")
    return sys.argv[index + 1]


_DATASET = Path(
    _argument("--dataset", str(DEFAULT_DATASET))
).resolve()
(
    _MANIFEST,
    _FIT_IDS,
    _VALIDATION_IDS,
    _SAFE_RECORDS,
    _MANIFEST_SHA256,
) = fit_inventory(
    _DATASET,
    EXPECTED_FIT_IDS,
    EXPECTED_VALIDATION_IDS,
    EXPECTED_HELDOUT_TRIALS,
)
_NMDA_MEAN, _NMDA_SCALE, _NORMALIZER_OPENED = (
    fit_nmda_normalizer(
        _DATASET,
        _FIT_IDS,
        _SAFE_RECORDS,
    )
)

core.FIT_IDS = _FIT_IDS
core.VALIDATION_IDS = _VALIDATION_IDS
core.SafeFitValidationDataset = make_manifest_safe_dataset(
    core,
    _DATASET,
    _MANIFEST,
    _MANIFEST_SHA256,
    _FIT_IDS,
    _VALIDATION_IDS,
    _SAFE_RECORDS,
)
install_positive_balanced_objective(core)


_ORIGINAL_LOAD_BRIDGE = core._load_bridge
_ORIGINAL_PARSE_ARGS = runner._parse_args
_ORIGINAL_SAVE_CHECKPOINT = runner._save_checkpoint


def _fit1024_bridge(path: Path):
    bridge, metadata = _ORIGINAL_LOAD_BRIDGE(path)
    bridge = {
        name: np.asarray(value)
        for name, value in bridge.items()
    }
    bridge["nmda_mean"] = _NMDA_MEAN.copy()
    bridge["nmda_scale"] = _NMDA_SCALE.copy()
    bridge = runner._reset_adam_bridge(bridge)
    metadata = dict(metadata)
    metadata["manifest_sha256"] = _MANIFEST_SHA256
    metadata["update_index"] = 0
    warmstart_argument = _argument("--checkpoint-in")
    warmstart = (
        Path(warmstart_argument).resolve()
        if warmstart_argument is not None
        else None
    )
    metadata["m1000_fit1024_mainline"] = {
        "dataset_root": str(_DATASET),
        "manifest_sha256": _MANIFEST_SHA256,
        "fit_ids": [1, 1_024],
        "fit_trials": len(_FIT_IDS),
        "validation_ids": [1_025, 1_032],
        "validation_trials": len(_VALIDATION_IDS),
        "heldout_trials_excluded": EXPECTED_HELDOUT_TRIALS,
        "heldout_opened": False,
        "nmda_normalizer": "fit1024-only",
        "nmda_mean": _NMDA_MEAN.tolist(),
        "nmda_scale": _NMDA_SCALE.tolist(),
        "normalizer_opened_safe_shards": list(_NORMALIZER_OPENED),
        "weight_initialization": "weights-only M1000 checkpoint warmstart",
        "warmstart_checkpoint": (
            str(warmstart) if warmstart is not None else None
        ),
        "warmstart_sha256": (
            sha256_file(warmstart)
            if warmstart is not None and warmstart.is_file()
            else None
        ),
        "optimizer": "fresh Julia-formula Adam",
        "optimizer_state_from_warmstart": False,
        "initial_update_index": 0,
        "updates": EXPECTED_UPDATES,
        "learning_rate": FIXED_LEARNING_RATE,
        "schedule": "fixed",
        "positive_aware": True,
        "spike_loss": "class-balanced BCE",
    }
    return bridge, metadata


def _restore_weights_only(
    path: Path,
    model: torch.nn.Module,
    optimizer: core.JuliaAdam,
    expected_memory: int,
) -> int:
    del optimizer
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != "paper_elm_torch_cpu_variable.v1":
        raise ValueError("input checkpoint schema differs")
    if int(payload["memory"]) != expected_memory:
        raise ValueError("input checkpoint memory size differs")
    if expected_memory != core.FULL_MEMORY:
        raise ValueError("fit1024 warmstart requires M1000")
    if int(payload["update_index"]) != 256:
        raise ValueError("fit1024 warmstart must be the adopted u256")
    source = payload["model"]
    named = dict(model.named_parameters())
    if set(named) != set(core.PARAMETER_NAMES):
        raise ValueError("model trainable parameter contract differs")
    for name in core.PARAMETER_NAMES:
        value = source[name]
        if tuple(value.shape) != tuple(named[name].shape):
            raise ValueError(f"warmstart parameter shape differs: {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"warmstart parameter is non-finite: {name}")
        with torch.no_grad():
            named[name].copy_(value)
    # Dataset-specific normalizer buffers and all Adam state intentionally
    # remain those freshly built for the audited fit1024 root.
    return 0


def _fixed_learning_rate(update_index: int) -> float:
    if update_index < 1:
        raise ValueError("fit1024 update index must be positive")
    return FIXED_LEARNING_RATE


def _parse_fit1024_args():
    options = _ORIGINAL_PARSE_ARGS()
    if options.dataset.resolve() != _DATASET:
        raise ValueError("dataset path changed after fit1024 audit")
    if options.mode != "train":
        raise ValueError("fit1024 mainline supports train mode only")
    if options.memory != core.FULL_MEMORY:
        raise ValueError("fit1024 mainline requires memory=1000")
    if options.backend != "torchscript":
        raise ValueError("fit1024 mainline requires TorchScript")
    if options.threads != EXPECTED_THREADS:
        raise ValueError("fit1024 mainline requires threads=6")
    if options.updates != EXPECTED_UPDATES:
        raise ValueError("fit1024 first stage requires 256 updates")
    if options.checkpoint_in is None:
        raise ValueError("fit1024 mainline requires weights checkpoint")
    if options.checkpoint_out is None:
        raise ValueError("fit1024 mainline requires checkpoint output")
    return options


def _atomic_save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: core.JuliaAdam,
    memory: int,
    update_index: int,
    source_metadata: dict,
    events,
    force: bool,
    zero_isolated_from_m256: bool = False,
) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite {path}; pass --force true"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    _ORIGINAL_SAVE_CHECKPOINT(
        temporary,
        model,
        optimizer,
        memory,
        update_index,
        source_metadata,
        events,
        False,
        zero_isolated_from_m256,
    )
    os.replace(temporary, path)


core._load_bridge = _fit1024_bridge
core._learning_rate = _fixed_learning_rate
runner._restore_checkpoint = _restore_weights_only
runner._parse_args = _parse_fit1024_args
runner._save_checkpoint = _atomic_save_checkpoint


if __name__ == "__main__":
    runner.main()
