"""Composite fit4096 M1000 stage: base1024 plus train-only aug3072."""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import torch

import train_paper_elm_torch_cpu as core
from paper_elm_torch_composite_fit_support import (
    composite_contract_sha256,
    fit_nmda_normalizer_many,
    make_composite_dataset,
    train_only_inventory,
)
from paper_elm_torch_fitn_support import (
    fit_inventory,
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


BASE_FIT_IDS = tuple(range(1, 1_025))
BASE_VALIDATION_IDS = tuple(range(1_025, 1_033))
AUGMENTATION_IDS = tuple(range(1, 3_073))
EXPECTED_UPDATES = 256
EXPECTED_THREADS = 6
FIXED_LEARNING_RATE = 2.0e-4


def _argument(name: str, default=None):
    try:
        index = sys.argv.index(name)
    except ValueError:
        return default
    if index + 1 >= len(sys.argv):
        raise ValueError(f"missing value for {name}")
    return sys.argv[index + 1]


_BASE_ROOT = Path(_argument("--dataset")).resolve()
_AUGMENTATION_ROOT = Path(
    _argument("--augmentation-dataset")
).resolve()
(
    _BASE_MANIFEST,
    _BASE_FIT_IDS,
    _BASE_VALIDATION_IDS,
    _BASE_RECORDS,
    _BASE_MANIFEST_SHA256,
) = fit_inventory(
    _BASE_ROOT,
    BASE_FIT_IDS,
    BASE_VALIDATION_IDS,
    8,
)
(
    _AUGMENTATION_MANIFEST,
    _AUGMENTATION_IDS,
    _AUGMENTATION_RECORDS,
    _AUGMENTATION_MANIFEST_SHA256,
) = train_only_inventory(
    _AUGMENTATION_ROOT,
    AUGMENTATION_IDS,
)
_COMPOSITE_SHA256 = composite_contract_sha256(
    _BASE_MANIFEST_SHA256,
    _AUGMENTATION_MANIFEST_SHA256,
)
_NMDA_MEAN, _NMDA_SCALE, _NORMALIZER_OPENED = (
    fit_nmda_normalizer_many((
        (_BASE_ROOT, _BASE_FIT_IDS, _BASE_RECORDS),
        (
            _AUGMENTATION_ROOT,
            _AUGMENTATION_IDS,
            _AUGMENTATION_RECORDS,
        ),
    ))
)

_BaseDataset = make_manifest_safe_dataset(
    core,
    _BASE_ROOT,
    _BASE_MANIFEST,
    _BASE_MANIFEST_SHA256,
    _BASE_FIT_IDS,
    _BASE_VALIDATION_IDS,
    _BASE_RECORDS,
)
_AugmentationDataset = make_manifest_safe_dataset(
    core,
    _AUGMENTATION_ROOT,
    _AUGMENTATION_MANIFEST,
    _AUGMENTATION_MANIFEST_SHA256,
    _AUGMENTATION_IDS,
    (),
    _AUGMENTATION_RECORDS,
)
_CompositeDataset, _FIT_HANDLES = make_composite_dataset(
    _BaseDataset,
    _AugmentationDataset,
    _BASE_ROOT,
    _AUGMENTATION_ROOT,
    _BASE_MANIFEST_SHA256,
    _AUGMENTATION_MANIFEST_SHA256,
    _COMPOSITE_SHA256,
    _BASE_FIT_IDS,
    _AUGMENTATION_IDS,
)
core.FIT_IDS = _FIT_HANDLES
core.VALIDATION_IDS = _BASE_VALIDATION_IDS
core.SafeFitValidationDataset = _CompositeDataset
install_positive_balanced_objective(core)


_ORIGINAL_LOAD_BRIDGE = core._load_bridge
_ORIGINAL_SAVE_CHECKPOINT = runner._save_checkpoint


def _fit4096_bridge(path: Path):
    bridge, metadata = _ORIGINAL_LOAD_BRIDGE(path)
    bridge = {
        name: np.asarray(value)
        for name, value in bridge.items()
    }
    bridge["nmda_mean"] = _NMDA_MEAN.copy()
    bridge["nmda_scale"] = _NMDA_SCALE.copy()
    bridge = runner._reset_adam_bridge(bridge)
    metadata = dict(metadata)
    metadata["manifest_sha256"] = _COMPOSITE_SHA256
    metadata["update_index"] = 0
    warmstart = Path(_argument("--checkpoint-in")).resolve()
    metadata["m1000_fit4096_mainline"] = {
        "schema": "paper_elm_composite_fit.v1",
        "base_root": str(_BASE_ROOT),
        "base_manifest_sha256": _BASE_MANIFEST_SHA256,
        "base_fit_trials": len(_BASE_FIT_IDS),
        "base_validation_ids": [1_025, 1_032],
        "base_validation_trials": len(_BASE_VALIDATION_IDS),
        "base_heldout_trials_excluded": 8,
        "augmentation_root": str(_AUGMENTATION_ROOT),
        "augmentation_manifest_sha256": (
            _AUGMENTATION_MANIFEST_SHA256
        ),
        "augmentation_fit_trials": len(_AUGMENTATION_IDS),
        "augmentation_heldout_trials": 0,
        "fit_trials": len(_FIT_HANDLES),
        "composite_sha256": _COMPOSITE_SHA256,
        "nmda_normalizer": "base1024-plus-augmentation3072 fit-only",
        "nmda_mean": _NMDA_MEAN.tolist(),
        "nmda_scale": _NMDA_SCALE.tolist(),
        "normalizer_opened_safe_shards": list(_NORMALIZER_OPENED),
        "weight_initialization": "weights-only fit1024 M1000 u512",
        "warmstart_checkpoint": str(warmstart),
        "warmstart_sha256": sha256_file(warmstart),
        "optimizer": "fresh Julia-formula Adam",
        "optimizer_state_from_warmstart": False,
        "initial_update_index": 0,
        "updates": EXPECTED_UPDATES,
        "learning_rate": FIXED_LEARNING_RATE,
        "schedule": "fixed",
        "positive_aware": True,
        "spike_loss": "class-balanced BCE",
        "validation_source": "base-only",
        "heldout_opened": False,
    }
    return bridge, metadata


def _restore_weights_only(path, model, optimizer, expected_memory):
    del optimizer
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != "paper_elm_torch_cpu_variable.v1":
        raise ValueError("input checkpoint schema differs")
    if int(payload["memory"]) != expected_memory:
        raise ValueError("input checkpoint memory size differs")
    if int(payload["update_index"]) != 512:
        raise ValueError("fit4096 warmstart requires fit1024 u512")
    source = payload["model"]
    named = dict(model.named_parameters())
    if set(named) != set(core.PARAMETER_NAMES):
        raise ValueError("model trainable parameter contract differs")
    for name in core.PARAMETER_NAMES:
        value = source[name]
        if tuple(value.shape) != tuple(named[name].shape):
            raise ValueError(f"warmstart shape differs: {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"warmstart is non-finite: {name}")
        with torch.no_grad():
            named[name].copy_(value)
    return 0


def _fixed_learning_rate(update_index: int) -> float:
    if update_index < 1:
        raise ValueError("fit4096 update index must be positive")
    return FIXED_LEARNING_RATE


def _parse_args():
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
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--seed", type=int, default=6077687918186389330)
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument(
        "--force",
        type=core._parse_bool,
        default=False,
    )
    options = parser.parse_args()
    if options.dataset.resolve() != _BASE_ROOT:
        raise ValueError("base dataset changed after audit")
    if options.augmentation_dataset.resolve() != _AUGMENTATION_ROOT:
        raise ValueError("augmentation dataset changed after audit")
    if options.threads != EXPECTED_THREADS:
        raise ValueError("fit4096 stage requires threads=6")
    if options.updates != EXPECTED_UPDATES:
        raise ValueError("fit4096 first stage requires 256 updates")
    return options


def _atomic_save_checkpoint(
    path,
    model,
    optimizer,
    memory,
    update_index,
    source_metadata,
    events,
    force,
    zero_isolated_from_m256=False,
):
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}")
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


core._load_bridge = _fit4096_bridge
core._learning_rate = _fixed_learning_rate
runner._restore_checkpoint = _restore_weights_only
runner._parse_args = _parse_args
runner._save_checkpoint = _atomic_save_checkpoint


if __name__ == "__main__":
    runner.main()
