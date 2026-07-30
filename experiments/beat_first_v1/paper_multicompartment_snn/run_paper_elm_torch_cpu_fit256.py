"""Fit256-safe entry point for the M-variable Paper-ELM CPU runner.

It derives fit/validation membership from the supplied manifest, excludes
every shard containing held-out trials, and fits the four-region normalizer
using fit targets only.  The selected r3/e31 import remains the M1000
parameter source; the dataset-specific normalizer is replaced only for M256
training/benchmark/validation on the expanded fit256 release.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import train_paper_elm_torch_cpu as _core


_core.FAST_MEMORY = 256
_core.FAST_HIDDEN = 512
_core.FULL_MEMORY = 1_000
_core.FULL_HIDDEN = 2_000


def _argument(name: str, default: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return default
    if index + 1 >= len(sys.argv):
        raise ValueError(f"missing value for {name}")
    return sys.argv[index + 1]


def _fit_inventory(root: Path) -> Tuple[dict, Tuple[int, ...], Tuple[int, ...], list]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_ids = tuple(
        int(value)
        for value in manifest["validation_from_train_indices"]
    )
    safe_records = []
    train_ids = set()
    for record in manifest["shards"]:
        split_counts = record.get("split_counts", {})
        heldout = int(split_counts.get("held_out_test", 0))
        train = int(split_counts.get("train", 0))
        if heldout != 0:
            continue
        if train != int(record["samples"]):
            raise ValueError("non-heldout shard is not entirely training")
        safe_records.append(record)
        train_ids.update(
            range(
                int(record["global_first"]),
                int(record["global_last"]) + 1,
            )
        )
    fit_ids = tuple(sorted(train_ids - set(validation_ids)))
    if len(fit_ids) != 256 or len(validation_ids) != 8:
        raise ValueError(
            f"expected fit256/validation8, got "
            f"{len(fit_ids)}/{len(validation_ids)}"
        )
    if set(fit_ids) & set(validation_ids):
        raise ValueError("fit and validation overlap")
    return manifest, fit_ids, validation_ids, safe_records


def _fit_nmda_normalizer(root: Path, fit_ids: Tuple[int, ...], records: list):
    fit_set = set(fit_ids)
    sums = np.zeros((4,), dtype=np.float64)
    sums2 = np.zeros((4,), dtype=np.float64)
    count = 0
    opened = []
    for record in records:
        record_ids = set(
            range(
                int(record["global_first"]),
                int(record["global_last"]) + 1,
            )
        )
        selected = sorted(record_ids & fit_set)
        if not selected:
            continue
        path = (root / str(record["path"])).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != str(record["sha256"]):
            raise ValueError(f"fit shard digest differs: {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            ids = [
                int(value)
                for value in archive["sample_indices"].reshape(-1)
            ]
            target = np.asarray(archive["target_nmda"], np.float32)
        for trial_id in selected:
            item = ids.index(trial_id)
            values = target[:, :, item]
            # Match Julia: a Float64 sum and Float64 sum(abs2) per region.
            for region in range(4):
                region_values = values[region].astype(
                    np.float64,
                    copy=False,
                )
                sums[region] += np.sum(
                    region_values,
                    dtype=np.float64,
                )
                sums2[region] += np.sum(
                    np.square(region_values),
                    dtype=np.float64,
                )
            count += values.shape[1]
        opened.append(str(path))
    expected_count = len(fit_ids) * 1_500
    if count != expected_count:
        raise ValueError(
            f"normalizer observation count {count} != {expected_count}"
        )
    mean64 = sums / count
    variance64 = np.maximum(sums2 / count - np.square(mean64), 0.0)
    mean = mean64.astype(np.float32)
    scale = np.maximum(
        np.sqrt(variance64).astype(np.float32),
        np.float32(1.0e-5),
    )
    if not np.all(np.isfinite(scale)) or not np.all(scale > 0):
        raise ValueError("fit-only NMDA normalizer is invalid")
    return mean, scale, opened


_DATASET = Path(
    _argument("--dataset", str(_core.DEFAULT_DATASET))
).resolve()
_MODE = _argument("--mode", "benchmark")
_MEMORY = int(_argument("--memory", "256"))
_MANIFEST, _FIT_IDS, _VALIDATION_IDS, _SAFE_RECORDS = _fit_inventory(
    _DATASET
)
_MANIFEST_SHA256 = hashlib.sha256(
    (_DATASET / "manifest.json").read_bytes()
).hexdigest()
_NMDA_MEAN, _NMDA_SCALE, _NORMALIZER_OPENED = _fit_nmda_normalizer(
    _DATASET,
    _FIT_IDS,
    _SAFE_RECORDS,
)

_core.FIT_IDS = _FIT_IDS
_core.VALIDATION_IDS = _VALIDATION_IDS


class _ManifestSafeDataset(_core.SafeFitValidationDataset):
    def __init__(self, root: Path, expected_manifest_sha256: str) -> None:
        self.root = root.resolve()
        if self.root != _DATASET:
            raise ValueError("dataset path changed after fit-only audit")
        actual = hashlib.sha256(
            (self.root / "manifest.json").read_bytes()
        ).hexdigest()
        if actual != _MANIFEST_SHA256:
            raise ValueError("dataset manifest changed after fit-only audit")
        self.manifest = _MANIFEST
        self.allowed_ids = frozenset(_FIT_IDS + _VALIDATION_IDS)
        self.records = list(_SAFE_RECORDS)
        covered = {
            trial_id
            for record in self.records
            for trial_id in range(
                int(record["global_first"]),
                int(record["global_last"]) + 1,
            )
        }
        if covered != self.allowed_ids:
            raise ValueError("fit/validation safe inventory differs")
        self.cache: Dict[str, Dict[str, np.ndarray]] = {}
        self.opened_shards = []


_core.SafeFitValidationDataset = _ManifestSafeDataset
_ORIGINAL_LOAD_BRIDGE = _core._load_bridge


def _load_bridge_with_fit256(path: Path):
    bridge, metadata = _ORIGINAL_LOAD_BRIDGE(path)
    if _MEMORY == 256 and _MODE in {"benchmark", "train", "validate"}:
        bridge = {
            name: np.asarray(value)
            for name, value in bridge.items()
        }
        bridge["nmda_mean"] = _NMDA_MEAN.copy()
        bridge["nmda_scale"] = _NMDA_SCALE.copy()
        metadata = dict(metadata)
        metadata["manifest_sha256"] = _MANIFEST_SHA256
        metadata["fit256_normalizer"] = {
            "mean": _NMDA_MEAN.tolist(),
            "scale": _NMDA_SCALE.tolist(),
            "fit_trials": len(_FIT_IDS),
            "heldout_opened": False,
        }
    return bridge, metadata


_core._load_bridge = _load_bridge_with_fit256

import train_paper_elm_torch_cpu_variable as _runner


if __name__ == "__main__":
    _runner.main()
