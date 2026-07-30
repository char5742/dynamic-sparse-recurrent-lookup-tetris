"""Manifest-safe helpers for single-root fit-N paper ELM training."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_manifest(
    manifest: Mapping,
    expected_fit_ids: Sequence[int],
    expected_validation_ids: Sequence[int],
    expected_heldout_trials: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], List[dict]]:
    """Audit split membership using manifest metadata without opening shards."""
    fit_ids = tuple(int(value) for value in expected_fit_ids)
    validation_ids = tuple(
        int(value) for value in expected_validation_ids
    )
    manifest_validation = tuple(
        int(value)
        for value in manifest["validation_from_train_indices"]
    )
    if manifest_validation != validation_ids:
        raise ValueError(
            "derived-validation membership differs: "
            f"{manifest_validation} != {validation_ids}"
        )
    expected_allowed = frozenset(fit_ids + validation_ids)
    if len(expected_allowed) != len(fit_ids) + len(validation_ids):
        raise ValueError("expected fit and validation IDs overlap")

    safe_records: List[dict] = []
    safe_ids = set()
    heldout_trials = 0
    for raw_record in manifest["shards"]:
        record = dict(raw_record)
        first = int(record["global_first"])
        last = int(record["global_last"])
        samples = int(record["samples"])
        if last - first + 1 != samples:
            raise ValueError("shard range and sample count differ")
        split_counts = record.get("split_counts", {})
        heldout = int(split_counts.get("held_out_test", 0))
        train = int(split_counts.get("train", 0))
        heldout_trials += heldout
        if heldout:
            if heldout != samples or train != 0:
                raise ValueError(
                    "heldout shard must not mix training samples"
                )
            continue
        if train != samples:
            raise ValueError(
                "non-heldout shard is not entirely training"
            )
        record_ids = set(range(first, last + 1))
        if safe_ids & record_ids:
            raise ValueError("safe shard ID ranges overlap")
        safe_ids.update(record_ids)
        safe_records.append(record)

    if heldout_trials != int(expected_heldout_trials):
        raise ValueError(
            f"expected {expected_heldout_trials} heldout trials, "
            f"got {heldout_trials}"
        )
    if safe_ids != expected_allowed:
        missing = sorted(expected_allowed - safe_ids)
        unexpected = sorted(safe_ids - expected_allowed)
        raise ValueError(
            "fit/validation safe inventory differs: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return fit_ids, validation_ids, safe_records


def fit_inventory(
    root: Path,
    expected_fit_ids: Sequence[int],
    expected_validation_ids: Sequence[int],
    expected_heldout_trials: int,
) -> Tuple[dict, Tuple[int, ...], Tuple[int, ...], List[dict], str]:
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    fit_ids, validation_ids, safe_records = audit_manifest(
        manifest,
        expected_fit_ids,
        expected_validation_ids,
        expected_heldout_trials,
    )
    return (
        manifest,
        fit_ids,
        validation_ids,
        safe_records,
        hashlib.sha256(manifest_bytes).hexdigest(),
    )


def fit_nmda_normalizer(
    root: Path,
    fit_ids: Sequence[int],
    safe_records: Sequence[Mapping],
    trial_steps: int = 1_500,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute the normalizer from fit IDs only; heldout records are absent."""
    fit_set = frozenset(int(value) for value in fit_ids)
    sums = np.zeros((4,), dtype=np.float64)
    sums2 = np.zeros((4,), dtype=np.float64)
    count = 0
    opened: List[str] = []
    for record in safe_records:
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
        if sha256_file(path) != str(record["sha256"]):
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
            if values.shape != (4, trial_steps):
                raise ValueError(
                    f"target_nmda shape differs for trial {trial_id}"
                )
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
    expected_count = len(fit_ids) * trial_steps
    if count != expected_count:
        raise ValueError(
            f"normalizer observation count {count} != {expected_count}"
        )
    mean64 = sums / count
    variance64 = np.maximum(
        sums2 / count - np.square(mean64),
        0.0,
    )
    mean = mean64.astype(np.float32)
    scale = np.maximum(
        np.sqrt(variance64).astype(np.float32),
        np.float32(1.0e-5),
    )
    if not np.all(np.isfinite(scale)) or not np.all(scale > 0):
        raise ValueError("fit-only NMDA normalizer is invalid")
    return mean, scale, opened


def make_manifest_safe_dataset(
    core,
    audited_root: Path,
    manifest: dict,
    manifest_sha256: str,
    fit_ids: Sequence[int],
    validation_ids: Sequence[int],
    safe_records: Sequence[Mapping],
):
    allowed_ids = frozenset(
        int(value) for value in tuple(fit_ids) + tuple(validation_ids)
    )
    audited_root = audited_root.resolve()
    safe_records = [dict(record) for record in safe_records]

    class ManifestSafeFitNDataset(core.SafeFitValidationDataset):
        def __init__(
            self,
            root: Path,
            expected_manifest_sha256: str,
        ) -> None:
            self.root = root.resolve()
            if self.root != audited_root:
                raise ValueError("dataset path changed after fit-only audit")
            actual = hashlib.sha256(
                (self.root / "manifest.json").read_bytes()
            ).hexdigest()
            if actual != manifest_sha256:
                raise ValueError(
                    "dataset manifest changed after fit-only audit"
                )
            if expected_manifest_sha256 != manifest_sha256:
                raise ValueError(
                    "checkpoint/dataset manifest digest differs"
                )
            self.manifest = manifest
            self.allowed_ids = allowed_ids
            self.records = list(safe_records)
            self.cache: Dict[str, Dict[str, np.ndarray]] = {}
            self.opened_shards: List[str] = []

    return ManifestSafeFitNDataset


def install_positive_balanced_objective(core) -> None:
    """Install the same positive-aware balanced composite used by fit256."""
    original_materialize_batch = core._materialize_batch
    positive_catalog: Dict[int, List[int]] = {}

    def catalog(dataset) -> Dict[int, List[int]]:
        if positive_catalog:
            return positive_catalog
        for trial_id in core.FIT_IDS:
            trial_id = int(trial_id)
            data, item = dataset.trial_arrays(trial_id)
            spike = data["target_spike"][:, item]
            prefix = spike.astype("int64").cumsum()
            prefix = core.np.concatenate(
                (core.np.zeros((1,), dtype=core.np.int64), prefix)
            )
            starts = []
            for start_julia in range(
                core.FIRST_RANDOM_START_JULIA,
                core.LAST_RANDOM_START_JULIA + 1,
            ):
                first = start_julia - 1
                last = first + core.TRAIN_WINDOW
                if int(prefix[last] - prefix[first]) > 0:
                    starts.append(start_julia)
            positive_catalog[trial_id] = starts
        if not any(positive_catalog.values()):
            raise ValueError("fit split has no spike-positive window")
        return positive_catalog

    def positive_materialize_batch(
        dataset,
        ids,
        starts_julia,
    ):
        batch = original_materialize_batch(
            dataset,
            ids,
            starts_julia,
        )
        if float(batch[2].sum()) > 0:
            return batch
        available = catalog(dataset)
        selected_ids = [int(value) for value in ids]
        selected_starts = [int(value) for value in starts_julia]
        eligible_slots = [
            slot
            for slot, trial_id in enumerate(selected_ids)
            if available[trial_id]
        ]
        if eligible_slots:
            slot = eligible_slots[0]
            selected_starts[slot] = available[
                selected_ids[slot]
            ][0]
        else:
            replacements = [
                int(trial_id)
                for trial_id in core.FIT_IDS
                if available[int(trial_id)]
                and int(trial_id) not in selected_ids
            ]
            if not replacements:
                raise ValueError(
                    "cannot choose a distinct positive anchor"
                )
            selected_ids[0] = replacements[0]
            selected_starts[0] = available[selected_ids[0]][0]
        batch = original_materialize_batch(
            dataset,
            selected_ids,
            selected_starts,
        )
        if float(batch[2].sum()) <= 0:
            raise AssertionError("positive-aware batch has no spike")
        return batch

    def balanced_objective(model, batch):
        inputs, target_voltage, target_spike, target_nmda = batch
        raw = model(inputs)
        spike_logit = raw[:, :, 0]
        voltage = raw[:, :, 1]
        nmda = raw[:, :, 2:]
        target_voltage_coordinate = (
            torch.minimum(
                target_voltage,
                target_voltage.new_tensor(core.SOMA_CLIP_MV),
            )
            - core.SOMA_BIAS_MV
        ) * core.SOMA_TRAIN_SCALE
        voltage_mse = torch.mean(
            torch.square(voltage - target_voltage_coordinate)
        )
        element_bce = F.binary_cross_entropy_with_logits(
            spike_logit,
            target_spike,
            reduction="none",
        )
        positives = torch.sum(target_spike)
        negatives = target_spike.numel() - positives
        if float(positives) <= 0 or float(negatives) <= 0:
            raise ValueError("balanced spike batch lacks one class")
        positive_bce = torch.sum(
            element_bce * target_spike
        ) / positives
        negative_bce = torch.sum(
            element_bce * (1.0 - target_spike)
        ) / negatives
        spike_balanced_bce = (
            0.5 * positive_bce + 0.5 * negative_bce
        )
        target_nmda_coordinate = (
            target_nmda
            - model.nmda_mean.view(1, 1, 4)
        ) / model.nmda_scale.view(1, 1, 4)
        nmda_extension_loss = torch.mean(
            torch.square(nmda - target_nmda_coordinate)
        )
        paper_loss = (
            0.5 * voltage_mse
            + 0.5 * spike_balanced_bce
        )
        total = paper_loss + nmda_extension_loss
        return total, {
            "total": total,
            "paper_loss": paper_loss,
            "voltage_mse": voltage_mse,
            "spike_balanced_bce": spike_balanced_bce,
            "positive_bce": positive_bce,
            "negative_bce": negative_bce,
            "nmda_extension_loss": nmda_extension_loss,
            "spike_positives": positives,
            "spike_negatives": negatives,
        }

    core._materialize_batch = positive_materialize_batch
    core._objective = balanced_objective
