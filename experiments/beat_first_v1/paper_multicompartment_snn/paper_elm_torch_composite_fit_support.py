"""Collision-safe two-root training support for paper ELM."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from paper_elm_torch_fitn_support import sha256_file


BASE_HANDLE_PREFIX = 1_000_000
AUGMENTATION_HANDLE_PREFIX = 2_000_000


def audit_train_only_manifest(
    manifest: Mapping,
    expected_ids: Sequence[int],
) -> Tuple[Tuple[int, ...], List[dict]]:
    expected_ids = tuple(int(value) for value in expected_ids)
    validation = tuple(
        int(value)
        for value in manifest.get(
            "validation_from_train_indices",
            (),
        )
    )
    if validation:
        raise ValueError("augmentation must not define validation trials")
    records: List[dict] = []
    covered = set()
    for raw_record in manifest["shards"]:
        record = dict(raw_record)
        first = int(record["global_first"])
        last = int(record["global_last"])
        samples = int(record["samples"])
        if last - first + 1 != samples:
            raise ValueError("augmentation shard range differs")
        counts = record.get("split_counts", {})
        heldout = int(counts.get("held_out_test", 0))
        train = int(counts.get("train", 0))
        if heldout != 0:
            raise ValueError("augmentation contains heldout samples")
        if train != samples:
            raise ValueError("augmentation shard is not train-only")
        ids = set(range(first, last + 1))
        if covered & ids:
            raise ValueError("augmentation shard ID ranges overlap")
        covered.update(ids)
        records.append(record)
    if covered != set(expected_ids):
        raise ValueError("augmentation train inventory differs")
    return expected_ids, records


def train_only_inventory(
    root: Path,
    expected_ids: Sequence[int],
) -> Tuple[dict, Tuple[int, ...], List[dict], str]:
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    ids, records = audit_train_only_manifest(
        manifest,
        expected_ids,
    )
    return (
        manifest,
        ids,
        records,
        hashlib.sha256(manifest_bytes).hexdigest(),
    )


def composite_contract_sha256(
    base_manifest_sha256: str,
    augmentation_manifest_sha256: str,
) -> str:
    contract = {
        "schema": "paper_elm_composite_fit.v1",
        "base_manifest_sha256": base_manifest_sha256,
        "augmentation_manifest_sha256": augmentation_manifest_sha256,
        "base_handle_prefix": BASE_HANDLE_PREFIX,
        "augmentation_handle_prefix": AUGMENTATION_HANDLE_PREFIX,
        "validation_source": "base-only",
    }
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fit_nmda_normalizer_many(
    inventories: Sequence[
        Tuple[Path, Sequence[int], Sequence[Mapping]]
    ],
    trial_steps: int = 1_500,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    sums = np.zeros((4,), dtype=np.float64)
    sums2 = np.zeros((4,), dtype=np.float64)
    count = 0
    expected_count = 0
    opened: List[str] = []
    for root, fit_ids, records in inventories:
        fit_set = frozenset(int(value) for value in fit_ids)
        expected_count += len(fit_ids) * trial_steps
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
            if sha256_file(path) != str(record["sha256"]):
                raise ValueError(
                    f"fit shard digest differs: {path.name}"
                )
            with np.load(path, allow_pickle=False) as archive:
                ids = [
                    int(value)
                    for value in archive[
                        "sample_indices"
                    ].reshape(-1)
                ]
                target = np.asarray(
                    archive["target_nmda"],
                    np.float32,
                )
            for trial_id in selected:
                item = ids.index(trial_id)
                values = target[:, :, item]
                if values.shape != (4, trial_steps):
                    raise ValueError(
                        f"target_nmda shape differs: {trial_id}"
                    )
                values64 = values.astype(np.float64, copy=False)
                sums += np.sum(values64, axis=1, dtype=np.float64)
                sums2 += np.sum(
                    np.square(values64),
                    axis=1,
                    dtype=np.float64,
                )
                count += trial_steps
            opened.append(str(path))
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
        raise ValueError("composite fit-only normalizer is invalid")
    return mean, scale, opened


def make_composite_dataset(
    base_dataset_class,
    augmentation_dataset_class,
    base_root: Path,
    augmentation_root: Path,
    base_manifest_sha256: str,
    augmentation_manifest_sha256: str,
    composite_sha256: str,
    base_fit_ids: Sequence[int],
    augmentation_ids: Sequence[int],
):
    base_root = base_root.resolve()
    augmentation_root = augmentation_root.resolve()
    handles: Dict[int, Tuple[str, int]] = {}
    for local_id in base_fit_ids:
        handles[BASE_HANDLE_PREFIX + int(local_id)] = (
            "base",
            int(local_id),
        )
    for local_id in augmentation_ids:
        handles[AUGMENTATION_HANDLE_PREFIX + int(local_id)] = (
            "augmentation",
            int(local_id),
        )

    class CompositeFitDataset:
        handle_map = dict(handles)

        def __init__(
            self,
            root: Path,
            expected_manifest_sha256: str,
        ) -> None:
            if root.resolve() != base_root:
                raise ValueError("composite base root differs")
            if expected_manifest_sha256 != composite_sha256:
                raise ValueError("composite dataset contract differs")
            self.base = base_dataset_class(
                base_root,
                base_manifest_sha256,
            )
            self.augmentation = augmentation_dataset_class(
                augmentation_root,
                augmentation_manifest_sha256,
            )

        @property
        def opened_shards(self):
            return (
                list(self.base.opened_shards)
                + list(self.augmentation.opened_shards)
            )

        def _resolve(self, handle: int):
            try:
                namespace, local_id = self.handle_map[int(handle)]
            except KeyError as error:
                raise PermissionError(
                    f"training handle {handle} is not allowed"
                ) from error
            dataset = (
                self.base
                if namespace == "base"
                else self.augmentation
            )
            return dataset, local_id

        def trial_arrays(self, handle: int):
            dataset, local_id = self._resolve(handle)
            return dataset.trial_arrays(local_id)

        def window(self, handle: int, *args, **kwargs):
            dataset, local_id = self._resolve(handle)
            return dataset.window(local_id, *args, **kwargs)

        def assert_no_heldout_opened(self) -> None:
            self.base.assert_no_heldout_opened()
            self.augmentation.assert_no_heldout_opened()

    fit_handles = tuple(sorted(handles))
    return CompositeFitDataset, fit_handles
