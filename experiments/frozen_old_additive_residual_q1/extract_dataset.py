from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


DATASET_SHA256 = "4f10cfcf545c97eb3f56e8511921a1a6b50fa5ab166fac2eb3575eacf84b71ba"
ACTIONS = 74
GAMMA = np.float32(0.997)
TRAIN_START, TRAIN_STOP = 1, 2160
OFFLINE_START, OFFLINE_STOP = 2161, 2660
EXPECTED_TRAIN_ELIGIBLE = 2124
EXPECTED_OFFLINE_ELIGIBLE = 494


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, value: object) -> None:
    if path.exists() or path.with_suffix(path.suffix + ".tmp").exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def require_dataset(path: Path) -> None:
    observed = sha256_file(path)
    if observed != DATASET_SHA256:
        raise RuntimeError(f"dataset SHA-256 mismatch: {observed}")


def terminal_values(file: h5py.File, rows_zero: np.ndarray) -> np.ndarray:
    dataset = file["terminal"]
    if isinstance(dataset, h5py.Dataset) and dataset.dtype.names and "chunks" in dataset.dtype.names:
        descriptor = dataset[()]
        if int(descriptor["len"]) != OFFLINE_STOP:
            raise RuntimeError("terminal BitVector length mismatch")
        chunks = file[descriptor["chunks"]]
        indices = np.unique(rows_zero // 64)
        packed = {int(index): int(value) for index, value in zip(indices, chunks[indices], strict=True)}
        return np.asarray(
            [bool((packed[int(row // 64)] >> int(row % 64)) & 1) for row in rows_zero],
            dtype=np.bool_,
        )
    return np.asarray(dataset[rows_zero], dtype=np.bool_)


def role_contract(
    file: h5py.File, start_one: int, stop_one: int, expected_episodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(file["episode_ids"])
    if count != OFFLINE_STOP:
        raise RuntimeError(f"aggregate row count mismatch: {count}")
    rows_zero = np.arange(start_one - 1, stop_one, dtype=np.int64)
    episodes = np.asarray(file["episode_ids"][rows_zero], dtype=np.int32)
    steps = np.asarray(file["episode_steps"][rows_zero], dtype=np.int32)
    terminals = terminal_values(file, rows_zero)
    if not np.array_equal(np.unique(episodes), expected_episodes):
        raise RuntimeError(f"rows {start_one}:{stop_one} episode role mismatch")
    if start_one == TRAIN_START:
        if not np.array_equal(np.unique(episodes[:1500]), np.arange(1, 7)):
            raise RuntimeError("aggregate rows 1:1500 are not exactly base episodes 1:6")
        if not np.array_equal(np.unique(episodes[1500:]), np.arange(7, 13)):
            raise RuntimeError("aggregate rows 1501:2160 are not exactly DAgger episodes 7:12")
    action_counts = np.asarray(file["action_counts"][rows_zero], dtype=np.int32)
    if int(action_counts.max()) != ACTIONS or int(action_counts.min()) < 1:
        raise RuntimeError("effective candidate dimension is not exactly 74")
    if len(file["teacher_q"].shape) != 2 or file["teacher_q"].shape[1] < ACTIONS:
        raise RuntimeError("physical candidate dimension is smaller than 74")
    return episodes, steps, terminals


def eligible_rows(
    episodes: np.ndarray,
    steps: np.ndarray,
    terminals: np.ndarray,
    start_one: int,
    stop_one: int,
) -> list[int]:
    result: list[int] = []
    for row_one in range(start_one, stop_one - 2):
        row = row_one - 1
        successor = slice(row, row + 4)
        if not np.all(episodes[successor] == episodes[row]):
            continue
        if not np.array_equal(steps[successor], np.arange(steps[row], steps[row] + 4)):
            continue
        if np.any(terminals[row : row + 3]):
            continue
        result.append(row_one)
    return result


def eligibility(dataset: Path, output: Path) -> None:
    require_dataset(dataset)
    with h5py.File(dataset, "r") as file:
        episodes, steps, terminals = role_contract(
            file, TRAIN_START, TRAIN_STOP, np.arange(1, 13)
        )
        # Arrays are role-local here, so the eligibility indices are local and
        # numerically identical to the one-based aggregate training rows.
        training = eligible_rows(episodes, steps, terminals, TRAIN_START, TRAIN_STOP)
    if len(training) != EXPECTED_TRAIN_ELIGIBLE:
        raise RuntimeError(f"training eligibility mismatch: {len(training)}")
    write_json_new(
        output,
        {
            "status": "eligibility_complete",
            "dataset_sha256": DATASET_SHA256,
            "training_rows_loaded": [TRAIN_START, TRAIN_STOP],
            "offline_rows_loaded": False,
            "training_episode_ids": list(range(1, 13)),
            "criterion": "same episode and consecutive steps t:t+3; no terminal at t:t+2",
            "training_eligible_rows": training,
            "training_eligible_count": len(training),
            "offline_eligibility_deferred_until_after_candidate_freeze": True,
            "validation_or_test_seed_loaded": False,
            "game_seed_loaded": False,
        },
    )


def scalar_at(file: h5py.File, key: str, row: int) -> object:
    dataset = file[key]
    if len(dataset.shape) == 1:
        return dataset[row]
    return dataset[row, 0]


def candidate_row(file: h5py.File, key: str, row: int, count: int) -> np.ndarray:
    dataset = file[key]
    # JLD2/HDF5 arrays are exposed to h5py with reversed Julia dimensions.
    return np.asarray(dataset[row, :count])


def allocate_role(file: h5py.File, rows_one: list[int]) -> dict[str, np.ndarray]:
    size = len(rows_one)
    arrays: dict[str, np.ndarray] = {
        "source_rows": np.asarray(rows_one, dtype=np.int32),
        "episode_ids": np.empty(size, dtype=np.int32),
        "episode_steps": np.empty(size, dtype=np.int32),
        "action_counts": np.empty(size, dtype=np.int16),
        "selected_actions": np.empty(size, dtype=np.int16),
        "boards": np.empty((size, 24, 10, 1), dtype=np.uint8),
        "placements": np.zeros((size, ACTIONS, 24, 10, 1), dtype=np.uint8),
        "ren": np.empty(size, dtype=np.float32),
        "back_to_back": np.empty(size, dtype=np.float32),
        "tspin": np.zeros((size, ACTIONS), dtype=np.float32),
        "queues": np.empty((size, 7, 6), dtype=np.uint8),
        "stored_q": np.full((size, ACTIONS), np.nan, dtype=np.float32),
        "targets": np.full(size, np.nan, dtype=np.float32),
        "target_residual": np.full(size, np.nan, dtype=np.float32),
        "target_valid": np.zeros(size, dtype=np.bool_),
        "behavior_is_old_argmax": np.zeros(size, dtype=np.bool_),
        "data_role": np.zeros(size, dtype=np.int8),
    }
    rows_zero = np.arange(rows_one[0] - 1, rows_one[-1], dtype=np.int64)
    episodes = np.asarray(file["episode_ids"][rows_zero], dtype=np.int32)
    steps = np.asarray(file["episode_steps"][rows_zero], dtype=np.int32)
    terminals = terminal_values(file, rows_zero)
    eligible_local = eligible_rows(episodes, steps, terminals, 1, len(rows_one))
    eligible = {rows_one[0] + local - 1 for local in eligible_local}
    for slot, row_one in enumerate(rows_one):
        row = row_one - 1
        count = int(scalar_at(file, "action_counts", row))
        selected = int(scalar_at(file, "selected_actions", row))
        if not 1 <= count <= ACTIONS or not 1 <= selected <= count:
            raise RuntimeError(f"invalid action metadata at row {row_one}")
        arrays["episode_ids"][slot] = int(scalar_at(file, "episode_ids", row))
        arrays["episode_steps"][slot] = int(scalar_at(file, "episode_steps", row))
        arrays["action_counts"][slot] = count
        arrays["selected_actions"][slot] = selected
        arrays["boards"][slot] = np.transpose(file["boards"][row], (2, 1, 0))
        placements = np.asarray(file["placements"][row, :count])
        arrays["placements"][slot, :count] = np.transpose(placements, (0, 3, 2, 1))
        arrays["ren"][slot] = float(scalar_at(file, "ren", row))
        arrays["back_to_back"][slot] = float(scalar_at(file, "back_to_back", row))
        arrays["tspin"][slot, :count] = candidate_row(file, "tspin", row, count)
        arrays["queues"][slot] = np.transpose(file["queues"][row], (1, 0))
        old_q = candidate_row(file, "teacher_q", row, count).astype(np.float32)
        if not np.all(np.isfinite(old_q)):
            raise RuntimeError(f"non-finite valid old-Q at row {row_one}")
        arrays["stored_q"][slot, :count] = old_q
        arrays["behavior_is_old_argmax"][slot] = selected == int(np.argmax(old_q)) + 1
        arrays["data_role"][slot] = 1 if row_one <= 1500 else (2 if row_one <= 2160 else 3)
        if row_one not in eligible:
            continue
        rewards = np.asarray(file["rewards"][row : row + 3], dtype=np.float32).reshape(-1)
        for offset, reward in enumerate(rewards):
            reward_row = row + offset
            score_after = int(scalar_at(file, "scores_after", reward_row))
            score_before = 0 if int(scalar_at(file, "episode_steps", reward_row)) == 1 else int(
                scalar_at(file, "scores_after", reward_row - 1)
            )
            expected = np.float32(score_after - score_before) / np.float32(600.0)
            if np.asarray(reward, dtype=np.float32).view(np.uint32) != np.asarray(expected).view(np.uint32):
                raise RuntimeError(f"stored reward mismatch at row {reward_row + 1}")
        bootstrap_row = row + 3
        bootstrap_count = int(scalar_at(file, "action_counts", bootstrap_row))
        bootstrap_q = candidate_row(file, "teacher_q", bootstrap_row, bootstrap_count).astype(np.float32)
        if not np.all(np.isfinite(bootstrap_q)):
            raise RuntimeError("non-finite bootstrap old-Q")
        # The DAgger behavior action is intentionally ignored here. The target
        # bootstraps from the raw stored old-Q maximum over valid actions.
        target = np.float32(rewards[0] + GAMMA * rewards[1] + GAMMA**2 * rewards[2])
        target = np.float32(target + GAMMA**3 * np.max(bootstrap_q))
        arrays["targets"][slot] = target
        arrays["target_residual"][slot] = np.float32(target - old_q[selected - 1])
        arrays["target_valid"][slot] = True
    if rows_one[0] == TRAIN_START:
        if not np.all(arrays["behavior_is_old_argmax"][:1500]):
            raise RuntimeError("base episodes contain a selected action that is not old-Q argmax")
        if np.all(arrays["behavior_is_old_argmax"][1500:]):
            raise RuntimeError("DAgger role contains no selected action differing from old-Q argmax")
    else:
        if not np.all(arrays["behavior_is_old_argmax"]):
            raise RuntimeError("offline guard contains a selected action that is not old-Q argmax")
    return arrays


def extract_role(dataset: Path, mode: str, output_npz: Path, output_json: Path) -> None:
    require_dataset(dataset)
    if output_npz.exists() or output_json.exists():
        raise RuntimeError("refusing to overwrite role extraction")
    rows = list(range(TRAIN_START, TRAIN_STOP + 1)) if mode == "training" else list(
        range(OFFLINE_START, OFFLINE_STOP + 1)
    )
    with h5py.File(dataset, "r") as file:
        if mode == "training":
            role_contract(file, TRAIN_START, TRAIN_STOP, np.arange(1, 13))
        else:
            role_contract(file, OFFLINE_START, OFFLINE_STOP, np.asarray([13, 14]))
        arrays = allocate_role(file, rows)
        valid_count = int(np.count_nonzero(arrays["target_valid"]))
        expected = EXPECTED_TRAIN_ELIGIBLE if mode == "training" else EXPECTED_OFFLINE_ELIGIBLE
        if valid_count != expected:
            raise RuntimeError(f"{mode} target eligibility mismatch: {valid_count}")
        np.savez_compressed(output_npz, **arrays)
    valid = arrays["target_valid"]
    selected_zero = arrays["selected_actions"].astype(np.int64) - 1
    row_indices = np.arange(len(rows), dtype=np.int64)
    selected_old_q = arrays["stored_q"][row_indices, selected_zero]
    def array_digest(value: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    write_json_new(
        output_json,
        {
            "status": f"{mode}_extraction_complete",
            "dataset_sha256": DATASET_SHA256,
            "rows_loaded": [rows[0], rows[-1]],
            "row_count": len(rows),
            "episodes": sorted(set(int(value) for value in arrays["episode_ids"])),
            "target_eligible_count": int(np.count_nonzero(arrays["target_valid"])),
            "behavior_old_argmax_fraction": float(np.mean(arrays["behavior_is_old_argmax"])),
            "bootstrap_source": "max(raw stored old-Q[1:action_count] at t+3)",
            "dagger_behavior_bootstrap_used": False,
            "initializer_exposed_to_offline_rows": True,
            "offline_role": "reused_development_guard",
            "offline_loaded_before_candidate_freeze": False,
            "eligible_rows_sha256": array_digest(arrays["source_rows"][valid]),
            "targets_sha256": array_digest(arrays["targets"][valid]),
            "selected_old_q_sha256": array_digest(selected_old_q[valid]),
            "target_residual_sha256": array_digest(arrays["target_residual"][valid]),
            "validation_or_test_seed_loaded": False,
            "game_seed_loaded": False,
            "npz_path": str(output_npz.resolve()),
            "npz_sha256": sha256_file(output_npz),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    eligible = subparsers.add_parser("eligibility")
    eligible.add_argument("dataset", type=Path)
    eligible.add_argument("output_json", type=Path)
    for mode in ("training", "offline"):
        command = subparsers.add_parser(mode)
        command.add_argument("dataset", type=Path)
        command.add_argument("output_npz", type=Path)
        command.add_argument("output_json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "eligibility":
        eligibility(args.dataset, args.output_json)
    else:
        extract_role(args.dataset, args.mode, args.output_npz, args.output_json)


if __name__ == "__main__":
    main()
