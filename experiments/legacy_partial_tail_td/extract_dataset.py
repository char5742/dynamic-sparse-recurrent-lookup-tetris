from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


DATASET_SHA256 = "e0d79e38daebb667bd8c248f5f64b8e5241a4ed56a29d31ffb4ee41bd0c26b8d"
GAMMA = np.float32(0.997)
N_STEP = 3
TRAIN_STOP = 1500
OFFLINE_START = 1501
OFFLINE_STOP = 2000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_dataset(path: Path) -> None:
    observed = sha256_file(path)
    if observed != DATASET_SHA256:
        raise RuntimeError(f"dataset SHA-256 mismatch: {observed}")


def write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def metadata_contract(file: h5py.File) -> None:
    metadata = file["metadata"][()]
    if int(metadata["seed_first"]) != 5742 or int(metadata["episodes"]) != 8:
        raise RuntimeError("dataset seed/episode metadata mismatch")
    if int(metadata["max_steps"]) != 250 or int(metadata["row_count"]) != 2000:
        raise RuntimeError("dataset row metadata mismatch")
    if bool(metadata["held_out_test_seeds_used"]):
        raise RuntimeError("dataset claims held-out test seed use")


def terminal_values(file: h5py.File, rows_zero: np.ndarray) -> np.ndarray:
    descriptor = file["terminal"][()]
    if int(descriptor["len"]) != 2000:
        raise RuntimeError("terminal BitVector length mismatch")
    chunks = file[descriptor["chunks"]]
    indices = np.unique(rows_zero // 64)
    packed = {int(index): int(value) for index, value in zip(indices, chunks[indices], strict=True)}
    return np.asarray(
        [bool((packed[int(row // 64)] >> int(row % 64)) & 1) for row in rows_zero],
        dtype=np.bool_,
    )


def eligibility(dataset: Path, output: Path) -> None:
    require_dataset(dataset)
    with h5py.File(dataset, "r") as file:
        metadata_contract(file)
        rows = np.arange(TRAIN_STOP, dtype=np.int64)
        episode_ids = np.asarray(file["episode_ids"][rows], dtype=np.int32)
        episode_steps = np.asarray(file["episode_steps"][rows], dtype=np.int16)
        terminal = terminal_values(file, rows)
        eligible: list[int] = []
        for row in range(TRAIN_STOP - N_STEP):
            episode = int(episode_ids[row])
            if not 1 <= episode <= 6:
                continue
            successors = slice(row, row + N_STEP + 1)
            if not np.all(episode_ids[successors] == episode):
                continue
            if not np.array_equal(
                episode_steps[successors],
                np.arange(int(episode_steps[row]), int(episode_steps[row]) + N_STEP + 1),
            ):
                continue
            if np.any(terminal[row : row + N_STEP]):
                continue
            eligible.append(row + 1)
    write_json_new(
        output,
        {
            "status": "training_eligibility_complete",
            "dataset_sha256": DATASET_SHA256,
            "rows_loaded": [1, TRAIN_STOP],
            "episode_ids_allowed": [1, 2, 3, 4, 5, 6],
            "seeds_allowed": [5742, 5743, 5744, 5745, 5746, 5747],
            "criterion": "same episode and consecutive step for t through t+3; no terminal in t through t+2",
            "eligible_rows": eligible,
            "eligible_count": len(eligible),
            "offline_rows_loaded": False,
            "validation_or_test_seed_loaded": False,
        },
    )


def read_row_freeze(path: Path) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [int(value) for value in data["ordered_rows"]]
    if data.get("rng") != "Xoshiro(0x1313_2026)" or len(rows) != 300:
        raise RuntimeError("invalid row freeze contract")
    if len(set(rows)) != 300 or not all(1 <= row <= TRAIN_STOP for row in rows):
        raise RuntimeError("row freeze escaped fixed training role")
    return rows


def allocate_rows(file: h5py.File, rows_one: list[int]) -> dict[str, np.ndarray]:
    rows_zero = np.asarray(rows_one, dtype=np.int64) - 1
    counts = np.asarray([file["action_counts"][row] for row in rows_zero], dtype=np.int16)
    selected = np.asarray([file["selected_actions"][row] for row in rows_zero], dtype=np.int16)
    max_count = int(np.max(counts))
    size = len(rows_one)
    arrays: dict[str, np.ndarray] = {
        "source_rows": np.asarray(rows_one, dtype=np.int32),
        "episode_ids": np.asarray([file["episode_ids"][row] for row in rows_zero], dtype=np.int32),
        "episode_steps": np.asarray([file["episode_steps"][row] for row in rows_zero], dtype=np.int16),
        "action_counts": counts,
        "selected_actions": selected,
        "boards": np.empty((size, 24, 10, 1), dtype=np.uint8),
        "placements": np.zeros((size, max_count, 24, 10, 1), dtype=np.uint8),
        "ren": np.empty(size, dtype=np.float32),
        "back_to_back": np.empty(size, dtype=np.float32),
        "tspin": np.zeros((size, max_count), dtype=np.float32),
        "queues": np.empty((size, 7, 6), dtype=np.uint8),
        "stored_q": np.full((size, max_count), np.nan, dtype=np.float32),
    }
    for slot, (row, count, selected_action) in enumerate(zip(rows_zero, counts, selected, strict=True)):
        count_int = int(count)
        if not 1 <= count_int <= 128 or not 1 <= int(selected_action) <= count_int:
            raise RuntimeError(f"invalid action metadata at row {row + 1}")
        arrays["boards"][slot] = np.transpose(file["boards"][row], (2, 1, 0))
        placements = np.asarray(file["placements"][row, :count_int])
        arrays["placements"][slot, :count_int] = np.transpose(placements, (0, 3, 2, 1))
        arrays["ren"][slot] = file["ren"][row, 0]
        arrays["back_to_back"][slot] = file["back_to_back"][row, 0]
        arrays["tspin"][slot, :count_int] = file["tspin"][row, :count_int]
        arrays["queues"][slot] = np.transpose(file["queues"][row], (1, 0))
        q = np.asarray(file["teacher_q"][row, :count_int], dtype=np.float32)
        if not np.all(np.isfinite(q)) or int(np.argmax(q)) + 1 != int(selected_action):
            raise RuntimeError(f"stored old-Q/action mismatch at row {row + 1}")
        arrays["stored_q"][slot, :count_int] = q
    return arrays


def extract_training(dataset: Path, row_freeze: Path, output_npz: Path, output_json: Path) -> None:
    require_dataset(dataset)
    if output_npz.exists() or output_json.exists():
        raise RuntimeError("refusing to overwrite training extraction")
    rows = read_row_freeze(row_freeze)
    with h5py.File(dataset, "r") as file:
        metadata_contract(file)
        arrays = allocate_rows(file, rows)
        episode_ids = arrays["episode_ids"]
        if not np.all((episode_ids >= 1) & (episode_ids <= 6)):
            raise RuntimeError("training extraction escaped episodes 1--6")
        targets = np.empty(len(rows), dtype=np.float32)
        target_rewards = np.empty((len(rows), N_STEP), dtype=np.float32)
        terminal_mask = np.empty((len(rows), N_STEP), dtype=np.bool_)
        bootstrap_rows = np.empty(len(rows), dtype=np.int32)
        bootstrap_actions = np.empty(len(rows), dtype=np.int16)
        bootstrap_q = np.empty(len(rows), dtype=np.float32)
        for slot, row_one in enumerate(rows):
            row = row_one - 1
            successor_rows = np.arange(row, row + N_STEP + 1, dtype=np.int64)
            successor_episodes = np.asarray(file["episode_ids"][successor_rows], dtype=np.int32)
            if not np.all(successor_episodes == episode_ids[slot]):
                raise RuntimeError(f"row {row_one} lacks three same-trajectory successors")
            rewards = np.asarray(file["rewards"][successor_rows[:N_STEP]], dtype=np.float32)
            terminals = terminal_values(file, successor_rows[:N_STEP])
            for reward_row, reward in zip(successor_rows[:N_STEP], rewards, strict=True):
                step = int(file["episode_steps"][reward_row])
                score_after = int(file["scores_after"][reward_row])
                if step == 1:
                    score_before = 0
                else:
                    if int(file["episode_ids"][reward_row - 1]) != int(file["episode_ids"][reward_row]):
                        raise RuntimeError("reward predecessor crossed an episode boundary")
                    score_before = int(file["scores_after"][reward_row - 1])
                expected_reward = np.float32(score_after - score_before) / np.float32(600.0)
                if reward != expected_reward:
                    raise RuntimeError(f"stored reward is not score delta / 600 at row {reward_row + 1}")
            target_rewards[slot] = rewards
            terminal_mask[slot] = terminals
            factor = np.float32(1.0)
            target = np.float32(0.0)
            terminated = False
            for reward, terminal in zip(rewards, terminals, strict=True):
                if not np.isfinite(reward):
                    raise RuntimeError("non-finite reward")
                target = np.float32(target + factor * reward)
                if terminal:
                    terminated = True
                    break
                factor = np.float32(factor * GAMMA)
            bootstrap_row = row + N_STEP
            bootstrap_rows[slot] = bootstrap_row + 1
            bootstrap_count = int(file["action_counts"][bootstrap_row])
            bootstrap_action = int(file["selected_actions"][bootstrap_row])
            values = np.asarray(file["teacher_q"][bootstrap_row, :bootstrap_count], dtype=np.float32)
            if bootstrap_action != int(np.argmax(values)) + 1:
                raise RuntimeError("bootstrap selected action is not stable old-policy argmax")
            bootstrap_actions[slot] = bootstrap_action
            bootstrap_q[slot] = values[bootstrap_action - 1]
            if not terminated:
                target = np.float32(target + factor * bootstrap_q[slot])
            targets[slot] = target
        arrays.update(
            {
                "targets": targets,
                "target_rewards": target_rewards,
                "terminal_mask": terminal_mask,
                "bootstrap_rows": bootstrap_rows,
                "bootstrap_actions": bootstrap_actions,
                "bootstrap_q": bootstrap_q,
            }
        )
        np.savez_compressed(output_npz, **arrays)
    write_json_new(
        output_json,
        {
            "status": "training_extraction_complete",
            "dataset_sha256": DATASET_SHA256,
            "row_freeze_sha256": sha256_file(row_freeze),
            "row_count": 300,
            "source_rows": rows,
            "episodes": sorted(set(int(value) for value in arrays["episode_ids"])),
            "seeds": [5742, 5743, 5744, 5745, 5746, 5747],
            "targets_frozen_before_training": True,
            "updated_model_used_for_targets": False,
            "offline_rows_loaded": False,
            "validation_or_test_seed_loaded": False,
            "npz_sha256": sha256_file(output_npz),
        },
    )


def extract_offline(dataset: Path, output_npz: Path, output_json: Path) -> None:
    require_dataset(dataset)
    if output_npz.exists() or output_json.exists():
        raise RuntimeError("refusing to overwrite offline extraction")
    rows = list(range(OFFLINE_START, OFFLINE_STOP + 1))
    with h5py.File(dataset, "r") as file:
        metadata_contract(file)
        arrays = allocate_rows(file, rows)
        if not np.array_equal(np.unique(arrays["episode_ids"]), np.asarray([7, 8])):
            raise RuntimeError("offline extraction is not exactly episodes 7--8")
        np.savez_compressed(output_npz, **arrays)
    write_json_new(
        output_json,
        {
            "status": "offline_extraction_complete",
            "dataset_sha256": DATASET_SHA256,
            "rows_loaded": [OFFLINE_START, OFFLINE_STOP],
            "row_count": 500,
            "episodes": [7, 8],
            "seeds": [5748, 5749],
            "training_rows_loaded": False,
            "validation_or_test_seed_loaded": False,
            "npz_sha256": sha256_file(output_npz),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    eligible = subparsers.add_parser("eligibility")
    eligible.add_argument("dataset", type=Path)
    eligible.add_argument("output_json", type=Path)
    training = subparsers.add_parser("training")
    training.add_argument("dataset", type=Path)
    training.add_argument("row_freeze", type=Path)
    training.add_argument("output_npz", type=Path)
    training.add_argument("output_json", type=Path)
    offline = subparsers.add_parser("offline")
    offline.add_argument("dataset", type=Path)
    offline.add_argument("output_npz", type=Path)
    offline.add_argument("output_json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "eligibility":
        eligibility(args.dataset, args.output_json)
    elif args.mode == "training":
        extract_training(args.dataset, args.row_freeze, args.output_npz, args.output_json)
    else:
        extract_offline(args.dataset, args.output_npz, args.output_json)


if __name__ == "__main__":
    main()
