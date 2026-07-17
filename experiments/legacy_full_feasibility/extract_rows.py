from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


DATASET_SHA256 = "e0d79e38daebb667bd8c248f5f64b8e5241a4ed56a29d31ffb4ee41bd0c26b8d"
SOURCE_ROWS_1 = (1, 251, 501, 751, 1001, 1251)
EXPECTED_EPISODES = (1, 2, 3, 4, 5, 6)
EXPECTED_SEEDS = (5742, 5743, 5744, 5745, 5746, 5747)
EXPECTED_TARGETS = np.asarray(
    [5.6397543, 5.567213, 5.616358, 5.702811, 5.58614, 5.53066],
    dtype=np.float32,
)
N_STEP = 3
GAMMA = np.float32(0.997)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()

    if args.output_npz.exists() or args.output_json.exists():
        raise RuntimeError("refusing to overwrite an existing F subset artifact")
    observed_hash = sha256_file(args.dataset)
    if observed_hash != DATASET_SHA256:
        raise RuntimeError(f"dataset SHA-256 mismatch: {observed_hash}")

    base_rows = np.asarray(SOURCE_ROWS_1, dtype=np.int64) - 1
    required_rows = np.asarray(
        sorted({int(row + offset) for row in base_rows for offset in range(N_STEP + 1)}),
        dtype=np.int64,
    )
    # This is an auditable hyperslab whitelist.  No full JLD2 array is loaded,
    # and no row outside the six training trajectories' steps 1--4 is touched.
    if len(required_rows) != 24 or np.any(required_rows >= 1500):
        raise RuntimeError("row whitelist escaped the preregistered training range")

    with h5py.File(args.dataset, "r") as file:
        metadata = file["metadata"][()]
        seed_first = int(metadata["seed_first"])
        if seed_first != 5742 or int(metadata["episodes"]) != 8:
            raise RuntimeError("dataset metadata seed/episode contract mismatch")
        if int(metadata["max_steps"]) != 250 or int(metadata["row_count"]) != 2000:
            raise RuntimeError("dataset metadata shape contract mismatch")
        if bool(metadata["held_out_test_seeds_used"]):
            raise RuntimeError("dataset claims held-out test seed use")

        episode_ids = np.asarray(file["episode_ids"][required_rows], dtype=np.int32)
        episode_steps = np.asarray(file["episode_steps"][required_rows], dtype=np.int16)
        lookup = {int(row): index for index, row in enumerate(required_rows)}
        for episode, base_row in zip(EXPECTED_EPISODES, base_rows, strict=True):
            for offset in range(N_STEP + 1):
                position = lookup[int(base_row + offset)]
                if int(episode_ids[position]) != episode or int(episode_steps[position]) != offset + 1:
                    raise RuntimeError("episode/step continuity mismatch in n-step whitelist")

        counts = np.asarray(file["action_counts"][base_rows], dtype=np.int16)
        selected = np.asarray(file["selected_actions"][base_rows], dtype=np.int16)
        if tuple(np.asarray(file["episode_ids"][base_rows], dtype=int)) != EXPECTED_EPISODES:
            raise RuntimeError("base row episode IDs are not exactly 1--6")
        if tuple(seed_first + np.asarray(EXPECTED_EPISODES) - 1) != EXPECTED_SEEDS:
            raise RuntimeError("base row seeds are not exactly 5742--5747")
        if np.any(counts <= 0) or np.any(counts > 128):
            raise RuntimeError("invalid candidate count")
        if np.any(selected <= 0) or np.any(selected > counts):
            raise RuntimeError("invalid selected action")

        max_count = int(np.max(counts))
        boards = np.empty((6, 24, 10, 1), dtype=np.float32)
        placements = np.zeros((6, max_count, 24, 10, 1), dtype=np.float32)
        ren = np.empty((6,), dtype=np.float32)
        back_to_back = np.empty((6,), dtype=np.float32)
        tspin = np.zeros((6, max_count), dtype=np.float32)
        queues = np.empty((6, 7, 6), dtype=np.float32)
        stored_q = np.full((6, max_count), np.nan, dtype=np.float32)
        targets = np.empty((6,), dtype=np.float32)
        target_rewards = np.empty((6, N_STEP), dtype=np.float32)
        bootstrap_rows_1 = np.empty((6,), dtype=np.int32)
        bootstrap_actions = np.empty((6,), dtype=np.int16)
        bootstrap_q = np.empty((6,), dtype=np.float32)

        for slot, base_row in enumerate(base_rows):
            count = int(counts[slot])
            # h5py exposes reversed Julia dimensions.  Transpose each selected
            # hyperslab back into the Lux (H,W,C,B) convention.
            boards[slot] = np.transpose(file["boards"][base_row], (2, 1, 0))
            placement_h5 = np.asarray(file["placements"][base_row, :count])
            placements[slot, :count] = np.transpose(placement_h5, (0, 3, 2, 1))
            ren[slot] = file["ren"][base_row, 0]
            back_to_back[slot] = file["back_to_back"][base_row, 0]
            tspin[slot, :count] = file["tspin"][base_row, :count]
            queues[slot] = np.transpose(file["queues"][base_row], (1, 0))
            q = np.asarray(file["teacher_q"][base_row, :count], dtype=np.float32)
            stored_q[slot, :count] = q
            if int(np.argmax(q)) + 1 != int(selected[slot]):
                raise RuntimeError("stored selected action is not stable argmax of old Q")

            reward_rows = np.arange(base_row, base_row + N_STEP, dtype=np.int64)
            rewards = np.asarray(file["rewards"][reward_rows], dtype=np.float32)
            scores_after = np.asarray(file["scores_after"][reward_rows], dtype=np.int32)
            previous_scores = np.concatenate((np.asarray([0], dtype=np.int32), scores_after[:-1]))
            expected_rewards = (scores_after - previous_scores).astype(np.float32) / np.float32(600.0)
            if not np.array_equal(rewards, expected_rewards):
                raise RuntimeError("stored rewards do not exactly equal score delta / 600")
            target_rewards[slot] = rewards

            bootstrap_row = int(base_row + N_STEP)
            bootstrap_rows_1[slot] = bootstrap_row + 1
            bootstrap_count = int(file["action_counts"][bootstrap_row])
            bootstrap_action = int(file["selected_actions"][bootstrap_row])
            bootstrap_values = np.asarray(
                file["teacher_q"][bootstrap_row, :bootstrap_count], dtype=np.float32
            )
            if bootstrap_action != int(np.argmax(bootstrap_values)) + 1:
                raise RuntimeError("bootstrap selected action is not stable argmax of old Q")
            bootstrap_actions[slot] = bootstrap_action
            bootstrap_q[slot] = bootstrap_values[bootstrap_action - 1]
            factor = np.float32(1.0)
            target = np.float32(0.0)
            for reward in rewards:
                target = np.float32(target + factor * reward)
                factor = np.float32(factor * GAMMA)
            targets[slot] = np.float32(target + factor * bootstrap_q[slot])

        if not np.array_equal(targets, EXPECTED_TARGETS):
            raise RuntimeError(
                f"frozen n-step targets changed: {targets.tolist()} != {EXPECTED_TARGETS.tolist()}"
            )

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        boards=boards,
        placements=placements,
        ren=ren,
        back_to_back=back_to_back,
        tspin=tspin,
        queues=queues,
        stored_q=stored_q,
        action_counts=counts,
        selected_actions=selected,
        targets=targets,
        target_rewards=target_rewards,
        bootstrap_rows=bootstrap_rows_1,
        bootstrap_actions=bootstrap_actions,
        bootstrap_q=bootstrap_q,
        source_rows=np.asarray(SOURCE_ROWS_1, dtype=np.int32),
        episode_ids=np.asarray(EXPECTED_EPISODES, dtype=np.int32),
        seeds=np.asarray(EXPECTED_SEEDS, dtype=np.int32),
    )
    record = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": observed_hash,
        "hyperslab_rows_one_based": [int(row + 1) for row in required_rows],
        "source_rows": list(SOURCE_ROWS_1),
        "episode_ids": list(EXPECTED_EPISODES),
        "seeds": list(EXPECTED_SEEDS),
        "n_step": N_STEP,
        "gamma": float(GAMMA),
        "candidate_counts": [int(value) for value in counts],
        "selected_actions": [int(value) for value in selected],
        "bootstrap_rows": [int(value) for value in bootstrap_rows_1],
        "bootstrap_actions": [int(value) for value in bootstrap_actions],
        "bootstrap_q": [float(value) for value in bootstrap_q],
        "targets": [float(value) for value in targets],
        "terminal_mask": [False] * 6,
        "terminal_mask_basis": "same-episode contiguous steps 1--4 exist for every source row",
        "validation_rows_loaded": False,
        "validation_or_test_seeds_used": False,
    }
    args.output_json.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
