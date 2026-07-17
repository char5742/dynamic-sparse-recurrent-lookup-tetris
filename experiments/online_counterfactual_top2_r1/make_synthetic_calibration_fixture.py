from __future__ import annotations

"""Build a model-free CLI fixture for the suspended-runner preflight only."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration_gate as gate  # noqa: E402
from test_calibration_gate_python import (  # noqa: E402
    collection_manifest,
    collection_table,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_table", type=Path)
    parser.add_argument("fitted_ridge_artifact", type=Path)
    parser.add_argument("fixture_ridge_artifact", type=Path)
    parser.add_argument("calibration_table", type=Path)
    parser.add_argument("calibration_manifest", type=Path)
    args = parser.parse_args()
    for path in (args.fixture_ridge_artifact, args.calibration_table, args.calibration_manifest):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite synthetic fixture: {path}")

    training = json.loads(args.training_table.read_text(encoding="utf-8"))
    artifact = json.loads(args.fitted_ridge_artifact.read_text(encoding="utf-8"))
    means = np.asarray(artifact["feature_mean"], dtype=np.float64)
    scales = np.asarray(artifact["feature_scale"], dtype=np.float64)
    constant = np.asarray(artifact["constant_feature"], dtype=np.bool_)
    feature = next(
        index for index, is_constant in enumerate(constant)
        if not is_constant and index not in (0, 1, 2, 6)
    )
    coefficients = np.zeros((71, 256), dtype=np.float64)
    coefficients[feature + 1, :] = 1.0
    artifact["coefficients"] = coefficients.tolist()
    artifact["synthetic_wrapper_fixture_coefficients"] = True
    write_json(args.fixture_ridge_artifact, artifact)

    high = means.copy()
    low = means.copy()
    high[feature] += scales[feature]
    low[feature] -= scales[feature]
    high[6] = 4.0
    low[6] = 4.0
    high = high.tolist()
    low = low.tolist()

    calibration = collection_table("calibration")
    calibration["metadata"]["stable_node_key_source_sha256"] = hashlib.sha256(
        b"r1-synthetic-calibration-stable-node-key-source"
    ).hexdigest()
    for row in calibration["rows"]:
        override = int(row["piece_index"]) in gate.SAMPLE_PIECES[:2]
        row["features"] = list(high if override else low)
        row["valid_action_count"] = int(row["features"][6])
        row["q_top1"] = row["features"][0]
        row["q_top2"] = row["features"][1]
        row["q_gap"] = row["features"][2]
        row["advantage"] = 0.8 if override else -0.2
        row["advantage_unclipped_A6"] = row["advantage"]
        row["clipped_target"] = row["advantage"]
        row["root_state_digest"] = f"synthetic-calibration-root-{row['episode_id']}-{row['piece_index']}"
        row["root_future_stream_digest"] = f"synthetic-calibration-future-{row['episode_id']}-{row['piece_index']}"
        row["a1_terminal_within_horizon"] = False
        row["a2_terminal_within_horizon"] = False
    positive_fraction = sum(
        float(row["advantage_unclipped_A6"]) > 0.0 for row in calibration["rows"]
    ) / len(calibration["rows"])
    calibration["metadata"]["positive_advantage_fraction"] = positive_fraction
    write_json(args.calibration_table, calibration)
    manifest = collection_manifest(args.calibration_table, calibration)
    manifest["stable_node_key_source_sha256"] = calibration["metadata"][
        "stable_node_key_source_sha256"
    ]
    manifest["positive_advantage_fraction"] = positive_fraction
    write_json(args.calibration_manifest, manifest)


if __name__ == "__main__":
    main()
