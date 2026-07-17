from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


FEATURE_DIGEST = "7e89c16b57dcebac56e3ab4c5be161d5e5430c682e60f3b565dd23ab3b04ac44"
FORBIDDEN = set(range(5742, 5756)) | set(range(8001, 8009)) | set(range(91001, 91033))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_role(
    julia: Path,
    experiment_dir: Path,
    role: str,
    output: Path,
    ridge_artifact: Path | None = None,
) -> tuple[Path, Path]:
    table = output / "table.json"
    manifest = output / "manifest.json"
    milestones = output / "milestones.jsonl"
    command = [
        str(julia),
        f"--project={experiment_dir.parent.parent}",
        "--startup-file=no",
        "--history-file=no",
        str(experiment_dir / "collect_online.jl"),
        role,
        str(table),
        str(manifest),
        str(milestones),
    ]
    if role == "calibration":
        assert ridge_artifact is not None
        command.append(str(ridge_artifact))
    command.append("--synthetic")
    subprocess.run(command, check=True)
    document = json.loads(table.read_text(encoding="utf-8"))
    evidence = json.loads(manifest.read_text(encoding="utf-8"))
    events = [json.loads(line) for line in milestones.read_text(encoding="utf-8").splitlines()]

    assert document["source_role"] == ("training" if role == "train" else "calibration")
    assert document["synthetic"] is True
    assert document["feature_schema_digest"] == FEATURE_DIGEST
    assert len(document["feature_names"]) == 70
    assert len(document["rows"]) == 48
    assert all(len(row["features"]) == 70 for row in document["rows"])
    assert all(
        (row["deployment_decision"] is not None) == (role == "calibration")
        for row in document["rows"]
    )
    assert not ({row["seed"] for row in document["rows"]} & FORBIDDEN)
    assert evidence["table_sha256"] == sha256(table)
    assert evidence["real_model_or_game_loaded"] is False
    assert evidence["counterfactual_states_completed"] == 48
    if role == "train":
        assert evidence["first32_projected_seconds"] <= 3300.0
        assert evidence["first32_projection_basis_states"] == 48
    else:
        assert evidence["first32_projected_seconds"] is None
    assert [event["event"] for event in events[:3]] == [
        "script_enter",
        "args_validated",
        "imports_begin",
    ]
    assert sum(event["event"] == "counterfactual_state_complete" for event in events) == 48
    assert sum(event["event"] == "first32_projection" for event in events) == (
        1 if role == "train" else 0
    )
    assert events[-1]["event"] == "script_complete"

    first = document["rows"][0]
    assert first["canonical_top1_candidate_index"] == 1
    assert first["canonical_top2_candidate_index"] == 2
    assert len(first["top1_branch"]["decision_evidence"]) == 7
    assert first["top1_branch"]["decision_evidence"][0]["selected_index"] == 1
    assert first["top2_branch"]["decision_evidence"][0]["selected_index"] == 2
    assert first["top1_branch"]["decision_evidence"][-1]["kind"] == "bootstrap"
    assert len(first["top1_branch"]["exact_outcome_digest"]) == 64
    assert first["top1_branch"]["pieces_played"] == 6
    if role == "calibration":
        deployment = first["deployment_decision"]
        assert deployment["deployment_decision_schema_version"] == "r1-live-ridge-decision-v1"
        assert deployment["production_feature_vector"] == first["features"]
        assert deployment["production_gate_incremental_elapsed_ns"] >= 0

    before = (sha256(table), sha256(manifest), sha256(milestones))
    collision = subprocess.run(command, check=False, capture_output=True, text=True)
    assert collision.returncode != 0
    assert before == (sha256(table), sha256(manifest), sha256(milestones))
    return table, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--julia", required=True, type=Path)
    args = parser.parse_args()
    experiment_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="r1-collector-cli-") as temporary:
        root = Path(temporary)
        train_output = root / "train"
        train_output.mkdir()
        run_role(
            args.julia.resolve(), experiment_dir, "train", train_output
        )

        eligibility = root / "eligibility.json"
        design = root / "design_freeze.json"
        ridge = root / "ridge.json"
        fit_training_table = root / "fit_training_table.json"
        fit_training_manifest = root / "fit_training_manifest.json"
        fit_milestones = root / "fit_milestones.jsonl"
        subprocess.run(
            [sys.executable, str(experiment_dir / "make_eligibility.py"), str(eligibility)],
            check=True,
        )
        subprocess.run(
            [
                str(args.julia.resolve()),
                f"--project={experiment_dir.parent.parent}",
                "--startup-file=no",
                "--history-file=no",
                str(experiment_dir / "freeze_design.jl"),
                str(eligibility),
                str(design),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(experiment_dir / "fit_ridge.py"),
                str(fit_training_table),
                str(fit_training_manifest),
                str(design),
                str(ridge),
                str(fit_milestones),
                "--synthetic",
            ],
            check=True,
        )
        calibration_output = root / "calibration"
        calibration_output.mkdir()
        run_role(
            args.julia.resolve(),
            experiment_dir,
            "calibration",
            calibration_output,
            ridge,
        )
    print("R1 synthetic collector CLI tests passed")


if __name__ == "__main__":
    main()
