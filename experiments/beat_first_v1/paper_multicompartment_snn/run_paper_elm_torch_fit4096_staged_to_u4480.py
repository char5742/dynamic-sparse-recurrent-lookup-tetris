"""Stage fit4096 batch32 to u4480 with checkpoint-before-base-val gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


TARGET_UPDATE = 4_480
STAGE_UPDATES = 256
FINAL_STAGE_UPDATES = 128
STAGE_RUNNER = Path(__file__).with_name(
    "run_paper_elm_torch_cpu_fit4096_batch32_cosine_stage.py"
)
BASE_VALIDATOR = Path(__file__).with_name(
    "evaluate_paper_elm_torch_fit4096_base_val8_m1000_safe.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint_update(path: Path) -> int:
    import torch

    payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != "paper_elm_torch_cpu_variable.v1":
        raise ValueError("input checkpoint schema differs")
    if int(payload["memory"]) != 1_000:
        raise ValueError("input checkpoint is not M1000")
    if bool(payload.get("heldout_opened", True)):
        raise ValueError("input checkpoint lacks heldout exclusion")
    return int(payload["update_index"])


def _parse_nmda_gate(value: str):
    values = [float(item) for item in value.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError(
            "--gate-nmda-max requires four comma-separated values"
        )
    return values


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--augmentation-dataset",
        type=Path,
        required=True,
    )
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-sha256",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--validator-warmstart",
        type=Path,
        required=True,
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=6077687918186389330)
    parser.add_argument("--target-update", type=int, default=TARGET_UPDATE)
    parser.add_argument("--gate-auroc-min", type=float)
    parser.add_argument("--gate-voltage-max", type=float)
    parser.add_argument("--gate-balanced-bce-max", type=float)
    parser.add_argument("--gate-nmda-max", type=_parse_nmda_gate)
    return parser.parse_args()


def _gate_config(options) -> dict:
    return {
        "exact_spike_auroc_min": options.gate_auroc_min,
        "clip_voltage_rmse_mv_max": options.gate_voltage_max,
        "spike_balanced_bce_max": options.gate_balanced_bce_max,
        "nmda_normalized_rmse_max": options.gate_nmda_max,
    }


def _gate_achieved(metrics: dict, gates: dict) -> bool:
    configured = False
    for name, threshold in gates.items():
        if threshold is None:
            continue
        configured = True
        if name == "exact_spike_auroc_min":
            if float(metrics["exact_spike_auroc"]) < threshold:
                return False
        elif name == "clip_voltage_rmse_mv_max":
            if float(metrics["clip_voltage_rmse_mv"]) > threshold:
                return False
        elif name == "spike_balanced_bce_max":
            if float(metrics["spike_balanced_bce"]) > threshold:
                return False
        elif name == "nmda_normalized_rmse_max":
            actual = metrics["nmda_normalized_rmse"]
            if any(
                float(value) > float(limit)
                for value, limit in zip(actual, threshold)
            ):
                return False
    return configured


def _best_key(record: dict):
    metrics = record["metrics"]
    return (
        float(metrics["exact_spike_auroc"]),
        -float(metrics["clip_voltage_rmse_mv"]),
        -float(metrics["spike_balanced_bce"]),
        -sum(
            float(value)
            for value in metrics["nmda_normalized_rmse"]
        ),
    )


def _run_logged(command, stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout:
        with stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                [str(value) for value in command],
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}); "
            f"see {stderr_path}"
        )


def main() -> None:
    options = _parse_args()
    if options.threads != 6:
        raise ValueError("staged fit4096 runner requires threads=6")
    if options.target_update != TARGET_UPDATE:
        raise ValueError("paper-equivalent target must be u4480")
    checkpoint = options.checkpoint_in.resolve()
    actual_sha256 = _sha256(checkpoint)
    if actual_sha256 != options.checkpoint_sha256.lower():
        raise ValueError("initial checkpoint SHA256 differs")
    current_update = _checkpoint_update(checkpoint)
    if current_update < 512 or current_update >= TARGET_UPDATE:
        raise ValueError("initial update is outside staged range")
    options.run_root.mkdir(parents=True, exist_ok=True)
    gates = _gate_config(options)
    summary = {
        "schema": "paper_elm_fit4096_staged_to_u4480.v1",
        "base_dataset": str(options.dataset.resolve()),
        "augmentation_dataset": str(
            options.augmentation_dataset.resolve()
        ),
        "initial_checkpoint": str(checkpoint),
        "initial_checkpoint_sha256": actual_sha256,
        "initial_update": current_update,
        "target_update": TARGET_UPDATE,
        "paper_equivalent_epochs": 35,
        "updates_per_epoch": 128,
        "stage_updates": STAGE_UPDATES,
        "final_stage_updates": FINAL_STAGE_UPDATES,
        "schedule": {
            "name": "paper-like cosine decay",
            "fixed_through_update": 512,
            "max_learning_rate": 2.0e-4,
            "floor_learning_rate": 2.0e-5,
            "final_update": TARGET_UPDATE,
        },
        "validation": "base val8 only after atomic checkpoint",
        "best_selection": (
            "lexicographic: max AUROC, min voltage RMSE, "
            "min balanced BCE, min NMDA RMSE sum"
        ),
        "gates": gates,
        "heldout_opened": False,
        "stages": [],
        "stop_reason": None,
    }
    best = None
    _atomic_json(options.run_root / "staged_summary.json", summary)

    while current_update < TARGET_UPDATE:
        updates = min(STAGE_UPDATES, TARGET_UPDATE - current_update)
        stage_end = current_update + updates
        stage_root = options.run_root / f"stage_u{stage_end:04d}"
        stage_root.mkdir(parents=True, exist_ok=False)
        checkpoint_out = stage_root / f"checkpoint_update_{stage_end}.pt"
        train_stdout = stage_root / "train_stdout.json"
        train_stderr = stage_root / "train_stderr.log"
        stage_command = [
            options.python,
            STAGE_RUNNER,
            "--dataset",
            options.dataset,
            "--augmentation-dataset",
            options.augmentation_dataset,
            "--bridge",
            options.bridge,
            "--mode",
            "train",
            "--memory",
            "1000",
            "--backend",
            "torchscript",
            "--threads",
            str(options.threads),
            "--updates",
            str(updates),
            "--batch-size",
            "32",
            "--seed",
            str(options.seed),
            "--checkpoint-in",
            checkpoint,
            "--checkpoint-sha256",
            actual_sha256,
            "--checkpoint-out",
            checkpoint_out,
        ]
        _run_logged(stage_command, train_stdout, train_stderr)
        if not checkpoint_out.is_file():
            raise RuntimeError("stage exited without atomic checkpoint")
        checkpoint_sha256 = _sha256(checkpoint_out)
        if _checkpoint_update(checkpoint_out) != stage_end:
            raise RuntimeError("stage checkpoint update differs")

        validation_stdout = stage_root / "base_joint_validation.json"
        validation_stderr = stage_root / "validation_stderr.log"
        validation_command = [
            options.python,
            BASE_VALIDATOR,
            "--dataset",
            options.dataset,
            "--augmentation-dataset",
            options.augmentation_dataset,
            "--bridge",
            options.bridge,
            "--checkpoint",
            checkpoint_out,
            "--checkpoint-in",
            options.validator_warmstart,
            "--threads",
            str(options.threads),
        ]
        _run_logged(
            validation_command,
            validation_stdout,
            validation_stderr,
        )
        validation = json.loads(
            validation_stdout.read_text(encoding="utf-8")
        )
        metrics = validation["metrics"]
        if bool(metrics.get("heldout_opened", True)):
            raise RuntimeError("validation lacks heldout exclusion")
        record = {
            "stage_start_update": current_update + 1,
            "stage_end_update": stage_end,
            "updates": updates,
            "checkpoint": str(checkpoint_out),
            "checkpoint_sha256": checkpoint_sha256,
            "train_stdout": str(train_stdout),
            "train_stderr": str(train_stderr),
            "validation_stdout": str(validation_stdout),
            "validation_stderr": str(validation_stderr),
            "metrics": metrics,
            "gate_achieved": _gate_achieved(metrics, gates),
            "heldout_opened": False,
        }
        summary["stages"].append(record)
        if best is None or _best_key(record) > _best_key(best):
            best = record
            summary["best"] = record
            _atomic_text(
                options.run_root / "best_checkpoint.txt",
                f"{checkpoint_out}\n{checkpoint_sha256}\n",
            )
        checkpoint = checkpoint_out
        actual_sha256 = checkpoint_sha256
        current_update = stage_end
        if record["gate_achieved"]:
            summary["stop_reason"] = "validation_gate_achieved"
            _atomic_json(
                options.run_root / "staged_summary.json",
                summary,
            )
            break
        _atomic_json(options.run_root / "staged_summary.json", summary)

    if summary["stop_reason"] is None:
        summary["stop_reason"] = "paper_equivalent_target_reached"
    summary["final_update"] = current_update
    summary["heldout_opened"] = False
    _atomic_json(options.run_root / "staged_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
