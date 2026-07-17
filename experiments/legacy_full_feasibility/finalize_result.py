from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_optional(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    root = args.output_directory
    destination = root / "final_result.json"
    if destination.exists():
        raise RuntimeError("refusing to overwrite final F result")

    freeze = read_optional(root / "freeze.json")
    extraction = read_optional(root / "subset.json")
    julia = read_optional(root / "julia_phase.json")
    openvino = read_optional(root / "openvino_phase.json")
    monitor = read_optional(root / "monitor.json")
    failures: list[str] = []

    if not freeze:
        failures.append("missing pre-execution freeze")
    if not extraction or extraction.get("validation_rows_loaded") is not False:
        failures.append("missing or invalid whitelisted dataset extraction")
    if not julia or julia.get("status") != "julia_phase_complete":
        failures.append("Julia six-update/export phase incomplete")
    if not openvino or openvino.get("status") != "openvino_phase_complete":
        failures.append("OpenVINO CPU/NPU refresh/equivalence phase incomplete")
    if not monitor or monitor.get("stop_reason") != "completed":
        failures.append(f"one-shot monitor did not complete: {monitor and monitor.get('stop_reason')}")

    if julia:
        if julia.get("zero_update", {}).get("max_abs_error", float("inf")) > 1.0e-2:
            failures.append("zero-update tolerance failed")
        if julia.get("head_bias_abs_change", 0.0) <= 0.0:
            failures.append("proof parameter did not change")
        if len(julia.get("warm_updates", [])) != 6:
            failures.append("not all six warm updates recorded")
        if julia.get("first_specialization_seconds", float("inf")) > 300.0:
            failures.append("first specialization exceeded 300 seconds")
        if any(
            update.get("timing", {}).get("seconds", float("inf")) > 120.0
            for update in julia.get("warm_updates", [])
        ):
            failures.append("a warm update exceeded 120 seconds")
    if openvino:
        if openvino.get("cpu", {}).get("max_abs_error", float("inf")) > 1.0e-4:
            failures.append("CPU updated-weight equivalence failed")
        if openvino.get("npu", {}).get("aggregate_max_abs_error", float("inf")) > 1.0e-2:
            failures.append("NPU updated-weight equivalence failed")
        if openvino.get("t1000_seconds_before_peak_memory_gate", float("inf")) > 1800.0:
            failures.append("T1000 exceeded 1800 seconds")
    if monitor:
        if monitor.get("peak_working_set_bytes", 2**63) > 8 * 1024**3:
            failures.append("peak working set exceeded 8 GiB")
        if monitor.get("wall_seconds", float("inf")) > 25 * 60:
            failures.append("hard wall exceeded 25 minutes")

    result = {
        "benchmark": "F legacy full-model continuation feasibility",
        "status": "F-feasible" if not failures else "F-infeasible",
        "success": not failures,
        "failures": failures,
        "freeze": freeze,
        "dataset_extraction": extraction,
        "julia_phase": julia,
        "openvino_phase": openvino,
        "monitor": monitor,
        "scope": "throughput/plumbing gate only; not a learned-policy or score result",
        "temporary_outputs_promoted": False,
        "score_or_game_evaluation_run": False,
        "validation_or_test_data_used": False,
    }
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
