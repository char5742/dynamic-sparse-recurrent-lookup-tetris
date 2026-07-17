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
    output = root / "final_result.json"
    if output.exists():
        raise RuntimeError("refusing to overwrite final P1 result")
    freeze = read_optional(root / "freeze.json")
    eligibility = read_optional(root / "eligibility.json")
    row_freeze = read_optional(root / "row_freeze.json")
    training_extraction = read_optional(root / "training_extraction.json")
    training = read_optional(root / "training_phase.json")
    training_failure = read_optional(root / "training_failure.json")
    openvino = read_optional(root / "openvino_gate.json")
    offline_extraction = read_optional(root / "offline_extraction.json")
    offline = read_optional(root / "offline_gate.json")
    development = read_optional(root / "development.json")
    monitor = read_optional(root / "monitor.json")
    failures: list[str] = []

    if not freeze or freeze.get("experiment", freeze.get("experiment_id")) != "legacy_partial_tail_td_P1":
        failures.append("missing or invalid pre-execution freeze")
    if not eligibility or eligibility.get("eligible_count") != 1482:
        failures.append("training eligibility is not the fixed 1,482-row set")
    if not row_freeze or row_freeze.get("row_count") != 300:
        failures.append("missing exact 300-row Xoshiro freeze")
    if not training_extraction or not training_extraction.get("targets_frozen_before_training"):
        failures.append("training targets were not frozen before learning")
    if not training or training.get("status") != "training_phase_complete":
        failures.append("300-update partial training phase incomplete")
    if training_failure:
        failures.append(f"training failure artifact exists at stage {training_failure.get('stage')}")
    if training:
        step0 = training.get("step0", {})
        if step0.get("split_tail_max_abs_error", float("inf")) > 1.0e-6:
            failures.append("split-tail step-0 equivalence failed")
        if step0.get("stored_old_q_max_abs_error", float("inf")) > 1.0e-2:
            failures.append("stored old-Q step-0 equivalence failed")
        if training.get("update_count") != 300:
            failures.append("training did not consume exactly 300 frozen rows")
        if training.get("gradient_elements") != 2_949_508 or not training.get(
            "gradient_elements_every_update", False
        ):
            failures.append("gradient element accounting mismatch")
        if training.get("optimizer_moment_elements") != 5_899_016:
            failures.append("AdamW moment element accounting mismatch")
        if not training.get("finite_differences") or not all(
            record.get("passed", False) for record in training["finite_differences"]
        ):
            failures.append("finite-difference gate incomplete")
        if not training.get("changed_trainable_arrays"):
            failures.append("no trainable array changed")
        if training.get("trainable_array_count") != 34:
            failures.append("trainable array-leaf count mismatch")
        if training.get("frozen_parameter_array_count") != 250:
            failures.append("frozen array-leaf count mismatch")
        if training.get("running_state_array_count") != 69:
            failures.append("running-state array-leaf count mismatch")
        if training.get("export_array_count") != 353:
            failures.append("fresh merged export array count mismatch")
        if training.get("final_merge_max_abs_error", float("inf")) > 1.0e-6:
            failures.append("fresh merged graph differs from split tail")
        if not training.get("frozen_parameter_sha_unchanged", False):
            failures.append("frozen parameter SHA changed")
        if not training.get("running_state_sha_unchanged", False):
            failures.append("running state changed")
        if training.get("first_update_seconds", float("inf")) > 180.0:
            failures.append("first update exceeded 180 seconds")
        warm = training.get("warm_update_seconds", [])
        if len(warm) != 6 or any(value > 15.0 for value in warm):
            failures.append("six warm-update hard gates failed")
        if training.get("warm_median_seconds", float("inf")) > 4.5:
            failures.append("warm-update median exceeded 4.5 seconds")
        later_updates = training.get("update_records", [])[1:]
        if len(later_updates) != 299 or any(
            record.get("timing", {}).get("seconds", float("inf")) > 15.0
            for record in later_updates
        ):
            failures.append("an update after the first exceeded 15 seconds or lacks timing")
        if training.get("projected_total_seconds", float("inf")) > 2100.0:
            failures.append("measured remaining-time projection exceeded 35 minutes")
    if not openvino or openvino.get("status") != "openvino_gate_pass":
        failures.append("fresh OpenVINO export/compile/equivalence incomplete")
    if openvino:
        if openvino.get("cpu", {}).get("max_abs_error", float("inf")) > 1.0e-4:
            failures.append("fresh CPU equivalence failed")
        if openvino.get("npu", {}).get("aggregate_max_abs_error", float("inf")) > 1.0e-2:
            failures.append("fresh NPU equivalence failed")
        if not openvino.get("fresh_weights_constructor_used", False):
            failures.append("OpenVINO did not bind the fresh candidate weights")
        if not openvino.get("fixed_ir") or not openvino.get("dynamic_tail_ir"):
            failures.append("fresh fixed/dynamic OpenVINO artifacts are missing")
    if not offline_extraction or offline_extraction.get("rows_loaded") != [1501, 2000]:
        failures.append("offline extraction is not exactly rows 1501--2000")
    if not offline or offline.get("status") != "offline_gate_pass":
        failures.append("offline safety gate rejected or is incomplete")
    if offline:
        if offline.get("top1_agreement", 0.0) < 0.95 or not offline.get("all_q_finite", False):
            failures.append("offline finite/top-1 threshold failed")
        if not offline.get("explicit_fresh_weight_constructor", False):
            failures.append("offline gate did not bind fresh candidate weights explicitly")
    if not development or development.get("status") != "P1-development-pass":
        failures.append("development two-seed gate rejected or is incomplete")
    if development:
        if not development.get("both_strictly_positive", False):
            failures.append("development differences were not 2/2 strictly positive")
        if development.get("paired_mean_difference", float("-inf")) < 500.0:
            failures.append("development paired mean was below +500")
        if development.get("paired_median_difference", float("-inf")) <= 0.0:
            failures.append("development paired median was not positive")
        evaluations = development.get("evaluations", [])
        if len(evaluations) != 4 or any(
            record.get("steps") != 100 or record.get("game_over", True)
            for record in evaluations
        ):
            failures.append("not all four paired episodes completed 100 pieces")
        accounting = development.get("accounting", {})
        if any(
            name not in accounting
            for name in (
                "candidate_evaluations",
                "logical_network_calls",
                "physical_network_calls",
                "generation_seconds",
                "inference_seconds",
                "episode_wall_seconds",
            )
        ):
            failures.append("development compute/time accounting is incomplete")
    if not monitor:
        failures.append("missing external process-tree monitor")
    else:
        if monitor.get("peak_process_tree_working_set_bytes", float("inf")) > 8 * 1024**3:
            failures.append("process-tree peak working set exceeded 8 GiB")
        if monitor.get("wall_seconds", float("inf")) > 2100.0:
            failures.append("hard 35-minute wall exceeded")
        if monitor.get("stop_reason") != "completed":
            failures.append(f"external monitor stopped: {monitor.get('stop_reason')}")

    passed = not failures
    result = {
        "experiment": "P1 conservative legacy partial-tail anchored TD",
        "status": "P1-development-pass" if passed else "P1-development-fail",
        "success": passed,
        "failures": failures,
        "freeze": freeze,
        "eligibility": eligibility,
        "row_freeze": row_freeze,
        "training_extraction": training_extraction,
        "training": training,
        "training_failure": training_failure,
        "openvino": openvino,
        "offline_extraction": offline_extraction,
        "offline": offline,
        "development": development,
        "monitor": monitor,
        "scope": "development evidence only; not statistical model-beat evidence",
        "checkpoint_frozen_for_sealed_review": False,
        "sealed_test_authorized": False,
        "validation_seed_used": False,
        "sealed_test_seed_used": False,
        "rescue_run_authorized": False,
        "original_checkpoint_overwritten": False,
        "existing_weight_artifact_overwritten": False,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
