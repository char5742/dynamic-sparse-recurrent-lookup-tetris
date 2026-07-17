from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


FIRST_ROW = 2161
LAST_ROW = 2660
ROW_COUNT = 500
EXPECTED_ELIGIBLE = 494
ACTIONS = 74
MIN_HUBER_IMPROVEMENT = 0.15
MIN_CORRELATION = 0.20
MIN_SIGN_AGREEMENT = 0.60
SIGN_TARGET_FLOOR = 0.10
MIN_TOP1_AGREEMENT = 0.95
MAX_TOP1_AGREEMENT_EXCLUSIVE = 0.995
MIN_TOP1_CHANGES = 3
MAX_CORRECTION_RMS = 0.25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json_new(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    temporary = Path(f"{path}.tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary output: {temporary}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def huber(value: np.ndarray) -> np.ndarray:
    absolute = np.abs(value)
    return np.where(absolute <= 1.0, 0.5 * value * value, absolute - 0.5)


def load_offline(path: Path) -> dict[str, np.ndarray]:
    expected = {
        "source_rows",
        "episode_ids",
        "episode_steps",
        "action_counts",
        "selected_actions",
        "boards",
        "placements",
        "ren",
        "back_to_back",
        "tspin",
        "queues",
        "stored_q",
        "targets",
        "target_residual",
        "target_valid",
        "behavior_is_old_argmax",
        "data_role",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise RuntimeError(
                f"offline NPZ keys differ: missing={sorted(expected - set(archive.files))}, "
                f"extra={sorted(set(archive.files) - expected)}"
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    shapes = {
        "source_rows": (ROW_COUNT,),
        "episode_ids": (ROW_COUNT,),
        "episode_steps": (ROW_COUNT,),
        "action_counts": (ROW_COUNT,),
        "selected_actions": (ROW_COUNT,),
        "boards": (ROW_COUNT, 24, 10, 1),
        "placements": (ROW_COUNT, ACTIONS, 24, 10, 1),
        "ren": (ROW_COUNT,),
        "back_to_back": (ROW_COUNT,),
        "tspin": (ROW_COUNT, ACTIONS),
        "queues": (ROW_COUNT, 7, 6),
        "stored_q": (ROW_COUNT, ACTIONS),
        "targets": (ROW_COUNT,),
        "target_residual": (ROW_COUNT,),
        "target_valid": (ROW_COUNT,),
        "behavior_is_old_argmax": (ROW_COUNT,),
        "data_role": (ROW_COUNT,),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise RuntimeError(f"offline {name} shape {arrays[name].shape} != {shape}")
    dtypes = {
        "source_rows": np.dtype(np.int32),
        "episode_ids": np.dtype(np.int32),
        "episode_steps": np.dtype(np.int32),
        "action_counts": np.dtype(np.int16),
        "selected_actions": np.dtype(np.int16),
        "boards": np.dtype(np.uint8),
        "placements": np.dtype(np.uint8),
        "ren": np.dtype(np.float32),
        "back_to_back": np.dtype(np.float32),
        "tspin": np.dtype(np.float32),
        "queues": np.dtype(np.uint8),
        "stored_q": np.dtype(np.float32),
        "targets": np.dtype(np.float32),
        "target_residual": np.dtype(np.float32),
        "target_valid": np.dtype(np.bool_),
        "behavior_is_old_argmax": np.dtype(np.bool_),
        "data_role": np.dtype(np.int8),
    }
    for name, dtype in dtypes.items():
        if arrays[name].dtype != dtype:
            raise RuntimeError(f"offline {name} dtype {arrays[name].dtype} != {dtype}")
    rows = arrays["source_rows"].astype(np.int64)
    if not np.array_equal(rows, np.arange(FIRST_ROW, LAST_ROW + 1)):
        raise RuntimeError("offline rows are not exactly 2161:2660")
    if sorted(set(arrays["episode_ids"].astype(int).tolist())) != [13, 14]:
        raise RuntimeError("offline episodes are not exactly 13:14")
    if not np.all(arrays["data_role"] == 3):
        raise RuntimeError("offline data_role is not reused development role 3")
    for name in ("boards", "placements", "ren", "back_to_back", "tspin", "queues"):
        if not np.all(np.isfinite(arrays[name])):
            raise RuntimeError(f"non-finite offline model input: {name}")
    counts = arrays["action_counts"].astype(np.int64)
    selected = arrays["selected_actions"].astype(np.int64)
    if not np.all((1 <= counts) & (counts <= ACTIONS)):
        raise RuntimeError("offline action count out of range")
    if not np.all((1 <= selected) & (selected <= counts)):
        raise RuntimeError("offline selected action out of range")
    eligible = arrays["target_valid"].astype(bool)
    if int(np.count_nonzero(eligible)) != EXPECTED_ELIGIBLE:
        raise RuntimeError("offline eligible target count is not exactly 494")
    if not np.all(np.isfinite(arrays["targets"][eligible])):
        raise RuntimeError("non-finite eligible target")
    if not np.all(np.isfinite(arrays["target_residual"][eligible])):
        raise RuntimeError("non-finite eligible target residual")
    for slot, count in enumerate(counts):
        if not np.all(np.isfinite(arrays["stored_q"][slot, :count])):
            raise RuntimeError(f"non-finite valid old-Q at offline slot {slot + 1}")
    selected_old = arrays["stored_q"][np.arange(ROW_COUNT), selected - 1]
    residual_check = arrays["targets"] - selected_old
    residual_error = float(
        np.max(
            np.abs(
                residual_check[eligible].astype(np.float64)
                - arrays["target_residual"][eligible].astype(np.float64)
            )
        )
    )
    if residual_error > 1.0e-6:
        raise RuntimeError(f"offline target residual mismatch: {residual_error}")
    return arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Q1 reused-development offline safety gate."
    )
    parser.add_argument("offline_npz", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("ir_xml", type=Path)
    parser.add_argument("output_json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    offline_path = args.offline_npz.resolve()
    output_directory = args.output_directory.resolve()
    weights = args.weights.resolve()
    ir_xml = args.ir_xml.resolve()
    ir_bin = ir_xml.with_suffix(".bin")
    output_json = args.output_json.resolve()
    training_path = output_directory / "training_phase.json"
    openvino_path = output_directory / "openvino_gate.json"
    if weights != output_directory / "correction_weights.npz":
        raise RuntimeError("weights must be the output-local fresh correction_weights.npz")
    if ir_xml != output_directory / "correction_dynamic.xml":
        raise RuntimeError("offline IR must be the output-local correction_dynamic.xml")
    if output_json != output_directory / "offline_gate.json":
        raise RuntimeError("offline gate must be output-local offline_gate.json")
    for path in (offline_path, training_path, openvino_path, weights, ir_xml, ir_bin):
        if not path.is_file():
            raise RuntimeError(f"required offline-gate input missing: {path}")
    if output_json.exists():
        raise RuntimeError(f"refusing to overwrite {output_json}")

    training = read_json(training_path)
    openvino_gate = read_json(openvino_path)
    if training.get("status") != "training_phase_complete":
        raise RuntimeError("Q1 training phase is incomplete")
    if openvino_gate.get("status") != "correction_openvino_gate_pass":
        raise RuntimeError("Q1 correction OpenVINO gate is incomplete")
    if openvino_gate.get("training_phase_sha256") != sha256_file(training_path):
        raise RuntimeError("OpenVINO gate does not bind the current training phase")
    weights_sha256 = sha256_file(weights)
    ir_xml_sha256 = sha256_file(ir_xml)
    ir_bin_sha256 = sha256_file(ir_bin)
    if training.get("weights_sha256") != weights_sha256:
        raise RuntimeError("offline weights differ from training export")
    if openvino_gate.get("weights_sha256") != weights_sha256:
        raise RuntimeError("offline weights differ from OpenVINO gate")
    if openvino_gate.get("ir_xml_sha256") != ir_xml_sha256:
        raise RuntimeError("offline IR XML differs from OpenVINO gate")
    if openvino_gate.get("ir_bin_sha256") != ir_bin_sha256:
        raise RuntimeError("offline IR BIN differs from OpenVINO gate")
    tolerance = openvino_gate.get("cpu_tolerance")
    correction_error = openvino_gate.get("correction_max_abs_error")
    combined_error = openvino_gate.get("combined_max_abs_error")
    if not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance):
        raise RuntimeError("OpenVINO gate CPU tolerance is missing or non-finite")
    if tolerance > 1.0e-4:
        raise RuntimeError("OpenVINO gate CPU tolerance exceeds preregistration")
    for error, label in (
        (correction_error, "correction"),
        (combined_error, "combined"),
        (openvino_gate.get("copack_vs_per_state_padded_max_abs_error"), "co-pack/padded"),
        (openvino_gate.get("actual_count_vs_padded_valid_max_abs_error"), "actual/padded"),
        (openvino_gate.get("actual_count_vs_copack_valid_max_abs_error"), "actual/co-pack"),
    ):
        if not isinstance(error, (int, float)) or not np.isfinite(error) or error > tolerance:
            raise RuntimeError(f"OpenVINO {label} equivalence is invalid")
    if openvino_gate.get("all_outputs_finite") is not True:
        raise RuntimeError("OpenVINO gate finite-output witness is missing")
    if openvino_gate.get("fresh_weights_constructor_used") is not True:
        raise RuntimeError("OpenVINO gate did not use the fresh weight constructor")
    if openvino_gate.get("fresh_combined_reference_used") is not True:
        raise RuntimeError("OpenVINO gate did not use the fresh combined reference")
    if openvino_gate.get("actual_count_fixed74_valid_output_invariance_verified") is not True:
        raise RuntimeError("OpenVINO actual-count/fixed-74 invariance witness is missing")
    if openvino_gate.get("four_state_copack_invariance_verified") is not True:
        raise RuntimeError("OpenVINO four-state co-pack invariance witness is missing")
    if openvino_gate.get("device") != "CPU" or openvino_gate.get("npu_required") is not False:
        raise RuntimeError("OpenVINO gate is not the preregistered CPU-only gate")
    for record, label in ((training, "training"), (openvino_gate, "OpenVINO")):
        if record.get("initializer_exposed_to_offline_rows") is not True:
            raise RuntimeError(f"{label} initializer exposure disclosure is missing")
        if record.get("validation_or_test_seed_loaded") is not False:
            raise RuntimeError(f"{label} reports validation/test seed use")
    if training.get("game_run") is not False:
        raise RuntimeError("training reports game use")
    if openvino_gate.get("game_evaluation_run") is not False:
        raise RuntimeError("OpenVINO gate reports game use")

    arrays = load_offline(offline_path)
    # Import only after artifact validation. --help remains data- and runtime-safe.
    import openvino as ov

    from correction_openvino import offline_row_inputs

    core = ov.Core()
    if "CPU" not in core.available_devices:
        raise RuntimeError(f"CPU OpenVINO device unavailable: {core.available_devices}")
    compile_started = time.perf_counter()
    compiled = core.compile_model(str(ir_xml), "CPU")
    compile_seconds = time.perf_counter() - compile_started
    counts = arrays["action_counts"].astype(np.int64)
    selected = arrays["selected_actions"].astype(np.int64)
    corrections: list[np.ndarray] = []
    all_finite = True
    inference_started = time.perf_counter()
    for slot in range(ROW_COUNT):
        output = np.asarray(
            compiled(offline_row_inputs(arrays, slot))[0], dtype=np.float32
        ).reshape(-1)
        if output.shape != (counts[slot],):
            raise RuntimeError(f"correction output count mismatch at offline slot {slot + 1}")
        all_finite = all_finite and bool(np.all(np.isfinite(output)))
        corrections.append(output)
    inference_seconds = time.perf_counter() - inference_started

    eligible = arrays["target_valid"].astype(bool)
    target_residual = arrays["target_residual"].astype(np.float64)
    selected_prediction = np.asarray(
        [corrections[slot][selected[slot] - 1] for slot in range(ROW_COUNT)],
        dtype=np.float64,
    )
    target = target_residual[eligible]
    prediction = selected_prediction[eligible]
    zero_huber = float(np.mean(huber(target)))
    selected_huber = float(np.mean(huber(prediction - target)))
    huber_improvement = (
        (zero_huber - selected_huber) / zero_huber if zero_huber > 0 else float("-inf")
    )
    correlation = (
        float(np.corrcoef(prediction, target)[0, 1])
        if np.std(prediction) > 0 and np.std(target) > 0
        else float("nan")
    )
    sign_subset = np.abs(target) >= SIGN_TARGET_FLOOR
    sign_count = int(np.count_nonzero(sign_subset))
    sign_agreement = (
        float(
            np.mean(
                np.signbit(prediction[sign_subset]) == np.signbit(target[sign_subset])
            )
        )
        if sign_count > 0
        else float("nan")
    )

    old_top1: list[int] = []
    combined_top1: list[int] = []
    changed_state_diagnostics: list[dict[str, object]] = []
    squared_sum = 0.0
    correction_count = 0
    combined_finite = True
    for slot, count in enumerate(counts):
        old_q = arrays["stored_q"][slot, :count].astype(np.float64)
        correction = corrections[slot].astype(np.float64)
        combined = old_q + correction
        combined_finite = combined_finite and bool(np.all(np.isfinite(combined)))
        old_top1.append(int(np.argmax(old_q)))
        combined_top1.append(int(np.argmax(combined)))
        old_action = old_top1[-1]
        candidate_action = combined_top1[-1]
        if old_action != candidate_action:
            if count < 2:
                raise RuntimeError("top-1 changed in a one-action state")
            top_two = np.partition(old_q, count - 2)[-2:]
            old_top2_margin = float(np.max(top_two) - np.min(top_two))
            combined_new_minus_old = float(
                combined[candidate_action] - combined[old_action]
            )
            changed_state_diagnostics.append(
                {
                    "source_row": int(arrays["source_rows"][slot]),
                    "old_action_index_one_based": old_action + 1,
                    "candidate_action_index_one_based": candidate_action + 1,
                    "old_top2_margin": old_top2_margin,
                    "old_top2_exact_tie": old_top2_margin == 0.0,
                    "correction_new_minus_old_action": float(
                        correction[candidate_action] - correction[old_action]
                    ),
                    "old_q_at_old_action": float(old_q[old_action]),
                    "old_q_at_new_action": float(old_q[candidate_action]),
                    "combined_q_at_old_action": float(combined[old_action]),
                    "combined_q_at_new_action": float(combined[candidate_action]),
                    "combined_new_minus_old_action": combined_new_minus_old,
                    "combined_old_new_exact_tie": combined_new_minus_old == 0.0,
                }
            )
        squared_sum += float(np.dot(correction, correction))
        correction_count += int(count)
    top1_agreements = int(np.count_nonzero(np.asarray(old_top1) == np.asarray(combined_top1)))
    top1_changes = ROW_COUNT - top1_agreements
    top1_agreement = top1_agreements / ROW_COUNT
    correction_rms = float(np.sqrt(squared_sum / correction_count))

    if len(changed_state_diagnostics) != top1_changes:
        raise RuntimeError("changed-state diagnostic count mismatch")
    gates = {
        "all_outputs_finite": bool(all_finite and combined_finite),
        "selected_huber_improvement_at_least_15_percent": bool(
            np.isfinite(huber_improvement) and huber_improvement >= MIN_HUBER_IMPROVEMENT
        ),
        "selected_residual_correlation_at_least_0_2": bool(
            np.isfinite(correlation) and correlation >= MIN_CORRELATION
        ),
        "sign_agreement_at_least_0_6_for_abs_target_at_least_0_1": bool(
            np.isfinite(sign_agreement) and sign_agreement >= MIN_SIGN_AGREEMENT
        ),
        "combined_top1_at_least_0_95": bool(top1_agreement >= MIN_TOP1_AGREEMENT),
        "combined_top1_below_0_995": bool(top1_agreement < MAX_TOP1_AGREEMENT_EXCLUSIVE),
        "combined_top1_changes_at_least_3": bool(top1_changes >= MIN_TOP1_CHANGES),
        "correction_rms_at_most_0_25": bool(correction_rms <= MAX_CORRECTION_RMS),
    }
    passed = all(bool(value) for value in gates.values())
    result: dict[str, object] = {
        "status": "Q1-offline-promoted" if passed else "Q1-offline-rejected",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rows": [FIRST_ROW, LAST_ROW],
        "row_count": ROW_COUNT,
        "episodes": [13, 14],
        "eligible_target_count": int(np.count_nonzero(eligible)),
        "offline_role": "reused_development_guard",
        "offline_is_held_out_generalization": False,
        "initializer_exposed": True,
        "initializer_exposed_to_offline_rows": True,
        "target_metrics_use_eligible_rows_only": True,
        "top1_and_rms_use_all_valid_actions_in_all_500_rows": True,
        "training_phase_sha256": sha256_file(training_path),
        "openvino_gate_sha256": sha256_file(openvino_path),
        "weights_sha256": weights_sha256,
        "offline_npz_path": str(offline_path),
        "offline_npz_sha256": sha256_file(offline_path),
        "ir_xml_sha256": ir_xml_sha256,
        "ir_bin_sha256": ir_bin_sha256,
        "device": "CPU",
        "npu_required": False,
        "compile_seconds": compile_seconds,
        "inference_seconds": inference_seconds,
        "logical_network_calls": ROW_COUNT,
        "candidate_evaluations": correction_count,
        "zero_correction_selected_huber": zero_huber,
        "candidate_selected_huber": selected_huber,
        "selected_huber_improvement_fraction": huber_improvement,
        "selected_residual_correlation": correlation,
        "sign_target_abs_floor": SIGN_TARGET_FLOOR,
        "sign_subset_count": sign_count,
        "sign_agreement": sign_agreement,
        "combined_top1_agreements": top1_agreements,
        "combined_top1_changes": top1_changes,
        "combined_top1_agreement": top1_agreement,
        "changed_state_diagnostics": changed_state_diagnostics,
        "correction_rms": correction_rms,
        "thresholds": {
            "minimum_huber_improvement_fraction": MIN_HUBER_IMPROVEMENT,
            "minimum_correlation": MIN_CORRELATION,
            "minimum_sign_agreement": MIN_SIGN_AGREEMENT,
            "minimum_top1_agreement": MIN_TOP1_AGREEMENT,
            "maximum_top1_agreement_exclusive": MAX_TOP1_AGREEMENT_EXCLUSIVE,
            "minimum_top1_changes": MIN_TOP1_CHANGES,
            "maximum_correction_rms": MAX_CORRECTION_RMS,
        },
        "gates": gates,
        "checkpoint_selection_performed": False,
        "earlier_checkpoint_rollback": False,
        "validation_or_test_seed_loaded": False,
        "game_evaluation_run": False,
    }
    write_json_new(output_json, result)
    if not passed:
        failed = sorted(name for name, value in gates.items() if not value)
        raise RuntimeError(f"Q1 offline safety gate rejected: {failed}")


if __name__ == "__main__":
    main()
