from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


CPU_TOLERANCE = 1.0e-4
WITNESS_STATES = 4
ACTIONS = 74
CANDIDATES = WITNESS_STATES * ACTIONS


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


def load_reference(path: Path) -> dict[str, np.ndarray]:
    expected = {
        "board",
        "placement",
        "ren",
        "back_to_back",
        "tspin",
        "queue",
        "mask",
        "old_q",
        "lux_correction",
        "lux_combined",
        "source_rows",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise RuntimeError(
                f"combined reference keys differ: missing={sorted(expected - set(archive.files))}, "
                f"extra={sorted(set(archive.files) - expected)}"
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    shapes = {
        "board": (24, 10, 1, CANDIDATES),
        "placement": (24, 10, 1, CANDIDATES),
        "ren": (1, CANDIDATES),
        "back_to_back": (1, CANDIDATES),
        "tspin": (1, CANDIDATES),
        "queue": (7, 6, CANDIDATES),
        "mask": (ACTIONS, WITNESS_STATES),
        "old_q": (ACTIONS, WITNESS_STATES),
        "lux_correction": (ACTIONS, WITNESS_STATES),
        "lux_combined": (ACTIONS, WITNESS_STATES),
        "source_rows": (WITNESS_STATES,),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise RuntimeError(f"combined reference {name} shape {arrays[name].shape} != {shape}")
    for name in expected - {"source_rows"}:
        if arrays[name].dtype != np.float32:
            raise RuntimeError(f"combined reference {name} dtype {arrays[name].dtype} != float32")
    if arrays["source_rows"].dtype != np.int32:
        raise RuntimeError("combined reference source_rows dtype is not int32")
    mask = arrays["mask"]
    if not np.all((mask == 0) | (mask == 1)):
        raise RuntimeError("combined reference mask is not binary")
    for column in range(WITNESS_STATES):
        count = int(np.count_nonzero(mask[:, column]))
        if count < 1 or not np.array_equal(
            mask[:, column],
            np.concatenate(
                [np.ones(count, dtype=mask.dtype), np.zeros(ACTIONS - count, dtype=mask.dtype)]
            ),
        ):
            raise RuntimeError("combined reference mask is not a valid prefix mask")
    valid = mask.astype(bool)
    for name in ("board", "placement", "ren", "back_to_back", "tspin", "queue", "lux_correction"):
        if not np.all(np.isfinite(arrays[name])):
            raise RuntimeError(f"non-finite combined reference input/output: {name}")
    if not np.all(np.isfinite(arrays["old_q"][valid])):
        raise RuntimeError("non-finite valid old-Q in combined reference")
    if not np.all(np.isfinite(arrays["lux_combined"][valid])):
        raise RuntimeError("non-finite valid combined Q in combined reference")
    expected_combined = arrays["old_q"][valid] + arrays["lux_correction"][valid]
    if not np.array_equal(arrays["lux_combined"][valid], expected_combined):
        raise RuntimeError("fresh combined reference is not old-Q + Lux correction")
    return arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the fresh Q1 correction weights and verify CPU equivalence."
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output_json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_directory = args.output_directory.resolve()
    weights = args.weights.resolve()
    reference_path = args.reference.resolve()
    output_json = args.output_json.resolve()
    training_path = output_directory / "training_phase.json"
    ir_xml = output_directory / "correction_dynamic.xml"
    ir_bin = output_directory / "correction_dynamic.bin"
    if weights != output_directory / "correction_weights.npz":
        raise RuntimeError("weights must be the output-local fresh correction_weights.npz")
    if reference_path != output_directory / "combined_reference.npz":
        raise RuntimeError("reference must be the output-local fresh combined_reference.npz")
    if output_json != output_directory / "openvino_gate.json":
        raise RuntimeError("OpenVINO gate must be output-local openvino_gate.json")
    for path in (output_json, ir_xml, ir_bin):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite {path}")
    for path in (training_path, weights, reference_path):
        if not path.is_file():
            raise RuntimeError(f"required fresh artifact missing: {path}")

    training = read_json(training_path)
    if training.get("status") != "training_phase_complete":
        raise RuntimeError("Q1 training phase is incomplete")
    if training.get("weights_sha256") != sha256_file(weights):
        raise RuntimeError("weights differ from the fresh training export")
    if training.get("reference_sha256") != sha256_file(reference_path):
        raise RuntimeError("combined reference differs from the fresh training export")
    if training.get("initializer_exposed_to_offline_rows") is not True:
        raise RuntimeError("initializer exposure disclosure is missing")
    if training.get("validation_or_test_seed_loaded") is not False:
        raise RuntimeError("training phase reports validation/test seed use")
    if training.get("game_run") is not False:
        raise RuntimeError("training phase reports a game run")

    arrays = load_reference(reference_path)
    # Import only after argument and immutable-artifact validation. --help is safe.
    import openvino as ov

    from correction_openvino import Q1CorrectionOpenVINOBuilder, julia_inputs

    started = time.perf_counter()
    builder = Q1CorrectionOpenVINOBuilder(weights)
    model = builder.build(batch_size=None)
    ov.save_model(model, ir_xml, compress_to_fp16=False)
    export_seconds = time.perf_counter() - started
    core = ov.Core()
    if "CPU" not in core.available_devices:
        raise RuntimeError(f"CPU OpenVINO device unavailable: {core.available_devices}")
    compile_started = time.perf_counter()
    compiled = core.compile_model(model, "CPU")
    compile_seconds = time.perf_counter() - compile_started
    inputs = julia_inputs(
        arrays["board"],
        arrays["placement"],
        arrays["ren"],
        arrays["back_to_back"],
        arrays["tspin"],
        arrays["queue"],
    )
    inference_started = time.perf_counter()
    correction = np.asarray(compiled(inputs)[0], dtype=np.float32).reshape(-1)
    inference_seconds = time.perf_counter() - inference_started
    lux_correction = arrays["lux_correction"].reshape(-1, order="F")
    correction_error = float(
        np.max(np.abs(correction.astype(np.float64) - lux_correction.astype(np.float64)))
    )
    old_q = arrays["old_q"].reshape(-1, order="F")
    mask = arrays["mask"].reshape(-1, order="F").astype(bool)
    lux_combined = arrays["lux_combined"].reshape(-1, order="F")
    combined = old_q + correction
    combined_error = float(
        np.max(
            np.abs(
                combined[mask].astype(np.float64)
                - lux_combined[mask].astype(np.float64)
            )
        )
    )
    copack_matrix = correction.reshape(ACTIONS, WITNESS_STATES, order="F")
    padded_errors: list[float] = []
    actual_padded_errors: list[float] = []
    actual_copack_errors: list[float] = []
    action_counts: list[int] = []
    invariance_started = time.perf_counter()
    for state in range(WITNESS_STATES):
        start = state * ACTIONS
        stop = start + ACTIONS
        state_inputs = {name: value[start:stop] for name, value in inputs.items()}
        padded = np.asarray(compiled(state_inputs)[0], dtype=np.float32).reshape(-1)
        count = int(np.count_nonzero(arrays["mask"][:, state]))
        action_counts.append(count)
        actual_inputs = {name: value[:count] for name, value in state_inputs.items()}
        actual = np.asarray(compiled(actual_inputs)[0], dtype=np.float32).reshape(-1)
        padded_errors.append(
            float(
                np.max(
                    np.abs(
                        padded.astype(np.float64)
                        - copack_matrix[:, state].astype(np.float64)
                    )
                )
            )
        )
        actual_padded_errors.append(
            float(
                np.max(
                    np.abs(
                        actual.astype(np.float64)
                        - padded[:count].astype(np.float64)
                    )
                )
            )
        )
        actual_copack_errors.append(
            float(
                np.max(
                    np.abs(
                        actual.astype(np.float64)
                        - copack_matrix[:count, state].astype(np.float64)
                    )
                )
            )
        )
    invariance_seconds = time.perf_counter() - invariance_started
    copack_vs_padded_error = max(padded_errors)
    actual_vs_padded_error = max(actual_padded_errors)
    actual_vs_copack_error = max(actual_copack_errors)
    if not np.all(np.isfinite(correction)):
        raise RuntimeError("non-finite OpenVINO correction output")
    if any(
        error > CPU_TOLERANCE
        for error in (
            correction_error,
            combined_error,
            copack_vs_padded_error,
            actual_vs_padded_error,
            actual_vs_copack_error,
        )
    ):
        raise RuntimeError(
            "CPU equivalence/invariance failed: "
            f"correction={correction_error}, combined={combined_error}, "
            f"copack_padded={copack_vs_padded_error}, "
            f"actual_padded={actual_vs_padded_error}, "
            f"actual_copack={actual_vs_copack_error}"
        )
    result: dict[str, object] = {
        "status": "correction_openvino_gate_pass",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "openvino_version": ov.__version__,
        "device": "CPU",
        "npu_required": False,
        "training_phase_path": str(training_path),
        "training_phase_sha256": sha256_file(training_path),
        "weights_path": str(weights),
        "weights_sha256": sha256_file(weights),
        "combined_reference_path": str(reference_path),
        "combined_reference_sha256": sha256_file(reference_path),
        "candidate_count": CANDIDATES,
        "witness_state_count": WITNESS_STATES,
        "witness_action_counts": action_counts,
        "valid_combined_count": int(np.count_nonzero(mask)),
        "ir_xml": str(ir_xml),
        "ir_xml_sha256": sha256_file(ir_xml),
        "ir_bin": str(ir_bin),
        "ir_bin_sha256": sha256_file(ir_bin),
        "export_seconds": export_seconds,
        "compile_seconds": compile_seconds,
        "inference_seconds": inference_seconds,
        "invariance_inference_seconds": invariance_seconds,
        "correction_max_abs_error": correction_error,
        "combined_max_abs_error": combined_error,
        "copack_vs_per_state_padded_max_abs_error": copack_vs_padded_error,
        "actual_count_vs_padded_valid_max_abs_error": actual_vs_padded_error,
        "actual_count_vs_copack_valid_max_abs_error": actual_vs_copack_error,
        "actual_count_fixed74_valid_output_invariance_verified": True,
        "four_state_copack_invariance_verified": True,
        "cpu_tolerance": CPU_TOLERANCE,
        "all_outputs_finite": True,
        "fresh_weights_constructor_used": True,
        "fresh_combined_reference_used": True,
        "initializer_exposed": True,
        "initializer_exposed_to_offline_rows": True,
        "offline_role": "reused_development_guard",
        "validation_or_test_seed_loaded": False,
        "game_evaluation_run": False,
    }
    write_json_new(output_json, result)


if __name__ == "__main__":
    main()
