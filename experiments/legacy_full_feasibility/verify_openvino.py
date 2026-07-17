from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov


CPU_TOLERANCE = 1.0e-4
NPU_TOLERANCE = 1.0e-2
BATCH = 16


def compile_actor_pair(weights: Path, device: str):
    from legacy_openvino import LegacyOpenVINOBuilder

    started = time.perf_counter()
    core = ov.Core()
    if device not in core.available_devices:
        raise RuntimeError(f"required OpenVINO device {device} unavailable: {core.available_devices}")
    fixed = core.compile_model(
        LegacyOpenVINOBuilder(weights).build(batch_size=BATCH), device
    )
    # The historical final short chunk is always actual-size CPU inference.
    tail = core.compile_model(
        LegacyOpenVINOBuilder(weights).build(batch_size=None), "CPU"
    )
    seconds = time.perf_counter() - started
    return core, fixed, tail, seconds


def infer_historical(fixed, tail, arrays: dict[str, np.ndarray]):
    from legacy_openvino import LegacyOpenVINOInference

    inputs = LegacyOpenVINOInference._inputs(
        arrays["board"],
        arrays["placement"],
        arrays["ren"],
        arrays["back_to_back"],
        arrays["tspin"],
        arrays["queue"],
    )
    count = len(inputs["board"])
    chunks: list[np.ndarray] = []
    chunk_records: list[dict[str, object]] = []
    started = time.perf_counter()
    for start in range(0, count, BATCH):
        stop = min(start + BATCH, count)
        batch = {name: value[start:stop] for name, value in inputs.items()}
        compiled = fixed if stop - start == BATCH else tail
        chunk_started = time.perf_counter()
        result = np.asarray(compiled(batch)[0], dtype=np.float32).reshape(-1)
        chunk_seconds = time.perf_counter() - chunk_started
        chunks.append(result)
        chunk_records.append(
            {
                "start_one_based": start + 1,
                "stop_one_based": stop,
                "size": stop - start,
                "device": "fixed" if stop - start == BATCH else "CPU-tail",
                "seconds": chunk_seconds,
            }
        )
    seconds = time.perf_counter() - started
    return np.concatenate(chunks), seconds, chunk_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("julia_phase", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite OpenVINO F result")
    if not args.weights.is_file() or not args.reference.is_file():
        raise RuntimeError("temporary updated weights/reference missing")
    sys.path.insert(0, str(args.repository / "tools"))

    phase = json.loads(args.julia_phase.read_text(encoding="utf-8"))
    if phase.get("status") != "julia_phase_complete":
        raise RuntimeError("Julia phase did not complete")
    arrays_npz = np.load(args.reference)
    arrays = {key: np.asarray(arrays_npz[key]) for key in arrays_npz.files}
    count = int(np.asarray(arrays["action_count"]).reshape(-1)[0])
    if count != 51:
        raise RuntimeError(f"reference row must have 51 candidates, observed {count}")
    reference = np.asarray(arrays["lux_output"], dtype=np.float32).reshape(-1)
    if len(reference) != count or not np.all(np.isfinite(reference)):
        raise RuntimeError("invalid Lux updated reference")

    cpu_core, cpu_fixed, cpu_tail, cpu_compile_seconds = compile_actor_pair(
        args.weights, "CPU"
    )
    cpu_output, cpu_inference_seconds, cpu_chunks = infer_historical(
        cpu_fixed, cpu_tail, arrays
    )
    cpu_error = float(np.max(np.abs(cpu_output.astype(np.float64) - reference)))
    if not np.all(np.isfinite(cpu_output)) or cpu_error > CPU_TOLERANCE:
        raise RuntimeError(
            f"updated CPU equivalence failed: max_abs={cpu_error}, tol={CPU_TOLERANCE}"
        )
    del cpu_fixed, cpu_tail, cpu_core
    gc.collect()

    npu_core, npu_fixed, npu_tail, npu_compile_seconds = compile_actor_pair(
        args.weights, "NPU"
    )
    npu_output, npu_inference_seconds, npu_chunks = infer_historical(
        npu_fixed, npu_tail, arrays
    )
    npu_error = float(np.max(np.abs(npu_output.astype(np.float64) - reference)))
    # Also report accelerator-only error separately from the canonical aggregate
    # whose final 3-candidate tail is CPU.
    full_stop = count - (count % BATCH)
    npu_full_chunk_error = float(
        np.max(np.abs(npu_output[:full_stop].astype(np.float64) - reference[:full_stop]))
    )
    tail_error = float(
        np.max(np.abs(npu_output[full_stop:].astype(np.float64) - reference[full_stop:]))
    )
    if not np.all(np.isfinite(npu_output)) or npu_error > NPU_TOLERANCE:
        raise RuntimeError(
            f"updated NPU+CPU-tail equivalence failed: max_abs={npu_error}, tol={NPU_TOLERANCE}"
        )

    actor_refresh_seconds = float(
        phase["export_seconds"]
        + npu_compile_seconds
        + phase["reference_seconds"]
        + npu_inference_seconds
    )
    t1000_seconds = float(
        phase["first_specialization_seconds"]
        + 1000.0 * phase["warm_median_seconds"]
        + 1000.0 * 0.411
        + 4.0 * actor_refresh_seconds
    )
    result = {
        "status": "openvino_phase_complete",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "openvino_version": ov.__version__,
        "reference_source_row": 1,
        "candidate_count": count,
        "historical_batch": BATCH,
        "cpu": {
            "compile_seconds_including_dynamic_cpu_tail": cpu_compile_seconds,
            "inference_seconds": cpu_inference_seconds,
            "max_abs_error": cpu_error,
            "tolerance": CPU_TOLERANCE,
            "chunks": cpu_chunks,
        },
        "npu": {
            "compile_seconds_including_dynamic_cpu_tail": npu_compile_seconds,
            "inference_seconds_including_cpu_tail": npu_inference_seconds,
            "aggregate_max_abs_error": npu_error,
            "npu_full_chunks_max_abs_error": npu_full_chunk_error,
            "cpu_tail_max_abs_error": tail_error,
            "tolerance": NPU_TOLERANCE,
            "chunks": npu_chunks,
        },
        "actor_refresh_definition": (
            "temporary NPZ export + NPU fixed16 compile + CPU dynamic-tail compile "
            "+ Lux-reference synchronization + NPU/CPU-tail inference synchronization"
        ),
        "actor_refresh_seconds": actor_refresh_seconds,
        "t1000_formula": (
            "first_specialization + 1000*warm_median + 1000*0.411 + 4*actor_refresh"
        ),
        "t1000_seconds_before_peak_memory_gate": t1000_seconds,
        "temporary_outputs_promoted": False,
        "score_or_game_evaluation_run": False,
        "validation_or_test_data_used": False,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
