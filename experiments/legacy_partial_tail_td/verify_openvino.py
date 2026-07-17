from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov


CPU_TOLERANCE = 1.0e-4
NPU_TOLERANCE = 1.0e-2
BATCH = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inputs_from_reference(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "board": np.ascontiguousarray(np.transpose(arrays["board"], (3, 2, 0, 1))),
        "placement": np.ascontiguousarray(np.transpose(arrays["placement"], (3, 2, 0, 1))),
        "ren": np.ascontiguousarray(arrays["ren"].T),
        "back_to_back": np.ascontiguousarray(arrays["back_to_back"].T),
        "tspin": np.ascontiguousarray(arrays["tspin"].T),
        "queue": np.ascontiguousarray(np.transpose(arrays["queue"], (2, 1, 0))),
    }


def infer_historical(fixed, tail, inputs: dict[str, np.ndarray]):
    count = len(inputs["board"])
    outputs: list[np.ndarray] = []
    chunks: list[dict[str, object]] = []
    started = time.perf_counter()
    for start in range(0, count, BATCH):
        stop = min(start + BATCH, count)
        batch = {name: value[start:stop] for name, value in inputs.items()}
        compiled = fixed if stop - start == BATCH else tail
        chunk_started = time.perf_counter()
        outputs.append(np.asarray(compiled(batch)[0], dtype=np.float32).reshape(-1))
        chunks.append(
            {
                "start_one_based": start + 1,
                "stop_one_based": stop,
                "size": stop - start,
                "device": "fixed" if stop - start == BATCH else "CPU-tail",
                "seconds": time.perf_counter() - chunk_started,
            }
        )
    return np.concatenate(outputs), time.perf_counter() - started, chunks


def compile_pair(core, fixed_model, dynamic_model, device: str):
    if device not in core.available_devices:
        raise RuntimeError(f"required OpenVINO device {device} unavailable: {core.available_devices}")
    started = time.perf_counter()
    fixed = core.compile_model(fixed_model, device)
    tail = core.compile_model(dynamic_model, "CPU")
    return fixed, tail, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite OpenVINO gate")
    for name in ("candidate_fixed16.xml", "candidate_fixed16.bin", "candidate_tail_dynamic.xml", "candidate_tail_dynamic.bin"):
        if (args.output_directory / name).exists():
            raise RuntimeError(f"refusing to overwrite {name}")
    if not args.weights.is_file() or not args.reference.is_file():
        raise RuntimeError("fresh candidate weights/reference missing")
    training = json.loads((args.output_directory / "training_phase.json").read_text(encoding="utf-8"))
    if training.get("status") != "training_phase_complete":
        raise RuntimeError("training phase is incomplete")
    if training.get("weights_sha256") != sha256_file(args.weights):
        raise RuntimeError("candidate weights differ from training export")
    sys.path.insert(0, str(args.repository / "tools"))
    from legacy_openvino import LegacyOpenVINOBuilder

    reference_npz = np.load(args.reference)
    arrays = {name: np.asarray(reference_npz[name]) for name in reference_npz.files}
    count = int(np.asarray(arrays["action_count"]).reshape(-1)[0])
    reference = np.asarray(arrays["lux_output"], dtype=np.float32).reshape(-1)
    if len(reference) != count or not np.all(np.isfinite(reference)):
        raise RuntimeError("invalid fresh Lux reference")
    builder = LegacyOpenVINOBuilder(args.weights)
    fixed_model = builder.build(batch_size=BATCH)
    dynamic_model = builder.build(batch_size=None)
    fixed_xml = args.output_directory / "candidate_fixed16.xml"
    dynamic_xml = args.output_directory / "candidate_tail_dynamic.xml"
    ov.save_model(fixed_model, fixed_xml, compress_to_fp16=False)
    ov.save_model(dynamic_model, dynamic_xml, compress_to_fp16=False)
    inputs = inputs_from_reference(arrays)
    core = ov.Core()

    cpu_fixed, cpu_tail, cpu_compile = compile_pair(core, fixed_model, dynamic_model, "CPU")
    cpu_output, cpu_inference, cpu_chunks = infer_historical(cpu_fixed, cpu_tail, inputs)
    cpu_error = float(np.max(np.abs(cpu_output.astype(np.float64) - reference)))
    if not np.all(np.isfinite(cpu_output)) or cpu_error > CPU_TOLERANCE:
        raise RuntimeError(f"fresh CPU equivalence failed: {cpu_error}")
    del cpu_fixed, cpu_tail
    gc.collect()

    npu_fixed, npu_tail, npu_compile = compile_pair(core, fixed_model, dynamic_model, "NPU")
    npu_output, npu_inference, npu_chunks = infer_historical(npu_fixed, npu_tail, inputs)
    npu_error = float(np.max(np.abs(npu_output.astype(np.float64) - reference)))
    if not np.all(np.isfinite(npu_output)) or npu_error > NPU_TOLERANCE:
        raise RuntimeError(f"fresh NPU+CPU-tail equivalence failed: {npu_error}")
    full_stop = count - (count % BATCH)
    result = {
        "status": "openvino_gate_pass",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "openvino_version": ov.__version__,
        "weights_path": str(args.weights.resolve()),
        "training_phase_sha256": sha256_file(args.output_directory / "training_phase.json"),
        "weights_sha256": sha256_file(args.weights),
        "reference_path": str(args.reference.resolve()),
        "reference_sha256": sha256_file(args.reference),
        "candidate_count": count,
        "historical_chunk": BATCH,
        "fixed_ir": {"xml": str(fixed_xml), "sha256": sha256_file(fixed_xml), "bin": str(fixed_xml.with_suffix('.bin')), "bin_sha256": sha256_file(fixed_xml.with_suffix('.bin'))},
        "dynamic_tail_ir": {"xml": str(dynamic_xml), "sha256": sha256_file(dynamic_xml), "bin": str(dynamic_xml.with_suffix('.bin')), "bin_sha256": sha256_file(dynamic_xml.with_suffix('.bin'))},
        "cpu": {"compile_seconds": cpu_compile, "inference_seconds": cpu_inference, "max_abs_error": cpu_error, "tolerance": CPU_TOLERANCE, "chunks": cpu_chunks},
        "npu": {
            "compile_seconds_including_cpu_tail": npu_compile,
            "inference_seconds_including_cpu_tail": npu_inference,
            "aggregate_max_abs_error": npu_error,
            "npu_full_chunks_max_abs_error": float(np.max(np.abs(npu_output[:full_stop].astype(np.float64) - reference[:full_stop]))),
            "cpu_tail_max_abs_error": float(np.max(np.abs(npu_output[full_stop:].astype(np.float64) - reference[full_stop:]))) if full_stop < count else 0.0,
            "tolerance": NPU_TOLERANCE,
            "chunks": npu_chunks,
        },
        "fresh_weights_constructor_used": True,
        "existing_weight_artifact_overwritten": False,
        "validation_or_test_seed_loaded": False,
        "game_evaluation_run": False,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
