"""Exclusive CPU/NPU HashStem benchmark and numerical/routing gate.

Expected witness NPZ keys:
  input                  [N, 559], N is a positive multiple of 16
  reference_output       [N, 256], from Julia `hashstem_reference!`
  probe_ids_l{1,2,3}     [N, C_l] int neuron IDs, -1 is padding
  probe_keys_l{1,2,3}    [N, C_l, 64] corresponding route keys
  query_residual_l{1,2,3}[N, 64], zero for L1 and CPU CountSketch for L2/L3

The route witness reuses each layer's exact CPU WTA candidate set and compares
stable exact reranking IDs. It is outside the timed HashStem region.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import openvino as ov


BATCH = 16
INPUTS = 559
OUTPUTS = 256
QUERY_RANGES = ((0, 64), (64, 128), (128, 192))
SCHEMA = "learned-hashstem-v1-conv-dw-pw-pool-1039x214-relu"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def stable_route_ids(query: np.ndarray, ids: np.ndarray, keys: np.ndarray, k: int) -> np.ndarray:
    valid = ids >= 0
    valid_ids = ids[valid].astype(np.int64, copy=False)
    valid_keys = keys[valid]
    if valid_ids.size < k:
        raise ValueError("route witness has fewer than k valid probe IDs")
    if np.unique(valid_ids).size != valid_ids.size:
        raise ValueError("route witness contains duplicate probe IDs")
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(valid_keys)):
        raise ValueError("route witness contains non-finite query/key values")
    scores = valid_keys @ query
    # neuron ID is the deterministic ascending tie-break after descending score
    order = np.lexsort((valid_ids, -scores))
    return valid_ids[order[:k]]


def route_conformance(witness, actual: np.ndarray, ks: tuple[int, int, int]) -> tuple[int, int]:
    expected = np.asarray(witness["reference_output"], dtype=np.float32)
    matches = 0
    total = 0
    for layer, ((start, end), k) in enumerate(zip(QUERY_RANGES, ks), start=1):
        ids = np.asarray(witness[f"probe_ids_l{layer}"])
        keys = np.asarray(witness[f"probe_keys_l{layer}"], dtype=np.float32)
        residual = np.asarray(witness[f"query_residual_l{layer}"], dtype=np.float32)
        if ids.shape[:1] != expected.shape[:1] or keys.shape[:2] != ids.shape:
            raise ValueError(f"layer {layer} route witness shape mismatch")
        if keys.shape[2] != 64:
            raise ValueError(f"layer {layer} route keys must have width 64")
        if residual.shape != (expected.shape[0], 64):
            raise ValueError(f"layer {layer} query residual must be [N,64]")
        for row in range(expected.shape[0]):
            reference_ids = stable_route_ids(
                expected[row, start:end] + residual[row], ids[row], keys[row], k
            )
            actual_ids = stable_route_ids(
                actual[row, start:end] + residual[row], ids[row], keys[row], k
            )
            matches += int(np.array_equal(reference_ids, actual_ids))
            total += 1
    return matches, total


def compile_device(core: ov.Core, model_path: Path, device: str):
    if device not in core.available_devices:
        raise RuntimeError(f"required device {device} unavailable: {core.available_devices}")
    start = time.perf_counter_ns()
    compiled = core.compile_model(model_path, device)
    return compiled, time.perf_counter_ns() - start


def infer_all(compiled, packed: np.ndarray, warmups: int, repeats: int):
    request = compiled.create_infer_request()
    output_port = compiled.output(0)
    chunks = [np.ascontiguousarray(packed[i : i + BATCH]) for i in range(0, len(packed), BATCH)]
    for _ in range(warmups):
        for chunk in chunks:
            request.infer([chunk], share_inputs=True, share_outputs=True)
            np.asarray(request.get_tensor(output_port).data, dtype=np.float32).copy()
    samples: list[int] = []
    last_outputs: list[np.ndarray] = []
    for repeat in range(repeats):
        repeat_outputs: list[np.ndarray] = []
        for chunk in chunks:
            start = time.perf_counter_ns()
            request.infer([chunk], share_inputs=True, share_outputs=True)
            output = np.asarray(request.get_tensor(output_port).data, dtype=np.float32).copy()
            samples.append(time.perf_counter_ns() - start)
            repeat_outputs.append(output)
        if repeat == repeats - 1:
            last_outputs = repeat_outputs
    return np.concatenate(last_outputs, axis=0), samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-ir", type=Path, required=True)
    parser.add_argument("--snapshot-metadata", type=Path, required=True)
    parser.add_argument("--witness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--k", type=int, nargs=3, default=(26, 22, 22))
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError("refusing to overwrite benchmark output")
    if args.warmups < 1 or args.repeats < 10:
        raise ValueError("benchmark requires >=1 warmup and >=10 repeats")
    if any(k <= 0 for k in args.k):
        raise ValueError("all k values must be positive")
    metadata_bytes = args.snapshot_metadata.read_bytes()
    metadata = json.loads(metadata_bytes.decode("utf-8"))
    fixed_bin = args.fixed_ir.with_suffix(".bin")
    if metadata.get("schema") != SCHEMA:
        raise ValueError("snapshot metadata schema mismatch")
    if metadata.get("fixed_batch") != BATCH or metadata.get("output_features") != OUTPUTS:
        raise ValueError("snapshot metadata geometry mismatch")
    if not isinstance(metadata.get("snapshot_version"), int) or metadata["snapshot_version"] <= 0:
        raise ValueError("snapshot metadata version mismatch")
    for field in ("weights_sha256", "normalization_sha256", "xml_sha256", "bin_sha256"):
        digest = metadata.get(field)
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest.lower()
        ):
            raise ValueError(f"snapshot metadata has invalid {field}")
    if metadata.get("xml_sha256") != sha256_file(args.fixed_ir):
        raise ValueError("fixed IR XML digest mismatch")
    if metadata.get("bin_sha256") != sha256_file(fixed_bin):
        raise ValueError("fixed IR BIN digest mismatch")
    with np.load(args.witness, allow_pickle=False) as witness:
        packed = np.asarray(witness["input"], dtype=np.float32)
        reference = np.asarray(witness["reference_output"], dtype=np.float32)
        if packed.ndim != 2 or packed.shape[1] != INPUTS or len(packed) % BATCH:
            raise ValueError("input must be [positive multiple of 16, 559]")
        if packed.shape[0] == 0 or reference.shape != (packed.shape[0], OUTPUTS):
            raise ValueError("reference_output shape mismatch")
        if not np.all(np.isfinite(packed)) or not np.all(np.isfinite(reference)):
            raise ValueError("input/reference contains non-finite values")

        core = ov.Core()
        cpu, cpu_compile_ns = compile_device(core, args.fixed_ir, "CPU")
        npu, npu_compile_ns = compile_device(core, args.fixed_ir, "NPU")
        cpu_output, cpu_samples = infer_all(cpu, packed, args.warmups, args.repeats)
        npu_output, npu_samples = infer_all(npu, packed, args.warmups, args.repeats)
        if not np.all(np.isfinite(cpu_output)) or not np.all(np.isfinite(npu_output)):
            raise ValueError("OpenVINO output contains non-finite values")
        cpu_error = float(np.max(np.abs(cpu_output - reference)))
        npu_error = float(np.max(np.abs(npu_output - reference)))
        route_matches, route_total = route_conformance(witness, npu_output, tuple(args.k))

    cpu_p50 = int(statistics.median(cpu_samples))
    cpu_p95 = percentile(cpu_samples, 0.95)
    npu_p50 = int(statistics.median(npu_samples))
    npu_p95 = percentile(npu_samples, 0.95)
    cpu_rate = BATCH * 1.0e9 / cpu_p50
    npu_rate = BATCH * 1.0e9 / npu_p50
    speedup = npu_rate / cpu_rate
    gates = {
        "cpu_max_abs_le_1e_5": cpu_error <= 1.0e-5,
        "npu_max_abs_le_1e_2": npu_error <= 1.0e-2,
        "throughput_speedup_ge_1_15": speedup >= 1.15,
        "npu_p95_no_worse": npu_p95 <= cpu_p95,
        "route_ids_exact": route_matches == route_total and route_total > 0,
    }
    result = {
        "schema": "learned-hashstem-openvino-gate-v1",
        "status": (
            "hashstem_component_pass_pending_integrated_gate"
            if all(gates.values())
            else "fallback_cpu"
        ),
        "openvino_version": ov.__version__,
        "fixed_batch": BATCH,
        "snapshot_version": metadata.get("snapshot_version"),
        "master_version": metadata.get("master_version"),
        "weights_sha256": metadata.get("weights_sha256"),
        "snapshot_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "fixed_ir_xml_sha256": sha256_file(args.fixed_ir),
        "fixed_ir_bin_sha256": sha256_file(fixed_bin),
        "witness_sha256": sha256_file(args.witness),
        "k": list(args.k),
        "cpu_compile_ns": cpu_compile_ns,
        "npu_compile_ns": npu_compile_ns,
        "cpu_p50_ns": cpu_p50,
        "cpu_p95_ns": cpu_p95,
        "npu_p50_ns": npu_p50,
        "npu_p95_ns": npu_p95,
        "cpu_candidates_per_second": cpu_rate,
        "npu_candidates_per_second": npu_rate,
        "throughput_speedup": speedup,
        "cpu_maximum_absolute_error": cpu_error,
        "npu_maximum_absolute_error": npu_error,
        "route_id_matches": route_matches,
        "route_id_total": route_total,
        "gates": gates,
        "timed_boundary": "InferRequest input binding through copied FP32 output materialization",
        "note": "Full-system packing, sparse routing, top-1 action, and overlap gates remain mandatory before adoption.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] == "fallback_cpu":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
