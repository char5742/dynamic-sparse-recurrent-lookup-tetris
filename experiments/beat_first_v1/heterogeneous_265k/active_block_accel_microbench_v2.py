"""Bounded selected-active-block CPU/NPU/iGPU component microbenchmark v2.

UNEXECUTED STATIC IMPLEMENTATION.  Importing this file does not import NumPy or
OpenVINO and does not enumerate or execute a device.  The benchmark deliberately
starts *after* WTA/LSH routing: an immutable witness supplies ordered selected
IDs and the corresponding contiguous FP32 row blocks.  Every measured path must
materialize those blocks into fresh fixed-shape host buffers.

This is component timing, never model-strength, game-score, routing, NNUE, or
heterogeneous-system adoption evidence.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


CONTRACT_SCHEMA = "heterogeneous-265k-active-block-accelerator-microbench-contract-v2"
WITNESS_SCHEMA = "heterogeneous-265k-active-block-witness-v2"
RESULT_SCHEMA = "heterogeneous-265k-active-block-accelerator-result-v2"
TIMING_SCHEMA = "heterogeneous-265k-active-block-raw-timing-v2"
CALL_SCHEMA = "heterogeneous-265k-active-block-physical-call-v2"
SOURCE_STATUS = "UNEXECUTED_STATIC_IMPLEMENTATION"
ARTIFACT_STATUS = "EXECUTED_COMPONENT_MICROBENCH_V2"

BATCH = 16
LAYER_DIMS = (560, 321, 321)
MAX_COUNTS = (96, 80, 80)
VARIANTS: dict[str, tuple[int, int, int]] = {
    "k64": (24, 20, 20),
    "k128": (48, 40, 40),
    "k256": (96, 80, 80),
}
REQUIRED_ARRAYS = tuple(
    [f"selected_rows_l{layer}" for layer in range(1, 4)]
    + [f"selected_ids_l{layer}" for layer in range(1, 4)]
    + [f"scaled_input_l{layer}" for layer in range(1, 4)]
)
FORBIDDEN_DEVICE_TOKENS = ("AUTO", "HETERO", "MULTI", "BATCH")


class EvidenceError(RuntimeError):
    pass


class DeviceUnavailable(EvidenceError):
    def __init__(
        self,
        cell: str,
        requested: str,
        enumerated: list[str],
        reason: str,
        identity: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.cell = cell
        self.requested = requested
        self.enumerated = enumerated
        self.reason = reason
        self.identity = identity or {}

    def artifact(self) -> dict[str, Any]:
        return {
            "status": "UNAVAILABLE_FAIL_CLOSED",
            "cell": self.cell,
            "requested_device": self.requested,
            "enumerated_devices": self.enumerated,
            "device_identity": self.identity,
            "reason": self.reason,
        }


class QpcClock:
    """Raw Windows QPC; no wall-clock substitute is accepted."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise EvidenceError("active-block v2 requires Windows QueryPerformanceCounter")
        kernel32 = ctypes.WinDLL("Kernel32", use_last_error=True)
        self._counter = kernel32.QueryPerformanceCounter
        self._counter.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
        self._counter.restype = ctypes.c_int
        frequency = ctypes.c_longlong()
        query_frequency = kernel32.QueryPerformanceFrequency
        query_frequency.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
        query_frequency.restype = ctypes.c_int
        if query_frequency(ctypes.byref(frequency)) == 0 or frequency.value <= 0:
            raise EvidenceError("QueryPerformanceFrequency failed")
        self.frequency = int(frequency.value)

    def now(self) -> int:
        value = ctypes.c_longlong()
        if self._counter(ctypes.byref(value)) == 0:
            raise EvidenceError("QueryPerformanceCounter failed")
        return int(value.value)

    def to_ns(self, ticks: int) -> int:
        if ticks < 0:
            raise EvidenceError("negative QPC interval")
        return int(round(ticks * 1_000_000_000 / self.frequency))


@dataclass(frozen=True)
class DeviceIdentity:
    requested_device: str
    execution_devices: tuple[str, ...]
    full_device_name: str
    device_type: str
    driver_version: str
    device_uuid: str
    openvino_version: str

    def json(self) -> dict[str, Any]:
        return {
            "requested_device": self.requested_device,
            "execution_devices": list(self.execution_devices),
            "full_device_name": self.full_device_name,
            "device_type": self.device_type,
            "driver_version": self.driver_version,
            "device_uuid": self.device_uuid,
            "openvino_version": self.openvino_version,
        }


@dataclass
class CompiledLayer:
    variant: str
    layer: int
    dimension: int
    active_count: int
    model_name: str
    compiled: Any
    request: Any
    compile_ticks: int
    identity: DeviceIdentity


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(np: Any, array: Any) -> str:
    canonical = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()


def valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def nearest_rank(values: list[int], fraction: float) -> int:
    if not values:
        raise EvidenceError("cannot summarize an empty timing sample")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return int(ordered[index])


def atomic_write(path: Path, data: bytes) -> None:
    target = path.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite temporary file {temporary}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        for row in rows
    )


def measure(clock: QpcClock, operation: Callable[[], Any]) -> tuple[Any, tuple[int, int]]:
    begin = clock.now()
    value = operation()
    end = clock.now()
    if end < begin:
        raise EvidenceError("QPC moved backwards")
    return value, (begin, end)


def append_interval(stages: dict[str, list[list[int]]], name: str, interval: tuple[int, int]) -> None:
    stages.setdefault(name, []).append([int(interval[0]), int(interval[1])])


def interval_ticks(intervals: list[list[int]]) -> int:
    return sum(end - begin for begin, end in intervals)


def stage_totals(stages: dict[str, list[list[int]]]) -> dict[str, int]:
    return {name: interval_ticks(intervals) for name, intervals in stages.items()}


def stable_scalar_silu(np: Any, value: Any) -> Any:
    z = np.float32(value)
    if z >= np.float32(0.0):
        sigma = np.float32(1.0) / (np.float32(1.0) + np.exp(np.float32(-z)))
    else:
        exponential = np.exp(z)
        sigma = exponential / (np.float32(1.0) + exponential)
    return np.float32(z * sigma)


def scalar_oracle(np: Any, rows: Any, features: Any) -> Any:
    """Strict ordered FP32 reference; never used in a latency sample."""
    batch, active, dimension = rows.shape
    output = np.empty((batch, active), dtype=np.float32)
    for candidate in range(batch):
        for neuron in range(active):
            accumulator = np.float32(0.0)
            for coordinate in range(dimension):
                product = np.float32(rows[candidate, neuron, coordinate] * features[candidate, coordinate])
                accumulator = np.float32(accumulator + product)
            output[candidate, neuron] = stable_scalar_silu(np, accumulator)
    return output


def optimized_cpu(np: Any, rows: Any, features: Any) -> Any:
    preactivation = np.einsum(
        "bkd,bd->bk", rows, features, dtype=np.float32, optimize=False
    )
    exponential = np.exp(-np.abs(preactivation))
    sigmoid = np.where(
        preactivation >= np.float32(0.0),
        np.float32(1.0) / (np.float32(1.0) + exponential),
        exponential / (np.float32(1.0) + exponential),
    )
    return np.asarray(preactivation * sigmoid, dtype=np.float32)


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"{label} is not strict UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} root must be an object")
    return value, data


def load_contract(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    if not valid_sha256(expected_sha256):
        raise EvidenceError("contract SHA-256 must be lowercase hexadecimal")
    value, data = _read_json(path, "contract")
    observed = sha256_bytes(data)
    if observed != expected_sha256:
        raise EvidenceError("contract SHA-256 mismatch")
    if value.get("schema") != CONTRACT_SCHEMA or value.get("status") != "UNEXECUTED_STATIC_CONTRACT":
        raise EvidenceError("contract schema/status mismatch")
    return value, observed


def load_witness(np: Any, npz_path: Path, metadata_path: Path, contract_sha256: str):
    metadata, metadata_bytes = _read_json(metadata_path, "witness metadata")
    observed_npz_sha256 = sha256_file(npz_path)
    if metadata.get("schema") != WITNESS_SCHEMA:
        raise EvidenceError("witness metadata schema mismatch")
    if metadata.get("npz_sha256") != observed_npz_sha256:
        raise EvidenceError("witness NPZ SHA-256 mismatch")
    if metadata.get("contract_sha256") != contract_sha256:
        raise EvidenceError("witness does not bind the executing contract")
    if metadata.get("split") not in ("teacher_v3_train", "synthetic_component_only"):
        raise EvidenceError("witness split is not allowed for this component benchmark")
    if metadata.get("reserved_seed_free") is not True:
        raise EvidenceError("witness does not certify reserved-seed exclusion")
    if metadata.get("development_validation_sealed_seeds_used") is not False:
        raise EvidenceError("witness does not explicitly forbid development/validation/sealed seeds")
    source_digest = metadata.get("selection_source_sha256")
    if not valid_sha256(source_digest):
        raise EvidenceError("witness selection source digest is missing or invalid")

    archive = np.load(npz_path, allow_pickle=False)
    if set(archive.files) != set(REQUIRED_ARRAYS):
        archive.close()
        raise EvidenceError(f"witness keys must be exactly {sorted(REQUIRED_ARRAYS)}")
    arrays: dict[str, Any] = {}
    try:
        for name in REQUIRED_ARRAYS:
            arrays[name] = archive[name]
    finally:
        archive.close()

    count = int(arrays["scaled_input_l1"].shape[0])
    if count <= 0 or count % BATCH:
        raise EvidenceError("candidate count must be a positive multiple of 16")
    if metadata.get("candidate_count") != count:
        raise EvidenceError("metadata candidate count mismatch")
    for layer, (dimension, maximum) in enumerate(zip(LAYER_DIMS, MAX_COUNTS), start=1):
        rows = arrays[f"selected_rows_l{layer}"]
        ids = arrays[f"selected_ids_l{layer}"]
        features = arrays[f"scaled_input_l{layer}"]
        if rows.dtype != np.float32 or rows.shape != (count, maximum, dimension):
            raise EvidenceError(f"layer {layer} selected rows have wrong dtype/shape")
        if ids.dtype != np.int32 or ids.shape != (count, maximum):
            raise EvidenceError(f"layer {layer} selected IDs have wrong dtype/shape")
        if features.dtype != np.float32 or features.shape != (count, dimension):
            raise EvidenceError(f"layer {layer} scaled input has wrong dtype/shape")
        if not rows.flags.c_contiguous or not ids.flags.c_contiguous or not features.flags.c_contiguous:
            raise EvidenceError(f"layer {layer} witness arrays must be C-contiguous")
        if not np.all(np.isfinite(rows)) or not np.all(np.isfinite(features)):
            raise EvidenceError(f"layer {layer} witness contains non-finite FP32 values")
        if np.any(ids <= 0):
            raise EvidenceError(f"layer {layer} selected IDs must be positive Julia-style IDs")
        ordered = np.sort(ids, axis=1)
        if np.any(np.diff(ordered, axis=1) == 0):
            raise EvidenceError(f"layer {layer} selected IDs contain duplicates")
        rows.setflags(write=False)
        ids.setflags(write=False)
        features.setflags(write=False)
    return arrays, metadata, sha256_bytes(metadata_bytes), observed_npz_sha256


def _safe_property(core: Any, device: str, name: str) -> str:
    try:
        value = core.get_property(device, name)
    except Exception:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _device_family(device: str) -> str:
    return device.upper().split(".", 1)[0]


def _forbidden_device_name(device: str) -> bool:
    upper = device.upper()
    return any(token in upper for token in FORBIDDEN_DEVICE_TOKENS)


def select_npu(core: Any, requested: str | None) -> tuple[str, dict[str, str]]:
    enumerated = [str(item) for item in core.available_devices]
    candidates = [item for item in enumerated if _device_family(item) == "NPU"]
    if requested is not None:
        if requested not in enumerated or _device_family(requested) != "NPU":
            raise DeviceUnavailable("A1v2_cpu_pack_npu_active", requested, enumerated, "requested exact NPU was not enumerated")
        selected = requested
    elif len(candidates) == 1:
        selected = candidates[0]
    else:
        raise DeviceUnavailable("A1v2_cpu_pack_npu_active", "NPU", enumerated, "NPU is absent or ambiguous; use --npu-device for an exact enumerated device")
    if _forbidden_device_name(selected):
        raise DeviceUnavailable("A1v2_cpu_pack_npu_active", selected, enumerated, "composite or automatic devices are forbidden")
    identity = {
        "full_device_name": _safe_property(core, selected, "FULL_DEVICE_NAME"),
        "device_type": _safe_property(core, selected, "DEVICE_TYPE"),
        "driver_version": _safe_property(core, selected, "DRIVER_VERSION"),
        "device_uuid": _safe_property(core, selected, "DEVICE_UUID"),
    }
    return selected, identity


def select_integrated_intel_gpu(core: Any, requested: str | None) -> tuple[str, dict[str, str]]:
    enumerated = [str(item) for item in core.available_devices]
    candidates: list[tuple[str, dict[str, str]]] = []
    for device in enumerated:
        if _device_family(device) != "GPU" or _forbidden_device_name(device):
            continue
        identity = {
            "full_device_name": _safe_property(core, device, "FULL_DEVICE_NAME"),
            "device_type": _safe_property(core, device, "DEVICE_TYPE"),
            "driver_version": _safe_property(core, device, "DRIVER_VERSION"),
            "device_uuid": _safe_property(core, device, "DEVICE_UUID"),
        }
        evidence = (identity["full_device_name"] + " " + identity["device_type"]).upper()
        if "INTEL" in evidence and "INTEGRATED" in evidence:
            candidates.append((device, identity))
    if requested is not None:
        matches = [item for item in candidates if item[0] == requested]
        if len(matches) != 1:
            raise DeviceUnavailable(
                "A2v2_cpu_pack_intel_igpu_active", requested, enumerated,
                "requested device lacks exact Intel integrated-GPU identity evidence",
            )
        return matches[0]
    if len(candidates) != 1:
        raise DeviceUnavailable(
            "A2v2_cpu_pack_intel_igpu_active", "explicit Intel integrated GPU", enumerated,
            "integrated Intel GPU is absent or ambiguous; use --igpu-device only for an exact verified candidate",
            {"verified_candidates": [item[0] for item in candidates]},
        )
    return candidates[0]


def build_active_model(ov: Any, ops: Any, np: Any, variant: str, layer: int, active: int, dimension: int):
    model_name = f"active_block_v2_{variant}_l{layer}_b{BATCH}_k{active}_d{dimension}"
    rows = ops.parameter([BATCH, active, dimension], np.float32, "selected_rows")
    features = ops.parameter([BATCH, dimension], np.float32, "scaled_input")
    expanded = ops.unsqueeze(features, ops.constant(np.asarray([1], dtype=np.int64)))
    products = ops.multiply(rows, expanded)
    preactivation = ops.reduce_sum(
        products, ops.constant(np.asarray([2], dtype=np.int64)), False
    )
    activated = ops.multiply(preactivation, ops.sigmoid(preactivation))
    return ov.Model([ops.result(activated, "activation")], [rows, features], model_name), model_name


def _execution_devices(compiled: Any) -> tuple[str, ...]:
    try:
        devices = tuple(str(item) for item in compiled.get_property("EXECUTION_DEVICES"))
    except Exception as error:
        raise EvidenceError(f"cannot verify OpenVINO execution device: {error}") from error
    if len(devices) != 1 or _forbidden_device_name(devices[0]):
        raise EvidenceError(f"compiled graph has non-exclusive execution devices: {devices}")
    return devices


def compile_layers(
    ov: Any,
    ops: Any,
    np: Any,
    core: Any,
    clock: QpcClock,
    device: str,
    device_properties: dict[str, str],
    expected_family: str,
) -> tuple[dict[tuple[str, int], CompiledLayer], list[dict[str, Any]]]:
    compiled_layers: dict[tuple[str, int], CompiledLayer] = {}
    receipts: list[dict[str, Any]] = []
    for variant, counts in VARIANTS.items():
        for layer, (active, dimension) in enumerate(zip(counts, LAYER_DIMS), start=1):
            model, model_name = build_active_model(ov, ops, np, variant, layer, active, dimension)
            begin = clock.now()
            try:
                compiled = core.compile_model(model, device)
            except Exception as error:
                raise DeviceUnavailable(
                    "A1v2_cpu_pack_npu_active" if expected_family == "NPU" else "A2v2_cpu_pack_intel_igpu_active",
                    device,
                    [str(item) for item in core.available_devices],
                    f"compile failed for {model_name}: {error}",
                    device_properties,
                ) from error
            end = clock.now()
            actual = _execution_devices(compiled)
            if _device_family(actual[0]) != expected_family or actual[0].upper() != device.upper():
                raise DeviceUnavailable(
                    "A1v2_cpu_pack_npu_active" if expected_family == "NPU" else "A2v2_cpu_pack_intel_igpu_active",
                    device,
                    [str(item) for item in core.available_devices],
                    f"compiled execution device {actual} is not exactly requested device {device}",
                    device_properties,
                )
            identity = DeviceIdentity(
                requested_device=device,
                execution_devices=actual,
                full_device_name=device_properties["full_device_name"],
                device_type=device_properties["device_type"],
                driver_version=device_properties["driver_version"],
                device_uuid=device_properties["device_uuid"],
                openvino_version=str(ov.__version__),
            )
            compiled_layers[(variant, layer)] = CompiledLayer(
                variant=variant,
                layer=layer,
                dimension=dimension,
                active_count=active,
                model_name=model_name,
                compiled=compiled,
                request=compiled.create_infer_request(),
                compile_ticks=end - begin,
                identity=identity,
            )
            receipts.append({
                "variant": variant,
                "layer": layer,
                "model_name": model_name,
                "compile_begin_qpc": begin,
                "compile_end_qpc": end,
                "compile_ticks": end - begin,
                "device_identity": identity.json(),
            })
    return compiled_layers, receipts


def _buffers(np: Any, variant: str) -> dict[int, tuple[Any, Any]]:
    return {
        layer: (
            np.empty((BATCH, active, dimension), dtype=np.float32),
            np.empty((BATCH, dimension), dtype=np.float32),
        )
        for layer, (active, dimension) in enumerate(zip(VARIANTS[variant], LAYER_DIMS), start=1)
    }


def logical_bytes(variant: str) -> dict[str, int]:
    counts = VARIANTS[variant]
    selected = sum(BATCH * active * dimension * 4 for active, dimension in zip(counts, LAYER_DIMS))
    packed_inputs = sum(BATCH * dimension * 4 for dimension in LAYER_DIMS)
    copied_outputs = sum(BATCH * active * 4 for active in counts)
    return {
        "selected_block_materialization_bytes_per_sample": selected,
        "input_pack_bytes_per_sample": packed_inputs,
        "host_to_device_logical_bytes_per_sample": selected + packed_inputs,
        "device_to_host_logical_bytes_per_sample": copied_outputs,
        "output_copy_bytes_per_sample": copied_outputs,
        "physical_calls_per_sample": 3,
        "synchronizations_per_sample": 3,
    }


def _source_slice(arrays: dict[str, Any], variant: str, layer: int, batch_index: int):
    active = VARIANTS[variant][layer - 1]
    begin = batch_index * BATCH
    end = begin + BATCH
    rows = arrays[f"selected_rows_l{layer}"][begin:end, :active, :]
    ids = arrays[f"selected_ids_l{layer}"][begin:end, :active]
    features = arrays[f"scaled_input_l{layer}"][begin:end, :]
    return rows, ids, features


def precompute_reference(
    np: Any,
    arrays: dict[str, Any],
    oracle_candidates: int,
) -> tuple[dict[tuple[str, int, int], Any], dict[str, Any], dict[tuple[str, int, int], dict[str, str]]]:
    batch_count = arrays["scaled_input_l1"].shape[0] // BATCH
    references: dict[tuple[str, int, int], Any] = {}
    digests: dict[tuple[str, int, int], dict[str, str]] = {}
    oracle_errors: dict[str, Any] = {}
    for variant in VARIANTS:
        maximum_error = 0.0
        per_layer: dict[str, float] = {}
        for layer in range(1, 4):
            layer_error = 0.0
            for batch_index in range(batch_count):
                rows, ids, features = _source_slice(arrays, variant, layer, batch_index)
                vectorized = optimized_cpu(np, rows, features)
                if not np.all(np.isfinite(vectorized)):
                    raise EvidenceError("optimized CPU reference produced non-finite output")
                references[(variant, layer, batch_index)] = vectorized
                digests[(variant, layer, batch_index)] = {
                    "selected_block_sha256": sha256_array(np, rows),
                    "selected_ids_sha256": sha256_array(np, ids),
                    "input_sha256": sha256_array(np, features),
                }
                if batch_index == 0:
                    take = min(BATCH, oracle_candidates)
                    scalar = scalar_oracle(np, rows[:take], features[:take])
                    error = float(np.max(np.abs(vectorized[:take] - scalar)))
                    layer_error = max(layer_error, error)
                    maximum_error = max(maximum_error, error)
            per_layer[f"layer_{layer}"] = layer_error
        oracle_errors[variant] = {
            "maximum_absolute_error": maximum_error,
            "per_layer": per_layer,
            "gate_le_1e_5": maximum_error <= 1.0e-5,
            "oracle_candidates_per_layer": min(BATCH, oracle_candidates),
        }
    return references, oracle_errors, digests


def run_cpu_sample(
    np: Any,
    clock: QpcClock,
    arrays: dict[str, Any],
    variant: str,
    batch_index: int,
    buffers: dict[int, tuple[Any, Any]],
) -> tuple[dict[int, Any], dict[str, list[list[int]]], tuple[int, int]]:
    stages: dict[str, list[list[int]]] = {}
    outputs: dict[int, Any] = {}
    total_begin = clock.now()
    for layer in range(1, 4):
        source_rows, _, source_features = _source_slice(arrays, variant, layer, batch_index)
        rows_buffer, feature_buffer = buffers[layer]
        _, interval = measure(clock, lambda: np.copyto(rows_buffer, source_rows))
        append_interval(stages, "host_gather", interval)
        _, interval = measure(clock, lambda: np.copyto(feature_buffer, source_features))
        append_interval(stages, "input_pack", interval)
        computed, interval = measure(clock, lambda: optimized_cpu(np, rows_buffer, feature_buffer))
        append_interval(stages, "optimized_cpu_compute", interval)
        copied, interval = measure(clock, lambda: np.array(computed, dtype=np.float32, copy=True))
        append_interval(stages, "output_copy", interval)
        outputs[layer] = copied
    total_end = clock.now()
    return outputs, stages, (total_begin, total_end)


def run_accelerator_sample(
    ov: Any,
    np: Any,
    clock: QpcClock,
    arrays: dict[str, Any],
    digests: dict[tuple[str, int, int], dict[str, str]],
    compiled_layers: dict[tuple[str, int], CompiledLayer],
    variant: str,
    batch_index: int,
    buffers: dict[int, tuple[Any, Any]],
    run_uuid: str,
    cell: str,
    phase: str,
    sample_index: int,
    call_index_start: int,
) -> tuple[dict[int, Any], dict[str, list[list[int]]], tuple[int, int], list[dict[str, Any]], int]:
    stages: dict[str, list[list[int]]] = {}
    outputs: dict[int, Any] = {}
    receipts: list[dict[str, Any]] = []
    total_begin = clock.now()
    for layer in range(1, 4):
        source_rows, source_ids, source_features = _source_slice(arrays, variant, layer, batch_index)
        rows_buffer, feature_buffer = buffers[layer]
        _, interval = measure(clock, lambda: np.copyto(rows_buffer, source_rows))
        append_interval(stages, "host_gather", interval)
        _, interval = measure(clock, lambda: np.copyto(feature_buffer, source_features))
        append_interval(stages, "input_pack", interval)
        compiled = compiled_layers[(variant, layer)]

        def bind() -> None:
            rows_tensor = ov.Tensor(rows_buffer, shared_memory=True)
            feature_tensor = ov.Tensor(feature_buffer, shared_memory=True)
            compiled.request.set_input_tensor(0, rows_tensor)
            compiled.request.set_input_tensor(1, feature_tensor)

        _, bind_interval = measure(clock, bind)
        append_interval(stages, "tensor_bind", bind_interval)
        _, submit_interval = measure(clock, compiled.request.start_async)
        append_interval(stages, "submit", submit_interval)
        _, wait_interval = measure(clock, compiled.request.wait)
        append_interval(stages, "wait_and_synchronize", wait_interval)

        def copy_output():
            return np.array(compiled.request.get_output_tensor(0).data, dtype=np.float32, copy=True)

        output, copy_interval = measure(clock, copy_output)
        append_interval(stages, "output_copy", copy_interval)
        outputs[layer] = output
    total_end = clock.now()
    call_index = call_index_start + 3

    # Numerical validation, hashing, and JSON receipt construction are evidence
    # work outside the latency boundary.  The copied output itself remains the
    # final timed operation.
    for layer in range(1, 4):
        _, source_ids, _ = _source_slice(arrays, variant, layer, batch_index)
        rows_buffer, feature_buffer = buffers[layer]
        output = outputs[layer]
        compiled = compiled_layers[(variant, layer)]
        bind_interval = stages["tensor_bind"][layer - 1]
        submit_interval = stages["submit"][layer - 1]
        wait_interval = stages["wait_and_synchronize"][layer - 1]
        copy_interval = stages["output_copy"][layer - 1]
        physical_call_index = call_index_start + layer
        expected_shape = (BATCH, VARIANTS[variant][layer - 1])
        if output.shape != expected_shape or not np.all(np.isfinite(output)):
            raise EvidenceError(f"{cell} {variant} layer {layer} output is malformed")
        receipt_digests = digests[(variant, layer, batch_index)]
        receipts.append({
            "schema": CALL_SCHEMA,
            "implementation_status": ARTIFACT_STATUS,
            "run_uuid": run_uuid,
            "phase": phase,
            "backend_cell": cell,
            "requested_device": compiled.identity.requested_device,
            "execution_devices": list(compiled.identity.execution_devices),
            "full_device_name": compiled.identity.full_device_name,
            "device_type": compiled.identity.device_type,
            "driver_version": compiled.identity.driver_version,
            "device_uuid": compiled.identity.device_uuid,
            "openvino_version": compiled.identity.openvino_version,
            "variant": variant,
            "layer": layer,
            "batch_index": batch_index,
            "sample_index": sample_index,
            "physical_call_index": physical_call_index,
            "model_name": compiled.model_name,
            **receipt_digests,
            "input_shape": list(feature_buffer.shape),
            "selected_rows_shape": list(rows_buffer.shape),
            "selected_ids_shape": list(source_ids.shape),
            "output_shape": list(output.shape),
            "host_to_device_bytes": int(rows_buffer.nbytes + feature_buffer.nbytes),
            "device_to_host_bytes": int(output.nbytes),
            "synchronization_count": 1,
            "bind_begin_qpc": bind_interval[0],
            "bind_end_qpc": bind_interval[1],
            "submit_begin_qpc": submit_interval[0],
            "submit_end_qpc": submit_interval[1],
            "wait_begin_qpc": wait_interval[0],
            "wait_end_qpc": wait_interval[1],
            "output_copy_begin_qpc": copy_interval[0],
            "output_copy_end_qpc": copy_interval[1],
            "output_sha256": sha256_array(np, output),
            "transfer_timing_semantics": "H2D/device/D2H visibility is opaque inside submit+wait; end-to-end total includes it",
        })
    return outputs, stages, (total_begin, total_end), receipts, call_index


def summarize_samples(clock: QpcClock, samples: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [sample["total"][1] - sample["total"][0] for sample in samples]
    stage_names = sorted({name for sample in samples for name in sample["stage_ticks"]})
    stage_summary: dict[str, Any] = {}
    for name in stage_names:
        values = [sample["stage_ticks"].get(name, 0) for sample in samples]
        stage_summary[name] = {
            "p50_ticks": nearest_rank(values, 0.50),
            "p95_ticks": nearest_rank(values, 0.95),
            "p50_ns": clock.to_ns(nearest_rank(values, 0.50)),
            "p95_ns": clock.to_ns(nearest_rank(values, 0.95)),
        }
    p50 = nearest_rank(totals, 0.50)
    p95 = nearest_rank(totals, 0.95)
    if p50 <= 0 or p95 <= 0:
        raise EvidenceError("QPC total timing is empty")
    return {
        "samples": len(samples),
        "p50_ticks": p50,
        "p95_ticks": p95,
        "p50_ns": clock.to_ns(p50),
        "p95_ns": clock.to_ns(p95),
        "batch16_bundles_per_second_at_p50": clock.frequency / p50,
        "candidates_per_second_at_p50": BATCH * clock.frequency / p50,
        "stages": stage_summary,
    }


def raw_timing_row(
    run_uuid: str,
    cell: str,
    variant: str,
    sample_index: int,
    batch_index: int,
    total: tuple[int, int],
    stages: dict[str, list[list[int]]],
    frequency: int,
) -> dict[str, Any]:
    for intervals in stages.values():
        for begin, end in intervals:
            if not (total[0] <= begin <= end <= total[1]):
                raise EvidenceError("stage interval lies outside total timing boundary")
    return {
        "schema": TIMING_SCHEMA,
        "implementation_status": ARTIFACT_STATUS,
        "run_uuid": run_uuid,
        "backend_cell": cell,
        "variant": variant,
        "sample_index": sample_index,
        "batch_index": batch_index,
        "qpc_frequency": frequency,
        "total": [total[0], total[1]],
        "stage_intervals": stages,
        "stage_ticks": stage_totals(stages),
    }


def benchmark_cpu(
    np: Any,
    clock: QpcClock,
    arrays: dict[str, Any],
    references: dict[tuple[str, int, int], Any],
    run_uuid: str,
    warmups: int,
    repeats: int,
    raw_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    batch_count = arrays["scaled_input_l1"].shape[0] // BATCH
    result: dict[str, Any] = {
        "cell": "A0v2_cpu_selected_dot_silu",
        "backend": "optimized NumPy CPU",
        "scalar_oracle_timed": False,
        "variants": {},
    }
    for variant in VARIANTS:
        buffers = _buffers(np, variant)
        first_outputs, _, first_total = run_cpu_sample(
            np, clock, arrays, variant, 0, buffers
        )
        for layer in range(1, 4):
            if not np.array_equal(first_outputs[layer], references[(variant, layer, 0)]):
                raise EvidenceError("optimized CPU first call differs from precomputed reference")
        for warmup in range(warmups):
            run_cpu_sample(np, clock, arrays, variant, warmup % batch_count, buffers)
        samples: list[dict[str, Any]] = []
        for sample_index in range(repeats):
            batch_index = sample_index % batch_count
            outputs, stages, total = run_cpu_sample(
                np, clock, arrays, variant, batch_index, buffers
            )
            for layer in range(1, 4):
                if not np.array_equal(outputs[layer], references[(variant, layer, batch_index)]):
                    raise EvidenceError("timed optimized CPU output differs from frozen CPU reference")
            row = raw_timing_row(
                run_uuid,
                "A0v2_cpu_selected_dot_silu",
                variant,
                sample_index,
                batch_index,
                total,
                stages,
                clock.frequency,
            )
            raw_rows.append(row)
            samples.append(row)
        result["variants"][variant] = {
            **summarize_samples(clock, samples),
            "first_call_ticks": first_total[1] - first_total[0],
            "first_call_ns": clock.to_ns(first_total[1] - first_total[0]),
            "active_counts": list(VARIANTS[variant]),
            "logical_bytes": logical_bytes(variant),
        }
    return result


def benchmark_accelerator(
    ov: Any,
    ops: Any,
    np: Any,
    core: Any,
    clock: QpcClock,
    arrays: dict[str, Any],
    references: dict[tuple[str, int, int], Any],
    digests: dict[tuple[str, int, int], dict[str, str]],
    run_uuid: str,
    cell: str,
    device: str,
    device_properties: dict[str, str],
    family: str,
    warmups: int,
    repeats: int,
    raw_rows: list[dict[str, Any]],
    call_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    compiled_layers, compile_receipts = compile_layers(
        ov, ops, np, core, clock, device, device_properties, family
    )
    batch_count = arrays["scaled_input_l1"].shape[0] // BATCH
    result: dict[str, Any] = {
        "cell": cell,
        "requested_device": device,
        "device_identity": next(iter(compiled_layers.values())).identity.json(),
        "compile_receipts": compile_receipts,
        "variants": {},
    }
    call_index = len(call_rows)
    for variant in VARIANTS:
        buffers = _buffers(np, variant)
        first_outputs, _, first_total, receipts, call_index = run_accelerator_sample(
            ov, np, clock, arrays, digests, compiled_layers, variant, 0, buffers,
            run_uuid, cell, "first_call", 0, call_index,
        )
        call_rows.extend(receipts)
        maximum_error = 0.0
        for layer in range(1, 4):
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(first_outputs[layer] - references[(variant, layer, 0)]))),
            )
        for warmup in range(warmups):
            batch_index = warmup % batch_count
            _, _, _, receipts, call_index = run_accelerator_sample(
                ov, np, clock, arrays, digests, compiled_layers, variant, batch_index,
                buffers, run_uuid, cell, "warmup", warmup, call_index,
            )
            call_rows.extend(receipts)
        samples: list[dict[str, Any]] = []
        for sample_index in range(repeats):
            batch_index = sample_index % batch_count
            outputs, stages, total, receipts, call_index = run_accelerator_sample(
                ov, np, clock, arrays, digests, compiled_layers, variant, batch_index,
                buffers, run_uuid, cell, "timed", sample_index, call_index,
            )
            call_rows.extend(receipts)
            for layer in range(1, 4):
                maximum_error = max(
                    maximum_error,
                    float(np.max(np.abs(outputs[layer] - references[(variant, layer, batch_index)]))),
                )
            row = raw_timing_row(
                run_uuid, cell, variant, sample_index, batch_index, total, stages,
                clock.frequency,
            )
            raw_rows.append(row)
            samples.append(row)
        expected_calls = 3 * (1 + warmups + repeats)
        observed_calls = sum(
            1 for row in call_rows
            if row["backend_cell"] == cell and row["variant"] == variant
        )
        result["variants"][variant] = {
            **summarize_samples(clock, samples),
            "first_call_ticks": first_total[1] - first_total[0],
            "first_call_ns": clock.to_ns(first_total[1] - first_total[0]),
            "active_counts": list(VARIANTS[variant]),
            "logical_bytes": logical_bytes(variant),
            "maximum_absolute_error_vs_optimized_cpu": maximum_error,
            "numeric_gate_le_1e_2": maximum_error <= 1.0e-2,
            "expected_physical_calls": expected_calls,
            "observed_physical_calls": observed_calls,
            "physical_call_count_exact": observed_calls == expected_calls,
        }
    return result


def add_comparisons(cpu: dict[str, Any], backend: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for variant in VARIANTS:
        control = cpu["variants"][variant]
        candidate = backend["variants"][variant]
        speedup = control["p50_ticks"] / candidate["p50_ticks"]
        p95_ratio = candidate["p95_ticks"] / control["p95_ticks"]
        gates = {
            "speedup_ge_1_15": speedup >= 1.15,
            "p95_ratio_le_1_0": p95_ratio <= 1.0,
            "numeric_le_1e_2": candidate["numeric_gate_le_1e_2"],
            "physical_calls_exact": candidate["physical_call_count_exact"],
        }
        comparisons[variant] = {
            "end_to_end_p50_speedup_vs_optimized_cpu": speedup,
            "end_to_end_p95_ratio_vs_optimized_cpu": p95_ratio,
            "gates": gates,
            "component_gate_pass_pending_integrated_benchmark": all(gates.values()),
        }
    return comparisons


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--contract-sha256", required=True)
    value.add_argument("--witness", type=Path, required=True)
    value.add_argument("--witness-metadata", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--raw-timings", type=Path, required=True)
    value.add_argument("--physical-call-receipts", type=Path, required=True)
    value.add_argument("--warmups", type=int, default=10)
    value.add_argument("--repeats", type=int, default=100)
    value.add_argument("--oracle-candidates", type=int, default=16)
    value.add_argument("--npu-device")
    value.add_argument("--igpu-device")
    return value


def execute(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], int]:
    if args.warmups < 3 or args.repeats < 20:
        raise EvidenceError("benchmark requires at least 3 warmups and 20 timed samples")
    if not 1 <= args.oracle_candidates <= BATCH:
        raise EvidenceError("oracle candidates must be in 1:16")
    contract, contract_sha256 = load_contract(args.contract.resolve(), args.contract_sha256)
    # Heavy packages are imported only after all static bindings and output
    # freshness checks have passed.
    import numpy as np
    try:
        import openvino as ov
        from openvino import opset13 as ops
    except (ImportError, OSError) as error:
        raise DeviceUnavailable(
            "A1v2/A2v2_openvino_runtime",
            "OpenVINO NPU plus explicit Intel integrated GPU",
            [],
            f"OpenVINO runtime import failed: {error}",
        ) from error

    arrays, witness_metadata, witness_metadata_sha256, witness_sha256 = load_witness(
        np, args.witness.resolve(), args.witness_metadata.resolve(), contract_sha256
    )
    clock = QpcClock()
    run_uuid = str(uuid.uuid4())
    references, oracle_evidence, digests = precompute_reference(
        np, arrays, args.oracle_candidates
    )
    if not all(value["gate_le_1e_5"] for value in oracle_evidence.values()):
        raise EvidenceError("optimized CPU implementation failed the scalar numerical oracle")

    raw_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    cpu = benchmark_cpu(
        np, clock, arrays, references, run_uuid, args.warmups, args.repeats, raw_rows
    )
    core = ov.Core()
    enumerated = [str(item) for item in core.available_devices]
    accelerator_results: dict[str, Any] = {}
    unavailable: list[dict[str, Any]] = []

    accelerator_specs: list[tuple[str, str, dict[str, str], str]] = []
    try:
        npu_device, npu_identity = select_npu(core, args.npu_device)
        accelerator_specs.append(("A1v2_cpu_pack_npu_active", npu_device, npu_identity, "NPU"))
    except DeviceUnavailable as error:
        unavailable.append(error.artifact())
    try:
        gpu_device, gpu_identity = select_integrated_intel_gpu(core, args.igpu_device)
        accelerator_specs.append(("A2v2_cpu_pack_intel_igpu_active", gpu_device, gpu_identity, "GPU"))
    except DeviceUnavailable as error:
        unavailable.append(error.artifact())

    for cell, device, identity, family in accelerator_specs:
        try:
            backend = benchmark_accelerator(
                ov, ops, np, core, clock, arrays, references, digests, run_uuid,
                cell, device, identity, family, args.warmups, args.repeats,
                raw_rows, call_rows,
            )
            backend["comparisons_vs_cpu"] = add_comparisons(cpu, backend)
            accelerator_results[cell] = backend
        except DeviceUnavailable as error:
            unavailable.append(error.artifact())
        except Exception as error:
            unavailable.append(DeviceUnavailable(
                cell, device, enumerated,
                f"runtime or evidence failure: {type(error).__name__}: {error}", identity,
            ).artifact())

    any_component_rejection = any(
        not comparison["component_gate_pass_pending_integrated_benchmark"]
        for backend in accelerator_results.values()
        for comparison in backend["comparisons_vs_cpu"].values()
    )
    if unavailable:
        status = "UNAVAILABLE_FAIL_CLOSED"
        exit_code = 2
    elif any_component_rejection:
        status = "REJECT_COMPONENT_OFFLOAD"
        exit_code = 3
    else:
        status = "COMPONENT_PASS_PENDING_INTEGRATED_BENCHMARK"
        exit_code = 0
    result = {
        "schema": RESULT_SCHEMA,
        "implementation_status": ARTIFACT_STATUS,
        "status": status,
        "exit_code": exit_code,
        "run_uuid": run_uuid,
        "created_utc": utc_now(),
        "evidence_class": "COMPONENT_MICROBENCH_ONLY",
        "promotion_authority": "NONE",
        "model_strength_evidence": False,
        "routing_search_timed": False,
        "nnue_or_search_reuse": False,
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "contract_sha256": contract_sha256,
        "contract_schema": contract["schema"],
        "witness_sha256": witness_sha256,
        "witness_metadata_sha256": witness_metadata_sha256,
        "witness_split": witness_metadata["split"],
        "selection_source_sha256": witness_metadata["selection_source_sha256"],
        "qpc_frequency": clock.frequency,
        "warmups": args.warmups,
        "timed_samples": args.repeats,
        "enumerated_openvino_devices": enumerated,
        "openvino_version": str(ov.__version__),
        "numpy_version": str(np.__version__),
        "scalar_oracle_evidence": oracle_evidence,
        "cpu_baseline": cpu,
        "accelerator_results": accelerator_results,
        "unavailable_backends": unavailable,
        "transfer_timing_semantics": "logical FP32 bytes are exact; H2D/device/D2H visibility remains included but opaque inside submit+wait",
        "prohibited_claims": [
            "old model beaten", "new model stronger", "routing accelerated",
            "heterogeneous system adopted", "NNUE accelerated",
        ],
    }
    return result, raw_rows, call_rows, exit_code


def main(arguments: list[str] | None = None) -> int:
    args = parser().parse_args(arguments)
    outputs = (args.output.resolve(), args.raw_timings.resolve(), args.physical_call_receipts.resolve())
    if len(set(outputs)) != 3:
        raise EvidenceError("summary, timing, and physical-call outputs must be distinct")
    for output in outputs:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    try:
        result, raw_rows, call_rows, exit_code = execute(args)
    except Exception as error:
        result = {
            "schema": RESULT_SCHEMA,
            "implementation_status": ARTIFACT_STATUS,
            "status": "UNAVAILABLE_FAIL_CLOSED" if isinstance(error, DeviceUnavailable) else "REJECT_COMPONENT_OFFLOAD",
            "exit_code": 2 if isinstance(error, DeviceUnavailable) else 3,
            "created_utc": utc_now(),
            "evidence_class": "COMPONENT_MICROBENCH_ONLY",
            "promotion_authority": "NONE",
            "failure_type": type(error).__name__,
            "failure_reason": str(error),
            "source_sha256": sha256_file(Path(__file__).resolve()),
        }
        raw_rows = []
        call_rows = []
        exit_code = int(result["exit_code"])
    raw_data = jsonl_bytes(raw_rows)
    call_data = jsonl_bytes(call_rows)
    atomic_write(args.raw_timings, raw_data)
    atomic_write(args.physical_call_receipts, call_data)
    result["raw_timing_artifact"] = {
        "path": args.raw_timings.name,
        "bytes": len(raw_data),
        "sha256": sha256_bytes(raw_data),
        "rows": len(raw_rows),
    }
    result["physical_call_artifact"] = {
        "path": args.physical_call_receipts.name,
        "bytes": len(call_data),
        "sha256": sha256_bytes(call_data),
        "rows": len(call_rows),
    }
    atomic_write(args.output, json_bytes(result))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
