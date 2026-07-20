"""Fail-closed native-Windows PyTorch-XPU Learned HashStem training probe.

This is a component experiment for the heterogeneous 265K track.  It cannot
authorize iGPU training, publish an inference snapshot, provide pure-CPU sparse
model evidence, or provide game-strength evidence.  In particular, the later
integrated P-core sparse slowdown gate remains external to this process.

The module is intentionally inert when imported.  Third-party imports happen
only after command-line parsing so that a missing PyTorch/XPU runtime can still
produce the required structured ``UNAVAILABLE_FAIL_CLOSED`` receipt.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROBE_SCHEMA = "heterogeneous-265k-igpu-hashstem-training-probe-result-v1"
CONTRACT_SCHEMA = "heterogeneous-265k-igpu-hashstem-training-probe-contract-v1"
SYSTEM_SCHEMA = "heterogeneous-265k-native-windows-system-contract-v1"
WITNESS_SCHEMA = "igpu-hashstem-training-witness-v1"
HASHSTEM_SCHEMA = "learned-hashstem-v1-conv-dw-pw-pool-1039x214-relu"
MASTER_SCHEMA = "learned-hashstem-master-checkpoint-v1"
SNAPSHOT_SOURCE_SCHEMA = "learned-hashstem-snapshot-source-v1"
SNAPSHOT_SOURCE_STATUS = "UNEXECUTED_STATIC_ONLY"
SNAPSHOT_WEIGHTS_FILENAME = "hashstem_snapshot_weights.npz"

CPU_SET_INFORMATION_TYPE = 0
CPU_SET_MINIMUM_RECORD_SIZE = 32
ERROR_INSUFFICIENT_BUFFER = 122

BATCH = 16
INPUTS = 559
OUTPUTS = 256
PARAMETERS = 223_504
AUXILIARY_START = 192
AUXILIARY_END = 202

LEARNING_RATE = 3.0e-4
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1.0e-8
WEIGHT_DECAY = 1.0e-4

CPU_OUTPUT_ATOL = 1.0e-5
LOSS_ATOL = 1.0e-4
GRADIENT_COSINE_MIN = 0.999
GRADIENT_RELATIVE_L2_MAX = 1.0e-3
GRADIENT_MAX_ABS = 1.0e-4
UPDATED_MAX_ABS = 1.0e-4
SPEEDUP_MIN = 1.15
P95_RATIO_MAX = 1.0

TRAINABLE_SHAPES = {
    "conv3": (16, 2, 3, 3),
    "conv3_bias": (16,),
    "depthwise5x1": (16, 5),
    "depthwise_bias": (16,),
    "pointwise": (32, 16),
    "pointwise_bias": (32,),
    "dense": (214, 1039),
    "dense_bias": (214,),
}
WEIGHT_SHAPES = {
    **TRAINABLE_SHAPES,
    "input_mean": (INPUTS,),
    "input_inv_std": (INPUTS,),
}
WITNESS_BASE_SHAPES = {
    "packed": (None, INPUTS),
    "output_cotangent": (None, OUTPUTS),
    "auxiliary_target": (None, 10),
    "auxiliary_mask": (None, 10),
    "state_mask": (None,),
    "reference_output": (BATCH, OUTPUTS),
    "reference_loss": (1,),
}
STAGES = (
    "h2d_submit",
    "h2d_sync",
    "zero_grad",
    "forward_submit",
    "forward_sync",
    "backward_submit",
    "backward_sync",
    "optimizer_submit",
    "optimizer_sync",
    "d2h_submit",
    "d2h_sync",
    "total",
)


class BackendUnavailable(RuntimeError):
    """The exact native-Windows XPU configuration cannot be proven usable."""


class ConfigurationRejected(RuntimeError):
    """An input/provenance/numeric contract was violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ConfigurationRejected(f"{label} must be a 64-digit SHA-256")
    lowered = value.lower()
    if any(character not in "0123456789abcdef" for character in lowered):
        raise ConfigurationRejected(f"{label} must be hexadecimal")
    return lowered


def strict_json_loads(text: str, label: str) -> Any:
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ConfigurationRejected(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ConfigurationRejected(f"{label} contains non-finite JSON constant {value}")

    try:
        return json.loads(text, object_pairs_hook=pairs_hook, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise ConfigurationRejected(f"{label} is invalid JSON: {exc}") from exc


def read_bound_json(path: Path, expected_sha256: str, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ConfigurationRejected(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != require_sha256(expected_sha256, f"expected {label} SHA-256"):
        raise ConfigurationRejected(f"{label} SHA-256 mismatch")
    try:
        value = strict_json_loads(path.read_text(encoding="utf-8"), label)
    except UnicodeDecodeError as exc:
        raise ConfigurationRejected(f"{label} is not canonical UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationRejected(f"{label} must be a JSON object")
    return value, observed


def write_fresh_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def nearest_rank(values: Iterable[int], fraction: float) -> int:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise ConfigurationRejected("cannot compute a percentile of an empty sample")
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize_timings(raw: list[dict[str, int]]) -> dict[str, Any]:
    if not raw:
        raise ConfigurationRejected("timing receipt is empty")
    result: dict[str, Any] = {}
    for stage in STAGES:
        values = [sample[stage + "_ns"] for sample in raw]
        result[stage] = {
            "p50_ns": nearest_rank(values, 0.50),
            "p95_ns": nearest_rank(values, 0.95),
            "sum_ns": sum(values),
        }
    total_ns = sum(sample["total_ns"] for sample in raw)
    result["updates_per_second"] = len(raw) * 1.0e9 / total_ns
    result["p50_reciprocal_updates_per_second"] = 1.0e9 / result["total"]["p50_ns"]
    return result


def _normalization_sha256(np, weights: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in ("input_mean", "input_inv_std"):
        digest.update(np.asarray(weights[name], dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def load_weights(np, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationRejected(f"HashStem NPZ is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(WEIGHT_SHAPES):
            raise ConfigurationRejected("HashStem NPZ key set changed")
        weights = {
            name: np.array(archive[name], dtype=np.float32, copy=True)
            for name in WEIGHT_SHAPES
        }
    for name, shape in WEIGHT_SHAPES.items():
        if weights[name].shape != shape:
            raise ConfigurationRejected(f"{name} shape {weights[name].shape} != {shape}")
        if not np.all(np.isfinite(weights[name])):
            raise ConfigurationRejected(f"{name} contains non-finite values")
        weights[name].setflags(write=False)
    if not np.all(weights["input_inv_std"] > 0):
        raise ConfigurationRejected("input_inv_std must be strictly positive")
    return weights


def load_master_checkpoint(np, root: Path) -> dict[str, Any]:
    manifest_path = root / "checkpoint_manifest.json"
    metadata_path = root / "master_metadata.json"
    weights_path = root / "hashstem_master_weights.npz"
    optimizer_path = root / "hashstem_optimizer.jld2"
    for path in (manifest_path, metadata_path, weights_path, optimizer_path):
        if not path.is_file():
            raise ConfigurationRejected(f"master checkpoint member is missing: {path}")
    manifest = strict_json_loads(
        manifest_path.read_text(encoding="utf-8"), "master checkpoint manifest"
    )
    metadata = strict_json_loads(
        metadata_path.read_text(encoding="utf-8"), "master checkpoint metadata"
    )
    if manifest.get("schema") != MASTER_SCHEMA or metadata.get("schema") != MASTER_SCHEMA:
        raise ConfigurationRejected("master checkpoint schema mismatch")
    fixed_members = {
        "metadata_file": ("master_metadata.json", metadata_path, "metadata_sha256"),
        "weights_file": ("hashstem_master_weights.npz", weights_path, "weights_sha256"),
        "optimizer_file": ("hashstem_optimizer.jld2", optimizer_path, "optimizer_sha256"),
    }
    for filename_field, (expected_name, path, digest_field) in fixed_members.items():
        if manifest.get(filename_field) != expected_name:
            raise ConfigurationRejected(f"master {filename_field} changed")
        expected_digest = require_sha256(manifest.get(digest_field), f"master {digest_field}")
        if sha256_file(path) != expected_digest:
            raise ConfigurationRejected(f"master {expected_name} digest mismatch")
    for field in ("weights_sha256", "optimizer_sha256"):
        if metadata.get(field) != manifest.get(field):
            raise ConfigurationRejected(f"master metadata/manifest {field} mismatch")
    if metadata.get("hashstem_schema") != HASHSTEM_SCHEMA:
        raise ConfigurationRejected("master HashStem graph schema mismatch")
    if metadata.get("runtime_validation_status") != "VALIDATED_RUNTIME":
        raise ConfigurationRejected("master is not explicitly runtime validated")
    if any(int(metadata.get(field, -1)) != 0 for field in (
        "master_version", "optimizer_step", "backend_updates",
    )):
        raise ConfigurationRejected(
            "v1 iGPU probe accepts only a version-zero CPU master so AdamW state is exact"
        )
    if int(metadata.get("last_snapshot_version", 0)) != 0 or int(
        metadata.get("last_snapshot_master_version", 0)
    ) != 0:
        raise ConfigurationRejected("version-zero source master must not claim a published snapshot")
    if not math.isclose(
        float(metadata.get("learning_rate", float("nan"))), LEARNING_RATE,
        rel_tol=0.0, abs_tol=1.0e-11,
    ):
        raise ConfigurationRejected("master learning rate differs from probe contract")
    if not math.isclose(
        float(metadata.get("weight_decay", float("nan"))), WEIGHT_DECAY,
        rel_tol=0.0, abs_tol=1.0e-11,
    ):
        raise ConfigurationRejected("master weight decay differs from probe contract")
    weights = load_weights(np, weights_path)
    if sha256_file(weights_path) != metadata.get("weights_sha256"):
        raise ConfigurationRejected("master metadata weight digest mismatch")
    if _normalization_sha256(np, weights) != metadata.get("normalization_sha256"):
        raise ConfigurationRejected("master normalization digest mismatch")
    return {
        "root": root.resolve(),
        "manifest": manifest,
        "metadata": metadata,
        "manifest_path": manifest_path.resolve(),
        "manifest_sha256": sha256_file(manifest_path),
        "metadata_sha256": sha256_file(metadata_path),
        "weights_path": weights_path.resolve(),
        "weights_sha256": sha256_file(weights_path),
        "optimizer_sha256": sha256_file(optimizer_path),
        "weights": weights,
    }


def load_snapshot_binding(np, path: Path, expected_sha256: str, weights_path: Path,
                          expected_weights_sha256: str, master: dict[str, Any]) -> dict[str, Any]:
    snapshot, digest = read_bound_json(path, expected_sha256, "source snapshot metadata")
    if snapshot.get("schema") != SNAPSHOT_SOURCE_SCHEMA:
        raise ConfigurationRejected("source snapshot schema mismatch")
    exact_exporter_contract = {
        "hashstem_schema": HASHSTEM_SCHEMA,
        "status": SNAPSHOT_SOURCE_STATUS,
        "immutable_source": True,
        "openvino_compiled": False,
        "publish_authorized": False,
        "weights_file": SNAPSHOT_WEIGHTS_FILENAME,
        "source_checkpoint_manifest_sha256": master["manifest_sha256"],
    }
    for field, expected in exact_exporter_contract.items():
        observed = snapshot.get(field)
        if type(observed) is not type(expected) or observed != expected:
            raise ConfigurationRejected(
                f"source snapshot exporter contract mismatch: {field}"
            )
    if weights_path.name != SNAPSHOT_WEIGHTS_FILENAME:
        raise ConfigurationRejected("source snapshot weights filename mismatch")
    metadata = master["metadata"]
    for field, expected in (
        ("model_id", metadata.get("model_id")),
        ("master_version", 0),
        ("weights_sha256", master["weights_sha256"]),
        ("normalization_sha256", metadata.get("normalization_sha256")),
    ):
        observed = snapshot.get(field)
        if type(observed) is not type(expected) or observed != expected:
            raise ConfigurationRejected(f"source snapshot/master {field} mismatch")
    if type(snapshot.get("snapshot_version")) is not int or snapshot["snapshot_version"] <= 0:
        raise ConfigurationRejected("source snapshot version must be positive")
    observed_weights_sha256 = sha256_file(weights_path) if weights_path.is_file() else None
    if observed_weights_sha256 != require_sha256(
        expected_weights_sha256, "expected source snapshot weights SHA-256"
    ):
        raise ConfigurationRejected("source snapshot weights SHA-256 mismatch")
    if snapshot.get("weights_sha256") != observed_weights_sha256:
        raise ConfigurationRejected("source snapshot metadata/weights digest mismatch")
    snapshot_weights = load_weights(np, weights_path)
    for name in WEIGHT_SHAPES:
        if not np.array_equal(
            snapshot_weights[name].view(np.uint32), master["weights"][name].view(np.uint32)
        ):
            raise ConfigurationRejected(f"source snapshot/master array differs bitwise: {name}")
    return {
        "metadata": snapshot,
        "sha256": digest,
        "path": path.resolve(),
        "weights_path": weights_path.resolve(),
        "weights_sha256": observed_weights_sha256,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != CONTRACT_SCHEMA or contract.get("status") != "UNEXECUTED_STATIC_CONTRACT":
        raise ConfigurationRejected("probe contract schema/status mismatch")
    numeric = contract.get("numeric_gates", {})
    performance = contract.get("performance_gates", {})
    expected_numeric = {
        "cpu_output_maximum_absolute_error": CPU_OUTPUT_ATOL,
        "training_loss_maximum_absolute_error": LOSS_ATOL,
        "training_gradient_cosine_minimum": GRADIENT_COSINE_MIN,
        "training_gradient_relative_l2_maximum": GRADIENT_RELATIVE_L2_MAX,
        "training_gradient_maximum_absolute_error": GRADIENT_MAX_ABS,
        "updated_parameter_maximum_absolute_error": UPDATED_MAX_ABS,
    }
    for field, expected in expected_numeric.items():
        if float(numeric.get(field, float("nan"))) != expected:
            raise ConfigurationRejected(f"contract numeric threshold drift: {field}")
    for field in (
        "real_backward_optimizer_weight_change_required",
        "all_eight_parameter_tensors_receive_step_one_adamw_state",
        "nonzero_first_and_second_moment_required",
        "normalization_bitwise_unchanged",
    ):
        if numeric.get(field) is not True:
            raise ConfigurationRejected(f"contract mandatory training gate drift: {field}")
    if float(performance.get(
        "xpu_end_to_end_total_p50_speedup_vs_cpu_minimum", float("nan")
    )) != SPEEDUP_MIN:
        raise ConfigurationRejected("contract speedup threshold drift")
    if float(performance.get(
        "xpu_end_to_end_p95_ratio_vs_cpu_maximum", float("nan")
    )) != P95_RATIO_MAX:
        raise ConfigurationRejected("contract p95 threshold drift")


def load_witness(np, npz_path: Path, metadata_path: Path, expected_metadata_sha256: str,
                 master: dict[str, Any], teacher_manifest_path: Path | None,
                 teacher_manifest_sha256: str | None) -> dict[str, Any]:
    metadata, metadata_sha256 = read_bound_json(
        metadata_path, expected_metadata_sha256, "training witness metadata"
    )
    if metadata.get("schema") != WITNESS_SCHEMA:
        raise ConfigurationRejected("training witness schema mismatch")
    if metadata.get("kind") not in (
        "teacher_v3_train", "deterministic_synthetic_component_only",
    ):
        raise ConfigurationRejected("training witness kind is not authorized")
    teacher_binding = None
    if metadata["kind"] == "teacher_v3_train":
        if teacher_manifest_path is None or teacher_manifest_sha256 is None:
            raise ConfigurationRejected("teacher_v3 witness requires a bound dataset manifest")
        teacher_manifest, observed_teacher_sha256 = read_bound_json(
            teacher_manifest_path, teacher_manifest_sha256, "teacher_v3 dataset manifest"
        )
        if metadata.get("teacher_manifest_sha256") != observed_teacher_sha256:
            raise ConfigurationRejected("witness/teacher manifest binding mismatch")
        if metadata.get("teacher_split") != "train":
            raise ConfigurationRejected("teacher witness may use only the train split")
        teacher_binding = {
            "path": teacher_manifest_path.resolve(),
            "sha256": observed_teacher_sha256,
            "schema": teacher_manifest.get("schema"),
        }
    else:
        if teacher_manifest_path is not None or teacher_manifest_sha256 is not None:
            raise ConfigurationRejected("synthetic witness must not claim a teacher manifest")
        require_sha256(metadata.get("generator_source_sha256"), "synthetic generator source")
        if metadata.get("seed_namespace") != "synthetic_component_only_no_game_rng":
            raise ConfigurationRejected("synthetic witness seed namespace is not isolated")
    if metadata.get("reserved_seed_free") is not True:
        raise ConfigurationRejected("training witness is not explicitly reserved-seed-free")
    for field in ("development_seeds_used", "validation_seeds_used", "sealed_seeds_used"):
        if metadata.get(field) is not False:
            raise ConfigurationRejected(f"training witness must declare {field}=false")
    if metadata.get("mask_applied") is not True:
        raise ConfigurationRejected("training witness must carry pre-masked cotangents/masks")
    if metadata.get("cpu_reference_backend") != "Lux+Reactant+EnzymeMLIR CPU FP32":
        raise ConfigurationRejected("CPU master reference backend identity mismatch")
    if metadata.get("weights_sha256") != master["weights_sha256"]:
        raise ConfigurationRejected("witness/master weight digest mismatch")
    if metadata.get("master_manifest_sha256") != master["manifest_sha256"]:
        raise ConfigurationRejected("witness/master manifest binding mismatch")
    if not npz_path.is_file():
        raise ConfigurationRejected("training witness NPZ is missing")
    npz_sha256 = sha256_file(npz_path)
    if metadata.get("npz_sha256") != npz_sha256:
        raise ConfigurationRejected("training witness NPZ digest mismatch")
    expected_keys = set(WITNESS_BASE_SHAPES)
    for prefix in ("reference_grad_", "reference_updated_"):
        expected_keys.update(prefix + name for name in TRAINABLE_SHAPES)
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            missing = sorted(expected_keys - set(archive.files))
            extra = sorted(set(archive.files) - expected_keys)
            raise ConfigurationRejected(f"training witness key mismatch; missing={missing}, extra={extra}")
        arrays = {
            name: np.array(archive[name], dtype=np.float32, copy=True)
            for name in expected_keys
        }
    rows = arrays["packed"].shape[0] if arrays["packed"].ndim == 2 else -1
    if rows <= 0 or rows % BATCH != 0 or int(metadata.get("row_count", -1)) != rows:
        raise ConfigurationRejected("training witness row count must be a positive multiple of 16")
    expected_dynamic = {
        "packed": (rows, INPUTS),
        "output_cotangent": (rows, OUTPUTS),
        "auxiliary_target": (rows, 10),
        "auxiliary_mask": (rows, 10),
        "state_mask": (rows,),
        "reference_output": (BATCH, OUTPUTS),
        "reference_loss": (1,),
    }
    for name, shape in expected_dynamic.items():
        if arrays[name].shape != shape:
            raise ConfigurationRejected(f"witness {name} shape {arrays[name].shape} != {shape}")
    for prefix in ("reference_grad_", "reference_updated_"):
        for name, shape in TRAINABLE_SHAPES.items():
            key = prefix + name
            if arrays[key].shape != shape:
                raise ConfigurationRejected(f"witness {key} shape mismatch")
    for name, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise ConfigurationRejected(f"witness {name} contains non-finite values")
        value.setflags(write=False)
    mask = arrays["state_mask"]
    if not np.all((mask == 0.0) | (mask == 1.0)):
        raise ConfigurationRejected("state_mask must contain only 0/1")
    aux_mask = arrays["auxiliary_mask"]
    if not np.all((aux_mask == 0.0) | (aux_mask == 1.0)):
        raise ConfigurationRejected("auxiliary_mask must contain only 0/1")
    masked_rows = mask == 0.0
    if np.any(arrays["output_cotangent"][masked_rows] != 0.0) or np.any(aux_mask[masked_rows] != 0.0):
        raise ConfigurationRejected("masked rows must have exactly zero cotangent and auxiliary mask")
    if float(metadata.get("auxiliary_weight", float("nan"))) != float(
        master["metadata"].get("auxiliary_weight", float("nan"))
    ):
        raise ConfigurationRejected("witness/master auxiliary weight mismatch")
    if float(metadata.get("huber_beta", float("nan"))) != float(
        master["metadata"].get("huber_beta", float("nan"))
    ):
        raise ConfigurationRejected("witness/master Huber beta mismatch")
    return {
        "metadata": metadata,
        "metadata_sha256": metadata_sha256,
        "npz_path": npz_path.resolve(),
        "npz_sha256": npz_sha256,
        "arrays": arrays,
        "rows": rows,
        "teacher_binding": teacher_binding,
    }


def _powershell_json(script: str) -> Any:
    executable = shutil.which("powershell.exe")
    if executable is None:
        raise BackendUnavailable("native Windows PowerShell is unavailable")
    completed = subprocess.run(
        [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        creationflags=0x08000000,
    )
    if completed.returncode != 0:
        raise BackendUnavailable(f"Windows CIM enumeration failed: {completed.stderr.strip()}")
    try:
        return strict_json_loads(completed.stdout, "Windows CIM enumeration")
    except ConfigurationRejected as exc:
        raise BackendUnavailable(f"Windows CIM enumeration returned invalid JSON: {exc}") from exc


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _derive_live_host_id(system_product: dict[str, Any]) -> tuple[str, dict[str, str]]:
    normalized = {
        "UUID": str(system_product.get("UUID", "")).strip().upper(),
        "Vendor": str(system_product.get("Vendor", "")).strip(),
        "Name": str(system_product.get("Name", "")).strip(),
        "IdentifyingNumber": str(system_product.get("IdentifyingNumber", "")).strip(),
    }
    if any(not value for value in normalized.values()):
        raise BackendUnavailable("Win32_ComputerSystemProduct identity is incomplete")
    digest = hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()
    return "sha256:" + digest, normalized


def _read_unsigned_le(raw: bytes, offset: int, width: int) -> int:
    end = offset + width
    if offset < 0 or end > len(raw):
        raise BackendUnavailable("CPU Set record field exceeds the returned buffer")
    return int.from_bytes(raw[offset:end], byteorder="little", signed=False)


def _parse_windows_cpu_set_buffer(raw: bytes) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = 0
    while offset < len(raw):
        if len(raw) - offset < 8:
            raise BackendUnavailable("truncated SYSTEM_CPU_SET_INFORMATION header")
        record_size = _read_unsigned_le(raw, offset, 4)
        if record_size < 8:
            raise BackendUnavailable("invalid SYSTEM_CPU_SET_INFORMATION record size")
        record_end = offset + record_size
        if record_end > len(raw):
            raise BackendUnavailable("SYSTEM_CPU_SET_INFORMATION record exceeds buffer")
        information_type = _read_unsigned_le(raw, offset + 4, 4)
        if information_type == CPU_SET_INFORMATION_TYPE:
            if record_size < CPU_SET_MINIMUM_RECORD_SIZE:
                raise BackendUnavailable("CpuSet record is smaller than audited Windows layout")
            flags = raw[offset + 19]
            records.append({
                "record_size": record_size,
                "id": _read_unsigned_le(raw, offset + 8, 4),
                "group": _read_unsigned_le(raw, offset + 12, 2),
                "logical_processor_index": raw[offset + 14],
                "core_index": raw[offset + 15],
                "last_level_cache_index": raw[offset + 16],
                "numa_node_index": raw[offset + 17],
                "efficiency_class": raw[offset + 18],
                "scheduling_class": raw[offset + 20],
                "all_flags": flags,
                "parked": bool(flags & 0x01),
                "allocated": bool(flags & 0x02),
                "allocated_to_target_process": bool(flags & 0x04),
                "realtime": bool(flags & 0x08),
                "reserved_flags": flags >> 4,
                "allocation_tag": _read_unsigned_le(raw, offset + 24, 8),
            })
        offset = record_end
    if offset != len(raw):
        raise BackendUnavailable("CPU Set record iteration did not end exactly")
    if not records:
        raise BackendUnavailable("CPU Set enumeration contained no CpuSet records")
    return records


def _enumerate_windows_cpu_sets() -> list[dict[str, Any]]:
    if os.name != "nt" or platform.system() != "Windows" or sys.maxsize <= 2**32:
        raise BackendUnavailable("audited CPU Set discovery requires 64-bit native Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = ctypes.c_void_p
    get_cpu_sets = kernel32.GetSystemCpuSetInformation
    get_cpu_sets.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    get_cpu_sets.restype = ctypes.c_int32
    process = get_current_process()
    for _ in range(4):
        required = ctypes.c_uint32(0)
        ctypes.set_last_error(0)
        success = get_cpu_sets(None, 0, ctypes.byref(required), process, 0)
        error = ctypes.get_last_error()
        if not success and error != ERROR_INSUFFICIENT_BUFFER:
            raise BackendUnavailable(
                f"GetSystemCpuSetInformation(size) failed with Win32 error {error}"
            )
        if required.value <= 0:
            raise BackendUnavailable("GetSystemCpuSetInformation returned no CPU Sets")
        buffer = (ctypes.c_ubyte * required.value)()
        returned = ctypes.c_uint32(required.value)
        ctypes.set_last_error(0)
        success = get_cpu_sets(
            ctypes.cast(buffer, ctypes.c_void_p), required.value,
            ctypes.byref(returned), process, 0,
        )
        error = ctypes.get_last_error()
        if success:
            if returned.value > required.value:
                raise BackendUnavailable("Win32 returned an oversized CPU Set buffer")
            return _parse_windows_cpu_set_buffer(bytes(buffer[:returned.value]))
        if error != ERROR_INSUFFICIENT_BUFFER:
            raise BackendUnavailable(
                f"GetSystemCpuSetInformation(data) failed with Win32 error {error}"
            )
    raise BackendUnavailable("CPU Set topology changed repeatedly during enumeration")


def _core_ultra_265k_topology_receipt(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len({record["id"] for record in records}) != len(records):
        raise BackendUnavailable("duplicate CPU Set IDs")
    if any(record["allocated_to_target_process"] and not record["allocated"] for record in records):
        raise BackendUnavailable("inconsistent CPU Set allocation flags")
    if any(
        record["parked"] or record["realtime"]
        or (record["allocated"] and not record["allocated_to_target_process"])
        for record in records
    ):
        raise BackendUnavailable("a CPU Set is unusable or allocated to another process")
    if len(records) != 20:
        raise BackendUnavailable("target requires exactly 20 usable logical processors")
    if len({record["group"] for record in records}) != 1:
        raise BackendUnavailable("target CPU Sets unexpectedly span processor groups")
    if len({(record["group"], record["logical_processor_index"]) for record in records}) != 20:
        raise BackendUnavailable("duplicate group-relative logical processor")
    if len({(record["group"], record["core_index"]) for record in records}) != 20:
        raise BackendUnavailable("SMT or duplicate group-relative core was discovered")
    classes = sorted({record["efficiency_class"] for record in records})
    if len(classes) != 2:
        raise BackendUnavailable("target requires exactly two intrinsic efficiency classes")
    efficiency_class, performance_class = classes
    order = lambda record: (
        record["group"], record["logical_processor_index"], record["core_index"], record["id"]
    )
    ordered = sorted(records, key=order)
    performance = sorted(
        (record for record in records if record["efficiency_class"] == performance_class),
        key=order,
    )
    efficiency = sorted(
        (record for record in records if record["efficiency_class"] == efficiency_class),
        key=order,
    )
    if len(performance) != 8 or len(efficiency) != 12:
        raise BackendUnavailable("discovered efficiency classes are not the target 8P+12E topology")
    cpu_sets = []
    for record in ordered:
        cpu_sets.append({
            "record_size": record["record_size"],
            "id": record["id"],
            "group": record["group"],
            "logical_processor_index": record["logical_processor_index"],
            "core_index": record["core_index"],
            "last_level_cache_index": record["last_level_cache_index"],
            "numa_node_index": record["numa_node_index"],
            "efficiency_class": record["efficiency_class"],
            "scheduling_class": record["scheduling_class"],
            "flags": {
                "all_flags": record["all_flags"],
                "parked": record["parked"],
                "allocated": record["allocated"],
                "allocated_to_target_process": record["allocated_to_target_process"],
                "realtime": record["realtime"],
                "reserved_flags": record["reserved_flags"],
            },
            "allocation_tag": record["allocation_tag"],
        })
    payload = {
        "schema": "windows-cpu-set-topology-v1",
        "discovery_api": "GetSystemCpuSetInformation",
        "cpu_set_count": len(ordered),
        "classification": {
            "adopted": True,
            "status": "target_8p12e",
            "reason": "two-class 8P+12E topology discovered without numbering assumptions",
            "performance_efficiency_class": performance_class,
            "efficiency_efficiency_class": efficiency_class,
            "performance_cpu_set_ids": [record["id"] for record in performance],
            "efficiency_cpu_set_ids": [record["id"] for record in efficiency],
        },
        "cpu_sets": cpu_sets,
    }
    return {
        "payload": payload,
        "topology_sha256": hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
        "fair_hashstem_control_threads": len(performance),
    }


def enumerate_windows_identity() -> dict[str, Any]:
    video = _as_list(_powershell_json(
        "[Console]::OutputEncoding=[Text.UTF8Encoding]::new();"
        "@(Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID,"
        "DriverVersion,AdapterCompatibility,AdapterRAM,Status,VideoProcessor) | "
        "ConvertTo-Json -Compress -Depth 4"
    ))
    processors = _as_list(_powershell_json(
        "[Console]::OutputEncoding=[Text.UTF8Encoding]::new();"
        "@(Get-CimInstance Win32_Processor | Select-Object Name,Manufacturer,"
        "NumberOfCores,NumberOfLogicalProcessors,ProcessorId) | "
        "ConvertTo-Json -Compress -Depth 4"
    ))
    system_products = _as_list(_powershell_json(
        "[Console]::OutputEncoding=[Text.UTF8Encoding]::new();"
        "@(Get-CimInstance Win32_ComputerSystemProduct | Select-Object UUID,Vendor,"
        "Name,IdentifyingNumber) | ConvertTo-Json -Compress -Depth 4"
    ))
    return {
        "video_controllers": video,
        "processors": processors,
        "system_products": system_products,
    }


def _property_receipt(properties: Any) -> dict[str, Any]:
    receipt: dict[str, Any] = {"repr": repr(properties)}
    for name in dir(properties):
        if name.startswith("_"):
            continue
        try:
            value = getattr(properties, name)
        except Exception:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            receipt[name] = value
    return receipt


def validate_xpu_identity(torch, system: dict[str, Any], requested_cpu_threads: int) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "validation_status": "INCOMPLETE_FAIL_CLOSED",
        "native_windows": os.name == "nt" and platform.system() == "Windows",
    }
    try:
        if not receipt["native_windows"]:
            raise BackendUnavailable("probe requires native Windows")

        # Capture the live host/topology before comparing caller-supplied identity
        # fields so a failed comparison still publishes the observed receipt.
        cim = enumerate_windows_identity()
        receipt["cim_video_controllers"] = cim["video_controllers"]
        receipt["cim_processors"] = cim["processors"]
        receipt["cim_system_products_raw"] = cim["system_products"]
        if len(cim["system_products"]) != 1:
            raise BackendUnavailable("Win32_ComputerSystemProduct identity is absent or ambiguous")
        derived_host_id, normalized_system_product = _derive_live_host_id(
            cim["system_products"][0]
        )
        receipt["cim_system_product"] = normalized_system_product
        receipt["derived_host_id"] = derived_host_id
        topology = _core_ultra_265k_topology_receipt(_enumerate_windows_cpu_sets())
        fair_cpu_threads = topology["fair_hashstem_control_threads"]
        receipt["cpu_set_topology_sha256"] = topology["topology_sha256"]
        receipt["cpu_set_topology"] = topology["payload"]
        receipt["fair_hashstem_control_threads"] = fair_cpu_threads

        if system.get("schema") != SYSTEM_SCHEMA or system.get("status") != "BOUND_LIVE_HOST":
            raise ConfigurationRejected("system contract schema/status mismatch")
        igpu = system.get("igpu")
        cpu = system.get("cpu")
        if not isinstance(igpu, dict) or not isinstance(cpu, dict):
            raise ConfigurationRejected("system contract CPU/iGPU blocks are missing")
        if not isinstance(system.get("host_id"), str) or not system["host_id"]:
            raise ConfigurationRejected("system contract host_id is missing")
        if system["host_id"] != derived_host_id:
            raise BackendUnavailable("live derived host_id differs from system contract")
        if not isinstance(cpu.get("exact_name"), str) or not cpu["exact_name"]:
            raise ConfigurationRejected("system contract CPU exact_name is missing")
        contract_topology_sha256 = require_sha256(
            cpu.get("cpu_set_topology_sha256"), "system contract CPU Set topology SHA-256"
        )
        if topology["topology_sha256"] != contract_topology_sha256:
            raise BackendUnavailable("live CPU Set topology differs from system contract")
        contract_cpu_threads = cpu.get("hashstem_control_threads")
        if isinstance(contract_cpu_threads, bool) or not isinstance(contract_cpu_threads, int):
            raise ConfigurationRejected(
                "system contract HashStem control threads must be an integer"
            )
        if contract_cpu_threads != fair_cpu_threads:
            raise ConfigurationRejected(
                "contract HashStem control threads differ from live P-core count"
            )
        if requested_cpu_threads != fair_cpu_threads:
            raise ConfigurationRejected(
                "CLI CPU thread count differs from contract-bound live P-core count"
            )
        for field in (
            "torch_xpu_exact_name", "cim_exact_name", "pnp_device_id", "driver_version",
        ):
            if not isinstance(igpu.get(field), str) or not igpu[field]:
                raise ConfigurationRejected(f"system contract iGPU {field} is missing")
        for field in ("torch_xpu_device_count", "torch_xpu_index"):
            if isinstance(igpu.get(field), bool) or not isinstance(igpu.get(field), int):
                raise ConfigurationRejected(f"system contract iGPU {field} must be an integer")
        if igpu.get("integrated") is not True:
            raise ConfigurationRejected("system contract does not bind an integrated GPU")

        processor_names = [str(item.get("Name", "")) for item in cim["processors"]]
        if processor_names != [str(cpu.get("exact_name", ""))]:
            raise BackendUnavailable("runtime CPU identity differs from system contract")
        expected_pnp = str(igpu.get("pnp_device_id", "")).upper()
        if not expected_pnp.startswith("PCI\\VEN_8086&"):
            raise ConfigurationRejected("system contract PNP identity is not an Intel PCI device")
        matches = [
            item for item in cim["video_controllers"]
            if str(item.get("PNPDeviceID", "")).upper() == expected_pnp
        ]
        if len(matches) != 1:
            raise BackendUnavailable(
                "system-contract Intel PNP identity is absent or ambiguous in CIM"
            )
        adapter = matches[0]
        receipt["cim_adapter"] = adapter
        exact_fields = (
            ("Name", "cim_exact_name"),
            ("DriverVersion", "driver_version"),
            ("PNPDeviceID", "pnp_device_id"),
        )
        for cim_field, contract_field in exact_fields:
            if str(adapter.get(cim_field, "")).upper() != str(igpu.get(contract_field, "")).upper():
                raise BackendUnavailable(f"CIM {cim_field} differs from system contract")
        if str(adapter.get("Status", "")).upper() != "OK":
            raise BackendUnavailable("Intel iGPU CIM status is not OK")
        if "INTEL" not in str(adapter.get("AdapterCompatibility", "")).upper():
            raise BackendUnavailable("CIM adapter compatibility does not identify Intel")

        if not hasattr(torch, "xpu"):
            raise BackendUnavailable("PyTorch build does not expose torch.xpu")
        try:
            available = bool(torch.xpu.is_available())
            count = int(torch.xpu.device_count())
        except Exception as exc:
            raise BackendUnavailable(f"torch.xpu enumeration failed: {exc}") from exc
        receipt["torch_xpu_available"] = available
        receipt["enumerated_xpu_count"] = count
        if not available or count != 1:
            raise BackendUnavailable(
                f"exactly one available XPU is required; available={available}, count={count}"
            )
        if int(igpu.get("torch_xpu_device_count", -1)) != count:
            raise BackendUnavailable("runtime XPU count differs from system contract")
        index = int(igpu.get("torch_xpu_index", -1))
        if index != 0:
            raise BackendUnavailable("the unambiguous single-XPU device must use index zero")
        try:
            torch.xpu.set_device(index)
            xpu_name = str(torch.xpu.get_device_name(index))
            properties = torch.xpu.get_device_properties(index)
        except Exception as exc:
            raise BackendUnavailable(f"XPU property query failed: {exc}") from exc
        receipt["torch_xpu_index"] = index
        receipt["torch_xpu_name"] = xpu_name
        receipt["torch_xpu_properties"] = _property_receipt(properties)
        if xpu_name != igpu.get("torch_xpu_exact_name"):
            raise BackendUnavailable("runtime XPU name differs from system contract")
        receipt["validation_status"] = "BOUND_LIVE_HOST_AND_XPU_VERIFIED"
        return receipt
    except (BackendUnavailable, ConfigurationRejected) as exc:
        receipt["validation_status"] = "FAIL_CLOSED"
        receipt["failure_reason"] = str(exc)
        exc.device_identity_receipt = receipt
        raise
    except Exception as exc:
        receipt["validation_status"] = "FAIL_CLOSED"
        receipt["failure_reason"] = str(exc)
        wrapped = BackendUnavailable(f"live identity validation failed: {exc}")
        wrapped.device_identity_receipt = receipt
        raise wrapped from exc


def build_types(torch):
    nn = torch.nn
    functional = torch.nn.functional

    class LearnedHashStem(nn.Module):
        def __init__(self, weights: dict[str, Any]):
            super().__init__()
            self.conv3 = nn.Conv2d(2, 16, (3, 3), padding=(1, 1), bias=True)
            self.depthwise5x1 = nn.Conv2d(
                16, 16, (5, 1), padding=(2, 0), groups=16, bias=True
            )
            self.pointwise = nn.Conv2d(16, 32, (1, 1), bias=True)
            self.dense = nn.Linear(1039, 214, bias=True)
            self.register_buffer("input_mean", torch.from_numpy(weights["input_mean"].copy()))
            self.register_buffer("input_inv_std", torch.from_numpy(weights["input_inv_std"].copy()))
            load_model_weights(torch, self, weights)

        def forward(self, packed):
            if packed.ndim != 2 or packed.shape[1] != INPUTS:
                raise RuntimeError("HashStem input must be Bx559")
            normalized = (packed - self.input_mean.reshape(1, -1)) * self.input_inv_std.reshape(1, -1)
            board = normalized[:, :480].reshape(-1, 2, 10, 24).permute(0, 1, 3, 2).contiguous()
            value = functional.relu(self.conv3(board))
            value = functional.relu(self.depthwise5x1(value))
            value = functional.relu(self.pointwise(value))
            value = functional.avg_pool2d(value, kernel_size=(4, 2), stride=(4, 2))
            pooled = value.flatten(start_dim=1)
            dense_input = torch.cat((pooled, normalized[:, 522:559], normalized[:, 480:522]), dim=1)
            learned = self.dense(dense_input)
            return torch.cat((learned, packed[:, 480:522]), dim=1)

    class TrainingObjective(nn.Module):
        def __init__(self, model, auxiliary_weight: float, huber_beta: float):
            super().__init__()
            self.model = model
            self.auxiliary_weight = float(auxiliary_weight)
            self.huber_beta = float(huber_beta)

        def forward(self, packed, output_cotangent, auxiliary_target, auxiliary_mask):
            output = self.model(packed)
            vjp_surrogate = torch.sum(output * output_cotangent)
            difference = output[:, AUXILIARY_START:AUXILIARY_END] - auxiliary_target
            absolute = torch.abs(difference)
            beta = self.huber_beta
            huber = torch.where(
                absolute <= beta,
                0.5 * difference * difference / beta,
                absolute - 0.5 * beta,
            )
            weighted = huber * auxiliary_mask
            denominator = torch.clamp(torch.sum(auxiliary_mask), min=1.0)
            auxiliary_loss = torch.sum(weighted) / denominator
            return vjp_surrogate + self.auxiliary_weight * auxiliary_loss, output

    return LearnedHashStem, TrainingObjective


def load_model_weights(torch, model, weights: dict[str, Any]) -> None:
    with torch.no_grad():
        model.conv3.weight.copy_(torch.from_numpy(weights["conv3"].copy()).to(model.conv3.weight.device))
        model.conv3.bias.copy_(torch.from_numpy(weights["conv3_bias"].copy()).to(model.conv3.bias.device))
        depthwise = weights["depthwise5x1"].reshape(16, 1, 5, 1)
        model.depthwise5x1.weight.copy_(torch.from_numpy(depthwise.copy()).to(model.depthwise5x1.weight.device))
        model.depthwise5x1.bias.copy_(
            torch.from_numpy(weights["depthwise_bias"].copy()).to(model.depthwise5x1.bias.device)
        )
        pointwise = weights["pointwise"].reshape(32, 16, 1, 1)
        model.pointwise.weight.copy_(torch.from_numpy(pointwise.copy()).to(model.pointwise.weight.device))
        model.pointwise.bias.copy_(
            torch.from_numpy(weights["pointwise_bias"].copy()).to(model.pointwise.bias.device)
        )
        model.dense.weight.copy_(torch.from_numpy(weights["dense"].copy()).to(model.dense.weight.device))
        model.dense.bias.copy_(torch.from_numpy(weights["dense_bias"].copy()).to(model.dense.bias.device))
        model.input_mean.copy_(torch.from_numpy(weights["input_mean"].copy()).to(model.input_mean.device))
        model.input_inv_std.copy_(
            torch.from_numpy(weights["input_inv_std"].copy()).to(model.input_inv_std.device)
        )


def canonical_tensors(np, model, gradient: bool = False) -> dict[str, Any]:
    def array(tensor):
        value = tensor.grad if gradient else tensor
        if value is None:
            raise ConfigurationRejected("missing gradient in full HashStem backward")
        return np.array(value.detach().to("cpu").contiguous().numpy(), dtype=np.float32, copy=True)

    return {
        "conv3": array(model.conv3.weight),
        "conv3_bias": array(model.conv3.bias),
        "depthwise5x1": array(model.depthwise5x1.weight).reshape(16, 5),
        "depthwise_bias": array(model.depthwise5x1.bias),
        "pointwise": array(model.pointwise.weight).reshape(32, 16),
        "pointwise_bias": array(model.pointwise.bias),
        "dense": array(model.dense.weight),
        "dense_bias": array(model.dense.bias),
    }


def make_optimizer(torch, model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        eps=EPSILON,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
        fused=False,
    )


def _sync(torch, device_type: str) -> None:
    if device_type == "xpu":
        torch.xpu.synchronize()


def torch_batch(torch, arrays: dict[str, Any], start: int, device: str, non_blocking: bool):
    keys = ("packed", "output_cotangent", "auxiliary_target", "auxiliary_mask")
    tensors = []
    for key in keys:
        host = torch.from_numpy(arrays[key][start:start + BATCH].copy())
        if device.startswith("xpu"):
            host = host.pin_memory()
            host = host.to(device, non_blocking=non_blocking)
        tensors.append(host)
    _sync(torch, "xpu" if device.startswith("xpu") else "cpu")
    return tuple(tensors)


@dataclass
class CompiledBundle:
    device: str
    objective: Any
    compiled: Any
    optimizer: Any
    compile_wrapper_ns: int
    first_forward_backward_ns: int


def compile_bundle(torch, LearnedHashStem, TrainingObjective, weights, witness, device: str,
                   auxiliary_weight: float, huber_beta: float) -> CompiledBundle:
    model = LearnedHashStem(weights).to(device=device, dtype=torch.float32)
    objective = TrainingObjective(model, auxiliary_weight, huber_beta).to(device=device)
    objective.train()
    optimizer = make_optimizer(torch, objective.model)
    first_batch = torch_batch(torch, witness["arrays"], 0, device, non_blocking=False)
    try:
        begin = time.perf_counter_ns()
        compiled = torch.compile(objective, backend="inductor", fullgraph=True, dynamic=False)
        wrapper_ns = time.perf_counter_ns() - begin
        begin = time.perf_counter_ns()
        loss, output = compiled(*first_batch)
        loss.backward()
        _sync(torch, "xpu" if device.startswith("xpu") else "cpu")
        first_ns = time.perf_counter_ns() - begin
    except Exception as exc:
        raise BackendUnavailable(f"torch.compile fixed-shape {device} training failed: {exc}") from exc
    if loss.device.type != ("xpu" if device.startswith("xpu") else "cpu"):
        raise BackendUnavailable("compiled loss executed on an unexpected device")
    if output.device.type != loss.device.type:
        raise BackendUnavailable("compiled output executed on an unexpected device")
    for parameter in objective.parameters():
        if parameter.device.type != loss.device.type or parameter.grad is None:
            raise BackendUnavailable("parameter/gradient device proves fallback or incomplete backward")
    optimizer.zero_grad(set_to_none=True)
    load_model_weights(torch, objective.model, weights)
    optimizer = make_optimizer(torch, objective.model)
    return CompiledBundle(device, objective, compiled, optimizer, wrapper_ns, first_ns)


def one_update(torch, np, bundle: CompiledBundle, witness: dict[str, Any]) -> dict[str, Any]:
    device_type = "xpu" if bundle.device.startswith("xpu") else "cpu"
    batch = torch_batch(torch, witness["arrays"], 0, bundle.device, non_blocking=False)
    before_normalization = (
        bundle.objective.model.input_mean.detach().to("cpu").contiguous().numpy().tobytes(),
        bundle.objective.model.input_inv_std.detach().to("cpu").contiguous().numpy().tobytes(),
    )
    bundle.optimizer.zero_grad(set_to_none=True)
    before_parameters = canonical_tensors(np, bundle.objective.model, gradient=False)
    loss, output = bundle.compiled(*batch)
    loss.backward()
    _sync(torch, device_type)
    gradients = canonical_tensors(np, bundle.objective.model, gradient=True)
    output_cpu = np.array(output.detach().to("cpu").numpy(), dtype=np.float32, copy=True)
    loss_cpu = float(loss.detach().to("cpu").item())
    bundle.optimizer.step()
    _sync(torch, device_type)
    updated = canonical_tensors(np, bundle.objective.model, gradient=False)
    changed_elements = sum(
        int(np.count_nonzero(updated[name].view(np.uint32) != before_parameters[name].view(np.uint32)))
        for name in TRAINABLE_SHAPES
    )
    maximum_parameter_change = max(
        float(np.max(np.abs(updated[name] - before_parameters[name])))
        for name in TRAINABLE_SHAPES
    )
    optimizer_state = bundle.optimizer.state_dict()["state"]
    optimizer_steps = []
    nonzero_first_moment = False
    nonzero_second_moment = False
    for state in optimizer_state.values():
        step = state.get("step")
        if torch.is_tensor(step):
            optimizer_steps.append(float(step.detach().to("cpu").item()))
        else:
            optimizer_steps.append(float(step))
        first = state.get("exp_avg")
        second = state.get("exp_avg_sq")
        if first is not None:
            nonzero_first_moment = nonzero_first_moment or bool(
                torch.count_nonzero(first.detach()).to("cpu").item()
            )
        if second is not None:
            nonzero_second_moment = nonzero_second_moment or bool(
                torch.count_nonzero(second.detach()).to("cpu").item()
            )
    after_normalization = (
        bundle.objective.model.input_mean.detach().to("cpu").contiguous().numpy().tobytes(),
        bundle.objective.model.input_inv_std.detach().to("cpu").contiguous().numpy().tobytes(),
    )
    return {
        "loss": loss_cpu,
        "output": output_cpu,
        "gradients": gradients,
        "updated": updated,
        "normalization_bitwise_unchanged": before_normalization == after_normalization,
        "real_update_evidence": {
            "changed_parameter_elements": changed_elements,
            "maximum_parameter_change": maximum_parameter_change,
            "optimizer_state_entries": len(optimizer_state),
            "optimizer_steps": optimizer_steps,
            "nonzero_first_moment": nonzero_first_moment,
            "nonzero_second_moment": nonzero_second_moment,
        },
    }


def flattened(np, tensors: dict[str, Any]) -> Any:
    return np.concatenate([np.asarray(tensors[name], dtype=np.float64).ravel() for name in TRAINABLE_SHAPES])


def compare_training(np, observed: dict[str, Any], reference: dict[str, Any], label: str,
                     output_tolerance: float) -> dict[str, Any]:
    observed_gradient = flattened(np, observed["gradients"])
    reference_gradient = flattened(np, reference["gradients"])
    dot = float(np.dot(observed_gradient, reference_gradient))
    observed_norm = float(np.linalg.norm(observed_gradient))
    reference_norm = float(np.linalg.norm(reference_gradient))
    if observed_norm == 0.0 and reference_norm == 0.0:
        cosine = 1.0
    elif observed_norm == 0.0 or reference_norm == 0.0:
        cosine = 0.0
    else:
        cosine = dot / (observed_norm * reference_norm)
    gradient_difference = observed_gradient - reference_gradient
    relative_l2 = float(np.linalg.norm(gradient_difference) / max(reference_norm, 1.0e-30))
    gradient_max = float(np.max(np.abs(gradient_difference)))
    updated_max = max(
        float(np.max(np.abs(observed["updated"][name] - reference["updated"][name])))
        for name in TRAINABLE_SHAPES
    )
    output_max = float(np.max(np.abs(observed["output"] - reference["output"])))
    loss_error = abs(float(observed["loss"]) - float(reference["loss"]))
    finite = all(math.isfinite(value) for value in (
        cosine, relative_l2, gradient_max, updated_max, output_max, loss_error,
    )) and all(np.all(np.isfinite(value)) for value in observed["updated"].values())
    gates = {
        "finite": finite,
        "output_max_abs": output_max <= output_tolerance,
        "loss_max_abs": loss_error <= LOSS_ATOL,
        "gradient_cosine": cosine >= GRADIENT_COSINE_MIN,
        "gradient_relative_l2": relative_l2 <= GRADIENT_RELATIVE_L2_MAX,
        "gradient_max_abs": gradient_max <= GRADIENT_MAX_ABS,
        "updated_parameter_max_abs": updated_max <= UPDATED_MAX_ABS,
        "normalization_bitwise_unchanged": bool(observed["normalization_bitwise_unchanged"]),
        "real_backward_optimizer_weight_change": bool(
            observed["real_update_evidence"]["changed_parameter_elements"] > 0
            and observed["real_update_evidence"]["maximum_parameter_change"] > 0.0
            and observed["real_update_evidence"]["optimizer_state_entries"]
                == len(TRAINABLE_SHAPES)
            and len(observed["real_update_evidence"]["optimizer_steps"])
                == len(TRAINABLE_SHAPES)
            and all(step == 1.0 for step in observed["real_update_evidence"]["optimizer_steps"])
            and observed["real_update_evidence"]["nonzero_first_moment"]
            and observed["real_update_evidence"]["nonzero_second_moment"]
        ),
    }
    return {
        "comparison": label,
        "output_maximum_absolute_error": output_max,
        "loss_absolute_error": loss_error,
        "gradient_cosine": cosine,
        "gradient_relative_l2": relative_l2,
        "gradient_maximum_absolute_error": gradient_max,
        "updated_parameter_maximum_absolute_error": updated_max,
        "real_update_evidence": observed["real_update_evidence"],
        "gates": gates,
        "passed": all(gates.values()),
    }


def external_cpu_reference(witness: dict[str, Any]) -> dict[str, Any]:
    arrays = witness["arrays"]
    return {
        "loss": float(arrays["reference_loss"][0]),
        "output": arrays["reference_output"],
        "gradients": {
            name: arrays["reference_grad_" + name] for name in TRAINABLE_SHAPES
        },
        "updated": {
            name: arrays["reference_updated_" + name] for name in TRAINABLE_SHAPES
        },
        "normalization_bitwise_unchanged": True,
    }


def _cpu_tree(torch, value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to("cpu").contiguous().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(torch, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(torch, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(torch, item) for item in value)
    return value


def _tree_bitwise_equal(torch, left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.dtype != right.dtype or left.shape != right.shape:
            return False
        left_array = left.detach().to("cpu").contiguous().numpy()
        right_array = right.detach().to("cpu").contiguous().numpy()
        return left_array.tobytes(order="C") == right_array.tobytes(order="C")
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _tree_bitwise_equal(torch, left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _tree_bitwise_equal(torch, a, b) for a, b in zip(left, right)
        )
    return left == right


def checkpoint_probe(torch, np, bundle: CompiledBundle, one_step: dict[str, Any],
                     master: dict[str, Any], snapshot: dict[str, Any], witness: dict[str, Any],
                     contract_sha256: str, system_sha256: str, destination: Path) -> dict[str, Any]:
    final = destination.resolve()
    temporary = final.with_name(final.name + ".tmp-" + uuid.uuid4().hex)
    if final.exists() or temporary.exists():
        raise ConfigurationRejected("probe checkpoint destination and transaction path must be fresh")
    temporary.mkdir(parents=True)
    model_state = _cpu_tree(torch, bundle.objective.model.state_dict())
    optimizer_state = _cpu_tree(torch, bundle.optimizer.state_dict())
    state_path = temporary / "igpu_one_step_state.pt"
    torch.save({
        "schema": "igpu-hashstem-one-step-state-v1",
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "probe_master_version": 1,
        "optimizer_step": 1,
    }, state_path)
    try:
        restored = torch.load(state_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise BackendUnavailable("PyTorch weights_only checkpoint reload is unavailable") from exc
    roundtrip = (
        restored.get("schema") == "igpu-hashstem-one-step-state-v1"
        and restored.get("probe_master_version") == 1
        and restored.get("optimizer_step") == 1
        and _tree_bitwise_equal(torch, restored.get("model_state"), model_state)
        and _tree_bitwise_equal(torch, restored.get("optimizer_state"), optimizer_state)
    )
    if not roundtrip:
        raise ConfigurationRejected("iGPU model/AdamW checkpoint did not reload bitwise")
    updated_weights = {name: one_step["updated"][name] for name in TRAINABLE_SHAPES}
    updated_weights["input_mean"] = master["weights"]["input_mean"]
    updated_weights["input_inv_std"] = master["weights"]["input_inv_std"]
    master_weights_path = temporary / "hashstem_probe_master_weights.npz"
    np.savez(master_weights_path, **updated_weights)
    snapshot_weights_path = temporary / "hashstem_probe_snapshot_weights.npz"
    shutil.copyfile(master_weights_path, snapshot_weights_path)
    master_weights_sha256 = sha256_file(master_weights_path)
    snapshot_weights_sha256 = sha256_file(snapshot_weights_path)
    if master_weights_sha256 != snapshot_weights_sha256:
        raise ConfigurationRejected("probe master/snapshot archives differ bytewise")
    normalization_sha256 = _normalization_sha256(np, updated_weights)
    if normalization_sha256 != master["metadata"].get("normalization_sha256"):
        raise ConfigurationRejected("probe update changed frozen normalization")
    master_metadata = {
        "schema": "igpu-hashstem-probe-master-v1",
        "status": "PROBE_FORK_ONLY_NOT_PRODUCTION",
        "model_id": master["metadata"].get("model_id"),
        "source_master_version": 0,
        "probe_master_version": 1,
        "optimizer_step": 1,
        "backend_updates": 1,
        "backend": "native Windows PyTorch XPU",
        "precision": "FP32",
        "weights_file": master_weights_path.name,
        "weights_sha256": master_weights_sha256,
        "normalization_sha256": normalization_sha256,
        "optimizer_file": state_path.name,
        "optimizer_sha256": sha256_file(state_path),
        "source_master_manifest_sha256": master["manifest_sha256"],
        "source_snapshot_metadata_sha256": snapshot["sha256"],
        "source_snapshot_weights_sha256": snapshot["weights_sha256"],
        "witness_sha256": witness["npz_sha256"],
        "contract_sha256": contract_sha256,
        "system_contract_sha256": system_sha256,
        "production_continuation_authorized": False,
        "adoption_authorized": False,
        "created_utc": utc_now(),
    }
    master_metadata_path = temporary / "probe_master_metadata.json"
    write_fresh_json(master_metadata_path, master_metadata)
    candidate_snapshot_version = int(snapshot["metadata"]["snapshot_version"]) + 1
    snapshot_metadata = {
        "schema": "igpu-hashstem-probe-snapshot-candidate-v1",
        "status": "UNPUBLISHED_PROBE_CANDIDATE",
        "model_id": master_metadata["model_id"],
        "source_snapshot_version": int(snapshot["metadata"]["snapshot_version"]),
        "candidate_snapshot_version": candidate_snapshot_version,
        "probe_master_version": 1,
        "weights_file": snapshot_weights_path.name,
        "weights_sha256": snapshot_weights_sha256,
        "normalization_sha256": normalization_sha256,
        "master_metadata_sha256": sha256_file(master_metadata_path),
        "immutable_source": True,
        "openvino_compiled": False,
        "publish_authorized": False,
        "adoption_authorized": False,
        "created_utc": utc_now(),
    }
    snapshot_metadata_path = temporary / "probe_snapshot_metadata.json"
    write_fresh_json(snapshot_metadata_path, snapshot_metadata)
    members = {}
    for path in (
        state_path, master_weights_path, snapshot_weights_path,
        master_metadata_path, snapshot_metadata_path,
    ):
        members[path.name] = sha256_file(path)
    manifest = {
        "schema": "igpu-hashstem-probe-checkpoint-manifest-v1",
        "status": "PROBE_FORK_ONLY_NOT_PRODUCTION",
        "members": members,
        "source_master_manifest_sha256": master["manifest_sha256"],
        "source_snapshot_metadata_sha256": snapshot["sha256"],
        "source_snapshot_weights_sha256": snapshot["weights_sha256"],
        "master_snapshot_weights_byte_identical": True,
        "state_roundtrip_bitwise": True,
        "normalization_unchanged": True,
        "probe_master_version": 1,
        "optimizer_step": 1,
        "candidate_snapshot_version": candidate_snapshot_version,
        "publish_authorized": False,
        "adoption_authorized": False,
    }
    manifest_path = temporary / "probe_checkpoint_manifest.json"
    write_fresh_json(manifest_path, manifest)
    temporary.replace(final)
    return {
        "directory": str(final),
        "manifest_sha256": sha256_file(final / manifest_path.name),
        "members": members,
        "state_roundtrip_bitwise": True,
        "master_snapshot_weights_byte_identical": True,
        "normalization_unchanged": True,
        "probe_master_version": 1,
        "optimizer_step": 1,
        "candidate_snapshot_version": candidate_snapshot_version,
        "publish_authorized": False,
        "adoption_authorized": False,
    }


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def memory_sample() -> dict[str, int]:
    if os.name != "nt":
        raise BackendUnavailable("Windows memory receipt requires native Windows")
    global_status = MEMORYSTATUSEX()
    global_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(global_status)):
        raise ctypes.WinError()
    process_status = PROCESS_MEMORY_COUNTERS_EX()
    process_status.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.psapi.GetProcessMemoryInfo(
        process, ctypes.byref(process_status), process_status.cb
    ):
        raise ctypes.WinError()
    return {
        "qpc_ns": time.perf_counter_ns(),
        "working_set_bytes": int(process_status.WorkingSetSize),
        "peak_working_set_bytes": int(process_status.PeakWorkingSetSize),
        "private_bytes": int(process_status.PrivateUsage),
        "peak_private_bytes": int(process_status.PeakPagefileUsage),
        "page_fault_count": int(process_status.PageFaultCount),
        "total_physical_bytes": int(global_status.ullTotalPhys),
        "available_physical_bytes": int(global_status.ullAvailPhys),
        "memory_load_percent": int(global_status.dwMemoryLoad),
    }


class MemorySampler:
    def __init__(self, period_ms: int):
        self.period_ms = period_ms
        self.samples: list[dict[str, int]] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self):
        self.samples.append(memory_sample())

        def loop():
            while not self.stop_event.wait(self.period_ms / 1000.0):
                self.samples.append(memory_sample())

        self.thread = threading.Thread(target=loop, name="hashstem-memory-receipt", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                raise ConfigurationRejected("memory receipt sampler did not stop")
        self.samples.append(memory_sample())

    def receipt(self) -> dict[str, Any]:
        if len(self.samples) < 2:
            raise ConfigurationRejected("memory receipt requires before/after samples")
        return {
            "schema": "igpu-hashstem-component-ram-contention-receipt-v1",
            "scope": "component process and host physical-memory pressure only",
            "sampling_period_ms": self.period_ms,
            "before": self.samples[0],
            "after": self.samples[-1],
            "working_set_peak_bytes": max(sample["working_set_bytes"] for sample in self.samples),
            "private_bytes_peak": max(sample["private_bytes"] for sample in self.samples),
            "minimum_available_physical_bytes": min(
                sample["available_physical_bytes"] for sample in self.samples
            ),
            "maximum_memory_load_percent": max(
                sample["memory_load_percent"] for sample in self.samples
            ),
            "raw_samples": self.samples,
            "dram_bandwidth_evidence": "NOT_MEASURED_EXTERNAL_ETW_IMC_GATE_REQUIRED",
            "p_sparse_contention_evidence": "NOT_MEASURED_EXTERNAL_GATE_REQUIRED",
        }


def host_batches(torch, witness: dict[str, Any], pin: bool) -> dict[str, Any]:
    result = {}
    for key in ("packed", "output_cotangent", "auxiliary_target", "auxiliary_mask"):
        tensor = torch.from_numpy(witness["arrays"][key].copy()).contiguous()
        if pin:
            try:
                tensor = tensor.pin_memory()
            except Exception as exc:
                raise BackendUnavailable(f"XPU pinned host memory is unavailable: {exc}") from exc
        result[key] = tensor
    return result


def timed_update(torch, bundle: CompiledBundle, host: dict[str, Any], batch_index: int) -> tuple[dict[str, int], Any]:
    device_type = "xpu" if bundle.device.startswith("xpu") else "cpu"
    start = batch_index * BATCH
    stop = start + BATCH
    total_begin = time.perf_counter_ns()
    begin = time.perf_counter_ns()
    source = tuple(host[key][start:stop] for key in (
        "packed", "output_cotangent", "auxiliary_target", "auxiliary_mask",
    ))
    if device_type == "xpu":
        batch = tuple(value.to(bundle.device, non_blocking=True) for value in source)
    else:
        batch = source
    h2d_submit = time.perf_counter_ns() - begin
    begin = time.perf_counter_ns()
    _sync(torch, device_type)
    h2d_sync = time.perf_counter_ns() - begin
    begin = time.perf_counter_ns()
    bundle.optimizer.zero_grad(set_to_none=True)
    zero_grad = time.perf_counter_ns() - begin
    begin = time.perf_counter_ns()
    loss, output = bundle.compiled(*batch)
    forward_submit = time.perf_counter_ns() - begin
    begin = time.perf_counter_ns()
    _sync(torch, device_type)
    forward_sync = time.perf_counter_ns() - begin
    begin = time.perf_counter_ns()
    loss.backward()
    backward_submit = time.perf_counter_ns() - begin
    begin = time.perf_counter_ns()
    _sync(torch, device_type)
    backward_sync = time.perf_counter_ns() - begin
    begin = time.perf_counter_ns()
    bundle.optimizer.step()
    optimizer_submit = time.perf_counter_ns() - begin
    begin = time.perf_counter_ns()
    _sync(torch, device_type)
    optimizer_sync = time.perf_counter_ns() - begin
    begin = time.perf_counter_ns()
    if device_type == "xpu":
        output_host = output.detach().to("cpu", non_blocking=True)
        loss_host = loss.detach().to("cpu", non_blocking=True)
    else:
        output_host = output.detach().clone()
        loss_host = loss.detach().clone()
    d2h_submit = time.perf_counter_ns() - begin
    begin = time.perf_counter_ns()
    _sync(torch, device_type)
    d2h_sync = time.perf_counter_ns() - begin
    total = time.perf_counter_ns() - total_begin
    if not bool(torch.isfinite(output_host).all()) or not bool(torch.isfinite(loss_host).all()):
        raise ConfigurationRejected("non-finite value during timed training")
    timing = {
        "h2d_submit_ns": h2d_submit,
        "h2d_sync_ns": h2d_sync,
        "zero_grad_ns": zero_grad,
        "forward_submit_ns": forward_submit,
        "forward_sync_ns": forward_sync,
        "backward_submit_ns": backward_submit,
        "backward_sync_ns": backward_sync,
        "optimizer_submit_ns": optimizer_submit,
        "optimizer_sync_ns": optimizer_sync,
        "d2h_submit_ns": d2h_submit,
        "d2h_sync_ns": d2h_sync,
        "total_ns": total,
    }
    return timing, output_host


def benchmark(torch, bundle: CompiledBundle, weights, witness, warmups: int, repeats: int,
              memory_period_ms: int) -> dict[str, Any]:
    load_model_weights(torch, bundle.objective.model, weights)
    bundle.optimizer = make_optimizer(torch, bundle.objective.model)
    host = host_batches(torch, witness, pin=bundle.device.startswith("xpu"))
    batch_count = witness["rows"] // BATCH
    for index in range(warmups):
        timed_update(torch, bundle, host, index % batch_count)
    raw: list[dict[str, int]] = []
    last_output = None
    with MemorySampler(memory_period_ms) as sampler:
        for index in range(repeats):
            sample, last_output = timed_update(torch, bundle, host, index % batch_count)
            sample["sample_index"] = index
            sample["witness_batch_index"] = index % batch_count
            raw.append(sample)
    if last_output is None:
        raise ConfigurationRejected("benchmark produced no output")
    output_sha256 = hashlib.sha256(last_output.contiguous().numpy().tobytes()).hexdigest()
    return {
        "device": bundle.device,
        "compile_wrapper_ns": bundle.compile_wrapper_ns,
        "first_fixed_shape_forward_backward_ns": bundle.first_forward_backward_ns,
        "compile_and_first_call_ns": bundle.compile_wrapper_ns + bundle.first_forward_backward_ns,
        "warmups": warmups,
        "timed_updates": repeats,
        "summary": summarize_timings(raw),
        "raw_qpc_timings": raw,
        "last_output_sha256": output_sha256,
        "ram_contention_receipt": sampler.receipt(),
    }


def _compile_amortized(compile_ns: int, summary: dict[str, Any], counts=(100, 1000)) -> dict[str, Any]:
    steady = summary["total"]["p50_ns"]
    return {
        str(count): {
            "total_ns": compile_ns + count * steady,
            "updates_per_second": count * 1.0e9 / (compile_ns + count * steady),
        }
        for count in counts
    }


def run_probe(args) -> dict[str, Any]:
    context: dict[str, Any] = {}
    try:
        import numpy as np
    except Exception as exc:
        raise BackendUnavailable(f"NumPy import failed: {exc}") from exc
    try:
        import torch
    except Exception as exc:
        raise BackendUnavailable(f"PyTorch import failed: {exc}") from exc

    contract, contract_sha256 = read_bound_json(args.contract, args.contract_sha256, "probe contract")
    validate_contract(contract)
    system, system_sha256 = read_bound_json(
        args.system_contract, args.system_contract_sha256, "system contract"
    )
    master = load_master_checkpoint(np, args.master_checkpoint)
    snapshot = load_snapshot_binding(
        np, args.snapshot_metadata, args.snapshot_metadata_sha256,
        args.snapshot_weights, args.snapshot_weights_sha256, master,
    )
    witness = load_witness(
        np, args.witness, args.witness_metadata, args.witness_metadata_sha256, master,
        args.teacher_manifest, args.teacher_manifest_sha256,
    )
    device_identity = validate_xpu_identity(torch, system, args.cpu_threads)

    if args.warmups < 5 or args.repeats < 50:
        raise ConfigurationRejected("probe requires at least 5 warmups and 50 timed updates")
    if not 5 <= args.memory_sample_period_ms <= 1000:
        raise ConfigurationRejected("memory sample period must be in [5,1000] ms")
    fair_cpu_threads = device_identity["fair_hashstem_control_threads"]
    torch.set_num_threads(fair_cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(args.seed)
    torch.xpu.manual_seed_all(args.seed)

    LearnedHashStem, TrainingObjective = build_types(torch)
    auxiliary_weight = float(master["metadata"]["auxiliary_weight"])
    huber_beta = float(master["metadata"]["huber_beta"])
    cpu_bundle = compile_bundle(
        torch, LearnedHashStem, TrainingObjective, master["weights"], witness,
        "cpu", auxiliary_weight, huber_beta,
    )
    xpu_device = f"xpu:{device_identity['torch_xpu_index']}"
    xpu_bundle = compile_bundle(
        torch, LearnedHashStem, TrainingObjective, master["weights"], witness,
        xpu_device, auxiliary_weight, huber_beta,
    )

    cpu_one = one_update(torch, np, cpu_bundle, witness)
    external = external_cpu_reference(witness)
    cpu_reference_comparison = compare_training(
        np, cpu_one, external, "PyTorch CPU vs bound Lux CPU master reference", CPU_OUTPUT_ATOL
    )
    if not cpu_reference_comparison["passed"]:
        return {
            "schema": PROBE_SCHEMA,
            "status": "REJECT_COMPONENT",
            "failure_reason": "PyTorch CPU graph does not match the bound CPU master reference",
            "device_identity": device_identity,
            "numeric_evidence": {"cpu_vs_master": cpu_reference_comparison},
            "adoption_authorized": False,
        }
    xpu_one = one_update(torch, np, xpu_bundle, witness)
    xpu_cpu_comparison = compare_training(
        np, xpu_one, cpu_one, "compiled XPU vs compiled PyTorch CPU", CPU_OUTPUT_ATOL
    )
    if not xpu_cpu_comparison["passed"]:
        return {
            "schema": PROBE_SCHEMA,
            "status": "REJECT_COMPONENT",
            "failure_reason": "XPU one-update numeric gate failed",
            "device_identity": device_identity,
            "numeric_evidence": {
                "cpu_vs_master": cpu_reference_comparison,
                "xpu_vs_cpu": xpu_cpu_comparison,
            },
            "adoption_authorized": False,
        }

    checkpoint = checkpoint_probe(
        torch, np, xpu_bundle, xpu_one, master, snapshot, witness,
        contract_sha256, system_sha256, args.checkpoint_output,
    )
    cpu_benchmark = benchmark(
        torch, cpu_bundle, master["weights"], witness,
        args.warmups, args.repeats, args.memory_sample_period_ms,
    )
    xpu_benchmark = benchmark(
        torch, xpu_bundle, master["weights"], witness,
        args.warmups, args.repeats, args.memory_sample_period_ms,
    )
    cpu_rate = cpu_benchmark["summary"]["updates_per_second"]
    xpu_rate = xpu_benchmark["summary"]["updates_per_second"]
    aggregate_updates_per_second_speedup = xpu_rate / cpu_rate
    cpu_total_p50_ns = cpu_benchmark["summary"]["total"]["p50_ns"]
    xpu_total_p50_ns = xpu_benchmark["summary"]["total"]["p50_ns"]
    total_p50_speedup = cpu_total_p50_ns / xpu_total_p50_ns
    p95_ratio = (
        xpu_benchmark["summary"]["total"]["p95_ns"]
        / cpu_benchmark["summary"]["total"]["p95_ns"]
    )
    performance_gates = {
        "xpu_total_p50_speedup_ge_1_15": total_p50_speedup >= SPEEDUP_MIN,
        "xpu_p95_no_worse": p95_ratio <= P95_RATIO_MAX,
        "numeric_gates": cpu_reference_comparison["passed"] and xpu_cpu_comparison["passed"],
        "checkpoint_sync": all((
            checkpoint["state_roundtrip_bitwise"],
            checkpoint["master_snapshot_weights_byte_identical"],
            checkpoint["normalization_unchanged"],
        )),
    }
    component_pass = all(performance_gates.values())
    source_path = Path(__file__).resolve()
    result = {
        "schema": PROBE_SCHEMA,
        "status": (
            "COMPONENT_PASS_PENDING_P_SPARSE_AND_SYSTEM_GATES"
            if component_pass else "REJECT_COMPONENT"
        ),
        "failure_reason": None if component_pass else "component performance gate failed",
        "timestamp_utc": utc_now(),
        "run_uuid": uuid.uuid4().hex,
        "scope": "HashStem component training only",
        "source_hashes": {
            "probe": sha256_file(source_path),
            "contract": contract_sha256,
            "system_contract": system_sha256,
            "master_manifest": master["manifest_sha256"],
            "master_metadata": master["metadata_sha256"],
            "master_weights": master["weights_sha256"],
            "master_optimizer": master["optimizer_sha256"],
            "source_snapshot_metadata": snapshot["sha256"],
            "source_snapshot_weights": snapshot["weights_sha256"],
            "witness": witness["npz_sha256"],
            "witness_metadata": witness["metadata_sha256"],
            **({
                "teacher_manifest": witness["teacher_binding"]["sha256"],
            } if witness["teacher_binding"] is not None else {}),
        },
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "torch_xpu_version": str(getattr(torch.version, "xpu", None)),
            "numpy": str(np.__version__),
            "compile_backend": "inductor",
            "compile_fullgraph": True,
            "compile_dynamic": False,
            "deterministic_algorithms": True,
            "dtype": "float32",
            "cpu_threads": fair_cpu_threads,
            "cpu_threads_source": "live contract-bound 8P+12E CPU Set topology P-core count",
            "seed": args.seed,
        },
        "device_identity": device_identity,
        "model": {
            "schema": HASHSTEM_SCHEMA,
            "parameters": PARAMETERS,
            "batch": BATCH,
            "input_features": INPUTS,
            "output_features": OUTPUTS,
            "master_version": 0,
            "source_snapshot_version": snapshot["metadata"]["snapshot_version"],
        },
        "numeric_evidence": {
            "cpu_vs_master": cpu_reference_comparison,
            "xpu_vs_cpu": xpu_cpu_comparison,
        },
        "checkpoint_sync": checkpoint,
        "cpu_control": cpu_benchmark,
        "xpu_candidate": xpu_benchmark,
        "performance": {
            "xpu_end_to_end_total_p50_speedup_vs_cpu": total_p50_speedup,
            "aggregate_updates_per_second_speedup_vs_cpu_supplemental": (
                aggregate_updates_per_second_speedup
            ),
            "xpu_p95_ratio_vs_cpu": p95_ratio,
            "gates": performance_gates,
            "cpu_compile_amortized": _compile_amortized(
                cpu_benchmark["compile_and_first_call_ns"], cpu_benchmark["summary"]
            ),
            "xpu_compile_amortized": _compile_amortized(
                xpu_benchmark["compile_and_first_call_ns"], xpu_benchmark["summary"]
            ),
        },
        "external_gates": {
            "p_core_sparse_slowdown_maximum": 1.10,
            "p_core_sparse_slowdown_status": "PENDING_EXTERNAL_INTEGRATED_GATE",
            "etw_residency_status": "PENDING_EXTERNAL_INTEGRATED_GATE",
            "imc_dram_bandwidth_status": "PENDING_EXTERNAL_INTEGRATED_GATE",
            "full_pipeline_status": "PENDING_EXTERNAL_INTEGRATED_GATE",
        },
        "adoption_authorized": False,
        "snapshot_publish_authorized": False,
        "old_model_beaten_claim": False,
    }
    return result


def unavailable_result(args, exc: BaseException, context: dict[str, Any] | None = None) -> dict[str, Any]:
    source = Path(__file__).resolve()
    enumerated: list[dict[str, Any]] = []
    runtime_versions: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch
        runtime_versions["torch"] = str(torch.__version__)
        runtime_versions["torch_xpu_version"] = str(getattr(torch.version, "xpu", None))
        if hasattr(torch, "xpu"):
            count = int(torch.xpu.device_count())
            runtime_versions["torch_xpu_available"] = bool(torch.xpu.is_available())
            runtime_versions["torch_xpu_device_count"] = count
            for index in range(count):
                try:
                    enumerated.append({
                        "index": index,
                        "name": str(torch.xpu.get_device_name(index)),
                        "properties": _property_receipt(torch.xpu.get_device_properties(index)),
                    })
                except Exception as enumerate_exc:
                    enumerated.append({
                        "index": index,
                        "enumeration_error": str(enumerate_exc),
                    })
    except Exception as runtime_exc:
        runtime_versions["torch_probe_error"] = str(runtime_exc)
    source_hashes = {"probe": sha256_file(source)}
    for label, path in (
        ("contract", args.contract),
        ("system_contract", args.system_contract),
        ("source_snapshot_metadata", args.snapshot_metadata),
        ("source_snapshot_weights", args.snapshot_weights),
        ("witness", args.witness),
        ("witness_metadata", args.witness_metadata),
        ("teacher_manifest", args.teacher_manifest),
    ):
        try:
            if path is not None and path.is_file():
                source_hashes[label] = sha256_file(path)
        except OSError:
            pass
    result = {
        "schema": PROBE_SCHEMA,
        "status": "UNAVAILABLE_FAIL_CLOSED",
        "failure_reason": str(exc),
        "exception_type": type(exc).__name__,
        "timestamp_utc": utc_now(),
        "requested_backend": "native Windows PyTorch XPU",
        "enumerated_backends": enumerated,
        "device_identity": getattr(exc, "device_identity_receipt", None),
        "runtime_versions": runtime_versions,
        "source_hashes": source_hashes,
        "adoption_authorized": False,
        "snapshot_publish_authorized": False,
    }
    if context:
        result.update(context)
    return result


def rejected_result(exc: BaseException) -> dict[str, Any]:
    return {
        "schema": PROBE_SCHEMA,
        "status": "REJECT_COMPONENT",
        "failure_reason": str(exc),
        "exception_type": type(exc).__name__,
        "traceback": traceback.format_exc(),
        "timestamp_utc": utc_now(),
        "device_identity": getattr(exc, "device_identity_receipt", None),
        "adoption_authorized": False,
        "snapshot_publish_authorized": False,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--system-contract", type=Path, required=True)
    parser.add_argument("--system-contract-sha256", required=True)
    parser.add_argument("--master-checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot-metadata", type=Path, required=True)
    parser.add_argument("--snapshot-metadata-sha256", required=True)
    parser.add_argument("--snapshot-weights", type=Path, required=True)
    parser.add_argument("--snapshot-weights-sha256", required=True)
    parser.add_argument("--witness", type=Path, required=True)
    parser.add_argument("--witness-metadata", type=Path, required=True)
    parser.add_argument("--witness-metadata-sha256", required=True)
    parser.add_argument("--teacher-manifest", type=Path)
    parser.add_argument("--teacher-manifest-sha256")
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--memory-sample-period-ms", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0x485354454D585055)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite probe output")
    try:
        result = run_probe(args)
        write_fresh_json(args.output, result)
        raise SystemExit(0 if result.get("status") == "COMPONENT_PASS_PENDING_P_SPARSE_AND_SYSTEM_GATES" else 3)
    except BackendUnavailable as exc:
        write_fresh_json(args.output, unavailable_result(args, exc))
        raise SystemExit(2)
    except SystemExit:
        raise
    except Exception as exc:
        write_fresh_json(args.output, rejected_result(exc))
        raise SystemExit(3)


if __name__ == "__main__":
    main()
