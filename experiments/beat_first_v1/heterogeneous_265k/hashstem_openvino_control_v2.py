"""Fair OpenVINO CPU/NPU execution boundary for Learned HashStem v2.

This module is intentionally separate from the frozen post-idle v1 source
closure.  Full batches compile the *same* immutable ``hashstem_b16.xml`` on
explicit OpenVINO ``CPU`` and ``NPU`` devices.  A remainder is executed at its
actual row count by the matching dynamic IR on explicit ``CPU``; it is never
padded into an accelerator call.

The scalar Julia ``hashstem_reference!`` output may be supplied as a numerical
oracle, but it is never called or timed here.  The CPU performance control is
the optimized OpenVINO CPU plugin using a persistent compiled model, persistent
InferRequest, and preallocated tensors.

OpenVINO's portable Python host-tensor API does not expose the NPU DMA engine as
independently synchronizable H2D/D2H events. Accordingly, receipts distinguish
the NPU input ``Tensor.copy_from`` from the prebound shared-host output and from
unmeasured hardware transfer. Actual device transfer is included in the
measured submit/wait/end-to-end boundary, and ``dma_timing_isolated`` is always
false. No kernel-only or falsely isolated DMA number is emitted.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import openvino as ov

try:
    from benchmark_hashstem_openvino import (
        BATCH,
        INPUTS,
        OUTPUTS,
        SCHEMA,
        percentile,
        route_conformance,
        sha256_file,
    )
except ImportError:  # pragma: no cover - package-style import used by tooling
    from .benchmark_hashstem_openvino import (
        BATCH,
        INPUTS,
        OUTPUTS,
        SCHEMA,
        percentile,
        route_conformance,
        sha256_file,
    )


RESULT_SCHEMA = "learned-hashstem-openvino-control-v2"
PROVENANCE_SCHEMA = "learned-hashstem-openvino-witness-provenance-v2"
FLOAT_BYTES = np.dtype(np.float32).itemsize
STAGES = ("pack", "bind", "h2d", "submit", "wait", "d2h", "copy")
SOURCE_PATH = Path(__file__).resolve()
REUSED_V1_SOURCE_PATH = SOURCE_PATH.with_name("benchmark_hashstem_openvino.py")

# The label is the total active-neuron budget.  Per-layer widths are frozen by
# K_SWEEP_EXPERIMENT.md and deliberately exclude the old k=(26,22,22) baseline.
SPARSE_K_TO_ACTIVE_COUNTS: dict[int, tuple[int, int, int]] = {
    64: (24, 20, 20),
    128: (48, 40, 40),
    256: (96, 80, 80),
}

FORBIDDEN_VALIDATION_SEEDS = frozenset(range(8001, 8009))
FORBIDDEN_SEALED_SEEDS = frozenset(range(91001, 91033))


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _nearest_rank(values: Sequence[int], numerator: int, denominator: int) -> int:
    if not values:
        raise ValueError("cannot summarize an empty timing sequence")
    if numerator <= 0 or denominator <= 0 or numerator > denominator:
        raise ValueError("invalid nearest-rank percentile")
    ordered = sorted(int(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(numerator * len(ordered) / denominator) - 1))
    return ordered[index]


def _read_json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw


def load_property_map(path: Path | None, default: Mapping[str, object]) -> tuple[dict[str, Any], str | None]:
    """Load an exact compile-property map without permitting AUTO routing."""

    if path is None:
        properties = dict(default)
        source_sha256 = None
    else:
        properties, raw = _read_json_object(path, "OpenVINO property file")
        source_sha256 = hashlib.sha256(raw).hexdigest()
    for key, value in properties.items():
        if not isinstance(key, str) or not key:
            raise ValueError("OpenVINO property keys must be nonempty strings")
        key_upper = key.upper()
        if key_upper in {"DEVICE_PRIORITIES", "MULTI_DEVICE_PRIORITIES"}:
            raise ValueError("AUTO/MULTI device routing properties are forbidden")
        if isinstance(value, str) and "AUTO" in value.upper():
            raise ValueError("AUTO device routing is forbidden in property values")
    return properties, source_sha256


def validate_snapshot_binding(
    metadata_path: Path,
    fixed_ir: Path,
    dynamic_cpu_ir: Path,
) -> tuple[dict[str, Any], bytes]:
    metadata, raw = _read_json_object(metadata_path, "snapshot metadata")
    fixed_bin = fixed_ir.with_suffix(".bin")
    dynamic_bin = dynamic_cpu_ir.with_suffix(".bin")
    if metadata.get("schema") != SCHEMA:
        raise ValueError("snapshot metadata schema mismatch")
    if metadata.get("fixed_batch") != BATCH:
        raise ValueError("snapshot fixed batch mismatch")
    if metadata.get("input_features") != INPUTS or metadata.get("output_features") != OUTPUTS:
        raise ValueError("snapshot input/output geometry mismatch")
    if not isinstance(metadata.get("snapshot_version"), int) or metadata["snapshot_version"] <= 0:
        raise ValueError("snapshot version must be positive")
    for field in (
        "weights_sha256",
        "normalization_sha256",
        "fixed_xml_sha256",
        "fixed_bin_sha256",
        "dynamic_xml_sha256",
        "dynamic_bin_sha256",
    ):
        if not _valid_sha256(metadata.get(field)):
            raise ValueError(f"snapshot metadata has invalid {field}")
    expected = {
        fixed_ir: metadata["fixed_xml_sha256"],
        fixed_bin: metadata["fixed_bin_sha256"],
        dynamic_cpu_ir: metadata["dynamic_xml_sha256"],
        dynamic_bin: metadata["dynamic_bin_sha256"],
    }
    for path, digest in expected.items():
        if not path.is_file():
            raise FileNotFoundError(f"snapshot member is missing: {path}")
        if sha256_file(path) != digest:
            raise ValueError(f"snapshot member SHA-256 mismatch: {path.name}")
    # Compatibility aliases must bind the exact same fixed IR, not a second
    # supposedly equivalent graph.
    if metadata.get("xml_sha256") != metadata["fixed_xml_sha256"]:
        raise ValueError("snapshot fixed XML alias mismatch")
    if metadata.get("bin_sha256") != metadata["fixed_bin_sha256"]:
        raise ValueError("snapshot fixed BIN alias mismatch")
    return metadata, raw


def validate_witness_provenance(
    provenance_path: Path,
    witness_path: Path,
    candidate_count: int,
    dataset_manifest_path: Path,
    expected_dataset_manifest_sha256: str,
    witness_generator_source: Path,
    expected_witness_generator_source_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    provenance, raw = _read_json_object(provenance_path, "witness provenance")
    for digest, label in (
        (expected_dataset_manifest_sha256, "expected dataset-manifest SHA-256"),
        (expected_witness_generator_source_sha256, "expected witness-generator SHA-256"),
    ):
        if not _valid_sha256(digest) or digest != digest.lower():
            raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    observed_manifest_sha256 = sha256_file(dataset_manifest_path)
    observed_generator_sha256 = sha256_file(witness_generator_source)
    if observed_manifest_sha256 != expected_dataset_manifest_sha256:
        raise ValueError("externally expected dataset-manifest SHA-256 mismatch")
    if observed_generator_sha256 != expected_witness_generator_source_sha256:
        raise ValueError("externally expected witness-generator SHA-256 mismatch")
    if provenance.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError("witness provenance schema mismatch")
    if provenance.get("split") != "teacher_v3_train":
        raise ValueError("v2 benchmark accepts only the teacher_v3 train split")
    if provenance.get("witness_sha256") != sha256_file(witness_path):
        raise ValueError("witness provenance SHA-256 mismatch")
    if provenance.get("candidate_count") != candidate_count:
        raise ValueError("witness provenance candidate count mismatch")
    if provenance.get("dataset_manifest_sha256") != observed_manifest_sha256:
        raise ValueError("witness does not bind the externally verified dataset manifest")
    if provenance.get("witness_generator_source_sha256") != observed_generator_sha256:
        raise ValueError("witness does not bind the externally verified generator source")
    manifest, _ = _read_json_object(dataset_manifest_path, "teacher_v3 dataset manifest")
    if manifest.get("format_version") != 3:
        raise ValueError("teacher_v3 dataset manifest format mismatch")
    run_metadata = manifest.get("run_metadata")
    if not isinstance(run_metadata, dict):
        raise ValueError("teacher_v3 manifest run metadata missing")
    if run_metadata.get("held_out_development_validation_sealed_seeds_used") is not False:
        raise ValueError("teacher_v3 manifest does not attest held-out seed exclusion")
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("teacher_v3 manifest parts are missing")
    indexed_parts: dict[str, dict[str, Any]] = {}
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError("teacher_v3 manifest part is malformed")
        episode_key = part.get("episode_key")
        if not isinstance(episode_key, str) or not episode_key or episode_key in indexed_parts:
            raise ValueError("teacher_v3 manifest episode key is invalid or duplicated")
        indexed_parts[episode_key] = part
    source_parts = provenance.get("source_parts")
    if not isinstance(source_parts, list) or not source_parts:
        raise ValueError("witness provenance must bind source teacher_v3 parts")
    observed_source_seeds: set[int] = set()
    observed_episode_keys: set[str] = set()
    dataset_root = dataset_manifest_path.parent.resolve()
    for binding in source_parts:
        if not isinstance(binding, dict) or set(binding) != {"episode_key", "part_sha256"}:
            raise ValueError("witness source-part binding is malformed")
        episode_key = binding["episode_key"]
        part_sha256 = binding["part_sha256"]
        if not isinstance(episode_key, str) or episode_key in observed_episode_keys:
            raise ValueError("witness source episode key is invalid or duplicated")
        part = indexed_parts.get(episode_key)
        if part is None:
            raise ValueError("witness source episode is absent from teacher_v3 manifest")
        if part.get("split") != "train" or not episode_key.startswith("v3|train|"):
            raise ValueError("witness source part is not in the teacher_v3 train split")
        if not _valid_sha256(part.get("sha256")) or part_sha256 != part["sha256"]:
            raise ValueError("witness source-part SHA-256 binding mismatch")
        relative_text = part.get("relative_path")
        if not isinstance(relative_text, str) or not relative_text:
            raise ValueError("teacher_v3 source-part path is missing")
        relative_path = Path(relative_text)
        if relative_path.is_absolute():
            raise ValueError("teacher_v3 source-part path must be relative")
        resolved_part = (dataset_root / relative_path).resolve()
        try:
            resolved_part.relative_to(dataset_root)
        except ValueError as error:
            raise ValueError("teacher_v3 source-part path escapes the dataset root") from error
        if not resolved_part.is_file():
            raise FileNotFoundError("teacher_v3 source-part file is missing")
        byte_length = part.get("bytes")
        if isinstance(byte_length, bool) or not isinstance(byte_length, int) or byte_length <= 0:
            raise ValueError("teacher_v3 source-part byte receipt is invalid")
        if resolved_part.stat().st_size != byte_length or sha256_file(resolved_part) != part_sha256:
            raise ValueError("teacher_v3 source-part bytes differ from the bound manifest")
        seed = part.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("teacher_v3 source seed is invalid")
        observed_source_seeds.add(seed)
        observed_episode_keys.add(episode_key)
    seeds = provenance.get("environment_seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("witness provenance must list environment seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("environment seeds must be integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("environment seeds must not be duplicated")
    if set(seeds) != observed_source_seeds:
        raise ValueError("environment seeds do not equal the manifest-bound source-part seeds")
    forbidden = (set(seeds) & FORBIDDEN_VALIDATION_SEEDS) | (set(seeds) & FORBIDDEN_SEALED_SEEDS)
    if forbidden:
        raise ValueError(f"validation/sealed environment seeds are forbidden: {sorted(forbidden)}")
    return provenance, raw


def load_witness(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    route_keys = tuple(
        name
        for layer in (1, 2, 3)
        for name in (
            f"probe_ids_l{layer}",
            f"probe_keys_l{layer}",
            f"query_residual_l{layer}",
        )
    )
    with np.load(path, allow_pickle=False) as archive:
        required = {"input", "reference_output", *route_keys}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"witness is missing required arrays: {missing}")
        packed = np.array(archive["input"], dtype=np.float32, order="C", copy=True)
        reference = np.array(archive["reference_output"], dtype=np.float32, order="C", copy=True)
        route_witness = {
            "reference_output": reference,
            **{name: np.array(archive[name], copy=True) for name in route_keys},
        }
    if packed.ndim != 2 or packed.shape[1] != INPUTS or packed.shape[0] <= 0:
        raise ValueError(f"witness input must be [N,{INPUTS}] with N > 0")
    if reference.shape != (packed.shape[0], OUTPUTS):
        raise ValueError("witness reference output shape mismatch")
    if not np.all(np.isfinite(packed)) or not np.all(np.isfinite(reference)):
        raise ValueError("witness input/reference contains non-finite values")
    return packed, reference, route_witness


def _execution_devices(compiled: object) -> list[str]:
    raw = compiled.get_property("EXECUTION_DEVICES")
    if isinstance(raw, str):
        values = [raw]
    else:
        values = list(raw)
    return [str(value).upper() for value in values]


@dataclass(frozen=True)
class CompiledReceipt:
    requested_device: str
    execution_devices: tuple[str, ...]
    compile_ns: int
    ir_xml_sha256: str
    ir_bin_sha256: str
    properties: dict[str, Any]


def compile_exact(
    core: object,
    model_path: Path,
    device: str,
    properties: Mapping[str, object],
) -> tuple[object, CompiledReceipt]:
    requested = device.upper()
    available = [str(value).upper() for value in core.available_devices]
    if requested not in available:
        raise RuntimeError(f"required exact device {requested} unavailable: {available}")
    if requested not in {"CPU", "NPU"}:
        raise ValueError("v2 HashStem boundary permits only explicit CPU or NPU")
    started = time.perf_counter_ns()
    compiled = core.compile_model(str(model_path), requested, dict(properties))
    finished = time.perf_counter_ns()
    execution = _execution_devices(compiled)
    if execution != [requested]:
        raise RuntimeError(
            f"compiled graph did not bind exclusively to {requested}: {execution}"
        )
    receipt = CompiledReceipt(
        requested_device=requested,
        execution_devices=tuple(execution),
        compile_ns=finished - started,
        ir_xml_sha256=sha256_file(model_path),
        ir_bin_sha256=sha256_file(model_path.with_suffix(".bin")),
        properties=dict(properties),
    )
    return compiled, receipt


@dataclass(frozen=True)
class CallTrace:
    path: str
    call_kind: str
    requested_device: str
    execution_devices: tuple[str, ...]
    sample_ordinal: int
    call_ordinal: int
    row_begin: int
    row_end: int
    intervals: dict[str, tuple[int, int]]
    call_begin_ns: int
    call_end_ns: int
    input_bytes: int
    output_bytes: int
    io_mode: str
    setup_bind_ns: int

    def to_receipt(self) -> dict[str, Any]:
        rows = self.row_end - self.row_begin
        stages_ns = {
            stage: self.intervals[stage][1] - self.intervals[stage][0] for stage in STAGES
        }
        is_npu = self.requested_device == "NPU"
        per_call_api_counts = {
            "pack": 1,
            "bind": 0,
            "h2d": 1 if is_npu else 0,
            "submit": 1,
            "wait": 1,
            "d2h": 0,
            "copy": 1,
        }
        return {
            "schema": "learned-hashstem-openvino-physical-call-v2",
            "path": self.path,
            "call_kind": self.call_kind,
            "requested_device": self.requested_device,
            "execution_devices": list(self.execution_devices),
            "sample_ordinal": self.sample_ordinal,
            "call_ordinal": self.call_ordinal,
            "row_begin_zero_based": self.row_begin,
            "row_end_exclusive": self.row_end,
            "rows": rows,
            "fixed_batch_16": rows == BATCH and self.call_kind != "CPU_TAIL",
            "actual_length_cpu_tail": self.call_kind == "CPU_TAIL",
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "host_to_runtime_tensor_copy_bytes": self.input_bytes if is_npu else 0,
            "runtime_tensor_to_host_copy_bytes": 0,
            "shared_host_input_bytes": 0 if is_npu else self.input_bytes,
            "shared_host_output_bytes": self.output_bytes,
            "logical_h2d_payload_bytes": self.input_bytes if is_npu else 0,
            "logical_d2h_payload_bytes": self.output_bytes if is_npu else 0,
            "hardware_dma_bytes_measured": False,
            "submission_count": 1,
            "synchronization_count": 1,
            "io_mode": self.io_mode,
            "persistent_setup_bind_ns": self.setup_bind_ns,
            "per_call_api_counts": per_call_api_counts,
            "physical_api_calls": {
                "pack": "numpy.copyto(application_pack_buffer, witness_view)",
                "bind": "persistent input/output prebind outside steady call",
                "h2d": (
                    "openvino.Tensor.copy_from(application_pack_buffer)"
                    if is_npu
                    else "not applicable: CPU Tensor shares the application pack buffer"
                ),
                "submit": "InferRequest.start_async",
                "wait": "InferRequest.wait",
                "d2h": "prebound host output becomes visible when InferRequest.wait completes",
                "copy": "numpy.copyto(application_output, shared_host_output_buffer)",
            },
            "dma_timing_isolated": False,
            "dma_timing_note": (
                "Portable OpenVINO Python host tensors do not expose independently "
                "synchronizable NPU DMA; actual transfer remains included in submit/wait/E2E."
            ),
            "call_begin_perf_counter_ns": self.call_begin_ns,
            "call_end_perf_counter_ns": self.call_end_ns,
            "end_to_end_ns": self.call_end_ns - self.call_begin_ns,
            "intervals_perf_counter_ns": {
                stage: [self.intervals[stage][0], self.intervals[stage][1]] for stage in STAGES
            },
            "stage_ns": stages_ns,
        }


class PersistentOpenVINOCall:
    """One persistent request/tensor set for one exact device and row count."""

    def __init__(
        self,
        compiled: object,
        compile_receipt: CompiledReceipt,
        rows: int,
        call_kind: str,
    ) -> None:
        if rows <= 0 or rows > BATCH:
            raise ValueError("physical HashStem call rows must be in 1:16")
        if call_kind == "CPU_TAIL" and not (rows < BATCH and compile_receipt.requested_device == "CPU"):
            raise ValueError("CPU_TAIL must be an actual-length CPU call shorter than 16")
        if call_kind != "CPU_TAIL" and rows != BATCH:
            raise ValueError("non-tail HashStem calls must use fixed batch 16")
        self.compiled = compiled
        self.compile_receipt = compile_receipt
        self.rows = rows
        self.call_kind = call_kind
        self.request = compiled.create_infer_request()
        self.pack_buffer = np.empty((rows, INPUTS), dtype=np.float32, order="C")
        self.output_buffer = np.empty((rows, OUTPUTS), dtype=np.float32, order="C")
        if compile_receipt.requested_device == "CPU":
            # Optimized CPU control: packing writes directly into the Tensor's
            # shared host allocation. There is no redundant steady-state H2D
            # copy merely to make the CPU baseline look slower.
            self.io_mode = "CPU_SHARED_HOST_PREBOUND"
            self.input_tensor = ov.Tensor(self.pack_buffer, shared_memory=True)
        else:
            self.io_mode = "NPU_PERSISTENT_RUNTIME_INPUT_PREBOUND_HOST_OUTPUT"
            self.input_tensor = ov.Tensor(ov.Type.f32, [rows, INPUTS])
        self.host_output_tensor = ov.Tensor(self.output_buffer, shared_memory=True)
        setup_begin = time.perf_counter_ns()
        self.request.set_input_tensor(self.input_tensor)
        self.request.set_output_tensor(self.host_output_tensor)
        setup_end = time.perf_counter_ns()
        self.setup_bind_ns = setup_end - setup_begin

    @staticmethod
    def _timed(intervals: dict[str, tuple[int, int]], stage: str, function):
        started = time.perf_counter_ns()
        value = function()
        finished = time.perf_counter_ns()
        intervals[stage] = (started, finished)
        return value

    @staticmethod
    def _mark(intervals: dict[str, tuple[int, int]], stage: str) -> None:
        tick = time.perf_counter_ns()
        intervals[stage] = (tick, tick)

    def run(
        self,
        source: np.ndarray,
        destination: np.ndarray,
        *,
        path: str,
        sample_ordinal: int,
        call_ordinal: int,
        row_begin: int,
    ) -> CallTrace:
        if source.shape != (self.rows, INPUTS) or source.dtype != np.float32:
            raise ValueError("physical-call source shape/dtype mismatch")
        if destination.shape != (self.rows, OUTPUTS) or destination.dtype != np.float32:
            raise ValueError("physical-call destination shape/dtype mismatch")
        intervals: dict[str, tuple[int, int]] = {}
        call_begin = time.perf_counter_ns()
        self._timed(
            intervals,
            "pack",
            lambda: np.copyto(self.pack_buffer, source, casting="no"),
        )
        self._mark(intervals, "bind")
        if self.compile_receipt.requested_device == "NPU":
            self._timed(
                intervals,
                "h2d",
                lambda: self.input_tensor.copy_from(self.pack_buffer),
            )
        else:
            self._mark(intervals, "h2d")
        self._timed(intervals, "submit", self.request.start_async)
        self._timed(intervals, "wait", self.request.wait)
        # Output is prebound to a shared host Tensor. Any opaque NPU D2H is
        # completed by wait; the portable API has no second synchronizable DMA
        # event, so the explicit D2H interval is a truthful zero marker.
        self._mark(intervals, "d2h")

        def copy_application_output() -> None:
            if self.output_buffer.dtype != np.float32 or self.output_buffer.shape != destination.shape:
                raise ValueError("OpenVINO host output tensor shape/dtype mismatch")
            np.copyto(destination, self.output_buffer, casting="no")

        self._timed(intervals, "copy", copy_application_output)
        call_end = time.perf_counter_ns()
        previous = call_begin
        for stage in STAGES:
            begin, end = intervals[stage]
            if not (previous <= begin <= end <= call_end):
                raise RuntimeError(f"non-monotonic physical-call interval for {stage}")
            previous = end
        return CallTrace(
            path=path,
            call_kind=self.call_kind,
            requested_device=self.compile_receipt.requested_device,
            execution_devices=self.compile_receipt.execution_devices,
            sample_ordinal=sample_ordinal,
            call_ordinal=call_ordinal,
            row_begin=row_begin,
            row_end=row_begin + self.rows,
            intervals=intervals,
            call_begin_ns=call_begin,
            call_end_ns=call_end,
            input_bytes=self.rows * INPUTS * FLOAT_BYTES,
            output_bytes=self.rows * OUTPUTS * FLOAT_BYTES,
            io_mode=self.io_mode,
            setup_bind_ns=self.setup_bind_ns,
        )


@dataclass(frozen=True)
class LogicalTrace:
    path: str
    sample_ordinal: int
    begin_ns: int
    end_ns: int
    calls: tuple[CallTrace, ...]

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": "learned-hashstem-openvino-logical-sample-v2",
            "path": self.path,
            "sample_ordinal": self.sample_ordinal,
            "begin_perf_counter_ns": self.begin_ns,
            "end_perf_counter_ns": self.end_ns,
            "end_to_end_ns": self.end_ns - self.begin_ns,
            "physical_call_count": len(self.calls),
            "physical_calls": [call.to_receipt() for call in self.calls],
        }


class HashStemPath:
    """A CPU-control or NPU-plus-CPU-tail logical execution path."""

    def __init__(
        self,
        name: str,
        full_call: PersistentOpenVINOCall,
        tail_call: PersistentOpenVINOCall | None,
        candidate_count: int,
    ) -> None:
        if candidate_count <= 0:
            raise ValueError("candidate count must be positive")
        expected_tail = candidate_count % BATCH
        if expected_tail == 0 and tail_call is not None:
            raise ValueError("unexpected CPU tail for a multiple of 16")
        if expected_tail and (tail_call is None or tail_call.rows != expected_tail):
            raise ValueError("missing or wrong-sized actual-length CPU tail")
        if full_call.rows != BATCH:
            raise ValueError("full HashStem call must use batch 16")
        self.name = name
        self.full_call = full_call
        self.tail_call = tail_call
        self.candidate_count = candidate_count
        self.full_rows = candidate_count - expected_tail
        self.output = np.empty((candidate_count, OUTPUTS), dtype=np.float32, order="C")

    def run(self, packed: np.ndarray, sample_ordinal: int) -> LogicalTrace:
        if packed.shape != (self.candidate_count, INPUTS) or packed.dtype != np.float32:
            raise ValueError("logical HashStem input shape/dtype mismatch")
        begin = time.perf_counter_ns()
        traces: list[CallTrace] = []
        call_ordinal = 0
        for row_begin in range(0, self.full_rows, BATCH):
            call_ordinal += 1
            traces.append(
                self.full_call.run(
                    packed[row_begin : row_begin + BATCH],
                    self.output[row_begin : row_begin + BATCH],
                    path=self.name,
                    sample_ordinal=sample_ordinal,
                    call_ordinal=call_ordinal,
                    row_begin=row_begin,
                )
            )
        if self.tail_call is not None:
            call_ordinal += 1
            traces.append(
                self.tail_call.run(
                    packed[self.full_rows :],
                    self.output[self.full_rows :],
                    path=self.name,
                    sample_ordinal=sample_ordinal,
                    call_ordinal=call_ordinal,
                    row_begin=self.full_rows,
                )
            )
        end = time.perf_counter_ns()
        if not traces:
            raise RuntimeError("logical HashStem path issued no physical call")
        if traces[0].call_begin_ns < begin or traces[-1].call_end_ns > end:
            raise RuntimeError("physical call escaped logical end-to-end interval")
        return LogicalTrace(self.name, sample_ordinal, begin, end, tuple(traces))


def _compile_receipt_json(receipt: CompiledReceipt) -> dict[str, Any]:
    return {
        "requested_device": receipt.requested_device,
        "execution_devices": list(receipt.execution_devices),
        "compile_ns": receipt.compile_ns,
        "ir_xml_sha256": receipt.ir_xml_sha256,
        "ir_bin_sha256": receipt.ir_bin_sha256,
        "properties": receipt.properties,
        "properties_sha256": _canonical_json_sha256(receipt.properties),
    }


def _summarize_logical_traces(traces: Sequence[LogicalTrace], candidate_count: int) -> dict[str, Any]:
    if not traces:
        raise ValueError("no logical traces to summarize")
    end_to_end = [trace.end_ns - trace.begin_ns for trace in traces]
    calls = [call for trace in traces for call in trace.calls]
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in sorted({call.call_kind for call in calls}):
        selected = [call for call in calls if call.call_kind == kind]
        stage_values = {
            stage: [call.intervals[stage][1] - call.intervals[stage][0] for call in selected]
            for stage in STAGES
        }
        by_kind[kind] = {
            "physical_calls": len(selected),
            "stage_p50_ns": {
                stage: int(statistics.median(values)) for stage, values in stage_values.items()
            },
            "stage_p95_ns": {
                stage: percentile(values, 0.95) for stage, values in stage_values.items()
            },
            "call_end_to_end_p50_ns": int(
                statistics.median([call.call_end_ns - call.call_begin_ns for call in selected])
            ),
            "call_end_to_end_p95_ns": percentile(
                [call.call_end_ns - call.call_begin_ns for call in selected], 0.95
            ),
        }
    total_ns = sum(end_to_end)
    return {
        "logical_samples": len(traces),
        "candidates": len(traces) * candidate_count,
        "physical_calls": len(calls),
        "logical_end_to_end_p50_ns": int(statistics.median(end_to_end)),
        "logical_end_to_end_p95_ns": _nearest_rank(end_to_end, 95, 100),
        "logical_end_to_end_total_ns": total_ns,
        "candidates_per_second": len(traces) * candidate_count * 1.0e9 / total_ns,
        "by_call_kind": by_kind,
    }


def _trace_receipts_exact(
    traces: Sequence[LogicalTrace],
    *,
    path: str,
    candidate_count: int,
    repeats: int,
    full_kind: str,
    full_device: str,
) -> bool:
    """Independently close call count, row coverage, device, and stage shape."""

    if len(traces) != repeats or [trace.sample_ordinal for trace in traces] != list(
        range(1, repeats + 1)
    ):
        return False
    expected_full = candidate_count // BATCH
    tail_rows = candidate_count % BATCH
    expected_calls = expected_full + int(tail_rows > 0)
    for trace in traces:
        if trace.path != path or len(trace.calls) != expected_calls:
            return False
        if trace.begin_ns > trace.end_ns:
            return False
        cursor = 0
        previous_call_end = trace.begin_ns
        for ordinal, call in enumerate(trace.calls, start=1):
            if call.path != path or call.sample_ordinal != trace.sample_ordinal:
                return False
            if call.call_ordinal != ordinal or call.row_begin != cursor:
                return False
            if call.call_begin_ns < trace.begin_ns or call.call_end_ns > trace.end_ns:
                return False
            if call.call_begin_ns < previous_call_end:
                return False
            if set(call.intervals) != set(STAGES):
                return False
            previous = call.call_begin_ns
            for stage in STAGES:
                begin, end = call.intervals[stage]
                if not (previous <= begin <= end <= call.call_end_ns):
                    return False
                previous = end
            rows = call.row_end - call.row_begin
            if call.input_bytes != rows * INPUTS * FLOAT_BYTES:
                return False
            if call.output_bytes != rows * OUTPUTS * FLOAT_BYTES:
                return False
            if ordinal <= expected_full:
                expected_io = (
                    "CPU_SHARED_HOST_PREBOUND"
                    if full_device == "CPU"
                    else "NPU_PERSISTENT_RUNTIME_INPUT_PREBOUND_HOST_OUTPUT"
                )
                if not (
                    rows == BATCH
                    and call.call_kind == full_kind
                    and call.requested_device == full_device
                    and call.execution_devices == (full_device,)
                    and call.io_mode == expected_io
                    and call.setup_bind_ns >= 0
                ):
                    return False
            else:
                if not (
                    tail_rows > 0
                    and rows == tail_rows
                    and call.call_kind == "CPU_TAIL"
                    and call.requested_device == "CPU"
                    and call.execution_devices == ("CPU",)
                    and call.io_mode == "CPU_SHARED_HOST_PREBOUND"
                    and call.setup_bind_ns >= 0
                ):
                    return False
            cursor = call.row_end
            previous_call_end = call.call_end_ns
        if cursor != candidate_count:
            return False
    return True


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("numeric comparison shape mismatch")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)), initial=0.0))


def benchmark_control(
    *,
    fixed_ir: Path,
    dynamic_cpu_ir: Path,
    snapshot_metadata: Path,
    witness_path: Path,
    witness_provenance: Path,
    dataset_manifest_path: Path,
    expected_dataset_manifest_sha256: str,
    witness_generator_source: Path,
    expected_witness_generator_source_sha256: str,
    sparse_k: int,
    warmups: int,
    repeats: int,
    cpu_properties: Mapping[str, object],
    npu_properties: Mapping[str, object],
) -> dict[str, Any]:
    """Run the component boundary; it deliberately performs no sparse layer."""

    if sparse_k not in SPARSE_K_TO_ACTIVE_COUNTS:
        raise ValueError("sparse_k must be exactly one of 64, 128, or 256")
    if warmups < 1 or repeats < 10:
        raise ValueError("benchmark requires >=1 warmup and >=10 timed repeats")
    metadata, metadata_raw = validate_snapshot_binding(
        snapshot_metadata, fixed_ir, dynamic_cpu_ir
    )
    packed, reference, route_witness = load_witness(witness_path)
    provenance, provenance_raw = validate_witness_provenance(
        witness_provenance,
        witness_path,
        packed.shape[0],
        dataset_manifest_path,
        expected_dataset_manifest_sha256,
        witness_generator_source,
        expected_witness_generator_source_sha256,
    )
    candidate_count = packed.shape[0]
    tail_rows = candidate_count % BATCH

    core = ov.Core()
    cpu_fixed, cpu_fixed_receipt = compile_exact(core, fixed_ir, "CPU", cpu_properties)
    npu_fixed, npu_fixed_receipt = compile_exact(core, fixed_ir, "NPU", npu_properties)
    # Strong fairness: CPU and NPU full calls bind the byte-identical fixed IR.
    if (
        cpu_fixed_receipt.ir_xml_sha256 != npu_fixed_receipt.ir_xml_sha256
        or cpu_fixed_receipt.ir_bin_sha256 != npu_fixed_receipt.ir_bin_sha256
    ):
        raise RuntimeError("CPU and NPU full calls did not compile the same fixed IR")

    cpu_tail = None
    cpu_dynamic_receipt = None
    if tail_rows:
        cpu_dynamic, cpu_dynamic_receipt = compile_exact(
            core, dynamic_cpu_ir, "CPU", cpu_properties
        )
        cpu_tail = PersistentOpenVINOCall(
            cpu_dynamic, cpu_dynamic_receipt, tail_rows, "CPU_TAIL"
        )

    cpu_full_call = PersistentOpenVINOCall(
        cpu_fixed, cpu_fixed_receipt, BATCH, "CPU_FIXED_B16"
    )
    npu_full_call = PersistentOpenVINOCall(
        npu_fixed, npu_fixed_receipt, BATCH, "NPU_FIXED_B16"
    )
    cpu_path = HashStemPath("CPU_CONTROL", cpu_full_call, cpu_tail, candidate_count)
    npu_path = HashStemPath("NPU_PLUS_CPU_TAIL", npu_full_call, cpu_tail, candidate_count)

    first_cpu = cpu_path.run(packed, sample_ordinal=0)
    first_cpu_output = cpu_path.output.copy()
    first_npu = npu_path.run(packed, sample_ordinal=0)
    first_npu_output = npu_path.output.copy()
    for warmup in range(1, warmups + 1):
        # Alternate order to avoid giving one backend every first/last thermal slot.
        order = (cpu_path, npu_path) if warmup % 2 else (npu_path, cpu_path)
        for path in order:
            path.run(packed, sample_ordinal=-warmup)

    cpu_traces: list[LogicalTrace] = []
    npu_traces: list[LogicalTrace] = []
    for repeat in range(1, repeats + 1):
        order = (cpu_path, npu_path) if repeat % 2 else (npu_path, cpu_path)
        for path in order:
            trace = path.run(packed, sample_ordinal=repeat)
            if path is cpu_path:
                cpu_traces.append(trace)
            else:
                npu_traces.append(trace)
    cpu_output = cpu_path.output.copy()
    npu_output = npu_path.output.copy()
    if not np.all(np.isfinite(cpu_output)) or not np.all(np.isfinite(npu_output)):
        raise ValueError("OpenVINO output contains non-finite values")

    full_rows = candidate_count - tail_rows
    cpu_error = _max_abs(cpu_output, reference)
    npu_full_error = _max_abs(npu_output[:full_rows], reference[:full_rows]) if full_rows else 0.0
    cpu_tail_error = _max_abs(cpu_output[full_rows:], reference[full_rows:]) if tail_rows else 0.0
    npu_path_tail_error = (
        _max_abs(npu_output[full_rows:], reference[full_rows:]) if tail_rows else 0.0
    )
    cross_backend_full_error = (
        _max_abs(npu_output[:full_rows], cpu_output[:full_rows]) if full_rows else 0.0
    )
    first_call_cpu_error = _max_abs(first_cpu_output, reference)
    first_call_npu_full_error = (
        _max_abs(first_npu_output[:full_rows], reference[:full_rows]) if full_rows else 0.0
    )

    active_counts = SPARSE_K_TO_ACTIVE_COUNTS[sparse_k]
    cpu_route_matches, cpu_route_total = route_conformance(
        route_witness, cpu_output, active_counts
    )
    npu_route_matches, npu_route_total = route_conformance(
        route_witness, npu_output, active_counts
    )
    cpu_summary = _summarize_logical_traces(cpu_traces, candidate_count)
    npu_summary = _summarize_logical_traces(npu_traces, candidate_count)
    cpu_p50_ns = cpu_summary["logical_end_to_end_p50_ns"]
    npu_p50_ns = npu_summary["logical_end_to_end_p50_ns"]
    cpu_p95_ns = cpu_summary["logical_end_to_end_p95_ns"]
    npu_p95_ns = npu_summary["logical_end_to_end_p95_ns"]
    p50_speedup = cpu_p50_ns / npu_p50_ns
    pooled_speedup = cpu_summary["logical_end_to_end_total_ns"] / npu_summary[
        "logical_end_to_end_total_ns"
    ]
    p95_ratio = npu_p95_ns / cpu_p95_ns
    p50_speed_gate = 115 * npu_p50_ns <= 100 * cpu_p50_ns
    p95_no_worse_gate = npu_p95_ns <= cpu_p95_ns
    expected_full_calls = candidate_count // BATCH
    expected_tail_calls = int(tail_rows > 0)
    cpu_full_calls_observed = sum(
        call.call_kind == "CPU_FIXED_B16" for trace in cpu_traces for call in trace.calls
    )
    cpu_control_tail_calls_observed = sum(
        call.call_kind == "CPU_TAIL" for trace in cpu_traces for call in trace.calls
    )
    npu_full_calls_observed = sum(
        call.call_kind == "NPU_FIXED_B16" for trace in npu_traces for call in trace.calls
    )
    npu_path_tail_calls_observed = sum(
        call.call_kind == "CPU_TAIL" for trace in npu_traces for call in trace.calls
    )
    cpu_trace_gate = _trace_receipts_exact(
        cpu_traces,
        path="CPU_CONTROL",
        candidate_count=candidate_count,
        repeats=repeats,
        full_kind="CPU_FIXED_B16",
        full_device="CPU",
    )
    npu_trace_gate = _trace_receipts_exact(
        npu_traces,
        path="NPU_PLUS_CPU_TAIL",
        candidate_count=candidate_count,
        repeats=repeats,
        full_kind="NPU_FIXED_B16",
        full_device="NPU",
    )
    receipt_gate = (
        cpu_full_calls_observed == repeats * expected_full_calls
        and cpu_control_tail_calls_observed == repeats * expected_tail_calls
        and npu_full_calls_observed == repeats * expected_full_calls
        and npu_path_tail_calls_observed == repeats * expected_tail_calls
        and cpu_trace_gate
        and npu_trace_gate
    )
    gates = {
        "same_fixed_ir_cpu_and_npu": True,
        "cpu_execution_device_exact": cpu_fixed_receipt.execution_devices == ("CPU",),
        "npu_execution_device_exact": npu_fixed_receipt.execution_devices == ("NPU",),
        "at_least_one_physical_npu_batch": expected_full_calls > 0,
        "physical_call_receipts_exact": receipt_gate,
        "cpu_max_abs_le_1e_5": cpu_error <= 1.0e-5,
        "npu_full_max_abs_le_1e_2": npu_full_error <= 1.0e-2,
        "cpu_tail_max_abs_le_1e_5": cpu_tail_error <= 1.0e-5,
        "npu_path_cpu_tail_max_abs_le_1e_5": npu_path_tail_error <= 1.0e-5,
        "cpu_routes_exact": cpu_route_matches == cpu_route_total and cpu_route_total > 0,
        "npu_routes_exact": npu_route_matches == npu_route_total and npu_route_total > 0,
        "component_p50_speedup_ge_1_15": p50_speed_gate,
        "component_p95_no_worse": p95_no_worse_gate,
    }
    passed = all(gates.values())
    compile_receipts: dict[str, Any] = {
        "cpu_fixed": _compile_receipt_json(cpu_fixed_receipt),
        "npu_fixed": _compile_receipt_json(npu_fixed_receipt),
        "cpu_dynamic_tail": (
            None if cpu_dynamic_receipt is None else _compile_receipt_json(cpu_dynamic_receipt)
        ),
    }
    return {
        "schema": RESULT_SCHEMA,
        "implementation_status": "MEASURED_COMPONENT_PENDING_INDEPENDENT_REVIEW",
        "status": (
            "COMPONENT_PASS_PENDING_INTEGRATED_SPARSE_K_GATE"
            if passed
            else "FALLBACK_CPU_OR_INCONCLUSIVE"
        ),
        "adoption_authorized": False,
        "system_adoption_authorized": False,
        "integrated_sparse_executed": False,
        "component_action_metrics_computed": False,
        "component_route_ids_are_final_action_ranking": False,
        "downstream_gate_status": {
            "integrated_sparse_action_evaluator_executed": False,
            "candidate_action_top1_checked": False,
            "candidate_action_top2_swap_checked": False,
            "action_margin_error_checked": False,
            "cpu_sparse_overlap_latency_checked": False,
            "teacher_ranking_accuracy_checked": False,
            "development_game_score_checked": False,
        },
        "downstream_integrated_gates_required": [
            "same sparse checkpoint, routing, gather, and compute path",
            "candidate/action top-1 exact agreement",
            "candidate/action top-2 swap rate",
            "action-margin error",
            "CPU sparse p50/p95 slowdown under overlap",
            "teacher ranking accuracy",
            "development game score",
        ],
        "scalar_reference_timed": False,
        "scalar_reference_role": "numeric oracle supplied in witness only",
        "openvino_version": ov.__version__,
        "fixed_batch": BATCH,
        "candidate_count": candidate_count,
        "full_batch_rows": full_rows,
        "tail_rows": tail_rows,
        "sparse_k_label": sparse_k,
        "sparse_active_counts": list(active_counts),
        "snapshot_version": metadata["snapshot_version"],
        "master_version": metadata.get("master_version"),
        "weights_sha256": metadata["weights_sha256"],
        "snapshot_metadata_sha256": hashlib.sha256(metadata_raw).hexdigest(),
        "control_source_sha256": sha256_file(SOURCE_PATH),
        "reused_v1_benchmark_source_sha256": sha256_file(REUSED_V1_SOURCE_PATH),
        "fixed_ir_xml_sha256": sha256_file(fixed_ir),
        "fixed_ir_bin_sha256": sha256_file(fixed_ir.with_suffix(".bin")),
        "dynamic_cpu_ir_xml_sha256": sha256_file(dynamic_cpu_ir),
        "dynamic_cpu_ir_bin_sha256": sha256_file(dynamic_cpu_ir.with_suffix(".bin")),
        "witness_sha256": sha256_file(witness_path),
        "witness_provenance_sha256": hashlib.sha256(provenance_raw).hexdigest(),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "witness_generator_source_sha256": sha256_file(witness_generator_source),
        "witness_split": provenance["split"],
        "environment_seed_count": len(provenance["environment_seeds"]),
        "compile_receipts": compile_receipts,
        "persistent_io_setup_receipts": {
            "cpu_fixed": {
                "io_mode": cpu_full_call.io_mode,
                "input_output_prebind_ns": cpu_full_call.setup_bind_ns,
            },
            "npu_fixed": {
                "io_mode": npu_full_call.io_mode,
                "input_output_prebind_ns": npu_full_call.setup_bind_ns,
            },
            "cpu_dynamic_tail": (
                None
                if cpu_tail is None
                else {
                    "io_mode": cpu_tail.io_mode,
                    "rows": cpu_tail.rows,
                    "input_output_prebind_ns": cpu_tail.setup_bind_ns,
                }
            ),
        },
        "warmups_per_path": warmups,
        "timed_repeats_per_path": repeats,
        "first_call": {
            "cpu": first_cpu.to_receipt(),
            "npu_plus_tail": first_npu.to_receipt(),
            "cpu_maximum_absolute_error": first_call_cpu_error,
            "npu_full_maximum_absolute_error": first_call_npu_full_error,
        },
        "cpu_control": {
            "summary": cpu_summary,
            "output_little_endian_fp32_sha256": hashlib.sha256(
                np.asarray(cpu_output, dtype="<f4").tobytes(order="C")
            ).hexdigest(),
            "timed_samples": [trace.to_receipt() for trace in cpu_traces],
        },
        "npu_plus_cpu_tail": {
            "summary": npu_summary,
            "output_little_endian_fp32_sha256": hashlib.sha256(
                np.asarray(npu_output, dtype="<f4").tobytes(order="C")
            ).hexdigest(),
            "timed_samples": [trace.to_receipt() for trace in npu_traces],
        },
        "physical_call_formula": {
            "full_calls_per_path_per_repeat": expected_full_calls,
            "tail_calls_per_path_per_repeat": expected_tail_calls,
            "observed_cpu_fixed_full_calls": cpu_full_calls_observed,
            "observed_cpu_control_tail_calls": cpu_control_tail_calls_observed,
            "observed_npu_full_calls": npu_full_calls_observed,
            "observed_npu_path_cpu_tail_calls": npu_path_tail_calls_observed,
            "cpu_trace_row_device_stage_gate": cpu_trace_gate,
            "npu_trace_row_device_stage_gate": npu_trace_gate,
            "npu_logical_h2d_payload_bytes_per_full_call": BATCH * INPUTS * FLOAT_BYTES,
            "npu_logical_d2h_payload_bytes_per_full_call": BATCH * OUTPUTS * FLOAT_BYTES,
            "hardware_dma_bytes_measured": False,
        },
        "numeric": {
            "cpu_maximum_absolute_error": cpu_error,
            "npu_full_maximum_absolute_error": npu_full_error,
            "cpu_tail_maximum_absolute_error": cpu_tail_error,
            "npu_path_cpu_tail_maximum_absolute_error": npu_path_tail_error,
            "cpu_npu_full_maximum_absolute_difference": cross_backend_full_error,
            "cpu_route_id_matches": cpu_route_matches,
            "cpu_route_id_total": cpu_route_total,
            "npu_route_id_matches": npu_route_matches,
            "npu_route_id_total": npu_route_total,
        },
        "performance": {
            "p50_end_to_end_speedup_cpu_over_npu": p50_speedup,
            "pooled_end_to_end_speedup_cpu_over_npu_supplemental": pooled_speedup,
            "npu_over_cpu_p95_ratio": p95_ratio,
            "component_speed_contract": {
                "p50_integer_inequality": "115 * npu_p50_ns <= 100 * cpu_p50_ns",
                "p95_inequality": "npu_p95_ns <= cpu_p95_ns",
                "pooled_speedup_is_gate": False,
            },
        },
        "gates": gates,
        "timing_semantics": {
            "clock": "time.perf_counter_ns (QueryPerformanceCounter on CPython/Windows)",
            "end_to_end": "first pack begin through final application-output copy end",
            "pack": "copy immutable witness view into persistent application pack buffer",
            "bind": "input/output Tensors prebound once; steady calls retain a zero bind marker",
            "h2d": (
                "NPU Tensor.copy_from into persistent runtime input; CPU shares the pack buffer; "
                "hardware DMA may remain in submit/wait"
            ),
            "submit": "InferRequest.start_async",
            "wait": "InferRequest.wait; includes device execution and any opaque runtime transfers",
            "d2h": "prebound shared host output is visible after wait; zero marker because no separate DMA API exists",
            "copy": "copy prebound shared host output into application-owned ordered output",
            "dma_timing_isolated": False,
        },
        "decision_scope": (
            "Non-adoptive HashStem component only. HashStem sparse-neuron route IDs are not "
            "candidate/action top-1 or top-2 rankings. An integrated evaluator using the same "
            "sparse checkpoint/routing/gather/compute path must pass action agreement, action "
            "margin, overlap latency, teacher accuracy, and development-game gates before "
            "system adoption."
        ),
    }
