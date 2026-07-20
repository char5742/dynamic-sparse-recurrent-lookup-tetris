"""Independent H0/H1 post-idle result validator.

Implementation status: UNEXECUTED_STATIC_ONLY.

This standard-library-only program does not import or execute the Julia runner,
provider, sparse model, or OpenVINO.  It reopens every byte named by a trusted
bundle manifest and derives the systems decision from raw records and separately
produced ETW/IMC evidence.  Runner summaries are cross-checks, never gate inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


IMPLEMENTATION_STATUS = "UNEXECUTED_STATIC_ONLY"
BUNDLE_SCHEMA = "heterogeneous-265k-post-idle-validation-bundle-v1"
VALIDATOR_CONTRACT_SCHEMA = "heterogeneous-265k-post-idle-validator-contract-v1"
BENCHMARK_CONTRACT_SCHEMA = "heterogeneous-265k-post-idle-benchmark-contract-v1"
SOURCE_MANIFEST_SCHEMA = "heterogeneous-265k-post-idle-source-manifest-v1"
SUMMARY_SCHEMA = "heterogeneous-265k-post-idle-result-v1"
RECORD_SCHEMA = "heterogeneous-265k-post-idle-record-v1"
ETW_SCHEMA = "heterogeneous-265k-etw-evidence-v1"
IMC_SCHEMA = "heterogeneous-265k-imc-evidence-v1"
OUTPUT_SCHEMA = "heterogeneous-265k-post-idle-independent-decision-v1"
SNAPSHOT_SCHEMA = "learned-hashstem-v1-conv-dw-pw-pool-1039x214-relu"
MASTER_SCHEMA = "learned-hashstem-master-checkpoint-v1"

A0 = "A0_cpu_route_cpu_active"
H0 = "H0_cpu_hashstem_cpu_sparse"
H1 = "H1_npu_hashstem_cpu_sparse"
CELLS = (A0, H0, H1)
REPETITIONS = (1, 2, 3)
ACTIVE_COUNTS = (26, 22, 22)
OUTPUT_WIDTH = 22
HASHSTEM_INPUT_WIDTH = 559
HASHSTEM_OUTPUT_WIDTH = 256
NPU_BATCH = 16
FLOAT_BYTES = 4
SPARSE_STAGES = tuple(
    stage
    for layer in range(1, 4)
    for stage in (
        f"l{layer}_route",
        f"l{layer}_gather",
        f"l{layer}_active",
        f"l{layer}_scatter",
    )
) + ("head",)


class RejectEvidence(RuntimeError):
    pass


class InconclusiveEvidence(RuntimeError):
    pass


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RejectEvidence(message)


def _require_int(value: Any, label: str, minimum: int | None = None) -> int:
    _require(_is_int(value), f"{label} must be an integer")
    if minimum is not None:
        _require(value >= minimum, f"{label} must be at least {minimum}")
    return value


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise RejectEvidence(f"cannot read {label}: {path}: {error}") from error


def _json_bytes(data: bytes, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RejectEvidence(f"{label} is not strict UTF-8 JSON: {error}") from error
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _resolve_within(root: Path, base: Path, relative_path: Any, label: str) -> Path:
    _require(isinstance(relative_path, str) and relative_path, f"{label} path is missing")
    candidate = Path(relative_path)
    _require(not candidate.is_absolute(), f"{label} path must be relative")
    resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RejectEvidence(f"{label} escapes the bundle root") from error
    _require(not resolved.is_symlink(), f"{label} must not be a symbolic link")
    return resolved


@dataclass(frozen=True)
class LoadedArtifact:
    label: str
    path: Path
    sha256: str
    data: bytes


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.loaded: dict[str, LoadedArtifact] = {}

    def load(
        self,
        reference: Any,
        label: str,
        *,
        base: Path | None = None,
        inconclusive_if_missing: bool = False,
    ) -> LoadedArtifact:
        if not isinstance(reference, dict):
            if inconclusive_if_missing:
                raise InconclusiveEvidence(f"missing {label} artifact reference")
            raise RejectEvidence(f"missing {label} artifact reference")
        expected = reference.get("sha256")
        if not _valid_sha256(expected):
            raise RejectEvidence(f"{label} has invalid SHA-256")
        byte_length = _require_int(reference.get("byte_length"), f"{label}.byte_length", 0)
        path = _resolve_within(self.root, base or self.root, reference.get("relative_path"), label)
        if not path.is_file():
            if inconclusive_if_missing:
                raise InconclusiveEvidence(f"{label} artifact is unavailable: {path}")
            raise RejectEvidence(f"{label} artifact is unavailable: {path}")
        data = _read_bytes(path, label)
        _require(len(data) == byte_length, f"{label} byte length changed")
        observed = _sha256_bytes(data)
        _require(observed == expected, f"{label} SHA-256 mismatch")
        loaded = LoadedArtifact(label, path, observed, data)
        prior = self.loaded.get(label)
        _require(prior is None or prior.path == path, f"duplicate logical artifact {label}")
        self.loaded[label] = loaded
        return loaded


def _round_ratio_ties_even(numerator: int, denominator: int) -> int:
    _require(denominator > 0 and numerator >= 0, "invalid QPC conversion ratio")
    quotient, remainder = divmod(numerator, denominator)
    doubled = 2 * remainder
    if doubled > denominator or (doubled == denominator and quotient % 2 == 1):
        quotient += 1
    return quotient


def nearest_rank(values: Iterable[int], numerator: int, denominator: int) -> int:
    ordered = sorted(values)
    _require(bool(ordered), "nearest-rank input is empty")
    _require(0 < numerator <= denominator, "invalid nearest-rank fraction")
    rank = (numerator * len(ordered) + denominator - 1) // denominator
    return ordered[rank - 1]


def _canonical_action_hash(digests: list[str]) -> str:
    payload = bytearray(struct.pack("<I", len(digests)))
    for digest in digests:
        _require(_valid_sha256(digest), "action digest must be lowercase SHA-256")
        payload.extend(bytes.fromhex(digest))
    return _sha256_bytes(bytes(payload))


def _validate_routes(raw: Any, candidate_count: int) -> list[list[list[int]]]:
    _require(isinstance(raw, list) and len(raw) == candidate_count, "route candidate count mismatch")
    normalized: list[list[list[int]]] = []
    for candidate_index, candidate in enumerate(raw, start=1):
        _require(isinstance(candidate, list) and len(candidate) == 3, f"candidate {candidate_index} routes malformed")
        layers: list[list[int]] = []
        for layer_index, (ids, expected_count) in enumerate(zip(candidate, ACTIVE_COUNTS), start=1):
            _require(isinstance(ids, list) and len(ids) == expected_count, f"candidate {candidate_index} layer {layer_index} active width mismatch")
            converted = [_require_int(value, "route ID", 1) for value in ids]
            _require(len(set(converted)) == len(converted), "route IDs contain duplicates")
            _require(all(value <= 2_147_483_647 for value in converted), "route ID exceeds Int32")
            layers.append(converted)
        normalized.append(layers)
    return normalized


def _canonical_route_hash(routes: list[list[list[int]]]) -> str:
    payload = bytearray(struct.pack("<I", len(routes)))
    for candidate in routes:
        for ids in candidate:
            payload.extend(struct.pack("<I", len(ids)))
            for neuron_id in ids:
                payload.extend(struct.pack("<i", neuron_id))
    return _sha256_bytes(bytes(payload))


def _decode_outputs(bits: Any, candidate_count: int) -> list[list[float]]:
    _require(isinstance(bits, list), "output bits must be a list")
    _require(len(bits) == candidate_count * OUTPUT_WIDTH, "output bit count mismatch")
    candidates: list[list[float]] = []
    position = 0
    for _ in range(candidate_count):
        row: list[float] = []
        for _ in range(OUTPUT_WIDTH):
            word = _require_int(bits[position], "output word", 0)
            _require(word <= 0xFFFFFFFF, "output word exceeds UInt32")
            value = struct.unpack("<f", struct.pack("<I", word))[0]
            _require(math.isfinite(value), "output contains NaN or infinity")
            row.append(value)
            position += 1
        candidates.append(row)
    return candidates


def _ranked_actions(outputs: list[list[float]], action_digests: list[str], output_index: int) -> list[str]:
    _require(1 <= output_index <= OUTPUT_WIDTH, "ranking output index is outside 1:22")
    ordered = sorted(
        range(len(outputs)),
        key=lambda index: (-outputs[index][output_index - 1], index),
    )
    return [action_digests[index] for index in ordered]


def _required_stage_counts(cell: str, candidate_count: int) -> dict[str, int]:
    counts = {"runner_total": 1, "pack": 1, **{stage: candidate_count for stage in SPARSE_STAGES}}
    if cell == A0:
        counts["hash_reference_copy"] = 1
    elif cell == H0:
        counts.update({"hash_pack": 1, "hash_cpu_compute": 1, "hash_copy": 1})
    else:
        npu_calls = candidate_count // NPU_BATCH
        tail_calls = int(candidate_count % NPU_BATCH != 0)
        counts.update({"hash_pack": 1, "hash_copy": 1})
        if npu_calls:
            counts.update(
                {
                    "hash_h2d": npu_calls,
                    "hash_bind": 2 * npu_calls,
                    "hash_submit": npu_calls,
                    "hash_wait": npu_calls,
                    "hash_d2h": npu_calls,
                }
            )
        if tail_calls:
            counts["hash_cpu_tail"] = 1
    return counts


def _validate_validator_contract(contract: dict[str, Any]) -> None:
    _require(contract.get("schema") == VALIDATOR_CONTRACT_SCHEMA, "validator contract schema mismatch")
    _require(contract.get("status") == IMPLEMENTATION_STATUS, "validator contract status mismatch")
    scope = contract.get("scope")
    _require(isinstance(scope, dict) and scope.get("promotion_scope") == "SYSTEMS_GATE_ONLY", "validator contract scope changed")
    bundle = contract.get("bundle")
    _require(isinstance(bundle, dict), "validator bundle contract is missing")
    _require(bundle.get("schema") == BUNDLE_SCHEMA, "validator bundle schema contract changed")
    _require(bundle.get("artifact_root") == ".", "validator artifact-root contract changed")
    _require(
        bundle.get("required_cells") == {cell: [1, 2, 3] for cell in CELLS},
        "validator cell/repetition contract changed",
    )
    receipts = contract.get("runner_receipts")
    _require(isinstance(receipts, dict), "validator runner-receipt contract is missing")
    _require(receipts.get("records_per_repetition") == 4096, "validator record-count contract changed")
    _require(receipts.get("sequences") == [257, 4352], "validator sequence contract changed")
    _require(receipts.get("warmup_count") == 256, "validator warmup contract changed")
    physical = contract.get("physical_h1_formulas")
    _require(isinstance(physical, dict), "validator physical-call contract is missing")
    _require(physical.get("h2d_bytes_per_npu_call") == 35_776, "validator H2D formula changed")
    _require(physical.get("d2h_bytes_per_npu_call") == 16_384, "validator D2H formula changed")
    _require(physical.get("physical_waits_per_npu_call") == 1, "validator wait formula changed")
    _require(physical.get("hash_bind_qpc_occurrences_per_npu_call") == 2, "validator bind count changed")
    _require(physical.get("hash_submit_qpc_occurrences_per_npu_call") == 1, "validator submit count changed")
    _require(physical.get("hash_wait_qpc_occurrences_per_npu_call") == 1, "validator wait count changed")
    _require(physical.get("hash_d2h_qpc_occurrences_per_npu_call") == 1, "validator D2H count changed")
    numeric = contract.get("numeric_and_rank_gates")
    _require(isinstance(numeric, dict), "validator numeric contract is missing")
    _require(numeric.get("h0_cpu_maximum_absolute_error") == 1.0e-5, "validator H0 numeric gate changed")
    _require(numeric.get("h1_npu_maximum_absolute_error") == 1.0e-2, "validator H1 numeric gate changed")
    _require(numeric.get("h1_actual_length_cpu_tail_maximum_absolute_error") == 1.0e-5, "validator tail gate changed")
    performance = contract.get("performance_gates")
    _require(isinstance(performance, dict), "validator performance contract is missing")
    _require(performance.get("h1_over_h0_speedup_minimum") == 1.15, "validator speed gate changed")
    _require(performance.get("h1_end_to_end_p95_ratio_maximum") == 1.0, "validator p95 gate changed")
    _require(performance.get("h1_over_h0_cpu_sparse_p50_slowdown_maximum") == 1.1, "validator sparse p50 gate changed")
    _require(performance.get("h1_over_h0_cpu_sparse_p95_slowdown_maximum") == 1.1, "validator sparse p95 gate changed")
    _require(set(contract.get("decisions", {})) == {"ADOPT", "REJECT", "INCONCLUSIVE_FAIL_CLOSED"}, "validator decision set changed")
    provenance_fields = {
        "provider_source_sha256",
        "benchmark_contract_sha256",
        "system_contract_sha256",
        "source_manifest_sha256",
        "evidence_contract_sha256",
        "captures",
    }
    for name, schema in (("etw_evidence", ETW_SCHEMA), ("imc_evidence", IMC_SCHEMA)):
        evidence = contract.get(name, {})
        _require(evidence.get("schema") == schema, f"validator {name} schema changed")
        _require(set(evidence.get("required_top_level", ())) == provenance_fields, f"validator {name} provenance contract changed")


def _validate_benchmark_contract(contract: dict[str, Any]) -> int:
    _require(contract.get("schema") == BENCHMARK_CONTRACT_SCHEMA, "benchmark contract schema mismatch")
    policy = contract.get("execution_policy")
    _require(isinstance(policy, dict), "benchmark execution policy missing")
    _require(policy.get("sequential_cells_only") is True, "benchmark lost sequential-cell policy")
    _require(policy.get("fresh_process_per_cell") is True, "benchmark lost fresh-process policy")
    _require(policy.get("independent_repetitions") == 3, "benchmark repetition count changed")
    _require(policy.get("warmup_candidate_sets") == 256, "benchmark warmup count changed")
    timed_count = _require_int(policy.get("timed_candidate_sets"), "benchmark timed count", 1)
    _require(timed_count == 4096, "benchmark timed count changed")
    _require(policy.get("tail_policy") == "fixed batches of 16 plus deterministic actual-length CPU tail; report both physical calls", "benchmark tail policy changed")
    matrix = contract.get("matrix")
    _require(isinstance(matrix, list), "benchmark matrix is missing")
    indexed = {entry.get("id"): entry for entry in matrix if isinstance(entry, dict)}
    _require(set(CELLS).issubset(indexed), "benchmark matrix lost an executable validator cell")
    _require(indexed[H0].get("required_backend") == "CPU", "H0 backend contract changed")
    _require(indexed[H1].get("comparison") == H0, "H1 comparison contract changed")
    _require("exactly NPU" in str(indexed[H1].get("required_backend")), "H1 exact-NPU contract changed")
    numeric = contract.get("numeric_gates")
    _require(isinstance(numeric, dict), "benchmark numeric gates are missing")
    _require(numeric.get("cpu_maximum_absolute_error") == 1.0e-5, "benchmark CPU numeric gate changed")
    _require(numeric.get("accelerator_maximum_absolute_error") == 1.0e-2, "benchmark accelerator numeric gate changed")
    _require(numeric.get("finite_outputs_required") is True, "benchmark finite-output gate changed")
    h1_gate = contract.get("performance_gates", {}).get("H1_vs_H0", {})
    _require(h1_gate.get("end_to_end_candidate_set_throughput_speedup_minimum") == 1.15, "benchmark H1 speed gate changed")
    _require(h1_gate.get("end_to_end_p95_ratio_maximum") == 1.0, "benchmark H1 p95 gate changed")
    _require(h1_gate.get("concurrent_cpu_sparse_slowdown_maximum") == 1.1, "benchmark sparse slowdown gate changed")
    return timed_count


@dataclass
class ValidatedRecord:
    raw: dict[str, Any]
    sequence: int
    set_id: str
    candidate_count: int
    action_digests: list[str]
    routes: list[list[list[int]]]
    outputs: list[list[float]]
    ranked_actions: list[str]
    top1: int
    stage_ns: dict[str, int]
    runner_interval: tuple[int, int]


def _validate_record(
    raw: dict[str, Any],
    *,
    cell: str,
    repetition: int,
    run_uuid: str,
    process_id: int,
    process_identity_qpc: int,
    qpc_frequency: int,
) -> ValidatedRecord:
    _require(raw.get("schema") == RECORD_SCHEMA, "record schema mismatch")
    _require(raw.get("cell_id") == cell, "record cell mismatch")
    _require(raw.get("repetition") == repetition, "record repetition mismatch")
    _require(raw.get("run_uuid") == run_uuid, "record run UUID mismatch")
    _require(raw.get("process_id") == process_id, "record process ID mismatch")
    _require(raw.get("process_identity_qpc") == process_identity_qpc, "record process identity mismatch")
    _require(raw.get("qpc_frequency") == qpc_frequency, "record QPC frequency mismatch")
    sequence = _require_int(raw.get("sequence"), "record sequence", 1)
    set_id = raw.get("set_id")
    _require(isinstance(set_id, str) and set_id, "record set ID is missing")
    candidate_count = _require_int(raw.get("candidate_count"), "candidate count", 1)
    actions = raw.get("action_digests")
    _require(isinstance(actions, list) and len(actions) == candidate_count, "raw action ordering is missing")
    action_digests = [str(value) for value in actions]
    _require(_canonical_action_hash(action_digests) == raw.get("action_order_sha256"), "action-order digest mismatch")
    routes = _validate_routes(raw.get("route_ids"), candidate_count)
    _require(_canonical_route_hash(routes) == raw.get("route_ids_sha256"), "route-ID digest mismatch")
    outputs = _decode_outputs(raw.get("output_bits_candidate_major"), candidate_count)
    ranking_index = _require_int(raw.get("ranking_output_index"), "ranking output index", 1)
    ranked_actions = _ranked_actions(outputs, action_digests, ranking_index)
    top1 = _require_int(raw.get("top1_index"), "top1 index", 1)
    _require(top1 <= candidate_count, "top1 index exceeds candidate count")
    _require(action_digests[top1 - 1] == ranked_actions[0], "stored top1 differs from full stable ordering")

    physical_calls = raw.get("physical_calls")
    _require(isinstance(physical_calls, dict), "physical call map is missing")
    h2d = _require_int(raw.get("host_to_device_bytes"), "H2D bytes", 0)
    d2h = _require_int(raw.get("device_to_host_bytes"), "D2H bytes", 0)
    synchronizations = _require_int(raw.get("synchronization_count"), "synchronization count", 0)
    packed_bytes = _require_int(raw.get("packed_bytes"), "packed bytes", 1)
    backends = raw.get("actual_backends")
    _require(isinstance(backends, list) and all(isinstance(value, str) for value in backends), "backend evidence is malformed")
    _require(not any("AUTO" in value.upper() for value in backends), "AUTO backend is forbidden")
    if cell == A0:
        _require(physical_calls == {"REFERENCE": 1}, "A0 physical-call receipt mismatch")
        _require(backends == ["REFERENCE", "CPU_SPARSE"], "A0 backend receipt mismatch")
        expected_packed = candidate_count * 496 * FLOAT_BYTES
        _require(h2d == d2h == synchronizations == 0, "A0 reported accelerator work")
    elif cell == H0:
        _require(physical_calls == {"CPU_REFERENCE": 1}, "H0 physical-call receipt mismatch")
        _require(backends == ["CPU_REFERENCE", "CPU_SPARSE"], "H0 backend receipt mismatch")
        expected_packed = candidate_count * (496 + HASHSTEM_INPUT_WIDTH) * FLOAT_BYTES
        _require(h2d == d2h == synchronizations == 0, "H0 reported accelerator work")
    else:
        npu_calls = candidate_count // NPU_BATCH
        tail_calls = int(candidate_count % NPU_BATCH != 0)
        _require(physical_calls == {"NPU": npu_calls, "CPU_TAIL": tail_calls}, "H1 NPU/tail call count mismatch")
        _require(h2d == npu_calls * NPU_BATCH * HASHSTEM_INPUT_WIDTH * FLOAT_BYTES, "H1 H2D byte receipt mismatch")
        _require(d2h == npu_calls * NPU_BATCH * HASHSTEM_OUTPUT_WIDTH * FLOAT_BYTES, "H1 D2H byte receipt mismatch")
        _require(synchronizations == npu_calls, "H1 physical wait count mismatch")
        expected_backends = ["CPU_TAIL", "CPU_SPARSE"] if npu_calls == 0 else (
            ["NPU", "CPU_SPARSE"] if tail_calls == 0 else ["NPU", "CPU_TAIL", "CPU_SPARSE"]
        )
        _require(backends == expected_backends, "H1 actual backend receipt mismatch")
        expected_packed = candidate_count * (496 + HASHSTEM_INPUT_WIDTH) * FLOAT_BYTES
    _require(packed_bytes == expected_packed, "packed-byte receipt mismatch")

    intervals = raw.get("raw_qpc_intervals")
    _require(isinstance(intervals, dict), "raw QPC intervals are missing")
    expected_counts = _required_stage_counts(cell, candidate_count)
    _require(set(intervals) == set(expected_counts), "raw QPC stage set differs from the frozen contract")
    for stage, expected_count in expected_counts.items():
        _require(isinstance(intervals.get(stage), list), f"required QPC stage {stage} is missing")
        _require(len(intervals[stage]) == expected_count, f"QPC occurrence count differs for {stage}")
    runner_raw = intervals.get("runner_total")
    _require(isinstance(runner_raw, list) and len(runner_raw) == 1, "runner_total must have one interval")
    runner_pair = runner_raw[0]
    _require(isinstance(runner_pair, list) and len(runner_pair) == 2, "runner_total interval malformed")
    runner = (
        _require_int(runner_pair[0], "runner_total begin", 1),
        _require_int(runner_pair[1], "runner_total end", 1),
    )
    _require(runner[0] <= runner[1], "runner_total QPC moved backwards")
    derived: dict[str, int] = {}
    for stage, occurrences in intervals.items():
        _require(isinstance(stage, str) and stage, "QPC stage name is invalid")
        _require(isinstance(occurrences, list) and occurrences, f"QPC stage {stage} is empty")
        ticks = 0
        for occurrence in occurrences:
            _require(isinstance(occurrence, list) and len(occurrence) == 2, f"QPC interval malformed for {stage}")
            started = _require_int(occurrence[0], f"{stage} begin", 1)
            finished = _require_int(occurrence[1], f"{stage} end", 1)
            _require(started <= finished, f"QPC moved backwards in {stage}")
            _require(runner[0] <= started <= finished <= runner[1], f"QPC stage {stage} lies outside runner_total")
            ticks += finished - started
        derived[stage] = _round_ratio_ties_even(ticks * 1_000_000_000, qpc_frequency)
    reported_stage_ns = raw.get("stage_ns")
    _require(isinstance(reported_stage_ns, dict), "record stage_ns is missing")
    _require(set(reported_stage_ns) == set(derived), "record stage_ns keys differ from raw QPC")
    for stage, value in derived.items():
        _require(reported_stage_ns.get(stage) == value, f"record stage_ns differs from raw QPC for {stage}")
    return ValidatedRecord(raw, sequence, set_id, candidate_count, action_digests, routes, outputs, ranked_actions, top1, derived, runner)


def _metrics(records: list[ValidatedRecord]) -> dict[str, Any]:
    stages = set().union(*(record.stage_ns for record in records))
    p50 = {
        stage: nearest_rank((record.stage_ns[stage] for record in records if stage in record.stage_ns), 1, 2)
        for stage in sorted(stages)
    }
    p95 = {
        stage: nearest_rank((record.stage_ns[stage] for record in records if stage in record.stage_ns), 95, 100)
        for stage in sorted(stages)
    }
    total_ns = sum(record.stage_ns["runner_total"] for record in records)
    candidates = sum(record.candidate_count for record in records)
    _require(total_ns > 0, "timed QPC total is zero")
    return {
        "timed_candidate_sets": len(records),
        "timed_candidates": candidates,
        "stage_p50_ns": p50,
        "stage_p95_ns": p95,
        "end_to_end_p50_ns": p50["runner_total"],
        "end_to_end_p95_ns": p95["runner_total"],
        "candidate_sets_per_second": len(records) * 1_000_000_000.0 / total_ns,
        "candidates_per_second": candidates * 1_000_000_000.0 / total_ns,
        "total_runner_ns": total_ns,
    }


@dataclass
class ValidatedRun:
    cell: str
    repetition: int
    summary_sha256: str
    records_sha256: str
    run_uuid: str
    process_id: int
    process_identity_qpc: int
    qpc_frequency: int
    records: list[ValidatedRecord]
    metrics: dict[str, Any]


def _compare_summary_metrics(summary: dict[str, Any], derived: dict[str, Any]) -> None:
    reported = summary.get("metrics")
    _require(isinstance(reported, dict), "runner summary metrics are missing")
    for name in (
        "timed_candidate_sets",
        "timed_candidates",
        "stage_p50_ns",
        "stage_p95_ns",
        "end_to_end_p50_ns",
        "end_to_end_p95_ns",
    ):
        _require(reported.get(name) == derived[name], f"runner summary {name} differs from raw recomputation")
    for name in ("candidate_sets_per_second", "candidates_per_second"):
        value = reported.get(name)
        _require(isinstance(value, (int, float)) and math.isfinite(value), f"runner summary {name} is invalid")
        _require(math.isclose(float(value), derived[name], rel_tol=1e-12, abs_tol=0.0), f"runner summary {name} differs from raw recomputation")


def _validate_record_series(
    records: list[ValidatedRecord],
    timed_count: int,
    process_identity_qpc: int,
    ranking_output_index: int,
) -> None:
    _require(len(records) == timed_count, "timed record count mismatch")
    expected_sequences = list(range(257, 257 + timed_count))
    _require([record.sequence for record in records] == expected_sequences, "timed sequences are duplicated, omitted, or reordered")
    _require(all(record.raw.get("ranking_output_index") == ranking_output_index for record in records), "record/provider ranking output index mismatch")
    _require(process_identity_qpc <= records[0].runner_interval[0], "process identity QPC occurs after timed work")
    _require(
        all(previous.runner_interval[1] <= current.runner_interval[0] for previous, current in zip(records, records[1:])),
        "timed candidate-set QPC intervals overlap or move backwards",
    )


def _validate_source_manifest(store: ArtifactStore, artifact: LoadedArtifact) -> dict[str, str]:
    manifest = _json_bytes(artifact.data, "source manifest")
    _require(manifest.get("schema") == SOURCE_MANIFEST_SCHEMA, "source manifest schema mismatch")
    _require(manifest.get("status") == IMPLEMENTATION_STATUS, "source manifest status mismatch")
    files = manifest.get("files")
    _require(isinstance(files, list) and files, "source manifest is empty")
    observed: dict[str, str] = {}
    for entry in files:
        _require(isinstance(entry, dict), "source manifest entry malformed")
        name = entry.get("name")
        _require(isinstance(name, str) and name not in observed, "source manifest name is invalid or duplicated")
        loaded = store.load(entry, f"source:{name}", base=artifact.path.parent)
        observed[name] = loaded.sha256
    required = {
        "benchmark_contract",
        "provider",
        "substrate",
        "runner",
        "validator",
        "validator_contract",
        "validator_test",
        "heterogeneous_module",
        "windows_cpu_sets",
        "windows_evidence_runtime_hook",
        "windows_residency_ram_evidence",
        "evidence_contract",
        "etw_evidence_validator",
        "imc_evidence_validator",
        "hashstem",
        "hashstem_openvino",
        "hashstem_master",
        "pipeline",
        "fixedshape_learner",
        "lifecycle",
        "sparse_hashstem_bridge",
        "sparse3_module",
        "sparse3_geometry",
        "sparse3_model",
        "sparse3_optimizer",
        "sparse3_runtime",
        "sparse3_checkpoint",
        "common_features",
        "common_wta_index",
        "project_toml",
        "manifest_toml",
    }
    _require(required.issubset(observed), "source manifest omits a required provider/validator source")
    return observed


def _validate_master_and_snapshot(store: ArtifactStore, artifacts: dict[str, LoadedArtifact]) -> dict[str, str]:
    master = _json_bytes(artifacts["master"].data, "master manifest")
    _require(master.get("schema") == MASTER_SCHEMA, "master manifest schema mismatch")
    expected_filenames = {
        "metadata": "master_metadata.json",
        "weights": "hashstem_master_weights.npz",
        "optimizer": "hashstem_optimizer.jld2",
    }
    master_digests: dict[str, str] = {}
    master_members: dict[str, LoadedArtifact] = {}
    for member in ("metadata", "weights", "optimizer"):
        filename = master.get(f"{member}_file")
        digest = master.get(f"{member}_sha256")
        _require(isinstance(filename, str) and _valid_sha256(digest), f"master {member} binding is missing")
        _require(filename == expected_filenames[member], f"master {member} filename changed")
        reference = {"relative_path": filename, "sha256": digest, "byte_length": (artifacts["master"].path.parent / filename).stat().st_size}
        master_members[member] = store.load(reference, f"master:{member}", base=artifacts["master"].path.parent)
        master_digests[member] = digest
    master_metadata = _json_bytes(master_members["metadata"].data, "master metadata")
    _require(master_metadata.get("schema") == MASTER_SCHEMA, "master metadata schema mismatch")
    _require(master_metadata.get("weights_file") == expected_filenames["weights"], "master metadata weights filename changed")
    _require(master_metadata.get("optimizer_file") == expected_filenames["optimizer"], "master metadata optimizer filename changed")
    _require(master_metadata.get("weights_sha256") == master_digests["weights"], "master metadata weights digest mismatch")
    _require(master_metadata.get("optimizer_sha256") == master_digests["optimizer"], "master metadata optimizer digest mismatch")
    snapshot = _json_bytes(artifacts["snapshot"].data, "snapshot metadata")
    _require(snapshot.get("schema") == SNAPSHOT_SCHEMA, "snapshot schema mismatch")
    _require(snapshot.get("fixed_batch") == 16, "snapshot fixed batch mismatch")
    _require(snapshot.get("input_features") == 559 and snapshot.get("output_features") == 256, "snapshot geometry mismatch")
    _require(_require_int(snapshot.get("snapshot_version"), "snapshot version", 1) >= 1, "snapshot version is invalid")
    weights_digest = snapshot.get("weights_sha256")
    _require(_valid_sha256(weights_digest), "snapshot weight digest missing")
    _require(weights_digest == master_digests["weights"], "snapshot/master weight digest mismatch")
    for suffix in ("xml", "bin"):
        digest = snapshot.get(f"{suffix}_sha256")
        _require(_valid_sha256(digest), f"snapshot {suffix} digest missing")
        filename = f"hashstem_b16.{suffix}"
        path = artifacts["snapshot"].path.parent / filename
        _require(path.is_file(), f"snapshot {suffix} artifact missing")
        reference = {"relative_path": filename, "sha256": digest, "byte_length": path.stat().st_size}
        store.load(reference, f"snapshot:{suffix}", base=artifacts["snapshot"].path.parent)
    source_receipts = {}
    for field in (
        "master_source_sha256",
        "hashstem_source_sha256",
        "backend_source_sha256",
        "project_sha256",
        "manifest_sha256",
    ):
        digest = master_metadata.get(field)
        _require(_valid_sha256(digest), f"master metadata {field} is missing")
        source_receipts[field] = digest
    return {
        "weights_sha256": weights_digest,
        "xml_sha256": str(snapshot.get("xml_sha256")),
        "bin_sha256": str(snapshot.get("bin_sha256")),
        **source_receipts,
    }


def _validate_run(
    store: ArtifactStore,
    entry: dict[str, Any],
    *,
    cell: str,
    repetition: int,
    artifact_hashes: dict[str, str],
    direct_source_hashes: dict[str, str],
    snapshot_receipts: dict[str, str],
    timed_count: int,
) -> ValidatedRun:
    summary_artifact = store.load(entry.get("summary"), f"{cell}:r{repetition}:summary")
    records_artifact = store.load(entry.get("records"), f"{cell}:r{repetition}:records")
    summary = _json_bytes(summary_artifact.data, f"{cell} repetition {repetition} summary")
    _require(summary.get("schema") == SUMMARY_SCHEMA, "runner summary schema mismatch")
    _require(summary.get("implementation_status") == IMPLEMENTATION_STATUS, "runner implementation status mismatch")
    _require(summary.get("status") == "MEASURED_PENDING_INDEPENDENT_VALIDATION", "runner result is not complete")
    _require(summary.get("decision") == "NO_ADOPTION_BEFORE_VALIDATOR_ETW_IMC_GATE", "runner attempted to make an adoption decision")
    _require(summary.get("matrix_cell_id") == cell and summary.get("repetition") == repetition, "summary identity mismatch")
    _require(summary.get("records_sha256") == records_artifact.sha256, "summary records digest mismatch")
    expected_records_path = (summary_artifact.path.parent / str(summary.get("records_file", ""))).resolve()
    _require(expected_records_path == records_artifact.path, "summary resolves a different records file")
    _require(summary.get("external_etw_evidence_bound") is False, "runner summary falsely claims ETW binding")
    _require(summary.get("external_imc_evidence_bound") is False, "runner summary falsely claims IMC binding")
    _require(summary.get("packing_transfer_sync_included") is True, "runner omitted physical-boundary declaration")
    run_uuid = summary.get("run_uuid")
    _require(isinstance(run_uuid, str) and len(run_uuid) == 36, "run UUID is missing")
    process_id = _require_int(summary.get("process_id"), "process ID", 1)
    process_identity_qpc = _require_int(summary.get("process_identity_qpc"), "process identity QPC", 1)
    _require(isinstance(summary.get("process_started_utc"), str) and summary["process_started_utc"], "process start receipt missing")
    qpc_frequency = _require_int(summary.get("qpc_frequency"), "summary QPC frequency", 1)
    _require(summary.get("warmup_candidate_sets") == 256, "warmup count mismatch")

    bindings = summary.get("bindings")
    _require(isinstance(bindings, dict), "summary artifact bindings are missing")
    for name, digest in artifact_hashes.items():
        binding = bindings.get(name)
        _require(isinstance(binding, dict) and binding.get("sha256") == digest, f"summary binding differs for {name}")
    sources = summary.get("source_hashes")
    _require(isinstance(sources, dict), "summary source closure missing")
    for name, digest in direct_source_hashes.items():
        _require(sources.get(name) == digest, f"summary source hash differs for {name}")
    metadata = summary.get("provider_metadata")
    _require(isinstance(metadata, dict), "provider metadata missing")
    _require(metadata.get("implementation_status") == IMPLEMENTATION_STATUS, "provider implementation status changed")
    _require(metadata.get("cell_id") == cell, "provider cell identity mismatch")
    for name in ("input", "sparse_checkpoint", "master", "snapshot", "system_contract"):
        _require(metadata.get(f"{name}_sha256") == artifact_hashes[name], f"provider metadata differs for {name}")
    _require(metadata.get("source_manifest_sha256") == artifact_hashes["source_manifest"], "provider source-manifest binding differs")
    _require(metadata.get("qpc_frequency") == qpc_frequency, "provider QPC frequency differs")
    _require(metadata.get("snapshot_weights_sha256") == snapshot_receipts["weights_sha256"], "provider snapshot-weight receipt differs")
    _require(metadata.get("master_weights_sha256") == snapshot_receipts["weights_sha256"], "provider master-weight receipt differs")
    _require(metadata.get("snapshot_xml_sha256") == snapshot_receipts["xml_sha256"], "provider snapshot XML receipt differs")
    _require(metadata.get("snapshot_bin_sha256") == snapshot_receipts["bin_sha256"], "provider snapshot BIN receipt differs")
    ranking_output_index = _require_int(metadata.get("ranking_output_index"), "provider ranking output index", 1)
    _require(ranking_output_index <= OUTPUT_WIDTH, "provider ranking output index exceeds output width")
    enumerated = metadata.get("enumerated_backends")
    _require(isinstance(enumerated, list) and all(isinstance(value, str) for value in enumerated), "provider backend inventory is malformed")
    _require(not any("AUTO" in value.upper() for value in enumerated), "provider inventory contains forbidden AUTO backend")
    execution_devices = metadata.get("NPU_execution_devices")
    if cell == H1:
        _require(execution_devices == ["NPU"], "H1 was not compiled exclusively for exact NPU")
        _require(any(value.upper() == "NPU" for value in enumerated), "H1 provider inventory omitted exact NPU")
    else:
        _require(execution_devices == [], "CPU/reference cell reports an NPU execution device")

    records: list[ValidatedRecord] = []
    for line_number, line in enumerate(records_artifact.data.splitlines(), start=1):
        _require(bool(line.strip()), f"blank records line {line_number}")
        records.append(
            _validate_record(
                _json_bytes(line, f"record line {line_number}"),
                cell=cell,
                repetition=repetition,
                run_uuid=run_uuid,
                process_id=process_id,
                process_identity_qpc=process_identity_qpc,
                qpc_frequency=qpc_frequency,
            )
        )
    _validate_record_series(records, timed_count, process_identity_qpc, ranking_output_index)
    metrics = _metrics(records)
    _compare_summary_metrics(summary, metrics)
    return ValidatedRun(cell, repetition, summary_artifact.sha256, records_artifact.sha256, run_uuid, process_id, process_identity_qpc, qpc_frequency, records, metrics)


def _maximum_error(left: list[list[float]], right: list[list[float]], start: int = 0, end: int | None = None) -> float:
    limit = len(left) if end is None else end
    return max((abs(left[candidate][output] - right[candidate][output]) for candidate in range(start, limit) for output in range(OUTPUT_WIDTH)), default=0.0)


def _numeric_and_rank_gates(runs: dict[str, dict[int, ValidatedRun]]) -> tuple[dict[str, Any], bool]:
    per_repetition: dict[str, Any] = {}
    passed = True
    for repetition in REPETITIONS:
        triplets = zip(runs[A0][repetition].records, runs[H0][repetition].records, runs[H1][repetition].records)
        h0_error = 0.0
        h1_npu_error = 0.0
        h1_tail_error = 0.0
        exact_actions = exact_routes = exact_rank = True
        for reference, cpu, npu in triplets:
            aligned = (
                reference.sequence == cpu.sequence == npu.sequence
                and reference.set_id == cpu.set_id == npu.set_id
                and reference.candidate_count == cpu.candidate_count == npu.candidate_count
                and reference.action_digests == cpu.action_digests == npu.action_digests
            )
            _require(aligned, "A0/H0/H1 candidate sets are not exactly aligned")
            exact_actions &= reference.ranked_actions == cpu.ranked_actions == npu.ranked_actions
            exact_routes &= reference.routes == cpu.routes == npu.routes
            exact_rank &= reference.top1 == cpu.top1 == npu.top1
            h0_error = max(h0_error, _maximum_error(reference.outputs, cpu.outputs))
            npu_candidates = (npu.candidate_count // NPU_BATCH) * NPU_BATCH
            h1_npu_error = max(h1_npu_error, _maximum_error(reference.outputs, npu.outputs, 0, npu_candidates))
            h1_tail_error = max(h1_tail_error, _maximum_error(reference.outputs, npu.outputs, npu_candidates, npu.candidate_count))
        gates = {
            "full_action_order_exact": exact_actions,
            "production_route_ids_exact": exact_routes,
            "top1_exact": exact_rank,
            "h0_cpu_max_abs": h0_error,
            "h1_npu_max_abs": h1_npu_error,
            "h1_cpu_tail_max_abs": h1_tail_error,
            "h0_cpu_max_abs_le_1e_5": h0_error <= 1.0e-5,
            "h1_npu_max_abs_le_1e_2": h1_npu_error <= 1.0e-2,
            "h1_cpu_tail_max_abs_le_1e_5": h1_tail_error <= 1.0e-5,
        }
        passed &= all(value for name, value in gates.items() if name.endswith("exact") or name.endswith("1e_5") or name.endswith("1e_2"))
        per_repetition[str(repetition)] = gates
    return per_repetition, passed


def _validate_cross_repetition_input_closure(runs: dict[str, dict[int, ValidatedRun]]) -> None:
    baseline = runs[A0][1].records
    for cell in CELLS:
        for repetition in REPETITIONS:
            observed = runs[cell][repetition].records
            _require(len(observed) == len(baseline), "cross-repetition record count changed")
            for expected, actual in zip(baseline, observed):
                _require(
                    expected.sequence == actual.sequence
                    and expected.set_id == actual.set_id
                    and expected.candidate_count == actual.candidate_count
                    and expected.action_digests == actual.action_digests,
                    "input set/action ordering differs across cells or repetitions",
                )
                if cell == A0:
                    _require(
                        expected.raw.get("output_bits_candidate_major")
                        == actual.raw.get("output_bits_candidate_major"),
                        "A0 frozen witness outputs differ across repetitions",
                    )


def _validate_h1_npu_presence(runs: dict[str, dict[int, ValidatedRun]]) -> dict[str, int]:
    calls: dict[str, int] = {}
    for repetition in REPETITIONS:
        total = sum(
            int(record.raw["physical_calls"]["NPU"])
            for record in runs[H1][repetition].records
        )
        _require(total > 0, f"H1 repetition {repetition} contains no physical NPU call")
        calls[str(repetition)] = total
    return calls


def _sparse_distribution(run: ValidatedRun) -> tuple[int, int]:
    values = [sum(record.stage_ns[stage] for stage in SPARSE_STAGES) for record in run.records]
    return nearest_rank(values, 1, 2), nearest_rank(values, 95, 100)


def _performance_gates(runs: dict[str, dict[int, ValidatedRun]]) -> tuple[dict[str, Any], bool]:
    results: dict[str, Any] = {}
    passed = True
    pooled_h0 = pooled_h1 = 0
    for repetition in REPETITIONS:
        h0 = runs[H0][repetition]
        h1 = runs[H1][repetition]
        h0_total = h0.metrics["total_runner_ns"]
        h1_total = h1.metrics["total_runner_ns"]
        h0_sparse_p50, h0_sparse_p95 = _sparse_distribution(h0)
        h1_sparse_p50, h1_sparse_p95 = _sparse_distribution(h1)
        _require(h0.metrics["end_to_end_p95_ns"] > 0, "H0 p95 is zero")
        _require(h0_sparse_p50 > 0 and h0_sparse_p95 > 0, "H0 sparse latency is zero")
        speed = 100 * h0_total >= 115 * h1_total
        p95 = h1.metrics["end_to_end_p95_ns"] <= h0.metrics["end_to_end_p95_ns"]
        sparse50 = 100 * h1_sparse_p50 <= 110 * h0_sparse_p50
        sparse95 = 100 * h1_sparse_p95 <= 110 * h0_sparse_p95
        result = {
            "speedup": h0_total / h1_total,
            "speedup_ge_1_15": speed,
            "p95_ratio": h1.metrics["end_to_end_p95_ns"] / h0.metrics["end_to_end_p95_ns"],
            "p95_no_worse": p95,
            "sparse_p50_slowdown": h1_sparse_p50 / h0_sparse_p50,
            "sparse_p95_slowdown": h1_sparse_p95 / h0_sparse_p95,
            "sparse_slowdown_le_1_10": sparse50 and sparse95,
        }
        passed &= speed and p95 and sparse50 and sparse95
        results[str(repetition)] = result
        pooled_h0 += h0_total
        pooled_h1 += h1_total
    pooled = 100 * pooled_h0 >= 115 * pooled_h1
    passed &= pooled
    results["pooled"] = {"speedup": pooled_h0 / pooled_h1, "speedup_ge_1_15": pooled}
    return results, passed


def _capture_index(evidence: dict[str, Any], schema: str, label: str) -> dict[tuple[str, int], dict[str, Any]]:
    _require(evidence.get("schema") == schema, f"{label} schema mismatch")
    status = evidence.get("status")
    if status != "MEASURED_COMPLETE":
        raise InconclusiveEvidence(f"{label} status is not MEASURED_COMPLETE")
    captures = evidence.get("captures")
    if not isinstance(captures, list):
        raise InconclusiveEvidence(f"{label} captures are missing")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for capture in captures:
        _require(isinstance(capture, dict), f"{label} capture malformed")
        key = (capture.get("cell_id"), capture.get("repetition"))
        _require(key not in indexed, f"duplicate {label} capture {key}")
        indexed[key] = capture
    required = {(cell, repetition) for cell in (H0, H1) for repetition in REPETITIONS}
    if set(indexed) != required:
        raise InconclusiveEvidence(f"{label} does not contain exactly six H0/H1 captures")
    return indexed


def _validate_invocation_identities(runs: dict[str, dict[int, ValidatedRun]]) -> None:
    _require(set(runs) == set(CELLS), "validated run set is not exactly A0/H0/H1")
    _require(all(set(runs[cell]) == set(REPETITIONS) for cell in CELLS), "validated repetition set is not exactly 1,2,3")
    all_runs = [runs[cell][repetition] for cell in CELLS for repetition in REPETITIONS]
    _require(len({run.run_uuid for run in all_runs}) == 9, "the nine cell repetitions do not have distinct run UUIDs")
    _require(len({(run.process_id, run.process_identity_qpc) for run in all_runs}) == 9, "the nine cell repetitions are not distinct process invocations")


def _validate_etw(
    store: ArtifactStore,
    artifact: LoadedArtifact,
    provider_sha256: str,
    runs: dict[str, dict[int, ValidatedRun]],
    contract_sha256: str,
    system_sha256: str,
    source_manifest_sha256: str,
    evidence_contract_sha256: str,
) -> dict[str, Any]:
    evidence = _json_bytes(artifact.data, "ETW evidence")
    _require(evidence.get("provider_source_sha256") == provider_sha256, "ETW provider digest mismatch")
    _require(evidence.get("benchmark_contract_sha256") == contract_sha256, "ETW contract binding mismatch")
    _require(evidence.get("system_contract_sha256") == system_sha256, "ETW system binding mismatch")
    _require(evidence.get("source_manifest_sha256") == source_manifest_sha256, "ETW source-manifest binding mismatch")
    _require(evidence.get("evidence_contract_sha256") == evidence_contract_sha256, "ETW evidence-contract binding mismatch")
    captures = _capture_index(evidence, ETW_SCHEMA, "ETW")
    for key, capture in captures.items():
        run = runs[key[0]][key[1]]
        _require(capture.get("records_sha256") == run.records_sha256, "ETW records binding mismatch")
        _require(capture.get("run_uuid") == run.run_uuid, "ETW run UUID mismatch")
        _require(capture.get("process_id") == run.process_id, "ETW process identity mismatch")
        _require(capture.get("process_identity_qpc") == run.process_identity_qpc, "ETW QPC process identity mismatch")
        _require(capture.get("qpc_frequency") == run.qpc_frequency, "ETW QPC frequency mismatch")
        if capture.get("complete_timed_window_covered") is not True:
            raise InconclusiveEvidence("ETW does not cover the complete timed window")
        if _require_int(capture.get("lost_events"), "ETW lost events", 0) != 0 or _require_int(capture.get("lost_buffers"), "ETW lost buffers", 0) != 0:
            raise InconclusiveEvidence("ETW reports lost events or buffers")
        _require(capture.get("role_assignments_verified") is True, "ETW CPU-set role assignment failed")
        _require(_require_int(capture.get("sparse_stage_events"), "ETW sparse events", 1) > 0, "ETW sparse-stage evidence is empty")
        _require(_require_int(capture.get("hash_stage_events"), "ETW hash events", 1) > 0, "ETW HashStem evidence is empty")
        _require(_require_int(capture.get("outside_assigned_cpu_set_events"), "ETW outside-set events", 0) == 0, "ETW found work outside assigned CPU sets")
        trace = store.load(capture.get("raw_trace"), f"ETW:{key}:raw_trace", base=artifact.path.parent, inconclusive_if_missing=True)
        _require(trace.sha256 == capture["raw_trace"]["sha256"], "ETW raw trace binding mismatch")
        minimum = min(record.runner_interval[0] for record in run.records)
        maximum = max(record.runner_interval[1] for record in run.records)
        if _require_int(capture.get("begin_qpc"), "ETW begin QPC", 1) > minimum:
            raise InconclusiveEvidence("ETW starts after timed work")
        if _require_int(capture.get("end_qpc"), "ETW end QPC", 1) < maximum:
            raise InconclusiveEvidence("ETW ends before timed work")
    return {"captures": len(captures), "complete": True}


def _validate_imc(
    store: ArtifactStore,
    artifact: LoadedArtifact,
    provider_sha256: str,
    runs: dict[str, dict[int, ValidatedRun]],
    contract_sha256: str,
    system_sha256: str,
    source_manifest_sha256: str,
    evidence_contract_sha256: str,
) -> dict[str, Any]:
    evidence = _json_bytes(artifact.data, "IMC evidence")
    _require(evidence.get("provider_source_sha256") == provider_sha256, "IMC provider digest mismatch")
    _require(evidence.get("benchmark_contract_sha256") == contract_sha256, "IMC contract binding mismatch")
    _require(evidence.get("system_contract_sha256") == system_sha256, "IMC system binding mismatch")
    _require(evidence.get("source_manifest_sha256") == source_manifest_sha256, "IMC source-manifest binding mismatch")
    _require(evidence.get("evidence_contract_sha256") == evidence_contract_sha256, "IMC evidence-contract binding mismatch")
    captures = _capture_index(evidence, IMC_SCHEMA, "IMC")
    derived: dict[str, Any] = {}
    for key, capture in captures.items():
        run = runs[key[0]][key[1]]
        _require(capture.get("records_sha256") == run.records_sha256, "IMC records binding mismatch")
        _require(capture.get("run_uuid") == run.run_uuid, "IMC run UUID mismatch")
        _require(capture.get("qpc_frequency") == run.qpc_frequency, "IMC QPC frequency mismatch")
        if capture.get("multiplexed") is not False or capture.get("overflow") is not False:
            raise InconclusiveEvidence("IMC counter is multiplexed or overflowed")
        enabled = _require_int(capture.get("time_enabled_ticks"), "IMC enabled ticks", 1)
        running = _require_int(capture.get("time_running_ticks"), "IMC running ticks", 1)
        if enabled != running:
            raise InconclusiveEvidence("IMC time-running differs from time-enabled")
        source = capture.get("counter_source")
        _require(isinstance(source, str) and any(token in source.upper() for token in ("IMC", "UNCORE", "MEMORY_CONTROLLER")), "IMC source is not a memory-controller counter")
        _require(not any(token in source.upper() for token in ("PROCESS_IO", "THEORETICAL", "DIMM_RATING")), "invalid IMC substitute")
        width = _require_int(capture.get("counter_width_bits"), "IMC counter width", 1)
        _require(width <= 64, "IMC counter width exceeds 64")
        numerator = _require_int(capture.get("bytes_per_count_numerator"), "IMC scale numerator", 1)
        denominator = _require_int(capture.get("bytes_per_count_denominator"), "IMC scale denominator", 1)
        read_start = _require_int(capture.get("read_counter_start"), "IMC read start", 0)
        read_end = _require_int(capture.get("read_counter_end"), "IMC read end", 0)
        write_start = _require_int(capture.get("write_counter_start"), "IMC write start", 0)
        write_end = _require_int(capture.get("write_counter_end"), "IMC write end", 0)
        if read_end < read_start or write_end < write_start:
            raise InconclusiveEvidence("IMC counter wrapped without an auditable overflow proof")
        read_scaled = (read_end - read_start) * numerator
        write_scaled = (write_end - write_start) * numerator
        _require(read_scaled % denominator == 0 and write_scaled % denominator == 0, "IMC byte scaling is fractional")
        read_bytes = read_scaled // denominator
        write_bytes = write_scaled // denominator
        _require(read_bytes > 0 and write_bytes >= 0, "IMC byte deltas are empty or invalid")
        begin = _require_int(capture.get("begin_qpc"), "IMC begin QPC", 1)
        end = _require_int(capture.get("end_qpc"), "IMC end QPC", 1)
        _require(begin < end, "IMC interval is empty")
        minimum = min(record.runner_interval[0] for record in run.records)
        maximum = max(record.runner_interval[1] for record in run.records)
        if begin > minimum or end < maximum:
            raise InconclusiveEvidence("IMC does not cover the full timed window")
        elapsed_ns = _round_ratio_ties_even((end - begin) * 1_000_000_000, run.qpc_frequency)
        reported_read = _require_int(capture.get("reported_read_bytes"), "reported IMC read bytes", 0)
        reported_write = _require_int(capture.get("reported_write_bytes"), "reported IMC write bytes", 0)
        _require(reported_read == read_bytes and reported_write == write_bytes, "reported IMC bytes differ from raw counters")
        raw = store.load(capture.get("raw_counter_artifact"), f"IMC:{key}:raw", base=artifact.path.parent, inconclusive_if_missing=True)
        derived[f"{key[0]}:r{key[1]}"] = {
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
            "elapsed_ns": elapsed_ns,
            "total_gbps": (read_bytes + write_bytes) / elapsed_ns,
            "raw_sha256": raw.sha256,
        }
    return derived


def validate_bundle(bundle_artifact: LoadedArtifact, validator_contract_artifact: LoadedArtifact) -> dict[str, Any]:
    bundle = _json_bytes(bundle_artifact.data, "validation bundle")
    validator_contract = _json_bytes(validator_contract_artifact.data, "validator contract")
    _require(bundle.get("schema") == BUNDLE_SCHEMA, "validation bundle schema mismatch")
    _require(bundle.get("status") == "READY_FOR_INDEPENDENT_VALIDATION", "validation bundle is not frozen")
    _require(bundle.get("promotion_scope") == "SYSTEMS_GATE_ONLY", "validation bundle scope changed")
    _validate_validator_contract(validator_contract)
    root_text = bundle.get("artifact_root")
    _require(root_text == ".", "bundle artifact_root must be the manifest directory")
    store = ArtifactStore(bundle_artifact.path.parent)
    artifacts_raw = bundle.get("artifacts")
    _require(isinstance(artifacts_raw, dict), "validation bundle artifacts are missing")
    required_names = (
        "benchmark_contract",
        "source_manifest",
        "input",
        "sparse_checkpoint",
        "master",
        "snapshot",
        "system_contract",
        "provider_source",
        "runner_source",
        "substrate_source",
    )
    artifacts = {name: store.load(artifacts_raw.get(name), name) for name in required_names}
    benchmark_contract = _json_bytes(artifacts["benchmark_contract"].data, "benchmark contract")
    timed_count = _validate_benchmark_contract(benchmark_contract)
    source_hashes = _validate_source_manifest(store, artifacts["source_manifest"])
    _require(source_hashes.get("validator") == _sha256_bytes(Path(__file__).read_bytes()), "source manifest does not bind the executing validator")
    _require(source_hashes.get("validator_contract") == validator_contract_artifact.sha256, "source manifest does not bind the validator contract")
    _require(source_hashes.get("benchmark_contract") == artifacts["benchmark_contract"].sha256, "source manifest does not bind the benchmark contract")
    snapshot_receipts = _validate_master_and_snapshot(store, artifacts)
    for receipt, source_name in (
        ("master_source_sha256", "hashstem_master"),
        ("hashstem_source_sha256", "hashstem"),
        ("backend_source_sha256", "fixedshape_learner"),
        ("project_sha256", "project_toml"),
        ("manifest_sha256", "manifest_toml"),
    ):
        _require(snapshot_receipts[receipt] == source_hashes[source_name], f"master source closure differs for {source_name}")

    artifact_hashes = {
        "contract": artifacts["benchmark_contract"].sha256,
        "input": artifacts["input"].sha256,
        "sparse_checkpoint": artifacts["sparse_checkpoint"].sha256,
        "master": artifacts["master"].sha256,
        "snapshot": artifacts["snapshot"].sha256,
        "system_contract": artifacts["system_contract"].sha256,
        "provider_source": artifacts["provider_source"].sha256,
        "source_manifest": artifacts["source_manifest"].sha256,
    }
    direct_sources = {
        "contract": artifacts["benchmark_contract"].sha256,
        "provider": artifacts["provider_source"].sha256,
        "substrate": artifacts["substrate_source"].sha256,
        "runner": artifacts["runner_source"].sha256,
        "source_manifest": artifacts["source_manifest"].sha256,
    }
    for name, digest in (("provider", artifacts["provider_source"].sha256), ("substrate", artifacts["substrate_source"].sha256), ("runner", artifacts["runner_source"].sha256)):
        _require(source_hashes.get(name) == digest, f"source manifest does not bind {name}")

    cell_entries = bundle.get("cells")
    _require(isinstance(cell_entries, dict) and set(cell_entries) == set(CELLS), "bundle must contain only A0/H0/H1")
    runs: dict[str, dict[int, ValidatedRun]] = {cell: {} for cell in CELLS}
    for cell in CELLS:
        entries = cell_entries[cell]
        _require(isinstance(entries, list) and len(entries) == 3, f"{cell} must contain three repetitions")
        by_repetition = {entry.get("repetition"): entry for entry in entries if isinstance(entry, dict)}
        _require(set(by_repetition) == set(REPETITIONS), f"{cell} repetition set is not exactly 1,2,3")
        for repetition in REPETITIONS:
            runs[cell][repetition] = _validate_run(
                store,
                by_repetition[repetition],
                cell=cell,
                repetition=repetition,
                artifact_hashes=artifact_hashes,
                direct_source_hashes=direct_sources,
                snapshot_receipts=snapshot_receipts,
                timed_count=timed_count,
            )

    _validate_invocation_identities(runs)
    h1_npu_calls_by_repetition = _validate_h1_npu_presence(runs)
    for repetition in REPETITIONS:
        a0 = runs[A0][repetition]
        for cell in (H0, H1):
            summary_artifact = store.loaded[f"{cell}:r{repetition}:summary"]
            summary = _json_bytes(summary_artifact.data, "non-reference summary")
            closure = summary.get("reference_closure")
            _require(isinstance(closure, dict), "A0 reference closure missing")
            _require(closure.get("summary_sha256") == a0.summary_sha256, "A0 summary closure mismatch")
            _require(closure.get("records_sha256") == a0.records_sha256, "A0 records closure mismatch")
            _require(closure.get("run_uuid") == a0.run_uuid, "A0 run-UUID closure mismatch")
            _require(closure.get("process_id") == a0.process_id, "A0 process-ID closure mismatch")
            _require(closure.get("process_identity_qpc") == a0.process_identity_qpc, "A0 process-QPC closure mismatch")
            bindings = summary.get("bindings")
            _require(isinstance(bindings, dict), "non-reference summary bindings are missing")
            _require(bindings.get("reference_summary", {}).get("sha256") == a0.summary_sha256, "A0 reference-summary binding mismatch")
            _require(bindings.get("reference_records", {}).get("sha256") == a0.records_sha256, "A0 reference-records binding mismatch")

    _validate_cross_repetition_input_closure(runs)
    numeric, numeric_pass = _numeric_and_rank_gates(runs)
    performance, performance_pass = _performance_gates(runs)
    if numeric_pass and performance_pass:
        evidence_refs = bundle.get("external_evidence")
        if not isinstance(evidence_refs, dict):
            raise InconclusiveEvidence("external ETW/IMC evidence is missing")
        etw_provider = store.load(evidence_refs.get("etw_provider_source"), "ETW provider", inconclusive_if_missing=True)
        imc_provider = store.load(evidence_refs.get("imc_provider_source"), "IMC provider", inconclusive_if_missing=True)
        benchmark_producer_hashes = {
            artifacts["provider_source"].sha256,
            artifacts["runner_source"].sha256,
            artifacts["substrate_source"].sha256,
        }
        _require(etw_provider.sha256 not in benchmark_producer_hashes, "ETW evidence provider is not separately produced")
        _require(imc_provider.sha256 not in benchmark_producer_hashes, "IMC evidence provider is not separately produced")
        _require(etw_provider.sha256 == source_hashes["etw_evidence_validator"], "ETW evidence provider is outside the frozen source closure")
        _require(imc_provider.sha256 == source_hashes["imc_evidence_validator"], "IMC evidence provider is outside the frozen source closure")
        etw_artifact = store.load(evidence_refs.get("etw"), "ETW evidence", inconclusive_if_missing=True)
        imc_artifact = store.load(evidence_refs.get("imc"), "IMC evidence", inconclusive_if_missing=True)
        etw = _validate_etw(
            store,
            etw_artifact,
            etw_provider.sha256,
            runs,
            artifacts["benchmark_contract"].sha256,
            artifacts["system_contract"].sha256,
            artifacts["source_manifest"].sha256,
            source_hashes["evidence_contract"],
        )
        imc = _validate_imc(
            store,
            imc_artifact,
            imc_provider.sha256,
            runs,
            artifacts["benchmark_contract"].sha256,
            artifacts["system_contract"].sha256,
            artifacts["source_manifest"].sha256,
            source_hashes["evidence_contract"],
        )
        decision = "ADOPT"
    else:
        decision = "REJECT"
        etw = {"status": "NOT_EVALUATED_AFTER_DECISIVE_RAW_GATE_REJECTION"}
        imc = {"status": "NOT_EVALUATED_AFTER_DECISIVE_RAW_GATE_REJECTION"}
    return {
        "schema": OUTPUT_SCHEMA,
        "implementation_status": IMPLEMENTATION_STATUS,
        "decision": decision,
        "promotion_scope": "SYSTEMS_GATE_ONLY",
        "track": "265K heterogeneous; H0 three-layer ~20M CPU-only vs H1 NPU HashStem plus identical CPU sparse",
        "excluded_tracks": ["pure-CPU dynamic-sparse promotion", "NNUE/search-reuse", "A1/A2 active-block offload"],
        "runner_summary_decisions_used": False,
        "bindings": {
            "bundle_sha256": bundle_artifact.sha256,
            "validator_contract_sha256": validator_contract_artifact.sha256,
            **{name: artifact.sha256 for name, artifact in artifacts.items()},
        },
        "numeric_and_rank_gates": numeric,
        "h1_physical_npu_calls_by_repetition": h1_npu_calls_by_repetition,
        "performance_gates": performance,
        "etw_gate": etw,
        "imc_gate": imc,
        "all_required_gates_pass": decision == "ADOPT",
        "note": "ADOPT authorizes only the H1 systems gate; it is not teacher, game, NNUE, A1, or A2 evidence.",
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    target = path.resolve()
    if target.exists():
        raise RejectEvidence(f"output must be fresh: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise RejectEvidence(f"temporary output must be fresh: {temporary}")
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--bundle-manifest-sha256", required=True)
    parser.add_argument("--validator-contract", type=Path, required=True)
    parser.add_argument("--validator-contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    decision = "REJECT"
    reason: str | None = None
    try:
        _require(_valid_sha256(args.bundle_manifest_sha256), "bundle manifest expected SHA-256 is invalid")
        _require(_valid_sha256(args.validator_contract_sha256), "validator contract expected SHA-256 is invalid")
        bundle_data = _read_bytes(args.bundle_manifest.resolve(), "bundle manifest")
        contract_data = _read_bytes(args.validator_contract.resolve(), "validator contract")
        _require(_sha256_bytes(bundle_data) == args.bundle_manifest_sha256, "bundle manifest SHA-256 mismatch")
        _require(_sha256_bytes(contract_data) == args.validator_contract_sha256, "validator contract SHA-256 mismatch")
        result = validate_bundle(
            LoadedArtifact("bundle", args.bundle_manifest.resolve(), args.bundle_manifest_sha256, bundle_data),
            LoadedArtifact("validator contract", args.validator_contract.resolve(), args.validator_contract_sha256, contract_data),
        )
        decision = result["decision"]
        _require(decision in {"ADOPT", "REJECT", "INCONCLUSIVE_FAIL_CLOSED"}, "validator returned an unknown decision")
    except InconclusiveEvidence as error:
        decision = "INCONCLUSIVE_FAIL_CLOSED"
        reason = str(error)
        result = {
            "schema": OUTPUT_SCHEMA,
            "implementation_status": IMPLEMENTATION_STATUS,
            "decision": decision,
            "promotion_scope": "SYSTEMS_GATE_ONLY",
            "failure_reason": reason,
            "runner_summary_decisions_used": False,
        }
    except (RejectEvidence, OSError, ValueError, KeyError, TypeError) as error:
        decision = "REJECT"
        reason = str(error)
        result = {
            "schema": OUTPUT_SCHEMA,
            "implementation_status": IMPLEMENTATION_STATUS,
            "decision": decision,
            "promotion_scope": "SYSTEMS_GATE_ONLY",
            "failure_reason": reason,
            "runner_summary_decisions_used": False,
        }
    result["validator_source_sha256"] = _sha256_bytes(Path(__file__).read_bytes())
    _atomic_write_json(args.output, result)
    return 0 if decision == "ADOPT" else (3 if decision == "INCONCLUSIVE_FAIL_CLOSED" else 2)


if __name__ == "__main__":
    sys.exit(main())
