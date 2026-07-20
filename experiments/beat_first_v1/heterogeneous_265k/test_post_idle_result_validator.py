"""Source-only contracts for the independent post-idle validator.

Implementation status: UNEXECUTED_STATIC_ONLY.

These tests are intentionally committed without being executed in the frozen
post-idle task.  They use only synthetic receipts and never import Julia,
OpenVINO, the benchmark provider, or a measured result bundle.
"""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import post_idle_result_validator as validator


IMPLEMENTATION_STATUS = "UNEXECUTED_STATIC_ONLY"
RUN_UUID = "00000000-0000-4000-8000-000000000001"
PROCESS_ID = 1234
PROCESS_IDENTITY_QPC = 777
QPC_FREQUENCY = 1_000_000_000


def _digest(index: int) -> str:
    return hashlib.sha256(f"action:{index}".encode("ascii")).hexdigest()


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _record(cell: str, candidate_count: int = 17) -> dict[str, object]:
    actions = [_digest(index) for index in range(candidate_count)]
    routes = [
        [
            list(range(1, validator.ACTIVE_COUNTS[0] + 1)),
            list(range(1, validator.ACTIVE_COUNTS[1] + 1)),
            list(range(1, validator.ACTIVE_COUNTS[2] + 1)),
        ]
        for _ in range(candidate_count)
    ]
    outputs: list[int] = []
    for candidate in range(candidate_count):
        row = [0.0] * validator.OUTPUT_WIDTH
        row[0] = float(candidate_count - candidate)
        outputs.extend(_float_bits(value) for value in row)

    counts = validator._required_stage_counts(cell, candidate_count)
    intervals: dict[str, list[list[int]]] = {}
    for stage, count in counts.items():
        if stage == "runner_total":
            intervals[stage] = [[1_000, 1_000_000]]
        else:
            intervals[stage] = [[1_001, 1_002] for _ in range(count)]
    stage_ns = {
        stage: sum(finished - started for started, finished in occurrences)
        for stage, occurrences in intervals.items()
    }

    if cell == validator.A0:
        physical_calls = {"REFERENCE": 1}
        backends = ["REFERENCE", "CPU_SPARSE"]
        packed_bytes = candidate_count * 496 * validator.FLOAT_BYTES
        npu_calls = 0
        tail_calls = 0
    elif cell == validator.H0:
        physical_calls = {"CPU_REFERENCE": 1}
        backends = ["CPU_REFERENCE", "CPU_SPARSE"]
        packed_bytes = candidate_count * (
            496 + validator.HASHSTEM_INPUT_WIDTH
        ) * validator.FLOAT_BYTES
        npu_calls = 0
        tail_calls = 0
    else:
        npu_calls = candidate_count // validator.NPU_BATCH
        tail_calls = int(candidate_count % validator.NPU_BATCH != 0)
        physical_calls = {"NPU": npu_calls, "CPU_TAIL": tail_calls}
        if npu_calls == 0:
            backends = ["CPU_TAIL", "CPU_SPARSE"]
        elif tail_calls == 0:
            backends = ["NPU", "CPU_SPARSE"]
        else:
            backends = ["NPU", "CPU_TAIL", "CPU_SPARSE"]
        packed_bytes = candidate_count * (
            496 + validator.HASHSTEM_INPUT_WIDTH
        ) * validator.FLOAT_BYTES

    return {
        "schema": validator.RECORD_SCHEMA,
        "cell_id": cell,
        "repetition": 1,
        "run_uuid": RUN_UUID,
        "process_id": PROCESS_ID,
        "process_identity_qpc": PROCESS_IDENTITY_QPC,
        "qpc_frequency": QPC_FREQUENCY,
        "sequence": 257,
        "set_id": "synthetic-set-257",
        "candidate_count": candidate_count,
        "action_digests": actions,
        "route_ids": routes,
        "action_order_sha256": validator._canonical_action_hash(actions),
        "route_ids_sha256": validator._canonical_route_hash(routes),
        "top1_index": 1,
        "ranking_output_index": 1,
        "output_bits_candidate_major": outputs,
        "actual_backends": backends,
        "physical_calls": physical_calls,
        "packed_bytes": packed_bytes,
        "host_to_device_bytes": (
            npu_calls
            * validator.NPU_BATCH
            * validator.HASHSTEM_INPUT_WIDTH
            * validator.FLOAT_BYTES
        ),
        "device_to_host_bytes": (
            npu_calls
            * validator.NPU_BATCH
            * validator.HASHSTEM_OUTPUT_WIDTH
            * validator.FLOAT_BYTES
        ),
        "synchronization_count": npu_calls,
        "raw_qpc_intervals": intervals,
        "stage_ns": stage_ns,
    }


def _validate(record: dict[str, object], cell: str) -> validator.ValidatedRecord:
    return validator._validate_record(
        record,
        cell=cell,
        repetition=1,
        run_uuid=RUN_UUID,
        process_id=PROCESS_ID,
        process_identity_qpc=PROCESS_IDENTITY_QPC,
        qpc_frequency=QPC_FREQUENCY,
    )


def _performance_run(
    cell: str,
    runner_ns: int,
    sparse_ns: int,
) -> validator.ValidatedRun:
    stage_ns = {stage: 0 for stage in validator.SPARSE_STAGES}
    stage_ns[validator.SPARSE_STAGES[0]] = sparse_ns
    stage_ns["runner_total"] = runner_ns
    record = validator.ValidatedRecord(
        {},
        257,
        "set",
        1,
        [_digest(0)],
        [[list(range(1, 27)), list(range(1, 23)), list(range(1, 23))]],
        [[0.0] * validator.OUTPUT_WIDTH],
        [_digest(0)],
        1,
        stage_ns,
        (1, 2),
    )
    metrics = {
        "total_runner_ns": runner_ns,
        "end_to_end_p95_ns": runner_ns,
    }
    return validator.ValidatedRun(
        cell,
        1,
        "a" * 64,
        "b" * 64,
        RUN_UUID,
        PROCESS_ID,
        PROCESS_IDENTITY_QPC,
        QPC_FREQUENCY,
        [record],
        metrics,
    )


def _numeric_runs(
    *,
    h0_record: validator.ValidatedRecord | None = None,
    h1_record: validator.ValidatedRecord | None = None,
) -> dict[str, dict[int, validator.ValidatedRun]]:
    validated = {
        validator.A0: _validate(_record(validator.A0), validator.A0),
        validator.H0: h0_record or _validate(_record(validator.H0), validator.H0),
        validator.H1: h1_record or _validate(_record(validator.H1), validator.H1),
    }
    runs: dict[str, dict[int, validator.ValidatedRun]] = {
        cell: {} for cell in validator.CELLS
    }
    for cell in validator.CELLS:
        for repetition in validator.REPETITIONS:
            runs[cell][repetition] = validator.ValidatedRun(
                cell,
                repetition,
                "a" * 64,
                "b" * 64,
                f"00000000-0000-4000-8000-{repetition:012d}",
                1000 + repetition,
                2000 + repetition,
                QPC_FREQUENCY,
                [copy.deepcopy(validated[cell])],
                {},
            )
    return runs


class StrictInputAndCliDecisionContracts(unittest.TestCase):
    def test_json_rejects_duplicate_keys_nonfinite_and_invalid_utf8(self) -> None:
        for payload in (
            b'{"a":1,"a":2}',
            b'{"a":NaN}',
            b'\xff',
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(validator.RejectEvidence):
                    validator._json_bytes(payload, "synthetic")

    def test_cli_decision_precedence_and_exit_codes(self) -> None:
        cases = (
            (
                {
                    "schema": validator.OUTPUT_SCHEMA,
                    "decision": "ADOPT",
                    "runner_summary_decisions_used": False,
                },
                None,
                "ADOPT",
                0,
            ),
            (None, validator.InconclusiveEvidence("missing ETW"), "INCONCLUSIVE_FAIL_CLOSED", 3),
            (None, validator.RejectEvidence("tampered records"), "REJECT", 2),
            ({"schema": validator.OUTPUT_SCHEMA, "decision": "MAYBE"}, None, "REJECT", 2),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle.json"
            contract = root / "contract.json"
            bundle.write_bytes(b"{}")
            contract.write_bytes(b"{}")
            bundle_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
            contract_sha = hashlib.sha256(contract.read_bytes()).hexdigest()
            for index, (result, exception, decision, exit_code) in enumerate(cases):
                output = root / f"decision-{index}.json"
                effect = exception if exception is not None else None
                with self.subTest(decision=decision):
                    with mock.patch.object(
                        validator,
                        "validate_bundle",
                        return_value=result,
                        side_effect=effect,
                    ):
                        observed = validator.main(
                            [
                                "--bundle-manifest",
                                str(bundle),
                                "--bundle-manifest-sha256",
                                bundle_sha,
                                "--validator-contract",
                                str(contract),
                                "--validator-contract-sha256",
                                contract_sha,
                                "--output",
                                str(output),
                            ]
                        )
                    payload = json.loads(output.read_text(encoding="utf-8"))
                    self.assertEqual(observed, exit_code)
                    self.assertEqual(payload["decision"], decision)
                    self.assertFalse(payload["runner_summary_decisions_used"])


class PercentileAndOrderingContracts(unittest.TestCase):
    def test_nearest_rank_4096_is_one_based_and_not_interpolated(self) -> None:
        values = range(1, 4097)
        self.assertEqual(validator.nearest_rank(values, 1, 2), 2048)
        self.assertEqual(validator.nearest_rank(values, 95, 100), 3892)

    def test_action_and_route_hashes_bind_order(self) -> None:
        record = _record(validator.H1)
        actions = list(record["action_digests"])
        actions[0], actions[1] = actions[1], actions[0]
        self.assertNotEqual(
            validator._canonical_action_hash(actions),
            record["action_order_sha256"],
        )
        routes = json.loads(json.dumps(record["route_ids"]))
        routes[0][0][0], routes[0][0][1] = routes[0][0][1], routes[0][0][0]
        self.assertNotEqual(
            validator._canonical_route_hash(routes),
            record["route_ids_sha256"],
        )

    def test_full_ranking_uses_ascending_candidate_tie_break(self) -> None:
        actions = [_digest(index) for index in range(3)]
        outputs = [[0.0] * validator.OUTPUT_WIDTH for _ in actions]
        self.assertEqual(validator._ranked_actions(outputs, actions, 1), actions)

    def test_exact_4096_record_series_and_nine_invocation_identities(self) -> None:
        records = [
            SimpleNamespace(
                sequence=257 + index,
                raw={"ranking_output_index": 1},
                runner_interval=(10_000 + 2 * index, 10_001 + 2 * index),
            )
            for index in range(4096)
        ]
        validator._validate_record_series(records, 4096, 9_000, 1)
        records[-1].sequence -= 1
        with self.assertRaisesRegex(validator.RejectEvidence, "duplicated"):
            validator._validate_record_series(records, 4096, 9_000, 1)

        runs: dict[str, dict[int, validator.ValidatedRun]] = {
            cell: {} for cell in validator.CELLS
        }
        identity = 0
        for cell in validator.CELLS:
            for repetition in validator.REPETITIONS:
                identity += 1
                run = _performance_run(cell, 115, 10)
                run.repetition = repetition
                run.run_uuid = f"00000000-0000-4000-8000-{identity:012d}"
                run.process_id = 10_000 + identity
                run.process_identity_qpc = 20_000 + identity
                runs[cell][repetition] = run
        validator._validate_invocation_identities(runs)
        runs[validator.H1][3].run_uuid = runs[validator.H0][1].run_uuid
        with self.assertRaisesRegex(validator.RejectEvidence, "distinct run UUID"):
            validator._validate_invocation_identities(runs)


class RawReceiptContracts(unittest.TestCase):
    def test_h1_physical_batch_tail_and_wait_formulas(self) -> None:
        for candidate_count in (15, 16, 17, 33):
            with self.subTest(candidate_count=candidate_count):
                validated = _validate(_record(validator.H1, candidate_count), validator.H1)
                self.assertEqual(validated.candidate_count, candidate_count)

    def test_each_physical_call_byte_wait_and_pack_receipt_rejects_tamper(self) -> None:
        mutations = (
            lambda record: record["physical_calls"].__setitem__("NPU", 2),
            lambda record: record.__setitem__(
                "host_to_device_bytes", record["host_to_device_bytes"] + 4
            ),
            lambda record: record.__setitem__(
                "device_to_host_bytes", record["device_to_host_bytes"] + 4
            ),
            lambda record: record.__setitem__(
                "synchronization_count", record["synchronization_count"] + 1
            ),
            lambda record: record.__setitem__(
                "packed_bytes", record["packed_bytes"] + 4
            ),
        )
        for mutate in mutations:
            record = _record(validator.H1, 17)
            mutate(record)
            with self.subTest(mutation=mutate):
                with self.assertRaises(validator.RejectEvidence):
                    _validate(record, validator.H1)

    def test_all_three_executable_cells_accept_complete_raw_receipts(self) -> None:
        for cell in validator.CELLS:
            with self.subTest(cell=cell):
                self.assertEqual(_validate(_record(cell), cell).top1, 1)

    def test_stage_ns_tamper_is_rejected(self) -> None:
        record = _record(validator.H1)
        record["stage_ns"]["hash_wait"] += 1
        with self.assertRaisesRegex(validator.RejectEvidence, "raw QPC"):
            _validate(record, validator.H1)

    def test_missing_wait_or_extra_uncontracted_stage_is_rejected(self) -> None:
        missing = _record(validator.H1)
        missing["raw_qpc_intervals"]["hash_wait"].clear()
        with self.assertRaises(validator.RejectEvidence):
            _validate(missing, validator.H1)

        extra = _record(validator.H1)
        extra["raw_qpc_intervals"]["hash_sync"] = [[1_003, 1_004]]
        extra["stage_ns"]["hash_sync"] = 1
        with self.assertRaises(validator.RejectEvidence):
            _validate(extra, validator.H1)

    def test_qpc_interval_outside_runner_total_is_rejected(self) -> None:
        record = _record(validator.H0)
        record["raw_qpc_intervals"]["pack"] = [[999, 1_000]]
        with self.assertRaisesRegex(validator.RejectEvidence, "outside runner_total"):
            _validate(record, validator.H0)

    def test_nonfinite_output_and_wrong_top1_are_rejected(self) -> None:
        nonfinite = _record(validator.H0)
        nonfinite["output_bits_candidate_major"][0] = 0x7FC00000
        with self.assertRaisesRegex(validator.RejectEvidence, "NaN or infinity"):
            _validate(nonfinite, validator.H0)

        wrong_top1 = _record(validator.H0)
        wrong_top1["top1_index"] = 2
        with self.assertRaisesRegex(validator.RejectEvidence, "stable ordering"):
            _validate(wrong_top1, validator.H0)

    def test_raw_float_bits_numeric_route_and_action_tamper_reach_gates(self) -> None:
        numeric = _record(validator.H1)
        numeric["output_bits_candidate_major"][0] = _float_bits(17.02)
        result, passed = validator._numeric_and_rank_gates(
            _numeric_runs(h1_record=_validate(numeric, validator.H1))
        )
        self.assertFalse(passed)
        self.assertFalse(result["1"]["h1_npu_max_abs_le_1e_2"])

        routed = _record(validator.H1)
        routed["route_ids"][0][0][0] = 27
        routed["route_ids_sha256"] = validator._canonical_route_hash(
            routed["route_ids"]
        )
        result, passed = validator._numeric_and_rank_gates(
            _numeric_runs(h1_record=_validate(routed, validator.H1))
        )
        self.assertFalse(passed)
        self.assertFalse(result["1"]["production_route_ids_exact"])

        reordered = _record(validator.H1)
        reordered["action_digests"][0], reordered["action_digests"][1] = (
            reordered["action_digests"][1],
            reordered["action_digests"][0],
        )
        reordered["action_order_sha256"] = validator._canonical_action_hash(
            reordered["action_digests"]
        )
        with self.assertRaisesRegex(validator.RejectEvidence, "aligned"):
            validator._numeric_and_rank_gates(
                _numeric_runs(h1_record=_validate(reordered, validator.H1))
            )

    def test_cross_repetition_input_closure_and_npu_presence_are_per_rep(self) -> None:
        runs = _numeric_runs()
        validator._validate_cross_repetition_input_closure(runs)
        self.assertEqual(
            set(validator._validate_h1_npu_presence(runs)),
            {"1", "2", "3"},
        )

        runs[validator.A0][2].records[0].set_id = "cherry-picked-set"
        with self.assertRaisesRegex(validator.RejectEvidence, "input set/action"):
            validator._validate_cross_repetition_input_closure(runs)

        runs = _numeric_runs()
        runs[validator.A0][2].records[0].raw["output_bits_candidate_major"][0] ^= 1
        with self.assertRaisesRegex(validator.RejectEvidence, "frozen witness outputs"):
            validator._validate_cross_repetition_input_closure(runs)

        runs = _numeric_runs()
        runs[validator.H1][2] = validator.ValidatedRun(
            validator.H1,
            2,
            "a" * 64,
            "b" * 64,
            RUN_UUID,
            PROCESS_ID,
            PROCESS_IDENTITY_QPC,
            QPC_FREQUENCY,
            [SimpleNamespace(raw={"physical_calls": {"NPU": 0}})],
            {},
        )
        with self.assertRaisesRegex(validator.RejectEvidence, "repetition 2"):
            validator._validate_h1_npu_presence(runs)


class GateAndFailClosedContracts(unittest.TestCase):
    def test_exact_performance_thresholds_are_inclusive(self) -> None:
        runs = {validator.H0: {}, validator.H1: {}}
        for repetition in validator.REPETITIONS:
            h0 = _performance_run(validator.H0, 115, 10)
            h1 = _performance_run(validator.H1, 100, 11)
            h0.repetition = repetition
            h1.repetition = repetition
            runs[validator.H0][repetition] = h0
            runs[validator.H1][repetition] = h1
        result, passed = validator._performance_gates(runs)
        self.assertTrue(passed)
        self.assertTrue(result["pooled"]["speedup_ge_1_15"])
        self.assertTrue(result["1"]["p95_no_worse"])
        self.assertTrue(result["1"]["sparse_slowdown_le_1_10"])

    def test_below_speed_threshold_is_rejected(self) -> None:
        runs = {validator.H0: {}, validator.H1: {}}
        for repetition in validator.REPETITIONS:
            runs[validator.H0][repetition] = _performance_run(validator.H0, 114, 10)
            runs[validator.H1][repetition] = _performance_run(validator.H1, 100, 10)
        _, passed = validator._performance_gates(runs)
        self.assertFalse(passed)

    def test_one_bad_repetition_cannot_hide_behind_pooled_or_other_repetitions(self) -> None:
        runs = {validator.H0: {}, validator.H1: {}}
        for repetition in validator.REPETITIONS:
            runs[validator.H0][repetition] = _performance_run(validator.H0, 230, 100)
            runs[validator.H1][repetition] = _performance_run(validator.H1, 100, 100)
        runs[validator.H1][2] = _performance_run(validator.H1, 231, 111)
        result, passed = validator._performance_gates(runs)
        self.assertFalse(passed)
        self.assertFalse(result["2"]["speedup_ge_1_15"])
        self.assertFalse(result["2"]["p95_no_worse"])
        self.assertFalse(result["2"]["sparse_slowdown_le_1_10"])

    def test_missing_etw_or_imc_capture_set_is_inconclusive(self) -> None:
        for schema, label in (
            (validator.ETW_SCHEMA, "ETW"),
            (validator.IMC_SCHEMA, "IMC"),
        ):
            evidence = {"schema": schema, "status": "MEASURED_COMPLETE", "captures": []}
            with self.subTest(label=label):
                with self.assertRaises(validator.InconclusiveEvidence):
                    validator._capture_index(evidence, schema, label)

    def test_etw_and_imc_provenance_fields_reject_independent_tamper(self) -> None:
        provider = "1" * 64
        benchmark = "2" * 64
        system = "3" * 64
        source_manifest = "4" * 64
        evidence_contract = "5" * 64
        expected = {
            "provider_source_sha256": provider,
            "benchmark_contract_sha256": benchmark,
            "system_contract_sha256": system,
            "source_manifest_sha256": source_manifest,
            "evidence_contract_sha256": evidence_contract,
            "status": "MEASURED_COMPLETE",
            "captures": [],
        }
        for schema, function in (
            (validator.ETW_SCHEMA, validator._validate_etw),
            (validator.IMC_SCHEMA, validator._validate_imc),
        ):
            for field in (
                "provider_source_sha256",
                "benchmark_contract_sha256",
                "system_contract_sha256",
                "source_manifest_sha256",
                "evidence_contract_sha256",
            ):
                document = {"schema": schema, **expected, field: "f" * 64}
                data = json.dumps(document).encode("utf-8")
                artifact = validator.LoadedArtifact(
                    schema,
                    Path(f"{schema}.json"),
                    hashlib.sha256(data).hexdigest(),
                    data,
                )
                with self.subTest(schema=schema, field=field):
                    with self.assertRaises(validator.RejectEvidence):
                        function(
                            None,
                            artifact,
                            provider,
                            {},
                            benchmark,
                            system,
                            source_manifest,
                            evidence_contract,
                        )

            document = {"schema": schema, **expected}
            data = json.dumps(document).encode("utf-8")
            artifact = validator.LoadedArtifact(
                schema,
                Path(f"{schema}.json"),
                hashlib.sha256(data).hexdigest(),
                data,
            )
            with self.assertRaises(validator.InconclusiveEvidence):
                function(
                    None,
                    artifact,
                    provider,
                    {},
                    benchmark,
                    system,
                    source_manifest,
                    evidence_contract,
                )

    def test_contract_freezes_scope_status_and_three_decisions(self) -> None:
        path = Path(__file__).with_name("post_idle_validator_contract.json")
        contract = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(contract["status"], IMPLEMENTATION_STATUS)
        self.assertEqual(contract["scope"]["promotion_scope"], "SYSTEMS_GATE_ONLY")
        self.assertEqual(
            set(contract["decisions"]),
            {"ADOPT", "REJECT", "INCONCLUSIVE_FAIL_CLOSED"},
        )
        self.assertEqual(
            contract["bundle"]["required_cells"],
            {cell: [1, 2, 3] for cell in validator.CELLS},
        )


if __name__ == "__main__":
    unittest.main()
