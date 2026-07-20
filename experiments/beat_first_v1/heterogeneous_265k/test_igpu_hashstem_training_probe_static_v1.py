"""Source-only contract checks for the unexecuted iGPU HashStem probe.

This test never imports the probe, PyTorch, NumPy, OpenVINO, or Julia.  Running
the test is intentionally deferred until the current single heavy Julia job is
authoritatively finished.
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROBE_PATH = ROOT / "igpu_hashstem_training_probe_v1.py"
CONTRACT_PATH = ROOT / "igpu_hashstem_training_probe_contract_v1.json"
PROBE = PROBE_PATH.read_text(encoding="utf-8")
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


class IGPUSparseTrainingProbeStaticTests(unittest.TestCase):
    def test_probe_is_valid_inert_python(self):
        tree = ast.parse(PROBE, filename=str(PROBE_PATH))
        top_level_calls = [
            node for node in tree.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        ]
        self.assertEqual(top_level_calls, [])
        self.assertIn('if __name__ == "__main__":', PROBE)
        self.assertNotIn("import torch\n", PROBE.split("def run_probe", 1)[0])
        self.assertNotIn("import numpy\n", PROBE.split("def run_probe", 1)[0])

    def test_real_backward_and_optimizer_change_are_mandatory(self):
        self.assertGreaterEqual(PROBE.count("loss.backward()"), 3)
        self.assertGreaterEqual(PROBE.count("bundle.optimizer.step()"), 2)
        for token in (
            "changed_parameter_elements",
            "maximum_parameter_change",
            "optimizer_state_entries",
            "optimizer_steps",
            "nonzero_first_moment",
            "nonzero_second_moment",
            '"real_backward_optimizer_weight_change"',
        ):
            self.assertIn(token, PROBE)
        numeric = CONTRACT["numeric_gates"]
        self.assertIs(numeric["real_backward_optimizer_weight_change_required"], True)
        self.assertIs(numeric["all_eight_parameter_tensors_receive_step_one_adamw_state"], True)
        self.assertIs(numeric["nonzero_first_and_second_moment_required"], True)

    def test_xpu_absence_and_compile_failure_fail_closed(self):
        self.assertIn('class BackendUnavailable', PROBE)
        self.assertIn('"UNAVAILABLE_FAIL_CLOSED"', PROBE)
        self.assertIn("raise SystemExit(2)", PROBE)
        self.assertIn("exactly one available XPU is required", PROBE)
        self.assertIn("torch.compile fixed-shape", PROBE)
        self.assertIn("except BackendUnavailable as exc:", PROBE)
        unavailable = CONTRACT["decisions"]
        self.assertEqual(unavailable["unavailable"], "UNAVAILABLE_FAIL_CLOSED")
        self.assertEqual(unavailable["unavailable_exit_code"], 2)

    def test_cpu_control_is_same_graph_weights_loss_shape_and_compile_policy(self):
        self.assertEqual(PROBE.count("compile_bundle("), 3)  # definition plus CPU and XPU
        self.assertIn('torch.compile(objective, backend="inductor", fullgraph=True, dynamic=False)', PROBE)
        self.assertIn('"PyTorch CPU vs bound Lux CPU master reference"', PROBE)
        self.assertIn('"compiled XPU vs compiled PyTorch CPU"', PROBE)
        self.assertIn("load_model_weights(torch, bundle.objective.model, weights)", PROBE)
        self.assertIn("output_cotangent", PROBE)
        self.assertIn("auxiliary_target", PROBE)
        self.assertIn("auxiliary_mask", PROBE)
        self.assertIn("BATCH = 16", PROBE)

    def test_transfer_sync_percentiles_and_external_contention_gate_remain_visible(self):
        for stage in (
            "h2d_submit", "h2d_sync", "forward_submit", "forward_sync",
            "backward_submit", "backward_sync", "optimizer_submit",
            "optimizer_sync", "d2h_submit", "d2h_sync", "total",
        ):
            self.assertIn(f'"{stage}"', PROBE)
        self.assertIn("torch.xpu.synchronize()", PROBE)
        self.assertIn("nearest_rank(values, 0.50)", PROBE)
        self.assertIn("nearest_rank(values, 0.95)", PROBE)
        self.assertIn("MemorySampler", PROBE)
        self.assertIn('"PENDING_EXTERNAL_INTEGRATED_GATE"', PROBE)
        self.assertEqual(
            CONTRACT["performance_gates"]["later_integrated_p_sparse_slowdown_maximum"],
            1.10,
        )

    def test_authoritative_speed_gate_is_total_p50_not_aggregate_throughput(self):
        self.assertIn("total_p50_speedup = cpu_total_p50_ns / xpu_total_p50_ns", PROBE)
        self.assertIn('"xpu_total_p50_speedup_ge_1_15": total_p50_speedup >= SPEEDUP_MIN', PROBE)
        self.assertIn('"aggregate_updates_per_second_speedup_vs_cpu_supplemental"', PROBE)
        self.assertNotIn('"xpu_speedup_ge_1_15"', PROBE)
        gates = CONTRACT["performance_gates"]
        self.assertEqual(gates["xpu_end_to_end_total_p50_speedup_vs_cpu_minimum"], 1.15)
        self.assertNotIn(
            "xpu_end_to_end_updates_per_second_speedup_vs_cpu_minimum", gates
        )

    def test_live_host_and_fair_cpu_control_are_contract_bound(self):
        for token in (
            "Win32_ComputerSystemProduct",
            'return "sha256:" + digest, normalized',
            "GetSystemCpuSetInformation",
            'cpu.get("cpu_set_topology_sha256")',
            'cpu.get("hashstem_control_threads")',
            'topology["fair_hashstem_control_threads"]',
            "CLI CPU thread count differs from contract-bound live P-core count",
            'parser.add_argument("--cpu-threads", type=int, required=True)',
            'exc.device_identity_receipt = receipt',
            'getattr(exc, "device_identity_receipt", None)',
            '"device_identity": device_identity',
        ):
            self.assertIn(token, PROBE)
        required = CONTRACT["backend"]["required_system_contract_fields"]
        self.assertIn("cpu.cpu_set_topology_sha256", required)
        self.assertIn("cpu.hashstem_control_threads", required)
        self.assertIn("host_id_derivation", CONTRACT["backend"])
        self.assertIn("cpu_control_policy", CONTRACT["backend"])

    def test_snapshot_matches_exact_exporter_contract(self):
        for token in (
            '"hashstem_schema": HASHSTEM_SCHEMA',
            '"status": SNAPSHOT_SOURCE_STATUS',
            '"immutable_source": True',
            '"openvino_compiled": False',
            '"publish_authorized": False',
            '"weights_file": SNAPSHOT_WEIGHTS_FILENAME',
            '"source_checkpoint_manifest_sha256": master["manifest_sha256"]',
            "type(observed) is not type(expected)",
        ):
            self.assertIn(token, PROBE)
        self.assertIn("exact hashstem_master.jl exporter contract", CONTRACT["hashstem"]["source_snapshot_requirement"])

    def test_checkpoint_is_probe_only_and_snapshot_cannot_publish(self):
        for member in CONTRACT["checkpoint_sync"]["required_members"]:
            self.assertIn(member, PROBE)
        self.assertIn('"state_roundtrip_bitwise": True', PROBE)
        self.assertIn('"master_snapshot_weights_byte_identical": True', PROBE)
        self.assertIn('"production_continuation_authorized": False', PROBE)
        self.assertIn('"publish_authorized": False', PROBE)
        self.assertIn('"adoption_authorized": False', PROBE)
        self.assertIn('left_array.tobytes(order="C")', PROBE)
        self.assertIn('right_array.tobytes(order="C")', PROBE)
        self.assertNotIn("torch.equal(left, right)", PROBE)

    def test_no_reserved_game_seed_is_embedded(self):
        self.assertIn("validation_seeds_used", PROBE)
        self.assertIn("sealed_seeds_used", PROBE)
        self.assertIn("development_seeds_used", PROBE)
        self.assertIn('metadata.get(field) is not False', PROBE)
        self.assertIn('teacher_split") != "train"', PROBE)
        self.assertIs(CONTRACT["witness"]["reserved_seed_free_required"], True)


if __name__ == "__main__":
    unittest.main()
