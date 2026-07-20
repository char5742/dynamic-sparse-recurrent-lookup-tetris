"""Standard-library-only static contract tests for active-block v2.

These tests parse source and JSON.  They do not import NumPy/OpenVINO, enumerate
devices, compile a graph, or execute the microbenchmark.
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "active_block_accel_microbench_v2.py"
CONTRACT_PATH = ROOT / "active_block_accel_microbench_v2_contract.json"


class ActiveBlockV2StaticContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(SOURCE_PATH))
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    def test_scope_is_component_only_and_excludes_model_claims(self) -> None:
        scope = self.contract["scope"]
        self.assertEqual(scope["evidence_class"], "COMPONENT_MICROBENCH_ONLY")
        self.assertEqual(scope["promotion_authority"], "NONE")
        self.assertFalse(scope["model_strength_evidence"])
        self.assertFalse(scope["game_score_evidence"])
        self.assertFalse(scope["nnue_or_parent_child_reuse"])
        self.assertFalse(scope["routing_search_timed"])
        self.assertIn("old model beaten", self.contract["prohibited_claims"])

    def test_exact_widths_and_transfer_bytes(self) -> None:
        variants = self.contract["fixed_geometry"]["variants"]
        expected = {
            "k64": ([24, 20, 20], 1_758_848, 4_096),
            "k128": ([48, 40, 40], 3_440_768, 8_192),
            "k256": ([96, 80, 80], 6_804_608, 16_384),
        }
        self.assertEqual(set(variants), set(expected))
        for name, (counts, h2d, d2h) in expected.items():
            self.assertEqual(variants[name]["active_counts"], counts)
            self.assertEqual(variants[name]["active_neurons"], sum(counts))
            self.assertEqual(variants[name]["host_to_device_bytes_per_three_call_sample"], h2d)
            self.assertEqual(variants[name]["device_to_host_bytes_per_three_call_sample"], d2h)

    def test_heavy_packages_are_lazy_imports(self) -> None:
        top_level_imports: set[str] = set()
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                top_level_imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level_imports.add(node.module.split(".")[0])
        self.assertNotIn("numpy", top_level_imports)
        self.assertNotIn("openvino", top_level_imports)
        execute = next(
            node for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "execute"
        )
        imported = {
            alias.name
            for node in ast.walk(execute)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_from = {
            node.module
            for node in ast.walk(execute)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertIn("numpy", imported)
        self.assertIn("openvino", imported)
        self.assertIn("openvino", imported_from)

    def test_source_contains_all_end_to_end_phases(self) -> None:
        required = (
            '"host_gather"',
            '"input_pack"',
            '"tensor_bind"',
            '"submit"',
            '"wait_and_synchronize"',
            '"output_copy"',
            "start_async",
            "get_output_tensor",
            "shared_memory=True",
            "QueryPerformanceCounter",
        )
        for marker in required:
            self.assertIn(marker, self.source)
        self.assertIn(
            "H2D/device/D2H visibility is opaque inside submit+wait",
            self.source,
        )

    def test_physical_receipts_bind_every_call(self) -> None:
        required = self.contract["physical_call_receipts"]["required_fields"]
        for field in required:
            self.assertIn(f'"{field}"', self.source)
        self.assertIn('"phase": phase', self.source)
        self.assertIn('"physical_call_index": physical_call_index', self.source)
        self.assertIn('"expected_physical_calls": expected_calls', self.source)
        self.assertIn('"physical_call_count_exact": observed_calls == expected_calls', self.source)

    def test_no_automatic_device_or_cpu_accelerator_fallback(self) -> None:
        self.assertIn('FORBIDDEN_DEVICE_TOKENS = ("AUTO", "HETERO", "MULTI", "BATCH")', self.source)
        compile_calls = [
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "compile_model"
        ]
        self.assertEqual(len(compile_calls), 1)
        call = compile_calls[0]
        self.assertGreaterEqual(len(call.args), 2)
        self.assertIsInstance(call.args[1], ast.Name)
        self.assertEqual(call.args[1].id, "device")
        self.assertIn('"UNAVAILABLE_FAIL_CLOSED"', self.source)
        self.assertNotIn('compile_model(model, "CPU")', self.source)
        self.assertNotIn('compile_model(model, "AUTO")', self.source)
        self.assertIn("actual[0].upper() != device.upper()", self.source)
        npu_selector = next(
            node for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "select_npu"
        )
        selector_source = ast.get_source_segment(self.source, npu_selector)
        self.assertIsNotNone(selector_source)
        self.assertNotIn('elif "NPU" in candidates', selector_source)
        self.assertIn("elif len(candidates) == 1", selector_source)

    def test_scalar_oracle_is_not_inside_timed_cpu_function(self) -> None:
        run_cpu = next(
            node for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_cpu_sample"
        )
        called_names = {
            node.func.id
            for node in ast.walk(run_cpu)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("optimized_cpu", called_names)
        self.assertNotIn("scalar_oracle", called_names)
        self.assertIn("scalar_oracle", self.source)

    def test_witness_is_preselected_and_reserved_seed_free(self) -> None:
        required_arrays = self.contract["fixed_geometry"]["largest_witness_blocks"]
        for name in required_arrays:
            self.assertIn(name, self.source)
        self.assertIn('metadata.get("reserved_seed_free") is not True', self.source)
        self.assertIn('metadata.get("development_validation_sealed_seeds_used") is not False', self.source)
        self.assertNotIn("8001", self.source)
        self.assertNotIn("91001", self.source)

    def test_outputs_are_fresh_and_separate(self) -> None:
        self.assertIn("refusing to overwrite", self.source)
        self.assertIn("if len(set(outputs)) != 3", self.source)
        self.assertIn("physical_call_artifact", self.source)
        self.assertIn("raw_timing_artifact", self.source)


if __name__ == "__main__":
    unittest.main()
