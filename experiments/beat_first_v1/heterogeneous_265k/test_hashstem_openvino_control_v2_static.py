"""Standard-library-only static contract checks for the unexecuted v2 source.

This test intentionally parses source text/AST and never imports NumPy,
OpenVINO, or the benchmark implementation. Running it cannot enumerate a
device, compile an IR, or issue inference.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / "hashstem_openvino_control_v2.py"
CLI = ROOT / "benchmark_hashstem_openvino_v2.py"


class HashStemOpenVINOControlV2StaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.control = CONTROL.read_text(encoding="utf-8")
        cls.cli = CLI.read_text(encoding="utf-8")
        cls.control_ast = ast.parse(cls.control, filename=str(CONTROL))
        cls.cli_ast = ast.parse(cls.cli, filename=str(CLI))

    @staticmethod
    def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
        matches = [
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        if len(matches) != 1:
            raise AssertionError(f"expected exactly one function {name}, got {len(matches)}")
        return matches[0]

    @staticmethod
    def _method(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef:
        classes = [
            node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        if len(classes) != 1:
            raise AssertionError(f"expected exactly one class {class_name}, got {len(classes)}")
        matches = [
            node
            for node in classes[0].body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"expected exactly one {class_name}.{method_name}, got {len(matches)}"
            )
        return matches[0]

    @staticmethod
    def _call_attributes(node: ast.AST) -> list[str]:
        return [
            call.func.attr
            for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        ]

    def test_sources_parse_without_importing_runtime(self) -> None:
        self.assertIsInstance(self.control_ast, ast.Module)
        self.assertIsInstance(self.cli_ast, ast.Module)

    def test_same_fixed_ir_is_compiled_on_explicit_cpu_and_npu(self) -> None:
        benchmark = self._function(self.control_ast, "benchmark_control")
        compile_devices = []
        for call in ast.walk(benchmark):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                continue
            if call.func.id != "compile_exact" or len(call.args) < 3:
                continue
            model = call.args[1]
            device = call.args[2]
            if isinstance(model, ast.Name) and model.id == "fixed_ir" and isinstance(device, ast.Constant):
                compile_devices.append(device.value)
        self.assertEqual(sorted(compile_devices), ["CPU", "NPU"])
        self.assertIn('execution != [requested]', self.control)
        self.assertIn('requested not in {"CPU", "NPU"}', self.control)
        self.assertNotIn('compile_model(str(model_path), "AUTO"', self.control)

    def test_actual_length_cpu_tail_is_not_npu_padding(self) -> None:
        self.assertIn('compile_exact(\n            core, dynamic_cpu_ir, "CPU"', self.control)
        self.assertIn('call_kind == "CPU_TAIL"', self.control)
        self.assertIn('rows < BATCH', self.control)
        self.assertIn('packed[self.full_rows :]', self.control)
        self.assertNotIn("np.pad", self.control)

    def test_each_required_physical_stage_has_an_api_receipt(self) -> None:
        for stage in ("pack", "bind", "h2d", "submit", "wait", "d2h", "copy"):
            self.assertIn(f'"{stage}"', self.control)
        for call in (
            "set_input_tensor",
            "copy_from",
            "start_async",
            "request.wait",
            "set_output_tensor",
            "np.copyto",
        ):
            self.assertIn(call, self.control)
        self.assertIn('"dma_timing_isolated": False', self.control)
        self.assertIn('"end_to_end_ns"', self.control)

    def test_cpu_baseline_is_openvino_and_scalar_is_never_called(self) -> None:
        self.assertIn('"scalar_reference_timed": False', self.control)
        self.assertNotIn("hashstem_reference!", self.control.split("class PersistentOpenVINOCall", 1)[1])
        self.assertIn('PersistentOpenVINOCall(\n        cpu_fixed', self.control)
        constructor = self._method(self.control_ast, "PersistentOpenVINOCall", "__init__")
        attributes = self._call_attributes(constructor)
        self.assertIn("set_input_tensor", attributes)
        self.assertIn("set_output_tensor", attributes)
        shared_tensor_calls = [
            call
            for call in ast.walk(constructor)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "Tensor"
            and any(
                keyword.arg == "shared_memory"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in call.keywords
            )
        ]
        self.assertGreaterEqual(len(shared_tensor_calls), 2)
        run = self._method(self.control_ast, "PersistentOpenVINOCall", "run")
        self.assertIn("copy_from", self._call_attributes(run))
        self.assertNotIn("set_input_tensor", self._call_attributes(run))
        self.assertNotIn("set_output_tensor", self._call_attributes(run))

    def test_only_registered_sparse_widths_are_accepted(self) -> None:
        for literal in (
            "64: (24, 20, 20)",
            "128: (48, 40, 40)",
            "256: (96, 80, 80)",
        ):
            self.assertIn(literal, self.control)
        self.assertIn("choices=(64, 128, 256)", self.cli)

    def test_validation_and_sealed_seeds_fail_closed(self) -> None:
        self.assertIn("range(8001, 8009)", self.control)
        self.assertIn("range(91001, 91033)", self.control)
        self.assertIn('split") != "teacher_v3_train"', self.control)
        provenance = self._function(self.control_ast, "validate_witness_provenance")
        provenance_text = ast.unparse(provenance)
        for binding in (
            "dataset_manifest_sha256",
            "witness_generator_source_sha256",
            "source_parts",
            "episode_key",
            "part_sha256",
            "held_out_development_validation_sealed_seeds_used",
            "resolved_part.relative_to(dataset_root)",
            "sha256_file(resolved_part)",
        ):
            self.assertIn(binding, provenance_text)
        for option in (
            "--dataset-manifest-sha256",
            "--witness-generator-source-sha256",
        ):
            self.assertIn(option, self.cli)

    def test_receipt_gate_checks_both_paths_and_contiguous_rows(self) -> None:
        validator = self._function(self.control_ast, "_trace_receipts_exact")
        validator_text = ast.unparse(validator)
        for invariant in (
            "call.row_begin != cursor",
            "cursor != candidate_count",
            "call.execution_devices",
            "call.call_kind",
            "call.intervals",
            "call.call_begin_ns < previous_call_end",
            "call.input_bytes != rows * INPUTS * FLOAT_BYTES",
        ):
            self.assertIn(invariant, validator_text)
        benchmark_text = ast.unparse(self._function(self.control_ast, "benchmark_control"))
        self.assertIn("path='CPU_CONTROL'", benchmark_text)
        self.assertIn("path='NPU_PLUS_CPU_TAIL'", benchmark_text)
        self.assertIn("cpu_trace_gate", benchmark_text)
        self.assertIn("npu_trace_gate", benchmark_text)

    def test_component_speed_gate_uses_p50_and_p95_not_pooled_total(self) -> None:
        benchmark = self._function(self.control_ast, "benchmark_control")
        benchmark_text = ast.unparse(benchmark)
        self.assertIn("p50_speed_gate = 115 * npu_p50_ns <= 100 * cpu_p50_ns", benchmark_text)
        self.assertIn("p95_no_worse_gate = npu_p95_ns <= cpu_p95_ns", benchmark_text)
        self.assertIn("'component_p50_speedup_ge_1_15': p50_speed_gate", benchmark_text)
        self.assertIn("'component_p95_no_worse': p95_no_worse_gate", benchmark_text)
        self.assertIn("'pooled_speedup_is_gate': False", benchmark_text)
        self.assertNotIn("'component_speedup_ge_1_15'", benchmark_text)

    def test_result_cannot_authorize_sparse_or_game_adoption(self) -> None:
        self.assertIn('"adoption_authorized": False', self.control)
        self.assertIn('"system_adoption_authorized": False', self.control)
        self.assertIn('"integrated_sparse_executed": False', self.control)
        self.assertIn('"component_action_metrics_computed": False', self.control)
        self.assertIn('"component_route_ids_are_final_action_ranking": False', self.control)
        self.assertIn("COMPONENT_PASS_PENDING_INTEGRATED_SPARSE_K_GATE", self.control)
        for required in (
            "candidate/action top-1 exact agreement",
            "candidate/action top-2 swap rate",
            "action-margin error",
            "CPU sparse p50/p95 slowdown under overlap",
            "teacher ranking accuracy",
            "development game score",
        ):
            self.assertIn(required, self.control)


if __name__ == "__main__":
    unittest.main()
