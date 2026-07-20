"""Static-only contract checks for the concrete v2 live CLI.

This file deliberately does not import Julia, PythonCall, OpenVINO, JLD2, or
touch a checkpoint/workload.  The post-idle gate may execute it later.
"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parent
CLI = (ROOT / "run_heterogeneous_pipeline_v2_live.jl").read_text(encoding="utf-8")
RUNNER = (ROOT / "heterogeneous_pipeline_runner_v2.jl").read_text(encoding="utf-8")
DOC = (ROOT / "HETEROGENEOUS_PIPELINE_V2.md").read_text(encoding="utf-8")


class ConcreteLiveCLIContract(unittest.TestCase):
    def test_runner_has_one_concrete_not_generic_production_builder(self) -> None:
        self.assertIn("build_concrete_live_stage_providers", RUNNER)
        self.assertIn(":BeatFirstHeterogeneousPipelineLiveCLI", RUNNER)
        self.assertIn("typeof(adapter) === getfield(live_module, :LiveAdapter)", RUNNER)
        self.assertIn("run_live_adapter_stage!", RUNNER)
        self.assertIn("validate_live_adapter_stage_outcome!", RUNNER)
        self.assertIn("_take_v2_health_aware!", RUNNER)
        self.assertIn("isready(runner.failures)", RUNNER)
        self.assertNotIn("seal_production_v2_stage_providers", RUNNER)
        self.assertNotIn("register_production_provider", RUNNER)

    def test_exact_train_only_workload_and_reserved_seed_rejection(self) -> None:
        for token in (
            'WORKLOAD_SCHEMA = "heterogeneous-265k-pipeline-v2-train-workload-v1"',
            '"teacher_v3_train"',
            '"reserved_seed_free"',
            '"development_seeds_used"',
            '"validation_seeds_used"',
            '"sealed_seeds_used"',
            "FORBIDDEN_VALIDATION_SEEDS = Set(8001:8008)",
            "FORBIDDEN_SEALED_SEEDS = Set(91001:91032)",
            '"workload_payload_sha256"',
            '"source_parts"',
            '"environment_seeds"',
        ):
            self.assertIn(token, CLI)

    def test_real_stage_adapter_and_physical_readback_are_mandatory(self) -> None:
        for token in (
            "Main._compile_npu(xml)",
            "Main._npu_chunk!",
            "H.hashstem_reference!",
            "S.route_forward!",
            '"NPU_FIXED_B16"',
            '"CPU_TAIL"',
            '"P_CORE_ALL_SPARSE_LAYERS_ONE_CALL"',
            '"physical_calls"',
            '"synchronization_count"',
            '"readback_complete"',
            "_validate_paired_results!",
        ):
            self.assertIn(token, CLI)
        self.assertIn("context.backend == :NPU && context.candidate_count == 16", CLI)
        self.assertIn("context.backend == :CPU_TAIL && context.candidate_count == 16", CLI)
        self.assertIn("any(item -> item.candidate_count == 16, items)", CLI)
        self.assertIn("any(item -> 1 <= item.candidate_count < 16, items)", CLI)

    def test_npu_is_authorized_only_by_byte_pinned_passing_scale_gate(self) -> None:
        for token in (
            '"--npu-gate"',
            '"--npu-gate-sha256"',
            '"hashstem-sparse3-direct-k-scale-decision-v1"',
            '"hashstem-sparse3-cpu-k64-vs-npu-k128-gate-v1"',
            '"npu_k128_route_matches"',
            '"npu_k128_action_top1_matches"',
            '"npu_k128_top2_mismatches"',
            '"npu_execution_devices"',
            '"packed_inputs_sha256"',
            '"state_order_sha256"',
            '"raw_records"',
            '"ADOPTED_MEASURED_BYTE_PINNED"',
        ):
            self.assertIn(token, CLI)
        self.assertIn("sparse_variant == :k128", CLI)

    def test_live_mode_has_no_mock_or_child_process_path(self) -> None:
        self.assertNotIn("V2StageProviders(pack=", CLI)
        self.assertNotIn("_start_v2_pipeline_source_test!", CLI)
        self.assertNotRegex(CLI, r"\baddprocs\b")
        self.assertNotRegex(CLI, r"\brun\s*\(")
        self.assertNotRegex(CLI, r"\bpipeline\s*\(")
        self.assertNotIn("Cmd(", CLI)
        self.assertIn('"single_process" => true', CLI)
        self.assertIn('"child_processes_launched" => false', CLI)

    def test_windows_cpu_sets_one_broker_and_optional_igpu_gate_are_preserved(self) -> None:
        self.assertIn("windows_v2_runtime_hooks()", CLI)
        self.assertIn("start_v2_pipeline!", CLI)
        self.assertIn("enable_igpu_stem_training=false", CLI)
        self.assertIn("iGPU live trainer is unavailable", CLI)
        self.assertIn("same broker", DOC)
        self.assertIn("run_heterogeneous_pipeline_v2_live.jl", DOC)

    def test_artifact_binds_sources_inputs_raw_receipts_and_comparator(self) -> None:
        for token in (
            '"provider_binding_sha256"',
            '"source_bindings"',
            '"input_bindings"',
            '"config_bindings"',
            '"workload_sequence"',
            '"stage_receipts"',
            '"physical_receipts"',
            '"makespan"',
            '"receipts.jsonl"',
            '"receipts_sha256"',
            '"overlap_comparator"',
            '"paired_physical_result_identity" => true',
        ):
            self.assertIn(token, CLI)
        self.assertIn("summarize_v2_receipts(overlapped; require_production=true)", CLI)
        self.assertIn("summarize_v2_receipts(phase; require_production=true)", CLI)
        self.assertIn("observe_overlap_comparator!", CLI)

    def test_no_per_layer_accelerator_roundtrip_or_adoption_claim(self) -> None:
        self.assertEqual(len(re.findall(r"Main\._npu_chunk!", CLI)), 1)
        self.assertIn('"adoption_allowed" => false', CLI)
        self.assertIn("no strength, ETW residency, IMC, or adoption claim", CLI)
        self.assertNotIn("OLD_MODEL_BEATEN", CLI)


if __name__ == "__main__":
    unittest.main()
