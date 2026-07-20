"""Source-only contract checks for the HashStem -> production sparse-3 bridge.

UNEXECUTED_STATIC_ONLY: this test deliberately imports no Julia, OpenVINO,
PyTorch, or device runtime.  Run it only in an idle validation window together
with the Julia runtime tests named in the adjacent contract.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
SPARSE = ROOT.parent / "sparse_dynamic_3layer"
CONTRACT_PATH = ROOT / "hashstem_sparse3_integration_contract_v1.json"


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def source_region(source: str, start: str, stop: str) -> str:
    begin = source.index(start)
    end = source.index(stop, begin + len(start))
    return source[begin:end]


class HashStemSparse3StaticContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(text(CONTRACT_PATH))
        cls.module = text(ROOT / "Heterogeneous265K.jl")
        cls.hashstem = text(ROOT / "hashstem.jl")
        cls.master = text(ROOT / "hashstem_master.jl")
        cls.bridge = text(ROOT / "sparse_hashstem_bridge.jl")
        cls.scale_gate = text(ROOT / "hashstem_sparse3_k_scale_gate.jl")
        cls.sparse_module = text(SPARSE / "SparseDynamic3Layer.jl")
        cls.geometry = text(SPARSE / "geometry.jl")
        cls.runtime = text(SPARSE / "runtime.jl")
        cls.teacher_training = text(SPARSE / "teacher_training.jl")

    def test_contract_freezes_small_stem_and_exact_sparse_geometry(self) -> None:
        contract = self.contract
        self.assertEqual(
            contract["schema"],
            "learned-hashstem-sparse3-integration-contract-v1",
        )
        self.assertEqual(contract["status"], "UNEXECUTED_STATIC_ONLY")
        self.assertEqual(contract["hashstem_abi"]["trainable_parameters"], 223_504)
        self.assertEqual(contract["hashstem_abi"]["macs_per_candidate"], 433_546)
        self.assertEqual(contract["cpu_sparse_path"]["total_parameters"], 19_924_022)
        self.assertEqual(
            contract["combined_accounting"]["trainable_parameters"], 20_147_526
        )
        variants = contract["cpu_sparse_path"]["named_variants"]
        self.assertEqual(variants["k64"]["active_counts"], [24, 20, 20])
        self.assertEqual(variants["k128"]["active_counts"], [48, 40, 40])
        self.assertEqual(variants["k256"]["active_counts"], [96, 80, 80])
        combined = contract["combined_accounting"]["by_named_sparse_variant"]
        self.assertEqual(
            combined["k64"][
                "parameters_touched_per_candidate_when_training_both_components"
            ],
            255_438,
        )
        self.assertEqual(combined["k128"]["learned_forward_macs_per_candidate"], 491_738)
        self.assertTrue(contract["separation"]["pure_cpu_sparse_model_is_unchanged"])
        self.assertFalse(contract["separation"]["nnue_or_parent_child_reuse"])

    def test_one_stem_graph_emits_every_base_query_and_context(self) -> None:
        expected = {
            "const QUERY_1_RANGE = 1:64",
            "const QUERY_2_RANGE = 65:128",
            "const QUERY_3_RANGE = 129:192",
            "const CONTEXT_RANGE = 193:214",
            "const AUXILIARY_RANGE = 193:202",
            "const NEXT_HOLD_PASSTHROUGH_RANGE = 215:256",
            "output = permutedims(vcat(learned, raw_next_hold))",
        }
        for needle in expected:
            self.assertIn(needle, self.hashstem)
        semantics = self.contract["hashstem_abi"]["single_graph_execution_semantics"]
        self.assertTrue(semantics["one_graph_execution_emits_all_sparse_layer_base_queries"])
        self.assertFalse(semantics["per_sparse_layer_hashstem_reinvocation"])
        self.assertEqual(semantics["cpu_npu_cpu_round_trips_between_sparse_layers"], 0)

    def test_adapter_consumes_a_complete_output_without_accelerator_callbacks(self) -> None:
        build_inputs = source_region(
            self.bridge,
            "function three_layer_inputs_from_hashstem(",
            "struct SparseHashStemForwardBatch",
        )
        route_batch = source_region(
            self.bridge,
            "function route_hashstem_batch!(",
            "struct SparseHashStemBackwardBatch",
        )
        for needle in (
            "QUERY_1_RANGE",
            "QUERY_2_RANGE",
            "QUERY_3_RANGE",
            "CONTEXT_RANGE",
            "NEXT_HOLD_PASSTHROUGH_RANGE",
            "SparseDynamic3Layer.ThreeLayerInput",
        ):
            self.assertIn(needle, build_inputs)
        self.assertIn("SparseDynamic3Layer.route_forward!", route_batch)
        for forbidden in (
            "hashstem_reference!",
            "OpenVINO",
            "PythonCall",
            "start_async",
            "compile_model",
            "train_hashstem_master_step!",
        ):
            self.assertNotIn(forbidden, build_inputs)
            self.assertNotIn(forbidden, route_batch)

    def test_l2_l3_queries_add_cpu_countsketch_residuals(self) -> None:
        route = source_region(
            self.runtime,
            "function route_forward!(\n    runtime::ThreeLayerRuntime,",
            "function route_forward!(\n    runtime::ThreeLayerRuntime,\n    workspace::ThreeLayerWorkspace,\n    q::AbstractVector",
        )
        self.assertIn(
            "_compose_deep_query!(q2, input.base_queries[2], ids1_copy, a1, 1)",
            route,
        )
        self.assertIn(
            "_compose_deep_query!(q3, input.base_queries[3], ids2_copy, a2, 2)",
            route,
        )
        compose = source_region(
            self.geometry,
            "function _compose_deep_query!(",
            "function _scatter_context_next_to_q!",
        )
        self.assertIn("_sketch_location", compose)
        self.assertIn("copyto!(destination, base_q)", compose)
        self.assertIn("muladd", compose)

    def test_pure_cpu_module_has_no_reverse_dependency(self) -> None:
        for forbidden in (
            "heterogeneous_265k",
            "HashStem",
            "OpenVINO",
            "PythonCall",
        ):
            self.assertNotIn(forbidden, self.sparse_module)
        self.assertNotIn('include("nnue.jl")', self.sparse_module.lower())
        self.assertIn("const PRODUCTION_DENSE_FALLBACK = false", self.geometry)
        self.assertIn("@assert TOTAL_PARAMETERS == 19_924_022", self.geometry)
        self.assertIn("@assert ACTIVE_PARAMETERS == 34_338", self.geometry)

    def test_module_include_order_and_inference_only_snapshot_boundary(self) -> None:
        hashstem_index = self.module.index('include("hashstem.jl")')
        sparse_index = self.module.index(
            'Base.include(\n        Main,\n        joinpath(@__DIR__, "..", '
            '"sparse_dynamic_3layer", "SparseDynamic3Layer.jl"),'
        )
        bridge_index = self.module.index('include("sparse_hashstem_bridge.jl")')
        self.assertLess(hashstem_index, sparse_index)
        self.assertLess(sparse_index, bridge_index)
        self.assertIn("using Main.SparseDynamic3Layer", self.module)
        self.assertIn('trainer.backend === :cpu || error(', self.master)
        self.assertIn('npu_backward=false', self.master)
        self.assertIn('igpu_authorized=false', self.master)
        self.assertIn('available=false', source_region(
            self.master,
            "function hashstem_master_backend_status",
            "mutable struct HashStemMasterTrainer",
        ))

    def test_bridge_binds_named_active_width_in_checkpoint_and_route(self) -> None:
        for needle in (
            "function sparse_bridge_variant_spec",
            'sparse_variant::String',
            'sparse_teacher_checkpoint_sha256::String',
            '"bridge_source_sha256" => _sha256_file(@__FILE__)',
            '"sparse_active_counts" => collect(variant.active_counts)',
            '"sparse_training_probes" => collect(variant.training_probes)',
            "record.active_counts == variant.active_counts",
            "variant.training_probes",
        ):
            self.assertIn(needle, self.bridge)
        self.assertNotIn(
            "record.active_counts == SparseDynamic3Layer.LAYER_ACTIVE_COUNTS",
            self.bridge,
        )
        gate = self.contract["primary_k_scale_gate"]
        self.assertEqual(gate["baseline"], "CPU HashStem plus CPU sparse k64")
        self.assertEqual(gate["candidate"], "NPU HashStem plus CPU sparse k128")
        self.assertEqual(gate["route_id_total_formula"], "3 * candidate_count")
        self.assertEqual(
            gate["action_top1_and_top2_total_formula"], "timed_sample_count"
        )
        self.assertEqual(gate["quality_direction"], "higher_is_better")
        self.assertEqual(
            gate["supported_quality_metrics"],
            [
                "teacher_top1_agreement",
                "teacher_ndcg",
                "paired_dev_game_score_mean",
            ],
        )
        for needle in (
            "active_counts=(24, 20, 20)",
            "active_counts=(48, 40, 40)",
            "active_counts=(96, 80, 80)",
            "training_probes=(3, 2, 2)",
            "training_probes=(6, 5, 5)",
            "training_probes=(12, 10, 10)",
        ):
            self.assertIn(needle, self.bridge)
            self.assertIn(needle, self.teacher_training)

    def test_teacher_checkpoint_has_an_explicit_bridge_bound_producer(self) -> None:
        producer = source_region(
            self.bridge,
            "function bind_teacher_checkpoint_for_hashstem_bridge(",
            '"""Load a byte-pinned production 3-layer runtime',
        )
        for needle in (
            "expected_teacher_checkpoint_sha256",
            "lineage.sparse_teacher_checkpoint_sha256",
            "SparseDynamic3Layer.load_checkpoint(source)",
            "_validate_production_topology",
            "_validate_teacher_checkpoint_binding",
            "training_state=loaded.training_state",
            "merge!(metadata, sparse_bridge_checkpoint_metadata(lineage))",
            "load_sparse_hashstem_bridge(",
        ):
            self.assertIn(needle, producer)
        self.assertIn('"variant" => String(config.variant)', self.teacher_training)

    def test_teacher_module_identity_and_strict_continuation_are_shared(self) -> None:
        self.assertIn("Base.include(\n        Main,", self.module)
        self.assertIn("using Main.SparseDynamic3Layer", self.module)
        self.assertIn("using Main.SparseDynamic3Layer", self.teacher_training)
        strict_loader = source_region(
            self.bridge,
            "function load_sparse_hashstem_bridge(",
            '"""Rebuild exact sparse `x496`',
        )
        self.assertIn(
            "_validate_teacher_continuation(loaded, lineage, variant)",
            strict_loader,
        )
        continuation = source_region(
            self.bridge,
            "function _validate_teacher_continuation(",
            '"""Create a fresh bridge-bound checkpoint',
        )
        for needle in (
            'state === nothing && error("teacher checkpoint has no continuation state")',
            "String(config.variant) == lineage.sparse_variant",
            "Tuple(Int.(config.training_probes)) == variant.training_probes",
            "UInt64(state.update) == step",
        ):
            self.assertIn(needle, continuation)

    def test_k_scale_gate_is_executable_not_only_a_json_label(self) -> None:
        for needle in (
            "struct HashStemSparse3GateCell",
            "function evaluate_hashstem_sparse3_k_scale_gate(",
            '"cpu_k64", "k64", "CPU"',
            '"cpu_k128", "k128", "CPU"',
            '"npu_k128", "k128", "NPU"',
            "115) * stem_p50_npu <= Int128(100) * stem_p50_cpu",
            "p50_128_npu <= p50_64",
            "p95_128_npu <= p95_64",
            "npu_k128.quality_value > cpu_k64.quality_value",
            "expected_route_total = 3 * cell.candidate_count",
            "expected_action_total = length(cell.sample_ids)",
            "_HASHSTEM_SPARSE3_HIGHER_IS_BETTER_METRICS",
            'cell.quality_direction == "higher_is_better"',
            "npu_k128.maximum_hashstem_absolute_error <= 1.0e-2",
        ):
            self.assertIn(needle, self.scale_gate)
        self.assertIn('include("hashstem_sparse3_k_scale_gate.jl")', self.module)


if __name__ == "__main__":
    unittest.main()
