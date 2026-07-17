from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import contract
import make_eligibility


class ContractTests(unittest.TestCase):
    def test_canonical_contract_and_feature_budget(self) -> None:
        value = contract.load_contract()
        self.assertEqual(
            value["authorized_implementation_parent_commit"],
            "ca0af2af8aa041147658fa4f494d1f26a5dc86a7",
        )
        self.assertEqual(len(value["feature_schema"]["names"]), 70)
        self.assertEqual(value["feature_schema"]["coefficient_count"], 71)
        self.assertLess(value["feature_schema"]["coefficient_count"], 100)
        self.assertEqual(
            contract.feature_names_digest(value["feature_schema"]["names"]),
            value["feature_schema"]["feature_names_sha256"],
        )
        self.assertEqual(value["feature_schema"]["names"][-42], "hold_I")
        self.assertEqual(value["feature_schema"]["names"][-1], "next5_T")
        self.assertEqual(value["feature_schema"]["piece_ids"], ["I", "O", "S", "Z", "J", "L", "T"])
        self.assertEqual(value["canonical_policy"]["candidate_order"], "stable_node_key")
        self.assertEqual(value["openvino_backend"]["complete_chunk"]["device"], "NPU")
        self.assertEqual(value["openvino_backend"]["tail_chunk"]["batch_semantics"], "actual candidate count")
        self.assertEqual(value["analytic_runtime"]["python_version"], "3.12.13")
        self.assertEqual(value["analytic_runtime"]["numpy_version"], "2.4.6")
        self.assertEqual(value["analytic_runtime"]["blas_threads"], 1)

    def test_contract_rejects_seed_or_threshold_tampering(self) -> None:
        value = contract.load_contract()
        changed = copy.deepcopy(value)
        changed["data_roles"]["training_seeds"][0] = 91001
        with self.assertRaises(ValueError):
            contract.validate_contract(changed)
        changed = copy.deepcopy(value)
        changed["fit"]["override_strict_threshold"] = 0.0
        with self.assertRaises(ValueError):
            contract.validate_contract(changed)
        changed = copy.deepcopy(value)
        changed["feature_schema"]["piece_ids"] = sorted(changed["feature_schema"]["piece_ids"])
        with self.assertRaises(ValueError):
            contract.validate_contract(changed)
        changed = copy.deepcopy(value)
        changed["openvino_backend"]["tail_chunk"]["padding"] = True
        with self.assertRaises(ValueError):
            contract.validate_contract(changed)

    def test_static_eligibility_is_exact_and_noninvasive(self) -> None:
        value = make_eligibility.eligibility_document()
        make_eligibility.validate_eligibility(value)
        self.assertEqual(value["planned_training_states"], 288)
        self.assertEqual(value["planned_calibration_states"], 144)
        self.assertFalse(value["game_run"])
        self.assertFalse(value["model_or_checkpoint_loaded"])
        self.assertFalse(value["validation_seed_loaded"])
        self.assertFalse(value["sealed_test_seed_loaded"])

    def test_atomic_write_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r1-contract-") as temporary:
            path = Path(temporary) / "artifact.json"
            contract.atomic_write_json(path, {"finite": 1.0})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"finite": 1.0})
            with self.assertRaises(FileExistsError):
                contract.atomic_write_json(path, {"changed": True})


if __name__ == "__main__":
    unittest.main()
