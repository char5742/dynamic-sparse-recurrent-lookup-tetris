from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import contract
import make_eligibility


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]


def julia_executable() -> str:
    configured = os.environ.get("R1_JULIA_EXE")
    executable = configured or shutil.which("julia")
    if executable is None:
        raise unittest.SkipTest("Julia executable is unavailable")
    return executable


class FreezeDesignProductionTests(unittest.TestCase):
    def run_freeze(self, eligibility: Path, output: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                julia_executable(),
                "--startup-file=no",
                "--history-file=no",
                f"--project={REPOSITORY}",
                str(ROOT / "freeze_design.jl"),
                str(eligibility),
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_fresh_process_production_branch_is_exact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r1-freeze-production-") as temporary:
            directory = Path(temporary)
            eligibility = directory / "eligibility.json"
            output = directory / "design_freeze.json"
            contract.atomic_write_json(eligibility, make_eligibility.eligibility_document())
            result = self.run_freeze(eligibility, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(output.read_text(encoding="utf-8"))
            canonical = contract.load_contract()
            self.assertEqual(value["status"], "r1_design_frozen")
            self.assertEqual(value["feature_names"], canonical["feature_schema"]["names"])
            self.assertEqual(
                value["feature_names_sha256"],
                canonical["feature_schema"]["feature_names_sha256"],
            )
            self.assertEqual(len(value["training_bootstrap_schedules"]), 256)
            self.assertTrue(all(len(draw) == 12 for draw in value["training_bootstrap_schedules"]))
            self.assertEqual(len(value["calibration_bootstrap_schedules"]), 2000)
            self.assertTrue(all(len(draw) == 6 for draw in value["calibration_bootstrap_schedules"]))
            self.assertEqual(value["canonical_policy"]["candidate_order"], "stable_node_key")
            self.assertEqual(value["openvino_backend"]["version"], "2026.2.1")
            self.assertEqual(value["openvino_backend"]["complete_chunk"], {
                "device": "NPU", "batch_size": 16, "shape": "static", "eligible_candidate_count": 16,
            })
            self.assertEqual(value["openvino_backend"]["tail_chunk"]["device"], "CPU")
            self.assertEqual(value["openvino_backend"]["tail_chunk"]["batch_semantics"], "actual candidate count")
            self.assertFalse(value["openvino_backend"]["tail_chunk"]["padding"])
            self.assertEqual(
                value["openvino_backend"]["weight_sha256"],
                value["immutable_inputs"]["old_openvino_weight_npz_sha256"],
            )
            self.assertFalse(value["game_run"])
            self.assertFalse(value["validation_seed_loaded"])
            self.assertFalse(value["sealed_test_seed_loaded"])

    def test_tampered_role_is_rejected_without_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r1-freeze-tamper-") as temporary:
            directory = Path(temporary)
            eligibility = directory / "eligibility.json"
            output = directory / "design_freeze.json"
            value = make_eligibility.eligibility_document()
            value["training_seed_ids"][0] = 91001
            contract.atomic_write_json(eligibility, value)
            result = self.run_freeze(eligibility, output)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
