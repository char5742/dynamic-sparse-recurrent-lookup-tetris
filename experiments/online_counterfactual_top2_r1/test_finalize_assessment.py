from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import contract
import finalize_assessment as finalizer
import make_eligibility


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
FROZEN_PYTHON = Path(r"D:\tetris-paper-plus\python-env\Scripts\python.exe")
CONCRETE_JULIA = Path(
    r"C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
)
FEATURES = contract.load_contract()["feature_schema"]["names"]
FEATURE_DIGEST = contract.load_contract()["feature_schema"]["feature_names_sha256"]
TRAIN = list(range(73001, 73013))
CALIBRATION = list(range(73101, 73107))
PIECES = list(range(10, 241, 10))
STABLE_DIGEST = hashlib.sha256(b"synthetic-stable-node-key-source").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def julia_executable() -> Path:
    if CONCRETE_JULIA.is_file():
        return CONCRETE_JULIA
    value = shutil.which("julia")
    if value is None:
        raise unittest.SkipTest("Julia is unavailable")
    return Path(value)


def make_collection(role: str, table_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    episodes = TRAIN if role == "training" else CALIBRATION
    total = len(episodes) * 24
    rows: list[dict[str, Any]] = []
    episode_evidence: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for episode_slot, episode in enumerate(episodes, 1):
        for state_slot, piece in enumerate(PIECES[:20], 1):
            top2_safe = state_slot <= 2
            q1 = 2.0 if top2_safe else 0.0
            q2 = 0.0
            gap = q1 - q2
            features = [0.0] * 70
            features[0] = q1
            features[1] = q2
            features[2] = gap
            features[6] = 3.0
            advantage = 0.80 + 0.01 * episode_slot if top2_safe else -0.20
            rows.append(
                {
                    "episode_id": episode,
                    "seed": episode,
                    "piece_index": piece,
                    "features": features,
                    "advantage": advantage,
                    "advantage_unclipped_A6": advantage,
                    "clipped_target": advantage,
                    "a1_terminal_within_horizon": False,
                    "a2_terminal_within_horizon": False,
                    "root_state_digest": hashlib.sha256(f"root-{role}-{episode}-{piece}".encode()).hexdigest(),
                    "root_future_stream_digest": hashlib.sha256(f"future-{role}-{episode}-{piece}".encode()).hexdigest(),
                    "canonical_top1_candidate_index": 1,
                    "canonical_top2_candidate_index": 2,
                    "canonical_top1_action_digest": f"a1-{episode}-{piece}",
                    "canonical_top2_action_digest": f"a2-{episode}-{piece}",
                    "q_top1": q1,
                    "q_top2": q2,
                    "q_gap": gap,
                    "valid_action_count": 3,
                }
            )
        for piece in PIECES[20:]:
            exclusions.append(
                {
                    "seed": episode,
                    "episode_id": episode,
                    "piece_index": piece,
                    "code": "synthetic_exclusion",
                    "detail": "fixture",
                    "root_state_digest": hashlib.sha256(f"excluded-{role}-{episode}-{piece}".encode()).hexdigest(),
                }
            )
        episode_evidence.append(
            {
                "seed": episode,
                "episode_id": episode,
                "canonical_pieces": 240,
                "canonical_score": 0,
                "canonical_terminal": False,
                "canonical_action_digests": [],
                "rows": 20,
                "exclusions": 4,
            }
        )
    positive_fraction = sum(row["advantage_unclipped_A6"] > 0.0 for row in rows) / len(rows)
    metadata = {
        "source_role": role,
        "feature_names": FEATURES,
        "feature_schema_digest": FEATURE_DIGEST,
        "training_seeds": episodes if role == "training" else [],
        "role_seeds": episodes,
        "sample_pieces": PIECES,
        "synthetic": True,
        "stable_node_key_source_sha256": STABLE_DIGEST,
        "backend_binding": None,
        "engine_dependency_graph": None,
        "immutable_input_end_hashes": None,
        "counterfactual_states_completed": total,
        "first32_elapsed_seconds": 0.01 if role == "training" else None,
        "first32_setup_seconds": 0.01 if role == "training" else None,
        "first32_projected_seconds": 0.155 if role == "training" else None,
        "first32_projection_limit_seconds": 3300.0,
        "first32_projection_basis_states": 432,
        "first32_projection_formula": "2*setup_seconds + first32_collection_seconds/32*432",
        "positive_advantage_fraction": positive_fraction,
        "validation_seed_used": False,
        "sealed_test_seed_used": False,
    }
    table = {
        "schema_version": f"r1-{role}-table-v1",
        "source_role": role,
        "metadata": metadata,
        "feature_names": FEATURES,
        "feature_schema_digest": FEATURE_DIGEST,
        "training_seeds": episodes if role == "training" else [],
        "synthetic": True,
        "validation_seed_used": False,
        "sealed_test_seed_used": False,
        "rows": rows,
    }
    write_json(table_path, table)
    manifest = {
        "schema_version": "r1-collection-manifest-v1",
        "source_role": role,
        "synthetic": True,
        "table_path": str(table_path.resolve()),
        "table_sha256": sha256(table_path),
        "feature_names": FEATURES,
        "feature_schema_digest": FEATURE_DIGEST,
        "stable_node_key_source_sha256": STABLE_DIGEST,
        "backend_binding": None,
        "engine_dependency_graph": None,
        "immutable_input_end_hashes": None,
        "episode_count": len(episodes),
        "row_count": len(rows),
        "exclusion_count": len(exclusions),
        "role_seeds": episodes,
        "counterfactual_states_completed": total,
        "first32_elapsed_seconds": metadata["first32_elapsed_seconds"],
        "first32_setup_seconds": metadata["first32_setup_seconds"],
        "first32_projected_seconds": metadata["first32_projected_seconds"],
        "first32_projection_limit_seconds": 3300.0,
        "first32_projection_basis_states": 432,
        "first32_projection_formula": "2*setup_seconds + first32_collection_seconds/32*432",
        "positive_advantage_fraction": positive_fraction,
        "exclusions": exclusions,
        "episodes": episode_evidence,
        "real_model_or_game_loaded": False,
        "validation_seed_used": False,
        "sealed_test_seed_used": False,
        "wall_seconds": 0.02,
    }
    return table, manifest


def make_ridge(training_table: Path, training_manifest: Path, freeze: Path) -> dict[str, Any]:
    coefficients = [[0.0] * 256 for _ in range(71)]
    coefficients[0] = [-1.0] * 256
    coefficients[1] = [1.0] * 256
    training_rows = json.loads(training_table.read_text(encoding="utf-8"))["rows"]
    row_order_sha256 = hashlib.sha256(
        "\n".join(
            f"{row['episode_id']},{row['piece_index']},{row['root_state_digest']}"
            for row in training_rows
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "r1-ridge-gate-v1",
        "feature_names": FEATURES,
        "feature_schema_digest": FEATURE_DIGEST,
        "feature_mean": [0.0] * 70,
        "feature_scale": [1.0] * 70,
        "constant_feature": [False] * 70,
        "coefficient_shape": [71, 256],
        "coefficients": coefficients,
        "lambda": 1.0,
        "bootstrap_rng": "Xoshiro(0x5231_2026)",
        "bootstrap_seed": 0x5231_2026,
        "lower_quantile": 0.10,
        "quantile_method": "linear_type7_position_1_plus_n_minus_1_p",
        "override_threshold": 0.05,
        "target_clamp": [-2.0, 2.0],
        "ensemble_size": 256,
        "experiment_id": "online_counterfactual_top2_R1",
        "fit_role": "training_only",
        "fit_backend": "python_numpy_analytic_ridge",
        "runtime_facts": {
            "python_version": "3.12.13",
            "python_executable": str(FROZEN_PYTHON.resolve()),
            "numpy_version": "2.4.6",
            "numpy_show_config": {},
            "linear_algebra": "numpy.linalg.cholesky+numpy.linalg.solve",
            "thread_environment": {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        },
        "source_table_sha256": sha256(training_table),
        "source_table_path": str(training_table.resolve()),
        "source_table_synthetic": True,
        "source_collection_manifest_path": str(training_manifest.resolve()),
        "source_collection_manifest_sha256": sha256(training_manifest),
        "engine_dependency_graph": None,
        "training_row_order_sha256": row_order_sha256,
        "training_row_order_encoding": "episode_id,piece_index,root_state_digest newline joined",
        "design_freeze_path": str(freeze.resolve()),
        "design_freeze_sha256": sha256(freeze),
        "training_bootstrap_schedule_sha256": "5b60a1e340b542dc8654a5c80777c254d8336aa086e51be6c2ba1251be20e5f7",
        "training_bootstrap_schedule_source_anchor_sha256": "5b60a1e340b542dc8654a5c80777c254d8336aa086e51be6c2ba1251be20e5f7",
        "training_bootstrap_schedule_consumed": True,
        "training_stats": {
            "row_count": 240,
            "episode_count": 12,
            "episode_ids": TRAIN,
            "positive_fraction_unclipped": 0.10,
            "advantage_unclipped_min": -0.20,
            "advantage_unclipped_max": 0.92,
            "advantage_unclipped_mean": -0.09,
            "target_clipped_min": -0.20,
            "target_clipped_max": 0.92,
            "target_clipped_mean": -0.09,
            "constant_feature_count": 0,
            "constant_feature_indices": [],
        },
        "all_finite": True,
        "validation_seed_used": False,
        "sealed_test_seed_used": False,
        "claim_scope": "analytic_training_artifact_not_calibration_or_game_strength",
    }


class FinalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        if not FROZEN_PYTHON.is_file():
            self.skipTest("frozen Python is unavailable")
        self.temporary = tempfile.TemporaryDirectory(prefix="r1-finalizer-")
        self.directory = Path(self.temporary.name)
        eligibility = self.directory / "eligibility.json"
        contract.atomic_write_json(eligibility, make_eligibility.eligibility_document())
        self.freeze = self.directory / "freeze.json"
        freeze = subprocess.run(
            [
                str(julia_executable()), "--startup-file=no", "--history-file=no",
                f"--project={REPOSITORY}", str(ROOT / "freeze_design.jl"),
                str(eligibility), str(self.freeze),
            ],
            check=False, capture_output=True, text=True,
        )
        self.assertEqual(freeze.returncode, 0, freeze.stderr)
        self.training_table = self.directory / "training.json"
        training, training_manifest = make_collection("training", self.training_table)
        self.training_manifest = self.directory / "training_manifest.json"
        write_json(self.training_manifest, training_manifest)
        self.calibration_table = self.directory / "calibration.json"
        calibration, calibration_manifest = make_collection("calibration", self.calibration_table)
        self.calibration_manifest = self.directory / "calibration_manifest.json"
        write_json(self.calibration_manifest, calibration_manifest)
        self.ridge = self.directory / "ridge.json"
        write_json(self.ridge, make_ridge(self.training_table, self.training_manifest, self.freeze))
        self.calibration_assessment = self.directory / "calibration_assessment.json"
        calibration_milestones = self.directory / "calibration_milestones.jsonl"
        handoff_environment = os.environ.copy()
        handoff_environment.update(
            {
                "R1_EXPECTED_CALIBRATION_TABLE_SHA256": sha256(self.calibration_table),
                "R1_EXPECTED_CALIBRATION_MANIFEST_SHA256": sha256(self.calibration_manifest),
                "R1_EXPECTED_RIDGE_ARTIFACT_SHA256": sha256(self.ridge),
                "R1_EXPECTED_DESIGN_FREEZE_SHA256": sha256(self.freeze),
            }
        )
        gate = subprocess.run(
            [
                str(FROZEN_PYTHON), str(ROOT / "calibration_gate.py"),
                str(self.calibration_table), str(self.calibration_manifest), str(self.ridge),
                str(self.freeze), str(self.calibration_assessment), str(calibration_milestones),
                "--synthetic",
            ],
            check=False, capture_output=True, text=True, env=handoff_environment,
        )
        self.assertEqual(gate.returncode, 0, gate.stderr)
        result = json.loads(self.calibration_assessment.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "R1-calibration-promoted")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_finalizer(
        self,
        suffix: str,
        *,
        table: Path | None = None,
        assessment: Path | None = None,
        ridge: Path | None = None,
        calibration_table: Path | None = None,
        calibration_manifest: Path | None = None,
    ) -> dict[str, Any]:
        output = self.directory / f"assessment_{suffix}.json"
        milestones = self.directory / f"finalizer_{suffix}.jsonl"
        environment = os.environ.copy()
        environment.update(
            {
                "R1_EXPECTED_TRAINING_TABLE_SHA256": (
                    sha256(table or self.training_table)
                    if (table or self.training_table).is_file() else "0" * 64
                ),
                "R1_EXPECTED_TRAINING_MANIFEST_SHA256": sha256(self.training_manifest),
                "R1_EXPECTED_CALIBRATION_TABLE_SHA256": sha256(calibration_table or self.calibration_table),
                "R1_EXPECTED_CALIBRATION_MANIFEST_SHA256": sha256(calibration_manifest or self.calibration_manifest),
                "R1_EXPECTED_RIDGE_ARTIFACT_SHA256": sha256(ridge or self.ridge),
                "R1_EXPECTED_DESIGN_FREEZE_SHA256": sha256(self.freeze),
                "R1_EXPECTED_CALIBRATION_ASSESSMENT_SHA256": sha256(assessment or self.calibration_assessment),
            }
        )
        result = subprocess.run(
            [
                str(FROZEN_PYTHON), str(ROOT / "finalize_assessment.py"),
                str(table or self.training_table), str(self.training_manifest), str(ridge or self.ridge),
                str(calibration_table or self.calibration_table), str(calibration_manifest or self.calibration_manifest),
                str(assessment or self.calibration_assessment), str(self.freeze),
                str(output), str(milestones), "--synthetic",
            ],
            check=False, capture_output=True, text=True, env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(output.is_file())
        stages = [json.loads(line)["stage"] for line in milestones.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(stages[0], "script_enter")
        self.assertEqual(stages[-1], "phase_complete")
        return json.loads(output.read_text(encoding="utf-8"))

    def test_exact_production_argv_promotes_synthetic_fixture(self) -> None:
        result = self.run_finalizer("pass")
        self.assertEqual(result["status"], "assessment-pass")
        self.assertEqual(result["promotion"], "R1-calibration-promoted")
        self.assertTrue(result["success"])
        self.assertEqual(result["training_row_count"], 240)
        self.assertEqual(result["calibration_override_count"], 12)
        for field in (
            "game_strength_evidence", "model_beat_claim", "game_evaluation_authorized",
            "development_authorized", "validation_seed_used", "sealed_test_seed_used",
            "sealed_test_authorized", "game_run",
        ):
            self.assertIs(result[field], False)

    def test_tampered_calibration_metric_fails_closed(self) -> None:
        value = json.loads(self.calibration_assessment.read_text(encoding="utf-8"))
        value["override_precision"] = 0.123
        tampered = self.directory / "tampered_assessment.json"
        write_json(tampered, value)
        result = self.run_finalizer("tamper", assessment=tampered)
        self.assertEqual(result["status"], "assessment-fail")
        self.assertFalse(result["success"])
        self.assertTrue(any("override_precision" in reason for reason in result["reasons"]))
        self.assertFalse(result["development_authorized"])

    def test_tampered_source_table_breaks_manifest_binding(self) -> None:
        value = json.loads(self.training_table.read_text(encoding="utf-8"))
        value["rows"][0]["advantage_unclipped_A6"] = 1.25
        value["rows"][0]["advantage"] = 1.25
        value["rows"][0]["clipped_target"] = 1.25
        tampered = self.directory / "tampered_training.json"
        write_json(tampered, value)
        # Preserve the original manifest deliberately: its absolute table path
        # and hash must both make this alternate source fail closed.
        result = self.run_finalizer("tampered_source", table=tampered)
        self.assertEqual(result["status"], "assessment-fail")
        self.assertTrue(
            any("manifest table path mismatch" in reason for reason in result["reasons"])
        )
        self.assertFalse(result["game_evaluation_authorized"])

    def test_invalid_training_row_order_digest_fails_closed(self) -> None:
        value = json.loads(self.ridge.read_text(encoding="utf-8"))
        value["training_row_order_sha256"] = "0" * 64
        tampered = self.directory / "tampered_row_order_ridge.json"
        write_json(tampered, value)
        result = self.run_finalizer("row_order", ridge=tampered)
        self.assertEqual(result["status"], "assessment-fail")
        self.assertTrue(
            any("row-order digest mismatch" in reason for reason in result["reasons"])
        )

    def test_fractional_integer_metric_fails_closed(self) -> None:
        value = json.loads(self.calibration_assessment.read_text(encoding="utf-8"))
        value["state_count"] = 120.9
        tampered = self.directory / "fractional_count_assessment.json"
        write_json(tampered, value)
        result = self.run_finalizer("fractional_count", assessment=tampered)
        self.assertEqual(result["status"], "assessment-fail")
        self.assertTrue(any("not an exact integer" in reason for reason in result["reasons"]))

    def test_forbidden_claim_field_fails_closed(self) -> None:
        value = json.loads(self.calibration_assessment.read_text(encoding="utf-8"))
        value["development_authorized"] = True
        tampered = self.directory / "forbidden_claim_assessment.json"
        write_json(tampered, value)
        result = self.run_finalizer("forbidden_claim", assessment=tampered)
        self.assertEqual(result["status"], "assessment-fail")
        self.assertTrue(any("forbidden fields" in reason for reason in result["reasons"]))

    def test_calibration_training_seed_field_fails_closed(self) -> None:
        table_value = json.loads(self.calibration_table.read_text(encoding="utf-8"))
        table_value["training_seeds"] = [999999]
        table_value["metadata"]["training_seeds"] = [999999]
        table_path = self.directory / "bad_role_calibration.json"
        write_json(table_path, table_value)
        manifest_value = json.loads(self.calibration_manifest.read_text(encoding="utf-8"))
        manifest_value["table_path"] = str(table_path.resolve())
        manifest_value["table_sha256"] = sha256(table_path)
        manifest_path = self.directory / "bad_role_calibration_manifest.json"
        write_json(manifest_path, manifest_value)
        result = self.run_finalizer(
            "bad_calibration_role",
            calibration_table=table_path,
            calibration_manifest=manifest_path,
        )
        self.assertEqual(result["status"], "assessment-fail")
        self.assertTrue(any("training-seed field mismatch" in reason for reason in result["reasons"]))

    def test_production_shaped_backend_requires_exact_openvino_build(self) -> None:
        backend = {
            "old_openvino_weight_npz_sha256": "2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba",
            "old_checkpoint_sha256": "7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1",
            "openvino_version": "2026.2.1",
            "openvino_full_build": "2026.2.1-21919-ede283a88e3-releases/2026/2",
            "complete_device": "NPU",
            "tail_device": "CPU",
            "complete_batch_size": 16,
            "evaluator_source_sha256": "a" * 64,
        }
        finalizer._validate_backend_evidence(
            backend, synthetic=False, label="production-shaped fixture"
        )
        short = dict(backend)
        short["openvino_full_build"] = "2026.2.1"
        with self.assertRaisesRegex(ValueError, "openvino_full_build mismatch"):
            finalizer._validate_backend_evidence(
                short, synthetic=False, label="short-version fixture"
            )

    def test_production_shaped_dependency_graph_binds_exact_live_closure(self) -> None:
        graph = finalizer._live_engine_dependency_graph(REPOSITORY)
        self.assertEqual(
            finalizer._validate_engine_dependency_graph(
                graph, synthetic=False, label="production-shaped fixture"
            ),
            graph,
        )
        changed = json.loads(json.dumps(graph))
        changed["records"][-1]["sha256"] = "0" * 64
        changed["graph_sha256"] = finalizer._dependency_digest(changed["records"])
        changed["runtime_closure_sha256"] = finalizer._dependency_digest(
            changed["records"][1:]
        )
        with self.assertRaisesRegex(ValueError, "frozen runtime dependency mismatch"):
            finalizer._validate_engine_dependency_graph(
                changed, synthetic=False, label="tampered production fixture"
            )

    def test_missing_input_fails_closed_with_durable_output(self) -> None:
        missing = self.directory / "missing_training.json"
        result = self.run_finalizer("missing", table=missing)
        self.assertEqual(result["status"], "assessment-fail")
        self.assertTrue(any("missing training_table" in reason for reason in result["reasons"]))


if __name__ == "__main__":
    unittest.main()
