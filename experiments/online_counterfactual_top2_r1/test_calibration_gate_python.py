from __future__ import annotations

import dataclasses
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
from unittest import mock

import calibration_gate as gate
import contract
import make_eligibility


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
FROZEN_PYTHON = Path(r"D:\tetris-paper-plus\python-env\Scripts\python.exe")
CONCRETE_JULIA = Path(
    r"C:\Users\fshuu\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collection_table(role: str) -> dict[str, Any]:
    episodes = gate.TRAINING_EPISODES if role == "training" else gate.CALIBRATION_EPISODES
    rows: list[dict[str, Any]] = []
    for episode_slot, episode in enumerate(episodes, 1):
        # The minimum valid cardinality is intentional: 20 unique frozen sample
        # positions per episode yields 240 training or 120 calibration states.
        for state_slot, piece in enumerate(gate.SAMPLE_PIECES[:20], 1):
            override = role == "calibration" and state_slot <= 2
            training_positive = role == "training" and state_slot == 1
            advantage = (
                0.8 + 0.01 * episode_slot
                if override or training_positive
                else -0.2
            )
            features = [0.0] * gate.EXPECTED_FEATURE_COUNT
            features[0] = 2.0 if override else 0.0
            features[6] = 3.0
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
                    "canonical_top1_candidate_index": 1,
                    "canonical_top2_candidate_index": 2,
                    "canonical_top1_action_digest": f"a1-{episode}-{piece}",
                    "canonical_top2_action_digest": f"a2-{episode}-{piece}",
                    "valid_action_count": 3,
                }
            )
    feature_names = contract.load_contract()["feature_schema"]["names"]
    stable_digest = "synthetic-no-engine"
    return {
        "schema_version": f"r1-{role}-table-v1",
        "source_role": role,
        "metadata": {
            "source_role": role,
            "feature_names": feature_names,
            "feature_schema_digest": gate.EXPECTED_FEATURE_NAMES_SHA256,
            "training_seeds": list(episodes) if role == "training" else [],
            "role_seeds": list(episodes),
            "sample_pieces": list(gate.SAMPLE_PIECES),
            "synthetic": True,
            "stable_node_key_source_sha256": stable_digest,
            "engine_dependency_graph": None,
            "counterfactual_states_completed": len(episodes) * len(gate.SAMPLE_PIECES),
            "validation_seed_used": False,
            "sealed_test_seed_used": False,
        },
        "feature_names": feature_names,
        "feature_schema_digest": gate.EXPECTED_FEATURE_NAMES_SHA256,
        "training_seeds": list(episodes) if role == "training" else [],
        "synthetic": True,
        "validation_seed_used": False,
        "sealed_test_seed_used": False,
        "rows": rows,
    }


def collection_manifest(table_path: Path, table: dict[str, Any]) -> dict[str, Any]:
    role = str(table["source_role"])
    episodes = gate.TRAINING_EPISODES if role == "training" else gate.CALIBRATION_EPISODES
    retained = {(int(row["episode_id"]), int(row["piece_index"])) for row in table["rows"]}
    exclusions = [
        {
            "seed": episode,
            "episode_id": episode,
            "piece_index": piece,
            "code": "synthetic_exclusion",
            "detail": "synthetic minimum-cardinality fixture",
            "root_state_digest": f"excluded-{episode}-{piece}",
        }
        for episode in episodes
        for piece in gate.SAMPLE_PIECES
        if (episode, piece) not in retained
    ]
    episode_evidence = [
        {
            "seed": episode,
            "episode_id": episode,
            "canonical_pieces": 240,
            "canonical_score": 0,
            "canonical_terminal": False,
            "canonical_action_digests": [],
            "rows": sum(int(row["episode_id"]) == episode for row in table["rows"]),
            "exclusions": sum(int(item["episode_id"]) == episode for item in exclusions),
        }
        for episode in episodes
    ]
    return {
        "schema_version": "r1-collection-manifest-v1",
        "source_role": role,
        "synthetic": True,
        "table_path": str(table_path.resolve()),
        "table_sha256": sha256(table_path),
        "feature_names": table["feature_names"],
        "feature_schema_digest": gate.EXPECTED_FEATURE_NAMES_SHA256,
        "stable_node_key_source_sha256": table["metadata"]["stable_node_key_source_sha256"],
        "engine_dependency_graph": table["metadata"]["engine_dependency_graph"],
        "episode_count": len(episodes),
        "row_count": len(table["rows"]),
        "exclusion_count": len(exclusions),
        "role_seeds": list(episodes),
        "counterfactual_states_completed": len(episodes) * len(gate.SAMPLE_PIECES),
        "exclusions": exclusions,
        "episodes": episode_evidence,
        "real_model_or_game_loaded": False,
        "validation_seed_used": False,
        "sealed_test_seed_used": False,
    }


def ridge_artifact(
    training_table_path: Path,
    training_manifest_path: Path,
    freeze_path: Path,
) -> dict[str, Any]:
    names = contract.load_contract()["feature_schema"]["names"]
    coefficients = [[0.0] * 256 for _ in range(71)]
    coefficients[0] = [-1.0] * 256
    coefficients[1] = [1.0] * 256
    return {
        "schema_version": "r1-ridge-gate-v1",
        "feature_names": names,
        "feature_schema_digest": gate.EXPECTED_FEATURE_NAMES_SHA256,
        "feature_mean": [0.0] * 70,
        "feature_scale": [1.0] * 70,
        "constant_feature": [False] * 70,
        "coefficient_shape": [71, 256],
        "coefficients": coefficients,
        "lambda": 1.0,
        "bootstrap_rng": "Xoshiro(0x5231_2026)",
        "bootstrap_seed": 0x5231_2026,
        "lower_quantile": 0.1,
        "quantile_method": gate.EXPECTED_QUANTILE_METHOD,
        "override_threshold": 0.05,
        "target_clamp": [-2.0, 2.0],
        "ensemble_size": 256,
        "experiment_id": "online_counterfactual_top2_R1",
        "fit_role": "training_only",
        "fit_backend": "python_numpy_analytic_ridge",
        "runtime_facts": {
            "python_version": gate.EXPECTED_PYTHON_VERSION,
            "numpy_version": gate.EXPECTED_NUMPY_VERSION,
            "thread_environment": {
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        },
        "source_table_path": str(training_table_path.resolve()),
        "source_table_sha256": sha256(training_table_path),
        "source_table_synthetic": True,
        "source_collection_manifest_path": str(training_manifest_path.resolve()),
        "source_collection_manifest_sha256": sha256(training_manifest_path),
        "design_freeze_path": str(freeze_path.resolve()),
        "design_freeze_sha256": sha256(freeze_path),
        "training_bootstrap_schedule_sha256": gate.EXPECTED_TRAINING_SCHEDULE_SHA256,
        "training_bootstrap_schedule_source_anchor_sha256": gate.EXPECTED_TRAINING_SCHEDULE_SHA256,
        "training_bootstrap_schedule_consumed": True,
        "engine_dependency_graph": None,
        "training_stats": {
            "row_count": 240,
            "episode_count": 12,
            "episode_ids": list(gate.TRAINING_EPISODES),
            "positive_fraction_unclipped": 0.05,
        },
        "all_finite": True,
        "validation_seed_used": False,
        "sealed_test_seed_used": False,
    }


def julia_executable() -> Path:
    if CONCRETE_JULIA.is_file():
        return CONCRETE_JULIA
    found = shutil.which("julia")
    if found is None:
        raise unittest.SkipTest("Julia executable is unavailable")
    return Path(found)


class PythonCalibrationGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="r1-python-calibration-")
        self.directory = Path(self.temporary.name)
        self.freeze = self.directory / "freeze.json"
        eligibility = self.directory / "eligibility.json"
        contract.atomic_write_json(eligibility, make_eligibility.eligibility_document())
        result = subprocess.run(
            [
                str(julia_executable()),
                "--startup-file=no",
                "--history-file=no",
                f"--project={REPOSITORY}",
                str(ROOT / "freeze_design.jl"),
                str(eligibility),
                str(self.freeze),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        self.training_table = self.directory / "training_table.json"
        self.training_manifest = self.directory / "training_manifest.json"
        training = collection_table("training")
        write_json(self.training_table, training)
        write_json(
            self.training_manifest,
            collection_manifest(self.training_table, training),
        )
        self.calibration_table = self.directory / "calibration_table.json"
        self.calibration_manifest = self.directory / "calibration_manifest.json"
        calibration = collection_table("calibration")
        write_json(self.calibration_table, calibration)
        write_json(
            self.calibration_manifest,
            collection_manifest(self.calibration_table, calibration),
        )
        self.artifact = self.directory / "ridge.json"
        write_json(
            self.artifact,
            ridge_artifact(self.training_table, self.training_manifest, self.freeze),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def handoff_environment(
        self,
        *,
        artifact: Path | None = None,
        artifact_sha256: str | None = None,
    ) -> dict[str, str]:
        selected_artifact = artifact or self.artifact
        environment = os.environ.copy()
        environment.update(
            {
                "R1_EXPECTED_CALIBRATION_TABLE_SHA256": sha256(self.calibration_table),
                "R1_EXPECTED_CALIBRATION_MANIFEST_SHA256": sha256(self.calibration_manifest),
                "R1_EXPECTED_RIDGE_ARTIFACT_SHA256": artifact_sha256
                or sha256(selected_artifact),
                "R1_EXPECTED_DESIGN_FREEZE_SHA256": sha256(self.freeze),
            }
        )
        return environment

    def test_cli_production_argv_and_provenance(self) -> None:
        output = self.directory / "result.json"
        milestones = self.directory / "milestones.jsonl"
        result = subprocess.run(
            [
                str(FROZEN_PYTHON),
                str(ROOT / "calibration_gate.py"),
                str(self.calibration_table),
                str(self.calibration_manifest),
                str(self.artifact),
                str(self.freeze),
                str(output),
                str(milestones),
                "--synthetic",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=self.handoff_environment(),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(value["status"], "R1-calibration-promoted")
        self.assertEqual(value["override_count"], 12)
        self.assertEqual(value["override_episode_count"], 6)
        self.assertEqual(value["bootstrap"]["schedule_sha256"], gate.EXPECTED_CALIBRATION_SCHEDULE_SHA256)
        self.assertTrue(value["checks"]["fallback_top1_exact"])
        self.assertTrue(value["checks"]["production_reference_exact"])
        self.assertLessEqual(
            value["runtime_evidence"]["production_reference_max_abs_error"], 1.0e-10
        )
        self.assertEqual(
            set(value["provenance"]["blas_thread_environment"].values()), {"1"}
        )
        stages = [json.loads(line)["stage"] for line in milestones.read_text().splitlines()]
        self.assertEqual(stages[0], "script_enter")
        self.assertEqual(stages[-1], "phase_complete")

    def test_exact_python_fit_artifact_schema_is_consumed(self) -> None:
        fit_table = self.directory / "fit_generated_training_table.json"
        fit_manifest = self.directory / "fit_generated_training_manifest.json"
        fit_artifact = self.directory / "fit_generated_ridge.json"
        fit_milestones = self.directory / "fit_generated_milestones.jsonl"
        fit = subprocess.run(
            [
                str(FROZEN_PYTHON),
                str(ROOT / "fit_ridge.py"),
                str(fit_table),
                str(fit_manifest),
                str(self.freeze),
                str(fit_artifact),
                str(fit_milestones),
                "--synthetic",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(fit.returncode, 0, fit.stderr)
        output = self.directory / "exact_fit_calibration_result.json"
        milestones = self.directory / "exact_fit_calibration_milestones.jsonl"
        calibration = subprocess.run(
            [
                str(FROZEN_PYTHON),
                str(ROOT / "calibration_gate.py"),
                str(self.calibration_table),
                str(self.calibration_manifest),
                str(fit_artifact),
                str(self.freeze),
                str(output),
                str(milestones),
                "--synthetic",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=self.handoff_environment(artifact=fit_artifact),
        )
        self.assertEqual(calibration.returncode, 0, calibration.stderr)
        value = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(all(value["gate_artifact_evidence"].values()))
        self.assertTrue(value["checks"]["production_reference_exact"])
        self.assertTrue(value["checks"]["selected_action_matches_decision"])
        self.assertTrue(value["checks"]["fallback_top1_exact"])
        self.assertEqual(
            value["provenance"]["training_table"]["actual_sha256"], sha256(fit_table)
        )
        self.assertEqual(
            value["provenance"]["training_manifest"]["actual_sha256"],
            sha256(fit_manifest),
        )

    def test_between_phase_artifact_tamper_is_rejected_before_parse(self) -> None:
        tampered = self.directory / "tampered_ridge.json"
        shutil.copyfile(self.artifact, tampered)
        published_sha = sha256(tampered)
        with tampered.open("a", encoding="utf-8") as stream:
            stream.write(" \n")
        output = self.directory / "tampered_result.json"
        milestones = self.directory / "tampered_milestones.jsonl"
        process = subprocess.run(
            [
                str(FROZEN_PYTHON),
                str(ROOT / "calibration_gate.py"),
                str(self.calibration_table),
                str(self.calibration_manifest),
                str(tampered),
                str(self.freeze),
                str(output),
                str(milestones),
                "--synthetic",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=self.handoff_environment(artifact=tampered, artifact_sha256=published_sha),
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertFalse(output.exists())
        stages = [json.loads(line)["stage"] for line in milestones.read_text().splitlines()]
        self.assertEqual(stages[:3], ["script_enter", "imports_begin", "imports_end"])
        self.assertEqual(stages[-1], "phase_failed")

    def test_single_byte_snapshot_rejects_replacement_during_json_decode(self) -> None:
        source = self.directory / "snapshot_race.json"
        bytes_a = b'{"version":"A","payload":1}'
        bytes_b = b'{"version":"B","payload":2}'
        self.assertEqual(len(bytes_a), len(bytes_b))
        source.write_bytes(bytes_a)
        expected = hashlib.sha256(bytes_a).hexdigest()
        original_loads = gate.json.loads
        parsed_payloads: list[bytes] = []

        def replace_after_snapshot(payload: Any, *args: Any, **kwargs: Any) -> Any:
            self.assertIsInstance(payload, bytes)
            parsed_payloads.append(payload)
            try:
                source.write_bytes(bytes_b)
            except OSError as error:
                # A platform-level read lock is also a fail-closed outcome.
                raise gate.CalibrationError("replacement blocked during snapshot") from error
            return original_loads(payload, *args, **kwargs)

        with mock.patch.object(gate.json, "loads", side_effect=replace_after_snapshot):
            with self.assertRaisesRegex(
                gate.CalibrationError,
                "replacement blocked|changed or was replaced",
            ):
                gate.load_json_byte_snapshot(source, expected)

        # The decoder was given the exact immutable bytes that were hashed;
        # bytes B were never consumed as JSON under bytes A's digest.
        self.assertEqual(parsed_payloads, [bytes_a])
        self.assertEqual(source.read_bytes(), bytes_b)

    def test_run_path_detects_wrong_production_fallback_mapping(self) -> None:
        expected = {
            "calibration_table": sha256(self.calibration_table),
            "calibration_manifest": sha256(self.calibration_manifest),
            "ridge_artifact": sha256(self.artifact),
            "design_freeze": sha256(self.freeze),
        }

        def wrong_selector(features: list[float], **keywords: Any) -> dict[str, Any]:
            decision = gate.production_select_action(features, **keywords)
            if not decision["use_top2"]:
                decision = {
                    **decision,
                    "selected_candidate_index": keywords[
                        "canonical_top2_candidate_index"
                    ],
                    "selected_action_digest": keywords[
                        "canonical_top2_action_digest"
                    ],
                }
            return decision

        result = gate.run_calibration(
            self.calibration_table,
            self.calibration_manifest,
            self.artifact,
            self.freeze,
            synthetic=True,
            expected_handoff_sha256=expected,
            production_selector=wrong_selector,
        )
        self.assertEqual(result["status"], "R1-calibration-rejected")
        self.assertFalse(result["checks"]["selected_action_matches_decision"])
        self.assertFalse(result["checks"]["fallback_top1_exact"])
        self.assertGreater(result["fallback_top1_mismatch_count"], 0)

    def test_strict_threshold_and_candidate_index_bounds(self) -> None:
        coefficients = gate.np.zeros((71, 256), dtype=gate.np.float64)
        coefficients[0, :] = 0.05
        features = [0.0] * 70
        features[6] = 3.0
        decision = gate.production_select_action(
            features,
            valid_action_count=3,
            canonical_top1_candidate_index=1,
            canonical_top2_candidate_index=2,
            canonical_top1_action_digest="a1",
            canonical_top2_action_digest="a2",
            means=gate.np.zeros(70),
            scales=gate.np.ones(70),
            constant=gate.np.zeros(70, dtype=gate.np.bool_),
            coefficients=coefficients,
        )
        self.assertEqual(decision["lower_bound"], 0.05)
        self.assertFalse(decision["use_top2"])
        self.assertEqual(decision["selected_candidate_index"], 1)

        artifact = json.loads(self.artifact.read_text(encoding="utf-8"))
        raw = json.loads(self.calibration_table.read_text(encoding="utf-8"))["rows"]
        raw[0]["canonical_top2_candidate_index"] = 4
        with self.assertRaisesRegex(gate.CalibrationError, "exceeds valid action count"):
            gate.calibration_rows_from_table(
                raw,
                gate.np.asarray(artifact["feature_mean"]),
                gate.np.asarray(artifact["feature_scale"]),
                gate.np.asarray(artifact["constant_feature"], dtype=gate.np.bool_),
                gate.np.asarray(artifact["coefficients"]),
            )

    def test_json_integer_fields_reject_floats_and_booleans(self) -> None:
        one_digest = hashlib.sha256(b"1").hexdigest()
        for invalid in (1.5, True):
            with self.subTest(schedule_item=invalid), self.assertRaisesRegex(
                gate.CalibrationError, "exact JSON integer"
            ):
                gate._validate_schedule(
                    [[invalid]],
                    expected_count=1,
                    expected_width=1,
                    allowed_episodes=(1,),
                    expected_digest=one_digest,
                    label="integer witness",
                )

        table = json.loads(self.calibration_table.read_text(encoding="utf-8"))
        manifest = json.loads(self.calibration_manifest.read_text(encoding="utf-8"))
        table["rows"][0]["episode_id"] = 73101.5
        with self.assertRaisesRegex(gate.CalibrationError, "exact JSON integer"):
            gate._validate_collection_table(
                table,
                manifest,
                role="calibration",
                expected_episodes=gate.CALIBRATION_EPISODES,
                minimum_rows=120,
                maximum_rows=144,
                synthetic=True,
                table_path=self.calibration_table,
                table_actual_sha256=sha256(self.calibration_table),
            )

        artifact = json.loads(self.artifact.read_text(encoding="utf-8"))
        for field, invalid in (
            ("canonical_top2_candidate_index", 2.5),
            ("valid_action_count", True),
        ):
            raw = json.loads(self.calibration_table.read_text(encoding="utf-8"))["rows"]
            raw[0][field] = invalid
            if field == "valid_action_count":
                raw[0]["features"][6] = 1.0
            with self.subTest(field=field), self.assertRaisesRegex(
                gate.CalibrationError, "exact JSON integer"
            ):
                gate.calibration_rows_from_table(
                    raw,
                    gate.np.asarray(artifact["feature_mean"]),
                    gate.np.asarray(artifact["feature_scale"]),
                    gate.np.asarray(artifact["constant_feature"], dtype=gate.np.bool_),
                    gate.np.asarray(artifact["coefficients"]),
                )

    def test_training_seed_fields_are_exactly_role_scoped(self) -> None:
        calibration_manifest = json.loads(
            self.calibration_manifest.read_text(encoding="utf-8")
        )
        for location in ("metadata", "top_level"):
            table = json.loads(self.calibration_table.read_text(encoding="utf-8"))
            if location == "metadata":
                table["metadata"]["training_seeds"] = [999999]
            else:
                table["training_seeds"] = [999999]
            with self.subTest(calibration_location=location), self.assertRaisesRegex(
                gate.CalibrationError, "training seed role mismatch"
            ):
                gate._validate_collection_table(
                    table,
                    calibration_manifest,
                    role="calibration",
                    expected_episodes=gate.CALIBRATION_EPISODES,
                    minimum_rows=120,
                    maximum_rows=144,
                    synthetic=True,
                    table_path=self.calibration_table,
                    table_actual_sha256=sha256(self.calibration_table),
                )

        training = json.loads(self.training_table.read_text(encoding="utf-8"))
        training_manifest = json.loads(
            self.training_manifest.read_text(encoding="utf-8")
        )
        training["metadata"]["training_seeds"] = list(gate.TRAINING_EPISODES[:-1])
        with self.assertRaisesRegex(gate.CalibrationError, "training seed role mismatch"):
            gate._validate_collection_table(
                training,
                training_manifest,
                role="training",
                expected_episodes=gate.TRAINING_EPISODES,
                minimum_rows=240,
                maximum_rows=288,
                synthetic=True,
                table_path=self.training_table,
                table_actual_sha256=sha256(self.training_table),
            )

    def test_engine_dependency_graph_tamper_and_ridge_propagation_are_rejected(self) -> None:
        table = json.loads(self.calibration_table.read_text(encoding="utf-8"))
        manifest = json.loads(self.calibration_manifest.read_text(encoding="utf-8"))
        table["metadata"]["engine_dependency_graph"] = {}
        with self.assertRaisesRegex(
            gate.CalibrationError,
            "engine dependency graph differs between table and manifest",
        ):
            gate._validate_collection_table(
                table,
                manifest,
                role="calibration",
                expected_episodes=gate.CALIBRATION_EPISODES,
                minimum_rows=120,
                maximum_rows=144,
                synthetic=True,
                table_path=self.calibration_table,
                table_actual_sha256=sha256(self.calibration_table),
            )

        tampered_artifact = self.directory / "ridge_graph_tampered.json"
        artifact = json.loads(self.artifact.read_text(encoding="utf-8"))
        artifact["engine_dependency_graph"] = {}
        write_json(tampered_artifact, artifact)
        expected = {
            "calibration_table": sha256(self.calibration_table),
            "calibration_manifest": sha256(self.calibration_manifest),
            "ridge_artifact": sha256(tampered_artifact),
            "design_freeze": sha256(self.freeze),
        }
        with self.assertRaisesRegex(
            gate.CalibrationError,
            "ridge artifact synthetic engine dependency graph must be null",
        ):
            gate.run_calibration(
                self.calibration_table,
                self.calibration_manifest,
                tampered_artifact,
                self.freeze,
                synthetic=True,
                expected_handoff_sha256=expected,
            )

    def _deterministic_rows_and_evidence(self) -> tuple[list[gate.CalibrationRow], gate.GateArtifactEvidence]:
        artifact = json.loads(self.artifact.read_text(encoding="utf-8"))
        raw = json.loads(self.calibration_table.read_text(encoding="utf-8"))["rows"]
        coefficients = gate.np.asarray(artifact["coefficients"], dtype=gate.np.float64)
        rows, _ = gate.calibration_rows_from_table(
            raw,
            gate.np.asarray(artifact["feature_mean"], dtype=gate.np.float64),
            gate.np.asarray(artifact["feature_scale"], dtype=gate.np.float64),
            gate.np.asarray(artifact["constant_feature"], dtype=gate.np.bool_),
            coefficients,
        )
        deterministic = [dataclasses.replace(row, overhead_ms=0.01) for row in rows]
        evidence = gate.GateArtifactEvidence(True, True, True, True, 70 + 70 + 71 * 256)
        return deterministic, evidence

    def _julia_result(self, rows: list[gate.CalibrationRow], suffix: str) -> dict[str, Any]:
        freeze = json.loads(self.freeze.read_text(encoding="utf-8"))
        artifact = json.loads(self.artifact.read_text(encoding="utf-8"))
        document = {
            "source_role": "calibration",
            "rows": [
                {
                    "episode_id": row.episode_id,
                    "advantage_unclipped_A6": row.advantage,
                    "a1_terminal_within_horizon": row.a1_terminal,
                    "a2_terminal_within_horizon": row.a2_terminal,
                    "production_decision": row.production_decision,
                    "reference_decision": row.reference_decision,
                    "production_selected_candidate_index": row.production_selected_candidate_index,
                    "canonical_top1_candidate_index": row.canonical_top1_candidate_index,
                    "canonical_top2_candidate_index": row.canonical_top2_candidate_index,
                    "production_selected_action_digest": row.production_selected_action_digest,
                    "canonical_top1_action_digest": row.canonical_top1_action_digest,
                    "canonical_top2_action_digest": row.canonical_top2_action_digest,
                    "decision_overhead_ms": row.overhead_ms,
                }
                for row in rows
            ],
            # The old Julia evaluator's evidence helper predates the explicit
            # 71x256 JSON orientation and consumes a flat coefficient vector.
            "gate_artifact": {
                "feature_names": artifact["feature_names"],
                "feature_mean": artifact["feature_mean"],
                "feature_scale": artifact["feature_scale"],
                "constant_feature": artifact["constant_feature"],
                "coefficients": [
                    value for coefficient_row in artifact["coefficients"] for value in coefficient_row
                ],
                "coefficient_count": 71,
                "ensemble_size": 256,
                "lambda": 1.0,
                "bootstrap_seed": 0x5231_2026,
                "lower_quantile": 0.1,
                "override_threshold": 0.05,
            },
            "calibration_bootstrap_schedules": freeze["calibration_bootstrap_schedules"],
            "calibration_bootstrap_schedule_sha256": gate.EXPECTED_CALIBRATION_SCHEDULE_SHA256,
            "forbidden_seed_used": False,
        }
        source = self.directory / f"julia_input_{suffix}.json"
        output = self.directory / f"julia_output_{suffix}.json"
        milestones = self.directory / f"julia_milestones_{suffix}.jsonl"
        write_json(source, document)
        environment = os.environ.copy()
        environment["R1_CHILD_MILESTONE_PATH"] = str(milestones)
        process = subprocess.run(
            [
                str(julia_executable()),
                "--startup-file=no",
                "--history-file=no",
                f"--project={REPOSITORY}",
                str(ROOT / "calibration_gate.jl"),
                str(source),
                str(output),
                sha256(source),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        return json.loads(output.read_text(encoding="utf-8"))

    def _python_result(self, rows: list[gate.CalibrationRow]) -> dict[str, Any]:
        freeze = json.loads(self.freeze.read_text(encoding="utf-8"))
        return gate.evaluate_calibration(
            rows,
            artifact_evidence=gate.GateArtifactEvidence(
                True, True, True, True, 70 + 70 + 71 * 256
            ),
            calibration_source_evidence=gate.SourceHashEvidence(
                str(self.calibration_table), sha256(self.calibration_table), sha256(self.calibration_table), True
            ),
            bootstrap_schedules=freeze["calibration_bootstrap_schedules"],
            engine_dependency_graph=None,
        )

    def assert_cross_language_equal(self, python: dict[str, Any], julia: dict[str, Any]) -> None:
        exact_keys = (
            "status",
            "promoted",
            "state_count",
            "calibration_episodes",
            "override_count",
            "override_episode_count",
            "override_episodes",
            "unsafe_top2_terminal_count",
            "production_reference_mismatch_count",
            "selected_action_mismatch_count",
            "fallback_top1_mismatch_count",
            "checks",
        )
        for key in exact_keys:
            self.assertEqual(python[key], julia[key], key)
        float_keys = (
            "override_rate",
            "override_precision",
            "override_mean_unclipped_A6",
            "median_decision_overhead_ms",
        )
        for key in float_keys:
            self.assertAlmostEqual(python[key], julia[key], places=13, msg=key)
        for key in ("precision_lower90", "mean_advantage_lower90"):
            self.assertAlmostEqual(
                python["bootstrap"][key], julia["bootstrap"][key], places=13, msg=key
            )
        self.assertEqual(
            python["bootstrap"]["schedule_sha256"],
            julia["bootstrap"]["schedule_sha256"],
        )

    def test_cross_language_gate_and_wrong_fallback_rejection(self) -> None:
        rows, _ = self._deterministic_rows_and_evidence()
        python_good = self._python_result(rows)
        julia_good = self._julia_result(rows, "good")
        self.assert_cross_language_equal(python_good, julia_good)
        self.assertTrue(python_good["promoted"])

        fallback = next(index for index, row in enumerate(rows) if not row.production_decision)
        wrong = list(rows)
        wrong[fallback] = dataclasses.replace(
            wrong[fallback],
            production_selected_candidate_index=wrong[fallback].canonical_top2_candidate_index,
            production_selected_action_digest=wrong[fallback].canonical_top2_action_digest,
        )
        python_bad = self._python_result(wrong)
        julia_bad = self._julia_result(wrong, "wrong_fallback")
        self.assert_cross_language_equal(python_bad, julia_bad)
        self.assertFalse(python_bad["promoted"])
        self.assertFalse(python_bad["checks"]["fallback_top1_exact"])
        self.assertEqual(python_bad["fallback_top1_mismatch_count"], 1)


if __name__ == "__main__":
    unittest.main()
