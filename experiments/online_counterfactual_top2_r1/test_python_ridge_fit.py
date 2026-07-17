from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path: Path):
    specification = importlib.util.spec_from_file_location("r1_fit_ridge", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--julia", required=True, type=Path)
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parent
    repository = experiment.parent.parent
    fit_script = experiment / "fit_ridge.py"
    fit = load_module(fit_script)
    np = fit._import_numpy()
    assert np.__version__ == "2.4.6"
    assert fit.linear_type7(np.arange(256, dtype=np.float64), 0.10) == 25.5
    assert all(os.environ[name] == "1" for name in (
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"
    ))
    runtime_binding = fit._contract_document()["runtime_source_binding"]
    live_graph = fit._expected_engine_dependency_graph(runtime_binding)
    assert len(live_graph["records"]) == 7
    assert fit._validate_engine_dependency_graph(
        live_graph,
        synthetic=False,
        runtime_binding=runtime_binding,
        label="test live",
    ) is live_graph
    assert fit._validate_engine_dependency_graph(
        None,
        synthetic=True,
        runtime_binding=runtime_binding,
        label="test synthetic",
    ) is None
    bad_graph = copy.deepcopy(live_graph)
    bad_graph["records"][0]["sha256"] = "0" * 64
    try:
        fit._validate_engine_dependency_graph(
            bad_graph,
            synthetic=False,
            runtime_binding=runtime_binding,
            label="test tampered",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("tampered production engine dependency graph accepted")
    try:
        fit._validate_engine_dependency_graph(
            live_graph,
            synthetic=True,
            runtime_binding=runtime_binding,
            label="test synthetic tamper",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("non-null synthetic engine dependency graph accepted")

    with tempfile.TemporaryDirectory(prefix="r1-python-ridge-conformance-") as temporary:
        root = Path(temporary)
        eligibility = root / "eligibility.json"
        freeze = root / "freeze.json"
        table = root / "table.json"
        manifest = root / "manifest.json"
        artifact = root / "artifact.json"
        milestones = root / "milestones.jsonl"
        reference = root / "julia_reference.json"
        subprocess.run(
            [str(args.python), str(experiment / "make_eligibility.py"), str(eligibility)],
            check=True,
            cwd=repository,
        )
        subprocess.run(
            [
                str(args.julia), "--startup-file=no", "--history-file=no",
                f"--project={repository}", str(experiment / "freeze_design.jl"),
                str(eligibility), str(freeze),
            ],
            check=True,
            cwd=repository,
        )
        python_command = [
            str(args.python), str(fit_script), str(table), str(manifest), str(freeze),
            str(artifact), str(milestones), "--synthetic",
        ]
        subprocess.run(python_command, check=True, cwd=repository)
        subprocess.run(
            [
                str(args.julia), "--startup-file=no", "--history-file=no",
                f"--project={repository}",
                str(experiment / "test_python_ridge_reference.jl"),
                str(table), str(freeze), str(artifact), str(reference),
            ],
            check=True,
            cwd=repository,
        )
        python_artifact = json.loads(artifact.read_text(encoding="utf-8"))
        fit._load_training_table(
            table, manifest, True, digest(table), digest(manifest)
        )
        fit._load_design_freeze(
            freeze, fit._contract_feature_names(), digest(freeze)
        )
        for call in (
            lambda: fit._load_training_table(table, manifest, True, "0" * 64, digest(manifest)),
            lambda: fit._load_training_table(table, manifest, True, digest(table), "0" * 64),
            lambda: fit._load_design_freeze(freeze, fit._contract_feature_names(), "0" * 64),
        ):
            try:
                call()
            except ValueError:
                pass
            else:
                raise AssertionError("producer/frozen SHA handoff mismatch was accepted")
        julia = json.loads(reference.read_text(encoding="utf-8"))
        julia_gate = julia["gate"]
        assert python_artifact["coefficient_shape"] == [71, 256]
        assert len(python_artifact["coefficients"]) == 71
        assert all(len(row) == 256 for row in python_artifact["coefficients"])
        means_difference = np.max(np.abs(
            np.asarray(python_artifact["feature_mean"]) - np.asarray(julia_gate["feature_mean"])
        ))
        scales_difference = np.max(np.abs(
            np.asarray(python_artifact["feature_scale"]) - np.asarray(julia_gate["feature_scale"])
        ))
        coefficient_difference = np.max(np.abs(
            np.asarray(python_artifact["coefficients"]) - np.asarray(julia_gate["coefficients"])
        ))
        assert python_artifact["constant_feature"] == julia_gate["constant_feature"]
        assert means_difference <= 5e-15, means_difference
        assert scales_difference <= 5e-15, scales_difference
        assert coefficient_difference <= 2e-10, coefficient_difference

        gate = fit.load_gate_payload(python_artifact)
        table_document = json.loads(table.read_text(encoding="utf-8"))
        probes = np.asarray([row["features"] for row in table_document["rows"][:12]])
        python_lower = fit.predict_lower_bounds(gate, probes)
        julia_lower = np.asarray(julia["lower_bounds"])
        lower_difference = np.max(np.abs(python_lower - julia_lower))
        assert lower_difference <= 2e-10, lower_difference
        python_decisions = [
            fit.decide_top2(gate, row, int(row[6])) for row in probes
        ]
        assert [value["use_top2"] for value in python_decisions] == [
            value["use_top2"] for value in julia["decisions"]
        ]
        assert [value["fallback_reason"] for value in python_decisions] == [
            value["fallback_reason"] for value in julia["decisions"]
        ]
        assert julia["python_artifact_loaded_by_julia"] is True
        artifact_load_difference = np.max(np.abs(
            python_lower - np.asarray(julia["python_artifact_lower_bounds"])
        ))
        assert artifact_load_difference <= 2e-10, artifact_load_difference
        assert [value["use_top2"] for value in python_decisions] == [
            value["use_top2"] for value in julia["python_artifact_decisions"]
        ]
        mismatch = fit.decide_top2(gate, probes[0], int(probes[0, 6]) + 1)
        assert mismatch["fallback_reason"] == "candidate_count_mismatch"
        huge = np.full(70, np.finfo(np.float64).max)
        huge[6] = probes[0, 6]
        overflow = fit.decide_top2(gate, huge, int(huge[6]))
        assert overflow["use_top2"] is False
        assert overflow["fallback_reason"] == "invalid_prediction"
        for field, replacement in (
            ("lambda", 123.0),
            ("lambda", True),
            ("lower_quantile", 0.9),
            ("override_threshold", -999.0),
            ("target_clamp", [-99.0, 99.0]),
            ("ensemble_size", 255),
            ("ensemble_size", 256.0),
            ("bootstrap_seed", float(fit.BOOTSTRAP_SEED)),
            ("coefficient_shape", [71.0, 256.0]),
        ):
            mutated_gate = dict(python_artifact)
            mutated_gate[field] = replacement
            try:
                fit.load_gate_payload(mutated_gate)
            except ValueError:
                pass
            else:
                raise AssertionError(f"mutated gate field accepted: {field}")

        events = [json.loads(line) for line in milestones.read_text(encoding="utf-8").splitlines()]
        stages = [event["stage"] for event in events]
        assert stages[:3] == ["script_enter", "args_verified", "synthetic_inputs_generated"]
        assert stages.index("imports_begin") < stages.index("numpy_import_end")
        assert stages[-2:] == ["artifact_published", "phase_complete"]
        assert python_artifact["training_bootstrap_schedule_sha256"] == fit.TRAINING_SCHEDULE_SHA256
        assert python_artifact["source_table_sha256"] == digest(table)
        assert python_artifact["source_collection_manifest_sha256"] == digest(manifest)
        assert python_artifact["engine_dependency_graph"] is None
        assert json.loads(table.read_text(encoding="utf-8"))["metadata"][
            "engine_dependency_graph"
        ] is None
        assert json.loads(manifest.read_text(encoding="utf-8"))[
            "engine_dependency_graph"
        ] is None
        assert python_artifact["runtime_facts"]["numpy_version"] == "2.4.6"

        # Publication is no-overwrite: an exact rerun fails without changing
        # any of the five already-created artifacts.
        before = tuple(digest(path) for path in (table, manifest, artifact, milestones))
        collision = subprocess.run(
            python_command, check=False, cwd=repository, capture_output=True, text=True
        )
        assert collision.returncode != 0
        assert before == tuple(digest(path) for path in (table, manifest, artifact, milestones))

        # The manifest/table binding and feature-7 candidate-count binding are
        # both fail-closed in a clean subprocess-style loader fixture.
        bad_table = root / "bad_table.json"
        bad_manifest = root / "bad_manifest.json"
        bad_document = json.loads(table.read_text(encoding="utf-8"))
        bad_document["rows"][0]["features"][6] += 1.0
        bad_table.write_text(json.dumps(bad_document, separators=(",", ":")) + "\n", encoding="utf-8")
        shutil.copyfile(manifest, bad_manifest)
        bad_manifest_document = json.loads(bad_manifest.read_text(encoding="utf-8"))
        bad_manifest_document["table_path"] = str(bad_table)
        bad_manifest_document["table_sha256"] = digest(bad_table)
        bad_manifest.write_text(json.dumps(bad_manifest_document, separators=(",", ":")) + "\n", encoding="utf-8")
        validation_code = (
            "import importlib.util,sys;from pathlib import Path;"
            "s=importlib.util.spec_from_file_location('r1_fit',sys.argv[1]);"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "m._load_training_table(Path(sys.argv[2]),Path(sys.argv[3]),True)"
        )
        invalid = subprocess.run(
            [str(args.python), "-c", validation_code, str(fit_script),
             str(bad_table), str(bad_manifest)],
            check=False,
            cwd=repository,
            capture_output=True,
            text=True,
        )
        assert invalid.returncode != 0
        assert "candidate-count feature mismatch" in invalid.stderr

        float_table = root / "float_identity_table.json"
        float_manifest = root / "float_identity_manifest.json"
        float_document = json.loads(table.read_text(encoding="utf-8"))
        float_document["rows"][0]["episode_id"] = 73001.9
        float_document["rows"][0]["seed"] = 73001.9
        float_document["rows"][0]["piece_index"] = 10.9
        float_table.write_text(
            json.dumps(float_document, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        float_manifest_document = json.loads(manifest.read_text(encoding="utf-8"))
        float_manifest_document["table_path"] = str(float_table)
        float_manifest_document["table_sha256"] = digest(float_table)
        float_manifest.write_text(
            json.dumps(float_manifest_document, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        invalid_float = subprocess.run(
            [str(args.python), "-c", validation_code, str(fit_script),
             str(float_table), str(float_manifest)],
            check=False,
            cwd=repository,
            capture_output=True,
            text=True,
        )
        assert invalid_float.returncode != 0
        assert "exact JSON integer" in invalid_float.stderr

        partition_table = root / "bad_partition_table.json"
        partition_manifest = root / "bad_partition_manifest.json"
        partition_document = json.loads(table.read_text(encoding="utf-8"))
        partition_document["rows"].pop()
        partition_table.write_text(
            json.dumps(partition_document, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        partition_manifest_document = json.loads(manifest.read_text(encoding="utf-8"))
        partition_manifest_document["table_path"] = str(partition_table)
        partition_manifest_document["table_sha256"] = digest(partition_table)
        partition_manifest_document["row_count"] = 287
        partition_manifest_document["exclusion_count"] = 1
        partition_manifest_document["exclusions"] = [{
            "seed": 73012,
            "episode_id": 73012,
            # A structurally valid-looking but wrong key overlaps the retained
            # piece-230 row and leaves the removed piece-240 key uncovered.
            "piece_index": 230,
            "code": "candidate_count_lt2",
            "detail": "synthetic partition corruption witness",
            "root_state_digest": "f" * 64,
        }]
        partition_manifest_document["episodes"][-1]["rows"] = 23
        partition_manifest_document["episodes"][-1]["exclusions"] = 1
        partition_manifest.write_text(
            json.dumps(partition_manifest_document, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        invalid_partition = subprocess.run(
            [str(args.python), "-c", validation_code, str(fit_script),
             str(partition_table), str(partition_manifest)],
            check=False,
            cwd=repository,
            capture_output=True,
            text=True,
        )
        assert invalid_partition.returncode != 0
        assert "duplicate row/exclusion state" in invalid_partition.stderr

        sha_bad_manifest = root / "sha_bad_manifest.json"
        sha_bad_document = json.loads(manifest.read_text(encoding="utf-8"))
        sha_bad_document["table_sha256"] = "0" * 64
        sha_bad_manifest.write_text(
            json.dumps(sha_bad_document, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        invalid_sha = subprocess.run(
            [str(args.python), "-c", validation_code, str(fit_script),
             str(table), str(sha_bad_manifest)],
            check=False,
            cwd=repository,
            capture_output=True,
            text=True,
        )
        assert invalid_sha.returncode != 0
        assert "manifest/table SHA-256 mismatch" in invalid_sha.stderr

        bad_freeze = root / "bad_freeze.json"
        bad_freeze_document = json.loads(freeze.read_text(encoding="utf-8"))
        original_schedule_value = bad_freeze_document["training_bootstrap_schedules"][0][0]
        bad_freeze_document["training_bootstrap_schedules"][0][0] = (
            73001 if original_schedule_value != 73001 else 73002
        )
        bad_freeze_document["training_bootstrap_schedule_sha256"] = fit.schedule_digest(
            bad_freeze_document["training_bootstrap_schedules"]
        )
        bad_freeze.write_text(
            json.dumps(bad_freeze_document, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        freeze_validation_code = (
            "import importlib.util,sys;from pathlib import Path;"
            "s=importlib.util.spec_from_file_location('r1_fit',sys.argv[1]);"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "m._load_design_freeze(Path(sys.argv[2]),m._contract_feature_names())"
        )
        invalid_schedule = subprocess.run(
            [str(args.python), "-c", freeze_validation_code, str(fit_script),
             str(bad_freeze)],
            check=False,
            cwd=repository,
            capture_output=True,
            text=True,
        )
        assert invalid_schedule.returncode != 0
        assert "independent source anchor" in invalid_schedule.stderr

        print(json.dumps({
            "status": "r1_python_julia_ridge_conformance_passed",
            "max_mean_abs_error": float(means_difference),
            "max_scale_abs_error": float(scales_difference),
            "max_coefficient_abs_error": float(coefficient_difference),
            "max_lower_bound_abs_error": float(lower_difference),
            "max_julia_artifact_load_abs_error": float(artifact_load_difference),
            "decisions_checked": len(python_decisions),
        }, separators=(",", ":")))


if __name__ == "__main__":
    main()
