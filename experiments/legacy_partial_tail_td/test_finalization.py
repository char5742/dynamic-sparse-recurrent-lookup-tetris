from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from finalize_result import (
    ELIGIBLE_ROWS,
    REQUIRED_FALSE_FIELDS,
    REQUIRED_OMISSION_CLASS_FIELDS,
    assess,
)
from validate_source_fingerprint import (
    source_aggregate,
    source_records,
    validate_fingerprint_document,
    validate_repository_binding,
)


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def fingerprint_document(repository: Path) -> dict:
    records = source_records(repository)
    return {
        "repository_root": str(repository.resolve()),
        "source_sha256": source_aggregate(records),
        "manifest_sha256": hashlib.sha256((repository / "Manifest.toml").read_bytes()).hexdigest(),
        "file_count": len(records),
        "files": records,
    }


def test_source_fingerprint_negative_cases() -> None:
    with tempfile.TemporaryDirectory(prefix="p1-fingerprint-") as temporary:
        repository = Path(temporary)
        write(repository / "Manifest.toml", "manifest\n")
        write(repository / "Project.toml", "project\n")
        write(repository / "src" / "model.jl", "x = 1\n")
        valid = fingerprint_document(repository)
        assert validate_fingerprint_document(repository, valid)["valid"]

        for missing in (
            "repository_root",
            "files",
            "source_sha256",
            "manifest_sha256",
            "file_count",
        ):
            invalid = copy.deepcopy(valid)
            invalid.pop(missing)
            assert not validate_fingerprint_document(repository, invalid)["valid"]

        duplicate = copy.deepcopy(valid)
        duplicate["files"].append(copy.deepcopy(duplicate["files"][0]))
        duplicate["file_count"] += 1
        assert not validate_fingerprint_document(repository, duplicate)["valid"]

        for field, replacement in (("path", "wrong.jl"), ("bytes", -1), ("sha256", "0" * 64)):
            invalid = copy.deepcopy(valid)
            invalid["files"][0][field] = replacement
            assert not validate_fingerprint_document(repository, invalid)["valid"]

        wrong_root = copy.deepcopy(valid)
        wrong_root["repository_root"] = str(repository / "elsewhere")
        assert not validate_fingerprint_document(repository, wrong_root)["valid"]


def run_git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_repository_binding_negative_cases() -> None:
    with tempfile.TemporaryDirectory(prefix="p1-binding-") as temporary:
        repository = Path(temporary)
        run_git(repository, "init", "-q")
        run_git(repository, "config", "user.email", "p1@example.invalid")
        run_git(repository, "config", "user.name", "P1 Test")
        write(repository / "base.txt", "base\n")
        run_git(repository, "add", ".")
        run_git(repository, "commit", "-q", "-m", "base")
        parent = run_git(repository, "rev-parse", "HEAD")
        write(repository / "experiments" / "legacy_partial_tail_td" / "hardening.py", "x=1\n")
        run_git(repository, "add", ".")
        run_git(repository, "commit", "-q", "-m", "hardening")
        head = run_git(repository, "rev-parse", "HEAD")
        assert validate_repository_binding(repository, head, parent)["valid"]
        assert not validate_repository_binding(repository, "0" * 40, parent)["valid"]
        assert not validate_repository_binding(repository, head, "1" * 40)["valid"]
        write(repository / "dirty.txt", "dirty\n")
        assert not validate_repository_binding(repository, head, parent)["valid"]


def test_incomplete_assessment_stubs_reject() -> None:
    with tempfile.TemporaryDirectory(prefix="p1-assessment-") as temporary:
        root = Path(temporary)
        empty = assess(root)
        assert not empty["success"]
        assert len(empty["failures"]) >= 8

        flattering = {
            "freeze.json": {"experiment": "legacy_partial_tail_td_P1", "repository_clean": True},
            "eligibility.json": {
                "status": "training_eligibility_complete",
                "eligible_count": 1482,
                "eligible_rows": list(ELIGIBLE_ROWS[:-1]) + [1498],
            },
            "row_freeze.json": {
                "status": "training_row_freeze_complete",
                "row_count": 300,
                "rng": "Xoshiro(0x1313_2026)",
                "ordered_rows": list(ELIGIBLE_ROWS[:300]),
            },
            "training_phase.json": {"status": "training_phase_complete", "update_count": 300},
            "openvino_gate.json": {"status": "openvino_gate_pass"},
            "offline_gate.json": {"status": "offline_gate_pass", "row_count": 500},
            "development.json": {
                "status": "P1-development-pass",
                "seeds": [5757, 5756],
                "validation_or_test_seed_loaded": True,
            },
        }
        for name, value in flattering.items():
            (root / name).write_text(json.dumps(value), encoding="utf-8")
        rejected = assess(root)
        assert not rejected["success"]
        assert any("exact row list" in failure for failure in rejected["failures"])
        assert any("exact order" in failure for failure in rejected["failures"])
        assert any("forbidden" in failure for failure in rejected["failures"])
        assert not (root / "final_result.json").exists()


def test_targeted_omission_classes_reject() -> None:
    payloads = {
        "eligibility": {"rows_loaded": [1, 1500]},
        "row_freeze": {"ordered_rows": [], "rng": "Xoshiro(0x1313_2026)"},
        "training_extraction": {"npz_sha256": "0" * 64},
        "training_phase": {"freeze_path": "freeze.json"},
        "development": {"accounting": {}},
    }
    for artifact_name, fields in REQUIRED_FALSE_FIELDS.items():
        payloads.setdefault(artifact_name, {}).update({field: False for field in fields})

    with tempfile.TemporaryDirectory(prefix="p1-omissions-") as temporary:
        root = Path(temporary)
        for omission_class, (artifact_name, field) in REQUIRED_OMISSION_CLASS_FIELDS.items():
            path = root / f"{artifact_name}.json"
            payload = copy.deepcopy(payloads[artifact_name])
            path.write_text(json.dumps(payload), encoding="utf-8")
            expected = f"{artifact_name} missing required {omission_class} field {field}"
            assert expected not in assess(root)["failures"]
            payload.pop(field)
            path.write_text(json.dumps(payload), encoding="utf-8")
            assert expected in assess(root)["failures"]
            path.unlink()

        for artifact_name, fields in REQUIRED_FALSE_FIELDS.items():
            path = root / f"{artifact_name}.json"
            for field in fields:
                payload = copy.deepcopy(payloads[artifact_name])
                path.write_text(json.dumps(payload), encoding="utf-8")
                missing = f"{artifact_name} missing required forbidden/scope field {field}"
                assert missing not in assess(root)["failures"]
                payload.pop(field)
                path.write_text(json.dumps(payload), encoding="utf-8")
                assert missing in assess(root)["failures"]
                payload[field] = True
                path.write_text(json.dumps(payload), encoding="utf-8")
                assert (
                    f"{artifact_name} forbidden/scope field {field} is not exactly false"
                    in assess(root)["failures"]
                )
            path.unlink()


if __name__ == "__main__":
    test_source_fingerprint_negative_cases()
    test_repository_binding_negative_cases()
    test_incomplete_assessment_stubs_reject()
    test_targeted_omission_classes_reject()
    print("P1 finalization/source negative checks passed")
