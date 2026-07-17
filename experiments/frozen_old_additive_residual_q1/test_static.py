from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_python_sources_parse() -> None:
    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_help_paths_are_safe() -> None:
    for name, token in (
        ("extract_dataset.py", "eligibility"),
        ("verify_openvino.py", "output_directory"),
        ("offline_gate.py", "offline_npz"),
        ("finalize_result.py", "output_directory"),
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / name), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert token in result.stdout


def test_q1_scope_and_fresh_export_are_explicit() -> None:
    verifier = (ROOT / "verify_openvino.py").read_text(encoding="utf-8")
    offline = (ROOT / "offline_gate.py").read_text(encoding="utf-8")
    trainer = (ROOT / "train_q1.jl").read_text(encoding="utf-8")
    assert "Q1CorrectionOpenVINOBuilder(weights)" in verifier
    assert '"fresh_weights_constructor_used": True' in verifier
    assert '"Q1-offline-promoted"' in offline
    assert '"checkpoint_selection_performed": False' in offline
    assert '"earlier_checkpoint_rollback": False' in offline
    assert "require_hash(old_checkpoint_path" in trainer
    assert "jldopen(old_checkpoint_path" not in trainer
    assert "Optimisers.ClipNorm" not in trainer
    assert 'clip_mode="single_global_tree_l2"' in trainer
    provenance = (ROOT / "validate_source_fingerprint.py").read_text(encoding="utf-8")
    wrapper = (ROOT / "invoke_once.ps1").read_text(encoding="utf-8")
    assert "AUTHORIZED_BASE_COMMIT" in provenance
    assert "merge-base" in provenance and "--is-ancestor" in provenance
    assert "EXPECTED_PARENT_COMMIT" not in provenance
    assert "expected_parent_commit" not in wrapper
    assert "authorized_base_commit" in wrapper
    assert "actual_parent_commit" in wrapper
    assert "test_freeze_order_production.py" in wrapper
    assert "freeze_order_production_branch_executed = $true" in wrapper


def test_no_forbidden_seed_or_prior_marker_escape() -> None:
    executable = tuple(ROOT.glob("*.py")) + tuple(ROOT.glob("*.jl")) + tuple(ROOT.glob("*.ps1"))
    for path in executable:
        if path.name == "test_static.py":
            continue
        source = path.read_text(encoding="utf-8", errors="strict")
        assert "legacy_partial_tail_td_P1.started.json" not in source
        assert "legacy_full_feasibility_F.started.json" not in source
        for forbidden_seed in ("8001", "8008", "91001", "91032"):
            assert forbidden_seed not in source


if __name__ == "__main__":
    test_python_sources_parse()
    test_help_paths_are_safe()
    test_q1_scope_and_fresh_export_are_explicit()
    test_no_forbidden_seed_or_prior_marker_escape()
    print("Q1 static Python checks passed")
