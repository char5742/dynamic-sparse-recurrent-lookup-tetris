from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_python_sources_parse() -> None:
    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_extract_help_is_safe() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "extract_dataset.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "eligibility" in result.stdout
    assert "training" in result.stdout
    assert "offline" in result.stdout


def test_candidate_inference_requires_explicit_weights() -> None:
    source = (ROOT / "weighted_inference.py").read_text(encoding="utf-8")
    assert "weights_path" in source
    assert "LegacyOpenVINOBuilder(weights)" in source
    assert "LegacyOpenVINOInference(" not in source


def test_preregistered_scripts_do_not_modify_f() -> None:
    for path in ROOT.iterdir():
        if path.is_file() and path.name != "test_static.py":
            source = path.read_text(encoding="utf-8", errors="ignore")
            assert "legacy_full_feasibility_F.started.json" not in source


if __name__ == "__main__":
    test_python_sources_parse()
    test_extract_help_is_safe()
    test_candidate_inference_requires_explicit_weights()
    test_preregistered_scripts_do_not_modify_f()
    print("P1 static Python checks passed")
