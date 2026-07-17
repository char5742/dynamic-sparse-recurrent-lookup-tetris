from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def assignment_literal(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"assignment {name} not found in {path.name}")


def test_python_sources_parse() -> None:
    for name in (
        "correction_openvino.py",
        "verify_openvino.py",
        "offline_gate.py",
    ):
        path = ROOT / name
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_help_is_runtime_safe() -> None:
    for name, token in (
        ("verify_openvino.py", "output_directory"),
        ("offline_gate.py", "offline_npz"),
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / name), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert token in result.stdout


def test_exact_compact_weight_contract() -> None:
    shapes = assignment_literal(ROOT / "correction_openvino.py", "WEIGHT_SHAPES")
    assert len(shapes) == 18
    assert sum(__import__("math").prod(shape) for shape in shapes.values()) == 165_051
    assert shapes["ps.stem.layer_1.weight"] == (3, 3, 2, 8)
    assert shapes["ps.head.layer_1.weight"] == (256, 547)
    assert shapes["ps.head.layer_3.weight"] == (1, 64)


def test_preregistered_gate_constants() -> None:
    path = ROOT / "offline_gate.py"
    assert assignment_literal(path, "FIRST_ROW") == 2161
    assert assignment_literal(path, "LAST_ROW") == 2660
    assert assignment_literal(path, "EXPECTED_ELIGIBLE") == 494
    assert assignment_literal(path, "MIN_HUBER_IMPROVEMENT") == 0.15
    assert assignment_literal(path, "MIN_CORRELATION") == 0.20
    assert assignment_literal(path, "MIN_SIGN_AGREEMENT") == 0.60
    assert assignment_literal(path, "SIGN_TARGET_FLOOR") == 0.10
    assert assignment_literal(path, "MIN_TOP1_AGREEMENT") == 0.95
    assert assignment_literal(path, "MAX_TOP1_AGREEMENT_EXCLUSIVE") == 0.995
    assert assignment_literal(path, "MIN_TOP1_CHANGES") == 3
    assert assignment_literal(path, "MAX_CORRECTION_RMS") == 0.25


def test_fail_closed_disclosures_and_cpu_only_gate() -> None:
    verify = (ROOT / "verify_openvino.py").read_text(encoding="utf-8")
    offline = (ROOT / "offline_gate.py").read_text(encoding="utf-8")
    for source in (verify, offline):
        assert '"initializer_exposed": True' in source
        assert '"initializer_exposed_to_offline_rows": True' in source
        assert '"validation_or_test_seed_loaded": False' in source
        assert '"game_evaluation_run": False' in source
        assert '"npu_required": False' in source
    assert 'core.compile_model(model, "CPU")' in verify
    assert 'core.compile_model(str(ir_xml), "CPU")' in offline
    for field in (
        "copack_vs_per_state_padded_max_abs_error",
        "actual_count_vs_padded_valid_max_abs_error",
        "actual_count_vs_copack_valid_max_abs_error",
        "actual_count_fixed74_valid_output_invariance_verified",
        "four_state_copack_invariance_verified",
    ):
        assert field in verify
        assert field in offline
    assert '"Q1-offline-promoted" if passed else "Q1-offline-rejected"' in offline
    assert "passed = all(bool(value) for value in gates.values())" in offline
    for field in (
        "source_row",
        "old_action_index_one_based",
        "candidate_action_index_one_based",
        "old_top2_margin",
        "old_top2_exact_tie",
        "correction_new_minus_old_action",
        "old_q_at_old_action",
        "old_q_at_new_action",
        "combined_q_at_old_action",
        "combined_q_at_new_action",
        "combined_new_minus_old_action",
        "combined_old_new_exact_tie",
    ):
        assert field in offline


def main() -> None:
    test_python_sources_parse()
    test_help_is_runtime_safe()
    test_exact_compact_weight_contract()
    test_preregistered_gate_constants()
    test_fail_closed_disclosures_and_cpu_only_gate()
    print("Q1 correction OpenVINO/offline static checks passed")


if __name__ == "__main__":
    main()
