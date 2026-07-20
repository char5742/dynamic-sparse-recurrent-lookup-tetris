"""Static-only contract for the direct k-scale CLI.

This test intentionally does not import Julia, PythonCall, OpenVINO, JLD2, or
hardware drivers. Runtime validation belongs to the exclusive post-training
idle window.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_hashstem_sparse3_k_scale_benchmark.jl"
VIEW = ROOT / "sparse_hashstem_named_inference_view.jl"
DOC = ROOT / "HASHSTEM_SPARSE3_K_SCALE_BENCHMARK.md"


def test_runner_binds_minimum_production_artifacts() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for token in (
        'module HashStemSparse3DirectRuntime',
        'include("sparse_hashstem_bridge.jl")',
        'include("hashstem_sparse3_k_scale_gate.jl")',
        'include("sparse_hashstem_named_inference_view.jl")',
        "H.three_layer_inputs_from_hashstem",
        "H.route_sparse_named_inputs!",
        "H.evaluate_hashstem_sparse3_k_scale_gate",
        '"cpu_k64"',
        '"cpu_k128"',
        '"npu_k128"',
        '"--teacher-checkpoint"',
        '"--teacher-checkpoint-sha256"',
        '"--hashstem-weights"',
        '"--snapshot-metadata"',
        '"--fixed-ir"',
        '"--fixed-bin"',
        '"--dynamic-ir"',
        '"--dynamic-bin"',
        '"--workload"',
        '"--workload-sha256"',
        '"raw_records.jsonl"',
        '"decision.json"',
    ):
        assert token in source


def test_runner_is_train_only_and_fail_closed() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    view = VIEW.read_text(encoding="utf-8")
    for token in (
        '"teacher_v3_train"',
        '"training_only"',
        '"reserved_seed_free"',
        "Set(8001:8008)",
        "Set(91001:91032)",
        "at least 30 timed repetitions are required",
        "timed repetitions must be a multiple of six",
        'uppercase.(values) == [requested]',
        '"NPU" in uppercase.(devices)',
        "verify(artifact)",
        "passed || exit(2)",
    ):
        assert token in source
    for token in (
        "named inference view requires an ordinary trainer checkpoint",
        "sparse_bridge_variant_spec",
        "SparseDynamic3Layer.route_forward!",
        "named view changed bank/head/optimizer state",
    ):
        assert token in view
    # The production runner embeds OpenVINO through PythonCall. It must not
    # silently replace the NPU cell with an external child or shell command.
    for forbidden in ("run(`", "pipeline(`", "Cmd(", "python.exe", "python3"):
        assert forbidden not in source


def test_required_stage_and_metric_surface_is_present() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for stage in (
        "packing",
        "hashstem",
        "routing",
        "gather",
        "sparse",
        "head",
        "sync",
        "end_to_end",
    ):
        assert f'"{stage}"' in source
    for metric in (
        "teacher_top1_agreement",
        "teacher_ndcg",
        "teacher_pairwise_accuracy",
        "route_id_matches",
        "action_top1_matches",
        "top2_swap_count",
        "packed_inputs_sha256",
        "timed_gc_observed",
        "maximum_q_error",
        "maximum_margin_budget_ratio",
    ):
        assert metric in source


def test_documentation_marks_unexecuted_and_explains_boundary() -> None:
    text = DOC.read_text(encoding="utf-8")
    for token in (
        "implemented, static-only, not yet executed",
        "ordinary",
        "No optimizer update occurs",
        "no Python child",
        "Validation seeds",
        "raw_records.jsonl",
        "decision.json",
    ):
        assert token in text
