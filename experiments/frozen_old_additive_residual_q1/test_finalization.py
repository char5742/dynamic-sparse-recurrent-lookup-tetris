from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path

import finalize_result as finalizer


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return finalizer.sha256_file(path)


def build_complete_fixture(root: Path) -> None:
    inputs = root / "immutable"
    inputs.mkdir()
    input_paths: dict[str, Path] = {}
    for index, label in enumerate(finalizer.EXPECTED_INPUTS):
        path = inputs / f"input-{index}.bin"
        path.write_bytes(f"Q1 synthetic immutable input {label}\n".encode())
        input_paths[label] = path
    hashes = {label: digest(path) for label, path in input_paths.items()}
    finalizer.EXPECTED_INPUTS = hashes
    finalizer.OLD_CHECKPOINT_SHA256 = hashes["old_checkpoint"]
    finalizer.OLD_OPENVINO_SHA256 = hashes["old_openvino"]
    finalizer.INITIALIZER_SHA256 = hashes["initializer"]
    finalizer.DATASET_SHA256 = hashes["dataset"]
    finalizer.AUTHORIZATION_SHA256 = hashes["authorization"]

    eligible = list(range(1, 2125))
    eligibility = {
        "status": "eligibility_complete",
        "dataset_sha256": finalizer.DATASET_SHA256,
        "training_rows_loaded": [1, 2160],
        "offline_rows_loaded": False,
        "training_episode_ids": list(range(1, 13)),
        "training_eligible_rows": eligible,
        "training_eligible_count": 2124,
        "offline_eligibility_deferred_until_after_candidate_freeze": True,
        "validation_or_test_seed_loaded": False,
        "game_seed_loaded": False,
    }
    write_json(root / "eligibility.json", eligibility)
    ordered = (eligible * 4)[:8000]
    order_digest = hashlib.sha256(",".join(map(str, ordered)).encode()).hexdigest()
    order = {
        "status": "q1_order_frozen",
        "eligibility_path": str(root / "eligibility.json"),
        "eligibility_sha256": digest(root / "eligibility.json"),
        "rng": "Xoshiro(0x5131_2026)",
        "epochs_touched": 4,
        "update_count": 2000,
        "state_batch": 4,
        "ordered_rows": ordered,
        "minibatches": [ordered[index : index + 4] for index in range(0, 8000, 4)],
        "ordered_rows_sha256": order_digest,
        "priority_sampling": False,
        "validation_or_test_seed_loaded": False,
        "game_seed_loaded": False,
        "offline_rows_loaded": False,
    }
    write_json(root / "order_freeze.json", order)
    source_commit = "a" * 40
    freeze = {
        "experiment": finalizer.EXPERIMENT_ID,
        "source_commit": source_commit,
        "authorized_hardening_commit": source_commit,
        "expected_parent_commit": finalizer.EXPECTED_PARENT_COMMIT,
        "repository_clean": True,
        "output_directory": str(root),
        "source_fingerprint": {"source_sha256": "b" * 64},
        "manifest_sha256": "c" * 64,
        "source_fingerprint_audit": {
            "valid": True,
            "repository_binding": {
                "head": source_commit,
                "authorized_commit": source_commit,
                "parent": finalizer.EXPECTED_PARENT_COMMIT,
                "repository_clean": True,
            },
            "fingerprint": {"source_sha256": "b" * 64, "manifest_sha256": "c" * 64},
        },
        "harness_sha256": "d" * 64,
        "harness_files": [f"file-{index}" for index in range(18)],
        "immutable_inputs": [
            {"label": label, "path": str(input_paths[label]), "expected": value, "observed": value}
            for label, value in hashes.items()
        ],
        "constants": {
            "actions": 74,
            "batch": 4,
            "updates": 2000,
            "rng": "Xoshiro(0x5131_2026)",
            "n_step": 3,
            "parameter_count": 165051,
            "hard_wall_seconds": 720,
            "max_process_tree_bytes": 4 * 1024**3,
        },
        "eligibility_path": str(root / "eligibility.json"),
        "eligibility_sha256": digest(root / "eligibility.json"),
        "order_freeze_path": str(root / "order_freeze.json"),
        "order_freeze_sha256": digest(root / "order_freeze.json"),
        "ordered_rows_sha256": order_digest,
    }
    write_json(root / "freeze.json", freeze)

    for name in (
        "training.npz",
        "correction_update2000.jld2",
        "correction_weights.npz",
        "combined_reference.npz",
        "correction_dynamic.xml",
        "correction_dynamic.bin",
        "offline.npz",
    ):
        (root / name).write_bytes(f"synthetic {name}\n".encode())
    extraction = {
        "status": "training_extraction_complete",
        "dataset_sha256": finalizer.DATASET_SHA256,
        "rows_loaded": [1, 2160],
        "row_count": 2160,
        "episodes": list(range(1, 13)),
        "target_eligible_count": 2124,
        "bootstrap_source": "max(raw stored old-Q[1:action_count] at t+3)",
        "dagger_behavior_bootstrap_used": False,
        "initializer_exposed_to_offline_rows": True,
        "offline_role": "reused_development_guard",
        "offline_loaded_before_candidate_freeze": False,
        "eligible_rows_sha256": "1" * 64,
        "targets_sha256": "2" * 64,
        "selected_old_q_sha256": "3" * 64,
        "target_residual_sha256": "4" * 64,
        "validation_or_test_seed_loaded": False,
        "game_seed_loaded": False,
        "npz_path": str(root / "training.npz"),
        "npz_sha256": digest(root / "training.npz"),
    }
    write_json(root / "training_extraction.json", extraction)
    updates = [
        {
            "update": index + 1,
            "source_rows": ordered[4 * index : 4 * index + 4],
            "loss": 1.0,
            "seconds": 0.01,
            "allocated_bytes": 10,
            "gradient_elements": 165051,
            "gradient_array_count": 1,
            "parameter_elements": 165051,
            "parameter_array_count": 1,
        }
        for index in range(2000)
    ]
    constants = {
        "experiment": finalizer.EXPERIMENT_ID,
        "train_rows": [1, 2160],
        "base_rows": [1, 1500],
        "dagger_rows": [1501, 2160],
        "offline_rows": [2161, 2660],
        "train_episodes": list(range(1, 13)),
        "offline_episodes": [13, 14],
        "expected_train_eligible": 2124,
        "expected_offline_eligible": 494,
        "actions": 74,
        "batch": 4,
        "updates": 2000,
        "rng": "Xoshiro(0x5131_2026)",
        "n_step": 3,
        "initializer_exposed_to_offline_rows": True,
        "offline_role": "reused_development_guard",
        "offline_is_held_out_generalization": False,
        "game_strength_evidence": False,
        "validation_seed_used": False,
        "sealed_test_seed_used": False,
    }
    input_hashes = {
        "initializer": finalizer.INITIALIZER_SHA256,
        "old_checkpoint": finalizer.OLD_CHECKPOINT_SHA256,
        "training_npz": digest(root / "training.npz"),
        "freeze": digest(root / "freeze.json"),
        "order": digest(root / "order_freeze.json"),
    }
    diagnostic = {
        "rows": 100,
        "mean_target_residual": 0.1,
        "mean_prediction": 0.1,
        "selected_huber": 0.1,
        "correlation": 0.5,
        "sign_agreement": 0.7,
    }
    training = {
        "status": "training_phase_complete",
        "constants": constants,
        "wall_seconds": 30.0,
        "update_count": 2000,
        "zero_gate": {"valid_outputs": 100, "bitwise_zero": True, "combined_stored_old_max_abs_error": 0.0, "top1_agreement": 1.0},
        "parameter_count": 165051,
        "parameter_array_count": 1,
        "gradient_paths": ["head.weight"],
        "gradient_elements_every_update": True,
        "first_update_seconds": 0.1,
        "warm_update_seconds": [0.01] * 20,
        "warm_median_seconds": 0.01,
        "projected_total_seconds": 50.0,
        "update_records": updates,
        "base_role_diagnostics": diagnostic,
        "dagger_role_diagnostics": diagnostic,
        "dagger_three_step_off_policy_bias_disclosed": True,
        "initializer_exposed_to_offline_rows": True,
        "offline_role": "reused_development_guard",
        "input_hashes_before": input_hashes,
        "input_hashes_after": input_hashes,
        "immutable_inputs_unchanged": True,
        "candidate_checkpoint": str(root / "correction_update2000.jld2"),
        "candidate_checkpoint_sha256": digest(root / "correction_update2000.jld2"),
        "weights_path": str(root / "correction_weights.npz"),
        "weights_sha256": digest(root / "correction_weights.npz"),
        "reference_path": str(root / "combined_reference.npz"),
        "reference_sha256": digest(root / "combined_reference.npz"),
        "validation_or_test_seed_loaded": False,
        "game_run": False,
    }
    write_json(root / "training_phase.json", training)
    openvino = {
        "status": "correction_openvino_gate_pass",
        "device": "CPU",
        "npu_required": False,
        "training_phase_path": str(root / "training_phase.json"),
        "training_phase_sha256": digest(root / "training_phase.json"),
        "weights_path": str(root / "correction_weights.npz"),
        "weights_sha256": digest(root / "correction_weights.npz"),
        "combined_reference_path": str(root / "combined_reference.npz"),
        "combined_reference_sha256": digest(root / "combined_reference.npz"),
        "ir_xml": str(root / "correction_dynamic.xml"),
        "ir_xml_sha256": digest(root / "correction_dynamic.xml"),
        "ir_bin": str(root / "correction_dynamic.bin"),
        "ir_bin_sha256": digest(root / "correction_dynamic.bin"),
        "witness_state_count": 4,
        "witness_action_counts": [10, 20, 30, 40],
        "export_seconds": 0.01,
        "compile_seconds": 0.01,
        "inference_seconds": 0.01,
        "invariance_inference_seconds": 0.01,
        "correction_max_abs_error": 0.0,
        "combined_max_abs_error": 0.0,
        "copack_vs_per_state_padded_max_abs_error": 0.0,
        "actual_count_vs_padded_valid_max_abs_error": 0.0,
        "actual_count_vs_copack_valid_max_abs_error": 0.0,
        "actual_count_fixed74_valid_output_invariance_verified": True,
        "four_state_copack_invariance_verified": True,
        "cpu_tolerance": 1.0e-4,
        "all_outputs_finite": True,
        "fresh_weights_constructor_used": True,
        "fresh_combined_reference_used": True,
        "initializer_exposed_to_offline_rows": True,
        "offline_role": "reused_development_guard",
        "validation_or_test_seed_loaded": False,
        "game_evaluation_run": False,
    }
    write_json(root / "openvino_gate.json", openvino)
    offline_extraction = copy.deepcopy(extraction)
    offline_extraction.update(
        status="offline_extraction_complete",
        rows_loaded=[2161, 2660],
        row_count=500,
        episodes=[13, 14],
        target_eligible_count=494,
        npz_path=str(root / "offline.npz"),
        npz_sha256=digest(root / "offline.npz"),
    )
    write_json(root / "offline_extraction.json", offline_extraction)
    changed = [
        {
            "source_row": 2161 + index,
            "old_action_index_one_based": 1,
            "candidate_action_index_one_based": 2,
            "old_top2_margin": 0.1,
            "correction_new_minus_old_action": 0.2,
            "old_q_at_old_action": 1.0,
            "old_q_at_new_action": 0.9,
            "combined_q_at_old_action": 1.0,
            "combined_q_at_new_action": 1.1,
            "combined_new_minus_old_action": 0.1,
        }
        for index in range(3)
    ]
    gates = {name: True for name in finalizer.GATE_NAMES}
    offline = {
        "status": "Q1-offline-promoted",
        "rows": [2161, 2660],
        "row_count": 500,
        "episodes": [13, 14],
        "eligible_target_count": 494,
        "offline_role": "reused_development_guard",
        "offline_is_held_out_generalization": False,
        "initializer_exposed": True,
        "initializer_exposed_to_offline_rows": True,
        "target_metrics_use_eligible_rows_only": True,
        "top1_and_rms_use_all_valid_actions_in_all_500_rows": True,
        "training_phase_sha256": digest(root / "training_phase.json"),
        "openvino_gate_sha256": digest(root / "openvino_gate.json"),
        "weights_sha256": digest(root / "correction_weights.npz"),
        "offline_npz_path": str(root / "offline.npz"),
        "offline_npz_sha256": digest(root / "offline.npz"),
        "ir_xml_sha256": digest(root / "correction_dynamic.xml"),
        "ir_bin_sha256": digest(root / "correction_dynamic.bin"),
        "device": "CPU",
        "npu_required": False,
        "compile_seconds": 0.01,
        "inference_seconds": 0.01,
        "logical_network_calls": 500,
        "candidate_evaluations": 1000,
        "zero_correction_selected_huber": 1.0,
        "candidate_selected_huber": 0.8,
        "selected_huber_improvement_fraction": 0.2,
        "selected_residual_correlation": 0.3,
        "sign_target_abs_floor": 0.1,
        "sign_subset_count": 10,
        "sign_agreement": 0.7,
        "combined_top1_agreements": 497,
        "combined_top1_changes": 3,
        "combined_top1_agreement": 497 / 500,
        "correction_rms": 0.1,
        "thresholds": {
            "minimum_huber_improvement_fraction": 0.15,
            "minimum_correlation": 0.2,
            "minimum_sign_agreement": 0.6,
            "minimum_top1_agreement": 0.95,
            "maximum_top1_agreement_exclusive": 0.995,
            "minimum_top1_changes": 3,
            "maximum_correction_rms": 0.25,
        },
        "gates": gates,
        "changed_state_diagnostics": changed,
        "checkpoint_selection_performed": False,
        "earlier_checkpoint_rollback": False,
        "validation_or_test_seed_loaded": False,
        "game_evaluation_run": False,
    }
    write_json(root / "offline_gate.json", offline)


def test_complete_and_fail_closed_mutations() -> None:
    with tempfile.TemporaryDirectory(prefix="q1-finalizer-") as temporary:
        root = Path(temporary)
        build_complete_fixture(root)
        result = finalizer.assess(root)
        assert result["success"], "\n".join(result["failures"])
        assert result["promotion"] == "Q1-offline-promoted"
        assert result["game_strength_evidence"] is False
        assert result["model_beat_claim"] is False

        path = root / "offline_gate.json"
        pristine = json.loads(path.read_text())
        cases = (
            ("validation_or_test_seed_loaded", True, "scope field"),
            ("checkpoint_selection_performed", True, "checkpoint selection"),
            ("status", "Q1-offline-rejected", "promotion status"),
        )
        for field, value, needle in cases:
            mutated = copy.deepcopy(pristine)
            mutated[field] = value
            write_json(path, mutated)
            rejected = finalizer.assess(root)
            assert not rejected["success"]
            assert any(needle in failure for failure in rejected["failures"])
        write_json(path, pristine)

        mutated = copy.deepcopy(pristine)
        mutated["gates"].pop("all_outputs_finite")
        write_json(path, mutated)
        assert not finalizer.assess(root)["success"]
        write_json(path, pristine)

        mutated = copy.deepcopy(pristine)
        mutated.pop("game_evaluation_run")
        write_json(path, mutated)
        missing_scope = finalizer.assess(root)
        assert not missing_scope["success"]
        assert any("missing required scope field game_evaluation_run" in item for item in missing_scope["failures"])
        write_json(path, pristine)

        openvino_path = root / "openvino_gate.json"
        openvino = json.loads(openvino_path.read_text())
        mutated_openvino = copy.deepcopy(openvino)
        mutated_openvino.pop("actual_count_fixed74_valid_output_invariance_verified")
        write_json(openvino_path, mutated_openvino)
        assert not finalizer.assess(root)["success"]
        write_json(openvino_path, openvino)

        training_path = root / "training_phase.json"
        training = json.loads(training_path.read_text())
        mutated_training = copy.deepcopy(training)
        mutated_training["zero_gate"]["bitwise_zero"] = False
        write_json(training_path, mutated_training)
        assert not finalizer.assess(root)["success"]


def test_incomplete_stub_rejected() -> None:
    with tempfile.TemporaryDirectory(prefix="q1-finalizer-empty-") as temporary:
        root = Path(temporary)
        empty = finalizer.assess(root)
        assert not empty["success"]
        assert len(empty["failures"]) >= 8
        write_json(root / "offline_gate.json", {"status": "Q1-offline-promoted"})
        flattering = finalizer.assess(root)
        assert not flattering["success"]
        assert flattering["game_strength_evidence"] is False
        assert flattering["model_beat_claim"] is False


if __name__ == "__main__":
    test_complete_and_fail_closed_mutations()
    test_incomplete_stub_rejected()
    print("Q1 finalization synthetic checks passed")
