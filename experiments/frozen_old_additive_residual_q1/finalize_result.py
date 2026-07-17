from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "frozen_old_additive_residual_Q1"
AUTHORIZED_BASE_COMMIT = "8d784985f300598d2a05ed4402902ae86dfb4908"
Q1_SOURCE_PREFIX = "experiments/frozen_old_additive_residual_q1/"
OLD_CHECKPOINT_SHA256 = "7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1"
OLD_OPENVINO_SHA256 = "2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba"
INITIALIZER_SHA256 = "1273b55b7616f912a3120718f77770af39c489f7fbe51052f4810d8a03291270"
DATASET_SHA256 = "4f10cfcf545c97eb3f56e8511921a1a6b50fa5ab166fac2eb3575eacf84b71ba"
AUTHORIZATION_SHA256 = "f0cd7bce2c39b353a3377dc2ebdd624ab485a2b96c5750f4bc97e7fd91a5cf00"
EXPECTED_INPUTS = {
    "old_checkpoint": OLD_CHECKPOINT_SHA256,
    "old_openvino": OLD_OPENVINO_SHA256,
    "initializer": INITIALIZER_SHA256,
    "dataset": DATASET_SHA256,
    "authorization": AUTHORIZATION_SHA256,
}
GATE_NAMES = {
    "all_outputs_finite",
    "selected_huber_improvement_at_least_15_percent",
    "selected_residual_correlation_at_least_0_2",
    "sign_agreement_at_least_0_6_for_abs_target_at_least_0_1",
    "combined_top1_at_least_0_95",
    "combined_top1_below_0_995",
    "combined_top1_changes_at_least_3",
    "correction_rms_at_most_0_25",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_optional(path: Path) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        return value
    except (OSError, json.JSONDecodeError):
        return None


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def finite_nonnegative(value: Any) -> bool:
    return finite(value) and value >= 0


def integer_nonnegative(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def path_is(root: Path, value: Any, expected_name: str) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        return Path(value).resolve() == (root / expected_name).resolve()
    except OSError:
        return False


def known_forbidden_flags_clear(value: Any) -> bool:
    forbidden = {
        "validation_or_test_seed_loaded",
        "validation_or_test_data_used",
        "validation_seed_used",
        "sealed_test_seed_used",
        "held_out_test_seeds_used",
        "sealed_test_authorized",
        "game_seed_loaded",
        "game_run",
        "game_evaluation_run",
        "game_evaluation_authorized",
        "game_strength_evidence",
        "model_beat_claim",
        "checkpoint_selection_performed",
        "earlier_checkpoint_rollback",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden and child is not False:
                return False
            if not known_forbidden_flags_clear(child):
                return False
    elif isinstance(value, list):
        return all(known_forbidden_flags_clear(child) for child in value)
    return True


def assess(root: Path) -> dict[str, Any]:
    root = root.resolve()
    names = (
        "freeze.json",
        "eligibility.json",
        "order_freeze.json",
        "training_extraction.json",
        "training_phase.json",
        "training_failure.json",
        "openvino_gate.json",
        "offline_extraction.json",
        "offline_gate.json",
    )
    artifacts = {name.removesuffix(".json"): read_optional(root / name) for name in names}
    freeze = artifacts["freeze"]
    eligibility = artifacts["eligibility"]
    order = artifacts["order_freeze"]
    extraction = artifacts["training_extraction"]
    training = artifacts["training_phase"]
    openvino = artifacts["openvino_gate"]
    offline_extraction = artifacts["offline_extraction"]
    offline = artifacts["offline_gate"]
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    def require_false(record: Any, label: str, *fields: str) -> None:
        if not isinstance(record, dict):
            return
        for field in fields:
            require(field in record, f"{label} missing required scope field {field}")
            require(record.get(field) is False, f"{label} scope field {field} is not exactly false")

    def file_hash_matches(name: str, expected: Any) -> None:
        path = root / name
        require(path.is_file(), f"missing artifact file {name}")
        require(
            isinstance(expected, str) and path.is_file() and sha256_file(path) == expected,
            f"artifact SHA-256 mismatch for {name}",
        )

    require(isinstance(freeze, dict), "missing or malformed freeze")
    if isinstance(freeze, dict):
        source_commit = freeze.get("source_commit")
        require(freeze.get("experiment") == EXPERIMENT_ID, "freeze experiment mismatch")
        require(isinstance(source_commit, str) and len(source_commit) == 40, "freeze source commit is not full")
        require(freeze.get("authorized_hardening_commit") == source_commit, "freeze authorization mismatch")
        require(freeze.get("authorized_base_commit") == AUTHORIZED_BASE_COMMIT, "freeze authorized base mismatch")
        actual_parent = freeze.get("actual_parent_commit")
        require(isinstance(actual_parent, str) and len(actual_parent) == 40, "freeze actual parent commit is not full")
        require(freeze.get("repository_clean") is True, "freeze repository was not clean")
        require(path_is(root, freeze.get("output_directory"), "."), "freeze output directory mismatch")
        audit = freeze.get("source_fingerprint_audit")
        require(isinstance(audit, dict) and audit.get("valid") is True, "source fingerprint audit rejected")
        if isinstance(audit, dict):
            binding = audit.get("repository_binding", {})
            fingerprint = audit.get("fingerprint", {})
            require(binding.get("head") == source_commit, "source audit HEAD mismatch")
            require(binding.get("authorized_commit") == source_commit, "source audit authorization mismatch")
            require(binding.get("parent") == actual_parent, "source audit actual parent mismatch")
            require(binding.get("authorized_base_commit") == AUTHORIZED_BASE_COMMIT, "source audit authorized base mismatch")
            require(binding.get("base_is_ancestor") is True, "source audit base ancestry failed")
            require(binding.get("repository_clean") is True, "source audit repository not clean")
            changed_paths = binding.get("changed_paths")
            require(
                isinstance(changed_paths, list)
                and len(changed_paths) > 0
                and all(isinstance(path, str) and path.startswith(Q1_SOURCE_PREFIX) for path in changed_paths),
                "source audit base-to-HEAD paths escaped Q1 namespace",
            )
            require(
                fingerprint.get("source_sha256") == freeze.get("source_fingerprint", {}).get("source_sha256"),
                "source fingerprint aggregate mismatch",
            )
            require(
                fingerprint.get("manifest_sha256") == freeze.get("manifest_sha256"),
                "source fingerprint manifest mismatch",
            )
        require(isinstance(freeze.get("harness_sha256"), str), "freeze harness aggregate missing")
        harness_files = freeze.get("harness_files")
        require(isinstance(harness_files, list) and len(harness_files) >= 18, "freeze harness inventory incomplete")
        inputs = freeze.get("immutable_inputs")
        require(isinstance(inputs, list), "freeze immutable input inventory missing")
        if isinstance(inputs, list):
            observed = {item.get("label"): item for item in inputs if isinstance(item, dict)}
            require(set(observed) == set(EXPECTED_INPUTS), "freeze immutable input roles mismatch")
            for label, digest in EXPECTED_INPUTS.items():
                item = observed.get(label, {})
                path_value = item.get("path")
                require(item.get("expected") == digest, f"freeze {label} expected hash mismatch")
                require(item.get("observed") == digest, f"freeze {label} observed hash mismatch")
                path = Path(path_value) if isinstance(path_value, str) else None
                require(path is not None and path.is_file(), f"freeze {label} path missing")
                require(path is not None and path.is_file() and sha256_file(path) == digest, f"live {label} hash mismatch")
        constants = freeze.get("constants", {})
        expected_constants = {
            "actions": 74,
            "batch": 4,
            "updates": 2000,
            "rng": "Xoshiro(0x5131_2026)",
            "n_step": 3,
            "parameter_count": 165051,
            "hard_wall_seconds": 720,
            "max_process_tree_bytes": 4 * 1024**3,
        }
        for key, expected in expected_constants.items():
            require(constants.get(key) == expected, f"freeze constant {key} mismatch")
        file_hash_matches("eligibility.json", freeze.get("eligibility_sha256"))
        file_hash_matches("order_freeze.json", freeze.get("order_freeze_sha256"))
        require(path_is(root, freeze.get("eligibility_path"), "eligibility.json"), "freeze eligibility path mismatch")
        require(path_is(root, freeze.get("order_freeze_path"), "order_freeze.json"), "freeze order path mismatch")

    require(isinstance(eligibility, dict), "missing or malformed eligibility")
    eligible_rows: list[int] = []
    if isinstance(eligibility, dict):
        eligible_rows = eligibility.get("training_eligible_rows") if isinstance(eligibility.get("training_eligible_rows"), list) else []
        require(eligibility.get("status") == "eligibility_complete", "eligibility status mismatch")
        require(eligibility.get("dataset_sha256") == DATASET_SHA256, "eligibility dataset mismatch")
        require(eligibility.get("training_rows_loaded") == [1, 2160], "eligibility training role mismatch")
        require(eligibility.get("offline_rows_loaded") is False, "offline rows loaded before candidate freeze")
        require(eligibility.get("training_episode_ids") == list(range(1, 13)), "eligibility episode role mismatch")
        require(eligibility.get("training_eligible_count") == 2124, "eligibility count mismatch")
        require(len(eligible_rows) == 2124 and len(set(eligible_rows)) == 2124, "eligibility rows missing or duplicate")
        require(all(isinstance(row, int) and 1 <= row <= 2160 for row in eligible_rows), "eligibility row escaped role")
        require(eligibility.get("offline_eligibility_deferred_until_after_candidate_freeze") is True, "offline eligibility was not deferred")
    require_false(eligibility, "eligibility", "validation_or_test_seed_loaded", "game_seed_loaded", "offline_rows_loaded")

    require(isinstance(order, dict), "missing or malformed order freeze")
    ordered_rows: list[int] = []
    if isinstance(order, dict):
        ordered_rows = order.get("ordered_rows") if isinstance(order.get("ordered_rows"), list) else []
        digest = hashlib.sha256(",".join(map(str, ordered_rows)).encode()).hexdigest()
        require(order.get("status") == "q1_order_frozen", "order freeze status mismatch")
        require(order.get("rng") == "Xoshiro(0x5131_2026)", "order RNG mismatch")
        require(order.get("update_count") == 2000 and order.get("state_batch") == 4, "order update/batch mismatch")
        require(len(ordered_rows) == 8000 and all(row in set(eligible_rows) for row in ordered_rows), "order rows mismatch")
        require(order.get("ordered_rows_sha256") == digest, "order digest mismatch")
        require(order.get("priority_sampling") is False, "priority sampling was enabled")
        batches = order.get("minibatches")
        require(
            isinstance(batches, list)
            and len(batches) == 2000
            and all(batch == ordered_rows[4 * index : 4 * index + 4] for index, batch in enumerate(batches)),
            "frozen minibatches do not match ordered rows",
        )
        file_hash_matches("eligibility.json", order.get("eligibility_sha256"))
        require(path_is(root, order.get("eligibility_path"), "eligibility.json"), "order eligibility path mismatch")
        if isinstance(freeze, dict):
            require(freeze.get("ordered_rows_sha256") == digest, "freeze/order digest mismatch")
    require_false(order, "order freeze", "validation_or_test_seed_loaded", "game_seed_loaded", "offline_rows_loaded")

    require(isinstance(extraction, dict), "missing or malformed training extraction")
    if isinstance(extraction, dict):
        require(extraction.get("status") == "training_extraction_complete", "training extraction status mismatch")
        require(extraction.get("dataset_sha256") == DATASET_SHA256, "training extraction dataset mismatch")
        require(extraction.get("rows_loaded") == [1, 2160] and extraction.get("row_count") == 2160, "training extraction row role mismatch")
        require(extraction.get("episodes") == list(range(1, 13)), "training extraction episodes mismatch")
        require(extraction.get("target_eligible_count") == 2124, "training extraction eligible count mismatch")
        require(extraction.get("bootstrap_source") == "max(raw stored old-Q[1:action_count] at t+3)", "training bootstrap mismatch")
        require(extraction.get("dagger_behavior_bootstrap_used") is False, "DAgger behavior bootstrap used")
        require(extraction.get("initializer_exposed_to_offline_rows") is True, "initializer exposure missing")
        require(extraction.get("offline_role") == "reused_development_guard", "training extraction offline role mismatch")
        require(extraction.get("offline_loaded_before_candidate_freeze") is False, "offline loaded before candidate freeze")
        for digest_name in ("eligible_rows_sha256", "targets_sha256", "selected_old_q_sha256", "target_residual_sha256"):
            require(isinstance(extraction.get(digest_name), str) and len(extraction[digest_name]) == 64, f"training extraction missing {digest_name}")
        file_hash_matches("training.npz", extraction.get("npz_sha256"))
        require(path_is(root, extraction.get("npz_path"), "training.npz"), "training extraction NPZ path mismatch")
    require_false(extraction, "training extraction", "dagger_behavior_bootstrap_used", "offline_loaded_before_candidate_freeze", "validation_or_test_seed_loaded", "game_seed_loaded")

    require(artifacts["training_failure"] is None, "training failure artifact exists")
    require(isinstance(training, dict), "missing or malformed training phase")
    if isinstance(training, dict):
        require(training.get("status") == "training_phase_complete", "training phase status mismatch")
        constants = training.get("constants", {})
        exact_constants = {
            "experiment": EXPERIMENT_ID,
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
            "clip_mode": "single_global_tree_l2",
            "initializer_exposed_to_offline_rows": True,
            "offline_role": "reused_development_guard",
            "offline_is_held_out_generalization": False,
            "game_strength_evidence": False,
            "validation_seed_used": False,
            "sealed_test_seed_used": False,
        }
        for key, expected in exact_constants.items():
            require(constants.get(key) == expected, f"training constant {key} mismatch")
        require(training.get("update_count") == 2000, "training update count mismatch")
        require(training.get("parameter_count") == 165051, "training parameter count mismatch")
        require(integer_nonnegative(training.get("parameter_array_count")), "training parameter array count invalid")
        zero = training.get("zero_gate", {})
        require(zero.get("bitwise_zero") is True, "update-0 correction was not bitwise +0")
        require(zero.get("combined_stored_old_max_abs_error") == 0, "update-0 combined identity failed")
        require(zero.get("top1_agreement") == 1, "update-0 top-1 identity failed")
        require(integer_nonnegative(zero.get("valid_outputs")) and zero.get("valid_outputs") > 0, "update-0 witness count invalid")
        require(training.get("gradient_elements_every_update") is True, "gradient coverage incomplete")
        require(training.get("clip_mode") == "single_global_tree_l2", "training clip mode mismatch")
        clip_tolerance = training.get("global_gradient_norm_tolerance")
        require(finite(clip_tolerance) and clip_tolerance == 1.0e-6, "training global gradient tolerance mismatch")
        require(training.get("global_gradient_norms_finite_every_update") is True, "non-finite global gradient norm recorded")
        require(training.get("global_gradient_post_norm_within_limit_every_update") is True, "global gradient post-norm limit failed")
        require(training.get("global_gradient_uniform_scale_every_update") is True, "global gradient uniform-scale gate failed")
        paths = training.get("gradient_paths")
        require(isinstance(paths, list) and len(paths) == training.get("parameter_array_count") and len(set(paths)) == len(paths), "gradient path inventory mismatch")
        require(finite_nonnegative(training.get("wall_seconds")) and training["wall_seconds"] <= 720, "training wall gate failed")
        require(finite_nonnegative(training.get("first_update_seconds")) and training["first_update_seconds"] <= 60, "first update gate failed")
        warm = training.get("warm_update_seconds")
        require(isinstance(warm, list) and len(warm) == 20 and all(finite_nonnegative(value) and value <= 1 for value in warm), "warm update gate failed")
        require(finite_nonnegative(training.get("warm_median_seconds")) and training["warm_median_seconds"] <= 0.25, "warm median gate failed")
        require(finite_nonnegative(training.get("projected_total_seconds")) and training["projected_total_seconds"] <= 720, "training projection gate failed")
        updates = training.get("update_records")
        require(isinstance(updates, list) and len(updates) == 2000, "update ledger count mismatch")
        if isinstance(updates, list) and len(updates) == 2000:
            require([entry.get("update") for entry in updates] == list(range(1, 2001)), "update ledger order mismatch")
            require([row for entry in updates for row in entry.get("source_rows", [])] == ordered_rows, "update ledger row order mismatch")
            require(all(entry.get("gradient_elements") == 165051 for entry in updates), "update gradient accounting mismatch")
            require(all(entry.get("parameter_elements") == 165051 for entry in updates), "update parameter accounting mismatch")
            require(all(entry.get("parameter_array_count") == training.get("parameter_array_count") for entry in updates), "update parameter-array accounting mismatch")
            for entry in updates:
                before_norm = entry.get("global_gradient_norm_before")
                after_norm = entry.get("global_gradient_norm_after")
                scale = entry.get("global_gradient_scale")
                require(entry.get("clip_mode") == "single_global_tree_l2", "update clip mode mismatch")
                require(finite_nonnegative(before_norm), "update global pre-norm invalid")
                require(finite_nonnegative(after_norm) and after_norm <= 1.0 + 1.0e-6, "update global post-norm invalid")
                require(finite(scale) and 0 < scale <= 1, "update global scale invalid")
                if finite_nonnegative(before_norm) and finite(scale):
                    expected_scale = 1.0 if before_norm <= 1.0 else 1.0 / before_norm
                    require(math.isclose(scale, expected_scale, rel_tol=1.0e-12, abs_tol=1.0e-15), "update global scale is inconsistent with pre-norm")
                    require(abs(after_norm - before_norm * scale) <= 1.0e-6 * max(1.0, before_norm * scale), "update global post-norm is inconsistent with shared scale")
                require(entry.get("all_gradient_leaves_same_scale") is True, "update did not apply one scale to all leaves")
                require(finite_nonnegative(entry.get("maximum_leaf_scale_error")) and entry["maximum_leaf_scale_error"] <= 1.0e-6, "update leaf scale error exceeded tolerance")
            require(all(finite(entry.get("loss")) and finite_nonnegative(entry.get("seconds")) for entry in updates), "update loss/time ledger invalid")
            require(all(entry["seconds"] <= 1 for entry in updates[5:25]), "warm update individual limit failed")
        for role in ("base_role_diagnostics", "dagger_role_diagnostics"):
            record = training.get(role, {})
            require(integer_nonnegative(record.get("rows")) and record.get("rows", 0) > 0, f"{role} row count invalid")
            for field in ("mean_target_residual", "mean_prediction", "selected_huber", "sign_agreement"):
                require(finite(record.get(field)), f"{role} {field} invalid")
            correlation = record.get("correlation")
            require(correlation is None or finite(correlation), f"{role} correlation invalid")
        require(training.get("dagger_three_step_off_policy_bias_disclosed") is True, "DAgger off-policy caveat missing")
        require(training.get("initializer_exposed_to_offline_rows") is True, "training initializer exposure missing")
        require(training.get("offline_role") == "reused_development_guard", "training offline role mismatch")
        before = training.get("input_hashes_before")
        after = training.get("input_hashes_after")
        require(isinstance(before, dict) and before == after and training.get("immutable_inputs_unchanged") is True, "training immutable input hashes changed")
        if isinstance(before, dict):
            expected_hashes = {
                "initializer": INITIALIZER_SHA256,
                "old_checkpoint": OLD_CHECKPOINT_SHA256,
                "training_npz": sha256_file(root / "training.npz") if (root / "training.npz").is_file() else None,
                "freeze": sha256_file(root / "freeze.json") if (root / "freeze.json").is_file() else None,
                "order": sha256_file(root / "order_freeze.json") if (root / "order_freeze.json").is_file() else None,
            }
            require(before == expected_hashes, "training input hash provenance mismatch")
        file_hash_matches("correction_update2000.jld2", training.get("candidate_checkpoint_sha256"))
        file_hash_matches("correction_weights.npz", training.get("weights_sha256"))
        file_hash_matches("combined_reference.npz", training.get("reference_sha256"))
        require(path_is(root, training.get("candidate_checkpoint"), "correction_update2000.jld2"), "candidate checkpoint path mismatch")
        require(path_is(root, training.get("weights_path"), "correction_weights.npz"), "correction weights path mismatch")
        require(path_is(root, training.get("reference_path"), "combined_reference.npz"), "combined reference path mismatch")
    require_false(training, "training phase", "validation_or_test_seed_loaded", "game_run")

    require(isinstance(openvino, dict), "missing or malformed OpenVINO gate")
    if isinstance(openvino, dict):
        require(openvino.get("status") == "correction_openvino_gate_pass", "OpenVINO status mismatch")
        file_hash_matches("training_phase.json", openvino.get("training_phase_sha256"))
        file_hash_matches("correction_weights.npz", openvino.get("weights_sha256"))
        file_hash_matches("combined_reference.npz", openvino.get("combined_reference_sha256"))
        file_hash_matches("correction_dynamic.xml", openvino.get("ir_xml_sha256"))
        file_hash_matches("correction_dynamic.bin", openvino.get("ir_bin_sha256"))
        for value, expected_name in ((openvino.get("training_phase_path"), "training_phase.json"), (openvino.get("weights_path"), "correction_weights.npz"), (openvino.get("combined_reference_path"), "combined_reference.npz"), (openvino.get("ir_xml"), "correction_dynamic.xml"), (openvino.get("ir_bin"), "correction_dynamic.bin")):
            require(path_is(root, value, expected_name), f"OpenVINO path mismatch for {expected_name}")
        require(openvino.get("device") == "CPU" and openvino.get("npu_required") is False, "OpenVINO device contract mismatch")
        require(openvino.get("all_outputs_finite") is True, "OpenVINO output finite gate failed")
        require(openvino.get("fresh_weights_constructor_used") is True and openvino.get("fresh_combined_reference_used") is True, "OpenVINO fresh-artifact binding missing")
        require(openvino.get("initializer_exposed_to_offline_rows") is True and openvino.get("offline_role") == "reused_development_guard", "OpenVINO scope disclosure mismatch")
        tolerance = openvino.get("cpu_tolerance")
        require(finite(tolerance) and tolerance == 1.0e-4, "OpenVINO CPU tolerance mismatch")
        for field in ("correction_max_abs_error", "combined_max_abs_error", "copack_vs_per_state_padded_max_abs_error", "actual_count_vs_padded_valid_max_abs_error", "actual_count_vs_copack_valid_max_abs_error"):
            require(finite_nonnegative(openvino.get(field)) and openvino[field] <= tolerance, f"OpenVINO {field} gate failed")
        require(openvino.get("actual_count_fixed74_valid_output_invariance_verified") is True, "actual-count/fixed-74 invariance missing")
        require(openvino.get("four_state_copack_invariance_verified") is True, "co-pack invariance missing")
        require(openvino.get("witness_state_count") == 4, "OpenVINO witness state count mismatch")
        require(isinstance(openvino.get("witness_action_counts"), list) and len(openvino["witness_action_counts"]) == 4, "OpenVINO witness counts missing")
        for field in ("export_seconds", "compile_seconds", "inference_seconds", "invariance_inference_seconds"):
            require(finite_nonnegative(openvino.get(field)), f"OpenVINO {field} invalid")
    require_false(openvino, "OpenVINO gate", "npu_required", "validation_or_test_seed_loaded", "game_evaluation_run")

    require(isinstance(offline_extraction, dict), "missing or malformed offline extraction")
    if isinstance(offline_extraction, dict):
        require(offline_extraction.get("status") == "offline_extraction_complete", "offline extraction status mismatch")
        require(offline_extraction.get("dataset_sha256") == DATASET_SHA256, "offline extraction dataset mismatch")
        require(offline_extraction.get("rows_loaded") == [2161, 2660] and offline_extraction.get("row_count") == 500, "offline extraction row role mismatch")
        require(offline_extraction.get("episodes") == [13, 14] and offline_extraction.get("target_eligible_count") == 494, "offline extraction episode/eligible mismatch")
        require(offline_extraction.get("dagger_behavior_bootstrap_used") is False, "offline DAgger behavior bootstrap used")
        require(offline_extraction.get("initializer_exposed_to_offline_rows") is True and offline_extraction.get("offline_role") == "reused_development_guard", "offline extraction scope mismatch")
        require(offline_extraction.get("offline_loaded_before_candidate_freeze") is False, "offline loaded before candidate freeze")
        for digest_name in ("eligible_rows_sha256", "targets_sha256", "selected_old_q_sha256", "target_residual_sha256"):
            require(isinstance(offline_extraction.get(digest_name), str) and len(offline_extraction[digest_name]) == 64, f"offline extraction missing {digest_name}")
        file_hash_matches("offline.npz", offline_extraction.get("npz_sha256"))
        require(path_is(root, offline_extraction.get("npz_path"), "offline.npz"), "offline extraction NPZ path mismatch")
    require_false(offline_extraction, "offline extraction", "dagger_behavior_bootstrap_used", "offline_loaded_before_candidate_freeze", "validation_or_test_seed_loaded", "game_seed_loaded")

    require(isinstance(offline, dict), "missing or malformed offline gate")
    if isinstance(offline, dict):
        require(offline.get("status") == "Q1-offline-promoted", "offline promotion status mismatch")
        require(offline.get("rows") == [2161, 2660] and offline.get("row_count") == 500, "offline exact rows mismatch")
        require(offline.get("episodes") == [13, 14] and offline.get("eligible_target_count") == 494, "offline episode/eligible mismatch")
        require(offline.get("offline_role") == "reused_development_guard", "offline role mismatch")
        require(offline.get("offline_is_held_out_generalization") is False, "offline incorrectly claims held-out generalization")
        require(offline.get("initializer_exposed") is True and offline.get("initializer_exposed_to_offline_rows") is True, "offline initializer exposure missing")
        require(offline.get("target_metrics_use_eligible_rows_only") is True, "offline target denominator mismatch")
        require(offline.get("top1_and_rms_use_all_valid_actions_in_all_500_rows") is True, "offline top1/RMS denominator mismatch")
        require(offline.get("device") == "CPU" and offline.get("npu_required") is False, "offline device mismatch")
        file_hash_matches("training_phase.json", offline.get("training_phase_sha256"))
        file_hash_matches("openvino_gate.json", offline.get("openvino_gate_sha256"))
        file_hash_matches("correction_weights.npz", offline.get("weights_sha256"))
        file_hash_matches("offline.npz", offline.get("offline_npz_sha256"))
        file_hash_matches("correction_dynamic.xml", offline.get("ir_xml_sha256"))
        file_hash_matches("correction_dynamic.bin", offline.get("ir_bin_sha256"))
        require(path_is(root, offline.get("offline_npz_path"), "offline.npz"), "offline NPZ path mismatch")
        gates = offline.get("gates")
        require(isinstance(gates, dict) and set(gates) == GATE_NAMES and all(value is True for value in gates.values()), "offline gate map mismatch")
        thresholds = offline.get("thresholds", {})
        require(thresholds == {"minimum_huber_improvement_fraction": 0.15, "minimum_correlation": 0.2, "minimum_sign_agreement": 0.6, "minimum_top1_agreement": 0.95, "maximum_top1_agreement_exclusive": 0.995, "minimum_top1_changes": 3, "maximum_correction_rms": 0.25}, "offline thresholds mismatch")
        zero_huber = offline.get("zero_correction_selected_huber")
        candidate_huber = offline.get("candidate_selected_huber")
        improvement = offline.get("selected_huber_improvement_fraction")
        require(finite(zero_huber) and zero_huber > 0 and finite_nonnegative(candidate_huber), "offline Huber metrics invalid")
        require(finite(improvement) and math.isclose(improvement, (zero_huber - candidate_huber) / zero_huber, rel_tol=1e-9, abs_tol=1e-12) and improvement >= 0.15, "offline Huber improvement mismatch")
        require(finite(offline.get("selected_residual_correlation")) and offline["selected_residual_correlation"] >= 0.2, "offline correlation gate failed")
        require(offline.get("sign_target_abs_floor") == 0.1 and integer_nonnegative(offline.get("sign_subset_count")) and offline["sign_subset_count"] > 0, "offline sign subset mismatch")
        require(finite(offline.get("sign_agreement")) and offline["sign_agreement"] >= 0.6, "offline sign gate failed")
        agreements = offline.get("combined_top1_agreements")
        changes = offline.get("combined_top1_changes")
        agreement = offline.get("combined_top1_agreement")
        require(integer_nonnegative(agreements) and integer_nonnegative(changes) and agreements + changes == 500, "offline top-1 count mismatch")
        require(finite(agreement) and math.isclose(agreement, agreements / 500, abs_tol=1e-12) and 0.95 <= agreement < 0.995 and changes >= 3, "offline top-1 gate mismatch")
        require(finite_nonnegative(offline.get("correction_rms")) and offline["correction_rms"] <= 0.25, "offline correction RMS failed")
        changed = offline.get("changed_state_diagnostics")
        require(isinstance(changed, list) and len(changed) == changes and len(changed) >= 3, "changed-state diagnostics mismatch")
        if isinstance(changed, list):
            numeric_fields = ("old_top2_margin", "correction_new_minus_old_action", "old_q_at_old_action", "old_q_at_new_action", "combined_q_at_old_action", "combined_q_at_new_action", "combined_new_minus_old_action")
            for record in changed:
                require(isinstance(record, dict), "changed-state diagnostic malformed")
                if isinstance(record, dict):
                    require(isinstance(record.get("source_row"), int) and 2161 <= record["source_row"] <= 2660, "changed-state row escaped role")
                    require(isinstance(record.get("old_action_index_one_based"), int) and isinstance(record.get("candidate_action_index_one_based"), int), "changed-state action index missing")
                    require(all(finite(record.get(field)) for field in numeric_fields), "changed-state numeric diagnostic invalid")
        require(offline.get("checkpoint_selection_performed") is False and offline.get("earlier_checkpoint_rollback") is False, "offline checkpoint selection/rollback occurred")
        for field in ("compile_seconds", "inference_seconds"):
            require(finite_nonnegative(offline.get(field)), f"offline {field} invalid")
        require(offline.get("logical_network_calls") == 500, "offline logical call count mismatch")
        require(integer_nonnegative(offline.get("candidate_evaluations")) and offline["candidate_evaluations"] >= 500, "offline candidate accounting invalid")
    require_false(offline, "offline gate", "npu_required", "checkpoint_selection_performed", "earlier_checkpoint_rollback", "validation_or_test_seed_loaded", "game_evaluation_run")

    require(known_forbidden_flags_clear(artifacts), "known forbidden validation/game/selection flag is non-false")
    unique_failures = list(dict.fromkeys(failures))
    passed = not unique_failures
    return {
        "assessment": "Q1 strict artifact/provenance/offline-scope assessment",
        "status": "assessment-pass" if passed else "assessment-fail",
        "success": passed,
        "failures": unique_failures,
        "promotion": "Q1-offline-promoted" if passed else "Q1-offline-rejected",
        "output_directory": str(root),
        "artifacts": artifacts,
        "scope": "offline reused-development safety evidence only",
        "monitor_consulted": False,
        "final_result_written": False,
        "initializer_exposed_to_offline_rows": True,
        "offline_is_held_out_generalization": False,
        "game_strength_evidence": False,
        "model_beat_claim": False,
        "game_evaluation_authorized": False,
        "validation_seed_used": False,
        "sealed_test_seed_used": False,
        "sealed_test_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    output = args.output_directory.resolve() / "assessment.json"
    if output.exists():
        raise RuntimeError("refusing to overwrite Q1 assessment")
    result = assess(args.output_directory)
    temporary = output.with_suffix(".json.tmp")
    if temporary.exists():
        raise RuntimeError("stale Q1 assessment temporary artifact")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()
