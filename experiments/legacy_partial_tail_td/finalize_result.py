from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "legacy_partial_tail_td_P1"
EXPECTED_PARENT_COMMIT = "c9ab1a94342752dc135725beabf4a6b10d73f92d"
DATASET_SHA256 = "e0d79e38daebb667bd8c248f5f64b8e5241a4ed56a29d31ffb4ee41bd0c26b8d"
CHECKPOINT_SHA256 = "7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1"
BASELINE_SHA256 = "2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba"
REPORT_SHA256 = "a079330917571824fdbb0dd92d37db92dc1df9701012206bb27bc672d24ca906"
ROW_ORDER_SHA256 = "7f8a24abc5000ad1cc13ee4c4d7b5227caf57923686fd17aea83ef664550efae"
ELIGIBLE_ROWS = tuple(
    row
    for start in (1, 251, 501, 751, 1001, 1251)
    for row in range(start, start + 247)
)
EXPECTED_EVALUATION_ORDER = (
    (5756, "candidate"),
    (5756, "canonical_old_baseline"),
    (5757, "candidate"),
    (5757, "canonical_old_baseline"),
)
REQUIRED_FALSE_FIELDS = {
    "eligibility": ("offline_rows_loaded", "validation_or_test_seed_loaded"),
    "row_freeze": ("validation_or_test_seed_loaded",),
    "training_extraction": ("offline_rows_loaded", "validation_or_test_seed_loaded"),
    "training_phase": (
        "original_checkpoint_overwritten",
        "existing_weight_artifact_overwritten",
        "validation_or_test_seed_loaded",
        "game_evaluation_run",
    ),
    "openvino_gate": (
        "existing_weight_artifact_overwritten",
        "validation_or_test_seed_loaded",
        "game_evaluation_run",
    ),
    "offline_extraction": ("training_rows_loaded", "validation_or_test_seed_loaded"),
    "offline_gate": (
        "checkpoint_selection_performed",
        "earlier_checkpoint_rollback",
        "validation_or_test_seed_loaded",
        "game_evaluation_run",
    ),
    "development": (
        "statistical_model_beat_claim",
        "sealed_test_authorized",
        "validation_or_test_seed_loaded",
    ),
}
REQUIRED_OMISSION_CLASS_FIELDS = {
    "role": ("eligibility", "rows_loaded"),
    "order": ("row_freeze", "ordered_rows"),
    "RNG": ("row_freeze", "rng"),
    "hash": ("training_extraction", "npz_sha256"),
    "provenance": ("training_phase", "freeze_path"),
    "accounting": ("development", "accounting"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_optional(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, json.JSONDecodeError):
        return None


def finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def path_is(root: Path, value: Any, expected_name: str) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        return Path(value).resolve() == (root / expected_name).resolve()
    except OSError:
        return False


def all_forbidden_flags_clear(value: Any) -> bool:
    forbidden_true = {
        "validation_or_test_seed_loaded",
        "validation_or_test_data_used",
        "validation_seed_used",
        "sealed_test_seed_used",
        "held_out_test_seeds_used",
        "sealed_test_authorized",
        "rescue_run_authorized",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden_true and child is not False:
                return False
            if not all_forbidden_flags_clear(child):
                return False
    elif isinstance(value, list):
        return all(all_forbidden_flags_clear(child) for child in value)
    return True


def assess(root: Path) -> dict[str, Any]:
    root = root.resolve()
    names = (
        "freeze.json",
        "eligibility.json",
        "row_freeze.json",
        "training_extraction.json",
        "training_phase.json",
        "training_failure.json",
        "openvino_gate.json",
        "offline_extraction.json",
        "offline_gate.json",
        "development.json",
    )
    artifacts = {name.removesuffix(".json"): read_optional(root / name) for name in names}
    freeze = artifacts["freeze"]
    eligibility = artifacts["eligibility"]
    row_freeze = artifacts["row_freeze"]
    training_extraction = artifacts["training_extraction"]
    training = artifacts["training_phase"]
    training_failure = artifacts["training_failure"]
    openvino = artifacts["openvino_gate"]
    offline_extraction = artifacts["offline_extraction"]
    offline = artifacts["offline_gate"]
    development = artifacts["development"]
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    def file_hash_matches(name: str, expected: Any) -> None:
        path = root / name
        require(path.is_file(), f"missing artifact file {name}")
        require(
            isinstance(expected, str) and path.is_file() and sha256_file(path) == expected,
            f"artifact SHA-256 mismatch for {name}",
        )

    for omission_class, (artifact_name, field) in REQUIRED_OMISSION_CLASS_FIELDS.items():
        artifact = artifacts[artifact_name]
        if isinstance(artifact, dict):
            require(
                field in artifact,
                f"{artifact_name} missing required {omission_class} field {field}",
            )

    for artifact_name, fields in REQUIRED_FALSE_FIELDS.items():
        artifact = artifacts[artifact_name]
        if not isinstance(artifact, dict):
            continue
        for field in fields:
            require(
                field in artifact,
                f"{artifact_name} missing required forbidden/scope field {field}",
            )
            require(
                artifact.get(field) is False,
                f"{artifact_name} forbidden/scope field {field} is not exactly false",
            )

    require(isinstance(freeze, dict), "missing or malformed freeze")
    if isinstance(freeze, dict):
        source_commit = freeze.get("source_commit")
        require(freeze.get("experiment") == EXPERIMENT_ID, "freeze experiment mismatch")
        require(
            isinstance(source_commit, str) and len(source_commit) == 40,
            "freeze source_commit is not a full commit",
        )
        require(
            freeze.get("authorized_hardening_commit") == source_commit,
            "freeze does not bind the explicit authorized hardening commit",
        )
        require(
            freeze.get("expected_parent_commit") == EXPECTED_PARENT_COMMIT,
            "freeze audited-parent binding mismatch",
        )
        require(freeze.get("repository_clean") is True, "freeze repository was not clean")
        require(path_is(root, freeze.get("output_directory"), "."), "freeze output path mismatch")
        audit = freeze.get("source_fingerprint_audit")
        require(isinstance(audit, dict) and audit.get("valid") is True, "source fingerprint audit missing or rejected")
        if isinstance(audit, dict):
            binding = audit.get("repository_binding", {})
            fingerprint = audit.get("fingerprint", {})
            require(binding.get("head") == source_commit, "fingerprint audit HEAD mismatch")
            require(binding.get("authorized_commit") == source_commit, "fingerprint authorization mismatch")
            require(binding.get("parent") == EXPECTED_PARENT_COMMIT, "fingerprint parent mismatch")
            require(binding.get("repository_clean") is True, "fingerprint audit tree not clean")
            require(
                fingerprint.get("source_sha256") == freeze.get("source_fingerprint", {}).get("source_sha256"),
                "fingerprint aggregate provenance mismatch",
            )
            require(
                fingerprint.get("manifest_sha256") == freeze.get("manifest_sha256"),
                "fingerprint manifest provenance mismatch",
            )
        require(isinstance(freeze.get("harness_sha256"), str), "freeze harness aggregate missing")
        harness_files = freeze.get("harness_files")
        require(isinstance(harness_files, list) and len(harness_files) >= 19, "freeze harness file inventory incomplete")
        inputs = freeze.get("immutable_inputs")
        expected_inputs = {
            "checkpoint": CHECKPOINT_SHA256,
            "dataset": DATASET_SHA256,
            "canonical baseline weights": BASELINE_SHA256,
            "authorization report": REPORT_SHA256,
        }
        require(isinstance(inputs, list), "freeze immutable input inventory missing")
        if isinstance(inputs, list):
            observed_inputs = {item.get("label"): item for item in inputs if isinstance(item, dict)}
            require(set(observed_inputs) == set(expected_inputs), "freeze immutable input roles mismatch")
            for label, digest in expected_inputs.items():
                item = observed_inputs.get(label, {})
                require(item.get("expected") == digest and item.get("observed") == digest, f"freeze {label} hash mismatch")
        constants = freeze.get("constants", {})
        require(constants.get("updates") == 300, "freeze update count mismatch")
        require(constants.get("data_order_rng") == "Xoshiro(0x1313_2026)", "freeze RNG mismatch")
        require(constants.get("trainable_parameter_count") == 2_949_508, "freeze trainable count mismatch")
        require(constants.get("optimizer_moment_elements") == 5_899_016, "freeze moment count mismatch")

    require(isinstance(eligibility, dict), "missing or malformed eligibility")
    if isinstance(eligibility, dict):
        require(eligibility.get("status") == "training_eligibility_complete", "eligibility status mismatch")
        require(eligibility.get("dataset_sha256") == DATASET_SHA256, "eligibility dataset mismatch")
        require(eligibility.get("rows_loaded") == [1, 1500], "eligibility row role mismatch")
        require(eligibility.get("episode_ids_allowed") == [1, 2, 3, 4, 5, 6], "eligibility episodes mismatch")
        require(eligibility.get("seeds_allowed") == [5742, 5743, 5744, 5745, 5746, 5747], "eligibility seeds mismatch")
        require(eligibility.get("eligible_count") == len(ELIGIBLE_ROWS), "eligibility count mismatch")
        require(eligibility.get("eligible_rows") == list(ELIGIBLE_ROWS), "eligibility exact row list mismatch")
        require(eligibility.get("offline_rows_loaded") is False, "eligibility loaded offline rows")
        require(eligibility.get("validation_or_test_seed_loaded") is False, "eligibility loaded forbidden seed")

    require(isinstance(row_freeze, dict), "missing or malformed row freeze")
    frozen_rows: list[int] = []
    if isinstance(row_freeze, dict):
        frozen_rows = row_freeze.get("ordered_rows") if isinstance(row_freeze.get("ordered_rows"), list) else []
        order_digest = hashlib.sha256(",".join(map(str, frozen_rows)).encode()).hexdigest()
        require(row_freeze.get("status") == "training_row_freeze_complete", "row freeze status mismatch")
        require(row_freeze.get("dataset_sha256") == DATASET_SHA256, "row freeze dataset mismatch")
        require(row_freeze.get("rng") == "Xoshiro(0x1313_2026)", "row freeze RNG mismatch")
        require(row_freeze.get("row_count") == 300 and len(frozen_rows) == 300, "row freeze count mismatch")
        require(len(set(frozen_rows)) == 300 and set(frozen_rows) <= set(ELIGIBLE_ROWS), "row freeze role/uniqueness mismatch")
        require(order_digest == ROW_ORDER_SHA256, "row freeze exact order mismatch")
        require(row_freeze.get("ordered_rows_sha256") == ROW_ORDER_SHA256, "row freeze order hash missing/mismatch")
        require(row_freeze.get("validation_or_test_seed_loaded") is False, "row freeze forbidden-seed flag mismatch")
        require(path_is(root, row_freeze.get("eligibility_path"), "eligibility.json"), "row freeze eligibility path mismatch")
        file_hash_matches("eligibility.json", row_freeze.get("eligibility_sha256"))

    require(isinstance(training_extraction, dict), "missing or malformed training extraction")
    if isinstance(training_extraction, dict):
        require(training_extraction.get("status") == "training_extraction_complete", "training extraction status mismatch")
        require(training_extraction.get("dataset_sha256") == DATASET_SHA256, "training extraction dataset mismatch")
        require(training_extraction.get("row_count") == 300, "training extraction count mismatch")
        require(training_extraction.get("source_rows") == frozen_rows, "training extraction row order mismatch")
        require(training_extraction.get("episodes") == [1, 2, 3, 4, 5, 6], "training extraction episodes mismatch")
        require(training_extraction.get("seeds") == [5742, 5743, 5744, 5745, 5746, 5747], "training extraction seeds mismatch")
        require(training_extraction.get("targets_frozen_before_training") is True, "targets not frozen")
        require(training_extraction.get("updated_model_used_for_targets") is False, "updated target leakage")
        file_hash_matches("row_freeze.json", training_extraction.get("row_freeze_sha256"))
        file_hash_matches("training.npz", training_extraction.get("npz_sha256"))
        require(path_is(root, training_extraction.get("row_freeze_path"), "row_freeze.json"), "training extraction row-freeze path mismatch")
        require(path_is(root, training_extraction.get("npz_path"), "training.npz"), "training extraction NPZ path mismatch")

    require(training_failure is None, "training failure artifact exists")
    require(isinstance(training, dict), "missing or malformed training phase")
    if isinstance(training, dict):
        require(training.get("status") == "training_phase_complete", "training status mismatch")
        require(training.get("source_rows") == frozen_rows, "training source row order mismatch")
        require(training.get("update_count") == 300, "training update count mismatch")
        file_hash_matches("training.npz", training.get("training_subset_sha256"))
        file_hash_matches("row_freeze.json", training.get("row_freeze_sha256"))
        file_hash_matches("freeze.json", training.get("freeze_sha256"))
        require(path_is(root, training.get("training_subset_path"), "training.npz"), "training subset path mismatch")
        require(path_is(root, training.get("row_freeze_path"), "row_freeze.json"), "training row-freeze path mismatch")
        require(path_is(root, training.get("freeze_path"), "freeze.json"), "training freeze path mismatch")
        require(training.get("source_checkpoint_sha256") == CHECKPOINT_SHA256, "training initializer mismatch")
        step0 = training.get("step0", {})
        require(step0.get("witness_source_row") == 1055, "step-0 witness row mismatch")
        require(step0.get("witness_candidate_count") == 52, "step-0 witness count mismatch")
        require(step0.get("witness_selected_action") == 11, "step-0 witness action mismatch")
        require(
            step0.get("witness_chunks")
            == [
                list(range(1, 17)),
                list(range(17, 33)),
                list(range(33, 49)),
                list(range(49, 53)),
            ],
            "step-0 witness chunks mismatch",
        )
        require(step0.get("split_tail_max_abs_error", math.inf) <= 1.0e-6, "split-tail equivalence failed")
        require(step0.get("stored_old_q_max_abs_error", math.inf) <= 1.0e-2, "stored old-Q equivalence failed")
        require(training.get("gradient_elements") == 2_949_508, "gradient element count mismatch")
        require(training.get("gradient_elements_every_update") is True, "gradient coverage incomplete")
        require(training.get("optimizer_moment_elements") == 5_899_016, "optimizer moment count mismatch")
        finite_differences = training.get("finite_differences")
        expected_fd = {
            "score_net.layer_3.bias",
            "score_net.layer_1.weight",
            "board_net.resblocks.layer_31.layer_1.weight",
        }
        require(
            isinstance(finite_differences, list)
            and {record.get("path") for record in finite_differences} == expected_fd
            and all(
                record.get("passed") is True
                and finite_nonnegative(record.get("abs_error"))
                and finite_nonnegative(record.get("tolerance"))
                and record["abs_error"] <= record["tolerance"]
                for record in finite_differences
            ),
            "finite-difference gate mismatch",
        )
        require(training.get("trainable_array_count") == 34, "trainable leaf count mismatch")
        require(training.get("frozen_parameter_array_count") == 250, "frozen leaf count mismatch")
        require(training.get("running_state_array_count") == 69, "state leaf count mismatch")
        require(training.get("export_array_count") == 353, "export array count mismatch")
        require(training.get("frozen_parameter_sha_unchanged") is True, "frozen SHA changed")
        require(training.get("running_state_sha_unchanged") is True, "running state changed")
        require(isinstance(training.get("changed_trainable_arrays"), list) and training["changed_trainable_arrays"], "no trainable array changed")
        require(training.get("final_merge_max_abs_error", math.inf) <= 1.0e-6, "fresh merge mismatch")
        require(training.get("first_update_seconds", math.inf) <= 180.0, "first update time gate failed")
        warm = training.get("warm_update_seconds")
        require(isinstance(warm, list) and len(warm) == 6 and all(finite_nonnegative(value) and value <= 15 for value in warm), "warm update gate mismatch")
        require(training.get("warm_median_seconds", math.inf) <= 4.5, "warm median gate failed")
        require(training.get("projected_total_seconds", math.inf) <= 2100.0, "projection gate failed")
        updates = training.get("update_records")
        require(isinstance(updates, list) and len(updates) == 300, "update ledger count mismatch")
        if isinstance(updates, list) and len(updates) == 300:
            require([record.get("update") for record in updates] == list(range(1, 301)), "update ledger order mismatch")
            require([record.get("source_row") for record in updates] == frozen_rows, "update ledger source rows mismatch")
            require(all(record.get("gradient_elements") == 2_949_508 for record in updates), "update gradient accounting mismatch")
            require(all(finite_nonnegative(record.get("timing", {}).get("seconds")) for record in updates), "update timing accounting invalid")
            require(all(isinstance(record.get("loss"), (int, float)) and math.isfinite(record["loss"]) for record in updates), "update loss ledger invalid")
            require(all(record["timing"]["seconds"] <= 15.0 for record in updates[1:]), "later update exceeded 15 seconds")
        file_hash_matches("candidate_merged.jld2", training.get("candidate_checkpoint_sha256"))
        file_hash_matches("candidate_weights.npz", training.get("weights_sha256"))
        file_hash_matches("final_reference.npz", training.get("reference_sha256"))
        require(path_is(root, training.get("candidate_checkpoint"), "candidate_merged.jld2"), "candidate checkpoint path mismatch")
        require(path_is(root, training.get("weights_path"), "candidate_weights.npz"), "candidate weights path mismatch")
        require(path_is(root, training.get("reference_path"), "final_reference.npz"), "reference path mismatch")

    require(isinstance(openvino, dict), "missing or malformed OpenVINO gate")
    if isinstance(openvino, dict):
        require(openvino.get("status") == "openvino_gate_pass", "OpenVINO status mismatch")
        require(openvino.get("candidate_count") == 52 and openvino.get("historical_chunk") == 16, "OpenVINO witness/chunk mismatch")
        file_hash_matches("training_phase.json", openvino.get("training_phase_sha256"))
        file_hash_matches("candidate_weights.npz", openvino.get("weights_sha256"))
        file_hash_matches("final_reference.npz", openvino.get("reference_sha256"))
        require(path_is(root, openvino.get("weights_path"), "candidate_weights.npz"), "OpenVINO weights path mismatch")
        require(openvino.get("fresh_weights_constructor_used") is True, "OpenVINO fresh-weight binding missing")
        for field, names in (("fixed_ir", ("candidate_fixed16.xml", "candidate_fixed16.bin")), ("dynamic_tail_ir", ("candidate_tail_dynamic.xml", "candidate_tail_dynamic.bin"))):
            record = openvino.get(field, {})
            require(path_is(root, record.get("xml"), names[0]), f"{field} XML path mismatch")
            require(path_is(root, record.get("bin"), names[1]), f"{field} BIN path mismatch")
            file_hash_matches(names[0], record.get("sha256"))
            file_hash_matches(names[1], record.get("bin_sha256"))
        require(openvino.get("cpu", {}).get("max_abs_error", math.inf) <= 1.0e-4, "CPU equivalence failed")
        require(openvino.get("npu", {}).get("aggregate_max_abs_error", math.inf) <= 1.0e-2, "NPU equivalence failed")
        for role in ("cpu", "npu"):
            record = openvino.get(role, {})
            compile_name = "compile_seconds" if role == "cpu" else "compile_seconds_including_cpu_tail"
            inference_name = "inference_seconds" if role == "cpu" else "inference_seconds_including_cpu_tail"
            require(finite_nonnegative(record.get(compile_name)) and finite_nonnegative(record.get(inference_name)), f"OpenVINO {role} time accounting invalid")
            require([chunk.get("size") for chunk in record.get("chunks", [])] == [16, 16, 16, 4], f"OpenVINO {role} chunk accounting mismatch")

    require(isinstance(offline_extraction, dict), "missing or malformed offline extraction")
    if isinstance(offline_extraction, dict):
        require(offline_extraction.get("status") == "offline_extraction_complete", "offline extraction status mismatch")
        require(offline_extraction.get("dataset_sha256") == DATASET_SHA256, "offline dataset mismatch")
        require(offline_extraction.get("rows_loaded") == [1501, 2000], "offline row role mismatch")
        require(offline_extraction.get("row_count") == 500, "offline row count mismatch")
        require(offline_extraction.get("episodes") == [7, 8], "offline episodes mismatch")
        require(offline_extraction.get("seeds") == [5748, 5749], "offline seeds mismatch")
        file_hash_matches("offline.npz", offline_extraction.get("npz_sha256"))
        require(path_is(root, offline_extraction.get("npz_path"), "offline.npz"), "offline extraction NPZ path mismatch")

    require(isinstance(offline, dict), "missing or malformed offline gate")
    if isinstance(offline, dict):
        require(offline.get("status") == "offline_gate_pass", "offline status mismatch")
        require(offline.get("rows") == [1501, 2000] and offline.get("row_count") == 500, "offline exact row role mismatch")
        require(offline.get("episodes") == [7, 8] and offline.get("seeds") == [5748, 5749], "offline episode/seed role mismatch")
        file_hash_matches("offline.npz", offline.get("offline_npz_sha256"))
        require(path_is(root, offline.get("offline_npz_path"), "offline.npz"), "offline gate NPZ path mismatch")
        file_hash_matches("training_phase.json", offline.get("training_phase_sha256"))
        file_hash_matches("openvino_gate.json", offline.get("openvino_gate_sha256"))
        file_hash_matches("candidate_weights.npz", offline.get("candidate_weights_sha256"))
        require(offline.get("explicit_fresh_weight_constructor") is True, "offline fresh-weight binding missing")
        require(offline.get("all_q_finite") is True and offline.get("top1_agreement", 0) >= 0.95, "offline finite/top-1 gate failed")
        require(offline.get("candidate_evaluations", -1) >= 0, "offline candidate accounting invalid")
        require(offline.get("logical_network_calls") == 500, "offline logical-call accounting mismatch")
        require(offline.get("physical_network_calls", -1) >= 500, "offline physical-call accounting invalid")
        require(all(finite_nonnegative(offline.get(name)) for name in ("compile_seconds", "inference_loop_seconds", "wall_seconds")), "offline time accounting invalid")

    require(isinstance(development, dict), "missing or malformed development gate")
    if isinstance(development, dict):
        require(development.get("status") == "P1-development-pass", "development status mismatch")
        require(development.get("seeds") == [5756, 5757], "development seed order mismatch")
        require(development.get("next_count") == 5 and development.get("hold_enabled") is True, "development NEXT/HOLD mismatch")
        require(development.get("candidate_order") == "stable_node_key", "development candidate order mismatch")
        require(development.get("max_pieces") == 100 and development.get("lookahead_expansions") == 0, "development budget mismatch")
        require(development.get("logical_full_candidate_score_calls_per_decision") == 1, "development logical-call contract mismatch")
        for name in ("training_phase", "openvino_gate", "offline_gate"):
            file_hash_matches(f"{name}.json", development.get(f"{name}_sha256"))
        compilation = development.get("compilation", {})
        file_hash_matches("candidate_weights.npz", compilation.get("candidate_weights_sha256"))
        require(compilation.get("baseline_weights_sha256") == BASELINE_SHA256, "development baseline hash mismatch")
        require(compilation.get("explicit_weight_constructor") is True, "development explicit weight binding missing")
        require(finite_nonnegative(compilation.get("candidate_seconds")) and finite_nonnegative(compilation.get("baseline_seconds")), "development compile accounting invalid")
        evaluations = development.get("evaluations")
        require(isinstance(evaluations, list) and len(evaluations) == 4, "development evaluation count mismatch")
        if isinstance(evaluations, list) and len(evaluations) == 4:
            require(tuple((record.get("seed"), record.get("role")) for record in evaluations) == EXPECTED_EVALUATION_ORDER, "development seed/role order mismatch")
            for record in evaluations:
                require(record.get("steps") == 100 and record.get("game_over") is False, "development episode incomplete")
                for name in ("candidate_evaluations", "logical_network_calls", "physical_network_calls"):
                    require(isinstance(record.get(name), int) and record[name] >= 0, f"development {name} invalid")
                for name in ("generation_seconds", "inference_seconds", "wall_seconds"):
                    require(finite_nonnegative(record.get(name)), f"development {name} invalid")
        require(development.get("both_strictly_positive") is True, "development sign gate failed")
        require(development.get("paired_mean_difference", -math.inf) >= 500, "development mean gate failed")
        require(development.get("paired_median_difference", -math.inf) > 0, "development median gate failed")
        pairs = development.get("pairs")
        require(isinstance(pairs, list) and [pair.get("seed") for pair in pairs] == [5756, 5757], "development pair order mismatch")
        if isinstance(pairs, list) and len(pairs) == 2:
            require(all(pair.get("difference") == pair.get("candidate_score") - pair.get("baseline_score") for pair in pairs), "development pair arithmetic mismatch")
            require(development.get("paired_differences") == [pair["difference"] for pair in pairs], "development paired-difference link mismatch")
        accounting = development.get("accounting", {})
        for name in ("candidate_evaluations", "logical_network_calls", "physical_network_calls"):
            require(isinstance(accounting.get(name), int) and accounting[name] >= 0, f"development aggregate {name} invalid")
        for name in ("generation_seconds", "inference_seconds", "episode_wall_seconds"):
            require(finite_nonnegative(accounting.get(name)), f"development aggregate {name} invalid")
        if isinstance(evaluations, list) and len(evaluations) == 4:
            require(accounting.get("candidate_evaluations") == sum(record["candidate_evaluations"] for record in evaluations), "development candidate total mismatch")
            require(accounting.get("logical_network_calls") == sum(record["logical_network_calls"] for record in evaluations), "development logical total mismatch")
            require(accounting.get("physical_network_calls") == sum(record["physical_network_calls"] for record in evaluations), "development physical total mismatch")

    require(all_forbidden_flags_clear(artifacts), "known validation/test/rescue flag is non-false")
    unique_failures = list(dict.fromkeys(failures))
    return {
        "assessment": "P1 strict artifact/provenance assessment",
        "status": "assessment-pass" if not unique_failures else "assessment-fail",
        "success": not unique_failures,
        "failures": unique_failures,
        "output_directory": str(root),
        "artifacts": artifacts,
        "scope": "artifact assessment only; external monitor is reconciled after completion",
        "monitor_consulted": False,
        "final_result_written": False,
        "sealed_test_authorized": False,
        "validation_seed_used": False,
        "sealed_test_seed_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    output = args.output_directory.resolve() / "assessment.json"
    if output.exists():
        raise RuntimeError("refusing to overwrite P1 assessment")
    result = assess(args.output_directory)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()
