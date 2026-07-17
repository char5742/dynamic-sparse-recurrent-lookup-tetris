from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "contract.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    validate_contract(contract)
    return contract


def _exact_range(values: Any, start: int, stop: int) -> bool:
    return values == list(range(start, stop + 1))


def feature_names_digest(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def validate_contract(contract: dict[str, Any]) -> None:
    failures: list[str] = []
    roles = contract.get("data_roles", {})
    schema = contract.get("feature_schema", {})
    fit = contract.get("fit", {})
    calibration = contract.get("calibration_gate", {})
    policy = contract.get("canonical_policy", {})
    openvino = contract.get("openvino_backend", {})
    analytic_runtime = contract.get("analytic_runtime", {})
    source_runtime = contract.get("runtime_source_binding", {})
    counterfactual = contract.get("counterfactual", {})
    claims = contract.get("claims", {})

    if contract.get("schema_version") != 1:
        failures.append("schema_version must be exactly 1")
    if contract.get("experiment_id") != "online_counterfactual_top2_R1":
        failures.append("unexpected experiment_id")
    if contract.get("authorized_base_commit") != "9b2f974d3f5950e4084dc27d546fffc25a736c90":
        failures.append("authorized base changed")
    if contract.get("authorized_implementation_parent_commit") != "ddd0f6f83c0e2931cd4cd415531b30b3c80cd2bd":
        failures.append("implementation parent changed")
    if not _exact_range(roles.get("training_seeds"), 73001, 73012):
        failures.append("training seed role changed")
    if not _exact_range(roles.get("calibration_seeds"), 73101, 73106):
        failures.append("calibration seed role changed")
    if roles.get("sample_pieces") != list(range(10, 241, 10)):
        failures.append("sample piece schedule changed")
    if roles.get("minimum_training_states") != 240:
        failures.append("training minimum changed")
    if roles.get("minimum_calibration_states") != 120:
        failures.append("calibration minimum changed")
    if roles.get("conditional_development_seeds") != [5756, 5757]:
        failures.append("conditional development order changed")
    if roles.get("forbidden_validation_seeds") != list(range(8001, 8009)):
        failures.append("validation role changed")
    if (roles.get("forbidden_sealed_test_first"), roles.get("forbidden_sealed_test_last")) != (91001, 91032):
        failures.append("sealed test role changed")
    all_roles = [
        set(roles.get("training_seeds", [])),
        set(roles.get("calibration_seeds", [])),
        set(roles.get("forbidden_development_seeds", [])),
        set(roles.get("conditional_development_seeds", [])),
        set(roles.get("forbidden_validation_seeds", [])),
        set(range(roles.get("forbidden_sealed_test_first", 0), roles.get("forbidden_sealed_test_last", -1) + 1)),
    ]
    if any(all_roles[i] & all_roles[j] for i in range(len(all_roles)) for j in range(i + 1, len(all_roles))):
        failures.append("seed roles overlap")

    names = schema.get("names")
    if not isinstance(names, list) or len(names) != 70 or len(set(names)) != 70:
        failures.append("feature names must contain 70 unique entries")
    if schema.get("feature_count") != 70:
        failures.append("feature_count must be 70")
    if isinstance(names, list) and schema.get("feature_names_sha256") != feature_names_digest(names):
        failures.append("feature name digest mismatch")
    if schema.get("coefficient_count") != 71 or schema.get("coefficient_count", 100) > schema.get("maximum_coefficient_count", 99):
        failures.append("ridge coefficient budget changed")
    expected_one_hot = [f"{slot}_{piece}" for slot in schema.get("piece_slots", []) for piece in schema.get("piece_ids", [])]
    if not isinstance(names, list) or names[-42:] != expected_one_hot:
        failures.append("HOLD/NEXT one-hot order changed")
    if schema.get("piece_ids") != ["I", "O", "S", "Z", "J", "L", "T"]:
        failures.append("piece one-hot order differs from Tetris.MINOS")

    exact_values = (
        (policy.get("next_count"), 5, "NEXT count"),
        (policy.get("candidate_chunk"), 16, "candidate chunk"),
        (counterfactual.get("horizon_pieces"), 6, "counterfactual horizon"),
        (counterfactual.get("gamma"), 0.997, "gamma"),
        (counterfactual.get("reward_scale"), 600.0, "reward scale"),
        (fit.get("ridge_lambda"), 1.0, "ridge lambda"),
        (fit.get("ensemble_count"), 256, "ensemble count"),
        (fit.get("bootstrap_seed_uint64"), int("52312026", 16), "training bootstrap seed"),
        (fit.get("prediction_lower_quantile"), 0.1, "lower quantile"),
        (fit.get("override_strict_threshold"), 0.05, "override threshold"),
        (calibration.get("cluster_bootstrap_count"), 2000, "calibration bootstrap count"),
        (calibration.get("cluster_bootstrap_seed_uint64"), int("523173106", 16), "calibration bootstrap seed"),
    )
    for observed, expected, label in exact_values:
        if observed != expected:
            failures.append(f"{label} changed")
    if fit.get("quantile_method") != "linear_type7_position_1_plus_n_minus_1_p":
        failures.append("prediction quantile interpolation changed")
    if fit.get("sweep_authorized") is not False:
        failures.append("hyperparameter sweep must be prohibited")
    if counterfactual.get("trajectory_writeback") is not False:
        failures.append("counterfactual trajectory writeback must be false")
    if policy.get("allowed_actions") != ["old_top1", "old_top2"] or policy.get("fallback_action") != "old_top1":
        failures.append("bounded action/fallback contract changed")
    if policy.get("candidate_order") != "stable_node_key":
        failures.append("candidate order changed")
    weights = contract.get("immutable_inputs", {}).get("old_openvino_weight_npz_sha256")
    if weights != "2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba":
        failures.append("canonical OpenVINO weight NPZ hash changed")
    if (
        openvino.get("version") != "2026.2.1"
        or openvino.get("full_build") != "2026.2.1-21919-ede283a88e3-releases/2026/2"
        or openvino.get("weight_sha256") != weights
    ):
        failures.append("OpenVINO version/weight binding changed")
    if any(openvino.get(name) != "Float32" for name in ("input_dtype", "output_dtype", "weight_dtype")):
        failures.append("OpenVINO FP32 graph contract changed")
    complete = openvino.get("complete_chunk", {})
    if complete != {"device": "NPU", "batch_size": 16, "shape": "static", "eligible_candidate_count": 16}:
        failures.append("OpenVINO complete-16 NPU contract changed")
    tail = openvino.get("tail_chunk", {})
    if tail != {"device": "CPU", "shape": "dynamic", "batch_semantics": "actual candidate count", "minimum_candidate_count": 1, "maximum_candidate_count": 15, "padding": False}:
        failures.append("OpenVINO dynamic CPU actual-tail contract changed")
    upstream = source_runtime.get("upstream_tetrisai", {})
    if upstream != {
        "repository_relative_path": "upstream/TetrisAI",
        "head": "6fdfb1d30197246fd862b716438e998f0315c830",
        "clean_required": True,
    }:
        failures.append("upstream TetrisAI binding changed")
    vendored = source_runtime.get("vendored_tetrisai", {})
    expected_vendor = {
        "node": (
            "upstream/TetrisAI/src/core/components/node.jl",
            "experiments/online_counterfactual_top2_r1/vendor/TetrisAI/src/core/components/node.jl",
            298,
            "e98d2052f9248f5c08c1eb58adaace1bd01533f287e682bf35a2fefa1325fe82",
        ),
        "analyzer": (
            "upstream/TetrisAI/src/core/analyzer.jl",
            "experiments/online_counterfactual_top2_r1/vendor/TetrisAI/src/core/analyzer.jl",
            8114,
            "24152e2549dcc6c3c25d928454268e8baaa4d45fea31044603917cfbabbe02bc",
        ),
    }
    for name, (original, path, size, digest) in expected_vendor.items():
        record = vendored.get(name, {})
        if (
            record.get("original_relative_path"),
            record.get("vendor_relative_path"),
            record.get("bytes"),
            record.get("sha256"),
        ) != (original, path, size, digest):
            failures.append(f"vendored TetrisAI {name} binding changed")
    expected_external = {
        "vendor/Tetris/lib/curses.jl": (2767, "4dd113316c4f82a226563d7ac3237c366417211582722b3d4b4277dcb12ff922"),
        "vendor/Tetris/lib/key_input.jl": (1448, "c09571b424a49f01278f6903c0018f9a2dfc652dfd18b804c4ad2b6a37f2fc53"),
        "vendor/Tetris/lib/game.so": (49870, "d63a03f494cb0a6f1704624923c58cb521a8a45873ca400e7085c02b1bf5bf46"),
        "vendor/Tetris/lib/pdcurses.dll": (176673, "0c770aa6721aa2155bbe2ef1d0f50ad2065da399085242454486c855c1f9fe67"),
    }
    external = source_runtime.get("external_tetris_runtime")
    observed_external = (
        {
            record.get("path"): (record.get("bytes"), record.get("sha256"))
            for record in external
            if isinstance(record, dict)
        }
        if isinstance(external, list)
        else {}
    )
    if not isinstance(external, list) or len(external) != 4 or observed_external != expected_external:
        failures.append("external Tetris runtime binding changed")
    cache = source_runtime.get("python_cache_policy", {})
    if cache != {
        "repository_cache_artifacts_forbidden": True,
        "required_environment_variable": "PYTHONPYCACHEPREFIX",
        "prefix_must_be_absolute": True,
        "prefix_must_be_outside_repository": True,
    }:
        failures.append("Python cache isolation policy changed")
    if source_runtime.get("dependency_graph_digest_encoding") != "sorted relative_path + NUL + lowercase sha256 + newline":
        failures.append("dependency graph digest encoding changed")
    if source_runtime.get("runtime_closure_sha256") != "fa908de68b6deb1581818bdb45c813b06d8886bc4fe33fd010830f7eef03a0e4":
        failures.append("runtime closure aggregate changed")
    if analytic_runtime.get("production_backend") != "Python NumPy analytic ridge":
        failures.append("analytic production backend changed")
    if (analytic_runtime.get("python_version"), analytic_runtime.get("numpy_version"), analytic_runtime.get("blas_threads")) != ("3.12.13", "2.4.6", 1):
        failures.append("analytic runtime version/thread contract changed")
    if analytic_runtime.get("base_python_sha256") != "3c6a206b7d93cca823934a83732220dcffd413fd1036d9fb82eebb64599cf7f3":
        failures.append("base Python hash changed")
    if analytic_runtime.get("venv_launcher_sha256") != "5912d0884b23c0343983a864c6064242391e2265536f50b88624857e353882c9":
        failures.append("venv launcher hash changed")
    if analytic_runtime.get("thread_environment") != {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}:
        failures.append("analytic thread environment changed")
    if any(claims.get(name) is not False for name in ("model_improvement", "validation_authorized", "sealed_test_authorized", "retry_authorized", "rescue_authorized")):
        failures.append("pre-development claim authorization must be false")
    if failures:
        raise ValueError("invalid R1 contract: " + "; ".join(failures))


def atomic_write_json(path: Path, value: Any) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary artifact: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def contract_identity(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    contract = load_contract(path)
    return {
        "contract_path": str(path.resolve()),
        "contract_sha256": sha256_file(path),
        "schema_version": contract["schema_version"],
        "experiment_id": contract["experiment_id"],
        "feature_schema_version": contract["feature_schema"]["version"],
    }
