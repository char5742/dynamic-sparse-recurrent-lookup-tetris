from __future__ import annotations

"""Frozen low-memory NumPy implementation of the R1 analytic ridge fit.

This is deliberately a small numerical program, not a configurable training
framework.  Every scientific choice is read back from, and checked against,
the already-frozen R1 design.  NumPy is imported lazily so the monitored child
can durably report entry and argument validation before its largest import.
"""

import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence


# These variables affect this process and its BLAS workers only.  They must be
# set before NumPy/OpenBLAS is imported; the Julia collector environment is not
# mutated by the wrapper.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

FEATURE_COUNT = 70
COEFFICIENT_COUNT = 71
ENSEMBLE_SIZE = 256
RIDGE_LAMBDA = 1.0
BOOTSTRAP_SEED = 0x5231_2026
LOWER_QUANTILE = 0.10
OVERRIDE_THRESHOLD = 0.05
TARGET_MIN = -2.0
TARGET_MAX = 2.0
FEATURE_SCHEMA_DIGEST = (
    "7e89c16b57dcebac56e3ab4c5be161d5e5430c682e60f3b565dd23ab3b04ac44"
)
TRAINING_SCHEDULE_SHA256 = (
    "5b60a1e340b542dc8654a5c80777c254d8336aa086e51be6c2ba1251be20e5f7"
)
QUANTILE_METHOD = "linear_type7_position_1_plus_n_minus_1_p"
TRAINING_IDS = list(range(73001, 73013))
SAMPLE_PIECES = list(range(10, 241, 10))
EXPECTED_NUMPY_VERSION = "2.4.6"
EXPECTED_PYTHON_VERSION = "3.12.13"
EXPECTED_BLAS_NAME = "scipy-openblas"
EXPECTED_BLAS_VERSION = "0.3.31.188.0"
EXPECTED_OPENVINO_FULL_VERSION = "2026.2.1-21919-ede283a88e3-releases/2026/2"
PROJECTION_FORMULA = (
    "2*setup_seconds + (first32_collection_seconds-repeatability_probe_seconds)/32*432 "
    "+ repeatability_probe_seconds"
)
EXPECTED_PYTHON_RUNTIME_ORIGIN = {
    "python_runtime_origin_schema": "r1-python-runtime-origin-v1",
    "python_bridge": "PythonCall",
    "python_executable": r"D:\tetris-paper-plus\python-env\Scripts\python.exe",
    "python_base_prefix": r"C:\Users\fshuu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python",
    "python_prefix": r"D:\tetris-paper-plus\python-env",
    "pythonpath_cleared": True,
    "pythonhome_cleared": True,
    "python_no_user_site": "1",
    "openvino_module_file": r"D:\tetris-paper-plus\python-env\Lib\site-packages\openvino\__init__.py",
    "openvino_package_root": r"D:\tetris-paper-plus\python-env\Lib\site-packages\openvino",
    "openvino_package_file_count": 1102,
    "openvino_package_bytes": 234324334,
    "openvino_package_tree_sha256": "c292b25245f36e937f21b105023737be80491e70de5f24b1704ad1ced8547e43",
    "openvino_package_tree_aggregate": "sorted relative-path NUL decimal-bytes NUL lowercase-sha256 newline",
    "openvino_loaded_native_sha256": "929dd49859750bfa59c850234c8eeb872c84db05c1b60510e9a9db8b7d756a74",
}
ENGINE_DEPENDENCY_ENCODING = "sorted relative_path + NUL + lowercase sha256 + newline"
UPSTREAM_TETRISAI_HEAD = "6fdfb1d30197246fd862b716438e998f0315c830"
ENGINE_DEPENDENCY_PATHS = [
    "experiments/online_counterfactual_top2_r1/engine_adapter.jl",
    "experiments/online_counterfactual_top2_r1/vendor/TetrisAI/src/core/analyzer.jl",
    "experiments/online_counterfactual_top2_r1/vendor/TetrisAI/src/core/components/node.jl",
    "vendor/Tetris/lib/curses.jl",
    "vendor/Tetris/lib/game.so",
    "vendor/Tetris/lib/key_input.jl",
    "vendor/Tetris/lib/pdcurses.dll",
]
NODE_SOURCE_DIGEST = "e98d2052f9248f5c08c1eb58adaace1bd01533f287e682bf35a2fefa1325fe82"
ANALYZER_SOURCE_DIGEST = "24152e2549dcc6c3c25d928454268e8baaa4d45fea31044603917cfbabbe02bc"
RUNTIME_CLOSURE_DIGEST = "fa908de68b6deb1581818bdb45c813b06d8886bc4fe33fd010830f7eef03a0e4"
ALLOWED_EXCLUSION_CODES = {
    "candidate_count_lt2",
    "nonfinite_q",
    "feature_schema",
    "nonfinite_feature",
    "q_evaluation_failure",
    "q_shape",
    "branch_construction_failure",
    "branch_apply_failure",
    "branch_clone_failure",
    "nonfinite_return",
    "feature_extraction_failure",
}

_NP: Any = None


def _import_numpy() -> Any:
    global _NP
    if _NP is None:
        import numpy as np

        if platform.python_version() != EXPECTED_PYTHON_VERSION:
            raise RuntimeError(
                f"R1 requires Python {EXPECTED_PYTHON_VERSION}, "
                f"observed {platform.python_version()}"
            )
        if np.__version__ != EXPECTED_NUMPY_VERSION:
            raise RuntimeError(
                f"R1 requires NumPy {EXPECTED_NUMPY_VERSION}, observed {np.__version__}"
            )
        configuration = np.show_config(mode="dicts")
        blas = configuration.get("Build Dependencies", {}).get("blas", {})
        if (
            blas.get("name") != EXPECTED_BLAS_NAME
            or blas.get("version") != EXPECTED_BLAS_VERSION
        ):
            raise RuntimeError(
                "R1 NumPy BLAS backend changed: "
                f"expected {EXPECTED_BLAS_NAME} {EXPECTED_BLAS_VERSION}, "
                f"observed {blas.get('name')} {blas.get('version')}"
            )
        _NP = np
    return _NP


def _contract_feature_names() -> list[str]:
    path = Path(__file__).with_name("contract.json")
    document, _ = _json_snapshot(path)
    names = document["feature_schema"]["names"]
    if not isinstance(names, list) or len(names) != FEATURE_COUNT:
        raise ValueError("contract does not contain the frozen 70-feature schema")
    names = [str(value) for value in names]
    if hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest() != FEATURE_SCHEMA_DIGEST:
        raise ValueError("contract feature schema digest changed")
    return names


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    """Parse and hash the same immutable byte snapshot of one JSON input."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON input: {path}") from error
    _require(isinstance(document, dict), f"JSON root must be an object: {path}")
    return document, digest


def _expected_handoff(name: str, *, required: bool) -> str | None:
    value = os.environ.get(name)
    if value is None:
        _require(not required, f"missing upstream SHA handoff: {name}")
        return None
    _require(
        len(value) == 64 and all(character in "0123456789abcdef" for character in value),
        f"invalid upstream SHA handoff: {name}",
    )
    return value


def _atomic_json_create(path: Path, value: Any) -> None:
    destination = path.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale temporary artifact: {temporary}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # On Windows rename does not replace an existing destination.  The
        # explicit second check also makes the intended no-overwrite contract
        # evident on other supported Python 3.12 hosts.
        if destination.exists():
            raise FileExistsError(f"destination appeared during publication: {destination}")
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


class Milestones:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        self.write("script_enter")

    def write(self, stage: str, **details: Any) -> None:
        json.dump(
            {
                "phase": "fit_ridge",
                "stage": stage,
                "pid": os.getpid(),
                "time_ns": time.time_ns(),
                **details,
            },
            self._stream,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        self._stream.write("\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        self._stream.close()


def schedule_digest(schedules: Sequence[Sequence[int]]) -> str:
    payload = ";".join(",".join(str(value) for value in schedule) for schedule in schedules)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dependency_digest(records: Sequence[dict[str, Any]]) -> str:
    payload = b"".join(
        str(record["path"]).encode("utf-8")
        + b"\0"
        + str(record["sha256"]).lower().encode("ascii")
        + b"\n"
        for record in sorted(records, key=lambda record: str(record["path"]))
    )
    return hashlib.sha256(payload).hexdigest()


def _contract_document() -> dict[str, Any]:
    document = json.loads(Path(__file__).with_name("contract.json").read_text(encoding="utf-8"))
    binding = document.get("runtime_source_binding")
    _require(isinstance(binding, dict), "contract runtime-source binding missing")
    _require(
        binding.get("dependency_graph_digest_encoding") == ENGINE_DEPENDENCY_ENCODING,
        "contract dependency-graph encoding changed",
    )
    _require(
        binding.get("runtime_closure_sha256") == RUNTIME_CLOSURE_DIGEST,
        "contract runtime closure changed",
    )
    upstream = binding.get("upstream_tetrisai")
    _require(
        isinstance(upstream, dict)
        and upstream.get("head") == UPSTREAM_TETRISAI_HEAD
        and upstream.get("clean_required") is True,
        "contract upstream TetrisAI binding changed",
    )
    return document


def _expected_engine_dependency_graph(runtime_binding: dict[str, Any]) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    records: list[dict[str, Any]] = []
    for relative_path in ENGINE_DEPENDENCY_PATHS:
        path = repository.joinpath(*relative_path.split("/"))
        _require(path.is_file(), f"missing live engine dependency: {relative_path}")
        records.append(
            {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    _require(records[1]["sha256"] == ANALYZER_SOURCE_DIGEST, "live analyzer digest changed")
    _require(records[2]["sha256"] == NODE_SOURCE_DIGEST, "live Node digest changed")
    runtime_records = [
        record
        for record in records
        if record["path"] != "experiments/online_counterfactual_top2_r1/engine_adapter.jl"
    ]
    runtime_digest = _dependency_digest(runtime_records)
    _require(runtime_digest == RUNTIME_CLOSURE_DIGEST, "live runtime closure changed")
    _require(
        runtime_digest == runtime_binding.get("runtime_closure_sha256"),
        "design runtime closure differs from live dependencies",
    )
    return {
        "schema_version": "r1-engine-dependency-graph-v1",
        "encoding": ENGINE_DEPENDENCY_ENCODING,
        "upstream_tetrisai": {"head": UPSTREAM_TETRISAI_HEAD, "clean": True},
        "records": records,
        "graph_sha256": _dependency_digest(records),
        "runtime_closure_sha256": runtime_digest,
        "node_source_sha256": NODE_SOURCE_DIGEST,
        "analyzer_source_sha256": ANALYZER_SOURCE_DIGEST,
    }


def _validate_engine_dependency_graph(
    value: Any,
    *,
    synthetic: bool,
    runtime_binding: dict[str, Any],
    label: str,
) -> dict[str, Any] | None:
    if synthetic:
        _require(value is None, f"{label} synthetic engine dependency graph must be null")
        return None
    _require(isinstance(value, dict), f"{label} engine dependency graph missing")
    expected = _expected_engine_dependency_graph(runtime_binding)
    _require(set(value) == set(expected), f"{label} engine dependency graph schema mismatch")
    records = value.get("records")
    _require(isinstance(records, list) and len(records) == 7, f"{label} dependency record count mismatch")
    _require(
        all(isinstance(record, dict) and set(record) == {"path", "bytes", "sha256"}
            for record in records),
        f"{label} dependency record schema mismatch",
    )
    _require(value == expected, f"{label} engine dependency graph differs from live frozen closure")
    return value


def row_order_digest(rows: Sequence[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{_exact_int(row['episode_id'], 'row-order episode_id')},"
        f"{_exact_int(row['piece_index'], 'row-order piece_index')},"
        f"{str(row['root_state_digest'])}"
        for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_int(value: Any, label: str) -> int:
    _require(type(value) is int, f"{label} must be an exact JSON integer")
    return value


def _exact_int_list(value: Any, expected: Sequence[int], label: str) -> list[int]:
    _require(isinstance(value, list), f"{label} must be a JSON array")
    observed = [_exact_int(item, f"{label}[{index}]") for index, item in enumerate(value)]
    _require(observed == list(expected), f"{label} changed")
    return observed


def _exact_float(value: Any, expected: float, label: str) -> float:
    # JSON3 may serialize an exactly integral Float64 such as 1.0 as `1`.
    # Accept either JSON numeric token, but never bool (a Python int subclass),
    # and require exact finite numeric equality to the frozen Float64 value.
    _require(
        type(value) in (int, float),
        f"{label} must be a non-boolean JSON number",
    )
    observed = float(value)
    _require(math.isfinite(observed) and observed == expected, f"{label} changed")
    return observed


def _exact_float_list(
    value: Any, expected: Sequence[float], label: str
) -> list[float]:
    _require(isinstance(value, list), f"{label} must be a JSON array")
    _require(len(value) == len(expected), f"{label} length changed")
    return [
        _exact_float(item, float(expected[index]), f"{label}[{index}]")
        for index, item in enumerate(value)
    ]


def _nonempty_string(value: Any, label: str) -> str:
    _require(type(value) is str and bool(value), f"{label} must be a nonempty JSON string")
    return value


def _sha256_string(value: Any, label: str) -> str:
    observed = _nonempty_string(value, label)
    _require(
        len(observed) == 64
        and all(character in "0123456789abcdef" for character in observed),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return observed


def _validate_python_runtime_origin(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} missing")
    for key, expected in EXPECTED_PYTHON_RUNTIME_ORIGIN.items():
        observed = value.get(key)
        if key in {
            "python_executable", "python_base_prefix", "python_prefix",
            "openvino_module_file", "openvino_package_root",
        }:
            _require(
                os.path.normcase(os.path.abspath(str(observed)))
                == os.path.normcase(os.path.abspath(str(expected))),
                f"{label} {key} mismatch",
            )
        else:
            _require(observed == expected, f"{label} {key} mismatch")
    modules = value.get("openvino_loaded_native_modules")
    _require(isinstance(modules, list) and len(modules) == 5, f"{label} native module list mismatch")
    for index, module in enumerate(modules, 1):
        _require(isinstance(module, dict), f"{label} native module {index} invalid")
        _nonempty_string(module.get("path"), f"{label} native module {index} path")
        _require(type(module.get("bytes")) is int and module["bytes"] > 0, f"{label} native module {index} bytes invalid")
        _sha256_string(module.get("sha256"), f"{label} native module {index} digest")
    return value


def _synthetic_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    names = _contract_feature_names()
    stable_digest = hashlib.sha256(b"r1-synthetic-stable-node-key-source").hexdigest()
    rows: list[dict[str, Any]] = []
    episode_evidence: list[dict[str, Any]] = []
    for episode_index, episode_id in enumerate(TRAINING_IDS, start=1):
        for state_index, piece_index in enumerate(SAMPLE_PIECES, start=1):
            features = [
                math.sin(0.013 * feature_index * state_index)
                + math.cos(0.021 * feature_index * episode_index)
                + 0.002 * episode_index
                for feature_index in range(1, FEATURE_COUNT + 1)
            ]
            valid_action_count = 4
            q_top1 = 0.40 + 0.01 * episode_index + 0.001 * state_index
            q_top2 = q_top1 - 0.05 - 0.0001 * state_index
            q_gap = q_top1 - q_top2
            features[0] = q_top1
            features[1] = q_top2
            features[2] = q_gap
            features[6] = float(valid_action_count)
            features[-1] = 1.0
            positive = (state_index + episode_index) % 8 == 0
            advantage = (
                0.30 + 0.01 * episode_index
                if positive
                else -0.20 - 0.001 * state_index
            )
            root_state_digest = hashlib.sha256(
                f"synthetic-root-{episode_id}-{piece_index}".encode("utf-8")
            ).hexdigest()
            root_future_stream_digest = hashlib.sha256(
                f"synthetic-future-{episode_id}-{piece_index}".encode("utf-8")
            ).hexdigest()
            rows.append(
                {
                    "episode_id": episode_id,
                    "seed": episode_id,
                    "piece_index": piece_index,
                    "features": features,
                    "advantage": advantage,
                    "advantage_unclipped_A6": advantage,
                    "clipped_target": max(TARGET_MIN, min(TARGET_MAX, advantage)),
                    "g6_top1": 0.0,
                    "g6_top2": advantage,
                    "q_top1": q_top1,
                    "q_top2": q_top2,
                    "q_gap": q_gap,
                    "valid_action_count": valid_action_count,
                    "root_state_digest": root_state_digest,
                    "root_future_stream_digest": root_future_stream_digest,
                    "canonical_top1_candidate_index": 1,
                    "canonical_top2_candidate_index": 2,
                    "canonical_top1_action_digest": f"synthetic-a1-{episode_id}-{piece_index}",
                    "canonical_top2_action_digest": f"synthetic-a2-{episode_id}-{piece_index}",
                    "a1_terminal_within_horizon": False,
                    "a2_terminal_within_horizon": False,
                }
            )
        episode_evidence.append(
            {
                "seed": episode_id,
                "episode_id": episode_id,
                "canonical_pieces": 240,
                "canonical_score": 0,
                "canonical_terminal": False,
                "canonical_action_digests": [],
                "rows": 24,
                "exclusions": 0,
                "repeatability_sentinel": None,
            }
        )
    positive_fraction = sum(
        float(row["advantage_unclipped_A6"]) > 0.0 for row in rows
    ) / len(rows)
    metadata = {
        "source_role": "training",
        "feature_names": names,
        "feature_schema_digest": FEATURE_SCHEMA_DIGEST,
        "training_seeds": TRAINING_IDS,
        "role_seeds": TRAINING_IDS,
        "sample_pieces": SAMPLE_PIECES,
        "synthetic": True,
        "stable_node_key_source_sha256": stable_digest,
        "engine_dependency_graph": None,
        "counterfactual_states_completed": 288,
        "counterfactual_states_attempted": 288,
        "scheduled_slots_accounted": 288,
        "positive_advantage_fraction": positive_fraction,
        "first32_elapsed_seconds": 0.01,
        "first32_setup_seconds": 0.01,
        "first32_projected_seconds": 0.155,
        "repeatability_probe_seconds": 0.0,
        "first32_projection_limit_seconds": 3300.0,
        "first32_projection_basis_states": 432,
        "first32_projection_formula": PROJECTION_FORMULA,
        "validation_seed_used": False,
        "sealed_test_seed_used": False,
    }
    table = {
        "schema_version": "r1-training-table-v1",
        "source_role": "training",
        "metadata": metadata,
        "feature_names": names,
        "feature_schema_digest": FEATURE_SCHEMA_DIGEST,
        "training_seeds": TRAINING_IDS,
        "synthetic": True,
        "validation_seed_used": False,
        "sealed_test_seed_used": False,
        "repeatability_sentinels": [],
        "rows": rows,
    }
    # The table hash is filled only after publication by main().
    manifest = {
        "schema_version": "r1-collection-manifest-v1",
        "source_role": "training",
        "synthetic": True,
        "table_path": "",
        "table_sha256": "",
        "feature_names": names,
        "feature_schema_digest": FEATURE_SCHEMA_DIGEST,
        "stable_node_key_source_sha256": stable_digest,
        "engine_dependency_graph": None,
        "episode_count": 12,
        "row_count": 288,
        "exclusion_count": 0,
        "role_seeds": TRAINING_IDS,
        "counterfactual_states_completed": 288,
        "counterfactual_states_attempted": 288,
        "scheduled_slots_accounted": 288,
        "positive_advantage_fraction": positive_fraction,
        "first32_elapsed_seconds": 0.01,
        "first32_setup_seconds": 0.01,
        "first32_projected_seconds": 0.155,
        "repeatability_probe_seconds": 0.0,
        "first32_projection_limit_seconds": 3300.0,
        "first32_projection_basis_states": 432,
        "first32_projection_formula": PROJECTION_FORMULA,
        "exclusions": [],
        "episodes": episode_evidence,
        "real_model_or_game_loaded": False,
        "validation_seed_used": False,
        "sealed_test_seed_used": False,
        "wall_seconds": 0.02,
    }
    return table, manifest


def _load_training_table(
    path: Path, manifest_path: Path, synthetic: bool,
    expected_table_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    np = _import_numpy()
    document, actual_table_sha = _json_snapshot(path)
    manifest, actual_manifest_sha = _json_snapshot(manifest_path)
    if expected_table_sha256 is not None:
        _require(actual_table_sha == expected_table_sha256, "training table differs from producer-completion SHA")
    if expected_manifest_sha256 is not None:
        _require(actual_manifest_sha == expected_manifest_sha256, "training manifest differs from producer-completion SHA")
    names = _contract_feature_names()
    metadata = document.get("metadata")
    _require(isinstance(metadata, dict), "missing training metadata")
    _require(document.get("schema_version") == "r1-training-table-v1", "table schema mismatch")
    _require(document.get("source_role") == "training", "table role mismatch")
    _require(document.get("synthetic") is synthetic, "table synthetic role mismatch")
    _exact_int_list(document.get("training_seeds"), TRAINING_IDS, "table training seeds")
    _require(document.get("feature_names") == names, "table feature order mismatch")
    _require(document.get("feature_schema_digest") == FEATURE_SCHEMA_DIGEST, "table feature digest mismatch")
    _require(document.get("validation_seed_used") is False, "validation contamination")
    _require(document.get("sealed_test_seed_used") is False, "sealed-test contamination")
    _require(metadata.get("source_role") == "training", "metadata role mismatch")
    _require(metadata.get("feature_names") == names, "metadata feature order mismatch")
    _require(metadata.get("feature_schema_digest") == FEATURE_SCHEMA_DIGEST, "metadata feature digest mismatch")
    _exact_int_list(metadata.get("training_seeds"), TRAINING_IDS, "metadata training seeds")
    _exact_int_list(metadata.get("role_seeds"), TRAINING_IDS, "metadata role seeds")
    _exact_int_list(metadata.get("sample_pieces"), SAMPLE_PIECES, "metadata sample schedule")
    _require(metadata.get("synthetic") is synthetic, "metadata synthetic role mismatch")
    _require(metadata.get("validation_seed_used") is False, "metadata validation contamination")
    _require(metadata.get("sealed_test_seed_used") is False, "metadata sealed contamination")

    _require(manifest.get("schema_version") == "r1-collection-manifest-v1", "collection manifest schema mismatch")
    _require(manifest.get("source_role") == "training", "collection manifest role mismatch")
    _require(manifest.get("synthetic") is synthetic, "collection manifest synthetic role mismatch")
    _require(Path(str(manifest.get("table_path", ""))).resolve() == path.resolve(), "manifest table path mismatch")
    _require(manifest.get("table_sha256") == actual_table_sha, "manifest/table SHA-256 mismatch")
    _require(manifest.get("feature_names") == names, "manifest feature order mismatch")
    _require(manifest.get("feature_schema_digest") == FEATURE_SCHEMA_DIGEST, "manifest feature digest mismatch")
    _exact_int_list(manifest.get("role_seeds"), TRAINING_IDS, "manifest training seeds")
    stable_digest = _sha256_string(
        manifest.get("stable_node_key_source_sha256"),
        "stable-order source digest",
    )
    _require(metadata.get("stable_node_key_source_sha256") == stable_digest, "stable-order evidence mismatch")
    metadata_backend = metadata.get("backend_binding")
    manifest_backend = manifest.get("backend_binding")
    _require(metadata_backend == manifest_backend, "training backend binding differs between table and manifest")
    if synthetic:
        _require(metadata_backend is None, "synthetic training backend binding must be null/absent")
    else:
        _require(isinstance(metadata_backend, dict), "production training backend binding missing")
        expected_backend = {
            "old_openvino_weight_npz_sha256": "2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba",
            "old_checkpoint_sha256": "7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1",
            "openvino_version": "2026.2.1",
            "openvino_full_build": EXPECTED_OPENVINO_FULL_VERSION,
            "complete_device": "NPU",
            "tail_device": "CPU",
            "complete_batch_size": 16,
        }
        for backend_key, expected_value in expected_backend.items():
            _require(
                metadata_backend.get(backend_key) == expected_value,
                f"production training backend {backend_key} mismatch",
            )
        _sha256_string(
            metadata_backend.get("evaluator_source_sha256"),
            "production evaluator source digest",
        )
        _validate_python_runtime_origin(
            metadata_backend.get("python_runtime_origin"),
            "production Python/OpenVINO runtime origin",
        )
    runtime_binding = _contract_document()["runtime_source_binding"]
    metadata_dependency_graph = _validate_engine_dependency_graph(
        metadata.get("engine_dependency_graph"),
        synthetic=synthetic,
        runtime_binding=runtime_binding,
        label="training metadata",
    )
    manifest_dependency_graph = _validate_engine_dependency_graph(
        manifest.get("engine_dependency_graph"),
        synthetic=synthetic,
        runtime_binding=runtime_binding,
        label="training manifest",
    )
    _require(
        metadata_dependency_graph == manifest_dependency_graph,
        "training engine dependency graph differs between table and manifest",
    )
    _require(manifest.get("validation_seed_used") is False, "manifest validation contamination")
    _require(manifest.get("sealed_test_seed_used") is False, "manifest sealed contamination")
    _require(manifest.get("real_model_or_game_loaded") is (not synthetic), "manifest model/game role mismatch")

    rows = document.get("rows")
    _require(isinstance(rows, list), "missing training rows")
    row_count = len(rows)
    _require(240 <= row_count <= 288, "training row count is outside [240,288]")
    _require(_exact_int(manifest.get("row_count"), "manifest row_count") == row_count, "manifest row count mismatch")
    exclusions = manifest.get("exclusions")
    _require(isinstance(exclusions, list), "manifest exclusions missing")
    _require(_exact_int(manifest.get("exclusion_count"), "manifest exclusion_count") == len(exclusions), "manifest exclusion count mismatch")
    _require(row_count + len(exclusions) == 288, "training rows/exclusions do not cover frozen 288 states")
    episodes_evidence = manifest.get("episodes")
    _require(isinstance(episodes_evidence, list) and len(episodes_evidence) == 12, "manifest episode evidence mismatch")
    _require(_exact_int(manifest.get("episode_count"), "manifest episode_count") == 12, "manifest episode count mismatch")
    _require(_exact_int(manifest.get("counterfactual_states_completed"), "manifest completed states") == 288, "manifest completed-state count mismatch")
    if not synthetic:
        _require(manifest.get("first32_projection_basis_states") == 432, "projection basis changed")
        _require(manifest.get("first32_projection_formula") == PROJECTION_FORMULA, "projection formula changed")
        projected = manifest.get("first32_projected_seconds")
        _require(isinstance(projected, (int, float)) and math.isfinite(projected), "projection is absent/non-finite")
        _require(0.0 <= float(projected) <= 3300.0, "projection exceeded frozen limit")

    features = np.empty((row_count, FEATURE_COUNT), dtype=np.float64, order="F")
    advantages = np.empty(row_count, dtype=np.float64)
    episode_ids = np.empty(row_count, dtype=np.int64)
    seen_keys: set[tuple[int, int]] = set()
    seen_roots: set[str] = set()
    previous_key: tuple[int, int] | None = None
    episode_counts = {episode: 0 for episode in TRAINING_IDS}
    for index, row in enumerate(rows):
        values = row.get("features")
        _require(isinstance(values, list) and len(values) == FEATURE_COUNT, f"row {index + 1} feature width mismatch")
        try:
            features[index, :] = values
            advantage = float(row["advantage"])
            episode = _exact_int(row["episode_id"], f"row {index + 1} episode_id")
            seed = _exact_int(row["seed"], f"row {index + 1} seed")
            piece = _exact_int(row["piece_index"], f"row {index + 1} piece_index")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"row {index + 1} has invalid numeric fields") from error
        _require(episode in TRAINING_IDS and seed == episode, f"row {index + 1} episode/seed mismatch")
        _require(piece in SAMPLE_PIECES, f"row {index + 1} is off sample schedule")
        key = (episode, piece)
        _require(key not in seen_keys, f"duplicate sampled state {key}")
        _require(previous_key is None or previous_key < key, "training row order is not canonical")
        previous_key = key
        seen_keys.add(key)
        episode_counts[episode] += 1
        root_digest = _sha256_string(
            row.get("root_state_digest"), f"row {index + 1} root digest"
        )
        _require(root_digest not in seen_roots, f"row {index + 1} root digest duplicated")
        seen_roots.add(root_digest)
        _sha256_string(
            row.get("root_future_stream_digest"),
            f"row {index + 1} future digest",
        )
        action1 = _nonempty_string(
            row.get("canonical_top1_action_digest"),
            f"row {index + 1} top-1 action digest",
        )
        action2 = _nonempty_string(
            row.get("canonical_top2_action_digest"),
            f"row {index + 1} top-2 action digest",
        )
        _require(action1 and action2 and action1 != action2, f"row {index + 1} action digests invalid")
        valid_count = _exact_int(
            row.get("valid_action_count"), f"row {index + 1} valid_action_count"
        )
        _require(valid_count >= 2, f"row {index + 1} has fewer than two candidates")
        top1_index = _exact_int(
            row.get("canonical_top1_candidate_index"),
            f"row {index + 1} canonical_top1_candidate_index",
        )
        top2_index = _exact_int(
            row.get("canonical_top2_candidate_index"),
            f"row {index + 1} canonical_top2_candidate_index",
        )
        _require(1 <= top1_index <= valid_count and 1 <= top2_index <= valid_count and top1_index != top2_index, f"row {index + 1} candidate indexes invalid")
        q1 = float(row.get("q_top1", math.nan))
        q2 = float(row.get("q_top2", math.nan))
        gap = float(row.get("q_gap", math.nan))
        _require(q1 >= q2 and gap == q1 - q2, f"row {index + 1} old-Q evidence invalid")
        _require(features[index, 0] == q1 and features[index, 1] == q2 and features[index, 2] == gap, f"row {index + 1} Q features mismatch")
        _require(features[index, 6] == valid_count, f"row {index + 1} candidate-count feature mismatch")
        _require(float(row.get("clipped_target", math.nan)) == max(TARGET_MIN, min(TARGET_MAX, advantage)), f"row {index + 1} clipped target mismatch")
        _require(float(row.get("advantage_unclipped_A6", math.nan)) == advantage, f"row {index + 1} unclipped advantage alias mismatch")
        g6_top1 = float(row.get("g6_top1", math.nan))
        g6_top2 = float(row.get("g6_top2", math.nan))
        _require(g6_top2 - g6_top1 == advantage, f"row {index + 1} G6/advantage mismatch")
        advantages[index] = advantage
        episode_ids[index] = episode
    exclusion_keys: set[tuple[int, int]] = set()
    exclusion_counts = {episode: 0 for episode in TRAINING_IDS}
    for index, exclusion in enumerate(exclusions):
        _require(isinstance(exclusion, dict), f"exclusion {index + 1} is not an object")
        episode = _exact_int(
            exclusion.get("episode_id"), f"exclusion {index + 1} episode_id"
        )
        seed = _exact_int(exclusion.get("seed"), f"exclusion {index + 1} seed")
        piece = _exact_int(
            exclusion.get("piece_index"), f"exclusion {index + 1} piece_index"
        )
        _require(episode in TRAINING_IDS and seed == episode, f"exclusion {index + 1} episode/seed mismatch")
        _require(piece in SAMPLE_PIECES, f"exclusion {index + 1} is off sample schedule")
        key = (episode, piece)
        _require(key not in seen_keys and key not in exclusion_keys, f"duplicate row/exclusion state {key}")
        exclusion_keys.add(key)
        exclusion_counts[episode] += 1
        code = _nonempty_string(exclusion.get("code"), f"exclusion {index + 1} code")
        _require(code in ALLOWED_EXCLUSION_CODES, f"exclusion {index + 1} code is invalid")
        _nonempty_string(exclusion.get("detail"), f"exclusion {index + 1} detail")
        _sha256_string(
            exclusion.get("root_state_digest"),
            f"exclusion {index + 1} root digest",
        )
    planned_keys = {
        (episode, piece) for episode in TRAINING_IDS for piece in SAMPLE_PIECES
    }
    _require(seen_keys | exclusion_keys == planned_keys, "rows/exclusions do not exactly partition the 12x24 design")
    for index, evidence in enumerate(episodes_evidence):
        _require(isinstance(evidence, dict), f"manifest episode {index + 1} is not an object")
        episode = TRAINING_IDS[index]
        _require(_exact_int(evidence.get("episode_id"), f"manifest episode {index + 1} id") == episode, "manifest episode order mismatch")
        _require(_exact_int(evidence.get("seed"), f"manifest episode {index + 1} seed") == episode, "manifest seed order mismatch")
        _require(_exact_int(evidence.get("rows"), f"manifest episode {index + 1} rows") == episode_counts[episode], "manifest per-episode row count mismatch")
        _require(_exact_int(evidence.get("exclusions"), f"manifest episode {index + 1} exclusions") == exclusion_counts[episode], "manifest per-episode exclusion count mismatch")
    _require(np.isfinite(features).all(), "non-finite training feature")
    _require(np.isfinite(advantages).all(), "non-finite training advantage")
    _require(all(0 < episode_counts[episode] <= 24 for episode in TRAINING_IDS), "episode row cardinality invalid")
    positive_fraction = float(np.mean(advantages > 0.0))
    _require(0.02 <= positive_fraction <= 0.40, "training positive fraction outside [0.02,0.40]")
    return {
        "document": document,
        "manifest": manifest,
        "features": features,
        "advantages": advantages,
        "episode_ids": episode_ids,
        "rows": rows,
        "row_count": row_count,
        "positive_fraction": positive_fraction,
        "source_table_sha256": actual_table_sha,
        "source_manifest_sha256": actual_manifest_sha,
        "row_order_sha256": row_order_digest(rows),
        # Preserve the collector object unchanged; downstream assessment can
        # compare the identical runtime lineage without re-encoding it.
        "engine_dependency_graph": metadata_dependency_graph,
        "backend_binding": metadata_backend,
    }


def _load_design_freeze(
    path: Path, names: list[str], expected_sha256: str | None = None
) -> dict[str, Any]:
    freeze, freeze_sha256 = _json_snapshot(path)
    if expected_sha256 is not None:
        _require(freeze_sha256 == expected_sha256, "design freeze differs from frozen SHA handoff")
    contract_path = Path(__file__).with_name("contract.json").resolve()
    contract, contract_sha256 = _json_snapshot(contract_path)
    runtime_binding = contract.get("runtime_source_binding")
    _require(isinstance(runtime_binding, dict), "contract runtime-source binding missing")
    _require(freeze.get("status") == "r1_design_frozen", "design freeze status mismatch")
    _require(freeze.get("experiment") == "online_counterfactual_top2_R1", "design freeze experiment mismatch")
    _require(Path(str(freeze.get("contract_path", ""))).resolve() == contract_path, "design freeze contract path mismatch")
    _require(freeze.get("contract_sha256") == contract_sha256, "design freeze contract SHA-256 mismatch")
    _require(contract.get("feature_schema", {}).get("names") == names, "live contract feature order mismatch")
    _require(freeze.get("feature_names") == names, "design freeze feature order mismatch")
    _require(freeze.get("feature_names_sha256") == FEATURE_SCHEMA_DIGEST, "design freeze feature digest mismatch")
    _require(
        _exact_int(freeze.get("feature_count"), "design freeze feature_count")
        == FEATURE_COUNT,
        "design freeze feature count mismatch",
    )
    _require(
        _exact_int(freeze.get("coefficient_count"), "design freeze coefficient_count")
        == COEFFICIENT_COUNT,
        "design freeze coefficient count mismatch",
    )
    _exact_int_list(freeze.get("training_seed_ids"), TRAINING_IDS, "design freeze training seeds")
    _exact_int_list(freeze.get("sample_piece_indices"), SAMPLE_PIECES, "design freeze sample schedule")
    _require(freeze.get("training_bootstrap_rng") == "Xoshiro(0x5231_2026)", "design freeze bootstrap RNG mismatch")
    _exact_float(freeze.get("ridge_lambda"), RIDGE_LAMBDA, "design freeze lambda")
    _exact_float(
        freeze.get("prediction_lower_quantile"),
        LOWER_QUANTILE,
        "design freeze lower quantile",
    )
    _exact_float(
        freeze.get("override_strict_threshold"),
        OVERRIDE_THRESHOLD,
        "design freeze override threshold",
    )
    _require(freeze.get("hyperparameter_sweep_authorized") is False, "design freeze authorizes sweep")
    for field in (
        "model_or_checkpoint_loaded",
        "game_run",
        "development_seed_loaded",
        "validation_seed_loaded",
        "sealed_test_seed_loaded",
    ):
        _require(freeze.get(field) is False, f"forbidden pre-freeze activity: {field}")
    _require(
        freeze.get("runtime_source_binding") == runtime_binding,
        "design freeze runtime-source binding differs from contract",
    )
    schedules = freeze.get("training_bootstrap_schedules")
    _require(isinstance(schedules, list) and len(schedules) == ENSEMBLE_SIZE, "bootstrap schedule count mismatch")
    _require(all(isinstance(schedule, list) and len(schedule) == 12 for schedule in schedules), "bootstrap schedule width mismatch")
    _require(all(all(type(value) is int and value in TRAINING_IDS for value in schedule) for schedule in schedules), "bootstrap schedule escaped training role")
    observed = schedule_digest(schedules)
    _require(observed == TRAINING_SCHEDULE_SHA256, "bootstrap schedule differs from independent source anchor")
    _require(freeze.get("training_bootstrap_schedule_sha256") == TRAINING_SCHEDULE_SHA256, "freeze bootstrap digest mismatch")
    return {
        "document": freeze,
        "schedules": schedules,
        "schedule_digest": observed,
        "source_sha256": freeze_sha256,
        "runtime_source_binding": runtime_binding,
    }


def _fit_gate(
    features: Any,
    advantages: Any,
    episode_ids: Any,
    schedules: Sequence[Sequence[int]],
    checkpoint: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    np = _import_numpy()
    x = np.asarray(features, dtype=np.float64, order="F")
    target = np.clip(np.asarray(advantages, dtype=np.float64), TARGET_MIN, TARGET_MAX)
    means = np.mean(x, axis=0, dtype=np.float64)
    scales = np.std(x, axis=0, dtype=np.float64, ddof=0)
    constant = np.logical_or(np.all(x == x[0, :], axis=0), scales == 0.0)
    scales = scales.copy()
    scales[constant] = 1.0
    standardized = np.asfortranarray((x - means) / scales)
    standardized[:, constant] = 0.0
    _require(np.isfinite(standardized).all(), "non-finite standardized feature")
    design = np.empty((x.shape[0], COEFFICIENT_COUNT), dtype=np.float64, order="F")
    design[:, 0] = 1.0
    design[:, 1:] = standardized
    coefficients = np.empty((COEFFICIENT_COUNT, ENSEMBLE_SIZE), dtype=np.float64, order="F")
    episode_rows = {episode: np.flatnonzero(episode_ids == episode) for episode in TRAINING_IDS}
    for member_index, schedule in enumerate(schedules):
        cluster_counts = {episode: 0 for episode in TRAINING_IDS}
        for episode in schedule:
            cluster_counts[int(episode)] += 1
        row_weights = np.zeros(x.shape[0], dtype=np.float64)
        for episode in TRAINING_IDS:
            row_weights[episode_rows[episode]] = cluster_counts[episode]
        sqrt_weight = np.sqrt(row_weights)
        weighted_design = np.asfortranarray(design * sqrt_weight[:, None])
        weighted_target = target * sqrt_weight
        gram = np.asfortranarray(weighted_design.T @ weighted_design)
        rhs = weighted_design.T @ weighted_target
        gram[
            np.arange(1, COEFFICIENT_COUNT),
            np.arange(1, COEFFICIENT_COUNT),
        ] += RIDGE_LAMBDA
        try:
            lower = np.linalg.cholesky(gram)
            intermediate = np.linalg.solve(lower, rhs)
            solution = np.linalg.solve(lower.T, intermediate)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"ridge Cholesky failed at member {member_index + 1}") from error
        _require(np.isfinite(solution).all(), "non-finite ridge solution")
        coefficients[:, member_index] = solution
        member = member_index + 1
        if checkpoint is not None and member in (1, 64, 128, 192, ENSEMBLE_SIZE):
            checkpoint(member)
    return {
        "feature_mean": means,
        "feature_scale": scales,
        "constant_feature": constant,
        "coefficients": coefficients,
    }


def _gate_payload(gate: dict[str, Any], names: list[str]) -> dict[str, Any]:
    np = _import_numpy()
    coefficients = np.asarray(gate["coefficients"], dtype=np.float64)
    _require(coefficients.shape == (COEFFICIENT_COUNT, ENSEMBLE_SIZE), "coefficient shape mismatch")
    return {
        "schema_version": "r1-ridge-gate-v1",
        "feature_names": names,
        "feature_schema_digest": FEATURE_SCHEMA_DIGEST,
        "feature_mean": gate["feature_mean"].tolist(),
        "feature_scale": gate["feature_scale"].tolist(),
        "constant_feature": gate["constant_feature"].tolist(),
        "coefficient_shape": [COEFFICIENT_COUNT, ENSEMBLE_SIZE],
        # Outer JSON arrays are coefficient rows, exactly matching gate_payload
        # in ridge_gate.jl: row 0 is the intercept, rows 1:70 are features.
        "coefficients": coefficients.tolist(),
        "lambda": RIDGE_LAMBDA,
        "bootstrap_rng": "Xoshiro(0x5231_2026)",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "lower_quantile": LOWER_QUANTILE,
        "quantile_method": QUANTILE_METHOD,
        "override_threshold": OVERRIDE_THRESHOLD,
        "target_clamp": [TARGET_MIN, TARGET_MAX],
        "ensemble_size": ENSEMBLE_SIZE,
    }


def load_gate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    np = _import_numpy()
    names = _contract_feature_names()
    _require(payload.get("schema_version") == "r1-ridge-gate-v1", "gate schema mismatch")
    _require(payload.get("feature_names") == names, "gate feature order mismatch")
    _require(payload.get("feature_schema_digest") == FEATURE_SCHEMA_DIGEST, "gate feature digest mismatch")
    _exact_int_list(
        payload.get("coefficient_shape"),
        [COEFFICIENT_COUNT, ENSEMBLE_SIZE],
        "gate coefficient_shape",
    )
    _require(
        _exact_int(payload.get("ensemble_size"), "gate ensemble_size")
        == ENSEMBLE_SIZE,
        "gate ensemble size mismatch",
    )
    _exact_float(payload.get("lambda"), RIDGE_LAMBDA, "gate ridge lambda")
    _require(
        payload.get("bootstrap_rng") == "Xoshiro(0x5231_2026)"
        and _exact_int(payload.get("bootstrap_seed"), "gate bootstrap_seed")
        == BOOTSTRAP_SEED,
        "gate bootstrap identity mismatch",
    )
    _exact_float(
        payload.get("lower_quantile"), LOWER_QUANTILE, "gate lower quantile"
    )
    _require(payload.get("quantile_method") == QUANTILE_METHOD, "gate quantile method mismatch")
    _exact_float(
        payload.get("override_threshold"),
        OVERRIDE_THRESHOLD,
        "gate override threshold",
    )
    _exact_float_list(
        payload.get("target_clamp"),
        [TARGET_MIN, TARGET_MAX],
        "gate target clamp",
    )
    coefficients = np.asarray(payload.get("coefficients"), dtype=np.float64)
    means = np.asarray(payload.get("feature_mean"), dtype=np.float64)
    scales = np.asarray(payload.get("feature_scale"), dtype=np.float64)
    constant_payload = payload.get("constant_feature")
    _require(
        isinstance(constant_payload, list)
        and len(constant_payload) == FEATURE_COUNT
        and all(type(value) is bool for value in constant_payload),
        "gate constant-feature mask must contain exactly 70 booleans",
    )
    constant = np.asarray(constant_payload, dtype=np.bool_)
    _require(coefficients.shape == (COEFFICIENT_COUNT, ENSEMBLE_SIZE), "gate coefficient matrix mismatch")
    _require(means.shape == scales.shape == constant.shape == (FEATURE_COUNT,), "gate standardization shape mismatch")
    _require(np.isfinite(coefficients).all() and np.isfinite(means).all() and np.isfinite(scales).all(), "non-finite gate")
    _require(np.all(scales > 0.0), "non-positive gate scale")
    return {
        "feature_names": names,
        "feature_mean": means,
        "feature_scale": scales,
        "constant_feature": constant,
        "coefficients": coefficients,
        "lower_quantile": float(payload["lower_quantile"]),
        "override_threshold": float(payload["override_threshold"]),
    }


def predict_ensemble(gate: dict[str, Any], features: Any) -> Any:
    np = _import_numpy()
    raw = np.asarray(features, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    _require(raw.ndim == 2 and raw.shape[1] == FEATURE_COUNT, "prediction feature shape mismatch")
    _require(np.isfinite(raw).all(), "non-finite prediction feature")
    standardized = (raw - gate["feature_mean"]) / gate["feature_scale"]
    standardized[:, gate["constant_feature"]] = 0.0
    # GEMV per state mirrors the deployment-stable Julia evaluator instead of
    # allowing GEMM blocking to vary with the number of calibration rows.
    result = np.empty((raw.shape[0], ENSEMBLE_SIZE), dtype=np.float64)
    weights = gate["coefficients"][1:, :]
    for row in range(raw.shape[0]):
        result[row, :] = standardized[row, :] @ weights + gate["coefficients"][0, :]
    _require(np.isfinite(result).all(), "non-finite ensemble prediction")
    return result


def linear_type7(values: Any, probability: float = LOWER_QUANTILE) -> float:
    np = _import_numpy()
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    _require(ordered.ndim == 1 and ordered.size > 0, "quantile requires a nonempty vector")
    _require(0.0 <= probability <= 1.0, "quantile probability out of range")
    position_zero_based = (ordered.size - 1) * probability
    lower = int(math.floor(position_zero_based))
    upper = int(math.ceil(position_zero_based))
    if lower == upper:
        return float(ordered[lower])
    fraction = position_zero_based - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def predict_lower_bounds(gate: dict[str, Any], features: Any) -> Any:
    np = _import_numpy()
    predictions = predict_ensemble(gate, features)
    return np.asarray(
        [linear_type7(predictions[row, :], gate["lower_quantile"]) for row in range(predictions.shape[0])],
        dtype=np.float64,
    )


def decide_top2(gate: dict[str, Any], features: Sequence[float], candidate_count: int) -> dict[str, Any]:
    np = _import_numpy()
    try:
        raw = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return {"use_top2": False, "lower_bound": math.nan, "fallback_reason": "invalid_features"}
    if type(candidate_count) is not int:
        return {
            "use_top2": False,
            "lower_bound": math.nan,
            "fallback_reason": "invalid_candidate_count",
        }
    if candidate_count < 2:
        return {"use_top2": False, "lower_bound": math.nan, "fallback_reason": "fewer_than_two_candidates"}
    if raw.shape != (FEATURE_COUNT,) or not np.isfinite(raw).all():
        return {"use_top2": False, "lower_bound": math.nan, "fallback_reason": "invalid_features"}
    if raw[6] != candidate_count:
        return {"use_top2": False, "lower_bound": math.nan, "fallback_reason": "candidate_count_mismatch"}
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            lower = float(predict_lower_bounds(gate, raw)[0])
    except (ArithmeticError, TypeError, ValueError):
        return {
            "use_top2": False,
            "lower_bound": math.nan,
            "fallback_reason": "invalid_prediction",
        }
    if not math.isfinite(lower):
        return {
            "use_top2": False,
            "lower_bound": lower,
            "fallback_reason": "invalid_prediction",
        }
    use_top2 = lower > gate["override_threshold"]
    return {
        "use_top2": bool(use_top2),
        "lower_bound": lower,
        "fallback_reason": "none" if use_top2 else "lower_bound_not_above_threshold",
    }


def _runtime_facts(np: Any) -> dict[str, Any]:
    config = np.show_config(mode="dicts")
    return {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy_version": np.__version__,
        "numpy_show_config": config,
        "linear_algebra": "numpy.linalg.cholesky+numpy.linalg.solve",
        "thread_environment": {
            name: os.environ[name]
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) not in (5, 6):
        raise SystemExit(
            "usage: python fit_ridge.py TRAINING_TABLE.json COLLECTION_MANIFEST.json "
            "DESIGN_FREEZE.json RIDGE_ARTIFACT.json MILESTONES.jsonl [--synthetic]"
        )
    synthetic = len(args) == 6
    if synthetic and args[5] != "--synthetic":
        raise ValueError("unknown fit_ridge option")
    table_path, manifest_path, freeze_path, artifact_path, milestone_path = (
        Path(value).resolve() for value in args[:5]
    )
    _require(len({table_path, manifest_path, freeze_path, artifact_path, milestone_path}) == 5, "fit_ridge paths must be distinct")
    milestones = Milestones(milestone_path)
    try:
        _require(not artifact_path.exists(), f"refusing to overwrite ridge artifact: {artifact_path}")
        _require(freeze_path.is_file(), f"missing design freeze: {freeze_path}")
        if synthetic:
            _require(not table_path.exists(), f"refusing to overwrite synthetic table: {table_path}")
            _require(not manifest_path.exists(), f"refusing to overwrite synthetic manifest: {manifest_path}")
        else:
            _require(table_path.is_file(), f"missing training table: {table_path}")
            _require(manifest_path.is_file(), f"missing collection manifest: {manifest_path}")
        milestones.write("args_verified", synthetic=synthetic)
        if synthetic:
            table_document, manifest_document = _synthetic_documents()
            _atomic_json_create(table_path, table_document)
            manifest_document["table_path"] = str(table_path)
            manifest_document["table_sha256"] = sha256_file(table_path)
            _atomic_json_create(manifest_path, manifest_document)
            milestones.write("synthetic_inputs_generated")
        milestones.write("imports_begin")
        milestones.write("numpy_import_begin")
        np = _import_numpy()
        milestones.write("numpy_import_end", numpy_version=np.__version__)
        expected_table_sha256 = _expected_handoff(
            "R1_EXPECTED_TRAINING_TABLE_SHA256", required=not synthetic
        )
        expected_manifest_sha256 = _expected_handoff(
            "R1_EXPECTED_TRAINING_MANIFEST_SHA256", required=not synthetic
        )
        expected_freeze_sha256 = _expected_handoff(
            "R1_EXPECTED_DESIGN_FREEZE_SHA256", required=not synthetic
        )
        milestones.write("table_and_manifest_load")
        table = _load_training_table(
            table_path, manifest_path, synthetic,
            expected_table_sha256, expected_manifest_sha256,
        )
        names = _contract_feature_names()
        milestones.write("design_freeze_load")
        frozen = _load_design_freeze(freeze_path, names, expected_freeze_sha256)
        milestones.write(
            "schema_verified",
            row_count=table["row_count"],
            episode_count=12,
            positive_fraction=table["positive_fraction"],
            row_order_sha256=table["row_order_sha256"],
        )
        gate = _fit_gate(
            table["features"],
            table["advantages"],
            table["episode_ids"],
            frozen["schedules"],
            checkpoint=lambda member: milestones.write(
                "bootstrap_checkpoint", member=member, ensemble_size=ENSEMBLE_SIZE
            ),
        )
        target = np.clip(table["advantages"], TARGET_MIN, TARGET_MAX)
        artifact = {
            **_gate_payload(gate, names),
            "experiment_id": "online_counterfactual_top2_R1",
            "fit_role": "training_only",
            "fit_backend": "python_numpy_analytic_ridge",
            "runtime_facts": _runtime_facts(np),
            "source_table_sha256": table["source_table_sha256"],
            "source_table_path": str(table_path),
            "source_table_synthetic": synthetic,
            "source_collection_manifest_path": str(manifest_path),
            "source_collection_manifest_sha256": table["source_manifest_sha256"],
            "engine_dependency_graph": table["engine_dependency_graph"],
            "backend_binding": table["backend_binding"],
            "training_row_order_sha256": table["row_order_sha256"],
            "training_row_order_encoding": "episode_id,piece_index,root_state_digest newline joined",
            "design_freeze_path": str(freeze_path),
            "design_freeze_sha256": frozen["source_sha256"],
            "training_bootstrap_schedule_sha256": frozen["schedule_digest"],
            "training_bootstrap_schedule_source_anchor_sha256": TRAINING_SCHEDULE_SHA256,
            "training_bootstrap_schedule_consumed": True,
            "training_stats": {
                "row_count": table["row_count"],
                "episode_count": 12,
                "episode_ids": TRAINING_IDS,
                "positive_fraction_unclipped": table["positive_fraction"],
                "advantage_unclipped_min": float(np.min(table["advantages"])),
                "advantage_unclipped_max": float(np.max(table["advantages"])),
                "advantage_unclipped_mean": float(np.mean(table["advantages"])),
                "target_clipped_min": float(np.min(target)),
                "target_clipped_max": float(np.max(target)),
                "target_clipped_mean": float(np.mean(target)),
                "constant_feature_count": int(np.count_nonzero(gate["constant_feature"])),
                "constant_feature_indices": [
                    int(index + 1) for index in np.flatnonzero(gate["constant_feature"])
                ],
            },
            "all_finite": bool(
                np.isfinite(gate["feature_mean"]).all()
                and np.isfinite(gate["feature_scale"]).all()
                and np.isfinite(gate["coefficients"]).all()
            ),
            "validation_seed_used": False,
            "sealed_test_seed_used": False,
            "claim_scope": "analytic_training_artifact_not_calibration_or_game_strength",
        }
        _require(artifact["all_finite"], "non-finite ridge artifact")
        _atomic_json_create(artifact_path, artifact)
        published_sha = sha256_file(artifact_path)
        milestones.write(
            "artifact_published",
            path=str(artifact_path),
            sha256=published_sha,
            coefficient_shape=[COEFFICIENT_COUNT, ENSEMBLE_SIZE],
        )
        milestones.write("phase_complete", artifact_sha256=published_sha)
        print(f"R1_RIDGE_ARTIFACT={artifact_path}")
        print(f"R1_RIDGE_SHA256={published_sha}")
        return 0
    finally:
        milestones.close()


if __name__ == "__main__":
    raise SystemExit(main())
