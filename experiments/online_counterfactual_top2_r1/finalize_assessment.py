from __future__ import annotations

"""Independent, fail-closed terminal assessment for preregistered R1.

This deliberately does not import the fitter or calibration-gate modules.  It
revalidates the immutable bindings and recomputes the calibration decisions
and gate statistics from the source tables and ridge coefficients.  The only
quantity it cannot recreate after the fact is measured decision latency; that
quantity is required to have mutually consistent runtime and assessment
evidence and is still checked against the frozen limit.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


EXPERIMENT_ID = "online_counterfactual_top2_R1"
FEATURE_DIGEST = "7e89c16b57dcebac56e3ab4c5be161d5e5430c682e60f3b565dd23ab3b04ac44"
TRAIN_SCHEDULE_DIGEST = "5b60a1e340b542dc8654a5c80777c254d8336aa086e51be6c2ba1251be20e5f7"
CALIBRATION_SCHEDULE_DIGEST = "c08341a6891997301f112912a4fe969c5493603834cc3dbc114b3e24401617db"
CHECKPOINT_DIGEST = "7b0f78edd0867d468c376f1b5375bb9a4d2195fa0fa5f76f94924723b26adfc1"
WEIGHT_DIGEST = "2ee741ebef7b7c0c5cbc0f86492e8b8d935989af149bff467a3ba8ca633375ba"
TRAIN_EPISODES = tuple(range(73001, 73013))
CALIBRATION_EPISODES = tuple(range(73101, 73107))
SAMPLE_PIECES = tuple(range(10, 241, 10))
PROJECTION_FORMULA = (
    "2*setup_seconds + (first32_collection_seconds-repeatability_probe_seconds)/32*432 "
    "+ repeatability_probe_seconds"
)
DEPLOYMENT_SCHEMA = "r1-live-ridge-decision-v1"
DEPLOYMENT_TIMING_SCOPE = (
    "feature_build+ridge_eval+selection_binding;"
    "excludes_candidate_enumeration+old_q_evaluation+artifact_load+clone_apply_verification"
)
DEPLOYMENT_FEATURE_ENCODING = (
    "feature_schema_sha256 newline Float64-bitstring-per-feature newline joined"
)
QUANTILE_METHOD = "linear_type7_position_1_plus_n_minus_1_p"
FEATURE_COUNT = 70
COEFFICIENT_COUNT = 71
ENSEMBLE_COUNT = 256
NUMERIC_VALUE_COUNT = FEATURE_COUNT * 2 + COEFFICIENT_COUNT * ENSEMBLE_COUNT
OPENVINO_FULL_VERSION = "2026.2.1-21919-ede283a88e3-releases/2026/2"
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
EXPECTED_OPENVINO_NATIVE_MODULES = [
    {"path": "_pyopenvino.cp312-win_amd64.pyd", "bytes": 3981304, "sha256": "602468f4ca37ba9859a6b22220e46267170abd0f744a72a839bb5e0874f2ad56"},
    {"path": "frontend/tensorflow/py_tensorflow_frontend.cp312-win_amd64.pyd", "bytes": 477176, "sha256": "859f76d40fb77074bf94de2bed3317aeb835debb2e98bd4b78ce24f0b1077c28"},
    {"path": "libs/openvino.dll", "bytes": 15888376, "sha256": "dc29dfd84048bed1b687426f770ee355d6c2933bbb3c6aa861f6dd1154e48908"},
    {"path": "libs/tbb12.dll", "bytes": 225272, "sha256": "5cdc63525bc1b7f9916b329a1304955a360803b3d5780f5b23e62fd4027622be"},
    {"path": "libs/tbbmalloc.dll", "bytes": 120824, "sha256": "11ac3ef8f213cc647b63f92dd115c2c005e489fd050fd41c3e69cf3cd4949921"},
]
COUNTERFACTUAL_GAMMA = 0.997
COUNTERFACTUAL_HORIZON = 6
COUNTERFACTUAL_SCORE_NORMALIZER = 600.0
# Exact binary64 outputs of Julia 1.12.6's `Float64(0.997)^n` for n=0:6.
# Python's `**` differs by one ulp at n=6, so spelling out the frozen Julia
# arithmetic is necessary for an exact serialized cross-runtime check.
COUNTERFACTUAL_GAMMA_POWERS = tuple(
    float.fromhex(value)
    for value in (
        "0x1p+0",
        "0x1.fe76c8b439581p-1",
        "0x1.fceebf65dbfcfp-1",
        "0x1.fb67e32cfa4ddp-1",
        "0x1.f9e2332258c9p-1",
        "0x1.f85dae5f6d82cp-1",
        "0x1.f6da53fe5f9d3p-1",
    )
)
ENGINE_DEPENDENCY_ENCODING = "sorted relative_path + NUL + lowercase sha256 + newline"
UPSTREAM_TETRISAI_HEAD = "6fdfb1d30197246fd862b716438e998f0315c830"
ENGINE_DEPENDENCY_PATHS = (
    "experiments/online_counterfactual_top2_r1/engine_adapter.jl",
    "experiments/online_counterfactual_top2_r1/vendor/TetrisAI/src/core/analyzer.jl",
    "experiments/online_counterfactual_top2_r1/vendor/TetrisAI/src/core/components/node.jl",
    "vendor/Tetris/lib/curses.jl",
    "vendor/Tetris/lib/game.so",
    "vendor/Tetris/lib/key_input.jl",
    "vendor/Tetris/lib/pdcurses.dll",
)
NODE_SOURCE_DIGEST = "e98d2052f9248f5c08c1eb58adaace1bd01533f287e682bf35a2fefa1325fe82"
ANALYZER_SOURCE_DIGEST = "24152e2549dcc6c3c25d928454268e8baaa4d45fea31044603917cfbabbe02bc"
FROZEN_RUNTIME_DEPENDENCIES = {
    "experiments/online_counterfactual_top2_r1/vendor/TetrisAI/src/core/analyzer.jl": (
        8114,
        ANALYZER_SOURCE_DIGEST,
    ),
    "experiments/online_counterfactual_top2_r1/vendor/TetrisAI/src/core/components/node.jl": (
        298,
        NODE_SOURCE_DIGEST,
    ),
    "vendor/Tetris/lib/curses.jl": (
        2767,
        "4dd113316c4f82a226563d7ac3237c366417211582722b3d4b4277dcb12ff922",
    ),
    "vendor/Tetris/lib/game.so": (
        49870,
        "d63a03f494cb0a6f1704624923c58cb521a8a45873ca400e7085c02b1bf5bf46",
    ),
    "vendor/Tetris/lib/key_input.jl": (
        1448,
        "c09571b424a49f01278f6903c0018f9a2dfc652dfd18b804c4ad2b6a37f2fc53",
    ),
    "vendor/Tetris/lib/pdcurses.dll": (
        176673,
        "0c770aa6721aa2155bbe2ef1d0f50ad2065da399085242454486c855c1f9fe67",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value}")


def load_json(path: Path) -> dict[str, Any]:
    return load_json_bytes(path.read_bytes(), path)


def load_json_bytes(payload: bytes, label: Any) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"), parse_constant=_json_constant)
    _require(isinstance(value, dict), f"JSON root is not an object: {label}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _required(document: dict[str, Any], name: str) -> Any:
    _require(name in document, f"missing required property: {name}")
    return document[name]


def _bool(value: Any, label: str) -> bool:
    _require(type(value) is bool, f"{label} is not an exact boolean")
    return value


def _number(value: Any, label: str) -> float:
    _require(type(value) in (int, float), f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is non-finite")
    return result


def _integer(value: Any, label: str) -> int:
    _require(type(value) is int, f"{label} is not an exact integer")
    return value


def _digest(value: Any, label: str) -> str:
    result = str(value).lower()
    _require(len(result) == 64 and all(ch in "0123456789abcdef" for ch in result), f"invalid {label} SHA-256")
    return result


def _lowercase_digest(value: Any, label: str) -> str:
    result = _digest(value, label)
    _require(type(value) is str and value == result, f"{label} SHA-256 is not canonical lowercase")
    return result


def _normal(path: Any) -> str:
    return os.path.normcase(str(Path(str(path)).resolve()))


def _validate_python_runtime_origin(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} missing")
    for key, expected in EXPECTED_PYTHON_RUNTIME_ORIGIN.items():
        observed = value.get(key)
        if key in {
            "python_executable", "python_base_prefix", "python_prefix",
            "openvino_module_file", "openvino_package_root",
        }:
            _require(_normal(observed) == _normal(expected), f"{label} {key} mismatch")
        else:
            _require(observed == expected, f"{label} {key} mismatch")
    modules = value.get("openvino_loaded_native_modules")
    _require(modules == EXPECTED_OPENVINO_NATIVE_MODULES, f"{label} native module list mismatch")
    for index, module in enumerate(modules, 1):
        _require(isinstance(module, dict), f"{label} native module {index} invalid")
        _require(type(module.get("path")) is str and bool(module["path"]), f"{label} native module {index} path invalid")
        _require(_integer(module.get("bytes"), f"{label} native module {index} bytes") > 0, f"{label} native module {index} bytes invalid")
        _digest(module.get("sha256"), f"{label} native module {index} digest")
    return value


def _feature_digest(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _dependency_digest(records: Sequence[dict[str, Any]]) -> str:
    payload = b"".join(
        str(record["path"]).encode("utf-8")
        + b"\0"
        + str(record["sha256"]).lower().encode("ascii")
        + b"\n"
        for record in sorted(records, key=lambda record: str(record["path"]))
    )
    return hashlib.sha256(payload).hexdigest()


def _validate_engine_dependency_graph(
    value: Any,
    *,
    synthetic: bool,
    label: str,
) -> dict[str, Any] | None:
    if synthetic:
        _require(value is None, f"{label} synthetic engine dependency graph must be null")
        return None
    _require(isinstance(value, dict), f"{label} engine dependency graph missing")
    expected_keys = {
        "schema_version",
        "encoding",
        "upstream_tetrisai",
        "records",
        "graph_sha256",
        "runtime_closure_sha256",
        "node_source_sha256",
        "analyzer_source_sha256",
    }
    _require(set(value) == expected_keys, f"{label} engine dependency graph schema mismatch")
    _require(
        value["schema_version"] == "r1-engine-dependency-graph-v1",
        f"{label} engine dependency graph version mismatch",
    )
    _require(value["encoding"] == ENGINE_DEPENDENCY_ENCODING, f"{label} dependency encoding mismatch")
    upstream = value["upstream_tetrisai"]
    _require(
        isinstance(upstream, dict) and set(upstream) == {"head", "clean"},
        f"{label} upstream TetrisAI evidence schema mismatch",
    )
    _require(upstream["head"] == UPSTREAM_TETRISAI_HEAD, f"{label} upstream TetrisAI HEAD mismatch")
    _require(_bool(upstream["clean"], f"{label} upstream clean") is True, f"{label} upstream TetrisAI is dirty")
    records = value["records"]
    _require(isinstance(records, list), f"{label} engine dependency records missing")
    _require(len(records) == len(ENGINE_DEPENDENCY_PATHS), f"{label} engine dependency cardinality mismatch")
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        _require(
            isinstance(record, dict) and set(record) == {"path", "bytes", "sha256"},
            f"{label} engine dependency record {index} schema mismatch",
        )
        relative_path = str(record["path"])
        _require("\\" not in relative_path, f"{label} engine dependency path is not canonical")
        normalized.append(
            {
                "path": relative_path,
                "bytes": _integer(record["bytes"], f"{label} engine dependency bytes"),
                "sha256": _lowercase_digest(
                    record["sha256"], f"{label} engine dependency"
                ),
            }
        )
    _require(
        [record["path"] for record in normalized] == list(ENGINE_DEPENDENCY_PATHS),
        f"{label} engine dependency path/order mismatch",
    )
    _require(all(record["bytes"] >= 0 for record in normalized), f"{label} negative dependency size")
    by_path = {record["path"]: record for record in normalized}
    node_path = "experiments/online_counterfactual_top2_r1/vendor/TetrisAI/src/core/components/node.jl"
    analyzer_path = "experiments/online_counterfactual_top2_r1/vendor/TetrisAI/src/core/analyzer.jl"
    _require(by_path[node_path]["sha256"] == NODE_SOURCE_DIGEST, f"{label} vendored node hash mismatch")
    _require(by_path[analyzer_path]["sha256"] == ANALYZER_SOURCE_DIGEST, f"{label} vendored analyzer hash mismatch")
    for relative_path, (expected_bytes, expected_sha256) in FROZEN_RUNTIME_DEPENDENCIES.items():
        _require(
            by_path[relative_path] == {
                "path": relative_path,
                "bytes": expected_bytes,
                "sha256": expected_sha256,
            },
            f"{label} frozen runtime dependency mismatch: {relative_path}",
        )
    _require(
        _lowercase_digest(value["node_source_sha256"], f"{label} node source") == NODE_SOURCE_DIGEST,
        f"{label} node convenience hash mismatch",
    )
    _require(
        _lowercase_digest(value["analyzer_source_sha256"], f"{label} analyzer source") == ANALYZER_SOURCE_DIGEST,
        f"{label} analyzer convenience hash mismatch",
    )
    _require(
        _lowercase_digest(value["graph_sha256"], f"{label} dependency graph") == _dependency_digest(normalized),
        f"{label} engine dependency graph digest mismatch",
    )
    runtime_records = [
        record
        for record in normalized
        if record["path"] != "experiments/online_counterfactual_top2_r1/engine_adapter.jl"
    ]
    _require(
        _lowercase_digest(value["runtime_closure_sha256"], f"{label} runtime closure")
        == _dependency_digest(runtime_records),
        f"{label} runtime closure digest mismatch",
    )
    return {
        "schema_version": value["schema_version"],
        "encoding": value["encoding"],
        "upstream_tetrisai": {"head": upstream["head"], "clean": True},
        "records": normalized,
        "graph_sha256": value["graph_sha256"],
        "runtime_closure_sha256": value["runtime_closure_sha256"],
        "node_source_sha256": value["node_source_sha256"],
        "analyzer_source_sha256": value["analyzer_source_sha256"],
    }


def _live_engine_dependency_graph(repository: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for relative_path in ENGINE_DEPENDENCY_PATHS:
        path = repository.joinpath(*relative_path.split("/"))
        _require(path.is_file(), f"missing live engine dependency: {relative_path}")
        payload = path.read_bytes()
        records.append(
            {
                "path": relative_path,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    upstream_path = repository / "upstream" / "TetrisAI"
    _require(upstream_path.is_dir(), "missing nested upstream TetrisAI repository")
    try:
        head_result = subprocess.run(
            ["git", "-C", str(upstream_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        status_result = subprocess.run(
            ["git", "-C", str(upstream_path), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot verify nested upstream TetrisAI repository: {error}") from error
    head = head_result.stdout.strip()
    _require(head == UPSTREAM_TETRISAI_HEAD, "live upstream TetrisAI HEAD mismatch")
    _require(not status_result.stdout, "live upstream TetrisAI repository is dirty")
    runtime_records = [
        record
        for record in records
        if record["path"] != "experiments/online_counterfactual_top2_r1/engine_adapter.jl"
    ]
    graph = {
        "schema_version": "r1-engine-dependency-graph-v1",
        "encoding": ENGINE_DEPENDENCY_ENCODING,
        "upstream_tetrisai": {"head": head, "clean": True},
        "records": records,
        "graph_sha256": _dependency_digest(records),
        "runtime_closure_sha256": _dependency_digest(runtime_records),
        "node_source_sha256": NODE_SOURCE_DIGEST,
        "analyzer_source_sha256": ANALYZER_SOURCE_DIGEST,
    }
    return _validate_engine_dependency_graph(graph, synthetic=False, label="live") or graph


def _schedule_digest(schedules: Sequence[Sequence[int]]) -> str:
    payload = ";".join(",".join(str(int(value)) for value in schedule) for schedule in schedules)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _same_number(left: Any, right: Any, *, tolerance: float = 1.0e-12) -> bool:
    try:
        lvalue = _number(left, "left comparison value")
        rvalue = _number(right, "right comparison value")
    except ValueError:
        return False
    return math.isclose(lvalue, rvalue, rel_tol=tolerance, abs_tol=tolerance)


def _require_scope_false(document: dict[str, Any], names: Sequence[str], label: str) -> None:
    for name in names:
        _require(name in document, f"{label} lacks scope flag {name}")
        _require(_bool(document[name], f"{label}.{name}") is False, f"{label} scope flag {name} is true")


class Milestones:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        self.write("script_enter")

    def write(self, stage: str, **details: Any) -> None:
        json.dump(
            {
                "phase": "finalize_assessment",
                "stage": stage,
                "pid": os.getpid(),
                "time_ns": time.time_ns(),
                **details,
            },
            self.stream,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.stream.write("\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()


def atomic_create_json(path: Path, value: dict[str, Any]) -> Path:
    destination = path.resolve()
    _require(not destination.exists(), f"refusing to overwrite assessment: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    _require(not temporary.exists(), f"stale assessment temporary: {temporary}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic create-if-absent.  Unlike rename(), hard-link publication
        # cannot replace a destination that appears after the prior check.
        os.link(temporary, destination)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _validate_schedule(
    schedules: Any,
    *,
    count: int,
    width: int,
    roles: tuple[int, ...],
    digest: str,
    label: str,
) -> list[list[int]]:
    _require(isinstance(schedules, list) and len(schedules) == count, f"{label} schedule count mismatch")
    normalized = [
        [_integer(value, f"{label} schedule entry") for value in schedule]
        for schedule in schedules
    ]
    allowed = set(roles)
    _require(all(len(schedule) == width for schedule in normalized), f"{label} schedule width mismatch")
    _require(all(all(value in allowed for value in schedule) for schedule in normalized), f"{label} schedule escaped role")
    _require(_schedule_digest(normalized) == digest, f"{label} schedule source-anchor mismatch")
    return normalized


def validate_freeze(path: Path, document: dict[str, Any]) -> dict[str, Any]:
    _require(_required(document, "status") == "r1_design_frozen", "design freeze status mismatch")
    _require(_required(document, "experiment") == EXPERIMENT_ID, "design freeze experiment mismatch")
    names = [str(value) for value in _required(document, "feature_names")]
    _require(len(names) == FEATURE_COUNT and len(set(names)) == FEATURE_COUNT, "freeze feature cardinality mismatch")
    _require(_feature_digest(names) == FEATURE_DIGEST, "freeze feature order mismatch")
    _require(_digest(_required(document, "feature_names_sha256"), "freeze feature") == FEATURE_DIGEST, "freeze feature digest mismatch")
    _require(_integer(_required(document, "feature_count"), "freeze feature count") == FEATURE_COUNT, "freeze feature count mismatch")
    _require(_integer(_required(document, "coefficient_count"), "freeze coefficient count") == COEFFICIENT_COUNT, "freeze coefficient count mismatch")
    _require([_integer(value, "freeze training seed") for value in _required(document, "training_seed_ids")] == list(TRAIN_EPISODES), "freeze training role mismatch")
    _require([_integer(value, "freeze calibration seed") for value in _required(document, "calibration_seed_ids")] == list(CALIBRATION_EPISODES), "freeze calibration role mismatch")
    _require([_integer(value, "freeze sample piece") for value in _required(document, "sample_piece_indices")] == list(SAMPLE_PIECES), "freeze sample schedule mismatch")
    _require(_number(_required(document, "ridge_lambda"), "freeze lambda") == 1.0, "freeze lambda mismatch")
    _require(_number(_required(document, "prediction_lower_quantile"), "freeze quantile") == 0.10, "freeze quantile mismatch")
    _require(
        _required(document, "prediction_quantile_method") == QUANTILE_METHOD,
        "freeze prediction quantile method mismatch",
    )
    _require(_number(_required(document, "override_strict_threshold"), "freeze threshold") == 0.05, "freeze threshold mismatch")
    _require(_required(document, "training_bootstrap_rng") == "Xoshiro(0x5231_2026)", "freeze training RNG mismatch")
    _require(_required(document, "calibration_bootstrap_rng") == "Xoshiro(0x5231_73106)", "freeze calibration RNG mismatch")
    _require_scope_false(
        document,
        (
            "hyperparameter_sweep_authorized",
            "model_or_checkpoint_loaded",
            "game_run",
            "development_seed_loaded",
            "validation_seed_loaded",
            "sealed_test_seed_loaded",
        ),
        "freeze",
    )
    immutable = _required(document, "immutable_inputs")
    _require(_digest(_required(immutable, "old_checkpoint_sha256"), "checkpoint") == CHECKPOINT_DIGEST, "freeze checkpoint binding mismatch")
    _require(_digest(_required(immutable, "old_openvino_weight_npz_sha256"), "OpenVINO weights") == WEIGHT_DIGEST, "freeze weight binding mismatch")
    backend = _required(document, "openvino_backend")
    _require(_required(backend, "version") == "2026.2.1", "freeze OpenVINO semantic version mismatch")
    _require(_required(backend, "full_build") == OPENVINO_FULL_VERSION, "freeze OpenVINO exact build mismatch")
    _require(_digest(_required(backend, "weight_sha256"), "backend weight") == WEIGHT_DIGEST, "freeze backend weight mismatch")
    _require(_required(backend, "complete_chunk") == {"device": "NPU", "batch_size": 16, "shape": "static", "eligible_candidate_count": 16}, "freeze complete-chunk backend mismatch")
    _require(_required(backend, "tail_chunk") == {"device": "CPU", "shape": "dynamic", "batch_semantics": "actual candidate count", "minimum_candidate_count": 1, "maximum_candidate_count": 15, "padding": False}, "freeze tail backend mismatch")
    contract_path = Path(str(_required(document, "contract_path"))).resolve()
    expected_contract = Path(__file__).with_name("contract.json").resolve()
    _require(_normal(contract_path) == _normal(expected_contract), "freeze contract path mismatch")
    _require(_digest(_required(document, "contract_sha256"), "contract") == sha256_file(expected_contract), "freeze contract hash mismatch")
    contract_document = load_json(expected_contract)
    runtime_source_binding = _required(document, "runtime_source_binding")
    _require(
        runtime_source_binding == _required(contract_document, "runtime_source_binding"),
        "freeze runtime source binding mismatch",
    )
    eligibility_path = Path(str(_required(document, "eligibility_path"))).resolve()
    _require(eligibility_path.is_file(), "missing freeze eligibility source")
    _require(_digest(_required(document, "eligibility_sha256"), "eligibility") == sha256_file(eligibility_path), "freeze eligibility hash mismatch")
    training = _validate_schedule(
        _required(document, "training_bootstrap_schedules"), count=256, width=12,
        roles=TRAIN_EPISODES, digest=TRAIN_SCHEDULE_DIGEST, label="training bootstrap",
    )
    calibration = _validate_schedule(
        _required(document, "calibration_bootstrap_schedules"), count=2000, width=6,
        roles=CALIBRATION_EPISODES, digest=CALIBRATION_SCHEDULE_DIGEST, label="calibration bootstrap",
    )
    _require(_digest(_required(document, "training_bootstrap_schedule_sha256"), "training schedule") == TRAIN_SCHEDULE_DIGEST, "freeze training schedule digest mismatch")
    _require(_digest(_required(document, "calibration_bootstrap_schedule_sha256"), "calibration schedule") == CALIBRATION_SCHEDULE_DIGEST, "freeze calibration schedule digest mismatch")
    return {
        "feature_names": names,
        "training_schedules": training,
        "calibration_schedules": calibration,
        "runtime_source_binding": runtime_source_binding,
    }


def _validate_backend_evidence(value: Any, *, synthetic: bool, label: str) -> None:
    if synthetic:
        _require(value is None, f"{label} synthetic backend evidence must be null")
        return
    _require(isinstance(value, dict), f"{label} backend evidence missing")
    expected = {
        "old_openvino_weight_npz_sha256": WEIGHT_DIGEST,
        "old_checkpoint_sha256": CHECKPOINT_DIGEST,
        "openvino_version": "2026.2.1",
        "openvino_full_build": OPENVINO_FULL_VERSION,
        "complete_device": "NPU",
        "tail_device": "CPU",
        "complete_batch_size": 16,
    }
    for name, wanted in expected.items():
        _require(_required(value, name) == wanted, f"{label} backend {name} mismatch")
    _digest(_required(value, "evaluator_source_sha256"), f"{label} evaluator source")
    _validate_python_runtime_origin(
        _required(value, "python_runtime_origin"),
        f"{label} Python/OpenVINO runtime origin",
    )


def _validate_immutable_evidence(value: Any, *, synthetic: bool, label: str) -> None:
    if synthetic:
        _require(value is None, f"{label} synthetic immutable evidence must be null")
        return
    _require(isinstance(value, dict), f"{label} immutable end hashes missing")
    _require(_digest(_required(value, "old_checkpoint_sha256"), "checkpoint end") == CHECKPOINT_DIGEST, f"{label} checkpoint changed")
    _require(_digest(_required(value, "old_openvino_weight_npz_sha256"), "weights end") == WEIGHT_DIGEST, f"{label} weights changed")
    _require(_bool(_required(value, "unchanged"), f"{label}.unchanged") is True, f"{label} immutable inputs changed")


_BRANCH_EVIDENCE_KEYS = {
    "return_G6",
    "terminal_within_horizon",
    "terminal_step",
    "bootstrap_q",
    "score_deltas",
    "pre_action_current_piece_tokens",
    "placed_piece_tokens",
    "post_action_rng_digests",
    "final_rng_digest",
    "selected_action_digests",
    "candidate_counts",
    "branch_start_state_digest",
    "branch_start_future_stream_digest",
    "start_score",
    "final_score",
    "pieces_played",
    "exact_outcome_digest",
    "decision_evidence",
}
_DECISION_EVIDENCE_KEYS = {
    "rollout_piece",
    "kind",
    "candidate_count",
    "candidate_order_digest",
    "q_vector_digest",
    "selected_index",
    "selected_action_digest",
    "selected_q",
    "max_q",
}


def _exact_number_equal(left: Any, right: Any, label: str) -> float:
    """Require equality of the parsed finite binary64 values, not tolerance."""
    value = _number(left, label)
    expected = _number(right, f"{label} recomputation")
    _require(value == expected, f"{label} exact serialized value mismatch")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    _require(type(value) is str and bool(value), f"{label} is not a non-empty string")
    return value


def _validate_decision_evidence(
    value: Any,
    *,
    label: str,
    rollout_piece: int,
    kind: str,
) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} is not an object")
    _require(set(value) == _DECISION_EVIDENCE_KEYS, f"{label} schema mismatch")
    _require(
        _integer(_required(value, "rollout_piece"), f"{label} rollout piece") == rollout_piece,
        f"{label} rollout piece mismatch",
    )
    _require(_required(value, "kind") == kind, f"{label} kind mismatch")
    candidate_count = _integer(_required(value, "candidate_count"), f"{label} candidate count")
    _require(candidate_count >= 1, f"{label} candidate count is not positive")
    selected_index = _integer(_required(value, "selected_index"), f"{label} selected index")
    _require(1 <= selected_index <= candidate_count, f"{label} selected index is out of range")
    return {
        "rollout_piece": rollout_piece,
        "kind": kind,
        "candidate_count": candidate_count,
        "candidate_order_digest": _lowercase_digest(
            _required(value, "candidate_order_digest"), f"{label} candidate order"
        ),
        "q_vector_digest": _lowercase_digest(
            _required(value, "q_vector_digest"), f"{label} Q vector"
        ),
        "selected_index": selected_index,
        "selected_action_digest": _lowercase_digest(
            _required(value, "selected_action_digest"), f"{label} selected action"
        ),
        "selected_q": _number(_required(value, "selected_q"), f"{label} selected Q"),
        "max_q": _number(_required(value, "max_q"), f"{label} maximum Q"),
    }


def _validate_production_branch(
    value: Any,
    *,
    label: str,
    root_state_digest: str,
    root_future_stream_digest: str,
    root_candidate_count: int,
    root_selected_index: int,
    root_selected_action_digest: str,
    root_selected_q: float,
    root_max_q: float,
) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} evidence missing")
    _require(set(value) == _BRANCH_EVIDENCE_KEYS, f"{label} evidence schema mismatch")

    terminal = _bool(_required(value, "terminal_within_horizon"), f"{label} terminal")
    terminal_step = _integer(_required(value, "terminal_step"), f"{label} terminal step")
    if terminal:
        _require(
            1 <= terminal_step <= COUNTERFACTUAL_HORIZON,
            f"{label} terminal step outside the six-piece horizon",
        )
        applied_steps = terminal_step
    else:
        _require(terminal_step == 0, f"{label} surviving branch has a terminal step")
        applied_steps = COUNTERFACTUAL_HORIZON

    score_deltas_raw = _required(value, "score_deltas")
    _require(isinstance(score_deltas_raw, list), f"{label} score deltas missing")
    score_deltas = [_number(item, f"{label} score delta") for item in score_deltas_raw]
    _require(len(score_deltas) == applied_steps, f"{label} score-delta horizon mismatch")
    start_score = _number(_required(value, "start_score"), f"{label} start score")
    final_score = _number(_required(value, "final_score"), f"{label} final score")
    pieces_played = _integer(_required(value, "pieces_played"), f"{label} pieces played")
    _require(pieces_played == applied_steps, f"{label} pieces-played horizon mismatch")
    _exact_number_equal(
        final_score - start_score,
        math.fsum(score_deltas),
        f"{label} score accounting",
    )

    candidate_counts_raw = _required(value, "candidate_counts")
    _require(isinstance(candidate_counts_raw, list), f"{label} candidate counts missing")
    candidate_counts = [
        _integer(item, f"{label} candidate count") for item in candidate_counts_raw
    ]
    _require(
        len(candidate_counts) == applied_steps and all(item >= 1 for item in candidate_counts),
        f"{label} candidate-count horizon mismatch",
    )
    _require(candidate_counts[0] == root_candidate_count, f"{label} root candidate count mismatch")

    selected_action_digests_raw = _required(value, "selected_action_digests")
    _require(isinstance(selected_action_digests_raw, list), f"{label} selected action digests missing")
    selected_action_digests = [
        _lowercase_digest(item, f"{label} selected action")
        for item in selected_action_digests_raw
    ]
    _require(
        len(selected_action_digests) == applied_steps,
        f"{label} selected-action horizon mismatch",
    )
    _require(
        selected_action_digests[0] == root_selected_action_digest,
        f"{label} root selected action digest mismatch",
    )

    for field in ("pre_action_current_piece_tokens", "placed_piece_tokens"):
        raw = _required(value, field)
        _require(isinstance(raw, list) and len(raw) == applied_steps, f"{label} {field} horizon mismatch")
        for item in raw:
            _nonempty_string(item, f"{label} {field} item")

    rng_raw = _required(value, "post_action_rng_digests")
    _require(isinstance(rng_raw, list), f"{label} post-action RNG digests missing")
    rng_digests = [_lowercase_digest(item, f"{label} post-action RNG") for item in rng_raw]
    _require(len(rng_digests) == applied_steps, f"{label} post-action RNG horizon mismatch")
    _require(
        _lowercase_digest(_required(value, "final_rng_digest"), f"{label} final RNG")
        == rng_digests[-1],
        f"{label} final RNG digest mismatch",
    )

    branch_future = _lowercase_digest(
        _required(value, "branch_start_future_stream_digest"),
        f"{label} branch-start future stream",
    )
    branch_state = _lowercase_digest(
        _required(value, "branch_start_state_digest"), f"{label} branch-start state"
    )
    _require(branch_state == root_state_digest, f"{label} branch-start state mismatch")
    exact_outcome_digest = _lowercase_digest(
        _required(value, "exact_outcome_digest"), f"{label} exact outcome"
    )
    _require(
        branch_future == root_future_stream_digest,
        f"{label} branch-start future stream mismatch",
    )

    raw_decisions = _required(value, "decision_evidence")
    expected_decisions = applied_steps + (0 if terminal else 1)
    _require(
        isinstance(raw_decisions, list) and len(raw_decisions) == expected_decisions,
        f"{label} decision-evidence horizon mismatch",
    )
    decisions = [
        _validate_decision_evidence(
            raw,
            label=f"{label} decision {step}",
            rollout_piece=step,
            kind="action",
        )
        for step, raw in enumerate(raw_decisions[:applied_steps], 1)
    ]
    for step, decision in enumerate(decisions, 1):
        offset = step - 1
        _require(
            decision["candidate_count"] == candidate_counts[offset],
            f"{label} decision {step} candidate count mismatch",
        )
        _require(
            decision["selected_action_digest"] == selected_action_digests[offset],
            f"{label} decision {step} selected action digest mismatch",
        )
        if step > 1:
            _require(
                decision["selected_q"] == decision["max_q"],
                f"{label} decision {step} did not select the old-policy maximum Q",
            )

    first = decisions[0]
    _require(first["selected_index"] == root_selected_index, f"{label} root selected index mismatch")
    _require(first["selected_q"] == root_selected_q, f"{label} root selected Q mismatch")
    _require(first["max_q"] == root_max_q, f"{label} root maximum Q mismatch")

    bootstrap_q = _number(_required(value, "bootstrap_q"), f"{label} bootstrap Q")
    if terminal:
        _require(bootstrap_q == 0.0, f"{label} terminal branch has nonzero bootstrap Q")
        bootstrap = None
    else:
        bootstrap = _validate_decision_evidence(
            raw_decisions[-1],
            label=f"{label} bootstrap decision",
            rollout_piece=COUNTERFACTUAL_HORIZON + 1,
            kind="bootstrap",
        )
        _require(
            bootstrap["selected_q"] == bootstrap["max_q"] == bootstrap_q,
            f"{label} bootstrap Q/selection mismatch",
        )

    recomputed_g6 = 0.0
    for step, score_delta in enumerate(score_deltas, 1):
        recomputed_g6 += (
            COUNTERFACTUAL_GAMMA_POWERS[step - 1]
            * score_delta
            / COUNTERFACTUAL_SCORE_NORMALIZER
        )
    if not terminal:
        recomputed_g6 += COUNTERFACTUAL_GAMMA_POWERS[COUNTERFACTUAL_HORIZON] * bootstrap_q
    return_g6 = _exact_number_equal(
        _required(value, "return_G6"), recomputed_g6, f"{label} return_G6"
    )
    return {
        "return_G6": return_g6,
        "terminal": terminal,
        "terminal_step": terminal_step,
        "branch_start_future_stream_digest": branch_future,
        "exact_outcome_digest": exact_outcome_digest,
        "root_decision": first,
        "bootstrap_decision": bootstrap,
    }


def _validate_production_row_science(row: dict[str, Any], *, label: str) -> None:
    """Independently reconstruct both six-piece returns and their advantage."""
    root_future = _lowercase_digest(
        _required(row, "root_future_stream_digest"), f"{label} root future stream"
    )
    root_state = _lowercase_digest(_required(row, "root_state_digest"), f"{label} root state")
    valid_count = _integer(_required(row, "valid_action_count"), f"{label} valid action count")
    top1_index = _integer(
        _required(row, "canonical_top1_candidate_index"), f"{label} top1 index"
    )
    top2_index = _integer(
        _required(row, "canonical_top2_candidate_index"), f"{label} top2 index"
    )
    top1_digest = _lowercase_digest(
        _required(row, "canonical_top1_action_digest"), f"{label} top1 action"
    )
    top2_digest = _lowercase_digest(
        _required(row, "canonical_top2_action_digest"), f"{label} top2 action"
    )
    q1 = _number(_required(row, "q_top1"), f"{label} q_top1")
    q2 = _number(_required(row, "q_top2"), f"{label} q_top2")

    top1 = _validate_production_branch(
        _required(row, "top1_branch"),
        label=f"{label} top1 branch",
        root_state_digest=root_state,
        root_future_stream_digest=root_future,
        root_candidate_count=valid_count,
        root_selected_index=top1_index,
        root_selected_action_digest=top1_digest,
        root_selected_q=q1,
        root_max_q=q1,
    )
    top2 = _validate_production_branch(
        _required(row, "top2_branch"),
        label=f"{label} top2 branch",
        root_state_digest=root_state,
        root_future_stream_digest=root_future,
        root_candidate_count=valid_count,
        root_selected_index=top2_index,
        root_selected_action_digest=top2_digest,
        root_selected_q=q2,
        root_max_q=q1,
    )
    _require(
        top1["branch_start_future_stream_digest"]
        == top2["branch_start_future_stream_digest"]
        == root_future,
        f"{label} branch-start future streams differ",
    )
    _require(
        top1["root_decision"]["candidate_order_digest"]
        == top2["root_decision"]["candidate_order_digest"],
        f"{label} root candidate-order evidence differs",
    )
    _require(
        top1["root_decision"]["q_vector_digest"]
        == top2["root_decision"]["q_vector_digest"],
        f"{label} root Q-vector evidence differs",
    )
    _require(
        _bool(_required(row, "a1_terminal_within_horizon"), f"{label} a1 terminal")
        is top1["terminal"],
        f"{label} top1 terminal alias mismatch",
    )
    _require(
        _bool(_required(row, "a2_terminal_within_horizon"), f"{label} a2 terminal")
        is top2["terminal"],
        f"{label} top2 terminal alias mismatch",
    )
    _exact_number_equal(_required(row, "g6_top1"), top1["return_G6"], f"{label} g6_top1")
    _exact_number_equal(_required(row, "g6_top2"), top2["return_G6"], f"{label} g6_top2")
    recomputed_advantage = top2["return_G6"] - top1["return_G6"]
    advantage = _exact_number_equal(
        _required(row, "advantage_unclipped_A6"),
        recomputed_advantage,
        f"{label} advantage_unclipped_A6",
    )
    _exact_number_equal(_required(row, "advantage"), advantage, f"{label} advantage alias")
    _exact_number_equal(
        _required(row, "clipped_target"),
        max(-2.0, min(2.0, advantage)),
        f"{label} clipped target",
    )


def _validate_repeatability_sentinels(
    table: dict[str, Any],
    metadata: dict[str, Any],
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    episode_evidence: list[dict[str, Any]],
    *,
    training: bool,
    synthetic: bool,
) -> float:
    sentinels = table.get("repeatability_sentinels", []) if synthetic else _required(table, "repeatability_sentinels")
    _require(isinstance(sentinels, list), "repeatability sentinel list missing")
    expected_count = 0 if synthetic or not training else 1
    _require(len(sentinels) == expected_count, "repeatability sentinel count mismatch")
    episode_sentinels = [item.get("repeatability_sentinel") for item in episode_evidence]
    _require(
        sum(item is not None for item in episode_sentinels) == expected_count,
        "per-episode repeatability sentinel count mismatch",
    )
    probe_seconds = _number(
        manifest.get("repeatability_probe_seconds", 0.0) if synthetic else _required(manifest, "repeatability_probe_seconds"),
        "repeatability duration",
    )
    _require(
        _same_number(metadata.get("repeatability_probe_seconds", 0.0) if synthetic else _required(metadata, "repeatability_probe_seconds"), probe_seconds),
        "repeatability duration differs between metadata and manifest",
    )
    if expected_count == 0:
        _require(not sentinels and all(item is None for item in episode_sentinels), "forged repeatability sentinel")
        _require(probe_seconds == 0.0, "unexpected repeatability duration")
        return probe_seconds

    sentinel = sentinels[0]
    expected_keys = {
        "schema", "seed", "episode_id", "piece_index", "root_state_digest",
        "root_future_stream_digest", "repetitions_per_branch",
        "added_branch_rollouts", "added_rollout_pieces",
        "reference_root_candidate_order_digest", "repeated_root_candidate_order_digest",
        "reference_root_q_vector_digest", "repeated_root_q_vector_digest",
        "reference_top1_outcome_digest", "repeated_top1_outcome_digest",
        "reference_top2_outcome_digest", "repeated_top2_outcome_digest",
        "elapsed_seconds",
    }
    _require(isinstance(sentinel, dict) and set(sentinel) == expected_keys, "repeatability sentinel schema mismatch")
    _require(_required(sentinel, "schema") == "r1-repeatability-sentinel-v1", "repeatability sentinel version mismatch")
    seed = _integer(_required(sentinel, "seed"), "repeatability seed")
    piece = _integer(_required(sentinel, "piece_index"), "repeatability piece")
    _require(seed == TRAIN_EPISODES[0], "repeatability sentinel seed is not frozen")
    _require(_integer(_required(sentinel, "episode_id"), "repeatability episode") == seed, "repeatability episode mismatch")
    matched = [row for row in rows if row["episode_id"] == seed and row["piece_index"] == piece]
    _require(len(matched) == 1, "repeatability sentinel does not identify one retained row")
    row = matched[0]
    _require(_required(sentinel, "root_state_digest") == _required(row, "root_state_digest"), "repeatability root mismatch")
    _require(_required(sentinel, "root_future_stream_digest") == _required(row, "root_future_stream_digest"), "repeatability future mismatch")
    for prefix, evidence_key in (
        ("root_candidate_order", "candidate_order_digest"),
        ("root_q_vector", "q_vector_digest"),
    ):
        reference = _digest(_required(sentinel, f"reference_{prefix}_digest"), f"repeatability {prefix}")
        repeated = _digest(_required(sentinel, f"repeated_{prefix}_digest"), f"repeated {prefix}")
        _require(reference == repeated, f"repeatability {prefix} mismatch")
        for branch_name in ("top1_branch", "top2_branch"):
            decision = _required(_required(row, branch_name), "decision_evidence")[0]
            _require(reference == _digest(_required(decision, evidence_key), f"{branch_name} {prefix}"), f"repeatability {prefix} differs from retained row")
    for branch in ("top1", "top2"):
        reference = _digest(_required(sentinel, f"reference_{branch}_outcome_digest"), f"repeatability {branch}")
        repeated = _digest(_required(sentinel, f"repeated_{branch}_outcome_digest"), f"repeated {branch}")
        _require(reference == repeated, f"repeatability {branch} outcome mismatch")
        row_digest = _digest(_required(_required(row, f"{branch}_branch"), "exact_outcome_digest"), f"row {branch} outcome")
        _require(reference == row_digest, f"repeatability {branch} differs from retained row")
    _require(_integer(_required(sentinel, "repetitions_per_branch"), "repeatability repetitions") == 2, "repeatability count mismatch")
    _require(_integer(_required(sentinel, "added_branch_rollouts"), "repeatability rollouts") == 2, "repeatability rollout count mismatch")
    expected_pieces = sum(
        _integer(_required(_required(row, name), "pieces_played"), f"{name} pieces")
        for name in ("top1_branch", "top2_branch")
    )
    _require(_integer(_required(sentinel, "added_rollout_pieces"), "repeatability pieces") == expected_pieces, "repeatability piece count mismatch")
    _require(_same_number(_required(sentinel, "elapsed_seconds"), probe_seconds), "repeatability duration mismatch")
    episode = next(item for item in episode_evidence if item["episode_id"] == seed)
    _require(_required(episode, "repeatability_sentinel") == sentinel, "episode/table sentinel mismatch")
    _require(probe_seconds >= 0.0, "negative repeatability duration")
    return probe_seconds


def validate_collection(
    table_path: Path,
    manifest_path: Path,
    table: dict[str, Any],
    manifest: dict[str, Any],
    *,
    role: str,
    synthetic: bool,
    feature_names: list[str],
    table_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    training = role == "training"
    episodes = TRAIN_EPISODES if training else CALIBRATION_EPISODES
    total = 288 if training else 144
    minimum = 240 if training else 120
    schema = "r1-training-table-v1" if training else "r1-calibration-table-v1"
    _require(_required(table, "schema_version") == schema, f"{role} table schema mismatch")
    _require(_required(table, "source_role") == role, f"{role} table role mismatch")
    _require(_bool(_required(table, "synthetic"), f"{role} table synthetic") is synthetic, f"{role} table synthetic mismatch")
    _require(_required(table, "feature_names") == feature_names, f"{role} table feature order mismatch")
    _require(_digest(_required(table, "feature_schema_digest"), f"{role} feature") == FEATURE_DIGEST, f"{role} table feature digest mismatch")
    _require_scope_false(table, ("validation_seed_used", "sealed_test_seed_used"), f"{role} table")
    metadata = _required(table, "metadata")
    _require(isinstance(metadata, dict), f"{role} metadata missing")
    _require(_required(metadata, "source_role") == role, f"{role} metadata role mismatch")
    _require(_bool(_required(metadata, "synthetic"), f"{role} metadata synthetic") is synthetic, f"{role} metadata synthetic mismatch")
    _require(_required(metadata, "feature_names") == feature_names, f"{role} metadata feature order mismatch")
    _require(_digest(_required(metadata, "feature_schema_digest"), f"{role} metadata feature") == FEATURE_DIGEST, f"{role} metadata feature digest mismatch")
    _require([_integer(value, f"{role} metadata seed") for value in _required(metadata, "role_seeds")] == list(episodes), f"{role} metadata seed role mismatch")
    expected_training_seeds = list(TRAIN_EPISODES) if training else []
    _require([_integer(value, f"{role} metadata training seed") for value in _required(metadata, "training_seeds")] == expected_training_seeds, f"{role} metadata training-seed field mismatch")
    _require([_integer(value, f"{role} table training seed") for value in _required(table, "training_seeds")] == expected_training_seeds, f"{role} table training-seed field mismatch")
    _require([_integer(value, f"{role} metadata piece") for value in _required(metadata, "sample_pieces")] == list(SAMPLE_PIECES), f"{role} metadata sample schedule mismatch")
    _require_scope_false(metadata, ("validation_seed_used", "sealed_test_seed_used"), f"{role} metadata")
    rows = _required(table, "rows")
    _require(isinstance(rows, list) and minimum <= len(rows) <= total, f"{role} row cardinality outside [{minimum},{total}]")
    seen: set[tuple[int, int]] = set()
    seen_episodes: set[int] = set()
    seen_roots: set[str] = set()
    previous_identity: tuple[int, int] | None = None
    positives = 0
    for index, row in enumerate(rows, 1):
        _require(isinstance(row, dict), f"{role} row {index} is not an object")
        episode = _integer(_required(row, "episode_id"), f"{role} row {index} episode")
        seed = _integer(_required(row, "seed"), f"{role} row {index} seed")
        piece = _integer(_required(row, "piece_index"), f"{role} row {index} piece")
        _require(episode == seed and episode in episodes and piece in SAMPLE_PIECES, f"{role} row {index} escaped its exact role")
        identity = (episode, piece)
        _require(identity not in seen, f"duplicate {role} row identity {identity}")
        _require(previous_identity is None or previous_identity < identity, f"{role} table row order is not canonical")
        previous_identity = identity
        seen.add(identity)
        seen_episodes.add(episode)
        features = _required(row, "features")
        _require(isinstance(features, list) and len(features) == FEATURE_COUNT, f"{role} row {index} feature width mismatch")
        numeric_features = [_number(value, f"{role} row {index} feature") for value in features]
        advantage = _number(_required(row, "advantage_unclipped_A6"), f"{role} row {index} A6")
        _require(_same_number(_required(row, "advantage"), advantage), f"{role} row {index} advantage aliases differ")
        _require(_same_number(_required(row, "clipped_target"), max(-2.0, min(2.0, advantage))), f"{role} row {index} clipped target mismatch")
        valid_count = _integer(_required(row, "valid_action_count"), f"{role} row {index} action count")
        _require(valid_count >= 2 and numeric_features[6] == valid_count, f"{role} row {index} candidate count mismatch")
        q1 = _number(_required(row, "q_top1"), f"{role} row {index} q1")
        q2 = _number(_required(row, "q_top2"), f"{role} row {index} q2")
        gap = _number(_required(row, "q_gap"), f"{role} row {index} q gap")
        _require(q1 >= q2 and _same_number(gap, q1 - q2), f"{role} row {index} old-Q ordering mismatch")
        _require(_same_number(numeric_features[0], q1) and _same_number(numeric_features[1], q2) and _same_number(numeric_features[2], gap), f"{role} row {index} Q feature mismatch")
        top1 = _integer(_required(row, "canonical_top1_candidate_index"), f"{role} row {index} top1 index")
        top2 = _integer(_required(row, "canonical_top2_candidate_index"), f"{role} row {index} top2 index")
        _require(1 <= top1 <= valid_count and 1 <= top2 <= valid_count and top1 != top2, f"{role} row {index} top-2 index mismatch")
        digest1 = str(_required(row, "canonical_top1_action_digest"))
        digest2 = str(_required(row, "canonical_top2_action_digest"))
        _require(digest1 and digest2 and digest1 != digest2, f"{role} row {index} action digest mismatch")
        root_digest = str(_required(row, "root_state_digest"))
        _require(root_digest and root_digest not in seen_roots and str(_required(row, "root_future_stream_digest")), f"{role} row {index} root/future evidence missing or duplicated")
        seen_roots.add(root_digest)
        _bool(_required(row, "a1_terminal_within_horizon"), f"{role} row {index} a1 terminal")
        _bool(_required(row, "a2_terminal_within_horizon"), f"{role} row {index} a2 terminal")
        if not synthetic:
            _validate_production_row_science(row, label=f"{role} row {index}")
        positives += advantage > 0.0
    row_order_sha256 = hashlib.sha256(
        "\n".join(
            f"{_integer(_required(row, 'episode_id'), f'{role} row episode')},"
            f"{_integer(_required(row, 'piece_index'), f'{role} row piece')},"
            f"{str(_required(row, 'root_state_digest'))}"
            for row in rows
        ).encode("utf-8")
    ).hexdigest()
    positive_fraction = positives / len(rows)
    _require(_same_number(_required(metadata, "positive_advantage_fraction"), positive_fraction), f"{role} metadata positive fraction mismatch")

    _require(_required(manifest, "schema_version") == "r1-collection-manifest-v1", f"{role} manifest schema mismatch")
    _require(_required(manifest, "source_role") == role, f"{role} manifest role mismatch")
    _require(_bool(_required(manifest, "synthetic"), f"{role} manifest synthetic") is synthetic, f"{role} manifest synthetic mismatch")
    _require(_normal(_required(manifest, "table_path")) == _normal(table_path), f"{role} manifest table path mismatch")
    _require(_digest(_required(manifest, "table_sha256"), f"{role} table") == table_sha256, f"{role} manifest/table hash mismatch")
    _require(_required(manifest, "feature_names") == feature_names, f"{role} manifest feature order mismatch")
    _require(_digest(_required(manifest, "feature_schema_digest"), f"{role} feature") == FEATURE_DIGEST, f"{role} manifest feature digest mismatch")
    _require([_integer(value, f"{role} manifest seed") for value in _required(manifest, "role_seeds")] == list(episodes), f"{role} manifest seed role mismatch")
    _require(_integer(_required(manifest, "episode_count"), f"{role} manifest episode count") == len(episodes), f"{role} manifest episode count mismatch")
    _require(_integer(_required(manifest, "row_count"), f"{role} manifest row count") == len(rows), f"{role} manifest row count mismatch")
    exclusions = _required(manifest, "exclusions")
    _require(isinstance(exclusions, list), f"{role} exclusions missing")
    _require(_integer(_required(manifest, "exclusion_count"), f"{role} manifest exclusion count") == len(exclusions), f"{role} exclusion count mismatch")
    _require(len(rows) + len(exclusions) == total, f"{role} rows/exclusions do not cover exact schedule")
    excluded_identities: set[tuple[int, int]] = set()
    for index, item in enumerate(exclusions, 1):
        _require(isinstance(item, dict), f"{role} exclusion {index} is not an object")
        episode = _integer(_required(item, "episode_id"), f"{role} exclusion {index} episode")
        seed = _integer(_required(item, "seed"), f"{role} exclusion {index} seed")
        piece = _integer(_required(item, "piece_index"), f"{role} exclusion {index} piece")
        identity = (episode, piece)
        _require(episode == seed and episode in episodes and piece in SAMPLE_PIECES, f"{role} exclusion {index} escaped role")
        _require(identity not in seen and identity not in excluded_identities, f"{role} exclusion identity duplicated/also eligible: {identity}")
        code = str(_required(item, "code"))
        detail = str(_required(item, "detail"))
        if code == "canonical_trajectory_unavailable":
            _require(
                detail in (
                    "reason=terminal;canonical trajectory unavailable before scheduled sample",
                    "reason=no_canonical_action;canonical trajectory unavailable before scheduled sample",
                ),
                f"{role} unavailable-slot reason is not frozen",
            )
        excluded_identities.add(identity)
    exact_schedule = {(episode, piece) for episode in episodes for piece in SAMPLE_PIECES}
    _require(seen | excluded_identities == exact_schedule, f"{role} eligible/excluded identities do not exactly cover the frozen schedule")
    episode_evidence = _required(manifest, "episodes")
    _require(isinstance(episode_evidence, list) and len(episode_evidence) == len(episodes), f"{role} per-episode evidence mismatch")
    _require([_integer(_required(item, "seed"), f"{role} episode seed") for item in episode_evidence] == list(episodes), f"{role} per-episode seed order mismatch")
    _require([_integer(_required(item, "episode_id"), f"{role} episode id") for item in episode_evidence] == list(episodes), f"{role} per-episode id order mismatch")
    actual_rows_by_episode = {episode: sum(row["episode_id"] == episode for row in rows) for episode in episodes}
    actual_exclusions_by_episode = {episode: sum(item["episode_id"] == episode for item in exclusions) for episode in episodes}
    for item in episode_evidence:
        episode = _integer(_required(item, "episode_id"), f"{role} episode id")
        _require(_integer(_required(item, "rows"), f"{role} episode rows") == actual_rows_by_episode[episode], f"{role} per-episode row count mismatch for {episode}")
        _require(_integer(_required(item, "exclusions"), f"{role} episode exclusions") == actual_exclusions_by_episode[episode], f"{role} per-episode exclusion count mismatch for {episode}")
    unavailable_count = sum(
        str(_required(item, "code")) == "canonical_trajectory_unavailable"
        for item in exclusions
    )
    attempted = len(rows) + len(exclusions) - unavailable_count
    for container, label in ((metadata, "metadata"), (manifest, "manifest")):
        _require(
            _integer(_required(container, "counterfactual_states_completed"), f"{role} {label} completed states") == attempted,
            f"{role} {label} completed-state mismatch",
        )
        _require(
            _integer(_required(container, "counterfactual_states_attempted"), f"{role} {label} attempted states") == attempted,
            f"{role} {label} attempted-state mismatch",
        )
        _require(
            _integer(_required(container, "scheduled_slots_accounted"), f"{role} {label} accounted slots") == total,
            f"{role} {label} schedule-accounting mismatch",
        )
    repeatability_probe_seconds = _validate_repeatability_sentinels(
        table,
        metadata,
        manifest,
        rows,
        episode_evidence,
        training=training,
        synthetic=synthetic,
    )
    _require(_same_number(_required(manifest, "positive_advantage_fraction"), positive_fraction), f"{role} manifest positive fraction mismatch")
    _require_scope_false(manifest, ("validation_seed_used", "sealed_test_seed_used"), f"{role} manifest")
    _require(_bool(_required(manifest, "real_model_or_game_loaded"), f"{role} real-model flag") is (not synthetic), f"{role} model/game mode mismatch")
    stable = _digest(_required(manifest, "stable_node_key_source_sha256"), f"{role} stable-order source")
    _require(_digest(_required(metadata, "stable_node_key_source_sha256"), f"{role} metadata stable source") == stable, f"{role} stable-order evidence mismatch")
    # The real collector always emits these fields.  The fitter's built-in
    # synthetic fixture predates them and legitimately omits them; synthetic
    # mode accepts only absent/null, never forged production evidence.
    metadata_backend = metadata.get("backend_binding")
    manifest_backend = manifest.get("backend_binding")
    metadata_dependency_graph = metadata.get("engine_dependency_graph")
    manifest_dependency_graph = manifest.get("engine_dependency_graph")
    metadata_immutable = metadata.get("immutable_input_end_hashes")
    manifest_immutable = manifest.get("immutable_input_end_hashes")
    _validate_backend_evidence(metadata_backend, synthetic=synthetic, label=f"{role} metadata")
    _validate_backend_evidence(manifest_backend, synthetic=synthetic, label=f"{role} manifest")
    _require(metadata_backend == manifest_backend, f"{role} backend evidence differs")
    normalized_metadata_graph = _validate_engine_dependency_graph(
        metadata_dependency_graph,
        synthetic=synthetic,
        label=f"{role} metadata",
    )
    normalized_manifest_graph = _validate_engine_dependency_graph(
        manifest_dependency_graph,
        synthetic=synthetic,
        label=f"{role} manifest",
    )
    _require(
        normalized_metadata_graph == normalized_manifest_graph,
        f"{role} engine dependency graph differs",
    )
    _validate_immutable_evidence(metadata_immutable, synthetic=synthetic, label=f"{role} metadata")
    _validate_immutable_evidence(manifest_immutable, synthetic=synthetic, label=f"{role} manifest")
    _require(metadata_immutable == manifest_immutable, f"{role} immutable evidence differs")
    if training:
        setup = _number(_required(manifest, "first32_setup_seconds"), "training first32 setup")
        elapsed = _number(_required(manifest, "first32_elapsed_seconds"), "training first32 elapsed")
        projected = _number(_required(manifest, "first32_projected_seconds"), "training projection")
        _require(_integer(_required(manifest, "first32_projection_basis_states"), "training projection basis") == 432, "training projection basis mismatch")
        _require(_required(manifest, "first32_projection_formula") == PROJECTION_FORMULA, "training projection formula mismatch")
        _require(_number(_required(manifest, "first32_projection_limit_seconds"), "projection limit") == 3300.0, "training projection limit mismatch")
        _require(0.0 <= repeatability_probe_seconds <= elapsed, "training repeatability duration exceeds first32 elapsed")
        _require(
            _same_number(
                projected,
                2.0 * setup
                + (elapsed - repeatability_probe_seconds) / 32.0 * 432.0
                + repeatability_probe_seconds,
                tolerance=1.0e-10,
            ),
            "training projection arithmetic mismatch",
        )
        _require(0.0 <= projected <= 3300.0, "training projection exceeds preregistration")
        for key in (
            "first32_setup_seconds", "first32_elapsed_seconds", "first32_projected_seconds",
            "first32_projection_limit_seconds", "first32_projection_basis_states", "first32_projection_formula",
        ):
            _require(_required(metadata, key) == _required(manifest, key), f"training projection evidence differs at {key}")
    return {
        "rows": rows,
        "positive_fraction": positive_fraction,
        "row_count": len(rows),
        "table_sha256": table_sha256,
        "manifest_sha256": manifest_sha256,
        "stable_node_key_source_sha256": stable,
        "backend_binding": manifest_backend,
        "engine_dependency_graph": normalized_manifest_graph,
        "row_order_sha256": row_order_sha256,
    }


def validate_ridge(
    path: Path,
    artifact: dict[str, Any],
    *,
    synthetic: bool,
    training: dict[str, Any],
    training_table_path: Path,
    training_manifest_path: Path,
    freeze_path: Path,
    feature_names: list[str],
    artifact_sha256: str,
    freeze_sha256: str,
) -> dict[str, Any]:
    _require(_required(artifact, "schema_version") == "r1-ridge-gate-v1", "ridge schema mismatch")
    _require(_required(artifact, "experiment_id") == EXPERIMENT_ID, "ridge experiment mismatch")
    _require(_required(artifact, "fit_role") == "training_only", "ridge fit role mismatch")
    _require(_required(artifact, "fit_backend") == "python_numpy_analytic_ridge", "ridge backend mismatch")
    _require(_bool(_required(artifact, "source_table_synthetic"), "ridge synthetic") is synthetic, "ridge synthetic mode mismatch")
    _require(_bool(_required(artifact, "all_finite"), "ridge finite") is True, "ridge not finite")
    _require_scope_false(artifact, ("validation_seed_used", "sealed_test_seed_used"), "ridge")
    _require(_required(artifact, "claim_scope") == "analytic_training_artifact_not_calibration_or_game_strength", "ridge claim scope mismatch")
    artifact_dependency_graph = _validate_engine_dependency_graph(
        _required(artifact, "engine_dependency_graph"),
        synthetic=synthetic,
        label="ridge artifact",
    )
    _require(
        artifact_dependency_graph == training["engine_dependency_graph"],
        "ridge artifact engine dependency graph differs from training collection",
    )
    _require(
        _required(artifact, "backend_binding") == training["backend_binding"],
        "ridge artifact backend binding differs from training collection",
    )
    _require(_normal(_required(artifact, "source_table_path")) == _normal(training_table_path), "ridge training table path mismatch")
    _require(_digest(_required(artifact, "source_table_sha256"), "ridge source table") == training["table_sha256"], "ridge training table hash mismatch")
    _require(_normal(_required(artifact, "source_collection_manifest_path")) == _normal(training_manifest_path), "ridge training manifest path mismatch")
    _require(_digest(_required(artifact, "source_collection_manifest_sha256"), "ridge source manifest") == training["manifest_sha256"], "ridge training manifest hash mismatch")
    _require(_normal(_required(artifact, "design_freeze_path")) == _normal(freeze_path), "ridge freeze path mismatch")
    _require(_digest(_required(artifact, "design_freeze_sha256"), "ridge freeze") == freeze_sha256, "ridge freeze hash mismatch")
    _require(_bool(_required(artifact, "training_bootstrap_schedule_consumed"), "ridge schedule consumed") is True, "ridge did not consume frozen schedule")
    _require(_digest(_required(artifact, "training_bootstrap_schedule_sha256"), "ridge schedule") == TRAIN_SCHEDULE_DIGEST, "ridge schedule mismatch")
    _require(_digest(_required(artifact, "training_bootstrap_schedule_source_anchor_sha256"), "ridge source anchor") == TRAIN_SCHEDULE_DIGEST, "ridge schedule anchor mismatch")
    _require(
        _required(artifact, "training_row_order_encoding")
        == "episode_id,piece_index,root_state_digest newline joined",
        "ridge training row-order encoding mismatch",
    )
    _require(
        _digest(_required(artifact, "training_row_order_sha256"), "ridge training row order")
        == training["row_order_sha256"],
        "ridge training row-order digest mismatch",
    )
    _require(_required(artifact, "feature_names") == feature_names, "ridge feature order mismatch")
    _require(_digest(_required(artifact, "feature_schema_digest"), "ridge feature") == FEATURE_DIGEST, "ridge feature digest mismatch")
    means = [_number(value, "ridge feature mean") for value in _required(artifact, "feature_mean")]
    scales = [_number(value, "ridge feature scale") for value in _required(artifact, "feature_scale")]
    constant = [_bool(value, "ridge constant feature") for value in _required(artifact, "constant_feature")]
    _require(len(means) == len(scales) == len(constant) == FEATURE_COUNT and all(value > 0.0 for value in scales), "ridge standardization shape/value mismatch")
    coefficient_rows = _required(artifact, "coefficients")
    _require(isinstance(coefficient_rows, list) and len(coefficient_rows) == COEFFICIENT_COUNT, "ridge coefficient row count mismatch")
    coefficients = [[_number(value, "ridge coefficient") for value in row] for row in coefficient_rows]
    _require(all(len(row) == ENSEMBLE_COUNT for row in coefficients), "ridge coefficient member count mismatch")
    _require([_integer(value, "ridge coefficient shape") for value in _required(artifact, "coefficient_shape")] == [COEFFICIENT_COUNT, ENSEMBLE_COUNT], "ridge coefficient shape metadata mismatch")
    _require(_integer(_required(artifact, "ensemble_size"), "ridge ensemble size") == ENSEMBLE_COUNT, "ridge ensemble mismatch")
    _require(_number(_required(artifact, "lambda"), "ridge lambda") == 1.0, "ridge lambda mismatch")
    _require(_required(artifact, "bootstrap_rng") == "Xoshiro(0x5231_2026)" and _integer(_required(artifact, "bootstrap_seed"), "ridge bootstrap seed") == 0x5231_2026, "ridge bootstrap identity mismatch")
    _require(_number(_required(artifact, "lower_quantile"), "ridge quantile") == 0.10, "ridge quantile mismatch")
    _require(_required(artifact, "quantile_method") == QUANTILE_METHOD, "ridge quantile method mismatch")
    _require(_number(_required(artifact, "override_threshold"), "ridge threshold") == 0.05, "ridge threshold mismatch")
    _require([_number(value, "ridge target clamp") for value in _required(artifact, "target_clamp")] == [-2.0, 2.0], "ridge target clamp mismatch")
    stats = _required(artifact, "training_stats")
    _require(_integer(_required(stats, "row_count"), "ridge training row count") == training["row_count"], "ridge training row count mismatch")
    _require(_integer(_required(stats, "episode_count"), "ridge training episode count") == 12 and [_integer(value, "ridge training episode id") for value in _required(stats, "episode_ids")] == list(TRAIN_EPISODES), "ridge training episode evidence mismatch")
    _require(_same_number(_required(stats, "positive_fraction_unclipped"), training["positive_fraction"]), "ridge training positive fraction mismatch")
    runtime = _required(artifact, "runtime_facts")
    _require(_required(runtime, "numpy_version") == "2.4.6", "ridge NumPy version mismatch")
    _require(_required(runtime, "python_version") == "3.12.13", "ridge Python version mismatch")
    _require(_required(runtime, "linear_algebra") == "numpy.linalg.cholesky+numpy.linalg.solve", "ridge linear algebra backend mismatch")
    _require(_normal(_required(runtime, "python_executable")) == _normal(sys.executable), "ridge recorded Python executable differs from finalizer")
    return {
        "means": means,
        "scales": scales,
        "constant": constant,
        "coefficients": coefficients,
        "artifact_sha256": artifact_sha256,
        "engine_dependency_graph": artifact_dependency_graph,
    }


def _type7(values: Sequence[float], probability: float = 0.10) -> float:
    _require(bool(values), "quantile of empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _lower_bound(features: Sequence[Any], ridge: dict[str, Any]) -> float:
    raw = [_number(value, "calibration feature") for value in features]
    predictions: list[float] = []
    for member in range(ENSEMBLE_COUNT):
        value = ridge["coefficients"][0][member]
        for feature in range(FEATURE_COUNT):
            standardized = 0.0 if ridge["constant"][feature] else (
                (raw[feature] - ridge["means"][feature]) / ridge["scales"][feature]
            )
            value += ridge["coefficients"][feature + 1][member] * standardized
        _require(math.isfinite(value), "non-finite independent ridge prediction")
        predictions.append(value)
    return _type7(predictions)


def _float64_bitstring(value: Any, label: str) -> str:
    number = _number(value, label)
    return f"{struct.unpack('>Q', struct.pack('>d', number))[0]:064b}"


def _deployment_evidence(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    value = _required(row, "deployment_decision")
    _require(isinstance(value, dict), f"calibration row {row_index} lacks deployment evidence")
    _require(_required(value, "deployment_decision_schema_version") == DEPLOYMENT_SCHEMA, "deployment schema mismatch")
    feature_schema = _digest(_required(value, "production_feature_schema_digest"), "deployment feature schema")
    _require(feature_schema == FEATURE_DIGEST, "deployment feature schema mismatch")
    _require(_required(value, "production_feature_digest_encoding") == DEPLOYMENT_FEATURE_ENCODING, "deployment feature encoding mismatch")
    live_features = list(_required(value, "production_feature_vector"))
    row_features = list(_required(row, "features"))
    _require(len(live_features) == FEATURE_COUNT and len(row_features) == FEATURE_COUNT, "deployment feature width mismatch")
    _require(
        all(
            _float64_bitstring(left, "deployment feature")
            == _float64_bitstring(right, "row feature")
            for left, right in zip(live_features, row_features, strict=True)
        ),
        "deployment feature vector differs from collected row",
    )
    feature_payload = "\n".join(
        [feature_schema, *(_float64_bitstring(item, "deployment feature") for item in live_features)]
    ).encode("utf-8")
    _require(
        _digest(_required(value, "production_feature_vector_sha256"), "deployment feature digest")
        == hashlib.sha256(feature_payload).hexdigest(),
        "deployment feature digest mismatch",
    )
    decision = _bool(_required(value, "production_decision"), "deployment decision")
    lower_bound = _number(_required(value, "production_gate_lower_bound"), "deployment lower bound")
    _require(decision is (lower_bound > 0.05), "deployment threshold decision mismatch")
    _require(
        _required(value, "production_gate_fallback_reason")
        == ("none" if decision else "lower_bound_not_above_threshold"),
        "deployment fallback reason mismatch",
    )
    top1 = _integer(_required(value, "canonical_top1_candidate_index"), "deployment top1 index")
    top2 = _integer(_required(value, "canonical_top2_candidate_index"), "deployment top2 index")
    _require(top1 == _integer(_required(row, "canonical_top1_candidate_index"), "row top1 index"), "deployment top1 index mismatch")
    _require(top2 == _integer(_required(row, "canonical_top2_candidate_index"), "row top2 index"), "deployment top2 index mismatch")
    top1_action = str(_required(value, "canonical_top1_action_digest"))
    top2_action = str(_required(value, "canonical_top2_action_digest"))
    _require(top1_action == str(_required(row, "canonical_top1_action_digest")), "deployment top1 action mismatch")
    _require(top2_action == str(_required(row, "canonical_top2_action_digest")), "deployment top2 action mismatch")
    top1_node = str(_required(value, "canonical_top1_node_identity"))
    top2_node = str(_required(value, "canonical_top2_node_identity"))
    selected_index = _integer(_required(value, "production_selected_candidate_index"), "deployment selected index")
    selected_action = str(_required(value, "production_selected_action_digest"))
    selected_node = str(_required(value, "production_selected_node_identity"))
    expected_index = top2 if decision else top1
    expected_action = top2_action if decision else top1_action
    expected_node = top2_node if decision else top1_node
    selected_mismatch = selected_index != expected_index or selected_action != expected_action or selected_node != expected_node
    applied_mismatch = (
        str(_required(value, "production_applied_action_digest")) != selected_action
        or str(_required(value, "production_applied_node_identity")) != selected_node
    )
    before = str(_required(value, "canonical_state_digest_before"))
    state_mismatch = (
        not before
        or str(_required(value, "canonical_state_digest_after")) != before
        or str(_required(value, "production_clone_state_digest_before")) != before
        or not str(_required(value, "production_applied_clone_state_digest"))
    )
    started = _integer(_required(value, "production_gate_incremental_started_ns"), "deployment start ns")
    finished = _integer(_required(value, "production_gate_incremental_finished_ns"), "deployment finish ns")
    elapsed = _integer(_required(value, "production_gate_incremental_elapsed_ns"), "deployment elapsed ns")
    _require(started >= 0 and finished >= started and elapsed == finished - started, "deployment timing mismatch")
    _require(_required(value, "production_gate_incremental_scope") == DEPLOYMENT_TIMING_SCOPE, "deployment timing scope mismatch")
    binding = "\n".join([DEPLOYMENT_SCHEMA, str(selected_index), selected_action, selected_node]).encode("utf-8")
    _require(
        _digest(_required(value, "production_selection_binding_sha256"), "deployment selection binding")
        == hashlib.sha256(binding).hexdigest(),
        "deployment selection binding mismatch",
    )
    return {
        "lower_bound": lower_bound,
        "decision": decision,
        "selected_mismatch": selected_mismatch or applied_mismatch or state_mismatch,
        "fallback_mismatch": (not decision) and (selected_mismatch or applied_mismatch or state_mismatch),
        "elapsed_ms": elapsed / 1_000_000.0,
    }


def recompute_calibration(
    rows: list[dict[str, Any]],
    ridge: dict[str, Any],
    schedules: list[list[int]],
    measured_overhead: float,
    *,
    synthetic: bool = False,
) -> dict[str, Any]:
    # Recreate the production vectorized NumPy path independently from the
    # calibration module, then compare it with the scalar reference below.
    import numpy as np

    _require(np.__version__ == "2.4.6", "finalizer NumPy version mismatch")
    feature_matrix = np.asarray([row["features"] for row in rows], dtype=np.float64)
    means = np.asarray(ridge["means"], dtype=np.float64)
    scales = np.asarray(ridge["scales"], dtype=np.float64)
    constant = np.asarray(ridge["constant"], dtype=np.bool_)
    coefficients = np.asarray(ridge["coefficients"], dtype=np.float64)
    standardized = (feature_matrix - means[None, :]) / scales[None, :]
    standardized[:, constant] = 0.0
    # Deployment deliberately issues one GEMV per state.  A single GEMM can
    # round across the strict 0.05 boundary differently on the same BLAS.
    predictions = np.empty((feature_matrix.shape[0], ENSEMBLE_COUNT), dtype=np.float64)
    for row_index in range(feature_matrix.shape[0]):
        predictions[row_index, :] = (
            standardized[row_index, :] @ coefficients[1:, :]
            + coefficients[0, :]
        )
    python_lower = np.quantile(predictions, 0.10, axis=1, method="linear")
    _require(bool(np.all(np.isfinite(python_lower))), "non-finite independent Python prediction")
    decisions: list[tuple[int, float, bool, bool, bool]] = []
    reference_lower: list[float] = []
    production_reference_mismatch_count = 0
    selected_action_mismatch_count = 0
    fallback_top1_mismatch_count = 0
    live_overheads: list[float] = []
    live_lower: list[float] = []
    for row_index, row in enumerate(rows):
        reference = _lower_bound(_required(row, "features"), ridge)
        live = (
            {
                "lower_bound": float(python_lower[row_index]),
                "decision": bool(float(python_lower[row_index]) > 0.05),
                "selected_mismatch": False,
                "fallback_mismatch": False,
                "elapsed_ms": measured_overhead,
            }
            if synthetic and row.get("deployment_decision") is None
            else _deployment_evidence(row, row_index + 1)
        )
        production = live["lower_bound"]
        tolerance = 1.0e-10 * max(1.0, abs(reference))
        _require(abs(production - reference) <= tolerance, f"independent production/reference numerical mismatch at row {row_index + 1}")
        _require(abs(float(python_lower[row_index]) - reference) <= tolerance, f"independent Python/reference numerical mismatch at row {row_index + 1}")
        production_decision = live["decision"]
        reference_decision = reference > 0.05
        production_reference_mismatch_count += production_decision != reference_decision
        selected_action_mismatch_count += bool(live["selected_mismatch"])
        fallback_top1_mismatch_count += bool(live["fallback_mismatch"])
        live_overheads.append(float(live["elapsed_ms"]))
        live_lower.append(production)
        reference_lower.append(reference)
        decisions.append(
            (
                int(_required(row, "episode_id")),
                _number(_required(row, "advantage_unclipped_A6"), "calibration A6"),
                production_decision,
                _bool(_required(row, "a1_terminal_within_horizon"), "calibration a1 terminal"),
                _bool(_required(row, "a2_terminal_within_horizon"), "calibration a2 terminal"),
            )
        )
    overridden = [item for item in decisions if item[2]]
    state_count = len(decisions)
    override_count = len(overridden)
    override_episodes = sorted({item[0] for item in overridden})
    precision = sum(item[1] > 0.0 for item in overridden) / override_count if override_count else 0.0
    mean_advantage = math.fsum(item[1] for item in overridden) / override_count if override_count else 0.0
    unsafe = sum(item[2] and item[4] and not item[3] for item in decisions)
    by_episode = {episode: [item[1] for item in overridden if item[0] == episode] for episode in CALIBRATION_EPISODES}
    precision_samples: list[float] = []
    mean_samples: list[float] = []
    empty = 0
    for schedule in schedules:
        sampled = [value for episode in schedule for value in by_episode[episode]]
        if sampled:
            precision_samples.append(sum(value > 0.0 for value in sampled) / len(sampled))
            mean_samples.append(math.fsum(sampled) / len(sampled))
        else:
            empty += 1
            precision_samples.append(0.0)
            mean_samples.append(0.0)
    bootstrap = {
        "precision_lower90": _type7(precision_samples),
        "mean_advantage_lower90": _type7(mean_samples),
        "replicates": 2000,
        "seed": 0x5231_73106,
        "schedule_sha256": CALIBRATION_SCHEDULE_DIGEST,
        "schedule_matches_regenerated": True,
        "lower_quantile": 0.10,
        "empty_override_replicate_count": empty,
    }
    derived_overhead = statistics.median(live_overheads)
    _require(_same_number(measured_overhead, derived_overhead, tolerance=1.0e-12), "reported overhead differs from live row evidence")
    checks = {
        "minimum_state_count": state_count >= 120,
        "exact_calibration_episodes": sorted({item[0] for item in decisions}) == list(CALIBRATION_EPISODES),
        "override_rate_in_range": 0.01 <= override_count / state_count <= 0.15,
        "minimum_override_count": override_count >= 12,
        "override_episode_distribution": len(override_episodes) >= 4,
        "precision_point": precision >= 0.70,
        "precision_lower_bound": bootstrap["precision_lower90"] > 0.50,
        "mean_advantage_point": mean_advantage >= 0.10,
        "mean_advantage_lower_bound": bootstrap["mean_advantage_lower90"] > 0.0,
        "no_top2_only_terminal": unsafe == 0,
        "production_reference_exact": production_reference_mismatch_count == 0,
        "selected_action_matches_decision": selected_action_mismatch_count == 0,
        "fallback_top1_exact": fallback_top1_mismatch_count == 0,
        "overhead_within_budget": derived_overhead <= 0.10,
        "artifact_finite": True,
        "feature_schema_exact": True,
        "coefficient_shape_exact": True,
        "hyperparameters_exact": True,
        "calibration_source_hash_verified": True,
        "no_forbidden_seed_used": True,
    }
    return {
        "state_count": state_count,
        "calibration_episodes": list(CALIBRATION_EPISODES),
        "override_count": override_count,
        "override_rate": override_count / state_count,
        "override_episode_count": len(override_episodes),
        "override_episodes": override_episodes,
        "override_precision": precision,
        "override_mean_unclipped_A6": mean_advantage,
        "unsafe_top2_terminal_count": unsafe,
        "production_reference_mismatch_count": production_reference_mismatch_count,
        "selected_action_mismatch_count": selected_action_mismatch_count,
        "fallback_top1_mismatch_count": fallback_top1_mismatch_count,
        "median_decision_overhead_ms": derived_overhead,
        "bootstrap": bootstrap,
        "checks": checks,
        "promoted": all(checks.values()),
        "production_reference_max_abs_error": float(
            np.max(np.abs(np.asarray(live_lower, dtype=np.float64) - np.asarray(reference_lower, dtype=np.float64)))
        ),
    }


def _validate_hash_evidence(value: Any, path: Path, digest: str, label: str) -> None:
    _require(isinstance(value, dict), f"{label} hash evidence missing")
    _require(_normal(_required(value, "path")) == _normal(path), f"{label} evidence path mismatch")
    _require(_digest(_required(value, "actual_sha256"), f"{label} actual") == digest, f"{label} actual hash mismatch")
    _require(_digest(_required(value, "expected_sha256"), f"{label} expected") == digest, f"{label} expected hash mismatch")
    _require(_bool(_required(value, "verified"), f"{label}.verified") is True, f"{label} hash not verified")


def validate_calibration_assessment(
    assessment: dict[str, Any],
    expected: dict[str, Any],
    *,
    synthetic: bool,
    training_table_path: Path,
    training_manifest_path: Path,
    calibration_table_path: Path,
    calibration_manifest_path: Path,
    ridge_path: Path,
    freeze_path: Path,
    input_sha256: dict[str, str],
    engine_dependency_graph: dict[str, Any] | None,
) -> None:
    allowed_keys = {
        "experiment", "status", "promoted", "scope", "state_count",
        "calibration_episodes", "expected_calibration_episodes",
        "override_count", "override_rate", "override_episode_count",
        "override_episodes", "override_precision",
        "override_mean_unclipped_A6", "unsafe_top2_terminal_count",
        "production_reference_mismatch_count", "selected_action_mismatch_count",
        "fallback_top1_mismatch_count", "median_decision_overhead_ms",
        "bootstrap", "gate_artifact_evidence", "calibration_source_evidence",
        "thresholds", "checks", "validation_seed_used", "sealed_test_seed_used",
        "game_strength_evidence", "model_improvement_evidence", "provenance",
        "runtime_evidence",
        "engine_dependency_graph",
    }
    _require(set(assessment) == allowed_keys, "calibration assessment schema has missing or forbidden fields")
    _require(_required(assessment, "experiment") == "R1_online_counterfactual_top2_safety_gate", "calibration experiment mismatch")
    _require(_required(assessment, "status") == ("R1-calibration-promoted" if expected["promoted"] else "R1-calibration-rejected"), "calibration status disagrees with recomputation")
    _require(_bool(_required(assessment, "promoted"), "calibration promoted") is expected["promoted"], "calibration promotion disagrees with recomputation")
    _require(_required(assessment, "scope") == "calibration_only_not_game_strength_not_model_improvement", "calibration scope mismatch")
    assessment_dependency_graph = _validate_engine_dependency_graph(
        _required(assessment, "engine_dependency_graph"),
        synthetic=synthetic,
        label="calibration assessment",
    )
    _require(
        assessment_dependency_graph == engine_dependency_graph,
        "calibration assessment engine dependency graph differs from collection/ridge",
    )
    integer_fields = (
        "state_count", "override_count", "override_episode_count", "unsafe_top2_terminal_count",
        "production_reference_mismatch_count", "selected_action_mismatch_count", "fallback_top1_mismatch_count",
    )
    for field in integer_fields:
        _require(_integer(_required(assessment, field), f"calibration {field}") == expected[field], f"calibration {field} mismatch")
    for field in ("override_rate", "override_precision", "override_mean_unclipped_A6", "median_decision_overhead_ms"):
        _require(_same_number(_required(assessment, field), expected[field], tolerance=1.0e-10), f"calibration {field} mismatch")
    _require([_integer(value, "calibration episode") for value in _required(assessment, "calibration_episodes")] == expected["calibration_episodes"], "calibration episodes mismatch")
    _require([_integer(value, "expected calibration episode") for value in _required(assessment, "expected_calibration_episodes")] == list(CALIBRATION_EPISODES), "expected calibration episodes mismatch")
    _require([_integer(value, "override episode") for value in _required(assessment, "override_episodes")] == expected["override_episodes"], "override episode identities mismatch")
    observed_bootstrap = _required(assessment, "bootstrap")
    for field, value in expected["bootstrap"].items():
        if type(value) is float:
            _require(_same_number(_required(observed_bootstrap, field), value, tolerance=1.0e-10), f"calibration bootstrap {field} mismatch")
        else:
            _require(_required(observed_bootstrap, field) == value, f"calibration bootstrap {field} mismatch")
    _require(_required(assessment, "checks") == expected["checks"], "calibration gate checks differ from independent recomputation")
    gate_evidence = _required(assessment, "gate_artifact_evidence")
    _require(gate_evidence == {
        "finite": True,
        "feature_schema_exact": True,
        "coefficient_shape_exact": True,
        "hyperparameters_exact": True,
        "numeric_value_count": NUMERIC_VALUE_COUNT,
    }, "calibration ridge evidence mismatch")
    thresholds = _required(assessment, "thresholds")
    _require(thresholds == {
        "minimum_states": 120,
        "override_rate": {"minimum": 0.01, "maximum": 0.15},
        "minimum_overrides": 12,
        "minimum_override_episodes": 4,
        "minimum_precision": 0.70,
        "minimum_precision_lower90": 0.50,
        "minimum_mean_unclipped_A6": 0.10,
        "minimum_mean_A6_lower90": 0.0,
        "maximum_median_overhead_ms": 0.10,
    }, "calibration thresholds changed")
    runtime = _required(assessment, "runtime_evidence")
    one_state_median = _number(_required(runtime, "one_state_latency_median_ms"), "calibration one-state median runtime")
    one_state_minimum = _number(_required(runtime, "one_state_latency_minimum_ms"), "calibration one-state minimum runtime")
    one_state_maximum = _number(_required(runtime, "one_state_latency_maximum_ms"), "calibration one-state maximum runtime")
    amortized = _number(_required(runtime, "batch_amortized_ms_per_state"), "calibration batch-amortized runtime")
    batch_elapsed_ns = _number(_required(runtime, "batch_elapsed_ns"), "calibration batch runtime")
    _require(_same_number(one_state_median, expected["median_decision_overhead_ms"], tolerance=1.0e-10), "calibration one-state runtime/metric overhead mismatch")
    _require(0.0 <= one_state_minimum <= one_state_median <= one_state_maximum, "calibration one-state latency ordering mismatch")
    _require(_same_number(amortized, batch_elapsed_ns / expected["state_count"] / 1_000_000.0, tolerance=1.0e-10), "calibration batch/amortized runtime arithmetic mismatch")
    observed_reference_error = _number(_required(runtime, "production_reference_max_abs_error"), "production/reference error")
    _require(0.0 <= observed_reference_error <= 1.0e-10, "production/reference error exceeds tolerance")
    _require(_same_number(observed_reference_error, expected["production_reference_max_abs_error"], tolerance=1.0e-10), "production/reference error differs from independent recomputation")
    _require_scope_false(assessment, ("validation_seed_used", "sealed_test_seed_used", "game_strength_evidence", "model_improvement_evidence"), "calibration assessment")
    source_evidence = _required(assessment, "calibration_source_evidence")
    _validate_hash_evidence(source_evidence, calibration_table_path, input_sha256["calibration_table"], "calibration source")
    provenance = _required(assessment, "provenance")
    _validate_hash_evidence(_required(provenance, "artifact"), ridge_path, input_sha256["ridge_artifact"], "ridge artifact")
    _validate_hash_evidence(_required(provenance, "training_table"), training_table_path, input_sha256["training_table"], "training table")
    _validate_hash_evidence(_required(provenance, "training_manifest"), training_manifest_path, input_sha256["training_manifest"], "training manifest")
    _validate_hash_evidence(_required(provenance, "design_freeze"), freeze_path, input_sha256["design_freeze"], "design freeze")
    _validate_hash_evidence(_required(provenance, "calibration_table"), calibration_table_path, input_sha256["calibration_table"], "calibration table")
    _validate_hash_evidence(_required(provenance, "calibration_manifest"), calibration_manifest_path, input_sha256["calibration_manifest"], "calibration manifest")
    _require(_bool(_required(provenance, "synthetic"), "calibration provenance synthetic") is synthetic, "calibration provenance synthetic mismatch")
    _require(_required(provenance, "numpy_version") == "2.4.6", "calibration NumPy version mismatch")
    _require(_required(provenance, "python_version") == "3.12.13", "calibration Python version mismatch")
    thread_environment = _required(provenance, "blas_thread_environment")
    _require(
        isinstance(thread_environment, dict)
        and set(thread_environment) == {
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
        }
        and set(thread_environment.values()) == {"1"},
        "calibration BLAS thread backend evidence mismatch",
    )


def assess(args: argparse.Namespace, milestones: Milestones) -> dict[str, Any]:
    observed_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    _require(observed_python == "3.12.13", f"finalizer requires Python 3.12.13, observed {observed_python}")
    inputs = {
        "training_table": args.training_table.resolve(),
        "training_manifest": args.training_manifest.resolve(),
        "ridge_artifact": args.ridge_artifact.resolve(),
        "calibration_table": args.calibration_table.resolve(),
        "calibration_manifest": args.calibration_manifest.resolve(),
        "calibration_assessment": args.calibration_assessment.resolve(),
        "design_freeze": args.design_freeze.resolve(),
    }
    for label, path in inputs.items():
        _require(path.is_file(), f"missing {label}: {path}")
    # Capture each explicit input exactly once.  All parsing, SHA handoff
    # checks, provenance validation, and output attestations below refer to
    # these same immutable byte snapshots, closing inter-call replacement.
    snapshots = {name: path.read_bytes() for name, path in inputs.items()}
    hashes = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in snapshots.items()
    }
    handoff_environment = {
        "training_table": "R1_EXPECTED_TRAINING_TABLE_SHA256",
        "training_manifest": "R1_EXPECTED_TRAINING_MANIFEST_SHA256",
        "calibration_table": "R1_EXPECTED_CALIBRATION_TABLE_SHA256",
        "calibration_manifest": "R1_EXPECTED_CALIBRATION_MANIFEST_SHA256",
        "ridge_artifact": "R1_EXPECTED_RIDGE_ARTIFACT_SHA256",
        "calibration_assessment": "R1_EXPECTED_CALIBRATION_ASSESSMENT_SHA256",
        "design_freeze": "R1_EXPECTED_DESIGN_FREEZE_SHA256",
    }
    for label, environment_name in handoff_environment.items():
        _require(environment_name in os.environ, f"missing upstream SHA handoff: {environment_name}")
        expected_handoff = _digest(os.environ[environment_name], environment_name)
        _require(expected_handoff == hashes[label], f"upstream SHA handoff mismatch: {label}")
    milestones.write("input_load_begin")
    documents = {
        name: load_json_bytes(snapshots[name], inputs[name])
        for name in inputs
    }
    milestones.write("input_load_complete", input_count=len(inputs))
    freeze = validate_freeze(inputs["design_freeze"], documents["design_freeze"])
    milestones.write("freeze_verified")
    training = validate_collection(
        inputs["training_table"], inputs["training_manifest"],
        documents["training_table"], documents["training_manifest"],
        role="training", synthetic=args.synthetic, feature_names=freeze["feature_names"],
        table_sha256=hashes["training_table"], manifest_sha256=hashes["training_manifest"],
    )
    _require(0.02 <= training["positive_fraction"] <= 0.40, "training positive fraction outside [0.02,0.40]")
    calibration = validate_collection(
        inputs["calibration_table"], inputs["calibration_manifest"],
        documents["calibration_table"], documents["calibration_manifest"],
        role="calibration", synthetic=args.synthetic, feature_names=freeze["feature_names"],
        table_sha256=hashes["calibration_table"], manifest_sha256=hashes["calibration_manifest"],
    )
    if not args.synthetic:
        repository = Path(__file__).resolve().parents[2]
        checkpoint_path = repository / "1313" / "mainmodel copy 3.jld2"
        weight_path = repository / "artifacts" / "legacy_openvino" / "legacy_1313_weights.npz"
        engine_path = repository / "scripts" / "benchmark_legacy_engine.jl"
        evaluator_path = repository / "scripts" / "evaluate_openvino_checkpoint.jl"
        for label, path in (
            ("canonical old checkpoint", checkpoint_path),
            ("canonical OpenVINO weights", weight_path),
            ("stable-node engine source", engine_path),
            ("OpenVINO evaluator source", evaluator_path),
        ):
            _require(path.is_file(), f"missing {label}: {path}")
        _require(sha256_file(checkpoint_path) == CHECKPOINT_DIGEST, "canonical old checkpoint changed before finalization")
        _require(sha256_file(weight_path) == WEIGHT_DIGEST, "canonical OpenVINO weights changed before finalization")
        stable_source_sha = sha256_file(engine_path)
        evaluator_source_sha = sha256_file(evaluator_path)
        _require(training["stable_node_key_source_sha256"] == stable_source_sha, "training stable-order source is not the live canonical engine")
        _require(calibration["stable_node_key_source_sha256"] == stable_source_sha, "calibration stable-order source is not the live canonical engine")
        _require(training["backend_binding"] == calibration["backend_binding"], "training/calibration backend bindings differ")
        _require(training["backend_binding"]["evaluator_source_sha256"] == evaluator_source_sha, "collector evaluator source is not the live canonical evaluator")
        _require(
            training["engine_dependency_graph"] == calibration["engine_dependency_graph"],
            "training/calibration engine dependency graphs differ",
        )
        live_dependency_graph = _live_engine_dependency_graph(repository)
        _require(
            training["engine_dependency_graph"] == live_dependency_graph,
            "collector engine dependency graph differs from live runtime closure",
        )
    milestones.write("collections_verified", training_rows=training["row_count"], calibration_rows=calibration["row_count"])
    ridge = validate_ridge(
        inputs["ridge_artifact"], documents["ridge_artifact"], synthetic=args.synthetic,
        training=training, training_table_path=inputs["training_table"],
        training_manifest_path=inputs["training_manifest"], freeze_path=inputs["design_freeze"],
        feature_names=freeze["feature_names"],
        artifact_sha256=hashes["ridge_artifact"], freeze_sha256=hashes["design_freeze"],
    )
    observed_assessment = documents["calibration_assessment"]
    measured_overhead = _number(_required(observed_assessment, "median_decision_overhead_ms"), "measured calibration overhead")
    milestones.write("independent_recompute_begin")
    expected = recompute_calibration(
        calibration["rows"],
        ridge,
        freeze["calibration_schedules"],
        measured_overhead,
        synthetic=args.synthetic,
    )
    milestones.write("independent_recompute_complete", promoted=expected["promoted"], override_count=expected["override_count"])
    validate_calibration_assessment(
        observed_assessment, expected, synthetic=args.synthetic,
        training_table_path=inputs["training_table"], training_manifest_path=inputs["training_manifest"],
        calibration_table_path=inputs["calibration_table"], calibration_manifest_path=inputs["calibration_manifest"],
        ridge_path=inputs["ridge_artifact"], freeze_path=inputs["design_freeze"],
        input_sha256=hashes,
        engine_dependency_graph=calibration["engine_dependency_graph"],
    )
    _require(expected["promoted"], "calibration gate did not satisfy every preregistered promotion criterion")
    return {
        "schema_version": "r1-final-assessment-v1",
        "experiment": EXPERIMENT_ID,
        "status": "assessment-pass",
        "promotion": "R1-calibration-promoted",
        "success": True,
        "reasons": [],
        "synthetic": args.synthetic,
        "input_sha256": hashes,
        "training_row_count": training["row_count"],
        "training_positive_fraction": training["positive_fraction"],
        "calibration_row_count": calibration["row_count"],
        "calibration_override_count": expected["override_count"],
        "calibration_override_precision": expected["override_precision"],
        "calibration_override_mean_unclipped_A6": expected["override_mean_unclipped_A6"],
        "calibration_assessment_sha256": hashes["calibration_assessment"],
        "ridge_artifact_sha256": hashes["ridge_artifact"],
        "design_freeze_sha256": hashes["design_freeze"],
        "scope": "calibration promotion only; separate future freeze review required before development",
        "game_strength_evidence": False,
        "model_beat_claim": False,
        "game_evaluation_authorized": False,
        "development_authorized": False,
        "validation_seed_used": False,
        "sealed_test_seed_used": False,
        "sealed_test_authorized": False,
        "game_run": False,
    }


def failure_assessment(reason: str, *, synthetic: bool) -> dict[str, Any]:
    return {
        "schema_version": "r1-final-assessment-v1",
        "experiment": EXPERIMENT_ID,
        "status": "assessment-fail",
        "promotion": "R1-calibration-rejected",
        "success": False,
        "reasons": [reason],
        "synthetic": synthetic,
        "scope": "fail-closed; no game, development, validation, or sealed-test authority",
        "game_strength_evidence": False,
        "model_beat_claim": False,
        "game_evaluation_authorized": False,
        "development_authorized": False,
        "validation_seed_used": False,
        "sealed_test_seed_used": False,
        "sealed_test_authorized": False,
        "game_run": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict terminal assessment for preregistered R1")
    parser.add_argument("training_table", type=Path)
    parser.add_argument("training_manifest", type=Path)
    parser.add_argument("ridge_artifact", type=Path)
    parser.add_argument("calibration_table", type=Path)
    parser.add_argument("calibration_manifest", type=Path)
    parser.add_argument("calibration_assessment", type=Path)
    parser.add_argument("design_freeze", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("milestones", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(argv)
    paths = [
        args.training_table, args.training_manifest, args.ridge_artifact,
        args.calibration_table, args.calibration_manifest, args.calibration_assessment,
        args.design_freeze, args.output, args.milestones,
    ]
    _require(len({_normal(path) for path in paths}) == len(paths), "finalizer paths must be distinct")
    _require(not args.output.exists(), f"refusing to overwrite assessment: {args.output}")
    _require(not args.milestones.exists(), f"refusing to overwrite milestones: {args.milestones}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    milestones = Milestones(args.milestones)
    try:
        milestones.write("args_verified", synthetic=args.synthetic)
        milestones.write("imports_begin")
        try:
            result = assess(args, milestones)
            milestones.write("assessment_evaluated", status=result["status"])
        except Exception as error:
            result = failure_assessment(
                f"{type(error).__name__}: {error}", synthetic=args.synthetic
            )
            milestones.write("assessment_evaluated", status="assessment-fail", reason=result["reasons"][0])
        milestones.write("artifact_write_begin")
        output = atomic_create_json(args.output, result)
        milestones.write("artifact_write_complete", sha256=sha256_file(output))
        milestones.write("phase_complete", status=result["status"])
        print(f"R1_ASSESSMENT_STATUS={result['status']}")
        print(f"R1_ASSESSMENT_RESULT={output}")
        # Scientific rejection/tamper is represented by the immutable fail
        # artifact.  Returning zero lets the wrapper compose its terminal
        # result instead of losing that evidence to phase-level exception flow.
        return 0
    finally:
        milestones.close()


if __name__ == "__main__":
    raise SystemExit(main())
