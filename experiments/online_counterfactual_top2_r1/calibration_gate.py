from __future__ import annotations

# Freeze every common CPU BLAS/OpenMP control before NumPy is imported.  The
# production wrapper launches a fresh process, so these values are effective
# for the complete calibration phase rather than being advisory after import.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


# A monitored production invocation always has six positional paths, with the
# milestone path last, plus an optional --synthetic flag.  Publish process
# identity before the only heavy import (NumPy), so the suspended-Job wrapper
# can prove that startup/import memory belongs to this exact root PID.
_EARLY_MILESTONE_PATH: Path | None = None
if __name__ == "__main__" and len(sys.argv) in (7, 8):
    _candidate = Path(sys.argv[6]).resolve()
    _candidate.parent.mkdir(parents=True, exist_ok=True)
    if _candidate.exists():
        raise RuntimeError(f"refusing to reuse milestone path: {_candidate}")
    with _candidate.open("x", encoding="utf-8", newline="\n") as _stream:
        for _stage in ("script_enter", "imports_begin"):
            _stream.write(
                json.dumps(
                    {
                        "phase": "calibration",
                        "stage": _stage,
                        "pid": os.getpid(),
                        "time_ns": time.time_ns(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            _stream.flush()
            os.fsync(_stream.fileno())
    _EARLY_MILESTONE_PATH = _candidate

import numpy as np

if _EARLY_MILESTONE_PATH is not None:
    with _EARLY_MILESTONE_PATH.open("a", encoding="utf-8", newline="\n") as _stream:
        _stream.write(
            json.dumps(
                {
                    "phase": "calibration",
                    "stage": "imports_end",
                    "pid": os.getpid(),
                    "time_ns": time.time_ns(),
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        _stream.flush()
        os.fsync(_stream.fileno())


CALIBRATION_EPISODES = tuple(range(73101, 73107))
TRAINING_EPISODES = tuple(range(73001, 73013))
SAMPLE_PIECES = tuple(range(10, 241, 10))
FORBIDDEN_DEVELOPMENT = frozenset(range(5742, 5756))
FORBIDDEN_VALIDATION = frozenset(range(8001, 8009))
FORBIDDEN_SEALED = frozenset(range(91001, 91033))

MINIMUM_STATES = 120
MAXIMUM_STATES = 144
MINIMUM_OVERRIDES = 12
MINIMUM_OVERRIDE_EPISODES = 4
MINIMUM_OVERRIDE_RATE = 0.01
MAXIMUM_OVERRIDE_RATE = 0.15
MINIMUM_PRECISION = 0.70
MINIMUM_PRECISION_LOWER_BOUND = 0.50
MINIMUM_MEAN_ADVANTAGE = 0.10
MAXIMUM_MEDIAN_OVERHEAD_MS = 0.10
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 0x5231_73106
ONE_SIDED_LOWER_QUANTILE = 0.10

EXPECTED_FEATURE_COUNT = 70
EXPECTED_COEFFICIENT_COUNT = 71
EXPECTED_ENSEMBLE_SIZE = 256
EXPECTED_FEATURE_NAMES_SHA256 = (
    "7e89c16b57dcebac56e3ab4c5be161d5e5430c682e60f3b565dd23ab3b04ac44"
)
EXPECTED_TRAINING_SCHEDULE_SHA256 = (
    "5b60a1e340b542dc8654a5c80777c254d8336aa086e51be6c2ba1251be20e5f7"
)
EXPECTED_CALIBRATION_SCHEDULE_SHA256 = (
    "c08341a6891997301f112912a4fe969c5493603834cc3dbc114b3e24401617db"
)
EXPECTED_QUANTILE_METHOD = "linear_type7_position_1_plus_n_minus_1_p"
EXPECTED_PYTHON_VERSION = "3.12.13"
EXPECTED_NUMPY_VERSION = "2.4.6"
EXPECTED_HANDOFF_ENVIRONMENT = {
    "calibration_table": "R1_EXPECTED_CALIBRATION_TABLE_SHA256",
    "calibration_manifest": "R1_EXPECTED_CALIBRATION_MANIFEST_SHA256",
    "ridge_artifact": "R1_EXPECTED_RIDGE_ARTIFACT_SHA256",
    "design_freeze": "R1_EXPECTED_DESIGN_FREEZE_SHA256",
}


class CalibrationError(ValueError):
    """Fail-closed input, provenance, or numerical conformance failure."""


@dataclass(frozen=True)
class CalibrationRow:
    episode_id: int
    advantage: float
    a1_terminal: bool
    a2_terminal: bool
    production_decision: bool
    reference_decision: bool
    production_selected_candidate_index: int
    canonical_top1_candidate_index: int
    canonical_top2_candidate_index: int
    production_selected_action_digest: str
    canonical_top1_action_digest: str
    canonical_top2_action_digest: str
    overhead_ms: float
    production_lower_bound: float
    reference_lower_bound: float


@dataclass(frozen=True)
class GateArtifactEvidence:
    finite: bool
    feature_schema_exact: bool
    coefficient_shape_exact: bool
    hyperparameters_exact: bool
    numeric_value_count: int


@dataclass(frozen=True)
class SourceHashEvidence:
    path: str
    actual_sha256: str
    expected_sha256: str
    verified: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationError(message)


def _require_key(value: dict[str, Any], key: str) -> Any:
    if key not in value:
        raise CalibrationError(f"missing required property: {key}")
    return value[key]


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise CalibrationError(f"{label} must be boolean")
    return value


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise CalibrationError(f"{label} must be an exact JSON integer")
    return value


def _require_digest(value: Any, label: str) -> str:
    digest = str(value).lower()
    _require(
        len(digest) == 64 and all(character in "0123456789abcdef" for character in digest),
        f"invalid SHA-256 for {label}",
    )
    return digest


def _normal_path(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash_evidence(path: str | Path, expected_sha256: str) -> SourceHashEvidence:
    source_path = Path(path).resolve()
    _require(source_path.is_file(), f"missing source artifact: {source_path}")
    expected = _require_digest(expected_sha256, str(source_path))
    actual = sha256_file(source_path)
    return SourceHashEvidence(str(source_path), actual, expected, actual == expected)


def load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CalibrationError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def atomic_create_json(path: str | Path, value: Any) -> Path:
    output = Path(path).resolve()
    _require(not output.exists(), f"refusing to overwrite calibration result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    _require(not temporary.exists(), f"stale temporary calibration result: {temporary}")
    try:
        encoded = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
            "utf-8"
        )
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _require(not output.exists(), f"calibration result appeared during write: {output}")
        temporary.rename(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


class Milestones:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        if _EARLY_MILESTONE_PATH is not None:
            _require(
                self.path == _EARLY_MILESTONE_PATH,
                "parsed milestone path differs from early production argv",
            )
            records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
            _require(
                [record.get("stage") for record in records]
                == ["script_enter", "imports_begin", "imports_end"],
                "early milestone sequence mismatch",
            )
            _require(
                all(record.get("pid") == os.getpid() for record in records),
                "early milestone root PID mismatch",
            )
        else:
            _require(not self.path.exists(), f"refusing to reuse milestone path: {self.path}")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.write("script_enter", create=True)
            self.write("imports_begin")
            self.write("imports_end")

    def write(self, stage: str, *, details: dict[str, Any] | None = None, create: bool = False) -> None:
        record = {
            "phase": "calibration",
            "stage": stage,
            "pid": os.getpid(),
            "time_ns": time.time_ns(),
        }
        if details is not None:
            record["details"] = details
        mode = "x" if create else "a"
        with self.path.open(mode, encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def schedule_digest(schedules: Sequence[Sequence[int]]) -> str:
    payload = ";".join(
        ",".join(str(_exact_int(item, "bootstrap schedule item")) for item in schedule)
        for schedule in schedules
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_names_digest(feature_names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest()


def _validate_schedule(
    schedules: Any,
    *,
    expected_count: int,
    expected_width: int,
    allowed_episodes: tuple[int, ...],
    expected_digest: str,
    label: str,
) -> list[list[int]]:
    _require(isinstance(schedules, list), f"{label} schedules must be an array")
    normalized = [
        [_exact_int(item, f"{label} schedule item") for item in schedule]
        for schedule in schedules
    ]
    _require(len(normalized) == expected_count, f"{label} schedule count mismatch")
    _require(
        all(len(schedule) == expected_width for schedule in normalized),
        f"{label} schedule width mismatch",
    )
    allowed = set(allowed_episodes)
    _require(
        all(all(episode in allowed for episode in schedule) for schedule in normalized),
        f"{label} schedule escaped its seed role",
    )
    digest = schedule_digest(normalized)
    _require(digest == expected_digest, f"{label} schedule differs from frozen source anchor")
    return normalized


def validate_freeze(freeze: dict[str, Any]) -> dict[str, Any]:
    _require(_require_key(freeze, "status") == "r1_design_frozen", "design freeze status mismatch")
    _require(
        _require_key(freeze, "experiment") == "online_counterfactual_top2_R1",
        "design freeze experiment mismatch",
    )
    names = [str(item) for item in _require_key(freeze, "feature_names")]
    _require(len(names) == EXPECTED_FEATURE_COUNT, "design freeze feature count mismatch")
    _require(len(set(names)) == EXPECTED_FEATURE_COUNT, "duplicate design freeze feature name")
    _require(feature_names_digest(names) == EXPECTED_FEATURE_NAMES_SHA256, "feature digest mismatch")
    _require(
        _require_digest(_require_key(freeze, "feature_names_sha256"), "freeze feature names")
        == EXPECTED_FEATURE_NAMES_SHA256,
        "design freeze feature digest mismatch",
    )
    _require(_exact_int(_require_key(freeze, "feature_count"), "freeze feature_count") == 70, "freeze feature count mismatch")
    _require(_exact_int(_require_key(freeze, "coefficient_count"), "freeze coefficient_count") == 71, "freeze coefficient count mismatch")
    _require(
        [_exact_int(item, "freeze training seed") for item in _require_key(freeze, "training_seed_ids")]
        == list(TRAINING_EPISODES),
        "freeze training role mismatch",
    )
    _require(
        [_exact_int(item, "freeze calibration seed") for item in _require_key(freeze, "calibration_seed_ids")]
        == list(CALIBRATION_EPISODES),
        "freeze calibration role mismatch",
    )
    _require(
        [_exact_int(item, "freeze sample piece") for item in _require_key(freeze, "sample_piece_indices")]
        == list(SAMPLE_PIECES),
        "freeze sample schedule mismatch",
    )
    _require(float(_require_key(freeze, "ridge_lambda")) == 1.0, "freeze ridge lambda mismatch")
    _require(
        float(_require_key(freeze, "prediction_lower_quantile")) == 0.10,
        "freeze lower quantile mismatch",
    )
    _require(
        float(_require_key(freeze, "override_strict_threshold")) == 0.05,
        "freeze override threshold mismatch",
    )
    _require(
        _require_key(freeze, "training_bootstrap_rng") == "Xoshiro(0x5231_2026)",
        "freeze training RNG mismatch",
    )
    _require(
        _require_key(freeze, "calibration_bootstrap_rng") == "Xoshiro(0x5231_73106)",
        "freeze calibration RNG mismatch",
    )
    for key in (
        "hyperparameter_sweep_authorized",
        "model_or_checkpoint_loaded",
        "game_run",
        "development_seed_loaded",
        "validation_seed_loaded",
        "sealed_test_seed_loaded",
    ):
        _require(_require_bool(_require_key(freeze, key), f"freeze {key}") is False, f"forbidden freeze flag: {key}")
    training = _validate_schedule(
        _require_key(freeze, "training_bootstrap_schedules"),
        expected_count=256,
        expected_width=12,
        allowed_episodes=TRAINING_EPISODES,
        expected_digest=EXPECTED_TRAINING_SCHEDULE_SHA256,
        label="training bootstrap",
    )
    calibration = _validate_schedule(
        _require_key(freeze, "calibration_bootstrap_schedules"),
        expected_count=2_000,
        expected_width=6,
        allowed_episodes=CALIBRATION_EPISODES,
        expected_digest=EXPECTED_CALIBRATION_SCHEDULE_SHA256,
        label="calibration bootstrap",
    )
    _require(
        _require_digest(
            _require_key(freeze, "training_bootstrap_schedule_sha256"), "freeze training schedule"
        )
        == EXPECTED_TRAINING_SCHEDULE_SHA256,
        "freeze training schedule digest mismatch",
    )
    _require(
        _require_digest(
            _require_key(freeze, "calibration_bootstrap_schedule_sha256"),
            "freeze calibration schedule",
        )
        == EXPECTED_CALIBRATION_SCHEDULE_SHA256,
        "freeze calibration schedule digest mismatch",
    )
    return {"feature_names": names, "training_schedules": training, "calibration_schedules": calibration}


def _validate_collection_table(
    table: dict[str, Any],
    manifest: dict[str, Any],
    *,
    role: str,
    expected_episodes: tuple[int, ...],
    minimum_rows: int,
    maximum_rows: int,
    synthetic: bool,
    table_path: Path,
) -> list[dict[str, Any]]:
    schema = "r1-training-table-v1" if role == "training" else "r1-calibration-table-v1"
    _require(_require_key(table, "schema_version") == schema, f"{role} table schema mismatch")
    _require(_require_key(table, "source_role") == role, f"{role} table source_role mismatch")
    _require(_require_bool(_require_key(table, "synthetic"), f"{role} table synthetic") == synthetic, f"{role} table synthetic mode mismatch")
    _require(
        _require_bool(_require_key(table, "validation_seed_used"), f"{role} validation flag") is False,
        f"{role} table used a validation seed",
    )
    _require(
        _require_bool(_require_key(table, "sealed_test_seed_used"), f"{role} sealed flag") is False,
        f"{role} table used a sealed-test seed",
    )
    names = [str(item) for item in _require_key(table, "feature_names")]
    _require(feature_names_digest(names) == EXPECTED_FEATURE_NAMES_SHA256, f"{role} feature schema mismatch")
    _require(
        _require_digest(_require_key(table, "feature_schema_digest"), f"{role} feature schema")
        == EXPECTED_FEATURE_NAMES_SHA256,
        f"{role} feature digest mismatch",
    )
    metadata = _require_key(table, "metadata")
    _require(isinstance(metadata, dict), f"{role} metadata must be an object")
    _require(_require_key(metadata, "source_role") == role, f"{role} metadata role mismatch")
    _require(_require_bool(_require_key(metadata, "synthetic"), f"{role} metadata synthetic") == synthetic, f"{role} metadata synthetic mismatch")
    _require(
        [_exact_int(item, f"{role} metadata role seed") for item in _require_key(metadata, "role_seeds")] == list(expected_episodes),
        f"{role} metadata seed role mismatch",
    )
    _require(
        [
            _exact_int(item, f"{role} metadata training seed")
            for item in _require_key(metadata, "training_seeds")
        ]
        == (list(TRAINING_EPISODES) if role == "training" else []),
        f"{role} metadata training seed role mismatch",
    )
    _require(
        [_exact_int(item, f"{role} metadata sample piece") for item in _require_key(metadata, "sample_pieces")] == list(SAMPLE_PIECES),
        f"{role} metadata sample schedule mismatch",
    )
    _require(
        [_exact_int(item, f"{role} top-level training seed") for item in _require_key(table, "training_seeds")]
        == (list(TRAINING_EPISODES) if role == "training" else []),
        f"{role} top-level training seed role mismatch",
    )
    stable_digest = str(_require_key(metadata, "stable_node_key_source_sha256"))
    if synthetic:
        _require(
            stable_digest == "synthetic-no-engine"
            or (
                len(stable_digest) == 64
                and all(character in "0123456789abcdef" for character in stable_digest)
            ),
            f"{role} synthetic stable-order evidence mismatch",
        )
    else:
        _require_digest(stable_digest, f"{role} stable-node-key source")
    for key in ("validation_seed_used", "sealed_test_seed_used"):
        _require(
            _require_bool(_require_key(metadata, key), f"{role} metadata {key}") is False,
            f"{role} metadata contamination: {key}",
        )
    rows = _require_key(table, "rows")
    _require(isinstance(rows, list), f"{role} rows must be an array")
    _require(minimum_rows <= len(rows) <= maximum_rows, f"{role} row count outside frozen range")
    identities: set[tuple[int, int]] = set()
    seen_episodes: set[int] = set()
    allowed_episodes = set(expected_episodes)
    for row_number, row in enumerate(rows, 1):
        _require(isinstance(row, dict), f"{role} row {row_number} must be an object")
        episode = _exact_int(_require_key(row, "episode_id"), f"{role} row {row_number} episode_id")
        seed = _exact_int(_require_key(row, "seed"), f"{role} row {row_number} seed")
        piece = _exact_int(_require_key(row, "piece_index"), f"{role} row {row_number} piece_index")
        _require(episode == seed, f"{role} row {row_number} episode/seed mismatch")
        _require(episode in allowed_episodes, f"{role} row {row_number} escaped seed role")
        _require(piece in SAMPLE_PIECES, f"{role} row {row_number} escaped sample schedule")
        identity = (episode, piece)
        _require(identity not in identities, f"duplicate {role} state identity {identity}")
        identities.add(identity)
        seen_episodes.add(episode)
    _require(seen_episodes == allowed_episodes, f"{role} table does not cover its exact seed role")

    _require(_require_key(manifest, "schema_version") == "r1-collection-manifest-v1", f"{role} manifest schema mismatch")
    _require(_require_key(manifest, "source_role") == role, f"{role} manifest role mismatch")
    _require(_require_bool(_require_key(manifest, "synthetic"), f"{role} manifest synthetic") == synthetic, f"{role} manifest synthetic mismatch")
    _require(_exact_int(_require_key(manifest, "row_count"), f"{role} manifest row_count") == len(rows), f"{role} manifest row count mismatch")
    _require(_exact_int(_require_key(manifest, "episode_count"), f"{role} manifest episode_count") == len(expected_episodes), f"{role} manifest episode count mismatch")
    _require(
        [_exact_int(item, f"{role} manifest role seed") for item in _require_key(manifest, "role_seeds")] == list(expected_episodes),
        f"{role} manifest seed role mismatch",
    )
    _require(
        _require_digest(_require_key(manifest, "table_sha256"), f"{role} manifest table")
        == sha256_file(table_path),
        f"{role} manifest does not bind the actual table",
    )
    _require(
        _normal_path(_require_key(manifest, "table_path")) == _normal_path(table_path),
        f"{role} manifest table path mismatch",
    )
    _require(
        _require_digest(_require_key(manifest, "feature_schema_digest"), f"{role} manifest schema")
        == EXPECTED_FEATURE_NAMES_SHA256,
        f"{role} manifest feature digest mismatch",
    )
    _require(
        _require_key(manifest, "stable_node_key_source_sha256") == stable_digest,
        f"{role} manifest stable-order evidence mismatch",
    )
    for key in ("validation_seed_used", "sealed_test_seed_used"):
        _require(
            _require_bool(_require_key(manifest, key), f"{role} manifest {key}") is False,
            f"{role} manifest contamination: {key}",
        )
    planned_state_count = len(expected_episodes) * len(SAMPLE_PIECES)
    _require(
        _exact_int(_require_key(manifest, "counterfactual_states_completed"), f"{role} manifest completed states") == planned_state_count,
        f"{role} manifest completed-state count mismatch",
    )
    _require(
        _exact_int(_require_key(metadata, "counterfactual_states_completed"), f"{role} metadata completed states") == planned_state_count,
        f"{role} metadata completed-state count mismatch",
    )
    exclusions = _require_key(manifest, "exclusions")
    _require(isinstance(exclusions, list), f"{role} manifest exclusions missing")
    _require(
        _exact_int(_require_key(manifest, "exclusion_count"), f"{role} manifest exclusion_count") == len(exclusions),
        f"{role} manifest exclusion count mismatch",
    )
    _require(
        len(rows) + len(exclusions) == planned_state_count,
        f"{role} rows/exclusions do not cover the frozen state grid",
    )
    exclusion_identities: set[tuple[int, int]] = set()
    for exclusion_number, exclusion in enumerate(exclusions, 1):
        _require(isinstance(exclusion, dict), f"{role} exclusion {exclusion_number} invalid")
        episode = _exact_int(_require_key(exclusion, "episode_id"), f"{role} exclusion {exclusion_number} episode_id")
        seed = _exact_int(_require_key(exclusion, "seed"), f"{role} exclusion {exclusion_number} seed")
        piece = _exact_int(_require_key(exclusion, "piece_index"), f"{role} exclusion {exclusion_number} piece_index")
        identity = (episode, piece)
        _require(episode == seed and episode in allowed_episodes, f"{role} exclusion escaped seed role")
        _require(piece in SAMPLE_PIECES, f"{role} exclusion escaped sample schedule")
        _require(identity not in identities, f"{role} exclusion duplicates a retained row")
        _require(identity not in exclusion_identities, f"duplicate {role} exclusion {identity}")
        exclusion_identities.add(identity)
    expected_grid = {(episode, piece) for episode in expected_episodes for piece in SAMPLE_PIECES}
    _require(
        identities | exclusion_identities == expected_grid,
        f"{role} rows/exclusions do not exactly cover the frozen state grid",
    )
    episode_evidence = _require_key(manifest, "episodes")
    _require(
        isinstance(episode_evidence, list) and len(episode_evidence) == len(expected_episodes),
        f"{role} manifest episode evidence mismatch",
    )
    _require(
        [_exact_int(_require_key(item, "episode_id"), f"{role} episode evidence episode_id") for item in episode_evidence]
        == list(expected_episodes),
        f"{role} manifest episode order mismatch",
    )
    _require(
        [_exact_int(_require_key(item, "seed"), f"{role} episode evidence seed") for item in episode_evidence]
        == list(expected_episodes),
        f"{role} manifest episode seed order mismatch",
    )
    _require(
        sum(_exact_int(_require_key(item, "rows"), f"{role} episode evidence rows") for item in episode_evidence) == len(rows),
        f"{role} per-episode retained-row total mismatch",
    )
    _require(
        sum(_exact_int(_require_key(item, "exclusions"), f"{role} episode evidence exclusions") for item in episode_evidence)
        == len(exclusions),
        f"{role} per-episode exclusion total mismatch",
    )
    if not synthetic:
        _require(
            _require_bool(_require_key(manifest, "real_model_or_game_loaded"), "real collection flag")
            is True,
            f"{role} production manifest did not load the real model/game",
        )
    return rows


def _load_production_gate(
    artifact: dict[str, Any],
    *,
    artifact_path: Path,
    artifact_expected_sha256: str,
    freeze_path: Path,
    freeze_sha256: str,
    freeze_details: dict[str, Any],
    synthetic: bool,
) -> tuple[dict[str, Any], GateArtifactEvidence, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _require(_require_key(artifact, "schema_version") == "r1-ridge-gate-v1", "ridge schema mismatch")
    _require(_require_key(artifact, "experiment_id") == "online_counterfactual_top2_R1", "ridge experiment mismatch")
    _require(_require_key(artifact, "fit_role") == "training_only", "ridge fit role mismatch")
    _require(
        _require_key(artifact, "fit_backend") == "python_numpy_analytic_ridge",
        "ridge fit backend mismatch",
    )
    _require(
        _require_bool(_require_key(artifact, "source_table_synthetic"), "ridge source synthetic")
        == synthetic,
        "ridge source synthetic mode mismatch",
    )
    _require(_require_bool(_require_key(artifact, "all_finite"), "ridge all_finite") is True, "ridge artifact is not finite")
    for key in ("validation_seed_used", "sealed_test_seed_used"):
        _require(_require_bool(_require_key(artifact, key), f"ridge {key}") is False, f"ridge artifact contamination: {key}")
    _require(
        _require_bool(
            _require_key(artifact, "training_bootstrap_schedule_consumed"),
            "training schedule consumed",
        )
        is True,
        "ridge fit did not consume the frozen training schedule",
    )
    _require(
        _normal_path(_require_key(artifact, "design_freeze_path")) == _normal_path(freeze_path),
        "ridge design freeze path mismatch",
    )
    _require(
        _require_digest(_require_key(artifact, "design_freeze_sha256"), "ridge design freeze")
        == freeze_sha256,
        "ridge design freeze SHA mismatch",
    )
    _require(
        _require_digest(
            _require_key(artifact, "training_bootstrap_schedule_sha256"),
            "ridge training schedule",
        )
        == EXPECTED_TRAINING_SCHEDULE_SHA256,
        "ridge training schedule mismatch",
    )
    _require(
        _require_digest(
            _require_key(artifact, "training_bootstrap_schedule_source_anchor_sha256"),
            "ridge training schedule source anchor",
        )
        == EXPECTED_TRAINING_SCHEDULE_SHA256,
        "ridge training schedule source anchor mismatch",
    )
    runtime_facts = _require_key(artifact, "runtime_facts")
    _require(isinstance(runtime_facts, dict), "ridge runtime facts missing")
    _require(
        _require_key(runtime_facts, "python_version") == EXPECTED_PYTHON_VERSION,
        "ridge Python version mismatch",
    )
    _require(
        _require_key(runtime_facts, "numpy_version") == EXPECTED_NUMPY_VERSION,
        "ridge NumPy version mismatch",
    )
    ridge_threads = _require_key(runtime_facts, "thread_environment")
    _require(
        isinstance(ridge_threads, dict)
        and all(ridge_threads.get(key) == "1" for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")),
        "ridge BLAS thread environment mismatch",
    )

    source_table_path = Path(str(_require_key(artifact, "source_table_path"))).resolve()
    source_manifest_path = Path(str(_require_key(artifact, "source_collection_manifest_path"))).resolve()
    table_evidence = source_hash_evidence(
        source_table_path, _require_key(artifact, "source_table_sha256")
    )
    manifest_evidence = source_hash_evidence(
        source_manifest_path, _require_key(artifact, "source_collection_manifest_sha256")
    )
    _require(table_evidence.verified, "ridge source training table SHA mismatch")
    _require(manifest_evidence.verified, "ridge source training manifest SHA mismatch")
    training_table = load_json(source_table_path)
    training_manifest = load_json(source_manifest_path)
    training_rows = _validate_collection_table(
        training_table,
        training_manifest,
        role="training",
        expected_episodes=TRAINING_EPISODES,
        minimum_rows=240,
        maximum_rows=288,
        synthetic=synthetic,
        table_path=source_table_path,
    )
    training_advantages = [float(_require_key(row, "advantage_unclipped_A6")) for row in training_rows]
    _require(all(math.isfinite(value) for value in training_advantages), "non-finite source training A6")
    positive_fraction = sum(value > 0.0 for value in training_advantages) / len(training_advantages)
    _require(0.02 <= positive_fraction <= 0.40, "source training A6 positive fraction outside frozen gate")
    training_stats = _require_key(artifact, "training_stats")
    _require(isinstance(training_stats, dict), "ridge training statistics missing")
    _require(_exact_int(_require_key(training_stats, "row_count"), "ridge training_stats row_count") == len(training_rows), "ridge training row statistic mismatch")
    _require(_exact_int(_require_key(training_stats, "episode_count"), "ridge training_stats episode_count") == 12, "ridge training episode statistic mismatch")
    _require(
        [_exact_int(item, "ridge training_stats episode_id") for item in _require_key(training_stats, "episode_ids")]
        == list(TRAINING_EPISODES),
        "ridge training episode IDs mismatch",
    )
    _require(
        math.isclose(
            float(_require_key(training_stats, "positive_fraction_unclipped")),
            positive_fraction,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "ridge training positive-fraction statistic mismatch",
    )

    names = [str(item) for item in _require_key(artifact, "feature_names")]
    means = np.asarray(_require_key(artifact, "feature_mean"), dtype=np.float64)
    scales = np.asarray(_require_key(artifact, "feature_scale"), dtype=np.float64)
    constant = np.asarray(_require_key(artifact, "constant_feature"), dtype=np.bool_)
    coefficients = np.asarray(_require_key(artifact, "coefficients"), dtype=np.float64)
    numeric_value_count = int(means.size + scales.size + coefficients.size)
    finite = bool(
        np.all(np.isfinite(means))
        and np.all(np.isfinite(scales))
        and np.all(scales > 0.0)
        and np.all(np.isfinite(coefficients))
    )
    feature_schema_exact = bool(
        len(names) == EXPECTED_FEATURE_COUNT
        and len(set(names)) == EXPECTED_FEATURE_COUNT
        and feature_names_digest(names) == EXPECTED_FEATURE_NAMES_SHA256
        and means.shape == (EXPECTED_FEATURE_COUNT,)
        and scales.shape == (EXPECTED_FEATURE_COUNT,)
        and constant.shape == (EXPECTED_FEATURE_COUNT,)
        and names == freeze_details["feature_names"]
        and _require_digest(_require_key(artifact, "feature_schema_digest"), "ridge feature schema")
        == EXPECTED_FEATURE_NAMES_SHA256
    )
    coefficient_shape_exact = bool(
        [_exact_int(item, "ridge coefficient shape") for item in _require_key(artifact, "coefficient_shape")]
        == [EXPECTED_COEFFICIENT_COUNT, EXPECTED_ENSEMBLE_SIZE]
        and _exact_int(_require_key(artifact, "ensemble_size"), "ridge ensemble_size") == EXPECTED_ENSEMBLE_SIZE
        and coefficients.shape == (EXPECTED_COEFFICIENT_COUNT, EXPECTED_ENSEMBLE_SIZE)
    )
    hyperparameters_exact = bool(
        float(_require_key(artifact, "lambda")) == 1.0
        and _exact_int(_require_key(artifact, "bootstrap_seed"), "ridge bootstrap_seed") == 0x5231_2026
        and _require_key(artifact, "bootstrap_rng") == "Xoshiro(0x5231_2026)"
        and float(_require_key(artifact, "lower_quantile")) == 0.10
        and _require_key(artifact, "quantile_method") == EXPECTED_QUANTILE_METHOD
        and float(_require_key(artifact, "override_threshold")) == 0.05
        and [float(item) for item in _require_key(artifact, "target_clamp")] == [-2.0, 2.0]
    )
    evidence = GateArtifactEvidence(
        finite,
        feature_schema_exact,
        coefficient_shape_exact,
        hyperparameters_exact,
        numeric_value_count,
    )
    _require(all((finite, feature_schema_exact, coefficient_shape_exact, hyperparameters_exact)), "ridge artifact gate evidence failed")
    artifact_source_evidence = source_hash_evidence(artifact_path, artifact_expected_sha256)
    _require(artifact_source_evidence.verified, "ridge artifact differs from fit-phase handoff SHA")
    provenance = {
        "artifact": asdict(artifact_source_evidence),
        "training_table": asdict(table_evidence),
        "training_manifest": asdict(manifest_evidence),
        "design_freeze": asdict(source_hash_evidence(freeze_path, freeze_sha256)),
    }
    return provenance, evidence, means, scales, constant, coefficients


def _type7_quantile(values: Sequence[float], probability: float) -> float:
    _require(len(values) > 0, "quantile of empty sequence")
    ordered = sorted(float(value) for value in values)
    position_zero_based = (len(ordered) - 1) * probability
    lower = math.floor(position_zero_based)
    upper = math.ceil(position_zero_based)
    if lower == upper:
        return ordered[lower]
    fraction = position_zero_based - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def production_lower_bounds(
    features: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    constant: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    _require(features.ndim == 2 and features.shape[1] == 70, "production feature shape mismatch")
    standardized = (features - means[None, :]) / scales[None, :]
    standardized[:, constant] = 0.0
    # Match the deployment-stable Julia and Python ridge evaluators: one GEMV
    # per state.  A calibration batch must not silently switch to a GEMM whose
    # blocking can move a strict 0.05 decision by an ulp.
    predictions = np.empty((features.shape[0], EXPECTED_ENSEMBLE_SIZE), dtype=np.float64)
    weights = coefficients[1:, :]
    for row in range(features.shape[0]):
        predictions[row, :] = standardized[row, :] @ weights + coefficients[0, :]
    _require(bool(np.all(np.isfinite(predictions))), "non-finite production prediction")
    # Sort and interpolate each state separately, as both deployment evaluators
    # do.  This freezes not merely the type-7 definition but the per-state path
    # whose strict 0.05 comparison is under test.
    result = np.empty(features.shape[0], dtype=np.float64)
    position = (EXPECTED_ENSEMBLE_SIZE - 1) * ONE_SIDED_LOWER_QUANTILE
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    for row in range(features.shape[0]):
        ordered = np.sort(predictions[row, :])
        result[row] = (
            ordered[lower]
            if lower == upper
            else ordered[lower] + fraction * (ordered[upper] - ordered[lower])
        )
    _require(bool(np.all(np.isfinite(result))), "non-finite production lower bound")
    return np.asarray(result, dtype=np.float64)


def production_select_action(
    features: Sequence[float],
    *,
    valid_action_count: int,
    canonical_top1_candidate_index: int,
    canonical_top2_candidate_index: int,
    canonical_top1_action_digest: str,
    canonical_top2_action_digest: str,
    means: np.ndarray,
    scales: np.ndarray,
    constant: np.ndarray,
    coefficients: np.ndarray,
) -> dict[str, Any]:
    """Exact production decision and action-mapping path used by R1 deployment."""
    raw = np.asarray(features, dtype=np.float64)
    if (
        raw.shape != (EXPECTED_FEATURE_COUNT,)
        or not bool(np.all(np.isfinite(raw)))
        or valid_action_count < 2
        or raw[6] != valid_action_count
    ):
        return {
            "use_top2": False,
            "lower_bound": math.nan,
            "selected_candidate_index": canonical_top1_candidate_index,
            "selected_action_digest": canonical_top1_action_digest,
            "fallback_reason": "invalid_features_or_candidate_count",
        }
    lower_bound = float(
        production_lower_bounds(
            raw.reshape(1, EXPECTED_FEATURE_COUNT),
            means,
            scales,
            constant,
            coefficients,
        )[0]
    )
    use_top2 = lower_bound > 0.05
    return {
        "use_top2": use_top2,
        "lower_bound": lower_bound,
        "selected_candidate_index": (
            canonical_top2_candidate_index if use_top2 else canonical_top1_candidate_index
        ),
        "selected_action_digest": (
            canonical_top2_action_digest if use_top2 else canonical_top1_action_digest
        ),
        "fallback_reason": "none" if use_top2 else "lower_bound_not_above_threshold",
    }


def scalar_lower_bound_reference(
    features: Sequence[float],
    means: Sequence[float],
    scales: Sequence[float],
    constant: Sequence[bool],
    coefficients: Sequence[Sequence[float]],
) -> float:
    # Deliberately use Python scalar loops and the independent quantile helper;
    # no NumPy dot, broadcasting, sorting, or quantile implementation is shared
    # with the production evaluator.
    _require(len(features) == EXPECTED_FEATURE_COUNT, "scalar feature width mismatch")
    predictions: list[float] = []
    for member in range(EXPECTED_ENSEMBLE_SIZE):
        value = float(coefficients[0][member])
        for feature in range(EXPECTED_FEATURE_COUNT):
            standardized = 0.0 if bool(constant[feature]) else (
                (float(features[feature]) - float(means[feature])) / float(scales[feature])
            )
            value += float(coefficients[feature + 1][member]) * standardized
        predictions.append(value)
    _require(all(math.isfinite(value) for value in predictions), "non-finite scalar prediction")
    return _type7_quantile(predictions, ONE_SIDED_LOWER_QUANTILE)


def calibration_rows_from_table(
    raw_rows: Sequence[dict[str, Any]],
    means: np.ndarray,
    scales: np.ndarray,
    constant: np.ndarray,
    coefficients: np.ndarray,
    production_selector: Any = production_select_action,
) -> tuple[list[CalibrationRow], dict[str, float]]:
    matrix_rows: list[list[float]] = []
    for row_number, row in enumerate(raw_rows, 1):
        features = [float(value) for value in _require_key(row, "features")]
        _require(len(features) == 70, f"calibration row {row_number} feature width mismatch")
        _require(all(math.isfinite(value) for value in features), f"calibration row {row_number} has non-finite features")
        _require(
            float(features[6]) == float(_require_key(row, "valid_action_count")),
            f"calibration row {row_number} candidate count feature mismatch",
        )
        _require(_exact_int(_require_key(row, "valid_action_count"), f"calibration row {row_number} valid_action_count") >= 2, f"calibration row {row_number} has fewer than two candidates")
        matrix_rows.append(features)
    features_matrix = np.asarray(matrix_rows, dtype=np.float64)

    coefficient_lists = coefficients.tolist()
    reference = np.asarray(
        [
            scalar_lower_bound_reference(features, means, scales, constant, coefficient_lists)
            for features in matrix_rows
        ],
        dtype=np.float64,
    )
    # Warm the exact one-state deploy selector before collecting one independent
    # latency observation per calibration state.  The batch path remains a
    # separate numerical diagnostic and is not used for the online budget.
    first = raw_rows[0]
    production_selector(
        matrix_rows[0],
        valid_action_count=_exact_int(_require_key(first, "valid_action_count"), "first calibration valid_action_count"),
        canonical_top1_candidate_index=_exact_int(_require_key(first, "canonical_top1_candidate_index"), "first calibration top1 index"),
        canonical_top2_candidate_index=_exact_int(_require_key(first, "canonical_top2_candidate_index"), "first calibration top2 index"),
        canonical_top1_action_digest=str(_require_key(first, "canonical_top1_action_digest")),
        canonical_top2_action_digest=str(_require_key(first, "canonical_top2_action_digest")),
        means=means,
        scales=scales,
        constant=constant,
        coefficients=coefficients,
    )
    production_values: list[float] = []
    per_state_overhead_ms: list[float] = []
    selections: list[dict[str, Any]] = []
    for features, raw in zip(matrix_rows, raw_rows, strict=True):
        started = time.perf_counter_ns()
        selection = production_selector(
            features,
            valid_action_count=_exact_int(_require_key(raw, "valid_action_count"), "calibration valid_action_count"),
            canonical_top1_candidate_index=_exact_int(_require_key(raw, "canonical_top1_candidate_index"), "calibration top1 index"),
            canonical_top2_candidate_index=_exact_int(_require_key(raw, "canonical_top2_candidate_index"), "calibration top2 index"),
            canonical_top1_action_digest=str(_require_key(raw, "canonical_top1_action_digest")),
            canonical_top2_action_digest=str(_require_key(raw, "canonical_top2_action_digest")),
            means=means,
            scales=scales,
            constant=constant,
            coefficients=coefficients,
        )
        per_state_overhead_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        production_values.append(float(selection["lower_bound"]))
        selections.append(selection)
    production = np.asarray(production_values, dtype=np.float64)
    difference = np.abs(production - reference)
    tolerance = 1.0e-10 * np.maximum(1.0, np.abs(reference))
    _require(bool(np.all(difference <= tolerance)), "production/scalar lower-bound numerical mismatch")

    batch_started = time.perf_counter_ns()
    batch_values = production_lower_bounds(features_matrix, means, scales, constant, coefficients)
    batch_elapsed_ns = time.perf_counter_ns() - batch_started
    _require(
        bool(np.all(np.abs(batch_values - production) <= tolerance)),
        "batch/one-state production lower-bound mismatch",
    )
    rows: list[CalibrationRow] = []
    for index, raw in enumerate(raw_rows):
        selection = selections[index]
        production_decision = bool(selection["use_top2"])
        reference_decision = bool(reference[index] > 0.05)
        _require(
            production_decision == reference_decision,
            f"production/reference decision mismatch at calibration row {index + 1}",
        )
        top1_index = _exact_int(_require_key(raw, "canonical_top1_candidate_index"), "calibration top1 index")
        top2_index = _exact_int(_require_key(raw, "canonical_top2_candidate_index"), "calibration top2 index")
        top1_digest = str(_require_key(raw, "canonical_top1_action_digest"))
        top2_digest = str(_require_key(raw, "canonical_top2_action_digest"))
        _require(top1_index > 0 and top2_index > 0 and top1_index != top2_index, "invalid canonical top-1/top-2 indices")
        valid_action_count = _exact_int(_require_key(raw, "valid_action_count"), "calibration valid_action_count")
        _require(
            top1_index <= valid_action_count and top2_index <= valid_action_count,
            "canonical top-1/top-2 index exceeds valid action count",
        )
        _require(top1_digest and top2_digest and top1_digest != top2_digest, "invalid canonical top-1/top-2 action digests")
        advantage = float(_require_key(raw, "advantage_unclipped_A6"))
        _require(math.isfinite(advantage), "non-finite unclipped A6")
        rows.append(
            CalibrationRow(
                episode_id=_exact_int(_require_key(raw, "episode_id"), "calibration episode_id"),
                advantage=advantage,
                a1_terminal=_require_bool(
                    _require_key(raw, "a1_terminal_within_horizon"), "a1 terminal"
                ),
                a2_terminal=_require_bool(
                    _require_key(raw, "a2_terminal_within_horizon"), "a2 terminal"
                ),
                production_decision=production_decision,
                reference_decision=reference_decision,
                production_selected_candidate_index=_exact_int(selection["selected_candidate_index"], "production selected index"),
                canonical_top1_candidate_index=top1_index,
                canonical_top2_candidate_index=top2_index,
                production_selected_action_digest=str(selection["selected_action_digest"]),
                canonical_top1_action_digest=top1_digest,
                canonical_top2_action_digest=top2_digest,
                overhead_ms=per_state_overhead_ms[index],
                production_lower_bound=float(production[index]),
                reference_lower_bound=float(reference[index]),
            )
        )
    return rows, {
        "batch_elapsed_ns": float(batch_elapsed_ns),
        "batch_amortized_ms_per_state": batch_elapsed_ns / 1_000_000.0 / len(raw_rows),
        "one_state_latency_median_ms": statistics.median(per_state_overhead_ms),
        "one_state_latency_minimum_ms": min(per_state_overhead_ms),
        "one_state_latency_maximum_ms": max(per_state_overhead_ms),
        "production_reference_max_abs_error": float(np.max(difference)),
    }


def cluster_bootstrap_override_metrics(
    rows: Sequence[CalibrationRow], schedules: Sequence[Sequence[int]]
) -> dict[str, Any]:
    normalized = _validate_schedule(
        list(schedules),
        expected_count=BOOTSTRAP_REPLICATES,
        expected_width=len(CALIBRATION_EPISODES),
        allowed_episodes=CALIBRATION_EPISODES,
        expected_digest=EXPECTED_CALIBRATION_SCHEDULE_SHA256,
        label="calibration bootstrap",
    )
    by_episode = {
        episode: [row.advantage for row in rows if row.episode_id == episode and row.production_decision]
        for episode in CALIBRATION_EPISODES
    }
    precision_samples: list[float] = []
    advantage_samples: list[float] = []
    empty_count = 0
    for schedule in normalized:
        sampled = [advantage for episode in schedule for advantage in by_episode[episode]]
        if not sampled:
            empty_count += 1
            precision_samples.append(0.0)
            advantage_samples.append(0.0)
        else:
            precision_samples.append(sum(advantage > 0.0 for advantage in sampled) / len(sampled))
            advantage_samples.append(math.fsum(sampled) / len(sampled))
    return {
        "precision_lower90": _type7_quantile(precision_samples, ONE_SIDED_LOWER_QUANTILE),
        "mean_advantage_lower90": _type7_quantile(
            advantage_samples, ONE_SIDED_LOWER_QUANTILE
        ),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "schedule_sha256": EXPECTED_CALIBRATION_SCHEDULE_SHA256,
        "schedule_matches_regenerated": True,
        "lower_quantile": ONE_SIDED_LOWER_QUANTILE,
        "empty_override_replicate_count": empty_count,
    }


def evaluate_calibration(
    rows: Sequence[CalibrationRow],
    *,
    artifact_evidence: GateArtifactEvidence,
    calibration_source_evidence: SourceHashEvidence,
    bootstrap_schedules: Sequence[Sequence[int]],
    forbidden_seed_used: bool = False,
    provenance: dict[str, Any] | None = None,
    runtime_evidence: dict[str, float] | None = None,
) -> dict[str, Any]:
    _require(bool(rows), "empty R1 calibration rows")
    episodes = sorted({row.episode_id for row in rows})
    override_rows = [row for row in rows if row.production_decision]
    override_count = len(override_rows)
    state_count = len(rows)
    override_rate = override_count / state_count
    override_episodes = sorted({row.episode_id for row in override_rows})
    precision = (
        sum(row.advantage > 0.0 for row in override_rows) / override_count
        if override_count
        else 0.0
    )
    mean_advantage = (
        math.fsum(row.advantage for row in override_rows) / override_count
        if override_count
        else 0.0
    )
    unsafe_terminal_count = sum(
        row.production_decision and row.a2_terminal and not row.a1_terminal for row in rows
    )
    decision_mismatch_count = sum(
        row.production_decision != row.reference_decision for row in rows
    )
    selected_action_mismatch_count = 0
    fallback_mismatch_count = 0
    for row in rows:
        expected_index = (
            row.canonical_top2_candidate_index
            if row.production_decision
            else row.canonical_top1_candidate_index
        )
        expected_digest = (
            row.canonical_top2_action_digest
            if row.production_decision
            else row.canonical_top1_action_digest
        )
        mismatch = (
            row.production_selected_candidate_index != expected_index
            or row.production_selected_action_digest != expected_digest
        )
        selected_action_mismatch_count += mismatch
        fallback_mismatch_count += (not row.production_decision) and mismatch
    median_overhead_ms = statistics.median(row.overhead_ms for row in rows)
    bootstrap = cluster_bootstrap_override_metrics(rows, bootstrap_schedules)
    checks = {
        "minimum_state_count": state_count >= MINIMUM_STATES,
        "exact_calibration_episodes": episodes == list(CALIBRATION_EPISODES),
        "override_rate_in_range": MINIMUM_OVERRIDE_RATE <= override_rate <= MAXIMUM_OVERRIDE_RATE,
        "minimum_override_count": override_count >= MINIMUM_OVERRIDES,
        "override_episode_distribution": len(override_episodes) >= MINIMUM_OVERRIDE_EPISODES,
        "precision_point": precision >= MINIMUM_PRECISION,
        "precision_lower_bound": bootstrap["precision_lower90"]
        > MINIMUM_PRECISION_LOWER_BOUND,
        "mean_advantage_point": mean_advantage >= MINIMUM_MEAN_ADVANTAGE,
        "mean_advantage_lower_bound": bootstrap["mean_advantage_lower90"] > 0.0,
        "no_top2_only_terminal": unsafe_terminal_count == 0,
        "production_reference_exact": decision_mismatch_count == 0,
        "selected_action_matches_decision": selected_action_mismatch_count == 0,
        "fallback_top1_exact": fallback_mismatch_count == 0,
        "overhead_within_budget": median_overhead_ms <= MAXIMUM_MEDIAN_OVERHEAD_MS,
        "artifact_finite": artifact_evidence.finite,
        "feature_schema_exact": artifact_evidence.feature_schema_exact,
        "coefficient_shape_exact": artifact_evidence.coefficient_shape_exact,
        "hyperparameters_exact": artifact_evidence.hyperparameters_exact,
        "calibration_source_hash_verified": calibration_source_evidence.verified,
        "no_forbidden_seed_used": not forbidden_seed_used,
    }
    promoted = all(checks.values())
    result = {
        "experiment": "R1_online_counterfactual_top2_safety_gate",
        "status": "R1-calibration-promoted" if promoted else "R1-calibration-rejected",
        "promoted": promoted,
        "scope": "calibration_only_not_game_strength_not_model_improvement",
        "state_count": state_count,
        "calibration_episodes": episodes,
        "expected_calibration_episodes": list(CALIBRATION_EPISODES),
        "override_count": override_count,
        "override_rate": override_rate,
        "override_episode_count": len(override_episodes),
        "override_episodes": override_episodes,
        "override_precision": precision,
        "override_mean_unclipped_A6": mean_advantage,
        "unsafe_top2_terminal_count": unsafe_terminal_count,
        "production_reference_mismatch_count": decision_mismatch_count,
        "selected_action_mismatch_count": selected_action_mismatch_count,
        "fallback_top1_mismatch_count": fallback_mismatch_count,
        "median_decision_overhead_ms": median_overhead_ms,
        "bootstrap": bootstrap,
        "gate_artifact_evidence": asdict(artifact_evidence),
        "calibration_source_evidence": asdict(calibration_source_evidence),
        "thresholds": {
            "minimum_states": MINIMUM_STATES,
            "override_rate": {"minimum": MINIMUM_OVERRIDE_RATE, "maximum": MAXIMUM_OVERRIDE_RATE},
            "minimum_overrides": MINIMUM_OVERRIDES,
            "minimum_override_episodes": MINIMUM_OVERRIDE_EPISODES,
            "minimum_precision": MINIMUM_PRECISION,
            "minimum_precision_lower90": MINIMUM_PRECISION_LOWER_BOUND,
            "minimum_mean_unclipped_A6": MINIMUM_MEAN_ADVANTAGE,
            "minimum_mean_A6_lower90": 0.0,
            "maximum_median_overhead_ms": MAXIMUM_MEDIAN_OVERHEAD_MS,
        },
        "checks": checks,
        "validation_seed_used": False,
        "sealed_test_seed_used": False,
        "game_strength_evidence": False,
        "model_improvement_evidence": False,
    }
    if provenance is not None:
        result["provenance"] = provenance
    if runtime_evidence is not None:
        result["runtime_evidence"] = runtime_evidence
    return result


def run_calibration(
    calibration_table_path: Path,
    calibration_manifest_path: Path,
    ridge_artifact_path: Path,
    freeze_path: Path,
    *,
    synthetic: bool,
    expected_handoff_sha256: dict[str, str],
    milestones: Milestones | None = None,
    production_selector: Any = production_select_action,
) -> dict[str, Any]:
    _require(sys.version.split()[0] == EXPECTED_PYTHON_VERSION, "calibration Python version mismatch")
    _require(np.__version__ == EXPECTED_NUMPY_VERSION, "calibration NumPy version mismatch")
    if milestones:
        milestones.write("input_load_begin")
    for path in (
        calibration_table_path,
        calibration_manifest_path,
        ridge_artifact_path,
        freeze_path,
    ):
        _require(path.is_file(), f"missing R1 calibration input: {path}")
    input_paths = {
        "calibration_table": calibration_table_path,
        "calibration_manifest": calibration_manifest_path,
        "ridge_artifact": ridge_artifact_path,
        "design_freeze": freeze_path,
    }
    _require(
        set(expected_handoff_sha256) == set(input_paths),
        "incomplete R1 upstream handoff SHA set",
    )
    handoff_evidence = {
        name: source_hash_evidence(path, expected_handoff_sha256[name])
        for name, path in input_paths.items()
    }
    _require(
        all(evidence.verified for evidence in handoff_evidence.values()),
        "R1 input differs from immutable upstream phase handoff",
    )
    calibration_table = load_json(calibration_table_path)
    calibration_manifest = load_json(calibration_manifest_path)
    artifact = load_json(ridge_artifact_path)
    freeze = load_json(freeze_path)
    freeze_sha256 = handoff_evidence["design_freeze"].actual_sha256
    freeze_details = validate_freeze(freeze)
    raw_rows = _validate_collection_table(
        calibration_table,
        calibration_manifest,
        role="calibration",
        expected_episodes=CALIBRATION_EPISODES,
        minimum_rows=MINIMUM_STATES,
        maximum_rows=MAXIMUM_STATES,
        synthetic=synthetic,
        table_path=calibration_table_path,
    )
    provenance, artifact_evidence, means, scales, constant, coefficients = _load_production_gate(
        artifact,
        artifact_path=ridge_artifact_path,
        artifact_expected_sha256=handoff_evidence["ridge_artifact"].expected_sha256,
        freeze_path=freeze_path,
        freeze_sha256=freeze_sha256,
        freeze_details=freeze_details,
        synthetic=synthetic,
    )
    if milestones:
        milestones.write("input_load_complete", details={"rows": len(raw_rows)})
        milestones.write("calibration_begin")
    rows, runtime = calibration_rows_from_table(
        raw_rows,
        means,
        scales,
        constant,
        coefficients,
        production_selector=production_selector,
    )
    forbidden = any(
        _exact_int(row["seed"], "calibration forbidden-seed check")
        in (FORBIDDEN_DEVELOPMENT | FORBIDDEN_VALIDATION | FORBIDDEN_SEALED)
        for row in raw_rows
    )
    calibration_evidence = handoff_evidence["calibration_table"]
    _require(calibration_evidence.verified, "calibration source SHA mismatch")
    _require(
        handoff_evidence["calibration_manifest"].verified,
        "calibration manifest differs from collection-phase handoff SHA",
    )
    provenance.update(
        {
            "calibration_table": asdict(calibration_evidence),
            "calibration_manifest": asdict(handoff_evidence["calibration_manifest"]),
            "blas_thread_environment": {
                key: os.environ[key]
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "BLIS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "numpy_version": np.__version__,
            "python_version": sys.version.split()[0],
            "synthetic": synthetic,
        }
    )
    result = evaluate_calibration(
        rows,
        artifact_evidence=artifact_evidence,
        calibration_source_evidence=calibration_evidence,
        bootstrap_schedules=freeze_details["calibration_schedules"],
        forbidden_seed_used=forbidden,
        provenance=provenance,
        runtime_evidence=runtime,
    )
    if milestones:
        milestones.write("calibration_complete", details={"status": result["status"]})
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen Python/NumPy R1 calibration gate",
    )
    parser.add_argument("calibration_table", type=Path)
    parser.add_argument("calibration_manifest", type=Path)
    parser.add_argument("ridge_artifact", type=Path)
    parser.add_argument("design_freeze", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("milestones", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(argv)
    paths = [
        args.calibration_table,
        args.calibration_manifest,
        args.ridge_artifact,
        args.design_freeze,
        args.output,
        args.milestones,
    ]
    _require(len({_normal_path(path) for path in paths}) == len(paths), "R1 calibration paths must be distinct")
    return args


def expected_handoff_from_environment() -> dict[str, str]:
    values: dict[str, str] = {}
    for name, variable in EXPECTED_HANDOFF_ENVIRONMENT.items():
        value = os.environ.get(variable)
        _require(value is not None, f"missing immutable upstream handoff: {variable}")
        values[name] = _require_digest(value, variable)
    return values


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    milestones = Milestones(args.milestones)
    try:
        milestones.write("args_verified", details={"synthetic": args.synthetic})
        result = run_calibration(
            args.calibration_table.resolve(),
            args.calibration_manifest.resolve(),
            args.ridge_artifact.resolve(),
            args.design_freeze.resolve(),
            synthetic=args.synthetic,
            expected_handoff_sha256=expected_handoff_from_environment(),
            milestones=milestones,
        )
        milestones.write("artifact_write_begin")
        output = atomic_create_json(args.output, result)
        milestones.write("artifact_write_complete", details={"sha256": sha256_file(output)})
        print(f"R1_CALIBRATION_STATUS={result['status']}")
        print(f"R1_CALIBRATION_RESULT={output}")
        milestones.write("phase_complete")
        return 0
    except Exception as error:
        milestones.write(
            "phase_failed",
            details={"error_type": type(error).__name__, "error": str(error)},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
