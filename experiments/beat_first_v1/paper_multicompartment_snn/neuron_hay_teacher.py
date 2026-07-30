#!/usr/bin/env python3
"""Official Hay-ModelDB teacher generator for HD-SWSNN-TwinProp.

This module deliberately keeps the downloaded ModelDB checkout read-only.  It
loads the published Hay et al. L5PC morphology, HOC biophysics and compiled
NMODL mechanisms, then applies the receptor kinetics reported by the
TwinProp preprint with explicit conductance clamps.

The resulting NPZ shards are the high-fidelity teacher boundary:

    detailed Hay cell -> digital twin -> 11-state distillation -> freeze

The authors' TwinProp implementation and exact dataset are not public.  This
is therefore a deterministic reconstruction of the published mechanism and
data contract, not a claim that the unpublished code was reproduced.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

try:
    from neuron import h
except ImportError as exc:  # pragma: no cover - exercised by WSL integration
    raise SystemExit(
        "NEURON is required. Run through generate_neuron_teacher.ps1 or "
        "activate /opt/hd_swsnn_twinprop_neuron under WSL."
    ) from exc


SCHEMA_NAME = "hd_swsnn_twinprop.neuron_teacher.v1"
MODEL_NAME = "HD-SWSNN-TwinProp"
MODELDB_REPOSITORY = "https://github.com/ModelDBRepository/139653.git"

EXCITATORY = np.uint8(1)
INHIBITORY = np.uint8(2)
TRAIN_SPLIT = np.uint8(1)
VALIDATION_SPLIT = np.uint8(2)
TEST_SPLIT = np.uint8(3)

REGION_NAMES = ("soma", "basal", "apical_trunk", "apical_tuft")
REGION_SOMA = 0
REGION_BASAL = 1
REGION_APICAL_TRUNK = 2
REGION_APICAL_TUFT = 3

# Public TwinProp synapse values.
AMPA_RISE_MS = 0.2
AMPA_DECAY_MS = 1.7
AMPA_MAX_NS = 0.4
NMDA_RISE_MS = 0.29
NMDA_DECAY_MS = 43.0
NMDA_MAX_NS = 0.3
GABAA_RISE_MS = 0.2
GABAA_DECAY_MS = 8.0
GABAA_MAX_NS = 0.7
NMDA_GAMMA_PER_MV = 0.062
MAGNESIUM_MM = 1.0
E_EXC_MV = 0.0
E_GABAA_MV = -80.0

DEFAULT_DT_MS = 0.025
DEFAULT_SAMPLE_DT_MS = 1.0
DEFAULT_V_INIT_MV = -80.0
DEFAULT_CELSIUS = 34.0
DEFAULT_SPIKE_THRESHOLD_MV = -20.0
DEFAULT_CA_EVENT_CAI_MM = 2.0e-4
DEFAULT_CA_EVENT_VOLTAGE_MV = -30.0


@dataclasses.dataclass(frozen=True)
class Preset:
    train_trials: int
    validation_trials: int
    test_trials: int
    duration_ms: int
    axons: int
    contacts_per_axon: int
    diagnostic_segments: int
    shard_size: int
    rate_hz: float
    burst_probability: float
    store_dense_events: bool


PRESETS = {
    "tiny": Preset(1, 0, 0, 8, 4, 2, 8, 1, 35.0, 0.0, True),
    "smoke": Preset(4, 1, 1, 100, 16, 20, 32, 2, 25.0, 0.01, True),
    # Dataset counts and duration follow the public TwinProp description.
    # Axon count is intentionally a configurable reconstruction value: the
    # preprint does not publish the exact random-drive teacher protocol.
    "production": Preset(
        49_000,
        1_000,
        2_000,
        10_000,
        128,
        20,
        64,
        4,
        25.0,
        0.005,
        False,
    ),
}


@dataclasses.dataclass(frozen=True)
class TeacherConfig:
    preset: str
    train_trials: int
    validation_trials: int
    test_trials: int
    duration_ms: int
    axons: int
    contacts_per_axon: int
    diagnostic_segments: int
    shard_size: int
    rate_hz: float
    burst_probability: float
    excitatory_fraction: float
    minimum_strength: float
    maximum_strength: float
    seed: int
    dt_ms: float
    sample_dt_ms: float
    v_init_mv: float
    celsius: float
    spike_threshold_mv: float
    ca_event_cai_mm: float
    ca_event_voltage_mv: float
    store_dense_events: bool

    @property
    def total_trials(self) -> int:
        return (
            self.train_trials + self.validation_trials + self.test_trials
        )

    @property
    def time_steps(self) -> int:
        return int(round(self.duration_ms / self.sample_dt_ms))

    @property
    def substeps_per_sample(self) -> int:
        return int(round(self.sample_dt_ms / self.dt_ms))

    @property
    def contacts(self) -> int:
        return self.axons * self.contacts_per_axon

    def validate(self) -> None:
        if self.train_trials < 1:
            raise ValueError("train_trials must be positive")
        if self.validation_trials < 0 or self.test_trials < 0:
            raise ValueError("validation/test trials must be nonnegative")
        if self.duration_ms < 1 or self.axons < 2:
            raise ValueError("duration_ms and axons are too small")
        if self.contacts_per_axon < 1 or self.diagnostic_segments < 2:
            raise ValueError("contacts/diagnostic segment count is too small")
        if self.shard_size < 1:
            raise ValueError("shard_size must be positive")
        if not 0.0 < self.excitatory_fraction < 1.0:
            raise ValueError("excitatory_fraction must be in (0,1)")
        if not 0.0 <= self.burst_probability <= 1.0:
            raise ValueError("burst_probability must be in [0,1]")
        if not 0.0 <= self.minimum_strength <= self.maximum_strength <= 1.0:
            raise ValueError("strength bounds must lie in [0,1]")
        ratio = self.sample_dt_ms / self.dt_ms
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("sample_dt_ms must be an integer multiple of dt_ms")
        duration_ratio = self.duration_ms / self.sample_dt_ms
        if not math.isclose(
            duration_ratio,
            round(duration_ratio),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("duration_ms must be divisible by sample_dt_ms")


@dataclasses.dataclass(frozen=True)
class SegmentInfo:
    index: int  # one-based public index
    section_name: str
    section_region: str
    x: float
    distance_um: float
    length_um: float
    diameter_um: float
    area_um2: float
    region_code: int
    has_calcium: bool
    segment: Any = dataclasses.field(repr=False, compare=False)

    def public_record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "section_name": self.section_name,
            "section_region": self.section_region,
            "x": self.x,
            "distance_um": self.distance_um,
            "length_um": self.length_um,
            "diameter_um": self.diameter_um,
            "area_um2": self.area_um2,
            "region_code": self.region_code,
            "region_name": REGION_NAMES[self.region_code],
            "has_calcium": self.has_calcium,
        }


@dataclasses.dataclass
class HayTeacher:
    root: pathlib.Path
    cell: Any
    soma: Any
    segments: list[SegmentInfo]
    basal_apical_indices: np.ndarray
    diagnostic_indices: np.ndarray
    hashes: dict[str, str]


def _sha256_bytes(chunks: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_named_files(paths: Sequence[pathlib.Path], root: pathlib.Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _git_output(root: pathlib.Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def _modeldb_hashes(root: pathlib.Path) -> dict[str, str]:
    biophysics = root / "models" / "L5PCbiophys3.hoc"
    template = root / "models" / "L5PCtemplate.hoc"
    morphology = root / "morphologies" / "cell1.asc"
    mechanism_library = root / "x86_64" / "libnrnmech.so"
    mod_sources = sorted((root / "mod").glob("*.mod"))
    tracked = _git_output(root, "ls-files", "-z")
    if tracked == "unavailable":
        tree_hash = "unavailable"
    else:
        tracked_paths = [
            root / entry for entry in tracked.split("\0") if entry
        ]
        tree_hash = _sha256_named_files(tracked_paths, root)
    return {
        "modeldb_repository_url": MODELDB_REPOSITORY,
        "modeldb_git_commit": _git_output(root, "rev-parse", "HEAD"),
        "modeldb_git_tree": _git_output(root, "rev-parse", "HEAD^{tree}"),
        "modeldb_tree_sha256": tree_hash,
        "modeldb_tracked_status": _git_output(
            root, "status", "--porcelain", "--untracked-files=no"
        ),
        "morphology_sha256": _sha256_file(morphology),
        "biophysics_sha256": _sha256_file(biophysics),
        "template_sha256": _sha256_file(template),
        "mechanism_library_sha256": _sha256_file(mechanism_library),
        "mechanism_sources_sha256": _sha256_named_files(mod_sources, root),
    }


def _validate_modeldb_root(root: pathlib.Path) -> None:
    required = (
        root / "models" / "L5PCbiophys3.hoc",
        root / "models" / "L5PCtemplate.hoc",
        root / "morphologies" / "cell1.asc",
        root / "x86_64" / "libnrnmech.so",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Hay ModelDB checkout is incomplete; missing: " + ", ".join(missing)
        )


def _mechanisms_are_loaded() -> bool:
    probe = h.Section(name="hd_swsnn_mechanism_probe")
    try:
        probe.insert("NaTa_t")
        return True
    except Exception:
        return False
    finally:
        h.delete_section(sec=probe)


def _load_modeldb(root: pathlib.Path) -> Any:
    _validate_modeldb_root(root)
    h.load_file("stdrun.hoc")
    if not _mechanisms_are_loaded():
        h.nrn_load_dll(str(root / "x86_64" / "libnrnmech.so"))
    h.load_file("import3d.hoc")
    h.load_file(str(root / "models" / "L5PCbiophys3.hoc"))
    h.load_file(str(root / "models" / "L5PCtemplate.hoc"))
    return h.L5PCtemplate(str(root / "morphologies" / "cell1.asc"))


def _section_names(section_list: Any) -> set[str]:
    return {section.name() for section in section_list}


def _segment_catalog(cell: Any) -> tuple[Any, list[SegmentInfo]]:
    soma = list(cell.somatic)[0]
    h.distance(0.0, 0.5, sec=soma)
    soma_names = _section_names(cell.somatic)
    basal_names = _section_names(cell.basal)
    apical_names = _section_names(cell.apical)
    axonal_names = _section_names(cell.axonal)
    records: list[SegmentInfo] = []
    for section in cell.all:
        name = section.name()
        if name in soma_names:
            section_region = "soma"
        elif name in basal_names:
            section_region = "basal"
        elif name in apical_names:
            section_region = "apical"
        elif name in axonal_names:
            section_region = "axon"
        else:
            section_region = "unknown"
        segment_length = float(section.L) / int(section.nseg)
        for segment in section:
            distance = float(h.distance(segment.x, sec=section))
            if section_region == "soma":
                region_code = REGION_SOMA
            elif section_region == "basal":
                region_code = REGION_BASAL
            elif section_region == "apical" and distance >= 650.0:
                region_code = REGION_APICAL_TUFT
            elif section_region == "apical":
                region_code = REGION_APICAL_TRUNK
            else:
                # Axonal values are diagnostic-only and share the soma group.
                region_code = REGION_SOMA
            records.append(
                SegmentInfo(
                    index=len(records) + 1,
                    section_name=name,
                    section_region=section_region,
                    x=float(segment.x),
                    distance_um=distance,
                    length_um=segment_length,
                    diameter_um=float(segment.diam),
                    area_um2=float(h.area(segment.x, sec=section)),
                    region_code=region_code,
                    has_calcium=hasattr(segment, "cai"),
                    segment=segment,
                )
            )
    return soma, records


def _evenly_spaced_indices(
    records: Sequence[SegmentInfo], count: int
) -> list[int]:
    if count <= 0 or not records:
        return []
    ordered = sorted(records, key=lambda record: record.distance_um)
    if count >= len(ordered):
        return [record.index for record in ordered]
    positions = np.linspace(0, len(ordered) - 1, count)
    return [ordered[int(round(position))].index for position in positions]


def _diagnostic_segments(
    segments: Sequence[SegmentInfo], requested: int
) -> np.ndarray:
    basal = [record for record in segments if record.section_region == "basal"]
    apical = [
        record for record in segments if record.section_region == "apical"
    ]
    basal_count = max(1, requested // 4)
    apical_count = max(1, requested - basal_count)
    selected = _evenly_spaced_indices(basal, basal_count)
    selected += _evenly_spaced_indices(apical, apical_count)
    # Quantile rounding can select duplicates on very small morphologies.
    selected = list(dict.fromkeys(selected))
    remaining = requested - len(selected)
    if remaining > 0:
        candidates = basal + apical
        for record in sorted(candidates, key=lambda item: item.distance_um):
            if record.index not in selected:
                selected.append(record.index)
                remaining -= 1
                if remaining == 0:
                    break
    return np.asarray(selected, dtype=np.int32)


def instantiate_teacher(
    modeldb_root: os.PathLike[str] | str,
    diagnostic_segments: int = 32,
) -> HayTeacher:
    root = pathlib.Path(modeldb_root).expanduser().resolve()
    cell = _load_modeldb(root)
    soma, segments = _segment_catalog(cell)
    eligible = np.asarray(
        [
            record.index
            for record in segments
            if record.section_region in ("basal", "apical")
        ],
        dtype=np.int32,
    )
    diagnostics = _diagnostic_segments(segments, diagnostic_segments)
    hashes = _modeldb_hashes(root)
    hashes["generator_source_sha256"] = _sha256_file(
        pathlib.Path(__file__).resolve()
    )
    return HayTeacher(
        root=root,
        cell=cell,
        soma=soma,
        segments=segments,
        basal_apical_indices=eligible,
        diagnostic_indices=diagnostics,
        hashes=hashes,
    )


def _double_exponential_scale(rise_ms: float, decay_ms: float) -> float:
    peak_time = (
        rise_ms
        * decay_ms
        / (decay_ms - rise_ms)
        * math.log(decay_ms / rise_ms)
    )
    peak = math.exp(-peak_time / decay_ms) - math.exp(
        -peak_time / rise_ms
    )
    return 1.0 / peak


def _nmda_magnesium_block(voltage_mv: np.ndarray) -> np.ndarray:
    return 1.0 / (
        1.0
        + (MAGNESIUM_MM / 3.57)
        * np.exp(np.clip(-NMDA_GAMMA_PER_MV * voltage_mv, -60.0, 60.0))
    )


def _trial_seed(seed: int, trial_index: int) -> int:
    mixed = (
        (seed & ((1 << 64) - 1))
        ^ ((trial_index * 0x9E3779B97F4A7C15) & ((1 << 64) - 1))
    )
    return mixed


def _draw_protocol(
    teacher: HayTeacher,
    config: TeacherConfig,
    global_trial: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(_trial_seed(config.seed, global_trial))
    axon_kind = np.full(config.axons, INHIBITORY, dtype=np.uint8)
    excitatory_axons = max(
        1, min(config.axons - 1, round(config.axons * config.excitatory_fraction))
    )
    permutation = rng.permutation(config.axons)
    axon_kind[permutation[:excitatory_axons]] = EXCITATORY
    contact_axon = np.repeat(
        np.arange(1, config.axons + 1, dtype=np.int32),
        config.contacts_per_axon,
    )
    contact_kind = axon_kind[contact_axon - 1]
    eligible_records = [
        teacher.segments[index - 1]
        for index in teacher.basal_apical_indices
    ]
    # One contact slot per micrometre, independently for E and I, implements
    # the public one-E/one-I-contact-per-micrometre constraint at segment
    # resolution.
    spatial_slots = np.concatenate(
        [
            np.full(
                max(1, int(math.floor(record.length_um))),
                record.index,
                dtype=np.int32,
            )
            for record in eligible_records
        ]
    )
    contact_segment = np.empty(config.contacts, dtype=np.int32)
    for kind in (EXCITATORY, INHIBITORY):
        contact_indices = np.flatnonzero(contact_kind == kind)
        if len(contact_indices) > len(spatial_slots):
            raise ValueError(
                f"{len(contact_indices)} contacts of kind {int(kind)} exceed "
                f"the morphology density capacity {len(spatial_slots)}"
            )
        selected = rng.choice(
            spatial_slots,
            size=len(contact_indices),
            replace=False,
        )
        contact_segment[contact_indices] = selected
    contact_strength = rng.uniform(
        config.minimum_strength,
        config.maximum_strength,
        size=config.contacts,
    ).astype(np.float32)
    event_probability = config.rate_hz * config.sample_dt_ms / 1000.0
    axon_events = (
        rng.random((config.axons, config.time_steps)) < event_probability
    )
    if config.burst_probability > 0.0:
        for time_bin in range(config.time_steps):
            if rng.random() < config.burst_probability:
                active_exc = np.flatnonzero(axon_kind == EXCITATORY)
                count = max(1, len(active_exc) // 4)
                recruited = rng.choice(active_exc, size=count, replace=False)
                axon_events[recruited, time_bin] = True
    event_axons, event_bins = np.nonzero(axon_events)
    return {
        "axon_kind": axon_kind,
        "contact_axon": contact_axon,
        "contact_kind": contact_kind,
        "contact_segment": contact_segment,
        "contact_strength": contact_strength,
        "axon_events": axon_events,
        "event_axon": (event_axons + 1).astype(np.int32),
        "event_time_bin": event_bins.astype(np.int32),
    }


def _zero_clamps(clamps: Sequence[Any]) -> None:
    for clamp in clamps:
        clamp.amp = 0.0


def _simulate_trial(
    teacher: HayTeacher,
    config: TeacherConfig,
    protocol: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    contact_segment = protocol["contact_segment"]
    contact_kind = protocol["contact_kind"]
    contact_strength = protocol["contact_strength"]
    contact_axon = protocol["contact_axon"]
    axon_events = protocol["axon_events"]
    target_indices = np.unique(contact_segment)
    target_slots = {
        int(segment_index): slot
        for slot, segment_index in enumerate(target_indices)
    }
    contact_target_slot = np.asarray(
        [target_slots[int(index)] for index in contact_segment],
        dtype=np.int32,
    )
    target_records = [
        teacher.segments[int(index) - 1] for index in target_indices
    ]
    clamps = [h.IClamp(record.segment) for record in target_records]
    for clamp in clamps:
        clamp.delay = 0.0
        clamp.dur = 1.0e9
        clamp.amp = 0.0
    target_count = len(target_records)
    receptor_state = {
        "ampa_rise": np.zeros(target_count, dtype=np.float64),
        "ampa_decay": np.zeros(target_count, dtype=np.float64),
        "nmda_rise": np.zeros(target_count, dtype=np.float64),
        "nmda_decay": np.zeros(target_count, dtype=np.float64),
        "gaba_rise": np.zeros(target_count, dtype=np.float64),
        "gaba_decay": np.zeros(target_count, dtype=np.float64),
    }
    decay = {
        "ampa_rise": math.exp(-config.dt_ms / AMPA_RISE_MS),
        "ampa_decay": math.exp(-config.dt_ms / AMPA_DECAY_MS),
        "nmda_rise": math.exp(-config.dt_ms / NMDA_RISE_MS),
        "nmda_decay": math.exp(-config.dt_ms / NMDA_DECAY_MS),
        "gaba_rise": math.exp(-config.dt_ms / GABAA_RISE_MS),
        "gaba_decay": math.exp(-config.dt_ms / GABAA_DECAY_MS),
    }
    scale = {
        "ampa": _double_exponential_scale(AMPA_RISE_MS, AMPA_DECAY_MS),
        "nmda": _double_exponential_scale(NMDA_RISE_MS, NMDA_DECAY_MS),
        "gaba": _double_exponential_scale(GABAA_RISE_MS, GABAA_DECAY_MS),
    }
    diagnostic_records = [
        teacher.segments[int(index) - 1
        ] for index in teacher.diagnostic_indices
    ]
    diagnostic_target_slots = np.asarray(
        [
            target_slots.get(int(record.index), -1)
            for record in diagnostic_records
        ],
        dtype=np.int32,
    )
    time_steps = config.time_steps
    diagnostic_count = len(diagnostic_records)
    soma_voltage = np.empty(time_steps, dtype=np.float32)
    soma_spike = np.zeros(time_steps, dtype=np.float32)
    nmda_region = np.zeros(
        (len(REGION_NAMES), time_steps), dtype=np.float32
    )
    dendritic_voltage = np.empty(
        (diagnostic_count, time_steps), dtype=np.float32
    )
    dendritic_nmda = np.zeros(
        (diagnostic_count, time_steps), dtype=np.float32
    )
    dendritic_cai = np.zeros(
        (diagnostic_count, time_steps), dtype=np.float32
    )
    dendritic_ica = np.zeros(
        (diagnostic_count, time_steps), dtype=np.float32
    )
    ca_event = np.zeros(
        (diagnostic_count, time_steps), dtype=np.uint8
    )
    soma_segment = teacher.soma(0.5)
    h.dt = config.dt_ms
    h.steps_per_ms = 1.0 / config.dt_ms
    h.celsius = config.celsius
    h.cvode_active(0)
    h.finitialize(config.v_init_mv)
    previous_soma_voltage = float(soma_segment.v)
    excitatory_contact = contact_kind == EXCITATORY
    inhibitory_contact = contact_kind == INHIBITORY
    nmda_current = np.zeros(target_count, dtype=np.float64)
    try:
        for time_bin in range(time_steps):
            axon_event_count = np.asarray(
                axon_events[:, time_bin], dtype=np.float64
            )
            active_axons = np.flatnonzero(axon_event_count > 0.0) + 1
            if len(active_axons):
                active_contacts = np.isin(contact_axon, active_axons)
                contact_event_count = axon_event_count[contact_axon - 1]
                active_exc = active_contacts & excitatory_contact
                active_inh = active_contacts & inhibitory_contact
                if np.any(active_exc):
                    increments = np.zeros(target_count, dtype=np.float64)
                    np.add.at(
                        increments,
                        contact_target_slot[active_exc],
                        contact_strength[active_exc]
                        * contact_event_count[active_exc],
                    )
                    receptor_state["ampa_rise"] += increments
                    receptor_state["ampa_decay"] += increments
                    receptor_state["nmda_rise"] += increments
                    receptor_state["nmda_decay"] += increments
                if np.any(active_inh):
                    increments = np.zeros(target_count, dtype=np.float64)
                    np.add.at(
                        increments,
                        contact_target_slot[active_inh],
                        contact_strength[active_inh]
                        * contact_event_count[active_inh],
                    )
                    receptor_state["gaba_rise"] += increments
                    receptor_state["gaba_decay"] += increments
            spike_in_bin = False
            for _ in range(config.substeps_per_sample):
                for key, factor in decay.items():
                    receptor_state[key] *= factor
                ampa_ns = (
                    AMPA_MAX_NS
                    * scale["ampa"]
                    * (
                        receptor_state["ampa_decay"]
                        - receptor_state["ampa_rise"]
                    )
                )
                nmda_unblocked_ns = (
                    NMDA_MAX_NS
                    * scale["nmda"]
                    * (
                        receptor_state["nmda_decay"]
                        - receptor_state["nmda_rise"]
                    )
                )
                gaba_ns = (
                    GABAA_MAX_NS
                    * scale["gaba"]
                    * (
                        receptor_state["gaba_decay"]
                        - receptor_state["gaba_rise"]
                    )
                )
                local_voltage = np.fromiter(
                    (float(record.segment.v) for record in target_records),
                    dtype=np.float64,
                    count=target_count,
                )
                mg_block = _nmda_magnesium_block(local_voltage)
                # Outward-current convention, matching PaperHayCell.jl.
                nmda_current[:] = (
                    nmda_unblocked_ns
                    * 1.0e-3
                    * mg_block
                    * (local_voltage - E_EXC_MV)
                )
                inward_current = (
                    ampa_ns * 1.0e-3 * (E_EXC_MV - local_voltage)
                    - nmda_current
                    + gaba_ns * 1.0e-3 * (E_GABAA_MV - local_voltage)
                )
                for slot, clamp in enumerate(clamps):
                    clamp.amp = float(inward_current[slot])
                h.fadvance()
                current_soma_voltage = float(soma_segment.v)
                if (
                    previous_soma_voltage < config.spike_threshold_mv
                    and current_soma_voltage >= config.spike_threshold_mv
                ):
                    spike_in_bin = True
                previous_soma_voltage = current_soma_voltage
            # Align the recorded NMDA target with the post-step voltage at
            # the exact 1-ms sample boundary rather than retaining the
            # voltage from the beginning of the final 0.025-ms substep.
            local_voltage = np.fromiter(
                (float(record.segment.v) for record in target_records),
                dtype=np.float64,
                count=target_count,
            )
            nmda_current[:] = (
                nmda_unblocked_ns
                * 1.0e-3
                * _nmda_magnesium_block(local_voltage)
                * (local_voltage - E_EXC_MV)
            )
            soma_voltage[time_bin] = float(soma_segment.v)
            soma_spike[time_bin] = 1.0 if spike_in_bin else 0.0
            for slot, record in enumerate(target_records):
                nmda_region[record.region_code, time_bin] += float(
                    nmda_current[slot]
                )
            for diagnostic, record in enumerate(diagnostic_records):
                voltage = float(record.segment.v)
                dendritic_voltage[diagnostic, time_bin] = voltage
                target_slot = diagnostic_target_slots[diagnostic]
                if target_slot >= 0:
                    dendritic_nmda[diagnostic, time_bin] = float(
                        nmda_current[target_slot]
                    )
                if hasattr(record.segment, "cai"):
                    cai = float(record.segment.cai)
                    ica = (
                        float(record.segment.ica)
                        if hasattr(record.segment, "ica")
                        else 0.0
                    )
                    dendritic_cai[diagnostic, time_bin] = cai
                    dendritic_ica[diagnostic, time_bin] = ica
                    ca_event[diagnostic, time_bin] = np.uint8(
                        cai >= config.ca_event_cai_mm
                        and voltage >= config.ca_event_voltage_mv
                    )
    finally:
        _zero_clamps(clamps)
    return {
        "target_voltage": soma_voltage,
        "target_spike": soma_spike,
        "target_nmda": nmda_region,
        "target_compartment_voltage": dendritic_voltage,
        "target_compartment_nmda": dendritic_nmda,
        "target_dendritic_cai": dendritic_cai,
        "target_dendritic_ica": dendritic_ica,
        "target_ca_event": ca_event,
    }


def _split_code(config: TeacherConfig, trial: int) -> np.uint8:
    if trial <= config.train_trials:
        return TRAIN_SPLIT
    if trial <= config.train_trials + config.validation_trials:
        return VALIDATION_SPLIT
    return TEST_SPLIT


def _json_canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _teacher_contract(
    teacher: HayTeacher, config: TeacherConfig
) -> dict[str, Any]:
    contract = {
        "schema_name": SCHEMA_NAME,
        "model_name": MODEL_NAME,
        "config": dataclasses.asdict(config),
        "source_hashes": teacher.hashes,
        "mechanisms": {
            "detailed_cell": "Hay et al. 2011 L5PCbiophys3/cell1.asc",
            "integration": "NEURON fixed-step",
            "dt_ms": config.dt_ms,
            "sample_dt_ms": config.sample_dt_ms,
            "receptors": {
                "AMPA": {
                    "rise_ms": AMPA_RISE_MS,
                    "decay_ms": AMPA_DECAY_MS,
                    "max_ns": AMPA_MAX_NS,
                },
                "NMDA": {
                    "rise_ms": NMDA_RISE_MS,
                    "decay_ms": NMDA_DECAY_MS,
                    "max_ns": NMDA_MAX_NS,
                    "gamma_per_mv": NMDA_GAMMA_PER_MV,
                    "magnesium_mm": MAGNESIUM_MM,
                },
                "GABAA": {
                    "rise_ms": GABAA_RISE_MS,
                    "decay_ms": GABAA_DECAY_MS,
                    "max_ns": GABAA_MAX_NS,
                },
            },
            "reversal_mv": {"exc": E_EXC_MV, "gabaa": E_GABAA_MV},
            "dale_law": True,
            "nonnegative_conductance": True,
            "contact_density": "at most one E and one I contact per um",
            "contacts_per_axon": config.contacts_per_axon,
        },
    }
    contract["teacher_contract_sha256"] = hashlib.sha256(
        _json_canonical(contract).encode("utf-8")
    ).hexdigest()
    return contract


ARRAY_CONTRACT = {
    "sample_indices": {"axes": ["trial"], "dtype": "int32"},
    "split_code": {
        "axes": ["trial"],
        "dtype": "uint8",
        "codes": {"train": 1, "validation": 2, "test": 3},
    },
    "contact_axon": {
        "axes": ["contact", "trial"],
        "dtype": "int32",
        "index_base": 1,
    },
    "contact_segment": {
        "axes": ["contact", "trial"],
        "dtype": "int32",
        "index_base": 1,
    },
    "contact_kind": {
        "axes": ["contact", "trial"],
        "dtype": "uint8",
        "codes": {"excitatory": 1, "inhibitory": 2},
    },
    "contact_strength": {
        "axes": ["contact", "trial"],
        "dtype": "float32",
        "units": "fraction_of_receptor_max",
    },
    "axon_kind": {
        "axes": ["axon", "trial"],
        "dtype": "uint8",
        "codes": {"excitatory": 1, "inhibitory": 2},
    },
    "event_trial_offset": {
        "axes": ["trial_plus_one"],
        "dtype": "int64",
    },
    "event_axon": {
        "axes": ["event"],
        "dtype": "int32",
        "index_base": 1,
    },
    "event_time_bin": {
        "axes": ["event"],
        "dtype": "int32",
        "index_base": 0,
        "units": "sample_dt_ms",
    },
    "target_voltage": {
        "axes": ["time", "trial"],
        "dtype": "float32",
        "units": "mV",
        "location": "soma_0.5",
    },
    "target_spike": {
        "axes": ["time", "trial"],
        "dtype": "float32",
        "units": "upward_threshold_crossing",
    },
    "target_nmda": {
        "axes": ["region", "time", "trial"],
        "dtype": "float32",
        "units": "nA",
        "sign": "outward_positive",
        "regions": list(REGION_NAMES),
    },
    "target_compartment_voltage": {
        "axes": ["diagnostic_segment", "time", "trial"],
        "dtype": "float32",
        "units": "mV",
    },
    "target_compartment_nmda": {
        "axes": ["diagnostic_segment", "time", "trial"],
        "dtype": "float32",
        "units": "nA",
        "sign": "outward_positive",
    },
    "target_dendritic_cai": {
        "axes": ["diagnostic_segment", "time", "trial"],
        "dtype": "float32",
        "units": "mM",
    },
    "target_dendritic_ica": {
        "axes": ["diagnostic_segment", "time", "trial"],
        "dtype": "float32",
        "units": "mA/cm2",
    },
    "target_ca_event": {
        "axes": ["diagnostic_segment", "time", "trial"],
        "dtype": "uint8",
    },
}


def _allocate_shard(
    config: TeacherConfig, batch: int, diagnostic_count: int
) -> dict[str, np.ndarray]:
    return {
        "sample_indices": np.empty(batch, dtype=np.int32),
        "split_code": np.empty(batch, dtype=np.uint8),
        "contact_axon": np.empty(
            (config.contacts, batch), dtype=np.int32
        ),
        "contact_segment": np.empty(
            (config.contacts, batch), dtype=np.int32
        ),
        "contact_kind": np.empty(
            (config.contacts, batch), dtype=np.uint8
        ),
        "contact_strength": np.empty(
            (config.contacts, batch), dtype=np.float32
        ),
        "axon_kind": np.empty((config.axons, batch), dtype=np.uint8),
        "target_voltage": np.empty(
            (config.time_steps, batch), dtype=np.float32
        ),
        "target_spike": np.empty(
            (config.time_steps, batch), dtype=np.float32
        ),
        "target_nmda": np.empty(
            (len(REGION_NAMES), config.time_steps, batch),
            dtype=np.float32,
        ),
        "target_compartment_voltage": np.empty(
            (diagnostic_count, config.time_steps, batch),
            dtype=np.float32,
        ),
        "target_compartment_nmda": np.empty(
            (diagnostic_count, config.time_steps, batch),
            dtype=np.float32,
        ),
        "target_dendritic_cai": np.empty(
            (diagnostic_count, config.time_steps, batch),
            dtype=np.float32,
        ),
        "target_dendritic_ica": np.empty(
            (diagnostic_count, config.time_steps, batch),
            dtype=np.float32,
        ),
        "target_ca_event": np.empty(
            (diagnostic_count, config.time_steps, batch),
            dtype=np.uint8,
        ),
    }


def _atomic_save_npz(path: pathlib.Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".npz", prefix=path.stem + ".", dir=path.parent,
        delete=False
    ) as stream:
        temporary = pathlib.Path(stream.name)
        np.savez_compressed(stream, **arrays)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=path.stem + ".",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = pathlib.Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def generate_dataset(
    modeldb_root: os.PathLike[str] | str,
    output_directory: os.PathLike[str] | str,
    config: TeacherConfig,
) -> dict[str, Any]:
    config.validate()
    output = pathlib.Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    teacher = instantiate_teacher(
        modeldb_root, diagnostic_segments=config.diagnostic_segments
    )
    contract = _teacher_contract(teacher, config)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "schema_name": SCHEMA_NAME,
        "model_name": MODEL_NAME,
        "stage": "official_hay_neuron_teacher",
        "completion_state": "generating",
        "teacher_contract_sha256": contract["teacher_contract_sha256"],
        "teacher_contract": contract,
        "source_hashes": teacher.hashes,
        "modeldb_source_modified_by_generator": False,
        "neuron_version": str(h.nrnversion()),
        "config": dataclasses.asdict(config),
        "total_segments": len(teacher.segments),
        "contact_segment_index_base": 1,
        "diagnostic_segment_indices": teacher.diagnostic_indices.tolist(),
        "segments": [
            record.public_record() for record in teacher.segments
        ],
        "regions": list(REGION_NAMES),
        "array_contract": ARRAY_CONTRACT,
        "paper_values_vs_reconstruction": {
            "paper_explicit": {
                "detailed_cell": "Hay et al. 2011 L5PC",
                "ampa": [AMPA_RISE_MS, AMPA_DECAY_MS, AMPA_MAX_NS],
                "nmda": [
                    NMDA_RISE_MS,
                    NMDA_DECAY_MS,
                    NMDA_MAX_NS,
                    NMDA_GAMMA_PER_MV,
                ],
                "gabaa": [GABAA_RISE_MS, GABAA_DECAY_MS, GABAA_MAX_NS],
                "average_contacts_per_axon": 20,
                "teacher_train_simulations": 50_000,
                "teacher_test_simulations": 2_000,
                "simulation_duration_ms": 10_000,
            },
            "public_hay_choice": {
                "modeldb_accession": 139653,
                "morphology": "cell1.asc",
                "biophysics": "L5PCbiophys3.hoc",
            },
            "reconstruction_choices": {
                "teacher_random_drive_protocol_public": False,
                "fixed_step_dt_ms": config.dt_ms,
                "celsius": config.celsius,
                "apical_tuft_boundary_um": 650.0,
                "receptors": "explicit double exponential IClamp; NMDA has local-voltage Mg block",
                "calcium_event_definition": {
                    "cai_at_least_mm": config.ca_event_cai_mm,
                    "voltage_at_least_mv": config.ca_event_voltage_mv,
                },
            },
        },
        "shards": [],
    }
    manifest_path = output / "manifest.json"
    _atomic_write_json(manifest_path, manifest)
    for shard_index, first in enumerate(
        range(1, config.total_trials + 1, config.shard_size), start=1
    ):
        last = min(first + config.shard_size - 1, config.total_trials)
        batch = last - first + 1
        arrays = _allocate_shard(
            config, batch, len(teacher.diagnostic_indices)
        )
        event_axons: list[np.ndarray] = []
        event_bins: list[np.ndarray] = []
        event_offsets = [0]
        dense_axon_events = (
            np.empty(
                (config.axons, config.time_steps, batch), dtype=np.uint8
            )
            if config.store_dense_events
            else None
        )
        dense_contact_events = (
            np.empty(
                (config.contacts, config.time_steps, batch), dtype=np.uint8
            )
            if config.store_dense_events
            else None
        )
        for local_trial, global_trial in enumerate(range(first, last + 1)):
            protocol = _draw_protocol(teacher, config, global_trial)
            trajectory = _simulate_trial(teacher, config, protocol)
            arrays["sample_indices"][local_trial] = global_trial
            arrays["split_code"][local_trial] = _split_code(
                config, global_trial
            )
            for key in (
                "contact_axon",
                "contact_segment",
                "contact_kind",
                "contact_strength",
                "axon_kind",
            ):
                arrays[key][:, local_trial] = protocol[key]
            for key, target in trajectory.items():
                arrays[key][..., local_trial] = target
            event_axons.append(protocol["event_axon"])
            event_bins.append(protocol["event_time_bin"])
            event_offsets.append(
                event_offsets[-1] + len(protocol["event_axon"])
            )
            if dense_axon_events is not None:
                dense_axon_events[..., local_trial] = protocol[
                    "axon_events"
                ].astype(np.uint8)
                dense_contact_events[..., local_trial] = protocol[
                    "axon_events"
                ][protocol["contact_axon"] - 1].astype(np.uint8)
        arrays["event_trial_offset"] = np.asarray(
            event_offsets, dtype=np.int64
        )
        arrays["event_axon"] = (
            np.concatenate(event_axons)
            if event_axons
            else np.empty(0, dtype=np.int32)
        )
        arrays["event_time_bin"] = (
            np.concatenate(event_bins)
            if event_bins
            else np.empty(0, dtype=np.int32)
        )
        arrays["diagnostic_segment_indices"] = (
            teacher.diagnostic_indices.copy()
        )
        arrays["time_ms"] = (
            np.arange(1, config.time_steps + 1, dtype=np.float32)
            * config.sample_dt_ms
        )
        if dense_axon_events is not None:
            arrays["axon_event_spike"] = dense_axon_events
            arrays["event_spike"] = dense_contact_events
        shard_metadata = {
            "schema_name": SCHEMA_NAME,
            "teacher_contract_sha256": contract[
                "teacher_contract_sha256"
            ],
            "shard_index": shard_index,
            "global_first": first,
            "global_last": last,
            "samples": batch,
            "spike_positive_bins": int(np.count_nonzero(
                arrays["target_spike"]
            )),
            "voltage_minimum_mv": float(np.min(arrays["target_voltage"])),
            "voltage_maximum_mv": float(np.max(arrays["target_voltage"])),
            "nmda_absolute_mean_na": float(
                np.mean(np.abs(arrays["target_nmda"]))
            ),
        }
        arrays["metadata_json"] = np.asarray(
            _json_canonical(shard_metadata)
        )
        shard_name = f"neuron_teacher_{shard_index:05d}.npz"
        shard_path = output / shard_name
        _atomic_save_npz(shard_path, arrays)
        manifest["shards"].append(
            {
                **shard_metadata,
                "path": shard_name,
                "sha256": _sha256_file(shard_path),
                "bytes": shard_path.stat().st_size,
                "split_counts": {
                    "train": int(np.count_nonzero(
                        arrays["split_code"] == TRAIN_SPLIT
                    )),
                    "validation": int(np.count_nonzero(
                        arrays["split_code"] == VALIDATION_SPLIT
                    )),
                    "test": int(np.count_nonzero(
                        arrays["split_code"] == TEST_SPLIT
                    )),
                },
            }
        )
        _atomic_write_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "event": "neuron_teacher_shard_complete",
                    "shard": shard_index,
                    "first": first,
                    "last": last,
                    "path": str(shard_path),
                }
            ),
            flush=True,
        )
    manifest["completion_state"] = "complete"
    manifest["completed_trials"] = config.total_trials
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _config_from_arguments(arguments: argparse.Namespace) -> TeacherConfig:
    preset = PRESETS[arguments.preset]

    def selected(name: str) -> Any:
        value = getattr(arguments, name)
        return getattr(preset, name) if value is None else value

    return TeacherConfig(
        preset=arguments.preset,
        train_trials=selected("train_trials"),
        validation_trials=selected("validation_trials"),
        test_trials=selected("test_trials"),
        duration_ms=selected("duration_ms"),
        axons=selected("axons"),
        contacts_per_axon=selected("contacts_per_axon"),
        diagnostic_segments=selected("diagnostic_segments"),
        shard_size=selected("shard_size"),
        rate_hz=selected("rate_hz"),
        burst_probability=selected("burst_probability"),
        excitatory_fraction=arguments.excitatory_fraction,
        minimum_strength=arguments.minimum_strength,
        maximum_strength=arguments.maximum_strength,
        seed=arguments.seed,
        dt_ms=arguments.dt_ms,
        sample_dt_ms=arguments.sample_dt_ms,
        v_init_mv=arguments.v_init_mv,
        celsius=arguments.celsius,
        spike_threshold_mv=arguments.spike_threshold_mv,
        ca_event_cai_mm=arguments.ca_event_cai_mm,
        ca_event_voltage_mv=arguments.ca_event_voltage_mv,
        store_dense_events=(
            preset.store_dense_events
            if arguments.store_dense_events is None
            else arguments.store_dense_events
        ),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modeldb-root",
        default=os.environ.get(
            "HAY_MODELDB_ROOT", "/mnt/c/tmp/hay_modeldb_139653"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--preset", choices=tuple(PRESETS), default="smoke"
    )
    for name, value_type in (
        ("train-trials", int),
        ("validation-trials", int),
        ("test-trials", int),
        ("duration-ms", int),
        ("axons", int),
        ("contacts-per-axon", int),
        ("diagnostic-segments", int),
        ("shard-size", int),
        ("rate-hz", float),
        ("burst-probability", float),
    ):
        parser.add_argument(f"--{name}", type=value_type)
    parser.add_argument("--excitatory-fraction", type=float, default=0.8)
    parser.add_argument("--minimum-strength", type=float, default=0.1)
    parser.add_argument("--maximum-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0x4841595457494E50)
    parser.add_argument("--dt-ms", type=float, default=DEFAULT_DT_MS)
    parser.add_argument(
        "--sample-dt-ms", type=float, default=DEFAULT_SAMPLE_DT_MS
    )
    parser.add_argument("--v-init-mv", type=float, default=DEFAULT_V_INIT_MV)
    parser.add_argument("--celsius", type=float, default=DEFAULT_CELSIUS)
    parser.add_argument(
        "--spike-threshold-mv",
        type=float,
        default=DEFAULT_SPIKE_THRESHOLD_MV,
    )
    parser.add_argument(
        "--ca-event-cai-mm", type=float, default=DEFAULT_CA_EVENT_CAI_MM
    )
    parser.add_argument(
        "--ca-event-voltage-mv",
        type=float,
        default=DEFAULT_CA_EVENT_VOLTAGE_MV,
    )
    dense = parser.add_mutually_exclusive_group()
    dense.add_argument(
        "--store-dense-events",
        dest="store_dense_events",
        action="store_true",
    )
    dense.add_argument(
        "--no-store-dense-events",
        dest="store_dense_events",
        action="store_false",
    )
    parser.set_defaults(store_dense_events=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    config = _config_from_arguments(arguments)
    manifest = generate_dataset(
        arguments.modeldb_root, arguments.output, config
    )
    print(
        json.dumps(
            {
                "event": "neuron_teacher_complete",
                "manifest": str(
                    pathlib.Path(arguments.output).resolve() / "manifest.json"
                ),
                "teacher_contract_sha256": manifest[
                    "teacher_contract_sha256"
                ],
                "trials": manifest["completed_trials"],
                "shards": len(manifest["shards"]),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
