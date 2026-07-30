#!/usr/bin/env python3
"""Canonical final Hay/NEURON teacher for HD-SWSNN-TwinProp.

This is the production data source for:

    detailed Hay model -> digital twin -> 11-state distillation -> freeze

Unlike ``neuron_hay_teacher.py`` (retained as a control), this generator
implements all published temporal and conductance elements of the TwinProp
random-drive protocol: 50,000 ten-second training simulations and 2,000
independent held-out simulations, Poisson spike trains driven by
Gaussian-smoothed piecewise-constant rates, uniform [0, 1] strengths, and the
one-E/one-I-contact-per-micrometre constraint. The Step-1 axon/contact scale is
not published; production therefore requires explicit acknowledgement of the
documented 8,000-contact interpretation and never claims byte-identical or
fully paper-scale reproduction.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import multiprocessing
import os
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

import neuron_hay_teacher as control


FINAL_SCHEMA_NAME = "hd_swsnn_twinprop.neuron_teacher.final.v2"
FINAL_MODEL_NAME = "HD-SWSNN-TwinProp"
PAPER_TRAIN_SIMULATIONS = 50_000
PAPER_TEST_SIMULATIONS = 2_000
PAPER_DURATION_MS = 10_000
PAPER_RATE_TIMESCALE_MIN_MS = 10.0
PAPER_RATE_TIMESCALE_MAX_MS = 1_000.0
STRENGTH_MINIMUM = 0.0
STRENGTH_MAXIMUM = 1.0


@dataclasses.dataclass(frozen=True)
class FinalPreset:
    train_trials: int
    validation_trials_from_train: int
    test_trials: int
    duration_ms: int
    axons: int
    mean_contacts_per_axon: float
    exact_contacts_per_axon: bool
    default_excitatory_fraction: float
    diagnostic_segments: int
    diagnostic_stride_bins: int
    shard_size: int
    store_dense_axon_events: bool


FINAL_PRESETS = {
    "tiny": FinalPreset(
        1, 0, 0, 8, 4, 2.0, False, 0.75, 8, 2, 1, True
    ),
    # 40 public-training-pool simulations are split 32 fit + 8 validation;
    # the 8 held-out trials remain untouched.
    "smoke": FinalPreset(
        40, 8, 8, 100, 64, 20.0, False, 0.80, 16, 5, 2, True
    ),
    "production": FinalPreset(
        PAPER_TRAIN_SIMULATIONS,
        0,
        PAPER_TEST_SIMULATIONS,
        PAPER_DURATION_MS,
        400,
        20.0,
        True,
        0.50,
        32,
        10,
        2,
        False,
    ),
}


@dataclasses.dataclass(frozen=True)
class FinalTeacherConfig:
    preset: str
    train_trials: int
    validation_trials_from_train: int
    test_trials: int
    duration_ms: int
    axons: int
    mean_contacts_per_axon: float
    exact_contacts_per_axon: bool
    diagnostic_segments: int
    diagnostic_stride_bins: int
    shard_size: int
    excitatory_fraction: float
    rate_min_hz: float
    rate_max_hz: float
    seed: int
    dt_ms: float
    sample_dt_ms: float
    v_init_mv: float
    celsius: float
    spike_threshold_mv: float
    ca_event_cai_mm: float
    ca_event_voltage_mv: float
    store_dense_axon_events: bool
    connectivity_interpretation_acknowledged: bool

    @property
    def total_trials(self) -> int:
        return self.train_trials + self.test_trials

    @property
    def time_steps(self) -> int:
        return int(round(self.duration_ms / self.sample_dt_ms))

    @property
    def diagnostic_time_indices(self) -> np.ndarray:
        return np.arange(
            self.diagnostic_stride_bins - 1,
            self.time_steps,
            self.diagnostic_stride_bins,
            dtype=np.int32,
        )

    def validate(self) -> None:
        if self.train_trials < 1 or self.test_trials < 0:
            raise ValueError("invalid train/test trial counts")
        if not 0 <= self.validation_trials_from_train < self.train_trials:
            raise ValueError(
                "validation_trials_from_train must be within train_trials"
            )
        if self.duration_ms < 2 or self.axons < 2:
            raise ValueError("duration and axon count are too small")
        if self.mean_contacts_per_axon <= 0:
            raise ValueError("mean_contacts_per_axon must be positive")
        if self.diagnostic_segments < 2 or self.diagnostic_stride_bins < 1:
            raise ValueError("invalid diagnostic sparsity")
        if self.shard_size < 1:
            raise ValueError("shard_size must be positive")
        if not 0.0 < self.excitatory_fraction < 1.0:
            raise ValueError("excitatory_fraction must be in (0,1)")
        if self.preset == "production":
            if not self.connectivity_interpretation_acknowledged:
                raise ValueError(
                    "production connectivity is ambiguous in the public "
                    "methods; pass --acknowledge-8000-contact-interpretation "
                    "to use 400 axons x 20 contacts, 50/50 E/I"
                )
            if not (
                self.axons == 400
                and self.exact_contacts_per_axon
                and self.mean_contacts_per_axon == 20.0
                and self.excitatory_fraction == 0.5
            ):
                raise ValueError(
                    "acknowledged production interpretation must yield "
                    "exactly 4,000 E and 4,000 I contacts"
                )
        if not 0.0 <= self.rate_min_hz < self.rate_max_hz:
            raise ValueError("require 0 <= rate_min_hz < rate_max_hz")
        ratio = self.sample_dt_ms / self.dt_ms
        if not math.isclose(ratio, round(ratio), abs_tol=1e-9):
            raise ValueError("sample_dt_ms must be divisible by dt_ms")
        duration_ratio = self.duration_ms / self.sample_dt_ms
        if not math.isclose(duration_ratio, round(duration_ratio), abs_tol=1e-9):
            raise ValueError("duration_ms must be divisible by sample_dt_ms")

    def simulation_config(self) -> control.TeacherConfig:
        """Adapter for the canonical detailed-cell integrator."""
        return control.TeacherConfig(
            preset=self.preset,
            train_trials=max(self.train_trials, 1),
            validation_trials=0,
            test_trials=self.test_trials,
            duration_ms=self.duration_ms,
            axons=self.axons,
            contacts_per_axon=max(1, round(self.mean_contacts_per_axon)),
            diagnostic_segments=self.diagnostic_segments,
            shard_size=self.shard_size,
            rate_hz=max(self.rate_max_hz, 1.0),
            burst_probability=0.0,
            excitatory_fraction=self.excitatory_fraction,
            minimum_strength=STRENGTH_MINIMUM,
            maximum_strength=STRENGTH_MAXIMUM,
            seed=self.seed,
            dt_ms=self.dt_ms,
            sample_dt_ms=self.sample_dt_ms,
            v_init_mv=self.v_init_mv,
            celsius=self.celsius,
            spike_threshold_mv=self.spike_threshold_mv,
            ca_event_cai_mm=self.ca_event_cai_mm,
            ca_event_voltage_mv=self.ca_event_voltage_mv,
            store_dense_events=False,
        )


@dataclasses.dataclass(frozen=True)
class LocationSlot:
    index: int
    section_index: int
    section_name: str
    micron_bin: int
    x: float
    path_distance_um: float
    segment_index: int


def _section_location_slots(
    teacher: control.HayTeacher,
) -> tuple[list[LocationSlot], list[dict[str, Any]]]:
    by_section: dict[str, list[control.SegmentInfo]] = {}
    for record in teacher.segments:
        if record.section_region in ("basal", "apical"):
            by_section.setdefault(record.section_name, []).append(record)
    slots: list[LocationSlot] = []
    section_catalog: list[dict[str, Any]] = []
    for section_index, name in enumerate(sorted(by_section), start=1):
        records = sorted(by_section[name], key=lambda item: item.x)
        section = records[0].segment.sec
        # NEURON increments L5PCtemplate[N] for each HOC instantiation. The
        # process-local object number is not part of the morphology and must
        # never enter a resumable content hash.
        canonical_name = name.split(".", 1)[-1]
        length_um = float(section.L)
        section_catalog.append(
            {
                "index": section_index,
                "name": canonical_name,
                "region": records[0].section_region,
                "length_um": length_um,
                "micron_slots": int(math.ceil(length_um)),
            }
        )
        for micron_bin in range(int(math.ceil(length_um))):
            position_um = min(
                micron_bin + 0.5,
                max(length_um - np.finfo(np.float32).eps, 0.0),
            )
            x = min(max(position_um / length_um, 1e-9), 1.0 - 1e-9)
            segment = section(x)
            target = min(records, key=lambda item: abs(item.x - segment.x))
            slots.append(
                LocationSlot(
                    index=len(slots) + 1,
                    section_index=section_index,
                    section_name=canonical_name,
                    micron_bin=micron_bin,
                    x=x,
                    path_distance_um=float(
                        control.h.distance(x, sec=section)
                    ),
                    segment_index=target.index,
                )
            )
    return slots, section_catalog


def _gaussian_smooth_reflect(
    piecewise_rate_hz: np.ndarray, sigma_bins: float
) -> np.ndarray:
    """Gaussian smoothing with reflect boundaries using an FFT convolution."""
    time_steps = piecewise_rate_hz.shape[1]
    if time_steps <= 1 or sigma_bins <= 1e-6:
        return piecewise_rate_hz.astype(np.float64, copy=True)
    radius = min(time_steps - 1, max(1, int(math.ceil(4.0 * sigma_bins))))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / sigma_bins))
    kernel /= np.sum(kernel)
    padded = np.pad(
        piecewise_rate_hz.astype(np.float64, copy=False),
        ((0, 0), (radius, radius)),
        mode="reflect",
    )
    full_length = padded.shape[1] + len(kernel) - 1
    fft_length = 1 << (full_length - 1).bit_length()
    signal_spectrum = np.fft.rfft(padded, n=fft_length, axis=1)
    kernel_spectrum = np.fft.rfft(kernel, n=fft_length)
    convolution = np.fft.irfft(
        signal_spectrum * kernel_spectrum[None, :],
        n=fft_length,
        axis=1,
    )[:, :full_length]
    same = convolution[:, radius : radius + padded.shape[1]]
    smoothed = same[:, radius : radius + time_steps]
    return np.maximum(smoothed, 0.0)


def _smoothed_piecewise_poisson(
    rng: np.random.Generator, config: FinalTeacherConfig
) -> tuple[np.ndarray, float, float, dict[str, float]]:
    window_ms = float(
        rng.uniform(
            PAPER_RATE_TIMESCALE_MIN_MS, PAPER_RATE_TIMESCALE_MAX_MS
        )
    )
    sigma_ms = float(
        rng.uniform(
            PAPER_RATE_TIMESCALE_MIN_MS, PAPER_RATE_TIMESCALE_MAX_MS
        )
    )
    window_bins = max(1, int(round(window_ms / config.sample_dt_ms)))
    windows = int(math.ceil(config.time_steps / window_bins))
    levels = rng.uniform(
        config.rate_min_hz,
        config.rate_max_hz,
        size=(config.axons, windows),
    )
    piecewise = np.repeat(levels, window_bins, axis=1)[
        :, : config.time_steps
    ]
    smoothed = _gaussian_smooth_reflect(
        piecewise, sigma_ms / config.sample_dt_ms
    )
    expected_count = smoothed * (config.sample_dt_ms / 1_000.0)
    count = rng.poisson(expected_count)
    if np.max(count, initial=0) > np.iinfo(np.uint8).max:
        raise OverflowError("Poisson count exceeds uint8 event representation")
    return (
        count.astype(np.uint8),
        window_ms,
        sigma_ms,
        {
            "mean_rate_hz": float(np.mean(smoothed)),
            "std_rate_hz": float(np.std(smoothed)),
            "minimum_rate_hz": float(np.min(smoothed)),
            "maximum_rate_hz": float(np.max(smoothed)),
        },
    )


def _draw_final_protocol(
    teacher: control.HayTeacher,
    slots: Sequence[LocationSlot],
    config: FinalTeacherConfig,
    global_trial: int,
) -> dict[str, np.ndarray | float | dict[str, float]]:
    rng = np.random.default_rng(control._trial_seed(config.seed, global_trial))
    axon_kind = np.full(config.axons, control.INHIBITORY, dtype=np.uint8)
    excitatory_axons = max(
        1, min(config.axons - 1, round(config.axons * config.excitatory_fraction))
    )
    axon_kind[rng.permutation(config.axons)[:excitatory_axons]] = (
        control.EXCITATORY
    )
    if config.exact_contacts_per_axon:
        contact_count_per_axon = np.full(
            config.axons,
            int(round(config.mean_contacts_per_axon)),
            dtype=np.int32,
        )
    else:
        contact_count_per_axon = rng.poisson(
            config.mean_contacts_per_axon, size=config.axons
        ).astype(np.int32)
    contact_axon = np.repeat(
        np.arange(1, config.axons + 1, dtype=np.int32),
        contact_count_per_axon,
    )
    contact_kind = axon_kind[contact_axon - 1]
    slot_indices = np.arange(len(slots), dtype=np.int32)
    selected_slot = np.empty(len(contact_axon), dtype=np.int32)
    for kind in (control.EXCITATORY, control.INHIBITORY):
        contact_indices = np.flatnonzero(contact_kind == kind)
        if len(contact_indices) > len(slots):
            raise ValueError(
                f"{len(contact_indices)} type-{int(kind)} contacts exceed "
                f"{len(slots)} one-micrometre dendritic slots"
            )
        # E and I draw independently; within each population a slot can occur
        # only once. This is exactly <=1 E and <=1 I per dendrite micrometre.
        selected_slot[contact_indices] = rng.choice(
            slot_indices, size=len(contact_indices), replace=False
        )
    chosen = [slots[int(index)] for index in selected_slot]
    contact_strength = rng.random(len(contact_axon)).astype(np.float32)
    axon_event_count, window_ms, sigma_ms, rate_statistics = (
        _smoothed_piecewise_poisson(rng, config)
    )
    event_axon_zero, event_time_bin = np.nonzero(axon_event_count)
    event_count = axon_event_count[event_axon_zero, event_time_bin]
    return {
        "axon_kind": axon_kind,
        "contact_count_per_axon": contact_count_per_axon,
        "contact_axon": contact_axon,
        "contact_kind": contact_kind,
        "contact_strength": contact_strength,
        "contact_location_slot": np.asarray(
            [slot.index for slot in chosen], dtype=np.int32
        ),
        "contact_section": np.asarray(
            [slot.section_index for slot in chosen], dtype=np.int32
        ),
        "contact_x": np.asarray(
            [slot.x for slot in chosen], dtype=np.float32
        ),
        "contact_path_distance_um": np.asarray(
            [slot.path_distance_um for slot in chosen], dtype=np.float32
        ),
        "contact_segment": np.asarray(
            [slot.segment_index for slot in chosen], dtype=np.int32
        ),
        "axon_events": axon_event_count,
        "event_axon": (event_axon_zero + 1).astype(np.int32),
        "event_time_bin": event_time_bin.astype(np.int32),
        "event_count": event_count.astype(np.uint8),
        "rate_window_ms": window_ms,
        "rate_sigma_ms": sigma_ms,
        "rate_statistics": rate_statistics,
    }


FINAL_ARRAY_CONTRACT = {
    "sample_indices": {"axes": ["trial"], "dtype": "int32"},
    "split_code": {
        "axes": ["trial"],
        "dtype": "uint8",
        "codes": {"train": 1, "held_out_test": 3},
    },
    "contact_trial_offset": {
        "axes": ["trial_plus_one"],
        "dtype": "int64",
    },
    "contact_axon": {
        "axes": ["ragged_contact"],
        "dtype": "int32",
        "index_base": 1,
    },
    "contact_segment": {
        "axes": ["ragged_contact"],
        "dtype": "int32",
        "index_base": 1,
    },
    "contact_location_slot": {
        "axes": ["ragged_contact"],
        "dtype": "int32",
        "index_base": 1,
        "constraint": "unique within each trial and E/I kind",
    },
    "contact_section": {
        "axes": ["ragged_contact"],
        "dtype": "int32",
        "index_base": 1,
    },
    "contact_x": {
        "axes": ["ragged_contact"],
        "dtype": "float32",
        "range": [0.0, 1.0],
    },
    "contact_path_distance_um": {
        "axes": ["ragged_contact"],
        "dtype": "float32",
        "units": "um",
    },
    "contact_kind": {
        "axes": ["ragged_contact"],
        "dtype": "uint8",
        "codes": {"excitatory": 1, "inhibitory": 2},
    },
    "contact_strength": {
        "axes": ["ragged_contact"],
        "dtype": "float32",
        "distribution": "Uniform[0,1)",
    },
    "event_trial_offset": {
        "axes": ["trial_plus_one"],
        "dtype": "int64",
    },
    "event_axon": {
        "axes": ["ragged_event"],
        "dtype": "int32",
        "index_base": 1,
    },
    "event_time_bin": {
        "axes": ["ragged_event"],
        "dtype": "int32",
        "index_base": 0,
    },
    "event_count": {
        "axes": ["ragged_event"],
        "dtype": "uint8",
        "meaning": "Poisson multiplicity in one sample bin",
    },
    "target_voltage": {
        "axes": ["time", "trial"],
        "dtype": "float32",
        "units": "mV",
    },
    "target_spike": {
        "axes": ["time", "trial"],
        "dtype": "float32",
    },
    "target_nmda": {
        "axes": ["region", "time", "trial"],
        "dtype": "float32",
        "units": "nA",
        "sign": "outward_positive",
    },
    "target_compartment_voltage": {
        "axes": ["diagnostic_segment", "diagnostic_time", "trial"],
        "dtype": "float32",
        "units": "mV",
    },
    "target_compartment_nmda": {
        "axes": ["diagnostic_segment", "diagnostic_time", "trial"],
        "dtype": "float32",
        "units": "nA",
    },
    "target_dendritic_cai": {
        "axes": ["diagnostic_segment", "diagnostic_time", "trial"],
        "dtype": "float32",
        "units": "mM",
    },
    "target_dendritic_ica": {
        "axes": ["diagnostic_segment", "diagnostic_time", "trial"],
        "dtype": "float32",
        "units": "mA/cm2",
    },
    "target_ca_event": {
        "axes": ["diagnostic_segment", "diagnostic_time", "trial"],
        "dtype": "uint8",
    },
}


_WORKER_CONTEXT: dict[str, Any] = {}


def _worker_context(
    modeldb_root: str, config: FinalTeacherConfig
) -> tuple[control.HayTeacher, list[LocationSlot]]:
    key = (
        str(pathlib.Path(modeldb_root).resolve()),
        config.diagnostic_segments,
    )
    if _WORKER_CONTEXT.get("key") != key:
        teacher = control.instantiate_teacher(
            modeldb_root, config.diagnostic_segments
        )
        slots, _ = _section_location_slots(teacher)
        _WORKER_CONTEXT.clear()
        _WORKER_CONTEXT.update(key=key, teacher=teacher, slots=slots)
    return _WORKER_CONTEXT["teacher"], _WORKER_CONTEXT["slots"]


def _split_code(config: FinalTeacherConfig, global_trial: int) -> np.uint8:
    return (
        control.TRAIN_SPLIT
        if global_trial <= config.train_trials
        else control.TEST_SPLIT
    )


def _generate_shard_worker(
    modeldb_root: str,
    output_directory: str,
    config_dictionary: dict[str, Any],
    teacher_contract_sha256: str,
    shard_index: int,
    first: int,
    last: int,
) -> dict[str, Any]:
    config = FinalTeacherConfig(**config_dictionary)
    teacher, slots = _worker_context(modeldb_root, config)
    simulation_config = config.simulation_config()
    batch = last - first + 1
    time_steps = config.time_steps
    diagnostic_time_indices = config.diagnostic_time_indices
    diagnostic_count = len(teacher.diagnostic_indices)
    diagnostic_times = len(diagnostic_time_indices)
    arrays: dict[str, Any] = {
        "sample_indices": np.arange(first, last + 1, dtype=np.int32),
        "split_code": np.asarray(
            [_split_code(config, trial) for trial in range(first, last + 1)],
            dtype=np.uint8,
        ),
        "axon_kind": np.empty((config.axons, batch), dtype=np.uint8),
        "contact_count_per_axon": np.empty(
            (config.axons, batch), dtype=np.int32
        ),
        "rate_window_ms": np.empty(batch, dtype=np.float32),
        "rate_sigma_ms": np.empty(batch, dtype=np.float32),
        "rate_mean_hz": np.empty(batch, dtype=np.float32),
        "rate_std_hz": np.empty(batch, dtype=np.float32),
        "target_voltage": np.empty((time_steps, batch), dtype=np.float32),
        "target_spike": np.empty((time_steps, batch), dtype=np.float32),
        "target_nmda": np.empty(
            (len(control.REGION_NAMES), time_steps, batch), dtype=np.float32
        ),
        "target_compartment_voltage": np.empty(
            (diagnostic_count, diagnostic_times, batch), dtype=np.float32
        ),
        "target_compartment_nmda": np.empty(
            (diagnostic_count, diagnostic_times, batch), dtype=np.float32
        ),
        "target_dendritic_cai": np.empty(
            (diagnostic_count, diagnostic_times, batch), dtype=np.float32
        ),
        "target_dendritic_ica": np.empty(
            (diagnostic_count, diagnostic_times, batch), dtype=np.float32
        ),
        "target_ca_event": np.empty(
            (diagnostic_count, diagnostic_times, batch), dtype=np.uint8
        ),
        "diagnostic_segment_indices": teacher.diagnostic_indices.copy(),
        "diagnostic_time_indices": diagnostic_time_indices.copy(),
        "time_ms": (
            np.arange(1, time_steps + 1, dtype=np.float32)
            * config.sample_dt_ms
        ),
        "diagnostic_time_ms": (
            (diagnostic_time_indices.astype(np.float32) + 1.0)
            * config.sample_dt_ms
        ),
    }
    contact_fields = (
        "contact_axon",
        "contact_segment",
        "contact_location_slot",
        "contact_section",
        "contact_x",
        "contact_path_distance_um",
        "contact_kind",
        "contact_strength",
    )
    ragged_contacts: dict[str, list[np.ndarray]] = {
        field: [] for field in contact_fields
    }
    contact_offsets = [0]
    event_axons: list[np.ndarray] = []
    event_bins: list[np.ndarray] = []
    event_counts: list[np.ndarray] = []
    event_offsets = [0]
    dense_events = (
        np.empty((config.axons, time_steps, batch), dtype=np.uint8)
        if config.store_dense_axon_events
        else None
    )
    for local_trial, global_trial in enumerate(range(first, last + 1)):
        protocol = _draw_final_protocol(
            teacher, slots, config, global_trial
        )
        trajectory = control._simulate_trial(
            teacher, simulation_config, protocol
        )
        arrays["axon_kind"][:, local_trial] = protocol["axon_kind"]
        arrays["contact_count_per_axon"][:, local_trial] = protocol[
            "contact_count_per_axon"
        ]
        arrays["rate_window_ms"][local_trial] = protocol["rate_window_ms"]
        arrays["rate_sigma_ms"][local_trial] = protocol["rate_sigma_ms"]
        arrays["rate_mean_hz"][local_trial] = protocol["rate_statistics"][
            "mean_rate_hz"
        ]
        arrays["rate_std_hz"][local_trial] = protocol["rate_statistics"][
            "std_rate_hz"
        ]
        arrays["target_voltage"][:, local_trial] = trajectory[
            "target_voltage"
        ]
        arrays["target_spike"][:, local_trial] = trajectory["target_spike"]
        arrays["target_nmda"][:, :, local_trial] = trajectory["target_nmda"]
        for field in (
            "target_compartment_voltage",
            "target_compartment_nmda",
            "target_dendritic_cai",
            "target_dendritic_ica",
            "target_ca_event",
        ):
            arrays[field][:, :, local_trial] = trajectory[field][
                :, diagnostic_time_indices
            ]
        for field in contact_fields:
            ragged_contacts[field].append(protocol[field])
        contact_offsets.append(
            contact_offsets[-1] + len(protocol["contact_axon"])
        )
        event_axons.append(protocol["event_axon"])
        event_bins.append(protocol["event_time_bin"])
        event_counts.append(protocol["event_count"])
        event_offsets.append(event_offsets[-1] + len(protocol["event_axon"]))
        if dense_events is not None:
            dense_events[:, :, local_trial] = protocol["axon_events"]
    arrays["contact_trial_offset"] = np.asarray(
        contact_offsets, dtype=np.int64
    )
    for field, parts in ragged_contacts.items():
        arrays[field] = np.concatenate(parts)
    arrays["event_trial_offset"] = np.asarray(event_offsets, dtype=np.int64)
    arrays["event_axon"] = np.concatenate(event_axons)
    arrays["event_time_bin"] = np.concatenate(event_bins)
    arrays["event_count"] = np.concatenate(event_counts)
    if dense_events is not None:
        arrays["axon_event_count"] = dense_events
    shard_metadata = {
        "schema_name": FINAL_SCHEMA_NAME,
        "teacher_contract_sha256": teacher_contract_sha256,
        "shard_index": shard_index,
        "global_first": first,
        "global_last": last,
        "samples": batch,
        "contacts": int(contact_offsets[-1]),
        "events": int(event_offsets[-1]),
        "spike_positive_bins": int(np.count_nonzero(arrays["target_spike"])),
        "voltage_minimum_mv": float(np.min(arrays["target_voltage"])),
        "voltage_maximum_mv": float(np.max(arrays["target_voltage"])),
        "nmda_absolute_mean_na": float(
            np.mean(np.abs(arrays["target_nmda"]))
        ),
    }
    arrays["metadata_json"] = np.asarray(
        control._json_canonical(shard_metadata)
    )
    output = pathlib.Path(output_directory)
    shard_name = f"neuron_hay_final_{shard_index:05d}.npz"
    shard_path = output / shard_name
    control._atomic_save_npz(shard_path, arrays)
    record = {
        **shard_metadata,
        "path": shard_name,
        "sha256": control._sha256_file(shard_path),
        "bytes": shard_path.stat().st_size,
        "split_counts": {
            "train": int(np.count_nonzero(
                arrays["split_code"] == control.TRAIN_SPLIT
            )),
            "held_out_test": int(np.count_nonzero(
                arrays["split_code"] == control.TEST_SPLIT
            )),
        },
    }
    done_path = output / f"neuron_hay_final_{shard_index:05d}.done.json"
    control._atomic_write_json(done_path, record)
    return record


def _valid_completion_record(
    output: pathlib.Path,
    expected_contract: str,
    shard_index: int,
    first: int,
    last: int,
) -> dict[str, Any] | None:
    done_path = output / f"neuron_hay_final_{shard_index:05d}.done.json"
    if not done_path.is_file():
        return None
    record = json.loads(done_path.read_text("utf-8"))
    if (
        record.get("teacher_contract_sha256") != expected_contract
        or record.get("shard_index") != shard_index
        or record.get("global_first") != first
        or record.get("global_last") != last
    ):
        raise RuntimeError(f"mismatched completion record: {done_path}")
    shard_path = output / record["path"]
    if not shard_path.is_file():
        raise RuntimeError(f"completion record has no shard: {shard_path}")
    if control._sha256_file(shard_path) != record["sha256"]:
        raise RuntimeError(f"shard hash mismatch: {shard_path}")
    return record


def _final_contract(
    teacher: control.HayTeacher,
    config: FinalTeacherConfig,
    section_catalog: list[dict[str, Any]],
    slots: Sequence[LocationSlot],
) -> dict[str, Any]:
    source_hashes = dict(teacher.hashes)
    source_hashes["final_generator_source_sha256"] = control._sha256_file(
        pathlib.Path(__file__).resolve()
    )
    slot_digest = hashlib.sha256()
    for slot in slots:
        slot_digest.update(
            control._json_canonical(
                {
                    "index": int(slot.index),
                    "section_index": int(slot.section_index),
                    "section_name": str(slot.section_name),
                    "micron_bin": int(slot.micron_bin),
                    "x": float(slot.x),
                    "path_distance_um": float(slot.path_distance_um),
                    "segment_index": int(slot.segment_index),
                }
            ).encode("utf-8")
        )
    contract: dict[str, Any] = {
        "schema_name": FINAL_SCHEMA_NAME,
        "model_name": FINAL_MODEL_NAME,
        "config": dataclasses.asdict(config),
        "source_hashes": source_hashes,
        "location_slot_sha256": slot_digest.hexdigest(),
        "paper_protocol": {
            "train_simulations": PAPER_TRAIN_SIMULATIONS,
            "held_out_test_simulations": PAPER_TEST_SIMULATIONS,
            "duration_ms": PAPER_DURATION_MS,
            "poisson_rate": "Gaussian-smoothed piecewise constant",
            "rate_window_range_ms": [
                PAPER_RATE_TIMESCALE_MIN_MS,
                PAPER_RATE_TIMESCALE_MAX_MS,
            ],
            "gaussian_sigma_range_ms": [
                PAPER_RATE_TIMESCALE_MIN_MS,
                PAPER_RATE_TIMESCALE_MAX_MS,
            ],
            "resample_timescales_independently_per_simulation": True,
            "strength_distribution": "Uniform[0,1)",
            "mean_contacts_per_axon": 20,
            "contact_density": "<=1 E and <=1 I per dendrite micron",
        },
        "connectivity_scale_conflict": {
            "task_protocol_wording": (
                "8,000 total synaptic count, described parenthetically as "
                "4,000 excitatory / 4,000 inhibitory"
            ),
            "other_wording": "average 20 contacts per input axon",
            "step1_random_twin_axon_and_contact_count_published": False,
            "literal_4000E_4000I_axons_compatible_with_density": False,
            "production_interpretation": (
                "400 axons (200 E, 200 I) x exact 20 contacts = "
                "4,000 E and 4,000 I contacts"
            ),
            "interpretation_explicitly_acknowledged": (
                config.connectivity_interpretation_acknowledged
            ),
            "fully_paper_scale_claim": False,
        },
        "reconstruction_choices_not_published": {
            "rate_level_distribution": (
                f"Uniform[{config.rate_min_hz},{config.rate_max_hz}) Hz"
            ),
            "timescale_sampling_distribution": (
                "continuous Uniform[10,1000] ms"
            ),
            "gaussian_boundary_mode": "reflect",
            "axon_count": config.axons,
            "excitatory_fraction": config.excitatory_fraction,
            "contacts_per_axon_distribution": (
                f"exact {config.mean_contacts_per_axon}"
                if config.exact_contacts_per_axon
                else f"Poisson(mean={config.mean_contacts_per_axon})"
            ),
            "location_within_each_micron": "bin midpoint",
            "neuron_discretization": (
                "official L5PCtemplate geom_nseg; sub-micron locations map "
                "to their containing electrical segment"
            ),
        },
        "section_catalog": section_catalog,
    }
    contract["teacher_contract_sha256"] = hashlib.sha256(
        control._json_canonical(contract).encode("utf-8")
    ).hexdigest()
    return contract


def generate_final_dataset(
    modeldb_root: os.PathLike[str] | str,
    output_directory: os.PathLike[str] | str,
    config: FinalTeacherConfig,
    *,
    workers: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    config.validate()
    if workers < 1:
        raise ValueError("workers must be positive")
    output = pathlib.Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    teacher = control.instantiate_teacher(
        modeldb_root, config.diagnostic_segments
    )
    slots, section_catalog = _section_location_slots(teacher)
    contract = _final_contract(teacher, config, section_catalog, slots)
    contract_hash = contract["teacher_contract_sha256"]
    manifest_path = output / "manifest.json"
    existing: dict[str, Any] | None = None
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text("utf-8"))
        if (
            existing.get("schema_name") != FINAL_SCHEMA_NAME
            or existing.get("teacher_contract_sha256") != contract_hash
        ):
            raise RuntimeError(
                "output contains a different teacher contract; choose a new "
                "directory rather than overwriting it"
            )
        if not resume:
            raise RuntimeError("output already exists and resume is disabled")
    manifest: dict[str, Any] = existing or {
        "schema_version": 2,
        "schema_name": FINAL_SCHEMA_NAME,
        "model_name": FINAL_MODEL_NAME,
        "stage": "official_hay_neuron_teacher_final",
        "completion_state": "generating",
        "teacher_contract_sha256": contract_hash,
        "teacher_contract": contract,
        "teacher_contract_canonical_json": control._json_canonical(
            {
                key: value
                for key, value in contract.items()
                if key != "teacher_contract_sha256"
            }
        ),
        "source_hashes": contract["source_hashes"],
        "modeldb_source_modified_by_generator": False,
        "neuron_version": str(control.h.nrnversion()),
        "config": dataclasses.asdict(config),
        "validation_from_train_indices": list(range(
            config.train_trials - config.validation_trials_from_train + 1,
            config.train_trials + 1,
        )),
        "paper_production_contract": {
            "train_trials": PAPER_TRAIN_SIMULATIONS,
            "held_out_test_trials": PAPER_TEST_SIMULATIONS,
            "duration_ms": PAPER_DURATION_MS,
        },
        "total_segments": len(teacher.segments),
        "diagnostic_segment_indices": teacher.diagnostic_indices.tolist(),
        "diagnostic_time_indices": config.diagnostic_time_indices.tolist(),
        "diagnostic_sparsity": {
            "segments": len(teacher.diagnostic_indices),
            "time_stride_bins": config.diagnostic_stride_bins,
            "soma_voltage_spike_and_region_nmda_remain_1ms": True,
        },
        "segments": [record.public_record() for record in teacher.segments],
        "section_catalog": section_catalog,
        "location_slots": len(slots),
        "array_contract": FINAL_ARRAY_CONTRACT,
        "shards": [],
    }
    expected_shards: list[tuple[int, int, int]] = []
    for shard_index, first in enumerate(
        range(1, config.total_trials + 1, config.shard_size), start=1
    ):
        last = min(first + config.shard_size - 1, config.total_trials)
        expected_shards.append((shard_index, first, last))
    records: dict[int, dict[str, Any]] = {}
    pending: list[tuple[int, int, int]] = []
    for shard_index, first, last in expected_shards:
        record = _valid_completion_record(
            output, contract_hash, shard_index, first, last
        )
        if record is None:
            pending.append((shard_index, first, last))
        else:
            records[shard_index] = record
    manifest["shards"] = [records[index] for index in sorted(records)]
    manifest["completion_state"] = (
        "complete" if not pending else "generating"
    )
    control._atomic_write_json(manifest_path, manifest)
    worker_arguments = (
        str(pathlib.Path(modeldb_root).expanduser().resolve()),
        str(output),
        dataclasses.asdict(config),
        contract_hash,
    )
    if workers == 1:
        for shard_index, first, last in pending:
            record = _generate_shard_worker(
                *worker_arguments, shard_index, first, last
            )
            records[shard_index] = record
            manifest["shards"] = [
                records[index] for index in sorted(records)
            ]
            control._atomic_write_json(manifest_path, manifest)
            print(json.dumps(
                {
                    "event": "final_neuron_shard_complete",
                    "shard": shard_index,
                    "first": first,
                    "last": last,
                    "resumed": False,
                }
            ), flush=True)
    elif pending:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        ) as executor:
            future_to_shard = {
                executor.submit(
                    _generate_shard_worker,
                    *worker_arguments,
                    shard_index,
                    first,
                    last,
                ): (shard_index, first, last)
                for shard_index, first, last in pending
            }
            for future in concurrent.futures.as_completed(future_to_shard):
                shard_index, first, last = future_to_shard[future]
                record = future.result()
                records[shard_index] = record
                manifest["shards"] = [
                    records[index] for index in sorted(records)
                ]
                control._atomic_write_json(manifest_path, manifest)
                print(json.dumps(
                    {
                        "event": "final_neuron_shard_complete",
                        "shard": shard_index,
                        "first": first,
                        "last": last,
                        "resumed": False,
                    }
                ), flush=True)
    manifest["shards"] = [records[index] for index in sorted(records)]
    manifest["completion_state"] = (
        "complete"
        if len(records) == len(expected_shards)
        else "generating"
    )
    manifest["completed_trials"] = sum(
        record["samples"] for record in manifest["shards"]
    )
    manifest["resumable_sidecars_verified"] = True
    control._atomic_write_json(manifest_path, manifest)
    return manifest


def _config_from_arguments(arguments: argparse.Namespace) -> FinalTeacherConfig:
    preset = FINAL_PRESETS[arguments.preset]

    def selected(name: str) -> Any:
        value = getattr(arguments, name)
        return getattr(preset, name) if value is None else value

    return FinalTeacherConfig(
        preset=arguments.preset,
        train_trials=selected("train_trials"),
        validation_trials_from_train=selected(
            "validation_trials_from_train"
        ),
        test_trials=selected("test_trials"),
        duration_ms=selected("duration_ms"),
        axons=selected("axons"),
        mean_contacts_per_axon=selected("mean_contacts_per_axon"),
        exact_contacts_per_axon=preset.exact_contacts_per_axon,
        diagnostic_segments=selected("diagnostic_segments"),
        diagnostic_stride_bins=selected("diagnostic_stride_bins"),
        shard_size=selected("shard_size"),
        excitatory_fraction=(
            preset.default_excitatory_fraction
            if arguments.excitatory_fraction is None
            else arguments.excitatory_fraction
        ),
        rate_min_hz=arguments.rate_min_hz,
        rate_max_hz=arguments.rate_max_hz,
        seed=arguments.seed,
        dt_ms=arguments.dt_ms,
        sample_dt_ms=arguments.sample_dt_ms,
        v_init_mv=arguments.v_init_mv,
        celsius=arguments.celsius,
        spike_threshold_mv=arguments.spike_threshold_mv,
        ca_event_cai_mm=arguments.ca_event_cai_mm,
        ca_event_voltage_mv=arguments.ca_event_voltage_mv,
        store_dense_axon_events=(
            preset.store_dense_axon_events
            if arguments.store_dense_axon_events is None
            else arguments.store_dense_axon_events
        ),
        connectivity_interpretation_acknowledged=(
            arguments.acknowledge_8000_contact_interpretation
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
        "--preset", choices=tuple(FINAL_PRESETS), default="smoke"
    )
    for name, value_type in (
        ("train-trials", int),
        ("validation-trials-from-train", int),
        ("test-trials", int),
        ("duration-ms", int),
        ("axons", int),
        ("mean-contacts-per-axon", float),
        ("diagnostic-segments", int),
        ("diagnostic-stride-bins", int),
        ("shard-size", int),
    ):
        parser.add_argument(f"--{name}", type=value_type)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--excitatory-fraction", type=float)
    parser.add_argument(
        "--acknowledge-8000-contact-interpretation", action="store_true"
    )
    parser.add_argument("--rate-min-hz", type=float, default=0.0)
    parser.add_argument("--rate-max-hz", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0x46494E414C484159)
    parser.add_argument("--dt-ms", type=float, default=control.DEFAULT_DT_MS)
    parser.add_argument(
        "--sample-dt-ms",
        type=float,
        default=control.DEFAULT_SAMPLE_DT_MS,
    )
    parser.add_argument(
        "--v-init-mv", type=float, default=control.DEFAULT_V_INIT_MV
    )
    parser.add_argument(
        "--celsius", type=float, default=control.DEFAULT_CELSIUS
    )
    parser.add_argument(
        "--spike-threshold-mv",
        type=float,
        default=control.DEFAULT_SPIKE_THRESHOLD_MV,
    )
    parser.add_argument(
        "--ca-event-cai-mm",
        type=float,
        default=control.DEFAULT_CA_EVENT_CAI_MM,
    )
    parser.add_argument(
        "--ca-event-voltage-mv",
        type=float,
        default=control.DEFAULT_CA_EVENT_VOLTAGE_MV,
    )
    dense = parser.add_mutually_exclusive_group()
    dense.add_argument(
        "--store-dense-axon-events",
        dest="store_dense_axon_events",
        action="store_true",
    )
    dense.add_argument(
        "--no-store-dense-axon-events",
        dest="store_dense_axon_events",
        action="store_false",
    )
    parser.set_defaults(store_dense_axon_events=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    config = _config_from_arguments(arguments)
    manifest = generate_final_dataset(
        arguments.modeldb_root,
        arguments.output,
        config,
        workers=arguments.workers,
        resume=not arguments.no_resume,
    )
    print(json.dumps(
        {
            "event": "final_neuron_teacher_complete",
            "manifest": str(
                pathlib.Path(arguments.output).resolve() / "manifest.json"
            ),
            "teacher_contract_sha256": manifest[
                "teacher_contract_sha256"
            ],
            "completion_state": manifest["completion_state"],
            "completed_trials": manifest["completed_trials"],
            "shards": len(manifest["shards"]),
        }
    ), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
