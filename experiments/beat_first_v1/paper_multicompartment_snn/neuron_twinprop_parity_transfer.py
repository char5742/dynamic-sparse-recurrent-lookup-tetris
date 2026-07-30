#!/usr/bin/env python3
"""Transfer a frozen TwinProp parity solution back to Hay/NEURON.

The input is an NPZ produced after the differentiable digital twin has been
frozen and only synaptic strength/location have been optimized.  This program
does not train, approximate, or read an analog classifier head.  It rebuilds
the optimized contacts on the public Hay ModelDB cell and classifies a trial
only by whether the soma emits at least one spike in the decision window.

The authors' implementation is not public.  The full-cell path reuses the
repository's read-only ModelDB 139653 loader and public TwinProp receptor
kinetics exactly.  The ablations are explicit reconstructions of the public
description and are always reported as such.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import pathlib
import tempfile
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

import neuron_hay_teacher as hay


SCHEMA = "hd_swsnn_twinprop.parity_neuron_transfer.v1"
INPUT_SCHEMA = "hd_swsnn_twinprop.parity_contact_export.v1"
VARIANTS = ("full", "passive", "no_nmda", "soma_only")

_DENDRITIC_CONDUCTANCES = (
    "gIhbar_Ih",
    "gSK_E2bar_SK_E2",
    "gCa_LVAstbar_Ca_LVAst",
    "gCa_HVAbar_Ca_HVA",
    "gSKv3_1bar_SKv3_1",
    "gNaTa_tbar_NaTa_t",
    "gImbar_Im",
)

_SOMA_MECHANISMS = (
    "pas",
    "Ca_LVAst",
    "Ca_HVA",
    "SKv3_1",
    "SK_E2",
    "K_Tst",
    "K_Pst",
    "Nap_Et2",
    "NaTa_t",
    "CaDynamics_E2",
    "Ih",
)

_SOMA_RANGE_PARAMETERS = (
    "g_pas",
    "e_pas",
    "gIhbar_Ih",
    "decay_CaDynamics_E2",
    "gamma_CaDynamics_E2",
    "gCa_LVAstbar_Ca_LVAst",
    "gCa_HVAbar_Ca_HVA",
    "gSKv3_1bar_SKv3_1",
    "gSK_E2bar_SK_E2",
    "gK_Tstbar_K_Tst",
    "gK_Pstbar_K_Pst",
    "gNap_Et2bar_Nap_Et2",
    "gNaTa_tbar_NaTa_t",
)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar_text(value: np.ndarray | Any) -> str:
    scalar = np.asarray(value).reshape(()).item()
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    return str(scalar)


def _scalar_int(value: np.ndarray | Any) -> int:
    return int(np.asarray(value).reshape(()).item())


def _require_array(
    data: np.lib.npyio.NpzFile,
    name: str,
    *,
    ndim: int,
) -> np.ndarray:
    if name not in data:
        raise ValueError(f"missing required NPZ field {name!r}")
    value = np.asarray(data[name])
    if value.ndim != ndim:
        raise ValueError(
            f"{name} must have {ndim} dimensions, received {value.shape}"
        )
    return value


def _load_export(path: pathlib.Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        schema = _scalar_text(data["schema"]) if "schema" in data else ""
        if schema != INPUT_SCHEMA:
            raise ValueError(
                f"expected input schema {INPUT_SCHEMA!r}, received {schema!r}"
            )
        axon_kind = _require_array(data, "axon_kind", ndim=1).astype(
            np.uint8, copy=False
        )
        contact_axon = _require_array(data, "contact_axon", ndim=1).astype(
            np.int32, copy=False
        )
        contact_kind = _require_array(data, "contact_kind", ndim=1).astype(
            np.uint8, copy=False
        )
        contact_segment = _require_array(
            data, "contact_segment", ndim=1
        ).astype(np.int32, copy=False)
        contact_strength = _require_array(
            data, "contact_strength", ndim=1
        ).astype(np.float32, copy=False)
        axon_events = _require_array(data, "axon_events", ndim=3).astype(
            np.uint8, copy=False
        )
        target = _require_array(data, "target", ndim=1).astype(
            np.uint8, copy=False
        )
        fields: dict[str, Any] = {
            "schema": schema,
            "model_name": _scalar_text(data["model_name"]),
            "task": _scalar_text(data["task"]),
            "dimension": _scalar_int(data["dimension"]),
            "variant": _scalar_text(data["variant"]),
            "sample_dt_ms": float(
                np.asarray(data["sample_dt_ms"]).reshape(()).item()
            ),
            "decision_first_step": _scalar_int(
                data["decision_first_step"]
            ),
            "contacts_per_axon": _scalar_int(data["contacts_per_axon"]),
            "axon_kind": axon_kind,
            "contact_axon": contact_axon,
            "contact_kind": contact_kind,
            "contact_segment": contact_segment,
            "contact_strength": contact_strength,
            "axon_events": axon_events,
            "target": target,
            "source_twin_sha256": _scalar_text(
                data["source_twin_sha256"]
            ),
            "source_parameter_sha256": _scalar_text(
                data["source_parameter_sha256"]
            ),
            "optimizer_result_sha256": _scalar_text(
                data["optimizer_result_sha256"]
            ),
        }
        if "modeldb_morphology_sha256" in data:
            fields["modeldb_morphology_sha256"] = _scalar_text(
                data["modeldb_morphology_sha256"]
            )
        if "contact_location_slot" in data:
            fields["contact_location_slot"] = _require_array(
                data, "contact_location_slot", ndim=1
            ).astype(np.int64, copy=False)
        return fields


def _validate_export(
    export: dict[str, Any],
    teacher: hay.HayTeacher,
    variant: str,
) -> dict[str, Any]:
    if export["model_name"] != hay.MODEL_NAME:
        raise ValueError("contact export has the wrong model family")
    if export["variant"] != variant:
        raise ValueError(
            "contact export variant differs from requested transfer variant"
        )
    axon_kind = export["axon_kind"]
    contact_axon = export["contact_axon"]
    contact_kind = export["contact_kind"]
    contact_segment = export["contact_segment"]
    contact_strength = export["contact_strength"]
    axon_events = export["axon_events"]
    target = export["target"]
    contacts_per_axon = export["contacts_per_axon"]
    axons = len(axon_kind)
    contacts = len(contact_axon)
    if axons < 2 or contacts < 1:
        raise ValueError("empty axon/contact export")
    if not (
        len(contact_kind)
        == len(contact_segment)
        == len(contact_strength)
        == contacts
    ):
        raise ValueError("ragged contact arrays differ in length")
    if axon_events.shape[0] != axons:
        raise ValueError("axon event axis differs from axon_kind")
    if axon_events.shape[2] != len(target):
        raise ValueError("trial target count differs from axon events")
    if export["decision_first_step"] < 1:
        raise ValueError("decision_first_step is one-based and must be >= 1")
    if export["decision_first_step"] > axon_events.shape[1]:
        raise ValueError("decision window begins after the trial")
    if not np.all(np.isin(axon_kind, (hay.EXCITATORY, hay.INHIBITORY))):
        raise ValueError("axon_kind violates fixed Dale classes")
    if np.any(contact_axon < 1) or np.any(contact_axon > axons):
        raise ValueError("contact_axon is out of bounds")
    if not np.array_equal(contact_kind, axon_kind[contact_axon - 1]):
        raise ValueError("contact kind violates the parent axon's Dale class")
    if np.any(~np.isfinite(contact_strength)):
        raise ValueError("contact strength contains a non-finite value")
    if np.any(contact_strength < 0.0) or np.any(contact_strength > 1.0):
        raise ValueError("contact strength is outside the public [0,1] bound")
    per_axon = np.bincount(contact_axon, minlength=axons + 1)[1:]
    if np.any(per_axon != contacts_per_axon):
        raise ValueError("each axon must have exactly contacts_per_axon")

    if variant != "soma_only":
        if np.any(contact_segment < 1) or np.any(
            contact_segment > len(teacher.segments)
        ):
            raise ValueError("contact segment is out of ModelDB bounds")
        eligible = set(int(index) for index in teacher.basal_apical_indices)
        if any(int(index) not in eligible for index in contact_segment):
            raise ValueError("full morphology contacts must target basal/apical")

        # The paper permits at most one E and one I contact per dendritic
        # micrometre.  When explicit slots are present, they are authoritative.
        location_slot = export.get("contact_location_slot")
        if location_slot is not None:
            if len(location_slot) != contacts:
                raise ValueError("contact_location_slot length mismatch")
            for kind in (hay.EXCITATORY, hay.INHIBITORY):
                selected = location_slot[contact_kind == kind]
                if len(np.unique(selected)) != len(selected):
                    raise ValueError(
                        "contact-location density exceeds one per kind/site"
                    )
        else:
            for segment_index in np.unique(contact_segment):
                record = teacher.segments[int(segment_index) - 1]
                capacity = max(1, int(np.floor(record.length_um)))
                selected = contact_segment == segment_index
                for kind in (hay.EXCITATORY, hay.INHIBITORY):
                    count = int(np.count_nonzero(selected & (contact_kind == kind)))
                    if count > capacity:
                        raise ValueError(
                            "segment contact count exceeds one-per-micrometre "
                            f"capacity: segment={segment_index}, kind={int(kind)}, "
                            f"count={count}, capacity={capacity}"
                        )
    morphology = export.get("modeldb_morphology_sha256")
    if morphology is not None and morphology != teacher.hashes[
        "morphology_sha256"
    ]:
        raise ValueError("contact export morphology hash mismatch")
    return {
        "axons": axons,
        "contacts": contacts,
        "trials": len(target),
        "time_steps": axon_events.shape[1],
        "contacts_per_axon": contacts_per_axon,
        "dale_fixed": True,
        "nonnegative_bounded_strength": True,
        "density_constraint_verified": True,
    }


def _zero_dendritic_voltage_gated_channels(
    teacher: hay.HayTeacher,
) -> None:
    """Keep cable morphology/passive current and soma channels intact."""

    for section in teacher.cell.basal:
        for name in _DENDRITIC_CONDUCTANCES:
            if hasattr(section, name):
                setattr(section, name, 0.0)
    for section in teacher.cell.apical:
        for name in _DENDRITIC_CONDUCTANCES:
            if hasattr(section, name):
                setattr(section, name, 0.0)


def _copy_soma_only_teacher(source: hay.HayTeacher) -> hay.HayTeacher:
    """Construct the public soma-only morphology ablation.

    The soma retains the original Hay somatic mechanisms and their fitted
    densities, while all dendritic cable is absent from this independently
    instantiated section.
    """

    original_section = source.soma
    original_segment = original_section(0.5)
    section = hay.h.Section(name="hd_swsnn_twinprop_soma_only")
    section.L = float(original_section.L)
    section.diam = float(original_segment.diam)
    section.nseg = 1
    section.Ra = float(original_section.Ra)
    section.cm = float(original_segment.cm)
    for mechanism in _SOMA_MECHANISMS:
        section.insert(mechanism)
    section.ena = float(original_segment.ena)
    section.ek = float(original_segment.ek)
    new_segment = section(0.5)
    for name in _SOMA_RANGE_PARAMETERS:
        if hasattr(original_segment, name) and hasattr(new_segment, name):
            setattr(new_segment, name, float(getattr(original_segment, name)))
    record = hay.SegmentInfo(
        index=1,
        section_name=section.name(),
        section_region="soma",
        x=0.5,
        distance_um=0.0,
        length_um=float(section.L),
        diameter_um=float(new_segment.diam),
        area_um2=float(hay.h.area(0.5, sec=section)),
        region_code=hay.REGION_SOMA,
        has_calcium=hasattr(new_segment, "cai"),
        segment=new_segment,
    )
    hashes = dict(source.hashes)
    hashes["soma_only_reconstruction"] = True
    return hay.HayTeacher(
        root=source.root,
        cell=source.cell,
        soma=section,
        segments=[record],
        basal_apical_indices=np.asarray([1], dtype=np.int32),
        diagnostic_indices=np.asarray([1], dtype=np.int32),
        hashes=hashes,
    )


@contextlib.contextmanager
def _nmda_variant(variant: str) -> Iterator[None]:
    original = hay.NMDA_MAX_NS
    if variant == "no_nmda":
        hay.NMDA_MAX_NS = 0.0
    try:
        yield
    finally:
        hay.NMDA_MAX_NS = original


def _simulation_config(export: dict[str, Any]) -> hay.TeacherConfig:
    duration = int(
        round(export["axon_events"].shape[1] * export["sample_dt_ms"])
    )
    config = hay.TeacherConfig(
        preset="twinprop_parity_transfer",
        train_trials=1,
        validation_trials=0,
        test_trials=0,
        duration_ms=duration,
        axons=len(export["axon_kind"]),
        contacts_per_axon=export["contacts_per_axon"],
        diagnostic_segments=32,
        shard_size=1,
        rate_hz=0.0,
        burst_probability=0.0,
        excitatory_fraction=float(
            np.mean(export["axon_kind"] == hay.EXCITATORY)
        ),
        minimum_strength=0.0,
        maximum_strength=1.0,
        seed=0,
        dt_ms=hay.DEFAULT_DT_MS,
        sample_dt_ms=export["sample_dt_ms"],
        v_init_mv=hay.DEFAULT_V_INIT_MV,
        celsius=hay.DEFAULT_CELSIUS,
        spike_threshold_mv=hay.DEFAULT_SPIKE_THRESHOLD_MV,
        ca_event_cai_mm=hay.DEFAULT_CA_EVENT_CAI_MM,
        ca_event_voltage_mv=hay.DEFAULT_CA_EVENT_VOLTAGE_MV,
        store_dense_events=False,
    )
    config.validate()
    return config


def _trial_protocol(
    export: dict[str, Any],
    trial: int,
    variant: str,
) -> dict[str, np.ndarray]:
    segment = export["contact_segment"]
    if variant == "soma_only":
        segment = np.ones_like(segment)
    return {
        "axon_kind": export["axon_kind"],
        "contact_axon": export["contact_axon"],
        "contact_kind": export["contact_kind"],
        "contact_segment": segment,
        "contact_strength": export["contact_strength"],
        "axon_events": export["axon_events"][:, :, trial],
    }


def run_transfer(
    *,
    modeldb_root: pathlib.Path,
    input_path: pathlib.Path,
    variant: str,
    trace_trials: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown transfer variant {variant!r}")
    export = _load_export(input_path)
    full_teacher = hay.instantiate_teacher(
        modeldb_root, diagnostic_segments=32
    )
    constraints = _validate_export(export, full_teacher, variant)
    teacher = full_teacher
    if variant == "passive":
        _zero_dendritic_voltage_gated_channels(teacher)
    elif variant == "soma_only":
        teacher = _copy_soma_only_teacher(full_teacher)
    config = _simulation_config(export)
    trials = len(export["target"])
    predictions = np.zeros(trials, dtype=np.uint8)
    spike_count = np.zeros(trials, dtype=np.int32)
    decision_first = export["decision_first_step"] - 1
    trace_count = min(max(0, trace_trials), trials)
    soma_voltage = np.empty(
        (config.time_steps, trace_count), dtype=np.float32
    )
    soma_spike = np.empty(
        (config.time_steps, trace_count), dtype=np.float32
    )
    nmda_region = np.empty(
        (len(hay.REGION_NAMES), config.time_steps, trace_count),
        dtype=np.float32,
    )
    dendritic_voltage = np.empty(
        (len(teacher.diagnostic_indices), config.time_steps, trace_count),
        dtype=np.float32,
    )
    dendritic_nmda = np.empty_like(dendritic_voltage)
    ca_event = np.empty_like(dendritic_voltage, dtype=np.uint8)
    all_abs_nmda_sum = 0.0
    all_nmda_values = 0
    with _nmda_variant(variant):
        for trial in range(trials):
            trajectory = hay._simulate_trial(
                teacher,
                config,
                _trial_protocol(export, trial, variant),
            )
            decision_spikes = trajectory["target_spike"][decision_first:]
            predictions[trial] = np.uint8(np.any(decision_spikes > 0.5))
            spike_count[trial] = int(np.count_nonzero(decision_spikes > 0.5))
            nmda = trajectory["target_nmda"]
            all_abs_nmda_sum += float(np.sum(np.abs(nmda), dtype=np.float64))
            all_nmda_values += int(nmda.size)
            if trial < trace_count:
                soma_voltage[:, trial] = trajectory["target_voltage"]
                soma_spike[:, trial] = trajectory["target_spike"]
                nmda_region[:, :, trial] = nmda
                dendritic_voltage[:, :, trial] = trajectory[
                    "target_compartment_voltage"
                ]
                dendritic_nmda[:, :, trial] = trajectory[
                    "target_compartment_nmda"
                ]
                ca_event[:, :, trial] = trajectory["target_ca_event"]
    target = export["target"].astype(np.uint8, copy=False)
    accuracy = float(np.mean(predictions == target))
    report = {
        "schema": SCHEMA,
        "model_name": hay.MODEL_NAME,
        "stage": "optimized_synapses_to_official_neuron",
        "task": export["task"],
        "dimension": export["dimension"],
        "variant": variant,
        "independently_retrained_variant": True,
        "input_path": str(input_path.resolve()),
        "input_sha256": _sha256_file(input_path),
        "source_twin_sha256": export["source_twin_sha256"],
        "source_parameter_sha256": export["source_parameter_sha256"],
        "optimizer_result_sha256": export["optimizer_result_sha256"],
        "modeldb": full_teacher.hashes,
        "constraints": constraints,
        "trials": trials,
        "decision_first_step_one_based": export["decision_first_step"],
        "decision_window_ms": (
            config.time_steps - decision_first
        )
        * config.sample_dt_ms,
        "accuracy": accuracy,
        "correct": int(np.count_nonzero(predictions == target)),
        "predictions": predictions.tolist(),
        "target": target.tolist(),
        "spike_count": spike_count.tolist(),
        "mean_abs_nmda_current": (
            all_abs_nmda_sum / all_nmda_values
            if all_nmda_values
            else 0.0
        ),
        "readout": "at_least_one_soma_spike_in_decision_window",
        "analog_readout_bypass": False,
        "simulation_dt_ms": config.dt_ms,
        "sample_dt_ms": config.sample_dt_ms,
        "receptor_kinetics": {
            "ampa_rise_ms": hay.AMPA_RISE_MS,
            "ampa_decay_ms": hay.AMPA_DECAY_MS,
            "ampa_max_ns": hay.AMPA_MAX_NS,
            "nmda_rise_ms": hay.NMDA_RISE_MS,
            "nmda_decay_ms": hay.NMDA_DECAY_MS,
            "nmda_max_ns": (
                0.0 if variant == "no_nmda" else hay.NMDA_MAX_NS
            ),
            "nmda_gamma_per_mv": hay.NMDA_GAMMA_PER_MV,
            "gabaa_rise_ms": hay.GABAA_RISE_MS,
            "gabaa_decay_ms": hay.GABAA_DECAY_MS,
            "gabaa_max_ns": hay.GABAA_MAX_NS,
        },
        "transfer_authority": "Hay ModelDB 139653 + NEURON",
        "julia_reduced_hay_is_control_only": True,
        "reconstruction_disclosure": (
            "The unpublished TwinProp author code is unavailable. Full-cell "
            "transfer uses the public Hay ModelDB mechanism and published "
            "receptor constants. Ablations are explicit reconstructions of "
            "the public methods description."
        ),
    }
    traces = {
        "soma_voltage": soma_voltage,
        "soma_spike": soma_spike,
        "nmda_region": nmda_region,
        "dendritic_voltage": dendritic_voltage,
        "dendritic_nmda": dendritic_nmda,
        "ca_event": ca_event,
        "prediction": predictions,
        "target": target,
        "diagnostic_segment": teacher.diagnostic_indices,
    }
    return report, traces


def _atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        temporary = pathlib.Path(stream.name)
    temporary.replace(path)


def _atomic_npz(path: pathlib.Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".npz",
        delete=False,
    ) as stream:
        temporary = pathlib.Path(stream.name)
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modeldb-root",
        default="/mnt/c/tmp/hay_modeldb_139653",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace-output")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--trace-trials", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    input_path = pathlib.Path(arguments.input).resolve()
    output_path = pathlib.Path(arguments.output).resolve()
    trace_path = (
        pathlib.Path(arguments.trace_output).resolve()
        if arguments.trace_output
        else output_path.with_suffix(".traces.npz")
    )
    report, traces = run_transfer(
        modeldb_root=pathlib.Path(arguments.modeldb_root).resolve(),
        input_path=input_path,
        variant=arguments.variant,
        trace_trials=arguments.trace_trials,
    )
    _atomic_npz(trace_path, traces)
    report["trace_path"] = str(trace_path)
    report["trace_sha256"] = _sha256_file(trace_path)
    _atomic_json(output_path, report)
    print(
        json.dumps(
            {
                "event": "twinprop_parity_neuron_transfer_complete",
                "output": str(output_path),
                "variant": arguments.variant,
                "accuracy": report["accuracy"],
                "readout": report["readout"],
                "input_sha256": report["input_sha256"],
                "trace_sha256": report["trace_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
