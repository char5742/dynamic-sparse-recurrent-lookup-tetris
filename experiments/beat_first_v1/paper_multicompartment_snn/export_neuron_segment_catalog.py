#!/usr/bin/env python3
"""Export the ordered Hay ModelDB segment/capacity catalog for TwinProp.

The JSON order is exactly the one used by ``neuron_hay_teacher.py`` and the
642-segment ELM input adapter.  It is generated from the read-only ModelDB
checkout rather than from the reduced Julia control cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import tempfile
from collections.abc import Sequence
from typing import Any

import neuron_hay_teacher as hay


SCHEMA = "hd_swsnn_twinprop.neuron_segment_catalog.v1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def build_catalog(modeldb_root: pathlib.Path) -> dict[str, Any]:
    teacher = hay.instantiate_teacher(modeldb_root, diagnostic_segments=8)
    eligible = set(int(index) for index in teacher.basal_apical_indices)
    records = []
    global_slot = 1
    for segment in teacher.segments:
        allowed = segment.index in eligible
        capacity = (
            max(1, int(segment.length_um // 1.0)) if allowed else 0
        )
        slot_first = global_slot if capacity else 0
        slot_last = global_slot + capacity - 1 if capacity else 0
        if capacity:
            global_slot = slot_last + 1
        record = segment.public_record()
        record.update(
            {
                "allowed_for_synapse": allowed,
                "one_micron_slots_per_kind": capacity,
                "slot_first_one_based": slot_first,
                "slot_last_one_based": slot_last,
            }
        )
        records.append(record)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "model_name": hay.MODEL_NAME,
        "modeldb": teacher.hashes,
        "segment_count": len(records),
        "eligible_segment_count": len(eligible),
        "one_micron_slots_per_kind": global_slot - 1,
        "segments": records,
        "ordering": (
            "cell.all section order, then NEURON segment order; one-based"
        ),
    }
    payload["catalog_sha256"] = hashlib.sha256(
        _canonical_json(payload)
    ).hexdigest()
    return payload


def _atomic_write(path: pathlib.Path, value: dict[str, Any]) -> None:
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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modeldb-root",
        default="/mnt/c/tmp/hay_modeldb_139653",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    output = pathlib.Path(arguments.output).resolve()
    catalog = build_catalog(pathlib.Path(arguments.modeldb_root).resolve())
    _atomic_write(output, catalog)
    print(
        json.dumps(
            {
                "event": "neuron_segment_catalog_complete",
                "output": str(output),
                "segment_count": catalog["segment_count"],
                "eligible_segment_count": catalog[
                    "eligible_segment_count"
                ],
                "one_micron_slots_per_kind": catalog[
                    "one_micron_slots_per_kind"
                ],
                "catalog_sha256": catalog["catalog_sha256"],
                "morphology_sha256": catalog["modeldb"][
                    "morphology_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
