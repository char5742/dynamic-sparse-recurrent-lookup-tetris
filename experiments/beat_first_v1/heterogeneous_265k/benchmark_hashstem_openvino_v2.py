"""CLI for the fair Learned HashStem OpenVINO CPU/NPU v2 control.

This source does nothing when imported.  It writes only a fresh result path and
never mutates an IR, snapshot, witness, checkpoint, or the frozen v1 closure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hashstem_openvino_control_v2 import benchmark_control, load_property_map


DEFAULT_CPU_PROPERTIES = {
    "PERFORMANCE_HINT": "LATENCY",
    "NUM_STREAMS": "1",
}
DEFAULT_NPU_PROPERTIES = {
    "PERFORMANCE_HINT": "LATENCY",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-ir", type=Path, required=True)
    parser.add_argument("--dynamic-cpu-ir", type=Path, required=True)
    parser.add_argument("--snapshot-metadata", type=Path, required=True)
    parser.add_argument("--witness", type=Path, required=True)
    parser.add_argument("--witness-provenance", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--witness-generator-source", type=Path, required=True)
    parser.add_argument("--witness-generator-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sparse-k", type=int, choices=(64, 128, 256), required=True)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--cpu-properties-json", type=Path)
    parser.add_argument("--npu-properties-json", type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError("refusing to overwrite v2 benchmark output")
    cpu_properties, cpu_properties_sha256 = load_property_map(
        args.cpu_properties_json, DEFAULT_CPU_PROPERTIES
    )
    npu_properties, npu_properties_sha256 = load_property_map(
        args.npu_properties_json, DEFAULT_NPU_PROPERTIES
    )
    result = benchmark_control(
        fixed_ir=args.fixed_ir,
        dynamic_cpu_ir=args.dynamic_cpu_ir,
        snapshot_metadata=args.snapshot_metadata,
        witness_path=args.witness,
        witness_provenance=args.witness_provenance,
        dataset_manifest_path=args.dataset_manifest,
        expected_dataset_manifest_sha256=args.dataset_manifest_sha256,
        witness_generator_source=args.witness_generator_source,
        expected_witness_generator_source_sha256=args.witness_generator_source_sha256,
        sparse_k=args.sparse_k,
        warmups=args.warmups,
        repeats=args.repeats,
        cpu_properties=cpu_properties,
        npu_properties=npu_properties,
    )
    result["compile_property_sources"] = {
        "cpu_properties_file_sha256": cpu_properties_sha256,
        "npu_properties_file_sha256": npu_properties_sha256,
        "cpu_used_built_in_defaults": args.cpu_properties_json is None,
        "npu_used_built_in_defaults": args.npu_properties_json is None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if result["status"] != "COMPONENT_PASS_PENDING_INTEGRATED_SPARSE_K_GATE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
