"""Build the fixed-batch NPU and dynamic-tail CPU Learned HashStem IRs.

This file is intentionally inert when imported. It consumes a fresh NPZ export
from the CPU master and never reads or overwrites a training checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import opset13 as ops


SCHEMA = "learned-hashstem-v1-conv-dw-pw-pool-1039x214-relu"
BATCH = 16
INPUTS = 559
DENSE_INPUTS = 1_039
LEARNED_OUTPUTS = 214
OUTPUTS = 256
PARAMETERS = 223_504
MACS = 433_546


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalization_sha256(weights: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in ("input_mean", "input_inv_std"):
        digest.update(np.asarray(weights[name], dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def load_weights(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        weights = {name: np.asarray(archive[name], dtype=np.float32) for name in archive.files}
    expected = {
        "conv3": (16, 2, 3, 3),
        "conv3_bias": (16,),
        "depthwise5x1": (16, 5),
        "depthwise_bias": (16,),
        "pointwise": (32, 16),
        "pointwise_bias": (32,),
        "dense": (LEARNED_OUTPUTS, DENSE_INPUTS),
        "dense_bias": (LEARNED_OUTPUTS,),
        "input_mean": (INPUTS,),
        "input_inv_std": (INPUTS,),
    }
    if set(weights) != set(expected):
        raise ValueError(f"NPZ keys must be exactly {sorted(expected)}")
    for name, shape in expected.items():
        if weights[name].shape != shape:
            raise ValueError(f"{name} shape {weights[name].shape} != {shape}")
        if not np.all(np.isfinite(weights[name])):
            raise ValueError(f"{name} contains non-finite values")
    if not np.all(weights["input_inv_std"] > 0):
        raise ValueError("input_inv_std must be strictly positive")
    return weights


def i64(values) -> object:
    return ops.constant(np.asarray(values, dtype=np.int64))


def feature_slice(value, start: int, stop: int):
    return ops.slice(value, i64([start]), i64([stop]), i64([1]), i64([1]))


def build_model(weights: dict[str, np.ndarray], batch: int | None) -> ov.Model:
    shape = [batch if batch is not None else -1, INPUTS]
    packed = ops.parameter(shape, np.float32, "packed_candidate")
    normalized = ops.multiply(
        ops.subtract(packed, ops.constant(weights["input_mean"])),
        ops.constant(weights["input_inv_std"]),
    )

    # Packed cells vary row-fastest inside each column. Reshape to [B,2,10,24]
    # first, then transpose into OpenVINO NCHW [B,2,24,10].
    board_flat = feature_slice(normalized, 0, 480)
    board_column_major = ops.reshape(board_flat, i64([0, 2, 10, 24]), True)
    board = ops.transpose(board_column_major, i64([0, 1, 3, 2]))

    value = ops.convolution(
        board,
        ops.constant(weights["conv3"]),
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
    )
    value = ops.add(value, ops.constant(weights["conv3_bias"].reshape(1, 16, 1, 1)))
    value = ops.relu(value)
    depthwise_weight = weights["depthwise5x1"].reshape(16, 1, 1, 5, 1)
    value = ops.group_convolution(
        value,
        ops.constant(depthwise_weight),
        [1, 1],
        [2, 0],
        [2, 0],
        [1, 1],
    )
    value = ops.add(value, ops.constant(weights["depthwise_bias"].reshape(1, 16, 1, 1)))
    value = ops.relu(value)
    pointwise_weight = weights["pointwise"].reshape(32, 16, 1, 1)
    value = ops.convolution(
        value,
        ops.constant(pointwise_weight),
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
    )
    value = ops.add(value, ops.constant(weights["pointwise_bias"].reshape(1, 32, 1, 1)))
    value = ops.relu(value)

    pooled = ops.avg_pool(
        value,
        strides=[4, 2],
        pads_begin=[0, 0],
        pads_end=[0, 0],
        kernel_shape=[4, 2],
        exclude_pad=True,
        rounding_type="floor",
    )
    pooled = ops.reshape(pooled, i64([0, 960]), True)

    next_hold_normalized = feature_slice(normalized, 480, 522)
    auxiliary = feature_slice(normalized, 522, 559)
    dense_input = ops.concat([pooled, auxiliary, next_hold_normalized], 1)
    learned = ops.add(
        ops.matmul(dense_input, ops.constant(weights["dense"]), False, True),
        ops.constant(weights["dense_bias"]),
    )
    next_hold_raw = feature_slice(packed, 480, 522)
    output = ops.concat([learned, next_hold_raw], 1)
    result = ops.result(output, "hashstem_output")
    model_name = f"learned_hashstem_v1_b{batch}" if batch is not None else "learned_hashstem_v1_dynamic"
    return ov.Model([result], [packed], model_name)


def serialize_fresh(model: ov.Model, xml_path: Path) -> tuple[Path, Path]:
    bin_path = xml_path.with_suffix(".bin")
    if xml_path.exists() or bin_path.exists():
        raise FileExistsError(f"refusing to overwrite {xml_path} or {bin_path}")
    ov.save_model(model, xml_path, compress_to_fp16=False)
    return xml_path, bin_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--master-version", type=int, required=True)
    parser.add_argument("--snapshot-version", type=int, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--normalization-sha256", required=True)
    args = parser.parse_args()

    if args.master_version < 0 or args.snapshot_version <= 0:
        raise ValueError("master version must be nonnegative and snapshot version positive")
    if len(args.normalization_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in args.normalization_sha256
    ):
        raise ValueError("normalization SHA-256 must contain exactly 64 hexadecimal digits")
    if args.output_directory.exists():
        raise FileExistsError("output directory must be fresh")
    weights = load_weights(args.weights)
    observed_normalization_sha256 = normalization_sha256(weights)
    if observed_normalization_sha256 != args.normalization_sha256.lower():
        raise ValueError("normalization SHA-256 does not match NPZ arrays")
    args.output_directory.mkdir(parents=True)
    fixed_xml, fixed_bin = serialize_fresh(
        build_model(weights, BATCH), args.output_directory / "hashstem_b16.xml"
    )
    dynamic_xml, dynamic_bin = serialize_fresh(
        build_model(weights, None), args.output_directory / "hashstem_dynamic_cpu.xml"
    )
    metadata = {
        "schema": SCHEMA,
        "model_id": args.model_id,
        "master_version": args.master_version,
        "snapshot_version": args.snapshot_version,
        "weights_sha256": sha256_file(args.weights),
        "normalization_sha256": observed_normalization_sha256,
        "openvino_version": ov.__version__,
        "device": "NPU",
        "fixed_batch": BATCH,
        "input_features": INPUTS,
        "output_features": OUTPUTS,
        "parameter_count": PARAMETERS,
        "macs_per_candidate": MACS,
        "fixed_xml_sha256": sha256_file(fixed_xml),
        "fixed_bin_sha256": sha256_file(fixed_bin),
        "xml_sha256": sha256_file(fixed_xml),
        "bin_sha256": sha256_file(fixed_bin),
        "dynamic_xml_sha256": sha256_file(dynamic_xml),
        "dynamic_bin_sha256": sha256_file(dynamic_bin),
        "npu_compiled_or_executed": False,
        "adoption_authorized": False,
        "exported_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_directory / "snapshot_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
