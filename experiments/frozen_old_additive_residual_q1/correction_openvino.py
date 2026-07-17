from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import openvino as ov
from openvino import opset13 as ops


WEIGHT_SHAPES: dict[str, tuple[int, ...]] = {
    "ps.stem.layer_1.weight": (3, 3, 2, 8),
    "ps.stem.layer_1.bias": (8,),
    "ps.trunk.layer_1.layer_1.layer_1.weight": (3, 3, 8, 8),
    "ps.trunk.layer_1.layer_1.layer_1.bias": (8,),
    "ps.trunk.layer_1.layer_1.layer_3.weight": (3, 3, 8, 8),
    "ps.trunk.layer_1.layer_1.layer_3.bias": (8,),
    "ps.projection.layer_1.weight": (1, 1, 8, 2),
    "ps.projection.layer_1.bias": (2,),
    "ps.queue_encoder.layer_1.weight": (64, 42),
    "ps.queue_encoder.layer_1.bias": (64,),
    "ps.queue_encoder.layer_2.weight": (64, 64),
    "ps.queue_encoder.layer_2.bias": (64,),
    "ps.head.layer_1.weight": (256, 547),
    "ps.head.layer_1.bias": (256,),
    "ps.head.layer_2.weight": (64, 256),
    "ps.head.layer_2.bias": (64,),
    "ps.head.layer_3.weight": (1, 64),
    "ps.head.layer_3.bias": (1,),
}


def validate_weight_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    observed = set(arrays)
    expected = set(WEIGHT_SHAPES)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(f"correction weight keys differ: missing={missing}, extra={extra}")
    total = 0
    for name, shape in WEIGHT_SHAPES.items():
        value = np.asarray(arrays[name])
        if value.shape != shape:
            raise RuntimeError(f"{name} shape {value.shape} != {shape}")
        if value.dtype != np.float32:
            raise RuntimeError(f"{name} dtype {value.dtype} != float32")
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"non-finite correction weight: {name}")
        total += value.size
    if total != 165_051:
        raise RuntimeError(f"correction parameter count {total} != 165051")


class Q1CorrectionOpenVINOBuilder:
    """Exact OpenVINO graph for CompactCandidateQ(8, 1, 2)."""

    def __init__(self, weights_path: Path):
        with np.load(weights_path, allow_pickle=False) as archive:
            loaded = {name: np.asarray(archive[name]) for name in archive.files}
        validate_weight_arrays(loaded)
        self.weights = {
            name: np.ascontiguousarray(value, dtype=np.float32)
            for name, value in loaded.items()
        }

    def array(self, name: str) -> np.ndarray:
        return self.weights[name]

    @staticmethod
    def swish(value):
        return ops.multiply(value, ops.sigmoid(value))

    @staticmethod
    def dense(value, weight: np.ndarray, bias: np.ndarray):
        value = ops.matmul(value, ops.constant(weight), False, True)
        return ops.add(value, ops.constant(bias))

    def named_dense(self, value, prefix: str, activate: bool = True):
        value = self.dense(
            value,
            self.array(f"{prefix}.weight"),
            self.array(f"{prefix}.bias"),
        )
        return self.swish(value) if activate else value

    @staticmethod
    def conv(value, weight: np.ndarray, bias: np.ndarray, padding: int):
        # Lux stores KH,KW,in,out and performs true convolution by default.
        # OpenVINO stores out,in,KH,KW and performs cross-correlation.
        ov_weight = np.transpose(weight[::-1, ::-1, :, :], (3, 2, 0, 1))
        value = ops.convolution(
            value,
            ops.constant(ov_weight),
            [1, 1],
            [padding, padding],
            [padding, padding],
            [1, 1],
        )
        return ops.add(value, ops.constant(bias.reshape(1, -1, 1, 1)))

    def named_conv(self, value, prefix: str, padding: int):
        return self.conv(
            value,
            self.array(f"{prefix}.weight"),
            self.array(f"{prefix}.bias"),
            padding,
        )

    def build(self, batch_size: int | None = None) -> ov.Model:
        batch = -1 if batch_size is None else batch_size
        board = ops.parameter([batch, 1, 24, 10], np.float32, name="board")
        placement = ops.parameter([batch, 1, 24, 10], np.float32, name="placement")
        ren = ops.parameter([batch, 1], np.float32, name="ren")
        back_to_back = ops.parameter([batch, 1], np.float32, name="back_to_back")
        tspin = ops.parameter([batch, 1], np.float32, name="tspin")
        queue = ops.parameter([batch, 6, 7], np.float32, name="queue")

        value = ops.concat(
            [
                ops.subtract(ops.constant(np.float32(1.0)), board),
                ops.subtract(ops.constant(np.float32(1.0)), placement),
            ],
            1,
        )
        value = self.named_conv(value, "ps.stem.layer_1", 1)
        value = self.swish(value)
        residual = self.named_conv(
            value, "ps.trunk.layer_1.layer_1.layer_1", 1
        )
        residual = self.swish(residual)
        residual = self.named_conv(
            residual, "ps.trunk.layer_1.layer_1.layer_3", 1
        )
        value = self.swish(ops.add(value, residual))
        value = self.named_conv(value, "ps.projection.layer_1", 0)
        value = self.swish(value)
        # Julia reshape traverses height, width, channel. Transpose makes height
        # the fastest logical OpenVINO axis before flattening.
        value = ops.transpose(value, ops.constant(np.asarray([0, 1, 3, 2], np.int64)))
        board_features = ops.reshape(
            value, ops.constant(np.asarray([0, 480], np.int64)), True
        )

        queue_features = ops.reshape(
            queue, ops.constant(np.asarray([0, 42], np.int64)), True
        )
        queue_features = self.named_dense(
            queue_features, "ps.queue_encoder.layer_1"
        )
        queue_features = self.named_dense(
            queue_features, "ps.queue_encoder.layer_2"
        )
        combined = ops.concat(
            [
                board_features,
                queue_features,
                ops.divide(ren, ops.constant(np.float32(30.0))),
                back_to_back,
                tspin,
            ],
            1,
        )
        correction = self.named_dense(combined, "ps.head.layer_1")
        correction = self.named_dense(correction, "ps.head.layer_2")
        correction = self.named_dense(correction, "ps.head.layer_3", activate=False)
        correction.set_friendly_name("correction")
        return ov.Model(
            [correction],
            [board, placement, ren, back_to_back, tspin, queue],
            "frozen_old_additive_residual_q1_correction",
        )


def julia_inputs(
    board: np.ndarray,
    placement: np.ndarray,
    ren: np.ndarray,
    back_to_back: np.ndarray,
    tspin: np.ndarray,
    queue: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert Lux WHCB/feature-batch arrays to OpenVINO NCHW rows."""

    return {
        "board": np.ascontiguousarray(np.transpose(board, (3, 2, 0, 1)), dtype=np.float32),
        "placement": np.ascontiguousarray(
            np.transpose(placement, (3, 2, 0, 1)), dtype=np.float32
        ),
        "ren": np.ascontiguousarray(ren.T, dtype=np.float32),
        "back_to_back": np.ascontiguousarray(back_to_back.T, dtype=np.float32),
        "tspin": np.ascontiguousarray(tspin.T, dtype=np.float32),
        "queue": np.ascontiguousarray(np.transpose(queue, (2, 1, 0)), dtype=np.float32),
    }


def offline_row_inputs(arrays: Mapping[str, np.ndarray], slot: int) -> dict[str, np.ndarray]:
    count = int(np.asarray(arrays["action_counts"])[slot])
    board = np.asarray(arrays["boards"])[slot]
    queue = np.asarray(arrays["queues"])[slot]
    return {
        "board": np.ascontiguousarray(
            np.repeat(np.transpose(board, (2, 0, 1))[None, :, :, :], count, axis=0),
            dtype=np.float32,
        ),
        "placement": np.ascontiguousarray(
            np.transpose(np.asarray(arrays["placements"])[slot, :count], (0, 3, 1, 2)),
            dtype=np.float32,
        ),
        "ren": np.full((count, 1), np.asarray(arrays["ren"])[slot], dtype=np.float32),
        "back_to_back": np.full(
            (count, 1), np.asarray(arrays["back_to_back"])[slot], dtype=np.float32
        ),
        "tspin": np.ascontiguousarray(
            np.asarray(arrays["tspin"])[slot, :count, None], dtype=np.float32
        ),
        "queue": np.ascontiguousarray(
            np.repeat(np.transpose(queue, (1, 0))[None, :, :], count, axis=0),
            dtype=np.float32,
        ),
    }
