from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import openvino as ov


class WeightedLegacyOpenVINOInference:
    """Historical fixed-16 + actual-length CPU tail bound to an explicit NPZ."""

    def __init__(self, repository: str, weights_path: str, device: str = "NPU", batch_size: int = 16):
        repository_path = Path(repository).resolve()
        weights = Path(weights_path).resolve()
        if not weights.is_file():
            raise RuntimeError(f"missing explicit weights: {weights}")
        sys.path.insert(0, str(repository_path / "tools"))
        from legacy_openvino import LegacyOpenVINOBuilder

        self.weights_path = str(weights)
        self.device = device
        self.batch_size = int(batch_size)
        self.core = ov.Core()
        if device not in self.core.available_devices:
            raise RuntimeError(f"required OpenVINO device {device} unavailable")
        builder = LegacyOpenVINOBuilder(weights)
        self.accelerator = self.core.compile_model(
            builder.build(batch_size=self.batch_size), device
        )
        self.tail = self.core.compile_model(builder.build(batch_size=None), "CPU")

    @staticmethod
    def _inputs(board, placement, ren, back_to_back, tspin, queue):
        return {
            "board": np.ascontiguousarray(np.transpose(np.asarray(board), (3, 2, 0, 1))),
            "placement": np.ascontiguousarray(
                np.transpose(np.asarray(placement), (3, 2, 0, 1))
            ),
            "ren": np.ascontiguousarray(np.asarray(ren).T),
            "back_to_back": np.ascontiguousarray(np.asarray(back_to_back).T),
            "tspin": np.ascontiguousarray(np.asarray(tspin).T),
            "queue": np.ascontiguousarray(np.transpose(np.asarray(queue), (2, 1, 0))),
        }

    def predict(self, board, placement, ren, back_to_back, tspin, queue):
        inputs = self._inputs(board, placement, ren, back_to_back, tspin, queue)
        count = len(inputs["board"])
        chunks = []
        for start in range(0, count, self.batch_size):
            stop = min(start + self.batch_size, count)
            batch = {name: value[start:stop] for name, value in inputs.items()}
            compiled = self.accelerator if stop - start == self.batch_size else self.tail
            chunks.append(np.asarray(compiled(batch)[0]).reshape(-1))
        return np.concatenate(chunks).astype(np.float32, copy=False)
