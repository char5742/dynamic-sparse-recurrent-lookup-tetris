from __future__ import annotations

from pathlib import Path

import numpy as np
import openvino as ov
from openvino import opset13 as ops


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "artifacts" / "legacy_openvino" / "legacy_1313_weights.npz"


class LegacyOpenVINOBuilder:
    def __init__(self, weights_path: Path = DEFAULT_WEIGHTS):
        self.weights = np.load(weights_path)
        self.debug_nodes = {}

    def array(self, name: str) -> np.ndarray:
        return np.asarray(self.weights[name], dtype=np.float32)

    def constant(self, name: str):
        return ops.constant(self.array(name))

    @staticmethod
    def swish(value):
        return ops.multiply(value, ops.sigmoid(value))

    @staticmethod
    def dense(value, weight: np.ndarray, bias: np.ndarray):
        result = ops.matmul(value, ops.constant(weight), False, True)
        return ops.add(result, ops.constant(bias))

    def named_dense(self, value, prefix: str, activation=None):
        value = self.dense(
            value, self.array(f"{prefix}.weight"), self.array(f"{prefix}.bias")
        )
        return activation(value) if activation is not None else value

    @staticmethod
    def conv(value, weight: np.ndarray, bias: np.ndarray, padding: int):
        # Lux stores convolution weights as KH, KW, input channels, output channels.
        # Historical Lux Conv used true convolution (cross_correlation=false),
        # while OpenVINO Convolution performs cross-correlation.
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

    def batch_norm(self, value, parameter_prefix: str, state_prefix: str, epsilon: float):
        scale = self.array(f"{parameter_prefix}.scale").reshape(1, -1, 1, 1)
        bias = self.array(f"{parameter_prefix}.bias").reshape(1, -1, 1, 1)
        mean = self.array(f"{state_prefix}.running_mean").reshape(1, -1, 1, 1)
        variance = self.array(f"{state_prefix}.running_var").reshape(1, -1, 1, 1)
        factor = scale / np.sqrt(variance + np.float32(epsilon))
        offset = bias - mean * factor
        return ops.add(ops.multiply(value, ops.constant(factor)), ops.constant(offset))

    def layer_norm(self, value, prefix: str, epsilon: float = 1.0e-5):
        # The historical LayerNorm used dims=Colon(), which normalizes the whole
        # features x tokens x candidate-batch tensor (including the batch axis).
        axes = ops.constant(np.asarray([0, 1, 2], dtype=np.int64))
        mean = ops.reduce_mean(value, axes, True)
        centered = ops.subtract(value, mean)
        variance = ops.reduce_mean(ops.multiply(centered, centered), axes, True)
        normalized = ops.divide(
            centered, ops.sqrt(ops.add(variance, ops.constant(np.float32(epsilon))))
        )
        # Julia tensor order is features, tokens, batch; OpenVINO is batch, tokens, features.
        scale = np.transpose(self.array(f"{prefix}.scale")[:, :, 0], (1, 0))[None, :, :]
        bias = np.transpose(self.array(f"{prefix}.bias")[:, :, 0], (1, 0))[None, :, :]
        return ops.add(ops.multiply(normalized, ops.constant(scale)), ops.constant(bias))

    def residual_block(self, value, block_index: int):
        chain_index = 2 * block_index - 1
        prefix = f"ps.board_net.resblocks.layer_{chain_index}"
        state_prefix = f"st.board_net.resblocks.layer_{chain_index}"
        residual = self.named_conv(value, f"{prefix}.layer_1", 1)
        residual = self.batch_norm(
            residual, f"{prefix}.layer_2", f"{state_prefix}.layer_2", 1.0e-5
        )
        residual = self.swish(residual)
        residual = self.named_conv(residual, f"{prefix}.layer_3", 1)
        residual = self.batch_norm(
            residual, f"{prefix}.layer_4", f"{state_prefix}.layer_4", 1.0e-5
        )
        gate = ops.reduce_mean(residual, ops.constant(np.asarray([2, 3], np.int64)), True)
        gate = self.named_conv(gate, f"{prefix}.layer_5.layer_2", 0)
        gate = self.swish(gate)
        gate = self.named_conv(gate, f"{prefix}.layer_5.layer_3", 0)
        residual = ops.multiply(residual, ops.sigmoid(gate))
        return self.swish(ops.add(value, residual))

    def board_network(self, board, placement):
        board = ops.subtract(ops.constant(np.float32(1.0)), board)
        placement = ops.subtract(ops.constant(np.float32(1.0)), placement)
        value = ops.concat([board, placement], 1)
        value = self.named_conv(value, "ps.board_net.conv1", 1)
        self.debug_nodes["board_conv1"] = value
        value = self.batch_norm(value, "ps.board_net.norm1", "st.board_net.norm1", 1.0e-6)
        value = self.swish(value)
        self.debug_nodes["board_norm1"] = value
        for block_index in range(1, 17):
            value = self.residual_block(value, block_index)
            if block_index == 1:
                self.debug_nodes["board_residual1"] = value
        value = self.named_conv(value, "ps.board_net.conv2", 1)
        value = self.batch_norm(value, "ps.board_net.norm2", "st.board_net.norm2", 1.0e-6)
        value = self.swish(value)
        # B,C,W,H makes H the fastest-changing dimension, matching Julia reshape.
        value = ops.transpose(value, ops.constant(np.asarray([0, 1, 3, 2], np.int64)))
        return ops.reshape(value, ops.constant(np.asarray([0, 240], np.int64)), True)

    def queue_mixer(self, queue):
        value = self.named_dense(queue, "ps.mino_list_encoder")
        self.debug_nodes["mino_features"] = value
        value = self.named_dense(value, "ps.attention.embedding")
        self.debug_nodes["queue_embedding"] = value
        positional = np.transpose(
            self.array("st.attention.positional_encoding.pos_enc")[:, :, 0], (1, 0)
        )[None, :, :]
        value = ops.add(value, ops.constant(positional))
        self.debug_nodes["queue_position"] = value
        for block_index in range(1, 13):
            prefix = f"ps.attention.blocks.layer_{block_index}"
            normalized = self.layer_norm(value, f"{prefix}.layer_1")
            if block_index % 2:
                if block_index == 1:
                    self.debug_nodes["queue_norm"] = normalized
                mixed = ops.transpose(
                    normalized, ops.constant(np.asarray([0, 2, 1], np.int64))
                )
                if block_index == 1:
                    self.debug_nodes["queue_token_order"] = mixed
                mixed = self.named_dense(mixed, f"{prefix}.layer_3")
                if block_index == 1:
                    self.debug_nodes["queue_token_dense1"] = mixed
                mixed = ops.gelu(mixed, "TANH")
                if block_index == 1:
                    self.debug_nodes["queue_token_gelu"] = mixed
                mixed = self.named_dense(mixed, f"{prefix}.layer_5")
                if block_index == 1:
                    self.debug_nodes["queue_token_dense2"] = mixed
                mixed = ops.transpose(
                    mixed, ops.constant(np.asarray([0, 2, 1], np.int64))
                )
                if block_index == 1:
                    self.debug_nodes["queue_token_result"] = mixed
            else:
                mixed = self.named_dense(normalized, f"{prefix}.layer_2")
                mixed = ops.gelu(mixed, "TANH")
                mixed = self.named_dense(mixed, f"{prefix}.layer_4")
            value = ops.add(value, mixed)
            if block_index == 1:
                self.debug_nodes["queue_block1"] = value
        value = ops.reshape(value, ops.constant(np.asarray([0, 768], np.int64)), True)
        return self.named_dense(value, "ps.attention.output.layer_2")

    def build(
        self,
        batch_size: int | None = None,
        debug: bool = False,
        output_kind: str = "score",
    ) -> ov.Model:
        batch = -1 if batch_size is None else batch_size
        board = ops.parameter([batch, 1, 24, 10], np.float32, name="board")
        placement = ops.parameter([batch, 1, 24, 10], np.float32, name="placement")
        ren = ops.parameter([batch, 1], np.float32, name="ren")
        back_to_back = ops.parameter([batch, 1], np.float32, name="back_to_back")
        tspin = ops.parameter([batch, 1], np.float32, name="tspin")
        queue = ops.parameter([batch, 6, 7], np.float32, name="queue")
        board_features = self.board_network(board, placement)
        queue_features = self.queue_mixer(queue)
        combined = ops.concat(
            [
                board_features,
                ops.divide(ren, ops.constant(np.float32(30.0))),
                back_to_back,
                tspin,
                queue_features,
            ],
            1,
        )
        score = self.named_dense(combined, "ps.score_net.layer_1", self.swish)
        score = self.named_dense(score, "ps.score_net.layer_2", self.swish)
        score = self.named_dense(score, "ps.score_net.layer_3")
        score.set_friendly_name("score")
        outputs = (
            [
                score,
                board_features,
                queue_features,
                combined,
                self.debug_nodes["board_conv1"],
                self.debug_nodes["board_norm1"],
                self.debug_nodes["board_residual1"],
                self.debug_nodes["mino_features"],
                self.debug_nodes["queue_embedding"],
                self.debug_nodes["queue_position"],
                self.debug_nodes["queue_norm"],
                self.debug_nodes["queue_token_order"],
                self.debug_nodes["queue_token_dense1"],
                self.debug_nodes["queue_token_gelu"],
                self.debug_nodes["queue_token_dense2"],
                self.debug_nodes["queue_token_result"],
                self.debug_nodes["queue_block1"],
            ]
            if debug
            else [score]
        )
        if output_kind == "features":
            outputs = [combined]
        elif output_kind != "score":
            raise ValueError(f"unknown output kind: {output_kind}")
        return ov.Model(
            outputs, [board, placement, ren, back_to_back, tspin, queue], "legacy_1313"
        )


def reference_inputs(reference: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    return {
        "board": np.transpose(reference["board"], (3, 2, 0, 1)),
        "placement": np.transpose(reference["placement"], (3, 2, 0, 1)),
        "ren": reference["ren"].T,
        "back_to_back": reference["back_to_back"].T,
        "tspin": reference["tspin"].T,
        "queue": np.transpose(reference["queue"], (2, 1, 0)),
    }


class LegacyOpenVINOInference:
    """Historical batch-16 policy with a static accelerator and dynamic CPU tail."""

    def __init__(self, device: str = "NPU", batch_size: int = 16):
        self.device = device
        self.batch_size = batch_size
        self.core = ov.Core()
        builder = LegacyOpenVINOBuilder()
        self.accelerator = self.core.compile_model(
            builder.build(batch_size=batch_size), device
        )
        # The legacy GPU implementation normalized each final short batch only
        # over its actual candidates. Dynamic CPU inference preserves that quirk.
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
        candidate_count = len(inputs["board"])
        results = []
        for start in range(0, candidate_count, self.batch_size):
            stop = min(start + self.batch_size, candidate_count)
            batch = {name: value[start:stop] for name, value in inputs.items()}
            compiled = self.accelerator if stop - start == self.batch_size else self.tail
            results.append(np.asarray(compiled(batch)[0]).reshape(-1))
        return np.concatenate(results).astype(np.float32, copy=False)

    def predict_overlap_tail(self, board, placement, ren, back_to_back, tspin, queue):
        """Overlap the unchanged dynamic CPU tail with static accelerator chunks.

        This is an opt-in scheduling optimization for the bounded teacher-data
        generator.  Chunk boundaries, devices, actual tail shape, output order,
        and FP32 conversion are identical to :meth:`predict`.  The method is
        intentionally single-caller: the dataset generator enters embedded
        CPython only from its main Julia thread.
        """
        inputs = self._inputs(board, placement, ren, back_to_back, tspin, queue)
        candidate_count = len(inputs["board"])
        if candidate_count == 0:
            return np.empty((0,), dtype=np.float32)

        tail_size = candidate_count % self.batch_size
        full_stop = candidate_count - tail_size
        results = []

        # With no full accelerator chunk there is nothing to overlap.  Keep
        # the original CompiledModel call path exactly as-is.
        if full_stop == 0:
            return np.asarray(self.tail(inputs)[0]).reshape(-1).astype(
                np.float32, copy=False
            )

        tail_request = None
        if tail_size:
            # Create lazily so ordinary serial users pay no request-allocation
            # cost and retain the existing construction/predict path.
            tail_request = getattr(self, "_overlap_tail_request", None)
            if tail_request is None:
                tail_request = self.tail.create_infer_request()
                self._overlap_tail_request = tail_request
            tail_batch = {
                name: value[full_stop:candidate_count]
                for name, value in inputs.items()
            }
            # share_inputs=False copies the actual-size dynamic inputs before
            # returning, so their lifetime cannot race the accelerator calls.
            tail_request.start_async(tail_batch, share_inputs=False)

        try:
            for start in range(0, full_stop, self.batch_size):
                stop = start + self.batch_size
                batch = {name: value[start:stop] for name, value in inputs.items()}
                results.append(
                    np.asarray(self.accelerator(batch)[0]).reshape(-1)
                )
        finally:
            # Do not leave the reusable dynamic request busy if an accelerator
            # call raises; the original accelerator exception still propagates.
            if tail_request is not None:
                tail_request.wait()

        if tail_request is not None:
            # Copy because the request owns this buffer and reuses it on the
            # next variable-shape tail.
            results.append(
                np.asarray(tail_request.results[0]).reshape(-1).copy()
            )
        return np.concatenate(results).astype(np.float32, copy=False)


class LegacyOpenVINOFeatures(LegacyOpenVINOInference):
    """Frozen 249-wide representation used to fine-tune only the value head."""

    def __init__(self, device: str = "NPU", batch_size: int = 16):
        self.device = device
        self.batch_size = batch_size
        self.core = ov.Core()
        builder = LegacyOpenVINOBuilder()
        self.accelerator = self.core.compile_model(
            builder.build(batch_size=batch_size, output_kind="features"), device
        )
        self.tail = self.core.compile_model(
            builder.build(batch_size=None, output_kind="features"), "CPU"
        )

    def predict(self, board, placement, ren, back_to_back, tspin, queue):
        inputs = self._inputs(board, placement, ren, back_to_back, tspin, queue)
        candidate_count = len(inputs["board"])
        results = []
        for start in range(0, candidate_count, self.batch_size):
            stop = min(start + self.batch_size, candidate_count)
            batch = {name: value[start:stop] for name, value in inputs.items()}
            compiled = self.accelerator if stop - start == self.batch_size else self.tail
            results.append(np.asarray(compiled(batch)[0]))
        return np.concatenate(results, axis=0).astype(np.float32, copy=False)
