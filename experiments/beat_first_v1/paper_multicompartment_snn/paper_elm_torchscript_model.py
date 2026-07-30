"""TorchScript-specialized M-variable Paper-ELM recurrence."""

from typing import List, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ScriptablePaperELMTwin(nn.Module):
    memory_size: int
    hidden_size: int

    def __init__(self, bridge: Mapping[str, np.ndarray]) -> None:
        super().__init__()
        self.memory_size = int(
            np.asarray(bridge["parameter_memory_bias"]).size
        )
        self.hidden_size = int(
            np.asarray(bridge["parameter_input_bias"]).size
        )
        if self.hidden_size != 2 * self.memory_size:
            raise ValueError("hidden size must be exactly 2M")
        for name in (
            "proto_w_s",
            "input_weight",
            "input_bias",
            "memory_weight",
            "memory_bias",
            "output_weight",
            "output_bias",
        ):
            setattr(
                self,
                name,
                nn.Parameter(
                    torch.from_numpy(
                        np.asarray(
                            bridge[f"parameter_{name}"],
                            np.float32,
                        ).copy()
                    )
                ),
            )
        self.register_buffer(
            "route_indices",
            torch.from_numpy(
                np.asarray(
                    bridge["route_indices_zero_based"],
                    np.int64,
                ).copy()
            ),
        )
        for name in (
            "valid_indices_mask",
            "kappa_b",
            "kappa_m",
            "kappa_lambda",
            "nmda_mean",
            "nmda_scale",
        ):
            self.register_buffer(
                name,
                torch.from_numpy(
                    np.asarray(bridge[name], np.float32).copy()
                ),
            )
        self._assert_contract()

    def _assert_contract(self) -> None:
        shapes = (
            (tuple(self.proto_w_s.shape), (4_500,)),
            (
                tuple(self.input_weight.shape),
                (self.hidden_size, 45 + self.memory_size),
            ),
            (tuple(self.input_bias.shape), (self.hidden_size,)),
            (
                tuple(self.memory_weight.shape),
                (self.memory_size, self.hidden_size),
            ),
            (tuple(self.memory_bias.shape), (self.memory_size,)),
            (
                tuple(self.output_weight.shape),
                (6, self.memory_size),
            ),
            (tuple(self.output_bias.shape), (6,)),
            (tuple(self.route_indices.shape), (4_500,)),
            (tuple(self.valid_indices_mask.shape), (4_500,)),
            (tuple(self.kappa_b.shape), (45,)),
            (tuple(self.kappa_m.shape), (self.memory_size,)),
            (tuple(self.kappa_lambda.shape), (self.memory_size,)),
            (tuple(self.nmda_mean.shape), (4,)),
            (tuple(self.nmda_scale.shape), (4,)),
        )
        for actual, expected in shapes:
            if actual != expected:
                raise ValueError(f"shape {actual} != {expected}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(2) != 1_278:
            raise RuntimeError("input must be time x batch x 1278")
        time_steps = x.size(0)
        batch_size = x.size(1)
        routed = x.index_select(2, self.route_indices)
        effective_weight = torch.clamp_min(self.proto_w_s, 0.0)
        weighted = (
            routed
            * self.valid_indices_mask.view(1, 1, -1)
            * effective_weight.view(1, 1, -1)
        )
        branch_input = weighted.reshape(
            time_steps,
            batch_size,
            45,
            100,
        ).sum(dim=3)
        branch = x.new_zeros((batch_size, 45))
        memory = x.new_zeros((batch_size, self.memory_size))
        memories = torch.jit.annotate(List[torch.Tensor], [])
        for step in range(time_steps):
            branch = branch * self.kappa_b + branch_input[step]
            decayed_memory = memory * self.kappa_m
            hidden_pre = F.linear(
                torch.cat((branch, decayed_memory), dim=1),
                self.input_weight,
                self.input_bias,
            )
            hidden = hidden_pre / (1.0 + torch.exp(-hidden_pre))
            delta_memory = 1.7159 * torch.tanh(
                (2.0 / 3.0)
                * F.linear(
                    hidden,
                    self.memory_weight,
                    self.memory_bias,
                )
            )
            memory = (
                decayed_memory
                + (1.0 - self.kappa_lambda) * delta_memory
            )
            memories.append(memory)
        trajectory = torch.stack(memories, dim=0)
        return F.linear(
            trajectory,
            self.output_weight,
            self.output_bias,
        )


def make_model(
    bridge: Mapping[str, np.ndarray],
    backend: str,
) -> nn.Module:
    model = ScriptablePaperELMTwin(bridge)
    if backend == "torchscript":
        return torch.jit.script(model)
    if backend == "eager":
        return model
    raise ValueError(f"unknown backend {backend}")
