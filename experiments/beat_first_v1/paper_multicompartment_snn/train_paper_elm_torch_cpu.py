"""Exact CPU PyTorch port of the canonical profiled Paper-ELM trainer.

The model contract is fixed to:

* ProfiledOfficialPaperELMTwin / twinprop_paper_reconstruction
* signed 1278 -> 45 x 100 NeuronIO routing
* M=1000, hidden=2000, output=6, SiLU
* Spieler ELM-v2 branch and memory recurrence
* the Julia paper loss plus four-region normalized NMDA extension

The bridge NPZ is produced by ``export_paper_elm_r3e31_to_npz.jl``.  This
trainer accepts only the 32 fit and 8 train-derived validation trials.  It
fails closed before opening any held-out shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple


_EARLY_THREADS = int(os.environ.get("TWINPROP_TORCH_THREADS", "6"))
os.environ.setdefault("OMP_NUM_THREADS", str(_EARLY_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_EARLY_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_EARLY_THREADS))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


INPUT_DIM = 1_278
MEMORY = 1_000
HIDDEN = 2_000
OUTPUT = 6
NMDA_REGIONS = 4
BRANCHES = 45
SYNAPSES_PER_BRANCH = 100
TRAIN_WINDOW = 500
FIRST_RANDOM_START_JULIA = 501
LAST_RANDOM_START_JULIA = 1_000
BATCH_SIZE = 8
FIT_IDS = tuple(range(1, 33))
VALIDATION_IDS = tuple(range(33, 41))
BASE_LEARNING_RATE = 5.0e-4
COSINE_T_MAX = 140
SOMA_CLIP_MV = -55.0
SOMA_BIAS_MV = -67.7
SOMA_TRAIN_SCALE = 0.1
PARAMETER_NAMES = (
    "proto_w_s",
    "input_weight",
    "input_bias",
    "memory_weight",
    "memory_bias",
    "output_weight",
    "output_bias",
)
DEFAULT_BRIDGE = Path(os.environ.get(
    "PAPER_ELM_TORCH_BRIDGE",
    r"C:\tmp\paper_elm_r3e31_torch_bridge.npz",
))
DEFAULT_DATASET = Path(
    r"C:\tmp\hd_swsnn_neuron_teacher_final_dev1500_release"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(array: np.ndarray) -> float:
    return float(np.asarray(array).reshape(-1)[0])


class PaperELMTwinTorch(nn.Module):
    """TorchScript-compatible exact recurrence with vectorized fixed routing."""

    def __init__(self, bridge: Mapping[str, np.ndarray]) -> None:
        super().__init__()
        self.proto_w_s = nn.Parameter(
            torch.from_numpy(
                np.asarray(bridge["parameter_proto_w_s"], np.float32).copy()
            )
        )
        self.input_weight = nn.Parameter(
            torch.from_numpy(
                np.asarray(bridge["parameter_input_weight"], np.float32).copy()
            )
        )
        self.input_bias = nn.Parameter(
            torch.from_numpy(
                np.asarray(bridge["parameter_input_bias"], np.float32).copy()
            )
        )
        self.memory_weight = nn.Parameter(
            torch.from_numpy(
                np.asarray(bridge["parameter_memory_weight"], np.float32).copy()
            )
        )
        self.memory_bias = nn.Parameter(
            torch.from_numpy(
                np.asarray(bridge["parameter_memory_bias"], np.float32).copy()
            )
        )
        self.output_weight = nn.Parameter(
            torch.from_numpy(
                np.asarray(bridge["parameter_output_weight"], np.float32).copy()
            )
        )
        self.output_bias = nn.Parameter(
            torch.from_numpy(
                np.asarray(bridge["parameter_output_bias"], np.float32).copy()
            )
        )
        self.register_buffer(
            "route_indices",
            torch.from_numpy(
                np.asarray(bridge["route_indices_zero_based"], np.int64).copy()
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
        expected = {
            "proto_w_s": (4_500,),
            "input_weight": (HIDDEN, BRANCHES + MEMORY),
            "input_bias": (HIDDEN,),
            "memory_weight": (MEMORY, HIDDEN),
            "memory_bias": (MEMORY,),
            "output_weight": (OUTPUT, MEMORY),
            "output_bias": (OUTPUT,),
            "route_indices": (4_500,),
            "valid_indices_mask": (4_500,),
            "kappa_b": (BRANCHES,),
            "kappa_m": (MEMORY,),
            "kappa_lambda": (MEMORY,),
            "nmda_mean": (NMDA_REGIONS,),
            "nmda_scale": (NMDA_REGIONS,),
        }
        for name, shape in expected.items():
            actual = tuple(getattr(self, name).shape)
            if actual != shape:
                raise ValueError(f"{name} shape {actual} != {shape}")
        if int(self.route_indices.min()) < 0:
            raise ValueError("route indices contain a negative entry")
        if int(self.route_indices.max()) >= INPUT_DIM:
            raise ValueError("route indices exceed signed-1278 input")
        if not torch.all(self.nmda_scale > 0):
            raise ValueError("NMDA normalizer scale is not positive")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is time x batch x signed-1278.  Route the entire window once;
        # routing has no recurrent dependency.
        if x.dim() != 3 or x.size(2) != INPUT_DIM:
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
            BRANCHES,
            SYNAPSES_PER_BRANCH,
        ).sum(dim=3)

        branch = x.new_zeros((batch_size, BRANCHES))
        memory = x.new_zeros((batch_size, MEMORY))
        memories = torch.jit.annotate(List[torch.Tensor], [])
        for step in range(time_steps):
            branch = branch * self.kappa_b + branch_input[step]
            decayed_memory = memory * self.kappa_m
            hidden_pre = F.linear(
                torch.cat((branch, decayed_memory), dim=1),
                self.input_weight,
                self.input_bias,
            )
            # Match Julia's explicit x / (1 + exp(-x)), rather than relying
            # on a backend-selected SiLU approximation.
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


class JuliaAdam:
    """Optimisers.jl 0.4 Adam, including its stored beta-power convention."""

    def __init__(
        self,
        model: nn.Module,
        bridge: Mapping[str, np.ndarray],
    ) -> None:
        named = dict(model.named_parameters())
        if set(named) != set(PARAMETER_NAMES):
            raise ValueError(
                f"unexpected parameter names: {sorted(named)}"
            )
        self.parameters = [(name, named[name]) for name in PARAMETER_NAMES]
        self.first: Dict[str, torch.Tensor] = {}
        self.second: Dict[str, torch.Tensor] = {}
        for name, parameter in self.parameters:
            first = torch.from_numpy(
                np.asarray(
                    bridge[f"adam_first_moment_{name}"],
                    np.float32,
                ).copy()
            )
            second = torch.from_numpy(
                np.asarray(
                    bridge[f"adam_second_moment_{name}"],
                    np.float32,
                ).copy()
            )
            if first.shape != parameter.shape or second.shape != parameter.shape:
                raise ValueError(f"Adam state shape differs for {name}")
            self.first[name] = first
            self.second[name] = second
        self.beta = torch.from_numpy(
            np.asarray(bridge["adam_beta"], np.float32).copy()
        )
        self.beta_power = torch.from_numpy(
            np.asarray(bridge["adam_beta_power"], np.float32).copy()
        )
        self.epsilon = np.float32(_scalar(bridge["adam_epsilon"]))

    def zero_grad(self) -> None:
        for _, parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self, learning_rate: float) -> None:
        beta1 = float(self.beta[0])
        beta2 = float(self.beta[1])
        one_minus_beta1_power = 1.0 - self.beta_power[0]
        one_minus_beta2_power = 1.0 - self.beta_power[1]
        for name, parameter in self.parameters:
            gradient = parameter.grad
            if gradient is None:
                raise RuntimeError(f"missing gradient for {name}")
            first = self.first[name]
            second = self.second[name]
            first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            second.mul_(beta2).addcmul_(
                gradient,
                gradient,
                value=1.0 - beta2,
            )
            denominator = torch.sqrt(
                second / one_minus_beta2_power
            ).add_(float(self.epsilon))
            update = (
                first
                / one_minus_beta1_power
                / denominator
                * float(learning_rate)
            )
            parameter.sub_(update)
        self.beta_power.mul_(self.beta)

    def serializable(self) -> dict:
        return {
            "first": {name: value.clone() for name, value in self.first.items()},
            "second": {
                name: value.clone() for name, value in self.second.items()
            },
            "beta": self.beta.clone(),
            "beta_power": self.beta_power.clone(),
            "epsilon": float(self.epsilon),
        }


class SafeFitValidationDataset:
    """Lazy NPZ loader which makes held-out access structurally invalid."""

    def __init__(self, root: Path, expected_manifest_sha256: str) -> None:
        self.root = root.resolve()
        manifest_path = self.root / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if actual_manifest_sha256 != expected_manifest_sha256:
            raise ValueError("checkpoint/dataset manifest digest differs")
        self.manifest = json.loads(manifest_bytes)
        validation = tuple(
            int(value)
            for value in self.manifest["validation_from_train_indices"]
        )
        if validation != VALIDATION_IDS:
            raise ValueError("derived-validation membership differs")
        self.allowed_ids = frozenset(FIT_IDS + VALIDATION_IDS)
        self.records = []
        for record in self.manifest["shards"]:
            first = int(record["global_first"])
            last = int(record["global_last"])
            heldout_count = int(
                record.get("split_counts", {}).get("held_out_test", 0)
            )
            if heldout_count == 0 and first >= 1 and last <= 40:
                self.records.append(record)
        covered = {
            trial_id
            for record in self.records
            for trial_id in range(
                int(record["global_first"]),
                int(record["global_last"]) + 1,
            )
        }
        if covered != self.allowed_ids:
            raise ValueError("fit/validation shard inventory differs")
        self.cache: MutableMapping[str, Dict[str, np.ndarray]] = {}
        self.opened_shards: List[str] = []

    def _record_for_id(self, trial_id: int) -> Tuple[dict, int]:
        if trial_id not in self.allowed_ids:
            raise PermissionError(
                f"trial {trial_id} is outside fit/derived-validation; "
                "heldout access is forbidden"
            )
        for record in self.records:
            first = int(record["global_first"])
            last = int(record["global_last"])
            if first <= trial_id <= last:
                return record, trial_id - first
        raise KeyError(f"trial {trial_id} is absent")

    def _load(self, record: dict) -> Dict[str, np.ndarray]:
        relative = str(record["path"]).replace("/", os.sep)
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise ValueError("shard path escapes dataset root")
        key = str(path)
        if key not in self.cache:
            if _sha256_file(path) != str(record["sha256"]):
                raise ValueError(f"shard digest differs: {path.name}")
            with np.load(path, allow_pickle=False) as archive:
                self.cache[key] = {
                    name: np.asarray(archive[name])
                    for name in archive.files
                }
            ids = tuple(
                int(value)
                for value in self.cache[key]["sample_indices"].reshape(-1)
            )
            expected = tuple(
                range(
                    int(record["global_first"]),
                    int(record["global_last"]) + 1,
                )
            )
            if ids != expected:
                raise ValueError("numeric shard IDs differ")
            self.opened_shards.append(key)
        return self.cache[key]

    @staticmethod
    def _expand_input(
        data: Mapping[str, np.ndarray],
        item: int,
        start_julia: int,
        steps: int,
    ) -> np.ndarray:
        if start_julia < 1 or steps < 1:
            raise ValueError("invalid input interval")
        first_bin = start_julia - 1
        last_bin = first_bin + steps - 1
        axon_kind = np.asarray(data["axon_kind"])
        axons = axon_kind.shape[0]
        event_offsets = np.asarray(
            data["event_trial_offset"], np.int64
        ).reshape(-1)
        event_first = int(event_offsets[item])
        event_last = int(event_offsets[item + 1])
        event_axon = np.asarray(data["event_axon"]).reshape(-1)
        event_time = np.asarray(data["event_time_bin"]).reshape(-1)
        event_count = np.asarray(data["event_count"]).reshape(-1)
        counts = np.zeros((axons, steps), dtype=np.uint16)
        for event in range(event_first, event_last):
            time_bin = int(event_time[event])
            if first_bin <= time_bin <= last_bin:
                axon = int(event_axon[event]) - 1
                local_time = time_bin - first_bin
                updated = int(counts[axon, local_time]) + int(
                    event_count[event]
                )
                if updated > np.iinfo(np.uint16).max:
                    raise OverflowError("event-count accumulation overflow")
                counts[axon, local_time] = updated

        contact_offsets = np.asarray(
            data["contact_trial_offset"], np.int64
        ).reshape(-1)
        contact_first = int(contact_offsets[item])
        contact_last = int(contact_offsets[item + 1])
        contact_axon = np.asarray(data["contact_axon"]).reshape(-1)
        contact_segment = np.asarray(
            data["contact_segment"]
        ).reshape(-1)
        contact_kind = np.asarray(data["contact_kind"]).reshape(-1)
        contact_strength = np.asarray(
            data["contact_strength"], np.float32
        ).reshape(-1)
        output = np.zeros((INPUT_DIM, steps), dtype=np.float32)
        for contact in range(contact_first, contact_last):
            axon = int(contact_axon[contact]) - 1
            segment = int(contact_segment[contact])
            kind = int(contact_kind[contact])
            strength = np.float32(contact_strength[contact])
            if kind == 1:
                feature = segment - 2
                signed_strength = strength
            elif kind == 2:
                feature = 639 + segment - 2
                signed_strength = -strength
            else:
                raise ValueError("contact kind is not E/I")
            output[feature] += (
                signed_strength * counts[axon].astype(np.float32)
            )
        return output

    def trial_arrays(
        self,
        trial_id: int,
    ) -> Tuple[Dict[str, np.ndarray], int]:
        record, item = self._record_for_id(trial_id)
        return self._load(record), item

    def window(
        self,
        trial_id: int,
        start_julia: int,
        steps: int = TRAIN_WINDOW,
        pad_to: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data, item = self.trial_arrays(trial_id)
        total_steps = int(data["target_voltage"].shape[0])
        actual_steps = min(steps, total_steps - start_julia + 1)
        if actual_steps < 1:
            raise ValueError("window starts after the trial")
        inputs = self._expand_input(
            data,
            item,
            start_julia,
            actual_steps,
        )
        first = start_julia - 1
        last = first + actual_steps
        voltage = np.asarray(
            data["target_voltage"][first:last, item],
            np.float32,
        )
        spike = np.asarray(
            data["target_spike"][first:last, item],
            np.float32,
        )
        nmda = np.asarray(
            data["target_nmda"][:, first:last, item],
            np.float32,
        ).T
        if pad_to is not None and actual_steps < pad_to:
            padded_input = np.zeros((INPUT_DIM, pad_to), np.float32)
            padded_voltage = np.zeros((pad_to,), np.float32)
            padded_spike = np.zeros((pad_to,), np.float32)
            padded_nmda = np.zeros((pad_to, NMDA_REGIONS), np.float32)
            padded_input[:, :actual_steps] = inputs
            padded_voltage[:actual_steps] = voltage
            padded_spike[:actual_steps] = spike
            padded_nmda[:actual_steps] = nmda
            return (
                padded_input,
                padded_voltage,
                padded_spike,
                padded_nmda,
            )
        return inputs, voltage, spike, nmda

    def assert_no_heldout_opened(self) -> None:
        allowed = {
            str((self.root / str(record["path"])).resolve())
            for record in self.records
        }
        unexpected = set(self.opened_shards) - allowed
        if unexpected:
            raise AssertionError(
                f"non-fit/validation shards were opened: {unexpected}"
            )


def _materialize_batch(
    dataset: SafeFitValidationDataset,
    ids: Sequence[int],
    starts_julia: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(ids) != len(starts_julia):
        raise ValueError("trial IDs and crop starts differ")
    if len(ids) != BATCH_SIZE:
        raise ValueError("canonical batch size must be 8")
    windows = [
        dataset.window(int(trial_id), int(start))
        for trial_id, start in zip(ids, starts_julia)
    ]
    inputs = torch.from_numpy(
        np.stack([window[0].T for window in windows], axis=1)
    )
    voltage = torch.from_numpy(
        np.stack([window[1] for window in windows], axis=1)
    )
    spike = torch.from_numpy(
        np.stack([window[2] for window in windows], axis=1)
    )
    nmda = torch.from_numpy(
        np.stack([window[3] for window in windows], axis=1)
    )
    return inputs, voltage, spike, nmda


def _objective(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    inputs, target_voltage, target_spike, target_nmda = batch
    raw = model(inputs)
    spike_logit = raw[:, :, 0]
    voltage = raw[:, :, 1]
    nmda = raw[:, :, 2:]
    target_voltage_coordinate = (
        torch.minimum(
            target_voltage,
            target_voltage.new_tensor(SOMA_CLIP_MV),
        )
        - SOMA_BIAS_MV
    ) * SOMA_TRAIN_SCALE
    voltage_mse = torch.mean(
        torch.square(voltage - target_voltage_coordinate)
    )
    spike_bce = F.binary_cross_entropy_with_logits(
        spike_logit,
        target_spike,
        reduction="mean",
    )
    paper_loss = 0.5 * voltage_mse + 0.5 * spike_bce
    target_nmda_coordinate = (
        target_nmda
        - model.nmda_mean.view(1, 1, NMDA_REGIONS)
    ) / model.nmda_scale.view(1, 1, NMDA_REGIONS)
    nmda_extension_loss = torch.mean(
        torch.square(nmda - target_nmda_coordinate)
    )
    total = paper_loss + nmda_extension_loss
    return total, {
        "total": total,
        "paper_loss": paper_loss,
        "voltage_mse": voltage_mse,
        "spike_bce": spike_bce,
        "nmda_extension_loss": nmda_extension_loss,
    }


def _float_components(components: Mapping[str, torch.Tensor]) -> dict:
    return {
        name: float(value.detach())
        for name, value in components.items()
    }


def _bridge_metadata(path: Path) -> dict:
    metadata_path = Path(str(path) + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["schema"] != "paper_elm_r3e31_torch_bridge.v1":
        raise ValueError("bridge metadata schema differs")
    if metadata["restart_index"] != 3 or metadata["epoch"] != 31:
        raise ValueError("bridge is not selected r3/e31")
    if metadata["model"] != {
        "input": 1278,
        "memory": 1000,
        "hidden": 2000,
        "output": 6,
        "activation": "silu",
        "profile": "twinprop_paper_reconstruction",
        "recurrence": "Spieler ELM v2",
        "routing": "signed-1278 neuronio_routing",
    }:
        raise ValueError("bridge model contract differs")
    return metadata


def _load_bridge(path: Path) -> Tuple[Dict[str, np.ndarray], dict]:
    metadata = _bridge_metadata(path)
    with np.load(path, allow_pickle=False) as archive:
        bridge = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "route_indices_zero_based",
        "valid_indices_mask",
        "kappa_b",
        "kappa_m",
        "kappa_lambda",
        "nmda_mean",
        "nmda_scale",
        "adam_beta",
        "adam_beta_power",
        "adam_epsilon",
    }
    for name in PARAMETER_NAMES:
        required.add(f"parameter_{name}")
        required.add(f"adam_first_moment_{name}")
        required.add(f"adam_second_moment_{name}")
    missing = required - set(bridge)
    if missing:
        raise ValueError(f"bridge arrays are missing: {sorted(missing)}")
    return bridge, metadata


def _make_model(
    bridge: Mapping[str, np.ndarray],
    backend: str,
) -> nn.Module:
    model = PaperELMTwinTorch(bridge)
    if backend == "torchscript":
        return torch.jit.script(model)
    if backend == "eager":
        return model
    raise ValueError(f"unknown backend {backend}")


def _oracle(
    model: nn.Module,
    bridge: Mapping[str, np.ndarray],
) -> dict:
    for name in (
        "oracle_input",
        "oracle_target_voltage",
        "oracle_target_spike",
        "oracle_target_nmda",
        "oracle_raw",
        "oracle_loss_components",
    ):
        if name not in bridge:
            raise ValueError("bridge was exported without --oracle true")
    inputs = torch.from_numpy(
        np.asarray(bridge["oracle_input"], np.float32)
        .transpose(1, 2, 0)
        .copy()
    )
    voltage = torch.from_numpy(
        np.asarray(bridge["oracle_target_voltage"], np.float32).copy()
    )
    spike = torch.from_numpy(
        np.asarray(bridge["oracle_target_spike"], np.float32).copy()
    )
    nmda = torch.from_numpy(
        np.asarray(bridge["oracle_target_nmda"], np.float32)
        .transpose(1, 2, 0)
        .copy()
    )
    with torch.no_grad():
        actual_raw = model(inputs)
        _, actual_components = _objective(
            model,
            (inputs, voltage, spike, nmda),
        )
    expected_raw = torch.from_numpy(
        np.asarray(bridge["oracle_raw"], np.float32)
        .transpose(1, 2, 0)
        .copy()
    )
    expected_components = np.asarray(
        bridge["oracle_loss_components"], np.float32
    ).reshape(-1)
    actual_component_array = np.asarray(
        [
            float(actual_components["total"]),
            float(actual_components["paper_loss"]),
            float(actual_components["voltage_mse"]),
            float(actual_components["spike_bce"]),
            float(actual_components["nmda_extension_loss"]),
        ],
        dtype=np.float32,
    )
    raw_difference = torch.abs(actual_raw - expected_raw)
    component_difference = np.abs(
        actual_component_array - expected_components
    )
    result = {
        "passed": bool(
            torch.allclose(
                actual_raw,
                expected_raw,
                rtol=2.0e-4,
                atol=5.0e-5,
            )
            and np.allclose(
                actual_component_array,
                expected_components,
                rtol=2.0e-4,
                atol=5.0e-5,
            )
        ),
        "raw_max_abs": float(raw_difference.max()),
        "raw_mean_abs": float(raw_difference.mean()),
        "component_max_abs": float(component_difference.max()),
        "julia_components": expected_components.tolist(),
        "torch_components": actual_component_array.tolist(),
        "rtol": 2.0e-4,
        "atol": 5.0e-5,
    }
    if not result["passed"]:
        raise AssertionError(f"Julia/PyTorch oracle differs: {result}")
    return result


def _gradient_statistics(model: nn.Module) -> dict:
    squared_norm = 0.0
    elements = 0
    finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad
        squared_norm += float(
            torch.sum(gradient.double().square())
        )
        elements += gradient.numel()
        finite = finite and bool(torch.all(torch.isfinite(gradient)))
    return {
        "norm": math.sqrt(squared_norm),
        "elements": elements,
        "finite": finite,
    }


def _learning_rate(update_index: int) -> float:
    return BASE_LEARNING_RATE * 0.5 * (
        1.0
        + math.cos(
            np.float32(math.pi)
            * np.float32(update_index - 1)
            / np.float32(COSINE_T_MAX)
        )
    )


def _one_update(
    model: nn.Module,
    optimizer: JuliaAdam,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    update_index: int,
) -> Tuple[dict, float]:
    optimizer.zero_grad()
    started = time.perf_counter()
    loss, components = _objective(model, batch)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite loss")
    loss.backward()
    gradient = _gradient_statistics(model)
    if not gradient["finite"] or gradient["norm"] <= 0:
        raise FloatingPointError("invalid gradient")
    learning_rate = _learning_rate(update_index)
    optimizer.step(learning_rate)
    elapsed = time.perf_counter() - started
    return {
        "update_index": update_index,
        "learning_rate": learning_rate,
        "components": _float_components(components),
        "gradient": gradient,
    }, elapsed


def _benchmark(
    model: nn.Module,
    optimizer: JuliaAdam,
    dataset: SafeFitValidationDataset,
    prior_update_index: int,
) -> dict:
    batch = _materialize_batch(
        dataset,
        FIT_IDS[:BATCH_SIZE],
        [FIRST_RANDOM_START_JULIA] * BATCH_SIZE,
    )
    # Warm MKL and TorchScript dispatch without building a full autograd graph.
    with torch.no_grad():
        model(batch[0][:2])
    update, elapsed = _one_update(
        model,
        optimizer,
        batch,
        prior_update_index + 1,
    )
    return {
        "kind": "forward_backward_optimizer_one_update",
        "elapsed_seconds": elapsed,
        "updates_per_second": 1.0 / elapsed,
        "baseline_seconds_per_update": 75.0,
        "speedup_vs_75_seconds": 75.0 / elapsed,
        "data_materialization_timed": False,
        "ids": list(FIT_IDS[:BATCH_SIZE]),
        "crop_starts_julia": [FIRST_RANDOM_START_JULIA] * BATCH_SIZE,
        "update": update,
    }


def _validation_rmse(
    model: nn.Module,
    dataset: SafeFitValidationDataset,
) -> float:
    squared_error = 0.0
    observations = 0
    starts_julia = (1, 351, 701, 1_051)
    with torch.no_grad():
        for trial_id in VALIDATION_IDS:
            data, item = dataset.trial_arrays(trial_id)
            target_full = np.asarray(
                data["target_voltage"][:, item],
                np.float32,
            )
            if target_full.shape != (1_500,):
                raise ValueError("validation trial does not have 1500 bins")
            for window_index, start_julia in enumerate(starts_julia):
                input_array, _, _, _ = dataset.window(
                    trial_id,
                    start_julia,
                    TRAIN_WINDOW,
                    pad_to=TRAIN_WINDOW,
                )
                inputs = torch.from_numpy(
                    input_array.T[:, None, :].copy()
                )
                raw = model(inputs)
                actual_steps = min(
                    TRAIN_WINDOW,
                    1_500 - start_julia + 1,
                )
                local_keep_first = 0 if window_index == 0 else 150
                global_keep_first = (
                    start_julia - 1 + local_keep_first
                )
                metric_global_first = max(global_keep_first, 500)
                global_keep_last = start_julia - 1 + actual_steps - 1
                if metric_global_first > global_keep_last:
                    continue
                local_metric_first = (
                    local_keep_first
                    + metric_global_first
                    - global_keep_first
                )
                local_last_exclusive = actual_steps
                predicted_mv = (
                    raw[
                        local_metric_first:local_last_exclusive,
                        0,
                        1,
                    ].double()
                    / SOMA_TRAIN_SCALE
                    + SOMA_BIAS_MV
                )
                target_mv = np.minimum(
                    target_full[
                        metric_global_first:global_keep_last + 1
                    ],
                    SOMA_CLIP_MV,
                )
                difference = (
                    predicted_mv
                    - torch.from_numpy(target_mv.astype(np.float64))
                )
                squared_error += float(torch.sum(difference.square()))
                observations += target_mv.size
    expected_observations = len(VALIDATION_IDS) * 1_000
    if observations != expected_observations:
        raise AssertionError(
            f"validation observations {observations} "
            f"!= {expected_observations}"
        )
    return math.sqrt(squared_error / observations)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: JuliaAdam,
    metadata: dict,
    update_index: int,
    events: Sequence[dict],
    force: bool,
) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite {path}; pass --force true"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "paper_elm_torch_cpu_checkpoint.v1",
            "source_bridge_metadata": metadata,
            "update_index": update_index,
            "model": model.state_dict(),
            "julia_adam": optimizer.serializable(),
            "events": list(events),
            "heldout_opened": False,
        },
        path,
    )


def _parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bridge",
        type=Path,
        default=DEFAULT_BRIDGE,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )
    parser.add_argument(
        "--mode",
        choices=("oracle", "benchmark", "train", "validate"),
        default="benchmark",
    )
    parser.add_argument(
        "--backend",
        choices=("torchscript", "eager"),
        default="torchscript",
    )
    parser.add_argument("--threads", type=int, default=_EARLY_THREADS)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6077687918186389330)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--force", type=_parse_bool, default=False)
    return parser.parse_args()


def main() -> None:
    options = _parse_args()
    if options.threads < 1:
        raise ValueError("--threads must be positive")
    if options.updates < 1:
        raise ValueError("--updates must be positive")
    torch.set_num_threads(options.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    bridge, metadata = _load_bridge(options.bridge.resolve())
    model = _make_model(bridge, options.backend)
    optimizer = JuliaAdam(model, bridge)
    result = {
        "schema": "paper_elm_torch_cpu_result.v1",
        "mode": options.mode,
        "backend": options.backend,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "source_restart": metadata["restart_index"],
        "source_epoch": metadata["epoch"],
        "source_update_index": metadata["update_index"],
        "model_contract": metadata["model"],
        "split_contract": {
            "fit_ids": [1, 32],
            "validation_ids": [33, 40],
            "heldout_opened": False,
        },
    }
    if options.mode == "oracle":
        result["oracle"] = _oracle(model, bridge)
    else:
        dataset = SafeFitValidationDataset(
            options.dataset,
            metadata["manifest_sha256"],
        )
        if options.mode == "benchmark":
            if options.updates != 1:
                raise ValueError("benchmark mode runs exactly one update")
            result["benchmark"] = _benchmark(
                model,
                optimizer,
                dataset,
                int(metadata["update_index"]),
            )
        elif options.mode == "validate":
            result["validation_physical_voltage_rmse_mv"] = (
                _validation_rmse(model, dataset)
            )
        else:
            rng = np.random.default_rng(options.seed)
            update_index = int(metadata["update_index"])
            events = []
            for _ in range(options.updates):
                ids = rng.choice(
                    np.asarray(FIT_IDS),
                    size=BATCH_SIZE,
                    replace=False,
                ).tolist()
                starts = rng.integers(
                    FIRST_RANDOM_START_JULIA,
                    LAST_RANDOM_START_JULIA + 1,
                    size=BATCH_SIZE,
                ).tolist()
                batch = _materialize_batch(dataset, ids, starts)
                update_index += 1
                update, elapsed = _one_update(
                    model,
                    optimizer,
                    batch,
                    update_index,
                )
                update["elapsed_seconds"] = elapsed
                update["ids"] = ids
                update["crop_starts_julia"] = starts
                events.append(update)
            result["updates"] = events
            result["final_update_index"] = update_index
            if options.checkpoint_out is not None:
                _save_checkpoint(
                    options.checkpoint_out.resolve(),
                    model,
                    optimizer,
                    metadata,
                    update_index,
                    events,
                    options.force,
                )
                result["checkpoint_out"] = str(
                    options.checkpoint_out.resolve()
                )
        dataset.assert_no_heldout_opened()
        result["opened_shards"] = dataset.opened_shards
        result["split_contract"]["heldout_opened"] = False
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
