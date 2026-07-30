"""M-variable runner and zero-isolated M256 -> M1000 promotion.

This is a thin orchestration layer over ``train_paper_elm_torch_cpu.py``.
M1000 imports the selected Julia r3/e31 weights and Adam state exactly.
M256 uses the leading 256-memory / 512-hidden block and resets Adam, while
retaining the same routing and the matching leading ELM-v2 decay factors.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch

import train_paper_elm_torch_cpu as core


SUPPORTED_MEMORY = (256, 1_000)


def _small_bridge(
    full: Mapping[str, np.ndarray],
    memory: int,
) -> Dict[str, np.ndarray]:
    if memory == 1_000:
        return {name: np.asarray(value) for name, value in full.items()}
    if memory != 256:
        raise ValueError("only M256 and M1000 are supported")
    hidden = 2 * memory
    result = {
        name: np.asarray(value)
        for name, value in full.items()
        if not (
            name.startswith("parameter_")
            or name.startswith("adam_first_moment_")
            or name.startswith("adam_second_moment_")
        )
    }
    result["kappa_m"] = np.asarray(full["kappa_m"])[:memory].copy()
    result["kappa_lambda"] = np.asarray(
        full["kappa_lambda"]
    )[:memory].copy()
    result["initial_proto_tau_m"] = np.asarray(
        full["initial_proto_tau_m"]
    )[:memory].copy()
    result["parameter_proto_w_s"] = np.asarray(
        full["parameter_proto_w_s"]
    ).copy()
    full_input_weight = np.asarray(full["parameter_input_weight"])
    result["parameter_input_weight"] = np.concatenate(
        (
            full_input_weight[:hidden, : core.BRANCHES],
            full_input_weight[
                :hidden,
                core.BRANCHES : core.BRANCHES + memory,
            ],
        ),
        axis=1,
    ).copy()
    result["parameter_input_bias"] = np.asarray(
        full["parameter_input_bias"]
    )[:hidden].copy()
    result["parameter_memory_weight"] = np.asarray(
        full["parameter_memory_weight"]
    )[:memory, :hidden].copy()
    result["parameter_memory_bias"] = np.asarray(
        full["parameter_memory_bias"]
    )[:memory].copy()
    result["parameter_output_weight"] = np.asarray(
        full["parameter_output_weight"]
    )[:, :memory].copy()
    result["parameter_output_bias"] = np.asarray(
        full["parameter_output_bias"]
    ).copy()
    for name in core.PARAMETER_NAMES:
        shape = result[f"parameter_{name}"].shape
        result[f"adam_first_moment_{name}"] = np.zeros(
            shape,
            dtype=np.float32,
        )
        result[f"adam_second_moment_{name}"] = np.zeros(
            shape,
            dtype=np.float32,
        )
    result["adam_beta_power"] = np.asarray(
        full["adam_beta"],
        dtype=np.float32,
    ).copy()
    return result


def _configure_core(memory: int) -> None:
    # The base implementation intentionally fixes the release model by module
    # constants.  Override them before construction/TorchScript compilation.
    core.MEMORY = memory
    core.HIDDEN = 2 * memory


def _build(
    bridge: Mapping[str, np.ndarray],
    memory: int,
    backend: str,
) -> torch.nn.Module:
    _configure_core(memory)
    return core._make_model(bridge, backend)


def _restore_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: core.JuliaAdam,
    expected_memory: int,
) -> int:
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != "paper_elm_torch_cpu_variable.v1":
        raise ValueError("input checkpoint schema differs")
    if int(payload["memory"]) != expected_memory:
        raise ValueError("input checkpoint memory size differs")
    model.load_state_dict(payload["model"])
    state = payload["julia_adam"]
    for name in core.PARAMETER_NAMES:
        optimizer.first[name].copy_(state["first"][name])
        optimizer.second[name].copy_(state["second"][name])
    optimizer.beta.copy_(state["beta"])
    optimizer.beta_power.copy_(state["beta_power"])
    return int(payload["update_index"])


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: core.JuliaAdam,
    memory: int,
    update_index: int,
    source_metadata: dict,
    events: Sequence[dict],
    force: bool,
    zero_isolated_from_m256: bool = False,
) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite {path}; pass --force true"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "paper_elm_torch_cpu_variable.v1",
            "memory": memory,
            "hidden": 2 * memory,
            "update_index": update_index,
            "model": model.state_dict(),
            "julia_adam": optimizer.serializable(),
            "source_bridge_metadata": source_metadata,
            "events": list(events),
            "zero_isolated_from_m256": zero_isolated_from_m256,
            "heldout_opened": False,
        },
        path,
    )


@torch.no_grad()
def _embed_leading_block_zero_isolated(
    small: torch.nn.Module,
    full: torch.nn.Module,
) -> None:
    """Make M1000 output/recurrence exactly equal to the M256 leading block."""

    full.proto_w_s.copy_(small.proto_w_s)
    full.input_weight.zero_()
    full.input_weight[
        : core.FAST_HIDDEN,
        : core.BRANCHES,
    ].copy_(
        small.input_weight[:, : core.BRANCHES]
    )
    full.input_weight[
        : core.FAST_HIDDEN,
        core.BRANCHES : core.BRANCHES + core.FAST_MEMORY,
    ].copy_(
        small.input_weight[:, core.BRANCHES :]
    )
    full.input_bias.zero_()
    full.input_bias[: core.FAST_HIDDEN].copy_(small.input_bias)
    full.memory_weight.zero_()
    full.memory_weight[
        : core.FAST_MEMORY,
        : core.FAST_HIDDEN,
    ].copy_(small.memory_weight)
    full.memory_bias.zero_()
    full.memory_bias[: core.FAST_MEMORY].copy_(small.memory_bias)
    full.output_weight.zero_()
    full.output_weight[:, : core.FAST_MEMORY].copy_(
        small.output_weight
    )
    full.output_bias.copy_(small.output_bias)


def _reset_adam_bridge(
    bridge: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    result = {name: np.asarray(value) for name, value in bridge.items()}
    for name in core.PARAMETER_NAMES:
        result[f"adam_first_moment_{name}"] = np.zeros_like(
            result[f"parameter_{name}"],
            dtype=np.float32,
        )
        result[f"adam_second_moment_{name}"] = np.zeros_like(
            result[f"parameter_{name}"],
            dtype=np.float32,
        )
    result["adam_beta_power"] = np.asarray(
        result["adam_beta"],
        np.float32,
    ).copy()
    return result


def _promote(
    full_bridge: Mapping[str, np.ndarray],
    metadata: dict,
    options: argparse.Namespace,
) -> dict:
    if options.checkpoint_in is None or options.checkpoint_out is None:
        raise ValueError(
            "promote mode requires --checkpoint-in and --checkpoint-out"
        )
    small_bridge = _small_bridge(full_bridge, core.FAST_MEMORY)
    small = _build(small_bridge, core.FAST_MEMORY, options.backend)
    small_optimizer = core.JuliaAdam(small, small_bridge)
    small_update = _restore_checkpoint(
        options.checkpoint_in.resolve(),
        small,
        small_optimizer,
        core.FAST_MEMORY,
    )

    full = _build(full_bridge, core.FULL_MEMORY, options.backend)
    _embed_leading_block_zero_isolated(small, full)
    probe = torch.from_numpy(
        np.asarray(full_bridge["oracle_input"], np.float32)
        .transpose(1, 2, 0)[:8]
        .copy()
    )
    with torch.no_grad():
        small_raw = small(probe)
        full_raw = full(probe)
    difference = torch.abs(small_raw - full_raw)
    if not torch.allclose(
        small_raw,
        full_raw,
        rtol=2.0e-5,
        atol=2.0e-6,
    ):
        raise AssertionError(
            "zero-isolated M1000 does not reproduce M256"
        )
    reset_bridge = _reset_adam_bridge(full_bridge)
    full_optimizer = core.JuliaAdam(full, reset_bridge)
    _save_checkpoint(
        options.checkpoint_out.resolve(),
        full,
        full_optimizer,
        core.FULL_MEMORY,
        0,
        metadata,
        (),
        options.force,
        zero_isolated_from_m256=True,
    )
    return {
        "small_update_index": small_update,
        "promoted_memory": core.FULL_MEMORY,
        "promoted_hidden": core.FULL_HIDDEN,
        "zero_isolation_raw_max_abs": float(difference.max()),
        "zero_isolation_passed": True,
        "checkpoint_out": str(options.checkpoint_out.resolve()),
        "adam_reset_for_full_finetune": True,
        "heldout_opened": False,
    }


def _parse_bool(value: str) -> bool:
    return core._parse_bool(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bridge",
        type=Path,
        default=core.DEFAULT_BRIDGE,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=core.DEFAULT_DATASET,
    )
    parser.add_argument(
        "--mode",
        choices=("oracle", "benchmark", "train", "validate", "promote"),
        default="benchmark",
    )
    parser.add_argument(
        "--memory",
        type=int,
        choices=SUPPORTED_MEMORY,
        default=core.FAST_MEMORY,
    )
    parser.add_argument(
        "--backend",
        choices=("torchscript", "eager"),
        default="torchscript",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(
            core.os.environ.get("TWINPROP_TORCH_THREADS", "6")
        ),
    )
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6077687918186389330)
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--force", type=_parse_bool, default=False)
    return parser.parse_args()


def main() -> None:
    options = _parse_args()
    if options.threads < 1 or options.updates < 1:
        raise ValueError("threads and updates must be positive")
    torch.set_num_threads(options.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    full_bridge, metadata = core._load_bridge(
        options.bridge.resolve()
    )
    result = {
        "schema": "paper_elm_torch_cpu_variable_result.v1",
        "mode": options.mode,
        "memory": options.memory,
        "hidden": 2 * options.memory,
        "backend": options.backend,
        "threads": torch.get_num_threads(),
        "heldout_opened": False,
    }
    if options.mode == "promote":
        result["promotion"] = _promote(
            full_bridge,
            metadata,
            options,
        )
        print(json.dumps(result, sort_keys=True))
        return

    bridge = _small_bridge(full_bridge, options.memory)
    model = _build(bridge, options.memory, options.backend)
    optimizer = core.JuliaAdam(model, bridge)
    prior_update = (
        int(metadata["update_index"])
        if options.memory == core.FULL_MEMORY
        else 0
    )
    if options.checkpoint_in is not None:
        prior_update = _restore_checkpoint(
            options.checkpoint_in.resolve(),
            model,
            optimizer,
            options.memory,
        )
    if options.mode == "oracle":
        if options.memory != core.FULL_MEMORY:
            raise ValueError("Julia oracle exists only for imported M1000")
        result["oracle"] = core._oracle(model, full_bridge)
    else:
        dataset = core.SafeFitValidationDataset(
            options.dataset,
            metadata["manifest_sha256"],
        )
        if options.mode == "benchmark":
            if options.updates != 1:
                raise ValueError("benchmark runs exactly one update")
            result["benchmark"] = core._benchmark(
                model,
                optimizer,
                dataset,
                prior_update,
            )
        elif options.mode == "validate":
            result["validation_physical_voltage_rmse_mv"] = (
                core._validation_rmse(model, dataset)
            )
        else:
            rng = np.random.default_rng(options.seed)
            update_index = prior_update
            events = []
            for _ in range(options.updates):
                ids = rng.choice(
                    np.asarray(core.FIT_IDS),
                    size=core.BATCH_SIZE,
                    replace=False,
                ).tolist()
                starts = rng.integers(
                    core.FIRST_RANDOM_START_JULIA,
                    core.LAST_RANDOM_START_JULIA + 1,
                    size=core.BATCH_SIZE,
                ).tolist()
                batch = core._materialize_batch(
                    dataset,
                    ids,
                    starts,
                )
                update_index += 1
                event, elapsed = core._one_update(
                    model,
                    optimizer,
                    batch,
                    update_index,
                )
                event["elapsed_seconds"] = elapsed
                event["ids"] = ids
                event["crop_starts_julia"] = starts
                events.append(event)
            result["updates"] = events
            result["final_update_index"] = update_index
            if options.checkpoint_out is not None:
                _save_checkpoint(
                    options.checkpoint_out.resolve(),
                    model,
                    optimizer,
                    options.memory,
                    update_index,
                    metadata,
                    events,
                    options.force,
                )
                result["checkpoint_out"] = str(
                    options.checkpoint_out.resolve()
                )
        dataset.assert_no_heldout_opened()
        result["opened_shards"] = dataset.opened_shards
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
