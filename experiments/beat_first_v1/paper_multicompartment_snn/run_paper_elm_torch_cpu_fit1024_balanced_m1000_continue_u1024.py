"""Resume fit1024 M1000 from u512 through u1024."""

from __future__ import annotations

import torch

import run_paper_elm_torch_cpu_fit1024_balanced_m1000 as mainline


core = mainline.core
runner = mainline.runner
_PREVIOUS_LOAD_BRIDGE = core._load_bridge
_BASE_PARSE_ARGS = mainline._ORIGINAL_PARSE_ARGS


def _continuation_bridge(path):
    bridge, metadata = _PREVIOUS_LOAD_BRIDGE(path)
    metadata = dict(metadata)
    details = dict(metadata["m1000_fit1024_mainline"])
    details.update({
        "weight_initialization": "resumed fit1024 M1000 checkpoint",
        "optimizer": "preserved Julia-formula Adam",
        "optimizer_state_from_warmstart": True,
        "initial_update_index": 512,
        "updates": 512,
        "final_update_index": 1_024,
        "schedule": "fixed continuation",
    })
    metadata["m1000_fit1024_mainline"] = details
    return bridge, metadata


def _restore_full_checkpoint(
    path,
    model,
    optimizer,
    expected_memory,
) -> int:
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != "paper_elm_torch_cpu_variable.v1":
        raise ValueError("input checkpoint schema differs")
    if int(payload["memory"]) != expected_memory:
        raise ValueError("input checkpoint memory size differs")
    if int(payload["update_index"]) != 512:
        raise ValueError("fit1024 continuation requires u512")
    source_metadata = payload.get("source_bridge_metadata", {})
    if (
        source_metadata.get("manifest_sha256")
        != mainline._MANIFEST_SHA256
    ):
        raise ValueError("checkpoint/dataset manifest digest differs")
    if bool(payload.get("heldout_opened", True)):
        raise ValueError("checkpoint does not assert heldout exclusion")
    model.load_state_dict(payload["model"])
    state = payload["julia_adam"]
    for name in core.PARAMETER_NAMES:
        optimizer.first[name].copy_(state["first"][name])
        optimizer.second[name].copy_(state["second"][name])
    optimizer.beta.copy_(state["beta"])
    optimizer.beta_power.copy_(state["beta_power"])
    return 512


def _parse_u1024_args():
    options = _BASE_PARSE_ARGS()
    if options.dataset.resolve() != mainline._DATASET:
        raise ValueError("dataset path changed after fit1024 audit")
    if options.mode != "train":
        raise ValueError("fit1024 continuation supports train mode only")
    if options.memory != core.FULL_MEMORY:
        raise ValueError("fit1024 continuation requires memory=1000")
    if options.backend != "torchscript":
        raise ValueError("fit1024 continuation requires TorchScript")
    if options.threads != mainline.EXPECTED_THREADS:
        raise ValueError("fit1024 continuation requires threads=6")
    if options.updates != 512:
        raise ValueError("u513-u1024 continuation requires 512 updates")
    if options.checkpoint_in is None:
        raise ValueError("fit1024 continuation requires input checkpoint")
    if options.checkpoint_out is None:
        raise ValueError("fit1024 continuation requires checkpoint output")
    return options


core._load_bridge = _continuation_bridge
runner._restore_checkpoint = _restore_full_checkpoint
runner._parse_args = _parse_u1024_args


if __name__ == "__main__":
    runner.main()
