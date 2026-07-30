#!/usr/bin/env python3
"""Execute public ELM-v1 fastpath; compact shards are axon-major."""

from __future__ import annotations

import importlib
import runpy
import sys
import types
from pathlib import Path

import numpy as np


repo_arg = sys.argv[sys.argv.index("--elm-repo") + 1]
sys.path.insert(0, str(Path(repo_arg).resolve()))
stub = types.ModuleType("src.neuronio.neuronio_data_utils")
stub.DEFAULT_Y_TRAIN_SOMA_SCALE = 0.1
sys.modules["src.neuronio.neuronio_data_utils"] = stub
actual = importlib.import_module("src.expressive_leaky_memory_neuron")
alias = types.ModuleType("src.expressive_leaky_memory_neuron_v2")
alias.ELM = actual.ELM
sys.modules["src.expressive_leaky_memory_neuron_v2"] = alias

module = runpy.run_path(
    str(Path(__file__).with_name("evaluate_spieler_published_best_fastpath.py")),
    run_name="spieler_fastpath_library",
)


def reconstruct(shard, local_index: int) -> np.ndarray:
    result = np.zeros((1500, 1278), dtype=np.float32)
    contacts_by_axon = {}
    offsets = shard["contact_trial_offset"]
    for contact in range(int(offsets[local_index]), int(offsets[local_index + 1])):
        segment = int(shard["contact_segment"][contact])
        kind = int(shard["contact_kind"][contact])
        if not 2 <= segment <= 640 or kind not in (1, 2):
            raise RuntimeError("illegal compact contact")
        channel = segment - 2 + (0 if kind == 1 else 639)
        strength = np.float32(shard["contact_strength"][contact])
        if kind == 2:
            strength = -strength
        axon = int(shard["contact_axon"][contact])
        contacts_by_axon.setdefault(axon, []).append((channel, strength))
    offsets = shard["event_trial_offset"]
    for event in range(int(offsets[local_index]), int(offsets[local_index + 1])):
        time = int(shard["event_time_bin"][event])
        axon = int(shard["event_axon"][event])
        count = np.float32(shard["event_count"][event])
        for channel, strength in contacts_by_axon.get(axon, ()):
            result[time, channel] += strength * count
    return result


module["_load_fit_validation"].__globals__["_reconstruct_trial"] = reconstruct
module["main"]()
