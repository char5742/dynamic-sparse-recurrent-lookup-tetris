#!/usr/bin/env python3
"""Run the public checkpoint with its actual pinned ELM-v1 implementation."""

from __future__ import annotations

import importlib
import runpy
import sys
import types
from pathlib import Path


stub = types.ModuleType("src.neuronio.neuronio_data_utils")
stub.DEFAULT_Y_TRAIN_SOMA_SCALE = 0.1
sys.modules["src.neuronio.neuronio_data_utils"] = stub

# The published checkpoint was produced by expressive_leaky_memory_neuron.py
# (synapse-state ELM v1), despite the repository also containing a v2 module.
actual = importlib.import_module("src.expressive_leaky_memory_neuron")
alias = types.ModuleType("src.expressive_leaky_memory_neuron_v2")
alias.ELM = actual.ELM
sys.modules["src.expressive_leaky_memory_neuron_v2"] = alias

runpy.run_path(
    str(Path(__file__).with_name("evaluate_spieler_published_best_fastpath.py")),
    run_name="__main__",
)
