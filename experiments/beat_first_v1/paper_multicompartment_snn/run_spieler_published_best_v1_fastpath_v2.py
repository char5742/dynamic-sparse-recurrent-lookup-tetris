#!/usr/bin/env python3
"""Dependency-light entry point for the actual public ELM-v1 checkpoint."""

from __future__ import annotations

import importlib
import runpy
import sys
import types
from pathlib import Path


try:
    repo_arg = sys.argv[sys.argv.index("--elm-repo") + 1]
except (ValueError, IndexError) as error:
    raise SystemExit("--elm-repo is required") from error
sys.path.insert(0, str(Path(repo_arg).resolve()))

stub = types.ModuleType("src.neuronio.neuronio_data_utils")
stub.DEFAULT_Y_TRAIN_SOMA_SCALE = 0.1
sys.modules["src.neuronio.neuronio_data_utils"] = stub
actual = importlib.import_module("src.expressive_leaky_memory_neuron")
alias = types.ModuleType("src.expressive_leaky_memory_neuron_v2")
alias.ELM = actual.ELM
sys.modules["src.expressive_leaky_memory_neuron_v2"] = alias

runpy.run_path(
    str(Path(__file__).with_name("evaluate_spieler_published_best_fastpath.py")),
    run_name="__main__",
)
