#!/usr/bin/env python3
"""Dependency-light entry point for the pinned Spieler fastpath evaluator."""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path


stub = types.ModuleType("src.neuronio.neuronio_data_utils")
stub.DEFAULT_Y_TRAIN_SOMA_SCALE = 0.1
sys.modules["src.neuronio.neuronio_data_utils"] = stub
runpy.run_path(
    str(Path(__file__).with_name("evaluate_spieler_published_best_fastpath.py")),
    run_name="__main__",
)
