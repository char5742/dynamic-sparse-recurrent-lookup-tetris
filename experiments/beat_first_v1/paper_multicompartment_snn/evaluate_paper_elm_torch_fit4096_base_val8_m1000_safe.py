"""CLI adapter retaining the fit1024 warmstart provenance for validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import evaluate_paper_elm_torch_fit4096_base_val8_m1000 as evaluator


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--augmentation-dataset",
        type=Path,
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    return parser.parse_args()


evaluator._parse_args = _parse_args


if __name__ == "__main__":
    evaluator.main()
