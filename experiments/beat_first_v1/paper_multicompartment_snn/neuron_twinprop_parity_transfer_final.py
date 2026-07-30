#!/usr/bin/env python3
"""Canonical wrapper for the official Hay/NEURON parity transfer runner.

Julia's NPZ writer stores UTF-8 metadata as one-dimensional ``uint8`` arrays.
The base implementation also accepts NumPy scalar strings.  This wrapper
installs the dual-format decoder and is the production entry point.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import neuron_twinprop_parity_transfer as implementation


def _metadata_text(value: np.ndarray | Any) -> str:
    array = np.asarray(value)
    if array.ndim == 1 and array.dtype == np.uint8:
        return array.tobytes().decode("utf-8")
    scalar = array.reshape(()).item()
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    return str(scalar)


implementation._scalar_text = _metadata_text

run_transfer = implementation.run_transfer
build_argument_parser = implementation.build_argument_parser


def main() -> int:
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
