"""Fit256-safe entry point using the TorchScript-specialized model."""

import train_paper_elm_torch_cpu as _core
from paper_elm_torchscript_model import make_model as _make_model

_core.FAST_MEMORY = 256
_core.FAST_HIDDEN = 512
_core.FULL_MEMORY = 1_000
_core.FULL_HIDDEN = 2_000
_core._make_model = _make_model

import run_paper_elm_torch_cpu_fit256 as _fit256


if __name__ == "__main__":
    _fit256._runner.main()
