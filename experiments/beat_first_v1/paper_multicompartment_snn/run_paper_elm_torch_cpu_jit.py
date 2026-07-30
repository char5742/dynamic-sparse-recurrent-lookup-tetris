"""M256/M1000 entry point using the TorchScript-specialized model."""

import train_paper_elm_torch_cpu as _core
from paper_elm_torchscript_model import make_model as _make_model

_core.FAST_MEMORY = 256
_core.FAST_HIDDEN = 512
_core.FULL_MEMORY = 1_000
_core.FULL_HIDDEN = 2_000
_core._make_model = _make_model

import train_paper_elm_torch_cpu_variable as _runner


if __name__ == "__main__":
    _runner.main()
