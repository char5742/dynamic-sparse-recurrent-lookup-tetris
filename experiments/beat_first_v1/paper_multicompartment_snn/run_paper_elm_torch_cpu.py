"""Stable entry point for the M256/M1000 Paper-ELM CPU runner."""

import train_paper_elm_torch_cpu as _core

# The release implementation fixes these at M1000.  The variable orchestrator
# sets MEMORY/HIDDEN before each model is constructed and scripted.
_core.FAST_MEMORY = 256
_core.FAST_HIDDEN = 512
_core.FULL_MEMORY = 1_000
_core.FULL_HIDDEN = 2_000

import train_paper_elm_torch_cpu_variable as _runner


if __name__ == "__main__":
    _runner.main()
