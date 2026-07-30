"""M1000 fit256 mainline: r3/e31 weights, fit-only scaling, fresh Adam."""

from __future__ import annotations

import numpy as np

import run_paper_elm_torch_cpu_fit256_balanced_jit as _balanced


core = _balanced.core
runner = _balanced.runner
fit256 = _balanced._entry._fit256
_previous_load_bridge = core._load_bridge


def _m1000_fit256_bridge(path):
    bridge, metadata = _previous_load_bridge(path)
    bridge = {
        name: np.asarray(value)
        for name, value in bridge.items()
    }
    bridge["nmda_mean"] = fit256._NMDA_MEAN.copy()
    bridge["nmda_scale"] = fit256._NMDA_SCALE.copy()
    bridge = runner._reset_adam_bridge(bridge)
    metadata = dict(metadata)
    metadata["manifest_sha256"] = fit256._MANIFEST_SHA256
    metadata["update_index"] = 0
    metadata["m1000_fit256_mainline"] = {
        "weight_initialization": "selected Julia r3/e31",
        "nmda_normalizer": "fit256-only",
        "optimizer": "fresh Julia-formula Adam",
        "cosine_schedule": "reset at update 1",
        "positive_aware": True,
        "spike_loss": "class-balanced BCE",
        "heldout_opened": False,
    }
    return bridge, metadata


core._load_bridge = _m1000_fit256_bridge


if __name__ == "__main__":
    runner.main()
