"""Continue M256 fit256 training with a non-cycling fixed 2e-4 LR."""

import run_paper_elm_torch_cpu_fit256_balanced_jit as _mainline


core = _mainline.core
runner = _mainline.runner


def _fixed_continuation_learning_rate(update_index: int) -> float:
    if update_index < 65:
        raise ValueError(
            "fixed continuation is valid only after the adopted u64 checkpoint"
        )
    return 2.0e-4


core._learning_rate = _fixed_continuation_learning_rate


if __name__ == "__main__":
    runner.main()
