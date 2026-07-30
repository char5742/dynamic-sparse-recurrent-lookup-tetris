"""Evaluate the public ELM NeuronIO checkpoint on dev-only Hay trials.

This driver is deliberately fail-closed around the data split:

* it opens only shards 00001 through 00020 (global trials 1 through 40);
* trials 1:32 are the fit split and 33:40 are derived validation;
* it never enumerates or opens any later shard.

The compact Hay input is expanded to the official signed 1278-channel layout:
639 excitatory dendritic locations followed by 639 inhibitory locations.
Official NeuronIO evaluation uses overlapping windows and starts every forward
call from zero recurrent state.  Its target-dependent voltage zero-scoring is
also retained for comparability with the public checkpoint's reported metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


FIT_IDS = tuple(range(1, 33))
VALIDATION_IDS = tuple(range(33, 41))
ALLOWED_IDS = FIT_IDS + VALIDATION_IDS
ALLOWED_SHARDS = tuple(range(1, 21))

INPUT_DIM = 1278
DENDRITIC_LOCATIONS = 639
TIME_STEPS = 1500
INPUT_WINDOW_SIZE = 500
BURN_IN_TIME = 150
METRIC_IGNORE_STEPS = 500

SOMA_CLIP_MV = -55.0
SOMA_BIAS_MV = -67.7
SOMA_TRAIN_SCALE = 0.1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _offset_range(offsets: np.ndarray, local_index: int) -> range:
    start = int(offsets[local_index])
    stop = int(offsets[local_index + 1])
    if not 0 <= start <= stop:
        raise RuntimeError("invalid compact ragged offsets")
    return range(start, stop)


def expand_official_signed_input(
    shard: np.lib.npyio.NpzFile, local_index: int
) -> np.ndarray:
    """Return one trial as time x 1278 exact signed official ELM input."""

    destination = np.zeros((TIME_STEPS, INPUT_DIM), dtype=np.float32)

    contact_axon = shard["contact_axon"]
    contact_segment = shard["contact_segment"]
    contact_kind = shard["contact_kind"]
    contact_strength = shard["contact_strength"]
    contacts_by_axon: dict[int, list[tuple[int, np.float32]]] = {}

    contact_range = _offset_range(shard["contact_trial_offset"], local_index)
    for contact in contact_range:
        axon = int(contact_axon[contact])
        segment = int(contact_segment[contact])
        kind = int(contact_kind[contact])
        strength = np.float32(contact_strength[contact])
        if not 2 <= segment <= 640:
            raise RuntimeError("contact targets soma or axon")
        if kind == 1:
            channel = segment - 2
            signed_strength = strength
        elif kind == 2:
            channel = DENDRITIC_LOCATIONS + segment - 2
            signed_strength = -strength
        else:
            raise RuntimeError("contact violates Dale E/I coding")
        if not np.isfinite(strength) or strength < 0:
            raise RuntimeError("invalid contact strength")
        contacts_by_axon.setdefault(axon, []).append((channel, signed_strength))

    event_axon = shard["event_axon"]
    event_time = shard["event_time_bin"]
    event_count = shard["event_count"]
    event_range = _offset_range(shard["event_trial_offset"], local_index)
    for event in event_range:
        time_bin = int(event_time[event])
        axon = int(event_axon[event])
        if not 0 <= time_bin < TIME_STEPS:
            raise RuntimeError("event time is outside the dev1500 trial")
        multiplicity = np.float32(event_count[event])
        if multiplicity <= 0:
            raise RuntimeError("event multiplicity is not positive")
        for channel, signed_strength in contacts_by_axon.get(axon, ()):
            destination[time_bin, channel] += signed_strength * multiplicity

    if not np.isfinite(destination).all():
        raise RuntimeError("expanded input contains non-finite values")
    if np.any(destination[:, :DENDRITIC_LOCATIONS] < 0):
        raise RuntimeError("excitatory input became negative")
    if np.any(destination[:, DENDRITIC_LOCATIONS:] > 0):
        raise RuntimeError("inhibitory input became positive")
    return destination


def official_window_starts() -> tuple[int, ...]:
    num_splits = int(
        2 + (TIME_STEPS - INPUT_WINDOW_SIZE) / (INPUT_WINDOW_SIZE - BURN_IN_TIME)
    )
    return tuple(
        split * (INPUT_WINDOW_SIZE - BURN_IN_TIME)
        for split in range(num_splits)
    )


def forward_official_windows(
    model: torch.nn.Module, input_batch: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Match compute_test_predictions stitching and per-window state reset."""

    batch = input_batch.shape[0]
    spike = np.zeros((batch, TIME_STEPS), dtype=np.float32)
    soma_relative_mv = np.zeros((batch, TIME_STEPS), dtype=np.float32)
    starts = official_window_starts()

    for split_index, start in enumerate(starts):
        stop = start + INPUT_WINDOW_SIZE
        window = input_batch[:, start:min(stop, TIME_STEPS), :]
        if window.shape[1] < INPUT_WINDOW_SIZE:
            pad = np.zeros(
                (batch, INPUT_WINDOW_SIZE - window.shape[1], INPUT_DIM),
                dtype=np.float32,
            )
            window = np.concatenate((window, pad), axis=1)

        # ELM.forward creates zero synapse and memory state on every call.
        # Calling neuronio_eval_forward once per overlapping window therefore
        # matches the official evaluator's recurrent-state reset boundary.
        with torch.no_grad():
            output = model.neuronio_eval_forward(torch.from_numpy(window)).numpy()
        current_spike = output[..., 0]
        current_soma_relative_mv = output[..., 1]

        if split_index == 0:
            fill_stop = min(stop, TIME_STEPS)
            duration = fill_stop
            spike[:, :fill_stop] = current_spike[:, :duration]
            soma_relative_mv[:, :fill_stop] = current_soma_relative_mv[:, :duration]
        elif split_index == len(starts) - 1:
            global_start = start + BURN_IN_TIME
            duration = TIME_STEPS - global_start
            spike[:, global_start:] = current_spike[
                :, BURN_IN_TIME : BURN_IN_TIME + duration
            ]
            soma_relative_mv[:, global_start:] = current_soma_relative_mv[
                :, BURN_IN_TIME : BURN_IN_TIME + duration
            ]
        else:
            global_start = start + BURN_IN_TIME
            spike[:, global_start:stop] = current_spike[:, BURN_IN_TIME:]
            soma_relative_mv[:, global_start:stop] = current_soma_relative_mv[
                :, BURN_IN_TIME:
            ]
    return spike, soma_relative_mv


def load_model(elm_root: Path, checkpoint: Path) -> tuple[torch.nn.Module, dict]:
    model_config_path = checkpoint.parent / "model_config.json"
    with model_config_path.open("r", encoding="utf-8") as stream:
        model_config = json.load(stream)
    sys.path.insert(0, str(elm_root))
    # The official model imports neuronio_data_utils, which imports seaborn
    # solely for optional visualization helpers. Evaluation never calls those
    # helpers, so avoid mutating the environment for this unused dependency.
    try:
        import seaborn  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["seaborn"] = types.ModuleType("seaborn")
    from src.expressive_leaky_memory_neuron import ELM

    torch.manual_seed(0)
    model = ELM(**model_config)
    state = torch.load(str(checkpoint), map_location="cpu")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"non-strict checkpoint load: {incompatible}")
    model.eval()
    return model, model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--elm-root", type=Path, default=Path(r"C:\tmp\elmneuron")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            r"C:\tmp\elmneuron\models\best_elm_neuron"
            r"\neuronio_best_model_state.pt"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"C:\tmp\hd_swsnn_neuron_teacher_final_dev1500_release"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    elm_root = args.elm_root.resolve()
    checkpoint = args.checkpoint.resolve()
    dataset = args.dataset.resolve()
    model, model_config = load_model(elm_root, checkpoint)

    predictions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    targets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    selected_shard_hashes: dict[str, str] = {}
    opened_shards: list[int] = []

    for shard_index in ALLOWED_SHARDS:
        shard_path = dataset / f"neuron_hay_final_{shard_index:05d}.npz"
        selected_shard_hashes[shard_path.name] = sha256_file(shard_path)
        with np.load(shard_path, allow_pickle=False) as shard:
            sample_ids = tuple(int(value) for value in shard["sample_indices"])
            expected_ids = (2 * shard_index - 1, 2 * shard_index)
            if sample_ids != expected_ids:
                raise RuntimeError(
                    f"dev-only shard membership differs: {sample_ids} != {expected_ids}"
                )
            if not set(sample_ids).issubset(ALLOWED_IDS):
                raise RuntimeError("attempted access outside fit32/validation8")
            if set(int(value) for value in shard["split_code"]) != {1}:
                raise RuntimeError("selected shard is not development train")
            if shard["target_voltage"].shape != (TIME_STEPS, 2):
                raise RuntimeError("unexpected target_voltage shape")
            if shard["target_spike"].shape != (TIME_STEPS, 2):
                raise RuntimeError("unexpected target_spike shape")

            input_batch = np.stack(
                [expand_official_signed_input(shard, local) for local in range(2)]
            )
            spike_prediction, soma_prediction = forward_official_windows(
                model, input_batch
            )
            for local, sample_id in enumerate(sample_ids):
                predictions[sample_id] = (
                    spike_prediction[local].copy(),
                    soma_prediction[local].copy(),
                )
                targets[sample_id] = (
                    shard["target_spike"][:, local].astype(np.float32, copy=True),
                    shard["target_voltage"][:, local].astype(np.float32, copy=True),
                )
        opened_shards.append(shard_index)

    if tuple(sorted(predictions)) != ALLOWED_IDS:
        raise RuntimeError("did not forward exactly fit32/validation8")
    if tuple(opened_shards) != ALLOWED_SHARDS:
        raise RuntimeError("opened shard set differs from dev-only allowlist")

    validation_spike_target = np.stack(
        [targets[sample_id][0] for sample_id in VALIDATION_IDS]
    )
    validation_spike_prediction = np.stack(
        [predictions[sample_id][0] for sample_id in VALIDATION_IDS]
    )
    validation_voltage_clipped_mv = np.minimum(
        np.stack([targets[sample_id][1] for sample_id in VALIDATION_IDS]),
        SOMA_CLIP_MV,
    )
    validation_voltage_target_relative_mv = (
        validation_voltage_clipped_mv - SOMA_BIAS_MV
    )
    validation_voltage_prediction_relative_mv = np.stack(
        [predictions[sample_id][1] for sample_id in VALIDATION_IDS]
    )

    # Preserve the public evaluator's target-distribution alignment.  It is
    # performed after stitching and before the post-500 ms metric slice.
    target_mean = float(validation_voltage_target_relative_mv.mean())
    target_std = float(validation_voltage_target_relative_mv.std())
    prediction_mean = float(validation_voltage_prediction_relative_mv.mean())
    prediction_std = float(validation_voltage_prediction_relative_mv.std())
    if prediction_std <= 0 or target_std <= 0:
        raise RuntimeError("voltage alignment encountered zero variance")
    aligned_prediction_relative_mv = (
        validation_voltage_prediction_relative_mv - prediction_mean
    ) / prediction_std
    aligned_prediction_relative_mv = (
        target_std * aligned_prediction_relative_mv + target_mean
    )
    aligned_prediction_physical_mv = (
        aligned_prediction_relative_mv + SOMA_BIAS_MV
    )

    metric_slice = slice(METRIC_IGNORE_STEPS, None)
    spike_target_metric = validation_spike_target[:, metric_slice]
    spike_prediction_metric = validation_spike_prediction[:, metric_slice]
    voltage_target_metric = validation_voltage_clipped_mv[:, metric_slice]
    voltage_prediction_metric = aligned_prediction_physical_mv[:, metric_slice]

    spike_auroc = roc_auc_score(
        spike_target_metric.ravel(), spike_prediction_metric.ravel()
    )
    clipped_physical_voltage_rmse_mv = float(
        np.sqrt(
            np.mean(
                np.square(
                    voltage_prediction_metric.astype(np.float64)
                    - voltage_target_metric.astype(np.float64)
                )
            )
        )
    )

    source_files = {
        "expressive_leaky_memory_neuron.py": (
            elm_root / "src" / "expressive_leaky_memory_neuron.py"
        ),
        "neuronio_eval_utils.py": (
            elm_root / "src" / "neuronio" / "neuronio_eval_utils.py"
        ),
        "neuronio_data_loader.py": (
            elm_root / "src" / "neuronio" / "neuronio_data_loader.py"
        ),
    }
    result = {
        "schema": "public_elm_checkpoint.dev40_validation.v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_config": model_config,
        "model_config_sha256": sha256_file(
            checkpoint.parent / "model_config.json"
        ),
        "official_source_sha256": {
            name: sha256_file(path) for name, path in source_files.items()
        },
        "driver_sha256": sha256_file(Path(__file__).resolve()),
        "dataset": str(dataset),
        "access_policy": {
            "opened_shards": opened_shards,
            "opened_global_ids": list(sorted(predictions)),
            "fit_ids": list(FIT_IDS),
            "validation_ids": list(VALIDATION_IDS),
            "later_shards_opened": False,
        },
        "selected_shards_sha256": selected_shard_hashes,
        "selected_shards_digest_sha256": canonical_json_sha256(
            selected_shard_hashes
        ),
        "contract": {
            "input_layout": "E[segments 2:640], then I[segments 2:640]",
            "input_semantics": "E:+strength*event_count, I:-strength*event_count",
            "input_normalization": "none",
            "input_window_size": INPUT_WINDOW_SIZE,
            "window_burn_in": BURN_IN_TIME,
            "window_starts_zero_based": list(official_window_starts()),
            "state_reset": "zero state at every overlapping window",
            "soma_training_target": (
                "(min(voltage_mv,-55)-(-67.7))*0.1"
            ),
            "soma_prediction_to_relative_mv": (
                "raw_model_output/0.1 via neuronio_eval_forward"
            ),
            "official_voltage_alignment": (
                "prediction zero-scored to clipped target mean/std"
            ),
            "physical_prediction_after_alignment": (
                "aligned_relative_mv + (-67.7)"
            ),
            "metric_ignore_first_steps": METRIC_IGNORE_STEPS,
        },
        "validation": {
            "trials": len(VALIDATION_IDS),
            "metric_bins": int(spike_target_metric.size),
            "spike_positive_bins": int(spike_target_metric.sum()),
            "spike_auroc": float(spike_auroc),
            "clipped_physical_voltage_rmse_mv": (
                clipped_physical_voltage_rmse_mv
            ),
        },
        "forwarded": {
            "fit_trials": len(FIT_IDS),
            "validation_trials": len(VALIDATION_IDS),
            "total_trials": len(predictions),
        },
        "torch_version": torch.__version__,
        "device": "cpu",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
