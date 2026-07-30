#!/usr/bin/env python3
"""Critical-path evaluation/export of Spieler's published best ELM-v2.

The script is deliberately fit/derived-validation only.  It verifies the
pinned public files, loads the PyTorch state dict strictly, evaluates the
project's 1500-bin development teacher with the paper overlap/reset protocol,
fits four NMDA readout rows only on fit memories, and exports a non-pickle NPZ
for the Julia importer.  It never opens shards containing held-out trials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


PINNED_COMMIT = "52e68a6d39523ac6613a586699b116e8e606dda3"
PINNED_CONFIG_SHA256 = (
    "3c54bb31199cdd4c814ffc2965827c2c7cc62802aa1330ae32806e9b5377b51a"
)
PINNED_CHECKPOINT_SHA256 = (
    "ee1252616cd2ad7dd60786e304a56e6d9d13b4f85ce9974fbef909efa9f43812"
)
EXPECTED_CONFIG = {
    "input_to_synapse_routing": "neuronio_routing",
    "learn_memory_tau": False,
    "memory_tau_max": 150.0,
    "memory_tau_min": 1.0,
    "mlp_activation": "silu",
    "num_branch": 45,
    "num_input": 1278,
    "num_memory": 100,
    "num_output": 2,
    "num_synapse_per_branch": 100,
}
WINDOW = 500
OVERLAP = 150
METRIC_FIRST_ZERO_BASED = 500
SOMA_CLIP_MV = -55.0
SOMA_BIAS_MV = -67.7
SOMA_SCALE = 0.1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elm-repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-npz", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.0e-4)
    return parser.parse_args()


def _load_model(repo: Path):
    config_path = repo / "models" / "best_elm_neuron" / "model_config.json"
    checkpoint_path = (
        repo / "models" / "best_elm_neuron" / "neuronio_best_model_state.pt"
    )
    if _sha256(config_path) != PINNED_CONFIG_SHA256:
        raise RuntimeError("pinned public model config SHA-256 differs")
    if _sha256(checkpoint_path) != PINNED_CHECKPOINT_SHA256:
        raise RuntimeError("pinned public checkpoint SHA-256 differs")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config != EXPECTED_CONFIG:
        raise RuntimeError(f"pinned public model config differs: {config}")
    sys.path.insert(0, str(repo))
    from src.expressive_leaky_memory_neuron_v2 import ELM

    model = ELM(**config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, state, config


def _offset_range(offsets: np.ndarray, local_index: int) -> range:
    return range(int(offsets[local_index]), int(offsets[local_index + 1]))


def _reconstruct_trial(shard, local_index: int) -> np.ndarray:
    result = np.zeros((1500, 1278), dtype=np.float32)
    contacts_by_axon: dict[int, list[tuple[int, np.float32]]] = {}
    contact_range = _offset_range(shard["contact_trial_offset"], local_index)
    for contact in contact_range:
        segment = int(shard["contact_segment"][contact])
        kind = int(shard["contact_kind"][contact])
        if not 2 <= segment <= 640:
            raise RuntimeError("contact is not on a legal Hay dendrite")
        location_zero = segment - 2
        if kind == 1:
            channel = location_zero
            sign = np.float32(1.0)
        elif kind == 2:
            channel = 639 + location_zero
            sign = np.float32(-1.0)
        else:
            raise RuntimeError("contact violates E/I coding")
        axon = int(shard["contact_axon"][contact])
        strength = np.float32(shard["contact_strength"][contact]) * sign
        contacts_by_axon.setdefault(axon, []).append((channel, strength))
    event_range = _offset_range(shard["event_trial_offset"], local_index)
    previous = (-1, -1)
    for event in event_range:
        time = int(shard["event_time_bin"][event])
        axon = int(shard["event_axon"][event])
        if (time, axon) < previous:
            raise RuntimeError("events violate time_then_axon order")
        previous = (time, axon)
        count = np.float32(shard["event_count"][event])
        for channel, strength in contacts_by_axon.get(axon, ()):
            result[time, channel] += strength * count
    if np.any(result[:, :639] < 0) or np.any(result[:, 639:] > 0):
        raise RuntimeError("signed 1278-input reconstruction differs")
    return result


def _load_fit_validation(dataset: Path):
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    validation_ids = {int(x) for x in manifest["validation_from_train_indices"]}
    records = {}
    # 24 shards x 2 trials; train IDs are exactly 1:40.  Stop before held-out.
    for shard_path in sorted(dataset.glob("neuron_hay_final_*.npz")):
        with np.load(shard_path, allow_pickle=False) as shard:
            sample_ids = [int(x) for x in shard["sample_indices"]]
            for local_index, sample_id in enumerate(sample_ids):
                if sample_id > 40:
                    continue
                if int(shard["split_code"][local_index]) != 1:
                    raise RuntimeError("fit/derived-validation member is not train-coded")
                records[sample_id] = {
                    "input": _reconstruct_trial(shard, local_index),
                    "voltage": np.asarray(
                        shard["target_voltage"][:, local_index], dtype=np.float32
                    ),
                    "spike": np.asarray(
                        shard["target_spike"][:, local_index], dtype=np.float32
                    ),
                    "nmda": np.asarray(
                        shard["target_nmda"][:, :, local_index].T, dtype=np.float32
                    ),
                }
    if set(records) != set(range(1, 41)):
        raise RuntimeError("fit/derived-validation membership differs")
    fit_ids = sorted(set(records) - validation_ids)
    return records, fit_ids, sorted(validation_ids), manifest


@torch.inference_mode()
def _predict_retained(model, input_array: np.ndarray):
    logits, voltage_coordinate, memories = [], [], []
    global_indices = []
    steps = input_array.shape[0]
    for window_index, start in enumerate(range(0, steps, WINDOW - OVERLAP)):
        actual = min(WINDOW, steps - start)
        if actual <= 0:
            break
        window = torch.from_numpy(input_array[start : start + actual]).unsqueeze(0)
        outputs, _, memory = model.neuronio_viz_forward(window)
        keep = 0 if window_index == 0 else OVERLAP
        metric_first = max(start + keep, METRIC_FIRST_ZERO_BASED)
        if metric_first >= start + actual:
            continue
        local_first = metric_first - start
        raw = model.forward(window)
        logits.append(raw[0, local_first:actual, 0].cpu().numpy())
        voltage_coordinate.append(raw[0, local_first:actual, 1].cpu().numpy())
        memories.append(memory[0, local_first:actual].cpu().numpy())
        global_indices.extend(range(metric_first, start + actual))
        if start + actual == steps:
            break
    indices = np.asarray(global_indices, dtype=np.int64)
    if indices.shape != (1000,) or not np.array_equal(indices, np.arange(500, 1500)):
        raise RuntimeError("paper overlap/reset retained-bin contract differs")
    return (
        np.concatenate(logits),
        np.concatenate(voltage_coordinate),
        np.concatenate(memories),
        indices,
    )


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    first = 0
    while first < scores.size:
        last = first + 1
        while last < scores.size and sorted_scores[last] == sorted_scores[first]:
            last += 1
        ranks[order[first:last]] = 0.5 * (first + 1 + last)
        first = last
    positive = labels != 0
    n_pos = int(positive.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    return float(
        (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def _evaluate_split(records, ids, predictions, nmda_weight, nmda_bias, nmda_scale):
    voltage_sq = 0.0
    count = 0
    all_spikes, all_logits = [], []
    nmda_sq = np.zeros(4, dtype=np.float64)
    for sample_id in ids:
        logits, voltage_coordinate, memory, indices = predictions[sample_id]
        target_voltage = np.minimum(records[sample_id]["voltage"][indices], SOMA_CLIP_MV)
        predicted_voltage = voltage_coordinate / SOMA_SCALE + SOMA_BIAS_MV
        voltage_sq += np.square(
            predicted_voltage.astype(np.float64)
            - target_voltage.astype(np.float64)
        ).sum()
        count += indices.size
        all_spikes.append(records[sample_id]["spike"][indices])
        all_logits.append(logits)
        predicted_nmda = memory @ nmda_weight.T + nmda_bias
        target_nmda = records[sample_id]["nmda"][indices]
        normalized_error = (predicted_nmda - target_nmda) / nmda_scale
        nmda_sq += np.square(normalized_error.astype(np.float64)).sum(axis=0)
    labels = np.concatenate(all_spikes)
    scores = np.concatenate(all_logits)
    return {
        "trial_ids": ids,
        "observations": count,
        "clip_voltage_rmse_mv": float(math.sqrt(voltage_sq / count)),
        "spike_auroc": _auc(labels, scores),
        "spike_positive_count": int(np.count_nonzero(labels)),
        "normalized_nmda_rmse_by_region": [
            float(math.sqrt(value / count)) for value in nmda_sq
        ],
    }


def _export_npz(
    path: Path,
    state,
    nmda_weight,
    nmda_bias,
    nmda_mean,
    nmda_scale,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "proto_w_s": state["_proto_w_s"].cpu().numpy(),
        "proto_tau_m": state["_proto_tau_m"].cpu().numpy(),
        "input_to_synapse_indices_zero_based": state[
            "input_to_synapse_indices"
        ].cpu().numpy(),
        "valid_indices_mask": state["valid_indices_mask"].cpu().numpy(),
        "input_weight": state["mlp.network.0.weight"].cpu().numpy(),
        "input_bias": state["mlp.network.0.bias"].cpu().numpy(),
        "memory_weight": state["mlp.network.2.weight"].cpu().numpy(),
        "memory_bias": state["mlp.network.2.bias"].cpu().numpy(),
        "output_weight": state["w_y.weight"].cpu().numpy(),
        "output_bias": state["w_y.bias"].cpu().numpy(),
        "nmda_weight": np.asarray(nmda_weight, dtype=np.float32),
        "nmda_bias": np.asarray(nmda_bias, dtype=np.float32),
        "nmda_mean": np.asarray(nmda_mean, dtype=np.float32),
        "nmda_scale": np.asarray(nmda_scale, dtype=np.float32),
        "source_config_sha256_ascii": np.frombuffer(
            PINNED_CONFIG_SHA256.encode("ascii"), dtype=np.uint8
        ),
        "source_checkpoint_sha256_ascii": np.frombuffer(
            PINNED_CHECKPOINT_SHA256.encode("ascii"), dtype=np.uint8
        ),
    }
    np.savez(path, **arrays)


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(max(1, min(20, torch.get_num_threads())))
    model, state, config = _load_model(args.elm_repo.resolve())
    records, fit_ids, validation_ids, manifest = _load_fit_validation(
        args.dataset.resolve()
    )
    predictions = {}
    for sample_id in fit_ids + validation_ids:
        predictions[sample_id] = _predict_retained(
            model, records[sample_id]["input"]
        )

    fit_memory = np.concatenate([predictions[i][2] for i in fit_ids])
    fit_nmda = np.concatenate(
        [records[i]["nmda"][predictions[i][3]] for i in fit_ids]
    )
    nmda_mean = fit_nmda.mean(axis=0, dtype=np.float64)
    nmda_scale = fit_nmda.std(axis=0, dtype=np.float64)
    nmda_scale = np.maximum(nmda_scale, 1.0e-5)
    design = np.concatenate(
        [fit_memory.astype(np.float64), np.ones((fit_memory.shape[0], 1))],
        axis=1,
    )
    gram = design.T @ design
    gram[:-1, :-1] += args.ridge * np.eye(fit_memory.shape[1])
    solution = np.linalg.solve(gram, design.T @ fit_nmda.astype(np.float64))
    nmda_weight = solution[:-1].T.astype(np.float32)
    nmda_bias = solution[-1].astype(np.float32)

    fit_metrics = _evaluate_split(
        records,
        fit_ids,
        predictions,
        nmda_weight,
        nmda_bias,
        nmda_scale,
    )
    validation_metrics = _evaluate_split(
        records,
        validation_ids,
        predictions,
        nmda_weight,
        nmda_bias,
        nmda_scale,
    )
    _export_npz(
        args.export_npz.resolve(),
        state,
        nmda_weight,
        nmda_bias,
        nmda_mean,
        nmda_scale,
    )
    report = {
        "schema": "hd_swsnn.spieler_published_best.fastpath.v1",
        "source": {
            "commit": PINNED_COMMIT,
            "config_sha256": PINNED_CONFIG_SHA256,
            "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
            "config": config,
            "checkpoint_shapes": {
                key: list(value.shape) for key, value in state.items()
            },
        },
        "compatibility": {
            "direct_m1000_load": False,
            "reason": (
                "published checkpoint is M=100/hidden=200/output=2/tau=1..150; "
                "target is M=1000/hidden=2000/output=6/tau=0.1..300"
            ),
            "exact_zero_invariant_embedding_possible": True,
            "embedding": (
                "copy public 100 memories and 200 hidden units, copy their "
                "stored proto_tau_m, zero the remaining 900/1800 units, and "
                "append fit-only four-row NMDA readout"
            ),
        },
        "evaluation": {
            "protocol": (
                "500-bin windows, stride 350, reset each window, discard 150 "
                "overlap bins, metric interval 501:1500"
            ),
            "heldout_opened": False,
            "fit": fit_metrics,
            "derived_validation": validation_metrics,
            "ridge": args.ridge,
        },
        "dataset": {
            "teacher_contract_sha256": manifest["teacher_contract_sha256"],
            "completed_trials": manifest["completed_trials"],
            "fit_trials": len(fit_ids),
            "derived_validation_trials": len(validation_ids),
        },
        "export_npz": str(args.export_npz.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
