#!/usr/bin/env python3
"""Fit-only linear readout ceiling for the pinned public ELM-v1 memories."""

from __future__ import annotations

import argparse
import importlib
import json
import runpy
import sys
import types
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elm-repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.0e-4)
    return parser.parse_args()


def _reconstruct(shard, local_index: int) -> np.ndarray:
    result = np.zeros((1500, 1278), dtype=np.float32)
    contacts_by_axon = {}
    offsets = shard["contact_trial_offset"]
    for contact in range(int(offsets[local_index]), int(offsets[local_index + 1])):
        segment = int(shard["contact_segment"][contact])
        kind = int(shard["contact_kind"][contact])
        channel = segment - 2 + (0 if kind == 1 else 639)
        strength = np.float32(shard["contact_strength"][contact])
        if kind == 2:
            strength = -strength
        contacts_by_axon.setdefault(
            int(shard["contact_axon"][contact]), []
        ).append((channel, strength))
    offsets = shard["event_trial_offset"]
    for event in range(int(offsets[local_index]), int(offsets[local_index + 1])):
        time = int(shard["event_time_bin"][event])
        axon = int(shard["event_axon"][event])
        count = np.float32(shard["event_count"][event])
        for channel, strength in contacts_by_axon.get(axon, ()):
            result[time, channel] += strength * count
    return result


def _ridge(design: np.ndarray, target: np.ndarray, alpha: float):
    augmented = np.concatenate(
        [design.astype(np.float64), np.ones((design.shape[0], 1))], axis=1
    )
    gram = augmented.T @ augmented
    gram[:-1, :-1] += alpha * np.eye(design.shape[1])
    solution = np.linalg.solve(gram, augmented.T @ target.astype(np.float64))
    return solution[:-1], solution[-1]


def main():
    args = _args()
    sys.path.insert(0, str(args.elm_repo.resolve()))
    stub = types.ModuleType("src.neuronio.neuronio_data_utils")
    stub.DEFAULT_Y_TRAIN_SOMA_SCALE = 0.1
    sys.modules["src.neuronio.neuronio_data_utils"] = stub
    actual = importlib.import_module("src.expressive_leaky_memory_neuron")
    alias = types.ModuleType("src.expressive_leaky_memory_neuron_v2")
    alias.ELM = actual.ELM
    sys.modules["src.expressive_leaky_memory_neuron_v2"] = alias
    library = runpy.run_path(
        str(Path(__file__).with_name("evaluate_spieler_published_best_fastpath.py")),
        run_name="spieler_fastpath_library",
    )
    library["_load_fit_validation"].__globals__["_reconstruct_trial"] = _reconstruct
    model, _, _ = library["_load_model"](args.elm_repo.resolve())
    records, fit_ids, validation_ids, _ = library["_load_fit_validation"](
        args.dataset.resolve()
    )
    predictions = {
        sample_id: library["_predict_retained"](
            model, records[sample_id]["input"]
        )
        for sample_id in fit_ids + validation_ids
    }
    fit_memory = np.concatenate([predictions[i][2] for i in fit_ids])
    fit_voltage = np.concatenate(
        [
            np.minimum(
                records[i]["voltage"][predictions[i][3]],
                library["SOMA_CLIP_MV"],
            )
            for i in fit_ids
        ]
    )
    fit_spike = np.concatenate(
        [records[i]["spike"][predictions[i][3]] for i in fit_ids]
    ).astype(np.int8)
    voltage_weight, voltage_bias = _ridge(
        fit_memory, fit_voltage, args.ridge
    )
    logistic = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    )
    logistic.fit(fit_memory, fit_spike)

    def metrics(ids):
        memory = np.concatenate([predictions[i][2] for i in ids])
        voltage = np.concatenate(
            [
                np.minimum(
                    records[i]["voltage"][predictions[i][3]],
                    library["SOMA_CLIP_MV"],
                )
                for i in ids
            ]
        )
        spike = np.concatenate(
            [records[i]["spike"][predictions[i][3]] for i in ids]
        ).astype(np.int8)
        predicted_voltage = memory @ voltage_weight + voltage_bias
        score = logistic.decision_function(memory)
        return {
            "trial_ids": ids,
            "observations": int(memory.shape[0]),
            "clip_voltage_rmse_mv": float(
                np.sqrt(np.mean(np.square(predicted_voltage - voltage)))
            ),
            "spike_auroc": library["_auc"](spike, score),
            "spike_positive_count": int(spike.sum()),
        }

    report = {
        "schema": "hd_swsnn.spieler_public_m100.readout_ceiling.v1",
        "heldout_opened": False,
        "features": "exact public ELM-v1 M=100 recurrent memory",
        "readouts": {
            "voltage": f"fit-only ridge alpha={args.ridge}",
            "spike": "fit-only balanced LogisticRegression C=1",
        },
        "fit": metrics(fit_ids),
        "derived_validation": metrics(validation_ids),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
