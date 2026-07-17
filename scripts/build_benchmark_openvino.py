from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from legacy_openvino import LegacyOpenVINOBuilder, reference_inputs  # noqa: E402


ARTIFACT_DIRECTORY = ROOT / "artifacts" / "legacy_openvino"
REFERENCE_PATH = ARTIFACT_DIRECTORY / "legacy_1313_reference.npz"


def benchmark(compiled_model, inputs, warmup: int = 3, iterations: int = 10):
    for _ in range(warmup):
        compiled_model(inputs)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        compiled_model(inputs)
        samples.append(time.perf_counter() - started)
    return {
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.mean(samples),
        "candidates_per_second": len(inputs["board"]) / statistics.median(samples),
    }


def main():
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    reference = np.load(REFERENCE_PATH)
    inputs = reference_inputs(reference)
    expected = reference["output"].reshape(-1)
    builder = LegacyOpenVINOBuilder()
    model = builder.build(batch_size=len(expected), debug=True)
    model_path = ARTIFACT_DIRECTORY / "legacy_1313_batch8.xml"
    ov.save_model(model, model_path, compress_to_fp16=False)

    core = ov.Core()
    results = {"openvino": ov.__version__, "devices": core.available_devices}
    for device in core.available_devices:
        print(f"Compiling {device}...", flush=True)
        try:
            compiled = core.compile_model(model, device)
            inferred = compiled(inputs)
            actual = np.asarray(inferred[0]).reshape(-1)
            error = np.abs(actual - expected)
            intermediate_names = [
                "board_conv1",
                "board_norm1",
                "board_residual1",
                "mino_features",
                "queue_embedding",
                "queue_position",
                "queue_norm",
                "queue_token_order",
                "queue_token_dense1",
                "queue_token_gelu",
                "queue_token_dense2",
                "queue_token_result",
                "queue_block1",
            ]
            intermediate_errors = {}
            for offset, name in enumerate(intermediate_names, start=4):
                ov_value = np.asarray(inferred[offset])
                julia_value = reference[name]
                if name.startswith("board_") and name != "board_features":
                    ov_value = np.transpose(ov_value, (2, 3, 1, 0))
                elif name in {
                    "queue_token_order",
                    "queue_token_dense1",
                    "queue_token_gelu",
                    "queue_token_dense2",
                }:
                    ov_value = np.transpose(ov_value, (2, 1, 0))
                elif name.startswith("queue_") or name == "mino_features":
                    ov_value = np.transpose(ov_value, (2, 1, 0))
                intermediate_errors[name] = float(
                    np.max(np.abs(ov_value - julia_value))
                )
            results[device] = {
                "maximum_absolute_error": float(np.max(error)),
                "mean_absolute_error": float(np.mean(error)),
                "maximum_relative_error": float(
                    np.max(error / np.maximum(np.abs(expected), 1.0e-6))
                ),
                "expected": expected.tolist(),
                "actual": actual.tolist(),
                "board_features_maximum_absolute_error": float(
                    np.max(
                        np.abs(
                            np.asarray(inferred[1]).T - reference["board_features"]
                        )
                    )
                ),
                "queue_features_maximum_absolute_error": float(
                    np.max(
                        np.abs(
                            np.asarray(inferred[2]).T - reference["queue_features"]
                        )
                    )
                ),
                "combined_maximum_absolute_error": float(
                    np.max(np.abs(np.asarray(inferred[3]).T - reference["combined"]))
                ),
                "intermediate_maximum_absolute_errors": intermediate_errors,
                **benchmark(compiled, inputs),
            }
        except Exception as error:
            results[device] = {"error": repr(error)}
        print(device, json.dumps(results[device], ensure_ascii=False), flush=True)

    output_path = ARTIFACT_DIRECTORY / "openvino_reference_benchmark.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
