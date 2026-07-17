from __future__ import annotations

import argparse
from pathlib import Path

from contract import atomic_write_json, contract_identity, load_contract


def eligibility_document() -> dict[str, object]:
    contract = load_contract()
    roles = contract["data_roles"]
    training = roles["training_seeds"]
    calibration = roles["calibration_seeds"]
    pieces = roles["sample_pieces"]
    return {
        "status": "r1_design_eligibility_complete",
        "experiment": contract["experiment_id"],
        **contract_identity(),
        "training_seed_ids": training,
        "calibration_seed_ids": calibration,
        "sample_piece_indices": pieces,
        "planned_training_states": len(training) * len(pieces),
        "planned_calibration_states": len(calibration) * len(pieces),
        "minimum_training_states": roles["minimum_training_states"],
        "minimum_calibration_states": roles["minimum_calibration_states"],
        "conditional_development_seed_ids": roles["conditional_development_seeds"],
        "forbidden_development_seed_ids": roles["forbidden_development_seeds"],
        "forbidden_validation_seed_ids": roles["forbidden_validation_seeds"],
        "forbidden_sealed_test_seed_range": [
            roles["forbidden_sealed_test_first"],
            roles["forbidden_sealed_test_last"],
        ],
        "real_data_loaded": False,
        "model_or_checkpoint_loaded": False,
        "game_run": False,
        "development_seed_loaded": False,
        "validation_seed_loaded": False,
        "sealed_test_seed_loaded": False,
        "existing_c10_c13_q1_dataset_loaded": False,
    }


def validate_eligibility(document: dict[str, object]) -> None:
    expected = eligibility_document()
    if document != expected:
        differing = sorted(
            key for key in set(document) | set(expected) if document.get(key) != expected.get(key)
        )
        raise ValueError("R1 design eligibility differs from contract: " + ", ".join(differing))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze R1 seed roles without opening any model, game, or dataset"
    )
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    document = eligibility_document()
    validate_eligibility(document)
    if args.self_check:
        if args.output is not None:
            parser.error("output is not accepted with --self-check")
        return
    if args.output is None:
        parser.error("output is required")
    atomic_write_json(args.output, document)


if __name__ == "__main__":
    main()
