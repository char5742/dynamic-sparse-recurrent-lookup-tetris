#!/usr/bin/env python3
"""Protocol and resume tests for the canonical final Hay teacher."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
import unittest

import numpy as np

import neuron_hay_teacher as control
import neuron_hay_teacher_final as final


MODELDB_ROOT = pathlib.Path(
    os.environ.get("HAY_MODELDB_ROOT", "/mnt/c/tmp/hay_modeldb_139653")
)


def tracked_status() -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(MODELDB_ROOT),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_config(
    *,
    train_trials: int = 1,
    test_trials: int = 0,
    duration_ms: int = 8,
    shard_size: int = 1,
) -> final.FinalTeacherConfig:
    return final.FinalTeacherConfig(
        preset="test",
        train_trials=train_trials,
        validation_trials_from_train=0,
        test_trials=test_trials,
        duration_ms=duration_ms,
        axons=8,
        mean_contacts_per_axon=4.0,
        exact_contacts_per_axon=False,
        diagnostic_segments=8,
        diagnostic_stride_bins=2,
        shard_size=shard_size,
        excitatory_fraction=0.75,
        rate_min_hz=5.0,
        rate_max_hz=250.0,
        seed=12345,
        dt_ms=0.025,
        sample_dt_ms=1.0,
        v_init_mv=-80.0,
        celsius=34.0,
        spike_threshold_mv=-20.0,
        ca_event_cai_mm=2.0e-4,
        ca_event_voltage_mv=-30.0,
        store_dense_axon_events=True,
        connectivity_interpretation_acknowledged=False,
    )


@unittest.skipUnless(
    (MODELDB_ROOT / "x86_64" / "libnrnmech.so").is_file(),
    "compiled Hay ModelDB mechanisms are unavailable",
)
class FinalHayTeacherTest(unittest.TestCase):
    def test_production_preset_matches_public_counts(self) -> None:
        preset = final.FINAL_PRESETS["production"]
        self.assertEqual(preset.train_trials, 50_000)
        self.assertEqual(preset.validation_trials_from_train, 0)
        self.assertEqual(preset.test_trials, 2_000)
        self.assertEqual(preset.duration_ms, 10_000)
        self.assertEqual(preset.axons, 400)
        self.assertTrue(preset.exact_contacts_per_axon)
        self.assertEqual(preset.default_excitatory_fraction, 0.5)
        self.assertEqual(final.STRENGTH_MINIMUM, 0.0)
        self.assertEqual(final.STRENGTH_MAXIMUM, 1.0)

    def test_production_connectivity_interpretation_is_fail_closed(self) -> None:
        parser = final.build_argument_parser()
        unacknowledged = final._config_from_arguments(parser.parse_args([
            "--output", "/tmp/final-teacher-test",
            "--preset", "production",
        ]))
        with self.assertRaisesRegex(ValueError, "acknowledge"):
            unacknowledged.validate()
        acknowledged = final._config_from_arguments(parser.parse_args([
            "--output", "/tmp/final-teacher-test",
            "--preset", "production",
            "--acknowledge-8000-contact-interpretation",
        ]))
        acknowledged.validate()
        self.assertEqual(acknowledged.axons, 400)
        self.assertEqual(acknowledged.excitatory_fraction, 0.5)
        self.assertTrue(acknowledged.exact_contacts_per_axon)
        self.assertEqual(
            acknowledged.axons * acknowledged.mean_contacts_per_axon,
            8_000,
        )

    def test_rate_and_continuous_location_protocol(self) -> None:
        config = test_config(duration_ms=100)
        teacher = control.instantiate_teacher(MODELDB_ROOT, 8)
        slots, _ = final._section_location_slots(teacher)
        protocol = final._draw_final_protocol(
            teacher, slots, config, global_trial=1
        )
        self.assertGreaterEqual(
            protocol["rate_window_ms"],
            final.PAPER_RATE_TIMESCALE_MIN_MS,
        )
        self.assertLessEqual(
            protocol["rate_window_ms"],
            final.PAPER_RATE_TIMESCALE_MAX_MS,
        )
        self.assertGreaterEqual(
            protocol["rate_sigma_ms"],
            final.PAPER_RATE_TIMESCALE_MIN_MS,
        )
        self.assertLessEqual(
            protocol["rate_sigma_ms"],
            final.PAPER_RATE_TIMESCALE_MAX_MS,
        )
        self.assertTrue(np.all(protocol["contact_strength"] >= 0.0))
        self.assertTrue(np.all(protocol["contact_strength"] < 1.0))
        for kind in (control.EXCITATORY, control.INHIBITORY):
            selected = protocol["contact_location_slot"][
                protocol["contact_kind"] == kind
            ]
            self.assertEqual(len(selected), len(np.unique(selected)))
        self.assertTrue(np.all(protocol["contact_x"] > 0.0))
        self.assertTrue(np.all(protocol["contact_x"] < 1.0))
        repeated = final._draw_final_protocol(
            teacher, slots, config, global_trial=1
        )
        np.testing.assert_array_equal(
            protocol["axon_events"], repeated["axon_events"]
        )
        np.testing.assert_array_equal(
            protocol["contact_location_slot"],
            repeated["contact_location_slot"],
        )

    def test_final_shard_and_hash_verified_resume(self) -> None:
        config = test_config(train_trials=1, test_trials=1, shard_size=1)
        before = tracked_status()
        with tempfile.TemporaryDirectory() as temporary:
            first = final.generate_final_dataset(
                MODELDB_ROOT, temporary, config, workers=1, resume=True
            )
            self.assertEqual(first["completion_state"], "complete")
            self.assertEqual(first["completed_trials"], 2)
            self.assertEqual(len(first["shards"]), 2)
            mtimes = {
                record["path"]: (
                    pathlib.Path(temporary) / record["path"]
                ).stat().st_mtime_ns
                for record in first["shards"]
            }
            second = final.generate_final_dataset(
                MODELDB_ROOT, temporary, config, workers=1, resume=True
            )
            self.assertEqual(
                first["teacher_contract_sha256"],
                second["teacher_contract_sha256"],
            )
            self.assertEqual(
                mtimes,
                {
                    record["path"]: (
                        pathlib.Path(temporary) / record["path"]
                    ).stat().st_mtime_ns
                    for record in second["shards"]
                },
            )
            manifest = json.loads(
                (pathlib.Path(temporary) / "manifest.json").read_text("utf-8")
            )
            self.assertEqual(manifest["schema_name"], final.FINAL_SCHEMA_NAME)
            self.assertEqual(
                hashlib.sha256(
                    manifest["teacher_contract_canonical_json"].encode("utf-8")
                ).hexdigest(),
                manifest["teacher_contract_sha256"],
            )
            with np.load(
                pathlib.Path(temporary) / manifest["shards"][0]["path"],
                allow_pickle=False,
            ) as shard:
                self.assertEqual(shard["target_voltage"].shape, (8, 1))
                self.assertEqual(shard["target_nmda"].shape, (4, 8, 1))
                self.assertEqual(
                    shard["target_compartment_voltage"].shape, (8, 4, 1)
                )
                self.assertEqual(
                    shard["diagnostic_time_indices"].tolist(), [1, 3, 5, 7]
                )
                self.assertEqual(len(shard["contact_trial_offset"]), 2)
                self.assertEqual(len(shard["event_trial_offset"]), 2)
                self.assertTrue(
                    np.all(np.isfinite(shard["target_voltage"]))
                )
                metadata = json.loads(str(shard["metadata_json"]))
                self.assertEqual(
                    metadata["teacher_contract_sha256"],
                    manifest["teacher_contract_sha256"],
                )
        self.assertEqual(before, tracked_status())


if __name__ == "__main__":
    unittest.main()
