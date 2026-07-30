#!/usr/bin/env python3
"""Smoke tests for the official Hay ModelDB teacher path."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tempfile
import unittest

import numpy as np

import neuron_hay_teacher as teacher_module


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


@unittest.skipUnless(
    (MODELDB_ROOT / "x86_64" / "libnrnmech.so").is_file(),
    "compiled Hay ModelDB mechanisms are unavailable",
)
class HayNeuronTeacherTest(unittest.TestCase):
    def test_instantiation_uses_official_morphology(self) -> None:
        before = tracked_status()
        teacher = teacher_module.instantiate_teacher(
            MODELDB_ROOT, diagnostic_segments=8
        )
        self.assertEqual(len(teacher.segments), 642)
        self.assertEqual(len(teacher.basal_apical_indices), 639)
        self.assertEqual(len(teacher.diagnostic_indices), 8)
        self.assertEqual(
            teacher.hashes["modeldb_git_commit"],
            "50a4aab3ce5c295ad16a134c5d9261b7cc3fbe58",
        )
        self.assertEqual(before, tracked_status())

    def test_one_trajectory_and_npz_contract(self) -> None:
        config = teacher_module.TeacherConfig(
            preset="test",
            train_trials=1,
            validation_trials=0,
            test_trials=0,
            duration_ms=4,
            axons=4,
            contacts_per_axon=2,
            diagnostic_segments=8,
            shard_size=1,
            rate_hz=250.0,
            burst_probability=0.25,
            excitatory_fraction=0.75,
            minimum_strength=0.5,
            maximum_strength=1.0,
            seed=12345,
            dt_ms=0.025,
            sample_dt_ms=1.0,
            v_init_mv=-80.0,
            celsius=34.0,
            spike_threshold_mv=-20.0,
            ca_event_cai_mm=2.0e-4,
            ca_event_voltage_mv=-30.0,
            store_dense_events=True,
        )
        before = tracked_status()
        with tempfile.TemporaryDirectory() as temporary:
            manifest = teacher_module.generate_dataset(
                MODELDB_ROOT, temporary, config
            )
            manifest_path = pathlib.Path(temporary) / "manifest.json"
            loaded_manifest = json.loads(manifest_path.read_text("utf-8"))
            self.assertEqual(
                loaded_manifest["schema_name"], teacher_module.SCHEMA_NAME
            )
            self.assertEqual(
                loaded_manifest["completion_state"], "complete"
            )
            self.assertEqual(manifest["completed_trials"], 1)
            shard_path = (
                pathlib.Path(temporary)
                / loaded_manifest["shards"][0]["path"]
            )
            with np.load(shard_path, allow_pickle=False) as shard:
                self.assertEqual(shard["target_voltage"].shape, (4, 1))
                self.assertEqual(shard["target_spike"].shape, (4, 1))
                self.assertEqual(shard["target_nmda"].shape, (4, 4, 1))
                self.assertEqual(
                    shard["target_compartment_voltage"].shape, (8, 4, 1)
                )
                self.assertEqual(
                    shard["target_compartment_nmda"].shape, (8, 4, 1)
                )
                self.assertEqual(
                    shard["target_dendritic_cai"].shape, (8, 4, 1)
                )
                self.assertEqual(shard["contact_axon"].shape, (8, 1))
                self.assertEqual(shard["event_spike"].shape, (8, 4, 1))
                self.assertTrue(np.all(np.isfinite(shard["target_voltage"])))
                self.assertTrue(
                    np.all(np.isfinite(shard["target_nmda"]))
                )
                self.assertTrue(np.all(shard["contact_strength"] >= 0.0))
                self.assertTrue(np.all(shard["contact_strength"] <= 1.0))
                self.assertTrue(
                    np.all(np.isin(shard["contact_kind"], (1, 2)))
                )
                metadata = json.loads(str(shard["metadata_json"]))
                self.assertEqual(
                    metadata["teacher_contract_sha256"],
                    loaded_manifest["teacher_contract_sha256"],
                )
        self.assertEqual(before, tracked_status())


if __name__ == "__main__":
    unittest.main()
