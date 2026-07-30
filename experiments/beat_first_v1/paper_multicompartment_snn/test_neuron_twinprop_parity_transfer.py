#!/usr/bin/env python3
"""Mechanism smoke tests for official NEURON parity transfer-back."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np

import neuron_hay_teacher as hay
import neuron_twinprop_parity_transfer as transfer


MODELDB_ROOT = pathlib.Path("/mnt/c/tmp/hay_modeldb_139653")


class NeuronParityTransferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not MODELDB_ROOT.is_dir():
            raise unittest.SkipTest("Hay ModelDB checkout is unavailable")
        cls.teacher = hay.instantiate_teacher(
            MODELDB_ROOT, diagnostic_segments=8
        )
        cls.eligible = cls.teacher.basal_apical_indices[:2]

    def _fixture(self, root: pathlib.Path, variant: str) -> pathlib.Path:
        path = root / f"{variant}.npz"
        events = np.zeros((2, 8, 2), dtype=np.uint8)
        events[0, 1, 0] = 1
        events[1, 2, 0] = 1
        events[0, 1, 1] = 1
        np.savez_compressed(
            path,
            schema=np.asarray(transfer.INPUT_SCHEMA),
            model_name=np.asarray(hay.MODEL_NAME),
            task=np.asarray("xor"),
            dimension=np.asarray(2, dtype=np.int32),
            variant=np.asarray(variant),
            sample_dt_ms=np.asarray(1.0, dtype=np.float32),
            decision_first_step=np.asarray(5, dtype=np.int32),
            contacts_per_axon=np.asarray(1, dtype=np.int32),
            axon_kind=np.asarray(
                [hay.EXCITATORY, hay.INHIBITORY], dtype=np.uint8
            ),
            contact_axon=np.asarray([1, 2], dtype=np.int32),
            contact_kind=np.asarray(
                [hay.EXCITATORY, hay.INHIBITORY], dtype=np.uint8
            ),
            contact_segment=np.asarray(self.eligible, dtype=np.int32),
            contact_location_slot=np.asarray([1, 2], dtype=np.int64),
            contact_strength=np.asarray([0.2, 0.2], dtype=np.float32),
            axon_events=events,
            target=np.asarray([0, 0], dtype=np.uint8),
            source_twin_sha256=np.asarray("1" * 64),
            source_parameter_sha256=np.asarray("2" * 64),
            optimizer_result_sha256=np.asarray("3" * 64),
            modeldb_morphology_sha256=np.asarray(
                self.teacher.hashes["morphology_sha256"]
            ),
        )
        return path

    def test_full_and_no_nmda_use_soma_spike_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for variant in ("full", "no_nmda"):
                report, traces = transfer.run_transfer(
                    modeldb_root=MODELDB_ROOT,
                    input_path=self._fixture(root, variant),
                    variant=variant,
                    trace_trials=1,
                )
                self.assertEqual(report["schema"], transfer.SCHEMA)
                self.assertEqual(
                    report["readout"],
                    "at_least_one_soma_spike_in_decision_window",
                )
                self.assertFalse(report["analog_readout_bypass"])
                self.assertEqual(report["transfer_authority"], "Hay ModelDB 139653 + NEURON")
                self.assertEqual(traces["soma_voltage"].shape, (8, 1))
                self.assertTrue(np.all(np.isfinite(traces["soma_voltage"])))
                self.assertTrue(np.all(np.isfinite(traces["nmda_region"])))
                if variant == "no_nmda":
                    self.assertEqual(report["mean_abs_nmda_current"], 0.0)
                    self.assertTrue(np.all(traces["nmda_region"] == 0.0))

    def test_passive_and_soma_only_are_independent_transfer_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for variant in ("passive", "soma_only"):
                report, traces = transfer.run_transfer(
                    modeldb_root=MODELDB_ROOT,
                    input_path=self._fixture(root, variant),
                    variant=variant,
                    trace_trials=1,
                )
                self.assertEqual(report["variant"], variant)
                self.assertTrue(report["independently_retrained_variant"])
                self.assertTrue(report["constraints"]["dale_fixed"])
                self.assertTrue(
                    report["constraints"]["density_constraint_verified"]
                )
                self.assertTrue(np.all(np.isfinite(traces["soma_voltage"])))

    def test_wrong_variant_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._fixture(pathlib.Path(directory), "full")
            with self.assertRaisesRegex(ValueError, "variant differs"):
                transfer.run_transfer(
                    modeldb_root=MODELDB_ROOT,
                    input_path=path,
                    variant="passive",
                    trace_trials=0,
                )


if __name__ == "__main__":
    unittest.main()
