"""Standard-library-only static checks for the active-block witness producer.

The test reads text only.  It does not launch Julia/Python subprocesses, import
NumPy/OpenVINO, read teacher_v3 parts, or deserialize a sparse checkpoint.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "active_block_witness_producer_v2.jl"
DOC = ROOT / "ACTIVE_BLOCK_WITNESS_PRODUCER_V2.md"


class ActiveBlockWitnessProducerV2Static(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.doc = DOC.read_text(encoding="utf-8")

    def test_requires_external_bindings_and_fresh_outputs(self) -> None:
        for flag in (
            "--dataset-root",
            "--dataset-manifest-sha256",
            "--checkpoint-sha256",
            "--contract-sha256",
            "--producer-sha256",
            "--output-npz",
            "--output-metadata",
        ):
            self.assertIn(f'"{flag}"', self.source)
        self.assertIn("refusing to overwrite", self.source)
        self.assertIn("_canonical_existing_file", self.source)
        self.assertIn("_reject_reparse_chain", self.source)
        self.assertIn("canonical alias", self.source)
        self.assertIn("physical_id = (status.device, status.inode)", self.source)

    def test_train_only_reserved_seed_and_role_barriers(self) -> None:
        self.assertIn('split == "train" || continue', self.source)
        self.assertIn('Set(("old_policy", "epsilon", "dagger"))', self.source)
        self.assertIn("Set(8001:8008)", self.source)
        self.assertIn("Set(91001:91032)", self.source)
        self.assertIn("development_validation_sealed_seeds_used=false", self.source)
        self.assertIn("held_out_development_validation_sealed_seeds_used", self.source)
        self.assertIn("Validation part bytes are never opened here", self.source)

    def test_only_trained_exact_k256_checkpoint_can_produce_rows(self) -> None:
        self.assertIn("SparseDynamic3Layer.load_checkpoint", self.source)
        self.assertIn("checkpoint must contain at least one completed sparse update", self.source)
        self.assertIn('String(config.variant) == "k256"', self.source)
        self.assertIn("Tuple(Int.(topology.active_counts)) == MAX_COUNTS", self.source)
        self.assertIn("dataset_manifest_sha256", self.source)
        forbidden_calls = ("initialize_model(", "initialize_exact_model(", "rand(", "randn(")
        for call in forbidden_calls:
            self.assertNotIn(call, self.source)

    def test_exact_consumer_abi_and_prefixes(self) -> None:
        self.assertIn("using .BeatFirstTrainingCore: allocate_host_batch, pack_batch!", self.source)
        self.assertIn("const LAYER_DIMS = (560, 321, 321)", self.source)
        self.assertIn("const MAX_COUNTS = (96, 80, 80)", self.source)
        for name in ("selected_rows_l$layer", "selected_ids_l$layer", "scaled_input_l$layer"):
            self.assertIn(name, self.source)
        self.assertIn("k64=(24, 20, 20)", self.source)
        self.assertIn("k128=(48, 40, 40)", self.source)
        self.assertIn("k256=(96, 80, 80)", self.source)
        self.assertIn("smaller_variants_use_ordered_prefixes=true", self.source)

    def test_npz_is_deterministic_c_order_not_npz_julia_fortran_order(self) -> None:
        self.assertIn("'fortran_order': False", self.source)
        self.assertIn("ZIP_LOCAL_FILE_HEADER_SIGNATURE", self.source)
        self.assertIn("ZIP_CENTRAL_DIRECTORY_SIGNATURE", self.source)
        self.assertIn("ZIP_END_DIRECTORY_SIGNATURE", self.source)
        for flag in ("JL_O_WRONLY", "JL_O_CREAT", "JL_O_EXCL"):
            self.assertEqual(self.source.count(f"Base.Filesystem.{flag}"), 2)
        self.assertIn("axes are deliberately reversed", self.source)
        self.assertNotIn("using NPZ", self.source)
        self.assertNotIn("NPZ.npzwrite", self.source)

    def test_no_synthetic_fallback_or_claim_of_strength(self) -> None:
        self.assertIn("no synthetic", self.source.lower())
        self.assertIn("UNEXECUTED STATIC IMPLEMENTATION", self.doc)
        self.assertIn("component witness", self.doc.lower())
        for claim in ("old model beaten", "stronger checkpoint", "game score improved"):
            self.assertNotIn(claim, self.doc.lower())

    def test_cli_documentation_contains_prerequisite_and_all_bindings(self) -> None:
        self.assertIn("trained k256", self.doc.lower())
        self.assertIn("no initialized-weight fallback", self.doc.lower())
        for flag in sorted(re.findall(r'"(--[a-z0-9-]+)"', self.source)):
            if flag in {
                "--dataset-root", "--dataset-manifest", "--dataset-manifest-sha256",
                "--checkpoint", "--checkpoint-sha256", "--contract",
                "--contract-sha256", "--producer-sha256", "--candidate-count",
                "--sampling-domain", "--output-npz", "--output-metadata",
            }:
                self.assertIn(flag, self.doc)


if __name__ == "__main__":
    unittest.main()
