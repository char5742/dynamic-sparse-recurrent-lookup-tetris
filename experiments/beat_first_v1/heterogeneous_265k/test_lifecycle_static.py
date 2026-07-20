"""Fail-closed source-contract tests for the heterogeneous P0 lifecycle.

UNEXECUTED_STATIC_ONLY: this file is intentionally not run while another heavy
process owns the machine. It performs source inspection only and creates no
worker, Julia process, OpenVINO request, or accelerator work.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
PIPELINE = (ROOT / "pipeline.jl").read_text(encoding="utf-8")
LIFECYCLE = (ROOT / "lifecycle.jl").read_text(encoding="utf-8")
MASTER = (ROOT / "hashstem_master.jl").read_text(encoding="utf-8")


def function_body(source: str, name: str, next_name: str) -> str:
    start = source.index(f"function {name}")
    end = source.index(f"function {next_name}", start)
    return source[start:end]


class LifecycleStaticContract(unittest.TestCase):
    def test_uncoordinated_cpu_tail_is_permanently_disabled(self):
        body = function_body(
            PIPELINE, "try_acquire_free_slot!", "cancel_slot_reservation!"
        )
        self.assertIn("uncoordinated HashStem admission is disabled", body)
        self.assertNotIn("_try_acquire_slot!", body)
        cpu = function_body(PIPELINE, "try_acquire_cpu_slot!", "try_acquire_npu_slot!")
        self.assertLess(cpu.index("lock(coordinator.lock)"), cpu.index("_admit_hash_slot_locked!"))
        self.assertIn("published_snapshot_version", cpu)
        self.assertIn("accepting_cpu", cpu)

    def test_first_adoption_probation_cannot_self_authorize(self):
        begin = function_body(
            PIPELINE,
            "begin_first_adoption_probation!",
            "complete_first_adoption_probation!",
        )
        admit = function_body(
            PIPELINE,
            "try_acquire_probation_npu_slot!",
            "cancel_hash_slot_reservation!",
        )
        complete = function_body(
            PIPELINE,
            "complete_first_adoption_probation!",
            "abort_first_adoption_probation!",
        )
        self.assertNotIn("npu_adopted = true", begin)
        self.assertNotIn("npu_adopted = true", admit)
        self.assertIn("ever_adopted", begin)
        self.assertIn("ever_adopted", admit)
        self.assertLess(
            complete.index("_validate_npu_evidence_binding"),
            complete.index("npu_adopted = true"),
        )
        self.assertLess(
            complete.index("_require_old_lineage_drained_locked!"),
            complete.index("npu_adopted = true"),
        )

    def test_refresh_closes_cpu_and_npu_before_whole_ring_drain(self):
        begin = function_body(
            PIPELINE, "begin_snapshot_refresh!", "_ring_references_snapshot"
        )
        drain = function_body(
            PIPELINE,
            "_require_old_lineage_drained_locked!",
            "require_old_lineage_drained!",
        )
        self.assertIn("accepting_npu = false", begin)
        self.assertIn("accepting_cpu = false", begin)
        self.assertIn("isempty(coordinator.inflight)", drain)
        self.assertIn("isempty(coordinator.active_tickets)", drain)
        self.assertIn("_ring_has_live_payload", drain)

    def test_frozen_superstep_has_one_barrier_update(self):
        accumulate = function_body(
            LIFECYCLE, "accumulate_stem_gradient!", "_frozen_master_batch"
        )
        update = function_body(
            LIFECYCLE,
            "apply_superstep_barrier_update!",
            "publish_updated_superstep!",
        )
        run = function_body(
            LIFECYCLE, "run_superstep_barrier!", "_coordinator_boundary_record"
        )
        for token in (
            "source_snapshot_version",
            "sparse_bank_version",
            "sparse_index_version",
            "master_superstep",
            "exact admission order",
        ):
            self.assertIn(token, accumulate)
        self.assertIn("barrier_update_count == 0", update)
        self.assertEqual(update.count("_apply_hashstem_master_batch!"), 1)
        self.assertIn("barrier_update_count = 1", update)
        self.assertLess(run.index("apply_superstep_barrier_update!"), run.index("export_and_validate"))
        self.assertLess(run.index("export_and_validate"), run.index("publish_updated_superstep!"))
        self.assertNotIn("@async", LIFECYCLE)
        self.assertNotIn("Threads.@spawn", LIFECYCLE)

    def test_publication_validation_defaults_fail_closed(self):
        bundle = LIFECYCLE[
            LIFECYCLE.index("Base.@kwdef struct SuperstepPublicationBundle") :
            LIFECYCLE.index("function begin_frozen_superstep")
        ]
        self.assertIn(
            "runtime_validation_status::String = UNEXECUTED_STATIC_ONLY", bundle
        )
        self.assertNotIn(
            "runtime_validation_status::String = VALIDATED_RUNTIME", bundle
        )

    def test_boundary_checkpoint_binds_every_mutable_lineage(self):
        save = function_body(
            LIFECYCLE,
            "save_heterogeneous_boundary_checkpoint!",
            "load_heterogeneous_boundary_checkpoint",
        )
        load = LIFECYCLE[LIFECYCLE.index("function load_heterogeneous_boundary_checkpoint") :]
        for token in (
            "save_hashstem_master_checkpoint!",
            "sparse_bank",
            "sparse_indexes",
            "sparse_optimizer_state",
            "next_sequence",
            "slot_generations",
            "completed_superstep",
            "snapshot",
            "evidence",
            "master_manifest_sha256",
            "sparse_sha256",
            "lifecycle_sha256",
        ):
            self.assertIn(token, save)
        for token in (
            "resume_hashstem_master",
            "sparse-bank version differs",
            "sparse-index version differs",
            "checkpoint barrier count is not one",
            "checkpoint does not bind exactly one master update",
            "checkpoint export-manifest binding is invalid",
            "restore_authorized_snapshot!",
            "ring cursor differs",
        ):
            self.assertIn(token, load)

    def test_online_update_primitive_is_not_the_public_offline_entry(self):
        public = function_body(
            MASTER, "train_hashstem_master_step!", "_apply_hashstem_master_batch!"
        )
        internal = MASTER[MASTER.index("function _apply_hashstem_master_batch!") :]
        self.assertIn("last_snapshot_version == 0", public)
        self.assertIn("_apply_hashstem_master_batch!", public)
        self.assertIn("trainer.master_version += UInt64(1)", internal)


if __name__ == "__main__":
    unittest.main()
