from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import validate_source_fingerprint as subject


class SourceFingerprintTests(unittest.TestCase):
    def test_powershell_is_fingerprinted_and_tampering_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="q1-fingerprint-") as temporary:
            repository = Path(temporary)
            (repository / "Manifest.toml").write_text("manifest\n", encoding="utf-8")
            experiment = repository / subject.EXPERIMENT_PREFIX
            experiment.mkdir(parents=True)
            wrapper = experiment / "synthetic.ps1"
            wrapper.write_text("Write-Output 'Q1'\n", encoding="utf-8")
            document = subject.fingerprint_document(repository)
            paths = {record["path"] for record in document["files"]}
            self.assertIn(
                "experiments/frozen_old_additive_residual_q1/synthetic.ps1", paths
            )
            self.assertTrue(
                subject.validate_fingerprint_document(repository, document)["valid"]
            )
            stale = copy.deepcopy(document)
            wrapper.write_text("Write-Output 'tampered'\n", encoding="utf-8")
            result = subject.validate_fingerprint_document(repository, stale)
            self.assertFalse(result["valid"])
            self.assertTrue(any("synthetic.ps1" in item for item in result["failures"]))

    def test_changed_paths_are_q1_only(self) -> None:
        self.assertEqual(
            subject.validate_changed_paths(
                ["experiments/frozen_old_additive_residual_q1/invoke_once.ps1"]
            ),
            [],
        )
        failures = subject.validate_changed_paths(["scripts/source_fingerprint.jl"])
        self.assertTrue(any("escape" in item for item in failures))

    def test_binding_accepts_multi_commit_q1_chain_from_authorized_base(self) -> None:
        authorized = "a" * 40
        actual_parent = "b" * 40
        responses = {
            ("rev-parse", "HEAD"): authorized,
            ("rev-list", "--parents", "-n", "1", "HEAD"): (
                f"{authorized} {actual_parent}"
            ),
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
            (
                "diff",
                "--name-only",
                f"{subject.AUTHORIZED_BASE_COMMIT}..HEAD",
            ): "\n".join((
                "experiments/frozen_old_additive_residual_q1/native_arguments.ps1",
                "experiments/frozen_old_additive_residual_q1/train_q1.jl",
            )),
        }
        with patch.object(subject, "git", side_effect=lambda _repo, *args: responses[args]), patch.object(
            subject, "git_is_ancestor", return_value=True
        ):
            result = subject.validate_repository_binding(Path.cwd(), authorized)
        self.assertTrue(result["valid"], result["failures"])
        self.assertEqual(result["parent"], actual_parent)
        self.assertEqual(result["authorized_base_commit"], subject.AUTHORIZED_BASE_COMMIT)
        self.assertTrue(result["base_is_ancestor"])

    def test_binding_rejects_wrong_base_and_escaped_base_to_head_path(self) -> None:
        authorized = "a" * 40
        responses = {
            ("rev-parse", "HEAD"): authorized,
            ("rev-list", "--parents", "-n", "1", "HEAD"): f"{authorized} {'b' * 40}",
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
        }
        with patch.object(subject, "git", side_effect=lambda _repo, *args: responses[args]), patch.object(
            subject, "git_is_ancestor", return_value=False
        ):
            wrong_base = subject.validate_repository_binding(Path.cwd(), authorized)
        self.assertFalse(wrong_base["valid"])
        self.assertTrue(any("not an ancestor" in item for item in wrong_base["failures"]))

        responses[("diff", "--name-only", f"{subject.AUTHORIZED_BASE_COMMIT}..HEAD")] = (
            "experiments/frozen_old_additive_residual_q1/train_q1.jl\n"
            "scripts/source_fingerprint.jl"
        )
        with patch.object(subject, "git", side_effect=lambda _repo, *args: responses[args]), patch.object(
            subject, "git_is_ancestor", return_value=True
        ):
            escaped = subject.validate_repository_binding(Path.cwd(), authorized)
        self.assertFalse(escaped["valid"])
        self.assertTrue(any("escape" in item for item in escaped["failures"]))

        responses[("diff", "--name-only", f"{subject.AUTHORIZED_BASE_COMMIT}..HEAD")] = (
            "experiments/frozen_old_additive_residual_q1/train_q1.jl"
        )
        with patch.object(subject, "git", side_effect=lambda _repo, *args: responses[args]), patch.object(
            subject, "git_is_ancestor", return_value=True
        ):
            wrong_head = subject.validate_repository_binding(Path.cwd(), "c" * 40)
        self.assertFalse(wrong_head["valid"])
        self.assertTrue(any("not live HEAD" in item for item in wrong_head["failures"]))


if __name__ == "__main__":
    unittest.main()
