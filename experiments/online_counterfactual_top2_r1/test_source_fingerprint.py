from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import validate_source_fingerprint as subject


class SourceFingerprintTests(unittest.TestCase):
    def test_json_and_powershell_are_covered_and_tampering_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r1-fingerprint-") as temporary:
            repository = Path(temporary)
            (repository / "Manifest.toml").write_text("manifest\n", encoding="utf-8")
            experiment = repository / subject.EXPERIMENT_PREFIX
            experiment.mkdir(parents=True)
            contract = experiment / "contract.json"
            wrapper = experiment / "invoke_once.ps1"
            contract.write_text('{"experiment":"R1"}\n', encoding="utf-8")
            wrapper.write_text("Write-Output 'R1'\n", encoding="utf-8")
            document = subject.fingerprint_document(repository)
            paths = {record["path"] for record in document["files"]}
            self.assertIn(subject.EXPERIMENT_PREFIX + "contract.json", paths)
            self.assertIn(subject.EXPERIMENT_PREFIX + "invoke_once.ps1", paths)
            self.assertTrue(subject.validate_fingerprint_document(repository, document)["valid"])
            stale = copy.deepcopy(document)
            wrapper.write_text("Write-Output 'tampered'\n", encoding="utf-8")
            self.assertFalse(subject.validate_fingerprint_document(repository, stale)["valid"])

    def test_changed_paths_are_fail_closed(self) -> None:
        self.assertEqual(
            subject.validate_changed_paths([subject.EXPERIMENT_PREFIX + "contract.json"]), []
        )
        self.assertTrue(subject.validate_changed_paths([]))
        self.assertTrue(
            subject.validate_changed_paths(
                [subject.EXPERIMENT_PREFIX + "contract.json", "reports/tamper.md"]
            )
        )


if __name__ == "__main__":
    unittest.main()
