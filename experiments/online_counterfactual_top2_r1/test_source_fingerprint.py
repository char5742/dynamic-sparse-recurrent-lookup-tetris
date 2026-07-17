from __future__ import annotations

import copy
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import contract
import validate_source_fingerprint as subject


LIVE_REPOSITORY = Path(__file__).resolve().parents[2]
ORIGINAL_NODE = LIVE_REPOSITORY / "upstream/TetrisAI/src/core/components/node.jl"
ORIGINAL_ANALYZER = LIVE_REPOSITORY / "upstream/TetrisAI/src/core/analyzer.jl"


def populate_runtime_closure(repository: Path) -> None:
    for relative in subject.RUNTIME_CLOSURE_PINS:
        destination = repository / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith("components/node.jl"):
            source = ORIGINAL_NODE
        elif relative.endswith("core/analyzer.jl"):
            source = ORIGINAL_ANALYZER
        else:
            source = LIVE_REPOSITORY / Path(relative)
        shutil.copyfile(source, destination)


class SourceFingerprintTests(unittest.TestCase):
    def make_repository(self, parent: Path) -> tuple[Path, Path]:
        repository = parent / "repository"
        repository.mkdir()
        external_cache = parent / "external-python-cache"
        (repository / "Manifest.toml").write_text("manifest\n", encoding="utf-8")
        experiment = repository / subject.EXPERIMENT_PREFIX
        experiment.mkdir(parents=True)
        (experiment / "contract.json").write_text('{"experiment":"R1"}\n', encoding="utf-8")
        (experiment / "invoke_once.ps1").write_text("Write-Output 'R1'\n", encoding="utf-8")
        populate_runtime_closure(repository)
        return repository, external_cache

    def test_full_runtime_closure_and_binary_mutation_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r1-fingerprint-") as temporary:
            repository, cache = self.make_repository(Path(temporary))
            with patch.dict(os.environ, {"PYTHONPYCACHEPREFIX": str(cache)}):
                document = subject.fingerprint_document(repository)
                paths = {record["path"] for record in document["files"]}
                for relative in subject.RUNTIME_CLOSURE_PINS:
                    self.assertIn(relative, paths)
                self.assertIn("vendor/Tetris/lib/game.so", paths)
                self.assertIn("vendor/Tetris/lib/pdcurses.dll", paths)
                self.assertTrue(subject.validate_fingerprint_document(repository, document)["valid"])

                stale = copy.deepcopy(document)
                analyzer = repository / subject.EXPERIMENT_PREFIX / "vendor/TetrisAI/src/core/analyzer.jl"
                analyzer.write_bytes(analyzer.read_bytes() + b"\n")
                result = subject.validate_fingerprint_document(repository, stale)
                self.assertFalse(result["valid"])
                self.assertTrue(any("analyzer.jl" in failure for failure in result["failures"]))

    def test_game_binary_mutation_breaks_runtime_aggregate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r1-binary-fingerprint-") as temporary:
            repository, cache = self.make_repository(Path(temporary))
            with patch.dict(os.environ, {"PYTHONPYCACHEPREFIX": str(cache)}):
                document = subject.fingerprint_document(repository)
                game = repository / "vendor/Tetris/lib/game.so"
                payload = bytearray(game.read_bytes())
                payload[0] ^= 0x01
                game.write_bytes(payload)
                result = subject.validate_fingerprint_document(repository, document)
                self.assertFalse(result["valid"])
                self.assertTrue(any("game.so" in failure for failure in result["failures"]))

    def test_repository_python_cache_and_internal_prefix_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r1-cache-policy-") as temporary:
            repository = Path(temporary) / "repository"
            repository.mkdir()
            outside = Path(temporary) / "cache"
            self.assertTrue(subject.validate_python_cache_policy(repository, str(outside))["valid"])
            self.assertFalse(subject.validate_python_cache_policy(repository, "")["valid"])
            internal = repository / "cache"
            self.assertFalse(subject.validate_python_cache_policy(repository, str(internal))["valid"])
            (repository / "standalone.pyc").write_bytes(b"not-code")
            cache = repository / "pkg/__pycache__"
            cache.mkdir(parents=True)
            (cache / "module.pyc").write_bytes(b"not-code")
            result = subject.validate_python_cache_policy(repository, str(outside))
            self.assertFalse(result["valid"])
            self.assertIn("standalone.pyc", result["repository_cache_artifacts"])
            self.assertTrue(any("__pycache__" in path for path in result["repository_cache_artifacts"]))

    def test_upstream_exact_clean_head_and_original_hashes(self) -> None:
        live = subject.validate_upstream_tetrisai_binding(LIVE_REPOSITORY)
        self.assertTrue(live["valid"], live["failures"])
        self.assertEqual(live["head"], subject.UPSTREAM_TETRISAI_HEAD)
        self.assertTrue(live["clean"])

        with tempfile.TemporaryDirectory(prefix="r1-upstream-") as temporary:
            repository = Path(temporary)
            upstream = repository / subject.UPSTREAM_TETRISAI_PATH
            node = upstream / "src/core/components/node.jl"
            analyzer = upstream / "src/core/analyzer.jl"
            node.parent.mkdir(parents=True)
            analyzer.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ORIGINAL_NODE, node)
            shutil.copyfile(ORIGINAL_ANALYZER, analyzer)

            def fake_git(_repository: Path, *arguments: str) -> str:
                if arguments == ("rev-parse", "HEAD"):
                    return subject.UPSTREAM_TETRISAI_HEAD
                if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
                    return ""
                raise AssertionError(arguments)

            with patch.object(subject, "git", side_effect=fake_git):
                self.assertTrue(subject.validate_upstream_tetrisai_binding(repository)["valid"])
                analyzer.write_bytes(analyzer.read_bytes() + b"tamper")
                result = subject.validate_upstream_tetrisai_binding(repository)
                self.assertFalse(result["valid"])
                self.assertTrue(any("analyzer.jl" in failure for failure in result["failures"]))

    def test_contract_and_validator_runtime_pins_are_identical(self) -> None:
        value = contract.load_contract()["runtime_source_binding"]
        observed = {
            value["vendored_tetrisai"]["node"]["vendor_relative_path"]: (
                value["vendored_tetrisai"]["node"]["bytes"],
                value["vendored_tetrisai"]["node"]["sha256"],
            ),
            value["vendored_tetrisai"]["analyzer"]["vendor_relative_path"]: (
                value["vendored_tetrisai"]["analyzer"]["bytes"],
                value["vendored_tetrisai"]["analyzer"]["sha256"],
            ),
        }
        observed.update(
            {record["path"]: (record["bytes"], record["sha256"]) for record in value["external_tetris_runtime"]}
        )
        self.assertEqual(observed, subject.RUNTIME_CLOSURE_PINS)
        self.assertEqual(value["upstream_tetrisai"]["head"], subject.UPSTREAM_TETRISAI_HEAD)
        self.assertEqual(
            value["runtime_closure_sha256"],
            subject.runtime_closure_records(LIVE_REPOSITORY)["sha256"],
        )

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
