from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_PARENT_COMMIT = "8d784985f300598d2a05ed4402902ae86dfb4908"
EXPERIMENT_PREFIX = "experiments/frozen_old_additive_residual_q1/"
SOURCE_ROOTS = (
    "Project.toml",
    "Manifest.toml",
    "upstream-lock.toml",
    "configs",
    "src",
    "scripts",
    "test",
    "tools",
    "vendor/Tetris/Project.toml",
    "vendor/Tetris/src",
    "experiments",
)
# PowerShell is executable source for Q1 and must be covered by the aggregate,
# unlike the older repository-wide fingerprint helper.
SOURCE_SUFFIXES = (".jl", ".toml", ".md", ".py", ".ps1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_linked_directory(path: Path) -> bool:
    if path.is_symlink():
        return True
    attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def source_records(repository: Path) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for relative in SOURCE_ROOTS:
        path = repository / relative
        if path.is_file():
            paths.add(path)
        elif path.is_dir():
            for directory, directory_names, file_names in os.walk(
                path, topdown=True, followlinks=False
            ):
                current = Path(directory)
                directory_names[:] = [
                    name
                    for name in directory_names
                    if not is_linked_directory(current / name)
                ]
                paths.update(
                    current / name
                    for name in file_names
                    if (current / name).suffix.lower() in SOURCE_SUFFIXES
                )
    ordered = sorted(paths, key=lambda path: path.relative_to(repository).as_posix())
    return [
        {
            "path": path.relative_to(repository).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in ordered
    ]


def source_aggregate(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(f"{record['path']}\0{record['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def fingerprint_document(repository: Path) -> dict[str, Any]:
    repository = repository.resolve()
    records = source_records(repository)
    return {
        "repository_root": str(repository),
        "source_sha256": source_aggregate(records),
        "manifest_sha256": sha256_file(repository / "Manifest.toml"),
        "file_count": len(records),
        "files": records,
    }


def write_fingerprint(repository: Path, output: Path) -> dict[str, Any]:
    document = fingerprint_document(repository)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary artifact: {temporary}")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return document


def normalized_root(path: str | Path) -> str:
    return str(Path(path).resolve()).rstrip("\\/").casefold()


def validate_fingerprint_document(repository: Path, document: dict[str, Any]) -> dict[str, Any]:
    repository = repository.resolve()
    failures: list[str] = []
    root_value = document.get("repository_root")
    if not isinstance(root_value, str) or not root_value.strip():
        failures.append("fingerprint repository_root is missing")
    elif normalized_root(root_value) != normalized_root(repository):
        failures.append("repository_root does not identify the live repository")
    live = source_records(repository)
    supplied = document.get("files")
    if "files" not in document:
        failures.append("fingerprint files field is missing")
    if not isinstance(supplied, list):
        supplied = []
        failures.append("fingerprint files is not a list")
    supplied_paths = [record.get("path") for record in supplied if isinstance(record, dict)]
    if len(supplied_paths) != len(set(supplied_paths)):
        failures.append("fingerprint contains duplicate paths")
    if supplied != live:
        live_by_path = {record["path"]: record for record in live}
        supplied_by_path = {
            record.get("path"): record for record in supplied if isinstance(record, dict)
        }
        if set(supplied_by_path) != set(live_by_path):
            failures.append("fingerprint file set/path list differs from live source set")
        for path in sorted(set(supplied_by_path) & set(live_by_path)):
            observed = supplied_by_path[path]
            expected = live_by_path[path]
            if observed.get("bytes") != expected["bytes"]:
                failures.append(f"fingerprint byte count mismatch: {path}")
            if observed.get("sha256") != expected["sha256"]:
                failures.append(f"fingerprint SHA-256 mismatch: {path}")
    aggregate = source_aggregate(live)
    if not isinstance(document.get("source_sha256"), str):
        failures.append("fingerprint source_sha256 is missing")
    elif document.get("source_sha256") != aggregate:
        failures.append("fingerprint aggregate source_sha256 mismatch")
    manifest_hash = sha256_file(repository / "Manifest.toml")
    if not isinstance(document.get("manifest_sha256"), str):
        failures.append("fingerprint manifest_sha256 is missing")
    elif document.get("manifest_sha256") != manifest_hash:
        failures.append("fingerprint manifest_sha256 mismatch")
    if not isinstance(document.get("file_count"), int):
        failures.append("fingerprint file_count is missing")
    elif document.get("file_count") != len(live):
        failures.append("fingerprint file_count mismatch")
    return {
        "valid": not failures,
        "failures": failures,
        "repository_root": str(repository),
        "manifest_sha256": manifest_hash,
        "source_sha256": aggregate,
        "file_count": len(live),
        "files": live,
    }


def git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_changed_paths(changed: list[str]) -> list[str]:
    failures: list[str] = []
    if not changed:
        failures.append("authorized Q1 commit changes no paths")
    escaped = [path for path in changed if not path.startswith(EXPERIMENT_PREFIX)]
    if escaped:
        failures.append("authorized Q1 commit changes escape the Q1 experiment directory")
    return failures


def validate_repository_binding(repository: Path, authorized_commit: str) -> dict[str, Any]:
    failures: list[str] = []
    head = git(repository, "rev-parse", "HEAD")
    parents = git(repository, "rev-list", "--parents", "-n", "1", "HEAD").split()
    parent = parents[1] if len(parents) == 2 else None
    if head != authorized_commit:
        failures.append("explicitly authorized Q1 commit is not live HEAD")
    if parent != EXPECTED_PARENT_COMMIT:
        failures.append("live HEAD is not the single-child Q1 commit of the audited base")
    status = git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        failures.append("repository is not clean, including untracked files")
    changed = (
        [
            line
            for line in git(
                repository, "diff", "--name-only", f"{EXPECTED_PARENT_COMMIT}..HEAD"
            ).splitlines()
            if line
        ]
        if parent == EXPECTED_PARENT_COMMIT
        else []
    )
    failures.extend(validate_changed_paths(changed))
    return {
        "valid": not failures,
        "failures": failures,
        "head": head,
        "parent": parent,
        "expected_parent": EXPECTED_PARENT_COMMIT,
        "authorized_commit": authorized_commit,
        "changed_paths": changed,
        "repository_clean": not status,
    }


def audit(
    repository: Path, fingerprint_path: Path, authorized_commit: str
) -> dict[str, Any]:
    document = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    fingerprint = validate_fingerprint_document(repository, document)
    binding = validate_repository_binding(repository, authorized_commit)
    failures = [*fingerprint["failures"], *binding["failures"]]
    return {
        "valid": not failures,
        "failures": failures,
        "fingerprint_path": str(fingerprint_path.resolve()),
        "fingerprint_file_sha256": sha256_file(fingerprint_path),
        "fingerprint": fingerprint,
        "repository_binding": binding,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate",
        action="store_true",
        help="write a fresh Q1 fingerprint instead of validating one",
    )
    parser.add_argument("repository", type=Path)
    parser.add_argument("fingerprint", type=Path)
    parser.add_argument("authorized_commit", nargs="?")
    args = parser.parse_args()
    if args.generate:
        if args.authorized_commit is not None:
            parser.error("authorized_commit is not accepted with --generate")
        result = write_fingerprint(args.repository.resolve(), args.fingerprint)
        print(json.dumps(result, separators=(",", ":")))
        return
    if args.authorized_commit is None:
        parser.error("authorized_commit is required for validation")
    if not __import__("re").fullmatch(r"[0-9a-fA-F]{40}", args.authorized_commit):
        parser.error("authorized_commit must be full 40-hex")
    result = audit(
        args.repository.resolve(),
        args.fingerprint.resolve(),
        args.authorized_commit.lower(),
    )
    print(json.dumps(result, separators=(",", ":")))
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
