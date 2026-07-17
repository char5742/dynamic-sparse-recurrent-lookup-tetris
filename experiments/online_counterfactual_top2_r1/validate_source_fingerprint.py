from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any


AUTHORIZED_BASE_COMMIT = "9b2f974d3f5950e4084dc27d546fffc25a736c90"
AUTHORIZED_IMPLEMENTATION_PARENT_COMMIT = "ddd0f6f83c0e2931cd4cd415531b30b3c80cd2bd"
EXPERIMENT_PREFIX = "experiments/online_counterfactual_top2_r1/"
UPSTREAM_TETRISAI_PATH = "upstream/TetrisAI"
UPSTREAM_TETRISAI_HEAD = "6fdfb1d30197246fd862b716438e998f0315c830"
RUNTIME_CLOSURE_PINS = {
    EXPERIMENT_PREFIX + "vendor/TetrisAI/src/core/components/node.jl": (
        298,
        "e98d2052f9248f5c08c1eb58adaace1bd01533f287e682bf35a2fefa1325fe82",
    ),
    EXPERIMENT_PREFIX + "vendor/TetrisAI/src/core/analyzer.jl": (
        8114,
        "24152e2549dcc6c3c25d928454268e8baaa4d45fea31044603917cfbabbe02bc",
    ),
    "vendor/Tetris/lib/curses.jl": (
        2767,
        "4dd113316c4f82a226563d7ac3237c366417211582722b3d4b4277dcb12ff922",
    ),
    "vendor/Tetris/lib/key_input.jl": (
        1448,
        "c09571b424a49f01278f6903c0018f9a2dfc652dfd18b804c4ad2b6a37f2fc53",
    ),
    "vendor/Tetris/lib/game.so": (
        49870,
        "d63a03f494cb0a6f1704624923c58cb521a8a45873ca400e7085c02b1bf5bf46",
    ),
    "vendor/Tetris/lib/pdcurses.dll": (
        176673,
        "0c770aa6721aa2155bbe2ef1d0f50ad2065da399085242454486c855c1f9fe67",
    ),
}
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
    "vendor/Tetris/lib",
    "experiments",
)
SOURCE_SUFFIXES = (".jl", ".toml", ".md", ".json", ".py", ".ps1", ".so", ".dll")
NON_SOURCE_DIRECTORY_NAMES = frozenset({"node_modules"})


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


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def repository_python_cache_artifacts(repository: Path) -> list[str]:
    repository = repository.resolve()
    artifacts: list[str] = []
    for directory, directory_names, file_names in os.walk(
        repository, topdown=True, followlinks=False
    ):
        current = Path(directory)
        kept: list[str] = []
        for name in directory_names:
            child = current / name
            if name == ".git" or is_linked_directory(child):
                continue
            if name.casefold() == "__pycache__":
                artifacts.append(child.relative_to(repository).as_posix() + "/")
                continue
            kept.append(name)
        directory_names[:] = kept
        artifacts.extend(
            (current / name).relative_to(repository).as_posix()
            for name in file_names
            if Path(name).suffix.casefold() in (".pyc", ".pyo")
        )
    return sorted(set(artifacts))


def validate_python_cache_policy(
    repository: Path, prefix: str | None = None
) -> dict[str, Any]:
    repository = repository.resolve()
    failures: list[str] = []
    raw_prefix = prefix if prefix is not None else os.environ.get("PYTHONPYCACHEPREFIX")
    resolved_prefix: Path | None = None
    if not raw_prefix:
        failures.append("PYTHONPYCACHEPREFIX is missing")
    else:
        candidate = Path(raw_prefix)
        if not candidate.is_absolute():
            failures.append("PYTHONPYCACHEPREFIX is not absolute")
        else:
            resolved_prefix = candidate.resolve()
            if is_within(resolved_prefix, repository) or resolved_prefix == repository:
                failures.append("PYTHONPYCACHEPREFIX is inside the repository")
    artifacts = repository_python_cache_artifacts(repository)
    if artifacts:
        failures.append("repository contains forbidden __pycache__/pyc artifacts")
    return {
        "valid": not failures,
        "failures": failures,
        "prefix": str(resolved_prefix) if resolved_prefix is not None else None,
        "repository_cache_artifacts": artifacts,
    }


def runtime_closure_records(repository: Path) -> dict[str, Any]:
    repository = repository.resolve()
    failures: list[str] = []
    records: list[dict[str, Any]] = []
    for relative, (expected_bytes, expected_sha256) in sorted(RUNTIME_CLOSURE_PINS.items()):
        path = repository / Path(relative)
        if not path.is_file():
            failures.append(f"missing runtime closure file: {relative}")
            continue
        if is_linked_directory(path):
            failures.append(f"runtime closure file is symlink/reparse: {relative}")
            continue
        observed_bytes = path.stat().st_size
        observed_sha256 = sha256_file(path)
        if observed_bytes != expected_bytes:
            failures.append(f"runtime closure byte count mismatch: {relative}")
        if observed_sha256 != expected_sha256:
            failures.append(f"runtime closure SHA-256 mismatch: {relative}")
        records.append(
            {"path": relative, "bytes": observed_bytes, "sha256": observed_sha256}
        )
    return {
        "valid": not failures,
        "failures": failures,
        "files": records,
        "sha256": source_aggregate(records),
    }


def source_records(repository: Path) -> list[dict[str, Any]]:
    repository = repository.resolve()
    paths: set[Path] = set()
    for relative in SOURCE_ROOTS:
        path = repository / relative
        if path.is_file():
            if is_linked_directory(path):
                raise ValueError(f"source file is symlink/reparse: {relative}")
            paths.add(path)
        elif path.is_dir():
            if is_linked_directory(path):
                raise ValueError(f"source root is symlink/reparse: {relative}")
            for directory, directory_names, file_names in os.walk(path, topdown=True, followlinks=False):
                current = Path(directory)
                kept: list[str] = []
                for name in directory_names:
                    child = current / name
                    if name == ".git":
                        continue
                    if name.casefold() in NON_SOURCE_DIRECTORY_NAMES:
                        continue
                    if is_linked_directory(child):
                        child_relative = child.relative_to(repository).as_posix()
                        raise ValueError(f"source directory is symlink/reparse: {child_relative}")
                    if name.casefold() == "__pycache__":
                        continue
                    kept.append(name)
                directory_names[:] = kept
                for name in file_names:
                    candidate = current / name
                    if candidate.suffix.lower() not in SOURCE_SUFFIXES:
                        continue
                    if is_linked_directory(candidate):
                        candidate_relative = candidate.relative_to(repository).as_posix()
                        raise ValueError(f"source file is symlink/reparse: {candidate_relative}")
                    paths.add(candidate)
    ordered = sorted(paths, key=lambda item: item.relative_to(repository).as_posix())
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
    cache_policy = validate_python_cache_policy(repository)
    if not cache_policy["valid"]:
        raise ValueError("; ".join(cache_policy["failures"]))
    records = source_records(repository)
    runtime = runtime_closure_records(repository)
    if not runtime["valid"]:
        raise ValueError("; ".join(runtime["failures"]))
    manifest = repository / "Manifest.toml"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing Manifest.toml: {manifest}")
    return {
        "repository_root": str(repository),
        "source_sha256": source_aggregate(records),
        "manifest_sha256": sha256_file(manifest),
        "file_count": len(records),
        "files": records,
        "runtime_closure_sha256": runtime["sha256"],
        "runtime_closure_files": runtime["files"],
        "upstream_tetrisai_expected_head": UPSTREAM_TETRISAI_HEAD,
        "python_cache_prefix": cache_policy["prefix"],
    }


def atomic_write_fingerprint(repository: Path, output: Path) -> dict[str, Any]:
    upstream = validate_upstream_tetrisai_binding(repository)
    if not upstream["valid"]:
        raise ValueError("; ".join(upstream["failures"]))
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
    cache_policy = validate_python_cache_policy(repository)
    failures.extend(cache_policy["failures"])
    supplied_root = document.get("repository_root")
    if not isinstance(supplied_root, str) or normalized_root(supplied_root) != normalized_root(repository):
        failures.append("fingerprint repository_root does not identify the live repository")
    try:
        live = source_records(repository)
    except (OSError, ValueError) as error:
        live = []
        failures.append(f"source enumeration failed: {error}")
    supplied = document.get("files")
    if not isinstance(supplied, list):
        supplied = []
        failures.append("fingerprint files is missing or not a list")
    supplied_paths = [record.get("path") for record in supplied if isinstance(record, dict)]
    if len(supplied_paths) != len(set(supplied_paths)):
        failures.append("fingerprint contains duplicate paths")
    if supplied != live:
        failures.append("fingerprint file records differ from live source")
    aggregate = source_aggregate(live)
    if document.get("source_sha256") != aggregate:
        failures.append("fingerprint aggregate source_sha256 mismatch")
    manifest_hash = sha256_file(repository / "Manifest.toml")
    if document.get("manifest_sha256") != manifest_hash:
        failures.append("fingerprint manifest_sha256 mismatch")
    if document.get("file_count") != len(live):
        failures.append("fingerprint file_count mismatch")
    runtime = runtime_closure_records(repository)
    failures.extend(runtime["failures"])
    if document.get("runtime_closure_files") != runtime["files"]:
        failures.append("fingerprint runtime closure records differ from live pinned closure")
    if document.get("runtime_closure_sha256") != runtime["sha256"]:
        failures.append("fingerprint runtime closure aggregate mismatch")
    if document.get("upstream_tetrisai_expected_head") != UPSTREAM_TETRISAI_HEAD:
        failures.append("fingerprint upstream TetrisAI expected HEAD mismatch")
    supplied_cache_prefix = document.get("python_cache_prefix")
    if (
        not isinstance(supplied_cache_prefix, str)
        or cache_policy["prefix"] is None
        or normalized_root(supplied_cache_prefix) != normalized_root(cache_policy["prefix"])
    ):
        failures.append("fingerprint Python cache prefix differs from live external prefix")
    return {
        "valid": not failures,
        "failures": failures,
        "repository_root": str(repository),
        "source_sha256": aggregate,
        "manifest_sha256": manifest_hash,
        "file_count": len(live),
        "files": live,
        "runtime_closure": runtime,
        "python_cache_policy": cache_policy,
    }


def git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_is_ancestor(repository: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip())
    return result.returncode == 0


def validate_upstream_tetrisai_binding(repository: Path) -> dict[str, Any]:
    repository = repository.resolve()
    upstream = repository / Path(UPSTREAM_TETRISAI_PATH)
    failures: list[str] = []
    head: str | None = None
    status: str | None = None
    if not upstream.is_dir():
        failures.append("missing nested upstream TetrisAI repository")
    elif is_linked_directory(upstream):
        failures.append("nested upstream TetrisAI repository is symlink/reparse")
    else:
        try:
            head = git(upstream, "rev-parse", "HEAD")
            status = git(upstream, "status", "--porcelain=v1", "--untracked-files=all")
        except (OSError, subprocess.CalledProcessError) as error:
            failures.append(f"nested upstream TetrisAI git audit failed: {error}")
        if head is not None and head != UPSTREAM_TETRISAI_HEAD:
            failures.append("nested upstream TetrisAI HEAD mismatch")
        if status:
            failures.append("nested upstream TetrisAI repository is not clean")
        for relative, expected in (
            (
                "src/core/components/node.jl",
                RUNTIME_CLOSURE_PINS[EXPERIMENT_PREFIX + "vendor/TetrisAI/src/core/components/node.jl"][1],
            ),
            (
                "src/core/analyzer.jl",
                RUNTIME_CLOSURE_PINS[EXPERIMENT_PREFIX + "vendor/TetrisAI/src/core/analyzer.jl"][1],
            ),
        ):
            source = upstream / Path(relative)
            if not source.is_file() or sha256_file(source) != expected:
                failures.append(f"nested upstream TetrisAI original hash mismatch: {relative}")
    return {
        "valid": not failures,
        "failures": failures,
        "path": str(upstream),
        "head": head,
        "expected_head": UPSTREAM_TETRISAI_HEAD,
        "clean": status == "" if status is not None else False,
    }


def validate_changed_paths(changed: list[str]) -> list[str]:
    failures: list[str] = []
    if not changed:
        failures.append("authorized R1 commit chain changes no paths")
    escaped = [path for path in changed if not path.startswith(EXPERIMENT_PREFIX)]
    if escaped:
        failures.append("authorized R1 commit chain escapes the R1 experiment namespace: " + ", ".join(escaped))
    return failures


def validate_repository_binding(repository: Path, authorized_commit: str) -> dict[str, Any]:
    failures: list[str] = []
    head = git(repository, "rev-parse", "HEAD")
    if head != authorized_commit:
        failures.append("explicitly authorized R1 commit is not live HEAD")
    base_is_ancestor = git_is_ancestor(repository, AUTHORIZED_BASE_COMMIT, head)
    if not base_is_ancestor:
        failures.append("preregistered R1 base is not an ancestor of live HEAD")
    status = git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        failures.append("repository is not clean, including untracked files")
    implementation_parent_is_ancestor = git_is_ancestor(
        repository, AUTHORIZED_IMPLEMENTATION_PARENT_COMMIT, head
    )
    if not implementation_parent_is_ancestor:
        failures.append("frozen R1 implementation parent is not an ancestor of live HEAD")
    changed = (
        [
            line
            for line in git(
                repository,
                "diff",
                "--name-only",
                f"{AUTHORIZED_IMPLEMENTATION_PARENT_COMMIT}..HEAD",
            ).splitlines()
            if line
        ]
        if implementation_parent_is_ancestor
        else []
    )
    failures.extend(validate_changed_paths(changed))
    parents = git(repository, "rev-list", "--parents", "-n", "1", "HEAD").split()
    return {
        "valid": not failures,
        "failures": failures,
        "head": head,
        "parent": parents[1] if len(parents) == 2 else None,
        "authorized_base_commit": AUTHORIZED_BASE_COMMIT,
        "base_is_ancestor": base_is_ancestor,
        "authorized_implementation_parent_commit": AUTHORIZED_IMPLEMENTATION_PARENT_COMMIT,
        "implementation_parent_is_ancestor": implementation_parent_is_ancestor,
        "authorized_commit": authorized_commit,
        "changed_paths": changed,
        "repository_clean": not status,
    }


def audit(repository: Path, fingerprint_path: Path, authorized_commit: str) -> dict[str, Any]:
    document = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    fingerprint = validate_fingerprint_document(repository, document)
    binding = validate_repository_binding(repository, authorized_commit)
    upstream = validate_upstream_tetrisai_binding(repository)
    failures = [*fingerprint["failures"], *binding["failures"], *upstream["failures"]]
    return {
        "valid": not failures,
        "failures": failures,
        "fingerprint_path": str(fingerprint_path.resolve()),
        "fingerprint_file_sha256": sha256_file(fingerprint_path),
        "fingerprint": fingerprint,
        "repository_binding": binding,
        "upstream_tetrisai_binding": upstream,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("repository", type=Path)
    parser.add_argument("fingerprint", type=Path)
    parser.add_argument("authorized_commit", nargs="?")
    args = parser.parse_args()
    if args.generate:
        if args.authorized_commit is not None:
            parser.error("authorized_commit is not accepted with --generate")
        print(json.dumps(atomic_write_fingerprint(args.repository, args.fingerprint), separators=(",", ":")))
        return
    if args.authorized_commit is None or not re.fullmatch(r"[0-9a-fA-F]{40}", args.authorized_commit):
        parser.error("authorized_commit must be full 40-hex")
    result = audit(args.repository.resolve(), args.fingerprint.resolve(), args.authorized_commit.lower())
    print(json.dumps(result, separators=(",", ":")))
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
