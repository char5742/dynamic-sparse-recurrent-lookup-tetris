from __future__ import annotations

import argparse
import ast
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_ENTRYPOINTS = ("freeze_design.jl",)
GUARDED_MAIN = re.compile(
    r"if\s+abspath\(PROGRAM_FILE\)\s*==\s*abspath\(@__FILE__\)\s*\n\s*main\(\)\s*\n\s*end",
    re.MULTILINE,
)


def python_syntax(paths: list[Path]) -> None:
    for path in paths:
        ast.parse(path.read_bytes(), filename=str(path))


def guarded_main_ok(path: Path) -> bool:
    return GUARDED_MAIN.search(path.read_text(encoding="utf-8")) is not None


def strict_julia_syntax(julia: Path, paths: list[Path]) -> None:
    if not paths:
        return
    code = (
        "for path in ARGS; "
        "Base.JuliaSyntax.parseall(Base.JuliaSyntax.SyntaxNode, read(path,String); "
        "filename=path, ignore_errors=false); end"
    )
    subprocess.run(
        [str(julia), "--startup-file=no", "--history-file=no", "-e", code, *map(str, paths)],
        check=True,
    )


def prove_strict_rejection(julia: Path) -> None:
    code = (
        "try; Base.JuliaSyntax.parseall(Base.JuliaSyntax.SyntaxNode, read(ARGS[1],String); "
        "filename=ARGS[1], ignore_errors=false); exit(2); catch; exit(0); end"
    )
    with tempfile.TemporaryDirectory(prefix="r1-julia-syntax-") as temporary:
        malformed = Path(temporary) / "macro_precedence.jl"
        malformed.write_text("abspath(PROGRAM_FILE) == @__FILE__ && main()\n", encoding="utf-8")
        subprocess.run(
            [str(julia), "--startup-file=no", "--history-file=no", "-e", code, str(malformed)],
            check=True,
        )


def recursive_sources(root: Path, suffix: str) -> list[Path]:
    return sorted(
        path for path in root.rglob(f"*{suffix}")
        if "__pycache__" not in path.parts
    )


def run_preflight(root: Path, julia: Path, entrypoints: tuple[str, ...]) -> dict[str, Any]:
    root = root.resolve()
    python_paths = recursive_sources(root, ".py")
    julia_paths = recursive_sources(root, ".jl")
    python_syntax(python_paths)
    strict_julia_syntax(julia.resolve(), julia_paths)
    prove_strict_rejection(julia.resolve())
    missing = [name for name in entrypoints if not (root / name).is_file()]
    malformed = [name for name in entrypoints if (root / name).is_file() and not guarded_main_ok(root / name)]
    if missing or malformed:
        raise ValueError(f"entrypoint guard failure; missing={missing}; malformed={malformed}")
    return {
        "status": "r1_syntax_preflight_passed",
        "python_files": len(python_paths),
        "julia_files": len(julia_paths),
        "entrypoints": list(entrypoints),
        "strict_julia_parser": True,
        "malformed_macro_precedence_fixture_rejected": True,
        "model_or_checkpoint_loaded": False,
        "game_run": False,
        "seed_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--julia", required=True, type=Path)
    parser.add_argument("--entrypoint", action="append")
    args = parser.parse_args()
    entrypoints = tuple(args.entrypoint or DEFAULT_ENTRYPOINTS)
    print(run_preflight(args.root, args.julia, entrypoints))


if __name__ == "__main__":
    main()
