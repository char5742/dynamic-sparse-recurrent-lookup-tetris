from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import syntax_preflight


class SyntaxPreflightTests(unittest.TestCase):
    def test_guarded_main_requires_unambiguous_idiom(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r1-syntax-") as temporary:
            path = Path(temporary) / "entry.jl"
            path.write_text(
                "if abspath(PROGRAM_FILE) == abspath(@__FILE__)\n    main()\nend\n",
                encoding="utf-8",
            )
            self.assertTrue(syntax_preflight.guarded_main_ok(path))
            path.write_text("abspath(PROGRAM_FILE) == @__FILE__ && main()\n", encoding="utf-8")
            self.assertFalse(syntax_preflight.guarded_main_ok(path))

    def test_python_parser_rejects_invalid_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r1-python-syntax-") as temporary:
            path = Path(temporary) / "bad.py"
            path.write_text("def broken(:\n", encoding="utf-8")
            with self.assertRaises(SyntaxError):
                syntax_preflight.python_syntax([path])


if __name__ == "__main__":
    unittest.main()
