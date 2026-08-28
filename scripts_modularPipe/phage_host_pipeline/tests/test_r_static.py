"""
Static checks over the R scripts — no R runtime required.

The Python side has test_static.py; this is its counterpart. It exists because
two real bugs reached a live run through the R layer, both introduced by merging
scripts, and neither visible without executing R:

  * a trailing comment was swallowed into a call, producing
    `library(patchwork# for marginal-hist composites)` — a syntax error
  * several `library(a); library(b)` statements ended up nested inside another
    `library(...)` call

Both are structural and detectable from the text alone.

  test_balanced_delimiters   braces, parens and brackets balance (comments and
                             string literals excluded)
  test_library_calls_valid   every library()/require() names one bare package
  test_no_duplicate_defs     no top-level name assigned twice in one file
  test_sourced_files_exist   every source(...) target is present

Run:  pytest tests/    or    python tests/test_r_static.py
"""
from __future__ import annotations

import collections
import pathlib
import re

PIPELINE_DIR = pathlib.Path(__file__).resolve().parent.parent
R_FILES = sorted(PIPELINE_DIR.glob("*.R"))


def _strip_noise(text: str) -> str:
    """Remove string literals and comments so delimiters inside them don't count."""
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":                       # string literal
            quote, i = ch, i + 1
            while i < n and text[i] != quote:
                i += 2 if text[i] == "\\" else 1
            i += 1
            out.append('""')
        elif ch == "#":                       # comment to end of line
            while i < n and text[i] != "\n":
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def test_balanced_delimiters() -> None:
    problems = []
    for path in R_FILES:
        code = _strip_noise(path.read_text(encoding="utf-8"))
        for opener, closer, label in (("{", "}", "braces"),
                                      ("(", ")", "parens"),
                                      ("[", "]", "brackets")):
            delta = code.count(opener) - code.count(closer)
            if delta:
                problems.append(f"{path.name}: {label} unbalanced by {delta:+d}")
    assert not problems, "unbalanced delimiters:\n  " + "\n  ".join(problems)


def test_library_calls_valid() -> None:
    """Each library()/require() call must name exactly one bare package.

    Catches `library(pkg# comment)` and `library(library(a); library(b))`, both
    of which a merge produced.
    """
    valid = re.compile(r"^(library|require)\(\s*[A-Za-z][\w.]*\s*\)$")
    problems = []
    for path in R_FILES:
        for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = raw.split("#", 1)[0].strip()
            if not code.startswith(("library(", "require(")):
                continue
            for stmt in [s.strip() for s in code.split(";") if s.strip()]:
                if not valid.match(stmt):
                    problems.append(f"{path.name}:{lineno}  {stmt}")
    assert not problems, "malformed library() calls:\n  " + "\n  ".join(problems)


def test_no_duplicate_defs() -> None:
    """A second top-level assignment to the same name shadows the first."""
    assign = re.compile(r"^([A-Za-z._][\w.]*)\s*(<-|=)(?!=)")
    problems = []
    for path in R_FILES:
        seen: dict[str, list[int]] = collections.defaultdict(list)
        for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if raw[:1].isspace() or raw.lstrip().startswith("#"):
                continue                      # indented => inside a block/local()
            m = assign.match(raw)
            if m:
                seen[m.group(1)].append(lineno)
        for name, lines in seen.items():
            if len(lines) > 1:
                problems.append(f"{path.name}: '{name}' assigned at top level "
                                f"on lines {lines}")
    assert not problems, "duplicate top-level assignments:\n  " + "\n  ".join(problems)


def test_sourced_files_exist() -> None:
    problems = []
    for path in R_FILES:
        for target in re.findall(r'"([\w.]+\.R)"', path.read_text(encoding="utf-8")):
            if not (PIPELINE_DIR / target).exists():
                problems.append(f"{path.name}: source target '{target}' does not exist")
    assert not problems, "missing sourced files:\n  " + "\n  ".join(problems)


def main() -> int:
    checks = (test_balanced_delimiters, test_library_calls_valid,
              test_no_duplicate_defs, test_sourced_files_exist)
    failed = 0
    for check in checks:
        try:
            check()
            print(f"PASS  {check.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {check.__name__}\n{exc}")
    print(f"\n{len(R_FILES)} R files checked, {failed} check(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
