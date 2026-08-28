"""
Static checks over every module — no data, no models, no torch required.

No runtime test in the repo reaches latent.py, which needs torch, and none
exercises the sklearn stages end to end. This file closes that gap with checks
that need nothing but the source.

It exists because of a real bug. Merging export_latent_space.py and
probe_latent_spaces.py into latent.py silently produced two module-level
functions both called `_model_targets`, with different signatures. Python keeps
the last definition, so every call to the first one broke — and it surfaced only
when the pipeline reached the latent stage, minutes into a real run.

  test_no_duplicate_definitions  no name is defined twice at module level
  test_call_arity                every call to a module-level function passes an
                                 argument count its signature accepts
  test_modules_import            every module imports cleanly (skipping any that
                                 need an optional dependency that is absent)

Run:  pytest tests/    or    python tests/test_static.py
"""
from __future__ import annotations

import ast
import collections
import importlib
import pathlib
import sys

PIPELINE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODULES = sorted(p for p in PIPELINE_DIR.glob("*.py") if p.name != "__init__.py")

# Modules whose import pulls in an optional heavyweight dependency.
OPTIONAL_DEPS = {"latent.py": "torch"}


def _parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _module_level_defs(tree: ast.Module) -> dict[str, list[int]]:
    """name -> line numbers, for everything defined at module level."""
    names: dict[str, list[int]] = collections.defaultdict(list)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names[node.name].append(node.lineno)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names[target.id].append(node.lineno)
    return names


def test_no_duplicate_definitions() -> None:
    """A second definition of the same name silently replaces the first."""
    problems = []
    for path in MODULES:
        for name, lines in _module_level_defs(_parse(path)).items():
            if len(lines) > 1:
                problems.append(f"{path.name}: '{name}' defined at lines {lines} "
                                f"— the later one shadows the earlier")
    assert not problems, "duplicate module-level definitions:\n  " + "\n  ".join(problems)


def test_call_arity() -> None:
    """Every call to a locally defined function must fit that function's signature.

    Catches the case where two merged files defined the same function name with
    different arities: the surviving definition makes some call sites invalid,
    which Python only reports when that line finally executes.
    """
    problems = []
    for path in MODULES:
        tree = _parse(path)
        sigs = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                pos = node.args.posonlyargs + node.args.args
                required = len(pos) - len(node.args.defaults)
                sigs[node.name] = (required, None if node.args.vararg else len(pos))

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            sig = sigs.get(node.func.id)
            if sig is None:
                continue
            if any(isinstance(a, ast.Starred) for a in node.args):
                continue                      # *args at the call site: unknowable
            required, maximum = sig
            given = len(node.args)
            kwargs = {k.arg for k in node.keywords if k.arg}
            if any(k.arg is None for k in node.keywords):
                continue                      # **kwargs at the call site
            if given + len(kwargs) < required:
                problems.append(f"{path.name}:{node.lineno} {node.func.id}() got "
                                f"{given} positional + {len(kwargs)} keyword, needs {required}")
            elif maximum is not None and given > maximum:
                problems.append(f"{path.name}:{node.lineno} {node.func.id}() got "
                                f"{given} positional, takes at most {maximum}")
    assert not problems, "call/signature mismatches:\n  " + "\n  ".join(problems)


def test_modules_import() -> None:
    """Every module imports cleanly, so a syntax or name error cannot lurk."""
    sys.path.insert(0, str(PIPELINE_DIR))
    failures, skipped = [], []
    for path in MODULES:
        dep = OPTIONAL_DEPS.get(path.name)
        if dep and importlib.util.find_spec(dep) is None:
            skipped.append(f"{path.name} (needs {dep})")
            continue
        try:
            importlib.import_module(path.stem)
        except SystemExit:
            pass                              # argparse in a __main__ guard
        except Exception as exc:              # noqa: BLE001 - report anything
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
    if skipped:
        print(f"  skipped (optional dependency missing): {', '.join(skipped)}")
    assert not failures, "modules failed to import:\n  " + "\n  ".join(failures)


def main() -> int:
    failed = 0
    for check in (test_no_duplicate_definitions, test_call_arity, test_modules_import):
        try:
            check()
            print(f"PASS  {check.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {check.__name__}\n{exc}")
    print(f"\n{len(MODULES)} modules checked, {failed} check(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
