# Copyright 2026 NVIDIA. Apache-2.0.
"""Nothing here may reach back into the archive repos.

The port moved files that used to load their neighbours by relative path -- the probes
imported the trainer's reward module out of `trl/`, the Section 5 scan imported an
attention-edit module, the head-selection screen imported an inference runner. Every one
of those paths still EXISTS on the machine this was ported on, so a missed rewiring does
not fail here: it silently runs the archive's copy of the code, and the new repo looks
correct while testing something else.

It would fail on anyone else's machine, at import, with a path in it that means nothing
to them. So the check belongs here, where the archive is present and a leak is
detectable.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Names that only ever resolved inside the archive repos.
FORBIDDEN_MODULES = {
    "sink_shift", "sink_location", "sink_location_probe", "vlm_family",
    "intervene_probe", "overlap_probe", "flow_correlation_probe", "laser", "laser_probe",
    "grad_maps", "glimpse_maps", "overlap_steps", "overlap_rewards", "grad_rewards",
    "glimpse_rewards", "placebo_rewards", "maskfree_rewards", "mismatch_rewards",
    "roll_null", "saliency_rewards", "aggregation_correlation", "run_experiment",
}

#: Directory names from the repos this was assembled from.
FORBIDDEN_PATHS = ("research/saliency_r1/", "research/vlm_reasoning/",
                   "scratch/research/saliency_r1", "scratch/research/vlm_reasoning")


def _python_files():
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if rel.parts[0] in (".git", "build", "dist") or "__pycache__" in rel.parts:
            continue
        if rel.parts[0] == "tests":          # this file names them on purpose
            continue
        yield path


@pytest.mark.parametrize("path", list(_python_files()), ids=lambda p: str(p.name))
def test_no_archive_module_imports(path):
    """No `import <archive module>` survived the rewiring."""
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_MODULES:
                    offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module.split(".")[0] in FORBIDDEN_MODULES:
                offenders.append(node.module)
    assert not offenders, (
        f"{path.relative_to(ROOT)} imports {sorted(set(offenders))}, which only resolve "
        f"inside the archive repos")


@pytest.mark.parametrize("path", list(_python_files()), ids=lambda p: str(p.name))
def test_no_archive_paths_outside_comments(path):
    """No archive path is used as a VALUE.

    Mentioning one in prose is fine and often useful -- several modules point at where a
    dropped experiment went. Loading from one is not, so this reads string literals out
    of the parsed tree and ignores comments and docstrings.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)

    offenders = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value not in docstrings
        and any(bad in node.value for bad in FORBIDDEN_PATHS)
    ]
    assert not offenders, (
        f"{path.relative_to(ROOT)} carries archive paths as values: "
        f"{[o[:80] for o in offenders]}")
