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


#: The archive's own file names. A module is not always reached by `import` -- the port
#: inherited a lot of `spec_from_file_location(name, "sink_location_probe.py")`, which is
#: an archive dependency spelled as a string and is invisible to the import check below.
#: Every one of these exists here under another name; `docs/provenance.md` maps them.
FORBIDDEN_FILENAMES = {
    "overlap_probe.py", "selfground_audit.py", "intervene_probe.py",
    "flow_correlation_probe.py", "sink_location.py", "sink_location_probe.py",
    "sink_location_xmodel_tables.py", "sink_location_html.py", "build_grpo_sets.py",
    "build_boxed_corpus.py", "overlap_metric_spread.py", "centre_box_probe.py",
    "patch_trl_qwen3.sh",
    "trl/overlap_steps.py", "trl/grad_maps.py", "trl/glimpse_maps.py",
    "trl/rewards/overlap_rewards.py", "trl/rewards/roll_null.py",
    "trl/rewards/placebo_rewards.py", "trl/rewards/maskfree_rewards.py",
    "trl/rewards/mismatch_rewards.py", "trl/rewards/length_guard_rewards.py",
    "trl/rewards/grad_rewards.py", "trl/rewards/glimpse_rewards.py",
}


@pytest.mark.parametrize("path", list(_python_files()), ids=lambda p: str(p.name))
def test_no_archive_module_imports(path):
    """No `import <archive module>` survived the rewiring.

    Checked at every component, not just the first. `from trl.rewards.overlap_rewards
    import x` has `trl` as its head, so a head-only check passes it -- and that is exactly
    what happened: the GRPO entry script carried seven such imports across the port and
    this test, which names `overlap_rewards` outright, said nothing.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        else:
            continue
        for name in names:
            if set(name.split(".")) & FORBIDDEN_MODULES:
                offenders.append(name)
    assert not offenders, (
        f"{path.relative_to(ROOT)} imports {sorted(set(offenders))}, which only resolve "
        f"inside the archive repos")


@pytest.mark.parametrize("path", list(_python_files()), ids=lambda p: str(p.name))
def test_no_archive_filenames_are_loaded(path):
    """No archive FILE NAME is used as a value either.

    The port's commonest survival was not an import but a path load: the archive was flat,
    so its scripts reached each other with
    `spec_from_file_location("_x", REPO / "sink_location_probe.py")`. Here they are
    packages, that file does not exist, and the load raises -- at import for some of them.
    Prose may still mention the old name (several modules point at where something went),
    so this reads string literals out of the parsed tree and ignores docstrings, exactly
    as the path check below does.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)

    # `unavailable("grad maps", "trl/grad_maps.py")` is the opposite of a load: it is a
    # SIGNPOST, and naming the archive path is the whole point of it -- that string ends
    # up in the NotImplementedError telling the reader where the code went. Exempted by
    # where the literal sits, not by its value, so a real load of the same path elsewhere
    # in the file is still caught.
    signposts = {
        id(arg) for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ((isinstance(node.func, ast.Name) and node.func.id == "unavailable")
             or (isinstance(node.func, ast.Attribute) and node.func.attr == "unavailable"))
        for arg in node.args
    }

    offenders = sorted({
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and id(node) not in signposts
        and node.value not in docstrings and node.value in FORBIDDEN_FILENAMES
    })
    assert not offenders, (
        f"{path.relative_to(ROOT)} loads {offenders} by file name. Those are the "
        f"ARCHIVE's names; docs/provenance.md maps each to what it is called here")


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
