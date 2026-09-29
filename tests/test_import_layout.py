# Copyright 2026 NVIDIA. Apache-2.0.
"""Every import in trl_patch/ must still resolve after the patch script moves the file.

`training/grpo/trl_patch/` is the tracked source; a TRL checkout is what executes; and
the two have DIFFERENT layouts. A relative import is resolved against the directory a
file ENDS UP in, so `from .x import y` can be correct where it is written and wrong where
it lands.

Nothing else catches this. The CPU tests load these modules where they are, and there
they import fine; the layout only differs in a tree no CPU test touches. So the failure
surfaces on a GPU, inside a live GRPO step, at whichever line first runs the bad import
-- which for the reward is the diagnostics drain, AFTER generation, the re-forward, the
backward and the detector calls of step 0. That is roughly forty minutes and eight GPUs
to learn about a typo.

It has bitten twice in the research tree: once a module moved package and its sibling
import broke, and once the patch script simply forgot a `cp` and the failure appeared
only on a second cluster, because the first had the file copied in by hand.

Two things are checked:

  1. every relative import in a patched file resolves at its DESTINATION
  2. every module the patched files import from `trl.*` is in the copy list, or is
     upstream TRL's own

Absolute imports into `selfsal` are exempt by construction -- an installed package
resolves the same wherever the importing file sits, which is the reason the method was
moved into one.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH_SCRIPT = ROOT / "env/patches/trl.sh"
SRC = ROOT / "training/grpo/trl_patch"


def copy_list() -> dict[str, str]:
    """{source relative to trl_patch : destination relative to the TRL checkout}.

    Parsed out of the patch script rather than duplicated here. A second copy of this
    mapping would be a second thing to keep in step, and keeping things in step by hand
    is the failure mode this whole file is about.
    """
    text = PATCH_SCRIPT.read_text()
    block = re.search(r"^COPIES=\((.*?)^\)", text, re.MULTILINE | re.DOTALL)
    assert block, "could not find the COPIES array in the patch script"
    pairs = re.findall(r'"([^":]+):([^"]+)"', block.group(1))
    assert pairs, "the COPIES array parsed to nothing"
    return dict(pairs)


@pytest.fixture(scope="module")
def copies():
    return copy_list()


def test_every_copied_source_exists(copies):
    missing = [s for s in copies if not (SRC / s).exists()]
    assert not missing, f"the patch script copies files that do not exist: {missing}"


def test_every_source_is_copied(copies):
    """A file in trl_patch/ that the script does not install is dead on the GPU."""
    on_disk = {
        str(p.relative_to(SRC)) for p in SRC.rglob("*.py")
        if "__pycache__" not in p.parts
    }
    uncopied = on_disk - set(copies)
    assert not uncopied, (
        f"{sorted(uncopied)} are in trl_patch/ but not in the patch script's copy list, "
        f"so they will not exist in the checkout that actually runs")


def _imports(path: Path):
    """(level, module, lineno) for every import in a file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            yield node.level, node.module, node.lineno
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield 0, alias.name, node.lineno


#: Packages upstream TRL provides. A relative import that lands on one of these is fine.
UPSTREAM_TRL = {
    "data_utils", "extras", "import_utils", "models", "trainer", "scripts", "rewards",
    "callbacks", "grpo_config", "utils", "modeling_value_head", "extras.profiling",
    "extras.vllm_client", "models.utils",
}


@pytest.mark.parametrize("source", sorted(copy_list()))
def test_relative_imports_resolve_at_the_destination(copies, source):
    """Resolve each relative import against where the file LANDS, not where it lives."""
    dest = Path(copies[source])
    problems = []

    for level, module, lineno in _imports(SRC / source):
        if level == 0:
            continue
        # `from . import x` inside trl/rewards/foo.py resolves to trl.rewards.x;
        # `from ..y import z` resolves to trl.y.z.
        package = dest.parent.parts            # e.g. ("trl", "rewards")
        if level - 1 > len(package):
            problems.append(f"line {lineno}: {'.' * level}{module or ''} escapes the tree")
            continue
        base = package[: len(package) - (level - 1)]
        target = ".".join(base + tuple((module or "").split(".")[:1])).strip(".")
        head = target.split(".", 1)[-1] if target.startswith("trl.") else target

        # Either it is one of ours (and therefore in the copy list at that path), or it
        # is upstream TRL's.
        as_path = "/".join(base[1:] + tuple((module or "").split("."))) + ".py"
        ours = as_path in set(copies.values()) or f"trl/{as_path}" in set(copies.values())
        if not ours and head not in UPSTREAM_TRL and (module or "") not in UPSTREAM_TRL:
            problems.append(
                f"line {lineno}: {'.' * level}{module} resolves to {target!r} at "
                f"{dest}, which is neither copied by the patch script nor upstream TRL")

    assert not problems, f"{source} -> {dest}:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("source", sorted(copy_list()))
def test_cross_module_imports_name_a_copied_file(copies, source):
    """`from trl.rewards.X import ...` must name something the script installs.

    THE PACKAGE BEING UPSTREAM'S IS NOT ENOUGH. `trl.rewards` is upstream TRL's own
    package, but every module inside it that this repository uses is one this repository
    installs -- so accepting an import because its SECOND component is in UPSTREAM_TRL
    accepts `trl.rewards.overlap_rewards` too, which is a research module that was left
    in the archive. That is exactly what happened: the entry script was copied across the
    port without being rewired, kept seven such imports, and this test passed on all of
    them. Training could not start.

    So `trl.<pkg>.<mod>` is checked at the MODULE, not at the package: either the patch
    script installs that file, or it must be an upstream module named in UPSTREAM_TRL in
    full. A bare `trl.<pkg>` is still allowed on the package name alone -- there is no
    file to point at -- and `test_from_trl_rewards_names_an_export` covers what may be
    taken out of it.
    """
    installed = set(copies.values())
    problems = []
    for level, module, lineno in _imports(SRC / source):
        if level != 0 or not module or not module.startswith("trl."):
            continue
        as_path = module.replace(".", "/") + ".py"
        if as_path in installed:
            continue
        parts = module.split(".")
        # `trl.x` (a package, or an upstream module) -- nothing deeper to resolve.
        if len(parts) == 2 and parts[1] in UPSTREAM_TRL:
            continue
        # `trl.x.y` and deeper: the whole tail has to be upstream's, not just its head.
        if ".".join(parts[1:]) in UPSTREAM_TRL:
            continue
        problems.append(f"line {lineno}: {module} is not installed by the patch script")
    assert not problems, f"{source}:\n  " + "\n  ".join(problems)


def _rewards_exports() -> set[str]:
    """Names `rewards/__init__.py` binds, read out of the file rather than imported.

    Importing it would need torch, transformers and an installed `selfsal`; this test has
    to run in the bare CPU environment, which is the whole point of it running at all.
    """
    tree = ast.parse((SRC / "rewards/__init__.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


@pytest.mark.parametrize("source", sorted(copy_list()))
def test_from_trl_rewards_names_an_export(copies, source):
    """`from trl.rewards import X` must name something `rewards/__init__.py` binds.

    The other half of the same hole. The entry script's line 87 asked this package for
    `openai_reward`, which moved to `selfsal.judge` during the port and was never
    re-exported here -- an ImportError on the first line of training, found on a GPU.

    The failure is worth catching cheaply because it is invisible on this machine: the
    archive's TRL checkout still exports every one of these names, so anyone with the
    research environment on their path imports the module and sees nothing wrong.
    """
    exports = _rewards_exports()
    tree = ast.parse((SRC / source).read_text())
    problems = [
        f"line {node.lineno}: `from trl.rewards import {alias.name}` but "
        f"rewards/__init__.py binds no such name"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0
        and node.module == "trl.rewards"
        for alias in node.names
        if alias.name not in exports
    ]
    assert not problems, f"{source}:\n  " + "\n  ".join(problems)


def test_the_method_is_imported_absolutely(copies):
    """selfsal must never be reached relatively -- that is why it is a package.

    A relative import of the method would put the layout hazard straight back.
    """
    offenders = []
    for source in copies:
        for level, module, lineno in _imports(SRC / source):
            if level > 0 and (module or "").startswith("selfsal"):
                offenders.append(f"{source}:{lineno}")
    assert not offenders, (
        f"selfsal is imported relatively in {offenders}; it is an installed package and "
        f"must be imported absolutely so the import cannot depend on where the file lands")


#: Modules of the research arms that did not come across. `docs/provenance.md` lists them.
DROPPED_VARIANT_MODULES = (
    "grad_maps", "glimpse_maps", "grad_rewards", "glimpse_rewards", "placebo_rewards",
    "maskfree_rewards", "mismatch_rewards", "length_guard_rewards", "roll_null",
    "overlap_rewards",
)


@pytest.mark.parametrize("source", sorted(copy_list()))
def test_no_patched_file_references_the_dropped_variants(source):
    """The gradient, GLIMPSE and control arms are not in the paper, and not installed.

    Checked over every patched file, not just the trainer. It was the trainer alone that
    this covered, and the entry script -- which is where all seven of these imports
    actually were -- went unchecked.
    """
    tree = ast.parse((SRC / source).read_text())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if any(k in node.module for k in DROPPED_VARIANT_MODULES):
                bad.append(f"line {node.lineno}: {node.module}")
    assert not bad, (
        f"{source} imports modules from research arms that are not in this repository "
        "and not installed by the patch script:\n  " + "\n  ".join(bad))
