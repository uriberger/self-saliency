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
    """`from trl.rewards.X import ...` must name something the script installs."""
    installed = set(copies.values())
    problems = []
    for level, module, lineno in _imports(SRC / source):
        if level != 0 or not module or not module.startswith("trl."):
            continue
        as_path = module.replace(".", "/") + ".py"
        if as_path in installed:
            continue
        if module.split(".")[1] in UPSTREAM_TRL:
            continue
        problems.append(f"line {lineno}: {module} is not installed by the patch script")
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


def test_trainer_no_longer_references_the_dropped_variants():
    """grad and glimpse are not in the paper and their modules are not installed."""
    trainer = (SRC / "grpo_trainer_qwen3.py").read_text()
    tree = ast.parse(trainer)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if any(k in node.module for k in ("grad_maps", "glimpse_maps", "grad_rewards",
                                              "glimpse_rewards", "placebo_rewards",
                                              "maskfree_rewards", "mismatch_rewards",
                                              "length_guard_rewards", "roll_null")):
                bad.append(f"line {node.lineno}: {node.module}")
    assert not bad, (
        "the trainer imports modules from research arms that are not in this repository "
        "and not installed by the patch script:\n  " + "\n  ".join(bad))
