# Copyright 2026 NVIDIA. Apache-2.0.
"""The probes must IMPORT even though the arms they were written for did not come across.

Several probes here were written against saliency maps and controls that were explored and
dropped -- the pixel-gradient map, the GLIMPSE map, the attention-rollout flow, the
roll-null, placebo, mask-free and length-guard rewards. `docs/provenance.md` lists them.
The probes came across because their ATTENTION path is the one the paper uses, so each
stands its missing dependency up as an `experiments._unavailable.Unavailable` sentinel
that raises when it is USED, naming the archive.

THAT ONLY WORKS IF NOTHING TOUCHES THE SENTINEL BEFORE THE FLAG IS READ, and two very
ordinary-looking lines do:

    def f(..., span_chunk=GM.SPAN_CHUNK_DEFAULT)     evaluated at MODULE IMPORT
    p.add_argument(..., default=GM.X)                evaluated while the PARSER is built

Both were present. `experiments/trained_model/probe.py` could not be imported at all --
so `--map attn`, the only map in the paper and the one Table 3 rests on, was as dead as
the two maps that are not here -- and two more would have fired while argparse was still
being assembled, before the flag that would have avoided them was read.

Nothing else catches this. Every module here imports torch and transformers, so no other
CPU test imports them; the failure surfaces the first time someone runs the probe, which
is on a GPU allocation.

Two checks, and the first is the one that matters: the modules import. The second is
static, and is what stops the pattern coming back.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Every module that stands a dropped dependency up as a sentinel.
SENTINEL_USERS = [
    "experiments/trained_model/probe.py",
    "experiments/figures/saliency_viz.py",
]


def _sentinel_names(tree: ast.Module) -> set[str]:
    """Names bound to `unavailable(...)` at module scope."""
    names = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        fn = node.value.func
        if (isinstance(fn, ast.Name) and fn.id == "unavailable") or (
                isinstance(fn, ast.Attribute) and fn.attr == "unavailable"):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _reads(node: ast.AST, sentinels: set[str]) -> list[str]:
    return [f"line {n.lineno}: {n.value.id}.{n.attr}"
            for n in ast.walk(node)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id in sentinels]


@pytest.mark.parametrize("source", SENTINEL_USERS)
def test_no_sentinel_is_read_before_a_flag_is(source):
    """No attribute of a sentinel may be read at import or while the parser is built."""
    tree = ast.parse((ROOT / source).read_text())
    sentinels = _sentinel_names(tree)
    assert sentinels, f"{source} binds no unavailable() sentinel; has the pattern moved?"

    problems = []
    # 1. module scope
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        problems += [f"module scope, {r}" for r in _reads(node, sentinels)]
    # 2. function default arguments -- evaluated at def time, i.e. at import
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.args.defaults + [x for x in node.args.kw_defaults if x]:
                problems += [f"default of {node.name}(), {r}" for r in _reads(d, sentinels)]
    # 3. add_argument(...) -- evaluated while the parser is assembled
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            problems += [f"add_argument, {r}" for r in _reads(node, sentinels)]

    assert not problems, (
        f"{source} reads a dropped-variant sentinel before a flag has been parsed, so the "
        f"whole module fails rather than the one map that needed it:\n  "
        + "\n  ".join(problems))


@pytest.mark.parametrize("module", [
    "experiments.trained_model.probe",
    "experiments.trained_model.audit",
    "experiments.figures.saliency_viz",
    "experiments.figures.step_referent",
    "experiments.attention_bias.probe",
    "experiments.attention_bias.observe_boxes",
    "experiments.alpha_calibration",
    "selfsal.steps.evaluate",
    "selfsal.data.boxed_corpus",
])
def test_the_probes_import(module):
    """The port left these loading archive paths by file name; they must import here.

    Skipped where the scientific stack is absent, since that is a missing environment and
    not a wiring fault -- but on any machine that can run the probes at all, this is the
    check that they are wired to this repository rather than to the one they came from.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    __import__(module)


def test_the_sentinel_raises_with_somewhere_to_go():
    """A sentinel's message must name what is missing AND where it went."""
    from experiments._unavailable import unavailable

    sentinel = unavailable("grad maps", "trl/grad_maps.py")
    with pytest.raises(NotImplementedError) as exc:
        sentinel.pixel_regroup
    message = str(exc.value)
    assert "grad maps" in message
    assert "trl/grad_maps.py" in message
    assert "research/saliency_r1" in message

    # Dunders must still raise AttributeError: inspect, copy and pytest's assertion
    # rewriting all probe for them while formatting a failure, and answering with the
    # wrong exception type turns a clear failure into a confusing one.
    with pytest.raises(AttributeError):
        sentinel.__wrapped__
