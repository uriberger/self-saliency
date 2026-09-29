# Copyright 2026 NVIDIA. Apache-2.0.
"""Every flag an experiment runner passes must be one the program it runs accepts.

`docs/reproduce.md` gives one command per table and figure, and for three of them that
command is a `run.sh`. A runner is a projection of a Python program's argument surface
into shell, and nothing holds the two together: argparse rejects an unknown flag with
SystemExit(2), so a renamed flag turns into a run that dies at launch -- after the
allocation, the model load and, for the sharded stages, on every shard at once.

It is not hypothetical. Writing these three found two: `head_selection/run.sh` passed
`--out` to a program whose flag is `--output`, and `--metric` to one whose flag is
`--metrics`. Both look right, neither is, and the usage examples in the archive's
launchers were the source of both.

Scoped to the three experiment runners, where every invocation is `python -m <module>` of
a module in this repository, so the check needs no allowlist and cannot rot into one.
`training/grpo/run.sh` is deliberately not covered: it also drives `accelerate launch`,
`trl.scripts.vllm_serve` and the patched entry script inside a TRL checkout, none of which
is introspectable from here -- `tests/test_entry_script_arguments.py` is the check that
its flags line up, by a different route.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

RUNNERS = [
    "experiments/attention_bias/run.sh",
    "experiments/trained_model/run.sh",
    "experiments/head_selection/run.sh",
]


def _module_flags(module: str) -> set[str] | None:
    """Every `--flag` string literal in an add_argument call, or None if not ours."""
    path = ROOT / (module.replace(".", "/") + ".py")
    if not path.exists():
        return None
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                        and arg.value.startswith("-"):
                    out.add(arg.value)
    return out


def _runner_text(runner: str) -> str:
    return (ROOT / runner).read_text()


def _own_flags(text: str) -> set[str]:
    """Flags the runner itself consumes, i.e. the labels of its own `case` arms."""
    own = set()
    for line in text.splitlines():
        m = re.match(r"\s*(-[^)]*)\)\s", line)
        if m:
            own.update(f for f in m.group(1).split("|") if f.startswith("-"))
    return own


def _invoked_modules(text: str) -> set[str]:
    """Modules the runner runs as `-m <dotted>`."""
    return set(re.findall(r"-m\s+([A-Za-z_][\w.]*)", text))


def _passed_flags(text: str) -> set[str]:
    """Every `--flag` the runner hands to a PYTHON invocation.

    Scoped to `$PYTHON`/`python` commands and their backslash continuations, because a
    runner also calls other programs and their flags are not argparse's to accept --
    `nvidia-smi --list-gpus` is the one that made this necessary. Comments are dropped:
    the usage headers spell flags the runner accepts anyway, and a prose mention of a
    forwarded flag is not a call.
    """
    flags: set[str] = set()
    in_python_cmd = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        starts = bool(re.search(r'(^|[|;&(]|\s)"?\$?\{?PYTHON\}?"?\s|(^|\s)python3?\s', line))
        if starts or in_python_cmd:
            flags.update(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", line))
        in_python_cmd = (starts or in_python_cmd) and line.endswith("\\")
    return flags


@pytest.mark.parametrize("runner", RUNNERS)
def test_runner_exists_and_is_executable(runner):
    path = ROOT / runner
    assert path.exists(), f"{runner} is promised by docs/reproduce.md and is missing"
    assert path.stat().st_mode & 0o111, f"{runner} is not executable"


@pytest.mark.parametrize("runner", RUNNERS)
def test_runner_invokes_only_modules_that_exist(runner):
    """`python -m experiments.x.y` must name a module in this repository."""
    text = _runner_text(runner)
    modules = _invoked_modules(text)
    assert modules, f"{runner} runs no module; has the invocation style changed?"
    missing = sorted(m for m in modules if _module_flags(m) is None)
    assert not missing, f"{runner} runs {missing}, which do not exist here"


@pytest.mark.parametrize("runner", RUNNERS)
def test_every_flag_the_runner_passes_is_accepted(runner):
    """The check that matters: no flag may be one argparse will reject.

    A flag is fine if the runner consumes it itself, or if ANY of the programs it runs
    declares it. Any is the right quantifier and not all: a runner drives several stages
    and forwards `--shard` to the sharded ones only.
    """
    text = _runner_text(runner)
    accepted = set(_own_flags(text))
    for module in _invoked_modules(text):
        flags = _module_flags(module)
        if flags:
            accepted |= flags

    unknown = sorted(_passed_flags(text) - accepted)
    assert not unknown, (
        f"{runner} passes {unknown}, which neither it nor any program it runs declares. "
        f"argparse exits 2 on an unknown flag, so this is a run that dies at launch")
