# Copyright 2026 NVIDIA. Apache-2.0.
"""No personal or machine-specific identifier may re-enter the tree.

The repository was assembled from two working trees on one cluster, and the first sweep
found identifiers in 293 tracked files -- 277 of them inside banked evaluation results,
where absolute paths are recorded as run metadata by lmms-eval itself. A path like
`/home/<user>/scratch/...` is harmless on the machine it came from and is a username on
any other, so this keeps them out rather than relying on remembering.

It also catches the ordinary bug underneath: a hardcoded home directory in a script is a
script that works for exactly one person, and fails for everyone else with a confusing
"no such file".

The pinned lmms-eval submodule is excluded -- it is upstream's tree, not ours.

ORGANISATIONAL identifiers are deliberately NOT checked here. `inference-api.nvidia.com`
is the LLM judge's gateway and is a real configuration default, not a leak. It does need
a decision before the repository is made public, for the reason in docs/publishing.md,
but failing a test on it every day until then would just train people to ignore this one.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Anything that names a person, a home directory, or one cluster's storage layout.
FORBIDDEN = [
    (re.compile(r"/home/[a-z][a-z0-9_-]*/", re.I), "a hardcoded home directory"),
    (re.compile(r"/lustre/[^/\s\"']+/portfolios/", re.I), "a cluster storage path"),
    (re.compile(r"/mnt/c/Users/", re.I), "a Windows user directory"),
    (re.compile(r"\bnvr_israel_rlop\b"), "an internal project account"),
    (re.compile(r"\buberger\b", re.I), "a username"),
]

#: Paths where a match is expected and correct.
ALLOWED = (
    "tests/",                 # this file names them on purpose, as do the archive tests
    "evaluation/lmms_eval/",  # upstream's submodule
    "docs/publishing.md",     # the checklist that discusses them
)


def tracked_files():
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files"],
                         capture_output=True, text=True, check=True).stdout
    for rel in out.split("\n"):
        if not rel or rel.startswith(ALLOWED):
            continue
        path = ROOT / rel
        if not path.is_file():
            continue
        yield rel, path


@pytest.mark.parametrize("rel,path", list(tracked_files()), ids=lambda x: x if isinstance(x, str) else "")
def test_no_private_identifier(rel, path):
    try:
        text = path.read_text(errors="strict")
    except (UnicodeDecodeError, OSError):
        return                                   # binary; nothing to read
    hits = []
    for pattern, what in FORBIDDEN:
        for m in pattern.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            hits.append(f"line {line}: {what} -- {m.group(0)!r}")
    assert not hits, f"{rel}\n  " + "\n  ".join(hits[:6])


def test_the_check_covers_the_banked_results():
    """A silent no-op would pass every case above.

    The banked results are where nearly every identifier was, so if they somehow stopped
    being scanned this test would go green while checking almost nothing.
    """
    scanned = {rel for rel, _ in tracked_files()}
    banked = {r for r in scanned if r.startswith("evaluation/results/")}
    assert len(banked) > 200, (
        f"only {len(banked)} banked result files are being scanned; the sweep that "
        f"motivated this test found identifiers in 277 of them")
