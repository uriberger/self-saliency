# Copyright 2026 NVIDIA. Apache-2.0.
"""A repo-relative path in the docs must be a file that is here.

`docs/reproduce.md` is one row per table and figure in the paper, and `docs/provenance.md`
is the map from every file here to where it came from. Both are read by someone trying to
rebuild a number, and both spent the port naming files that do not exist: three `run.sh`
runners, the question-box builder, a corpus loader, and a `configs/dapo.yaml` that was
never the mechanism at all -- the DAPO arm is a flag on `baselines/ease/run.sh`.

A wrong path in a document is worse than a missing document, because it reads as a
promise that the thing exists and was checked. Nothing else catches it: the docs are
prose, the tests are code, and until now the two never met.

WHAT COUNTS AS A PATH, deliberately narrowly. Only backtick-quoted strings that look like
a repo-relative path with a known source extension, plus bare directory prefixes in a
`bash ...` command line. Prose that mentions the archive's old names is fine and often
useful -- `docs/provenance.md` is largely a table of them -- so anything naming the
archive is skipped, and so is anything under a directory this repository does not own.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

DOCS = sorted(p.relative_to(ROOT) for p in (ROOT / "docs").glob("*.md"))
# NEXT_SESSION.md is the handoff, and it is the document most likely to name a file that
# has moved -- the previous one promised six that did not exist.
DOCS += [Path("README.md"), Path("NEXT_SESSION.md")]

#: Extensions whose paths are checkable. A `.json`/`.jsonl`/`.parquet` is an OUTPUT --
#: it exists only after something is run -- so naming one is not a promise about the tree.
CHECKED_SUFFIXES = (".py", ".sh", ".yaml", ".yml", ".md", ".toml")

#: The top-level directories this repository owns. A path outside them is someone else's
#: (the archive, a TRL checkout, an EasyR1 checkout, the user's own output tree).
OWNED = ("baselines/", "docs/", "env/", "evaluation/", "experiments/", "selfsal/",
         "tests/", "tools/", "training/")

#: Prefixes that name a tree this repository does not contain.
FOREIGN = ("research/", "archive/", "A ", "V ", "third_party/", "ease_repo/",
           "trl_repo/", "outputs/", "checkpoint/", "data/", "cold_data/",
           "evaluation/lmms_eval/")

#: `docs/publishing.md` discusses paths that deliberately do not exist yet.
SKIP_DOCS = {Path("docs/publishing.md")}


def _expand_braces(token: str) -> list[str]:
    """`a/{x,y}.py` -> ['a/x.py', 'a/y.py'].

    The docs use this shorthand throughout (`trained_model/{probe,audit}.py`), and it is
    worth expanding rather than skipping: each branch is a separate promise, and a table
    row that lists four files is wrong if any one of them is absent.
    """
    m = re.search(r"\{([^{}]*)\}", token)
    if not m:
        return [token]
    out = []
    for part in m.group(1).split(","):
        out += _expand_braces(token[:m.start()] + part.strip() + token[m.end():])
    return out


def _candidate_paths(text: str):
    """(path, line number) for every backtick-quoted repo path in a document.

    SECTION-RELATIVE PATHS COUNT TOO. `docs/provenance.md` is organised as `## training/`,
    `## experiments/` and so on, and its rows name files relative to that heading --
    `coldstart/submit.sh`, not `training/coldstart/submit.sh`. Those rows ARE the map, so
    a checker that only understood absolute-from-root paths would skip almost all of the
    document and pass while the map was wrong. The heading is tracked and used as a
    prefix; a token that resolves under neither the root nor the section is left alone.
    """
    section = ""
    for lineno, line in enumerate(text.splitlines(), 1):
        heading = re.match(r"##\s+([a-z_]+/)", line)
        if heading:
            section = heading.group(1)
        for token in re.findall(r"`([^`\s]+)`", line):
            token = token.rstrip(".,;:)")
            if token.startswith(FOREIGN) or "*" in token:
                continue    # a glob names a set, not a promise about one file
            if token.startswith(OWNED):
                candidates = [token]
            elif section and "/" in token and (ROOT / section).is_dir():
                candidates = [section + token]
            else:
                continue
            for candidate in candidates:
                for expanded in _expand_braces(candidate):
                    if expanded.endswith(CHECKED_SUFFIXES):
                        yield expanded, lineno


@pytest.mark.parametrize("doc", [d for d in DOCS if d not in SKIP_DOCS],
                         ids=lambda d: str(d))
def test_every_path_the_doc_names_exists(doc):
    path = ROOT / doc
    if not path.exists():
        pytest.skip(f"{doc} is not in this checkout")
    missing = [f"line {n}: {p}" for p, n in _candidate_paths(path.read_text())
               if not (ROOT / p).exists()]
    assert not missing, (
        f"{doc} names files that are not here:\n  " + "\n  ".join(missing))


def test_reproduce_names_a_runner_for_every_experiment_row():
    """The three `run.sh` rows of docs/reproduce.md are the ones that went missing."""
    text = (ROOT / "docs/reproduce.md").read_text()
    for runner in ("experiments/trained_model/run.sh",
                   "experiments/attention_bias/run.sh",
                   "experiments/head_selection/run.sh"):
        assert runner in text, f"docs/reproduce.md no longer routes through {runner}"
        assert (ROOT / runner).exists(), f"{runner} is promised and missing"


def _declared_flags(target: str) -> set[str] | None:
    """Every `--flag` in an add_argument call of a module or script path, or None."""
    path = ROOT / (target.replace(".", "/") + ".py" if "/" not in target else target)
    if not path.exists():
        return None
    import ast

    tree = ast.parse(path.read_text())
    return {
        arg.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        and arg.value.startswith("-")
    }


def test_reproduce_commands_use_flags_that_exist():
    """A flag docs/reproduce.md tells someone to pass must be one the program accepts.

    This is the `--arms` bug. `docs/reproduce.md` routed four of the paper's tables --
    2, 5, 6 and 7 -- through `evaluation/tables.py --arms paper|ablation|appendix-c`, and
    that flag has never existed: the script takes no arm selector, because the banked tree
    holds the paper's ten runs and nothing else. Anyone following the document got
    `error: unrecognized arguments` on the first command they tried.

    Shell runners are covered by `tests/test_experiment_runners.py`; this is the Python
    half, and it reads the commands out of the document rather than being told them.
    """
    text = (ROOT / "docs/reproduce.md").read_text()
    problems = []
    # `python -m a.b.c --x`, `python path/to/s.py --x`, and bare `evaluation/tables.py --x`
    pattern = re.compile(
        r"(?:python3?\s+(?:-m\s+([\w.]+)|([\w./]+\.py))|(?<![\w/])([\w./]+\.py))"
        r"((?:\s+--[a-z][\w-]*)*)")
    for lineno, line in enumerate(text.splitlines(), 1):
        for m in pattern.finditer(line):
            target = m.group(1) or m.group(2) or m.group(3)
            flags = re.findall(r"--[a-z][\w-]*", m.group(4) or "")
            if not flags:
                continue
            declared = _declared_flags(target)
            if declared is None:
                continue        # not a program in this tree
            unknown = [f for f in flags if f not in declared]
            if unknown:
                problems.append(f"line {lineno}: {target} does not accept {unknown}")
    assert not problems, (
        "docs/reproduce.md gives commands that would not run:\n  " + "\n  ".join(problems))
