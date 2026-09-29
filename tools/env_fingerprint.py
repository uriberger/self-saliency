#!/usr/bin/env python
"""Print a diffable fingerprint of the environment that actually runs GRPO training.

Run it on two clusters and `diff` the outputs. Anything that differs is a reason a
checkpoint copied from one to the other will not resume identically -- or at all.

Why this is not just `pip freeze`: the training code is not a pip package. It lives in a
TRL checkout (`third_party/trl_repo` by default, gitignored) that `env/patches/trl.sh`
overwrites with the tracked sources under `training/grpo/trl_patch/`. Two clusters can
report identical package versions and still run different reward functions. So this
checks three layers:

  1. packages      -- the pip layer
  2. patch set     -- every file env/patches/trl.sh copies, installed copy vs source
  3. reward modules-- every module `rewards/__init__.py` binds, and whether it is installed

Layer 3 exists because the patch script need not copy every file the entry script
imports. `grpo_vlm_qwen3.py` does `from trl.rewards import think_format_reward, ...`, and
a checkout that predates a change to those resolves the import against stock upstream
TRL, where `think_format_reward` has a different signature entirely. Any module this
prints with "NOT patch-copied" beside it is one to look at for that reason.
(`tests/test_import_layout.py` holds the copy list and those exports in step on the
source side; this is the check on the INSTALLED side, which no test can reach.)

Usage:
    <env>/bin/python -I tools/env_fingerprint.py   # -I matters: keeps cwd off sys.path

Exit status is 0 unless a check could not be performed at all.
"""

import glob
import hashlib
import importlib.metadata as md
import os
import re
import subprocess
import sys

# This file sits in tools/, so the repository root is its PARENT. The same two
# environment variables the runner and the patch script honour, with the same defaults,
# so all three agree on which checkout is being described.
REPO = (os.environ.get("SELFSAL_ROOT") or os.environ.get("SALIENCY_REPO")
        or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRL_REPO = os.environ.get("TRL_REPO") or os.path.join(REPO, "third_party", "trl_repo")

#: The script that installs the patch set. Its COPIES array is the source of truth for
#: which files are patched, and is parsed rather than duplicated here.
PATCH_SCRIPT = os.path.join(REPO, "env", "patches", "trl.sh")

#: The tracked sources it copies from.
PATCH_SRC = os.path.join(REPO, "training", "grpo", "trl_patch")

PKGS = [
    "torch", "transformers", "trl", "peft", "accelerate", "deepspeed",
    "datasets", "tokenizers", "safetensors", "numpy", "vllm",
    "qwen-vl-utils", "pillow", "wandb",
]

# Patched in place inside site-packages, so they carry no version of their own. Paths are
# relative to site-packages and hashed directly -- importing transformers just to learn
# where it lives would make the cheapest check in this file depend on the slowest step.
SITE_PATCHES = [
    "transformers/integrations/sdpa_attention.py",
    "vllm/transformers_utils/tokenizer.py",
    "vllm/model_executor/models/qwen3_vl.py",
]

NO_IMPORTS = "--no-imports" in sys.argv

WIDTH = 26


def row(label, value):
    print(f"{label:<{WIDTH}} {value}", flush=True)  # flush: partial output survives a hang


def site_packages():
    """Locate site-packages from the interpreter prefix, without importing anything."""
    for cand in glob.glob(os.path.join(sys.prefix, "lib", "python*", "site-packages")):
        if os.path.isdir(cand):
            return cand
    return None


def digest(path):
    try:
        with open(path, "rb") as fh:
            return hashlib.md5(fh.read()).hexdigest()
    except OSError:
        return None


def sh(*cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or "(none)"
    except Exception as exc:
        return f"ERROR {exc}"


def module_dir(name):
    """Import for its location only, tolerating a broken install."""
    try:
        return os.path.dirname(__import__(name).__file__)
    except Exception:
        return None


def norm(path):
    """Rewrite the two cluster-specific prefixes to placeholders.

    The repo root and the conda prefix differ between clusters by definition, so printing
    them raw makes every import-origin line a false diff hit. What actually matters is
    which of the two a module resolves under -- `$REPO/trl_repo/trl` vs `$ENV/.../trl`.
    """
    # TRL_REPO first: in a worktree it is a symlink to the central tree, so it resolves
    # somewhere outside $REPO entirely and would otherwise print as a raw absolute path.
    for prefix, name in ((TRL_REPO, "$REPO/trl_repo"), (REPO, "$REPO"), (sys.prefix, "$ENV")):
        real = os.path.realpath(prefix)
        for cand in (prefix, real):
            if path.startswith(cand):
                return name + path[len(cand):]
    return path


def copied_pairs():
    """(src, dst) for every entry in env/patches/trl.sh's COPIES array.

    Hardcoding the file list is how you miss a file the patch script started copying last
    month. Parse the script instead -- it is the source of truth, and it is the same array
    `tests/test_import_layout.py` reads. `src` is relative to training/grpo/trl_patch/ and
    `dst` to the TRL checkout, exactly as the array spells them.
    """
    try:
        with open(PATCH_SCRIPT) as fh:
            text = fh.read()
    except OSError:
        return None
    block = re.search(r"^COPIES=\((.*?)^\)", text, re.MULTILINE | re.DOTALL)
    if not block:
        return None
    return re.findall(r'"([^":]+):([^"]+)"', block.group(1))


def declared_reward_modules():
    """Module names `rewards/__init__.py` binds, from the tracked source.

    Read out of trl_patch/ rather than out of the installed checkout, and by the names
    the `__init__` imports rather than by a lazy `_import_structure` table -- this
    package's __init__ imports eagerly on purpose (a lazy failure would surface on a GPU,
    mid-step, rather than at import; its docstring is the argument).
    """
    path = os.path.join(PATCH_SRC, "rewards", "__init__.py")
    try:
        with open(path) as fh:
            src = fh.read()
    except OSError:
        return None
    return sorted({m.group(1) for m in re.finditer(r"^from \.(\w+) import", src, re.M)})


print("## host  (these lines differ between clusters by design -- ignore them in a diff)")
row("hostname", sh("hostname"))
row("$REPO", REPO)
row("$ENV", sys.prefix)

print("\n## interpreter")
row("python", sys.version.split()[0])

def dirty(path):
    """Tracked-file edits, named; untracked files only counted.

    The repo root accumulates dozens of untracked run-output directories, and listing
    them buries the one line that matters -- an edit to a tracked file that makes this
    checkout differ from the other cluster's at the same HEAD. Untracked files cannot,
    so a count is enough.
    """
    out = sh("git", "-C", path, "status", "--porcelain")
    if out in ("(none)", "") or out.startswith("ERROR"):
        return out or "(clean)"
    tracked, untracked = [], 0
    for line in out.splitlines():
        if line.startswith("??"):
            untracked += 1
        else:
            tracked.append(line.strip())
    parts = []
    if tracked:
        parts.append(", ".join(tracked))
    parts.append(f"+{untracked} untracked" if untracked else "0 untracked")
    return " | ".join(parts) if tracked else parts[0]


print("\n## git")
row("repo HEAD", sh("git", "-C", REPO, "rev-parse", "--short", "HEAD"))
row("repo dirty", dirty(REPO))
row("trl_repo HEAD", sh("git", "-C", TRL_REPO, "rev-parse", "--short", "HEAD"))
row("trl_repo dirty", dirty(TRL_REPO))

print("\n## patch set (env/patches/trl.sh: installed copy vs tracked source)")
pairs = copied_pairs()
if pairs is None:
    row("patchset", "ERROR (env/patches/trl.sh unreadable or no COPIES array)")
else:
    roll, stale, absent = hashlib.md5(), [], []
    for src, dst in sorted(pairs, key=lambda p: p[1]):
        d_dst = digest(os.path.join(TRL_REPO, dst))
        d_src = digest(os.path.join(PATCH_SRC, src))
        if d_dst is None:
            absent.append(dst)
            continue
        if d_src is not None and d_src != d_dst:
            stale.append(dst)
        roll.update(f"{dst}:{d_dst}".encode())
    row("files", len(pairs))
    row("patchset", roll.hexdigest())
    row("not installed", ", ".join(absent) if absent else "(none)")
    row("stale vs trl/", ", ".join(stale) if stale else "(none)")

print("\n## reward modules (bound by rewards/__init__.py)")
mods = declared_reward_modules()
if mods is None:
    row("rewards", "ERROR (trl_patch/rewards/__init__.py unreadable)")
else:
    copied = {dst for _, dst in (pairs or [])}
    for mod in sorted(mods):
        rel = f"trl/rewards/{mod}.py"
        d = digest(os.path.join(TRL_REPO, rel))
        tag = "" if rel in copied else "   <- NOT patch-copied"
        row(mod, (d or "MISSING") + tag)

print("\n## site-package patches")
sp = site_packages()
if sp is None:
    row("site-packages", f"NOT FOUND under {sys.prefix}")
else:
    for rel in SITE_PATCHES:
        row(rel, digest(os.path.join(sp, rel)) or "MISSING")

print("\n## packages")
for p in PKGS:
    try:
        row(p, md.version(p))
    except Exception:
        row(p, "ABSENT")

# Everything above reads files. Everything below imports them, which on a cold lustre
# cache costs minutes and can stall outright on an HF library that reaches for the
# network. Keep it last so a hang here still leaves you with every hash that decides
# whether a checkpoint resumes, and let --no-imports skip it entirely.
if NO_IMPORTS:
    print("\n## torch build / import origins   SKIPPED (--no-imports)")
    raise SystemExit(0)

print("\n## torch build")
try:
    import torch
    row("torch.version.cuda", torch.version.cuda)
    row("torch.cuda.nccl", ".".join(map(str, torch.cuda.nccl.version())))
except Exception as exc:
    row("torch", f"ERROR {type(exc).__name__}: {exc}")

print("\n## import origins   (slow: ~25s warm, minutes on a cold cache)")
for name in ("trl", "transformers", "peft", "deepspeed"):
    d = module_dir(name)
    row(name, norm(d) if d else "ERROR (import failed)")
