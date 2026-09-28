#!/usr/bin/env python
"""Print a diffable fingerprint of the environment that actually runs GRPO training.

Run it on two clusters and `diff` the outputs. Anything that differs is a reason a
checkpoint copied from one to the other will not resume identically -- or at all.

Why this is not just `pip freeze`: the training code is not a pip package. It lives in
`trl_repo/`, a gitignored clone that `patch_trl_qwen3.sh` overwrites with tracked sources
from `trl/`. Two clusters can report identical package versions and still run different
reward functions. So this checks three layers:

  1. packages      -- the pip layer
  2. patch set     -- every file patch_trl_qwen3.sh copies, verified against its source
  3. reward modules-- every module trl/rewards/__init__.py promises to expose

Layer 3 exists because the patch script does NOT copy every file the entrypoint imports.
`grpo_vlm_qwen3.py` does `from trl.rewards import think_format_reward, ...`, and
format_rewards.py / saliency_rewards.py / answer_format_rewards.py are modified (or added)
in this repo without being in the patch script's copy list. A cluster whose trl_repo
predates that work resolves those imports against stock upstream TRL, where
think_format_reward has a different signature entirely.

Usage:
    ./env_fingerprint.sh                      # activates the env, then runs this
    <env>/bin/python -I env_fingerprint.py    # -I matters: keeps cwd off sys.path

Exit status is 0 unless a check could not be performed at all.
"""

import glob
import hashlib
import importlib.metadata as md
import os
import re
import subprocess
import sys

REPO = os.environ.get("SALIENCY_REPO") or os.path.dirname(os.path.abspath(__file__))
TRL_REPO = os.path.join(REPO, "trl_repo")

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
    """(src, dst) for every `cp` in patch_trl_qwen3.sh, so the list cannot drift.

    Hardcoding the file list is how you miss a file the patch script started copying
    last month. Parse the script instead -- it is the source of truth. Two shapes appear
    there: literal `cp "$REPO/a" "$TRL_REPO/b"`, and grpo_trainer_qwen3.py's
    `cp "$SRC" "$DST"` after SRC/DST are assigned -- resolve the assignments too, since
    that indirection hides the single most important file in the patch set.
    """
    script = os.path.join(REPO, "patch_trl_qwen3.sh")
    literal = re.compile(
        r'^\s*cp\s+"?\$(?:\{)?REPO(?:\})?/([^"\s]+)"?\s+"?\$(?:\{)?TRL_REPO(?:\})?/([^"\s]+)"?'
    )
    assign = re.compile(r'^\s*(SRC|DST)="\$(?:\{)?(?:REPO|TRL_REPO)(?:\})?/([^"]+)"')
    indirect = re.compile(r'^\s*cp\s+"\$(?:\{)?SRC(?:\})?"\s+"\$(?:\{)?DST(?:\})?"')
    pairs, held = [], {}
    try:
        with open(script) as fh:
            for line in fh:
                m = literal.match(line)
                if m:
                    pairs.append((m.group(1), m.group(2)))
                    continue
                m = assign.match(line)
                if m:
                    held[m.group(1)] = m.group(2)
                    continue
                if indirect.match(line) and "SRC" in held and "DST" in held:
                    pairs.append((held["SRC"], held["DST"]))
    except OSError:
        return None
    return pairs


def declared_reward_modules():
    """Module names trl/rewards/__init__.py promises via its lazy _import_structure."""
    path = os.path.join(TRL_REPO, "trl", "rewards", "__init__.py")
    try:
        with open(path) as fh:
            src = fh.read()
    except OSError:
        return None
    block = re.search(r"_import_structure\s*=\s*\{(.*?)\}", src, re.DOTALL)
    if not block:
        return None
    return re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', block.group(1))


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

print("\n## patch set (patch_trl_qwen3.sh: trl_repo copy vs tracked source)")
pairs = copied_pairs()
if pairs is None:
    row("patchset", "ERROR (patch_trl_qwen3.sh unreadable)")
else:
    roll, stale, absent = hashlib.md5(), [], []
    for src, dst in sorted(pairs, key=lambda p: p[1]):
        d_dst = digest(os.path.join(TRL_REPO, dst))
        d_src = digest(os.path.join(REPO, src))
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

print("\n## reward modules (declared in trl/rewards/__init__.py, NOT all patch-copied)")
mods = declared_reward_modules()
if mods is None:
    row("rewards", "ERROR (trl_repo/trl/rewards/__init__.py unreadable)")
else:
    copied = {dst for _, dst in (pairs or [])}
    for mod in sorted(mods):
        rel = f"trl/rewards/{mod}.py"
        d = digest(os.path.join(TRL_REPO, rel))
        tag = "" if rel in copied else "   <- not patch-copied"
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
