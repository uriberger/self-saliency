#!/usr/bin/env bash
# Register the VGA model wrapper in an lmms-eval checkout. Idempotent, reversible.
#
#   bash install_lmms_vga.sh [--lmms-eval-dir DIR] [--copy] [--uninstall] [--check]
#
# WHY THIS EXISTS. lmms-eval lives in a clone outside this repo
# (~/scratch/research/lmms-eval), shared by every session and every running job.
# A wrapper edited into that clone by hand is a change that exists on exactly one
# machine and in no history -- and this project already has a second cluster with
# its own checkout on a filesystem this one cannot see.
#
# So the wrapper's source of truth is lmms_eval_plugin/qwen3_vl_vga.py in THIS repo,
# and this script wires a checkout up to it. It does two things, both additive:
#
#   1. Links (or copies) the wrapper to lmms_eval/models/chat/qwen3_vl_vga.py.
#      A symlink by default, so two clusters cannot drift: editing the file here
#      changes what both of them run.
#   2. Adds ONE key to AVAILABLE_CHAT_TEMPLATE_MODELS in lmms_eval/models/__init__.py.
#
# NOTHING ELSE IS TOUCHED, and that is deliberate. lmms-eval imports only the model it
# is asked for, so a job running `--model qwen3_vl` never opens the new file. Adding a
# key to a dict cannot change what another key resolves to. The registry edit is written
# to a temporary file and renamed into place, so a process reading it mid-write sees the
# old whole file or the new whole file -- never half of one. `--check` re-verifies
# afterwards that `qwen3_vl` still resolves to exactly the class it did before.
#
# Then:
#   VGA_ARGS="beta=0.2,start_layer=4,end_layer=16" \
#     bash scripts/run_lmms_eval_suite.sh --model Qwen/Qwen3-VL-8B-Instruct \
#          --model-type qwen3_vl_vga
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# A symlink into a worktree is a symlink into a directory built to be deleted:
# ./worktree.sh done removes it and every eval on every cluster then fails on a dangling
# path, long after the change that caused it. So the link is always to the central tree,
# which means the wrapper has to be merged before it can be installed from a worktree.
if [[ "$REPO" == */.worktrees/* ]]; then
    CENTRAL=$(cd "$REPO/../.." && pwd)
    echo "NOTE: running from a worktree; linking to the central tree $CENTRAL" >&2
    REPO="$CENTRAL"
fi
SRC="$REPO/lmms_eval_plugin/qwen3_vl_vga.py"

LMMS_EVAL_DIR=${LMMS_EVAL_DIR:-${SELFSAL_ROOT:-.}/../lmms-eval}
MODE=install
LINK=symlink

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lmms-eval-dir) LMMS_EVAL_DIR="$2"; shift 2 ;;
        --copy)          LINK=copy;          shift ;;
        --uninstall)     MODE=uninstall;     shift ;;
        --check)         MODE=check;         shift ;;
        -h|--help)       sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -f "$SRC" ]] || { echo "ERROR: no wrapper source at $SRC (merge the branch first?)" >&2; exit 2; }
INIT="$LMMS_EVAL_DIR/lmms_eval/models/__init__.py"
DEST="$LMMS_EVAL_DIR/lmms_eval/models/chat/qwen3_vl_vga.py"
[[ -f "$INIT" ]] || { echo "ERROR: $INIT not found -- is --lmms-eval-dir right?" >&2; exit 2; }

KEY='    "qwen3_vl_vga": "Qwen3_VL_VGA",'

# ---------------------------------------------------------------- check
if [[ "$MODE" == check ]]; then
    python - "$LMMS_EVAL_DIR" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from lmms_eval.models import get_model
stock = get_model("qwen3_vl")
print(f"  qwen3_vl     -> {stock.__module__}.{stock.__name__}")
try:
    ours = get_model("qwen3_vl_vga")
except Exception as exc:
    print(f"  qwen3_vl_vga -> NOT REGISTERED ({type(exc).__name__}: {exc})")
    sys.exit(1)
print(f"  qwen3_vl_vga -> {ours.__module__}.{ours.__name__}")
# The one regression that would matter: the stock wrapper must be untouched.
assert stock.__module__ == "lmms_eval.models.chat.qwen3_vl", stock.__module__
assert issubclass(ours, stock), "the variant must subclass the stock wrapper"
assert ours.is_simple is False, "must resolve as a chat model, like qwen3_vl"
print("  OK: qwen3_vl unchanged, and the variant subclasses it")
PY
    exit $?
fi

# ---------------------------------------------------------------- uninstall
if [[ "$MODE" == uninstall ]]; then
    rm -f "$DEST" && echo "removed $DEST"
    python - "$INIT" "$KEY" <<'PY'
import os, sys, tempfile
init, key = sys.argv[1], sys.argv[2] + "\n"
src = open(init).read()
if key not in src:
    print("  registry key was not present"); raise SystemExit(0)
d = os.path.dirname(os.path.abspath(init))
fd, tmp = tempfile.mkstemp(dir=d, prefix=".init_", suffix=".py")
with os.fdopen(fd, "w") as fh:
    fh.write(src.replace(key, ""))
os.chmod(tmp, os.stat(init).st_mode & 0o777)
os.replace(tmp, init)
print("  removed the registry key")
PY
    exit 0
fi

# ---------------------------------------------------------------- install
echo "lmms-eval : $LMMS_EVAL_DIR"
echo "wrapper   : $SRC"

if [[ "$LINK" == symlink ]]; then
    ln -sfn "$SRC" "$DEST"
    echo "  linked  $DEST -> $SRC"
else
    cp -f "$SRC" "$DEST"
    echo "  copied  $DEST   (a copy can drift from the repo; --copy was asked for)"
fi

python - "$INIT" "$KEY" <<'PY'
import os, sys, tempfile
init, key = sys.argv[1], sys.argv[2] + "\n"
src = open(init).read()
if key in src:
    print("  registry key already present"); raise SystemExit(0)
anchor = "AVAILABLE_CHAT_TEMPLATE_MODELS"
i = src.index(anchor)
mark = '    "qwen3_vl": "Qwen3_VL",\n'
j = src.index(mark, i) + len(mark)
d = os.path.dirname(os.path.abspath(init))
fd, tmp = tempfile.mkstemp(dir=d, prefix=".init_", suffix=".py")
with os.fdopen(fd, "w") as fh:
    fh.write(src[:j] + key + src[j:])
os.chmod(tmp, os.stat(init).st_mode & 0o777)
# Atomic within a filesystem: a concurrent eval reads the old whole file or the new
# whole file. This is a shared clone and other jobs import it while we write.
os.replace(tmp, init)
print("  added the registry key next to qwen3_vl")
PY

echo "verifying:"
bash "$0" --lmms-eval-dir "$LMMS_EVAL_DIR" --check
