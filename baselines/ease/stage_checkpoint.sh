#!/usr/bin/env bash
# Stage one of our merged checkpoints so transformers 4.57.0 can load it.
#
# Our checkpoints were written by transformers 5.13.0.dev0. The `ease` env is
# pinned to exactly 4.57.0 (see setup_ease_env.sh for why that is a point and
# not a range), and three metadata files changed schema between the two:
#
#   config.json            text_config.rope_parameters{rope_theta,...}  became
#                          text_config.rope_scaling + text_config.rope_theta,
#                          and vision_config.model_type went qwen3_vl_vision ->
#                          qwen3_vl. Without the rename 4.57 dies in
#                          Qwen3VLTextRotaryEmbedding: 'NoneType' has no 'get'.
#   tokenizer_config.json  extra_special_tokens became a list; 4.57 calls
#                          .keys() on it -> AttributeError.
#   processor_config.json  5.x nests image_processor/video_processor inside it;
#                          4.57 wants separate preprocessor_config.json and
#                          video_preprocessor_config.json.
#
# All three are metadata. The weights are unaffected: our merged checkpoint has
# the same 750 tensors, with the same names and shapes, that 4.57 builds from
# the stock config. So we take the stock repo's 4.x metadata, symlink our
# weights next to it, and verify the substitution rather than assume it.
#
# The verification is the point of this script. It refuses to stage unless:
#   * the stock chat template is byte-identical to ours (a differing template
#     would silently change the prompt the policy was SFT'd on),
#   * the stock tokenizer's vocab, merges and added tokens match ours,
#   * every tensor 4.57 expects from the stock config is present in our
#     checkpoint with the right shape, and there are no extras.
#
# Weights are symlinked, never copied -- the merged checkpoint is 17 GB.
#
# Usage:
#   bash stage_ease_checkpoint.sh
#   bash stage_ease_checkpoint.sh --src checkpoint/other_merged --dest checkpoint/other__tf457
#   bash stage_ease_checkpoint.sh --stock Qwen/Qwen3-VL-4B-Instruct
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$REPO"

SRC="checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged"
DEST=""
STOCK="Qwen/Qwen3-VL-8B-Instruct"
FORCE=0
PYBIN=${PYBIN:-/home/uberger/scratch/miniconda3/envs/ease/bin/python}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src)   SRC="$2";   shift 2 ;;
        --dest)  DEST="$2";  shift 2 ;;
        --stock) STOCK="$2"; shift 2 ;;
        --force) FORCE=1;    shift ;;
        -h|--help) sed -n '2,36p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ -d "$SRC" ]] || { echo "ERROR: --src $SRC does not exist." >&2; exit 1; }
# `pwd -P`, not `pwd`: checkpoint/ is a symlink into the shared tree, and a
# logical path would make the weight symlinks below point through
# .worktrees/<branch>/ -- which `worktree.sh done` deletes, silently breaking a
# checkpoint that outlives the branch that staged it.
SRC="$(cd "$SRC" && pwd -P)"
DEST=${DEST:-"$REPO/checkpoint/$(basename "$SRC")__tf457"}
[[ "$DEST" = /* ]] || DEST="$REPO/$DEST"
DEST="$(cd "$(dirname "$DEST")" && pwd -P)/$(basename "$DEST")"

export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}

echo "=========================================================================="
echo "src   : $SRC"
echo "stock : $STOCK   (4.x metadata donor)"
echo "dest  : $DEST"
echo "=========================================================================="

if [[ -e "$DEST" && $FORCE -eq 0 ]]; then
    echo "$DEST already exists; pass --force to restage." >&2
    exit 0
fi
rm -rf "$DEST"
mkdir -p "$DEST"

"$PYBIN" - "$SRC" "$DEST" "$STOCK" <<'PYEOF'
import glob
import json
import os
import shutil
import sys

import torch
import transformers

# Populate the lazy qwen3_vl module before the auto-mapping is consulted: in
# 4.57.0 AutoModelForImageTextToText cannot resolve Qwen3VLForConditionalGeneration
# out of _LazyModule until the modeling submodule has actually been imported.
from transformers.models.qwen3_vl import modeling_qwen3_vl  # noqa: F401
from safetensors import safe_open

src, dest, stock = sys.argv[1], sys.argv[2], sys.argv[3]

# ── locate the stock metadata (cache first; these files are a few MB) ────────
if os.path.isdir(stock):
    stock_dir = stock
else:
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.txt"]
    try:
        stock_dir = snapshot_download(stock, allow_patterns=patterns, local_files_only=True)
    except Exception:
        stock_dir = snapshot_download(stock, allow_patterns=patterns)
print(f"stock metadata: {stock_dir}")

# ── check 1: the chat template must be the one the policy was SFT'd against ──
def read_template(directory):
    jinja = os.path.join(directory, "chat_template.jinja")
    if os.path.isfile(jinja):
        return open(jinja, encoding="utf-8").read()
    js = os.path.join(directory, "chat_template.json")
    if os.path.isfile(js):
        loaded = json.load(open(js, encoding="utf-8"))
        return loaded["chat_template"] if isinstance(loaded, dict) else loaded
    tok = os.path.join(directory, "tokenizer_config.json")
    if os.path.isfile(tok):
        return json.load(open(tok, encoding="utf-8")).get("chat_template")
    return None

ours_template, stock_template = read_template(src), read_template(stock_dir)
if ours_template is None or stock_template is None:
    sys.exit("FAIL: could not read a chat template from both checkpoints")
if ours_template.strip() != stock_template.strip():
    sys.exit(
        "FAIL: the stock chat template differs from this checkpoint's. Substituting it "
        "would change the prompt format the policy was trained on. Refusing to stage."
    )
print("  chat template      identical")

# ── check 2: the tokenizer must be the same tokenizer ────────────────────────
def normalise_merges(model):
    return [" ".join(m) if isinstance(m, list) else m for m in model.get("merges", [])]

ours_tok = json.load(open(os.path.join(src, "tokenizer.json"), encoding="utf-8"))
stock_tok = json.load(open(os.path.join(stock_dir, "tokenizer.json"), encoding="utf-8"))
if ours_tok["model"]["vocab"] != stock_tok["model"]["vocab"]:
    sys.exit("FAIL: tokenizer vocab differs between this checkpoint and the stock repo")
if normalise_merges(ours_tok["model"]) != normalise_merges(stock_tok["model"]):
    sys.exit("FAIL: tokenizer merges differ between this checkpoint and the stock repo")
ours_added = [t["content"] for t in ours_tok.get("added_tokens", [])]
stock_added = [t["content"] for t in stock_tok.get("added_tokens", [])]
if ours_added != stock_added:
    sys.exit("FAIL: added tokens differ between this checkpoint and the stock repo")
print(f"  tokenizer          identical ({len(ours_tok['model']['vocab'])} vocab, {len(ours_added)} added)")

# ── copy the 4.x metadata, symlink the weights ───────────────────────────────
META = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "chat_template.json",
]
copied = []
for name in META:
    path = os.path.join(stock_dir, name)
    if os.path.isfile(path):
        shutil.copy2(path, os.path.join(dest, name))
        copied.append(name)
print(f"  copied from stock  {', '.join(copied)}")

weights = sorted(
    glob.glob(os.path.join(src, "*.safetensors"))
    + glob.glob(os.path.join(src, "*.safetensors.index.json"))
    + glob.glob(os.path.join(src, "*.bin"))
)
if not weights:
    sys.exit(f"FAIL: no weight files under {src}")
for path in weights:
    link = os.path.join(dest, os.path.basename(path))
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(path, link)
print(f"  symlinked weights  {', '.join(os.path.basename(p) for p in weights)}")

# A stock index.json would point at shard names our checkpoint does not have.
stray = os.path.join(dest, "model.safetensors.index.json")
if os.path.isfile(stray) and not os.path.islink(stray):
    ours_index = os.path.join(src, "model.safetensors.index.json")
    if not os.path.isfile(ours_index):
        os.remove(stray)
        print("  removed the stock model.safetensors.index.json (our weights are unsharded)")

# ── check 3: every tensor 4.57 wants, at the right shape ─────────────────────
config = transformers.AutoConfig.from_pretrained(dest)
with torch.device("meta"):
    model = transformers.AutoModelForImageTextToText.from_config(config)
expected = model.state_dict()

have = {}
for path in glob.glob(os.path.join(dest, "*.safetensors")):
    with safe_open(path, framework="pt") as handle:
        for key in handle.keys():
            have[key] = tuple(handle.get_slice(key).get_shape())

missing = sorted(set(expected) - set(have))
extra = sorted(set(have) - set(expected))
mismatched = sorted(k for k in set(expected) & set(have) if tuple(expected[k].shape) != have[k])
if missing or extra or mismatched:
    print(f"  missing    : {missing[:8]}")
    print(f"  extra      : {extra[:8]}")
    print(f"  mismatched : {mismatched[:8]}")
    sys.exit("FAIL: the staged config does not describe these weights")
print(f"  weights            {len(expected)}/{len(expected)} tensors match by name and shape")

# ── check 4: the things that failed before must now load ─────────────────────
transformers.AutoTokenizer.from_pretrained(dest, use_fast=True)
processor = transformers.AutoProcessor.from_pretrained(dest, use_fast=True)
print(f"  processor          {type(processor).__name__} loads under {transformers.__version__}")
PYEOF

echo "=========================================================================="
echo "staged: $DEST"
echo "Point MODEL_PATH at it:"
echo "  bash launch_ease_train.sh --model $DEST ..."
echo "=========================================================================="
