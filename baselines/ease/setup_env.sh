#!/usr/bin/env bash
#SBATCH --job-name=ease-env
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/ease_env_%j.log
#
# Build the conda env that runs EASE (the EasyR1/verl fork at ease_repo/).
#
# This is a NEW env, `ease`. It does not touch saliency_r1_qwen3 or
# saliency_r1_qwen3_vllm, which are patched for Qwen3-VL against a transformers
# 5.x dev build and are shared by every session and running job.
#
# Why a separate env is mandatory rather than merely tidy:
#
#   EASE's requirements.txt pins transformers>=4.54.0,<=4.57.0. Qwen3-VL landed
#   in transformers v4.57.0 and does NOT exist in v4.56.0 (checked against the
#   upstream tags), so that range collapses to exactly 4.57.0. Our existing envs
#   run transformers 5.13.0.dev0. The two cannot coexist.
#
# Everything else we already have and simply re-pin: their Dockerfile targets
# torch 2.8.0+cu128 and vllm 0.11.0, which is what saliency_r1_qwen3_vllm
# already runs.
#
# Python 3.12 to match their NGC base image (nvcr.io/nvidia/pytorch:25.05-py3).
#
# Usage:
#   sbatch setup_ease_env.sh
#   bash   setup_ease_env.sh --name ease-test     # build under another name
#
# Their Dockerfile points pip and apt at Tsinghua mirrors; we deliberately do
# not, and we install the flash-attn release wheel rather than compiling it.

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

CONDA_ROOT=/home/uberger/scratch/miniconda3
ENV_NAME=ease
PYVER=3.12
TORCH=2.8.0
VLLM=0.11.0
TRANSFORMERS=4.57.0
FLASH_ATTN=2.8.3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) ENV_NAME="$2"; shift 2 ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$REPO/logs"
ENV_PREFIX="$CONDA_ROOT/envs/$ENV_NAME"

say() { echo "[$(date '+%F %T')] $*"; }

say "building conda env '$ENV_NAME' (python $PYVER) at $ENV_PREFIX"

if [[ -d "$ENV_PREFIX" ]]; then
    say "env already exists -- reusing it (delete it to rebuild from scratch)"
else
    "$CONDA_ROOT/bin/conda" create -y -n "$ENV_NAME" "python=$PYVER"
fi

PY="$ENV_PREFIX/bin/python"
PIP="$ENV_PREFIX/bin/pip"

say "python: $($PY -V)"

# One resolver pass for the torch/vllm pair, exactly as their Dockerfile does --
# splitting these across pip calls lets vllm drag in a different torch.
say "installing torch $TORCH + vllm $VLLM and the verl runtime deps"
$PIP install --no-cache-dir \
    "vllm==$VLLM" "torch==$TORCH" "torchvision==0.23.0" "torchaudio==$TORCH" \
    tensordict torchdata \
    accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" "optree>=0.13.0" pandas \
    "ray[default]" codetiming pylatexenc qwen-vl-utils wandb liger-kernel mathruler \
    pillow omegaconf ruff pytest

# Pin transformers LAST: vllm's own resolution will happily install something
# outside EASE's supported range, and Qwen3-VL only exists at exactly 4.57.0.
say "pinning transformers==$TRANSFORMERS"
$PIP install --no-cache-dir --no-deps "transformers==$TRANSFORMERS"

# ...and then walk tokenizers back. vllm 0.11.0 resolves tokenizers to 0.23.2,
# but transformers 4.57.0 hard-requires >=0.22.0,<=0.23.0 and raises ImportError
# at import time otherwise. --no-deps above means the transformers pin does not
# correct it on its own.
say "pinning tokenizers into the range transformers $TRANSFORMERS accepts"
$PIP install --no-cache-dir "tokenizers>=0.22.0,<=0.23.0"

say "installing flash-attn $FLASH_ATTN from the release wheel (no compile)"
ABI_FLAG=$($PY -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
CPTAG=$($PY -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
WHEEL="flash_attn-${FLASH_ATTN}+cu12torch2.8cxx11abi${ABI_FLAG}-${CPTAG}-${CPTAG}-linux_x86_64.whl"
URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN}/${WHEEL}"
say "  wheel: $WHEEL"
TMPWHEEL="$(mktemp -d)/$WHEEL"
if curl -fsSL -o "$TMPWHEEL" "$URL"; then
    $PIP install --no-cache-dir "$TMPWHEEL"
else
    say "WARNING: could not fetch $URL"
    say "         flash-attn is unset; EASE will fall back to sdpa. Investigate before training."
fi

say "installing ease_repo in editable mode"
$PIP install --no-cache-dir --no-deps -e "$REPO/ease_repo"

say "--- verification ---"
$PY - <<'PY'
import importlib.metadata as m
def v(p):
    try: return m.version(p)
    except Exception: return "MISSING"
for p in ["torch","transformers","vllm","flash-attn","ray","tensordict","accelerate","datasets","liger-kernel"]:
    print(f"  {p:16s} {v(p)}")
import transformers
from transformers.models.qwen3_vl import modeling_qwen3_vl  # noqa: F401
print("  qwen3_vl modeling import: OK")
import torch
print(f"  torch.cuda available: {torch.cuda.is_available()} (False is expected on a CPU node)")
PY

say "done. activate with:  conda activate $ENV_NAME"
