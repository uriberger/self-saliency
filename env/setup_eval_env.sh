#!/bin/bash
# Build the `lmms_eval` conda env on this cluster (linux-aarch64, GB200 / sm_100).
#
# Why this exists instead of `conda env create -f lmms_eval.environment.yml`:
# that YAML is a linux-64 snapshot from the old cluster and cannot be solved
# here -- it pins `ld_impl_linux-64`, an x86_64-only package. Two further
# deviations are forced by the hardware, not by preference:
#
#   1. torch 2.6.0+cu124 / torchvision 0.21.0+cu124 are unusable. GB200 is
#      compute capability 10.0 (sm_100); CUDA 12.4 tops out at sm_90, so those
#      wheels cannot launch a kernel here even if an aarch64 build existed.
#      We install the cu128 pair instead (2.7.1 / 0.22.1) -- the same build
#      already proven on this cluster by the `saliency_r1_qwen3` env.
#   2. All nvidia-*/cuda-* pins and `triton` are dropped from the freeze; they
#      are torch's own transitive deps and must match the cu128 wheel.
#
# Everything else -- transformers 5.13.1, accelerate 1.14.0, datasets 5.0.0,
# sqlitedict 2.1.0, python 3.12 -- is held at the snapshotted version.
#
# Usage:  bash scripts/slurm/setup_lmms_eval_env.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONDA_SH=${CONDA_SH:-${CONDA_ROOT:?set CONDA_ROOT}/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-lmms_eval}
PYTHON_VERSION=${PYTHON_VERSION:-3.12.13}
LMMS_EVAL_DIR=${LMMS_EVAL_DIR:-${SELFSAL_ROOT:-.}/../lmms-eval}
TORCH_VERSION=${TORCH_VERSION:-2.7.1}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.22.1}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}

if [[ ! -d "$LMMS_EVAL_DIR" ]]; then
    echo "lmms-eval clone not found at $LMMS_EVAL_DIR" >&2
    exit 1
fi

# This script exists only because the linux-64 snapshot cannot be replayed on
# aarch64. On x86_64 the original snapshot is the more faithful reproduction --
# use it rather than this, or the pinned versions silently drift.
if [[ "$(uname -m)" != "aarch64" && "${ALLOW_NON_AARCH64:-0}" != "1" ]]; then
    cat >&2 <<EOF
This is the aarch64/GB200 env builder, but this host is $(uname -m).
On linux-64 use the original snapshot instead:
    conda env create -n $ENV_NAME -f $SCRIPT_DIR/lmms_eval.environment.yml
    pip install -r $SCRIPT_DIR/lmms_eval.pip-freeze.txt
    pip install -e $LMMS_EVAL_DIR
Set ALLOW_NON_AARCH64=1 to override.
EOF
    exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"

# conda-forge with --override-channels on purpose: the `defaults` channels
# (repo.anaconda.com) refuse to solve until their commercial Terms of Service
# are accepted, which is not a call this script should make. conda-forge
# carries the exact snapshotted python for linux-aarch64 anyway.
echo "==> creating conda env '$ENV_NAME' (python $PYTHON_VERSION, conda-forge)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "    env already exists; reusing it (delete it to start clean)"
else
    conda create -y -n "$ENV_NAME" --override-channels -c conda-forge "python=$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"

echo "==> torch $TORCH_VERSION + torchvision $TORCHVISION_VERSION from $TORCH_INDEX"
pip install --no-cache-dir "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    --index-url "$TORCH_INDEX"

# --no-deps is deliberate. A pip freeze is already the full transitive closure of
# the source env, so nothing needs resolving -- and resolving it actively fails:
# the snapshot pins antlr4-python3-runtime==4.13.2 while latex2sympy2 1.9.1
# declares ==4.7.2. That inconsistency was present in the old cluster's env too
# (pip freeze records what is installed, not what is consistent). Installing the
# closure verbatim reproduces that env exactly instead of letting the resolver
# pick a different, untested set.
echo "==> pinned dependency set (aarch64 freeze, --no-deps)"
pip install --no-cache-dir --no-deps -r "$SCRIPT_DIR/lmms_eval.pip-freeze.aarch64.txt"

# Our patched fork, editable -- NOT upstream EvolvingLMMs-Lab/lmms-eval. The
# cache-correctness and task fixes live here; upstream silently corrupts runs.
echo "==> our lmms-eval fork (editable) from $LMMS_EVAL_DIR"
pip install --no-cache-dir --no-deps -e "$LMMS_EVAL_DIR"

echo "==> verifying"
python - <<'PY'
import lmms_eval, torch, transformers, accelerate, datasets, sqlitedict
print("lmms_eval  :", lmms_eval.__file__)
print("torch      :", torch.__version__, "cuda", torch.version.cuda)
print("transformers:", transformers.__version__)
print("accelerate :", accelerate.__version__)
print("datasets   :", datasets.__version__)
assert "site-packages" not in lmms_eval.__file__, \
    "lmms_eval resolved to site-packages -- the editable install of our fork did not take"
print("OK: lmms_eval resolves to the local fork")
PY
