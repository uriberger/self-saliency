#!/usr/bin/env bash
# Create the `selfsal-grpo` conda env: GRPO training, every attention probe, Section 5.
#
#   bash env/setup_grpo_env.sh [env-name]
#
# Python 3.10. Builds fine on a machine with no GPU -- flash-attn is deliberately absent
# (it is a source build needing nvcc, and training runs with --attn_implementation sdpa
# while vLLM ships its own kernels), so this can be prepared on a login node.
#
# The pinned list carries an `-e git+...#egg=trl` line for reference and it is STRIPPED
# below: TRL comes from the local checkout you install separately, not from git. Letting
# pip resolve it would install an unpatched TRL over the patched one, and the failure is
# silent -- the trainer class simply would not exist.
set -euo pipefail
REPO=${SELFSAL_ROOT:-$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.." && pwd)}
ENV_NAME=${1:-selfsal-grpo}

conda create -y -n "$ENV_NAME" python=3.10
grep -v '#egg=trl' "$REPO/env/grpo.txt" > /tmp/selfsal-grpo-reqs.txt
conda run -n "$ENV_NAME" pip install -r /tmp/selfsal-grpo-reqs.txt
conda run -n "$ENV_NAME" pip install -e "$REPO"
rm -f /tmp/selfsal-grpo-reqs.txt

cat <<MSG

  $ENV_NAME is ready. Still to do, in this order:

    git clone --branch v0.21-release https://github.com/huggingface/trl third_party/trl_repo
    conda run -n $ENV_NAME pip install -e third_party/trl_repo --no-deps
    bash env/patches/trl.sh
    bash env/patches/transformers.sh
    bash env/patches/vllm.sh

  Then: conda run -n $ENV_NAME python -c 'from trl import GRPOTrainerQwen3'
MSG
