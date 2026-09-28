#!/usr/bin/env bash
# Create the `selfsal-sft` conda env: the cold start, via LLaMA-Factory.
#
#   bash env/setup_sft_env.sh [env-name]
#
# Separate from selfsal-grpo because it pins transformers 5.7 and torch 2.6/cu124 against
# that env's 5.x-dev and torch 2.8/cu128.
#
# This env needs NO attention patch. Supervised fine-tuning emits no attention weights,
# so the whole reason the training env is patched does not apply here.
set -euo pipefail
REPO=${SELFSAL_ROOT:-$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.." && pwd)}
ENV_NAME=${1:-selfsal-sft}

conda create -y -n "$ENV_NAME" python=3.10
grep -v '#egg=llamafactory' "$REPO/env/sft.txt" > /tmp/selfsal-sft-reqs.txt
conda run -n "$ENV_NAME" pip install -r /tmp/selfsal-sft-reqs.txt
rm -f /tmp/selfsal-sft-reqs.txt

cat <<MSG

  $ENV_NAME is ready. Still to do:

    git clone https://github.com/hiyouga/LLaMA-Factory third_party/LLaMA-Factory
    conda run -n $ENV_NAME pip install -e third_party/LLaMA-Factory

  The two cold-start corpora must also be registered in LLaMA-Factory's
  dataset_info.json; training/coldstart/configs/qwen3_vl_8b.yaml names them
  (saliency_r1_llava_cot_full, saliency_r1_mulberry_sft_full) and
  training/coldstart/data/ prepares them.
MSG
