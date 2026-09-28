#!/usr/bin/env bash
# Print a diffable fingerprint of the env that runs GRPO training. See env_fingerprint.py
# for what is checked and why.
#
# Run it on both clusters and diff:
#     ./env_fingerprint.sh > /tmp/fp-$(hostname).txt
#     diff /tmp/fp-clusterA.txt /tmp/fp-clusterB.txt
#
# The conda base differs per cluster (miniconda3 here, miniforge3 elsewhere), so this
# searches the usual spots rather than hardcoding one. Override either half explicitly:
#     ./env_fingerprint.sh --python /path/to/envs/foo/bin/python
#     ./env_fingerprint.sh --env saliency_r1_qwen3          # different env, same search
#
# Takes ~30s warm. The last section imports torch/peft/deepspeed, which on a cold
# lustre cache costs minutes; --no-imports drops it and returns in under a second.
# Every hash that decides whether a checkpoint resumes is printed before that point.
#     ./env_fingerprint.sh --no-imports
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_NAME=${CONDA_ENV:-saliency_r1_qwen3_vllm}
PYTHON=""
PROBE_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --python) PYTHON="$2"; shift 2 ;;
        --env)    ENV_NAME="$2"; shift 2 ;;
        --no-imports) PROBE_ARGS+=("--no-imports"); shift ;;
        -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$PYTHON" ]; then
    # Candidate conda/mamba installs, most specific first. `conda info --base` is last
    # because it needs conda on PATH, which is exactly what a fresh login shell lacks.
    BASES=(
        "${CONDA_BASE:-}"
        "$HOME/scratch/miniconda3"
        "$HOME/scratch/miniforge3"
        "$HOME/scratch/research/miniconda3"
        "$HOME/scratch/research/miniforge3"
        "$HOME/miniconda3"
        "$HOME/miniforge3"
        "$(command -v conda >/dev/null 2>&1 && conda info --base 2>/dev/null || true)"
    )
    for base in "${BASES[@]}"; do
        [ -n "$base" ] || continue
        if [ -x "$base/envs/$ENV_NAME/bin/python" ]; then
            PYTHON="$base/envs/$ENV_NAME/bin/python"
            break
        fi
    done
fi

if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    echo "ERROR: could not find a python for conda env '$ENV_NAME'." >&2
    for base in "${BASES[@]}"; do
        [ -n "$base" ] && echo "       searched: $base/envs/$ENV_NAME/bin/python" >&2
    done
    echo "       Pass it directly:  $0 --python /path/to/envs/$ENV_NAME/bin/python" >&2
    exit 1
fi

# -I keeps the cwd off sys.path. Without it a stray /tmp/inspect.py (or the repo's own
# trl/ directory, when run from the repo root) shadows a stdlib or installed module and
# every import in the probe dies with a traceback that looks like a broken env.
SALIENCY_REPO="$REPO" exec "$PYTHON" -I "$REPO/env_fingerprint.py" ${PROBE_ARGS[@]+"${PROBE_ARGS[@]}"}
