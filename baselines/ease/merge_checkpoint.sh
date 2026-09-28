#!/usr/bin/env bash
#SBATCH --job-name=ease-merge
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/ease_merge_%j.log
#
# Turn an EASE run's FSDP checkpoint into a standalone HF model the bench
# harness can score, and give it a name that harness can tell apart.
#
# Two things this exists to encode:
#
# 1. IT DOES NOT FIT ON A LOGIN NODE. ease_repo/scripts/model_merger.py holds all
#    eight rank shards in memory at once, then the merged state dict, then a
#    materialised CPU model -- north of 50 GB for an 8B checkpoint. On the login
#    node that is an exit-137 SIGKILL with no message. The `cpu` partition's
#    nodes have ~2 TB, which is why this is an sbatch script and not a one-liner.
#
# 2. THE MERGED DIRECTORY IS ALWAYS CALLED `huggingface`. lmms-eval derives its
#    results directory from the model path's basename, so scoring two arms whose
#    paths both end in `huggingface` files both under the same name and the
#    response cache hands the second one the first one's answers. Every arm
#    therefore gets a distinctly named symlink under checkpoint/, matching the
#    `*_merged` convention the other baselines use.
#
# Usage:
#   sbatch merge_ease_checkpoint.sh --exp ease_8k_v2
#   sbatch merge_ease_checkpoint.sh --exp dapo_8k_v2 --step 100
#   bash   merge_ease_checkpoint.sh --exp ease_8k_v2 --step 124   # on a compute node
#
# Idempotent: a checkpoint whose huggingface/ already holds weights is skipped
# unless --force is passed.
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$REPO"

EXP=""
STEP=124
OUT_ROOT="$REPO/outputs/ease"
LINK_DIR="$REPO/checkpoint"
FORCE=0
PYBIN=${PYBIN:-/home/uberger/scratch/miniconda3/envs/ease/bin/python}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp)      EXP="$2";      shift 2 ;;
        --step)     STEP="$2";     shift 2 ;;
        --out-root) OUT_ROOT="$2"; shift 2 ;;
        --link-dir) LINK_DIR="$2"; shift 2 ;;
        --force)    FORCE=1;       shift ;;
        -h|--help)  sed -n '2,38p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$EXP" ]] || { echo "ERROR: --exp is required." >&2; exit 2; }

ACTOR="$OUT_ROOT/$EXP/checkpoints/global_step_$STEP/actor"
# Resolve physically before the symlink below is written. outputs/ and checkpoint/
# are symlinks into the shared tree, so under a worktree $REPO is
# .worktrees/<branch>/... -- and `worktree.sh done` would then break a merged model
# that outlives the branch that produced it. Same reason stage_ease_checkpoint.sh
# uses `pwd -P`.
[[ -d "$ACTOR" ]] && ACTOR="$(cd "$ACTOR" && pwd -P)"
HF="$ACTOR/huggingface"
[[ -d "$ACTOR" ]] || { echo "ERROR: $ACTOR does not exist." >&2; exit 1; }

mkdir -p logs "$LINK_DIR"

echo "=========================================================================="
echo "exp   : $EXP   step $STEP"
echo "actor : $ACTOR"
echo "=========================================================================="

if compgen -G "$HF/*.safetensors" > /dev/null && [[ $FORCE -eq 0 ]]; then
    echo "already merged ($(ls "$HF"/*.safetensors | wc -l) shards); pass --force to redo."
else
    export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
    # Not `cd ease_repo && python scripts/...` with the output piped through a
    # filter: the merger's only failure signal is a traceback or a SIGKILL, and a
    # pipeline's exit status is the last command's. Keep it unpiped.
    ( cd "$REPO/ease_repo" && "$PYBIN" scripts/model_merger.py --local_dir "$ACTOR" )
    compgen -G "$HF/*.safetensors" > /dev/null || {
        echo "ERROR: the merger exited without writing weights to $HF" >&2; exit 1; }
fi

LINK="$LINK_DIR/${EXP}-step${STEP}_merged"
ln -sfn "$HF" "$LINK"

echo
echo "merged : $HF"
echo "linked : $LINK"
du -sh "$HF"
echo
echo "Register it in run_bench_baselines.sh's BASELINES as:"
echo "    \"${EXP//_/-}|checkpoint/$(basename "$LINK")|BENCH_MODEL_TYPE=qwen3_vl\""
echo
echo "The BENCH_MODEL_TYPE is required: launch_lmms_eval_job.sh infers the"
echo "lmms-eval model class from a qwen3-vl substring in the model PATH, and"
echo "these names have none. Without it every eval unit fails instantly and the"
echo "suite banks nothing."
