#!/usr/bin/env bash
# Run one arm of the EASE replication on saliency-r1-8k, inside an existing
# allocation. launch_ease_train_job.sh is what asks SLURM for the allocation.
#
#   bash launch_ease_train.sh --arm ease --exp ease_8k
#   bash launch_ease_train.sh --arm dapo --exp dapo_8k
#
# TWO ARMS, AND BOTH ARE REQUIRED. EASE is DAPO-in-EasyR1 plus an auxiliary
# attention loss; our overlap runs are GRPO-in-TRL. Scoring an EASE checkpoint
# against an overlap checkpoint compares method AND framework at once. The
# quantity that means something is
#
#     (EASE - DAPO) inside EasyR1   vs   (overlap - placebo) inside ours
#
# which is why --arm dapo runs their unmodified baseline on the same data from
# the same checkpoint. Run it.
#
# WHAT DIFFERS FROM THEIR PUBLISHED RECIPE, and nothing else does:
#
#   data              saliency-r1-8k, not their (unreleased) evidence pools.
#                     8,080 rows, all single-evidence: our `bbox` is already a
#                     union of the source corpus's boxes, so the 1:1
#                     single/multi mixture the paper samples has no counterpart
#                     and K=1 for every row.
#   base model        our SFT cold start, staged for transformers 4.57 by
#                     stage_ease_checkpoint.sh.
#   rollout batch     128, not 512. At 512 our 8,080 rows give 31 steps over
#                     two epochs against the <=~158 their own run had. This
#                     does NOT change the optimizer batch: EasyR1 multiplies
#                     worker.actor.global_batch_size by rollout.n internally,
#                     so at global_batch_size=64 the update still sees 64
#                     prompts x 5 rollouts either way, and the run still takes
#                     252 optimizer steps. What 128 buys is step granularity --
#                     126 reported steps to checkpoint and log against.
#   reward            ease/reward_function/judged_perception.py: their rule
#                     matcher with our gpt-4o-mini judge behind it. See that
#                     file for why. Pass --no-judge for their reward exactly --
#                     but pass it to BOTH arms or NEITHER. A judged EASE arm
#                     against a rule-scored DAPO arm confounds the attention
#                     loss with the reward, which is the one thing the paired
#                     design exists to prevent.
#
# Every other hyperparameter comes from examples/config.yaml and their
# train_ease_dapo_qwen3vl.sh: lr 1e-6, 2 epochs, n=5, clip 0.2/0.3, KL off,
# lambda_attn 0.001, background alpha 0.1, sigma scale 0.25, layer floor(2L/3),
# tau 0.5, <=64 response tokens for the aux loss, padding_free false.
#
# CHECKPOINTS ARE FULL, not model-only, because this run will be interrupted.
# 124 steps is ~6 h against 4 h allocations, and save_model_only=true writes no
# optimizer state, no LR scheduler and no RNG (fsdp_checkpoint_manager.py:100) --
# every resume would restart AdamW's moments from zero, at whatever step each arm
# happened to be cut at. ~48 GB a checkpoint against ~16; take the disk.
# --model-only-checkpoints for the other trade.
#
# The judge needs a key: NVIDIA_API_KEY=... bash launch_ease_train.sh ...
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$REPO"

ARM=ease
EXP=""
PROJ=ease_saliency_r1_8k
MODEL="$REPO/checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged__tf457"
DATA="$REPO/cold_data/ease/saliency_r1_8k"
OUT_ROOT="$REPO/outputs/ease"
GPUS=${SLURM_GPUS_ON_NODE:-8}
TP=4
ROLLOUT_BATCH=128
GLOBAL_BATCH=64
EPOCHS=2
LAMBDA_ATTN=0.001
JUDGE=1
VAL_FREQ=10
SAVE_FREQ=25
SAVE_LIMIT=-1
SAVE_MODEL_ONLY=false
DRY_RUN=0
PREFLIGHT=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm)              ARM="$2";           shift 2 ;;
        --exp)              EXP="$2";           shift 2 ;;
        --project)          PROJ="$2";          shift 2 ;;
        --model)            MODEL="$2";         shift 2 ;;
        --data)             DATA="$2";          shift 2 ;;
        --out-root)         OUT_ROOT="$2";      shift 2 ;;
        --gpus)             GPUS="$2";          shift 2 ;;
        --tp)               TP="$2";            shift 2 ;;
        --rollout-batch)    ROLLOUT_BATCH="$2"; shift 2 ;;
        --global-batch)     GLOBAL_BATCH="$2";  shift 2 ;;
        --epochs)           EPOCHS="$2";        shift 2 ;;
        --lambda-attn)      LAMBDA_ATTN="$2";   shift 2 ;;
        --val-freq)         VAL_FREQ="$2";      shift 2 ;;
        --save-freq)        SAVE_FREQ="$2";     shift 2 ;;
        --save-limit)       SAVE_LIMIT="$2";    shift 2 ;;
        --model-only-checkpoints) SAVE_MODEL_ONLY=true; shift ;;
        --no-judge)         JUDGE=0;            shift ;;
        --dry-run)          DRY_RUN=1;          shift ;;
        --preflight)        PREFLIGHT=1;        shift ;;
        --) shift; EXTRA+=("$@"); break ;;
        -h|--help) sed -n '2,45p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

case "$ARM" in
    ease|dapo) ;;
    *) echo "ERROR: --arm must be ease or dapo, got '$ARM'." >&2; exit 2 ;;
esac
EXP=${EXP:-"${ARM}_8k"}

EASE_REPO="$REPO/ease_repo"
TRAIN_FILE="$DATA/parquet/train.parquet"
VAL_FILE="$DATA/parquet/val.parquet"
IMAGE_DIR="$DATA/images"
REWARD_FILE="$REPO/ease/reward_function/judged_perception.py"

for path in "$EASE_REPO" "$MODEL" "$IMAGE_DIR"; do
    [[ -e "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }
done
for path in "$TRAIN_FILE" "$VAL_FILE"; do
    [[ -f "$path" ]] || {
        echo "ERROR: missing $path -- run prepare_ease_saliency_data.sh first." >&2; exit 1; }
done

# ── environment ─────────────────────────────────────────────────────────────
CONDA_ROOT=${CONDA_ROOT:-/home/uberger/scratch/miniconda3}
CONDA_ENV=${CONDA_ENV:-ease}
# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"
set +u; conda activate "$CONDA_ENV"; set -u

export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# judged_perception.py imports their perception.py through this.
export EASE_REPO
export JUDGE_MODEL=${JUDGE_MODEL:-azure/openai/gpt-4o-mini}
# 64, not 32. At rollout batch 128 a step judges ~480 completions (measured:
# judge_called runs 0.73-0.90), and the reward actor fans out inside one Ray
# actor. 32 workers is ~15 sequential rounds; 64 halves that, keeping the judge
# inside the advantage step rather than adding to the step time.
export JUDGE_MAX_WORKERS=${JUDGE_MAX_WORKERS:-64}
[[ -n "${NVIDIA_API_KEY:-}" ]] && export NVIDIA_API_KEY
[[ -n "${OPENAI_API_KEY:-}"  ]] && export OPENAI_API_KEY
[[ -n "${OPENAI_BASE_URL:-}" ]] && export OPENAI_BASE_URL

if [[ $JUDGE -eq 1 ]]; then
    REWARD_FUNCTION="$REWARD_FILE:compute_score"
    if [[ -z "${NVIDIA_API_KEY:-}${OPENAI_API_KEY:-}" ]]; then
        echo "WARNING: no NVIDIA_API_KEY/OPENAI_API_KEY. Every judged row falls back to" >&2
        echo "         the rule score, which on flickr30k means 0. Pass --no-judge if" >&2
        echo "         that is what you want." >&2
    fi
else
    # Their reward, byte for byte -- not our file with the judge switched off,
    # so that --no-judge is a real control rather than a near-copy of one.
    REWARD_FUNCTION="$EASE_REPO/examples/reward_function/perception.py:compute_score"
fi

SAVE_PATH="$OUT_ROOT/$EXP/checkpoints"
LOG_DIR="$OUT_ROOT/$EXP"
mkdir -p "$SAVE_PATH" "$LOG_DIR"

# ── arm-specific overrides ──────────────────────────────────────────────────
ARM_ARGS=(
    worker.actor.padding_free=false
)
if [[ "$ARM" == "ease" ]]; then
    ARM_ARGS+=(
        worker.actor.attention.use_evidence_anchor=true
        worker.actor.attention.lambda_attn="$LAMBDA_ATTN"
        worker.actor.attention.bbox_weight_mode=gaussian
        worker.actor.attention.gaussian_sigma_scale=0.25
        worker.actor.attention.background_alpha=0.1
        worker.actor.attention.attn_loss_mode=vision_kl
        worker.actor.attention.kl_direction=model_to_target
        worker.actor.attention.layer_index=-1
        worker.actor.attention.reward_threshold=0.5
        worker.actor.attention.max_attn_response_tokens=64
    )
else
    ARM_ARGS+=(
        worker.actor.attention.use_evidence_anchor=false
        worker.actor.attention.lambda_attn=0.0
    )
fi

ROWS=$("$CONDA_ROOT/envs/$CONDA_ENV/bin/python" -c \
    "import pandas;print(len(pandas.read_parquet('$TRAIN_FILE')))" 2>/dev/null || echo "?")
STEPS="?"
[[ "$ROWS" != "?" ]] && STEPS=$(( ROWS * EPOCHS / ROLLOUT_BATCH ))

echo "=========================================================================="
echo "arm       : $ARM     experiment: $EXP"
echo "model     : $MODEL"
echo "train     : $TRAIN_FILE   ($ROWS rows)"
echo "reward    : $REWARD_FUNCTION"
echo "judge     : $([[ $JUDGE -eq 1 ]] && echo "$JUDGE_MODEL, $JUDGE_MAX_WORKERS workers, key $([[ -n "${NVIDIA_API_KEY:-}${OPENAI_API_KEY:-}" ]] && echo set || echo MISSING)" || echo 'off (their rule reward)')"
echo "batching  : rollout $ROLLOUT_BATCH x n5, global $GLOBAL_BATCH prompts, $EPOCHS epochs -> ~$STEPS steps"
echo "gpus      : $GPUS   (rollout tp $TP)"
echo "save      : $SAVE_PATH  every $SAVE_FREQ steps, model_only=$SAVE_MODEL_ONLY, limit $SAVE_LIMIT"
echo "=========================================================================="

CMD=(
    python3 -m verl.trainer.main
    config=examples/config.yaml
    data.train_files="$TRAIN_FILE"
    data.val_files="$VAL_FILE"
    data.image_dir="$IMAGE_DIR"
    data.rollout_batch_size="$ROLLOUT_BATCH"
    data.format_prompt=./examples/format_prompt/perception.jinja
    worker.actor.model.model_path="$MODEL"
    worker.actor.global_batch_size="$GLOBAL_BATCH"
    worker.actor.clip_ratio_low=0.2
    worker.actor.clip_ratio_high=0.3
    worker.rollout.temperature=1.0
    worker.rollout.val_override_config.temperature=0.0
    worker.rollout.val_override_config.top_p=1.0
    worker.rollout.val_override_config.n=1
    worker.rollout.tensor_parallel_size="$TP"
    worker.reward.reward_function="$REWARD_FUNCTION"
    # The reward manager is one Ray actor and the judge fans out inside it;
    # their default of 1 CPU would serialise 640 HTTP calls per step.
    worker.reward.num_cpus=8
    algorithm.adv_estimator=grpo
    algorithm.disable_kl=true
    algorithm.use_kl_loss=false
    algorithm.kl_coef=0.0
    algorithm.online_filtering=false
    trainer.save_checkpoint_path="$SAVE_PATH"
    trainer.project_name="$PROJ"
    trainer.experiment_name="$EXP"
    trainer.total_epochs="$EPOCHS"
    "trainer.logger=['console','file']"
    trainer.n_gpus_per_node="$GPUS"
    trainer.val_freq="$VAL_FREQ"
    trainer.save_freq="$SAVE_FREQ"
    trainer.save_limit="$SAVE_LIMIT"
    trainer.save_model_only="$SAVE_MODEL_ONLY"
    "${ARM_ARGS[@]}"
)
CMD+=(${EXTRA[@]+"${EXTRA[@]}"})

printf '%q ' "${CMD[@]}"; echo
[[ $DRY_RUN -eq 1 ]] && { echo "[dry-run] not running."; exit 0; }

if [[ $PREFLIGHT -eq 1 ]]; then
    # Hand the checker the same overrides the trainer would get, so what is
    # verified is this command and not a paraphrase of it. Everything after
    # `config=...` is a key=value dotlist.
    echo
    exec python3 "$REPO/verify_ease_setup.py" --ease-repo "$EASE_REPO" -- "${CMD[@]:4}"
fi

cd "$EASE_REPO"
# Not exec: `set -o pipefail` is what makes the trainer's exit status survive
# the tee, and exec would hand it to tee instead.
"${CMD[@]}" 2>&1 | tee -a "$LOG_DIR/train.log"
