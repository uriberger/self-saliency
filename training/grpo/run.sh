#!/usr/bin/env bash
# Train one arm of the paper.
#
#   bash training/grpo/run.sh self_saliency
#   bash training/grpo/run.sh center_rect --num-gpus 8
#   bash training/grpo/run.sh self_saliency --dry-run     # print the plan, launch nothing
#
# The arm is the config. Everything that distinguishes the six arms lives in
# training/grpo/configs/, and this script reads it through `training.grpo.config` rather
# than reimplementing any of it -- so the configs, the checks against the published
# checkpoints, and what actually reaches the trainer are one implementation.
#
# That is the whole design change from the research launcher this replaces, which was
# 2,377 lines of shell parsing 72 flags, where an arm was a particular combination of
# them and roughly 47 belonged to experiments that never reached the paper.
#
# GPU LAYOUT. The number of TRAINING ranks is stated by the config, not inferred, because
# prompts-per-step is the rank count and therefore the total optimizer steps:
#
#     steps = 7,980 rows * 3 epochs / training_ranks
#
# center_rect and question_boxes need no detector, so the obvious inference is "one more
# GPU for training" -- 7 ranks, 3,420 steps. Both published checkpoints have 3,990, i.e.
# 6 ranks, with one card left idle. Inferring here would silently retrain two of the five
# arms at a different length. The script warns about the idle card rather than using it.
set -euo pipefail

REPO=${SELFSAL_ROOT:-$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../.." && pwd)}
cd "$REPO"

ARM=""; NUM_GPUS=""; OUTPUT_DIR=""; DRY_RUN=false; EXTRA=()
TRL_REPO=${TRL_REPO:-$REPO/third_party/trl_repo}
PYTHON=${SELFSAL_PYTHON:-python}
VLLM_PORT=${VLLM_PORT:-8000}
DINO_PORT=${DINO_PORT:-8100}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus)   NUM_GPUS="$2";   shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true;    shift ;;
        -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
        -*)           EXTRA+=("$1"); shift ;;
        *)            if [[ -z "$ARM" ]]; then ARM="$1"; else EXTRA+=("$1"); fi; shift ;;
    esac
done
[[ -n "$ARM" ]] || { echo "usage: run.sh <arm> [--num-gpus N] [--dry-run]" >&2; exit 2; }

CONFIG="training/grpo/configs/${ARM}.yaml"
[[ -f "$CONFIG" ]] || { echo "no such arm: $CONFIG" >&2; ls training/grpo/configs/ >&2; exit 2; }

# ---- the config decides everything ------------------------------------------
eval "$("$PYTHON" -m training.grpo.config "$CONFIG" --emit-layout)"
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/checkpoint/$ARM}
mapfile -t TRAIN_ARGS < <("$PYTHON" -m training.grpo.config "$CONFIG" --emit-flags "$OUTPUT_DIR")

# ---- allocate the GPUs -------------------------------------------------------
if [[ -z "$NUM_GPUS" ]]; then
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    [[ "$NUM_GPUS" -gt 0 ]] || { echo "no GPUs found; pass --num-gpus" >&2; exit 1; }
fi
SIDECARS=0
[[ "$GENERATION" == vllm ]] && SIDECARS=$(( SIDECARS + 1 ))
[[ "$NEEDS_DINO" == true ]] && SIDECARS=$(( SIDECARS + 1 ))
NEEDED=$(( TRAINING_RANKS + SIDECARS ))
if (( NUM_GPUS < NEEDED )); then
    echo "ERROR: $ARM needs $TRAINING_RANKS training ranks + $SIDECARS sidecar(s) = $NEEDED GPUs, have $NUM_GPUS." >&2
    echo "       Reducing the rank count changes prompts-per-step and therefore the" >&2
    echo "       total optimizer steps, so it is not a free knob -- see the config." >&2
    exit 1
fi
NEXT=0
VLLM_GPU=""; if [[ "$GENERATION" == vllm ]]; then VLLM_GPU=$NEXT; NEXT=$(( NEXT + 1 )); fi
DINO_GPU=""; if [[ "$NEEDS_DINO" == true ]]; then DINO_GPU=$NEXT; NEXT=$(( NEXT + 1 )); fi
DINO_DESC=$([[ -n "$DINO_GPU" ]] && echo "GPU $DINO_GPU, port $DINO_PORT" || echo "not needed by this arm")
GEN_DESC=$([[ -n "$VLLM_GPU" ]] && echo "vLLM on GPU $VLLM_GPU, port $VLLM_PORT" || echo "in-process (use_vllm=False)")
TRAIN_GPUS=$(seq -s, "$SIDECARS" $(( SIDECARS + TRAINING_RANKS - 1 )))
IDLE=$(( NUM_GPUS - NEEDED ))

cat <<EOF
=== $ARM ===
  config          $CONFIG
  output          $OUTPUT_DIR
  training ranks  $TRAINING_RANKS   (GPUs $TRAIN_GPUS)
  Grounding-DINO  $DINO_DESC
  generation      $GEN_DESC
  expected steps  $EXPECT_STEPS
EOF
(( IDLE > 0 )) && echo "  NOTE            $IDLE GPU(s) idle. Not reassigned to training: the rank count sets the step count."
echo

if [[ "$DRY_RUN" == true ]]; then
    echo "--- trainer arguments ---"
    printf '  %s\n' "${TRAIN_ARGS[@]}" | paste - - 2>/dev/null || printf '  %s\n' "${TRAIN_ARGS[@]}"
    echo; echo "dry run -- nothing launched."
    exit 0
fi

[[ -d "$TRL_REPO/trl" ]] || {
    echo "no TRL checkout at $TRL_REPO. See docs/install.md, then: bash env/patches/trl.sh" >&2
    exit 1; }

mkdir -p "$OUTPUT_DIR/sidecar_logs"
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

if [[ -n "$DINO_GPU" ]]; then
    echo "starting Grounding-DINO on GPU $DINO_GPU ..."
    CUDA_VISIBLE_DEVICES="$DINO_GPU" "$PYTHON" -m selfsal.grounding.server --port "$DINO_PORT" \
        > "$OUTPUT_DIR/sidecar_logs/dino.log" 2>&1 &
    PIDS+=($!)
    TRAIN_ARGS+=(--dino_api_base "http://127.0.0.1:$DINO_PORT")
fi

if [[ -n "$VLLM_GPU" ]]; then
echo "starting vLLM on GPU $VLLM_GPU ..."
CUDA_VISIBLE_DEVICES="$VLLM_GPU" "$PYTHON" -m trl.scripts.vllm_serve \
    --model "$(sed -n '/--model_name_or_path/{n;p;}' <<<"$(printf '%s\n' "${TRAIN_ARGS[@]}")")" \
    --port "$VLLM_PORT" --gpu_memory_utilization "${VLLM_GPU_MEM:-0.85}" \
    > "$OUTPUT_DIR/sidecar_logs/vllm.log" 2>&1 &
PIDS+=($!)
echo "waiting for vLLM ..."
for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1 && break
    sleep 5
done
fi

cd "$TRL_REPO"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    --num_processes "$TRAINING_RANKS" \
    examples/scripts/grpo_vlm_qwen3.py \
    "${TRAIN_ARGS[@]}" \
    ${VLLM_GPU:+--vllm_server_host 127.0.0.1 --vllm_server_port "$VLLM_PORT"} \
    --report_to wandb \
    "${EXTRA[@]:-}"

echo "finished $ARM"
