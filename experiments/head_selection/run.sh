#!/usr/bin/env bash
# Section 3.5: which attention heads the saliency score reads.  Selects (22,28) and (22,31).
#
#   bash experiments/head_selection/run.sh --out-dir DIR                  # the whole screen
#   bash experiments/head_selection/run.sh --out-dir DIR --stages analyze # re-sweep, CPU only
#   bash experiments/head_selection/run.sh --out-dir DIR --cross-dir DIR2 # footnote 3
#
# THE PROCEDURE, and why it is three programs rather than one:
#
#   generate   run the BASE model (Qwen3-VL-8B-Instruct, not a trained arm) over the
#              dataset with the step prompt, and write one JSONL of responses. GPU.
#   collect    for every sample: segment the chain, keep the observe steps, ground each
#              one, and capture raw per-layer per-head attention from the observe tokens
#              to the image patches in ONE teacher-forced pass. Saves
#              (n_steps, n_layers, n_heads, n_patches) per sample. GPU.
#   analyze    sweep every (layer, head) and score each by the correlation between its
#              phi and whether the model answered correctly. The top two are the answer.
#              CPU, and re-runnable without touching a GPU -- which is the point of the
#              split, because the sweep is where the choices are and the capture is where
#              the hours are.
#
# WHAT IT SHOULD PRINT. Layer 22, heads 28 and 31, out of 36 layers. That constant lives
# in `selfsal/saliency/heads.py` and is read by the reward, by Section 4.4 and by
# Figure 5; this script is where it came from. If a re-run disagrees, `heads.py` is what
# the paper trained against and this is the measurement -- reconcile, do not quietly edit
# one to match the other.
#
# FOOTNOTE 3 is the cross-dataset check: select on one dataset, report that same head's
# correlation on another. --cross-dir points at a second completed screen directory.
# Nothing is split train/test, because selection already happened on different data.
#
# The screen is run with `--bbox-source human` for the headline table (the question-level
# box the corpus ships) and with the default DINO grounding for the per-step variant;
# --bbox-source is forwarded, so pass it to choose.
set -euo pipefail

REPO=${SELFSAL_ROOT:-$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../.." && pwd)}
cd "$REPO"

PYTHON=${SELFSAL_PYTHON:-python}
GENERATE=(-m experiments.head_selection.generate)
SCREEN=(-m experiments.head_selection.screen)
CROSS=(-m experiments.head_selection.cross_dataset)

OUT_DIR=""; MODEL="Qwen/Qwen3-VL-8B-Instruct"; DATASET=""; SPLIT=""
RESULTS=""; STAGES=""; CROSS_DIR=""; METRIC="mean_in"; DRY_RUN=false; EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir)   OUT_DIR="$2";   shift 2 ;;
        --model)     MODEL="$2";     shift 2 ;;
        --dataset)   DATASET="$2";   shift 2 ;;
        --split)     SPLIT="$2";     shift 2 ;;
        --results)   RESULTS="$2";   shift 2 ;;
        --stages)    STAGES="$2";    shift 2 ;;
        --cross-dir) CROSS_DIR="$2"; shift 2 ;;
        --metrics)   METRIC="$2";    shift 2 ;;
        --dry-run)   DRY_RUN=true;   shift   ;;
        -h|--help)   sed -n '2,33p' "$0"; exit 0 ;;
        *)           EXTRA+=("$1");  shift   ;;
    esac
done

[[ -n "$OUT_DIR" ]] || { echo "--out-dir is required" >&2; exit 2; }
STAGES=${STAGES:-generate,collect,analyze}
RESULTS=${RESULTS:-$OUT_DIR/responses.jsonl}

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

mkdir -p "$OUT_DIR/logs"

DS_ARGS=()
[[ -n "$DATASET" ]] && DS_ARGS+=(--dataset "$DATASET")
[[ -n "$SPLIT"   ]] && DS_ARGS+=(--split "$SPLIT")

DS_DESC="$DATASET"; [[ -z "$DS_DESC" ]] && DS_DESC="(the screen's own default)"
[[ -n "$SPLIT" ]] && DS_DESC="$DS_DESC  split $SPLIT"
EXTRA_DESC="${EXTRA[*]:-}"; [[ -z "$EXTRA_DESC" ]] && EXTRA_DESC="(none)"

cat <<EOF
=== head selection (Section 3.5) ===
  out        $OUT_DIR
  model      $MODEL
  dataset    $DS_DESC
  responses  $RESULTS
  stages     $STAGES
  extra      $EXTRA_DESC
EOF

has_stage () { [[ ",$STAGES," == *",$1,"* ]]; }

if [[ "$DRY_RUN" == true ]]; then
    echo
    has_stage generate && echo "  $PYTHON ${GENERATE[*]} --model $MODEL ${DS_ARGS[*]:-} --output $RESULTS"
    has_stage collect  && echo "  $PYTHON ${SCREEN[*]} collect --model $MODEL ${DS_ARGS[*]:-} --results $RESULTS --output $OUT_DIR"
    has_stage analyze  && echo "  $PYTHON ${SCREEN[*]} analyze --output $OUT_DIR"
    [[ -n "$CROSS_DIR" ]] && echo "  $PYTHON ${CROSS[*]} --dir-a $OUT_DIR --dir-b $CROSS_DIR --metrics $METRIC"
    echo "dry run -- nothing launched."
    exit 0
fi

# ---- 1. generate, on the BASE model -----------------------------------------
# Deliberately the base model and not a trained arm: the heads are chosen before any
# saliency training, and choosing them on a model that was already rewarded for its
# attention would select for the reward rather than for correctness.
if has_stage generate; then
    echo "[generate] $MODEL -> $RESULTS"
    "$PYTHON" "${GENERATE[@]}" --model "$MODEL" "${DS_ARGS[@]+"${DS_ARGS[@]}"}" \
        --output "$RESULTS" "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 | tee "$OUT_DIR/logs/generate.log"
fi

# ---- 2. collect the per-head attention (GPU, resumable) ----------------------
if has_stage collect; then
    [[ -f "$RESULTS" ]] || { echo "no responses at $RESULTS; run --stages generate first" >&2; exit 2; }
    echo "[collect] per-layer per-head attention"
    "$PYTHON" "${SCREEN[@]}" collect --model "$MODEL" "${DS_ARGS[@]+"${DS_ARGS[@]}"}" \
        --results "$RESULTS" --output "$OUT_DIR" "${EXTRA[@]+"${EXTRA[@]}"}" \
        2>&1 | tee "$OUT_DIR/logs/collect.log"
fi

# ---- 3. the sweep (CPU) ------------------------------------------------------
if has_stage analyze; then
    echo "[analyze] sweeping every (layer, head)"
    "$PYTHON" "${SCREEN[@]}" analyze --output "$OUT_DIR" \
        "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 | tee "$OUT_DIR/analyze.txt"
    echo
    echo "The two heads with the highest correlation are the answer; the paper's are"
    echo "layer 22, heads 28 and 31, recorded in selfsal/saliency/heads.py."
fi

# ---- 4. footnote 3: does the pick transfer to another dataset? ---------------
if [[ -n "$CROSS_DIR" ]]; then
    echo "[cross-dataset] $OUT_DIR vs $CROSS_DIR"
    "$PYTHON" "${CROSS[@]}" --dir-a "$OUT_DIR" --label-a "$(basename "$OUT_DIR")" \
        --dir-b "$CROSS_DIR" --label-b "$(basename "$CROSS_DIR")" \
        --metrics "$METRIC" "${EXTRA[@]+"${EXTRA[@]}"}" \
        2>&1 | tee "$OUT_DIR/cross_dataset.txt"
fi
