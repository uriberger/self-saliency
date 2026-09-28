#!/usr/bin/env bash
#SBATCH --job-name=ease-data
#SBATCH --account=nvr_israel_rlop
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/ease_data_%j.log
#
# Build EASE's train/val parquet from saliency-r1-8k.
#
#   saliency-r1-8k --[export_saliency_r1_8k_for_ease.py]--> raw/ + images/
#   raw/ --[ease_repo/scripts/prepare_ease_dataset.py]--> train.parquet, val.parquet
#
# The second step is THEIR converter, unmodified. It owns the box -> pixel
# conversion, the `<image>` prefix, the reward_model block and the split, so the
# only thing we contribute is the raw layout it reads.
#
# Two flags are load-bearing and not their defaults:
#
#   --min_pixels / --max_pixels   The converter defaults to 50176/1048576 but
#       examples/config.yaml trains at 262144/4194304. They only matter when
#       --model is given (prompt_length is computed under them), but a
#       prompt_length measured at a different resolution than training uses
#       would filter the wrong rows. We pass the config.yaml values.
#
#   --val_ratio 0.0124            Their default 0.1 would hold out 808 rows.
#       Our TRL overlap runs on this same corpus hold out 100
#       (train_test_split(test_size=100, seed=42)), so we match that: ~95 rows
#       after their per-source max(1, int(n*ratio)) rounding, leaving ~7,985 to
#       train on against our runs' 7,980. Comparing training-set sizes across
#       the two stacks is the point.
#
# Note what EASE's loader will do to these images: saliency-r1-8k ships them
# pre-resized to a long side of <=512, so nearly every one is BELOW min_pixels
# 262144 and process_image UPSCALES it. That is their setting and both EASE and
# the DAPO baseline see it, but it is not the resolution our overlap runs used.
#
# Usage:
#   sbatch prepare_ease_saliency_data.sh
#   bash   prepare_ease_saliency_data.sh --limit 64 --out /tmp/ease_smoke   # smoke
#   bash   prepare_ease_saliency_data.sh --no-prompt-length                 # skip the processor pass
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$REPO"

OUT="$REPO/cold_data/ease/saliency_r1_8k"
DATASET="peterant330/saliency-r1-8k"
MODEL="$REPO/checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged__tf457"
VAL_RATIO=0.0124
MIN_PIXELS=262144
MAX_PIXELS=4194304
SEED=42
LIMIT=0
PROMPT_LENGTH=1
PYBIN=${PYBIN:-/home/uberger/scratch/miniconda3/envs/ease/bin/python}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)                OUT="$2";       shift 2 ;;
        --dataset)            DATASET="$2";   shift 2 ;;
        --model)              MODEL="$2";     shift 2 ;;
        --val-ratio)          VAL_RATIO="$2"; shift 2 ;;
        --min-pixels)         MIN_PIXELS="$2"; shift 2 ;;
        --max-pixels)         MAX_PIXELS="$2"; shift 2 ;;
        --seed)               SEED="$2";      shift 2 ;;
        --limit)              LIMIT="$2";     shift 2 ;;
        --no-prompt-length)   PROMPT_LENGTH=0; shift ;;
        -h|--help) sed -n '2,48p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ "$OUT" = /* ]] || OUT="$REPO/$OUT"
EASE_REPO="$REPO/ease_repo"
[[ -d "$EASE_REPO" ]] || { echo "ERROR: $EASE_REPO not found." >&2; exit 1; }

export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

mkdir -p "$(dirname "$OUT")" logs

echo "=========================================================================="
echo "dataset : $DATASET"
echo "out     : $OUT"
echo "model   : $([[ $PROMPT_LENGTH -eq 1 ]] && echo "$MODEL" || echo '(prompt_length skipped)')"
echo "=========================================================================="

# ── 1. raw layout + image files ─────────────────────────────────────────────
"$PYBIN" "$REPO/export_saliency_r1_8k_for_ease.py" \
    --dataset "$DATASET" \
    --out-dir "$OUT" \
    ${LIMIT:+$([[ "$LIMIT" -gt 0 ]] && echo "--limit $LIMIT")}

# ── 2. their converter ──────────────────────────────────────────────────────
mapfile -t SOURCES < <(cd "$OUT/raw" && ls -d */ 2>/dev/null | sed 's|/$||' | sort)
[[ ${#SOURCES[@]} -gt 0 ]] || { echo "ERROR: no source dirs under $OUT/raw" >&2; exit 1; }
echo
echo "sources: ${SOURCES[*]}"

if [[ $PROMPT_LENGTH -eq 1 && ! -d "$MODEL" ]]; then
    echo "ERROR: --model $MODEL not found. Run stage_ease_checkpoint.sh first," >&2
    echo "       or pass --no-prompt-length." >&2
    exit 1
fi

# Deliberately WITHOUT --model_path. That flag makes their converter compute
# prompt_length, which is worth having (verl/utils/dataset.py takes a fast path
# on the column instead of re-tokenizing 8,080 images at every start), but it
# does it in one process at ~1.7 s/row -- four hours. Step 3 does the same
# computation, importing their function, across a process pool.
cd "$EASE_REPO"
"$PYBIN" scripts/prepare_ease_dataset.py \
    --input_dir "$OUT/raw" \
    --output_dir "$OUT/parquet" \
    --image_root "$OUT/images" \
    --datasets "${SOURCES[@]}" \
    --question_key question \
    --answer_key answer \
    --image_path_key image_path \
    --bbox_key evidence_bboxes \
    --val_ratio "$VAL_RATIO" \
    --seed "$SEED" \
    --min_pixels "$MIN_PIXELS" \
    --max_pixels "$MAX_PIXELS"

cd "$REPO"

# ── 3. prompt_length, in parallel ───────────────────────────────────────────
if [[ $PROMPT_LENGTH -eq 1 ]]; then
    echo
    EASE_REPO="$EASE_REPO" "$PYBIN" "$REPO/add_ease_prompt_length.py" \
        --data-dir "$OUT" \
        --model "$MODEL" \
        --ease-repo "$EASE_REPO" \
        --jobs "${SLURM_CPUS_PER_TASK:-16}" \
        --min-pixels "$MIN_PIXELS" \
        --max-pixels "$MAX_PIXELS"
fi
echo
echo "=========================================================================="
"$PYBIN" - "$OUT" <<'PYEOF'
import sys
import pandas as pd

out = sys.argv[1]
for split in ("train", "val"):
    df = pd.read_parquet(f"{out}/parquet/{split}.parquet")
    print(f"{split}: {len(df)} rows")
    print("  per source:", df["data_source"].value_counts().to_dict())
    print("  boxes/row :", df["bbox"].map(len).value_counts().to_dict())
    if "prompt_length" in df.columns:
        print(f"  prompt_len: median {int(df.prompt_length.median())}  "
              f"p99 {int(df.prompt_length.quantile(0.99))}  max {int(df.prompt_length.max())}"
              f"   (max_prompt_length is 2048)")
PYEOF
echo
echo "train : $OUT/parquet/train.parquet"
echo "val   : $OUT/parquet/val.parquet"
echo "images: $OUT/images"
echo "=========================================================================="
