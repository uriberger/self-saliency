#!/usr/bin/env bash
# Section 5: where inside the picture the attention piles up.  Table 4, Figure 4, Table 8.
#
#   bash experiments/attention_bias/run.sh --stage corpus   --out-dir DIR
#   bash experiments/attention_bias/run.sh --stage selftest --out-dir DIR --model M
#   bash experiments/attention_bias/run.sh --stage scan     --out-dir DIR --model M --gpus 8
#   bash experiments/attention_bias/run.sh --stage arms     --out-dir DIR --model M --gpus 8
#   bash experiments/attention_bias/run.sh --stage report   --out-dir DIR
#
# Table 4 is four models, so `corpus` once and then selftest/scan per model into its own
# directory; `tables.py --dirs A,B,C,D` puts them side by side (that is also Figure 5's
# command, with --panels). Table 8 is the model characteristics and needs no run.
#
#   for M in Qwen/Qwen3-VL-8B-Instruct OpenGVLab/InternVL3_5-8B \
#            zai-org/GLM-4.1V-9B-Thinking nvidia/Nemotron-Nano-Omni; do
#       bash experiments/attention_bias/run.sh --stage selftest --out-dir out/$M --model $M
#       bash experiments/attention_bias/run.sh --stage scan     --out-dir out/$M --model $M
#   done
#   python -m experiments.attention_bias.tables --dirs out/A,out/B,out/C,out/D --out-dir out/tables
#
# SELFTEST GATES scan AND arms and is not optional. It checks that the scan reproduces
# stock SDPA (it edits nothing, and that has to be measured), that the ring's area is what
# the formula says, that the negative controls come back at 1.0 -- and, above all, that
# every transform's pixel-to-patch mapping decodes where `patch_correspondence` claims. An
# off-by-one there answers the content-versus-position question confidently and backwards,
# and no later table would look wrong.
#
# RUN corpus FIRST, on CPU. It is the only stage that touches `datasets`, so the GPU
# stages are deterministic, offline and shard trivially. It is also where the grid census
# comes from, and the grid census is what stops a cross-type comparison of raw ring
# percentages: the ring is 23% of a 16x16 grid and 50% of a 6x8 one.
#
# The human-box row of Table 4 is a different corpus -- the 1,800 Visual-CoT pairs that
# carry an annotated box -- built by `python -m selfsal.data.boxed_corpus`.
#
# COST. One picture is one prefill at batch size 1, well under a second, so the scan is
# minutes rather than hours: this experiment generates nothing, grounds nothing and judges
# nothing. Roughly: corpus ~1 h on CPU, scan ~10 min on 8 GPUs, arms ~30 min, report
# instant.
#
# Resuming: re-run the identical command. Results are append-only JSONL keyed by unit and
# the bulk arrays are flushed in parts, so a killed shard loses at most one part.
set -euo pipefail

REPO=${SELFSAL_ROOT:-$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../.." && pwd)}
cd "$REPO"

PYTHON=${SELFSAL_PYTHON:-python}
PROBE=(-m experiments.attention_bias.probe)

GPUS=""; STAGE=scan; OUT_DIR=""; MODEL=""; DRY_RUN=false; EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)     GPUS="$2";    shift 2 ;;
        --stage)    STAGE="$2";   shift 2 ;;
        --out-dir)  OUT_DIR="$2"; shift 2 ;;
        --model)    MODEL="$2";   shift 2 ;;
        --dry-run)  DRY_RUN=true; shift   ;;
        -h|--help)  sed -n '2,42p' "$0"; exit 0 ;;
        *)          EXTRA+=("$1"); shift  ;;
    esac
done

[[ -n "$OUT_DIR" ]] || { echo "--out-dir is required" >&2; exit 2; }
case "$STAGE" in
    selftest|scan|arms)
        [[ -n "$MODEL" ]] || { echo "--model is required for stage $STAGE" >&2; exit 2; } ;;
    corpus|report|crossmodel|verify) ;;
    *) echo "unknown --stage $STAGE" >&2; exit 2 ;;
esac

# Offline by default: every stage after `corpus` reads only what `corpus` wrote, and a
# stage that silently reaches the Hub is a stage whose inputs are not the ones on disk.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

if [[ -z "$GPUS" ]]; then
    GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    [[ "$GPUS" -gt 0 ]] || GPUS=1
fi

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/progress"

cat <<EOF
=== attention bias (Section 5) ===
  stage      $STAGE
  out        $OUT_DIR
  model      ${MODEL:-(not needed by this stage)}
  shards     $GPUS
  extra      ${EXTRA[*]:-(none)}
EOF

COMMON=(--out-dir "$OUT_DIR")
[[ -n "$MODEL" ]] && COMMON+=(--model "$MODEL")

if [[ "$DRY_RUN" == true ]]; then
    echo; echo "would run: $PYTHON ${PROBE[*]} --stage $STAGE ${COMMON[*]} ${EXTRA[*]:-}"
    echo "dry run -- nothing launched."
    exit 0
fi

# corpus is CPU-only; selftest deliberately runs on ONE GPU, because a check that passed
# on some shards and not others is not a gate.
if [[ "$STAGE" == "corpus" || "$STAGE" == "selftest" ]]; then
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PYTHON" "${PROBE[@]}" --stage "$STAGE" \
        "${COMMON[@]}" "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 | tee "$OUT_DIR/logs/$STAGE.log"
    exit "${PIPESTATUS[0]}"
fi

if [[ "$STAGE" == "report" || "$STAGE" == "crossmodel" || "$STAGE" == "verify" ]]; then
    "$PYTHON" "${PROBE[@]}" --stage "$STAGE" "${COMMON[@]}" \
        "${EXTRA[@]+"${EXTRA[@]}"}" | tee "$OUT_DIR/$STAGE.txt"
    exit 0
fi

# ---- the two sharded GPU stages ---------------------------------------------
if [[ ! -f "$OUT_DIR/logs/selftest.log" ]] || ! grep -q "SELFTEST PASS" "$OUT_DIR/logs/selftest.log"; then
    echo "refusing to run: no passing selftest in $OUT_DIR/logs/selftest.log" >&2
    echo "  bash experiments/attention_bias/run.sh --stage selftest --out-dir $OUT_DIR --model $MODEL" >&2
    exit 2
fi
if [[ ! -f "$OUT_DIR/corpus/manifest.jsonl" ]]; then
    echo "refusing to run: no corpus at $OUT_DIR/corpus/manifest.jsonl" >&2
    echo "  bash experiments/attention_bias/run.sh --stage corpus --out-dir $OUT_DIR" >&2
    exit 2
fi

# Drop the previous attempt's heartbeats: a shard writes its first only after the model
# loads, so on a resume the monitor reads the dead run's files, calls them stale and exits
# while the shards it was watching are fine. Resume state lives in the results files.
rm -f "$OUT_DIR"/progress/*.json

pids=()
for ((i = 0; i < GPUS; i++)); do
    CUDA_VISIBLE_DEVICES="$i" "$PYTHON" "${PROBE[@]}" \
        --stage "$STAGE" --shard "$i" --num-shards "$GPUS" \
        "${COMMON[@]}" "${EXTRA[@]+"${EXTRA[@]}"}" \
        >"$OUT_DIR/logs/${STAGE}_shard${i}.log" 2>&1 &
    pids+=($!)
    echo "[launch] shard $i -> GPU $i (pid ${pids[-1]})"
done

sleep 5
"$PYTHON" "${PROBE[@]}" --stage monitor --out-dir "$OUT_DIR" &
mon=$!

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[done] shard $i ok"
    else
        echo "[FAIL] shard $i -- see $OUT_DIR/logs/${STAGE}_shard${i}.log" >&2
        tail -20 "$OUT_DIR/logs/${STAGE}_shard${i}.log" >&2 || true
        fail=1
    fi
done
kill "$mon" 2>/dev/null || true
wait "$mon" 2>/dev/null || true

if [[ $fail -ne 0 ]]; then
    echo "WARNING: a shard failed; re-run the identical command to resume." >&2
    exit 1
fi
"$PYTHON" "${PROBE[@]}" --stage report --out-dir "$OUT_DIR" | tee "$OUT_DIR/report.txt"
