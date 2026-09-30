#!/usr/bin/env bash
# Section 4.4: what the trained model changed -- its text, or its attention.  Table 3.
#
#   bash experiments/trained_model/run.sh --arm coldstart=CKPT --arm self_saliency=CKPT \
#       --out-dir DIR [--gpus 8] [--n-samples 100]
#
# Table 3 is four columns over two arms, and they are not all the same kind of number:
#
#   completion length   \  TEXT statistics, on each model's OWN generations, because the
#   box union area      /  question is what the policy chose to say
#   in-region share     \  ATTENTION statistics, TEACHER-FORCED on a fixed set of chains
#     (selected heads)  |  (the 100 each model generated, so 200 per model), because
#   in-region share     /  otherwise a change in attention and a change in text are one
#     (layer-22 heads)     number and the section's whole question is which of the two moved
#
# So the run is two programs. `probe` generates once per arm on the SAME prompts and
# scores every reward the trainer computes; `audit` takes those generations and does the
# rest, including the cross-pass that teacher-forces each arm's chains through each model.
#
#   stage      what it does                                  where
#   probe      generate + score per arm                      GPU, sharded
#   text       H1/H2/H4: what the policy says                CPU
#   dino       ground the stored sentences                   GPU, sharded
#   crosspass  phi(text of arm A, attention of model B)      GPU, sharded  <- the teacher forcing
#   report     the table                                     CPU
#
# --arm NAME=PATH is repeatable and is how the two rows are named; PATH is a merged
# checkpoint, or BASE:ADAPTER for a LoRA. The paper's two are the cold start and
# SELF-SALIENCY; docs/reproduce.md maps every arm to its published checkpoint.
#
# COST. The probe generates 8 rollouts per prompt through an 8B model with Grounding-DINO
# and the step classifier on the same card, so budget ~45 min per arm on 8 GPUs at 100
# samples. `crosspass` is one teacher-forced pass per (chain, model) and is much cheaper.
#
# Resuming: re-run the identical command. Every stage is append-only per unit.
set -euo pipefail

REPO=${SELFSAL_ROOT:-$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../.." && pwd)}
cd "$REPO"

PYTHON=${SELFSAL_PYTHON:-python}
PROBE=(-m experiments.trained_model.probe)
AUDIT=(-m experiments.trained_model.audit)

GPUS=""; OUT_DIR=""; N_SAMPLES=100; DRY_RUN=false; STAGES=""; ARMS=(); EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm)        ARMS+=("$2");    shift 2 ;;
        --gpus)       GPUS="$2";       shift 2 ;;
        --out-dir)    OUT_DIR="$2";    shift 2 ;;
        --n-samples)  N_SAMPLES="$2";  shift 2 ;;
        --stages)     STAGES="$2";     shift 2 ;;
        --dry-run)    DRY_RUN=true;    shift   ;;
        -h|--help)    sed -n '2,35p' "$0"; exit 0 ;;
        *)            EXTRA+=("$1");   shift   ;;
    esac
done

[[ -n "$OUT_DIR" ]] || { echo "--out-dir is required" >&2; exit 2; }
[[ ${#ARMS[@]} -ge 1 ]] || { echo "at least one --arm NAME=PATH is required" >&2; exit 2; }
STAGES=${STAGES:-probe,text,dino,crosspass,report}

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

if [[ -z "$GPUS" ]]; then
    GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    [[ "$GPUS" -gt 0 ]] || GPUS=1
fi

# R_llm is one of the rewards the probe reports, so the judge key is needed for the same
# reason training needs it. Without it every judged sample MASKS rather than scoring zero
# (see selfsal/judge.py), which is safe but makes that column empty -- so say so once,
# here, rather than letting it show up as a blank column in the table.
if [[ -z "${OPENAI_API_KEY:-}${NVIDIA_API_KEY:-}" ]]; then
    echo "NOTE: neither OPENAI_API_KEY nor NVIDIA_API_KEY is set; the judge reward will be"
    echo "      masked on every sample. Table 3 does not use it, so this is only a warning."
fi

mkdir -p "$OUT_DIR/logs"
cat <<EOF
=== trained-model audit (Section 4.4, Table 3) ===
  out        $OUT_DIR
  arms       ${ARMS[*]}
  samples    $N_SAMPLES     shards $GPUS
  stages     $STAGES
EOF

run_sharded () {   # $1 = label, rest = the command before --shard/--num-shards
    local label="$1"; shift
    local pids=() fail=0 i
    for ((i = 0; i < GPUS; i++)); do
        CUDA_VISIBLE_DEVICES="$i" "$@" --shard "$i" --num-shards "$GPUS" --device cuda:0 \
            >"$OUT_DIR/logs/${label}_shard${i}.log" 2>&1 &
        pids+=($!)
        echo "  [launch] $label shard $i -> GPU $i (pid ${pids[-1]})"
    done
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then echo "  [done] $label shard $i ok"
        else echo "  [FAIL] $label shard $i -- see $OUT_DIR/logs/${label}_shard${i}.log" >&2
             tail -20 "$OUT_DIR/logs/${label}_shard${i}.log" >&2 || true; fail=1; fi
    done
    return $fail
}

has_stage () { [[ ",$STAGES," == *",$1,"* ]]; }

PROBE_ARGS=()
for spec in "${ARMS[@]}"; do
    name=${spec%%=*}; path=${spec#*=}
    PROBE_ARGS+=(--probe "$OUT_DIR/probe/$name/probe_merged.json")
done

if [[ "$DRY_RUN" == true ]]; then
    echo; echo "stages: $STAGES"
    for spec in "${ARMS[@]}"; do echo "  probe  ${spec%%=*}  <- ${spec#*=}"; done
    echo "  audit  ${PROBE_ARGS[*]}"
    echo "dry run -- nothing launched."
    exit 0
fi

# ---- 1. generate and score, once per arm, on the SAME prompts ----------------
# The prompts are the same across arms because --dataset, --n-samples and --seed are, and
# the probe draws its sample from those alone. That is what makes every delta below paired.
if has_stage probe; then
    for spec in "${ARMS[@]}"; do
        name=${spec%%=*}; path=${spec#*=}
        base=${path%%:*}; adapter=${path#*:}
        arm_args=(--base-model "$base" --out-dir "$OUT_DIR/probe/$name"
                  --n-samples "$N_SAMPLES" --map attn)
        [[ "$adapter" != "$path" ]] && arm_args+=(--trained-adapter "$adapter")
        echo "[probe] $name"
        run_sharded "probe_$name" "$PYTHON" "${PROBE[@]}" "${arm_args[@]}" \
            "${EXTRA[@]+"${EXTRA[@]}"}" || exit 1
        "$PYTHON" "${PROBE[@]}" --render --out-dir "$OUT_DIR/probe/$name"
    done
fi

# ---- 2. the audit stages -----------------------------------------------------
if has_stage text; then
    echo "[audit] text"
    "$PYTHON" "${AUDIT[@]}" --stage text "${PROBE_ARGS[@]}" --out-dir "$OUT_DIR" \
        2>&1 | tee "$OUT_DIR/logs/text.log"
fi

for stage in dino crosspass; do
    has_stage "$stage" || continue
    echo "[audit] $stage"
    run_sharded "$stage" "$PYTHON" "${AUDIT[@]}" --stage "$stage" \
        "${PROBE_ARGS[@]}" --out-dir "$OUT_DIR" "${EXTRA[@]+"${EXTRA[@]}"}" || exit 1
    "$PYTHON" "${AUDIT[@]}" --stage "$stage" --out-dir "$OUT_DIR" --merge
done

if has_stage report; then
    echo "[audit] report"
    "$PYTHON" "${AUDIT[@]}" --stage report --out-dir "$OUT_DIR" | tee "$OUT_DIR/report.txt"
    echo
    echo "Table 3 is the text columns of $OUT_DIR/report.txt beside the crosspass diagonal."
fi
