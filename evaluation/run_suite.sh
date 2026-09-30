#!/bin/bash
# Run lmms-eval on every benchmark listed in a file, for one model, resumably.
#
# Each benchmark is run as its OWN lmms_eval invocation. This gives three
# levels of resume, so re-running with the same model just continues:
#   * finished benchmark  -> a *_results.json already records it   -> SKIPPED
#   * started benchmark   -> no results.json yet, but the sqlite response
#                            cache holds its finished samples       -> RESUMED
#                            (cache replays done samples, runs the rest)
#   * untouched benchmark -> no cache, no results.json              -> STARTED
# Because each benchmark writes its own results.json, a crash / wall-clock kill
# on benchmark N never loses the completion record of benchmarks 1..N-1.
#
# lmms-eval: the pinned submodule at evaluation/lmms_eval. Conda env: selfsal-eval.
# Results land in evaluation/results/<model_slug>/<model_subdir>/.
#
# Usage:
#   bash evaluation/run_suite.sh --model Qwen/Qwen2.5-VL-7B-Instruct
#   bash evaluation/run_suite.sh --model /path/to/ckpt --model-type qwen2_5_vl
#   bash evaluation/run_suite.sh --model ... --benchmarks-file my_list.txt
#   bash evaluation/run_suite.sh --model ... --num-gpus 4      # accelerate DP
#   bash evaluation/run_suite.sh --model ... --direct          # run here (GPU node)
#   bash evaluation/run_suite.sh --model ... --limit 8         # extra args forwarded
#
# MMBench-style answer extraction uses an OpenAI-compatible API; export
# OPENAI_API_KEY before launching or it falls back to exact match. The default is
# OpenAI's public API with gpt-4o-mini, the judge the paper reports (Appendix C).
# OPENAI_API_URL and MODEL_VERSION move TOGETHER -- a gateway addresses the same
# model by a provider-prefixed name; see docs/install.md. evaluation/submit.sh has
# the same env-override knobs.
#
# Environment overrides:  PARTITION=batch_block1  DURATION=4 (hours)
#   OPENAI_API_KEY=...  MODEL_VERSION=gpt-4o-mini
#   OPENAI_API_URL=https://api.openai.com/v1/chat/completions
set -e

# ADLR cluster-interface tools (submit_job, etc.) on PATH.
# A site's batch-submission tooling, if any. Empty by default; --direct needs none.
[ -n "${SUBMIT_JOB_BIN:-}" ] && export PATH="$SUBMIT_JOB_BIN:$PATH"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---------- cluster / project constants ----------
ACCOUNT=${SLURM_ACCOUNT:?set SLURM_ACCOUNT}
PARTITION=${PARTITION:-batch_block1}
DURATION=${DURATION:-4}
PROJECT=${SELFSAL_ROOT:-.}/../vlm_reasoning
LMMS_EVAL_DIR=${SELFSAL_ROOT:-.}/../lmms-eval
CONDA_SH=${CONDA_ROOT:?set CONDA_ROOT}/etc/profile.d/conda.sh
CONDA_ENV=lmms_eval
HF_HOME=${HF_HOME:-${HF_HOME:?set HF_HOME}}

# ---------- experiment defaults ----------
MODEL="Qwen/Qwen3-VL-8B-Instruct"
MODEL_TYPE=""     # auto-detected from the model name unless given
BENCHMARKS_FILE="$SCRIPT_DIR/lmms_eval_benchmarks.txt"
NUM_GPUS=1
DIRECT=0
TAG=""            # suffix for the results dir, isolating config variants from each other
VGA_MODE=0        # --vga: Vision-Guided Attention (arXiv:2511.20032); see below
VGA_ARGS_CLI=""
EXTRA_ARGS=""     # forwarded verbatim to lmms_eval (e.g. --limit 8)

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)            MODEL="$2";           shift 2 ;;
        --model-type)       MODEL_TYPE="$2";      shift 2 ;;
        --benchmarks-file)  BENCHMARKS_FILE="$2"; shift 2 ;;
        --num-gpus)         NUM_GPUS="$2";        shift 2 ;;
        --tag)              TAG="$2";             shift 2 ;;
        --vga)              VGA_MODE=1;           shift ;;
        --vga-args)         VGA_ARGS_CLI="$2"; VGA_MODE=1; shift 2 ;;
        --direct)           DIRECT=1;             shift ;;
        *)                  EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# --vga: VGA is training-free, so the arm lives entirely in the configuration --
# there is no checkpoint whose name would distinguish it. That makes the results
# directory the only thing between a guided run and the stock run of the same
# weights, and the loop below RESUMES from whatever it finds there: an untagged
# guided run would skip every benchmark the stock run had already finished and
# report those numbers as its own. So the tag encodes the arm.
if [[ $VGA_MODE -eq 1 ]]; then
    : "${MODEL_TYPE:=qwen3_vl_vga}"
    VGA_ARGS="${VGA_ARGS_CLI:-${VGA_ARGS:-}}"
    _vga_get() { echo "$VGA_ARGS" | tr ',' '\n' | sed -n "s/^ *$1 *= *//p" | tail -1; }
    _b=$(_vga_get beta);        : "${_b:=0.2}"
    _s=$(_vga_get start_layer); : "${_s:=4}"
    _e=$(_vga_get end_layer);   : "${_e:=16}"
    _m=$(_vga_get mode)
    _rest=$(echo "$VGA_ARGS" | tr ',' '\n' \
        | grep -vE '^ *(beta|start_layer|end_layer|mode) *=' \
        | sed 's/ *//g; s/=/-/' | paste -sd_ -)
    : "${TAG:=vga_b${_b}_l${_s}-${_e}${_m:+_${_m}}${_rest:+_${_rest}}}"
fi

[[ -f "$BENCHMARKS_FILE" ]] || { echo "Benchmarks file not found: $BENCHMARKS_FILE" >&2; exit 1; }

# Auto-detect the lmms-eval model class from the model name.
if [[ -z "$MODEL_TYPE" ]]; then
    if echo "$MODEL" | grep -qiE "qwen3[-_.]?vl"; then
        MODEL_TYPE=qwen3_vl
    elif echo "$MODEL" | grep -qiE "qwen2[-_.]?5[-_.]?vl"; then
        MODEL_TYPE=qwen2_5_vl
    else
        echo "Cannot infer --model-type from '$MODEL'; pass e.g. --model-type qwen2_5_vl" >&2
        exit 1
    fi
fi

MODEL_SLUG=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g')
TAG_SLUG=${TAG:+_${TAG}}
OUTPUT_DIR="$PROJECT/results/lmms_eval/${MODEL_SLUG}${TAG_SLUG}"
JOB_NAME="lmms_eval_suite_${MODEL_SLUG}${TAG_SLUG}"
LOG_ROOT="$PROJECT/outputs/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_ROOT"

echo "Model:           $MODEL ($MODEL_TYPE)"
[[ $VGA_MODE -eq 1 ]] && echo "VGA:             ${VGA_ARGS:-(defaults: beta=0.2 layers 4-16)}"
echo "Benchmarks file: $BENCHMARKS_FILE"
echo "GPUs:            $NUM_GPUS"
echo "Output dir:      $OUTPUT_DIR"
OPENAI_API_URL=${OPENAI_API_URL:-https://api.openai.com/v1/chat/completions}
MODEL_VERSION=${MODEL_VERSION:-gpt-4o-mini}
echo "OpenAI key:      $([[ -n "$OPENAI_API_KEY" ]] && echo "(set — mmbench uses $MODEL_VERSION via $OPENAI_API_URL)" || echo '(unset — mmbench falls back to exact match)')"
[[ -n "$EXTRA_ARGS" ]] && echo "Extra args:      $EXTRA_ARGS"
echo ""

# The resume loop runs INSIDE the job (whether --direct or slurm), so an
# autoresume after the wall-clock limit re-scans results.json and skips
# whatever already finished. One lmms_eval invocation per pending benchmark.
INNER_CMD="
    set -e;
    source $CONDA_SH;
    conda activate $CONDA_ENV;
    export HF_HOME=$HF_HOME;
    export HF_TOKEN=${HF_TOKEN:-};
    export OPENAI_API_KEY=${OPENAI_API_KEY:-};
    export OPENAI_API_URL=$OPENAI_API_URL;
    export MODEL_VERSION=$MODEL_VERSION;
    export LMMS_LOCAL_CACHE=/nonexistent;
    export LMMS_CACHE_EVAL_VERSION=${LMMS_CACHE_EVAL_VERSION:-cache-v1};
    export VGA_ARGS='${VGA_ARGS:-}';
    export VGA_REPO=${VGA_REPO:-$PROJECT};
    cd $LMMS_EVAL_DIR;
    if [[ $NUM_GPUS -gt 1 ]]; then
        LAUNCH=\"accelerate launch --num_processes=$NUM_GPUS -m lmms_eval\";
    else
        LAUNCH=\"python -m lmms_eval\";
    fi;

    # Pending = benchmarks in the file not yet recorded finished. Recomputed at
    # loop start; each benchmark's own results.json updates it across restarts.
    mapfile -t FINISHED < <(python3 $SCRIPT_DIR/lmms_eval_finished_tasks.py $OUTPUT_DIR);
    is_finished() { local t=\$1; for f in \"\${FINISHED[@]}\"; do [[ \"\$f\" == \"\$t\" ]] && return 0; done; return 1; };

    while IFS= read -r line; do
        TASK=\"\${line%%#*}\";                 # strip inline comments
        TASK=\"\$(echo \$TASK | xargs)\";       # trim whitespace
        [[ -z \"\$TASK\" ]] && continue;
        if is_finished \"\$TASK\"; then
            echo \"[skip]   \$TASK — already finished\";
            continue;
        fi;
        echo \"[run ]   \$TASK\";
        \$LAUNCH \
            --model $MODEL_TYPE \
            --model_args pretrained=$MODEL \
            --tasks \"\$TASK\" \
            --batch_size 1 \
            --log_samples \
            --use_cache $OUTPUT_DIR/cache \
            --output_path $OUTPUT_DIR \
            $EXTRA_ARGS \
        && echo \"[done]   \$TASK\" \
        || echo \"[fail]   \$TASK — see log above; other benchmarks continue\";
    done < $BENCHMARKS_FILE;
    echo \"All benchmarks processed for $MODEL_SLUG\";
"

if [[ $DIRECT -eq 1 ]]; then
    bash -c "$INNER_CMD"
    echo "Finished $JOB_NAME"
    exit 0
fi

source "$SCRIPT_DIR/slurm/slurm_job_cap.sh"
wait_for_slurm_capacity "$JOB_NAME"

# Bare node (no --image): the conda env provides the full stack; /lustre is
# natively mounted.
submit_job \
    --account "$ACCOUNT" \
    --partition "$PARTITION" \
    --name "$JOB_NAME" \
    --gpu "$NUM_GPUS" \
    --duration "$DURATION" \
    --autoresume_uninstrumented \
    --outfile "$LOG_ROOT/${JOB_NAME}.%j.out" \
    --logroot "$LOG_ROOT" \
    -c "bash -c '$INNER_CMD'"

echo "Submitted $JOB_NAME"
