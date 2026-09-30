#!/bin/bash
# Submit an lmms-eval benchmark job replicating the Saliency-R1 evaluation
# (arXiv 2604.04500 §4.3: "The evaluation utilizes the lmms-eval framework").
# Default task suite = the paper's Table 2 benchmarks: MMMU-Pro, MMBench (en
# test), POPE, MME, MME-RealWorld, MMStar, ChartQA, IllusionVQA, ScienceQA
# (img), SalBench (P3).
#
# lmms-eval clone: ~/scratch/research/lmms-eval, conda env: lmms_eval.
# Results land in results/lmms_eval/<model_slug>/ (json + per-sample logs).
#
# Usage:
#   bash scripts/slurm/launch_lmms_eval_job.sh                                   # Qwen3-VL-8B-Instruct, full paper suite
#   bash scripts/slurm/launch_lmms_eval_job.sh --model Qwen/Qwen2.5-VL-7B-Instruct
#   bash scripts/slurm/launch_lmms_eval_job.sh --tasks mmstar,pope --limit 64    # quick check
#   bash scripts/slurm/launch_lmms_eval_job.sh --model /path/to/checkpoint --model-type qwen2_5_vl
#   bash scripts/slurm/launch_lmms_eval_job.sh --direct                          # run in place (interactive GPU node)
#   bash scripts/slurm/launch_lmms_eval_job.sh --num-gpus 4                      # data-parallel via accelerate
#   bash scripts/slurm/launch_lmms_eval_job.sh --max-new-tokens 8192            # cap generation; results land in <slug>_mnt8192/
#   bash scripts/slurm/launch_lmms_eval_job.sh --tasks mmerealworld --max-pixels 12845056
#       # raise input resolution for high-res tasks (MME-RealWorld); dir gets _px<N>
#   bash scripts/slurm/launch_lmms_eval_job.sh --model <coldstart-or-grpo-model> --r1-mode
#       # one flag = R1 system prompt + repetition_penalty 1.05 + max_new_tokens 4096
#       # + tag "r1" (results in <slug>_mnt4096_r1/). Individual flags still override.
#   bash scripts/slurm/launch_lmms_eval_job.sh --model M --r1-mode --print-output-dir
#       # print the results dir this config resolves to and exit (no job, no mkdir)
#   bash scripts/slurm/launch_lmms_eval_job.sh --model M --ease-mode --tasks vstar_bench
#       # --r1-mode, PLUS strip each task's "answer with the letter directly"
#       # instruction and raise max_pixels to 4194304, so the numbers are
#       # comparable with EASE (arXiv 2605.30912) instead of with lmms-eval's
#       # defaults. Results land in <slug>_mnt4096_ease. See the block by the
#       # --r1-mode defaults for what it deliberately does NOT align.
#
# To sweep a whole benchmark suite one task at a time (resumable, skips what is
# already done), use scripts/eval_saliency_r1_benchmarks.sh /
# scripts/eval_our_benchmarks.sh instead of a comma-separated --tasks list.
#
# MMBench answer extraction uses an OpenAI-compatible API for answer matching.
# Export OPENAI_API_KEY before launching; the launcher defaults to OpenAI's public
# API with gpt-4o-mini, which is the judge the paper reports (Appendix C).
# OPENAI_API_URL and MODEL_VERSION override the pair, and they move TOGETHER: a
# gateway addresses the same model by a provider-prefixed name. See docs/install.md.
# Without a key it falls back to exact matching and under-reports.
#
# Task-name notes (paper name -> lmms-eval task): MME-RealWorld -> mmerealworld
# (13K high-res samples; mmerealworld_lite is the cheap variant), SalBench P3 ->
# p3, IllusionVQA -> illusionvqa (tag = comprehension + soft_localization),
# CV-Bench -> cv_bench (cv_bench_2d / cv_bench_3d are the halves), V* ->
# vstar_bench, HR-Bench -> hrbench4k + hrbench8k (`hrbench` is their group).
#
# A relaunch (or --autoresume after the 4h wall limit) resumes from the sqlite
# response cache in <output_dir>/cache, so completed requests are not re-run.
#
# Environment overrides:
#   PARTITION=batch   DURATION=4 (hours)
#   OPENAI_API_KEY=...   MODEL_VERSION=gpt-4o-mini (default)
#   OPENAI_API_URL=https://api.openai.com/v1/chat/completions (default)
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---------- cluster detection ----------
# This script runs on two clusters with different schedulers, so the backend is
# detected rather than hard-coded:
#   adlr  - NVIDIA ADLR/OCI, submits through the cluster-interface submit_job
#           wrapper (oci-nrt-cs-001 and friends)
#   slurm - stock Slurm, submits with sbatch (oci-hsg-cs-001, GB200/aarch64)
# Force either with CLUSTER=adlr / CLUSTER=slurm.
# Site-specific; set it to wherever your `submit_job` wrapper lives, or leave it unset
# and use --direct.
ADLR_CLUSTER_INTERFACE=${ADLR_CLUSTER_INTERFACE:-}
if [[ -z "${CLUSTER:-}" ]]; then
    if [[ -x "$ADLR_CLUSTER_INTERFACE/submit_job" ]] || command -v submit_job >/dev/null 2>&1; then
        CLUSTER=adlr
    else
        CLUSTER=slurm
    fi
fi

# ---------- cluster / project constants ----------
ACCOUNT=${ACCOUNT:-${SLURM_ACCOUNT:?set SLURM_ACCOUNT}}
DURATION=${DURATION:-4}
if [[ "$CLUSTER" == "adlr" ]]; then
    # ADLR cluster-interface tools (submit_job, etc.) on PATH.
    export PATH="$ADLR_CLUSTER_INTERFACE:$PATH"
    PARTITION=${PARTITION:-batch_block1}
else
    # oci-hsg-cs-001: GB200 nodes, 4 GPUs / 144 CPUs / 920G each. `batch`
    # caps at 4h; `batch_long` allows 7d -- set PARTITION=batch_long if you
    # raise DURATION past 4.
    PARTITION=${PARTITION:-batch}
fi
# Derived from this script's own location rather than hard-coded, so a run
# launched from a worktree writes results into that worktree instead of the
# shared central tree.
PROJECT=${PROJECT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
LMMS_EVAL_DIR=${LMMS_EVAL_DIR:-${SELFSAL_ROOT:-.}/../lmms-eval}
CONDA_SH=${CONDA_SH:-${CONDA_ROOT:?set CONDA_ROOT}/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-lmms_eval}
HF_HOME=${HF_HOME:-${HF_HOME:?set HF_HOME}}
# sbatch-only sizing. Per-GPU share of a GB200 node (144 CPUs / 920G split 4
# ways), and the QOS floor: every QOS on oci-hsg-cs-001 sets MinTRES=gres/gpu=4,
# so a job asking for fewer is rejected outright with QOSMinGRES. On ADLR
# submit_job does its own sizing, so these are unused there.
CPUS_PER_GPU=${CPUS_PER_GPU:-36}
MEM_PER_GPU_GB=${MEM_PER_GPU_GB:-220}
if [[ "$CLUSTER" == "adlr" ]]; then
    MIN_GPUS_PER_JOB=${MIN_GPUS_PER_JOB:-1}
else
    MIN_GPUS_PER_JOB=${MIN_GPUS_PER_JOB:-4}
fi

# ---------- experiment defaults ----------
MODEL="Qwen/Qwen3-VL-8B-Instruct"
MODEL_TYPE=""     # auto-detected from the model name unless given
# The Saliency-R1 paper suite (Table 2).
TASKS="mmmu_pro,mmbench_en_test,pope,mme,mmerealworld,mmstar,chartqa,illusionvqa,scienceqa_img,p3"
NUM_GPUS=1
DIRECT=0
R1_MODE=0         # --r1-mode: preset the full Saliency-R1 reasoning-eval recipe
                  # (R1 system prompt + repetition_penalty 1.05 + max_new_tokens 4096
                  # + tag "r1"). Individual flags below still override each piece.
EASE_MODE=0       # --ease-mode: --r1-mode, plus the two things that stop a
                  # reasoning model from reasoning on these benchmarks. See the
                  # block below --r1-mode for what it does and does not align.
STRIP_ANSWER_FORMAT=0  # exported as LMMS_STRIP_ANSWER_FORMAT_INSTRUCTION so the model
                       # wrapper drops "answer with the letter directly" from the prompt
MAX_NEW_TOKENS=""  # if set, forwarded as --gen_kwargs max_new_tokens=N and suffixes the output/cache dir with _mnt<N>
REPETITION_PENALTY=""  # if set, forwarded as --gen_kwargs repetition_penalty=N (paper uses 1.05 for Saliency-R1)
SYSTEM_PROMPT_FILE=""  # if set, exported as LMMS_SYSTEM_PROMPT_FILE so the model wrapper reads the system prompt from it
MAX_PIXELS=""     # if set, forwarded as --model_args max_pixels=N (raise for high-res tasks like MME-RealWorld;
                  # lmms-eval default 1605632 downsamples MME-RW's ~36MP images ~22x; native Qwen2.5-VL is 12845056)
VGA_MODE=0        # --vga: evaluate with Vision-Guided Attention (arXiv:2511.20032).
                  # Selects the qwen3_vl_vga wrapper and tags the output dir with the
                  # arm's own settings, so a guided run can never resume from -- or be
                  # confused with -- the stock run of the same checkpoint.
VGA_ARGS_CLI=""   # --vga-args "beta=0.25,start_layer=2,end_layer=18"; also VGA_ARGS in the env
TAG=""            # optional suffix for the output/cache dir, to isolate config variants (e.g. r1sys)
PRINT_OUTPUT_DIR=0  # --print-output-dir: echo the resolved results dir and exit (used by the
                    # benchmark-suite drivers to decide what has already been evaluated)
EXTRA_ARGS=""     # forwarded verbatim to lmms_eval (e.g. --limit 8)

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2";      shift 2 ;;
        --model-type) MODEL_TYPE="$2"; shift 2 ;;
        --tasks|--task) TASKS="$2";    shift 2 ;;
        --num-gpus)   NUM_GPUS="$2";   shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --repetition-penalty) REPETITION_PENALTY="$2"; shift 2 ;;
        --system-prompt-file) SYSTEM_PROMPT_FILE="$2"; shift 2 ;;
        --max-pixels) MAX_PIXELS="$2"; shift 2 ;;
        --tag)        TAG="$2";        shift 2 ;;
        --r1-mode)    R1_MODE=1;       shift ;;
        --ease-mode)  EASE_MODE=1;     shift ;;
        --vga)        VGA_MODE=1;      shift ;;
        --vga-args)   VGA_ARGS_CLI="$2"; VGA_MODE=1; shift 2 ;;
        --strip-answer-format) STRIP_ANSWER_FORMAT=1; shift ;;
        --direct)     DIRECT=1;        shift ;;
        --print-output-dir) PRINT_OUTPUT_DIR=1; shift ;;
        *)            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# --ease-mode: make the run comparable with the numbers in EASE (arXiv
# 2605.30912, Table 1) rather than with lmms-eval's own defaults.
#
# It exists because of a measurement, not a preference. Most multiple-choice
# tasks end their prompt with a sentence like "Answer with the option's letter
# from the given choices directly." That sentence is in the USER turn, so it
# beats the R1 system prompt asking for a reasoning chain. Run cv_bench,
# vstar_bench and hrbench4k under plain --r1-mode and the median response is ONE
# character, with no <think> block in any of 3,629 samples -- an RL-trained
# reasoner scoring at base-model level because the benchmark never lets it
# reason. EASE appends the opposite instruction, so their models reason on every
# item and their figures are not comparable with ours.
#
# Two changes, and it implies --r1-mode for the rest:
#   strip the instruction   LMMS_STRIP_ANSWER_FORMAT_INSTRUCTION=1; the wrapper
#                           edits the RENDERED prompt, because hrbench hardcodes
#                           its sentence and vstar-bench ships one in the dataset
#   max_pixels 4194304      EASE's setting (applied further down, next to the
#                           per-task defaults it overrides)
#
# What it does NOT align, deliberately -- these keep --r1-mode's values, and
# changing them silently would confound the comparison the flag exists to make:
#   max_new_tokens 4096     EASE trains at 1024 and never states its eval cap
#   repetition_penalty 1.05 EASE uses none (greedy, temperature 0.0, top_p 1.0)
#   system prompt           ours wraps reasoning in <think>; EASE wraps the answer
#                           in <answer> and appends it to the user turn
# Pass --max-new-tokens 1024 --repetition-penalty 1.0 to close the first two.
#
# The tag is "ease", so results land in <slug>_mnt4096_ease and can neither be
# confused with nor resumed from the _r1 runs. They are different measurements.
if [[ $EASE_MODE -eq 1 ]]; then
    R1_MODE=1
    STRIP_ANSWER_FORMAT=1
    : "${TAG:=ease}"
fi

# --vga: VGA is training-free, so the arm lives entirely in the configuration --
# there is no checkpoint whose name would distinguish it. That makes the output
# directory the only thing standing between a guided run and the stock run of the
# same weights, and this suite RESUMES from whatever it finds there: an untagged
# guided run would skip every benchmark the stock run had already finished and
# report those numbers as its own. So the tag encodes the arm, and the settings
# that define it go in the name.
VGA_TAG=""
if [[ $VGA_MODE -eq 1 ]]; then
    : "${MODEL_TYPE:=qwen3_vl_vga}"
    VGA_ARGS="${VGA_ARGS_CLI:-${VGA_ARGS:-}}"
    # beta=0.25,start_layer=2,end_layer=18  ->  vga_b0.25_l2-18
    _vga_get() { echo "$VGA_ARGS" | tr ',' '\n' | sed -n "s/^ *$1 *= *//p" | tail -1; }
    _b=$(_vga_get beta);        : "${_b:=0.2}"
    _s=$(_vga_get start_layer); : "${_s:=4}"
    _e=$(_vga_get end_layer);   : "${_e:=16}"
    _m=$(_vga_get mode)
    _rest=$(echo "$VGA_ARGS" | tr ',' '\n' \
        | grep -vE '^ *(beta|start_layer|end_layer|mode) *=' \
        | sed 's/ *//g; s/=/-/' | paste -sd_ -)
    # Held, not applied: --r1-mode and --ease-mode set TAG below, and a guided
    # r1 run has to land in a directory that says BOTH -- it is neither
    # comparable with a plain-instruct VGA run nor with the stock r1 run. The
    # slug is composed after those blocks, so the result is <mode>_vga_<config>.
    VGA_TAG="vga_b${_b}_l${_s}-${_e}${_m:+_${_m}}${_rest:+_${_rest}}"
    if [[ "$MODEL_TYPE" != *vga* ]]; then
        echo "--vga was given but --model-type is '$MODEL_TYPE', which does not apply it" >&2
        exit 1
    fi
fi

# --r1-mode: apply the Saliency-R1 reasoning-eval recipe as defaults. Uses := so
# any explicitly-passed flag (e.g. --max-new-tokens 8192) still wins.
if [[ $R1_MODE -eq 1 ]]; then
    : "${SYSTEM_PROMPT_FILE:=$PROJECT/configs/prompts/saliency_r1_system.txt}"
    : "${REPETITION_PENALTY:=1.05}"
    : "${MAX_NEW_TOKENS:=4096}"
    : "${TAG:=r1}"
fi

# Compose the VGA slug onto whatever mode tag was resolved above, so an r1-mode
# guided run lands in <slug>_mnt4096_r1_vga_b0.2_l4-16 -- next to the stock
# <slug>_mnt4096_r1 it is paired against, and distinct from a plain-instruct
# guided run, which is a different measurement entirely.
if [[ -n "$VGA_TAG" ]]; then
    TAG="${TAG:+${TAG}_}${VGA_TAG}"
fi

# Resolve the system-prompt file to an absolute path: the inner command cd's
# into $LMMS_EVAL_DIR, so a relative path would break.
if [[ -n "$SYSTEM_PROMPT_FILE" ]]; then
    if [[ ! -f "$SYSTEM_PROMPT_FILE" ]]; then
        echo "System prompt file not found: $SYSTEM_PROMPT_FILE" >&2
        exit 1
    fi
    SYSTEM_PROMPT_FILE=$(realpath "$SYSTEM_PROMPT_FILE")
fi

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

# Strip trailing slashes before deriving the slug. Shell tab-completion on a
# checkpoint directory produces `--model /path/to/ckpt/`, and `sed 's|.*/||'` is
# greedy: it eats through that final slash and yields an EMPTY slug. Results
# then land in `<results>/_mnt4096_r1` instead of the model's own directory --
# a fresh dir with no records, so the suite skips nothing and silently re-runs
# benchmarks that were already finished.
while [[ "$MODEL" == */ ]]; do
    MODEL="${MODEL%/}"
done

MODEL_SLUG=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g')

# Belt and braces: never write to a dir whose name is only the suffixes. Any
# future path shape that slugs to nothing should fail loudly, not quietly
# orphan a run's results.
if [[ -z "$MODEL_SLUG" ]]; then
    echo "Could not derive a model slug from --model '$MODEL'" >&2
    exit 1
fi

# Default max_pixels for the high-resolution benchmarks, run one task at a time.
# MME-RealWorld's images are ~36MP and the wrapper default (1.6M) downsamples
# them ~22x, so the model over-abstains ("E") and under-scores; 3.2M replicates
# the paper (58.7). The same argument applies to the three benchmarks whose
# whole subject is resolution -- HR-Bench 4K/8K (3840x2160 and 7680x4320) and
# V* (fine-grained search in large scenes) -- where 1.6M shrinks the small
# region the question is about until the benchmark measures something else.
#
# 3.2M is the tier this cluster is known to run 4-way at 4096 generated tokens,
# not a per-benchmark calibration: only MME-RealWorld's value is validated
# against a published number, and even 3.2M is ~10x below HR-Bench 8K's native
# resolution. Raise it with --max-pixels 12845056 (Qwen's native cap) to test
# that; being explicit also suffixes the results dir, so a sweep cannot
# overwrite these runs.
#
# Matched against the whole --tasks string, so it applies only when the suite
# drivers run the benchmark on its own and never silently changes the other
# tasks in a combined `--tasks a,b,c` run.
MAX_PIXELS_EXPLICIT=0
[[ -n "$MAX_PIXELS" ]] && MAX_PIXELS_EXPLICIT=1
if [[ -z "$MAX_PIXELS" ]]; then
    if [[ $EASE_MODE -eq 1 ]]; then
        # EASE's own setting, and it applies to EVERY task rather than only the
        # high-res ones: matching them on resolution is half the point of the
        # flag, and 4.19M is above the 3.2M tier anyway, so this never lowers a
        # task's pixels. Un-suffixed like the mmerealworld default, because the
        # "ease" tag already separates these runs.
        MAX_PIXELS=4194304
    else
        case "$TASKS" in
            mmerealworld|hrbench4k|hrbench8k|vstar_bench) MAX_PIXELS=3211264 ;;
        esac
    fi
fi

# Suffix the slug so runs with different generation caps get isolated result
# dirs (and, crucially, isolated response caches — see --use_cache below).
MNT_SLUG=${MAX_NEW_TOKENS:+_mnt${MAX_NEW_TOKENS}}
# Only an *explicit* --max-pixels suffixes the dir (for sweeps); the mmerealworld
# default stays un-suffixed so it lands in the plain <model_slug> dir.
PX_SLUG=""
[[ "$MAX_PIXELS_EXPLICIT" -eq 1 ]] && PX_SLUG="_px${MAX_PIXELS}"
TAG_SLUG=${TAG:+_${TAG}}
OUTPUT_DIR="$PROJECT/results/lmms_eval/${MODEL_SLUG}${MNT_SLUG}${PX_SLUG}${TAG_SLUG}"
JOB_NAME="lmms_eval_${MODEL_SLUG}${MNT_SLUG}${PX_SLUG}${TAG_SLUG}"

# Query mode: report where this exact config would write, without creating or
# running anything. Keeps the suite drivers from re-deriving the slug logic.
if [[ $PRINT_OUTPUT_DIR -eq 1 ]]; then
    echo "$OUTPUT_DIR"
    exit 0
fi
# Build --model_args (max_pixels raises input resolution for high-res tasks).
MODEL_ARGS="pretrained=$MODEL"
[[ -n "$MAX_PIXELS" ]] && MODEL_ARGS="$MODEL_ARGS,max_pixels=$MAX_PIXELS"
# Build a single --gen_kwargs from the cap + repetition penalty. Prepended to
# EXTRA_ARGS so an explicit --gen_kwargs in EXTRA_ARGS (if any) still wins by
# appearing later.
GEN_KWARGS=""
[[ -n "$MAX_NEW_TOKENS" ]] && GEN_KWARGS="max_new_tokens=$MAX_NEW_TOKENS"
if [[ -n "$REPETITION_PENALTY" ]]; then
    GEN_KWARGS="${GEN_KWARGS:+$GEN_KWARGS,}repetition_penalty=$REPETITION_PENALTY"
fi
if [[ -n "$GEN_KWARGS" ]]; then
    EXTRA_ARGS="--gen_kwargs $GEN_KWARGS $EXTRA_ARGS"
fi
LOG_ROOT="$PROJECT/outputs/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_ROOT"

echo "Model:      $MODEL ($MODEL_TYPE)"
[[ $VGA_MODE -eq 1 ]] && echo "VGA:        ${VGA_ARGS:-(defaults: beta=0.2 layers 4-16)}"
echo "Tasks:      $TASKS"
echo "GPUs:       $NUM_GPUS"
[[ -n "$MAX_NEW_TOKENS" ]] && echo "Max new tok: $MAX_NEW_TOKENS"
[[ -n "$REPETITION_PENALTY" ]] && echo "Rep penalty: $REPETITION_PENALTY"
[[ -n "$SYSTEM_PROMPT_FILE" ]] && echo "Sys prompt: $SYSTEM_PROMPT_FILE"
[[ -n "$MAX_PIXELS" ]] && echo "Max pixels: $MAX_PIXELS"
[[ $STRIP_ANSWER_FORMAT -eq 1 ]] && echo "Prompt:     answer-format instructions stripped (reasoning-mode eval)"
echo "Output dir: $OUTPUT_DIR"
OPENAI_API_URL=${OPENAI_API_URL:-https://api.openai.com/v1/chat/completions}
MODEL_VERSION=${MODEL_VERSION:-gpt-4o-mini}
echo "OpenAI key: $([[ -n "$OPENAI_API_KEY" ]] && echo "(set — mmbench uses $MODEL_VERSION via $OPENAI_API_URL)" || echo '(unset — mmbench falls back to exact match)')"
[[ -n "$EXTRA_ARGS" ]] && echo "Extra args: $EXTRA_ARGS"
echo ""

# lmms-eval data-parallel: one process per GPU via accelerate; single GPU runs
# plain python. --use_cache makes reruns/auto-resumes skip finished requests.
INNER_CMD="
    source $CONDA_SH;
    conda activate $CONDA_ENV;
    export HF_HOME=$HF_HOME;
    export HF_TOKEN=${HF_TOKEN:-};
    export OPENAI_API_KEY=${OPENAI_API_KEY:-};
    export OPENAI_API_URL=$OPENAI_API_URL;
    export MODEL_VERSION=$MODEL_VERSION;
    export LMMS_LOCAL_CACHE=/nonexistent;
    export LMMS_CACHE_EVAL_VERSION=${LMMS_CACHE_EVAL_VERSION:-cache-v1};
    export LMMS_SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-};
    export LMMS_STRIP_ANSWER_FORMAT_INSTRUCTION=$STRIP_ANSWER_FORMAT;
    export VGA_ARGS='${VGA_ARGS:-}';
    export VGA_REPO=${VGA_REPO:-$PROJECT};
    cd $LMMS_EVAL_DIR;
    if [[ $NUM_GPUS -gt 1 ]]; then
        LAUNCH=\"accelerate launch --num_processes=$NUM_GPUS -m lmms_eval\";
    else
        LAUNCH=\"python -m lmms_eval\";
    fi;
    \$LAUNCH \
        --model $MODEL_TYPE \
        --model_args $MODEL_ARGS \
        --tasks $TASKS \
        --batch_size 1 \
        --log_samples \
        --use_cache $OUTPUT_DIR/cache \
        --output_path $OUTPUT_DIR \
        $EXTRA_ARGS
"

if [[ $DIRECT -eq 1 ]]; then
    bash -c "$INNER_CMD"
    echo "Finished $JOB_NAME"
    exit 0
fi

source "$SCRIPT_DIR/slurm_job_cap.sh"
wait_for_slurm_capacity "$JOB_NAME"

# Bare node (no --image): the conda env provides the full stack; /lustre is
# natively mounted.
if [[ "$CLUSTER" == "adlr" ]]; then
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
    exit 0
fi

# ---------- stock Slurm (sbatch) ----------
#
# Requeue replaces ADLR's --autoresume_uninstrumented. This Slurm has
# JobRequeue=1, so we ask for the batch shell to be signalled 5 minutes before
# the wall limit (--signal=B:USR1@300) and requeue ourselves from the trap. The
# requeued job re-runs this same command and resumes from the sqlite response
# cache in $OUTPUT_DIR/cache, exactly as autoresume did on ADLR.
#
# The `& wait` is not cosmetic: with the command in the foreground bash defers
# the trap until it returns, which is precisely the case we need to catch.
# NUM_GPUS stays the data-parallel process count; the allocation is floored
# separately at the QOS minimum. Asking for 1 GPU is a submission error here, so
# a 1-GPU run still reserves the node's 4 -- it just leaves 3 idle.
ALLOC_GPUS=$NUM_GPUS
if [[ $ALLOC_GPUS -lt $MIN_GPUS_PER_JOB ]]; then
    ALLOC_GPUS=$MIN_GPUS_PER_JOB
    echo "Note: QOS requires >=${MIN_GPUS_PER_JOB} GPUs; allocating $ALLOC_GPUS but running $NUM_GPUS process(es)."
    echo "      Pass --num-gpus $MIN_GPUS_PER_JOB to use the whole allocation."
fi

SBATCH_SCRIPT="$LOG_ROOT/${JOB_NAME}.sbatch"
cat > "$SBATCH_SCRIPT" <<SBATCH_EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:$ALLOC_GPUS
#SBATCH --cpus-per-task=$((CPUS_PER_GPU * ALLOC_GPUS))
#SBATCH --mem=$((MEM_PER_GPU_GB * ALLOC_GPUS))G
#SBATCH --time=${DURATION}:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=$LOG_ROOT/${JOB_NAME}.%j.out

trap 'echo "[autoresume] wall limit near; requeueing \$SLURM_JOB_ID"; scontrol requeue \$SLURM_JOB_ID; exit 0' USR1

bash -c '$INNER_CMD' &
wait \$!
SBATCH_EOF

sbatch "$SBATCH_SCRIPT"

echo "Submitted $JOB_NAME (batch script: $SBATCH_SCRIPT)"
