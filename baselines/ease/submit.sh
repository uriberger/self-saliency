#!/usr/bin/env bash
# Ask SLURM for a node and run one EASE arm on it. Same `_job.sh` split as
# launch_sink_location_job.sh: this builds a runner and hands it to submit_job;
# launch_ease_train.sh is what runs inside the allocation.
#
#   NVIDIA_API_KEY=... bash launch_ease_train_job.sh --arm ease --exp ease_8k
#   NVIDIA_API_KEY=... bash launch_ease_train_job.sh --arm dapo --exp dapo_8k
#
# Both arms, always -- see the header of launch_ease_train.sh for why a lone
# EASE number cannot be compared against our overlap runs.
#
# DURATION. 126 steps at rollout_batch 128 will not finish in one allocation.
# EasyR1 checkpoints every --save-freq steps and its config sets
# find_last_checkpoint: true, so resubmitting the same --exp resumes from the
# last checkpoint under outputs/ease/<exp>/checkpoints. Resubmit; do not
# restart from scratch, and do not raise --duration past what the queue grants.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

ARM=ease
EXP=""
DURATION=4
GPUS=8
PARTITION_OVERRIDE=""
DRY_RUN=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm)       ARM="$2";                shift 2 ;;
        --exp)       EXP="$2";                shift 2 ;;
        --duration)  DURATION="$2";           shift 2 ;;
        --gpus)      GPUS="$2";               shift 2 ;;
        --partition) PARTITION_OVERRIDE="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=1;               shift ;;
        --)          shift; EXTRA+=("$@"); break ;;
        -h|--help)   sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)           EXTRA+=("$1");           shift ;;
    esac
done

case "$ARM" in ease|dapo) ;; *) echo "ERROR: --arm must be ease or dapo." >&2; exit 2 ;; esac
EXP=${EXP:-"${ARM}_8k"}
NAME="ease-$EXP"

# shellcheck source=/dev/null
source "$REPO/cluster_env.sh"
PARTITION=${PARTITION_OVERRIDE:-${PARTITION:-$(SR1_JOB_HOURS=$DURATION sr1_pick_partition)}}
ACCOUNT=${ACCOUNT:-nvr_israel_rlop}

sr1_find_submit_job || [[ $DRY_RUN -eq 1 ]] || {
    echo "ERROR: submit_job not found under the cluster-interface paths." >&2; exit 1; }

LOG_ROOT="$REPO/outputs/logs"
mkdir -p "$LOG_ROOT" "$REPO/outputs/ease/$EXP"

RUNNER="$LOG_ROOT/$NAME.runner.sh"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    printf 'cd %q\n' "$REPO"
    echo "export HF_HOME=${HF_HOME:-/home/uberger/scratch/cache/hf_cache}"
    echo "export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
    # Every judge setting has to be written into the runner explicitly. submit_job
    # does not carry the submitting shell's environment into the allocation, so a
    # variable that is merely exported here is silently dropped -- and the failure
    # is quiet: a missing key makes every judged row fall back to its rule score,
    # which on flickr30k means 0.
    [[ -n "${NVIDIA_API_KEY:-}" ]] && printf 'export NVIDIA_API_KEY=%q\n' "$NVIDIA_API_KEY"
    [[ -n "${OPENAI_API_KEY:-}"  ]] && printf 'export OPENAI_API_KEY=%q\n' "$OPENAI_API_KEY"
    [[ -n "${OPENAI_BASE_URL:-}" ]] && printf 'export OPENAI_BASE_URL=%q\n' "$OPENAI_BASE_URL"
    [[ -n "${JUDGE_MODEL:-}"       ]] && printf 'export JUDGE_MODEL=%q\n' "$JUDGE_MODEL"
    [[ -n "${JUDGE_MAX_WORKERS:-}" ]] && printf 'export JUDGE_MAX_WORKERS=%q\n' "$JUDGE_MAX_WORKERS"
    printf 'bash launch_ease_train.sh --arm %q --exp %q --gpus %q' "$ARM" "$EXP" "$GPUS"
    for a in ${EXTRA[@]+"${EXTRA[@]}"}; do printf ' %q' "$a"; done
    echo
} > "$RUNNER"
chmod +x "$RUNNER"

echo "=========================================================================="
echo "Job    : $NAME   ($ACCOUNT, $PARTITION, ${DURATION}h, ${GPUS} GPU)"
echo "Arm    : $ARM"
echo "Judge  : $([[ -n "${NVIDIA_API_KEY:-}${OPENAI_API_KEY:-}" ]] && echo 'key set' || echo 'NO KEY -- judged rows fall back to the rule score')"
echo "Runner : $RUNNER"
echo "=========================================================================="
cat "$RUNNER"
echo "=========================================================================="

[[ $DRY_RUN -eq 1 ]] && { echo "[dry-run] not submitting."; exit 0; }

submit_job \
    --account "$ACCOUNT" \
    --partition "$PARTITION" \
    --name "$NAME" \
    --gpu "$GPUS" \
    --duration "$DURATION" \
    --outfile "$LOG_ROOT/$NAME.%j.out" \
    --logroot "$LOG_ROOT" \
    -c "bash $RUNNER"
