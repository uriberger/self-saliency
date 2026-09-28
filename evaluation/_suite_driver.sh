#!/bin/bash
# Shared driver for the per-suite benchmark scripts
# (eval_saliency_r1_benchmarks.sh, eval_our_benchmarks.sh).
#
# Not meant to be run directly. A suite script sources it, sets SUITE_NAME and
# BENCHMARKS, and calls run_benchmark_suite "$@":
#
#   SUITE_NAME="saliency-r1 paper suite"
#   BENCHMARKS=(chartqa mme mmstar ...)
#   source "$(dirname "${BASH_SOURCE[0]}")/eval_benchmark_suite.sh"
#   run_benchmark_suite "$@"
#
# Each benchmark is launched as its own lmms-eval invocation, in list order, so
# results for finished benchmarks are on disk while the rest are still running.
# A benchmark whose per-sample log already exists in the run's output dir is
# skipped, which makes re-running the script a resume.
set -uo pipefail

SUITE_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LAUNCHER="$SUITE_SCRIPT_DIR/slurm/launch_lmms_eval_job.sh"

# Flags every benchmark in every suite runs with.
#
# SUITE_NUM_GPUS is how many accelerate processes each lmms-eval invocation
# spawns -- not how big an allocation it gets -- so it has to match the GPUs the
# suite is actually running on, or the processes oversubscribe them. When the
# suite is submitted by slurm/launch_eval_suite_job.sh, that script exports it to
# match the allocation it asked for; running a driver by hand keeps the 4 this
# was hardcoded to before.
#
# --r1-mode is a DEFAULT here, not a fixture: SUITE_R1_MODE=0 (or --no-r1-mode)
# drops it, which evaluates the plain instruct recipe -- no R1 system prompt, no
# repetition_penalty 1.05. That also moves where the run lands, since the "r1"
# tag is what suffixes the output dir: <slug>_mnt4096_r1 with it,
# <slug>_mnt4096 without. So the two modes resume independently, and a benchmark
# already finished in one is still not-done in the other.
SUITE_R1_MODE=${SUITE_R1_MODE:-1}
COMMON_LAUNCH_ARGS=(--max-new-tokens 4096 --num-gpus "${SUITE_NUM_GPUS:-4}" --direct)

# Appended after argument parsing, so --no-r1-mode can still turn it off. Every
# consumer of COMMON_LAUNCH_ARGS (resolve_output_dir, the run loop) is reached
# from run_benchmark_suite after that point.
_apply_r1_mode() {
    [[ "$SUITE_R1_MODE" == 1 ]] && COMMON_LAUNCH_ARGS+=(--r1-mode)
    return 0
}

usage() {
    cat <<EOF
Usage: bash $0 --model <MODEL-PATH> [options] [-- <extra launcher args>]

Options:
  --model <path>   model or checkpoint to evaluate (required)
  --skip <task>    drop a benchmark from this run; repeatable, and accepts a
                   comma-separated list (--skip mme,pope). A name that is not in
                   this suite is reported and otherwise ignored, so the same
                   --skip can be passed to a multi-suite run.
  --force          re-run benchmarks that already have results
  --no-r1-mode     evaluate WITHOUT the Saliency-R1 recipe (no R1 system prompt,
                   no repetition_penalty). Results land in <slug>_mnt4096
                   instead of <slug>_mnt4096_r1, so this is a separate run with
                   its own resume state, not a re-do of the r1 one.
  --dry-run        print what would run, then exit
  --ease-mode      forwarded to the launcher: strip each task's "answer with the
                   letter directly" instruction so the model actually reasons,
                   and raise max_pixels to EASE's 4194304. Results land in a
                   separate <slug>_mnt4096_ease dir, so a suite run in this mode
                   neither resumes from nor overwrites the _r1 numbers.
  -h, --help       show this message

Anything after -- (or any unrecognised flag) is forwarded verbatim to
scripts/slurm/launch_lmms_eval_job.sh, e.g. --limit 8 for a smoke test.

Environment:
  SUITE_NUM_GPUS   accelerate processes per benchmark (default 4). Set it to the
                   number of GPUs this shell actually holds; more than that
                   oversubscribes them. slurm/launch_eval_suite_job.sh sets it
                   from its own --num-gpus.
  SUITE_R1_MODE    1 (default) runs --r1-mode; 0 is the same as --no-r1-mode.

Benchmarks run one job at a time, in the order listed in this suite. Re-running
the script skips benchmarks that already produced per-sample logs, so it is safe
to interrupt and resume.
EOF
}

# Is $1 among the remaining arguments? Written to tolerate an empty tail, since
# every caller can legitimately pass no candidates (no --skip was given).
_suite_contains() {
    local needle="$1"; shift
    local candidate
    for candidate in "$@"; do
        [[ "$candidate" == "$needle" ]] && return 0
    done
    return 1
}

# Path of the results dir the launcher would use for this model/config. Asking
# the launcher avoids duplicating its slug rules (mnt/px/tag suffixes) here.
resolve_output_dir() {
    local task="$1"; shift
    bash "$LAUNCHER" --model "$MODEL" --tasks "$task" \
        "${COMMON_LAUNCH_ARGS[@]}" "$@" --print-output-dir
}

# A benchmark counts as done once lmms-eval has written its per-sample log
# (<timestamp>_samples_<task>.jsonl); that file is only produced after the task
# finishes evaluating, so a killed mid-task run is correctly seen as not-done.
#
# The per-sample logs are gitignored and ~1.3GB, so they are NOT transferred
# between clusters -- only the aggregated *_results.json is committed. On a
# fresh checkout the samples check therefore reports every benchmark as
# not-done and the suite re-runs work that already finished elsewhere. Fall
# back to the results.json record, which is the canonical "this ran" evidence
# (same rule as scripts/lmms_eval_finished_tasks.py, reused here rather than
# reimplemented).
benchmark_is_done() {
    local output_dir="$1" task="$2"
    [[ -d "$output_dir" ]] || return 1
    local hit
    hit=$(find "$output_dir" -name "*_samples_${task}.jsonl" -size +0c -print -quit 2>/dev/null)
    [[ -n "$hit" ]] && return 0
    python3 "$SUITE_SCRIPT_DIR/lmms_eval_finished_tasks.py" "$output_dir" "$task" 2>/dev/null
}

run_benchmark_suite() {
    local MODEL="" FORCE=0 DRY_RUN=0
    local -a EXTRA_ARGS=() SKIP=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)     MODEL="$2"; shift 2 ;;
            # Consumed here rather than falling through to the catch-all: every
            # unrecognised flag is forwarded to launch_lmms_eval_job.sh, which
            # forwards what IT does not know to lmms-eval, so a --skip that got
            # that far would fail every benchmark on an unknown argument.
            --skip)      [[ $# -ge 2 ]] || { echo "error: --skip needs a benchmark name" >&2; return 2; }
                         SKIP+=("$2"); shift 2 ;;
            --skip=*)    SKIP+=("${1#*=}"); shift ;;
            --force)     FORCE=1;    shift ;;
            # Consumed here, like --skip: forwarded on, launch_lmms_eval_job.sh
            # would not recognise it and would hand it to lmms-eval, which fails
            # every benchmark on an unknown argument.
            --no-r1-mode) SUITE_R1_MODE=0; shift ;;
            --dry-run)   DRY_RUN=1;  shift ;;
            -h|--help)   usage; return 0 ;;
            --)          shift; EXTRA_ARGS+=("$@"); break ;;
            *)           EXTRA_ARGS+=("$1"); shift ;;
        esac
    done

    _apply_r1_mode

    if [[ -z "$MODEL" ]]; then
        echo "error: --model is required" >&2
        usage >&2
        return 2
    fi

    # ---------- apply --skip ----------
    # Split on commas so --skip mme,pope works as well as two --skip flags.
    local -a skip_names=()
    local raw
    for raw in ${SKIP[@]+"${SKIP[@]}"}; do
        local -a parts=()
        IFS=',' read -r -a parts <<< "$raw"
        skip_names+=(${parts[@]+"${parts[@]}"})
    done

    # A name that matches nothing is REPORTED but not fatal. It has to be
    # tolerated: `launch_eval_suite_job.sh --suite both` hands the same --skip to
    # both drivers, so skipping one suite's benchmark is always unknown to the
    # other. Saying so out loud is what keeps a typo from silently running the
    # full suite anyway.
    local -a run_tasks=() skipped_tasks=()
    local task name
    for task in "${BENCHMARKS[@]}"; do
        if _suite_contains "$task" ${skip_names[@]+"${skip_names[@]}"}; then
            skipped_tasks+=("$task")
        else
            run_tasks+=("$task")
        fi
    done
    for name in ${skip_names[@]+"${skip_names[@]}"}; do
        _suite_contains "$name" "${BENCHMARKS[@]}" \
            || echo "note: --skip $name is not a benchmark in this suite — ignored here"
    done

    if [[ ${#run_tasks[@]} -eq 0 ]]; then
        echo "Suite:      $SUITE_NAME"
        echo "Nothing to run: --skip excluded all ${#BENCHMARKS[@]} benchmarks."
        return 0
    fi

    local output_dir
    output_dir=$(resolve_output_dir "${run_tasks[0]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}") || return 1

    echo "Suite:      $SUITE_NAME (${#run_tasks[@]} of ${#BENCHMARKS[@]} benchmarks)"
    echo "Model:      $MODEL"
    echo "Mode:       $([[ "$SUITE_R1_MODE" == 1 ]] && echo "r1 (Saliency-R1 system prompt + repetition_penalty 1.05)" || echo "plain instruct (--no-r1-mode)")"
    echo "Output dir: $output_dir"
    echo "Order:      ${run_tasks[*]}"
    [[ ${#skipped_tasks[@]} -gt 0 ]] && echo "Skipped:    ${skipped_tasks[*]} (--skip)"
    echo ""

    local -a done_tasks=() ok_tasks=() failed_tasks=()
    local rc
    for task in "${run_tasks[@]}"; do
        # Re-resolved per task: the launcher picks task-specific defaults (e.g.
        # max_pixels for mmerealworld) that can change where results land.
        output_dir=$(resolve_output_dir "$task" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}") || return 1

        if [[ $FORCE -eq 0 ]] && benchmark_is_done "$output_dir" "$task"; then
            echo "[skip] $task — results already in $output_dir"
            done_tasks+=("$task")
            continue
        fi

        echo ""
        echo "=============================================================="
        echo "[run ] $task"
        echo "=============================================================="
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "bash $LAUNCHER --model $MODEL --tasks $task ${COMMON_LAUNCH_ARGS[*]} ${EXTRA_ARGS[*]+${EXTRA_ARGS[*]}}"
            continue
        fi

        bash "$LAUNCHER" --model "$MODEL" --tasks "$task" \
            "${COMMON_LAUNCH_ARGS[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        rc=$?
        if [[ $rc -eq 0 ]]; then
            ok_tasks+=("$task")
            echo "[done] $task"
        else
            # Keep going: a benchmark that OOMs or lacks its dataset should not
            # block the ones after it. Re-running the script retries it.
            failed_tasks+=("$task")
            echo "[FAIL] $task (exit $rc) — continuing with the next benchmark" >&2
        fi
    done

    echo ""
    echo "=============================================================="
    echo "Suite summary: $SUITE_NAME"
    echo "  already done: ${#done_tasks[@]} ${done_tasks[*]+(${done_tasks[*]})}"
    echo "  completed now: ${#ok_tasks[@]} ${ok_tasks[*]+(${ok_tasks[*]})}"
    echo "  failed: ${#failed_tasks[@]} ${failed_tasks[*]+(${failed_tasks[*]})}"
    [[ ${#skipped_tasks[@]} -gt 0 ]] && \
        echo "  excluded by --skip: ${#skipped_tasks[@]} (${skipped_tasks[*]})"
    echo "=============================================================="

    [[ ${#failed_tasks[@]} -eq 0 ]]
}
