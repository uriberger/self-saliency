#!/usr/bin/env python3
"""Print the lmms-eval tasks that have FINISHED under an output directory.

A task is considered finished when a `*_results.json` written by lmms-eval
contains it as a key in the top-level ``results`` dict. lmms-eval only writes
that file when an invocation completes and scores, so — because
run_lmms_eval_suite.sh runs one task per invocation — the presence of the task
in any results.json is a durable "this benchmark is done" record that survives
job restarts.

Usage:
    lmms_eval_finished_tasks.py <output_dir>          # print finished tasks
    lmms_eval_finished_tasks.py <output_dir> <task>   # exit 0 if <task> finished
"""
import glob
import json
import os
import sys


def finished_tasks(output_dir):
    done = set()
    pattern = os.path.join(output_dir, "**", "*_results.json")
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue  # unreadable / half-written file — treat as not-done
        # submissions/<task>_results.json (mathverse) matches the glob but is a
        # list of per-question predictions, not a run summary.
        if not isinstance(data, dict):
            continue
        done.update(data.get("results", {}).keys())
    return done


def main(argv):
    if len(argv) < 2:
        sys.exit(f"usage: {argv[0]} <output_dir> [task]")
    done = finished_tasks(argv[1])
    if len(argv) >= 3:
        sys.exit(0 if argv[2] in done else 1)
    for task in sorted(done):
        print(task)


if __name__ == "__main__":
    main(sys.argv)
