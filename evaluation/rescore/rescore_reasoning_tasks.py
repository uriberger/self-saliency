#!/usr/bin/env python
"""Re-score banked lmms-eval `*_reasoning` results with the current parser.

Every task built on `lmms_eval/tasks/_task_utils/reasoning_utils.py` scores a
free-text response by extracting an answer from it. When that extraction
changes, every number already on disk becomes a number produced by code that no
longer exists -- and nothing in a results.json says which build wrote it. On
2026-09-10 that put two LogicVista runs of the same checkpoint in one table at
0.45% and 56.2%.

Responses are deterministic and banked in full (each samples row keeps the raw
text under `resps`), so re-scoring needs no GPU: this replays the stored text
through today's `compute_score` and rewrites

  * `<stamp>_samples_<task>.jsonl`  -- per-question `acc_score` / `format_score`
  * `<stamp>_results.json`          -- the aggregates, their stderrs, and a
                                      `reasoning_parser_version` stamp

Only tasks whose results.json carries `acc_score,none` are touched: that metric
name is what identifies the shared reasoning scorer. Anything else in the tree
is scored by its own utils and is left alone.

Dry run by default -- it prints what would move and exits without writing.

Usage:
    python scripts/rescore_reasoning_tasks.py                    # report only
    python scripts/rescore_reasoning_tasks.py --apply            # rewrite
    python scripts/rescore_reasoning_tasks.py --task logicvista_reasoning
    python scripts/rescore_reasoning_tasks.py --results-dir /path/to/results/lmms_eval

The originals are kept beside each rewritten file as `<name>.orig_buggy_parse.bak`,
following what the earlier mmmu_pro re-score left behind. A backup is written
once and never overwritten, so re-running this can not bury the pre-fix state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

LMMS_EVAL_DIR = Path(os.environ.get(
    "LMMS_EVAL_DIR", Path(__file__).resolve().parents[1] / "lmms_eval"))
sys.path.insert(0, str(LMMS_EVAL_DIR))

from lmms_eval.api.reasoning import (  # noqa: E402
    parse_reasoning_tags_config,
    strip_reasoning_tags,
)
from lmms_eval.tasks._task_utils.reasoning_utils import (  # noqa: E402
    PARSER_VERSION,
    compute_score,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BACKUP_SUFFIX = ".orig_buggy_parse.bak"
# The metric name that marks a task as scored by the shared reasoning scorer.
MARKER_METRIC = "acc_score,none"
# lmms-eval's own CLI default, and what every run in this tree was launched
# with. results.json does not record the flag, so a run that overrode it would
# need the override passed here too.
DEFAULT_REASONING_TAGS = '[["<think>", "</think>"], ["<analysis>", "</analysis>"]]'
# Below this an aggregate is "unchanged": results.json carries full precision,
# so anything real moves further than float noise.
EPS = 1e-9


def response_of(row: dict) -> str:
    """The raw model output for one sample.

    `resps` is the text as generated; `filtered_resps` is what the scorer saw
    after reasoning-tag stripping. Prefer the raw one and re-strip, so the
    result does not depend on which stripping ran at generation time. Runs
    whose model emitted no tags have no `resps` key at all -- there the two are
    the same text.
    """
    value = row.get("resps")
    if value is None:
        value = row.get("filtered_resps")
    while isinstance(value, list):
        value = value[0] if value else ""
    return str(value)


def stderr_of(values: list[float]) -> float:
    """Sample stderr of the mean, matching lmms-eval's `*_stderr,none`."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def samples_path(results_path: Path, task: str) -> Path:
    base = str(results_path)[: -len("_results.json")]
    return Path(f"{base}_samples_{task}.jsonl")


def rescore_task(samples: Path, task: str, tags) -> tuple[list[dict], dict]:
    """Replay one samples file. Returns (rows with metrics replaced, aggregates)."""
    rows = [json.loads(line) for line in samples.open() if line.strip()]
    accs: list[float] = []
    fmts: list[float] = []
    for row in rows:
        segment = strip_reasoning_tags(response_of(row), tags)
        scored = compute_score(
            data_source=task,
            solution_str=segment.strip(),
            ground_truth=str(row.get("target", "")),
            extra_info={"question": row.get("input", "")},
        )
        accs.append(float(scored["acc_score"]))
        fmts.append(float(scored["format_reward_score"]))
        row["acc_score"] = accs[-1]
        row["format_score"] = fmts[-1]
    aggregates = {
        "acc_score,none": statistics.fmean(accs) if accs else 0.0,
        "acc_score_stderr,none": stderr_of(accs),
        "acc_score_stderr_clt,none": stderr_of(accs),
        "format_score,none": statistics.fmean(fmts) if fmts else 0.0,
        "format_score_stderr,none": stderr_of(fmts),
        "format_score_stderr_clt,none": stderr_of(fmts),
    }
    return rows, aggregates


def back_up(path: Path, apply: bool) -> None:
    """Keep the pre-fix file once. Never overwrite an existing backup."""
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if backup.exists() or not apply:
        return
    backup.write_bytes(path.read_bytes())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--task", action="append", default=None,
                        help="only this task (repeatable); default = all reasoning tasks")
    parser.add_argument("--apply", action="store_true",
                        help="rewrite the files; without it nothing is written")
    parser.add_argument("--reasoning-tags", default=DEFAULT_REASONING_TAGS,
                        help="JSON pairs stripped before scoring, or 'none'")
    args = parser.parse_args()

    tags = parse_reasoning_tags_config(cli_value=args.reasoning_tags)
    changed = rescored = skipped = 0

    print(f"parser version: {PARSER_VERSION}")
    print(f"{'run / task':72s} {'stored':>8s} {'rescored':>9s} {'delta':>8s}")
    for results_path in sorted(args.results_dir.glob("*/*/*_results.json")):
        if results_path.relative_to(args.results_dir).parts[1] == "submissions":
            continue
        if results_path.name.endswith(BACKUP_SUFFIX):
            continue
        try:
            results = json.loads(results_path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(results, dict):
            continue

        slug = results_path.relative_to(args.results_dir).parts[0]
        touched = False
        for task, task_results in (results.get("results") or {}).items():
            if not isinstance(task_results, dict) or MARKER_METRIC not in task_results:
                continue
            if args.task and task not in args.task:
                continue
            samples = samples_path(results_path, task)
            if not samples.exists():
                print(f"  SKIP {slug}/{task}: no samples file beside {results_path.name}")
                skipped += 1
                continue

            rows, aggregates = rescore_task(samples, task, tags)
            before = float(task_results[MARKER_METRIC])
            after = aggregates[MARKER_METRIC]
            rescored += 1
            moved = abs(after - before) > EPS
            changed += moved
            flag = "  <-- changed" if moved else ""
            print(f"{(slug + ' / ' + task)[:72]:72s} {before:8.4f} {after:9.4f} "
                  f"{after - before:+8.4f}{flag}")

            if args.apply:
                back_up(samples, args.apply)
                write_jsonl(samples, rows)
                task_results.update(aggregates)
                touched = True

        if touched:
            back_up(results_path, args.apply)
            results["reasoning_parser_version"] = PARSER_VERSION
            results_path.write_text(json.dumps(results, indent=2, default=str))

    verb = "rewrote" if args.apply else "would rewrite"
    print(f"\n{rescored} task-entries re-scored, {changed} moved, {skipped} skipped "
          f"(no samples file); {verb} the ones listed above.")
    if not args.apply:
        print("dry run -- pass --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
