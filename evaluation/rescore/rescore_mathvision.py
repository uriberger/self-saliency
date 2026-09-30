#!/usr/bin/env python
"""Re-score banked `mathvision_testmini` results with the current parser.

The sibling of scripts/rescore_reasoning_tasks.py, for the one benchmark that
does *not* go through the shared reasoning scorer. MathVision brings its own
`mathvision_process_results` (lmms_eval/tasks/mathvision/utils.py), and until
2026-09-17 that function could not read an answer of the form "<letter>. <option
text>" -- the exact shape a model produces when the prompt asks it for "the
option's letter from the given choices" and it names the letter and restates the
choice. "B. B" was folded to "b.b" and compared against a ground truth of "B".

The penalty was proportional to how often a checkpoint used that style, so the
column ranked models by output formatting: the EASE checkpoint used it for 125
of 304 answers and scored 18.42%, below the chance rate on the multiple-choice
half, while the plain-instruct baselines never used it and were untouched.

Responses are deterministic and banked in full, so re-scoring needs no GPU: this
replays the stored text through today's `mathvision_process_results` and rewrites

  * `<stamp>_samples_mathvision_*.jsonl`  -- per-question `scores`
  * `<stamp>_results.json`                -- the aggregate and a
                                             `mathvision_parser_version` stamp

`mathvision_process_results` needs the document's options, which a samples row
does not carry -- but the prompt it does carry was built from them, so they are
recovered from `input` (see parse_doc). A row whose prompt cannot be parsed is
reported and left alone rather than guessed at.

Dry run by default -- it prints what would move and exits without writing.

Usage:
    python scripts/rescore_mathvision.py                    # report only
    python scripts/rescore_mathvision.py --apply            # rewrite
    python scripts/rescore_mathvision.py --results-dir /path/to/results/lmms_eval

The originals are kept beside each rewritten file as `<name>.orig_buggy_parse.bak`,
matching rescore_reasoning_tasks.py. A backup is written once and never
overwritten, so re-running this can not bury the pre-fix state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# The scorer comes from the PINNED submodule, not from a checkout beside this one. A
# re-score is only meaningful against the scorer the paper's numbers were produced by.
# `evaluation/lmms_eval` is checked out at 4ea4f15 = a9a806b, the commit those numbers
# were scored under, plus one commit that only stamps a version and changes no score.
LMMS_EVAL_DIR = Path(
    os.environ.get("LMMS_EVAL_DIR", Path(__file__).resolve().parents[1] / "lmms_eval")
)
sys.path.insert(0, str(LMMS_EVAL_DIR))

from lmms_eval.tasks.mathvision.utils import (  # noqa: E402
    PARSER_VERSION,
    mathvision_aggregate_results_eval,
    mathvision_process_results,
)

# `evaluation/results` IS the lmms-eval tree here. The archive kept two trees side by side
# (`<root>/inference` and `<root>/lmms_eval`) and the port flattened them; this constant
# still had the archive's extra level, so the default glob matched a directory that does
# not exist and the script reported nothing to do. `rescore_reasoning_tasks.py` was
# updated and this one was not -- they glob the same `*/*/*_results.json` below, so they
# have to agree on the root.
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BACKUP_SUFFIX = ".orig_buggy_parse.bak"
# The metric name that marks a task as scored by mathvision's own rule-based
# scorer. The `_reason_` variant of the task is judged by an LLM instead and
# reports llm_as_judge_eval, so it is not ours to re-score.
MARKER_METRIC = "mathvision_standard_eval,none"
VERSION_KEY = "mathvision_parser_version"

# doc_to_text builds the prompt as
#   <preamble><question>\nChoices: A. <c1>\nB. <c2>...\n<mc_prompt>
# so the choices start at the first "\nChoices: " that is followed by an "A. "
# line, and each later option opens a line of its own.
_CHOICES_MARKER = "\nChoices: "
_OPTION_LINE = re.compile(r"\n(?=[A-Z][.] )")
# lmms_eval_specific_kwargs.mc_prompt, appended after the last choice. Runs
# launched with --strip-answer-format have none, hence the tolerance.
_MC_PROMPTS = (
    "Answer the question with the option's letter from the given choices directly.",
    "Answer the question with a number directly.",
)


def parse_doc(row: dict) -> dict | None:
    """Rebuild the {answer, options} a scorer needs from a banked samples row.

    `target` is doc["answer"] as written by doc_to_target. The options are read
    back out of the prompt, which is the only place they survive. Returns None
    when the prompt does not look like one doc_to_text produced, so the caller
    can skip the run loudly instead of scoring against invented choices.
    """
    prompt = row.get("input")
    if not isinstance(prompt, str):
        return None
    answer = str(row.get("target", "")).strip()

    if _CHOICES_MARKER not in prompt:
        # doc_to_text emits the marker whenever doc["options"] is non-empty, so
        # its absence means there were none. That is not only the free-form
        # questions: testmini doc 31 ("Which side is opposite to <image3>?")
        # answers "C" against options that exist solely as pixels, and the
        # scorer handles it -- gt_answer_value stays "" and only the letter is
        # compared.
        return {"answer": answer, "options": []}

    tail = prompt.split(_CHOICES_MARKER)[-1].rstrip()
    for mc_prompt in _MC_PROMPTS:
        if tail.endswith(mc_prompt):
            tail = tail[: -len(mc_prompt)].rstrip()
            break

    options = []
    for part in _OPTION_LINE.split("\n" + tail)[0:]:
        part = part.strip("\n")
        if not part:
            continue
        expected = chr(ord("A") + len(options))
        if not part.startswith(f"{expected}. "):
            return None
        options.append(part[len(expected) + 2 :])
    if not options:
        return None
    # doc["options"][ord(answer) - ord("A")] has to be in range, or the scorer
    # raises rather than mis-scoring.
    if len(answer) == 1 and answer.isalpha() and not (0 <= ord(answer) - ord("A") < len(options)):
        return None
    return {"answer": answer, "options": options}


def response_of(row: dict) -> str:
    """The text the scorer saw: post-reasoning-tag-stripping, as generated.

    Unlike the reasoning tasks, mathvision's scorer runs on whatever
    process_results is handed, which evaluator.py has already stripped. Prefer
    the banked `mathvision_standard_eval.response`, which is literally that
    argument, and fall back to `filtered_resps` for older files.
    """
    stored = (row.get("mathvision_standard_eval") or {}).get("response")
    value = stored if stored is not None else row.get("filtered_resps")
    while isinstance(value, list):
        value = value[0] if value else ""
    return str(value)


def rescore_task(samples: Path) -> tuple[list[dict], float, int, int] | None:
    """Replay one samples file.

    Returns (rows with `scores` replaced, aggregate, n_gained, n_lost), or None
    when a row's prompt could not be parsed back into a document.
    """
    rows = [json.loads(line) for line in samples.open() if line.strip()]
    gained = lost = 0
    scored = []
    for row in rows:
        doc = parse_doc(row)
        if doc is None:
            return None
        before = bool((row.get("mathvision_standard_eval") or {}).get("scores", [False])[0])
        result = mathvision_process_results(doc, [response_of(row)])[MARKER_METRIC.split(",")[0]]
        after = bool(result["scores"][0])
        gained += after and not before
        lost += before and not after
        row["mathvision_standard_eval"] = result
        scored.append(result)
    return rows, mathvision_aggregate_results_eval(scored), gained, lost


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


def samples_path(results_path: Path, task: str) -> Path:
    base = str(results_path)[: -len("_results.json")]
    return Path(f"{base}_samples_{task}.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--task", action="append", default=None,
                        help="only this task (repeatable); default = all of them. "
                             "`mathvision_testmini` is the benchmark the results "
                             "table reports; `mathvision_testmini_mini` is the "
                             "100-doc training-curve eval, which the table drops "
                             "but report_bench_evals.sh plots")
    parser.add_argument("--apply", action="store_true",
                        help="rewrite the files; without it nothing is written")
    args = parser.parse_args()

    changed = rescored = skipped = total_lost = 0

    print(f"parser version: {PARSER_VERSION}")
    print(f"{'run / task':70s} {'stored':>8s} {'rescored':>9s} {'delta':>7s} {'+/-':>9s}")
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

            replayed = rescore_task(samples)
            if replayed is None:
                print(f"  SKIP {slug}/{task}: prompt in {samples.name} is not one "
                      f"doc_to_text produced -- options could not be recovered")
                skipped += 1
                continue
            rows, after, gained, lost = replayed
            before = float(task_results[MARKER_METRIC])
            rescored += 1
            total_lost += lost
            moved = abs(after - before) > 0.005  # the aggregate is rounded to 2dp
            changed += moved
            flag = "  <-- LOST" if lost else ""
            print(f"{(slug + ' / ' + task)[:70]:70s} {before:8.2f} {after:9.2f} "
                  f"{after - before:+7.2f} {f'+{gained}/-{lost}':>9s}{flag}")

            if args.apply:
                back_up(samples, args.apply)
                write_jsonl(samples, rows)
                task_results[MARKER_METRIC] = after
                touched = True

        if touched:
            back_up(results_path, args.apply)
            results[VERSION_KEY] = PARSER_VERSION
            results_path.write_text(json.dumps(results, indent=2, default=str))

    verb = "rewrote" if args.apply else "would rewrite"
    print(f"\n{rescored} task-entries re-scored, {changed} moved, {skipped} skipped; "
          f"{verb} the ones listed above.")
    if total_lost:
        print(f"WARNING: {total_lost} answer(s) went correct -> wrong. The fix only "
              f"adds a reading, so this should be 0 -- check before applying.")
    if not args.apply:
        print("dry run -- pass --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
