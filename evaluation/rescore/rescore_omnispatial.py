#!/usr/bin/env python3
"""Re-score cached OmniSpatial lmms-eval runs with the fixed answer parser.

The original scorer accepted only a literal ``Answer: X`` line and, finding
nothing, *predicted "A"*. Reasoning checkpoints (cold-start / GRPO) answer as
``<think>...</think> D`` and never emit that line, so every sample was graded as
"A" and the score collapsed to the base rate of gold-A items -- identical across
checkpoints because it no longer depended on the model at all.

``answer_parsing.extract_answer_letter`` now falls back to a bare letter and then
to the last option letter mentioned, and returns None (scored incorrect) when
there is genuinely no answer. This script replays every cached
``*_samples_omnispatial_test.jsonl`` through that parser, rewrites the per-sample
``is_correct`` / ``pred_letter``, and updates the omnispatial metrics in the
sibling ``*_results.json``.

Originals are backed up next to each file with a ``.parse_bak`` suffix.
Run from the repo root:  python scripts/rescore_omnispatial.py [--dry-run]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# `evaluation/results` IS the lmms-eval tree here. The archive kept `<root>/inference`
# and `<root>/lmms_eval` side by side and the port flattened them, so the extra level
# below made the default glob match nothing and the script report no work to do.
LMMS_EVAL_DIR = REPO / "results"
# The parser comes from the PINNED submodule, not from a checkout beside this one. A
# re-score is only meaningful against the scorer the paper's numbers were produced by.
# `evaluation/lmms_eval` is checked out at 4ea4f15 = a9a806b, the commit those numbers
# were scored under, plus one commit that only stamps a version and changes no score.
# The LMMS_EVAL_DIR environment variable overrides it; note that it means the CHECKOUT,
# not the constant of the same name above.
LMMS_EVAL_SRC = Path(os.environ.get("LMMS_EVAL_DIR", REPO / "lmms_eval"))
PARSER_PATH = LMMS_EVAL_SRC / "lmms_eval/tasks/omnispatial/answer_parsing.py"

# Load the deployed parser directly from the lmms-eval checkout so this script
# and the live scorer can never drift apart. answer_parsing.py is import-safe
# (no dataset download) precisely so this works.
_spec = importlib.util.spec_from_file_location("omnispatial_answer_parsing", PARSER_PATH)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

TASK_KEY = "omnispatial_test"
SUFFIX = "_samples_omnispatial_test.jsonl"


def rescore_file(samples_path: Path, dry_run: bool) -> dict:
    samples = [json.loads(l) for l in samples_path.read_text().splitlines() if l.strip()]

    per_subtask: dict[str, list[int]] = defaultdict(list)
    old_correct = new_correct = flips = unparsed = 0

    for s in samples:
        # Every sample carries the same submission dict under "omnispatial" and
        # under "omnispatial_<sub_task>"; keep both in sync.
        blocks = [v for k, v in s.items() if k == "omnispatial" or k.startswith("omnispatial_")]
        if not blocks:
            continue
        sub = blocks[0]
        letter = P.extract_answer_letter(sub["pred"])
        flag = letter == sub["gt_content"]

        old_correct += int(bool(sub["is_correct"]))
        new_correct += int(flag)
        flips += int(bool(sub["is_correct"]) != flag)
        unparsed += int(letter is None)
        per_subtask[sub["sub_task"]].append(int(flag))

        for b in blocks:
            b["pred_letter"] = letter
            b["is_correct"] = flag

    n = len(samples)
    new_block = {"omnispatial,none": new_correct / n if n else 0.0}
    for sub_task, scores in per_subtask.items():
        new_block[f"{sub_task},none"] = sum(scores) / len(scores)

    summary = {
        "n": n,
        "old_acc": old_correct / n if n else 0.0,
        "new_acc": new_block["omnispatial,none"],
        "flips": flips,
        "unparsed": unparsed,
    }
    if dry_run:
        return summary

    shutil.copy2(samples_path, samples_path.with_suffix(samples_path.suffix + ".parse_bak"))
    with samples_path.open("w") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")

    ts = samples_path.name[: -len(SUFFIX)]
    results_path = samples_path.parent / f"{ts}_results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        block = results.get("results", {}).get(TASK_KEY)
        if block is not None:
            shutil.copy2(results_path, results_path.with_suffix(".json.parse_bak"))
            for k, v in new_block.items():
                if k in block:  # only overwrite metrics the run actually reported
                    block[k] = v
            results_path.write_text(json.dumps(results, indent=2))
            summary["results_json"] = results_path.name
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="report old->new without writing")
    args = ap.parse_args()

    files = sorted(LMMS_EVAL_DIR.glob(f"*/*/*{SUFFIX}"))
    print(f"Found {len(files)} cached omnispatial run(s). Parser: {PARSER_PATH}\n")
    for f in files:
        run = f.relative_to(LMMS_EVAL_DIR).parts[0]
        s = rescore_file(f, args.dry_run)
        print(
            f"{run:72s} n={s['n']:5d}  acc {s['old_acc']:.4f} -> {s['new_acc']:.4f}"
            f"  flips={s['flips']:4d} unparsed={s['unparsed']:4d}"
            f"  ({'DRY' if args.dry_run else s.get('results_json', 'no results.json')})"
        )


if __name__ == "__main__":
    main()
