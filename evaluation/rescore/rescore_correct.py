"""Recompute the per-sample ``correct`` field of existing result JSONL files.

Older results were scored with a raw bidirectional substring match, which
false-positives on single-letter (multiple-choice) ground truths: any verbose
answer that merely *mentions* the GT letter (e.g. "None of the options A, B, C,
or D...") counted as correct. run_experiment.is_correct now extracts the option
the model actually committed to for single-letter GTs (see extract_mcq_choice).

This script re-applies the current is_correct to every scoreable record and
rewrites the ``correct`` field in place (atomic temp-file swap). Only files whose
scoring actually changes are rewritten. Non-letter (open-ended) ground truths are
unaffected — their scoring is identical to before.

Usage:
  python scripts/rescore_correct.py            # dry-run: report changes only
  python scripts/rescore_correct.py --apply     # rewrite changed files in place
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_experiment import extract_mcq_choice, is_correct, parse_mcq_options  # noqa: E402


def _rescore(final_answer: str, ground_truth, old: bool, question: str | None = None) -> bool:
    """Conservative re-score of one record's ``correct`` field.

    Only flips a value when we can judge it confidently, so that files whose
    answers are option *text* rather than letters (and whose option lists are not
    recoverable from the question) are not silently turned into a new kind of
    wrong:

      * open-ended GT (no MCQ options in the question) -> is_correct substring;
      * MCQ GT (lone letter, or text matching a parsed option) and the answer
        resolves to a letter -> letter comparison via is_correct(question);
      * single-letter GT with no option list and no letter commitment (ambiguous
        / stated as option text we cannot map) -> keep the old value.
    """
    fa = final_answer or ""
    g = "" if ground_truth is None else str(ground_truth).strip()
    options = parse_mcq_options(question)
    # MCQ with a recoverable option list, or open-ended: is_correct is confident.
    if options or not (len(g) == 1 and g.isalpha()):
        return is_correct(fa, g, question)
    # Single-letter GT, no option list: only flip when the answer commits.
    choice = extract_mcq_choice(fa)
    if choice is not None:
        return choice == g.upper()
    low = fa.lower()
    if "none of the" in low or "none of these" in low:
        return False
    return old  # uncertain — preserve prior label


def _is_scoreable(first_record: dict) -> bool:
    return all(k in first_record for k in ("correct", "final_answer", "ground_truth"))


def _load(path: str):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_atomic(path: str, records: list) -> None:
    tmp = path + ".rescore.tmp"
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="results", help="Directory to scan recursively.")
    ap.add_argument("--apply", action="store_true",
                    help="Rewrite changed files in place (default: dry-run report only).")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "**", "*.jsonl"), recursive=True))
    tot_files = tot_changed_files = tot_flips = tot_t2f = tot_f2t = 0

    for path in paths:
        try:
            records = _load(path)
        except (json.JSONDecodeError, OSError):
            continue
        if not records or not _is_scoreable(records[0]):
            continue
        tot_files += 1

        flips = t2f = f2t = 0
        old_correct = new_correct = 0
        for r in records:
            gt = r.get("ground_truth")
            old = bool(r.get("correct", False))
            new = _rescore(r.get("final_answer") or "", gt, old, r.get("question"))
            old_correct += int(old)
            new_correct += int(new)
            if new != old:
                flips += 1
                if old and not new:
                    t2f += 1
                else:
                    f2t += 1
                r["correct"] = new

        n = len(records)
        if flips:
            tot_changed_files += 1
            tot_flips += flips
            tot_t2f += t2f
            tot_f2t += f2t
            print(f"{path}")
            print(f"    n={n}  acc {old_correct/n:.3f} -> {new_correct/n:.3f}  "
                  f"flips={flips} (T->F={t2f}, F->T={f2t})")
            if args.apply:
                _write_atomic(path, records)

    action = "rewrote" if args.apply else "would rewrite"
    print()
    print(f"Scanned {tot_files} scoreable files; {action} {tot_changed_files} changed files.")
    print(f"Total flips: {tot_flips}  (T->F={tot_t2f}, F->T={tot_f2t})")
    if not args.apply and tot_changed_files:
        print("Dry-run only. Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
