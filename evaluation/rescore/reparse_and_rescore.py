"""Re-parse final_answer from response and recompute is_correct for target files.

Applies the current parse_response (which now handles a missing </think> tag when
an <answer> block is still present) and then re-scores is_correct using is_correct.

Only files whose basename starts with one of the TARGET_PREFIXES are touched.

Usage:
  python scripts/reparse_and_rescore.py            # dry-run: report changes only
  python scripts/reparse_and_rescore.py --apply    # rewrite changed files in place
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_experiment import is_correct, parse_response  # noqa: E402

TARGET_PREFIXES = (
    "qwen2_5_vl_7b_instruct-",
    "cosmos3_nano_reasoner-",
    "saliency_r1_ci_v2-",
    "saliency_r1_7b-",
    "grpo_saliency_r1_ci_v2_saliency_r1-",
)


def _load(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_atomic(path: str, records: list[dict]) -> None:
    tmp = path + ".reparse.tmp"
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="results", help="Directory to scan recursively.")
    ap.add_argument("--apply", action="store_true",
                    help="Rewrite changed files in place (default: dry-run).")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "**", "*.jsonl"), recursive=True))
    tot_files = tot_changed = tot_fa_flips = tot_score_flips = 0

    for path in paths:
        basename = os.path.basename(path)
        if not any(basename.startswith(p) for p in TARGET_PREFIXES):
            continue

        try:
            records = _load(path)
        except (json.JSONDecodeError, OSError):
            continue
        if not records or "response" not in records[0]:
            continue
        tot_files += 1

        fa_flips = score_flips = 0
        old_correct = new_correct = 0
        for r in records:
            old_fa = r.get("final_answer") or ""
            old_score = bool(r.get("correct", False))
            old_correct += int(old_score)

            reasoning, new_fa = parse_response(r.get("response") or "")
            new_score = is_correct(new_fa, str(r.get("ground_truth") or ""), r.get("question"))
            new_correct += int(new_score)

            if new_fa != old_fa:
                fa_flips += 1
                r["reasoning"] = reasoning
                r["final_answer"] = new_fa
            if new_score != old_score:
                score_flips += 1
            r["correct"] = new_score

        if fa_flips or score_flips:
            tot_changed += 1
            tot_fa_flips += fa_flips
            tot_score_flips += score_flips
            n = len(records)
            print(f"{path}")
            print(f"    n={n}  acc {old_correct/n:.3f} -> {new_correct/n:.3f}"
                  f"  final_answer_changes={fa_flips}  score_flips={score_flips}")
            if args.apply:
                _write_atomic(path, records)

    action = "rewrote" if args.apply else "would rewrite"
    print()
    print(f"Scanned {tot_files} target files; {action} {tot_changed} changed files.")
    print(f"Total final_answer changes: {tot_fa_flips}  score flips: {tot_score_flips}")
    if not args.apply and tot_changed:
        print("Dry-run only. Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
