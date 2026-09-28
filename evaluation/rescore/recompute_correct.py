"""Recompute the 'correct' field in all inference result JSONL files.

Uses the same parse_response / is_correct logic as run_experiment.py.
Run from the repo root:

    python scripts/recompute_correct.py

Each file is rewritten in-place; a summary of changed records is printed.
"""

import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Copied verbatim from run_experiment.py so this script is self-contained.
# Keep in sync if parse_response / is_correct change there.
# ---------------------------------------------------------------------------

def parse_response(response: str) -> tuple[str, str]:
    if re.search(r"<think>", response) and not re.search(r"</think>", response):
        return "", ""

    think_m = re.search(r"<think>(.*?)</think>", response, flags=re.DOTALL)
    reasoning = think_m.group(1).strip() if think_m else ""

    answer_m = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
    if answer_m:
        return reasoning, answer_m.group(1).strip()

    after_think = re.search(r"</think>(.*)", response, flags=re.DOTALL)
    if after_think:
        return reasoning, after_think.group(1).strip()

    return "", ""


def is_correct(prediction: str, ground_truth: str) -> bool:
    if not prediction or not ground_truth:
        return False
    p = prediction.lower().strip()
    g = ground_truth.lower().strip()
    return g in p or p in g


# ---------------------------------------------------------------------------

def recompute_file(path: Path) -> tuple[int, int]:
    """Return (n_changed, n_total)."""
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n_changed = 0
    for rec in records:
        response = rec.get("response", "")
        gt = rec.get("ground_truth")
        _, final_answer = parse_response(response)
        new_correct = is_correct(final_answer, str(gt) if gt is not None else "")

        old_correct = rec.get("correct")
        if new_correct != old_correct:
            n_changed += 1

        rec["correct"] = new_correct
        rec["final_answer"] = final_answer

    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return n_changed, len(records)


def main() -> None:
    results_dir = Path(__file__).parent.parent / "results" / "inference"
    if not results_dir.exists():
        print(f"ERROR: {results_dir} not found", file=sys.stderr)
        sys.exit(1)

    files = sorted(results_dir.glob("*.jsonl"))
    if not files:
        print("No JSONL files found.")
        return

    total_changed = 0
    total_records = 0
    for path in files:
        n_changed, n_total = recompute_file(path)
        total_changed += n_changed
        total_records += n_total
        status = f"  {n_changed:4d} changed / {n_total:4d} total"
        print(f"{path.name:<60s} {status}")

    print(f"\nDone. {total_changed} records updated across {len(files)} files ({total_records} total records).")


if __name__ == "__main__":
    main()
