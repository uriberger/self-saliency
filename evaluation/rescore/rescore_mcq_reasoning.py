#!/usr/bin/env python
"""GPU-free re-score of a cached MMMU-Pro (MCQ) samples file for reasoning models.

lmms-eval's response cache stores per-sample *results*, so re-running after a
parser change reuses the stale parsed_pred. This re-applies post-</think> answer
extraction to the cached raw responses and rewrites parsed_pred, so the results
table reflects the fix without a GPU re-generation (responses are deterministic).

Extraction mirrors mmmu_pro/utils.py::_extract_reasoned_choice: the option letter
is taken ONLY from the segment after the last </think> (\\boxed / 'answer is X' /
leading 'X.' / bare letter), falling back to the existing parsed_pred otherwise —
so non-reasoning outputs are unchanged.

Usage:
    python scripts/rescore_mcq_reasoning.py <samples.jsonl> [--in-place | --out FILE]
"""
import argparse, json, os, re

CHOICES = set("ABCDEFGHIJKLMNOP")


def extract_post_think_letter(pred, choices=CHOICES):
    if "</think>" not in pred:
        return None
    seg = pred.rsplit("</think>", 1)[1].strip()
    if not seg:
        return None
    m = re.search(r"\\boxed\{\s*([A-Z])\b", seg)
    if m and m.group(1) in choices:
        return m.group(1)
    ms = re.findall(r"(?:answer|option)(?:\s+is)?\s*[:\-]?\s*\(?([A-Z])\b", seg, re.I)
    for letter in reversed(ms):
        if letter.upper() in choices:
            return letter.upper()
    m = re.match(r"\(?([A-Z])[).:\s]", seg)
    if m and m.group(1) in choices:
        return m.group(1)
    stripped = seg.strip("()*.\n ")
    if len(stripped) == 1 and stripped.upper() in choices:
        return stripped.upper()
    return None


def resp_of(r):
    v = r.get("resps") or r.get("filtered_resps")
    while isinstance(v, list):
        v = v[0] if v else ""
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--acc-field", default="mmmu_acc")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.samples)]
    old_correct = new_correct = recovered = lost = changed = 0
    for r in rows:
        acc = r[args.acc_field]
        target = r.get("target") or acc.get("answer")
        old = acc.get("parsed_pred")
        old_correct += (old == target)
        e = extract_post_think_letter(resp_of(r))
        new = e if e is not None else old
        if new != old:
            changed += 1
            acc["parsed_pred"] = new
        new_correct += (new == target)
        if new == target and old != target:
            recovered += 1
        if new != target and old == target:
            lost += 1

    n = len(rows)
    print(f"samples: {n}")
    print(f"old acc: {old_correct}/{n} = {100*old_correct/n:.1f}%")
    print(f"new acc: {new_correct}/{n} = {100*new_correct/n:.1f}%  "
          f"(changed {changed}, recovered {recovered}, lost {lost})")

    out = args.samples if args.in_place else args.out
    if out:
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote corrected parsed_pred -> {out}")

    # Update the aggregated <ts>_results.json (what the results table reads) only
    # when rewriting the canonical samples file in place.
    if args.in_place:
        results_path = re.sub(r"_samples_.*\.jsonl$", "_results.json", args.samples)
        task = os.path.basename(args.samples).split("_samples_", 1)[1].rsplit(".jsonl", 1)[0]
        metric_key = f"{args.acc_field},none"
        if os.path.exists(results_path):
            rj = json.loads(open(results_path).read())
            task_res = rj.get("results", {}).get(task)
            if task_res is not None and metric_key in task_res:
                task_res[metric_key] = new_correct / n
                with open(results_path, "w") as f:
                    json.dump(rj, f, indent=2)
                print(f"updated {metric_key}={new_correct/n:.4f} -> {results_path}")
            else:
                print(f"WARN: {task}/{metric_key} not in {results_path}")
        else:
            print(f"WARN: results.json not found at {results_path}")
    if not out:
        print("(dry run; pass --in-place or --out to write)")


if __name__ == "__main__":
    main()
