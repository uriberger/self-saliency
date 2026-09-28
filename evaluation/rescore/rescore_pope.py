#!/usr/bin/env python
"""GPU-free re-score of a cached POPE samples file using the (patched) real metric.

The response cache stores per-sample results, so a metric change (yes/no label
extraction) doesn't apply on a cache-hit re-run. This re-applies the actual
lmms_eval pope_process_results + aggregations to the cached raw responses and
reports accuracy/precision/recall/F1, optionally rewriting the samples file so
the results table reflects the fix. Deterministic responses -> no re-generation.

Run in the lmms_eval conda env:
    python scripts/rescore_pope.py <samples_pope.jsonl> [--in-place]
"""
import argparse, json, os, re, sys

sys.path.insert(0, "${SELFSAL_ROOT:-.}/../lmms-eval")
from lmms_eval.tasks.pope.utils import (  # noqa: E402
    pope_process_results,
    pope_aggregate_accuracy,
    pope_aggregate_precision,
    pope_aggregate_recall,
    pope_aggregate_f1_score,
    pope_aggregate_yes_ratio,
)

METRICS = ["pope_accuracy", "pope_precision", "pope_recall", "pope_f1_score", "pope_yes_ratio"]
AGG = {
    "pope_accuracy": pope_aggregate_accuracy,
    "pope_precision": pope_aggregate_precision,
    "pope_recall": pope_aggregate_recall,
    "pope_f1_score": pope_aggregate_f1_score,
    "pope_yes_ratio": pope_aggregate_yes_ratio,
}


def raw(r):
    v = r.get("resps")
    while isinstance(v, list):
        v = v[0] if v else ""
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("--in-place", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.samples)]
    old = {m: [] for m in METRICS}
    new = {m: [] for m in METRICS}
    for r in rows:
        for m in METRICS:
            old[m].append(r[m])
        gt = r["pope_accuracy"]["ground_truth"]
        qid = r["pope_accuracy"]["question_id"]
        out = pope_process_results({"answer": gt, "question_id": qid}, [raw(r)])
        for m in METRICS:
            new[m].append(out[m])
            r[m] = out[m]  # update in-memory record

    def line(tag, data):
        vals = {m.replace("pope_", ""): 100 * AGG[m](data[m]) for m in METRICS}
        return f"{tag}: " + "  ".join(f"{k}={v:.1f}" for k, v in vals.items())

    print(line("old (exact-match)", old))
    print(line("new (label-extract)", new))

    if args.in_place:
        with open(args.samples, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote corrected per-sample metrics -> {args.samples}")
        # Also update the aggregated <ts>_results.json (what the results table reads).
        results_path = re.sub(r"_samples_.*\.jsonl$", "_results.json", args.samples)
        if os.path.exists(results_path):
            rj = json.loads(open(results_path).read())
            task_res = rj.get("results", {}).get("pope")
            if task_res is not None:
                for m in METRICS:
                    task_res[f"{m},none"] = AGG[m](new[m])
                with open(results_path, "w") as f:
                    json.dump(rj, f, indent=2)
                print(f"updated aggregated results -> {results_path}")
            else:
                print(f"WARN: no 'pope' block in {results_path}")
        else:
            print(f"WARN: results.json not found at {results_path}")
    else:
        print("(dry run; pass --in-place to write)")


if __name__ == "__main__":
    main()
