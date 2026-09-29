#!/usr/bin/env python3
"""Re-score cached p3 (SalBench) lmms-eval runs with the fixed answer-token
normalization in salbench/utils.py.

The original p3 exact_match scorer lowercased predictions but only stripped
brackets, so prose-formatted answers ("Size.", "Color.") failed the set match
against the bare labels ("size", "color"). The fixed process_results strips
surrounding whitespace/punctuation. This script replays every cached
``*_samples_p3.jsonl`` through the fixed scorer and rewrites the p3 block of the
sibling ``*_results.json`` (and the per-sample scores in the samples file).

Originals are backed up next to each file with a ``.prefix_bak`` suffix.
Run from the repo root:  python scripts/rescore_p3.py [--dry-run]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# `evaluation/results` IS the lmms-eval tree here. The archive kept `<root>/inference`
# and `<root>/lmms_eval` side by side and the port flattened them, so the extra level
# below made the default glob match nothing and the script report no work to do.
LMMS_EVAL_DIR = REPO / "results"
UTILS_PATH = Path.home() / "scratch/research/lmms-eval/lmms_eval/tasks/salbench/utils.py"

# Load the (fixed) salbench scorer directly from the lmms-eval checkout so this
# script always reflects the deployed normalization logic.
_spec = importlib.util.spec_from_file_location("salbench_utils", UTILS_PATH)
U = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(U)
CATS = U.P3_CATEGORIES

# Per-sample metrics get a CLT standard error; category metrics are pooled (N/A).
SAMPLE_METRICS = ["exact_match", "sample_precision", "sample_recall", "sample_f1"]


def _get_score(sample: dict, key: str):
    v = sample.get(key)
    return v["score"] if isinstance(v, dict) else v


def rescore_file(samples_path: Path, dry_run: bool) -> dict:
    lines = samples_path.read_text().splitlines()
    samples = [json.loads(l) for l in lines]

    per_sample_new = {m: [] for m in SAMPLE_METRICS}
    per_sample_old = {m: [] for m in SAMPLE_METRICS}
    all_cat_results = []            # list of {"id":.., "pred": cat_preds}
    per_cat_results = {c: [] for c in CATS}

    for s in samples:
        resp = s["filtered_resps"]
        resp = resp[0] if isinstance(resp, list) else resp
        doc = {"answer": str(s["target"]), "image_id": s.get("doc_id", 0)}
        out = U.process_results(doc, [resp], CATS)

        for m in SAMPLE_METRICS:
            per_sample_new[m].append(out[m]["score"])
            per_sample_old[m].append(_get_score(s, m))
            s[m] = out[m]                       # rewrite per-sample score in place
        all_cat_results.append(out["all_cat_f1"])
        for c in CATS:
            per_cat_results[c].append(out[f"{c}_f1"])
        # keep the per-category / all_cat structured fields consistent too
        for key in ("all_cat_precision", "all_cat_recall", "all_cat_f1"):
            s[key] = out[key]
        for c in CATS:
            for stat in ("precision", "recall", "f1"):
                s[f"{c}_{stat}"] = out[f"{c}_{stat}"]

    n = len(samples)

    def mean(xs):
        return sum(xs) / n

    def clt_stderr(xs):
        m = mean(xs)
        var = sum((x - m) ** 2 for x in xs) / (n - 1)
        return math.sqrt(var) / math.sqrt(n)

    # Aggregate exactly as lmms-eval does, via the utils aggregation functions.
    all_p, all_r, all_f1 = U._aggregate_all_category(all_cat_results, CATS)
    new_block = {}
    for m in SAMPLE_METRICS:
        new_block[f"{m},none"] = mean(per_sample_new[m])
        new_block[f"{m}_stderr_clt,none"] = clt_stderr(per_sample_new[m])
    new_block["all_cat_precision,none"] = all_p
    new_block["all_cat_recall,none"] = all_r
    new_block["all_cat_f1,none"] = all_f1
    for c in CATS:
        p, r, f1 = U._aggregate_per_category(per_cat_results[c])
        new_block[f"{c}_precision,none"] = p
        new_block[f"{c}_recall,none"] = r
        new_block[f"{c}_f1,none"] = f1

    summary = {
        "old_exact_match": mean(per_sample_old["exact_match"]),
        "new_exact_match": new_block["exact_match,none"],
        "n": n,
    }

    if dry_run:
        return summary

    # --- write updated samples file (backup first) ---
    shutil.copy2(samples_path, samples_path.with_suffix(samples_path.suffix + ".prefix_bak"))
    with samples_path.open("w") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")

    # --- write updated results.json p3 block (backup first) ---
    ts = samples_path.name.split("_samples_p3.jsonl")[0]
    results_path = samples_path.parent / f"{ts}_results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        if "p3" in results.get("results", {}):
            shutil.copy2(results_path, results_path.with_suffix(".json.prefix_bak"))
            block = results["results"]["p3"]
            for k, v in new_block.items():
                block[k] = v            # only overwrite the keys we recomputed
            results_path.write_text(json.dumps(results, indent=2))
            summary["results_json"] = results_path.name
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report old->new without writing")
    args = ap.parse_args()

    files = sorted(LMMS_EVAL_DIR.glob("*/*/*_samples_p3.jsonl"))
    print(f"Found {len(files)} cached p3 run(s). Scorer: {UTILS_PATH}\n")
    for f in files:
        run = f.relative_to(LMMS_EVAL_DIR).parts[0]
        s = rescore_file(f, args.dry_run)
        print(f"{run:70s} n={s['n']:5d}  "
              f"exact_match {s['old_exact_match']:.3f} -> {s['new_exact_match']:.3f}"
              f"  ({'DRY' if args.dry_run else s.get('results_json','no results.json')})")


if __name__ == "__main__":
    main()
