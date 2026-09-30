"""Cross-dataset head-transfer test.

Question: does the single (layer, head) combo that wins on dataset A still
correlate with correctness on dataset B (and vice versa)?

Unlike analysis/cv_generalization.py (within-dataset 80/20 splits measuring
selection *optimism*), head selection here happens on a *different* dataset than
evaluation, so the target dataset's full-set point-biserial r is already an
unbiased estimate of transfer — no train/test split is needed.

Per the user's constraint, only **positively**-correlated heads are considered:
the winner on each dataset is the per-(layer,head) combo with the largest
*positive* r (attend-to-region -> more correct), and we report that same combo's
r on the other dataset. Head-averaged / layer-aggregated combos are excluded;
only single (layer, head) picks (keys containing 'li' and 'hi').

Scoring reuses aggregation_correlation.compute_combo_scores so it is identical to
the headline tables. The same box filter is applied to BOTH dirs (so a DINO run
uses --box-threshold/--max-box-area; a human run uses neither).

Usage
-----
  # human boxes (no filter)
  ./venv/bin/python analysis/cross_dataset_head_transfer.py \
      --dir-a results/analysis/agg_corr_saliency_r1_8k_human_bbox --label-a saliency_human \
      --dir-b results/analysis/agg_corr_visual_cot                 --label-b viscot_human \
      --metric mean_in

  # DINO boxes (flagship filter, needs boxes_scored in both dirs)
  ./venv/bin/python analysis/cross_dataset_head_transfer.py \
      --dir-a results/analysis/agg_corr_saliency_r1_8k --label-a saliency_dino \
      --dir-b results/analysis/agg_corr_visual_cot     --label-b viscot_dino \
      --metric mean_in --box-threshold 0.10 --max-box-area 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# parents[2] is the REPOSITORY root. `parent.parent` is `experiments/`, which put
# every default output path under `experiments/results/` instead of `results/`.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# `screen.py` IS the archive's `V analysis/aggregation_correlation.py` -- same functions,
# same line numbers -- so this is the same implementation, reached by the name it has here.
from experiments.head_selection.screen import compute_combo_scores  # noqa: E402


def _is_head_combo(key: str) -> bool:
    """Single (layer, head) picks only — exclude head/layer aggregations."""
    return ("li" in key) and ("hi" in key)


def load_dir_scores(out_dir, box_thr, max_area):
    """Load a dataset once; return (combo_scores, correct_arr, head_keys)."""
    combo_scores, correct_arr, all_keys, _valid = compute_combo_scores(
        out_dir, box_thr, max_area,
    )
    head_keys = [k for k in all_keys if _is_head_combo(k)]
    return combo_scores, correct_arr.astype(np.float64), head_keys


def per_combo_r(combo_scores, y, keys, metric):
    """{combo_key: (r, n)} for every per-(layer,head) combo, one metric.

    r is the point-biserial correlation (masked Pearson vs 0/1 correctness),
    computed the same way as aggregation_correlation.analyze.
    """
    out = {}
    for k in keys:
        x = np.array(
            [np.nan if v is None else v for v in combo_scores[k][metric]],
            dtype=np.float64,
        )
        m = ~np.isnan(x)
        n = int(m.sum())
        if n < 10:
            continue
        xv, yv = x[m], y[m]
        if xv.std() == 0 or yv.std() == 0:
            continue
        r = float(np.corrcoef(xv, yv)[0, 1])
        if np.isfinite(r):
            out[k] = (r, n)
    return out


def transfer_report(sel_label, tgt_label, r_sel, r_tgt, topk):
    """Top-k positive heads selected on `sel`, with their r on `sel` and `tgt`."""
    common = [k for k in r_sel if k in r_tgt and r_sel[k][0] > 0]
    common.sort(key=lambda k: -r_sel[k][0])
    rows = []
    for k in common[:topk]:
        rs, ns = r_sel[k]
        rt, nt = r_tgt[k]
        rows.append({
            "combo": k,
            f"r_{sel_label}": round(rs, 4),
            f"n_{sel_label}": ns,
            f"r_{tgt_label}": round(rt, 4),
            f"n_{tgt_label}": nt,
            "retained_sign": rt > 0,
            "frac_retained": round(rt / rs, 3) if rs > 0 else None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-a", required=True)
    ap.add_argument("--dir-b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--metrics", default="mean_in,sum_in,enrichment",
                    help="Comma-separated metrics to report (all share one data load).")
    ap.add_argument("--box-threshold", type=float, default=None)
    ap.add_argument("--max-box-area", type=float, default=None)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--output", default=None,
                    help="Optional JSON path to write the full report.")
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    # Load each dataset ONCE, reduce to per-combo r for all metrics, then FREE the
    # big combo_scores dict before loading the next dataset — holding both datasets'
    # full scores at once (~14 GB each for the 18k set) OOM-kills under a session cap.
    def _reduce(dir_, box_thr, max_area, label):
        import gc
        print(f"\n== Scoring {label} ({dir_}) ==", flush=True)
        cs, y, keys = load_dir_scores(dir_, box_thr, max_area)
        r_by_metric = {m: per_combo_r(cs, y, keys, m) for m in metrics}
        del cs, y, keys
        gc.collect()
        return r_by_metric

    ra = _reduce(args.dir_a, args.box_threshold, args.max_box_area, args.label_a)
    rb = _reduce(args.dir_b, args.box_threshold, args.max_box_area, args.label_b)

    def _show(title, rows, sel, tgt):
        print(f"\n{'='*78}\n{title}\n{'='*78}")
        print(f"{'combo':30s} {'r@'+sel:>12s} {'r@'+tgt:>12s}  {'sign kept':>9s} {'frac':>6s}")
        for row in rows:
            print(f"{row['combo']:30s} {row['r_'+sel]:+12.4f} {row['r_'+tgt]:+12.4f}"
                  f"  {str(row['retained_sign']):>9s} {str(row['frac_retained']):>6s}")

    report = {}
    for metric in metrics:
        r_a = ra[metric]
        r_b = rb[metric]
        a2b = transfer_report(args.label_a, args.label_b, r_a, r_b, args.topk)
        b2a = transfer_report(args.label_b, args.label_a, r_b, r_a, args.topk)
        _show(f"[{metric}] Winning POSITIVE heads on {args.label_a} -> tested on "
              f"{args.label_b}", a2b, args.label_a, args.label_b)
        _show(f"[{metric}] Winning POSITIVE heads on {args.label_b} -> tested on "
              f"{args.label_a}", b2a, args.label_b, args.label_a)
        report[metric] = {"a_to_b": a2b, "b_to_a": b2a}

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "metrics": metrics,
                "box_threshold": args.box_threshold,
                "max_box_area": args.max_box_area,
                "label_a": args.label_a, "label_b": args.label_b,
                "by_metric": report,
            }, f, indent=2)
        print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
