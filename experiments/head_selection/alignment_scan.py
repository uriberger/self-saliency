"""Phase 0 of the hack-resistant reward search: score every head for box alignment.

The reward heads (22,28)/(22,31) were picked by correlation with *correctness*, and
turn out to be BELOW chance on actually attending to the box (AUROC 0.39-0.45) while
most heads are above it. See wiki/hack-resistant-overlap-reward-plan.md.

This sweeps all 36 x 32 heads over every collected dataset and box source, computing
six per-(step, head) statistics, and persists them **per sample** so the downstream
work (correctness correlation, cross-validation, attack simulation) never has to
re-read the ~200 GB of .npz again.

Metrics
-------
    mean_in     mean over in-box patches of A / A.max()      the CURRENT reward.
                Divides by the map's own PEAK, so a map that merely flattens scores
                higher. Included as the control that should disagree with the rest.
    lift        (in-box mass / total image mass) / box area   chance = 1.0
                Area-normalised, so inflating the box buys nothing.
    auroc       P(random in-box patch outranks a random out-box patch), average
                ranks for ties                                chance = 0.5
                Depends only on patch ORDER, so it is exactly invariant to any
                monotone reshaping m -> m**gamma. The candidate.
    enrichment  mean_in_raw / mean_out_raw                    chance = 1.0
                Stored as mean AND median over steps: its per-step distribution is
                unbounded above with skew up to 21, so the old mean-based head
                rankings are suspect.
    mass_in     in-box mass / total image mass                chance = box area
                Diagnostic: separates the area effect from the alignment effect.
    sum_in      RAW sum of the map inside the box, no normalisation at all
                NB the collected maps were never renormalised over visual tokens:
                they are softmax weights over ALL keys with the image columns
                sliced out, so this is a fraction of the whole attention row
                (~1e-3), and sum_in == mass_in * image_mass. It is therefore
                sensitive to both box area (H2) and total image attention (H1) --
                kept as a diagnostic, not proposed as a reward.
                NB2 this is NOT the `sum_in` of aggregation_correlation, which
                sums the PEAK-NORMALISED map; that one is recoverable here as
                mean_in * box_area * n_patches.
    image_mass  sum of the map over image patches
                Same reasoning: the fraction of the row spent on the image. It is
                the magnitude guard AUROC needs (AUROC is blind to a model that
                withdraws from the image but keeps a good ranking).

Box source
----------
`--box-source dino` reads the per-box scores (`boxes_scored`) at `--box-threshold`;
`--box-source human` reads the `boxes` field as-is. This mirrors
aggregation_correlation._filtered_boxes, so `agg_corr_visual_cot` — which carries
human boxes in `boxes` and DINO boxes in `boxes_scored` — is scanned twice, once per
source, off the same directory.

Heavy: ~55 GB of reads and 1-2 CPU-hours single-threaded over all collections. Run it
on a compute node with --jobs, not on the login node.

    python analysis/head_alignment_scan.py \
        --dir results/analysis/agg_corr_visual_cot \
        --name visual_cot_human --box-source human --jobs 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

# parents[2] is the REPOSITORY root. `parent.parent` is `experiments/`, which put
# every default output path under `experiments/results/` instead of `results/`.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# `screen.py` IS the archive's `V analysis/aggregation_correlation.py` -- same functions,
# same line numbers -- so this is the same implementation, reached by the name it has here.
from experiments.head_selection.screen import _filtered_boxes  # noqa: E402

METRICS = ("mean_in", "lift", "auroc", "enrichment", "enrichment_med",
           "mass_in", "sum_in", "image_mass")
DEFAULT_OUT = ROOT / "results" / "analysis" / "head_alignment"


def _mask(step, grid_h, grid_w, box_thr, max_area):
    """Flat boolean patch mask for a step's boxes, or None if degenerate.

    Rasterisation is identical to aggregation_correlation / the online reward.
    """
    boxes = _filtered_boxes(step, box_thr, max_area)
    if not boxes:
        return None
    m = np.zeros((grid_h, grid_w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        r0 = max(0, int(y1 * grid_h))
        r1 = min(grid_h, max(r0 + 1, round(y2 * grid_h)))
        c0 = max(0, int(x1 * grid_w))
        c1 = min(grid_w, max(c0 + 1, round(x2 * grid_w)))
        m[r0:r1, c0:c1] = True
    n_in = int(m.sum())
    if n_in == 0 or n_in == grid_h * grid_w:
        return None
    return m.ravel()


def _auroc(A, mask):
    """Rank-based AUROC per head. A: (L, H, P) -> (L, H).

    Average ranks for ties: attention maps have many near-identical near-zero
    patches, and argsort would break those ties arbitrarily, biasing the estimate.
    """
    from scipy.stats import rankdata

    r = rankdata(A, method="average", axis=-1)      # 1-based average ranks
    n_in = int(mask.sum())
    n_out = A.shape[-1] - n_in
    u = r[..., mask].sum(-1) - n_in * (n_in + 1) / 2.0
    return u / (n_in * n_out)


def _score_sample(args, run_dir, box_thr, max_area):
    """-> (sample_id, correct, n_grounded, box_area, n_patches, {metric: (L,H) f32}) or None."""
    rec = args
    npz = run_dir / rec["npz_file"]
    if not npz.exists():
        return None
    try:
        with np.load(npz) as z:
            # npz members decompress lazily -- touch ONLY attn_tr_mean (the token
            # reduction training used), never the min/max/v_norms siblings.
            attn = z["attn_tr_mean"].astype(np.float32)
    except Exception:  # noqa: BLE001  truncated / partially written file
        return None

    gh, gw = rec["grid_h"], rec["grid_w"]
    acc = {k: [] for k in METRICS}
    areas = []
    for step in rec.get("steps", []):
        si = step["step_index"]
        if si >= attn.shape[0]:
            continue
        mask = _mask(step, gh, gw, box_thr, max_area)
        if mask is None:
            continue
        A = attn[si]                                   # (L, H, P)
        tot = A.sum(-1)                                # (L, H) == image_mass
        if not np.any(tot > 0):
            continue
        area = float(mask.mean())
        safe = np.where(tot > 0, tot, 1.0)

        peak = A.max(-1)
        peak_safe = np.where(peak > 0, peak, 1.0)
        mean_in = A[..., mask].mean(-1) / peak_safe

        sum_in = A[..., mask].sum(-1)      # raw, un-normalised (see module docstring)
        mass_in = sum_in / safe
        lift = mass_in / area
        mo = A[..., ~mask].mean(-1)
        enr = np.where(mo > 0, A[..., mask].mean(-1) / np.where(mo > 0, mo, 1.0), np.nan)

        acc["mean_in"].append(mean_in)
        acc["mass_in"].append(mass_in)
        acc["sum_in"].append(sum_in)
        acc["lift"].append(lift)
        acc["enrichment"].append(enr)
        acc["enrichment_med"].append(enr)
        acc["image_mass"].append(tot)
        acc["auroc"].append(_auroc(A, mask))
        areas.append(area)

    if not acc["lift"]:
        return None
    out = {}
    with np.errstate(invalid="ignore"):
        for k, v in acc.items():
            stack = np.stack(v, 0)
            # enrichment is NaN wherever a head had zero mass outside the box
            if k.startswith("enrichment"):
                allnan = np.all(np.isnan(stack), axis=0)
                red = np.nanmedian(np.where(allnan, 0.0, stack), 0) if k == "enrichment_med" \
                    else np.nanmean(np.where(allnan, 0.0, stack), 0)
                red = np.where(allnan, np.nan, red)
            else:
                red = stack.mean(0)
            out[k] = red.astype(np.float32)
    return (str(rec["sample_id"]), rec.get("correct"), len(acc["lift"]),
            float(np.mean(areas)), int(attn.shape[-1]), out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="a collect output dir (has metadata.jsonl)")
    ap.add_argument("--name", required=True, help="label for the output file")
    ap.add_argument("--box-source", choices=["dino", "human"], default="dino")
    ap.add_argument("--box-threshold", type=float, default=0.10,
                    help="DINO score cut; ignored for --box-source human")
    ap.add_argument("--max-box-area", type=float, default=0.5)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = ap.parse_args()

    run_dir = Path(args.dir)
    meta = run_dir / "metadata.jsonl"
    if not meta.exists():
        sys.exit(f"No metadata.jsonl in {run_dir}")
    records = [json.loads(l) for l in meta.open() if l.strip()]
    if args.limit:
        records = records[: args.limit]

    # human boxes live in `boxes`; _filtered_boxes only consults `boxes_scored`
    # when a threshold is given, so passing None selects the human annotation.
    box_thr = args.box_threshold if args.box_source == "dino" else None
    print(f"[{args.name}] {len(records)} samples from {run_dir}", flush=True)
    print(f"[{args.name}] box_source={args.box_source} box_threshold={box_thr} "
          f"max_box_area={args.max_box_area} jobs={args.jobs}", flush=True)

    worker = partial(_score_sample, run_dir=run_dir,
                     box_thr=box_thr, max_area=args.max_box_area)

    t0 = time.time()
    results = []
    if args.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(args.jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(worker, records, chunksize=16), 1):
                if r is not None:
                    results.append(r)
                if i % 1000 == 0:
                    el = time.time() - t0
                    print(f"[{args.name}]   {i}/{len(records)}  {i / el:.0f}/s  "
                          f"ETA {(len(records) - i) / (i / el) / 60:.1f}min", flush=True)
    else:
        for i, rec in enumerate(records, 1):
            r = worker(rec)
            if r is not None:
                results.append(r)
            if i % 500 == 0:
                print(f"[{args.name}]   {i}/{len(records)}", flush=True)

    if not results:
        sys.exit(f"[{args.name}] nothing scored — check --box-source / --box-threshold")

    results.sort(key=lambda x: x[0])
    sample_ids = np.array([r[0] for r in results])
    correct = np.array([-1 if r[1] is None else int(bool(r[1])) for r in results], dtype=np.int8)
    n_grounded = np.array([r[2] for r in results], dtype=np.int32)
    # box_area / n_patches make the H2 (box-inflation) monitor possible and let the
    # peak-normalised sum_in of aggregation_correlation be recovered as
    # mean_in * box_area * n_patches, for comparability with the old sweeps.
    box_area = np.array([r[3] for r in results], dtype=np.float32)
    n_patches = np.array([r[4] for r in results], dtype=np.int32)
    arrays = {k: np.stack([r[5][k] for r in results], 0) for k in METRICS}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}.npz"
    np.savez_compressed(
        out_path, sample_ids=sample_ids, correct=correct, n_grounded=n_grounded,
        box_area=box_area, n_patches=n_patches, **arrays
    )
    (out_dir / f"{args.name}.json").write_text(json.dumps({
        "name": args.name, "dir": str(run_dir), "box_source": args.box_source,
        "box_threshold": box_thr, "max_box_area": args.max_box_area,
        "n_records": len(records), "n_scored": len(results),
        "n_layers": int(arrays["lift"].shape[1]), "n_heads": int(arrays["lift"].shape[2]),
        "labelled": int((correct >= 0).sum()), "base_rate": float(correct[correct >= 0].mean())
        if (correct >= 0).any() else None,
        "mean_box_area": float(box_area.mean()),
        "mean_grounded_steps": float(n_grounded.mean()),
    }, indent=1))

    L = arrays["lift"].shape[1]
    lift = np.nanmean(arrays["lift"], 0)
    auc = np.nanmean(arrays["auroc"], 0)
    print(f"\n[{args.name}] scored {len(results)}/{len(records)} in "
          f"{(time.time() - t0) / 60:.1f}min -> {out_path}")
    print(f"[{args.name}] trained heads (22,28)/(22,31): "
          f"lift {lift[22, 28]:.3f}/{lift[22, 31]:.3f}  auroc {auc[22, 28]:.3f}/{auc[22, 31]:.3f}")
    bi = int(np.nanargmax(auc))
    print(f"[{args.name}] best auroc L{bi // auc.shape[1]}H{bi % auc.shape[1]}={auc.ravel()[bi]:.3f}  "
          f"| heads>0.5: {int(np.nansum(auc > 0.5))}/{auc.size}  "
          f"| heads lift>1: {int(np.nansum(lift > 1))}/{lift.size}   (L={L})")


if __name__ == "__main__":
    main()
