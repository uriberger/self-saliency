#!/usr/bin/env python
"""Score every observe step of a saliency_viz run against ITS OWN referent box.

`saliency_viz.py` draws maps and loads no detector; the probes score maps against the
dataset's *answer* box. Neither answers the question a Figure-1 panel actually asks --
"in this step, did the model look at the thing the step is talking about?" -- because the
answer box is the same box for every step of the chain, so a step that describes some
other object is marked wrong for describing it.

This script asks the per-step question instead, with the same instrument R_sal used during
training -- literally the same functions, not a reimplementation: Grounding-DINO grounds
the step's whole sentence (`selfsal.grounding.ground`), the surviving boxes are rasterised
onto the map's patch grid (`union_mask`, so the per-box area cap applies), and the step's
map is scored inside that union by phi (`selfsal.saliency`). That makes the number
reward-aligned by construction -- it is the training target, not independent evidence --
which is the right thing for an illustration and the wrong thing for a headline metric.

The cross-model search is the point: within one sample, pair a step of model A with a step
of model B whose referent masks agree (grid IoU >= --iou-min, i.e. the two sentences are
about the same region), then rank by how much better A's map sits inside that shared region
than B's. That pairing is what makes a single panel honest -- the two rows are then the
same picture and the same target, differing only in where the attention went.

CPU-only: Grounding-DINO falls back to CPU when no GPU is visible, which is ~100 (image,
sentence) pairs for a 20-sample 3-model run and a few minutes on a login node.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from selfsal.grounding import ground, union_mask
from selfsal.saliency import auroc, phi, phi_mean

REPO = Path(__file__).resolve().parents[2]   # the repository root

# The grounding and scoring settings for this run. In the archive these were reached out
# of the reward module's module-level `_CFG`, which is why this file used to build a fake
# `trl.rewards` package and load `overlap_rewards.py` by path. The method is a library
# here -- `selfsal.grounding` grounds and `selfsal.saliency` scores, and both take their
# settings as arguments -- so there is no global to reach into, and this script holds its
# own. Same shape as `experiments/trained_model/probe.py`'s CFG, and for the same reason.
CFG = {
    "box_threshold": 0.10,
    "max_box_area": 0.5,
    "max_union_area": None,
    "dino_batch_size": 32,
    "dino_device": None,
}


def border_frac(smap: np.ndarray) -> float:
    """Share of the map's mass on the outer one-patch ring.

    Reported on every step because the encoder stamps the border in all image types, and a
    box that happens to touch an edge can otherwise read as a spectacular hit.
    """
    inner = np.zeros(smap.shape, dtype=bool)
    inner[1:-1, 1:-1] = True
    total = float(smap.sum())
    return 1.0 - float(smap[inner].sum()) / total if total > 0 else float("nan")


def score_one(smap: np.ndarray, mask: np.ndarray) -> dict:
    # The historical key names are kept: stored runs of this script are keyed on them, and
    # the figure scripts that read its JSON index them. `mean_in` IS phi (Equation 1) and
    # `mean_in_v2` IS phi_mean (Appendix C) -- see selfsal.saliency.score.
    return {
        "mean_in": phi(smap, mask),
        "mean_in_v2": phi_mean(smap, mask),
        "auroc": auroc(smap, mask),
        "box_area_frac": float(mask.sum()) / float(mask.size),
        "peak_in": bool(mask[np.unravel_index(np.argmax(smap), smap.shape)]),
    }


def collect(run_dir: Path, models: dict[str, str], map_key: str) -> list[dict]:
    """One record per (model, sample, step): the map, the step text and the image."""
    items = []
    for tag, sub in models.items():
        sample_root = run_dir / sub / "samples"
        if not sample_root.is_dir():
            raise SystemExit(f"no samples under {sample_root}")
        for sample in sorted(p.name for p in sample_root.iterdir() if p.is_dir()):
            sdir = sample_root / sample
            meta = json.loads((sdir / "meta.json").read_text())
            if meta.get("dropped") or not (sdir / "maps.npz").exists():
                continue
            maps = np.load(sdir / "maps.npz")
            if map_key not in maps.files:
                continue
            z = maps[map_key]
            image = Image.open(sdir / "original.png").convert("RGB")
            for i, step in enumerate(meta["steps"]):
                items.append({
                    "model": tag, "sample": sample, "row": int(sample.split("row")[1]),
                    "step": i, "text": step["text"], "question": meta.get("question", ""),
                    "gt_answer": meta.get("gt_answer", ""), "format_ok": meta.get("format_ok"),
                    "map": np.clip(z[i], 0, None).astype(np.float64), "image": image,
                })
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="a saliency_viz output root, e.g. outputs/saliency_viz/sviz-3models")
    ap.add_argument("--model", action="append", default=[], metavar="NAME=SUBDIR",
                    help="repeatable; default is every subdir holding samples/")
    ap.add_argument("--map", default="glimpse", help="which map in maps.npz to score")
    ap.add_argument("--out", default=None, help="JSON to write (default <run-dir>/step_referent.json)")
    ap.add_argument("--iou-min", type=float, default=0.5,
                    help="grid-IoU above which two steps count as the same referent")
    ap.add_argument("--box-threshold", type=float, default=None, help="DINO threshold (default: reward's)")
    ap.add_argument("--max-box-area", type=float, default=None, help="per-box area cap (default: reward's)")
    ap.add_argument("--max-union-area", type=float, default=None, help="per-step union cap (default: reward's)")
    ap.add_argument("--dino-device", default=None)
    ap.add_argument("--dino-batch-size", type=int, default=8)
    ap.add_argument("--top", type=int, default=25, help="how many matched pairs to print")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if args.model:
        models = dict(m.split("=", 1) for m in args.model)
    else:
        models = {p.name: p.name for p in sorted(run_dir.iterdir())
                  if (p / "samples").is_dir()}
    if len(models) < 2:
        print(f"[warn] only {len(models)} model(s); the cross-model pairing needs two")

    CFG["dino_batch_size"] = args.dino_batch_size
    for key, value in (("box_threshold", args.box_threshold),
                       ("max_box_area", args.max_box_area),
                       ("max_union_area", args.max_union_area),
                       ("dino_device", args.dino_device)):
        if value is not None:
            CFG[key] = value
    used_cfg = {k: CFG[k] for k in ("box_threshold", "max_box_area", "max_union_area")}
    print(f"[cfg] models={models}  map={args.map}  dino={used_cfg}", flush=True)

    items = collect(run_dir, models, args.map)
    print(f"[dino] grounding {len(items)} (image, step) pairs ...", flush=True)
    boxes_per_item = ground([it["image"] for it in items], [it["text"] for it in items],
                            box_threshold=CFG["box_threshold"],
                            batch_size=CFG["dino_batch_size"],
                            device=CFG["dino_device"])

    records = []
    for it, boxes in zip(items, boxes_per_item):
        smap = it["map"]
        gh, gw = smap.shape
        mask = union_mask(boxes or [], gh, gw,
                          max_box_area=CFG["max_box_area"],
                          max_union_area=CFG["max_union_area"])
        rec = {k: it[k] for k in
               ("model", "sample", "row", "step", "text", "question", "gt_answer", "format_ok")}
        rec.update(grid=[gh, gw], n_boxes=len(boxes or []), border_frac=border_frac(smap),
                   boxes=[[round(float(v), 5) for v in b] for b in (boxes or [])])
        if mask is None:
            # Exactly the reward's behaviour: an ungrounded or degenerate step is SKIPPED,
            # not scored zero, so it must not enter any mean taken over this file.
            rec.update(grounded=False, note="DINO grounded nothing, or the union was degenerate/capped")
        else:
            rec.update(grounded=True, **score_one(smap, mask))
        rec["_mask"] = mask
        rec["_map"] = smap
        records.append(rec)

    n_ok = sum(r["grounded"] for r in records)
    print(f"[dino] grounded {n_ok}/{len(records)} steps", flush=True)

    # ---- cross-model pairs on a shared referent ------------------------------------
    pairs = []
    tags = list(models)
    by_sample: dict[str, list[dict]] = {}
    for r in records:
        by_sample.setdefault(r["sample"], []).append(r)
    for sample, recs in by_sample.items():
        for a_tag, b_tag in [(x, y) for x in tags for y in tags if x != y]:
            for ra in [r for r in recs if r["model"] == a_tag and r["grounded"]]:
                for rb in [r for r in recs if r["model"] == b_tag and r["grounded"]]:
                    ma, mb = ra["_mask"], rb["_mask"]
                    if ma.shape != mb.shape:
                        continue
                    inter = float((ma & mb).sum())
                    union = float((ma | mb).sum())
                    iou = inter / union if union > 0 else 0.0
                    if iou < args.iou_min:
                        continue
                    shared = ma & mb
                    pairs.append({
                        "sample": sample, "row": ra["row"], "question": ra["question"],
                        "gt_answer": ra["gt_answer"], "iou": iou,
                        "a_model": a_tag, "a_step": ra["step"], "a_text": ra["text"],
                        "b_model": b_tag, "b_step": rb["step"], "b_text": rb["text"],
                        "a_own_v2": ra["mean_in_v2"], "b_own_v2": rb["mean_in_v2"],
                        "a_own_mean_in": ra["mean_in"], "b_own_mean_in": rb["mean_in"],
                        # Both maps against the SAME region, so the panel's two rows differ
                        # only in where the attention went.
                        "a_shared_v2": phi_mean(ra["_map"], shared),
                        "b_shared_v2": phi_mean(rb["_map"], shared),
                        "a_border": ra["border_frac"], "b_border": rb["border_frac"],
                        "a_peak_in": ra["peak_in"], "b_peak_in": rb["peak_in"],
                        "shared_area_frac": float(shared.sum()) / float(shared.size),
                    })
    for p in pairs:
        p["gap_shared_v2"] = (p["a_shared_v2"] or 0.0) - (p["b_shared_v2"] or 0.0)
    pairs.sort(key=lambda p: p["gap_shared_v2"], reverse=True)

    out = Path(args.out) if args.out else run_dir / "step_referent.json"
    for r in records:
        r.pop("_mask", None)
        r.pop("_map", None)
    out.write_text(json.dumps(
        {"run_dir": str(run_dir), "map": args.map, "models": models, "dino_cfg": used_cfg,
         "iou_min": args.iou_min, "steps": records, "pairs": pairs}, indent=1))
    print(f"[out] {out}  ({len(records)} steps, {len(pairs)} matched pairs)")

    print(f"\n=== per-step, against the step's own referent (chance for v2 = 1.0) ===")
    for tag in tags:
        vals = [r["mean_in_v2"] for r in records if r["model"] == tag and r["grounded"]]
        ng = sum(1 for r in records if r["model"] == tag and not r["grounded"])
        if vals:
            print(f"  {tag:32s} n={len(vals):3d} (+{ng} ungrounded)  "
                  f"mean_in_v2 {np.mean(vals):5.2f}   median {np.median(vals):5.2f}   "
                  f"frac>1 {np.mean([v > 1 for v in vals]):.2f}")

    print(f"\n=== top {args.top} matched pairs, ranked by A-minus-B on the SHARED region ===")
    for p in pairs[:args.top]:
        print(f"row {p['row']:4d} IoU {p['iou']:.2f} shared {p['shared_area_frac']*100:5.1f}%  "
              f"gap {p['gap_shared_v2']:+5.2f}   Q: {p['question'][:60]}")
        print(f"    A {p['a_model']:30s} s{p['a_step']} v2 {p['a_shared_v2']:5.2f} "
              f"peak_in {str(p['a_peak_in']):5s} border {p['a_border']*100:4.1f}%  {p['a_text'][:88]}")
        print(f"    B {p['b_model']:30s} s{p['b_step']} v2 {p['b_shared_v2']:5.2f} "
              f"peak_in {str(p['b_peak_in']):5s} border {p['b_border']*100:4.1f}%  {p['b_text'][:88]}")


if __name__ == "__main__":
    main()
