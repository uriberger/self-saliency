#!/usr/bin/env python
"""Where the model LOOKS, where the human says the answer IS, and what the model TALKS about.

    python sink_three_legs.py --dirs A,B,C --out DIR/three_legs.txt

Three claims, and this file is the measurement of each:

  1. the model attends the geometric BORDER
  2. the human says the answer is somewhere else -- the corpus's annotated box
  3. the model REASONS about somewhere else -- Grounding-DINO on its own observe steps

They are only one argument if all three are on one scale, so everything below is the same
statistic -- a region's share over its share of the patches, on the picture's OWN grid,
1.0 being a fair share -- and is read two ways:

  WHERE THE REGION SITS      ring enrichment of the region itself (geometry)
  WHAT THE MODEL GAVE IT     the attention mass inside the region over its area (attention)

The second is the one that needs no null: `map_obs` is a distribution over patches and
dividing by the region's area share makes 1.0 mean "its fair share" whatever the region's
size. The first does need one, and the null is the point of this file.

THE NULL. A region's ring enrichment is a function of its SIZE and SHAPE before it is a
function of its position: the border is a thin frame, so a blob dropped anywhere covers
proportionally little of it and scores below 1 without preferring the centre at all. So
every geometric row carries a matched null -- the SAME mask, rigidly translated to every
position it fits on the SAME grid, averaged exactly rather than sampled. Size, shape,
contiguity and grid are held; only position moves. A ratio of 1.0 to that null means the
apparent centre preference was the region's size and nothing else.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(REPO))

#: Which query set's pooled map to read. The user's question is about the observe steps --
#: the sentences in which the model says what it is looking at -- but the scan stores all
#: four, so the aggregation can be changed here without rescanning anything.
QUERY_MAPS = {"obs": "map_obs", "gen": "map_gen", "prompt": "map_q", "all": "map_all"}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def mask_from_boxes(boxes, gh, gw, max_area=0.5):
    """Normalised boxes -> the patches whose CENTRES they cover. -> [gh,gw] bool or None.

    The centre convention is the one `sink_observe_boxes.box_stats` and the corpus box
    already use, so a region rasterises the same way wherever it came from.
    """
    keep = [b for b in boxes
            if max_area <= 0 or (b[2] - b[0]) * (b[3] - b[1]) <= max_area]
    if not keep:
        return None
    c = (np.arange(gw) + 0.5) / gw
    r = (np.arange(gh) + 0.5) / gh
    m = np.zeros((gh, gw), dtype=bool)
    for x0, y0, x1, y1 in keep:
        m |= ((c[None, :] >= x0) & (c[None, :] <= x1)
              & (r[:, None] >= y0) & (r[:, None] <= y1))
    if not m.any():
        # A box thinner than one patch still points somewhere. Snap it rather than
        # dropping the picture, which would silently select for large boxes.
        x0, y0, x1, y1 = keep[0]
        m[min(gh - 1, max(0, int((y0 + y1) / 2 * gh))),
          min(gw - 1, max(0, int((x0 + x1) / 2 * gw)))] = True
    return m


def ring_grid(gh, gw):
    from experiments.attention_bias import measure as SL
    return np.asarray(SL.ring_set(gh, gw)).reshape(gh, gw)


def ring_enrich(mask, ring):
    k = int(mask.sum())
    ra = float(ring.mean())
    if k == 0 or ra <= 0:
        return float("nan")
    return float((mask & ring).sum()) / k / ra


def shift_null(mask, ring):
    """The mask's ring enrichment averaged over every position it fits. -> float.

    EXACT, not sampled: the valid offsets number (gh-bh+1)(gw-bw+1), a few hundred at
    most on these grids, so they are enumerated. The mask is translated rigidly, so its
    area, its shape and its contiguity are all held fixed and position is the only thing
    that varies -- which is precisely the null "this region is where it is for a reason
    other than its size" has to be tested against.
    """
    gh, gw = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return float("nan")
    y0, x0 = int(ys.min()), int(xs.min())
    bh, bw = int(ys.max()) - y0 + 1, int(xs.max()) - x0 + 1
    base = mask[y0:y0 + bh, x0:x0 + bw]
    k, ra = int(base.sum()), float(ring.mean())
    if k == 0 or ra <= 0:
        return float("nan")
    hits = [int((base & ring[dy:dy + bh, dx:dx + bw]).sum())
            for dy in range(gh - bh + 1) for dx in range(gw - bw + 1)]
    return float(np.mean(hits)) / k / ra


def attn_enrich(pmap, mask):
    """Attention mass inside the region over the region's share of the patches. -> float.

    Size-controlled: `pmap` sums to 1 over the patches, so dividing the mass in a region
    by that region's area share gives 1.0 for "its fair share" at every region size.
    """
    k = int(mask.sum())
    if k == 0 or not np.isfinite(pmap).all():
        return float("nan")
    return float(pmap[mask].sum()) / (k / mask.size)


def attn_shift_null(pmap, mask):
    """`attn_enrich` averaged over every position the mask fits. -> float.

    Size-controlling is not enough here and this is the control that finishes the job.
    Attention is not uniform over the picture, so ANY region in a favoured part of the
    frame scores above 1 whether or not it is the region that matters -- and the human
    box is central, which is exactly the part of the frame under contest. Translating the
    box and re-reading the SAME map asks the sharper question: does the model attend THIS
    region more than an identical region somewhere else? The ratio is the answer; the
    raw enrichment on its own cannot distinguish the two.
    """
    gh, gw = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0 or not np.isfinite(pmap).all():
        return float("nan")
    y0, x0 = int(ys.min()), int(xs.min())
    bh, bw = int(ys.max()) - y0 + 1, int(xs.max()) - x0 + 1
    base = mask[y0:y0 + bh, x0:x0 + bw]
    k = int(base.sum())
    if k == 0:
        return float("nan")
    vals = [pmap[dy:dy + bh, dx:dx + bw][base].sum()
            for dy in range(gh - bh + 1) for dx in range(gw - bw + 1)]
    return float(np.mean(vals)) / (k / mask.size)


def topk_mask(pmap, k):
    """The k most-attended patches. The size-matched way to ask where attention sits."""
    flat = np.argsort(pmap, axis=None)[::-1][:k]
    m = np.zeros(pmap.size, dtype=bool)
    m[flat] = True
    return m.reshape(pmap.shape)


def fmt(v, w=8, p=2):
    return f"{'--':>{w}}" if v is None or not np.isfinite(v) else f"{v:>{w}.{p}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--query", default="obs", choices=sorted(QUERY_MAPS),
                    help="which query set's pooled map to read (default: observe steps)")
    ap.add_argument("--max-box-area", type=float, default=0.5)
    args = ap.parse_args()

    P = _load("_3l_probe", "sink_location_probe.py")
    key = QUERY_MAPS[args.query]
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"Query set: {args.query}  (pooled map `{key}`, all heads over the mass floor)")
    emit("Every number is a share over a fair share: 1.00 = exactly proportional.")
    emit("")
    emit("GEOMETRY -- where each region sits, as ring enrichment, against a null that is")
    emit("the same region rigidly translated over every position it fits on that grid.")
    emit("`ratio` < 1 is a real pull to the centre; ratio ~ 1 means it was only size.")
    emit("")
    emit(f"{'model':<15} {'n':>5} | {'top-k attn':>10} {'null':>7} {'ratio':>6} |"
         f" {'human':>7} {'null':>7} {'ratio':>6} | {'referent':>8} {'null':>7} {'ratio':>6}")
    emit("-" * 108)

    attn_rows, step_rows = [], []
    for d in [x for x in args.dirs.split(",") if x]:
        meta, arrays = P.read_stage(d, "scan")
        if not meta:
            emit(f"{Path(d).name:<15} (no scan results)")
            continue
        man = {r["key"]: r for r in P.read_manifest(d)}
        boxes = {}
        bpath = Path(d) / "observe_boxes.jsonl"
        if bpath.exists():
            for line in bpath.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                boxes[r["key"]] = r

        g = {k: [] for k in ("tk", "tk0", "hu", "hu0", "rf", "rf0", "a_ring",
                             "a_hu", "a_hu0", "a_rf", "a_rf0", "hu_area", "rf_area",
                             "sf", "sf0", "a_sf", "a_sf0", "sf_area")}
        n_ref = 0
        for m in meta:
            row = man.get(m["key"])
            pm = arrays.get(m["unit"], {}).get(key)
            if row is None or pm is None:
                continue
            gh, gw = m["grid"]
            pmap = np.asarray(pm, dtype=np.float64).reshape(gh, gw)
            if not np.isfinite(pmap).all() or pmap.sum() <= 0:
                continue
            pmap = pmap / pmap.sum()
            ring = ring_grid(gh, gw)

            g["a_ring"].append(attn_enrich(pmap, ring))

            hm = mask_from_boxes([[float(v) for v in row["bbox"]]], gh, gw, 0.0) \
                if row.get("bbox") else None
            if hm is not None and hm.any():
                g["hu"].append(ring_enrich(hm, ring))
                g["hu0"].append(shift_null(hm, ring))
                g["a_hu"].append(attn_enrich(pmap, hm))
                g["a_hu0"].append(attn_shift_null(pmap, hm))
                g["hu_area"].append(float(hm.mean()))
                tk = topk_mask(pmap, int(hm.sum()))     # matched to the human box's size
                g["tk"].append(ring_enrich(tk, ring))
                g["tk0"].append(shift_null(tk, ring))

            rec = boxes.get(m["key"]) or {}
            allb = [q for st in rec.get("steps", []) for q in (st.get("boxes") or [])]
            rm = mask_from_boxes([[float(v) for v in q] for q in allb], gh, gw,
                                 args.max_box_area) if allb else None
            if rm is not None and rm.any():
                g["rf"].append(ring_enrich(rm, ring))
                g["rf0"].append(shift_null(rm, ring))
                g["a_rf"].append(attn_enrich(pmap, rm))
                g["a_rf0"].append(attn_shift_null(pmap, rm))
                g["rf_area"].append(float(rm.mean()))
                n_ref += 1
            # The union above is one region per PICTURE and it is large, because a chain
            # with six observe steps contributes six boxes to it. The per-STEP region is
            # what "the model reasons about the centre" is really a claim about, so it is
            # measured separately rather than inferred from the union.
            for st in rec.get("steps", []):
                sb = st.get("boxes") or []
                sm = mask_from_boxes([[float(v) for v in q] for q in sb], gh, gw,
                                     args.max_box_area) if sb else None
                if sm is None or not sm.any():
                    continue
                g["sf"].append(ring_enrich(sm, ring))
                g["sf0"].append(shift_null(sm, ring))
                g["a_sf"].append(attn_enrich(pmap, sm))
                g["a_sf0"].append(attn_shift_null(pmap, sm))
                g["sf_area"].append(float(sm.mean()))

        def mean(k):
            v = [x for x in g[k] if np.isfinite(x)]
            return float(np.mean(v)) if v else float("nan")

        name = Path(d).name
        tk, tk0 = mean("tk"), mean("tk0")
        hu, hu0 = mean("hu"), mean("hu0")
        rf, rf0 = mean("rf"), mean("rf0")
        emit(f"{name:<15} {len(g['hu']):>5} | {fmt(tk,10)} {fmt(tk0,7)} {fmt(tk/tk0,6)} |"
             f" {fmt(hu,7)} {fmt(hu0,7)} {fmt(hu/hu0,6)} |"
             f" {fmt(rf,8)} {fmt(rf0,7)} {fmt(rf/rf0,6)}")
        sf, sf0 = mean("sf"), mean("sf0")
        step_rows.append((name, len(g["sf"]), sf, sf0, mean("sf_area"),
                          mean("a_sf"), mean("a_sf0")))
        attn_rows.append((name, mean("a_ring"), mean("a_hu"), mean("a_hu0"),
                          mean("a_rf"), mean("a_rf0"), mean("hu_area"),
                          mean("rf_area"), len(g["hu"]), n_ref))

    emit("")
    emit("The referent column above is the UNION over a chain's observe steps, which is")
    emit("large by construction. Per STEP, the region each observe sentence names:")
    emit("")
    emit(f"{'model':<15} {'steps':>6} | {'ring':>6} {'null':>7} {'ratio':>6} |"
         f" {'attn in':>8} {'null':>7} {'ratio':>6} | {'area':>6}")
    emit("-" * 80)
    for name, ns, sf, sf0, sa, asf, asf0 in step_rows:
        emit(f"{name:<15} {ns:>6} | {fmt(sf,6)} {fmt(sf0,7)} {fmt(sf/sf0,6)} |"
             f" {fmt(asf,8)} {fmt(asf0,7)} {fmt(asf/asf0,6)} | {fmt(sa,6,3)}")

    emit("")
    emit("ATTENTION -- the mass the model put inside each region over that region's share")
    emit("of the patches. 1.00 is a fair share at any region size. BORDER needs no null;")
    emit("the box columns do, because attention is not flat and a central box would score")
    emit("high wherever the answer was -- so `null` is the SAME box moved over every")
    emit("position it fits, and `ratio` is the part that is about THIS region.")
    emit("")
    emit(f"{'model':<15} {'BORDER':>8} | {'human box':>10} {'null':>7} {'ratio':>6} |"
         f" {'referent':>9} {'null':>7} {'ratio':>6} | {'hu area':>8} {'rf area':>8}"
         f" {'n box':>6} {'n ref':>6}")
    emit("-" * 116)
    for name, ar, ah, ah0, af, af0, ha, fa, nb, nr in attn_rows:
        emit(f"{name:<15} {fmt(ar,8)} | {fmt(ah,10)} {fmt(ah0,7)} {fmt(ah/ah0,6)} |"
             f" {fmt(af,9)} {fmt(af0,7)} {fmt(af/af0,6)} |"
             f" {fmt(ha,8,3)} {fmt(fa,8,3)} {nb:>6} {nr:>6}")

    emit("")
    emit("Read the two blocks together. The border column above is a property of the")
    emit("attention and needs no null; the geometry block says whether the regions the")
    emit("border is being contrasted with are actually central, or merely small.")
    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
