#!/usr/bin/env python
"""Render the sink-location results as one self-contained HTML page.

    python sink_location_html.py --cold outputs/sink_location/coldstart \
        --base outputs/sink_location/base_qwen3vl --out docs/sink-location.html

Everything is inlined: no CDN, no fonts to fetch, no build step. The eval nodes run with
`HF_HUB_OFFLINE=1` and the page has to open on a laptop with no network, so a script tag
pointing at a chart library would be a page that works on exactly one machine.

The numbers are not retyped. This reads the same npz/JSONL the report reads and calls the
report's own functions with stdout swallowed, so a figure and the text of
`docs/sink-location-by-image-type.md` cannot drift apart -- which is the failure mode a
hand-built dashboard has, and it is silent.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(REPO))


def _probe():
    """The harness, whose readers this page calls rather than retyping their numbers.

    A function rather than a plain import only because it is bound to `SP` below and the
    name `probe` would shadow nothing useful here; it is called at module scope, so this
    does pull in torch.
    """
    from experiments.attention_bias import probe
    return probe


SP = _probe()
from experiments.attention_bias import measure as SL

LOC_LABELS = {"ring": "ring", "depth1": "one in", "middle": "middle", "top": "top row",
              "bottom": "bottom row", "left": "left col", "right": "right col",
              "topleft": "top-left patch", "botright": "bottom-right patch"}


class Args:
    """The report's argument surface, without argparse."""
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.n_cells, self.min_mass, self.n_boot = 16, 0.002, 2000
        self.blank_min, self.sink_x, self.sink_cv = 0.15, 10.0, 0.5
        self.types = list(SP.CORPUS)


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def collect(out_dir):
    """Every number the page shows, from one run directory."""
    args = Args(str(out_dir))
    meta, arrays = SP.read_stage(out_dir, "scan")
    if not meta:
        return None
    cells, _n = quiet(SP.choose_cells, meta, arrays, args.n_cells, args.min_mass)
    d = {"n_pictures": len(meta), "cells": [list(c) for c in cells],
         "types": sorted({m["type"] for m in meta})}

    # the budget
    spans = {s: [] for s in SL.SPANS}
    for m in meta:
        a = arrays.get(m["unit"], {}).get("spans")
        if a is not None:
            for i, s in enumerate(SL.SPANS):
                spans[s].append(float(np.nanmean(a[..., i])))
    d["budget"] = {s: float(np.mean(v)) for s, v in spans.items() if v}

    # per type: E_ring with a CI, and the full location profile at the selected cells
    d["per_type"] = []
    for t in d["types"]:
        e, prof, grids, area = {}, {loc: [] for loc in LOC_LABELS}, [], []
        for m in meta:
            if m["type"] != t or m.get("dev"):
                continue
            a = arrays.get(m["unit"], {}).get("stats")
            if a is None:
                continue
            gh, gw = m["grid"]
            sets = SL.named_sets(gh, gw)
            e[m["key"]] = SP.at_cells(a, "ring_share", cells) / m["ring_area_frac"]
            for loc, stat, key in SP.LOCATIONS:
                if loc not in prof:
                    continue
                one = 1.0 / (gh * gw)
                frac = one if key is None else sets[key].mean()
                if frac > 0:
                    prof[loc].append(SP.at_cells(a, stat, cells) / frac)
            grids.append(tuple(m["grid"]))
            area.append(m["ring_area_frac"])
        if not e:
            continue
        mm, lo, hi, n = SP.boot_mean(list(e.values()), args.n_boot)
        modal = max(set(grids), key=grids.count)
        d["per_type"].append({
            "type": t, "n": n, "grid": f"{modal[0]}x{modal[1]}",
            "ring_area": float(np.mean(area)), "E_ring": mm, "lo": lo, "hi": hi,
            "profile": {loc: (float(np.nanmean(v)) if v else None)
                        for loc, v in prof.items()}})
    d["per_type"].sort(key=lambda r: -r["E_ring"])

    # the radial cliff
    radial = {k: [] for k in ("ring", "depth1", "depth2", "deep")}
    for m in meta:
        if m.get("dev"):
            continue
        a = arrays.get(m["unit"], {}).get("stats")
        if a is None:
            continue
        gh, gw = m["grid"]
        sets = SL.named_sets(gh, gw)
        for k in radial:
            f = sets[k].mean()
            if f > 0:
                radial[k].append(SP.at_cells(a, f"{k}_share", cells) / f)
    d["radial"] = {k: float(np.nanmean(v)) for k, v in radial.items() if v}

    # the head-set x token-set grid, pooled and per type
    d["grid"] = {}
    d["grid_per_type"] = {}
    for tok_name, field in SP.TOKEN_SETS:
        for head_name, head_cells in SP.HEAD_SETS:
            rows = {}
            for m in meta:
                a = arrays.get(m["unit"], {}).get(field)
                if a is None:
                    continue
                gh, gw = m["grid"]
                sets = SL.named_sets(gh, gw)
                v = {}
                for loc, stat, key in SP.LOCATIONS:
                    if loc not in LOC_LABELS:
                        continue
                    area = 1.0 / (gh * gw) if key is None else sets[key].mean()
                    v[loc] = SP._enrich(a, head_cells, stat, area, args.min_mass)
                rows.setdefault(m["type"], []).append(v)
            if not rows:
                continue
            key = f"{tok_name}|{head_name}"
            d["grid_per_type"][key] = {
                t: {loc: float(np.nanmean([r[loc] for r in rs])) for loc in LOC_LABELS}
                for t, rs in rows.items()}
            d["grid"][key] = {
                loc: float(np.nanmean([v for t in d["grid_per_type"][key]
                                       for v in [d["grid_per_type"][key][t][loc]]]))
                for loc in LOC_LABELS}

    # S3, the blank test, the norms, the key split, the arms and the verdict
    facts = {"shape": quiet(SP.report_shape, meta, arrays, cells, args) or {}}
    facts["blank"] = quiet(SP.report_blank, meta, arrays, cells, args) or {}
    facts["content"] = quiet(SP.report_content, meta, arrays, args) or {}
    facts["norms"] = quiet(SP.report_norms, meta, arrays, args) or {}
    facts["keys"] = quiet(SP.report_key_split, meta, arrays, cells, args) or {}
    facts["arms"] = quiet(SP.report_arms, out_dir, cells, args) or {}
    # `report_shape` names its sets for the text report ("first", "last", "deep"); the page
    # names them for a reader ("topleft", "botright", "middle"). Aliased HERE rather than
    # renamed at the source, because the pre-registered hypothesis readers index the
    # report's own names and quietly return None for a key that moved.
    alias = {"first": "topleft", "last": "botright", "deep": "middle"}
    d["shape"] = {alias.get(k, k): (None if not np.isfinite(v) else float(v))
                  for k, v in facts["shape"].items()}
    d["facts"] = {k: {kk: (None if not np.isfinite(vv) else float(vv))
                      for kk, vv in v.items() if isinstance(vv, (int, float, np.floating))}
                  for k, v in facts.items() if k != "arms"}
    d["arms"] = {a: {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v))
                         else (bool(v) if isinstance(v, bool) else float(v)))
                     for k, v in vals.items()}
                 for a, vals in facts["arms"].items()}
    d["hypotheses"] = []
    for name, why, preds in SP.HYPOTHESES:
        rows = []
        for text, reader in preds:
            r = reader(facts)
            rows.append({"text": text, "met": None if r is None else bool(r[0]),
                         "evidence": "" if r is None else r[1]})
        d["hypotheses"].append({"name": name, "why": why, "preds": rows})

    # S3 -- is it a sink
    s3 = {"ring_cells": [], "best_cell": []}
    mass_i = SL.STAT_INDEX["image_mass"]
    for m in meta:
        if m.get("dev"):
            continue
        a = arrays.get(m["unit"], {}).get("stats")
        if a is None:
            continue
        s3["ring_cells"].append((SP.at_cells(a, "peak_uniform_x", cells),
                                 SP.at_cells(a, "peak_cv", cells)))
        mag = np.where(np.asarray(a[..., mass_i], dtype=float) >= args.min_mass,
                       np.asarray(a[..., SL.STAT_INDEX["peak_uniform_x"]], dtype=float),
                       np.nan)
        if np.isfinite(mag).any():
            l, h = divmod(int(np.nanargmax(mag)), a.shape[1])
            s3["best_cell"].append((float(a[l, h, SL.STAT_INDEX["peak_uniform_x"]]),
                                    float(a[l, h, SL.STAT_INDEX["peak_cv"]])))
    d["s3"] = {k: {"x": float(np.nanmean([r[0] for r in v])),
                   "cv": float(np.nanmean([r[1] for r in v]))}
               for k, v in s3.items() if v}
    d["thresholds"] = {"sink_x": args.sink_x, "sink_cv": args.sink_cv}
    return d


def build(cold, base, out):
    data = {"cold": collect(cold) if cold else None,
            "base": collect(base) if base else None,
            "loc_labels": LOC_LABELS,
            "token_sets": [t[0] for t in SP.TOKEN_SETS],
            "head_sets": [h[0] for h in SP.HEAD_SETS]}
    html = TEMPLATE.replace("__DATA__", json.dumps(data, allow_nan=False))
    Path(out).write_text(html, encoding="utf-8")
    return out


TEMPLATE = (REPO / "assets" / "sink_location_template.html").read_text() \
    if (REPO / "assets" / "sink_location_template.html").exists() else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cold", default="outputs/sink_location/coldstart")
    ap.add_argument("--base", default="outputs/sink_location/base_qwen3vl")
    ap.add_argument("--out", default="docs/sink-location.html")
    a = ap.parse_args()
    if not TEMPLATE:
        raise SystemExit("assets/sink_location_template.html is missing")
    print(f"written to {build(a.cold, a.base, a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
