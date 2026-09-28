#!/usr/bin/env python3
"""One page putting several models' saliency_viz runs side by side, sample by sample.

`saliency_viz.py --stage render` writes one index.html per run, which is the right unit
when there is one model. With three of them the question is no longer "what does this map
look like" but "what does THIS model look at that THAT one does not", and flipping between
three pages does not answer it. This script re-uses the PNGs those runs already wrote --
it renders nothing itself and needs no GPU -- and lays them out as:

    per sample:  the image, then the chain-level map of every model in a row
                 then, per model, its own chain and its own per-step strip

    python saliency_viz_compare.py \
        --run vanilla=outputs/saliency_viz/sviz-3models/vanilla-qwen3vl8b \
        --run overlap=outputs/saliency_viz/sviz-3models/overlap-wov0.4-2head-trmean \
        --out outputs/saliency_viz/sviz-3models/compare.html

Samples are matched by `row_index`, not by position, so a run that dropped a sample (a
malformed chain, an OOM) leaves a gap in its column instead of shifting every model below
it onto the wrong picture.

WHAT IS AND IS NOT COMPARABLE. Every run draws the same rows -- same seed, same dataset,
same images -- so the CHAIN-LEVEL row is like for like. The per-step strips are not: each
model wrote its own chain, so its step 2 is not the other's step 2, and there need not even
be the same number of them. They are here to read against that model's own text, which is
why they sit under it rather than in a shared grid.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

# docs/saliency-maps.md order, so a run scanned with several methods lays them out the
# way every other page in this repo does.
MAP_ORDER = ("glimpse", "grad", "direct", "rollout_mean", "rollout_wnorm")


def load_run(tag: str, root: Path):
    """-> {row_index: sample record}. Reads meta.json only; the PNGs are linked, not read."""
    out = {}
    for d in sorted((root / "samples").glob("sample_*")):
        mj = d / "meta.json"
        if not mj.exists():
            continue
        meta = json.loads(mj.read_text())
        out[int(meta["row_index"])] = {"dir": d, "meta": meta, "tag": tag}
    return out


def pick_map(runs, want: str) -> str:
    """The map to put in the chain-level row: the requested one if every run has it.

    Per RUN, the maps any of its samples drew -- not per sample. A sample the segmenter
    dropped has a meta.json and no PNGs at all, so intersecting over samples would make
    one such row empty the whole set and report that no map exists anywhere.
    """
    have = [{m for m in MAP_ORDER
             for rec in r["samples"].values() if (rec["dir"] / f"sal_{m}.png").exists()}
            for r in runs]
    common = set.intersection(*have) if have else set()
    if want in common:
        return want
    for m in MAP_ORDER:
        if m in common:
            print(f"[compare] no '{want}' map in every run; using '{m}'")
            return m
    raise SystemExit(f"no map is present in every run (asked for '{want}'). "
                     "Run --stage render first, or pass --map.")


def rel(target: Path, page: Path) -> str:
    return os.path.relpath(target, page.parent).replace(os.sep, "/")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", default=[], metavar="NAME=DIR",
                   help="a saliency_viz out-dir and the label to give it; repeat")
    p.add_argument("--out", default="", help="the page to write (default <common parent>/compare.html)")
    p.add_argument("--map", default="glimpse", help="which map goes in the chain-level row")
    p.add_argument("--title", default="saliency maps by model")
    p.add_argument("--no-steps", action="store_true", help="chain level only")
    args = p.parse_args()

    if len(args.run) < 2:
        raise SystemExit("give at least two --run NAME=DIR")
    runs = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run '{spec}' is not NAME=DIR")
        tag, d = spec.split("=", 1)
        root = Path(d).resolve()
        if not (root / "samples").is_dir():
            raise SystemExit(f"{root} has no samples/ -- is it a saliency_viz out-dir?")
        samples = load_run(tag, root)
        if not samples:
            raise SystemExit(f"{root}/samples holds no meta.json")
        runs.append({"tag": tag, "root": root, "samples": samples})

    page = Path(args.out) if args.out else (
        Path(os.path.commonpath([str(r["root"]) for r in runs])) / "compare.html")
    page.parent.mkdir(parents=True, exist_ok=True)
    mp = pick_map(runs, args.map)

    # The union, ordered: a row every model drew comes first in the natural order, and a
    # row only some drew still appears rather than being silently dropped.
    rows = sorted({ri for r in runs for ri in r["samples"]})
    e = html.escape
    n_all = sum(1 for ri in rows if all(ri in r["samples"] for r in runs))

    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{e(args.title)}</title>",
        "<style>body{background:#111;color:#ddd;font:13px/1.5 -apple-system,sans-serif;"
        "margin:24px}h2{margin:36px 0 4px;border-top:1px solid #333;padding-top:18px}"
        "img{border:1px solid #333;max-width:100%}pre{white-space:pre-wrap;background:#181818;"
        "padding:8px;border-radius:4px;margin:4px 0}.q{color:#9cf;font-size:15px}"
        ".grid{display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap}"
        ".cell{max-width:340px}.cell b{color:#fc9}.miss{color:#a55;font-style:italic}"
        ".m{margin:14px 0 0 0;border-left:2px solid #444;padding-left:12px}"
        ".s{margin:8px 0 8px 14px}details summary{cursor:pointer;color:#9cf}"
        "a{color:#9cf}</style>",
        f"<h1>{e(args.title)}</h1>",
        f"<p>map <b>{e(mp)}</b> &middot; {len(runs)} models &middot; {len(rows)} rows "
        f"({n_all} drawn by every model). Per-model pages: "
        + " &middot; ".join(f"<a href='{rel(r['root'] / 'index.html', page)}'>{e(r['tag'])}</a>"
                            for r in runs) + "</p>",
        "<p>The <b>chain-level</b> row is the map averaged over that model's own observe "
        "steps, on the same image for every model &mdash; that row is like for like. The "
        "per-step strips below it are not: each model wrote its own chain, so its step 2 "
        "is not the other's step 2.</p>",
    ]

    for ri in rows:
        present = [r for r in runs if ri in r["samples"]]
        head = present[0]["samples"][ri]["meta"]
        parts.append(f"<h2>row {ri} &mdash; {e(str(head.get('dataset')))}</h2>")
        parts.append(f"<p class='q'>{e(head.get('question', ''))}</p>")
        parts.append(f"<p>gold: <b>{e(str(head.get('gt_answer')))}</b></p>")

        cells = [f"<div class='cell'><b>image</b><br>"
                 f"<img src='{rel(present[0]['samples'][ri]['dir'] / 'original.png', page)}'></div>"]
        for r in runs:
            rec = r["samples"].get(ri)
            if rec is None:
                cells.append(f"<div class='cell'><b>{e(r['tag'])}</b><br>"
                             "<span class='miss'>not in this run</span></div>")
                continue
            png = rec["dir"] / f"sal_{mp}.png"
            n = len(rec["meta"].get("steps", []))
            why = rec["meta"].get("dropped")
            body = (f"<img src='{rel(png, page)}'><br>{n} observe step(s)" if png.exists()
                    else f"<span class='miss'>no map ({e(str(why or 'not rendered'))})</span>")
            cells.append(f"<div class='cell'><b>{e(r['tag'])}</b><br>{body}</div>")
        parts.append("<div class='grid'>" + "".join(cells) + "</div>")

        for r in runs:
            rec = r["samples"].get(ri)
            if rec is None:
                continue
            meta = rec["meta"]
            flag = ("" if meta.get("format_ok", True)
                    is True else " <i>(malformed completion)</i>")
            parts.append(f"<div class='m'><b>{e(r['tag'])}</b>{flag}"
                         f"<details><summary>chain "
                         f"({len(meta.get('steps', []))} observe steps)</summary>"
                         f"<pre>{e(meta.get('generation', ''))}</pre></details>")
            if not args.no_steps:
                for st in meta.get("steps", []):
                    sp = rec["dir"] / "steps" / f"step{st['index']:02d}" / f"sal_{mp}.png"
                    if not sp.exists():
                        continue
                    parts.append(
                        f"<div class='s'>step {st['index']} ({st['n_tokens']} tok): "
                        f"{e(st['text'])}<br><img src='{rel(sp, page)}' width='340'></div>")
            parts.append("</div>")

    page.write_text("\n".join(parts))
    print(f"[compare] {len(rows)} row(s), {len(runs)} model(s) -> {page}")


if __name__ == "__main__":
    main()
