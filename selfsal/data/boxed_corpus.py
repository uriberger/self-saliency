#!/usr/bin/env python
"""A sink-location corpus of pictures that carry a HUMAN box.

    python -m selfsal.data.boxed_corpus --out-dir DIR --n 1800

WHY A SECOND CORPUS. The twelve-image-type corpus was built to separate "the border" from
"the background", and boxes came along incidentally -- only 548 of its 1,800 pictures have
one, and eight of its twelve types have none at all. The question this corpus is for is a
different one:

    the model's attention sits on the BORDER (1.64x its share on Qwen3-VL)
    the human box sits in the CENTRE (0.36x on the border, measured on those 548)

and answering it needs pictures that have a box, in numbers, balanced across the sources
rather than across image types.

WHERE THE BOXES COME FROM. Visual-CoT, whose sub-datasets each ship one
answer-region box per question: GQA, Flickr30k, OpenImages, Visual7W, DocVQA, VSR,
TextCaps, V7W, InfographicsVQA, TextVQA, VisDrone and CUB. The boxes are normalised
[x0, y0, x1, y1] in 0-1, so `prepare_image`'s resize does not touch them -- which is the
one property that makes this corpus reusable at any resolution.

Rows are deduplicated by (source dataset, question id) across every set on disk, then
sampled in proportion to each source's share of the boxed pool, so the mix is Visual-CoT's
own rather than an artefact of which set happened to be largest.

THE UNIT IS A (PICTURE, QUESTION) PAIR, not a picture. Visual-CoT stores up to two
questions per image for several sources and the attention being measured depends on the
question, so a picture may legitimately appear twice with different boxes. The manifest
records the source and question id of each row so that can be checked, and `--one-per-image`
is there for the analysis that needs distinct pictures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from selfsal.data.paths import grpo_sets_dir

#: Every set on disk that can carry a box. The val_* splits are listed first so that a
#: row present in both a validation split and a training set is attributed to the
#: validation one -- which is what lets `--val-only` mean something later.
SETS = ("val_natural", "val_nonnatural", "val_c_natural", "val_c_nonnatural",
        "val_d_natural", "val_d_nonnatural", "val_e_natural", "val_e_nonnatural",
        "set_c", "set_d", "set_e", "set_a", "set_b")

#: Visual-CoT sources whose pictures are photographs. The rest (DocVQA, InfographicsVQA)
#: are documents, and the split is carried into the manifest because the border-vs-centre
#: contrast is expected to differ between them -- documents run ink to the margins.
NATURAL = {"gqa", "flickr30k", "openimages", "visual7w", "vsr", "textcap", "v7w",
           "textvqa", "visdrone", "cub"}

DEV_FRAC = 0.25


def collect(sets, verbose=True):
    """Every boxed row on disk, deduplicated by (source dataset, question id)."""
    from datasets import load_from_disk

    seen, pool = set(), []
    for name in sets:
        path = grpo_sets_dir() / name
        if not path.is_dir():
            continue
        try:
            ds = load_from_disk(str(path))
        except Exception as exc:
            if verbose:
                print(f"  {name}: unreadable ({type(exc).__name__})", flush=True)
            continue
        if hasattr(ds, "keys"):
            ds = ds["train"]
        if "bbox" not in ds.column_names:
            continue
        kept = 0
        cols = [c for c in ("dataset", "question_id", "bbox") if c in ds.column_names]
        for i, r in enumerate(ds.select_columns(cols)):
            b = r.get("bbox")
            if b in (None, "", "[]", "null"):
                continue
            key = (r["dataset"], str(r["question_id"]))
            if key in seen:
                continue
            seen.add(key)
            pool.append({"set": name, "dataset": r["dataset"],
                         "question_id": str(r["question_id"]), "row": i, "bbox": b})
            kept += 1
        if verbose:
            print(f"  {name:<18} {len(ds):>6} rows -> {kept:>6} new boxed", flush=True)
    return pool


def parse_box(b):
    """-> (x0, y0, x1, y1) in 0-1, or None. Refuses anything not a unit-square box."""
    if isinstance(b, str):
        try:
            b = json.loads(b)
        except json.JSONDecodeError:
            return None
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    x0, y0, x1, y1 = (float(v) for v in b)
    if not all(-1e-6 <= v <= 1 + 1e-6 for v in (x0, y0, x1, y1)):
        return None                       # not normalised: refuse rather than rescale
    x0, x1 = sorted((max(0.0, x0), min(1.0, x1)))
    y0, y1 = sorted((max(0.0, y0), min(1.0, y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--one-per-image", action="store_true",
                    help="at most one row per source picture; costs rows on the sources "
                         "that store two questions per image")
    ap.add_argument("--sets", default=",".join(SETS))
    args = ap.parse_args()

    from datasets import load_from_disk

    # The probes' image preparation, from the module that owns it. In the archive this
    # was reached through `overlap_probe.py`, which carried its own copy; both resize
    # BILINEAR, so this corpus is prepared exactly as the probes that read it are.
    # NOT the training resize, which is bicubic -- see prepare_image's docstring.
    from selfsal.data.prompt import prepare_image

    print("indexing boxed rows")
    pool = collect([s for s in args.sets.split(",") if s])
    by_src = {}
    for r in pool:
        by_src.setdefault(r["dataset"], []).append(r)
    total = sum(len(v) for v in by_src.values())
    print(f"\n{total} boxed rows over {len(by_src)} Visual-CoT sources")

    # Proportional allocation, largest-remainder, so the mix is the pool's own.
    rng = np.random.default_rng(args.seed)
    exact = {s: len(v) / total * args.n for s, v in by_src.items()}
    take = {s: min(len(by_src[s]), int(np.floor(e))) for s, e in exact.items()}
    for s in sorted(by_src, key=lambda k: -(exact[k] - np.floor(exact[k]))):
        if sum(take.values()) >= args.n:
            break
        if take[s] < len(by_src[s]):
            take[s] += 1

    out = Path(args.out_dir) / "corpus"
    (out / "images").mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    cache, n_written, skipped = {}, 0, 0
    print(f"\n{'source':<18} {'pool':>7} {'take':>6}")
    with open(manifest, "w") as fh:
        for src in sorted(by_src):
            rows = by_src[src]
            idx = rng.permutation(len(rows))[: take[src] * 3]     # headroom for skips
            print(f"{src:<18} {len(rows):>7} {take[src]:>6}", flush=True)
            kept, seen_img = 0, set()
            for j in idx:
                if kept >= take[src]:
                    break
                r = rows[int(j)]
                bb = parse_box(r["bbox"])
                if bb is None:
                    skipped += 1
                    continue
                setname = r["set"]
                if setname not in cache:
                    ds = load_from_disk(str(grpo_sets_dir() / setname))
                    cache[setname] = ds["train"] if hasattr(ds, "keys") else ds
                rec = cache[setname][int(r["row"])]
                try:
                    im = prepare_image(rec["image"])
                except Exception:
                    skipped += 1
                    continue
                if args.one_per_image:
                    h = hashlib.sha1(im.tobytes()).hexdigest()
                    if h in seen_img:
                        continue
                    seen_img.add(h)
                key = f"{src}-{kept:05d}"
                im.save(out / "images" / f"{key}.png")
                h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                fh.write(json.dumps({
                    "key": key, "type": src, "source": f"{setname}:{src}",
                    "ref": r["question_id"], "question": rec["problem"],
                    "image": f"images/{key}.png", "size": list(im.size),
                    "dev": h < DEV_FRAC, "bbox": list(bb),
                    "natural": src in NATURAL,
                }) + "\n")
                kept += 1
                n_written += 1
    print(f"\n{n_written} pictures under {out}   ({skipped} skipped: unreadable image "
          "or a box that is not a unit-square rectangle)")
    print("Every row carries `bbox` as normalised [x0, y0, x1, y1]; the image is resized "
          "by\nprepare_image and the box is resolution-free, so the two never drift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
