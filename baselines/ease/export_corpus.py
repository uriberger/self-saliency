#!/usr/bin/env python3
# Copyright 2026 NVIDIA. Apache-2.0.
"""Export saliency-r1-8k into the raw layout EASE's converter reads.

EASE (arXiv:2605.30912) trains on its own evidence-box pools, which were never
released. We substitute our own corpus: `peterant330/saliency-r1-8k` already
ships a per-row `bbox`, so no annotation pipeline is needed at all -- every one
of its 8,080 rows carries a box, in all ten source corpora (verified: 0 empty,
0 unparseable, 0 degenerate).

This script writes the *raw* half of the pipeline. Their
`scripts/prepare_ease_dataset.py` then turns it into `train.parquet` /
`val.parquet`; we deliberately do not reimplement that step, so the box ->
pixel conversion, the `<image>` prefix, the `reward_model` block and the split
all stay theirs.

    saliency-r1-8k (HF)  --[this script]-->  raw/<source>/*.parquet + images/
    raw/ --[their prepare_ease_dataset.py]-->  train.parquet, val.parquet

Layout written under --out-dir:

    images/<ab>/<sha256>.<ext>        deduplicated by content hash
    raw/<source>/data-00000-of-00001.parquet

with one parquet directory per source corpus (flickr30k, gqa, ...) so that
their converter's `--datasets` flag gives each row a per-corpus `data_source`.

## The one transformation that is not a copy

`bbox` is a JSON string of four floats normalized to [0, 1] -- a union of the
source corpus's boxes, so exactly one box per row. Their `box_to_pixels`
decides normalized-vs-pixel by `max(abs(coords)) <= 1.0`, and 31 of our rows
carry a coordinate just outside that, up to 1.334, from `round(x, 3)` at the
image edge upstream. Left alone those 31 would be read as *pixel* coordinates
and collapse into a sub-pixel box in the top-left corner -- silently, since the
result is still a valid non-degenerate box. So we clamp into [0, 1] here and
report how many rows it touched.

Usage:
    python3 export_saliency_r1_8k_for_ease.py --out-dir cold_data/ease/saliency_r1_8k
    python3 export_saliency_r1_8k_for_ease.py --limit 64 --out-dir /tmp/ease_smoke
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from io import BytesIO
from pathlib import Path

# Extensions we are willing to hand to PIL on the training side, keyed by the
# format PIL sniffs out of the original bytes. Anything else is re-encoded to
# PNG rather than written under a name that misdescribes it.
_FORMAT_EXT = {
    "JPEG": "jpg",
    "PNG": "png",
    "WEBP": "webp",
    "BMP": "bmp",
    "GIF": "gif",
    "TIFF": "tiff",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="peterant330/saliency-r1-8k")
    p.add_argument("--split", default="train")
    p.add_argument("--out-dir", type=Path, required=True, help="Root for images/ and raw/.")
    p.add_argument("--source-key", default="dataset", help="Column naming the origin corpus.")
    p.add_argument("--limit", type=int, default=0, help="Export only the first N rows (smoke tests).")
    p.add_argument("--overwrite", action="store_true", help="Rewrite image files that already exist.")
    return p.parse_args()


def clamp_unit_box(box: list[float]) -> tuple[list[float], bool]:
    """Clamp a normalized box into [0, 1]. Returns (box, was_clamped)."""
    clamped = [min(1.0, max(0.0, float(v))) for v in box]
    return clamped, clamped != [float(v) for v in box]


def image_bytes_and_ext(raw: dict) -> tuple[bytes, str, int, int]:
    """Return (bytes to write, extension, width, height) for one stored image.

    The bytes are copied verbatim whenever PIL recognises the container, so the
    file on disk is byte-identical to what the dataset shipped and its
    dimensions cannot drift from the ones we record in the parquet.
    """
    from PIL import Image

    data = raw["bytes"]
    if data is None:
        # `datasets` stores either inline bytes or a path; a path here means the
        # cache was built from loose files.
        with open(raw["path"], "rb") as fh:
            data = fh.read()

    with Image.open(BytesIO(data)) as img:
        fmt, width, height = img.format, img.width, img.height

    ext = _FORMAT_EXT.get(fmt or "")
    if ext is not None:
        return data, ext, width, height

    with Image.open(BytesIO(data)) as img:
        buf = BytesIO()
        img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue(), "png", width, height


def main() -> None:
    args = parse_args()

    import datasets
    import pandas as pd

    out_dir: Path = args.out_dir
    image_root = out_dir / "images"
    raw_root = out_dir / "raw"
    image_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.dataset}:{args.split}", flush=True)
    ds = datasets.load_dataset(args.dataset, split=args.split)
    # Undecoded, so the original encoded bytes reach us instead of a PIL object
    # that would have to be re-encoded (and could change the dimensions the
    # evidence mask is scaled against).
    ds = ds.cast_column("image", datasets.Image(decode=False))
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[load] {len(ds)} rows, columns: {ds.column_names}", flush=True)

    rows_by_source: dict[str, list[dict]] = collections.defaultdict(list)
    n_clamped = 0
    n_dropped = 0
    written_images: set[str] = set()
    sizes: list[tuple[int, int]] = []

    for i, row in enumerate(ds):
        if i and i % 1000 == 0:
            print(f"[export] {i}/{len(ds)}", flush=True)

        raw_box = row.get("bbox")
        if raw_box is None or not str(raw_box).strip():
            n_dropped += 1
            continue
        try:
            box = [float(v) for v in json.loads(raw_box)]
        except Exception:
            n_dropped += 1
            continue
        if len(box) != 4:
            n_dropped += 1
            continue

        box, was_clamped = clamp_unit_box(box)
        n_clamped += int(was_clamped)
        if box[2] <= box[0] or box[3] <= box[1]:
            n_dropped += 1
            continue

        data, ext, width, height = image_bytes_and_ext(row["image"])
        digest = hashlib.sha256(data).hexdigest()
        rel_path = f"{digest[:2]}/{digest}.{ext}"
        abs_path = image_root / rel_path
        if digest not in written_images:
            written_images.add(digest)
            sizes.append((width, height))
            if args.overwrite or not abs_path.exists():
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = abs_path.with_suffix(abs_path.suffix + ".tmp")
                tmp.write_bytes(data)
                os.replace(tmp, abs_path)

        source = str(row.get(args.source_key) or "unknown")
        rows_by_source[source].append(
            {
                "question": str(row["problem"]).strip(),
                "answer": str(row["solution"]).strip(),
                "image_path": rel_path,
                # A list, because EASE's target is a mixture over K boxes. Ours
                # is always K=1: the upstream `bbox` is already a union of the
                # source corpus's boxes, so the per-entity mixture EASE builds
                # for its multi-evidence pool has no counterpart here.
                "evidence_bboxes": [box],
                "sample_id": f"{source}-{row.get('question_id', i)}",
                "source_dataset": source,
                "source_split": str(row.get("split") or ""),
            }
        )

    for source, rows in sorted(rows_by_source.items()):
        target = raw_root / source
        target.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(target / "data-00000-of-00001.parquet", index=False)

    total = sum(len(r) for r in rows_by_source.values())
    print()
    print(f"{'source':<20}{'rows':>8}")
    for source, rows in sorted(rows_by_source.items(), key=lambda kv: -len(kv[1])):
        print(f"{source:<20}{len(rows):>8}")
    print(f"{'TOTAL':<20}{total:>8}")
    print()
    print(f"distinct images     {len(written_images)}  ({total / max(len(written_images), 1):.2f} questions/image)")
    if sizes:
        areas = sorted(w * h for w, h in sizes)
        print(f"image area          median {areas[len(areas) // 2]:,} px  "
              f"min {areas[0]:,}  max {areas[-1]:,}")
        print(f"long side           max {max(max(w, h) for w, h in sizes)}")
    print(f"boxes clamped       {n_clamped}   (coordinate outside [0,1] before clamping)")
    print(f"rows dropped        {n_dropped}")
    print()
    print(f"images -> {image_root}")
    print(f"raw    -> {raw_root}")
    print()
    print("Next, from ease_repo/:")
    print(f"  python3 scripts/prepare_ease_dataset.py --input_dir {raw_root} \\")
    print(f"      --output_dir {out_dir / 'parquet'} --image_root {image_root} \\")
    print(f"      --datasets {' '.join(sorted(rows_by_source))} ...")

    if total == 0:
        sys.exit("no rows exported")


if __name__ == "__main__":
    main()
