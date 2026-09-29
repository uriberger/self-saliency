#!/usr/bin/env python
"""Ground each dataset row ONCE, on its question, and write the boxes for training to read.

This builds the input file the `question_boxes` arm of Section 5.3 trains against:

    python training/grpo/precompute_question_boxes.py \
        --dataset peterant330/saliency-r1-8k \
        --out data/question_boxes/saliency_r1_8k_bt0.10.json

    bash training/grpo/run.sh question_boxes

R_sal normally calls Grounding-DINO once per observe step, on that step's own sentence.
The ablation replaces that with one call per dataset ROW, on the row's QUESTION, so every
observe step of a row is scored against the same region -- which is what prior work does,
target regions fixed by the image and question alone. The gap to SELF-SALIENCY is then the
value of conditioning the region on the chain the policy is generating. Doing it offline
is not just an optimisation: it is what lets the arm train with no detector at all, which
is why `training/grpo/config.needs_detector` returns False for it and the launcher starts
no Grounding-DINO sidecar.

WHAT IS STORED is the RAW box list -- every box above --box-threshold, before any area
filter -- so `--max_box_area` and `--max_union_area` stay run-time knobs. `--box-threshold`
is applied inside the detector and cannot be re-applied later, so it is recorded and the
trainer REFUSES a file built at a different one. So is the image cap: the detector sees a
different picture at a different resolution. Those two refusals are what pin the arm's
settings after the fact -- the published file records 0.10 and 512, so the published run
cannot have used anything else (see `training/grpo/configs/question_boxes.yaml`).

Rows are keyed by (dataset, split, question_id). The question TEXT is not part of the key:
questions repeat across images, and keying on the text would collapse different pictures
onto one box list. `selfsal.data.question_boxes` owns that key and the file version, and
is imported by both this script and the reward, so the writer and the reader cannot drift.

One GPU. Roughly 20 groundings/second on an A100, so about 7 minutes for saliency-r1-8k;
`--shard`/`--num-shards` splits it across cards and `--merge` puts the pieces back:

    python training/grpo/precompute_question_boxes.py --dataset ... --out out.shard0.json \
        --shard 0 --num-shards 8
    python training/grpo/precompute_question_boxes.py --merge out.shard*.json --out out.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from selfsal.data.prompt import MAX_IMAGE_SIDE
from selfsal.data.question_boxes import QBOX_KEY_COLUMNS, QBOX_VERSION, qbox_key
from selfsal.data.saliency_r1_8k import load_corpus
from selfsal.grounding import DEFAULT_BOX_THRESHOLD, GROUNDING_DINO_HF_ID, box_area, ground, union_mask

# Qwen3-VL's vision tower: 16px patches merged 2x2, so one cell of the grid the reward
# scores on is 32px of the (already resized) image. Recorded per row for the SUMMARY ONLY
# -- the reward reads the true grid off the step's own attention map and never looks at
# this. It is here because the alternative, a fixed nominal grid, biases the union
# fraction: rasterisation gives every box at least one row and column, so a coarser grid
# reports MORE coverage for the same boxes, and the summary could not then be compared
# with the per-step numbers it exists to be compared with. Checked against the probe's
# stored grids: a 500x332 image gives (10, 16), which is what the probe recorded.
GRID_CELL_PX = 32


def prepare_image(image):
    """The trainer's image preparation, which is NOT `selfsal.data.prompt.prepare_image`.

    BICUBIC, because that is what `training/grpo/trl_patch/grpo_vlm_qwen3.py` does and
    these boxes are only the boxes the run would have got if the detector sees the same
    picture the policy does. The shared helper in `selfsal.data.prompt` is the PROBES'
    resize and is bilinear; using it here would ground a different picture from the one
    trained on, silently. docs/provenance.md has the measurement and the history.

    MAX_IMAGE_SIDE is imported rather than retyped, and is written into the file and
    checked back at load, so the two copies of the cap cannot drift even though the two
    copies of the filter deliberately differ.
    """
    from PIL import Image

    width, height = image.size
    if max(width, height) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(width, height)
        image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.BICUBIC,
        )
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def patch_grid(image) -> list[int]:
    """(gh, gw) the reward will score on. Diagnostic only -- see GRID_CELL_PX."""
    w, h = image.size
    return [max(1, round(h / GRID_CELL_PX)), max(1, round(w / GRID_CELL_PX))]


# ---------------------------------------------------------------------------
def build(args) -> dict:
    # The same loader the trainer uses. The 100-row carve is deliberately NOT applied:
    # it holds out rows the run never trains on, but the file costs nothing to build
    # for them and one covering the whole corpus survives a change of seed or size.
    ds = load_corpus(args.dataset, split=args.dataset_split)

    for col in (*QBOX_KEY_COLUMNS, args.text_column, "image"):
        if col not in ds.column_names:
            raise SystemExit(
                f"{args.dataset}: no `{col}` column (has {ds.column_names}). "
                f"The file is keyed by {', '.join(QBOX_KEY_COLUMNS)} and grounds "
                f"`{args.text_column}`."
            )

    # Every row of the corpus, then this shard's slice of it. Sharded by STRIDE rather
    # than by block so each shard sees the same mix of source datasets and the timing of
    # one shard predicts the rest.
    idx = list(range(len(ds)))
    if args.limit:
        idx = idx[: args.limit]
    mine = [i for i in idx if i % args.num_shards == args.shard]

    # Two different batch sizes, and conflating them is what makes every call OOM once:
    # --rows-per-call is how many rows are decoded and handed over at a time, and
    # --batch-size is how many of those Grounding-DINO forwards together. `ground` halves
    # its own batch on OOM, so it only has room to do that if it is given more than one
    # batch's worth.
    print(f"[question_boxes] {args.dataset}: {len(ds)} rows, shard {args.shard}/"
          f"{args.num_shards} takes {len(mine)}; box_threshold={args.box_threshold} "
          f"max_image_side={MAX_IMAGE_SIDE} dino_batch={args.batch_size} "
          f"rows_per_call={args.rows_per_call}", flush=True)

    out: dict[str, list] = {}
    grids: dict[str, list] = {}
    done = 0
    for start in range(0, len(mine), args.rows_per_call):
        rows = [ds[i] for i in mine[start:start + args.rows_per_call]]
        images = [prepare_image(r["image"]) for r in rows]
        # An empty question would ground the whole frame; "object" is the archive's
        # fallback and keeps the row in the file rather than silently dropping it.
        texts = [r[args.text_column] or "object" for r in rows]
        boxes = ground(images, texts, box_threshold=args.box_threshold,
                       batch_size=args.batch_size, device=args.device)
        for r, im, b in zip(rows, images, boxes):
            key = qbox_key(*(r[c] for c in QBOX_KEY_COLUMNS))
            if key in out:
                raise SystemExit(
                    f"{args.dataset}: duplicate row key {key!r}. The file is a mapping, "
                    "so two rows sharing a key would silently share one box list."
                )
            out[key] = [[round(float(v), 5) for v in box] for box in (b or [])]
            grids[key] = patch_grid(im)
        done += len(rows)
        print(f"[question_boxes] {done}/{len(mine)}", flush=True)

    return {
        "grids": grids,
        "version": QBOX_VERSION,
        "config": {
            "box_threshold": float(args.box_threshold),
            "max_image_side": int(MAX_IMAGE_SIDE),
            "dino_hf_id": GROUNDING_DINO_HF_ID,
            "text_column": args.text_column,
            "dataset": args.dataset,
            "dataset_rows": len(ds),
            "key_columns": list(QBOX_KEY_COLUMNS),
        },
        "shard": {"shard": args.shard, "num_shards": args.num_shards, "rows": len(mine)},
        "boxes": out,
    }


def merge(paths):
    """Combine shard files. Their configs must agree, or the halves are not comparable."""
    merged, grids, cfg, version = {}, {}, None, None
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        if cfg is None:
            cfg, version = d["config"], d["version"]
        elif d["config"] != cfg or d["version"] != version:
            raise SystemExit(
                f"{p} was built with a different configuration than {paths[0]}:\n"
                f"  {json.dumps(d['config'], sort_keys=True)}\n"
                f"  {json.dumps(cfg, sort_keys=True)}"
            )
        for k, v in d["boxes"].items():
            if k in merged:
                raise SystemExit(f"{p}: key {k!r} already came from an earlier shard")
            merged[k] = v
        grids.update(d.get("grids") or {})
        print(f"[merge] {p}: {len(d['boxes'])} rows", flush=True)

    want = cfg.get("dataset_rows")
    if want is not None and len(merged) != want:
        raise SystemExit(
            f"merged {len(merged)} rows but the corpus has {want}. A shard is missing or "
            "was built with --limit; refusing to write a partial file, because the "
            "trainer treats a key it cannot find as a hard failure and would die "
            "mid-epoch on whichever rows happen to be absent."
        )
    return {"version": version, "config": cfg, "grids": grids, "boxes": merged}


# ---------------------------------------------------------------------------
def summarise(d) -> str:
    """What the run will actually see, so a bad file is visible before training on it."""
    boxes = d["boxes"]
    grids = d.get("grids") or {}
    n = len(boxes)
    raw = np.array([len(v) for v in boxes.values()], dtype=float)
    cap = 0.5  # the --max_box_area default; the run's own value may differ
    kept = np.array([sum(1 for b in v if box_area(b) <= cap) for v in boxes.values()],
                    dtype=float)

    # Union coverage on each row's own patch grid, which is the whole point of recording
    # it -- see GRID_CELL_PX. A row whose union swallows the grid rasterises to None and
    # the REWARD SKIPS IT, so it is counted here rather than folded in as 1.0.
    fracs, degenerate = [], 0
    for k, v in boxes.items():
        g = grids.get(k)
        if not g:
            continue
        m = union_mask([b for b in v if box_area(b) <= cap], g[0], g[1],
                       max_box_area=None, max_union_area=None)
        if m is None:
            degenerate += 1
            continue
        fracs.append(float(m.sum()) / m.size)
    fr = np.array(fracs, dtype=float)

    L = []
    P = L.append
    P(f"  rows                              {n}")
    P(f"  grounded nothing at all           {int((raw == 0).sum())}  "
      f"({(raw == 0).mean():.1%} -- these rows are MASKED by the reward, not scored 0)")
    P(f"  boxes per row (raw)               mean {raw.mean():.1f}, median {np.median(raw):.0f}")
    P(f"  boxes per row (after area<={cap})   mean {kept.mean():.1f}, median {np.median(kept):.0f}")
    if not fr.size:
        P("  union coverage                    (no per-row grids stored; rebuild to get it)")
        return "\n".join(L)
    P(f"  union covers the whole grid       {degenerate}  "
      f"({degenerate / n:.1%} -- also MASKED, like an ungroundable row)")
    P(f"  union coverage, per-row grid      mean {fr.mean():.3f}, "
      f"median {np.median(fr):.3f}, p10 {np.percentile(fr, 10):.3f}, "
      f"p90 {np.percentile(fr, 90):.3f}")
    P("  (the question mask's median coverage was measured at 0.568 against the per-step")
    P("   masks' 0.578 on a natural corpus -- a median far from that is a bug, not a")
    P("   finding. It is also the number that says the per-step call buys little.)")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="what --dataset_name would be given to the trainer")
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--text-column", default="problem",
                    help="the column grounded once per row (default: the question). The "
                         "arm's whole point is that this is the QUESTION and not the "
                         "chain, so changing it makes a different experiment")
    ap.add_argument("--out", required=True)
    ap.add_argument("--box-threshold", type=float, default=DEFAULT_BOX_THRESHOLD,
                    help="must match the run's --box_threshold; written into the file, "
                         "and the trainer refuses a mismatch")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="images Grounding-DINO forwards at once. 32 OOMs and halves "
                         "itself on an 80GB card at 512px, which costs a retry per call")
    ap.add_argument("--rows-per-call", type=int, default=256,
                    help="rows decoded and handed to the detector at a time. Larger than "
                         "--batch-size on purpose, so its OOM halving has room to work")
    ap.add_argument("--device", default=None, help="default: cuda if visible, else cpu")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="0 = the whole corpus")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="combine shard files into --out instead of grounding anything")
    args = ap.parse_args(argv)

    if args.merge:
        d = merge(args.merge)
    else:
        if not args.dataset:
            raise SystemExit("--dataset is required unless --merge is given")
        d = build(args)

    Path(os.path.dirname(os.path.abspath(args.out)) or ".").mkdir(parents=True, exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, args.out)          # atomic: a killed shard leaves no half-written file

    print("")
    print(f"[question_boxes] wrote {args.out}")
    if not d.get("shard") or d["shard"]["num_shards"] == 1:
        print(summarise(d))
    else:
        print(f"  shard {d['shard']['shard']}/{d['shard']['num_shards']}, "
              f"{len(d['boxes'])} rows. Merge the shards before training on them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
