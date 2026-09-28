#!/usr/bin/env python
"""
vga_layer_scan.py — pick VGA's ``start_layer`` on Qwen3-VL, instead of assuming it.

    "The layer window is the one thing that does not transfer. Upstream picks
     start_layer by the PAI heuristic — the layer where attention to BOS starts
     dominating. We can measure that directly with vlm/saliency.py: take the
     attention from the last prompt token to position 0, per layer, and start
     where it jumps."   — wiki/vga-implementation.md

What it measures, per layer, averaged over samples, for the last prompt token:

  bos          attention mass on position 0 (head-mean). PAI's quantity: the
               layer where this takes off is where the sink starts eating the
               distribution, and injecting below it fights the model's own
               early visual processing for no gain.
  visual       total mass on the visual span. Guidance can only redirect what
               is actually being spent on the picture.
  ratio        bos / visual. The cleanest single ordering: high means the token
               is looking at the sink instead of the image.

Qwen3-VL also injects vision-tower features at ``config.deepstack_visual_indexes``
(8/16/24 by default), so those layers already receive an extra visual signal;
they are marked in the table because they are a second reason not to assume the
Qwen2.5-VL window of 4-16 transfers.

This suggests a start_layer.  It does not choose one — the wiki asks for a small
grid over (start_layer, end_layer, beta) on a held-out split, and this narrows
where to centre it.

Usage
-----
    conda run -n lmms_eval python analysis/vga_layer_scan.py --limit 32 \
        --out results/analysis/vga_layer_scan.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from analysis import vga_common as common  # noqa: E402
from vlm import vga as vga_mod  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p)
    p.set_defaults(limit=32)
    p.add_argument("--out", default=None, help="Write the per-layer table as JSON here.")
    p.add_argument("--jump", type=float, default=2.0,
                   help="A layer 'jumps' when its bos/visual ratio exceeds this multiple of "
                        "the running median of the layers below it (default: 2.0).")
    args = p.parse_args()

    model, processor = common.load(args, need_attention=True)
    img_id = vga_mod.image_token_id(model, processor)
    deepstack = list(getattr(model.config, "deepstack_visual_indexes", []) or [])

    rows: dict[int, list[tuple[float, float]]] = {}
    n = 0
    for sample in common.samples(args):
        inputs = common.build_inputs(processor, model, sample["image"], sample["question"])
        span = vga_mod.visual_span(inputs["input_ids"], img_id)
        if span is None:
            continue
        per_layer = common.last_token_attention(model, inputs, span)
        for li, d in per_layer.items():
            rows.setdefault(li, []).append((d["bos"], d["visual_mass"]))
        n += 1
        if n % 8 == 0:
            print(f"  {n} samples", flush=True)

    if not rows:
        print("no samples produced attention — is the model eager?")
        return 1

    layers = sorted(rows)
    table = []
    for li in layers:
        bos = float(np.mean([b for b, _ in rows[li]]))
        vis = float(np.mean([v for _, v in rows[li]]))
        table.append(dict(layer=li, bos=bos, visual=vis,
                          ratio=bos / vis if vis > 0 else float("inf"),
                          deepstack=li in deepstack))

    print(f"\nQwen3-VL {args.model}, {n} samples, last prompt token")
    print(f"deepstack_visual_indexes = {deepstack}\n")
    print(f"{'layer':>5} {'bos':>9} {'visual':>9} {'bos/visual':>11}  ")
    jump_at = None
    ratios = []
    for r in table:
        mark = " <- deepstack" if r["deepstack"] else ""
        if jump_at is None and len(ratios) >= 3:
            med = float(np.median(ratios))
            if med > 0 and r["ratio"] > args.jump * med:
                jump_at = r["layer"]
                mark += "  <== jump"
        ratios.append(r["ratio"])
        print(f"{r['layer']:>5} {r['bos']:>9.4f} {r['visual']:>9.4f} "
              f"{r['ratio']:>11.3f}{mark}")

    print()
    if jump_at is None:
        print(f"No layer's bos/visual exceeded {args.jump}x the running median below it. "
              f"On this evidence the PAI heuristic does not pick a start_layer here; "
              f"sweep (start_layer, end_layer) directly.")
    else:
        print(f"PAI-style suggestion: start_layer = {jump_at} "
              f"(first layer whose bos/visual is >{args.jump}x the median below it).")
    print("This narrows the grid; it does not settle it. Sweep beta and end_layer "
          "on a held-out split before trusting any single window.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(model=args.model, dataset=args.dataset,
                                       n_samples=n, deepstack=deepstack,
                                       jump_threshold=args.jump, suggested_start=jump_at,
                                       layers=table), indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
