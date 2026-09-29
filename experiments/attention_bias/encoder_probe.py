#!/usr/bin/env python
"""Does the vision encoder mark extreme grid positions -- with the content held constant?

`docs/sink-location-cross-model.md` established that the border preference is positional
(A10), that in Qwen3-VL and InternVL it rides inside the patch embedding while in
LLaVA-1.5 it is held by the language model's slot (A9), and that it is an alignment effect
rather than a magnitude one. What none of those arms can say is WHY an extreme grid
coordinate is attractive, or whether LLaVA-1.5's encoder fails to write the mark or its
language model fails to read it. A9 cannot separate those: it only reports whether
attention travelled with the vector.

This separates them, and it needs no language model at all.

    Feed the encoder a UNIFORM image. Every patch then has identical content, so any
    difference between the patch embeddings it emits is POSITIONAL BY CONSTRUCTION --
    there is nothing else left for it to be.

Measured on the rows the language model would actually consume (after the merger /
pixel shuffle / projector, i.e. `vlm_family.Family.row_module`):

  deviation      ||x_i - xbar|| per patch, reported as ring / interior. Above 1 means the
                 encoder pushes border patches further from the average patch than
                 interior ones -- a positional mark, written with no content to explain it.
  separation     ||mean(ring) - mean(interior)|| over the per-patch spread. A large value
                 means "border" is a DIRECTION in feature space, not just scatter: there
                 is a single vector a query could aim at to select the border.
  profile        the deviation by grid row and column. A 2-D positional mark is a U on
                 both axes; anything else is not the story the ring result tells.
  content check  the same statistics on real pictures, where content and position are
                 confounded, so the uniform-image numbers can be read against them.

    python sink_encoder_probe.py --model M --out DIR [--images corpus/manifest.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(REPO))


from experiments.attention_bias import measure as SL          # noqa: E402
from experiments.trained_model import probe as PROBE          # noqa: E402
from selfsal.models import families as VF                     # noqa: E402


class RowTap:
    """Keep the actual rows the language model is handed, not just their norms."""

    def __init__(self, model, family):
        self.model, self.family, self.rows, self._h = model, family, None, []

    def _hook(self, module, args, out):
        pool = getattr(out, "pooler_output", None)
        t = pool if pool is not None and hasattr(pool, "detach") else out
        if hasattr(t, "detach"):
            self.rows = t.detach().float().reshape(-1, t.shape[-1]).cpu().numpy()
        return out

    def __enter__(self):
        self._h.append(self.family.row_module(self.model).register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []
        return False


def uniform_image(size, kind):
    from PIL import Image
    if kind == "noise":
        rng = np.random.default_rng(0)
        return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8),
                               "RGB")
    return Image.new("RGB", size, {"grey": (128,) * 3, "white": (255,) * 3,
                                   "black": (0,) * 3}[kind])


def positional_structure(rows, gh, gw):
    """-> dict. Every quantity here is about DISTANCE FROM THE AVERAGE PATCH.

    On a uniform image the average patch is what every patch would be if the encoder
    carried no position information at all, so the deviation from it is the mark, and
    nothing else is available to confound it.
    """
    x = np.asarray(rows, dtype=np.float64)[: gh * gw]
    if x.shape[0] != gh * gw:
        return None
    ring = SL.ring_set(gh, gw).reshape(-1)
    xbar = x.mean(0)
    dev = np.linalg.norm(x - xbar, axis=1)                     # [N]
    spread = float(dev.mean()) or 1e-30
    sep = float(np.linalg.norm(x[ring].mean(0) - x[~ring].mean(0)))
    d2 = dev.reshape(gh, gw)
    return {
        "dev_ring_over_interior": float(dev[ring].mean() / max(dev[~ring].mean(), 1e-30)),
        "separation_over_spread": sep / spread,
        "row_profile": (d2.mean(1) / spread).tolist(),
        "col_profile": (d2.mean(0) / spread).tolist(),
        "dev_by_depth": [float(dev[(SL.depth_map(gh, gw).reshape(-1) == k)].mean())
                         / spread for k in range(min(4, min(gh, gw) // 2 + 1))],
        "grid": [gh, gw],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", default=None, help="a corpus manifest, for the content check")
    ap.add_argument("--n-real", type=int, default=24)
    ap.add_argument("--size", default="512x512")
    args = ap.parse_args()

    import torch
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = PROBE.load_model(args.model, None, device, "sdpa")
    fam = VF.family_for(model, processor)
    W, H = (int(v) for v in args.size.lower().split("x"))
    out = {"model": args.model, "family": fam.name, "size": [W, H], "uniform": {},
           "real": None}
    print(f"\nencoder probe  model={args.model}  family={fam.name}", flush=True)

    def rows_for(im):
        with RowTap(model, fam) as tap, torch.no_grad():
            inputs = fam.build_inputs(processor, [im], "x", device)
            model(**inputs, use_cache=False)
        grids = SL.locate_image_runs(inputs["input_ids"], inputs, fam)[1]
        return tap.rows, grids[0][1], grids[0][2]

    print(f"\n  CONTENT HELD CONSTANT -- every patch is the same colour, so every")
    print(f"  difference between the emitted rows is positional by construction.")
    print(f"\n    {'image':<8} {'grid':>8} {'dev ring/interior':>19} "
          f"{'separation/spread':>19}")
    for kind in ("grey", "white", "black", "noise"):
        rows, gh, gw = rows_for(uniform_image((W, H), kind))
        st = positional_structure(rows, gh, gw)
        if st is None:
            print(f"    {kind:<8} could not be read on this family")
            continue
        out["uniform"][kind] = st
        print(f"    {kind:<8} {f'{gh}x{gw}':>8} {st['dev_ring_over_interior']:>19.3f} "
              f"{st['separation_over_spread']:>19.3f}")

    g = out["uniform"].get("grey")
    if g:
        print("\n  the grey image's deviation profile (1.00 = the average patch's "
              "deviation):")
        print("    row:", " ".join(f"{v:4.2f}" for v in g["row_profile"]))
        print("    col:", " ".join(f"{v:4.2f}" for v in g["col_profile"]))
        print("    by depth from the border:",
              " ".join(f"{v:4.2f}" for v in g["dev_by_depth"]))

    if args.images:
        rows_meta = [json.loads(l) for l in Path(args.images).read_text().splitlines() if l]
        base = Path(args.images).parent
        acc = []
        for r in rows_meta[: args.n_real]:
            im = Image.open(base / r["image"]).convert("RGB")
            rows, gh, gw = rows_for(im)
            st = positional_structure(rows, gh, gw)
            if st:
                acc.append(st)
        if acc:
            out["real"] = {
                "n": len(acc),
                "dev_ring_over_interior": float(np.mean(
                    [a["dev_ring_over_interior"] for a in acc])),
                "separation_over_spread": float(np.mean(
                    [a["separation_over_spread"] for a in acc])),
            }
            print(f"\n  CONTENT AND POSITION CONFOUNDED -- {len(acc)} real pictures, for "
                  "the uniform numbers to be read against:")
            print(f"    dev ring/interior {out['real']['dev_ring_over_interior']:.3f}   "
                  f"separation/spread {out['real']['separation_over_spread']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
