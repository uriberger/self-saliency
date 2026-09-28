#!/usr/bin/env python
"""
vga_grounding_probe.py — rung 1 of the wiki's ladder, the go/no-go.

    "The entire method rests on the premise that unembedding a *visual* token's
     hidden state yields a meaningful word distribution. That was shown for
     LLaVA-1.5 and Qwen2.5-VL; Qwen3-VL has a different vision stack and
     deepstack injection, and the premise is not guaranteed to survive. Measure
     it first ... If VSC does not beat attention here, stop — the rest will not
     work."   — wiki/vga-implementation.md

Four maps over the same patches, scored against the same ground-truth region:

  vsc       Visual Semantic Confidence — the object-directed map VGA injects.
  vss       Visual Semantic Salience — the object-agnostic entropy map, scored
            here too because the sign of its formula is unsettled (see
            salience_map() in vlm/vga.py) and this is the cheapest place to
            find out which way it points.
  attn      the model's own attention over the visual span. THE ARM THAT
            MATTERS: VGA's claim is that VSC grounds objects better than this.
  uniform   a constant map. Not a strawman — it fixes what each metric scores
            when a map knows nothing, which for Dice is the region's area
            fraction and is easy to mistake for a result.

and three metrics, because one is not enough:

  dice      the paper's metric, thresholded at the region's own size so both
            maps predict exactly as many patches as the region has. Dice then
            reduces to precision = recall and cannot be won on area.
  auroc     threshold-free and area-free, chance-corrected at 0.5. This repo's
            reward work moved to AUROC for exactly that reason.
  mean_in   the repo's peak-normalised in-region mean, for continuity with the
            overlap-reward numbers.

The region is the dataset's human key-region box (saliency_r1_8k, visual_cot),
which is what "the object the question is about" means here. The paper uses COCO
instance masks; those are not cached on this cluster and a box is a coarser
target, so read the absolute Dice as a floor and the VSC-vs-attn *difference* —
which is paired, same region, same patches — as the result.

Usage
-----
    conda run -n lmms_eval python analysis/vga_grounding_probe.py \
        --limit 500 --require-boxes --out results/analysis/vga_grounding.json

Verdict: VSC must beat attn on dice and auroc with a bootstrap CI clear of zero.
Anything else is a stop.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis import vga_common as common  # noqa: E402
from vlm import vga as vga_mod  # noqa: E402

ARMS = ("vsc", "vss", "attn", "uniform")
METRICS = ("dice", "auroc", "mean_in")


def _bootstrap_ci(diffs: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    """Percentile CI for the mean of a paired difference. Resamples samples, so
    it carries the pairing — every arm sees the same region on the same image."""
    if diffs.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    means = diffs[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def _sign_test(diffs: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided sign test; ties dropped. numpy-only (no scipy here)."""
    pos = int((diffs > 0).sum())
    neg = int((diffs < 0).sum())
    n = pos + neg
    if n == 0:
        return pos, neg, float("nan")
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return pos, neg, min(1.0, 2 * tail)


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p)
    p.set_defaults(limit=500, require_boxes=True)
    p.add_argument("--out", default=None, help="Write per-sample scores + summary here.")
    p.add_argument("--topk", type=int, default=10, help="K for the VSS entropy.")
    p.add_argument("--vss-invert", action="store_true",
                   help="Score 1 - normalised entropy instead. Run both; the sign is unsettled.")
    p.add_argument("--object-variants", default="both", choices=["both", "plain"])
    p.add_argument("--attn-layers", default=None,
                   help="Comma/range list limiting the attention baseline to these layers "
                        "(default: all). e.g. '4-16' to match VGA's own window.")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    attn_layers = None
    if args.attn_layers:
        attn_layers = []
        for part in args.attn_layers.split(","):
            if "-" in part:
                a, b = part.split("-")
                attn_layers.extend(range(int(a), int(b) + 1))
            else:
                attn_layers.append(int(part))

    model, processor = common.load(args, need_attention=True)
    tok = getattr(processor, "tokenizer", processor)
    img_id = vga_mod.image_token_id(model, processor)
    cfg = vga_mod.VGAConfig(topk=args.topk, vss_invert=args.vss_invert,
                            object_variants=args.object_variants, verbose=False)

    per_sample: list[dict] = []
    n_seen = n_no_object = n_no_grid = 0

    for sample in common.samples(args):
        n_seen += 1
        if not sample.get("boxes"):
            continue
        inputs = common.build_inputs(processor, model, sample["image"], sample["question"])
        span = vga_mod.visual_span(inputs["input_ids"], img_id)
        if span is None:
            continue
        grid = vga_mod.patch_grid(inputs, processor)
        m = span[1] - span[0]
        if grid is None or grid[0] * grid[1] != m:
            # A mask needs the patch layout; VGA itself never does.
            n_no_grid += 1
            continue

        mask = common.boxes_to_patch_mask(sample["boxes"], grid)
        if mask.sum() == 0 or mask.all():
            continue

        fwd = {k: v for k, v in inputs.items() if k != "input_ids"}
        pre = vga_mod.run_prefill(model, inputs["input_ids"], fwd, span, cfg)

        objects = vga_mod.extract_objects(sample["question"], cfg.max_objects)
        tids = [vga_mod.first_token_ids(tok, o, cfg.object_variants) for o in objects]
        tids = [t for t in tids if t]
        if not tids:
            n_no_object += 1

        maps = {
            "vss": vga_mod.salience_map(model, pre, cfg).float().cpu().numpy(),
            "attn": common.attention_baseline(model, inputs, span, attn_layers),
            "uniform": np.full(m, 1.0 / m),
        }
        if tids:
            maps["vsc"] = vga_mod.object_map(model, pre, tids, cfg).float().cpu().numpy()

        row = dict(sample_id=str(sample["sample_id"]), m=m, grid=list(grid),
                   objects=objects, region_frac=float(mask.mean()))
        for arm, g in maps.items():
            row[f"{arm}_dice"] = common.dice(g, mask)
            row[f"{arm}_auroc"] = common.auroc(g, mask)
            row[f"{arm}_mean_in"] = common.mean_in(g, mask)
        per_sample.append(row)

        del pre
        if len(per_sample) % 25 == 0:
            print(f"  {len(per_sample)} scored ({n_seen} seen)", flush=True)

    if not per_sample:
        print("nothing scored — does this dataset ship boxes? try --dataset saliency_r1_8k")
        return 1

    # ---------------------------------------------------------------- report
    n = len(per_sample)
    with_vsc = [r for r in per_sample if f"vsc_dice" in r]
    print(f"\n{args.model} on {args.dataset}: {n} scored of {n_seen} seen "
          f"({n_no_object} had no extractable object, {n_no_grid} had no usable patch grid)")
    print(f"region covers {np.mean([r['region_frac'] for r in per_sample]):.3f} of the patches "
          f"on average\n")

    print(f"{'arm':>8} {'n':>5} " + " ".join(f"{me:>18}" for me in METRICS))
    summary: dict = {}
    for arm in ARMS:
        rows = [r for r in per_sample if f"{arm}_dice" in r]
        if not rows:
            continue
        cells = []
        summary[arm] = {"n": len(rows)}
        for me in METRICS:
            v = np.array([r[f"{arm}_{me}"] for r in rows], dtype=float)
            v = v[~np.isnan(v)]
            mean = float(v.mean()) if v.size else float("nan")
            sem = float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else float("nan")
            summary[arm][me] = {"mean": mean, "sem": sem, "n": int(v.size)}
            cells.append(f"{mean:>10.4f}±{sem:<7.4f}")
        print(f"{arm:>8} {len(rows):>5} " + " ".join(cells))

    print(f"\npaired differences over the {len(with_vsc)} samples that have both arms "
          f"(95% bootstrap CI, {args.n_boot} resamples):")
    verdicts = {}
    for other in ("attn", "vss", "uniform"):
        for me in METRICS:
            d = np.array([r[f"vsc_{me}"] - r[f"{other}_{me}"] for r in with_vsc], dtype=float)
            d = d[~np.isnan(d)]
            lo, hi = _bootstrap_ci(d, args.n_boot, args.seed)
            pos, neg, pval = _sign_test(d)
            key = f"vsc-{other}.{me}"
            verdicts[key] = dict(mean=float(d.mean()) if d.size else float("nan"),
                                 ci=[lo, hi], wins=pos, losses=neg, sign_p=pval,
                                 n=int(d.size))
            star = "" if (lo <= 0 <= hi) else "  *"
            print(f"  vsc − {other:<8} {me:<8} {d.mean():+.4f}  "
                  f"[{lo:+.4f}, {hi:+.4f}]  {pos}W/{neg}L  sign p={pval:.2g}{star}")

    beats = all(verdicts[f"vsc-attn.{me}"]["ci"][0] > 0 for me in ("dice", "auroc"))
    print("\n" + "=" * 72)
    if not with_vsc:
        print("VERDICT: no sample produced a VSC map — object extraction failed everywhere. "
              "Fix extraction (or pass objects explicitly) before reading this as a "
              "statement about the model.")
    elif beats:
        print("VERDICT: GO. VSC grounds the question's object better than the model's own "
              "attention on both dice and auroc, with the CI clear of zero. The premise "
              "survives the port to Qwen3-VL; continue to rung 2.")
    else:
        print("VERDICT: STOP, per the wiki. VSC does not beat the attention baseline on "
              "both dice and auroc with a CI clear of zero. Before abandoning it, check "
              "the two things that would fake this result: object extraction quality "
              "(the `objects` column) and the box→patch mask (--limit 5 with a "
              "visualisation).")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(
            model=args.model, dataset=args.dataset, n_scored=n, n_seen=n_seen,
            n_no_object=n_no_object, n_no_grid=n_no_grid,
            vss_invert=args.vss_invert, topk=args.topk,
            object_variants=args.object_variants, attn_layers=attn_layers,
            summary=summary, paired=verdicts, go=bool(beats and with_vsc),
            per_sample=per_sample), indent=2, default=float))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
