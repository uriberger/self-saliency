#!/usr/bin/env python
"""Reduce a `fig1_multistep.py` JSON to the numbers a Figure-1 caption has to survive.

A panel is one picture and proves nothing on its own. These are the three tables that say
whether the picture is representative:

  1. per step, inside the step's OWN referent -- does the map land on the thing the step
     names at all? Reported for every (model, map, referent) so the two rewarded heads and
     GLIMPSE are visibly different objects, and so the reward's own union is visibly a
     different question from a tight box.
  2. the crossover rate -- of the disjoint within-chain step pairs, how many move the
     right way. This is the aggregate the panel is an instance of.
  3. answers -- how often each model is right on the same rows, so "the baseline answers
     worse" is a rate rather than an anecdote, with the truncated chains counted
     separately because a chain cut off at --max-new-tokens has no answer to grade.

Pure CPU, no detector: everything is already in the JSON.

    python fig1_report.py --json outputs/fig1-multistep/all.json [--md docs/fig1-multistep.md]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def two_proportion_p(k1, n1, k2, n2) -> float:
    """Two-sided p for two independent proportions, normal approximation.

    scipy is not in every env this repo uses and a chi-square on four counts does not
    need it. n is in the hundreds to thousands here, so the approximation is not the
    weak link.
    """
    if min(n1, n2) == 0:
        return float("nan")
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan")
    z = (k1 / n1 - k2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def mannwhitney_p(a, b) -> tuple[float, float]:
    """-> (AUC of a over b, two-sided p). Ties get average ranks; normal approximation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = a.size, b.size
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    v = np.concatenate([a, b])
    order = v.argsort(kind="stable")
    ranks = np.empty(v.size)
    ranks[order] = np.arange(1, v.size + 1)
    _u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    auc = u / (n1 * n2)
    mu = n1 * n2 / 2
    # tie correction on the variance, or a map with many equal near-zero patches makes
    # the p optimistic
    tie = (cnt ** 3 - cnt).sum()
    n = n1 + n2
    sd = math.sqrt(n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1))))
    if sd == 0:
        return auc, float("nan")
    return auc, math.erfc(abs(u - mu) / sd / math.sqrt(2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True, action="append")
    ap.add_argument("--md", default=None, help="also write a markdown version here")
    args = ap.parse_args()

    blobs = [json.loads(Path(p).read_text()) for p in args.json]
    blob = blobs[0]
    steps = [s for b in blobs for s in b["steps"]]
    models = list(blob["models"])
    maps = blob["maps"]
    ours = blob["ours"]
    out = []

    def emit(line=""):
        print(line)
        out.append(line)

    runs = sorted({s["run"] for s in steps})
    n_img = len({(s["run"], s["sample"]) for s in steps})
    emit(f"# Figure-1 search: {n_img} (model, picture) chains over {', '.join(runs)}")
    emit()
    emit(f"ours = `{ours}`; maps = {', '.join(maps)}; "
         f"tight referent = boxes within {blob['cfg']['tight_score_frac']:.0%} of the "
         f"step's best Grounding-DINO score, each under "
         f"{blob['cfg']['tight_max_box_area']:.0%} of the image; the reward referent is "
         f"the union the overlap reward would have scored (threshold "
         f"{blob['cfg']['box_threshold']}, per-box cap {blob['cfg']['max_box_area']}).")
    emit()

    # ---- 1. per step ---------------------------------------------------------------
    emit("## 1. Per step, inside the step's own referent")
    emit()
    emit("| model | map | referent | steps | median v2 | median AUROC | AUROC > 0.5 | "
         "peak inside | median border mass |")
    emit("|---|---|---|---|---|---|---|---|---|")
    auroc_by = {}
    for mdl in models:
        for mp in maps:
            for ref in ("tight", "reward"):
                v2, au, pk, bd = [], [], [], []
                for s in steps:
                    if s["model"] != mdl:
                        continue
                    sc = (s.get("scores") or {}).get(ref, {}).get(mp)
                    if not sc:
                        continue
                    if sc["mean_in_v2"] is not None:
                        v2.append(sc["mean_in_v2"])
                    if sc["auroc"] is not None:
                        au.append(sc["auroc"])
                    pk.append(sc["peak_in"])
                    bd.append(s["border"][mp])
                if not au:
                    continue
                auroc_by[(mdl, mp, ref)] = au
                emit(f"| {mdl} | {mp} | {ref} | {len(au)} | {np.median(v2):.2f} | "
                     f"{np.median(au):.3f} | {np.mean([a > 0.5 for a in au]):.0%} | "
                     f"{np.mean(pk):.0%} | {np.median(bd):.0%} |")
    emit()
    emit("Chance is 1.00 for v2 and 0.500 for AUROC. Ours against base, on the tight "
         "referent (Mann-Whitney over steps):")
    emit()
    for mp in maps:
        a = auroc_by.get((ours, mp, "tight"))
        for other in models:
            if other == ours or (other, mp, "tight") not in auroc_by:
                continue
            auc, p = mannwhitney_p(a, auroc_by[(other, mp, "tight")])
            emit(f"- `{mp}`: P(a random {ours} step beats a random {other} step) = "
                 f"{auc:.3f}, p = {p:.2g}")
    emit()

    # ---- 1b. per benchmark ----------------------------------------------------------
    chains = [c for b in blobs for c in b.get("chains", [])]
    if chains:
        emit("## 1b. Per benchmark, and per whole chain")
        emit()
        emit("`clean` is the figure's actual claim: every grounded step of that chain "
             "above chance inside its own referent, over at least "
             f"{blob.get('chain_cfg', {}).get('min_chain_steps', '?')} scored steps and "
             f"{blob.get('chain_cfg', {}).get('min_chain_regions', '?')} disjoint places. "
             "It is harder for a model that writes SHORT chains, so read it next to the "
             "median-steps column rather than on its own.")
        emit()
        emit("| benchmark | model | chains | scored steps | median AUROC | AUROC > 0.5 | "
             "median steps | clean |")
        emit("|---|---|---|---|---|---|---|---|")
        for bench in sorted({c["dataset"] for c in chains}):
            for mdl in models:
                rows = [c for c in chains if c["dataset"] == bench and c["model"] == mdl]
                if not rows:
                    continue
                au = [p["auroc"] for c in rows for p in c["steps"]]
                emit(f"| {bench} | {mdl} | {len(rows)} | {len(au)} | "
                     f"{np.median(au):.3f} | {np.mean([a > 0.5 for a in au]):.0%} | "
                     f"{np.median([c['n_scored'] for c in rows]):.0f} | "
                     f"{sum(c['clean'] for c in rows)} |")
        emit()

    # ---- 2. crossover --------------------------------------------------------------
    emit("## 2. Crossover rate over disjoint within-chain step pairs")
    emit()
    emit(f"A pair qualifies when the two tight referents have grid IoU <= "
         f"{blob['cfg']['max_pair_iou']} and each covers <= "
         f"{blob['cfg']['max_referent_area']:.0%} of the patch grid. It *moves the right "
         f"way* when min over the two steps of (v2 in its own region - v2 in the other) "
         f"is positive: one step firing everywhere cannot carry the pair.")
    emit()
    emit("| map | model | pairs | move the right way | rate |")
    emit("|---|---|---|---|---|")
    rates = {}
    for key, r in blob["crossover_rates"].items():
        rates[(r["model"], r["map"])] = (r["pos"], r["n"])
    for mp in maps:
        for mdl in models:
            pos, n = rates.get((mdl, mp), (0, 0))
            emit(f"| {mp} | {mdl} | {n} | {pos} | "
                 + (f"{pos / n:.0%} |" if n else "n/a |"))
    emit()
    for mp in maps:
        po, no = rates.get((ours, mp), (0, 0))
        for other in models:
            if other == ours:
                continue
            pb, nb = rates.get((other, mp), (0, 0))
            if not (no and nb):
                continue
            emit(f"- `{mp}`: {ours} {po/no:.0%} vs {other} {pb/nb:.0%}, "
                 f"p = {two_proportion_p(po, no, pb, nb):.2g}")
    emit()

    # ---- 3. answers ----------------------------------------------------------------
    emit("## 3. Answers on the same pictures")
    emit()
    emit("One row per (model, picture), graded on the scan's own greedy completion -- "
         "not the benchmark's, which runs at a longer token budget and full image "
         "resolution. A chain with no end-of-turn token hit the cap and is counted "
         "as truncated rather than wrong.")
    emit()
    emit("| model | pictures | soft-correct | strict-correct | hit the token cap | "
         "`<think>` format |")
    emit("|---|---|---|---|---|---|")
    per_pic, truncated = {}, {}
    for s in steps:
        key = (s["model"], s["run"], s["sample"])
        if key in per_pic:
            continue
        per_pic[key] = s
        # No end-of-turn token means --max-new-tokens cut the chain off, so the model has
        # no answer to grade and "it got it wrong" would be a statement about the budget.
        gen = json.loads((Path(s["sdir"]) / "meta.json").read_text()).get("generation", "")
        truncated[key] = "<|im_end|>" not in gen
    for mdl in models:
        keys = [k for k in per_pic if k[0] == mdl]
        rows = [per_pic[k] for k in keys]
        soft = [bool((s.get("grade") or {}).get("soft")) for s in rows]
        strict = [bool((s.get("grade") or {}).get("strict")) for s in rows]
        fmt = [bool(s.get("format_ok")) for s in rows]
        emit(f"| {mdl} | {len(rows)} | {np.mean(soft):.0%} | {np.mean(strict):.0%} | "
             f"{np.mean([truncated[k] for k in keys]):.0%} | {np.mean(fmt):.0%} |")
    emit()
    common = sorted({(r, s) for (m, r, s) in per_pic if m == ours}
                    & {(r, s) for (m, r, s) in per_pic if m != ours})
    for other in models:
        if other == ours:
            continue
        both = [(per_pic[(ours, r, s)], per_pic[(other, r, s)], (r, s))
                for (r, s) in common if (other, r, s) in per_pic]
        if not both:
            continue
        ow = sum(1 for a, b, k in both
                 if (a["grade"] or {}).get("soft") and not (b["grade"] or {}).get("soft"))
        bw = sum(1 for a, b, k in both
                 if (b["grade"] or {}).get("soft") and not (a["grade"] or {}).get("soft"))
        ow_ok = sum(1 for a, b, k in both
                    if (a["grade"] or {}).get("soft") and not (b["grade"] or {}).get("soft")
                    and not truncated[(other, *k)])
        emit(f"- vs `{other}` over {len(both)} shared pictures: {ours} only "
             f"{ow} ({ow_ok} with the baseline's chain complete), {other} only {bw}.")
    emit()

    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text("\n".join(out) + "\n")
        print(f"\n[out] {args.md}")


if __name__ == "__main__":
    main()
