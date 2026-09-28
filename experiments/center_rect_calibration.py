#!/usr/bin/env python
"""If DINO's per-step union were a FIXED CENTRED RECTANGLE, how much would we lose?

The overlap reward calls a detector once per observe step, on that step's own sentence,
and scores the step's attention map against the union of the boxes that come back. That
detector is the whole cost of the reward and the whole reason it needs a serving GPU.
This script asks the cheapest possible question about it: replace the union with an
axis-aligned rectangle in the middle of the frame -- no detector, no sentence, no image
content at all -- and see what the numbers do.

Everything comes off disk. `overlap_probe.py --store-maps` already wrote, per observe
step: the DINO union raster (`mask_q`), the attention map quantised to its own peak
(`map_q`) with the peak alongside (`map_max`), the boxes (`boxes_kept`), and the patch
grid. No GPU, no model, no DINO call; a minute on a login node.

THE RECTANGLE.  Axis-aligned, centred on the grid, preserving the grid's aspect:

    rows = round(gh * sqrt(f)),  cols = round(gw * sqrt(f)),  clamped to [1, gh/gw]

so it covers a fraction f of the frame before rounding. Two variants:

    c_fixed    f = 0.565 for every step of every image. This is the deployable one:
               it depends on nothing but the grid.
    c_matched  f = that step's OWN union area fraction. Not deployable (it needs
               DINO to tell it how big to be), but it is the diagnostic that matters:
               with the area held equal, the IoU is measuring POSITION AND SHAPE only
               and cannot be flattered or punished by an area mismatch.

The rectangle is rasterised on the PATCH GRID, not kept as box geometry, because the
grid mask is what the metric actually scores -- the same reason `_union_mask` rasterises.

WHAT IS REPORTED.  Four rows on one scale. A step's union is compared against

    within      another step of the SAME chain          <- what per-step DINO buys
    same-image  a step of a different chain, same image  <- is it just the picture?
    diff-image  another image's boxes on this grid       <- the floor
    centre      the centred rectangle                    <- the detector-free stand-in

Raw IoU on its own says very little here: these unions are huge (the median step covers
well over half the grid) and two big blobs overlap a lot whatever they contain. So each
IoU is reported against two reference points and then as one number between them:

    chance     the IoU two INDEPENDENT masks of exactly those two sizes would score
    best       the IoU those two sizes score when the smaller sits inside the larger
    closeness  (IoU - chance) / (best - chance);  0 = no better than the sizes alone,
               1 = as close as masks of those sizes can be

Then the test that actually decides it, on the reward itself (`mean_in` by default):
per step, does the map score higher on its own DINO union than on the rectangle; and
per chain, does the reward built on the rectangle rank the 8 rollouts of a prompt the
way the real per-step-DINO reward did. GRPO subtracts the group mean, so the ranking is
all that survives -- a rectangle that reproduces the ranking reproduces the gradient.

REPRODUCING `step_box_similarity.py`.  The within / same-image / diff-image rows are the
same three populations that script reports, recomputed here from the same stored bytes
by independent code. The two control rows draw one random partner per within-chain pair,
so they are exact only at that script's seed and draw order, which `--validate` checks
against its published report (outputs/step_box_similarity/mean_in/report.txt):

    base_coldstart:  within 0.635 / same-image 0.637 / diff-image 0.449,
                     chance 0.383, best 0.727, on 30 images / 205 chains / 833 steps,
                     own-mask rank 0.526, DINO-once-per-chain rho 0.621, top-1 53.6%

Usage:

    python centre_box_probe.py outputs/overlap_probe/<dir>/probe_merged.json \
        --models base_coldstart,mean_in_saliency_r1_8k --validate
"""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import math
import os
import sys

import numpy as np

# The seed and the draw order below are `step_box_similarity.py`'s, so the two control
# rows land on the numbers already published rather than on a second sample of the same
# population. Nothing else in this script is random.
SEED = 20260903

# Area fraction of the fixed rectangle. Not tuned here: it is the mean union coverage
# the detector produces across these runs, so `c_fixed` gives away nothing on size.
F_FIXED = 0.565

# How many independent redraws for the numbers that depend on a random partner. The
# published report takes ONE draw; a spread over many says whether that draw mattered.
N_REDRAW = 200


# ---------------------------------------------------------------------------
# stored bytes -> arrays
# ---------------------------------------------------------------------------
def _u8(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.uint8)


def decode_mask(b64: str, gh: int, gw: int) -> np.ndarray:
    a = _u8(b64)
    if a.size != gh * gw:
        raise ValueError("mask size does not match the grid")
    return a.astype(bool).reshape(gh, gw)


def decode_map(b64: str, gh: int, gw: int, map_max: float) -> np.ndarray:
    """The absolute map: the stored byte is the patch's value as a fraction of the peak."""
    a = _u8(b64)
    if a.size != gh * gw:
        raise ValueError("map size does not match the grid")
    return a.astype(np.float64).reshape(gh, gw) * (float(map_max) / 255.0)


def rasterise(boxes, gh: int, gw: int):
    """Relative-coordinate boxes -> patch-grid union, or None if degenerate.

    Same rule as `overlap_rewards._union_mask`: every box claims at least one patch row
    and column. `boxes_kept` is already past the per-box area filter, so no filter here.
    Needed for the different-image control, where a step's boxes have to land on a grid
    they were never drawn for, and nowhere else.
    """
    if not boxes:
        return None
    m = np.zeros((gh, gw), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        r0 = max(0, int(y1 * gh))
        r1 = min(gh, max(r0 + 1, round(y2 * gh)))
        c0 = max(0, int(x1 * gw))
        c1 = min(gw, max(c0 + 1, round(x2 * gw)))
        m[r0:r1, c0:c1] = True
    n = int(m.sum())
    return None if n == 0 or n == gh * gw else m


# ---------------------------------------------------------------------------
# the rectangle
# ---------------------------------------------------------------------------
_RECT_CACHE: dict = {}


def centre_rect(gh: int, gw: int, f: float) -> np.ndarray:
    """Centred axis-aligned rectangle covering ~f of the grid, at the grid's aspect.

    Splitting the area equally between the two axes (sqrt(f) on each) keeps the shape of
    the frame, so the rectangle is not secretly a wide or tall band -- the comparison is
    against a neutral centre bias and nothing else. Rounding to whole patches on a grid
    this coarse (10x16 is typical) moves the realised area a few points off f; the
    realised mean is reported rather than assumed.
    """
    key = (gh, gw, round(float(f), 6))
    hit = _RECT_CACHE.get(key)
    if hit is not None:
        return hit
    s = math.sqrt(min(1.0, max(0.0, float(f))))
    rows = min(gh, max(1, int(round(gh * s))))
    cols = min(gw, max(1, int(round(gw * s))))
    m = np.zeros((gh, gw), dtype=bool)
    r0, c0 = (gh - rows) // 2, (gw - cols) // 2
    m[r0:r0 + rows, c0:c0 + cols] = True
    _RECT_CACHE[key] = m
    return m


# ---------------------------------------------------------------------------
# the similarity measures
# ---------------------------------------------------------------------------
def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = float(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum()) / u if u > 0 else float("nan")


def iou_chance(na: int, nb: int, n: int) -> float:
    """IoU of two INDEPENDENT masks covering na and nb of n patches.

    Ratio of expectations: expected intersection na*nb/n over expected union
    na+nb-na*nb/n. Closed form, so the reference point costs nothing and is exact.
    """
    if n <= 0:
        return float("nan")
    inter = na * nb / n
    den = na + nb - inter
    return inter / den if den > 0 else float("nan")


def iou_best(na: int, nb: int) -> float:
    """IoU when the smaller mask sits entirely inside the larger one."""
    hi = max(na, nb)
    return min(na, nb) / hi if hi > 0 else float("nan")


def closeness(a: np.ndarray, b: np.ndarray):
    """(IoU, chance IoU, best IoU, closeness on the 0-1 scale between the last two)."""
    na, nb, n = int(a.sum()), int(b.sum()), a.size
    o, c, m = iou(a, b), iou_chance(na, nb, n), iou_best(na, nb)
    span = m - c
    return o, c, m, ((o - c) / span if span > 1e-12 else float("nan"))


# ---------------------------------------------------------------------------
# the metric the reward uses (a copy, so this script needs no training env)
# ---------------------------------------------------------------------------
def m_mean_in(smap, mask):
    """mean of the max-normalised map inside the mask."""
    vmax = float(smap.max())
    m = smap / vmax if vmax > 0 else smap
    inside = m[mask]
    return float(inside.mean()) if inside.size else None


def m_mean_in_v2(smap, mask):
    inside = smap[mask]
    if inside.size == 0:
        return None
    den = float(smap.mean())
    return float(inside.mean()) / den if den > 0 else None


def m_auroc(smap, mask):
    v = np.asarray(smap, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    n_in = int(m.sum())
    n_out = v.size - n_in
    if n_in == 0 or n_out == 0:
        return None
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=np.float64)
    ranks[order] = np.arange(1, v.size + 1, dtype=np.float64)
    _u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size, dtype=np.float64)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    u = ranks[m].sum() - n_in * (n_in + 1) / 2.0
    return float(u / (n_in * n_out))


METRICS = {"mean_in": m_mean_in, "mean_in_v2": m_mean_in_v2, "auroc": m_auroc}


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    x, y = x[ok], y[ok]

    def rank(v):
        o = np.argsort(v, kind="stable")
        r = np.empty(v.size, dtype=float)
        r[o] = np.arange(v.size, dtype=float)
        _u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        s = np.zeros(cnt.size)
        np.add.at(s, inv, r)
        return (s / cnt)[inv]

    rx, ry = rank(x), rank(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
class Step:
    __slots__ = ("gh", "gw", "mask", "smap", "boxes", "union_frac", "stored")

    def __init__(self, rec):
        self.gh, self.gw = rec["grid"]
        self.mask = decode_mask(rec["mask_q"], self.gh, self.gw)
        self.smap = (decode_map(rec["map_q"], self.gh, self.gw, rec.get("map_max") or 0.0)
                     if rec.get("map_q") else None)
        self.boxes = rec.get("boxes_kept") or []
        self.union_frac = float(self.mask.sum()) / self.mask.size
        self.stored = {k: rec.get(k + "_raw") for k in METRICS}


def load_model(model_rec, min_steps=2):
    """-> [[ [Step, ...] per completion ] per image ].

    A step with no stored mask is dropped: DINO grounded nothing there, or the union
    swallowed the whole grid, and the reward did not score it either. A completion with
    fewer than two usable steps is dropped, because the within-chain pairing is what
    every population here is weighted by.
    """
    out = []
    for s in model_rec["samples"]:
        comps = []
        for c in s["completions"]:
            steps = []
            for rec in c.get("observe_steps") or []:
                if not rec.get("mask_q") or not rec.get("grid"):
                    continue
                try:
                    steps.append(Step(rec))
                except ValueError:
                    continue
            if len(steps) >= min_steps:
                comps.append(steps)
        if comps:
            out.append(comps)
    return out


# ---------------------------------------------------------------------------
# the four rows
# ---------------------------------------------------------------------------
def _row(rows):
    """[(iou, chance, best, closeness), ...] -> the summary line."""
    if not rows:
        return None
    a = np.asarray(rows, dtype=np.float64)
    return {
        "n": int(a.shape[0]),
        "iou": float(a[:, 0].mean()),
        "chance": float(a[:, 1].mean()),
        "best": float(a[:, 2].mean()),
        "closeness": float(np.nanmean(a[:, 3])),
        "iou_p10": float(np.percentile(a[:, 0], 10)),
        "iou_p50": float(np.percentile(a[:, 0], 50)),
        "iou_p90": float(np.percentile(a[:, 0], 90)),
        "frac_iou_ge_90": float((a[:, 0] >= 0.90).mean()),
    }


def similarity_pass(images, rng, f_fixed=F_FIXED, max_steps_per_comp=40):
    """Every within-chain pair, the two control pairings, and the two rectangles.

    The controls draw one partner per within-chain pair, so all three pairings are
    weighted by the same completions and a long chain cannot dominate one comparison
    without dominating the others.

    The rectangle rows are per STEP -- there is exactly one rectangle comparison a step
    can make, and nothing to draw. Because that reweights chains slightly relative to the
    pair rows (a chain with k steps contributes k here and k(k-1)/2 there), the same rows
    are also accumulated pair-weighted, and both are reported.
    """
    within, cross_comp, cross_img = [], [], []
    c_fix, c_mat = [], []          # per step
    c_fix_pw, c_mat_pw = [], []    # the same, pair-weighted

    flat = [(si, ci, k) for si, cs in enumerate(images)
            for ci, c in enumerate(cs) for k in range(len(c))]

    for si, comps in enumerate(images):
        for ci, comp in enumerate(comps):
            steps = comp[:max_steps_per_comp]
            pairs = list(itertools.combinations(range(len(steps)), 2))
            if not pairs:
                continue
            for i, j in pairs:
                within.append(closeness(steps[i].mask, steps[j].mask))

            for st in steps:
                c_fix.append(closeness(st.mask, centre_rect(st.gh, st.gw, f_fixed)))
                c_mat.append(closeness(st.mask, centre_rect(st.gh, st.gw, st.union_frac)))
            # A pair contributes the mean of its two steps, so the rectangle rows carry
            # exactly the pair weighting of the three pairings above.
            for i, j in pairs:
                for st in (steps[i], steps[j]):
                    c_fix_pw.append(closeness(st.mask, centre_rect(st.gh, st.gw, f_fixed)))
                    c_mat_pw.append(closeness(st.mask, centre_rect(st.gh, st.gw, st.union_frac)))

            # control A: same image, a step of a different rollout of the same prompt
            others = [c for cj, c in enumerate(comps) if cj != ci]
            if others:
                for _ in pairs:
                    a = steps[rng.integers(len(steps))]
                    oc = others[rng.integers(len(others))]
                    b = oc[rng.integers(len(oc))]
                    if a.mask.shape == b.mask.shape:
                        cross_comp.append(closeness(a.mask, b.mask))

            # control B: a different image's boxes, re-rasterised onto this grid
            for _ in pairs:
                a = steps[rng.integers(len(steps))]
                for _try in range(8):
                    tsi, tci, tk = flat[rng.integers(len(flat))]
                    if tsi == si:
                        continue
                    bm = rasterise(images[tsi][tci][tk].boxes, a.gh, a.gw)
                    if bm is None:
                        continue
                    cross_img.append(closeness(a.mask, bm))
                    break

    return {
        "within": _row(within),
        "cross_comp": _row(cross_comp),
        "cross_img": _row(cross_img),
        "centre_fixed": _row(c_fix),
        "centre_matched": _row(c_mat),
        "centre_fixed_pairweighted": _row(c_fix_pw),
        "centre_matched_pairweighted": _row(c_mat_pw),
    }


# ---------------------------------------------------------------------------
# does the reward notice?
# ---------------------------------------------------------------------------
def swap_pass(images, metric_name, rng, f_fixed=F_FIXED, max_steps_per_comp=25,
              n_redraw=N_REDRAW):
    """Rescore every step on the rectangle, and rebuild the chain reward on it.

    Per step: does the map score higher on its own DINO union than on the rectangle
    (0.5 = coin flip, ties counted half)? The same number against another step of the
    same chain is computed too -- that is the published `own_rank`, and it is the
    comparator the rectangle has to be read against.

    Per chain, five versions of the reward the trainer would have seen:
      true      each step on its own union (what happened)
      first     every step on the chain's FIRST union (DINO once per chain)
      other     every step on a union from a different rollout of the same image
                (DINO once per image, on somebody else's sentence) -- one random
                partner, so it is redrawn `n_redraw` times and reported as a spread
      centre    every step on the rectangle (no DINO at all)
      flat      every step on the WHOLE grid -- no mask at all, so the score collapses
                to mean(map)/max(map). Not a proposal, a floor: any rectangle result
                has to be read against it, because a chain ranking the rectangle
                reproduces but the whole grid reproduces too was never spatial.
    scored by the within-image rank correlation against `true`, plus whether the
    top-ranked rollout is still top-ranked. GRPO only ever sees that ranking.
    """
    fn = METRICS[metric_name]
    own_rank, own_vs_fix, own_vs_mat, own_vs_flat, gap = [], [], [], [], []
    groups = []            # per image: dict of arrays over the image's chains
    n_steps = 0

    for comps in images:
        v_true, v_first, v_fix, v_mat, v_flat, others_pool = [], [], [], [], [], []
        for ci, comp in enumerate(comps):
            steps = [st for st in comp if st.smap is not None][:max_steps_per_comp]
            if len(steps) < 2:
                continue
            n = len(steps)
            S = np.full((n, n), np.nan)
            for i, a in enumerate(steps):
                for j, b in enumerate(steps):
                    v = fn(a.smap, b.mask)
                    if v is not None:
                        S[i, j] = v
            diag = np.diag(S).copy()
            spread = float(np.nanstd(diag))

            fx = np.array([fn(a.smap, centre_rect(a.gh, a.gw, f_fixed)) for a in steps],
                          dtype=np.float64)
            mt = np.array([fn(a.smap, centre_rect(a.gh, a.gw, a.union_frac)) for a in steps],
                          dtype=np.float64)
            fl = np.array([fn(a.smap, np.ones((a.gh, a.gw), dtype=bool)) for a in steps],
                          dtype=np.float64)

            for i in range(n):
                if not np.isfinite(diag[i]):
                    continue
                off = np.delete(S[i], i)
                off = off[np.isfinite(off)]
                if off.size:
                    own_rank.append(float((off < diag[i]).sum() + 0.5 * (off == diag[i]).sum())
                                    / off.size)
                    gap.append((diag[i] - off.mean()) / spread if spread > 1e-12
                               else float("nan"))
                own_vs_fix.append(1.0 if diag[i] > fx[i] else (0.5 if diag[i] == fx[i] else 0.0))
                own_vs_mat.append(1.0 if diag[i] > mt[i] else (0.5 if diag[i] == mt[i] else 0.0))
                own_vs_flat.append(1.0 if diag[i] > fl[i] else (0.5 if diag[i] == fl[i] else 0.0))
                n_steps += 1

            v_true.append(float(np.nanmean(diag)))
            v_first.append(float(np.nanmean(S[:, 0])))
            v_fix.append(float(np.nanmean(fx)))
            v_mat.append(float(np.nanmean(mt)))
            v_flat.append(float(np.nanmean(fl)))
            others_pool.append((ci, steps))

        if len(v_true) >= 3:
            groups.append({"true": v_true, "first": v_first, "fix": v_fix, "mat": v_mat,
                           "flat": v_flat, "comps": comps, "pool": others_pool})

    def summarise(key):
        rho = [spearman(g["true"], g[key]) for g in groups]
        top = [float(np.argmax(g["true"]) == np.argmax(np.nan_to_num(g[key], nan=-1e18)))
               for g in groups]
        return float(np.nanmean(rho)), float(np.mean(top))

    out = {
        "metric": metric_name,
        "n_steps": n_steps,
        "n_groups": len(groups),
        "own_rank_mean": float(np.nanmean(own_rank)) if own_rank else float("nan"),
        "swap_gap_mean": float(np.nanmean(gap)) if gap else float("nan"),
        "own_beats_fixed": float(np.mean(own_vs_fix)) if own_vs_fix else float("nan"),
        "own_beats_matched": float(np.mean(own_vs_mat)) if own_vs_mat else float("nan"),
        "own_beats_flat": float(np.mean(own_vs_flat)) if own_vs_flat else float("nan"),
    }
    out["rho_first"], out["top1_first"] = summarise("first")
    out["rho_fixed"], out["top1_fixed"] = summarise("fix")
    out["rho_matched"], out["top1_matched"] = summarise("mat")
    out["rho_flat"], out["top1_flat"] = summarise("flat")

    # `other`: one random partner chain per chain, so it is a draw, not a number.
    rho_o, top_o = [], []
    for _ in range(n_redraw):
        r, t = [], []
        for g in groups:
            vals = []
            for ci, steps in g["pool"]:
                cand = [c for cj, c in enumerate(g["comps"]) if cj != ci]
                om = None
                if cand:
                    oc = cand[rng.integers(len(cand))]
                    fit = [st for st in oc if st.mask.shape == steps[0].mask.shape]
                    if fit:
                        om = fit[0].mask
                if om is None:
                    vals.append(float("nan"))
                    continue
                v = [fn(a.smap, om) for a in steps]
                v = [x for x in v if x is not None]
                vals.append(float(np.mean(v)) if v else float("nan"))
            r.append(spearman(g["true"], vals))
            t.append(float(np.argmax(g["true"]) ==
                           np.argmax(np.nan_to_num(vals, nan=-1e18))))
        rho_o.append(float(np.nanmean(r)))
        top_o.append(float(np.mean(t)))
    if rho_o:
        out["rho_other_mean"] = float(np.mean(rho_o))
        out["rho_other_p10"], out["rho_other_p90"] = [float(x) for x in
                                                      np.percentile(rho_o, [10, 90])]
        out["top1_other_mean"] = float(np.mean(top_o))
        out["top1_other_p10"], out["top1_other_p90"] = [float(x) for x in
                                                        np.percentile(top_o, [10, 90])]
    return out


# ---------------------------------------------------------------------------
# self-check on the decode
# ---------------------------------------------------------------------------
def verify_decode(images, metric_name):
    """Largest disagreement between the metric recomputed here and the stored one.

    The map is stored quantised to 8 bits, so a small error is expected and a large one
    means the decode is wrong. `step_box_similarity` published 0.00093 for mean_in on
    base_coldstart.
    """
    fn = METRICS[metric_name]
    worst = 0.0
    for comps in images:
        for comp in comps:
            for st in comp:
                ref = st.stored.get(metric_name)
                if st.smap is None or ref is None:
                    continue
                got = fn(st.smap, st.mask)
                if got is not None:
                    worst = max(worst, abs(got - float(ref)))
    return worst


# ---------------------------------------------------------------------------
def analyse(tag, images, metric_name, f_fixed, n_redraw, verify=False):
    rng = np.random.default_rng(SEED)
    res = {
        "model": tag,
        "n_images": len(images),
        "n_completions": sum(len(cs) for cs in images),
        "n_steps": sum(len(c) for cs in images for c in cs),
        "f_fixed": f_fixed,
    }
    sim = similarity_pass(images, rng, f_fixed=f_fixed)
    res.update(sim)
    res["n_within_pairs"] = sim["within"]["n"] if sim["within"] else 0

    fr = [st.union_frac for cs in images for c in cs for st in c]
    res["union_frac_p50"] = float(np.median(fr)) if fr else float("nan")
    res["union_frac_mean"] = float(np.mean(fr)) if fr else float("nan")
    rects = [centre_rect(st.gh, st.gw, f_fixed).mean()
             for cs in images for c in cs for st in c]
    res["rect_frac_mean"] = float(np.mean(rects)) if rects else float("nan")
    shapes = {}
    for cs in images:
        for c in cs:
            for st in c:
                shapes[(st.gh, st.gw)] = shapes.get((st.gh, st.gw), 0) + 1
    gh, gw = max(shapes, key=shapes.get)
    r = centre_rect(gh, gw, f_fixed)
    res["modal_grid"] = [gh, gw]
    res["modal_rect"] = [int(r.any(axis=1).sum()), int(r.any(axis=0).sum())]

    # A fresh stream for the redraws: the numbers above must not move when n_redraw does.
    res["swap"] = swap_pass(images, metric_name, np.random.default_rng(SEED + 1),
                            f_fixed=f_fixed, n_redraw=n_redraw)
    if verify:
        res["verify_max_abs_err"] = verify_decode(images, metric_name)
    return res


# ---------------------------------------------------------------------------
# published constants, for --validate
# ---------------------------------------------------------------------------
PUBLISHED = {
    "base_coldstart": {
        "n_images": 30, "n_completions": 205, "n_steps": 833, "n_within_pairs": 1651,
        "within": (0.635, 0.383, 0.727, 0.721),
        "cross_comp": (0.637, 0.387, 0.727, 0.704),
        "cross_img": (0.449, 0.386, 0.700, 0.235),
        "own_rank_mean": 0.526, "rho_first": 0.621, "top1_first": 0.536,
        "rho_other": 0.506, "top1_other": 0.357,
    },
    "mean_in_saliency_r1_8k": {
        "n_images": 28, "n_completions": 120, "n_steps": 357, "n_within_pairs": 508,
        "within": (0.687, 0.414, 0.777, 0.728),
        "cross_comp": (0.709, 0.433, 0.790, 0.772),
        "cross_img": (0.450, 0.397, 0.716, 0.192),
        "own_rank_mean": 0.515, "rho_first": 0.781, "top1_first": 0.714,
        "rho_other": 0.294, "top1_other": 0.524,
    },
}


def validate(results):
    """Reproduce step_box_similarity's published rows before anything here is believed."""
    ok = True
    L = ["VALIDATION against outputs/step_box_similarity/mean_in/report.txt",
         "(the two control rows are one random draw there; same seed, same draw order here)",
         ""]
    for res in results:
        key = res["model"].split("::")[-1]
        pub = PUBLISHED.get(key)
        if not pub:
            continue
        L.append(f"  {key}")
        for fld in ("n_images", "n_completions", "n_steps", "n_within_pairs"):
            good = res[fld] == pub[fld]
            ok &= good
            L.append(f"    {fld:<16} {res[fld]:>6}   published {pub[fld]:>6}   "
                     f"{'ok' if good else 'MISMATCH'}")
        for tag, label in (("within", "within"), ("cross_comp", "same-image"),
                           ("cross_img", "diff-image")):
            got = (res[tag]["iou"], res[tag]["chance"], res[tag]["best"],
                   res[tag]["closeness"])
            good = all(abs(g - p) < 5e-4 for g, p in zip(got, pub[tag]))
            ok &= good
            L.append(f"    {label:<16} IoU {got[0]:.3f} chance {got[1]:.3f} "
                     f"best {got[2]:.3f} close {got[3]:.3f}   published "
                     + " ".join(f"{p:.3f}" for p in pub[tag])
                     + f"   {'ok' if good else 'MISMATCH'}")
        sw = res["swap"]
        for fld, label in (("own_rank_mean", "own-mask rank"), ("rho_first", "rho once/chain"),
                           ("top1_first", "top-1 once/chain")):
            good = abs(sw[fld] - pub[fld]) < 5e-4
            ok &= good
            L.append(f"    {label:<16} {sw[fld]:.3f}   published {pub[fld]:.3f}   "
                     f"{'ok' if good else 'MISMATCH'}")
        L.append(f"    {'rho once/image':<16} {sw['rho_other_mean']:.3f} "
                 f"[{sw['rho_other_p10']:.3f}, {sw['rho_other_p90']:.3f}] over redraws   "
                 f"published (1 draw) {pub['rho_other']:.3f}")
        L.append(f"    {'top-1 once/image':<16} {sw['top1_other_mean']:.3f} "
                 f"[{sw['top1_other_p10']:.3f}, {sw['top1_other_p90']:.3f}] over redraws   "
                 f"published (1 draw) {pub['top1_other']:.3f}")
        if "verify_max_abs_err" in res:
            L.append(f"    {'decode err':<16} {res['verify_max_abs_err']:.5f}   "
                     f"(8-bit map, small is expected)")
        L.append("")
    L.append("  ALL MATCHED" if ok else "  *** MISMATCH -- do not read anything below ***")
    return ok, "\n".join(L)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def ff_lbl(res):
    return f"f={res['f_fixed']:.3f}"


def render(results, metric_name):
    W = 96
    L = ["=" * W,
         "IF DINO'S PER-STEP UNION WERE A FIXED CENTRED RECTANGLE, HOW MUCH WOULD WE LOSE?",
         "=" * W, "",
         "  IoU        overlap of two masks: shared patches / patches in either.",
         "  chance     the IoU two INDEPENDENT masks of those two sizes would get.",
         "  best       the IoU those two sizes get when one sits inside the other.",
         "  closeness  (IoU - chance) / (best - chance).  0 = the sizes explain it all,",
         "             1 = as close as masks of those sizes can be.",
         "",
         "  within      two steps of the SAME chain            <- what per-step DINO buys",
         "  same-image  two steps of two different chains      <- is it just the picture?",
         "  diff-image  another image's boxes on this grid     <- the floor",
         "  centre      a centred rectangle, no detector       <- the question",
         ""]
    for res in results:
        L += ["-" * W,
              f"{res['model']}   ({res['n_images']} images, {res['n_completions']} chains, "
              f"{res['n_steps']} steps, {res['n_within_pairs']} within-chain step pairs)",
              "-" * W]
        L.append(f"  a step's union covers {res['union_frac_mean']:.3f} of the patch grid on "
                 f"average ({res['union_frac_p50']:.3f} at the median); the fixed rectangle "
                 f"covers {res['rect_frac_mean']:.3f}")
        L.append(f"  f = {res['f_fixed']:.3f} -> {res['modal_rect'][0]} x "
                 f"{res['modal_rect'][1]} patches on the modal {res['modal_grid'][0]} x "
                 f"{res['modal_grid'][1]} grid")
        L += ["",
              "                     n      IoU  (p10   p50   p90)   chance   best  closeness",
              ]
        rows = (("within", "within"), ("cross_comp", "same-image"),
                ("cross_img", "diff-image"),
                ("centre_fixed", f"centre f={res['f_fixed']:.3f}"),
                ("centre_matched", "centre area-matched"))
        for tag, label in rows:
            r = res.get(tag)
            if not r:
                continue
            L.append(f"  {label:<20}{r['n']:>5}  {r['iou']:.3f}  ({r['iou_p10']:.3f} "
                     f"{r['iou_p50']:.3f} {r['iou_p90']:.3f})   {r['chance']:.3f}  "
                     f"{r['best']:.3f}    {r['closeness']:.3f}")
        pf, pm = res["centre_fixed_pairweighted"], res["centre_matched_pairweighted"]
        L += ["",
              "  the two centre rows are per step; weighted per within-chain pair instead "
              "(so the chains",
              f"  carry exactly the weight they carry above): IoU {pf['iou']:.3f} / "
              f"{pm['iou']:.3f}, closeness {pf['closeness']:.3f} / {pm['closeness']:.3f}"]
        # The headline: where the rectangle lands on the floor-to-ceiling span the first
        # three rows define. Quoted on both scales because they disagree, and the
        # disagreement is the finding -- the rectangle looks much closer on raw IoU than
        # it does once the sizes it gets for free are discounted.
        L += ["  where the rectangle lands on the span the first three rows define",
              "  (0% = the different-image floor, 100% = the within-chain level):"]
        if res["n_within_pairs"] < 100:
            # A run whose within-chain level rests on a handful of pairs has no span to
            # be a fraction of; the percentage would be division by noise.
            L.append(f"    only {res['n_within_pairs']} within-chain pairs -- the floor "
                     "and the ceiling are both noise here, no fraction quoted")
        else:
            for tag, label in (("centre_fixed", ff_lbl(res)),
                               ("centre_matched", "area-matched")):
                r = res[tag]
                f = []
                for scale in ("iou", "closeness"):
                    lo, hi = res["cross_img"][scale], res["within"][scale]
                    f.append(f"{100 * (r[scale] - lo) / (hi - lo):6.1f}%"
                             if hi - lo >= 0.05 else "   n/a")
                L.append(f"    {label:<16}{f[0]} on IoU   {f[1]} on closeness")
        L.append("")

        sw = res["swap"]
        ff = ff_lbl(res)
        L += [f"  DOES THE REWARD NOTICE?  scored with `{metric_name}` "
              f"({sw['n_steps']} steps, {sw['n_groups']} groups of chains)",
              "    a step's map scores higher on its OWN union than on   (0.5 = coin flip)",
              f"      {'another step of the same chain':<48}{sw['own_rank_mean']:.3f}",
              f"      {'the centre rectangle, ' + ff:<48}{sw['own_beats_fixed']:.3f}",
              f"      {'the centre rectangle, area-matched':<48}{sw['own_beats_matched']:.3f}",
              f"      {'the whole grid (no mask at all)':<48}{sw['own_beats_flat']:.3f}",
              "    rank correlation of the per-chain reward against the real per-step-DINO one,",
              "    and whether the top-ranked rollout is still top-ranked:",
              f"      {'DINO once per chain (first step boxes)':<48}"
              f"{sw['rho_first']:.3f}   top-1 kept {100 * sw['top1_first']:.1f}%",
              f"      {'DINO once per image (another chain sentence)':<48}"
              f"{sw['rho_other_mean']:.3f}   top-1 kept {100 * sw['top1_other_mean']:.1f}%"
              f"   (mean of {N_REDRAW} draws)",
              f"      {'centre rectangle, ' + ff:<48}"
              f"{sw['rho_fixed']:.3f}   top-1 kept {100 * sw['top1_fixed']:.1f}%",
              f"      {'centre rectangle, area-matched':<48}"
              f"{sw['rho_matched']:.3f}   top-1 kept {100 * sw['top1_matched']:.1f}%",
              f"      {'no mask at all (whole grid = mean/max)':<48}"
              f"{sw['rho_flat']:.3f}   top-1 kept {100 * sw['top1_flat']:.1f}%",
              ""]
        if "verify_max_abs_err" in res:
            L.append(f"  recomputed-vs-stored max abs error: {res['verify_max_abs_err']:.5f}")
            L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("merged", nargs="+", help="probe_merged.json file(s) from overlap_probe")
    ap.add_argument("--out-dir", default="outputs/centre_box_probe")
    ap.add_argument("--models", default=None, help="comma-separated model keys (default: all)")
    ap.add_argument("--metric", default="mean_in", choices=sorted(METRICS))
    ap.add_argument("--f", type=float, default=F_FIXED,
                    help=f"area fraction of the fixed rectangle (default {F_FIXED})")
    ap.add_argument("--redraws", type=int, default=N_REDRAW,
                    help="redraws for the once-per-image control")
    ap.add_argument("--validate", action="store_true",
                    help="reproduce step_box_similarity's published rows first, and "
                         "refuse to write the report if they do not match")
    args = ap.parse_args()

    keep = set(args.models.split(",")) if args.models else None
    results = []
    for path in args.merged:
        d = json.load(open(path))
        if not d["config"].get("store_maps"):
            print(f"[skip] {path}: store_maps was off, no masks stored", file=sys.stderr)
            continue
        for mk, mv in d["models"].items():
            if keep and mk not in keep:
                continue
            images = load_model(mv)
            if not images:
                continue
            tag = f"{os.path.basename(os.path.dirname(path))}::{mk}"
            print(f"[run] {tag} ...", file=sys.stderr, flush=True)
            res = analyse(tag, images, args.metric, args.f, args.redraws,
                          verify=args.validate)
            res["source"] = path
            res["max_box_area"] = d["config"].get("max_box_area")
            res["max_union_area"] = d["config"].get("max_union_area")
            results.append(res)

    if not results:
        print("nothing to analyse", file=sys.stderr)
        return 1

    ok = True
    if args.validate:
        ok, txt = validate(results)
        print(txt)
        print()
        if not ok:
            print("validation failed; report not written", file=sys.stderr)
            return 2

    txt = render(results, args.metric)
    print(txt)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "report.txt"), "w") as f:
        f.write(txt + "\n")
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwritten to {args.out_dir}/report.txt and report.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
