#!/usr/bin/env python
"""Is the self-grounding loop hackable -- does the policy change WHAT IT SAYS so that the
saliency reward gets easier, rather than looking where it says it is looking?

    # after overlap_probe.py has generated on the same prompts for every arm
    python selfground_audit.py --stage text \
        --probe outputs/overlap_probe/align-A/probe_merged.json \
        --probe outputs/overlap_probe/align-C/probe_merged.json \
        --out-dir outputs/selfground/holdout
    python selfground_audit.py --stage crossmap --out-dir ... --base base_coldstart
    python selfground_audit.py --stage report   --out-dir ...

WHY THIS EXISTS. R_sal grounds the policy's OWN observation sentences, so the policy sits
on both sides of the reward: it writes the text that picks the target region, and it
produces the attention that is scored against that region. Everything the reward can be
raised by that is not "look where you say you are looking" is a hack, and there are four
of them, each with its own column here:

  H1 ground more often   an ungroundable step is SKIPPED, not scored 0, so naming things
                         DINO can find changes which steps are scored at all
  H2 ground bigger       phi = mean(map in union)/max(map); a union that swallows the
                         image regresses phi to mean/max, a mask-free statistic
  H3 ground where the    the map peaks on the border ring and the top-left corner in this
     attention already   model family, so naming whatever sits there scores without
     is                  moving any attention
  H4 say less            fewer observe steps = fewer terms in the mean, and a chain that
                         keeps only its best step raises the mean by pruning

None of these needs the attention to move. The stages measure each one before vs after
RL on the SAME prompts, against two controls that share everything but the reward:
`no_sal` (GRPO, same data, R_sal removed) separates "RL drift" from "R_sal drift", and
`center_rect` (a fixed centred rectangle instead of DINO's boxes) is trained by a reward
whose mask cannot depend on the text at all.

STAGES
  text      CPU, seconds. Every per-step and per-completion statistic that comes off
            probe_merged.json: grounding rate, box count, box area, box location, union
            ring share, step counts, duplicate rate, sentence-label mix, the vocabulary
            of the observation sentences, and what phi correlates with inside an arm.
  crossmap  CPU. The decomposition: score each arm's OWN boxes under ANOTHER arm's
            attention on the same image, and under the image-independent border prior.
            Text effect and attention effect, separated.
  crosspass GPU, ~an hour on 8. The same decomposition WITHOUT the proxy: teacher-force
            each arm's completions through each model and score the stored boxes with the
            map that model actually produces while reading those exact tokens. The
            off-diagonal cells are the fixed-chain comparison -- same sentences, same
            boxes, different weights -- so anything they show is attention alone.
            `--all-heads` adds every statistic on the layer's full-head mean.
  dino      GPU, minutes. Re-grounds every observe step to record what the reward's DINO
            call throws away -- the box confidences and the phrase each box matched --
            plus the wrong-image control that says whether a sentence grounds because of
            THIS image or would ground anywhere.
  sheet     CPU. A stratified sample of grounded steps rendered with their boxes, as an
            HTML sheet for the manual "is the phrase actually supported?" pass, and the
            same sample as JSONL for a vision judge.
  report    CPU. One markdown page from whatever stages have run.

Every interval is a prompt-clustered bootstrap: the 8 rollouts of one prompt see one
image and one question, so resampling completions would understate the spread.
"""

from __future__ import annotations

import argparse
import ast
import base64
import importlib.util
import json
import math
import os
import re
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

N_BOOT = 2000
SEED = 20260921


# ---------------------------------------------------------------------------
# stored bytes -> arrays (the probe's own encoding; see overlap_probe.quantize_map)
# ---------------------------------------------------------------------------
def _u8(b64):
    return np.frombuffer(base64.b64decode(b64), dtype=np.uint8)


def decode_mask(b64, gh, gw):
    return _u8(b64).astype(bool).reshape(gh, gw)


def decode_map(b64, gh, gw, mx):
    """The ABSOLUTE map: the stored byte is the patch's value as a fraction of the peak."""
    return _u8(b64).astype(np.float64).reshape(gh, gw) * (float(mx) / 255.0)


def _raster(boxes, gh=10, gw=16):
    """Relative boxes -> the patch mask, the way overlap_rewards._raster_union does it
    (a patch is in if any box covers any of it). Only used to compare two box lists."""
    m = np.zeros((gh, gw), bool)
    for b in boxes or []:
        x1, y1, x2, y2 = [float(v) for v in b]
        c0, c1 = int(np.floor(x1 * gw)), int(np.ceil(x2 * gw))
        r0, r1 = int(np.floor(y1 * gh)), int(np.ceil(y2 * gh))
        m[max(r0, 0):min(r1, gh), max(c0, 0):min(c1, gw)] = True
    return m


def ring_mask(gh, gw):
    m = np.zeros((gh, gw), dtype=bool)
    m[0, :] = m[-1, :] = True
    m[:, 0] = m[:, -1] = True
    return m


def mean_in(smap, mask):
    """The trained objective, phi(s) = mean(map over the union) / max(map over the image).

    Reimplemented rather than imported so this file stays numpy-only, and checked against
    the shipped reward in test_selfground_audit_cpu.py.
    """
    mx = float(np.max(smap))
    if mx <= 0 or mask is None or not mask.any():
        return float("nan")
    return float(smap[mask].mean() / mx)


_ROLL_NULL = None


def roll_null_mod():
    """trl/rewards/roll_null.py by path -- it is numpy-only, and importing the package
    would pull torch into a report that has no business needing it."""
    global _ROLL_NULL
    if _ROLL_NULL is None:
        _ROLL_NULL = _load_probe_module("_sg_roll_null", "trl/rewards/roll_null.py")
    return _ROLL_NULL


def null_stats(smap, mask, sib_masks=(), seed=0, n_offsets=16):
    """The two matched nulls for `share_in`, both read off the SAME map.

    A union covering half the grid scores well on any map that is not uniform, and the
    named regions of this corpus sit nearer the centre than the attention does -- so "more
    mass inside the union" is not on its own evidence that the map followed the sentence.
    Two controls hold everything that is not the sentence:

      roll  the step's OWN union translated over the grid (the repo's roll null, in-frame
            offsets when there are enough). Area and shape fixed, only the position
            moves, so `enr - enr_roll` is what survives the region's geometry.
      sib   the union of ANOTHER observe step of the same completion. Picture, chain and
            weights fixed, only the region changes, so `enr - enr_sib` is
            step-specificity: does this map follow the STEP or just the image?

    `seed` fixes the offsets, so the same step draws the same translates under both
    models and the cross-model comparison stays paired.
    """
    RN = roll_null_mod()
    smap = np.asarray(smap, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    tot, nan = float(smap.sum()), float("nan")
    out = {"share_roll": nan, "enr_roll": nan, "n_roll": 0,
           "share_sib": nan, "enr_sib": nan, "n_sib": 0}
    if tot <= 0 or not mask.any():
        return out
    offs, _ = RN.sample_offsets(mask, int(n_offsets), np.random.default_rng(seed))
    if offs:
        sh = [float(smap[np.roll(mask, o, axis=(0, 1))].sum() / tot) for o in offs]
        out["share_roll"] = float(np.mean(sh))
        out["enr_roll"] = float(np.mean(sh)) / float(mask.mean())
        out["n_roll"] = len(offs)
    sibs = [np.asarray(s, dtype=bool) for s in sib_masks
            if np.asarray(s).shape == mask.shape and np.asarray(s).any()]
    if sibs:
        sh = [float(smap[s].sum() / tot) for s in sibs]
        out["share_sib"] = float(np.mean(sh))
        out["enr_sib"] = float(np.mean([v / float(s.mean()) for v, s in zip(sh, sibs)]))
        out["n_sib"] = len(sibs)
    return out


def map_stats(smap, mask):
    """Everything the fixed-chain comparison reads off ONE step map.

    The capture keeps the attention weights themselves -- a completion token's row sums to
    1 over the whole prefix, and the map is the slice of that row on the image tokens --
    so `vis_mass`, the map's sum, is the share of the step's attention that reached the
    image AT ALL. That makes `share_in` a within-visual-token normalisation: it says where
    on the image the mass went, and cannot be raised by sending more mass to the image.

        phi       mean(union) / max(image)          the trained reward; no chance level
        share_in  mass(union) / mass(image)         redistribution only
        enr       share_in / union's area share     chance 1.0, scale-free (= mean_in_v2)
        vis_mass  mass(image) / mass(everything)    the confound share_in divides out
    """
    smap = np.asarray(smap, dtype=np.float64)
    tot, mx, nan = float(smap.sum()), float(smap.max()), float("nan")
    ok = tot > 0 and mask is not None and mask.any()
    ring = ring_mask(*smap.shape)
    m_in = float(smap[mask].sum()) if ok else nan
    return dict(
        phi=mean_in(smap, mask),
        flat=(float(smap.mean() / mx) if mx > 0 else nan),
        vis_mass=tot,
        mass_in=m_in,
        share_in=(m_in / tot) if ok else nan,
        enr=((m_in / tot) / float(mask.mean())) if ok else nan,
        ring_mass=(float(smap[ring].sum() / tot) if tot > 0 else nan),
        ring_en=(float((smap[ring].sum() / tot) / (ring.sum() / ring.size))
                 if tot > 0 else nan),
        tl_en=(float((smap[0, 0] / tot) * smap.size) if tot > 0 else nan),
    )


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _probe_paths(p: str) -> Path:
    q = Path(p)
    if q.is_dir():
        q = q / "probe_merged.json"
    return q


def load_arms(paths, keep=None, drop=None):
    """-> {arm: {"path":..., "config":..., "samples":[...]}} over every model in every file.

    An arm is one (probe run, model) pair. Arm names collide only if the same model name
    was probed twice, and then the second copy is suffixed with its run directory so both
    survive rather than one silently overwriting the other.
    """
    arms = {}
    for p in paths:
        path = _probe_paths(p)
        d = json.load(open(path))
        run = path.parent.name
        for name, rec in d["models"].items():
            if keep and name not in keep:
                continue
            if drop and name in drop:
                continue
            arm = name if name not in arms else f"{name}@{run}"
            arms[arm] = {
                "path": str(path), "run": run, "config": d["config"],
                "model_path": rec.get("path"), "adapter": rec.get("adapter"),
                "samples": rec["samples"],
            }
    return arms


# ---------------------------------------------------------------------------
# flattening: one row per observe step, one per completion
# ---------------------------------------------------------------------------
GENERIC_FRAMES = {
    # The sentence frames a hacked run converged to on set_a (docs/HANDOFF.md result 1):
    # they name no object at all, so DINO grounds them on whatever the image opens with.
    "background": r"\bbackground\b",
    "image/picture/scene": r"\b(the (image|picture|scene|photo|photograph))\b",
    "no_object_deixis": r"\b(there (is|are)|it (is|appears)|we can see)\b",
}

STOP = set("""a an the this that these those there here is are was were be been being am
it its it's they them their he she his her him we us our you your i me my of in on at to
for from with without by as and or but if then than so such very more most less least
some any all both each few many much no not only other others same own into over under
about above below between out up down off again further once because while during before
after which who whom whose what when where why how can could should would may might must
will shall do does did done doing have has had having appears appear appearing seems seem
seen looks look looking shows show showing showing suggests suggest indicates indicate
likely probably possibly clearly visible visibly given also just even still yet well
image picture photo photograph scene side left right top bottom center centre middle
part parts area areas region regions thing things something anything nothing one two
three four five s t re ve ll d m""".split())

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")


def content_terms(text):
    """Lower-cased content words of a sentence: a transparent stand-in for 'the objects it
    mentions'. The `dino` stage replaces this with the phrase DINO actually matched."""
    return [w for w in (t.lower() for t in TOKEN_RE.findall(text or ""))
            if w not in STOP and len(w) > 2]


def _norm_step(text):
    return re.sub(r"[^a-z0-9 ]+", "", (text or "").lower()).strip()


def box_stats(boxes):
    """-> (areas, centre distances) for relative [x1,y1,x2,y2] boxes.

    The centre distance is the box centre's distance from the image centre over the
    distance to a corner, so 0 is dead centre and 1 is a corner -- the same scale as the
    union's `ecc`, which the reward's own diagnostics log.
    """
    areas, dists = [], []
    for b in boxes or []:
        x1, y1, x2, y2 = [float(v) for v in b]
        areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1))
        dists.append(math.hypot((x1 + x2) / 2 - 0.5, (y1 + y2) / 2 - 0.5) / math.hypot(0.5, 0.5))
    return areas, dists


def flatten(arms):
    """-> (step rows, completion rows). One flat table each, arms concatenated."""
    steps, comps = [], []
    for arm, rec in arms.items():
        for s in rec["samples"]:
            qid = str(s.get("question_id"))
            for c in s["completions"]:
                osteps = c.get("observe_steps") or []
                sents = c.get("all_sentences") or []
                labels = Counter(x.get("label") for x in sents)
                norm = [_norm_step(x["text"]) for x in osteps]
                dup = 1.0 - (len(set(norm)) / len(norm)) if norm else float("nan")
                phis = [r.get("mean_in_raw") for r in osteps
                        if r.get("grounded") and r.get("mean_in_raw") is not None]
                # the completion's distinct content terms, and the two per-completion
                # counts as RATES. Trained arms write much shorter completions, so a
                # per-completion count mixes "how much it says about the image" with
                # "how much it says"; the rates separate them.
                comp_terms = set()
                for r in osteps:
                    comp_terms |= set(content_terms(r.get("text")))
                clen = c.get("n_completion_tokens") or 0
                toks = [r.get("n_tokens") for r in osteps if r.get("grounded")]
                unions = [r.get("union_frac_uncapped") for r in osteps
                          if r.get("grounded") and r.get("union_frac_uncapped") is not None]
                flats = [(r.get("map_mean") / r.get("map_max")) for r in osteps
                         if r.get("grounded") and r.get("map_max")]
                comps.append(dict(
                    arm=arm, qid=qid, sample=s.get("sample_index"), comp=c.get("index"),
                    # per-completion aggregates of the scored steps, so the within-group
                    # correlations below are over the same unit as the reward
                    mean_step_tokens=float(np.mean(toks)) if toks else float("nan"),
                    n_scored=len(toks),
                    mean_union=float(np.mean(unions)) if unions else float("nan"),
                    mean_flatness=float(np.mean(flats)) if flats else float("nan"),
                    image=s.get("image_file"), question=s.get("question"),
                    gt=s.get("gt_answer"), text=c.get("text"),
                    n_tokens=c.get("n_completion_tokens"),
                    truncated=bool(c.get("truncated_at_max_tokens")),
                    format_valid=bool(c.get("format_valid")),
                    n_sentences=len(sents),
                    n_observe_total=c.get("n_observe_steps_total"),
                    n_observe_scored=c.get("n_observe_steps_scored"),
                    n_terms=len(comp_terms),
                    observe_per_100tok=(100.0 * (c.get("n_observe_steps_total") or 0) / clen
                                        if clen else float("nan")),
                    terms_per_100tok=(100.0 * len(comp_terms) / clen
                                      if clen else float("nan")),
                    sentences_per_100tok=(100.0 * len(sents) / clen
                                          if clen else float("nan")),
                    frac_observe=(labels.get("observe", 0) / len(sents)) if sents else float("nan"),
                    n_plan=labels.get("plan", 0), n_deduce=labels.get("deduce", 0),
                    n_none=labels.get("none", 0),
                    dup_step_frac=dup,
                    phi=float(np.mean(phis)) if phis else float("nan"),
                    reward=(c.get("rewards") or {}).get("think_overlap_reward"),
                    accuracy=(c.get("rewards") or {}).get("accuracy_reward"),
                    judge=(c.get("rewards") or {}).get("openai_reward"),
                ))
                for r in osteps:
                    gh, gw = (r.get("grid") or [0, 0])
                    mask = (decode_mask(r["mask_q"], gh, gw)
                            if r.get("mask_q") and gh else None)
                    ring = ring_mask(gh, gw) if gh else None
                    a_raw, d_raw = box_stats(r.get("boxes_raw"))
                    a_kept, d_kept = box_stats(r.get("boxes_kept"))
                    steps.append(dict(
                        arm=arm, qid=qid, sample=s.get("sample_index"), comp=c.get("index"),
                        step=r.get("step_index"), text=r.get("text"),
                        n_tokens=r.get("n_tokens"),
                        n_boxes_raw=r.get("n_boxes_raw"), n_boxes_kept=r.get("n_boxes_kept"),
                        any_box=bool((r.get("n_boxes_raw") or 0) > 0),
                        grounded=bool(r.get("grounded")),
                        dropped_by_union_cap=bool(r.get("dropped_by_union_cap")),
                        union_frac=r.get("union_frac_uncapped"),
                        box_area_frac=r.get("box_area_frac"),
                        max_box_area=r.get("max_box_area"),
                        mean_box_area=float(np.mean(a_raw)) if a_raw else float("nan"),
                        med_box_area=float(np.median(a_raw)) if a_raw else float("nan"),
                        mean_box_area_kept=float(np.mean(a_kept)) if a_kept else float("nan"),
                        mean_box_ecc=float(np.mean(d_raw)) if d_raw else float("nan"),
                        mean_box_ecc_kept=float(np.mean(d_kept)) if d_kept else float("nan"),
                        ecc=r.get("ecc"),
                        ring_share=(float(mask[ring].sum() / mask.sum())
                                    if mask is not None and mask.any() else float("nan")),
                        ring_cover=(float(mask[ring].sum() / ring.sum())
                                    if mask is not None and mask.any() else float("nan")),
                        phi=r.get("mean_in_raw"),
                        # mean_in_v2 = mean(map in union) / mean(map over the image):
                        # the same map on both sides, so the scale cancels and there is
                        # no peak to flatten against. Chance is 1.0. It is the
                        # enrichment factor phi hides -- phi = flatness x this.
                        enrichment=r.get("mean_in_v2_raw"),
                        auroc=r.get("auroc_raw"),
                        logratio=r.get("logratio_raw"),
                        map_max=r.get("map_max"), map_mean=r.get("map_mean"),
                        image_mass=r.get("image_mass"),
                        flatness=((r.get("map_mean") / r.get("map_max"))
                                  if r.get("map_max") else float("nan")),
                    ))
    return steps, comps


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def _clusters(rows, value, where=None):
    """-> {qid: [values]} for a prompt-clustered bootstrap."""
    out = defaultdict(list)
    for r in rows:
        if where and not where(r):
            continue
        v = r.get(value)
        if v is None:
            continue
        v = float(v)
        if not np.isfinite(v):
            continue
        out[r["qid"]].append(v)
    return out


def boot_mean(rows, value, where=None, n_boot=N_BOOT, seed=SEED):
    """Mean with a prompt-clustered percentile CI. -> (mean, lo, hi, n)."""
    cl = _clusters(rows, value, where)
    keys = list(cl)
    if not keys:
        return float("nan"), float("nan"), float("nan"), 0
    flat = np.concatenate([np.asarray(cl[k], float) for k in keys])
    n = flat.size
    mean = float(flat.mean())
    sums = np.array([np.sum(cl[k]) for k in keys])
    cnts = np.array([len(cl[k]) for k in keys], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    bs = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return mean, float(lo), float(hi), int(n)


def boot_diff(rows_a, rows_b, value, where=None, n_boot=N_BOOT, seed=SEED):
    """mean(a) - mean(b) with a PAIRED prompt-clustered CI: the same prompts are resampled
    on both sides, which is what makes an arm-to-arm difference a within-prompt contrast."""
    ca, cb = _clusters(rows_a, value, where), _clusters(rows_b, value, where)
    keys = sorted(set(ca) & set(cb))
    if not keys:
        return float("nan"), float("nan"), float("nan"), 0
    sa = np.array([np.sum(ca[k]) for k in keys], float)
    na = np.array([len(ca[k]) for k in keys], float)
    sb = np.array([np.sum(cb[k]) for k in keys], float)
    nb = np.array([len(cb[k]) for k in keys], float)
    point = float(sa.sum() / na.sum() - sb.sum() / nb.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    bs = (sa[idx].sum(1) / np.maximum(na[idx].sum(1), 1)
          - sb[idx].sum(1) / np.maximum(nb[idx].sum(1), 1))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return point, float(lo), float(hi), len(keys)


def within_group_corr(rows, field, reward="reward"):
    """Group-centred correlation of `field` with the reward, inside a prompt's rollouts.

    This is the only signal GRPO has. The advantage subtracts the group mean, so a
    property that does not vary BETWEEN the 8 rollouts of one prompt contributes nothing
    however large it is; what the policy can learn is exactly what this correlates with.
    Measured on the COLD START it is a prediction about which way the run will drift.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[r["qid"]].append(r)
    xs, ys = [], []
    for g in groups.values():
        a = np.array([r.get(field, np.nan) if r.get(field) is not None else np.nan
                      for r in g], float)
        b = np.array([r.get(reward, np.nan) if r.get(reward) is not None else np.nan
                      for r in g], float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            continue
        xs.append(a[ok] - a[ok].mean())
        ys.append(b[ok] - b[ok].mean())
    if not xs:
        return float("nan"), 0
    return pearson(np.concatenate(xs), np.concatenate(ys)), int(sum(len(v) for v in xs))


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or x[ok].std() < 1e-15 or y[ok].std() < 1e-15:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    def rank(v):
        order = v.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        return r
    return pearson(rank(x[ok]), rank(y[ok]))


def logodds_delta(counts_a, counts_b, prior=None, alpha0=500.0):
    """Monroe et al.'s log-odds ratio with an informative Dirichlet prior, z-scored.

    Raw frequency deltas are dominated by whichever word is commonest; this is the
    standard fix and is what makes "the trained model says X more" a claim rather than a
    ranking of stopwords.
    """
    prior = prior if prior is not None else (counts_a + counts_b)
    tot_p = sum(prior.values()) or 1.0
    na, nb = sum(counts_a.values()), sum(counts_b.values())
    out = {}
    for w in set(counts_a) | set(counts_b):
        ap = alpha0 * prior.get(w, 0) / tot_p
        ya, yb = counts_a.get(w, 0), counts_b.get(w, 0)
        if ya + yb < 3:
            continue
        num_a = ya + ap
        num_b = yb + ap
        den_a = na + alpha0 - num_a
        den_b = nb + alpha0 - num_b
        if min(num_a, num_b, den_a, den_b) <= 0:
            continue
        d = math.log(num_a / den_a) - math.log(num_b / den_b)
        var = 1.0 / num_a + 1.0 / num_b
        out[w] = d / math.sqrt(var)
    return out


# ---------------------------------------------------------------------------
# stage: text
# ---------------------------------------------------------------------------
STEP_METRICS = [
    ("n_boxes_raw", "boxes DINO proposed"),
    ("n_boxes_kept", "boxes past the 0.5 area cap"),
    ("mean_box_area", "mean box area (frac of image)"),
    ("max_box_area", "largest box area"),
    ("union_frac", "union area (frac of grid)"),
    ("mean_box_ecc", "box centre distance from image centre"),
    ("ecc", "union centroid distance from centre"),
    ("ring_share", "share of the union on the border ring"),
    ("ring_cover", "share of the border ring the union covers"),
    ("n_tokens", "step length (tokens)"),
    ("phi", "phi = mean_in (the reward's per-step score)"),
    ("flatness", "flatness = map mean / map max (box-blind)"),
    ("enrichment", "enrichment = mean_in_v2 = phi / flatness (chance 1.0)"),
    ("logratio", "logratio vs the union's own translates"),
    ("auroc", "auroc of the map inside the union"),
]

COMP_METRICS = [
    ("n_tokens", "completion length (tokens)"),
    ("n_sentences", "sentences per completion"),
    ("sentences_per_100tok", "... per 100 completion tokens"),
    ("n_observe_total", "observe steps per completion"),
    ("observe_per_100tok", "... per 100 completion tokens"),
    ("n_observe_scored", "observe steps actually scored"),
    ("n_terms", "distinct content terms per completion"),
    ("terms_per_100tok", "... per 100 completion tokens"),
    ("frac_observe", "share of sentences labelled observe"),
    ("n_plan", "plan steps"), ("n_deduce", "deduce steps"), ("n_none", "none steps"),
    ("dup_step_frac", "duplicate observe steps (frac)"),
    ("phi", "mean phi over the completion's scored steps"),
    ("reward", "think_overlap_reward"),
    ("accuracy", "exact-match accuracy reward"),
    ("judge", "judge reward"),
]


def stage_text(args, out_dir):
    arms = load_arms(args.probe, keep=set(args.keep_arm) if args.keep_arm else None,
                     drop=set(args.drop_arm) if args.drop_arm else None)
    if not arms:
        raise SystemExit("no arms loaded")
    steps, comps = flatten(arms)
    by_arm_s = defaultdict(list)
    by_arm_c = defaultdict(list)
    for r in steps:
        by_arm_s[r["arm"]].append(r)
    for r in comps:
        by_arm_c[r["arm"]].append(r)

    base = args.base if args.base in by_arm_s else list(by_arm_s)[0]
    order = [base] + [a for a in by_arm_s if a != base]

    res = {"arms": {}, "base": base, "order": order,
           "config": {a: {k: v for k, v in arms[a].items() if k != "samples"} for a in arms}}

    for a in order:
        S, C = by_arm_s[a], by_arm_c[a]
        gs = [r for r in S if r["grounded"]]
        arm = {"n_completions": len(C), "n_observe_steps": len(S),
               "n_grounded_steps": len(gs), "n_prompts": len({r["qid"] for r in C})}
        # H1: does a step ground at all? `any_box` is DINO returning something above the
        # threshold; `grounded` is the stricter thing the reward needs -- a union that
        # also survived rasterisation and the area caps.
        arm["any_box_rate"] = boot_mean(S, "any_box")
        arm["grounded_rate"] = boot_mean(S, "grounded")
        arm["union_cap_rate"] = boot_mean(S, "dropped_by_union_cap")
        arm["format_valid"] = boot_mean(C, "format_valid")
        arm["truncated"] = boot_mean(C, "truncated")
        # per-step metrics, over the steps the reward actually scored
        arm["steps"] = {k: boot_mean(gs, k) for k, _ in STEP_METRICS}
        arm["steps_all"] = {k: boot_mean(S, k) for k in ("n_boxes_raw", "n_tokens")}
        arm["comps"] = {k: boot_mean(C, k) for k, _ in COMP_METRICS}
        # what does phi ride on, INSIDE this arm? (group-free, step level)
        for k in ("union_frac", "ecc", "ring_share", "n_boxes_raw", "flatness",
                  "mean_box_area", "n_tokens"):
            arm.setdefault("phi_corr", {})[k] = [
                pearson([r["phi"] for r in gs], [r[k] for r in gs]),
                spearman([r["phi"] for r in gs], [r[k] for r in gs])]
        # what does the reward PAY FOR, inside a prompt's eight rollouts? -- the only
        # direction GRPO can move in, and on the cold start a prediction about the drift
        for k in ("mean_step_tokens", "n_scored", "mean_union", "mean_flatness",
                  "n_tokens", "n_observe_total"):
            r, n = within_group_corr(C, k)
            arm.setdefault("within_group", {})[k] = [r, n]
        # generic sentence frames
        for name, pat in GENERIC_FRAMES.items():
            rx = re.compile(pat, re.I)
            rows = [dict(r, hit=bool(rx.search(r["text"] or ""))) for r in S]
            arm.setdefault("frames", {})[name] = boot_mean(rows, "hit")
        # vocabulary of the observation sentences (document frequency: a term counts once
        # per step, so one sentence repeating "cup" four times does not become four)
        terms = Counter()
        for r in S:
            terms.update(set(content_terms(r["text"])))
        arm["terms"] = terms.most_common(60)
        arm["n_terms"] = sum(terms.values())
        # "how many different things does it name" -- over EVERY completion, including the
        # ones with no observe step at all. Averaging only over completions that have one
        # silently changes the denominator between arms: 40% of center_rect's completions
        # have none.
        arm["distinct_terms_per_completion"] = boot_mean(C, "n_terms")
        arm["distinct_terms_per_100tok"] = boot_mean(C, "terms_per_100tok")
        arm["completions_with_a_step"] = boot_mean(
            [dict(qid=c["qid"], v=1.0 if c["n_observe_total"] else 0.0) for c in C], "v")
        res["arms"][a] = arm

    # paired deltas against the base arm
    res["deltas"] = {}
    for a in order[1:]:
        d = {}
        for k, _ in STEP_METRICS:
            d[k] = boot_diff([r for r in by_arm_s[a] if r["grounded"]],
                             [r for r in by_arm_s[base] if r["grounded"]], k)
        for k in ("any_box", "grounded", "dropped_by_union_cap"):
            d[k] = boot_diff(by_arm_s[a], by_arm_s[base], k)
        for k, _ in COMP_METRICS:
            d["c_" + k] = boot_diff(by_arm_c[a], by_arm_c[base], k)
        res["deltas"][a] = d

    # vocabulary shift against the base arm
    res["vocab"] = {}
    def _df(rows):
        c = Counter()
        for r in rows:
            c.update(set(content_terms(r["text"])))
        return c
    cb = _df(by_arm_s[base])
    for a in order[1:]:
        z = logodds_delta(_df(by_arm_s[a]), cb)
        top = sorted(z.items(), key=lambda kv: -kv[1])[:25]
        bot = sorted(z.items(), key=lambda kv: kv[1])[:25]
        res["vocab"][a] = {"rises": top, "falls": bot}

    (out_dir / "text_stats.json").write_text(json.dumps(res, indent=1))
    # the flat tables, for the later stages and for anybody who wants a different cut
    slim = [{k: v for k, v in r.items() if k != "text"} | {"text": r["text"]} for r in steps]
    (out_dir / "steps.json").write_text(json.dumps(slim))
    (out_dir / "completions.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "text"} for r in comps]))
    print(f"[text] {len(arms)} arms, {len(comps)} completions, {len(steps)} observe steps"
          f" -> {out_dir}/text_stats.json")
    return res


# ---------------------------------------------------------------------------
# stage: crossmap -- the text effect and the attention effect, separated
# ---------------------------------------------------------------------------
def _image_key(sample):
    return sample.get("image_file") or str(sample.get("question_id"))


def per_image_maps(arms):
    """-> {arm: {image: mean absolute map}} and the global mean map per arm.

    The reward reads the map the model produced WHILE WRITING that step, and we cannot
    have that for text the model did not write. The substitute is this arm's mean map over
    its own steps on the same image; `agreement` below reports what the substitution costs
    on the one case where both are available (an arm scoring its own steps).
    """
    per_arm = {}
    for arm, rec in arms.items():
        acc = {}
        for s in rec["samples"]:
            key = _image_key(s)
            for c in s["completions"]:
                for r in c.get("observe_steps") or []:
                    if not (r.get("map_q") and r.get("grid")):
                        continue
                    gh, gw = r["grid"]
                    m = decode_map(r["map_q"], gh, gw, r.get("map_max") or 0.0)
                    cur = acc.get(key)
                    if cur is None:
                        acc[key] = [m.copy(), 1, (gh, gw)]
                    elif cur[2] == (gh, gw):
                        cur[0] += m
                        cur[1] += 1
        per_arm[arm] = {k: (v[0] / v[1], v[2]) for k, v in acc.items()}
    return per_arm


def stage_crossmap(args, out_dir):
    arms = load_arms(args.probe, keep=set(args.keep_arm) if args.keep_arm else None,
                     drop=set(args.drop_arm) if args.drop_arm else None)
    maps = per_image_maps(arms)
    base = args.base if args.base in arms else list(arms)[0]

    # the image-independent prior: this arm's mean map over every image, resized by grid
    priors = {}
    for arm, per_img in maps.items():
        acc = {}
        for m, grid in per_img.values():
            cur = acc.get(grid)
            acc[grid] = (m.copy(), 1) if cur is None else (cur[0] + m, cur[1] + 1)
        priors[arm] = {g: v[0] / v[1] for g, v in acc.items()}

    rows = []           # one row per (text arm, scoring arm, step)
    agree = []          # own per-step map vs own per-image mean map, on the same mask
    for t_arm, rec in arms.items():
        for s in rec["samples"]:
            key = _image_key(s)
            for c in s["completions"]:
                for r in c.get("observe_steps") or []:
                    if not (r.get("mask_q") and r.get("grid") and r.get("grounded")):
                        continue
                    gh, gw = r["grid"]
                    mask = decode_mask(r["mask_q"], gh, gw)
                    own = decode_map(r["map_q"], gh, gw, r.get("map_max") or 0.0)
                    row = dict(arm=t_arm, qid=str(s.get("question_id")),
                               sample=s.get("sample_index"), comp=c.get("index"),
                               step=r.get("step_index"), own=mean_in(own, mask),
                               union_frac=r.get("union_frac_uncapped"))
                    for m_arm in arms:
                        got = maps[m_arm].get(key)
                        if got is not None and got[1] == (gh, gw):
                            row["map:" + m_arm] = mean_in(got[0], mask)
                        pr = priors[m_arm].get((gh, gw))
                        if pr is not None:
                            row["prior:" + m_arm] = mean_in(pr, mask)
                    if "map:" + t_arm in row:
                        agree.append((row["own"], row["map:" + t_arm]))
                    rows.append(row)

    res = {"base": base, "arms": list(arms), "n_rows": len(rows)}
    if agree:
        a = np.array([x[0] for x in agree]); b = np.array([x[1] for x in agree])
        res["proxy_check"] = {
            "n": int(a.size), "mean_own": float(np.nanmean(a)),
            "mean_image_mean_map": float(np.nanmean(b)),
            "pearson": pearson(a, b), "spearman": spearman(a, b)}

    cells, deltas = {}, {}
    for t_arm in arms:
        R = [r for r in rows if r["arm"] == t_arm]
        cells[t_arm] = {"own": boot_mean(R, "own")}
        for m_arm in arms:
            cells[t_arm]["map:" + m_arm] = boot_mean(R, "map:" + m_arm)
            cells[t_arm]["prior:" + m_arm] = boot_mean(R, "prior:" + m_arm)
    for t_arm in arms:
        if t_arm == base:
            continue
        A = [r for r in rows if r["arm"] == t_arm]
        B = [r for r in rows if r["arm"] == base]
        deltas[t_arm] = {
            # TEXT effect: this arm's boxes minus the base arm's boxes, both scored under
            # the SAME (base) attention. Everything here is language.
            "text_under_base": boot_diff(A, B, "map:" + base),
            "text_under_prior": boot_diff(A, B, "prior:" + base),
            # ATTENTION effect: the base arm's own boxes, scored under this arm's
            # attention minus under the base's. Everything here is attention.
            "attention_on_base_text": boot_diff(
                [dict(r, v=r.get("map:" + t_arm)) for r in B],
                [dict(r, v=r.get("map:" + base)) for r in B], "v"),
            # the whole move, own text under own attention
            "total": boot_diff(A, B, "own"),
        }
    res["cells"], res["deltas"] = cells, deltas
    (out_dir / "crossmap.json").write_text(json.dumps(res, indent=1))
    print(f"[crossmap] {len(rows)} scored steps, {len(arms)} arms -> {out_dir}/crossmap.json")
    return res


# ---------------------------------------------------------------------------
# stage: dino -- what the reward's grounding call throws away
# ---------------------------------------------------------------------------
def _dino_key(image_file, text):
    return f"{image_file}||{text}"


def _unique_steps(arms):
    """-> {key: {"image": abs path, "text": str, "stored": [boxes], "arms": [...]}}.

    One DINO call per distinct (image, sentence): the 8 rollouts of a prompt repeat
    sentences, and two arms that write the same sentence must get the same boxes or the
    comparison is measuring sampling noise in the detector, which has none.
    """
    uniq = {}
    for arm, rec in arms.items():
        root = Path(rec["path"]).parent
        for s in rec["samples"]:
            img = root / (s.get("image_file") or "")
            for c in s["completions"]:
                for r in c.get("observe_steps") or []:
                    if not r.get("text"):
                        continue
                    k = _dino_key(s.get("image_file"), r["text"])
                    u = uniq.setdefault(k, {"image": str(img), "image_file": s.get("image_file"),
                                            "text": r["text"], "qid": str(s.get("question_id")),
                                            "stored": r.get("boxes_raw"), "arms": set()})
                    u["arms"].add(arm)
    for u in uniq.values():
        u["arms"] = sorted(u["arms"])
    return uniq


def dino_call(proc, model, device, images, texts, threshold):
    """Grounding-DINO exactly as trl/rewards/overlap_rewards._dino_boxes_local calls it,
    but keeping the two fields the reward drops: the per-box confidence and the phrase
    the box matched. Boxes come back in relative [x1,y1,x2,y2] so they compare directly
    with what the probe stored."""
    import torch

    prompts = [(t.strip() + ".") if not t.strip().endswith(".") else t.strip() for t in texts]
    inputs = proc(images=images, text=prompts, return_tensors="pt",
                  padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = [(im.size[1], im.size[0]) for im in images]
    results = proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=threshold, text_threshold=threshold,
        target_sizes=target_sizes)
    out = []
    for j, res in enumerate(results):
        w, h = images[j].size
        boxes = [[b[0] / w, b[1] / h, b[2] / w, b[3] / h] for b in res["boxes"].tolist()]
        out.append({
            "boxes": [[round(float(v), 5) for v in b] for b in boxes],
            "scores": [round(float(s), 4) for s in res["scores"].tolist()],
            "labels": [str(x) for x in (res.get("text_labels") or res.get("labels") or [])],
        })
    return out


def stage_dino(args, out_dir):
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    arms = load_arms(args.probe, keep=set(args.keep_arm) if args.keep_arm else None,
                     drop=set(args.drop_arm) if args.drop_arm else None)
    uniq = _unique_steps(arms)
    keys = sorted(uniq)
    if args.limit:
        keys = keys[: args.limit]
    keys = [k for i, k in enumerate(keys) if i % args.num_shards == args.shard]

    # The control pool: every image any arm was probed on, one entry per picture. Each
    # probe run saved its own copy of the same 100 images, so keying on the file name
    # rather than the path keeps a picture from being drawn four times over. A sentence
    # that grounds just as well on someone else's picture is not grounding on THIS one.
    by_file = {}
    for u in uniq.values():
        by_file.setdefault(u["image_file"], u["image"])
    pool = sorted(by_file.items())
    rng = np.random.default_rng(SEED)

    device = args.device
    proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base").to(device).eval()

    cache = {}

    def _img(path):
        im = cache.get(path)
        if im is None:
            im = Image.open(path).convert("RGB")
            if len(cache) > 64:
                cache.clear()
            cache[path] = im
        return im

    # every (text, image) job: the step's own image first, then its controls
    jobs = []
    for k in keys:
        u = uniq[k]
        jobs.append((k, "own", u["image_file"], u["image"], u["text"]))
        others = [p for p in pool if p[0] != u["image_file"]]
        if others and args.controls:
            pick = rng.choice(len(others), size=min(args.controls, len(others)), replace=False)
            for j, i in enumerate(pick):
                jobs.append((k, f"ctl{j}", others[int(i)][0], others[int(i)][1], u["text"]))

    res = defaultdict(dict)
    f = out_dir / f"dino_shard{args.shard:02d}.json"

    def _save():
        # Written every so often, not only at the end: Grounding-DINO's own processor
        # upsamples to 800x1333, so a shard is an hour of calls and a wall-clock kill
        # that lost all of them has happened. Steps with no `own` run yet are dropped, so
        # a partial file is a smaller audit rather than a broken one.
        done = [k for k in keys if "own" in res.get(k, {})]
        f.write_text(json.dumps({
            "box_threshold": args.box_threshold, "controls": args.controls,
            "shard": args.shard, "num_shards": args.num_shards,
            "n_planned": len(keys), "n_done": len(done),
            "steps": {k: {"text": uniq[k]["text"], "image_file": uniq[k]["image_file"],
                          "qid": uniq[k]["qid"], "arms": uniq[k]["arms"],
                          "stored": uniq[k]["stored"], "runs": res[k]} for k in done}}))

    bs = args.dino_batch_size
    for start in range(0, len(jobs), bs):
        chunk = jobs[start:start + bs]
        images = [_img(j[3]) for j in chunk]
        got = dino_call(proc, model, device, images, [j[4] for j in chunk], args.box_threshold)
        for j, g in zip(chunk, got):
            g["image_file"] = j[2]
            res[j[0]][j[1]] = g
        if (start // bs) % 50 == 0:
            print(f"[dino] {start}/{len(jobs)} calls", flush=True)
        if (start // bs) % 200 == 199:
            _save()
    _save()
    print(f"[dino] {len(keys)} unique steps, {len(jobs)} calls -> {f}")


def merge_dino(out_dir):
    """Merge every dino_shard*.json in the directory.

    Two arms that wrote the same sentence about the same picture share a key -- that is
    the point, the detector is called once -- so a later shard must UNION the arm list
    rather than replace it. Shards from a second run (dropped in as dino_shard1*.json)
    merge the same way, which is how an arm probed later joins the audit.
    """
    steps, meta = {}, None
    for f in sorted(out_dir.glob("dino_shard*.json")):
        d = json.load(open(f))
        meta = meta or {k: v for k, v in d.items() if k != "steps"}
        for k, v in d["steps"].items():
            prev = steps.get(k)
            if prev is not None:
                v = dict(v, arms=sorted(set(prev.get("arms", [])) | set(v.get("arms", []))))
            steps[k] = v
    if not steps:
        return None
    payload = dict(meta or {}, steps=steps)
    (out_dir / "dino_audit.json").write_text(json.dumps(payload))
    return payload


# ---------------------------------------------------------------------------
# stage: crosspass -- phi(text of arm A, attention of model B), exactly
# ---------------------------------------------------------------------------
def _load_probe_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stage_crosspass(args, out_dir):
    """Teacher-force each arm's completions through each model and score the SAME stored
    masks with the resulting maps.

    This is the decomposition the proxy in `crossmap` approximates, without the proxy: the
    map is the one that model produces while reading that exact sentence, which is the
    quantity the reward reads. The diagonal cells re-derive the numbers the probe already
    stored, and `--check` reports the agreement, so a cell that disagrees with the probe
    is a misconfiguration rather than a result.
    """
    import torch

    PROBE = _load_probe_module("_sg_probe", "overlap_probe.py")
    OSTEPS = PROBE.OSTEPS
    from PIL import Image

    arms = load_arms(args.probe, keep=set(args.keep_arm) if args.keep_arm else None,
                     drop=set(args.drop_arm) if args.drop_arm else None)
    text_arms = args.text_arm or list(arms)
    map_arms = args.map_arm or list(arms)

    # the completions to push through, sharded
    work = []
    for a in text_arms:
        rec = arms[a]
        root = Path(rec["path"]).parent
        for s in rec["samples"]:
            for c in s["completions"]:
                osteps = [r for r in (c.get("observe_steps") or [])
                          if r.get("mask_q") and r.get("grid") and r.get("grounded")]
                if not osteps:
                    continue
                work.append(dict(arm=a, qid=str(s.get("question_id")),
                                 sample=s.get("sample_index"), comp=c.get("index"),
                                 image=str(root / (s.get("image_file") or "")),
                                 question=s.get("question"), text=c.get("text"),
                                 steps=[{"text": r["text"], "grid": r["grid"],
                                         "mask_q": r["mask_q"], "step": r.get("step_index"),
                                         "phi_stored": r.get("mean_in_raw")} for r in osteps]))
    work = [w for i, w in enumerate(work) if i % args.num_shards == args.shard]
    if args.limit:
        work = work[: args.limit]

    heads = [int(h) for h in args.overlap_heads.split(",")]
    rows, comp_rows = [], []
    skipped = Counter()
    for m_arm in map_arms:
        rec = arms[m_arm]
        base = rec["model_path"] or rec["config"]["base_model"]
        processor, model = PROBE.load_model(base, rec.get("adapter"), args.device, "sdpa")
        attn_mod = PROBE.find_attn_module(model, args.overlap_layer)
        # One capture, two head sets. The rewarded heads are the lens R_sal was trained
        # through; the layer's full-head mean is "where the model looks" without that lens,
        # and it costs nothing because the eager re-run already materialises every head.
        n_heads = int(getattr(attn_mod.config, "num_attention_heads", 0)
                      or model.config.text_config.num_attention_heads)
        cap_heads = list(range(n_heads)) if args.all_heads else heads
        sel = [cap_heads.index(h) for h in heads]
        tok = processor.tokenizer
        for wi, w in enumerate(work):
            image = Image.open(w["image"]).convert("RGB")
            text = PROBE.build_prompt(processor, w["question"])
            inputs = processor(text=[text], images=[[image]], return_tensors="pt",
                               padding=True, padding_side="left",
                               add_special_tokens=False).to(args.device)
            prompt_len = inputs["input_ids"].shape[1]
            # One tokenisation for both the forward and the spans, with no special tokens
            # added: the probe's spans live in the same space as its comp_ids, and a BOS
            # here would shift every span by one and quietly misattribute the map.
            out = tok([w["text"]], add_special_tokens=False)
            comp_ids = out["input_ids"][0]
            ts = re.search(r"<think>\s*(\S\S*)", w["text"], re.DOTALL | re.MULTILINE)
            te = re.search(r"(\S)\s*</think>", w["text"], re.DOTALL | re.MULTILINE)
            if not (ts and te):
                skipped["no_think_span"] += 1
                continue
            ts_idx, te_idx = ts.start(1), te.start(1)
            t_start = out.char_to_token(0, ts_idx)
            t_end = out.char_to_token(0, te_idx)
            if t_start is None or t_end is None or t_end <= t_start:
                skipped["span_not_tokenisable"] += 1
                continue
            # The stored steps already ARE the observe sentences the classifier picked, so
            # this re-derives only their token spans in this tokenisation. Running the
            # classifier again would risk a different step set for the two sides of the
            # comparison, which is the one thing the cross pass must not have.
            steps, owner = [], []
            n_chars = len(w["text"])
            for st in w["steps"]:
                cs = w["text"].find(st["text"])
                if cs < 0:
                    skipped["step_text_not_found"] += 1
                    continue
                ce = cs + len(st["text"])
                tok_a = OSTEPS._char_to_tok(out, 0, cs, n_chars)
                tok_b_incl = out.char_to_token(0, ce - 1)
                if tok_b_incl is None:
                    tok_b_incl = OSTEPS._char_to_tok(out, 0, ce - 1, n_chars)
                if tok_a is None or tok_b_incl is None:
                    skipped["step_not_tokenisable"] += 1
                    continue
                a = max(tok_a, t_start)
                b = min(tok_b_incl + 1, t_end + 1)
                if b <= a:
                    skipped["empty_span"] += 1
                    continue
                steps.append((st["text"], a, b))
                owner.append(st)
            if not steps:
                continue
            gh = int(inputs["image_grid_thw"][0, 1].item()) // 2
            gw = int(inputs["image_grid_thw"][0, 2].item()) // 2
            per_tok = PROBE.capture_layer_attention(
                model, attn_mod, inputs, prompt_len, comp_ids, cap_heads, args.device)
            maps = PROBE.step_maps_from_attention(per_tok[sel], steps, gh, gw, "mean")
            by_text = {_norm_step(m["text"]): m["map"] for m in maps}
            by_text_all = {}
            if args.all_heads:
                by_text_all = {_norm_step(m["text"]): m["map"] for m in
                               PROBE.step_maps_from_attention(per_tok, steps, gh, gw, "mean")}
            # How much of the chain's attention reaches the image at all, over the WHOLE
            # think span rather than the observe steps: the quantity a "the trained model
            # simply looks at the image more" objection is about, with the text held fixed.
            hi = min(t_end + 1, per_tok.shape[1])
            if hi > t_start:
                span = per_tok[:, t_start:hi, :]
                comp_rows.append(dict(
                    text_arm=w["arm"], map_arm=m_arm, qid=w["qid"], sample=w["sample"],
                    comp=w["comp"], n_span_tokens=int(hi - t_start),
                    vis_chain=float(span[sel].mean(axis=(0, 1)).sum()),
                    vis_chain_all=(float(span.mean(axis=(0, 1)).sum())
                                   if args.all_heads else float("nan"))))
            # every step's mask up front: a step's null is the OTHER steps of its own
            # completion, so the whole chain's masks have to be in hand before scoring one
            masks = [decode_mask(st["mask_q"], *st["grid"]) for st in owner]
            for i, ((_, tok_a, tok_b), st) in enumerate(zip(steps, owner)):
                m = by_text.get(_norm_step(st["text"]))
                if m is None or list(m.shape) != list(st["grid"]):
                    skipped["grid_mismatch"] += 1
                    continue
                mask = masks[i]
                # A sibling that IS this step's union measures nothing -- two sentences
                # that grounded to the same patches -- so it is not a sibling here.
                sibs = [s for j, s in enumerate(masks)
                        if j != i and s.shape == mask.shape and not np.array_equal(s, mask)]
                seed = zlib.crc32(
                    f"{w['qid']}|{w['sample']}|{w['comp']}|{st['step']}".encode())
                # The map's own shape, recorded alongside phi: with the text held fixed,
                # these say whether the trained weights moved the attention at all -- the
                # corner sink of Figure 5 and the flatness phi mostly rides on.
                row = dict(text_arm=w["arm"], map_arm=m_arm, qid=w["qid"],
                           sample=w["sample"], comp=w["comp"], step=st["step"],
                           phi_stored=st["phi_stored"], union_frac=float(mask.mean()),
                           n_tokens=int(tok_b - tok_a))
                st_rw = map_stats(m, mask)
                if args.nulls:
                    st_rw.update(null_stats(m, mask, sibs, seed=seed))
                row.update(st_rw)
                m_all = by_text_all.get(_norm_step(st["text"]))
                if m_all is not None and list(m_all.shape) == list(st["grid"]):
                    st_all = map_stats(m_all, mask)
                    if args.nulls:
                        st_all.update(null_stats(m_all, mask, sibs, seed=seed))
                    row.update({k + "_all": v for k, v in st_all.items()})
                rows.append(row)
            del per_tok
            if wi % 25 == 0:
                torch.cuda.empty_cache()
                print(f"[crosspass] map={m_arm} {wi}/{len(work)}", flush=True)
        del model
        torch.cuda.empty_cache()

    f = out_dir / f"crosspass_shard{args.shard:02d}.json"
    f.write_text(json.dumps({"rows": rows, "comp_rows": comp_rows, "shard": args.shard,
                             "num_shards": args.num_shards, "base": args.base,
                             "skipped": dict(skipped), "all_heads": bool(args.all_heads),
                             "n_heads": n_heads if map_arms else None,
                             "layer": args.overlap_layer, "heads": args.overlap_heads}))
    print(f"[crosspass] {len(rows)} (step, map arm) scores over {len(comp_rows)} "
          f"(completion, map arm) pairs, skipped {dict(skipped)} -> {f}")


def merge_crosspass(out_dir):
    rows, comp_rows, meta, skipped = [], [], None, Counter()
    for f in sorted(out_dir.glob("crosspass_shard*.json")):
        d = json.load(open(f))
        meta = meta or {k: v for k, v in d.items()
                        if k not in ("rows", "comp_rows", "skipped")}
        skipped.update(d.get("skipped") or {})
        rows += d["rows"]
        comp_rows += d.get("comp_rows") or []
    if not rows:
        return None
    payload = dict(meta or {}, rows=rows, comp_rows=comp_rows, skipped=dict(skipped))
    (out_dir / "crosspass.json").write_text(json.dumps(payload))
    return payload


# ---------------------------------------------------------------------------
# stage: sheet -- the manual "is this phrase actually supported?" pass
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = (
    "You check whether a sentence written by a vision-language model about an image is "
    "supported by that image, and whether a detector's boxes landed on what the sentence "
    "refers to. You are strict and literal: a sentence is SUPPORTED only if everything it "
    "asserts about the picture is visibly true."
)

JUDGE_USER = """You are given the same photograph twice: first unmarked, then with red rectangles drawn on it.

Sentence written by the model: "{text}"

Answer with exactly three lines and nothing else:
Support: <yes|partly|no>          (is the sentence's claim about the image visibly true?)
Boxes: <yes|partly|no>            (do the red rectangles cover the thing(s) the sentence refers to?)
Named: <a short comma-separated list of the concrete objects the sentence names, or NONE>"""


def draw_boxes(image, boxes, width=3):
    from PIL import ImageDraw

    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w, h = im.size
    for b in boxes or []:
        d.rectangle([b[0] * w, b[1] * h, b[2] * w, b[3] * h], outline=(255, 0, 0), width=width)
    return im


def stage_sheet(args, out_dir):
    from PIL import Image

    arms = load_arms(args.probe, keep=set(args.keep_arm) if args.keep_arm else None,
                     drop=set(args.drop_arm) if args.drop_arm else None)
    rng = np.random.default_rng(SEED)
    items = []
    for arm, rec in arms.items():
        root = Path(rec["path"]).parent
        pool = []
        for s in rec["samples"]:
            for c in s["completions"]:
                for r in c.get("observe_steps") or []:
                    if not r.get("grounded") or not r.get("boxes_kept"):
                        continue
                    pool.append((s, c, r))
        if not pool:
            continue
        # One step per PROMPT at most, so a sheet is not eight rollouts of one picture.
        by_q = defaultdict(list)
        for s, c, r in pool:
            by_q[str(s.get("question_id"))].append((s, c, r))
        qs = sorted(by_q)
        pick_q = rng.choice(len(qs), size=min(args.n_sheet, len(qs)), replace=False)
        for qi in pick_q:
            cand = by_q[qs[int(qi)]]
            s, c, r = cand[int(rng.integers(0, len(cand)))]
            items.append(dict(arm=arm, qid=str(s.get("question_id")),
                              question=s.get("question"), gt=s.get("gt_answer"),
                              image=str(root / (s.get("image_file") or "")),
                              text=r["text"], boxes=r.get("boxes_kept"),
                              n_boxes=r.get("n_boxes_kept"),
                              union_frac=r.get("union_frac_uncapped"),
                              phi=r.get("mean_in_raw")))
    rng.shuffle(items)          # blind the manual pass to the arm's order
    for i, it in enumerate(items):
        it["item_id"] = f"{i:04d}"

    sheet_dir = out_dir / "sheet"
    sheet_dir.mkdir(exist_ok=True)
    for it in items:
        im = Image.open(it["image"]).convert("RGB")
        draw_boxes(im, it["boxes"]).save(sheet_dir / f"{it['item_id']}_boxed.png")
        im.save(sheet_dir / f"{it['item_id']}_plain.png")
    (out_dir / "sheet_items.json").write_text(json.dumps(items, indent=1))

    # the blind review page: arm hidden behind a toggle so a human can score first
    rows = []
    for it in items:
        rows.append(
            f"<tr><td>{it['item_id']}</td>"
            f"<td><img src='sheet/{it['item_id']}_boxed.png' width='320'></td>"
            f"<td style='max-width:32em'><b>{it['text']}</b><br><small>Q: {it['question']}</small>"
            f"<br><small>boxes {it['n_boxes']}, union {it['union_frac']:.2f}, "
            f"phi {it['phi']:.3f}</small></td>"
            f"<td class='arm' style='display:none'>{it['arm']}</td>"
            f"<td><label><input type=radio name='s{it['item_id']}'>yes</label> "
            f"<label><input type=radio name='s{it['item_id']}'>partly</label> "
            f"<label><input type=radio name='s{it['item_id']}'>no</label></td></tr>")
    html = ("<html><head><meta charset='utf-8'><style>"
            "body{font-family:system-ui;margin:2em}td{vertical-align:top;padding:6px;"
            "border-bottom:1px solid #ddd}</style></head><body>"
            "<h1>Is the grounded phrase supported by the image?</h1>"
            "<p>Red rectangles are the boxes Grounding-DINO returned for the sentence and "
            "that the reward's mask was built from. Score support first, then reveal the "
            "arm.</p>"
            "<button onclick=\"document.querySelectorAll('.arm').forEach(e=>e.style.display="
            "e.style.display=='none'?'table-cell':'none')\">reveal arm</button>"
            "<table><tr><th>id</th><th>image + boxes</th><th>sentence</th><th>arm</th>"
            "<th>supported?</th></tr>" + "".join(rows) + "</table></body></html>")
    (out_dir / "sheet.html").write_text(html)
    print(f"[sheet] {len(items)} items over {len(arms)} arms -> {out_dir}/sheet.html")

    if args.judge:
        judge_sheet(items, out_dir, args)


def judge_sheet(items, out_dir, args):
    """The same sheet, scored by a vision judge (GPT-4o mini through the NVIDIA gateway).

    A model judge is not the manual pass and does not replace it: it is the part that
    scales to every arm, and the HTML sheet is there so a person can check the judge on a
    subsample. Failures are recorded, never silently scored.
    """
    import base64 as b64
    import io
    from concurrent.futures import ThreadPoolExecutor

    import openai
    from PIL import Image

    key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("--judge needs NVIDIA_API_KEY (or OPENAI_API_KEY) in the "
                         "environment; the sheet itself is already written")
    client = openai.OpenAI(api_key=key,
                           base_url=os.environ.get("OPENAI_BASE_URL",
                                                   "https://inference-api.nvidia.com"))
    model = os.environ.get("JUDGE_MODEL", "azure/openai/gpt-4o-mini")

    def _b64(path):
        im = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return b64.b64encode(buf.getvalue()).decode("ascii")

    def _one(it):
        plain = _b64(out_dir / "sheet" / f"{it['item_id']}_plain.png")
        boxed = _b64(out_dir / "sheet" / f"{it['item_id']}_boxed.png")
        try:
            r = client.chat.completions.create(
                model=model, temperature=0, max_tokens=200,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": [
                              {"type": "image_url",
                               "image_url": {"url": f"data:image/png;base64,{plain}"}},
                              {"type": "image_url",
                               "image_url": {"url": f"data:image/png;base64,{boxed}"}},
                              {"type": "text",
                               "text": JUDGE_USER.format(text=it["text"])}]}],
                timeout=120)
            txt = r.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            return dict(it, judge_error=f"{type(e).__name__}: {str(e)[:200]}")
        def _f(name):
            m = re.search(rf"{name}:\s*(\w+)", txt, re.I)
            return m.group(1).lower() if m else None
        named = re.search(r"Named:\s*(.+)", txt, re.I)
        return dict(it, support=_f("Support"), boxes_ok=_f("Boxes"),
                    named=(named.group(1).strip() if named else None), raw=txt)

    with ThreadPoolExecutor(max_workers=int(os.environ.get("JUDGE_MAX_WORKERS", "8"))) as ex:
        scored = list(ex.map(_one, items))
    (out_dir / "sheet_judged.json").write_text(json.dumps(scored, indent=1))
    ok = [s for s in scored if s.get("support")]
    print(f"[sheet] judged {len(ok)}/{len(scored)} -> {out_dir}/sheet_judged.json")


# ---------------------------------------------------------------------------
# stage: humanbox -- do the named regions get closer to the annotated one?
# ---------------------------------------------------------------------------
def stage_humanbox(args, out_dir):
    """Score every step's DINO union against the row's HUMAN box.

    The crossmap prior says the trained arms' unions sit where this model generically
    looks. That has two readings -- "it names what it already stares at" (the reviewer's
    hypothesis) and "it names what matters, and the attention already knew" -- and they
    are told apart by a target that owes nothing to the model: Saliency-R1-8K ships one
    annotated box per row, the same boxes Saliency-R1 trains on.

    Reported per step, against that box H and the union U on the step's own patch grid:
      recall      |U n H| / |H|          how much of the annotated region is named
      precision   |U n H| / |U|          how much of what is named is annotated
      enrichment  precision / (|H|/|P|)  1.0 = the union is placed at chance w.r.t. H
    """
    arms = load_arms(args.probe, keep=set(args.keep_arm) if args.keep_arm else None,
                     drop=set(args.drop_arm) if args.drop_arm else None)
    from datasets import load_dataset

    ds = load_dataset(args.human_box_dataset, split="train")
    boxes = {}
    for qid, bb in zip(ds["question_id"], ds["bbox"]):
        # the column is stored as a STRING, and a row may carry one box or several
        if isinstance(bb, str):
            try:
                bb = ast.literal_eval(bb)
            except (ValueError, SyntaxError):
                continue
        if bb and not isinstance(bb[0], (list, tuple)):
            bb = [bb]
        boxes[str(qid)] = [[float(v) for v in b] for b in bb] if bb else None
    print(f"[humanbox] {len(boxes)} annotated rows from {args.human_box_dataset}", flush=True)

    rows = []
    for arm, rec in arms.items():
        for s in rec["samples"]:
            qid = str(s.get("question_id"))
            hb = boxes.get(qid)
            if not hb:
                continue
            for c in s["completions"]:
                for r in c.get("observe_steps") or []:
                    if not (r.get("mask_q") and r.get("grid") and r.get("grounded")):
                        continue
                    gh, gw = r["grid"]
                    u = decode_mask(r["mask_q"], gh, gw)
                    h = _raster(hb, gh, gw)
                    nu, nh = float(u.sum()), float(h.sum())
                    inter = float(np.logical_and(u, h).sum())
                    if nu <= 0 or nh <= 0:
                        continue
                    prec = inter / nu
                    rows.append(dict(
                        arm=arm, qid=qid, sample=s.get("sample_index"), comp=c.get("index"),
                        step=r.get("step_index"), recall=inter / nh, precision=prec,
                        enrichment=prec / (nh / u.size),
                        iou=inter / float(np.logical_or(u, h).sum()),
                        human_area=nh / u.size, union_area=nu / u.size))
    if not rows:
        raise SystemExit("no steps matched an annotated row")
    (out_dir / "humanbox.json").write_text(json.dumps({"rows": rows}))
    print(f"[humanbox] {len(rows)} steps over {len(arms)} arms -> {out_dir}/humanbox.json")


# ---------------------------------------------------------------------------
# stage: figs -- the distributions behind the means
# ---------------------------------------------------------------------------
def stage_figs(args, out_dir):
    """The reviewer asked for distributions, not means: a box-area histogram, where the
    boxes sit, what the detector's confidence looks like, and what phi tracks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads((out_dir / "steps.json").read_text())
    rows = [r for r in rows if r.get("grounded")]
    arms = args.keep_arm or [args.base, "ours"]
    arms = [a for a in arms if any(r["arm"] == a for r in rows)]
    colors = {a: c for a, c in zip(arms, ["#4c6ef5", "#f03e3e", "#37b24d", "#f59f00",
                                          "#7048e8", "#0ca678"])}

    def _hist(ax, field, bins, title, xlabel):
        for a in arms:
            v = np.array([r[field] for r in rows if r["arm"] == a
                          and r.get(field) is not None
                          and np.isfinite(r.get(field, np.nan))], float)
            if v.size:
                ax.hist(v, bins=bins, density=True, histtype="step", linewidth=1.8,
                        color=colors[a], label=f"{a} (n={v.size})")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    _hist(axes[0][0], "mean_box_area", np.linspace(0, 1, 41),
          "box area (mean over the step's boxes)", "fraction of the image")
    _hist(axes[0][1], "union_frac", np.linspace(0, 1, 41),
          "union area, before the cap", "fraction of the patch grid")
    _hist(axes[0][2], "mean_box_ecc", np.linspace(0, 1, 41),
          "box centre distance from the image centre", "0 = centre, 1 = corner")
    _hist(axes[1][0], "ring_share", np.linspace(0, 1, 41),
          "share of the union on the border ring", "fraction of the union")
    _hist(axes[1][1], "phi", np.linspace(0, 0.3, 61),
          "phi, the per-step reward", "mean(map in union) / max(map)")
    _hist(axes[1][2], "n_boxes_raw", np.arange(0, 61, 2),
          "boxes the detector returned", "boxes per step")
    fig.suptitle("What the self-grounding loop targets, before and after RL "
                 "(held-out prompts, 8 rollouts each)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_boxes.png", dpi=140)
    plt.close(fig)

    # phi against the box-blind statistic it is supposed to improve on
    fig, axes = plt.subplots(1, len(arms), figsize=(4.6 * len(arms), 4.2), squeeze=False)
    for ax, a in zip(axes[0], arms):
        x = np.array([r["flatness"] for r in rows if r["arm"] == a], float)
        y = np.array([r["phi"] for r in rows if r["arm"] == a], float)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=4, alpha=0.25, color=colors[a], edgecolors="none")
        lim = [0, max(0.3, float(np.nanpercentile(np.concatenate([x[ok], y[ok]]), 99.5)))]
        ax.plot(lim, lim, color="#adb5bd", linewidth=1, linestyle="--")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(f"{a}  (r = {pearson(x, y):+.2f})", fontsize=10)
        ax.set_xlabel("map mean / map max  (no boxes)", fontsize=9)
        ax.set_ylabel("phi  (with boxes)", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("phi against the statistic that ignores the boxes; the dashed line is "
                 "'the union scores exactly the image average'", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_phi_flatness.png", dpi=140)
    plt.close(fig)

    made = ["fig_boxes.png", "fig_phi_flatness.png"]

    dino = (json.loads((out_dir / "dino_audit.json").read_text())
            if (out_dir / "dino_audit.json").exists() else None)
    if dino:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for a in arms:
            own, ctl = [], []
            for v in dino["steps"].values():
                if a not in v["arms"]:
                    continue
                runs = v.get("runs") or {}
                s = (runs.get("own") or {}).get("scores") or []
                if s:
                    own.append(max(s))
                for n, c in runs.items():
                    if n.startswith("ctl") and c.get("scores"):
                        ctl.append(max(c["scores"]))
            if own:
                axes[0].hist(own, bins=np.linspace(0, 1, 41), density=True,
                             histtype="step", linewidth=1.8, color=colors[a],
                             label=f"{a} (n={len(own)})")
            if ctl:
                axes[1].hist(ctl, bins=np.linspace(0, 1, 41), density=True,
                             histtype="step", linewidth=1.8, color=colors[a],
                             label=f"{a} (n={len(ctl)})")
        axes[0].set_title("best box confidence, own image", fontsize=10)
        axes[1].set_title("best box confidence, a WRONG image", fontsize=10)
        for ax in axes:
            ax.set_xlabel("Grounding-DINO score", fontsize=9)
            ax.set_ylabel("density", fontsize=9)
            ax.legend(fontsize=7, frameon=False)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_confidence.png", dpi=140)
        plt.close(fig)
        made.append("fig_confidence.png")

    print(f"[figs] {', '.join(made)} -> {out_dir}")


# ---------------------------------------------------------------------------
# stage: report
# ---------------------------------------------------------------------------
def _c(cell, dec=3, pct=False):
    """[mean, lo, hi, n] -> 'mean [lo, hi]'."""
    if cell is None:
        return "—"
    m, lo, hi = cell[0], cell[1], cell[2]
    if m is None or not np.isfinite(m):
        return "—"
    k = 100.0 if pct else 1.0
    s = "%" if pct else ""
    return f"{m*k:.{dec}f}{s} [{lo*k:.{dec}f}, {hi*k:.{dec}f}]"


def _d(cell, dec=3, pct=False):
    """A delta cell, starred when the interval excludes zero."""
    if cell is None or cell[0] is None or not np.isfinite(cell[0]):
        return "—"
    m, lo, hi = cell[0], cell[1], cell[2]
    k = 100.0 if pct else 1.0
    s = "%" if pct else ""
    star = "" if (lo <= 0 <= hi) else " \\*"
    return f"{m*k:+.{dec}f}{s} [{lo*k:+.{dec}f}, {hi*k:+.{dec}f}]{star}"


def union_bin_table(steps_rows, base, arms, n_bins=5, field="phi", by="union_frac"):
    """Δ`field` against `base`, inside bins of `by` -- the size-matched comparison.

    phi's ceiling moves with the union's area, so an arm that grounds bigger scores
    differently for a reason that has nothing to do with where its attention is. Cutting
    on the pooled quantiles of the union area and comparing inside a bin removes that,
    at the price of conditioning on a post-treatment variable -- which is why it is a
    check on the headline number rather than a replacement for it.
    """
    vals = np.array([r[by] for r in steps_rows
                     if r.get(by) is not None and np.isfinite(r.get(by, np.nan))], float)
    if vals.size < n_bins * 10:
        return None
    edges = np.quantile(vals, np.linspace(0, 1, n_bins + 1))
    labels = [f"{edges[b]:.3g}–{edges[b + 1]:.3g}" for b in range(n_bins)]
    edges[0], edges[-1] = -np.inf, np.inf
    out = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = lambda r: (r.get(by) is not None and np.isfinite(r.get(by, np.nan))
                         and lo <= r[by] < hi)
        base_rows = [r for r in steps_rows if r["arm"] == base and sel(r)]
        row = {"bin": labels[b], "n_base": len(base_rows),
               "base": boot_mean(base_rows, field)}
        for a in arms:
            if a == base:
                continue
            arm_rows = [r for r in steps_rows if r["arm"] == a and sel(r)]
            row[a] = boot_diff(arm_rows, base_rows, field)
        out.append(row)
    return out


FIXED_CHAIN_FIELDS = [("phi", "phi", 4), ("share_in", "in-region share", 4),
                      ("enr", "enrichment", 3), ("vis_mass", "visual attention", 4)]


def _fixed_chain_tables(cpass, by, t_arms, m_arms):
    """The one table the whole cross pass exists for: same chain, same boxes, new weights.

    Every row of a block is the SAME sentences scored against the SAME masks, so a
    difference down a block is attention and nothing else. `in-region share` and
    `enrichment` normalise within the visual tokens, which is what separates "the model
    redistributed its attention over the image" from "the model sends more attention to
    the image", the last column.
    """
    rows = cpass["rows"]
    if not any(r.get("vis_mass") is not None for r in rows):
        return []
    layer, heads = cpass.get("layer"), cpass.get("heads")
    L = ["", "### the fixed-chain table", "",
         "Each block holds the chain and the boxes fixed and swaps only the weights, so "
         "every difference inside a block is attention. `in-region share` is the step's "
         "attention mass inside the union over its mass on the image -- normalised WITHIN "
         "the visual tokens, so it measures redistribution across the image and not how "
         "much attention reaches the image; `enrichment` divides that by the union's area "
         "share, so 1.0 is chance; `visual attention` is the share of the step's attention "
         "that lands on image tokens at all.", ""]
    fields = list(FIXED_CHAIN_FIELDS)
    if any(r.get("enr_roll") is not None for r in rows):
        fields += [("enr_vroll", "enr − roll null", 3), ("enr_vsib", "enr − sibling", 3)]
        L += ["`enr − roll null` subtracts the same union translated over the grid (area "
              "and shape held fixed, chance 0), and `enr − sibling` subtracts the other "
              "observe steps' unions on the same picture (chance 0), so neither can be "
              "bought by the map drifting toward wherever the named regions tend to sit.",
              ""]
    blocks = [("", f"layer {layer}, heads {heads} -- the rewarded heads")]
    if any(r.get("phi_all") is not None for r in rows):
        n_heads = cpass.get("n_heads")
        blocks.append(("_all", f"layer {layer}, all {n_heads or ''} heads".replace("  ", " ")))
    for suffix, label in blocks:
        L += [f"**{label}**", "",
              "| chains from | attention from | steps | " +
              " | ".join(lbl for _, lbl, _ in fields) + " |",
              "|---" * (len(fields) + 3) + "|"]
        for t in t_arms:
            present = [m for m in m_arms if by[(t, m)]]
            for m in present:
                L.append(f"| {t} | {m} | {len(by[(t, m)])} | " + " | ".join(
                    _c(boot_mean(by[(t, m)], f + suffix), d)
                    for f, _, d in fields) + " |")
            for m in present[1:]:
                L.append(f"| {t} | Δ ({m} − {present[0]}) | — | " + " | ".join(
                    _d(boot_diff(by[(t, m)], by[(t, present[0])], f + suffix), d)
                    for f, _, d in fields) + " |")
        L += [""]

    crows = cpass.get("comp_rows") or []
    if crows:
        cby = defaultdict(list)
        for r in crows:
            cby[(r["text_arm"], r["map_arm"])].append(r)
        L += ["### how much attention reaches the image at all", "",
              "Mean over the whole `<think>` span rather than the observe steps, one row "
              "per completion. Same chains, so a difference here is the weights sending "
              "more (or less) of the chain's attention to the image.", "",
              "| chains from | attention from | completions | rewarded heads | all heads |",
              "|---|---|---|---|---|"]
        for t in sorted({r["text_arm"] for r in crows}):
            present = [m for m in sorted({r["map_arm"] for r in crows}) if cby[(t, m)]]
            for m in present:
                L.append(f"| {t} | {m} | {len(cby[(t, m)])} | "
                         f"{_c(boot_mean(cby[(t, m)], 'vis_chain'), 4)} | "
                         f"{_c(boot_mean(cby[(t, m)], 'vis_chain_all'), 4)} |")
            for m in present[1:]:
                L.append(f"| {t} | Δ ({m} − {present[0]}) | — | "
                         f"{_d(boot_diff(cby[(t, m)], cby[(t, present[0])], 'vis_chain'), 4)} | "
                         f"{_d(boot_diff(cby[(t, m)], cby[(t, present[0])], 'vis_chain_all'), 4)} |")
        L += [""]
    return L


def stage_report(args, out_dir):
    text = json.loads((out_dir / "text_stats.json").read_text()) \
        if (out_dir / "text_stats.json").exists() else None
    cross = json.loads((out_dir / "crossmap.json").read_text()) \
        if (out_dir / "crossmap.json").exists() else None
    cpass = json.loads((out_dir / "crosspass.json").read_text()) \
        if (out_dir / "crosspass.json").exists() else None
    dino = json.loads((out_dir / "dino_audit.json").read_text()) \
        if (out_dir / "dino_audit.json").exists() else None
    judged = json.loads((out_dir / "sheet_judged.json").read_text()) \
        if (out_dir / "sheet_judged.json").exists() else None

    L = ["# Does the policy talk its way to the saliency reward?", "",
         "Every number is a prompt-clustered bootstrap mean with a 95% interval; a delta is",
         "paired on the prompt and starred when its interval excludes zero.", ""]

    if text:
        base = text["base"]
        order = text["order"]
        L += [f"Arms: {', '.join(order)}. Base (before RL): **{base}**.", ""]
        L += ["| arm | completions | observe steps | prompts |", "|---|---|---|---|"]
        for a in order:
            x = text["arms"][a]
            L.append(f"| {a} | {x['n_completions']} | {x['n_observe_steps']} "
                     f"| {x['n_prompts']} |")
        L += ["", "## 1. Does the model ground more often after RL?", "",
              "| arm | DINO returned a box | step scored by the reward | dropped by the union cap |",
              "|---|---|---|---|"]
        for a in order:
            x = text["arms"][a]
            L.append(f"| {a} | {_c(x['any_box_rate'], 1, pct=True)} | "
                     f"{_c(x['grounded_rate'], 1, pct=True)} | "
                     f"{_c(x['union_cap_rate'], 1, pct=True)} |")
        L += ["", "Deltas against the base arm:", "",
              "| arm | any box | scored | union cap |", "|---|---|---|---|"]
        for a in order[1:]:
            d = text["deltas"][a]
            L.append(f"| {a} | {_d(d['any_box'], 1, pct=True)} | "
                     f"{_d(d['grounded'], 1, pct=True)} | "
                     f"{_d(d['dropped_by_union_cap'], 1, pct=True)} |")

        L += ["", "## 2. What do the boxes look like?", "",
              "Over the steps the reward scored.", "",
              "| metric | " + " | ".join(order) + " |",
              "|---" * (len(order) + 1) + "|"]
        for k, label in STEP_METRICS:
            L.append(f"| {label} | " +
                     " | ".join(_c(text["arms"][a]["steps"][k]) for a in order) + " |")
        L += ["", "Deltas against the base arm:", "",
              "| metric | " + " | ".join(order[1:]) + " |",
              "|---" * len(order) + "|"]
        for k, label in STEP_METRICS:
            L.append(f"| {label} | " +
                     " | ".join(_d(text["deltas"][a][k]) for a in order[1:]) + " |")

        L += ["", "## 3. How much does the model say?", "",
              "| metric | " + " | ".join(order) + " |",
              "|---" * (len(order) + 1) + "|"]
        for k, label in COMP_METRICS:
            L.append(f"| {label} | " +
                     " | ".join(_c(text["arms"][a]["comps"][k]) for a in order) + " |")
        L += ["", "| metric | " + " | ".join(order[1:]) + " |",
              "|---" * len(order) + "|"]
        for k, label in COMP_METRICS:
            L.append(f"| Δ {label} | " +
                     " | ".join(_d(text["deltas"][a]["c_" + k]) for a in order[1:]) + " |")

        L += ["", "## 4. Sentence frames and vocabulary", "",
              "Share of observe steps matching a frame that names no object:", "",
              "| frame | " + " | ".join(order) + " |", "|---" * (len(order) + 1) + "|"]
        for name in GENERIC_FRAMES:
            L.append(f"| {name} | " +
                     " | ".join(_c(text["arms"][a]["frames"][name], 1, pct=True)
                                for a in order) + " |")
        L += ["", "| arm | distinct content terms per completion | ... per 100 completion "
              "tokens | completions with at least one observe step |", "|---|---|---|---|"]
        for a in order:
            x = text["arms"][a]
            L.append(f"| {a} | {_c(x['distinct_terms_per_completion'], 2)} | "
                     f"{_c(x['distinct_terms_per_100tok'], 2)} | "
                     f"{_c(x['completions_with_a_step'], 1, pct=True)} |")
        L += ["", "The commonest things the observation sentences name (share of that "
              "arm's observe steps that use the term at least once):", ""]
        for a in order:
            n = max(text["arms"][a]["n_observe_steps"], 1)
            top = text["arms"][a]["terms"][:15]
            L.append(f"- **{a}**: " + ", ".join(f"{w} ({100.0*c/n:.0f}%)" for w, c in top))
        for a in order[1:]:
            v = text["vocab"][a]
            L += ["", f"**{a}** vs {base}, log-odds z (informative Dirichlet prior):", "",
                  "- rises: " + ", ".join(f"{w} ({z:+.1f})" for w, z in v["rises"][:15]),
                  "- falls: " + ", ".join(f"{w} ({z:+.1f})" for w, z in v["falls"][:15])]

        steps_file = out_dir / "steps.json"
        if steps_file.exists():
            rows = json.loads(steps_file.read_text())
            rows = [r for r in rows if r.get("grounded")]
            tab = union_bin_table(rows, base, order)
            if tab:
                L += ["", "### phi inside matched union-area bins", "",
                      "Δphi against the base arm, computed separately inside quintiles of "
                      "the union's area, so an arm cannot win the row by grounding bigger.",
                      "", "| union area | n (base) | base phi | " +
                      " | ".join(order[1:]) + " |",
                      "|---" * (len(order) + 2) + "|"]
                for r in tab:
                    L.append(f"| {r['bin']} | {r['n_base']} | {_c(r['base'], 4)} | " +
                             " | ".join(_d(r.get(a), 4) for a in order[1:]) + " |")

        if "within_group" in text["arms"][order[0]]:
            wg = list(text["arms"][order[0]]["within_group"])
            L += ["", "### what the reward pays for inside a prompt's rollouts", "",
                  "GRPO subtracts the group mean, so only what varies BETWEEN the eight "
                  "rollouts of one prompt can be learned. Read the base row as a "
                  "prediction about which way a run will drift.", "",
                  "| arm | " + " | ".join(wg) + " |", "|---" * (len(wg) + 1) + "|"]
            for a in order:
                L.append(f"| {a} | " + " | ".join(
                    f"{text['arms'][a]['within_group'][k][0]:+.3f}" for k in wg) + " |")

        L += ["", "## 5. What is phi riding on, inside each arm?", "",
              "Pearson (Spearman) of the per-step score against the step's own geometry.", "",
              "| arm | " + " | ".join(k for k in text["arms"][order[0]]["phi_corr"]) + " |",
              "|---" * (len(text["arms"][order[0]]["phi_corr"]) + 1) + "|"]
        for a in order:
            pc = text["arms"][a]["phi_corr"]
            L.append(f"| {a} | " + " | ".join(f"{v[0]:+.2f} ({v[1]:+.2f})"
                                              for v in pc.values()) + " |")

    if cross:
        L += ["", "## 6. Text effect vs attention effect (per-image mean map proxy)", ""]
        pc = cross.get("proxy_check") or {}
        if pc:
            L += [f"Proxy check: scoring a step with its own map gives "
                  f"{pc['mean_own']:.4f}, with this arm's mean map over the same image "
                  f"{pc['mean_image_mean_map']:.4f} (r = {pc['pearson']:.2f}, "
                  f"rho = {pc['spearman']:.2f}, n = {pc['n']}).", ""]
        arms = cross["arms"]
        L += ["| text from ↓ / attention from → | " + " | ".join(arms) + " | own |",
              "|---" * (len(arms) + 2) + "|"]
        for t in arms:
            cells = cross["cells"][t]
            L.append(f"| {t} | " + " | ".join(_c(cells.get("map:" + m), 4) for m in arms)
                     + f" | {_c(cells.get('own'), 4)} |")
        L += ["", "`prior` scores the same boxes under the base model's mean map over "
              "EVERY image of that grid shape -- an attention target that knows nothing "
              "about this picture, so a gain there is the boxes drifting toward where the "
              "model looks in general.", "",
              "| arm | text effect (its boxes − base's, under base attention) | "
              "text effect under the image-independent prior | "
              "attention effect (base's boxes, its attention − base's) | total |",
              "|---|---|---|---|---|"]
        for a, d in cross["deltas"].items():
            L.append(f"| {a} | {_d(d['text_under_base'], 4)} | "
                     f"{_d(d['text_under_prior'], 4)} | "
                     f"{_d(d['attention_on_base_text'], 4)} | {_d(d['total'], 4)} |")

    if cpass:
        rows = cpass["rows"]
        by = defaultdict(list)
        for r in rows:
            # phi = flatness x enrichment, and enrichment is exactly mean_in_v2: the same
            # map on both sides of the ratio, so it cannot be moved by flattening. Shards
            # written before `map_stats` existed carry neither, so derive both here; a
            # newer shard already has them and must not be overwritten.
            if r.get("enr") is None:
                r["enr"] = (r["phi"] / r["flat"]
                            if r.get("flat") and r.get("phi") is not None else float("nan"))
            if r.get("share_in") is None:
                r["share_in"] = r["enr"] * r["union_frac"]
            # what is left of the in-region enrichment once the region's own geometry
            # (roll) and the picture the chain is about (sibling) are held fixed
            for suf in ("", "_all"):
                for null in ("roll", "sib"):
                    if r.get(f"enr_{null}{suf}") is not None:
                        r[f"enr_v{null}{suf}"] = r[f"enr{suf}"] - r[f"enr_{null}{suf}"]
            by[(r["text_arm"], r["map_arm"])].append(r)
        t_arms = sorted({r["text_arm"] for r in rows})
        m_arms = sorted({r["map_arm"] for r in rows})
        L += ["", "## 7. Text effect vs attention effect (exact teacher-forced cross pass)",
              "", "phi for each arm's completions, re-read through each model's attention "
              "at the rewarded heads, against the same stored boxes.", "",
              "| text from ↓ / attention from → | " + " | ".join(m_arms) + " |",
              "|---" * (len(m_arms) + 1) + "|"]
        for t in t_arms:
            L.append(f"| {t} | " + " | ".join(_c(boot_mean(by[(t, m)], "phi"), 4)
                                              for m in m_arms) + " |")
        # the reproduction check: diagonal cells against what the probe stored
        diag = [r for r in rows if r["text_arm"] == r["map_arm"]
                and r.get("phi_stored") is not None]
        if diag:
            a = np.array([r["phi"] for r in diag]); b = np.array([r["phi_stored"] for r in diag])
            L += ["", f"Diagonal reproduction: n = {len(diag)}, mean re-read "
                  f"{np.nanmean(a):.4f} vs stored {np.nanmean(b):.4f}, "
                  f"r = {pearson(a, b):.3f}."]
        L += _fixed_chain_tables(cpass, by, t_arms, m_arms)
        base = cpass.get("base") or (t_arms[0] if t_arms else None)
        if base:
            for field, label, dec in (("phi", "phi", 4), ("enr", "enrichment", 3)):
                L += ["", f"**{label}**", "",
                      "| arm | text effect | attention effect | total |", "|---|---|---|---|"]
                for t in t_arms:
                    if t == base:
                        continue
                    L.append(
                        f"| {t} | {_d(boot_diff(by[(t, base)], by[(base, base)], field), dec)} | "
                        f"{_d(boot_diff(by[(base, t)], by[(base, base)], field), dec)} | "
                        f"{_d(boot_diff(by[(t, t)], by[(base, base)], field), dec)} |")

        # With the text held fixed, did the weights move the MAP at all? phi can be flat
        # because nothing moved or because two things cancelled; these say which.
        shapes = [("flat", "map mean / map max"), ("ring_en", "border-ring enrichment"),
                  ("tl_en", "top-left patch enrichment"),
                  ("phi", "phi"), ("enr", "enrichment (= mean_in_v2)")]
        if any(r.get("flat") is not None for r in rows):
            L += ["", "### the map's own shape, with the text held fixed", "",
                  "Every row is the SAME completions, re-read through each model.", "",
                  "| text from | statistic | " + " | ".join(m_arms) + " | Δ |",
                  "|---" * (len(m_arms) + 3) + "|"]
            for t in t_arms:
                for k, label in shapes:
                    cells = [boot_mean(by[(t, m)], k) for m in m_arms]
                    d = (boot_diff(by[(t, m_arms[-1])], by[(t, m_arms[0])], k)
                         if len(m_arms) == 2 else None)
                    L.append(f"| {t} | {label} | " +
                             " | ".join(_c(c, 4) for c in cells) +
                             f" | {_d(d, 4) if d else '—'} |")
        # and the length-matched version of the text effect
        if any(r.get("n_tokens") for r in rows):
            same_map = [dict(r, arm=r["text_arm"]) for r in rows if r["map_arm"] == base]
            tab = union_bin_table(same_map, base, t_arms, field="phi", by="n_tokens")
            if tab:
                L += ["", "### the text effect inside matched step-length bins", "",
                      "All of these are read through the BASE model's attention, so only "
                      "the sentences differ. If the effect were only that trained "
                      "sentences are longer -- a longer span averages more token maps and "
                      "a flatter map scores higher -- it would vanish here.", "",
                      "| step length (tokens) | n (base) | base phi | " +
                      " | ".join(a for a in t_arms if a != base) + " |",
                      "|---" * (len(t_arms) + 2) + "|"]
                for r in tab:
                    L.append(f"| {r['bin']} | {r['n_base']} | {_c(r['base'], 4)} | " +
                             " | ".join(_d(r.get(a), 4) for a in t_arms if a != base) + " |")

    if dino:
        L += ["", "## 8. Grounding confidence, and the wrong-image control", ""]
        # The re-grounding must reproduce the MASK the probe stored, or it is measuring a
        # different detector call from the one the reward made. Box-for-box identity is
        # the wrong test: this stage batches by (step, control image) where the probe
        # batched a sample's steps on one image, and Grounding-DINO's processor pads a
        # batch to a common canvas, so borderline boxes cross the 0.1 threshold either
        # way. What has to survive is the union.
        dn, ious = [], []
        for v in dino["steps"].values():
            own = (v.get("runs") or {}).get("own") or {}
            st, got = v.get("stored"), own.get("boxes")
            if st is None or got is None:
                continue
            dn.append(len(got) - len(st))
            a, b = _raster(st), _raster(got)
            u = float(np.logical_or(a, b).sum())
            ious.append(float(np.logical_and(a, b).sum() / u) if u else 1.0)
        if ious:
            dn, ious = np.array(dn), np.array(ious)
            L += [f"Reproduction of the probe's own call, over {len(ious)} steps: union "
                  f"IoU on a 10x16 raster is {ious.mean():.3f} on average "
                  f"({100.0*np.mean(ious > 0.9):.1f}% above 0.9, median "
                  f"{np.median(ious):.3f}), and the box count differs by a median of "
                  f"{np.median(dn):+.0f} ({100.0*np.mean(np.abs(dn) <= 2):.0f}% within "
                  f"two).", ""]
        rec = []
        for k, v in dino["steps"].items():
            own = (v.get("runs") or {}).get("own") or {}
            ctls = [c for n, c in (v.get("runs") or {}).items() if n.startswith("ctl")]
            sc = own.get("scores") or []
            crow = dict(qid=v["qid"], arms=v["arms"],
                        n_boxes=len(own.get("boxes") or []),
                        max_score=max(sc) if sc else float("nan"),
                        mean_score=float(np.mean(sc)) if sc else float("nan"),
                        any_box=float(bool(sc)),
                        ctl_any=float(np.mean([1.0 if c.get("scores") else 0.0 for c in ctls]))
                        if ctls else float("nan"),
                        ctl_max=float(np.mean([max(c["scores"]) if c.get("scores") else 0.0
                                               for c in ctls])) if ctls else float("nan"))
            crow["specificity"] = (crow["max_score"] - crow["ctl_max"]
                                   if np.isfinite(crow["max_score"]) else float("nan"))
            rec.append(crow)
        arms_here = sorted({a for r in rec for a in r["arms"]})
        L += ["One row per distinct (image, sentence); a sentence written by several arms "
              "is counted in each.", "",
              "| arm | boxes | max confidence | mean confidence | grounds on a WRONG image | "
              "max confidence there | own − wrong |", "|---|---|---|---|---|---|---|"]
        for a in arms_here:
            sub = [dict(r, qid=r["qid"]) for r in rec if a in r["arms"]]
            L.append(f"| {a} | {_c(boot_mean(sub, 'n_boxes'), 2)} | "
                     f"{_c(boot_mean(sub, 'max_score'), 3)} | "
                     f"{_c(boot_mean(sub, 'mean_score'), 3)} | "
                     f"{_c(boot_mean(sub, 'ctl_any'), 1, pct=True)} | "
                     f"{_c(boot_mean(sub, 'ctl_max'), 3)} | "
                     f"{_c(boot_mean(sub, 'specificity'), 3)} |")

    if judged:
        L += ["", "## 9. Is the grounded phrase supported by the image?", "",
              "Vision judge over a blind sample, one step per prompt per arm.", "",
              "| arm | n | support yes | partly | no | boxes on target (yes) | failed |",
              "|---|---|---|---|---|---|---|"]
        by_arm = defaultdict(list)
        for r in judged:
            by_arm[r["arm"]].append(r)
        for a in sorted(by_arm):
            R = by_arm[a]
            n = len(R)
            def frac(field, val):
                ok = [r for r in R if r.get(field)]
                return f"{100.0 * sum(1 for r in ok if r[field] == val) / max(len(ok), 1):.0f}%"
            L.append(f"| {a} | {n} | {frac('support','yes')} | {frac('support','partly')} | "
                     f"{frac('support','no')} | {frac('boxes_ok','yes')} | "
                     f"{sum(1 for r in R if r.get('judge_error'))} |")

    hb_file = out_dir / "humanbox.json"
    if hb_file.exists():
        hb = json.loads(hb_file.read_text())["rows"]
        hb_arms = sorted({r["arm"] for r in hb})
        base = (text or {}).get("base") or hb_arms[0]
        L += ["", "## 10. The named region against the ANNOTATED one", "",
              "Saliency-R1-8K ships one human box per row. `enrichment` is the share of "
              "the union that falls inside that box over the share expected if the union "
              "were placed at chance, so 1.0 is chance and the union's size is divided "
              "out.", "",
              "| arm | steps | recall of the human box | precision | enrichment | IoU |",
              "|---|---|---|---|---|---|"]
        for a in [base] + [x for x in hb_arms if x != base]:
            R = [r for r in hb if r["arm"] == a]
            L.append(f"| {a} | {len(R)} | {_c(boot_mean(R, 'recall'), 3)} | "
                     f"{_c(boot_mean(R, 'precision'), 3)} | "
                     f"{_c(boot_mean(R, 'enrichment'), 3)} | {_c(boot_mean(R, 'iou'), 3)} |")
        L += ["", "| arm | Δ recall | Δ precision | Δ enrichment | Δ IoU |",
              "|---|---|---|---|---|"]
        B = [r for r in hb if r["arm"] == base]
        for a in hb_arms:
            if a == base:
                continue
            A_ = [r for r in hb if r["arm"] == a]
            L.append(f"| {a} | {_d(boot_diff(A_, B, 'recall'), 3)} | "
                     f"{_d(boot_diff(A_, B, 'precision'), 3)} | "
                     f"{_d(boot_diff(A_, B, 'enrichment'), 3)} | "
                     f"{_d(boot_diff(A_, B, 'iou'), 3)} |")

    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    print(f"[report] -> {out_dir}/report.md")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True,
                   choices=["text", "crossmap", "dino", "crosspass", "sheet", "humanbox", "figs", "report"])
    p.add_argument("--probe", action="append", default=[],
                   help="probe_merged.json (or its directory); repeatable")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--base", default="base_coldstart",
                   help="the BEFORE arm every delta is taken against")
    p.add_argument("--keep-arm", action="append", default=[])
    p.add_argument("--drop-arm", action="append", default=[])
    # dino
    p.add_argument("--controls", type=int, default=3,
                   help="wrong-image DINO calls per step (0 = off)")
    p.add_argument("--box-threshold", type=float, default=0.1,
                   help="must match the run's --overlap_box_threshold or the boxes are "
                        "not the ones the reward saw")
    p.add_argument("--dino-batch-size", type=int, default=8)
    # crosspass
    p.add_argument("--text-arm", action="append", default=[])
    p.add_argument("--map-arm", action="append", default=[])
    p.add_argument("--overlap-layer", type=int, default=22)
    p.add_argument("--overlap-heads", default="28,31")
    p.add_argument("--nulls", action="store_true",
                   help="also score each step's map against the two matched nulls -- its "
                        "own union translated, and the other observe steps' unions -- so "
                        "an in-region gain can be told apart from the map drifting "
                        "toward where the named regions happen to sit")
    p.add_argument("--all-heads", action="store_true",
                   help="also score every statistic on the mean over ALL heads of the "
                        "layer (fields suffixed `_all`), off the same capture -- so "
                        "'the attention did not move' cannot hide in the 34 heads the "
                        "reward never read")
    # humanbox
    p.add_argument("--human-box-dataset", default="peterant330/saliency-r1-8k",
                   help="the corpus whose `bbox` column is the annotated region")
    # sheet
    p.add_argument("--n-sheet", type=int, default=25,
                   help="items per arm on the manual review sheet")
    p.add_argument("--judge", action="store_true",
                   help="also score the sheet with a vision judge (needs NVIDIA_API_KEY)")
    # shared
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--merge", action="store_true",
                   help="merge this stage's shard files instead of running it")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "text":
        stage_text(args, out_dir)
    elif args.stage == "crossmap":
        stage_crossmap(args, out_dir)
    elif args.stage == "dino":
        if args.merge:
            merge_dino(out_dir)
        else:
            stage_dino(args, out_dir)
    elif args.stage == "crosspass":
        if args.merge:
            merge_crosspass(out_dir)
        else:
            stage_crosspass(args, out_dir)
    elif args.stage == "sheet":
        stage_sheet(args, out_dir)
    elif args.stage == "humanbox":
        stage_humanbox(args, out_dir)
    elif args.stage == "figs":
        stage_figs(args, out_dir)
    elif args.stage == "report":
        merge_dino(out_dir)
        merge_crosspass(out_dir)
        stage_report(args, out_dir)


if __name__ == "__main__":
    main()
