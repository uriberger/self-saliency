#!/usr/bin/env python
"""Measure the per-sample SPREAD of each overlap metric from a probe run, and derive the
w_overlap that reproduces a reference metric's training pressure.

Why spread and not mean: GRPO normalises within the group
(``advantages = (reward - group_mean) / (group_std + 1e-4)``, scale_rewards defaults
True), so a constant offset in a reward term does nothing and only its VARIATION across
the completions of one prompt reaches the gradient. Two metrics with the same mean but
different sd apply different pressure at the same w_overlap. This is exactly how auroc's
0.11 was derived from mean_in's 0.4 (sd 0.13 vs 0.036 -> 0.4 x 0.036/0.13 ~ 0.11), and
it is the number a new metric needs before it can be trained with.

THREE SPREADS ARE REPORTED, and only two of them are the right one:

    sd/sample   mean over prompts of that prompt's own sd (ddof=1) across its scored
                completions. The incumbent weights were set from this column.
    sd_within   the same completions, group-mean-centred and then POOLED --
                sqrt( sum_g sum_i (x_gi - mean_g)^2 / sum_g (n_g - 1) ). This is the
                quantity the advantage actually sees, and it is what a new weight
                should be set from. It differs from sd/sample because a mean of
                standard deviations is not the standard deviation of the pool: small
                groups pull the mean around, and E[s] < sqrt(E[s^2]) always, so
                sd/sample sits slightly BELOW sd_within (a few % at n=8).
    sd all      across every scored completion in the corpus, ignoring groups. This is
                the WRONG one for weighting -- most of it is between-prompt variation
                that the group-mean subtraction removes before the gradient sees it --
                and it is reported only to show how much larger it is.

The trainer now logs the same quantity live, as
``train/rewards/<func>/within_group_std``, so a run can be checked against the weight it
was launched with instead of being reconstructed from the completions table afterwards.

The probe records every metric ungated on every grounded step (mean_in_raw,
mean_in_v2_raw, auroc_raw), so all of them are measured on IDENTICAL maps, masks and
completions -- a paired comparison, not one across runs that generated different text.

    python overlap_metric_spread.py outputs/overlap_probe/<run>/probe_merged.json

ONE PAIRING TRAP, on a `--map grad` probe run. Every row in the table is then computed on
the GRADIENT map, including `mean_in` -- so the in-table "w that matches mean_in's
pressure" is scaling against mean_in-on-gradients, which nothing was ever trained with.
The incumbent's spread is a property of the ATTENTION map: sd_per_sample = 0.0086, from
the 1074-grounded-step set_a run (see trl/rewards/overlap_rewards.py). So for w_grad, take
`sd_per_sample` for `logratio` from the grad run and compute

    w_grad = 0.4 * 0.0086 / sd_logratio

by hand, rather than reading the table's mean_in column. This is the one comparison that
cannot be paired -- the two maps cannot both be the incumbent -- so it inherits the +-25%
the reference weights already carry. On an ATTENTION run there is no trap: all four rows,
`logratio` included, are the same map and the table is paired as designed.

W_GLIMPSE, the same arithmetic, and the one thing that is EASIER here. `--map glimpse`
records all four `*_raw` metrics on every grounded step like every other run, so a single
glimpse probe run calibrates every variant on identical maps, masks and completions -- the
rows are paired against each other even though none is paired against the incumbent. Take
`sd_per_sample` for the variant being trained and compute

    w_glimpse = 0.4 * 0.0086 / sd_variant

by hand, exactly as for w_grad and with the same caveat: 0.0086 is the ATTENTION map's
spread (1074 grounded steps, set_a), so this is the one comparison that cannot be paired
and it inherits the +-25% the reference weights already carry. Do NOT reuse mean_in_v2's
incumbent 0.033 -- that was derived from mean_in_v2 on the ATTENTION map, and the GLIMPSE
map's spread is a different number.

    python overlap_probe.py --map glimpse --n-samples 40 \
        --out-dir outputs/overlap_probe/glimpse_spread --dataset cold_data/grpo_sets/set_a
    python overlap_metric_spread.py outputs/overlap_probe/glimpse_spread/probe_merged.json

Read `union_frac`/`box_area_frac` in the same report while you are there: mean_in_v2's
ceiling is n_patches/n_in, so its spread is partly a spread of union AREAS, and a w set
from it is paying for that too. auroc has no such term, which is the main practical
argument for the auroc variant over the mean_in_v2 one.

THIS SCRIPT'S TABLE IS ALWAYS THE UNCAPPED ONE. It has no `--max-union-area`, so every row
is computed over every grounded step -- while the glimpse runs on record train with
`--max-union-area 0.5`, which skips the 61% of steps whose union covers more than half the
image and moves each weight by up to 1.5x (measured 2026-08-24; the four uncapped/capped
w on set_a are mean_in 0.13/0.088, mean_in_v2 0.024/0.013, auroc 0.071/0.060, logratio
0.032/0.020). To get the capped column, filter the steps in `probe_merged.json` by
`box_area_frac` -- that field IS the union fraction `--max-union-area` gates on -- and
rerun `summarise`'s arithmetic over what survives.

The per-completion reward is reconstructed the way think_overlap_reward builds it:
mean over the completion's grounded observe steps, times the format gate, None when
there is no grounded step (masked, so it never enters the group). The optional mass
floor is NOT applied -- it is a per-step multiplier in [0,1] that would confound a
comparison of the metrics themselves.

THE PLACEBO TABLE (--placebo roll|random|length, docs/next-reward-experiments.md) is
printed underneath, from the SAME completions, and it is what those runs' weights come
from: w_placebo = w_ref x sd_within(reference) / sd_within(placebo). The three controls
are computed here by importing trl/rewards/placebo_rewards.py itself rather than
reimplementing them, so the number a run is launched with is measured on the same
function the run will use, and they inherit the reward's parity rule -- a completion
enters the placebo table only if the reference metric scored it.

`roll` needs the maps, so it is only available on a probe run made with
``--store-maps`` (the default). It is scored on the stored uint8 map, which is
peak-normalised, so `mean_in` and `auroc` are reproduced exactly up to 1/255 and
`mean_in_v2` up to the same (both are scale-invariant); `logratio` is not offered at
all, and neither is the placebo machinery -- see placebo_rewards.configure().
"""

from __future__ import annotations

import argparse
import base64
import math
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

# (json key on a step, display name). The probe writes all of these per grounded step.
# `logratio` is the roll-null score the gradient reward uses; it is recorded on every
# step whatever map the probe built, so on an attention run it is a paired comparison
# against the other three, and on a `--map grad` run it is the number w_grad comes from.
METRICS = [
    ("mean_in_raw", "mean_in"),
    ("mean_in_v2_raw", "mean_in_v2"),
    ("auroc_raw", "auroc"),
    ("logratio_raw", "logratio"),
]
CHANCE = {"mean_in": None, "mean_in_v2": 1.0, "auroc": 0.5, "logratio": 0.0}


def _load_placebo_rewards():
    """Import trl/rewards/placebo_rewards.py without importing the trl package.

    Same trick the CPU tests use, and for the same reason: `import trl` runs the lazy
    module machinery and drags in torch/transformers, none of which this script needs.
    The stub's __path__ points only at this checkout's trl/rewards/, so the sibling
    imports inside placebo_rewards (roll_null, overlap_rewards) resolve there and
    nowhere else. Returns None if the sources are missing (an older checkout).
    """
    src = ROOT / "trl" / "rewards" / "placebo_rewards.py"
    if not src.exists():
        return None
    for name, path in (("trl_spread", src.parent.parent), ("trl_spread.rewards", src.parent)):
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m
    spec = importlib.util.spec_from_file_location("trl_spread.rewards.placebo_rewards", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trl_spread.rewards.placebo_rewards"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_maskfree_rewards():
    """Import trl/rewards/maskfree_rewards.py the same way, and for the same reason.

    Safe to call after _load_placebo_rewards(): both register the same `trl_spread`
    stub packages, and re-registering them is idempotent.
    """
    src = ROOT / "trl" / "rewards" / "maskfree_rewards.py"
    if not src.exists():
        return None
    for name, path in (("trl_spread", src.parent.parent), ("trl_spread.rewards", src.parent)):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [str(path)]
            sys.modules[name] = m
    spec = importlib.util.spec_from_file_location("trl_spread.rewards.maskfree_rewards", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trl_spread.rewards.maskfree_rewards"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_length_guard_rewards():
    """Import trl/rewards/length_guard_rewards.py the same way, and for the same reason.

    Safe to call after the two loaders above: they all register the same `trl_spread`
    stub package, and this module imports nothing from its siblings.
    """
    src = ROOT / "trl" / "rewards" / "length_guard_rewards.py"
    pkg = sys.modules.get("trl_spread")
    if pkg is None:
        pkg = types.ModuleType("trl_spread"); pkg.__path__ = [str(ROOT / "trl")]
        sys.modules["trl_spread"] = pkg
        sub = types.ModuleType("trl_spread.rewards"); sub.__path__ = [str(src.parent)]
        sys.modules["trl_spread.rewards"] = sub
    spec = importlib.util.spec_from_file_location("trl_spread.rewards.length_guard_rewards", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trl_spread.rewards.length_guard_rewards"] = mod
    spec.loader.exec_module(mod)
    return mod


def length_guard_group_vals(samples, ref_key, lgr, l_ref, restrict_to_reference):
    """Per-group length-guard penalties, from the probe's stored `n_completion_tokens`.

    TWO SETS, and the difference is the point of printing both rows.

    `restrict_to_reference=True` keeps only the completions `ref_key` scored, which is the
    set every other row in this table is measured on. That is the set the WEIGHT must come
    from: `w = w_ref x sd_within(reference) / sd_within(guard)` is only the ratio it claims
    to be if both sds are over the same completions.

    `restrict_to_reference=False` is what the reward actually does at training time -- it
    scores EVERY completion and never returns None, because masking it to the overlap
    reward's scored set would make "produce no groundable observe step" an escape hatch
    from the leash (see length_guard_rewards.__doc__, SCORED SET). That row is the one
    whose sd_within governs the pressure a run really applies, so it is printed too and
    the two are expected to differ.

    No format gate either, for the same reason: an unparseable completion still has a
    length, and the guard is not conditional on the rest of the reward stack agreeing.
    """
    group_vals = []
    for s in samples:
        vals = []
        for c in s["completions"]:
            if restrict_to_reference:
                if not any(st.get("grounded") and st.get(ref_key) is not None
                           for st in c["observe_steps"]):
                    continue
            n = c.get("n_completion_tokens")
            if n is None:
                continue
            vals.append(lgr.penalty(int(n), l_ref))
        if vals:
            group_vals.append(vals)
    return group_vals


def maskfree_group_vals(samples, ref_key, kind, mfr):
    """Per-group mask-free rewards, on exactly the completions `ref_key` scored.

    NOT what the reward does at training time -- there it scores every completion with a
    gradeable observe step, because without DINO the reference's gate cannot be evaluated.
    Here the reference's set is imposed so the sd is measured on the SAME completions as
    the mean_in row it is about to be divided by; a weight derived from a different set
    would not be the ratio it claims to be. The two sets were measured identical on
    val_natural anyway (231/240 either way) -- see maskfree_rewards.__doc__.

    `mass` MUST NOT read `_decode_step_map`. That decoder returns the map normalised to
    its own peak, which is exactly right for every metric offered until now -- mean_in
    divides by the peak, mean_in_v2 is a ratio of two means of the same map, auroc is
    rank-only, logratio is a log ratio -- and exactly wrong for the first metric that is
    NOT scale-invariant: a peak-normalised map sums to ~8, not to the ~0.004 of real
    attention mass, and its spread is the spread of a normalisation constant. The probe
    stores the true total as `image_mass` (= smap.sum(), written on every step whether or
    not maps were kept), so that is what is read here. `flatness` is scale-invariant and
    can use the decoder safely.
    """
    group_vals = []
    for s in samples:
        vals = []
        for c in s["completions"]:
            steps = [st for st in c["observe_steps"]
                     if st.get("grounded") and st.get(ref_key) is not None]
            if not steps:
                continue                       # masked by the reference -> masked here
            gate = 1.0 if c["format_valid"] else 0.0
            per_step = []
            for st in steps:
                if kind == "mass":
                    total = st.get("image_mass")
                    if total is None or not (total > 0):
                        continue
                    per_step.append(math.log(float(total)) + mfr._CFG["mass_anchor"])
                    continue
                m, _mask = _decode_step_map(st)
                if m is None:
                    continue
                v = mfr.flatness(m)
                if v is not None:
                    per_step.append(v)
            if not per_step:
                continue
            vals.append(float(np.mean(per_step)) * gate)
        if vals:
            group_vals.append(vals)
    return group_vals


def pooled_within_sd(group_vals):
    """sqrt( sum_g sum_i (x_gi - mean_g)^2 / sum_g (n_g - 1) ) -- see the module docstring."""
    ss, dof = 0.0, 0
    for vals in group_vals:
        if len(vals) < 2:
            continue
        a = np.asarray(vals, dtype=np.float64)
        ss += float(((a - a.mean()) ** 2).sum())
        dof += len(a) - 1
    return float(np.sqrt(ss / dof)) if dof else float("nan")


def completion_reward(completion, key):
    """think_overlap_reward for one completion under one metric, mass floor off."""
    vals = [s[key] for s in completion["observe_steps"]
            if s.get("grounded") and s.get(key) is not None]
    if not vals:
        return None                                    # masked: no grounded step
    return float(np.mean(vals)) * (1.0 if completion["format_valid"] else 0.0)


def _stats(name, group_vals, per_step=None):
    """Row shared by the metric table and the placebo table."""
    per_completion = [v for g in group_vals for v in g]
    if not per_completion:
        return None
    group_sds = [float(np.std(g, ddof=1)) for g in group_vals if len(g) >= 2]
    out = {
        "name": name,
        "n_steps": len(per_step) if per_step is not None else 0,
        "n_completions": len(per_completion),
        "n_groups": len(group_sds),
        "step_mean": float(np.mean(per_step)) if per_step else float("nan"),
        "reward_mean": float(np.mean(per_completion)),
        "reward_sd_all": float(np.std(per_completion, ddof=1)) if len(per_completion) > 1 else float("nan"),
        # Mean of per-group sds: the column the incumbent weights were set from.
        "sd_per_sample": float(np.mean(group_sds)) if group_sds else float("nan"),
        # Group-mean-centred and pooled: the spread the advantage actually sees.
        "sd_within": pooled_within_sd(group_vals),
    }
    chance = CHANCE.get(name)
    if chance is not None and per_step:
        out["frac_steps_above_chance"] = float(np.mean(np.asarray(per_step) > chance))
    return out


def summarise(samples, key, name):
    per_step, group_vals = [], []
    for s in samples:
        vals = []
        for c in s["completions"]:
            per_step += [st[key] for st in c["observe_steps"]
                         if st.get("grounded") and st.get(key) is not None]
            r = completion_reward(c, key)
            if r is not None:
                vals.append(r)
        if vals:
            group_vals.append(vals)
    return _stats(name, group_vals, per_step)


# ---------------------------------------------------------------------------
# the placebo controls (--placebo roll|random|length)
# ---------------------------------------------------------------------------

_METRIC_FN = {"mean_in": "_mean_in", "mean_in_v2": "_mean_in_v2", "auroc": "_auroc"}


def _decode_step_map(step):
    """(map, mask) from a --store-maps probe step, or (None, None).

    `map_q` is the map normalised to its OWN peak and quantised to uint8, which is all
    three offered metrics need: mean_in divides by that peak anyway, mean_in_v2 is a
    ratio of two means of the same map, and auroc depends only on the patch order.
    """
    grid, mq, kq = step.get("grid"), step.get("map_q"), step.get("mask_q")
    if not grid or not mq or not kq:
        return None, None
    gh, gw = int(grid[0]), int(grid[1])
    m = np.frombuffer(base64.b64decode(mq), dtype=np.uint8)
    k = np.frombuffer(base64.b64decode(kq), dtype=np.uint8)
    if m.size != gh * gw or k.size != gh * gw:
        return None, None
    return m.reshape(gh, gw).astype(np.float64) / 255.0, k.reshape(gh, gw).astype(bool)


def placebo_group_vals(samples, ref_key, kind, plc, orw, metric, seed):
    """Per-group placebo rewards, on exactly the completions `ref_key` scored.

    Mirrors think_placebo_reward: the reference metric decides WHICH completions and
    which steps count, the placebo only replaces the per-step value. Returns
    (group_vals, n_dropped) -- n_dropped counts completions the reference scored but the
    placebo could not, which is 0 for random/length and non-zero for `roll` only when a
    step's stored map is missing or its union cannot be translated at all.
    """
    metric_fn = getattr(orw, _METRIC_FN[metric])
    group_vals, dropped = [], 0
    for s in samples:
        question = s.get("question", "") or ""
        vals = []
        for c in s["completions"]:
            steps = [st for st in c["observe_steps"]
                     if st.get("grounded") and st.get(ref_key) is not None]
            if not steps:
                continue                       # masked by the reference -> masked here
            gate = 1.0 if c["format_valid"] else 0.0
            if kind == "length":
                v = plc.length_score(c["n_completion_tokens"])
            elif kind == "random":
                v = plc.uniform01(c["text"], seed)
            else:
                per_step = []
                for st in steps:
                    m, mask = _decode_step_map(st)
                    if m is None:
                        continue
                    rolled, _info = plc.roll_mask(mask, question, st["text"], seed=seed)
                    if rolled is None:
                        continue
                    r = metric_fn(m, rolled)
                    if r is not None:
                        per_step.append(r)
                if not per_step:
                    dropped += 1
                    continue
                v = float(np.mean(per_step))
            vals.append(v * gate)
        if vals:
            group_vals.append(vals)
    return group_vals, dropped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("probe_json", help="probe_merged.json from a probe run (or its directory)")
    p.add_argument("--reference", default="mean_in",
                   help="metric whose training pressure is being matched (default mean_in)")
    p.add_argument("--reference-weights", default="0.2,0.4",
                   help="w_overlap values already trained with the reference metric")
    p.add_argument("--placebo-seed", type=int, default=0,
                   help="--rollnull_seed the placebo run will use; sets both the roll "
                        "offsets and the `random` draw, so the weight is measured on the "
                        "same numbers the run will see")
    p.add_argument("--no-placebo", action="store_true",
                   help="skip the --placebo roll|random|length table")
    # The length guard's shape knobs, so the weight can be measured for the exact band a
    # run will be launched with rather than for the default one.
    p.add_argument("--length-guard-ref", type=float, default=None,
                   help="reference length in tokens for the lenguard rows. Default: each "
                        "model's own mean completion length, which is what --length-guard "
                        "wants when the model is the cold-start policy you train from")
    p.add_argument("--length-guard-band-lo", type=float, default=0.30,
                   help="lower edge of the free window, as a MULTIPLE of the reference length")
    p.add_argument("--length-guard-band-hi", type=float, default=3.0,
                   help="upper edge of the free window, as a MULTIPLE of the reference length")
    p.add_argument("--length-guard-knee", type=float, default=1.0)
    p.add_argument("--no-length-guard", action="store_true",
                   help="skip the --length-guard rows")
    args = p.parse_args()

    path = Path(args.probe_json)
    if path.is_dir():
        path = path / "probe_merged.json"
    payload = json.loads(path.read_text())
    cfg = payload.get("config", {})
    print(f"probe: {path}")
    print(f"  dataset={cfg.get('dataset')} n_samples={cfg.get('n_samples')} "
          f"gens={cfg.get('num_generations')} temp={cfg.get('temperature')}")
    print(f"  layer={cfg.get('overlap_layer')} heads={cfg.get('overlap_heads')} "
          f"token_reduction={cfg.get('token_reduction')} "
          f"box_threshold={cfg.get('box_threshold')} max_box_area={cfg.get('max_box_area')}")

    w_refs = [float(w) for w in args.reference_weights.split(",") if w.strip()]

    plc = None if args.no_placebo else _load_placebo_rewards()
    # The mask-free rows need the stored maps for the same reason `placebo:roll` does.
    mfr = None if args.no_placebo else _load_maskfree_rewards()
    # The length guard needs neither maps nor boxes, only the stored token counts, so it
    # is measurable on any probe -- including one run without --store-maps.
    lgr = None if args.no_length_guard else _load_length_guard_rewards()
    ref_key = next((k for k, n in METRICS if n == args.reference), None)

    for model, m in payload["models"].items():
        rows = [r for r in (summarise(m["samples"], k, n) for k, n in METRICS) if r]
        if not rows:
            print(f"\n=== {model}: no scored steps ===")
            continue
        print(f"\n=== {model} ===")
        print(f"{'metric':<18} {'steps':>7} {'compl':>6} {'step mean':>10} {'reward mean':>12} "
              f"{'sd_within':>10} {'sd/sample':>10} {'sd all':>8} {'>chance':>8}")
        for r in rows:
            ac = r.get("frac_steps_above_chance")
            print(f"{r['name']:<18} {r['n_steps']:>7} {r['n_completions']:>6} "
                  f"{r['step_mean']:>10.4f} {r['reward_mean']:>12.4f} "
                  f"{r['sd_within']:>10.4f} {r['sd_per_sample']:>10.4f} {r['reward_sd_all']:>8.4f} "
                  f"{'--' if ac is None else f'{ac:>8.3f}'}")

        # --- the placebo controls, on the same completions -------------------------
        if plc is not None and ref_key is not None and args.reference in _METRIC_FN:
            for kind in plc.KINDS:
                gv, dropped = placebo_group_vals(
                    m["samples"], ref_key, kind, plc, plc._ORW, args.reference, args.placebo_seed
                )
                row = _stats(f"placebo:{kind}", gv)
                if row is None:
                    print(f"placebo:{kind:<6} -- not measurable here"
                          + (" (no stored maps: rerun the probe with --store-maps)"
                             if kind == "roll" else ""))
                    continue
                rows.append(row)
                note = f"   [{dropped} completions the reference scored had no rollable step]" if dropped else ""
                ac = row.get("frac_steps_above_chance")
                print(f"{row['name']:<18} {'--':>7} {row['n_completions']:>6} "
                      f"{'--':>10} {row['reward_mean']:>12.4f} "
                      f"{row['sd_within']:>10.4f} {row['sd_per_sample']:>10.4f} "
                      f"{row['reward_sd_all']:>8.4f} {'--' if ac is None else f'{ac:>8.3f}'}{note}")

        # --- the mask-free rewards (--maskfree flatness|mass), same completions --------
        if mfr is not None and ref_key is not None:
            for kind in mfr.KINDS:
                row = _stats(f"maskfree:{kind}", maskfree_group_vals(m["samples"], ref_key, kind, mfr))
                if row is None:
                    print(f"maskfree:{kind:<5} -- not measurable here "
                          f"(no stored maps: rerun the probe with --store-maps)")
                    continue
                rows.append(row)
                ac = row.get("frac_steps_above_chance")
                print(f"{row['name']:<18} {'--':>7} {row['n_completions']:>6} "
                      f"{'--':>10} {row['reward_mean']:>12.4f} "
                      f"{row['sd_within']:>10.4f} {row['sd_per_sample']:>10.4f} "
                      f"{row['reward_sd_all']:>8.4f} {'--' if ac is None else f'{ac:>8.3f}'}")

        # --- the length guard (--length-guard REF_TOKENS) ------------------------------
        # Needs no maps, no boxes and no metric -- only the stored token counts -- so it is
        # the one row that is always measurable. l_ref defaults to THIS model's own mean
        # completion length, which is exactly the number --length-guard wants: the base
        # policy's mean length on the corpus being trained on. Pass --length-guard-ref to
        # score a different reference (e.g. a trained checkpoint against the cold start's).
        if lgr is not None:
            _all_n = [int(c["n_completion_tokens"]) for s in m["samples"] for c in s["completions"]
                      if c.get("n_completion_tokens") is not None]
            if _all_n:
                l_ref = float(args.length_guard_ref) if args.length_guard_ref else float(np.mean(_all_n))
                lgr.configure(l_ref=l_ref, band_lo=args.length_guard_band_lo,
                              band_hi=args.length_guard_band_hi,
                              knee=args.length_guard_knee)
                print(f"lenguard: l_ref={l_ref:.1f} tokens"
                      f"{'  (--length-guard-ref)' if args.length_guard_ref else '  (= this model mean length)'}"
                      f"  free band {args.length_guard_band_lo}x..{args.length_guard_band_hi}x ="
                      f" [{l_ref * args.length_guard_band_lo:.0f}, {l_ref * args.length_guard_band_hi:.0f}] tok"
                      f"  n={len(_all_n)} completions, mean {np.mean(_all_n):.1f}")
                for restrict, label in ((True, "lenguard"), (False, "lenguard:all")):
                    row = _stats(label, length_guard_group_vals(
                        m["samples"], ref_key, lgr, l_ref, restrict))
                    if row is None:
                        continue
                    # Only the reference-restricted row joins `rows`: the weight table
                    # below divides sd_within(reference) by each row's, and that ratio is
                    # meaningless across two different completion sets.
                    if restrict:
                        rows.append(row)
                    print(f"{row['name']:<18} {'--':>7} {row['n_completions']:>6} "
                          f"{'--':>10} {row['reward_mean']:>12.4f} "
                          f"{row['sd_within']:>10.4f} {row['sd_per_sample']:>10.4f} "
                          f"{row['reward_sd_all']:>8.4f} {'--':>8}"
                          f"{'' if restrict else '   [every completion: the set the reward really scores]'}")
                # The weight table below answers "what w makes this term as strong as
                # mean_in", which is the right question for a metric taking the overlap
                # reward's slot and the WRONG one for a regulator. The guard is meant to be
                # silent in the middle and expensive at the edges, not co-equal everywhere,
                # so its k comes from the two cost figures printed here -- not from the
                # `lenguard` row of that table. See length_guard_rewards.__doc__.
                _k = 0.20
                print(f"{'':18} k={_k}: pressure {_k * (row['sd_within'] if row else 0):.4f} "
                      f"| cost at 0.2x ref {_k * lgr.penalty(int(0.2 * l_ref), l_ref):+.3f}, "
                      f"at 0.1x {_k * lgr.penalty(int(0.1 * l_ref), l_ref):+.3f}, "
                      f"at 4x {_k * lgr.penalty(int(4 * l_ref), l_ref):+.3f}"
                      f"   <- set k from THESE, not from the w table below")

        ref = next((r for r in rows if r["name"] == args.reference), None)
        if not ref:
            continue
        for basis in ("sd_within", "sd_per_sample"):
            if not np.isfinite(ref[basis]) or ref[basis] <= 0:
                continue
            print(f"\n  w that matches {args.reference}'s pressure, on {basis} "
                  f"(w_ref x {basis}_{args.reference}/{basis}_row):")
            print("  " + f"{'metric':<18}" + "".join(f"{f'w_ref={w}':>14}" for w in w_refs)
                  + f"{'sd ratio':>11}")
            for r in rows:
                if not np.isfinite(r[basis]) or r[basis] <= 0:
                    continue
                ratio = ref[basis] / r[basis]
                cells = "".join(f"{w * ratio:>14.3f}" for w in w_refs)
                print(f"  {r['name']:<18}{cells}{ratio:>11.3f}")
        print("\n  sd_within is the one to weight from; sd/sample is what the incumbent "
              "weights were set from and is kept for continuity.")


if __name__ == "__main__":
    main()
