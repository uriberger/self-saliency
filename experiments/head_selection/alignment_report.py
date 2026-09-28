"""Phase 0 gate: is there a head-set with BOTH box alignment AND correctness signal?

Consumes the per-sample x per-head arrays written by head_alignment_scan.py and
answers the question that decides the reward search
(wiki/hack-resistant-overlap-reward-plan.md):

    The current reward heads (22,28)/(22,31) were chosen by correlation with
    correctness alone, and are BELOW chance at actually attending to the box. Is
    there any head that is above chance AND predicts correctness -- consistently
    across datasets and across box sources?

Three screens:

  1. ALIGNMENT   mean lift (chance 1.0) and mean AUROC (chance 0.5) per head.
  2. CORRECTNESS point-biserial r between each metric and the correctness label.
  3. JOINT       heads clearing both, on >=2 datasets and >=2 box sources.

Plus two audits the plan calls for:

  - selection optimism, via repeated 80/20 splits: pick the argmax-|r| head on
    train, score it on held-out test. The gap is how much of a "best head" is
    fitting noise.
  - enrichment mean vs median rankings. The old sweeps ranked heads by the MEAN
    of a statistic with skew up to 21; if mean- and median-based rankings
    disagree, those rankings are unreliable.

    python analysis/head_alignment_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "results" / "analysis" / "head_alignment"

# Collections used to SELECT heads. The GRPO-trained runs are deliberately excluded:
# their attention was shaped by the very reward under review, so selecting on them
# would be circular. They are used only for the drift monitors at the end.
SELECT = [
    ("saliency_r1_8k_dino",   "dino"),
    ("saliency_r1_8k_human",  "human"),
    ("visual_cot_dino",       "dino"),
    ("visual_cot_human",      "human"),
    ("virl39k_dino",          "dino"),
    ("mmstar_sft",            "dino"),
    ("pope_sft",              "dino"),
]
TRAINED = [(22, 28), (22, 31)]
ALIGN_METRICS = ("lift", "auroc")
CORR_METRICS = ("mean_in", "lift", "auroc", "enrichment", "enrichment_med", "sum_in", "mass_in")


def _pointbiserial(x, y):
    """r between (N, L, H) metric values and (N,) binary y, vectorised over heads."""
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 < 10 or n0 < 10:
        return np.full(x.shape[1:], np.nan, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        m1 = np.nanmean(x[y], 0)
        m0 = np.nanmean(x[~y], 0)
        sd = np.nanstd(x, 0)
        n = x.shape[0]
        r = (m1 - m0) / np.where(sd > 0, sd, np.nan) * np.sqrt(n1 * n0 / (n * n))
    return r.astype(np.float32)


def load(path: Path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--splits", type=int, default=40, help="repeated 80/20 splits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    d = Path(args.dir)
    align, corr, meta = {}, {}, {}
    rng = np.random.default_rng(args.seed)
    optimism = {}

    print("=" * 100)
    print("SCREEN 1+2 -- alignment (chance: lift 1.0, AUROC 0.5) and correctness correlation")
    print("=" * 100)
    hdr = (f"{'collection':<22} {'src':<6} {'n':>6} | {'(22,28)':>16} {'(22,31)':>16} | "
           f"{'best AUROC head':>22} {'>chance':>9}")
    print(hdr)
    print(f"{'':22} {'':6} {'':6} | {'lift / auroc':>16} {'lift / auroc':>16} |")
    print("-" * 100)

    for name, src in SELECT:
        p = d / f"{name}.npz"
        if not p.exists():
            print(f"{name:<22} MISSING")
            continue
        z = load(p)
        n = len(z["sample_ids"])
        a = {m: np.nanmean(z[m], 0) for m in ALIGN_METRICS}
        align[name] = a
        meta[name] = (src, n, z)

        auc = a["auroc"]
        bi = int(np.nanargmax(auc))
        L, H = auc.shape
        best = f"L{bi // H}H{bi % H}={auc.ravel()[bi]:.3f}"
        t1 = f"{a['lift'][22,28]:.3f} / {auc[22,28]:.3f}"
        t2 = f"{a['lift'][22,31]:.3f} / {auc[22,31]:.3f}"
        print(f"{name:<22} {src:<6} {n:>6} | {t1:>16} {t2:>16} | {best:>22} "
              f"{int(np.nansum(auc > 0.5)):>5}/{auc.size}")

        y = z["correct"]
        ok = y >= 0
        corr[name] = {m: _pointbiserial(z[m][ok], y[ok]) for m in CORR_METRICS if m in z}

        # selection optimism: argmax-|r| on train, scored on held-out test
        xs = z["lift"][ok].reshape(int(ok.sum()), -1)
        yy = y[ok].astype(bool)
        gaps = []
        for _ in range(args.splits):
            idx = rng.permutation(len(yy))
            k = int(0.8 * len(yy))
            tr, te = idx[:k], idx[k:]
            rtr = _pointbiserial(xs[tr][:, None, :], yy[tr])[0]
            j = int(np.nanargmax(np.abs(rtr)))
            rte = _pointbiserial(xs[te][:, None, j:j + 1], yy[te])[0, 0]
            gaps.append((abs(rtr[j]), abs(rte)))
        gaps = np.array(gaps)
        optimism[name] = (float(gaps[:, 0].mean()), float(np.nanmean(gaps[:, 1])))

    if not align:
        sys.exit("No scans found -- run scripts/run_head_alignment_scan.sh first")

    # ---------------- correctness table ----------------
    print("\n" + "=" * 100)
    print("SCREEN 2 -- |r| with correctness at the TRAINED heads, and the best head per metric")
    print("=" * 100)
    print(f"{'collection':<22} {'metric':<16} {'r(22,28)':>10} {'r(22,31)':>10} "
          f"{'best |r| head':>22} {'align of that head':>20}")
    print("-" * 100)
    for name, _src in SELECT:
        if name not in corr:
            continue
        auc = align[name]["auroc"]
        H = auc.shape[1]
        for m in ("mean_in", "lift", "auroc"):
            if m not in corr[name]:
                continue
            r = corr[name][m]
            bi = int(np.nanargmax(np.abs(r)))
            bh = f"L{bi // H}H{bi % H} r={r.ravel()[bi]:+.3f}"
            al = f"auroc={auc.ravel()[bi]:.3f}"
            print(f"{name:<22} {m:<16} {r[22,28]:>+10.3f} {r[22,31]:>+10.3f} {bh:>22} {al:>20}")
        print()

    # ---------------- joint screen ----------------
    print("=" * 100)
    print("SCREEN 3 -- JOINT GATE: above chance on alignment AND correlated with correctness")
    print("=" * 100)
    names = [n for n, _ in SELECT if n in align]
    srcs = {n: meta[n][0] for n in names}
    shape = align[names[0]]["auroc"].shape
    H = shape[1]

    above = np.stack([align[n]["auroc"] > 0.5 for n in names])          # (D, L, H)
    n_above = above.sum(0)
    n_src_above = np.zeros(shape, dtype=int)
    for s in ("dino", "human"):
        sel = [i for i, n in enumerate(names) if srcs[n] == s]
        if sel:
            n_src_above += above[sel].any(0).astype(int)

    # correctness: consistent SIGN and non-trivial magnitude, using lift
    rl = np.stack([corr[n]["lift"] for n in names if "lift" in corr[n]])
    pos = (rl > 0.05).sum(0)
    neg = (rl < -0.05).sum(0)

    gate = (n_above >= len(names) - 1) & (n_src_above >= 2) & (np.maximum(pos, neg) >= 3)
    print(f"heads above chance on >= {len(names)-1}/{len(names)} datasets and both box sources: "
          f"{int(((n_above >= len(names)-1) & (n_src_above >= 2)).sum())}/{shape[0]*shape[1]}")
    print(f"... of those, also |r(lift, correct)| > 0.05 with a consistent sign on >= 3 datasets: "
          f"{int(gate.sum())}")

    if gate.any():
        mean_auc = np.nanmean(np.stack([align[n]["auroc"] for n in names]), 0)
        mean_r = np.nanmean(rl, 0)
        cand = [(int(l), int(h)) for l, h in zip(*np.where(gate))]
        cand.sort(key=lambda lh: -mean_auc[lh])
        print(f"\n{'head':<10} {'mean AUROC':>11} {'min AUROC':>10} {'mean r(lift)':>13} "
              f"{'sign-consistent':>16}")
        print("-" * 64)
        for l, h in cand[:15]:
            per = np.array([align[n]["auroc"][l, h] for n in names])
            rs = rl[:, l, h]
            sgn = f"{int(max((rs>0).sum(), (rs<0).sum()))}/{len(rs)}"
            print(f"L{l}H{h:<7} {mean_auc[l,h]:>11.3f} {per.min():>10.3f} "
                  f"{mean_r[l,h]:>+13.3f} {sgn:>16}")
        print("\nfor reference, the two currently-rewarded heads:")
        for l, h in TRAINED:
            per = np.array([align[n]["auroc"][l, h] for n in names])
            rs = rl[:, l, h]
            sgn = f"{int(max((rs>0).sum(), (rs<0).sum()))}/{len(rs)}"
            print(f"L{l}H{h:<7} {np.nanmean(per):>11.3f} {per.min():>10.3f} "
                  f"{np.nanmean(rs):>+13.3f} {sgn:>16}")

    # ---------------- audits ----------------
    print("\n" + "=" * 100)
    print("AUDIT A -- selection optimism (argmax-|r| on 80%, scored on held-out 20%)")
    print("=" * 100)
    print(f"{'collection':<22} {'train |r|':>10} {'test |r|':>10} {'optimism':>10}")
    print("-" * 56)
    for n, (tr, te) in optimism.items():
        print(f"{n:<22} {tr:>10.3f} {te:>10.3f} {tr - te:>10.3f}")

    print("\n" + "=" * 100)
    print("AUDIT B -- enrichment mean vs median: do the old head rankings survive?")
    print("=" * 100)
    print(f"{'collection':<22} {'top-10 overlap':>15} {'rank corr':>11} {'argmax mean':>13} {'argmax med':>12}")
    print("-" * 78)
    for n in names:
        z = meta[n][2]
        if "enrichment" not in z or "enrichment_med" not in z:
            continue
        a = np.nanmean(z["enrichment"], 0).ravel()
        b = np.nanmean(z["enrichment_med"], 0).ravel()
        good = np.isfinite(a) & np.isfinite(b)
        ta = set(np.argsort(-np.where(good, a, -np.inf))[:10].tolist())
        tb = set(np.argsort(-np.where(good, b, -np.inf))[:10].tolist())
        from scipy.stats import spearmanr
        rho = spearmanr(a[good], b[good]).statistic
        ia, ib = int(np.nanargmax(np.where(good, a, -np.inf))), int(np.nanargmax(np.where(good, b, -np.inf)))
        print(f"{n:<22} {len(ta & tb):>13}/10 {rho:>11.3f} "
              f"{'L%dH%d' % (ia // H, ia % H):>13} {'L%dH%d' % (ib // H, ib % H):>12}")

    if args.json_out:
        out = {
            "collections": {n: {"src": meta[n][0], "n": meta[n][1],
                                "auroc": align[n]["auroc"].tolist(),
                                "lift": align[n]["lift"].tolist(),
                                "r_lift": corr[n]["lift"].tolist()} for n in names},
            "gate_heads": [[int(l), int(h)] for l, h in zip(*np.where(gate))],
            "optimism": optimism,
        }
        Path(args.json_out).write_text(json.dumps(out))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
