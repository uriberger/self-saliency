#!/usr/bin/env python
"""Where the attention goes, beside where the human boxes ARE -- as two maps of one shape.

    python sink_box_coverage.py --dirs A,B,C,D --out-dir DIR

Two quantities per patch, on each model's own modal lattice:

  ATTENTION   the mean pooled attention map over the GENERATED tokens, exactly as
              `sink_location_xmodel_tables.py` builds it -- this file imports that
              module's `model_maps`/`resample_map`/`table_from_map` rather than
              reimplementing them, so its attention table IS that table.

  HUMAN BOX   the fraction of the 1,800 pictures in which this patch fell inside the
              corpus's annotated answer box. A frequency, not an average of weights.

Both are then divided by their own mean over the lattice, so both read on the same
fair-share scale: 1.0 is what a patch would get if the quantity were spread evenly, above
1.0 is over-represented. That is what makes the two panels of a row comparable at all --
the raw units (a probability mass, a coverage frequency) are not.

WHY THE BOX MAP IS A PLAIN FREQUENCY. Averaging `mask / mask.sum()` instead would weight
each picture by 1/(its box's patch count), so a handful of tiny boxes would dominate the
map. The question here is "how often is this patch inside the answer region", which is the
unweighted mean of the indicator -- one picture, one vote -- and its grand mean over the
lattice is then just the mean box area. Both conventions are reported; they differ, and
the difference is entirely about box size.

WHAT THE BOX MAP DOES *NOT* DEPEND ON. The boxes are the corpus's, identical for every
model. A row's right-hand panel therefore differs from another row's ONLY through the
lattice its model's grids were resampled onto -- so four near-identical box panels are the
expected result, not a bug, and any visible difference between them is a geometry
artefact rather than a finding about that model.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(REPO))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def box_mask(box, gh, gw):
    """A normalised box -> the patches whose CENTRES it covers. -> [gh,gw] bool.

    The same centre-in-box convention `sink_three_legs` and `sink_observe_boxes` use, so
    a coverage map and a per-picture ring enrichment are rasterised identically.
    """
    x0, y0, x1, y1 = box
    c = (np.arange(gw) + 0.5) / gw
    r = (np.arange(gh) + 0.5) / gh
    m = ((c[None, :] >= x0) & (c[None, :] <= x1)
         & (r[:, None] >= y0) & (r[:, None] <= y1))
    if not m.any():
        m[min(gh - 1, max(0, int((y0 + y1) / 2 * gh))),
          min(gw - 1, max(0, int((x0 + x1) / 2 * gw)))] = True
    return m


def coverage_map(meta, man, XM, lattice, weighted=False):
    """Per-patch box statistic on a common lattice. -> ([GH,GW] enrichment, n, mean area).

    `weighted=False` is the frequency the docstring describes: the mean over pictures of
    the 0/1 indicator, area-weighted onto the lattice. `weighted=True` is the
    distribution convention the attention map uses -- `mask / mask.sum()` -- kept so the
    two can be printed side by side rather than argued about.
    """
    GH, GW = lattice
    acc, n, areas = np.zeros((GH, GW)), 0, []
    for m in meta:
        row = man.get(m["key"])
        if row is None or not row.get("bbox"):
            continue
        gh, gw = m["grid"]
        mk = box_mask([float(v) for v in row["bbox"]], gh, gw)
        k = int(mk.sum())
        if k == 0:
            continue
        field = mk.astype(np.float64) / k if weighted else mk.astype(np.float64)
        # `resample_map` multiplies by gh*gw on the way in (it takes a distribution and
        # returns enrichment), so dividing first makes it a plain area-weighted MEAN of
        # whatever field is handed to it.
        acc += XM.resample_map(field / (gh * gw), gh, gw, GH, GW)
        areas.append(k / (gh * gw))
        n += 1
    if n == 0:
        return None, 0, float("nan")
    mean_field = acc / n
    grand = float(mean_field.mean())
    if grand <= 0:
        return None, 0, float("nan")
    return mean_field / grand, n, float(np.mean(areas))


def draw_grid(rows, path, note=""):
    """One figure: a row per model, a column per quantity. Shared diverging scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    XM = sys.modules["_bc_tables"]
    cmap = LinearSegmentedColormap.from_list("fair", XM.DIVERGING)
    CLIP = XM.CLIP
    nr, nc = len(rows), 2

    # Constrained layout, because every panel is aspect-locked by `imshow` and a fixed
    # subplot grid leaves a band of dead space above each row that no `bbox_inches`
    # trim can reach -- it is between the axes, not around them.
    fig, axes = plt.subplots(nr, nc, figsize=(6.4, 2.0 * nr + 0.8), dpi=200,
                             squeeze=False, layout="constrained")
    fig.patch.set_facecolor(XM.SURFACE)
    for i, (label, mats) in enumerate(rows):
        for j, (col_title, mat) in enumerate(mats):
            ax = axes[i][j]
            ax.set_facecolor(XM.SURFACE)
            if mat is None:
                ax.axis("off")
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.log2(np.where(mat > 0, mat, np.nan))
            ax.imshow(np.clip(z, -CLIP, CLIP), cmap=cmap, vmin=-CLIP, vmax=CLIP,
                      interpolation="nearest")
            GH, GW = mat.shape
            ax.set_xticks(np.arange(-0.5, GW, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, GH, 1), minor=True)
            ax.grid(which="minor", color=XM.SURFACE, linewidth=0.4)
            ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
            for s in ax.spines.values():
                s.set_visible(False)
            # The four corners carry the claim, so they are always labelled outright
            # rather than left to the eye and the colour ramp.
            for (r, c) in ((0, 0), (0, GW - 1), (GH - 1, 0), (GH - 1, GW - 1)):
                v = mat[r, c]
                if not np.isfinite(v):
                    continue
                ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=5.5,
                        color="#ffffff" if abs(np.log2(max(v, 1e-9))) > 1.1
                        else XM.INK)
            if i == 0:
                ax.set_title(col_title, fontsize=9, color=XM.INK, pad=6)
            if j == 0:
                ax.set_ylabel(label, fontsize=8.5, color=XM.INK, labelpad=6)

    sm = ScalarMappable(norm=Normalize(-CLIP, CLIP), cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.030, pad=0.02,
                      ticks=[-CLIP, -1, 0, 1, CLIP])
    cb.ax.set_yticklabels(["0.25x", "0.5x", "fair share", "2x", "4x"], fontsize=7,
                          color=XM.MUTED)
    cb.outline.set_visible(False)
    fig.suptitle("attention over generated tokens, beside where the human boxes are",
                 fontsize=10.5, color=XM.INK, x=0.01, ha="left")
    if note:
        fig.text(0.01, -0.012, note, fontsize=6.4, color=XM.MUTED, ha="left", va="top")
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", facecolor=XM.SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", required=True, help="comma-separated scan directories")
    ap.add_argument("--labels", default=None, help="comma-separated display names")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--field", default="map_gen",
                    help="the pooled map to read (default map_gen: generated tokens)")
    args = ap.parse_args()

    XM = _load("_bc_tables", "sink_location_xmodel_tables.py")
    P = _load("_bc_probe", "sink_location_probe.py")
    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    dirs = [d for d in args.dirs.split(",") if d]
    labels = ([x for x in args.labels.split(",")] if args.labels
              else [Path(d).name.replace("box_", "") for d in dirs])
    if len(labels) != len(dirs):
        raise SystemExit("--labels must have as many entries as --dirs")

    cols = [c for c, _s, _a in XM.COLUMNS]
    attn_rows, cov_rows, wcov_rows, fig_rows, lines = [], [], [], [], []
    starred = {}

    for label, d in zip(labels, dirs):
        meta, arrays = P.read_stage(d, "scan")
        man = {r["key"]: r for r in P.read_manifest(d)}
        shapes = [tuple(m["grid"]) for m in meta]
        if not shapes:
            raise SystemExit(f"{d} holds no scan results")
        modal = max(set(shapes), key=shapes.count)
        n_grids = len(set(shapes))
        starred[label] = n_grids > 1

        amat, an, _g = XM.model_maps(meta, arrays, args.field, lattice=modal)
        cmat, cn, area = coverage_map(meta, man, XM, modal, weighted=False)
        wmat, _wn, _wa = coverage_map(meta, man, XM, modal, weighted=True)

        attn_rows.append((label, an, modal, n_grids, XM.table_from_map(amat)))
        cov_rows.append((label, cn, modal, area, XM.table_from_map(cmat)))
        wcov_rows.append((label, cn, modal, area, XM.table_from_map(wmat)))
        fig_rows.append((f"{label}\n{modal[0]}x{modal[1]}",
                         [("attention (generated tokens)", amat),
                          ("human box coverage", cmat)]))

    def emit(s=""):
        print(s)
        lines.append(s)

    def table(title, rows, extra_hdr, extra_key):
        emit(f"## {title}")
        emit("")
        emit("| model | n | grid | " + extra_hdr + " | "
             + " | ".join(cols) + " |")
        emit("|---" * (4 + len(cols)) + "|")
        for label, n, grid, ex, vals in rows:
            # The star means "this model's grid varies and the modal shape is shown", so
            # it belongs to the model, not to the table: InternVL is fixed at 16x16 and
            # starring it would claim a variation it does not have.
            star = "*" if starred.get(label) else ""
            emit(f"| {label} | {n} | {grid[0]}x{grid[1]}{star} | "
                 + (f"{ex:.3f}" if isinstance(ex, float) else str(ex)) + " | "
                 + " | ".join(f"{vals[c]:.2f}" for c in cols) + " |")
        emit("")

    emit("# Attention vs. human-box coverage — boxed corpus")
    emit("")
    emit("Every cell is an enrichment: the quantity's share of a patch set over that")
    emit("set's share of the patches, so 1.00 is a fair share on both tables. Grids vary")
    emit("per picture and are area-weighted onto each model's own modal lattice.")
    emit("")
    table(f"Attention — {args.field}, all heads over the mass floor",
          attn_rows, "distinct grids", None)
    emit("\\* the grid varies per picture; the modal shape is shown.")
    emit("")
    table("Human-box coverage — fraction of pictures in which the patch is inside the box",
          cov_rows, "mean box area", None)
    emit("The human boxes are the corpus's and do not depend on the model. These four")
    emit("rows differ only through the lattice each model's grids were resampled onto,")
    emit("so they SHOULD be near-identical; the spread across them is the size of this")
    emit("measurement's geometry artefact.")
    emit("")
    table("Human-box coverage, area-weighted per picture (mask / mask.sum())",
          wcov_rows, "mean box area", None)
    emit("The same boxes under the attention map's own convention, where each picture")
    emit("contributes one unit of mass however large its box. It is reported because it")
    emit("is the like-for-like comparison with the attention table; the frequency table")
    emit("above is the one that answers \"how often is this patch in the box\".")

    (out / "box_coverage.md").write_text("\n".join(lines) + "\n")
    note = ("Left: mean pooled attention over the generated tokens. Right: the fraction "
            "of pictures whose human box covers that patch.\nBoth divided by their own "
            "mean, so 1.0 is a fair share on either panel. Corners labelled outright.")
    draw_grid(fig_rows, str(out / "figures" / "attention_vs_boxes_4x2"), note)
    print(f"\nwrote {out}/box_coverage.md and figures/attention_vs_boxes_4x2.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
