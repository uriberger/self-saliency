# Copyright 2026 NVIDIA. Apache-2.0.
"""Grounded regions as boolean masks on the model's patch grid.

Section 3.3 defines the grounded region of a step as the union of the boxes the
phrase-grounding model returned, `u_s = union_i b_i`, with each `b_i` a set of image
patches. This module is that conversion: relative [x0, y0, x1, y1] boxes in, a boolean
(grid_h, grid_w) mask out.

Rasterisation is the load-bearing detail. The metric scores the GRID, not the box
geometry, and the two are not the same area: every surviving box claims at least one
patch row and one patch column, so a scatter of small boxes covers more grid than the
sum of their areas suggests. Anything comparing a box area to a mask coverage has to
know which of the two it is holding.

The centred rectangle used by the `center_rect` ablation (Section 5.3) lives here too,
because it occupies exactly the same slot: it is a region the step is scored against,
just one that cannot depend on the step's text at all. That independence is the point
of the arm.
"""

from __future__ import annotations

import math

import numpy as np


def box_area(box) -> float:
    """Area of a relative [x0, y0, x1, y1] box, clamped at zero."""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def raster_union(boxes, grid_h: int, grid_w: int, max_box_area: float | None = 0.5):
    """Boxes rasterised onto the patch grid, with nothing rejected.

    `max_box_area` drops individual boxes before rasterisation (None or <= 0 disables
    it). The paper's runs use 0.5: Grounding-DINO on a whole sentence will sometimes
    return one box covering the frame, and a region that is the whole image makes
    "inside the region" not a question.

    May return an all-False or an all-True mask. Anything that scores inside-vs-outside
    must go through `union_mask`, which refuses both.
    """
    if max_box_area is not None and float(max_box_area) > 0:
        boxes = [b for b in boxes if box_area(b) <= float(max_box_area)]
    mask = np.zeros((int(grid_h), int(grid_w)), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        r0 = max(0, int(y1 * grid_h))
        r1 = min(grid_h, max(r0 + 1, round(y2 * grid_h)))
        c0 = max(0, int(x1 * grid_w))
        c1 = min(grid_w, max(c0 + 1, round(x2 * grid_w)))
        mask[r0:r1, c0:c1] = True
    return mask


def union_mask(boxes, grid_h: int, grid_w: int,
               max_box_area: float | None = 0.5,
               max_union_area: float | None = None):
    """The grounded region u_s, or None when it is degenerate.

    Two independent filters, each disabled by None or a non-positive value:

      max_box_area    per box, on the relative coordinates, BEFORE rasterisation.
                      Drops individual boxes; the survivors still form a union.
      max_union_area  per step, on the RASTERISED union. Rejects the whole step when
                      the union covers more than this fraction of the grid.

    The per-box cap does not bound the union -- N disjoint boxes each under the cap can
    cover the image between them, and the median union on this corpus already covers
    about 56% of the grid. `max_union_area` is the only filter that closes that. It is
    off in the paper's runs; it is here because turning it on changes the scored
    population, which makes cross-run comparisons of phi's spread invalid unless the
    scored sets are matched.

    None means SKIP the step, never score it zero. See `saliency.score.phi`.
    """
    mask = raster_union(boxes, grid_h, grid_w, max_box_area=max_box_area)
    n_in = int(mask.sum())
    if n_in == 0 or n_in == grid_h * grid_w:
        return None
    if max_union_area is not None and float(max_union_area) > 0:
        if n_in > float(max_union_area) * grid_h * grid_w:
            return None
    return mask


# One rectangle per (grid, fraction). The grids repeat for a whole run and the mask is a
# pure function of them.
_RECT_CACHE: dict = {}


def center_rect_mask(grid_h: int, grid_w: int, frac: float):
    """The `center_rect` ablation's region: a centred rectangle covering ~frac of the grid.

    The area is split equally between the axes (sqrt(frac) on each), so the rectangle
    keeps the frame's aspect and is not secretly a wide or a tall band -- the only thing
    it differs from a grounded union in is WHERE it sits, which is what Section 5.3 is
    testing.

    Rounded to whole patches like `raster_union`, for the same reason: the grid mask is
    what gets scored. On a grid this coarse (10x16 is typical) that rounding moves the
    realised area a few points off `frac`. That is deliberate and it is shared with the
    calibration probe that chose 0.565, so the trained arm scores the rectangle the
    probe measured.

    None if the result covers everything or nothing.
    """
    key = (int(grid_h), int(grid_w), round(float(frac), 6))
    hit = _RECT_CACHE.get(key)
    if hit is not None:
        return hit
    s = math.sqrt(min(1.0, max(0.0, float(frac))))
    rows = min(grid_h, max(1, int(round(grid_h * s))))
    cols = min(grid_w, max(1, int(round(grid_w * s))))
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    r0, c0 = (grid_h - rows) // 2, (grid_w - cols) // 2
    mask[r0:r0 + rows, c0:c0 + cols] = True
    n_in = int(mask.sum())
    if n_in == 0 or n_in == grid_h * grid_w:
        return None
    _RECT_CACHE[key] = mask
    return mask


def ring_fraction(mask) -> float:
    """Share of a mask's patches lying on the grid's one-patch border.

    Reported alongside every mask the reward scores, because Section 5 says the border
    ring is where the attention already is. A mask that drifts onto the ring scores well
    for a reason that has nothing to do with grounding, so this is the number that says
    whether an arm is still measuring what it was named for.
    """
    n_in = int(np.asarray(mask, dtype=bool).sum())
    if n_in == 0:
        return float("nan")
    m = np.asarray(mask, dtype=bool)
    ring = np.zeros_like(m)
    ring[0, :] = ring[-1, :] = True
    ring[:, 0] = ring[:, -1] = True
    return float(np.logical_and(m, ring).sum()) / n_in


def centroid_eccentricity(mask) -> float:
    """How far the region's centroid sits from the grid centre. 0 = centre, 1 = a corner.

    A DIAGNOSTIC, not a score. Section 5.3 asks whether the gain is just a centre bias,
    and the way that would show up inside a real run is this rising over training,
    together with its correlation with phi -- the policy naming central objects because a
    centre-heavy map scores well for free. Reported per step by the trained-model probe
    so the question is answerable from a run rather than only from the ablation.
    """
    m = np.asarray(mask, dtype=bool)
    gh, gw = m.shape
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return float("nan")
    cy = (ys.mean() + 0.5) / gh - 0.5
    cx = (xs.mean() + 0.5) / gw - 0.5
    return float(np.hypot(cy, cx) / np.hypot(0.5, 0.5))
