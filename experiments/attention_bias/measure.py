#!/usr/bin/env python
"""Where inside the picture does attention pile up -- on the BORDER, or on the BACKGROUND?

`docs/sink-location-by-image-type.md` is the design. This module is the measurement:
geometry on the patch grid, the reducer that turns raw attention columns into endpoints,
the attention implementation that collects them, the image transforms that separate
position from content, and the two model-level interventions (patch permutation, vision
tap) that the mechanism stage needs.

WHAT IS BEING MEASURED, AND WHY IT IS NOT THE SAME AS `sink_shift.py`

`sink_shift` reads ROWS: one query, where did its weight go. That is the right view for an
edit, because an edit rewrites a row. This module reads COLUMNS: one image patch, how much
weight did it receive, averaged over a fixed set of queries. A sink is a column property --
"everyone looks here" -- and the column view is what makes the three definitions separable:

    S1  peak      argmax over image patches of the column mean. This is what
                  peak_location_probe.py measured and what "82% on the outer ring" is.
                  It is the maximum of a distribution, which exists whether or not
                  anything is a sink.
    S2  mass      the ring's share of the image's attention, over the ring's share of the
                  patches. 1.0 is no effect. Robust to ties and to quantisation.
    S3  sink      large AND query-invariant: a column whose mean is many times uniform and
                  whose coefficient of variation ACROSS QUERIES is small. This is the
                  literature's definition, and it is the one that can come back empty
                  while S1 and S2 are both strong.

Every number is reported against the geometry it came from. The ring's own area fraction
is (2gh + 2gw - 4) / (gh*gw), which is 0.234 on a 16x16 grid, 0.300 on 10x16 and 0.500 on
6x8 -- a spread larger than most effects anyone reports. Nothing here returns a raw
percentage; `cell_stats` returns shares, and the report divides by `ring_area_frac`.

THE BUDGET COMES FIRST. `SPANS` splits a prompt into first-token / pre-image /
vision-start / image / vision-end / post-image, and every (layer, head) records what
fraction of its row landed in each. At layer 22 heads 28/31 the whole picture receives
0.4%-1.4% of a row (docs/inference-intervention.md), so if that holds broadly then the
image's "sink" is a ripple on a 1% budget and the honest claim is "within the picture, the
border is favoured". The budget is what licenses the wording.

    from experiments.attention_bias import measure as SL
    scan = SL.install(model)                     # registers the attention implementation
    model(**inputs)                              # one prefill, batch 1
    per_image = scan.result()                    # raw columns -> SL.reduce_image(...)
    scan.uninstall()

WHICH MODEL. Everything below is family-agnostic and reaches the model through
`vlm_family.py`, which holds the image token, the grid, the decoder's attention class,
the module that emits the LLM-facing rows and -- the one that will bite you -- the VIEW
BOX, the part of the picture the patch grid actually covers. `install(model)` picks the
family off `config.model_type`; the module-level ids below are Qwen3-VL's and are kept
only so a caller that predates the seam still resolves.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from selfsal.saliency import maps as CAP
from selfsal.models import families as VF

#: The centred-rectangle fraction. Shared with the `center_rect` training arm, and with
#: the "donut" image transform below, which blanks exactly that rectangle -- so "what the
#: reward's rectangle covered" and "what the transform removed" are one number.
CENTER_RECT_FRAC = 0.565

QWEN3VL = VF.Qwen3VL()
IMAGE_TOKEN_ID = CAP.IMAGE_TOKEN_ID       # <|image_pad|>
VISION_START_ID = 151652
VISION_END_ID = 151653
IMPL_NAME = "sink_scan"

# Query sets, all accumulated from the SAME forward.
#
#   text        every prompt position after the last image token -- the question and the
#               assistant header. The primary, and the only one a prefill-only run has.
#   generated   the tokens the model WROTE. Present only when the forward is teacher-forced
#               over prompt ++ completion, which is exactly how the training reward read
#               its maps (grpo_trainer_qwen3.py, overlap_layer=22, overlap_heads=(28, 31)),
#               so this is not an approximation of the reward's view -- it is the reward's
#               view.
#   image       the picture attending to itself, kept because a sink that exists only in
#               the image->image block is a vision-tower artefact wearing a language
#               model's clothes. Needs the causal correction; the other two do not.
#
#   observe     a SUBSET of `generated`: only the tokens of the sentences the reward's own
#               FLAN-T5 classifier labels "observe". Everything the overlap reward scores
#               is read off those rows and no others, so `generated` -- which averages
#               them together with the plan and deduce sentences and the answer -- is a
#               wider set than the reward ever looked at. It cannot be derived from the
#               other accumulators: which rows belong to it is a function of the text the
#               model wrote, not of any position the scan can compute.
#
# "all tokens" is not a fifth accumulator: column sums are additive, so text + generated
# with the row counts added is exactly the union, and deriving it costs nothing.
Q_SETS = ("text", "generated", "observe", "image")
PRIMARY_Q = "text"

SPANS = ("first", "pre_image", "vision_start", "image", "vision_end", "post_image")

#: The part of a picture the patch grid covers, normalised, when the processor shows the
#: encoder all of it. See `vlm_family.Family.view_box`.
FULL_VIEW = (0.0, 0.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# geometry on the patch grid -- pure, no torch, no model
# ---------------------------------------------------------------------------
def depth_map(gh, gw):
    """Chebyshev distance from each patch to the nearest edge. -> [gh, gw] int.

    Depth 0 is the one-patch border, depth 1 the ring inside it, and so on. Every radial
    statistic in this module is a groupby on this array, which is what makes the result
    independent of anyone's choice of "ring width".
    """
    r = np.minimum(np.arange(gh)[:, None], gh - 1 - np.arange(gh)[:, None])
    c = np.minimum(np.arange(gw)[None, :], gw - 1 - np.arange(gw)[None, :])
    return np.minimum(np.broadcast_to(r, (gh, gw)), np.broadcast_to(c, (gh, gw)))


def ring_set(gh, gw, depth=0):
    """The patches at exactly `depth` from the border. depth=0 is `sink_shift.frame_set`."""
    return depth_map(gh, gw) == int(depth)


def ring_area_frac(gh, gw):
    """(2gh + 2gw - 4) / (gh*gw) -- the chance level for a uniformly placed peak."""
    if gh <= 0 or gw <= 0:
        return float("nan")
    if gh == 1 or gw == 1:
        return 1.0
    return (2 * gh + 2 * gw - 4) / float(gh * gw)


def edge_sets(gh, gw):
    """The border, cut into the four sides and the corners.

    The corners belong to two sides each and are counted in both, plus separately. This is
    deliberate: `top` means "the top row", which is the set an asymmetry claim is about,
    and subtracting the corners from it would make the four sides not tile the ring.
    """
    z = lambda: np.zeros((gh, gw), dtype=bool)          # noqa: E731
    top, bottom, left, right, corner = z(), z(), z(), z(), z()
    top[0, :] = bottom[-1, :] = True
    left[:, 0] = right[:, -1] = True
    corner[0, 0] = corner[0, -1] = corner[-1, 0] = corner[-1, -1] = True
    return {"top": top, "bottom": bottom, "left": left, "right": right, "corner": corner}


def random_contiguous_set(gh, gw, n_patches, seed=0):
    """A control block of `n_patches` contiguous patches, placed away from the border.

    The metric's negative control: a set of the same SIZE as the ring but in no special
    place must come back at enrichment 1.0. Without it, an enrichment of 1.6 has nothing
    to be 1.6 against, and a bug in the normalisation looks exactly like a result.
    """
    n_patches = int(min(max(1, n_patches), gh * gw))
    rng = np.random.default_rng(seed)
    rows = int(min(gh, max(1, round(math.sqrt(n_patches * gh / max(1, gw))))))
    cols = int(min(gw, max(1, math.ceil(n_patches / rows))))
    r0 = int(rng.integers(0, max(1, gh - rows + 1)))
    c0 = int(rng.integers(0, max(1, gw - cols + 1)))
    m = np.zeros((gh, gw), dtype=bool)
    m[r0:r0 + rows, c0:c0 + cols] = True
    flat = np.flatnonzero(m.reshape(-1))
    if flat.size > n_patches:                            # trim to the exact size
        m = np.zeros(gh * gw, dtype=bool)
        m[flat[:n_patches]] = True
        m = m.reshape(gh, gw)
    return m


def named_sets(gh, gw, seed=0):
    """Every patch set the reducer scores, as flat boolean arrays over gh*gw."""
    sets = {"ring": ring_set(gh, gw, 0)}
    for name, m in edge_sets(gh, gw).items():
        sets[name] = m
    d = depth_map(gh, gw)
    for k in (1, 2):
        sets[f"depth{k}"] = d == k
    sets["interior"] = d >= 1
    sets["deep"] = d >= 3
    # the two negative controls: same size as the ring, no special place
    sets["ctrl_block"] = random_contiguous_set(gh, gw, int(ring_set(gh, gw).sum()), seed)
    sets["ring2_ctrl"] = ring_set(gh, gw, 1)
    return {k: v.reshape(-1) for k, v in sets.items()}


# ---------------------------------------------------------------------------
# the reducer -- raw columns to endpoints. Pure numpy, CPU-testable.
# ---------------------------------------------------------------------------
#: One row of the stats array, per (layer, head). Order is the file format; append only.
STAT_NAMES = (
    "image_mass",        # image's share of the whole attention row
    "peak_share",        # the top image patch's share of the IMAGE's mass
    "peak_abs",          # ... and of the whole row. S3 needs the absolute magnitude
    "peak_uniform_x",    # peak_abs / (1/kv_len): how many times uniform the top column is
    "peak_cv",           # its coefficient of variation ACROSS QUERIES. S3's invariance leg
    "peak_in_ring",      # 1.0 if the argmax patch is on the border. S1
    "peak_row_frac",     # normalised row/col of the argmax, for the asymmetry story
    "peak_col_frac",
    "ring_share",        # S2's numerator: the border's share of the image's mass
    "top_share", "bottom_share", "left_share", "right_share", "corner_share",
    # The FIRST and LAST image tokens, on their own. H1's sharpest readout: the first
    # patch is top-left in raster order and sits immediately after <|vision_start|>, so a
    # sequence-position sink lands on it and a 2D border effect has no reason to prefer it
    # over the other three corners.
    "first_patch_share", "last_patch_share",
    "depth1_share", "depth2_share", "deep_share",
    "ctrl_block_share",  # the negative control, same size as the ring
    "ring2_ctrl_share",
    "entropy_norm",      # normalised entropy of the image distribution; 1.0 = flat
    "knorm_ring",        # mean ||k|| over border patches, and over the interior. M2
    "knorm_interior",
    "align_ring",        # mean logit / (scaling * ||k||): the query-alignment leg of M2
    "align_interior",
    # The four corners SEPARATELY. `corner_share` averages them, which is the right
    # summary for "is the geometry enriched" and the wrong one for "which end of the
    # raster order is". Top-left and bottom-right coincide with `first_patch_share` and
    # `last_patch_share` on a single-tile grid and are carried again here so a table can
    # name all four without the reader having to know that.
    "corner_tl_share", "corner_tr_share", "corner_bl_share", "corner_br_share",
)
STAT_INDEX = {n: i for i, n in enumerate(STAT_NAMES)}


def reduce_cells(col_sum, col_sq, n_rows, row_total, gh, gw, kv_len,
                 knorm=None, logit_sum=None, scaling=1.0, seed=0, col_null=None):
    """[L, H, N] raw column sums -> ([L, H, len(STAT_NAMES)] endpoints, [L, H] peak index).

    The peak index is returned rather than recovered from `peak_row_frac`, because the
    stats are stored as float16 and a rounded normalised coordinate lands on the wrong
    patch near a boundary. The arms compare peak indices for a living, so they get the
    exact one.

    col_sum    [L, H, N]  sum over the query set of the attention each image patch got
    col_sq     [L, H, N]  sum of the squares, or None -- only `peak_cv` needs it
    n_rows     int        how many queries were summed
    row_total  [L, H]     sum over the query set of the WHOLE row (image and text), so
                          `image_mass` is a share of the row and not of the picture
    knorm      [L, H, N]  ||k|| per image patch, or None
    logit_sum  [L, H, N]  sum over the query set of the pre-softmax logit, or None
    col_null   [N]        the column mean a POSITION-BLIND model would produce, or None

    `col_null` exists for one reason and it is a trap worth naming. Attention is causal
    and the picture is raster-ordered, so in the IMAGE->IMAGE query set patch 0 is visible
    to all N queries and the last patch is visible to one. That alone manufactures a
    top-row gradient of exactly the shape this experiment is testing for.

    A per-column QUERY COUNT does not fix it, which is worth stating because it is the
    obvious fix and it is wrong: an early column is also competing against fewer rivals in
    every row it appears in, so its mean is genuinely higher under a model with no
    position preference at all. What removes the artefact is dividing by that model's own
    prediction -- `SinkScan.column_null` computes it in closed form -- and only then
    renormalising. The text query set sits entirely after the picture, sees every column
    with the same competition, and needs no correction, which is one more reason it is the
    primary readout.

    Magnitudes (`peak_abs`, `peak_uniform_x`, `peak_cv`) are read off the RAW mean, since
    "this column receives 30x uniform" is a claim about the attention that actually
    happened. The distribution -- every *_share, the peak's location, the entropy -- is
    read off the corrected one, since that is a claim about position.

    A cell whose image mass is exactly zero -- possible in a head that never looks at the
    picture -- returns NaN throughout rather than a division artefact or a silent 0.
    """
    col_sum = np.asarray(col_sum, dtype=np.float64)
    L, H, N = col_sum.shape
    n_rows = max(1, int(n_rows))
    out = np.full((L, H, len(STAT_NAMES)), np.nan, dtype=np.float64)
    if N != gh * gw:
        raise ValueError(f"{N} image columns against a {gh}x{gw} grid")

    p, raw, tot, img_abs, live, mass = _cell_distribution(col_sum, row_total, n_rows,
                                                          col_null)
    keep = lambda x: np.where(live, x, np.nan)               # noqa: E731
    out[..., STAT_INDEX["image_mass"]] = mass

    peak = np.argmax(np.where(np.isfinite(p), p, -np.inf), axis=-1)       # [L,H]
    li, hi = np.meshgrid(np.arange(L), np.arange(H), indexing="ij")
    out[..., STAT_INDEX["peak_share"]] = keep(p[li, hi, peak])
    peak_abs = raw[li, hi, peak] / np.where(tot > 0, tot, np.nan)   # share of the ROW
    out[..., STAT_INDEX["peak_abs"]] = keep(peak_abs)
    out[..., STAT_INDEX["peak_uniform_x"]] = keep(peak_abs * max(1, int(kv_len)))

    if col_sq is not None:
        sq = np.asarray(col_sq, dtype=np.float64)[li, hi, peak] / float(n_rows)
        mu = raw[li, hi, peak]
        var = np.maximum(sq - mu ** 2, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[..., STAT_INDEX["peak_cv"]] = keep(np.where(mu > 0, np.sqrt(var) / mu,
                                                            np.nan))

    sets = named_sets(gh, gw, seed)
    out[..., STAT_INDEX["peak_in_ring"]] = keep(sets["ring"][peak].astype(float))
    out[..., STAT_INDEX["peak_row_frac"]] = keep(((peak // gw) + 0.5) / gh)
    out[..., STAT_INDEX["peak_col_frac"]] = keep(((peak % gw) + 0.5) / gw)

    for key, stat in (("ring", "ring_share"), ("top", "top_share"),
                      ("bottom", "bottom_share"), ("left", "left_share"),
                      ("right", "right_share"), ("corner", "corner_share"),
                      ("depth1", "depth1_share"), ("depth2", "depth2_share"),
                      ("deep", "deep_share"), ("ctrl_block", "ctrl_block_share"),
                      ("ring2_ctrl", "ring2_ctrl_share")):
        out[..., STAT_INDEX[stat]] = keep(np.nansum(p[..., sets[key]], axis=-1))
    out[..., STAT_INDEX["first_patch_share"]] = keep(p[..., 0])
    out[..., STAT_INDEX["last_patch_share"]] = keep(p[..., -1])
    for stat, idx in (("corner_tl_share", 0), ("corner_tr_share", gw - 1),
                      ("corner_bl_share", (gh - 1) * gw),
                      ("corner_br_share", gh * gw - 1)):
        out[..., STAT_INDEX[stat]] = keep(p[..., idx])

    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(np.isfinite(p) & (p > 0), p, 1.0)
        ent = -np.nansum(np.where(p > 0, p * np.log(safe), 0.0), axis=-1)
    out[..., STAT_INDEX["entropy_norm"]] = keep(ent / math.log(max(2, N)))

    if knorm is not None:
        kn = np.asarray(knorm, dtype=np.float64)
        out[..., STAT_INDEX["knorm_ring"]] = kn[..., sets["ring"]].mean(-1)
        out[..., STAT_INDEX["knorm_interior"]] = kn[..., sets["interior"]].mean(-1)
        if logit_sum is not None:
            lg = np.asarray(logit_sum, dtype=np.float64) / n_rows
            # logit_i = scaling * <qbar, k_i> = scaling * ||k_i|| * (||qbar|| cos t_i).
            # Dividing out the key norm leaves the alignment leg, so a ring advantage can
            # be split into "the border's keys are bigger" and "the border's keys point
            # where the queries point". Those have different fixes.
            with np.errstate(divide="ignore", invalid="ignore"):
                align = np.where(kn > 0, lg / (float(scaling) * kn), np.nan)
            out[..., STAT_INDEX["align_ring"]] = np.nanmean(align[..., sets["ring"]], -1)
            out[..., STAT_INDEX["align_interior"]] = np.nanmean(
                align[..., sets["interior"]], -1)
    return out, np.where(live, peak, -1).astype(np.int32)


def _cell_distribution(col_sum, row_total, n_rows, col_null=None):
    """One (layer, head) cell's patch distribution, and how much of its row the picture got.

    Factored out of `reduce_cells` so that `pooled_patch_map` -- the per-patch
    decomposition the figures draw -- is computed from the same arithmetic as the tables
    rather than from a second copy of it. A drifted copy would produce a heatmap that
    disagrees with the numbers printed beside it, in the third decimal, silently.

    -> (p [L,H,N] distribution, raw weight, row total, image weight, live mask, image mass)
    """
    col_sum = np.asarray(col_sum, dtype=np.float64)
    n_rows = max(1, int(n_rows))
    raw = col_sum / float(n_rows)                            # [L,H,N] absolute weight
    tot = np.asarray(row_total, dtype=np.float64) / n_rows
    img_abs = raw.sum(-1)                                    # [L,H]
    live = img_abs > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        mass = np.where(tot > 0, img_abs / tot, np.nan)
        # the position-corrected distribution: what the picture's attention looks like
        # once the shape a position-blind model would produce anyway is divided out
        corr = raw if col_null is None else raw / np.maximum(
            np.asarray(col_null, dtype=np.float64), 1e-30)
        cimg = corr.sum(-1)
        p = np.where(live[..., None], corr / np.where(cimg > 0, cimg, 1.0)[..., None],
                     np.nan)
    return p, raw, tot, img_abs, live, mass


def pooled_patch_map(col_sum, row_total, n_rows, min_mass=0.002, col_null=None,
                     cells=None):
    """The per-patch decomposition of a head set's table. -> [N], sums to ~1.

    Every cross-model table is "the mean, over the (layer, head) cells that clear the
    image-mass floor, of that cell's share of the picture's attention on some patch set".
    This is that same average taken per PATCH instead of per set, so summing it over any
    set reproduces the table's numerator exactly. It is what the heatmaps draw, and the
    reason they cannot drift away from the numbers printed beside them.

    `cells` is None for the ALL-HEAD map, or an explicit list of (layer, head) -- the two
    the overlap reward trains on. The two treatments of the floor differ on purpose, and
    it is the same split `sink_location_probe._enrich` makes:

      all-head    floored. A head that puts no weight on the picture still has a patch
                  distribution, and it is noise wearing a statistic's name.
      named       never floored. These cells were named by the reward, not selected for
                  being large, so dropping them on the pictures where they happen to look
                  away would quietly turn "where these two heads look" into "where they
                  look on the pictures they bother to look at" -- a different claim, and a
                  flattering one.

    A cell outside the array is ignored rather than an error: L22 exists in every family
    here, but a 20-layer model would otherwise make this raise from inside a scan.
    """
    p, _raw, _tot, _img, _live, mass = _cell_distribution(col_sum, row_total, n_rows,
                                                          col_null)
    if cells is None:
        ok = np.isfinite(mass) & (mass >= float(min_mass))
    else:
        ok = np.zeros(mass.shape, dtype=bool)
        for layer, head in cells:
            if 0 <= layer < ok.shape[0] and 0 <= head < ok.shape[1]:
                ok[layer, head] = True
    if not ok.any():
        return np.full(p.shape[-1], np.nan)
    sel = np.where(ok[..., None], p, np.nan).reshape(-1, p.shape[-1])
    if not np.isfinite(sel).any():
        return np.full(p.shape[-1], np.nan)
    return np.nanmean(sel, axis=0)


def content_stats(image, gh, gw):
    """Per-patch content, on the model's own grid: is this patch worth looking at?

    Returns {name: [gh*gw] float}. These are the covariates that decide whether "outer
    ring" survives once "background" is controlled for, which is the whole question:

      pix_var    variance of the greyscale pixels in the patch. Flat sky and blank paper
                 are near zero; texture and text are not
      edge       mean absolute gradient -- ink, edges, structure
      blank      1.0 when the patch is within a hair of the image's modal colour AND flat.
                 A chart's white middle and a photo's white sky both score 1
      sat        mean saturation, which separates "white paper" from "flat blue sky"
    """
    from PIL import Image

    a = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    H, W = a.shape[:2]
    grey = a.mean(-1)
    gy, gx = np.gradient(grey)
    grad = np.abs(gy) + np.abs(gx)
    sat = a.max(-1) - a.min(-1)

    def pool(x, how):
        # The grid is the processor's, not a divisor of the pixel size in general, so the
        # patch boundaries are taken by resizing the per-pixel field to (gw, gh). BOX
        # resampling of a float field is exactly a box mean over each patch.
        im = Image.fromarray(x.astype(np.float32), mode="F")
        if how == "max":
            im = im.resize((gw * 2, gh * 2), Image.BILINEAR)
        return np.asarray(im.resize((gw, gh), Image.BOX), dtype=np.float64).reshape(-1)

    mean_g = pool(grey, "mean")
    mean_g2 = pool(grey ** 2, "mean")
    var = np.maximum(mean_g2 - mean_g ** 2, 0.0)
    edge = pool(grad, "mean")
    satp = pool(sat, "mean")
    modal = float(np.median(np.concatenate([grey[0, :], grey[-1, :],
                                            grey[:, 0], grey[:, -1]])))
    blank = ((var < 1e-3) & (np.abs(mean_g - modal) < 0.06)).astype(float)
    return {"pix_var": var, "edge": edge, "sat": satp, "blank": blank,
            "mean_grey": mean_g, "modal_border_grey": np.full(gh * gw, modal)}


# ---------------------------------------------------------------------------
# locating the picture in the prompt
# ---------------------------------------------------------------------------
def locate_image_runs(input_ids, inputs, family=None):
    """-> ([run positions per picture], [(t, gh, gw) per TILE, in column order]).

    Deliberately a second implementation of what `SinkShift._locate_images` does for its
    own purposes, in the same way `sink_shift.rect_set` is a second implementation of the
    reward's rectangle -- and `test_sink_location_cpu.py` asserts the two agree on the
    same input. A drifted locator would put every column statistic on the wrong patch and
    nothing downstream would notice.

    The grids come back per TILE rather than per picture. In every family but InternVL
    with tiling switched on those are the same thing; where they are not, a picture is up
    to twelve 448px tiles plus a thumbnail, each its own 16x16 grid, and scoring them as
    one grid would call a tile boundary an interior edge and produce a confident,
    meaningless number.
    """
    fam = family or QWEN3VL
    ids = input_ids[0]
    is_img = ids == fam.image_token_id
    if not bool(is_img.any()):
        return [], []
    pos = torch.nonzero(is_img, as_tuple=True)[0]
    brk = torch.nonzero(pos[1:] - pos[:-1] != 1, as_tuple=True)[0]
    starts = [0] + (brk + 1).tolist()
    ends = (brk + 1).tolist() + [pos.numel()]
    runs = [pos[a:b] for a, b in zip(starts, ends)]
    return runs, fam.grids_for(runs, inputs)


def span_index(input_ids, runs, family=None):
    """{span name: LongTensor of key positions}. A partition of the prompt.

    The picture's share of a row means nothing without the rest of the row beside it, and
    "the rest" is not one thing: the first token, the system prompt, the two vision
    delimiters and the question are four different candidate sinks with four different
    stories. `first` is a subset of `pre_image` and is reported separately.

    LLaVA-1.5 has no delimiter tokens at all -- `<image>` expands in place with nothing
    around it -- so its `vision_start`/`vision_end` spans come back empty. That is a fact
    about the model and the budget table prints it as 0.0000 rather than hiding it.
    """
    fam = family or QWEN3VL
    ids = input_ids[0]
    n = int(ids.numel())
    img = torch.cat(runs) if runs else torch.zeros(0, dtype=torch.long, device=ids.device)
    lo = int(img.min()) if img.numel() else n
    hi = int(img.max()) + 1 if img.numel() else n
    dev = ids.device
    ar = torch.arange(n, device=dev)
    vs = _ids_at(ids, fam.vision_start_ids)
    ve = _ids_at(ids, fam.vision_end_ids)
    is_img = torch.zeros(n, dtype=torch.bool, device=dev)
    is_img[img] = True
    special = is_img.clone()
    special[vs] = True
    special[ve] = True
    return {
        "first": ar[:1],
        "pre_image": ar[(ar < lo) & ~special],
        "vision_start": vs,
        "image": img,
        "vision_end": ve,
        "post_image": ar[(ar >= hi) & ~special],
    }


def _ids_at(ids, wanted):
    """Positions of any of `wanted` token ids. An empty `wanted` gives an empty span."""
    if not wanted:
        return torch.zeros(0, dtype=torch.long, device=ids.device)
    hit = torch.zeros_like(ids, dtype=torch.bool)
    for i in wanted:
        hit |= ids == int(i)
    return torch.nonzero(hit, as_tuple=True)[0]


# ---------------------------------------------------------------------------
# the collector
# ---------------------------------------------------------------------------
class SinkScan:
    """Accumulates the column view of one prompt, at every layer and every head.

    Installed the way `sink_shift.SinkShift` is -- a registered attention implementation
    plus a forward pre-hook that locates the picture -- because that is the only place the
    real post-RoPE keys and the real softmax both exist. It EDITS NOTHING; the identity is
    checked against stock SDPA by the selftest and by the CPU tests.

    Memory: the accumulators are [n_layers, n_heads, n_image_tokens], which at 36 x 32 x
    256 is 1.2 MB per array. The per-layer [heads, q, kv] weights never outlive their own
    forward.
    """

    def __init__(self, model, want_key_stats=True, q_sets=Q_SETS, family=None):
        self.model = model
        self.family = family or QWEN3VL
        self.want_key_stats = bool(want_key_stats)
        self.q_sets = tuple(q_sets)
        # `paused` lets the caller generate at full speed through the fused kernel and then
        # measure one teacher-forced forward, instead of paying the explicit softmax on
        # every layer of every decode step.
        self.paused = False
        # A teacher-forced forward carries prompt ++ completion, so the prompt length the
        # pre-hook would infer is the WHOLE thing and `generated` would come back empty.
        # The caller sets this to the real prompt length before that forward.
        self.prompt_len_override = None
        # Absolute query positions of the observe-step tokens, in the same space as
        # `prompt_len` and `img_cols`. Set by the caller alongside `prompt_len_override`
        # and cleared with it: which rows those are cannot be inferred from any position,
        # only from the sentences the model wrote and how the classifier labelled them.
        self.observe_rows = None
        self.img_cols = None
        self.grids = []
        self.runs = []
        self.spans = {}
        self.prompt_len = 0
        self.kv_len = 0
        self.scaling = 1.0
        self._acc = {}
        self._handles = []
        self._prev_impl = None
        self._text_cfg = None
        self.reset()

    # -- accumulators -----------------------------------------------------
    def reset(self):
        self._acc = {q: {} for q in self.q_sets}
        self._span_acc = {}
        # Counted PER LAYER, not per query set: every layer of one forward sees the same
        # queries, so a single counter incremented inside the per-layer reducer would come
        # out 36x too large and every mean would be 36x too small.
        self._rows = {q: {} for q in self.q_sets}
        self._key = {}
        self._logit = {}
        self.n_forwards = 0

    def _slot(self, store, layer, shape, device):
        got = store.get(layer)
        if got is None:
            got = torch.zeros(shape, dtype=torch.float32, device=device)
            store[layer] = got
        return got

    def rows_for(self, name, q_start, q_len, device):
        """Query positions of one set, as offsets into this forward's query block."""
        pos = torch.arange(q_start, q_start + q_len, device=device)
        if name == "text":
            lo = int(self.img_cols.max()) + 1
            sel = (pos >= lo) & (pos < self.prompt_len)
        elif name == "generated":
            sel = pos >= self.prompt_len
        elif name == "observe":
            # A named set of absolute positions, never a range: observe sentences are
            # scattered through the chain, and a picture whose completion has none is an
            # empty set rather than a fallback to the whole of `generated` -- which would
            # quietly answer a different question on exactly the rows that failed.
            if self.observe_rows is None or len(self.observe_rows) == 0:
                return pos.new_empty(0, dtype=torch.long)
            want = torch.as_tensor(self.observe_rows, device=device, dtype=pos.dtype)
            sel = torch.isin(pos, want)
        elif name == "image":
            sel = torch.isin(pos, self.img_cols)
        else:
            sel = torch.ones_like(pos, dtype=torch.bool)
        return torch.nonzero(sel, as_tuple=True)[0]

    def column_null(self, q_set):
        """The column mean a POSITION-BLIND model would produce anyway. -> [N] or None.

        "Position-blind" means every query spreads its row uniformly over the keys it is
        allowed to see. Column j then gets 1/(p_i + 1) from each query at absolute
        position p_i >= p_j, and nothing from the rest, so

            null[j] = (1 / n_queries) * sum over allowed i of 1 / (p_i + 1)

        For `text` that sum does not depend on j at all -- those queries sit after the
        whole picture and every column faces the same competition -- so the correction is
        a constant, the renormalisation eats it, and this returns None.

        For `image` it is a steep gradient, and it is NOT the same as the query count.
        The count says an early column is seen more often; this says it is also competing
        against fewer rivals each time. Dividing by the count leaves most of the artefact
        standing, which `test_sink_location_cpu.py` measures rather than asserts.
        """
        if q_set != "image" or self.img_cols is None:
            return None
        pos = self.img_cols.detach().cpu().numpy().astype(np.float64)
        allowed = pos[:, None] >= pos[None, :]        # [query, column]
        share = 1.0 / (pos + 1.0)                     # a causal row's uniform weight
        return (allowed * share[:, None]).sum(0) / max(1, pos.size)

    # -- results ----------------------------------------------------------
    def result(self):
        """-> dict of numpy arrays, one entry per query set, plus the span budget.

        Layers are stacked in index order and a layer that never fired is refused rather
        than zero-filled: a missing layer means the attention implementation was not
        installed on it, and a zero row would read as "this layer ignores the picture".

        "Every layer" is not the same as "0 to n-1". A Mamba-Transformer hybrid has
        attention in only some of its layers -- Nemotron-Nano-Omni in 6 of 52, at the
        positions its `hybrid_override_pattern` marks -- and there is nothing for this
        scan to see in the rest, because they have no attention matrix. So the family
        says which layers SHOULD have fired and the check is against that; the
        contiguous-from-zero rule is what a dense model's family returns.
        """
        if not self._acc[self.q_sets[0]]:
            return None
        layers = sorted(self._acc[self.q_sets[0]])
        want = self.family.attention_layers(self.model)
        if want is None:
            want = list(range(len(layers)))
        if layers != list(want):
            raise RuntimeError(
                f"the scan saw layers {layers} but this family's attention layers are "
                f"{list(want)}: the attention implementation was not installed on every "
                "one of them, and a missing layer is not a layer that ignores the picture")
        out = {"layers": layers, "kv_len": self.kv_len, "scaling": self.scaling,
               "grids": self.grids, "n_image_tokens": int(self.img_cols.numel())}
        for q in self.q_sets:
            if not self._acc[q]:
                out[q] = None
                continue
            out[q] = {
                "col_sum": torch.stack([self._acc[q][l]["sum"] for l in layers]).cpu().numpy(),
                "col_sq": (torch.stack([self._acc[q][l]["sq"] for l in layers]).cpu().numpy()
                           if "sq" in self._acc[q][layers[0]] else None),
                "row_total": torch.stack([self._acc[q][l]["tot"] for l in layers]).cpu().numpy(),
                "n_rows": int(self._rows[q][layers[0]]),
                "col_null": self.column_null(q),
            }
        out["spans"] = {
            name: torch.stack([self._span_acc[l][i] for l in layers]).cpu().numpy()
            for i, name in enumerate(SPANS)
        } if self._span_acc else None
        out["knorm"] = (torch.stack([self._key[l] for l in layers]).cpu().numpy()
                        if self._key else None)
        out["logit_sum"] = (torch.stack([self._logit[l] for l in layers]).cpu().numpy()
                            if self._logit else None)
        return out

    # -- per-prompt layout ------------------------------------------------
    def _locate(self, input_ids, inputs):
        runs, grids = locate_image_runs(input_ids, inputs, self.family)
        if not runs:
            self.img_cols = None
            return False
        self.runs, self.grids = runs, grids
        self.img_cols = torch.cat(runs).to(input_ids.device)
        self.spans = span_index(input_ids, runs, self.family)
        self.prompt_len = int(self.prompt_len_override or input_ids.shape[1])
        self.reset()
        return True

    def _pre_hook(self, module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is None or ids.dim() != 2:
            return None
        if ids.shape[0] != 1:
            raise RuntimeError(
                f"batch of {ids.shape[0]}: sink_location locates the picture per prompt "
                "and is only correct at batch size 1")
        if bool((ids == self.family.image_token_id).any()):
            self._locate(ids, kwargs)
        return None

    # -- install ----------------------------------------------------------
    def install(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS.register(IMPL_NAME, _make_scan_attention(self))
        cfg = getattr(self.model.config, "text_config", None) or self.model.config
        self._text_cfg = cfg
        self._prev_impl = cfg._attn_implementation
        cfg._attn_implementation = IMPL_NAME
        for m in self.model.modules():
            if type(m).__name__ in self.family.attn_classes and hasattr(m, "layer_idx"):
                m.config._attn_implementation = IMPL_NAME
        self._handles.append(self.model.register_forward_pre_hook(self._pre_hook,
                                                                  with_kwargs=True))
        return self

    def uninstall(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._text_cfg is not None and self._prev_impl is not None:
            self._text_cfg._attn_implementation = self._prev_impl
            for m in self.model.modules():
                if type(m).__name__ in self.family.attn_classes:
                    m.config._attn_implementation = self._prev_impl
        self._prev_impl = None


def install(model, family=None, **kwargs):
    return SinkScan(model, family=family or VF.family_for(model), **kwargs).install()


def _make_scan_attention(state: SinkScan):
    """Stock attention, with the columns read off on the way past. Edits nothing."""

    def sink_scan_attention_forward(module, query, key, value, attention_mask,
                                    dropout=0.0, scaling=None, is_causal=None, **kwargs):
        if state.img_cols is None or state.paused:
            return CAP.sdpa(module, query, key, value, attention_mask, dropout, scaling,
                            is_causal, **kwargs)
        layer_idx = int(getattr(module, "layer_idx", -1))
        n_rep = getattr(module, "num_key_value_groups", 1)
        k, v = CAP.repeat_kv(key, n_rep), CAP.repeat_kv(value, n_rep)
        q_len, kv_len = query.shape[2], k.shape[2]
        q_start = kv_len - q_len
        if scaling is None:
            scaling = module.head_dim ** -0.5
        state.scaling = float(scaling)
        state.kv_len = int(kv_len)

        logits = torch.matmul(query.float(), k.float().transpose(2, 3)) * scaling
        if attention_mask is not None:
            logits = logits + attention_mask[..., :kv_len].float()
        elif q_len > 1:
            qpos = torch.arange(q_start, q_start + q_len, device=query.device)[:, None]
            kpos = torch.arange(kv_len, device=query.device)[None, :]
            logits = logits.masked_fill(kpos > qpos, float("-inf"))
        a = torch.softmax(logits, dim=-1)

        _record(state, a, logits, k, layer_idx, q_start, q_len)

        out = torch.matmul(a.to(v.dtype), v).transpose(1, 2).contiguous()
        return out, None

    return sink_scan_attention_forward


def _record(state: SinkScan, a, logits, k, layer_idx, q_start, q_len):
    """Reduce this layer's [1, H, q, kv] weights to [H, N] columns, then let them go."""
    img = state.img_cols
    H = a.shape[1]
    dev = a.device
    for name in state.q_sets:
        rows = state.rows_for(name, q_start, q_len, dev)
        if rows.numel() == 0:
            continue
        sub = a[0, :, rows, :]                                     # [H, R, kv]
        w = sub[..., img]                                          # [H, R, N]
        slot = state._acc[name].setdefault(layer_idx, {})
        for key_, val in (("sum", w.sum(1)), ("tot", sub.sum(-1).sum(1))):
            prev = slot.get(key_)
            slot[key_] = val.float() if prev is None else prev + val.float()
        state._rows[name][layer_idx] = (state._rows[name].get(layer_idx, 0)
                                        + int(rows.numel()))
        if name != PRIMARY_Q:
            continue          # the variance, the budget and the key stats are primary-only
        prev = slot.get("sq")
        sq = (w ** 2).sum(1).float()
        slot["sq"] = sq if prev is None else prev + sq
        # the span budget, on the same rows
        spans = torch.stack([sub[..., state.spans[s]].sum(-1).sum(1)
                             if state.spans[s].numel() else torch.zeros(H, device=dev)
                             for s in SPANS]).float()              # [S, H]
        prevs = state._span_acc.get(layer_idx)
        state._span_acc[layer_idx] = spans if prevs is None else prevs + spans
        if state.want_key_stats:
            kn = k[0, :, img, :].float().norm(dim=-1)               # [H, N]
            lg = logits[0, :, rows, :][..., img].float().sum(1)     # [H, N]
            pk, pl = state._key.get(layer_idx), state._logit.get(layer_idx)
            state._key[layer_idx] = kn if pk is None else pk        # keys do not vary
            state._logit[layer_idx] = lg if pl is None else pl + lg


# ---------------------------------------------------------------------------
# A9 -- permute which patch embedding sits in which grid slot
# ---------------------------------------------------------------------------
class PatchPermute:
    """Swap the vision tower's output rows around, leaving the slots where they are.

    The cleanest dissociation this experiment has. Every pixel-space transform can be
    answered with "your transform changed the content in some way you did not model"; this
    one changes nothing but WHICH SLOT HOLDS WHICH VECTOR. Position ids, the grid, the
    prompt and the number of tokens are all untouched, and the vision tower has already
    run, so no feature is recomputed.

    The model's answer becomes nonsense. That is expected and it is not the readout: the
    readout is where the attention went.

    One picture only. With two, the permutation would have to respect each image's own
    span, and silently mixing two pictures' patches is a different experiment.
    """

    def __init__(self, model, mode="shuffle", seed=0, family=None):
        self.model, self.mode, self.seed = model, mode, seed
        self.family = family or VF.family_for(model)
        self.perm = None
        self._handles = []

    def _visual(self):
        return self.family.row_module(self.model)

    def _permutation(self, n, device):
        g = torch.Generator(device="cpu").manual_seed(int(self.seed))
        if self.mode == "shuffle":
            p = torch.randperm(n, generator=g)
        elif self.mode == "roll":
            p = torch.roll(torch.arange(n), shifts=max(1, n // 3))
        elif self.mode == "identity":
            p = torch.arange(n)
        else:
            raise ValueError(f"unknown permutation mode {self.mode!r}")
        return p.to(device)

    def _hook(self, module, args, out):
        got, perm = self.family.permute_rows(out, self._permutation)
        self.perm = perm
        return got

    def install(self):
        self._handles.append(self._visual().register_forward_hook(self._hook))
        return self

    def uninstall(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class VisionTap:
    """Keep the norm of every patch embedding the vision tower hands to the LLM.

    M1: if border patches are norm outliers HERE -- before a single text token exists --
    then the sink was decided by the encoder and the language model inherited it. The
    three deepstack features are kept too, because those are injected into the LLM's early
    layers and are a second way the encoder can plant one. Only Qwen3-VL has them; the
    other two families run a single projector, which is one fewer place a mark can be
    planted and makes any difference in the result interpretable.
    """

    def __init__(self, model, family=None):
        self.model = model
        self.family = family or VF.family_for(model)
        self.norms = None
        self.deepstack_norms = None
        self._handles = []

    def _hook(self, module, args, out):
        got = self.family.row_norms(out)
        if got is not None:
            self.norms = got
        self.deepstack_norms = self.family.deepstack_norms(out)
        return out

    def install(self):
        self._handles.append(
            self.family.row_module(self.model).register_forward_hook(self._hook))
        return self

    def uninstall(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# ---------------------------------------------------------------------------
# the arms -- image transforms, and the grid correspondence each one implies
# ---------------------------------------------------------------------------
#: name -> (pixel transform, inverse coordinate map). The inverse map takes a point in
#: NORMALISED output coordinates and returns the normalised INPUT point it came from, or
#: None if the output point shows something the input never had (padding, canvas). Every
#: correspondence in the analysis goes through it, so a transform cannot be added in pixel
#: space without also saying where its pixels came from.
def _inv_identity(p):
    return p


def _inv_rot90(p):
    u, v = p
    return (1.0 - v, u)              # PIL ROTATE_90 / np.rot90 are counter-clockwise


def _inv_rot180(p):
    u, v = p
    return (1.0 - u, 1.0 - v)


def _inv_hflip(p):
    u, v = p
    return (1.0 - u, v)


def _inv_zoom(frac):
    def f(p):
        u, v = p
        return (0.5 + (u - 0.5) * frac, 0.5 + (v - 0.5) * frac)
    return f


def _inv_pad(px, py):
    def f(p):
        u, v = p
        if not (px <= u <= 1 - px and py <= v <= 1 - py):
            return None
        return ((u - px) / (1 - 2 * px), (v - py) / (1 - 2 * py))
    return f


def _inv_canvas(scale, ox, oy):
    def f(p):
        u, v = p
        u0, v0 = (u - ox) / scale, (v - oy) / scale
        if not (0.0 <= u0 <= 1.0 and 0.0 <= v0 <= 1.0):
            return None
        return (u0, v0)
    return f


def _fill(image, colour, size):
    from PIL import Image
    if colour == "noise":
        rng = np.random.default_rng(12345)
        a = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
        return Image.fromarray(a, "RGB")
    named = {"grey": (128, 128, 128), "white": (255, 255, 255), "black": (0, 0, 0)}
    return Image.new("RGB", size, named.get(colour, (128, 128, 128)))


def transform(name, image, patch_px=32):
    """(image, inverse coordinate map, meta). `image` keeps its PIXEL SIZE wherever it can.

    Keeping the size fixed keeps the patch grid fixed, so an arm and its baseline are
    compared on the same geometry and a change in `ring_share` cannot be a change in the
    ring's area. The two arms that deliberately break that -- `rot90`, which transposes it,
    and the resolution ladder -- say so in `meta`.
    """
    from PIL import Image

    W, H = image.size
    if name == "identity":
        return image, _inv_identity, {}
    if name == "rot90":
        return image.transpose(Image.ROTATE_90), _inv_rot90, {"grid_transposed": True}
    if name == "rot180":
        return image.transpose(Image.ROTATE_180), _inv_rot180, {}
    if name == "hflip":
        return image.transpose(Image.FLIP_LEFT_RIGHT), _inv_hflip, {}
    if name.startswith("zoom"):
        frac = float(name[4:] or 60) / 100.0
        w, h = max(1, int(round(W * frac))), max(1, int(round(H * frac)))
        box = ((W - w) // 2, (H - h) // 2, (W - w) // 2 + w, (H - h) // 2 + h)
        return (image.crop(box).resize((W, H), Image.BICUBIC), _inv_zoom(frac),
                {"crop_frac": frac})
    if name.startswith("pad_"):
        # <colour>[_<k>], k patches of border on every side. The CONTENT is shrunk so the
        # canvas keeps its size: a bigger canvas would be resized by the processor and the
        # grid would change underneath the comparison.
        parts = name.split("_")
        colour = parts[1]
        k = int(parts[2]) if len(parts) > 2 else 1
        bx, by = k * patch_px, k * patch_px
        if W - 2 * bx < patch_px or H - 2 * by < patch_px:
            return None, None, {"skipped": "picture too small to pad"}
        out = _fill(image, colour, (W, H))
        out.paste(image.resize((W - 2 * bx, H - 2 * by), Image.BICUBIC), (bx, by))
        return out, _inv_pad(bx / W, by / H), {"pad_patches": k, "colour": colour}
    if name.startswith("canvas"):
        # canvas<pos>, pos in 0..8 reading the 3x3 placements in raster order.
        pos = int(name[6:] or 4)
        scale = 0.5
        ox = (pos % 3) * (1 - scale) / 2.0
        oy = (pos // 3) * (1 - scale) / 2.0
        out = _fill(image, "grey", (W, H))
        out.paste(image.resize((max(1, int(W * scale)), max(1, int(H * scale))),
                               Image.BICUBIC),
                  (int(round(ox * W)), int(round(oy * H))))
        return out, _inv_canvas(scale, ox, oy), {"canvas_pos": pos, "scale": scale}
    if name == "donut":
        # Blank the middle, keep the border. If the sink is content-seeking it must move
        # here; if it is positional it will not.
        a = np.asarray(image.convert("RGB"))
        border = np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]]).reshape(-1, 3)
        col = tuple(int(x) for x in np.median(border, axis=0))
        s = math.sqrt(CENTER_RECT_FRAC)
        w, h = int(round(W * s)), int(round(H * s))
        out = image.copy()
        out.paste(Image.new("RGB", (w, h), col), ((W - w) // 2, (H - h) // 2))
        return out, _inv_identity, {"blank_frac": CENTER_RECT_FRAC, "fill": col}
    if name.startswith("res"):
        side = int(name[3:])
        s = side / float(max(W, H))
        out = image.resize((max(1, round(W * s)), max(1, round(H * s))), Image.BICUBIC)
        return out, _inv_identity, {"max_side": side, "grid_changes": True}
    raise ValueError(f"unknown transform {name!r}")


ARMS = (
    "identity",
    "rot90", "rot180", "hflip",                     # A1  position vs content vs raster
    "zoom60",                                       # A2  the ring becomes foreground
    "pad_grey_1", "pad_white_1", "pad_noise_1", "pad_grey_2",   # A3
    "canvas0", "canvas4", "canvas8",                # A4  blank everywhere, object moves
    "donut",                                        # A5  blank centre, content on the ring
    "res256", "res384",                             # A7  the resolution ladder
)
#: Arms handled outside `transform()`: A6 needs a second picture in the prompt, A8 needs a
#: different question, A9 is an embedding permutation, A10 needs the grid to cut the
#: picture on, and `tiled` needs a different processor setting. The probe owns those.
SPECIAL_ARMS = ("two_images", "prompt_swap", "permute", "permute_identity",
                "permute_pixels", "permute_pixels_identity", "tiled")

#: What `--arms` runs when it is not given one. `tiled` is left out on purpose: it is
#: InternVL-only, it costs one forward per tile, and it answers a different question --
#: whether the ring tracks the ENCODER'S INPUT BOUNDARY rather than the picture -- which
#: deserves to be asked deliberately.
DEFAULT_ARMS = ARMS + ("two_images", "prompt_swap", "permute", "permute_identity",
                       "permute_pixels", "permute_pixels_identity")

#: Arms for which "did the peak follow the content / the slot" is not a question. Both
#: columns compare a patch index in the arm's grid against one in the baseline's, which
#: means something only when the two grids frame the same thing. A TILE's grid does not:
#: slot 0 of tile 3 and slot 0 of the whole picture are different places, and the two
#: numbers would read as a follow rate while measuring a coincidence.
NO_FOLLOW = ("tiled",)

#: An arm whose baseline is not `identity`. A10 has to resize the picture so the grid
#: divides it exactly before it can shuffle whole cells, and that resize is not free, so
#: its control is the SAME resize with the identity permutation. Pairing it against
#: `identity` would price the resize as part of the result.
ARM_BASELINE = {"permute_pixels": "permute_pixels_identity"}


def patch_correspondence(inv, gh, gw, gh0, gw0, view=FULL_VIEW, view0=FULL_VIEW):
    """For each patch of the transformed grid, the baseline patch it shows. -> [gh*gw] int.

    -1 where the output shows something the input never had. This is the only thing that
    lets "did the sink follow the CONTENT" be asked at all, and it is the single most
    dangerous piece of arithmetic in the experiment: an off-by-one here decodes the wrong
    coordinate frame and answers the question confidently and wrongly. `--stage selftest`
    checks it against a picture with one bright patch at a known place.

    THREE FRAMES, NOT TWO. A patch is a cell of the grid; `inv` speaks in normalised
    PICTURE coordinates; and the grid only covers the picture where the processor hands
    the whole thing to the encoder. LLaVA-1.5's does not -- it resizes the short side and
    centre-crops -- so `view` and `view0` are the sub-rectangles the two grids cover, and
    the walk is grid -> picture -> (inv) -> baseline picture -> baseline grid. Both
    default to the whole picture, which is what Qwen3-VL and InternVL do and what every
    published number was computed under.
    """
    u0v, v0v, u1v, v1v = view
    u0b, v0b, u1b, v1b = view0
    dw, dh = (u1v - u0v), (v1v - v0v)
    bw, bh = (u1b - u0b), (v1b - v0b)
    out = np.full(gh * gw, -1, dtype=np.int64)
    for r in range(gh):
        for c in range(gw):
            p = inv((u0v + ((c + 0.5) / gw) * dw, v0v + ((r + 0.5) / gh) * dh))
            if p is None:
                continue
            u0, v0 = p
            if not (u0b <= u0 < u1b and v0b <= v0 < v1b):
                continue
            r0 = min(gh0 - 1, max(0, int((v0 - v0b) / bh * gh0)))
            c0 = min(gw0 - 1, max(0, int((u0 - u0b) / bw * gw0)))
            out[r * gw + c] = r0 * gw0 + c0
    return out
