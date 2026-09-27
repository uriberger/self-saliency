# Copyright 2026 NVIDIA. Apache-2.0.
"""`selfsal` must score exactly what the paper's runs scored.

`selfsal/saliency/score.py` and `selfsal/grounding/mask.py` were extracted from the
reward module the published checkpoints were actually trained against
(`trl/rewards/overlap_rewards.py` in the archive repo). An extraction is a rewrite, and
a rewrite that is a few percent off would leave every number in the paper attached to
code that no longer computes it -- silently, because phi has no ground truth to check
against.

So this holds the two implementations side by side on randomised grids, box sets, map
dtypes and filter settings, and asserts they agree. Skipped when the archive repo is
not on this machine, which is the only reason it would not run.

    pytest tests/test_score_matches_paper_runs.py

WHAT IS ALLOWED TO DIFFER, AND ONLY THIS. The original divided phi in the map's own
dtype while computing phi_mean in float64; `selfsal` uses float64 for both. On float32
maps that is ~4e-8 relative and on the float16 maps the offline screen stores it
reaches ~6e-4 -- in both cases the float64 answer is the correct one, and in both cases
the difference is orders of magnitude below the within-group reward spread the GRPO
advantage divides by (~0.0086). Masks, rectangles, ring fractions and phi_mean must be
bit-identical, and are.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ARCHIVE = Path(os.environ.get(
    "SELFSAL_ARCHIVE",
    Path.home() / "scratch/research/saliency_r1"))

pytestmark = pytest.mark.skipif(
    not (ARCHIVE / "trl/rewards/overlap_rewards.py").exists(),
    reason=f"archive repo not present at {ARCHIVE}; set SELFSAL_ARCHIVE to point at it")


def _load_original():
    """Import the archive's reward module without importing the trainer package.

    It lives inside `trl.rewards`, whose `__init__` lazily pulls in the whole trainer.
    Loading the two files under a throwaway package name gets the functions with none
    of that.
    """
    pkg = types.ModuleType("_selfsal_archive")
    pkg.__path__ = []
    sys.modules["_selfsal_archive"] = pkg
    for name in ("roll_null", "overlap_rewards"):
        spec = importlib.util.spec_from_file_location(
            f"_selfsal_archive.{name}", ARCHIVE / f"trl/rewards/{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"_selfsal_archive.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["_selfsal_archive.overlap_rewards"]


def _random_boxes(rng, n):
    out = []
    for _ in range(n):
        x1, x2 = sorted(rng.random(2))
        y1, y2 = sorted(rng.random(2))
        out.append([float(x1), float(y1), float(x2), float(y2)])
    return out


# float16 is the dtype the offline head-selection screen stores maps in; float32 is what
# the trainer produces. Both appear, so both are checked.
_TOL = {np.float16: 1e-3, np.float32: 1e-6, np.float64: 1e-12}


def test_masks_and_scores_agree():
    from selfsal.grounding import center_rect_mask, ring_fraction, union_mask
    from selfsal.saliency import phi, phi_mean

    old = _load_original()
    rng = np.random.default_rng(0)
    seen = dict(phi=0, phi_mean=0, mask=0, rect=0)

    for trial in range(600):
        gh, gw = int(rng.integers(2, 20)), int(rng.integers(2, 20))
        boxes = _random_boxes(rng, int(rng.integers(0, 6)))

        for dtype in _TOL:
            step_map = rng.random((gh, gw)).astype(dtype)
            if trial % 7 == 0:          # the all-zero map: phi_mean must decline to score
                step_map[:] = 0.0

            for box_cap in (None, 0.5, 0.2):
                for union_cap in (None, 0.6):
                    old._CFG["max_box_area"] = box_cap
                    old._CFG["max_union_area"] = union_cap

                    want = old._union_mask(boxes, gh, gw)
                    got = union_mask(boxes, gh, gw,
                                     max_box_area=box_cap, max_union_area=union_cap)
                    assert (want is None) == (got is None), "disagree on degenerate union"
                    if want is None:
                        continue
                    assert np.array_equal(want, got)
                    seen["mask"] += 1

                    a, b = old._mean_in(step_map, want), phi(step_map, got)
                    assert (a is None) == (b is None)
                    if a is not None:
                        assert abs(a - b) <= _TOL[dtype] * max(abs(a), 1e-30)
                        seen["phi"] += 1

                    a, b = old._mean_in_v2(step_map, want), phi_mean(step_map, got)
                    assert (a is None) == (b is None)
                    if a is not None:
                        assert abs(a - b) <= 1e-12 * max(abs(a), 1.0)
                        seen["phi_mean"] += 1

                    assert abs(old._ring_frac(want) - ring_fraction(got)) < 1e-12

        for frac in (0.565, 0.2, 0.9, 1.0):   # 0.565 is the center_rect arm's fraction
            want = old._centre_rect_mask(gh, gw, frac)
            got = center_rect_mask(gh, gw, frac)
            assert (want is None) == (got is None)
            if want is not None:
                assert np.array_equal(want, got)
                seen["rect"] += 1

    # A silent no-op would pass every assertion above.
    assert min(seen.values()) > 100, f"too few comparisons actually ran: {seen}"


def test_ungrounded_step_is_skipped_not_zero():
    """None, never 0.0 -- the distinction the reward depends on.

    An observe step Grounding-DINO could not ground says nothing about where the model
    looked. Scoring it zero would make "name something ungroundable" a move with a
    payoff, since it would drag the completion's mean down for a rival and not for you.
    """
    from selfsal.saliency import phi, phi_mean, saliency_reward

    empty = np.zeros((4, 5), dtype=bool)
    m = np.random.default_rng(0).random((4, 5))
    assert phi(m, empty) is None
    assert phi_mean(m, empty) is None
    assert saliency_reward([None, 0.5, None]) == pytest.approx(0.5)
    assert saliency_reward([None, None]) is None


def test_metric_aliases_resolve():
    """Configs written against the research tree still name a real metric."""
    from selfsal.saliency import resolve_metric

    assert resolve_metric("mean_in") == "phi"
    assert resolve_metric("mean_in_v2") == "phi_mean"
    assert resolve_metric(None) == "phi"
    with pytest.raises(ValueError):
        resolve_metric("auroc")      # a research metric, left in the archive
