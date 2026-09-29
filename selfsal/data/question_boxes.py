# Copyright 2026 NVIDIA. Apache-2.0.
"""Precomputed per-question regions, for the `question_boxes` ablation (Section 5.3).

One box list per dataset ROW, grounded once on the row's question before training starts
by `training/grpo/precompute_question_boxes.py`. The trainer then loads the file and
never constructs a detector.

IN `selfsal` RATHER THAN IN THE REWARD, because this is a FILE FORMAT and two programs
depend on it: the builder that writes the file and the reward that reads it. They must
agree on the version, on the key columns, and on what the refusals below mean, and the
builder must not need a patched TRL checkout to learn any of that. Both import it
absolutely from here, which is the same reason phi lives in `selfsal.saliency` -- see
`trl/rewards/self_saliency.py`, whose docstring calls itself a shell for exactly this.

This is what makes the ablation the comparison it claims to be: prior work fixes the
target regions from the image and question alone, and the gap to SELF-SALIENCY is the
value of conditioning them on the chain the policy is generating instead.

The loader REFUSES a file whose box threshold or image cap disagrees with the run's,
rather than training against regions built under different settings. That refusal is
also what pins the ablation's threshold after the fact -- the file records 0.10, so the
run cannot have used anything else.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# One box list per dataset ROW, grounded once on the row's question by
# precompute_question_boxes.py before the run, and reused for every observe step of
# every completion of that row. See the module docstring for why the per-step call is
# not buying a per-step mask.

QBOX_VERSION = 1

# The row identity. Every corpus this trainer accepts carries all three -- the
# saliency-r1-8k default and every cold_data/grpo_sets/* built by build_grpo_sets.py --
# and the triple is unique in each of them (the builder checks, and so does the loader).
# `problem` is deliberately NOT part of it: questions repeat across images (35343 distinct
# strings over set_a's 50000 rows), so keying on the text would collapse different
# pictures onto one box list.
QBOX_KEY_COLUMNS = ("dataset", "split", "question_id")

# Loaded once per process, on first use.
_QBOX: dict = {"path": None, "boxes": None, "meta": None}


def qbox_key(dataset, split, question_id) -> str:
    """The cache key for one dataset row.

    Joined with '|' rather than JSON-encoded so the file stays readable; no corpus here
    has a separator in any of the three fields, and the builder refuses one that does
    instead of silently producing a key that two rows could share.
    """
    parts = [str(dataset), str(split), str(question_id)]
    for name, part in zip(QBOX_KEY_COLUMNS, parts):
        if "|" in part:
            raise ValueError(
                f"question-box key column `{name}` contains the '|' separator: {part!r}. "
                "Two rows could then collide onto one key."
            )
    return "|".join(parts)


def load_question_boxes(path: str, box_threshold=None, max_image_side=None) -> dict:
    """Read (once per process) and validate a precomputed question-box file.

    Both checks are hard failures, because both fail SILENTLY otherwise -- the run would
    train happily on boxes that are not the ones its own configuration describes:

      box_threshold   applied inside DINO, so it cannot be re-applied here. A cache built
                      at 0.10 cannot serve a run asking for 0.25.
      max_image_side  the detector sees a different picture at a different resolution.
                      The trainer passes its own constant so the two cannot drift.
    """
    if _QBOX["path"] == path and _QBOX["boxes"] is not None:
        return _QBOX["boxes"]

    import json

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"--overlap_question_boxes {path} does not exist. Build it first with "
            f"precompute_question_boxes.py (it needs a GPU, and it is a separate job)."
        )
    with open(path) as f:
        d = json.load(f)

    got_v = d.get("version")
    if got_v != QBOX_VERSION:
        raise ValueError(f"{path}: question-box file version {got_v}, expected {QBOX_VERSION}")

    meta = d.get("config") or {}
    if box_threshold is not None:
        want, have = float(box_threshold), meta.get("box_threshold")
        if have is None or abs(float(have) - want) > 1e-9:
            raise ValueError(
                f"{path} was built with box_threshold={have}, but this run asks for "
                f"{want}. The threshold is applied inside Grounding-DINO, so the cached "
                "boxes cannot be re-filtered to it -- rebuild the cache, or match the flag."
            )
    if max_image_side is not None:
        want, have = int(max_image_side), meta.get("max_image_side")
        if have is None or int(have) != want:
            raise ValueError(
                f"{path} was built with max_image_side={have}, but the trainer resizes to "
                f"{want}. Grounding-DINO sees a different picture at a different "
                "resolution -- rebuild the cache against this trainer."
            )

    boxes = d.get("boxes")
    if not isinstance(boxes, dict) or not boxes:
        raise ValueError(f"{path}: no `boxes` mapping")

    _QBOX.update(path=path, boxes=boxes, meta=meta)
    n_empty = sum(1 for v in boxes.values() if not v)
    print(f"[overlap_reward] question boxes: {len(boxes)} rows from {path} "
          f"(box_threshold={meta.get('box_threshold')}, "
          f"max_image_side={meta.get('max_image_side')}, "
          f"{n_empty} rows grounded nothing). Grounding-DINO will not be loaded.",
          flush=True)
    return boxes


def boxes_per_row(path, kwargs, wanted):
    """-> {completion index: raw box list} for the completions in `wanted`.

    Only the completions that actually have steps to score are looked up, for the same
    reason masked rows never reach DINO on the per-step path: a row this call is not
    scoring must not be able to fail it.

    A key the cache does not hold IS a hard failure. It means the cache was built for a
    different corpus, and masking those rows instead would show up only as a quietly
    smaller reward on part of the batch.
    """
    if not wanted:
        return {}
    absent = [c for c in QBOX_KEY_COLUMNS if kwargs.get(c) is None]
    if absent:
        raise KeyError(
            f"--overlap_question_boxes needs the {', '.join(QBOX_KEY_COLUMNS)} columns to "
            f"identify a row, but {', '.join(absent)} did not reach the reward function. "
            "Use a corpus built by build_grpo_sets.py (cold_data/grpo_sets/*) or the "
            "saliency-r1-8k default."
        )
    boxes = load_question_boxes(path)
    cols = [kwargs[c] for c in QBOX_KEY_COLUMNS]
    out = {}
    for c in sorted(wanted):
        key = qbox_key(*(col[c] for col in cols))
        if key not in boxes:
            raise KeyError(
                f"row {key!r} is not in {path}. The cache does not cover "
                "the dataset being trained on -- rebuild it for this corpus."
            )
        out[c] = boxes[key]
    return out


