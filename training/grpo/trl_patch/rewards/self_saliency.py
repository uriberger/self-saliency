# Copyright 2026 NVIDIA. Apache-2.0.
"""R_sal: the self-grounded saliency reward (Section 3.4).

Per completion: segment the chain into steps, keep the `observe` ones, ground each one's
own sentence, and average phi over the steps that grounded.

    R_sal(c) = (1/|O|) * sum over grounded observe steps s of phi(s)

THIS MODULE IS A SHELL. phi is `selfsal.saliency.score`, the grounded region is
`selfsal.grounding.mask`, the detector is `selfsal.grounding.dino`, and the segmentation
is `selfsal.steps`. What is left here is the glue TRL needs: a function with TRL's reward
signature, a configuration the launcher sets from flags, and the per-step diagnostics the
trainer drains.

That split is the point. The head-selection screen of Section 3.5 imports exactly the
same four things, so the reward the policy is trained against and the screen that chose
its two attention heads cannot compute different phis.

TWO CONTRACTS THE TRAINER DEPENDS ON

None is not zero. A step the detector could not ground returns no score and is dropped
from the mean; a completion with none of them returns None and the trainer imputes its
group's mean, so it is neutral in the advantage. Scoring either as 0 would make "say
something ungroundable" a move with a payoff.

The format gate is multiplicative. An invalid completion scores 0 here rather than None,
because that IS a measurement: the policy produced something outside the required shape.
"""

from __future__ import annotations

import numpy as np

from selfsal.data.question_boxes import boxes_per_row, load_question_boxes  # noqa: F401
from selfsal.grounding import center_rect_mask, ground, ring_fraction, union_mask
from selfsal.saliency import resolve_metric, score_step

#: Set by the launcher from the CLI; see training/grpo/configs/.
_CFG = {
    "box_threshold": 0.10,
    "max_box_area": 0.5,      # per box, before rasterisation
    "max_union_area": None,   # per step, on the rasterised union; off in the paper's runs
    "metric": "phi",          # phi (Eq. 1) | phi_mean (App C)
    "dino_api_base": None,    # served detector; None loads one in-process
    "dino_device": None,
    "dino_batch_size": 32,
    "natural_only": False,
    "rect_frac": None,        # set -> the center_rect ablation, and no detector at all
    "rect_placement": "center",
    "question_boxes": None,   # set -> the question_boxes ablation, and no detector
}

#: Fixed key set, always drained in full. The trainer gathers these across ranks, and a
#: key set that depended on what a rank happened to see would make the NUMBER of
#: collectives rank-dependent, which hangs rather than fails.
MASK_DIAG_KEYS = ("union_frac", "ring_frac", "n_placements")
_MASK_DIAG: dict[str, list[float]] = {}


def _diag(key: str, value: float) -> None:
    _MASK_DIAG.setdefault(key, []).append(float(value))


def mask_diag_active() -> bool:
    """True when the mask came from somewhere other than per-step grounding.

    A rank-uniform CLI decision, so the trainer may branch its collectives on it.
    """
    return rect_active()


def pop_mask_diagnostics() -> dict[str, float]:
    """Mean of each mask diagnostic since the last call, then clear. Always all keys."""
    out = {k: (float(np.mean(_MASK_DIAG[k])) if _MASK_DIAG.get(k) else float("nan"))
           for k in MASK_DIAG_KEYS}
    _MASK_DIAG.clear()
    return out


def rect_active() -> bool:
    f = _CFG.get("rect_frac")
    return f is not None and float(f) > 0


def configure(**kwargs) -> None:
    """Set the reward's configuration. None values leave the default in place."""
    for k, v in kwargs.items():
        if v is not None:
            if k not in _CFG:
                raise KeyError(f"unknown saliency-reward setting {k!r}")
            _CFG[k] = v
    _CFG["metric"] = resolve_metric(_CFG.get("metric"))
    _validate()


def _validate() -> None:
    """Refuse the combinations that fail silently rather than loudly."""
    if _CFG.get("rect_placement") not in ("center", "centre"):
        raise ValueError(
            f"rect_placement must be 'center', got {_CFG['rect_placement']!r}. The "
            "interior placements were a research arm and are in the archive repo.")
    if rect_active() and _CFG.get("question_boxes"):
        raise ValueError(
            "rect_frac and question_boxes both REPLACE the per-step grounded region, in "
            "the same place. Whichever won, the other arm's name would be on a run it "
            "did not describe. Pick one.")
    if rect_active():
        f = float(_CFG["rect_frac"])
        if not 0 < f <= 1:
            raise ValueError(f"rect_frac must be in (0, 1], got {f}")
        cap = _CFG.get("max_union_area")
        if cap and f > float(cap):
            raise ValueError(
                f"rect_frac {f} exceeds max_union_area {cap}. The rectangle is the SAME "
                "on every step, so unlike a grounded union that cap is all-or-nothing: "
                "it would drop every step of every completion, and the logs would read "
                "as a run with no saliency signal rather than as a misconfiguration.")


def self_saliency_reward(completions=None, saliency_map=None, valid_list=None,
                         image=None, natural=None, **kwargs):
    """-> one score per completion, or None where nothing was scored.

    `saliency_map[c]` is the list of that completion's observe steps, each a dict with
    `map` (the patch-level saliency of Section 3.3) and `text` (the sentence to ground).
    The weight alpha_sal is applied by TRL through --reward_weights, not here.
    """
    n = len(saliency_map)
    if valid_list is None:
        valid_list = [True] * n

    if _CFG.get("natural_only"):
        if natural is None:
            raise KeyError(
                "natural_only needs a boolean 'natural' column in the dataset, but none "
                "reached the reward function.")
        scored = [bool(x) for x in natural]
    else:
        scored = [True] * n

    owner = [(c, si) for c, steps in enumerate(saliency_map)
             if steps and scored[c] for si in range(len(steps))]

    rect, qbox = rect_active(), bool(_CFG.get("question_boxes"))
    if rect:
        # The region is a function of the GRID alone. The detector is never constructed
        # -- `selfsal.grounding.dino` loads lazily, so not calling it is the whole
        # mechanism, and the launcher can give DINO's GPU to training.
        boxes = [None] * len(owner)
    elif qbox:
        # One grounding per dataset ROW, done before the run. Every step of a completion
        # gets the same region. Deliberately not hoisted out of the loop below: keeping
        # one scoring path means the cached and per-step arms differ in where the boxes
        # came from and in nothing else.
        per_row = boxes_per_row(_CFG["question_boxes"], kwargs, {c for c, _ in owner})
        boxes = [per_row[c] for c, _ in owner]
    else:
        boxes = ground([image[c] for c, _ in owner],
                       [saliency_map[c][si]["text"] for c, si in owner],
                       box_threshold=_CFG["box_threshold"],
                       api_base=_CFG["dino_api_base"],
                       batch_size=_CFG["dino_batch_size"],
                       device=_CFG["dino_device"]) if owner else []

    diag_on = mask_diag_active()
    per_completion: list[list[float]] = [[] for _ in range(n)]
    for (c, si), box_list in zip(owner, boxes):
        step_map = saliency_map[c][si]["map"]
        gh, gw = step_map.shape
        mask = (center_rect_mask(gh, gw, _CFG["rect_frac"]) if rect
                else union_mask(box_list or [], gh, gw,
                                max_box_area=_CFG["max_box_area"],
                                max_union_area=_CFG["max_union_area"]))
        if mask is None:
            continue                      # ungrounded or degenerate -> SKIP, not zero
        if diag_on:
            _diag("union_frac", float(mask.sum()) / mask.size)
            _diag("ring_frac", ring_fraction(mask))
            _diag("n_placements", 1.0)
        s = score_step(step_map, mask, _CFG["metric"])
        if s is not None:
            per_completion[c].append(s)

    rewards: list[float | None] = []
    for c in range(n):
        if not scored[c] or not per_completion[c]:
            rewards.append(None)          # masked -> neutral in the GRPO advantage
            continue
        value = float(np.mean(per_completion[c]))
        rewards.append(value * (1.0 if valid_list[c] else 0.0))   # format gate
    return rewards


#: The name the trainer and the launcher have always used.
think_overlap_reward = self_saliency_reward
