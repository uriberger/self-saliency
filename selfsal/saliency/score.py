# Copyright 2026 NVIDIA. Apache-2.0.
"""The saliency score phi, Equation 1 -- and the Appendix C variant.

This module is the reason `selfsal` is one package. Two things must compute the same
phi or the paper does not hold together:

  * the GRPO reward of Section 3.4, which pays the policy for raising it, and
  * the head-selection screen of Section 3.5, which chose layer 22 heads 28 and 31 by
    correlating phi with correctness.

While the work was in progress those were two implementations in two repos, and
keeping them in step was a manual act. Here there is one.

DEFINITIONS

Given a step's patch-level saliency map `sal_s` and the grounded region `u_s` (a
boolean mask over the same patch grid),

    phi(s)      = mean_{p in u_s} sal_s(p) / max_{p in I} sal_s(p)      (Equation 1)
    phi_mean(s) = mean_{p in u_s} sal_s(p) / mean_{p in I} sal_s(p)     (Appendix C)

The two differ only in the denominator, and the difference is not cosmetic.

`phi` divides by the map's own PEAK. Section 5.2 is why that matters and why it is the
variant that works: attention peaks at a grid corner while grounded regions sit near
the centre, so the argmax patch is usually OUTSIDE u_s -- and then raising attention
inside u_s raises the numerator without moving the denominator. The reward has a
gradient to climb.

`phi_mean` divides by the map's MEAN. Chance is exactly 1.0 and the value is invariant
to `m -> c*m`, which is tidier. It is also the variant that did not beat the no-sal
baseline (Table 7). Both are provided; `phi` is the default because it is what the
paper's headline arm was trained on.

A NOTE ON WHAT MAKES phi MOVE. Because the denominator is the peak, a map that merely
FLATTENS scores higher inside the mask without attending it any better. That is a real
property of the metric, not a bug in this code, and Section 4.4 is the measurement of
how much of the trained model's gain took that route. Anything reading phi as "how well
the model attends the region" should read Section 4.4 first.

PRECISION. Both functions accumulate in float64 regardless of the map's dtype. The
research implementations did not agree on this -- phi divided in the map's own dtype
while phi_mean already upcast -- so on the float16 maps the offline screen stores and
the float32 maps the trainer produces, the two metrics were being computed at different
precisions. The disagreement against the original phi is ~4e-8 relative, which is
float32 epsilon and orders of magnitude below the within-group reward spread the GRPO
advantage is normalised by (~0.0086). It is not a difference any run would notice; it
is fixed here so the reward and the head-selection screen cannot drift apart.
"""

from __future__ import annotations

import numpy as np

# Historical spellings, kept so a command line or a config written against the research
# tree still resolves. `mean_in` was phi; `mean_in_v2` was phi_mean.
_METRIC_ALIASES = {
    "mean_in": "phi",
    "mean_in_v2": "phi_mean",
}

METRICS = ("phi", "phi_mean")


def resolve_metric(name: str | None) -> str:
    """Canonical metric name, accepting the historical spellings."""
    if not name:
        return "phi"
    name = _METRIC_ALIASES.get(name, name)
    if name not in METRICS:
        raise ValueError(
            f"unknown saliency metric {name!r}; expected one of {'|'.join(METRICS)} "
            f"(or the historical {'|'.join(_METRIC_ALIASES)})")
    return name


def phi(step_map, mask) -> float | None:
    """Equation 1: mean of the max-normalised saliency inside the grounded region.

    Returns None for an empty mask -- the step is then SKIPPED, not scored zero. That
    distinction matters to the reward: a step Grounding-DINO could not ground carries no
    information about where the model looked, and scoring it zero would make "say
    something ungroundable" a move with a payoff.
    """
    v = np.asarray(step_map, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    inside = v[m]
    if inside.size == 0:
        return None
    vmax = float(v.max())
    return float(inside.mean() / vmax) if vmax > 0 else float(inside.mean())


def phi_mean(step_map, mask) -> float | None:
    """Appendix C: the same numerator, over the mean of the map rather than its peak.

    None on an empty mask, and also on an all-zero map -- there the ratio is 0/0 and the
    honest answer is "this step tells us nothing", which is the same skip.
    """
    v = np.asarray(step_map, dtype=np.float64)
    inside = v[np.asarray(mask, dtype=bool)]
    if inside.size == 0:
        return None
    denom = float(v.mean())
    if denom <= 0:
        return None
    return float(inside.mean()) / denom


def score_step(step_map, mask, metric: str | None = None) -> float | None:
    """phi or phi_mean for one step. None means skip, not zero."""
    metric = resolve_metric(metric)
    return phi(step_map, mask) if metric == "phi" else phi_mean(step_map, mask)


def saliency_reward(step_scores) -> float | None:
    """R_sal (Section 3.4): the mean of phi over a completion's grounded observe steps.

    Steps that scored None are dropped rather than counted as zero, so the mean is over
    `O` as the paper defines it -- the observe steps that were actually grounded. A
    completion with none of them returns None and is masked out of the GRPO advantage.
    """
    vals = [float(s) for s in step_scores if s is not None]
    return float(np.mean(vals)) if vals else None


def auroc(step_map, mask) -> float | None:
    """P(a random in-region patch outranks a random out-region one). 0.5 is chance.

    A DIAGNOSTIC, deliberately absent from METRICS so it cannot be selected as a training
    objective: no arm in the paper was trained on it, and offering it here would invite a
    run that no published number describes.

    It is worth reporting beside phi because it answers a question phi cannot. phi divides
    by the map's peak, so it moves when a map merely flattens; auroc depends only on the
    ORDER of the patches and is exactly invariant to any monotone reshaping m -> m**gamma.
    Reading the two together is how "the region got more attention" is told apart from
    "the map got flatter", which is the distinction Section 4.4 turns on.

    Average ranks for ties: attention maps carry many near-identical near-zero patches and
    argsort would break those ties arbitrarily, biasing the estimate.
    """
    v = np.asarray(step_map, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    n_in = int(m.sum())
    n_out = v.size - n_in
    if n_in == 0 or n_out == 0:
        return None
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=np.float64)
    ranks[order] = np.arange(1, v.size + 1, dtype=np.float64)
    _uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size, dtype=np.float64)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    u = ranks[m].sum() - n_in * (n_in + 1) / 2.0
    return float(u / (n_in * n_out))
