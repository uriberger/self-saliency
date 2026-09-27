# Copyright 2026 NVIDIA. Apache-2.0.
"""Which attention heads the saliency score reads (Section 3.5).

One constant, in one place, because three things have to agree on it: the reward that
trains against these heads, the analysis in Section 4.4 that asks whether they moved,
and Figure 5, which draws them before and after.

HOW THEY WERE CHOSEN. `experiments/head_selection/` is the procedure. The base model is
run over Visual-CoT, phi is computed per sample for every one of the 36 x 32 heads, and
each head is scored by the correlation between its phi and whether the model answered
correctly. The top two are layer 22, heads 28 and 31. Footnote 3's cross-dataset check
is `experiments/head_selection/cross_dataset.py`.

Layer 22 is not arbitrary in the literature either: Gandikota & Bau (2026) report the
strongest attention-to-described-region alignment in layers 20-28, and their Figure 1B
puts Qwen3-VL-8B's peak at layer 22.

ONE THING TO KNOW BEFORE READING THESE HEADS AS "WHERE THE MODEL LOOKS". They were
selected for correlation with CORRECTNESS, not for attending the region, and they are
more border-biased than the layer's average head. Section 4.4 finds in-region attention
flat at this pair and rising across layer 22 as a whole -- so a claim about the model's
attention wants the layer, and a claim about the reward wants the pair. They are
different measurements and the paper keeps them apart.
"""

from __future__ import annotations

#: The layer the saliency map is read from.
REWARD_LAYER = 22

#: The heads within that layer, averaged to form the map.
REWARD_HEADS = (28, 31)

#: How a step's per-token maps are collapsed into one map for the step.
TOKEN_REDUCTION = "mean"


def head_spec() -> str:
    """Human-readable form, as it appears in run names and figure captions."""
    return "L{}H{}".format(REWARD_LAYER, "+".join(str(h) for h in REWARD_HEADS))
