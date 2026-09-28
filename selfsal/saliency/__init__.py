# Copyright 2026 NVIDIA. Apache-2.0.
"""Attention -> patch map (Section 3.3), and the score over it (Equation 1).

`maps` needs torch and is imported lazily, so `from selfsal.saliency import phi` works
in an environment that has only numpy -- which is what the CPU tests and the offline
table builders run in.
"""

from .heads import REWARD_HEADS, REWARD_LAYER, TOKEN_REDUCTION, head_spec
from .score import METRICS, phi, phi_mean, resolve_metric, saliency_reward, score_step

__all__ = [
    "REWARD_HEADS", "REWARD_LAYER", "TOKEN_REDUCTION", "head_spec",
    "METRICS", "phi", "phi_mean", "resolve_metric", "saliency_reward", "score_step",
    "AttentionCollector", "IMAGE_TOKEN_ID",
]


def __getattr__(name):
    if name in ("AttentionCollector", "IMAGE_TOKEN_ID", "repeat_kv", "sdpa",
                "attention_weights"):
        from . import maps
        return getattr(maps, name)
    raise AttributeError(name)
