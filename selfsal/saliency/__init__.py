# Copyright 2026 NVIDIA. Apache-2.0.
"""Attention -> patch map (Section 3.3), and the score over it (Equation 1)."""

from .heads import REWARD_HEADS, REWARD_LAYER, TOKEN_REDUCTION, head_spec
from .score import METRICS, phi, phi_mean, resolve_metric, saliency_reward, score_step

__all__ = [
    "REWARD_HEADS", "REWARD_LAYER", "TOKEN_REDUCTION", "head_spec",
    "METRICS", "phi", "phi_mean", "resolve_metric", "saliency_reward", "score_step",
]
