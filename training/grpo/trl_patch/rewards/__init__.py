# Copyright 2026 NVIDIA. Apache-2.0.
"""The reward functions of Section 3.4, as TRL reward callables.

    R = a_format*R_format + a_sal*R_sal + a_direct*R_direct + a_llm*R_llm

`self_saliency` is ours. `saliency_r1` is the Appendix D.2 baseline's, kept beside it so
the two arms differ in the attention term and in nothing else. R_direct is inlined in the
entry script and R_llm is `selfsal.judge`, because both are needed outside training too.

Eager imports, not TRL's lazy _LazyModule. The trainer imports `is_active` and
`pop_mask_diagnostics` from here inside its metrics block, which runs on the FIRST
optimizer step of every run -- a lazy failure there surfaces after generation, the
re-forward and the detector calls, on a GPU, rather than at import.
"""

from .answer import answer_format_reward
from .format import think_format_reward
from .saliency_r1 import think_saliency_reward
from .self_saliency import (configure, mask_diag_active, pop_mask_diagnostics,
                            self_saliency_reward, think_overlap_reward)

__all__ = [
    "answer_format_reward", "think_format_reward", "think_saliency_reward",
    "configure", "mask_diag_active", "pop_mask_diagnostics",
    "self_saliency_reward", "think_overlap_reward",
]
