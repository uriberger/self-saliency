# Copyright 2026 NVIDIA. Apache-2.0.
"""The training contract: the system prompt, the output format, and the image budget.

These three are not incidental settings. Each one is depended on by more than one part
of the method, and a disagreement between any two of them is silent:

  SYSTEM_PROMPT    the cold start was supervised against it and GRPO samples under it.
                   A probe or an analysis that generates under a different prompt is
                   measuring a different policy, and the chains it produces cannot be
                   compared with training-time ones.

  FORMAT_PATTERN   R_format is 1 when a completion matches this and 0 otherwise, and the
                   saliency reward is gated by it multiplicatively. It is also what the
                   analyses use to decide whether a completion is well-formed, so a
                   mismatch would make "the reward saw a valid completion" and "the
                   table counted a valid completion" different claims.

  MAX_IMAGE_SIDE   the longer side every image is resized to before the model sees it.
                   The patch grid follows from it, and the patch grid is what every map
                   and every mask is defined on.

These lived in three places -- the reward, the probe and the launcher -- and agreed by
inspection. They agree by construction now.
"""

from __future__ import annotations

import re

#: Saliency-R1's prompt, which the cold start was trained against and GRPO samples under.
SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user asks a question, and the "
    "assistant solves it. The assistant first thinks about the reasoning process in the "
    "mind and then provides the user with the answer. The reasoning process and answer "
    "are enclosed within <think></think> tags, "
    "i.e., <think>\nThis is my reasoning.\n</think>\nThis is my answer."
)

#: A valid completion: a non-empty <think> block, then a non-empty answer. R_format is
#: 1 on a match and 0 otherwise.
FORMAT_PATTERN = r"^<think>\s*([^\s].*?)\s*</think>\s*([^\s].*?)\s*$"
FORMAT_RE = re.compile(FORMAT_PATTERN, re.DOTALL | re.MULTILINE)

#: Longer side, in pixels. Appendix A.2.
MAX_IMAGE_SIDE = 512

#: Qwen3-VL's <|image_pad|>, and the vision-span delimiters around a picture.
IMAGE_TOKEN_ID = 151655
VISION_START_ID = 151652
VISION_END_ID = 151653


def is_valid(completion: str) -> bool:
    """Whether R_format scores this completion 1."""
    return bool(FORMAT_RE.match(completion or ""))


def split_completion(completion: str):
    """(reasoning, answer) for a valid completion, else (None, None)."""
    m = FORMAT_RE.match(completion or "")
    return (m.group(1), m.group(2)) if m else (None, None)


def prepare_image(image, max_side: int = MAX_IMAGE_SIDE):
    """Resize so the longer side is at most `max_side`, then convert to RGB.

    Images at or under the cap are returned unresized rather than upscaled: the patch
    grid follows from the size the model is handed, and inventing pixels would invent
    patches for regions carrying no evidence.

    TWO DETAILS ARE LOAD-BEARING, AND BOTH READ AWKWARDLY ON PURPOSE.

    BILINEAR, AND THIS IS THE PROBES' FILTER, NOT TRAINING'S. The two really did differ,
    and the comment in the archive is what hides it. `A overlap_probe.py` (here
    `experiments/trained_model/probe.py`) and the RoPE-phase probes write
    `image.resize(..., 2)` with a trailing `# BICUBIC` -- but PIL's resample 2 is
    BILINEAR, so those measurements, including Section 4.4's, are bilinear. The TRAINING
    path is not: `A trl/grpo_vlm_qwen3.py`, `A build_grpo_sets.py` and
    `A precompute_question_boxes.py` all pass `Image.BICUBIC` symbolically, so every image
    the published runs trained on was resized bicubically.

    This function is the probes' filter, and it is DELIBERATELY not wired into
    `training/grpo/trl_patch/grpo_vlm_qwen3.py`, which keeps its own bicubic resize.
    Adopting it there would change the pixels of every training image, hence every patch
    grid's contents, every map and every phi. Which of the two to standardise on is a
    decision about which published numbers move; it is not a cleanup, and it has not been
    made. See docs/provenance.md.

    Resize first, convert second. Converting a palette-mode image to RGB before resizing
    is a different operation from resizing in palette space and converting after --
    interpolating palette INDICES is not interpolating colours -- and the two give
    different pixels.
    """
    from PIL import Image

    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                             Image.BILINEAR)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image
