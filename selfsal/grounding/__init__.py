# Copyright 2026 NVIDIA. Apache-2.0.
"""Localising the objects a reasoning step mentions (Section 3.3).

`dino` pulls in transformers and is imported lazily, so the mask helpers stay usable in
a numpy-only environment.
"""

from .mask import box_area, center_rect_mask, centroid_eccentricity, raster_union, ring_fraction, union_mask

__all__ = [
    "box_area", "center_rect_mask", "centroid_eccentricity", "raster_union", "ring_fraction", "union_mask",
    "ground", "ground_local", "ground_served", "ground_scored", "ground_claim",
    "GROUNDING_DINO_HF_ID",
    "DEFAULT_BOX_THRESHOLD",
]


def __getattr__(name):
    if name in ("ground", "ground_local", "ground_served", "ground_scored",
                "ground_claim", "load_local",
                "GROUNDING_DINO_HF_ID", "DEFAULT_BOX_THRESHOLD"):
        from . import dino
        return getattr(dino, name)
    raise AttributeError(name)
