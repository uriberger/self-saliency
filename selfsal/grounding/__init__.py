# Copyright 2026 NVIDIA. Apache-2.0.
"""Localising the objects a reasoning step mentions (Section 3.3)."""

from .mask import (box_area, center_rect_mask, raster_union, ring_fraction, union_mask)

__all__ = ["box_area", "center_rect_mask", "raster_union", "ring_fraction", "union_mask"]
