# Copyright 2026 NVIDIA. Apache-2.0.
"""The attention collector's numerical core, against the archive and against itself.

`selfsal/saliency/maps.py` was extracted from an attention-EDIT framework that the
Section 5 measurement used purely as a collector (it installed it with `alpha=0.0`). The
edit stayed behind; this checks that the capture path came across unchanged.

The hook wiring needs a real Qwen3-VL to exercise and is not covered here. What IS
covered is everything numerical: the grouped-query expansion, the fused path taken by
the 35 layers that are not being read, and the explicit softmax taken by the one that
is -- which must reproduce the fused path's output, or the collector would be changing
the model's behaviour while claiming to observe it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ARCHIVE = Path(os.environ.get(
    "SELFSAL_ARCHIVE", Path.home() / "scratch/research/saliency_r1"))


class _Module:
    """The handful of attributes the attention functions read off a real one."""
    num_key_value_groups = 4
    is_causal = True


def _archive():
    path = ARCHIVE / "sink_shift.py"
    if not path.exists():
        pytest.skip(f"archive not present at {ARCHIVE}")
    spec = importlib.util.spec_from_file_location("_selfsal_sink_shift", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_selfsal_sink_shift"] = mod
    spec.loader.exec_module(mod)
    return mod


_SHAPES = [(1, 8, 1, 40, 64),      # one query: the decode step
           (1, 8, 5, 40, 64),
           (1, 2, 3, 7, 16),       # a tiny head count and a short context
           (1, 8, 12, 60, 64)]     # a prefill chunk


def _tensors(shape, seed):
    b, kv_heads, q, kv, d = shape
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, kv_heads * _Module.num_key_value_groups, q, d, generator=g),
            torch.randn(b, kv_heads, kv, d, generator=g),
            torch.randn(b, kv_heads, kv, d, generator=g))


def test_image_token_id_matches_archive():
    from selfsal.saliency.maps import IMAGE_TOKEN_ID
    assert IMAGE_TOKEN_ID == _archive().IMAGE_TOKEN_ID


@pytest.mark.parametrize("n_rep", [1, 2, 4])
@pytest.mark.parametrize("shape", _SHAPES)
def test_repeat_kv_matches_archive(shape, n_rep):
    from selfsal.saliency.maps import repeat_kv
    _q, key, _v = _tensors(shape, 0)
    assert torch.equal(repeat_kv(key, n_rep), _archive()._repeat_kv(key, n_rep))


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("masked", [False, True])
def test_sdpa_matches_archive(shape, masked):
    from selfsal.saliency.maps import sdpa
    query, key, value = _tensors(shape, 1)
    mask = torch.zeros(shape[0], 1, shape[2], shape[3]) if masked else None

    got, got_w = sdpa(_Module(), query, key, value, mask, 0.0, None, None)
    want, want_w = _archive()._sdpa(_Module(), query, key, value, mask, 0.0, None, None)

    assert torch.allclose(got, want, atol=1e-6)
    assert got_w is None and want_w is None      # the fused path returns no weights


@pytest.mark.parametrize("shape", _SHAPES)
def test_explicit_softmax_reproduces_the_fused_output(shape):
    """Observing must not change the model.

    The collected layer leaves the fused kernel so the weights exist as a tensor. If its
    output drifted from SDPA's, every map would describe a model the paper never
    evaluated.
    """
    from selfsal.saliency.maps import attention_weights, sdpa
    query, key, value = _tensors(shape, 2)
    mask = torch.zeros(shape[0], 1, shape[2], shape[3])   # explicit, so both agree

    fused, _ = sdpa(_Module(), query, key, value, mask, 0.0, None, None)
    explicit, weights = attention_weights(_Module(), query, key, value, mask, None)

    assert torch.allclose(explicit, fused, atol=1e-5)
    rows = weights.sum(-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5)


def test_collected_map_refuses_multi_picture_prompts():
    """One map has one shape; two grids concatenated would be a map of nothing."""
    from selfsal.saliency.maps import AttentionCollector

    c = AttentionCollector.__new__(AttentionCollector)
    c._maps = [(10, torch.rand(6))]
    c.grids = [(1, 2, 3), (1, 2, 3)]
    assert c.collected_map() is None
    assert c.token_maps() == []

    c.grids = [(1, 2, 3)]
    got = c.collected_map()
    assert got is not None and got.shape == (2, 3)


def test_collected_map_is_empty_before_generation():
    from selfsal.saliency.maps import AttentionCollector

    c = AttentionCollector.__new__(AttentionCollector)
    c._maps, c.grids = [], [(1, 2, 3)]
    assert c.collected_map() is None
