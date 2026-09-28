# Copyright 2026 NVIDIA. Apache-2.0.
"""Attention -> a saliency map over image patches (Section 3.3).

For a generated token `t` and an image patch `p`, `sal_t(p)` is the attention weight `t`
assigns to the visual token corresponding to `p`. This module records exactly that,
during generation, without changing what the model does.

    from selfsal.saliency import AttentionCollector, REWARD_LAYER, REWARD_HEADS

    with AttentionCollector(model, layer=REWARD_LAYER, heads=REWARD_HEADS) as c:
        model.generate(**inputs, max_new_tokens=256)
    patch_map = c.collected_map()            # (grid_h, grid_w), or None

HOW IT WORKS, AND WHY IT IS NOT A HOOK ON THE MODULE. transformers dispatches attention
through `config._attn_implementation` into `ALL_ATTENTION_FUNCTIONS`, so an
implementation registered under a name and selected by that config is the only place the
weights exist as a tensor. A module forward hook sees the OUTPUT, by which point the
weights have been contracted with the values and are gone; asking for
`output_attentions=True` instead forces every layer off the fused kernel for the whole
generation. This takes the explicit path at ONE layer and leaves the other 35 on SDPA.

PROMPT POSITIONS ARE SKIPPED. Only tokens the model wrote are recorded. A map averaged
over the question's own tokens is a different statistic wearing the same name.

N generated tokens give N-1 rows: the last token is never a query, because nothing
follows it. That is causal generation, not a dropped row.

PROVENANCE. Extracted from the archive's `sink_shift.py`, which is an attention-EDIT
framework that the Section 5 measurement was using purely as a collector -- it installed
it with `alpha=0.0`, i.e. with the edit disabled. The edit, its arms and its mass
matching are not part of this paper and stayed behind; what is here is the capture path
and nothing else.
"""

from __future__ import annotations

import torch

#: Qwen3-VL's <|image_pad|>. The visual tokens are the runs of this id in the prompt.
IMAGE_TOKEN_ID = 151655

#: The name this implementation registers under in ALL_ATTENTION_FUNCTIONS.
IMPL_NAME = "selfsal_collect"


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Grouped-query expansion: [b, kv_heads, s, d] -> [b, kv_heads * n_rep, s, d]."""
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


def sdpa(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kw):
    """The stock fused path, for every layer this collector is not reading."""
    n_rep = getattr(module, "num_key_value_groups", 1)
    k, v = repeat_kv(key, n_rep), repeat_kv(value, n_rep)
    if attention_mask is not None:
        attention_mask = attention_mask[..., :k.shape[2]]
    causal = (query.shape[2] > 1 and attention_mask is None
              and (is_causal if is_causal is not None
                   else getattr(module, "is_causal", True)))
    out = torch.nn.functional.scaled_dot_product_attention(
        query, k, v, attn_mask=attention_mask, dropout_p=dropout, scale=scaling,
        is_causal=bool(causal))
    return out.transpose(1, 2).contiguous(), None


def attention_weights(module, query, key, value, attention_mask, scaling):
    """The explicit softmax, for the one layer being read. -> [b, heads, q, kv]."""
    n_rep = getattr(module, "num_key_value_groups", 1)
    k, v = repeat_kv(key, n_rep), repeat_kv(value, n_rep)
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    logits = torch.matmul(query, k.transpose(2, 3)) * scale
    if attention_mask is not None:
        logits = logits + attention_mask[..., :k.shape[2]]
    weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(weights, v).transpose(1, 2).contiguous()
    return out, weights


class AttentionCollector:
    """Record image attention at one layer, over the generated tokens.

    `heads=None` merges every head in the layer. The paper reads two different things
    off this and keeps them apart: the rewarded pair (22, 28+31), which is what the
    reward saw, and the whole of layer 22, which is where Section 4.4 finds the
    grounded-region attention actually moving.
    """

    def __init__(self, model, layer: int, heads=None, image_token_id: int = IMAGE_TOKEN_ID):
        self.model = model
        self.layer = int(layer)
        self.heads = None if heads is None else sorted(int(h) for h in heads)
        self.image_token_id = int(image_token_id)

        self._maps: list[tuple[int, torch.Tensor]] = []
        self.img_cols = None      # absolute key positions of the visual tokens
        self.grids: list[tuple[int, int, int]] = []
        self.prompt_len = 0

        self._handles: list = []
        self._text_cfg = None
        self._prev_impl = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()
        return False

    def install(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS.register(IMPL_NAME, self._make_attention())
        cfg = self._find_text_config()
        self._text_cfg = cfg
        self._prev_impl = cfg._attn_implementation
        cfg._attn_implementation = IMPL_NAME
        # The per-module copies matter: transformers caches the choice on each attention
        # module's own config, so setting only the top-level one leaves every layer on
        # the previous implementation and the collector silently records nothing.
        for m in self.model.modules():
            if type(m).__name__ == "Qwen3VLTextAttention":
                m.config._attn_implementation = IMPL_NAME
        self._handles.append(
            self.model.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
        return self

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._text_cfg is not None and self._prev_impl is not None:
            self._text_cfg._attn_implementation = self._prev_impl
            for m in self.model.modules():
                if type(m).__name__ == "Qwen3VLTextAttention":
                    m.config._attn_implementation = self._prev_impl
        self._prev_impl = None

    def _find_text_config(self):
        cfg = getattr(self.model, "config", None)
        return getattr(cfg, "text_config", None) or cfg

    def reset(self):
        self._maps = []
        return self

    # -- locating the picture ---------------------------------------------

    def _pre_hook(self, _module, args, kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        grid = kwargs.get("image_grid_thw")
        if input_ids is not None and input_ids.dim() == 2:
            self._locate_images(input_ids, grid)
        return None

    def _locate_images(self, input_ids, image_grid_thw):
        """Absolute key positions of every visual token, and each picture's patch grid.

        Runs of the image token are matched to rows of `image_grid_thw` in order, so a
        prompt carrying two pictures of different shapes gets two correctly shaped grids
        rather than one averaged wrong one. A count mismatch is refused rather than
        guessed at.
        """
        ids = input_ids[0]
        is_img = ids == self.image_token_id
        if not bool(is_img.any()):
            return False
        pos = torch.nonzero(is_img, as_tuple=True)[0]
        brk = torch.nonzero(pos[1:] - pos[:-1] != 1, as_tuple=True)[0]
        starts = [0] + (brk + 1).tolist()
        ends = (brk + 1).tolist() + [pos.numel()]
        runs = [pos[a:b] for a, b in zip(starts, ends)]

        n_grids = 0 if image_grid_thw is None else len(image_grid_thw)
        if n_grids != len(runs):
            raise RuntimeError(
                f"{len(runs)} image token runs but {n_grids} grids: refusing to guess "
                "which picture is which")

        cols, grids = [], []
        for run, thw in zip(runs, image_grid_thw):
            t, h, w = (int(x) for x in thw)
            gh, gw = h // 2, w // 2        # Qwen3-VL merges 2x2 patches into one token
            if run.numel() != t * gh * gw:
                raise RuntimeError(
                    f"image run of {run.numel()} tokens against a {t}x{gh}x{gw} grid: "
                    "the patch-merge assumption is wrong for this model")
            cols.append(run)
            grids.append((t, gh, gw))

        self.img_cols = torch.cat(cols).to(input_ids.device)
        self.grids = grids
        self.prompt_len = int(input_ids.shape[1])
        # A forward carrying image tokens starts a generation, so anything kept from the
        # previous one belongs to a different picture.
        self._maps = []
        return True

    # -- the attention implementation --------------------------------------

    def _make_attention(self):
        collector = self

        def forward(module, query, key, value, attention_mask,
                    dropout=0.0, scaling=None, is_causal=None, **kwargs):
            layer_idx = getattr(module, "layer_idx", -1)
            if collector.img_cols is None or layer_idx != collector.layer:
                return sdpa(module, query, key, value, attention_mask,
                            dropout, scaling, is_causal, **kwargs)
            out, weights = attention_weights(module, query, key, value,
                                             attention_mask, scaling)
            collector._record(weights, query.shape[2], key.shape[2])
            return out, None

        return forward

    def _record(self, weights, n_query: int, n_key: int):
        """Keep one image-attention row per generated token, merged over the heads."""
        q_start = n_key - n_query           # absolute position of this chunk's first row
        rows = torch.arange(n_query, device=weights.device)
        pos = rows + q_start
        keep = pos >= self.prompt_len
        if not bool(keep.any()):
            return
        rows, pos = rows[keep], pos[keep]
        heads = (slice(None) if self.heads is None
                 else torch.tensor(self.heads, device=weights.device, dtype=torch.long))
        w = weights[:, heads][:, :, rows, :][..., self.img_cols]    # [1, H', R, n_img]
        merged = w.mean(dim=1)[0].detach().float().cpu()            # [R, n_img]
        for i, p in enumerate(pos.tolist()):
            self._maps.append((int(p), merged[i]))

    # -- the result --------------------------------------------------------

    def collected_map(self):
        """Mean over the generated tokens' attention rows, on the picture's patch grid.

        None when nothing was collected, or when the prompt carried more than one
        picture -- a single map has one shape, and silently concatenating two grids would
        be a map of nothing.
        """
        if not self._maps or len(self.grids) != 1:
            return None
        t, gh, gw = self.grids[0]
        if t != 1:
            return None
        rows = torch.stack([m for _p, m in self._maps])             # [n_tok, n_img]
        return rows.mean(0).clamp_min(0).reshape(gh, gw).numpy()

    def token_maps(self):
        """[(absolute position, map over patches)], one per recorded generated token.

        What `saliency.score` needs: phi is computed per STEP, so the caller reduces the
        rows of that step's token span rather than the whole completion's.
        """
        if not self.grids or len(self.grids) != 1:
            return []
        _t, gh, gw = self.grids[0]
        return [(p, m.clamp_min(0).reshape(gh, gw).numpy()) for p, m in self._maps]
