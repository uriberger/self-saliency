# Copyright 2026 NVIDIA. Apache-2.0.
# Ported from the archive's vlm/saliency.py. `models/families.py` is the other half of
# this: what differs BETWEEN model families for the Section 5 analysis, where this is
# what can be read off any one of them generically -- layer and head counts, the image
# token, the KV cache, and the decode-time saliency capture the head-selection screen
# runs over Visual-CoT.
"""
saliency.py — Per-token attention saliency map extraction for VLMs.

For each generated response token, produces a spatial saliency map showing
which image regions that token attended to, weighted by value-state norms.

Algorithm
---------
Single forward pass over the full [prompt + response] sequence with eager
attention.  A hook on each LM self-attention layer captures the attention
sub-matrix from response tokens to image-patch tokens.  The captured weights
are then multiplied by the L2 norm of the corresponding GQA-expanded value
states and summed across layers and heads, giving one saliency map per token.

Maps are returned at the model's native patch-grid resolution (grid_h × grid_w)
rather than the original image resolution.  Upsample to image dimensions at
visualization time.

Requirements
------------
The model must be loaded with ``attn_implementation="eager"``.

Supported model families
------------------------
* Qwen2.5-VL  (Qwen/Qwen2.5-VL-*)
* Qwen3-VL    (Qwen/Qwen3-VL-*)

Extend ``_get_model_config`` for other architectures.

Usage
-----
    from selfsal.models.introspect import extract_saliency

    saliency, tokens = extract_saliency(model, processor, messages, response)
    # saliency : float32 ndarray of shape (n_tokens, grid_h, grid_w), values in [0, 1]
    # tokens   : list of str, length n_tokens — decoded text of each response token
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _get_value_tensor(past_key_values, layer_idx: int):
    """Extract the value tensor for *layer_idx* from any HF cache format.

    Handles three generations of transformers cache APIs:
      - tuple-of-tuples       (old):  pkv[i][1]
      - DynamicCache v1       (mid):  pkv.value_cache[i]
      - Cache w/ layers list  (new):  pkv.layers[i].values
    """
    if hasattr(past_key_values, "layers"):
        return past_key_values.layers[layer_idx].values
    if hasattr(past_key_values, "value_cache"):
        return past_key_values.value_cache[layer_idx]
    return past_key_values[layer_idx][1]


# ---------------------------------------------------------------------------
# Model-specific helpers
# ---------------------------------------------------------------------------

def _get_model_config(model) -> dict:
    """
    Return backbone components for supported model families.

    Keys: lm, layers (the ModuleList of decoder layers), num_groups (GQA expansion
    factor = n_heads / n_kv_heads).
    """
    # Qwen3.5-VL: model.model.language_model (one extra nesting level)
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        tcfg = getattr(model.config, "text_config", model.config)
        n_h = tcfg.num_attention_heads
        n_kv = tcfg.num_key_value_heads
        if hasattr(lm, "layers"):
            layers = lm.layers
        elif hasattr(lm, "blocks"):
            layers = lm.blocks
        else:
            raise ValueError(
                f"Cannot find decoder layer list on {type(lm).__name__}. "
                "Expected .layers or .blocks. Add support in _get_model_config."
            )
        return dict(lm=lm, layers=layers, num_groups=n_h // n_kv)
    if hasattr(model, "language_model"):
        lm = model.language_model
        tcfg = getattr(model.config, "text_config", model.config)
        n_h = tcfg.num_attention_heads
        n_kv = tcfg.num_key_value_heads
        # Qwen uses .layers; Molmo2 uses .blocks
        if hasattr(lm, "layers"):
            layers = lm.layers
        elif hasattr(lm, "blocks"):
            layers = lm.blocks
        else:
            raise ValueError(
                f"Cannot find decoder layer list on {type(lm).__name__}. "
                "Expected .layers or .blocks. Add support in _get_model_config."
            )
        return dict(lm=lm, layers=layers, num_groups=n_h // n_kv)
    raise ValueError(
        f"Cannot locate language_model on {type(model).__name__}. "
        "Add support in saliency.py:_get_model_config."
    )


def _get_image_token_id(processor, model=None) -> int:
    """Auto-detect the image-patch token ID from the processor's tokenizer."""
    tok = getattr(processor, "tokenizer", processor)
    # Molmo2 inherits the Qwen2.5 vocab (so <|image_pad|> resolves to a valid ID)
    # but uses <im_patch> for actual image patches.  Detect by model_type.
    if model is not None:
        model_type = getattr(getattr(model, "config", None), "model_type", "")
        if "molmo" in model_type.lower():
            tid = tok.convert_tokens_to_ids("<im_patch>")
            if tid not in (None, tok.unk_token_id):
                return tid
    for candidate in ("<|image_pad|>", "<IMG_CONTEXT>", "<image>", "<|vision_start|>"):
        tid = tok.convert_tokens_to_ids(candidate)
        if tid not in (None, tok.unk_token_id):
            return tid
    raise ValueError(
        "Cannot determine image token ID automatically. "
        "Pass image_token_id= explicitly to extract_saliency()."
    )


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_saliency(
    model,
    processor,
    messages: list,
    response: str,
    image_token_id: int | None = None,
    value_weighting: bool = True,
    head_reduction: str = "mean",
    layer_reduction: str = "sum",
) -> tuple[np.ndarray, list[str]]:
    """
    Compute per-token spatial saliency maps for a VLM response.

    Parameters
    ----------
    model            : VLM loaded with attn_implementation="eager".
    processor        : corresponding AutoProcessor.
    messages         : chat messages used for the original inference call
                       (user turn only; no assistant turn).
    response         : the model's full generated response string.
    image_token_id   : override the auto-detected image-patch token ID.
    value_weighting  : if True (default), multiply attention weights by the L2 norm
                       of the corresponding value states before reducing.
    head_reduction   : how to reduce across attention heads — "mean" (default), "min", "max".
    layer_reduction  : how to reduce across transformer layers — "sum" (default), "min", "max", "last".

    Returns
    -------
    saliency : float32 ndarray of shape (n_tokens, grid_h, grid_w), values in [0, 1].
               One map per generated response token at the model's native patch-grid
               resolution.  Upsample to image dimensions at visualization time.
    tokens   : list of str, length n_tokens — decoded text of each response token.
    """
    device = next(model.parameters()).device
    tok    = getattr(processor, "tokenizer", processor)
    cfg    = _get_model_config(model)
    lm, layers, num_groups = cfg["lm"], cfg["layers"], cfg["num_groups"]

    if image_token_id is None:
        image_token_id = _get_image_token_id(processor, model)

    # ------------------------------------------------------------------ #
    # Build full [prompt + response] token sequence                        #
    # ------------------------------------------------------------------ #
    full_messages = messages + [
        {"role": "assistant", "content": [{"type": "text", "text": response}]}
    ]
    inputs = processor.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    ids = inputs["input_ids"][0].tolist()

    # ------------------------------------------------------------------ #
    # Locate image-patch token positions                                   #
    # ------------------------------------------------------------------ #
    image_pos = torch.tensor(
        [i for i, t in enumerate(ids) if t == image_token_id],
        device=device, dtype=torch.long,
    )
    if len(image_pos) == 0:
        raise ValueError(f"No image tokens (id={image_token_id}) found. Check image_token_id.")
    n_image = len(image_pos)

    # ------------------------------------------------------------------ #
    # Determine response token positions                                   #
    # ------------------------------------------------------------------ #
    prompt_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    prompt_len  = prompt_inputs["input_ids"].shape[1]
    total_len   = inputs["input_ids"].shape[1]
    n_response  = total_len - prompt_len

    if n_response <= 0:
        raise ValueError("Response token span is empty.")

    token_texts = [
        tok.decode([ids[p]], skip_special_tokens=False)
        for p in range(prompt_len, total_len)
    ]

    # ------------------------------------------------------------------ #
    # Register hooks to capture attention from response → image tokens     #
    # ------------------------------------------------------------------ #
    # Hybrid models (e.g. Qwen3.5) interleave linear-attention layers that
    # have no self_attn — skip those; only hook full-attention layers.
    hookable = [i for i in range(len(layers)) if hasattr(layers[i], "self_attn")]
    n_layers   = len(hookable)
    layer_attn = [None] * n_layers  # [n_heads, n_response, n_image] per hookable layer

    def _make_hook(hook_idx):
        def _hook(module, inp, out):
            if not (isinstance(out, tuple) and len(out) > 1 and out[1] is not None):
                return
            attn_w = out[1]  # [1, n_heads, S, S]
            # response tokens → image tokens: [n_heads, n_response, n_image]
            a = attn_w[0][:, prompt_len:, :][:, :, image_pos].detach().clone()
            layer_attn[hook_idx] = a
            return (out[0], None) + out[2:]
        return _hook

    handles = [
        layers[hookable[hi]].self_attn.register_forward_hook(_make_hook(hi))
        for hi in range(n_layers)
    ]

    try:
        outputs = model(
            **inputs,
            output_attentions=True,
            use_cache=True,
        )
    finally:
        for h in handles:
            h.remove()

    if any(d is None for d in layer_attn):
        raise RuntimeError(
            "Attention hooks did not capture weights. "
            "Ensure the model is loaded with attn_implementation='eager'."
        )

    # Value states from KV cache (index by original layer index, then re-order)
    kv = outputs.past_key_values
    val_cache = [_get_value_tensor(kv, hookable[hi]) for hi in range(n_layers)]

    # ------------------------------------------------------------------ #
    # Reduce attention across heads and layers                             #
    # ------------------------------------------------------------------ #
    layer_results = []

    for li in range(n_layers):
        a = layer_attn[li].float()  # [n_heads, n_response, n_image]

        if value_weighting:
            vs = val_cache[li][:, :, image_pos, :].float()  # [1, n_kv, n_image, d_head]
            vs = vs.repeat_interleave(num_groups, dim=1)    # [1, n_heads, n_image, d_head]
            vs = vs.squeeze(0)                              # [n_heads, n_image, d_head]
            v_norms = vs.norm(dim=-1)                       # [n_heads, n_image]
            weighted = a * v_norms.unsqueeze(1)             # [n_heads, n_response, n_image]
        else:
            weighted = a

        if head_reduction == "mean":
            head_reduced = weighted.mean(dim=0)
        elif head_reduction == "min":
            head_reduced = weighted.min(dim=0).values
        else:  # max
            head_reduced = weighted.max(dim=0).values

        layer_results.append(head_reduced)  # [n_response, n_image]

    stacked = torch.stack(layer_results, dim=0)  # [n_layers, n_response, n_image]
    if layer_reduction == "sum":
        saliency_acc = stacked.sum(dim=0)
    elif layer_reduction == "min":
        saliency_acc = stacked.min(dim=0).values
    elif layer_reduction == "max":
        saliency_acc = stacked.max(dim=0).values
    else:  # last
        saliency_acc = layer_results[-1]

    saliency_flat = torch.relu(saliency_acc).cpu().numpy()  # [n_response, n_image]

    # ------------------------------------------------------------------ #
    # Reshape flat patch scores → 2D spatial grid                          #
    # ------------------------------------------------------------------ #
    if "image_grid_thw" in inputs:
        grid_thw = inputs["image_grid_thw"][0]
        # Qwen VL models merge 2×2 patches, so effective grid is thw // 2
        grid_h = int(grid_thw[1]) // 2
        grid_w = int(grid_thw[2]) // 2
    elif "image_grids" in inputs:
        # Molmo2: image_grids[0] = [resized_h, resized_w, height, width]
        # input_ids has low-res then high-res <im_patch> tokens; use high-res.
        g = inputs["image_grids"][0]
        lr_count = int(g[0]) * int(g[1])
        saliency_flat = saliency_flat[:, lr_count:]
        grid_h = int(g[2])
        grid_w = int(g[3])
        n_image = grid_h * grid_w
    else:
        side   = int(np.sqrt(n_image))
        grid_h = side
        grid_w = (n_image + side - 1) // side

    expected = grid_h * grid_w
    if saliency_flat.shape[1] != expected:
        if saliency_flat.shape[1] > expected:
            saliency_flat = saliency_flat[:, :expected]
        else:
            pad = expected - saliency_flat.shape[1]
            saliency_flat = np.pad(saliency_flat, ((0, 0), (0, pad)))

    saliency_2d = saliency_flat.reshape(n_response, grid_h, grid_w)

    # Normalize each token's map to [0, 1]
    vmaxs = saliency_2d.max(axis=(1, 2), keepdims=True)
    vmaxs = np.where(vmaxs > 0, vmaxs, 1.0)
    saliency_2d = (saliency_2d / vmaxs).astype(np.float32)

    return saliency_2d, token_texts


# ---------------------------------------------------------------------------
# Decode-time saliency (collect during model.generate() — no extra fwd pass)
# ---------------------------------------------------------------------------

def prepare_decode_saliency(
    model,
    processor,
    inputs: dict,
    image_token_id: int | None = None,
) -> dict:
    """
    Register forward hooks on LM self-attention layers to capture per-token
    attention during model.generate().  Call before generate().

    The vision encoder is NOT hooked and does not need to use eager attention.
    Only the LM layers need attn_implementation="eager".

    Parameters
    ----------
    model          : VLM with LM layers loaded with attn_implementation="eager".
    processor      : corresponding AutoProcessor.
    inputs         : preprocessed inputs dict (output of apply_chat_template).
    image_token_id : override auto-detected image-patch token ID.

    Returns
    -------
    state : opaque dict to pass to finalize_decode_saliency().
    """
    cfg = _get_model_config(model)
    lm, layers, num_groups = cfg["lm"], cfg["layers"], cfg["num_groups"]
    tok = getattr(processor, "tokenizer", processor)

    if image_token_id is None:
        image_token_id = _get_image_token_id(processor, model)

    device = inputs["input_ids"].device
    ids = inputs["input_ids"][0].tolist()
    all_patch_pos = [i for i, t in enumerate(ids) if t == image_token_id]
    if not all_patch_pos:
        raise ValueError(f"No image tokens (id={image_token_id}) found.")

    # Molmo2: input_ids has low-res patches followed by high-res patches.
    # Capture attention to ALL patch positions; record how many are low-res
    # so finalize_decode_saliency can slice to the high-res spatial grid.
    hr_start = 0
    grid_hw = None
    if "image_grids" in inputs:
        g = inputs["image_grids"][0]
        hr_start = int(g[0]) * int(g[1])   # low-res count to skip
        grid_hw = (int(g[2]), int(g[3]))    # (height, width) of high-res grid

    image_pos = torch.tensor(all_patch_pos, device=device, dtype=torch.long)
    prompt_len = inputs["input_ids"].shape[1]
    # Hybrid models (e.g. Qwen3.5) interleave linear-attention layers without self_attn.
    hookable_indices = [i for i in range(len(layers)) if hasattr(layers[i], "self_attn")]
    n_layers = len(hookable_indices)

    # Some models (e.g. Molmo2) don't propagate attn_implementation="eager" to
    # their nested text configs, so self_attn falls back to SDPA which returns
    # None for attention weights — causing hooks to silently skip every step.
    # Patch each layer config to "eager" before generate() and restore after.
    _patched_attn: dict = {}
    for i, layer in enumerate(layers):
        sa_cfg = getattr(getattr(layer, "self_attn", None), "config", None)
        if sa_cfg is None:
            continue
        cur = getattr(sa_cfg, "_attn_implementation", None)
        if cur != "eager":
            sa_cfg._attn_implementation = "eager"
            _patched_attn[i] = cur
    if _patched_attn:
        import sys
        print(
            f"[saliency] patched {len(_patched_attn)} LM layer config(s): "
            f"_attn_implementation {set(_patched_attn.values())!r} → 'eager'",
            file=sys.stderr, flush=True,
        )

    state: dict = {
        "lm": lm,
        "layers": layers,
        "num_groups": num_groups,
        "image_pos": image_pos,
        "prompt_len": prompt_len,
        "n_layers": n_layers,
        "hookable_indices": hookable_indices,
        "n_image": len(all_patch_pos),
        "tok": tok,
        "image_grid_thw": inputs.get("image_grid_thw"),
        "hr_start": hr_start,
        "grid_hw": grid_hw,
        "_patched_attn": _patched_attn,
        "_layer_buf": {},   # hook_idx → [n_heads, n_image] for current step
        "_steps": [],       # list of [[n_heads, n_image] * n_layers] per decode step
        "handles": [],
    }

    def _make_hook(hook_idx):
        def _hook(module, inp, out):
            if not (isinstance(out, tuple) and len(out) > 1 and out[1] is not None):
                return
            attn_w = out[1]  # [1, n_heads, q_len, kv_len]
            if attn_w.shape[2] != 1:
                return  # prefill step — skip
            a = attn_w[0, :, 0, image_pos].detach().float()  # [n_heads, n_image] — stays on GPU
            state["_layer_buf"][hook_idx] = a
            if len(state["_layer_buf"]) == n_layers:
                state["_steps"].append(
                    [state["_layer_buf"][hi] for hi in range(n_layers)]
                )
                state["_layer_buf"] = {}
            return (out[0], None) + out[2:]  # discard attn_w to free memory
        return _hook

    state["handles"] = [
        layers[hookable_indices[hi]].self_attn.register_forward_hook(_make_hook(hi))
        for hi in range(n_layers)
    ]
    return state


def finalize_decode_saliency(
    state: dict,
    gen_sequences: torch.Tensor,
    past_key_values,
    value_weighting: bool = True,
    head_reduction: str = "mean",
    layer_reduction: str = "sum",
) -> tuple[np.ndarray, list[str]]:
    """
    Compute saliency maps from hooks collected during model.generate().

    Removes the registered hooks.  Must be called after model.generate().

    Parameters
    ----------
    state            : dict returned by prepare_decode_saliency().
    gen_sequences    : [1, total_len] tensor — use gen_out.sequences when
                       model.generate() was called with return_dict_in_generate=True.
    past_key_values  : KV cache from generate() (gen_out.past_key_values);
                       used for value-norm weighting.  Pass None to skip weighting.
    value_weighting  : if True (default), multiply attention weights by the L2 norm
                       of the corresponding value states before reducing.
    head_reduction   : how to reduce across attention heads — "mean" (default), "min", "max".
    layer_reduction  : how to reduce across transformer layers — "sum" (default), "min", "max", "last".

    Returns
    -------
    saliency : float32 ndarray of shape (n_tokens, grid_h, grid_w), values in [0, 1].
    tokens   : list of str, one per generated response token.
    """
    for h in state["handles"]:
        h.remove()
    state["handles"] = []

    # Restore any _attn_implementation values patched in prepare_decode_saliency
    for i, orig in state.get("_patched_attn", {}).items():
        sa_cfg = getattr(getattr(state["layers"][i], "self_attn", None), "config", None)
        if sa_cfg is not None:
            sa_cfg._attn_implementation = orig

    num_groups = state["num_groups"]
    image_pos  = state["image_pos"]
    prompt_len = state["prompt_len"]
    n_layers   = state["n_layers"]
    n_image    = state["n_image"]
    tok        = state["tok"]
    steps      = state["_steps"]

    if not steps:
        raise RuntimeError(
            "No decode steps were collected. "
            "Ensure LM layers use attn_implementation='eager'."
        )

    n_response = len(steps)

    # ------------------------------------------------------------------
    # Value norms from KV cache — image positions are fixed after prefill
    # ------------------------------------------------------------------
    if value_weighting and past_key_values is not None:
        hookable_indices = state.get("hookable_indices")
        if hookable_indices is not None:
            val_cache = [_get_value_tensor(past_key_values, hookable_indices[hi]) for hi in range(n_layers)]
        else:
            n_cache = len(past_key_values.layers) if hasattr(past_key_values, "layers") else len(past_key_values)
            val_cache = [_get_value_tensor(past_key_values, i) for i in range(n_cache)]
    else:
        val_cache = None

    # ------------------------------------------------------------------
    # Reduce attention across heads and layers
    # ------------------------------------------------------------------
    layer_results = []

    for li in range(n_layers):
        # [n_response, n_heads, n_image] → [n_heads, n_response, n_image]
        layer_attn = torch.stack([steps[t][li] for t in range(n_response)], dim=0)
        layer_attn = layer_attn.permute(1, 0, 2)

        if val_cache is not None:
            vc = val_cache[li]
            ip = image_pos.to(vc.device)
            vs = vc[:, :, ip, :].float()                      # [1, n_kv, n_image, d]
            vs = vs.repeat_interleave(num_groups, dim=1)       # [1, n_heads, n_image, d]
            vs = vs.squeeze(0)                                 # [n_heads, n_image, d] — stays on GPU
            v_norms = vs.norm(dim=-1)                          # [n_heads, n_image]
            weighted = layer_attn * v_norms.unsqueeze(1)       # [n_heads, n_response, n_image]
        else:
            weighted = layer_attn

        if head_reduction == "mean":
            head_reduced = weighted.mean(dim=0)
        elif head_reduction == "min":
            head_reduced = weighted.min(dim=0).values
        else:  # max
            head_reduced = weighted.max(dim=0).values

        layer_results.append(head_reduced)  # [n_response, n_image]

    stacked = torch.stack(layer_results, dim=0)  # [n_layers, n_response, n_image]
    if layer_reduction == "sum":
        saliency_acc = stacked.sum(dim=0).cpu().numpy()
    elif layer_reduction == "min":
        saliency_acc = stacked.min(dim=0).values.cpu().numpy()
    elif layer_reduction == "max":
        saliency_acc = stacked.max(dim=0).values.cpu().numpy()
    else:  # last
        saliency_acc = layer_results[-1].cpu().numpy()

    saliency_flat = np.maximum(saliency_acc, 0)

    # ------------------------------------------------------------------
    # Decode response token texts
    # ------------------------------------------------------------------
    ids = gen_sequences[0].tolist()
    token_texts = [
        tok.decode([ids[p]], skip_special_tokens=False)
        for p in range(prompt_len, prompt_len + n_response)
    ]

    # ------------------------------------------------------------------
    # Reshape flat patch scores → 2-D spatial grid
    # ------------------------------------------------------------------
    hr_start = state.get("hr_start", 0)
    grid_hw  = state.get("grid_hw")

    if hr_start > 0:
        # Molmo2: discard low-res prefix, keep high-res patches only
        saliency_flat = saliency_flat[:, hr_start:]

    if grid_hw is not None:
        grid_h, grid_w = grid_hw
    else:
        grid_thw = state.get("image_grid_thw")
        if grid_thw is not None:
            g = grid_thw[0]
            grid_h = int(g[1]) // 2
            grid_w = int(g[2]) // 2
        else:
            n_hr = saliency_flat.shape[1]
            side   = int(np.sqrt(n_hr))
            grid_h = side
            grid_w = (n_hr + side - 1) // side

    expected = grid_h * grid_w
    if saliency_flat.shape[1] != expected:
        if saliency_flat.shape[1] > expected:
            saliency_flat = saliency_flat[:, :expected]
        else:
            pad = expected - saliency_flat.shape[1]
            saliency_flat = np.pad(saliency_flat, ((0, 0), (0, pad)))

    saliency_2d = saliency_flat.reshape(n_response, grid_h, grid_w)

    vmaxs = saliency_2d.max(axis=(1, 2), keepdims=True)
    vmaxs = np.where(vmaxs > 0, vmaxs, 1.0)
    return (saliency_2d / vmaxs).astype(np.float32), token_texts
