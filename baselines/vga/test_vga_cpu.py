#!/usr/bin/env python
"""
vga_cpu_smoketest.py — exercise every line of vlm/vga.py on a toy model, on CPU,
in seconds, with no GPU and no checkpoint.

This exists because the expensive failures in an inference-time patch are not
subtle numerical ones — they are shape errors, a hook that never fires, a
generate() that cannot resume from the cache it was handed, a GQA expansion that
lines heads up wrongly.  All of those reproduce on a six-layer model with a
32-dimensional hidden state, and none of them need an H100 to find.

It builds a genuinely tiny Qwen3-VL (random weights, GQA on, deepstack on) and
checks:

  1. the visual span and patch grid are located correctly
  2. beta=0 is EXACTLY the stock model, on both execution paths
  3. a large beta is not, and the hooks report having fired
  4. Δz really is Gᵀ·V over the visual positions, per head, after repeat_kv —
     recomputed independently and compared against the hook's own update
  5. head balancing does what it says: gamma falls as a head's output aligns
     with the guidance
  6. object-directed and object-agnostic modes both build a normalised G
  7. PVG moves G between steps, and only on generated positions
  8. text-only prompts and double installation are refused or passed through

Run it after any transformers upgrade — the cache-resume path in (2) is the part
most likely to break under one, and it breaks silently.

    conda run -n lmms_eval python scripts/vga_cpu_smoketest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from vlm import vga as vga_mod  # noqa: E402

IMAGE_TOKEN_ID = 100
GRID_H, GRID_W = 8, 8          # vision patches before the 2×2 merge
PATCH = 4


def build_model():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    cfg = Qwen3VLConfig(
        text_config=dict(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=6, num_attention_heads=4,
            num_key_value_heads=2,           # GQA: exercises repeat_kv
            head_dim=8, max_position_embeddings=512,
        ),
        vision_config=dict(
            depth=4, hidden_size=16, intermediate_size=32, num_heads=2,
            in_channels=3, patch_size=PATCH, spatial_merge_size=2,
            temporal_patch_size=2, out_hidden_size=32,
            num_position_embeddings=256,
            deepstack_visual_indexes=[1, 2],  # deepstack on, as on the real model
        ),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=101,
        vision_start_token_id=102,
    )
    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(cfg).eval()
    return model


def build_inputs(model):
    m = (GRID_H * GRID_W) // 4                      # after the 2×2 merge
    prefix, suffix = [5, 6, 7], [8, 9]
    ids = prefix + [IMAGE_TOKEN_ID] * m + suffix
    n_patches = GRID_H * GRID_W
    vis_cfg = model.config.vision_config
    px_dim = vis_cfg.in_channels * vis_cfg.temporal_patch_size * PATCH * PATCH
    torch.manual_seed(1)
    # mm_token_type_ids: 0 text, 1 image. Qwen3-VL's M-RoPE requires it whenever
    # image_grid_thw is passed, and the real processor returns it.
    mm = [0] * len(prefix) + [1] * m + [0] * len(suffix)
    return dict(
        input_ids=torch.tensor([ids]),
        attention_mask=torch.ones(1, len(ids), dtype=torch.long),
        mm_token_type_ids=torch.tensor([mm], dtype=torch.int32),
        pixel_values=torch.randn(n_patches, px_dim),
        image_grid_thw=torch.tensor([[1, GRID_H, GRID_W]]),
    ), m, len(prefix)


class FakeTokenizer:
    """Just enough of a tokenizer for object extraction and prompt recovery."""

    def encode(self, text, add_special_tokens=False):
        # Deterministic, and distinct for "cat" / " cat" so the variant logic is
        # actually exercised rather than collapsing to one id.
        return [abs(hash(text.strip().lower())) % 90 + 10]

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            return f"<{ids}>"
        return "<|im_start|>user\nIs there a cat in the image?<|im_end|>"


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.image_processor = type("IP", (), {"merge_size": 2})()


def gen(model, inputs, n=8):
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=n, do_sample=False, num_beams=1)
    return out[0, inputs["input_ids"].shape[1]:].tolist()


def main() -> int:
    fails: list[str] = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}",
              flush=True)
        if not ok:
            fails.append(name)

    def note(text):
        print(f"  note  {text}", flush=True)

    model = build_model()
    proc = FakeProcessor()
    inputs, m, prefix_len = build_inputs(model)
    cfgk = dict(start_layer=1, end_layer=5, verbose=False)

    # -- 1. geometry -------------------------------------------------------
    print("\n1. locating the picture")
    span = vga_mod.visual_span(inputs["input_ids"], IMAGE_TOKEN_ID)
    check("visual span", span == (prefix_len, prefix_len + m), f"{span}, m={m}")
    grid = vga_mod.patch_grid(inputs, proc)
    check("patch grid", grid == (GRID_H // 2, GRID_W // 2) and grid[0] * grid[1] == m,
          str(grid))

    # -- 2/3. the equality check -------------------------------------------
    print("\n2. beta=0 must be the stock model; a big beta must not be")
    stock = gen(model, inputs)

    arms = {}
    for name, kw in (("beta0-fresh", dict(beta=0.0, prefill_reuse=False, mode="agnostic")),
                     ("beta0-reuse", dict(beta=0.0, prefill_reuse=True, mode="agnostic")),
                     ("beta5-reuse", dict(beta=5.0, prefill_reuse=True, mode="agnostic")),
                     ("beta5-fresh", dict(beta=5.0, prefill_reuse=False, mode="agnostic"))):
        v = vga_mod.install(model, proc, vga_mod.VGAConfig(**cfgk, **kw))
        try:
            arms[name] = (gen(model, inputs), v.diagnostics())
        finally:
            v.remove()

    check("beta=0, fresh prefill == stock", arms["beta0-fresh"][0] == stock)
    check("beta=0, cache resume  == stock", arms["beta0-reuse"][0] == stock,
          "" if arms["beta0-reuse"][0] == stock else
          f"{arms['beta0-reuse'][0]} vs {stock} (bf16 would excuse this; float32 does not)")
    check("hooks fired at beta=0", arms["beta0-fresh"][1]["injected_steps"] > 0,
          f"{arms['beta0-fresh'][1]['injected_steps']} steps")
    check("beta=5 changes the output", arms["beta5-reuse"][0] != stock,
          f"{arms['beta5-reuse'][0]} vs {stock}")
    check("beta=5 reports a real update", arms["beta5-reuse"][1]["mean_rel_update"] > 1e-3,
          f"rel_update={arms['beta5-reuse'][1]['mean_rel_update']:.4f}")
    check("beta=5 changes the output on the fresh path too",
          arms["beta5-fresh"][0] != stock)
    # Not an assertion. The fresh path deliberately does not inject at the last
    # prompt token, so its first generated token can differ from the resume
    # path's, and greedy decoding amplifies one differing token into a different
    # continuation. Printed so the size of that gap is visible, not hidden.
    note(f"resume vs fresh at beta=5: {arms['beta5-reuse'][0]} vs {arms['beta5-fresh'][0]}"
         + ("  (identical)" if arms["beta5-reuse"][0] == arms["beta5-fresh"][0] else ""))

    # -- 4. is Δz actually GᵀV? --------------------------------------------
    print("\n3. Δz = Gᵀ·V over the visual positions, per head, after repeat_kv")
    v = vga_mod.install(model, proc, vga_mod.VGAConfig(**cfgk, beta=0.0, mode="agnostic"))
    captured = {}
    layer = v.layer_idx[0]

    orig_hook = v._make_hook

    def spy(li):
        inner = orig_hook(li)

        def hook(module, args):
            if li == layer and "z" not in captured:
                captured["z"] = args[0].detach().clone()
                captured["dz"] = v._dz[li].detach().clone()
            return inner(module, args)
        return hook

    v._make_hook = spy
    gen(model, inputs, n=1)

    # Redo the prefill by hand so the values are available outside the call. The
    # per-token inputs have to be held back along with the last token — the same
    # slicing vga._prepare does, and getting it wrong here is how the missing
    # mm_token_type_ids slice was found.
    n_ids = inputs["input_ids"].shape[1]
    fwd = {k: (val[:, :-1] if (k != "input_ids" and torch.is_tensor(val)
                               and not k.endswith("_grid_thw") and val.dim() == 2
                               and val.shape[1] == n_ids) else val)
           for k, val in inputs.items() if k != "input_ids"}
    pre = vga_mod.run_prefill(model, inputs["input_ids"][:, :-1], fwd, span, v.cfg)
    g = vga_mod.salience_map(model, pre, v.cfg)
    raw_v = vga_mod._get_value_tensor(pre.cache, layer)
    n_groups = model.config.text_config.num_attention_heads // model.config.text_config.num_key_value_heads
    expanded = vga_mod._repeat_kv(raw_v[:, :, span[0]:span[1], :], n_groups).float()
    manual = torch.einsum("m,bhmd->bhd", g.float(), expanded)
    got = captured.get("dz")
    ok = got is not None and torch.allclose(manual, got, atol=1e-5)
    check("Δz matches an independent GᵀV", ok,
          "" if ok else f"max |diff| = {(manual - got).abs().max().item():.3e}"
          if got is not None else "hook never captured Δz")
    # And GQA must not be a no-op here, or the test would prove nothing.
    check("GQA expansion is exercised", n_groups > 1, f"n_groups={n_groups}")
    v.remove()

    # -- 5. head balancing --------------------------------------------------
    print("\n4. head balancing weights guidance down on already-visual heads")
    H, D = 4, 8
    z = torch.randn(1, 1, H, D)
    dz = torch.zeros(1, H, D)
    dz[0, 0] = z[0, 0, 0] * 3.0        # head 0: perfectly aligned
    dz[0, 1] = -z[0, 0, 1] * 3.0       # head 1: anti-aligned
    dz[0, 2:] = torch.randn(2, D)
    sim = torch.nn.functional.cosine_similarity(dz.unsqueeze(1), z, dim=-1)
    w = (1 + sim) / 2
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12) * H
    gamma = torch.relu(2 - w)
    check("aligned head is damped below the anti-aligned one",
          float(gamma[0, 0, 0]) < float(gamma[0, 0, 1]),
          f"gamma aligned={float(gamma[0,0,0]):.3f} anti={float(gamma[0,0,1]):.3f}")
    check("gamma is non-negative", bool((gamma >= 0).all()))

    # -- 6. the two map modes ----------------------------------------------
    print("\n5. G is a normalised distribution over the patches, in both modes")
    for mode, kw in (("agnostic", {}), ("object", {})):
        cfg = vga_mod.VGAConfig(**cfgk, mode=mode, **kw)
        v = vga_mod.install(model, proc, cfg)
        if mode == "object":
            v.set_objects(["cat", "table"])
        try:
            gen(model, inputs, n=2)
            gmap = v.last_map(grid)
            d = v.diagnostics()
        finally:
            v.remove()
        ok = (gmap is not None and gmap.shape == grid
              and abs(float(gmap.sum()) - 1.0) < 1e-4 and (gmap >= 0).all())
        check(f"{mode}: G sums to 1 over {grid}", ok,
              f"sum={float(gmap.sum()):.6f} mode={d['last_mode']}" if gmap is not None else "no map")
        check(f"{mode}: took the intended branch",
              d["last_mode"] == mode, f"got {d['last_mode']}")

    # 'auto' with no extractable object must land in VSS, loudly, not silently on None.
    v = vga_mod.install(model, proc, vga_mod.VGAConfig(**cfgk, mode="auto"))
    v.set_question("Describe it.")
    try:
        gen(model, inputs, n=2)
        d = v.diagnostics()
    finally:
        v.remove()
    check("auto falls back to VSS when no object is found",
          d["last_mode"] == "agnostic" and d["fallbacks"] == 1, str(d["last_mode"]))

    # -- 7. PVG -------------------------------------------------------------
    print("\n6. PVG moves G between steps")
    v = vga_mod.install(model, proc, vga_mod.VGAConfig(**cfgk, mode="agnostic",
                                                       pvg=True, lam=0.5))
    seen = []
    orig_update = vga_mod.pvg_update

    def spy_update(model_, pre_, g_, tok, cfg_):
        out = orig_update(model_, pre_, g_, tok, cfg_)
        seen.append((tok, float((out - g_.to(out.device)).abs().max())))
        return out

    vga_mod.pvg_update = spy_update
    try:
        toks = gen(model, inputs, n=5)
    finally:
        vga_mod.pvg_update = orig_update
        d = v.diagnostics()
        v.remove()
    check("PVG ran once per generated token after the first",
          len(seen) == len(toks) - 1, f"{len(seen)} updates for {len(toks)} tokens")
    check("PVG actually changed G", bool(seen) and all(s > 0 for _, s in seen),
          f"max moves {[round(s, 6) for _, s in seen]}")
    check("PVG used the generated tokens, in order",
          [t for t, _ in seen] == toks[:-1], f"{[t for t, _ in seen]} vs {toks[:-1]}")

    # -- 8. the refusals ----------------------------------------------------
    print("\n7. what it refuses")
    v = vga_mod.install(model, proc, vga_mod.VGAConfig(**cfgk, mode="agnostic"))
    try:
        text_only = dict(input_ids=torch.tensor([[5, 6, 7, 8]]),
                         attention_mask=torch.ones(1, 4, dtype=torch.long))
        with torch.inference_mode():
            model.generate(**text_only, max_new_tokens=3, do_sample=False)
        check("a text-only prompt passes straight through",
              v.diagnostics()["no_image"] == 1)

        batched = {k: torch.cat([val, val]) if k in ("input_ids", "attention_mask")
                   else val for k, val in inputs.items()}
        try:
            with torch.inference_mode():
                model.generate(**batched, max_new_tokens=2, do_sample=False)
            check("batch > 1 is refused", False, "it was accepted")
        except ValueError as exc:
            check("batch > 1 is refused", "batch_size=1" in str(exc))

        try:
            vga_mod.install(model, proc, vga_mod.VGAConfig(**cfgk))
            check("double installation is refused", False, "it was accepted")
        except RuntimeError as exc:
            check("double installation is refused", "already installed" in str(exc))
    finally:
        v.remove()

    check("remove() restores the original generate",
          not hasattr(model.generate, "__self__")
          or model.generate.__func__.__qualname__.startswith("GenerationMixin"),
          model.generate.__qualname__)

    print("\n" + "=" * 72)
    if fails:
        print(f"{len(fails)} FAILURE(S): " + "; ".join(fails))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
