#!/usr/bin/env python3
"""Side-by-side pictures of five saliency maps, on the model's own observe steps.

Every number in docs/saliency-maps.md is a scalar against a Grounding-DINO union.
This script draws the maps themselves instead: one directory per sample holding the
question, the generated chain, the image the model actually saw, and that image with
each map laid over it as a heatmap. No DINO, no scoring -- looking, not measuring.

Five maps, all reduced over the same tokens (the step's own span), so the only
difference between the five pictures is how a patch's saliency is defined:

  direct         mean_{p in S} mean_{h in {28,31}} A^{22,h}_{p, I_j}
                 docs/saliency-maps.md map 1 -- what the GRPO overlap reward paid for.

  rollout_mean   docs map 2. Abnar-Zuidema rollout, heads merged by the mean, read at
                 the last layer, averaged over the step's tokens.

  rollout_wnorm  docs map 3. Same recursion, edge weight ||sum_h A^h_{p,q} W_O^h v^h_q||
                 -- the value-norm correction, so attention sinks stop dominating.

  grad           docs map 5, moved from the image embeddings down to the PIXELS, which
                 is what was asked for:

                     g_j = mean_{n in S} || d f_n / d (pixels of patch j) ||_2

                 f_n is the log-prob (or the raw logit, --grad-target) of the token the
                 model actually generated at n, teacher-forced on its own chain. Taking
                 the norm per token and averaging the norms -- rather than differentiating
                 the summed logit once -- keeps the "average over the step's tokens"
                 identical in meaning to the three attention maps, at the price of one
                 backward per token. Differentiating w.r.t. pixels rather than embeddings
                 means the vision tower is inside the graph, so the deepstack taps are
                 counted automatically and there is no `_ds` variant to choose.

  glimpse        docs map 6 -- GLIMPSE (arXiv 2506.18985v1), gradient-weighted attention
                 propagated with adaptive layer weights, aggregated over the step's
                 tokens by a confidence x prompt-alignment weight rather than a plain
                 mean. It is the one map here that is neither pure attention (1-3) nor
                 pure gradient (4): the gradient only says which heads and which layers
                 to believe, and the attention still says where to look.

The GLIMPSE deviations from the paper are in `glimpse_map` and in docs map 6; the one
that is not cosmetic is the relevance ROW. Taken literally, `R(t, :)` for the token at
position `t` is identically zero -- causality means position `t`'s attention row cannot
affect the logit that generated the token sitting there -- so the row read here is the
query that produced the token, `t-1`, which is what the method has to mean.

The pixel map has to be regrouped from the processor's patch layout to the language
model's token grid. `Qwen2VLImageProcessor` flattens to
[gh, gw, merge, merge, channel, temporal, patch, patch], so language-model token
(i, j) owns pixel rows 4*(i*gw + j) .. +4, and within a row the temporal axis is a
duplicate of the same pixels -- its two gradient halves are SUMMED (chain rule through
the duplication) before the norm is taken. `--stage selftest` proves that layout on the
real processor with a synthetic image; it needs no GPU and no weights.

Stages
------
  scan     GPU, shardable. Per sample: greedy chain -> observe-step segmentation
           (the FLAN-T5 classifier the reward uses) -> the five maps. Writes
           maps.npz + question.txt + generation.txt + original.png per sample.
  render   CPU, single process. Turns maps.npz into the overlays and an index.html.
           Re-runnable with different --norm/--cmap/--overlay-alpha; no GPU needed.
  selftest CPU. Gates the pixel->token regrouping.

  bash launch_saliency_viz.sh --gpus 8 --out-dir outputs/saliency_viz/run1

Output layout
-------------
  <out-dir>/samples/sample_000_row001234/
      question.txt  generation.txt  meta.json  maps.npz  original.png
      sal_direct.png  sal_rollout_mean.png  sal_rollout_wnorm.png  sal_grad.png
      sal_glimpse.png  contact_sheet.png
      steps/step00/  step.txt + the same six images for that step alone
  <out-dir>/index.html

The six images at sample level are the maps averaged over every observe step of that
sample; steps/stepNN/ keeps them separated, which is the thing the maps are actually
defined on.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]   # the repository root


def repo_path(rel: str) -> Path:
    """Resolve a repo-relative path, falling back to the central tree (see overlap_probe)."""
    p = REPO / rel
    if p.exists():
        return p
    if REPO.parent.name == ".worktrees":
        alt = REPO.parent.parent / rel
        if alt.exists():
            return alt
    return p


from experiments import _progress as IV                            # noqa: E402
from experiments._unavailable import unavailable                   # noqa: E402
from experiments.trained_model import probe as PROBE               # noqa: E402
from selfsal.data.prompt import IMAGE_TOKEN_ID                     # noqa: E402
from selfsal.steps import StepClassifier                           # noqa: E402
from selfsal.steps import segment as OSTEPS                        # noqa: E402

# ONE OF THE FIVE MAPS BELOW IS IN THIS REPOSITORY. `direct` is the attention map of
# Section 3.3, the one the reward reads and the one every published number comes from, and
# it needs nothing but the probe. The other four were explored and dropped:
#
#   rollout_mean, rollout_wnorm   the attention-rollout flow, `A flow_correlation_probe.py`
#   grad                          the pixel-gradient map, `A trl/grad_maps.py`
#   glimpse                       the GLIMPSE map, `A trl/glimpse_maps.py`
#
# so `--method direct` works here and the other four raise when the map is built, naming
# the archive. See experiments/_unavailable.py. Note that no figure in the paper is
# produced by this script -- docs/reproduce.md routes Figure 3 through
# `figures/steps_figure.py` and Figure 5 through `attention_bias/tables.py --panels`.
FC = unavailable("the attention-rollout flow", "flow_correlation_probe.py")
GM = unavailable("grad maps", "trl/grad_maps.py")


def pixel_regroup(*args, **kwargs):
    """`GM.pixel_regroup`, reached through the sentinel so it raises with the message.

    A plain `pixel_regroup = GM.pixel_regroup` here would be an attribute read at import
    time, which is the one thing the sentinel forbids.
    """
    return GM.pixel_regroup(*args, **kwargs)

METHODS = ("direct", "rollout_mean", "rollout_wnorm", "grad", "glimpse")
TITLES = {
    "direct": "direct  L{layer} heads {heads}",
    "rollout_mean": "rollout_mean  (L{rl})",
    "rollout_wnorm": "rollout_wnorm  (L{rl})",
    "grad": "grad  d {target} / d pixels",
    "glimpse": "glimpse  L{gfirst}+  ld={gdepth}",
}


# ---------------------------------------------------------------------------
# chain + observe-step segmentation
# ---------------------------------------------------------------------------
def segment_case(tok, clf, question: str, comp_ids: list[int]):
    """-> (text, steps, format_ok) or (text, None, reason).

    `steps` are (step_text, tok_a, tok_b) half-open spans into `comp_ids`, exactly the
    space intervene_probe.build_case works in.
    """
    text = tok.decode(comp_ids, skip_special_tokens=False,
                      clean_up_tokenization_spaces=False)
    enc = tok(text, add_special_tokens=False)
    # Everything downstream indexes `comp_ids` with spans found in `enc`'s space. That
    # is only sound if the decode/encode round trip is the identity, which it is for
    # this tokeniser but is an assumption, not a guarantee -- so check it rather than
    # silently mislabelling a sample's tokens.
    if list(enc["input_ids"]) != list(comp_ids):
        return text, None, "retokenise_mismatch"

    ms = re.search(r"<think>\s*(\S\S*)", text, re.DOTALL | re.MULTILINE)
    me = re.search(r"(\S)\s*</think>", text, re.DOTALL | re.MULTILINE)
    format_ok = bool(ms and me and me.start(1) > ms.start(1))
    if format_ok:
        ts_char, te_char = ms.start(1), me.start(1)
    else:
        # A malformed completion is still worth drawing -- the interesting failures are
        # often the malformed ones -- so fall back to "the whole completion is the
        # chain" and record that this happened.
        stripped = text.rstrip()
        ts_char = len(text) - len(text.lstrip())
        te_char = max(len(stripped) - 1, ts_char)
    ts = enc.char_to_token(0, ts_char)
    te = enc.char_to_token(0, te_char)
    if ts is None or te is None or te <= ts:
        return text, None, "bad_think_tokens"

    steps = OSTEPS.segment_sentences(text, ts_char, te_char, enc, 0, ts, te,
                                         question, clf)
    steps = [(s, a, b) for (s, a, b) in steps if 0 <= a < b <= len(comp_ids)]
    if not steps:
        return text, None, "no_observe_steps"
    return text, (steps, format_ok), None


# ---------------------------------------------------------------------------
# map 1: the direct map, layer 22 heads 28/31
# ---------------------------------------------------------------------------
def direct_map(model, inputs, prompt_len, comp_ids, steps, gh, gw, layer, heads, device):
    attn_mod = PROBE.find_attn_module(model, layer)
    if attn_mod is None:
        raise SystemExit(f"no Qwen3VLTextAttention with layer_idx={layer}")
    per_tok = PROBE.capture_layer_attention(model, attn_mod, inputs, prompt_len,
                                            comp_ids, list(heads), device)
    if per_tok.shape[-1] != gh * gw:
        raise RuntimeError(f"direct map has {per_tok.shape[-1]} patches, grid is {gh}x{gw}")
    out = np.zeros((len(steps), gh, gw), dtype=np.float32)
    for si, (_t, a, b) in enumerate(steps):
        # mean over the step's tokens, then over the two rewarded heads -- the reward's
        # own reduction (token_reduction="mean", then a head mean).
        out[si] = per_tok[:, a:b, :].mean(axis=1).mean(axis=0).reshape(gh, gw)
    return np.maximum(out, 0.0)


# ---------------------------------------------------------------------------
# maps 2/3: the rollout, heads merged by the mean or by value norm
# ---------------------------------------------------------------------------
def rollout_map(model, inputs, ids, prompt_len, steps, gh, gw, weighting, args, device):
    img_cols = (inputs["input_ids"][0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
    if img_cols.numel() != gh * gw:
        raise RuntimeError(f"{img_cols.numel()} image tokens, grid is {gh}x{gw}")
    need = sorted({prompt_len + p for _t, a, b in steps for p in range(a, b)})
    pos = {p: i for i, p in enumerate(need)}
    rows = torch.tensor(need, device=device)
    seq = ids.shape[1]

    engine = FC.RolloutFlow(model, weighting, args.alpha, args.chunk)
    try:
        engine.arm(seq, img_cols, rows,
                   IV.causal_mask(seq, next(model.parameters()).dtype, device))
        with torch.no_grad():
            model(**FC.build_forward(inputs, ids, prompt_len), use_cache=False)
        snaps = torch.stack(engine.snaps).cpu().numpy()      # [n_layers, n_rows, M]
    finally:
        engine.disarm()
        engine.close()

    li = args.rollout_layer if args.rollout_layer >= 0 else snaps.shape[0] - 1
    if not 0 <= li < snaps.shape[0]:
        raise SystemExit(f"--rollout-layer {args.rollout_layer} outside 0..{snaps.shape[0]-1}")
    sal = snaps[li]
    out = np.zeros((len(steps), gh, gw), dtype=np.float32)
    for si, (_t, a, b) in enumerate(steps):
        sel = [pos[prompt_len + p] for p in range(a, b)]
        out[si] = sal[sel].mean(axis=0).reshape(gh, gw)
    return out


# ---------------------------------------------------------------------------
# map 4: the gradient w.r.t. the pixels
#
# `pixel_regroup` moved to trl/grad_maps.py (imported above as GM and re-exported), so
# the map drawn here and the one the GRPO gradient reward scores are the same function.
# ---------------------------------------------------------------------------
def grad_map(model, processor, inputs, ids, prompt_len, steps, gh, gw, args, device):
    ip = processor.image_processor
    ps = int(getattr(ip, "patch_size", 16))
    tps = int(getattr(ip, "temporal_patch_size", 2))
    grid = inputs["image_grid_thw"][0].tolist()

    pv = inputs["pixel_values"].detach().float().requires_grad_(True)
    fwd = FC.build_forward(inputs, ids, prompt_len)
    # The leaf is fp32 and the cast into the model's dtype is part of the graph, so the
    # gradient that arrives back at `pv` is fp32 even though the tower runs in bf16.
    fwd["pixel_values"] = pv.to(inputs["pixel_values"].dtype)

    n_chain = ids.shape[1] - prompt_len
    out = np.zeros((len(steps), gh, gw), dtype=np.float32)
    with torch.enable_grad():
        res = model(**fwd, use_cache=False, logits_to_keep=n_chain + 1)
        logits = res.logits[0]
        # logits_to_keep trimmed the leading rows: absolute position p-1 predicts the
        # token at p and now sits at index p - prompt_len.
        for si, (_t, a, b) in enumerate(steps):
            acc = torch.zeros(gh, gw, dtype=torch.float32, device=device)
            for p in range(prompt_len + a, prompt_len + b):
                row = logits[p - prompt_len].float()
                tgt = int(ids[0, p])
                f = row[tgt] if args.grad_target == "logit" else row.log_softmax(-1)[tgt]
                (g,) = torch.autograd.grad(f, pv, retain_graph=True)
                acc += pixel_regroup(g, grid, ps, tps)
                del g
            out[si] = (acc / max(b - a, 1)).cpu().numpy()
    del res, logits
    return out


# ---------------------------------------------------------------------------
# map 5: GLIMPSE -- gradient-weighted attention, adaptively propagated
#
# arXiv 2506.18985v1, sections 3.2-3.5. The algebra lived in `A trl/glimpse_maps.py`,
# beside `A trl/grad_maps.py`, because the GLIMPSE grounding reward needed the same map.
# Neither the map nor that reward is in this repository, so this is the sentinel and
# `--method glimpse` raises when the map is built.
# ---------------------------------------------------------------------------
GLM = unavailable("GLIMPSE maps", "trl/glimpse_maps.py", "trl/rewards/glimpse_rewards.py")


def glimpse_map(*args, **kwargs):
    """`GLM.glimpse_map`, reached through the sentinel so it raises with the message."""
    return GLM.glimpse_map(*args, **kwargs)


# ---------------------------------------------------------------------------
# stage: scan
# ---------------------------------------------------------------------------
def sample_dir(out: Path, i: int, row_index: int) -> Path:
    return out / "samples" / f"sample_{i:03d}_row{row_index:06d}"


def scan(args, device):
    out = Path(args.out_dir)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    # `overlap_probe.prepare_image` reads this at call time, so setting it before
    # load_samples is what makes --max-image-side take effect; there is no per-call
    # argument to thread. 512 is the training value (trl/grpo_vlm_qwen3.MAX_IMAGE_SIDE)
    # and stays the default, because a map drawn at another resolution is not the map the
    # reward saw. Raising it is for benchmarks whose whole point is resolution --
    # HR-Bench 4K/8K -- where 512 px throws away the thing being asked about. GLIMPSE
    # builds an [N, N] edge matrix per layer, so cost grows with the SQUARE of the token
    # count: 1024 px is ~4x the image tokens of 512 and about 16x the propagation work.
    if args.max_image_side != PROBE.MAX_IMAGE_SIDE:
        print(f"[scan] max image side {PROBE.MAX_IMAGE_SIDE} -> {args.max_image_side}",
              flush=True)
        PROBE.MAX_IMAGE_SIDE = args.max_image_side
    rows = PROBE.load_samples(args.dataset, args.n_samples, args.seed,
                              cache_tag=f"_sv{args.shard}", split=args.split)
    todo = list(enumerate(rows))[args.shard::args.num_shards]
    print(f"[scan] shard {args.shard}/{args.num_shards}: {len(todo)} of {len(rows)} samples",
          flush=True)
    if not todo:
        return

    methods = [m for m in args.methods.split(",") if m]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise SystemExit(f"unknown method(s) {bad}; pick from {list(METHODS)}")
    heads = [int(h) for h in args.direct_heads.split(",") if h != ""]

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    processor, model = PROBE.load_model(args.base_model, args.adapter or None, device,
                                        args.attn_impl)
    model.requires_grad_(False)          # only `pixel_values` is ever differentiated
    if args.grad_checkpointing and "grad" in methods:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    clf = StepClassifier.load(args.steps_ckpt, device=device)
    tok = processor.tokenizer

    for i, row in todo:
        d = sample_dir(out, i, row["row_index"])
        if (d / "maps.npz").exists() and not args.overwrite:
            print(f"[scan] {d.name}: already done", flush=True)
            continue
        d.mkdir(parents=True, exist_ok=True)
        try:
            note = scan_one(model, processor, tok, clf, row, d, methods, heads,
                            args, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            note = "FAILED: out of memory"
        except Exception as e:                    # one bad sample must not kill a shard
            note = f"FAILED: {type(e).__name__}: {e}"
        print(f"[scan] {d.name}: {note}", flush=True)


def scan_one(model, processor, tok, clf, row, d: Path, methods, heads, args, device):
    image = row["image"]
    inputs, prompt_len, comp_ids = IV.greedy_chain(
        processor, model, image, row["question"], args.max_new_tokens, device)

    text, seg, reason = segment_case(tok, clf, row["question"], comp_ids)
    image.save(d / "original.png")
    (d / "question.txt").write_text(row["question"] + "\n")
    (d / "generation.txt").write_text(text + "\n")
    meta = {"row_index": row["row_index"], "dataset": row.get("dataset"),
            "question": row["question"], "gt_answer": row.get("gt_answer"),
            "generation": text, "methods": [], "steps": [],
            "direct_layer": args.direct_layer, "direct_heads": heads,
            "grad_target": args.grad_target, "rollout_layer": args.rollout_layer,
            "image_size": list(image.size)}
    if seg is None:
        meta["dropped"] = reason
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        return f"no maps ({reason})"
    steps, format_ok = seg
    meta["format_ok"] = format_ok
    if args.max_steps and len(steps) > args.max_steps:
        steps = steps[: args.max_steps]

    gh = int(inputs["image_grid_thw"][0, 1].item()) // 2
    gw = int(inputs["image_grid_thw"][0, 2].item()) // 2
    ids = torch.tensor([inputs["input_ids"][0].tolist() + list(comp_ids)], device=device)

    maps = {}
    for m in methods:
        if m == "direct":
            maps[m] = direct_map(model, inputs, prompt_len, comp_ids, steps, gh, gw,
                                 args.direct_layer, heads, device)
        elif m.startswith("rollout_"):
            maps[m] = rollout_map(model, inputs, ids, prompt_len, steps, gh, gw,
                                  m.split("_", 1)[1], args, device)
        elif m == "grad":
            maps[m] = grad_map(model, processor, inputs, ids, prompt_len, steps, gh, gw,
                               args, device)
        elif m == "glimpse":
            maps[m], meta["glimpse"] = glimpse_map(model, processor, inputs, ids,
                                                   prompt_len, steps, gh, gw,
                                                   row["question"], args, device)
        torch.cuda.empty_cache()

    meta["methods"] = list(maps)
    meta["grid"] = [gh, gw]
    meta["steps"] = [{"index": si, "text": t, "tok_a": int(a), "tok_b": int(b),
                      "n_tokens": int(b - a)}
                     for si, (t, a, b) in enumerate(steps)]
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    np.savez_compressed(d / "maps.npz", **{m: v for m, v in maps.items()})
    return f"{len(steps)} observe step(s), maps: {','.join(maps)}"


# ---------------------------------------------------------------------------
# stage: render
# ---------------------------------------------------------------------------
def normalize_map(m, mode: str, lo: float, hi: float):
    m = np.asarray(m, dtype=np.float64)
    if mode == "rank":
        flat = m.ravel()
        order = flat.argsort(kind="stable")
        ranks = np.empty(flat.size, dtype=np.float64)
        ranks[order] = np.arange(flat.size, dtype=np.float64)
        return (ranks / max(flat.size - 1, 1)).reshape(m.shape)
    if mode == "minmax":
        a, b = float(m.min()), float(m.max())
    else:                                          # percentile (default)
        a, b = float(np.percentile(m, lo)), float(np.percentile(m, hi))
    if not b > a:
        return np.zeros_like(m)
    return np.clip((m - a) / (b - a), 0.0, 1.0)


def overlay(img, m, cmap, args):
    """The image with `m` painted over it, upsampled from the patch grid."""
    from PIL import Image

    x = normalize_map(m, args.norm, args.norm_lo, args.norm_hi)
    rgb = (np.asarray(cmap(x))[..., :3] * 255).astype(np.uint8)
    resample = Image.NEAREST if args.upsample == "nearest" else Image.BILINEAR
    heat = Image.fromarray(rgb).resize(img.size, resample)
    if args.overlay_mode == "alpha":
        # transparency tracks saliency: the picture stays visible where nothing fires
        a = Image.fromarray((x * 255 * args.overlay_alpha).astype(np.uint8)).resize(
            img.size, resample)
        base = img.convert("RGB").copy()
        base.paste(heat, (0, 0), a)
        return base
    return Image.blend(img.convert("RGB"), heat, args.overlay_alpha)


def caption(img, text: str):
    from PIL import Image, ImageDraw, ImageFont

    bar = 16
    canvas = Image.new("RGB", (img.size[0], img.size[1] + bar), (16, 16, 16))
    canvas.paste(img, (0, bar))
    ImageDraw.Draw(canvas).text((3, 3), text[: max(1, img.size[0] // 6)],
                                fill=(235, 235, 235), font=ImageFont.load_default())
    return canvas


def contact_sheet(panels, pad: int = 6):
    from PIL import Image

    w = sum(p.size[0] for p in panels) + pad * (len(panels) + 1)
    h = max(p.size[1] for p in panels) + 2 * pad
    sheet = Image.new("RGB", (w, h), (16, 16, 16))
    x = pad
    for p in panels:
        sheet.paste(p, (x, pad))
        x += p.size[0] + pad
    return sheet


def render_sample(d: Path, cmap, args):
    from PIL import Image

    meta = json.loads((d / "meta.json").read_text())
    if not (d / "maps.npz").exists():
        return meta, 0
    img = Image.open(d / "original.png").convert("RGB")
    z = np.load(d / "maps.npz")
    methods = [m for m in METHODS if m in z]
    steps = meta["steps"]
    gm = meta.get("glimpse", {})
    label = {m: TITLES[m].format(layer=meta.get("direct_layer"),
                                 heads="/".join(str(h) for h in meta.get("direct_heads", [])),
                                 rl=("last" if meta.get("rollout_layer", -1) < 0
                                     else meta["rollout_layer"]),
                                 target=meta.get("grad_target", "logprob"),
                                 gfirst=gm.get("first_layer", "?"),
                                 gdepth=gm.get("depth_temp", "?"))
             for m in METHODS}

    def draw(dst: Path, get):
        dst.mkdir(parents=True, exist_ok=True)
        panels = [caption(img, "original")]
        for m in methods:
            o = overlay(img, get(m), cmap, args)
            o.save(dst / f"sal_{m}.png")
            panels.append(caption(o, label[m]))
        if not args.no_contact_sheet:
            contact_sheet(panels).save(dst / "contact_sheet.png")

    # sample level: the maps averaged over every observe step
    draw(d, lambda m: z[m].mean(axis=0))
    if not args.no_per_step:
        for si, st in enumerate(steps):
            sd = d / "steps" / f"step{si:02d}"
            sd.mkdir(parents=True, exist_ok=True)
            (sd / "step.txt").write_text(
                f"tokens [{st['tok_a']}, {st['tok_b']}) ({st['n_tokens']})\n\n{st['text']}\n")
            draw(sd, lambda m, si=si: z[m][si])
    return meta, len(steps)


def render(args):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import cm

    # colormaps[] is the modern spelling; cm.get_cmap was removed in matplotlib 3.9.
    cmap = (matplotlib.colormaps[args.cmap] if hasattr(matplotlib, "colormaps")
            else cm.get_cmap(args.cmap))
    out = Path(args.out_dir)
    dirs = sorted((out / "samples").glob("sample_*"))
    if not dirs:
        raise SystemExit(f"no samples under {out/'samples'} -- run --stage scan first")
    cards = []
    for d in dirs:
        if not (d / "meta.json").exists():
            continue
        meta, n = render_sample(d, cmap, args)
        cards.append((d, meta, n))
        print(f"[render] {d.name}: {n} step(s)", flush=True)
    write_index(out, cards, args)
    print(f"[render] {len(cards)} sample(s) -> {out/'index.html'}")


def write_index(out: Path, cards, args):
    e = html.escape
    parts = [
        "<!doctype html><meta charset='utf-8'><title>saliency maps</title>",
        "<style>body{background:#111;color:#ddd;font:13px/1.5 -apple-system,sans-serif;"
        "margin:24px}h2{margin:32px 0 4px}img{max-width:100%;border:1px solid #333}"
        "pre{white-space:pre-wrap;background:#181818;padding:8px;border-radius:4px}"
        ".q{color:#9cf}.s{margin:10px 0 10px 18px;border-left:2px solid #333;padding-left:12px}"
        "a{color:#9cf}</style>",
        f"<h1>saliency maps &mdash; {len(cards)} samples</h1>",
        f"<p>norm={e(args.norm)} ({args.norm_lo}&ndash;{args.norm_hi} pct), cmap={e(args.cmap)}, "
        f"overlay={e(args.overlay_mode)} alpha={args.overlay_alpha}. "
        "Left to right in each strip: original, then the maps that were scanned, in "
        "docs/saliency-maps.md order.</p>",
    ]
    for d, meta, n in cards:
        r = d.name
        parts.append(f"<h2>{e(r)} &mdash; {e(str(meta.get('dataset')))}</h2>")
        parts.append(f"<p class='q'>{e(meta['question'])}</p>")
        parts.append(f"<p>gold: <b>{e(str(meta.get('gt_answer')))}</b>"
                     + ("" if meta.get("format_ok", True) else " &nbsp;<i>(malformed completion)</i>")
                     + (f" &nbsp;<i>({e(meta['dropped'])})</i>" if meta.get("dropped") else "")
                     + "</p>")
        parts.append(f"<pre>{e(meta.get('generation',''))}</pre>")
        if (d / "contact_sheet.png").exists():
            parts.append(f"<p><i>mean over all {n} observe step(s)</i><br>"
                         f"<img src='samples/{r}/contact_sheet.png'></p>")
        for st in meta.get("steps", []):
            sp = d / "steps" / f"step{st['index']:02d}" / "contact_sheet.png"
            if sp.exists():
                parts.append(
                    f"<div class='s'><b>step {st['index']}</b> "
                    f"({st['n_tokens']} tokens)<br>{e(st['text'])}<br>"
                    f"<img src='samples/{r}/steps/step{st['index']:02d}/contact_sheet.png'></div>")
    (out / "index.html").write_text("\n".join(parts))


# ---------------------------------------------------------------------------
# stage: selftest -- the pixel -> token regrouping, on the real processor
# ---------------------------------------------------------------------------
def selftest(args):
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.base_model, padding_side="left")
    ip = processor.image_processor
    ps, tps = int(ip.patch_size), int(ip.temporal_patch_size)
    tok_px = ps * 2                       # merge_size 2: one LM token is 2x2 patches

    # Mid-grey, not black: image_mean/std are 0.5/0.5, so black normalises to -1 and a
    # black background would carry as much magnitude as a white square. Grey is ~0.
    rows_t, cols_t = 12, 16               # 384x512 -- above min_pixels, so no resize
    img = Image.new("RGB", (tok_px * cols_t, tok_px * rows_t), (128, 128, 128))
    ti, tj = 7, 11                        # the one bright token
    for y in range(ti * tok_px, (ti + 1) * tok_px):
        for x in range(tj * tok_px, (tj + 1) * tok_px):
            img.putpixel((x, y), (255, 255, 255))

    text = PROBE.build_prompt(processor, "test")
    inputs = processor(text=[text], images=[[img]], return_tensors="pt", padding=True,
                       padding_side="left", add_special_tokens=False)
    grid = inputs["image_grid_thw"][0].tolist()
    gh, gw = grid[1] // 2, grid[2] // 2
    if (gh, gw) != (rows_t, cols_t):
        raise SystemExit(f"selftest FAILED: processor resized the image; grid is {gh}x{gw}, "
                         f"expected {rows_t}x{cols_t}. Pick a size the processor keeps.")

    # pixel_values as a stand-in for a gradient: the regrouping is a pure reindexing,
    # so feeding it the values themselves must localise the bright token exactly.
    pv = inputs["pixel_values"].float()
    m = pixel_regroup(pv, grid, ps, tps).numpy()
    hot = np.unravel_index(int(m.argmax()), m.shape)
    n_img = int((inputs["input_ids"][0] == IMAGE_TOKEN_ID).sum())
    ok = hot == (ti, tj) and n_img == gh * gw
    share = float(m[ti, tj] / m.sum())
    print(f"[selftest] grid {gh}x{gw}, {n_img} image tokens, patch={ps} temporal={tps}")
    print(f"[selftest] brightest token {hot}, expected ({ti}, {tj}); "
          f"it holds {share:.1%} of the total")
    if not ok:
        raise SystemExit("[selftest] FAILED: pixel->token regrouping does not match the "
                         "processor's layout. The grad map would be scrambled.")
    print("[selftest] OK")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="scan", choices=["scan", "render", "selftest"])
    p.add_argument("--out-dir", default="")
    p.add_argument("--base-model", default=str(repo_path(
        "checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged")))
    p.add_argument("--adapter", default="")
    p.add_argument("--dataset", default=str(repo_path("cold_data/grpo_sets/val_natural")))
    p.add_argument("--split", default="all", choices=["train", "holdout", "all"],
                   help="'all' for the val_* sets (they have no holdout to carve)")
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    # glimpse replays one layer at a time in eager on top of this; `--attn-impl eager`
    # would put all 36 layers' [H, N, N] back in the graph and undo that.
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-image-side", type=int, default=PROBE.MAX_IMAGE_SIDE,
                   help="long-side cap on the input picture. The default is the training "
                        "value; raise it only for a benchmark whose question is about "
                        "resolution, and note that GLIMPSE's cost grows with the square "
                        "of the token count")
    p.add_argument("--max-steps", type=int, default=0,
                   help="cap observe steps per sample (0 = all)")
    p.add_argument("--steps-ckpt", default=os.environ.get(
        "OVERLAP_STEPS_CKPT", str(repo_path("checkpoint/steps_classifier/best"))))
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--direct-layer", type=int, default=22)
    p.add_argument("--direct-heads", default="28,31")
    p.add_argument("--alpha", type=float, default=0.5, help="rollout retention constant")
    p.add_argument("--rollout-layer", type=int, default=-1, help="-1 = last layer")
    p.add_argument("--chunk", type=int, default=256, help="wnorm Gram chunk")
    p.add_argument("--grad-target", default="logprob", choices=["logprob", "logit"],
                   help="logprob is docs/saliency-maps.md map 5; logit is the raw score")
    p.add_argument("--grad-checkpointing", action="store_true")
    # glimpse (docs/saliency-maps.md map 6). The defaults are the paper's, including
    # --glimpse-depth-temp: see glimpse_layer_alphas for why 0.36 is the other candidate.
    p.add_argument("--glimpse-temp", type=float, default=0.5,
                   help="lambda, head-fusion temperature (eq 6)")
    p.add_argument("--glimpse-depth-temp", type=float, default=0.2,
                   help="lambda_d, depth prior temperature (eq 9)")
    p.add_argument("--glimpse-layer-frac", type=float, default=1.0,
                   help="propagate the last frac of the stack; the paper's ablation loses "
                        "nothing at 0.6. A method knob, not a memory knob -- peak memory "
                        "no longer scales with the number of propagated layers")
    # The choices are written out rather than read off GM: building the parser must not
    # touch the sentinel, or --method direct dies before it has read the flag that would
    # have avoided the sentinel entirely. Same list as the archive's GM.GRAD_TARGETS.
    p.add_argument("--glimpse-target", default="logit",
                   choices=["clogit", "logit", "logprob"],
                   help="z_t in eqs 5 and 16; the paper's is the raw logit")
    # The reward's two cost dials, so the picture can be drawn for the map a capped or
    # trimmed run would actually score rather than only for the full one.
    p.add_argument("--glimpse-token-cap", type=int, default=0,
                   help="target tokens per step, drawn at random; 0 = all")
    p.add_argument("--glimpse-seed", type=int, default=0,
                   help="seed for the --glimpse-token-cap draw")
    p.add_argument("--glimpse-token-weight", default="full",
                   choices=["full", "confidence", "prompt", "uniform"],
                   help="eq 18, and the paper's token-saliency ablation")
    p.add_argument("--tf32", action="store_true", default=True)
    p.add_argument("--no-tf32", dest="tf32", action="store_false")
    # render-only
    p.add_argument("--norm", default="percentile", choices=["percentile", "minmax", "rank"])
    p.add_argument("--norm-lo", type=float, default=1.0)
    p.add_argument("--norm-hi", type=float, default=99.0)
    p.add_argument("--cmap", default="jet")
    p.add_argument("--overlay-mode", default="blend", choices=["blend", "alpha"])
    p.add_argument("--overlay-alpha", type=float, default=0.5)
    p.add_argument("--upsample", default="bilinear", choices=["bilinear", "nearest"])
    p.add_argument("--no-per-step", action="store_true")
    p.add_argument("--no-contact-sheet", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.stage == "selftest":
        selftest(args)
        return
    if not args.out_dir:
        raise SystemExit("--out-dir is required for --stage scan|render")
    if args.stage == "render":
        render(args)
        return
    device = args.device if torch.cuda.is_available() else "cpu"
    scan(args, device)


if __name__ == "__main__":
    main()
