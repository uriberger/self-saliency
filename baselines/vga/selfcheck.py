#!/usr/bin/env python
"""
vga_selfcheck.py — rung 2 of the wiki's validation ladder: is the injection wired
up, and is it inert when it should be?

    "Sanity-check the injection is live: with β large, attention maps and
     outputs should visibly change; with β = 0, generation must be
     bit-identical to the unpatched model. That equality check is the cheapest
     bug-catcher available."   — wiki/vga-implementation.md

Four arms, greedy, same prompt, same sample:

  stock         model.generate() with VGA not installed.
  beta0-fresh   VGA at β=0 with --vga-no-prefill-reuse. This is the strict
                equality check: the generate call is byte-for-byte the stock
                one, plus hooks that add zero. Any difference at all is the
                injection leaking, so this arm MUST match. It is a FAIL if not.
  beta0-reuse   VGA at β=0 on the default path, which resumes generation from
                the guidance pass's KV cache. Expected to match, but not
                guaranteed to: the prompt is prefilled as L-1 tokens plus one
                rather than L at once, and bf16 reductions over different shapes
                are not bit-identical. This arm therefore REPORTS divergence
                rather than failing on it -- it is a measurement of what the
                prefill-reuse trick costs, and the number to look at is how far
                into the answer the two agree.
  beta-live     VGA at --big-beta. Must differ from stock; if it does not, the
                hooks are not landing and every later result is the stock model.

Usage
-----
    conda run -n lmms_eval python scripts/vga_selfcheck.py --limit 5
    conda run -n lmms_eval python scripts/vga_selfcheck.py --image cat.png \
        --question "Is there a cat in the image?" --show-vocab

``--show-vocab`` additionally prints what each visual patch predicts, which is
the whole premise of the method in one glance: if the top-scoring patches do not
emit anything object-like, VSC has not survived the port and rung 1 will say so
at length.

Exit status is non-zero when a MUST arm fails, so this can gate a launcher.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from analysis import vga_common as common  # noqa: E402
from vlm import vga as vga_mod  # noqa: E402


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


@torch.inference_mode()
def _generate(model, inputs, max_new_tokens: int) -> tuple[list[int], float]:
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         do_sample=False, num_beams=1)
    dt = time.time() - t0
    prompt_len = inputs["input_ids"].shape[1]
    return out[0, prompt_len:].tolist(), dt


@torch.inference_mode()
def _vsc_report(model, processor, inputs, question, cfg, n_show: int = 3) -> None:
    """Print what the strongest and weakest visual patches actually predict."""
    tok = getattr(processor, "tokenizer", processor)
    ids = inputs["input_ids"]
    span = vga_mod.visual_span(ids, vga_mod.image_token_id(model, processor))
    fwd = {k: v for k, v in inputs.items() if k != "input_ids"}
    pre = vga_mod.run_prefill(model, ids, fwd, span, cfg)

    objects = vga_mod.extract_objects(question, cfg.max_objects)
    print(f"    objects extracted : {objects or '(none — VSS fallback)'}")
    if objects:
        tids = [vga_mod.first_token_ids(tok, o, cfg.object_variants) for o in objects]
        shown = {o: [tok.decode([t]) for t in ids_] for o, ids_ in zip(objects, tids)}
        print(f"    first tokens      : {shown}")
        g = vga_mod.object_map(model, pre, [t for t in tids if t], cfg)
        label = "VSC"
    else:
        g = vga_mod.salience_map(model, pre, cfg)
        label = "VSS"

    grid = vga_mod.patch_grid(inputs, processor)
    order = torch.argsort(g, descending=True)
    picks = list(order[:n_show].tolist()) + list(order[-n_show:].tolist())
    lm_head = model.get_output_embeddings()
    rows = pre.vis_hidden[picks]
    lg = vga_mod._unembed(lm_head, rows)
    top = lg.topk(5, dim=-1).indices

    print(f"    {label} map: m={pre.m} grid={grid} "
          f"max={float(g.max()):.5f} mean={float(g.mean()):.5f} "
          f"peak/mean={float(g.max() / g.mean()):.1f}")
    for k, patch in enumerate(picks):
        rank = "top" if k < n_show else "bottom"
        rc = f"({patch // grid[1]},{patch % grid[1]})" if grid else f"#{patch}"
        words = [repr(tok.decode([int(t)])) for t in top[k]]
        print(f"      {rank:6s} {rc:>10s} G={float(g[patch]):.6f}  {' '.join(words)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p)
    p.set_defaults(limit=5)
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="Tokens per arm (default: 64; equality shows up early).")
    p.add_argument("--beta", type=float, default=0.2, help="The operating point to report.")
    p.add_argument("--big-beta", type=float, default=2.0,
                   help="Deliberately over-driven beta for the 'it is landing' arm.")
    p.add_argument("--start-layer", type=int, default=4)
    p.add_argument("--end-layer", type=int, default=16)
    p.add_argument("--mode", default="auto", choices=["auto", "object", "agnostic"])
    p.add_argument("--show-vocab", action="store_true",
                   help="Print the top words the strongest/weakest patches predict.")
    args = p.parse_args()

    model, processor = common.load(args)

    def cfg(**kw):
        return vga_mod.VGAConfig(start_layer=args.start_layer, end_layer=args.end_layer,
                                 mode=args.mode, verbose=False, **kw)

    failures, notes = [], []
    n = 0
    for sample in common.samples(args):
        n += 1
        print(f"\n[{n}] sample={sample['sample_id']}  q={sample['question'][:90]!r}")
        inputs = common.build_inputs(processor, model, sample["image"], sample["question"])

        if args.show_vocab:
            _vsc_report(model, processor, inputs, sample["question"], cfg())

        stock, t_stock = _generate(model, inputs, args.max_new_tokens)

        arms = {}
        for name, kw in (
            ("beta0-fresh", dict(beta=0.0, prefill_reuse=False)),
            ("beta0-reuse", dict(beta=0.0, prefill_reuse=True)),
            (f"beta{args.beta}", dict(beta=args.beta)),
            ("beta-live", dict(beta=args.big_beta)),
        ):
            vga = vga_mod.install(model, processor, cfg(**kw))
            vga.set_question(sample["question"])
            try:
                toks, dt = _generate(model, inputs, args.max_new_tokens)
            finally:
                d = vga.diagnostics()
                vga.remove()
            arms[name] = (toks, dt, d)

        print(f"    stock                : {len(stock)} tok, {t_stock:.2f}s")
        for name, (toks, dt, d) in arms.items():
            div = _first_divergence(stock, toks)
            same = div is None
            agree = len(stock) if same else div
            print(f"    {name:20s}: {len(toks)} tok, {dt:.2f}s "
                  f"({dt / t_stock:.2f}x), agrees with stock for {agree}/{len(stock)} tokens"
                  f"{'' if same else f', first diff at {div}'}"
                  f" | injected {d['injected_steps']} steps, "
                  f"rel_update {d['mean_rel_update']:.4f}, mode {d['last_mode']}")

        # --- the assertions -------------------------------------------------
        toks, _, d = arms["beta0-fresh"]
        if toks != stock:
            failures.append(f"[{n}] beta0-fresh differs from stock at token "
                            f"{_first_divergence(stock, toks)} — the injection is not "
                            f"inert at beta=0")
        if d["injected_steps"] == 0:
            failures.append(f"[{n}] beta0-fresh never injected — the hooks did not fire, "
                            f"so this arm proves nothing")

        toks, _, _ = arms["beta0-reuse"]
        if toks != stock:
            notes.append(f"[{n}] beta0-reuse diverges from stock at token "
                         f"{_first_divergence(stock, toks)} of {len(stock)} — prefill-reuse "
                         f"is not bit-exact here (expected on bf16; see the header)")

        toks, _, d = arms["beta-live"]
        if toks == stock:
            failures.append(f"[{n}] beta={args.big_beta} left the output identical — the "
                            f"injection is not landing")
        if d["mean_rel_update"] <= 0:
            failures.append(f"[{n}] beta={args.big_beta} produced a zero-size update")

    print("\n" + "=" * 72)
    for note in notes:
        print(f"NOTE {note}")
    if failures:
        for f in failures:
            print(f"FAIL {f}")
        print(f"\n{len(failures)} failure(s) over {n} sample(s).")
        return 1
    print(f"PASS — over {n} sample(s): beta=0 is exactly the stock model, and "
          f"beta={args.big_beta} is not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
