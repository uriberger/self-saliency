#!/usr/bin/env python
"""Make a cold-start checkpoint loadable by the `laser` env. Idempotent.

    python fix_coldstart_tokenizer.py <checkpoint-dir> [...]
    python fix_coldstart_tokenizer.py /path/to/run/checkpoint-*

THE CHECKPOINT IS WRITTEN BY ONE TRANSFORMERS AND READ BY ANOTHER

Stage 1 trains in `sr1_coldstart` (transformers **5.7.0**) and Stage 2 — the LASER RL run
— loads the resulting checkpoint in `laser` (transformers **4.57.6**, their pin). 5.7.0
serialises `extra_special_tokens` into `tokenizer_config.json` as a **list**; 4.57.6's
`_set_model_specific_special_tokens` does `special_tokens.keys()` on it and dies with

    AttributeError: 'list' object has no attribute 'keys'

The base `Qwen/Qwen3-VL-8B-Instruct` has **no such key**, and loads fine in both. So the
fix is to drop the key rather than translate it: every token it lists (`<|im_start|>`,
`<|image_pad|>`, …) is already present in `tokenizer.json`'s `added_tokens`, and the key
only creates attribute aliases.

WHAT THIS DELIBERATELY DOES NOT TOUCH

The four trained tags live in `tokenizer.json` -> `added_tokens` as
`<think>`/`</think>`/`<answer>`/`</answer>` at ids 151667-151670 with `special: true`.
They are NOT in `tokenizer_config.json`'s `added_tokens_decoder`, which looks alarming and
is not: that is simply not where a fast tokenizer keeps them. Do not "repair" that.

`resize_vocab: true` was also a no-op worth knowing about — Qwen3-VL's `vocab_size` is
151,936 and the highest added id is 151,670, so the embedding already had room and no
rows were added. The tags are still learned; nothing was resized because nothing needed
to be.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


#: Qwen3-VL's `Qwen3VLTextRotaryEmbedding.__init__` in 4.57.6 does
#: `config.rope_scaling.get("mrope_section", ...)` and raises AttributeError on None.
#: 5.7.0 writes `rope_scaling: null` into the saved config; 5.x tolerates it, 4.57.6 does
#: not. This is the base model's value, restored verbatim.
BASE_ROPE_SCALING = {
    "mrope_interleaved": True,
    "mrope_section": [24, 20, 20],
    "rope_type": "default",
}


def fix_config(ckpt: Path) -> str:
    """Restore `rope_scaling` if 5.x nulled it out. Second half of the same bug."""
    p = ckpt / "config.json"
    if not p.is_file():
        return "no config.json"
    d = json.loads(p.read_text())
    tcfg = d.get("text_config", d)
    if tcfg.get("rope_scaling") is not None:
        return "rope_scaling already set"
    if not (ckpt / "config.json.orig").exists():
        shutil.copy2(p, ckpt / "config.json.orig")
    tcfg["rope_scaling"] = dict(BASE_ROPE_SCALING)
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    return "restored rope_scaling from the base model; backup at config.json.orig"


def fix(ckpt: Path) -> str:
    cfg = ckpt / "tokenizer_config.json"
    if not cfg.is_file():
        return "no tokenizer_config.json"
    d = json.loads(cfg.read_text())
    val = d.get("extra_special_tokens")
    if val is None:
        return "already clean"
    if not isinstance(val, list):
        return f"left alone (extra_special_tokens is {type(val).__name__}, not a list)"

    # Verify the trained tags survive in tokenizer.json before touching anything. If they
    # are missing, this checkpoint cannot teach the format and silently "fixing" its
    # tokenizer would hide that.
    tj = ckpt / "tokenizer.json"
    want = {"<think>", "</think>", "<answer>", "</answer>"}
    if tj.is_file():
        got = {a.get("content") for a in json.loads(tj.read_text()).get("added_tokens", [])}
        missing = want - got
        if missing:
            return f"REFUSED: {sorted(missing)} absent from tokenizer.json added_tokens"

    if not (ckpt / "tokenizer_config.json.orig").exists():
        shutil.copy2(cfg, ckpt / "tokenizer_config.json.orig")
    d.pop("extra_special_tokens")
    cfg.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    return f"dropped extra_special_tokens (list of {len(val)}); backup at *.orig"


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    for p in argv:
        ckpt = Path(p)
        print(f"{ckpt.name:<18} tokenizer: {fix(ckpt)}")
        print(f"{'':<18} config:    {fix_config(ckpt)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
