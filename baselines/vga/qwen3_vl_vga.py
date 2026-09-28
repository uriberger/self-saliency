"""Qwen3-VL with Vision-Guided Attention applied while it answers.

`qwen3_vl` with one thing added: `vlm.vga.install()` from the vlm_reasoning repo
wraps `model.generate`, so before each answer the model runs one guidance pass
over the prompt, reads a per-patch relevance map off the visual tokens' own word
distributions, and injects it into the attention that answer tokens pay to the
picture.  Nothing else differs — prompts, generation kwargs, decoding and
scoring are inherited unchanged, so a score from this model sits on the same
axis as a score from `qwen3_vl`.

    bash scripts/run_lmms_eval_suite.sh --model CKPT --model-type qwen3_vl_vga ...

VGA is training-free, so there is no checkpoint here: `--model` is stock
Qwen3-VL (or any of our own), and the arm is entirely in the configuration.

CONFIGURED THROUGH THE ENVIRONMENT, because the launchers build `--model_args`
themselves and have no passthrough:

    VGA_ARGS      comma-separated overrides, e.g.
                  "beta=0.25,start_layer=2,end_layer=18,mode=object".
                  Keys are VGAConfig field names and an unknown one is a hard
                  error, not a silent no-op — a typo'd knob that quietly does
                  nothing is a benchmark run of the wrong thing.
    VGA_REPO      where vlm/vga.py lives (default below).

Every field is also settable as a model_arg with a `vga_` prefix
(`vga_beta=0.25`) where a caller has a way to set them; the model_arg wins.

WHY THIS IS A SEPARATE FILE.  lmms-eval imports only the model actually
requested, so a job running `--model qwen3_vl` never opens this one.
Registering it costs a single added key in `lmms_eval/models/__init__.py`, which
cannot change what any other key resolves to.  That is the whole reason it is
not a flag on `qwen3_vl`: no concurrent evaluation should be able to notice that
this exists.

BATCH SIZE 1 ONLY.  The guidance map is per-sample and the visual span is
located from the prompt's own token ids; left padding in a wider batch moves
every image column, so a batch is refused rather than silently guided in the
wrong place.
"""

import os
import pathlib
import sys
from typing import List

from loguru import logger as eval_logger

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.qwen3_vl import Qwen3_VL

# Where to import `baselines.vga.vga` from. lmms-eval loads this module out of its own
# package, so the repository is not on sys.path by the time it runs; SELFSAL_ROOT is how
# the eval launcher tells it. Falls back to walking up from this file, which works for an
# editable install and fails loudly rather than silently importing nothing.
DEFAULT_REPO = os.environ.get(
    "SELFSAL_ROOT", str(pathlib.Path(__file__).resolve().parents[2]))


def _coerce(field, raw):
    """Turn a string from the environment into what VGAConfig expects.

    Typed off the field's *default value*, not its annotation: vlm/vga.py uses
    ``from __future__ import annotations``, so ``field.type`` is the string
    ``"float"`` rather than ``float`` and every isinstance check against it
    would quietly fall through to str.
    """
    want = type(field.default)
    if isinstance(raw, want) and not (want is not bool and isinstance(raw, bool)):
        return raw
    text = str(raw).strip()
    if want is bool:                       # before int: bool subclasses int
        low = text.lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"cannot read {text!r} as a boolean")
    if want is int:
        return int(text)
    if want is float:
        return float(text)
    return text


def _parse_overrides(spec, fields: dict) -> dict:
    """"beta=0.25,start_layer=2" → {"beta": 0.25, "start_layer": 2}."""
    out = {}
    if not spec:
        return out
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"VGA_ARGS entry {part!r} is not key=value")
        key, value = (s.strip() for s in part.split("=", 1))
        if key not in fields:
            raise ValueError(
                f"VGA_ARGS: {key!r} is not a VGAConfig field. Known fields: "
                f"{', '.join(sorted(fields))}")
        out[key] = _coerce(fields[key], value)
    return out


@register_model("qwen3_vl_vga")
class Qwen3_VL_VGA(Qwen3_VL):
    is_simple = False

    def __init__(self, vga_repo=None, **kwargs):
        # Anything the caller passed as vga_<field>= is an override; strip those
        # before they reach the stock wrapper, which would reject them.
        prefixed = {k[len("vga_"):]: v for k, v in list(kwargs.items())
                    if k.startswith("vga_")}
        for k in list(kwargs):
            if k.startswith("vga_"):
                kwargs.pop(k)

        super().__init__(**kwargs)

        repo = vga_repo or os.environ.get("VGA_REPO") or DEFAULT_REPO
        if repo not in sys.path:
            sys.path.insert(0, repo)
        try:
            from vlm.vga import VGAConfig, install
        except ImportError as exc:
            raise ImportError(
                f"cannot import vlm.vga from {repo}; set VGA_REPO") from exc

        fields = VGAConfig.__dataclass_fields__
        cfg_kwargs = _parse_overrides(os.environ.get("VGA_ARGS"), fields)
        for k, v in prefixed.items():
            if k not in fields:
                raise ValueError(f"vga_{k} is not a VGAConfig field")
            cfg_kwargs[k] = _coerce(fields[k], v)

        if int(self.batch_size) != 1:
            raise ValueError(
                f"qwen3_vl_vga needs batch_size=1, got {self.batch_size}: the guidance "
                "map is per-sample and the visual span is located from the prompt's "
                "token ids, so left padding in a wider batch moves every image column.")

        cfg = VGAConfig(**cfg_kwargs)
        self.vga = install(self.model, self.processor, cfg)
        eval_logger.warning(
            f"qwen3_vl_vga ACTIVE: beta={cfg.beta} layers=[{cfg.start_layer},"
            f"{cfg.end_layer}{']' if cfg.end_layer_inclusive else ')'} "
            f"mode={cfg.mode} head_balancing={cfg.head_balancing} "
            f"attn_norm={cfg.attn_norm} sparsity_scaling={cfg.sparsity_scaling} "
            f"object_variants={cfg.object_variants} topk={cfg.topk} "
            f"vss_invert={cfg.vss_invert} pvg={cfg.pvg} lam={cfg.lam} "
            f"prefill_reuse={cfg.prefill_reuse}")
        if cfg.beta == 0:
            eval_logger.warning(
                "qwen3_vl_vga: beta=0 is exactly the identity, so THIS RUN IS THE "
                "STOCK MODEL however the output directory is named.")

    def generate_until(self, requests: List[Instance]):
        out = super().generate_until(requests)
        # The landing check, in the eval log. A run that guided nothing is a run
        # of the stock model, and no other line of the output would say so.
        d = self.vga.diagnostics()
        eval_logger.warning(
            f"qwen3_vl_vga landed: guided {d['guided']}/{d['calls']} generate calls "
            f"({d['no_image']} had no image), {d['object_mode']} object-directed / "
            f"{d['agnostic_mode']} object-agnostic ({d['fallbacks']} fell back to VSS), "
            f"{d['injected_steps']} decode steps injected on layers {d['layers']}, "
            f"mean relative update {d['mean_rel_update']:.4f}, "
            f"beta_eff {d['last_beta_eff']:.4f}")
        if d["injected_steps"] == 0:
            eval_logger.error(
                "qwen3_vl_vga: NOTHING WAS EVER INJECTED — these are stock-model scores.")
        return out
