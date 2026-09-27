# Copyright 2026 NVIDIA. Apache-2.0.
"""The reasoning-step classifier of Appendix A.1.

Four functional categories for a step of a reasoning chain:

    plan      a forward-looking statement about an intended action
              "let me check the left side of the image"
    observe   a direct statement about the visual content of the image
              "there is a red car on the left"
    deduce    a conclusion drawn from earlier reasoning rather than from new visual
              evidence -- "therefore it is a traffic sign"
    none      a statement that does not contribute to the reasoning

Only `observe` steps are scored. That is the whole reason this model exists: phi asks
whether the model looked at what it said it saw, and a plan or a deduction makes no
claim about the image for the attention to be aligned with.

ARCHITECTURE. A frozen-topology FLAN-T5 encoder, masked mean-pooling over the token
dimension, and a two-layer MLP head. Each step is classified individually but the FULL
chain is supplied as context, so an ambiguous fragment can be resolved by what surrounds
it. Trained by distilling labels from a proprietary model (`make_data.py`); it reaches
91.9% on the held-out test set and 93% on a human-labelled one (`evaluate.py`).

ONE DEFINITION, THREE CALLERS. This was previously two classes in two repos -- a
trainable one in the analysis repo and an inference-only transcription of it in the
trainer -- which had to be kept architecturally identical by hand, with a comment in
each saying so. They are one class now. The inference-only concerns that lived only in
the trainer's copy (the ZeRO-3 guard, the apex fallback, batched prediction) are all
here, because the training script wants none of them and is not harmed by them.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
from pathlib import Path

import torch
import torch.nn as nn

#: Order is load-bearing: it is the head's output order, so it must match training.
LABELS = ("plan", "observe", "deduce", "none")
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

_STEP_SEP = "[STEP]"
_CHAIN_SEP = "[CHAIN]"

#: Where the trained checkpoint lives, overridable for a different one.
CKPT_ENV = "SELFSAL_STEPS_CKPT"


def default_checkpoint() -> Path:
    return Path(os.environ.get(CKPT_ENV, "checkpoint/steps_classifier/best"))


def build_input(step_text: str, chain: str, question: str, include_chain: bool) -> str:
    """The exact string the model was trained on.

    Changing this silently changes what every stored prediction meant, so it lives in
    one place and both the training script and both segmenters call it.
    """
    parts = [_STEP_SEP, step_text.strip()]
    if include_chain:
        ctx = (f"Question: {question.strip()} Chain: {chain.strip()}"
               if question.strip() else chain.strip())
        parts += [_CHAIN_SEP, ctx]
    return " ".join(parts)


@contextlib.contextmanager
def _no_deepspeed_zero3_init():
    """Hide HF's global ZeRO-3 config from `from_pretrained` for the duration.

    The GRPO trainer runs under DeepSpeed ZeRO-3 with `zero3_init_flag: true`, which
    registers a process-global HfDeepSpeedConfig. Every later `from_pretrained` -- this
    auxiliary encoder included -- is then wrapped in `deepspeed.zero.Init` and has its
    parameters partitioned into 1-D shards. A sharded `embed_tokens.weight` is no longer
    2-D and the embedding lookup raises `RuntimeError: 'weight' must be 2-D`, at
    inference, with nothing in the message naming this model. This is a small frozen
    single-device encoder that must be fully materialised. No-op without the deepspeed
    integration.
    """
    try:
        import transformers.integrations.deepspeed as ds
    except Exception:
        yield
        return
    saved = getattr(ds, "_hf_deepspeed_config_weak_ref", None)
    ds._hf_deepspeed_config_weak_ref = None
    try:
        yield
    finally:
        ds._hf_deepspeed_config_weak_ref = saved


@contextlib.contextmanager
def _t5_layernorm_fallback():
    """Survive an apex install whose fused LayerNorm extension is broken.

    apex monkey-patches transformers' `T5LayerNorm` with a FusedLayerNorm. When its CUDA
    extension does not load (undefined symbol, version skew) the substitution is still
    in place and loading the encoder crashes. Swap in a plain-PyTorch RMS-norm for the
    load, then put the original back.
    """
    import transformers.models.t5.modeling_t5 as t5_mod

    try:
        importlib.import_module("fused_layer_norm_cuda")
        yield
        return
    except ImportError:
        pass

    class _FallbackT5LN(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            v = x.float().pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(v + self.variance_epsilon)
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                x = x.to(self.weight.dtype)
            return self.weight * x

    original = t5_mod.T5LayerNorm
    t5_mod.T5LayerNorm = _FallbackT5LN
    try:
        yield
    finally:
        t5_mod.T5LayerNorm = original


def _make_head(d: int, num_labels: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d), nn.Dropout(dropout),
        nn.Linear(d, d // 2), nn.GELU(),
        nn.Dropout(dropout), nn.Linear(d // 2, num_labels),
    )


class StepClassifier(nn.Module):
    """FLAN-T5 encoder + masked mean-pooling + MLP head."""

    def __init__(self, encoder_name: str = "google/flan-t5-base",
                 num_labels: int = len(LABELS), dropout: float = 0.1,
                 include_chain: bool = True, _encoder=None):
        super().__init__()
        if _encoder is not None:
            self.encoder = _encoder
        else:
            from transformers import T5EncoderModel
            self.encoder = T5EncoderModel.from_pretrained(encoder_name)
        self.head = _make_head(self.encoder.config.d_model, num_labels, dropout)
        self.include_chain = include_chain
        self._tokenizer = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled)

    def save(self, path: Path, tokenizer=None) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path / "encoder")
        torch.save(self.head.state_dict(), path / "head.pt")
        (path / "cfg.json").write_text(json.dumps({
            "include_chain": bool(self.include_chain),
            "labels": list(LABELS),
        }, indent=2))
        if tokenizer is not None:
            tokenizer.save_pretrained(path / "tokenizer")

    @classmethod
    def load(cls, path: str | Path | None = None, device: str | None = None,
             num_labels: int = len(LABELS)) -> "StepClassifier":
        from transformers import AutoTokenizer, T5EncoderModel

        ckpt = Path(path) if path else default_checkpoint()
        cfg_path = ckpt / "cfg.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

        with _t5_layernorm_fallback(), _no_deepspeed_zero3_init():
            encoder = T5EncoderModel.from_pretrained(ckpt / "encoder")

        obj = cls(num_labels=num_labels,
                  include_chain=bool(cfg.get("include_chain", True)),
                  _encoder=encoder)
        obj.head.load_state_dict(torch.load(ckpt / "head.pt", map_location="cpu"))
        obj._tokenizer = AutoTokenizer.from_pretrained(ckpt / "tokenizer")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return obj.eval().to(device)

    @torch.no_grad()
    def predict(self, step_text: str, chain: str, question: str = "") -> str:
        return self.predict_many([step_text], chain, question)[0]

    @torch.no_grad()
    def predict_many(self, step_texts: list[str], chain: str,
                     question: str = "") -> list[str]:
        """One padded encoder forward for all the steps of a chain.

        Masked mean-pooling makes this numerically identical to classifying each step on
        its own, and it is the difference between one forward per completion and
        hundreds of serial CPU T5 forwards per optimizer step.
        """
        if not step_texts:
            return []
        if self._tokenizer is None:
            raise RuntimeError("no tokenizer on this classifier; use StepClassifier.load()")
        enc = self._tokenizer(
            [build_input(t, chain, question, self.include_chain) for t in step_texts],
            return_tensors="pt", truncation=True, max_length=512, padding=True)
        device = next(self.parameters()).device
        logits = self(input_ids=enc["input_ids"].to(device),
                      attention_mask=enc["attention_mask"].to(device))
        return [ID2LABEL[int(i)] for i in logits.argmax(dim=-1).tolist()]
