"""
vga.py — Vision-Guided Attention (VGA) on Qwen3-VL, as an inference-time patch.

Implements *Tell Model Where to Look: Mitigating Hallucinations in MLLMs by
Vision-Guided Attention* (CVPR 2026, arXiv:2511.20032).  ``wiki/vga-implementation.md``
is the derivation; this module is the code it asks for.  VGA is training-free:
there is no checkpoint, only stock weights plus the hooks installed here.

The method in one paragraph
---------------------------
Push the LLM's hidden state at each *visual* token position through the
unembedding matrix, as if predicting the next token there.  The resulting word
distribution is semantically meaningful — the probability a patch assigns to
"dog" tracks whether that patch holds a dog — and gives a per-patch relevance
map ``G`` over the ``m`` visual tokens.  VGA adds ``G`` into the attention that
answer tokens pay to visual tokens.  Crucially that never needs the attention
matrix, because adding a constant row to the attention distribution distributes
over the value multiplication::

    ẑ = (α + β·G) V  =  z + β·Δz        where   Δz = Gᵀ · V[s:e]

So the whole method is: read ``Δz`` off the value states of the visual
positions, and add a scaled copy to the attention output before ``o_proj``.
FlashAttention / SDPA stay on, and no modeling file is forked.

No forked modeling file
-----------------------
Upstream ships a vendored ``modeling_qwen2_5_vl.py`` with the injection spliced
into the attention forward, plus a forked greedy sampler.  We do it with hooks:

* **Values** come out of the KV cache (``vlm.saliency._get_value_tensor``).
  Visual tokens live in the prompt, so their value vectors are written once
  during prefill and never move.  Capturing them from a ``v_proj`` hook during
  decode would not work — at that point ``v_proj`` only sees the new token.
* **Injection** is a ``forward_pre_hook`` on ``self_attn.o_proj``, whose input is
  the concatenated head outputs ``(B, T_q, H·D)``.

Qwen3-VL is GQA, so cached values arrive as ``(B, H_kv, T, D)`` and are
``repeat_kv``'d up to ``H_q`` before the per-head ``Δz`` is formed.

Usage
-----
::

    from vlm.vga import VGAConfig, install

    vga = install(model, processor, VGAConfig(beta=0.2, start_layer=4, end_layer=16))
    out = model.generate(**inputs, max_new_tokens=128)   # VGA applies transparently
    print(vga.diagnostics())
    vga.remove()

``install`` wraps ``model.generate``, so any caller — ``run_experiment.py``,
lmms-eval, a notebook — gets VGA without knowing it exists.  Batch size 1 only:
``G`` is per-sample and the visual span is located from the sample's own token
ids, so left padding in a wider batch would move every image column.

Two things in here are NOT settled, and both are flagged at their definition:
``VSS_ENTROPY_SIGN`` and ``LAYER_WINDOW``.  Read those before trusting a sweep.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# The wiki points at these by name; they already handle every cache API
# generation and every Qwen backbone nesting level this project has met.
from vlm.saliency import _get_model_config, _get_value_tensor, _get_image_token_id


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VGAConfig:
    """Every knob VGA has.  Defaults are the paper's, except where noted.

    LAYER_WINDOW.  ``start_layer``/``end_layer`` are the one thing that does not
    transfer between backbones: the paper uses 4–16 for Qwen2.5-VL-7B, 2–16 for
    LLaVA-1.5, 0–16 for LLaVA-Next.  The defaults here are Qwen2.5-VL's and are
    a *starting point, not a setting* — ``analysis/vga_layer_scan.py`` measures
    the PAI heuristic (the layer where attention to position 0 takes off) on
    Qwen3-VL, and Qwen3-VL additionally injects vision features at
    ``config.deepstack_visual_indexes`` (default 8/16/24), which is one more
    reason not to assume 4.  The window is half-open by default
    (``range(start, end)``, so ``end_layer`` is excluded); set
    ``end_layer_inclusive=True`` if a comparison needs the other reading.
    """

    # --- guidance strength -------------------------------------------------
    beta: float = 0.2                  # 0.25 for POPE and for larger models
    sparsity_scaling: bool = True      # β ← β·‖G‖₀/m, upstream's automatic damping

    # --- where it applies --------------------------------------------------
    start_layer: int = 4
    end_layer: int = 16
    end_layer_inclusive: bool = False

    # --- how G is built ----------------------------------------------------
    mode: str = "auto"                 # "auto" | "object" (VSC) | "agnostic" (VSS)
    topk: int = 10                     # K for the VSS entropy
    vss_invert: bool = False           # see VSS_ENTROPY_SIGN in salience_map()
    object_variants: str = "both"      # "plain" | "both"; see first_token_ids()
    max_objects: int = 4

    # --- head balancing ----------------------------------------------------
    head_balancing: str = "simg"       # "simg" (cosine scheme) | "none"
    attn_norm: bool = False            # convex-blend variant; upstream defaults it off

    # --- progressive visual guidance (captioning only) ---------------------
    pvg: bool = False
    lam: float = 0.02
    pvg_normalize: bool = True         # see pvg_update()

    # --- execution ---------------------------------------------------------
    prefill_reuse: bool = True         # resume generation from pass A's cache
    logit_chunk: int = 128             # rows of vis_hidden unembedded at a time
    sparsity_eps: float = 1e-8
    verbose: bool = True

    def layers(self, n_layers: int) -> list[int]:
        end = self.end_layer + 1 if self.end_layer_inclusive else self.end_layer
        return [i for i in range(max(0, self.start_layer), min(end, n_layers))]


def _log(cfg: VGAConfig | None, msg: str) -> None:
    if cfg is None or cfg.verbose:
        print(f"[vga] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Model plumbing
# ---------------------------------------------------------------------------

def _text_config(model):
    return getattr(model.config, "text_config", model.config)


def _repeat_kv(v: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, H_kv, T, D) → (B, H_kv·n_rep, T, D), matching transformers' repeat_kv.

    Query head i reads kv head i // n_rep, so kv head h expands into the
    contiguous block [h·n_rep, (h+1)·n_rep).
    """
    if n_rep == 1:
        return v
    b, h, t, d = v.shape
    return v[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def image_token_id(model, processor) -> int:
    """The id of the per-patch placeholder token, config first."""
    tid = getattr(getattr(model, "config", None), "image_token_id", None)
    if tid is not None:
        return int(tid)
    return _get_image_token_id(processor, model)


def visual_span(input_ids: torch.Tensor, img_id: int) -> tuple[int, int] | None:
    """(start, end) of the visual block, end-exclusive.  None if there is no image.

    Qwen3-VL uses dynamic resolution, so ``m = e - s`` varies per sample.  The
    span is taken as first..last placeholder: a single image is contiguous, and
    for several images this covers them all plus whatever sits between, which is
    what a single flat G over "the picture" means anyway.
    """
    pos = (input_ids[0] == img_id).nonzero().flatten()
    if pos.numel() == 0:
        return None
    return int(pos[0].item()), int(pos[-1].item()) + 1


def patch_grid(inputs, processor=None) -> tuple[int, int] | None:
    """(grid_h, grid_w) after the 2×2 patch merge — for visualising G only.

    Nothing in VGA needs a square grid, or any grid at all; this is the
    assumption that breaks KLAL and VGA is free of it.
    """
    thw = inputs.get("image_grid_thw") if hasattr(inputs, "get") else None
    if thw is None:
        return None
    merge = 2
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        merge = int(getattr(ip, "merge_size", 2) or 2)
    g = thw[0]
    return int(g[1]) // merge, int(g[2]) // merge


# ---------------------------------------------------------------------------
# Object extraction  (object-directed mode)
# ---------------------------------------------------------------------------

# Upstream uses spaCy noun chunks.  spaCy is not installed in venv_qwen35 and a
# pip install there is global — it would hit every session and every running job
# — so the default is the dependency-free extractor below, with spaCy used
# automatically when it happens to be importable.  Object-extraction quality
# bounds object-directed mode, so `VGA.set_objects()` exists for callers that
# know the objects exactly, and every extraction is logged.

_STOPWORDS = frozenset("""
a an the this that these those some any each every no another both either neither
is are was were be been being am do does did doing done have has had having
can could will would shall should may might must
i you he she it we they me him her them us my your his its our their
and or but if then than so because while when where which who whom whose what
how why not none nor as at by for from in into of off on onto out over to under
with within without about above across after against along among around before
behind below beneath beside between beyond during near through toward towards up
upon there here it's dont don't cant can't
please tell describe explain say answer respond choose select given give shown
show shows see seen look looking observe according based following option options
question choice choices letter correct incorrect true false yes maybe
image images picture pictures photo photos photograph figure figures scene diagram
view frame visible depicted present presence
color colour colours colors shape shapes size sizes kind kinds type types name
names side sides part parts number numbers amount many much most least more less
left right top bottom front back middle centre center corner corners area areas
region regions position positions place places thing things something anything
one two three four five six seven eight nine ten first second third last next
detail details detailed brief briefly caption overall approximately roughly
""".split())

# Verbs and adjectives that survive the stoplist but never name an object.
_NON_NOUN_SUFFIXES = ("ing", "ed", "ly", "est")


def _spacy_chunks(text: str) -> list[str] | None:
    """Upstream's extractor, when spaCy is available.  None when it is not."""
    try:
        import spacy  # type: ignore
    except Exception:
        return None
    try:
        nlp = _spacy_chunks._nlp  # type: ignore[attr-defined]
    except AttributeError:
        nlp = None
        for name in ("en_core_web_sm", "en_core_web_md"):
            try:
                nlp = spacy.load(name, disable=["ner", "lemmatizer"])
                break
            except Exception:
                continue
        _spacy_chunks._nlp = nlp  # type: ignore[attr-defined]
    if nlp is None:
        return None
    doc = nlp(text)
    out = []
    for chunk in doc.noun_chunks:
        toks = [t.text for t in chunk if t.pos_ not in ("DET", "PRON")]
        if toks:
            out.append(" ".join(toks))
    return out


# Benchmark harnesses append an answer-format instruction to the user turn, and
# it is prose full of content words: lmms-eval's POPE prompt ends "Answer the
# question using a single word or phrase.", out of which the extractor below
# happily produced the objects 'single word' and 'phrase'. Those are then
# max'ed into G alongside the real object and dilute it, so the guidance is
# weaker than the method deserves -- which for a baseline is the expensive
# direction to be wrong in. Strip the instruction before extracting, rather
# than stoplisting every word that appears in one.
_ANSWER_FORMAT_RE = re.compile(
    r"""(
        answer\s+the\s+question\s+using\s+a\s+single\s+word\s+or\s+phrase\.?
      | answer\s+with\s+the\s+option'?s?\s+letter\s+from\s+the\s+given\s+choices\s+directly\.?
      | please\s+answer\s+(directly\s+)?with[^.?!]*[.?!]?
      | answer\s+the\s+question\s+with[^.?!]*[.?!]?
      | (please\s+)?(select|choose)\s+the\s+(correct|best)\s+(answer|option)[^.?!]*[.?!]?
      | your\s+answer\s+should\s+be[^.?!]*[.?!]?
      | respond\s+with[^.?!]*[.?!]?
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def strip_answer_format(question: str) -> str:
    """Drop the harness's answer-format instruction from a question."""
    return " ".join(_ANSWER_FORMAT_RE.sub(" ", question or "").split())


def extract_objects(question: str, max_objects: int = 4) -> list[str]:
    """Objects named in the question, best effort, most-specific first.

    spaCy noun chunks when spaCy is importable; otherwise consecutive runs of
    content words, which is enough for the question shapes these benchmarks use
    ("Is there a *chair* in the image?", "What is the *man* holding?").
    """
    if not question:
        return []
    question = strip_answer_format(question)
    if not question:
        return []
    chunks = _spacy_chunks(question)
    if chunks is None:
        text = re.sub(r"[^a-z0-9\s'-]+", " ", question.lower())
        phrase: list[str] = []
        chunks = []
        for word in text.split():
            word = word.strip("'-")
            keep = (
                len(word) >= 2
                and word not in _STOPWORDS
                and not word.isdigit()
                and not word.endswith(_NON_NOUN_SUFFIXES)
            )
            if keep:
                phrase.append(word)
            elif phrase:
                chunks.append(" ".join(phrase))
                phrase = []
        if phrase:
            chunks.append(" ".join(phrase))

    seen, out = set(), []
    for c in chunks:
        c = c.strip()
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out[:max_objects]


def first_token_ids(tokenizer, phrase: str, variants: str = "both") -> list[int]:
    """The ids VSC reads for one object.

    The paper takes ``o₀``, the first token of the object's tokenization.  With
    BPE that is ambiguous: "chair" and " chair" are different tokens, and which
    one a visual position would predict is not obvious.  ``variants="both"``
    (default) scores the space-prefixed and capitalised forms too and keeps the
    largest, which only ever helps a real object beat the floor;
    ``variants="plain"`` is the literal reading, for exact comparisons.
    """
    forms = [phrase]
    if variants == "both":
        forms += [" " + phrase, phrase.capitalize(), " " + phrase.capitalize()]
    ids: list[int] = []
    for f in forms:
        enc = tokenizer.encode(f, add_special_tokens=False)
        if enc and enc[0] not in ids:
            ids.append(int(enc[0]))
    return ids


# ---------------------------------------------------------------------------
# Pass A — prefill, and harvest the visual positions' word distributions
# ---------------------------------------------------------------------------

@dataclass
class Prefill:
    """What one prefill leaves behind for the rest of the method."""
    cache: object                     # past_key_values, resumable by generate()
    vis_hidden: torch.Tensor          # (m, hidden) float32 — the visual states
    vis_lse: torch.Tensor             # (m,) float32 — logsumexp per visual row
    span: tuple[int, int]
    n_prompt: int

    @property
    def m(self) -> int:
        return self.span[1] - self.span[0]


def _final_hidden_module(model):
    """The submodule whose output[0] is the last hidden state (pre-lm_head)."""
    inner = getattr(model, "model", None)
    if isinstance(inner, torch.nn.Module):
        return inner
    return None


@torch.no_grad()
def run_prefill(model, input_ids, forward_inputs: dict, span: tuple[int, int],
                cfg: VGAConfig | None = None) -> Prefill:
    """One forward pass that both fills the KV cache and captures visual states.

    We keep the *hidden* states rather than the logits: unembedding the whole
    prompt would allocate (T × vocab), and the visual rows' logits are all we
    need — they are recomputed from ``vis_hidden`` on demand, in chunks.  That
    also lets PVG gather an arbitrary token's column later for the price of one
    matrix-vector product.
    """
    captured: list[torch.Tensor] = []
    target = _final_hidden_module(model)
    if target is None:
        raise RuntimeError(
            "cannot locate the decoder wrapper (model.model) to capture hidden "
            "states from; add support in vga._final_hidden_module.")

    def _grab(module, args, output):
        h = output[0] if isinstance(output, (tuple, list)) or hasattr(output, "keys") else output
        captured.append(h.detach())

    handle = target.register_forward_hook(_grab)
    try:
        kw = dict(forward_inputs)
        kw["input_ids"] = input_ids
        kw["use_cache"] = True
        try:
            out = model(**kw, logits_to_keep=1)
        except TypeError:
            # A wrapper (PEFT, custom) without the argument: pay for the logits.
            out = model(**kw)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("the hidden-state hook never fired during prefill")

    s, e = span
    hidden = captured[-1]
    if hidden.shape[1] <= s:
        raise RuntimeError(
            f"captured hidden states are {tuple(hidden.shape)}, which does not "
            f"reach the visual span {span}")
    vis_hidden = hidden[0, s:e, :].float()

    lm_head = model.get_output_embeddings()
    lse = _row_logsumexp(lm_head, vis_hidden, (cfg or VGAConfig()).logit_chunk)

    return Prefill(cache=out.past_key_values, vis_hidden=vis_hidden, vis_lse=lse,
                   span=span, n_prompt=int(input_ids.shape[1]))


def _unembed(lm_head, h: torch.Tensor) -> torch.Tensor:
    w = lm_head.weight
    return lm_head(h.to(device=w.device, dtype=w.dtype)).float()


@torch.no_grad()
def _row_logsumexp(lm_head, vis_hidden: torch.Tensor, chunk: int) -> torch.Tensor:
    """logsumexp over the vocabulary for each visual row, chunked.

    Every probability this module needs is ``exp(logit − lse)``, so this one
    (m,) vector plus ``vis_hidden`` replaces the (m × vocab) softmax the paper
    writes down — which for a 1024-patch image would be ~600 MB.
    """
    outs = []
    for i in range(0, vis_hidden.shape[0], chunk):
        outs.append(torch.logsumexp(_unembed(lm_head, vis_hidden[i:i + chunk]), dim=-1))
    return torch.cat(outs)


# ---------------------------------------------------------------------------
# Building G
# ---------------------------------------------------------------------------

@torch.no_grad()
def object_map(model, pre: Prefill, token_ids_per_object: list[list[int]],
               cfg: VGAConfig) -> torch.Tensor:
    """Visual Semantic Confidence — the object-directed map.

        G_O[i] = softmax(logit_{v_i})[o₀]     i = 1..m
        G_O    = G_O / G_O.sum()
        G      = max_j G_{O_j}                 (elementwise, over objects)
        G      = G / G.sum()

    The per-object normalisation *before* the max is load-bearing: without it a
    common word would swamp a rare one purely through its prior.
    """
    lm_head = model.get_output_embeddings()
    flat = sorted({t for ids in token_ids_per_object for t in ids})
    if not flat:
        raise ValueError("no object token ids to score")
    index = torch.tensor(flat, device=lm_head.weight.device)

    cols = []
    for i in range(0, pre.vis_hidden.shape[0], cfg.logit_chunk):
        lg = _unembed(lm_head, pre.vis_hidden[i:i + cfg.logit_chunk])
        cols.append(lg.index_select(1, index))
    logits = torch.cat(cols)                                   # (m, n_ids)
    probs = torch.exp(logits - pre.vis_lse.to(logits.device).unsqueeze(1))

    at = {t: k for k, t in enumerate(flat)}
    per_object = []
    for ids in token_ids_per_object:
        if not ids:
            continue
        g = probs[:, [at[t] for t in ids]].max(dim=1).values     # variant max
        total = g.sum()
        if total <= 0:
            continue
        per_object.append(g / total)
    if not per_object:
        raise ValueError("every object scored zero across the visual span")

    g = torch.stack(per_object).max(dim=0).values
    return g / g.sum().clamp_min(1e-12)


@torch.no_grad()
def salience_map(model, pre: Prefill, cfg: VGAConfig) -> torch.Tensor:
    """Visual Semantic Salience — the object-agnostic map, for captioning.

        p    = topk(softmax(logit_{v_i}), K)
        G[i] = -Σ_k p_k · log(p_k) / log(K)
        G    = G / G.sum()

    VSS_ENTROPY_SIGN — READ THIS BEFORE TUNING.  The paper motivates VSS as
    scoring how *definite* a patch's semantics are, i.e. **low** entropy should
    mean **high** relevance; the formula upstream actually computes is plain
    normalised entropy, which scores the opposite way.  The wiki's instruction
    is "follow the code", so the code is what runs here — but the sign is
    genuinely unresolved, so ``cfg.vss_invert=True`` gives ``1 − H`` (the
    paper's stated intent) and any sweep of the agnostic mode should try both.
    The top-K probabilities are deliberately *not* renormalised to sum to 1,
    matching the formula as written; G is normalised afterwards regardless.
    """
    lm_head = model.get_output_embeddings()
    ent = []
    for i in range(0, pre.vis_hidden.shape[0], cfg.logit_chunk):
        lg = _unembed(lm_head, pre.vis_hidden[i:i + cfg.logit_chunk])
        top = lg.topk(cfg.topk, dim=-1).values
        p = torch.exp(top - pre.vis_lse[i:i + cfg.logit_chunk].to(lg.device).unsqueeze(1))
        p = p.clamp_min(1e-12)
        ent.append(-(p * p.log()).sum(dim=-1) / math.log(cfg.topk))
    g = torch.cat(ent)
    if cfg.vss_invert:
        g = (1.0 - g).clamp_min(0.0)
    return g / g.sum().clamp_min(1e-12)


@torch.no_grad()
def pvg_update(model, pre: Prefill, g: torch.Tensor, token_id: int,
               cfg: VGAConfig) -> torch.Tensor:
    """Progressive Visual Guidance — suppress what the caption already said.

        G_{t+1} = Norm(ReLU((1+λ)·G_t − λ·G_w))    where  G_w[i] = softmax(v_i)[w]

    Free-form captioning has moving visual demands, so a static G is wrong.
    Short-answer VQA does not need this; it is off by default.

    ``G_w`` as written is a raw softmax column (values ~1e-5) while ``G_t`` sums
    to 1 (values ~1/m), so at λ=0.02 the literal subtraction would do nothing at
    all.  ``cfg.pvg_normalize`` (default on) puts G_w on G_t's scale first,
    which is the only reading under which λ is a mixing coefficient; set it
    False for the literal formula.
    """
    w = model.get_output_embeddings().weight[token_id]
    logit = pre.vis_hidden.to(device=w.device, dtype=torch.float32) @ w.float()
    g_w = torch.exp(logit - pre.vis_lse.to(logit.device))
    if cfg.pvg_normalize:
        g_w = g_w / g_w.sum().clamp_min(1e-12)
    nxt = F.relu((1.0 + cfg.lam) * g.to(g_w.device) - cfg.lam * g_w)
    total = nxt.sum()
    return g if total <= 0 else nxt / total


# ---------------------------------------------------------------------------
# The injection
# ---------------------------------------------------------------------------

class VGA:
    """A live VGA installation.  Created by :func:`install`; call ``remove()``.

    Wraps ``model.generate``.  Per call it runs pass A (prefill, no injection),
    builds G, forms one ``Δz`` per hooked layer, then resumes generation with
    the ``o_proj`` hooks armed.
    """

    # Forward arguments that describe the *input*, as opposed to how to sample.
    _FORWARD_KEYS = ("attention_mask", "pixel_values", "pixel_values_videos",
                     "image_grid_thw", "video_grid_thw", "mm_token_type_ids")

    def __init__(self, model, processor, cfg: VGAConfig):
        self.model = model
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.cfg = cfg

        mc = _get_model_config(model)
        self.decoder_layers = mc["layers"]
        self.num_groups = mc["num_groups"]
        tcfg = _text_config(model)
        self.n_heads = int(tcfg.num_attention_heads)
        self.head_dim = int(getattr(tcfg, "head_dim",
                                    tcfg.hidden_size // tcfg.num_attention_heads))
        self.image_token_id = image_token_id(model, processor)

        wanted = cfg.layers(len(self.decoder_layers))
        self.layer_idx = [i for i in wanted if hasattr(self.decoder_layers[i], "self_attn")]
        if len(self.layer_idx) != len(wanted):
            _log(cfg, f"WARNING: {len(wanted) - len(self.layer_idx)} layer(s) in "
                      f"[{cfg.start_layer}, {cfg.end_layer}) have no self_attn and were skipped")
        if not self.layer_idx:
            raise ValueError(
                f"the layer window [{cfg.start_layer}, {cfg.end_layer}) selects no "
                f"layer of {len(self.decoder_layers)}")

        # Per-call state
        self._dz: dict[int, torch.Tensor] = {}
        self._v_vis: dict[int, torch.Tensor] = {}
        self._active = False
        self._handles: list = []
        self._prefill: Prefill | None = None
        self._g: torch.Tensor | None = None
        self._beta_eff = 0.0
        self._prompt_len = 0
        self._question: str | None = None
        self._objects: list[str] | None = None

        # Diagnostics — a run that silently did nothing is the expensive failure.
        self.stats = dict(calls=0, guided=0, no_image=0, injected_steps=0,
                          object_mode=0, agnostic_mode=0, fallbacks=0,
                          rel_update_sum=0.0, rel_update_n=0,
                          last_objects=None, last_mode=None, last_beta_eff=0.0,
                          last_g_entropy=None, last_g_max=None, last_m=None)

        if getattr(model, "_vga", None) is not None:
            raise RuntimeError(
                "VGA is already installed on this model. Stacking two installations "
                "would run two guidance passes and inject twice; call .remove() on "
                f"the existing one first (model._vga, beta={model._vga.cfg.beta}).")
        self._orig_generate = model.generate
        model.generate = self._generate
        model._vga = self
        self._installed = True

    # -- lifecycle ---------------------------------------------------------

    def remove(self) -> None:
        if not self._installed:
            return
        self._detach()
        self.model.generate = self._orig_generate
        if getattr(self.model, "_vga", None) is self:
            self.model._vga = None
        self._installed = False
        self._release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False

    # -- caller overrides --------------------------------------------------

    def set_question(self, text: str | None) -> None:
        """Give the exact question, instead of recovering it from the prompt."""
        self._question = text

    def set_objects(self, objects: list[str] | None) -> None:
        """Pin the objects, bypassing extraction entirely."""
        self._objects = objects

    # -- reporting ---------------------------------------------------------

    def diagnostics(self) -> dict:
        d = dict(self.stats)
        n = d.pop("rel_update_n") or 1
        d["mean_rel_update"] = d.pop("rel_update_sum") / n
        d["beta"] = self.cfg.beta
        d["layers"] = self.layer_idx
        return d

    def last_map(self, grid_hw: tuple[int, int] | None = None):
        """The most recent G, as a (grid_h, grid_w) array when a grid is given."""
        if self._g is None:
            return None
        g = self._g.detach().float().cpu().numpy()
        if grid_hw is None:
            return g
        h, w = grid_hw
        if h * w != g.size:
            return g
        return g.reshape(h, w)

    # -- generation --------------------------------------------------------

    def _generate(self, *args, **kwargs):
        self.stats["calls"] += 1
        if args:
            if "input_ids" in kwargs:
                raise TypeError("input_ids given both positionally and by keyword")
            kwargs["input_ids"] = args[0]
            args = args[1:]
        if args:
            raise TypeError("vga: model.generate takes at most one positional argument")

        ids = kwargs.get("input_ids")
        if ids is None:
            # inputs_embeds path: the visual span cannot be located from token ids.
            return self._orig_generate(**kwargs)
        if ids.shape[0] != 1:
            raise ValueError(
                f"vga needs batch_size=1, got {ids.shape[0]}: G is per-sample and the "
                "visual span is located from the sample's own token ids, so left "
                "padding in a wider batch moves every image column.")

        span = visual_span(ids, self.image_token_id)
        if span is None:
            self.stats["no_image"] += 1
            return self._orig_generate(**kwargs)

        try:
            self._prepare(ids, kwargs, span)
        except Exception as exc:                       # noqa: BLE001 — see below
            # A guidance failure must not be mistaken for a model answer, and
            # must not silently become an unguided run either.
            self._release()
            raise RuntimeError(f"vga could not build guidance for this sample: {exc}") from exc

        gen_kwargs = self._generation_kwargs(kwargs)
        self._attach()
        try:
            self._active = True
            out = self._orig_generate(**gen_kwargs)
        finally:
            self._active = False
            self._detach()
            self._release()
        self.stats["guided"] += 1
        return out

    def _prepare(self, ids, kwargs, span) -> None:
        cfg = self.cfg
        self._prompt_len = int(ids.shape[1])
        fwd = {k: kwargs[k] for k in self._FORWARD_KEYS if kwargs.get(k) is not None}

        prefill_ids = ids[:, :-1] if cfg.prefill_reuse else ids
        fwd_a = dict(fwd)
        if cfg.prefill_reuse:
            # Holding back the last token means every PER-TOKEN input has to be
            # held back with it, not just attention_mask: Qwen3-VL also requires
            # mm_token_type_ids, and an off-by-one there misaligns M-RoPE for the
            # whole prompt rather than failing.  Matched by shape so a future
            # per-token argument is handled too; the *_grid_thw tensors are
            # per-image and must not be touched.
            n = int(ids.shape[1])
            for k, val in list(fwd_a.items()):
                if k.endswith("_grid_thw") or not torch.is_tensor(val):
                    continue
                if val.dim() == 2 and val.shape[0] == ids.shape[0] and val.shape[1] == n:
                    fwd_a[k] = val[:, :-1]

        pre = run_prefill(self.model, prefill_ids, fwd_a, span, cfg)
        self._prefill = pre
        self._g = self._build_map(ids, pre)
        self._refresh_dz()

        self.stats["last_m"] = pre.m
        p = self._g.clamp_min(1e-12)
        self.stats["last_g_entropy"] = float(-(p * p.log()).sum() / math.log(pre.m))
        self.stats["last_g_max"] = float(self._g.max())

    def _build_map(self, ids, pre: Prefill) -> torch.Tensor:
        cfg = self.cfg
        mode = cfg.mode
        if mode in ("auto", "object"):
            objects = self._objects
            if objects is None:
                question = self._question or self._prompt_text(ids)
                objects = extract_objects(question, cfg.max_objects)
            token_ids = [first_token_ids(self.tokenizer, o, cfg.object_variants)
                         for o in objects]
            token_ids = [t for t in token_ids if t]
            if token_ids:
                self.stats.update(last_objects=list(objects), last_mode="object")
                self.stats["object_mode"] += 1
                _log(cfg, f"object-directed on {objects}")
                return object_map(self.model, pre, token_ids, cfg)
            if mode == "object":
                raise ValueError("mode='object' but no object could be extracted "
                                 "from the question; pass set_objects(...)")
            # The wiki asks for this fallback to be explicit, not a silent G=None.
            self.stats["fallbacks"] += 1
            _log(cfg, "no object extracted from the question — falling back to VSS")
        self.stats.update(last_objects=None, last_mode="agnostic")
        self.stats["agnostic_mode"] += 1
        return salience_map(self.model, pre, cfg)

    def _prompt_text(self, ids) -> str:
        """The last user turn, with the thousands of patch placeholders dropped."""
        tid = self.image_token_id
        vid = getattr(getattr(self.model, "config", None), "video_token_id", None)
        keep = [int(t) for t in ids[0].tolist() if t != tid and t != vid]
        text = self.tokenizer.decode(keep, skip_special_tokens=False)
        turns = re.findall(r"<\|im_start\|>\s*user\s*(.*?)<\|im_end\|>", text, re.S)
        seg = turns[-1] if turns else text
        seg = re.sub(r"<\|[^|>]*\|>", " ", seg)
        return " ".join(seg.split())

    @torch.no_grad()
    def _refresh_dz(self) -> None:
        """Δz = Gᵀ·V[s:e], per hooked layer, per head.  Constant across queries.

        PVG re-enters this once per generated token, so under PVG the sliced
        value block is kept rather than re-read: the visual positions' values
        were written during prefill and never change.
        """
        assert self._prefill is not None and self._g is not None
        s, e = self._prefill.span
        cache = self._prefill.cache
        g = self._g
        for li in self.layer_idx:
            v = self._v_vis.get(li)
            if v is None:
                v = _get_value_tensor(cache, li)                 # (B, H_kv, T, D)
                v = _repeat_kv(v[:, :, s:e, :], self.num_groups)  # (B, H_q,  m, D)
                if self.cfg.pvg:
                    self._v_vis[li] = v
            self._dz[li] = torch.einsum("m,bhmd->bhd",
                                        g.to(v.device, torch.float32),
                                        v.to(torch.float32))
        n_nonzero = int((g > self.cfg.sparsity_eps).sum())
        self._beta_eff = (self.cfg.beta * n_nonzero / g.numel()
                          if self.cfg.sparsity_scaling else self.cfg.beta)
        self.stats["last_beta_eff"] = float(self._beta_eff)

    def _generation_kwargs(self, kwargs: dict) -> dict:
        cfg = self.cfg
        out = {k: v for k, v in kwargs.items()
               if k not in self._FORWARD_KEYS and k != "past_key_values"}
        if kwargs.get("attention_mask") is not None:
            out["attention_mask"] = kwargs["attention_mask"]

        if cfg.prefill_reuse:
            # The prompt is already encoded; generate resumes from the cache and
            # only ever sees one uncached token.  pixel_values must NOT be
            # forwarded — the image tokens are behind us and re-embedding them
            # has nowhere to scatter to.
            out["past_key_values"] = self._prefill.cache
            out["use_cache"] = True
        else:
            for k in self._FORWARD_KEYS:
                if kwargs.get(k) is not None:
                    out[k] = kwargs[k]

        return out

    # -- hooks -------------------------------------------------------------

    def _attach(self) -> None:
        self._detach()
        for li in self.layer_idx:
            o_proj = self.decoder_layers[li].self_attn.o_proj
            self._handles.append(o_proj.register_forward_pre_hook(self._make_hook(li)))
        if self.cfg.pvg:
            self._handles.append(
                self.model.register_forward_pre_hook(self._pvg_hook, with_kwargs=True))

    def _pvg_hook(self, module, args, kwargs):
        """Fold the token the model just committed to into G, before this step runs.

        A LogitsProcessor is the obvious place for this and is wrong by one
        step: it is called *after* the step's forward, so a token generated at
        step t could not affect the map until step t+1.  A pre-hook on the whole
        model sees the step's own input, so ``G_{t+1}`` is in place for exactly
        the forward that should use it.

        Query positions before ``prompt_len`` are prompt tokens, not things the
        model said, and must not suppress anything — that includes the last
        prompt token, which the cache-resume path feeds through generate.
        """
        if not self._active or self._prefill is None or self._g is None:
            return None
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is None or ids.dim() != 2 or ids.shape[1] != 1:
            return None                                  # prefill, not a decode step
        pos = kwargs.get("cache_position")
        at = int(pos[0]) if pos is not None and len(pos) else None
        if at is None:
            cache = kwargs.get("past_key_values")
            at = cache.get_seq_length() if cache is not None else self._prompt_len
        if at < self._prompt_len:
            return None
        self._g = pvg_update(self.model, self._prefill, self._g, int(ids[0, -1]), self.cfg)
        self._refresh_dz()
        return None

    def _detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _release(self) -> None:
        self._dz = {}
        self._v_vis = {}
        self._prefill = None
        # _g survives for last_map(); it is (m,) floats, not the cache.

    def _make_hook(self, layer_idx: int):
        cfg = self.cfg

        def hook(module, args):
            if not self._active:
                return None
            dz = self._dz.get(layer_idx)
            if dz is None:
                return None
            z = args[0]
            if z.dim() != 3:
                return None
            b, t, hd = z.shape
            if not cfg.prefill_reuse and t > 1:
                # Fresh-prefill mode: the prompt's own positions are not answer
                # positions.  (The reference path injects at the last prompt
                # token as well; this one starts one token later.)
                return None

            h = self.n_heads
            d = hd // h
            zv = z.view(b, t, h, d).float()
            dzq = dz.to(zv.device).unsqueeze(1)                  # (B, 1, H, D)

            if cfg.head_balancing == "simg":
                # Weight guidance DOWN on heads already doing visual work and UP
                # on the rest: over-driving a specialised head damages what the
                # model does well.
                sim = F.cosine_similarity(dzq, zv, dim=-1)       # (B, T, H)
                w = (1.0 + sim) / 2.0
                w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12) * h
                gamma = F.relu(2.0 - w)
            else:
                gamma = torch.ones(b, t, h, device=zv.device, dtype=zv.dtype)

            coef = (self._beta_eff * gamma).unsqueeze(-1)        # (B, T, H, 1)
            if cfg.attn_norm:
                c = coef / (1.0 + coef)
                out = (1.0 - c) * zv + c * dzq
            else:
                out = zv + coef * dzq

            if layer_idx == self.layer_idx[0]:
                self.stats["injected_steps"] += 1
                denom = zv.norm().clamp_min(1e-12)
                self.stats["rel_update_sum"] += float((out - zv).norm() / denom)
                self.stats["rel_update_n"] += 1

            return (out.to(z.dtype).view(b, t, hd),) + tuple(args[1:])

        return hook


def install(model, processor, config: VGAConfig | None = None, **overrides) -> VGA:
    """Install VGA on ``model``.  Returns the handle; call ``.remove()`` when done.

    ``install(model, processor, beta=0.25, start_layer=2, end_layer=18)`` works
    as shorthand for building the config inline.
    """
    cfg = config or VGAConfig()
    if overrides:
        cfg = VGAConfig(**{**cfg.__dict__, **overrides})
    vga = VGA(model, processor, cfg)
    if cfg.beta == 0:
        # The one mistake worth a whole wasted benchmark: an arm's directory
        # holding the stock model's scores.
        _log(cfg, "WARNING: beta=0 — the injection is the identity and this run "
                  "is the stock model.")
    _log(cfg, f"installed: beta={cfg.beta} layers={vga.layer_idx} "
              f"mode={cfg.mode} head_balancing={cfg.head_balancing} "
              f"pvg={cfg.pvg} prefill_reuse={cfg.prefill_reuse}")
    return vga
