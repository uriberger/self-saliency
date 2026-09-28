#!/usr/bin/env python
"""Everything the sink-location experiment knows about the model it is measuring.

`docs/sink-location-by-image-type.md` is the design and `sink_location.py` is the
measurement. Both were written against Qwen3-VL, and the question
`docs/sink-location-cross-model.md` asks is whether the result is a property of that
model or of VLMs -- which cannot be answered without running the SAME measurement on a
different family. This module is the seam: one class per family, selected off
`config.model_type`, holding the handful of facts that differ.

WHAT DIFFERS, AND WHY EACH ONE IS HERE RATHER THAN BRANCHED INLINE

    the image token, and the delimiters around it   locating the picture in the prompt
    the patch grid                                  Qwen3-VL varies it per picture;
                                                    InternVL and LLaVA-1.5 are fixed
    the decoder's attention class                   which modules the scan installs on
    the module that emits the LLM-facing rows       where the permutation hook goes
    the VIEW BOX                                    see below
    the prompt                                      each model's own chat template

THE VIEW BOX IS THE ONE THAT WILL BITE YOU. Qwen3-VL and InternVL hand the whole picture
to the encoder; LLaVA-1.5's processor resizes the short side to 336 and then CENTRE-CROPS
to 336x336, so its 24x24 grid covers a centred square of the picture and nothing else.
`view_box` is that square in normalised picture coordinates, and every piece of geometry
that relates a pixel to a patch -- the per-patch content statistics, the frame check, the
arms' `patch_correspondence` -- composes through it. Without it, `follow content` on a
rotated picture would be decoded in the wrong frame and would answer the
content-versus-position question confidently and backwards, which is exactly the failure
mode the selftest exists to catch.

    import vlm_family as VF
    fam = VF.family_for(model, processor)     # -> Qwen3VL() / InternVL() / Llava15()
    scan = SL.install(model, family=fam)
"""

from __future__ import annotations

import numpy as np

# The families, by `config.model_type`. `family_for` is the only way to get one, so a
# model this file has never heard of fails loudly at the seam rather than silently
# measuring a Qwen3-VL-shaped hole in a different model.
_REGISTRY = {}


def register(cls):
    for t in cls.model_types:
        _REGISTRY[t] = cls
    return cls


def family_for(model=None, processor=None, config=None, model_type=None):
    """The adapter for this model. -> a bound Family instance."""
    if model_type is None:
        cfg = config if config is not None else getattr(model, "config", None)
        model_type = getattr(cfg, "model_type", None)
    cls = _REGISTRY.get(model_type)
    if cls is None:
        raise SystemExit(
            f"sink_location has no adapter for model_type {model_type!r}; "
            f"known families are {sorted(_REGISTRY)}. Add one to vlm_family.py -- do not "
            "branch on the model inside the measurement.")
    return cls().bind(model=model, processor=processor, config=config)


#: The decoder attention class for each text tower a connector-style VLM can be built on.
#: Several of these architectures are SOCKETS, not models -- the same
#: `LlavaForConditionalGeneration` holds Vicuna + CLIP in LLaVA-1.5 and Qwen2 + SigLIP in
#: llava-interleave-qwen -- so the class has to be read off the text config rather than
#: assumed. Guessing installs the scan on nothing, and a model that reports empty cells
#: looks exactly like a model with no effect.
TEXT_ATTENTION = {
    "llama": "LlamaAttention", "qwen2": "Qwen2Attention", "qwen3": "Qwen3Attention",
    "mistral": "MistralAttention", "gemma": "GemmaAttention",
    "gemma2": "Gemma2Attention", "gemma3_text": "Gemma3Attention",
    "phi3": "Phi3Attention", "olmo2": "Olmo2Attention",
    # A Mamba-Transformer hybrid: only the layers marked '*' in its
    # `hybrid_override_pattern` have attention at all. See NemotronVL.
    "nemotron_h": "NemotronHAttention",
}


def text_attention(config, default):
    mt = getattr(getattr(config, "text_config", None), "model_type", None)
    if not mt:
        return default
    cls = TEXT_ATTENTION.get(mt)
    if cls is None:
        raise SystemExit(
            f"a {mt!r} text tower: add its attention class to vlm_family.TEXT_ATTENTION. "
            "Guessing would install the scan on nothing and report empty cells as a "
            "result.")
    return (cls,)


# ---------------------------------------------------------------------------
class Family:
    """The surface `sink_location.py` is allowed to know about a model.

    Subclasses fill in the class attributes and override only the methods whose answer
    is not the common one. Every default here is the answer that is true for a model
    which shows the encoder the whole picture on a fixed grid, which is two of the three.
    """

    name = ""
    model_types = ()
    #: the decoder's attention module class(es). The scan registers its own attention
    #: implementation and switches these over to it; the vision tower's attention has a
    #: different class name in all three families, which is what keeps it untouched.
    attn_classes = ()
    #: the module whose forward OUTPUT carries the rows the language model will consume,
    #: one row per image token, in token order. That is where A9's permutation goes.
    row_classes = ()
    #: True when this family should be run with the project's own trainer system prompt
    #: (so the published Qwen3-VL numbers reproduce), False when it gets its own.
    uses_project_prompt = False

    image_token_id = None
    vision_start_ids = ()
    vision_end_ids = ()
    #: token strings to resolve against the tokenizer at bind time, when the ids are not
    #: a fixed part of the family.
    start_tokens = ()
    end_tokens = ()
    #: True when the processor stacks a multi-image batch's `pixel_values` into one dense
    #: tensor and therefore cannot take two pictures of different sizes. Only the A6
    #: two-image arm ever sends more than one, and it resizes the PARTNER to match rather
    #: than zero-padding either -- padding would staple a black border onto a picture in
    #: an experiment about borders.
    batch_needs_equal_size = False

    def __init__(self):
        self.system_prompt = None
        self.config = None
        self.processor = None

    # -- binding ---------------------------------------------------------
    def bind(self, model=None, processor=None, config=None):
        self.config = config if config is not None else getattr(model, "config", None)
        self.processor = processor
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            if self.start_tokens:
                self.vision_start_ids = self._ids(tok, self.start_tokens)
            if self.end_tokens:
                self.vision_end_ids = self._ids(tok, self.end_tokens)
        if self.image_token_id is None and self.config is not None:
            got = getattr(self.config, "image_token_id", None)
            if got is None:
                got = getattr(self.config, "image_token_index", None)
            self.image_token_id = None if got is None else int(got)
        return self

    @staticmethod
    def _ids(tok, names):
        out = []
        for n in names:
            i = tok.convert_tokens_to_ids(n)
            if i is not None and i >= 0:
                out.append(int(i))
        return tuple(out)

    # -- the prompt ------------------------------------------------------
    #: Processor keyword arguments this family must pin. Empty for most; see InternVL.
    proc_defaults = {}
    #: Processor OUTPUTS that the model's forward does not accept. Empty for most.
    drop_inputs = ()

    def model_inputs(self, inputs):
        """The processor's output, reduced to what `forward()` will accept."""
        if not self.drop_inputs:
            return inputs
        return {k: v for k, v in inputs.items() if k not in self.drop_inputs}

    #: Keys that `forward()` needs but `generate()` rejects. `generate` validates its
    #: kwargs against the LANGUAGE model's signature, so a multimodal-wrapper argument
    #: that the wrapper consumes itself is an error there and required here.
    drop_for_generate = ()

    def generate_inputs(self, inputs):
        """The same inputs, reduced to what `generate()` will accept."""
        if not self.drop_for_generate:
            return inputs
        return {k: v for k, v in inputs.items() if k not in self.drop_for_generate}

    def image_arg(self, images):
        """How this processor wants the pictures: flat, or nested one list per sample."""
        return list(images)

    def build_inputs(self, processor, images, question, device, **proc_kwargs):
        """The prompt, at batch size 1, with one or more pictures.

        Each model gets its OWN chat template -- putting Qwen3-VL's `<think>` system
        prompt in front of LLaVA-1.5 would measure an off-distribution model, and the
        thing being compared is where attention lands, which is a property of the model
        and not of our prompt conventions. `--system-prompt none` is what puts all three
        on the same footing when the cross-model tables are produced.
        """
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": question})
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": content})
        text = processor.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        kw = dict(self.proc_defaults)
        kw.update(proc_kwargs)
        return processor(text=[text], images=self.image_arg(images),
                         return_tensors="pt", padding=True, padding_side="left",
                         add_special_tokens=False, **kw).to(device)

    def teacher_forced_case(self, prompt_inputs, comp_ids, device):
        """prompt ++ one completion, as the single measured forward.

        The multimodal inputs are carried over verbatim and the completion is appended as
        text. Families with extra per-token multimodal bookkeeping (Qwen3-VL's
        `mm_token_type_ids`) extend it; forgetting to would make this forward differ from
        the one the training run computes its reward on.
        """
        import torch

        ids = torch.cat([prompt_inputs["input_ids"],
                         torch.tensor([comp_ids], device=device)], dim=1)
        case = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        for k in self.passthrough_inputs:
            if prompt_inputs.get(k) is not None:
                case[k] = prompt_inputs[k]
        return case

    passthrough_inputs = ("pixel_values",)

    # -- the grid --------------------------------------------------------
    #: tokens per tile as (gh, gw), for families whose grid does not depend on the
    #: picture. `grids_for` uses it and never has to ask the processor.
    fixed_grid = None
    #: one grid cell, in the ENCODER's own input pixels. A10 cuts its blocks at exactly
    #: this size so the processor's resize is the identity and the shuffle is lossless.
    encoder_px = 32

    def grids_for(self, runs, inputs):
        """-> [(t, gh, gw)] one entry per TILE, in column order.

        One tile per picture in every family except InternVL with tiling switched on,
        where a picture becomes up to 12 tiles plus a thumbnail and each tile is its own
        16x16 grid. Returning tiles rather than pictures is what lets the tiled arm be
        scored on the geometry the encoder actually saw.
        """
        gh, gw = self.fixed_grid
        out = []
        for run in runs:
            n = int(run.numel())
            if n % (gh * gw):
                raise RuntimeError(
                    f"{self.name}: an image run of {n} tokens is not a multiple of the "
                    f"{gh}x{gw} tile this family emits")
            out += [(1, gh, gw)] * (n // (gh * gw))
        return out

    def grid_of(self, processor, image):
        """The grid THIS picture will get. Used by the frame check, per transform."""
        return tuple(self.fixed_grid)

    def view_box(self, image):
        """The part of the picture the patch grid covers, normalised. (u0, v0, u1, v1).

        The whole picture, unless the processor crops.
        """
        return (0.0, 0.0, 1.0, 1.0)

    def patch_px(self, image):
        """One grid cell, in the PICTURE's own pixels. Used by the padding arms.

        Derived from the view rather than assumed, because a padded picture is only 'one
        patch of border' if the pad is the width of a patch in the frame the grid uses.
        Needs no processor: every family whose grid depends on the picture overrides it.
        """
        u0, v0, u1, v1 = self.view_box(image)
        W, H = image.size
        gh, gw = self.fixed_grid
        return max(1, int(round(min((u1 - u0) * W / max(1, gw),
                                    (v1 - v0) * H / max(1, gh)))))

    # -- the vision side -------------------------------------------------
    #: An attribute path to the row module, for families whose projector is a bare
    #: `nn.Sequential` -- the class name "Sequential" would match dozens of modules, so
    #: the path is the only unambiguous handle. Tried before the class-name search.
    row_attr = None

    def attention_layers(self, model):
        """Which decoder layer indices have an attention matrix. None = all of them.

        Dense decoders answer None and the scan checks 0..n-1. A hybrid cannot: most of
        its layers are state-space or MLP blocks with no attention at all, so "the scan
        saw fewer layers than the model has" is the correct outcome there and a bug
        anywhere else. Returning the real list is what keeps that distinction.
        """
        return None

    def row_module(self, model):
        """The module whose output holds the LLM-facing image rows."""
        if self.row_attr:
            m = model
            for part in self.row_attr.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is not None:
                return m
        for m in model.modules():
            if type(m).__name__ in self.row_classes:
                return m
        raise RuntimeError(f"{self.name}: no module at {self.row_attr!r} and none of "
                           f"{self.row_classes} on this model")

    def permute_rows(self, out, make_perm):
        """Apply a row permutation to that module's output. -> (new out, perm).

        The default is for a module that returns a plain [batch, tokens, dim] tensor,
        which is what both projectors do. `batch` is the TILE axis, and the rows reach
        the language model in row-major order, so flattening before permuting is what
        makes a tiled picture's permutation a permutation of its tokens.
        """
        import torch  # noqa: F401

        flat = out.reshape(-1, out.shape[-1])
        perm = make_perm(int(flat.shape[0]), out.device)
        return flat[perm].reshape(out.shape), perm

    def row_norms(self, out):
        """||row|| per image token, as a numpy array in token order."""
        return out.reshape(-1, out.shape[-1]).detach().float().norm(dim=-1).cpu().numpy()

    def deepstack_norms(self, out):
        """Norms at any extra injection point this family has. -> list or None."""
        return None


# ---------------------------------------------------------------------------
@register
class Qwen3VL(Family):
    """Native resolution, a per-picture grid, and three extra injection points.

    The published result (`docs/sink-location-by-image-type.md` §16-17) is this family,
    so nothing here may change: an adapter that silently moves the baseline invalidates
    the comparison the other two families exist to make.
    """

    name = "qwen3_vl"
    model_types = ("qwen3_vl",)
    attn_classes = ("Qwen3VLTextAttention",)
    row_classes = ("Qwen3VLVisionModel", "Qwen3VLVisionTransformerPretrainedModel")
    uses_project_prompt = True
    image_token_id = 151655              # <|image_pad|>
    vision_start_ids = (151652,)
    vision_end_ids = (151653,)
    passthrough_inputs = ("pixel_values", "image_grid_thw")

    def image_arg(self, images):
        # one list per sample, which is what this processor's batching expects
        return [list(images)]

    def teacher_forced_case(self, prompt_inputs, comp_ids, device):
        import torch

        case = super().teacher_forced_case(prompt_inputs, comp_ids, device)
        if prompt_inputs.get("mm_token_type_ids") is not None:
            zeros = torch.zeros(1, len(comp_ids), dtype=torch.long, device=device)
            case["mm_token_type_ids"] = torch.cat(
                [prompt_inputs["mm_token_type_ids"], zeros], dim=1)
        return case

    def grids_for(self, runs, inputs):
        thw = inputs if not hasattr(inputs, "get") else inputs.get("image_grid_thw")
        if thw is None or len(thw) != len(runs):
            raise RuntimeError(
                f"{len(runs)} image token runs but "
                f"{0 if thw is None else len(thw)} grids: refusing to guess which "
                "picture is which")
        out = []
        for run, g in zip(runs, thw):
            t, h, w = (int(x) for x in g)
            gh, gw = h // 2, w // 2          # this processor merges 2x2 patches per token
            if int(run.numel()) != t * gh * gw:
                raise RuntimeError(
                    f"image run of {int(run.numel())} tokens against a {t}x{gh}x{gw} "
                    "grid: the patch merge assumption is wrong for this model")
            out.append((t, gh, gw))
        return out

    def grid_of(self, processor, image):
        got = processor(text=["x"], images=[[image]], return_tensors="pt",
                        add_special_tokens=False)
        t, h, w = (int(x) for x in got["image_grid_thw"][0])
        return (h // 2, w // 2)

    def patch_px(self, image):
        # 32 exactly -- this processor's own token is 2x2 patches of 16px, and the
        # picture's size is rounded to a multiple of it, so deriving it from the
        # picture's width would round to 31 on some sizes and change the padding arms
        # away from the published run.
        return 32

    def permute_rows(self, out, make_perm):
        pool = out.pooler_output
        perm = make_perm(int(pool.shape[0]), pool.device)
        out.pooler_output = pool[perm]
        feats = getattr(out, "deepstack_features", None)
        if feats:
            # The three DeepStack features are injected into the LLM's early layers with
            # the same row indexing, so the permutation is only honest if all three move
            # with it. This is the piece the other two families do not have.
            out.deepstack_features = [f[perm] for f in feats]
        return out, perm

    def row_norms(self, out):
        pool = getattr(out, "pooler_output", None)
        if pool is None:
            return None
        return pool.detach().float().norm(dim=-1).cpu().numpy()

    def deepstack_norms(self, out):
        feats = getattr(out, "deepstack_features", None)
        if not feats:
            return None
        return [f.detach().float().norm(dim=-1).cpu().numpy() for f in feats]


# ---------------------------------------------------------------------------
@register
class Glm4V(Qwen3VL):
    """GLM-4.1V -- a third language-model family, on a native-resolution tower of its own.

    Subclassed off Qwen3-VL because the two really are the same shape where this module
    touches them, and re-typing that would be two copies to keep in step rather than one:
    a native-resolution ViT whose grid comes from `image_grid_thw`, a 2x2 spatial merge so
    the token grid is (h//2, w//2), and a vision model that hands the language model its
    rows as `pooler_output`. What differs is the identifiers, the 14px patch (so a token
    is 28px, not Qwen3-VL's 32), and the absence of DeepStack -- a single injection point,
    like InternVL and LLaVA, which is one fewer place a positional mark can be planted.

    The decoder is GLM-4, 40 layers by 32 heads: neither Qwen nor Llama, which is the
    point. It is also a reasoning model, so its `generated` query set is a long chain
    rather than the handful of tokens the instruct models write.
    """

    name = "glm4v"
    model_types = ("glm4v",)
    attn_classes = ("Glm4vTextAttention",)
    row_classes = ("Glm4vVisionModel",)
    uses_project_prompt = False
    image_token_id = 151343
    vision_start_ids = (151339,)          # <|begin_of_image|>
    vision_end_ids = (151340,)            # <|end_of_image|>
    encoder_px = 28                       # patch 14 x spatial_merge 2

    def patch_px(self, image):
        # 28 exactly, for the reason Qwen3-VL's is 32: the picture's own width is a
        # rounded multiple of the token size, so deriving it would round differently on
        # some sizes and quietly change what the padding arms pad by.
        return 28


# ---------------------------------------------------------------------------
@register
class InternVL(Family):
    """A Qwen3 text tower with someone else's eyes -- the best-controlled comparison.

    InternVL3.5-8B's language model is Qwen3, 36 layers and 32 heads, the same shape as
    Qwen3-VL-8B's. So this family holds the LLM nearly fixed and swaps the vision encoder
    and the connector: if the ring survives, it is not about the language model; if it
    dies, the encoder is implicated directly.

    TILING, AND THE TRAP INSIDE IT. `max_dynamic_patch: 12` means a picture can become
    twelve 448px tiles plus a thumbnail, each its own 16x16 grid, and then "the outer
    ring" is ambiguous -- the ring of a tile is an interior edge of the picture. The
    primary comparison therefore runs with tiling OFF, one 16x16 grid whose geometry
    matches Qwen3-VL's, and the tiled configuration is a separate, clearly labelled arm.

    `crop_to_patches` HAS TO BE PASSED. `image_processor.crop_to_patches` is False on
    this repo, and the processor tiles anyway -- `InternVLProcessor.__call__` carries its
    own default and overrides the attribute. Reading the attribute and believing it gives
    seven tiles where the analysis assumes one, and every column statistic then describes
    the top-left corner of the picture while claiming to describe the picture. That is
    what `proc_defaults` pins, and the selftest's frame check is what would have caught
    it if it had not been caught here.
    """

    name = "internvl"
    model_types = ("internvl",)
    attn_classes = ("Qwen3Attention",)             # the text tower IS Qwen3
    row_classes = ("InternVLMultiModalProjector",)
    start_tokens = ("<img>",)
    end_tokens = ("</img>",)
    fixed_grid = (16, 16)                          # 448/14 = 32, pixel-shuffled by 0.5
    proc_defaults = {"crop_to_patches": False}

    def bind(self, model=None, processor=None, config=None):
        super().bind(model=model, processor=processor, config=config)
        cfg = self.config
        if cfg is not None:
            # 256 tokens per tile is `image_seq_length`; derive rather than assume, so a
            # different downsample_ratio fails here instead of mislabelling the geometry.
            n = int(getattr(cfg, "image_seq_length", 256))
            side = int(round(n ** 0.5))
            if side * side != n:
                raise SystemExit(f"internvl: {n} tokens per tile is not a square grid")
            self.fixed_grid = (side, side)
            vc = getattr(cfg, "vision_config", None)
            px = getattr(vc, "image_size", 448) if vc is not None else 448
            self.encoder_px = int((px[0] if isinstance(px, (list, tuple)) else px) / side)
        return self


# ---------------------------------------------------------------------------
@register
class Llava15(Family):
    """A frozen CLIP encoder at its native resolution -- the best falsification target.

    LLaVA-1.5 is the clean case geometrically (fixed 336px, one 24x24 grid, no tiling)
    and the hard case for the claim: the literature reports a BOTTOM-of-image bias for
    this family (MCA-LLaVA, VisPruner), its CLIP tower is frozen at the resolution it was
    trained at so it barely interpolates position embeddings, and its connector is a
    two-layer MLP. If the ring is going to fail anywhere, it is here.

    THE CENTRE CROP. The stock `llava-hf` processor resizes the short side to 336 and
    centre-crops to 336x336 -- which is how LLaVA-1.5 is normally run, and is therefore
    what this uses. The consequence is that the 24x24 grid covers a centred square of the
    picture, not the picture, and `view_box` is that square. Every pixel-to-patch
    statement in the experiment composes through it.
    """

    name = "llava"
    model_types = ("llava",)
    attn_classes = ("LlamaAttention",)
    row_classes = ("LlavaMultiModalProjector",)
    # No delimiter tokens at all: `<image>` expands in place, with nothing around it.
    # `span_index` reports an empty vision_start/vision_end span, which is a fact about
    # this model rather than a gap in the measurement.
    fixed_grid = (24, 24)                          # 336/14

    def bind(self, model=None, processor=None, config=None):
        super().bind(model=model, processor=processor, config=config)
        vc = getattr(self.config, "vision_config", None)
        if vc is not None:
            px = int(getattr(vc, "patch_size", 14))
            side = int(getattr(vc, "image_size", 336)) // px
            self.fixed_grid, self.encoder_px = (side, side), px
        self.attn_classes = text_attention(self.config, self.attn_classes)
        return self

    # -- the centre crop, in closed form ---------------------------------
    def view_box(self, image):
        """The part of the picture this processor's grid covers.

        TWO CASES, and reading the wrong one puts every patch statistic in the wrong
        frame. `size` with a `shortest_edge` plus `do_center_crop` -- LLaVA-1.5's
        CLIPImageProcessor -- resizes the short side and CENTRE-CROPS, so the grid covers
        a centred square. `size` with an explicit height and width -- SigLIP's processor,
        which llava-interleave-qwen uses -- resizes the whole picture to that square, so
        the grid covers all of it.

        The crop case reproduces `CLIPImageProcessor`'s arithmetic exactly rather than
        approximating it: the short side goes to `shortest_edge`, the long side is
        TRUNCATED, then `center_crop` takes `(size - crop) // 2` off the top and the
        left. A half-pixel of slop here is harmless; getting the direction wrong is not.
        """
        ip = getattr(self.processor, "image_processor", None)
        size, crop = getattr(ip, "size", None), getattr(ip, "crop_size", None)
        short = _size_get(size, "shortest_edge")
        if short is None or not getattr(ip, "do_center_crop", False):
            return (0.0, 0.0, 1.0, 1.0)          # resized to a square: the whole picture
        short = int(short)
        ch = int(_size_get(crop, "height") or short)
        cw = int(_size_get(crop, "width") or short)
        W, H = image.size
        if W <= H:
            nw, nh = short, int(short * H / W)
        else:
            nh, nw = short, int(short * W / H)
        left, top = (nw - cw) // 2, (nh - ch) // 2
        return (left / nw, top / nh, (left + cw) / nw, (top + ch) / nh)


# ---------------------------------------------------------------------------
@register
class Idefics3(Family):
    """A Llama-3 text tower on a SigLIP encoder -- the cell the other four do not fill.

    Qwen3-VL, InternVL3.5 and llava-interleave-qwen all run a Qwen language model, and
    LLaVA-1.5 is the only non-Qwen one, which leaves "the language model's queries favour
    border keys" confounded with everything that differs between those checkpoints. This
    is a second non-Qwen decoder -- Llama-3-8B, 32 layers by 32 heads -- on a SigLIP tower
    that WAS trained inside the VLM, which is the combination none of the others has.

    SPLITTING, like InternVL's tiling: `do_image_splitting` is True by default and cuts a
    picture into sub-images plus a global view. Pinned off, so one 13x13 grid covers the
    whole picture. The processor then resizes to 364x364 with no padding -- checked, the
    pixel attention mask comes back fully valid -- so the view box is the whole picture
    and the border of the grid really is the border of the image.
    """

    name = "idefics3"
    model_types = ("idefics3", "smolvlm")
    attn_classes = ("LlamaAttention",)
    row_classes = ("Idefics3Connector", "SmolVLMConnector")
    # One token is used on BOTH sides of the picture, so it is reported entirely in
    # `vision_start` rather than counted twice by naming it as the closer as well.
    start_tokens = ("<fake_token_around_image>",)
    fixed_grid = (13, 13)
    encoder_px = 28
    proc_defaults = {"do_image_splitting": False}

    def image_arg(self, images):
        return [list(images)]

    def bind(self, model=None, processor=None, config=None):
        super().bind(model=model, processor=processor, config=config)
        cfg = self.config
        vc = getattr(cfg, "vision_config", None)
        if vc is not None:
            px = int(getattr(vc, "patch_size", 14))
            # pixel shuffle by `scale_factor` in the connector, exactly as InternVL's
            # 2x2 shuffle does: 26x26 patches of 14px become 13x13 tokens of 28px.
            sf = int(getattr(cfg, "scale_factor", 2))
            side = (int(getattr(vc, "image_size", 364)) // px) // sf
            self.fixed_grid, self.encoder_px = (side, side), px * sf
        self.attn_classes = text_attention(cfg, self.attn_classes)
        return self


# ---------------------------------------------------------------------------
@register
class NemotronVL(Family):
    """NVIDIA's VLMs, on the RADIO tower -- a fourth encoder lineage, and a hybrid decoder.

    Two checkpoints, one family. `Llama-3.1-Nemotron-Nano-VL-8B-V1` is a plain Llama-3.1
    decoder, 32 layers all attention. `Nemotron-3-Nano-Omni-30B-A3B` is a MAMBA-TRANSFORMER
    HYBRID whose `hybrid_override_pattern` is

        MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
        M = Mamba (23)      E = MoE (23)      * = attention (6)

    so **only 6 of its 52 decoder layers have an attention matrix at all**. Everything
    this module measures is defined per (layer, head), and for the other 46 layers there
    is no such thing. The scan therefore reports 6 x 32 = 192 cells for it rather than the
    ~1,152 the dense models give, and that row is a different claim -- "where the only
    attention layers look" -- not a smaller sample of the same one. It is labelled, never
    pooled.

    GEOMETRY. Despite an InternVL-shaped config (`force_image_size`, `downsample_ratio`,
    `use_thumbnail`) this does NOT tile: the processor buckets the picture to an
    aspect-matched resolution and emits one grid, so the token grid is (H//32, W//32) from
    `imgs_sizes` -- a 16px patch with a 2x2 shuffle, the same 32px token as Qwen3-VL.

    RADIO is the reason to want it: distilled from CLIP ViT-H/14, SigLIP-SO400M, DINOv2-g
    WITH REGISTERS, and SAM -- the only tower in the panel whose ancestry includes the
    model Darcet et al. found registers in.
    """

    name = "nemotron_vl"
    model_types = ("NemotronH_Nano_Omni_Reasoning_V3", "Llama_Nemotron_Nano_VL")
    attn_classes = ("NemotronHAttention",)
    row_attr = "mlp1"                     # an nn.Sequential; the class name is useless
    row_classes = ()
    start_tokens = ("<img>",)
    end_tokens = ("</img>",)
    encoder_px = 32                       # patch 16 x the 2x2 pixel shuffle
    #: RADIO's own ViT also has modules called `Attention`; the decoder's class is what
    #: `bind` resolves off the text config, so the tower is never switched over.
    passthrough_inputs = ("pixel_values", "image_flags")
    #: Processor outputs that `forward()` does not accept. They describe the geometry
    #: rather than feed the model, and passing them through raises TypeError.
    drop_inputs = ("num_patches", "num_tokens", "imgs_sizes")
    #: `image_flags` is consumed by the wrapper's own forward and is not a kwarg the
    #: language model's `generate` will accept.
    drop_for_generate = ("image_flags",)
    #: Its processor hands `pixel_values` to `BatchFeature` as a LIST and lets that stack
    #: them, so two pictures of different sizes raise. RADIO is native-resolution, so on
    #: this corpus that is nearly every pair: the A6 arm took down a whole 8-GPU run 8
    #: pictures in before this flag existed.
    batch_needs_equal_size = True

    def build_inputs(self, processor, images, question, device, **proc_kwargs):
        out = super().build_inputs(processor, images, question, device, **proc_kwargs)
        # The grid lives in `imgs_sizes`, which is NOT a forward kwarg and so is stripped
        # before the model is called -- which means the scan's pre-hook never sees it.
        # Cached here, at the one place that knows the pictures and the sizes belong
        # together, and read back by `grids_for`. Safe only because this module is batch
        # size 1 throughout and `build_inputs` immediately precedes the forward; both are
        # enforced elsewhere, and `grids_for` re-checks the run length against the grid
        # it derives, so a stale cache fails loudly rather than mislabelling patches.
        self._sizes = out.get("imgs_sizes")
        # `image_flags` marks which of the encoder's outputs are real pictures. The
        # forward requires it -- `image_flags.squeeze(-1)` on the way in -- and this
        # processor does not emit it, so it is synthesised here as [N, 1] ones, N being
        # the number of images the encoder was given. Shape matters: squeeze(-1) on a
        # bare [N] would collapse a single image to a 0-d tensor.
        if out.get("image_flags") is None and out.get("pixel_values") is not None:
            import torch
            n = int(out["pixel_values"].shape[0])
            out["image_flags"] = torch.ones(n, 1, dtype=torch.long,
                                            device=out["pixel_values"].device)
        return out

    def bind(self, model=None, processor=None, config=None):
        super().bind(model=model, processor=processor, config=config)
        cfg = self.config
        if self.image_token_id is None and cfg is not None:
            got = getattr(cfg, "img_context_token_id", None)
            if got is None and processor is not None:
                tok = getattr(processor, "tokenizer", None) or processor
                name = getattr(cfg, "img_context_token", None) or "<image>"
                got = tok.convert_tokens_to_ids(name)
            self.image_token_id = None if got is None else int(got)
        self.attn_classes = text_attention(cfg, self.attn_classes)
        return self

    def grids_for(self, runs, inputs):
        """The token grid, from the size the processor actually resized to.

        `imgs_sizes` is the resized (H, W) per picture and the grid is that over the 32px
        token. Derived rather than assumed, because this processor buckets to an
        aspect-matched resolution instead of a fixed square -- 512x320 and 352x224 both
        come back 416x672, and a fixed-grid assumption would be right for neither.
        """
        sizes = inputs if not hasattr(inputs, "get") else inputs.get("imgs_sizes")
        if sizes is None:
            sizes = getattr(self, "_sizes", None)      # see build_inputs
        if sizes is None or len(sizes) != len(runs):
            raise RuntimeError(
                f"{len(runs)} image token runs but "
                f"{0 if sizes is None else len(sizes)} sizes: this family reads its grid "
                "from `imgs_sizes` and will not guess it")
        out = []
        for run, hw in zip(runs, sizes):
            h, w = (int(x) for x in hw)
            gh, gw = h // self.encoder_px, w // self.encoder_px
            if int(run.numel()) != gh * gw:
                raise RuntimeError(
                    f"image run of {int(run.numel())} tokens against a {gh}x{gw} grid "
                    f"derived from {h}x{w}: the {self.encoder_px}px token assumption is "
                    "wrong for this checkpoint")
            out.append((1, gh, gw))
        return out

    def grid_of(self, processor, image):
        got = processor(text=["<image>"], images=[image], return_tensors="pt")
        h, w = (int(x) for x in got["imgs_sizes"][0])
        return (h // self.encoder_px, w // self.encoder_px)

    def patch_px(self, image):
        return self.encoder_px

    def attention_layers(self, model):
        """The `*` positions of `hybrid_override_pattern`, read off the loaded model.

        Read from the modules rather than from the pattern string, so it reports what
        was actually built. On the 8B -- a plain Llama decoder -- this returns all 32 and
        is equivalent to the dense answer.
        """
        idx = sorted(int(m.layer_idx) for m in model.modules()
                     if type(m).__name__ in self.attn_classes
                     and getattr(m, "layer_idx", None) is not None)
        return idx or None


# ---------------------------------------------------------------------------
@register
class NemotronVLV2(NemotronVL):
    """`NVIDIA-Nemotron-Nano-12B-v2-VL` -- the same decoder, a DIFFERENT geometry.

    Structurally this is its Omni sibling: a RADIO tower, an `mlp1` projector, an
    `image_flags` forward, and a Nemotron-H hybrid decoder whose pattern

        M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M-
        M = Mamba (28)      - = MLP (28)      * = attention (6)

    again leaves only 6 of 62 layers with an attention matrix, so `attention_layers`
    and the "192 cells, labelled, never pooled" caveat carry over verbatim.

    THE GEOMETRY DOES NOT CARRY OVER, AND THAT IS THE WHOLE REASON THIS IS A SEPARATE
    CLASS. The Omni is native-resolution: it buckets a picture to an aspect-matched
    size, emits ONE grid, and reports it in `imgs_sizes`. This checkpoint is InternVL:
    `image_processing.dynamic_preprocess` picks the closest aspect ratio up to
    `max_num_tiles: 12`, cuts the resize into 512x512 tiles, and -- because
    `use_thumbnail` is true -- appends a thumbnail of the WHOLE picture whenever there
    is more than one tile. On the boxed corpus that is 12 tiles plus a thumbnail for
    most pictures, 3,328 visual tokens against the Omni's 269, and no `imgs_sizes` key
    at all: `NemotronVL.grids_for` would raise on the first picture.

    SO TILING IS OFF, for exactly the reason it is off for `InternVL`. With thirteen
    16x16 sub-grids "the outer ring" is ambiguous -- the ring of a tile is an interior
    edge of the picture -- and the thumbnail covers the picture a second time at a
    different scale, so a peak could be counted twice at two different places. Forcing
    one tile gives one 16x16 grid over the whole picture, which is the geometry
    Qwen3-VL, InternVL3.5 and the Omni rows were all measured on. The tiled
    configuration is a separate, clearly labelled arm, not the primary comparison.

    HOW tiling is switched off matters. `max_num_tiles` is not declared in
    `NemotronNanoVLV2ImagesKwargs`, so passing it to `processor(...)` is silently
    dropped rather than honoured -- the failure mode `InternVL.proc_defaults` documents,
    and the reason it is set on the image processor as an ATTRIBUTE here. That is also
    what NVIDIA's own `processing.py` does to force single-tile video frames, and it is
    what keeps the text side consistent: the placeholder run is
    `num_patches * num_image_token`, read back from the same object.
    """

    name = "nemotron_vl_v2"
    model_types = ("NemotronH_Nano_VL_V2",)
    #: 512px tile / 16px patch = 32, pixel-shuffled by `downsample_ratio` 0.5 -> 16x16.
    #: Re-derived from the config in `bind`, so a checkpoint that changes either one
    #: fails loudly instead of mislabelling every patch.
    fixed_grid = (16, 16)
    encoder_px = 32
    #: This processor reports tile COUNTS, not sizes: `imgs_sizes` and `num_tokens` are
    #: Omni-only keys. `num_patches` is still not a forward kwarg.
    drop_inputs = ("num_patches",)

    def bind(self, model=None, processor=None, config=None):
        super().bind(model=model, processor=processor, config=config)
        cfg = self.config
        if cfg is not None:
            px = int(getattr(cfg, "force_image_size", 512))
            patch = int(getattr(cfg, "patch_size", 16))
            ratio = float(getattr(cfg, "downsample_ratio", 0.5))
            side = int(round(px / patch * ratio))
            if side <= 0 or abs(px / patch * ratio - side) > 1e-6:
                raise SystemExit(
                    f"nemotron_vl_v2: a {px}px tile at patch {patch} downsampled by "
                    f"{ratio} is not a whole grid")
            self.fixed_grid = (side, side)
            self.encoder_px = px // side
        ip = getattr(processor, "image_processor", None)
        if ip is not None:
            # See the class docstring: an attribute, not a call kwarg.
            ip.max_num_tiles = 1
            n = int(getattr(ip, "num_image_token", self.fixed_grid[0] ** 2))
            if n != self.fixed_grid[0] * self.fixed_grid[1]:
                raise SystemExit(
                    f"nemotron_vl_v2: the processor emits {n} tokens per tile but the "
                    f"config implies {self.fixed_grid[0]}x{self.fixed_grid[1]}")
        tok = getattr(processor, "tokenizer", None)
        if tok is not None and tok.pad_token is None:
            # This repo ships no pad token where the Omni ships `<|im_end|>`, and
            # `Family.build_inputs` asks for `padding=True` unconditionally. The module
            # is batch size 1 throughout -- `batch_needs_equal_size` exists because two
            # pictures in ONE sample is as wide as it gets -- so nothing is ever padded
            # and this cannot move a measured number; it only stops the tokenizer
            # refusing a request that has no work in it.
            tok.pad_token = tok.eos_token
        return self

    def build_inputs(self, processor, images, question, device, **proc_kwargs):
        """The parent's inputs, with `pixel_values` cast to the vision tower's dtype.

        NVIDIA's own line, from the release that has it. The Omni's `extract_feature`
        opens with

            pixel_values = pixel_values.to(dtype=self.vision_model.config.torch_dtype)

        and this checkpoint's does not -- it casts only inside `generate`, which is the
        path its model card demonstrates. Everything this module measures goes through
        `forward` instead, where a float32 processor output meets a bfloat16 RADIO and
        dies in the patch embedder's `F.linear` with "mat1 and mat2 have the same
        dtype".

        Cast HERE rather than by patching the remote module, for the reason
        `_shim_masking_api` gives: the modules cache is re-downloaded whenever the repo
        changes, so an edit on disk is silently lost.
        """
        out = super().build_inputs(processor, images, question, device, **proc_kwargs)
        vc = getattr(self.config, "vision_config", None)
        dt = getattr(vc, "torch_dtype", None) if vc is not None else None
        if isinstance(dt, str):
            import torch
            dt = getattr(torch, dt, None)
        if dt is not None and out.get("pixel_values") is not None:
            out["pixel_values"] = out["pixel_values"].to(dtype=dt)
        return out

    def grids_for(self, runs, inputs):
        """The fixed-grid answer, NOT the Omni's `imgs_sizes` one.

        `Family.grids_for` already divides a run into whole tiles, so this stays correct
        if the tiled arm is ever run -- it would report 13 grids rather than raising.
        """
        return Family.grids_for(self, runs, inputs)

    def grid_of(self, processor, image):
        return tuple(self.fixed_grid)

    def patch_px(self, image):
        return Family.patch_px(self, image)


# ---------------------------------------------------------------------------
# geometry helpers that need the view box
# ---------------------------------------------------------------------------
def _size_get(size, key):
    """One field of a processor's size spec, whether it is a dict or a `SizeDict`.

    transformers hands these back in both shapes depending on the processor and the
    version, and `"shortest_edge" in size` raises on one of them. A view box read off the
    wrong branch silently frames every patch statistic on the wrong region.
    """
    if size is None:
        return None
    if isinstance(size, dict):
        return size.get(key)
    return getattr(size, key, None)


def view_crop(image, view):
    """The part of the picture the grid covers, as a picture of its own.

    The per-patch content statistics and the frame check's marker both have to be read on
    the region the model saw. Handing them the whole picture where the model centre-cropped
    it would line every covariate up against the wrong patch.
    """
    u0, v0, u1, v1 = view
    if (u0, v0, u1, v1) == (0.0, 0.0, 1.0, 1.0):
        return image
    W, H = image.size
    box = (int(round(u0 * W)), int(round(v0 * H)),
           max(int(round(u0 * W)) + 1, int(round(u1 * W))),
           max(int(round(v0 * H)) + 1, int(round(v1 * H))))
    return image.crop(box)


def block_permute(image, grid, block_px, perm=None, seed=0, mode="shuffle"):
    """A10 -- shuffle the PIXEL BLOCKS that will become grid cells, before the encoder.

    A9 shuffles the encoder's OUTPUT rows and shows the attractor travels with the token,
    which rules out the language model's positional slot. It does not rule out the
    ENCODER's own position embeddings, which could have stamped the token on the way
    through. This does: the pixels of grid cell j are moved to cell `perm[j]` before the
    vision tower runs, so if the attractor appears at whatever content now occupies the
    top-left OF THE VIT GRID, the encoder's position embedding is writing it, and if it
    follows the original content instead, it is content-driven.

    The picture is first resized to the ENCODER'S OWN input size -- `block_px` pixels per
    grid cell -- so the blocks are integer, the shuffle is lossless, and the processor's
    own resize is the identity rather than a second resampling that would blur every
    block boundary. That resize is still not free on the source side, so
    `mode="identity"` runs it with the identity permutation and is the baseline this arm
    is paired against.

    -> (image, perm) where slot j of the new picture holds the block that was at perm[j].
    """
    from PIL import Image

    gh, gw = int(grid[0]), int(grid[1])
    n = gh * gw
    bw = bh = max(1, int(block_px))
    base = image.resize((bw * gw, bh * gh), Image.BICUBIC)
    if perm is None:
        if mode == "identity":
            perm = np.arange(n)
        elif mode == "shuffle":
            perm = np.random.default_rng(int(seed)).permutation(n)
        else:
            raise ValueError(f"unknown block permutation mode {mode!r}")
    perm = np.asarray(perm, dtype=np.int64)
    if perm.shape != (n,) or not np.array_equal(np.sort(perm), np.arange(n)):
        raise ValueError(f"block_permute needs a permutation of {n} blocks")
    if mode == "identity":
        return base, perm
    out = Image.new("RGB", base.size)
    for j in range(n):
        src = int(perm[j])
        sr, sc = divmod(src, gw)
        dr, dc = divmod(j, gw)
        out.paste(base.crop((sc * bw, sr * bh, (sc + 1) * bw, (sr + 1) * bh)),
                  (dc * bw, dr * bh))
    return out, perm
