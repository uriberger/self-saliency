"""
vga_common.py — sample loading and prompt building shared by the VGA probes.

The three probes on the wiki's validation ladder
(``analysis/vga_layer_scan.py``, ``analysis/vga_grounding_probe.py``,
``scripts/vga_selfcheck.py``) all need the same three things: a model, a stream
of (image, question, region) samples, and the exact prompt ``run_experiment.py``
would have built.  That last point matters — a probe measured on a different
prompt than the eval runs is measuring a different model.
"""

from __future__ import annotations

import ast
import json
import string
import sys
import zlib
from pathlib import Path

# The probes live in analysis/ but import the repo's own loaders.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.head_selection.generate import (  # noqa: E402
    GROUND_TRUTH_FIELD_CANDIDATES,
    QUESTION_FIELD_CANDIDATES,
    _load_dataset_resolved,
    _resolve_image,
    get_field,
    load_model,
)

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_DATASET = "saliency_r1_8k"


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def add_common_args(p) -> None:
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Model id or checkpoint path (default: {DEFAULT_MODEL}).")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"Dataset slug understood by run_experiment.py (default: {DEFAULT_DATASET}). "
                        "saliency_r1_8k and visual_cot ship human key-region boxes, which is what "
                        "the grounding probe scores against.")
    p.add_argument("--split", default=None, help="Override the dataset split.")
    p.add_argument("--subset", default=None, help="Override the dataset config/subset.")
    p.add_argument("--limit", type=int, default=500,
                   help="How many samples to use (default: 500, the wiki's go/no-go size).")
    p.add_argument("--skip", type=int, default=0, help="Skip this many samples first.")
    p.add_argument("--image", default=None,
                   help="Use one image from disk instead of a dataset. Requires --question.")
    p.add_argument("--question", default=None, help="The question to pair with --image.")
    p.add_argument("--attn-impl", default=None,
                   help="Force an attention implementation ('eager' is required by anything "
                        "that reads attention weights).")
    p.add_argument("--require-boxes", action="store_true",
                   help="Skip samples with no ground-truth region instead of erroring.")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load(args, need_attention: bool = False):
    """Load model + processor the same way run_experiment.py does."""
    from transformers import AutoProcessor

    impl = args.attn_impl or ("eager" if need_attention else None)
    model = load_model(args.model, attn_impl=impl)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    if need_attention and hasattr(model, "visual") and hasattr(model.visual, "config"):
        # Only the LM needs eager; materialising the vision tower's full attention
        # matrix costs a lot of memory for nothing.
        vcfg = model.visual.config
        if getattr(vcfg, "_attn_implementation", None) == "eager":
            vcfg._attn_implementation = "sdpa"
    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

def _parse_boxes(row) -> list[list[float]] | None:
    """Human key-region box(es) in relative [0, 1] xyxy, or None.

    Same two on-disk shapes analysis/aggregation_correlation.py handles:
    saliency-r1-8k stores one flat ``[x1,y1,x2,y2]``, visual_cot a list of them.
    """
    raw = row.get("bbox") if hasattr(row, "get") else None
    if raw is None:
        return None
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else list(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    if len(parsed) == 4 and all(isinstance(v, (int, float)) for v in parsed):
        return [[float(v) for v in parsed]]
    out = [[float(v) for v in b] for b in parsed
           if isinstance(b, list) and len(b) == 4
           and all(isinstance(v, (int, float)) for v in b)]
    return out or None


def samples(args):
    """Yield dicts: sample_id, image (RGB PIL), question, boxes (or None)."""
    if args.image:
        if not args.question:
            raise ValueError("--image needs --question")
        from PIL import Image
        img = Image.open(args.image)
        yield dict(sample_id=Path(args.image).stem,
                   image=img.convert("RGB") if img.mode != "RGB" else img,
                   question=args.question, boxes=None)
        return

    ds, hf_id = _load_dataset_resolved(args.dataset, args.split, args.subset)
    n_yielded = 0
    for i, row in enumerate(ds):
        if i < args.skip:
            continue
        if n_yielded >= args.limit:
            return

        raw = get_field(row, ["decoded_image", "image", "img"])
        if isinstance(raw, list):
            raw = raw[0]
        image = _resolve_image(raw, hf_id)
        if image is None:
            continue
        if hasattr(image, "convert") and image.mode != "RGB":
            image = image.convert("RGB")

        question = get_field(row, QUESTION_FIELD_CANDIDATES, default="")
        choices = get_field(row, ["choices", "options"])
        if isinstance(choices, str):
            try:
                choices = ast.literal_eval(choices)
            except Exception:
                choices = None
        if choices:
            labels = string.ascii_uppercase
            question = question + "\n" + "\n".join(
                f"{labels[j]}. {c}" for j, c in enumerate(choices))

        boxes = _parse_boxes(row)
        if boxes is None and getattr(args, "require_boxes", False):
            continue

        n_yielded += 1
        yield dict(sample_id=get_field(row, ["id", "qid", "question_id", "pid"], default=i),
                   image=image, question=question, boxes=boxes,
                   ground_truth=get_field(row, GROUND_TRUTH_FIELD_CANDIDATES))


def build_inputs(processor, model, image, question, system_prompt=None):
    """The prompt run_experiment.py would have built, tokenised and on device."""
    messages = []
    if system_prompt:
        messages.append({"role": "system",
                         "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]})
    return processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)


# ---------------------------------------------------------------------------
# Region → patch-grid mask
# ---------------------------------------------------------------------------

def boxes_to_patch_mask(boxes, grid_hw) -> np.ndarray:
    """Relative xyxy boxes → a boolean (grid_h·grid_w,) patch mask.

    A patch counts as inside if its *centre* falls in any box.  Centre rather
    than overlap because overlap makes every box at least one patch wide in each
    direction, which flatters a small box on a coarse grid.
    """
    gh, gw = grid_hw
    ys = (np.arange(gh) + 0.5) / gh
    xs = (np.arange(gw) + 0.5) / gw
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    mask = np.zeros((gh, gw), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        mask |= (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
    return mask.reshape(-1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice(score: np.ndarray, mask: np.ndarray, quantile: float | None = None) -> float:
    """Dice between a thresholded score map and a binary mask.

    The paper reports Dice, which needs a threshold that Dice itself does not
    supply.  Thresholding at the mask's own size (``quantile=None``) is the only
    choice that does not hand one method an area advantage over another: both
    maps are cut to exactly ``mask.sum()`` patches, so Dice reduces to
    precision = recall and the comparison is purely about *where* the mass is.
    """
    k = int(mask.sum())
    if k == 0 or k == mask.size:
        return float("nan")
    if quantile is None:
        # Ties are broken by jitter, not by index order: argsort alone hands
        # every flat map the top-left patches, which is where boxes
        # disproportionately are (a constant map scored 0.50 instead of its true
        # 0.25 on a 4×4 toy before this). The seed is a checksum of the map, so
        # the result is reproducible for a given map and uncorrelated across
        # samples — a single fixed seed would repeat one arbitrary permutation
        # over the whole dataset.
        seed = zlib.crc32(np.ascontiguousarray(score, dtype=np.float64).tobytes())
        jitter = np.random.default_rng(seed).random(score.size)
        pred = np.zeros_like(mask)
        pred[np.lexsort((jitter, -score))[:k]] = True
    else:
        pred = score >= np.quantile(score, quantile)
    inter = int((pred & mask).sum())
    return 2.0 * inter / (int(pred.sum()) + k)


def auroc(score: np.ndarray, mask: np.ndarray) -> float:
    """Patch-level AUROC of the score against the mask — chance-corrected at 0.5.

    Dice depends on the mask's area; AUROC does not, which is why the repo's
    reward work prefers it.  Reported alongside Dice so a result is not an
    artefact of box size.
    """
    pos, neg = int(mask.sum()), int((~mask).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    # Average ranks within ties, or a constant map scores 1.0 instead of 0.5.
    s_sorted = score[order]
    i = 0
    while i < s_sorted.size:
        j = i
        while j + 1 < s_sorted.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[mask].sum() - pos * (pos + 1) / 2.0) / (pos * neg)


def mean_in(score: np.ndarray, mask: np.ndarray) -> float:
    """The repo's mean_in: mean score inside the region, peak-normalised."""
    peak = score.max()
    if peak <= 0 or mask.sum() == 0:
        return float("nan")
    return float(score[mask].mean() / peak)


@torch.no_grad()
def last_token_attention(model, inputs, span) -> dict[int, dict]:
    """Per layer, where the last prompt token's attention goes.

    Returns ``{layer_idx: {"visual": (m,) head-mean over the visual span,
    "bos": head-mean mass on position 0, "visual_mass": total on the span}}``.
    Requires ``attn_implementation='eager'``.

    The weights are reduced inside the hook and the full matrix is dropped
    before the next layer runs — ``output_attentions=True`` alone would hold
    36 × (heads × T × T), which is ~9 GB at T=2000.  Same trick as
    ``vlm/saliency.py``.
    """
    from vlm.saliency import _get_model_config

    decoder = _get_model_config(model)["layers"]
    hookable = [i for i in range(len(decoder)) if hasattr(decoder[i], "self_attn")]
    s, e = span
    buf: dict[int, dict] = {}

    def _mk(li):
        def hook(module, inp, out):
            if not (isinstance(out, tuple) and len(out) > 1 and out[1] is not None):
                return None
            a = out[1]                                   # (1, H, q_len, kv_len)
            row = a[0, :, -1, :].detach().float()        # (H, kv_len)
            vis = row[:, s:e].mean(dim=0)
            buf[li] = {"visual": vis.cpu().numpy(),
                       "bos": float(row[:, 0].mean()),
                       "visual_mass": float(vis.sum())}
            return (out[0], None) + out[2:]              # drop the matrix
        return hook

    handles = [decoder[i].self_attn.register_forward_hook(_mk(i)) for i in hookable]
    try:
        model(**inputs, output_attentions=True, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    if not buf:
        raise RuntimeError(
            "no attention weights were captured — load the model with attn_impl='eager'")
    return buf


def attention_baseline(model, inputs, span, layers=None) -> np.ndarray:
    """The model's own attention to the visual patches — VSC's comparison arm.

    Attention from the last prompt token to each visual position, averaged over
    heads and over ``layers`` (all of them by default).

    CAVEAT, and it is why this is a *baseline* rather than a measurement of what
    any reward saw: it is prefill attention from a single query position, which
    the sink-location work found reads about a third high versus the
    teacher-forced generated-token readout.  It is the right arm here anyway,
    because it is exactly the map VGA claims to beat — the one distorted by
    sinks.
    """
    per_layer = last_token_attention(model, inputs, span)
    keep = per_layer if layers is None else {k: v for k, v in per_layer.items()
                                             if k in set(layers)}
    if not keep:
        raise ValueError(f"no captured layer matched {layers}")
    return np.mean([v["visual"] for v in keep.values()], axis=0)
