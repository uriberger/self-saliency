"""
Two-step aggregation-correlation pipeline.

Step 1 — collect (GPU required):
  For each sample in an inference JSONL:
  - Skip if the response hit the max_new_tokens limit (finish_reason == "length").
  - Identify <observe> steps via the FLAN-T5 steps classifier.
  - Ground each step to an image bounding box, using one of two sources
    (--bbox-source):
    - "dino"  (default) — Grounding DINO localises the region each step's
      text refers to, independently per step.
    - "human" — the single human-annotated bounding box shipped with the
      dataset (e.g. peterant330/saliency-r1-8k's "bbox" column), applied to
      every observe step in the sample. This is a question-level box (the
      region a human judged relevant to answering the question), not a
      per-step box, so no Grounding DINO call is made in this mode.
  - Run a single teacher-forced forward pass capturing raw per-layer, per-head
    attention from observe tokens to image patches.
  - Collapse the within-step token dimension (mean / min / max), producing
    three (n_steps, n_layers, n_heads, n_patches) float16 arrays per sample.
  - Save per-sample .npz files + a metadata.jsonl index (resumable).

  Each saliency map in the .npz is traceable via the metadata record to:
    layer index, head index, step text, sample id / sample number.

Step 2 — analyze (CPU only):
  Load the saved metadata and .npz files, then sweep all aggregation combos:

  Aggregated combos (36):
    - value_weighting : {True, False}
    - head_reduction  : {mean, min, max}
    - layer_reduction : {sum, last}
    - token_reduction : {mean, min, max}   (pre-computed in Step 1)

  Per-layer-per-head combos (n_layers × n_heads × 3 tr × 2 vw):
    Select a single (layer, head) pair, ignoring all others.

  For each aggregation combo, three bbox metrics are computed:
    - mean_in    : mean saliency inside bbox (heatmap normalized to [0,1])
    - sum_in     : sum of saliency inside bbox (heatmap normalized to [0,1])
    - enrichment : mean_inside / (mean_outside + 1e-8)

  Correlation metrics reported (sorted by point-biserial r):
    - r           : point-biserial correlation with correctness (ranking metric)
    - p           : raw p-value for r
    - q           : Benjamini-Hochberg FDR-adjusted p-value, corrected across
                    all combos tested for that bbox metric (hundreds to
                    thousands of comparisons — raw p-values alone would
                    produce many false positives)
    - mean_diff   : mean(metric | correct) − mean(metric | wrong)
    - rank_biserial : rank-biserial correlation (2×AUROC − 1; non-parametric)

  Output: top 20 combos per bbox metric printed to screen and saved to
  analysis_results.json in the output directory.

  Note: value weighting multiplies attention by value-state L2 norms.  For
  token_reduction=mean this is exact.  For min/max it is an approximation
  (vw is applied after token reduction rather than before), because the raw
  per-token data is not retained.

Usage
-----
  source venv/bin/activate.fish

  # Step 1 — collect raw saliency maps
  python analysis/aggregation_correlation.py collect \\
      --results results/inference/qwen3_vl_8b_instruct-mathvision-step_prompt_mnt8192.jsonl \\
      --output  results/analysis/agg_corr_mathvision/

  # Step 2 — sweep combos and print correlation table (no GPU)
  python analysis/aggregation_correlation.py analyze \\
      --output  results/analysis/agg_corr_mathvision/
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_RESULTS = (
    ROOT / "results/inference"
    / "qwen3_vl_8b_instruct-mathvision-step_prompt_mnt8192.jsonl"
)
DEFAULT_OUTPUT  = ROOT / "results/analysis/agg_corr_mathvision"
DEFAULT_MODEL   = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_DATASET = "MathLLMs/MathVision"
DEFAULT_SPLIT   = "testmini"

STEP_SYSTEM_PROMPT = (
    "You are a meticulous and precise AI assistant, an expert in visual reasoning. "
    "Your primary goal is to solve the user's query by providing a detailed, "
    "step-by-step thought process.\n"
    "First, share your detailed reasoning inside <think>...</think> tags. "
    "Within your reasoning, break it into explicit steps using <step>...</step> tags "
    "(e.g. <step>Observe the image...</step>, <step>Conclude that...</step>). "
    "Then, provide only your final answer inside <answer>...</answer> tags."
)

HEAD_REDUCTIONS  = ["mean", "min", "max"]
LAYER_REDUCTIONS = ["sum", "last"]
TOKEN_REDUCTIONS = ["mean", "min", "max"]
VALUE_WEIGHTINGS = [True, False]

BBOX_METRICS = ["mean_in", "sum_in", "enrichment"]
TOP_N = 20

BBOX_SOURCES = ["dino", "human"]
HUMAN_BBOX_FIELD = "bbox"  # column name in the human-annotated dataset (e.g. saliency-r1-8k)

COMBOS   = list(product(VALUE_WEIGHTINGS, HEAD_REDUCTIONS, LAYER_REDUCTIONS, TOKEN_REDUCTIONS))
ALL_KEYS = [f"vw{int(vw)}_hr{hr}_lr{lr}_tr{tr}" for vw, hr, lr, tr in COMBOS]


# ---------------------------------------------------------------------------
# Multi-GPU helpers (same pattern as run_experiment.py)
# ---------------------------------------------------------------------------

def _claim_next_sample(counter_path: Path) -> int:
    """Atomically claim and return the next sample index from a shared counter file."""
    import fcntl
    with open(counter_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            idx = int(f.read().strip())
            f.seek(0)
            f.write(str(idx + 1))
            f.truncate()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return idx


def _increment_file_counter(path: Path) -> int:
    """Atomically increment a counter file and return the new value."""
    import fcntl
    with open(path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            val = int(f.read().strip() or "0") + 1
            f.seek(0)
            f.write(str(val))
            f.truncate()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return val


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _score_saliency_flat(
    flat_map: np.ndarray,    # (n_patches,) float32, already ReLU'd
    mask_flat: np.ndarray,   # (n_patches,) bool
) -> dict:
    """
    Normalize map to [0, 1] then compute all three bbox metrics.
    Returns dict with keys mean_in, sum_in, enrichment.
    """
    vmax = flat_map.max()
    m = flat_map / vmax if vmax > 0 else flat_map
    inside  = m[mask_flat]
    outside = m[~mask_flat]
    mean_in = float(inside.mean())
    return {
        "mean_in":    mean_in,
        "sum_in":     float(inside.sum()),
        "enrichment": mean_in / (float(outside.mean()) + 1e-8),
    }


def _score_saliency_batch(
    maps: np.ndarray,        # (n_maps, n_patches) float32
    mask_flat: np.ndarray,   # (n_patches,) bool
) -> tuple:
    """
    Vectorized version of _score_saliency_flat over many maps at once.
    Returns (mean_in, sum_in, enrichment), each shape (n_maps,) float64.
    """
    vmaxes = maps.max(axis=1, keepdims=True)
    vmaxes = np.where(vmaxes > 0, vmaxes, 1.0)
    m       = maps / vmaxes                               # (n_maps, n_patches)
    inside  = m[:, mask_flat]                             # (n_maps, n_in)
    outside = m[:, ~mask_flat]                            # (n_maps, n_out)
    mean_in = inside.mean(axis=1)                         # (n_maps,)
    sum_in  = inside.sum(axis=1)                          # (n_maps,)
    enrich  = mean_in / (outside.mean(axis=1) + 1e-8)    # (n_maps,)
    return (
        mean_in.astype(np.float64),
        sum_in.astype(np.float64),
        enrich.astype(np.float64),
    )


def _is_truncated(sample: dict) -> bool:
    """Return True if the model hit the max_new_tokens limit during inference."""
    if sample.get("finish_reason") == "length":
        return True
    if sample.get("truncated") is True:
        return True
    return False


def _parse_key(k: str):
    m = re.match(r"vw(\d)_hr(\w+)_lr(\w+)_tr(\w+)", k)
    return (bool(int(m.group(1))), m.group(2), m.group(3), m.group(4)) if m else None


def _parse_human_bbox(row: dict) -> list | None:
    """
    Parse the human-annotated bounding box(es) shipped with the dataset row.
    Returns a *list of boxes* ([[x1, y1, x2, y2], ...] in relative [0, 1]
    coordinates), or None if absent/malformed. Two on-disk formats are handled:

    - saliency-r1-8k's "bbox": a JSON flat "[x1, y1, x2, y2]" (one box per row).
    - visual_cot's "bbox": a JSON list of boxes "[[x1, y1, x2, y2], ...]" (the
      question-level key region(s); usually one, occasionally several).

    Both are question-level annotations (not per-step); collect applies the
    returned box list to every observe step in the sample.
    """
    raw = row.get(HUMAN_BBOX_FIELD)
    if raw is None:
        return None
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else list(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, list):
        return None
    # Flat single box [x1, y1, x2, y2] (saliency-r1-8k).
    if len(parsed) == 4 and all(isinstance(v, (int, float)) for v in parsed):
        return [[float(v) for v in parsed]]
    # List of boxes [[x1, y1, x2, y2], ...] (visual_cot).
    out = [
        [float(v) for v in b]
        for b in parsed
        if isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)
    ]
    return out or None


# ---------------------------------------------------------------------------
# Step 1 helpers
# ---------------------------------------------------------------------------

def _free_after_exception(e) -> str:
    """Return str(e) after dropping every reference the exception holds.

    The traceback keeps the raising stack frames alive — including any GPU
    tensors in their locals — and so do the tracebacks of chained
    __context__/__cause__ exceptions (nulling e.__traceback__ alone leaves
    those). empty_cache() can only return memory once all references and
    cycles are gone, hence gc.collect() first.
    """
    msg = str(e)
    e.__traceback__ = None
    e.__context__   = None
    e.__cause__     = None
    del e
    gc.collect()
    torch.cuda.empty_cache()
    return msg


def encode_teacher_forced(processor, messages, response):
    """Tokenise the prompt and the teacher-forced [prompt + response] sequence.

    Returns ``(full_inputs, prompt_len)`` on CPU, or None if templating fails.
    This is the *step-prompt* encoding (chat template over `messages`), used by
    the inference-JSONL collect path.  Callers that reproduce a different
    generation-time prompt (e.g. an lmms-eval run) build the same pair
    themselves and pass it to extract_obs_spans / extract_raw_attention via
    `encoded=`, so both functions stay a single implementation.
    """
    full_messages = messages + [
        {"role": "assistant", "content": [{"type": "text", "text": response}]}
    ]
    try:
        full_inputs = processor.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt",
        )
        prompt_inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
    except Exception as e:
        print(f"  [warn] tokenisation failed: {_free_after_exception(e)}", flush=True)
        return None
    return full_inputs, prompt_inputs["input_ids"].shape[1]


def extract_obs_spans(response, processor, messages, clf_model, clf_tok, include_chain,
                      *, encoded=None, segment_fn=None):
    """
    Tokenise [prompt + response], classify response tokens, return observe spans.

    Returns (spans, resp_tokens) or None on failure.
    spans : list of (step_text, resp_tok_start, resp_tok_end)

    encoded    : optional pre-built (full_inputs, prompt_len) from a caller that
                 reproduces a non-step-prompt generation setup.
    segment_fn : optional callable(resp_tokens) -> spans, replacing the default
                 <step>-tag segmenter (which only fires on STEP_SYSTEM_PROMPT
                 output; freeform <think> chains need a sentence splitter).
    """
    from selfsal.steps import segment_tagged as extract_observe_spans_via_classifier

    if encoded is None:
        encoded = encode_teacher_forced(processor, messages, response)
        if encoded is None:
            return None
    full_inputs, prompt_len = encoded

    total_len  = full_inputs["input_ids"].shape[1]
    tok        = getattr(processor, "tokenizer", processor)
    ids        = full_inputs["input_ids"][0].tolist()
    resp_tokens = [
        tok.decode([ids[p]], skip_special_tokens=False)
        for p in range(prompt_len, total_len)
    ]
    if segment_fn is not None:
        spans = segment_fn(resp_tokens)
    else:
        spans = extract_observe_spans_via_classifier(resp_tokens, clf_model, clf_tok, include_chain)
    return spans, resp_tokens


@torch.no_grad()
def extract_raw_attention(model, processor, messages, response, obs_spans, image_token_id,
                          *, encoded=None, with_v_norms=True):
    """
    Single forward pass capturing per-layer, per-head attention from observe
    tokens to image patches (no generation — response is teacher-forced).

    Returns
    -------
    layer_attns  : list[n_layers] of float16 ndarray (n_heads, n_obs_tok, n_patches)
    v_norms_list : list[n_layers] of float32 ndarray (n_heads, n_patches)
    step_slices  : list[n_steps] of (obs_start, obs_end) into the obs-token axis
    grid_h, grid_w : patch-grid dimensions (n_patches == grid_h * grid_w)

    Returns None on any failure.
    """
    from selfsal.models.introspect import _get_model_config, _get_value_tensor

    dev        = next(model.parameters()).device
    cfg        = _get_model_config(model)
    layers     = cfg["layers"]
    num_groups = cfg["num_groups"]

    if encoded is None:
        encoded = encode_teacher_forced(processor, messages, response)
        if encoded is None:
            return None
    full_inputs, prompt_len = encoded
    try:
        inputs = full_inputs.to(dev)
    except Exception as e:
        print(f"  [warn] moving inputs to device failed: {_free_after_exception(e)}", flush=True)
        return None

    # Eager attention materialises a full (n_heads, seq, seq) weight matrix per
    # layer — bf16 scores + a transient fp32 softmax copy, ~8 bytes per element
    # at peak.  We DON'T pass output_attentions=True (see the model() call
    # below): that would make transformers' @check_model_inputs retain every
    # layer's matrix in outputs.attentions for the whole forward (~n_layers×
    # this estimate → tens of GiB → OOM-wedge).  Our per-layer forward hook
    # copies each layer's slice to CPU and lets the full matrix free
    # immediately, so the real peak is a SINGLE layer's matrix, ≈ est_peak.
    # Require 2× headroom so the skip fires before a mid-forward OOM (which can
    # leave the allocator wedged for the rest of the run).
    seq_len = inputs["input_ids"].shape[1]
    n_heads = getattr(getattr(model.config, "text_config", model.config),
                      "num_attention_heads", 32)
    est_peak  = seq_len * seq_len * n_heads * 8
    free_b, _ = torch.cuda.mem_get_info(dev)
    if est_peak * 2 > free_b:
        print(f"  [warn] sequence too long for eager attention: seq={seq_len}, "
              f"est. peak {est_peak / 2**30:.1f} GiB vs {free_b / 2**30:.1f} GiB free",
              flush=True)
        del inputs, full_inputs
        torch.cuda.empty_cache()
        return None

    ids        = inputs["input_ids"][0].tolist()
    image_pos  = torch.tensor(
        [i for i, t in enumerate(ids) if t == image_token_id],
        device=dev, dtype=torch.long,
    )
    if len(image_pos) == 0:
        print("  [warn] no image tokens found", flush=True)
        return None
    n_patches = len(image_pos)

    # Build observe-token index list and per-step slices into the obs axis
    obs_tok_list: list[int] = []
    step_slices:  list[tuple[int, int]] = []
    for _text, t_start, t_end in obs_spans:
        s = len(obs_tok_list)
        obs_tok_list.extend(range(t_start, t_end))
        step_slices.append((s, len(obs_tok_list)))
    if not obs_tok_list:
        return None

    abs_obs = torch.tensor(
        [prompt_len + i for i in obs_tok_list], device=dev, dtype=torch.long,
    )

    # Register hooks on LM self-attention layers
    hookable  = [i for i in range(len(layers)) if hasattr(layers[i], "self_attn")]
    n_layers  = len(hookable)
    layer_buf = [None] * n_layers

    def _make_hook(hi: int):
        def _hook(module, inp, out):
            if not (isinstance(out, tuple) and len(out) > 1 and out[1] is not None):
                return
            attn_w = out[1]  # (1, n_heads, seq_len, seq_len)
            a = attn_w[0][:, abs_obs, :][:, :, image_pos]  # (n_heads, n_obs, n_patches)
            layer_buf[hi] = a.detach().cpu().half().numpy()
            # Drop the decoder's remaining reference to the full matrix so it
            # frees this layer before the next one allocates.  (This is only
            # effective because we do NOT pass output_attentions=True — that
            # would make transformers' @check_model_inputs keep its own copy of
            # every layer's matrix for the whole forward; see model() below.)
            return (out[0], None) + out[2:]
        return _hook

    handles = [
        layers[hookable[hi]].self_attn.register_forward_hook(_make_hook(hi))
        for hi in range(n_layers)
    ]
    try:
        # NB: no output_attentions=True.  With force_eager the attention module
        # always returns real attn_weights in out[1] (our hook reads them), so
        # we don't need transformers to collect them — and collecting them
        # (@check_model_inputs) would retain all n_layers matrices for the whole
        # forward instead of one at a time, OOM-wedging on long sequences.
        outputs = model(**inputs, use_cache=with_v_norms)
    except Exception as e:
        # Delete GPU tensors that are still alive in this scope before the
        # gc.collect() inside _free_after_exception, so they can be reclaimed
        # and the CUDA allocator is not left wedged.
        del inputs, full_inputs, abs_obs, image_pos, layer_buf
        print(f"  [warn] forward pass failed: {_free_after_exception(e)}", flush=True)
        return None
    finally:
        for h in handles:
            h.remove()

    if any(x is None for x in layer_buf):
        print("  [warn] some hooks did not fire — load model with eager attention?", flush=True)
        return None

    # Value norms from KV cache for value-weighting support in Step 2. Callers
    # that never value-weight (with_v_norms=False) skip both the KV cache and
    # this pass — it is the only thing here that depends on the cache layout.
    kv           = outputs.past_key_values if with_v_norms else None
    v_norms_list = []
    for hi in range(n_layers if with_v_norms else 0):
        vc = _get_value_tensor(kv, hookable[hi])
        ip = image_pos.to(vc.device)
        vs = vc[:, :, ip, :].float()                   # (1, n_kv_heads, n_patches, d)
        vs = vs.repeat_interleave(num_groups, dim=1)   # expand GQA → (1, n_heads, n_patches, d)
        vn = vs.squeeze(0).norm(dim=-1).cpu().numpy()  # (n_heads, n_patches)
        v_norms_list.append(vn)
    del outputs, kv

    if "image_grid_thw" in inputs:
        g      = inputs["image_grid_thw"][0]
        grid_h = int(g[1]) // 2
        grid_w = int(g[2]) // 2
    else:
        side   = int(np.sqrt(n_patches))
        grid_h = side
        grid_w = (n_patches + side - 1) // side

    return layer_buf, v_norms_list, step_slices, grid_h, grid_w


def compute_step_maps(
    layer_attns: list[np.ndarray],    # [n_layers] of (n_heads, n_obs_tok, n_patches)
    step_slices: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Token-reduce within each step using mean / min / max, keeping layer and
    head dimensions intact.

    Returns three float16 arrays, each shape (n_steps, n_layers, n_heads, n_patches):
      attn_tr_mean, attn_tr_min, attn_tr_max
    """
    n_layers  = len(layer_attns)
    n_steps   = len(step_slices)
    n_heads   = layer_attns[0].shape[0]
    n_patches = layer_attns[0].shape[2]

    out_mean = np.zeros((n_steps, n_layers, n_heads, n_patches), dtype=np.float16)
    out_min  = np.zeros((n_steps, n_layers, n_heads, n_patches), dtype=np.float16)
    out_max  = np.zeros((n_steps, n_layers, n_heads, n_patches), dtype=np.float16)

    for si, (sli_s, sli_e) in enumerate(step_slices):
        if sli_s >= sli_e:
            continue
        for li in range(n_layers):
            seg = layer_attns[li][:, sli_s:sli_e, :].astype(np.float32)
            seg = np.maximum(seg, 0)   # ReLU before reduction
            out_mean[si, li] = seg.mean(axis=1).astype(np.float16)
            out_min[si, li]  = seg.min(axis=1).astype(np.float16)
            out_max[si, li]  = seg.max(axis=1).astype(np.float16)

    return out_mean, out_min, out_max


# ---------------------------------------------------------------------------
# Step 1 — collection loop
# ---------------------------------------------------------------------------

def collect(args):
    out_dir    = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rank       = args.rank
    world_size = args.world_size
    multi_gpu  = world_size > 1
    pfx        = f"[r{rank}] " if multi_gpu else ""

    # Determine per-rank vs flat metadata file paths
    if multi_gpu:
        index_path        = out_dir / f"metadata.rank{rank}.jsonl"
        flat_path         = out_dir / "metadata.jsonl"
        all_rank_paths    = [out_dir / f"metadata.rank{r}.jsonl" for r in range(world_size)]
        counter_path      = out_dir / "collect.counter"
        done_counter_path = out_dir / "collect.done_ranks"
        counter_ready     = out_dir / "collect.counter.ready"
        progress_path     = out_dir / "collect.progress"
    else:
        index_path        = out_dir / "metadata.jsonl"
        flat_path         = index_path
        all_rank_paths    = None
        counter_path      = None
        done_counter_path = None
        counter_ready     = None

    # Resume: build done_ids from own rank file + sibling rank files + flat file
    done_ids: set[str] = set()
    sources = [index_path]
    if multi_gpu:
        sources += [p for p in all_rank_paths if p != index_path]
        if flat_path != index_path:
            sources.append(flat_path)
    for src in sources:
        if src.exists():
            for line in src.open():
                if line.strip():
                    done_ids.add(str(json.loads(line)["sample_id"]))
    if done_ids:
        print(f"{pfx}[resume] {len(done_ids)} samples already processed", flush=True)

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        sys.exit(1)

    samples = [json.loads(l) for l in results_path.open() if l.strip()]
    if args.limit:
        samples = samples[: args.limit]
    print(f"{pfx}Loaded {len(samples)} samples from {results_path.name}", flush=True)

    # Initialize dynamic work queue (rank 0 creates counter; others wait)
    if multi_gpu:
        if getattr(args, "reuse_queue", False) and counter_ready.exists():
            # Restarted mid-run (e.g. by collect_with_restart.sh after a
            # wedge exit): keep the shared counter so the other rank's
            # position is preserved and work is not re-claimed.
            print(f"{pfx}Reusing existing work queue (--reuse-queue)", flush=True)
        elif rank == 0:
            with open(counter_path, 'w') as _f:
                _f.write("0")
            with open(done_counter_path, 'w') as _f:
                _f.write("0")
            with open(progress_path, 'w') as _f:
                _f.write(str(len(done_ids)))
            counter_ready.touch()
            print(f"{pfx}Initialized work queue over {len(samples)} samples", flush=True)
        else:
            print(f"{pfx}Waiting for work queue...", flush=True)
            _t0_wait = time.time()
            while not counter_ready.exists():
                time.sleep(0.1)
                if time.time() - _t0_wait > 60:
                    raise RuntimeError(f"Timed out waiting for {counter_ready}")
            print(f"{pfx}Work queue ready", flush=True)

    n_todo = sum(
        1 for s in samples
        if str(s.get("sample_id", s.get("id", ""))) not in done_ids
    )
    if n_todo == 0 and not multi_gpu:
        print("All samples already processed. Run 'analyze' to compute correlations.")
        return

    # Pin to a specific GPU before model loading
    import os
    if args.gpu is not None:
        if args.dino_gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu},{args.dino_gpu}"
            print(f"{pfx}Pinned VLM to GPU {args.gpu}, DINO to GPU {args.dino_gpu} "
                  f"(CUDA_VISIBLE_DEVICES={args.gpu},{args.dino_gpu})", flush=True)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            print(f"{pfx}Pinned to GPU {args.gpu} (CUDA_VISIBLE_DEVICES={args.gpu})", flush=True)

    # Qwen3-VL forward pass + hooks can exceed Python's default 1000-frame limit
    # on some IJSO/olympiad samples with deep call stacks.
    import sys as _sys
    _sys.setrecursionlimit(3000)

    # Load all required models
    from datasets import load_dataset
    from transformers import AutoProcessor

    from experiments.head_selection.generate import (load_model, get_field,
                                                 _resolve_image, _load_dataset_resolved)
    from selfsal.steps import StepClassifier
    from selfsal.models.introspect import _get_image_token_id

    print(f"{pfx}Loading dataset...", flush=True)
    # Route through run_experiment's dispatch so custom-loader datasets (e.g.
    # ViRL39K, whose images live inside images.zip rather than as addressable repo
    # files) are loaded identically to inference — same rows, same row indices, so
    # the `image_filename` index written at inference time still maps to ds[i].
    ds, _ds_hf_id = _load_dataset_resolved(args.dataset, args.split)
    print(f"{pfx}  {len(ds)} rows", flush=True)

    if args.bbox_source == "human" and HUMAN_BBOX_FIELD not in ds.column_names:
        print(
            f"{pfx}ERROR: --bbox-source human requires a '{HUMAN_BBOX_FIELD}' column "
            f"in {args.dataset}, but columns are: {ds.column_names}",
            flush=True,
        )
        sys.exit(1)

    print(f"{pfx}Loading model {args.model} with eager attention...", flush=True)
    # Pin VLM to cuda:0 so it doesn't spill to cuda:1 (the aux-model GPU) when
    # two GPUs are visible via CUDA_VISIBLE_DEVICES.  Prevents both the
    # device-mismatch error in attention hooks and competition with DINO/FLAN-T5.
    model = load_model(args.model, force_eager=True, device_map={"": "cuda:0"})
    if hasattr(model, "visual") and hasattr(model.visual, "config"):
        vis_cfg = model.visual.config
        if getattr(vis_cfg, "_attn_implementation", None) == "eager":
            vis_cfg._attn_implementation = "sdpa"
            print("  Patched vision encoder: eager → sdpa", flush=True)
    model.eval()

    print(f"{pfx}Loading processor...", flush=True)
    processor      = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    # Default Qwen3-VL processor allows up to 16M pixels (4096×4096).  At 16px
    # patches that yields 65536 patches → the vision encoder's sdpa falls back to
    # math attention and tries to allocate 65536²×16 heads ≈ 137 GB → OOM.
    # Cap at 4M pixels (≈ 2048×2048) → ≤16384 patches → ≤4096 visual tokens.
    _MAX_PIXELS = 4 * 1024 * 1024
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "max_pixels"):
        old = processor.image_processor.max_pixels
        if old is None or old > _MAX_PIXELS:
            processor.image_processor.max_pixels = _MAX_PIXELS
            print(f"  Capped image_processor.max_pixels: {old} → {_MAX_PIXELS}", flush=True)
    image_token_id = _get_image_token_id(processor, model)
    print(f"{pfx}  image_token_id={image_token_id}", flush=True)

    aux_device = args.dino_device or ("cuda:1" if args.dino_gpu is not None else None)

    if args.bbox_source == "dino":
        from selfsal.grounding import ground_claim
        from selfsal.grounding.dino import load_local as load_grounding_dino
        print(f"{pfx}Loading Grounding DINO...", flush=True)
        dino_proc, dino_model, dino_device = load_grounding_dino(device=aux_device)
    else:
        print(f"{pfx}bbox-source=human — skipping Grounding DINO", flush=True)
        dino_proc = dino_model = dino_device = None

    print(f"{pfx}Loading steps classifier...", flush=True)
    clf = StepClassifier.load(device=aux_device)
    clf_model, clf_tok, include_chain = clf, clf._tokenizer, clf.include_chain

    # Wedge detection: if a mid-forward OOM leaves tens of GiB *allocated*
    # (live references, so empty_cache() cannot return it), every subsequent
    # sample fails and the run silently produces nothing.  Track the healthy
    # post-load baseline on the VLM GPU; if allocated memory stays far above
    # it after cleanup on consecutive failures, exit with code 17 so
    # scripts/collect_with_restart.sh can relaunch this rank with a fresh
    # CUDA context (--reuse-queue keeps the shared work queue intact).
    gc.collect()
    torch.cuda.empty_cache()
    wedge_baseline = torch.cuda.memory_allocated(0)
    WEDGE_MARGIN   = 10 * 2**30
    WEDGE_EXIT     = 17
    wedge_streak   = 0
    print(f"{pfx}VLM GPU baseline: {wedge_baseline / 2**30:.1f} GiB allocated", flush=True)

    def _check_wedge():
        nonlocal wedge_streak
        gc.collect()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(0)
        if allocated <= wedge_baseline + WEDGE_MARGIN:
            wedge_streak = 0
            return
        wedge_streak += 1
        print(
            f"{pfx}[wedge] {allocated / 2**30:.1f} GiB still allocated after cleanup "
            f"(baseline {wedge_baseline / 2**30:.1f} GiB) — strike {wedge_streak}/2",
            flush=True,
        )
        if wedge_streak >= 2:
            print(
                f"{pfx}[wedge] GPU memory unrecoverable — exiting {WEDGE_EXIT} "
                f"so the wrapper can restart this rank",
                flush=True,
            )
            sys.exit(WEDGE_EXIT)

    print(f"\n{pfx}Starting collection: {n_todo} samples to process", flush=True)
    t0 = time.time()
    n_processed_this_run = 0
    n_done_at_start = len(done_ids)

    PROGRESS_INTERVAL = 10

    def _update_progress():
        if multi_gpu:
            global_done = _increment_file_counter(progress_path)
        else:
            global_done = len(done_ids)
        if global_done % PROGRESS_INTERVAL == 0 or global_done == len(samples):
            elapsed = time.time() - t0
            new_done = global_done - n_done_at_start
            rate = new_done / elapsed if elapsed > 0 else 0
            remaining = len(samples) - global_done
            eta_str = f"{remaining / rate / 60:.0f}min" if rate > 0 else "?"
            print(
                f"{pfx}  [progress] {global_done}/{len(samples)}  "
                f"rate={rate:.2f}/s  ETA={eta_str}",
                flush=True,
            )

    def _sample_iter():
        if multi_gpu:
            n = len(samples)
            while True:
                i = _claim_next_sample(counter_path)
                if i >= n:
                    break
                yield i, samples[i]
        else:
            yield from enumerate(samples)

    with index_path.open("a") as idx_f:
        for sample_idx, sample in _sample_iter():
            sample_id = str(sample.get("sample_id", sample.get("id", sample_idx)))
            if sample_id in done_ids:
                continue

            # Discard samples that hit the max_new_tokens limit
            if _is_truncated(sample):
                print(
                    f"{pfx}  [{sample_idx+1}] sample={sample_id} SKIP: hit max_new_tokens limit",
                    flush=True,
                )
                done_ids.add(sample_id)
                _update_progress()
                continue

            response        = sample.get("response", "")
            correct         = sample.get("correct")
            image_row_index = sample.get("image_filename", sample_idx)
            question        = sample.get("question", "")
            t_sample        = time.time()

            # Load image
            try:
                row   = ds[int(image_row_index)]
                image = _resolve_image(
                    get_field(row, ["decoded_image", "image", "img"]), args.dataset
                )
                if hasattr(image, "convert") and image.mode != "RGB":
                    image = image.convert("RGB")
            except Exception as e:
                print(
                    f"{pfx}  [{sample_idx+1}] sample={sample_id} SKIP: image load error: {e}",
                    flush=True,
                )
                done_ids.add(sample_id)
                _update_progress()
                continue

            messages = [
                {"role": "system",
                 "content": [{"type": "text", "text": STEP_SYSTEM_PROMPT}]},
                {"role": "user",
                 "content": [
                     {"type": "image", "image": image},
                     {"type": "text",  "text":  question},
                 ]},
            ]

            # Identify observe spans
            span_result = extract_obs_spans(
                response, processor, messages, clf_model, clf_tok, include_chain,
            )
            if span_result is None:
                done_ids.add(sample_id)
                _update_progress()
                print(
                    f"{pfx}  [{sample_idx+1}] sample={sample_id} SKIP: tokenisation failed",
                    flush=True,
                )
                _check_wedge()
                continue

            obs_spans, _ = span_result
            if not obs_spans:
                done_ids.add(sample_id)
                _update_progress()
                print(
                    f"{pfx}  [{sample_idx+1}] sample={sample_id} SKIP: no observe steps",
                    flush=True,
                )
                continue

            # Ground each step to a bounding box, from the source selected by --bbox-source
            boxes_per_step: list[list] = []
            boxes_scored_per_step: list[list] = []
            if args.bbox_source == "dino":
                # With --save-all-boxes, ground at a low base threshold and keep every
                # box with its score so box_threshold / box-area can be tuned later in
                # analyze without re-running DINO.
                ground_thr = args.dino_base_threshold if args.save_all_boxes else args.box_threshold
                dino_oom = False
                for step_text, _, _ in obs_spans:
                    try:
                        boxes_and_scores = ground_claim(
                            image, step_text,
                            box_threshold=ground_thr, device=dino_device,
                        )
                    except torch.OutOfMemoryError as e:
                        print(f"  [warn] DINO OOM: {_free_after_exception(e)}", flush=True)
                        dino_oom = True
                        break
                    boxes = [b for b, s in boxes_and_scores if s >= args.box_threshold]
                    boxes_per_step.append(boxes)
                    if args.save_all_boxes:
                        boxes_scored_per_step.append([[*b, float(s)] for b, s in boxes_and_scores])
                if dino_oom:
                    done_ids.add(sample_id)
                    _update_progress()
                    print(
                        f"{pfx}  [{sample_idx+1}] sample={sample_id} SKIP: DINO OOM",
                        flush=True,
                    )
                    continue
            else:
                # "human" — question-level box(es) from the dataset, reused for
                # every observe step (the annotation isn't tied to a specific step).
                human_boxes = _parse_human_bbox(row)
                if not human_boxes:
                    done_ids.add(sample_id)
                    _update_progress()
                    print(
                        f"{pfx}  [{sample_idx+1}] sample={sample_id} SKIP: no human bbox",
                        flush=True,
                    )
                    continue
                boxes_per_step = [list(human_boxes) for _ in obs_spans]
            n_grounded = sum(1 for b in boxes_per_step if b)

            # Forward pass → raw per-layer, per-head attention
            torch.cuda.empty_cache()
            raw = extract_raw_attention(
                model, processor, messages, response, obs_spans, image_token_id,
            )
            if raw is None:
                done_ids.add(sample_id)
                _update_progress()
                print(
                    f"{pfx}  [{sample_idx+1}] sample={sample_id} SKIP: attention extraction failed",
                    flush=True,
                )
                _check_wedge()
                continue

            layer_attns, v_norms_list, step_slices, grid_h, grid_w = raw

            # Cap observe tokens to stay within memory budget
            n_obs_tok = layer_attns[0].shape[1]
            if n_obs_tok > args.max_obs_tokens:
                cap = args.max_obs_tokens
                print(f"{pfx}  [{sample_idx+1}] truncating obs tokens {n_obs_tok} → {cap}", flush=True)
                layer_attns = [a[:, :cap, :] for a in layer_attns]
                step_slices = [(min(s, cap), min(e, cap)) for s, e in step_slices]
                n_obs_tok   = cap

            # Token-reduce within each step, preserve layer and head dims
            attn_tr_mean, attn_tr_min, attn_tr_max = compute_step_maps(
                layer_attns, step_slices,
            )

            # v_norms: (n_layers, n_heads, n_patches) — used for value weighting in Step 2
            v_norms = np.stack(v_norms_list, axis=0).astype(np.float32)

            n_layers  = attn_tr_mean.shape[1]
            n_heads   = attn_tr_mean.shape[2]
            n_patches = attn_tr_mean.shape[3]

            # Save per-sample .npz
            npz_filename = f"sample_{sample_id}.npz"
            np.savez_compressed(
                out_dir / npz_filename,
                attn_tr_mean=attn_tr_mean,   # (n_steps, n_layers, n_heads, n_patches) float16
                attn_tr_min=attn_tr_min,
                attn_tr_max=attn_tr_max,
                v_norms=v_norms,             # (n_layers, n_heads, n_patches) float32
            )

            # Metadata: full traceability for every saliency map
            meta = {
                "sample_id":        sample_id,
                "sample_number":    sample_idx,   # index in the results JSONL
                "correct":          correct,
                "bbox_source":      args.bbox_source,
                "n_obs_steps":      len(obs_spans),
                "n_grounded_steps": n_grounded,
                "n_layers":         n_layers,
                "n_heads":          n_heads,
                "n_patches":        n_patches,
                "grid_h":           grid_h,
                "grid_w":           grid_w,
                "npz_file":         npz_filename,
                # attn[step_index, layer_index, head_index, :] for each step below
                "steps": [
                    {
                        "step_index": si,
                        "step_text":  text,
                        "boxes":      boxes_per_step[si],
                        **({"boxes_scored": boxes_scored_per_step[si]}
                           if (args.save_all_boxes and args.bbox_source == "dino")
                           else {}),
                    }
                    for si, (text, _, _) in enumerate(obs_spans)
                ],
            }
            idx_f.write(json.dumps(meta) + "\n")
            idx_f.flush()
            done_ids.add(sample_id)
            n_processed_this_run += 1
            _update_progress()

            elapsed = time.time() - t_sample
            print(
                f"{pfx}  [{sample_idx+1}] sample={sample_id}  "
                f"obs={len(obs_spans)}  grounded={n_grounded}  "
                f"layers={n_layers}  heads={n_heads}  "
                f"correct={correct}  {elapsed:.1f}s",
                flush=True,
            )

    print(f"\n{pfx}Collection done. Output directory: {out_dir}")

    # Auto-merge metadata shards when all ranks have finished
    if multi_gpu:
        n_ranks_done = _increment_file_counter(done_counter_path)
        print(f"{pfx}Rank {rank} done ({n_ranks_done}/{world_size} ranks finished)", flush=True)
        if n_ranks_done >= world_size:
            all_records = []
            for rp in all_rank_paths:
                if rp.exists():
                    for line in rp.open():
                        if line.strip():
                            all_records.append(json.loads(line))
            all_records.sort(key=lambda r: r.get("sample_number", 0))
            with flat_path.open("w") as f:
                for rec in all_records:
                    f.write(json.dumps(rec) + "\n")
            print(f"{pfx}Merged {len(all_records)} records → {flat_path}", flush=True)
            for rp in all_rank_paths:
                try:
                    rp.unlink()
                except FileNotFoundError:
                    pass
            for cp in (counter_path, done_counter_path, counter_ready, progress_path):
                try:
                    cp.unlink()
                except FileNotFoundError:
                    pass


# ---------------------------------------------------------------------------
# Step 2 — analysis
# ---------------------------------------------------------------------------

def _box_area(b):
    x1, y1, x2, y2 = b[:4]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _filtered_boxes(step, box_threshold=None, max_box_area=None):
    """Return a step's box list, optionally re-thresholded and area-filtered.

    If the step carries a 'boxes_scored' field (written by the 'reground'
    subcommand — each entry [x1, y1, x2, y2, score]) and box_threshold is given,
    boxes are selected by score >= box_threshold; otherwise the pre-thresholded
    'boxes' list is used as-is. max_box_area (a fraction in [0, 1]) drops boxes
    whose area exceeds the cap — e.g. to discard near-full-frame degenerate boxes.
    """
    scored = step.get("boxes_scored")
    if scored is not None and box_threshold is not None:
        boxes = [b[:4] for b in scored if b[4] >= box_threshold]
    else:
        boxes = [b[:4] for b in step.get("boxes", [])]
    if max_box_area is not None:
        boxes = [b for b in boxes if _box_area(b) <= max_box_area]
    return boxes


def _merged_record_boxes(rec, box_threshold=None, max_box_area=None):
    """Union every observe step's filtered boxes into one chain-level box list.

    Implements the 'merged DINO' bbox source — a granularity sitting between
    per-step DINO (a different box per step) and the question-level human box (a
    single annotation reused across steps): here we take the boxes DINO produced
    for *all* the chain's steps and merge them into one shape (the union of the
    rectangles — generally not itself a rectangle), then score every observe step
    against that same shape. Each per-step box is first passed through the same
    box_threshold / max_box_area filter as the per-step path, so degenerate
    near-full-frame detections are dropped *before* the union (per the user-chosen
    'filter per-box, then union' semantics).

    NB: reading DINO boxes requires box_threshold to be set (so boxes_scored is
    used); with box_threshold=None `_filtered_boxes` falls back to the 'boxes'
    field, which on a human-collected dir (e.g. visual_cot) is the human box, not
    DINO. Callers using this for merged-DINO must pass a box_threshold.

    Returns a flat list of [x1, y1, x2, y2] boxes (empty if nothing survives).
    """
    merged: list = []
    for step in rec.get("steps", []):
        merged.extend(_filtered_boxes(step, box_threshold, max_box_area))
    return merged


def compute_combo_scores(out_dir, box_thr=None, max_area=None, merge_boxes=False):
    """Load collected saliency maps and reduce each to a per-sample scalar for
    every aggregation combo (36 aggregated + n_layers*n_heads per-head).

    Shared by `analyze` and by the cross-validation generalization test
    (analysis/cv_generalization.py) so both score combos identically.

    merge_boxes=True switches the bbox source to chain-level merged DINO: every
    observe step is scored against the union of the whole chain's filtered boxes
    (see `_merged_record_boxes`) instead of its own per-step box. The attention
    map is still per-step; only the target box changes. This is the DINO analogue
    of the question-level human box and keeps the correlation chain-level (one
    per-chain scalar per combo, as always).

    Returns (combo_scores, correct_arr, all_keys, valid) where
    combo_scores[key][bbox_metric] is a list[float | None] aligned element-wise
    with `valid` (and with correct_arr).
    """
    out_dir    = Path(out_dir)
    index_path = out_dir / "metadata.jsonl"

    if not index_path.exists():
        print(f"No metadata.jsonl found in {out_dir}")
        sys.exit(1)

    records = [json.loads(l) for l in index_path.open() if l.strip()]
    if merge_boxes:
        # Chain-level merged DINO: a record is usable iff the union of its steps'
        # filtered boxes is non-empty. box_thr/max_area select and clean the
        # per-step boxes before the union (see _merged_record_boxes).
        valid = [r for r in records
                 if r.get("correct") is not None
                 and _merged_record_boxes(r, box_thr, max_area)]
        print(f"Box source: merged-DINO chain box (union of per-step boxes)  "
              f"box_threshold={box_thr}  max_box_area={max_area}", flush=True)
    elif box_thr is not None or max_area is not None:
        # Re-derive "grounded" from the filtered boxes (needs boxes_scored for a
        # threshold change; area filter works on legacy boxes too).
        def _n_grounded(rec):
            return sum(1 for s in rec["steps"] if _filtered_boxes(s, box_thr, max_area))
        valid = [r for r in records
                 if r.get("correct") is not None and _n_grounded(r) > 0]
        print(f"Box filter: box_threshold={box_thr}  max_box_area={max_area}", flush=True)
    else:
        valid = [r for r in records
                 if r.get("correct") is not None and r.get("n_grounded_steps", 0) > 0]

    if not valid:
        print("No valid records for analysis.")
        return

    print(f"Loaded {len(records)} records, {len(valid)} with grounded steps", flush=True)

    # Determine model dimensions from the first valid record
    n_layers = valid[0]["n_layers"]
    n_heads  = valid[0]["n_heads"]
    print(f"Model dimensions: {n_layers} layers × {n_heads} heads", flush=True)

    correct_arr = np.array([int(r["correct"]) for r in valid], dtype=float)

    # Build combo lists
    # Aggregated combos (2 vw × 3 hr × 2 lr × 3 tr = 36)
    agg_params = list(product(VALUE_WEIGHTINGS, HEAD_REDUCTIONS, LAYER_REDUCTIONS, TOKEN_REDUCTIONS))
    agg_keys   = [f"vw{int(vw)}_hr{hr}_lr{lr}_tr{tr}" for vw, hr, lr, tr in agg_params]

    # Per-layer-per-head combos (2 vw × n_layers × n_heads × 3 tr)
    ph_params = list(product(VALUE_WEIGHTINGS, range(n_layers), range(n_heads), TOKEN_REDUCTIONS))
    ph_keys   = [f"vw{int(vw)}_li{li:02d}_hi{hi:02d}_tr{tr}" for vw, li, hi, tr in ph_params]

    all_keys = agg_keys + ph_keys
    print(
        f"Sweeping {len(agg_keys)} aggregated combos + {len(ph_keys)} per-head combos"
        f" = {len(all_keys)} total",
        flush=True,
    )

    # combo_scores[key][metric] = list[float | None], one per valid sample
    combo_scores: dict = {key: {m: [] for m in BBOX_METRICS} for key in all_keys}

    for ri, rec in enumerate(valid):
        npz_path = out_dir / rec["npz_file"]
        if not npz_path.exists():
            print(f"  [warn] missing {npz_path.name}", flush=True)
            for key in all_keys:
                for m in BBOX_METRICS:
                    combo_scores[key][m].append(None)
            continue

        data = np.load(npz_path)
        attn_by_tr = {
            "mean": data["attn_tr_mean"].astype(np.float32),
            "min":  data["attn_tr_min"].astype(np.float32),
            "max":  data["attn_tr_max"].astype(np.float32),
        }
        v_norms = data["v_norms"].astype(np.float32)   # (n_layers, n_heads, n_patches)
        data.close()

        grid_h  = rec["grid_h"]
        grid_w  = rec["grid_w"]
        n_steps = attn_by_tr["mean"].shape[0]

        # Precompute flat boolean masks for each step (independent of combo).
        # For merged-DINO every step shares the chain's unioned box; otherwise
        # each step uses its own per-step box.
        merged_boxes = _merged_record_boxes(rec, box_thr, max_area) if merge_boxes else None
        mask_flat_info = []   # (mask_flat, n_in, n_out) or None per step
        for step in rec["steps"]:
            boxes = merged_boxes if merge_boxes else _filtered_boxes(step, box_thr, max_area)
            if not boxes:
                mask_flat_info.append(None)
                continue
            mask = np.zeros((grid_h, grid_w), dtype=bool)
            for x1, y1, x2, y2 in boxes:
                r0 = max(0, int(y1 * grid_h))
                r1 = min(grid_h, max(r0 + 1, round(y2 * grid_h)))
                c0 = max(0, int(x1 * grid_w))
                c1 = min(grid_w, max(c0 + 1, round(x2 * grid_w)))
                mask[r0:r1, c0:c1] = True
            n_in  = int(mask.sum())
            n_out = grid_h * grid_w - n_in
            if n_in == 0 or n_out == 0:
                mask_flat_info.append(None)
            else:
                mask_flat_info.append((mask.ravel(), n_in, n_out))
        while len(mask_flat_info) < n_steps:
            mask_flat_info.append(None)

        # Outer loop: group by (vw, tr) so we build the value-weighted base only once
        # per group and reuse it for both aggregated and per-head combos.
        for vw in VALUE_WEIGHTINGS:
            for tr in TOKEN_REDUCTIONS:
                a = attn_by_tr[tr]   # (n_steps, n_layers, n_heads, n_patches)
                if vw:
                    base4d = a * v_norms[np.newaxis, :, :, :]  # new array, same shape
                else:
                    base4d = a   # view — no copy

                # ---- Aggregated combos for this (vw, tr) ----
                for hr in HEAD_REDUCTIONS:
                    for lr in LAYER_REDUCTIONS:
                        key = f"vw{int(vw)}_hr{hr}_lr{lr}_tr{tr}"

                        # Head reduction → (n_steps, n_layers, n_patches)
                        if hr == "mean":
                            b = base4d.mean(axis=2)
                        elif hr == "min":
                            b = base4d.min(axis=2)
                        else:
                            b = base4d.max(axis=2)

                        # Layer reduction → (n_steps, n_patches)
                        if lr == "sum":
                            b = b.sum(axis=1)
                        else:
                            b = b[:, -1, :]

                        step_scores = {m: [] for m in BBOX_METRICS}
                        for si in range(n_steps):
                            mi = mask_flat_info[si]
                            if mi is None:
                                continue
                            mask_flat, _, _ = mi
                            s = _score_saliency_flat(b[si], mask_flat)
                            for m in BBOX_METRICS:
                                step_scores[m].append(s[m])

                        for m in BBOX_METRICS:
                            vals = step_scores[m]
                            combo_scores[key][m].append(
                                float(np.mean(vals)) if vals else None
                            )

                # ---- Per-layer-per-head combos for this (vw, tr) (vectorized) ----
                _n_s, _n_l, _n_h, _n_p = base4d.shape
                n_lh      = _n_l * _n_h
                flat_base = base4d.reshape(_n_s, n_lh, _n_p)   # (n_steps, n_lh, n_patches)

                sum_mean_in = np.zeros(n_lh, dtype=np.float64)
                sum_sum_in  = np.zeros(n_lh, dtype=np.float64)
                sum_enrich  = np.zeros(n_lh, dtype=np.float64)
                n_valid     = 0

                for si in range(_n_s):
                    mi = mask_flat_info[si]
                    if mi is None:
                        continue
                    mask_flat, _, _ = mi
                    mi_arr, si_arr, en_arr = _score_saliency_batch(flat_base[si], mask_flat)
                    sum_mean_in += mi_arr
                    sum_sum_in  += si_arr
                    sum_enrich  += en_arr
                    n_valid     += 1

                if n_valid > 0:
                    avg_mi = sum_mean_in / n_valid
                    avg_si = sum_sum_in  / n_valid
                    avg_en = sum_enrich  / n_valid
                    for idx in range(n_lh):
                        li  = idx // _n_h
                        hi  = idx % _n_h
                        key = f"vw{int(vw)}_li{li:02d}_hi{hi:02d}_tr{tr}"
                        combo_scores[key]["mean_in"].append(float(avg_mi[idx]))
                        combo_scores[key]["sum_in"].append(float(avg_si[idx]))
                        combo_scores[key]["enrichment"].append(float(avg_en[idx]))
                else:
                    for li in range(_n_l):
                        for hi in range(_n_h):
                            key = f"vw{int(vw)}_li{li:02d}_hi{hi:02d}_tr{tr}"
                            for m in BBOX_METRICS:
                                combo_scores[key][m].append(None)

                if vw:
                    del base4d   # free value-weighted copy

        if (ri + 1) % 10 == 0:
            print(f"  Processed {ri+1}/{len(valid)}", flush=True)

    return combo_scores, correct_arr, all_keys, valid


def analyze(args):
    out_dir     = Path(args.output)
    box_thr     = getattr(args, "box_threshold", None)
    max_area    = getattr(args, "max_box_area", None)
    merge_boxes = getattr(args, "merge_dino_boxes", False)
    combo_scores, correct_arr, all_keys, valid = compute_combo_scores(
        out_dir, box_thr, max_area, merge_boxes=merge_boxes,
    )

    # ---- Correlation statistics ----
    try:
        from scipy import stats as sp_stats
    except ImportError:
        print("scipy not available — pip install scipy")
        return

    n_c = int(correct_arr.sum())
    n_w = len(correct_arr) - n_c
    print(
        f"\nComputing correlations for {len(all_keys)} combos × {len(BBOX_METRICS)} metrics "
        f"({len(valid)} samples, {n_c} correct / {n_w} wrong)...",
        flush=True,
    )

    def _compute_stats_matrix(X, y):
        """Vectorized point-biserial correlation for every combo at once.

        X : (n_combos, n_samples) float64, np.nan where the combo is undefined
            for that sample.  y : (n_samples,) 0/1 correctness.

        Point-biserial r == Pearson correlation between a continuous variable and
        a 0/1 variable, so this is a masked, per-row Pearson computed in a handful
        of numpy reductions — identical r to a per-combo
        `scipy.stats.pointbiserialr` loop but ~10^3x faster on large n (the loop's
        cost scaled with sample count, which made the 25k-sample sweeps take ~80
        min).  p is the standard two-sided t-test p-value on r (df = n-2); it feeds
        only the BH q-values / significance flags, never the |r| ranking.

        Returns per-combo arrays (r, p, mean_diff, n, valid).  Invalid combos
        (n < 10, or zero variance in x or y — e.g. an all-zero saliency map) are
        flagged in `valid` and NaN'd in r/p/mean_diff so the caller drops them,
        exactly as the old n<10 / NaN-r guards did.
        """
        V = ~np.isnan(X)
        n = V.sum(axis=1).astype(np.float64)
        Xf = np.where(V, X, 0.0)
        Yf = np.where(V, y[np.newaxis, :], 0.0)        # y is 0/1
        sx  = Xf.sum(axis=1)
        sy  = Yf.sum(axis=1)                            # per-combo count of correct (n1)
        sxx = np.einsum("ij,ij->i", Xf, Xf)
        sxy = np.einsum("ij,ij->i", Xf, Yf)
        with np.errstate(divide="ignore", invalid="ignore"):
            mx  = sx / n
            my  = sy / n
            cov = sxy / n - mx * my
            vx  = sxx / n - mx * mx
            vy  = my - my * my                         # var of a 0/1 variable = p(1-p)
            r   = cov / np.sqrt(vx * vy)
            n1  = sy
            n0  = n - sy
            mean_diff = sxy / n1 - (sx - sxy) / n0      # mean(correct) - mean(wrong)
            df  = n - 2.0
            t   = r * np.sqrt(df / (1.0 - r * r))
        p = 2.0 * sp_stats.t.sf(np.abs(t), df)
        valid = (n >= 10) & (vx > 0) & (vy > 0) & np.isfinite(r)
        r         = np.where(valid, r, np.nan)
        p         = np.where(valid, p, np.nan)
        mean_diff = np.where(valid, mean_diff, np.nan)
        return {"r": r, "p": p, "mean_diff": mean_diff, "n": n, "valid": valid}

    def _bh_adjust(pvals: list) -> list:
        """
        Benjamini-Hochberg FDR-adjusted p-values (q-values), in the same order
        as `pvals`.  Corrects for the hundreds/thousands of combos tested per
        metric — raw p-values alone would produce many false positives.
        """
        m = len(pvals)
        if m == 0:
            return []
        order = sorted(range(m), key=lambda i: pvals[i])
        q_sorted = [0.0] * m
        prev = 1.0
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            q = pvals[i] * m / (rank + 1)
            prev = min(prev, q)
            q_sorted[rank] = prev
        q = [0.0] * m
        for rank, i in enumerate(order):
            q[i] = min(1.0, q_sorted[rank])
        return q

    # Build and sort results per metric (vectorized point-biserial across all combos)
    results: dict = {}
    for metric in BBOX_METRICS:
        # (n_combos, n_samples) score matrix; None -> np.nan (numpy converts a
        # float-dtype array with None entries to NaN directly).
        X = np.array([combo_scores[key][metric] for key in all_keys], dtype=np.float64)
        st = _compute_stats_matrix(X, correct_arr)
        rows = []
        for i, key in enumerate(all_keys):
            if not st["valid"][i]:
                continue
            rows.append({"key": key, "_idx": i,
                         "r": float(st["r"][i]), "p": float(st["p"][i]),
                         "mean_diff": float(st["mean_diff"][i]),
                         "rank_biserial": None, "n": int(st["n"][i])})
        qvals = _bh_adjust([row["p"] for row in rows])
        for row, q in zip(rows, qvals):
            row["q"] = q
        rows.sort(key=lambda x: -abs(x["r"]))
        # rank_biserial (Mann-Whitney effect size) is printed context only — not
        # used for ranking or significance — so compute it just for the top-N
        # printed rows, at negligible cost instead of once per combo.
        for row in rows[:TOP_N]:
            col = X[row["_idx"]]
            v = ~np.isnan(col)
            vals_c = col[v & (correct_arr == 1.0)]
            vals_w = col[v & (correct_arr == 0.0)]
            if len(vals_c) > 0 and len(vals_w) > 0:
                U, _ = sp_stats.mannwhitneyu(vals_c, vals_w, alternative="two-sided")
                row["rank_biserial"] = float(2 * U / (len(vals_c) * len(vals_w)) - 1)
        for row in rows:
            row.pop("_idx", None)
        del X
        results[metric] = rows

    # ---- Print top-N per metric ----
    for metric in BBOX_METRICS:
        rows = results[metric]
        print(f"\n{'=' * 82}")
        print(f"  Metric: {metric}  —  Top {TOP_N} aggregation methods  (ranked by |r|)")
        print(f"{'=' * 82}")
        hdr = (
            f"{'Rank':>4}  {'Key':<38}  {'r':>8}  {'p':>9}  {'q(BH)':>9}  "
            f"{'mean_diff':>10}  {'rank_bisrl':>10}  {'n':>5}"
        )
        print(hdr)
        print("-" * len(hdr))
        for rank, row in enumerate(rows[:TOP_N], 1):
            star = "*" if row["q"] < 0.05 else " "
            md_s = f"{row['mean_diff']:>+10.4f}" if row["mean_diff"] is not None else f"{'n/a':>10}"
            rb_s = f"{row['rank_biserial']:>+10.4f}" if row["rank_biserial"] is not None else f"{'n/a':>10}"
            print(
                f"{rank:>4}  {row['key']:<38}  {row['r']:>+8.4f}  {row['p']:>9.4f}  "
                f"{row['q']:>8.4f}{star}  {md_s}  {rb_s}  {row['n']:>5}"
            )
        print(f"* q(BH) < 0.05  (Benjamini-Hochberg FDR-adjusted across {len(rows)} combos tested for this metric)")

    # ---- Save JSON ----
    json_name = getattr(args, "json_name", None) or "analysis_results.json"
    json_path = out_dir / json_name
    json_out = {
        metric: {
            "top20": [
                {
                    "rank":          i + 1,
                    "key":           row["key"],
                    "r":             row["r"],
                    "p":             row["p"],
                    "q":             row["q"],
                    "mean_diff":     row["mean_diff"],
                    "rank_biserial": row["rank_biserial"],
                    "n":             row["n"],
                }
                for i, row in enumerate(results[metric][:TOP_N])
            ]
        }
        for metric in BBOX_METRICS
    }
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nSaved analysis results → {json_path}", flush=True)


def reground(args):
    """Re-run Grounding DINO ONLY on an already-collected dir, capturing per-box
    confidence scores so box_threshold / box-area can be tuned later on CPU.

    Reuses the stored observe-step texts and leaves the .npz attention untouched
    (attention is threshold-independent). Adds a 'boxes_scored' field
    ([x1, y1, x2, y2, score]) to each step in metadata.jsonl, grounding at a low
    base threshold so any higher threshold can be applied in 'analyze'.
    Resumable: samples that already have boxes_scored on every step are skipped,
    and metadata is checkpointed periodically.
    """
    import os
    out_dir    = Path(args.output)
    index_path = out_dir / "metadata.jsonl"
    if not index_path.exists():
        print(f"No metadata.jsonl found in {out_dir}")
        sys.exit(1)

    records = [json.loads(l) for l in index_path.open() if l.strip()]
    results = [json.loads(l) for l in Path(args.results).open() if l.strip()]

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from experiments.head_selection.generate import (get_field, _resolve_image,
                                                 _load_dataset_resolved)
    from selfsal.grounding import ground_claim
    from selfsal.grounding.dino import load_local as load_grounding_dino

    print(f"Loading dataset {args.dataset} [split={args.split}] ...", flush=True)
    ds, ds_hf_id = _load_dataset_resolved(args.dataset, args.split)
    print(f"Loading Grounding DINO ...", flush=True)
    dino_proc, dino_model, dino_device = load_grounding_dino()

    def _needs(rec):
        return any("boxes_scored" not in s for s in rec["steps"])

    todo = sum(1 for r in records if _needs(r))
    print(f"{len(records)} records, {todo} to reground at base threshold "
          f"{args.dino_base_threshold}", flush=True)

    def _checkpoint():
        tmp = index_path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        tmp.replace(index_path)

    n_done = 0
    for rec in records:
        if not _needs(rec):
            continue
        try:
            sn        = rec["sample_number"]
            row_index = results[sn].get("image_filename", sn)
            row       = ds[int(row_index)]
            image     = _resolve_image(
                get_field(row, ["decoded_image", "image", "img"]), ds_hf_id
            )
            if hasattr(image, "convert") and image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            print(f"  [warn] sample={rec.get('sample_id')} image load failed: {e}", flush=True)
            continue

        for s in rec["steps"]:
            bas = ground_claim(
                image, s["step_text"],
                box_threshold=args.dino_base_threshold, device=dino_device,
            )
            s["boxes_scored"] = [[*b, float(sc)] for b, sc in bas]

        n_done += 1
        if n_done % 100 == 0:
            _checkpoint()
            print(f"  regrounded {n_done}/{todo} (checkpointed)", flush=True)

    _checkpoint()
    print(f"Done. Wrote scored boxes to {index_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap  = argparse.ArgumentParser(description="Two-step aggregation-correlation analysis")
    sub = ap.add_subparsers(dest="cmd", required=True)

    cp = sub.add_parser("collect", help="Collect raw saliency maps (GPU required)")
    cp.add_argument("--results", default=str(DEFAULT_RESULTS),
                    help="Inference JSONL (needs response / correct fields).")
    cp.add_argument("--output",  default=str(DEFAULT_OUTPUT),
                    help="Output directory for metadata.jsonl and per-sample .npz files.")
    cp.add_argument("--model",   default=DEFAULT_MODEL)
    cp.add_argument("--dataset", default=DEFAULT_DATASET)
    cp.add_argument("--split",   default=DEFAULT_SPLIT)
    cp.add_argument("--box-threshold", type=float, default=0.3,
                    help="Grounding DINO confidence threshold (default: 0.3). Ignored when "
                         "--bbox-source=human.")
    cp.add_argument("--bbox-source", choices=BBOX_SOURCES, default="dino",
                    help="Source of ground-truth bounding boxes to score saliency against "
                         "(default: dino). 'dino' grounds each observe step independently "
                         "via Grounding DINO. 'human' reuses the dataset's single "
                         f"human-annotated '{HUMAN_BBOX_FIELD}' column (e.g. "
                         "peterant330/saliency-r1-8k) for every observe step in the sample.")
    cp.add_argument("--save-all-boxes", action="store_true",
                    help="Ground DINO at a low base threshold (--dino-base-threshold) and store "
                         "per-box confidence scores (boxes_scored) so box_threshold / box-area "
                         "can be tuned later in analyze without re-running DINO. Recommended for "
                         "large collects. Only affects --bbox-source=dino.")
    cp.add_argument("--dino-base-threshold", type=float, default=0.05,
                    help="Base DINO threshold used when --save-all-boxes is set (default: 0.05).")
    cp.add_argument("--dino-device", type=str, default=None,
                    help="Device for Grounding DINO (default: cuda). Pass 'cpu' to avoid OOM "
                         "when the VLM already fills GPU memory.")
    cp.add_argument("--dino-gpu", type=int, default=None,
                    help="Physical GPU index to dedicate to Grounding DINO. When set, "
                         "CUDA_VISIBLE_DEVICES exposes both --gpu and --dino-gpu; the VLM "
                         "loads on cuda:0 and DINO on cuda:1. Typical use: --gpu 0 --dino-gpu 1.")
    cp.add_argument("--limit",   type=int, default=None,
                    help="Process only the first N samples.")
    cp.add_argument("--max-obs-tokens", type=int, default=4000,
                    help="Cap on total observe tokens per sample (default: 4000).")
    cp.add_argument("--num-gpus", type=int, default=1,
                    help="Number of GPUs to use. Spawns one subprocess per GPU automatically "
                         "(default: 1, i.e. single-process mode).")
    cp.add_argument("--rank", type=int, default=0,
                    help=argparse.SUPPRESS)
    cp.add_argument("--world-size", type=int, default=1,
                    help=argparse.SUPPRESS)
    cp.add_argument("--gpu", type=int, default=None,
                    help=argparse.SUPPRESS)
    cp.add_argument("--reuse-queue", action="store_true",
                    help="Keep the existing shared work-queue counter instead of resetting "
                         "it. Passed by scripts/collect_with_restart.sh when relaunching a "
                         "rank mid-run after an OOM-wedge exit (code 17).")

    ap2 = sub.add_parser("analyze", help="Sweep aggregation combos and print correlations (CPU only)")
    ap2.add_argument("--output", default=str(DEFAULT_OUTPUT),
                     help="Directory written by 'collect'.")
    ap2.add_argument("--box-threshold", type=float, default=None,
                     help="Re-threshold DINO boxes at analysis time (requires 'boxes_scored' "
                          "from the 'reground' subcommand). Keeps boxes with score >= this value.")
    ap2.add_argument("--max-box-area", type=float, default=None,
                     help="Drop boxes whose area fraction exceeds this cap (e.g. 0.6 to remove "
                          "near-full-frame degenerate boxes). Works with or without boxes_scored.")
    ap2.add_argument("--merge-dino-boxes", action="store_true",
                     help="Chain-level merged-DINO bbox source: union every observe step's "
                          "filtered DINO boxes into one shape and score EVERY step against it "
                          "(the DINO analogue of a question-level human box). Per-box "
                          "--box-threshold/--max-box-area filtering is applied BEFORE the union. "
                          "Pass --box-threshold (e.g. 0.10) so boxes_scored (DINO) is read rather "
                          "than the 'boxes' field. Correlation stays chain-level.")
    ap2.add_argument("--json-name", type=str, default=None,
                     help="Filename for the saved results JSON (default: analysis_results.json). "
                          "Use a distinct name to avoid clobbering canonical results, e.g. "
                          "analysis_results_merged_dino.json.")

    rp = sub.add_parser("reground",
                        help="Re-run DINO only on a collected dir, saving scored boxes for "
                             "threshold tuning (GPU; reuses stored attention).")
    rp.add_argument("--output", required=True,
                    help="Collected directory containing metadata.jsonl.")
    rp.add_argument("--results", required=True,
                    help="The SAME inference JSONL used at collect time (for image mapping).")
    rp.add_argument("--dataset", default=DEFAULT_DATASET)
    rp.add_argument("--split",   default=DEFAULT_SPLIT)
    rp.add_argument("--dino-base-threshold", type=float, default=0.05,
                    help="Low base threshold to capture all boxes with scores (default: 0.05).")
    rp.add_argument("--gpu", type=int, default=None,
                    help="Pin to a specific GPU via CUDA_VISIBLE_DEVICES.")

    args = ap.parse_args()
    if args.cmd == "collect":
        if args.num_gpus > 1 and args.world_size == 1:
            import subprocess

            def _strip_arg(argv, name):
                out, i = [], 0
                while i < len(argv):
                    if argv[i] == name:
                        i += 2
                    elif argv[i].startswith(name + "="):
                        i += 1
                    else:
                        out.append(argv[i])
                        i += 1
                return out

            base_argv = _strip_arg(sys.argv[1:], "--num-gpus")
            procs = []
            for k in range(args.num_gpus):
                cmd = [sys.executable, __file__] + base_argv + [
                    f"--world-size={args.num_gpus}",
                    f"--rank={k}",
                    f"--gpu={k}",
                ]
                print(f"Launching rank {k}: GPU {k}", flush=True)
                procs.append(subprocess.Popen(cmd))
            exit_codes = [p.wait() for p in procs]
            if any(c != 0 for c in exit_codes):
                print(f"Some ranks failed: {exit_codes}", flush=True)
                sys.exit(1)
        else:
            collect(args)
    elif args.cmd == "reground":
        reground(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
