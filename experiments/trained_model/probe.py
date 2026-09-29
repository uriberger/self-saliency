#!/usr/bin/env python
"""Offline probe of the attention-overlap reward: no training, no optimizer.

Generates completions from one or more checkpoints on the *same* prompts and reports,
for every completion, each reward the trainer computes plus the per-observe-step
breakdown the reward averages over -- which training discards (`np.mean(vals)` in
think_overlap_reward keeps only the mean).

Mirrors the training path exactly: same SYSTEM_PROMPT, same 512px image cap, same
sampling params (temperature 1.0, top_p 1.0, top_k off), same format regex, same
layer-L single-layer attention re-forward, same FLAN-T5 observe segmentation, same
Grounding-DINO grounding, same metric, same weighted-nansum reward and the same
group-normalised advantage.

Per shard it writes one JSON. `--render` merges shard JSONs into a readable report.

    # one shard per GPU
    CUDA_VISIBLE_DEVICES=0 python overlap_probe.py --shard 0 --num-shards 8 ...
    # then
    python overlap_probe.py --render --out-dir <dir>
"""

from __future__ import annotations

import argparse
import base64
import copy
import gc
import importlib.util
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]   # the repository root


def repo_path(rel: str) -> Path:
    """Resolve a repo-relative path, falling back to the central tree.

    Worktrees only symlink the paths in .worktree-links; gitignored corpora such as
    cold_data/grpo_sets/ are untracked in the central tree, so they exist neither in
    the worktree checkout nor as a link. Resolve those against the central tree
    (.worktrees/<branch>/ -> two levels up) instead of failing with a confusing
    "couldn't find any data file" from load_dataset.
    """
    p = REPO / rel
    if p.exists():
        return p
    if REPO.parent.name == ".worktrees":
        alt = REPO.parent.parent / rel
        if alt.exists():
            return alt
    return p


# The training contract, from the one place that owns it. Must match or the model is
# off-distribution and the probe measures the wrong policy.
from selfsal.data.prompt import (FORMAT_PATTERN, IMAGE_TOKEN_ID,   # noqa: E402
                                 MAX_IMAGE_SIDE, SYSTEM_PROMPT, prepare_image)
from selfsal import grounding as GROUND                            # noqa: E402
from selfsal import steps as STEPS                                 # noqa: E402
from selfsal import saliency as SAL                                # noqa: E402


# The probe supported three maps -- the attention map the published arms were trained on,
# plus a pixel-gradient one and a GLIMPSE one that were explored and dropped. Only the
# attention map has a number in the paper, so only it came across. `--map grad` and
# `--map glimpse` therefore fail when the map is BUILT, with a sentence saying where the
# code went; see experiments/_unavailable.py for the rule that keeps that failure late.
from experiments._unavailable import unavailable                   # noqa: E402

GM = unavailable("grad maps", "trl/grad_maps.py")
GREW = unavailable("the roll-null gradient reward", "trl/rewards/grad_rewards.py")
GLM = unavailable("GLIMPSE maps", "trl/glimpse_maps.py", "trl/rewards/glimpse_rewards.py")

# Which per-step number `score` (and therefore the aggregated overlap) reports:
# "metric" = the configured overlap metric, "logratio" = the roll-null gradient score.
# Set from --score in main(); a module global because score_steps is called from both
# the shard path and the re-scoring path.
SCORE_KIND = "metric"


# The grounding/scoring settings for this run. These used to be reached out of the
# reward module's module-level _CFG; the reward is a library here, so the probe holds
# its own. Set once by main() from the flags.
CFG = {
    "box_threshold": GROUND.DEFAULT_BOX_THRESHOLD,
    "max_box_area": 0.5,
    "max_union_area": None,
    "metric": "phi",
    "dino_api_base": None,
    "dino_device": None,
    "dino_batch_size": 32,
}


def _union(boxes, gh, gw, apply_union_cap=True):
    """This run's union mask, with the caps the flags asked for."""
    return GROUND.union_mask(
        boxes, gh, gw,
        max_box_area=CFG["max_box_area"],
        max_union_area=CFG["max_union_area"] if apply_union_cap else None)


# ---------------------------------------------------------------------------
# accuracy reward (verbatim from grpo_vlm_qwen3.py, inlined to avoid importing
# the trainer module and its deepspeed/accelerate stack)
# ---------------------------------------------------------------------------
def accuracy_reward(completions, solution, **kwargs):
    from math_verify import LatexExtractionConfig, parse, verify
    from math_verify.parser import NormalizationConfig

    rewards = []
    contents = [c[0]["content"] for c in completions]
    for content, sol in zip(contents, solution):
        m = re.search(r"</think>\s*(.*)", content, re.DOTALL)
        answer_text = m.group(1).strip() if m else content.strip()
        try:
            gold_parsed = parse(sol, extraction_mode="first_match")
        except Exception:
            gold_parsed = []
        if len(gold_parsed) != 0:
            try:
                answer_parsed = parse(
                    answer_text,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False, malformed_operators=False, basic_latex=True,
                                boxed="all", units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception:
                reward = None
        else:
            reward = float(answer_text.lower() == sol.strip().lower())
        rewards.append(reward)
    return rewards


def judge_format(response: str) -> bool:
    """The trainer's format-validity test (gates the overlap reward multiplicatively)."""
    return (
        re.match(FORMAT_PATTERN, response, re.DOTALL | re.MULTILINE) is not None
        and response.count("<think>") == 1
        and response.count("</think>") == 1
    )


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
# `prepare_image` is imported from `selfsal.data.prompt` at the top of this file.
# A verbatim copy of it used to sit here and shadow that import -- same filter, same
# order, and verified pixel-identical over 42 cases (7 sizes x 6 image modes), so nothing
# about the probe's numbers changes by deleting it. What changes is that there is now one
# implementation: the copy carried the archive's `resize(..., 2)  # BICUBIC`, whose
# comment is wrong, and a reader fixing the comment would have silently switched this
# probe to a filter the published runs never used.


def load_samples(dataset_path: str, n: int, seed: int, cache_tag: str = "",
                 split: str = "train", exclude=None):
    """Sample n rows from `dataset_path`.

    `split` selects which side of the trainer's holdout to draw from:
      train   -- the rows the model trained on (the trainer's 100-row holdout removed)
      holdout -- those 100 held-out rows, still from the *training* corpus
      all     -- the whole dataset, no carve-out. This is the one to use with the
                 dedicated validation sets (cold_data/grpo_sets/val_natural,
                 val_nonnatural): they are image-disjoint from set_a/set_b, so there
                 is no holdout to carve and taking 'train' would silently discard
                 100 of their 256 rows.

    `exclude` is a set of row indices -- in the same (post-carve) space as the
    returned `row_index` -- to draw AROUND, so a confirmation set can be made
    disjoint from an earlier draw.
    """
    from datasets import load_dataset, load_from_disk

    if os.path.isfile(os.path.join(dataset_path, "state.json")):
        ds = load_from_disk(dataset_path)
        if hasattr(ds, "keys"):
            ds = ds["train"]
    elif os.path.isdir(dataset_path):
        ds = load_dataset(dataset_path, split="train")
    else:
        raise SystemExit(
            f"dataset not found: {dataset_path}\n"
            "If you are running from a worktree, cold_data/ is untracked and is not "
            "symlinked in -- pass --dataset with an absolute path to the central tree."
        )
    # The trainer holds out 100 rows with seed 42 before training; reproduce that same
    # split here so 'train' means exactly the prompts the model trained on.
    # Per-shard indices cache files: 8 concurrent shards writing the dataset's shared
    # cache-*.arrow would race on the same filenames.
    if split != "all":
        # A val set carved by --split train would silently drop 100 of its rows and the
        # report would still be labelled "val". Refuse rather than mislabel.
        if Path(dataset_path).name.startswith("val"):
            raise SystemExit(
                f"{dataset_path} looks like a validation set but --split is '{split}', "
                "which carves out the trainer's 100-row holdout. Pass --split all."
            )
        if len(ds) <= 100:
            raise SystemExit(
                f"--split {split} needs the trainer's 100-row holdout, but {dataset_path} "
                f"has only {len(ds)} rows. Validation sets have no holdout to carve: "
                "pass --split all."
            )
        tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"probe_split{cache_tag}"
        tmp.mkdir(parents=True, exist_ok=True)
        parts = ds.train_test_split(
            test_size=100, seed=42,
            train_indices_cache_file_name=str(tmp / "train.arrow"),
            test_indices_cache_file_name=str(tmp / "test.arrow"),
        )
        ds = parts["train"] if split == "train" else parts["test"]
    rng = np.random.default_rng(seed)
    # Two independent draws of 2,000 from 50,000 overlap by ~48 rows. Small, but a
    # confirmation set that shares rows with the selection set is not one.
    pool = np.arange(len(ds))
    if exclude:
        pool = pool[~np.isin(pool, np.asarray(sorted(exclude), dtype=np.int64))]
        if len(pool) < n:
            raise SystemExit(f"only {len(pool)} rows remain after excluding "
                             f"{len(exclude)}; asked for {n}")
    idx = sorted(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())
    rows = []
    for i in idx:
        r = ds[int(i)]
        rows.append(
            {
                "row_index": int(i),
                "dataset": r.get("dataset"),
                "question_id": r.get("question_id"),
                "question": r["problem"],
                "gt_answer": r.get("solution"),
                "image": prepare_image(r["image"]),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def load_model(base_path: str, adapter: str | None, device: str, attn_impl: str):
    import transformers
    from transformers import AutoConfig, AutoProcessor

    processor = AutoProcessor.from_pretrained(base_path, padding_side="left")
    # Resolve the architecture from the config exactly as GRPOTrainer does, so the
    # probe instantiates the same class the training run did.
    config = AutoConfig.from_pretrained(base_path)
    architecture = getattr(transformers, config.architectures[0])
    model = architecture.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, attn_implementation=attn_impl
    )
    if adapter:
        from peft import PeftModel

        # checkpoint-N from the GRPO run is a PEFT adapter (q_proj/v_proj LoRA), not a
        # full model. Merging in-memory avoids a separate merge job AND keeps the
        # module tree plain, so the layer-L attention hook still finds
        # Qwen3VLTextAttention rather than a PEFT wrapper.
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model = model.to(device).eval()
    return processor, model


def build_prompt(processor, question: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(processor, model, image, question, n_gen, max_new_tokens, temperature, device):
    text = build_prompt(processor, question)
    inputs = processor(
        text=[text], images=[[image]], return_tensors="pt",
        padding=True, padding_side="left", add_special_tokens=False,
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        top_k=0,
        max_new_tokens=max_new_tokens,
        num_return_sequences=n_gen,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    comp = out[:, prompt_len:]
    eos = processor.tokenizer.eos_token_id
    seqs = []
    for row in comp:
        ids = row.tolist()
        if eos in ids:
            ids = ids[: ids.index(eos) + 1]          # keep the eos, drop right padding
            truncated = False
        else:
            truncated = True                          # hit max_new_tokens: no </think>
        seqs.append((ids, truncated))
    return inputs, prompt_len, seqs


# ---------------------------------------------------------------------------
# layer-L attention capture (mirrors GRPOTrainer._compute_overlap_step_maps)
# ---------------------------------------------------------------------------
def find_attn_module(model, layer: int):
    for m in model.modules():
        if type(m).__name__ == "Qwen3VLTextAttention" and getattr(m, "layer_idx", None) == layer:
            return m
    return None


def teacher_forced_case(prompt_inputs, comp_ids, device):
    """prompt ++ one completion, as the single teacher-forced forward both maps read off.

    `comp_ids` is generate()'s `list[int]` (it returns `row.tolist()`), not a tensor.
    `mm_token_type_ids` must be extended over the completion with zeros -- text -- or the
    forward differs from the one training runs, which would silently change the quantity
    being measured. Both saliency paths call this so they cannot drift apart again.
    """
    ids = torch.cat(
        [prompt_inputs["input_ids"], torch.tensor([comp_ids], device=device)], dim=1
    )
    case = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if "pixel_values" in prompt_inputs:
        case["pixel_values"] = prompt_inputs["pixel_values"]
        case["image_grid_thw"] = prompt_inputs["image_grid_thw"]
    if prompt_inputs.get("mm_token_type_ids") is not None:
        zeros = torch.zeros(1, len(comp_ids), dtype=torch.long, device=device)
        case["mm_token_type_ids"] = torch.cat(
            [prompt_inputs["mm_token_type_ids"], zeros], dim=1
        )
    return case


@torch.no_grad()
def capture_layer_attention(model, attn_mod, prompt_inputs, prompt_len, comp_ids, heads, device):
    """Return [n_heads_sel, think_span_all, n_image_patches] float32 for one completion.

    Re-runs only layer L in eager mode via a forward hook, exactly as the trainer does,
    so the weights are numerically identical to an all-eager forward at ~1/36 the cost.
    """
    cap = {"attn": None}
    mask_holder = {"m": None}
    reentry = {"in": False}
    mdtype = next(model.parameters()).dtype
    min_val = torch.finfo(mdtype).min

    def hook(module, args, kwargs, output):
        if reentry["in"]:
            return
        reentry["in"] = True
        kw = dict(kwargs)
        kw["attention_mask"] = mask_holder["m"]
        kw["past_key_values"] = None
        kw["use_cache"] = False
        prev = module.config._attn_implementation
        module.config._attn_implementation = "eager"
        try:
            cap["attn"] = module(*args, **kw)[1]
        finally:
            module.config._attn_implementation = prev
            reentry["in"] = False

    handle = attn_mod.register_forward_hook(hook, with_kwargs=True)
    try:
        case = teacher_forced_case(prompt_inputs, comp_ids, device)
        ids = case["input_ids"]
        seq = ids.shape[-1]
        masked = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=device), diagonal=1)
        add = torch.zeros(seq, seq, dtype=mdtype, device=device)
        add.masked_fill_(masked, min_val)
        mask_holder["m"] = add[None, None]
        cap["attn"] = None
        model(**case)
        attn = cap["attn"]
        del add, masked
        image_mask = prompt_inputs["input_ids"][0] == IMAGE_TOKEN_ID
        raw = attn[:, heads, prompt_len:, :prompt_len][:, :, :, image_mask]
        per_tok = torch.relu(raw)[0].float().cpu().numpy()
        del attn, raw
        return per_tok
    finally:
        handle.remove()


def step_maps_from_attention(per_tok, steps, gh, gw, token_reduction):
    """steps: [(text, tok_a, tok_b)] in `out` token space.

    per_tok rows start at completion token 0 (the capture keeps every completion row,
    not just the think span), so the step spans index it directly -- unlike the trainer,
    which slices to the think span first and subtracts `ts`.
    """
    maps = []
    for text, tok_a, tok_b in steps:
        la, lb = tok_a, tok_b
        if lb <= la or lb > per_tok.shape[1]:
            continue
        seg = per_tok[:, la:lb, :]
        if token_reduction == "max":
            red = seg.max(axis=1)
        elif token_reduction == "min":
            red = seg.min(axis=1)
        else:
            red = seg.mean(axis=1)
        m = np.maximum(red.mean(axis=0), 0.0)
        if m.size != gh * gw:
            continue
        maps.append({"map": m.reshape(gh, gw).astype(np.float32), "text": text,
                     "tok_a": int(tok_a), "tok_b": int(tok_b)})
    return maps


# ---------------------------------------------------------------------------
# per-step scoring (think_overlap_reward, but retaining the per-step values)
# ---------------------------------------------------------------------------
def _b64_u8(arr) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).decode("ascii")


def quantize_map(smap):
    """uint8 map normalised to its OWN peak, base64'd, row-major (gh, gw).

    Peak-normalised rather than absolute so one byte spans the full within-map range;
    `map_max` is stored alongside, so the absolute value of a patch is exactly
    `q / 255 * map_max`. Keeping both is the point: the flattening this probe is
    chasing shows up as a *shape* change under peak-normalisation and as a *dimming*
    on the absolute scale, and the viewer needs to switch between the two.
    """
    mx = float(smap.max())
    if mx <= 0:
        q = np.zeros(smap.shape, dtype=np.uint8)
    else:
        q = np.clip(np.rint(255.0 * (smap / mx)), 0, 255).astype(np.uint8)
    return _b64_u8(q)


def score_steps(all_step_maps, images_per_completion, store_maps=True):
    """all_step_maps: list (per completion) of step dicts. Returns per-completion detail.

    One batched DINO call over every (completion, step) pair, exactly like
    think_overlap_reward does.
    """
    flat_images, flat_texts, owner = [], [], []
    for c, steps in enumerate(all_step_maps):
        for si, st in enumerate(steps):
            flat_images.append(images_per_completion[c])
            flat_texts.append(st["text"])
            owner.append((c, si))
    boxes_per_item = GROUND.ground(flat_images, flat_texts, box_threshold=CFG["box_threshold"],
                                 api_base=CFG["dino_api_base"],
                                 batch_size=CFG["dino_batch_size"],
                                 device=CFG["dino_device"]) if flat_images else []

    detail = [[] for _ in all_step_maps]
    for (c, si), boxes in zip(owner, boxes_per_item):
        st = all_step_maps[c][si]
        smap = st["map"]
        gh, gw = smap.shape
        max_area = CFG.get("max_box_area")
        box_cap_on = max_area is not None and float(max_area) > 0
        kept = [b for b in (boxes or []) if not box_cap_on or GROUND.box_area(b) <= max_area]
        mask = _union(boxes or [], gh, gw)
        # _union_mask returns None for three different reasons and the viewer should not
        # conflate them: nothing grounded, a degenerate/full union, or the union cap.
        # Re-rasterise the kept boxes without the cap to tell the last one apart.
        uncapped = _union(boxes or [], gh, gw, apply_union_cap=False)
        union_frac_uncapped = (
            float(uncapped.sum()) / float(uncapped.size) if uncapped is not None else None
        )
        dropped_by_union_cap = mask is None and uncapped is not None
        rec = {
            "step_index": si,
            "text": st["text"],
            "tok_a": st["tok_a"],
            "tok_b": st["tok_b"],
            "n_tokens": st["tok_b"] - st["tok_a"],
            "n_boxes_raw": len(boxes or []),
            "n_boxes_kept": len(kept),
            "max_box_area": max(( GROUND.box_area(b) for b in (boxes or [])), default=None),
            # Union coverage BEFORE --max-union-area, so a capped step still reports the
            # coverage that got it dropped (box_area_frac below is None in that case).
            "union_frac_uncapped": union_frac_uncapped,
            "dropped_by_union_cap": dropped_by_union_cap,
            "image_mass": float(smap.sum()),
            "map_max": float(smap.max()),
            "map_mean": float(smap.mean()),
        }
        if store_maps:
            # Everything the viewer needs to redraw this step over the image: the
            # patch grid, the peak-normalised map, and the DINO boxes both as proposed
            # and as they survive the max_box_area filter (only `kept` feeds the mask,
            # so seeing what was thrown away explains a low box_area_frac).
            rec["grid"] = [int(gh), int(gw)]
            rec["map_q"] = quantize_map(smap)
            rec["boxes_raw"] = [[round(float(v), 5) for v in b] for b in (boxes or [])]
            rec["boxes_kept"] = [[round(float(v), 5) for v in b] for b in kept]
            # Draw the capped-away union too, otherwise a dropped step renders with no
            # overlay at all and looks identical to an ungrounded one.
            shown = mask if mask is not None else uncapped
            rec["mask_q"] = _b64_u8(shown.astype(np.uint8)) if shown is not None else None
        if mask is None:
            rec.update(grounded=False, box_area_frac=None, score=None,
                       mean_in_raw=None, mean_in_v2_raw=None, auroc_raw=None,
                       logratio_raw=None, ecc=None,
                       note=(f"union covers {union_frac_uncapped:.2f} of the image > "
                             f"--max-union-area {CFG['max_union_area']} -> SKIPPED, not scored 0"
                             if dropped_by_union_cap else
                             "DINO could not ground this step (or union degenerate) -> SKIPPED, not scored 0"))
        else:
            # `score` is the configured metric (with the mass gate) and is what the
            # reward uses. Every metric is also recorded ungated on every step, so they
            # can be compared on identical maps and masks -- a paired comparison of
            # metrics rather than one across runs with different completions.
            # logratio_raw is the roll-null score (chance 0), recorded on every step
            # whatever the map is -- it is the number --map grad exists to measure, and
            # on an attention map it is the paired comparison against mean_in/auroc.
            # logratio_raw is the roll-null score (chance 0), recorded once per step
            # whatever the map is -- it is the number --map grad exists to measure, and
            # on an attention map it is the paired comparison against mean_in/auroc.
            # Computed ONCE: it draws random control placements, so calling it twice
            # would give the step two different scores and double the diagnostics.
            _lr = GREW.step_logratio(smap, mask)
            rec.update(
                grounded=True,
                box_area_frac=float(mask.sum()) / float(mask.size),
                # The roll-null DRAWS random control placements, so asking for it twice
                # gives the step two different scores. `_lr` above is the single draw;
                # --overlap-metric logratio therefore reuses it rather than letting
                # _step_score draw again with the other module's rng and knobs.
                score=(_lr if (SCORE_KIND == "logratio"
                               or CFG.get("metric") == "logratio")
                       else SAL.score_step(smap, mask, CFG["metric"])),
                mean_in_raw=SAL.phi(smap, mask),
                mean_in_v2_raw=SAL.phi_mean(smap, mask),
                auroc_raw=SAL.auroc(smap, mask),
                logratio_raw=_lr,
                ecc=GROUND.centroid_eccentricity(mask),
                note="",
            )
        detail[c].append(rec)
    return detail


def overlap_from_detail(detail, format_valid):
    vals = [d["score"] for d in detail if d["grounded"] and d["score"] is not None]
    if not vals:
        return None, vals
    return float(np.mean(vals)) * (1.0 if format_valid else 0.0), vals


# ---------------------------------------------------------------------------
# main shard
# ---------------------------------------------------------------------------
def grad_step_maps(model, processor, prompt_inputs, prompt_len, comp_ids, steps, device,
                   grad_target="clogit", span_chunk=None):
    """The GRPO gradient reward's map, per observe step -- the same call the trainer makes.

    Same [{"map", "text", "tok_a", "tok_b"}] contract as step_maps_from_attention, so the
    scoring, the HTML and every aggregate below are shared. `comp_ids` is one completion;
    the prompt is re-forwarded with it, teacher-forced, exactly as in training.
    """
    ip = processor.image_processor
    ps = int(getattr(ip, "patch_size", 16))
    tps = int(getattr(ip, "temporal_patch_size", 2))
    case = teacher_forced_case(prompt_inputs, comp_ids, device)
    ids = case["input_ids"]
    kept, spans = [], []
    for text, tok_a, tok_b in steps:
        a, b = prompt_len + tok_a, prompt_len + tok_b
        if b <= a or a <= 0 or b > ids.shape[1]:
            continue
        spans.append((a, b))
        kept.append((text, tok_a, tok_b))
    if not spans:
        return []
    grid = prompt_inputs["image_grid_thw"][0].tolist()
    cuda = isinstance(device, str) and device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    with GM.frozen_params(model):
        maps = GM.step_grad_maps(model, case, spans, grid, ps, tps, target=grad_target,
                                 span_chunk=span_chunk)
    if cuda:
        # The dial that decides whether this pass fits: peak scales with the chunk, not
        # with the completion's step count. Logged every call so a run that OOMs says how
        # far over it went, and one that survives says how much room --grad-span-chunk has.
        print(f"[grad-mem] spans={len(spans)} chunk={span_chunk or len(spans)} "
              f"len={ids.shape[1]} peak={torch.cuda.max_memory_allocated(device) / 2**30:.1f}GiB",
              file=sys.stderr, flush=True)
    return [{"map": maps[k].astype(np.float32), "text": t, "tok_a": a, "tok_b": b}
            for k, (t, a, b) in enumerate(kept)]


def glimpse_step_maps(model, processor, prompt_inputs, prompt_len, comp_ids, steps,
                      question, device, args):
    """The GLIMPSE reward's map, per observe step -- the same call the trainer makes.

    Same [{"map", "text", "tok_a", "tok_b"}] contract as the other two producers, so the
    scoring, the HTML and every aggregate below are shared. This is the run that sets
    `w_glimpse`: `score_steps` records ALL FOUR `*_raw` metrics on every grounded step, so
    ONE probe run calibrates every variant on identical maps and masks -- a paired
    comparison, which separate runs could not give. The 2026-08-24 run did exactly that
    (result 7 in docs/probe-results.md); the weights it produced are in
    trl/rewards/glimpse_rewards.py, and they depend on `--max-union-area`.
    """
    case = teacher_forced_case(prompt_inputs, comp_ids, device)
    ids = case["input_ids"]
    kept, spans = [], []
    for text, tok_a, tok_b in steps:
        a, b = prompt_len + tok_a, prompt_len + tok_b
        if b <= a or a <= 0 or b > ids.shape[1]:
            continue
        spans.append((a, b))
        kept.append((text, tok_a, tok_b))
    if not spans:
        return []
    grid = prompt_inputs["image_grid_thw"][0].tolist()
    cuda = isinstance(device, str) and device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    with GM.frozen_params(model):
        maps, info = GLM.step_glimpse_maps(
            model, case, spans, grid, question=question, tokenizer=processor.tokenizer,
            prompt_len=prompt_len, target=args.glimpse_target,
            layer_frac=args.glimpse_layer_frac, temp=args.glimpse_temp,
            depth_temp=args.glimpse_depth_temp, token_weight=args.glimpse_token_weight,
            token_cap=args.glimpse_token_cap, seed=args.glimpse_seed)
    if cuda:
        # Cost is linear in target tokens and this is the only place that says how many a
        # real completion actually had -- the number every projection off this run rests on.
        print(f"[glimpse-mem] spans={len(spans)} tokens={info.get('n_target_tokens')} "
              f"layers={info.get('n_layers', 0) - info.get('first_layer', 0)} "
              f"len={ids.shape[1]} "
              f"peak={torch.cuda.max_memory_allocated(device) / 2**30:.1f}GiB",
              file=sys.stderr, flush=True)
    return [{"map": maps[k].astype(np.float32), "text": t, "tok_a": a, "tok_b": b}
            for k, (t, a, b) in enumerate(kept)]


def run_model(spec, rows, args, device):
    processor, model = load_model(spec["path"], spec.get("adapter"), device, args.attn_impl)
    attn_mod = None
    if args.map == "attn":
        attn_mod = find_attn_module(model, args.overlap_layer)
        if attn_mod is None:
            raise RuntimeError(
                f"no Qwen3VLTextAttention with layer_idx={args.overlap_layer}; "
                "the single-layer capture would silently fall back to an all-eager forward"
            )
    clf = STEPS.StepClassifier.load(args.steps_ckpt, device=args.steps_device)
    heads = [int(h) for h in args.overlap_heads.split(",")]
    tok = processor.tokenizer
    out_samples = []
    # Import once, not per sample: the module builds an OpenAI client at import time.
    judge_mod = importlib.import_module("selfsal.judge") if args.judge else None

    for si, row in enumerate(rows):
        prompt_inputs, prompt_len, seqs = generate(
            processor, model, row["image"], row["question"],
            args.num_generations, args.max_new_tokens, args.temperature, device,
        )
        comp_ids = [s[0] for s in seqs]
        truncated = [s[1] for s in seqs]
        output_text = tok.batch_decode(comp_ids, skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
        completions_text = tok.batch_decode(comp_ids, skip_special_tokens=True)
        completions = [[{"role": "assistant", "content": c}] for c in completions_text]
        fmt_valid = [judge_format(c) for c in completions_text]

        # think spans, in the re-tokenised `out` space (identical to the trainer)
        out = tok(output_text)
        ts_idx = [re.search(r"<think>\s*(\S\S*)", t, re.DOTALL | re.MULTILINE) for t in output_text]
        te_idx = [re.search(r"(\S)\s*</think>", t, re.DOTALL | re.MULTILINE) for t in output_text]
        ts_idx = [m.start(1) if m else -1 for m in ts_idx]
        te_idx = [m.start(1) if m else -1 for m in te_idx]
        think_start = [out.char_to_token(b, i) if i >= 0 else -1 for b, i in enumerate(ts_idx)]
        think_end = [out.char_to_token(b, i) if i >= 0 else -1 for b, i in enumerate(te_idx)]
        think_end = [i if v else 0 for i, v in zip(think_end, fmt_valid)]
        think_start = [min(i, z) if v else 0 for i, v, z in zip(think_start, fmt_valid, think_end)]

        gh = int(prompt_inputs["image_grid_thw"][0, 1].item()) // 2
        gw = int(prompt_inputs["image_grid_thw"][0, 2].item()) // 2

        all_maps, all_sentences = [], []
        for c in range(len(comp_ids)):
            ts, te = think_start[c], think_end[c]
            sents = STEPS.split_sentences_with_spans(
                output_text[c][ts_idx[c]: te_idx[c] + 1] if (ts_idx[c] >= 0 and te_idx[c] >= 0) else "",
                base_offset=max(ts_idx[c], 0),
            )
            labels = clf.predict_many([s for s, _, _ in sents],
                                      output_text[c][max(ts_idx[c], 0): te_idx[c] + 1] if te_idx[c] >= 0 else "",
                                      row["question"]) if sents else []
            all_sentences.append([{"text": s, "label": lab} for (s, _, _), lab in zip(sents, labels)])

            if not fmt_valid[c] or te <= ts:
                all_maps.append([])
                continue
            steps = STEPS.segment_sentences(
                output_text[c], ts_idx[c], te_idx[c], out, c, ts, te, row["question"], clf
            )
            if args.map == "grad":
                all_maps.append(grad_step_maps(
                    model, processor, prompt_inputs, prompt_len, comp_ids[c], steps,
                    device, grad_target=args.grad_target, span_chunk=args.grad_span_chunk,
                ))
                continue
            if args.map == "glimpse":
                all_maps.append(glimpse_step_maps(
                    model, processor, prompt_inputs, prompt_len, comp_ids[c], steps,
                    row["question"], device, args,
                ))
                continue
            per_tok = capture_layer_attention(
                model, attn_mod, prompt_inputs, prompt_len, comp_ids[c], heads, device
            )
            # per_tok rows are completion tokens from 0; step spans are in the same space
            all_maps.append(step_maps_from_attention(per_tok, steps, gh, gw, args.token_reduction))
            del per_tok

        # The attention re-forward leaves tens of GB cached but fragmented; DINO then
        # wants one large contiguous block on the same GPU. Return the cache before
        # handing over, or the first sample of a shard can OOM outright.
        torch.cuda.empty_cache()
        detail = score_steps(all_maps, [row["image"]] * len(comp_ids), store_maps=args.store_maps)
        overlap = []
        for c in range(len(comp_ids)):
            ov, _ = overlap_from_detail(detail[c], fmt_valid[c])
            overlap.append(ov)

        # think_format_reward is the same validity flag that gates the overlap reward
        fmt_r = [1.0 if v else 0.0 for v in fmt_valid]
        acc_r = accuracy_reward(completions, [row["gt_answer"]] * len(comp_ids))
        if judge_mod is not None:
            try:
                judge_r = judge_mod.openai_reward(completions, [row["gt_answer"]] * len(comp_ids),
                                                  [row["question"]] * len(comp_ids))
            except Exception as e:  # judge must never kill the probe
                print(f"[probe] judge failed on sample {si}: {e}", flush=True)
                judge_r = [None] * len(comp_ids)
        else:
            judge_r = [None] * len(comp_ids)

        # weighted nansum + group-normalised advantage, exactly as the trainer does
        w = [float(x) for x in args.reward_weights.split()]
        per_func = np.array(
            [[np.nan if v is None else float(v) for v in fmt_r],
             [np.nan if v is None else float(v) for v in overlap],
             [np.nan if v is None else float(v) for v in acc_r],
             [np.nan if v is None else float(v) for v in judge_r]],
            dtype=np.float64,
        ).T
        total = np.nansum(per_func * np.array(w)[None, :], axis=1)
        mean_g = total.mean()
        std_g = total.std(ddof=1)
        adv = (total - mean_g) / (std_g + 1e-4)

        out_samples.append({
            "sample_index": si,
            "row_index": row["row_index"],
            "dataset": row["dataset"],
            "question_id": row["question_id"],
            "question": row["question"],
            "gt_answer": row["gt_answer"],
            "image_file": f"images/{row['dataset']}_{row['question_id']}.png",
            "group": {"reward_mean": float(mean_g), "reward_std": float(std_g)},
            "completions": [
                {
                    "index": c,
                    "text": completions_text[c],
                    "n_completion_tokens": len(comp_ids[c]),
                    "truncated_at_max_tokens": bool(truncated[c]),
                    "format_valid": bool(fmt_valid[c]),
                    "rewards": {
                        "think_format_reward": fmt_r[c],
                        "think_overlap_reward": overlap[c],
                        "accuracy_reward": acc_r[c],
                        "openai_reward": judge_r[c],
                        "weighted_total": float(total[c]),
                        "advantage": float(adv[c]),
                    },
                    "n_observe_steps_scored": sum(1 for d in detail[c] if d["grounded"]),
                    "n_observe_steps_total": len(detail[c]),
                    "observe_steps": detail[c],
                    "all_sentences": all_sentences[c],
                }
                for c in range(len(comp_ids))
            ],
        })
        print(f"[probe] {spec['name']} sample {si + 1}/{len(rows)} done", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(repo_path("cold_data/grpo_sets/set_a")))
    # Which side of the trainer's holdout to probe. Default stays "train" so existing
    # reports remain reproducible; --split all is what the val_* sets want.
    p.add_argument("--split", default="train", choices=["train", "holdout", "all"],
                   help="train: prompts the model trained on (default). holdout: the "
                        "trainer's 100 held-out rows. all: the whole dataset, for the "
                        "dedicated validation sets (val_natural / val_nonnatural).")
    p.add_argument("--n-samples", type=int, default=30)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--base-model", default=str(repo_path("checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged")))
    p.add_argument("--trained-adapter", default=str(repo_path("checkpoint/grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-overlap__wov0.4_2head_trmean_50k_set_a/checkpoint-2000")))
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--overlap-layer", type=int, default=22)
    p.add_argument("--overlap-heads", default="28,31")
    p.add_argument("--token-reduction", default="mean", choices=["mean", "max", "min"])
    p.add_argument("--overlap-metric", default="mean_in",
                   choices=["mean_in", "mean_in_v2", "auroc", "logratio"],
                   help="how a step's map is scored against its box union. 'logratio' is "
                        "the roll-null and is identical to --score logratio; its knobs are "
                        "the --grad-* ones, since the probe computes it once per step.")
    # --- the gradient reward (trl/rewards/grad_rewards.py) ---
    p.add_argument("--map", default="attn", choices=["attn", "grad", "glimpse"],
                   help="which per-step map to build: raw attention at --overlap-layer, "
                        "the PIXEL GRADIENT of the step's own tokens (the map reward_variant"
                        "='grad' scores), or GLIMPSE's gradient-weighted attention (the map "
                        "reward_variant='glimpse' scores). 'grad' and 'glimpse' ignore "
                        "--overlap-layer/--overlap-heads/--token-reduction. 'glimpse' is "
                        "~0.2 s per target token, i.e. 55-59x 'grad' -- budget accordingly.")
    p.add_argument("--score", default="metric", choices=["metric", "logratio"],
                   help="which per-step number is aggregated into the reported overlap: the "
                        "--overlap-metric, or the roll-null log ratio (chance 0). logratio_raw "
                        "is recorded on every step either way, so the two are comparable on "
                        "identical maps and masks. Use --map grad --score logratio to measure "
                        "the spread that sets w_grad.")
    p.add_argument("--grad-target", default="clogit", choices=["clogit", "logit", "logprob"])
    # 4 is the archive's SPAN_CHUNK_DEFAULT, written out rather than read off GM for the
    # same reason --grad-target's choices are: building the parser must not touch the
    # sentinel, or the probe dies before it has read the flag that would have avoided it.
    p.add_argument("--grad-span-chunk", type=int, default=4,
                   help="steps per vmapped backward; 0 = all at once. Pure memory/speed "
                        "dial, identical maps at every value. Lower it if [grad-mem] "
                        "peak approaches the card")
    p.add_argument("--grad-null-offsets", type=int, default=16)
    p.add_argument("--grad-logratio-clip", type=float, default=1.0)
    # --- the GLIMPSE reward (trl/rewards/glimpse_rewards.py). Defaults match the
    # trainer's, so `--map glimpse` measures the map a `--glimpse` run would score.
    p.add_argument("--glimpse-target", default="clogit",
                   choices=["clogit", "logit", "logprob"])
    p.add_argument("--glimpse-layer-frac", type=float, default=1.0,
                   help="fraction of the stack propagated, off the top. 0.6 is 1.64x "
                        "cheaper and a METHOD change; the screened map is 1.0.")
    p.add_argument("--glimpse-token-cap", type=int, default=0,
                   help="target tokens scored per step, drawn at random; 0 = all. Cost is "
                        "exactly linear in this.")
    p.add_argument("--glimpse-temp", type=float, default=0.5)
    p.add_argument("--glimpse-depth-temp", type=float, default=0.2)
    p.add_argument("--glimpse-token-weight", default="full",
                   choices=["full", "confidence", "prompt", "uniform"])
    p.add_argument("--glimpse-seed", type=int, default=0)
    p.add_argument("--grad-inframe-rolls", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--box-threshold", type=float, default=0.10)
    p.add_argument("--max-box-area", type=float, default=0.5,
                   help="per-BOX area cap; 0 disables it (keep every box above --box-threshold)")
    p.add_argument("--max-union-area", type=float, default=None,
                   help="per-STEP cap on the rasterised box union as a fraction of the image; "
                        "steps above it are skipped, not scored 0. None/0 (default) = off")
    p.add_argument("--mass-floor-tau", type=float, default=None)
    p.add_argument("--reward-weights", default="1.0 0.4 1.0 1.0")
    p.add_argument("--dino-device", default=None)
    p.add_argument("--dino-api-base", default=None)
    # Lower than the trainer's 32: here DINO shares the GPU with the 8B VLM and the
    # attention re-forward. _dino_boxes_local halves this on OOM, so it is a starting
    # point, not a ceiling.
    p.add_argument("--dino-batch-size", type=int, default=8)
    p.add_argument("--steps-device", default=None)
    p.add_argument("--steps-ckpt", default=os.environ.get("OVERLAP_STEPS_CKPT", str(repo_path("checkpoint/steps_classifier/best"))))
    # BooleanOptionalAction gives both --judge and --no-judge, so this matches the
    # spelling launch_overlap_probe.sh accepts (the two interfaces drifting apart is
    # exactly what made the hand-run commands fail).
    p.add_argument("--judge", action=argparse.BooleanOptionalAction, default=False,
                   help="query the LLM judge for openai_reward (needs NVIDIA_API_KEY)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--store-maps", action=argparse.BooleanOptionalAction, default=True,
                   help="persist each step's quantised attention map + DINO boxes for the viewer")
    p.add_argument("--skip-base", action="store_true",
                   help="evaluate only the adapters, not the un-adapted base model")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--out-dir", default=str(REPO / "outputs/overlap_probe"))
    p.add_argument("--render", action="store_true", help="merge shard JSONs into a report and exit")
    p.add_argument("--markdown", action="store_true", help="also emit the old probe_report.md")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if args.render:
        render(out_dir, markdown=args.markdown)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    CFG.update(
        box_threshold=args.box_threshold,
        max_box_area=args.max_box_area,
        max_union_area=args.max_union_area,
        metric=SAL.resolve_metric(args.overlap_metric),
        dino_api_base=args.dino_api_base,
        dino_device=args.dino_device or args.device,
        dino_batch_size=args.dino_batch_size,
    )
    GREW.configure(
        null_offsets=args.grad_null_offsets,
        logratio_clip=args.grad_logratio_clip,
        inframe_rolls=args.grad_inframe_rolls,
        seed=args.seed,
    )
    global SCORE_KIND
    SCORE_KIND = args.score
    if args.steps_device is None:
        args.steps_device = args.device

    rows = load_samples(args.dataset, args.n_samples, args.seed,
                        cache_tag=f"_s{args.shard}", split=args.split)
    mine = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    # Each shard exports only its own images: concurrent shards writing the same PNG
    # can interleave and leave a truncated file.
    for r in mine:
        f = out_dir / "images" / f"{r['dataset']}_{r['question_id']}.png"
        # Suffix the whole name, not with_suffix(): that would replace ".png" and PIL
        # then cannot infer the format. Pass format explicitly regardless.
        tmp_f = f.with_name(f.name + f".tmp{args.shard}")
        r["image"].save(tmp_f, format="PNG")
        os.replace(tmp_f, f)
    print(f"[probe] shard {args.shard}/{args.num_shards}: {len(mine)}/{len(rows)} samples", flush=True)
    if not mine:
        return

    # --trained-adapter takes a comma-separated list so several checkpoints are compared
    # on identical prompts in one pass; the label comes from the adapter directory name
    # rather than being hardcoded, so a report can't silently mislabel which step it is.
    # --skip-base still loads the base weights (the LoRA needs them) but does not
    # evaluate the un-adapted model, which is a third of the run when it isn't needed.
    specs = [] if args.skip_base else [{"name": "base_coldstart", "path": args.base_model, "adapter": None}]
    # Each entry may be written NAME=PATH: these directory names differ only by a
    # trailing suffix, so the auto-label is ambiguous and a report could silently
    # mislabel which run a column belongs to.
    for entry in (a.strip() for a in (args.trained_adapter or "").split(",")):
        if not entry or entry.lower() == "none":
            continue
        name, sep, path = entry.partition("=")
        name, path = (name.strip(), path.strip()) if sep else ("", entry)
        if not os.path.isfile(os.path.join(path, "adapter_config.json")):
            raise SystemExit(f"not a LoRA adapter directory (no adapter_config.json): {path}")
        specs.append({"name": name or ("grpo_" + Path(path).name), "path": args.base_model, "adapter": path})

    result = {"config": vars(args), "models": {}}
    for spec in specs:
        print(f"[probe] === model {spec['name']} ===", flush=True)
        result["models"][spec["name"]] = {
            "path": spec["path"], "adapter": spec["adapter"],
            "samples": run_model(spec, mine, args, args.device),
        }
    f = out_dir / f"probe_shard{args.shard:02d}.json"
    f.write_text(json.dumps(result, indent=1, default=str))
    print(f"[probe] wrote {f}", flush=True)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def render(out_dir: Path, markdown: bool = False):
    shards = sorted(out_dir.glob("probe_shard*.json"))
    if not shards:
        # Return rather than raise: the launcher renders unconditionally, and aborting
        # here under `set -e` would swallow its own "shard N failed" summary.
        print(f"[render] nothing to merge: no probe_shard*.json in {out_dir}")
        return
    merged, cfg = {}, None
    for s in shards:
        d = json.loads(s.read_text())
        cfg = cfg or d["config"]
        for name, m in d["models"].items():
            merged.setdefault(name, {"path": m["path"], "adapter": m["adapter"], "samples": []})
            merged[name]["samples"] += m["samples"]
    for m in merged.values():
        m["samples"].sort(key=lambda x: x["row_index"])

    payload = {"config": cfg, "models": merged}
    (out_dir / "probe_merged.json").write_text(json.dumps(payload, indent=1))

    # HTML is the primary report: the thing worth looking at is which sentences became
    # scored observe steps and what each was worth, shown in place in the text.
    from probe_html import write_html

    write_html(payload, out_dir / "probe_report.html", REPO / "assets/validate_palette.js")
    print(f"wrote {out_dir / 'probe_report.html'} and probe_merged.json")
    if not markdown:
        return

    def f(v, nd=4):
        return "None" if v is None else f"{float(v):.{nd}f}"

    L = ["# Attention-overlap reward probe", "",
         "Generation only -- no training, no optimizer step. Same prompts for every model.",
         "",
         f"- dataset: `{cfg['dataset']}` split={cfg.get('split', 'train')}  "
         f"n_samples={cfg['n_samples']} seed={cfg['seed']}",
         f"- generations/sample: {cfg['num_generations']}  temperature={cfg['temperature']}  max_new_tokens={cfg['max_new_tokens']}",
         f"- overlap: layer={cfg['overlap_layer']} heads=[{cfg['overlap_heads']}] token_reduction={cfg['token_reduction']} metric={cfg['overlap_metric']}",
         f"- DINO: box_threshold={cfg['box_threshold']} "
         f"max_box_area={cfg['max_box_area'] or 'off'} "
         f"max_union_area={cfg.get('max_union_area') or 'off'}  mass_floor_tau={cfg['mass_floor_tau']}",
         f"- reward_weights (format, overlap, accuracy, judge): `{cfg['reward_weights']}`",
         "",
         "`score` per observe step is the value the reward averages over. Steps DINO cannot",
         "ground are SKIPPED (not scored 0) -- shown with `grounded=False`.", ""]

    # summary
    L += ["## Summary", "", "| model | completions | mean len (tok) | truncated | format ok | overlap | steps/compl | scored steps/compl | mean step score | mean box area frac | accuracy | judge |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in merged.items():
        cs = [c for s in m["samples"] for c in s["completions"]]
        st = [d for c in cs for d in c["observe_steps"]]
        sc = [d["score"] for d in st if d["grounded"] and d["score"] is not None]
        ba = [d["box_area_frac"] for d in st if d["grounded"] and d["box_area_frac"] is not None]
        ov = [c["rewards"]["think_overlap_reward"] for c in cs if c["rewards"]["think_overlap_reward"] is not None]
        ac = [c["rewards"]["accuracy_reward"] for c in cs if c["rewards"]["accuracy_reward"] is not None]
        ju = [c["rewards"]["openai_reward"] for c in cs if c["rewards"]["openai_reward"] is not None]
        L.append(
            f"| {name} | {len(cs)} | {np.mean([c['n_completion_tokens'] for c in cs]):.0f} | "
            f"{np.mean([c['truncated_at_max_tokens'] for c in cs]):.3f} | "
            f"{np.mean([c['format_valid'] for c in cs]):.3f} | {np.mean(ov) if ov else float('nan'):.4f} | "
            f"{np.mean([c['n_observe_steps_total'] for c in cs]):.2f} | "
            f"{np.mean([c['n_observe_steps_scored'] for c in cs]):.2f} | "
            f"{np.mean(sc) if sc else float('nan'):.4f} | {np.mean(ba) if ba else float('nan'):.3f} | "
            f"{np.mean(ac) if ac else float('nan'):.3f} | {np.mean(ju) if ju else float('nan'):.3f} |"
        )
    L.append("")

    for name, m in merged.items():
        L += [f"# Model: {name}", "", f"path: `{m['path']}`", f"adapter: `{m['adapter']}`", ""]
        for s in m["samples"]:
            L += [f"## {name} — sample row {s['row_index']} ({s['dataset']} / question_id={s['question_id']})", "",
                  f"**Question:** {s['question']}", "",
                  f"**GT answer:** `{s['gt_answer']}`", "",
                  f"**Image:** `{s['image_file']}`", "",
                  "| # | tok | trunc | fmt | overlap | acc | judge | total | adv | steps | scored |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for c in sorted(s["completions"], key=lambda c: -(c["rewards"]["think_overlap_reward"] or -9)):
                r = c["rewards"]
                L.append(f"| {c['index']} | {c['n_completion_tokens']} | {'Y' if c['truncated_at_max_tokens'] else ''} | "
                         f"{f(r['think_format_reward'],2)} | {f(r['think_overlap_reward'])} | {f(r['accuracy_reward'],2)} | "
                         f"{f(r['openai_reward'],2)} | {f(r['weighted_total'],3)} | {f(r['advantage'],3)} | "
                         f"{c['n_observe_steps_total']} | {c['n_observe_steps_scored']} |")
            L.append("")
            for c in sorted(s["completions"], key=lambda c: -(c["rewards"]["think_overlap_reward"] or -9)):
                r = c["rewards"]
                L += [f"### completion {c['index']} — overlap={f(r['think_overlap_reward'])}, adv={f(r['advantage'],3)}, "
                      f"{c['n_completion_tokens']} tokens", "", "```", c["text"].strip(), "```", "",
                      "**All sentences (FLAN-T5 step label):**", ""]
                for i, sn in enumerate(c["all_sentences"]):
                    mark = "**observe**" if sn["label"] == "observe" else sn["label"]
                    L.append(f"{i}. [{mark}] {sn['text']}")
                L += ["", "**Scored observe steps:**", "",
                      "| step | tokens | boxes (raw/kept) | max box area | box area frac | image mass | score |",
                      "|---|---|---|---|---|---|---|"]
                for d in c["observe_steps"]:
                    L.append(f"| {d['step_index']}: {d['text'][:70]} | {d['n_tokens']} | "
                             f"{d['n_boxes_raw']}/{d['n_boxes_kept']} | {f(d['max_box_area'],3)} | "
                             f"{f(d['box_area_frac'],3)} | {f(d['image_mass'],5)} | "
                             f"{f(d['score']) if d['grounded'] else 'SKIPPED (ungrounded)'} |")
                L.append("")
    (out_dir / "probe_report.md").write_text("\n".join(L))
    print(f"wrote {out_dir / 'probe_report.md'}")


if __name__ == "__main__":
    main()
