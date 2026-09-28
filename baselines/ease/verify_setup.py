#!/usr/bin/env python3
# Copyright 2026 NVIDIA. Apache-2.0.
"""Preflight for an EASE run: everything that can fail without a GPU.

Run by `launch_ease_train.sh --preflight`, which hands it the exact override
list the real command would use -- so this checks the run you are about to
submit, not a paraphrase of it.

Three things, in the order they would break:

  1. `import verl` and parse the config. Their verl imports vllm at package
     import time, so a vllm API drift is a startup crash before any config is
     read. That is not hypothetical: `from vllm.lora.lora_model import
     LoRAModel` does not resolve under vllm 0.11.0, the version their own
     Dockerfile pins (patch_ease_repo.sh fixes it). Then the full override list
     goes through OmegaConf and `deep_post_init()`, which catches a mistyped
     key -- OmegaConf's structured merge rejects unknown fields, so a typo here
     is a crash 20 minutes into an allocation otherwise.

  2. Build the RLHFDataset over val.parquet with the staged checkpoint's
     tokenizer and processor. Catches a checkpoint that transformers 4.57
     cannot load, a broken image path, and a prompt_length column that does not
     match what the filter expects.

  3. Build EASE's own attention target over real rows and check it is a
     probability distribution confined to the vision span. This is the method
     touching our boxes: if `bbox`, `image_height` and `image_width` did not
     survive the parquet -> dataset -> actor path, or if the box were in the
     wrong coordinate frame, the target would silently fall back to uniform
     over vision tokens and the aux loss would train toward nothing.

Usage (normally via the launcher):
    python3 verify_ease_setup.py --ease-repo ease_repo -- data.train_files=... ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# A concentrated target is the point of the method; a flat one means the boxes
# never arrived. Row-level ratio of the peak vision token's mass to what a
# uniform distribution would give it.
MIN_PEAK_RATIO = 2.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ease-repo", type=Path, default=None, help="Defaults to $EASE_REPO or ./ease_repo.")
    p.add_argument("--config", default="examples/config.yaml", help="Relative to --ease-repo.")
    p.add_argument("--rows", type=int, default=8, help="How many rows to build a target for.")
    p.add_argument("overrides", nargs="*", help="key=value overrides, after a bare --.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ease_repo = args.ease_repo or Path(os.environ.get("EASE_REPO", "ease_repo"))
    ease_repo = ease_repo.resolve()
    if not (ease_repo / "verl").is_dir():
        sys.exit(f"not an ease_repo checkout: {ease_repo}")
    # Their config.yaml carries relative paths (./examples/...), and verl is
    # imported as a package from the repo root.
    os.chdir(ease_repo)
    sys.path.insert(0, str(ease_repo))

    from omegaconf import OmegaConf

    # ── 1. import + config ───────────────────────────────────────────────────
    from verl.trainer.config import PPOConfig

    config = OmegaConf.merge(OmegaConf.structured(PPOConfig()), OmegaConf.load(args.config))
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(args.overrides)))
    config = OmegaConf.to_object(config)
    config.deep_post_init()

    attention = config.worker.actor.attention
    print("[1] verl imports, config parses, deep_post_init() succeeds")
    print(f"    rollout_batch={config.data.rollout_batch_size}  "
          f"global_batch={config.worker.actor.global_batch_size}  "
          f"n={config.worker.rollout.n}  epochs={config.trainer.total_epochs}")
    print(f"    attention: anchor={attention.use_evidence_anchor} lambda={attention.lambda_attn} "
          f"mode={attention.bbox_weight_mode} sigma={attention.gaussian_sigma_scale} "
          f"alpha={attention.background_alpha} layer={attention.layer_index} "
          f"tau={attention.reward_threshold}")
    print(f"    reward: {config.worker.reward.reward_function}:{config.worker.reward.reward_function_name}")
    if attention.use_evidence_anchor and config.worker.actor.padding_free:
        sys.exit("FAIL: the aux attention loss requires worker.actor.padding_free=false")
    print(f"    padding_free={config.worker.actor.padding_free}")

    # ── 2. dataset ───────────────────────────────────────────────────────────
    # Populate the lazy qwen3_vl module: under 4.57 the auto-mapping cannot
    # resolve Qwen3VLForConditionalGeneration until this import has happened.
    from transformers.models.qwen3_vl import modeling_qwen3_vl  # noqa: F401

    from verl.utils.dataset import RLHFDataset
    from verl.utils.tokenizer import get_processor, get_tokenizer

    model_path = config.worker.actor.model.model_path
    tokenizer = get_tokenizer(model_path, use_fast=True)
    processor = get_processor(model_path, use_fast=True)
    dataset = RLHFDataset(
        data_path=config.data.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.data.prompt_key,
        answer_key=config.data.answer_key,
        image_key=config.data.image_key,
        video_key=config.data.video_key,
        image_dir=config.data.image_dir,
        max_prompt_length=config.data.max_prompt_length,
        format_prompt=config.data.format_prompt,
        min_pixels=config.data.min_pixels,
        max_pixels=config.data.max_pixels,
        filter_overlong_prompts=config.data.filter_overlong_prompts,
        filter_overlong_prompts_workers=4,
    )
    if len(dataset) == 0:
        sys.exit("FAIL: the val set is empty after filtering")
    print(f"\n[2] RLHFDataset over {config.data.val_files}: {len(dataset)} rows")

    # ── 3. the evidence target ───────────────────────────────────────────────
    from verl.workers.actor.evidence_mask import (
        get_attention_target_distribution,
        get_vision_token_range,
    )

    print(f"\n[3] evidence target, {min(args.rows, len(dataset))} rows")
    print(f"    {'vis tok':>8}{'box area':>10}{'vis mass':>10}{'peak':>9}{'peak/unif':>11}{'total':>8}")
    failures = []
    for index in range(min(args.rows, len(dataset))):
        example = dataset[index]
        for key in ("bbox", "image_height", "image_width", "ground_truth"):
            if key not in example:
                sys.exit(f"FAIL: row {index} lost `{key}` on the way through the dataset")

        input_ids = example["input_ids"]
        vis_start, vis_end = get_vision_token_range(input_ids)
        if vis_start < 0:
            sys.exit(f"FAIL: row {index} has no vision token span")

        grid_thw = example.get("multi_modal_inputs", {}).get("image_grid_thw")
        target = get_attention_target_distribution(
            input_ids=input_ids,
            bbox=example["bbox"],
            image_height=int(example["image_height"]),
            image_width=int(example["image_width"]),
            image_grid_thw=grid_thw[0] if grid_thw is not None else None,
            weight_mode=attention.bbox_weight_mode,
            gaussian_sigma_scale=attention.gaussian_sigma_scale,
            background_alpha=attention.background_alpha,
        )

        vision = target[vis_start:vis_end]
        n_vision = vis_end - vis_start
        uniform = 1.0 / n_vision
        box = list(example["bbox"][0])
        width, height = int(example["image_width"]), int(example["image_height"])
        area = ((box[2] - box[0]) * (box[3] - box[1])) / (width * height)
        peak, vision_mass, total = float(vision.max()), float(vision.sum()), float(target.sum())
        ratio = peak / uniform
        print(f"    {n_vision:>8}{area:>10.3f}{vision_mass:>10.4f}{peak:>9.4f}{ratio:>11.1f}{total:>8.4f}")

        if abs(total - 1.0) > 1e-4:
            failures.append(f"row {index}: target sums to {total}, not 1")
        if vision_mass < 0.999:
            failures.append(f"row {index}: {1 - vision_mass:.4f} of the mass is outside the vision span")
        if ratio < MIN_PEAK_RATIO:
            failures.append(
                f"row {index}: peak is only {ratio:.1f}x uniform -- the target is flat, which is what "
                f"get_attention_target_distribution returns when it cannot use the box"
            )

    if failures:
        print()
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    print("\nAll preflight checks passed. Nothing here exercises FSDP, vLLM or the judge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
