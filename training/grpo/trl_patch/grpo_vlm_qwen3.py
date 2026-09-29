# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GRPO entry point for the paper's arms. Derived from TRL's `examples/scripts/grpo_vlm.py`.

DO NOT LAUNCH THIS BY HAND. The arm is the config:

    bash training/grpo/run.sh self_saliency
    bash training/grpo/run.sh center_rect --dry-run     # print the command, launch nothing

`training/grpo/run.sh` reads `training/grpo/configs/<arm>.yaml` through
`training.grpo.config`, which is the ONLY producer of the flags below. Six arms, six YAML
files, one projection -- so the configs, the checks against the published checkpoints, and
what actually reaches this script are one implementation rather than three that agree.
The flags themselves are documented in `trl/scripts/utils.py`; the environment is
`docs/install.md`; the patch script that installs this file into a TRL checkout is
`env/patches/trl.sh`.

WHAT THIS FILE DOES, AND WHAT IT DELEGATES

It builds four things and hands them to `GRPOTrainerQwen3`: the corpus, the prompt, the
list of reward callables, and the saliency reward's configuration. It computes no saliency
and grounds nothing. R_sal is `trl.rewards.self_saliency`, which is itself a shell over
`selfsal` -- so the reward the policy is trained against and the head-selection screen of
Section 3.5 compute the same phi, by construction rather than by inspection.

REWARD ORDER IS LOAD-BEARING. `--reward_weights` is positional against `reward_funcs`,
so the order built below IS the meaning of Section 3.4's four alphas:

    R = a_format*R_format + a_sal*R_sal + a_direct*R_direct + a_llm*R_llm

`training.grpo.config.reward_weights` builds that vector in this order, and
`tests/test_configs_match_paper_checkpoints.py` checks it against the vector in each
published `training_args.bin`.
"""

import torch
from latex2sympy2_extended import NormalizationConfig
from PIL import Image
from math_verify import LatexExtractionConfig, parse, verify

from trl import (
    GRPOConfig,
    GRPOTrainerQwen3 as GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.rewards import think_format_reward, think_saliency_reward

# The method package. Everything reached here is shared with something outside training
# and must not be a second copy of it: the judge with the offline re-scoring tools, the
# corpus and its holdout with `precompute_question_boxes.py` and with the step count
# `training/grpo/config.py` checks, and the system prompt with the cold start that was
# supervised against it and with every probe that generates under it.
from selfsal.data.prompt import SYSTEM_PROMPT
from selfsal.data.saliency_r1_8k import DEFAULT_CORPUS, load_corpus, train_holdout_split
from selfsal.judge import openai_reward

# The long-side cap every image is resized to before the model or any detector sees it.
# Module-level because --overlap_question_boxes grounds the same images in a separate
# offline job (precompute_question_boxes.py), and Grounding-DINO shown a different
# resolution returns different boxes -- the value is written into that file and checked
# back when it is loaded, so the two copies cannot drift apart in silence.
#
# NOT `selfsal.data.prompt.MAX_IMAGE_SIDE`'s neighbour `prepare_image`. That helper
# resizes BILINEAR, which is what the archive's PROBES did (`resize(..., 2)`, whose
# trailing `# BICUBIC` comment is wrong). The training runs went through the line below,
# which is bicubic and always was. The two really did differ; wiring this onto the shared
# helper would change the pixels of every training image, hence every patch grid, every
# map and every phi. See docs/provenance.md.
MAX_IMAGE_SIDE = 512


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    ################
    # Model & Processor
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    ################
    # Dataset
    ################
    # Defaults to the paper's corpus, so existing launchers keep working unchanged.
    # --dataset_name points the same pipeline at any other VQA corpus exposing the
    # columns this script consumes: `problem`, `solution` and `image`, plus `bbox`
    # when --reward_variant saliency_r1.
    #
    # The loader and the carve live in `selfsal.data.saliency_r1_8k` rather than here,
    # because the offline `precompute_question_boxes.py` has to load the SAME corpus the
    # same way, and `training/grpo/config.py` turns the resulting row count into the step
    # count it checks against every published checkpoint. Three readers, one definition.
    dataset_name = script_args.dataset_name or DEFAULT_CORPUS
    dataset = load_corpus(dataset_name, split=script_args.dataset_train_split)
    dataset = train_holdout_split(dataset)

    def make_conversation(example):
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ]
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    # Cap the long side at 512px, preserving aspect ratio. saliency-r1-8k ships
    # pre-resized within this bound, but larger sources (full Visual-CoT, V*,
    # VisDrone) do not -- dropping oversized samples would silently discard most
    # of such a dataset, so downscale instead of filtering. Boxes in the `bbox`
    # column are normalized to [0, 1], so they survive the resize unchanged.
    #
    # Resize first, convert to RGB second. Converting a palette-mode image before
    # resizing is a different operation from resizing in palette space and converting
    # after -- interpolating palette INDICES is not interpolating colours -- and the two
    # give different pixels.

    def prepare_image(example):
        image = example["image"]
        width, height = image.size
        if max(width, height) > MAX_IMAGE_SIDE:
            scale = MAX_IMAGE_SIDE / max(width, height)
            image = image.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.BICUBIC,
            )
        if image.mode != "RGB":
            image = image.convert("RGB")
        example["image"] = image
        return example

    dataset = dataset.map(prepare_image)

    train_dataset = dataset["train"]

    ################
    # R_direct: the exact-match accuracy reward
    ################
    def accuracy_reward(completions, solution: list[str], **kwargs):
        """Reward function that checks if the completion matches the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If not parseable → compare as normalized text.

        Returns None where the answer could not be graded at all, which the trainer
        imputes to the group mean. Scoring an ungradeable completion 0 would conflate
        "got it wrong" with "could not be read".
        """
        import re as _re

        rewards = []
        contents = [completion[0]["content"] for completion in completions]
        for content, sol in zip(contents, solution):
            # Extract only the answer portion after </think>
            m = _re.search(r"</think>\s*(.*)", content, _re.DOTALL)
            answer_text = m.group(1).strip() if m else content.strip()

            try:
                gold_parsed = parse(sol, extraction_mode="first_match")
            except Exception:
                gold_parsed = []

            if len(gold_parsed) != 0:
                # Try parsing predicted answer too
                try:
                    answer_parsed = parse(
                        answer_text,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    boxed="all",
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode="first_match",
                    )
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception as e:
                    print(f"verify failed: {e}, answer: {answer_text}, gold: {sol}")
                    reward = None
            else:
                # fallback to text match
                reward = float(answer_text.lower() == sol.strip().lower())

            rewards.append(reward)

        return rewards

    ################
    # Reward function selection
    ################
    # The saliency term takes the SECOND slot, so --reward_weights lines up positionally
    # whichever of the two it is:
    #
    #   ours | saliency_r1   ->  [format, saliency, direct, judge]
    #   none                 ->  [format,           direct, judge]
    #
    # `none` is not the paper's No-Sal arm. That one ran `ours` with alpha_sal = 0, so the
    # term was installed, scored, logged and weighted zero -- the same gradient, but the
    # run on disk is the one the number came from. See training/grpo/configs/no_sal.yaml.

    # Both region overrides are read by R_sal and by nothing else. Under any other variant
    # the flag would be silently ignored and the run would be named after an arm that did
    # not happen -- a null result that looks like a finding. Refuse instead of ignoring.
    # (rect_frac and question_boxes conflicting with EACH OTHER is caught inside
    # configure(), which is the one place that knows they replace the same thing.)
    for _flag, _value in (("--overlap_rect_frac", script_args.overlap_rect_frac),
                          ("--overlap_question_boxes", script_args.overlap_question_boxes)):
        if _value is not None and script_args.reward_variant != "ours":
            raise SystemExit(
                f"{_flag} {_value} replaces the region R_sal scores a step against, but "
                f"--reward_variant {script_args.reward_variant} puts a different reward in "
                "that slot and never reads it. The run would be identical to one without "
                "the flag."
            )

    if script_args.reward_variant == "ours":
        from selfsal.data.question_boxes import load_question_boxes
        from trl.rewards.self_saliency import configure as configure_saliency
        from trl.rewards.self_saliency import self_saliency_reward

        configure_saliency(
            # Unset resolves to phi (Equation 1) inside selfsal.saliency.resolve_metric,
            # which is also where the historical mean_in / mean_in_v2 spellings are
            # accepted. Not defaulted here, so an unset metric acquires its value in one
            # place and the reward and the offline screen cannot disagree about it.
            metric=script_args.overlap_metric,
            box_threshold=script_args.box_threshold,
            max_box_area=script_args.max_box_area,
            max_union_area=script_args.max_union_area,
            dino_api_base=script_args.dino_api_base,
            natural_only=script_args.overlap_natural_only,
            # None keeps the per-step Grounding-DINO union; a fraction switches the region
            # to a centred rectangle and stops the detector being constructed at all.
            rect_frac=script_args.overlap_rect_frac,
            rect_placement=script_args.overlap_rect_placement,
            # The other way to stop constructing the detector: one union per dataset row,
            # grounded on its question before the run. configure() refuses the pair.
            question_boxes=script_args.overlap_question_boxes,
        )
        if script_args.overlap_question_boxes:
            # Load and validate NOW rather than on the first reward call: a threshold or
            # resolution mismatch is a configuration error, and finding it after the model
            # has loaded and the first generations are done costs an allocation.
            load_question_boxes(script_args.overlap_question_boxes,
                                box_threshold=script_args.box_threshold,
                                max_image_side=MAX_IMAGE_SIDE)
        reward_funcs = [think_format_reward, self_saliency_reward, accuracy_reward, openai_reward]
    elif script_args.reward_variant == "saliency_r1":
        # Appendix D.2's baseline reward, on the question-level box the corpus ships. The
        # trainer refuses a dataset with no `bbox` column rather than scoring against
        # nothing.
        reward_funcs = [think_format_reward, think_saliency_reward, accuracy_reward, openai_reward]
    elif script_args.reward_variant == "none":
        # No saliency term at all: format + accuracy + judge. The trainer skips the
        # attention re-forward entirely, so this is also the cheap variant.
        reward_funcs = [think_format_reward, accuracy_reward, openai_reward]
    else:
        raise SystemExit(f"unknown --reward_variant {script_args.reward_variant!r}")

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_args),
        reforward_saliency=script_args.reforward_saliency,
        reward_variant=script_args.reward_variant,
        overlap_layer=script_args.overlap_layer,
        overlap_heads=script_args.overlap_heads,
        token_reduction=script_args.token_reduction,
        overlap_natural_only=script_args.overlap_natural_only,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
