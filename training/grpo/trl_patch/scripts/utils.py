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

import argparse
import importlib
import inspect
import logging
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional, Union

import yaml
from transformers import HfArgumentParser
from transformers.hf_argparser import DataClass, DataClassType
from transformers.utils import is_rich_available


logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments common to all scripts.

    Args:
        dataset_name (`str`):
            Dataset name.
        dataset_config (`str` or `None`, *optional*, defaults to `None`):
            Dataset configuration name. Corresponds to the `name` argument of the [`~datasets.load_dataset`] function.
        dataset_train_split (`str`, *optional*, defaults to `"train"`):
            Dataset split to use for training.
        dataset_test_split (`str`, *optional*, defaults to `"test"`):
            Dataset split to use for evaluation.
        dataset_streaming (`bool`, *optional*, defaults to `False`):
            Whether to stream the dataset. If True, the dataset will be loaded in streaming mode.
        gradient_checkpointing_use_reentrant (`bool`, *optional*, defaults to `False`):
            Whether to apply `use_reentrant` for gradient checkpointing.
        ignore_bias_buffers (`bool`, *optional*, defaults to `False`):
            Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid scalar
            type, inplace operation. See
            https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992.
    """

    dataset_name: Optional[str] = field(default=None, metadata={"help": "Dataset name."})
    dataset_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset configuration name. Corresponds to the `name` argument of the `datasets.load_dataset` "
            "function."
        },
    )
    dataset_train_split: str = field(default="train", metadata={"help": "Dataset split to use for training."})
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})
    val_sets_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory holding the held-out validation sets written by "
            "`build_grpo_sets.py --build-val` (val_natural/ and val_nonnatural/). Whichever "
            "of the two exist are scored separately, so natural and non-natural imagery get "
            "their own curves. Their images are disjoint from set_a and set_b, so this does "
            "not touch the training corpus. Scored on answer accuracy only, by one greedy "
            "completion per prompt through vLLM -- not by the Trainer's evaluation loop, "
            "which would re-run the whole reward pipeline."
        },
    )
    val_eval_steps: int = field(
        default=100,
        metadata={
            "help": "How often (in optimizer steps) to score the validation sets. A step-0 "
            "baseline is always taken. 0 disables periodic scoring, leaving only step 0."
        },
    )
    dataset_streaming: bool = field(
        default=False,
        metadata={"help": "Whether to stream the dataset. If True, the dataset will be loaded in streaming mode."},
    )
    gradient_checkpointing_use_reentrant: bool = field(
        default=False,
        metadata={"help": "Whether to apply `use_reentrant` for gradient checkpointing."},
    )
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": "Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid "
            "scalar type, inplace operation. See "
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992."
        },
    )
    reforward_saliency: bool = field(
        default=True,
        metadata={
            "help": "Whether to compute saliency via a separate re-forward pass instead of capturing attention "
            "weights during generate(). Required when max_completion_length > 1024 to avoid OOM."
        },
    )
    # ---- attention-overlap reward (reward_variant="ours"; see grpo-reward-port-plan) ----
    saliency_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "WHICH SALIENCY MAP the per-observe-step reward scores. This is the "
            "flag to use; --reward_variant is the older spelling and is kept only so "
            "existing command lines keep working. "
            "'attention' = raw observe->patch attention at --overlap_layer, mean of "
            "--overlap_heads (docs/saliency-maps.md map 1); 'grad' = the PIXEL GRADIENT "
            "of the step's own tokens (map 5b); 'glimpse' = GLIMPSE's gradient-weighted "
            "attention (map 6, ~55x the cost of grad -- see trl/glimpse_maps.py). "
            "All three produce the same per-step map contract and are scored by "
            "--overlap_metric, so the map and the metric are now independent choices. "
            "Maps to --reward_variant ours|grad|glimpse; set both and this one wins. "
            "Leave unset (with --reward_variant saliency_r1|none) for the two variants "
            "that are not per-observe-step rewards at all.",
            "choices": ["attention", "grad", "glimpse"],
        },
    )
    reward_variant: str = field(
        default="saliency_r1",
        metadata={
            "help": "Which reward + attention-extraction mode to use. 'saliency_r1' = the paper's "
            "whole-completion rollout saliency reward. 'ours' = raw per-head observe->patch "
            "attention overlap reward (layer 22, 2-head mean, per observe step, DINO-grounded). "
            "'none' = accuracy + judge + format only (no saliency/overlap reward; skips the "
            "attention re-forward pass entirely). 'grad' = the roll-null gradient reward: "
            "same per-observe-step, DINO-grounded shape as 'ours', but the map is the "
            "PIXEL GRADIENT of the step's own tokens and the score is "
            "log(||g_U|| / ||g_rolled||) -- see trl/rewards/grad_rewards.py. "
            "'glimpse' = the GLIMPSE grounding reward: same per-observe-step, "
            "DINO-grounded shape, but the map is GLIMPSE's gradient-weighted attention "
            "(docs/saliency-maps.md map 6) and the score is --overlap_metric. It costs "
            "55-59x the 'grad' variant per case (100-145 s added to a ~40 s optimizer "
            "step) and its screened correlation with correctness is null to slightly "
            "negative -- read trl/rewards/glimpse_rewards.py before using it.",
            "choices": ["saliency_r1", "ours", "none", "grad", "glimpse"],
        },
    )
    # ---- roll-null gradient reward (reward_variant="grad") ----
    grad_target: str = field(
        default="clogit",
        metadata={
            "help": "reward_variant='grad': the scalar differentiated per generated token. "
            "'clogit' (default) = the raw logit minus the vocabulary mean: it does not "
            "saturate as the model grows confident (d log P/dz -> 0 as p -> 1, which would "
            "make the reward pay for uncertain steps) and it drops the common-mode channel "
            "shared by every vocabulary item, which is otherwise the SAME map for every "
            "step and so a one-shot lift for all of them. 'logit' keeps that channel; "
            "'logprob' saturates. Both are for probes.",
            "choices": ["clogit", "logit", "logprob"],
        },
    )
    grad_null_offsets: int = field(
        default=16,
        metadata={
            "help": "reward_variant='grad': how many translated copies of the box union "
            "form the null. Their SQUARED norms are pooled before the log, so one control "
            "landing on a dead region cannot dominate. Pure numpy on a ~16x16 map: free."
        },
    )
    grad_logratio_clip: float = field(
        default=1.0,
        metadata={
            "help": "reward_variant='grad': clip |log(||g_U||/||g_null||)| to this. A ratio "
            "has a heavy tail, and with scale_rewards=True one outlier completion takes "
            "most of its group's normalised advantage. Set it from the measured spread "
            "(overlap_metric_spread.py) rather than trusting the default."
        },
    )
    grad_inframe_rolls: bool = field(
        default=True,
        metadata={
            "help": "reward_variant='grad': draw the control placements so the translated "
            "union stays inside the grid, instead of wrapping toroidally across the image "
            "border. Falls back to toroidal (counted in grad/toroidal_frac) when a "
            "near-full-frame union leaves too few in-frame positions."
        },
    )
    grad_dedupe_steps: bool = field(
        default=True,
        metadata={
            "help": "reward_variant='grad': drop repeated observe-step texts before the "
            "mean over steps. The score is a mean, so re-quoting one easily-grounded "
            "sentence pulls it up and dilutes the hard perception steps -- measured going "
            "0.00 -> 0.19 in the wov0.4 run. grad/dup_frac is logged either way."
        },
    )
    grad_natural_only: bool = field(
        default=False,
        metadata={
            "help": "reward_variant='grad': score only rows whose 'natural' column is True. "
            "Same rationale as --overlap_natural_only: Grounding-DINO is a photograph "
            "detector, so on charts/documents the box union -- and the whole score -- is "
            "noise."
        },
    )
    grad_seed: int = field(
        default=0,
        metadata={"help": "reward_variant='grad': seed for the control placements."},
    )
    # ---- GLIMPSE grounding reward (reward_variant="glimpse") ----
    glimpse_target: str = field(
        default="clogit",
        metadata={
            "help": "reward_variant='glimpse': the scalar differentiated per generated "
            "token to get dz/dA. Means exactly what --grad_target means, and for the same "
            "reasons: 'clogit' does not saturate as the model grows confident and drops "
            "the common-mode channel shared by every vocabulary item.",
            "choices": ["clogit", "logit", "logprob"],
        },
    )
    glimpse_layer_frac: float = field(
        default=1.0,
        metadata={
            "help": "reward_variant='glimpse': fraction of the decoder stack propagated, "
            "taken off the TOP. THE FIRST COST DIAL: 0.6 measured 1.64x cheaper (34-36x "
            "the gradient reward instead of 55-59x) and the paper's own ablation loses "
            "nothing there -- but it is a METHOD change, not a memory one, and the map "
            "that was screened is 1.0."
        },
    )
    glimpse_token_cap: int = field(
        default=0,
        metadata={
            "help": "reward_variant='glimpse': score at most this many target tokens per "
            "observe step, drawn uniformly at random without replacement (never the first "
            "k -- these maps carry a reading-order prior). THE SECOND COST DIAL, and the "
            "strongest one: cost is exactly linear in it, so a cap of 6 against a median "
            "~20 tokens per step is ~3.5x. Eq 18 renormalises beta inside the step, so a "
            "random subset estimates the same weighted mean -- but what the cap does to "
            "the SCORE has not been measured at scale, which is why 0 (every token) is "
            "the default."
        },
    )
    glimpse_temp: float = field(
        default=0.5,
        metadata={
            "help": "reward_variant='glimpse': lambda in eq 6, the head-fusion softmax "
            "temperature. The paper's value."
        },
    )
    glimpse_depth_temp: float = field(
        default=0.2,
        metadata={
            "help": "reward_variant='glimpse': lambda_d in eq 9, the exponential depth "
            "prior. 0.2 is the paper's text; it was tuned on a 64-layer backbone where it "
            "spans 7.8% of the depth, and 0.36 is what matches that SHAPE on this "
            "36-layer model. The ablation calls this the single most important component."
        },
    )
    glimpse_token_weight: str = field(
        default="full",
        metadata={
            "help": "reward_variant='glimpse': eq 18's beta_t. 'full' = p_t x prompt "
            "alignment (the paper); the other three reproduce its token-saliency "
            "ablation. Eq 17 crosses the modalities on purpose -- a token earns its say "
            "in where the model looked by being about the QUESTION, since weighting it by "
            "its own visual alignment would be circular.",
            "choices": ["full", "confidence", "prompt", "uniform"],
        },
    )
    glimpse_dedupe_steps: bool = field(
        default=True,
        metadata={
            "help": "reward_variant='glimpse': drop repeated observe-step texts before "
            "the mean over steps. Same hack and same rationale as --grad_dedupe_steps; "
            "glimpse/dup_frac is logged either way."
        },
    )
    glimpse_natural_only: bool = field(
        default=False,
        metadata={
            "help": "reward_variant='glimpse': score only rows whose 'natural' column is "
            "True. Same rationale as --overlap_natural_only: Grounding-DINO is a "
            "photograph detector, so on charts/documents the box union -- and the whole "
            "score -- is noise."
        },
    )
    glimpse_seed: int = field(
        default=0,
        metadata={
            "help": "reward_variant='glimpse': seed for the --glimpse_token_cap draw. "
            "Irrelevant when the cap is 0."
        },
    )
    # ---- the roll-null, when it is used as a METRIC (--overlap_metric logratio /
    # metric). reward_variant='grad' keeps its own --grad_* copies of
    # these, because there the roll-null IS the reward rather than one metric of four,
    # and separating them keeps every existing grad run reproducible byte for byte.
    rollnull_offsets: int = field(
        default=16,
        metadata={
            "help": "--overlap_metric logratio: how many translated copies of the box union form "
            "the null. Their SQUARED masses are pooled BEFORE the log, so one control "
            "landing on a dead region cannot dominate. Pure numpy on a ~16x16 map: free."
        },
    )
    rollnull_clip: float = field(
        default=1.0,
        metadata={
            "help": "--overlap_metric logratio: clip |log(N(U)/N_0)| to this. A ratio has a heavy "
            "tail, and with scale_rewards=True one outlier completion takes most of its "
            "group's normalised advantage. 1.0 == a ratio of e."
        },
    )
    rollnull_inframe: bool = field(
        default=True,
        metadata={
            "help": "--overlap_metric logratio: draw the control placements so the translated "
            "union stays INSIDE the grid rather than wrapping toroidally across the image "
            "border. Falls back to toroidal when a near-full-frame union leaves too few "
            "in-frame positions -- watch <variant>/toroidal_frac, because that fallback "
            "changes what the null means."
        },
    )
    rollnull_seed: int = field(
        default=0,
        metadata={"help": "--overlap_metric logratio: seed for the control placements."},
    )
    token_reduction: str = field(
        default="mean",
        metadata={
            "help": "reward_variant='ours': reduce per-token saliency maps within an observe step "
            "(mean|max). Sweep dimension — appears in the model/wandb name as trmean/trmax.",
            "choices": ["mean", "max", "min"],
        },
    )
    overlap_layer: int = field(
        default=22,
        metadata={"help": "reward_variant='ours': transformer layer to read raw attention from."},
    )
    overlap_heads: str = field(
        default="28,31",
        metadata={
            "help": "reward_variant='ours': comma-separated head indices at overlap_layer to mean "
            "together (default the fixed 2-head (22,28)+(22,31) option)."
        },
    )
    overlap_metric: Optional[str] = field(
        default=None,
        metadata={
            "help": "How to score a step's map against its DINO box union, for EVERY "
            "--saliency_method (attention, grad and glimpse alike). UNSET means the "
            "historical default OF THAT MAP -- mean_in for attention, logratio for grad, "
            "mean_in_v2 for glimpse -- so an existing command line keeps its behaviour "
            "now that one flag serves all three. "
            "'mean_in' (default, the incumbent) = mean of the MAX-normalized map inside the "
            "box; it divides by the map's own peak, so a map that merely FLATTENS scores "
            "higher (measured: 32x more movement under flattening than under real "
            "grounding). 'mean_in_v2' = the same mean over the box divided by the mean over "
            "the whole map instead of by its peak: chance is 1.0, rescale-invariant, and "
            "unlike auroc it still sees magnitudes. Measured on the cold-start policy it runs "
            "median 0.74 / p99 1.36 with 12x mean_in's per-sample spread, so its w_overlap is "
            "0.033 (= mean_in's wov0.4 pressure), applied by the launchers. 'auroc' = "
            "P(in-box patch outranks out-box patch), which depends "
            "only on patch order and is therefore exactly invariant to that flattening, and "
            "predicts correctness more stably (mean |r| 0.238 vs 0.181, sd 0.028 vs 0.089). "
            "'logratio' = the ROLL-NULL: log of the map's mass inside the union over its "
            "mass in the SAME union translated to random in-frame offsets. Chance is "
            "exactly 0 and the control has the union's own shape and area by "
            "construction, so this is the only one of the four that closes the box-size "
            "confound rather than merely bounding it -- at the cost of being random "
            "(it draws control placements) and of squaring the mass, which weights peaks "
            "more than a plain sum. Its knobs are --rollnull_*. "
            "Sweep dimension — appears in the model/wandb name.",
            "choices": ["mean_in", "mean_in_v2", "auroc", "logratio"],
        },
    )
    mass_floor_tau: Optional[float] = field(
        default=None,
        metadata={
            "help": "reward_variant='ours': if set, multiply each step's score by "
            "min(1, image_mass/tau), where image_mass is the fraction of the attention row "
            "spent on image tokens. Closes the one hole a rank-based metric cannot see — a "
            "model withdrawing attention from the image while keeping a good ranking. Also "
            "raises the correctness correlation (0.227 -> 0.238) because image_mass is "
            "itself predictive. Recommended 0.0022 = the 10th percentile of the reference "
            "model's image_mass. Keep near p10: much above p25 it stops being a floor and "
            "'raise image attention uniformly' becomes its own exploitable direction. "
            "Sweep dimension — appears in the model/wandb name.",
        },
    )
    overlap_rect_frac: Optional[float] = field(
        default=None,
        metadata={
            "help": "reward_variant='ours': score each step's map against a CENTRED "
            "RECTANGLE covering this fraction of the patch grid instead of against the "
            "step's Grounding-DINO box union. NO DETECTOR IS RUN — no DINO GPU, no DINO "
            "server, and the launchers drop the sidecar. Everything else is unchanged "
            "(same --overlap_metric, same format gate, same reward slot and weights), so "
            "the run differs from a DINO reference in the mask and nothing else. "
            "0.565 is the fraction to use for that comparison: it is the mean union "
            "coverage DINO produces on these runs, so the rectangle gives away nothing on "
            "mask SIZE and differs in PLACEMENT alone. Measured motivation, from "
            "centre_box_probe.py: the rectangle is NOT DINO's mask (area-matched "
            "closeness 0.230, against a 0.235 different-image floor), yet the "
            "per-completion reward built on it reproduces the real per-step-DINO reward's "
            "ranking of a group at rho 0.651 vs 0.621 for DINO once per chain — so this "
            "arm asks whether the detector was buying the gradient at all. Note every "
            "step becomes scoreable (nothing is ungroundable), so the scored set is larger "
            "than its DINO reference's, and w_overlap should be re-measured rather than "
            "carried over. Sweep dimension — appears in the model/wandb name.",
        },
    )
    overlap_rect_placement: str = field(
        default="centre",
        metadata={
            "help": "reward_variant='ours', read only with --overlap_rect_frac: WHERE the "
            "rectangle sits. 'centre' (default) is the incumbent and is byte-identical to "
            "what --overlap_rect_frac did before this flag existed. The other two restore a "
            "per-COMPLETION mask without a detector, which the centred rectangle cannot "
            "have: a mask that is the same for all 8 rollouts cancels out of the advantage "
            "except through the map, and the surviving contrast is 0.90-0.97 correlated "
            "with the box-blind `flatness` statistic against 0.57-0.78 for the per-step "
            "DINO union (11 checkpoints, mask_variance_probe.py). 'interior_centre' = the "
            "same construction on the grid's INTERIOR (everything but the one-patch "
            "border), sized to --overlap_rect_frac of the interior's area; the matched "
            "control. 'interior_hash' = those dims placed at one of the strictly-interior "
            "positions, chosen by blake2b(--overlap_rect_seed | completion text), so it is "
            "fixed per completion and stable across ranks and restarts. The interior "
            "restriction is load-bearing and not a detail: the border is 30% of a 10x16 "
            "grid but carries 48-52% of the attention mass and 76-85% of map peaks, and "
            "mean_in divides by that peak. A rectangle merely displaced IN FRAME takes 21% "
            "of the border and tracks a completion's border mass LESS strongly than "
            "`flatness` does in 10 of 11 checkpoints -- it dilutes the mechanism it was "
            "meant to preserve. Confined to the interior, border coverage is exactly 0 for "
            "every completion and the reward tracks border mass more strongly than "
            "`flatness` in 8 of 11 and than the DINO union in 11 of 11. Against its own "
            "centred control the hashed draw raises within-group sd in 11 of 11 (median "
            "ratio to the per-step reward 0.85 vs 0.69) and drops the flatness correlation "
            "in 11 of 11 (0.709 vs 0.891, where the per-step union is 0.723), at the cost "
            "of a weaker border term (median -0.605 vs -0.714). Note the fraction is read "
            "against the INTERIOR under both interior modes, so the mask is 0.412 of a "
            "10x16 grid rather than 0.600 -- re-measure w_overlap. Which reference to "
            "match depends on the comparator: against a CENTRED --overlap_rect_frac run, "
            "0.32 for interior_hash and 0.45 for interior_centre; against mean_in w0.4, "
            "0.37 and 0.45. The first is the steadier number (0.28-0.36 over 11 "
            "checkpoints against 0.37-0.58) because it is the same statistic on two "
            "similar masks. Sweep dimension - appears in the model/wandb name."
        },
    )
    overlap_rect_seed: int = field(
        default=0,
        metadata={
            "help": "reward_variant='ours': seed mixed into the "
            "--overlap_rect_placement interior_hash draw. A second run at another seed is a "
            "replicate over a different lottery, which is how a result is told from a draw."
        },
    )
    overlap_chain_boxes: Optional[str] = field(
        default=None,
        metadata={
            "help": "reward_variant='ours': call Grounding-DINO ONCE PER COMPLETION, on one "
            "observe step's sentence ('first'|'last'), and score every step of that "
            "completion against the union it returns - instead of once per step on each "
            "step's own sentence. The middle rung between the incumbent and "
            "--overlap_question_boxes (once per row), and unlike those two it still needs "
            "the detector: it is cheaper, not detector-free. Detector calls fall from the "
            "completion's step count (2.1 on the trained 8k policy, 3.7 at the cold start) "
            "to one. It is also the only mask source measured that RAISES the reward's "
            "within-group spread rather than lowering it - median ratio to the per-step "
            "reward 1.45 for 'last' and 1.14 for 'first', where every fixed-mask arm sits "
            "at 0.67-0.75 - and its correlation with the box-blind `flatness` statistic is "
            "0.50/0.56 against those arms' 0.89-0.93 (11 checkpoints, "
            "mask_variance_probe.py). Use 'last': first steps are the most stereotyped in a "
            "chain - two chains' opening steps are more alike (median closeness 0.842) "
            "than two steps of ONE chain (0.614), where last steps sit at 0.699 and it "
            "holds pairwise in 9 of 11 checkpoints - so grounding on the first gives a "
            "prompt's 8 rollouts the most nearly identical masks available, which is the "
            "opposite of the point. Changes WHICH completions are scored: if the chosen "
            "step grounds nothing the whole completion is unscored (masked, not zeroed), "
            "with no fallback, because a fallback costs a second call; watch "
            "mask/chain_ungrounded_frac. --max_union_area likewise becomes per completion. "
            "w_overlap needs re-measuring: the cold-start match at w_ref 0.4 is 0.32 for "
            "'last'. Sweep dimension - appears in the model/wandb name."
        },
    )
    overlap_merge_boxes: bool = field(
        default=False,
        metadata={
            "help": "reward_variant='ours': ground every observe step exactly as the "
            "incumbent does - same calls, same sentences, same count - then MERGE the "
            "completion's box lists and score every one of its steps against that single "
            "union. The question stops being 'did this step look where its own sentence "
            "points' and becomes 'did this step look anywhere the chain ever mentions'. "
            "Not a rung of the --overlap_chain_boxes / --overlap_question_boxes / "
            "--overlap_rect_frac ladder, which trades detector calls for a coarser mask: "
            "this buys nothing on cost and changes only the target, so it is the arm that "
            "isolates granularity from budget. It keeps the property those fixed-mask arms "
            "lack - the mask is constant inside a completion but still varies between a "
            "prompt's 8 rollouts, and a mask constant inside a group cancels out of the "
            "GRPO advantage. It also widens the scored set: a step that grounds nothing is "
            "scored against its neighbours' boxes instead of being skipped, and only a "
            "completion where NOTHING grounded is lost. Measured on the val_natural probe, "
            "11 checkpoints (mask_variance_probe.py, scheme `chain_union`): w_overlap "
            "TRANSFERS - within-group sd ratio 1.02 and matched weight 0.39 against the "
            "incumbent's 0.4, the only mask source for which that is true (chain_last wants "
            "0.32, question_boxes 0.55, a centred rectangle 0.60); r with the box-blind "
            "`flatness` statistic 0.687, BELOW the per-step union's own 0.723 where every "
            "fixed-mask arm sits at 0.89-0.93; and r 0.827 with the per-step reward's own "
            "within-group ranking, the closest of any arm, which makes it both the least "
            "confounded comparison and the least likely to land anywhere different. THE "
            "RISK is saturation: merging only grows the mask, the per-step union already "
            "covers a median 0.562 of the grid, and two steps of one chain are the LEAST "
            "alike pairing measured (closeness 0.614), so the area really does add up - "
            "median merged coverage 0.754, and at exactly 100% the union is refused, which "
            "here costs the WHOLE completion. That refusal rate is 0.0-67.9% across the 11 "
            "checkpoints and is a deterministic function of chain length (Spearman +0.991 "
            "with observe steps per completion): 6.5% at the cold start, 67.9% at 14 steps. "
            "Since this reward is known to lengthen chains, the drift switches the arm off "
            "rather than diluting it, silently, because masked is neutral. Watch "
            "mask/merged_cover (recorded before the refusal, so the dropped completions are "
            "in it) and mask/merged_unscored_frac, in that order, before the reward. "
            "--max_union_area applies per COMPLETION here and bounds the damage without "
            "removing it - at the cold start a cap of 0.9 drops 26.4% of completions up "
            "front, 0.8 drops 45.9%. See docs/merged-boxes.md. Sweep dimension - appears in "
            "the model/wandb name."
        },
    )
    box_threshold: float = field(
        default=0.10,
        metadata={"help": "reward_variant='ours': Grounding-DINO confidence threshold for per-step boxes. "
                          "Ignored under --overlap_rect_frac (no boxes are requested); under "
                          "--overlap_chain_boxes it still applies, to the one call per completion."},
    )
    max_box_area: float = field(
        default=0.5,
        metadata={
            "help": "reward_variant='ours': drop INDIVIDUAL DINO boxes whose area fraction exceeds this "
            "cap. Set to 0 to disable the per-box cap entirely (keep every box above --box_threshold). "
            "This bounds no. of pixels per box, not the union — see --max_union_area."
        },
    )
    max_union_area: Optional[float] = field(
        default=None,
        metadata={
            "help": "reward_variant='ours': skip (do not score) any observe step whose rasterised box "
            "UNION covers more than this fraction of the image, e.g. 0.4. The step is masked exactly "
            "like an ungroundable one — SKIPPED, not scored 0 — so it drops out of the per-completion "
            "mean. None/0 (default) disables the cap, leaving only the existing 100%-coverage "
            "degenerate guard. Needed because --max_box_area is per-box: N disjoint boxes each under "
            "the cap can still cover the whole image. Sweep dimension — appears in the model/wandb name."
        },
    )
    dino_api_base: Optional[str] = field(
        default=None,
        metadata={
            "help": "reward_variant='ours': base URL of a served batched Grounding-DINO endpoint. "
            "If unset, DINO runs locally on each training process's device."
        },
    )
    overlap_question_boxes: Optional[str] = field(
        default=None,
        metadata={
            "help": "reward_variant='ours': path to a precompute_question_boxes.py file. Ground ONCE "
            "per dataset row, on the row's question, before the run, and score every observe step of "
            "that row against that one box union — instead of calling Grounding-DINO once per step on "
            "the step's own sentence. No detector is loaded at training time at all. Justified by "
            "dino_text_sensitivity.py: the question recovers a step's real mask at IoU 0.649 / "
            "closeness 0.785, where a different real step of the SAME chain gets 0.635 / 0.721, at the "
            "same mask size — so the per-step call is not buying a per-step mask. Changes WHICH steps "
            "are scored: a row now grounds for all of its steps or for none, where per-step grounding "
            "skips only the steps whose own sentence grounds nothing. --box_threshold and the "
            "trainer's 512px image cap are baked into the file and a mismatch is refused. Sweep "
            "dimension — appears in the model/wandb name."
        },
    )
    placebo: Optional[str] = field(
        default=None,
        metadata={
            "help": "reward_variant='ours': REPLACE the attention-overlap reward with a "
            "control that has its within-group spread but not its grounding, to find out "
            "whether the reward's DIRECTION matters at all. 'roll' = the configured "
            "--overlap_metric on the same map, scored against the step's own box union "
            "MOVED to a deterministic wrong place (same area, same shape); 'random' = a "
            "stable hash of the completion text -> U(0,1), pure variance with no "
            "direction; 'length' = -n_completion_tokens/1000, i.e. the brevity reward the "
            "overlap term is suspected of being in disguise (within a group it correlates "
            "-0.04 to -0.11 with completion length and ~0.00 with accuracy). Each takes "
            "the overlap reward's slot in reward_funcs, so --reward_weights is unchanged, "
            "and each must be given the weight that matches mean_in w0.4's WITHIN-GROUP "
            "sd -- the launcher resolves it. Every placebo returns unscored on exactly "
            "the completions the real reward would leave unscored (it runs the same "
            "segmentation, the same Grounding-DINO call and the same metric, and uses the "
            "real score only as a gate), so the comparison has one variable and not two. "
            "Not available with --overlap_metric logratio. See "
            "docs/next-reward-experiments.md and trl/rewards/placebo_rewards.py.",
            "choices": ["roll", "random", "length"],
        },
    )
    maskfree: Optional[str] = field(
        default=None,
        metadata={
            "help": "reward_variant='ours': REPLACE the attention-overlap reward with a "
            "MASK-FREE one that needs no boxes and never calls Grounding-DINO. 'flatness' "
            "= mean(m)/max(m) over the whole patch grid, i.e. mean_in with the union "
            "replaced by the image -- scale-invariant, so it scores only the map's shape; "
            "'mass' = log(sum(m)) + anchor, the probability mass the step's think-tokens "
            "put on image patches. These test whether mean_in's benefit was ever about "
            "grounding: on the val_natural probe, within-group, mean_all/max predicts the "
            "real mean_in reward at r=+0.69 while a genuinely relocated union predicts it "
            "at r=-0.015. Takes the overlap reward's slot in reward_funcs, so "
            "--reward_weights is unchanged, and needs the weight that matches mean_in "
            "w0.4's WITHIN-GROUP sd -- the launcher resolves it (flatness 0.45, mass "
            "0.006). Segmentation and the layer-22 re-forward are kept, so the maps are "
            "the ones think_overlap_reward would have scored. See "
            "docs/next-reward-experiments.md and trl/rewards/maskfree_rewards.py.",
            "choices": ["flatness", "mass"],
        },
    )
    maskfree_parity: bool = field(
        default=False,
        metadata={
            "help": "--maskfree: re-impose the overlap reward's scored/unscored set by "
            "running the real Grounding-DINO pipeline and using its verdict only as a "
            "boolean. Off by default: measured on val_natural the two sets are IDENTICAL "
            "(231/240 completions either way), and turning it on costs the full grounding "
            "call -- 16.6 s of a 40.5 s optimizer step -- which is the entire saving "
            "--maskfree exists for. Turn it on only to verify that equality on a new "
            "corpus."
        },
    )
    maskfree_mass_anchor: float = field(
        default=18.0,
        metadata={
            "help": "--maskfree mass: constant added to log(sum(m)). Invisible to GRPO "
            "(the advantage subtracts the group mean); it exists so a scored completion "
            "never reads negative, because a `.nansum` fold reads an UNSCORED reward as 0 "
            "and that would make 'produce no gradeable observe step' the best move on the "
            "auxiliary dimension. 18.0 against a measured minimum image mass of 1.4e-04 "
            "(13,648 observe steps, val_natural), which needs 8.87 -- four orders of "
            "magnitude of margin. The guarantee is empirical: no finite anchor makes a "
            "log positive for an arbitrarily small mass, so on a much dimmer corpus read "
            "maskfree/mass in the logs and raise this."
        },
    )
    mismatch_bank: Optional[str] = field(
        default=None,
        metadata={
            "help": "reward_variant='ours': REPLACE the attention-overlap reward with the "
            "MISMATCHED-BOX control -- the same metric on the same map, but scored against "
            "real Grounding-DINO boxes computed for a DIFFERENT question about a DIFFERENT "
            "picture. Tests the assumption every other variant is downstream of: that "
            "running DINO on THAT SENTENCE matters. Takes the path of a bank written by "
            "build_mismatch_bank.py; no DINO is loaded at training time at all, which is "
            "16.6 s off a 40.5 s optimizer step. The donor row is fixed per training row "
            "and shared by all of a prompt's rollouts -- drawing one per completion would "
            "put 0.0117 of donor noise inside a group whose whole real reward spans 0.0115, "
            "i.e. re-run --placebo random. Within-group spread is 0.82x the reference's, so "
            "the launcher gives it ~1.22x mean_in's weight. Takes the overlap reward's slot "
            "in reward_funcs, so --reward_weights is unchanged. See "
            "docs/mismatch-boxes.md and trl/rewards/mismatch_rewards.py."
        },
    )
    mismatch_seed: int = field(
        default=0,
        metadata={
            "help": "--mismatch_bank: chooses which donor row each training row is paired "
            "with, and which of that donor's chains is used at each length. Two runs "
            "differing only in it are independent replicates of the control over a "
            "different random pairing, which is the way to tell a result from a pairing."
        },
    )
    length_guard_ref: Optional[float] = field(
        default=None,
        metadata={
            "help": "Install the LENGTH GUARD, an ADDITIONAL reward term (it does not "
            "replace anything) that is exactly 0 while a completion's token count stays "
            "near the base policy's and goes negative outside a band around it. The value "
            "is the REFERENCE LENGTH in tokens: the cold-start model's mean completion "
            "length on the corpus you are training on, read off completions/mean_length "
            "at step 0 of a matched run. There is no safe default -- 217 is set_c's and "
            "would be wrong on a corpus with shorter chains. Its strength is its entry in "
            "--reward_weights, appended automatically by grpo_vlm_qwen3.py so existing "
            "--reward_weights command lines are unchanged. Works with every "
            "--saliency_method and with --reward_variant none, costs nothing (it reads "
            "completion_ids and no map), and scores every completion rather than "
            "inheriting any reward's unscored set -- masking it would make 'produce no "
            "groundable observe step' an escape hatch from the leash. See "
            "trl/rewards/length_guard_rewards.py."
        },
    )
    length_guard_weight: float = field(
        default=0.20,
        metadata={
            "help": "--length-guard: the term's entry in --reward_weights, i.e. its "
            "strength k. Appended by grpo_vlm_qwen3.py rather than typed into "
            "--reward_weights, so every existing command line keeps working. Measured on "
            "the cold-start policy (val_natural, 240 completions) 0.20 gives 0.0036 of "
            "effective pressure -- the same order as mean_in w0.4's 0.0028, but "
            "concentrated on the 1.3% of completions outside the band rather than spread "
            "over all of them as a brevity gradient -- while costing a collapsed "
            "completion 0.018 reward units at 49 tokens, 0.12 at 31 and 0.45 at 13. "
            "Re-measure on your corpus with overlap_metric_spread.py. If baseline noise "
            "and collapse cost conflict there, WIDEN the band rather than lowering this."
        },
    )
    length_guard_band_lo: float = field(
        default=0.30,
        metadata={
            "help": "--length-guard: lower edge of the free window, as a MULTIPLE of the "
            "reference length (0.30 = a third of the base length is still free). Measured, "
            "not guessed: it clears the healthy 8k run's own short tail (p1 = 0.37x) while "
            "sitting above every length collapse on record -- auroc 0.22x, --maskfree mass "
            "0.14x, --placebo length 0.06x. This is the side that does the work, because "
            "collapse has no other guard: a 13-token completion can be perfectly formatted "
            "and scored correct."
        },
    )
    length_guard_band_hi: float = field(
        default=3.0,
        metadata={
            "help": "--length-guard: upper edge of the free window, as a MULTIPLE of the "
            "reference length. 3.0 sits between the cold-start policy's p99 (3.5x) and its "
            "bulk, and below max_completion_length (4.6x at l_ref=221). The long side is "
            "necessarily weak: truncation already costs a completion BOTH accuracy and "
            "format (~2.0 reward units), so all this adds is a soft ramp into an existing "
            "cliff. Do not tighten it to chase a set_c-style inflation -- that run's mean "
            "sat at 1.15x and its tail at 1.81x, and a band tight enough to reach them "
            "penalises 22% of healthy completions."
        },
    )
    length_guard_knee: float = field(
        default=1.0,
        metadata={
            "help": "--length-guard: excess (in LOG units) at which the penalty stops "
            "growing quadratically and grows linearly instead; value and slope are both "
            "continuous there. Bounds how far one runaway completion can inflate its "
            "group's reward std, which under scale_rewards=True would shrink every OTHER "
            "completion's advantage in that group."
        },
    )
    overlap_natural_only: bool = field(
        default=False,
        metadata={
            "help": "reward_variant='ours': apply the overlap reward ONLY to rows whose "
            "'natural' column is True; non-natural rows (charts, documents, diagrams) are "
            "scored by format + accuracy + judge alone. Grounding-DINO is trained on "
            "photographs, so its boxes -- and hence the overlap score -- are noise on "
            "non-natural imagery. Requires a boolean 'natural' column (build_grpo_sets.py "
            "emits one). Off by default, so mixed-corpus runs stay reproducible. Sweep "
            "dimension — appears in the model/wandb name as natonly."
        },
    )


def init_zero_verbose():
    """
    Perform zero verbose init - use this method on top of the CLI modules to make logging and warning output cleaner.
    Uses Rich if available, falls back otherwise.
    """
    import logging
    import warnings

    FORMAT = "%(message)s"

    if is_rich_available():
        from rich.logging import RichHandler

        handler = RichHandler()
    else:
        handler = logging.StreamHandler()

    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[handler], level=logging.ERROR)

    # Custom warning handler to redirect warnings to the logging system
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        logging.warning(f"{filename}:{lineno}: {category.__name__}: {message}")

    # Add the custom warning handler - we need to do that before importing anything to make sure the loggers work well
    warnings.showwarning = warning_handler


class TrlParser(HfArgumentParser):
    """
    A subclass of [`transformers.HfArgumentParser`] designed for parsing command-line arguments with dataclass-backed
    configurations, while also supporting configuration file loading and environment variable management.

    Args:
        dataclass_types (`Union[DataClassType, Iterable[DataClassType]]` or `None`, *optional*, defaults to `None`):
            Dataclass types to use for argument parsing.
        **kwargs:
            Additional keyword arguments passed to the [`transformers.HfArgumentParser`] constructor.

    Examples:

    ```yaml
    # config.yaml
    env:
        VAR1: value1
    arg1: 23
    ```

    ```python
    # main.py
    import os
    from dataclasses import dataclass
    from trl import TrlParser


    @dataclass
    class MyArguments:
        arg1: int
        arg2: str = "alpha"


    parser = TrlParser(dataclass_types=[MyArguments])
    training_args = parser.parse_args_and_config()

    print(training_args, os.environ.get("VAR1"))
    ```

    ```bash
    $ python main.py --config config.yaml
    (MyArguments(arg1=23, arg2='alpha'),) value1

    $ python main.py --arg1 5 --arg2 beta
    (MyArguments(arg1=5, arg2='beta'),) None
    ```
    """

    def __init__(
        self,
        dataclass_types: Optional[Union[DataClassType, Iterable[DataClassType]]] = None,
        **kwargs,
    ):
        # Make sure dataclass_types is an iterable
        if dataclass_types is None:
            dataclass_types = []
        elif not isinstance(dataclass_types, Iterable):
            dataclass_types = [dataclass_types]

        # Check that none of the dataclasses have the "config" field
        for dataclass_type in dataclass_types:
            if "config" in dataclass_type.__dataclass_fields__:
                raise ValueError(
                    f"Dataclass {dataclass_type.__name__} has a field named 'config'. This field is reserved for the "
                    f"config file path and should not be used in the dataclass."
                )

        super().__init__(dataclass_types=dataclass_types, **kwargs)

    def parse_args_and_config(
        self,
        args: Optional[Iterable[str]] = None,
        return_remaining_strings: bool = False,
        fail_with_unknown_args: bool = True,
    ) -> tuple[DataClass, ...]:
        """
        Parse command-line args and config file into instances of the specified dataclass types.

        This method wraps [`transformers.HfArgumentParser.parse_args_into_dataclasses`] and also parses the config file
        specified with the `--config` flag. The config file (in YAML format) provides argument values that replace the
        default values in the dataclasses. Command line arguments can override values set by the config file. The
        method also sets any environment variables specified in the `env` field of the config file.
        """
        args = list(args) if args is not None else sys.argv[1:]
        if "--config" in args:
            # Get the config file path from
            config_index = args.index("--config")
            args.pop(config_index)  # remove the --config flag
            config_path = args.pop(config_index)  # get the path to the config file
            with open(config_path) as yaml_file:
                config = yaml.safe_load(yaml_file)

            # Set the environment variables specified in the config file
            if "env" in config:
                env_vars = config.pop("env", {})
                if not isinstance(env_vars, dict):
                    raise ValueError("`env` field should be a dict in the YAML file.")
                for key, value in env_vars.items():
                    os.environ[key] = str(value)

            # Set the defaults from the config values
            config_remaining_strings = self.set_defaults_with_config(**config)
        else:
            config_remaining_strings = []

        # Parse the arguments from the command line
        output = self.parse_args_into_dataclasses(args=args, return_remaining_strings=return_remaining_strings)

        # Merge remaining strings from the config file with the remaining strings from the command line
        if return_remaining_strings:
            args_remaining_strings = output[-1]
            return output[:-1] + (config_remaining_strings + args_remaining_strings,)
        elif fail_with_unknown_args and config_remaining_strings:
            raise ValueError(
                f"Unknown arguments from config file: {config_remaining_strings}. Please remove them, add them to the "
                "dataclass, or set `fail_with_unknown_args=False`."
            )
        else:
            return output

    def set_defaults_with_config(self, **kwargs) -> list[str]:
        """
        Overrides the parser's default values with those provided via keyword arguments, including for subparsers.

        Any argument with an updated default will also be marked as not required if it was previously required.

        Returns a list of strings that were not consumed by the parser.
        """

        def apply_defaults(parser, kw):
            used_keys = set()
            for action in parser._actions:
                # Handle subparsers recursively
                if isinstance(action, argparse._SubParsersAction):
                    for subparser in action.choices.values():
                        used_keys.update(apply_defaults(subparser, kw))
                elif action.dest in kw:
                    action.default = kw[action.dest]
                    action.required = False
                    used_keys.add(action.dest)
            return used_keys

        used_keys = apply_defaults(self, kwargs)
        # Remaining args not consumed by the parser
        remaining = [
            item for key, value in kwargs.items() if key not in used_keys for item in (f"--{key}", str(value))
        ]
        return remaining


def get_git_commit_hash(package_name):
    try:
        # Import the package to locate its path
        package = importlib.import_module(package_name)
        # Get the path to the package using inspect
        package_path = os.path.dirname(inspect.getfile(package))

        # Navigate up to the Git repository root if the package is inside a subdirectory
        git_repo_path = os.path.abspath(os.path.join(package_path, ".."))
        git_dir = os.path.join(git_repo_path, ".git")

        if os.path.isdir(git_dir):
            # Run the git command to get the current commit hash
            commit_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=git_repo_path).strip().decode("utf-8")
            )
            return commit_hash
        else:
            return None
    except Exception as e:
        return f"Error: {str(e)}"
