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
    Arguments common to all scripts, plus the saliency reward's own.

    The first block is upstream TRL's and is shared with every other example script in
    the checkout, so it stays as upstream has it. The second block, from
    `reforward_saliency` down, is this repository's: the knobs of Section 3.4's saliency
    reward. There is one for each thing the six arm configs in `training/grpo/configs/`
    vary, and nothing else -- `training.grpo.config.training_flags` is the only producer
    of these flags, and `tests/test_entry_script_arguments.py` holds the two in step in
    both directions.

    The research launcher carried 59 of these. The ones that are gone belonged to the
    gradient and GLIMPSE saliency maps, the AUROC and roll-null metrics, and the placebo,
    mask-free, mismatched-box and length-guard controls -- none of which are in the paper
    and none of whose modules this repository installs. They are intact in the archive;
    see `docs/provenance.md`.

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
    reward_variant: str = field(
        default="saliency_r1",
        metadata={
            "help": "WHICH SALIENCY TERM takes the second slot of reward_funcs. "
            "'ours' = R_sal, Section 3.4: raw observe->patch attention at --overlap_layer "
            "meaned over --overlap_heads, scored per observe step against that step's own "
            "grounded region by --overlap_metric. 'saliency_r1' = the Appendix D.2 "
            "baseline's whole-completion rollout saliency against the question-level box "
            "shipped with the corpus. 'none' = format + accuracy + judge only, which also "
            "skips the attention re-forward entirely. "
            "Note that the paper's No-Sal arm is NOT 'none': it ran 'ours' at alpha_sal 0, "
            "so the term was installed and weighted zero. The gradient is the same and the "
            "checkpoint on disk is the one the number came from -- see "
            "training/grpo/configs/no_sal.yaml.",
            "choices": ["saliency_r1", "ours", "none"],
        },
    )
    token_reduction: str = field(
        default="mean",
        metadata={
            "help": "reward_variant='ours': how a step's per-token saliency maps are "
            "collapsed into one map for the step. 'mean' is Section 3.3's sal_s and is what "
            "every published arm used; the other two are probes.",
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
            "help": "reward_variant='ours': the saliency score phi, i.e. how a step's map "
            "is scored against its grounded region. Unset means 'phi'. "
            "'phi' = Equation 1, the mean of the map inside the region over the map's PEAK "
            "over the whole image. It is what every published arm but one was trained on, "
            "and Section 5.2 is why it works: attention peaks at a grid corner while "
            "grounded regions sit near the centre, so the argmax is usually outside the "
            "region and raising attention inside it raises the numerator without moving "
            "the denominator. It also moves when a map merely FLATTENS, which is a real "
            "property of the metric and the subject of Section 4.4. "
            "'phi_mean' = Appendix C, the same numerator over the map's MEAN. Chance is "
            "exactly 1.0 and the value is invariant to m -> c*m. Its within-group spread is "
            "~12x phi's, so alpha_sal must be rescaled with it (0.4 -> 0.033, see "
            "experiments/alpha_calibration.py); it did not beat the no-sal baseline. "
            "'mean_in' and 'mean_in_v2' are the historical spellings of the two and "
            "resolve to them, so a command line written against the research tree still "
            "runs. selfsal.saliency.score is the one implementation, shared with the "
            "head-selection screen of Section 3.5.",
            "choices": ["phi", "phi_mean", "mean_in", "mean_in_v2"],
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
            "experiments/center_rect_calibration.py: the rectangle is NOT DINO's mask "
            "(area-matched closeness 0.230, against a 0.235 different-image floor), yet the "
            "per-completion reward built on it reproduces the real per-step-DINO reward's "
            "ranking of a group at rho 0.651 vs 0.621 for DINO once per chain — so this "
            "arm asks whether the detector was buying the gradient at all. Note every "
            "step becomes scoreable (nothing is ungroundable), so the scored set is larger "
            "than its DINO reference's. See training/grpo/configs/center_rect.yaml.",
        },
    )
    overlap_rect_placement: str = field(
        default="centre",
        metadata={
            "help": "reward_variant='ours', read only with --overlap_rect_frac: WHERE the "
            "rectangle sits. 'center' (or the British spelling) is the only placement here "
            "and is what Table 5's arm ran -- read back off its logged mask/ring_frac of "
            "0.0008, which the interior placements make exactly zero by construction. "
            "Those placements were a research arm and are in the archive.",
            "choices": ["center", "centre"],
        },
    )
    box_threshold: float = field(
        default=0.10,
        metadata={"help": "reward_variant='ours': Grounding-DINO confidence threshold for per-step boxes. "
                          "Ignored under --overlap_rect_frac (no boxes are requested). Under "
                          "--overlap_question_boxes it is not applied here at all -- the boxes were "
                          "already filtered when the file was built -- but it is still checked against "
                          "the file's, which is what refuses a run against boxes it does not describe."},
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
            "help": "reward_variant='ours': the `question boxes` ablation of Section 5.3. "
            "Path to a file written by training/grpo/precompute_question_boxes.py, holding one "
            "box list per dataset ROW, grounded once on that row's QUESTION before the run. Every "
            "observe step of a row is then scored against that single union, instead of calling "
            "Grounding-DINO once per step on the step's own sentence — which is what prior work "
            "does, so the gap to SELF-SALIENCY is the value of conditioning on the chain. No "
            "detector is loaded at training time at all. Changes WHICH steps are scored: a row "
            "grounds for all of its steps or for none, where per-step grounding decides that step "
            "by step. --box_threshold and the trainer's 512px image cap are recorded in the file "
            "and a mismatch is refused rather than trained through."
        },
    )
    overlap_natural_only: bool = field(
        default=False,
        metadata={
            "help": "reward_variant='ours': apply R_sal ONLY to rows whose 'natural' column "
            "is True; non-natural rows (charts, documents, diagrams) are scored by format + "
            "accuracy + judge alone, with the saliency term MASKED rather than zeroed, so "
            "they stay neutral in the GRPO advantage. Grounding-DINO is trained on "
            "photographs, so its boxes -- and hence the score -- are noise on non-natural "
            "imagery. Requires a boolean 'natural' column, which saliency-r1-8k does not "
            "have; OFF in every arm of the paper, which trains on the whole corpus."
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
