# Copyright 2026 NVIDIA. Apache-2.0.
"""Each arm config must describe the run whose number is in the paper.

A config file is a claim: "launching this reproduces Table 2's `Ours` column". Nothing
enforces that by itself, and the failure mode is silent -- a wrong default produces a
run that trains fine, finishes, and disagrees with the paper by a point, months later.

So this reads the published checkpoints and compares. Every field below came off disk:

  training_args.bin     the TrainingArguments the run actually used -- learning rate,
                        schedule, beta, G, accumulation, the reward WEIGHT VECTOR
  adapter_config.json   the LoRA that was actually trained
  checkpoint-N          the optimizer step count that was actually reached

Skipped when the archive repo is not on this machine; set SELFSAL_ARCHIVE.

WHY THE WEIGHT VECTOR IS THE CENTRE OF THIS. `reward_weights` is positional against
`reward_funcs = [format, saliency, direct, judge]`, so it encodes all four alphas of
Section 3.4 in one authoritative artefact. If a config's alphas are right, the arm's
objective is right.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

ARCHIVE = Path(os.environ.get(
    "SELFSAL_ARCHIVE", Path.home() / "scratch/research/saliency_r1"))

#: arm -> the archive checkpoint directory whose 25-benchmark mean is the paper's number.
#: Identified by matching the score, not the name -- see docs/reproduce.md.
CHECKPOINTS = {
    "self_saliency":
        "grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged"
        "-overlap__wov0.4_2head_trmean",
    "self_saliency_mean":
        "grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged"
        "-overlap__wov0.033_2head_trmean_saliency_r1_8k_mean_in_v2",
    "no_sal":
        "grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged"
        "-no-saliency_saliency_r1_8k",
    "center_rect":
        "grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-rect-frac",
    "question_boxes":
        "grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-question-boxes",
    "saliency_r1":
        "grpo-coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged-saliency-r1-qwen3",
}

pytestmark = pytest.mark.skipif(
    not (ARCHIVE / "checkpoint").is_dir(),
    reason=f"archive checkpoints not present at {ARCHIVE}")


class _Stub:
    """Stand-in for a class this environment does not have.

    Unpickling only needs something to hang the attribute dict on.
    """

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


@contextlib.contextmanager
def _stub_modules(*names: str):
    """Make `import <name>` succeed with a module whose every attribute is a class.

    `training_args.bin` is a pickled `GRPOConfig`, so loading it normally requires a
    working TRL install -- i.e. the training environment. The test only wants the
    attribute dict, so the classes are stubbed and any environment with torch can run
    it. Without this the check that matters most would be skipped exactly where it is
    most useful: on a fresh clone.
    """
    def _getattr(attr):
        # Dunders must still raise: `inspect` asks a module for __file__ and friends
        # while formatting a traceback, and handing it a class instead turns any real
        # assertion failure into an unrelated AttributeError from inspect.
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return _Stub

    saved = {n: sys.modules.get(n) for n in names}
    try:
        for name in names:
            module = types.ModuleType(name)
            module.__getattr__ = _getattr                  # type: ignore[attr-defined]
            module.__path__ = []                           # allow submodule imports
            sys.modules[name] = module
        yield
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _training_args(arm: str) -> dict:
    torch = pytest.importorskip("torch")
    path = ARCHIVE / "checkpoint" / CHECKPOINTS[arm] / "training_args.bin"
    if not path.exists():
        pytest.skip(f"{arm}: no training_args.bin at {path}")
    try:
        loaded = torch.load(path, weights_only=False)
    except ModuleNotFoundError:
        missing = ("trl", "trl.trainer", "trl.trainer.grpo_config",
                   "trl.trainer.grpo_trainer_qwen3")
        with _stub_modules(*(m for m in missing if m not in sys.modules)):
            loaded = torch.load(path, weights_only=False)
    return vars(loaded)


def _reached_steps(arm: str) -> int:
    root = ARCHIVE / "checkpoint" / CHECKPOINTS[arm]
    steps = [int(p.name.split("-")[1]) for p in root.glob("checkpoint-*")
             if p.name.split("-")[-1].isdigit()]
    if not steps:
        pytest.skip(f"{arm}: no checkpoint-N directory")
    return max(steps)


@pytest.fixture(scope="module")
def cfgmod():
    from training.grpo import config
    return config


@pytest.mark.parametrize("arm", list(CHECKPOINTS))
def test_alphas_match_the_trained_weight_vector(arm, cfgmod):
    """Section 3.4's four alphas, against the vector the run was trained with."""
    cfg = cfgmod.load_arm(arm)
    want = _training_args(arm).get("reward_weights")
    got = cfgmod.reward_weights(cfg)

    if want is None:
        assert got is None, (
            f"{arm}: the run used reward_weights=None (all weights 1.0) but the config "
            f"implies {got}")
        return
    want = [round(float(x), 6) for x in want]
    assert got is not None, f"{arm}: the run used {want} but the config implies all-ones"
    assert [round(x, 6) for x in got] == want


@pytest.mark.parametrize("arm", list(CHECKPOINTS))
def test_optimizer_steps_match(arm, cfgmod):
    """training_ranks is not derivable from the GPU layout, so it is checked directly.

    center_rect and question_boxes need no detector, and the launcher gives DINO's GPU
    to training when one is not needed -- which would be 7 ranks and 3,420 steps. Both
    reached 3,990. A config that inferred ranks from the sidecars would quietly retrain
    two of the five arms at a different length.
    """
    cfg = cfgmod.load_arm(arm)
    assert cfgmod.optimizer_steps(cfg) == _reached_steps(arm)
    assert cfg["expect_optimizer_steps"] == _reached_steps(arm)


@pytest.mark.parametrize("arm", list(CHECKPOINTS))
def test_optimizer_and_sampling_match(arm, cfgmod):
    cfg = cfgmod.load_arm(arm)
    args = _training_args(arm)
    o, g = cfg["optim"], cfg["grpo"]

    assert float(args["learning_rate"]) == float(o["learning_rate"])
    assert str(args["lr_scheduler_type"]).lower().endswith(o["lr_scheduler"])
    assert int(args["warmup_steps"]) == int(o["warmup_steps"])
    assert float(args["weight_decay"]) == float(o["weight_decay"])
    assert float(args["max_grad_norm"]) == float(o["max_grad_norm"])
    assert float(args["beta"]) == float(o["beta"])

    for key in ("num_generations", "per_device_train_batch_size",
                "gradient_accumulation_steps", "max_prompt_length",
                "max_completion_length", "seed"):
        assert int(args[key]) == int(g[key]), f"{arm}: {key}"
    for key in ("temperature", "top_p", "repetition_penalty"):
        assert float(args[key]) == float(g[key]), f"{arm}: {key}"
    assert bool(args["scale_rewards"]) == bool(g["scale_rewards"])
    assert int(args["num_train_epochs"]) == int(g["num_train_epochs"])


@pytest.mark.parametrize("arm", list(CHECKPOINTS))
def test_lora_matches_the_trained_adapter(arm, cfgmod):
    """Appendix A.2's rank 16 / alpha 32 / dropout 0.05 on q_proj and v_proj.

    The launcher's default moved to q,k,v after these runs, so this is exactly the field
    a config built from current defaults would get wrong.
    """
    path = ARCHIVE / "checkpoint" / CHECKPOINTS[arm] / "adapter_config.json"
    if not path.exists():
        pytest.skip(f"{arm}: no adapter_config.json")
    trained = json.loads(path.read_text())
    lora = cfgmod.load_arm(arm)["lora"]

    assert int(trained["r"]) == int(lora["rank"])
    assert int(trained["lora_alpha"]) == int(lora["alpha"])
    assert float(trained["lora_dropout"]) == float(lora["dropout"])
    assert sorted(trained["target_modules"]) == sorted(lora["targets"])


@pytest.mark.parametrize("arm", list(CHECKPOINTS))
def test_config_is_self_consistent(arm, cfgmod):
    assert cfgmod.validate(cfgmod.load_arm(arm)) == []


def test_center_rect_placement_is_center_not_interior(cfgmod):
    """Read off the run's logged mask diagnostics, not assumed.

    Both interior placements guarantee a ring fraction of exactly 0.000 by construction.
    The run logged 0.0008 -- small, because a few grids in the corpus are small enough
    that a centred rectangle reaches their border, but not zero. That rules the interior
    placements out and leaves `center`.
    """
    cfg = cfgmod.load_arm("center_rect")
    regions = cfg["reward"]["saliency"]["regions"]
    assert regions["source"] == "center_rect"
    assert regions["placement"] == "center"
    assert regions["frac"] == 0.565

    state = (ARCHIVE / "checkpoint" / CHECKPOINTS["center_rect"]
             / "checkpoint-3990" / "trainer_state.json")
    if not state.exists():
        pytest.skip("no trainer_state.json for center_rect")
    history = json.loads(state.read_text())["log_history"]
    ring = [r["mask/ring_frac"] for r in history if "mask/ring_frac" in r]
    assert ring, "the run logged no mask/ring_frac, so the placement cannot be read back"
    assert max(ring) > 0.0, (
        "ring_frac is identically zero, which is the interior placements' contract -- "
        "this run would not be `center`")

    # And the realised coverage is the 0.565 rectangle's, not some other fraction's.
    union = [r["mask/union_frac"] for r in history if "mask/union_frac" in r]
    assert union and 0.55 < sum(union) / len(union) < 0.60


def test_question_boxes_threshold_matches_its_box_file(cfgmod):
    """The trainer refuses a file built at a different threshold, so these must agree."""
    cfg = cfgmod.load_arm("question_boxes")
    regions = cfg["reward"]["saliency"]["regions"]

    box_file = ARCHIVE / "outputs/question_boxes/saliency_r1_8k_bt0.10.json"
    if not box_file.exists():
        pytest.skip("precomputed question boxes not present")
    built = json.loads(box_file.read_text())["config"]

    assert float(built["box_threshold"]) == float(regions["box_threshold"])
    assert built["text_column"] == regions["text_column"]
    assert built["dataset"] == cfg["data"]["dataset"]
    assert int(built["max_image_side"]) == int(cfg["data"]["max_image_side"])


@pytest.mark.parametrize("arm", list(CHECKPOINTS))
def test_generation_backend_matches(arm, cfgmod):
    """vLLM or in-process, as the run actually did it.

    Not cosmetic. vLLM occupies a GPU of its own, so an arm that generates in-process has
    that card available for training -- which is how Saliency-R1 fits 8 training ranks on
    8 GPUs while our arms fit 6. Get this wrong and the launcher either demands a ninth
    GPU or silently reduces the rank count, and the rank count sets the step count.
    """
    cfg = cfgmod.load_arm(arm)
    want = bool(_training_args(arm).get("use_vllm"))
    got = cfg.get("generation", "vllm") == "vllm"
    assert got == want, (
        f"{arm}: the run had use_vllm={want}, the config says generation="
        f"{cfg.get('generation', 'vllm')!r}")


@pytest.mark.parametrize("arm", list(CHECKPOINTS))
def test_emitted_flags_carry_the_alphas_and_the_lora(arm, cfgmod):
    """What `run.sh` hands the trainer must still be the arm.

    config.py builds the command line, and everything above checks the config. This
    checks the projection of it -- the step between "the config is right" and "the
    trainer was told the right thing", which is where a launcher usually goes wrong.
    """
    cfg = cfgmod.load_arm(arm)
    flags = cfgmod.training_flags(cfg, "/tmp/out")

    def value_of(flag):
        return flags[flags.index(flag) + 1] if flag in flags else None

    lora = cfg["lora"]
    assert value_of("--lora_r") == str(lora["rank"])
    assert value_of("--lora_alpha") == str(lora["alpha"])
    i = flags.index("--lora_target_modules")
    assert flags[i + 1:i + 1 + len(lora["targets"])] == lora["targets"]

    weights = cfgmod.reward_weights(cfg)
    if weights is None:
        assert "--reward_weights" not in flags, (
            f"{arm} ran with reward_weights=None (all ones); passing them explicitly "
            f"would be equivalent but would not be what the checkpoint records")
    else:
        i = flags.index("--reward_weights")
        assert [float(x) for x in flags[i + 1:i + 1 + len(weights)]] == weights

    assert ("--use_vllm" in flags) == (cfg.get("generation", "vllm") == "vllm")
