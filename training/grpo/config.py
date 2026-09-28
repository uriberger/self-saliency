# Copyright 2026 NVIDIA. Apache-2.0.
"""Load an arm config, resolve `extends`, and check it is internally consistent.

    python -m training.grpo.config training/grpo/configs/self_saliency.yaml
    python -m training.grpo.config --all          # every arm, as a table

`extends` is a shallow chain with recursive dict merge, so an arm file states only what
it changes and the diff between two arms is the experimental difference. Lists replace
rather than append -- a head set or a LoRA target list is a whole answer, never a
fragment of one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"

#: Every arm the paper reports, in the order they appear in Tables 2 and 5.
ARMS = ("self_saliency", "self_saliency_mean", "no_sal",
        "center_rect", "question_boxes", "saliency_r1")

#: saliency-r1-8k, less the 100-row holdout.
TRAIN_ROWS = 7980


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: str | Path, _seen: tuple = ()) -> dict:
    """Read a config with its `extends` chain resolved."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path.name
    if str(path) in _seen:
        raise ValueError(f"circular extends: {' -> '.join(_seen)} -> {path}")
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _merge(load(CONFIG_DIR / Path(parent).name, _seen + (str(path),)), cfg)
    return cfg


def load_arm(name: str) -> dict:
    return load(CONFIG_DIR / f"{name}.yaml")


def optimizer_steps(cfg: dict) -> int:
    """Steps implied by the config.

    completions/step = ranks * per_device_batch * grad_accum
    prompts/step     = completions/step / G  ==  ranks  (at the paper's settings)
    steps            = rows * epochs / prompts_per_step
    """
    g = cfg["grpo"]
    ranks = int(cfg["training_ranks"])
    completions = ranks * int(g["per_device_train_batch_size"]) * int(g["gradient_accumulation_steps"])
    prompts_per_step = completions // int(g["num_generations"])
    return (TRAIN_ROWS // prompts_per_step) * int(g["num_train_epochs"])


def validate(cfg: dict) -> list[str]:
    """Problems that would produce a run other than the one the config names."""
    problems = []

    want = cfg.get("expect_optimizer_steps")
    got = optimizer_steps(cfg)
    if want is not None and int(want) != got:
        problems.append(
            f"expect_optimizer_steps {want} but training_ranks "
            f"{cfg['training_ranks']} implies {got}")

    sal = cfg["reward"]["saliency"]
    kind = sal.get("kind")
    if kind == "self_saliency":
        if sal.get("metric") not in ("phi", "phi_mean"):
            problems.append(f"unknown saliency metric {sal.get('metric')!r}")
        src = (sal.get("regions") or {}).get("source")
        if src not in ("dino_per_step", "center_rect", "question_boxes"):
            problems.append(f"unknown region source {src!r}")
        # A zero weight is legitimate (no_sal) but never accidental.
        if float(sal.get("alpha", 0)) == 0.0 and cfg.get("name") != "no_sal":
            problems.append("alpha 0.0 on an arm that is not no_sal: the saliency term "
                            "would be installed and contribute nothing")
    elif kind != "saliency_r1":
        problems.append(f"unknown saliency kind {kind!r}")

    lora = cfg["lora"]
    if lora["targets"] != ["q_proj", "v_proj"]:
        problems.append(
            f"LoRA targets {lora['targets']} are not the paper's [q_proj, v_proj]; "
            "every published adapter reads back q_proj+v_proj, and changing this "
            "invalidates resuming from one")
    return problems


def reward_weights(cfg: dict) -> list[float] | None:
    """The weight vector, in `reward_funcs` order: [format, saliency, direct, judge]."""
    r = cfg["reward"]
    w = [float(r["format"]["alpha"]), float(r["saliency"]["alpha"]),
         float(r["direct"]["alpha"]), float(r["judge"]["alpha"])]
    # All-ones is expressed as None upstream, and the Saliency-R1 arm was run that way.
    return None if all(x == 1.0 for x in w) else w


def _summary(name: str) -> dict:
    cfg = load_arm(name)
    sal = cfg["reward"]["saliency"]
    return {
        "arm": name,
        "kind": sal.get("kind"),
        "alpha_sal": sal.get("alpha"),
        "metric": sal.get("metric", "-"),
        "regions": (sal.get("regions") or {}).get("source", "-"),
        "ranks": cfg["training_ranks"],
        "steps": optimizer_steps(cfg),
        "weights": reward_weights(cfg),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="path to an arm config")
    ap.add_argument("--all", action="store_true", help="summarise every paper arm")
    ap.add_argument("--emit-flags", metavar="OUTPUT_DIR",
                    help="print the trainer's argument list for this config, one per "
                         "line. This is what training/grpo/run.sh consumes, so the "
                         "shell never reimplements any of the logic above.")
    ap.add_argument("--emit-layout", action="store_true",
                    help="print TRAINING_RANKS / NEEDS_DINO / EXPECT_STEPS as shell "
                         "assignments, for the same reason.")
    args = ap.parse_args(argv)

    if args.config and args.emit_flags:
        cfg = load(args.config)
        problems = validate(cfg)
        for p in problems:
            print(f"!! {p}", file=sys.stderr)
        if problems:
            return 1
        for flag in training_flags(cfg, args.emit_flags):
            print(flag)
        return 0

    if args.config and args.emit_layout:
        cfg = load(args.config)
        print(f"TRAINING_RANKS={int(cfg['training_ranks'])}")
        print(f"NEEDS_DINO={'true' if needs_detector(cfg) else 'false'}")
        print(f"GENERATION={cfg.get('generation', 'vllm')}")
        print(f"EXPECT_STEPS={optimizer_steps(cfg)}")
        print(f"ARM_NAME={cfg.get('name', 'unnamed')}")
        return 0

    if args.all or not args.config:
        cols = ["arm", "kind", "alpha_sal", "metric", "regions", "ranks", "steps", "weights"]
        rows = [_summary(a) for a in ARMS]
        widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
        print("  ".join(c.ljust(widths[c]) for c in cols))
        print("  ".join("-" * widths[c] for c in cols))
        bad = 0
        for a, r in zip(ARMS, rows):
            print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
            for p in validate(load_arm(a)):
                print(f"    !! {a}: {p}")
                bad += 1
        return 1 if bad else 0

    cfg = load(args.config)
    print(yaml.safe_dump(cfg, sort_keys=False))
    problems = validate(cfg)
    for p in problems:
        print(f"!! {p}", file=sys.stderr)
    return 1 if problems else 0



# ---------------------------------------------------------------------------
# Turning a config into a command line
# ---------------------------------------------------------------------------

def needs_detector(cfg: dict) -> bool:
    """Whether this arm calls Grounding-DINO during training.

    Only per-step grounding does. The center_rect region is a function of the patch grid,
    and question_boxes reads a file grounded before the run -- so both arms run with no
    detector at all, and the launcher does not start the sidecar.
    """
    sal = cfg["reward"]["saliency"]
    if sal.get("kind") != "self_saliency":
        return False
    return (sal.get("regions") or {}).get("source") == "dino_per_step"


def training_flags(cfg: dict, output_dir: str) -> list[str]:
    """The argument list for examples/scripts/grpo_vlm_qwen3.py.

    Built here rather than in the shell so that the six arm configs, the checks in
    `validate`, and what actually reaches the trainer are one implementation. The
    research launcher carried 72 flags across 2,377 lines of shell and the arms differed
    by which of them were set; here the arm IS the config and this is a projection of it.
    """
    sal = cfg["reward"]["saliency"]
    regions = sal.get("regions") or {}
    g, o, lora = cfg["grpo"], cfg["optim"], cfg["lora"]

    weights = reward_weights(cfg)
    args = [
        "--model_name_or_path", str(cfg["base_model"]),
        "--dataset_name", str(cfg["data"]["dataset"]),
        "--output_dir", str(output_dir),
        "--attn_implementation", "sdpa",
        "--torch_dtype", "bfloat16",
        "--learning_rate", repr(float(o["learning_rate"])),
        "--lr_scheduler_type", str(o["lr_scheduler"]),
        "--warmup_steps", str(int(o["warmup_steps"])),
        "--weight_decay", str(float(o["weight_decay"])),
        "--max_grad_norm", str(float(o["max_grad_norm"])),
        "--beta", str(float(o["beta"])),
        "--max_prompt_length", str(int(g["max_prompt_length"])),
        "--max_completion_length", str(int(g["max_completion_length"])),
        "--num_generations", str(int(g["num_generations"])),
        "--per_device_train_batch_size", str(int(g["per_device_train_batch_size"])),
        "--gradient_accumulation_steps", str(int(g["gradient_accumulation_steps"])),
        "--num_train_epochs", str(int(g["num_train_epochs"])),
        "--temperature", str(float(g["temperature"])),
        "--top_p", str(float(g["top_p"])),
        "--repetition_penalty", str(float(g["repetition_penalty"])),
        "--seed", str(int(g["seed"])),
        "--use_peft",
        "--lora_r", str(int(lora["rank"])),
        "--lora_alpha", str(int(lora["alpha"])),
        "--lora_dropout", str(float(lora["dropout"])),
        "--lora_target_modules", *lora["targets"],
        "--log_completions",
        "--logging_steps", "5",
    ]
    if weights is not None:
        args += ["--reward_weights", *(str(w) for w in weights)]
    if cfg.get("generation", "vllm") == "vllm":
        args += ["--use_vllm", "--vllm_mode", "server"]

    if sal.get("kind") == "saliency_r1":
        args += ["--reward_variant", "saliency_r1"]
        return args

    args += [
        "--reward_variant", "ours",
        "--overlap_metric", str(sal["metric"]),
        "--overlap_layer", str(int(sal["layer"])),
        "--overlap_heads", ",".join(str(h) for h in sal["heads"]),
        "--token_reduction", str(sal["token_reduction"]),
    ]
    source = regions.get("source")
    if source == "dino_per_step":
        args += ["--box_threshold", str(float(regions["box_threshold"])),
                 "--max_box_area", str(float(regions["max_box_area"]))]
        if regions.get("max_union_area"):
            args += ["--max_union_area", str(float(regions["max_union_area"]))]
    elif source == "center_rect":
        args += ["--overlap_rect_frac", str(float(regions["frac"])),
                 "--overlap_rect_placement", str(regions["placement"])]
    elif source == "question_boxes":
        args += ["--overlap_question_boxes", str(regions["file"]),
                 "--box_threshold", str(float(regions["box_threshold"]))]
    return args

if __name__ == "__main__":
    raise SystemExit(main())
