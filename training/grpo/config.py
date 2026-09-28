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
    args = ap.parse_args(argv)

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


if __name__ == "__main__":
    raise SystemExit(main())
