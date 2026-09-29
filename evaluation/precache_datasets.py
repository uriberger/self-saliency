#!/usr/bin/env python
"""Pre-populate the HF datasets cache for lmms-eval benchmarks, unauthenticated.

Why this is needed
------------------
335 lmms-eval task configs set `dataset_kwargs: {token: True}`, which makes
`datasets.load_dataset` call `build_hf_headers(token=True)` and raise
`LocalTokenNotFoundError` when no HF token is configured -- even though the
datasets themselves are public and load fine without one. This cluster has no
HF token, so every not-yet-cached benchmark fails at load time.

Loading each dataset *without* `token=True` populates the same cache that the
task loader later reads, so the evaluation run then succeeds. lmms-eval logs a
misleading "couldn't be found on the Hugging Face Hub" and silently falls back
to that cache -- which is exactly the behaviour we are relying on here.

This does not patch the task YAMLs, so results stay identical to the old
cluster's. Setting HF_TOKEN would also fix it and makes this script unnecessary.

Usage:
    HF_HOME=/path/to/hf_cache python evaluation/precache_datasets.py
    ... --tasks mmstar,chartqa      # subset
    ... --list                      # resolve and print, download nothing

Run it on a login node: pure network + disk, no GPU.
"""
import argparse
import os
import sys
import traceback
from pathlib import Path

import yaml

# The union of scripts/eval_saliency_r1_benchmarks.sh and eval_our_benchmarks.sh.
DEFAULT_TASKS = [
    # Saliency-R1 paper suite
    "chartqa",
    "illusionvqa_soft_localization",
    "mme",
    "mmstar",
    "p3",
    "pope",
    "scienceqa_img",
    "mmerealworld",
    # Our reasoning suite
    "algopuzzlevqa",
    "cv_bench",
    "dailyclue",
    "hallusion_bench_image",
    "logicvista_reasoning",
    "mathverse_testmini_vision_only",
    "mathvision_testmini",
    "mathvista_testmini_cot",
    "mathvista_testmini_solution",
    "mmk12",
    "mmmu_pro_standard",
    "omnispatial_test",
    "realworldqa",
    "visulogic",
    "vstar_bench",
    "wemath_testmini_reasoning",
    # HR-Bench is the biggest download here by a wide margin: 800 4K images and
    # 800 8K ones, stored full-size.
    "hrbench4k",
    "hrbench8k",
]


def _load_yaml(path):
    """YAML with lmms-eval's !function tags ignored -- we only want the metadata."""

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_constructor("!function", lambda loader, node: None)
    _Loader.add_multi_constructor("!", lambda loader, suffix, node: None)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_Loader) or {}


def find_task_configs(tasks_root):
    """Map task name -> resolved config, following `include:` templates."""
    by_name = {}
    for dirpath, _, filenames in os.walk(tasks_root):
        for fn in filenames:
            if not fn.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(dirpath, fn)
            try:
                cfg = _load_yaml(path)
            except Exception:
                continue
            if not isinstance(cfg, dict):
                continue
            # `include` supplies defaults; keys in the task file win. Resolved
            # BEFORE reading `task`, because a task's name is not always written
            # in its own file: vstar_bench.yaml carries only the metric list and
            # inherits `task:` (and the dataset) from _default_template_yaml, so
            # requiring the key here dropped the benchmark entirely.
            inherited = not isinstance(cfg.get("task"), str)
            inc = cfg.get("include")
            if inc:
                inc_path = os.path.join(dirpath, inc)
                if os.path.exists(inc_path):
                    try:
                        merged = _load_yaml(inc_path)
                        merged.update(cfg)
                        cfg = merged
                    except Exception:
                        pass
            name = cfg.get("task")
            if not isinstance(name, str):
                continue
            # A file that spells out its own `task` outranks one that only
            # inherits the name, so a sibling sharing the same template cannot
            # displace the real config for that name.
            if inherited and name in by_name:
                continue
            cfg["__path__"] = path
            by_name[name] = cfg
    return by_name


def precache_one(cfg, name):
    """Return (status, detail). Never raises."""
    import datasets

    path = cfg.get("dataset_path")
    if not path:
        return "skip", "no dataset_path"

    kwargs = cfg.get("dataset_kwargs") or {}
    if kwargs.get("load_from_disk"):
        # Built locally (e.g. DailyClue via build_dailyclue_hf.py), never fetched.
        return "skip", f"load_from_disk ({path})"

    subset = cfg.get("dataset_name")
    split = cfg.get("test_split")
    label = f"{path}" + (f" [{subset}]" if subset else "") + (f" split={split}" if split else "")

    # Deliberately no token= argument: that is the whole point.
    load_kwargs = {}
    if subset:
        load_kwargs["name"] = subset
    if split:
        load_kwargs["split"] = split

    try:
        ds = datasets.load_dataset(path, **load_kwargs)
        n = ds.num_rows if hasattr(ds, "num_rows") else "?"
        return "ok", f"{label} -> {n} rows"
    except Exception as e:
        return "fail", f"{label} -> {type(e).__name__}: {str(e)[:200]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="comma-separated task names")
    ap.add_argument(
        "--lmms-eval-dir",
        default=os.environ.get(
            "LMMS_EVAL_DIR", str(Path(__file__).resolve().parent / "lmms_eval")),
    )
    ap.add_argument("--list", action="store_true", help="resolve and print, download nothing")
    args = ap.parse_args()

    tasks_root = os.path.join(args.lmms_eval_dir, "lmms_eval", "tasks")
    if not os.path.isdir(tasks_root):
        sys.exit(f"no lmms-eval tasks dir at {tasks_root}")

    print(f"HF_HOME={os.environ.get('HF_HOME', '<unset>')}")
    configs = find_task_configs(tasks_root)
    wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]

    results = []
    for name in wanted:
        cfg = configs.get(name)
        if cfg is None:
            results.append((name, "missing", "no task config found"))
            continue
        if args.list:
            kw = cfg.get("dataset_kwargs") or {}
            results.append(
                (
                    name,
                    "list",
                    f"{cfg.get('dataset_path')} name={cfg.get('dataset_name')} "
                    f"split={cfg.get('test_split')} token={kw.get('token')}",
                )
            )
            continue
        print(f"\n=== {name} ===", flush=True)
        try:
            status, detail = precache_one(cfg, name)
        except Exception:
            status, detail = "fail", traceback.format_exc(limit=1).strip()[:200]
        print(f"  {status}: {detail}", flush=True)
        results.append((name, status, detail))

    print("\n================ SUMMARY ================")
    counts = {}
    for name, status, detail in results:
        counts[status] = counts.get(status, 0) + 1
        print(f"{status.upper():8} {name:32} {detail}")
    print("-----------------------------------------")
    print(", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 1 if counts.get("fail") or counts.get("missing") else 0


if __name__ == "__main__":
    sys.exit(main())
