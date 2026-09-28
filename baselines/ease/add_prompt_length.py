#!/usr/bin/env python3
# Copyright 2026 NVIDIA. Apache-2.0.
"""Add the `prompt_length` column to EASE's train/val parquet, in parallel.

Their `scripts/prepare_ease_dataset.py --model_path ...` already computes this
column, and their RLHFDataset has a fast path that uses it instead of
re-tokenizing every sample at start-up:

    if "prompt_length" in self.dataset.column_names:   # verl/utils/dataset.py
        ...filter by the column...

But their converter computes it in one process, and one row costs ~1.7 s: the
image is resized to `min_pixels` (262144 in examples/config.yaml, which for
saliency-r1-8k means UPSCALING nearly every picture) and then patchified. That
is about four hours for 8,080 rows. Without the column, the same work happens
at every training start across 16 dataset workers -- for both the train and the
val file, and again on every resume.

So this does exactly their computation, in a process pool. `build_prompt_length_fn`
is imported from their converter rather than reimplemented, which is the point:
a prompt_length measured by different code than the one that filters on it is
worse than no column at all.

Usage:
    python3 add_ease_prompt_length.py --data-dir cold_data/ease/saliency_r1_8k \
        --model checkpoint/..._tf457 --jobs 16
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

_COMPUTE = None
_ARGS = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True, help="Holds parquet/ and images/.")
    p.add_argument("--model", type=Path, required=True, help="Processor path (a tf-4.57 staged checkpoint).")
    p.add_argument("--ease-repo", type=Path, default=None, help="Defaults to $EASE_REPO or ./ease_repo.")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    p.add_argument("--min-pixels", type=int, default=262144)
    p.add_argument("--max-pixels", type=int, default=4194304)
    p.add_argument("--max-prompt-length", type=int, default=2048, help="Only used for the report.")
    p.add_argument("--splits", nargs="*", default=["train", "val"])
    p.add_argument("--force", action="store_true", help="Recompute even if the column exists.")
    return p.parse_args()


def resolve_ease_repo(explicit: Path | None) -> Path:
    for candidate in (explicit, Path(os.environ["EASE_REPO"]) if os.environ.get("EASE_REPO") else None,
                      Path.cwd() / "ease_repo"):
        if candidate is not None and (candidate / "scripts/prepare_ease_dataset.py").is_file():
            return candidate
    sys.exit("Could not find ease_repo/. Pass --ease-repo or set EASE_REPO.")


def load_their_converter(ease_repo: Path):
    path = ease_repo / "scripts/prepare_ease_dataset.py"
    spec = importlib.util.spec_from_file_location("ease_prepare_dataset", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ease_prepare_dataset"] = module
    spec.loader.exec_module(module)
    return module


def _init_worker(payload: dict) -> None:
    """Build the processor once per worker; it is the expensive part."""
    global _COMPUTE, _ARGS
    import torch

    torch.set_num_threads(1)
    module = load_their_converter(Path(payload["ease_repo"]))
    _ARGS = SimpleNamespace(
        model_path=payload["model_path"],
        format_prompt=Path(payload["format_prompt"]),
        min_pixels=payload["min_pixels"],
        max_pixels=payload["max_pixels"],
    )
    _COMPUTE = module.build_prompt_length_fn(_ARGS)


def _one(task: tuple[str, str]) -> int:
    problem, image_path = task
    return int(_COMPUTE(problem, Path(image_path)))


def main() -> None:
    args = parse_args()

    # Before any child is forked, and before torch is imported anywhere. Each
    # worker's torch/numpy would otherwise open one OMP thread per core, so
    # --jobs 16 becomes 16 x ncores threads fighting over ncores: measured at
    # 13% CPU per worker and ~40x slower than the single-process path it is
    # supposed to replace. This work is per-row and embarrassingly parallel;
    # the parallelism belongs in the pool, not inside each row.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import pandas as pd

    ease_repo = resolve_ease_repo(args.ease_repo)
    image_root = args.data_dir / "images"
    payload = {
        "ease_repo": str(ease_repo),
        "model_path": str(args.model),
        "format_prompt": str(ease_repo / "examples/format_prompt/perception.jinja"),
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
    }

    for split in args.splits:
        path = args.data_dir / "parquet" / f"{split}.parquet"
        if not path.is_file():
            print(f"[skip] {path} does not exist")
            continue
        df = pd.read_parquet(path)
        if "prompt_length" in df.columns and not args.force:
            print(f"[skip] {path} already has prompt_length ({len(df)} rows)")
            continue

        tasks = [
            (row["problem"], str(image_root / row["images"][0]))
            for _, row in df.iterrows()
        ]
        print(f"[{split}] {len(tasks)} rows on {args.jobs} workers", flush=True)
        started = time.monotonic()
        lengths = []
        with ProcessPoolExecutor(
            max_workers=args.jobs, initializer=_init_worker, initargs=(payload,)
        ) as pool:
            for done, length in enumerate(pool.map(_one, tasks, chunksize=8), start=1):
                lengths.append(length)
                if done % 500 == 0 or done == len(tasks):
                    rate = done / max(time.monotonic() - started, 1e-6)
                    eta = (len(tasks) - done) / max(rate, 1e-6)
                    print(f"[{split}] {done}/{len(tasks)}  {rate:.1f} rows/s  eta {eta / 60:.1f} min",
                          flush=True)

        df["prompt_length"] = lengths
        df.to_parquet(path, index=False)

        over = sum(1 for n in lengths if n > args.max_prompt_length)
        ordered = sorted(lengths)
        print(
            f"[{split}] median {ordered[len(ordered) // 2]}  "
            f"p99 {ordered[int(0.99 * (len(ordered) - 1))]}  max {ordered[-1]}  "
            f"over max_prompt_length={args.max_prompt_length}: {over}"
        )
        print(f"[{split}] wrote {path}")


if __name__ == "__main__":
    main()
