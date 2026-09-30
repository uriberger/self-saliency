#!/usr/bin/env python3
"""Re-score cached MMMU-Pro lmms-eval runs after the random-fallback fix.

``_task_utils/mmmu_mcq_utils.py`` used to return ``random.choice(all_choices)``
when no option letter could be parsed out of a response, so a parse failure was
graded at ~1/n_choices instead of 0 -- and the run was not reproducible. It now
returns ``""``. Cached runs still carry the old randomly-guessed predictions.

This replays every cached ``*_samples_mmmu_pro_*.jsonl`` through the *full*
deployed chain (``mmmu_pro.utils.mmmu_pro_process_results``: reasoned-choice ->
explicit "Answer: X" -> general multi-choice parser), re-aggregates with the
task's own ``mmmu_pro_aggregate_results``, and rewrites ``mmmu_acc,none`` in the
sibling ``*_results.json``.

The Qwen3-VL-8B checkpoints are unaffected (0 no-parse samples); the movement is
in the 7B baselines. Originals are backed up with a ``.mmmu_bak`` suffix.

Needs the MMMU-Pro dataset for the per-question options (samples files store the
gold answer but not the option list). Reads it from the HF cache; set
HF_HOME=${HF_HOME:?set HF_HOME} and HF_HUB_OFFLINE=1 to avoid a
network round-trip.

Run from the repo root:  python scripts/rescore_mmmu_pro.py [--dry-run]
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# `evaluation/results` IS the lmms-eval tree here. The archive kept `<root>/inference`
# and `<root>/lmms_eval` side by side and the port flattened them, so the extra level
# below made the default glob match nothing and the script report no work to do.
LMMS_EVAL_DIR = REPO / "results"
# The scorer comes from the PINNED submodule, not from a checkout beside this one. A
# re-score is only meaningful against the scorer the paper's numbers were produced by.
# `evaluation/lmms_eval` is checked out at 4ea4f15 = a9a806b, the commit those numbers
# were scored under, plus one commit that only stamps a version and changes no score.
# The LMMS_EVAL_DIR environment variable overrides it; note that it means the CHECKOUT,
# not the constant of the same name above.
LMMS_EVAL_SRC = Path(os.environ.get("LMMS_EVAL_DIR", REPO / "lmms_eval"))

# task -> HF dataset config holding that task's options
TASK_CONFIG = {
    "mmmu_pro_standard": "standard (10 options)",
    "mmmu_pro_standard_cot": "standard (10 options)",
    "mmmu_pro_vision": "vision",
    "mmmu_pro_vision_cot": "vision",
}


def _load_task_utils():
    """Import the deployed mmmu_pro/utils.py without importing all of lmms_eval.

    ``lmms_eval.tasks.__init__`` pulls in loguru and the whole task registry,
    which this script has no use for. mmmu_pro/utils.py only needs
    ``lmms_eval.tasks._task_utils.mmmu_mcq_utils``, so stub the parent packages
    and load that one module from its real path -- the scorer we exercise is
    then byte-for-byte the one lmms-eval runs.
    """
    for name in ("lmms_eval", "lmms_eval.tasks", "lmms_eval.tasks._task_utils"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []  # mark as a package
            sys.modules[name] = mod

    mcq_path = LMMS_EVAL_SRC / "lmms_eval/tasks/_task_utils/mmmu_mcq_utils.py"
    spec = importlib.util.spec_from_file_location("lmms_eval.tasks._task_utils.mmmu_mcq_utils", mcq_path)
    mcq = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mcq
    spec.loader.exec_module(mcq)

    utils_path = LMMS_EVAL_SRC / "lmms_eval/tasks/mmmu_pro/utils.py"
    spec = importlib.util.spec_from_file_location("mmmu_pro_utils", utils_path)
    U = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(U)
    return U, mcq


def _load_docs(config: str) -> dict:
    from datasets import load_dataset

    ds = load_dataset("MMMU/MMMU_Pro", config, split="test")
    return {d["id"]: {"id": d["id"], "subject": d["subject"], "answer": d["answer"], "options": d["options"]} for d in ds}


def rescore_file(samples_path: Path, task: str, docs: dict, U, mcq, dry_run: bool) -> dict:
    samples = [json.loads(l) for l in samples_path.read_text().splitlines() if l.strip()]
    ts = samples_path.name.split(f"_samples_{task}.jsonl")[0]
    results_path = samples_path.parent / f"{ts}_results.json"

    aggregate_input = []
    old_correct = new_correct = noparse = missing = 0
    # Two distinct reasons a stored prediction can change, worth keeping apart:
    #   to_noanswer -- the random.choice fallback used to invent a letter here
    #   drift       -- the run was scored by an older parser version and now
    #                  resolves to a *different* letter (nothing to do with the
    #                  fallback fix; re-scoring normalizes it to the current one)
    to_noanswer = drift = 0

    for s in samples:
        block = s.get("mmmu_acc")
        if block is None:
            continue
        doc = docs.get(block["id"])
        if doc is None:  # id not in this config (shouldn't happen)
            missing += 1
            continue
        resp = s["filtered_resps"]
        resp = resp[0] if isinstance(resp, list) else resp

        out = U.mmmu_pro_process_results(doc, [str(resp)])["mmmu_acc"]
        old_pred, new_pred = block.get("parsed_pred"), out["parsed_pred"]
        old_correct += int(old_pred == block["answer"])
        new_correct += int(new_pred == out["answer"])
        noparse += int(new_pred == mcq.NO_ANSWER)
        if old_pred != new_pred:
            if new_pred == mcq.NO_ANSWER:
                to_noanswer += 1
            else:
                drift += 1

        block["parsed_pred"] = new_pred
        aggregate_input.append(out)

    n = len(aggregate_input)
    # Use the task's own aggregation so the written metric matches what a live
    # run would produce. It prints a per-subject table; keep that out of the way.
    with contextlib.redirect_stdout(io.StringIO()):
        new_acc = U.mmmu_pro_aggregate_results(aggregate_input)

    # The number that is actually reported (and that results_table.py reads) is
    # the one in results.json, NOT a recomputation from the samples file: an
    # earlier MMMU-Pro parse fix rewrote results.json for the vision variants
    # without rewriting their samples, so the stored parsed_pred there is stale.
    reported = None
    if results_path.exists():
        blk = json.loads(results_path.read_text()).get("results", {}).get(task, {})
        v = blk.get("mmmu_acc,none")
        if isinstance(v, (int, float)):
            reported = float(v)

    summary = {
        "n": n,
        "reported": reported,
        "from_samples": old_correct / n if n else 0.0,
        "new_acc": new_acc,
        "to_noanswer": to_noanswer,
        "drift": drift,
        "noparse": noparse,
        "missing": missing,
        "ts": ts,
    }
    if dry_run:
        return summary

    shutil.copy2(samples_path, samples_path.with_suffix(samples_path.suffix + ".mmmu_bak"))
    with samples_path.open("w") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")

    if results_path.exists():
        results = json.loads(results_path.read_text())
        block = results.get("results", {}).get(task)
        if block is not None and "mmmu_acc,none" in block:
            shutil.copy2(results_path, results_path.with_suffix(".json.mmmu_bak"))
            block["mmmu_acc,none"] = new_acc
            block["rescore_note"] = f"mmmu_acc re-scored {reported if reported is not None else float('nan'):.5f} -> {new_acc:.5f} after the mmmu_mcq_utils random.choice fallback was replaced with NO_ANSWER ({noparse} unparseable of {n})"
            results_path.write_text(json.dumps(results, indent=2))
            summary["results_json"] = results_path.name
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="report old->new without writing")
    args = ap.parse_args()

    # No HF_HOME default: huggingface_hub already falls back to ~/.cache/huggingface, and
    # a default pointing into one person's scratch is a default that works for one person.
    # HF_HUB_OFFLINE stays -- this script must not reach the network to re-score.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    U, mcq = _load_task_utils()
    if getattr(mcq, "NO_ANSWER", None) is None:
        sys.exit("mmmu_mcq_utils has no NO_ANSWER -- the lmms-eval checkout predates the fix.")

    doc_cache: dict[str, dict] = {}
    files = sorted(p for task in TASK_CONFIG for p in LMMS_EVAL_DIR.glob(f"*/*/*_samples_{task}.jsonl"))
    print(f"Found {len(files)} cached MMMU-Pro run(s). Scorer: {LMMS_EVAL_SRC}/lmms_eval/tasks/mmmu_pro/utils.py\n")

    for f in files:
        task = f.name.split("_samples_")[-1][: -len(".jsonl")]
        config = TASK_CONFIG[task]
        if config not in doc_cache:
            doc_cache[config] = _load_docs(config)
        s = rescore_file(f, task, doc_cache[config], U, mcq, args.dry_run)
        run = f.relative_to(LMMS_EVAL_DIR).parts[0]
        rep = s["reported"]
        rep_s = f"{rep:.4f}" if rep is not None else "  n/a "
        moved = rep is None or abs(s["new_acc"] - rep) > 5e-5
        why = []
        if s["to_noanswer"]:
            why.append(f"fallback:{s['to_noanswer']}")
        if s["drift"]:
            why.append(f"parser-drift:{s['drift']}")
        print(
            f"{task:22s} {run[:40]:40s} {s['ts']} n={s['n']:5d} reported {rep_s} -> {s['new_acc']:.4f}"
            f"  [{', '.join(why) if why else 'no change'}]"
            f"  ({'DRY' if args.dry_run else s.get('results_json', 'no results.json')})"
            f"{'  <-- MOVED' if moved else ''}"
        )
        if s["missing"]:
            print(f"    WARNING: {s['missing']} sample(s) had ids absent from config {config!r} and were skipped")


if __name__ == "__main__":
    main()
