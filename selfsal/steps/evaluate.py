#!/usr/bin/env python
"""Evaluate the FLAN-T5 POD step classifier that gates the overlap reward.

The deployed checkpoint (checkpoint/steps_classifier/best, 4-class, include_chain=True)
has no recorded metrics: the only logged run in the vlm_reasoning repo is an earlier
3-class --no_chain_context model. This script measures the deployed one.

It scores through the exact inference path the reward uses — OverlapStepsClassifier.load()
plus predict_many() per chain — so the numbers describe the code that runs in training,
padding and truncation included, not a re-implementation of it.

READ THIS BEFORE QUOTING A NUMBER
---------------------------------
The training-time val split is **unrecoverable**. train_classifier.split_by_chain does
`chains = list({...})` and then shuffles: set iteration order over tuples of str depends
on per-process hash randomisation, so the 15% held out on 25 Jun cannot be reproduced,
not even with the same --seed. Verified: three processes, three different orderings.

So the default `--split val` here is a *fresh* deterministic split (chain ids sorted
before the shuffle, hence stable across processes) that overlaps the model's original
training chains by ~85% in expectation. Its accuracy is inflated by memorisation and is
NOT a held-out score. It is useful only as a sanity check that the checkpoint works and
as an upper bound.

To get a real number: make split_by_chain deterministic (`chains = sorted({...})`) in
vlm_reasoning/steps_classifier/train_classifier.py, retrain (~5 min on an H100), then
either rerun this with the same --seed/--val-fraction or pass the held-out chain ids via
--holdout-json. `--dump-split` writes the ids this script used, in that format.

Where to run it
---------------
A GPU node. The classifier is small, but the login node is heavily oversubscribed
(188 cores, load ~265) and a smoke test there measured ~32 s per fragment — the full
~4.8k-fragment split would take on the order of 40 h. On one GPU it is minutes.

Usage (fish)
------------
    conda activate saliency_r1_qwen3_vllm
    set -x HF_HOME /home/uberger/scratch/cache/hf_cache
    set -x HF_HUB_OFFLINE 1

    # smoke test, 20 chains
    python eval_steps_classifier.py --device cuda:0 --limit-chains 20

    # full re-derived split, with the per-source-model breakdown
    python eval_steps_classifier.py --device cuda:0 --by-source \
        --json outputs/steps_clf_eval/rederived_val.json

    # a genuinely held-out set, after a retrain that saved its split
    python eval_steps_classifier.py --device cuda:0 --holdout-json outputs/val_chains.json
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent

# The distilled fragment labels of Appendix A.1, written by `make_data.py`.
_DEFAULT_DATA = DATA_ROOT / "steps_classifier" / "labeled_steps.jsonl"


def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


OSTEPS = _load_module("_eval_overlap_steps", "trl/overlap_steps.py")
LABELS = OSTEPS.LABELS


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_chains(path: Path) -> dict:
    """Group the labelled fragments by chain, in step_index order.

    Mirrors train_classifier.load_samples: chain_id is (source_file, str(sample_id)),
    records whose label is outside the taxonomy are dropped.
    """
    chains: dict = collections.defaultdict(
        lambda: {"question": "", "chain": "", "steps": [], "labels": [], "idx": []}
    )
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("label") not in LABELS:
                continue
            cid = (rec["source_file"], str(rec["sample_id"]))
            c = chains[cid]
            c["question"] = rec.get("question", "")
            c["chain"] = rec.get("original_chain", "")
            c["steps"].append(rec["step_text"])
            c["labels"].append(rec["label"])
            c["idx"].append(rec["step_index"])
    for c in chains.values():
        order = sorted(range(len(c["idx"])), key=lambda i: c["idx"][i])
        c["steps"] = [c["steps"][i] for i in order]
        c["labels"] = [c["labels"][i] for i in order]
        c["idx"] = [c["idx"][i] for i in order]
    return dict(chains)


def derive_split(chain_ids, val_fraction: float, seed: int):
    """Deterministic chain-level split.

    Sorts before shuffling, unlike train_classifier.split_by_chain, so the result does
    not depend on PYTHONHASHSEED. This is what makes it a *different* split from the
    one the deployed checkpoint trained on — see the module docstring.
    """
    chains = sorted(chain_ids)
    random.Random(seed).shuffle(chains)
    n_val = max(1, int(len(chains) * val_fraction))
    return set(chains[n_val:]), set(chains[:n_val])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(true: list[str], pred: list[str], chain_of: list[tuple]) -> dict:
    """Fragment accuracy, per-class P/R/F1, macro-F1, confusion matrix, chain EM."""
    n = len(true)
    acc = sum(t == p for t, p in zip(true, pred)) / n if n else 0.0

    tp = collections.Counter()
    fp = collections.Counter()
    fn = collections.Counter()
    support = collections.Counter()
    conf = collections.Counter()
    for t, p in zip(true, pred):
        support[t] += 1
        conf[(t, p)] += 1
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    per_class = {}
    for lbl in LABELS:
        prec = tp[lbl] / (tp[lbl] + fp[lbl]) if tp[lbl] + fp[lbl] else 0.0
        rec = tp[lbl] / (tp[lbl] + fn[lbl]) if tp[lbl] + fn[lbl] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[lbl] = {"precision": prec, "recall": rec, "f1": f1, "support": support[lbl]}

    by_chain: dict = collections.defaultdict(lambda: [0, 0])
    for t, p, cid in zip(true, pred, chain_of):
        by_chain[cid][1] += 1
        if t == p:
            by_chain[cid][0] += 1
    exact = sum(c == tot for c, tot in by_chain.values()) / len(by_chain) if by_chain else 0.0

    # Macro-average over classes that actually occur: an absent class would otherwise
    # contribute a spurious 0 (visible on --limit-chains runs, where a label can be missing).
    present = [l for l in LABELS if support[l]]

    return {
        "n_fragments": n,
        "n_chains": len(by_chain),
        "accuracy": acc,
        "macro_f1": sum(per_class[l]["f1"] for l in present) / len(present) if present else 0.0,
        "macro_f1_classes": present,
        "per_class": per_class,
        "confusion": {f"{t}->{p}": c for (t, p), c in sorted(conf.items())},
        "chain_exact_match": exact,
    }


def print_report(m: dict, title: str) -> None:
    print(f"\n{title}")
    print(f"  fragments {m['n_fragments']}   chains {m['n_chains']}")
    print(f"  accuracy          {m['accuracy']:.4f}")
    covered = m.get("macro_f1_classes", LABELS)
    note = "" if len(covered) == len(LABELS) else f"   (over {', '.join(covered)} only)"
    print(f"  macro-F1          {m['macro_f1']:.4f}{note}")
    print(f"  chain exact-match {m['chain_exact_match']:.4f}")
    print(f"\n  {'label':10s} {'prec':>7s} {'rec':>7s} {'F1':>7s} {'support':>8s}")
    for lbl in LABELS:
        d = m["per_class"][lbl]
        print(f"  {lbl:10s} {d['precision']:7.4f} {d['recall']:7.4f} {d['f1']:7.4f} {d['support']:8d}")
    print(f"\n  confusion (rows = true, cols = predicted)")
    print(f"  {'':10s}" + "".join(f"{l:>9s}" for l in LABELS))
    for t in LABELS:
        row = "".join(f"{m['confusion'].get(f'{t}->{p}', 0):9d}" for p in LABELS)
        print(f"  {t:10s}{row}")


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the deployed POD step classifier")
    p.add_argument("--ckpt", default=str(REPO / "checkpoint/steps_classifier/best"))
    p.add_argument("--data", default=_DEFAULT_DATA)
    p.add_argument("--device", default=None, help="default: cuda if available else cpu")
    p.add_argument("--split", choices=("val", "train", "all"), default="val")
    p.add_argument("--seed", type=int, default=42, help="split seed (train_classifier default)")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--holdout-json", default=None,
                   help='held-out chains to evaluate on: either a JSON list of '
                        '[source_file, sample_id] pairs or a training run\'s '
                        'val_chains.json; overrides --split and suppresses the '
                        'contamination warning')
    p.add_argument("--dump-split", default=None, help="write the evaluated chain ids here")
    p.add_argument("--batch-size", type=int, default=32, help="fragments per encoder forward")
    p.add_argument("--limit-chains", type=int, default=0, help="0 = no limit (smoke tests)")
    p.add_argument("--by-source", action="store_true", help="also break results down per source model")
    p.add_argument("--json", default=None, help="write the full metrics dict here")
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"No labelled data at {data_path} (it lives in the vlm_reasoning repo).")

    t0 = time.time()
    chains = load_chains(data_path)
    print(f"{len(chains)} chains, {sum(len(c['steps']) for c in chains.values())} fragments "
          f"from {data_path} ({time.time() - t0:.1f}s)")

    if args.holdout_json:
        # Either a bare list of pairs, or the val_chains.json a training run writes
        # (a dict with the split metadata around a "val_chains" list).
        blob = json.loads(Path(args.holdout_json).read_text())
        pairs = blob["val_chains"] if isinstance(blob, dict) else blob
        keep = {tuple(map(str, x)) for x in pairs}
        missing = keep - set(chains)
        if missing:
            print(f"[warn] {len(missing)} held-out chain ids are not in the data file")
        keep &= set(chains)
        split_name = f"holdout ({Path(args.holdout_json).name})"
    else:
        train_ids, val_ids = derive_split(chains.keys(), args.val_fraction, args.seed)
        keep = {"val": val_ids, "train": train_ids, "all": set(chains)}[args.split]
        split_name = f"re-derived {args.split} (seed {args.seed}, val_fraction {args.val_fraction})"
        print(
            "\n" + "=" * 78
            + "\n[WARNING] This is NOT the split the checkpoint was held out on."
              "\n  train_classifier.split_by_chain shuffles a set, so its split depends on"
              "\n  PYTHONHASHSEED and cannot be reproduced. Expect ~85% of these chains to"
              "\n  be chains the model trained on: the score below is inflated by"
              "\n  memorisation and is an upper bound, not a held-out result.\n"
            + "=" * 78
        )

    keep = sorted(keep)
    if args.limit_chains:
        keep = keep[:args.limit_chains]
    if not keep:
        sys.exit("Empty evaluation set.")
    if args.dump_split:
        Path(args.dump_split).write_text(json.dumps([list(c) for c in keep], indent=1))
        print(f"wrote {len(keep)} chain ids to {args.dump_split}")

    print(f"\nloading classifier from {args.ckpt}", flush=True)
    clf = OSTEPS.OverlapStepsClassifier.load(args.ckpt, device=args.device)
    device = next(clf.parameters()).device
    print(f"  device {device}   include_chain {clf.include_chain}")

    true: list[str] = []
    pred: list[str] = []
    chain_of: list[tuple] = []
    source_of: list[str] = []
    t0 = time.time()
    for i, cid in enumerate(keep):
        c = chains[cid]
        for lo in range(0, len(c["steps"]), args.batch_size):
            batch = c["steps"][lo:lo + args.batch_size]
            pred += clf.predict_many(batch, c["chain"], c["question"])
        true += c["labels"]
        chain_of += [cid] * len(c["steps"])
        source_of += [cid[0]] * len(c["steps"])
        if (i + 1) % 50 == 0 or i + 1 == len(keep):
            el = time.time() - t0
            # flush: this normally runs redirected to a log, where stdout is block-buffered
            print(f"  {i + 1}/{len(keep)} chains  {len(pred)} fragments  "
                  f"{el:.0f}s  ({el / (i + 1):.2f}s/chain)", flush=True)

    metrics = compute_metrics(true, pred, chain_of)
    metrics["split"] = split_name
    metrics["ckpt"] = str(args.ckpt)
    metrics["include_chain"] = bool(clf.include_chain)
    print_report(metrics, f"RESULTS — {split_name}")

    if args.by_source:
        print("\n  per source model (accuracy / observe-F1 / n)")
        by_src: dict = collections.defaultdict(lambda: ([], [], []))
        for t, pr, cid, src in zip(true, pred, chain_of, source_of):
            by_src[src][0].append(t)
            by_src[src][1].append(pr)
            by_src[src][2].append(cid)
        per_source = {}
        for src, (t, pr, ci) in sorted(by_src.items()):
            ms = compute_metrics(t, pr, ci)
            per_source[src] = ms
            print(f"    {src.replace('-step_prompt.jsonl', ''):52s} "
                  f"{ms['accuracy']:.3f}  {ms['per_class']['observe']['f1']:.3f}  {ms['n_fragments']:6d}")
        metrics["per_source"] = per_source

    obs = metrics["per_class"]["observe"]
    print(f"\n  observe is the only class the overlap reward consumes: "
          f"precision {obs['precision']:.4f} (share of rewarded spans that are real "
          f"observations), recall {obs['recall']:.4f}.")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
