#!/usr/bin/env python
"""Ground each OBSERVE step of a stored completion, and say where those boxes sit.

    python sink_observe_boxes.py --scan-dir DIR --out DIR/observe_boxes.jsonl

THE THIRD LEG. The scan says where the model's attention went; the corpus says where the
human put the answer box. Neither says what the model was TALKING about. This grounds the
sentences it wrote -- Grounding-DINO on each observe step, exactly as
`overlap_rewards._dino_boxes` did during training and as `fig1_step_referent.py` scores
with -- so "the model reasons about the centre" becomes a measurement rather than a
reading of the transcript.

A SEPARATE PASS, not part of the scan, for three reasons. The completions are already
stored, so nothing has to be generated twice -- and generation was ~99% of the scan's
cost. A detector does not have to share a card with a 30B model. And the grounding can be
redone with a different threshold, or a different detector, without touching the
attention.

WHAT IT WRITES, per (picture, observe step): the step's text, its DINO boxes as
normalised [x0, y0, x1, y1], and where the union of those boxes falls on the picture's
own patch grid -- ring share, centre share, and the enrichment of each. Those are the
same quantities the attention tables report, computed on the same grids, so the three
legs are directly comparable:

    model attention   ring enrichment from the scan
    human box         ring enrichment of the corpus box
    model's referent  ring enrichment of the grounded observe step   <- this file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]   # the repository root
sys.path.insert(0, str(REPO))


def box_stats(boxes, gh, gw, max_box_area=0.5):
    """Where a set of normalised boxes lands on a gh x gw grid. -> dict or None.

    The union is rasterised at patch resolution -- a patch counts as covered when its
    CENTRE is inside a box -- which is the same convention the corpus box and the
    attention sets use, so the three numbers are on one footing. Boxes larger than
    `max_box_area` are dropped before the union, as the overlap reward dropped them:
    a detection covering half the picture grounds nothing.
    """
    from experiments.attention_bias import measure as SL

    keep = [b for b in boxes
            if max_box_area <= 0 or (b[2] - b[0]) * (b[3] - b[1]) <= max_box_area]
    if not keep:
        return None
    c = (np.arange(gw) + 0.5) / gw
    r = (np.arange(gh) + 0.5) / gh
    m = np.zeros((gh, gw), dtype=bool)
    for x0, y0, x1, y1 in keep:
        m |= ((c[None, :] >= x0) & (c[None, :] <= x1)
              & (r[:, None] >= y0) & (r[:, None] <= y1))
    if not m.any():
        # A box thinner than one patch still points somewhere; snap it to the patch its
        # centre lands in rather than discarding the step.
        x0, y0, x1, y1 = keep[0]
        m[min(gh - 1, max(0, int((y0 + y1) / 2 * gh))),
          min(gw - 1, max(0, int((x0 + x1) / 2 * gw)))] = True
    p = m.astype(float) / m.sum()
    ring = SL.ring_set(gh, gw)
    ra = ring.mean()
    return {"n_boxes": len(keep), "area": float(m.mean()),
            "ring_share": float(p[ring].sum()),
            "ring_enrich": float(p[ring].sum() / ra) if ra > 0 else float("nan"),
            "centre_enrich": (float((1 - p[ring].sum()) / (1 - ra))
                              if ra < 1 else float("nan"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--box-threshold", type=float, default=0.10)
    ap.add_argument("--max-box-area", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    # Imported here rather than at module scope: `probe` pulls in torch, and --help
    # should not.
    from experiments.attention_bias import measure as SL
    from experiments.attention_bias import probe as P
    from PIL import Image

    from selfsal.grounding import ground
    from selfsal.steps import StepClassifier

    # The detector, called directly. In the archive this went through the reward module's
    # module-level _CFG, reached either by importing it or by borrowing whichever copy the
    # probe had already loaded -- hence the `sys.modules.get(...) or ...` and the
    # configure()/._CFG fallback, which straddled two versions of that module. `ground`
    # takes the threshold as an argument, so there is no global to set and no version to
    # straddle. --max-box-area is not a grounding setting at all: it is applied below by
    # box_stats, per box, when the boxes are rasterised onto the grid.

    meta, arrays = P.read_stage(args.scan_dir, "scan")
    man = {r["key"]: r for r in P.read_manifest(args.scan_dir)}
    out_path = Path(args.out or (Path(args.scan_dir) / "observe_boxes.jsonl"))
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue
    todo = [m for m in meta if m["key"] not in done and m["key"] in man]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(meta)} scanned, {len(done)} already grounded, {len(todo)} to do",
          flush=True)

    clf = StepClassifier.load(device="cuda" if _cuda() else "cpu")
    proc, _tok = None, None
    n_steps = 0
    with open(out_path, "a") as fh:
        for i, m in enumerate(todo):
            comp = arrays.get(m["unit"], {}).get("completion")
            if comp is None:
                continue
            row = man[m["key"]]
            if proc is None:
                from transformers import AutoTokenizer
                proc = _Tok(AutoTokenizer.from_pretrained(
                    m.get("tokenizer") or _guess_tokenizer(args.scan_dir),
                    trust_remote_code=True))
            spans, diag = P.observe_spans(proc, [int(t) for t in comp],
                                          row["question"], clf)
            if not spans:
                fh.write(json.dumps({"key": m["key"], "n_steps": 0,
                                     "span_rule": diag.get("span")}) + "\n")
                continue
            text = proc.tokenizer.decode([int(t) for t in comp],
                                         skip_special_tokens=False)
            sents = [proc.tokenizer.decode([int(t) for t in comp[a:b]],
                                           skip_special_tokens=True).strip()
                     for a, b in spans]
            im = Image.open(row["path"]).convert("RGB")
            gh, gw = m["grid"]
            got = ground([im] * len(sents), sents,
                         box_threshold=args.box_threshold)
            recs = []
            for (a, b), s, bx in zip(spans, sents, got):
                st = box_stats([list(map(float, q)) for q in (bx or [])], gh, gw,
                               args.max_box_area)
                recs.append({"span": [int(a), int(b)], "text": s[:300],
                             "boxes": [[round(float(v), 4) for v in q]
                                       for q in (bx or [])],
                             "stats": st})
                n_steps += 1
            fh.write(json.dumps({"key": m["key"], "type": m["type"],
                                 "grid": [gh, gw], "span_rule": diag.get("span"),
                                 "n_steps": len(recs), "steps": recs}) + "\n")
            fh.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(todo)} pictures, {n_steps} observe steps grounded",
                      flush=True)
    print(f"\nwrote {out_path}")
    return 0


class _Tok:
    """`observe_spans` wants a processor; it only ever touches `.tokenizer`."""

    def __init__(self, tok):
        self.tokenizer = tok


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _guess_tokenizer(scan_dir):
    """The model a scan directory was produced by, from its own selftest log."""
    import re
    log = Path(scan_dir) / "logs" / "selftest.log"
    if log.exists():
        m = re.search(r"selftest\s+model=(\S+)", log.read_text())
        if m:
            return m.group(1)
    raise SystemExit(f"cannot tell which model produced {scan_dir}; pass a tokenizer")


if __name__ == "__main__":
    sys.exit(main())
