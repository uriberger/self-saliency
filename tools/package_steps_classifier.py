#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package the step classifier for a Hugging Face repository.

    python tools/package_steps_classifier.py --ckpt checkpoint/steps_classifier/best \
                                             --output /tmp/selfsal-steps-classifier

The classifier is needed by GRPO training and by every probe, and it is NOT derivable
from anything else in the repository: `selfsal/steps/make_data.py` distils its labels
from a model no public OpenAI endpoint serves, and from reasoning chains that are not
released either. So it has to be published. It is one of the TWO models that ship -- the
other is the merged flagship -- see docs/publishing.md section 5.

WHAT THIS DOES, AND THE ONE THING IT CHANGES. The encoder, tokenizer and cfg.json are
copied verbatim -- the encoder is a stock T5EncoderModel and loads with
`from_pretrained` on its own. `head.pt` is rewritten as `head.safetensors`, same tensors,
because a 1.2 MB `torch.load` of a file fetched from the internet executes whatever is
inside it and there is no reason to ask that of a reader. `StepClassifier.load` reads
either.

WHAT IT DOES NOT DO. It does not repackage this into an AutoModel. There is no stock
T5 sequence-classification head matching this one, so an `AutoModelForSequenceClassification`
path would mean shipping a custom modeling file behind `trust_remote_code` -- inventing a
mechanism to fake a standalone load that is not real. The card says plainly that loading
the head needs `selfsal`, and gives the six lines to rebuild it for anyone who would
rather not.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default="checkpoint/steps_classifier/best",
                    help="the trained checkpoint to package")
    ap.add_argument("--output", required=True, help="directory to write the HF repo into")
    ap.add_argument("--card", default=str(REPO_ROOT / "docs/model_cards/steps_classifier.md"),
                    help="model card to copy in as README.md")
    args = ap.parse_args()

    ckpt, out = Path(args.ckpt), Path(args.output)
    for required in ("encoder/config.json", "encoder/model.safetensors", "tokenizer", "cfg.json"):
        if not (ckpt / required).exists():
            raise SystemExit(f"{ckpt} has no {required} -- is this a step-classifier checkpoint?")
    head_pt = ckpt / "head.pt"
    if not head_pt.exists():
        raise SystemExit(f"{ckpt} has no head.pt")

    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ckpt / "encoder", out / "encoder", dirs_exist_ok=True)
    shutil.copytree(ckpt / "tokenizer", out / "tokenizer", dirs_exist_ok=True)

    # cfg.json carries `include_chain`, which is part of the input contract rather than a
    # tuning knob. Older checkpoints omit `labels`; write it in, because the head's output
    # index means nothing without the order.
    cfg = json.loads((ckpt / "cfg.json").read_text())
    cfg.setdefault("include_chain", True)
    cfg["labels"] = ["plan", "observe", "deduce", "none"]
    (out / "cfg.json").write_text(json.dumps(cfg, indent=2) + "\n")

    state = torch.load(head_pt, map_location="cpu")
    state = {k: v.contiguous() for k, v in state.items()}
    save_file(state, str(out / "head.safetensors"))

    card = Path(args.card)
    if card.exists():
        shutil.copyfile(card, out / "README.md")
    else:
        print(f"WARNING: no card at {card}; the repository will have no README")

    # Reload through the real loader, from the OUTPUT directory, and check it predicts.
    # Packaging that produces a directory nobody can load is the failure worth catching
    # here rather than after an upload.
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from selfsal.steps import StepClassifier

    clf = StepClassifier.load(out, device="cpu")
    label = clf.predict("There is a red car on the left.",
                        chain="Let me look at the image. There is a red car on the left.",
                        question="What is on the left?")
    print(f"\nwrote {out}")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out)}  ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"\nloaded back from {out} and predicted {label!r} on a sample observe step.")
    if label not in cfg["labels"]:
        raise SystemExit(f"loaded model returned {label!r}, which is not one of {cfg['labels']}")


if __name__ == "__main__":
    main()
