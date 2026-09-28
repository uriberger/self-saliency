"""Train a 4-class step classifier (plan / observe / deduce / none) for reasoning chains.

Architecture
------------
T5 encoder (flan-t5-small, pre-cached) + attention-mean-pooling + 2-layer MLP head.
Each step is classified individually, but the FULL reasoning chain is provided as
context so the model can resolve ambiguous steps.

Input:  "[STEP] {step_text} [CHAIN] {full_chain}"   (truncated to max_length tokens)
Output: one of {plan, observe, deduce}

System interface (training vs. inference)
-----------------------------------------
During training each (step_text, label, original_chain) triple is one example.
At inference the user passes only the raw reasoning chain; the classifier splits it
internally with extract_steps() (rule-based, no LLM) and classifies each fragment:

  python train_classifier.py --infer \\
      --chain "I will look at the image. The car is red. Therefore it approaches." \\
      --question "What color is the car?"

Output format (boundary info included):
  <plan>I will look at the image.</plan>
  <observe>The car is red.</observe>
  <deduce>Therefore it approaches.</deduce>

Step texts are output verbatim so the caller can locate every boundary by searching
the original chain.

Usage
-----
  # Training
  python steps_classifier/train_classifier.py
  python steps_classifier/train_classifier.py --model_name google/flan-t5-base
  python steps_classifier/train_classifier.py --no_chain_context   # ablation

  # Inference
  python steps_classifier/train_classifier.py --infer \\
      --chain "..." --question "..."
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5EncoderModel, get_linear_schedule_with_warmup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABELS = ("plan", "observe", "deduce", "none")
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

_DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "labeled_steps.jsonl"
_DEFAULT_CKPT = Path(__file__).resolve().parent / "checkpoints"

_STEP_SEP = "[STEP]"
_CHAIN_SEP = "[CHAIN]"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _build_input(step_text: str, chain: str, question: str, include_chain: bool) -> str:
    parts = [_STEP_SEP, step_text.strip()]
    if include_chain:
        ctx = f"Question: {question.strip()} Chain: {chain.strip()}" if question.strip() else chain.strip()
        parts += [_CHAIN_SEP, ctx]
    return " ".join(parts)


def load_samples(
    data_path: Path,
    include_chain: bool = True,
) -> list[dict]:
    """Load one record per (step_text, label) pair from the JSONL."""
    records: list[dict] = []
    with data_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("label") not in LABELS:
                continue
            records.append({
                "input_text": _build_input(
                    rec["step_text"],
                    rec.get("original_chain", ""),
                    rec.get("question", ""),
                    include_chain,
                ),
                "label": rec["label"],
                "label_id": LABEL2ID[rec["label"]],
                "chain_id": (rec["source_file"], str(rec["sample_id"])),
                "step_index": rec["step_index"],
            })
    return records


def split_by_chain(
    samples: list[dict], val_fraction: float = 0.15, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Hold out whole chains so no chain appears in both train and val.

    sorted(), not list(): set iteration order over tuples of str depends on the
    per-process hash seed, so `list({...})` gave a different split on every run and
    the held-out chains of an already-trained checkpoint could not be recovered.
    Sorting first makes the split a pure function of (chain ids, seed, val_fraction).
    """
    chains = sorted({s["chain_id"] for s in samples})
    random.Random(seed).shuffle(chains)
    n_val = max(1, int(len(chains) * val_fraction))
    val_chains = set(chains[:n_val])
    return (
        [s for s in samples if s["chain_id"] not in val_chains],
        [s for s in samples if s["chain_id"] in val_chains],
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StepDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        enc = self.tokenizer(
            s["input_text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(s["label_id"], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StepClassifier(nn.Module):
    """T5 encoder with attention-weighted mean pooling and an MLP classification head."""

    def __init__(self, encoder_name: str, num_labels: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(encoder_name)
        d = self.encoder.config.d_model
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(dropout),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d // 2, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (B, T, D)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (enc_out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # (B, D)
        return self.head(pooled)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path / "encoder")
        torch.save(self.head.state_dict(), path / "head.pt")

    @classmethod
    def load(cls, path: Path, num_labels: int = 4) -> "StepClassifier":
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        # apex may replace T5LayerNorm with FusedLayerNorm; if its CUDA extension is
        # broken (undefined symbol / version mismatch) loading the encoder crashes.
        # Patch transformers' T5LayerNorm with a plain-PyTorch RMS-norm fallback.
        import importlib, sys
        import transformers.models.t5.modeling_t5 as _t5_mod
        _apex_broken = False
        try:
            importlib.import_module("fused_layer_norm_cuda")
        except ImportError:
            _apex_broken = True
        _orig_t5_ln = _t5_mod.T5LayerNorm
        if _apex_broken:
            class _FallbackT5LN(torch.nn.Module):
                def __init__(self, hidden_size, eps=1e-6):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.ones(hidden_size))
                    self.variance_epsilon = eps
                def forward(self, x):
                    v = x.float().pow(2).mean(-1, keepdim=True)
                    x = x * torch.rsqrt(v + self.variance_epsilon)
                    if self.weight.dtype in (torch.float16, torch.bfloat16):
                        x = x.to(self.weight.dtype)
                    return self.weight * x
            _t5_mod.T5LayerNorm = _FallbackT5LN
        obj.encoder = T5EncoderModel.from_pretrained(path / "encoder")
        if _apex_broken:
            _t5_mod.T5LayerNorm = _orig_t5_ln
        d = obj.encoder.config.d_model
        obj.head = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(0.1),
            nn.Linear(d, d // 2), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(d // 2, num_labels),
        )
        obj.head.load_state_dict(torch.load(path / "head.pt", map_location="cpu"))
        return obj


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: StepClassifier,
    samples: list[dict],
    tokenizer,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 512,
) -> dict:
    """Fragment-level accuracy, per-class breakdown, and chain exact-match."""
    model.eval()
    ds = StepDataset(samples, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)

    all_preds: list[int] = []
    all_true: list[int] = []
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_true.extend(batch["label"].tolist())

    label_acc = sum(p == t for p, t in zip(all_preds, all_true)) / len(all_true)

    per_class: dict[str, dict] = {lbl: {"c": 0, "n": 0} for lbl in LABELS}
    for p, t in zip(all_preds, all_true):
        lbl = ID2LABEL[t]
        per_class[lbl]["n"] += 1
        if p == t:
            per_class[lbl]["c"] += 1

    by_chain: dict = defaultdict(lambda: {"pred": [], "true": []})
    for s, p, t in zip(samples, all_preds, all_true):
        by_chain[s["chain_id"]]["pred"].append((s["step_index"], p))
        by_chain[s["chain_id"]]["true"].append((s["step_index"], t))
    exact = sum(
        sorted(v["pred"]) == sorted(v["true"]) for v in by_chain.values()
    ) / len(by_chain)

    return {
        "label_acc": label_acc,
        "exact_match": exact,
        "per_class": {
            lbl: d["c"] / d["n"] if d["n"] else 0.0 for lbl, d in per_class.items()
        },
        "n_fragments": len(all_true),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    include_chain = not args.no_chain_context
    print(f"Chain context: {'enabled' if include_chain else 'disabled'}")

    print(f"Loading data from {args.data}")
    samples = load_samples(args.data, include_chain)
    if not samples:
        sys.exit(f"No valid samples in {args.data}. Run generate_data.py first.")

    counts = Counter(s["label"] for s in samples)
    n_chains = len({s["chain_id"] for s in samples})
    print(f"Fragments: {len(samples)}  Chains: {n_chains}  Labels: {dict(counts)}")

    train_samples, val_samples = split_by_chain(samples, args.val_fraction, args.seed)
    n_train_chains = len({s["chain_id"] for s in train_samples})
    n_val_chains = len({s["chain_id"] for s in val_samples})
    print(f"Train: {len(train_samples)} frags ({n_train_chains} chains)  "
          f"Val: {len(val_samples)} frags ({n_val_chains} chains)")

    # Persist the split next to the checkpoints: without it a later evaluation cannot
    # tell which chains this run held out, and any score it reports is contaminated.
    args.output.mkdir(parents=True, exist_ok=True)
    split_path = args.output / "val_chains.json"
    json.dump(
        {
            "data": str(args.data),
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "include_chain": include_chain,
            "labels": list(LABELS),
            "val_chains": sorted({s["chain_id"] for s in val_samples}),
        },
        open(split_path, "w"),
        indent=1,
    )
    print(f"Wrote held-out chain ids to {split_path}")

    # Inverse-frequency class weights
    total = len(train_samples)
    train_counts = Counter(s["label"] for s in train_samples)
    class_weights = torch.tensor(
        [total / (len(LABELS) * train_counts.get(lbl, 1)) for lbl in LABELS],
        dtype=torch.float32,
    ).to(device)
    print(f"Class weights: { {l: f'{w:.2f}' for l, w in zip(LABELS, class_weights.tolist())} }")

    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = StepClassifier(args.model_name, num_labels=len(LABELS)).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable:,} trainable")

    train_ds = StepDataset(train_samples, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.06 * total_steps), total_steps,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    args.output.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    print(f"\nTraining for {args.epochs} epochs...\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = loss_fn(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        metrics = evaluate(
            model, val_samples, tokenizer, device, args.batch_size, args.max_length
        )
        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"train_loss={avg_loss:.4f}  "
            f"val_label_acc={metrics['label_acc']:.3f}  "
            f"val_exact_match={metrics['exact_match']:.3f}"
        )
        for cls, acc in metrics["per_class"].items():
            print(f"  {cls:8s}: {acc:.1%}")

        if metrics["label_acc"] > best_val_acc:
            best_val_acc = metrics["label_acc"]
            model.save(args.output / "best")
            tokenizer.save_pretrained(args.output / "best" / "tokenizer")
            json.dump({"include_chain": include_chain}, open(args.output / "best" / "cfg.json", "w"))
            print(f"  → saved best (label_acc={metrics['label_acc']:.3f})")

    model.save(args.output / "final")
    tokenizer.save_pretrained(args.output / "final" / "tokenizer")
    json.dump({"include_chain": include_chain}, open(args.output / "final" / "cfg.json", "w"))
    print(f"\nBest val label_acc: {best_val_acc:.3f}")
    print(f"Saved final model to {args.output / 'final'}")


# ---------------------------------------------------------------------------
# Inference  (user passes raw chain; system splits and classifies)
# ---------------------------------------------------------------------------

def infer(args: argparse.Namespace) -> None:
    """Classify steps in a reasoning chain.  Output: labeled XML, one block per step.

    The step text in each XML block is verbatim, so the caller can recover the
    exact split boundary by searching for that text in the original chain.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from selfsal.steps.make_data import extract_steps  # noqa: PLC0415

    ckpt = Path(args.ckpt) if args.ckpt else args.output / "best"
    if not ckpt.exists():
        sys.exit(f"Checkpoint not found: {ckpt}. Train first.")

    cfg = json.load(open(ckpt / "cfg.json")) if (ckpt / "cfg.json").exists() else {}
    include_chain = cfg.get("include_chain", True)

    tokenizer = AutoTokenizer.from_pretrained(ckpt / "tokenizer")
    model = StepClassifier.load(ckpt).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    steps = extract_steps(args.chain)
    if not steps:
        print("No steps extracted from chain.", file=sys.stderr)
        return

    results: list[str] = []
    with torch.no_grad():
        for step in steps:
            inp = _build_input(step, args.chain, args.question, include_chain)
            enc = tokenizer(
                inp, return_tensors="pt", truncation=True, max_length=args.max_length
            )
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            label = ID2LABEL[logits.argmax(dim=-1).item()]
            results.append(f"<{label}>{step}</{label}>")

    print("\n".join(results))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="POD step classifier (train / infer)")
    # shared
    parser.add_argument("--data", type=Path, default=_DEFAULT_DATA)
    parser.add_argument(
        "--model_name", default="google/flan-t5-small",
        help="T5-family encoder; google/flan-t5-small and google/flan-t5-base are pre-cached",
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="Max tokens per example.  Steps are short; chain context fills the rest.",
    )
    parser.add_argument("--output", type=Path, default=_DEFAULT_CKPT)
    parser.add_argument("--seed", type=int, default=42)
    # training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument(
        "--no_chain_context", action="store_true",
        help="Classify steps without the full chain as context (ablation)",
    )
    # inference
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--chain", type=str, default="")
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")

    args = parser.parse_args()

    if args.infer:
        if not args.chain:
            sys.exit("--chain is required with --infer")
        infer(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
