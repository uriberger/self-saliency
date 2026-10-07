---
license: apache-2.0
library_name: transformers
tags:
  - text-classification
  - reasoning
base_model: google/flan-t5-base
---

# SELF-SALIENCY — reasoning-step classifier

Labels one fragment of a reasoning chain as **plan**, **observe**, **deduce** or **none**
(Appendix A.1). It is what SELF-SALIENCY uses to find the *observe* steps — the sentences
that make a claim about the image — because those are the only steps the saliency reward
scores.

Without this model there is no saliency reward and no Section 5 analysis. The GRPO
trainer and all four attention probes load it.

## The input contract

**This is the part that matters.** The model was trained on one exact string shape, and
classifying a bare sentence instead is running a different model:

```
[STEP] <the step's text> [CHAIN] Question: <the question> Chain: <the full chain>
```

With no question, the `Question: ...` clause is dropped and the chain follows `[CHAIN]`
directly. `selfsal.steps.classifier.build_input()` is the one implementation of this;
both the training script and both segmenters call it.

`cfg.json` carries `include_chain: true`. That is part of the contract, not a tuning
knob — set it false and the chain context disappears from the input the model was fitted
on.

**Label order is `plan, observe, deduce, none`.** The head emits an index into that
tuple, so the order is load-bearing.

## Architecture

A **FLAN-T5 encoder** (`T5EncoderModel`, from `google/flan-t5-base`), masked mean-pooling
over the token states, then a small MLP head:

```python
nn.Sequential(
    nn.LayerNorm(d), nn.Dropout(0.1),
    nn.Linear(d, d // 2), nn.GELU(),
    nn.Dropout(0.1), nn.Linear(d // 2, 4),
)   # d = encoder.config.d_model
```

Pooling is `(hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)`.

## Files

| file | what |
|---|---|
| `encoder/` | the T5 encoder — a stock `T5EncoderModel`, loads with `from_pretrained` alone |
| `tokenizer/` | its tokenizer |
| `head.safetensors` | the MLP head's state dict, ~1.2 MB |
| `cfg.json` | `include_chain`, and the label order |

The head is safetensors rather than a `.pt` on purpose: loading a pickle fetched over the
network executes whatever is in it.

## Loading it

The encoder is standard, the head is not, so there is no `AutoModel` that returns both.
The supported route is this repository:

```bash
pip install git+https://github.com/uriberger/self-saliency
```

```python
from selfsal.steps import StepClassifier

clf = StepClassifier.load("path/to/this/repo")        # or set SELFSAL_STEPS_CKPT
clf.predict("There is a red car on the left.",
            chain="Let me look at the image. There is a red car on the left.",
            question="What is on the left?")          # -> 'observe'
```

If you would rather not take the dependency, the head is six lines — rebuild the
`nn.Sequential` above, `load_file("head.safetensors")` into it, and pool as shown.

## Evaluation

91.9% / 93% on the held-out split (Appendix A.1). Reproduce with:

```bash
python -m selfsal.steps.evaluate
```

## Training data

Distilled from a closed LLM over reasoning chains produced by the cold-start checkpoint;
`selfsal/steps/make_data.py` is the generator. Neither those chains nor the resulting
labels are released, which is why this model is published rather than left to be
regenerated.

## Citation

> *Look Where You Say You're Looking: Self-Grounded Attention for Visual Reasoning.*

Apache-2.0, matching the repository. The encoder derives from `google/flan-t5-base`
(Apache-2.0).
