# Copyright 2026 NVIDIA. Apache-2.0.
"""The GRPO corpus, and the 100-row holdout every arm was trained against (Section 4.1).

    from selfsal.data.saliency_r1_8k import load_corpus, train_holdout_split

    ds = train_holdout_split(load_corpus())["train"]      # 7,980 rows

Saliency-R1-8K is 8,080 rows; the paper holds out 100 of them, so every arm trains on
7,980. The split is `train_test_split(test_size=100, seed=42)`, which is not an arbitrary
choice and is the reason this is one function rather than a line in each caller:

  * it is SALIENCY-R1'S OWN split, from their published command (Appendix D.2). Our arms
    and the Appendix D.2 baseline therefore train on the same 7,980 rows, which is what
    makes that comparison isolate the RL stage.
  * 7,980 is what sets the step count. `steps = 7,980 * epochs / prompts_per_step`, and
    `training/grpo/config.py` asserts the result against the step each published
    checkpoint actually reached (3,990 for our arms, 2,991 for Saliency-R1). Change the
    size or the seed and every one of those assertions is describing a different run.
  * `datasets` derives the shuffle from the seed alone, so this is reproducible across
    processes -- unlike the step classifier's split, which is not; see
    `selfsal/steps/evaluate.py` for what that costs.

THE HOLDOUT IS NOT AN EVAL SET. Nothing in the paper reports a number on it: it exists so
the arms train on less than the whole corpus, in the same way the baseline's command does.
Table 3's 100 held-out validation samples are a different thing and a different corpus.

WHAT A CALLER GETS. `load_corpus` returns the rows unmodified -- no prompt, no resize.
Those belong to whoever is consuming them, and they differ: the trainer resizes bicubic
and wraps each row in `SYSTEM_PROMPT`, while the probes resize bilinear. See
`selfsal/data/prompt.py`.
"""

from __future__ import annotations

import os

#: The corpus the paper trains on. Appendix D.2's baseline uses it too.
DEFAULT_CORPUS = "peterant330/saliency-r1-8k"

#: Rows in the corpus, and the carve applied to it.
TOTAL_ROWS = 8080
HOLDOUT_SIZE = 100
HOLDOUT_SEED = 42

#: What is left to train on. `training/grpo/config.py` turns this into the step count.
TRAIN_ROWS = TOTAL_ROWS - HOLDOUT_SIZE


def _is_saved_to_disk(path: str) -> bool:
    """Whether `path` is a directory written by `Dataset.save_to_disk`.

    Worth detecting, because `load_dataset` does NOT refuse such a directory -- it falls
    back to the generic arrow builder, which ignores `dataset_info.json` and so hands back
    `image` as a raw {bytes, path} struct instead of a decoded PIL image, after copying
    the whole corpus into the HF cache. The failure is a column of the wrong type much
    later, not an error here.
    """
    return (os.path.isfile(os.path.join(path, "dataset_info.json"))
            or os.path.isfile(os.path.join(path, "dataset_dict.json")))


def load_corpus(dataset_name: str | None = None, split: str = "train"):
    """The corpus, from the Hub or from a `save_to_disk` directory. Rows unmodified.

    `datasets` is imported here rather than at module scope so the constants above can be
    read in an environment that has only numpy -- which is what the CPU tests and the
    config checks run in.
    """
    from datasets import load_dataset, load_from_disk

    name = dataset_name or DEFAULT_CORPUS
    if _is_saved_to_disk(name):
        ds = load_from_disk(name)
        # save_to_disk on a DatasetDict keeps the splits; take the requested one.
        if not hasattr(ds, "train_test_split"):
            ds = ds[split]
        return ds
    return load_dataset(name, split=split)


def train_holdout_split(dataset, size: int = HOLDOUT_SIZE, seed: int = HOLDOUT_SEED):
    """-> a DatasetDict with `train` (7,980) and `test` (100). Section 4.1's carve.

    Left exactly as it ran. `train` must stay byte-identical to what the published arms
    trained on, so neither argument should be touched to "tidy" anything -- see the module
    docstring for the three things that would move if it were.
    """
    return dataset.train_test_split(test_size=size, seed=seed)
