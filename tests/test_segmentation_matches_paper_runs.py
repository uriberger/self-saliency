# Copyright 2026 NVIDIA. Apache-2.0.
"""Appendix A.1's segmentation must split chains exactly as the paper's runs did.

Where the sentence boundaries fall decides which fragments get classified, which decides
which steps are `observe`, which decides what phi is averaged over. A boundary that
moves by one character can add or drop a whole step from `O` -- so this holds the
extracted splitter against the one the trained checkpoints were rewarded through, over
randomised chains covering the punctuation and line-break cases the rule names.

Skipped when the archive repo is absent; set SELFSAL_ARCHIVE to point at it.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

ARCHIVE = Path(os.environ.get(
    "SELFSAL_ARCHIVE", Path.home() / "scratch/research/saliency_r1"))

pytestmark = pytest.mark.skipif(
    not (ARCHIVE / "trl/overlap_steps.py").exists(),
    reason=f"archive repo not present at {ARCHIVE}")


def _archive_splitter():
    """Lift just the splitter out of the archive module.

    Importing it would pull in torch and transformers for a pure-string function.
    """
    src = (ARCHIVE / "trl/overlap_steps.py").read_text()
    start = src.index("def split_sentences_with_spans")
    end = src.index("def _char_to_tok")
    ns: dict = {}
    exec("import re\n"
         '_SENT_SEP = re.compile(r"(?<=[.!?])\\s+|\\n+")\n' + src[start:end], ns)
    return ns["split_sentences_with_spans"]


#: The separators the Appendix A.1 rule names, plus a bare space that must NOT split.
_ENDINGS = (". ", "! ", "? ", "\n", "\n\n", ". \n", " ", "?  ", "...  ")
_WORDS = ("the", "car", "is", "red", "image", "shows", "a", "left",
          "therefore", "short", "x", "traffic", "sign")


def test_sentence_spans_match_archive():
    from selfsal.steps import split_sentences_with_spans

    old = _archive_splitter()
    rng = random.Random(0)
    compared = 0

    for _ in range(2000):
        chain = "".join(
            " ".join(rng.choice(_WORDS) for _ in range(rng.randint(1, 12)))
            + rng.choice(_ENDINGS)
            for _ in range(rng.randint(0, 8)))
        for offset in (0, 17):
            want = old(chain, base_offset=offset)
            got = split_sentences_with_spans(chain, base_offset=offset)
            assert want == got, f"split differs at offset {offset}: {chain!r}"
            compared += 1

    assert compared == 4000


def test_short_fragments_are_dropped():
    """Appendix A.1: fragments under ten characters are discarded."""
    from selfsal.steps import MIN_FRAGMENT_CHARS, split_sentences_with_spans

    assert MIN_FRAGMENT_CHARS == 10
    spans = split_sentences_with_spans("Yes. The car on the left is red.")
    assert [t for t, _, _ in spans] == ["The car on the left is red."]


def test_spans_index_the_original_string():
    """Offsets must locate the fragment, not merely describe its length.

    A chain that repeats a sentence -- and trained chains do -- cannot be mapped back to
    its tokens by searching for the text.
    """
    from selfsal.steps import split_sentences_with_spans

    chain = "The car is red here. Something else entirely. The car is red here."
    spans = split_sentences_with_spans(chain)
    assert len(spans) == 3
    for text, start, end in spans:
        assert chain[start:end] == text
    assert spans[0][1] != spans[2][1]
