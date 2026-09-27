# Copyright 2026 NVIDIA. Apache-2.0.
"""Reasoning chain -> atomic steps -> the observe steps, with their token spans.

Appendix A.1: "We segment the reasoning chain at sentence-final punctuation marks and
line breaks, and discard fragments shorter than ten characters," then classify each
fragment and keep only those labelled `observe`. `segment_sentences` is that rule.

TWO SEGMENTERS, AND THE SEAM BETWEEN THEM
------------------------------------------
There are two, because the two experiments that need one run on different prompts:

  `segment_sentences`  the paper's method (Appendix A.1). The policy emits a freeform
                       <think>...</think> block with no markup, so the chain is split on
                       punctuation and newlines. This is what the GRPO reward uses --
                       every trained checkpoint in the paper was rewarded through it.

  `segment_tagged`     the head-selection screen (Section 3.5) prompts for explicit
                       <step>...</step> markup and segments on the tags instead, so
                       fragment boundaries are the model's own rather than a regex's.

The classifier, the label set and the `[STEP] ... [CHAIN] ...` input format are shared,
so an `observe` means the same thing on both paths. The BOUNDARIES do not: a regex
sentence and a tagged step are not guaranteed to be the same fragment, so the heads were
selected over one segmentation and are used under another.

This is a real seam in the method and it is written down here rather than left for a
reader to discover. It is also, so far as the evidence goes, a benign one -- the labels
come from the same model and the head ranking replicates across a second dataset
(`experiments/head_selection/cross_dataset.py`, footnote 3). But anyone extending the
head-selection screen should know which segmenter their numbers were produced under,
and `segment_sentences` is the one to prefer for new work, because it is the one the
reward runs.
"""

from __future__ import annotations

import re

#: Sentence-final punctuation or a line break (Appendix A.1).
_SENTENCE_SEP = re.compile(r"(?<=[.!?])\s+|\n+")

#: Fragments shorter than this are discarded (Appendix A.1).
MIN_FRAGMENT_CHARS = 10

_STEP_TAG = re.compile(r"<step>(.*?)</step>", re.DOTALL)
_OBSERVE_TAG = re.compile(r"<observe>(.*?)</observe>", re.DOTALL)


def split_sentences_with_spans(text: str, base_offset: int = 0,
                               min_len: int = MIN_FRAGMENT_CHARS):
    """Appendix A.1's split, as (fragment, char_start, char_end) triples.

    Offsets are absolute in the original string (`base_offset` + local index) and refer
    to the STRIPPED fragment, so a caller can map a fragment back to the tokens that
    produced it without re-finding it by text -- which would go wrong the moment a chain
    repeats a sentence, and chains do repeat sentences.
    """
    spans = []
    pos = 0

    def emit(segment: str, start: int):
        stripped = segment.strip()
        if len(stripped) < min_len:
            return
        lead = len(segment) - len(segment.lstrip())
        s = start + lead
        spans.append((stripped, base_offset + s, base_offset + s + len(stripped)))

    for m in _SENTENCE_SEP.finditer(text):
        emit(text[pos:m.start()], pos)
        pos = m.end()
    emit(text[pos:], pos)
    return spans


def _char_to_token(encoding, case_id: int, char_idx: int, total_chars: int,
                   lookahead: int = 8):
    """`encoding.char_to_token`, walking forward over whitespace that maps to None."""
    for c in range(char_idx, min(char_idx + lookahead, total_chars)):
        tok = encoding.char_to_token(case_id, c)
        if tok is not None:
            return tok
    return None


def segment_sentences(output_text: str, think_start_char: int, think_end_char: int,
                      encoding, case_id: int, tok_lo: int, tok_hi: int,
                      question: str, classifier):
    """The paper's segmentation: observe steps of a freeform chain, as token spans.

    Returns `[(step_text, tok_start, tok_end), ...]`, half-open, in `encoding`'s
    tokenisation space and clamped to `[tok_lo, tok_hi + 1]` -- the think-token range
    whose attention rows were captured. Steps whose span is empty after clamping are
    dropped, as are sentences the classifier does not label `observe`.

    One batched classifier forward per completion, not one per sentence.
    """
    if think_start_char < 0 or think_end_char < think_start_char:
        return []
    think_text = output_text[think_start_char:think_end_char + 1]
    sentences = split_sentences_with_spans(think_text, base_offset=think_start_char)
    if not sentences:
        return []

    total_chars = len(output_text)
    labels = classifier.predict_many([s for s, _, _ in sentences], think_text, question)

    out = []
    for (text, cs, ce), label in zip(sentences, labels):
        if label != "observe":
            continue
        tok_a = _char_to_token(encoding, case_id, cs, total_chars)
        tok_b = encoding.char_to_token(case_id, ce - 1)
        if tok_b is None:
            tok_b = _char_to_token(encoding, case_id, ce - 1, total_chars)
        if tok_a is None or tok_b is None:
            continue
        tok_a = max(tok_a, tok_lo)
        tok_b = min(tok_b + 1, tok_hi + 1)
        if tok_b > tok_a:
            out.append((text, tok_a, tok_b))
    return out


def segment_tagged(tokens: list[str], classifier, question: str = "",
                   tag: str = "step"):
    """The head-selection screen's segmentation: observe steps of a <step>-tagged chain.

    Operates on a list of token STRINGS rather than a tokenizer encoding, because the
    offline pipeline holds the decoded tokens, and returns spans in that same index
    space. `tag="observe"` reads explicit <observe> markup instead and skips the
    classifier entirely -- a prompt format from before the classifier existed, kept
    because stored runs use it.
    """
    full = "".join(tokens)

    starts, pos = [], 0
    for t in tokens:
        starts.append(pos)
        pos += len(t)
    total = pos

    def first_token(char_start: int) -> int:
        for i, s in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else total
            if end > char_start:
                return i
        return len(tokens)

    def end_token(char_end: int) -> int:
        for i, s in enumerate(starts):
            if s >= char_end:
                return i
        return len(tokens)

    pattern = _OBSERVE_TAG if tag == "observe" else _STEP_TAG
    matches = [(m.group(1).strip(), m.start(1), m.end(1))
               for m in pattern.finditer(full)]
    matches = [m for m in matches if m[0]]
    if not matches:
        return []

    if tag == "observe":
        keep = [True] * len(matches)
    else:
        labels = classifier.predict_many([t for t, _, _ in matches], full, question)
        keep = [l == "observe" for l in labels]

    out = []
    for (text, cs, ce), wanted in zip(matches, keep):
        if not wanted:
            continue
        a, b = first_token(cs), end_token(ce)
        if a < b:
            out.append((text, a, b))
    return out
