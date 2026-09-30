# Re-score audit: the six unstamped cells

Run on branch `rescore/parser-stamp-audit`, then **applied**. The ten paper arms were
re-scored on the three benchmarks with a versioned answer reader; every one of their
cells now carries the pinned reader's version. The pre-re-score state is kept beside each
rewritten file as `*.orig_buggy_parse.bak` (18 files).

Non-paper runs in the archive were left alone. One of them was reported to move
38.00 -> 37.00 and lose an answer, which the re-score script itself flags as impossible
for a fix that only adds readings. That was the reason this was applied per-arm rather
than across the whole tree.

**It does not reproduce.** Replaying the pinned scorer (`2026-09-17-option-letter`) over
every banked MathVision run that still exists:

| tree | files | skipped | moved | LOST an answer |
|---|---|---|---|---|
| `V results/lmms_eval` (the non-paper tree) | 446 | 0 | 4 | **0** |
| `A results/vga_mini_sweep` (App D.1's window sweep) | 6 | 0 | 0 | **0** |

and of the 435 MathVision cells that carry a `*.orig_buggy_parse.bak`, **none** scores
lower than its backup. No cell anywhere reads 38.00 in the self-saliency tree; the 40 that
do in `vlm_reasoning` are all already stamped, and none of them regressed.

So the "impossible" case cannot be exhibited against the scorer as pinned. That is not the
same as explaining it: the likeliest reading is that the run it came from has since been
deleted, or that the dry run predates the pin and saw a scorer build that no longer
exists. Either way there is nothing on disk today that the current reader scores DOWN, and
the per-arm caution it motivated cost nothing.

Re-run the check with:

```bash
LMMS_EVAL_DIR=evaluation/lmms_eval python evaluation/rescore/rescore_mathvision.py
```

which is a dry run by default and prints `+gained/-lost` per task-entry, with `<-- LOST`
against any that regressed.

In a fresh clone it will find all ten arms and skip all ten with `no samples file beside
<stamp>_results.json`: re-scoring replays the model's stored answers, and those per-sample
`*_samples_*.jsonl` files are the ~5 GB release asset, not in git. That is the same asset
`tables.py --bootstrap` needs — see [publishing.md](publishing.md). Skipping loudly is the
intended behaviour; finding *nothing at all* was a bug, and was one until the four
re-score scripts were pointed at `evaluation/results` instead of the archive's
`results/lmms_eval`.

## What this was checking

A benchmark score is produced by code that reads the model's free-text answer and decides
whether it is right. That code has had bugs: on LogicVista it once found the "A" inside
the word "Answer" and scored nearly every answer as "A", giving results near 0% where the
fixed version gives 51–57%.

So a banked number is only meaningful together with the version of the answer reader that
produced it. The re-score scripts write that version into the result file. A normal
evaluation run does not. Six cells used in the paper therefore carry no version, and could
not be told apart from stale ones by inspection.

Re-scoring needs no GPU and no re-generation: the model's answers are stored in full, so
the current reader is simply replayed over them.

## Result: two cells move, both on LogicVista

| arm | benchmark | banked | re-scored | change |
|---|---|---|---|---|
| Coldstart | LogicVista | 54.91% | **55.13%** | +0.22 |
| No-Sal | LogicVista | 55.13% | **54.69%** | −0.45 |
| question boxes | LogicVista | 53.35% | 53.35% | — |
| SELF-SALIENCY_mean | LogicVista | 53.12% | 53.12% | — |
| SELF-SALIENCY_mean | WeMath | 73.39% | 73.39% | — |
| SELF-SALIENCY_mean | MathVision | 36.84% | 36.84% | — |

All ten paper arms were then re-scored on all three versioned benchmarks, not just the six
unstamped cells, so that a stamped-but-wrong cell could not hide. Nothing else moved:
MathVision 0 of 10 changed, WeMath 0 of 10, LogicVista 2 of 12.

## Effect on the paper

| | mean score | mean rank | wins |
|---|---|---|---|
| Coldstart | 62.69 → 62.70 | −0.04 | 0 |
| No-Sal | 63.47 → 63.45 | +0.04 | 3 |
| every other arm | unchanged | unchanged | unchanged |

SELF-SALIENCY is untouched: 64.26, best mean rank, 9 wins.

Nothing reorders. The ordering by mean score is identical in Table 2 and Table 5. In the
LogicVista column itself Coldstart and No-Sal swap (54.9 / 55.1 becomes 55.1 / 54.7), and
neither was that column's best — Saliency-R1's 56.5 is, before and after.

One apparent reordering in the mean-rank column is an artefact, not a result: after the
change Coldstart, VGA and EASE are all exactly 4.52, and a three-way tie sorts
arbitrarily.

The mean ranks above are deltas from a recomputation in this audit, which does not
reproduce the paper's absolute mean ranks exactly (it handles ties more crudely). The
deltas are computed the same way before and after, so they are comparable with each
other; the absolute values in the paper are not restated here.

## Provenance

No stamp was written by hand. The re-score script produced every one of them by actually
re-reading the stored answers, which is the only thing that makes a version label mean
anything.

## The underlying gap

The version stamp is written by the re-score scripts and by nothing else. A fresh
evaluation run produces no stamp at all. So the staleness check can confirm "this was
re-scored"; it cannot confirm "this is current", and every freshly-scored result looks
exactly like a stale one.

Fixing that belongs upstream in the lmms-eval fork -- the evaluator should stamp on the
way out, not only on the way back through a repair script.

## The gap is closed, and the pin moved to close it

`save_results_aggregated()` in the fork now writes `reasoning_parser_version` and
`mathvision_parser_version` into every `results.json` as it is written, beside `results`
and never inside it. A file with no stamp now means an old file, and a stamp that does
not match the scorer on disk means a stale one. Both were previously indistinguishable
from a correct fresh run.

**The pin moved, deliberately, and this is the record of it.**

| | |
|---|---|
| was | `a9a806b`, on the fork's `main` |
| now | `4ea4f15`, on the fork's `paper-pin` branch |
| between them | one commit: `feat(results): stamp the scorer's PARSER_VERSION at evaluation time` |

`paper-pin` is branched from `a9a806b` and **not** from the fork's `main`. `main` had
already moved one commit ahead, to a response-cache resume fix; basing the stamp there
would have pulled that into the pinned tree as well. Branching from the pin keeps the
promise this section is making: the tree this repository pins is the scoring code the
published numbers came from, plus a label.

**No number changes, and that is checked rather than asserted.** The commit adds
top-level keys to a dict and touches nothing under `results`;
`test/eval/test_parser_version_stamp.py` in the fork asserts exactly that, alongside each
stamp matching the `PARSER_VERSION` actually in its scorer's source. On this side, all 32
banked results files that use a tracked scorer already carried the correct stamp, written
by the re-score scripts, and `tables.py` flags none of them stale. So the change affects
runs made from now on and no banked number at all.

What it does not do is retrofit provenance. A `results.json` written before this commit
still carries a stamp only if a repair script put one there.

## None of this can be re-run from a clone

Every script in `evaluation/rescore/` reads the per-sample `*_samples_*.jsonl` files. Those
are about 5 GB, are not in git, and are not being released. In a fresh clone each script
finds the ten arms and skips all ten with `no samples file beside <stamp>_results.json`.
That is correct and loud.

So a re-score means **re-running the benchmark suite on GPUs first**. What comes out is
then a new measurement, not the banked one this audit is about -- the numbers here were
produced from the stored answers of the paper's own runs, and a fresh run does not
reproduce those answers, it replaces them. The same applies to Appendix B's error bars;
[reproduce.md](reproduce.md) says it from the other side.
