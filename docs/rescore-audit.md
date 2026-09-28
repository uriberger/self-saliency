# Re-score audit: the six unstamped cells

Run on branch `rescore/parser-stamp-audit`, then **applied**. The ten paper arms were
re-scored on the three benchmarks with a versioned answer reader; every one of their
cells now carries the pinned reader's version. The pre-re-score state is kept beside each
rewritten file as `*.orig_buggy_parse.bak` (18 files).

Non-paper runs in the archive were left alone. One of them moved 38.00 -> 37.00 and lost
an answer, which the re-score script itself flags as impossible for a fix that only adds
readings. That is unexplained, it is outside every paper arm, and it is the reason this
was applied per-arm rather than across the whole tree.

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
