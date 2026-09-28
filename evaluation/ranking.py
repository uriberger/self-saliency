"""Rank models by their average per-benchmark rank across the lmms-eval suite.

For every benchmark in BENCHMARKS the selected models are sorted by score
(higher is better) to give a per-benchmark rank; a model's final score is the
mean of its ranks, over the benchmarks it actually has a result for. Models
below --min-benchmarks are dropped *before* ranking, so a half-finished run
never shifts anybody else's ranks.

Which models take part is decided by the patterns in MODEL_RULES, not by a
hardcoded list: new GRPO overlap checkpoints are picked up automatically the
next time the script runs, as long as they follow the existing naming.

    python visualize/rank_models.py
    python visualize/rank_models.py --min-benchmarks 10 --csv ranking.csv
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tables import LMMS_EVAL_DIR, collect_lmms_eval_scores  # noqa: E402

# The benchmarks the ranking is computed over (lmms-eval task names).
BENCHMARKS = (
    "algopuzzlevqa",
    "chartqa",
    "dailyclue",
    "illusionvqa_soft_localization",
    "mathvision_testmini",
    "mathvista_testmini_cot",
    "mme",
    "mmerealworld",
    "mmmu_pro_standard",
    "mmstar",
    "omnispatial_test",
    "p3",
    "pope",
    "realworldqa",
    "scienceqa_img",
    "visulogic",
)

# A model directory in results/lmms_eval is <checkpoint slug><eval-config suffix>,
# e.g. ..._merged + _mnt4096_r1. The suffix records how the model was evaluated
# (max new tokens, R1 system prompt), not which checkpoint it is, so strip it
# before matching: the same checkpoint evaluated twice must land in one group.
EVAL_SUFFIX_RE = re.compile(r"(_mnt\d+|_r1sys\d+|_r1)+$")

# (display name, regex matched against the suffix-stripped slug). Patterns, not
# literal names, so that checkpoints added later are included automatically.
# First matching rule wins; a rule may match many slugs (then each becomes its
# own row, named after its slug).
MODEL_RULES = (
    ("Qwen3-VL-8B-Instruct", r"^qwen3_vl_8b_instruct$"),
    ("Cold-start SFT", r"^coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged$"),
    ("GRPO Saliency-R1", r"^grpo_.*_saliency_r1_qwen3_merged$"),
    ("GRPO accuracy-only", r"^grpo_qwen3_vl_8b_instruct_no_sal_merged$"),
    # Every GRPO overlap run whose checkpoint slug ends in _merged.
    ("GRPO overlap", r"^grpo_coldstart.*overlap.*_merged$"),
)


def strip_eval_suffix(slug: str) -> str:
    """`..._merged_mnt4096_r1` -> `..._merged` (the checkpoint identity)."""
    return EVAL_SUFFIX_RE.sub("", slug)


def match_rule(checkpoint: str) -> Optional[str]:
    """Display name of the first MODEL_RULES entry matching a checkpoint slug."""
    for label, pattern in MODEL_RULES:
        if re.match(pattern, checkpoint):
            return label
    return None


def select_models(scores: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Pick the model dirs to rank and key them by checkpoint slug.

    One checkpoint can have several eval-config dirs (``_mnt4096`` vs
    ``_mnt4096_r1``, ...). They are not interchangeable — the R1 system prompt
    tanks models that were not trained for it — so rather than merge them we
    keep the variant with the most results on BENCHMARKS and say which one that
    was. Ties go to the alphabetically first slug, to keep runs reproducible.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for slug in scores:
        checkpoint = strip_eval_suffix(slug)
        if match_rule(checkpoint):
            groups[checkpoint].append(slug)

    selected: dict[str, dict[str, float]] = {}
    for checkpoint, slugs in groups.items():
        best = max(
            sorted(slugs),
            key=lambda s: sum(1 for b in BENCHMARKS if b in scores[s]),
        )
        selected[checkpoint] = {b: scores[best][b] for b in BENCHMARKS if b in scores[best]}
        if len(slugs) > 1:
            others = ", ".join(s for s in sorted(slugs) if s != best)
            print(
                f"NOTE: {checkpoint} has several eval configs; using {best} "
                f"({len(selected[checkpoint])} benchmarks), ignoring: {others}",
                file=sys.stderr,
            )
    return selected


def rank_column(model_scores: dict[str, float]) -> dict[str, float]:
    """model -> rank (1 = best) for one benchmark. Ties share the average rank."""
    ordered = sorted(model_scores.items(), key=lambda kv: -kv[1])
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        shared = (i + j) / 2 + 1  # average of the 1-based positions i..j
        for model, _ in ordered[i:j + 1]:
            ranks[model] = shared
        i = j + 1
    return ranks


def compute_ranking(selected: dict[str, dict[str, float]], min_benchmarks: int):
    """Return (ranking rows, per-benchmark ranks, dropped models).

    Each ranking row is (checkpoint, mean rank, n benchmarks), best mean first.
    """
    kept = {m: s for m, s in selected.items() if len(s) >= min_benchmarks}
    dropped = {m: len(s) for m, s in selected.items() if m not in kept}

    per_benchmark: dict[str, dict[str, float]] = {}
    ranks: dict[str, list[float]] = defaultdict(list)
    for benchmark in BENCHMARKS:
        column = {m: s[benchmark] for m, s in kept.items() if benchmark in s}
        if not column:
            continue
        per_benchmark[benchmark] = rank_column(column)
        for model, rank in per_benchmark[benchmark].items():
            ranks[model].append(rank)

    rows = [
        (model, sum(rs) / len(rs), len(rs))
        for model, rs in ranks.items()
    ]
    rows.sort(key=lambda row: (row[1], -row[2], row[0]))
    return rows, per_benchmark, dropped


def print_ranking(rows, n_benchmarks, dropped, min_benchmarks) -> None:
    if dropped:
        print(
            f"Excluded (fewer than {min_benchmarks} of {len(BENCHMARKS)} benchmarks):",
            file=sys.stderr,
        )
        for model, n in sorted(dropped.items(), key=lambda kv: -kv[1]):
            print(f"  {n:2d}  {model}", file=sys.stderr)
        print(file=sys.stderr)

    if not rows:
        print("No model has enough results to rank.")
        return

    width = max(len(model) for model, _, _ in rows)
    print(f"Ranking over {n_benchmarks} benchmarks "
          f"({len(rows)} models, >= {min_benchmarks} results each)\n")
    print(f"{'#':>3}  {'model':<{width}}  {'mean rank':>9}  {'n':>3}")
    print(f"{'-' * 3}  {'-' * width}  {'-' * 9}  {'-' * 3}")
    for position, (model, mean_rank, n) in enumerate(rows, 1):
        print(f"{position:>3}  {model:<{width}}  {mean_rank:>9.2f}  {n:>3}")


def print_detail(rows, per_benchmark, selected) -> None:
    """Per-benchmark score and rank for every ranked model."""
    order = [model for model, _, _ in rows]
    width = max(len(m) for m in order)
    for benchmark in BENCHMARKS:
        ranks = per_benchmark.get(benchmark)
        print(f"\n{benchmark}")
        if not ranks:
            print("  (no results)")
            continue
        for model in sorted(ranks, key=lambda m: ranks[m]):
            print(f"  {ranks[model]:>5.1f}  {model:<{width}}  {selected[model][benchmark]:8.2f}")


def write_csv(path: Path, rows, per_benchmark, selected) -> None:
    import csv

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "mean_rank", "n_benchmarks"]
                        + [f"{b}_score" for b in BENCHMARKS]
                        + [f"{b}_rank" for b in BENCHMARKS])
        for model, mean_rank, n in rows:
            scores = [selected[model].get(b, "") for b in BENCHMARKS]
            ranks = [per_benchmark.get(b, {}).get(model, "") for b in BENCHMARKS]
            writer.writerow([model, f"{mean_rank:.4f}", n] + scores + ranks)
    print(f"\nWrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=LMMS_EVAL_DIR,
                        help="lmms-eval results root (default: %(default)s).")
    parser.add_argument("--min-benchmarks", type=int, default=13,
                        help="Drop models with fewer results than this (default: %(default)s).")
    parser.add_argument("--detail", action="store_true",
                        help="Also print the score and rank of every model per benchmark.")
    parser.add_argument("--csv", type=Path, help="Write the full table to this CSV file.")
    args = parser.parse_args()

    # collect_lmms_eval_scores() reads the module global, so point it at the
    # requested root (a worktree's results/ is not the central tree's).
    import results_table
    results_table.LMMS_EVAL_DIR = args.results_dir
    scores = collect_lmms_eval_scores()
    if not scores:
        sys.exit(f"No lmms-eval results found in {args.results_dir}")

    selected = select_models(scores)
    if not selected:
        sys.exit("No model directory matched MODEL_RULES.")

    rows, per_benchmark, dropped = compute_ranking(selected, args.min_benchmarks)
    print_ranking(rows, len(per_benchmark), dropped, args.min_benchmarks)
    if args.detail:
        print_detail(rows, per_benchmark, selected)
    if args.csv:
        write_csv(args.csv, rows, per_benchmark, selected)


if __name__ == "__main__":
    main()
