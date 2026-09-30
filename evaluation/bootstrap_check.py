"""Check the bootstrap error bars against the closed-form ones they replace.

The results table reads each score's error bar off a formula (see eval_stats),
on the argument that resampling the questions would land in the same place.
``results_table.py --bootstrap`` does the resampling instead; this script is how
that argument stops being an argument -- and stays checked as new benchmarks
arrive, since a wrongly chosen unit is the one error both methods would make
together.

Two things get checked, over the real runs banked under results/lmms_eval:

``--check-sampler``
    that the counts-based resampler in ``_bootstrap_std`` draws from the same
    distribution as the literal one that draws every unit individually. The fast
    one is what makes 10,000 resamples of a 23,609-question run take
    milliseconds, and it is only worth having if it is the same bootstrap. Both
    are first held against an exactly known answer -- for one group of binary
    units the bootstrap spread is sqrt(p(1-p)/n) with no simulation needed --
    and only then against each other on the grouped tasks, where nothing exact
    exists to compare to.

the default run
    every table cell and every model-vs-model gap, computed both ways, reported
    as the ratio bootstrap/closed-form. Two independent estimates of the same
    quantity, so the interesting number is how far from 1.0 the worst one lands.

Both are Monte Carlo, so they are seeded and the residual disagreement is
quoted, not hidden: at R resamples the bootstrap's own standard deviation is
about 1/sqrt(2R) of the bar itself -- 0.7% at R=10,000 -- and nothing here
should agree more closely than that.

    python scripts/bootstrap_check.py --check-sampler
    python scripts/bootstrap_check.py --resamples 10000
"""

import argparse
import itertools
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stats as eval_stats  # noqa: E402
import results_table  # noqa: E402


def _fmt_ratio(ratios: list[float]) -> str:
    worst = max(ratios, key=lambda r: abs(math.log(r)))
    return (f"n={len(ratios)}  median={statistics.median(ratios):.4f}  "
            f"worst={worst:.4f}  mean={statistics.fmean(ratios):.4f}")


def check_exact(resamples: int, seed: int, seeds: int) -> int:
    """Both resamplers against an answer that is known exactly.

    One group of n binary units, k of them 1: a bootstrap resample draws
    Binomial(n, k/n) ones, so the resampled score is that over n and the spread
    the bootstrap is estimating is exactly sqrt(p(1-p)/n). No simulation needed
    to know what the right answer is, which makes this a sharper test than
    comparing two noisy estimators to each other -- either one can be wrong
    without the comparison noticing, if they are wrong the same way.

    The sizes are the ones the real suite actually uses: MathVision's 304
    questions, MathVista's 1000, MME-RealWorld's 23,609.
    """
    print(f"exact check: {resamples} resamples x {seeds} seeds against "
          f"sqrt(p(1-p)/n)\n")
    print(f"{'n':>7} {'p':>6} {'exact':>10} {'counts':>10} {'sigma':>7} "
          f"{'per-unit':>10} {'sigma':>7}")
    worst = 0.0
    for n, k in ((304, 100), (1000, 308), (1500, 1040), (23609, 16010)):
        p = k / n
        exact = math.sqrt(p * (1 - p) / n)
        groups, values = ("",) * n, tuple([1.0] * k + [0.0] * (n - k))
        estimates = {}
        for name, fn in (("counts", eval_stats._bootstrap_std),
                         ("per-unit", eval_stats.bootstrap_units_std)):
            # The per-unit resampler is O(n) per resample; at 23,609 units it
            # would dominate the runtime of this check for no extra coverage.
            if name == "per-unit" and n > 1500:
                continue
            draws = [fn(groups, values, eval_stats.Bootstrap(resamples, seed + i))
                     for i in range(seeds)]
            mean = statistics.fmean(draws)
            sem = statistics.stdev(draws) / math.sqrt(seeds)
            estimates[name] = (mean, abs(mean - exact) / sem if sem else 0.0)
            worst = max(worst, estimates[name][1])
        cells = "".join(f" {mean:>10.7f} {sigma:>7.1f}"
                        for mean, sigma in estimates.values())
        print(f"{n:>7} {p:>6.3f} {exact:>10.7f}{cells}"
              + ("" if "per-unit" in estimates else f"{'-':>11}{'-':>8}"))
    print(f"\nfurthest from exact: {worst:.1f} sigma (tolerance 4.0): "
          f"{'OK' if worst <= 4 else 'FAILED'}")
    return 0 if worst <= 4 else 1


def check_sampler(runs: dict, resamples: int, seed: int, max_units: int,
                  seeds: int) -> int:
    """Counts-based resampling vs drawing every unit, on real runs.

    Restricted to the smaller runs: the literal version is O(units x resamples)
    and the whole point of the other one is that that gets expensive.

    Each estimate is itself noisy, and on the grouped tasks noticeably noisier
    than 1/sqrt(2R) -- MMStar splits 1500 questions across 18 groups, so each
    group's mean is resampled from ~83 units and the score is a sum of 18 such.
    Comparing one draw against one draw therefore fails on a ~2-sigma run for no
    reason. So both are averaged over ``seeds`` independent repeats, and the
    tolerance comes from the scatter actually observed across those repeats
    rather than from a formula that does not know the group structure.
    """
    seen, rows = set(), []
    print(f"sampler check: {resamples} resamples x {seeds} seeds, "
          f"runs up to {max_units} units\n")
    print(f"{'task':<34} {'units':>7} {'counts':>10} {'per-unit':>10} "
          f"{'ratio':>8} {'sigma':>7}")
    for cells in runs.values():
        for task, (run, _unit) in sorted(cells.items()):
            if task in seen or len(run.values) > max_units:
                continue
            seen.add(task)
            fast, slow = [], []
            for offset in range(seeds):
                config = eval_stats.Bootstrap(resamples, seed + offset)
                fast.append(eval_stats._bootstrap_std(run.groups, run.values, config))
                slow.append(eval_stats.bootstrap_units_std(run.groups, run.values,
                                                           config))
            if any(v is None for v in fast + slow):
                continue
            fast_mean, slow_mean = statistics.fmean(fast), statistics.fmean(slow)
            # Standard error on the *difference* of the two means, from the
            # repeats themselves: the yardstick the ratio has to be judged by.
            sem = math.sqrt(statistics.variance(fast) + statistics.variance(slow)) \
                / math.sqrt(seeds)
            sigma = abs(fast_mean - slow_mean) / sem if sem else 0.0
            rows.append((sigma, task, fast_mean, slow_mean))
            print(f"{task:<34} {len(run.values):>7} {fast_mean:>10.6f} "
                  f"{slow_mean:>10.6f} {fast_mean / slow_mean:>8.4f} {sigma:>7.1f}")

    if not rows:
        print("\nno runs small enough to check", file=sys.stderr)
        return 1
    ratios = [f / s for _sig, _t, f, s in rows]
    worst_sigma, worst_task, _f, _s = max(rows)
    print(f"\n{_fmt_ratio(ratios)}")
    print(f"furthest apart: {worst_task} at {worst_sigma:.1f} sigma "
          f"(tolerance 4.0): {'OK' if worst_sigma <= 4 else 'FAILED'}")
    return 0 if worst_sigma <= 4 else 1


def plugin_ratio(groups: tuple, values: tuple) -> float:
    """Where the bootstrap is *expected* to land relative to the closed form.

    Not 1.0, and not because either is wrong. The closed form divides by n-1
    (statistics.variance, the unbiased estimator of the population variance);
    the bootstrap resamples the observed units, whose variance is the plug-in
    one with n in the denominator. So each group contributes
    ``(n-1)/n * s^2/n`` rather than ``s^2/n``, and the ratio of the two square
    roots is what this returns.

    It matters only where groups are small. A 2500-question single-group task
    lands at 0.9998; MME splits 1187 images across 14 categories, four of them
    holding 20 images, and lands near 0.98 -- which is the bulk of the largest
    disagreement in this whole comparison.
    """
    buckets: dict = {}
    for group, value in zip(groups, values):
        buckets.setdefault(group, []).append(value)
    plugin = biased = 0.0
    for vals in buckets.values():
        term = statistics.variance(vals) / len(vals)
        biased += term
        plugin += term * (len(vals) - 1) / len(vals)
    return math.sqrt(plugin / biased) if biased else 1.0


def compare_bars(runs: dict, rows: list, resamples: int, seed: int) -> int:
    """Every cell and every gap, bootstrap vs closed form."""
    config = eval_stats.Bootstrap(resamples, seed)

    def both(groups, values):
        closed = eval_stats._closed_form_std(groups, values)
        boot = eval_stats._bootstrap_std(groups, values, config)
        return closed, boot

    started = time.time()
    cell_ratios, worst_cells = [], []
    for (model, notes), cells in runs.items():
        for task, (run, _unit) in cells.items():
            closed, boot = both(run.groups, run.values)
            if not closed or boot is None:
                continue
            cell_ratios.append(boot / closed)
            worst_cells.append((abs(boot / closed - 1), task, model, closed, boot,
                                len(run.values),
                                plugin_ratio(run.groups, run.values)))

    gap_ratios, worst_gaps = [], []
    for (model_a, notes_a), (model_b, notes_b) in itertools.combinations(rows, 2):
        for task, (run_a, _u) in runs.get((model_a, notes_a), {}).items():
            entry_b = runs.get((model_b, notes_b), {}).get(task)
            if entry_b is None:
                continue
            b_values = dict(zip(entry_b[0].units, entry_b[0].values))
            groups, deltas = [], []
            for group, unit, value in zip(run_a.groups, run_a.units, run_a.values):
                other = b_values.get(unit)
                if other is not None:
                    groups.append(group)
                    deltas.append(value - other)
            if len(deltas) < 2:
                continue
            closed, boot = both(tuple(groups), tuple(deltas))
            if not closed or boot is None:
                continue
            gap_ratios.append(boot / closed)
            worst_gaps.append((abs(boot / closed - 1), task,
                               f"{model_a} - {model_b}", closed, boot, len(deltas),
                               plugin_ratio(tuple(groups), tuple(deltas))))

    elapsed = time.time() - started
    noise = 1 / math.sqrt(2 * resamples)
    print(f"{resamples} resamples per bar, seed {seed}, {elapsed:.1f}s total")
    print(f"expected Monte Carlo noise on one bootstrap bar: +/-{noise:.2%}\n")

    for label, ratios, worst in (("table cells", cell_ratios, worst_cells),
                                 ("model-vs-model gaps", gap_ratios, worst_gaps)):
        if not ratios:
            continue
        print(f"{label}: bootstrap / closed form   {_fmt_ratio(ratios)}")
        print(f"  within +/-{noise:.2%}: "
              f"{sum(1 for r in ratios if abs(r - 1) <= noise) / len(ratios):.0%}"
              f"   within +/-1%: "
              f"{sum(1 for r in ratios if abs(r - 1) <= 0.01) / len(ratios):.0%}")
        print(f"  {'furthest apart':<22} {'closed':>9} {'boot':>9} {'ratio':>7} "
              f"{'expected':>9} {'units':>7}")
        for _d, task, who, closed, boot, n, expected in sorted(worst, reverse=True)[:5]:
            print(f"  {task:<22} {closed:>9.5f} {boot:>9.5f} {boot / closed:>7.4f} "
                  f"{expected:>9.4f} {n:>7}   {who[:44]}")
        print()

    ratios = cell_ratios + gap_ratios
    expected = [e for *_rest, e in worst_cells + worst_gaps]
    # A real discrepancy shows up as a systematic shift, not one noisy cell, so
    # the gate is on the median -- and on the median of what the n-vs-(n-1)
    # denominator alone predicts, which is not quite 1.0.
    median, median_expected = statistics.median(ratios), statistics.median(expected)
    ok = abs(median - median_expected) <= noise
    print(f"median ratio {median:.4f} vs {median_expected:.4f} predicted by the "
          f"n/(n-1) denominator alone")
    print(f"difference {abs(median - median_expected):.4f} vs noise floor "
          f"{noise:.4f}: {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resamples", type=int, default=eval_stats.BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=eval_stats.BOOTSTRAP_SEED)
    parser.add_argument("--check-sampler", action="store_true",
                        help="Compare the counts-based resampler against the "
                             "literal draw-every-unit one, instead of comparing "
                             "the bootstrap against the closed form.")
    parser.add_argument("--max-units", type=int, default=3000,
                        help="Largest run --check-sampler will use (default "
                             "%(default)s); the literal resampler is O(units).")
    parser.add_argument("--seeds", type=int, default=8,
                        help="Repeats per run for --check-sampler (default "
                             "%(default)s), averaged to see past Monte Carlo noise.")
    parser.add_argument("--results-root", type=Path, metavar="DIR",
                        help="Read runs from this results/ directory instead of "
                             "this tree's -- same reason as results_table.py's "
                             "flag of the same name.")
    args = parser.parse_args()

    if args.results_root:
        # No `/ "lmms_eval"`. The archive kept `<root>/inference` and `<root>/lmms_eval`
        # side by side; here the lmms-eval runs ARE `results/`, so the extra level made
        # this flag find nothing. Same shape as `tables.py`'s flag of the same name.
        results_table.LMMS_EVAL_DIR = args.results_root

    _data, runs = results_table.collect_lmms_eval()
    if not runs:
        print("no runs with per-question scores found", file=sys.stderr)
        return 1
    rows = sorted(runs.keys())

    if args.check_sampler:
        # The exact check pins both resamplers to a known answer on the shape
        # that covers most of the suite; the run-by-run one then extends that to
        # the grouped, many-valued cases (MME, MMStar) that have no closed form
        # to be pinned to.
        status = check_exact(args.resamples, args.seed, args.seeds)
        print()
        return check_sampler(runs, args.resamples, args.seed, args.max_units,
                             args.seeds) or status
    return compare_bars(runs, rows, args.resamples, args.seed)


if __name__ == "__main__":
    sys.exit(main())
