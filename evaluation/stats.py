"""Per-question eval scores, and the error bars they support.

A results-table cell holds one number from one run -- 69.8% on MMStar, say. That
number still carries an error bar, because the benchmark's questions are
themselves a sample: a different draw of 1500 questions from the same pool would
have produced a slightly different score. Estimating that spread needs the
per-question results, which lmms-eval writes next to every results.json as
``<timestamp>_samples_<task>.jsonl``.

Everything here first reduces a run to one shape -- independent *units*, each
carrying a value, partitioned into *groups*::

    score = sum over groups of (mean of that group's unit values)

For most tasks there is a single group and one unit per question, so the score
is a plain mean and the formula above is an identity. Two headline benchmarks
are not plain means, and each is folded into this shape rather than being
approximated by one (see ``mme_run`` and ``nested_macro_run``):

* **MME** totals ``(accuracy + accuracy_plus) * 100`` over its categories, and
  asks two questions about each image, so the image is the unit.
* **MMStar** macro-averages twice -- over 6 categories, then over each one's 3
  sub-categories -- so questions in small sub-categories count for more. Unequal
  weights are folded into the values, which leaves the shape above intact.
* **HR-Bench** asks every question four times with its options cyclically
  rotated, so the question -- not the row -- is the unit (see ``hrbench_run``).
* **HallusionBench** puts no score in its samples rows at all -- a GPT judge
  makes the call at aggregation time -- so the values come from the judge's own
  output file instead (see ``hallusion_run``).

Once a run is in that shape, the two error bars this module exposes are:

``run_std``
    ``var = sum over groups of (variance of the unit values) / n_units``

``paired``
    the same formula applied to the per-unit difference ``a - b``

Either can be arrived at two ways, and they agree. The default is the formula
above, exact and instant. ``use_bootstrap()`` swaps in the resampling instead:
draw the units with replacement, rebuild the score, and take the spread of the
rebuilt scores (see "the bootstrap" below). The table reaches that through
``results_table.py --bootstrap``, which nothing needs to pass routinely.

The closed form is the bootstrap's answer without the bootstrap. Drawing units
with replacement and taking the spread of the resulting scores converges to
``sd/sqrt(n)``, and for the plain-mean tasks it agrees to the digit with the
``*_stderr_clt`` values lmms-eval reports where it emits them. (For MMStar it
does not, and shouldn't: lmms-eval's stderr there is the spread of a plain mean,
while the score it sits beside is the nested macro-average above.) Measured over
every banked run, the two methods put the median bar 0.55% apart -- less than
the bootstrap's own Monte Carlo wobble at 10,000 resamples. The one systematic
difference is the n vs n-1 denominator, visible only where groups are small:
MME's four 20-image categories move its bar by ~2%, from 39.6 to 38.2 on a
2360-point score. scripts/bootstrap_check.py is where those numbers come from.

``paired`` is for comparing two runs. Both saw identical questions, so the luck
of the question draw is shared and cancels: a question they both got right, or
both got wrong, contributes a zero difference and no uncertainty at all. Only
disagreements carry information about which run is better. That makes the error
bar on a *gap* much tighter than combining the two runs' separate bars, which
would wrongly assume they had been measured on unrelated question sets.

Neither error bar says anything about seeds or decoding nondeterminism. A single
run carries no information about run-to-run variance; the question these answer
is "would a different draw of questions have changed this?".

Parsing the samples files is the expensive part (several GB across every run
here), so the reduced values are cached under analysis_cache/item_scores, keyed
by the path, mtime and size of every file that went into them -- which for
HallusionBench means the judge's output as well as the samples file.
"""

import hashlib
import json
import math
import random
import statistics
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

CACHE_DIR = Path(__file__).parent.parent / "analysis_cache" / "item_scores"
# Bump when the extraction logic changes, so stale entries are recomputed.
CACHE_VERSION = 6

# How close a replayed score must land to the one lmms-eval reported before the
# run's error bar is trusted -- see load_run(). Loose enough to absorb reported
# scores that were rounded on the way out (mathvision rounds to 2 decimals of a
# percentage, i.e. 5e-5 here), tight enough to still reject a genuinely
# different aggregation: reading MMStar as a plain mean misses its reported
# score by 4.2e-3, forty times this bar.
SCORE_TOLERANCE = 1e-4

# A task reporting 0-100 while its samples rows score questions 0/1 is common
# enough to try both readings rather than dropping the run (mathvista).
SCALES = (1.0, 100.0)


@dataclass(frozen=True)
class Run:
    """One (model, task) evaluation, reduced to independent units.

    ``groups``, ``units`` and ``values`` are parallel: unit ``units[i]`` sits in
    group ``groups[i]`` and contributes ``values[i]``. ``score`` is the sum over
    groups of each group's mean value, which reproduces the number lmms-eval
    reported -- on the scale the table displays, since load_run() rescales the
    values to match.
    """

    task: str
    metric: str
    groups: tuple
    units: tuple
    values: tuple

    @property
    def score(self) -> float:
        return math.fsum(mean for mean, _n in _group_means(self.groups, self.values))


# --- extracting one question's score -----------------------------------------

# Keys holding a ready-made per-question score inside a samples-row metric entry.
# Which one is present depends on the task's process_results: most write "score",
# omnispatial writes "is_correct", mathvista's judge writes "true_false",
# hrbench's writes "gpt_score".
SCORE_KEYS = ("score", "is_correct", "true_false", "acc", "correct", "gpt_score")

# Tasks that log the prediction and the gold answer but leave the comparison to
# their aggregation function (mmerealworld, mmmu_pro). (prediction key, gold key)
PREDICTION_KEYS = (("pred_answer", "answer"), ("parsed_pred", "answer"))

# Tasks whose headline metric macro-averages over two nested levels instead of
# meaning over questions. Value is the field on the metric entry naming the
# inner level; the outer level is the sibling metric key on the same samples row
# (MMStar rows carry "average" plus the row's own coarse category). A run only
# gets used if it reproduces the reported score, so a task listed here that
# turns out to be a plain mean still gets the plain-mean treatment.
NESTED_MACRO_TASKS = {"mmstar": "l2_category", "mmstar_qwen": "l2_category"}


def item_value(entry) -> Optional[float]:
    """The 0/1 (or graded) score of a single question, from one metric entry of
    a samples row. None when the entry's shape isn't one we can read -- the
    caller then leaves that run without an error bar rather than guessing.
    """
    if isinstance(entry, (bool, int, float)):
        return float(entry)
    if isinstance(entry, dict):
        for key in SCORE_KEYS:
            value = entry.get(key)
            if isinstance(value, (bool, int, float)):
                return float(value)
        # mathvision logs one verdict per sampled response ("scores": [false])
        # instead of a single number; with one response per question that list's
        # mean is the question's score.
        verdicts = entry.get("scores")
        if (isinstance(verdicts, list) and verdicts
                and all(isinstance(v, (bool, int, float)) for v in verdicts)):
            return statistics.fmean(float(v) for v in verdicts)
        for pred_key, gold_key in PREDICTION_KEYS:
            if pred_key in entry and gold_key in entry:
                return float(entry[pred_key] == entry[gold_key])
    return None


def _unit(row: dict, index: int) -> str:
    return str(row.get("doc_id", index))


def plain_run(rows: list[dict], task: str, metric: str) -> Optional[Run]:
    """The common case: score = mean over questions, one group."""
    units, values = [], []
    for i, row in enumerate(rows):
        if metric not in row:
            return None
        value = item_value(row[metric])
        if value is None:
            return None
        units.append(_unit(row, i))
        values.append(value)
    return Run(task, metric, ("",) * len(units), tuple(units), tuple(values))


def _sibling_category(row: dict, metric: str) -> Optional[str]:
    """The one other scored metric key on a samples row -- MMStar's rows carry
    the overall "average" entry plus an entry named after the row's own coarse
    category, which is how the outer macro level is recovered."""
    siblings = [
        key for key, value in row.items()
        if key != metric and isinstance(value, dict) and item_value(value) is not None
    ]
    return siblings[0] if len(siblings) == 1 else None


def nested_macro_run(rows: list[dict], task: str, metric: str,
                     inner_field: str) -> Optional[Run]:
    """Score = mean over outer categories of the mean over their inner
    sub-categories of the mean over questions (MMStar).

    A question in a small sub-category counts for more than one in a large
    sub-category. Folding each question's weight -- 1 / (n_outer * n_inner_of_
    its_outer) -- into its value turns the weighted average back into a plain
    sum of group means over (outer, inner) groups, so the variance formula in
    this module applies unchanged.
    """
    grouped: dict = defaultdict(list)
    for i, row in enumerate(rows):
        entry = row.get(metric)
        if not isinstance(entry, dict):
            return None
        value = item_value(entry)
        outer = _sibling_category(row, metric)
        inner = entry.get(inner_field)
        if value is None or outer is None or inner is None:
            return None
        grouped[(outer, inner)].append((_unit(row, i), value))

    if not grouped:
        return None
    inner_counts: dict = defaultdict(int)
    for outer, _inner in grouped:
        inner_counts[outer] += 1
    n_outer = len(inner_counts)

    groups, units, values = [], [], []
    for (outer, inner), items in grouped.items():
        weight = 1.0 / (n_outer * inner_counts[outer])
        for unit, value in items:
            groups.append(f"{outer}/{inner}")
            units.append(unit)
            values.append(value * weight)
    return Run(task, metric, tuple(groups), tuple(units), tuple(values))


MME_METRICS = ("mme_perception_score", "mme_cognition_score")


def mme_run(rows: list[dict], task: str = "mme") -> Optional[Run]:
    """MME's total, reduced to per-image units.

    MME's headline number is not a mean. It sums, over each category,
    ``(accuracy + accuracy_plus) * 100`` -- where accuracy_plus is the fraction
    of *images* on which every question was answered correctly. Two questions
    are asked about each image, so questions are not independent and the image
    is the unit that may be resampled.

    With k questions on every image of a category, both halves are means over
    that category's images::

        100 * accuracy_c      = mean over images of 100 * (image's total)/k
        100 * accuracy_plus_c = mean over images of 100 * (image scored k of k)

    so a category contributes the mean of one per-image value and the total is a
    sum of group means. Perception and cognition partition the categories, so
    their two sub-scores are simply more groups in the same sum.

    Returns None unless every image in a category carries the same number of
    questions, which is what makes ``accuracy`` a mean over images rather than a
    ratio of two quantities that both move under resampling.
    """
    per_image: dict = defaultdict(list)
    for row in rows:
        for metric in MME_METRICS:
            entry = row.get(metric)
            if not isinstance(entry, dict):
                continue
            score = item_value(entry)
            if score is None:
                return None
            image = str(entry.get("question_id", "")).split("/")[-1]
            per_image[(entry.get("category"), image)].append(score)
    if not per_image:
        return None

    per_category: dict = defaultdict(list)
    for (category, image), scores in per_image.items():
        per_category[category].append((image, scores))

    groups, units, values = [], [], []
    for category, images in per_category.items():
        counts = {len(scores) for _image, scores in images}
        if len(counts) != 1:
            return None
        k = counts.pop()
        for image, scores in images:
            groups.append(str(category))
            units.append(f"{category}/{image}")
            values.append(100.0 * (sum(scores) / k + float(all(s == 1 for s in scores))))
    return Run(task, "mme_total", tuple(groups), tuple(units), tuple(values))


HRBENCH_TASKS = ("hrbench4k", "hrbench8k")
HRBENCH_CYCLES = 4


def hrbench_run(rows: list[dict], task: str, metric: str) -> Optional[Run]:
    """HR-Bench's average, reduced to per-question units.

    HR-Bench is scored CircularEval-style: each question is asked four times
    with its four options cyclically rotated, and the rows of one question sit
    at consecutive ``index`` values with ``cycle_category == index % 4``. Those
    four rows are one question seen four times, not four questions, so the
    question is the unit that may be resampled -- the same situation as MME's
    two questions per image.

    The reported metric macro-averages over the four cycle_categories. Each
    category holds exactly one rotation of every question, so that average is
    also the mean over questions of the question's own mean over its four
    rotations -- one group, one unit per question.

    Returns None unless every question has all four rotations, which is what
    makes the two readings equal. A run that fails that check is left without an
    error bar rather than falling back to a per-row one, which would treat four
    views of one question as four independent draws and understate the spread.
    """
    per_question: dict = defaultdict(list)
    for row in rows:
        entry = row.get(metric)
        if not isinstance(entry, dict):
            return None
        value = item_value(entry)
        index = entry.get("index")
        if value is None or not isinstance(index, int):
            return None
        per_question[index // HRBENCH_CYCLES].append((entry.get("cycle_category"), value))

    units, values = [], []
    for question, rotations in sorted(per_question.items()):
        if len({cycle for cycle, _value in rotations}) != HRBENCH_CYCLES:
            return None
        units.append(str(question))
        values.append(statistics.fmean(value for _cycle, value in rotations))
    if not units:
        return None
    return Run(task, metric, ("",) * len(units), tuple(units), tuple(values))


HALLUSION_TASKS = ("hallusion_bench_image",)
# The metric the table shows, and the only one that is a mean over questions:
# fAcc groups by figure and qAcc by question-pair, each with its own unit.
HALLUSION_METRIC = "aAcc"
# Where lmms-eval leaves the judge's verdicts, relative to the run directory
# that holds the samples file.
HALLUSION_SIDECARS = ("gpt_response/hallusion_output_vd_model.json",
                      "gpt_response/hallusion_output_vs_model.json")
HALLUSION_KEY_FIELDS = ("category", "subcategory", "set_id", "figure_id",
                        "question_id")


def hallusion_sidecars(samples_path: Path) -> list:
    """The judge-verdict files belonging to a HallusionBench samples file.

    They sit one level above the run directory --
    ``<model_slug>/gpt_response/`` against
    ``<model_slug>/<model_name>/<timestamp>_samples_<task>.jsonl`` -- because
    lmms-eval writes them per output_path, not per run.
    """
    return [samples_path.parent.parent / name for name in HALLUSION_SIDECARS]


def _hallusion_key(entry: dict) -> Optional[str]:
    if not all(field in entry for field in HALLUSION_KEY_FIELDS):
        return None
    return "/".join(str(entry[field]) for field in HALLUSION_KEY_FIELDS)


def hallusion_run(rows: list[dict], task: str, metric: str,
                  samples_path: Path) -> Optional[Run]:
    """HallusionBench's aAcc, from the verdicts the samples file doesn't hold.

    HallusionBench is the one benchmark here whose samples rows carry no score
    at all. Its process_results writes the question, the gold answer and the
    model's free text, and the right/wrong call is made later, at aggregation
    time, by a GPT judge -- so item_value() has nothing to read and the run
    would otherwise be dropped. (lmms-eval is in the same position: it reports
    ``aAcc_stderr_clt`` as "N/A".)

    The judge's answers do survive, in the gpt_response sidecar, as
    ``gpt4v_output_gpt_check`` in {0, 1, 2}. Turning those into 0/1 has one
    wrinkle, taken verbatim from lmms-eval's assign_correctness: a "2"
    (the judge could not tell) counts as *correct* for a VS question with
    figure_id 0, where there is no visual supplement and not knowing is the
    right answer, and as incorrect everywhere else.

    The sidecar is keyed by model slug rather than by run, so re-evaluating a
    model overwrites it with no record of which run it came from. The score
    gate in load_run() would catch a stale one, but only if the two runs
    happened to score differently; so this also checks that the sidecar holds
    exactly the questions the samples file does, with exactly the same model
    predictions, and declines the run if not. That is a direct test of "same
    run", not an inference from the score.

    Which is not a hypothetical. One banked model has two HallusionBench runs
    five seconds apart; the sidecar belongs to the later one, 83 of its 951
    predictions differ from the earlier one's -- and *both runs scored 55.7308*,
    so the score gate would have waved the earlier one straight through with a
    bar computed from the wrong run's answers. The later run still gets its bar,
    which is the one the table shows anyway.
    """
    if metric != HALLUSION_METRIC:
        return None

    verdicts: dict = {}
    for path in hallusion_sidecars(samples_path):
        if not path.exists():
            return None
        try:
            entries = json.loads(path.read_text())
        except ValueError:
            return None
        if not isinstance(entries, list):
            return None
        for entry in entries:
            key = _hallusion_key(entry)
            if key is None or key in verdicts:
                return None  # missing identity, or two entries claiming one
            verdicts[key] = entry

    units, values = [], []
    for i, row in enumerate(rows):
        sample = row.get(metric)
        if not isinstance(sample, dict):
            return None
        key = _hallusion_key(sample)
        judged = verdicts.get(key) if key else None
        if judged is None:
            return None
        if judged.get("model_prediction") != sample.get("model_prediction"):
            return None  # sidecar belongs to some other run of this model
        try:
            check = int(judged["gpt4v_output_gpt_check"])
            no_supplement = (sample["category"] == "VS"
                             and int(sample["figure_id"]) == 0)
        except (KeyError, TypeError, ValueError):
            return None
        units.append(key)
        values.append(float(check in ((1, 2) if no_supplement else (1,))))

    if len(units) != len(verdicts):
        return None  # the sidecar judged questions this run never asked
    return Run(task, metric, ("",) * len(units), tuple(units), tuple(values))


def candidate_runs(rows: list[dict], task: str, metric: str,
                   samples_path: Path) -> list:
    """Every reduction we know how to try for this task, best guess first. The
    caller keeps whichever one reproduces the reported score."""
    if task == "mme":
        return [r for r in (mme_run(rows, task),) if r is not None]
    if task in HALLUSION_TASKS:
        # No plain-mean fallback for the same reason as HR-Bench below: there is
        # nothing in the rows to take a mean of, so a fallback could only invent
        # one.
        return [r for r in (hallusion_run(rows, task, metric, samples_path),)
                if r is not None]
    if task in HRBENCH_TASKS:
        # No plain-mean fallback: with all four rotations present it reproduces
        # the score too, so the caller would accept it and quietly report an
        # error bar computed as if the rotations were independent questions.
        return [r for r in (hrbench_run(rows, task, metric),) if r is not None]
    candidates = []
    inner_field = NESTED_MACRO_TASKS.get(task)
    if inner_field:
        candidates.append(nested_macro_run(rows, task, metric, inner_field))
    candidates.append(plain_run(rows, task, metric))
    return [r for r in candidates if r is not None]


# --- loading a run -----------------------------------------------------------

def _stamp(path: Path) -> str:
    """A path's identity for cache purposes -- empty when it isn't there."""
    if not path.exists():
        return f"{path}|-"
    stat = path.stat()
    return f"{path}|{stat.st_mtime_ns}|{stat.st_size}"


def _cache_path(samples_path: Path, task: str, metric: str,
                expected: Optional[float]) -> Path:
    # HallusionBench reads a second file (see hallusion_run), so that file has
    # to be in the key too: re-running the judge changes the answer without
    # touching the samples file.
    extra = (hallusion_sidecars(samples_path) if task in HALLUSION_TASKS else [])
    key = (f"{CACHE_VERSION}|{_stamp(samples_path)}"
           + "".join(f"|{_stamp(p)}" for p in extra)
           + f"|{task}|{metric}|{'' if expected is None else format(expected, '.12g')}")
    return CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()}.json"


def _rescaled(run: Run, expected: Optional[float]) -> Optional[Run]:
    """The run with its values put on ``expected``'s scale, or None if no
    supported scale reproduces ``expected``."""
    if expected is None:
        return run
    score = run.score
    for scale in SCALES:
        if abs(score / scale - expected) <= SCORE_TOLERANCE * max(1.0, abs(expected)):
            if scale == 1.0:
                return run
            return replace(run, values=tuple(v / scale for v in run.values))
    return None


def _build(samples_path: Path, task: str, metric: str,
           expected: Optional[float]) -> Optional[Run]:
    with samples_path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        return None
    for candidate in candidate_runs(rows, task, metric, samples_path):
        rescaled = _rescaled(candidate, expected)
        if rescaled is not None:
            return rescaled
    return None


def load_run(samples_path: Path, task: str, metric: str,
             expected_score: Optional[float] = None) -> Optional[Run]:
    """Per-question scores for one (model, task), or None if unavailable.

    ``expected_score`` is the score the table displays for this run. It acts as
    a gate: unless replaying the samples file reproduces it, our reading of that
    file disagrees with the task's own aggregation, and the run is dropped
    rather than annotated with an error bar computed from the wrong quantity.
    Across ~30 tasks whose samples rows all look different, this is what keeps a
    silently-misread file from turning into a confident-looking number.
    """
    if not samples_path.exists():
        return None

    cache_file = _cache_path(samples_path, task, metric, expected_score)
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
        except ValueError:
            cached = None
        if cached is None:
            return None
        try:
            return Run(cached["task"], cached["metric"], tuple(cached["groups"]),
                       tuple(cached["units"]), tuple(cached["values"]))
        except (KeyError, TypeError):
            pass  # written by an older layout; fall through and rebuild

    run = _build(samples_path, task, metric, expected_score)
    payload = None if run is None else {
        "task": run.task, "metric": run.metric, "groups": list(run.groups),
        "units": list(run.units), "values": list(run.values),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(f".{id(run)}.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(cache_file)  # atomic: analysis_cache is shared between worktrees
    return run if run is not None and len(run.values) > 1 else None


# --- the error bars ----------------------------------------------------------

def _group_means(groups: tuple, values: tuple) -> list:
    """[(group mean, group size)] over the run's groups."""
    buckets: dict = defaultdict(list)
    for group, value in zip(groups, values):
        buckets[group].append(value)
    return [(statistics.fmean(vals), len(vals)) for vals in buckets.values()]


def _closed_form_std(groups: tuple, values: tuple) -> Optional[float]:
    """Standard deviation of ``sum over groups of group mean`` under resampling
    of the units -- the bootstrap std of the score itself.

    Each group's mean has variance ``s^2 / n``, and the groups hold disjoint
    units, so their variances add.
    """
    buckets: dict = defaultdict(list)
    for group, value in zip(groups, values):
        buckets[group].append(value)
    variance = 0.0
    for vals in buckets.values():
        if len(vals) < 2:
            return None
        variance += statistics.variance(vals) / len(vals)
    return math.sqrt(variance)


def _sum_of_means_std(groups: tuple, values: tuple) -> Optional[float]:
    """The error bar every public entry point here goes through -- closed-form
    by default, resampled when use_bootstrap() has been called."""
    if _BOOTSTRAP is None:
        return _closed_form_std(groups, values)
    return _bootstrap_std(groups, values, _BOOTSTRAP)


# --- the bootstrap -----------------------------------------------------------
#
# The closed form above is what the bootstrap converges to, so this buys no
# accuracy -- it is the same bar, measured rather than derived, for a reader who
# would rather see it that way. Draw the run's units with replacement, rebuild
# the score, repeat, report the spread of the rebuilt scores. It also stays
# useful as a check on the reduction itself: if a new task's unit were chosen
# wrongly, both methods would be wrong together, which is what
# scripts/bootstrap_check.py exists to catch.
#
# Two things the resampling must get right, and both fall out of the reduction
# above rather than needing new logic:
#
# * The unit is what gets drawn -- the image for MME, the question for HR-Bench.
#   Resampling rows there would quietly assume independence the data does not
#   have, which is the same error the closed form would make on the same input.
# * Groups are resampled independently, each to its own original size. The score
#   is a sum of group means, and a category's mean is estimated only from that
#   category's units; pooling the draw across groups would let one category's
#   size wander and is not the sampling the score's error bar is about.
#
# Drawing n units one at a time is the textbook description but not the cheapest
# implementation, and at 23,609 units times 10,000 resamples the difference is
# minutes against milliseconds. A group's resampled mean depends only on *how
# many times* each distinct value was drawn, and those counts are multinomial
# over the distinct values -- two of them for a 0/1 task, three for MME. So each
# resample draws the counts directly, via the chain of conditional binomials
# that a multinomial factors into. This is the same distribution, not an
# approximation of it; scripts/bootstrap_check.py checks that against the
# literal draw-every-unit version.

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class Bootstrap:
    resamples: int
    seed: int


# Module-level, because the point of the switch is that every bar on a page is
# computed the same way; results_table.py sets it once from its --bootstrap flag.
_BOOTSTRAP: Optional[Bootstrap] = None


def use_bootstrap(resamples: int = BOOTSTRAP_RESAMPLES,
                  seed: int = BOOTSTRAP_SEED) -> None:
    """Estimate every error bar by resampling units instead of in closed form.

    Affects run_std() and paired() alike, since both go through
    _sum_of_means_std(). Call with resamples=0 to switch back.
    """
    global _BOOTSTRAP
    if resamples and resamples < 2:
        raise ValueError("a bootstrap needs at least 2 resamples to have a spread")
    _BOOTSTRAP = Bootstrap(resamples, seed) if resamples else None


def bootstrap_config() -> Optional[Bootstrap]:
    return _BOOTSTRAP


def _call_seed(seed: int, groups: tuple, values: tuple) -> int:
    """A seed derived from the data being resampled, so a cell's error bar is
    the same number whether the whole table was built or that one cell -- and
    doesn't shift when an unrelated model is added ahead of it in the loop."""
    digest = hashlib.blake2b(struct.pack(f"<q{len(values)}d", seed, *values),
                             digest_size=8)
    digest.update("\x00".join(groups).encode())
    return int.from_bytes(digest.digest(), "big")


def _bootstrap_std(groups: tuple, values: tuple,
                   config: Bootstrap) -> Optional[float]:
    """Spread of the score over ``config.resamples`` bootstrap resamples.

    None on the same input the closed form declines: a group of one unit has no
    spread to resample, and reporting zero uncertainty for it would be worse
    than reporting none.
    """
    buckets: dict = defaultdict(list)
    for group, value in zip(groups, values):
        buckets[group].append(value)

    tallies = []
    for vals in buckets.values():
        if len(vals) < 2:
            return None
        tallies.append((len(vals), tuple(Counter(vals).items())))

    binomial = random.Random(_call_seed(config.seed, groups, values)).binomialvariate
    scores = []
    for _ in range(config.resamples):
        score = 0.0
        for n, counts in tallies:
            total = 0.0
            remaining = n
            mass = 1.0  # probability left among the categories not yet drawn
            for value, count in counts[:-1]:
                if not remaining:
                    break
                p = count / n
                drawn = binomial(remaining, min(1.0, p / mass)) if mass > p else remaining
                total += drawn * value
                remaining -= drawn
                mass -= p
            # Whatever is left belongs to the last category by construction.
            score += (total + remaining * counts[-1][0]) / n
        scores.append(score)
    return statistics.stdev(scores)


def bootstrap_units_std(groups: tuple, values: tuple,
                        config: Bootstrap) -> Optional[float]:
    """The same bootstrap written the obvious way -- draw every unit, every
    time. Only used by scripts/bootstrap_check.py, to confirm that the counts
    shortcut in _bootstrap_std() samples the same distribution."""
    buckets: dict = defaultdict(list)
    for group, value in zip(groups, values):
        buckets[group].append(value)
    if any(len(vals) < 2 for vals in buckets.values()):
        return None

    rng = random.Random(_call_seed(config.seed, groups, values))
    grouped = list(buckets.values())
    scores = []
    for _ in range(config.resamples):
        scores.append(math.fsum(
            statistics.fmean(rng.choices(vals, k=len(vals))) for vals in grouped
        ))
    return statistics.stdev(scores)


def run_std(run: Optional[Run]) -> Optional[float]:
    """Error bar on one run's score, on the scale the score is displayed in.

    Answers "if the benchmark had drawn a different sample of questions, how far
    would this score move?" -- and nothing about seeds or decoding, which a
    single run cannot speak to.
    """
    if run is None:
        return None
    return _sum_of_means_std(run.groups, run.values)


@dataclass(frozen=True)
class Comparison:
    gap: float          # run_a.score - run_b.score
    std: float          # error bar on that gap, over the questions the runs share
    n_units: int        # questions (images, for MME) both runs answered
    n_differing: int    # of those, how many the two runs scored differently

    @property
    def z(self) -> Optional[float]:
        """Gap in error bars. Above ~2 the two runs are separated by more than
        question-sampling noise."""
        return None if self.std == 0 else self.gap / self.std


def paired(run_a: Optional[Run], run_b: Optional[Run]) -> Optional[Comparison]:
    """Compare two runs over the questions they both answered.

    The difference is taken per question and only then aggregated, so the shared
    luck of the question draw cancels instead of being counted twice: questions
    the two runs scored identically contribute nothing to the error bar.
    ``n_differing`` is how many questions actually carry the comparison.
    """
    if run_a is None or run_b is None:
        return None
    b_values = dict(zip(run_b.units, run_b.values))
    groups, deltas = [], []
    for group, unit, value in zip(run_a.groups, run_a.units, run_a.values):
        other = b_values.get(unit)
        if other is not None:
            groups.append(group)
            deltas.append(value - other)
    if len(deltas) < 2:
        return None

    groups, deltas = tuple(groups), tuple(deltas)
    std = _sum_of_means_std(groups, deltas)
    if std is None:
        return None
    gap = math.fsum(mean for mean, _n in _group_means(groups, deltas))
    return Comparison(gap, std, len(deltas), sum(1 for d in deltas if d != 0))
