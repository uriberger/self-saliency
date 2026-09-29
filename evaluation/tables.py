"""Print a summary table of results and open an HTML version in the browser."""

import argparse
import html
import itertools
import json
import os
import re
import socket
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

import stats as eval_stats

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "inference"
LMMS_EVAL_DIR = Path(__file__).resolve().parent / "results"
# Where the finished page is written. Always this tree's own results/, even when
# --results-root points the *reading* somewhere else: a worktree that builds the
# table from the central tree's runs must not write its output back into it.
HTML_DIR = Path(__file__).resolve().parent / "results"
BENCHMARKS_PATH = Path(__file__).resolve().parent / "benchmark_meta.json"
LMMS_EVAL_BENCHMARKS_PATH = Path(__file__).resolve().parent / "benchmark_meta.json"
LMMS_EVAL_SUITE_PATH = Path(__file__).resolve().parent / "suite.yaml"
# Hand-entered scores for --paper-comparison benchmarks that can't be scored
# locally (see load_manual_scores).
MANUAL_SCORES_PATH = Path(__file__).resolve().parent / "paper_comparison_manual.json"

# --- Scorer staleness ---------------------------------------------------------
# A benchmark scores free text by extracting an answer from it. Change that
# extraction and every number already on disk was produced by code that no
# longer exists -- which is not hypothetical: on 2026-09-10 two LogicVista runs
# of the SAME checkpoint sat in this table at 0.45% and 56.2%, and nothing in
# either file said they had been scored by different builds.
#
# So a re-scored results.json carries a version stamp, and a cell whose stamp
# does not match the parser on disk is marked with STALE_MARK rather than
# printed as if it were current. Read the version out of the source text rather
# than importing it: these modules pull in math_verify and construct an OpenAI
# client at import time, and a table should not need either.
#
# Two scorers can move under a banked number, so each gets an entry below: the
# shared one behind every `*_reasoning` task, and MathVision's own, which is not
# built on it.
# The pinned submodule is the source of truth. It is checked out at the commit the
# paper's numbers were scored under (a9a806b), so the PARSER_VERSION strings below are
# not a hand-maintained copy of something elsewhere -- they are checkable, and
# tests/test_eval_scorers.py fails if they and the submodule ever disagree.
LMMS_EVAL_REPO = Path(os.environ.get(
    "LMMS_EVAL_DIR", Path(__file__).resolve().parent / "lmms_eval"))
STALE_MARK = "!"

# marker metric -> (source file, results.json stamp key, fallback version,
# re-score command). The marker is the metric name whose presence in a task's
# results identifies which scorer produced it. The fallback is used when the
# lmms-eval checkout is not on this machine; keep each in step with the
# PARSER_VERSION in the file above it.
PARSERS = {
    "acc_score,none": (
        LMMS_EVAL_REPO / "lmms_eval/tasks/_task_utils/reasoning_utils.py",
        "reasoning_parser_version",
        "2026-09-16-mcq-cue",
        "python evaluation/rescore/rescore_reasoning_tasks.py --apply",
    ),
    "mathvision_standard_eval,none": (
        LMMS_EVAL_REPO / "lmms_eval/tasks/mathvision/utils.py",
        "mathvision_parser_version",
        "2026-09-17-option-letter",
        "python evaluation/rescore/rescore_mathvision.py --apply",
    ),
}

# Models (as (model, notes) pairs) checked by default in the HTML view.
DEFAULT_MODELS = {
    ("coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged_mnt4096_r1", ""),
    ("grpo_coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged_overlap__wov0_2_2head_trmean_merged_mnt4096_r1", ""),
    ("grpo_coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged_saliency_r1_qwen3_merged_mnt4096_r1", ""),
    ("qwen3_vl_8b_instruct_mnt4096", ""),
}

# Checked by default in the --lmms-eval HTML view, which until now passed an
# EMPTY set -- so that page opened with no rows at all and every reader had to
# know which of ~40 run directories to tick before seeing anything. These two
# are a matched pair: identical weights and settings, injection off and on.
DEFAULT_LMMS_MODELS = {
    ("qwen3_vl_8b_instruct_mnt4096", ""),
    ("qwen3_vl_8b_instruct_mnt4096_vga_b0.2_l4-16", ""),
}

# Caveats rendered next to a run's name. lmms-eval rows have no `notes` of their
# own -- unlike the inference rows, which carry one in the filename -- so this is
# how a run that will be misread gets to say so in the table itself rather than
# only in the wiki.
MODEL_NOTES = {
    # The two rows below are the baselines for every column, and outside the
    # eight Saliency-R1 perception benchmarks they are the ONLY rows evaluated
    # without --r1-mode: qwen3_vl_8b_instruct_mnt4096_r1 was never run on
    # anything else. So a base-vs-trained delta here is partly a prompt-mode
    # delta, and nothing in the numbers says so.
    #
    # It bites wherever the benchmark's own prompt asks for a direct answer.
    # MathVerse vision-only pins query_type: query_wo ("please DIRECTLY answer
    # ... and provide the correct option letter"); the base model obeys and the
    # R1 system prompt stops the trained rows from obeying. Median output
    # tokens there are 2 vs 542, and 34.4% vs 51.0% is that gap -- the failures
    # are genuine wrong answers, not extraction misses. Same split on
    # algopuzzlevqa (4/3725), visulogic (2/698), mathvista_testmini_format
    # (3/270), mmerealworld (2/190), hrbench4k/8k (2/167), mme (2/164),
    # vstar_bench (4/151), cv_bench (2/138) and realworldqa (2/135).
    #
    # Where the prompt itself asks for reasoning the base model complies and
    # these rows are fine: mathvision_testmini ("solve the problem step by
    # step") is 4096 vs 1402, mmk12 1070 vs 972, logicvista_reasoning 638 vs
    # 497, wemath_testmini_reasoning 484 vs 501. Read the note as "check the
    # column", not "discard the row".
    #
    # Note the direction is the opposite of the _r1 rows below, which are broken
    # on exactly the columns these two are sound on. Fixing this needs a re-run
    # with --r1-mode, not a re-score. See wiki/saliency-r1-lmms-eval-replication.md.
    "qwen3_vl_8b_instruct_mnt4096":
        "plain instruct: answers in ~2 tokens where the benchmark asks it to -- "
        "reasoning columns are not comparable with the _r1 rows",
    "qwen3_vl_8b_instruct_mnt4096_vga_b0.2_l4-16":
        "VGA b=0.2 L4-16; control is qwen3_vl_8b_instruct_mnt4096. "
        "Plain instruct, same caveat as that control",
    # This row shows ScienceQA 12.7% and MME perception ~940. Neither is a
    # perception result: under --r1-mode these tasks score whether the model
    # closed its </think> block, and the STOCK _r1 run below is broken the same
    # way (ScienceQA 26.3% vs 94.4% plain). Read the plain-instruct rows.
    "qwen3_vl_8b_instruct_mnt4096_r1_vga_b0.2_l4-16":
        "r1-mode: scores </think>-closure, not accuracy -- do not read as perception",
    "qwen3_vl_8b_instruct_mnt4096_r1":
        "r1-mode: scores </think>-closure, not accuracy -- do not read as perception",
}

# Datasets checked by default in the HTML view.
DEFAULT_DATASETS = {
    "algopuzzlevqa",
    "chartqa",
    "cv_bench",
    "dailyclue",
    "hallusion_bench_image",
    "hrbench4k",
    "hrbench8k",
    "illusionvqa_soft_localization",
    "logicvista_reasoning",
    "mathverse_testmini_vision_only",
    "mathvision_testmini",
    "mathvista_testmini_cot",
    "mathvista_testmini_solution",
    "mme",
    "mmerealworld",
    "mmk12",
    "mmmu_pro_standard",
    "mmstar",
    "omnispatial_test",
    "p3",
    "pope",
    "realworldqa",
    "scienceqa_img",
    "visulogic",
    "vstar_bench",
    "wemath_testmini_reasoning",
}

# --- Benchmark content tags ---------------------------------------------------
# What kind of images / questions a benchmark is made of, so the table shows at a
# glance whether a score moved on photos, math figures, or abstract puzzles.
# tag -> (chip background, chip text colour, tooltip)
TAG_INFO = {
    "natural":       ("#d0ebff", "#1864ab", "Photographs of real-world scenes and objects"),
    "synthetic":     ("#e5dbff", "#5f3dc4", "Rendered/synthetic imagery (shapes, patterns, generated figures)"),
    "math":          ("#ffe3e3", "#c92a2a", "Math problems (geometry, algebra, plots) posed over a figure"),
    "science":       ("#d3f9d8", "#2b8a3e", "Exam-style science / multi-discipline subject questions"),
    "chart":         ("#fff3bf", "#e67700", "Charts, plots and other data graphics"),
    "puzzle":        ("#ffd8a8", "#d9480f", "Puzzles: algorithmic, riddle-style or abstract logic"),
    "spatial":       ("#c5f6fa", "#0b7285", "Spatial relations, layout and viewpoint reasoning"),
    "illusion":      ("#fcc2d7", "#a61e4d", "Optical illusions and impossible/counter-intuitive scenes"),
    "perception":    ("#e9ecef", "#495057", "Low-level perceptual judgements (colour/shape/orientation pop-out)"),
    "hallucination": ("#ffec99", "#8c6d1f", "Probes object hallucination (yes/no existence questions)"),
    "mixed":         ("#f1f3f5", "#5c5f66", "Deliberately mixed image and question types"),
}

# Ordered (regex on the task/dataset name, tags) rules -- first match wins, so
# variants (mathvista_testmini_cot, pope_adv, mmstar_qwen, ...) inherit the tags
# of their base benchmark. Anything unmatched is reported by _warn_untagged().
TAG_RULES = (
    (r"^algopuzzlevqa",       ("puzzle", "synthetic")),
    (r"^chartqa",             ("chart",)),
    (r"^cv_bench",            ("natural", "spatial")),
    (r"^dailyclue",           ("natural", "puzzle")),
    (r"^detective_bench",     ("natural", "puzzle")),
    (r"^hallusion",           ("illusion", "hallucination")),
    (r"^hrbench",             ("natural", "perception")),
    (r"^illusionvqa",         ("illusion",)),
    (r"^logicvista",          ("puzzle", "synthetic")),
    (r"^mathverse",           ("math",)),
    (r"^mathvision",          ("math",)),
    (r"^mathvista",           ("math", "chart")),
    (r"^mmerealworld",        ("natural",)),
    (r"^mme$|^mme_",          ("natural",)),
    (r"^mmk12$|^k12$",        ("science", "math")),
    (r"^mmmu_pro",            ("science", "mixed")),
    (r"^mmstar",              ("mixed",)),
    (r"^omni_?spatial",       ("natural", "spatial")),
    (r"^p3$|^salbench",       ("synthetic", "perception")),
    (r"^pope",                ("natural", "hallucination")),
    (r"^realworldqa",         ("natural", "spatial")),
    (r"^saliency_r1",         ("natural",)),
    (r"^scienceqa",           ("science",)),
    (r"^virl39k",             ("math", "science")),
    (r"^visual_cot",          ("natural",)),
    (r"^visulogic",           ("puzzle", "synthetic")),
    (r"^vstar_bench",         ("natural", "perception")),
    (r"^wemath",              ("math",)),
)


def dataset_tags(dataset: str) -> tuple[str, ...]:
    """Content tags for a task/dataset name ( () when we have no rule for it)."""
    for pattern, tags in TAG_RULES:
        if re.match(pattern, dataset):
            return tags
    return ()


def _warn_untagged(datasets: list[str]) -> None:
    missing = [ds for ds in datasets if not dataset_tags(ds)]
    if missing:
        print(
            f"NOTE: no content tag for: {', '.join(missing)} -- add a rule to "
            f"TAG_RULES in {Path(__file__).name} to tag them in the table.",
            file=sys.stderr,
        )


# Fields expected per dataset entry in dataset_benchmarks.json.
BENCHMARK_FIELDS = (
    "best_score", "best_model", "best_source",
    "similar_score", "similar_model", "similar_source",
)

# --- Saliency-R1 paper comparison (--paper-comparison) -----------------------
# The two models to compare against the paper, as
# (display name, lmms-eval model-dir slug in results/lmms_eval).
PAPER_MODELS = [
    ("Qwen2.5-VL-7B", "qwen2_5_vl_7b_instruct"),
    ("Saliency-R1-7B", "saliency_r1_7b"),
]

# Per-task, use the best-recipe Saliency-R1 variants instead of the bare
# saliency_r1_7b dir.  mnt2048 is preferred; mnt4096 is the fallback when
# mnt2048 has no score for a given task.
SALIENCY_R1_SLUG = "saliency_r1_7b"
SALIENCY_R1_CANDIDATES = [
    "saliency_r1_7b_mnt2048_r1sys3",
    "saliency_r1_7b_mnt4096_r1sys3",
]

# Table 2 of the Saliency-R1 paper (arXiv 2604.04500, §4.3 "Effective
# Performance Compared to the SOTA Model"), transcribed from the PDF. For each
# paper benchmark we list the candidate lmms-eval task name(s) that correspond
# to it in our runs; when several of our variants exist, the one whose
# Qwen2.5-VL-7B score is closest to the paper's is chosen (see the note below).
# "unit" is "%" for accuracy benchmarks or "score" for raw-scale MME
# (perception + cognition, matching the paper's ~2800-max column).
PAPER_TABLE2 = [
    dict(name="MMMU-Pro", unit="%",
         candidates=["mmmu_pro_standard", "mmmu_pro_vision", "mmmu_pro_vision_cot"],
         paper={"qwen2_5_vl_7b_instruct": 36.2, "saliency_r1_7b": 37.6}),
    dict(name="MMBench", unit="%", candidates=["mmbench_en_test"], manual=True,
         paper={"qwen2_5_vl_7b_instruct": 82.8, "saliency_r1_7b": 81.8}),
    dict(name="POPE", unit="%", candidates=["pope"],
         paper={"qwen2_5_vl_7b_instruct": 86.7, "saliency_r1_7b": 88.1}),
    dict(name="MME", unit="score", candidates=["mme"],
         paper={"qwen2_5_vl_7b_instruct": 2302, "saliency_r1_7b": 2385}),
    dict(name="MME-RW", unit="%", candidates=["mmerealworld"],
         paper={"qwen2_5_vl_7b_instruct": 58.7, "saliency_r1_7b": 62.9}),
    dict(name="MMStar", unit="%", candidates=["mmstar", "mmstar_qwen"],
         paper={"qwen2_5_vl_7b_instruct": 62.4, "saliency_r1_7b": 62.6}),
    dict(name="ChartQA", unit="%", candidates=["chartqa"],
         paper={"qwen2_5_vl_7b_instruct": 84.0, "saliency_r1_7b": 88.2}),
    dict(name="IllusionVQA", unit="%",
         candidates=["illusionvqa_soft_localization", "illusionvqa"],
         paper={"qwen2_5_vl_7b_instruct": 37.5, "saliency_r1_7b": 38.4}),
    dict(name="ScienceQA", unit="%", candidates=["scienceqa_img"],
         paper={"qwen2_5_vl_7b_instruct": 88.2, "saliency_r1_7b": 94.3}),
    dict(name="SalBench", unit="%", candidates=["p3"],
         paper={"qwen2_5_vl_7b_instruct": 49.1, "saliency_r1_7b": 63.7}),
]


def load_lmms_eval_suite() -> list[str]:
    """The 25 task names of Table 1, from evaluation/suite.yaml.

    One list, read by the table, by the run driver and by the tests -- so "the suite" is
    a file rather than three lists that have to agree. It used to be a bare text file of
    task names; the YAML carries the split and the reason for each choice alongside,
    which several of these need (mathverse_testmini_vision_only is not
    mathverse_testmini, and the difference is larger than anything the paper reports).
    """
    import yaml

    spec = yaml.safe_load(LMMS_EVAL_SUITE_PATH.read_text())
    return [b["task"] for b in spec["benchmarks"]]


def load_suite_protocol() -> dict:
    """The Appendix A.3 generation settings, from the same file."""
    import yaml

    return yaml.safe_load(LMMS_EVAL_SUITE_PATH.read_text())["protocol"]


def load_benchmarks(datasets: list[str], path: Path) -> dict[str, dict]:
    """Load per-dataset reference numbers, adding blank templates for any
    new dataset so the file can be filled in by hand. Existing values are
    preserved and the file is rewritten with entries for all known datasets.
    """
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text() or "{}")

    benchmarks = {}
    for ds in datasets:
        entry = existing.get(ds, {})
        benchmarks[ds] = {field: entry.get(field, None if field.endswith("_score") else "") for field in BENCHMARK_FIELDS}

    path.write_text(json.dumps(benchmarks, indent=2, sort_keys=True) + "\n")
    return benchmarks


def load_records(path: Path) -> list[dict]:
    text = path.read_text().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def accuracy(records: list[dict]) -> tuple:
    """Return (display string, accuracy ratio, answered ratio, std), the last
    three None when unavailable."""
    total = len(records)
    if total == 0:
        return "—", None, None, None
    correct = sum(1 for r in records if r.get("correct"))
    answered = sum(1 for r in records if r.get("final_answer", "") not in ("", None))
    acc_ratio = correct / total
    ans_ratio = answered / total
    std = eval_stats.run_std(inference_run(records)) if total > 1 else None
    return f"{acc_ratio:.1%} ({correct}/{total})", acc_ratio, ans_ratio, std


def inference_run(records: list[dict]) -> eval_stats.Run:
    """One run_experiment.py jsonl as per-question scores, so the inference
    table gets the same error bars as the lmms-eval one. sample_id is the unit,
    which is what lets two models be compared question by question."""
    units = tuple(str(r.get("sample_id", i)) for i, r in enumerate(records))
    values = tuple(1.0 if r.get("correct") else 0.0 for r in records)
    return eval_stats.Run("", "correct", ("",) * len(units), units, values)


def parse_filename(stem: str) -> tuple[str, str, str]:
    """Return (model, dataset, notes) from a result file stem."""
    parts = stem.split("-", 2)
    model = parts[0]
    dataset = parts[1] if len(parts) > 1 else ""
    notes = parts[2] if len(parts) > 2 else ""
    return model, dataset, notes


def collect_inference() -> tuple[dict, dict]:
    """Collect results from run_experiment.py jsonl files in results/inference.
    Returns (data, runs) in the same shape as collect_lmms_eval()."""
    data: dict[tuple, dict[str, tuple]] = defaultdict(dict)
    runs: dict[tuple, dict[str, tuple]] = defaultdict(dict)
    for path in sorted(RESULTS_DIR.glob("*.json*")):
        # Skip per-rank temporary files (e.g. ...rank0.jsonl) written by run_experiment.py.
        if re.search(r"\.rank\d+\b", path.name):
            continue
        model, dataset, notes = parse_filename(path.stem)
        records = load_records(path)
        data[(model, notes)][dataset] = accuracy(records)
        if len(records) > 1:
            runs[(model, notes)][dataset] = (inference_run(records), "%")
    return data, runs


PREFERRED_METRICS = (
    # wemath_strict first, ahead of acc_score: WeMath emits both, and they are
    # different quantities. acc_score is plain per-question accuracy, while the
    # paper's headline "Score (Strict)" credits a question only when the model
    # also gets its constituent knowledge steps right, so it runs much lower.
    # acc_score sorts first below and would silently take the column.
    "wemath_strict",
    "acc_score", "acc", "accuracy", "exact_match", "relaxed_overall", "average",
    # Tasks whose overall metric is named after the task itself and which also
    # emit per-sub-task metrics. Without an entry here the arbitrary fallthrough
    # below picks whichever sub-task happens to be first in the JSON -- for
    # omnispatial that silently reported the "motion_analysis" sub-task (346 of
    # 1533 samples) as if it were the benchmark score.
    "omnispatial",
    # dailyclue: same shape -- the overall metric sits alongside four category
    # metrics, and file order was surfacing "daily_commonsense" instead.
    "dailyclue_overall",
    # pope reports accuracy/f1/precision/recall/yes_ratio; accuracy is the
    # headline. It was already being picked, but only by dict order.
    "pope_accuracy",
    # HallusionBench reports three accuracies over the same 951 answers and none
    # of them is "the" score by dict order: aAcc is per-question, qAcc is per
    # question-pair (both must be right) and fAcc per figure, so qAcc/fAcc run
    # far lower. aAcc is the one comparable with every other column here.
    "aAcc",
)


def _primary_metric_entry(task_results: dict, task_name: str = "") -> Optional[tuple]:
    """Pick the main (metric name, score) from an lmms-eval per-task results dict
    (keys look like "acc_score,none", "exact_match,flexible-extract", ...).

    The name matters as much as the score: it is the key under which the same
    metric's per-question entry appears in the task's samples jsonl, which is how
    error bars are computed (see eval_stats).
    """
    metrics = {}
    for key, value in task_results.items():
        name = key.split(",")[0]
        if name == "alias" or "stderr" in name or not isinstance(value, (int, float)):
            continue
        metrics[name] = float(value)
    if not metrics:
        return None
    for preferred in PREFERRED_METRICS:
        if preferred in metrics:
            return preferred, metrics[preferred]
    if len(metrics) > 1:
        # Dict order is not a ranking. Guessing here is how a sub-task metric
        # ends up masquerading as the headline number -- say so instead.
        print(
            f"WARNING: no known primary metric for task {task_name or '?'}; "
            f"candidates: {sorted(metrics)}. Picking {next(iter(metrics))!r} by "
            f"file order -- add the right name to PREFERRED_METRICS.",
            file=sys.stderr,
        )
    return next(iter(metrics.items()))


def _primary_metric(task_results: dict, task_name: str = "") -> Optional[float]:
    """The main score from an lmms-eval per-task results dict."""
    entry = _primary_metric_entry(task_results, task_name)
    return None if entry is None else entry[1]


def is_mini_task(task: str) -> bool:
    """True for the 100-document *_mini tasks of the training-curve evals.

    saliency_r1's run_bench_eval.sh scores a run's intermediate checkpoints on
    subsampled copies of these benchmarks (its own task yamls: mmstar_mini,
    algopuzzlevqa_mini, ...). It banks progress per unit, and gives every unit
    its own `--tag r1_<unit>` so the units cannot overwrite each other's output
    dir -- which means one checkpoint produces ~15 directories
    (<slug>_mnt4096_r1_natural, <slug>_mnt4096_r1_algopuzzlevqa, ...). Since a
    directory here is one table row, that read as a separate *model* per
    benchmark, and 1,325 of the 1,353 dirs under results/lmms_eval are of this
    kind: they buried the ~28 real models.

    They are excluded rather than folded into their checkpoint's row because a
    100-document score is not comparable with the full-benchmark scores next to
    it in the same column. The curve they belong to is reported where it is
    comparable -- WandB, and saliency_r1's report_bench_evals.sh.
    """
    return task.endswith("_mini")


def _samples_path(results_path: Path, task: str) -> Path:
    """The per-question jsonl lmms-eval writes beside a results.json."""
    base = str(results_path)[: -len("_results.json")]
    return Path(f"{base}_samples_{task}.jsonl")


def expected_parser_versions() -> dict[str, str]:
    """marker metric -> the parser build a banked score must have come from."""
    versions = {}
    for marker, (source, _key, fallback, _cmd) in PARSERS.items():
        try:
            text = source.read_text()
        except OSError:
            versions[marker] = fallback
            continue
        match = re.search(r'^PARSER_VERSION\s*=\s*"([^"]+)"', text, re.M)
        versions[marker] = match.group(1) if match else fallback
    return versions


def _stale_score(task_results: dict, results: dict,
                 expected: dict[str, str]) -> Optional[str]:
    """The marker of the scorer whose stamp is missing or out of date, if any.

    Only the tasks scored by a parser in PARSERS can be stale: a task with its
    own untracked scorer is not stale just because it carries no stamp.
    """
    for marker, version in expected.items():
        if marker in task_results and results.get(PARSERS[marker][1]) != version:
            return marker
    return None


def _iter_lmms_eval_results():
    """Yield (model_slug, results_dict) for every real lmms-eval run file.

    The layout is results/lmms_eval/<model_slug>/<model_name_sanitized>/
    <timestamp>_results.json, but a sibling <model_slug>/submissions/ dir sits
    at the same depth and matches the same glob. Most tasks drop non-JSON there
    (mmbench writes an .xlsx), yet mathverse writes
    submissions/mathverse_testmini_results.json -- a bare *list* of per-question
    predictions, not a run summary. Filter those out by directory rather than by
    shape alone, so a future task's submission json cannot sneak in either.
    """
    for path in sorted(LMMS_EVAL_DIR.glob("*/*/*_results.json")):
        parts = path.relative_to(LMMS_EVAL_DIR).parts
        if parts[1] == "submissions":
            continue
        results = json.loads(path.read_text())
        if not isinstance(results, dict):
            continue
        yield parts[0], path, results


def collect_lmms_eval() -> tuple[dict, dict]:
    """Collect results from lmms-eval output dirs:
    results/lmms_eval/<model_slug>/<model_name_sanitized>/<timestamp>_results.json
    Files are processed in timestamp order, so the latest run wins per task.

    Returns (data, runs):
      data[(model, notes)][task] = (display, ratio, answered_ratio, std)
      runs[(model, notes)][task] = (Run, unit_label)

    ``std`` is on the same scale as ``ratio`` and is None when the task's
    per-question scores could not be read back (see eval_stats.load_run). The
    Runs are kept for the pairwise comparison table, which needs the individual
    question scores, not just their spread.

    The mini training-curve evals are skipped -- see is_mini_task().
    """
    data: dict[tuple, dict[str, tuple]] = defaultdict(dict)
    runs: dict[tuple, dict[str, tuple]] = defaultdict(dict)
    expected_parsers = expected_parser_versions()
    stale: dict[str, list[str]] = defaultdict(list)
    for model, path, results in _iter_lmms_eval_results():
        n_samples = results.get("n-samples", {})
        for task, task_results in results.get("results", {}).items():
            if is_mini_task(task):
                continue
            # Flagged, not dropped: the number is still what that run produced,
            # and hiding it would look like the run never happened.
            stale_marker = _stale_score(task_results, results, expected_parsers)
            is_stale = stale_marker is not None
            if is_stale:
                stale[stale_marker].append(f"{model} / {task}")
            # MME reports raw sub-scores (perception + cognition on a ~2800 scale),
            # not an accuracy — the generic 0-1 handling below turns the cognition
            # sub-score into a nonsensical "%" and count. Sum them and show the raw
            # paper-scale total instead.
            if task == "mme":
                perc = task_results.get("mme_perception_score,none")
                cog = task_results.get("mme_cognition_score,none")
                if isinstance(perc, (int, float)) and isinstance(cog, (int, float)):
                    total = perc + cog
                    run = eval_stats.load_run(_samples_path(path, task), task,
                                              "mme_total", expected_score=total)
                    data[(model, "")][task] = (f"{total:.1f}", total, None,
                                               eval_stats.run_std(run))
                    if run:
                        runs[(model, "")][task] = (run, "pts")
                continue
            entry = _primary_metric_entry(task_results, task)
            if entry is None:
                continue
            metric, score = entry
            ratio = score if score <= 1 else score / 100  # some tasks report 0-100
            n = (n_samples.get(task) or {}).get("effective")
            display = f"{ratio:.1%}" + (f" ({round(ratio * n)}/{n})" if n else "")
            if is_stale:
                display += f" {STALE_MARK}"
            # format_score (fraction of well-formatted answers) is the closest
            # analogue of the inference path's answered ratio.
            fmt = task_results.get("format_score,none")
            ans_ratio = float(fmt) if isinstance(fmt, (int, float)) else None
            # load_run puts the per-question values on ratio's scale, so its std
            # needs no further conversion.
            run = eval_stats.load_run(_samples_path(path, task), task, metric,
                                      expected_score=ratio)
            data[(model, "")][task] = (display, ratio, ans_ratio,
                                       eval_stats.run_std(run))
            if run:
                runs[(model, "")][task] = (run, "%")
    for marker, entries in stale.items():
        print(
            f"WARNING: {len(entries)} score(s) were not produced by parser "
            f"{expected_parsers[marker]!r} and are marked {STALE_MARK!r} in the "
            f"table. Re-score them with:\n"
            f"    {PARSERS[marker][3]}\n"
            + "".join(f"    {s}\n" for s in sorted(entries)[:10])
            + (f"    ... and {len(entries) - 10} more\n" if len(entries) > 10 else ""),
            file=sys.stderr,
        )
    return data, runs


def _paper_scale_score(task: str, task_results: dict) -> Optional[float]:
    """Return a task's score on the same scale the paper's Table 2 uses:
    accuracy as a 0-100 percentage, and MME as the raw perception + cognition
    sum (the paper reports MME on its native ~2800-point scale, not a percent).
    """
    if task == "mme":
        perc = task_results.get("mme_perception_score,none")
        cog = task_results.get("mme_cognition_score,none")
        if isinstance(perc, (int, float)) and isinstance(cog, (int, float)):
            return float(perc) + float(cog)
        return None
    val = _primary_metric(task_results, task)
    if val is None:
        return None
    return val * 100 if val <= 1 else val


def collect_lmms_eval_scores() -> dict[str, dict[str, float]]:
    """model_slug -> task -> score (paper scale). Latest run wins per task."""
    data: dict[str, dict[str, float]] = defaultdict(dict)
    for model, _path, results in _iter_lmms_eval_results():
        for task, task_results in results.get("results", {}).items():
            # Same exclusion as collect_lmms_eval: a mini variant of a Table-2
            # benchmark must not become the variant build_paper_columns picks.
            if is_mini_task(task):
                continue
            score = _paper_scale_score(task, task_results)
            if score is not None:
                data[model][task] = score
    return data


def load_manual_scores(path: Path) -> dict[str, dict[str, float]]:
    """Load hand-entered scores for Table-2 benchmarks flagged ``manual`` in
    PAPER_TABLE2 — benchmarks whose answers live only on the eval server and so
    cannot be scored locally by lmms-eval.

    MMBench's ``en_test`` split is the case in point: lmms-eval only writes a
    submission file (results/lmms_eval/<slug>/submissions/mmbench_en_test_results.xlsx),
    which you upload to https://mmbench.opencompass.org.cn to get the accuracy.
    Paste that accuracy into ``paper_comparison_manual.json`` (task -> model-slug
    -> score, on the paper scale: 0-100 for "%", raw for MME).

    A template with null values is (re)written for every ``manual`` benchmark so
    the file always lists exactly what needs filling. Returns
    {model_slug: {task: score}} for the entries that have been filled in.
    """
    existing = json.loads(path.read_text() or "{}") if path.exists() else {}

    template: dict[str, dict] = {}
    for bench in PAPER_TABLE2:
        if not bench.get("manual"):
            continue
        task = bench["candidates"][0]
        entry = existing.get(task, {})
        template[task] = {slug: entry.get(slug) for _, slug in PAPER_MODELS}

    path.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n")

    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for task, per_model in template.items():
        for slug, value in per_model.items():
            if isinstance(value, (int, float)):
                scores[slug][task] = float(value)
    return scores


def _report_missing_manual(manual: dict[str, dict[str, float]]) -> None:
    """Warn about ``manual`` benchmarks not yet filled for both models, pointing
    at the submission file to upload and where to paste the returned score."""
    for bench in PAPER_TABLE2:
        if not bench.get("manual"):
            continue
        task = bench["candidates"][0]
        missing = [slug for _, slug in PAPER_MODELS if task not in manual.get(slug, {})]
        if not missing:
            continue
        print(f"[{bench['name']}] no manual score for: {', '.join(missing)}")
        for slug in missing:
            xlsx = LMMS_EVAL_DIR / slug / "submissions" / f"{task}_results.xlsx"
            print(f"    upload {xlsx}")
        print(f"    to https://mmbench.opencompass.org.cn, then paste the accuracy into")
        print(f"    {MANUAL_SCORES_PATH} (key '{task}'), and re-run.\n")


def build_paper_columns(scores: dict[str, dict[str, float]]) -> list[dict]:
    """Select the Table-2 benchmarks we can compare on. A benchmark is kept
    only when at least one candidate task was evaluated for *both* paper models
    (the intersection). When several variants qualify, keep the one whose
    Qwen2.5-VL-7B score is closest to the paper's Qwen2.5-VL-7B score.
    Returns a list of {name, unit, task, paper} dicts in the paper's order.
    """
    qwen_slug = PAPER_MODELS[0][1]
    columns = []
    for bench in PAPER_TABLE2:
        both = [
            t for t in bench["candidates"]
            if all(t in scores.get(slug, {}) for _, slug in PAPER_MODELS)
        ]
        if not both:
            continue
        qwen_paper = bench["paper"][qwen_slug]
        task = min(both, key=lambda t: abs(scores[qwen_slug][t] - qwen_paper))
        columns.append(
            {"name": bench["name"], "unit": bench["unit"], "task": task,
             "paper": bench["paper"], "manual": bench.get("manual", False)}
        )
    return columns


def _fmt_paper_score(value: Optional[float], unit: str) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}%" if unit == "%" else f"{value:.0f}"


def _paper_closeness(delta: Optional[float], unit: str) -> Optional[float]:
    """Map |ours - paper| to a [0, 1] closeness for cell colouring (1 = exact)."""
    if delta is None:
        return None
    span = 20.0 if unit == "%" else 400.0  # points that count as "far off"
    return max(0.0, 1.0 - abs(delta) / span)


def build_paper_html(scores: dict, columns: list[dict], title: str) -> str:
    def _note_suffix(c: dict) -> str:
        if c.get("manual"):
            return " (server)"
        return " (raw)" if c["unit"] == "score" else ""

    def _tag_chips(task: str) -> str:
        tags = dataset_tags(task)
        if not tags:
            return ""
        chips = "".join(
            f'<span class="tag" style="background:{TAG_INFO[t][0]};color:{TAG_INFO[t][1]}"'
            f' title="{html.escape(TAG_INFO[t][2], quote=True)}">{t}</span>'
            for t in tags
        )
        return f'<div class="tags">{chips}</div>'

    header_cells = "".join(
        f'<th>{c["name"]}<br><span class="bench-note">{c["task"]}'
        f'{_note_suffix(c)}</span>{_tag_chips(c["task"])}</th>'
        for c in columns
    )

    body_rows = []
    for display, slug in PAPER_MODELS:
        cells = []
        for c in columns:
            ours = scores.get(slug, {}).get(c["task"])
            paper = c["paper"].get(slug)
            delta = (ours - paper) if (ours is not None and paper is not None) else None
            closeness = _paper_closeness(delta, c["unit"])
            style = f'style="background:{_ratio_to_color(closeness)}"' if closeness is not None else ""
            delta_html = (
                f'<div class="delta">Δ {delta:+.1f}</div>' if delta is not None else ""
            )
            cells.append(
                f'<td {style}>'
                f'<div class="ours">ours {_fmt_paper_score(ours, c["unit"])}</div>'
                f'<div class="paper">paper {_fmt_paper_score(paper, c["unit"])}</div>'
                f'{delta_html}</td>'
            )
        body_rows.append(f'<tr><td class="model">{display}</td>{"".join(cells)}</tr>')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; padding: 2rem; background: #f8f9fa; color: #212529; }}
  h1 {{ font-size: 1.2rem; margin-bottom: .4rem; color: #495057; }}
  p.caption {{ font-size: .85rem; color: #868e96; max-width: 60rem; margin: 0 0 1.2rem; }}
  table {{ border-collapse: collapse; background: white; box-shadow: 0 1px 4px rgba(0,0,0,.1); border-radius: 6px; overflow: hidden; }}
  th, td {{ padding: .55rem 1.1rem; text-align: center; border: 1px solid #dee2e6; font-size: .9rem; white-space: nowrap; }}
  th {{ background: #343a40; color: white; font-weight: 600; }}
  td.model {{ text-align: left; font-family: monospace; font-size: .85rem; background: #f1f3f5; }}
  span.bench-note {{ font-size: .72rem; color: #adb5bd; font-weight: 400; font-family: monospace; }}
  .ours {{ font-weight: 600; }}
  .paper {{ font-size: .8rem; color: #495057; }}
  .delta {{ font-size: .72rem; color: #868e96; margin-top: .15rem; }}
  .tags {{ margin-top: .25rem; display: flex; gap: .25rem; justify-content: center; flex-wrap: wrap; }}
  span.tag {{ font-size: .65rem; font-weight: 600; letter-spacing: .02em; text-transform: lowercase;
              padding: .05rem .35rem; border-radius: 999px; white-space: nowrap; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="caption">Each cell shows our lmms-eval score (<b>ours</b>) and the
Saliency-R1 paper's Table&nbsp;2 score (<b>paper</b>), with their difference
(Δ = ours&nbsp;−&nbsp;paper). Columns are the intersection of Table&nbsp;2
benchmarks and benchmarks we ran for both models; when we had several variants
of a benchmark, the one whose Qwen2.5-VL-7B score is closest to the paper's was
chosen (its lmms-eval task name is shown under each column). MME is the raw
perception+cognition sum. Greener cells are closer to the paper.
Saliency-R1-7B scores use the <code>mnt2048_r1sys3</code> recipe per task,
falling back to <code>mnt4096_r1sys3</code> when mnt2048 has no result.</p>
<table>
  <thead><tr><th>model</th>{header_cells}</tr></thead>
  <tbody>{"".join(body_rows)}</tbody>
</table>
</body>
</html>"""


def run_paper_comparison() -> None:
    scores = collect_lmms_eval_scores()
    if not scores:
        print(f"No lmms-eval results found in {LMMS_EVAL_DIR}")
        return

    # Build per-task Saliency-R1 scores from the best-recipe variants
    # (mnt2048_r1sys3 preferred; mnt4096_r1sys3 used when mnt2048 has no score).
    # reversed() so mnt2048 overwrites mnt4096 when both have the task.
    merged: dict[str, float] = {}
    for candidate in reversed(SALIENCY_R1_CANDIDATES):
        merged.update(scores.get(candidate, {}))
    if merged:
        scores[SALIENCY_R1_SLUG] = merged

    # Fold in hand-entered scores (e.g. MMBench en_test from the eval server).
    manual = load_manual_scores(MANUAL_SCORES_PATH)
    for slug, tasks in manual.items():
        scores[slug].update(tasks)
    _report_missing_manual(manual)

    columns = build_paper_columns(scores)
    if not columns:
        print("No Table-2 benchmarks were evaluated for both paper models.")
        return

    title = "Our lmms-eval vs Saliency-R1 paper (Table 2)"

    # --- terminal table (cells show: our-eval / paper) ---
    label_w = max(len(name) for name, _ in PAPER_MODELS)
    col_w = max(
        15,
        *(len(c["name"]) for c in columns),
        *(len("/".join(dataset_tags(c["task"]))) for c in columns),
    )
    print(title)
    print("(cells show: our-eval / paper)\n")
    def _task_label(c: dict) -> str:
        return c["task"] + ("*" if c.get("manual") else "")

    print(f"{'model':<{label_w}}  " + "  ".join(f"{c['name']:^{col_w}}" for c in columns))
    print(f"{'':<{label_w}}  " + "  ".join(f"{_task_label(c):^{col_w}}" for c in columns))
    print(f"{'':<{label_w}}  " + "  ".join(
        f"{'/'.join(dataset_tags(c['task'])):^{col_w}}" for c in columns))
    print("-" * (label_w + 2 + (col_w + 2) * len(columns)))
    if any(c.get("manual") for c in columns):
        print("(* score entered by hand from the eval server, not scored locally)")
    for display, slug in PAPER_MODELS:
        cells = []
        for c in columns:
            ours = scores.get(slug, {}).get(c["task"])
            paper = c["paper"].get(slug)
            cell = f"{_fmt_paper_score(ours, c['unit'])} / {_fmt_paper_score(paper, c['unit'])}"
            cells.append(f"{cell:^{col_w}}")
        print(f"{display:<{label_w}}  " + "  ".join(cells))

    # --- HTML page ---
    html = build_paper_html(scores, columns, title)
    html_name = "results_table_paper_comparison.html"
    html_path = HTML_DIR / html_name
    html_path.write_text(html)
    host = socket.gethostname()
    print(f"\nHTML table written to: {html_path}")
    print(f"scp {host}:{html_path} ~/Downloads/{html_name}")


def _ratio_to_color(ratio: float) -> str:
    """Map [0, 1] accuracy to a green-shaded background colour."""
    # pale yellow → green
    r = int(255 - ratio * 80)
    g = int(220 + ratio * 35)
    b = int(180 - ratio * 120)
    return f"rgb({r},{g},{b})"


# Diverging scale for the signed gaps in the comparison table: two opposed hues
# with a neutral grey midpoint, so "no difference" reads as no colour at all.
# The arms stop short of the full poles to keep dark text on them legible.
DIVERGING_NEUTRAL = (240, 239, 236)
DIVERGING_AHEAD = (42, 120, 214)     # blue: the row's first model scored higher
DIVERGING_BEHIND = (227, 73, 72)     # red:  the row's second model scored higher
DIVERGING_MAX = 0.75
# Gaps of this many error bars or more get the fully saturated arm.
DIVERGING_FULL_Z = 3.0
# A gap needs to clear this many error bars before it is called out as more than
# question-sampling noise (~95% under a normal approximation).
SIGNIFICANT_Z = 2.0


def _z_to_color(z: Optional[float]) -> str:
    """Background for a gap that is ``z`` error bars from zero."""
    if z is None:
        return ""
    pole = DIVERGING_AHEAD if z > 0 else DIVERGING_BEHIND
    strength = min(abs(z) / DIVERGING_FULL_Z, 1.0) * DIVERGING_MAX
    rgb = tuple(round(n + (p - n) * strength) for n, p in zip(DIVERGING_NEUTRAL, pole))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


EMPTY_CELL = ("—", None, None, None)


def build_html(data: dict, datasets: list[str], rows: list[tuple], benchmarks: dict[str, dict],
               default_models: set, default_datasets: set, title: str,
               runs: Optional[dict] = None) -> str:
    runs = runs or {}

    # Per-dataset min/max for relative colouring within each column
    col_values: dict[str, list[float]] = defaultdict(list)
    for model, notes in rows:
        for ds in datasets:
            _, ratio, _ans, _std = data[(model, notes)].get(ds, EMPTY_CELL)
            if ratio is not None:
                col_values[ds].append(ratio)

    def column_unit(ds: str) -> str:
        """"%" for the accuracy columns, "pts" for MME's raw ~2800-point total —
        which is the only column whose scores aren't fractions."""
        vals = col_values[ds]
        return "pts" if vals and max(vals) > 1.5 else "%"

    def spread_unit(ds: str) -> str:
        """The unit an error bar or a gap is quoted in for this column."""
        return "pp" if column_unit(ds) == "%" else "pts"

    def fmt_spread(ds: str, value: Optional[float], sign: bool = False) -> str:
        """An error bar or a gap, in the units the column is read in:
        percentage points for accuracies, raw points for MME."""
        if value is None:
            return ""
        scaled = value * 100 if column_unit(ds) == "%" else value
        digits = 2 if column_unit(ds) == "%" else 1
        return f"{scaled:+.{digits}f}" if sign else f"{scaled:.{digits}f}"

    def cell_style(ds: str, ratio: Optional[float]) -> str:
        if ratio is None:
            return ""
        vals = col_values[ds]
        if len(vals) < 2:
            return f'style="background:{_ratio_to_color(ratio)}"'
        lo, hi = min(vals), max(vals)
        norm = (ratio - lo) / (hi - lo) if hi > lo else 0.5
        return f'style="background:{_ratio_to_color(norm)}"'

    def tag_chips(ds: str, cls: str = "tags") -> str:
        tags = dataset_tags(ds)
        if not tags:
            return ""
        chips = "".join(
            f'<span class="tag" style="background:{TAG_INFO[t][0]};color:{TAG_INFO[t][1]}"'
            f' title="{html.escape(TAG_INFO[t][2], quote=True)}">{t}</span>'
            for t in tags
        )
        return f'<div class="{cls}">{chips}</div>'

    header_cells = "".join(
        f'<th data-col="{i}">{ds}{tag_chips(ds)}</th>' for i, ds in enumerate(datasets)
    )

    # Legend: only the tags actually used by the columns on this page.
    used_tags = [t for t in TAG_INFO if any(t in dataset_tags(ds) for ds in datasets)]
    tag_legend = "".join(
        f'<span class="tag" style="background:{TAG_INFO[t][0]};color:{TAG_INFO[t][1]}">{t}</span>'
        f'<span class="tag-desc">{html.escape(TAG_INFO[t][2])}</span>'
        for t in used_tags
    )

    def benchmark_row(label: str, score_key: str, model_key: str, source_key: str) -> Optional[str]:
        cells = []
        has_any = False
        for i, ds in enumerate(datasets):
            entry = benchmarks.get(ds, {})
            score = entry.get(score_key)
            if score in (None, ""):
                cells.append(f'<td data-col="{i}"></td>')
                continue
            has_any = True
            model_name = entry.get(model_key) or ""
            source = entry.get(source_key) or ""
            model_html = f"<br><span class='bench-note'>{model_name}</span>" if model_name else ""
            source_html = f'<br><a class="bench-source" href="{source}" target="_blank">source</a>' if source else ""
            cells.append(f'<td data-col="{i}">{score:.1f}%{model_html}{source_html}</td>')
        if not has_any:
            return None
        return f'<tr class="benchmark"><td class="model">{label}</td>{"".join(cells)}</tr>'

    benchmark_rows = "".join(
        r for r in (
            benchmark_row("Best recorded", "best_score", "best_model", "best_source"),
            benchmark_row("Similar models", "similar_score", "similar_model", "similar_source"),
        )
        if r
    )

    def model_label(model: str, notes: str) -> str:
        note = notes or MODEL_NOTES.get(model, "")
        return model + (f" <span class='notes'>[{html.escape(note)}]</span>" if note else "")

    def row_identity(model: str, notes: str) -> str:
        """Stable per-model key, shared by the results row and by every
        comparison row that names the model, so a rename reaches all of them --
        and so drag-order/renames in localStorage survive re-runs even when the
        alphabetical row index shifts. The separator keeps ("ab", "c") and
        ("a", "bc") from colliding."""
        return html.escape(f"{model}\x1f{notes}", quote=True)

    body_rows = []
    for ri, (model, notes) in enumerate(rows):
        label = model_label(model, notes)
        row_key = row_identity(model, notes)
        cells = []
        for i, ds in enumerate(datasets):
            text, ratio, ans_ratio, std = data[(model, notes)].get(ds, EMPTY_CELL)
            ans_str = f"<br><span class='answered'>{ans_ratio:.1%} answered</span>" if ans_ratio is not None else ""
            std_str = (f"<br><span class='std' title='If the benchmark had drawn a "
                       f"different sample of questions, roughly how far this score "
                       f"would move'>± {fmt_spread(ds, std)}</span>"
                       if std is not None else "")
            # data-ratio lets the page re-find the column max whenever the
            # visible set of models changes.
            ratio_attr = f' data-ratio="{ratio}"' if ratio is not None else ""
            cells.append(f'<td data-col="{i}"{ratio_attr} {cell_style(ds, ratio)}>'
                         f'<span class="score">{text}</span>{std_str}{ans_str}</td>')
        body_rows.append(
            f'<tr data-row="{ri}" data-key="{row_key}" draggable="true">'
            f'<td class="model"><span class="drag-handle" title="drag to reorder">⠿</span>'
            f'<span class="model-name" data-key="{row_key}" title="double-click to rename">{label}</span></td>'
            f'{"".join(cells)}</tr>'
        )

    # --- pairwise comparison table ---
    # One row per unordered pair of models, C(n,2) of them, each cell the gap
    # between the two on that benchmark measured question by question. Pairing
    # matters: both models answered the identical questions, so the luck of the
    # question draw cancels and the error bar on the gap is much tighter than
    # the two cells' individual error bars would suggest. A pair that shares no
    # benchmark simply comes out blank.
    pair_rows = []
    for pi, ((ia, (model_a, notes_a)), (ib, (model_b, notes_b))) in enumerate(
            itertools.combinations(enumerate(rows), 2)):
        cells = []
        for i, ds in enumerate(datasets):
            entry_a = runs.get((model_a, notes_a), {}).get(ds)
            entry_b = runs.get((model_b, notes_b), {}).get(ds)
            comparison = eval_stats.paired(entry_a[0] if entry_a else None,
                                           entry_b[0] if entry_b else None)
            if comparison is None:
                cells.append(f'<td data-col="{i}"></td>')
                continue
            z = comparison.z
            tip = (f"{model_a} − {model_b} on {ds}: "
                   f"{fmt_spread(ds, comparison.gap, sign=True)} ± "
                   f"{fmt_spread(ds, comparison.std)} {spread_unit(ds)}. "
                   f"{comparison.n_differing} of {comparison.n_units} questions "
                   f"scored differently"
                   + (f"; the gap is {abs(z):.1f} error bars from zero" if z else ""))
            z_attr = "" if z is None else f"{z:.4f}"
            gap_class = "gap significant" if z and abs(z) >= SIGNIFICANT_Z else "gap"
            cells.append(
                f'<td data-col="{i}" data-z="{z_attr}" '
                f'style="background:{_z_to_color(z)}" title="{html.escape(tip, quote=True)}">'
                f'<span class="{gap_class}">{fmt_spread(ds, comparison.gap, sign=True)}</span>'
                f'<br><span class="std">± {fmt_spread(ds, comparison.std)}</span></td>'
            )
        pair_rows.append(
            f'<tr data-pair="{pi}" data-a="{ia}" data-b="{ib}">'
            f'<td class="model">'
            f'<span class="pair-name" data-key="{row_identity(model_a, notes_a)}">'
            f'{model_label(model_a, notes_a)}</span>'
            f'<span class="pair-minus">−</span>'
            f'<span class="pair-name" data-key="{row_identity(model_b, notes_b)}">'
            f'{model_label(model_b, notes_b)}</span></td>'
            f'{"".join(cells)}</tr>'
        )

    model_checkboxes = "".join(
        f'<label><input type="checkbox" class="model-cb" value="{ri}"'
        f'{" checked" if (m, n) in default_models else ""}> '
        f'{m}{"  [" + n + "]" if n else ""}</label>'
        for ri, (m, n) in enumerate(rows)
    )
    dataset_checkboxes = "".join(
        f'<label><input type="checkbox" class="dataset-cb" value="{i}"'
        f'{" checked" if ds in default_datasets else ""}> {ds}'
        f'{tag_chips(ds, "tags inline")}</label>'
        for i, ds in enumerate(datasets)
    )

    # How the bars on this particular page were arrived at. The two methods
    # answer the same question and land on the same numbers, so the caption says
    # which one produced what the reader is looking at rather than leaving them
    # to guess from the filename.
    bootstrap = eval_stats.bootstrap_config()
    method_clause = (
        f"those questions resampled with replacement {bootstrap.resamples:,} times, "
        f"and the spread of the resulting scores taken"
        if bootstrap else
        "resampling those questions with replacement gives the same spread"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; padding: 2rem; background: #f8f9fa; color: #212529; }}
  h1 {{ font-size: 1.2rem; margin-bottom: 1rem; color: #495057; }}
  .filters {{ display: flex; gap: 2rem; margin-bottom: 1.5rem; background: white;
              padding: 1rem 1.4rem; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.1); flex-wrap: wrap; }}
  .filter-group {{ display: flex; flex-direction: column; gap: .35rem; }}
  .filter-group h2 {{ font-size: .8rem; font-weight: 700; text-transform: uppercase;
                      letter-spacing: .05em; color: #868e96; margin: 0 0 .2rem; }}
  .filter-group label {{ font-size: .85rem; display: flex; align-items: center; gap: .4rem;
                         white-space: nowrap; cursor: pointer; }}
  .toggle-btns {{ display: flex; gap: .4rem; margin-top: .2rem; }}
  .toggle-btns button {{ font-size: .75rem; padding: .15rem .5rem; border: 1px solid #ced4da;
                          background: #f1f3f5; border-radius: 4px; cursor: pointer; }}
  .toggle-btns button:hover {{ background: #dee2e6; }}
  table {{ border-collapse: collapse; background: white; box-shadow: 0 1px 4px rgba(0,0,0,.1); border-radius: 6px; overflow: hidden; }}
  th, td {{ padding: .55rem 1.1rem; text-align: center; border: 1px solid #dee2e6; font-size: .9rem; white-space: nowrap; }}
  th {{ background: #343a40; color: white; font-weight: 600; }}
  td.model {{ text-align: left; font-family: monospace; font-size: .85rem; background: #f1f3f5; }}
  span.notes {{ color: #868e96; }}
  span.answered {{ font-size: .78rem; color: #868e96; }}
  span.std {{ font-size: .78rem; color: #868e96; }}
  td.best span.score {{ font-weight: 700; }}
  h2.section {{ font-size: 1.05rem; margin: 2.5rem 0 .3rem; color: #495057; }}
  p.caption {{ font-size: .82rem; color: #868e96; max-width: 62rem; margin: 0 0 1rem; line-height: 1.5; }}
  span.gap {{ font-variant-numeric: tabular-nums; }}
  span.gap.significant {{ font-weight: 700; }}
  span.pair-minus {{ color: #adb5bd; margin: 0 .4rem; }}
  tr[data-pair] td.model {{ font-size: .78rem; }}
  span.bench-note {{ font-size: .78rem; color: #868e96; }}
  a.bench-source {{ font-size: .78rem; color: #4263eb; }}
  tr.benchmark td {{ background: #f1f3f5; color: #495057; font-style: italic; }}
  tr.benchmark td.model {{ background: #e9ecef; font-style: italic; }}
  tr:hover td {{ filter: brightness(.95); }}
  .hidden {{ display: none !important; }}
  tr[data-row].dragging td {{ opacity: .4; }}
  tr[data-row].drop-target td {{ box-shadow: inset 0 2px 0 #4263eb; }}
  span.drag-handle {{ cursor: grab; color: #adb5bd; margin-right: .4rem; user-select: none; }}
  .tags {{ margin-top: .25rem; display: flex; gap: .25rem; justify-content: center; flex-wrap: wrap; }}
  .tags.inline {{ display: inline-flex; margin: 0 0 0 .1rem; vertical-align: middle; }}
  span.tag {{ font-size: .65rem; font-weight: 600; letter-spacing: .02em; text-transform: lowercase;
              padding: .05rem .35rem; border-radius: 999px; white-space: nowrap; }}
  .tag-legend {{ display: flex; flex-wrap: wrap; align-items: center; gap: .3rem 1rem;
                 margin: -.5rem 0 1rem; font-size: .78rem; color: #868e96; }}
  .tag-legend .tag {{ margin-right: .25rem; }}
  span.tag-desc {{ margin-right: .6rem; }}
  span.model-name {{ cursor: text; }}
  span.renamed {{ color: #1971c2; }}
  p.layout-hint {{ font-size: .8rem; color: #868e96; margin: -.5rem 0 1rem; }}
  p.layout-hint button {{ font-size: .75rem; padding: .1rem .5rem; border: 1px solid #ced4da;
                          background: #f1f3f5; border-radius: 4px; cursor: pointer; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="filters">
  <div class="filter-group">
    <h2>Models</h2>
    {model_checkboxes}
    <div class="toggle-btns">
      <button onclick="setAll('model-cb', true)">All</button>
      <button onclick="setAll('model-cb', false)">None</button>
    </div>
  </div>
  <div class="filter-group">
    <h2>Datasets</h2>
    {dataset_checkboxes}
    <div class="toggle-btns">
      <button onclick="setAll('dataset-cb', true)">All</button>
      <button onclick="setAll('dataset-cb', false)">None</button>
    </div>
  </div>
</div>
<div class="tag-legend">{tag_legend}</div>
<p class="layout-hint">Drag the <span class="drag-handle">⠿</span> handle to reorder rows;
double-click a model name to rename it (blank to reset). Layout is saved in your
browser. <button id="reset-layout">Reset layout</button></p>
<p class="caption">Each score carries <b>±</b> its error bar, in percentage points
(raw points for MME): if the benchmark had drawn a different sample of questions,
about how far this score would move. It comes from the single run's own
per-question results — {method_clause} — so it covers the luck of the question
draw and says <b>nothing</b> about
seeds or decoding, which one run cannot speak to. It is blank where a run's
per-question scores were unavailable, or where replaying them did not reproduce
the reported score. Two overlapping bars here do <b>not</b> mean two models are
tied — for that, see the paired comparison below.</p>
<table>
  <thead><tr><th>model</th>{header_cells}</tr></thead>
  <tbody>{benchmark_rows}{''.join(body_rows)}</tbody>
</table>

<h2 class="section">Model vs model</h2>
<p class="caption">Every pair of the models ticked above, {len(pair_rows)} in all
when every model is ticked. Each cell is <b>first model − second model</b> on that
benchmark, in percentage points (raw points for MME), with its own error bar
below.
<br>These bars are <b>not</b> the two cells' bars combined. The two models
answered the identical questions, so the luck of the question draw is shared and
cancels: a question they both got right, or both got wrong, contributes nothing.
Only the questions they disagreed on carry the comparison, which makes the bar on
a gap far tighter — often about half — than comparing the two scores above as if
they had been measured separately. <b>Bold</b> marks a gap of at least
{SIGNIFICANT_Z:.0f} error bars; blue means the first model scored higher, red the
second, and the colour deepens with the number of error bars. Hover a cell for how
many questions the two models actually scored differently.</p>
<table>
  <thead><tr><th>comparison</th>{header_cells}</tr></thead>
  <tbody>{''.join(pair_rows)}</tbody>
</table>
<script>
  function update() {{
    const models = new Set([...document.querySelectorAll('.model-cb:checked')].map(cb => cb.value));
    const cols   = new Set([...document.querySelectorAll('.dataset-cb:checked')].map(cb => cb.value));
    document.querySelectorAll('tr[data-row]').forEach(row => {{
      row.classList.toggle('hidden', !models.has(row.dataset.row));
    }});
    // A comparison row needs both of its models ticked to mean anything.
    document.querySelectorAll('tr[data-pair]').forEach(row => {{
      row.classList.toggle('hidden', !(models.has(row.dataset.a) && models.has(row.dataset.b)));
    }});
    document.querySelectorAll('[data-col]').forEach(cell => {{
      cell.classList.toggle('hidden', !cols.has(cell.dataset.col));
    }});
    markColumnBest();
  }}

  // Bold the best score in each column, over the currently visible model rows
  // only -- the leader can change as models are ticked on and off. Reference
  // rows (best recorded / similar models) never take part.
  function markColumnBest() {{
    const best = new Map();  // col -> max ratio among visible rows
    document.querySelectorAll('tr[data-row]:not(.hidden) td[data-ratio]').forEach(td => {{
      const r = parseFloat(td.dataset.ratio);
      if (isNaN(r)) return;
      const col = td.dataset.col;
      if (!best.has(col) || r > best.get(col)) best.set(col, r);
    }});
    document.querySelectorAll('td[data-ratio]').forEach(td => {{
      const visible = !td.closest('tr').classList.contains('hidden');
      td.classList.toggle('best', visible && parseFloat(td.dataset.ratio) === best.get(td.dataset.col));
    }});
  }}
  function setAll(cls, checked) {{
    document.querySelectorAll('.' + cls).forEach(cb => cb.checked = checked);
    update();
  }}
  document.querySelectorAll('.model-cb, .dataset-cb').forEach(cb => cb.addEventListener('change', update));
  update();

  // --- row reorder + rename, persisted per-page in localStorage ---
  const NS = {json.dumps(title)};
  const orderKey = 'rt-order:' + NS;
  const renameKey = 'rt-rename:' + NS;
  const loadJSON = (k, def) => {{ try {{ return JSON.parse(localStorage.getItem(k)) || def; }} catch (e) {{ return def; }} }};
  const saveJSON = (k, v) => localStorage.setItem(k, JSON.stringify(v));

  const tbody = document.querySelector('table tbody');
  const dataRows = () => [...tbody.querySelectorAll('tr[data-row]')];

  // A model's name appears once in the results table and once per comparison
  // row that involves it; renaming has to reach all of them.
  const spansByKey = {{}};
  document.querySelectorAll('.model-name[data-key], .pair-name[data-key]').forEach(span => {{
    (spansByKey[span.dataset.key] = spansByKey[span.dataset.key] || []).push(span);
  }});
  const originalNames = {{}};
  Object.entries(spansByKey).forEach(([key, spans]) => {{ originalNames[key] = spans[0].innerHTML; }});

  const renames = loadJSON(renameKey, {{}});
  function applyRename(key) {{
    (spansByKey[key] || []).forEach(span => {{
      if (renames[key] != null) {{ span.textContent = renames[key]; span.classList.add('renamed'); }}
      else {{ span.innerHTML = originalNames[key]; span.classList.remove('renamed'); }}
    }});
  }}

  function applyOrder() {{
    const order = loadJSON(orderKey, []);
    if (!order.length) return;
    const byKey = new Map(dataRows().map(r => [r.dataset.key, r]));
    order.forEach(key => {{ const r = byKey.get(key); if (r) tbody.appendChild(r); }});
    byKey.forEach((r, key) => {{ if (!order.includes(key)) tbody.appendChild(r); }});  // new rows go last
  }}
  const saveOrder = () => saveJSON(orderKey, dataRows().map(r => r.dataset.key));

  applyOrder();
  Object.keys(spansByKey).forEach(applyRename);

  document.querySelectorAll('.model-name').forEach(span => {{
    span.addEventListener('dblclick', () => {{
      const key = span.dataset.key;
      const current = renames[key] != null ? renames[key] : span.textContent.trim();
      const val = prompt('Rename model (blank to reset):', current);
      if (val === null) return;
      if (val.trim() === '') delete renames[key]; else renames[key] = val;
      saveJSON(renameKey, renames);
      applyRename(key);
    }});
  }});

  let dragged = null;
  dataRows().forEach(row => {{
    row.addEventListener('dragstart', () => {{ dragged = row; row.classList.add('dragging'); }});
    row.addEventListener('dragend', () => {{
      row.classList.remove('dragging');
      dataRows().forEach(r => r.classList.remove('drop-target'));
      saveOrder();
    }});
    row.addEventListener('dragover', e => {{
      e.preventDefault();
      if (!dragged || dragged === row) return;
      const rect = row.getBoundingClientRect();
      const after = (e.clientY - rect.top) > rect.height / 2;
      dataRows().forEach(r => r.classList.remove('drop-target'));
      row.classList.add('drop-target');
      tbody.insertBefore(dragged, after ? row.nextSibling : row);
    }});
  }});

  document.getElementById('reset-layout').addEventListener('click', () => {{
    localStorage.removeItem(orderKey);
    localStorage.removeItem(renameKey);
    location.reload();
  }});
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lmms-eval", action=argparse.BooleanOptionalAction, default=True,
        help="Build the table from the lmms-eval outputs in results/ (the default, and "
             "what every paper table is built from). --no-lmms-eval reads the "
             "results/inference jsonl files instead, which this tree does not carry.",
    )
    parser.add_argument(
        "--paper-comparison", action="store_true",
        help="Compare our lmms-eval results for Qwen2.5-VL-7B and Saliency-R1-7B "
             "against the Saliency-R1 paper's Table 2 (off by default).",
    )
    parser.add_argument(
        "--results-root", type=Path, metavar="DIR",
        help="Read runs from this results/ directory instead of this tree's. "
             "results/ is not shared between worktrees (see CLAUDE.md), so a "
             "worktree otherwise sees only the runs that have been committed — "
             "point this at the central tree to build the full table. The page "
             "is still written to this tree's results/.",
    )
    parser.add_argument(
        "--bootstrap", action="store_true",
        help="Measure every error bar by resampling questions with replacement "
             "instead of reading it off the closed form the two agree on. Same "
             "bars to within 0.55%%, and ~30x slower: 13 minutes against 26 "
             "seconds for the full table. Writes to <name>_bootstrap.html so "
             "the default table is left alone (see eval_stats).",
    )
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=eval_stats.BOOTSTRAP_RESAMPLES,
        metavar="N", help="Resamples per error bar (default %(default)s).",
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=eval_stats.BOOTSTRAP_SEED,
        metavar="S", help="Seed for the resampling (default %(default)s).",
    )
    args = parser.parse_args()

    if args.results_root:
        global RESULTS_DIR, LMMS_EVAL_DIR
        if not args.results_root.is_dir():
            parser.error(f"--results-root {args.results_root} is not a directory")
        # The same shape as the module-level defaults above. The archive kept
        # `<root>/inference` and `<root>/lmms_eval` side by side; here the lmms-eval runs
        # ARE `results/`, so `--results-root evaluation/results` has to mean that
        # directory and not a `lmms_eval/` inside it, or the flag finds nothing.
        RESULTS_DIR = args.results_root / "inference"
        LMMS_EVAL_DIR = args.results_root

    if args.bootstrap:
        eval_stats.use_bootstrap(args.bootstrap_resamples, args.bootstrap_seed)
        # Not a progress bar, just the one number that explains the wait: the
        # pairwise table is ~9,500 gaps, and each one is resampled from scratch.
        print(f"Bootstrapping every error bar from {args.bootstrap_resamples:,} "
              f"resamples; this takes minutes, not seconds. Drop --bootstrap "
              f"for the closed form.", file=sys.stderr)

    if args.paper_comparison:
        run_paper_comparison()
        return

    # data[(model, notes)][dataset] = (display_string, ratio, answered_ratio, std)
    # runs[(model, notes)][dataset] = (per-question Run, unit label)
    data, runs = collect_lmms_eval() if args.lmms_eval else collect_inference()
    if not data:
        source = LMMS_EVAL_DIR if args.lmms_eval else RESULTS_DIR
        print(f"No results found in {source}")
        return

    datasets = sorted({ds for cell in data.values() for ds in cell})
    rows = sorted(data.keys())

    _warn_untagged(datasets)

    # --- terminal table ---
    row_label_width = max(
        len(f"{m}" + (f"  [{n}]" if n else "")) for m, n in rows
    )
    tag_labels = {ds: "/".join(dataset_tags(ds)) for ds in datasets}
    col_width = max(20, *(len(ds) for ds in datasets), *(len(t) for t in tag_labels.values()))

    print(f"{'model':<{row_label_width}}  " + "  ".join(f"{ds:^{col_width}}" for ds in datasets))
    print(f"{'':<{row_label_width}}  " + "  ".join(f"{tag_labels[ds]:^{col_width}}" for ds in datasets))
    print("-" * (row_label_width + 2 + (col_width + 2) * len(datasets)))

    for model, notes in rows:
        label = model + (f"  [{notes}]" if notes else "")
        cells = "  ".join(
            f"{data[(model, notes)].get(ds, EMPTY_CELL)[0]:^{col_width}}" for ds in datasets
        )
        print(f"{label:<{row_label_width}}  {cells}")

    # --- HTML page ---
    if args.lmms_eval:
        benchmarks_path = LMMS_EVAL_BENCHMARKS_PATH
        html_name = "results_table_lmms_eval.html"
        title = "lmms-eval Results"
        suite_tasks = load_lmms_eval_suite()
        # One matched PAIR ticked, not everything and not nothing. The original
        # reasoning stands and is why this is two rows rather than forty: every
        # checkpoint ever evaluated is a row here, so "all on" is a wall of
        # numbers with no comparison in it, and "All" is one button away when
        # that is what is wanted. But opening on a genuinely empty grid means
        # the page shows nothing until the reader already knows which of ~40 run
        # directories to tick, which is the wrong thing to require of whoever
        # opens it next. DEFAULT_LMMS_MODELS is a pair that differs in exactly
        # one thing, so the page opens on a comparison rather than a list.
        default_models: set = DEFAULT_LMMS_MODELS
        default_datasets = set(suite_tasks) & set(datasets)
        benchmarks = load_benchmarks(suite_tasks, benchmarks_path)
    else:
        benchmarks_path = BENCHMARKS_PATH
        html_name = "results_table.html"
        title = "VLM Inference Results"
        default_models, default_datasets = DEFAULT_MODELS, DEFAULT_DATASETS
        benchmarks = load_benchmarks(datasets, benchmarks_path)
    if args.bootstrap:
        # A separate file, not an overwrite: the point of building this one is
        # to hold it next to the default table and see whether the bars moved.
        html_name = html_name.replace(".html", "_bootstrap.html")
        title += " (bootstrap error bars)"
    html = build_html(data, datasets, rows, benchmarks, default_models, default_datasets,
                      title, runs)
    html_path = HTML_DIR / html_name
    html_path.write_text(html)
    host = socket.gethostname()
    print(f"\nHTML table written to: {html_path}")
    print(f"scp {host}:{html_path} ~/Downloads/{html_name}")


if __name__ == "__main__":
    main()
