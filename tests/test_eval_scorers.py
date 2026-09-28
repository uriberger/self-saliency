# Copyright 2026 NVIDIA. Apache-2.0.
"""The evaluation side, held against the paper.

Three things are checked, and each guards a failure that is silent rather than loud.

1. THE SUITE IS THE SUITE. Exactly the 25 tasks of Table 1. A benchmark quietly added or
   dropped changes every mean score in the paper, and nothing else would notice -- the
   mean is over "whatever was found".

2. THE NUMBERS COME BACK. Each arm's mean over the suite, recomputed from the banked
   results.json by the ported table code, against the published figure. This is the
   end-to-end check that the port did not change scoring: it exercises the collector,
   the per-task metric selection and the MME rescale together.

3. THE STALENESS GUARD IS ITSELF GUARDED. tables.py refuses to print a cell whose scorer
   version does not match the parser on disk -- the mechanism that caught one LogicVista
   checkpoint sitting in the table at both 0.45% and 56.2%. It compares against a
   hardcoded fallback string, which used to be kept in step with lmms-eval by hand. The
   fork is a pinned submodule now, so that string is checkable, and here it is checked.
   A fallback that drifts from the pin would make the guard pass on stale numbers, which
   is worse than not having it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
sys.path.insert(0, str(EVAL))

pytest.importorskip("yaml")


@pytest.fixture(scope="module")
def tables():
    return pytest.importorskip("tables")


#: Table 1, as lmms-eval task names. Written out rather than read from suite.yaml, so
#: that editing the suite cannot silently edit what it is being checked against.
TABLE_1 = {
    "mathvista_testmini_cot", "mathvision_testmini", "mathverse_testmini_vision_only",
    "wemath_testmini_reasoning", "mmk12",
    "logicvista_reasoning", "algopuzzlevqa", "visulogic", "dailyclue",
    "cv_bench", "omnispatial_test",
    "vstar_bench", "hrbench4k", "hrbench8k", "mmerealworld",
    "pope", "hallusion_bench_image", "illusionvqa_soft_localization",
    "chartqa", "p3",
    "mmstar", "mme", "realworldqa", "mmmu_pro_standard", "scienceqa_img",
}

#: run directory -> (paper label, expected mean over the 25 benchmarks).
#:
#: Two of these are NOT the printed figure. LogicVista was re-scored against the pinned
#: answer parser (docs/rescore-audit.md): Coldstart 54.91 -> 55.13 and No-Sal 55.13 ->
#: 54.69, which moves their means by +0.01 and -0.02. The published figures are kept in
#: PUBLISHED below so the difference is recorded rather than quietly absorbed. Nothing
#: reorders and SELF-SALIENCY is untouched.
PAPER_ARMS = {
    "grpo_coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged_overlap__wov0_4_2head_trmean_merged_mnt4096_r1":
        ("SELF-SALIENCY", 64.26),
    "grpo_coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged_saliency_r1_qwen3_merged_mnt4096_r1":
        ("Saliency-R1", 63.59),
    "grpo_coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged_no_saliency_saliency_r1_8k_merged_mnt4096_r1":
        ("No-Sal", 63.45),   # published 63.47
    "grpo_coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged_rect_frac_merged_mnt4096_r1":
        ("center rect", 63.38),
    "grpo_coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged_question_boxes_merged_mnt4096_r1":
        ("question boxes", 63.33),
    "grpo_coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged_overlap__wov0_033_2head_trmean_saliency_r1_8k_mean_in_v2_merged_mnt4096_r1":
        ("SELF-SALIENCY_mean", 63.17),
    "ease_8k_v2_step124_merged_mnt4096_r1": ("EASE", 62.88),
    "coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged_mnt4096_r1": ("Coldstart", 62.70),   # published 62.69
    "qwen3_vl_8b_instruct_mnt4096_vga_b0.2_l4-16": ("VGA", 61.16),
    "qwen3_vl_8b_instruct_mnt4096": ("Qwen3-VL-8B-Instruct", 61.14),
}


#: What Table 2 / Table 7 print, for the two arms the re-score moved.
PUBLISHED = {"No-Sal": 63.47, "Coldstart": 62.69}
MAX_RESCORE_DRIFT = 0.03


# --- 1. the suite -----------------------------------------------------------

def test_suite_is_table_1(tables):
    tasks = tables.load_lmms_eval_suite()
    assert len(tasks) == 25, f"the suite has {len(tasks)} benchmarks, not 25"
    assert len(set(tasks)) == 25, "the suite lists a benchmark twice"
    assert set(tasks) == TABLE_1, (
        f"suite.yaml and Table 1 disagree: only in suite {set(tasks) - TABLE_1}, "
        f"only in Table 1 {TABLE_1 - set(tasks)}")


def test_protocol_matches_appendix_a3(tables):
    p = tables.load_suite_protocol()
    assert p["temperature"] == 0
    assert p["num_beams"] == 1
    assert p["max_new_tokens"] == 4096
    assert p["repetition_penalty"] == 1.05
    assert p["batch_size"] == 1
    assert p["dtype"] == "bfloat16"
    assert p["max_pixels"] == 1605632          # 2,048 visual tokens at 28x28
    assert p["max_pixels_highres"] == 3211264  # 4,096, for the resolution benchmarks
    assert set(p["highres_tasks"]) == {"mmerealworld", "hrbench4k", "hrbench8k"}


def test_suite_default_pixel_budget_matches_the_pinned_fork():
    """1,605,632 is asserted to be lmms-eval's default, so read it back off the pin."""
    model = EVAL / "lmms_eval/lmms_eval/models/simple/qwen3_vl.py"
    if not model.exists():
        pytest.skip("submodule not checked out")
    m = re.search(r"max_pixels:\s*int\s*=\s*(\d+)", model.read_text())
    assert m, "could not find the max_pixels default in the pinned fork"
    assert int(m.group(1)) == 1605632


# --- 2. the numbers ---------------------------------------------------------

@pytest.fixture(scope="module")
def scores(tables):
    if not (EVAL / "results").is_dir():
        pytest.skip("no banked results")
    return tables.collect_lmms_eval_scores()


@pytest.mark.parametrize("run,expected", list(PAPER_ARMS.items()),
                         ids=[v[0] for v in PAPER_ARMS.values()])
def test_arm_mean_score_matches_the_paper(tables, scores, run, expected):
    label, published = expected
    per_task = scores.get(run)
    assert per_task is not None, f"{label}: no banked results under {run}"

    tasks = tables.load_lmms_eval_suite()
    found = [t for t in tasks if per_task.get(t) is not None]
    assert len(found) == 25, (
        f"{label}: {len(found)} of 25 benchmarks present; missing "
        f"{[t for t in tasks if t not in found]}")

    # MME totals to 2800, so it is rescaled into [0, 100] before the mean.
    values = [per_task[t] / 28.0 if t == "mme" else per_task[t] for t in found]
    got = sum(values) / len(values)
    assert got == pytest.approx(published, abs=0.02), (
        f"{label}: recomputed {got:.3f}, paper says {published}")


def test_arms_rank_as_published(tables, scores):
    """The ordering is the claim; a tie or a swap would not show up in any one cell."""
    tasks = tables.load_lmms_eval_suite()
    means = {}
    for run, (label, _) in PAPER_ARMS.items():
        per_task = scores.get(run) or {}
        values = [per_task[t] / 28.0 if t == "mme" else per_task[t]
                  for t in tasks if per_task.get(t) is not None]
        if len(values) == 25:
            means[label] = sum(values) / len(values)

    order = [label for label, _ in sorted(means.items(), key=lambda kv: -kv[1])]
    assert order[0] == "SELF-SALIENCY", f"the best arm is {order[0]}"
    assert order[-1] == "Qwen3-VL-8B-Instruct"
    assert order == [label for _, (label, _) in
                     sorted(PAPER_ARMS.items(), key=lambda kv: -kv[1][1])]


# --- 3. the guard on the guard ----------------------------------------------

def test_parser_fallbacks_match_the_pinned_submodule(tables):
    """tables.PARSERS' fallback versions must equal the pin's PARSER_VERSION.

    The fallback is what the staleness check compares against when it cannot read the
    checkout. If it drifts from the pin, the check passes on numbers produced by a
    scorer that no longer exists -- the exact failure it was written to catch, with the
    warning light wired to the wrong sensor.
    """
    assert tables.PARSERS, "no parsers registered; the staleness check is inert"
    for marker, (source, _stamp_key, fallback, _cmd) in tables.PARSERS.items():
        if not Path(source).exists():
            pytest.skip(f"submodule not checked out ({source})")
        m = re.search(r'^PARSER_VERSION\s*=\s*["\']([^"\']+)["\']',
                      Path(source).read_text(), re.MULTILINE)
        assert m, f"{marker}: no PARSER_VERSION in {source}"
        assert m.group(1) == fallback, (
            f"{marker}: the pinned fork says {m.group(1)!r} but tables.py falls back to "
            f"{fallback!r}. Update the fallback, and re-score anything banked under the "
            f"old one.")


def test_parsers_point_inside_the_submodule(tables):
    """Not at a checkout that happens to be on the porting machine."""
    submodule = (EVAL / "lmms_eval").resolve()
    for marker, (source, *_rest) in tables.PARSERS.items():
        assert submodule in Path(source).resolve().parents, (
            f"{marker} reads its parser version from {source}, outside the pinned "
            f"submodule")


def test_banked_results_are_scored_by_the_pinned_parser(tables):
    """Every cell the table would actually USE carries the pinned parser's version.

    Two details this gets right that an obvious version would not.

    The stamp sits at the JSON's TOP level, beside `results`, not inside the per-task
    metrics. Looking in the wrong place makes the check vacuous.

    Only the WINNING file per (run, task) matters. The collector takes the latest
    timestamp, so a superseded file left in the tree is not a stale number -- it is a
    number nothing reads. Flagging those would train the reader to ignore this test.
    """
    results = EVAL / "results"
    if not results.is_dir():
        pytest.skip("no banked results")

    # marker metric -> (stamp key, expected version)
    expected = {marker: (stamp_key, fallback)
                for marker, (_src, stamp_key, fallback, _cmd) in tables.PARSERS.items()}

    winning: dict[tuple[str, str], tuple[str, dict]] = {}
    for path in results.rglob("*results.json"):
        run = path.relative_to(results).parts[0]
        timestamp = path.name.split("_results")[0]
        try:
            blob = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(blob, dict):
            continue
        for task, metrics in (blob.get("results") or {}).items():
            if not isinstance(metrics, dict):
                continue
            if not any(marker in metrics for marker in expected):
                continue
            key = (run, task)
            if key not in winning or timestamp > winning[key][0]:
                winning[key] = (timestamp, blob)

    assert winning, "no versioned task results found; this check is inert"

    stale = []
    for (run, task), (_ts, blob) in sorted(winning.items()):
        metrics = blob["results"][task]
        for marker, (stamp_key, want) in expected.items():
            if marker in metrics and blob.get(stamp_key) != want:
                stale.append(f"{run}/{task}: {blob.get(stamp_key)!r} != {want!r}")
    assert not stale, (
        f"{len(stale)} banked cell(s) were scored by a parser other than the pinned "
        f"one:\n  " + "\n  ".join(stale[:8]))


def test_rescore_drift_from_the_published_figures_is_small(tables, scores):
    """The two re-scored arms must stay within a rounding step of what was printed.

    This is not a formality. If a future re-score moved one of these by a point, the
    repository and the paper would disagree materially and someone has to decide which is
    right. Catching it here makes that a decision rather than a discovery.
    """
    task_lists = tables.load_lmms_eval_suite()
    for run, (label, expected) in PAPER_ARMS.items():
        if label not in PUBLISHED:
            continue
        per_task = scores.get(run)
        assert per_task is not None
        values = [per_task[t] / 28.0 if t == "mme" else per_task[t] for t in task_lists]
        got = sum(values) / len(values)
        drift = abs(got - PUBLISHED[label])
        assert drift < MAX_RESCORE_DRIFT, (
            f"{label}: re-scored mean {got:.3f} is {drift:.3f} from the published "
            f"{PUBLISHED[label]}, past the {MAX_RESCORE_DRIFT} rounding allowance")
