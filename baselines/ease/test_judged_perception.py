#!/usr/bin/env python3
# Copyright 2026 NVIDIA. Apache-2.0.
"""Regression test for the span gate in judged_perception.py.

The cases are real: they are read out of a run's own
`checkpoints/generations.log`, which EasyR1 writes as

    [prompt] ...
    [output] ...
    [ground_truth] ...
    [score] 1.0

The first pair of runs (ease_8k / dapo_8k) hacked this reward by emitting
`<answer> X <answer> X <answer> ...` to the 1024-token cap without ever closing
the tag, and were scored 1.0 for it. Those completions are the point of this
test -- it asserts the gate now rejects every response the old reward scored
while it was degenerating, without an API call, and still accepts well-formed
ones.

No network: the judge is never reached, because a rejected span short-circuits
to 0 before any API call and an accepted span is only checked for acceptance.

Usage:
    python3 ease/test_judged_perception.py
    python3 ease/test_judged_perception.py --generations outputs/ease/ease_8k/checkpoints/generations.log
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_reward():
    os.environ.setdefault("EASE_REPO", str(REPO / "ease_repo"))
    path = REPO / "ease/reward_function/judged_perception.py"
    spec = importlib.util.spec_from_file_location("judged_perception", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["judged_perception"] = module
    spec.loader.exec_module(module)
    return module


def parse_generations(path: Path) -> list[dict[str, str]]:
    """Parse EasyR1's generations.log into records."""
    records, current, field = [], {}, None
    for line in path.read_text(errors="replace").splitlines():
        for tag in ("prompt", "output", "ground_truth", "score"):
            marker = f"[{tag}]"
            if line.startswith(marker):
                if tag == "prompt" and current:
                    records.append(current)
                    current = {}
                field = tag
                current[field] = line[len(marker):].strip()
                break
        else:
            if field:
                current[field] = current.get(field, "") + "\n" + line
    if current:
        records.append(current)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generations",
        type=Path,
        default=REPO / "outputs/ease/ease_8k/checkpoints/generations.log",
        help="A run's generations.log to replay.",
    )
    args = parser.parse_args()
    reward = load_reward()
    failures: list[str] = []

    # ── synthetic cases, pinning the contract ────────────────────────────────
    cases = [
        ("closed tag, short", "<think>x</think><answer>horse</answer>", "horse"),
        ("after </think>", "<think>reasoning</think>The answer is dusk.", "The answer is dusk."),
        ("flickr30k-length sentence",
         "<answer>" + "Yes, the image conveys that it is around dusk due to the light." + "</answer>",
         "Yes, the image conveys that it is around dusk due to the light."),
        # The hack, in its exact shape.
        ("unclosed tag, repeated to cap", "<answer> Down " * 200, ""),
        ("no delimiter at all", "The direction is Down. " * 60, ""),
        ("closed tag but over 300 chars", "<answer>" + "x" * 400 + "</answer>", ""),
        ("empty", "", ""),
    ]
    for name, response, expected in cases:
        got = reward.extract_prediction(response)
        if got != expected:
            failures.append(f"extract_prediction({name}): expected {expected[:40]!r}, got {got[:60]!r}")

    print(f"[1] {len(cases)} synthetic cases")

    # ── replay the real degenerate completions ───────────────────────────────
    if not args.generations.is_file():
        print(f"[2] skipped: {args.generations} not found")
    else:
        records = parse_generations(args.generations)
        scored_one = [r for r in records if r.get("score", "").strip().startswith("1")]
        # A response is "degenerate" if the old reward would have had to fall
        # back to the whole response: no closed <answer>, no </think>.
        degenerate = [
            r for r in scored_one
            if "</answer>" not in r.get("output", "") and "</think>" not in r.get("output", "")
        ]
        print(f"[2] {len(records)} logged generations, {len(scored_one)} scored 1.0, "
              f"{len(degenerate)} of those with no closing delimiter")

        leaked = [r for r in degenerate if reward.extract_prediction(r["output"])]
        if leaked:
            failures.append(
                f"{len(leaked)}/{len(degenerate)} undelimited completions still yield a span; "
                f"first: {reward.extract_prediction(leaked[0]['output'])[:80]!r}"
            )

        # And the gate must not be so strict that nothing survives: well-formed
        # short answers in the same log still have to pass.
        wellformed = [
            r for r in records
            if "</answer>" in r.get("output", "") and len(r["output"]) < 2000
        ]
        passing = [r for r in wellformed if reward.extract_prediction(r["output"])]
        print(f"    {len(passing)}/{len(wellformed)} well-formed completions still yield a span")
        if wellformed and not passing:
            failures.append("the gate rejects every well-formed completion too -- too strict")

    # ── scoring end to end, judge disabled (no network) ──────────────────────
    inputs = [
        {"response": "<answer> Down " * 200, "response_length": 1024, "ground_truth": "DOWN", "question": "?"},
        {"response": "<think>x</think><answer>DOWN</answer>", "response_length": 20,
         "ground_truth": "DOWN", "question": "?"},
    ]
    scores = reward.compute_score(inputs, judge_disabled=True)
    if scores[0]["no_answer_span"] != 1.0:
        failures.append("no_answer_span should be 1.0 for the hacked completion")
    if scores[1]["no_answer_span"] != 0.0:
        failures.append("no_answer_span should be 0.0 for the well-formed completion")
    if scores[1]["accuracy"] != 1.0:
        failures.append(f"well-formed exact match should score 1.0, got {scores[1]['accuracy']}")
    print(f"[3] no_answer_span: hacked={scores[0]['no_answer_span']} "
          f"well-formed={scores[1]['no_answer_span']}")

    if failures:
        print()
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
