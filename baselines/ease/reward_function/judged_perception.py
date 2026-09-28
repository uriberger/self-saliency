# Copyright 2026 NVIDIA. Apache-2.0.
"""EASE's perception reward, with our gpt-4o-mini judge behind it.

Loaded by their `AutoRewardManager` via
`worker.reward.reward_function=<abs path>:compute_score`.

## Why this file exists

EASE scores answers with `examples/reward_function/perception.py`: exact string
match, MCQ letter, or `mathruler.grade_answer`. On saliency-r1-8k that is fine
for 63% of rows, but flickr30k contributes 2,715 rows (34%) whose gold answers
are whole sentences -- exactly one of the 2,715 is <= 3 words. Under a rule
reward those rows score 0 on essentially every rollout, which costs twice over:
the GRPO group has zero advantage spread, *and* the sample never clears EASE's
tau=0.5 gate, so it contributes no attention supervision either. A third of the
corpus would burn rollout compute to teach nothing.

Our own overlap runs score this corpus with a gpt-4o-mini judge
(`trl/rewards/openai_rewards.py`). Using the same judge here also makes the two
stacks' accuracy curves mean the same thing.

## What it computes

  span      = a WELL-FORMED answer span, or nothing (see below)
  rule      = their perception.py accuracy, verbatim (imported, not copied)
  judge     = gpt-4o-mini on `span`, 1-5, mapped to (s-1)/4 -- only when rule == 0
  accuracy  = max(rule, judge)
  overall   = (1 - format_weight) * accuracy + format_weight * format

The rule pass runs first and short-circuits: a row their matcher already calls
correct never reaches the API. That is both cheaper and monotone -- this reward
is >= theirs on every sample, never below it.

## The span gate, and the run that made it necessary

A first pair of runs (2026-09-10, ease_8k / dapo_8k) reward-hacked this file.
Both arms converged on emitting

    <answer> Down <answer> The direction mentioned ... <answer> Down <answer> ...

repeated to the 1024-token cap and never closing the tag, and this reward scored
it 1.0. By step ~40 `format` had fallen 0.97 -> 0.05, response length had
saturated at the cap on 90% of rollouts, and judged accuracy read 0.88.

Their perception.py is immune to that shape by accident of construction: its
regex needs a CLOSING `</answer>`, so an unclosed tag falls through to
`answer_text = response.strip()` and is then killed by `len(answer_text) < 300`.
Rambling earns nothing, so rambling never pays. The first version of this file
kept the fallback and dropped the length guard, which opened exactly the door
their matcher closes -- and with `format_weight` at 0.0 (their default, which is
safe only alongside their guarded matcher) nothing pushed back.

So the judge is now shown a span or nothing at all:

  * a CLOSED `<answer>...</answer>`, or text after `</think>` (our cold start's
    own format), and nothing else -- never the whole response as a fallback,
    because "the model never delimited its answer" is not an answer;
  * at most `max_answer_chars` (300, their number) -- long enough for
    flickr30k's sentence answers, which run 100-150 characters.

Anything else scores 0 without an API call. The rule half is untouched and still
runs their code verbatim, so the DAPO arm's reward is unchanged.

`accuracy` is graded, not binary, because the judge's 1-5 scale is. EASE's gate
is `reward_threshold: 0.5`, so a judge score of 3/5 passes it and 2/5 does not.

## Configuration

Through `worker.reward.reward_function_kwargs` (all optional):

    format_weight   0.0    their default; format is reported either way
    rule_shortcut   true   skip the judge when the rule already matched
    judge_disabled  false  rule-only, i.e. exactly their reward (for A/B)

and through the environment, matching our TRL stack so one key serves both:

    NVIDIA_API_KEY / OPENAI_API_KEY   required, or every judged row falls back
    OPENAI_BASE_URL                   default https://inference-api.nvidia.com
    JUDGE_MODEL                       default azure/openai/gpt-4o-mini
    JUDGE_MAX_WORKERS                 default 32
    EASE_REPO                         where to import their perception.py from

A judge failure is never fatal: the row falls back to its rule score and is
counted in the `judge_failed` metric.
"""

from __future__ import annotations

import importlib.util
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REWARD_NAME = "judged_perception"
REWARD_TYPE = "batch"


# ── their matcher, imported rather than reimplemented ────────────────────────
def _load_their_perception():
    """Import EASE's own perception.py so the rule half is literally their code."""
    candidates = []
    env_file = os.environ.get("EASE_PERCEPTION_PY")
    if env_file:
        candidates.append(Path(env_file))
    env_repo = os.environ.get("EASE_REPO")
    if env_repo:
        candidates.append(Path(env_repo) / "examples/reward_function/perception.py")
    # Fall back to walking up from this file to a sibling ease_repo/ checkout.
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "ease_repo/examples/reward_function/perception.py")

    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("ease_perception_reward", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["ease_perception_reward"] = module
            spec.loader.exec_module(module)
            print(f"[judged_perception] rule half imported from {path}", flush=True)
            return module

    raise FileNotFoundError(
        "Could not locate EASE's examples/reward_function/perception.py. "
        "Set EASE_REPO or EASE_PERCEPTION_PY."
    )


_perception = _load_their_perception()


# ── the judge ────────────────────────────────────────────────────────────────
_JUDGE_KEY = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not _JUDGE_KEY:
    print(
        "[judged_perception] WARNING: neither NVIDIA_API_KEY nor OPENAI_API_KEY is set. "
        "Every judged row will fall back to the rule score, which on flickr30k means 0.",
        flush=True,
    )

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "azure/openai/gpt-4o-mini")
JUDGE_MAX_WORKERS = max(1, int(os.environ.get("JUDGE_MAX_WORKERS", "32")))
_JUDGE_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://inference-api.nvidia.com")

_client = None


def _get_client():
    """Build the OpenAI client lazily: this module is imported on a Ray actor."""
    global _client
    if _client is None:
        import openai

        _client = openai.OpenAI(api_key=_JUDGE_KEY or "xxx", base_url=_JUDGE_BASE_URL)
    return _client


# Byte-for-byte the prompt trl/rewards/openai_rewards.py sends, so a judge score
# here and a judge score there are the same measurement. It mentions an image it
# is not given; that is true of our runs too, and changing it would break the
# comparison this whole exercise exists to make.
_JUDGE_SYSTEM = (
    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
    "------"
    "##INSTRUCTIONS: "
    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
    "- Consider synonyms or paraphrases as valid matches.\n"
    "- Evaluate the correctness of the prediction compared to the answer."
)


def _judge_user_prompt(question: str, ground_truth: str, prediction: str) -> str:
    return (
        f"I will give you an image and the following text as inputs:\n\n"
        f"1. **Question Related to the Image**: {question}\n"
        f"2. **Ground Truth Answer**: {ground_truth}\n"
        f"3. **Model Predicted Answer**: {prediction}\n\n"
        "Your task is to evaluate the model's predicted answer against the ground truth answer, based on the context provided by the image and the question. Consider the following criteria for evaluation:"
        "- **Relevance**: Does the predicted answer directly address the question posed, considering the information provided in the image?"
        "- **Accuracy**: Compare the predicted answer to the ground truth answer. Does the prediction accurately reflect the information given in the ground truth answer without introducing factual inaccuracies?"
        "**Output Format**:"
        "Score: <a integer score of quality from 1-5>"
    )


_TRANSIENT_NAMES = ("RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError")


def _is_transient(exc: BaseException) -> bool:
    return type(exc).__name__ in _TRANSIENT_NAMES


def _judge_once(question: str, ground_truth: str, prediction: str) -> float | None:
    completion = _get_client().chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        max_tokens=512,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": _judge_user_prompt(question, ground_truth, prediction)},
        ],
        timeout=120,
    )
    text = completion.choices[0].message.content or ""
    match = re.search(r"Score:\s*(\d+)", text)
    if not match:
        return None
    return (min(5, max(1, int(match.group(1)))) - 1.0) / 4.0


def _judge(question: str, ground_truth: str, prediction: str, attempts: int = 5) -> float | None:
    """Judge one answer, retrying transient failures with exponential backoff.

    `retrying` is not installed in the `ease` env and adding a dependency to a
    file that gets exec'd inside a Ray actor is not worth it, so the backoff is
    inline. Returns None on give-up; the caller falls back to the rule score.
    """
    delay = 0.2
    for attempt in range(attempts):
        try:
            return _judge_once(question, ground_truth, prediction)
        except Exception as exc:  # noqa: BLE001 - a judge failure must not kill training
            if attempt == attempts - 1 or not _is_transient(exc):
                print(
                    f"[judged_perception] judge failed, falling back to rule score: "
                    f"{type(exc).__name__}: {str(exc)[:160]}",
                    flush=True,
                )
                return None
            time.sleep(delay + random.random() * delay)
            delay = min(delay * 2, 4.0)
    return None


# ── answer extraction ────────────────────────────────────────────────────────
# Both require a CLOSING delimiter. That is the whole point: an opening tag the
# model never closed means it did not finish delimiting an answer.
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_AFTER_THINK = re.compile(r"</think>\s*(.*)", re.DOTALL)

MAX_ANSWER_CHARS = 300  # their perception.py's own bound


def extract_prediction(response: str, max_answer_chars: int = MAX_ANSWER_CHARS) -> str:
    """Return the answer span the judge may see, or "" if there isn't one.

    Their prompt template asks for `<answer>...</answer>`; our cold-start SFT
    checkpoint was trained to emit `<think>...</think>` followed by the answer.
    Both shapes are accepted. Nothing else is: there is deliberately no
    whole-response fallback, and a span longer than `max_answer_chars` is
    rejected rather than truncated. See "The span gate" above -- the first pair
    of runs hacked precisely the fallback this used to have.
    """
    match = _ANSWER_TAG.search(response)
    if match:
        span = match.group(1).strip()
    else:
        match = _AFTER_THINK.search(response)
        span = match.group(1).strip() if match else ""

    if not span or len(span) > max_answer_chars:
        return ""
    return span


# ── the reward ───────────────────────────────────────────────────────────────
def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.0,
    rule_shortcut: bool = True,
    judge_disabled: bool = False,
    max_answer_chars: int = MAX_ANSWER_CHARS,
) -> list[dict[str, float]]:
    if not 0.0 <= format_weight <= 1.0:
        raise ValueError(f"format_weight must be in [0, 1], got {format_weight}")

    # Their matcher, unmodified, on the whole batch. format_weight=0 here because
    # we recombine below against the judged accuracy.
    rule_scores = _perception.compute_score(reward_inputs, format_weight=0.0)

    pending: list[int] = []
    for i, (reward_input, rule) in enumerate(zip(reward_inputs, rule_scores)):
        if judge_disabled:
            continue
        if rule_shortcut and rule["accuracy"] >= 1.0:
            continue
        pending.append(i)

    judged: dict[int, float | None] = {}
    if pending:
        def run(i: int) -> tuple[int, float | None]:
            reward_input = reward_inputs[i]
            prediction = extract_prediction(str(reward_input.get("response", "")), max_answer_chars)
            if not prediction:
                # No delimited answer, or one too long to be one. Score 0 and do
                # not spend an API call asking a judge to find an answer inside a
                # ramble -- it will, and that is the hack.
                return i, 0.0
            # `question` is supplied by patch_ease_repo.sh; without it the judge
            # still works, but grades a bare answer against a bare gold string.
            question = str(reward_input.get("question", ""))
            ground_truth = str(reward_input.get("ground_truth", ""))
            return i, _judge(question, ground_truth, prediction)

        with ThreadPoolExecutor(max_workers=min(JUDGE_MAX_WORKERS, len(pending))) as pool:
            judged = dict(pool.map(run, pending))

    scores: list[dict[str, float]] = []
    for i, rule in enumerate(rule_scores):
        judge_score = judged.get(i)
        accuracy = max(float(rule["accuracy"]), float(judge_score or 0.0))
        format_score = float(rule["format"])
        scores.append(
            {
                "overall": (1.0 - format_weight) * accuracy + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy,
                "rule_accuracy": float(rule["accuracy"]),
                "judge_called": 1.0 if i in judged else 0.0,
                "judge_failed": 1.0 if (i in judged and judge_score is None) else 0.0,
                # The canary. If this climbs, the policy is drifting back toward
                # undelimited or over-long answers and the run is going the way
                # ease_8k/dapo_8k went.
                "no_answer_span": 0.0 if extract_prediction(
                    str(reward_inputs[i].get("response", "")), max_answer_chars
                ) else 1.0,
            }
        )
    return scores
