# Copyright 2026 NVIDIA. Apache-2.0.
"""R_llm: the LLM-as-a-judge accuracy reward (Section 3.4, Appendix A.2).

GPT-4o-mini is shown the question, the ground-truth answer and the model's extracted
answer span, and asked for a correctness rating on a 1-5 scale, rescaled to [0, 1].

WHY A JUDGE AT ALL, BESIDE THE EXACT-MATCH REWARD. A third of the training corpus is
Flickr30k, whose gold answers are whole sentences -- almost none of them under four
words. Under exact match those rows score 0 on essentially every rollout, so the GRPO
group has no advantage spread and the sample teaches nothing. R_direct and R_llm are
complementary for that reason and both carry weight 1.

MASKED, NEVER ZERO, ON FAILURE. A sample the judge could not score returns None and is
imputed to its group's mean, so it neither gains nor loses advantage on that dimension.
Scoring it zero would make an unreachable API into a training signal.

Shared between the trainer and the offline probes, so "the reward the run computed" and
"the reward the probe reports" are the same function rather than two that agree.
"""

import re
import ast
import openai
from retrying import retry
import os
from concurrent.futures import ThreadPoolExecutor

# API errors worth retrying (transient). Everything else (e.g. a 400 content_filter
# from Azure) is deterministic -> fail fast and mask that sample rather than crash.
_TRANSIENT = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)

# LLM-as-judge via the NVIDIA inference API. Key is supplied at run time through
# NVIDIA_API_KEY (falls back to OPENAI_API_KEY). NVIDIA_API_KEY wins because the
# default base_url is the NVIDIA gateway: a stale OPENAI_API_KEY in the shell must
# not be sent there. Endpoint/model overridable via env.
_JUDGE_KEY = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not _JUDGE_KEY:
    # openai.OpenAI() raises on a None key, and this module is imported whether or
    # not the judge reward is actually used -- a missing key must not kill the run
    # at import. Use a placeholder and warn; every judged sample then masks to None.
    print("[selfsal.judge] WARNING: neither NVIDIA_API_KEY nor OPENAI_API_KEY is set. "
          "If openai_reward is among the reward funcs, every sample's judge reward "
          "will be masked (401).", flush=True)
    _JUDGE_KEY = "xxx"

client = openai.OpenAI(
    api_key=_JUDGE_KEY,
    base_url=os.environ.get("OPENAI_BASE_URL", "https://inference-api.nvidia.com"),
)

# The NVIDIA inference gateway requires provider-prefixed model names
# (e.g. "azure/openai/gpt-4o-mini"); the bare "gpt-4o-mini" alias returns a
# 403 key_model_access_denied. Override with the JUDGE_MODEL env var.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "azure/openai/gpt-4o-mini")

@retry(wait_exponential_multiplier=200, wait_exponential_max=2000, retry_on_exception=lambda e: isinstance(e, openai.RateLimitError) or isinstance(e, openai.APIConnectionError))
def openai_reward(completions, solution, problem, **kwargs):
    ground_truth_list = solution
    problem_list = problem
    contents = [completion[0]["content"] for completion in completions]
    #prediction_list = [i.split("</think>")[-1] for i in contents]
    prediction_list = [re.search(r"</think>\s*([^<]*(?:(?!<think>|</think>).)*?)\s*$", i, re.DOTALL | re.MULTILINE) for i in contents]
    #prediction_list = [re.search(r"<answer>\s*(.*?)\s*</answer>", i, re.DOTALL | re.MULTILINE) for i in contents]
    prediction_list = [i.group(1) if i else None for i in prediction_list]
    @retry(stop_max_attempt_number=5, wait_exponential_multiplier=200,
           retry_on_exception=lambda e: isinstance(e, _TRANSIENT))
    def query_gpt4o(question, ground_truth, prediction):
        # Compute the correctness score
        chat_completion = client.chat.completions.create(
            model=JUDGE_MODEL,  # override via JUDGE_MODEL env, e.g. "azure/openai/gpt-4o"
            temperature=0,
            max_tokens=512,
            messages=[
                {
                    "role": "system",
                    "content": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                               "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                               "------"
                               "##INSTRUCTIONS: "
                               "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                               "- Consider synonyms or paraphrases as valid matches.\n"
                               "- Evaluate the correctness of the prediction compared to the answer.",
                },
                {
                    "role": "user",
                    "content": f"I will give you an image and the following text as inputs:\n\n"
                               f"1. **Question Related to the Image**: {question}\n"
                               f"2. **Ground Truth Answer**: {ground_truth}\n"
                               f"3. **Model Predicted Answer**: {prediction}\n\n"
                               "Your task is to evaluate the model's predicted answer against the ground truth answer, based on the context provided by the image and the question. Consider the following criteria for evaluation:"
                               "- **Relevance**: Does the predicted answer directly address the question posed, considering the information provided in the image?"
                               "- **Accuracy**: Compare the predicted answer to the ground truth answer. Does the prediction accurately reflect the information given in the ground truth answer without introducing factual inaccuracies?"
                               "**Output Format**:"
                               "Score: <a integer score of quality from 1-5>",
                },
            ], timeout=120)

        response_message = chat_completion.choices[0].message.content
        # print(f"Response Message: {response_message}")
        score_match = re.search(r'Score:\s*(\d+)', response_message)
        if score_match:
            score = (int(score_match.group(1)) - 1.0) / 4.0
            return score

    def _score(args):
        question, ground_truth, prediction = args
        if not prediction:
            return 0
        try:
            return query_gpt4o(question, ground_truth, prediction)
        except Exception as e:
            # Non-retryable judge failure (e.g. Azure content_filter 400) must never
            # crash training. Mask this sample's judge reward -> None -> NaN downstream.
            print(f"[selfsal.judge] judge failed, masking sample: "
                  f"{type(e).__name__}: {str(e)[:160]}", flush=True)
            return None

    args_list = list(zip(problem_list, ground_truth_list, prediction_list))
    max_workers = max(1, int(os.environ.get("JUDGE_MAX_WORKERS", "8")))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_score, args_list))


#: Historical name, from when this lived in the TRL rewards package.
judge_reward = openai_reward
