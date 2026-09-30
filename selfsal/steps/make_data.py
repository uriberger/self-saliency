"""Generate step-classification training data using a closed LLM (default: Gemini 2.5 Pro).

Reads reasoning chains from non-pod_prompt JSONL files in results/inference/,
splits each chain into atomic fragments, and calls the LLM to classify each
fragment as plan / observe / deduce.

Two splitting modes (--split_mode):

  llm   (default) — The LLM both splits and classifies in one call. It is
                    asked to break each coarse step into atomic sub-fragments
                    where each fragment is a pure single type. Best quality.

  rules           — Rule-based splitting on discourse connectives (e.g.
                    ", indicating that", ", therefore") is applied before
                    calling the LLM, which then only classifies.

Output (steps_classifier/data/labeled_steps.jsonl), one line per fragment:
  {
    "source_file":    str,
    "sample_id":      str,
    "question":       str,
    "original_chain": str,   # full raw response text from the inference file
    "step_index":     int,   # 0-based position in the flat fragment list
    "step_text":      str,
    "label":          "plan" | "observe" | "deduce",
    "split_mode":     "llm" | "rules"
  }

The script is resumable: already-written (source_file, sample_id) pairs are
skipped so you can kill and restart without losing progress.

Usage
-----
  python -m selfsal.steps.make_data --api_key YOUR_KEY
  python -m selfsal.steps.make_data --api_key YOUR_KEY --split_mode rules
  python -m selfsal.steps.make_data --api_key YOUR_KEY --max_samples 50

The endpoint is OPENAI_BASE_URL and the model is --model; they are a pair, and the
defaults are a gateway that serves Gemini 2.5 Pro rather than OpenAI's public API. See
docs/publishing.md §2 for why this one script is not defaulted to api.openai.com.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_INFERENCE_DIR = _PROJECT_ROOT / "results" / "inference"
_OUTPUT_FILE = Path(__file__).resolve().parent / "data" / "labeled_steps.jsonl"
#: THE ENDPOINT AND `--model` ARE A PAIR. Unlike the judge in `selfsal/judge.py`, this
#: default is NOT OpenAI's public API, because `--model` defaults to Gemini 2.5 Pro and
#: api.openai.com does not serve it. Override both together, with OPENAI_BASE_URL and
#: --model, to distil from any OpenAI-compatible endpoint. See docs/publishing.md §2.
_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://inference-api.nvidia.com/v1")

# ---------------------------------------------------------------------------
# Coarse step extraction (format-aware, no type classification yet)
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 10]


def extract_steps(response: str) -> list[str]:
    """Split a reasoning chain into coarse ordered steps, regardless of format."""
    pod_steps = re.findall(
        r"<(plan|observe|deduce)>(.*?)</\1>", response, re.DOTALL | re.IGNORECASE
    )
    if pod_steps:
        return [text.strip() for _, text in pod_steps if text.strip()]

    step_tags = re.findall(r"<step>(.*?)</step>", response, re.DOTALL)
    if step_tags:
        return [s.strip() for s in step_tags if s.strip()]

    m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if m:
        return _split_sentences(m.group(1))

    text = re.sub(r"<answer>.*?</answer>", "", response, flags=re.DOTALL)
    return _split_sentences(text)


# ---------------------------------------------------------------------------
# Rule-based sub-sentence splitting (Option A)
# ---------------------------------------------------------------------------

# Split just before these discourse connectives when they follow a comma or
# semicolon, signalling a transition from an observation to a deduction.
_SPLIT_CUE = re.compile(
    r"(?<=[,;])\s+(?="
    r"indicating\s+(?:that\s+)?"
    r"|suggesting\s+(?:that\s+)?"
    r"|implying\s+(?:that\s+)?"
    r"|meaning\s+(?:that\s+)?"
    r"|which\s+means\s+"
    r"|therefore\b"
    r"|thus\b"
    r"|hence\b"
    r"|as\s+a\s+result\b"
    r")",
    re.IGNORECASE,
)


def split_step_rules(step: str) -> list[str]:
    """Further split one coarse step into atomic fragments using discourse cues."""
    fragments = _SPLIT_CUE.split(step)
    # Strip trailing commas/semicolons left on the first fragment after splitting
    cleaned = [f.rstrip(",;").strip() for f in fragments]
    return [f for f in cleaned if len(f) >= 5]


def apply_rules_split(steps: list[str]) -> list[str]:
    """Apply rule-based splitting to all coarse steps, returning a flat list."""
    fragments = []
    for step in steps:
        fragments.extend(split_step_rules(step))
    return fragments


# ---------------------------------------------------------------------------
# LLM prompts and API calls
# ---------------------------------------------------------------------------

# --- Mode: rules (classify only) ---

_CLASSIFY_SYSTEM_PROMPT = """\
You classify fragments of a visual reasoning chain. Each fragment belongs to exactly one category:

- plan: A forward-looking intention — what the model intends to examine or do next \
("I will look at...", "I need to identify...", "Let me check...").
- observe: A direct claim about visual content in the image \
("I see...", "The image shows...", "There is a red car on the left...").
- deduce: An inference or conclusion drawn from prior observations or reasoning \
("Based on...", "Therefore...", "Indicating that...", "Since X, it follows that..."). \
These fragments synthesise earlier steps rather than introducing new visual evidence.
- none: A fragment that does not contribute to the reasoning process. This includes \
filler words or hesitations ("Hmm.", "Wait", "No, wait-"), fragments that restate or \
list the question options ("The options are A. ..., B. ..., C. ..."), and fragments \
that repeat or paraphrase the question ("The user is asking me to count how many dogs \
are in this image").

You will receive a question and a numbered list of fragments. Return ONLY a JSON \
array of labels in the same order. Each label must be one of: "plan", "observe", "deduce", "none". \
No explanation, no extra text.

Example output for 4 fragments:
["none", "plan", "observe", "deduce"]
"""

# --- Mode: llm (split + classify) ---

_SPLIT_CLASSIFY_SYSTEM_PROMPT = """\
You split and classify a visual reasoning chain into atomic fragments. Each fragment \
must be purely one type — no fragment should mix types.

Categories:
- plan: A forward-looking intention — what the model intends to examine or do next \
("I will look at...", "I need to identify...", "Let me check...").
- observe: A direct claim about visual content in the image \
("I see...", "The image shows...", "There is a red car on the left...").
- deduce: An inference or conclusion drawn from prior observations or reasoning \
("Based on...", "Therefore...", "Indicating that...", "Since X, it follows that..."). \
These fragments synthesise earlier steps rather than introducing new visual evidence.
- none: A fragment that does not contribute to the reasoning process. This includes \
filler words or hesitations ("Hmm.", "Wait", "No, wait-"), fragments that restate or \
list the question options ("The options are A. ..., B. ..., C. ..."), and fragments \
that repeat or paraphrase the question ("The user is asking me to count how many dogs \
are in this image").

You will receive a question and a raw reasoning chain. Split it into fine-grained \
atomic fragments and classify each one.

Return ONLY a flat JSON array of {"text": "...", "label": "..."} objects in order. \
No explanation, no extra text.

Example:
[
  {"text": "The user is asking me to identify the animal in the image", "label": "none"},
  {"text": "I will examine the left side of the image", "label": "plan"},
  {"text": "The dog's head is turned to the left", "label": "observe"},
  {"text": "indicating that it is facing that direction", "label": "deduce"},
  {"text": "I will now check the background for additional clues", "label": "plan"}
]
"""


def _build_user_message(question: str, steps: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    return f"Question: {question}\n\nSteps:\n{numbered}"


def _call_api(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_message: str,
    max_retries: int,
) -> str | None:
    """Make a single API call with retries. Returns raw content string or None."""
    delay = 2.0
    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
                max_tokens=32768,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [warn] API error ({e}), retrying…", file=sys.stderr)
        time.sleep(delay)
        delay = min(delay * 2, 60.0)
    return None


def _extract_json(raw: str) -> str | None:
    """Extract the outermost JSON array from a raw LLM response."""
    # Find the first '[' and the matching ']' using a simple bracket counter
    start = raw.find("[")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(raw[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return None


def classify_fragments(
    client: OpenAI,
    model_name: str,
    question: str,
    fragments: list[str],
    max_retries: int = 5,
) -> list[str] | None:
    """Classify pre-split fragments. Returns a list of labels or None."""
    prompt = _build_user_message(question, fragments)
    raw = _call_api(client, model_name, _CLASSIFY_SYSTEM_PROMPT, prompt, max_retries)
    if raw is None:
        return None

    json_str = _extract_json(raw)
    if json_str is None:
        print(f"  [warn] No JSON array in response: {raw[:200]!r}", file=sys.stderr)
        return None

    try:
        labels = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  [warn] JSON parse error ({e}): {json_str[:200]!r}", file=sys.stderr)
        return None

    if (
        isinstance(labels, list)
        and len(labels) == len(fragments)
        and all(l in ("plan", "observe", "deduce", "none") for l in labels)
    ):
        return labels

    print(f"  [warn] Unexpected classify response shape: {labels!r}", file=sys.stderr)
    return None


def split_and_classify(
    client: OpenAI,
    model_name: str,
    question: str,
    chain: str,
    max_retries: int = 5,
) -> list[dict] | None:
    """Ask the LLM to split the raw reasoning chain and classify fragments in one call.

    Returns a flat list of {"text": str, "label": str} dicts, or None on failure.
    """
    prompt = f"Question: {question}\n\nReasoning chain:\n{chain}"
    raw = _call_api(client, model_name, _SPLIT_CLASSIFY_SYSTEM_PROMPT, prompt, max_retries)
    if raw is None:
        return None

    json_str = _extract_json(raw)
    if json_str is None:
        print(f"  [warn] No JSON array in response: {raw[:200]!r}", file=sys.stderr)
        return None

    try:
        fragments = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  [warn] JSON parse error ({e}): {json_str[:200]!r}", file=sys.stderr)
        return None

    if not isinstance(fragments, list):
        print(f"  [warn] Expected a list, got: {type(fragments)}", file=sys.stderr)
        return None

    for frag in fragments:
        if (
            not isinstance(frag, dict)
            or "text" not in frag
            or "label" not in frag
            or frag["label"] not in ("plan", "observe", "deduce", "none")
        ):
            print(f"  [warn] Invalid fragment: {frag!r}", file=sys.stderr)
            return None

    return [{"text": f["text"], "label": f["label"]} for f in fragments]


# ---------------------------------------------------------------------------
# Source file discovery
# ---------------------------------------------------------------------------

def _is_eligible(path: Path) -> bool:
    name = path.name
    if not name.endswith(".jsonl"):
        return False
    if "pod_prompt" in name or "poc_prompt" in name:
        return False
    return True


def discover_inference_files() -> list[Path]:
    return sorted(p for p in _INFERENCE_DIR.iterdir() if _is_eligible(p))


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_done_ids(output_path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not output_path.exists():
        return done
    with output_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add((rec["source_file"], str(rec["sample_id"])))
            except Exception:
                pass
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate POD step-classification data via Gemini")
    parser.add_argument("--api_key", required=True,
                        help="Bearer token for whichever endpoint OPENAI_BASE_URL names")
    parser.add_argument(
        "--model",
        default="gcp/google/gemini-2.5-pro",
        help="Model name as used by the NVIDIA inference API",
    )
    parser.add_argument(
        "--split_mode",
        choices=["llm", "rules"],
        default="llm",
        help="'llm': LLM splits and classifies (default). 'rules': rule-based split then LLM classifies.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to process per source file (default: all)",
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        default=None,
        help="Process a single inference JSONL file instead of discovering all files in results/inference/",
    )
    parser.add_argument(
        "--input_list",
        type=Path,
        default=None,
        help="Text file with one filename per line (no results/inference/ prefix); those files are processed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_OUTPUT_FILE,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--rpm_limit",
        type=float,
        default=10.0,
        help="Max requests per minute (default: 10). Adds inter-call sleep.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=args.api_key, base_url=_BASE_URL)

    if args.input_file is not None:
        if not args.input_file.exists():
            print(f"Input file not found: {args.input_file}", file=sys.stderr)
            sys.exit(1)
        inference_files = [args.input_file]
    elif args.input_list is not None:
        if not args.input_list.exists():
            print(f"Input list not found: {args.input_list}", file=sys.stderr)
            sys.exit(1)
        with args.input_list.open() as f:
            names = [line.strip() for line in f if line.strip()]
        inference_files = [_INFERENCE_DIR / name for name in names]
        missing = [p for p in inference_files if not p.exists()]
        if missing:
            for p in missing:
                print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)
    else:
        inference_files = discover_inference_files()
        if not inference_files:
            print("No eligible inference files found.", file=sys.stderr)
            sys.exit(1)
    print(f"Found {len(inference_files)} eligible inference file(s). Split mode: {args.split_mode}")

    done_ids = load_done_ids(args.output)
    print(f"Resuming: {len(done_ids)} (source_file, sample_id) pairs already done.")

    min_sleep = 60.0 / args.rpm_limit
    total_fragments = 0
    total_samples = 0

    with args.output.open("a") as out_f:
        for src_path in inference_files:
            src_name = src_path.name
            print(f"\n--- {src_name} ---")

            with src_path.open() as f:
                records = [json.loads(line) for line in f if line.strip()]

            random.shuffle(records)

            if args.max_samples is not None:
                records = records[: args.max_samples]

            pbar = tqdm(records, desc=src_name, unit="sample")
            for rec in pbar:
                sample_id = str(rec.get("sample_id", rec.get("image_filename", "?")))
                if (src_name, sample_id) in done_ids:
                    continue

                question = rec.get("question", "")
                response = rec.get("response", "")
                if not response.strip():
                    continue

                t0 = time.time()

                if args.split_mode == "rules":
                    coarse_steps = extract_steps(response)
                    if not coarse_steps:
                        continue
                    fragments_text = apply_rules_split(coarse_steps)
                    labels = classify_fragments(client, args.model, question, fragments_text)
                    if labels is None:
                        print(f"  [skip] {sample_id} — classification failed after retries")
                        continue
                    labeled = [{"text": t, "label": l} for t, l in zip(fragments_text, labels)]
                    n_coarse = len(coarse_steps)
                else:  # llm
                    labeled = split_and_classify(client, args.model, question, response)
                    if labeled is None:
                        print(f"  [skip] {sample_id} — split+classify failed after retries")
                        continue
                    n_coarse = len(labeled)

                elapsed = time.time() - t0

                for idx, frag in enumerate(labeled):
                    out_f.write(json.dumps({
                        "source_file": src_name,
                        "sample_id": sample_id,
                        "question": question,
                        "original_chain": response,
                        "step_index": idx,
                        "step_text": frag["text"],
                        "label": frag["label"],
                        "split_mode": args.split_mode,
                    }) + "\n")
                out_f.flush()

                total_fragments += len(labeled)
                total_samples += 1
                done_ids.add((src_name, sample_id))

                sleep_needed = max(0.0, min_sleep - elapsed)
                if sleep_needed > 0:
                    time.sleep(sleep_needed)

                label_counts = {k: sum(1 for f in labeled if f["label"] == k) for k in ("plan", "observe", "deduce")}
                pbar.set_postfix({
                    "frags": len(labeled),
                    "plan": label_counts["plan"],
                    "obs": label_counts["observe"],
                    "ded": label_counts["deduce"],
                })

    print(f"\nDone. {total_samples} samples, {total_fragments} fragments written to {args.output}")


if __name__ == "__main__":
    main()
