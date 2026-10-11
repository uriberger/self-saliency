# Attribution

This repository is released under Apache-2.0 (see [LICENSE](LICENSE)), copyright NVIDIA
CORPORATION & AFFILIATES. It contains and depends on work by others. This file is the
complete list; [README.md](README.md#derived-work-and-its-licences) is the short version.

Everything below is under a permissive licence compatible with Apache-2.0 distribution.

---

## 1. Third-party code **in** this repository

### 1.1 HuggingFace TRL — `training/grpo/trl_patch/`

| | |
|---|---|
| **Upstream** | https://github.com/huggingface/trl |
| **Version** | `v0.21.0` (the tag `v0.21-release`, which `docs/install.md` §2 pins) |
| **Licence** | Apache-2.0 |
| **Copyright** | Copyright 2020-2025 The HuggingFace Team. All rights reserved. |
| **Form** | Modified copies of seven upstream files. TRL itself is cloned at install time, not vendored; `env/patches/trl.sh` copies these over the clone. |

Every file retains TRL's copyright and Apache-2.0 block and carries `# Modified by NVIDIA`
with NVIDIA's SPDX header beneath it.

| File here | Upstream file | Upstream lines | Lines here | Diff |
|---|---|---|---|---|
| `grpo_trainer_qwen3.py` | `trl/trainer/grpo_trainer.py` | 2,063 | 2,979 | +941 / −25 |
| `grpo_vlm_qwen3.py` | `examples/scripts/grpo_vlm.py` | 208 | 312 | +181 / −77 |
| `scripts/utils.py` | `trl/scripts/utils.py` | 284 | 461 | +178 / −1 |
| `models/utils.py` | `trl/models/utils.py` | 449 | 459 | +12 / −2 |
| `rewards/__init__.py` | `trl/rewards/__init__.py` | 32 | 25 | +25 / −32 |
| `rewards/format.py` | `trl/rewards/format_rewards.py` | 50 | 60 | +14 / −4 |
| `rewards/answer.py` | `trl/rewards/format_rewards.py` | 50 | 59 | +12 / −3 |

**What NVIDIA changed:**

* `grpo_trainer_qwen3.py` — the largest change. Adds the attention re-forward that makes
  the saliency reward computable: capture of attention weights during or after generation,
  the per-observe-step scoring path, the reward-masking policy when the LLM judge is
  unreachable, and the diagnostics the arms are checked against.
* `grpo_vlm_qwen3.py` — rewritten as this paper's entry point: the corpus and its seed-42
  holdout, the system prompt, the reward list and its positional ordering, and the saliency
  reward's configuration.
* `scripts/utils.py` — adds this repository's flags to `ScriptArguments`
  (`reforward_saliency`, `reward_variant`, `overlap_*`, `token_reduction`, the
  Grounding-DINO knobs). The upstream block above them is unchanged.
* `models/utils.py` — **effectively verbatim upstream.** One behavioural change: the
  ZeRO-3 hook re-registration in `unwrap_model_for_generation` is wrapped in `try/finally`
  so hooks are restored if the yielded block raises. Note that this upstream file itself
  contains code TRL attributes to `accelerate` and to PyTorch Lightning, both Apache-2.0.
* `rewards/__init__.py` — rewritten: eager imports instead of TRL's `_LazyModule`, and it
  exports this repository's rewards.
* `rewards/format.py`, `rewards/answer.py` — upstream's `think_format_reward` adapted to
  take a precomputed validity list; the upstream implementation is retained in-file.

### 1.2 HuggingFace transformers — `env/patches/transformers_patch/sdpa_attention_tf5x.py`

| | |
|---|---|
| **Upstream** | https://github.com/huggingface/transformers |
| **Upstream file** | `transformers/integrations/sdpa_attention.py` |
| **Commit** | `612c371ac6e4f9e86baf9b5cf4b8048f5ca80774` (5.13.0.dev0), the build `env/grpo.txt` pins |
| **Licence** | Apache-2.0 |
| **Copyright** | Copyright 2020-2025 The HuggingFace Team. All rights reserved. |
| **Form** | Modified copy. 104 upstream lines; about 100 of them survive verbatim. `env/patches/transformers.sh` installs it over the stock file in the `selfsal-grpo` environment. |

**What NVIDIA changed:** one thing. `sdpa_attention_forward` returns the attention
probability matrix as its second value instead of `None`, by calling SDPA a second time
with `V` replaced by the identity — `softmax(QKᵀ) @ I == softmax(QKᵀ)`. Stock SDPA never
materialises that matrix, so without this the saliency reward and all of Section 5 have
nothing to read.

### 1.3 lmms-eval — `evaluation/lmms_eval` (git submodule)

| | |
|---|---|
| **Upstream** | https://github.com/EvolvingLMMs-Lab/lmms-eval |
| **Fork used** | https://github.com/uriberger/lmms-eval, branch `paper-pin` |
| **Commit** | `4ea4f151e1ddd41e194a2d7f18a84a9ca50cfe45`, branched from upstream `a9a806b` |
| **Licence** | **MIT** for the pipeline (inherited from `lm-evaluation-harness`) **and Apache-2.0** for `lmms_eval/tasks` and `lmms_eval/models`, per its own `LICENSE` |
| **Copyright** | Copyright (c) 2025 LMMs-Lab |
| **Form** | Unmodified submodule of a fork. 3,338 files. No file in it is edited by this repository. |

The fork carries ~13 commits over upstream: task definitions and answer-parser fixes that
are not upstream, plus the parser-version stamp. See [docs/rescore-audit.md](docs/rescore-audit.md).

A `--recurse-submodules` clone also brings in nested third-party licences that belong to
lmms-eval, not to this repository: `lmms_eval/tasks/mdpbench/LICENSE-MDPBENCH`,
`lmms_eval/tasks/vbvr/vbvr_bench/LICENSE`, `lmms_eval/tasks/mdpbench/cdm/LICENSE`,
`.../cdm/modules/tokenize_latex/LICENSE-IM2MARKUP`, and
`.../tokenize_latex/third_party/katex/LICENSE.txt`.

---

## 2. Third-party code obtained at install time — **not in this repository**

Cloned and, where noted, patched by scripts here. No source from these is committed.

| Project | Upstream | Version / commit | Licence | Used by |
|---|---|---|---|---|
| **TRL** | https://github.com/huggingface/trl | tag `v0.21-release` | Apache-2.0 | the GRPO trainer; `env/patches/trl.sh` copies §1.1 into the clone |
| **LLaMA-Factory** | https://github.com/hiyouga/LLaMA-Factory | `76a0391dddc07741ff3e8fa2c82ebed106508280` | Apache-2.0 | the cold-start SFT (`training/coldstart/`) |
| **EASE** (on EasyR1/verl) | https://github.com/Nrich-sunny/Attend-to-Evidence | `b90f83301c9c6738651ef151da6dbb696d5bcaa7` | Apache-2.0 | the EASE and DAPO baselines (`baselines/ease/`) |

`baselines/ease/patch_repo.sh` makes two additive edits to the EASE clone — extra keys in
the reward-function payload, and an import fallback in `verl/utils/vllm_utils.py`. It
leaves `verl/workers/actor/` — the EASE method itself — byte-identical, and
`baselines/ease/reward_function/judged_perception.py` *imports* their
`examples/reward_function/perception.py` rather than copying it.

---

## 3. Independent reimplementations — no third-party code taken

| Here | Reimplements | Note |
|---|---|---|
| `baselines/vga/` | *Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention*, CVPR 2026, arXiv:2511.20032 | Written from the paper as an inference-time hook on Qwen3-VL. No upstream VGA source is used; upstream's approach (a forked modeling file and sampler) is deliberately not the approach here. `qwen3_vl_vga.py` subclasses lmms-eval's `Qwen3_VL`, which is an API dependency on §1.3, not a copy. |

---

## 4. Files of unconfirmed origin

This file carries **no NVIDIA copyright header**, deliberately, because authorship has not
been confirmed. Do not assume it is NVIDIA's.

| Path | What | Status |
|---|---|---|
| `assets/validate_palette.js` | Data-visualisation palette validator (OKLCH / OKLab / CVD / WCAG checks) | Arrived from the archive repository. Its docstring refers to checks "the skill enforces", which suggests generation from or adaptation of a tooling template. **Origin unconfirmed.** |

See [docs/publishing.md](docs/publishing.md) §6.

---

## 5. Models and corpora

Used under their own terms; none are redistributed by this repository.

| | Source | Used for |
|---|---|---|
| Qwen3-VL-8B-Instruct / -Thinking | `Qwen/Qwen3-VL-8B-*` | the base policy, and evaluation |
| Grounding-DINO | `IDEA-Research/grounding-dino-base` | phrase grounding for the saliency reward (§3.3) |
| FLAN-T5 | `google/flan-t5-base` | encoder of the step classifier (App. A.1) |
| InternVL3.5-8B, Nemotron Nano VL | `OpenGVLab/InternVL3_5-8B`, `nvidia/...` | the Section 5 cross-model scan |
| GPT-4o mini | OpenAI API | the LLM judge (`R_llm`, answer extraction) |
| Gemini 2.5 Pro | via an OpenAI-compatible gateway | distils the step classifier's training labels (`selfsal/steps/make_data.py`) |
| saliency-r1-8k | `peterant330/saliency-r1-8k` | the GRPO training corpus |
| LLaVA-CoT-100k, Mulberry-SFT | `Xkev/LLaVA-CoT-100k`, `HuanjinYao/Mulberry-SFT` | the cold start (§4.1) |
| Visual-CoT | `deepcs233/Visual-CoT` | the Section 5 boxed corpus |
| the 25 benchmarks of Table 1 | see [docs/reproduce.md](docs/reproduce.md) | evaluation |

Benchmark and corpus licences are the publishers'. Where `baselines/ease/download_sources.sh`
pulls additional corpora it records each one's licence inline: `inclusionAI/ZwZ-RL-VQA`
(Apache-2.0), `TIGER-Lab/ViRL39K` (MIT), `Cusyoung/SpaCE-10` (MIT).
