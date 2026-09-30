# Self-Saliency

Code for *Look Where You Say You're Looking: Self-Grounded Attention for Visual
Reasoning*.

SELF-SALIENCY is a post-training method for vision-language models. During GRPO, the
policy's own reasoning chain is segmented into steps; each *observe* step is grounded
with a phrase-grounding model; and the policy is rewarded for placing its visual
attention inside the regions its own sentences just named. Prior attention-steering
work fixes the target regions from the image and question alone — here they are
derived from the text the policy is generating, which is what makes the two pressures
in Figure 1 complementary: say things that can be grounded, and look at what you say.

Start at **[docs/reproduce.md](docs/reproduce.md)**: one row per table and figure, each
pointing at the config that produced it. [docs/install.md](docs/install.md) is the setup,
and [docs/provenance.md](docs/provenance.md) says where every file came from and what was
checked against what.

```bash
pytest                                              # 2,300+ CPU tests, no GPU
python -m training.grpo.config --all                # the six arms of the paper
bash training/grpo/run.sh self_saliency --dry-run   # the plan, without launching
python evaluation/tables.py                         # rebuild Table 2 from banked results
```

## Layout

```
selfsal/          the method, as an importable package
  steps/          §3.2 + App A.1  chain -> atomic steps -> plan/observe/deduce/none
  grounding/      §3.3            Grounding-DINO boxes -> union mask on the patch grid
  saliency/       §3.3            attention -> patch map; phi (Eq. 1) and phi_mean (App C)
  models/         §5              what differs between the four VLM families
  data/                           saliency-r1-8k + its seed-42 holdout; the §5 corpus

training/
  coldstart/      §4.1, App A.2   LLaVA-CoT + Mulberry SFT via LLaMA-Factory
  grpo/           §3.4, App A.2   the GRPO trainer, and one config per paper arm

experiments/
  head_selection/ §3.5            why layer 22, heads 28 and 31
  attention_bias/ §5              the border-ring result across four VLMs
  trained_model/  §4.4            does the text move, or does the attention?

baselines/        App D           VGA, Saliency-R1, EASE (+ the DAPO reference)
evaluation/       §4.2, App A/B   the 25-benchmark suite, its scorers, its error bars
```

`selfsal/` is one package on purpose. The reward the policy is trained against and the
screen that selected its two attention heads must compute the *same* φ, or §3.5 does
not justify §3.4 — so both import `selfsal.saliency.score`, rather than each carrying
its own copy as they did while the work was in progress.

## Install

Four environments, because the pins genuinely conflict (EASE requires
`transformers<=4.57`; the trainer requires a 5.x dev build). See
[docs/install.md](docs/install.md).

| env | runs |
|---|---|
| `selfsal-grpo` | GRPO training, every attention probe, §5 |
| `selfsal-sft` | the cold start, via LLaMA-Factory |
| `selfsal-eval` | the 25-benchmark suite, and VGA |
| `selfsal-ease` | the EASE and DAPO baselines |

Three external repos are cloned and patched at install time rather than vendored:
TRL, LLaMA-Factory, and EasyR1/verl. `lmms-eval` is a pinned submodule — the suite
needs tasks and answer-parser fixes that are not upstream.

Training and the benchmarks that score by answer extraction both call GPT-4o mini as a
judge, so export `OPENAI_API_KEY`. Without it training masks that reward rather than
scoring it zero, and those benchmarks fall back to exact matching and under-report.
[docs/install.md §4](docs/install.md) has the rest, including how to point it at another
OpenAI-compatible endpoint.

## Status

Private, pending review. The paper is under double-blind review and promises code and
weights on acceptance; **[docs/publishing.md](docs/publishing.md) is what has to happen
first**, starting with the fact that the git history is authored under a real name.

## Derived work, and its licences

This repository is Apache-2.0 (see [LICENSE](LICENSE)). Parts of it are derived from other
people's work, all Apache-2.0-compatible, and are marked as such in their own headers:

| here | derived from |
|---|---|
| `training/grpo/trl_patch/` | HuggingFace **TRL** (Apache-2.0). `grpo_trainer_qwen3.py` and `grpo_vlm_qwen3.py` are modified copies of its GRPO trainer and VLM example; the files carry TRL's copyright header and `env/patches/trl.sh` installs them into a checkout rather than vendoring one |
| `baselines/saliency_r1/reward.py` and `training/grpo/trl_patch/rewards/saliency_r1.py` | the **Saliency-R1** authors' reward, from [their repository](https://github.com/peterant330/Saliency_R1), kept as released so the Appendix D.2 arm differs from ours in the attention term alone |
| `baselines/ease/` | **EASE** on top of **EasyR1**/**verl** (Apache-2.0). The method files are untouched; what is here patches their reward interface and an import, and drives their trainer |
| `baselines/vga/` | **VGA**, reimplemented as an inference-time patch on Qwen3-VL (Appendix D.1) |
| `evaluation/lmms_eval` | **lmms-eval** (Apache-2.0), a pinned submodule of a fork carrying task definitions and answer-parser fixes that are not upstream |
| `training/coldstart/` | driven by **LLaMA-Factory** (Apache-2.0), cloned rather than vendored |

The corpora are used under their own terms: Saliency-R1-8K, Visual-CoT, LLaVA-CoT,
Mulberry-SFT and the 25 benchmarks of [Table 1](docs/reproduce.md). Grounding-DINO and
Qwen3-VL-8B-Instruct are used under theirs.

## Citation

```bibtex
@inproceedings{selfsaliency,
  title     = {Look Where You Say You're Looking: Self-Grounded Attention for Visual Reasoning},
  booktitle = {Under review},
  year      = {2027}
}
```
