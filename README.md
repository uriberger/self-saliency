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
  figures/                        Figures 3 and 5

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

## Status

Private, pending review. The paper is under double-blind review and promises code and
weights on acceptance; **[docs/publishing.md](docs/publishing.md) is what has to happen
first**, starting with the fact that the git history is authored under a real name.

## Citation

```bibtex
@inproceedings{selfsaliency,
  title     = {Look Where You Say You're Looking: Self-Grounded Attention for Visual Reasoning},
  booktitle = {Under review},
  year      = {2027}
}
```
