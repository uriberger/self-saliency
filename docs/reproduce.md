# Reproducing the paper

One row per table and figure in *Look Where You Say You're Looking: Self-Grounded
Attention for Visual Reasoning*. Every arm below was identified by **matching its
score**, not by matching its name: the "source run" column is the run directory in
the archive repo (`research/saliency_r1`) whose 25-benchmark mean reproduces the
published number.

## The arms

| Paper column | Mean score | Source run (archive) | Config here |
|---|---|---|---|
| Qwen3-VL-8B-Instruct | 61.14 | `qwen3_vl_8b_instruct_mnt4096` | no training |
| VGA | 61.16 | `qwen3_vl_8b_instruct_mnt4096_vga_b0.2_l4-16` | `baselines/vga/` |
| Coldstart | 62.70 | `coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged` | `training/coldstart/configs/qwen3_vl_8b.yaml` |
| EASE | 62.88 | `ease_8k_v2-step124_merged` | `baselines/ease/` |
| SELF-SALIENCY_mean | 63.17 | `…overlap__wov0.033_2head_trmean_saliency_r1_8k_mean_in_v2` | `training/grpo/configs/self_saliency_mean.yaml` |
| question boxes | 63.33 | `…-question-boxes` | `training/grpo/configs/question_boxes.yaml` |
| center rect | 63.38 | `…-rect-frac` | `training/grpo/configs/center_rect.yaml` |
| No-Sal | 63.45 | `…-no-saliency_saliency_r1_8k` | `training/grpo/configs/no_sal.yaml` |
| Saliency-R1 | 63.59 | `…-saliency-r1-qwen3` | `training/grpo/configs/saliency_r1.yaml` |
| **SELF-SALIENCY** | **64.26** | `…overlap__wov0.4_2head_trmean` | `training/grpo/configs/self_saliency.yaml` |

The published mean for *center rect* is 63.39 against 63.38 recomputed; the gap is
rounding in the MME `/2800` rescale, not a different run.

Two means above differ from the printed paper. LogicVista was re-scored against the
pinned answer reader, moving Coldstart 54.91 -> 55.13 and No-Sal 55.13 -> 54.69, hence
62.69 -> 62.70 and 63.47 -> 63.45. Nothing reorders and SELF-SALIENCY is untouched. See
[rescore-audit.md](rescore-audit.md).

The EASE row carries a DAPO reference that appears in no table and without which the EASE
number means nothing (App D.3), because EASE is DAPO-in-EasyR1 plus an attention loss
while our arms are GRPO-in-TRL — so EASE against an arm of ours compares the method and
the framework at once. The quantity that means something is

    (EASE − DAPO) inside EasyR1    vs    (SELF-SALIENCY − No-Sal) inside ours

so run both arms, from the same cold start on the same data:

```bash
bash baselines/ease/run.sh --arm ease --exp ease_8k
bash baselines/ease/run.sh --arm dapo --exp dapo_8k
```

It is `--arm`, not a config file of ours: EasyR1 takes its own `examples/config.yaml` and
every deviation is a `key=value` override, so the arm is the override list in `run.sh`.
Pass `--no-judge` to both arms or to neither — a judged EASE arm against a rule-scored
DAPO arm confounds the attention loss with the reward, which is the one thing the paired
design exists to prevent.

## The flagship, exactly

`training/grpo/configs/self_saliency.yaml` encodes, and these were read back off the
released adapter rather than off a command line:

| | |
|---|---|
| base | `coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged` |
| corpus | `peterant330/saliency-r1-8k`, `train_test_split(test_size=100, seed=42)` |
| LoRA | rank 16, α 32, dropout 0.05, on `q_proj` + `v_proj` |
| α_sal | 0.4, φ (max-normalised; Eq. 1) |
| heads | layer 22, heads 28 and 31; mean token reduction |
| boxes | Grounding-DINO, `box_threshold 0.10`, per-box area cap 0.5 |
| steps | 3,990 (three epochs, 6 prompts × G=8 = 48 completions/step) |

## Table by table

| Artifact | Command |
|---|---|
| Table 1 (the suite) | `evaluation/suite.yaml` is the list, with App A.3's splits |
| Table 2, Table 5, Table 6, Table 7 | `python evaluation/tables.py` |
| Table 3 (§4.4) | `bash experiments/trained_model/run.sh --out-dir DIR --arm coldstart=CKPT --arm self_saliency=CKPT` |
| Table 4, Figure 4, Table 8 (§5) | `bash experiments/attention_bias/run.sh --stage corpus\|selftest\|scan\|report --out-dir DIR` |
| Figure 3 | `python -m experiments.figures.steps_figure` |
| Figure 5 | `python -m experiments.attention_bias.tables --panels ...` |
| §3.5 head selection | `bash experiments/head_selection/run.sh --out-dir DIR` — selects (22,28) and (22,31) |
| App C's α_sal_mean = 0.033 | `python -m experiments.alpha_calibration <probe_merged.json>` |
| App A.1 classifier (91.9% / 93%) | `python -m selfsal.steps.evaluate` |

**Four tables, one command.** `evaluation/tables.py` reads the banked `*_results.json`
under `evaluation/results/` and prints every arm it finds, which is exactly the ten of the
table above — the tree carries the paper's runs and nothing else. Tables 2, 5, 6 and 7 are
four readings of those ten rows, not four invocations: Table 5 is the ablation subset,
Table 7 adds SELF-SALIENCY_mean, and Table 6 is Table 2 with error bars. Pass
`--bootstrap` for App B's bars, which needs the per-sample `*_samples_*.jsonl` files —
about 5 GB, not in git, and a release asset (see [publishing.md](publishing.md)).
