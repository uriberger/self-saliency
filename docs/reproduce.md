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
| Coldstart | 62.69 | `coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged` | `training/coldstart/configs/qwen3_vl_8b.yaml` |
| EASE | 62.88 | `ease_8k_v2-step124_merged` | `baselines/ease/` |
| SELF-SALIENCY_mean | 63.17 | `…overlap__wov0.033_2head_trmean_saliency_r1_8k_mean_in_v2` | `training/grpo/configs/self_saliency_mean.yaml` |
| question boxes | 63.33 | `…-question-boxes` | `training/grpo/configs/question_boxes.yaml` |
| center rect | 63.38 | `…-rect-frac` | `training/grpo/configs/center_rect.yaml` |
| No-Sal | 63.47 | `…-no-saliency_saliency_r1_8k` | `training/grpo/configs/no_sal.yaml` |
| Saliency-R1 | 63.59 | `…-saliency-r1-qwen3` | `training/grpo/configs/saliency_r1.yaml` |
| **SELF-SALIENCY** | **64.26** | `…overlap__wov0.4_2head_trmean` | `training/grpo/configs/self_saliency.yaml` |

The published mean for *center rect* is 63.39 against 63.38 recomputed; the gap is
rounding in the MME `/2800` rescale, not a different run.

Two arms carry a DAPO reference that never appears in a table but without which the
EASE number means nothing (App D.3): `baselines/ease/configs/dapo.yaml`. Run it.

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
| Table 2, Table 6 | `evaluation/tables.py --arms paper` (reads banked `results.json`; `--bootstrap` for App B's error bars, which needs the per-sample release asset) |
| Table 3 (§4.4) | `experiments/trained_model/run.sh` |
| Table 4, Figure 4, Table 8 (§5) | `experiments/attention_bias/run.sh` |
| Table 5 (ablations) | `evaluation/tables.py --arms ablation` |
| Table 7 (App C) | `evaluation/tables.py --arms appendix-c`; α_sal_mean = 0.033 is re-derived by `experiments/alpha_calibration.py` |
| Figure 3 | `experiments/figures/steps_figure.py` |
| Figure 5 | `experiments/attention_bias/tables.py --panels` |
| §3.5 head selection | `experiments/head_selection/run.sh` — selects (22,28) and (22,31) |
| App A.1 classifier (91.9% / 93%) | `selfsal/steps/evaluate.py` |
