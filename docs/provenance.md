# Provenance

Where every file here came from. This repo was assembled from two working repos that
stay in place as the archive:

* `A` = `research/saliency_r1` — training, rewards, the §5 attention analysis
* `V` = `research/vlm_reasoning` — head selection, the steps classifier, evaluation

Work that did not reach the paper (the glimpse / gradient / AUROC reward variants, the
placebo, mismatch, mask-free and length-guard controls, LASER, the RoPE-phase probes,
the sink-shift edits, the flow and intervention probes, the set_a–set_e corpus builders)
stays in `A` and `V` and is deliberately **not** here.

`[ ]` = not yet ported · `[x]` = ported and importable

## selfsal/ — the method

| Here | From | Notes |
|---|---|---|
| `steps/segment.py` | `A trl/overlap_steps.py` + `V grpo/reward.py` | merged; these were two copies of one thing |
| `steps/classifier.py` | `V steps_classifier/train_classifier.py` (model def) | shared by training and inference |
| `steps/train.py` | `V steps_classifier/train_classifier.py` | |
| `steps/make_data.py` | `V steps_classifier/generate_data.py` | App A.1's distillation |
| `steps/evaluate.py` | `A eval_steps_classifier.py` | the 91.9% / 93% of fn. 9 |
| `grounding/dino.py` | `A trl/rewards/overlap_rewards.py` (`_dino_boxes`, `_union_mask`) + `V step1/grounding.py` | merged |
| `grounding/server.py` | `A serve_grounding_dino.py` | the reward's sidecar |
| `saliency/maps.py` | `A trl/rewards/overlap_rewards.py` + `V vlm/saliency.py` | merged |
| `saliency/score.py` | `A trl/rewards/overlap_rewards.py` (`_step_score`) | φ and φ_mean |
| `saliency/heads.py` | new | the (22,28)/(22,31) constant, one place |
| `models/families.py` | `A vlm_family.py` | §5's four backbones |
| `data/saliency_r1_8k.py` | `A trl/grpo_vlm_qwen3.py` (loader + seed-42 split) | |
| `data/boxed_corpus.py` | `A build_boxed_corpus.py` | §5's 1,800 VisualCoT pairs |

## training/

| Here | From |
|---|---|
| `coldstart/configs/qwen3_vl_8b.yaml` | `A train/cold_start/qwen3_vl_8b_instruct_sft/train.yaml` |
| `coldstart/prepare_data.py` | `A cold_data/{dl_llavacot,dl_mulberry,extract_needed,sanitize_to_jsonl,split_records_to_jsonl}.py` |
| `coldstart/run.sh` | `A launch_coldstart_job.sh` |
| `grpo/trl_patch/grpo_trainer_qwen3.py` | `A trl/grpo_trainer_qwen3.py` |
| `grpo/trl_patch/grpo_vlm_qwen3.py` | `A trl/grpo_vlm_qwen3.py` |
| `grpo/trl_patch/rewards/self_saliency.py` | `A trl/rewards/overlap_rewards.py` |
| `grpo/trl_patch/rewards/saliency_r1.py` | `A trl/rewards/saliency_rewards.py` |
| `grpo/trl_patch/rewards/{format,answer,judge}.py` | `A trl/rewards/{format,answer_format,openai}_rewards.py` |
| `grpo/trl_patch/{scripts,models}/utils.py` | `A trl/{scripts,models}/utils.py` |
| `grpo/run.sh` + `grpo/configs/*.yaml` | `A launch_grpo_qwen3_overlap_colocated_job.sh`, split |
| `grpo/precompute_question_boxes.py` | `A precompute_question_boxes.py` |

The 2,377-line launcher parsed 72 flags, roughly 47 of them for experiments outside
the paper. Here the six paper arms are six YAML files and the runner is thin.

## experiments/

| Here | From |
|---|---|
| `head_selection/{collect,correlate}.py` | `V analysis/aggregation_correlation.py`, split by stage |
| `head_selection/cross_dataset.py` | `V analysis/cross_dataset_head_transfer.py` (fn. 3) |
| `head_selection/generate.py` | `V run_experiment.py` |
| `attention_bias/*` | `A sink_location{,_probe,_xmodel_tables,_html}.py`, `A sink_{box_coverage,three_legs,observe_boxes,encoder_probe}.py` |
| `trained_model/{probe,audit}.py` | `A overlap_probe.py`, `A selfground_audit.py` |
| `figures/steps_figure.py` | `A fig1_steps_figure.py` |
| `figures/saliency_viz.py` | `A saliency_viz.py`, `A saliency_viz_compare.py` |
| `alpha_calibration.py` | `A overlap_metric_spread.py` — App C's α_sal_mean = 0.033 |
| `center_rect_calibration.py` | `A centre_box_probe.py` — where 0.565 comes from |

## baselines/

| Here | From |
|---|---|
| `vga/` | `V vlm/vga.py`, `V lmms_eval_plugin/qwen3_vl_vga.py`, `V install_lmms_vga.sh`, `V analysis/vga_*.py` |
| `saliency_r1/` | `A trl/rewards/saliency_rewards.py` + a GRPO config |
| `ease/` | `A ease/`, `A {setup_ease_env,patch_ease_repo,launch_ease_train*,prepare_ease_saliency_data,download_ease_sources,merge_ease_checkpoint,stage_ease_checkpoint}.sh`, `A {export_saliency_r1_8k_for_ease,add_ease_prompt_length,verify_ease_setup}.py` |

## evaluation/

| Here | From |
|---|---|
| `lmms_eval/` | submodule → `uriberger/lmms-eval`, pinned |
| `suite.yaml` | `V scripts/lmms_eval_benchmarks.txt` + App A.3's splits |
| `run_suite.sh` | `V scripts/{run_lmms_eval_suite,eval_benchmark_suite,eval_our_benchmarks,eval_saliency_r1_benchmarks}.sh` |
| `stats.py` | `V visualize/eval_stats.py` — the bootstrap units of App B |
| `tables.py` | `V visualize/{results_table,rank_models}.py` |
| `bootstrap_check.py` | `V scripts/bootstrap_check.py` |
| `rescore/` | `V scripts/{rescore_*,reparse_and_rescore,recompute_correct}.py` |
| `results/` | `V results/lmms_eval/<arm>/**/*results.json`, paper arms only |

`stats.py` carries the non-obvious unit choices App B depends on: MME's unit is the
image (two questions each), HR-Bench's is the question rather than the row (four
cyclic option rotations), HallusionBench reads the judge's output file because its
sample rows hold no score. `tables.py` carries the parser-version stamp check — the
guard that caught one LogicVista checkpoint sitting in the table at both 0.45% and
56.2%.

## tests/

The CPU tests that cover kept code, as pytest: the reward, the two ablation masks, the
`trl_patch` import layout, launcher flag forwarding, §5, §4.4, and the eval scorers.

`test_import_layout.py` is load-bearing and not optional. `trl_patch/` is the tracked
source and the patched TRL clone is what executes, and the two have different layouts,
so a relative import that is correct where it is written can break where it lands.
Omitting one `cp` from the patch script once produced a failure that surfaced only on
a second cluster, mid-run, after generation.
