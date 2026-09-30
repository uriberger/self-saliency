# Provenance

Where every file here came from. This repo was assembled from two working repos that
stay in place as the archive:

* `A` = `research/saliency_r1` — training, rewards, the §5 attention analysis
* `V` = `research/vlm_reasoning` — head selection, the steps classifier, evaluation

Work that did not reach the paper (the glimpse / gradient / AUROC reward variants, the
placebo, mismatch, mask-free and length-guard controls, LASER, the RoPE-phase probes,
the sink-shift edits, the flow and intervention probes, the set_a–set_e corpus builders)
stays in `A` and `V` and is deliberately **not** here.

## Status

Everything below is ported. The repository is self-contained: no file reaches into the
archive repos, and `tests/test_no_archive_dependencies.py` fails if one starts to --
which matters because every archive path still exists on the machine this was ported on,
so a missed rewiring would run the archive's code and look correct here while failing
everywhere else. It caught eleven hardcoded paths.

Checked against the originals rather than eyeballed:

| what | how |
|---|---|
| phi, phi_mean, the regions, the rectangle | 600 randomised grids x 3 map dtypes x 6 filter settings |
| Appendix A.1 segmentation | 4,000 randomised chains |
| image preparation | 42 cases over 7 sizes x 6 image modes, pixel-identical |
| attention capture | the fused path, and that the explicit softmax reproduces it |
| the six arm configs | against `training_args.bin`, `adapter_config.json`, `trainer_state.json` and the reached step count |
| the 25-benchmark suite | all ten arms reproduce their published mean |

Four things the port turned up that were not visible from the outside:

1. **Training and the probes resized images with different filters**, and one of the two
   says so wrongly. The training path — `A trl/grpo_vlm_qwen3.py`, `A build_grpo_sets.py`
   and `A precompute_question_boxes.py` — passes `Image.BICUBIC`, so every image the
   published runs trained on, and every image the `question_boxes` arm grounded, is
   bicubic. The post-hoc probes — `A overlap_probe.py` (here
   `experiments/trained_model/probe.py`) and the RoPE-phase probes — pass `resize(..., 2)`
   with a trailing comment saying BICUBIC, and PIL's 2 is BILINEAR, so those measurements
   are bilinear. Both filters are preserved as they ran; neither file was "corrected",
   because following the comment would change every patch grid's contents and hence every
   map and every φ.

   The consequence to know about: `selfsal/data/prompt.py`'s `prepare_image` is the
   probes' filter. It has no callers today — `experiments/trained_model/probe.py` imports
   it and then shadows it with its own copy — and **the GRPO entry script must not adopt
   it**, which is why that script keeps its own bicubic resize with a comment saying so.
   Unifying the two is a decision about which published numbers to move, not a cleanup.
2. **`phi` and `phi_mean` ran at different precisions** on the same map -- one divided in
   the map's dtype, the other had already upcast. Both are float64 now.
3. **Head selection and the reward segment chains differently.** See
   `selfsal/steps/segment.py`; both segmenters are here and the seam is recorded.
4. **The Saliency-R1 arm generated in-process, not through vLLM** (`use_vllm=False`),
   which is how all eight of its GPUs could train.

What deliberately did NOT come across, and is intact in the archive: the gradient and
GLIMPSE saliency maps, the AUROC and roll-null metrics, the placebo, mask-free,
mismatched-box and length-guard controls, the attention-intervention and sink-shift
experiments, the RoPE-phase probes, and the set_a-set_e corpus builders.

HOW THE PROBES REACH WHAT DID COME ACROSS. The archive was flat, so its scripts reached
each other by file name — `spec_from_file_location("_x", REPO / "sink_location_probe.py")`
— and several kept doing that after the port, naming files that do not exist here. They
are ordinary imports now. The mapping, which is the one to read when a traceback still
names an old file:

| archive | here |
|---|---|
| `overlap_probe.py` | `experiments/trained_model/probe.py` |
| `sink_location{,_probe,_xmodel_tables,_html}.py` | `experiments/attention_bias/{measure,probe,tables,report_html}.py` |
| `trl/overlap_steps.py` | `selfsal/steps/segment.py` (`segment_observe_steps` → `segment_sentences`, `OverlapStepsClassifier` → `StepClassifier`) |
| `trl/rewards/overlap_rewards.py` | `selfsal/{grounding,saliency}` — `_dino_boxes` → `ground`, `_union_mask` → `union_mask`, `_mean_in`/`_mean_in_v2`/`_auroc` → `phi`/`phi_mean`/`auroc` |
| `V analysis/aggregation_correlation.py` | `experiments/head_selection/screen.py` |
| `intervene_probe.py` (`Progress`, `monitor` only) | `experiments/_progress.py` |
| `trl/rewards/roll_null.py` (`inframe_offsets`, `sample_offsets` only) | `selfsal/grounding/mask.py` |

The last two are the pattern for a module that did not come across whole: the geometry a
probe needs is taken, the reward it belonged to is not. `sample_offsets` was checked
against the archive's over 400 random masks and is identical.

Where the missing thing has no equivalent — the gradient and GLIMPSE maps, the
attention-rollout flow, the three control rewards — the probe stands it up as an
`experiments/_unavailable.py` sentinel, so `--map attn` works and `--map grad` raises with
a sentence naming the archive. That pattern was already here and was broken in three
places: a sentinel read as a function default or an `add_argument` default is evaluated at
import and parser-build time respectively, which made `experiments/trained_model/probe.py`
unimportable — so the one map in the paper was as dead as the two that are not here.
`tests/test_dropped_variants_fail_late.py` now enforces it.

Two WandB callbacks in the GRPO entry script went with them: `ValidationAccuracyCallback`
(one greedy completion per held-out prompt, scored on answer accuracy) and
`BenchmarkResultsCallback` (mini-benchmark scores produced out of process). Both read
inputs that only the dropped builders produce — `A build_grpo_sets.py --build-val` and
`A run_bench_eval.sh` / `A eval_mini/` — so neither could be satisfied from this
repository, and neither is reachable from any of the six arm configs. Their `--val_sets_dir`
and `--val_eval_steps` flags went too. Training is byte-identical without them; the one
thing worth carrying forward was the note on why the vLLM prompt needs an explicit image
placeholder, which now sits at the conversion site in `grpo_trainer_qwen3.py`.

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
| `data/prompt.py` | the reward, the probe and the launcher | three copies, now one |
| `data/saliency_r1_8k.py` | `A trl/grpo_vlm_qwen3.py` (loader + seed-42 split) | also sets `config.py`'s step count |
| `data/question_boxes.py` | `A trl/rewards/overlap_rewards.py` (the cache format) | shared with the builder |
| `data/boxed_corpus.py` | `A build_boxed_corpus.py` | §5's 1,800 VisualCoT pairs |
| `judge.py` | `A trl/rewards/openai_rewards.py` | R_llm, shared with re-scoring |

## training/

| Here | From |
|---|---|
| `coldstart/configs/qwen3_vl_8b.yaml` | `A train/cold_start/qwen3_vl_8b_instruct_sft/train.yaml` |
| `coldstart/data/{dl_llavacot,dl_mulberry,extract_needed,sanitize_to_jsonl,split_records_to_jsonl}.py` | the same names under `A cold_data/` |
| `coldstart/submit.sh` | `A launch_coldstart_job.sh` |
| `grpo/trl_patch/grpo_trainer_qwen3.py` | `A trl/grpo_trainer_qwen3.py` |
| `grpo/trl_patch/grpo_vlm_qwen3.py` | `A trl/grpo_vlm_qwen3.py` |
| `grpo/trl_patch/rewards/self_saliency.py` | `A trl/rewards/overlap_rewards.py` |
| `grpo/trl_patch/rewards/saliency_r1.py` | `A trl/rewards/saliency_rewards.py` |
| `grpo/trl_patch/rewards/{format,answer}.py` | `A trl/rewards/{format,answer_format}_rewards.py` |
| `grpo/trl_patch/{scripts,models}/utils.py` | `A trl/{scripts,models}/utils.py` |
| `grpo/run.sh` + the six `grpo/configs/` YAMLs | `A launch_grpo_qwen3_overlap_colocated_job.sh`, split |
| `grpo/precompute_question_boxes.py` | `A precompute_question_boxes.py` |

R_llm is not in that table: it is `selfsal/judge.py` (from `A trl/rewards/openai_rewards.py`),
because the offline re-scoring tools need the same judge and must not reach into a TRL
checkout to get it. `selfsal/data/question_boxes.py` moved out of the rewards package for
the same reason — it is a FILE FORMAT, and `grpo/precompute_question_boxes.py` writes what
the reward reads.

The 2,377-line launcher parsed 72 flags, roughly 47 of them for experiments outside
the paper. Here the six paper arms are six YAML files and the runner is thin.

## experiments/

| Here | From |
|---|---|
| `head_selection/screen.py` | `V analysis/aggregation_correlation.py` |
| `head_selection/cross_dataset.py` | `V analysis/cross_dataset_head_transfer.py` (fn. 3) |
| `head_selection/generate.py` | `V run_experiment.py` |
| `head_selection/run.sh` | `V scripts/slurm/launch_agg_corr_job.sh`, minus the cluster |
| `attention_bias/run.sh` | `A launch_sink_location{,_job}.sh`, minus the cluster |
| `trained_model/run.sh` | `A launch_overlap_probe{,_job}.sh` + `A launch_selfground_audit{,_job}.sh` |
| `_progress.py` | `A intervene_probe.py` (`Progress`, `monitor` only) |
| `_unavailable.py` | new | the sentinel for a dropped arm's module |
| `attention_bias/*` | `A sink_location{,_probe,_xmodel_tables,_html}.py`, `A sink_{box_coverage,three_legs,observe_boxes,encoder_probe}.py` |
| `trained_model/{probe,audit}.py` | `A overlap_probe.py`, `A selfground_audit.py` |
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
