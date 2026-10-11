# Before this repository is made public

The paper is under double-blind review. Its first page reads *"Anonymous authors — Paper
under double-blind review"*, and the reproducibility statement says code and weights are
released **upon acceptance**. This repository is therefore **private**, and publishing it
as it stands would deanonymise the submission.

Personal identifiers have already been removed and `tests/test_no_private_paths.py` keeps
them out: no usernames, no home directories, no cluster storage layout, no internal
project account. That sweep touched 293 files, 277 of them banked evaluation results,
where lmms-eval records absolute paths as run metadata. No score changed — the scrub
asserted all 15,230 numeric values across those files, and all ten arms still reproduce
their published mean.

What is left is the list below. None of it is urgent while the repository is private;
all of it matters the day it is not.

## 1. Authorship in the git history

All commits are authored under a real name and an employer address. That alone
deanonymises, whatever the file contents say.

```bash
git log --format='%an <%ae>' | sort -u
```

**The decision is taken: the history stays as it is, and is not to be rewritten.** A
preprint may go on arXiv while the paper is under double-blind review, so the real name in
the history is not a problem to solve.

This is also no longer free to undo. The repository is pushed to
`git@github.com:uriberger/self-saliency.git`, so rewriting the history now means a force
push over a remote that exists.

If an anonymised link for reviewers is ever needed before acceptance, the way to make one
is a fresh repository from a single squashed commit under a neutral identity — not a
rewrite of this one.

## 2. The organisational endpoints — done

Every judge default now points at OpenAI's public API with the bare `gpt-4o-mini`, which
is the model the paper reports (§5 Auxiliary models, App. A.2, App. C) — only the route
to it changed. A reader with an `OPENAI_API_KEY` needs no other configuration;
`docs/install.md` §4 has the five exports that put it back on the NVIDIA gateway, and
says why the URL and the model name have to move together.

Six defaults moved, in `selfsal/judge.py`,
`baselines/ease/reward_function/judged_perception.py`, `baselines/ease/run.sh`,
`experiments/trained_model/audit.py`, `evaluation/submit.sh` and
`evaluation/run_suite.sh`. The key precedence inverted with them —
`OPENAI_API_KEY` is now tried before `NVIDIA_API_KEY`, because a stale key in the shell
must not be the one sent to the default endpoint.

```bash
git grep -lI "nvidia\.com" -- . ':!evaluation/lmms_eval'
```

What that still finds is not a default:

* `experiments/head_selection/generate.py` — `--api-base` defaults to `None`, and
  `integrate.api.nvidia.com` appears only as an example of an OpenAI-compatible endpoint
  in its help text, beside a local NIM.
* `selfsal/steps/make_data.py` — distils the step classifier's training labels from
  Gemini 2.5 Pro, addressed by the gateway name `gcp/google/gemini-2.5-pro`. It is the
  one judge-adjacent default not flipped, because there is no public OpenAI endpoint that
  serves that model, and changing the model would change what a reader regenerates rather
  than how they reach it. The shipped classifier checkpoint is what the paper used; this
  script is only for retraining from scratch.

`tests/test_no_private_paths.py` deliberately does **not** fail on these. A test that
fails every day until an unrelated decision is made is a test people learn to ignore.

## 3. The per-sample evaluation files

`evaluation/results/` carries `results.json` only — 13 MB, enough to rebuild Tables 2, 5
and 7 with no GPU and no downloads. Appendix B's bootstrap error bars need the
`*_samples_*.jsonl` files beside them, which are about 5 GB for the ten arms and are
excluded by `.gitignore`.

Publish them as a release asset, and point `docs/reproduce.md` at it. Do not commit them:
5 GB in git history is permanent, and it is a download every clone pays for whether or not
it wants error bars.

## 4. The submodule has to be reachable

`evaluation/lmms_eval` is pinned to `4ea4f15` on `github.com/uriberger/lmms-eval`, on the
`paper-pin` branch. If that fork is private or is renamed, `git clone
--recurse-submodules` fails for everyone and the 25 benchmarks are simply absent.

`paper-pin` is branched from `a9a806b`, the commit the paper's numbers were scored under,
and carries one commit on top of it: the parser-version stamp. It is NOT the fork's
`main`, which has moved on — deliberately, so that the tree this repository pins stays the
scoring code the published numbers came from. See [rescore-audit.md](rescore-audit.md).

Two things to check before publishing:

* the fork is public, and `4ea4f15` is reachable. Being on a non-default branch is fine:
  `git clone` fetches every branch, so an anonymous clone of the fork carries the commit
  and can check it out. Checked, with every credential helper disabled:

  ```bash
  GIT_TERMINAL_PROMPT=0 git -c credential.helper= clone https://github.com/uriberger/lmms-eval
  git -C lmms-eval cat-file -e 4ea4f15   # exits 0
  ```

  What would break it is deleting the `paper-pin` branch. The commit is on no other
  branch, so nothing else keeps it alive;
* the fork's `lmms_eval/models/chat/qwen3_vl_vga.py` is a **symlink** into a sibling
  research repository — one line, no content. A plain clone of the fork gets a dangling
  link. This repository does not depend on it (VGA installs from `baselines/vga/` as a
  plugin, which is how the paper's VGA run actually worked), but anyone cloning the fork
  on its own will hit it.

## 5. Model weights

The paper promises weights as well as code. **Two models go out, not ten.** They are not in
this repository and should not be; a Hugging Face repository each, linked from
`docs/reproduce.md`, is the home.

| | size | note |
|---|---|---|
| **SELF-SALIENCY 8B** | ~16 GB | the Table 2 flagship, mean 64.26. Merged, not an adapter — see below |
| **step classifier** | **422 MB** | **a clone cannot train or probe without it** |

An earlier plan released ten: the cold start, six GRPO adapters, EASE, DAPO and the
classifier. That is no longer the plan, and the two sections below say what the narrower
one costs so nobody has to rediscover it.

### Why the flagship ships merged

`training/grpo/configs/self_saliency.yaml` trains a **rank-16 LoRA adapter** — 116 MB, on
`q_proj` and `v_proj` — and its base is
`coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged`, an SFT checkpoint of ours.
Publishing that adapter on its own would ship 116 MB nobody can load, because the base is
not on the Hub and is not being published either.

So the released artefact is the merge. `tools/merge_lora.py` produces it:

```bash
python tools/merge_lora.py \
    --base   checkpoint/coldstart_qwen3_vl_8b_instruct_sft_epoch2_lr5e5_merged \
    --adapter checkpoint/<the self_saliency run>/checkpoint-3990 \
    --output /tmp/self-saliency-8b
```

It loads with `from_pretrained` and needs nothing else, which is the point.

What that costs, stated plainly: **a reader cannot re-run GRPO from the paper's exact
starting point**, because the starting point is not published. The recipe is here —
`training/coldstart/configs/qwen3_vl_8b.yaml`, two public corpora, LLaMA-Factory — so it
can be rebuilt, but that is an 8B supervised run rather than a download, and a rebuild
will not be bit-identical to the checkpoint the published adapter was trained on.

The flagship still needs what the classifier already has: a model card, and a packaging
step. `tools/package_steps_classifier.py` and `docs/model_cards/steps_classifier.md` are
the pattern; there is no equivalent for the flagship yet. Its card must carry the prompt
contract — `selfsal/data/prompt.py`'s `SYSTEM_PROMPT` is what the cold start was
supervised against and what `evaluation/suite.yaml` evaluates under, so a reader prompting
it differently is measuring a different policy.

### What is not published, and what that forecloses

The other eight stay unreleased: the cold-start base, the four remaining GRPO adapters
(`no_sal`, `center_rect`, `question_boxes`, `self_saliency_mean`), and the
EASE and DAPO baselines.

Every number they produced is still here. `evaluation/results/` carries their
`results.json` and `python evaluation/tables.py` rebuilds Tables 2, 5 and 7 from them with
no GPU and no downloads. What is foreclosed is *re-measuring* them:

* **The ablation rows of Table 2 can be read, not re-evaluated.** Checking one means
  retraining it from its config.
* **Appendix D.3's paired comparison** — (EASE − DAPO) against (SELF-SALIENCY − No-Sal) —
  needs four checkpoints, of which one is published. It becomes a retraining exercise,
  and `baselines/ease/` is the only route to two of the four.
* **Section 4.4** (`experiments/trained_model/`) asks whether the text moves or the
  attention does, by comparing arms. With one arm released it degenerates to a
  single-model probe unless the others are retrained.

The configs themselves are unaffected: all six stay in `training/grpo/configs/`, and
`tests/test_configs_match_paper_checkpoints.py` still checks each against its published
run's `training_args.bin`, `adapter_config.json` and `trainer_state.json`. The arms remain
fully specified; only their weights stop shipping.

**Still to do:** `docs/reproduce.md`'s arms table lists ten rows and implies a checkpoint
behind each. It needs a column, or a note, marking which row is downloadable (one) and
which are retrain-only (the rest). Leaving it as it stands reads as a promise of nine
checkpoints that will not exist.

### The step classifier is the one that blocks everything else

The GRPO trainer loads it to segment every chain, and so do all four attention probes: no
classifier, no saliency reward, no Section 5. It is also not derivable from what is
released. `docs/install.md` says "to train it from scratch, see `selfsal/steps/make_data.py`",
but that distils from Gemini 2.5 Pro — which no public OpenAI endpoint serves, see section
2 — over reasoning chains that are not released either. Both routes are closed, so the
weights have to go out.

`tools/package_steps_classifier.py` builds the repository, and
`docs/model_cards/steps_classifier.md` is its card. The packaging rewrites `head.pt` as
`head.safetensors` — the same tensors, but a 1.2 MB `torch.load` of a file fetched over
the network executes whatever is in it. It deliberately does NOT repackage the model as an
`AutoModel`: the encoder is a stock `T5EncoderModel`, the head is a bespoke MLP, and there
is no stock T5 classification head matching it, so an Auto path would mean shipping a
custom modeling file behind `trust_remote_code` to fake a standalone load that is not
real. The card states the dependency instead, and gives the six lines to rebuild the head.

The card carries the input contract, which is the part that silently breaks: the model was
fitted on `[STEP] ... [CHAIN] Question: ... Chain: ...` with `include_chain: true`, and on
the label order `plan, observe, deduce, none`. Classify a bare sentence and you are running
a different model. Same argument as `selfsal/data/prompt.py` makes for the policy weights.

## 6. Licence

`LICENSE` is Apache-2.0. Note the derived work:

* `training/grpo/trl_patch/` derives from HuggingFace TRL (Apache-2.0)
* `baselines/ease/` patches EasyR1/verl, from `github.com/Nrich-sunny/Attend-to-Evidence` (Apache-2.0)

Each is compatible. **Done:** the README now has a "Derived work, and its licences"
section naming all of them, plus `baselines/vga/`, the pinned lmms-eval fork and
LLaMA-Factory, and the corpora and models used under their own terms — so the
acknowledgement is not only in the file headers.
