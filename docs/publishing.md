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

If the repository goes public **before** acceptance — an anonymised link for reviewers —
the history has to be rewritten or replaced with a single squashed commit under a neutral
identity. Nothing is pushed yet, so that is currently free. It stops being free the
moment there is a remote anyone has fetched from.

If it goes public **on acceptance**, anonymity no longer applies and the history can stay
as it is.

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

`evaluation/lmms_eval` is pinned to `a9a806b` on `github.com/uriberger/lmms-eval`. If that
fork is private or is renamed, `git clone --recurse-submodules` fails for everyone and the
25 benchmarks are simply absent.

Two things to check before publishing:

* the fork is public, and `a9a806b` is reachable on its default branch;
* the fork's `lmms_eval/models/chat/qwen3_vl_vga.py` is a **symlink** into a sibling
  research repository — one line, no content. A plain clone of the fork gets a dangling
  link. This repository does not depend on it (VGA installs from `baselines/vga/` as a
  plugin, which is how the paper's VGA run actually worked), but anyone cloning the fork
  on its own will hit it.

## 5. Model weights

The paper promises weights as well as code. They are not in this repository and should
not be: the adapters are ~116 MB each and the merged models are far larger. A Hugging Face
repository per arm, linked from `docs/reproduce.md`, is the obvious home.

## 6. Licence

`LICENSE` is Apache-2.0, matching every file header. Note the derived work:

* `training/grpo/trl_patch/` derives from HuggingFace TRL (Apache-2.0)
* `baselines/saliency_r1/reward.py` derives from the Saliency-R1 authors' repository
* `baselines/ease/` patches EasyR1/verl

Each is compatible. **Done:** the README now has a "Derived work, and its licences"
section naming all of them, plus `baselines/vga/`, the pinned lmms-eval fork and
LLaMA-Factory, and the corpora and models used under their own terms — so the
acknowledgement is not only in the file headers.
