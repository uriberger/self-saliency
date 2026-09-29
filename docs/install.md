# Install

Four environments. They are separate because their pins genuinely conflict, not for
tidiness: EASE requires `transformers<=4.57`, and Qwen3-VL landed in 4.57 while the GRPO
trainer needs a 5.x dev build. There is no single resolution.

| env | runs | why it is its own |
|---|---|---|
| `selfsal-grpo` | GRPO training, every attention probe, Section 5 | transformers 5.x dev + vllm 0.11; vllm drags torch to 2.8/cu128 |
| `selfsal-sft` | the cold start, via LLaMA-Factory | transformers 5.7, torch 2.6/cu124 |
| `selfsal-eval` | the 25-benchmark suite, and VGA | its own lmms-eval stack |
| `selfsal-ease` | the EASE and DAPO baselines | `transformers<=4.57`, which nothing else can hold |

Only `selfsal-eval` is needed to rebuild the paper's tables from the banked results. The
other three are needed to retrain.

## 0. The repository

```bash
git clone --recurse-submodules <url> self-saliency
cd self-saliency
```

`--recurse-submodules` matters. `evaluation/lmms_eval` is a submodule pinned at the
commit the paper's numbers were scored under. Without it the evaluation code is present
and the benchmarks it evaluates are not.

If you already cloned without it: `git submodule update --init`.

## 1. The environments

Each script creates one conda env and installs from a pinned file in `env/`.

```bash
bash env/setup_grpo_env.sh      # selfsal-grpo
bash env/setup_sft_env.sh       # selfsal-sft
bash env/setup_eval_env.sh      # selfsal-eval
bash baselines/ease/setup_env.sh  # selfsal-ease
```

Install this repository into whichever ones you will use:

```bash
conda run -n selfsal-grpo pip install -e .
conda run -n selfsal-eval pip install -e .
```

That is what puts `selfsal` on the path. It is not optional for training: the trainer
imports the method absolutely from the installed package, which is what makes its imports
independent of where the patch script lands each file.

## 2. The external trainers

TRL and LLaMA-Factory are cloned, not vendored. Vendoring someone else's repository makes
every upstream bump a merge in this tree.

```bash
git clone --branch v0.21-release https://github.com/huggingface/trl third_party/trl_repo
conda run -n selfsal-grpo pip install -e third_party/trl_repo --no-deps

bash env/patches/trl.sh            # install our trainer into it
bash env/patches/transformers.sh   # the attention-weights patch
bash env/patches/vllm.sh           # two vllm-0.11-vs-transformers-5.x breakages
```

Re-run `env/patches/trl.sh` after editing anything under `training/grpo/trl_patch/`.
`tests/test_import_layout.py` checks that the script's copy list still covers every file
the trainer needs; run it after touching either.

For the cold start:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory third_party/LLaMA-Factory
conda run -n selfsal-sft pip install -e third_party/LLaMA-Factory
```

## 3. Data

Nothing large is tracked. `SELFSAL_DATA` points at where it lives; it defaults to
`./data`, and anything missing fails with the path and what it was for rather than
quietly finding nothing.

```bash
export SELFSAL_DATA=/somewhere/with/room      # optional
python training/coldstart/data/dl_llavacot.py     # cold-start SFT corpora
python training/coldstart/data/dl_mulberry.py
python -m selfsal.data.boxed_corpus --out-dir "$SELFSAL_DATA/boxed_corpus" --n 1800
```

The GRPO corpus (`peterant330/saliency-r1-8k`) is pulled from the Hub on first use.

Everything that touches the Hub honours `HF_HOME` and nothing overrides it, so the cache
lands in `~/.cache/huggingface` unless you say otherwise. On a cluster, where that is
usually a small quota, export it alongside `SELFSAL_DATA` — every `submit.sh` here
already requires it.

The step classifier's checkpoint is needed by training and by every probe; point
`SELFSAL_STEPS_CKPT` at it, or put it in `checkpoint/steps_classifier/best`. To train it
from scratch, see `selfsal/steps/make_data.py` and `selfsal/steps/train.py`.

## 4. Check it

```bash
pytest                                        # 2,300+ CPU tests, no GPU
python -m training.grpo.config --all          # the six arms and their settings
bash training/grpo/run.sh self_saliency --dry-run
```

Several tests compare against the original research repositories and skip when those are
absent — which is the normal case outside NVIDIA. Set `SELFSAL_ARCHIVE` if you have them.
Everything else runs anywhere.

## Then

[reproduce.md](reproduce.md) has one row per table and figure.
