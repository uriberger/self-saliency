#!/usr/bin/env bash
# Patch vLLM 0.11.0 for transformers 5.x compatibility (Qwen3-VL).
#
# transformers 5.x moved/removed attributes that vLLM 0.11.0 reads directly, crashing the
# vLLM server at startup. Each fix falls back to the value's new location / a safe default:
#
#   1. vllm/transformers_utils/tokenizer.py  (get_cached_tokenizer)
#      transformers 5.x removed tokenizer.all_special_tokens_extended ->
#      AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended.
#      Fall back to all_special_tokens (vLLM types the field as list[str] anyway).
#
#   2. vllm/model_executor/models/qwen3_vl.py
#      The text sub-config (Qwen3VLTextConfig) no longer carries `tie_word_embeddings`
#      (it now lives only on the top-level Qwen3VLConfig, value False for the 8B) ->
#      AttributeError: 'Qwen3VLTextConfig' object has no attribute 'tie_word_embeddings'.
#      Guard with getattr(config, "tie_word_embeddings", False).
#
# Idempotent: backs up each file's .orig on first change; safe to re-run.
# Usage:  bash env/patches/vllm.sh [env_name]     # default env: selfsal-grpo
set -euo pipefail

# `selfsal-grpo`, not a separate vLLM env: training/grpo/run.sh starts the server with the
# same interpreter it trains with, so one env carries both and both patches apply to it.
ENV=${1:-selfsal-grpo}
# Derived, not demanded -- see the note in transformers.sh.
CONDA_ROOT=${CONDA_ROOT:-$(conda info --base 2>/dev/null)}
[ -n "$CONDA_ROOT" ] || { echo "no conda on PATH; set CONDA_ROOT" >&2; exit 1; }
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"
echo "=== env/patches/vllm.sh: env=$ENV ==="

# py_patch FILE NEEDLE REPLACEMENT ALREADY_MARKER
py_patch() {
    local f="$1" marker="$4"
    if grep -qF "$marker" "$f"; then echo "  ($(basename "$f"): already patched)"; return 0; fi
    [ -f "$f.orig" ] || cp "$f" "$f.orig"
    F="$f" NEEDLE="$2" REPL="$3" python - <<'PY'
import os
f = os.environ["F"]; needle = os.environ["NEEDLE"]; repl = os.environ["REPL"]
s = open(f).read()
n = s.count(needle)
if n < 1:
    raise SystemExit(f"  PATTERN NOT FOUND in {f}: {needle!r} (vLLM version differs)")
open(f, "w").write(s.replace(needle, repl))
print(f"  patched {os.path.basename(f)} ({n} occurrence(s))")
PY
}

# Locate files via sysconfig (importing vllm prints logging that pollutes $(...) capture).
VLLM_DIR="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/vllm"
TOK="$VLLM_DIR/transformers_utils/tokenizer.py"
QV="$VLLM_DIR/model_executor/models/qwen3_vl.py"
CFGPY="$VLLM_DIR/transformers_utils/config.py"
[ -f "$TOK" ]   || { echo "  MISSING: $TOK"; exit 1; }
[ -f "$QV" ]    || { echo "  MISSING: $QV"; exit 1; }
[ -f "$CFGPY" ] || { echo "  MISSING: $CFGPY"; exit 1; }
echo "  tokenizer.py: $TOK"
echo "  qwen3_vl.py:  $QV"
echo "  config.py:    $CFGPY"

# NOTE: vLLM 0.11.1 refactored get_cached_tokenizer -- the attribute is now read into a
# local instead of being passed inline, so the 0.11.0 needle (which ended in ")") no longer
# matches. Needle below targets the 0.11.1 form.
# THE NEEDLE IS THE ATTRIBUTE ACCESS, NOT THE ASSIGNMENT AROUND IT. It used to be
# `= tokenizer.all_special_tokens_extended`, which encoded the line formatting and so
# matched nothing in vllm 0.11.0 -- the pinned version -- where the assignment is wrapped:
#
#     tokenizer_all_special_tokens_extended = (
#         tokenizer.all_special_tokens_extended)
#
# The patch failed with "PATTERN NOT FOUND ... (vLLM version differs)" on exactly the
# version env/grpo.txt installs. Matching the dotted access is robust to the wrapping and
# to both 0.11.0 and 0.11.1; the local `tokenizer_all_special_tokens_extended` does not
# collide with it, because that one is an underscore and this is a dot.
#
# AND NO TRAILING COMMENT IN THE REPLACEMENT. The expression being replaced is followed by
# the `)` that closes the wrapped assignment, so a `# ...` after it comments that paren out
# and the file stops parsing: "'(' was never closed". The marker argument is what makes the
# patch idempotent; it does not need a comment in the source to do that.
py_patch "$TOK" \
    "tokenizer.all_special_tokens_extended" \
    'getattr(tokenizer, "all_special_tokens_extended", tokenizer.all_special_tokens)' \
    'all_special_tokens_extended", tokenizer.all_special_tokens'

py_patch "$QV" \
    "config.tie_word_embeddings" \
    'getattr(config, "tie_word_embeddings", False)' \
    'getattr(config, "tie_word_embeddings", False)'

# General fix (covers all 75+ model-code sites): backfill tie_word_embeddings onto the
# shared text sub-config in get_hf_text_config(), which returns the config vLLM's model
# receives. Qwen3VLTextConfig declares it on the class but loaded instances lack it.
py_patch "$CFGPY" \
    '        assert hasattr(text_config, "num_attention_heads")' \
    $'        assert hasattr(text_config, "num_attention_heads")\n        if not hasattr(text_config, "tie_word_embeddings"):  # transformers 5.x compat\n            text_config.tie_word_embeddings = getattr(config, "tie_word_embeddings", False)' \
    'text_config.tie_word_embeddings = getattr(config'

echo "Done. Verify import: python -c 'import vllm.model_executor.models.qwen3_vl'"
