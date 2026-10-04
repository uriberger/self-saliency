#!/usr/bin/env bash
# Patch a transformers-5.x install with ONLY the SDPA identity trick (no modeling-file
# replacement -- transformers 5.x threads attn weights up generically via the record-outputs
# hook system, so the whole-file modeling patch that Qwen2.5-VL needed is NOT required here).
#
# Target: the `selfsal-grpo` env, which is where both training and the vLLM server run.
#
# Idempotent: backs up the original to *.orig on first run.
set -euo pipefail

REPO=${SELFSAL_ROOT:-$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../.." && pwd)}
ENV=${1:-selfsal-grpo}
# Derived, not demanded. Requiring CONDA_ROOT made `bash env/patches/transformers.sh` --
# which is what docs/install.md tells a reader to run -- fail on a correctly installed
# machine with `CONDA_ROOT: set CONDA_ROOT` and nothing else.
CONDA_ROOT=${CONDA_ROOT:-$(conda info --base 2>/dev/null)}
[ -n "$CONDA_ROOT" ] || { echo "no conda on PATH; set CONDA_ROOT" >&2; exit 1; }
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV"

SP=$(python -c "import transformers, os; print(os.path.dirname(transformers.__file__))")
VER=$(python -c "import transformers; print(transformers.__version__)")
echo "env=$ENV  transformers=$VER  at: $SP"
case "$VER" in
    5.*|4.5[7-9]*|4.[6-9]*) : ;;  # 4.57+ / 5.x expected
    *) echo "WARNING: transformers $VER may not support Qwen3-VL or the record-outputs system." ;;
esac

SDPA="$SP/integrations/sdpa_attention.py"
[ -f "$SDPA" ] || { echo "MISSING target: $SDPA"; exit 1; }
[ -f "$SDPA.orig" ] || cp "$SDPA" "$SDPA.orig"

cp "$REPO/env/patches/transformers_patch/sdpa_attention_tf5x.py" "$SDPA"
echo "  ✓ patched $SDPA"
echo "Done. Restore with: mv '$SDPA.orig' '$SDPA'"
