#!/usr/bin/env bash
# Patch a transformers-5.x install with ONLY the SDPA identity trick (no modeling-file
# replacement -- transformers 5.x threads attn weights up generically via the record-outputs
# hook system, so the whole-file modeling patch that Qwen2.5-VL needed is NOT required here).
#
# Target: the `saliency_r1_qwen3` conda env (transformers from git main, Qwen3-VL capable).
# Leave the original `saliency_r1` env + patch_transformers.sh untouched (that path is for
# Qwen2.5-VL on transformers 4.55 and needs BOTH sdpa + modeling patches).
#
# Idempotent: backs up the original to *.orig on first run.
set -euo pipefail

REPO=${SELFSAL_ROOT:-$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../.." && pwd)}
ENV=${1:-saliency_r1_qwen3}
source /home/uberger/scratch/miniconda3/etc/profile.d/conda.sh
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

cp "$REPO/transformer/sdpa_attention_tf5x.py" "$SDPA"
echo "  ✓ patched $SDPA"
echo "Done. Restore with: mv '$SDPA.orig' '$SDPA'"
