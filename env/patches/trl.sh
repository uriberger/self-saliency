#!/usr/bin/env bash
# Install this repository's GRPO trainer into a TRL checkout.
#
#   bash env/patches/trl.sh [/path/to/trl_repo]
#
# TRL is cloned, not vendored: it is someone else's repository and pinning a copy of it
# here would make every upstream bump a merge in this tree. What IS here is the handful
# of files that differ, under training/grpo/trl_patch/, and this script copies them in.
#
# THE SOURCE LAYOUT AND THE INSTALLED LAYOUT ARE NOT THE SAME, and that is the whole
# reason tests/test_import_layout.py exists. A relative import is resolved against the
# directory a file ENDS UP in, so `from .x import y` can be correct where it is written
# and wrong where it lands. In the research tree that bit once: a reward module moved
# into a different package and its sibling import raised ModuleNotFoundError on a GPU,
# mid-run, after generation, the re-forward and the detector calls of step 0. No CPU test
# touched it because no CPU test looked at the installed tree.
#
# It is much harder to hit now, because the method itself lives in the `selfsal` package
# and is imported ABSOLUTELY. An absolute import into an installed package resolves the
# same wherever the importing file sits. The copy list below is still checked against the
# sources by that test, because the other half of the failure -- a file the trainer needs
# that this script forgets to copy -- is still possible, and was also hit once.
#
# Idempotent. Re-run it after editing anything under training/grpo/trl_patch/.
set -euo pipefail

REPO=${SELFSAL_ROOT:-$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../.." && pwd)}
TRL_REPO=${1:-${TRL_REPO:-$REPO/third_party/trl_repo}}
SRC="$REPO/training/grpo/trl_patch"

echo "=== installing the Self-Saliency trainer into $TRL_REPO ==="
[ -d "$TRL_REPO/trl" ] || { echo "not a TRL checkout: $TRL_REPO" >&2; exit 1; }

# ---- the copy list. Keep in step with tests/test_import_layout.py. -------------
#      source (relative to trl_patch/)      ->  destination (relative to TRL_REPO)
COPIES=(
  "grpo_trainer_qwen3.py:trl/trainer/grpo_trainer_qwen3.py"
  "grpo_vlm_qwen3.py:examples/scripts/grpo_vlm_qwen3.py"
  "rewards/self_saliency.py:trl/rewards/self_saliency.py"
  "rewards/saliency_r1.py:trl/rewards/saliency_r1.py"
  "rewards/format.py:trl/rewards/format.py"
  "rewards/answer.py:trl/rewards/answer.py"
  "rewards/__init__.py:trl/rewards/__init__.py"
  "scripts/utils.py:trl/scripts/utils.py"
  "models/utils.py:trl/models/utils.py"
)
for pair in "${COPIES[@]}"; do
    s="$SRC/${pair%%:*}"; d="$TRL_REPO/${pair##*:}"
    [ -f "$s" ] || { echo "MISSING source: $s" >&2; exit 1; }
    mkdir -p "$(dirname "$d")"
    cp "$s" "$d"
    echo "  ${pair%%:*}  ->  ${pair##*:}"
done

# ---- register the trainer in TRL's two lazy-import structures -------------------
TINIT="$TRL_REPO/trl/trainer/__init__.py"
if grep -q 'grpo_trainer_qwen3' "$TINIT"; then
    echo "  (trainer/__init__.py already registers it)"
else
    sed -i 's|"grpo_trainer": \["GRPOTrainer"\],|"grpo_trainer": ["GRPOTrainer"],\n    "grpo_trainer_qwen3": ["GRPOTrainerQwen3"],|' "$TINIT"
    echo "  registered GRPOTrainerQwen3 in trl/trainer/__init__.py"
fi

TINIT2="$TRL_REPO/trl/__init__.py"
if grep -q 'GRPOTrainerQwen3' "$TINIT2"; then
    echo "  (trl/__init__.py already registers it)"
else
    sed -i 's|"GRPOTrainer",|"GRPOTrainer",\n        "GRPOTrainerQwen3",|' "$TINIT2"
    sed -i 's|        GRPOTrainer,|        GRPOTrainer,\n        GRPOTrainerQwen3,|' "$TINIT2"
    echo "  registered GRPOTrainerQwen3 in trl/__init__.py"
fi

# ---- transformers 5.x availability shim ----------------------------------------
# _is_package_available() changed from returning a bool to returning (bool, version) in
# transformers 5.x. TRL assigns it straight into module-level flags, and a non-empty
# tuple is truthy, so EVERY availability flag reads True -- including ones for packages
# that are not installed.
IU="$TRL_REPO/trl/import_utils.py"
if grep -q '_pkg_available' "$IU"; then
    echo "  (import_utils.py already shimmed)"
else
    [ -f "$IU.orig" ] || cp "$IU" "$IU.orig"
    python3 - "$IU" <<'PYEOF'
import re, sys
path = sys.argv[1]
src = open(path).read()
shim = '''

def _pkg_available(name: str) -> bool:
    """A plain bool, for transformers <=4.x (bool) and >=5.x ((bool, version))."""
    result = _is_package_available(name)
    return result[0] if isinstance(result, tuple) else result
'''
src = src.replace(
    "from transformers.utils.import_utils import _is_package_available\n",
    "from transformers.utils.import_utils import _is_package_available\n" + shim, 1)
src = re.sub(r'_is_package_available\("([^"]+)"\)(?!\s*,\s*return_version)',
             lambda m: f'_pkg_available("{m.group(1)}")', src)
open(path, "w").write(src)
PYEOF
    echo "  shimmed import_utils.py"
fi

echo "=== done. Verify: python -c 'from trl import GRPOTrainerQwen3' ==="
